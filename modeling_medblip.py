import logging
import math
from typing import List, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F


from medblip.mae_vit import mae_vit_base_patch16, mae_vit_small_patch16
from medblip.mae_bert import BertConfig
from medblip.modeling_table import SimpleTableRestorer
from medblip.modeling_fttransformer import TableFTTRestorer, FTTransformer

from medblip.utils import itc_loss, distillation_loss


# 轻量级交叉注意力模块
class LightweightCrossAttention(nn.Module):
    """
    轻量级交叉注意力模块，用于增强对比学习中的特征表示
    实现思路：
    1. 对输入的两个特征进行线性投影，生成查询、键、值
    2. 计算注意力权重，捕捉特征间的相关性
    3. 应用注意力并通过轻量级前馈网络
    4. 添加残差连接，确保信息流动
    """
    def __init__(self, dim, num_heads=2, reduction=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # 轻量级投影
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        
        # 降维以减少计算量
        self.reduction = nn.Linear(dim, dim // reduction)
        self.expansion = nn.Linear(dim // reduction, dim)
        
        # 层归一化
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x1, x2):
        # 保存原始输入用于残差连接
        residual = x1
        batch_size = x1.shape[0]
        
        # 生成查询、键、值
        q = self.q_proj(x1)
        k = self.k_proj(x2)
        v = self.v_proj(x2)
        
        # 多头注意力
        q = q.view(batch_size, self.num_heads, self.head_dim)
        k = k.view(batch_size, self.num_heads, self.head_dim)
        v = v.view(batch_size, self.num_heads, self.head_dim)
        
        # 计算注意力权重
        attn = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim), dim=-1)
        
        # 应用注意力
        out = torch.matmul(attn, v)
        out = out.view(batch_size, -1)
        
        # 轻量级前馈网络
        out = self.expansion(F.gelu(self.reduction(out)))
        
        # 残差连接和层归一化
        out = self.norm(out + residual)
        
        return out


# 简化版的FT-Transformer，用于学生模型
class StudentFTTransformer(nn.Module):
    """
    简化版的Feature Tokenizer Transformer，用于学生模型的蒸馏
    相比原版FTTransformer，减少了层数、注意力头数和前馈网络维度
    """
    def __init__(self, 
                 num_numerical_features: int,
                 categorical_cardinalities: List[int] = None,
                 d_token: int = 192,
                 num_heads: int = 4,  # 减少注意力头数
                 d_ff: int = 384,     # 减少前馈网络维度
                 num_layers: int = 2,  # 减少层数
                 dropout: float = 0.1,
                 use_cls_token: bool = True,
                 output_dim: int = 768,
                 use_positional_embedding: bool = True):
        super().__init__()
        
        if categorical_cardinalities is None:
            categorical_cardinalities = []
            
        # 复用原版FTTransformer的FeatureTokenizer
        self.feature_tokenizer = FTTransformer(num_numerical_features=num_numerical_features, 
                                              categorical_cardinalities=categorical_cardinalities, 
                                              d_token=d_token,
                                              use_positional_embedding=use_positional_embedding).feature_tokenizer
        
        # 是否使用CLS token
        self.use_cls_token = use_cls_token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))
        
        # 简化的Transformer编码器
        from medblip.modeling_fttransformer import TransformerEncoder
        self.encoder = TransformerEncoder(
            d_model=d_token,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # 输出投影层（用于匹配MedBLIP模型的维度）
        self.output_projection = nn.Sequential(
            nn.Linear(d_token, d_token * 2),
            nn.GELU(),
            nn.Linear(d_token * 2, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, numerical_features: Optional[torch.Tensor] = None, 
                categorical_features: Optional[torch.Tensor] = None):
        # 对特征进行分词
        tokens = self.feature_tokenizer(numerical_features, categorical_features)
        batch_size = tokens.shape[0]
        
        # 添加CLS token（如果使用）
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            tokens = torch.cat([cls_tokens, tokens], dim=1)
        
        # 通过简化的Transformer编码器
        encoder_output = self.encoder(tokens)
        
        # 如果使用CLS token，只使用CLS token的输出
        if self.use_cls_token:
            output = encoder_output[:, 0]
        else:
            # 否则，对所有token的输出进行平均池化
            output = encoder_output.mean(dim=1)
        
        # 通过输出投影层
        output = self.output_projection(output)
        
        return output

class MedBLIPModel(nn.Module):
    def __init__(
        self,
        max_txt_len=60,
        hidden_size=768,
        num_class = 2,
        num_numerical_features=60,  # 语音特征中的数值特征数量
        num_cog_features=60,  # 认知量表特征中的数值特征数量
        use_ft_transformer=True,    # 是否使用FT-Transformer替代SimpleTableRestorer
        use_positional_embedding=True,  # 是否使用位置嵌入
        use_cross_attention=True,      # 是否使用跨模态注意力
        # 训练模式参数
        training_mode='teacher',     # 训练模式: 'teacher'(教师模型), 'student'(学生模型蒸馏), 'single'(单模态)
        freeze_teacher=False,       # 是否冻结教师模型参数（仅在student模式下有效）
        modalities='both',          # 训练使用的模态: 'both'(多模态), 'vision'(仅视觉), 'text'(仅文本/语音), 'cog'(仅认知量表), 'speech_cog'(语音+认知量表)
        # 蒸馏相关参数
        kd_temp=2.0,                # 蒸馏温度
        kd_weight=0.5,              # 蒸馏损失权重
        student_cls_weight=1.0,     # 学生分类损失权重
        feat_distill_weight=0.5,    # 特征级蒸馏权重
    ):
        super().__init__()

        # 训练模式配置
        self.training_mode = training_mode
        self.modalities = modalities
        self.freeze_teacher = freeze_teacher
        
        # 蒸馏相关配置参数
        self.distillation_temp = nn.Parameter(torch.tensor(kd_temp))
        self.kd_weight = kd_weight
        self.student_cls_weight = student_cls_weight
        self.feat_distill_weight = feat_distill_weight
        
        # 数据增强和重构参数
        self.image_mask_ratio = 0.75  # 图像掩码率，用于图像重构损失
        self.text_mask_ratio = 0.2    # 文本掩码率，用于文本重构损失
        self.text_res_weight = 1.0  # 文本恢复损失权重，可配置
        
        # 根据训练模式设置disable_student
        self.disable_student = training_mode in ['teacher', 'single']
        
        # 单模态分类器
        if modalities == 'vision':
            # 仅视觉模态分类器
            self.mlp_vision = nn.Sequential(
                nn.Linear(768, 256),
                nn.ReLU(),
                nn.Linear(256, num_class)
            )
        elif modalities == 'text':
            # 仅文本/语音模态分类器
            self.mlp_text = nn.Sequential(
                nn.Linear(768, 256),
                nn.ReLU(),
                nn.Linear(256, num_class)
            )
        elif modalities == 'cog':
            # 仅认知量表模态分类器
            self.mlp_cog = nn.Sequential(
                nn.Linear(768, 256),
                nn.ReLU(),
                nn.Linear(256, num_class)
            )
        elif modalities == 'speech_cog':
            # 语音+认知量表融合分类器
            self.mlp_speech_cog = nn.Sequential(
                nn.Linear(768*2, 256),
                nn.ReLU(),
                nn.Linear(256, num_class)
            )
        


        self.vision_transformer = mae_vit_base_patch16()

        config = BertConfig.from_pretrained("/data-pool/data/data2/qiuhui/tokenizer")
        config.hidden_size = 768
        config.num_attention_heads = 12
        config.num_hidden_layers = 24
        
        # 使用FT-Transformer替代简单的表格处理器
        self.use_ft_transformer = use_ft_transformer
        if use_ft_transformer:
            # 使用FT-Transformer处理表格特征（教师模型）
            self.text_restorer = TableFTTRestorer(
                num_numerical_features=num_numerical_features,
                categorical_cardinalities=None,  # 假设语音特征都是数值型的
                hidden_dim=hidden_size,
                dropout=0.1,
                num_cross_attn_heads=config.num_attention_heads,  # 使用与视觉编码器相同的头数
                use_positional_embedding=use_positional_embedding
            )
            # 设置表格特征维度
            self.table_feature_dim = num_numerical_features
            
            # 与教师模型相同的FT-Transformer用于学生模型
            self.student_text_restorer = TableFTTRestorer(
                num_numerical_features=num_numerical_features,
                categorical_cardinalities=None,  # 假设语音特征都是数值型的
                hidden_dim=hidden_size,
                dropout=0.1,
                num_cross_attn_heads=config.num_attention_heads,  # 使用与视觉编码器相同的头数
                use_positional_embedding=use_positional_embedding
            )
            
            # 为认知量表模态添加FT-Transformer处理器
            self.cog_restorer = TableFTTRestorer(
                num_numerical_features=num_cog_features,
                categorical_cardinalities=None,  # 假设认知量表特征都是数值型的
                hidden_dim=hidden_size,
                dropout=0.1,
                num_cross_attn_heads=config.num_attention_heads,
                use_positional_embedding=use_positional_embedding
            )
            # 设置认知量表特征维度
            self.cog_feature_dim = num_cog_features
        else:
            # 保持原有实现作为备选
            self.text_restorer = SimpleTableRestorer(num_numerical_features, hidden_size)
            # 为认知量表模态添加简单表格处理器
            self.cog_restorer = SimpleTableRestorer(num_cog_features, hidden_size)
        
        # 保存参数
        self.max_txt_len = max_txt_len

        self.vision_proj = nn.Sequential(
            nn.Linear(self.vision_transformer.dim, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size)
        )
#        self.vision_proj = nn.Sequential(
#            nn.Linear(self.vision_transformer.dim, hidden_size),
#            nn.LayerNorm(hidden_size)
#        )
        
        self.text_proj = nn.Sequential(
            nn.Linear(max_txt_len, hidden_size),
            nn.LayerNorm(hidden_size)
        )
        
        # 认知量表特征投影层
        self.cog_proj = nn.Sequential(
            nn.Linear(num_cog_features, hidden_size),
            nn.LayerNorm(hidden_size)
        )

        # 多模态融合分类器 [image_feat, speech_feat] - 教师模型
        self.mlp = nn.Sequential(
            nn.Linear(768*2, 256),
            nn.ReLU(),
            nn.Linear(256, num_class)
        )
        # === 语音单模态头（student）===
        self.mlp_speech = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, num_class)
        )
        
        # 特征级蒸馏损失
        self.feat_criterion = nn.MSELoss()
        
        # 分类损失函数（初始化为None，在训练时设置权重）
        self.criterion = None

        self.itc_temp = nn.Parameter(0.07 * torch.ones([]))  # 对比学习温度
        self.mlm_probability = 0.5
        
        # 交叉注意力模块，用于增强对比学习的特征表示
        self.cross_attention = LightweightCrossAttention(hidden_size)
        self.use_cross_attention = use_cross_attention  # 是否使用交叉注意力增强对比学习
        
        # 根据训练模式设置参数的可训练性
        if self.disable_student:
            # 禁用学生模型模式：训练教师模型参数，完全禁用学生模型相关组件
            if hasattr(self, 'student_text_restorer'):
                for param in self.student_text_restorer.parameters():
                    param.requires_grad = False
            if hasattr(self, 'mlp_speech'):
                for param in self.mlp_speech.parameters():
                    param.requires_grad = False
            print(f"模型初始化完成 - 训练模式: {'单模态' if training_mode == 'single' else '教师模型训练'} (禁用学生模型)")
        elif self.freeze_teacher:
            # 学生模型蒸馏阶段：冻结教师模型参数，训练学生模型参数
            self._freeze_teacher_parameters()
        else:
            # 教师模型训练阶段：训练教师模型参数，冻结学生模型参数
            if hasattr(self, 'student_text_restorer'):
                for param in self.student_text_restorer.parameters():
                    param.requires_grad = False
            if hasattr(self, 'mlp_speech'):
                for param in self.mlp_speech.parameters():
                    param.requires_grad = False
            
        # 打印训练模式信息
        mode_name = {
            'teacher': '教师模型训练',
            'student': '学生模型蒸馏训练',
            'single': f'{"视觉" if modalities == "vision" else "文本/语音"}单模态模型训练'
        }[training_mode]
        
        print(f"模型初始化完成 - 训练模式: {mode_name}")
        print(f"使用的表格处理器: {'FT-Transformer' if use_ft_transformer else 'SimpleTableRestorer'}")
        if training_mode == 'student':
            print(f"蒸馏参数: temperature={kd_temp}, kd_weight={kd_weight}, student_cls_weight={student_cls_weight}, feat_distill_weight={feat_distill_weight}")
    def _freeze_teacher_parameters(self):
        """冻结教师模型参数，确保蒸馏过程中不更新教师模型"""
        # 添加静态标志位避免重复打印
        if not hasattr(self, '_has_printed_params_status'):
            self._has_printed_params_status = False
            
        # 冻结教师模型相关组件
        frozen_layers = []
        
        # 冻结视觉编码器参数（如果存在）
        if hasattr(self, 'vision_transformer'):
            for param in self.vision_transformer.parameters():
                param.requires_grad = False
            frozen_layers.append('vision_transformer')
        
        # 冻结视觉投影层参数（如果存在）
        if hasattr(self, 'vision_proj'):
            for param in self.vision_proj.parameters():
                param.requires_grad = False
            frozen_layers.append('vision_proj')
            
        # 根据模态冻结相应的分类器
        if self.modalities == 'both':
            # 多模态融合分类器
            if hasattr(self, 'mlp'):
                for param in self.mlp.parameters():
                    param.requires_grad = False
                frozen_layers.append('mlp')
        elif self.modalities == 'vision':
            # 仅视觉模态分类器
            if hasattr(self, 'mlp_vision'):
                for param in self.mlp_vision.parameters():
                    param.requires_grad = False
                frozen_layers.append('mlp_vision')
        elif self.modalities == 'text':
            # 仅文本/语音模态分类器
            if hasattr(self, 'mlp_text'):
                for param in self.mlp_text.parameters():
                    param.requires_grad = False
                frozen_layers.append('mlp_text')
        elif self.modalities == 'cog':
            # 仅认知量表模态分类器
            if hasattr(self, 'mlp_cog'):
                for param in self.mlp_cog.parameters():
                    param.requires_grad = False
                frozen_layers.append('mlp_cog')
        elif self.modalities == 'speech_cog':
            # 语音+认知量表融合分类器
            if hasattr(self, 'mlp_speech_cog'):
                for param in self.mlp_speech_cog.parameters():
                    param.requires_grad = False
                frozen_layers.append('mlp_speech_cog')
            # 冻结认知量表处理器
            if hasattr(self, 'cog_restorer'):
                for param in self.cog_restorer.parameters():
                    param.requires_grad = False
                frozen_layers.append('cog_restorer')
            
        # 冻结文本处理相关参数 - 仅冻结用于教师模型的部分
        if hasattr(self, 'text_restorer'):
            for param in self.text_restorer.parameters():
                param.requires_grad = False
            frozen_layers.append('text_restorer')
            
        # 学生模型可训练参数
        trainable_layers = []
        
        # 确保学生TableFTTRestorer可训练
        if hasattr(self, 'student_text_restorer'):
            for param in self.student_text_restorer.parameters():
                param.requires_grad = True
            trainable_layers.append('student_text_restorer')
            
        # 确保学生模型分类头可训练
        if hasattr(self, 'mlp_speech'):
            for param in self.mlp_speech.parameters():
                param.requires_grad = True
            trainable_layers.append('mlp_speech')
        
        # 只打印一次参数状态摘要
        if not self._has_printed_params_status:
            print(f"[模型参数状态] 已冻结: {', '.join(frozen_layers)}")
            print(f"[模型参数状态] 可训练: {', '.join(trainable_layers)}")
            print(f"[模型参数状态] 学生模型现在使用与教师模型相同的FT-Transformer进行特征提取")
            
            # 确认冻结状态 - 打印参数数量统计
            total_params = sum(p.numel() for p in self.parameters())
            frozen_params = sum(p.numel() for p in self.parameters() if not p.requires_grad)
            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            
            print(f"[参数统计] 总参数: {total_params:,} 个")
            print(f"[参数统计] 冻结参数: {frozen_params:,} 个 ({frozen_params/total_params*100:.1f}%)")
            print(f"[参数统计] 可训练参数: {trainable_params:,} 个 ({trainable_params/total_params*100:.1f}%)")
            print(f"[参数统计] 学生模型训练约 {trainable_params/1000:.0f}K 个参数")
            
            # 标记已打印
            self._has_printed_params_status = True

    def augment_image(self, image):
        """MAE专用的简单图像增强"""
        aug_image = image.clone()
        device = aug_image.device
        
        # 60%概率增强
        if torch.rand(1, device=device) > 0.4:
            # 1. 高斯噪声 - 对MAE有意义
            if torch.rand(1, device=device) > 0.5:
                noise_std = 0.05 * torch.std(aug_image)
                noise = torch.randn_like(aug_image) * noise_std
                aug_image = aug_image + noise
            
            # 2. 对比度调整 - 对MAE有意义
            if torch.rand(1, device=device) > 0.5:
                contrast = 0.9 + 0.2 * torch.rand(1, device=device)  # [0.9, 1.1] 更保守
                mean_val = aug_image.mean()
                aug_image = contrast * (aug_image - mean_val) + mean_val
        
        return aug_image

    def augment_text(self, text):
        """简单有效的表格增强"""
        aug_text = text.clone()
        device = aug_text.device
        
        # 60%概率增强
        if torch.rand(1, device=device) > 0.4:
            # 1. 高斯噪声
            if torch.rand(1, device=device) > 0.5:
                noise_std = 0.05 * torch.std(aug_text, dim=0, keepdim=True)
                noise = torch.randn_like(aug_text) * noise_std
                aug_text = aug_text + noise
            
            # 2. 特征丢弃
            if torch.rand(1, device=device) > 0.5:
                drop_rate = 0.1  # 丢弃10%的特征
                mask = torch.rand_like(aug_text) > drop_rate
                aug_text = aug_text * mask.float()
        
        return aug_text

    def mask_table_features(self, table_features, mask_ratio=0.2, targets=None, masked_indices=None):
        """
        表格数据的mask策略
        Args:
            table_features: [batch_size, feature_dim] 表格特征
            mask_ratio: mask比例
            targets: 可选的目标标签
            masked_indices: 可选的预定义mask位置
        """
        batch_size, feature_dim = table_features.shape
        device = table_features.device
        
        # 1. 创建mask位置
        if masked_indices is None:
            # 随机选择mask位置，每个样本独立
            probability_matrix = torch.full((batch_size, feature_dim), mask_ratio, device=device)
            masked_indices = torch.bernoulli(probability_matrix).bool()
            
            # 确保每个样本至少有一个特征被mask
            for i in range(batch_size):
                if not masked_indices[i].any():
                    # 随机选择一个特征mask
                    rand_idx = torch.randint(0, feature_dim, (1,), device=device)
                    masked_indices[i, rand_idx] = True
        
        # 2. 创建被mask的特征（将masked位置置0）
        masked_features = table_features.clone()
        masked_features[masked_indices] = 0
        
        # 3. 如果提供了targets，只计算被mask位置的loss
        if targets is not None:
            # 对于表格数据，targets通常是原始特征本身
            targets = targets.clone()
            targets[~masked_indices] = -100  # 忽略未被mask的位置
        
        if targets is not None:
            return masked_features, targets, masked_indices
        else:
            return masked_features, masked_indices

    
    def _extract_features(self, image, text, cog, augment=True):
        """
        统一的特征提取方法，处理不同模态的特征提取和增强
        
        Args:
            image: 图像数据 [batch_size, 1, 128, 128, 128]
            text: 文本/语音特征 [batch_size, feature_dim]
            cog: 认知量表特征 [batch_size, cog_feature_dim]
            augment: 是否进行数据增强
            
        Returns:
            dict: 包含所有提取的特征
        """
        features = {}
        
        # ========== 数据增强 ==========
        if self.modalities == 'both' or self.modalities == 'vision':
            features['image'] = self.augment_image(image) if augment else image
        if self.modalities == 'both' or self.modalities == 'text' or self.modalities == 'speech_cog':
            features['text'] = self.augment_text(text) if augment else text
        if self.modalities == 'cog' or self.modalities == 'speech_cog':
            features['cog'] = self.augment_text(cog) if augment else cog

        # ========== 特征提取 ==========
        # 1. 首先提取所有基础特征（不包含重构损失计算）
        
        # 提取图像特征（如果需要）
        if self.modalities == 'both' or self.modalities == 'vision':
            image_embeds, attns, _, _ = self.vision_transformer.forward_encoder(
                img=features['image'], text_emb=None, mask_ratio=0)
            features['image_embeds'] = self.vision_proj(image_embeds)
            features['image_feat'] = F.normalize(features['image_embeds'][:,0,:], dim=-1)

        # 提取语音表格特征（如果需要）
        if self.modalities == 'both' or self.modalities == 'text' or self.modalities == 'speech_cog':
            if hasattr(self, 'use_ft_transformer') and self.use_ft_transformer:
                # 提取分类特征（不使用交叉注意力）
                _, text_features_cls, text_attn_probs = self.text_restorer(
                    original_feat=features['text'],
                    masked_embeddings=features['text'].unsqueeze(1),
                    image_embeddings=None,
                    mask=None,
                    compute_loss=False
                )
                
                # 保存特征
                features['text_features'] = text_features_cls
                features['text_embeds'] = text_features_cls.unsqueeze(1)
                features['text_feat'] = F.normalize(text_features_cls, dim=-1)
                features['text_attn_probs'] = text_attn_probs  # 保存注意力权重
                
                # 准备掩码特征用于后续的重构
                if augment:
                    masked_text, mask_indices = self.mask_table_features(features['text'], mask_ratio=self.text_mask_ratio)
                    masked_text_embeds = masked_text.unsqueeze(1)
                    features['masked_text'] = masked_text
                    features['mask_indices'] = mask_indices
                else:
                    features['mask_indices'] = None
            else:
                # 原始处理方式
                features['text_embeds'] = self.text_proj(features['text']).unsqueeze(1)
                features['text_feat'] = F.normalize(features['text_embeds'][:,0,:], dim=-1)
                features['text_features'] = features['text_embeds'][:,0]
                
                # 准备掩码特征用于后续的重构
                if augment:
                    masked_text, mask_indices = self.mask_table_features(features['text'], mask_ratio=self.text_mask_ratio)
                    masked_text_embeds_proj = self.text_proj(masked_text).unsqueeze(1)
                    features['masked_text'] = masked_text
                    features['mask_indices'] = mask_indices
                    features['masked_text_embeds_proj'] = masked_text_embeds_proj

        # 提取认知量表特征（如果需要）
        if self.modalities == 'cog' or self.modalities == 'speech_cog':
            if hasattr(self, 'use_ft_transformer') and self.use_ft_transformer:
                # 提取分类特征（不使用交叉注意力）
                _, cog_features_cls, cog_attn_probs = self.cog_restorer(
                    original_feat=features['cog'],
                    masked_embeddings=features['cog'].unsqueeze(1),
                    image_embeddings=None,
                    mask=None,
                    compute_loss=False
                )
                
                # 保存特征
                features['cog_features'] = cog_features_cls
                features['cog_embeds'] = cog_features_cls.unsqueeze(1)
                features['cog_feat'] = F.normalize(cog_features_cls, dim=-1)
                features['cog_attn_probs'] = cog_attn_probs  # 保存注意力权重
                
                # 准备掩码特征用于后续的重构
                if augment:
                    masked_cog, cog_mask_indices = self.mask_table_features(features['cog'], mask_ratio=self.text_mask_ratio)
                    masked_cog_embeds = masked_cog.unsqueeze(1)
                    features['masked_cog'] = masked_cog
                    features['cog_mask_indices'] = cog_mask_indices
                else:
                    features['cog_mask_indices'] = None
            else:
                # 原始处理方式
                features['cog_embeds'] = self.cog_proj(features['cog']).unsqueeze(1)
                features['cog_feat'] = F.normalize(features['cog_embeds'][:,0,:], dim=-1)
                features['cog_features'] = features['cog_embeds'][:,0]
                
                # 准备掩码特征用于后续的重构
                if augment:
                    masked_cog, cog_mask_indices = self.mask_table_features(features['cog'], mask_ratio=self.text_mask_ratio)
                    masked_cog_embeds_proj = self.cog_proj(masked_cog).unsqueeze(1)
                    features['masked_cog'] = masked_cog
                    features['cog_mask_indices'] = cog_mask_indices
                    features['masked_cog_embeds_proj'] = masked_cog_embeds_proj

        # 2. 然后计算所有重构损失（此时所有基础特征都已提取完成）
        
        # 计算语音表格重构损失（如果需要）
        if (self.modalities == 'both' or self.modalities == 'text' or self.modalities == 'speech_cog') and augment:
            if hasattr(self, 'use_ft_transformer') and self.use_ft_transformer:
                # 确定用于文本恢复的交叉注意力特征
                # 在speech_cog模态下使用认知量表特征，在both模态下使用图像特征，其他情况不使用
                if self.modalities == 'speech_cog':
                    cross_embeddings_for_text = features['cog_embeds']
                elif self.modalities == 'both':
                    cross_embeddings_for_text = features['image_embeds'][:,0:1,:]
                else:
                    cross_embeddings_for_text = None
                
                # 计算重构损失
                text_res_loss, _, _ = self.text_restorer(
                    original_feat=features['text'],
                    masked_embeddings=features['masked_text'].unsqueeze(1),
                    image_embeddings=cross_embeddings_for_text,
                    mask=features['mask_indices']
                )
                features['text_res_loss'] = text_res_loss
            else:
                # 原始处理方式
                image_embeddings_for_text = features['image_embeds'] if self.modalities == 'both' else None
                text_res_loss = self.text_restorer(
                    original_feat=features['text'],
                    masked_embeddings=features['masked_text_embeds_proj'],
                    image_embeddings=image_embeddings_for_text,
                    mask=features['mask_indices']
                )
                features['text_res_loss'] = text_res_loss
        else:
            # 不增强时，设置重构损失为0
            device = image.device if 'image' in features else (text.device if 'text' in features else cog.device)
            features['text_res_loss'] = torch.tensor(0.0, device=device)

        # 计算认知量表重构损失（如果需要）
        if (self.modalities == 'cog' or self.modalities == 'speech_cog') and augment:
            if hasattr(self, 'use_ft_transformer') and self.use_ft_transformer:
                # 确定用于认知量表恢复的交叉注意力特征
                # 在speech_cog模态下使用语音表格数据特征，其他情况不使用
                if self.modalities == 'speech_cog':
                    cross_embeddings_for_cog = features['text_embeds']
                else:
                    cross_embeddings_for_cog = None
                
                # 计算重构损失
                cog_res_loss, _, _ = self.cog_restorer(
                    original_feat=features['cog'],
                    masked_embeddings=features['masked_cog'].unsqueeze(1),
                    image_embeddings=cross_embeddings_for_cog,
                    mask=features['cog_mask_indices']
                )
                features['cog_res_loss'] = cog_res_loss
            else:
                # 原始处理方式
                cog_res_loss = self.cog_restorer(
                    original_feat=features['cog'],
                    masked_embeddings=features['masked_cog_embeds_proj'],
                    image_embeddings=None,
                    mask=features['cog_mask_indices']
                )
                features['cog_res_loss'] = cog_res_loss
        else:
            # 不增强时，设置重构损失为0
            device = text.device if 'text' in features else cog.device
            features['cog_res_loss'] = torch.tensor(0.0, device=device)
        
        return features

    def _teacher_forward(self, image, text, cog, label, id, patient_id, device):
        """
        教师模型前向传播逻辑，处理不同模态的特征提取和损失计算
        
        Args:
            image: 图像数据 [batch_size, 1, 128, 128, 128]
            text: 文本/语音特征 [batch_size, feature_dim]
            cog: 认知量表特征 [batch_size, cog_feature_dim]
            label: 标签数据
            id: 样本ID
            patient_id: 患者ID，用于对比学习中的正样本识别
            device: 设备类型
            
        Returns:
            dict: 包含所有损失值和预测结果
        """
        # 初始化损失变量
        loss_itc = torch.tensor(0.0, device=device)
        image_res_loss = torch.tensor(0.0, device=device)
        text_res_loss = torch.tensor(0.0, device=device)
        cog_res_loss = torch.tensor(0.0, device=device)
        
        # ========== 特征提取 ==========
        features = self._extract_features(image, text, cog, augment=True)

        # ========== 交叉注意力增强 ==========
        # 使用交叉注意力增强特征表示，用于对比学习
        if self.modalities == 'both' and self.use_cross_attention:
            # 应用交叉注意力增强特征
            enhanced_image_feat = self.cross_attention(features['image_feat'], features['text_feat'])
            enhanced_text_feat = self.cross_attention(features['text_feat'], features['image_feat'])
            # 对增强后的特征进行归一化，确保与原始特征格式一致
            enhanced_image_feat = F.normalize(enhanced_image_feat, dim=-1)
            enhanced_text_feat = F.normalize(enhanced_text_feat, dim=-1)
            # 更新特征字典
            features['enhanced_image_feat'] = enhanced_image_feat
            features['enhanced_text_feat'] = enhanced_text_feat
        elif self.modalities == 'speech_cog' and self.use_cross_attention:
            # 应用交叉注意力增强语音和认知量表特征
            enhanced_speech_feat = self.cross_attention(features['text_feat'], features['cog_feat'])
            enhanced_cog_feat = self.cross_attention(features['cog_feat'], features['text_feat'])
            # 对增强后的特征进行归一化
            enhanced_speech_feat = F.normalize(enhanced_speech_feat, dim=-1)
            enhanced_cog_feat = F.normalize(enhanced_cog_feat, dim=-1)
            # 更新特征字典
            features['enhanced_speech_feat'] = enhanced_speech_feat
            features['enhanced_cog_feat'] = enhanced_cog_feat

        # ========== 损失计算 ==========
        # 对比学习损失（仅多模态时使用）
        # 保护可学习温度参数，避免出现非有限值传递到ITC损失
        with torch.no_grad():
            if not torch.isfinite(self.itc_temp):
                self.itc_temp.fill_(0.07)

        if self.modalities == 'both':
            # 使用增强后的特征进行对比学习
            if self.use_cross_attention:
                loss_itc = itc_loss(features['enhanced_image_feat'], features['enhanced_text_feat'], patient_id, label, temp=self.itc_temp)
            else:
                loss_itc = itc_loss(features['image_feat'], features['text_feat'], patient_id, label, temp=self.itc_temp)
        elif self.modalities == 'speech_cog':
            # 使用增强后的特征进行对比学习
            if self.use_cross_attention:
                loss_itc = itc_loss(features['enhanced_speech_feat'], features['enhanced_cog_feat'], patient_id, label, temp=self.itc_temp)
            else:
                loss_itc = itc_loss(features['text_feat'], features['cog_feat'], patient_id, label, temp=self.itc_temp)

        # 图像重构损失（仅视觉模态或多模态时使用）
        if self.modalities == 'both' or self.modalities == 'vision':
            # 多模态时传入文本嵌入，单模态视觉时不传入
            text_emb_for_vision = features['text_embeds'] if self.modalities == 'both' else None
            image_res_loss, _, _ = self.vision_transformer(
                features['image'], text_emb=text_emb_for_vision, mask_ratio=self.image_mask_ratio
            )

        # 文本重构损失（仅文本/语音模态或多模态时使用）
        if self.modalities == 'both' or self.modalities == 'text' or self.modalities == 'speech_cog':
            # 文本重构损失已在特征提取阶段计算
            text_res_loss = features['text_res_loss']

        # 认知量表重构损失（仅认知量表模态或语音+认知量表模态时使用）
        if self.modalities == 'cog' or self.modalities == 'speech_cog':
            # 认知量表重构损失已在特征提取阶段计算
            cog_res_loss = features['cog_res_loss']

        # 教师模型推理
        if self.modalities == 'both':
            # 多模态融合
            h_concat = torch.cat([features['image_embeds'][:,0], features['text_features']], dim=-1)
            y_hat_teacher = self.mlp(h_concat)
            loss_cls_tea = self.criterion(y_hat_teacher, label)
        elif self.modalities == 'vision':
            # 仅视觉模态
            y_hat_teacher = self.mlp_vision(features['image_embeds'][:,0])
            loss_cls_tea = self.criterion(y_hat_teacher, label)
        elif self.modalities == 'text':
            # 仅文本/语音模态
            y_hat_teacher = self.mlp_text(features['text_features'])
            loss_cls_tea = self.criterion(y_hat_teacher, label)
        elif self.modalities == 'cog':
            # 仅认知量表模态
            y_hat_teacher = self.mlp_cog(features['cog_features'])
            loss_cls_tea = self.criterion(y_hat_teacher, label)
        elif self.modalities == 'speech_cog':
            # 语音+认知量表融合
            h_concat = torch.cat([features['text_features'], features['cog_features']], dim=-1)
            y_hat_teacher = self.mlp_speech_cog(h_concat)
            loss_cls_tea = self.criterion(y_hat_teacher, label)

        # ========== 总损失计算 ==========
        loss_cls_tea = loss_cls_tea * 1
        
        # 根据选择的模态调整损失权重
        if self.modalities == 'both':
            text_res_loss = text_res_loss * 1
            image_res_loss = image_res_loss * 1
            total_loss = (
                loss_cls_tea +  # 教师分类损失
                loss_itc +      # 对比学习损失
                text_res_loss + # 文本恢复损失
                image_res_loss  # 图像恢复损失
            )
        elif self.modalities == 'vision':
            # 仅视觉模态
            image_res_loss = image_res_loss * 1
            total_loss = (
                loss_cls_tea +  # 教师分类损失
                image_res_loss  # 图像恢复损失
            )
        elif self.modalities == 'text':
            # 仅文本/语音模态
            text_res_loss = text_res_loss * 1
            total_loss = (
                loss_cls_tea +  # 教师分类损失
                text_res_loss   # 文本恢复损失
            )
        elif self.modalities == 'cog':
            # 仅认知量表模态
            cog_res_loss = cog_res_loss * 1
            total_loss = (
                loss_cls_tea +  # 教师分类损失
                cog_res_loss    # 认知量表恢复损失
            )
        elif self.modalities == 'speech_cog':
            # 语音+认知量表模态
            text_res_loss = text_res_loss * 1
            cog_res_loss = cog_res_loss * 1
            total_loss = (
                loss_cls_tea +  # 教师分类损失
                loss_itc +      # 对比学习损失
                text_res_loss + # 文本恢复损失
                cog_res_loss    # 认知量表恢复损失
            )

        # 返回所有损失值和预测结果
        return {
            "loss": total_loss,
            'loss_itc': loss_itc,
            'loss_text_res': text_res_loss,
            'loss_image_res': image_res_loss,
            'loss_cog_res': cog_res_loss,
            'loss_cls': loss_cls_tea,
            'loss_cls_teacher': loss_cls_tea,
            'loss_cls_student': 0.0,
            'loss_kl': 0.0,
            'loss_feat': 0.0,
            'y_hat_teacher': y_hat_teacher,
            'y_hat_student': None
        }

    def forward(self, samples):
        """
        模型前向传播函数，根据不同的训练模式执行相应的逻辑
        
        训练模式说明：
        1. 教师模型训练模式 (training_mode='teacher'):
           - 训练完整的多模态教师模型
           - 同时使用图像和文本/语音模态
           - 包含对比学习损失和重构损失
           
        2. 学生模型蒸馏模式 (training_mode='student'):
           - 冻结教师模型参数
           - 训练单模态学生模型
           - 使用教师模型的软目标和中间特征进行监督
           - 包含分类损失、蒸馏损失和特征级蒸馏损失
           
        3. 单模态训练模式 (training_mode='single'):
           - 只训练指定模态的模型
           - 模态由modalities参数指定
           - 适用于单独训练视觉或文本/语音模型
        
        Args:
            samples: 输入样本，包含image, text, cog, label, id, patient_id
            
        Returns:
            dict: 包含所有损失值和预测结果
        """
        # 解包样本
        if len(samples) == 4:
            # 兼容旧格式（无patient_id和cog）
            image, text, label, id = samples
            patient_id = id  # 当没有patient_id时，使用样本ID作为默认值
            cog = torch.zeros_like(text)  # 当没有cog时，使用零向量
        elif len(samples) == 5:
            # 新格式（包含cog但无patient_id）- 数据集返回格式：image, features, cog_features, label, id
            image, text, cog, label, id = samples
            patient_id = id  # 当没有patient_id时，使用样本ID作为默认值
        else:
            # 新格式（包含patient_id和cog）
            image, text, cog, label, id, patient_id = samples
        
        image = image.unsqueeze(1).cuda()  # [bs, 1, 128, 128, 128]
        text = text.cuda()
        cog = cog.cuda()
        label = label.cuda()
        id = id.cuda()
        patient_id = patient_id.cuda() if isinstance(patient_id, torch.Tensor) else patient_id
        device = label.device
        
        # 禁用学生模型模式 - 只计算教师模型
        if self.disable_student:
            return self._teacher_forward(image, text, cog, label, id, patient_id, device)

        # ===== 教师模型训练阶段 =====
        if not self.freeze_teacher:
            return self._teacher_forward(image, text, cog, label, id, patient_id, device)

        # ===== 学生模型蒸馏训练阶段 =====
        else:
            # 确保教师模型参数已冻结
            self._freeze_teacher_parameters()
            
            # ========== 数据准备 ==========
            aug_text = self.augment_text(text)  # 增强后的表格

            # 教师模型 - 使用原始数据生成软目标和特征 (禁用梯度计算)
            with torch.no_grad():
                # 使用统一的特征提取方法获取教师特征
                teacher_features = self._extract_features(image, text, cog, augment=False)
                
                # 根据模态类型处理教师模型
                if self.modalities == 'speech_cog':
                    # 语音+认知量表模态
                    # 从提取的特征中获取文本和认知量表特征
                    text_features_tea = teacher_features['text_features']
                    cog_features_tea = teacher_features['cog_features']
                    
                    # 融合特征并使用相应分类器
                    h_concat_tea = torch.cat([text_features_tea, cog_features_tea], dim=-1)
                    y_hat_teacher = self.mlp_speech_cog(h_concat_tea)
                    
                    # 获取教师特征用于特征级蒸馏
                    teacher_text_features = text_features_tea
                else:
                    # 语音-图像模态或其他模态
                    # 教师模型特征提取
                    image_embeds_tea, _, _, _ = self.vision_transformer.forward_encoder(
                        img=image, text_emb=None, mask_ratio=0)
                    image_embeds_tea = self.vision_proj(image_embeds_tea)
                    
                    # 文本特征处理
                    if hasattr(self, 'use_ft_transformer') and self.use_ft_transformer:
                        # 生成软目标
                        _, text_features_tea, _ = self.text_restorer(
                            original_feat=text,
                            masked_embeddings=text.unsqueeze(1),
                            image_embeddings=image_embeds_tea,
                            mask=None,
                            compute_loss=False
                        )
                        h_concat_tea = torch.cat([image_embeds_tea[:,0], text_features_tea], dim=-1)
                        y_hat_teacher = self.mlp(h_concat_tea)
                        
                        # 获取教师特征用于特征级蒸馏
                        teacher_text_features = teacher_features['text_features']
                    else:
                        # 原始处理方式
                        text_embeds_tea = self.text_proj(text).unsqueeze(1)
                        h_concat_tea = torch.cat([image_embeds_tea[:,0], text_embeds_tea[:,0]], dim=-1)
                        y_hat_teacher = self.mlp(h_concat_tea)
                        
                        # 获取教师特征用于特征级蒸馏
                        teacher_text_features = teacher_features['text_features']
            
            # 学生模型 - 使用与教师模型相同的FT-Transformer处理语音特征
            if hasattr(self, 'use_ft_transformer') and self.use_ft_transformer:
                # 使用学生的TableFTTRestorer
                _, student_text_features, _ = self.student_text_restorer(
                    original_feat=aug_text,
                    masked_embeddings=aug_text.unsqueeze(1),
                    image_embeddings=None,  # 学生模型只使用语音特征
                    mask=None,
                    compute_loss=False
                )
                y_hat_student = self.mlp_speech(student_text_features)
            else:
                # 原始处理方式（备用）
                aug_text_embeds = self.text_proj(aug_text).unsqueeze(1)
                student_text_features = aug_text_embeds[:,0]
                y_hat_student = self.mlp_speech(student_text_features)
            
            # ========== 损失计算 ==========
            # 计算学生模型分类损失 (硬目标监督)
            loss_cls_stu = self.criterion(y_hat_student, label) * self.student_cls_weight
            
            # 计算蒸馏损失 (软目标监督)
            loss_kd = distillation_loss(y_hat_teacher, y_hat_student, T=self.distillation_temp) * self.kd_weight
            
            # 计算特征级蒸馏损失 (中间特征对齐)
            loss_feat = self.feat_criterion(student_text_features, teacher_text_features) * self.feat_distill_weight
            
            # 组合总损失 - 学生模型只关注分类和蒸馏
            total_loss = loss_cls_stu + loss_kd + loss_feat

        # 返回所有损失值和预测结果
        return {
            "loss": total_loss,
            'loss_itc': 0.0,
            'loss_text_res': 0.0,
            'loss_image_res': 0.0,
            'loss_cls': loss_cls_stu + loss_kd,
            'loss_cls_teacher': 0.0,
            'loss_cls_student': loss_cls_stu,
            'loss_kl': loss_kd,
            'loss_feat': loss_feat,
            'y_hat_teacher': y_hat_teacher,
            'y_hat_student': y_hat_student
        }

        
    def predict(self, samples):
        """预测函数，同时返回教师模型和学生模型的预测结果，以及注意力权重"""
        # 支持字典形式输入以增强灵活性
        if isinstance(samples, dict):
            image = samples.get('image', None)
            text = samples.get('text', None)
            cog = samples.get('cog', None)
            label = samples.get('label', None)
            id = samples.get('id', None)
            
            if image is not None:
                image = image.unsqueeze(1).cuda()
            if text is not None:
                text = text.cuda()
            if cog is not None:
                cog = cog.cuda()
            if label is not None:
                label = label.cuda()
        else:
            # 保持原有接口兼容性
            if len(samples) == 4:
                image,text,label,id = samples
                cog = torch.zeros_like(text)  # 当没有cog时，使用零向量
            elif len(samples) == 5:
                # 新格式（包含cog但无patient_id）- 数据集返回格式：image, features, cog_features, label, id
                image,text,cog,label,id = samples
            else:
                image,text,cog,label,id,_ = samples
            image = image.unsqueeze(1).cuda() # bs c 128 128 128
            text = text.cuda()
            cog = cog.cuda()
            label = label.cuda()
        # id = id.cuda()
        
        # 初始化注意力权重
        text_attn_probs = None
        cog_attn_probs = None
        
        # 教师模型推理
        if self.modalities == 'both' or self.modalities == 'vision':
            image_embeds,attns, mask,ids_restore = self.vision_transformer.forward_encoder(
                img = image,
                text_emb=None,
                mask_ratio=0)
            image_embeds = self.vision_proj(image_embeds) #[24, 513, 768]
        else:
            image_embeds = None

        # 根据模态类型处理特征
        if self.modalities == 'cog':
            # 仅认知量表模态
            if hasattr(self, 'use_ft_transformer') and self.use_ft_transformer:
                # 使用cog_restorer处理认知量表特征
                _, cog_embeds_proj_teacher, cog_attn_probs = self.cog_restorer(
                    original_feat=cog,
                    masked_embeddings=cog.unsqueeze(1),
                    image_embeddings=None,
                    mask=None,
                    compute_loss=False
                )
            else:
                # 原始处理方式
                cog_embeds_proj_teacher = self.cog_proj(cog)
            
            # 使用认知量表分类器
            y_hat_logits = self.mlp_cog(cog_embeds_proj_teacher)
            # 学生模型预测（使用相同特征）
            y_hat_speech_logits = self.mlp_speech(cog_embeds_proj_teacher)
        elif self.modalities == 'speech_cog':
            # 语音+认知量表模态
            if hasattr(self, 'use_ft_transformer') and self.use_ft_transformer:
                # 处理文本/语音特征
                _, text_embeds_proj_teacher, text_attn_probs = self.text_restorer(
                    original_feat=text,
                    masked_embeddings=text.unsqueeze(1),
                    image_embeddings=None,
                    mask=None,
                    compute_loss=False
                )
                # 处理认知量表特征
                _, cog_embeds_proj_teacher, cog_attn_probs = self.cog_restorer(
                    original_feat=cog,
                    masked_embeddings=cog.unsqueeze(1),
                    image_embeddings=None,
                    mask=None,
                    compute_loss=False
                )
            else:
                # 原始处理方式
                text_embeds_proj_teacher = self.text_proj(text)
                cog_embeds_proj_teacher = self.cog_proj(cog)
            
            # 融合特征并使用相应分类器
            h_concat_teacher = torch.cat([text_embeds_proj_teacher, cog_embeds_proj_teacher], dim=-1)
            y_hat_logits = self.mlp_speech_cog(h_concat_teacher)
            # 学生模型预测（使用文本特征）
            _, text_embeds_proj_student, _ = self.student_text_restorer(
                original_feat=text,
                masked_embeddings=text.unsqueeze(1),
                image_embeddings=None,  # 学生模型只使用语音特征
                mask=None,
                compute_loss=False
            )
            y_hat_speech_logits = self.mlp_speech(text_embeds_proj_student)
        else:
            # 文本/语音或视觉模态
            if hasattr(self, 'use_ft_transformer') and self.use_ft_transformer:
                # 教师模型处理 - 使用与forward方法一致的接口
                _, text_embeds_proj_teacher, text_attn_probs = self.text_restorer(
                    original_feat=text,
                    masked_embeddings=text.unsqueeze(1),
                    image_embeddings=image_embeds,
                    mask=None,
                    compute_loss=False
                )
                
                # 定义text_embeds_teacher以确保所有代码路径都有该变量
                text_embeds_teacher = text_embeds_proj_teacher.unsqueeze(1)
                
                # 学生模型处理
                _, text_embeds_proj_student, _ = self.student_text_restorer(
                    original_feat=text,
                    masked_embeddings=text.unsqueeze(1),
                    image_embeddings=None,  # 学生模型只使用语音特征
                    mask=None,
                    compute_loss=False
                )
            else:
                # 原始处理方式
                text_embeds_teacher = self.text_proj(text).unsqueeze(1)  # bs txt_len 768 
                text_embeds_proj_teacher = text_embeds_teacher[:,0]
                text_embeds_proj_student = text_embeds_proj_teacher
            
            # 教师模型预测 - 根据模态类型选择正确的分类器
            if self.modalities == 'both':
                # 多模态融合
                h_concat_teacher = torch.cat([image_embeds[:,0], text_embeds_proj_teacher], dim=-1)
                y_hat_logits = self.mlp(h_concat_teacher)
            elif self.modalities == 'vision':
                # 仅视觉模态
                y_hat_logits = self.mlp_vision(image_embeds[:,0])
            elif self.modalities == 'text':
                # 仅文本/语音模态
                y_hat_logits = self.mlp_text(text_embeds_proj_teacher)
            else:
                # 默认使用文本/语音分类器
                y_hat_logits = self.mlp_text(text_embeds_proj_teacher)
            
            # 学生模型预测
            y_hat_speech_logits = self.mlp_speech(text_embeds_proj_student)
        
        # 使用softmax激活函数转换为概率值
        y_hat_prob = torch.softmax(y_hat_logits, dim=-1)
        y_hat_speech_prob = torch.softmax(y_hat_speech_logits, dim=-1)

        # 返回概率值、特征和注意力权重
        return y_hat_prob, y_hat_speech_prob, image_embeds[:,0] if image_embeds is not None else None, text_embeds_proj_teacher if 'text_embeds_proj_teacher' in locals() else None, text_attn_probs, cog_attn_probs

    
    

def load_pretrained_vision_encoder(model, pretrained_path, freeze_layers=0):
    """加载三分类预训练的视觉部分到二分类模型
    
    Args:
        model: 目标模型
        pretrained_path: 预训练权重路径
        freeze_layers: 要冻结的视觉Transformer层数，0表示不冻结，-1表示冻结所有
    """
    
    # 加载预训练权重
    pretrained_dict = torch.load(pretrained_path)
    model_dict = model.state_dict()
    
    # 只保留视觉编码器相关的权重
    vision_keys = []
    for k in pretrained_dict.keys():
        if k.startswith('vision_transformer.') or k.startswith('vision_proj.'):
            vision_keys.append(k)
    
    # 更新当前模型的视觉部分权重
    update_count = 0
    for k in vision_keys:
        if k in model_dict and model_dict[k].shape == pretrained_dict[k].shape:
            model_dict[k] = pretrained_dict[k]
            update_count += 1
        elif k.startswith('vision_proj.'):
            # 处理vision_proj的特殊情况：当前模型可能是Sequential，而预训练是简单的Linear
            if hasattr(model, 'vision_proj') and isinstance(model.vision_proj, nn.Sequential):
                # 检查Sequential中的第一个线性层
                if len(model.vision_proj) > 0 and isinstance(model.vision_proj[0], nn.Linear):
                    # 构建对应的键名
                    seq_key = f"vision_proj.0.{k.split('.')[-1]}"
                    if seq_key in model_dict and model_dict[seq_key].shape == pretrained_dict[k].shape:
                        model_dict[seq_key] = pretrained_dict[k]
                        update_count += 1
                        print(f"✅ 加载到序列层: {seq_key}")
                    else:
                        print(f"⚠️  跳过: {k} - 序列层形状不匹配")
            else:
                print(f"⚠️  跳过: {k} - 形状不匹配")
        else:
            print(f"⚠️  跳过: {k} - 形状不匹配")
    
    # 加载权重（strict=False允许分类层不匹配）
    model.load_state_dict(model_dict, strict=False)
    
    print(f"✅ 视觉预训练权重加载完成")
    print(f"📊 更新了 {update_count}/{len(vision_keys)} 个视觉层")
    print(f"🎯 分类层保持二分类随机初始化")

    # 分层冻结策略
    if freeze_layers == 0:
        print(f"✅ 视觉预训练权重不冻结，将参与训练")
    else:
        # 统计可训练和冻结的参数
        total_params = 0
        frozen_params = 0
        trainable_params = 0
        
        # 冻结视觉Transformer的层
        if hasattr(model, 'vision_transformer'):
            vit = model.vision_transformer
            
            # 确定要冻结的层数
            if freeze_layers == -1:
                # 冻结所有层
                freeze_count = len(vit.blocks) if hasattr(vit, 'blocks') else len(vit.encoder.layers)
            else:
                freeze_count = freeze_layers
            
            print(f"\n===== 分层冻结配置 =====")
            print(f"要冻结的视觉Transformer层数: {freeze_count}")
            
            # 冻结patch embedding和位置编码
            if hasattr(vit, 'patch_embed_3d'):
                for param in vit.patch_embed_3d.parameters():
                    param.requires_grad = False
                print("✅ 冻结: patch_embed_3d")
            
            if hasattr(vit, 'pos_embed_3d'):
                vit.pos_embed_3d.requires_grad = False
                print("✅ 冻结: pos_embed_3d")
            
            if hasattr(vit, 'cls_token'):
                vit.cls_token.requires_grad = False
                print("✅ 冻结: cls_token")
            
            # 冻结Transformer层
            if hasattr(vit, 'blocks'):
                # MAE ViT结构
                for i, block in enumerate(vit.blocks):
                    if i < freeze_count:
                        for param in block.parameters():
                            param.requires_grad = False
                        print(f"✅ 冻结: block_{i}")
                    else:
                        print(f"✅ 可训练: block_{i}")
            elif hasattr(vit, 'encoder') and hasattr(vit.encoder, 'layers'):
                # 标准ViT结构
                for i, layer in enumerate(vit.encoder.layers):
                    if i < freeze_count:
                        for param in layer.parameters():
                            param.requires_grad = False
                        print(f"✅ 冻结: layer_{i}")
                    else:
                        print(f"✅ 可训练: layer_{i}")
            
            # 保持norm层可训练
            if hasattr(vit, 'norm'):
                print("✅ 可训练: norm")
        
        # 保持vision_proj可训练，因为它是连接视觉编码器和其他模块的桥梁
        print("✅ 可训练: vision_proj")
        
        # 统计参数状态
        for name, param in model.named_parameters():
            if "vision_transformer" in name or "vision_proj" in name:
                total_params += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
                else:
                    frozen_params += param.numel()
        
        print(f"\n===== 参数状态统计 =====")
        print(f"视觉相关总参数: {total_params:,} 个")
        print(f"冻结参数: {frozen_params:,} 个 ({frozen_params/total_params*100:.1f}%)")
        print(f"可训练参数: {trainable_params:,} 个 ({trainable_params/total_params*100:.1f}%)")
        print("=====================")
    
    return model

# 添加教师模型权重加载方法到MedBLIPModel类
import types

def load_teacher_weights(self, teacher_model_path):
    """从保存的教师模型中加载权重，支持分步训练策略"""
    print(f"从 {teacher_model_path} 加载教师模型权重...")
    
    # 加载教师模型权重
    checkpoint = torch.load(teacher_model_path, map_location='cpu')
    
    # 处理可能的checkpoint结构（如包含'model'键）
    if 'model' in checkpoint:
        teacher_dict = checkpoint['model']
    else:
        teacher_dict = checkpoint
    
    model_dict = self.state_dict()
    
    # 过滤并复制匹配的权重
    update_count = 0
    skip_count = 0
    for k, v in teacher_dict.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            model_dict[k] = v
            update_count += 1
        else:
            skip_count += 1
    
    # 加载权重
    self.load_state_dict(model_dict, strict=False)
    
    print(f"✅ 教师模型权重加载完成")
    print(f"📊 成功更新: {update_count} 个参数")
    print(f"⚠️  跳过不匹配: {skip_count} 个参数")
    
    # 如果设置了冻结教师模型，则重新冻结参数
    if self.freeze_teacher:
        self._freeze_teacher_parameters()
        print("✅ 教师模型参数已重新冻结")
    
    return self

# 动态添加方法到MedBLIPModel类
MedBLIPModel.load_teacher_weights = load_teacher_weights