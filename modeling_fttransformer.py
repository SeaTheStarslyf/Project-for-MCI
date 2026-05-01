from typing import List, Dict, Optional, Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from medblip.mae_vit import CrossAttention

class FeatureTokenizer(nn.Module):
    """
    特征分词器：将表格特征转换为token嵌入
    处理两种类型的特征：数值型和类别型
    """
    def __init__(self, num_numerical_features: int, categorical_cardinalities: List[int], d_token: int, use_positional_embedding: bool = True):
        super().__init__()
        
        # 数值特征处理
        self.numerical_embeddings = nn.ModuleList([
            nn.Linear(1, d_token) for _ in range(num_numerical_features)
        ])
        
        # 类别特征处理
        self.categorical_embeddings = nn.ModuleList([
            nn.Embedding(cat_size, d_token) for cat_size in categorical_cardinalities
        ])
        
        self.num_numerical = num_numerical_features
        self.num_categorical = len(categorical_cardinalities)
        self.d_token = d_token
        self.use_positional_embedding = use_positional_embedding
        
        # 添加位置嵌入
        if self.use_positional_embedding:
            total_features = num_numerical_features + len(categorical_cardinalities)
            self.position_embedding = nn.Parameter(torch.randn(1, total_features, d_token))
    
    def forward(self, numerical_features: Optional[torch.Tensor] = None, 
                categorical_features: Optional[torch.Tensor] = None):
        tokens = []
        
        # 处理数值特征
        if numerical_features is not None and self.num_numerical > 0:
            for i in range(self.num_numerical):
                feat = numerical_features[:, i].unsqueeze(1)  # [B, 1]
                token = self.numerical_embeddings[i](feat)  # [B, d_token]
                tokens.append(token)
        
        # 处理类别特征
        if categorical_features is not None and self.num_categorical > 0:
            for i in range(self.num_categorical):
                feat = categorical_features[:, i]  # [B]
                token = self.categorical_embeddings[i](feat)  # [B, d_token]
                tokens.append(token)
        
        # 将所有token拼接起来
        tokens = torch.stack(tokens, dim=1)  # [B, num_features, d_token]
        
        # 添加位置嵌入
        if self.use_positional_embedding:
            tokens = tokens + self.position_embedding
        
        return tokens

class MultiHeadAttention(nn.Module):
    """
    多头自注意力机制
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 确保d_model能被num_heads整除
        assert d_model % num_heads == 0
        
        # 线性投影层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        # x: [B, N, d_model], where N is the number of tokens
        batch_size, num_tokens, _ = x.shape
        
        # 应用层归一化
        residual = x
        x = self.layer_norm(x)
        
        # 线性投影并分割为多个头
        q = self.W_q(x).view(batch_size, num_tokens, self.num_heads, self.d_k).transpose(1, 2)  # [B, num_heads, N, d_k]
        k = self.W_k(x).view(batch_size, num_tokens, self.num_heads, self.d_k).transpose(1, 2)  # [B, num_heads, N, d_k]
        v = self.W_v(x).view(batch_size, num_tokens, self.num_heads, self.d_k).transpose(1, 2)  # [B, num_heads, N, d_k]
        
        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)  # [B, num_heads, N, N]
        
        # 应用注意力掩码（如果提供）
        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(attention_mask == 0, -1e9)
        
        # 应用softmax和dropout
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # 计算加权和
        attn_output = torch.matmul(attn_probs, v)  # [B, num_heads, N, d_k]
        
        # 合并多个头
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.d_model)  # [B, N, d_model]
        
        # 应用输出投影并添加残差连接
        output = self.W_o(attn_output) + residual  # [B, N, d_model]
        
        return output, attn_probs

class CrossAttention(nn.Module):
    """
    多头交叉注意力机制
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x, y):
        B, N, C = x.shape
        B, M, C = y.shape
        
        # 应用层归一化
        residual = x
        x = self.layer_norm(x)
        y = self.layer_norm(y)
        
        q = self.q(x).reshape(B,N,self.num_heads, C // self.num_heads).permute(0,2,1,3)
        k = self.k(y).reshape(B,M,self.num_heads, C // self.num_heads).permute(0,2,1,3)
        v = self.v(y).reshape(B,M,self.num_heads, C // self.num_heads).permute(0,2,1,3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        # 添加残差连接
        output = x + residual
        
        return output

class FeedForward(nn.Module):
    """
    前馈神经网络
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor):
        # x: [B, N, d_model]
        
        # 应用层归一化
        residual = x
        x = self.layer_norm(x)
        
        # 应用前馈网络
        x = self.dropout(F.gelu(self.linear1(x)))
        x = self.linear2(x)
        
        # 添加残差连接
        output = x + residual  # [B, N, d_model]
        
        return output

class TransformerBlock(nn.Module):
    """
    Transformer块，包含自注意力、交叉注意力和前馈神经网络
    """
    def __init__(self, dim, num_heads, d_ff, dropout=0.0):
        super().__init__()
        self.self_attn = MultiHeadAttention(dim, num_heads, dropout)
        self.cross_attn = CrossAttention(dim, num_heads, qkv_bias=True, attn_drop=dropout, proj_drop=dropout)
        self.ffn = FeedForward(dim, d_ff, dropout)
        
    def forward(self, x, cross_emb=None, attention_mask=None):
        # 自注意力
        x, attn_probs = self.self_attn(x, attention_mask)
        # 交叉注意力（如果提供了交叉嵌入）
        if cross_emb is not None:
            x = self.cross_attn(x, cross_emb)
        # 前馈神经网络
        x = self.ffn(x)
        return x, attn_probs

class TransformerEncoder(nn.Module):
    """
    Transformer编码器
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, num_layers: int, dropout: float = 0.0):
        super().__init__()
        
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
    
    def forward(self, x: torch.Tensor, cross_emb: Optional[torch.Tensor] = None, attention_mask: Optional[torch.Tensor] = None):
        # x: [B, N, d_model]
        # cross_emb: [B, M, d_model] (例如图像嵌入)
        
        all_attn_probs = []
        for block in self.layers:
            x, attn_probs = block(x, cross_emb, attention_mask)
            all_attn_probs.append(attn_probs)
        
        return x, all_attn_probs

class FTTransformer(nn.Module):
    """
    Feature Tokenizer Transformer for tabular data
    """
    def __init__(self, 
                 num_numerical_features: int,
                 categorical_cardinalities: List[int] = None,
                 d_token: int = 192,
                 num_heads: int = 8,
                 d_ff: int = 768,
                 num_layers: int = 6,
                 dropout: float = 0.1,
                 use_cls_token: bool = True,
                 output_dim: int = 768,
                 use_positional_embedding: bool = True):
        super().__init__()
        
        if categorical_cardinalities is None:
            categorical_cardinalities = []
            
        # 特征分词器
        self.feature_tokenizer = FeatureTokenizer(
            num_numerical_features=num_numerical_features,
            categorical_cardinalities=categorical_cardinalities,
            d_token=d_token,
            use_positional_embedding=use_positional_embedding
        )
        
        # 是否使用CLS token
        self.use_cls_token = use_cls_token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))
        
        # Transformer编码器
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
                categorical_features: Optional[torch.Tensor] = None,
                cross_emb: Optional[torch.Tensor] = None):
        # 对特征进行分词
        tokens = self.feature_tokenizer(numerical_features, categorical_features)
        batch_size = tokens.shape[0]
        
        # 添加CLS token（如果使用）
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            tokens = torch.cat([cls_tokens, tokens], dim=1)
        
        # 通过Transformer编码器
        encoder_output, all_attn_probs = self.encoder(tokens, cross_emb)
        
        # 如果使用CLS token，只使用CLS token的输出
        if self.use_cls_token:
            output = encoder_output[:, 0]
        else:
            # 否则，对所有token的输出进行平均池化
            output = encoder_output.mean(dim=1)
        
        # 通过输出投影层
        output = self.output_projection(output)
        
        return output, all_attn_probs

class TableFTTRestorer(nn.Module):
    """
    使用FT-Transformer处理表格数据，并提供与SimpleTableRestorer兼容的接口
    采用与图像处理相同的单向交叉注意力机制
    """
    def __init__(self, 
                 num_numerical_features: int,
                 categorical_cardinalities: List[int] = None,
                 hidden_dim: int = 768,
                 dropout: float = 0.1,
                 num_cross_attn_heads: int = 8,
                 use_positional_embedding: bool = True):
        super().__init__()
        
        # 保存参数以便在forward中使用
        self.num_numerical_features = num_numerical_features
        self.categorical_cardinalities = categorical_cardinalities if categorical_cardinalities is not None else []
        
        # 检查是否有任何特征
        has_numerical = num_numerical_features is not None and num_numerical_features > 0
        has_categorical = categorical_cardinalities is not None and len(categorical_cardinalities) > 0
        
        if not (has_numerical or has_categorical):
            raise ValueError("必须提供至少一种类型的特征：数值特征或类别特征")
        
        # FT-Transformer用于表格特征处理
        # 确保d_token与hidden_dim匹配，避免维度不匹配问题
        self.ft_transformer = FTTransformer(
            num_numerical_features=num_numerical_features,
            categorical_cardinalities=self.categorical_cardinalities,
            d_token=hidden_dim,  # 使用与hidden_dim相同的d_token
            output_dim=hidden_dim,
            dropout=dropout,
            use_positional_embedding=use_positional_embedding
        )
        
        # 添加单向Cross-Attention层：与图像处理一致，表格特征作为query，图像特征作为key和value
        self.cross_attn = CrossAttention(
            dim=hidden_dim,
            num_heads=num_cross_attn_heads,
            qkv_bias=True,
            attn_drop=dropout,
            proj_drop=dropout
        )
        
        # LayerNorm层，与图像处理保持一致
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # 重构层：尝试重构表格特征
        self.reconstruction_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_numerical_features)
        )
    
    def forward(self, original_feat, masked_embeddings=None, image_embeddings=None, mask=None, compute_loss=True):
        """
        与SimpleTableRestorer兼容的接口，使用与图像处理相同的单向交叉注意力机制
        
        Args:
            original_feat: 原始表格特征，可以是仅包含数值特征的张量，或包含数值和类别特征的元组
            masked_embeddings: 掩码后的表格嵌入 (未使用，但保留接口兼容性)
            image_embeddings: 图像嵌入
            mask: 掩码位置
            compute_loss: 是否计算重构损失
        
        Returns:
            如果compute_loss为True: (loss, table_embedding)
            如果compute_loss为False: (0, table_embedding)
        """
        # 输入验证
        if original_feat is None:
            raise ValueError("original_feat cannot be None")
        
        # 验证至少有一个特征被提供
        has_numerical = self.num_numerical_features > 0
        has_categorical = len(self.categorical_cardinalities) > 0
        
        if not (has_numerical or has_categorical):
            raise ValueError("模型未配置任何特征处理能力")
        
        # 处理不同类型的输入
        numerical_features = None
        categorical_features = None
        
        # 判断输入是元组(数值特征, 类别特征)还是单个数值特征张量
        if isinstance(original_feat, tuple) and len(original_feat) == 2:
            numerical_features, categorical_features = original_feat
        else:
            # 假设是数值型特征
            numerical_features = original_feat
        
        # 使用FT-Transformer处理表格特征，并在每个Transformer块中进行跨模态交互
        table_embedding, all_attn_probs = self.ft_transformer(
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            cross_emb=image_embeddings  # 将图像嵌入传递给FTTransformer
        )
        
        # 如果提供了图像嵌入，表格嵌入已经通过每个Transformer块中的交叉注意力层与图像特征交互
        # 不需要再进行额外的跨模态交互
        
        # 如果不需要计算损失，可以直接返回
        loss = torch.tensor(0.0, device=original_feat.device if not isinstance(original_feat, tuple) else numerical_features.device)
        
        if compute_loss:
            # 重构表格特征
            restored_feat = self.reconstruction_layer(table_embedding)
            
            # 计算损失
            if mask is not None and mask.sum() > 0:
                # 确保我们只对数值特征计算损失
                target_feat = numerical_features if isinstance(original_feat, tuple) else original_feat
                loss = F.smooth_l1_loss(restored_feat[mask], target_feat[mask])
        
        # 返回处理后的表格嵌入和注意力权重，用于后续的多模态融合和可视化
        return loss, table_embedding, all_attn_probs