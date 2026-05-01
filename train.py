import os
import sys

# 先设置CUDA_VISIBLE_DEVICES，再导入torch
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import random
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils import class_weight
from dataset.dataset_RMI import get_dataloader, get_kfold_dataloaders
from medblip.modeling_medblip import MedBLIPModel, load_pretrained_vision_encoder
from medblip.trainer import Trainer
from medblip.utils import TrainingVisualizer

# Focal Loss 实现
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Focal Loss for handling class imbalance
        Args:
            alpha: 类别权重，None时自动计算
            gamma: 焦点参数，控制难易样本的权重
            reduction: 损失聚合方式
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = alpha
        if alpha is not None:
            if isinstance(alpha, list):
                self.alpha = torch.FloatTensor(alpha)
            elif isinstance(alpha, torch.Tensor):
                self.alpha = alpha
    
    def forward(self, inputs, targets):
        # 获取批次大小
        batch_size = inputs.size(0)
        
        # 计算softmax概率
        prob = F.softmax(inputs, dim=1)
        
        # 获取目标类别的概率
        pt = prob[torch.arange(batch_size), targets]
        
        # 计算交叉熵损失（不包含权重）
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # 处理类别权重
        if self.alpha is not None:
            # 确保alpha在正确的设备上
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            
            # 获取目标类别的权重
            alpha_t = self.alpha[targets]
        else:
            alpha_t = 1.0
        
        # 计算focal loss（符合原始论文公式）
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        
        # 聚合损失
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# GPU设备设置与验证
def setup_gpu(gpu_id='1'):
    """设置并验证GPU设备是否可用"""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if device_count == 0:
            print(f"警告: GPU ID {gpu_id} 不可用，将使用CPU")
            return torch.device('cpu')
        else:
            print(f"使用GPU ID {gpu_id}, 可用GPU数量: {device_count}")
            return torch.device('cuda')
    else:
        print("警告: CUDA不可用，将使用CPU")
        return torch.device('cpu')

# 添加类别平衡权重计算函数
def calculate_class_weights(trainloader):
    """计算训练集中各类别的权重，用于平衡类别不平衡问题"""
    labels = []
    try:
        # 遍历训练集收集所有标签
        for batch in trainloader:
            # 适应新的数据格式，包含认知量表特征
            if len(batch) == 5:
                # 新格式：image, features, cog_features, label, id
                _, _, _, label, _ = batch
            else:
                # 旧格式：image, text, label, id
                _, _, label, _ = batch
            labels.extend(label.cpu().numpy())
        
        # 检查是否收集到足够的标签
        if len(labels) == 0:
            print("警告: 无法收集到训练标签，使用默认权重")
            return torch.FloatTensor([1.0])
        
        # 计算类别权重
        class_weights = class_weight.compute_class_weight(
            'balanced', 
            classes=np.unique(labels), 
            y=labels
        )
        
        # 转换为tensor并移至设备
        class_weights_tensor = torch.FloatTensor(class_weights)
        print(f"类别权重: {class_weights_tensor}")
        return class_weights_tensor
    except Exception as e:
        print(f"计算类别权重时出错: {e}，使用默认权重")
        return torch.FloatTensor([1.0])

# 设备设置
device = setup_gpu(gpu_id='1')  # 默认使用第一个GPU
print(f"Using device: {device}")

# set random seed
seed = 42
def set_random_seed(seed_value):
    """设置随机种子以确保结果可复现"""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # 多GPU情况
    os.environ['PYTHONASHSEED'] = str(seed_value)
    os.environ['TOKENIZERS_PARALLELISM']='false'
    # 确保确定性操作
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_random_seed(seed)

# 数据集配置
config = {
    'txt_len': 209,  # 语音pca维度
    'cog_len': 80,  # 认知量表特征数量（将在运行时动态更新）
    'batch_size_train': 4,
    'batch_size_val': 24,
    'num_workers': 2,
    'train_datalist': ['RMT_MRI-train'],
    'test_datalist': ['RMT_MRI-test'],  # 测试集
    'val_split_ratio': 0.2,  # 从训练数据中划分验证集的比例
    'checkpoint_dir': 'checkpoints',
    'scaler_path': os.path.join('checkpoints', 'scalers.pkl'),  # 保存训练集标准化器，供测试阶段复用
    'teacher_model_name': 'teacher_model',
    'student_model_name': 'student_distilled',
    'training_phase': 1,  # 1=教师模型, 2=学生模型, 3=两步都执行
    'teacher_epochs': 50,
    'student_epochs': 50,
    'teacher_learning_rate': 1e-5,
    'student_learning_rate': 1e-5,
    'kd_weight': 5, # 蒸馏损失权重，控制软目标在总损失中的贡献度
    'student_cls_weight': 1, # 学生模型分类损失权重，控制硬标签在总损失中的贡献度
    'temperature': 2.0,
    'feat_distill_weight': 0.01, # 特征级蒸馏损失权重，控制特征对齐在总损失中的贡献度
    'modalities': 'speech_cog',  # 训练使用的模态: 'speech_cog'(多模态：语音+认知量表+MRI数值特征), 'text'(仅语音), 'cog'(仅认知量表)
    'single_modality_model_name': 'single_modality_model',  # 单模态模型保存名称
    'save_freq': 10,  # 每n个epoch保存一次模型
    'patience': 10,  # 提前停止的耐心值
    'use_pretrained_vision_encoder': False,  # 是否加载预训练视觉编码器
    'use_position_embedding': True,  # 是否使用位置嵌入
    'use_cross_attention': False,      # 是否使用跨模态注意力
    'balance_data': False,  # 是否在数据加载时进行类别平衡
    'enable_cross_validation': True,  # 是否启用K折交叉验证
    'cv_folds': 5,  # 交叉验证折数
    'cv_datalist': ['RMT_MRI-train', 'RMT_MRI-test'],  # 交叉验证时合并后再划分
}

# 创建数据加载器（普通模式下立即创建；交叉验证模式下在每折动态创建）
trainloader, valloader, testloader = None, None, None
if not config['enable_cross_validation']:
    try:
        print("加载训练数据集并划分验证集...")
        trainloader, valloader = get_dataloader(
            datalist=config['train_datalist'],
            batch_size=config['batch_size_train'],
            txt_len=config['txt_len'],
            shuffle=True,
            num_workers=config['num_workers'],
            drop_last=True,
            val_split_ratio=config['val_split_ratio'],
            random_state=seed,
            balance_data=config['balance_data'],
            scaler_save_path=config['scaler_path'],
        )

        print("加载测试数据集...")
        testloader = get_dataloader(
            datalist=config['test_datalist'],
            batch_size=config['batch_size_val'],
            txt_len=config['txt_len'],
            shuffle=False,
            num_workers=config['num_workers'],
            drop_last=False
        )

        # 动态计算认知量表特征的维度
        print("计算认知量表特征维度...")
        # 从训练数据加载器中获取一个批次的数据，计算认知量表特征的维度
        for batch in trainloader:
            if len(batch) == 5:
                # 新格式：image, features, cog_features, label, id
                _, _, cog_features, _, _ = batch
                config['cog_len'] = cog_features.shape[1]
                print(f"动态更新认知量表特征维度: {config['cog_len']}")
                break

        print(f"成功创建数据加载器，训练集批次: {len(trainloader)}, 验证集批次: {len(valloader)}, 测试集批次: {len(testloader)}")
    except Exception as e:
        print(f"创建数据加载器失败: {e}")
        sys.exit(1)

# 检查点保存路径
os.makedirs(config['checkpoint_dir'], exist_ok=True)
TEACHER_MODEL_PATH = os.path.join(config['checkpoint_dir'], f'{config["teacher_model_name"]}_best.pth')

def train_teacher_model():
    """训练多模态教师模型"""
    print("\n" + "="*50)
    print("          开始训练多模态教师模型          ")
    print("="*50)
    
    try:
        # 计算类别权重
        print("计算训练集类别权重...")
        class_weights = calculate_class_weights(trainloader)
        
        # 创建模型 - 完全专注于教师模型训练，不使用蒸馏或学生模型
        model = MedBLIPModel(max_txt_len=config['txt_len'], 
                            num_numerical_features=config['txt_len'],
                            num_cog_features=config['cog_len'],  # 认知量表特征数量
                            freeze_teacher=False,  # 不冻结教师模型
                            kd_temp=1.0,          # 蒸馏温度
                            kd_weight=0.0,         # 蒸馏权重为0
                            student_cls_weight=0.0,
                            use_positional_embedding=config['use_position_embedding'],
                            use_cross_attention=config['use_cross_attention'],
                            modalities=config['modalities'],  # 添加模态参数
                            training_mode='teacher')  # 禁用学生模型
        
        # 加载预训练权重到视觉模态模型
        if config['use_pretrained_vision_encoder']:
            pretrained_path = 'pretrained/latest.pth'
            if config['modalities'] == 'both' or config['modalities'] == 'vision':
                print(f"加载视觉预训练权重: {pretrained_path}")
                # 使用分层冻结策略，冻结前6层视觉Transformer
                model = load_pretrained_vision_encoder(model, pretrained_path, freeze_layers=0)
        
        # 设置Focal Loss
        model.criterion = FocalLoss(alpha=class_weights.to(device), gamma=2.0)
        
        model.to(device)
        # 为教师模型训练创建专用的可视化器，指定不同的保存目录
        trainer = Trainer(phase_name="teacher", training_mode=config['training_phase'])
        # 修改默认的可视化目录，避免被第二阶段覆盖
        trainer.visualizer = TrainingVisualizer(save_dir="training_plots_teacher", phase_name="teacher")
        
        # 打印训练配置
        print("\n===== 教师模型训练配置 =====")
        print(f"冻结教师模型: {model.freeze_teacher}")
        print(f"使用蒸馏: False (教师模型模式)")
        print(f"训练轮数: {config['teacher_epochs']}")
        print(f"学习率: {config['teacher_learning_rate']}")
        print(f"权重衰减: 1e-4")
        print(f"批次大小: {trainloader.batch_size}")
        print(f"类别权重: {class_weights}")
        print("训练目标: 多模态特征融合 + 分类任务")
        print("=====================\n")
        
        # 训练教师模型 - 只优化教师模型相关参数
        trainer.train(
            model,
            trainloader,
            valloader,
            warmup_ratio=0.1,
            epochs=config['teacher_epochs'],
            optimizer_params={'lr': config['teacher_learning_rate']},
            output_path=os.path.join(config['checkpoint_dir'], config['teacher_model_name']),
            metric_path=os.path.join(config['checkpoint_dir'], f'{config["teacher_model_name"]}_metrics.txt'),
            weight_decay=1e-4,
            save_interval=config['save_freq'],
            patience=config['patience'],
        )
        
        # 加载训练过程中保存的最佳模型，然后保存到最终路径
        best_model_path = os.path.join(config['checkpoint_dir'], config['teacher_model_name'], 'best_model.pth')
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            torch.save(model.state_dict(), TEACHER_MODEL_PATH)
            print(f"\n✅ 已加载并保存最佳教师模型到: {TEACHER_MODEL_PATH}")
        else:
            # 如果没有找到最佳模型文件，使用当前模型
            torch.save(model.state_dict(), TEACHER_MODEL_PATH)
            print(f"\n⚠️  未找到最佳模型文件，保存当前模型到: {TEACHER_MODEL_PATH}")
        print("教师模型训练阶段完成！")
        
        return model
    except Exception as e:
        print(f"训练教师模型时出错: {e}")
        # 尝试保存当前状态以便恢复
        try:
            if 'model' in locals():
                error_model_path = os.path.join(config['checkpoint_dir'], f'{config["teacher_model_name"]}_error.pth')
                torch.save(model.state_dict(), error_model_path)
                print(f"已保存错误状态模型到: {error_model_path}")
        except:
            pass
        return None

def train_student_model(teacher_model=None):
    """固定教师模型，蒸馏训练单模态学生模型"""
    print("\n" + "="*50)
    print("         开始蒸馏训练单模态学生模型         ")
    print("="*50)
    
    try:
        # 检查教师模型是否可用
        if teacher_model is None and not os.path.exists(TEACHER_MODEL_PATH):
            print(f"错误: 找不到教师模型文件: {TEACHER_MODEL_PATH}")
            return None
        
        # 创建模型 - 启用蒸馏，冻结教师模型
        model = MedBLIPModel(max_txt_len=config['txt_len'], 
                            num_numerical_features=config['txt_len'],
                            num_cog_features=config['cog_len'],  # 认知量表特征数量
                            freeze_teacher=True,  # 强制冻结教师模型
                            kd_temp=config['temperature'],  # 蒸馏温度
                            kd_weight=config['kd_weight'],   # 蒸馏损失权重
                            student_cls_weight=config['student_cls_weight'],
                            feat_distill_weight=config['feat_distill_weight'],
                            use_positional_embedding=config['use_position_embedding'],
                            use_cross_attention=config['use_cross_attention'],
                            modalities=config['modalities'],  # 添加模态参数
                            training_mode='student') # 特征级蒸馏权重
        
        # 设置Focal Loss
        class_weights = calculate_class_weights(trainloader)
        model.criterion = FocalLoss(alpha=class_weights.to(device), gamma=2.0)
        
        # 加载训练好的教师模型权重
        teacher_loaded = False
        if teacher_model is not None:
            # 从内存加载教师模型权重，忽略缺少的键（如student_text_restorer）
            model.load_state_dict(teacher_model.state_dict(), strict=False)
            print("✅ 从内存加载教师模型权重")
            teacher_loaded = True
        elif os.path.exists(TEACHER_MODEL_PATH):
            # 从文件加载教师模型权重，忽略缺少的键（如student_text_restorer）
            try:
                model.load_state_dict(torch.load(TEACHER_MODEL_PATH, map_location=device), strict=False)
                print(f"✅ 从文件加载教师模型权重: {TEACHER_MODEL_PATH}")
                teacher_loaded = True
            except Exception as e:
                print(f"❌ 加载教师模型失败: {e}")
        else:
            print(f"❌ 未找到训练好的教师模型: {TEACHER_MODEL_PATH}")
        
        if not teacher_loaded:
            print("错误: 无法继续训练，没有有效的教师模型权重")
            return None
        
        # 强制冻结教师模型参数 - 确保所有教师相关参数不可训练
        model._freeze_teacher_parameters()
        # 重新设置蒸馏温度，确保使用配置中的值
        model.distillation_temp = nn.Parameter(torch.tensor(config['temperature']))
        model.to(device)
        # 为学生模型训练创建专用的可视化器，指定不同的保存目录
        trainer = Trainer(phase_name="student", training_mode=config['training_phase'])
        # 修改默认的可视化目录，避免与第一阶段冲突
        trainer.visualizer = TrainingVisualizer(save_dir="training_plots_student", phase_name="student")
        
        # 打印训练配置
        print("\n===== 学生模型蒸馏配置 =====")
        print(f"冻结教师模型: {model.freeze_teacher}")
        print(f"使用蒸馏: True (学生模型模式)")
        print(f"蒸馏权重: {model.kd_weight}")
        print(f"学生分类权重: {model.student_cls_weight}")
        print(f"蒸馏温度: {model.distillation_temp.item()}")
        print(f"训练轮数: {config['student_epochs']}")
        print(f"学习率: {config['student_learning_rate']}")
        print(f"类别权重: {class_weights}")
        print("训练目标: 语音单模态分类 + 教师模型软目标")
        print(f"可训练参数: student_ft_transformer (特征提取), mlp_speech (分类头)")
        print(f"训练方式: 响应式蒸馏模式 - 同时进行logits级和特征级对齐")
        print(f"特征级蒸馏权重: {config['feat_distill_weight']}")
        print("=====================\n")
        
        # 蒸馏训练学生模型
        trainer.train(
            model,
            trainloader,
            valloader,
            warmup_ratio=0.1,
            epochs=config['student_epochs'],
            optimizer_params={'lr': config['student_learning_rate']},
            output_path=os.path.join(config['checkpoint_dir'], config['student_model_name']),
            metric_path=os.path.join(config['checkpoint_dir'], f'{config["student_model_name"]}_metrics.txt'),
            weight_decay=1e-4,
            save_interval=config['save_freq'],
            patience=config['patience'],
        )
        
        # 加载训练过程中保存的最佳学生模型，然后保存到最终路径
        best_student_model_path = os.path.join(config['checkpoint_dir'], config['student_model_name'], 'best_model.pth')
        student_model_path = os.path.join(config['checkpoint_dir'], f'{config["student_model_name"]}_best.pth')
        if os.path.exists(best_student_model_path):
            model.load_state_dict(torch.load(best_student_model_path, map_location=device))
            torch.save(model.state_dict(), student_model_path)
            print(f"\n✅ 已加载并保存最佳学生模型到: {student_model_path}")
        else:
            # 如果没有找到最佳模型文件，使用当前模型
            torch.save(model.state_dict(), student_model_path)
            print(f"\n⚠️  未找到最佳学生模型文件，保存当前模型到: {student_model_path}")
        print("学生模型蒸馏训练完成！")
        
        return model
    except Exception as e:
        print(f"训练学生模型时出错: {e}")
        # 尝试保存当前状态以便恢复
        try:
            if 'model' in locals():
                error_model_path = os.path.join(config['checkpoint_dir'], f'{config["student_model_name"]}_error.pth')
                torch.save(model.state_dict(), error_model_path)
                print(f"已保存错误状态模型到: {error_model_path}")
        except:
            pass
        return None

def train_single_modality_model():
    """训练单模态模型（仅视觉、仅文本/语音或仅认知量表）"""
    print("\n" + "="*50)
    modality_name = '视觉' if config['modalities'] == 'vision' else '文本/语音' if config['modalities'] == 'text' else '认知量表' if config['modalities'] == 'cog' else '语音+认知量表'
    print(f"          开始训练{modality_name}单模态模型          ")
    print("="*50)
    
    try:
        # 计算类别权重
        print("计算训练集类别权重...")
        class_weights = calculate_class_weights(trainloader)
        
        # 创建模型 - 单模态训练模式
        model = MedBLIPModel(max_txt_len=config['txt_len'], 
                            num_numerical_features=config['txt_len'],
                            num_cog_features=config['cog_len'],  # 认知量表特征数量
                            freeze_teacher=False,  # 不冻结模型参数
                            kd_temp=1.0,          # 蒸馏温度（单模态时无意义）
                            kd_weight=0.0,         # 蒸馏权重为0
                            student_cls_weight=0.0,
                            modalities=config['modalities'],
                            use_positional_embedding=config['use_position_embedding'],
                            use_cross_attention=config['use_cross_attention'],
                            training_mode='single')  # 设置训练模态
        
        # 加载预训练权重到视觉模态模型
        if config['use_pretrained_vision_encoder']:
            pretrained_path = 'pretrained/latest.pth'
            if config['modalities'] == 'vision':
                print(f"加载视觉预训练权重: {pretrained_path}")
                # 使用分层冻结策略，冻结前6层视觉Transformer
                model = load_pretrained_vision_encoder(model, pretrained_path, freeze_layers=6)
        
        # 设置Focal Loss
        model.criterion = FocalLoss(alpha=class_weights.to(device), gamma=2.0)
        
        model.to(device)
        # 为单模态模型训练创建专用的可视化器
        trainer = Trainer(phase_name=f"single_{config['modalities']}", training_mode=config['training_phase'])
        # 创建专用的保存目录
        trainer.visualizer = TrainingVisualizer(save_dir=f"training_plots_single_{config['modalities']}", phase_name=f"single_{config['modalities']}")
        
        # 打印训练配置
        print("\n===== 单模态模型训练配置 =====")
        print(f"训练模态: {config['modalities']}")
        print(f"冻结教师模型: {model.freeze_teacher}")
        print(f"训练轮数: {config['teacher_epochs']}")
        print(f"学习率: {config['teacher_learning_rate']}")
        print(f"权重衰减: 1e-4")
        print(f"批次大小: {trainloader.batch_size}")
        print(f"类别权重: {class_weights}")
        modality_target = '视觉' if config['modalities'] == 'vision' else '文本/语音' if config['modalities'] == 'text' else '认知量表' if config['modalities'] == 'cog' else '语音+认知量表'
        print(f"训练目标: {modality_target}单模态分类")
        print("=====================\n")
        
        # 训练单模态模型
        trainer.train(
            model,
            trainloader,
            valloader,
            warmup_ratio=0.1,
            epochs=config['teacher_epochs'],
            optimizer_params={'lr': config['teacher_learning_rate']},
            output_path=os.path.join(config['checkpoint_dir'], f"{config['single_modality_model_name']}_{config['modalities']}"),
            metric_path=os.path.join(config['checkpoint_dir'], f"{config['single_modality_model_name']}_{config['modalities']}_metrics.txt"),
            weight_decay=1e-4,
            save_interval=config['save_freq'],
            patience=config['patience'],
        )
        
        # 加载训练过程中保存的最佳单模态模型，然后保存到最终路径
        best_single_model_path = os.path.join(config['checkpoint_dir'], f"{config['single_modality_model_name']}_{config['modalities']}", 'best_model.pth')
        single_modality_model_path = os.path.join(config['checkpoint_dir'], f"{config['single_modality_model_name']}_{config['modalities']}_best.pth")
        if os.path.exists(best_single_model_path):
            model.load_state_dict(torch.load(best_single_model_path, map_location=device))
            torch.save(model.state_dict(), single_modality_model_path)
            print(f"\n✅ 已加载并保存最佳单模态模型到: {single_modality_model_path}")
        else:
            # 如果没有找到最佳模型文件，使用当前模型
            torch.save(model.state_dict(), single_modality_model_path)
            print(f"\n⚠️  未找到最佳单模态模型文件，保存当前模型到: {single_modality_model_path}")
        print("单模态模型训练阶段完成！")
        
        return model
    except Exception as e:
        print(f"训练单模态模型时出错: {e}")
        # 尝试保存当前状态以便恢复
        try:
            if 'model' in locals():
                error_model_path = os.path.join(config['checkpoint_dir'], f"{config['single_modality_model_name']}_{config['modalities']}_error.pth")
                torch.save(model.state_dict(), error_model_path)
                print(f"已保存错误状态模型到: {error_model_path}")
        except:
            pass
        return None

def evaluate_on_loader(model, eval_loader, metric_path, phase_name='eval'):
    """在指定数据集上评估并返回用于汇总的AUC/ACC/F1/Precision/Recall"""
    if model is None or eval_loader is None:
        return None

    trainer = Trainer(phase_name=phase_name, training_mode=config['training_phase'])
    teacher_auc, teacher_acc, student_auc, student_acc, *rest_metrics = trainer.test(
        model=model,
        eval_dataloader=eval_loader,
        metric_path=metric_path,
        epoch=9999,
    )
    teacher_f1, teacher_precision, teacher_recall, student_f1, student_precision, student_recall = rest_metrics[-6:]

    # 根据训练模式选择最终汇总指标
    if config['modalities'] in ['vision', 'text', 'cog']:
        return {
            'auc': teacher_auc,
            'acc': teacher_acc,
            'f1': teacher_f1,
            'precision': teacher_precision,
            'recall': teacher_recall,
        }
    if config['training_phase'] == 1:
        return {
            'auc': teacher_auc,
            'acc': teacher_acc,
            'f1': teacher_f1,
            'precision': teacher_precision,
            'recall': teacher_recall,
        }
    return {
        'auc': student_auc,
        'acc': student_acc,
        'f1': student_f1,
        'precision': student_precision,
        'recall': student_recall,
    }


def evaluate_on_test(model, metric_path, phase_name='test'):
    """在测试集上评估并返回用于汇总的AUC/ACC/F1/Precision/Recall"""
    return evaluate_on_loader(model, testloader, metric_path=metric_path, phase_name=phase_name)


def write_cv_result_table(cv_rows, output_path):
    """将交叉验证逐折结果与汇总结果写入CSV表格"""
    if len(cv_rows) == 0:
        return

    metric_keys = ['auc', 'acc', 'f1', 'precision', 'recall']
    split_names = ['val', 'test']

    with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['fold', 'split', 'stat', 'auc', 'acc', 'f1', 'precision', 'recall'])

        for row in cv_rows:
            for split in split_names:
                metrics = row[split]
                writer.writerow([
                    row['fold'],
                    split,
                    'raw',
                    metrics['auc'],
                    metrics['acc'],
                    metrics['f1'],
                    metrics['precision'],
                    metrics['recall'],
                ])

        for split in split_names:
            values = {k: [row[split][k] for row in cv_rows] for k in metric_keys}
            mean_row = [float(np.mean(values[k])) for k in metric_keys]
            std_row = [float(np.std(values[k])) for k in metric_keys]
            writer.writerow(['summary', split, 'mean', *mean_row])
            writer.writerow(['summary', split, 'std', *std_row])

    print(f"交叉验证结果表格已保存: {output_path}")


def run_training_flow():
    """执行单次训练流程，返回训练得到的最终模型"""
    teacher_model = None
    student_model = None
    single_modality_model = None

    if config['modalities'] in ['vision', 'text', 'cog']:
        single_modality_model = train_single_modality_model()
        return single_modality_model

    if config['training_phase'] == 1 or config['training_phase'] == 3:
        teacher_model = train_teacher_model()
        if teacher_model is None:
            return None

    if config['training_phase'] == 2 or config['training_phase'] == 3:
        student_model = train_student_model(teacher_model)

    if config['training_phase'] == 1:
        return teacher_model
    return student_model if student_model is not None else teacher_model


def main():
    """主训练流程控制（支持普通训练与K折交叉验证）"""
    try:
        global trainloader, valloader, testloader, TEACHER_MODEL_PATH

        if config['enable_cross_validation']:
            print("\n" + "=" * 60)
            print(f"开始 {config['cv_folds']} 折交叉验证")
            print("=" * 60)

            fold_loaders = get_kfold_dataloaders(
                datalist=config['cv_datalist'],
                n_splits=config['cv_folds'],
                val_split_ratio=config['val_split_ratio'],
                batch_size_train=config['batch_size_train'],
                batch_size_eval=config['batch_size_val'],
                txt_len=config['txt_len'],
                num_workers=config['num_workers'],
                random_state=seed,
                balance_data=config['balance_data'],
            )

            fold_results = []
            original_teacher_name = config['teacher_model_name']
            original_student_name = config['student_model_name']
            original_single_name = config['single_modality_model_name']

            for fold_idx, (tr_loader, va_loader, te_loader) in enumerate(fold_loaders, start=1):
                print("\n" + "-" * 60)
                print(f"Fold {fold_idx}/{config['cv_folds']} 开始")
                print("-" * 60)

                trainloader, valloader, testloader = tr_loader, va_loader, te_loader

                # 动态计算认知量表特征的维度
                print("计算认知量表特征维度...")
                for batch in trainloader:
                    if len(batch) == 5:
                        # 新格式：image, features, cog_features, label, id
                        _, _, cog_features, _, _ = batch
                        config['cog_len'] = cog_features.shape[1]
                        print(f"动态更新认知量表特征维度: {config['cog_len']}")
                        break

                # 每折使用独立的模型名/路径，避免互相覆盖
                config['teacher_model_name'] = f"{original_teacher_name}_fold{fold_idx}"
                config['student_model_name'] = f"{original_student_name}_fold{fold_idx}"
                config['single_modality_model_name'] = f"{original_single_name}_fold{fold_idx}"
                TEACHER_MODEL_PATH = os.path.join(config['checkpoint_dir'], f'{config["teacher_model_name"]}_best.pth')

                model = run_training_flow()
                if model is None:
                    print(f"警告: Fold {fold_idx} 训练失败，跳过该折统计")
                    continue

                val_metric_path = os.path.join(
                    config['checkpoint_dir'],
                    f'cv_fold{fold_idx}_val_metrics.txt'
                )
                test_metric_path = os.path.join(
                    config['checkpoint_dir'],
                    f'cv_fold{fold_idx}_test_metrics.txt'
                )
                val_result = evaluate_on_loader(
                    model,
                    valloader,
                    metric_path=val_metric_path,
                    phase_name=f"cv_fold{fold_idx}_val"
                )
                test_result = evaluate_on_test(
                    model,
                    metric_path=test_metric_path,
                    phase_name=f"cv_fold{fold_idx}_test"
                )
                if val_result is None or test_result is None:
                    print(f"警告: Fold {fold_idx} 验证或测试评估失败")
                    continue

                fold_results.append({
                    'fold': fold_idx,
                    'val': val_result,
                    'test': test_result,
                })
                print(
                    f"Fold {fold_idx} 验证结果 -> "
                    f"AUC: {val_result['auc']:.4f}, ACC: {val_result['acc']:.4f}, "
                    f"F1: {val_result['f1']:.4f}, Precision: {val_result['precision']:.4f}, Recall: {val_result['recall']:.4f}"
                )
                print(
                    f"Fold {fold_idx} 测试结果 -> "
                    f"AUC: {test_result['auc']:.4f}, ACC: {test_result['acc']:.4f}, "
                    f"F1: {test_result['f1']:.4f}, Precision: {test_result['precision']:.4f}, Recall: {test_result['recall']:.4f}"
                )

                torch.cuda.empty_cache()

            # 恢复原始命名，避免影响后续其他流程
            config['teacher_model_name'] = original_teacher_name
            config['student_model_name'] = original_student_name
            config['single_modality_model_name'] = original_single_name
            TEACHER_MODEL_PATH = os.path.join(config['checkpoint_dir'], f'{config["teacher_model_name"]}_best.pth')

            if len(fold_results) == 0:
                print("交叉验证未获得有效结果")
                return

            mean_auc_val = float(np.mean([x['val']['auc'] for x in fold_results]))
            mean_acc_val = float(np.mean([x['val']['acc'] for x in fold_results]))
            mean_f1_val = float(np.mean([x['val']['f1'] for x in fold_results]))
            mean_precision_val = float(np.mean([x['val']['precision'] for x in fold_results]))
            mean_recall_val = float(np.mean([x['val']['recall'] for x in fold_results]))
            std_auc_val = float(np.std([x['val']['auc'] for x in fold_results]))
            std_acc_val = float(np.std([x['val']['acc'] for x in fold_results]))
            std_f1_val = float(np.std([x['val']['f1'] for x in fold_results]))
            std_precision_val = float(np.std([x['val']['precision'] for x in fold_results]))
            std_recall_val = float(np.std([x['val']['recall'] for x in fold_results]))

            mean_auc_test = float(np.mean([x['test']['auc'] for x in fold_results]))
            mean_acc_test = float(np.mean([x['test']['acc'] for x in fold_results]))
            mean_f1_test = float(np.mean([x['test']['f1'] for x in fold_results]))
            mean_precision_test = float(np.mean([x['test']['precision'] for x in fold_results]))
            mean_recall_test = float(np.mean([x['test']['recall'] for x in fold_results]))
            std_auc_test = float(np.std([x['test']['auc'] for x in fold_results]))
            std_acc_test = float(np.std([x['test']['acc'] for x in fold_results]))
            std_f1_test = float(np.std([x['test']['f1'] for x in fold_results]))
            std_precision_test = float(np.std([x['test']['precision'] for x in fold_results]))
            std_recall_test = float(np.std([x['test']['recall'] for x in fold_results]))

            print("\n" + "=" * 60)
            print("交叉验证完成（验证集 + 测试集结果）")
            print("=" * 60)
            for i, r in enumerate(fold_results, start=1):
                print(
                    f"Fold {i} Val: "
                    f"AUC={r['val']['auc']:.4f}, ACC={r['val']['acc']:.4f}, "
                    f"F1={r['val']['f1']:.4f}, Precision={r['val']['precision']:.4f}, Recall={r['val']['recall']:.4f}"
                )
                print(
                    f"Fold {i} Test: "
                    f"AUC={r['test']['auc']:.4f}, ACC={r['test']['acc']:.4f}, "
                    f"F1={r['test']['f1']:.4f}, Precision={r['test']['precision']:.4f}, Recall={r['test']['recall']:.4f}"
                )
            print(f"Val Mean AUC: {mean_auc_val:.4f} ± {std_auc_val:.4f}")
            print(f"Val Mean ACC: {mean_acc_val:.4f} ± {std_acc_val:.4f}")
            print(f"Val Mean F1: {mean_f1_val:.4f} ± {std_f1_val:.4f}")
            print(f"Val Mean Precision: {mean_precision_val:.4f} ± {std_precision_val:.4f}")
            print(f"Val Mean Recall: {mean_recall_val:.4f} ± {std_recall_val:.4f}")
            print(f"Test Mean AUC: {mean_auc_test:.4f} ± {std_auc_test:.4f}")
            print(f"Test Mean ACC: {mean_acc_test:.4f} ± {std_acc_test:.4f}")
            print(f"Test Mean F1: {mean_f1_test:.4f} ± {std_f1_test:.4f}")
            print(f"Test Mean Precision: {mean_precision_test:.4f} ± {std_precision_test:.4f}")
            print(f"Test Mean Recall: {mean_recall_test:.4f} ± {std_recall_test:.4f}")

            cv_table_path = os.path.join(config['checkpoint_dir'], 'cv_results_table.csv')
            write_cv_result_table(fold_results, cv_table_path)
            print("=" * 60)
        else:
            model = run_training_flow()
            if model is None:
                print("警告: 训练失败")
                return

            result = evaluate_on_test(
                model,
                metric_path=os.path.join(config['checkpoint_dir'], 'test_metrics.txt'),
                phase_name='test'
            )
            if result is not None:
                print(
                    f"\n最终测试结果 -> "
                    f"AUC: {result['auc']:.4f}, ACC: {result['acc']:.4f}, "
                    f"F1: {result['f1']:.4f}, Precision: {result['precision']:.4f}, Recall: {result['recall']:.4f}"
                )

        print("\n训练流程全部完成!")
        
        # 清理CUDA缓存
        torch.cuda.empty_cache()
        
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"训练过程中发生错误: {e}")
    finally:
        # 确保清理资源
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()