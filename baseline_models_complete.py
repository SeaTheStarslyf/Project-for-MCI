#!/usr/bin/env python3
"""
基线单语音模态模型实现

本脚本实现了四种基线模型：
1. CNN (卷积神经网络)
2. 随机森林
3. XGBoost
4. Transformer

使用方法：
1. 确保已安装所需依赖：
   pip install numpy torch scikit-learn xgboost

2. 运行脚本：
   python baseline_models_complete.py

注意：本脚本使用dataset_RMI.py中的get_dataloader函数加载数据
"""

import os
import sys
import random
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.utils import class_weight

# 尝试导入数据加载器
try:
    from dataset.dataset_RMI import get_dataloader, get_kfold_dataloaders
except ImportError:
    print("Error: dataset_RMI.py not found. Please ensure the file exists in the dataset directory.")
    sys.exit(1)

# 设置随机种子
def set_random_seed(seed_value):
    """设置随机种子以确保结果可复现"""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    os.environ['PYTHONASHSEED'] = str(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_random_seed(42)

# 设备设置
def setup_gpu(gpu_id='0'):
    """设置并返回设备"""
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return torch.device('cuda')
    else:
        print("Using CPU")
        return torch.device('cpu')

device = setup_gpu()

# === 与 train.py 保持一致的 FocalLoss 与类别权重计算 ===
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = alpha
        if alpha is not None:
            if isinstance(alpha, list):
                self.alpha = torch.FloatTensor(alpha)
            elif isinstance(alpha, torch.Tensor):
                self.alpha = alpha

    def forward(self, inputs, targets):
        batch_size = inputs.size(0)
        prob = F.softmax(inputs, dim=1)
        pt = prob[torch.arange(batch_size, device=inputs.device), targets]
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
        else:
            alpha_t = 1.0

        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        if self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


def calculate_class_weights_from_labels(labels: np.ndarray) -> torch.FloatTensor:
    """按 train.py 口径计算 balanced class weights (alpha)。"""
    labels = np.asarray(labels)
    if labels.size == 0:
        return torch.FloatTensor([1.0])
    classes = np.unique(labels)
    weights = class_weight.compute_class_weight('balanced', classes=classes, y=labels)
    # 确保按 class index 对齐（0..C-1）
    max_c = int(classes.max())
    full = np.ones((max_c + 1,), dtype=np.float32)
    for c, w in zip(classes, weights):
        full[int(c)] = float(w)
    return torch.FloatTensor(full)


def calculate_class_weights_from_dataloader(trainloader) -> torch.FloatTensor:
    labels = []
    for batch in trainloader:
        # dataset_RMI 返回: image, features, cog_features, label, id
        label = batch[3]
        labels.extend(label.cpu().numpy().tolist())
    return calculate_class_weights_from_labels(np.array(labels, dtype=np.int64))

# 数据配置
config = {
    'batch_size_train': 32,
    'batch_size_val': 32,
    'num_workers': 2,
    'train_datalist': ['RMT_MRI-train'],
    'test_datalist': ['RMT_MRI-test'],
    'val_split_ratio': 0.2,
    'enable_cross_validation': True,
    'cv_folds': 5,
    'cv_datalist': ['RMT_MRI-train', 'RMT_MRI-test'],
}

# 加载数据
def load_data():
    """加载训练、验证和测试数据"""
    print("Loading training and validation data...")
    trainloader, valloader = get_dataloader(
        datalist=config['train_datalist'],
        batch_size=config['batch_size_train'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True,
        val_split_ratio=config['val_split_ratio'],
        random_state=42
    )

    print("Loading test data...")
    testloader = get_dataloader(
        datalist=config['test_datalist'],
        batch_size=config['batch_size_val'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False
    )
    return trainloader, valloader, testloader

# 提取特征和标签用于 sklearn 模型
def extract_features_labels(dataloader):
    """从数据加载器中提取特征和标签"""
    features = []
    labels = []
    for batch in dataloader:
        _, feat, _, label, _ = batch
        features.extend(feat.numpy())
        labels.extend(label.numpy())
    return np.array(features), np.array(labels)

# 评估函数
def evaluate_model(model, features, labels, model_name):
    """评估模型性能"""
    if isinstance(model, (RandomForestClassifier, XGBClassifier)):
        predictions = model.predict(features)
        # 获取概率预测
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(features)[:, 1]  # 二分类问题，取正类概率
        else:
            probs = model.predict(features)
    else:
        model.eval()
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).to(device)
            outputs = model(features_tensor)
            _, predictions = torch.max(outputs, 1)
            predictions = predictions.cpu().numpy()
            # 获取概率预测
            probs = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()  # 二分类问题，取正类概率
    
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='macro', zero_division=0)
    recall = recall_score(labels, predictions, average='macro', zero_division=0)
    f1 = f1_score(labels, predictions, average='macro', zero_division=0)
    cm = confusion_matrix(labels, predictions)
    
    # 按train.py(eval_metrics)一致口径计算AUC：
    # - 二分类：使用正类概率
    # - 多分类：使用OvR macro
    try:
        probs = np.asarray(probs)
        unique_classes = np.unique(labels)
        num_classes = len(unique_classes)

        if probs.ndim == 1:
            auc = roc_auc_score(labels, probs)
        elif probs.ndim == 2 and probs.shape[1] == 2:
            auc = roc_auc_score(labels, probs[:, 1])
        elif probs.ndim == 2 and probs.shape[1] > 2:
            auc = roc_auc_score(labels, probs, multi_class='ovr', average='macro')
        else:
            # 与eval_metrics保持一致：无有效概率形状时退化为离散预测
            if num_classes <= 2:
                auc = roc_auc_score(labels, predictions)
            else:
                all_classes = np.unique(np.concatenate([labels, predictions]))
                class_to_idx = {c: i for i, c in enumerate(all_classes)}
                y_true_mapped = np.array([class_to_idx[c] for c in labels], dtype=np.int64)
                y_pred_mapped = np.array([class_to_idx[c] for c in predictions], dtype=np.int64)
                y_pred_one_hot = np.eye(len(all_classes), dtype=np.float32)[y_pred_mapped]
                auc = roc_auc_score(y_true_mapped, y_pred_one_hot, multi_class='ovr', average='macro')
    except Exception:
        auc = 0.0
    
    print(f"\n{model_name} Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    
    return accuracy, precision, recall, f1, auc

def _auc_from_probs(labels: np.ndarray, probs: np.ndarray, predictions: Optional[np.ndarray] = None) -> float:
    """
    与 train.py(eval_metrics)一致口径计算AUC：
    - 二分类：使用正类概率
    - 多分类：使用OvR macro
    当probs形状不可用时，退化到离散预测。
    """
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    try:
        if probs.ndim == 1:
            return float(roc_auc_score(labels, probs))
        if probs.ndim == 2 and probs.shape[1] == 2:
            return float(roc_auc_score(labels, probs[:, 1]))
        if probs.ndim == 2 and probs.shape[1] > 2:
            return float(roc_auc_score(labels, probs, multi_class='ovr', average='macro'))
    except Exception:
        pass

    # fallback: use discrete predictions
    if predictions is None:
        return 0.0
    try:
        unique_classes = np.unique(labels)
        num_classes = len(unique_classes)
        if num_classes <= 2:
            return float(roc_auc_score(labels, predictions))
        all_classes = np.unique(np.concatenate([labels, predictions]))
        class_to_idx = {c: i for i, c in enumerate(all_classes)}
        y_true_mapped = np.array([class_to_idx[c] for c in labels], dtype=np.int64)
        y_pred_mapped = np.array([class_to_idx[c] for c in predictions], dtype=np.int64)
        y_pred_one_hot = np.eye(len(all_classes), dtype=np.float32)[y_pred_mapped]
        return float(roc_auc_score(y_true_mapped, y_pred_one_hot, multi_class='ovr', average='macro'))
    except Exception:
        return 0.0


def _evaluate_torch_epoch(model: nn.Module, dataloader, criterion: nn.Module):
    """返回 (val_loss, val_acc, val_auc)；AUC口径与 train.py 对齐。"""
    model.eval()
    total_loss = 0.0
    total_n = 0
    correct = 0
    all_probs = []
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            _, features, _, labels, _ = batch
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            n = labels.size(0)
            total_loss += float(loss.item()) * n
            total_n += n

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()

            probs = F.softmax(outputs, dim=1).detach().cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.detach().cpu().numpy())
            all_preds.append(preds.detach().cpu().numpy())

    if total_n == 0:
        return 0.0, 0.0, 0.0

    val_loss = total_loss / total_n
    val_acc = correct / total_n
    labels_np = np.concatenate(all_labels, axis=0) if all_labels else np.array([], dtype=np.int64)
    probs_np = np.concatenate(all_probs, axis=0) if all_probs else np.array([], dtype=np.float32)
    preds_np = np.concatenate(all_preds, axis=0) if all_preds else np.array([], dtype=np.int64)
    val_auc = _auc_from_probs(labels_np, probs_np, predictions=preds_np)
    return val_loss, val_acc, val_auc


def _metrics_dict_from_eval_tuple(eval_tuple):
    return {
        'acc': eval_tuple[0],
        'precision': eval_tuple[1],
        'recall': eval_tuple[2],
        'f1': eval_tuple[3],
        'auc': eval_tuple[4],
    }


def write_cv_result_table(model_results_by_split, output_path):
    """导出CSV：包含每折 val/test 与汇总 mean/std。"""
    metric_keys = ['acc', 'precision', 'recall', 'f1', 'auc']
    with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'fold', 'split', 'stat', 'acc', 'precision', 'recall', 'f1', 'auc'])

        for model_name, split_data in model_results_by_split.items():
            for split in ['val', 'test']:
                metrics_list = split_data[split]
                for i, metrics in enumerate(metrics_list, start=1):
                    writer.writerow([
                        model_name,
                        i,
                        split,
                        'raw',
                        metrics['acc'],
                        metrics['precision'],
                        metrics['recall'],
                        metrics['f1'],
                        metrics['auc'],
                    ])

                if len(metrics_list) > 0:
                    mean_row = [float(np.mean([m[k] for m in metrics_list])) for k in metric_keys]
                    std_row = [float(np.std([m[k] for m in metrics_list])) for k in metric_keys]
                    writer.writerow([model_name, 'summary', split, 'mean', *mean_row])
                    writer.writerow([model_name, 'summary', split, 'std', *std_row])

    print(f"交叉验证结果表格已保存: {output_path}")

# 1. 随机森林模型
def train_random_forest(train_features, train_labels, val_features, val_labels, test_features, test_labels):
    """训练随机森林模型"""
    print("\n" + "="*50)
    print("        Training Random Forest Model        ")
    print("="*50)
    
    # sklearn 基线无法使用 focal loss，这里用 class_weight 对齐 train.py 的“类别重加权”口径
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    model.fit(train_features, train_labels)
    
    evaluate_model(model, val_features, val_labels, "Random Forest (Validation)")
    evaluate_model(model, test_features, test_labels, "Random Forest (Test)")
    
    return model

# 2. XGBoost模型
def train_xgboost(train_features, train_labels, val_features, val_labels, test_features, test_labels):
    """训练XGBoost模型"""
    print("\n" + "="*50)
    print("        Training XGBoost Model        ")
    print("="*50)
    
    # sklearn 基线无法使用 focal loss，这里用样本权重/scale_pos_weight 对齐 train.py 的“类别重加权”口径
    train_labels_np = np.asarray(train_labels)
    unique = np.unique(train_labels_np)
    sample_weight = None
    scale_pos_weight = None
    if unique.size >= 2 and set(unique.tolist()) <= {0, 1}:
        neg = float((train_labels_np == 0).sum())
        pos = float((train_labels_np == 1).sum())
        if pos > 0:
            scale_pos_weight = neg / pos
    else:
        cls_w = calculate_class_weights_from_labels(train_labels_np).numpy()
        sample_weight = cls_w[train_labels_np.astype(np.int64)]

    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight
    )
    
    model.fit(train_features, train_labels, sample_weight=sample_weight)
    
    evaluate_model(model, val_features, val_labels, "XGBoost (Validation)")
    evaluate_model(model, test_features, test_labels, "XGBoost (Test)")
    
    return model

# 3. MLP模型（改进版本，适合处理扁平特征）
class MLPModel(nn.Module):
    """多层感知器模型，适合处理扁平特征"""
    def __init__(self, input_dim, num_classes=2, hidden_dims=[256, 128, 64]):
        super(MLPModel, self).__init__()
        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

def train_mlp(trainloader, valloader, test_features, test_labels, save_path='checkpoints/mlp_best.pth'):
    """训练MLP模型"""
    print("\n" + "="*50)
    print("        Training MLP Model        ")
    print("="*50)
    
    # 获取输入维度
    for batch in trainloader:
        _, features, _, _, _ = batch
        input_dim = features.shape[1]
        break
    
    # 与 train.py 一致：FocalLoss(alpha=balanced class weights, gamma=2.0)
    class_weights = calculate_class_weights_from_dataloader(trainloader).to(device)
    num_classes = int(class_weights.numel())
    model = MLPModel(input_dim, num_classes=num_classes).to(device)

    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 50
    patience = 50
    min_delta_auc = 0.0
    min_delta_loss = 0.0
    tie_breaker_eps = 1e-4

    best_f1 = -1.0
    best_val_loss_for_best_f1 = float("inf")
    best_val_loss_for_early_stop = float("inf")
    no_improve_epochs = 0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for batch in trainloader:
            _, features, _, labels, _ = batch
            features = features.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * features.size(0)
        
        epoch_loss = running_loss / len(trainloader.dataset)
        
        # 验证：按F1保存最佳模型，早停按val loss
        val_loss, val_acc, val_auc = _evaluate_torch_epoch(model, valloader, criterion)
        
        # 计算F1分数
        val_features_list = []
        val_labels_list = []
        for batch in valloader:
            _, features, _, labels, _ = batch
            val_features_list.append(features.numpy())
            val_labels_list.append(labels.numpy())
        val_features = np.concatenate(val_features_list, axis=0)
        val_labels = np.concatenate(val_labels_list, axis=0)
        
        model.eval()
        with torch.no_grad():
            features_tensor = torch.FloatTensor(val_features).to(device)
            outputs = model(features_tensor)
            _, predictions = torch.max(outputs, 1)
            predictions = predictions.cpu().numpy()
        
        val_f1 = f1_score(val_labels, predictions, average='macro', zero_division=0)
        
        print(
            f"Epoch {epoch+1}/{epochs}, "
            f"TrainLoss: {epoch_loss:.4f}, ValLoss: {val_loss:.4f}, ValACC: {val_acc:.4f}, ValF1: {val_f1:.4f}"
        )

        improved_f1 = val_f1 > (best_f1 + min_delta_auc)  # 使用相同的阈值
        f1_almost_same = abs(val_f1 - best_f1) <= tie_breaker_eps
        improved_tie = f1_almost_same and (val_loss < (best_val_loss_for_best_f1 - min_delta_loss))

        if improved_f1 or improved_tie:
            best_f1 = float(val_f1)
            best_val_loss_for_best_f1 = float(val_loss)
            os.makedirs(os.path.dirname(save_path) or 'checkpoints', exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"[MLP] best已保存: epoch={epoch+1}, F1={best_f1:.4f}, ValLoss={best_val_loss_for_best_f1:.4f}")

        if val_loss < (best_val_loss_for_early_stop - min_delta_loss):
            best_val_loss_for_early_stop = float(val_loss)
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            print(f"[MLP] Val loss 未下降 ({no_improve_epochs}/{patience})")

        if no_improve_epochs >= patience:
            print(f"[MLP] 早停触发: {patience}轮 val loss 未下降")
            break
    
    # 加载最佳模型
    model.load_state_dict(torch.load(save_path, map_location=device))
    
    # 评估模型 - 使用整个验证集
    val_features_list = []
    val_labels_list = []
    for batch in valloader:
        _, features, _, labels, _ = batch
        val_features_list.append(features.numpy())
        val_labels_list.append(labels.numpy())
    val_features = np.concatenate(val_features_list, axis=0)
    val_labels = np.concatenate(val_labels_list, axis=0)
    
    evaluate_model(model, val_features, val_labels, "MLP (Validation)")
    evaluate_model(model, test_features, test_labels, "MLP (Test)")
    
    return model

# 4. CNN模型
class CNNModel(nn.Module):
    """卷积神经网络模型（改进版，适合处理扁平特征）"""
    def __init__(self, input_dim, num_classes=2):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        # 计算特征维度，确保即使input_dim不是8的倍数也能正常工作
        self.feature_dim = 128 * ((input_dim + 1) // 8)
        self.fc1 = nn.Linear(self.feature_dim, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.3)
        self.batch_norm1 = nn.BatchNorm1d(32)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.batch_norm3 = nn.BatchNorm1d(128)
        self.batch_norm_fc = nn.BatchNorm1d(256)
    
    def forward(self, x):
        x = x.unsqueeze(1)  # 添加通道维度
        x = self.pool(F.relu(self.batch_norm1(self.conv1(x))))
        x = self.pool(F.relu(self.batch_norm2(self.conv2(x))))
        x = self.pool(F.relu(self.batch_norm3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.batch_norm_fc(self.fc1(x))))
        x = self.fc2(x)
        return x

def train_cnn(trainloader, valloader, test_features, test_labels, save_path='checkpoints/cnn_best.pth'):
    """训练CNN模型"""
    print("\n" + "="*50)
    print("        Training CNN Model        ")
    print("="*50)
    
    # 获取输入维度
    for batch in trainloader:
        _, features, _, _, _ = batch
        input_dim = features.shape[1]
        break
    
    # 与 train.py 一致：FocalLoss(alpha=balanced class weights, gamma=2.0)
    class_weights = calculate_class_weights_from_dataloader(trainloader).to(device)
    num_classes = int(class_weights.numel())
    model = CNNModel(input_dim, num_classes=num_classes).to(device)

    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 50
    patience = 50
    min_delta_auc = 0.0
    min_delta_loss = 0.0
    tie_breaker_eps = 1e-4

    best_f1 = -1.0
    best_val_loss_for_best_f1 = float("inf")
    best_val_loss_for_early_stop = float("inf")
    no_improve_epochs = 0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for batch in trainloader:
            _, features, _, labels, _ = batch
            features = features.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * features.size(0)
        
        epoch_loss = running_loss / len(trainloader.dataset)
        
        # 验证：按F1保存最佳模型，早停按val loss
        val_loss, val_acc, val_auc = _evaluate_torch_epoch(model, valloader, criterion)
        
        # 计算F1分数
        val_features_list = []
        val_labels_list = []
        for batch in valloader:
            _, features, _, labels, _ = batch
            val_features_list.append(features.numpy())
            val_labels_list.append(labels.numpy())
        val_features = np.concatenate(val_features_list, axis=0)
        val_labels = np.concatenate(val_labels_list, axis=0)
        
        model.eval()
        with torch.no_grad():
            features_tensor = torch.FloatTensor(val_features).to(device)
            outputs = model(features_tensor)
            _, predictions = torch.max(outputs, 1)
            predictions = predictions.cpu().numpy()
        
        val_f1 = f1_score(val_labels, predictions, average='macro', zero_division=0)
        
        print(
            f"Epoch {epoch+1}/{epochs}, "
            f"TrainLoss: {epoch_loss:.4f}, ValLoss: {val_loss:.4f}, ValACC: {val_acc:.4f}, ValF1: {val_f1:.4f}"
        )

        improved_f1 = val_f1 > (best_f1 + min_delta_auc)  # 使用相同的阈值
        f1_almost_same = abs(val_f1 - best_f1) <= tie_breaker_eps
        improved_tie = f1_almost_same and (val_loss < (best_val_loss_for_best_f1 - min_delta_loss))

        if improved_f1 or improved_tie:
            best_f1 = float(val_f1)
            best_val_loss_for_best_f1 = float(val_loss)
            os.makedirs(os.path.dirname(save_path) or 'checkpoints', exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"[CNN] best已保存: epoch={epoch+1}, F1={best_f1:.4f}, ValLoss={best_val_loss_for_best_f1:.4f}")

        if val_loss < (best_val_loss_for_early_stop - min_delta_loss):
            best_val_loss_for_early_stop = float(val_loss)
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            print(f"[CNN] Val loss 未下降 ({no_improve_epochs}/{patience})")

        if no_improve_epochs >= patience:
            print(f"[CNN] 早停触发: {patience}轮 val loss 未下降")
            break
    
    # 加载最佳模型
    model.load_state_dict(torch.load(save_path, map_location=device))
    
    # 评估模型 - 使用整个验证集
    val_features_list = []
    val_labels_list = []
    for batch in valloader:
        _, features, _, labels, _ = batch
        val_features_list.append(features.numpy())
        val_labels_list.append(labels.numpy())
    val_features = np.concatenate(val_features_list, axis=0)
    val_labels = np.concatenate(val_labels_list, axis=0)
    
    evaluate_model(model, val_features, val_labels, "CNN (Validation)")
    evaluate_model(model, test_features, test_labels, "CNN (Test)")
    
    return model

# 4. Transformer模型
class TransformerModel(nn.Module):
    """Transformer模型（改进版，适合处理扁平特征）"""
    def __init__(self, input_dim, num_classes=2, d_model=64, nhead=4, num_layers=1, dim_feedforward=128):
        super(TransformerModel, self).__init__()
        # 确保d_model能被nhead整除
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        
        self.embedding = nn.Linear(input_dim, d_model)
        self.batch_norm = nn.BatchNorm1d(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward,
            batch_first=True,  # 设置batch_first=True
            dropout=0.3
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        batch_size = x.size(0)
        # 输入形状: (batch_size, input_dim)
        x = self.embedding(x)  # 输出形状: (batch_size, d_model)
        x = self.batch_norm(x)
        x = self.dropout(x)
        
        # 不需要位置编码，因为特征是扁平的
        x = x.unsqueeze(1)  # 输出形状: (batch_size, 1, d_model)
        
        x = self.transformer_encoder(x)  # 输出形状: (batch_size, 1, d_model)
        x = x.squeeze(1)  # 输出形状: (batch_size, d_model)
        x = self.fc(x)  # 输出形状: (batch_size, num_classes)
        return x

def train_transformer(trainloader, valloader, test_features, test_labels, save_path='checkpoints/transformer_best.pth'):
    """训练Transformer模型"""
    print("\n" + "="*50)
    print("        Training Transformer Model        ")
    print("="*50)
    
    # 获取输入维度
    for batch in trainloader:
        _, features, _, _, _ = batch
        input_dim = features.shape[1]
        break
    
    # 与 train.py 一致：FocalLoss(alpha=balanced class weights, gamma=2.0)
    class_weights = calculate_class_weights_from_dataloader(trainloader).to(device)
    num_classes = int(class_weights.numel())
    model = TransformerModel(input_dim, num_classes=num_classes).to(device)

    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 50
    patience = 50
    min_delta_auc = 0.0
    min_delta_loss = 0.0
    tie_breaker_eps = 1e-4

    best_f1 = -1.0
    best_val_loss_for_best_f1 = float("inf")
    best_val_loss_for_early_stop = float("inf")
    no_improve_epochs = 0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for batch in trainloader:
            _, features, _, labels, _ = batch
            features = features.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * features.size(0)
        
        epoch_loss = running_loss / len(trainloader.dataset)
        
        # 验证：按F1保存最佳模型，早停按val loss
        val_loss, val_acc, val_auc = _evaluate_torch_epoch(model, valloader, criterion)
        
        # 计算F1分数
        val_features_list = []
        val_labels_list = []
        for batch in valloader:
            _, features, _, labels, _ = batch
            val_features_list.append(features.numpy())
            val_labels_list.append(labels.numpy())
        val_features = np.concatenate(val_features_list, axis=0)
        val_labels = np.concatenate(val_labels_list, axis=0)
        
        model.eval()
        with torch.no_grad():
            features_tensor = torch.FloatTensor(val_features).to(device)
            outputs = model(features_tensor)
            _, predictions = torch.max(outputs, 1)
            predictions = predictions.cpu().numpy()
        
        val_f1 = f1_score(val_labels, predictions, average='macro', zero_division=0)
        
        print(
            f"Epoch {epoch+1}/{epochs}, "
            f"TrainLoss: {epoch_loss:.4f}, ValLoss: {val_loss:.4f}, ValACC: {val_acc:.4f}, ValF1: {val_f1:.4f}"
        )

        improved_f1 = val_f1 > (best_f1 + min_delta_auc)  # 使用相同的阈值
        f1_almost_same = abs(val_f1 - best_f1) <= tie_breaker_eps
        improved_tie = f1_almost_same and (val_loss < (best_val_loss_for_best_f1 - min_delta_loss))

        if improved_f1 or improved_tie:
            best_f1 = float(val_f1)
            best_val_loss_for_best_f1 = float(val_loss)
            os.makedirs(os.path.dirname(save_path) or 'checkpoints', exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"[Transformer] best已保存: epoch={epoch+1}, F1={best_f1:.4f}, ValLoss={best_val_loss_for_best_f1:.4f}")

        if val_loss < (best_val_loss_for_early_stop - min_delta_loss):
            best_val_loss_for_early_stop = float(val_loss)
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            print(f"[Transformer] Val loss 未下降 ({no_improve_epochs}/{patience})")

        if no_improve_epochs >= patience:
            print(f"[Transformer] 早停触发: {patience}轮 val loss 未下降")
            break
    
    # 加载最佳模型
    model.load_state_dict(torch.load(save_path, map_location=device))
    
    # 评估模型 - 使用整个验证集
    val_features_list = []
    val_labels_list = []
    for batch in valloader:
        _, features, _, labels, _ = batch
        val_features_list.append(features.numpy())
        val_labels_list.append(labels.numpy())
    val_features = np.concatenate(val_features_list, axis=0)
    val_labels = np.concatenate(val_labels_list, axis=0)
    
    evaluate_model(model, val_features, val_labels, "Transformer (Validation)")
    evaluate_model(model, test_features, test_labels, "Transformer (Test)")
    
    return model

# 主函数
def summarize_cv_results(model_results):
    """汇总并打印交叉验证结果"""
    for model_name, metrics_list in model_results.items():
        if len(metrics_list) == 0:
            print(f"\n{model_name}: 无有效结果")
            continue

        accs = [m['acc'] for m in metrics_list]
        precisions = [m['precision'] for m in metrics_list]
        recalls = [m['recall'] for m in metrics_list]
        f1s = [m['f1'] for m in metrics_list]
        aucs = [m['auc'] for m in metrics_list]

        print("\n" + "=" * 60)
        print(f"{model_name} 交叉验证结果")
        print("=" * 60)
        for i, metric in enumerate(metrics_list, start=1):
            print(
                f"Fold {i}: "
                f"ACC={metric['acc']:.4f}, Precision={metric['precision']:.4f}, "
                f"Recall={metric['recall']:.4f}, F1={metric['f1']:.4f}, AUC={metric['auc']:.4f}"
            )
        print(f"Mean ACC: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
        print(f"Mean Precision: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
        print(f"Mean Recall: {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
        print(f"Mean F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
        print(f"Mean AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")


def run_single_split(trainloader, valloader, testloader, fold_idx=None):
    """执行单次训练/验证/测试流程并返回各模型验证集+测试集结果"""
    print("Extracting features...")
    train_features, train_labels = extract_features_labels(trainloader)
    val_features, val_labels = extract_features_labels(valloader)
    test_features, test_labels = extract_features_labels(testloader)

    print(f"Training data shape: {train_features.shape}")
    print(f"Validation data shape: {val_features.shape}")
    print(f"Test data shape: {test_features.shape}")

    rf_model = train_random_forest(train_features, train_labels, val_features, val_labels, test_features, test_labels)
    xgb_model = train_xgboost(train_features, train_labels, val_features, val_labels, test_features, test_labels)

    fold_suffix = f"_fold{fold_idx}" if fold_idx is not None else ""
    mlp_save_path = os.path.join('checkpoints', f'mlp_best{fold_suffix}.pth')
    cnn_save_path = os.path.join('checkpoints', f'cnn_best{fold_suffix}.pth')
    transformer_save_path = os.path.join('checkpoints', f'transformer_best{fold_suffix}.pth')
    mlp_model = train_mlp(trainloader, valloader, test_features, test_labels, save_path=mlp_save_path)
    cnn_model = train_cnn(trainloader, valloader, test_features, test_labels, save_path=cnn_save_path)
    transformer_model = train_transformer(trainloader, valloader, test_features, test_labels, save_path=transformer_save_path)

    rf_val_metrics = evaluate_model(rf_model, val_features, val_labels, "Random Forest (Validation Summary)")
    rf_test_metrics = evaluate_model(rf_model, test_features, test_labels, "Random Forest (Test Summary)")
    xgb_val_metrics = evaluate_model(xgb_model, val_features, val_labels, "XGBoost (Validation Summary)")
    xgb_test_metrics = evaluate_model(xgb_model, test_features, test_labels, "XGBoost (Test Summary)")
    mlp_val_metrics = evaluate_model(mlp_model, val_features, val_labels, "MLP (Validation Summary)")
    mlp_test_metrics = evaluate_model(mlp_model, test_features, test_labels, "MLP (Test Summary)")
    cnn_val_metrics = evaluate_model(cnn_model, val_features, val_labels, "CNN (Validation Summary)")
    cnn_test_metrics = evaluate_model(cnn_model, test_features, test_labels, "CNN (Test Summary)")
    transformer_val_metrics = evaluate_model(transformer_model, val_features, val_labels, "Transformer (Validation Summary)")
    transformer_test_metrics = evaluate_model(transformer_model, test_features, test_labels, "Transformer (Test Summary)")

    return {
        'Random Forest': {
            'val': _metrics_dict_from_eval_tuple(rf_val_metrics),
            'test': _metrics_dict_from_eval_tuple(rf_test_metrics),
        },
        'XGBoost': {
            'val': _metrics_dict_from_eval_tuple(xgb_val_metrics),
            'test': _metrics_dict_from_eval_tuple(xgb_test_metrics),
        },
        'MLP': {
            'val': _metrics_dict_from_eval_tuple(mlp_val_metrics),
            'test': _metrics_dict_from_eval_tuple(mlp_test_metrics),
        },
        'CNN': {
            'val': _metrics_dict_from_eval_tuple(cnn_val_metrics),
            'test': _metrics_dict_from_eval_tuple(cnn_test_metrics),
        },
        'Transformer': {
            'val': _metrics_dict_from_eval_tuple(transformer_val_metrics),
            'test': _metrics_dict_from_eval_tuple(transformer_test_metrics),
        }
    }


def main():
    """主函数（支持普通训练与K折交叉验证）"""
    print("Starting baseline model training...")
    os.makedirs('checkpoints', exist_ok=True)

    if config.get('enable_cross_validation', False):
        print("\n" + "=" * 60)
        print(f"开始 {config['cv_folds']} 折交叉验证")
        print("=" * 60)

        fold_loaders = get_kfold_dataloaders(
            datalist=config['cv_datalist'],
            n_splits=config['cv_folds'],
            val_split_ratio=config['val_split_ratio'],
            batch_size_train=config['batch_size_train'],
            batch_size_eval=config['batch_size_val'],
            num_workers=config['num_workers'],
            random_state=42,
        )

        model_results = {
            'Random Forest': {'val': [], 'test': []},
            'XGBoost': {'val': [], 'test': []},
            'MLP': {'val': [], 'test': []},
            'CNN': {'val': [], 'test': []},
            'Transformer': {'val': [], 'test': []},
        }

        for fold_idx, (trainloader, valloader, testloader) in enumerate(fold_loaders, start=1):
            print("\n" + "-" * 60)
            print(f"Fold {fold_idx}/{config['cv_folds']} 开始")
            print("-" * 60)

            fold_metrics = run_single_split(trainloader, valloader, testloader, fold_idx=fold_idx)
            for model_name, metrics in fold_metrics.items():
                model_results[model_name]['val'].append(metrics['val'])
                model_results[model_name]['test'].append(metrics['test'])

            torch.cuda.empty_cache()

        summarize_cv_results({k: v['test'] for k, v in model_results.items()})
        cv_table_path = os.path.join('checkpoints', 'baseline_cv_results_table.csv')
        write_cv_result_table(model_results, cv_table_path)
    else:
        trainloader, valloader, testloader = load_data()
        run_single_split(trainloader, valloader, testloader)

    print("\n" + "=" * 50)
    print("        All Baseline Models Trained        ")
    print("=" * 50)

if __name__ == "__main__":
    main()
