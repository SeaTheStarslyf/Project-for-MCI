import os
import sys
import glob
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from dataset.dataset_RMI import get_dataloader, get_kfold_dataloaders
from medblip.modeling_medblip import MedBLIPModel

# 设置设备
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 配置
config = {
    'txt_len': 209,  # 语音pca维度
    'cog_len': 80,  # 认知量表特征数量（会在运行时动态更新）
    'batch_size_val': 1,
    'num_workers': 2,
    'cv_datalist': ['RMT_MRI-train', 'RMT_MRI-test'],  # 交叉验证数据列表
    'cv_folds': 5,  # 交叉验证折数
    'val_split_ratio': 0.2,
    'checkpoint_dir': 'checkpoints',
    'teacher_model_name': 'teacher_model',
    'modalities': 'speech_cog',  # 训练使用的模态
    'use_ft_transformer': True,
    'use_position_embedding': True,
    'use_cross_attention': False,
    'enable_cross_validation': True,  # 是否启用交叉验证
}

# 获取特征名称
def get_feature_names():
    """从CSV文件中获取特征名称及来源"""
    # 语音特征名称
    filename = 'local_data/RMT_MRI-test.csv'
    df = pd.read_csv(filename, header=0)
    speech_feature_columns = [col for col in df.columns if df.columns.get_loc(col) >= df.columns.get_loc('SCREENER_SCORE')]

    # 认知量表特征名称
    filename_cog = 'local_data/information-test.csv'
    df_cog = pd.read_csv(filename_cog, header=0)
    cog_keys = list(df_cog.columns)
    trails_index = cog_keys.index('TOTSCORE') if 'TOTSCORE' in cog_keys else 0
    cog_feature_columns = cog_keys[trails_index:]
    
    # 标记认知量表特征的来源
    cog_feature_sources = ['cognitive' for _ in cog_feature_columns]

    # MRI信息特征名称
    filename_mri_info = 'local_data/MRI_information_matched_to_RMI_MRI_split.csv'
    df_mri_info = pd.read_csv(filename_mri_info, header=0)
    mri_feature_columns = [col for col in df_mri_info.columns if col not in ['df2_index', 'PTID']]
    
    # 标记MRI特征的来源
    mri_feature_sources = ['mri' for _ in mri_feature_columns]

    # 合并认知量表和MRI信息特征名称及来源
    combined_feature_columns = cog_feature_columns + mri_feature_columns
    combined_feature_sources = cog_feature_sources + mri_feature_sources

    return speech_feature_columns, combined_feature_columns, combined_feature_sources

speech_feature_names, cog_feature_names, cog_feature_sources = get_feature_names()
print(f"语音特征数量: {len(speech_feature_names)}")
print(f"认知量表特征数量: {len(cog_feature_names)}")
print(f"认知特征来源分布: cognitive={cog_feature_sources.count('cognitive')}, mri={cog_feature_sources.count('mri')}")

# 检测可用的模型文件
def find_available_models():
    """查找所有可用的模型文件（支持普通模式和交叉验证模式）"""
    models = []

    # 普通模式模型
    single_model_path = os.path.join(config['checkpoint_dir'], f'{config["teacher_model_name"]}_best.pth')
    if os.path.exists(single_model_path):
        models.append({
            'path': single_model_path,
            'name': 'single_model',
            'fold': 0
        })

    # 交叉验证模式模型
    if config['enable_cross_validation']:
        for fold in range(1, config['cv_folds'] + 1):
            cv_model_path = os.path.join(config['checkpoint_dir'], f'{config["teacher_model_name"]}_fold{fold}_best.pth')
            if os.path.exists(cv_model_path):
                models.append({
                    'path': cv_model_path,
                    'name': f'fold{fold}',
                    'fold': fold
                })

    return models

# 处理注意力权重
def process_attention_weights(attn_probs):
    """处理注意力权重，计算每个特征的平均注意力"""
    if attn_probs is None:
        return None

    # 取最后一层的注意力权重
    last_layer_attn = attn_probs[-1]

    # 取CLS token对其他token的注意力（如果使用了CLS token）
    # 假设第一个token是CLS token
    cls_attn = last_layer_attn[:, :, 0, 1:].mean(dim=1).mean(dim=0)

    return cls_attn.cpu().detach().numpy()

# 生成热力图
def generate_heatmap(attention_weights, feature_names, title, save_path, feature_sources=None, show_top_n=50):
    """生成并保存注意力热力图"""
    # 调整图大小以适应竖轴特征名称
    plt.figure(figsize=(10, 12))

    # 限制特征数量，避免图表过大
    if len(attention_weights) > show_top_n:
        # 选择注意力权重最高的前show_top_n个特征
        top_indices = np.argsort(attention_weights)[-show_top_n:]
        attention_weights = attention_weights[top_indices]
        feature_names = [feature_names[i] for i in top_indices]
        if feature_sources:
            feature_sources = [feature_sources[i] for i in top_indices]

    # 转置数据以将特征名称放在y轴
    attention_weights = np.array(attention_weights).reshape(-1, 1)

    # 创建热力图
    ax = sns.heatmap(attention_weights, cmap='YlOrRd', annot=False, yticklabels=feature_names, xticklabels=['Attention'])

    # 根据特征来源设置不同颜色
    if feature_sources:
        # 获取y轴标签
        labels = ax.get_yticklabels()
        for i, label in enumerate(labels):
            if feature_sources[i] == 'mri':
                label.set_color('blue')
            else:  # cognitive
                label.set_color('green')

    plt.title(title, fontsize=14, fontweight='bold')
    if feature_sources:
        plt.ylabel('Features (Green: Cognitive, Blue: MRI)', fontsize=12)
    else:
        plt.ylabel('Features', fontsize=12)
    plt.xlabel('Attention Weight', fontsize=12)
    plt.yticks(rotation=0, fontsize=8)
    plt.xticks(fontsize=10)
    plt.tight_layout()

    # 保存图表
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"热力图已保存到: {save_path}")
    plt.close()

# 生成综合热力图（多模型平均）
def generate_combined_heatmap(all_attention_weights, feature_names, title, save_path, feature_sources=None, show_top_n=50):
    """生成并保存多模型平均的注意力热力图"""
    # 调整图大小以适应竖轴特征名称
    plt.figure(figsize=(12, 14))

    # 计算平均注意力权重和标准差
    avg_attention = np.mean(all_attention_weights, axis=0)
    std_attention = np.std(all_attention_weights, axis=0)

    # 限制特征数量，避免图表过大
    if len(avg_attention) > show_top_n:
        # 选择平均注意力权重最高的前show_top_n个特征
        top_indices = np.argsort(avg_attention)[-show_top_n:]
        avg_attention = avg_attention[top_indices]
        std_attention = std_attention[top_indices]
        feature_names = [feature_names[i] for i in top_indices]
        if feature_sources:
            feature_sources = [feature_sources[i] for i in top_indices]

    # 创建热力图数据（转置以将特征名称放在y轴）
    heatmap_data = np.array(avg_attention).reshape(-1, 1)

    # 创建热力图
    ax = sns.heatmap(heatmap_data, cmap='YlOrRd', annot=False, yticklabels=feature_names,
                     xticklabels=['Mean Attention'], cbar_kws={'label': 'Attention Weight'})

    # 根据特征来源设置不同颜色
    if feature_sources:
        # 获取y轴标签
        labels = ax.get_yticklabels()
        for i, label in enumerate(labels):
            if feature_sources[i] == 'mri':
                label.set_color('blue')
            else:  # cognitive
                label.set_color('green')

    plt.title(f'{title}\n(Averaged over {len(all_attention_weights)} models)', fontsize=14, fontweight='bold')
    if feature_sources:
        plt.ylabel('Features (Green: Cognitive, Blue: MRI)', fontsize=12)
    else:
        plt.ylabel('Features', fontsize=12)
    plt.xlabel('Attention Statistics', fontsize=12)
    plt.yticks(rotation=0, fontsize=8)
    plt.xticks(fontsize=10)
    plt.tight_layout()

    # 保存图表
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"综合热力图已保存到: {save_path}")
    plt.close()

    # 返回特征重要性排序
    importance_ranking = np.argsort(avg_attention)[::-1]
    return [(feature_names[i], avg_attention[i], std_attention[i]) for i in importance_ranking]

# 为单个模型生成热力图
def generate_heatmaps_for_model(model, testloader, model_name, save_dir):
    """为单个模型生成热力图"""
    print(f"\n{'='*60}")
    print(f"为模型 {model_name} 生成热力图")
    print(f"{'='*60}")

    model.to(device)
    model.eval()

    # 收集所有样本的注意力权重
    all_speech_attn = []
    all_cog_attn = []

    with torch.no_grad():
        for i, batch in enumerate(testloader):
            if (i + 1) % 20 == 0 or i == 0:
                print(f"  处理样本 {i+1}/{len(testloader)}")

            # 准备输入
            image, text, cog, label, id = batch
            image = image.unsqueeze(1).to(device)
            text = text.to(device)
            cog = cog.to(device)

            # 预测并获取注意力权重
            y_hat_prob, y_hat_speech_prob, image_feat, text_feat, text_attn_probs, cog_attn_probs = model.predict(
                {'image': image, 'text': text, 'cog': cog, 'label': label, 'id': id}
            )

            # 处理注意力权重
            speech_attn = process_attention_weights(text_attn_probs)
            cog_attn = process_attention_weights(cog_attn_probs)

            if speech_attn is not None:
                all_speech_attn.append(speech_attn)
            if cog_attn is not None:
                all_cog_attn.append(cog_attn)

    # 生成单独模型的热力图
    if all_speech_attn:
        avg_speech_attn = np.mean(all_speech_attn, axis=0)
        model_save_dir = os.path.join(save_dir, model_name)
        generate_heatmap(
            avg_speech_attn,
            speech_feature_names,
            f'Speech Features Attention - {model_name}',
            os.path.join(model_save_dir, 'speech_attention.png')
        )

    if all_cog_attn:
        avg_cog_attn = np.mean(all_cog_attn, axis=0)
        model_save_dir = os.path.join(save_dir, model_name)
        generate_heatmap(
            avg_cog_attn,
            cog_feature_names,
            f'Cognitive Features Attention - {model_name}',
            os.path.join(model_save_dir, 'cog_attention.png'),
            feature_sources=cog_feature_sources
        )

    return all_speech_attn, all_cog_attn

# 主函数
def main():
    """主函数，处理数据并生成热力图"""
    print("\n" + "="*60)
    print("注意力热力图可视化工具（支持交叉验证）")
    print("="*60)

    # 查找可用的模型
    available_models = find_available_models()
    if not available_models:
        print("错误: 未找到任何模型文件！")
        sys.exit(1)

    print(f"\n找到 {len(available_models)} 个可用模型:")
    for m in available_models:
        print(f"  - {m['name']}: {m['path']}")

    # 确定是否使用交叉验证模式
    use_cv = any(m['fold'] > 0 for m in available_models)
    print(f"\n使用模式: {'交叉验证模式' if use_cv else '单模型模式'}")

    # 加载数据（交叉验证模式需要合并训练和测试数据）
    if use_cv:
        print("\n加载交叉验证数据...")
        dataloaders = get_kfold_dataloaders(
            datalist=config['cv_datalist'],
            n_splits=config['cv_folds'],
            val_split_ratio=config['val_split_ratio'],
            batch_size_train=config['batch_size_val'],
            batch_size_eval=config['batch_size_val'],
            txt_len=config['txt_len'],
            num_workers=config['num_workers'],
            random_state=42,
            balance_data=False,
        )
        # 使用所有数据的合并测试集
        # 获取最后一个fold的测试集作为通用测试集
        _, _, testloader = dataloaders[-1]

        # 动态更新认知量表特征维度
        for batch in testloader:
            if len(batch) == 5:
                _, _, cog_features, _, _ = batch
                config['cog_len'] = cog_features.shape[1]
                print(f"动态更新认知量表特征维度: {config['cog_len']}")
                break
    else:
        print("\n加载测试数据...")
        testloader = get_dataloader(
            datalist=config['test_datalist'],
            batch_size=config['batch_size_val'],
            txt_len=config['txt_len'],
            shuffle=False,
            num_workers=config['num_workers'],
            drop_last=False
        )

        # 动态更新认知量表特征维度
        for batch in testloader:
            if len(batch) == 5:
                _, _, cog_features, _, _ = batch
                config['cog_len'] = cog_features.shape[1]
                print(f"动态更新认知量表特征维度: {config['cog_len']}")
                break

    # 保存目录
    save_dir = 'attention_heatmaps'
    os.makedirs(save_dir, exist_ok=True)

    # 收集所有模型的注意力权重
    all_speech_attn_list = []
    all_cog_attn_list = []

    # 为每个模型生成热力图
    for model_info in available_models:
        # 创建模型
        model = MedBLIPModel(
            max_txt_len=config['txt_len'],
            num_numerical_features=config['txt_len'],
            num_cog_features=config['cog_len'],
            use_ft_transformer=config['use_ft_transformer'],
            use_positional_embedding=config['use_position_embedding'],
            use_cross_attention=config['use_cross_attention'],
            training_mode='teacher',
            modalities=config['modalities']
        )

        # 加载模型权重
        model.load_state_dict(torch.load(model_info['path'], map_location=device))
        print(f"\n成功加载模型: {model_info['path']}")

        # 生成热力图并收集注意力权重
        speech_attn, cog_attn = generate_heatmaps_for_model(model, testloader, model_info['name'], save_dir)

        if speech_attn:
            all_speech_attn_list.extend(speech_attn)
        if cog_attn:
            all_cog_attn_list.extend(cog_attn)

        # 释放内存
        del model
        torch.cuda.empty_cache()

    # 生成综合热力图（多模型平均）
    if len(available_models) > 1:
        print(f"\n{'='*60}")
        print(f"生成 {len(available_models)} 个模型的综合热力图")
        print(f"{'='*60}")

        # 计算每个模型的平均注意力，然后对所有模型取平均
        if all_speech_attn_list:
            n_samples = len(all_speech_attn_list) // len(available_models)
            model_avg_speech = []
            for i in range(len(available_models)):
                start_idx = i * n_samples
                end_idx = start_idx + n_samples
                model_avg = np.mean(all_speech_attn_list[start_idx:end_idx], axis=0)
                model_avg_speech.append(model_avg)

            # 生成综合热力图
            generate_combined_heatmap(
                model_avg_speech,
                speech_feature_names,
                'Combined Speech Features Attention (All Folds)',
                os.path.join(save_dir, 'combined_speech_attention.png')
            )

        if all_cog_attn_list:
            n_samples = len(all_cog_attn_list) // len(available_models)
            model_avg_cog = []
            for i in range(len(available_models)):
                start_idx = i * n_samples
                end_idx = start_idx + n_samples
                model_avg = np.mean(all_cog_attn_list[start_idx:end_idx], axis=0)
                model_avg_cog.append(model_avg)

            # 生成综合热力图
            importance_ranking = generate_combined_heatmap(
                model_avg_cog,
                cog_feature_names,
                'Combined Cognitive Features Attention (All Folds)',
                os.path.join(save_dir, 'combined_cog_attention.png'),
                feature_sources=cog_feature_sources
            )

            # 保存特征重要性排名到CSV
            ranking_df = pd.DataFrame(importance_ranking, columns=['Feature Name', 'Mean Attention', 'Std Attention'])
            ranking_df.to_csv(os.path.join(save_dir, 'feature_importance_ranking.csv'), index=False)
            print(f"特征重要性排名已保存到: {os.path.join(save_dir, 'feature_importance_ranking.csv')}")

    print(f"\n{'='*60}")
    print("注意力热力图生成完成！")
    print(f"{'='*60}")
    print(f"热力图保存在: {os.path.abspath(save_dir)}")

if __name__ == "__main__":
    main()
