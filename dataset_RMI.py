import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import SimpleITK as sitk
import pandas as pd
from sklearn.preprocessing import StandardScaler
# joblib 用于保存/加载标准化器（scaler）
try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None
# 删除PCA导入

label_map = {
    'AD': 1,
    'MCI': 1,
    'CN': 0,
}

class Dataset(torch.utils.data.Dataset):
    """
    Loads data and corresponding label and returns pytorch float tensor.
    """
    def __init__(self, data):
        self.files = data

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        Read data and label and return them.
        """
        img_path = self.files[idx]['image_path']

        # 使用SimpleITK
        image_sitk = sitk.ReadImage(img_path)

        # 1. 标准化
        normalizer = sitk.NormalizeImageFilter()
        normalized_image = normalizer.Execute(image_sitk)

        # 2. 直接重采样到目标尺寸，避免后续padding
        original_size = normalized_image.GetSize()
        target_size = [128, 128, 128]

        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(target_size)
        # 保持原有的物理间距比例
        # 修正重采样逻辑，保持物理空间的一致性
        original_spacing = normalized_image.GetSpacing()
        output_spacing = [orig_sp * orig_sz / target_sz 
                        for orig_sp, orig_sz, target_sz in 
                        zip(original_spacing, original_size, target_size)]
        resampler.SetOutputSpacing(output_spacing)
        resampler.SetInterpolator(sitk.sitkLinear)  # 线性插值
        resampled_image = resampler.Execute(normalized_image)

        # 3. 直接转换为tensor，无需padding和resize
        data = sitk.GetArrayFromImage(resampled_image).astype(np.float32)
        image = torch.FloatTensor(data)  # 已经是128x128x128

        features = self.files[idx]['features']
        cog_features = self.files[idx]['cog_features']
        label = self.files[idx]['label']
        id = self.files[idx]['id']

        return image, features, cog_features, label, id


def balance_dataset(files, random_state=None):
    """
    平衡数据集，创建1:1比例的子数据集
    
    参数:
        files: 原始数据文件列表
        random_state: 随机种子
    
    返回:
        平衡后的数据集文件列表
    """
    # 按标签分组
    label_groups = {}
    for file in files:
        label = file['label']
        if label not in label_groups:
            label_groups[label] = []
        label_groups[label].append(file)
    
    # 计算每个类别的样本数
    label_counts = {label: len(files) for label, files in label_groups.items()}
    print(f"原始数据分布: {label_counts}")
    
    # 确定最小样本数
    min_count = min(label_counts.values())
    print(f"最小类别样本数: {min_count}")
    print("创建1:1比例的平衡数据集...")
    
    # 从每个类别中随机选取min_count个样本
    balanced_files = []
    rng = np.random.RandomState(random_state)
    for label, files in label_groups.items():
        # 随机采样
        sampled_files = rng.choice(files, size=min_count, replace=False)
        balanced_files.extend(sampled_files.tolist())
    
    # 打乱顺序
    rng.shuffle(balanced_files)
    print(f"平衡后数据集大小: {len(balanced_files)}")
    
    # 验证平衡后的分布
    balanced_counts = {}
    for file in balanced_files:
        label = file['label']
        balanced_counts[label] = balanced_counts.get(label, 0) + 1
    print(f"平衡后数据分布: {balanced_counts}")
    
    return balanced_files

def get_dataloader(
    datalist=['ADNI-train'],
    batch_size=1,
    txt_len=60,
    shuffle=False,
    num_workers=12,
    drop_last=False,
    val_split_ratio=None,
    random_state=None,
    balance_data=False,
    scaler=None,
    cog_scaler=None,
    scaler_save_path=None,
):
    """
    创建数据加载器，可以选择性地划分训练集和验证集
    
    参数:
        datalist: 数据集列表
        batch_size: 批次大小
        txt_len: 文本长度（保留参数，不再使用）
        shuffle: 是否打乱数据
        num_workers: 工作线程数
        drop_last: 是否丢弃最后一个不完整的批次
        val_split_ratio: 验证集划分比例，如果为None则不划分
        random_state: 随机种子，用于划分数据集
        balance_data: 是否平衡数据集（创建1:1比例的子数据集）
    
    返回:
        如果val_split_ratio为None: 返回一个数据加载器
        否则: 返回(train_dataloader, val_dataloader)两个数据加载器
    """
    all_features = []
    all_cog_features = []
    all_img_paths = []
    all_labels = []
    all_ptids = []  # 保存所有病人ID
    
    # 第一步：收集所有数据
    for data in datalist:
        filename = f'local_data/{data}.csv'
        print('load data from', filename)
        
        df = pd.read_csv(filename, header=0)
        
        # 提取特征列
        feature_columns = [col for col in df.columns if df.columns.get_loc(col) >= df.columns.get_loc('SCREENER_SCORE')]
        print(f"使用 {len(feature_columns)} 个特征列")
        
        # 批量提取所有特征
        features_batch = df[feature_columns].values.astype(np.float32)
        features_batch = np.nan_to_num(features_batch)  # 处理缺失值

        # 处理对应的认知量表模态
        if 'train' in data:
            filename_cog = f'local_data/information-train.csv'
        elif 'test' in data:
            filename_cog = f'local_data/information-test.csv'
        df_cog = pd.read_csv(filename_cog, header=0)
        # 删除重复的PTID值，只保留第一个出现的
        df_cog = df_cog.drop_duplicates(subset=['PTID'], keep='first')
        # 根据病人ID匹配语音特征和认知量表数据（包含一系列检测结果数据，不止SCORE）
        cog_dict = df_cog.set_index('PTID').to_dict(orient='index')
        cog_features = []
        for pid in df['PTID']:
            if pid in cog_dict:
                cog_data = cog_dict[pid]
                # 提取认知量表数据（将非数值的字母转化为数值（需要区分大小写））
                cog_row = []
                # 将dict_keys转换为列表
                cog_keys = list(cog_data.keys())
                # 找到TRAILS的索引
                trails_index = cog_keys.index('TOTSCORE') if 'TOTSCORE' in cog_keys else 0
                for key, value in cog_data.items():
                    # 只提取在TRAILS之后的列
                    if cog_keys.index(key) < trails_index:
                        continue
                    try:
                        cog_row.append(float(value))
                    except ValueError:
                        cog_row.append(float(ord(value)))  # 转化为ASCII码数值
                cog_features.append(cog_row)
            else:
                # 如果没有对应的认知量表数据，填充NaN
                # 将dict_keys转换为列表
                if len(cog_dict) > 0:
                    sample_cog = next(iter(cog_dict.values()))
                    sample_keys = list(sample_cog.keys())
                    trails_index = sample_keys.index('TOTSCORE') if 'TOTSCORE' in sample_keys else 0
                    fill_len = len(sample_cog) - trails_index
                else:
                    fill_len = 1
                cog_features.append([np.nan] * fill_len)
        
        # 读取并合并新的MRI信息数据
        filename_mri_info = f'local_data/MRI_information_matched_to_RMI_MRI_split.csv'
        df_mri_info = pd.read_csv(filename_mri_info, header=0)
        # 删除重复的PTID值，只保留第一个出现的
        df_mri_info = df_mri_info.drop_duplicates(subset=['PTID'], keep='first')
        # 根据病人ID匹配MRI信息数据
        mri_info_dict = df_mri_info.set_index('PTID').to_dict(orient='index')
        
        # 合并MRI信息到认知量表特征中
        for i, pid in enumerate(df['PTID']):
            if pid in mri_info_dict:
                mri_data = mri_info_dict[pid]
                # 提取所有数值特征（跳过非数值列）
                mri_row = []
                for key, value in mri_data.items():
                    if key != 'df2_index' and key != 'PTID':  # 跳过索引和ID列
                        try:
                            mri_row.append(float(value))
                        except (ValueError, TypeError):
                            mri_row.append(float(0))  # 处理非数值或缺失值
                # 将MRI信息添加到认知量表特征中
                cog_features[i].extend(mri_row)
            else:
                # 如果没有对应的MRI信息，填充0
                if len(mri_info_dict) > 0:
                    sample_mri = next(iter(mri_info_dict.values()))
                    # 计算需要填充的长度（跳过非数值列）
                    fill_len = sum(1 for key in sample_mri.keys() if key != 'df2_index' and key != 'PTID')
                else:
                    fill_len = 0
                cog_features[i].extend([0.0] * fill_len)
#        # 补充的认知量表数据
#        if 'train' in data:
#            filename_cog = f'local_data/MMSE-train.csv'
#        elif 'test' in data:
#            filename_cog = f'local_data/MMSE-test.csv'
#        df_cog = pd.read_csv(filename_cog, header=0)
#        # 删除重复的PTID值，只保留第一个出现的
#        df_cog = df_cog.drop_duplicates(subset=['PTID'], keep='first')
#        # 根据病人ID匹配语音特征和认知量表数据（包含一系列检测结果数据，不止SCORE）
#        cog_dict = df_cog.set_index('PTID').to_dict(orient='index')
#        for i, pid in enumerate(df['PTID']):
#            if pid in cog_dict:
#                cog_data = cog_dict[pid]
#                # 提取认知量表数据（将非数值的字母转化为数值（需要区分大小写））
#                # 将dict_keys转换为列表
#                cog_keys = list(cog_data.keys())
#                # 找到DONE的索引
#                done_index = cog_keys.index('DONE') if 'DONE' in cog_keys else 0
#                for key, value in cog_data.items():
#                    # 只提取在DONE之后的列
#                    if cog_keys.index(key) < done_index:
#                        continue
#                    try:
#                        cog_features[i].append(float(value))
#                    except ValueError:
#                        cog_features[i].append(float(ord(value)))  # 转化为ASCII码数值
#            else:
#                # 如果没有对应的认知量表数据，填充NaN
#                # 将dict_keys转换为列表
#                cog_keys = list(cog_data.keys())
#                # 找到DONE的索引
#                done_index = cog_keys.index('DONE') if 'DONE' in cog_keys else 0
#                cog_features[i].extend([np.nan] * (len(cog_data) - done_index))
        # 将认知量表特征转为numpy数组 
        cog_features = np.array(cog_features, dtype=np.float32)
        cog_features = np.nan_to_num(cog_features)  # 处理缺失值
        
        # 收集数据
        all_features.append(features_batch)
        all_cog_features.append(cog_features)
        all_img_paths.extend(df['MRI_Path'].tolist())
        all_labels.extend([label_map.get(group, -1) for group in df['Group']])
        all_ptids.extend(df['PTID'].tolist())  # 保存病人ID
    
    # 生成all_ids
    all_ids = [abs(hash(path)) for path in all_img_paths]
    
    # 合并所有特征
    all_features = np.vstack(all_features)
    all_cog_features = np.vstack(all_cog_features)
    print(f"总样本数: {len(all_features)}")
    print(f"特征矩阵形状: {all_features.shape}")
    print(f"认知量表特征矩阵形状: {all_cog_features.shape}")
    
    # 第三步：创建数据列表，同时保存病人ID
    files = []
    patient_data = {}
    
    # 首先收集所有数据并按病人ID分组
    for i in range(len(all_img_paths)):
        # 获取病人ID（从保存的列表中读取）
        ptid = all_ptids[i]
        
        file_item = {
            'image_path': all_img_paths[i],
            'features': torch.FloatTensor(all_features[i]),  # 使用原始特征
            'cog_features': torch.FloatTensor(all_cog_features[i]),  # 使用原始认知量表特征
            'label': all_labels[i],
            'id': all_ids[i],
            'ptid': ptid  # 保存病人ID
        }
        files.append(file_item)
        
        # 按病人ID分组
        if ptid not in patient_data:
            patient_data[ptid] = []
        patient_data[ptid].append(file_item)
    
    # 如果不需要划分验证集，直接返回一个数据加载器
    if val_split_ratio is None:
        all_features_np = np.vstack([item['features'].numpy() for item in files])
        all_cog_features_np = np.vstack([item['cog_features'].numpy() for item in files])

        # 标准化策略：
        # - 如果传入了 scaler/cog_scaler（例如测试阶段），则直接 transform
        # - 否则（例如单独加载一个数据集），在该数据集上 fit_transform
        if scaler is None:
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(all_features_np)
        else:
            features_scaled = scaler.transform(all_features_np)

        if cog_scaler is None:
            cog_scaler = StandardScaler()
            cog_features_scaled = cog_scaler.fit_transform(all_cog_features_np)
        else:
            cog_features_scaled = cog_scaler.transform(all_cog_features_np)
        
        # 更新特征
        for i, item in enumerate(files):
            item['features'] = torch.FloatTensor(features_scaled[i])
            item['cog_features'] = torch.FloatTensor(cog_features_scaled[i])
        
        print("标准化完成")

        # 可选：保存标准化器（一般用于训练阶段）
        if scaler_save_path is not None:
            if joblib is None:
                raise ImportError("未找到 joblib，无法保存 scaler。请先安装：pip install joblib")
            import os
            os.makedirs(os.path.dirname(scaler_save_path) or ".", exist_ok=True)
            joblib.dump({'scaler': scaler, 'cog_scaler': cog_scaler}, scaler_save_path)
            print(f"已保存标准化器到: {scaler_save_path}")
        
        # 不再进行PCA降维，直接使用所有特征
        # 注意：txt_len参数现在不再使用，保留是为了兼容性
        if txt_len is not None and txt_len < features_scaled.shape[1]:
            print(f"警告：txt_len参数({txt_len})小于特征数({features_scaled.shape[1]})，但不再进行降维")
            print(f"使用所有{features_scaled.shape[1]}个特征")
        
        # 如果需要平衡数据
        if balance_data:
            files = balance_dataset(files, random_state)
        
        dataset = Dataset(data=files)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)
        return dataloader
    else:
        # 基于病人ID进行划分，确保同一个病人的所有数据都在同一个集合中
        from sklearn.model_selection import train_test_split
        print(f"将数据按 {1-val_split_ratio:.2f}:{val_split_ratio:.2f} 划分为训练集和验证集...")
        
        # 获取所有病人ID
        patient_ids = list(patient_data.keys())
        
        # 计算每个病人的标签（使用该病人的第一个样本的标签）
        patient_labels = [patient_data[pid][0]['label'] for pid in patient_ids]
        
        # 按病人ID划分，保持标签分层
        train_patient_ids, val_patient_ids = train_test_split(
            patient_ids, 
            test_size=val_split_ratio, 
            random_state=random_state, 
            stratify=patient_labels
        )
        
        # 根据划分的病人ID收集训练集和验证集数据
        train_files = []
        for pid in train_patient_ids:
            train_files.extend(patient_data[pid])
        
        val_files = []
        for pid in val_patient_ids:
            val_files.extend(patient_data[pid])
        
        # 在训练集上进行标准化
        scaler = StandardScaler()
        train_features_np = np.vstack([item['features'].numpy() for item in train_files])
        scaler.fit(train_features_np)
        
        cog_scaler = StandardScaler()
        train_cog_features_np = np.vstack([item['cog_features'].numpy() for item in train_files])
        cog_scaler.fit(train_cog_features_np)
        
        # 转换训练集
        for item in train_files:
            item['features'] = torch.FloatTensor(scaler.transform(item['features'].numpy().reshape(1, -1)).flatten())
            item['cog_features'] = torch.FloatTensor(cog_scaler.transform(item['cog_features'].numpy().reshape(1, -1)).flatten())
        
        # 用训练集的标准化器转换验证集
        for item in val_files:
            item['features'] = torch.FloatTensor(scaler.transform(item['features'].numpy().reshape(1, -1)).flatten())
            item['cog_features'] = torch.FloatTensor(cog_scaler.transform(item['cog_features'].numpy().reshape(1, -1)).flatten())
        
        print("标准化完成")

        # 可选：保存训练集拟合得到的标准化器（推荐用于后续测试集复用）
        if scaler_save_path is not None:
            if joblib is None:
                raise ImportError("未找到 joblib，无法保存 scaler。请先安装：pip install joblib")
            import os
            os.makedirs(os.path.dirname(scaler_save_path) or ".", exist_ok=True)
            joblib.dump({'scaler': scaler, 'cog_scaler': cog_scaler}, scaler_save_path)
            print(f"已保存标准化器到: {scaler_save_path}")
        
        # 不再进行PCA降维，直接使用所有特征
        # 注意：txt_len参数现在不再使用，保留是为了兼容性
        if txt_len is not None and txt_len < train_features_np.shape[1]:
            print(f"警告：txt_len参数({txt_len})小于特征数({train_features_np.shape[1]})，但不再进行降维")
            print(f"使用所有{train_features_np.shape[1]}个特征")
        
        # 如果需要平衡训练集数据
        if balance_data:
            print("平衡训练集数据...")
            train_files = balance_dataset(train_files, random_state)
        
        print(f"训练集大小: {len(train_files)}, 验证集大小: {len(val_files)}")
        print(f"训练集病人数: {len(train_patient_ids)}, 验证集病人数: {len(val_patient_ids)}")
        
        # 保存划分结果到CSV文件，方便检查
        import os
        
        # 创建保存目录
        output_dir = 'local_data/split_check'
        os.makedirs(output_dir, exist_ok=True)
        
        # 准备训练集数据
        train_data = []
        for item in train_files:
            train_data.append({
                'PTID': item['ptid'],
                'Label': item['label'],
                'ImagePath': item['image_path']
            })
        
        # 准备验证集数据
        val_data = []
        for item in val_files:
            val_data.append({
                'PTID': item['ptid'],
                'Label': item['label'],
                'ImagePath': item['image_path']
            })
        
        # 保存为CSV文件
        train_df = pd.DataFrame(train_data)
        val_df = pd.DataFrame(val_data)
        
        train_df.to_csv(os.path.join(output_dir, 'train_split.csv'), index=False)
        val_df.to_csv(os.path.join(output_dir, 'val_split.csv'), index=False)
        
        print(f"划分结果已保存到: {output_dir}")
        print(f"训练集CSV: {os.path.join(output_dir, 'train_split.csv')}")
        print(f"验证集CSV: {os.path.join(output_dir, 'val_split.csv')}")
        
        # 创建训练集数据加载器
        train_dataset = Dataset(data=train_files)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)
        
        # 创建验证集数据加载器
        val_dataset = Dataset(data=val_files)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
        
        return train_dataloader, val_dataloader


def get_kfold_dataloaders(
    datalist,
    n_splits=5,
    val_split_ratio=0.2,
    batch_size_train=4,
    batch_size_eval=24,
    txt_len=60,
    num_workers=2,
    random_state=42,
    balance_data=False,
):
    """
    基于病人ID进行分层K折划分：
    - 每一折使用不同病人子集作为测试集
    - 剩余病人再按 val_split_ratio 划分训练/验证
    - 标准化器仅在训练集上拟合，再应用到验证和测试集
    """
    from sklearn.model_selection import StratifiedKFold, train_test_split

    all_features = []
    all_cog_features = []
    all_img_paths = []
    all_labels = []
    all_ptids = []

    for data in datalist:
        filename = f'local_data/{data}.csv'
        print('load data from', filename)
        df = pd.read_csv(filename, header=0)

        feature_columns = [col for col in df.columns if df.columns.get_loc(col) >= df.columns.get_loc('SCREENER_SCORE')]
        print(f"使用 {len(feature_columns)} 个特征列")
        features_batch = df[feature_columns].values.astype(np.float32)
        features_batch = np.nan_to_num(features_batch)

        # 认知量表数据加载逻辑与现有流程保持一致
        if 'train' in data:
            filename_cog = f'local_data/information-train.csv'
        elif 'test' in data:
            filename_cog = f'local_data/information-test.csv'
        else:
            filename_cog = f'local_data/information-train.csv'

        df_cog = pd.read_csv(filename_cog, header=0)
        df_cog = df_cog.drop_duplicates(subset=['PTID'], keep='first')
        cog_dict = df_cog.set_index('PTID').to_dict(orient='index')
        cog_features = []
        for pid in df['PTID']:
            if pid in cog_dict:
                cog_data = cog_dict[pid]
                cog_row = []
                cog_keys = list(cog_data.keys())
                trails_index = cog_keys.index('TOTSCORE') if 'TOTSCORE' in cog_keys else 0
                for key, value in cog_data.items():
                    if cog_keys.index(key) < trails_index:
                        continue
                    try:
                        cog_row.append(float(value))
                    except ValueError:
                        cog_row.append(float(ord(value)))
                cog_features.append(cog_row)
            else:
                # 缺失认知数据时，使用与已有数据一致的列数填充
                if len(cog_dict) > 0:
                    sample_cog = next(iter(cog_dict.values()))
                    sample_keys = list(sample_cog.keys())
                    trails_index = sample_keys.index('TOTSCORE') if 'TOTSCORE' in sample_keys else 0
                    fill_len = len(sample_cog) - trails_index
                else:
                    fill_len = 1
                cog_features.append([np.nan] * fill_len)
        
        # 读取并合并新的MRI信息数据
        filename_mri_info = f'local_data/MRI_information_matched_to_RMI_MRI_split.csv'
        df_mri_info = pd.read_csv(filename_mri_info, header=0)
        # 删除重复的PTID值，只保留第一个出现的
        df_mri_info = df_mri_info.drop_duplicates(subset=['PTID'], keep='first')
        # 根据病人ID匹配MRI信息数据
        mri_info_dict = df_mri_info.set_index('PTID').to_dict(orient='index')
        
        # 合并MRI信息到认知量表特征中
        for i, pid in enumerate(df['PTID']):
            if pid in mri_info_dict:
                mri_data = mri_info_dict[pid]
                # 提取所有数值特征（跳过非数值列）
                mri_row = []
                for key, value in mri_data.items():
                    if key != 'df2_index' and key != 'PTID':  # 跳过索引和ID列
                        try:
                            mri_row.append(float(value))
                        except (ValueError, TypeError):
                            mri_row.append(float(0))  # 处理非数值或缺失值
                # 将MRI信息添加到认知量表特征中
                cog_features[i].extend(mri_row)
            else:
                # 如果没有对应的MRI信息，填充0
                if len(mri_info_dict) > 0:
                    sample_mri = next(iter(mri_info_dict.values()))
                    # 计算需要填充的长度（跳过非数值列）
                    fill_len = sum(1 for key in sample_mri.keys() if key != 'df2_index' and key != 'PTID')
                else:
                    fill_len = 0
                cog_features[i].extend([0.0] * fill_len)

        cog_features = np.array(cog_features, dtype=np.float32)
        cog_features = np.nan_to_num(cog_features)

        all_features.append(features_batch)
        all_cog_features.append(cog_features)
        all_img_paths.extend(df['MRI_Path'].tolist())
        all_labels.extend([label_map.get(group, -1) for group in df['Group']])
        all_ptids.extend(df['PTID'].tolist())

    all_ids = [abs(hash(path)) for path in all_img_paths]
    all_features = np.vstack(all_features)
    all_cog_features = np.vstack(all_cog_features)

    files = []
    patient_data = {}
    for i in range(len(all_img_paths)):
        ptid = all_ptids[i]
        file_item = {
            'image_path': all_img_paths[i],
            'features': torch.FloatTensor(all_features[i]),
            'cog_features': torch.FloatTensor(all_cog_features[i]),
            'label': all_labels[i],
            'id': all_ids[i],
            'ptid': ptid
        }
        files.append(file_item)
        if ptid not in patient_data:
            patient_data[ptid] = []
        patient_data[ptid].append(file_item)

    def _count_labels_from_files(split_files):
        """统计样本级标签数量。"""
        counts = {}
        for item in split_files:
            label = int(item['label'])
            counts[label] = counts.get(label, 0) + 1
        return counts

    def _count_labels_from_patients(split_patient_ids):
        """统计病人级标签数量（每个病人计1次）。"""
        counts = {}
        for pid in split_patient_ids:
            label = int(patient_data[pid][0]['label'])
            counts[label] = counts.get(label, 0) + 1
        return counts

    def _format_counts_and_ratio(counts):
        """将标签计数格式化为可读字符串（含比例）。"""
        total = sum(counts.values())
        if total == 0:
            return "total=0"
        labels_sorted = sorted(counts.keys())
        parts = [f"total={total}"]
        for label in labels_sorted:
            cnt = counts[label]
            ratio = cnt / total
            parts.append(f"class{label}={cnt}({ratio:.2%})")
        return ", ".join(parts)

    patient_ids = list(patient_data.keys())
    patient_labels = [patient_data[pid][0]['label'] for pid in patient_ids]
    all_patient_counts = _count_labels_from_patients(patient_ids)
    all_sample_counts = _count_labels_from_files(files)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_loaders = []

    for fold_idx, (trainval_idx, test_idx) in enumerate(skf.split(patient_ids, patient_labels), start=1):
        trainval_patient_ids = [patient_ids[i] for i in trainval_idx]
        test_patient_ids = [patient_ids[i] for i in test_idx]
        trainval_labels = [patient_data[pid][0]['label'] for pid in trainval_patient_ids]

        train_patient_ids, val_patient_ids = train_test_split(
            trainval_patient_ids,
            test_size=val_split_ratio,
            random_state=random_state + fold_idx,
            stratify=trainval_labels
        )

        train_files = []
        for pid in train_patient_ids:
            train_files.extend(patient_data[pid])
        val_files = []
        for pid in val_patient_ids:
            val_files.extend(patient_data[pid])
        test_files = []
        for pid in test_patient_ids:
            test_files.extend(patient_data[pid])

        # 统计每折病人级/样本级标签分布，便于检查分层效果
        fold_train_patient_counts = _count_labels_from_patients(train_patient_ids)
        fold_val_patient_counts = _count_labels_from_patients(val_patient_ids)
        fold_test_patient_counts = _count_labels_from_patients(test_patient_ids)

        fold_train_sample_counts = _count_labels_from_files(train_files)
        fold_val_sample_counts = _count_labels_from_files(val_files)
        fold_test_sample_counts = _count_labels_from_files(test_files)

        print(f"[Fold {fold_idx}/{n_splits}] 病人级分布:")
        print(f"  overall -> {_format_counts_and_ratio(all_patient_counts)}")
        print(f"  train   -> {_format_counts_and_ratio(fold_train_patient_counts)}")
        print(f"  val     -> {_format_counts_and_ratio(fold_val_patient_counts)}")
        print(f"  test    -> {_format_counts_and_ratio(fold_test_patient_counts)}")

        print(f"[Fold {fold_idx}/{n_splits}] 样本级分布:")
        print(f"  overall -> {_format_counts_and_ratio(all_sample_counts)}")
        print(f"  train   -> {_format_counts_and_ratio(fold_train_sample_counts)}")
        print(f"  val     -> {_format_counts_and_ratio(fold_val_sample_counts)}")
        print(f"  test    -> {_format_counts_and_ratio(fold_test_sample_counts)}")

        # 标准化：仅在训练集拟合
        scaler = StandardScaler()
        train_features_np = np.vstack([item['features'].numpy() for item in train_files])
        scaler.fit(train_features_np)

        cog_scaler = StandardScaler()
        train_cog_features_np = np.vstack([item['cog_features'].numpy() for item in train_files])
        cog_scaler.fit(train_cog_features_np)

        def _apply_scale(split_files):
            for item in split_files:
                item['features'] = torch.FloatTensor(
                    scaler.transform(item['features'].numpy().reshape(1, -1)).flatten()
                )
                item['cog_features'] = torch.FloatTensor(
                    cog_scaler.transform(item['cog_features'].numpy().reshape(1, -1)).flatten()
                )
            return split_files

        train_files = _apply_scale(train_files)
        val_files = _apply_scale(val_files)
        test_files = _apply_scale(test_files)

        if balance_data:
            print(f"[Fold {fold_idx}] 平衡训练集数据...")
            train_files = balance_dataset(train_files, random_state + fold_idx)

        train_dataset = Dataset(data=train_files)
        val_dataset = Dataset(data=val_files)
        test_dataset = Dataset(data=test_files)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size_train,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size_eval,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size_eval,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False
        )

        print(f"[Fold {fold_idx}/{n_splits}] train/val/test 样本数: {len(train_files)}/{len(val_files)}/{len(test_files)}")
        fold_loaders.append((train_loader, val_loader, test_loader))

    return fold_loaders