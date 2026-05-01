# 方法设计（论文写作草稿）

## 1. 多模态教师模型

### 1.1 输入数据
教师模型使用两条实际参与训练的结构化模态路径：
- 语音表格特征（由 `RMT_MRI-*.csv` 中从 `SCREENER_SCORE` 起始列截取的数值向量）。
- 认知量表+MRI数值特征（`information-*.csv` 与 `MRI_information_matched_to_RMI_MRI_split.csv` 按 `PTID` 对齐后拼接）。

样本标签采用二分类映射：
\[
y \in \{0,1\},\quad \text{CN}\mapsto 0,\; \text{MCI/AD}\mapsto 1.
\]

设第 \(i\) 个样本的输入为：
\[
\mathbf{x}^{(i)}_s\in\mathbb{R}^{d_s},\quad \mathbf{x}^{(i)}_c\in\mathbb{R}^{d_c},\quad y^{(i)}\in\{0,1\}.
\]
其中 \(\mathbf{x}_s\) 为语音数值特征，\(\mathbf{x}_c\) 为认知量表与MRI数值特征拼接向量。

### 1.2 整体框架
教师模型遵循“**分类分支 + 重建分支**”并行结构：
1. 每个模态各有两条分支：
   - 分类分支：输出判别特征用于最终分类；
   - 重建分支：通过掩码重建进行自监督约束。
2. 仅重建分支使用跨模态交叉注意力；分类分支不显式做跨模态注意力。
3. 两条分支除交叉注意力模块外共享同源编码表示与主干参数。

记语音与认知模态编码器输出为：
\[
\mathbf{h}_s = E_s(\mathbf{x}_s),\qquad \mathbf{h}_c = E_c(\mathbf{x}_c).
\]
重建分支引入跨模态增强：
\[
\tilde{\mathbf{h}}_s = \operatorname{CA}_{s\leftarrow c}(\mathbf{h}_s,\mathbf{h}_c),\qquad
\tilde{\mathbf{h}}_c = \operatorname{CA}_{c\leftarrow s}(\mathbf{h}_c,\mathbf{h}_s).
\]

### 1.3 数据处理
数据处理流程可概括为：
1. **语音表格特征提取**：从 `SCREENER_SCORE` 起的连续列读取并转为 `float32`。
2. **认知量表数值化**：对非数值字符按 ASCII 码映射，缺失置 `NaN` 后统一处理。
3. **MRI结构化信息拼接**：按 `PTID` 对齐后与认知量表特征拼接。
4. **标准化**：训练集拟合 `StandardScaler`，验证/测试复用训练统计量。
5. **类别不均衡处理**：训练阶段可选类别重加权或构建 1:1 平衡子集。

设标准化算子为 \(\mathcal{N}(\cdot)\)，则：
\[
\hat{\mathbf{x}}_s = \mathcal{N}_s(\mathbf{x}_s),\qquad
\hat{\mathbf{x}}_c = \mathcal{N}_c(\mathbf{x}_c).
\]

### 1.4 特征提取器

#### 1.4.1 单模态编码器（输出后续用于分类的特征）
两模态均采用 `TableFTTRestorer`（FT-Transformer风格）得到 768 维判别表示：
\[
\mathbf{z}_s = f_s(\hat{\mathbf{x}}_s),\qquad \mathbf{z}_c = f_c(\hat{\mathbf{x}}_c),
\quad \mathbf{z}_s,\mathbf{z}_c\in\mathbb{R}^{768}.
\]

#### 1.4.2 跨模态编码器（重建部分编码器：利用另一模态信息进行交叉注意力）
在重建分支中，查询来自当前模态，键值来自另一模态：
\[
\operatorname{Attn}(Q,K,V)=\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right)V.
\]
语音重建增强表示为：
\[
\tilde{\mathbf{z}}_s = \operatorname{Attn}(W_q\mathbf{z}_s, W_k\mathbf{z}_c, W_v\mathbf{z}_c) + \mathbf{z}_s.
\]
认知分支同理。

#### 1.4.3 单模态解码器（重建部分解码器）
解码器将增强表示还原到原始特征空间，构造重建目标：
\[
\hat{\mathbf{x}}^{rec}_s = D_s(\tilde{\mathbf{z}}_s),\qquad
\hat{\mathbf{x}}^{rec}_c = D_c(\tilde{\mathbf{z}}_c).
\]
重建损失采用均方误差：
\[
\mathcal{L}^{rec}_s = \lVert \hat{\mathbf{x}}^{rec}_s-\hat{\mathbf{x}}_s \rVert_2^2,
\qquad
\mathcal{L}^{rec}_c = \lVert \hat{\mathbf{x}}^{rec}_c-\hat{\mathbf{x}}_c \rVert_2^2.
\]

### 1.5 融合分类器
教师模型在 `speech_cog` 设置下将两模态分类特征拼接：
\[
\mathbf{z}_{fus}=[\mathbf{z}_s;\mathbf{z}_c]\in\mathbb{R}^{1536}.
\]
之后输入 MLP 分类头：
\[
\mathbf{p}_t = \operatorname{softmax}(W_2\sigma(W_1\mathbf{z}_{fus}+b_1)+b_2).
\]

### 1.6 优化目标
教师阶段总体目标由分类损失与重建损失组成：
\[
\mathcal{L}_{teacher} = \lambda_{cls}\mathcal{L}_{cls}
+ \lambda_s\mathcal{L}^{rec}_s
+ \lambda_c\mathcal{L}^{rec}_c
+ \lambda_{itc}\mathcal{L}_{itc}.
\]
分类项可用交叉熵或 Focal Loss：
\[
\mathcal{L}_{focal}=-\alpha_t(1-p_t)^\gamma\log(p_t).
\]
其中 \(\alpha_t\) 为类别权重，\(\gamma\) 为难样本聚焦系数。

---

## 2. 单语音学生模型蒸馏

### 2.1 输入数据
学生模型仅使用语音数值特征：
\[
\mathbf{x}_s\in\mathbb{R}^{d_s},\quad y\in\{0,1\}.
\]
不再输入认知量表/MRI结构化特征。

### 2.2 整体框架
学生阶段采用“冻结教师 + 训练学生”策略：
1. 教师模型前向产生软目标与中间特征；
2. 学生模型仅语音单模态编码-解码与分类；
3. 通过响应蒸馏（logits）+ 特征蒸馏（feature）联合约束学生。

教师与学生输出分别记为 \(\mathbf{p}_t,\mathbf{p}_s\)，特征记为 \(\mathbf{z}_t,\mathbf{z}_s\)。

### 2.3 数据处理
学生数据处理继承教师语音分支流程：缺失值处理、标准化、批量采样与类别平衡策略一致，保证教师-学生输入分布一致。

### 2.4 特征提取器

#### 2.4.1 单模态编码器
学生语音编码器使用与教师同构（或轻量改造）FT-Transformer 表格编码器：
\[
\mathbf{z}_s = E^{stu}_s(\hat{\mathbf{x}}_s),\quad \mathbf{z}_s\in\mathbb{R}^{768}.
\]

#### 2.4.2 单模态解码器
学生重建分支对语音特征进行自重建：
\[
\hat{\mathbf{x}}^{stu,rec}_s = D^{stu}_s(\mathbf{z}_s),
\qquad
\mathcal{L}^{stu}_{rec}=\lVert \hat{\mathbf{x}}^{stu,rec}_s-\hat{\mathbf{x}}_s\rVert_2^2.
\]

### 2.5 分类器
学生分类头为单模态 MLP：
\[
\mathbf{p}_s=\operatorname{softmax}(W_s\mathbf{z}_s+b_s).
\]

### 2.6 优化目标
学生总损失由三部分构成：
\[
\mathcal{L}_{student} =
\beta_{cls}\mathcal{L}^{stu}_{cls}
+ \beta_{kd}\mathcal{L}_{KD}
+ \beta_{feat}\mathcal{L}_{feat}
+ \beta_{rec}\mathcal{L}^{stu}_{rec}.
\]

其中：
1. **硬标签分类损失**
\[
\mathcal{L}^{stu}_{cls}=\operatorname{CE}(\mathbf{p}_s,y).
\]
2. **响应蒸馏损失（KL）**
\[
\mathcal{L}_{KD}=T^2\,\operatorname{KL}\Big(\operatorname{softmax}(\tfrac{\mathbf{l}_t}{T})\;\big\|\;\operatorname{softmax}(\tfrac{\mathbf{l}_s}{T})\Big),
\]
其中 \(T\) 为蒸馏温度，\(\mathbf{l}_t,\mathbf{l}_s\) 为教师/学生 logits。
3. **特征蒸馏损失**
\[
\mathcal{L}_{feat}=\lVert \phi(\mathbf{z}_s)-\phi(\mathbf{z}_t)\rVert_2^2.
\]

最终通过调节 \(\beta_{cls},\beta_{kd},\beta_{feat},\beta_{rec}\) 平衡任务监督与知识迁移强度。
