---
title: "HiLoTs-High-Low-Temporal-Sensitive-Representation-Learning-f"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Lin_HiLoTs_High-Low_Temporal_Sensitive_Representation_Learning_for_Semi-Supervised_LiDAR_Segmentation_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 07:09:46"
field: "自动驾驶LiDAR语义分割"
keywords: ["semi-supervised LiDAR segmentation", "temporal sensitivity", "point cloud", "Mean Teacher", "cylindrical voxelization", "cross-attention"]
innovations: ["提出距离感知的双路高/低时间敏感性流（HTSF/LTSF）建模远近端时序差异", "设计Multi-Voxel Aggregation降低Transformer时序注意力计算复杂度", "纯LiDAR半监督方法达到与LiDAR+Camera多模态方法相近的性能水平"]
benchmarks: ["SemanticKITTI", "nuScenes"]
---

# 论文速读：HiLoTs-High-Low-Temporal-Sensitive-Representation-Learning-f

## 一句话总结
本文提出 **HiLoTs**，一种面向自动驾驶半监督 LiDAR 点云语义分割的新方法，通过捕捉不同距离目标的**高/低时间敏感性**差异，设计 HTSF（远端高敏感流）与 LTSF（近端低敏感流），结合交叉注意力融合与 Mean Teacher 架构，在仅用少量标注数据下显著提升了分割性能。

---

## 研究问题与动机
1. **标注成本高**：现有 LiDAR 语义分割方法多为全监督，需大量逐点标注，难以扩展到新场景。
2. **现有半监督方法偏重空间、忽视时序**：主流 SSL 方法（如 GPC、LaserMix、DDSemi）聚焦点云空间分布，对室外点云的时序特性利用不足。
3. **已有时序方法仅利用短邻帧**：如 Shi et al. [33] 仅基于两帧做时序匹配，无法编码更丰富的时序变化。
4. **关键观察未被利用**：驾驶经验表明，近处物体（道路、车辆）类别和形状稳定（低时间敏感），远处物体（植被、建筑、行人）变化频繁（高时间敏感），LiDAR 点云亦呈现此规律（图1），但现有工作未系统建模该现象。

---

## 核心贡献（创新点）
1. **发现并利用距离-时间敏感性关联**：首次将"近处稳定、远处多变"这一驾驶场景先验形式化为高/低时间敏感性表示学习，与以往仅关注空间或短邻帧时序的方法有本质区别。
2. **HiLoTs Embedding Unit (HEU)**：设计双路流（HTSF + LTSF）分别处理远/近端体素，并通过交叉注意力实现高/低敏感性特征交互，区别于简单的多帧拼接或单流特征提取。
3. **Multi-Voxel Aggregation (MVA)**：将邻近体素聚合成超体素以降低 Transformer 注意力的 $O(n^2)$ 计算复杂度，在减少 token 数的同时保留空间连贯语义，为高效时序建模提供新思路。
4. **仅 LiDAR 达到 multimodal 水平**：HiLoTs 在 SemanticKITTI/nuScenes 上以纯 LiDAR 输入实现了与 LiDAR+Camera 多模态方法（如 LaserMix++）相近的 mIoU，在低标注率下优势尤为突出。

---

## 方法详解
1. **圆柱体素化（Cylindrical Voxelization）**：将笛卡尔坐标 $(x,y,z)$ 转为 $(\rho, \theta, z)$，按距离递增设置体素尺寸以平衡远近密度；保留最近点与反射强度 $r_\epsilon$；输出 $\boldsymbol{x}_f \in \mathbb{R}^{R \times \Theta \times H \times 4}$，经 3D ResNet50 提取初始特征 $x_f \in \mathbb{R}^{M \times d}$（$d=256$）。

2. **Multi-Voxel Aggregation (MVA)**：
   - **空间聚合**：$F' = \mathrm{MLP}_{\theta_1}(\mathrm{NNGroup}(F, m))$，将 $M$ 个体素聚为 $m$ 个超体素（$m < M$）。
   - **时序融合**：$V = \mathrm{MLP}_{\theta_2}(\mathrm{AvgPool}(F'))$，沿 $t$ 帧取平均后过 MLP。

3. **HTSF（高时间敏感性流）**：处理最远 70% 距离范围内的超体素，采用多层 self-attention：
   $$V_i = \mathrm{Softmax}\!\left(\frac{\mathrm{dot}(V_i W_Q, V_i W_K)}{\sqrt{d_k}}\right) V_i W_V, \quad V_{i+1} = \mathrm{MLP}_i(V_i)$$
   共 $N=6$ 层 Transformer Encoder。

4. **LTSF（低时间敏感性流）**：处理近端体素，结构与 HTSF 类似，但其 self-attention map 作为 **Key/Value** 与 LTSF 作为 **Query** 进行**交叉注意力**，实现远/近特征交互。

5. **特征融合**：
   $$S = \alpha_1 \cdot \mathrm{Bottleneck}_{\mathrm{En}}(x_f) + \alpha_2 \cdot \mathrm{HEU}(F)$$
   其中 $\alpha_1, \alpha_2$ 为可学习参数；融合后送入 Minkowski-UNet Decoder 输出体素级特征，再经 RefineNet 还原为点级预测。

6. **半监督训练（Mean Teacher）**：
   - 学生网络处理标注帧，教师网络处理未标注帧。
   - 监督损失：$L_{sup} = \mathrm{FocalLoss}(\hat{y}_s, y)$
   - 一致性损失：$L_{con} = \|\hat{y}_s - \hat{y}_t\|_2$
   - 总损失：$L = \alpha L_{sup} + \beta L_{con}$（$\alpha=\beta=1$）
   - 教师权重EMA更新：$W'_t = \gamma W'_{t-1} + (1-\gamma)W_t$，最优 $\gamma=0.99$
   - 时序窗口 $t=5$（中心帧±2帧），Transformer 层数 $N=6$。

---

## 实验与结果
- **数据集**：SemanticKITTI（19类，序列00-07/09-10训练，08验证）与 nuScenes（16类，27287/5850场景）。
- **评估**：标注比例 {1%, 10%, 20%, 50%}，指标为 mIoU。
- **主要结果（SemanticKITTI 10%）**：HiLoTs 达 **65.7 mIoU**，超越 DDSemi（65.1）、FRNet（64.8）、LaserMix（60.0）；在 1% 标注率下达 **58.6 mIoU**，远超 Cylinder3D（45.4）和 MarS3D（44.5）。
- **nuScenes 10%**：HiLoTs 达 **72.2 mIoU**，超越 DDSemi（70.2）和 FRNet（72.2，持平）；在 1% 标注率下达 **58.7 mIoU**。
- **多模态对比**：HiLoTs（纯LiDAR）在 SemanticKITTI 10% 上达 65.7，略低于 LaserMix++（LiDAR+Camera，67.5），但在 1% 标注率下（58.6 vs 63.2）差距缩小，整体性能与多模态方法**相当**。
- **类别分析**：HiLoTs 在 "parking" 类上表现优秀（60.2），"terrain" 类相对较弱（52.3），说明远距离物体识别能力提升显著；"other-ground""bicycle""traffic-sign" 仍是挑战。
- **鲁棒性**：在 SemanticKITTI-C 和 nuScenes-C 的风雪、Beam丢失、Echo干扰等扰动下，与 CENet/FRNet 持平。

---

## 相关工作脉络
1. **GPC [15]**：基于伪标签引导的对比学习 SSL 方法，聚焦空间特征，未建模时序动态——HiLoTs 在时序维度做了本质补充。
2. **LaserMix [18]**：分析点云分布先验并进行 labeled/unlabeled 混合增强，仍为纯空间方法——HiLoTs 引入距离感知的时序敏感性流。
3. **DDSemi [23]**：密度引导对比学习解决 SSL 离群点问题，但同样侧重空间——HiLoTs 以 HTSF/LTSF 双路设计捕捉远近时空差异。
4. **MarS3D [29]**：运动感知多帧融合方法，将多帧直接输入编码器——HiLoTs 在特征级别区分高/低敏感流，计算更高效。
5. **Shi et al. [33]**：基于两帧时序匹配做弱监督分割——HiLoTs 利用 $t=5$ 帧并引入距离分解的时序建模，编码能力更强。
6. **Cylinder3D [46] / RangeViT [1]**：全监督体素化基准——HiLoTs 在其基础上叠加半监督时序敏感性学习，实现小样本下大幅超越。

---

## 局限性与未来方向
1. **领域局限性**：方法专为自动驾驶连续帧 LiDAR 序列设计，不适用于静态/姿态固定的通用点云分割任务（如 ShapeNet、ScanNet）。
2. **远距离小目标识别仍有瓶颈**：ablation 显示 terrain 类表现较弱，且 Fig.4 可视化表明复杂远处区域仍存在误分。
3. **未探索更多时序窗口长度**：实验固定 $t=5$，未系统研究更长时序对远距离建模的影响。
4. **与多模态融合结合空间**：虽达到 multimodal 水平，但未探索与相机特征的深度融合，未来可进一步验证 HTSF/LTSF 在跨模态场景的迁移性。

---

## 研究启发与可借鉴点
1. **距离-时间敏感性先验的可迁移性**：该观察可推广至其他时序点云任务（如追踪、预测），为"不同距离目标建模不同时序感受野"提供设计范式。
2. **Multi-Voxel Aggregation 的效率设计**：超体素聚合+时序平均池化的轻量方案，可用于其他基于 Transformer 的点云时序模型中降低计算开销。
3. **交叉注意力融合策略**：以 HTSF 作为 K/V、LTSF 作为 Q 的交叉注意力设计，使稳定近端特征"锚定"动态远端特征，可在多尺度/多模态特征融合中复用。
4. **Mean Teacher + Focal Loss 的组合**：适用于类别极度不平衡的稀疏点云场景，可作为 SSL 点云分割的通用训练配方。
5. **创新机会**：将 HiLoTs 的 HTSF/LTSF 思想与深度估计/法线估计结合，或迁移至视频/雷达点云序列的时序分割任务中，具有较高研究价值。

---

## 关键术语表
- **HiLoTs**：High-Low Temporal Sensitive 的缩写，本文提出的高/低时间敏感性表示学习方法。
- **HEU（HiLoTs Embedding Unit）**：包含 HTSF 和 LTSF 双路流的嵌入单元，是实现距离感知时序建模的核心模块。
- **HTSF（High Temporal Sensitivity Flow）**：处理远端（最远70%范围）体素的 self-attention 流，捕捉类别和形状高频变化的区域。
- **LTSF（Low Temporal Sensitivity Flow）**：处理近端体素的流，结合来自 HTSF 的交叉注意力实现远近特征交互。
- **MVA（Multi-Voxel Aggregation）**：将邻近体素聚合成超体素以降低 Transformer 计算复杂度，包含空间聚合与时序平均两步。
- **Mean Teacher**：半监督学习框架，通过 EMA 更新教师网络权重，使教师逐步"长成"学生，用于对齐有标/无标分支的一致性。
- **Cylindrical Voxelization**：将 LiDAR 点云转换为圆柱坐标系下的规则体素网格，自适应远近密度差异。
- **Focal Loss**：用于缓解点云语义分割中类别极度不平衡问题的损失函数。

---

## 可复现要素
- **数据集**：SemanticKITTI 和 nuScenes 均为公开数据集。
- **代码**：已开源，GitHub: https://github.com/rdlin118/HiLoTs
- **权重**：论文未明确声明预训练权重开源情况。
- **关键超参**：标注比例 {1%, 10%, 20%, 50%}；时序窗口 $t=5$；Transformer 层数 $N=6$；EMA 比率 $\gamma=0.99$（最优）；最大范围 $(0,50)$m，方位角 $(-\pi,\pi)$，高度 $(-4,2)$m；体素分辨率 $(240,180,20)$；初始学习率 $1\mathrm{e}{-3}$；Batch size 16（4×RTX 3090）；训练 50,000 步；$\alpha=\beta=1$。
