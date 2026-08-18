---
title: "DeepLA-Net-Very-Deep-Local-Aggregation-Networks-for-Point-Cl"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Zeng_DeepLA-Net_Very_Deep_Local_Aggregation_Networks_for_Point_Cloud_Analysis_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 07:00:33"
---

# 论文速读：DeepLA-Net: Very Deep Local Aggregation Networks for Point Cloud Analysis

## 一句话总结
针对点云分析中局部表示方法日益复杂但性能饱和的问题，本文反其道而行，提出通过构建极深网络来提升性能；核心创新在于轻量高效的 ResLFE 模块与混合深度监督（HDS）策略，在单卡 RTX 3090 上成功训练了高达 120 个残差块的点云网络，并在 S3DIS、ScanNet v2、ScanObjectNN 和 ShapeNetPart 四项基准上同时刷新 SOTA。

## 研究问题与动机
1. **性能饱和现象：** 过去数年点云主流基准（S3DIS、ScanObjectNN、ShapeNetPart 等）的精度提升已趋近停滞（如 PointNeXt 之后多年仅获 +0.4% 左右增益），表明现有的复杂局部表示设计已触及边际收益天花板。
2. **设计路径同质化：** 现有 LANet（Local Aggregation Networks）普遍聚焦于引入更精细的局部交互机制（如向量表示、深度卷积、注意力等），导致计算冗余且难以继续加深。
3. **深度路径被忽视：** 2D 视觉早已通过残差连接走向超深网络并持续突破性能上限，而点云领域因数据结构不规则且计算昂贵，缺乏对“极深化”路线的系统性探索。
4. **两大工程阻碍：** 直接加深网络会面临（1）FLOPs 随深度线性增长导致显存/算力不可承受；（2）深层网络梯度传播困难、收敛缓慢甚至优化崩溃。

## 核心贡献（创新点）
1. **提出 ResLFE（残差局部特征提取）模块：** 通过阶段级位置编码缓存、前置线性抽象与向量特征加法融合，将单块 FLOPs 降低至基线的 10%，为构建极深网络提供计算可行性。**与已有工作的本质区别：** 放弃不断堆砌复杂局部交互的设计，转而以效率换深度，用极简模块换取百倍以上的深度空间。
2. **提出 HDS（混合深度监督）策略：** 在各阶段同时施加语义概率交叉熵监督与空间分布 MSE 监督，并引入以 epoch 倒数为指数的衰减权重，使深度监督在训练早期强引导、后期自然退火。**与已有工作的本质区别：** 传统深度监督仅依赖分类头；本文额外引入贴合点云几何特性的空间先验，专门针对点云稀疏离散带来的深层优化难题设计。
3. **构建 DeepLA-Net 系列（DeepLA-24/60/120）：** 在单张 RTX 3090 上完成训练，网络深度达现有方法 5 倍以上，并在分割与分类任务上全面超越 SOTA。**与已有工作的本质区别：** 首次在点云 LANet 上验证“极深架构+轻量模块+混合监督”组合的普适性与有效性，打破“点云只能靠复杂局部设计提升性能”的固有认知。

## 方法详解
- **演进基线：** 以 PointNet++ 为起点，沿用 PointNeXt 的数据增强（随机高度/颜色 Dropout）与优化配置，Baseline 为 63.4% mIoU / 11.3G FLOPs。
- **Stage-Level Positional Embedding：** 传统 LANet 每层实时计算相对位置 $PE = \mathcal{M}(p_i - p_i^k)$，耗时约占单层 50%。本文利用同阶段内点分辨率恒定的特性，将位置编码提升至阶段级预计算并缓存复用，FLOPs 由 92.2G 降至 44.6G，精度仅微降 0.4%。
- **Vector Feature Representation (VFR) Module：** 
  1. **Front-linear：** 在分组前对原始特征执行线性抽象 $f'_i = \mathcal{M}(f_i)$，理论降 K 倍计算（44.6G → 6.3G）；
  2. **向量特征替代分组特征：** 使用 $F^k_i = f'_i - f'^k_i$ 替代 $f_i^k$，既保留局部相对关系，又与相对位置编码语义对齐；
  3. **加法融合代替拼接：** 将 $F^k_i \oplus PE$ 改为 $F^k_i + PE$，避免通道翻倍（6.3G → 4.4G）；
  4. **Max Pooling 聚合：** 输出 $F_{out} = max(F^k_i + PE)$。
- **Modernization Structure：** 借鉴 Transformer 设计，归一化后接入 Dropout 路径抑制过拟合；仿照 Inverted Bottleneck 但取消 4× 通道扩张，保持 FFN 两层维度一致，以极小计算代价（4.4G → 9.2G）换来 +2.1% 精度。
- **Hybrid Deep Supervision (HDS)：**
  - **语义监督 $\mathcal{L}_{sem}$：** 各阶段输出经无位置编码的 VFR 模块后接线性分类头，使用加权交叉熵 $\mathcal{L}_{sem} = -\frac{1}{N}\sum_{i=0}^{N}\sum_{c=0}^{C} y_i^c \log p_i^c$。
  - **空间分布监督 $\mathcal{L}_{spa}$：** 用权重初始化为单位矩阵、无偏置的轻量线性层预测点云空间分布 $y_i$，与阶段分组相对特征 $p_i$ 计算 MSE：$\mathcal{L}_{spa} = -\frac{1}{N}\sum_{i=0}^{N}(p_i - y_i)^2$，引导早期拟合几何结构。
  - **混合损失：** $\mathcal{L}_{H} = \alpha^n \mathcal{L}_{sem} + \beta^n \mathcal{L}_{spa} + (1-\alpha^n-\beta^n)\mathcal{L}_{pred}$，其中 $n$ 为当前 epoch 的倒数，α=0.3，β=0.005，实现深度监督的动态衰减。整体贡献 +3.1% mIoU。

## 实验与结果
- **数据集与指标：** S3DIS（6-fold / Area5）、ScanNet v2（val / test）、ScanObjectNN（OA / mAcc）、ShapeNetPart（Cls.mIoU / Ins.mIoU）。
- **语义分割：**
  - S3DIS Area5：DeepLA-120 达到 **75.7%** mIoU（
