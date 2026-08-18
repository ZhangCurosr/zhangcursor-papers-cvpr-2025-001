---
title: "MaRI-Material-Retrieval-Integration-across-Domains"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_MaRI_Material_Retrieval_Integration_across_Domains_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 07:38:20"
---

# 论文速读：MaRI-Material-Retrieval-Integration-across-Domains

## 一句话总结
本文提出 MaRI 框架，通过对比学习在共享嵌入空间中直接对齐 2D 图像与标准化材质球特征，并结合自建的合成+真实世界混合数据集，实现了跨域（合成↔真实）高精度材质检索，显著优于通用视觉基线与现有 GPT-4V 驱动的两阶段检索方法。

## 研究问题与动机
1. **材质检索的域间鸿沟**：现有方法多依赖纯合成数据或通用图像检索模型，合成渲染与真实世界材质在光照、表面微观起伏、反射特性上分布差异显著，导致直接迁移时检索精度骤降。
2. **传统视觉检索不适用材质空间**：材质检索需捕捉纹理、粗糙度、金属度等物理属性，而非单纯的外观/语义相似性；直接套用 ViT、DINOv2 或 CLIP 会忽略材质的固有物理表征规律。
3. **缺乏大规模高质量配对数据**：现有材质数据集（如 ABO、MatSynth）侧重分类、重建或生成任务，缺乏“真实图像-标准化材质球”的大规模配对样本，制约了对比学习范式的落地。
4. **现有检索管线依赖 LLM 中介**：MaPa、Make-it-Real 等方法借助 GPT-4V 先粗分类再检索，计算成本高且细粒度纹理匹配易受提示词噪声干扰，泛化至未见材质时表现受限。

## 核心贡献（创新点）
1. **提出 MaRI 跨域共享嵌入框架**：借鉴 CLIP 架构思想，构建图像编码器与材质编码器并行的双分支网络，直接在视觉子域内完成跨域对齐；与已有工作的本质区别在于摒弃文本/多模态中介，面向材质物理属性专门设计对比学习空间。
2. **构建合成+真实融合的规模化配对数据集**：整合 Objaverse、AmbientCG、HDRI Haven 以及 ZeST 材质迁移技术，生成约 42.4 万组高质量配对数据；与以往单一来源数据集的本质区别在于显式弥合合成控制与真实复杂度的分布偏移。
3. **轻量级域适应微调策略**：仅微调 DINOv2 最后一个 Transformer block 而冻结其余参数，结合 InfoNCE 损失；与全量微调或 Triplet Loss 相比，在保留预训练通用特征的同时显著提升实例级检索精度。
4. **未见材质强泛化性能**：在 Trained 与 Unseen 两套独立测试集上均取得当前最优结果，尤其 Unseen 集合 T1I 达 54.0%，证明框架具备从已知材质库向未知材质分布外推的能力。

## 方法详解
1. **双编码器共享嵌入空间**：采用两个基于 DINOv2 的编码器 $E_I$（图像）与 $E_M$（材质球）。输入图像先经二值 Mask 剔除背景（$\odot$ 表示逐元素乘），材质输入为标准化的球面 PBR 渲染图。映射公式为 $\mathbf{z}_I^i = E_I(x_i \odot \text{mask}_i)$，$\mathbf{z}_M^i = E_M(m_i)$，两者共同落入共享特征空间 $\mathcal{F} \subset \mathbb{R}^d$。
2. **域自适应对比损失**：使用 InfoNCE 损失，温度系数 $\tau = 0.07$。相似度采用缩放点积 $\text{sim}(\mathbf{z}_I^i, \mathbf{z}_M^j) = \frac{\mathbf{z}_I^i \cdot \mathbf{z}_M^j}{\sqrt{d}}$。优化目标为最大化正对相似度、最小化 batch 内所有负对相似度：
   $\mathcal{L}_{\text{contrast}} = -\frac{1}{N} \sum_{i=1}^N \log \frac{\exp(\text{sim}(\mathbf{z}_I^i, \mathbf{z}_M^i)/\tau)}{\sum_{j=1}^N \exp(\text{sim}(\mathbf{z}_I^i, \mathbf{z}_M^j)/\tau)}$。
3. **检索推理机制**：测试时冻结双编码器，将查询图像与材质库中所有材质球分别编码，计算余弦相似度后执行最近邻搜索：$m^* = \arg\max_{m \in \mathcal{M}} \text{sim}(\mathbf{z}_{I_q}, \mathbf{z}_M)$，直接输出 Top-k 检索结果。
4. **数据构造流水线**：合成数据通过 Blender 将 AmbientCG 的 1605 种 PBR 材质贴合到 Objaverse 几何体，配合 712 张 HDRI 环境光与随机半球相机位姿渲染；真实数据利用 Grounded-SAM 截取前景材质区域，再经 ZeST 迁移至中性球面，形成配对样本。

## 实验与结果
1. **数据集与评测设置**：训练集 $\mathcal{D}_{\text{synthetic}}$（394,560 对）+ $\mathcal{D}_{\text{real}}$（30,000 对）；测试集划分为 Trained（约 200 种训练集同源材质，8 大类）与 Unseen（约 200 种独立来源材质）。指标涵盖 T1I、T5I、T1C、T3IoU。
2. **定量结果**：MaRI 在 Trained 数据集上 T1I=26.0%、T5I=90.0%、T1C=81.5%，全面超越 DINOv2（T1I 7.5%）、Make-it-Real（T8.5%）、MaPa（T1I 2.5%）；在 Unseen 数据集上 T1I=54.0%、T5I=89.0%，显著领先 Make-it-Real（42.5%）与 DINOv2（31.0%）。
3. **消融结论**：
   - **数据规模**：合成数据从 25% 增至 100%，Trained T1I 从 19.5% 升至 26.0%，T5I 从 55.5% 升至 90.0%，实例级检索对数据量高度敏感。
   - **架构与数据组合**：双编码器（DE）+ 合成数据（SD）+ 真实数据（RD）为最优配置；移除 DE 或任一类数据均导致 T1I 大幅下降（如仅用 SD+
