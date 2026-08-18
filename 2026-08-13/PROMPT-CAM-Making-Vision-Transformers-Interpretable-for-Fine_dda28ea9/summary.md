---
title: "PROMPT-CAM-Making-Vision-Transformers-Interpretable-for-Fine"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Chowdhury_Prompt-CAM_Making_Vision_Transformers_Interpretable_for_Fine-Grained_Analysis_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 07:45:32"
---

# 论文速读：PROMPT-CAM-Making-Vision-Transformers-Interpretable-for-Fine

## 一句话总结
本文提出 PROMPT-CAM，通过在冻结的预训练 ViT 中注入可学习的类别特定 prompt token 并配合共享分类头，使模型在微调过程中被迫关注各类独有的局部判别特征，从而实现细粒度图像中 trait 的精准识别与可解释定位。

## 研究问题与动机
1. **核心问题**：现有基于预训练 ViT（如 DINO/DINOv2）的显著性图方法（Grad-CAM、attention rollout 等）在细粒度任务上往往产生模糊、粗糙的热力图，倾向于高亮整个物体轮廓，无法定位区分相似类别的关键细微特征。
2. **[CLS] token 注意力不具备类别特异性**：ViT 的 [CLS] token 虽能聚焦局部区域，但其关注的是跨类别共有的解剖结构（如鸟的头、翅、尾），无法反映特定类独有的判别性 traits。
3. **现有可解释方法工程复杂度高**：ProtoPNet、INTR 等方法需重新设计网络架构、定制损失函数或进行全量微调，难以直接复用主流预训练 ViT，落地成本高。
4. **细粒度场景的生物学/语义分析需求**：在物种分类、医学影像等领域，研究者不仅需要分类结果，更需要可视化并自动化提取具有领域意义的形态学特征（traits）与分类依据（taxonomy keys）。

## 核心贡献（创新点）
1. **提出 PROMPT-CAM 轻量级可解释框架**：仅向冻结的预训练 ViT 注入 C 个可学习 prompt 并修改预测头，利用标准交叉熵即可训练，无需额外模块或复杂损失；与 ProtoPNet/INTR 等需定制架构与全量微调的方法本质不同，实现近乎“免费午餐”式的即插即用。
2. **实现类别特定提示驱动的细粒度特征定位**：通过训练约束使真类 prompt 的注意力图集中关注其他类别图像中不存在的独有 patch，推理时直接可视化该 prompt 的多头注意力即可精确定位判别 traits；相比 INTR 依赖 encoder-decoder 检测预训练权重，PROMPT-CAM 纯编码器架构兼容性更强且热力图更锐利。
3. **设计贪心头部掩码进行特征重要性排序与误判诊断**：提出逐步模糊低贡献注意力头直至模型误判的自动化策略，以识别每类最核心的少数判别特征；同时支持并行对比真类与预测类的 attention maps，直观解释因遮挡、姿态异常或光照相似导致的误分类原因。
4. **拓展至层次化 Taxonomy 键发现**：证明 PROMPT-CAM 可在层级树的不同节点上训练独立的 prompt 集合，自动学习并定位从科到种逐级细化的形态学分类依据，为生物分类学研究提供可扩展的分析工具。

## 方法详解
- **整体流程**：给定预训练 ViT（N 层 Transformer）与 C 类数据集，ViT 输入为图像分块嵌入 $E_0 \in \mathbb{R}^{D \times M}$ 与 [CLS] token $x_0$。PROMPT-CAM 引入 C 个 D 维可学习 prompt 矩阵 $P \in \mathbb{R}^{D \times C}$。
- **两种注入变体**：
  - `PROMPT-CAM-SHALLOW`：将 $P$ 直接拼接至第一层 $L_1$ 输入，所有层 patch 特征参与计算。
  - `PROMPT-CAM-DEEP`（主方法）：借鉴 VPT-Deep，在最后一层 $L_N$ 输入前注入 C 个**类别特定** prompt，在前 $N-1$ 层注入 C 个**类别无关** prompt。特定 prompt 仅关注高层语义 $E_{N-1}$
