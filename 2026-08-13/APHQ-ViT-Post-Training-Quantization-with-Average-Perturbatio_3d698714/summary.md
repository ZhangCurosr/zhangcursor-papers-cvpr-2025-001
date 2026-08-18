---
title: "APHQ-ViT-Post-Training-Quantization-with-Average-Perturbatio"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Wu_APHQ-ViT_Post-Training_Quantization_with_Average_Perturbation_Hessian_Based_Reconstruction_for_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 06:59:50"
field: "模型压缩与量化"
keywords: ["Post-Training Quantization", "Vision Transformer", "Hessian-based Reconstruction", "Low-bit Quantization", "MLP Reconstruction"]
innovations: ["提出平均扰动海森（APH）损失以提升块重建的重要性估计准确性", "设计 MLP 重建（MR）将 GELU 替换为 ReLU 以改善激活分布和量化性能"]
benchmarks: ["ImageNet Classification", "COCO Object Detection", "COCO Instance Segmentation"]
---

# 论文速读：APHQ-ViT: Post-Training Quantization with Average Perturbation Hessian Based Reconstruction for Vision Transformers

## 一句话总结
本文提出 **APHQ-ViT**，一种面向 Vision Transformer (ViT) 的后训练量化（PTQ）方法，通过改进的**平均扰动海森（APH）损失**精准估计输出重要性，并设计 **MLP 重建（MR）** 将 GELU 替换为 ReLU，在仅使用线性量化器的条件下显著提升了 3-bit 和 4-bit 低比特量化的精度。

## 研究问题与动机
1. **ViT 低比特 PTQ 精度骤降**：ViT 在量化部署时面临巨大挑战，尤其是 3/4-bit 等极低比特下，现有基于重建的 PTQ 方法直接套用会失效。
2. **现有海森近似不准确**：主流方法（如 BRECQ）依赖 Fisher 信息矩阵（FIM）近似 Hessian，且基于平方梯度，在 ViT 及复杂任务（检测、分割）中误差大，不如简单 MSE Loss。
3. **Post-GELU 激活量化困难**：ViT 中 GELU 激活分布极度不平衡（负值密集、正值稀疏）且激活值范围差异大（最高达 40），导致量化误差显著。

## 核心贡献（创新点）
1. **提出平均扰动海森（APH）损失**：通过数值微分直接计算对角 Hessian，避免 FIM 近似的误差，并通过对多个校准样本取平均降低梯度方差，提升重建稳定性。
2. **设计 MLP 重建（MR）方法**：将 MLP 模块中的 GELU 激活替换为 ReLU，结合截断损失（Clamp Loss）压缩激活范围，同时保留模型表达能力。
3. **统一的轻量级 PTQ 框架**：仅使用均匀量化器（Uniform Quantizer）即可在多种 ViT 架构和视觉任务上实现 SOTA 性能，无需复杂硬件友好的特殊量化器。

## 方法详解
1. **APH 损失推导**：基于泰勒展开，忽略高阶项，利用均值定理通过微小扰动（$\Delta O = 10^{-6}$）计算 Jacobian 差值来近似对角 Hessian 元素。对 batch 内所有样本计算 Hessian 并取平均，得到 $\mathcal{L}_{\mathrm{APH}} = \sum_i (\hat{O}_i - O_i)^2 \cdot \bar{H}_{i,i}$，比单次扰动海森（PH）方差更低。
2. **MLP 重建（MR）**：将原 MLP 的 GELU 替换为 ReLU，通过特征知识蒸馏最小化 ReLU 输出与原始 GELU 输出的差异。损失由两部分组成：直接蒸馏损失 $\mathcal{L}_{\mathrm{Direct}}$（加权 $L_2$）和截断蒸馏损失 $\mathcal{L}_{\mathrm{Clamp}}$（限制激活值在 p 分位数内），总损失 $\mathcal{L}_{\mathrm{Distill}} = \mathcal{L}_{\mathrm{Direct}} + 2 \cdot \mathcal{L}_{\mathrm{Clamp}}$。
3. **块重建流程**：遵循 QDrop 框架，对每个 Transformer Block 先执行 MLP 重建，再执行基于 APH 损失的权重/激活量化重建。

## 实验与结果
- **数据集与模型**：ImageNet（分类，ViT/DeiT/Swin）、COCO（检测与分割，Swin Backbone）。
- **主要结果（ImageNet, Table 1）**：
  - **4-bit**：APHQ-ViT 在 ViT-S 上达 76.07%，优于第二名 DopQ-ViT（75.69%）；在 ViT-B 上达 82.41%，与 SOTA 持平。
  - **3-bit**：APHQ-ViT 优势显著，在 DeiT-T 上达到 55.42%，较第二名 DopQ-ViT（53.82%）提升约 1.6%，**平均提升 7.21%**。
- **主要结果（COCO, Table 2）**：在 Mask R-CNN Swin-S 4-bit 检测任务上，APHQ-ViT 达到 APb 44.1，优于所有对比方法。
- **消融（Table 3-5）**：APH 损失相比 MSE/BH/PH 均有提升；MR 方法在 ViT-B 上甚至超越了全精度模型（84.84% vs 84.54%）。

## 相关工作脉络
1. **BRECQ**：CNN 上的块重建基准，使用 FIM 近似 Hessian，本文指出其在 ViT 上近似误差大且不适用于多任务。
2. **PTQ4ViT**：针对 ViT 的校准-only 方法，使用孪生均匀量化器（TUQ）处理 GELU，但在极低比特下精度较差。
3. **QDrop**：引入随机 Dropout 的重建方法，本文在其基础上引入了 APH 损失和 MR 模块以进一步提升 ViT 量化效果。
4. **I&S-ViT / DopQ-ViT**：近期 SOTA 重建方法，依赖特殊的量化器（SULQ/TanQ）或平滑优化，而本文仅用均匀量化器即达到同等或更优效果。
5. **AdaLog**：使用任意底数对数量化器处理幂律分布激活，需特定硬件支持；本文 MR 方法从激活分布层面解决问题，更具通用性。

## 局限性与未来方向
1. **MLP 重建的适用性**：将 GELU 替换为 ReLU 在深层网络中可能面临“死神经元”问题，本文仅验证了浅层 MLP 模块的有效性和安全性。
2. **超参数依赖**：截断损失中的分位数 $p$（设为 0.99）和平衡系数 $\alpha$（设为 2）需要针对特定模型调整。
3. **未来方向**：可将 APH 损失扩展至更复杂的模型架构（如 Diffusion Models），或探索与非均匀量化器的结合。

## 研究启发与可借鉴点
1. **APH 损失的数值近似思路**：通过扰动估计对角 Hessian 替代 FIM 近似，是一种更通用、更准确的损失函数设计方式，可迁移至其他模型的低比特量化研究。
2. **激活函数重构策略**：用 ReLU 替代 GELU/Swish 等非饱和激活并结合蒸馏损失重建，是解决激活分布异常、压缩量化范围的有效手段，值得在其他非线性模块中尝试。
3. **块重建中的重要性加权**：在重建损失中引入基于 Hessian 的输出重要性权重，能更精准地保护对任务关键的特征维度，对 CNN 和 ViT 均有借鉴意义。

## 关键术语表
- **Post-Training Quantization (PTQ)**：后训练量化，在预训练模型训练完成后，仅用小规模校准数据调整量化参数，无需重新训练。
- **Average Perturbation Hessian (APH)**：平均扰动海森，通过对多个校准样本进行微小扰动并平均梯度，估计的对角 Hessian 矩阵，用于量化损失的重要性加权。
- **MLP Reconstruction (MR)**：MLP 重建，将 Transformer 块中 MLP 层的 GELU 激活函数替换为 ReLU，并通过蒸馏损失优化以保留原始输出特征。
- **Block-wise Reconstruction**：块重建，将神经网络按模块（Block）划分，逐块进行量化参数的优化调整。
- **Distillation Loss**：蒸馏损失，在量化重建过程中，衡量量化后模型输出与原始全精度模型输出差异的监督信号。

## 可复现要素
- **数据集**：ImageNet（分类，公开）、COCO（检测/分割，公开）。
- **代码与权重**：代码已开源（https://github.com/GoatWu/APHQ-ViT）；预训练模型来自 timm 和 MMDetection。
- **关键超参**：校准集大小（分类 1024 张，检测 256 张）；学习率（激活 4e-5，权重 1e-3）；迭代次数（20000）；分位数 $p=0.99$；损失平衡系数 $\alpha=2$。
