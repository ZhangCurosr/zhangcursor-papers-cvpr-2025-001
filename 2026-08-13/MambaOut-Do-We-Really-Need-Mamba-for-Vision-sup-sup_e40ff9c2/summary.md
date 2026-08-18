---
title: "MambaOut-Do-We-Really-Need-Mamba-for-Vision-sup-sup"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Yu_MambaOut_Do_We_Really_Need_Mamba_for_Vision_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 07:35:43"
field: "视觉基础模型与架构设计"
keywords: ["Mamba", "State Space Model", "Vision Transformer", "Image Classification", "Object Detection", "Semantic Segmentation", "Gated CNN", "Visual Mamba"]
innovations: ["从 SSM 因果机制出发提出视觉任务适用性假设并实验验证", "构建去除 SSM 的 MambaOut 作为 Visual Mamba 的 Occam's razor 基线", "提出基于 FLOPs 比的长序列定量判据 r_L=L/(6D)"]
benchmarks: ["ImageNet", "COCO 2017", "ADE20K"]
---

# 论文速读：MambaOut-Do-We-Really-Need-Mamba-for-Vision

## 一句话总结
本文从机制层面剖析 Mamba（SSM）的适用条件，提出**ImageNet 图像分类不需要 SSM**的假设，并通过构建去除了 SSM 的 MambaOut（基于 Gated CNN）验证：MambaOut 在 ImageNet 上全面超越所有视觉 Mamba 模型；但在 COCO 检测和 ADE20K 分割等长序列任务上仍落后于 SOTA 视觉 Mamba，说明 SSM 在长序列视觉任务中仍有潜力。

## 研究问题与动机
- **视觉 Mamba 性能不及预期**：Vision Mamba、VMamba、LocalMamba、PlainMamba 等在图像分类任务上的表现普遍弱于同等规模的卷积/注意力模型，引发"Mamba 是否真的对视觉必要"的疑问。
- **缺乏对 SSM 机制与本质的理论分析**：现有工作将 Mamba 直接应用于视觉，但未系统讨论 SSM 的因果记忆机制与视觉任务的匹配性。
- **视觉任务特性与 SSM 假设不匹配**：视觉识别是理解任务而非生成任务，需要全可见（fully-visible）token mixing，而 SSM 的递归性质天然对应因果（causal）模式。
- **长序列 vs 短序列的区分标准缺失**：未见统一标准判断视觉任务是否属于"长序列"，本文提出基于 FLOPs 比的定量判据（$L > 6D$ 视为长序列）。

## 核心贡献（创新点）
1. **从 SSM 机制层面给出 Mamba 适用条件的理论分析**：证明 SSM 适合"长序列 + 自回归"两类特征的任务，而多数视觉任务不同时具备这两类特征；与已有工作的本质区别在于首次系统建立 SSM 递归因果特性与任务属性的映射关系。
2. **提出两个可验证的假设（Hypothesis 1 & 2）**：H1 认为 ImageNet 分类不需要 SSM；H2 认为 COCO 检测/分割和 ADE20K 语义分割因具有长序列特性，仍值得探索 SSM 潜力；本质区别是将直觉判断转化为可实验验证的科学假设。
3. **构建 MambaOut 系列模型作为 Occam's razor 基线**：基于 Gated CNN block（去掉 SSM）堆叠而成，结构简单但 ImageNet 精度全面超越视觉 Mamba；本质区别在于以"做减法"的方式揭示 SSM 在短序列分类任务中的冗余性。
4. **提供任务-架构匹配的定量判据**：通过 $r_L = L/(6D)$ 划分短/长序列，为后续视觉架构设计提供可操作的序列长度评估标准。

## 方法详解
- **Mamba Block 结构**：基于 MetaFormer 范式，核心为 Token Mixer + MLP。Mamba 的 Token Mixer 为 `SSM(σ(Conv(Z)))`，其中 SSM 由选择性状态空间模型构成，参数 $(\Delta, A, B, C)$ 经离散化变换为 $(\overline{A}, \overline{B}, \overline{C})$，递推公式为 $h_t = \overline{A}h_{t-1} + \overline{B}x_t$，$y_t = Ch_t$。
- **Gated CNN Block 结构**：MambaOut 的 Token Mixer 为 `Conv(Z)`，即仅保留深度卷积（$7\times7$ depthwise conv），去除 SSM 模块；其余结构（Norm、Gating、MLP expansion、shortcut）与 Mamba 保持一致。
- **Meta-Architecture 统一形式**：
  $$X' = \text{Norm}(X), \quad Y = (\text{TokenMixer}(X'W_1) \odot \sigma(X'W_2))W_3 + X$$
- **两阶段假设验证策略**：先在 ImageNet 分类（短序列、非自回归）上验证 H1，再在 COCO 检测/分割和 ADE20K 语义分割（长序列、非自回归）上验证 H2。
- **长序列判据**：以 Transformer 块 FLOPs 中二次项与线性项之比 $r_L = L/(6D)$ 为阈值，当 $r_L > 1$ 即 $L > 6D$ 时视为长序列任务。

## 实验与结果
**数据集**：ImageNet（分类）、COCO 2017（检测+实例分割，Mask R-CNN）、ADE20K（语义分割，UperNet）。

**ImageNet 分类（关键结果）**：
- MambaOut-Small：84.1% top-1（27M 参数，4.5G MAC），超越 VMamba-S（82.2%）、LocalVMamba-S（81.2%）、PlainMamba-L1（77.9%）等所有视觉 Mamba，且 MAC 更低。
- MambaOut-Base：84.2%（85M 参数，15.8G MAC），同样超越 VMamba-B（83.7%）、LocalVMamba-S（83.7%）等。
- **最强对比**：CAFormer-M36（简单 Conv+Attn，7 年前设计）以 85.2% 超越所有视觉 Mamba >1%，说明 SOTA 视觉 Mamba 仍有显著差距。

**COCO 检测/实例分割（关键结果）**：
- MambaOut-Tiny：APb=45.1，APm=41.0，落后 VMamba-T（APb=46.5，APm=42.1）约 1.4 APb / 1.1 APm。
- MambaOut-Small：APb=47.4，APm=42.7，落后 VMamba-S（APb=48.2，APm=43.0）。
- **结论**：MambaOut 在长序列任务上无法匹敌 SOTA 视觉 Mamba，支撑 H2。

**ADE20K 语义分割（关键结果）**：
- MambaOut-Tiny：mIoU(SS)=47.4，mIoU(MS)=48.6，落后 LocalVMamba-T（47.9/49.1）约 0.5 mIoU。
- MambaOut-Small：mIoU(SS)=49.5，mIoU(MS)=50.6，落后 LocalVMamba-S（50.0/51.0）。
- **结论**：同样支撑 H2，SSM 在长序列稠密预测任务中有增益。

## 相关工作脉络
- **Vision Mamba [112]**：首个各向同性视觉 Mamba，本文指出其在 ImageNet 上仅 76.1%（Tiny），远低于 MambaOut-Femto 的 78.9%。
- **VMamba [56]**：分层视觉 Mamba，SOTA 视觉 Mamba 之一；本文在分类任务上全面超越，但在长序列任务上仍领先 MambaOut。
- **LocalMamba [41]**：引入局部归纳偏置的视觉 Mamba；在 COCO/ADE20K 上仍优于 MambaOut，凸显 SSM 在长序列任务的价值。
- **PlainMamba [96]**：改进各向同性 Mamba；在分类上性能低于 MambaOut，说明纯 SSM 扩展并非有效路径。
- **MILA [32]（同期工作）**：从线性注意力角度解构 Mamba，本文从任务特性角度切入，两者视角互补。
- **MetaFormer [99] / MetaNeXt [101]**：本文 MambaOut 的架构继承自 MetaFormer 范式，以 Gated CNN block 替换 SSM block。

## 局限性与未来方向
- **领域泛化受限**：仅验证了三种通用视觉任务，未涵盖医疗影像、遥感等特定领域任务，也未验证 NLP 任务。
- **MambaOut 未充分扩展**：论文未进一步探索 MambaOut 在更大规模或更长序列上的性能边界（参考文献 [89] 表明其在扩展后表现良好）。
- **视觉 Mamba 自身仍有差距**：与 TransNeXt、SG-Former 等 Conv+Attn 混合模型相比，视觉 Mamba 在检测和分割任务上仍存在明显性能劣势，需进一步验证 SSM 在视觉检测中的真实有效性。
- **因果 vs 全可见的理论断言缺乏更广泛的实证**：仅通过 ViT 的消融实验展示了因果模式的性能下降，尚未在更多架构上验证。

## 研究启发与可借鉴点
1. **"做减法"驱动的科学假说验证**：通过去除核心组件（SSM）构建简化模型来检验其必要性，是高效且有说服力的研究范式，可迁移至其他"新架构是否必要"的争议性问题。
2. **任务特性-架构机制匹配的分析框架**：本文建立的双维度分析（序列长度 + token mixing 模式）可作为后续视觉架构设计的评估工具，帮助判断 SSM/Attention/Conv 的适用场景。
3. **长序列定量判据的可迁移性**：$r_L = L/(6D)$ 的阈值法为快速判断任务是否需要线性复杂度建模提供了实用启发，可推广到视频、全景图像等任务。
4. **MambaOut 作为 Visual Mamba 研究的自然基线**：后续所有视觉 Mamba 工作可先与 MambaOut 对比，以明确 SSM 的真实贡献量。
5. **Conv+SSM 组合的探索方向**：既然 SSM 在长序列任务中有价值，如何将 SSM 与局部卷积先验更好融合（类似 LocalMamba 的思路但更深）是一个值得探索的方向。

## 关键术语表
- **SSM（State Space Model）**：状态空间模型，Mamba 的核心 token mixer，具有 RNN 递归特性，通过固定大小的隐状态压缩历史信息，复杂度线性于序列长度。
- **MambaOut**：本文提出的模型系列，基于 Gated CNN block 堆叠，去除了 Mamba 中的 SSM 组件，作为视觉 Mamba 的简化基线。
- **Gated CNN**：带门控的卷积网络 block，Mamba block 的基础结构，不含 SSM；Token Mixer 仅为卷积操作。
- **Causal token mixing**：因果 token 混合模式，token t 只能聚合前面及当前位置的信息（$y_t = f(x_1,...,x_t)$），适合自回归生成任务。
- **Fully-visible token mixing**：全可见 token 混合模式，每个 token 可访问所有其他 token 的信息，适合理解类任务。
- **Visual Mamba**：将 Mamba/SSM 应用于视觉任务的模型系列，包括 Vision Mamba、VMamba、LocalMamba、PlainMamba 等。
- **MetaFormer**：统一的 vision backbone 范式，将网络分解为 Norm + Token Mixer + MLP 三个组件，ViT、ConvNeXt、PoolFormer 等均属于此类。

## 可复现要素
- **数据集**：ImageNet（公开）、COCO 2017（公开）、ADE20K（公开），均公开可获取。
- **代码**：已开源，GitHub: https://github.com/yuweihao/MambaOut
- **权重**：论文未提及是否开源预训练权重，但代码链接指向官方仓库。
- **关键超参**：
  - ImageNet：batch_size=4096，lr=0.004（按 batch_size/1024 × 10⁻³ 缩放），AdamW，输入 224²，含 RandAugment/Mixup/CutMix/Random Erasing/Color Jitter 等增强。
  - COCO：batch_size=16，lr=0.0001，AdamW，1× schedule（12 epochs），FP16，4× NVIDIA 4090。
  - ADE20K：batch_size=16，lr=0.0001，AdamW，160,000 iters，FP16，4× NVIDIA 4090，UperNet 检测头。
