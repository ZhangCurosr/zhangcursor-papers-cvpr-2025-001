---
title: "SketchFusion-Learning-Universal-Sketch-Features-through"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Koley_SketchFusion_Learning_Universal_Sketch_Features_through_Fusing_Foundation_Models_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 09:43:10"
field: "草图视觉理解"
keywords: ["草图理解", "稳定扩散", "CLIP", "基础模型融合", "特征提取", "零样本检索", "频域分析"]
innovations: ["揭示SD在草图特征提取中的频域高频偏向并系统量化", "提出CLIP视觉特征向SD UNet去噪过程的轻量1D卷积注入机制", "设计动态特征聚合网络实现跨任务自适应通用草图表示"]
benchmarks: ["Sketchy", "Tu-Berlin", "Quick Draw!", "PSC6K"]
---

# 论文速读：SketchFusion: Learning Universal Sketch Features through Fusing Foundation Models

## 一句话总结
本文系统分析了 Stable Diffusion (SD) 在抽象稀疏草图特征提取上的局限（缺乏低频语义、对草图表征力弱），并提出通过轻量级 1D 卷积将 CLIP 的语义特征注入 SD 去噪过程，结合自适应特征聚合，实现跨多种草图判别/密集预测任务的通用特征表示，无需重训练基础模型。

## 研究问题与动机
1. **草图的抽象稀疏性挑战**：手绘草图缺乏纹理与颜色，语义线索极少，传统特征提取方法难以从中提取有效表示。
2. **SD 在草图特征提取上的缺陷**：SD 虽在照片任务上表现优异，但在草图上 PCA 可视化显示特征质量显著低于照片，且其 UNet 内在偏向高频(HF)分量、压制低频(LF)语义上下文，不利于密集预测任务（分割、对应学习）。
3. **全量微调代价高昂**：在有限草图数据上微调 SD 会导致灾难性遗忘，丢失大规模预训练知识；而仅用 CLIP 特征空间精度不足。
4. **需要跨任务通用表示**：不同草图任务（检索、识别、分割、对应）对特征的粒度要求不同，手工选择特征层数繁琐，需自适应机制。

## 核心贡献（创新点）
1. **首次系统揭示 SD 在草图理解上的双重局限**：通过 PCA 可视化和 2D 傅里叶频谱分析，量化了 SD 对草图特征表征力弱及频域偏向高频的问题——区别于此前仅将 SD 作为现成 backbone 的使用方式。
2. **提出 CLIP→SD 互补特征注入机制**：用轻量可学习 1D 卷积将 CLIP 视觉特征注入 SD UNet 各层去噪过程，以"冻结基础模型+注入适配器"替代全量微调——区别于简单拼接两类特征（B-SD+CLIP）的做法。
3. **动态特征聚合网络**：训练包含 3 个 ResNet 块的聚合器 + 可学习分支权重，自动选择最优语义层次组合——区别于手工挑选固定层数的策略。
4. **首个真正通用的草图特征表示**：在检索(+3.35%)、识别(+1.06%)、分割(+29.42%)、对应学习(+21.22%)四类任务上均达 SOTA，平均提升约 39.49%——覆盖判别与密集预测任务，建立新范式。

## 方法详解
- **骨干网络冻结**：SD v2.1 UNet（含 VAE 编码器 E 和解码器 D）和 CLIP ViT-L/14 均保持 frozen。
- **SD 特征提取**：输入图像经 VAE 编码为潜变量 z₀，加噪至 timestep t 得 zₜ，送入 UNet，提取第 4 个上采样跳连层之前的 4 个上采样块特征 {fᵤⁿ}ₙ₌₁⁴，尺寸分别为 h/32, h/16, h/8, h/8。文本 prompt 使用 null（空串），因草图-照片数据集无配对文本。
- **CLIP 特征提取**：同一输入图像经 CLIP 视觉编码器得 patch 级特征 fᵥ ∈ R^(h/p × w/p × d)，取倒数第二层输出。
- **1D 卷积注入**：CLIP 特征 fᵥ 经轻量 1D 卷积层 C(·) 调整维度后，加到每个 timestep 的每个 UNet 上采样块特征上：f̂ᵤⁿ = fᵤⁿ + C(fᵥ)。
- **动态特征聚合**：对前三个增强特征 f̂ᵤ¹,²,³ 分别经 3 个 ResNet 块 A(·) 映射到 R^(60×60×d)，再按可学习权重 αₙ 加权求和得到最终特征图。所有任务共享该聚合网络。
- **训练策略**：仅训练 C(·)、A(·) 和 αₙ，SD 与 CLIP 全程冻结；不同任务用各自损失函数（triplet loss、CE loss、对比+EPPE loss、BCE loss）微调适配器。

## 实验与结果
- **数据集**：Sketchy（125类）、Sketchy-extended（+60K ImageNet照片）、Tu-Berlin（250类）、Quick, Draw!（345类）、PSC6K（1250张照片×5草图）；分割任务使用自建 Sketchy 子集（10类，5K三元组）。
- **关键基线**：B-CLIP、B-DINO、B-DINOv2、B-SD、B-SD+CLIP（简单组合）、B-Finetuning（全量微调）、SAKE、LVM、SD-PL、SketchMate、SketchGNN、SketchXAI、Self-Supervised、Sketch-a-Segmenter、ZS-Seg 等。
- **主要结果**：
  - ZS-SBIR（mAP@200）：Sketchy 0.761（↑3.35% vs SD-PL）、Tu-Berlin 0.695（↑2.21%）、Quick Draw! 0.242（↑4.78%）
  - FG-SBIR（Acc.@1）：Sketchy 33.01%（↑3.68% vs SD-PL 31.94%）
  - 草图识别（Acc.@1）：Quick Draw! 87.02%（↑0.92% vs SketchXAI 86.10%），Tu-Berlin 84.96%
  - 草图-照片对应（PCK@5）：PSC6K 70.31%（↑21.22% vs Self-Supervised 58.00%）
  - 草图分割（mIoU）：60.12%（↑29.42% vs Sketch-a-Segmenter 46.45%）
- **消融结论**：去掉聚合网络（PCK@5 降 28.65%）、去掉可学习权重（mAP@200 降 4.82%）、去掉 1D 卷积（PCK@5 降 25.62%）均显著损害性能；SD v2.1 + CLIP ViT-L/14 为最优组合；t=195 时效果最佳但方法对 t 选择较鲁棒。

## 相关工作脉络
1. **SD 作为特征提取器**（SD-PL [46]、B-SD 系列）：直接使用冻结 SD UNet 中间激活，未处理草图稀疏性；本文在此基础上揭示了 SD 的频域偏差并引入 CLIP 补偿。
2. **CLIP/DINO 单骨干**（LVM [76]、B-CLIP、B-DINOv2）：语义对齐好但空间精度不足；本文证明单一视觉语言模型无法同时满足 HF 结构和 LF 语义需求。
3. ** hybrid 模型**（如 SD+DINO [107]）：已有工作探索多模型组合，但本文首次针对草图场景并系统化分析频域偏差，提出有理论依据的注入策略而非简单拼接。
4. **草图专用模型**（SketchMate、SketchGNN、SketchXAI）：依赖手工设计或 Graph 结构，泛化到开放世界能力弱；本文基于基础模型实现更强零样本泛化。
5. **草图分割方法**（Sketch-a-Segmenter [35]、ZS-Seg [95]）：此前 mIoU 最高仅 46.45%；本文凭借 SD+CLIP 互补达到 60.12%，拉开近 14 个百分点差距。
6. **特征注入/适配器方法**（T2I-Adapter、Prompt-to-Prompt）：本文用 1D 卷积而非复杂 Attention 模块实现轻量注入，参数量极小但效果显著。

## 局限性与未来方向
- **依赖预训练基础模型**：若目标域与 SD/CLIP 预训练分布差异极大（如医学草图、专业领域），性能可能下降；未来可扩展至领域适配。
- **空 prompt 假设**：当前所有实验使用 null prompt，未利用文本条件引导特征提取；未来可探索 task-aware prompt learning。
- **仅覆盖 2D 草图**：3D 草图/VR 草图（如 Doodle Your 3D [2]）尚未验证；未来可扩展到三维形态表示学习。
- **对 timestep t 有一定敏感性**：虽有一定鲁棒性但最优值 t=195，未来可设计 timestep 自适应机制。
- **未测试极低数据场景**：few-shot/zero-shot 下 adapter 训练量有限，极端低资源情况下的泛化有待验证。

## 研究启发与可借鉴点
1. **"冻结 backbone + 轻量适配器"范式**：保持 SD/CLIP 冻结，仅训练 C(·)、A(·) 和 αₙ，避免灾难性遗忘且计算成本低，可迁移至其他foundation model微调场景。
2. **频域分析诊断模型偏差**：通过 2D 傅里叶频谱揭示 SD UNet 偏向 HF 的内在偏差，为其他生成模型的特征诊断提供了可复用的分析方法。
3. **互补融合优于简单拼接**：B-SD+CLIP 比本文方法低约 35.78%，证明特征注入的时序一致性（在 denoising 各 timestep 逐步注入）比后验拼接更重要。
4. **动态聚合替代手工选层**：可学习权重 αₙ 自动适配不同任务，这一设计思路可推广到其他多尺度特征融合场景。
5. **跨任务统一框架验证**：同一套 extractor 在检索、识别、分割、对应四类任务上均达 SOTA，证明通用特征表示的价值，可作为后续多任务联合训练的起点。

## 关键术语表
- **Stable Diffusion (SD)**：基于 latent diffusion 的文本到图像生成模型，其冻结 UNet 可作为强空间感知特征提取器。
- **CLIP**：Contrastive Language-Image Pretraining，通过大规模图文对训练的语言-视觉对齐模型，提供强语义但空间稀疏的特征。
- **高频(HF) / 低频(LF)分量**：傅里叶频谱中，HF 对应边缘/细节，LF 对应全局语义结构；SD 偏向 HF，CLIP 偏向 LF。
- **草图-照片对应学习**：在抽象草图和真实照片间建立像素级语义关键点匹配。
- **动态特征聚合**：通过可学习权重自适应融合不同语义层次的特征，无需手动选择。
- **ZS-SBIR（零样本草图图像检索）**：在训练阶段未见类别上测试草图检索照片的能力。
- **FG-SBIR（细粒度草图图像检索）**：跨类别实例级匹配，要求区分同类不同实例。
- **1D 卷积注入**：用轻量 1D conv 对 CLIP patch 特征做维度变换后加到 UNet 每层，而非用大模块。

## 可复现要素
- **数据集**：Sketchy、Sketchy-extended、Tu-Berlin、Quick Draw!、PSC6K 均为公开数据集；分割数据集为作者自标注（论文未公开代码/权重）。
- **代码/权重**：论文未明确声明代码开源（截至 CVPR 2025 截稿时间）；SD v2.1 和 CLIP ViT-L/14 为 HuggingFace 公开权重。
- **关键超参**：timestep t=195（最优）；patch 尺寸 p=16（CLIP ViT-L/14）；聚合网络 3 个 ResNet 块；分割阈值 0.47；学习率/优化器等未在摘要中详述（论文正文未给出完整超参表，需查阅 supplementary）。
- **训练细节**：SD 和 CLIP 全程 frozen；仅训练 C(·)、A(·)、αₙ；使用 null prompt。
