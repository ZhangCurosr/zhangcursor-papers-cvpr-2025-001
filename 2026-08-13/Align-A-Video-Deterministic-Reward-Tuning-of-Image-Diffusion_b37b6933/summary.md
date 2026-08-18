---
title: "Align-A-Video-Deterministic-Reward-Tuning-of-Image-Diffusion"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_Align-A-Video_Deterministic_Reward_Tuning_of_Image_Diffusion_Models_for_Consistent_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:55:50"
field: "视频生成与编辑"
keywords: ["视频编辑", "奖励微调", "扩散模型", "时序一致性", "确定性优化", "跨帧注意力"]
innovations: ["确定性奖励微调策略：固定初始噪声实现分钟级稳定优化", "跨帧特征传播机制：锚帧优化特征经注意力传播至全视频"]
benchmarks: ["V2VBench", "DOVER Score", "Pick Score", "CLIP Score", "ViCLIP Score"]
---

# 论文速读：Align-A-Video: Deterministic Reward Tuning of Image Diffusion Models for Consistent Video Editing

## 一句话总结
提出 Align-A-Video，一种通过**确定性奖励微调**预训练图像扩散模型并**跨帧传播**优化特征的视频编辑方法，在分钟级时间内实现文本对齐、视觉质量与时间一致性的平衡。

## 研究问题与动机
1. 零样本视频编辑方法因缺乏针对任务训练的奖励信号，常无法充分遵循用户文本指令，生成视频视觉吸引力不足且语义偏离
2. 现有 SOTA 方法（如 Flatten、TokenFlow）通过 DDIM inversion、深度引导、光流约束等**固定先验**强制保持时间一致性，却过度保留源视频像素信息，导致语义保真度下降、编辑灵活性受限
3. 直接将逐帧图像奖励应用于视频编辑会破坏时间一致性——帧级优化与视频整体时序连贯性之间存在内在矛盾（InstructVideo、VADER 采用稀疏帧采样缓解，但成本高）
4. 训练专用视频扩散模型需要海量数据与算力，而利用预训练图像扩散模型进行零样本/单样本迁移更为高效

## 核心贡献（创新点）
1. **确定性奖励微调策略**：固定初始噪声并复用反演特征，将奖励调优收敛稳定性大幅提升；与 DDPO/DRaFT 等在多 prompt 上优化噪声分布不同，本文聚焦单 prompt 确定性优化，训练仅需数分钟
2. **跨帧特征传播机制**：以锚帧为优化核心，通过改编版空间-时序注意力实现跨帧 attention，使优化结果可靠传播至其余帧；与 TokenFlow/Flatten 的直接像素注入不同，本文传播的是经奖励调优后的语义特征而非原始结构
3. **单样本/单帧调优范式**：仅对锚帧进行奖励微调，避免对整个视频序列优化；与 InstructVideo/VADER 的全视频或稀疏帧奖励优化相比，训练成本极低且精度更高
4. **端到端视频编辑 Pipeline**：整合奖励微调与特征传播两个模块，在 V2VBench 上同时超越 TokenFlow、Flatten（视觉质量/语义保真）、ControlVideo（时序一致）等各类基线

## 方法详解
**1. 确定性奖励微调（Deterministic Reward Tuning）**
- 对输入视频选取一个锚帧 $I_{anc}$，应用 DDIM inversion 获得**确定性初始噪声 z**，在所有训练迭代中复用该 z，消除噪声随机性引入的梯度波动
- 提取 DDIM inversion 最后 k 步的反演特征，在去噪过程前 k 步重新注入（沿用 PnP-diffusion 策略），增强输出结构确定性
- 对剩余去噪步骤，每轮随机选取截断点 $l \sim \text{Uniform}(\{k, \ldots, T\})$，在 l 处截断反向传播，缓解短依赖偏差
- 损失函数简化为 $\mathcal{L}_{\mathcal{R}}(\theta) = -\mathcal{R}(x, \mathcal{P})$，使用 HPSv2 作为奖励模型，结合 LoRA 高效微调注意力模块投影矩阵

**2. 锚帧特征跨帧传播（Feature Propagation Across Frames）**
- 生成过程中采样一组关键帧 $\{\mathcal{K}^i\}_{i \in \kappa}$，构建跨帧注意力：keyframe 的 Query 来自当前帧，Key/Value 来自锚帧，公式为 $\phi(\mathcal{K}^i) = \text{softmax}\left(\frac{Q_{curr} K_{anc}^T}{\sqrt{d}}\right) V_{anc}$
- 对非关键帧 s，根据距前后最近关键帧的距离计算权重 $w_s = |s - p^-| / (|s - p^-| + |s - p^+|)$，加权融合相邻关键帧的传播特征：$F_s = w_s \cdot \phi(\mathcal{K}^{p^-}) + (1 - w_s) \cdot \phi(\mathcal{K}^{p^+})$

**3. 推理流程**
- 输入视频 $\mathcal{I}$ 与编辑 prompt $\mathcal{P}$ → 对锚帧执行确定性奖励微调 → 在逆向采样中注入锚帧优化特征 → 通过跨帧注意力传播至其余帧 → 输出编辑后视频 $\mathcal{J}$

## 实验与结果
- **数据集**：V2VBench（50 个标准视频，4 类编辑任务，每视频 3 个 prompt）
- **基线方法**：Tune-A-Video、TokenFlow、Flatten、Text2Video-Zero、ControlVideo（5 种）
- **评估指标**：视觉质量（Aesthetic S.、Pick S.、DOVER S.）、语义保真（CLIP S.、ViCLIP S.）、时间一致性（CLIP C.、DINO C.、EPE）
- **核心结果**：
  - DOVER 得分 **0.761**，显著优于第二名的 ControlVideo（0.708），提升 **+7.2%**
  - Pick Score **21.847**，优于所有基线（第二名 ControlVideo 21.039，+3.8%）
  - CLIP Score 0.284、ViCLIP Score 0.265，语义保真与最优基线持平或略优
  - CLIP Consistency 0.955、DINO Consistency 0.944，时序一致性接近 TokenFlow（0.957/0.956），显著优于 Tune-A-Video（0.873/0.824）
- **结论**：本方法在视觉质量上全面领先，语义保真与时间一致性达到最佳平衡，验证了奖励微调+特征传播的有效性

## 相关工作脉络
1. **Tune-A-Video**：单样本时序注意力扩展方法，依赖学习时序先验，存在复杂运动下时序不一致问题；本文在其基础上引入奖励微调直接优化文本对齐，无需从源视频学习运动先验
2. **TokenFlow / Flatten**：通过 diffusion feature 空间对齐或光流约束保持时序一致性；本文指出这类方法过度依赖源视频像素信息导致语义偏离，改用奖励微调让注意力特征与文本指令对齐
3. **InstructVideo / VADER**：面向 T2V 模型的奖励微调工作，使用梯度裁剪和 checkpointing 技术；本文聚焦 V2V 场景，提出确定性单帧优化避免视频时序退化，训练效率从小时级降至分钟级
4. **DDPO / DRaFT / Align-Prop**：图像领域的奖励微调方法；本文将其适配至视频编辑场景，核心区别是解决了逐帧优化导致时序不一致的根本难题
5. **ControlVideo**：基于 ControlNet 的深度引导方法，时序一致性较好但在源视频运动对齐上不足；本文在保持相似时序表现的同时显著提升视觉质量和文本对齐

## 局限性与未来方向
1. **单锚帧限制**：当前方法仅优化单个锚帧，对于长视频或多主体编辑场景，单一锚帧可能无法覆盖全片语义变化
2. **奖励模型依赖**：使用 HPSv2 等图像级奖励模型，视频级奖励建模（如 DOVER 直接优化）尚未探索
3. **缺乏时序运动奖励**：当前奖励仅评估单帧视觉质量和文本对齐，未对时序平滑性/运动一致性设计专项奖励函数
4. **通用性验证有限**：仅在 V2VBench 上评估，其他视频编辑基准（如 VBench 完整版、自定义场景）泛化性待验证

## 研究启发与可借鉴点
1. **确定性噪声固定策略**可迁移至其他奖励微调场景：固定初始噪声消除梯度方差，是提升 diffusion 模型 reward tuning 稳定性的通用技巧
2. **跨帧注意力传播机制**可推广至其他零样本视频编辑任务：以关键帧为锚点传播优化特征，避免全序列计算的性价比思路
3. **单帧优化+传播范式**适用于资源受限的微调任务：相比训练整个视频，先优化代表性帧再传播可大幅降低显存和时间开销
4. **混合截断梯度策略**（randomized truncation at step l）可有效缓解短序列优化中的短期依赖偏差，值得在其他生成模型微调中尝试
5. 可结合本团队视频理解/生成方向：将 ViCLIP 等多模态一致性指标作为奖励函数，进一步改进语义保真评估

## 关键术语表
- **Deterministic Reward Tuning**：固定初始噪声并复用 DDIM 反演特征，使奖励梯度优化在确定性条件下收敛的训练策略
- **Anchor Frame**：视频中被选中用于奖励微调的关键帧，其优化特征通过跨帧注意力传播至其他帧
- **Cross-Frame Attention**：将自注意力扩展为关键帧间共享 Q/K/V 的注意力机制，使关键帧能查询锚帧特征以实现一致性传播
- **V2VBench**：专为视频编辑任务设计的综合评测基准，包含 50 个视频和 4 类编辑任务
- **HPSv2**：Human Preference Score v2，用于评估文本-图像对生成质量的视觉偏好奖励模型
- **DOVER Score**：基于大规模人工评分视频数据集训练的端到端视频质量评估指标，衡量伪影、模糊等伪影程度
- **DDIM Inversion**：DDIM 确定性采样逆过程，将真实图像转为可复用的初始噪声，用于保持生成结构一致性

## 可复现要素
- **数据集**：V2VBench（论文声明公开）
- **代码/权重**：论文未提及代码开源声明；使用 Stable Diffusion v1.5 预训练权重和 HPSv2 奖励模型（均为公开）
- **关键超参**：DDIM 总步数 T（未明示具体值）、反演特征复用步数 k、随机截断点范围 $l \sim \text{Uniform}(\{k, \ldots, T\})$、LoRA 秩 r（未明示）、关键帧采样策略（未明示）
