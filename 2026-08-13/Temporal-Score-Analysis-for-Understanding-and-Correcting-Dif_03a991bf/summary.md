---
title: "Temporal-Score-Analysis-for-Understanding-and-Correcting-Dif"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Cao_Temporal_Score_Analysis_for_Understanding_and_Correcting_Diffusion_Artifacts_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 09:46:32"
---

# 论文速读：Temporal-Score-Analysis-for-Understanding-and-Correcting-Dif

## 一句话总结
本文提出 ASCED，一种无监督的扩散模型视觉伪影检测与实时矫正框架。通过分析去噪过程中像素级评分（score）的时序动态演变，在“突变期”识别异常评分轨迹（score traps），并在推理途中注入可控扰动打破局部锁定状态，从而在无需人工标注的情况下有效抑制伪影并保留生成多样性。

## 研究问题与动机
- **核心问题**：扩散模型即使在大尺度数据上训练，仍会生成局部纹理畸变、结构断裂等视觉伪影；现有方法多依赖后期监督检测器或最终输出的空间不确定性分析，缺乏对伪影“为何产生、何时产生”的机理认知。
- **现有方法不足1**：基于空间不确定性的方法（如 BayesDiff）仅捕捉最终输出的像素级方差 Var(x₀)，忽略了生成过程中的关键时间动态，导致伪影定位不准。
- **现有方法不足2**：监督式伪影检测器（如 PAL、SARGD）需大量人工标注，且跨域泛化能力有限，难以直接迁移至新数据分布。
- **动机**：通过细粒度分析扩散推理过程，发现生成必经“轮廓绘制(Profiling)-突变(Mutation)-细化(Refinement)”三阶段，伪影恰在突变期因局部评分动态异常而固化，这为早期无监督检测与干预提供了理论突破口。

## 核心贡献（创新点）
1. **揭示扩散生成三阶段与伪影成因**：从评分动力学视角将去噪过程解构为 Profiling、Mutation、Refinement 三个阶段，首次将视觉伪影的形成机制归因于突变期的异常评分动态锁定（score traps）。
2. **提出无监督时序检测框架 ASCED**：通过监控扩散步间归一化评分差分的时序变化，无需任何人工标注或域特定训练即可实时定位潜在伪影区域。
3. **设计轨迹感知实时矫正机制 TTC**：在推理中途（T_c）向异常区域注入可控随机扰动，打破评分锁定状态；相比直接状态替换或梯度裁剪，能更有效地保留生成多样性。
4. **提供理论分析支撑**：从概率流散度角度推导了时序加权函数 w(t) 的必要性，并证明受控扰动如何帮助被困像素重新与上下文耦合。

## 方法详解
- **核心思想**：将扩散模型的评分函数 $s_\theta(x_t, t)$ 视为像素演化的向量场。正常区域的评分随时间平稳变化；伪影区域在 Mutation 阶段会出现剧烈加速后突然停滞的异常轨迹。
- **异常检测**：定义相邻步的评分动态 $\Delta s_\theta(x_t^{i,j}, t) = s_\theta(x_t^{i,j}, t) - s_\theta(x_{t-1}^{i,j}, t-1)$。引入时序权重 $w(t) = (1-\bar{\alpha}_t)/\sqrt{\bar{\alpha}_t}$ 消除不同步长下的评分尺度衰减。设定自适应阈值 $\tau = \max\{\text{MAD}(\Delta(w\cdot s)), \text{mean}(\mathcal{S})\}$，累积标记异常区域 $\Omega^a$。
- **实时矫正（TTC）**：当去噪步数到达预设修正点 $T_c$ 时，对检测到伪影的区域执行：$\hat{x}_{T_c} = x_{T_c} \cdot \mathbb{1}_{\overline{\Omega}^a} + (\sqrt{\bar{\alpha}_{T_c}} x_0' + \sqrt{1-\bar{\alpha}_{T_c}}\epsilon) \cdot \gamma \xi \cdot \mathbb{1}_{\Omega^a}$。其中 $x_0'$ 为当前步预测的干净图像，$\gamma \xi$ 为高斯扰动。该操作打断局部评分锁定的演化轨迹，使异常像素重新进入正常的上下文耦合去噪过程。
- **理论支撑**：通过概率流散度分析证明，$w(t)$ 的引入可使评分动态在不同扩散步长下保持可比性；TTC 的扰动项为被困像素提供随机逃逸机会，且不破坏非伪影区域的协同演化。

## 实验与结果
- **数据集**：FFHQ、ImageNet、LSUN-Cat、LSUN-Horse、LSUN-Bedroom。
- **评估指标**：FID（越低越好）、Precision（保真度）、Recall（多样性）。
- **对比基线**：无监督 BayesDiff、State Replacement、Score Clipping；监督 SARGD、PAL+TTC；原始扩散模型。
- **主要结果**：
  - **生成质量**：ASCED 作为无监督方法，在全部五个数据集上均取得最佳或次佳的 FID 与 Precision，且 Recall 始终高于对比方法，说明在去除伪影的同时更好保留了多样性。
  - **跨域泛化**：在 FFHQ、LSUN-Horse、LSUN-Bedroom 上超越监督方法 SARGD 与 PAL；在 ImageNet/LSUN-Cat 上与监督方法持平（后者因针对特定域微调而占优）。
  - **检测精度**：在 FFHQ/ImageNet/LSUN 系列上的伪影检测准确率与监督模型 PAL 及零样本 LLaVA-v1.5 接近（误差范围约 1.5%~10.9%）。
  - **效率**：单图检测与修正耗时约 0.09s，较 PAL（0.79s）提速 **8.8倍**。
  - **参数分析**：最优修正时间点 $T_c^*/T \approx 0.48$，过早或过晚干预均会导致 Precision/Recall 下降。
- **最强提升**：在 LSUN-Bedroom 上 FID 从 12.96 降至 **12.53**（-0.43）；LSUN-Horse 上 FID 从 29.36 降至 **27.66**（-1.70），显著提升。

## 相关工作脉络
1. **视觉伪影检测**：早期关注超分伪影（空间/频域特征），近期转向通用生成模型的监督分类器（PAL[43]）或 LMM 零样本检测（LLaVA[22]）。本文与之区别在于**不依赖最终图像的空间特征，而是挖掘生成过程内部的时序动态**。
2. **不确定性量化**：BayesDiff[20] 利用 Last-layer Laplace Approximation 估计像素级方差，属空间后验分析。本文指出其**仅捕捉 Var(x₀) 而丢失了时间维度信息**，证明时序异常比最终不确定性更能精准定位伪影。
3. **生成质量增强**：截断技巧（BigGAN）、分类器引导（Classifier Guidance）、SARGD[49] 等通过约束采样空间或监督引导修正。本文**在无需额外引导信号的情况下，于推理途中进行轨迹级扰动**，避免了对特定域标注数据的依赖。
4. **扩散表示学习**：部分工作（如 [28, 42, 48]）观察到生成阶段的属性逐步显现，但聚焦于语义解耦或自监督表征。本文**聚焦于故障机制（artifacts）的物理/数学成因**，为可控编辑与纠错提供底层解释。

## 局限性与未来方向
- **低对比度图像误判**：当伪影与正常细节的评分动态差异不明显时，易产生漏检（False Negatives）。
- **过度修正假阳性**：若模型在 Refinement 阶段成功“合理化”了初始异常模式，时序检测可能误将其判为伪影并施加扰动，导致轻微失真。
- **未覆盖幻觉（Hallucinations）**：本文仅针对局部纹理/结构伪影，对语义级幻觉（如多余肢体）无效。
- **未来方向**：提升低对比度场景的异常区分能力
