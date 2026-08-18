---
title: "TFCustom-Customized-Image-Generation-with-Time-Aware-Frequen"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_TFCustom_Customized_Image_Generation_with_Time-Aware_Frequency_Feature_Guidance_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 09:45:06"
field: "个性化图像生成"
keywords: ["subject-driven generation", "ReferenceNet", "time-aware frequency", "customized image generation", "fine-tuning-free", "DINO reward loss"]
innovations: ["提出同步ReferenceNet实现参考特征与时序去噪的动态对齐", "设计TA-FFR模块通过可学习频域滤波与时嵌入自适应注入高低频特征", "引入DINOv2奖励损失优化多对象生成中的身份一致性"]
benchmarks: ["DreamBench", "MS-Bench"]
---

# 论文速读：TFCustom-Customized-Image-Generation-with-Time-Aware-Frequen

## 一句话总结
本文提出 TFCustom，一种无需测试时微调（fine-tuning-free）的个性化图像生成框架，通过同步 ReferenceNet 与时频感知特征精炼模块（TA-FFR），在不同扩散时间步自适应注入高低频参考特征，显著提升了单对象与多对象定制的图像一致性与文本遵循能力。

## 研究问题与动机
- **ReferenceNet 时间对齐缺失**：现有 ReferenceNet 仅作为潜在层面（latent-level）特征提取器，在不同时间步提供固定强度的特征，无法匹配去噪骨干网络在不同阶段对轮廓（早期）与细节（晚期）的差异化需求。
- **频域信息利用不足**：传统方法未显式分离参考图像的高低频成分，导致细节纹理与主体结构难以协同优化。
- **多对象冲突问题**：多对象生成中参考对象间易产生身份混淆或遗漏，缺乏有效的对象级一致性约束机制。
- **细粒度细节保留困难**：现有方法在复杂纹理、文字图案等精细结构生成上表现欠佳，文本与图像对齐度有待提升。

## 核心贡献（创新点）
1. **同步 ReferenceNet 设计**：通过在每个时间步对参考图像施加与去噪过程同步的噪声注入，使 ReferenceNet 提取的特征与 noisy latent 在时序上对齐，区别于传统 ReferenceNet 仅在干净图像上提取特征的静态模式。
2. **时频感知特征精炼模块（TA-FFR）**：首次将可学习的高通/低通滤波器与时嵌入结合，实现粗到细（coarse-to-fine）的频域特征自适应注入，与已有频域方法（如 FCDiffusion、FBSdiff）的本质区别在于显式建模扩散时间步与频率成分的动态匹配关系。
3. **基于奖励模型的损失函数**：提出利用 DINOv2 计算去噪输出与参考分支预测的相似度作为奖励信号，有效缓解多对象生成中的身份冲突，区别于纯对比学习或交叉注意力方法的对象级优化策略。

## 方法详解
- **同步 ReferenceNet Guidance（Sec 3.2）**：将 ReferenceNet 从去噪网络复制而来，在每个训练时间步 t 对参考图像 $x_{\text{ref}}$ 施加 t 步噪声得到 $x_{\text{ref}}^t$，通过 ReferenceNet 提取特征 $\mathbf{F}_{\text{ref}}$ 并估计噪声，优化目标为扩散损失 $\mathcal{L}_{\text{diff}}^{\text{ref}}$，确保参考特征与去噪网络的时序对齐。
- **时频感知特征精炼模块 TA-FFR（Sec 3.3）**：使用可学习的 $5 \times 5$ 卷积核（Kirsch 算子 $\mathbf{H}_{\text{conv}}$ 与 Gaussian 算子 $\mathbf{L}_{\text{conv}}$）分离高频与低频成分：
  $$\mathbf{H}_{\text{out}} = \text{Conv}_{\mathbf{H}}(\mathbf{F}_{\text{ref}}) \cdot \mathbf{W}_{\text{H}}, \quad \mathbf{L}_{\text{out}} = \text{Conv}_{\mathbf{L}}(\mathbf{F}_{\text{ref}}) \cdot \mathbf{W}_{\text{L}}$$
  两路特征分别经过 Time-Aware Attention 与 Time-Aware FFN，通过 AdaLN 注入时间嵌入 $\mathbf{t}_{\text{emb}}$，最终合并为增强特征 $\mathbf{F}_{\text{enh}} = \mathbf{F}_{\text{high}} + \mathbf{F}_{\text{low}}$。
- **奖励模型优化（Sec 3.4）**：利用一步预测公式 $x_0' \approx (x_t - \sqrt{1-\alpha_t}\epsilon_\theta)/\sqrt{\alpha_t}$ 估计源图像，在 $t < T_0$ 时计算奖励损失 $\mathcal{L}_{\text{reward}} = \text{Dis}(\text{DINO}_{v2}(x_0), \text{DINO}_{v2}(x_0'))$，整体损失为 $\mathcal{L} = \mathcal{L}_{\text{diff}}^{\text{denoising}} + \lambda_1 \mathcal{L}_{\text{diff}}^{\text{ref}} + \lambda_2 \mathcal{L}_{\text{reward}}$。
- **特征注入机制**：通过 Shared Self-Attention 将参考特征与去噪网络特征拼接，使生成特征在每个位置可 attend 到所有参考特征，实现多尺度信息融合。

## 实验与结果
- **数据集**：训练使用 2.3M 内部图像对；测试使用 DreamBench（单对象，30 主题 × 25 prompt）与 MS-Bench（多对象，40 主题 × 1148 组合）。
- **评估指标**：CLIP-I（图像相似度）、DINO（细节保真度）、CLIP-T（文本遵循度）、M-DINO（多对象保真度）。
- **单对象 SOTA**：零样本下 CLIP-I=84.1、DINO=71.4、CLIP-T=33.9，分别超越第二名的 Kosmos-G（+1.9）、MS-Diffusion（+4.3）、MS-Diffusion（+1.8）；微调后达到 85.9/74.1/35.1。
- **多对象 SOTA**：CLIP-I=74.3、DINO=43.8、M-DINO=12.9、CLIP-T=36.9，全面超越 λ-ECLIPSE、SSR-Encoder、MS-Diffusion。
- **消融验证**：同步去噪（$\mathcal{L}_{\text{ref}}$）提升 CLIP-I 从 81.3 到 83.2；频率模块提升 CLIP-I 从 83.2 到 84.3；奖励模型进一步提升至 85.9。

## 相关工作脉络
- **IP-Adapter/ELITE**：基于预训练 CLIP 编码器的零样本特征注入方法，依赖高层抽象特征，缺乏细粒度细节控制；本文通过 ReferenceNet 提取多尺度特征弥补此不足。
- **Subject-Diffusion/SSR-Encoder/MS-Diffusion**：采用 Adapter 或 Mask 机制的多对象生成方法；本文通过同步噪声注入与频域分离实现更精确的对象级特征对齐。
- **AnimateAnyone/FlashFace**：早期 ReferenceNet 应用，将参考特征作为静态输入；本文通过时序同步使 ReferenceNet 参与动态去噪过程。
- **FCDiffusion/FBSdiff**：利用 DCT 变换增强纹理的方法；本文在特征空间而非像素空间进行频域分离，并与时间步自适应结合。
- **DreamBooth/CustomDiffusion**：测试时微调方法；本文保持 fine-tuning-free 架构，通过预训练实现同等甚至更优效果。

## 局限性与未来方向
- **计算开销**：同步噪声注入与 ReferenceNet 双分支结构增加了训练与推理的计算成本。
- **复杂场景泛化**：论文主要在标准benchmark上验证，对极端遮挡、高度艺术化风格等场景的能力未充分探索。
- **时间步截断限制**：奖励损失仅在 $t < T_0$ 时计算，限制了全时间步的细粒度优化。
- **多对象数量上限**：当前方法在少量对象（2-4个）上表现良好，大规模多对象（>6个）场景的扩展性待验证。

## 研究启发与可借鉴点
- **时频联合建模思路**：将扩散时间步与频率成分显式关联的设计可有效迁移至视频生成、3D生成等时序任务。
- **同步噪声注入范式**：ReferenceNet 与去噪网络共享时间步的训练策略可推广至其他条件生成任务（如 depth-conditioned、pose-conditioned generation）。
- **奖励模型辅助损失**：利用 DINOv2 等自监督特征计算相似度奖励的思路可应用于其他需要对象级对齐的任务（如多属性编辑、跨模态检索）。
- **可学习频域滤波器**：Kirsch/Gaussian 算子与可学习缩放因子的组合设计灵活且高效，可作为通用组件嵌入不同架构。

## 关键术语表
- **ReferenceNet**：在扩散去噪网络旁引入的参考图像编码网络，用于提取并注入参考特征以保持生成一致性。
- **TA-FFR（Time-Aware Frequency Feature Refinement）**：时频感知特征精炼模块，通过高通/低通滤波与时嵌入实现自适应频域特征注入。
- **Sync ReferenceNet**：在训练中对参考图像施加与去噪时间步同步的噪声，使参考特征与 noisy latent 对齐的优化策略。
- **DINOv2**：自监督视觉Transformer模型，本文用作奖励损失的特征提取器计算对象级相似度。
- **AdaLN（Adaptive Layer Normalization）**：DiT中提出的机制，通过时间嵌入动态调整归一化参数以实现条件注入。
- **M-DINO**：多对象DINO指标，评估多对象生成中每个独立对象的保真度。
- **Shared Self-Attention**：将参考特征与去噪特征拼接后共享的自注意力机制，实现跨模态特征交互。

## 可复现要素
- **数据集**：训练数据为内部 2.3M 图像对（未公开）；测试数据 DreamBench 与 MS-Bench 需从原论文获取或自行构建。
- **代码/权重**：论文未明确提及代码开源状态与预训练权重是否可用。
- **关键超参**：AdamW optimizer，learning rate 1e-4，100K steps，32×A100 80GB，batch size 4/GPU，$\lambda_1=0.4$，$\lambda_2=0.6$，DDIM 采样 50 步，guidance scale 7.5。
- **随机种子与配置**：论文未详细报告随机种子设置。
