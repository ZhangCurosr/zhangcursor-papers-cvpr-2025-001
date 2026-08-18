---
title: "Nested-Diffusion-Models-Using-Hierarchical-Latent-Priors"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_Nested_Diffusion_Models_Using_Hierarchical_Latent_Priors_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 07:40:08"
field: "生成模型与表征学习"
keywords: ["diffusion models", "hierarchical generative models", "image generation", "semantic representation", "conditional generation", "self-supervised learning"]
innovations: ["提出分层语义引导的嵌套扩散模型框架，每层以更高抽象层级输出为条件", "通过SVD降维与高斯噪声注入控制层级间信息容量，防止模型退化", "无条件生成效能超越类别条件基线，同时保持计算高效"]
benchmarks: ["ImageNet-1K", "COCO-2014"]
---

# 论文速读：Nested-Diffusion-Models-Using-Hierarchical-Latent-Priors

## 一句话总结
本文提出**嵌套扩散模型（Nested Diffusion Models）**，通过一系列分层扩散模型从低维语义特征逐步生成到高分辨率图像，每一层以更高抽象层级的输人为条件，实现了无需外部条件（如类别标签）即可生成高质量图像的层级化生成框架。

## 研究问题与动机
- **核心问题**：现有扩散模型在生成复杂场景图像时，难以准确建模全局结构一致性与细粒度局部细节之间的平衡，尤其是无条件生成质量远低于条件生成。
- **现有方法的不足**：
  - 传统分层VAE存在高方差和表征坍缩问题，高层变量常被忽略；
  - 现有方法多依赖图像金字塔或低层纹理特征，缺乏结构化语义引导；
  - 直接使用预训练编码器的特征作为条件会导致生成模型退化为自编码器，中间层级被"绕过"。

## 核心贡献（创新点）
1. **分层语义引导的嵌套扩散框架**：提出$L$层扩散模型级联结构，从高层抽象语义逐步细化到像素级图像，每层以更高层的生成结果作为条件输入。
2. **基于预训练编码器的分层潜在变量构造**：利用ViT编码器对patch化图像提取特征，并通过SVD降维与高斯噪声注入控制各层信息容量，避免中间层级失效。
3. **无条件生成超越条件基线**：5层嵌套模型在ImageNet-1K上实现了FID从45.19降至11.05（无条件），超越了对比实验中的类别条件基线（FID=19.74）。
4. **计算高效的结构化设计**：高层使用更低维表征，使5层模型相比单模型仅增加约25%的GFlops计算开销，却带来FID 68.29%的显著改善。

## 方法详解
- **分层架构**：共$L$层，第$l$层扩散模型$D_{\theta_l}$生成$\mathbf{z}_l \in \mathbb{R}^{d_l}$，满足$d_l > d_{l+1}$，最底层$\mathbf{z}_1 = \mathbf{x}$（原始图像）。
- **非马尔可夫生成**：每层输出依赖于所有更高层级的已生成潜在变量$\mathbf{z}_{>l} = \{\mathbf{z}_m\}_{m>l}$。
- **训练目标**：优化分层扩散模型的变分下界（Eq. 2），实际训练等价于噪声预测损失（Eq. 3）。
- **分层潜在变量构造**：
  - 将图像划分为$M^2$个不重叠patch，用预训练编码器$E$提取特征；
  - 使用SVD将特征通道数压缩至$d/(L-l+1)$，使channel线性增长；
  - 高斯噪声注入：$\hat{\mathbf{z}}_l \sim \mathcal{N}(\mathbf{z}_l, \sigma_l^2\mathbf{I})$控制信息容量。
- **CFG噪声调度**：推理时通过$(t/T)^\gamma$动态衰减噪声强度，CFG场景取$\gamma=0.3$，无CFG时取$\gamma=\infty$（不加噪）。

## 实验与结果
- **数据集**：ImageNet-1K（256×256）、COCO-2014。
- **评估指标**：FID（Fréchet Inception Distance）。
- **主要结果**：
  - ImageNet-1K无条件生成：L=5时FID=11.05（w/o CFG），对比基线L=1的45.19提升68.29%；
  - ImageNet-1K条件生成：L=5时FID=9.87（w/o CFG），对比基线31.13；
  - 使用CFG后：无条件FID=5.03，条件FID=3.97，均显著优于基线；
  - COCO-2014文本到图像生成：L=3时FID=14.98（w/o CFG），优于多数参数量更大的模型；
  - 与DiT-XL/2、U-ViT等基线相比，仅需约34 GFlops即可超越更大计算量模型。
- **消融实验**：噪声水平$\sigma_2$越深模型效果越好；不同视觉编码器中CLIP、DINO表现最优。

## 相关工作脉络
- **分层VAE（HVAE）**：如Nhvae、HQ-VAE等，但存在训练不稳定、方差大等问题；本文用冻结潜变量+扩散模型替代。
- **级联扩散模型**：如Matryoshka Diffusion、Cascaded Diffusion等，聚焦多分辨率而非语义层级。
- **条件扩散模型**：Class-conditional、Text-conditional扩散模型依赖显式条件输入；本文通过分层语义隐式实现类似效果。
- **表示学习辅助生成**：DiffAE、SODA等学习低维潜在变量辅助生成；本文更进一步，构建多层语义层次结构。
- **语义特征引导生成**：如SOGA、Self-guided Diffusion等；本文强调特征空间的最近邻语义结构对扩散去噪的重要性。

## 局限性与未来方向
- **超参数搜索成本**：噪声水平$\{\sigma_l\}$需逐层搜索，虽有贪心策略但仍需调参；
- **编码器依赖性**：当前使用MoCo-v3/CLIP等预训练编码器，泛化性受限于编码器质量；
- **深层模型的性能波动**：Table 1显示L=5在部分设置下不如L=4，说明更深层级不一定单调改善；
- **扩展至高维模态**：目前仅验证了图像生成，未探索视频或3D场景。

## 研究启发与可借鉴点
1. **冻结潜在变量+扩散重建**的思路可有效规避分层VAE的训练不稳定性，可迁移至其他模态的层次化生成任务。
2. **SVD降维+高斯噪声注入**的"容量控制"机制为防止中间层级退化提供了实用方案。
3. **CFG权重递减策略**（高层强CFG、低层弱CFG）为层级生成模型的条件控制提供了新思路。
4. **非马尔可夫层级依赖**：每层依赖所有高层输出而非仅相邻层，可启发更通用的层级条件生成架构设计。

## 关键术语表
**Nested Diffusion Model**：嵌套扩散模型，指多层级联的扩散模型，每层以高层生成结果作为条件输入。
**Hierarchical Latent Prior**：分层潜变量先验，指通过预训练编码器在不同语义尺度上提取的层次化特征表示。
**Classifier-Free Guidance (CFG)**：无分类器引导，一种无需单独训练分类器即可实现条件控制的扩散模型技术。
**SVD Channel Reduction**：SVD通道压缩，通过对特征向量进行奇异值分解来降低信息维度。
**Gaussian Noise Injection**：高斯噪声注入，通过向潜变量添加高斯噪声来控制信息容量。
**Non-Markovian Generation**：非马尔可夫生成，指当前层输出依赖于所有高层潜在变量而非仅相邻层。

## 可复现要素
- **数据集**：ImageNet-1K（公开）、COCO-2014（公开）
- **代码/权重**：论文声明代码已开源（"Our code is available"），具体链接见原文
- **关键超参**：
  - $L$（层级数）：2~5
  - $\sigma_l$（噪声水平）：通过top-down贪心搜索确定
  - $\gamma$（CFG噪声衰减指数）：0.3（有CFG）、$\infty$（无CFG）
  - U-ViT-B/16作为基础架构（12 blocks，channel 768，attention heads 12）
  - MoCo-v3 ViT-B/16和CLIP ViT-B/16作为特征编码器
