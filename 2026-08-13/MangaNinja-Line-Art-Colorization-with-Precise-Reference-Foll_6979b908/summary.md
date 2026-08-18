---
title: "MangaNinja-Line-Art-Colorization-with-Precise-Reference-Foll"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_MangaNinja_Line_Art_Colorization_with_Precise_Reference_Following_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 07:37:12"
field: "图像生成与编辑"
keywords: ["Line Art Colorization", "Reference-based Generation", "Visual Correspondence", "Diffusion Models", "Point Control"]
innovations: ["Patch Shuffling 模块强制学习局部语义匹配能力", "PointNet 驱动的细粒度交互式点控制方案", "基于视频帧对的自动化训练数据构建策略"]
benchmarks: ["自建 200 对图像 benchmark", "CLIP/DINO 语义相似度", "PSNR/MS-SSIM/LPIPS"]
---

# 论文速读：MangaNinja-Line-Art-Colorization-with-Precise-Reference-Foll

## 一句话总结
论文提出 MangaNinja，一种基于参考图像的线稿自动着色方法，通过 Patch Shuffling 模块增强局部语义对应学习能力，并结合 PointNet 驱动的交互点控制实现细粒度着色，在动漫线稿着色任务上显著优于现有方法。

## 研究问题与动机
- **参考图像差异大时语义错配**：现有方法依赖参考图像与线稿高度相似，当两者存在姿态、视角或细节差异时容易出现颜色混淆和语义不匹配。
- **缺乏细粒度控制能力**：已有方法只能实现全局风格迁移，无法精确还原参考图中的关键细节（如鼻影、服装图案等在线稿中难以表达的细节）。
- **多参考/跨角色场景受限**：实际应用中可能需要多张参考图协同着色，或参考图为不同角色时仍保持风格一致性，现有算法难以支持。
- **评测基准不统一**：现有工作测试集规模小、领域单一、评估指标不一致，缺乏系统化 benchmark。

## 核心贡献（创新点）
1. **Patch Shuffling 模块**：通过将参考图划分为 patch 并随机打乱，打破全局结构连贯性，迫使模型学习局部匹配能力而非依赖粗略的全局风格迁移；与已有双分支架构的本质区别在于主动破坏参考图的空间结构以激发细粒度对应学习。
2. **PointNet 驱动的交互式点控制方案**：使用用户定义的匹配点对实现细粒度颜色控制，支持复杂场景（如极端姿态、遮挡缺失区域）；与已有方法的区别在于提供零样本交互能力且无需重新训练即可控制局部颜色。
3. **基于动漫视频帧对的训练数据构建策略**：利用同一视频中连续帧的自然语义对应和姿态/光影变化自动构建训练对，无需人工标注；与已有工作的区别在于利用视频数据的内在一致性而非配对数据集。
4. **多 classifier-free guidance 策略**：分别控制参考图和点条件的引导强度，使模型在不同场景下可灵活切换自动匹配与手动指导模式。

## 方法详解
**整体架构**：双分支 U-Net 结构，包含 Reference U-Net 和 Denoising U-Net。

**Reference U-Net**：
- 使用 VAE 将参考图编码为 4 通道 latent，通过 Reference U-Net 提取多级特征
- 将参考分支和去噪分支的 self-attention 层的 Key 和 Value 拼接，通过 cross-attention 注入主分支（公式 1）：
$$\mathrm{Attn} = \mathrm{softmax}\left(\frac{Q_{\mathrm{tar}}[K_{\mathrm{tar}}, K_{\mathrm{ref}}]^\top}{\sqrt{d}}\right)[V_{\mathrm{tar}}, V_{\mathrm{ref}}]$$

**Denoising U-Net**：
- 线稿经 LineartAnimeDetector 提取后重复 3 次与噪声 latent 拼接为 8 通道输入
- 使用 CLIP encoder 替代原始文本 embedding

**Progressive Patch Shuffle（渐进式 Patch 打乱）**：
- 训练时从 2×2 逐步增加到 32×32 的 patch 数量并随机打乱参考图
- 策略迫使模型关注局部 patch 级别匹配而非全局结构

**PointNet 点控制模块**：
- 用户定义最多 24 对匹配点，编码为单通道点图
- PointNet（多层 CNN + SiLU）提取多尺度 embedding $E_{\mathrm{tar}}$ 和 $E_{\mathrm{ref}}$
- 通过加到 Query 和 Key 上实现条件注入（公式 2）：
$$Q'_{\mathrm{tar}} = Q_{\mathrm{tar}} + E_{\mathrm{tar}}, \quad K'_{\mathrm{tar}} = K_{\mathrm{tar}} + E_{\mathrm{tar}}, \quad K'_{\mathrm{ref}} = K_{\mathrm{ref}} + E_{\mathrm{ref}}$$

**多 Classifier-Free Guidance**（公式 3）：
$$\epsilon_\theta = \epsilon_\theta(z_t, \emptyset, \emptyset) + \omega_{ref}(\epsilon_\theta(z_t, c_{ref}, \emptyset) - \epsilon_\theta(z_t, \emptyset, \emptyset)) + \omega_{points}(\epsilon_\theta(z_t, c_{ref}, c_{points}) - \epsilon_\theta(z_t, c_{ref}, \emptyset))$$

**两阶段训练策略**：
- 第一阶段：同时对参考图和点信号做 condition dropping，学习无条件生成和点控制能力（180k 步）
- 第二阶段：仅训练 PointNet 模块，增强点编码能力（20k 步）

**Condition Dropping 策略**：训练时随机丢弃线稿条件，迫使模型学习从参考图重建目标图的能力。

## 实验与结果
**数据集**：使用 sakuga-42m 数据集（4200 万动漫关键帧），筛选 30 万视频片段（帧间隔设为 36，去除高相似重复帧）。

**评测基准**：构建包含 200 对图像的 benchmark，涵盖人机角色、多样表情和服饰，使用 CLIP/DINO 语义相似度、PSNR、MS-SSIM、LPIPS 及 50 对预定义匹配点的 MSE 进行评估。

**主要结果**（Table 1）：

| 方法 | DINO ↑ | CLIP ↑ | PSNR ↑ | MS-SSIM ↑ | LPIPS ↓ |
|------|--------|--------|--------|-----------|---------|
| BasicPBC | 42.64 | 79.64 | 17.58 | 0.894 | 0.33 |
| IP-Adapter | 55.42 | 82.39 | 16.19 | 0.845 | 0.30 |
| AnyDoor | 51.36 | 80.73 | 15.12 | 0.874 | 0.32 |
| AnyDoor* (带mask) | 63.79 | 83.91 | 16.24 | 0.827 | 0.27 |
| **Ours (无点)** | **68.23** | **88.34** | **20.37** | **0.962** | **0.22** |
| **Ours (full, 有点)** | **69.91** | **90.02** | **21.34** | **0.972** | **0.21** |

- 最强结果：Ours (full) 在 CLIP 相似度达到 90.02，PSNR 达 21.34，较 Best 基线 AnyDoor* 提升约 6.11 个 CLIP 分数、5.1 个 PSNR
- 消融实验（Table 2）表明 Condition Dropping 和 Progressive Patch Shuffle 均显著提升自动匹配能力，且后者贡献最大（DINO 从 64.79 提升至 67.12）

**训练配置**：8×A100-80G GPU，学习率 $10^{-3}$，每 30k 步衰减，总计 200k 步（一天完成）。

## 相关工作脉络
1. **BasicPBC [14]**：非生成式颜料桶着色方法，通过邻域采样着色，但在参考-目标差异大时表现不佳，无生成能力无法处理光影。
2. **Animediffusion [9]**：基于扩散模型的动漫人脸线稿着色，专注于特定领域，泛化能力有限。
3. **IP-Adapter [77]**：文本到图像扩散模型的图像提示适配器，具备生成能力但缺乏细粒度匹配，需手动标注 mask。
4. **AnyDoor [12]**：零样本对象级图像定制方法，使用 DINOv2 提取特征，但聚焦通用对象而非细粒度颜色对应。
5. **Paint-by-Example [76]**：使用 CLIP 编码参考图进行图像编辑，缺乏针对线稿的精细对应学习能力。
6. **Diffusart [10]**：条件扩散模型着色方法，但未解决参考图与线稿差异大的场景。

## 局限性与未来方向
- **点控制依赖用户标注**：虽然提供细粒度控制，但仍需用户手动定义匹配点，自动化程度有待提高。
- **参考图与线稿差异过大时仍有局限**：极端姿态或完全无关的参考图可能导致匹配失败。
- **视频帧对构建的假设局限**：依赖于同一角色在视频中的连续帧，对于跨角色或跨作品的着色能力仍需人工点指导。
- **计算资源需求较高**：双分支 U-Net 架构推理速度可能较慢。

## 研究启发与可借鉴点
1. **Patch Shuffle 训练策略可迁移**：打破空间结构以强制学习局部匹配的范式，可应用于其他参考驱动的生成分割/配准任务。
2. **视频帧对自动构建训练数据**：利用视频内在时序一致性和姿态变化构建配对数据，为缺乏标注数据的问题提供低成本数据方案。
3. **渐进式困难采样设计**：从粗粒度到细粒度的渐进式训练策略（2×2 → 32×32 patches）值得在其他需要精细匹配的任务中借鉴。
4. **多条件 Classifier-free Guidance**：独立控制不同条件来源的引导强度，为多模态条件生成提供了灵活的推理调控手段。
5. **两阶段训练解耦目标**：先学习通用能力再专门强化控制模块的策略，可有效避免多目标训练时的梯度冲突。

## 关键术语表
- **Line Art Colorization**：将黑白线稿填充为彩色图像的任务，广泛应用于动漫、漫画制作。
- **Reference U-Net**：专门用于编码参考图像特征的双分支架构中的一个 U-Net 子网络。
- **Patch Shuffling**：将参考图划分为小块并随机打乱，破坏全局结构以强制学习局部语义对应。
- **PointNet**：基于 CNN 的点特征编码器，将用户定义的匹配点对编码为多尺度 embedding。
- **Classifier-free Guidance**：通过随机丢弃条件来训练无条件生成能力，推理时通过调整权重控制条件影响。
- **Condition Dropping**：训练时随机丢弃输入条件（线稿/参考图/点），增强模型生成鲁棒性。
- **Sakuga-42m**：包含 4200 万动漫关键帧的大规模数据集，覆盖多样艺术风格和地域。
- **Semantic Correspondence**：识别并匹配不同图像中相关特征或点的任务，用于跨图像对齐。

## 可复现要素
- **数据集**：sakuga-42m [51]（公开可用）
- **代码**：论文声明代码和模型可在指定链接获取（论文未提供具体 URL）
- **权重**：使用 Stable Diffusion 1.5 [54] 预训练权重初始化
- **关键超参**：训练步数 200k（Stage 1: 180k, Stage 2: 20k），学习率 $10^{-3}$，衰减周期 30k 步，GPU: 8×A100-80G
- **line art 提取**：LineartAnimeDetector [84]
- **点匹配提取**：LightGlue [37]
- **点图最大对数**：24 对
- **Patch shuffle 范围**：2×2 到 32×32
