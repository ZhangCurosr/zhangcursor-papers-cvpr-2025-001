---
title: "EDEN-Enhanced-Diffusion-for-High-quality-Large-motion-Video"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_EDEN_Enhanced_Diffusion_for_High-quality_Large-motion_Video_Frame_Interpolation_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 07:01:36"
field: "视频帧插值与生成模型"
keywords: ["Video Frame Interpolation", "Diffusion Model", "Large Motion", "Transformer", "Latent Representation"]
innovations: ["基于Transformer的1D token化tokenizer替代2D卷积VAE", "双路径上下文注入：时间注意力+帧差嵌入", "Diffusion Transformer适配大运动帧插值"]
benchmarks: ["DAVIS", "DAIN-HD544p", "SNU-FILM"]
---

# 论文速读：EDEN-Enhanced-Diffusion-for-High-quality-Large-motion-Video Frame Interpolation

## 一句话总结
本文提出EDEN，一种增强型扩散框架，专门解决视频帧插值中大运动场景下的模糊和时序不一致问题。通过设计基于Transformer的tokenizer、引入时间注意力机制和起始-终止帧差异嵌入，显著提升了复杂运动的重建质量。

## 研究问题与动机
- 现有扩散方法在处理大运动、非线性运动时难以生成清晰、时序一致的中间帧
- 分析发现当前扩散方法的去噪效果对生成质量影响甚微（随机噪声解码与去噪潜变量解码的感知差异很小）
- 传统光流方法在大运动场景下存在固有局限
- 需要同时从latent表示质量、扩散架构和训练范式三个层面进行系统性改进

## 核心贡献（创新点）
- **Transformer-based Tokenizer**：用基于Transformer的编码器-解码器替代传统2D卷积VAE，将中间帧压缩为紧凑的1D latent tokens，提升语义表达能力
- **Pyramid Feature Fusion Module**：通过多尺度token融合机制捕捉不同运动尺度的细粒度细节
- **Dual-stream Context Integration**：在扩散Transformer中引入时间注意力流和帧差嵌入流的条件注入机制
- **Start-end Frame Difference Embedding**：显式编码起始与终止帧之间的运动差异，引导扩散过程适应不同运动幅度
- **State-of-the-art Performance**：在DAVIS上LPIPS降低近10%，在DAIN-HD上提升8%，同时仅需2步去噪即可实现高质量生成

## 方法详解

### 1. Transformer-based Tokenizer
- 编码器/解码器均包含N个连续的Transformer block
- 每个block除标准self-attention和FFN外，还包含金字塔特征融合模块和时间注意力模块
- 使用投影层将特征降维为tokens，再升维还原

### 2. Pyramid Feature Fusion Module
- **编码器**：将输入图像划分为patches得到大尺度tokens，通过平均池化得到小尺度tokens，拼接后进入self-attention，仅保留小尺度tokens
- **解码器**：从小尺度tokens开始，插值得到大尺度tokens，拼接处理后仅保留小尺度tokens
- 公式表达：
  - $\tilde{I}_t = \text{Concat}(\tilde{I}_t^l, \tilde{I}_t^s) \in \mathbf{R}^{b \times (m+n) \times d}$
  - $\tilde{I}_t^s = \text{Self-Attention}(\tilde{I}_t)[;m:,!] \in \mathbf{R}^{b \times n \times d}$

### 3. Temporal-Attention Module
- 在编码过程中注入起始帧$I_0$和终止帧$I_1$的信息
- 公式：将$I_0, I_t^s, I_1$拼接后通过时间注意力，仅保留中间帧位置的信息
- 支持隐式空间对齐，对分辨率变化更具鲁棒性

### 4. Diffusion Architecture (DiT)
- 采用Diffusion Transformer而非U-Net，更好地处理时间依赖和token输入
- 使用adaptive Layer Normalization (adaLN)进行条件注入
- 每层后添加时间注意力层进行上下文融合

### 5. Difference Context Integration
- 计算训练集中所有起始-终止帧对的余弦相似度均值和标准差
- 对每个样本的余弦相似度进行归一化，通过MLP转换为差异嵌入
- 将该嵌入与timestep嵌入相加，作为adaLN的条件输入

### 6. 训练策略
- **Tokenizer损失**：$\mathcal{L}_{tok} = \lambda_1 \mathcal{L}_1 + \lambda_p \mathcal{L}_p + \lambda_G \mathcal{L}_G + \lambda_{kl} \mathcal{L}_{kl}$
- **扩散损失**：Velocity matching loss $\mathcal{L}_{dit} = \mathbb{E}_{t, p_t(z)} ||v_\Theta(z,t) - u_t(z)||_2^2$
- **多分辨率/多帧间隔微调**：先在256×448固定分辨率训练，再进行随机分辨率和帧间隔的微调

## 实验与结果

### 数据集
- **训练**：LAVIB（284,484 clips，1296×1296，每clip 60帧）
- **测试**：DAVIS（2,849 triplets，480×854）、DAIN-HD544p（124 triplets，544×1280）、SNU-FILM（Easy/Hard子集）

### 评估指标
- LPIPS（越低越好）和FloLPIPS（越低越好），不使用PSNR/SSIM等重建指标

### 主要结果

| 数据集 | 指标 | EDEN | 最佳基线 | 提升 |
|--------|------|------|----------|------|
| DAVIS | LPIPS | **0.0874** | LBBDM: 0.0963 | ~10%↓ |
| DAVIS | FloLPIPS | **0.1201** | LBBDM: 0.1313 | ~8%↓ |
| DAIN-HD544p | LPIPS | **0.1321** | VFIMamba: 0.1426 | 8%↓ |
| DAIN-HD544p | FloLPIPS | **0.2184** | VFIMamba: 0.2471 | ~12%↓ |
| SNU-FILM(Hard) | LPIPS | **0.0986** | LBBDM: 0.1101 | ~10%↓ |

- **推理速度**：仅需2步去噪，推理时间0.130s（DAVIS），优于多数扩散方法
- **Ablation关键发现**：
  - 时间注意力同时加入Encoder和Decoder效果最佳
  - 时间注意力在跨分辨率场景下优于交叉注意力
  - 差异嵌入带来显著提升（DAVIS LPIPS从0.0976降至0.0874）

## 相关工作脉络

- **LDMVFI**：首个基于潜扩散的视频插值方法，使用2D卷积VAE，存在大运动模糊问题
- **LBBDM**：采用Brownian Bridge扩散模型，在感知质量上有提升但仍受限于架构
- **MADIFF/VIDIM**：引入运动条件或级联扩散，但未系统解决latent表示质量
- **传统光流方法（AMT/SGM-VFI/VFIMamba）**：依赖显式光流估计，在非线性大运动中受限
- **TiTok**：证明1D序列表示在图像压缩中优于2D网格表示，本文受其启发设计tokenizer
- **DiT**：扩散Transformer架构，本文将其适配到视频插值任务并加入时序模块

## 局限性与未来方向

- **文本/精细细节模糊**：在快速变化的精细细节（如文字）上仍存在模糊问题
- **推理步数**：虽然仅需2步去噪，但仍比传统方法的推理慢
- **计算成本**：Transformer-based tokenizer和DiT的训练开销较大
- **未来方向**：可探索更高效的采样策略、结合物理运动约束、处理极端大运动场景

## 研究启发与可借鉴点

- **1D Token化思路**：将图像压缩为1D序列而非2D网格的latent representation，对图像/视频压缩任务有借鉴价值
- **双路径上下文注入**：时间注意力流+差异嵌入的设计模式可迁移到其他条件生成任务
- **多尺度特征融合在token空间的应用**：Pyramid Feature Fusion在encoder/decoder中的对称设计
- **仅需2步去噪的高效推理**：证明了在特定任务上扩散模型可以实现接近实时的推理速度
- **跨分辨率鲁棒性验证**：在训练分辨率之外的数据集上测试，验证了时间注意力相比交叉注意力的优势

## 关键术语表

**Video Frame Interpolation (VFI)**：给定起始帧和终止帧，合成中间帧的任务

**Diffusion Transformer (DiT)**：基于Transformer架构的扩散模型，相比U-Net更适合处理序列输入和长程依赖

**Temporal Attention**：在跨帧维度的注意力机制，用于融合起始帧和终止帧的上下文信息

**LPIPS**：Learned Perceptual Image Patch Similarity，基于深度特征的感知相似度指标

**FloLPIPS**：结合光流信息的LPIPS变体，专为视频插值设计的感知质量评估指标

**Patch Embedding**：将图像划分为patches并通过线性投影转换为token序列

**AdaLN**：Adaptive Layer Normalization，通过条件信息调制层归一化的参数

**Brownian Bridge**：一种连续时间扩散过程，保证终点确定性，被LBBDM用于帧插值

## 可复现要素

- **训练数据集**：LAVIB（需确认访问权限）
- **测试数据集**：DAVIS、DAIN-HD544p、SNU-FILM（公开可用）
- **代码开源状态**：论文未明确声明GitHub仓库，需查看会议 supplement
- **预训练权重**：论文未提及开源计划
- **关键超参数**：
  - 训练分辨率：256×448（初始），480×854（微调）
  - Batch size：256（初始），64（微调）
  - Optimizer：AdamW，β₁=0.9，β₂=0.99，weight decay=1e-4
  - Tokenizer：4个Transformer blocks，hidden dim=768
  - DiT：12个blocks，hidden dim=768
  - Latent dimension：16（最终选择）
  - Denoising steps：2
