---
title: "High-Dynamic-Range-Video-Compression-A-Large-Scale-Benchmark"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Tian_High_Dynamic_Range_Video_Compression_A_Large-Scale_Benchmark_Dataset_and_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 07:10:59"
field: "高分辨率视频压缩"
keywords: ["HDR视频压缩", "学习视频压缩", "位深可扩展编码", "大规模数据集", "动态范围先验", "LVC"]
innovations: ["首个大规模HDR视频压缩数据集HDRVD2K", "首个学习位深可扩展HDR视频压缩框架LBSVC", "压缩友好的位深增强模块BEM通过动态范围先验高效预测HDR内容"]
benchmarks: ["HDRVD2K", "JVET", "HDM"]
---

# 论文速读：High-Dynamic-Range-Video-Compression-A-Large-Scale-Benchmark

## 一句话总结
本文提出了首个大规模HDR视频压缩基准数据集 HDRVD2K，并设计了首个学习位深可扩展视频压缩网络 LBSVC，通过动态范围先验引导的位深增强模块（BEM）实现高效HDR视频压缩，在 PU-SSIM 指标上较传统可扩展编码器 SHM 节省 32.5% 码率。

## 研究问题与动机
1. **缺乏大规模训练数据**：现有HDR视频数据集（如HDM、JVET）规模小、场景和运动模式有限，无法支撑学习视频压缩（LVC）模型训练，导致泛化能力不足。
2. **现有LVC方法不适配HDR**：已有LVC方法主要针对LDR视频设计，未考虑HDR视频的独特动态范围特性，直接应用于HDR视频效果次优。
3. **传统编码方法优化瓶颈**：传统HDR视频压缩算法依赖手工设计的模块和统计特性进行优化，整体系统协同优化空间有限，难以进一步突破性能。
4. **位深冗余利用不充分**：现有方法在处理HDR与tone-mapped LDR之间的位深冗余时，多采用简单线性缩放或消耗码率的映射方法，效率较低。

## 核心贡献（创新点）
1. **首个大规模HDR视频压缩数据集 HDRVD2K**：收集500个高质量HDR视频并提取2200个多样化视频片段，涵盖丰富场景和多种运动类型，填补了LVC在HDR视频训练数据的空白；与现有数据集相比，在FHLP、EHL、SI、CF等多项多样性指标上全面领先。
2. **首个学习位深可扩展HDR视频压缩框架 LBSVC**：基于HDRVD2K提出第一个针对HDR的视频学习位深可扩展压缩网络，有效挖掘多动态范围视频间的位深冗余；与现有单图层LVC方法不同，本方法采用双层可扩展架构，同时兼容LDR和HDR显示设备。
3. **压缩友好的位深增强模块 BEM**：通过微分直方图层提取动态范围先验，利用压缩后的LDR视频特征和动态范围先验预测原始HDR视频，而非仅通过时空预测减少冗余；相比传统线性缩放或位消耗映射方法，BEM能用极少比特重建高质量HDR视频。
4. **全面的实验验证与开源承诺**：在JVET和HDM两个标准测试集上，LBSVC在PU-SSIM指标上较SHM节省32.5%码率，在PU-PSNR指标上节省49.8%，代码和数据集将在GitHub开源。

## 方法详解
**整体架构**：LBSVC采用双层可扩展编码结构，包括基础层（BL）和优化层（EL）。BL压缩tone-mapped LDR视频，EL基于BL信息和BEM预测原始HDR视频。

**基础层（BL）编码**：
- 使用光流网络估计运动向量 $v_t^b$，编码后得到 $\hat{v}_t^b$
- 基于 $\hat{v}_t^b$ 和前帧传播特征 $\hat{F}_{t-1}^b$，提取运动对齐的时间上下文特征 $C_t^b$
- $f_{frame}$ 将LDR帧编码为量化潜在表示 $\hat{y}_t^b$，经熵编码和解码器重建得到 $\hat{x}_t^b$
- BL基于DCVC-HEM方法实现

**优化层（EL）编码**：
- 估计HDR帧间的运动向量 $v_{t_0}^e$，基于BL运动向量 $\hat{v}_t^b$ 压缩后得到 $\hat{v}_{t_0}^e$
- 混合时间-层上下文挖掘模块提取重构时间信息 $(\hat{v}_{t_0}^e)$ 和增强纹理信息 $(\overline{F}_t^b)$，生成混合上下文 $C_t^1, C_t^2, C_t^3$
- $\overline{y}_t^b$ 输入熵模型更有效地估计HDR帧潜在码 $y_t^e$
- 帧生成器重建HDR帧 $\hat{x}_{t_0}^e$

**位深增强模块（BEM）核心设计**：
1. **动态范围先验提取**：使用微分直方图层从HDR帧提取动态范围先验：
   $$t_j(x) = \exp\left(-\frac{(x-c_j)^2}{\sigma_j^2}\right)$$
   其中 $c_j$ 为亮度范围切片中心，$\sigma_j$ 为切片长度（$j=1..k$）
2. **先验压缩**：提取 $c_j^e$ 和 $\sigma_j^e$ 并用额外少量比特无损压缩，仅需256个float-32值即可有效表示亮度范围的bin中心和长度
3. **特征增强**：聚合HDR帧的动态范围先验 $t_j^e$ 和BL信息（$\hat{F}_t^b, \hat{y}_t^b$）提取的阈值函数 $t_j^b$，产生位深增强输出 $\overline{F}_t^b, \overline{y}_t^b$
4. **运动向量处理**：利用深度卷积块最小化增强特征 $\overline{v}_t^b$ 与 $\hat{v}_t^e$ 之间的表示冗余

**损失函数**：
- 单层训练：$L_{single} = \lambda \cdot D(x_t^n, \hat{x}_t^n) + R_t^n$，其中 $D$ 为MSE失真，$R$ 为熵模型估计的码率
- 双层联合训练：$L_{joint} = (\omega_b \cdot \lambda_b \cdot D(x_t^b, \hat{x}_t^b) + R_t^b) + (\lambda_e \cdot D(x_t^e, \hat{x}_t^e) + R_t^e)$，其中 $\omega_b = 0.5$

## 实验与结果
**数据集**：
- 训练集：HDRVD2K，包含500个HDR视频、2200个独立片段，分辨率固定为1920×1080，每片段180帧，随机选取15连续帧
- 测试集：JVET（6个序列）和HDM（10个序列），分辨率均为1920×1080

**评估基线**：
- SHM-12.4（SHVC参考软件）
- Mai11 + HM-16.20（传统backward-compatible方法）
- LSSVC（修改版，用于位深可扩展编码）
- HEM*（基于DCVC-HEM的分离编码方法）

**主要结果**（BD-Rate，负值表示码率节省）：

| 方法 | PU-PSNR (JVET) | PU-PSNR (HDM) | PU-SSIM (JVET) | PU-SSIM (HDM) |
|------|----------------|---------------|----------------|---------------|
| SHM | 0.0 | 0.0 | 0.0 | 0.0 |
| LSSVC | 8.7 | 14.7 | -69.2 | 8.2 |
| HEM* | -30.6 | -27.4 | -81.6 | -9.4 |
| **Ours** | **-45.2** | **-49.8** | **-93.9** | **-32.5** |

- **最强结果**：在HDM数据集PU-SSIM指标上较SHM节省32.5%码率，在JVET数据集PU-SSIM上节省93.9%
- **VDP2/VDP3指标**：由于这些指标评估人眼可见的完整亮度范围，与MSE损失优化方向不完全一致，但本文方法仍取得不错表现
- **模型复杂度**：参数量41.8M，FLOPs 257.35G，与其他LVC方法相当
- **消融实验**：去除BEM后，JVET/HDM的PU-PSNR BD-Rate分别增加8.51%/7.49%和8.05%/7.11%，验证BEM有效性

**训练细节**：
- BL阶段：$\lambda^b = (85, 170, 380, 840)$
- EL阶段：$\lambda^e = (85, 170, 380, 840)$
- 联合微调：使用$L_{joint}$损失函数
- 优化器：Adam，默认设置，学习率从$1 \times 10^{-4}$衰减至$1 \times 10^{-5}$
- 训练时间：约240小时（Nvidia RTX4090）

## 相关工作脉络
1. **DCVC-HEM [31]**：混合时空熵建模的神经视频压缩方法，本文将其作为BL基础实现，未对HDR特殊处理。
2. **LSSVC [7]**：学习的空间可扩展视频编码方案，本文修改其模块以实现位深可扩展HDR压缩，但原有设计面向空间分辨率而非位深。
3. **SHM [11]**： scalable HEVC (SHVC) 测试模型，传统双向兼容HDR视频编解码器，采用手工设计的层间预测和线性位深增强，作为主要对比基线。
4. **Mai11 [38]**：优化的tone curve映射方法，利用分段函数从tone-mapped视频重建HDR，本文指出其在PU指标上表现较低的原因。
5. **HDR-VDP-2/3 [42,43]**：HDR视觉质量评估指标，本文发现LVC方法在此类指标上优势减弱，因为MSE损失与PQ变换不直接优化人眼感知。
6. **PU21 [5]**：感知均匀编码方法，将HDR内容转换到感知均匀域以优化LVC损失，本文采用12-bit PQ预处理结合此编码。

## 局限性与未来方向
1. **主观质量评估有限**：实验主要依赖客观指标（PU-PSNR、PU-SSIM、VDP2/3），缺乏大规模主观用户体验测试验证。
2. **VDP指标优化不足**：由于MSE损失和PQ变换的优化方向与VDP2/VDP3指标不完全一致，在评估完整人眼可见亮度范围时表现有提升空间。
3. **数据集覆盖范围**：虽然HDRVD2K是首个大规模HDR视频压缩数据集，但仍可能存在特定场景（如极端低光、快速运动）覆盖不足的问题。
4. **实时性未充分讨论**：训练耗时240小时且未详细讨论推理速度，实际部署应用需进一步优化计算效率。
5. **未来方向**：可扩展至更高动态范围（如HDR10+）、支持更多显示设备兼容性、探索端到端联合优化策略以提升VDP指标。

## 研究启发与可借鉴点
1. **数据集构建方法论**：HDRVD2K通过专业视频平台采集、DaVinci Resolve Studio编辑、主观测试验证的流程，为其他媒体领域的大规模数据集构建提供了可复用的范式。
2. **动态范围先验的高效表示**：BEM使用微分直方图层提取仅需256个float-32值的动态范围先验，这种紧凑表示方式可有效指导特征增强，值得迁移到其他位深扩展任务。
3. **双层可扩展LVC架构设计**：将BL（LDR）和EL（HDR）解耦设计，先冻结BL训练EL再联合微调的策略，平衡了训练稳定性和最终性能，可推广至其他多分辨率/多位深压缩场景。
4. **多样性评估指标体系**：采用8项指标（FHLP、EHL、SI、CF、stdL、ALL、DR、TI）全面评估数据集多样性，并结合t-SNE可视化，为数据集质量评估提供了系统化方法。
5. **感知均匀编码结合LVC**：使用12-bit PQ转换将HDR内容映射到感知均匀域，再结合MSE损失优化，有效发挥了LVC方法的潜力，这一思路可应用于其他高比特深度内容压缩。

## 关键术语表
**HDR（High Dynamic Range）**：高动态范围，表示能呈现更宽亮度范围和更丰富色彩的视频/图像格式，通常使用10-16bit浮点或整型编码。

**LDR（Low Dynamic Range）**：低动态范围，传统视频格式，通常使用8bit整数编码每像素，动态范围有限。

**Tone Mapping**：色调映射，将HDR内容的亮度范围映射到LDR显示设备可显示的范围内，是backward-compatible压缩的核心步骤。

**Bit-depth Scalable Video Coding**：位深可扩展视频编码，同时压缩不同位深版本（如8bit LDR和16bit HDR）的视频，实现兼容性。

**DCVC-HEM**：Hybrid Spatial-Temporal Entropy Modeling for Deep Video Compression，混合时空熵建模的深度学习视频压缩方法，本文用作BL基础。

**PU-SSIM/PU-PSNR**：Perceptually Uniform encoding下的SSIM/PSNR指标，在感知均匀域计算的压缩质量评估指标。

**HDRVD2K**：本文提出的首个大规模HDR视频压缩基准数据集，包含500个HDR视频和2200个clip。

**BEM（Bit-depth Enhancement Module）**：位深增强模块，通过动态范围先验和压缩LDR信息预测HDR内容的核心组件。

## 可复现要素
- **数据集**：HDRVD2K，论文声明将在 https://github.com/sdkinda/HDR-Learned-Video-Coding 开源
- **代码**：论文声明代码和权重将在上述GitHub仓库发布
- **训练参数**：
  - BL/EL的$\lambda$值：$(85, 170, 380, 840)$
  - 优化器：Adam，默认设置
  - 学习率：从$1 \times 10^{-4}$衰减至$1 \times 10^{-5}$
  - Batch size：1-16（随训练阶段变化）
  - 训练设备：Nvidia RTX4090
  - 训练时间：约240小时
- **测试设置**： intra period=32，每序列编码96帧，低延迟公共测试条件
- **数据集规格**：分辨率1920×1080，SMPTE ST 2086/HDR10兼容，BT.2020色域，.exr格式
