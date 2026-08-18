---
title: "Binarized-Mamba-Transformer-for-Lightweight-Quad-Bayer-Hybri"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Zhou_Binarized_Mamba-Transformer_for_Lightweight_Quad_Bayer_HybridEVS_Demosaicing_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:55:58"
field: "计算摄影与边缘AI"
keywords: ["去马赛克", "二值神经网络", "Mamba", "HybridEVS", "Quad Bayer", "轻量级视觉", "边缘部署"]
innovations: ["首次提出二值化Mamba（Bi-Mamba），对投影层二值化而保留Selective Scan全精度", "将全局视觉嵌入融入控制矩阵B以增强长程依赖建模", "首个面向Quad Bayer HybridEVS去马赛克的二值化网络框架"]
benchmarks: ["MIPI Workshop 2024", "Kodak", "BSD100", "Urban100", "REDS", "Vid4"]
---

# 论文速读：Binarized-Mamba-Transformer-for-Lightweight-Quad-Bayer-Hybri

## 一句话总结
本文提出了首个面向Quad Bayer HybridEVS相机去马赛克的轻量级二值化网络BMTNet，通过二值化Mamba的非关键投影层、保持核心Selective Scan全精度，并结合全局视觉嵌入，在参数量仅1.28M的情况下实现了超越其他BNN、接近全精度方法的去马赛克性能。

## 研究问题与动机
- **HybridEVS去马赛克的特殊性**：Quad Bayer CFA色彩排列复杂，且事件像素导致额外颜色损失，传统及现有深度学习方法难以在边缘设备上高效部署。
- **全局建模与计算效率的矛盾**：Transformer等架构能有效捕获长距离依赖，但复杂度极高，不适合移动设备实时ISP部署。
- **Mamba的二值化空白**：Mamba虽以线性复杂度建模长程依赖，但其投影层仍带来较大计算负担；现有二值化工作集中于CNN/Transformer，Mamba的二值化研究尚未开展。
- **BNN在去马赛克中的探索不足**：二值神经网络（BNN）是极致压缩手段，但其在去马赛克任务中的应用仍处于空白，亟待探索。

## 核心贡献（创新点）
- **首个二值Mamba-Transformer去马赛克框架**：首次将Mamba引入二值网络并进行系统化二值化设计，填补了Mamba二值化的研究空白。
- **Bi-Mamba：选择性二值化策略**：提出对非关键投影（Linear/Conv）进行全二值化，而保留核心Selective Scan全精度计算，在Mamba块上实现79%参数量减少和88%运算量降低。
- **全局视觉嵌入增强全局建模**：设计二值化全局视觉编码器，并将全局视觉向量嵌入至控制矩阵B，通过累加方式稳定地增强长程依赖建模能力。
- **双分支混合结构**：结合Bi-Mamba（全局）与Bi-SwinT（局部），以极低的计算代价实现全局与局部特征的并行捕获。

## 方法详解
**整体架构**：网络分为两阶段 $\mathcal{N}_1$（基于BBCU的事件像素粗略修复）和 $\mathcal{N}_2$（主去马赛克分支），辅以独立的全局视觉编码器分支。

**二值化视觉编码器**：以预训练RAM大视觉编码器为教师，蒸馏训练紧凑的二值化视觉编码器；主分支训练时该编码器冻结，通过单层Bi-Linear适配器将全局视觉嵌入注入各层Bi-Mamba。

**Bi-Mamba设计**：
- **二值化策略**：全精度权重 $\mathbf{W}^f$ 经Sign函数二值化为 $\mathbf{W}^b \in \{+1, -1\}$；激活值 $\mathbf{A}^f$ 经RSign函数（含可学习阈值 $\alpha$）二值化，并引入可学习缩放因子 $\mathbf{S}$ 补偿精度损失，计算公式为 $\text{Bi-Linear}(\mathbf{A}^f) = \text{bitcount}(\text{XNOR}(\mathbf{W}^b, \mathbf{A}^b)) * \mathbf{S}$。
- **核心Selective Scan保持全精度**：输入经多方向扫描后提取SSM参数 $\mathbf{B}_i, \mathbf{C}_i, \pmb{\Delta}_i$；其中控制矩阵 $\mathbf{B}_i$ 拼接了来自二值视觉编码器的全局视觉向量 $\mathbf{S}$，即 $\mathbf{B}_i = \text{Bi-Linear}(\text{Concat}(\mathbf{X}_i, \mathbf{S}))$，使全局信息以累加方式稳定影响隐状态。
- **最终输出**：对所有扫描方向的SS结果求和后，与另一Bi-Linear分支的输出做Hadamard积，再经Layer Normalization得到 $\mathbf{X}^{out}$。

**Bi-SwinT**：采用二值化Swin Transformer块，负责局部特征提取与信息交互。

## 实验与结果
**数据集**：训练集为MIPI Workshop 2024（800训练对+26测试对）；测试集包括Kodak、McM、BSD100、Urban100、Wed、REDS、Vid4，以及Alpsentek采集的真实HybridEVS数据。

**主要定量结果（MIPI数据集，PSNR/SSIM）**：
- BMTNet：36.95/0.975，显著优于所有其他BNN（ReActNet 36.47/0.971、BBCU 36.06/0.970等）。
- 参数量1.28M，OPs 6.56G，在BNN中最优的同时参数量最少。
- 视频数据集上平均PSNR达37.70/0.985，超越多个全精度方法（如NAFNet 38.31/0.982、Restormer 38.50/0.985）。

**消融结论**：
- 引入Bi-Mamba相较无Mamba提升1.78dB PSNR；全精度Mamba（6.15M/54.07G）降至Bi-Mamba（1.28M/6.56G）仅损失0.79dB。
- 全局视觉嵌入置于B矩阵取得最佳效果（PSNR 36.95），置于$\Delta$矩阵反而下降（36.83），说明累加融合方式更稳定。

## 相关工作脉络
- **去马赛克**：传统插值方法→CNN方法（如DFormer、NAFNet等）→针对Quad Bayer的GAN方法（SAGAN、PIPNet）；本文聚焦Edge部署的轻量化BNN方案。
- **事件相机/ HybridEVS**：已有DemosaicFormer等全精度方法；本文是首个面向该任务的二值化轻量级方案。
- **Mamba在视觉中的应用**：VMamba、Mambair等利用SSM的长程建模能力；本文首次对Mamba进行系统化二值化。
- **二值神经网络**：BNN、ReActNet、BBCU等专注于CNN；BiViT、BTM扩展至Transformer；本文首次探索Mamba的二值化。
- **全局视觉先验**：Seesr等工作利用大视觉编码器增强局部任务；本文将其与二值Mamba结合，用于控制矩阵的增强。
- **定位差异**：本文不是单纯追求精度，而是面向边缘ISP的极致轻量化，在BNN中首次实现了Mamba级别的全局建模。

## 局限性与未来方向
- 仅在一个真实传感器（Alpix-Eiger）上进行了定性验证，缺乏更广泛的跨设备泛化测试。
- 二值化视觉编码器的蒸馏依赖预训练大模型（RAM），实际部署时需额外考虑离线训练成本。
- 视频实验仅在两个数据集上评估，未涉及更高帧率或更长序列的稳定性分析。
- 未来可将Bi-Mamba策略扩展至其他状态空间模型（如S4、VMamba变体），或探索部分二值化（Partial Binarization）以进一步平衡精度与效率。

## 研究启发与可借鉴点
- **核心-投影分离的二值化范式**：对Mamba中关键操作（Selective Scan）保持全精度、非关键线性/卷积投影二值化的思路，可迁移至其他SSM架构或混合架构中。
- **全局视觉嵌入的控制矩阵设计**：将全局信息融入B矩阵而非$\Delta$矩阵的设计具有理论解释力，可推广至其他需要全局感知的序列建模任务。
- **两阶段去马赛克架构**：先用轻量网络修复事件像素损失、再用主网络去马赛克的pipeline设计，对处理传感器异常像素（如坏点、缺失像素）具有通用参考价值。
- **BNN在图像恢复任务中的适用性分析**：本文系统对比了多种BNN在去马赛克上的表现，其实验设计可作为后续BNN恢复类工作的基准评估框架。

## 关键术语表
- **Quad Bayer CFA**：一种4×4像素色块排列的非拜耳彩色滤镜阵列，每个色块由4个同色像素组成，提升低光性能但增加去马赛克难度。
- **HybridEVS**：结合传统帧式成像与事件检测的混合事件视觉传感器，兼具高空间分辨率与高时间动态响应。
- **Binarized Neural Network (BNN)**：将网络权重和激活值二值化为{+1, -1}的极端压缩技术，可大幅降低计算与存储开销。
- **Selective Scan (SS)**：Mamba的核心机制，沿多个扫描方向对序列进行状态空间模型递推计算，捕获长程依赖。
- **Mamba**：基于状态空间模型（SSM）的高效序列建模架构，具有线性复杂度，适用于视觉和NLP任务。
- **Bi-Mamba**：本文提出的二值化Mamba块，对投影层二值化而保留Selective Scan全精度，并嵌入全局视觉信息。
- **RSign函数**：带可学习阈值的反转符号函数，用于激活值二值化，缓解精度损失。
- **SSM参数**：状态空间模型的参数，包括$\mathbf{A}$（状态转移）、$\mathbf{B}$（输入控制）、$\mathbf{C}$（输出投影）和$\pmb{\Delta}$（步长）。

## 可复现要素
- **数据集**：MIPI Workshop 2024（公开）；真实HybridEVS数据（与Alpsentek合作采集，论文未公开）；测试基准Kodak、BSD100、Urban100等为公开数据集。
- **代码/权重**：论文声明代码和模型已开源：https://github.com/Clausy9/BMTNet
- **关键超参**：输入裁剪尺寸128×128，batch size=32，学习率$2\times10^{-4}$至$1\times10^{-7}$余弦退火，总迭代$1\times10^6$次，L1损失；upsampling/downsampling保持FP精度。
