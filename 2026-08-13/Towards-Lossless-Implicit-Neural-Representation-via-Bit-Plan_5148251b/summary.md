---
title: "Towards-Lossless-Implicit-Neural-Representation-via-Bit-Plan"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Han_Towards_Lossless_Implicit_Neural_Representation_via_Bit_Plane_Decomposition_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 09:47:35"
field: "隐式神经表示的高精度无损建模"
keywords: ["隐式神经表示", "位平面分解", "无损拟合", "比特精度", "Bit Bias", "极端量化"]
innovations: ["首次从UAT推导INR参数上界并揭示指数增长规律", "提出位平面分解方法实现16-bit图像/音频的BER=0无损表示", "发现并量化Bit Bias与Bit-Spectral Bias现象"]
benchmarks: ["TESTIMAGES", "MIT-5K", "DIV2K", "Kodak", "Librispeech"]
---

# 论文速读：Towards-Lossless-Implicit-Neural-Representation-via-Bit-Plan

## 一句话总结
本文从数字信号视角理论量化了隐式神经表示（INR）的参数上界，提出**位平面分解**方法将高比特深度信号拆分为多个1-bit平面分别拟合，从而实现了此前无法达成的**无损16位图像/音频拟合**，并进一步拓展至无损压缩、位深扩展与极端网络量化三大学术应用。

## 研究问题与动机
1. **连续表示与离散现实之间的鸿沟**：INR以坐标型MLP拟合连续信号，但计算机实际处理的是固定比特精度的数字信号（8-bit/16-bit图像、24-bit音频），要求模型在给定比特精度下达到理论上的**无损逼近**。
2. **现有INR方法无法达到真正无损**：即便大幅提升参数量，现有方法（Tanh+P.E、SIREN、FINER等）在16-bit等高动态范围图像上仍存在不可忽略的位错误率（BER），且训练收敛极慢。
3. **理论缺口——缺少对INR参数上界的严格量化**：现有工作主要关注激活函数设计或坐标编码，从未从**比特精度→误差容限→参数上界**的角度建立联系，导致设计思路缺乏理论约束。
4. **缺乏对INR训练动态中"数字域偏置"的系统认识**：研究者已知频谱偏置（Spectral Bias），但不知道在比特轴上同样存在类似偏置（即MSB比LSB更容易收敛）。

## 核心贡献（创新点）
1. **首次从UAT（通用近似定理）出发推导INR参数上界 $\mathcal{U}_d(n)$**，揭示上界随比特精度 $n$ 呈指数增长：$\mathcal{U}_d(n) \propto (2^{n+1}-2)^{2d}$。
2. **提出位平面分解（Bit-Plane Decomposition）方法**：将n位信号拆分为n个1-bit平面，每个平面独立用INR拟合，使得有效参数上界从 $O(2^{2dn})$ 骤降至 $O(2^{2d})$。
3. **发现并量化"Bit Bias"与"Bit-Spectral Bias"**：证明INR对MSB收敛快、对LSB收敛慢；且高位比特的频率轴也存在类似Spectral Bias的偏置。
4. **实现首个真正无损的16-bit图像/音频INR拟合**：在TESTIMAGES、MIT-5K、Kodak等数据集上BER=0，PSNR=∞，且收敛迭代数显著低于所有基线。
5. **拓展三大学术应用**：极端量化（三元权重 Ternary INR）、位深扩展（8-bit→16-bit自监督外推）、无损压缩（结合RECOMBINER超越PNG/JPEG2000/WebP）。

## 方法详解
1. **参数上界量化**：基于Jentzen et al. [16] 的UAT显式界，给定Lipschitz常数 $L$ 与误差容限 $\epsilon(n) = \frac{1}{2(2^n-1)}$，证明存在MLP $h_\theta$ 满足：
   - $\mathcal{P}(h_\theta) \leq \mathfrak{C} \epsilon(n)^{-2d} = \mathfrak{C}(2^{n+1}-2)^{2d}$
   - $\sup_x |h_\theta(x) - h_n(x)|_1 \leq \epsilon(n)$
2. **位平面分解公式**：将n位图像 $\mathbf{I}_n$ 分解为：
   $$\mathbf{I}_n = \frac{1}{2^n-1}\sum_{i=0}^{n-1} 2^i \mathbf{B}^{(i)}, \quad \mathbf{B}^{(i)} \in \{0,1\}^{H\times W\times 3}$$
   每个位平面 $\mathbf{B}^{(i)}$ 由单个INR $f_\theta(\mathbf{x}, i)$ 拟合，其中 $\mathbf{x}\in\mathbb{R}^2$ 为空间坐标，$i$ 为额外的**位轴坐标**。
3. **损失函数设计**：针对每个位平面的二值预测，使用**二元交叉熵（BCE）损失**：
   $$\hat{\theta} = \arg\min_\theta \mathcal{L}_{BCE}(\mathbf{B}^{(i)}(\mathbf{x}), f_\theta(\mathbf{x}, i))$$
   消融实验表明BCE收敛最快，优于MSE与MAE。
4. **推理重建流程**：训练后通过式(11)重组：
   $$\mathbf{I}_n(\mathbf{x};\theta) = \frac{1}{2^n-1}\sum_{i=0}^{n-1} 2^i \mathcal{Q}(\hat{f}_\theta(\mathbf{x}, i))$$
   其中 $\mathcal{Q}$ 为量化操作，保证输出落在 $Q_n$ 集合内，从而实现理论无损。
5. **高维扩展（消融）**：实验验证当输入维度从3维 $(h,w,i)$ 增至4维 $(h,w,i,c)$ 时，迭代数从790增至1438，印证了维度灾难与参数上界增长的理论预期。

## 实验与结果
1. **数据集**：16-bit数据集（TESTIMAGES 40张、MIT-5K 1000张）、8-bit数据集（DIV2K 100张、Kodak 24张）、音频（Librispeech FP32）。
2. **评估指标**：PSNR(dB)、SSIM、RMSE、**Bit-Error-Rate (BER)**、收敛迭代数。
3. **16-bit图像拟合**：
   - Ours vs SIREN：PSNR 80.17 dB vs 78.52 dB，BER **0.0000** vs 0.1544，迭代数 **8** vs 5000（Tab.1）。
   - Ours vs FINER：PSNR 86.48 dB vs 78.39 dB，BER **0.0000** vs 0.1138。
4. **8-bit图像拟合**：在DIV2K与Kodak上同样实现BER=0、PSNR=∞，迭代数显著低于所有基线（Tab.1）。
5. **假设验证实验**（Tab.3）：当参数上界 $\mathcal{U}_d(n)$ 越接近实际参数量时，收敛越快，验证了"降低上界加速收敛"的核心假设。
6. **音频拟合**（Tab.9、Fig.12）：对Librispeech FP32音频实现无损拟合，STT任务下WER显著低于SIREN与DINER。
7. **组合应用**（Tab.2）：与Instant-NGP、DINER、Gauss、FINER等结合后，均能实现无损且减少迭代数（如Instant-NGP迭代从5000降至4668）。
8. **极端量化**（Tab.4、Fig.10）：三元权重（1.58-bit）下仍保持无损，模型体积从1.27MB（FP32）降至656KB，BitOps降低8倍。
9. **位深扩展**（Tab.5）：自监督8-bit→16-bit外推，PSNR 55.92/SSIM 0.9993，优于BitNet与BECNN等监督方法。
10. **无损压缩**（Tab.6、7）：结合RECOMBINER后bpp降至2.58（MNIST）与6.11（Fashion MNIST），PSNR=∞，BER=0，超越PNG/JPEG2000/WebP/TIFF等传统无损编解码器。

## 相关工作脉络
1. **NeRF（Mildenhall et al. [25]）**：开创性将INR用于辐射场表示，但未关注高比特精度下的无损问题。
2. **SIREN（Sitzmann et al. [37]）**：提出正弦激活函数改善高频学习，本文指出即使使用SIREN也无法达到16-bit无损。
3. **Spectral Bias（Rahaman et al. [31]）**：揭示神经网络优先学习低频分量，本文将其推广至**比特轴方向**提出Bit Bias。
4. **Instant-NGP（Müller et al. [26]）** & **DINER（Xie et al. [46]）**：利用Hash表加速训练，本文证明其仍需结合位平面分解才能达到无损。
5. **BCON / Bayesian INR（Dupont et al. [8,9,11]）**：贝叶斯INR压缩方法，本文与其互补——BCON关注有损压缩，本文关注无损。
6. **Bit-plane去量化（Han et al. [12] ABCD）**：位平面技术此前用于图像去量化，本文首次将其引入INR训练过程。
7. **Ternary Networks（Wang et al. [45] / Ma et al. [23] BitNet）**：极端量化LLM技术，本文证明其在INR中同样可实现无损。

## 局限性与未来方向
1. **高维场景受限**：当维度 $d>5$（如3D辐射场）时，参数上界 $\mathcal{U}_d(8) \approx 1.23\mathfrak{C}\times 10^{27}$ 过大，方法难以直接扩展。
2. **位偏置本质未充分解释**：Bit Bias的发现具有启发性，但其与Spectral Bias的深层数学联系仍需理论剖析。
3. **三值权重的激活函数限制**：三元INR需使用GELU激活且依赖严格权重初始化，扩展性受限。
4. **未来方向**：探索位平面分解与Hash编码、稀疏化技术的结合；将位轴外推思想推广至视频/4D信号；研究自适应位平面选择策略以减少冗余计算。

## 研究启发与可借鉴点
1. **"降维打击"思路**：通过引入额外坐标轴（位轴）将高比特问题转化为低比特序列问题，这一**坐标扩展降界**策略可迁移至其他需要高精度表示的领域（如科学计算、医学影像）。
2. **BCE优于MSE的启示**：对于二值或离散预测任务，BCE损失在收敛速度与最终精度上均优于传统回归损失，这一经验可复用于其他离散信号表示任务。
3. **偏置发现驱动方法设计**：发现Bit Bias后，作者不仅解释它，还主动通过位平面分解缓解它——这种"发现→解释→消除"的研究范式值得学习。
4. **理论量化指导实验设计**：从UAT推导参数上界，再据此设计位平面分解，使实验结果具有明确的理论支撑，提升了论文的说服力。
5. **开源代码与可复现性**：代码已公开，包含完整的训练脚本与消融实验，为后续研究提供了可靠起点。

## 关键术语表
- **隐式神经表示（INR）**：将连续信号（图像、辐射场、音频等）参数化为坐标型MLP的表示方法。
- **位平面分解（Bit-Plane Decomposition）**：将n位信号拆分为n个1位二进制平面，每个平面独立拟合以降低参数上界。
- **Bit Bias（位偏置）**：INR在训练中对MSB收敛快、对LSB收敛慢的现象，类比于频谱偏置。
- **Bit-Spectral Bias（位谱偏置）**：在位轴方向上，特定频率的比特值（如全1或全0）更容易收敛的现象。
- **无损表示（Lossless Representation）**：在给定比特精度n下，预测值与真实值在所有坐标上的量化结果完全一致。
- **参数上界（$\mathcal{U}_d(n)$）**：基于UAT推导的、保证达到指定误差容限所需的最小参数数量上限。
- **Bit-Error-Rate（BER）**：预测值与真实值在比特层面的错误率，是衡量无损精度的关键指标。
- **Ternary INR**：使用三元权重 $\{-1, 0, 1\}$ 的隐式神经表示，实现极端量化下的无损拟合。

## 可复现要素
- **数据集**：TESTIMAGES（16-bit，40张）、MIT-5K（16-bit，1000张）、DIV2K（8-bit，100张）、Kodak（8-bit，24张）、Librispeech（FP32音频）。
- **代码开源**：https://github.com/WooKyoungHan/LosslessINR ✓
- **网络结构**：5个隐藏层，每层512维，所有方法统一参数规模。
- **激活函数**：默认使用SINE（SIREN风格），消融实验涉及TANH、ReLU、Gauss、WIRE等。
- **损失函数**：默认BCE，消融对比MSE与MAE。
- **优化器**：Adam，学习率1e-4。
- **硬件**：NVIDIA RTX 3090 24GB。
- **关键超参**：位轴坐标 $i \in [0, n-1]$ 作为额外输入维度；量化公式 $\epsilon(n) = \frac{1}{2(2^n-1)}$。
