---
title: "SINR-Sparsity-Driven-Compressed-Implicit-Neural-Representati"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Jayasundara_SINR_Sparsity_Driven_Compressed_Implicit_Neural_Representations_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 09:39:40"
field: "隐式神经表示压缩"
keywords: ["隐式神经表示", "压缩感知", "稀疏编码", "信号压缩", "Neural Radiance Fields"]
innovations: ["利用INR权重的Gaussian分布通过压缩感知寻找稀疏码", "基于CLT用随机种子生成感知矩阵免去传输开销", "提出Tiny INR的矩阵展平策略提升小型网络压缩效率"]
benchmarks: ["KODAK图像压缩", "Stanford shape占据场重建", "NeRF渲染"]
---

# 论文速读：SINR: Sparsity Driven Compressed Implicit Neural Representations

## 一句话总结
本文提出SINR，一种基于稀疏驱动的隐式神经表示（INR）压缩算法，利用INR权重向量的Gaussian分布特性，通过压缩感知理论在高维字典中寻找稀疏码，实现存储需求的显著降低，同时保持高质量重建。

## 研究问题与动机
- **INR压缩效率不足**：现有INR压缩方法（如COIN、COIN++、INRIC、SHACIRA）主要依赖对权重的直接量化与熵编码，或通过学习变换提取潜在码，性能严重依赖量化和熵编码方案，未能从权重空间本身挖掘冗余。
- **跨模态通用性缺失**：传统深度学习压缩方法（如基于自编码器）通常针对单一数据模态设计，适配不同模态需重新训练专门架构，缺乏统一性。
- **存储与传输瓶颈**：随着数据量激增（日均400TB+），INR虽能以较少参数表示信号，但其权重仍需高效压缩以降低存储和传输开销。
- **权重空间固有稀疏性未被利用**：INR权重向量呈现Gaussian分布，暗示存在可利用的字典压缩潜力，但现有工作未探索此方向。

## 核心贡献（创新点）
- **首次将压缩感知应用于INR权重压缩**：利用权重向量的Gaussian分布特性，通过$L_1$最小化在高维字典中寻找稀疏码，而非直接对权重量化。
- **基于中心极限定理无需学习/传输感知矩阵**：证明随机采样的高维矩阵可满足CLT，接收端只需种子即可复现矩阵，消除额外传输开销。
- **提出Tiny INR的矩阵展平策略**：针对每层神经元数$k<50$的小型INR，将$k\times k$权重矩阵展平为$k^2$维向量，提升稀疏表示可行性。
- **通用可插拔设计**：SINR作为基础压缩技术，可嵌入任意现有INR压缩算法（如COIN++）的预处理阶段，提升其压缩率。
- **跨模态验证**：在图像、占据场（occupancy fields）和NeRF三种数据模态上均实现显著存储降低，同时保持高质量解码。

## 方法详解
- **INR权重压缩框架**：
  - 将INR每层权重向量$\mathbf{w}\in\mathbb{R}^{k_1}$表示为$\mathbf{w}=\mathbf{A}\mathbf{x}$，其中$\mathbf{A}\in\mathbb{R}^{k_1\times k_2}$为感知矩阵（字典），$\mathbf{x}\in\mathbb{R}^{k_2}$为稀疏码，$\|\mathbf{x}\|_0=s\ll k_1$。
  - 通过$L_1$最小化求解稀疏码：$\min\|\mathbf{x}\|_1$，subject to $\mathbf{w}=\mathbf{A}\mathbf{x}$，约束$2s < k_1$。
  - 使用正交匹配追踪（OMP）算法求解。

- **感知矩阵的随机生成**：
  - 根据中心极限定理（CLT），权重向量的Gaussian分布可由随机变量的线性组合生成。
  - 感知矩阵$\mathbf{A}$的元素随机采样自标准正态分布，通过种子控制随机数生成器，接收端用相同种子复现$\mathbf{A}$，无需传输。

- **参数存储转换**：
  - 原始：存储$k_1$个浮点数（32-bit）。
  - SINR后：仅存储$s$个非零浮点数（32-bit）+$s$个索引整数（16-bit），压缩比为$\frac{2s}{k_1}$。

- **Tiny INR策略**：
  - 当隐藏层神经元数$k<50$时，直接对$k$维向量稀疏化困难。
  - 将$i$层到$(i+1)$层的$k\times k$权重矩阵展平为$k^2$维向量，应用SINR，使$k^2\gg 50$，满足$2s<k^2$约束。

- **与量化/熵编码结合**：
  - SINR输出稀疏码后，仅对非零值和索引进行均匀量化（16-bit，65536级）和Brotli熵编码。
  - 解码流程：熵解码→反量化→稀疏码$\mathbf{x}$→与种子生成的$\mathbf{A}$相乘恢复$\mathbf{w}$。

## 实验与结果
- **数据集**：
  - 图像：KODAK数据集（24张RGB自然图像，768×512）。
  - 占据场：Stanford shape dataset（Thai Statue、Lucy等）。
  - NeRF：论文提及但未展示具体数字。

- **评估指标**：
  - 图像：bpp（bits per pixel）、PSNR。
  - 占据场：IoU、文件大小。

- **关键结果**：
  - **图像实验$C_1$**：平均PSNR 30 dB时，COIN需3.7 bpp，INRIC需2.0 bpp，SINR仅需1.7 bpp（相同量化器+熵编码器）。
  - **图像实验$C_2$**：SINR在相同PSNR下始终低于baseline的bpp。
  - **Meta-learning实验（$C_3$、$C_4$、$C_5$）**：
    - INRIC无位置编码：SINR在相同PSNR下bpp更低。
    - COIN++：达到约24.2 dB PSNR时，原方法需>1.5 bpp，SINR仅需<1 bpp（节省约33%）。
  - **占据场实验**：SINR在所有测试形状上实现最小文件大小和最高IoU（图6）。
  - **配置影响**：隐藏层神经元数越多（如$m=128$），压缩潜力越大；层数多但神经元少（如$h=7, m=64$）压缩增益较小。

- **最强结果**：图像压缩在$C_1$实验中，SINR以1.7 bpp达到30 dB PSNR，比INRIC（2.0 bpp）降低15%，比COIN（3.7 bpp）降低54%。

## 相关工作脉络
- **COIN/COIN++**： pioneered INR用于图像压缩，引入量化+熵编码及meta-learning；SINR在其基础上进一步压缩权重/调制参数，无需额外传输基网络。
- **INRIC**：使用meta-learning提升泛化，结合量化；SINR可插入其pipeline，独立于其量化方案进一步提升压缩。
- **SHACIRA**：对潜在权重量化并施加熵正则化；SINR从权重空间固有稀疏性出发，不依赖正则化。
- **WIRE**：使用小波隐式表示；SINR通用适用于各类INR架构（含Sinusoidal、Gaussian激活）。
- **传统压缩感知**：如JPEG（DCT）、MP3（心理声学模型）针对特定模态；SINR将压缩感知引入INR权重域，实现跨模态统一压缩。
- **字典学习方法**：如Deep Dictionary Learning（[36]）；SINR摒弃显式字典学习，利用CLT随机矩阵替代，免去字典传输开销。

## 局限性与未来方向
- **s的选择依赖经验**：稀疏度$s$需通过实验确定（随神经元数递增直至$2s=k_1$），缺乏理论最优解公式。
- **Tiny INR策略的扩展性**：矩阵展平虽提升稀疏性，但对极小型网络仍可能受限。
- **未覆盖音频模态**：实验仅验证图像、占据场、NeRF，音频未测试。
- **计算开销**：OMP算法求解$sparse code$在训练后增加额外计算，但未讨论延迟。
- **未来方向**：探索权重空间其他统计模式、开发自适应$s$预测机制、扩展至更多模态（视频、点云）。

## 研究启发与可借鉴点
- **权重空间Gaussian分布假设**：可迁移至其他隐式表示模型（如SIREN、Fourier Features），验证并优化压缩。
- **随机感知矩阵替代字典学习**：免去字典训练与传输开销，适用于任何高维向量压缩场景。
- **Tiny INR的矩阵展平技巧**：对小型网络参数压缩具有普适价值，可结合结构化稀疏进一步优化。
- **SINR与meta-learning结合**：已验证与INRIC/COIN++兼容，可探索与其他元学习压缩方法（如MAML-based）的结合。
- **跨模态统一压缩框架**：为构建通用信号压缩基础设施提供新思路，减少模态特定设计的依赖。

## 关键术语表
- **隐式神经表示（INR）**：用MLP的权重和偏置编码连续信号的表示方法，支持无限分辨率查询。
- **压缩感知（Compressed Sensing）**：利用信号稀疏性，通过少于奈奎斯特采样的测量重建信号的理论。
- **稀疏码（Sparse Code）**：信号在高维字典下的表示，大部分系数为零，仅保留少数非零元素。
- **正交匹配追踪（OMP）**：迭代贪心算法，用于求解$L_1$最小化问题的稀疏恢复。
- **中心极限定理（CLT）**：大量独立随机变量之和趋于正态分布，此处用于论证随机矩阵可生成Gaussian权重。
- **bpp（Bits Per Pixel）**：图像压缩中每个像素的比特数，衡量压缩效率。
- **占据场（Occupancy Fields）**：用二进制值表示3D空间中点是否属于目标物体的隐式表示。
- **Brotli熵编码**：通用数据压缩算法，用于量化后数据的无损压缩。

## 可复现要素
- **数据集**：KODAK（公开）、Stanford shape dataset（公开）。
- **代码/权重**：论文未提供开源代码；使用PyTorch实现，基于WIRE [29]代码库。
- **关键超参**：量化位宽16-bit（65536级）、Brotli熵编码、OMP算法、稀疏度$s$由实验确定（$2s<k_1$）。
