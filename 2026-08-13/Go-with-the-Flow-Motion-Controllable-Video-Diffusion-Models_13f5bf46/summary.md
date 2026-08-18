---
title: "Go-with-the-Flow-Motion-Controllable-Video-Diffusion-Models"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Burgert_Go-with-the-Flow_Motion-Controllable_Video_Diffusion_Models_Using_Real-Time_Warped_Noise_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 07:09:10"
---

# 论文速读：Go-with-the-Flow-Motion-Controllable-Video-Diffusion-Models

## 一句话总结
本文提出了一种仅修改噪声采样分布、无需改动模型架构的黑盒微调方案，通过基于光流场的实时线性复杂度噪声形变算法，将视频扩散模型的随机高斯噪声转化为带运动先验的结构性噪声，从而实现对局部物体运动、全局相机轨迹及任意运动迁移的高效精准控制。

## 研究问题与动机
- **核心问题**：现有视频扩散模型缺乏细粒度、交互式的运动控制能力，难以在精准运动引导与高质量时空生成之间取得平衡。
- **架构侵入性强**：主流运动控制方法需引入额外引导注意力、显式运动编码器或LoRA适配器，难以兼容现代全注意力时空Token架构（如CogVideoX）。
- **条件信号获取困难**：依赖精确相机位姿、边界框轨迹或稀疏关键点，标注成本高且泛化性受限，无法直接迁移至真实拍摄素材。
- **算法效率瓶颈**：现有噪声形变方法（如HIWYN）计算复杂度随帧数二次增长，无法满足大规模视频扩散模型微调的预处理与推理时效需求。

## 核心贡献（创新点）
1. **黑盒运动控制范式**：证明仅需将训练噪声替换为基于光流的形变噪声，即可让视频扩散模型天然具备运动控制能力，无需任何架构改动或条件模块注入。
2. **实时线性复杂度噪声形变算法**：提出迭代式帧间形变算法，通过像素级流密度追踪统一处理膨胀与收缩动态，严格保持空间i.i.d.高斯性并将时间复杂度降至$O(D)$，较HIWYN加速约26倍。
3. **噪声退化（Noise Degradation）可控机制**：引入可调节系数$\gamma$将干净形变噪声与随机高斯噪声混合，推理时灵活权衡运动遵循度与生成多样性，实现局部控制、相机运动与运动迁移的一站式切换。

## 方法详解
- **实时噪声形变算法**：摒弃HIWYN逐帧多边形光栅化的昂贵操作，仅维护前一帧噪声矩阵$q \in \mathbb{R}^{H \times W \times C}$与流密度矩阵$p \in \mathbb{R}^{H \times W}$。根据正向光流$f$与反向光流$f'$构建二分图，同步计算膨胀（单像素映射到多像素）与收缩（未覆盖区域通过后向流拉取填充）动态，并以流密度加权补偿概率质量，确保输出分布守恒。
- **高斯性与复杂度保证**：Proposition 1证明若输入为i.i.d.标准高斯，输出$q'$同样保持空间独立同分布高斯特性；Proposition 2证明每帧时间复杂度为$O(D)$（$D=H \times W$），总边数不超过$2D$。
- **黑盒微调流程**：以CogVideoX-5B（T2V与I2V双变体）为例，保留原始去噪MSE损失与训练管线，唯一改动是在每个训练视频前计算光流并生成形变噪声张量$\mathbf{Q} \in \mathbb{R}^{F \times C \times H \times W}$。由于模型工作于潜空间，先在图像空间生成噪声，再经时空下采样（空间8×、时间4×，最近邻插值+均值池化后乘8保持方差）注入Latent空间。
- **推理期控制策略**：采用确定性采样（如DDIM）初始化扩散过程。定义退化水平$\gamma \in [0, 1]$，混合公式为$\frac{(1-\gamma)\mathbf{Q} + \zeta\gamma}{\sqrt{(1-\gamma)^2 + \gamma^2}}$（$\zeta$为独立高斯噪声）。$\gamma \to 0$严格遵循输入流场，$\gamma \to 1$退化为随机噪声。针对三类应用场景分别适配流场来源：局部物体控制通过用户UI拖拽多边形生成合成流；相机运动控制复用参考视频光流；运动迁移支持光流、3D渲染引擎流（如Blender）及深度形变流（如WonderJourney）。

## 实验与结果
- **评估设置**：数据集涵盖DAVIS（43视频）、DL3DV（100视频）、WonderJourney（19视频）、VIPSeg及自建40条用户研究样本。基线包括HIWYN、InfRes、PYoCo、CaV以及SG-I2V、MotionClone、DragAnything、DMT、MotionCtrl、ImageConductor等。
- **高斯性与效率**：Moran’s I指数达0.00014（p=0.84），K-S检验p=0.44，确认空间高斯性完好；1024×1024分辨率下GPU耗时仅2.14ms，较HIWYN（55.2ms）加速约**26倍**，满足实时要求。
- **免训练图像编辑验证**：在DeepFloyd IF超分与DifFRelight重光照任务中，形变噪声显著降低时序不稳定，Warping Error分别降至152.04与85.82，全面优于插值基线与同类形变方法。
- **视频运动控制量化结果**：
  - 本地物体控制（VIPSeg）：FID降至**41.1**，CoTracker mIoU达**0.75**，显著优于SG-I2V（FID 61.4, mIoU 0.63）。
  - 运动迁移（DAVIS I2V）：FID 78.6，FVD降至1.21×10³，CoTracker mIoU 0.74；消融显示$\gamma=0.5$为最优平衡点。
  - 相机运动控制（DL3DV）：FID大幅下降至**48.4**，CoTracker mIoU
