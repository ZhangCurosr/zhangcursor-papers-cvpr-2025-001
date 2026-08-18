---
title: "Touch2Shape-Touch-Conditioned-3D-Diffusion-for-Shape-Explora"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_Touch2Shape_Touch-Conditioned_3D_Diffusion_for_Shape_Exploration_and_Reconstruction_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 09:47:39"
field: "触觉引导的3D形状重建与探索"
keywords: ["3D reconstruction", "touch-based sensing", "diffusion model", "shape exploration", "multimodal fusion", "reinforcement learning"]
innovations: ["提出触觉条件3D扩散模型，在低维潜空间引导触觉探索策略", "设计对比触觉编码器与Touch Shape Fusion模块，增强多模态对齐与局部细节重建"]
benchmarks: ["ABC dataset", "ShapeNet dataset"]
---

# 论文速读：Touch2Shape-Touch-Conditioned-3D-Diffusion-for-Shape-Explora

## 一句话总结
本文提出 Touch2Shape，一种触觉条件驱动的 3D 扩散模型，利用低维潜向量指导触觉探索策略并重建目标 3D 形状，在 ABC 与 ShapeNet 数据集上均显著优于 ActiveVT 等现有视觉-触觉重建方法。

## 研究问题与动机
- 现有 3D 重建方法多依赖视觉输入，在遮挡、光照变化及局部细节刻画上存在不足。
- 纯视觉方法难以应对复杂局部几何与透明/反射表面，需引入触觉模态获取高保真局部 3D 信息。
- 已有触觉探索方法（如 ActiveVT）需在每一步生成完整高分辨率形状，计算开销大且策略学习能力受限。
- 缺少能够有效融合触觉与视觉信息、并在低维潜空间中进行高效探索与重建的统一框架。

## 核心贡献（创新点）
- 提出 Touch2Shape 触觉条件 3D 扩散模型，在低维潜空间生成紧凑形状表示以引导后续触觉探索。
- 设计对比触觉编码器，将触觉特征与形状特征映射到联合隐空间，增强多模态对齐。
- 引入 Touch Shape Fusion 模块，在解码阶段融合多尺度触觉体素特征，提升局部细节重建精度。
- 将扩散模型与强化学习结合训练触觉探索策略，利用扩散损失差值设计奖励函数，避免每步生成完整形状。

## 方法详解
- **形状编码与去噪扩散**：采用预训练的 3D VQ-VAE 将 T-SDF 体素编码为低维潜向量 z，并在随机时间点加入高斯噪声，使用触觉条件去噪网络 $E_\theta$ 预测噪声，损失为 $L_{diff} = \|E_\theta(z_t, r(t), C(T_0,...,T_{n-1})) - \epsilon_t\|_2$。
- **触觉嵌入**：每个触觉图像生成大小为 $M \times 4$ 的触觉图张量，拼接后通过位置编码、卷积与池化得到 N 个触觉 token。
- **对比触觉编码器**：基于 moco 框架，以触觉特征为 query、形状潜特征为 key，最小化 InfoNCE 损失 $L_{rl} = -\log \frac{e^{q \cdot k_p / \tau}}{\sum_{i=0}^{K} e^{q \cdot k_i / \tau}}$。
- **视觉-触觉融合**：在视觉-触觉设置中，使用 ResNet 提取图像特征 token，与触觉 token 经 dropout 后共同输入去噪网络。
- **Touch Shape Fusion**：将历史触觉信息体素化后通过额外编码器提取多尺度特征，与解码器特征按公式 $M_1 = \frac{\lambda \cdot F_3^e \cdot e^{F_1^d}}{\sum_{c'} e^{F_1^{d}(c')}}$ 进行加权融合，细化重建形状。
- **策略训练与奖励设计**：每步输入真实潜向量 z 加噪后得到去噪潜向量 $z'$，通过 Q-network 预测动作价值，奖励为 $R = H(L_{diff}(t,n-1) - L_{diff}(t,n))$，强化学习损失采用 DQN 形式。

## 实验与结果
- **数据集**：ABC（40,000 对象）、ShapeNet（1,650 对象，6 类）。
- **评估指标**：Chamfer Distance（CD）、Earth Mover's Distance（EMD）。
- **ABC 数据集**（Table 1）：视觉-触觉单握持下，Ours CD=1.475，ActiveVT=2.486，VTRecon=2.637；触觉仅模式下 Ours CD=6.794，优于 ActiveVT 的 8.220。
- **ShapeNet 数据集**（Table 2）：20 次触摸下，Ours_T 平均 EMD=0.053，Ours_TV=0.042，均低于 TouchSDF 的 0.081。
- **策略对比**（Table 3）：5 次触摸后 CD 下降比例，Ours_T 从 4.88 降至 6.63，优于 Random、Even 及 ActiveVT。
- **消融实验**（Table 4）：去除对比学习或融合模块均导致性能下降，验证各组件有效性。

## 相关工作脉络
- **VTRecon [33]**：基于图卷积网络的视觉-触觉局部图融合重建，本文在其基础上引入扩散模型与对比学习实现更紧凑的潜空间表示。
- **ActiveVT [34]**：基于网格的重构与 RL 探索策略，需每步生成完整形状；本文仅末步生成完整形状，降低计算负担。
- **TouchSDF [7]**：基于隐式神经函数的触觉重建，仅支持纯触觉；本文扩展至视觉-触觉多模态融合。
- **SDFusion [5] / DiffusionSDF [6]**：视觉条件 3D 扩散重建工作，本文将其推广至触觉条件并引入探索策略。
- **IC3D [32] / UniTouch [42]**：图像-3D 联合编码与多模态触觉表征学习，本文借鉴其对比学习思想用于触觉-形状对齐。
- **主动视觉/触觉探索**：多数方法基于不确定性采样或视角规划，本文利用扩散潜向量直接引导触觉动作选择。

## 局限性与未来方向
- 当前实验基于仿真环境，尚未在真实机器人平台上验证触觉探索与重建的泛化性。
- 方法仅针对单一物体重建，未扩展到完整场景的触觉探索与重建。
- 视觉-触觉融合依赖初始 RGB 图像，在纯触觉设定下性能仍有提升空间。
- 未来可结合神经渲染技术，利用主动触觉感知合成多视角视觉图像以进一步提升重建质量。

## 研究启发与可借鉴点
- **潜空间策略引导**：将扩散模型生成的低维潜向量用于指导 RL 探索策略，避免每步高分辨率重建，可迁移至其他多模态主动感知任务。
- **对比触觉-形状对齐**：基于 moco 的对比学习框架可有效融合触觉与 3D 形状表征，适用于多传感器模态对齐问题。
- **触觉形状融合机制**：在解码阶段引入多尺度触觉特征加权融合，可增强局部细节重建，值得借鉴于点云/网格补全任务。
- **奖励函数设计**：利用扩散损失差值作为探索奖励，提供了一种无需完整形状生成的策略训练信号，可拓展至其他生成式探索场景。

## 关键术语表
- **Touch2Shape**：本文提出的触觉条件 3D 扩散模型，用于基于触觉交互的形状探索与重建。
- **T-SDF**：截断符号距离场，用于表示 3D 形状体积的隐式函数。
- **VQ-VAE**：矢量量化变分自编码器，将 3D 形状编码为低维离散潜向量。
- **Contrastive Touch Encoder**：基于对比学习的触觉-形状联合编码模块，实现对齐触觉与 3D 特征。
- **Touch Shape Fusion**：融合多尺度触觉体素特征与解码器特征的模块，提升局部重建精度。
- **Exploration Policy**：基于强化学习的触觉探索策略，指导机械臂选择下一触摸位置。
- **Chamfer Distance (CD)**：评估重建点云与地面真实点云之间点对点距离的常用指标。
- **Earth Mover's Distance (EMD)**：衡量两个点集之间最小“搬运工作量”的评估指标。

## 可复现要素
- **数据集**：ABC、ShapeNet（公开）；触觉与视觉图像由 [34] 提供的仿真环境渲染生成。
- **代码/权重**：论文未明确开源声明；神经网络基于 PyTorch 实现，训练硬件为 GeForce RTX 4090。
- **关键超参**：体素分辨率 64×64×64；扩散模型训练 1M 迭代，学习率 1e-5，batch size 12；触觉形状融合训练 250K 迭代，学习率 1e-4，batch size 8；策略训练 200 epochs，学习率 3e-4，batch size 16。
