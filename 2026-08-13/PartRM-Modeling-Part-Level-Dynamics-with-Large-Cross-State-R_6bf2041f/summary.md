---
title: "PartRM-Modeling-Part-Level-Dynamics-with-Large-Cross-State-R"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Gao_PartRM_Modeling_Part-Level_Dynamics_with_Large_Cross-State_Reconstruction_Model_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 07:46:15"
field: "3D视觉与动态重建"
keywords: ["Part-level dynamics", "4D reconstruction", "3D Gaussian Splatting", "Drag conditioning", "World models", "Catastrophic forgetting", "Sim-to-real"]
innovations: ["提出PartRM框架，实现外观、几何与部件级运动同步的4D前馈重建", "构建PartDrag-4D数据集，提供20,548个部件状态的多视图观测", "设计多尺度拖拽嵌入与两阶段训练策略，缓解灾难性遗忘"]
benchmarks: ["PartDrag-4D", "Objaverse-Animation-HQ"]
---

# 论文速读：PartRM-Modeling-Part-Level-Dynamics-with-Large-Cross-State-Reconstruction-Model

## 一句话总结
PartRM提出了一种基于3D高斯表示的4D重建框架，能够从单视图图像和用户拖拽指令中同时快速建模物体的外观、几何与部件级运动，解决了现有方法仅输出2D视频、推理耗时长以及缺乏3D状态表示的瓶颈。

## 研究问题与动机
- 现有方法（如Puppet-Master）依赖微调视频扩散模型，只能生成单视图视频，缺乏3D表示，无法直接服务于多视角仿真器与机器人应用。
- 扩散去噪过程耗时数分钟，难以提供快速试错反馈，制约了其在实时操控策略生成中的落地。
- 4D动态数据稀缺，限制了以数据驱动的大规模重建模型对部件级运动的学习。
- 直接微调预训练的静态3D重建模型易引发灾难性遗忘，破坏模型原有的外观与几何建模能力。

## 核心贡献（创新点）
- 提出PartRM框架，首次实现外观、几何与部件级动力学的同步前馈式4D重建，基于3D高斯表示实现秒级推理。
- 构建PartDrag-4D数据集，包含738个对象、20,548个部件状态的多视图观测，显著缓解4D动态数据稀缺。
- 设计多尺度拖拽嵌入模块，在不同空间尺度（128×128/32×32/8×8）下将拖拽条件注入UNet各下采样层，提升多粒度动态理解。
- 引入两阶段训练策略（运动学习→外观学习），以知识蒸馏和光度损失分步优化，避免灾难性遗忘。
- 在多个基准上建立SOTA，并证明该方法可直接用于Isaac Gym中的机器人零样本操作任务。

## 方法详解
- **基础架构**：以预训练LGM为骨干，采用非对称U-Net处理多视图图像，输出高分辨率3D高斯（14维参数）。
- **多视图生成**：使用微调的Zero123++从单视图生成多视图图像，增强视图一致性；多视图图像与原始图像共同输入网络。
- **拖拽传播模块**：利用SAM对输入拖拽起点进行部件分割，得到移动部件掩码；在掩码上采样$N$个起始点，保持与输入相同的相对位移向量$\Delta a_t$，生成多个传播拖拽$a_{t,i}$，消除单拖拽歧义。
- **多尺度拖拽嵌入**：对每个拖拽的起点与终点经Fourier embedder + 3层MLP编码，在高斯特征图上按位置写入通道拼接向量$F(a_{src})\oplus F(a_{dst})$；对$l$层输出$O_l$，构造拖拽地图$M_{t,l,i}$后叠加所有拖拽得$M_{t,l}=\sum_i M_{t,l,i}$，通过零初始化卷积层将$M_{t,l}\oplus O_l$融合回特征：$I_{l+1}=O_l+\text{Conv}(M_{t,l}\oplus O_l)$。
- **两阶段训练**：
  - **阶段一（运动学习）**：以预训练网络对目标状态生成的3D高斯作为教师，对学生网络输出像素级14维高斯参数计算L2损失：$\mathcal{L}_1=\sum\|\mathcal{G}S_i-\mathcal{G}S_j\|_2^2$，实现知识蒸馏。
  - **阶段二（外观学习）**：改用真实渲染RGB与Alpha图作为监督，损失为$\mathcal{L}_2=L_{mse}(I_{color},I_{gt\_color})+\lambda_1 L_{lpips}(I_{color},I_{gt\_color})+\lambda_2 L_{mse}(I_{alpha},I_{gt\_alpha})$，其中$\lambda_1=\lambda_2=1.0$。

## 实验与结果
- **数据集**：PartDrag-4D（738 mesh、8类，20,548状态，12视图）；Objaverse-Animation-HQ（约15,000状态）。
- **评估协议**：新视角合成（NVS），渲染8个视角，计算PSNR、SSIM、LPIPS。
- **基线对比**：DiffEditor、DragAPart、Puppet-Master（均按统一协议重新测试或微调）。
- **定量结果**：
  - PartDrag-4D：PartRM **PSNR 28.15 / SSIM 0.9531 / LPIPS 0.0356**，显著领先次优DragAPart（24.91 PSNR）。
  - Objaverse-Animation-HQ：PartRM **PSNR 21.38 / SSIM 0.9209 / LPIPS 0.0758**。
  - **推理速度**：仅**4.2s**（含多视图生成），较Puppet-Master（64.9s/187.5s）快数十倍。
- **消融**：
  - 拖拽数量：10个拖拽最优（PSNR 28.15）。
  - 训练阶段：两阶段组合最优（仅阶段1：22.05；仅阶段2：25.87）。
  - 多尺度嵌入：多尺度融合优于任何单一尺度（128×128: 25.48；32×32: 27.99；8×8: 26.87）。
- **Sim-to-Real**：在Isaac Gym中训练的操作策略可零样本泛化到真实URDF环境，成功完成抽屉与门的开合任务。

## 相关工作脉络
- **Drag-conditioned图像/视频生成**：DiffEditor、DragAPart、Puppet-Master等方法仅输出2D单视图视频，缺乏3D表示；PartRM直接输出3D高斯状态，支持任意视角渲染。
- **Large Reconstruction Models**：LGM、L RM、GS-LRM等专注静态3D重建；本文将其扩展至4D，引入拖拽条件与动态建模。
- **动态3D生成**：L4GM从视频生成动态3D但不支持动作条件；本文以单视图+拖拽为条件，实现可控部件级运动。
- **灾难性遗忘缓解**：与直接端到端微调不同，本文通过两阶段训练与知识蒸馏保留预训练静态建模能力。
- **机器人操作仿真**：传统方法依赖affordance预测或手工特征；本文利用生成的部件网格与运动轴直接训练操作策略，实现端到端sim2real。

## 局限性与未来方向
- 对偏离训练分布的复杂articulated物体泛化有限（野外数据展示部分失败案例）。
- 数据集基于PartNet-Mobility，类别与运动模式仍受限于现有标注资源。
- 多视图生成依赖Zero123++，输入质量波动可能影响下游重建稳定性。
- 未来可拓展至软体变形、真实世界数据采集，以及结合物理引擎提升仿真保真度。

## 研究启发与可借鉴点
- **3D高斯作为状态表示**：兼顾渲染效率与可微性，适合需要多视角输出的动态世界模型。
- **多尺度条件注入**：在UNet各下采样层融合不同空间分辨率的条件特征，可同时捕捉局部精确运动与全局上下文，可迁移至其他条件生成任务。
- **两阶段蒸馏训练**：先以伪标签（预训练教师输出）学习新动态，再以真实监督联合优化外观，有效平衡新技能获取与旧知识保留。
- **数据构建思路**：基于PartNet-Mobility动画生成多视图状态数据，为4D动态研究提供了可复用的数据流水线。
- **Sim2Real验证范式**：在模拟环境中提取部件网格与运动轴训练策略，直接部署到真实物理引擎，展示端到端落地潜力。

## 关键术语表
- **Part-Level Dynamics**：描述对象各独立部件运动规律及其相互耦合的动态特性。
- **3D Gaussian Splatting**：基于3D高斯椭球分布的高效渲染表示，支持快速新视角合成。
- **Large Reconstruction Model (LRM)**：基于大规模数据预训练、前向推理直接生成3D表示的神经网络。
- **Drag Embedding**：将用户拖拽操作的起止点编码为空间特征图，注入生成网络实现条件控制。
- **Catastrophic Forgetting**：神经网络在学习新任务时显著遗忘原有知识的现象。
- **PartDrag-4D**：本文构建的部件级动态多视图数据集，包含超2万个状态。
- **NVS-First / Drag-First**：两种评估协议，前者先生成多视图再施加拖拽，后者先施加拖拽再生成多视图。
- **Sim-to-Real**：在模拟器中训练策略并直接部署到真实物理环境的技术路线。

## 可复现要素
- **数据集**：PartDrag-4D已公开；Objaverse-Animation-HQ为已有公开数据子集。
- **代码/模型**：代码、数据与模型已公开于 https://PartRM.c7w.tech/。
- **关键超参**：
  - 拖拽传播采样点数：10
  - 多尺度嵌入分辨率：128×128、32×32、8×8
  - 外观损失权重：$\lambda_1=1.0$，$\lambda_2=1.0$
  - 渲染分辨率：256×256
  - 训练损失：Stage1 L2 on 14-dim Gaussians；Stage2 MSE+LPIPS+Alpha MSE
- **依赖组件**：Zero123++（微调）、LGM backbone、SAM、Isaac Gym。
