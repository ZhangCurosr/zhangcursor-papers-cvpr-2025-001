---
title: "AffordDP-Generalizable-Diffusion-Policy-with-Transferable-Af"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Wu_AffordDP_Generalizable_Diffusion_Policy_with_Transferable_Affordance_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 06:59:58"
field: "具身智能/机器人学习"
keywords: ["扩散策略", "可转移可供性", "跨类别泛化", "机器人操作", "模仿学习", "扩散采样引导"]
innovations: ["提出可转移的3D静态接触点与动态轨迹作为统一affordance表征", "基于基础视觉模型与解耦6D变换实现跨物体affordance迁移", "在扩散采样中引入自适应affordance引导以增强动作精度"]
benchmarks: ["IsaacGym仿真PullDrawer/OpenDoor", "真机PullDrawer/OpenDoor/Pick&Place"]
---

# 论文速读：AffordDP-Generalizable-Diffusion-Policy-with-Transferable-Affordance

## 一句话总结
AffordDP提出了一种基于可转移可供性（affordance）的扩散策略，通过迁移3D接触点和接触后轨迹，显著增强了机器人在未见过的物体实例和全新类别上的零样本泛化能力，解决了现有扩散策略跨类别泛化的瓶颈。

## 研究问题与动机
*   **核心问题：** 现有基于扩散的机器人操作策略在跨类别（unseen categories）泛化上存在严重不足。
*   **现有方法局限1：** 仅依赖视觉特征编码改进（如3D点云、等变网络、3D语义场），其泛化通常局限于外观相似的同类别物体，难以处理完全不同的物体。
*   **现有方法局限2：** 现有的affordance定义多为2D接触点、任务关键帧或价值图，无法完整表达“在哪里（where）”和“如何（how）”交互的静态与动态先验知识。
*   **人类启发：** 人类凭借丰富的物体级交互先验（即affordance），能够轻松将技能（如开柜子）迁移到外观或类别不同的新物体（如微波炉、冰箱）上。

## 核心贡献（创新点）
*   **提出AffordDP框架：** 首次将**可转移的3D静态接触点与动态接触后轨迹**作为统一affordance表征，并作为扩散策略的额外条件，以桥接训练与部署间的分布差距，实现跨类别泛化。
*   **端到端affordance转移：** 利用基础视觉模型（SD-DINOv2、Point-SAM）进行语义/几何对应估计，并独立计算6D变换矩阵的旋转与平移分量，实现了从源物体到目标物体的静态和动态affordance可靠迁移。
*   **自适应affordance引导采样：** 在扩散采样过程中引入基于末端执行器与静态affordance距离的自适应引导损失，将生成的动作序列逐步推向所需的操作轨迹，同时保持动作在有效流形内，提升了高精度操作的准确性。

## 方法详解
*   **Affordance定义与记忆库：** Affordance表示为 $\Phi = (c, \tau)$，其中 $c \in \mathbb{R}^3$ 是3D接触点（静态），$\tau \in \mathbb{R}^{3 \times N}$ 是接触后的轨迹点序列（动态）。构建包含任务名、affordance、CLIP图像特征 $z$ 和可操作物体点云 $\mathcal{P}$ 的记忆库 $\mathcal{M}$。
*   **静态affordance转移：** 使用超分辨率的SD-DINOv2特征图，在2D图像上通过特征相似度匹配找到源接触点对应的目标接触点2D坐标，再反投影到3D世界坐标得到 $c_{3D}^T$。
*   **动态affordance转移：** 平移向量 $\mathbf{t}$ 由源和目标3D接触点之差计算。旋转矩阵 $\mathbf{R}$ 通过将源和目标物体的可操作部分点云（由Point-SAM基于接触点提示分割得到）进行ICP配准来独立估计。最终通过 $\tau^T = \mathbf{R}\tau^S + \mathbf{t}$ 映射动态轨迹。
*   **条件扩散策略：** 策略条件 $\mathcal{C} = \{O_t, S_t, \Phi\}$，包含场景点云、机器人本体感知和affordance。使用DDIM进行去噪，训练损失为标准噪声预测均方误差 $\mathcal{L} = || \epsilon_k - \epsilon_\theta(a_t^k, k, f_t) ||^2$。
*   **自适应引导采样：** 定义引导损失 $\mathcal{L}_g$：当末端执行器位置 $p_{ee}$ 与静态affordance $c_{3D}$ 的距离小于阈值 $\theta$ 时，$\mathcal{L}_g = \| p_{ee} - c_{3D} \|_2$，否则为0。在DDIM采样步骤中，沿 $\mathcal{L}_g$ 的梯度方向调整动作估计，实现自适应引导。

## 实验与结果
*   **数据集与环境：** 仿真使用IsaacGym和GAPartnet物体；真机使用Franka机械臂。任务包括PullDrawer、OpenDoor、Pick&Place。
*   **评估基线：** Diffusion Policy (DP) [6] 和 3D Diffusion Policy (DP3) [41]。
*   **训练设置：** 分为单对象特定训练和统一多对象训练两种设置。
*   **关键结果（仿真-统一训练）：** 在PullDrawer任务中，对**未见类别**的泛化成功率，AffordDP达到73.3%，而DP和DP3仅为3.3%和3.3%。在OpenDoor任务的未见类别上，AffordDP达到26.7%，大幅优于DP的3.3%和DP3的5.6%。
*   **关键结果（真机）：** 在全部三种任务的**未见类别**测试中，AffordDP成功率在40%-50%之间，而DP在所有任务中均为0%成功，DP3最高仅为30%（OpenDoor未见类别）。
*   **消融实验：** 移除动态轨迹或引导模块均导致性能下降，验证了各组件的有效性。静态+动态+引导的组合效果最佳。

## 相关工作脉络
*   **Diffusion Policy [6] & DP3 [41]：** 本文的直接对比基线。DP使用2D图像，泛化受限；DP3使用点云改善了3D空间泛化，但仍局限于同类别或相似外观的实例。本文通过引入可转移的affordance实现了跨越类别的泛化。
*   **3D Diffusion Policy / Equibot / G3Flow / GenDP：** 这些工作通过引入3D点云、SIM(3)等变网络、3D语义场等来提升视觉特征编码的泛化能力，但其目标主要是改善同一类别内的位置、姿态、外观变化。本文的核心区别是利用**交互先验（affordance）的迁移**来实现跨类别泛化。
*   **Robo-ABC [18] & RAM [19]：** 利用基础视觉模型进行零样本affordance转移的代表作。但它们只能预测2D/带方向的3D接触点，且需要调用grasp generator或motion planner，在复杂高维操作中效果受限。本文进一步迁移了**轨迹级别的动态affordance**，并直接将其融入端到端的扩散策略中。
*   **RLafford [12] / Where2Act [24]：** 从大规模数据或强化学习中学习affordance的方法，开销大且任务特定。本文方法仅需少量域内演示数据，通过基础模型实现零样本/少样本的affordance转移。

## 局限性与未来方向
*   **基础模型的空间理解限制：** 当前使用的基础视觉模型（如SD-DINOv2）对空间信息的理解有限，可能限制对特定部件affordance的精确转移。
*   **复杂场景提取挑战：** 在难以建模或提取affordance的场景中（如高度可变形的物体、缺乏明确操作部分的物体），本文方法的有效性会受限。
*   **未来方向：** 提升基础视觉模型的空间感知能力；改进在复杂或非常规场景下的affordance提取与建模技术。

## 研究启发与可借鉴点
*   **Affordance作为泛化桥梁：** 将“可供性”从一种辅助预测信号提升为扩散策略的**核心条件变量**，并通过几何变换实现跨物体/类别的迁移，这一思路可有效缓解视觉分布偏移问题。
*   **解耦的6D变换估计策略：** 独立估计平移（基于接触点）和旋转（基于部分点云ICP配准）比整体实例级配准对形状差异更鲁棒，该策略可借鉴于其他需要跨物体属性迁移的任务。
*   **采样阶段的自适应引导：** 在扩散采样过程中，引入一个可微的、基于任务关键几何约束的自适应引导损失，可以弥补纯数据驱动策略在高精度任务上动作不精确的缺陷，且无需重新训练模型。
*   **统一多对象训练范式：** 论文展示了“统一策略训练”（用多个不同物体训练）比“单对象特定训练”更能激发出跨类别的泛化潜力，这与VLA模型中大规模多样化数据训练的思路一致。

## 关键术语表
*   **Affordance (可供性)：** 指物体本身所蕴含的、允许或提示特定交互方式的环境属性，回答了“在哪里（where）接触”和“如何（how）操作”的问题。
*   **Static Affordance (静态可供性)：** 以3D接触点形式表示的操作关键位置信息。
*   **Dynamic Affordance (动态可供性)：** 以接触后末端执行器轨迹点序列形式表示的操作过程先验信息。
*   **6D Transformation Matrix (6D变换矩阵)：** 由旋转矩阵和平移向量组成，用于将源物体的点集或轨迹映射到目标物体坐标系下的几何变换。
*   **Adaptive Affordance Guidance (自适应affordance引导)：** 在扩散去噪采样阶段，当末端执行器接近目标接触点时，动态施加一个将其拉向该接触点的梯度引导力。
*   **Unified Policy Training (统一策略训练)：** 使用多个不同（但属于同任务类别）的物体实例数据进行策略训练，以学习更通用的任务策略，而非过拟合到单一物体。

## 可复现要素
*   **数据集：** GAPartnet [11]（仿真）。真机使用自行收集的teleoperation数据。论文未明确提及GAPartnet是否完全公开于本工作。
*   **代码/权重：** 项目主页为 https://afforddp.github.io/，通常此类工作会开源代码，但需访问主页确认。论文中未明确声明代码和模型权重已开源。
*   **关键超参：** ICP配准迭代次数、点云采样/降采样参数、diffusion去噪步数、引导阈值 $\theta$ 和权重 $\gamma$、运动规划器CuRoBo的参数等。论文正文中**未提及**具体数值，可能在补充材料中。
