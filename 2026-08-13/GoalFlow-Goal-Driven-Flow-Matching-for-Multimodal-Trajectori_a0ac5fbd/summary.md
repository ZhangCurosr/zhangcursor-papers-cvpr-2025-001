---
title: "GoalFlow-Goal-Driven-Flow-Matching-for-Multimodal-Trajectori"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Xing_GoalFlow_Goal-Driven_Flow_Matching_for_Multimodal_Trajectories_Generation_in_End-to-End_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 07:31:16"
field: "自动驾驶轨迹规划"
keywords: ["端到端自动驾驶", "多模态轨迹生成", "Flow Matching", "目标点引导", "Rectified Flow", "Navsim"]
innovations: ["目标点驱动的双阶段轨迹生成框架（目标点构建+Flow Matching生成）", "基于密集词汇库与双评分机制的目标点筛选方法", "单步推理的Rectified Flow轨迹生成实现实时部署"]
benchmarks: ["Navsim"]
---

# 论文速读：GoalFlow-Goal-Driven-Flow-Matching-for-Multimodal-Trajectori

## 一句话总结
本文提出了 GoalFlow，一种面向端到端自动驾驶的目标驱动流匹配方法，通过构建密集目标点词汇库并结合场景评分机制筛选最优目标点，再用 Rectified Flow 高效生成高质量多模态轨迹，在 Navsim 基准上以 PDMS 90.3 取得 SOTA 性能。

## 研究问题与动机
- **轨迹发散与生成质量低**：基于 Diffusion 的多模态轨迹生成方法（如 Diffusion-ES、MotionDiffuser）缺乏约束时易产生高度发散的轨迹，模态边界模糊，且需要复杂的 HD map 对齐后处理。
- **引导信息偏差导致低质量轨迹**：现有方法（如 VAD、SparseDrive）依赖离散指令或预定义目标点集指导轨迹生成，但当引导信息与真实值差距较大时，会生成低质量轨迹。
- **忽略可行驶区域约束**：现有端到端系统过度关注碰撞率和 L2 误差，忽略了车辆是否保持在可行驶区域内（DAC 指标被忽视）。
- **推理效率不足**：传统 Diffusion 模型需要多步去噪，推理延迟高，难以满足自动驾驶实时性需求。

## 核心贡献（创新点）
1. **目标点驱动的高约束轨迹生成框架**：将规划任务解耦为"目标点构建→目标点引导轨迹生成"两步，用精确的目标点约束生成过程，与无约束 Diffusion 方法本质不同。
2. **无需 HD map 的密集目标点词汇库构建**：借鉴 VADv2 思想，对轨迹终点进行聚类生成大规模目标点词汇（4096/8192个），避免依赖昂贵的 HD map 车道信息。
3. **双评分机制筛选最优目标点**：设计 Distance Score（接近 GT 终点）和 Drivable Area Compliance Score（位于可行驶区域）联合评分，从词汇中选出最适配目标点，显著提升 DAC 得分。
4. **Flow Matching 替代 Diffusion 实现单步推理**：采用 Rectified Flow 建模轨迹分布转移，推理时仅需单步去噪即可达到接近最优的性能，相比多步 Diffusion 推理速度提升约 17 倍。
5. **阴影轨迹辅助的目标点可靠性检测**：通过遮蔽目标点生成阴影轨迹，若阴影轨迹偏离主轨迹则判定目标点不可靠并切换输出，增强系统鲁棒性。

## 方法详解
### 整体架构
GoalFlow 分为三部分：感知模块（Perception Module）、目标点构建模块（Goal Point Construction Module）、轨迹规划模块（Trajectory Planning Module）。

### 感知模块
- 采用 Transfuser 架构融合相机图像（I）和 LiDAR 数据（L）
- 通过多层 Transformer 交叉注意力融合不同模态特征，输出 BEV 特征 $F_{bev}$
- 附加监督：HD map 交叉熵损失 $L_{HD}$、3D bounding box 分类损失 $L_{bbox}$ 和位置损失 $L_{loc}$

### 目标点构建模块
- **目标点词汇库** $\mathbb{V} = \{g_i\}^{N}$：对训练数据中轨迹终点 $(x_i, y_i, \theta_i)$ 进行聚类得到 N 个中心点，通常 $N = 4096$ 或 $8192$
- **双评分机制**：
  - Distance Score：$\delta_i^{dis} = \frac{\exp(-\|g_i - g^{gt}\|_2)}{\sum_j \exp(-\|g_j - g^{gt}\|_2)}$，衡量与 GT 终点的接近程度
  - DAC Score：基于"影子车辆"的四个角点是否全部落在可行驶区域多边形内部，输出二进制值
  - 最终得分：$\hat{\delta}_i^{final} = w_1 \log \hat{\delta}_i^{dis} + w_2 \log \hat{\delta}_i^{dac}$
- 选择得分最高的目标点 $g^*$ 用于轨迹生成

### 轨迹规划模块（Flow Matching）
- **Rectified Flow 原理**：从标准正态分布 $\pi_0 = \mathcal{N}(0, I)$ 线性映射到目标轨迹分布 $\pi_1$，中间状态 $x_t = (1-t)x_0 + t\tau^{norm}$
- 网络预测方向场 $\hat{v}_t = \mathcal{G}(F_{env}, F_{goal}, F_{traj}, F_t)$
- 多步重构轨迹：$\hat{\tau}^{norm} = x_0 + \frac{1}{n}\sum_{i=1}^{n} \hat{v}_{t_i}$，再反归一化得到 $\hat{\tau}$
- **训练采用 classifier-free guidance**：随机遮蔽条件特征增强鲁棒性

### 轨迹选择机制
- 轨迹评分函数：$f(\hat{\tau}_i) = -\lambda_1 \Phi(f_{dis}(\hat{\tau}_i)) + \lambda_2 \Phi(f_{pg}(\hat{\tau}_i))$，权衡轨迹到目标点距离与自车进度
- 生成 M 条候选轨迹（128/256条），通过评分器选出最优轨迹

### 训练损失
$$L_{perception} = 10 L_{HD} + 1 L_{bbox} + 10 L_{loc}$$
$$L_{goal} = 1 \cdot L_{dis} + 0.005 \cdot L_{dac}$$
$$L_{planner} = |\mathbf{v}_t - \hat{\mathbf{v}}_t|$$

## 实验与结果
- **数据集**：OpenScene → Navsim 环境（1192 trainval + 136 test 场景，2Hz，超10万样本）
- **评估指标**：PDMS = $S_{NC} \times S_{DAC} \times S_{TTC} \times \frac{5S_{EP}+5S_{CF}+2S_{DDC}}{12}$
- **主要结果**：
  | 方法 | $S_{NC}$ | $S_{DAC}$ | $S_{TTC}$ | $S_{EP}$ | PDMS |
  |------|----------|-----------|-----------|----------|------|
  | TransFuser | 97.7 | 92.8 | 92.8 | 79.0 | 84.0 |
  | PARA-Drive | 97.9 | 92.4 | 93.0 | 79.3 | 84.0 |
  | **GoalFlow (Ours)** | **98.4** | **98.3** | **94.6** | **85.0** | **90.3** |
  | GoalFlow† (GT目标点) | 99.8 | 97.9 | 98.6 | 85.4 | 92.1 |
  | Human | 100 | 100 | 100 | 87.5 | 94.8 |

- **消融实验关键结论**：
  - 距离评分引入带来最大提升（M0→M1: +2.9 PDMS）
  - DAC 评分进一步提升 DAC 指标（M1→M2: +0.9）
  - 轨迹选择器贡献最后提升（M2→M3: +0.9）

- **推理步数鲁棒性**：从20步降至1步仅损失1.6% PDMS，推理时间从177.8ms降至10.4ms（约1/17）
- **噪声方差影响**：$\sigma = 0.1$ 时最优；$\sigma > 0.1$ 时性能骤降（$\sigma=0.3$ 时 PDMS 仅18.8）
- **模型缩放**：隐藏维度从256增至1024，PDMS 从86.5提升至89.4

## 相关工作脉络
1. **Transfuser [3]**：早期多模态融合感知+规划方法，GoalFlow 沿用其 Transfuser 架构作为感知骨干，但将规划部分从回归改为生成式 Flow Matching。
2. **VAD/VADv2 [17, 1]**：使用离散驾驶指令（左转/直行/右转）引导多模态轨迹生成；GoalFlow 借鉴其空间离散化思想构建目标点词汇库，但采用连续目标点而非离散指令。
3. **SparseDrive [27]**：基于稀疏向量化场景表示，通过聚类导航点引导轨迹；GoalFlow 不依赖 HD map，通过数据驱动聚类生成更密集的目标点集合。
4. **MotionDiffuser [18]**：使用扩散模型生成多模态轨迹，以 GT 终点为强约束；GoalFlow 使用学习到的目标点而非 GT 终点，且用 Flow Matching 替代 Diffusion 提升效率。
5. **Diffusion-ES [32]**：无约束扩散轨迹生成，需 HD map 后处理对齐；GoalFlow 通过目标点约束从根本上避免发散问题，无需外部地图对齐。
6. **GoalGAN [8]**：先预测目标点再用 GAN 生成轨迹；GoalFlow 同样采用目标点引导，但使用 Flow Matching 替代 GAN，推理更高效。
7. **UniAD [15] / PARA-Drive [30]**：统一多任务端到端框架；GoalFlow 聚焦于规划模块的生成式改进，可与这些感知框架结合。

## 局限性与未来方向
- **目标点词汇库依赖数据分布**：聚类得到的目标点词汇可能无法覆盖所有极端场景，对未见过的场景泛化能力存疑。
- **DAC 评分使用二分逻辑**：当前 DAC 评分为硬约束（0/1），可能导致边缘可行驶区域的目标点被错误排除。
- **未集成视频时序信息**：当前方法仅使用单帧多视角图像，未充分利用时序信息建模轨迹连续性。
- **未来方向**：探索不同引导信息（如自然语言指令、意图标签）对多模态轨迹生成的影响（论文自述）。

## 研究启发与可借鉴点
1. **任务解耦策略**：将复杂规划任务分解为"目标点预测→目标点引导轨迹生成"两步，降低生成模型的条件复杂度，此思路可迁移到其他生成式决策任务。
2. **Flow Matching 在自动驾驶中的应用**：证明 Rectified Flow 在轨迹生成中的高效性（单步推理），为实时生成式模型部署提供可行路径。
3. **双评分目标点筛选机制**：结合"接近性"与"合规性"双重约束筛选引导信息，可有效平衡轨迹多样性与安全性，适用于其他条件生成场景。
4. **影子轨迹辅助的鲁棒性设计**：通过遮蔽关键条件生成替代输出来检测引导信息可靠性，是一种轻量级的不确定性量化方法。
5. **无 HD map 的替代方案**：通过数据驱动的密集词汇库替代 HD map 依赖，对无高精地图区域部署具有实用价值。

## 关键术语表
**Flow Matching / Rectified Flow**：一种生成建模方法，学习从简单分布到数据分布的可逆变换，相比 Diffusion 推理步骤更少、路径更直。
**Goal Point Vocabulary**：通过聚类轨迹终点构建的大规模目标点候选集合（4096/8192个），作为轨迹生成的条件引导。
**DAC (Drivable Area Compliance)**：可行驶区域合规性指标，衡量轨迹是否完全位于合法驾驶区域内。
**PDMS (Performance Distance Metrics Score)**：Navsim 基准的综合评估分数，由无责碰撞率、可行驶区域合规性、碰撞时间、自车进度等加权计算。
**Shadow Trajectory**：遮蔽目标点条件后重新生成的轨迹，用于检测目标点可靠性，若偏离过大则切换输出。
**Classifier-Free Guidance**：训练时随机遮蔽条件特征以提升模型鲁棒性，推理时结合条件/无条件输出增强生成质量。
**BEV (Bird's Eye View)**：鸟瞰图特征表示，融合多模态感知信息后的统一场景表征。

## 可复现要素
- **数据集**：OpenScene / Navsim（论文声明代码已开源：https://github.com/YvanYin/GoalFlow）
- **代码**：已开源
- **权重**：未明确提及是否公开预训练权重
- **关键超参**：
  - 目标点词汇量 $N = 4096$ 或 $8192$
  - 噪声标准差 $\sigma = 0.1$
  - 生成候选轨迹数 $M = 128$ 或 $256$
  - 推理步数 $T = 1 \sim 20$（推荐5步平衡速度/质量）
  - 训练 GPU：4节点 × 8× RTX 4090/3090
  - 损失权重：$w_1=10.0, w_2=1.0, w_3=10.0, w_4=1.0, w_5=0.005$
