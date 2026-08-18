---
title: "CORE4D-A-4D-Human-Object-Human-Interaction-Dataset-for-Colla"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_CORE4D_A_4D_Human-Object-Human_Interaction_Dataset_for_Collaborative_Object_REarrangement_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:58:21"
---

# 论文速读：CORE4D-A-4D-Human-Object-Human-Interaction-Dataset-for-Colla

## 一句话总结
本文提出了 CORE4D，一个大规模的人-物-人（HOH）协作物体重排 4D 数据集，通过结合真实 mocap 捕捉与创新的迭代协作重定向合成策略，构建了包含 11K 序列和 3K 物体形状的混合数据集，并在运动预测、交互合成及人形机器人技能学习上进行了基准测试与迁移验证。

## 研究问题与动机
1. **数据匮乏**：现有数据集多聚焦单人交互或简单的双人交接（handover），缺乏针对复杂多人协作物体重排行为的大规模、高质量、类别级数据集。
2. **采集瓶颈**：仅靠视觉跟踪难以解决多人协作中严重的自遮挡与姿态歧义问题，而纯 mocap 数据成本高昂，难以覆盖重物重排所需的海量物体几何变体。
3. **时空解耦假设**：作者观察到多人协作中，**时间维度的协作动态模式**变化多样，而**空间维度的握持/接触关系**在同形类别物体间具有强同质性，这为利用算法扩展空间多样性提供了理论依据。

## 核心贡献（创新点）
1. **构建 CORE4D 混合数据集**：包含 1K 真实捕捉序列（CORE4D-Real）与 10K 合成序列（CORE4D-Synthetic），覆盖 37 个真实模型与 3K 虚拟物体形状、多种协作模式及不同复杂度的 3D 场景。
2. **提出迭代协作重定向策略**：结合 mocap 真实捕捉与自动空间重定向，在保留真实时间协作模式的同时，将交互泛化至 novel 物体几何，避免了针对数千种物体逐一捕捉的成本。
3. **设计人以接触选择机制**：突破传统以物体为中心的重定向局限，引入 human pose discriminator 与 beam search 迭代优化接触约束，有效解决了大几何差异物体上的接触语义一致性问题。
4. **提供双任务基准与机器人应用验证**：在运动预测与交互合成任务上建立了 baselines，揭示了现有生成模型在新物体泛化性与运动自然度上的挑战，并成功将数据迁移至 Unitree H1 人形机器人的箱式搬运模仿学习任务。

## 方法详解
**1. 真实数据采集与标注 (CORE4D-Real)**
* **采集系统**：采用惯性-光学混合 mocap 系统（12 红外相机 + 每人身穿 8 追踪器服装与数据手套），配合 4 个 allocentric Kinect Azure DK（RGB-D）与 1 个 egocentric Osmo Action3 摄像头，运行于 15 FPS。
* **人体网格拟合**：将 BVH 骨架重定向至 SMPL-X 模型，优化形状参数 $\beta \in \mathbb{R}^{10}$ 与全身姿态 $\theta \in \mathbb{R}^{159}$。损失函数 $\mathcal{L}$ 综合了正则项、关节 3D 位置/旋转对齐、手指定位、时序平滑及手-物接触约束。
* **物体与分割标注**：物体表面粘贴 4-5 个标记点计算 6D 位姿；37 个刚性物体经工业 3D 扫描仪（最高 100K 三角面）建模并手动去噪；2D 分割通过 DEVA 生成初始掩码，再渲染 mesh 选取最大 IoU 实例。

**2. 协作重定向合成管线 (CORE4D-Synthetic)**
 pipeline 包含三个核心组件，实现从源物体到目标 ShapeNet 物体的交互迁移：
* **Object-centric Contact Retargeting**：在所有物体上训练 DeepSDF 学习隐式符号距离场（SDF）。对源物体 latent $o_s$ 与目标物体 latent $o_t$ 进行线性插值，生成 $N$ 个中间 mesh 序列 $\mathcal{M}$，通过最近邻搜索将接触点逐级传播至目标物体，构建接触候选池。
* **Contact-guided Interaction Retargeting**：联合优化物体运动 $\{R_o, T_o\}$ 与人体参数 $\{\theta_{1,2}, T_{1,2}, O_{1,2}\}$。目标函数包含保真度损失 $\mathcal{L}_f$（保持原始运动趋势）、地面穿透限制 $\mathcal{L}_{spat}$、时序平滑 $\mathcal{L}_{smooth}$ 及接触约束 $\mathcal{L}_c$。
* **Human-centric Contact Selection**：训练基于 ranking loss 的人体姿势鉴别器，正样本取自 CORE4D-Real，负样本为添加 6D 位姿噪声 $\Delta(\alpha,\beta,\gamma,x,y,z)$ 后生成的运动。利用 beam search 迭代更新接触约束，在穿透惩罚约束下选取鉴别器得分最高的候选，实现多轮精修。

**3. 数据集多样性设计**
* **物体几何**：6 大类（box, board, barrel, stick, chair, desk），涵盖简单对称体与复杂非对称体。
* **协作模式**：5 种（带/不带目标知识的协作搬运、交接、加入、中途退出）。
* **3D 场景**：3 级复杂度（无障碍、单障碍、多障碍）。

## 实验与结果
* **数据划分**：训练集由随机采样的真实物体及其对应合成数据组成；测试集 S1 含已知物体，S2 含未见物体（合成数据不进入测试集以避免算法偏差）。
* **运动预测基准**：评估 MDM、InterDiff、CAHMP。S2（新物体）上人体关节误差 $J_e$ 从 ~169-170 上升至 ~186，而物体位姿误差相对稳定，表明**新物体形状下的协作 motion forecasting 泛化仍是显著挑战**。
* **交互合成基准**：评估 MDM、OMOMO、CHOIS。合成结果的 FID 高于真实数据，且多人体协作建模误差大于单人交互（Li et al. [50]），说明**多主体协同动力学与自然度生成难度更高**。
* **重定向消融（Table 4）**：
  * 移除候选池（仅用源轨迹接触点，Abl.1）：物理合理性与用户偏好骤降，证明候选池对大几何差异物体的必要性。
  * 移除鉴别器（Abl.2）：性能明显下降；改用纯接触准确度选择（Abl.3）的用户偏好亦逊于鉴别器方案。
  * 移除接触更新（Abl.4）：穿透距离略增，用户感知质量下降。
  * **真实性对比**：43% 用户认为合成与真实数据质量相当，14% 甚至更偏好合成数据。
* **数据增强效果（Table 5）**：使用 5K 训练集（1K Real + 4K Synthetic）时，CAHMP 在 S2 上的 $J_e$ 降至 116.2，$T_e$ 降至 112.1，$R_e$ 降至 6.99，显著优于仅用 1K 真实数据的 baseline。
* **人形机器人迁移（Table 6）**：在 Isaac Gym 中搭建 Unitree H1 搬运箱子任务。模仿学习（IL）方法 HumanPlus 成功率达 21.0%，HST+ACT 达 26.5%，而免演示的 PPO 为 0.0，验证了 CORE4D 对机器人协作技能学习的促进作用。

## 相关工作脉络
1. **HOH / CoChair / HOI-M³**：虽涉及多人交互，但缺乏类别级物体多样性与自/他双视角融合；CORE4D 首次系统性地覆盖了协作重排这一细粒度任务类型。
2. **OakInk [119] / Tink 等重定向工作**：以物体表面几何对应为核心，受限于相似拓扑与尺度；本文转换视角至人本位接触选择，突破了复杂形变物体的重定向瓶颈。
3. **MDM [90] / InterDiff [110]**：在单人 HOI 生成上表现优异，但在 CORE4D 的多主体耦合与新物体泛化设定下出现性能回落，确立了新的 benchmark 难度基线。
4. **HumanPlus [25] / ACT [134]**：主流 humanoid 模仿学习框架；本文将其应用域从纯 locomotion 拓展至 multi-human-collaboration-derived  manipulation，证明了跨模态数据迁移的可行性。

## 局限性与未来方向
1. **场景局限**：受限于 mocap 设备，目前仅包含室内场景，未覆盖室外非结构化环境。
2. **视觉模态缺失**：CORE4D-Synthetic 缺乏真实的 RGB/RGB-D 视觉信号；未来可将真实世界视频重定向至合成运动，实现视觉-动作模态的完整对齐。
3. **多机器人扩展待探索**：当前仅验证了单机器人搬运，多机器人协同协作技能学习仍有较大挖掘空间。

## 研究启发与可借鉴点
1. **时间与空间解耦的数据构建范式**：对于高
