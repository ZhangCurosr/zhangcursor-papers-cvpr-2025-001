---
title: "Improve-Representation-for-Imbalanced-Regression-through-Geo"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Dong_Improve_Representation_for_Imbalanced_Regression_through_Geometric_Constraints_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 07:31:48"
---

# 论文速读：Improve-Representation-for-Imbalanced-Regression-through-Geo

## 一句话总结
本文首次将特征空间均匀性（uniformity）引入深度不均衡回归（DIR），通过提出包裹损失与同质性损失对连续有序表征轨迹施加几何约束，并配合代理驱动表示学习（SRL）框架，显著提升中/少样本区域的回归精度；同时首创不均衡算子学习（IOL）基准任务。

## 研究问题与动机
- **表征分布未被建模**：现有DIR工作主要聚焦训练无偏回归器或标签-特征关系，却忽视了网络输出在完整特征空间（单位超球面）中的分布形态。
- **分类方法不可直接迁移**：分类领域的均匀性提升依赖离散簇划分，而回归要求表征具有连续性与有序性，直接套用会导致特征坍缩或边界模糊。
- **单batch无法覆盖全局**：几何约束需作用于全标签范围的有序轨迹，但随机mini-batch往往遗漏部分bin，导致正则无法有效计算。
- **连续有序轨迹缺乏量化标准**：既往方法未定义如何衡量一条连续路径对特征空间的“填充效率”与“沿迹均匀性”。

## 核心贡献（创新点）
- **首次为DIR引入几何均匀性正则**：提出包裹损失与同质性损失，从轨迹覆盖度与沿迹平滑性两个维度定义回归均匀性，与以往侧重误差重加权或对比学习的设计形成本质差异。
- **SRL跨epoch代理机制**：通过记忆并动量更新缺失bin质心，将全局几何约束安全注入本地批次训练，解决了单batch标签覆盖不全的工程瓶颈。
- **首创IOL任务**：在算子学习（Operator Learning）中人为构造输出域采样不均衡，将DIR研究边界拓展至函数空间映射。
- **正交可组合性验证**：证明SRL可与LDS+FDS、RankSim、BalancedMSE等现有DIR方法叠加使用且稳定增益，体现方法设计的通用性。

## 方法详解
- **表征归一化与分箱**：网络输出 $z_i=f(x_i)$ 投影至单位超球面（$\|z_i\|=1$），标签空间划分为 $K$ 个bin，batch内同bin表征取平均得质心 $c_y$。
- **包裹损失 $\mathcal{L}_{\mathrm{env}}$**：定义潜轨迹 $l$ 的 $\epsilon$-管状邻域 $T(l,\epsilon)$，目标为最大化 $\mathrm{vol}(T(l,\epsilon))/\mathrm{vol}(\mathcal{U})$。由于管体积难以直接求导，采用蒙特卡洛采样 $N$ 个均匀点，将硬判断松弛为 $\max_y \{p_i \cdot l(y)\} > \epsilon$ 的soft余弦相似度最大化。
- **同质性损失 $\mathcal{L}_{\mathrm{homo}}$**：离散形式为 $\sum_{k=1}^{K-1} \|l(y_{k+1})-l(y_k)\|^2 / (y_{k+1}-y_k)$，等价于惩罚轨迹弧长；定理1证明其最小化当且仅当表征沿轨迹均匀分布（$\|\nabla_y l(y)\|=c$）。
- **对比损失 $\mathcal{L}_{\mathrm{con}}$**：以同bin质心为正、其余质心为负计算InfoNCE，促使个体表征向对应bin中心收敛并保持类间分离。
- **SRL代理更新**：当前batch缺失bin的质心由上一epoch存储的质心填补，构成完整代理 $S$；每轮结束以动量 $S^{e+1} \leftarrow \alpha S^e + (1-\alpha)\hat{S}^e$ 平滑更新，防止轨迹跳变。
- **总损失**：$\mathcal{L}_\theta = \mathcal{L}_{\mathrm{reg}} + \lambda_e \mathcal{L}_{\mathrm{env}} + \lambda_h \mathcal{L}_{\mathrm{homo}} + \mathcal{L}_{\mathrm{con}}$，最优超参 $\lambda_e=\lambda_h=0.1$。

## 实验与结果
- **数据集与划分**：AgeDB-DIR、IMDB-WIKI-DIR、STS-B-DIR、自构建UCI-DIR（Airfoil/Abalone/Real Estate/Concrete）、新基准OL-DIR（线性积分算子与非线性PDE）；按训练样本数划分Many(>100)/Med(20-100)/Few(<20)区域。
- **最强结果**：SRL在全部5个数据集、所有shot分区上均优于VANILLA、LDS+FDS、RankSim、BalancedMSE、Ordinal Entropy与ConR。AgeDB-Few MAE 12.61→9.81，IMDB-WIKI-Few MAE 26.11→22.78，STS-B-Few MSE 1.075→0.877；UCI-DIR四个表格任务全面领先。
- **IOL结果**：线性算子Few-shot MAE 27.00→9.33，非线性Few-shot MAE 19.77→17.00，显著优于基线。
- **定量分析**：几何约束使少样本区域特征占比大幅提升（AgeDB: 1.98%→15.80%，上限23%；STS-B: 4.52%→22.39%，上限38%），与性能提升强相关。
- **消融**：缺少 $\mathcal{L}_{\mathrm{env}}$ 或 $\mathcal{L}_{\mathrm{homo}}$ 均导致性能下降；$N$ 过小会使包裹损失估计不准；与现有DIR方法结合仍带来稳定增益（Table 5）。

## 相关工作脉络
- **不均衡分类均匀性**（Wang & Isola [22], PaCo [5], TSC [12], BCL [33]）：关注离散簇的均匀展开，本文将其思想抽象为连续轨迹覆盖问题，摒弃了簇假设。
- **深度不均衡回归先验**（LDS+FDS [25], RankSim [6], BalancedMSE [16], ConR [10], RNC [27], SupReMix [24]）：多聚焦标签密度平滑、对比学习或MSE平衡，本文首次从特征空间几何填充角度补全DIR表征研究空白。
- **算子学习**（Neural Operator [11], DeepONet [14]）：传统假设输出域均匀采样，本文通过控制采样密度构造IOL，为科学机器学习引入不均衡评估维度。
- **表示学习正则化**：将几何先验（弧长/覆盖）与InfoNCE结合，区别于分类中对齐-均匀分解范式，适配回归的连续有序约束。

## 局限性与未来方向
- **仅支持标量连续标签**：无法直接推广至高维标签（如图像、视频、结构化输出）的回归任务。
- **代理维护引入额外内存**：需跨epoch存储质心并进行动量平滑，工程复杂度略高于纯batch-wise方法。
- **未来方向**：拓展至多维/分层标签的均匀性建模；探索与方差估计（如VIR [23]）、不确定性量化方法的融合；将IOL推广至更复杂的偏微分方程求解与物理仿真场景。

## 研究启发与可借鉴点
- **“潜轨迹+管状邻域”几何建模**：为连续属性预测（时序、流形、拓扑）提供了刻画表征覆盖度的新范式，可迁移至分布外检测或领域泛化。
- **跨epoch动量代理机制**：解决局部采样无法覆盖全局分布的经典痛点，设计可复用于长尾生成、持续学习、联邦学习等需全局正则的场景。
- **几何正则与对比损失的解耦组合**：本文证明 $\mathcal{L}_{\mathrm{env}}$/$\mathcal{L}_{\mathrm{homo}}$ 与 $\mathcal{L}_{\mathrm{con}}$ 正交协同，未来可移植至分位数回归、多任务回归或半监督回归中以增强表征质量。
- **IOL采样构造思路**：通过控制输出域采样密度制造region-wise不均衡，为PINN/ML4Sci的不均衡训练与评估提供了可复用的基准构建模板。

## 关键术语表
- **Uniformity（均匀性）**：原指分类中特征簇的均匀展开，本文重定义为回归潜轨迹对超球面的覆盖效率与沿迹
