---
title: "InterAct-Advancing-Large-Scale-Versatile-3D-Human-Object-Int"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Xu_InterAct_Advancing_Large-Scale_Versatile_3D_Human-Object_Interaction_Generation_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 07:35:05"
---

# 论文速读：InterAct-Advancing-Large-Scale-Versatile-3D-Human-Object-Int

## 一句话总结
本文提出 InterAct，首个面向序列式3D人-物交互（HOI）生成的大规模统一基准，通过数据整合、统一优化去伪与接触不变性数据增强，将有效数据扩展至 30.70 小时并附带 48,630 条细粒度文本标注；在此基础上定义 6 项生成任务，并提出基于 Marker 表征与多任务接触联合建模的扩散模型，在多项 HOI 生成基准上达到 SOTA。

## 研究问题与动机
- **数据集规模小且异构**：现有 HOI 数据通常仅含数小时片段，人体表示（SMPL/SMPL-H/SMPL-X）、坐标系、物体种类与标注格式高度不一致，难以直接合并训练。
- **标注匮乏且粗糙**：多数数据集缺乏细粒度语义描述，仅有的文本标注无法覆盖交互状态、参与部位与动态过程，阻碍视觉-语言对齐生成。
- **Motion Capture 伪影普遍**：现有数据中常出现人体-物体穿透、悬浮接触、手部姿态失真及明显运动抖动，直接用于训练会导致生成模型学习错误的物理与空间先验。
- **缺乏统一基准与公平对比**：不同工作往往针对不同子任务或特定物体设计，评估指标与切分不统一，难以系统衡量数据质量与模型泛化能力。

## 核心贡献（创新点）
1. **构建 InterAct / InterAct-X 大规模统一基准**：整合 7 个异构 MoCap 数据集，标准化至 21.81 小时（增强后 30.70 小时），并提供 34,050 / 48,630 条高质量文本标注与 217 类物体，规模与标注完备性大幅领先现有资源。
2. **提出统一优化去伪与合成增强框架**：解耦全身修正与手部修正，结合接触促进、穿透惩罚与运动先验损失；基于“接触不变性”原理通过物体位移+重优化生成高质量合成数据，验证了对现有数据集的“无采集扩展”可行性。
3. **设计 Marker 统一表征与多任务接触联合建模**：以固定顶点集合替代传统关节点表征人体，显式编码人体-物体距离向量（ω）；将 5 类运动学生成任务统一为共享扩散骨干的多任务学习，显著提升空间一致性与跨任务泛化。
4. **建立 6 项标准化 Benchmark 与评估协议**：覆盖文本/动作条件生成、双向生成、交互预测与物理模仿，提供可扩展的基线（HOI-Diff、OMOMO、InterDiff、PhysHOI 等）与完整消融实验，填补领域统一评测空白。

## 方法详解
- **数据整合与文本标注**：从 GRAB、BEHAVE、InterCap、Chairs、HODome、OMOMO、IMHD 六个来源合并数据；采用“人工详细标注 → GPT-4 改写/简化 → GPT-4 分类至 15 类动作标签”的两阶段流程，OMOMO 子集沿用其原有标注。
- **统一表征设计**：人体统一转换为 SMPL 坐标系下的 Marker 集合（SMPL-H 与 SMPL-X 间映射误差<1 cm）；物体几何采用 Basis Point Set (BPS)；引入关系向量 $\omega$（每个 Marker 指向最近物体表面的有符号距离向量），作为多任务共享辅助输出。
- **交互修正（Optimization-based Correction）**：
  - **手部修正**：接触损失 $E_{\text{cont}} = \sum_{i=1}^{L} c_i \sum_j d_j[i]$（$c_i$ 为接触指示，$d_j[i]$ 为手顶点到物体表面最近距离），配合穿透损失、时间平滑损失、手指 RoM 先验损失，防止过度接触导致生物不合理姿态。
  - **全身修正**：在以上损失基础上加入重建损失，接触/穿透计算扩展至全身顶点，去除 RoM 先验（重建损失已足够约束整体姿态）。
- **交互增强（Contact Invariance Augmentation）**：(1) 对物体轨迹施加均匀随机平移；(2) 以接触一致性损失 $E_{\text{align}} = \sum_i \sum_{j,k} \frac{1}{(\hat{D}_{jk}+\epsilon)^2} |\hat{D}_{jk} - D_{jk}|^2$ 重优化人体运动，保留初始近距离顶点对的相对距离；(3) 过滤初始位移过大、穿透严重或优化发散的低质量样本。
- **统一多任务生成模型**：以 HOI-Diff 为骨干，实施四项改进：① 用序列级 InfoNCE 对比学习预训练交互感知文本编码器，替代原始 CLIP；② 用 BPS 替代 PointNet++ 编码物体形状；③ 将接触向量 $\eta$ 作为额外输出参与多任务联合训练；④ 在去噪过程中引入接触预测作为 Classifier Guidance。六大任务形式化定义为：$\mathcal{G}_{t2i}(t)\mapsto\langle h,o\rangle$、$\mathcal{G}_{a2i}(a)\mapsto\langle h,o\rangle$、$\mathcal{G}_{o2h}(o)\mapsto h$、$\mathcal{G}_{h2o}(h)\mapsto o$、$\mathcal{G}_{p2f}(\langle h_p,o_p\rangle)\mapsto\langle h_f,o_f\rangle$、$\mathcal{G}_{i2f}(\langle h,o\rangle)\mapsto f$（PD 控制目标）。

## 实验与结果
- **数据集统计**：InterAct 基础版 21.81 小时 / 11,350 clip / 34,050 文本 / 217 类物体；InterAct-X 增强版 30.70 小时 / 16,201 clip / 48,630 文本。
- **修正与增强质量**：Table 2 显示修正后穿透深度（Pene）显著降低，接触比例（Cont Ratio）提升；双盲用户研究中约 39.7% 评审将修正数据评为最优，增强数据评分与修正数据相当。
- **文本条件生成（Table 3）**：FID 降至 1.
