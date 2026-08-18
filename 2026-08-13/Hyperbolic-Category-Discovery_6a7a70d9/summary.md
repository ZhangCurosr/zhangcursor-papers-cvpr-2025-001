---
title: "Hyperbolic-Category-Discovery"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_Hyperbolic_Category_Discovery_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 07:12:40"
---

# 论文速读：Hyperbolic-Category-Discovery

## 一句话总结
本文针对广义类别发现（GCD）任务，首次将双曲几何系统引入表征学习与分类环节，提出轻量即插即用的 HypCD 框架，通过 Poincaré 球映射与距离-角度混合对比损失有效捕获样本的层次化部件结构，在 SSB 及通用基准上持续刷新 SOTA 并显著缩小新旧类别性能差距。

## 研究问题与动机
- **核心问题**：GCD 需要在仅部分标注的数据中完成已知与未知类别的联合分类/聚类，现有方法普遍在欧氏或球面空间操作，难以充分建模物体部件的层次关系与多尺度差异。
- **球面/欧氏空间的几何局限**：两者体积随半径呈多项式或恒定增长，对树状/层次化样本的嵌入容量不足，导致类别内部件关系与跨尺度变化被压制。
- **已有层次建模的缺口**：现有 GCD 层次化工作多关注“类别间”预定义语义树，缺乏对“类别内”非刚性、多尺度部件组合的灵活刻画。
- **双曲空间的天然优势**：体积随半径指数增长，能低成本容纳层次结构，理论上更适合捕获 GCD 中已知→未知类别迁移所需的细粒度局部语义。

## 核心贡献（创新点）
1. **范式转换**：指出 GCD 主流欧氏/球面表征的几何瓶颈，提出以双曲流形替代的传统空间视角。与已有方法本质不同在于：不依赖人工预置类别树，而是利用双曲指数膨胀性质自动建模样本内部的层次化部件关系。
2. **统一框架 HypCD**：提出可同时适配非参数与参数化 GCD 基线的统一双曲框架。与 prior 工作的区别在于：完整打通从欧氏骨干输出→特征裁剪→指数映射→双曲对比→双曲分类的端到端流水线，而非仅修改后处理聚类。
3. **双曲混合对比学习**：设计同时利用测地距离与角度相似度的混合对比损失，并引入 $\alpha_d$ 线性调度策略。区别于仅用余弦相似度或纯欧氏距离的先驱工作，该混合度量在双曲流形上实现了更稳定的优化轨迹。
4. **双曲参数化分类头（HypFFN）**：用双曲线性层替换传统 MLP，基于 Möbius 运算实现类别原型预测。与现有球面分类器相比，其几何对齐性更强且在自蒸馏框架下直接兼容均值熵正则。

## 方法详解
- **欧氏→双曲映射**：骨干网络输出 $\mathbf{z}\in\mathbb{E}^n$ 先经特征裁剪 $\mathcal{C}(\mathbf{z})=\min\{1, r/\|\mathbf{z}\|\}\cdot\mathbf{z}$ 防止靠近球边界导致梯度消失，再通过指数映射 $\exp_{\mathbf{o}}^c(\cdot)$ 投射至曲率 $-c^2$ 的 Poincaré 球 $\mathbb{H}^n$。
- **双曲混合对比损失**：统一自监督与监督对比形式，相似度函数 $S$ 同时支持 $S_d=-\mathcal{D}_{\mathbb{H}}$（测地距离）与 $S_a$（余弦相似度，因保角性在双曲空间等价）。总表征损失为 $\mathcal{L}_{hrep}=\alpha_d\mathcal{L}_{dis}+(1-\alpha_d)\mathcal{L}_{ang}$，其中 $\alpha_d$ 在训练中从 0 线性增至 $\alpha_d^{\max}$，初期优先优化角度分布，后期强化距离判别。
- **双曲分类器 HypFFN**：针对参数化基线，将原 MLP 最后一层替换为双曲线性层：$\text{HypLinear}(\mathbf{z}^{\mathbb{H}},\mathbf{w},\mathbf{s})=\text{Proj}[(\mathbf{w}\otimes_c\mathbf{z}^{\mathbb{H}})\oplus_c\mathbf{s}]$，结合 Möbius 矩阵乘法与安全投影保证数值稳定，分类损失沿用原交叉熵形式但作用于双曲输出。
- **标签分配兼容设计**：非参数方法（GCD、SelEx）保留 Euclidean 空间上的半监督 k-means，因双曲训练已将层次结构知识迁移至 $\mathbb{E}^n$；参数化方法（SimGCD）直接使用 HypFFN 在双曲空间预测。
- **整体目标**：$\mathcal{L}_{rep}^{\mathbb{H}}=(1-\lambda_b^{\mathbb{H}})\mathcal{L}_{hrep}^u+\lambda_b^{\mathbb{H}}\mathcal{L}_{hrep}^s$，联合双曲分类损失 $\mathcal{L}_{cls}^{\mathbb{H}}$ 共同优化。

## 实验与结果
- **数据集与划分**：SSB 细粒度基准（CUB、Stanford-Cars、FGVC-Aircraft）及通用基准（CIFAR-10、CIFAR-100、ImageNet-100）；沿用 Vaze et al. [51] 标准 Old/New 划分。
- **评估基线**：GCD [51]、SimGCD [59]、SelEx [45] 及 ORCA、XCon、OpenCon、PromptCAL、DCCL、GPC、CiPR、SPTNet、CMS、AMEND 等；骨干使用 DINO 与 DINOv2。
- **关键结果**：Hyp-SelEx 在 SSB 上均取得最高 ACC；Stanford-Cars 上 Hyp-GCD 较基线提升 11.8%（All）、13.3%（Old）、15.9%（New）；CUB 上 DINOv2 backbone 达到 90.7% All ACC；CIFAR-100 上 Hyp-SimGCD 达 82.4%（DINO）/91.5%（DINOv2）All ACC；ImageNet-100 上较基线最高提升 3.7%。
- **最强结果与结论**：Hyp-SelEx + DINOv2 在 CUB 上达到 90.7% All ACC，建立新 SOTA；双曲方法在全部数据集上 Old-New 差距显著缩小，证明层次化表征有效促进了已知→未知类别的知识迁移。

## 相关工作脉络
- **GCD 主线工作**：从 NCD [24] 到 GCD [51]、参数化基线 SimGCD [59]、最新 SOTA SelEx [45] 与 SPTNet [55]；本文定位为用几何视角统一升级这些方法的表征/分类模块，而非修改数据增强或提示机制。
- **层次化 GCD 尝试**：如 CiPR [27]、SelEx [45] 等关注类别间预定义层次；本文区别于它们，强调利用双曲空间自然建模类别内部件层次，无需
