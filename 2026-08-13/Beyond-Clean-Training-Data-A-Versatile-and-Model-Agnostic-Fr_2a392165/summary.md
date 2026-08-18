---
title: "Beyond-Clean-Training-Data-A-Versatile-and-Model-Agnostic-Fr"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Li_Beyond_Clean_Training_Data_A_Versatile_and_Model-Agnostic_Framework_for_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:55:40"
field: "分布外检测与数据污染鲁棒学习"
keywords: ["out-of-distribution detection", "contaminated training data", "model-agnostic framework", "iterative training", "sample weighting", "negative samples"]
innovations: ["提出可即插即用的模型无关迭代框架，适配使用与不使用负样本的 OOD 方法", "设计带过渡机制的样本权重函数，缓解早期不可靠 OOD 评分的负面影响", "首次实现对污染训练数据中未知 OOD 比例的无监督准确估计"]
benchmarks: ["CIFAR-10", "CIFAR-100", "ImageNet-1k", "iNaturalist", "OpenImage-O", "CelebA", "Tiny-ImageNet", "Mini-ImageNet"]
---

# 论文速读：Beyond-Clean-Training-Data-A-Versatile-and-Model-Agnostic-Framework-for-OOD-Detection-with-Contaminated-Training-Data

## 一句话总结
本文提出一个通用、模型无关的迭代框架，使现有大多数 OOD 检测模型能在含 OOD 样本的污染训练集上训练，并首次实现从训练数据中准确估计未知 OOD 样本比例。

## 研究问题与动机
- 现实 AI 应用中训练数据普遍被污染（混入无标签 OOD 样本），但几乎所有现有 OOD 检测方法都假设训练集完全干净。
- 污染会严重破坏 OOD 检测性能（如仅 5% 污染即可导致性能大幅下降）。
- 针对不同 OOD 方法分别定制以处理污染非常不现实，需要一种可即插即用的模型无关框架。
- 除提升检测性能外，准确估计训练集中未知 OOD 占比本身也是一个尚未被探索的重要且困难的问题。

## 核心贡献（创新点）
1. 提出一个与几乎所有现有 OOD 检测方法兼容的模型无关迭代框架，显著提升在污染训练集上的 OOD 检测性能。
2. 设计新型样本重要性权重函数与过渡机制，解决早期不可靠 OOD 评分导致的误分类放大问题。
3. 首次实现从污染训练数据中直接、准确地估计未知 OOD 样本比例，并通过迭代优化持续修正该比例。
4. 扩展框架以兼容使用负样本的 OOD 方法，将权重范围调整为[-1,1]，避免对 ID/OOD 样本进行硬划分。
5. 在多数据集与多种骨干模型上的实验验证了方法的通用性与有效性，包括 ImageNet-1k 等大规模场景。

## 方法详解
- 整体为迭代框架，训练轮次为 $t$，第 $t$ 轮使用当前模型 $f_{\theta^{(t)}}$ 对所有样本计算归一化 OOD 分数 $s_i^{(t+1)} \in [0,1]$，并在每轮用样本级权重 $w_i^{(t)}$ 加权原始损失训练：$\mathcal{L}^{(t)} = \frac{1}{n}\sum_i w_i^{(t)}L(x_i)$。
- 不使用负样本的骨干方法采用权重：
  - 初始阶段 $t=0$ 令 $w_i^{(0)}=1$；
  - 中间阶段引入过渡函数 $\tau^{(t)} = \frac{k}{t}+1$，并先用移动平均 OOD 分 $\bar{s}_i^{(t)}$ 通过递归阈值估计 $\hat{P}_{\text{OOD}}^{(t)}$，再计算 $w_i^{(t)} = \frac{1}{1+e^{\alpha(\frac{s_i^{(t)}}{\tau^{(t)}}-\hat{P}_{\text{ID}}^{(t)})}}$；
  - 结束阶段 $\tau^{(t)}\to 1$，权重退化为标准形式。
- 使用负样本的骨干方法将权重映射到 $[-1,1]$：
  - $t=0$ 仍为 1；
  - 中间阶段保持同一过渡机制，使用 $w_i^{(t)} = \frac{2}{1+e^{\alpha(\frac{s_i^{(t)}}{\tau^{(t)}}-\hat{P}_{\text{ID}}^{(t)})}}-1$；
  - 结束阶段去掉过渡：$w_i^{(t)} = \frac{2}{1+e^{\alpha(s_i^{(t)}-\hat{P}_{\text{ID}}^{(t)})}}-1$。
- 终止条件为 OOD 占比估计变化较小：$|\hat{P}_{\text{OOD}}^{(t)} - \hat{P}_{\text{OOD}}^{(t-1)}| < 0.05 \cdot \hat{P}_{\text{OOD}}^{(t)}$。
- 过渡机制的核心动机是：早期 OOD 评分不可靠，可能把易学的 ID 样本误判为 OOD 并降权，从而削弱训练；过渡机制保证早期即使 $s_i^{(t)}$ 偏大，$\frac{s_i^{(t)}}{\tau^{(t)}}$ 仍较小，使 ID 样本仍能保持较高权重。
- OOD 占比估计通过逐轮更新阈值实现：$\hat{P}_{\text{OOD}}^{(t)} = \frac{1}{n}\sum_i \mathbf{1}_{\{\bar{s}_i^{(t)} > (1-\hat{P}_{\text{OOD}}^{(t-1)})\}}$，初始 $\hat{P}_{\text{OOD}}^{(0)}=0$。

## 实验与结果
- 数据集：CIFAR-10、CIFAR-100、CelebA、SVHN、GTSRB、Mini-ImageNet、Tiny-ImageNet、ImageNet-1k、iNaturalist、OpenImage-O。
- 评估指标：OOD 检测用 AUROC；OOD 比例估计用归一化误差 $\varepsilon = \frac{|P_{\text{OOD}}^* - \hat{P}_{\text{OOD}}^{\text{(END)}}|}{0.5}$。
- 骨干对比基线包括不使用负样本的 DB、LReg、LRat、WAIC、G-ODIN、Energy、ReAct，以及使用负样本的 CSI、CnC、VOS，并与模型无关异常检测框架 IAD 对比。
- OOD 检测性能（$P_{\text{OOD}}^*=1\%$）：
  - ImageNet-1k + iNaturalist：G-ODIN 从 60.4 提升至 64.8；ReAct 从 63.9 提升至 70.3；OpenImage-O：ReAct 从 57.6 提升至 63.0。
  - CIFAR-10 作为 ID、CelebA 作为 OOD（高难度小样本设置）：
    - DB 在 1% 污染下从 58.6 提升到 62.2；ReAct 从 65.7 提升到 72.2。
    - 2% 污染下 ReAct 从 61.9 提升到 69.8。
  - 使用负样本的方法：VOS 在 CIFAR-100 上从 57.2 提升到 62.1；ImageNet-1k + iNaturalist 从 61.0 提升到 63.2。
- OOD 比例估计精度（$P_{\text{OOD}}^*=1\%$）：
  - BB+Ours 在各种骨干上的归一化误差 $\varepsilon$ 接近 0（多数为 0.000–0.002）。
  - IAD 因假设污染比例为 0.5，无法给出有效估计，表中报告 N/A 并以固定 0.5 计算的误差在 0.900–0.980。
- 不同污染比例（0.01、0.02、0.05）下估计误差依然很小，DB/LReg/Energy/ReAct/CnC/VOS 等均在 0.016 以内。
- 消融显示：去掉 OOD 比例估计或去掉过渡函数都会显著降低性能，验证两项设计的关键作用。

## 相关工作脉络
- OOD 检测主流方法分为判别类、密度估计类、距离类、重建类，但普遍假设干净训练数据；本文针对实际污染场景。
- IAD 是最接近的模型无关方法，用于异常检测与污染数据，但不能利用负样本，也无法估计训练数据中的污染比例；本文在目标、损失结构与适应性方面均与其本质不同。
- 使用负样本的 OOD 方法（CSI、CnC、VOS 等）原先依赖合成负样本；本文指出污染训练集天然包含真实 OOD 样本，可通过权重机制被利用。
- 异常检测领域部分工作考虑污染数据，但直接将 AD 方法迁移到 OOD 检测通常无效，因为任务目标、假设和数据特性存在差异。
- Curriculum learning、简单性偏差和噪声标签相关研究为早期让模型优先学习“容易”的 ID 样本提供了理论依据，支撑了本文的过渡机制设计。

## 局限性与未来方向
- 迭代训练依赖 OOD 分数逐步稳定，在极重度污染或评估指标设计不合理的情况下可能收敛较慢或效果受限。
- 超参数（如 $\alpha$、$k$）与终止阈值的经验性设置可能影响在不同数据集上的稳定性。
- 方法面向无监督场景，未引入少量标签信息，可能存在进一步结合弱监督或自监督信号的潜力。
- 本文未详细讨论在分布偏移非常大或跨域语义完全不同的极端污染场景下的鲁棒性边界。
- 未来可将框架推广到时间序列、多模态或其他检测任务，并研究自动化超参数选择策略。

## 研究启发与可借鉴点
- 将“样本重要性权重 + 迭代更新 + 过渡机制”的思路复用到其他需要处理污染训练数据的任务中，尤其是缺乏标注的异常/分布外检测场景。
- 用移动平均与递归阈值交替估计污染比例，是一种可迁移的无监督占比校正技术，可用于课程学习与噪声剔除。
- 使用负样本的方法原本需要合成负样本，本文提示可利用真实 OOD 信号并通过连续权重统一处理，这一设计思路可扩展到其他对比学习与生成式检测框架。
- 消融中对估计模块与过渡模块的分离验证方式清晰，可作为类似迭代框架的标准评估范式。
- 该方法可与团队当前的检测或表征学习方向结合，例如在大规模预训练数据清洗与可靠性评估中提供无监督校准手段。

## 关键术语表
**Out-of-Distribution Detection (OOD Detection)**：判断输入是否来自与训练分布不同的数据分布的检测任务。
**Contaminated Training Data**：训练集中混入未标注 OOD 样本的情况。
**Model-Agnostic Framework**：不依赖特定模型结构、可直接嵌入多种骨干 OOD 方法的通用框架。
**Sample-wise Importance Weight**：对每个训练样本分配独立权重，用于调节其在损失中的贡献。
**Transition Mechanism**：在迭代早期降低不可靠 OOD 评分对权重的影响，避免误降 ID 样本权重。
**OOD Percentage Estimation**：从污染训练集中估计真实 OOD 样本比例。
**Negative Sample Utilization**：部分 OOD 方法通过正/负样本对比增强判别，本文将其扩展到污染数据场景。
**Normalized Estimation Error**：用于量化 OOD 比例估计精度的归一化误差指标。

## 可复现要素
- 数据集：CIFAR-10、CIFAR-100、CelebA、SVHN、GTSRB、Mini-ImageNet、Tiny-ImageNet、ImageNet-1k、iNaturalist、OpenImage-O；论文声明基于公开数据集构建。
- 代码/权重：论文未明确说明开源状态。
- 关键超参：权重斜率 $\alpha>0$、过渡函数陡峭度 $k>0$、终止阈值系数 0.05（经验设定），具体数值论文正文未全部给出，需查阅附录或实现细节。
