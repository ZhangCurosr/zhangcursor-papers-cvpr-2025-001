---
title: "Self-Expansion-of-Pre-trained-Models-with-Mixture-of-Adapter"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_Self-Expansion_of_Pre-trained_Models_with_Mixture_of_Adapters_for_Continual_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 09:40:44"
field: "持续学习/类增量学习"
keywords: ["持续学习", "Pre-trained Models", "Modularized Adaptation", "Self-Expansion", "Mixture of Adapters", "Catastrophic Forgetting", "Parameter-Efficient Fine-Tuning"]
innovations: ["提出基于表征描述符的模块化适配器，实现自适应分布偏移检测与按需自扩展", "设计可展开加权路由器，以软混合方式组合多适配器输出，支持增量扩展而不遗忘", "实现亚线性参数增长且无回放，在多个视觉 CL 基准上达到 SOTA"]
benchmarks: ["CIFAR-100", "ImageNet-R", "ImageNet-A", "VTAB"]
---

# 论文速读：Self-Expansion-of-Pre-trained-Models-with-Mixture-of-Adapter

## 一句话总结
本文提出 **SEMA（Self-Expansion of pre-trained models with Modularized Adaptation）**，一种基于预训练模型（PTM）的无回放持续学习方法，通过引入**模块化适配器**与**表征描述符（RD）**，自动检测表征层面的分布偏移并按需触发模型自扩展，以亚线性速率增长参数规模，在保证知识重用的同时实现更优的稳定-可塑平衡。

---

## 研究问题与动机

1. **PTM-Based CL 方法存在适应潜力受限问题**：现有基于预训练模型（如 ViT）的持续学习方法通常冻结 PTM 并在其上添加固定大小的 prompt/adapters 池进行跨任务共享适配，为避免灾难性遗忘而限制参数更新（如仅在第一任务更新），导致持续适应能力受限。
2. **任务特异性扩展带来线性增长和知识复用受损**：部分工作（如 Expansive Adapter、InfLoRA）为每个任务新增独立的任务特定模块，虽减少跨任务干扰，但导致模型参数量随任务数线性增长，且知识共享与复用能力受限。
3. **缺乏对稳定-可塑平衡的细粒度控制**：现有方法在"固定参数集适配"与"每任务全扩展"之间只有两种极端策略，无法根据新任务的实际分布偏移程度动态决定扩展的**时机**与**位置**（即哪一层需要新增适配器）。
4. **无回放（rehearsal-free）场景下的挑战**：存储全部历史数据并重训在经济与隐私上不可行，需要在无经验回放的前提下有效抑制灾难性遗忘。

---

## 核心贡献（创新点）

1. **提出 SEMA：一种基于自扩展的模块化适配持续学习新方法**——与固定池适配方法（如 L2P、DualPrompt）和每任务线性扩展方法（如 InfLoRA、Expandable Subspace Ensemble）的本质区别在于：SEMA 根据分布偏移检测信号**按需自动决定**何时、在何层扩展，而非预定义固定大小或任务数线性增长。
2. **设计模块化适配器（Modular Adapter）：功能适配器 + 表征描述符（RD）的成对结构**——与 ADAM/SimpleCIL 等仅使用单一功能适配器的本质区别在于：RD 以**密度估计/异常检测**的方式建模输入特征的局部分布，作为扩展触发的分布偏移指示器，使扩展决策可在表征层面进行。
3. **提出可展开加权路由器（Expandable Weighting Router）**——将多个适配器的输出以加权混合（Mixture of Experts 形式）组合，并随新适配器的加入在线扩展，而非采用硬选择（Top-1 Select）或随机分配，显著提升知识复用效率。
4. **实现亚线性参数扩展并达到无回放 SOTA 性能**——在 CIFAR-100、ImageNet-R、ImageNet-A、VTAB 等多个基准上超过既有 PTM-Based CL 方法，且在含对抗样本的数据集（如 ImageNet-A）上提升更显著。

---

## 方法详解

### 整体架构
SEMA 以冻结的预训练 ViT（ViT-B/16-IN1K）为骨干网络，在其 Transformer 层的 MLP 模块旁路插入可学习的适配器。模型支持在任意层（默认最后三层）进行自扩展。

### 3.3 模块化适配器（Representation-Aware Modular Adapter）

每个模块化适配器由**成对**的两个组件构成：

**功能适配器（Functional Adapter）** $f_{\phi_k^l}(\cdot)$：
- 插入第 $l$ 层 Transformer MLP 的旁路，负责桥接预训练表示与下游任务的差异。
- 默认实现为轻量级 Adapter（Houlsby et al., 2019）：含下投影层 $\mathbf{W}_{\text{down},k}^l \in \mathbb{R}^{d \times r}$、ReLU 激活、上投影层 $\mathbf{W}_{\text{up},k}^l \in \mathbb{R}^{r \times d}$，其中 $r=16$ 为隐藏维度。
- 输出公式：
$$f_{\phi_k^l}(\mathbf{x}^l) = \mathrm{ReLU}(\mathbf{x}^l \cdot \mathbf{W}_{\text{down},k}^l) \cdot \mathbf{W}_{\text{up},k}^l$$
- 第 $l$ 层 MLP 输出调整为：$\mathbf{x}_{\text{out}}^l = \mathrm{MLP}(\mathbf{x}^l) + \sum_{k=1}^{K^l} w_k^l \cdot f_{\phi_k^l}(\mathbf{x}^l)$

**表征描述符（Representation Descriptor, RD）** $g_{\varphi_k^l}(\cdot)$：
- 以**自编码器（AE）** 实现（亦可用 VAE），用于捕捉与配对适配器相关的输入特征分布。
- 训练目标为仅基于重建损失（与分类损失梯度隔离）：
$$\mathcal{L}_{\text{RD},k}^l(x) = \sum_{\mathbf{x} \in \mathcal{X}_k^l} ||\mathbf{x} - g_{\varphi_k^l}(\mathbf{x})||_2^2$$
- $\mathcal{X}_k^l$ 为该适配器所对应的所有样本的第 $l$ 层输入特征。
- RD 独立于分类损失梯度优化，仅维护局部特征分布状态，用于后续**分布偏移检测**。

### 3.4 可展开加权路由器（Expandable Weighting Router）

- 路由函数 $h_{\psi^l}(\cdot): \mathbb{R}^d \to \mathbb{R}^{K^l}$ 实现为线性映射 + softmax：
$$\mathbf{w}^l = h_{\psi^l}(\mathbf{x}^l) = \mathrm{softmax}(\mathbf{x}^l \cdot \mathbf{W}_{\text{mix}}^l), \quad \mathbf{W}_{\text{mix}}^l \in \mathbb{R}^{d \times K^l}$$
- 最终第 $l$ 层输出：$\mathbf{x}_{\text{out}}^l = \mathrm{MLP}(\mathbf{x}^l) + \sum_{k=1}^{K^l} w_k^l \cdot f_{\phi_k^l}(\mathbf{x}^l)$
- **关键设计**：当新适配器在第 $l$ 层加入时，路由器 $\mathbf{W}_{\text{mix}}^l$ 相应扩展（增加一列），但**已有列参数冻结**，仅训练新增列——与分类头增量训练策略一致，有效抑制路由器遗忘。

### 3.5 持续学习目标函数

$$\min_{\{\phi_k^l\},\{\psi^l\},\{\varphi_k^l\}} \sum_{t=1}^{T} \mathbb{E}_{(x,y) \in \mathcal{D}^t}\left[\mathcal{L}_{\text{CE}}(F(x),y) + \sum_{l=1}^{L}\sum_{k=1}^{K^l} \mathcal{L}_{\text{RD},k}^l(x;\varphi_k^l)\right]$$

- 功能适配器和路由器通过交叉熵损失 $\mathcal{L}_{\text{CE}}$ 优化；
- RD 通过重建损失 $\mathcal{L}_{\text{RD}}$ 独立优化（梯度不回流至分类路径）；
- 当某任务未触发扩展时，现有冻结适配器直接复用，无需额外训练。

### 3.6 自扩展策略（Self-Expansion Strategy）

**任务导向扩展（Task-oriented Expansion）**：
- 每个任务在每个层**最多新增一个适配器**，避免过度膨胀；
- 在任务到达后**首个 epoch** 扫描所有样本决定是否需要扩展。

**Z-score 扩展信号检测**：
- 维护每个 RD 的 reconstruction error 的滑动均值 $\mu_k^l$ 和标准差 $\sigma_k^l$；
- 对每个新样本计算 z-score：$z_k^l = (r_k^l - \mu_k^l) / \sigma_k^l$，其中 $r_k^l = ||\mathbf{x}^l - g_{\varphi_k^l}(\mathbf{x}^l)||_2^2$；
- 若某层所有现有 RD 的 z-score 均超过阈值 $\tau$，则触发扩展信号；
- 该方法经 z-score 归一化后对阈值 $\tau$ 设置**不敏感**（消融实验证实）。

**多层扩展（Multi-layer Expansion）**：
- 从浅层到深层逐层执行扩展决策；
- 浅层新增适配器后先完成训练，再据此判断是否需要继续向深层扩展；
- 允许不同层面对不同程度的分布偏移。

---

## 实验与结果

### 数据集与设置
- **CIFAR-100**、**ImageNet-R（IN-R，5/10/20任务）**、**ImageNet-A**、**VTAB**
- 骨干：ViT-B/16-IN1K（冻结）
- 优化器：SGD，adapter lr=0.005，RD lr=0.01，cosine decay
- 隐藏维度 $r=16$，batch size=32
- 默认在**最后三层** Transformer 层进行自扩展

### 主要结果（Tab. 1，$\mathcal{A}_N$ 为最终平均准确率）

| 方法 | CIFAR-100 A | IN-R (5-task) A | IN-R (10-task) A | IN-R (20-task) A | ImageNet-A A | VTAB A |
|------|------------|-----------------|------------------|------------------|-------------|--------|
| FT Adapter | 47.88 | 53.91 | 45.31 | 38.51 | 29.78 | 59.98 |
| L2P | 84.77 | 77.40 | 66.97 | 70.67 | 47.16 | 81.19 |
| DualPrompt | 86.60 | 76.39 | 72.83 | 62.33 | 59.54 | 82.89 |
| CODA-P | 91.55 | 81.63 | 81.11 | 75.00 | 47.29 | 79.88 |
| SimpleCIL | 82.31 | 65.83 | 67.09 | 67.59 | 60.05 | 85.29 |
| ADAM | 90.55 | 79.91 | 79.11 | 75.84 | 60.15 | 85.29 |
| InfLoRA | 90.51 | 78.58 | 81.39 | 78.87 | 59.71 | 88.90 |
| **SEMA（Ours）** | **91.37** | **84.75** | **83.56** | **81.75** | **64.53** | **91.26** |

**最强结果**：SEMA 在所有基准上均取得最优或次优的 $\mathcal{A}_N$，在 IN-R 和 VTAB 上超越第二名约 2~3%，在 ImageNet-A（对抗样本数据集）上领先约 4~5 个百分点。

### 消融实验要点（Tab. 2）
- **移除自扩展（No Exp.）**：ImageNet-A $\mathcal{A}_N$ 从 53.32 降至 49.90，VTAB 从 89.64 降至 83.66，验证自扩展必要性。
- **不同路由器策略**：加权软混合（SEMA）显著优于平均权重（Avg.W.）、随机权重（Rand.W.）、Top-1 硬选择（Top-1 Sel.）及随机选择（Rand. Sel.）。
- **阈值敏感性**：在宽范围阈值下性能稳定，证实 z-score 设计的鲁棒性。
- **多层 vs 单层扩展**：在多层（最后 3 层）扩展显著优于单层的设置。
- **不同适配器变体**：SEMA 兼容 Adapter [9]、LoRA [30]、Convpass [34]，均取得类似表现（Tab. 3）。
- **亚线性增长**：图 8 显示模型参数量随任务数呈**亚线性**增长。

---

## 相关工作脉络

1. **经验回放方法（A-GEM、DER、GEM）**：通过存储旧任务样本或生成伪样本防止遗忘——SEMA 与其本质区别在于**完全不依赖任何历史数据或记忆缓冲**，实现纯无回放学习。
2. **正则化方法（EWC、SI、MAS）**：通过惩罚重要参数变化来抑制遗忘——SEMA 不依赖参数重要性评估，而是通过**架构动态扩展**从根本上隔离不同任务的知识。
3. **固定池适配方法（L2P、DualPrompt、SimpleCIL、ADAM）**：使用预定义的固定大小 prompt/adapters 池——SEMA 与其区别在于**池大小动态扩展**且按需触发，而非静态固定。
4. **任务特异性扩展方法（InfLoRA、ES-CE）**：每个任务新增独立模块——SEMA 与其本质区别在于**跨任务复用已有适配器**且扩展速率呈亚线性，而非线性增长。
5. **模块化网络（Neural Module Networks、CLMN）**：通过模块隔离实现知识复用——SEMA 与其区别在于**扩展决策由表征描述符自动驱动**，无需人工预设模块分配规则。
6. **参数高效微调（PEFT，LoRA、Adapter、Prompt Tuning）**：SEMA 在此基础上引入了**分布感知+按需扩展+加权混合**的持续学习框架，填补了 PEFT 在持续学习场景中动态扩展能力的空白。

---

## 局限性与未来方向

1. **任务导向扩展的限制**：目前每个任务每层最多扩展一个适配器，设计上更适合 CIL（类增量学习）场景；对于**任务内存在多样性**的数据或真正的**在线持续学习**场景，扩展策略可能不够灵活。
2. **RD 的分布偏移检测能力受限**：当前使用 AE 作为 RD 实现 novelty detection，其检测精度直接影响扩展决策质量；可进一步提升 RD 的优化方法和扩展协议。
3. **阈值鲁棒性虽好但仍需设置**：虽然 z-score 归一化使方法对阈值不敏感，但用户仍需选择一个合理的阈值 $\tau$，完全免超参的自扩展仍有改进空间。
4. **仅在三类层进行扩展**：默认只在最后三层 Transformer 层扩展，早期层的信息整合能力未被充分利用。
5. **未来方向**：① 支持 fully online dynamic expansion，适用于 intra-task diversity；② 将 RD 优化和扩展协议提升至 meta-level，构建闭环学习系统；③ 拓展至在线持续学习（online CL）场景。

---

## 研究启发与可借鉴点

1. **"表征描述符"作为分布偏移检测器的设计思路可迁移**——将密度估计/异常检测模块与功能模块成对绑定，作为扩展触发器，这一思想可迁移至其他需要动态容量管理的学习场景（如多任务学习、元学习）。
2. **路由器扩展时冻结已有参数仅训练新增列的策略**——是一种简洁有效的防止路由器自身遗忘的方法，可借鉴于其他 MoE 类架构的增量训练。
3. **任务内首个 epoch 进行扩展决策的离线扫描机制**——以极低的额外开销（仅需第一 epoch 数据）换取精确的扩展时机判断，避免了在线高频扩展的计算负担，对实际部署有参考价值。
4. **与 LoRA/Convpass 等不同适配器变体的兼容性验证**——说明 SEMA 的核心创新在于**扩展决策机制**而非特定适配器形式，这种解耦设计为后续替换更先进的 PEFT 模块预留了接口。
5. **RD 梯度与分类梯度隔离的设计**——确保分布描述不被分类信号污染，可借鉴于其他需要"辅助监督信号独立训练"的多任务框架。

---

## 关键术语表

**SEMA（Self-Expansion of pre-trained models with Modularized Adaptation）**：本文提出的持续学习方法，通过模块化适配器与表征描述符自动检测分布偏移并按需扩展预训练模型。

**Modular Adapter（模块化适配器）**：由功能适配器（$f_\phi$）和表征描述符（$g_\varphi$）成对组成的适配模块，功能适配器负责任务适配，表征描述符负责感知输入特征分布。

**Representation Descriptor（RD，表征描述符）**：以 AE/VAE 实现的分布感知模块，训练目标为重建损失，用于检测输入特征是否超出已知分布范围，触发自扩展。

**Expandable Weighting Router（可展开加权路由器）**：实现为线性映射 + softmax 的路由模块，将多个适配器的输出以软加权混合方式组合，随新适配器加入而扩展（新增列），已有列参数冻结。

**Catastrophic Forgetting（灾难性遗忘）**：持续学习中模型在学习新任务时严重丢失旧任务知识的问题，是 CL 的核心挑战。

**Class-Incremental Learning（CIL，类增量学习）**：持续学习的一种设定，各任务间类别不重叠（$Y_t \cap Y_{t'} = \emptyset$），模型需逐步学习全新类别。

**Rehearsal-free（无回放）**：指在持续学习过程中不使用历史样本记忆缓冲（memory buffer）或生成模型进行回放的训练设定。

**Z-score Based Expansion Signal（基于 Z-score 的扩展信号）**：以各 RD 重建误差的滑动均值和标准差进行归一化后计算的 z-score 作为扩展触发判断依据，使方法对阈值设置不敏感。

---

## 可复现要素

- **数据集**：CIFAR-100、ImageNet-R（IN-R）、ImageNet-A、VTAB — 均为公开数据集。
- **代码**：已开源，地址为 https://github.com/huiyiwang01/SEMA-CL
- **权重**：ViT-B/16-IN1K 预训练权重（公开可用）
- **关键超参**：
  - 骨干网络：ViT-B/16-IN1K（冻结）
  - 优化器：SGD，adapter lr=0.005，RD lr=0.01，cosine decay
  - Batch size：32
  - Adapter 隐藏维度 $r$：16
  - 自扩展层数：默认最后三层 Transformer 层
  - 扩展阈值 $\tau$：论文未给出具体数值（消融实验图 6 显示在宽范围内鲁棒）

---
