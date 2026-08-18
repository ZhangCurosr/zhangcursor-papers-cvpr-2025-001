---
title: "Do-Your-Best-and-Get-Enough-Rest-for-Continual-Learning"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Kang_Do_Your_Best_and_Get_Enough_Rest_for_Continual_Learning_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 07:00:35"
field: "持续学习"
keywords: ["continual learning", "forgetting curve", "spaced repetition", "self-supervised learning", "view-batch", "catastrophic forgetting", "replay buffer"]
innovations: ["提出View-Batch Model，通过多视角增强重构训练调度以优化recall interval", "设计one-to-many KL散度自监督损失，无需额外结构即可充分学习单一样本", "作为drop-in replacement在CIL/TIL/DIL多种协议下 consistently 提升SOTA基线性能"]
benchmarks: ["S-CIFAR-10", "S-CIFAR-100", "S-Tiny-ImageNet", "DomainNet", "S-ImageNet-R"]
---

# 论文速读：Do-Your-Best-and-Get-Enough-Rest-for-Continual-Learning

## 一句话总结
论文受艾宾浩斯遗忘曲线理论与间隔效应启发，提出 **View-Batch Model (VBM)**，通过构造多视角 view-batch 延长重训练同一样本的 recall interval，并结合 one-to-many KL 散度自监督损失实现充分学习，作为一种即插即用模块显著提升多种持续学习（CL）方法在 CIL/TIL/DIL 协议下的长期记忆保持能力。

## 研究问题与动机
- **灾难性遗忘**：持续学习在连续学习任务序列中学习新知识时，旧知识迅速衰退是核心难题。
- **现有方法 recall interval 过短**：当前主流 CL 方法（如 iCaRL、DER、ER 等）训练调度中重训练同一批样本的间隔（recall interval）等同于一个 epoch 内的 batch 总数，时间空间不足，难以发挥间隔效应（spacing effect）。
- **遗忘曲线未被充分利用**：Ebbinghaus 遗忘曲线与 spaced repetition 在神经网络的 long-term memory retention 中虽有初步分析性应用，但从未被系统性地作为训练调度优化器引入 CL。
- **自监督学习优势未被结合**：自监督学习已证明能提取 task-agnostic、更鲁棒的表征，但未与 recall interval 优化协同使用。

## 核心贡献（创新点）
1. **提出 View-Batch Model**：以 recall interval 为核心设计指标，将样本级 replay 扩展至 view-batch 级，通过扩大重训练间隔来增强 long-term memory retention；与以往仅把间隔理论作分析工具的方法本质不同，VBM 是主动训练调度器。
2. **基于多视角增强延长 recall interval 的 replay 方法**：将 view-batch 内同一图像的 V 个增强视图依次训练，使 recall interval 从 $B \times T$ 扩展为 $B \times T \times V$，同时以 $1/V$ epoch 数保持总样本暴露量不变；与重复增强（repeated augmentation）研究仅聚焦增强效果不同，本文首次显式控制 recall interval。
3. **One-to-many divergence self-supervised loss**：提出弱增强-强增强之间的 KL 散度损失，让网络在一个 view-batch 内充分泛化地学习单一样本；相比 MoCo、SimCLR 等需要额外 queue/teacher 的 SSL 方法，本方法无需改架构或额外训练阶段，以 drop-in replacement 形式接入任意 CL 方法。
4. **全面验证 + 即插即用泛化性**：在 rehearsal 与 non-rehearsal、不同 buffer 尺寸、不同 backbone（ResNet-18 / ViT-B/16）、三种协议（CIL/TIL/DIL）下均一致提升 SOTA 基线，且计算开销几乎不变。

## 方法详解
**整体框架（Algorithm 1）：**
- 输入数据集 $\mathcal{D}^{\mathrm{train}}$，网络参数 $\theta$，学习率 $\gamma$，batch size $B$，view-batch size $V$。
- 将 batch size 调整为 $B \leftarrow B / V$（保证训练样本总数与 baseline 一致）。
- 每步从数据集中采样 $B$ 个独立样本，每个样本生成 $V$ 个增强视图构成 $\mathcal{V}_i = \{I_{i,j}\}_{j=1}^{V}$，再对 $i=1,\dots,B$ 组成 view-batch $\mathcal{B}^{\mathcal{V}}$。

**增强策略：**
- 弱增强 $A_w$：仅水平翻转（horizontal flip）。
- 强增强 $A_s$：AutoAugment（来自 [13]）。

**学习调度重构：**
- 传统调度：$\mathcal{A}_{\mathrm{conventional}} = [\mathcal{B}^I_1, \cdots, \mathcal{B}^I_T, \mathcal{B}^I_1, \cdots, \mathcal{B}^I_T]$，recall interval $= B \times T$。
- VBM 调度：$\mathcal{A}_{\mathrm{ours}} = [\mathcal{B}^{\mathcal{V}}_1, \cdots, \mathcal{B}^{\mathcal{V}}_T, \mathcal{B}^{\mathcal{V}}_1, \cdots, \mathcal{B}^{\mathcal{V}}_T]$，recall interval $= B \times T \times V$（扩大 $V$ 倍）。

**自监督损失（Eq. 3）：**
$$L_{\mathrm{ssl}}(f_\theta, \mathcal{B}^{\mathcal{V}}) = \frac{1}{B \cdot (V-1)} \sum_{i=1}^{B} \sum_{j=2}^{V} D_{\mathrm{KL}}(p_i^1 \| p_i^j)$$
其中 $p_i^1 = \sigma(f_\theta(A_w(\mathcal{V}_{i,1})))$，$p_i^j = \sigma(f_\theta(A_s(\mathcal{V}_{i,j})))$，即 1 个弱增强视图预测与其余 $V-1$ 个强增强视图预测之间的平均 KL 散度。

**总目标（Eq. 4）：**
$$\min_{f_\theta} \; L_{\mathrm{sup}}(f_\theta, \mathcal{B}^{\mathcal{V}}) + L_{\mathrm{ssl}}(f_\theta, \mathcal{B}^{\mathcal{V}})$$
其中 $L_{\mathrm{sup}} = \frac{1}{B \cdot V}\sum_{i=1}^{B}\sum_{j=1}^{V} \mathcal{H}(y_i, p_i^j)$ 为各视图上的交叉熵损失。

**实证支撑（Figure 4）：**
- Recall interval 越大，遗忘程度越高（Figure 4a），验证"充分遗忘"前提成立。
- x3（interval 为 baseline 的 3 倍）下记忆衰减最慢（Figure 4b），并在 formal learning 结束阶段取得更高精度（Figure 4c）。

## 实验与结果
**数据集与基线：** S-CIFAR-10、S-CIFAR-100、S-Tiny-ImageNet、S-ImageNet-R、DomainNet；基线覆盖 iCaRL、ER、DER、DER++、TCIL、LwF、LwF-MC、BiC、WA、UCIR、DyTox、SLCA、L2P、DualPrompt、CODA-Prompt 等。

**主要结果（保留关键数值）：**

- **S-CIFAR-10 / S-Tiny-ImageNet（Table 1）**：
  - Buffer=0（LwF）：VBM-LwF 在 S-CIFAR-10 上 TIL 77.53%（+15.55），S-Tiny-ImageNet 上 51.21%（+35.95）。
  - Buffer=200，iCaRL：VBM-iCaRL 在 S-CIFAR-10 上 Avg 81.25%（+4.09）；S-Tiny-ImageNet 上 38.85%（+2.77）。
  - Buffer=200，DER++：VBM-DER++ 在 S-CIFAR-10 上 Avg 80.65%（+4.51）；S-Tiny-ImageNet 上 29.02%（+2.18）。
  - Buffer=500，iCaRL：VBM-iCaRL 在 S-CIFAR-10 上 Avg 81.04%（+5.50）；S-Tiny-ImageNet 上 44.75%（+3.34）。
  - Buffer=5120，DER：VBM-DER 在 S-CIFAR-10 上 Avg 88.16%（+2.61）；S-Tiny-ImageNet 上 50.99%（+3.16）。

- **DIL（DomainNet，Table 2）**：VBM-DER++ Avg 42.16%（+7.24），为最强提升之一。

- **CIL on S-CIFAR-100（Table 3，5-step）**：
  - VBM-DER：Avg 78.60%（+1.83），Last 70.60%（+2.54）。
  - VBM-TCIL：Avg 79.23%（+1.90），Last 71.23%（+1.75）。

- **Non-rehearsal CIL（Table 4，S-CIFAR-100）**：
  - VBM-TCIL 5-step：Avg 66.40%（+2.00），Last 54.69%（+2.32）。
  - VBM-TCIL 10-step：Avg 61.12%（+4.28），Last 45.92%（+5.61）。

- **Pre-trained model + CIL（Table 5）**：
  - VBM-SLCA 在 S-CIFAR-100 上 Avg 94.52%，Last 91.78%（小幅提升但 consistent）。

- **消融（Table 6）**：
  - iCaRL：Replay(+2.53) + Strong augment(+0.76) + SSL(+4.09) 逐项叠加，共同贡献最大 (+4.09)。
  - DER++：四项全开达 Avg 80.65%（+4.51）。

- **最优 Recall Interval（Figure 6）**：在多数 setting 下 x3 或 x4 取得最佳 Last accuracy 并最小化 Forgetting。

## 相关工作脉络
1. **遗忘曲线与神经网络**：Amiri et al. [1] 提出 spaced repetition scheduler 降低训练成本；Tirumala et al. [39] 分析大模型规模与遗忘曲线的关系；本文首次将其作为 CL 训练调度的核心设计依据，而非事后分析工具。
2. **间隔效应 (Spacing Effect)**：Cepeda et al. [6,7] 奠定人类记忆中的理论框架；Zhong et al. [48] 在 LLM 对话中应用；本文将其引入视觉持续学习的 replay 调度。
3. **经验重放 CL**：ER [34]、iCaRL [32]、DER [3]、DER++ 以 memory buffer 缓解灾难性遗忘；VBM 在不改变 buffer 大小前提下，通过调整同一 buffer 样本的重训间隔来提升性能（drop-in replacement）。
4. **自监督学习**：SimCLR [11]、MoCo [5]、BYOL [20] 强调 architecture/queue/teacher 设计；本文 SSL loss 仅修改 objective function，不增加结构开销。
5. **重复增强**：Augment Your Batch [22] 研究 batch 内重复 augmentation 的性能增益但未涉及 recall interval；本文首次把多视角增强与间隔效应关联。
6. **Prompt-based CL**：L2P [42]、DualPrompt [41]、CODA-Prompt [38]、SLCA [46] 在预训练 ViT 上做增量学习；本文证明 VBM 同样可提升 prompt-based 方法性能。

## 局限性与未来方向
- **最优 recall interval 需手动调参**：实验显示 x3–x4 效果最佳，但未提供自适应间隔选择机制，对不同数据集/模型规模可能需要重新搜索。
- **视觉分类任务为主**：全部实验基于图像分类（CIFAR / ImageNet / DomainNet），未验证在检测、分割、NLP 等任务上的泛化性。
- **非 rehearsal 场景提升幅度相对较小**：Table 5 中 VBM-SLCA 提升仅约 +0.2~+0.4%，说明在预训练大模型 + prompt 设定下 VBM 增益有限。
- **view-batch 的视图多样性依赖 AutoAugment**：强增强策略固定为 AutoAugment，未探讨其他增强组合（如 CUTMIX、RandAugment）对间隔效应的交互影响。
- **理论分析较浅**：遗忘曲线公式仅在 supplementary material 给出，缺乏对 optimal interval 的理论下界/上界分析。

## 研究启发与可借鉴点
1. **间隔效应作为 CL 训练调度器**：将心理学遗忘曲线理论形式化为训练 schedule 的设计范式，可直接迁移至序列推荐、在线学习、NER 等领域的时间敏感型任务。
2. **One-to-many divergence loss 的设计思想**：以弱增强为 anchor、强增强为 target 的 KL 散度模式简洁高效，无需额外网络结构，可作为通用正则化项嵌入任何监督/半监督训练 pipeline。
3. **Recall interval 可视化分析工具**：Figure 6 中以 Last Accuracy vs. Forgetting 的二维网格评估不同间隔，可为团队后续研究提供一套标准化的 schedule ablation 评估模板。
4. **Drop-in replacement 的设计哲学**：VBM 不与底层 CL 算法耦合，只修改 replay unit 与 loss，这种"最小改动最大化收益"的思路适用于大多数模块替换型改进研究。
5. **与 prompt-based / 预训练模型的结合潜力**：Table 5 证明 VBM 在 ViT-B/16 + SLCA 下同样有效，未来可探索在更大预训练模型（如 SigLIP、DINOv2）上测试 VBM 的增益边界。

## 关键术语表
- **View-Batch Model (VBM)**：将单个样本的多个增强视图作为一个 batch 单元进行顺序训练的调度模型，目的是拉长重训练间隔。
- **Recall Interval**：同一训练样本两次被重训之间的时间步（batch 数），是控制遗忘-巩固节奏的核心超参。
- **Spacing Effect（间隔效应）**：心理学中表明在最优间隔重复学习可显著提升长期记忆保留率的现象。
- **One-to-Many Divergence Loss**：以弱增强视图预测为锚，对所有强增强视图预测计算 KL 散度的自监督损失。
- **Catastrophic Forgetting（灾难性遗忘）**：神经网络在学习新任务时迅速遗忘旧任务知识的现象。
- **Rehearsal / Non-Rehearsal CL**：是否使用 memory buffer 重放旧数据区分的两种持续学习设定。
- **CIL / TIL / DIL**：Class Incremental Learning（类别增量）、Task Incremental Learning（任务增量）、Domain Incremental Learning（域增量）。

## 可复现要素
- **数据集**：S-CIFAR-10、S-CIFAR-100、S-Tiny-ImageNet、S-ImageNet-R、DomainNet（公开，标准 CL benchmark）。
- **代码**：开源，GitHub：https://github.com/hankyul2/ViewBatchModel。
- **权重**：论文未提及额外预训练权重要求；使用 ResNet-18（随机初始化）与 ViT-B/16（ImageNet 预训练）。
- **关键超参**：View-batch size $V=4$（Figure 6 显示 x3–x4 最优）；弱增强为水平翻转，强增强为 AutoAugment；buffer 尺寸 0/200/500/5120（S-CIFAR-10）及 DomainNet 每类 50 样本。
