---
title: "Debiasing-Multimodal-Large-Language-Models-via-Noise-Aware-P"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_Debiasing_Multimodal_Large_Language_Models_via_Noise-Aware_Preference_Optimization_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 06:00:30"
field: "多模态大语言模型对齐与去偏"
keywords: ["模态偏差", "偏好优化", "噪声鲁棒", "多模态大语言模型", "幻觉缓解", "DPO", "Box-Cox变换"]
innovations: ["提出 NaPO 算法，通过负 Box-Cox 变换融合 DPO 的 BCE 与 MAE 实现噪声鲁棒的偏好优化", "构造 RLAIF-V-Bias 去偏偏好优化数据集，利用模态掩码扰动生成语言/视觉有偏响应作为负样本", "设计基于奖励边距的动态噪声评估机制，自适应调整噪声鲁棒系数和 loss 权重"]
benchmarks: ["VLind-Bench", "Object HalBench", "AM-BER", "MMHalBench"]
---

# 论文速读：Debiasing-Multimodal-Large-Language-Models-via-Noise-Aware-P

## 一句话总结
本文提出通过偏好优化范式缓解多模态大语言模型（MLLM）的模态偏差问题，构造了去偏偏好优化数据集 RLAIF-V-Bias，并设计了噪声感知偏好优化（NaPO）算法，结合噪声鲁棒的 MAE 与 DPO 中的 BCE，动态调整噪声鲁棒系数，在有效减少模态偏差的同时显著降低了幻觉。

## 研究问题与动机
- **模态偏差问题**：MLLM 在图文输入下常过度依赖单一模态（语言先验或视觉细节），忽略其他模态的关键信息，导致错误回答和无关内容输出。
- **现有方法的局限**：已有去偏方法多依赖大规模监督微调（SFT），存在损失 MLLM 已有知识的风险；且专门针对 MLLM 去偏的偏好优化数据集极为匮乏。
- **自动构造数据的噪声挑战**：基于模型自动生成的有偏响应中混入了部分高质量（无噪）样本，标准偏好优化算法（如 DPO 的 BCE 损失）对噪声数据敏感，易过拟合劣质样本。
- **偏好优化的优势**：通过扩大偏好（无偏）响应与拒绝（有偏）响应的概率差距进行对齐，可在不破坏已有知识的前提下调整模型的响应偏好。

## 核心贡献（创新点）
1. **构造了 RLAIF-V-Bias 去偏偏好优化数据集**：通过在 RLAIF-V 基础上引入模态掩码扰动生成语言/视觉有偏响应作为负样本；与现有工作本质不同在于这是首个专为 MLLM 模态去偏设计的偏好优化数据集，而非通用多模态偏好数据。
2. **提出了 NaPO 噪声感知偏好优化算法**：将 DPO 的 BCE 损失通过负 Box-Cox 变换与噪声鲁棒的 MAE 结合，实现收敛速度与噪声鲁棒性的可控平衡；与既有方法本质不同在于首次将 Generalized Cross Entropy 框架引入 DPO 损失设计以应对自动构造数据的噪声。
3. **设计了动态噪声评估与自适应权重机制**：基于奖励边距（LogP margin）动态计算噪声鲁棒系数 q，并引入基于边距的动态 loss 权重 γ，实现训练过程中对不同噪声水平样本的自适应调整；与固定系数方法本质不同在于可根据每条数据的质量实时调节优化策略。
4. **实验验证了去偏与减幻觉的双重效果**：在 VLind-Bench、Object HalBench、AM-BER、MMHalBench 四个基准上均取得显著提升；与 MDPO 等幻觉缓解方法相比，本文方法在模态偏差基准上平均提升约 19%，且在多个幻觉指标上优于基线。

## 方法详解
- **有偏响应生成**：对输入 $x=(v, t)$ 分别遮蔽视觉信息 $v$（用 [MASK] 替换）生成语言有偏响应 $y_{lb}=MLLM([MASK]; t)$，以及遮蔽文本信息 $t$ 生成视觉有偏响应 $y_{vb}=MLLM(v; [MASK])$，两者均附加至 RLAIF-V 作为负样本，形成 RLAIF-V-Bias 数据集。
- **NaPO 损失函数**：基于 Generalized Cross Entropy，引入噪声鲁棒系数 $q \in (0, 1]$：
  $\mathcal{L}_{\mathrm{NaPO}} = \frac{1}{q}\left(1 - \sigma\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)^q\right)$。当 $q \to 0$ 时退化为 DPO 的 BCE 损失；$q=1$ 时接近 MAE，噪声鲁棒性最强。
- **动态噪声鲁棒系数 q**：通过对 100 条样本的人工标注发现，噪声样本的 LogP 边距普遍小于无噪样本，因此设定 $q = 1 - \sigma(\alpha \cdot \psi(x, y_w, y_l))$，其中 $\psi$ 为奖励边距（语言偏差用 $\psi_\mu$，视觉偏差用 $\psi_\Sigma$），$\alpha$ 为缩放因子（语言偏差 $\alpha=0.5$，视觉偏差 $\alpha=0.01$）。
- **最终优化目标**：引入基于 $\psi_\Sigma$ 计算的动态权重 $\gamma_i$，将原始 DPO 损失与两种有偏 NaPO 损失加权融合：
  $\mathcal{L}_\gamma = \gamma_{y_l}\mathcal{L}_{\mathrm{DPO}} + \gamma_{y_{lb}}\mathcal{L}_{\mathrm{NaPO}}(y_{lb}) + \gamma_{y_{vb}}\mathcal{L}_{\mathrm{NaPO}}(y_{vb})$。

## 实验与结果
- **实验设置**：基础模型为 LLaVA-v1.5-7B（及 13B），$\beta=0.1$，学习率 5e-7，训练 4 个 epoch，batch size=4，8×A100 80GB。
- **评测基准**：VLind-Bench（语言先验 LP↑、常识偏差 CB↑）、Object HalBench（CHAIRs↓、CHAIRi↓）、AM-BER（覆盖率↑、幻觉率↓）、MMHalBench（GPT-4 评分↑、幻觉率↓）。
- **主要结果（LLaVA-v1.5-7B）**：
  - VLind-Bench：CB=58.9（vs. DPO 的 39.4，+19.5%）、LP=44.0（vs. DPO 的 25.4，+18.6%）。
  - Object HalBench：CHAIRs=25.7↓（vs. DPO 的 32.0）、CHAIRi=6.2↓（vs. DPO 的 8.5）。
  - AM-BER：CHAIRs=4.0↓、覆盖↑、HalRate=20.7↓。
  - MMHalBench：Score=3.31↑、HalRate=0.35↓。
- **13B 模型**：CB=42.1、LP=25.1、CHAIRs=23.7、CHAIRi=5.9，同样全面优于 DPO 基线。
- **结论**：方法平均在模态偏差基准上提升约 19%，同时在幻觉评估上显著优于 DPO 和 MDPO 基线。

## 相关工作脉络
- **RLAIF-V [58]**：开源模型反馈迭代的多模态偏好优化数据集；本文在其基础上扩展出含偏样本的 RLAIF-V-Bias，定位差异在于本文聚焦模态去偏而非通用偏好对齐。
- **MDPO [48]**：面向幻觉的偏好优化，通过降低无图时偏好样本的概率来抑制幻觉；本文方法直接降低有偏响应的概率，且同时处理语言偏差和视觉偏差。
- **VCD [29]**：通过视觉对比解码缓解幻觉；本文从训练端（偏好优化）入手而非推理端（解码策略），方法正交可结合。
- **VLind-Bench [28]**：专门评测 MLLM 语言先验与常识偏差的基准；本文在其上进行系统性评估，建立了去偏效果与幻觉减少之间的关联。
- **Ghosh et al. [18]**：证明 MAE 是对称损失具有噪声鲁棒性；本文将其思想引入 DPO 的 BCE 框架，通过 Box-Cox 变换实现两者平滑过渡。
- **β-DPO [53]**：动态调整 DPO 的超参数 β；本文关注的是噪声鲁棒系数 q 的动态调整，二者作用于不同维度。

## 局限性与未来方向
- **噪声处理的泛化性待验证**：NaPO 在更广泛的 LLM 合成数据场景下的噪声鲁棒性尚未充分检验。
- **"偏差是否总是有害"的反思**：论文末尾承认视觉上偏差（bias）并不一定总是有害，现有方法假设偏差有害可能过于简化。
- **仅评估了语言/视觉两种模态偏差**：未涉及音频、视频等其他模态的偏差问题。
- **动态权重的超参敏感性**：缩放因子 α 的选取依赖人工调参，缺乏自动化搜索机制。
- **基线对比的公平性限制**：部分对比方法使用不同的基础模型和训练数据，难以直接比较。

## 研究启发与可借鉴点
- **Box-Cox 变换融合 BCE 与 MAE**：这一损失设计思路可迁移到其他存在噪声标签的偏好优化任务（如 RLHF、KTO）中，提升自动构造数据的训练稳定性。
- **基于 LogP 边距的噪声估计**：利用奖励边距与数据质量的反比关系进行噪声评估，是一种无需额外标注的自监督信号，可推广至其他 LLM 合成数据的清洗场景。
- **掩码扰动构造负样本**：通过遮蔽单模态信息诱导偏差响应的策略，可扩展到视频-语言、音频-语言等多模态组合的去偏数据构造。
- **动态 loss 加权机制**：基于边距的动态权重 $\gamma_i$ 设计可借鉴于多任务学习中的损失平衡，避免手动设定权重比例。
- **去偏与减幻觉的联合优化**：本文揭示了模态偏差与幻觉之间的内在联系，启示后续研究可将两类问题统一建模，而非分别处理。

## 关键术语表
- **Modality Bias（模态偏差）**：MLLM 过度依赖某一模态（语言或视觉）而忽视其他模态信息的现象，分为语言偏差和视觉偏差两种。
- **NaPO（Noise-Aware Preference Optimization）**：噪声感知偏好优化算法，通过负 Box-Cox 变换将 DPO 的 BCE 与 MAE 融合，动态调整噪声鲁棒系数。
- **RLAIF-V-Bias**：本文构造的包含语言/视觉有偏响应的多模态偏好优化数据集，基于 RLAIF-V 扩展而来。
- **BCE（Binary Cross-Entropy）**：二元交叉熵损失，DPO 中用于拟合奖励边距的标准损失，收敛快但对噪声敏感。
- **MAE（Mean Absolute Error）**：平均绝对误差损失，具有对称性因此对噪声数据更鲁棒，但收敛较慢。
- **LogP Margin（LogP 边距）**：偏好与拒绝响应之间的对数概率差，用于评估数据质量，边距越小暗示噪声越多。
- **VLind-Bench**：专门评测 MLLM 语言先验和常识偏差的基准，包含 CB（常识偏差）和 LP（语言先验）两个指标。
- **CHAIR Score**：对象幻觉评估指标，CHAIRs 为响应级幻觉率，CHAIRi 为对象级幻觉率。

## 可复现要素
- **数据集**：RLAIF-V-Bias，基于 RLAIF-V 构造；代码和数据已开源（https://github.com/zhangzef/NaPO）。
- **代码/权重**：代码已开源，论文未提及预训练权重发布。
- **关键超参**：β=0.1，学习率=5e-7，训练 4 epoch（7B）/1 epoch（13B），batch size=4，α（语言偏差）=0.5，α（视觉偏差）=0.01，q 和动态权重截断至 [0.01, 1] 范围。
- **硬件**：8×A100 80GB。
