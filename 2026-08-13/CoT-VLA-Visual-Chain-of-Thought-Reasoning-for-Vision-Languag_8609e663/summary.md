---
title: "CoT-VLA-Visual-Chain-of-Thought-Reasoning-for-Vision-Languag"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Zhao_CoT-VLA_Visual_Chain-of-Thought_Reasoning_for_Vision-Language-Action_Models_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:59:29"
field: "具身智能/机器人学习"
keywords: ["VLA", "Chain-of-Thought", "机器人操作", "视觉推理", "多模态生成", "模仿学习"]
innovations: ["首次将视觉链式思维引入VLA，以子目标图像作为中间推理步骤", "设计混合注意力机制（因果注意力+全注意力）统一处理图像生成与动作预测", "利用无动作视频数据增强VLA视觉推理能力"]
benchmarks: ["LIBERO", "Bridge-V2", "Franka-Tabletop"]
---

# 论文速读：CoT-VLA: Visual Chain-of-Thought Reasoning for Vision-Language-Action Models

## 一句话总结
论文提出 CoT-VLA，一种将视觉链式思维（Visual Chain-of-Thought）引入视觉-语言-动作（VLA）模型的方法，通过在生成动作前先自回归预测未来图像帧（子目标图像）作为中间推理步骤，结合混合注意力机制，显著提升机器人操作任务的泛化能力，在真实世界任务中较最强VLA基线提升17%，仿真基准提升6%。

## 研究问题与动机
- **现有VLA缺乏显式推理能力**：当前VLA模型直接从观测映射到动作，缺少对复杂操作任务所需的时间规划与中间推理步骤，导致可解释性差且性能受限。
- **难以利用无动作标注的视频数据**：传统VLA仅依赖带动作标注的机器人演示数据，而互联网上存在大量无动作标注的普通视频数据，未能被有效利用来提升视觉推理能力。
- **已有CoT扩展依赖抽象中间表示**：现有将链式思维引入机器人的工作多使用文本描述、关键点或边界框等抽象表示，需要额外的预处理管道。
- **需要可解释且高效的中间推理形式**：子目标图像天然存在于机器人演示数据中，无需额外标注即可作为中间推理步骤，同时为动作生成提供明确目标。

## 核心贡献（创新点）
1. **首次将视觉链式思维引入VLA框架**：以子目标图像生成作为中间推理步骤，区别于Prior工作使用文本计划、关键点或边界框等抽象表示，提供更直观的视觉推理形式。
2. **混合注意力机制设计**：对文本/图像生成使用因果注意力，对动作预测使用全注意力，使所有动作维度能同时交互预测，与传统VLA的顺序token预测形成本质差异。
3. **联合利用有动作机器人数据和无动作视频数据**：子目标图像生成步骤仅需图像数据，从而解锁EPIC-KITCHEN-100等大规模无动作视频数据的利用，提升视觉推理能力。
4. **7B参数VLA在多维基准上达到SOTA**：在LIBERO、Bridge-V2和Franka-Tabletop上全面验证，真实世界任务较OpenVLA提升17%，仿真基准提升6%。

## 方法详解
- **总体框架**：基于7B参数的VILA-U（统一多模态基础模型），在两个阶段进行训练：预训练阶段在Open X-Embodiment数据集和无动作视频数据集（EPIC-KITCHEN-100、Something-Something V2）上联合训练；微调阶段在目标机器人平台上针对特定任务进行闭环节制微调。
- **两阶段推理过程**：
  - 阶段一（视觉CoT推理）：给定当前观测图像$\mathbf{s}_t$和语言指令$l$，自回归预测未来第$t+n$帧的子目标图像$\hat{\mathbf{s}}_{t+n}$，公式：$\hat{\mathbf{s}}_{t+n} \sim P_\theta(\mathbf{s}_{t+n}|\mathbf{s}_t, l)$
  - 阶段二（动作生成）：基于当前观测、语言指令和生成的子目标图像，预测长度为$m$的动作序列：$\{\hat{\mathbf{a}}_t, ..., \hat{\mathbf{a}}_{t+m}\} \sim P_\theta(\{\mathbf{a}_t, ..., \mathbf{a}_{t+m}\}|\mathbf{s}_t, l, \mathbf{s}_{t+n})$
- **混合注意力机制**：图像和文本生成使用因果注意力（causal attention）保证自回归生成顺序性；动作预测使用全注意力（full attention）使所有动作token并行交互，提升动作预测一致性。
- **视觉token预测损失**：使用残差量化（RQ-VAE）将图像编码为$16 \times 16 \times 4$个离散token，通过深度transformer逐层预测残差token，损失函数为$\mathcal{L}_{visual} = -\sum_j \sum_{d=1}^D \log P_\delta(k_{jd}|k_{j,<d})$
- **动作token预测损失**：每个连续动作维度量化为256个离散bin，将文本tokenizer中最不常用的256个token复用为动作bin token，使用交叉熵损失$\mathcal{L}_{action} = -\sum_{i=1}^m \log P_\theta(\mathbf{a}_{t...t+m}|l, s_t, s_{t+n})$
- **总损失函数**：$\mathcal{L} = \mathcal{L}_{action} + \mathcal{L}_{visual}$
- **闭环节制部署**：执行预测动作序列后获取新观测，迭代执行，算法伪代码见原文Algorithm 1

## 实验与结果
- **评估设置**：三个互补场景——LIBERO仿真基准、Bridge-V2真实机器人平台（45k演示数据）、Franka-Tabletop真实机器人（10-150演示数据/任务）
- **LIBERO仿真结果**：CoT-VLA-7B平均成功率81.13%（+4.63% vs OpenVLA fine-tuned），在Spatial、Object、Goal、Long四个子任务上均取得最佳或最具竞争力结果，长程任务（Long）提升尤为显著（69.0% vs 53.7%）
- **Bridge-V2真实实验结果**：在四项泛化类别中，Motion泛化达到60%（较OpenVLA 45%提升15个百分点），Semantic泛化达到50%（较OpenVLA 40%提升10个百分点）；Visual和Language泛化与OpenVLA相当
- **Franka-Tabletop真实实验结果**：在6项任务上取得最高平均成功率，单指令和多指令任务均表现优异
- **消融实验结论**：
  - 三组件有效性：action chunking > hybrid attention > visual CoT，每项均带来持续提升
  - 预训练贡献：加入OpenX+无动作视频预训练后，Franka-Tabletop任务相对提升46.7%（53.7%→78.8%）
  - 视觉推理价值：使用ground-truth子目标图像替代生成图像，OOD任务成功率绝对提升40%，表明视觉推理能力直接决定动作执行效果
- **最强结果**：LIBERO平均成功率81.13%，较SOTA VLA提升6%；真实世界任务综合提升17%

## 相关工作脉络
1. **OpenVLA [29]**：开源VLA模型，在OpenX上微调预训练VLM直接预测动作；本文在其基础上引入视觉CoT和混合注意力，利用无动作视频数据增强推理能力。
2. **Octo [59]**：在OpenX上预训练的通用机器人策略模型，不使用VLM初始化；本文基于VLM架构，强调语言理解和视觉推理的结合。
3. **Diffusion Policy [10]**：基于扩散模型的模仿学习算法；本文对比显示在复杂多指令任务上VLA类模型更具优势，但Diffusion Policy在单指令简单任务上仍有竞争力。
4. **SUSIE [2]**：两阶段方法，使用指令引导图像编辑生成目标图像再用goal-conditioned policy生成动作；本文与SUSIE的核心差异在于CoT-VLA的子目标图像由自回归模型直接生成而非扩散模型，且端到端统一训练。
5. **EmbodiedCoT [45] / Robotic Control via Embodied CoT [44]**：将链式思维引入机器人，使用文本描述或关键点对作为中间表示；本文使用子目标图像作为CoT，保留像素级视觉信息，无需额外预处理。
6. **Video Language Planning [14] / Dreamitate [35]**：生成未来图像轨迹用于机器人控制；本文聚焦单步子目标图像生成而非完整轨迹，并结合VLA架构实现闭环节制。

## 局限性与未来方向
- **推理速度瓶颈**：生成256个图像token后再生成动作token，比直接动作生成慢约7倍（action chunk size=10时），图像生成为主要瓶颈，可探索快速图像生成或快速LLM推理技术。
- **图像生成质量限制**：自回归图像生成质量低于SOTA扩散模型，可结合Janus、Emu3等新一代统一多模态模型改进。
- **动作chunking的连续性缺陷**：不同chunk间可能出现不连续动作，且缺乏高频反馈，可通过时间平滑或per-step预测方法改进。
- **OOD视觉推理泛化不足**：虽利用无动作视频预训练，但当前计算约束下对全新任务的子目标生成泛化仍有限，需结合世界模型或更大规模视频生成模型。

## 研究启发与可借鉴点
1. **无动作视频数据的有效利用范式**：通过子目标图像生成这一中间步骤，将EPIC-KITCHEN等大规模无动作视频纳入训练，为"如何让VLA模型从非机器人视频中学到有用知识"提供了可复用的框架设计思路。
2. **混合注意力机制的可迁移性**：因果注意力处理生成任务、全注意力处理条件化输出的设计模式，可推广至其他需要"先理解/生成中间表示再决策"的具身智能场景。
3. **视觉CoT替代抽象CoT的灵感**：相比文本/关键点等抽象中间表示，子目标图像保留了丰富的空间细节和视觉上下文，在复杂操作任务中可能具有更强的判别性，值得在其他视觉-动作任务中探索。
4. **消融设计范式**：逐步剥离visual CoT、hybrid attention、action chunking三个组件验证各自贡献，且通过ground-truth子目标图像ablation揭示视觉推理上限，实验设计严谨完整。
5. **与视频生成/世界模型结合的空间**：论文明确指出快速视频生成模型的进展可直接提升CoT-VLA能力，为后续将更强大的world model（如Gaia-1、Pandora等）作为视觉推理引擎提供了明确方向。

## 关键术语表
**CoT (Chain-of-Thought)**：链式思维，通过生成中间推理步骤提升大模型复杂任务解决能力的方法，本文将其扩展至视觉域。
**VLA (Vision-Language-Action)**：视觉-语言-动作模型，将预训练VLM与机器人控制结合，直接从视觉观测和语言指令映射到机器人动作的端到端模型。
**Subgoal Image**：子目标图像，作为视觉链式思维中间表示的未来状态图像，由模型自回归生成，用于指导后续动作序列生成。
**Action Chunking**：动作分块，一次性预测多个时间步的动作序列而非单步动作，提升执行效率和策略平滑性。
**Hybrid Attention**：混合注意力机制，对图像/文本生成使用因果注意力（自回归），对动作预测使用全注意力（并行交互）。
**Open X-Embodiment (OpenX)**：大规模跨机器人平台演示数据集集合，本文主要预训练数据来源之一。
**RQ-VAE (Residual Quantization VAE)**：残差量化变分自编码器，将连续视觉输入编码为多层离散token，支持自回归图像生成。
**Bridge-V2**：包含45k语言标注轨迹的真实机器人操作数据集，用于评估模型的泛化和语言 grounding 能力。

## 可复现要素
- **数据集**：Open X-Embodiment（公开）、EPIC-KITCHEN-100（公开）、Something-Something V2（公开）、LIBERO（公开）、Bridge-V2（公开）、Franka-Tabletop（论文未明确说明开源状态）
- **代码/权重**：论文未明确声明代码开源状态，官方主页为 https://cot-vla.github.io/；VILA-U基座模型权重可参考其原始论文
- **关键超参**：图像分辨率256×256，视觉token结构16×16×4（残差深度4），动作量化bin数256，action chunk size=10，未来帧预测步长n从数据集特定范围均匀采样；完整超参见supplementary material
- **训练策略**：预训练阶段联合OpenX和无动作视频数据，微调阶段仅用目标平台演示数据；优化LLM backbone、projector和depth transformer，冻结vision tower
