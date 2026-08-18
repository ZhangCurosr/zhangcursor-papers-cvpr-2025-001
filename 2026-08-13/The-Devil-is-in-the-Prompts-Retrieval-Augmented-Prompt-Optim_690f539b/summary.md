---
title: "The-Devil-is-in-the-Prompts-Retrieval-Augmented-Prompt-Optim"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Gao_The_Devil_is_in_the_Prompts_Retrieval-Augmented_Prompt_Optimization_for_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 09:46:11"
field: "Text-to-Video 生成"
keywords: ["Text-to-Video", "Prompt Optimization", "Retrieval-Augmented Generation", "Large Language Model", "Video Generation", "Prompt Engineering"]
innovations: ["关系图检索增强的双分支提示词优化框架，首次将训练数据结构化知识图谱与LLM迭代合并结合用于T2V提示词优化", "句子重构模块通过指令微调LLaMA 3.1使优化提示词长度/格式分布与训练数据对齐", "判别器LLM双路择优机制缓解LLM生成不准确和歧义问题"]
benchmarks: ["VBench", "EvalCrafter", "T2V-CompBench"]
---

# 论文速读：The Devil is in the Prompts: Retrieval-Augmented Prompt Optimization for Text-to-Video Generation

## 一句话总结
本文提出 RAPO（Retrieval-Augmented Prompt Optimization），一个面向 Text-to-Video（T2V）生成的检索增强提示词优化框架。通过关系图检索+迭代合并的词汇增强分支、指令微调的句子重构分支，以及判别器 LLM 的双路择优机制，将用户简短提示词转换为与训练数据分布一致的优化提示词，显著提升生成视频的空间与时间质量。

## 研究问题与动机
- T2V 生成模型对输入提示词高度敏感，用户提供的原始提示词通常过于简短、缺乏必要细节，无法充分激发预训练模型的生成潜力。
- 现有 T2I 提示优化方法（如仅用 LLM 添加修饰词）侧重于空间元素增强，但难以有效提升视频的时间维度（如动作流畅性、时序闪烁），且直接迁移到 T2V 效果有限。
- 现有基于 LLM 的提示优化方法缺乏对训练提示词语料库的系统性分析，生成的优化提示词在长度分布、句式结构上与训练分布存在偏差，反而可能干扰模型理解。
- 多对象/复杂构图场景的 T2V 生成质量显著下降，CLIP 文本编码器的混合上下文导致信息绑定不当，而现有工作多从模型侧改进，较少从提示词层面探索。

## 核心贡献（创新点）
1. **提出 RAPO 双分支检索增强提示词优化框架**：首次将关系图检索与 LLM 迭代合并、句子重构、判别器选择有机结合，形成面向 T2V 的端到端提示词优化流水线。
2. **关系图驱动的词汇增强模块**：从 Vimeo25M 训练数据构建场景-修饰词关系图（场景节点+主体/动作/氛围子节点），通过句子向量检索与 LLM 逐步合并，精确注入与用户提示语义相关的修饰词，区别于以往随机添加或纯 LLM 扩写的粗放方式。
3. **指令微调的句子重构模块**：设计专门的数据构造流程（将训练提示改写为格式破坏版本再让 LLM 学习还原），使重构后的提示词在长度和句式分布上与训练数据高度一致，而非简单依赖通用 LLM 改写。
4. **判别器 LLM 驱动的双路择优**：训练 LLM 判别器在"重构路径"和"直接重写路径"之间选择最优提示词，有效缓解 LLM 生成中的不准确和歧义细节问题。
5. **实验验证了提示词分布对齐的重要性**：通过文本统计分析和注意力图可视化，揭示了优化提示词长度分布与训练集的一致性、以及空间关系描述对多对象生成的增益机制。

## 方法详解
RAPO 由三个模块组成，给定用户提示词 $x_i$：

**1. 词汇增强模块（Word Augmentation Module）**
- **关系图构建**：使用 Mistral LLM 从 Vimeo25M（约 2.1M 有效句子）中提取每个训练提示的场景及对应修饰词（主体 subject、动作 action、氛围 atmosphere），构建无向图 $\mathcal{G}$，场景作为核心节点，修饰词作为子节点连接。
- **检索**：使用 all-MiniLM-L6-v2 作为 sentence transformer，提取用户提示特征，通过余弦相似度检索 $\mathcal{G}$ 中 top-k 最相关场景，再获取所有连接修饰词，选取 top-k 修饰词 $\{p_n\}_{n=0}^{k-1}$。
- **迭代合并**：通过固定 LLM $\mathcal{L}$ 将检索到的修饰词逐一合并入提示词，形式化表示为 $x_i^{m+1} = f(x_i^m, p_i^m)$，$m = 0, 1, ..., k-1$。LLM 被指令为"Text Merger"角色，在保留原文描述的基础上以自然方式融合修饰词，输出连贯文本。

**2. 句子重构模块（Sentence Refactoring Module）**
- **数据准备**：构造配对数据集 $D_r = \{(w_i, c_i)\}_{i=1}^{N^r}$，其中 $w_i$ 为模拟词汇增强后但格式被打乱的提示词（通过 LLM 改写训练提示 $c_i$ 获得），$c_i$ 为原始训练提示词，两者语义相同但格式/长度不同。共约 86K 配对。
- **指令微调**：对 LLaMA 3.1 进行 LoRA 微调（rank=64，batch size=32，8 epochs，单卡 A100），指令模板要求模型"精炼句式格式和词长，保留原始主体/动作/场景描述，必要时补充直接的动作描述使句子更动态"，使增强后提示词的格式与训练数据分布对齐。

**3. 提示词选择模块（Prompt Selection Module）**
- **第二分支**：使用冻结 LLM $\mathcal{L}$ 直接按指令重写用户提示词，得到 naive 增强提示词 $x_n$。
- **判别器训练数据**：构造三元组数据集 $D_d = \{(x_i, x_r, x_n, y_d)\}_{i=1}^{N^d}$，其中 $y_d$ 为判别标签（$x_r$ 与 $x_n$ 哪个更优）。通过 T2V 模型生成视频并用对应维度的 VBench 指标自动评估确定 $y_d$，共约 7K 训练样本。
- **指令微调**：对 LLaMA 3.1 进行 LoRA 微调（rank=64，batch size=32，3 epochs，单卡 A100），判别器需判断哪个提示词包含更多合理修饰词且忠实于原语义。

**整体流程**：用户提示词 → 词汇增强（图检索+迭代合并）→ 句子重构（微调 LLM）得 $x_r$；同时用户提示词 → 直接重写得 $x_n$；判别器 LLM 在 $x_r$ 和 $x_n$ 中选择最优者输入 T2V 模型。

## 实验与结果
- **评估基准**：VBench（16 个细粒度维度）、EvalCrafter（约 17 个客观指标）、T2V-CompBench（ composition 任务专用基准）。
- **T2V 模型**：LaVie（扩散架构）、Latte（DiT 架构），均以 Vimeo25M 为主要训练数据之一。
- **对比方法**：原始短提示词、GPT-4 提示优化、Open-sora Prompt Refiner（基于微调 LLaMA 3.1）。
- **VBench 主要结果（Table 4）**：
  - LaVie-RAPO 总分 82.38%（Baseline LaVie 80.89%），多个对象维度从 37.71% 跃升至 64.86%（+27.15pp）。
  - Latte-RAPO 总分 79.97%（Baseline Latte 77.03%），多个对象维度从 29.55% 跃升至 52.78%（+23.23pp）。
  - 在 temporal flickering、human action、object class、spatial relationship 等维度均优于对比方法。
- **EvalCrafter 结果（Table 5）**：LaVie-RAPO 总分 256，text-video alignment 达 74.38，visual quality 达 66.62，均最高。
- **T2V-CompBench 结果（Table 6）**：在 consistent attribute binding（0.692）、action binding（0.635）、object interactions（0.839）等复杂组合任务上全面领先。
- **消融实验**：全模块组合达到最优（VBench 82.38%）；单独词汇增强得 80.37%，单独句子重构得 79.75%，证明三者协同有效（Table 7）。不同 LLM（GPT-4/Mistral/LLaMA）下性能差异微小（82.38%/82.25%/82.10%），说明方法鲁棒（Table 8）。

## 相关工作脉络
- **T2I 提示优化（Hao et al. [17]）**：使用强化学习优化 T2I 提示词以获得更美学的图像。本文定位：聚焦 T2V 的时间维度优化，且采用检索增强而非 RL，解决了 LLM 改写与训练分布不一致的问题。
- **Prompt Auto-Editing (Mo et al. [27])**：T2I 场景下自动决定每个词的权重和注入 timestep。本文定位：T2V 场景，从提示词内容层面而非扩散过程参数层面进行优化。
- **CogVideoX (Yang et al. [44])**：使用 LLM 将短提示扩展为详细提示。本文定位：发现单纯 LLM 扩写会导致提示词过长、与训练分布偏离，提出关系图检索+句子重构的更有针对性的方案。
- **MovieGen (Polyak et al. [31])**：教师-学生蒸馏用于提示优化以提升效率。本文定位：不追求推理效率，而是追求提示词与训练分布的精细对齐，在质量上超越单纯 LLM 扩写。
- **CLIP 文本编码混淆问题（Chen et al. [7], Feng et al. [14]）**：从模型侧（嵌入优化、结构化扩散引导）解决多对象绑定问题。本文定位：从提示词侧通过添加空间关系描述和结构优化来缓解同一问题，提供互补视角。
- **Open-sora Prompt Refiner**：基于微调 LLaMA 3.1 的直接重写方法。本文定位：对比基线，证明本文的双分支+判别器选择+关系图检索方案在格式对齐和多对象生成上显著优于直接重写。

## 局限性与未来方向
- 论文自述局限：仅在 LaVie 和 Latte 两个 T2V 模型上验证，未扩展到更多模型架构（如 SVD、AnimateDiff 等）。
- 关系图构建依赖 LLM 提取修饰词，可能存在提取不完整或噪声问题；仅覆盖 subject/action/atmosphere 三类修饰词，未考虑相机运动、镜头角度等其他视频生成重要维度。
- 判别器训练数据量较小（约 7K），且依赖自动评估指标确定标签，可能存在评估偏差。
- 未来方向：扩展到更多 T2V 模型；深入分析优化机制背后的改进原因；探索更丰富的修饰词类型。

## 研究启发与可借鉴点
- **关系图检索增强策略**可迁移至其他生成模态（如 T2I、3D generation）的提示词优化任务，通过结构化知识图谱指导 LLM 的扩写方向，避免"盲扩"导致的分布偏移。
- **句子重构的"格式破坏-重建"数据构造方法**具有通用性：可应用于任何需要提示词与训练分布对齐的场景，核心思想是构造风格差异大但语义相同的配对数据进行指令微调。
- **双分支+判别器择优的设计范式**可有效缓解 LLM 生成的不确定性：一个分支负责"精准增强"（检索+重构），另一个分支负责"灵活重写"，由判别器权衡，这种设计可复用到其他 LLM 辅助生成任务中。
- **多维度自动评估确定训练标签**的策略：利用现有 T2V 基准的多维度评估指标自动判定哪个提示词更优，可作为后续研究构建大规模提示词偏好数据集的高效方案。
- **提示词长度分布对齐的定量分析**提供了可操作的诊断工具：研究者可通过对比优化前后提示词的长度/结构分布与训练分布的 KL 散度，量化提示词优化的有效性。

## 关键术语表
- **RAPO（Retrieval-Augmented Prompt Optimization）**：本文提出的面向 T2V 生成的检索增强提示词优化框架，通过关系图检索和 LLM 双重优化分支提升生成质量。
- **Relation Graph $\mathcal{G}$**：从训练提示词数据库构建的图结构，节点为场景和修饰词（主体/动作/氛围），边表示语义关联关系，用于检索增强。
- **Word Augmentation**：通过关系图检索相关修饰词，并由 LLM 迭代合并入用户提示词，以丰富描述的丰富性和相关性。
- **Sentence Refactoring**：利用指令微调 LLM 将词汇增强后的提示词重构为与训练数据格式一致的句式结构。
- **Prompt Discriminator**：经过指令微调的 LLM，用于在两条优化路径的结果中选择更适合 T2V 生成的提示词。
- **VBench**：包含 16 个细粒度维度的综合视频生成评估基准，涵盖视觉质量、时序一致性、文本-视频对齐等。
- **EvalCrafter**：包含约 17 个客观指标的综合性视频生成评估基准。
- **T2V-CompBench**：专为组合式 Text-to-Video 生成设计的基准，评估多对象绑定、动态属性绑定、动作绑定等复杂 compositional 任务。

## 可复现要素
- **数据集**：Vimeo25M（25M 文本-视频对，用于构建关系图和训练数据），论文未明确说明是否可直接使用；VBench、EvalCrafter、T2V-CompBench 为标准公开基准。
- **代码/权重**：论文在摘要中提到项目网站为 GitHub，但未给出具体链接；关系图构建使用 Mistral + all-MiniLM-L6-v2；微调使用 LLaMA 3.1（LoRA, rank=64）。
- **关键超参**：关系图检索 top-k 值论文未明确；句子重构训练 8 epochs、判别器训练 3 epochs；LoRA rank=64、batch size=32、单卡 A100。
- **基线对比**：GPT-4 提示优化模板见 supplementary materials；Open-sora Prompt Refiner 为开源实现。
