---
title: "HarmonySet-A-Comprehensive-Dataset-for-Understanding-Video-M"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Zhou_HarmonySet_A_Comprehensive_Dataset_for_Understanding_Video-Music_Semantic_Alignment_and_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 07:10:40"
field: "多模态理解与评测"
keywords: ["video-music alignment", "multimodal large language models", "dataset", "temporal synchronization", "semantic alignment", "instruction tuning", "HarmonySet"]
innovations: ["首个视频-音乐多维对齐数据集HarmonySet，覆盖节奏同步、情感对齐、主题一致性、文化相关性四维度", "多步骤人机协作标注框架，结合人工时序锚定与机器描述生成", "双轨评估基准OE+MC，首次系统评测MLLMs的视频-音乐对齐理解能力"]
benchmarks: ["HarmonySet-OE", "HarmonySet-MC"]
---

# 论文速读：HarmonySet-A-Comprehensive-Dataset-for-Understanding-Video-M

## 一句话总结
本文提出了 **HarmonySet**，这是首个面向多模态大语言模型（MLLMs）的视频-音乐语义对齐与时间同步理解数据集，包含 48,328 个高质量视频-音乐对及其多维度标注（节奏同步、情感对齐、主题一致性、文化相关性），并配套构建了开放回答（OE）与多项选择（MC）双轨评测基准。

---

## 研究问题与动机

1. **现有数据集标注不足**：已有视频-音乐数据集（如 TT-150K、MovieClips、SVM-10K 等）仅提供配对内容或基础描述，缺乏对视频-音乐间细粒度语义对齐和时间同步的深度标注，难以支撑 MLLMs 的训练。
2. **模型理解停留在表面**：当前视频-音频 MLLMs（如 VideoLLaMA2、video-SALMONN、Macaw-LLM 等）对视频-音乐关系的分析仅能给出表层解读，无法捕捉节奏同步、情感对齐、主题连贯性等多维深层语义。
3. **高质量标注成本高昂**：视频-音乐对齐标注需同步观看视频与音乐、识别关键转场时刻，且涉及主观判断（个人喜好、文化背景），标准化难度极大。
4. **缺乏系统化评估基准**：现有视频语言基准（如 VideoMME、MM-Vet、Q-Bench）聚焦通用视觉语言任务，缺少针对视频-音乐跨模态对齐的专项评测体系。

---

## 核心贡献（创新点）

1. **提出 HarmonySet 数据集**：构建 48,328 对视频-音乐数据，覆盖 6 大类 43 子类别，每个样本均标注节奏同步、情感对齐、主题一致性、文化相关性四个维度——与已有数据集仅依赖自动化推荐对话或简单描述的本质区别在于，HarmonySet 提供了结构化、多维度的语义与时间同步对齐标注。
2. **设计多步骤人机协作标注框架**：先由人工标注关键时间戳与四维标签，再由 Gemini 1.5 Pro 基于时标与视频元数据生成详细上下文感知描述——与完全自动化生成或纯人工标注的区别在于，该框架以人工见解锚定关键转场，同时借助机器提升标注效率与描述丰富度。
3. **构建双轨评估基准（HarmonySet-OE 与 HarmonySet-MC）**：OE 采用 LLM-as-a-judge 评估开放回答质量，MC 由 GPT-4o 生成干扰项构成多选题——与依赖 BLEU/ROUGE 等传统文本指标的区别在于，该框架专门针对视频-音乐对齐任务设计了语义匹配与时序同步评测维度。

---

## 方法详解

### 3.1 视频采集
- 采用分层标签体系，涵盖 **Life & Emotions、Arts & Performance、Travel & Events、Sports & Fitness、Knowledge、Technology & Fashion** 六大主类别及 43 个子类别。
- 生成 293 个关键词指导 YouTube Shorts 视频爬取。
- 使用 **PANNs** 模型验证音乐存在性，仅保留含用户添加背景音乐且与视觉内容互补的视频，人工剔除无音乐样本。

### 3.2 标注构建
**同步标注（Synchronization Annotation）**：
- 标注员识别视频关键转场时刻（如场景切换、情节转折），评估音乐变化是否与视觉转场对齐，并记录精确时间戳。

**多维标签标注（Labeling）**：
- 采用结构化标签体系，四个维度包括：
  - **节奏与同步（Rhythm & Synchronization）**
  - **主题与内容（Theme & Content）**：标签包括 "strongly related"、"indirectly related"、"unrelated"、"conflicting"
  - **叙事增强（Narrative Enhancement）**：标签包括 "enhancing"、"suggesting"、"reversing"、"independent narrative"、"no supplement"
  - **情感（Emotion）**、**文化（Culture）**
- 每个视频由 **3 名独立标注员** 标注，最终取共识结果。

**质量控制（Quality Control）**：
- 专职审稿人交叉核查关键时间戳准确性、标签一致性及事实依据。

**自动标注增强（Automated Annotation Curation）**：
- 使用 **Gemini 1.5 Pro** 输入视频内容、人工验证标注及视频元数据（标题、描述），生成针对四个维度的详细上下文描述。

### 3.3 指令微调数据集
- 构建 44,470 对视频-音乐的指令微调数据，每对附带结构化解释。
- 视频时长 2.96–63.38 秒（平均 31.5 秒），总时长 458.8 小时，平均标注长度 352.65 词。

### 3.4 评估框架
**HarmonySet-OE**（3,858 对）：
- 开放回答任务，要求模型分析视频-音乐对齐关系。
- 采用 **LLM-as-a-judge** 评估（参照 [64] 方法），比较模型输出与 ground truth。

**HarmonySet-MC**（3,858 对）：
- 多项选择扩展，由 GPT-4o 基于正确选项生成 3 个干扰项。
- 干扰项设计准则：主题相关、长度/句式相似、语义可区分。
- 每个样本含 4 道多选题，分别对应节奏、情感、主题、文化四个维度。

---

## 实验与结果

### 评测设置
- **基线模型**：闭源 Gemini-1.5 Pro；开源 VideoLLaMA2（Qwen2-7B）、video-SALMONN（Vicuna-13B-v1.5）
- **零样本设置**：统一 prompt 输入
- **视频输入**：开源模型使用 16 帧；Gemini-1.5 Pro 支持长上下文，采样 1 fps

### 主要结果（HarmonySet-OE）

| 模型 | R&S | Theme | Emotion | Culture | Overall |
|------|-----|-------|---------|---------|---------|
| Gemini-1.5 Pro | 5.43 | 5.18 | 5.15 | 4.51 | — |
| VideoLLaMA2（未微调） | 4.15 | 4.29 | 4.38 | 3.05 | — |
| video-SALMONN | 2.83 | 3.55 | 3.38 | 2.12 | — |
| **VideoLLaMA2 + HarmonySet** | **5.55** | **5.06** | **5.26** | **4.62** | — |

- VideoLLaMA2 微调后 R&S 提升 **+1.40**，Culture 提升 **+1.57**，在多数类别上超越未训练的开源模型。
- Gemini-1.5 Pro 在 R&S（+1.28）和 Culture（+1.46）上显著领先未训练模型，得益于其长上下文与时间戳输出能力。

### HarmonySet-MC 结果

| 模型 | R&S (Acc.) | Theme (Acc.) | Emotion (Acc.) | Culture (Acc.) |
|------|------------|--------------|----------------|----------------|
| Gemini-1.5 Pro | 41.84% | 45.45% | 44.43% | 50.40% |
| Video-LLaMA2 | 21.76% | 48.95% | 52.76% | 24.29% |
| Video-LLaMA2 (HarmonySet) | 10.63% | 54.16% | 47.32% | 36.66% |
| **Human** | **85.26%** | **88.19%** | **84.49%** | **93.81%** |

- 人类性能显著优于所有模型，表明任务挑战性高，模型与人类理解仍存在巨大差距。

### 消融实验
1. **帧数影响**（Table 6）：16 帧 → 32 帧性能提升；64 帧反而下降，说明短视频（<1 分钟）中过多帧引入冗余或过拟合。
2. **人机协作有效性**（Table 4）：使用完全自动化生成的 10k 样本训练 vs. HarmonySet 10k 样本训练，后者在 R&S（+0.27）、Theme（+0.54）、Emotion（+0.38）、Culture（+0.45）均显著更优，验证人工标注价值。
3. **AI 生成音乐评估**（Figure 5）：微调后模型能更好区分人类创作与 AI 生成配乐，提供更精确的时序与语义分析。

### 数据集质量
- 共识评估：随机抽样 10%，92% 标注获得"High"共识（Table 3）。

---

## 相关工作脉络

1. **视频-音乐数据集**：TT-150K [1]、MovieClips [2]、SVM-10K [5] 仅提供配对数据或基础推荐，无细粒度语义/时间标注；BGM909 [4] 含节奏/和弦信息但缺情感与语义分析；MMTrail [6] 有指令微调描述但未深入视频-音乐对齐；HarmonySet 在标注深度与维度完整性上显著领先。
2. **音视频预训练数据集**：AudioSet [33]、VGGSound [34]、FSD50K [35]、ESC-50 [36] 聚焦通用音频事件识别，非音乐语义对齐；HarmonySet 专门针对音乐与视频的跨模态语义建模。
3. **视频-音频 MLLMs**：VideoLLaMA2 [23]、video-SALMONN [24]、Macaw-LLM [26]、VALOR [25] 可处理视频-音频输入，但缺乏专项对齐训练；本文证明其在节奏同步、文化理解等维度存在明显短板。
4. **视频语言基准**：VideoMME [13]、MM-Vet [12]、Q-Bench [14]、EgoSchema [10]、MMBench [11] 评估通用视频理解能力；HarmonySet-OE/MC 首次专门针对视频-音乐对齐任务设计评测。
5. **音乐生成相关工作**：M2Ugen [67]、VidMuse [29]、Diff-BGM [4] 等关注视频到音乐生成；本文反向利用理解能力评估 AI 生成配乐与视频的和谐度。

---

## 局限性与未来方向

1. **视频时长限制**：数据集以 1 分钟以内短视频为主，长视频的时间同步理解尚未充分探索。
2. **时标覆盖率有限**：仅 58% 数据包含关键时间戳标注，部分视频缺乏精细时序对齐信息。
3. **模型性能距人类差距显著**：即使在最优设置下，模型多项选择准确率（最高 54.16%）仍远低于人类（88%+），表明视频-音乐深层对齐理解仍是开放挑战。
4. **评估指标依赖 LLM-as-a-judge**：开放回答评估依赖 LLM 判分，可能存在系统性偏差。
5. **未来方向**：开发专为视频-音乐分析定制的 MLLM 架构；探索视频-音乐跨模态知识迁移；扩大长视频覆盖；完善时标标注比例。

---

## 研究启发与可借鉴点

1. **多步骤人机协作标注范式**：人工锚定时序关键点 → 人工标注多维标签 → 机器生成详细描述——此流程可迁移至其他需要时序对齐的多模态标注任务（如视频-文本叙事对齐、语音-唇语同步等）。
2. **四维对齐标注体系**：节奏同步、情感对齐、主题一致性、文化相关性构成结构化分析框架，可作为视频-音频理解任务的通用标注标准。
3. **OE + MC 双轨评测设计**：开放回答评估语义深度，多项选择提供客观准确率，二者互补——此设计可推广至其他跨模态理解基准构建。
4. **帧数选择的实证经验**：短视频场景下 16–32 帧为最优区间，64 帧反而退化，提示在资源受限场景下需审慎选择采样策略而非盲目增加。
5. **AI 生成内容评估应用**：微调后的模型可有效区分人类创作与 AI 生成配乐，为 AIGC 内容质量评估提供新思路。

---

## 关键术语表

**HarmonySet**：本文提出的首个面向 MLLMs 的视频-音乐语义对齐与时间同步理解数据集，包含 48,328 对视频-音乐及其多维度标注。

**Rhythm & Synchronization (R&S)**：视频画面节奏与音乐节拍/转折点的时序对齐程度，是本文强调的核心能力之一。

**LLM-as-a-judge**：使用大语言模型作为评判器，对开放回答进行语义质量评分的方法（参见 [64]）。

**HarmonySet-OE**：Open-Ended 版本评测集，包含 3,858 对视频-音乐及开放回答型标注，评估模型生成式理解能力。

**HarmonySet-MC**：Multiple-Choice 版本评测集，由 GPT-4o 生成干扰项构建的多选题，提供客观准确率评测。

**VideoLLaMA2**：开源视频多模态大语言模型（基于 Qwen2-7B），本文主要微调与评测对象。

**PANNs**：Pre-trained Audio Neural Networks，用于视频音乐存在性验证的音频模式识别模型 [61]。

**Multi-step human-machine collaborative framework**：多步骤人机协作标注框架，结合人工时间戳标注、多维标签与机器自动生成描述的高效标注流程。

---

## 可复现要素

- **数据集**：HarmonySet 已公开，项目页面 https://harmonyset.github.io/
- **代码/权重**：论文未提及开源代码；模型权重可基于开源 VideoLLaMA2 复现微调
- **关键超参**：视频输入 16 帧（最优）；采样频率 1 fps（Gemini）；训练样本数 10k（消融实验）；视频时长 2.96–63.38 秒（平均 31.5 秒）
- **标注工具**：Gemini 1.5 Pro 用于自动描述生成；GPT-4o 用于 MC 干扰项生成

---
