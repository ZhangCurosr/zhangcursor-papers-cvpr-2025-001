---
title: "Motion-Grounded-Video-Reasoning-Understanding-and-Perceiving"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Deng_Motion-Grounded_Video_Reasoning_Understanding_and_Perceiving_Motion_at_Pixel_Level_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 07:39:43"
field: "视频理解与 grounding"
keywords: ["Motion-Grounded Video Reasoning", "Video Segmentation", "Spatiotemporal Grounding", "Video Reasoning", "Large Multimodal Models", "GROUNDMORE", "Pixel-Level Video Understanding"]
innovations: ["提出 Motion-Grounded Video Reasoning 新任务，要求基于运动相关问题生成时空分割掩码；设计 GROUNDMORE 数据集（1,715 视频、249K 掩码、4 类推理问题）；提出 MORA 基线模型，引入 [LOC] token 实现端到端时空定位"]
benchmarks: ["GROUNDMORE", "Ref-YouTubeVOS", "MeViS", "Perception Test", "NExT-GQA"]
---

# 论文速读：Motion-Grounded Video Reasoning: Understanding and Perceiving Motion at Pixel Level

## 一句话总结
本文提出**Motion-Grounded Video Reasoning**新任务，要求模型根据视频中的运动相关问题生成目标对象的**时空分割掩码**（像素级视觉答案），从而同时评估模型的隐式时空推理与感知能力。围绕该任务，作者构建了大规模基准数据集 **GROUNDMORE**（1,715 个视频片段、249K 个物体掩码、4 类问题），并提出基线模型 **MORA**（基于 LLaVA + SAM），在 GROUNDMORE 上取得 SOTA 结果。

## 研究问题与动机
- **现有任务只能覆盖运动理解的某个单一维度**：动作识别依赖空间特征而忽略细粒度时序模式；时序动作定位缺少物体级空间分析；时空动作检测聚焦预定义人为主体的动作；视频推理分割仅做空间定位、缺乏时序感知。
- **运动本质上是具有双向交互与时序上下文关系的时空概念**，现有任务（如 AVA、MeViS）要么忽略了交互对象的完整空间感知，要么忽略了相邻事件之间的时序因果/序列关系。
- **纯文本答案无法精确表达运动的时空属性**：语言难以避免偏差（如球赛视频中问"play"易答"balls"），且无法直接说明"运动何时/何地发生"；视觉化的像素级响应才是更可解释的评估方式。
- **隐式推理与像素级输出的结合仍属空白**：已有的 Referring VOS 和 Video Reasoning Segmentation 任务依赖显式表达式或忽略时间定位，缺乏基于问答格式的隐式时空推理挑战。

## 核心贡献（创新点）
1. **提出 Motion-Grounded Video Reasoning 新任务**：要求模型输入视频+运动相关问题→输出时空分割掩码，填补了 Referring VOS/动作检测与运动视频推理之间的空白，同时覆盖空间上下文、时序上下文、运动抽象和像素级输出四个维度。
2. **构建 GROUNDMORE 大规模数据集**：收录 1,715 个 720p 视频片段、7,577 个问题、249K 物体掩码，涵盖家庭/动物/球赛/户外活动四类场景，设计 4 类问题（因果、序列、反事实、描述），在数量级和挑战性上均超过现有相关数据集（如 MeViS）。
3. **系统性地评估现有基线并揭示其运动理解能力不足**：包括纯视觉 RVOS 模型、图像推理分割模型（LISA/PixelLM）、视频推理分割模型（PG-Video-LLaVA/VISA）及两阶段基线，结果表明即便在其他基准上表现优异的模型在此新任务上普遍失败。
4. **提出 MORA 基线模型并达到当前 SOTA**：基于 LISA 框架（LLaVA + SAM），引入 **[LOC] 特殊 token** 实现端到端的时序定位，辅以时空池化策略；较最强视频推理接地基线 PG-Video-LLaVA 平均相对提升约 11.28% J&F。

## 方法详解
### 整体架构
MORA（Motion-Grounded Video Reasoning Assistant）基于 **LISA** 框架构建（本身集成了 **LLaVA** 和 **SAM**），并针对视频模态进行了扩展：
- **视频编码**：采用 Video-ChatGPT 中的**时空池化策略**（spatiotemporal pooling）进行高效帧编码，在保留关键时序信息的同时降低计算开销。
- **空间分割**：沿用 LISA 中的 **[SEG] token**，通过 MLP 投影到 SAM 解码器，逐帧生成空间分割掩码。
- **时序定位（核心创新）**：引入新的 **[LOC] 特殊 token**，其嵌入经由 MLP 层解码为**二元时序掩码**（temporal mask），用于抑制帧级掩码解码过程中的误激活，弥补纯帧级推理缺失的时序感知能力。

### 训练流程（两阶段）
1. **预训练阶段**：使用 **Ref-YouTubeVOS** 和 **MeViS** 两个现有数据集（将原始文本标注转换为问答格式以强制模型遵循指令）进行 20 个 epoch 的预训练，**不含时序定位模块**，可作为 zero-shot 评估使用。
2. **微调阶段**：在 GROUNDMORE 训练集上，加入时序定位模块，再微调 20 个 epoch，得到完整 MoRA-ft 模型。

### 损失与评估
- 评估采用 **Jaccard index (J)** 和 **F-measure (F)**，以及综合指标 **J&F**（反映预测与 GT 掩码的 IoU 和轮廓精度）。
- 论文未详细给出训练损失函数，但从架构可推断：空间分割损失来自 SAM 解码分支；时序定位损失来自 [LOC] token 对应的二元掩码监督。

## 实验与结果
### 数据集与评估设置
- **GROUNDMORE**：1,715 个视频片段，平均时长 9.61s，涵盖 3,942 种不同物体，分为因果（Causal）、序列（Sequential）、反事实（Counterfactual）、描述（Descriptive）4 类问题。
- **所有方法均在 zero-shot 设置下评测**。

### 基线对比（Table 3 关键结果）
| 类别 | 最佳模型 | Overall J&F |
|---|---|---|
| RVOS 基线 | HTR | 17.49 |
| 两阶段基线 | SeViLA+HTR | **22.34** |
| MORA (Ours) | — | **23.13** |
| PG-Video-LLaVA | — | 11.17 |
| VISA | — | 5.31 |

- MORA 较最强 RVOS 基线（HTR, 17.49）绝对提升 **+5.64**；较最强两阶段基线（SeViLA+HTR, 22.34）提升 **+0.80**；较最强视频推理接地模型 PG-Video-LLaVA 相对提升约 **21.5%**（摘要所述）/ 平均绝对提升约 11.28。
- 不同问题类型存在差异：在 Causal/Descriptive 问题上，SeViLA/ViLA 两阶段基线表现优于 MORA（因强推理能力未受接地模块干扰）；在 Sequential/Counterfactual 等时序相关问题上，MORA 的 [LOC] 时序定位头发挥关键作用。

### 消融实验（Table 5）
- **去掉 [LOC] 时序定位分支**：MoRA-ft w/o loc 较 MoRA-ft 下降约 **1.53 J&F**，各问题类型均有损失，证明时序定位模块不可或缺（相对提升约 6.0%）。
- **零样本评估**：MoRA-zs（仅预训练，无 GROUNDMORE 微调）即达到 23.13，说明预训练阶段已学习到有效的视频 grounding 能力。

### 数据集诊断（Table 4）
- **隐式推理验证**：将问题替换为 GT 答案后，J&F 平均提升 **14.29**，证明隐式推理确实是核心挑战。
- **时序上下文验证**：仅输入带时间标注的运动密集片段（去除时序上下文）后，J&F 平均下降 **4.68**，证实时序信息的重要性。

## 相关工作脉络
1. **视频动作理解任务（Action Recognition / Temporal Action Localization / Spatiotemporal Action Detection）**：AVA、Kinetics 等聚焦单一维度（空间或时序），缺乏运动对象的完整交互感知；本文通过问答格式统一整合空间+时序上下文。
2. **Referring VOS / MeViS（MeViS, Refer-YouTubeVOS）**：依赖**显式**物体指代表达进行分割，无法评估隐式推理；本文使用**隐含的问答形式**作为输入，要求模型推导目标对象。
3. **Video Reasoning Segmentation（VISA, VideoReasonSeg, ReVOS）**：仅做空间定位，缺乏**时序定位**能力，且基于已有 VOS 数据集构建限制了问题设计空间；本文强调**时空双维**输出与更丰富的 Q&A 设计。
4. **Video-LLM  grounded 模型（PG-Video-LLaVA）**：基于 Video-LLaVA 做像素接地，但对隐式推理任务表现不佳（因 LLM 冗余响应导致大量误激活）；本文通过 [SEG]+[LOC] token 端到端联合优化规避此问题。
5. **两阶段 Video QA + Segmentation 范式（SeViLA, ViLA+HTR）**：先推理后分割的两阶段方法在 GROUNDMORE 上表现强劲，但缺少统一的时空建模；本文 MORA 提供端到端的统一框架。

## 局限性与未来方向
- **MORA 架构较为简单**：时空池化策略不可避免地会造成信息损失；[LOC] 时序定位头设计也较简单，有待改进。
- **LLaVA 的视觉-语言对齐在动态场景中仍有局限**：可能需要更运动敏感的语料进行训练。
- **GROUNDMORE 规模有限**（1,715 视频/249K 掩码），相对其他大规模视频数据集（如 NExT-GQA 的 5.4K 视频）偏小。
- **现有最强 RVOS 模型（如 HTR 在 Ref-YouTubeVOS 上 J&F=67.1）** 在 GROUNDMORE 上仅 10.41，说明任务难度大幅提升，但同时也意味着模型仍有巨大进步空间。
- 代码与权重**均未开源**（论文中注明"Project available at: https://groundmore.github.io/"，但未明确提供代码仓库链接）；**GROUNDMORE 数据集也未开放**。

## 研究启发与可借鉴点
1. **用 [LOC] token 实现端到端时序定位**：通过特殊 token + MLP 解码为二元掩码的方式，将时序边界学习融入 LLM 统一框架，无需额外后处理，可迁移至任何需要时序定位的视频 grounding 任务。
2. **数据集诊断方法值得借鉴**：通过"替换为 GT 答案"和"仅用运动密集片段"两种诊断实验，量化隐式推理与上下文缺失带来的难度，为后续研究提供任务分解的评估视角。
3. **将 Referring VOS/MeViS 数据转换为问答格式进行预训练**的策略：在不改变原始数据标注的前提下，利用 LLM 的指令遵循能力适配新的 QA 范式，是一种低成本的数据利用方式。
4. **四类型问题设计**（因果/序列/反事实/描述）为系统化评估运动理解的不同维度提供了模板，后续研究可在此基础上扩展或调整问题类型。
5. **视觉答案优于纯文本**的思路：对于运动理解等任务，像素级输出比语言输出更具可解释性和精确性，可与团队的方向（如视频 grounding/分割）结合探索。

## 关键术语表
- **Motion-Grounded Video Reasoning**：一种新型视频理解任务，要求模型根据运动相关的隐式问题生成目标物体的时空分割掩码作为像素级答案。
- **GROUNDMORE**：为 Motion-Grounded Video Reasoning 任务构建的大规模数据集，包含 1,715 个视频、7,577 个问题及 249K 物体掩码，设计 4 类问题以评估不同维度的运动推理能力。
- **MORA**（Motion-Grounded Video Reasoning Assistant）：本文提出的基线模型，基于 LISA（LLaVA + SAM）框架，新增 [LOC] token 和时序定位头以实现端到端的时空推理与分割。
- **[LOC] token**：新增的特殊 token，用于在语言空间中编码时序边界信息，经 MLP 解码为二元时序掩码，抑制帧级分割中的误激活。
- **[SEG] token**：LISA 框架中用于触发 SAM 空间分割的特殊 token，MORA 沿用该 token 进行帧级空间掩码生成。
- **Spaciotemporal pooling**：Video-ChatGPT 中提出的视频编码策略，在时空两个维度上进行池化以实现高效的视频帧特征提取。
- **J&F metric**：Jaccard index（J）与 F-measure（F）的综合指标，分别衡量掩码的 IoU 和轮廓精度，是视频分割任务的常用评估标准。

## 可复现要素
- **GROUNDMORE 数据集**：**论文未公开**（项目主页链接为 https://groundmore.github.io/，但未明确提供下载链接）
- **MORA 代码与权重**：**论文未开源**（文中声明提供 project page，但未给出 GitHub 仓库地址）
- **预训练数据**：Ref-YouTubeVOS（公开）、MeViS（公开）
- **基座模型**：LLaVA-7B/13B（开源）、SAM（开源）
- **关键超参**：预训练 20 epoch、微调 20 epoch；其他超参数论文未提及
