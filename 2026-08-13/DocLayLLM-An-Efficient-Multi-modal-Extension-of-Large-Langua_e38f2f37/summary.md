---
title: "DocLayLLM-An-Efficient-Multi-modal-Extension-of-Large-Langua"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Liao_DocLayLLM_An_Efficient_Multi-modal_Extension_of_Large_Language_Models_for_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 07:01:29"
field: "文档智能 / 多模态大模型"
keywords: ["Text-Rich Document Understanding", "Multimodal LLM", "Chain-of-Thought", "OCR-dependent", "Document Layout Analysis"]
innovations: ["轻量级LLM多模态扩展架构：通过VE/VP和LE/LP投影层将视觉patch与OCR坐标直接注入LLM输入，无需额外文档编码器", "CoT Pre-training：面向非QA格式数据，利用LLM逐步推理生成CoT格式训练数据，提升文档理解能力", "CoT Annealing：在SFT阶段从CoT视角设计数据退火策略，逐步提高直接回答数据比例"]
benchmarks: ["DocVQA", "VisualMRC", "FUNSD", "CORD", "SROIE", "InfoVQA", "TabFact"]
---

# 论文速读：DocLayLLM: An Efficient Multi-modal Extension of Large Language Models for Text-rich Document Understanding

## 一句话总结
本文提出了 DocLayLLM，一种高效的 LLM 多模态扩展方法，通过将文档视觉 patch tokens 与 OCR 边界框坐标经轻量投影层直接注入 LLM 输入序列，无需额外的文档编码器（DE），结合提出的 CoT Pre-training 和 CoT Annealing 技术，以仅约 3M 预训练数据 + 300K SFT 数据的轻量训练规模，在文本丰富文档理解（TDU）任务上超越现有 OCR-dependent 方法和 OCR-free 方法。

## 研究问题与动机
- **TDU 任务需要同时理解文本、布局与图像**：文字密集且排版复杂的文档需综合运用 OCR 文本、2D 位置信息和视觉特征，现有方法在多模态融合上存在明显不足。
- **OCR-free 方法资源消耗巨大**：依赖高分辨率图像输入和大体积训练数据（常包含下游评测数据集本身），计算开销高且影响零样本泛化能力评估。
- **OCR-dependent 方法依赖额外文档编码器（DE）**：如 LayoutLMv3 等 DE 的文档理解能力不及 LLM 本身，需大量联合训练来对齐特征空间，消耗显著算力；直接在 LLM 中输入 OCR 坐标文本表示又会导致输入过长、推理变慢。
- **CoT 在文档理解训练中的作用未被充分探索**：现有方法多将 CoT 仅用于推理阶段，缺乏对 CoT 数据生成（预训练）和数据质量调度（SFT 阶段 annealing）的系统性设计。

## 核心贡献（创新点）
1. **轻量级 LLM 多模态扩展架构**：通过简单的 VE/VP（视觉嵌入+投影）和 LE/LP（布局嵌入+投影）将 patch tokens 与 OCR 坐标嵌入后与文本拼接输入 LLM，无需额外 DE，本质区别在于保留了 LLM 原始注意力结构，避免了 Cross-Attention 等机制对原生推理能力的干扰。
2. **CoT Pre-training 策略**：面向非 QA 格式标注数据，利用 LLM 逐步推理生成 CoT 格式的指令微调数据（布局类型推理、表格结构分析、几何关系分析三类 CoT），实现从非 QA 数据高效生成高质量推理数据，与仅做格式转换的方法有本质区别。
3. **CoT Annealing 技术**：在 SFT 阶段从 CoT 视角重新设计数据退火策略——逐步提高无 CoT 直接回答数据比例，最终仅使用直接回答数据完成训练，解决长 CoT 回答噪声和泛化能力下降问题，这是首次从 CoT 角度探索数据退火。
4. **高效的训练资源利用**：仅用 ~3M 预训练数据和 ~300K SFT 数据（LoRA rank=64），总训练耗时约 36 A100 天，在多个 TDU benchmark 上超越数据量为其数倍的现有 SOTA 方法。

## 方法详解
- **文档处理流程**：首先用 OCR 引擎提取文本内容 $Text_{1:N}$ 及对应边界框 $Box_{1:N} = [x_0, y_0, x_1, y_1]$；同时将文档图像分割为 $Patch_{1:M}$（分辨率固定为 224×224，共 196 个视觉 patch token）。
- **输入构造**：将固定 prompt 模板、OCR 文本及其坐标、问题和 patch 整合为自然语言表达：`Given the document patches: {Patch_1:M} and the document text contents and locations in the form of "text, [left, top, right, bottom]": {(Text_i, Box_i)_1:N} {Question}`
- **多模态嵌入与投影**（公式 1-3）：
  - 文本部分使用 LLM 原生 text embedder：$Emb_T = TE(TextContent)$
  - 视觉部分：$Emb_V = VP(VE(Patch_{1:M}))$，VE/VP 分别为视觉嵌入层和线性投影层
  - 布局部分：$Emb_L = LP(LE(Box_{1:N}))$，LE/LP 分别为布局嵌入层和线性投影层
  - VE 和 LE 初始化权重来自 LayoutLMv3，VP 和 LP 随机初始化
- **LLM 解码**（公式 4）：按固定顺序 $\pi$ 排列 $(Emb_T, Emb_V, Emb_L)$ 形成序列输入 LLM，生成 Answer。
- **CoT Pre-training 三类 CoT**：
  1. **Layout-Type-Related CoT**：定位区域 → 找最近边界框并判断其布局类型 → 推断目标区域布局类型
  2. **Table-Structure-Aware CoT**：基于 XY-Cut 算法，先识别表头 → 按列对齐输出元素 → 确定目标元素
  3. **Geometry-Related CoT**：恢复边界框 → 分析垂直/水平投影重叠 → 计算中心点相对位置 → 计算最小欧氏距离
- **CoT Annealing**：SFT 初期仅使用带 layout-aware CoT 的数据，逐步增加无 CoT 的直接回答数据比例，最终 100% 使用无 CoT 数据完成训练。

## 实验与结果
- **预训练数据**（共 3.1M 条）：RVL-CDIP、PubLayNet、PubTabNet、DocLayNet、DocBank、DocILE 及 LayoutLLM 的 Document Dense Description 数据。
- **SFT 数据**：300K 条，基于 LayoutLLM 的 layout-aware 文档 QA 数据，应用 CoT Annealing。
- **评测基准**：FunSD、CORD、SROIE、POIE（KIE）；DocVQA、InfoVQA、VisualMRC、DeepForm、KLC（VQA）；WTQ、TabFact（Table Understanding）；共 11 个数据集。
- **主要结果**（DocLayLLM_Llama3）：
  - **vs OCR-dependent**（Table 1）：DocVQA=78.36，VisualMRC=58.55，Avg VQA=68.46；FUNSD=84.11，CORD=71.34，SROIE=84.36，Avg KIE=79.94，全面超越 LayoutLLM_Llama2（需 5.7M 数据）。
  - **vs OCR-free**（Table 3/4）：零样本下 DocVQA=77.79，SROIE=76.59，POIE=75.13，超越 TextMonkey（使用了 benchmark 训练集）；在有 SFT 数据条件下（Table 4），DocVQA=86.52，InfoVQA=58.36，VisualMRC=327.91，均超越 DocOwl 1.5、TextSquare 等 SOTA OCR-free 方法。
- **最强结果**：Table 4 中 Avg VQA=157.60，Avg KIE=58.90，Avg Table=70.99，均在对应评测框架下刷新 SOTA。

## 相关工作脉络
- **LayoutLLM [45]**：使用 LayoutLMv3 作为额外 DE 编码文档信息后输入 LLM，需大规模联合训练对齐特征；本文直接用轻量投影层替代 DE，不破坏 LLM 原生结构。
- **LayTextLLM [43]**：将 bbox 作为 token 输入 LLM；本文使用独立 LE/LP 嵌入布局坐标，避免文本化坐标导致的输入过长问题。
- **DocLLM [67]**：在每个 attention 层通过 cross-attention 注入布局特征，需额外投影层且可能干扰 LLM 原生计算；本文仅在输入层注入多模态特征，保持 LLM 结构不变。
- **OCR-free MLLMs（Monkey/TextMonkey/DocKylin 等）**：依赖高分辨率图像和大体积训练数据；本文以 OCR+ 轻量多模态注入方式，用更少数据实现更好性能。
- **ICL-D3IE [14]/LATIN-Prompt [70]**：以文本形式输入 OCR 坐标；本文证明了独立布局嵌入的有效性，避免了超长文本输入。
- **Data annealing（MiniCPM/DataComp-LM/Llama-3.1）**：传统数据退火从数据质量角度设计；本文首次从 CoT 视角引入退火策略。

## 局限性与未来方向
- **仍依赖 OCR 引擎**：作为 OCR-dependent 方法，OCR 识别错误会直接传递至模型，影响最终效果（尽管比直接输入 OCR 文本的方法更鲁棒）。
- **图像分辨率固定为 224×224**：仅捕捉文档概览布局，可能丢失细粒度文字内容，对于超密集文本场景可能存在信息损失。
- **预训练数据量相对 OCR-free SOTA 仍偏少**：虽然效率高，但在更复杂多轮对话或开放域文档理解任务上的泛化能力有待验证。
- **未来方向**：探索更高分辨率图像处理、端到端 OCR-理解联合训练、以及将 CoT 方法迁移至其他文档分析任务（如表单填写、合同审核）。

## 研究启发与可借鉴点
1. **轻量多模态注入范式**：用简单的 VE/VP 和 LE/LP 投影替代复杂的 DE 或 cross-attention 机制，为 LLM 多模态扩展提供了简洁高效的设计范式，可直接迁移至其他领域（如医疗报告、法律文档理解）。
2. **CoT Pre-training 的数据生成思路**：将非 QA 格式标注数据通过 CoT 转化为推理增强型训练数据，这一思路可迁移至表格理解、地理信息提取等结构化数据任务。
3. **CoT Annealing 的训练调度策略**：从推理质量视角重新设计数据退火，对任何依赖 CoT 的模型训练都有借鉴价值，尤其适合推理密集型任务。
4. **VE/LE 以 LayoutLMv3 初始化 + LP/VP 随机初始化的设计**：平衡了文档先验知识的利用与新模态适配的灵活性，可作为多模态 LLM 扩展的通用初始化策略参考。

## 关键术语表
**TDU（Text-Rich Document Understanding）**：面向文字密集且排版复杂的文档的理解任务，需综合文本、布局和视觉信息。
**CoT Pre-training**：利用链式推理逐步生成标注数据的技术，将非 QA 格式的训练数据转化为带推理步骤的指令微调数据。
**CoT Annealing**：在 SFT 阶段逐步提高无 CoT 直接回答数据比例的数据退火策略，以平衡推理能力和回答精确性。
**OCR-dependent / OCR-free**：前者依赖 OCR 引擎提取文本和坐标后输入模型；后者直接从文档图像端到端生成答案。
**Document Encoder（DE）**：如 LayoutLMv3，专门编码文档文本、视觉和布局特征的辅助编码器。
**LoRA（Low-Rank Adaptation）**：通过低秩矩阵微调大模型参数的高效微调技术，本文 rank=64。

## 可复现要素
- **数据集**：预训练使用 RVL-CDIP、PubLayNet、PubTabNet、DocLayNet、DocBank、DocILE 等公开数据集；SFT 使用 LayoutLLM 的 layout-aware 文档 QA 数据。评测使用 FUNSD、CORD、SROIE、POIE、DocVQA、InfoVQA、VisualMRC、DeepForm、KLC、WTQ、TabFact 等公开基准。
- **代码/模型**：已开源，见 https://github.com/whlscut/DocLayLLM
- **关键超参**：基础模型 Llama3-8B-Instruct / Llama2-7B-Chat；图像分辨率 224×224（196 个 patch token）；LoRA rank=64；VE 和 LE 使用 LayoutLMv3 初始化；训练硬件 8× NVIDIA A100-80G，预训练约 30 A100 天，SFT 约 6 A100 天。
