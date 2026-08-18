---
title: "Molmo-and-PixMo-Open-Weights-and-Open-Data-for-State-of-the"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Deitke_Molmo_and_PixMo_Open_Weights_and_Open_Data_for_State-of-the-Art_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 07:39:11"
field: "多模态大模型"
keywords: ["Vision-Language Model", "Open-Source VLM", "PixMo Dataset", "Dense Captioning", "Grounding", "Counting", "Overlapping Multi-crop", "Text-only Dropout"]
innovations: ["构建完全不依赖闭源VLM蒸馏的PixMo数据集（语音描述+dense caption+2D点标注）", "提出交叠多裁剪（overlapping multi-crop）策略提升ViT高分辨率建模能力", "Point-then-Count链式推理策略实现计数SOTA，配合text-only dropout提升预训练质量"]
benchmarks: ["VQA v2.0", "ChartQA", "DocVQA", "CountBenchQA", "PixMo-Count", "MMMU", "MathVista", "RealWorldQA", "AI2D", "InfographicVQA", "TextVQA"]
---

# 论文速读：Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Vision-Language Models

## 一句话总结
本文提出了 **Molmo** 系列最先进的开源视觉-语言模型（VLM），其核心贡献是构建了名为 **PixMo** 的全新数据集（全部由人工标注或合成生成，**完全不依赖任何外部 VLM 的蒸馏数据**）；在此基础上训练的 Molmo-72B 模型在学术基准上排名第一、Human Evaluation Elo 排名第二，超越了 Claude 3.5 Sonnet 和 Gemini 1.5 Pro/Flash，仅次于 GPT-4o。

## 研究问题与动机
- **核心问题**：目前最强的 VLM 均为闭源（GPT-4o、Gemini、Claude），而开源模型要么性能差距较大，要么严重依赖闭源 VLM 生成的合成数据进行蒸馏（如 ShareGPT4V），学术界缺少"从零构建高性能 VLM"的基础知识。
- **现有方法的不足**：
  1. 早期全开源模型（如 LLaVA-1.5）与 SOTA 差距显著。
  2. 近期强开源模型（PaliGemma、LLaVA-OneVision、Cambrian 等）的训练数据大量依赖闭源 VLM 生成的合成 caption/QA 数据，本质上是蒸馏产物。
  3. 高质量、细粒度多模态数据（尤其是 dense caption 和 grounding 数据）收集成本高、质量难以保证。
  4. 现有 VLM 在 OCR 密集型任务、计数任务和自然图像理解方面仍有明显短板。

## 核心贡献（创新点）
1. **全新 PixMo 数据集体系**：首次完全绕过闭源 VLM，通过"语音转述+LLM润色"收集 712k 张高分辨率详细 caption 图片，并引入 2.3M 级 2D 点标注 grounding 数据，数据集本身即构成可复用的科研资源。
2. **交叠多裁剪策略（Overlapping Multi-crop）**：在不降低 tiled 图像分辨率的前提下，允许 ViT crop 之间部分重叠以保留边界 patch 的上下文信息，相比无交叠裁剪显著提升 cap F₁ 与下游平均分。
3. **文本只 dropout（Text-only Dropout）与 Length-conditioned Captioning**：预训练阶段仅对 text tokens 施加 dropout，迫使模型更依赖视觉信号而非语言先验；同时引入长度提示使 caption 生成成为更强的预训练目标。
4. **Efficient Multi-annotated Image 训练**：将同一张图的多条 annotation 拼接为单条长序列并施加跨 annotation 的 attention mask，减少约 2/3 的图片编码计算，缩短训练时间超 50%。
5. **Point-then-Count 链式推理**：利用点标注数据训练模型"先定位、后计数"的思维链方式，使 Molmo 在 CountBenchQA 和新增 PixMo-Count 基准上均取得 SOTA 成绩。

## 方法详解
### 模型架构
Molmo 采用标准 VLM 架构（Vision Encoder + Connector + Decoder-only LLM），四大组件：
1. **预处理**：将输入图像划分为多个正方形 crop（含全局低分辨率缩略图），crop 之间允许重叠。
2. **ViT 编码器**：使用 OpenAI ViT-L/14 336px（亦可替换 SigLIP 或 MetaCLIP），对每个 crop 独立提取 per-patch 特征。
3. **Vision-Language Connector**：
   - 拼接 ViT 倒数第三层与第十层特征（略优于单层）。
   - 对每 2×2 patch window 使用 **多头注意力池化**（mean of patches 作为 query），替代简单拼接，性能更优。
   - MLP 将 pooled 特征投影到 LLM 嵌入空间。
4. **Decoder-only LLM**：使用 OLMo-7B、Qwen2 7B/72B 或 OLMoE-1B-7B（MoE）。

### 关键设计细节
- **Vision token 排列**：全局低分辨率 crop tokens 在前，高分辨率 crop tokens 按行主序排列，插入特殊 token 标记序列起止及行分隔。
- **Dropout 策略**：LLM 残差 dropout；预训练阶段**仅对 text tokens 启用 dropout**（fine-tuning 不开启），提升 caption 质量和下游性能。
- **多标注高效训练**：将一张图的多条 text annotation tokens 拼接为长序列，mask 使每条 annotation 内部互相 attend，但不同 annotation 之间不 attend，等价于独立训练但节省约 2/3 图像编码开销。

### PixMo 数据集构建
| 数据集 | 规模 | 构建方法 | 用途 |
|---|---|---|---|
| **PixMo-Cap** | 712k 图 / 1.3M transcript / 平均 196 words | 标注者语音描述 60-90s → STT → LLM 润色 | 预训练 dense caption |
| **PixMo-AskModelAnything** | 73k 图 / 162k QA | 标注者与纯 LLM 交互式编辑 QA | Fine-tuning 指令遵循 |
| **PixMo-Points** | 223k 图 / 2.3M Q-points / 14k 解释点 | 标注者点选目标并描述、枚举所有实例 | Fine-tuning grounding & counting |
| **PixMo-CapQA** | 165k 图 / 214k QA | LLM 仅基于 ground-truth caption 自问自答 | Fine-tuning |
| **PixMo-Docs** | 255k 图 / 2.3M QA | LLM 生成 HTML 代码 → 基于代码生成 QA | 文档/图表理解 |
| **PixMo-Clocks** | 826k 示例 | 合成钟面图 + 随机时间 QA | 时钟读数技能 |
| **PixMo-Count** | 36k 训练 + 验证/测试各 540 | 非 VLM 目标检测 + 点标注 + QA | 计数技能 |

### 训练流程
- **预训练**：仅在 PixMo-Cap 上进行 4 轮 AdamW，含 length hint prompt（90% 概率），**跳过传统的 connector-only 预训练阶段**，改用更高学习率+更短 warmup 直接联合训练。
- **Fine-tuning**：混合 PixMo 自采数据 + 多个开源学术数据集（VQA v2.0、ChartQA、DocVQA 等 15+ 数据集），采样率按数据集大小的平方根比例，手动 down-weight 超大合成集（PlotQA、FigureQA 等），pointing 任务 up-weight。
- **Style tag**：所有学术数据集使用风格前缀（如 "vqa2:"）避免评测时答案风格被污染。
- **点坐标输出**：归一化到 0-100 的纯文本坐标，多点按从上到下、从左到右排序并编号。

## 实验与结果
### 评测设置
- **11 个学术基准**：AI2D、ChartQA、VQA v2.0、DocVQA、InfographicVQA、RealWorldQA、MMMU、MathVista、CountBenchQA（CBQA）、PixMo-Count（PCQA）、TextVQA（隐含在 DocQA 类中）。
- **Human Evaluation**：15k 多样图文 prompt，870 名标注者 pairwise 比较，总计 >325k 评分，用 Bradley-Terry 模型计算 Elo 分。

### 主要结果（表 1）

| 模型 | 11-Avg Acc | Elo Score | Elo Rank |
|---|---|---|---|
| **GPT-4o** (proprietary) | **78.5** | **1079** | **1** |
| Gemini 1.5 Pro | 78.3 | 1074 | 3 |
| Claude 3.5 Sonnet | 76.7 | 1069 | 4 |
| **Molmo-72B** (ours) | **76.9** (≈78.5) | **1032** | **13** |
| Qwen2-VL-72B | 79.4 | 1037 | 12 |
| Llama-3.2V-90B | 74.5 | 1063 | 5 |
| Molmo-7B-D | ~73 | ~1014 | — |
| MolmoE-1B | ~70 | ~1010 (≈GPT-4V) | — |

- **Molmo-72B** 学术平均分最高，Elo 排名第二（仅次于 GPT-4o），超越 Claude 3.5 Sonnet 和 Gemini 1.5 Pro/Flash。
- **MolmoE-1B**（1B MoE 参数）在学术基准和 Elo 上几乎匹配 GPT-4V。
- **Molmo-7B-O / 7B-D** 介于 GPT-4V 与 GPT-4o 之间。
- **计数任务**：Molmo 在 CBQA（89.4%）和 PCQA（86.3%）均领先所有开源与闭源模型，归功于 point-then-count 能力。
- **时钟读数**：Molmo 全系列大幅超越所有对比 VLM（包括闭源），但仍落后于专用非 VLM 模型。
- **Android Control**：Molmo-72B 达到 88.7% 低级准确率 / 69.0% 高级准确率，与原文报告结果相当。
- **Chatbot Arena 独立 Elo**：Molmo-72B 在所有开源模型中排名第一，低于 GPT-4o 和 Claude 3.5 Sonnet。

### Ablation 关键结论（表 2-5）
- **交叠裁剪** vs 无交叠：cap F₁ 54.1 vs 53.4，11-avg 76.9 vs 75.7。
- **Attention pooling** vs stacking：76.9 vs 76.1。
- **Text-only dropout**：预训练 cap F₁ 54.1 vs 53.7。
- **Length conditioning**：cap F₁ 54.1 vs 53.0（关闭）。
- **PixMo-Cap 规模缩放**：0 → 712k 图像，11-avg 从 74.9 提升至 76.9。
- **PixMo-Cap 原始 transcript vs GPT-4o 生成 caption**：cap F₁ 54.1 vs 52.9，性能相当，证明人工语音数据不弱于蒸馏数据。
- **Point order**：有序点（top-down, left-right）CBQA 89.4% vs 无序 85.4%。
- **Point-then-Count**：CBQA 89.4% vs count-only 87.9% vs count-then-point 81.5%。

## 相关工作脉络
1. **LLaVA 系列（LLaVA-1.5, LLaVA-OneVision）**：早期全开源 VLM 的代表，但性能已落后；LLaVA-OneVision 等更强模型依赖 ShareGPT4V 等闭源 VLM 蒸馏数据。本文与其本质区别在于**完全不用外部 VLM 生成训练数据**。
2. **PaliGemma / Phi-3.5-Vision**：使用部分闭源数据的高质量开源模型，但训练数据不完全公开。本文强调数据**100% 开源且无蒸馏**。
3. **ShareGPT4V / xGen-MM-interleave / Cambrian**：典型的"用 GPT-4V 生成 caption/QA → 蒸馏到开源模型"范式。本文用人工语音描述+LLM润色替代，证明质量相当且更具科学透明度。
4. **Qwen2-VL**：当前最强开源 VLM 之一，在 OCR 密集 benchmark 上略优于 Molmo，但在计数和自然图像理解上落后；本文定位是**完全开放 pipeline + 更优计数/ grounding 能力**。
5. **Prismatic VLMs (Karamcheti et al.)**：同时探索 VLM 设计空间的工作，但与本文不同的核心在于数据集的构建方式（非蒸馏 vs 可能依赖合成数据）。
6. **Fuyu-8B / MiniGPT-4 / BLIP-2**：早期 VLM 架构探索，性能已明显落后；本文架构较传统但通过数据质量与训练技巧实现突破。

## 局限性与未来方向
- **推理/复杂推理任务仍有差距**：Molmo 在 MMMU、MathVista 等复杂推理 benchmark 上落后于 GPT-4o 和部分专有模型，作者承认训练数据集中缺乏高级推理相关数据。
- **Elo 排名与 GPT-4o 仍有差距**：虽然超越多数专有模型，但在 Chatbot Arena 人类偏好评估中仍落后于 GPT-4o 和 Claude 3.5 Sonnet。
- **计数泛化受限**：改变测试时 crop 数量会影响 pointing 能力，需额外 high-res post-training 缓解。
- **数据成本虽降低但仍不菲**：712k 张高分辨率图像的语音标注需要大量人工时间和 STT/LLM 处理管线。
- **未来方向**：作者指出 pointing 数据为 VLM agent（机器人、网页 agent）的导航/操作提供了新思路；后续可增加推理/数学类数据以补齐短板。

## 研究启发与可借鉴点
1. **语音描述替代文字写 caption**：让标注者语音描述 60-90 秒可大幅提升 caption 详细度和效率，同时保留 audio receipt 防止 VLM 作弊蒸馏——这一范式可迁移到任何需要高质量图像描述的领域。
2. **交叠多裁剪策略**：对 ViT 的 crop 设计做了关键改进，解决边界 patch 上下文缺失问题，可普遍应用于任何基于 ViT 的多模态模型高分辨率推理。
3. **Text-only Dropout 在预训练中的应用**：仅对 text tokens 施加 dropout 以迫使模型依赖视觉信号，这一简单技巧值得在其他多模态预训练中尝试。
4. **Point-then-Count 思维链**：将 grounding 能力与链式推理结合用于计数任务，既提升准确性又提供可解释性，可推广至其他需要定位+推理的视觉任务。
5. **Multi-annotated Image 高效训练**：将多图注序列拼接并加 attention mask 的方法可显著降低训练成本，适用于任何含有多条图像标注的数据集。
6. **纯人工+合成数据路线的可行性验证**：证明了不用闭源 VLM 蒸馏也能达到接近 SOTA 的性能，为开源 VLM 社区提供了重要的方法论参考。

## 关键术语表
- **Molmo（Multimodal Open Language Model）**：本文提出的开源 VLM 系列，支持从 1B 到 72B 多种规模，全部开源权重、数据和训练代码。
- **PixMo（Pixels for Molmo）**：本文构建的 7 个全新视觉-语言数据集的总称，全部不依赖外部 VLM 生成。
- **Dense Caption**：对图像进行详细描述（平均 196 词，远超 COCO 的 11 词）的图像-文本对，用于预训练。
- **Overlapping Multi-crop**：将图像切分为多个有重叠的正方形 crop 送入 ViT，以在保持高分辨率的同时保留边界 patch 上下文。
- **Attention Pooling（Vision-Language Connector）**：使用多头注意力机制将 2×2 patch window 的特征聚合为一个向量，替代简单的特征拼接。
- **Text-only Dropout**：仅在预训练阶段对 text tokens 施加 dropout，鼓励模型依赖视觉信号而非语言先验。
- **Point-then-Count**：模型先输出目标物体的像素坐标点，再基于点数得出计数结果的一种链式推理策略。
- **Style Tag**：在 fine-tuning 时给不同学术数据集的问题加上风格前缀（如 "vqa2:"），防止评测时答案风格被特定数据集污染。

## 可复现要素
- **数据集**：PixMo 全部 7 个数据集均随论文开源（https://molmo.allenai.org/blog）。
- **代码**：训练代码完全开源。
- **权重**：MolmoE-1B、Molmo-7B-O、Molmo-7B-D、Molmo-72B 等多个规格权重全部开源；另有 100% 完全开源版本（MetaCLIP + OLMo）。
- **关键超参**：预训练 4 epochs，AdamW optimizer；connector 使用更高学习率+更短 warmup；fine-tuning 数据集采样按大小平方根比例，pointing 任务 up-weight；图像使用 12 crops 训练、11-avg 评测时用 36 crops（计数任务保持训练 crop 数）。完整超参见 Appendix。
