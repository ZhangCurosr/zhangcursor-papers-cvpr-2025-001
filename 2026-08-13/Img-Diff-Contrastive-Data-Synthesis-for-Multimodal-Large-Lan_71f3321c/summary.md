---
title: "Img-Diff-Contrastive-Data-Synthesis-for-Multimodal-Large-Lan"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Jiao_Img-Diff_Contrastive_Data_Synthesis_for_Multimodal_Large_Language_Models_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 07:12:09"
field: "多模态大模型数据合成"
keywords: ["Multimodal Large Language Models", "Contrastive Data Synthesis", "Image Difference Captioning", "Object Replacement", "Visual Instruction Tuning", "Stable Diffusion", "Prompt-to-Prompt"]
innovations: ["提出全自动 IMG-DIFF 对比数据合成流水线，通过四级过滤生成细粒度图像差异数据", "设计 Difference Area Generator（CLIP相似度+FastSAM+BLIP匹配+IoU去重）和两阶段 Difference Captions Generator", "在 MMVP 上超越 GPT-4V/Gemini 高达 12 分，8 个通用 MLLM 基准平均最高提升 3.06%"]
benchmarks: ["MMVP", "Spot-the-Diff", "Image-Edit-Request", "VQAv2", "GQA", "MMBench", "MMBench-CN", "MM-Vet", "ScienceQA", "SEED-Bench", "POPE"]
---

# 论文速读：Img-Diff: Contrastive Data Synthesis for Multimodal Large Language Models

## 一句话总结
本文提出了一种完全自动化的对比式数据合成方法，通过 Stable Diffusion XL + Prompt-to-Prompt 生成"物体替换"图像对，并结合 Difference Area Generator 和 Difference Captions Generator 产出生成高质量的细粒度图像差异数据集 IMG-Diff，用于微调 MLLM（如 InternVL2、LLaVA-1.5、MGM），在 MMVP、Spot-the-Diff 等图像差异基准上超越 GPT-4V 和 Gemini，并在 8 个通用 MLLM 基准上实现平均最高 3.06% 的综合提升。

---

## 研究问题与动机

1. **MLLM 性能高度依赖数据质量**：当前 SOTA MLLM（LLaVA、MGM、InternVL2）采用两阶段训练（预训练+指令微调），但视觉指令微调数据的多样性仍不足，尤其缺乏针对"细粒度图像差异识别"的专项数据。

2. **已有图像差异数据集覆盖范围有限**：Spot-the-Diff（监控摄像头街景）、CLEVR-Change（几何图形）、CUB-Bird（鸟类物种）等数据集场景单一，难以支撑通用 MLLM 的细粒度识别能力。

3. **现有合成方法（如 InstructPix2Pix）缺乏质量控制机制**：InstructPix2Pix 使用 Prompt-to-Prompt + GPT-3 生成图像对，但未引入多维度过滤（相似度、文本匹配、差异检测），导致数据质量参差不齐，且其差异描述偏向整图而非细粒度区域。

4. **对比学习思想尚未充分融入 MLLM 数据合成**：对比学习通过正负样本对提升模型判别能力，但如何将这一范式自动化应用于大规模 MLLM 数据合成仍缺少系统方案。

---

## 核心贡献（创新点）

1. **提出了 IMG-DIFF 数据合成流水线**：利用 Stable Diffusion XL + Prompt-to-Prompt 生成物体替换图像对，并通过三阶段自动化过滤（相似度过滤→分割检测→差异描述）构建高质量"object replacement"数据集，区别于 InstructPix2Pix 缺少过滤机制的做法。

2. **设计了 Difference Area Generator**：结合 CLIP 图像相似度过滤、FastSAM 分割、BLIP 图像-文本匹配过滤和 IoU 去重，精确提取图像对中包含物体差异的边界框区域，克服了传统目标检测模型类别覆盖有限的问题。

3. **设计了 Difference Captions Generator（两阶段）**：Stage 1 使用 LLaVA-NEXT 描述边界框内容并做文本匹配和 caption 相似度过滤；Stage 2 用红框高亮差异区域指导 LLaVA-NEXT 生成细粒度差异描述，确保单条 caption 聚焦单一差异而非整图混合描述。

4. **实证验证了对比数据对 MLLM 的多维度提升**：在 MMVP 基准上超过 GPT-4V 和 Gemini 高达 12 分；在 Spot-the-Diff 和 Image-Edit-Request 上达到新 SOTA；在 8 个通用 MLLM 基准上平均提升最高达 3.06%（LLaVA-1.5-7B），且代码与数据集已开源。

---

## 方法详解

### 整体流程（Figure 2）
三步流水线：① 图像对生成 → ② Difference Area Generator（差异区域定位）→ ③ Difference Captions Generator（差异描述生成）。全程在 Data-Juicer 框架中以 operator + recipe 形式实现，支持可复现配置。

### Step 1：图像对生成
- 从 MSCOCO 或 LLaVA 预训练 caption 库获取原始图像描述；
- 使用 LLM 执行替换 prompt："Here is a sentence: 'INPUT'. Please only replace one of the objects in this sentence with another object."
- 以原始 caption 和替换后 caption 为条件，利用 Stable Diffusion XL + Prompt-to-Prompt 生成只含单一物体替换的图像对。

### Step 2：Difference Area Generator（Figure 3）
包含三个子模块串联：

- **Image Similarity Filter**：用 CLIP 提取两图特征，计算余弦相似度，筛选出"高度相似但不完全相同"的图像对（在分割前和差异检测后各使用一次）。
- **FastSAM 分割 + 裁剪**：对每张图进行实例分割，按 bbox 裁剪子图。
- **Image-text Matching Filter**：用 BLIP 提取子图特征，与物体名称文本特征做匹配，保留包含有效替换物体的 bbox。
- **Difference Detector + IoU 过滤**：对同坐标 bbox 对应的两个子图再次用 CLIP 计算相似度，仅保留差异足够大的 bbox；最后用 IoU 去除重叠 bbox，得到最终有效 bbox 集合。

### Step 3：Difference Captions Generator（Figure 4，两阶段）
- **Stage 1 — Object Labeling & Filtering**：
  - 对每对图像，选取 N=5 个相似度最低的 bbox 作为候选区域；
  - 用 LLaVA-NEXT 描述每个 bbox 内容；
  - 用 Image-text Matching Filter 验证描述准确性；
  - 用 **Captions Similarity Filter**（CLIP 文本特征余弦相似度）判断两个 bbox 的 caption 是否足够不同——分数越低差异越大。
- **Stage 2 — Difference Captions Generating**：
  - 在原图上用红框标注差异 bbox；
  - 将红框图像 + Stage 1 caption 输入 LLaVA-NEXT，指令模型生成差异描述；
  - 最终产出"图像对 + bbox + 差异描述"三元组及对应的 QA 对。

### 数据规模（Section 3.5）
- MSCOCO captions 起点：118K 图像对 → 经 Image Similarity Filter 得 38,533 对 → Difference Area Generator 产 117,779 个有效 bbox（每对最多 5 个）→ Difference Captions Generator 产 12,688 条高质量 "object replacement" 样本。
- LLaVA pretrain captions 起点：额外生成 34,538 条样本（用于消融实验，验证质量优先于数量）。

---

## 实验与结果

### 数据集与评估基准
- **图像差异基准**：MMVP（CLIP-blind 图像对，需识别内容差异）、Spot-the-Diff（街景时间差图像对）、Image-Edit-Request（编辑指令理解）。
- **通用 MLLM 基准（8 个）**：VQAv2、GQA（综合 VQA）、MMBench、MMBench-CN、MM-Vet、ScienceQA、SEED-Bench（感知与推理）、POPE（细粒度定位/幻觉检测）。

### 微调设置
- 基线模型：LLaVA-1.5-7B、MGM-7B、InternVL2-8B（主实验）；InternVL2-1B、LLaVA-1.5-13B（补充实验）。
- 将 IMG-DIFF 数据混合入原始视觉指令微调数据后重新微调。
- Spot-the-Diff 和 Image-Edit-Request 有训练划分，微调后额外在这些数据的训练集上再训 2 epoch。

### 关键结果

**MMVP（Figure 5）**：
- MGM-7B + RP 超过 GPT-4V 和 Gemini 高达 **12 分**。
- LLaVA-1.5-7B + RP 超过 LLaVA-1.5-13B。

**Spot-the-Diff（Table 1）**：
| 模型 | BLEU | METEOR | CIDEr-D | ROUGE-L |
|---|---|---|---|---|
| LLaVA-1.5-7B → +RP | 8.5→**9.7** | 12.0→**13.0** | 38.3→**43.2** | 30.1→**30.8** |
| MGM-7B → +RP | 9.9→**10.8** | 12.0→**13.1** | 46.3→**53.5** | 31.5→**33.0** |
| InternVL2-8B-FT → +RP | 6.6→**8.4** | 11.7→**12.8** | 26.5→**32.2** | 27.3→**28.5** |

**Image-Edit-Request（Table 2）**：所有三个模型 + RP 后均刷新 SOTA，BLEU 最高达 16.6（MGM-7B+RP）。

**8 个通用 MLLM 基准（Table 3，△ 为平均提升）**：
- LLaVA-1.5-7B：**+3.06%**（最强提升，VQAv2 78.5→80.4，MMB 64.3→66.1）
- MGM-7B：+1.28%
- InternVL2-8B：+1.01%
- InternVL2 基础模型本身已含大量高质量训练数据，故提升相对较小但仍为正。

### 数据质量评估（Section 5.1，1,000 样本人工标注）
- **Bounding Box Difference**：仅 4.5% 为"low"，近 80% 有效差异。
- **Content Caption Accuracy**：80.1% 准确。
- **Difference Caption Accuracy**：超 70% 完全准确，21.8% 仅特征描述有误。

### 数据多样性（Section 5.2）
- 覆盖 **1,203** 个物体类别，**3,680** 种唯一"物体替换对"。
- Object365 类别出现约 52.06%，长尾类别占近半，兼顾常见与罕见物体。

---

## 相关工作脉络

1. **InstructPix2Pix [6]**：使用 Prompt-to-Prompt + SD-1.5 生成图像对并用 GPT-3 生成编辑文本描述。本文沿用 Prompt-to-Prompt 思路但改用 SDXL 提高真实感，并引入多维度过滤 + 细粒度区域描述，解决 InstructPix2Pix 缺过滤、描述偏整图的问题。

2. **Spot-the-Diff [26]**：最早的 IDC 数据集，监控街景图像对 + LSTM 解码器生成差异描述。本文强调"object replacement"场景的泛化性，且完全自动化，无需人工标注。

3. **VIXEN [5]**：首个用 MLLM 做 IDC 的工作，将差异图像特征映射到文本空间后用 LLM 生成描述。本文反其道而行——先合成带精确 bbox 标注的数据，再用 MLLM 生成描述，更适配 MLLM 指令微调范式。

4. **CLIP4IDC [18] / DUDA [51] / VARD [65] / NCT [66]**：各类专用 IDC 模型（ResNet+LSTM、Transformer、disentanglement）。本文不走模型改进路线，而是通过数据合成直接从数据侧赋能通用 MLLM。

5. **LLaVA-1.5 / MGM / InternVL2**：当前主流 MLLM 基线，均采用视觉指令微调范式。本文不修改模型架构，仅以新增对比数据增强微调，证明了数据侧的增量价值。

---

## 局限性与未来方向

1. **仅聚焦"物体替换"一种差异类型**：未覆盖场景级变化（光照、季节、天气）、结构变化（添加/删除背景元素）等，数据多样性受限于单一操作类型（附录中有"object removal"探索，但未深入）。

2. **合成图像可能包含人工痕迹**：Section 15 评估了 unnatural images 的影响，说明 SDXL 生成的图像在某些情况下可能不够自然，可能影响模型泛化。

3. **基础模型训练数据规模差异导致提升不均衡**：InternVL2 本身训练数据量大、质量高，因此 +RP 后仅获 +1.01% 提升，说明该方法对"训练数据较匮乏"的模型更有价值，对 Already-strong 模型边际效益递减。

4. **依赖 LLM 做 caption 替换存在随机性**：温度设置和词库大小影响多样性（Section 11 有讨论），但未系统分析替换策略对最终数据质量的因果影响。

5. **未探索跨域泛化**：全部在英文 caption 和自然场景图像上生成，对 OCR、图表、医学影像等专业领域无验证。

---

## 研究启发与可借鉴点

1. **"相似度过滤→分割→匹配→差异检测"四级过滤架构**可复用于其他类型的合成数据清洗流程，尤其是任何基于生成模型的图像对构建任务。

2. **细粒度 bbox 约束 + 红框高亮引导 MLLM 生成差异描述**的策略，比直接让模型描述整图差异更准确，可作为 MLLM 数据标注的通用技巧。

3. **用 CLIP 文本相似度作为 caption 差异的自动判定器**（Captions Similarity Filter），低成本替代人工标注，值得迁移到其他需要自动去重的数据 pipeline 中。

4. **质量>数量原则的实证**（LLaVA pretrain 的 34,538 条 vs MSCOCO 的 12,688 条高质量样本，后者反而带来更好效果），为数据筛选策略提供了直接参考。

5. **在 Data-Juicer 中以 operator + recipe 形式发布**的做法，为可复现的数据合成工程提供了样板，便于后续研究直接复用或扩展。

---

## 关键术语表

**IMG-DIFF**：本文提出的自动化对比数据合成方法与数据集名称，核心为"object replacement"图像对及其细粒度差异描述。

**Prompt-to-Prompt [20]**：通过控制 cross-attention 层实现文本到图像生成的局部编辑技术，本文用它保证生成图像对之间仅有单一物体被替换。

**Stable Diffusion XL (SDXL) [52]**：OpenAI 开源的高分辨率潜在扩散模型，本文取代 InstructPix2Pix 使用的 SD-1.5 以获得更真实的图像。

**Difference Area Generator**：本文提出的差异区域定位模块，串联 CLIP 相似度过滤、FastSAM 分割、BLIP 文本匹配和 IoU 去重四个步骤。

**Difference Captions Generator**：两阶段差异描述生成模块，Stage 1 用 LLaVA-NEXT 标注 bbox 内容并过滤，Stage 2 用红框高亮图生成最终差异 caption。

**Captions Similarity Filter**：用 CLIP 文本编码器计算两个 bbox caption 的余弦相似度，相似度低则视为有效差异。

**MMVP [63]**：多模态大模型视觉能力基准，构造 CLIP-blind 图像对（CLIP 特征相似但内容不同），要求模型识别差异。

**Object Replacement (RP)**：数据合成中的核心操作类型，即在图像中替换单一物体，构成 IMG-DIFF 数据集的基本单元。

---

## 可复现要素

- **数据集**：12,688 条 MSCOCO 源的高质量"object replacement"样本 + 34,538 条 LLaVA pretrain 源样本；**已开源**于 [GitHub](https://github.com/modelscope/data-juicer/tree/ImgDiff)。
- **代码**：全部以 Data-Juicer operator + recipe 形式开源，可在 Data-Juicer 沙箱环境中复现。
- **关键超参**：N=5（候选 bbox 数量）、FastSAM 分割、CLIP 余弦相似度阈值（论文附录 Section 17 详述，主文未列具体数值）、SDXL + Prompt-to-Prompt 生成。
- **训练设置**：LLaVA-1.5-7B / MGM-7B / InternVL2-8B 混合原始视觉指令数据 + IMG-DIFF 微调；Spot-the-Diff/Image-Edit-Request 额外 2 epoch。
- **基线模型权重**：LLaVA-1.5、MGM、InternVL2 均为开源模型，可复现。

---
