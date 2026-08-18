---
title: "Multi-Layer-Visual-Feature-Fusion-in-Multimodal-LLMs-Methods"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Lin_Multi-Layer_Visual_Feature_Fusion_in_Multimodal_LLMs_Methods_Analysis_and_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 07:39:55"
field: "多模态大模型架构设计"
keywords: ["多模态大语言模型", "视觉特征融合", "层选择", "内部融合", "外部融合", "直接融合", "MMFuser", "Mini-LLaVA"]
innovations: ["提出相似度与比例双标准视觉层选择框架", "建立 2x2 融合策略正交分类并系统消融", "发现外部直接融合是性能最强且最稳定的最优实践"]
benchmarks: ["GQA", "MMBench", "MME", "TextVQA", "OCRBench", "CV-Bench", "POPE"]
---

# 论文速读：Multi-Layer Visual Feature Fusion in Multimodal LLMs: Methods, Analysis, and Best Practices

## 一句话总结
本文系统研究了多模态大语言模型（MLLMs）中多层视觉特征的选择与融合策略，发现外部直接融合（将不同阶段的视觉特征在输入阶段简单合并）是性能最强且最稳定的方案，且从开头、中间、结尾三个阶段各选一个代表层能获得最佳泛化能力。

## 研究问题与动机
- **视觉层选择缺乏系统性**：现有工作（如 MiniCPM、LLaVA 使用单层特征；Dense Connector、EVLM 使用多层特征）多依赖人工设计或任意选择，缺少对"哪些层最优"的系统分析。
- **融合策略多样性但无对比研究**：内部融合（在 LLM 中间层注入特征）vs 外部融合（在输入阶段融合）、模块化融合（引入 cross-attention 等额外模块）vs 直接融合（简单拼接/相加）——已有工作各自为战，缺少统一实验对比。
- **参数效率与训练稳定性问题**：内部融合随层数增加需引入更多投影器与模块，参数开销大，且深层插入模块易破坏已有特征处理流。
- **核心研究问题**：① 如何更有效地选择视觉层？② 如何设计更有效的融合策略？

## 核心贡献（创新点）
1. **提出双标准视觉层选择框架**：基于表征相似度（将 24 层分为开头/中间/结尾三阶段，各取代表层 {3, 18, 23}）和基于编码器深度的比例划分（前 12 层 Former、后 12 层 Latter），系统评估不同层组合。
2. **建立 2×2 融合策略正交分类**：按融合位置（内部 Internal vs 外部 External）和融合模式（模块化 Modular vs 直接 Direct）划分为四种策略，填补现有研究缺乏统一对照的空白。
3. **发现"跨阶段选择 > 同阶段堆叠"原则**：从不同表征阶段（开头 + 中间 + 结尾）各选一层显著优于在同一阶段内选取多层（如仅用 Latter 或 All 均表现下降）。
4. **确立"外部直接融合"为最优实践**：在所有配置下，外部直接融合（E+D）平均得分最高（49.88）、方差最小；内部融合需大规模数据才能逼近，模块化融合对层选择更敏感。

## 方法详解
**视觉编码器与基座模型**：采用 CLIP-ViT-L/14（24 层）作为视觉编码器，MobileLLaMA 1.4B（24 层）作为 LLM，构建轻量基座 Mini-LLaVA（替换 LLaVA-1.5 中的 Vicuna 7B）。

**层选择方案**：
- 相似度划分：第 1–8 层为开头阶段（Beginning），9–16 层为中间阶段（Middle），17–24 层为结尾阶段（Ending），代表性层为第 3、18、23 层。
- 比例划分：前 12 层（Former）、后 12 层（Latter）、全部 24 层（All）。

**内部融合（Internal Fusion）**：在 LLM 第 i 层将投影后的视觉特征 $v_i$ 与隐藏状态 $h_i$ 结合：
$$h'_i = \mathcal{I}(h_i, \mathbf{P}_i(v_i)) + h_i$$
- 模块化：引入 pre-cross / post-cross / parallel cross-attention。
- 直接：将视觉 token 直接加到对应层的 hidden state。

**外部融合（External Fusion）**：在输入阶段将视觉特征集 $\mathbf{F}$ 与视觉 token $\mathbf{V}$ 融合：
$$\mathbf{V}' = \mathcal{E}(\mathbf{V}, \mathbf{P}(\mathbf{F}))$$
- 模块化：采用 MMFuser 架构。
- 直接：维度拼接（stacking）或逐元素相加后平均（summation + averaging）。

**训练设置**：预训练 558K 图像描述 + 指令微调 665K 对话；预训练阶段仅训新增模块（投影器/新模块），微调阶段解冻除视觉编码器外所有参数。

## 实验与结果
**评测基准**（四大类共 10 项）：
- General：GQA、MMBench（MMB）、MME（认知 MME$^C$、感知 MME$^P$）
- OCR：TextVQA、OCRBench
- CV-Centric：CV-Bench 2D / 3D
- Hallucination：POPE

**关键结果**（Mini-LLaVA 基线 Avg = 48.51）：

| 配置 | GQA | MMB | MME$^C$ | MME$^P$ | TextVQA | OCRBench | CVBench$^{2D}$ | CVBench$^{3D}$ | POPE | **Avg.** |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | 56.95 | 46.91 | 262 | 1200 | 35.47 | 239 | 39.74 | 55.00 | 85.83 | **48.51** |
| Internal + Pre-Cross (Double) | 58.41 | 50.93 | 218 | 1182 | 34.42 | 261 | 46.34 | 51.08 | 85.06 | **48.74** |
| Internal + Direct (Latter) | 58.79 | 47.02 | 221 | 1179 | 37.28 | 241 | 42.64 | 51.92 | 85.57 | **48.21** |
| External + Modular (Former) | 58.37 | **54.30** | 258 | 1181 | 35.85 | 244 | 42.59 | 54.92 | 86.29 | **49.78** |
| **External + Direct (All)** | **59.54** | 51.46 | 236 | **1200** | 38.01 | **265** | 44.78 | **53.08** | **86.40** | **49.88** |

- 最强结果：**External + Direct (All)** 平均 49.88，较基线 +1.37 分；GQA +2.59、TextVQA +2.54、OCRBench +26。
- 内部融合中，Latter 直接融合在 GQA/TextVQA 上表现突出（+8.87 / +18.32），说明 LLM 深层对视觉 token 注意力较弱，注入干扰小。
- 外部融合整体优于内部融合；直接融合比模块化融合更稳定、对层数不敏感。
- 数据规模实验（332K/665K/737K）：E+D 和 E+M 在 665K 以上保持高表现；I 随数据增大提升明显，暗示内部融合需要更大 SFT 数据。
- 组件替换实验（SigLIP / MobileLLaMA 2.7B）：E+D 从 49.18 → 51.76 → 52.63 持续增益；I+M 配合 SigLIP 仅 43.27，严重过拟合/参数效率差。

## 相关工作脉络
- **LLaVA-1.5 / MiniCPM / InternVL**：采用单层（最后一层）视觉特征 + 外部直接注入，本文将其设为基线并证明改进空间。
- **Dense Connector [45]**：按比例选取后半段层特征，本文扩展为 Former/Latter/All 三套对照，揭示"同阶段堆叠"不一定更优。
- **EVLM [9] / Lion [8]**：利用多层 ViT 特征增强感知，但层选择经验化；本文首次系统验证"跨三阶段各取一点"优于"全量或后半段"。
- **DeepStack [35] / mplug-owl3 [46]**：在 LLM 中间层插入视觉 token（内部融合）；本文对比发现内部融合参数开销大、易受层位置影响。
- **MMFuser [6]**：外部模块化融合代表；本文表明其虽在特定任务（如 MMB）上有优势，但整体不如外部直接融合稳定。
- **SigLIP [50] / Cambrian-1 [44]**：先进视觉编码器与开源 VLM 平台；本文用其做组件替换验证，证明 E+D 的可扩展性。

## 局限性与未来方向
- **基座模型规模较小**（MobileLLaMA 1.4B），结论在 7B/70B 级模型上需进一步验证。
- **仅覆盖 CLIP/SigLIP 类 ViT 编码器**，对于卷积型（ConvNeXt）或 DINO 型编码器的泛化未探索。
- **融合策略以静态注入为主**，未考虑动态自适应选择（如按 query 内容决定使用哪些层）。
- **训练数据规模上限 737K**，更大语料（千万级）下内部融合的潜力未充分挖掘。
- 未来方向：① 探索自适应层选择机制；② 扩展到更大型 LLM（7B+）和更长图像序列；③ 研究多模态检索/定位任务中的层特异性。

## 研究启发与可借鉴点
1. **2×2 正交分类框架可直接迁移**：将任何新融合模块按"位置×模式"二分，可快速建立消融体系，避免重复摸索。
2. **"跨阶段代表性采样"设计值得复用**：用余弦相似度聚类确定层阶段划分，再每阶段取 1 个 representative，既保留多样性又控制 token 开销。
3. **外部直接融合作为强 baseline**：在尝试复杂模块化设计前，先跑通 E+D 并与之对比，可更公平地评估新增模块的真实增益。
4. **数据规模对融合策略选择性显著**：小数据（<665K）优先外部融合，大数据（>1M）可考虑内部融合，这一规律可指导资源受限场景的架构选择。
5. **直接融合在后期层更稳定**：LLM 注意力分布随深度递减的经验（An Image is Worth 1/2 Tokens [10]）可解释为何 Latter 直接融合在 GQA 上表现突出，为"在哪层注入"提供理论依据。

## 关键术语表
- **Internal Fusion（内部融合）**：将视觉特征在 LLM 中间层（而非输入端）注入，可与每一 LLM 层对齐。
- **External Fusion（外部融合）**：在视觉 token 进入 LLM 之前于输入阶段完成多源特征合并。
- **Modular Fusion（模块化融合）**：引入 cross-attention、MLP 等额外模块处理视觉特征后再融合。
- **Direct Fusion（直接融合）**：通过拼接（stacking）或逐元素相加（addition + averaging）简单合并特征，无额外模块。
- **Pre-Cross / Post-Cross / Parallel Attention**：三种 cross-attention 插入位置变体，分别位于 LLM 层前、层后、并行分支。
- **Similarity-based Selection（相似度选择）**：基于层间特征余弦相似度将编码器层划分为三个表征阶段。
- **Proportion-based Selection（比例选择）**：按编码器深度等比切分前/后 halves。
- **Mini-LLaVA**：本文构建的轻量基座模型，基于 LLaVA-1.5 架构替换为 MobileLLaMA 1.4B。

## 可复现要素
- **代码**：公开于 https://github.com/EIT-NLP/Layer_Select_Fuse_for_MLLM
- **视觉编码器**：CLIP-ViT-L/14（公开权重）；替代实验用 SigLIP。
- **LLM**：MobileLLaMA 1.4B / 2.7B（公开）。
- **预训练数据**：558K image caption（Conceptual 12M / LAION-5B 子集）。
- **SFT 数据**：665K（LLaVA-1.5 原版）/ 332K（减半）/ 737K（Cambrian-1）。
- **超参**：论文未详细列出学习率/步数等，仅在补充材料提供新增模块参数量；建议参照 LLaVA-1.5 官方训练脚本。
- **评测脚本**：未明确开源，需依据各 benchmark 官方实现复现。

---
