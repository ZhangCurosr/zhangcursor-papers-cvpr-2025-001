---
title: "Hybrid-Level-Instruction-Injection-for-Video-Token-Compressi"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_Hybrid-Level_Instruction_Injection_for_Video_Token_Compression_in_Multi-modal_Large_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 07:11:39"
field: "多模态大语言模型高效推理"
keywords: ["视频理解", "多模态大语言模型", "token压缩", "条件压缩", "指令注入", "HICom"]
innovations: ["混合层级指令注入实现条件token压缩，局部保持时空结构、全局提取关键信息", "新增条件预训练阶段（align→cond-pretrain→instruction tuning）解决对齐不充分导致的引导混乱问题"]
benchmarks: ["VideoMME", "MVBench", "EgoSchema", "ActivityNet", "VideoChatGPT Bench"]
---

# 论文速读：Hybrid-Level-Instruction-Injection-for-Video-Token-Compressi

## 一句话总结
论文提出了 HICom 方法，通过混合层级指令注入（局部层级 + 全局层级）实现多模态大语言模型的视频 token 条件压缩，在减少 78.8% token 数量的同时，在三个多选题评测上平均提升 2.43% 性能，并新增条件预训练阶段与自构数据集 HICom-248K。

## 研究问题与动机
1. **视频 token 数量爆炸导致计算开销过大**：MLLM 处理视频时通常对帧采样后拼接视觉 token，视频帧数远超图像，造成显著推理负担。
2. **现有压缩方法为无条件压缩**：平均池化、Q-Former、Resampler 等方法缺乏显式指令引导，无法区分哪些视觉内容对用户问题更重要，容易丢失关键信息。
3. **已有指令引导方法探索不足**：LLaMA-VID 等方法仅在粗略层面对帧进行选择或交互，未深入探索指令对 token 压缩过程的指导作用；部分方法简化为单 token 辅助信息，破坏了时空结构。
4. **用户关注的视觉内容与指令强相关**：视频各部分对用户指令的贡献并不均等，应通过条件压缩突出指令相关部分、丢弃无关内容。

## 核心贡献（创新点）
1. **混合层级指令注入的条件压缩框架（HICom）**：同时在局部层级（分组内压缩）和全局层级（可学习 token 压缩）注入指令条件，与现有无条件压缩方法的本质区别在于引入了显式用户指令作为压缩条件，实现"问什么保什么"。
2. **三种指令注入方式的设计与选择**：提出了 direct injection、coarse injection、fine injection 三种机制，并系统性验证后确定局部层级用 direct、全局层级用 coarse 的组合最优，这是本文在条件注入设计上的创新。
3. **新增条件预训练阶段（Conditional Pre-training Stage）**：在传统的 align → instruction tuning 流程中插入新的条件预训练阶段，专门用于训练指令注入模块，解决了直接指令微调时对齐不充分导致引导混乱的问题。
4. **自构指令跟随描述数据集 HICom-248K**：收集 248K 视频片段、生成 739K 条指令跟随描述，数据来源于 Panda-70M 和 Ego4D，采用 Qwen2-VL-72B-Instruct 与 CoT 生成，为条件预训练提供高质量支撑。

## 方法详解
**整体框架**：HICom 位于视觉编码器与 LLM 之间，负责将视频的原始视觉 token 压缩为少量 token 输入 LLM。

**局部层级压缩（Local-Level Compression）**：
- 将编码后的视频帧特征 $V \in \mathbb{R}^{T \times H \times W \times D}$ 按降采样比 $(\alpha_T, \alpha_H, \alpha_W)$ 分为 $N_T \cdot N_H \cdot N_W$ 组，每组含 $\alpha_T \cdot \alpha_H \cdot \alpha_W$ 个 token。
- 对每组先做池化得到 $V_p^{t,h,w}$，再将指令文本特征 $C$ 注入（direct injection：经 MLP 变换），通过组内注意力完成压缩：
$$Z_l^{t,h,w} = \text{Attn}[\text{Inj}(V_p^{t,h,w}, C), V^{t,h,w}, V^{t,h,w}]$$
- 每组压缩为 1 个 token，拼接后得到 $Z_l \in \mathbb{R}^{N_T \times N_H \times N_W \times D}$，再经 MLP 投影到 LLM 嵌入空间。
- **优势**：分组注意力保留了视频的时空结构归纳偏置。

**全局层级压缩（Global-Level Compression）**：
- 初始化 $N_L$ 个可学习 token $L$，注入指令条件（coarse injection：自适应 LayerNorm，由 MLP 回归 scale/shift 作用于 $C_p$）。
- 对未分组的 flattened 视频特征加 3D 位置编码后，通过注意力从全局视角提取指令相关信息：
$$Z_g = \text{Attn}[\text{Inj}(L, C), V + \text{Pos}(V), V]$$
- 经 MLP 投影后与 $Z_l$ 拼接输入 LLM。
- **作用**：弥补局部压缩视野受限的不足，提供全局指令相关信息的汇总表示。

**指令注入方式**：
- **Direct Injection**：MLP 直接将池化文本特征 $C_p$ 映射为 query，仅适用于局部压缩（输出为单 token）。
- **Coarse Injection**：MLP 从 $C_p$ 回归 scale/shift，通过 Adaptive LayerNorm 作用于视觉输入：$\text{Inj}(A, C) = \text{LN}(A) \cdot \text{scale}(C_p) + \text{shift}(C_p)$。
- **Fine Injection**：以细粒度文本 token $C_f$ 为条件，通过 cross-attention 注入：$\text{Inj}(A, C) = \text{Attn}(\text{LN}(A), C_f, C_f)$。

**训练流程（三阶段）**：
1. **Alignment 阶段**：使用 LLaVA-558K 图像-描述对，冻结视觉编码器和 LLM，训练压缩模块基础参数，学习率 1e-3。
2. **Conditional Pre-training 阶段（新增）**：使用 HICom-248K，冻结视觉编码器和 LLM，学习率 condition injection 子模块 1e-3、其他参数 1e-4。
3. **Instruction Tuning 阶段**：使用 2.68M 视频指令数据，解冻 LLM，学习率 1e-5。

## 实验与结果
**评测基准**：VideoMME（short/mid/long，有/无字幕）、MVBench、EgoSchema、ActivityNet、VideoChatGPT Bench。

**主要结果（多选题）**：
- HICom-7B（32 帧，680 token）在 VideoMME all（无字幕）上达 58.1，优于 LLaVA-Video-7B（6272 token，60.6），但 token 节省 89.1%；以 1328 token 超越 LLaVA-Video，平均提升 2.43%，节省 78.8% token。
- HICom-7B（128 帧，2624 token）在 VideoMME all 达 60.3，优于 LLaVA-Video 的 60.6，token 仅为其 41.8%，节省 58.2%。
- 在长视频（VideoMME long）上优势更明显，证明了条件压缩对稀疏采样的有效性。

**开放问答结果**：
- HICom-7B 在 ActivityNet 达到 SOTA（59.5/3.7）。
- 在 VideoChatGPT Bench 各子项均处于第二梯队，显著优于多倍 token 的基线。

**消融关键结论**：
- local+global 条件压缩较 avg pool（2592 token）减少 73.8% token 并提升 0.47% 平均性能。
- 条件预训练阶段使条件压缩平均提升 1.17%。
- Direct + Coarse 注入组合在各基准上表现最佳。

## 相关工作脉络
1. **LLaVA-Video / LLaVA-OneVision**：主流视频 MLLM，采用固定数量帧采样 + 空间池化压缩，属于无条件压缩范式，本文在其基础上引入指令条件。
2. **VideoLLaMA2**：使用时空卷积下采样 token，同样是 unconditional 压缩；本文方法在更少 token 下达到更高性能。
3. **Q-Former / Resampler**：将视觉 token 压缩为固定数量，用于图像-语言对齐，但未考虑视频时序结构；本文分组注意力天然保持时空结构。
4. **LLaMA-VID**：唯一采用指令引导思路的近期工作，但仅将交互结果简化为每帧一个 token 的辅助信息，不压缩 token；本文探索了完整的条件压缩机制。
5. **Chat-Univi**：使用 DPC-KNN 聚类动态视觉 token，无需额外训练；本文通过可学习压缩模块实现可微的条件压缩。
6. **VAQuA / Text-based video representation**：基于指令相似度采样关键帧，但仅解决帧选择问题，未深入 token 级压缩；本文方法在 token 层面实现了更细粒度的条件压缩。

## 局限性与未来方向
1. **训练阶段帧采样策略固定**：使用均匀采样（32 帧）训练，限制了模型对超长视频的理解能力；文中提到未来将尝试 fps 采样策略。
2. **极长视频（30-60 分钟）性能提升有限**：32 帧对极长视频过于稀疏，128 帧改善也不明显，需要更多长视频训练数据。
3. **仅探索了视频场景**：结论中明确提到未来将探索条件压缩在高精度图像上的潜力。
4. **局部压缩视野受限**：分组注意力虽保持结构但视野局部化，全局压缩会遗忘部分细节（如可视化所示）。

## 研究启发与可借鉴点
1. **条件预训练阶段的插入策略**：在 align 和 instruction tuning 之间加入专门训练条件注入模块的阶段，解决了"模块未充分对齐即进行指令微调"的问题，这一三段式训练范式可迁移至其他条件生成/压缩任务。
2. **分组注意力保持时空结构的思想**：将视频特征按时空维度分组并在组内压缩，既实现了降维又保留了结构归纳偏置，可用于其他需要保留空间结构的视觉压缩任务。
3. **HICom-248K 数据集的构建范式**：基于 CLIP 相似度选 keyframe、使用大模型 CoT 生成指令跟随描述、过滤无效回答的 pipeline，为视频-语言预训练数据构建提供了可复用的模板。
4. **可迁移到高分辨率图像场景**：作者提出探索条件压缩在图像上的应用，结合分组思想可将高分辨率图像特征图分块压缩，降低长序列 LLM 的计算开销。
5. **三种注入方式的系统对比**：direct/coarse/fine 三种机制的分类与实验设计，为后续研究者理解条件注入机制提供了清晰的分析框架。

## 关键术语表
**HICom**：Hybrid-Level Instruction Injection for Conditional Token Compression，本文提出的混合层级指令注入条件 token 压缩方法。
**Conditional Compression（条件压缩）**：以用户指令为条件，选择性保留与指令相关的视觉信息、丢弃无关内容的 token 压缩策略。
**Direct Injection**：通过 MLP 将池化文本特征直接映射为 attention query 的指令注入方式，适用于单 token 输出场景。
**Coarse Injection**：通过自适应 LayerNorm（MLP 回归 scale/shift）将全局文本特征注入视觉输入的指令注入方式。
**Fine Injection**：以细粒度文本 token 为 key/value、通过 cross-attention 注入指令条件的方式，能捕获更细粒度的文本-视觉交互。
**HICom-248K**：本文自构的条件预训练数据集，包含 248K 视频片段和 739K 条指令跟随描述，来源于 Panda-70M 和 Ego4D。
**Conditional Pre-training Stage**：插入在 align 和 instruction tuning 之间的新增训练阶段，专门用于训练指令注入压缩模块。

## 可复现要素
- **代码**：开源，地址 https://github.com/lntzm/HICom
- **数据集**：HICom-248K 未声明开源地址，论文未明确说明是否公开
- **视觉编码器**：SigLIP so400mpatch14-384，冻结
- **LLM**：Qwen2.5 系列（1.5B / 7B）
- **训练配置**：alignment 阶段 batch size=512、lr=1e-3；conditional pre-training 阶段 batch size=256、condition injection lr=1e-3、其他参数 lr=1e-4；instruction tuning 阶段 lr=1e-5；各阶段均训练 1 epoch
- **降采样比**：默认 (α_T, α_H, α_W) = (4, 3, 3)
- **帧采样**：训练 32 帧，推理可扩展至 64/128 帧
- **硬件**：消融实验使用 8 × A800 GPU，总训练时间约 28 小时（896K 指令微调数据 + Qwen2.5-0.5B）
