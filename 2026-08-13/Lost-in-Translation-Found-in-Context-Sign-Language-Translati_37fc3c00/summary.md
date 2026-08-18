---
title: "Lost-in-Translation-Found-in-Context-Sign-Language-Translati"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Jang_Lost_in_Translation_Found_in_Context_Sign_Language_Translation_with_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 09:39:09"
field: "手语理解与翻译"
keywords: ["手语翻译", "大语言模型微调", "上下文感知", "BOBSL", "多模态融合", "LoRA"]
innovations: ["提出结合背景描述、前句翻译和伪gloss的多线索LLM手语翻译框架", "设计完全自动化的上下文利用方案，无需oracle信息", "引入LLM-based评估指标并系统消融各线索贡献"]
benchmarks: ["BOBSL SENT-TEST", "How2Sign"]
---

# 论文速读：Lost-in-Translation-Found-in-Context-Sign-Language-Translati

## 一句话总结
本文提出了一种结合上下文线索（背景描述、前句翻译、伪 gloss）的大语言模型微调方法，将连续手语视频翻译为口语文本，在 BOBSL 数据集上显著超越了现有 SOTA 方法。

## 研究问题与动机
1. 手语缺乏标准化书写形式，且依赖语境和空间指代，仅凭视觉特征难以准确翻译（如三分之一的句子需要额外话语语境才能完整翻译）。
2. 现有 SLT 方法多忽略上下文信息，或仅使用 GT 前句/spotting 等 oracle 信息，缺乏完全自动化的多线索融合方案。
3. 大规模低资源手语数据集（如 BOBSL）的训练数据标注质量弱——字幕与手语存在时序错位且语义不对齐。
4. 手语中存在同音手势、话题-评论结构等语法特性，导致单句级翻译往往定义不清，需要上下文消歧。

## 核心贡献（创新点）
1. 提出首个结合背景描述、前句预测、伪 gloss 与视觉特征的多线索 LLM 手语翻译框架，实现完全自动化的非 oracle 上下文利用。
2. 引入 LLM-based 评估指标（GPT-4o-mini 打分），提供更细粒度的翻译质量评估，弥补 BLEU 等传统指标的不足。
3. 在 BOBSL 上系统消融各上下文线索的贡献，证明每类线索均对翻译质量有正向且互补的提升作用。
4. 将方法泛化至美式手语数据集 How2Sign，验证了模型跨语言/跨场景的通用性。

## 方法详解
1. **整体架构**：以预训练 Llama3-8B 为解码器，通过 LoRA 微调，将视觉特征、伪 gloss、背景描述、前句翻译拼接为 prompt 序列输入 LLM。
2. **视觉特征提取**：使用在 BOBSL 上训练的 Video-Swin ISLR 模型（8697 类），以步长 s=2 滑动窗口提取 16 帧 clip 的 768 维嵌入（时空平均），共约 56 个特征向量。
3. **伪 gloss 生成**：将同一 ISLR 模型以滑动窗口方式输出分类预测，得到 G 个噪声标签序列（约 22 个/句），供 LLM 消歧利用。
4. **背景描述提取**：裁剪掉手语者后，用 BLIP2 图像描述模型在 1 fps 采样帧上生成字幕，取去停用词后的唯一关键词列表（约 14 词/句）。
5. **前句上下文**：推理时自回归使用模型自身预测的前句；训练时 50% 概率使用 GT 前句、50% 概率使用预计算的前句预测。
6. **映射网络**：2 层 MLP（GELU 激活），将 768 维视觉特征投影至 LLM embedding 维度（4096）。
7. **训练策略**：LoRA（rank 4, alpha 16, dropout 0.05），仅微调 query/value 投影层，冻结文本嵌入层；学习率 1e-4，bfloat16，FlashAttention-2；同时对文本输入做词级 dropout（0-50%）和整 cue dropout（50%）。

## 实验与结果
1. **数据集**：BOBSL（英国手语，689k 训练对，SENT-TEST 含 20,870 句）；How2Sign（美式手语，31k 训练句）。
2. **评估指标**：BLEU-4、BLEURT、ROUGE-L、CIDEr、IoU、LLM 打分（0-5）。
3. **BOBSL 主结果**：本文方法（Vid+PG+Prev+BG）在 SENT-TEST 上取得 B-RT 40.3、IoU 14.8、LLM 1.20，全面超越 Sign2GPT w/PGP（B-RT 35.2, IoU 8.7）和 GFSLT（B-RT 27.7, IoU 5.2）。
4. **消融结论**：仅视觉基线已超 SOTA（B-RT 37.8 vs 35.2）；依次加入 PG、Prev、BG 各提升约 +0.7~+1.0 B-RT；全部线索合并较基线提升 +2.5 B-RT、+2.2 IoU、+0.27 LLM。
5. **How2Sign 结果**：Vid+PG+Prev 达到 B-RT 45.3、R-L 32.5，优于 VAP（R-L 27.8）近 5 个点，与 SOTA 具有竞争力。
6. **Oracle 对比**：即使在使用 GT 前句+spotting 的最强 oracle 设置下，本文方法仍大幅领先 Sincan（45.9 vs 37.0 B-RT）。

## 相关工作脉络
1. **Albanie [2] / Sincan [64]**：基于 I3D + Transformer 编码器的传统 SLT 基线，本文方法在相同设定下大幅超越，且无需 oracle 信息。
2. **GFSLT [84]**：采用 CLIP-style 对比预训练 + 掩码自监督的 gloss-free SLT，本文方法不依赖此类预训练，而是直接微调 LLM 并融入上下文。
3. **Sign2GPT [75]**：利用 DINOv2 + XGLM + LoRA 的 SOTA 方法，本文在相同 BOBSL 基准上复现后仍领先约 5 个 B-RT 点。
4. **SignLLM [27] / VAP [35] / SSVP-SLT [60]**：同样微调 T5 或 LLM 的 SLT 方法，本文强调上下文线索的显式引入，而非仅靠预训练视觉表征对齐。
5. **SSLT [82] / Fla-LLM [17]**：利用大规模 YouTube-ASL 预训练的 SLT 方案，本文在较小数据集上通过上下文增强实现可比甚至更优效果。
6. **Sincan et al. [64]**：最接近的上下文 SLT 工作，但其依赖 GT 前句字幕和 spotting，本文方法完全自动化且不依赖任何 oracle 信息。

## 局限性与未来方向
1. 背景描述可能引入噪声或与当前手语内容冲突，导致翻译质量下降（如图 3 右下角所示）。
2. 依赖前句预测存在误差累积风险，早期错误会传播至后续句子。
3. 伪 gloss 因同音手势产生大量误报，模型尚未完全学会抑制。
4. 模型难以处理指向手势（pointing）和手指拼写（fingerspelling），以及否定句、疑问句等语法结构。
5. BOBSL 字幕监督本身存在时序错位和语义不对齐问题，限制了翻译上限。

## 研究启发与可借鉴点
1. **多模态上下文融合范式**：将背景描述、前文等文本线索与视觉特征统一输入 LLM 的 prompt 机制，可迁移至其他跨模态翻译任务（如视频描述生成、多语言机器翻译）。
2. **LoRA 微调大模型做跨模态生成**：冻结文本嵌入层、仅微调 attention 投影 + 轻量 MLP 映射视觉特征，兼顾效率与性能，适合资源受限场景。
3. **LLM-based 评估指标**：用 GPT-4o-mini 对翻译结果进行 0-5 分打分并给出 reasoning，比 BLEU 更能反映语义正确性，可作为辅助评估手段。
4. **训练时 cue dropout 增强鲁棒性**：随机丢弃部分输入线索，使模型在推理时某个模态缺失仍能工作，对实际部署有价值。
5. **跨手语数据集泛化验证**：在 BSL 上训练、在 ASL 的 How2Sign 上测试，证明方法不局限于单一手语，可扩展至多语言场景。

## 关键术语表
**Sign Language Translation (SLT)**：将手语视频自动翻译为口语文本的任务，旨在消除听障人士与健听人士之间的沟通障碍。
**Pseudo-gloss**：由 ISLR 模型滑动窗口预测得到的手语词级标签序列，作为噪声文本输入辅助 LLM 消歧。
**BOBSL**：BBC-Oxford 英国手语数据集，包含 1500 小时带英文字幕的电视手语口译视频，是目前最大的 BSL 数据集。
**Video-Swin Transformer**：基于 Swin Transformer 的视频时空特征提取器，在 ISLR 任务上表现优异，本文用作视觉编码器。
**LoRA (Low-Rank Adaptation)**：低秩自适应微调技术，通过冻结预训练 LLM 权重仅在 attention 层注入低秩矩阵，实现高效参数微调。
**BLEURT**：基于回归模型训练的语义评估指标，能更好捕捉预测与参考句之间的非平凡语义相似性。
**Pointing gesture**：手语中指针对空间中特定位置的手势，用于指代上下文中已引入的人或物，是 SLT 的难点之一。
**Topic-comment structure**：手语中常见的话题-评论语法结构，话题一旦设立后不会每句重复，导致单句翻译需依赖上下文推断时态与指代。

## 可复现要素
- **数据集**：BOBSL（公开）、How2Sign（公开）；训练前需使用 [7] 的方法进行手语-字幕对齐。
- **代码/权重**：论文未提供开源代码，但使用了 Llama3（Apache 2.0）、Video-Swin、BLIP2 等开源模型。
- **关键超参**：LoRA rank=4, alpha=16, dropout=0.05；学习率 1e-4；batch size=2/GPU；10 epoch（BOBSL）/15 epoch（How2Sign）；warmup 5 epoch；gradient clipping=1.0。
- **硬件**：4×H100 GPU，bfloat16 精度，FlashAttention-2。
