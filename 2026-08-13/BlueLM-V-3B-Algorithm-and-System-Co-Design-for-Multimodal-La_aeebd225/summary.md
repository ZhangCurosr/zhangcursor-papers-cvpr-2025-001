---
title: "BlueLM-V-3B-Algorithm-and-System-Co-Design-for-Multimodal-La"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Lu_BlueLM-V-3B_Algorithm_and_System_Co-Design_for_Multimodal_Large_Language_Models_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:56:29"
field: "多模态大语言模型高效部署"
keywords: ["多模态大语言模型", "移动端部署", "算法-系统协同设计", "动态分辨率", "NPU加速", "模型量化"]
innovations: ["松驰宽高比匹配方法缓解动态分辨率过度放大问题", "NPU硬件感知的批处理图像编码与流水线并行优化", "Token下采样与分块计算策略适配移动端NPU推理"]
benchmarks: ["OpenCompass", "MMBench", "MMStar", "MMMU", "MathVista", "OCRBench", "TextVQA", "DocVQA", "MTVQA", "VQAv2", "ChartQA"]
---

# 论文速读：BlueLM-V-3B: Algorithm and System Co-Design for Multimodal Large Language Models on Mobile Devices

## 一句话总结
本文提出 BlueLM-V-3B，一款专为手机端高效部署设计的 3B 参数多模态大语言模型，通过**算法与系统协同设计**（松驰宽高比匹配 + NPU 硬件感知优化）在 MediaTek Dimensity 9300 上实现 24.4 token/s 的推理速度，同时在 OpenCompass 上以 66.1 平均分超越多个更大参数模型（如 MiniCPM-V-2.6、InternVL2-8B）。

## 研究问题与动机
1. **移动端部署内存受限**：4-bit 量化 LLaMA-7B 约需 4.5 GB 内存，严重影响手机系统流畅性，而小参数模型在性能上难以匹敌大模型。
2. **移动端计算能力有限**：在 Dimensity 9300 上 4-bit 量化 LLaMA-7B 仅生成 ~10-15 token/s，无法满足实时交互需求。
3. **动态分辨率导致图像过度放大**：主流 MLLM（InternVL 1.5、LLaVA-NeXT）的动态分辨率方案会产生大量图像 patch（如 25× 面积放大），生成超长图像 token 序列，显著增加 NPU 推理延迟。
4. **NPU 长 token 处理能力弱**：传统 GPU 并行处理所有输入 token 的策略在 NPU 上效率低下，单 token 顺序处理同样次优，需寻找平衡点。

## 核心贡献（创新点）
1. **松驰宽高比匹配方法（Relaxed Aspect Ratio Matching）**：针对传统动态分辨率方案的过度放大问题，引入阈值机制避免总是选择更大分辨率，在不牺牲准确率的前提下显著减少图像 token 数量；与 LLaVA-NeXT/InternVL 1.5 的本质区别在于其主动抑制不必要的上采样，而非一味追求高分辨率。
2. **批处理图像 Patch 编码（Batched Image Patch Encoding）**：设计 NPU 并行编码策略，每次固定处理 4 个 patch，充分利用 NPU 算力加速 ViT 推理；与已有工作的区别在于针对 NPU 低级别计算资源（显存布局、寄存器粒度）做了硬件感知的细粒度调度。
3. **图像编码流水线并行（Pipeline Parallelism in Image Patch Encoding）**：在 CPU 上的 SigLIP Conv2D 层与 NPU 上的 Vision Transformer block 之间实现流水线并行，隐藏 Conv2D 执行延迟（约 200ms）；此设计区别于以往仅在单一硬件上串行处理的方案。
4. **Token 下采样 + Chunked Computing**：采用 VILA 的 downsampler（2×2 拼接 + 线性融合）将 729 个 ViT token 降至 196 个，并实现每次并行处理 128 个输入 token 的分块计算策略（t128）；与纯 GPU 全并行或纯序列处理的方案相比，该策略在 NPU 算力与延迟间取得最佳平衡。
5. **混合精度部署 + 图像处理与指令处理解耦**：ViT/Projector 使用 INT8、LLM 使用 INT4 量化，激活保持 INT16/FP16；同时将图像编码与用户指令处理分离，图像上传后 ViT 立即开始处理而用户可同步输入指令，首 token 延迟显著降低，峰值内存控制在 2.2 GB。

## 方法详解
**模型架构**：继承 LLaVA 经典设计，由 SigLIP ViT（400M 参数，384×384 输入）+ 2 层 MLP Projector + 自研 2.7B BlueLM 语言模型组成。

**松驰宽高比匹配**：
- 定义有效分辨率 $R_e$ 和浪费分辨率 $R_w = 384m \times 384n - R_e$
- 当 $R_e - R_{e,\max} > \alpha \cdot R_{e,\max}$ 或 $(R_{e,\max} - R_e) < \alpha \cdot R_{e,\max}$ 且 $R_w < R_{w,\min}$ 时更新最优比例
- 候选宽高比按降序枚举（如 3:3 → 1:1），优先选择较小 $R_w$，避免过度放大

**批处理与流水线并行**：
- 训练时：GPU 上对图像 patch 进行 batch 编码，获得 10% 加速
- 推理时（NPU）：每次固定处理 4 个 patch（2:4 宽高比情形共 9 个 patch，采用 4+4+1 策略）
- 流水线：SigLIP 的 Conv2D 层（CPU）与 ViT block（NPU）并行执行，隐藏 ~200ms 延迟

**Token 下采样**：
- 采用 VILA 的 downsampler：每 2×2 个 token 拼接后经线性层融合
- 将 729×9=6561 个 ViT token 降至 196×9=1764 个（2:4 宽高比场景）

**Chunked Computing**：
- 在 NPU 上每次并行处理 128 个输入 token（t128），平衡并行度与 NPU 算力
- 实验表明 t128/t1 组合达到最低预填充延迟与最快生成速度

**混合精度**：
- 权重：ViT/Projector INT8，LLM INT4
- 激活：LLM INT16，ViT/Projector FP16
- KV Cache：INT8
- 峰值内存：2.2 GB

**解耦框架**：ViT 与 LLM 同时加载，图像上传后 ViT 立即处理，用户同步输入指令，图像编码完成后提交 LLM 生成，ViT 随即释放内存。

## 实验与结果
**训练数据**：
- 预训练：2.5M 图像-文本对（LLaVA 558k + ShareGPT4V 1200k + ALLaVA 708k）
- 微调：645M 图像-文本对（开源 55.8M + 自有 589.3M，涵盖纯文本、Caption、VQA、OCR）

**关键结果**：

| 模型 | 参数量 | OpenCompass 平均 |
|------|--------|-----------------|
| **BlueLM-V-3B (Ours)** | **3B** | **66.1** |
| Qwen2-VL | 8B | 67.0 |
| MiniCPM-V-2.6 | 8B | 65.2 |
| InternVL2 | 8B | 64.1 |

- 在 8 个评测任务中 4 个取得 SOTA（MMBench 82.7、MMStar 62.3、AI2D 85.3、MMVet 61.8）
- **OCRBench**: 829；**TextVQA**: 78.4；**DocVQA**: 87.8；**MTVQA**: 32.7（多语言能力显著领先）

**部署效率（MediaTek Dimensity 9300，vivo X100）**：
- 图像编码（768×1536）：~2.1 秒
- LLM Prefilling：2.7 秒
- Token 吞吐：**24.4 token/s**（vs MiniCPM-V-2.5 的 4.9 token/s，提升 ~5×）
- 峰值内存：2.2 GB

**宽高比匹配有效性**：在 LLaVA 665k 数据上，相比 LLaVA-NeXT 在 29k 样本中选择更小的宽高比，相比 InternVL 1.5 在 523k 样本中选择更小的宽高比；同时各项基准准确率均有提升（如 MiniCPM-2B 上 VQAv2 从 70.5→71.8，OCRBench 从 327→343）。

## 相关工作脉络
1. **LLaVA-NeXT [31]**：主流动态分辨率方案，但倾向于选择更高分辨率（导致图像放大），本文在此基础上引入阈值机制缓解过度放大问题。
2. **InternVL 1.5 [12]**：采用宽高比直接匹配原图比例的策略，极端情况下产生 25× 面积放大（如 380×76 图像放大至 1920×384），本文方法有效避免此类情况。
3. **MiniCPM-V [58]**：此前在移动端部署的代表性工作，但部署在 CPU（llama.cpp）上，吞吐仅 4.9 token/s；本文首次在 NPU 上实现硬件感知部署，吞吐提升约 5 倍。
4. **VILA [29]**：本文借鉴其 token downsampler 设计（2×2 拼接+线性融合），用于减少 ViT 输出的图像 token 数量。
5. **MobileVLM [14, 15]**：移动端 VLM 早期工作，本文在其基础上进行了更细粒度的算法-系统协同设计，性能更优。
6. **Qwen2-VL [54]**：8B 参数 SOTA 模型，本文 3B 模型在多数 benchmark 上逼近甚至超过其表现，验证了小模型的高效设计潜力。

## 局限性与未来方向
1. **模型规模限制**：仅 3B 参数，在复杂多轮对话、深度推理等任务上可能不如 7B+ 模型，可扩展至更大参数规模尚待验证。
2. **仅测试 Dimensity 9300**：部署实验仅在单一芯片平台上验证，对其他移动端 SoC 的泛化性未知。
3. **动态分辨率候选比例固定**：目前仅枚举 9 种宽高比（1:1 至 3:3），对于更极端的图像比例可能不够灵活。
4. **微调数据依赖自有数据**：645M 数据中 91% 为自有数据（589.3M/645M），开源可复现性受限。
5. **未来方向**：论文提到将探索更广泛移动设备的支持以及增强性能与可用性的先进算法。

## 研究启发与可借鉴点
1. **算法-系统协同设计的思路**：不应孤立地优化模型架构或系统部署，而应在训练阶段就考虑推理硬件特性（如 NPU 的长 token 处理瓶颈），反哺算法设计（如松驰宽高比匹配）。
2. **动态分辨率的重新审视**：主流方案追求高分辨率，但本文证明"不过度放大"反而更优——这一洞察可迁移至其他视觉-语言模型的训练设计中。
3. **NPU 硬件感知的细粒度优化**：批处理大小（4 patches）、流水线并行（CPU Conv2D + NPU ViT）、chunk 大小（128 tokens）等超参均需针对具体硬件反复调优，这种"硬件驱动调参"方法论值得借鉴。
4. **解耦图像编码与指令处理**：该设计可通用化地应用于其他端侧多模态系统，以降低首 token 延迟并减少峰值内存。
5. **混合精度策略的精细化设计**：权重 INT4/INT8、激活 INT16/FP16 的分类量化策略，兼顾了内存效率与模型精度，可作为移动端部署的参考范式。

## 关键术语表
**MLLM（Multimodal Large Language Model）**：融合视觉、语言等多种模态信息的大语言模型，本文聚焦于图像-文本场景。
**Dynamic Resolution（动态分辨率）**：让模型根据输入图像尺寸自适应选择处理分辨率的策略，主流方案包括 LLaVA-NeXT 和 InternVL 1.5 的方法。
**Relaxed Aspect Ratio Matching（松驰宽高比匹配）**：本文提出的改进方案，通过阈值机制避免动态分辨率中的过度图像放大。
**NPU（Neural Processing Unit）**：神经网络专用处理器，本文目标部署平台，相比 CPU/GPU 在深度学习推理上更高效。
**Token Downsampler（Token 下采样器）**：将相邻的 2×2 个 ViT token 拼接并通过线性层融合，从而减少图像 token 数量的模块。
**Chunked Computing（分块计算）**：将输入 token 分批（每批 128 个）并行处理以适配 NPU 算力限制的推理策略。
**Mixed-Precision Quantization（混合精度量化）**：对不同模型组件采用不同量化精度（如 INT4/INT8/INT16/FP16）以平衡效率与精度。
**Pipeline Parallelism（流水线并行）**：在不同硬件单元（CPU/NPU）之间重叠执行不同阶段的计算以隐藏延迟的技术。

## 可复现要素
- **数据集**：预训练使用公开数据（LLaVA 558k、ShareGPT4V 1200k、ALLaVA 708k）；微调包含 55.8M 公开数据 + 589.3M 自有数据（自有数据未公开）
- **代码/权重**：论文未明确说明代码和模型权重是否开源
- **关键超参**：基线分辨率 384×384；候选宽高比 1:1~3:3（9 种）；Patch 批大小 4；Chunk 大小 128 tokens；KV Cache 长度 2048；预训练 2 epochs，微调 2 epochs；ViT INT8 / LLM INT4 / LLM 激活 INT16 / ViT 激活 FP16
