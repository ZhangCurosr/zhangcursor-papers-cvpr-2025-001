---
title: "DivPrune-Diversity-based-Visual-Token-Pruning-for-Large-Mult"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Alvar_DivPrune_Diversity-based_Visual_Token_Pruning_for_Large_Multimodal_Models_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 07:00:32"
field: "多模态大模型高效推理"
keywords: ["Visual Token Pruning", "Large Multimodal Models", "Max-Min Diversity", "Efficient Inference", "Token Selection"]
innovations: ["将视觉Token剪枝形式化为MMDP并通过贪心算法精确求解", "提出无需微调/校准的即插即用多样性剪枝方法DivPrune", "在16个图像/视频理解数据集上实现极端压缩（≥80%）下的SOTA精度"]
benchmarks: ["COCO", "GQA", "MMMU", "POPE", "MMBench", "MME", "SeedBench", "ActivityNet", "NextQA", "EgoSchema"]
---

# 论文速读：DivPrune: Diversity-based Visual Token Pruning for Large Multimodal Models

## 一句话总结
本文提出 DivPrune，一种将视觉 Token 剪枝形式化为**最大最小多样性问题（MMDP）**的即插即用、无训练/无校准方法，通过最大化选定 Token 间的最小余弦距离来降低冗余，在高达 80%-90% 的极端剪枝比下仍能保持 SOTA 精度，无需微调即可直接部署于各类 LMMs。

## 研究问题与动机
- **推理延迟与内存瓶颈**：LMMs 将图像/视频编码为数千个视觉 Token，总序列长度剧增，Transformer 复杂度随序列长度二次增长，严重影响端到端延迟与显存占用。
- **现有注意力剪枝的缺陷**：FastV、PruMerge 等方法依赖注意力分数筛选 Token，但注意力分数并非最优的重要性指标（部分重要 Token 被忽略），且倾向于保留相互相似的 Token，导致高压缩比下冗余严重、代表性下降。
- **校准/微调类方法的高成本**：FitPrune、M³ 等方法需要为每个模型单独校准或重新训练，耗时耗力，难以泛化至新模型。
- **极端压缩下的性能骤降**：实验表明在 TFLOP 比例 ≤ 15% 时，基于注意力的方法性能下降超过 40%，而多样性导向的方法能更优雅地退化。

## 核心贡献（创新点）
- **首次将 Token 剪枝建模为 MMDP**：以最大化子集内最小成对距离为目标，从数学上保证所保留 Token 的多样性；与已有工作以"重要性评分"为目标的本质区别在于目标函数从"保留最显著 Token"转向"保留最多样 Token"。
- **训练无关、校准无关的即插即用方案**：无需额外微调或校准数据集，可直接嵌入任何 LMM（兼容任意 LLM 架构与视觉编码器）；与 FitPrune/M³ 等需要离线优化的方法形成鲜明对比。
- **支持 KV 缓存等推理优化技术**：剪枝发生在 Prefill 阶段一次性计算距离矩阵，不影响自回归解码过程；与 FastV/VTW 等在每次解码步需重新计算的重要性指标相比更具工程优势。
- **16 数据集 SOTA 与跨模型泛化验证**：在图像理解（11 数据集）与视频理解（5 数据集）上全面评测，LLaVA 1.5/1.6 及 LLaVA-NeXT-Video 系列均取得最优或最具竞争力的结果，尤其极端剪枝（TFLOP 比例 ~15%）下优势显著。
- **揭示"冗余去除可提升性能"的实证发现**：在 POPE 等幻觉敏感数据集上，DivPrune 在不增加 FLOPs 的情况下反而略微提升了原始模型精度，表明冗余 Token 的去除具有正则化效果。

## 方法详解
- **问题形式化**：给定视觉 Token 集合 E_v（|E_v| = M），目标是选取大小为 M̃（M̃ < M）的子集 Ẽ_v，最小化剪枝前后模型输出分布的差异（公式 2）。
- **MMDP 建模**：将目标替换为最大化子集内最小成对距离（公式 3）：
  - 目标：find Ẽ_v = arg max [ min_{γ,ω∈S} d(γ,ω) ]，其中 S ⊂ E_v，|S| = M̃。
  - 距离度量采用余弦距离：d(γ, ω) = 1 - (γ·ω)/(‖γ‖‖ω‖)（公式 4）。
- **精确求解算法（Algorithm 1）**：
  - **初始化**：候选集 R = E_v，选定集 Ẽ_v = ∅。
  - **第一阶段（选首个 Token）**：对每个候选 Token，计算其到其他所有候选 Token 的最小成对距离 d_min；选择 d_min 最大的 Token 加入 Ẽ_v。
  - **第二阶段（迭代选择）**：对每个候选 Token，计算其到已选集合 Ẽ_v 中所有 Token 的最小距离；重复选择最小距离最大的候选 Token，直至达到目标数量 M̃。
  - **计算优化**：预先一次性计算完整距离矩阵（一次矩阵乘法），避免迭代中重复计算；GPU 上开销可忽略不计。
- **剪枝位置**：默认在视觉 Token 进入 LLM 第一个 Decoder 层之前（Layer 0）执行剪枝；论文也验证了可在中间层应用（Ablation Tab. 3）。
- **与注意力剪枝的本质差异**：注意力方法按"单 Token 重要性"排序后截断，易聚集同类 Token；DivPrune 按"集体多样性"贪心选择，确保覆盖更广泛的视觉语义空间。

## 实验与结果
- **评测模型**：LLaVA 1.5-7B / 1.5-13B / 1.6-7B、LLaVA-NeXT-Video-7B，均使用 CLIP ViT 视觉编码器。
- **图像理解数据集（11个）**：COCO-CIDEr、Flickr30k-CIDEr、GQA-EM、MMB-Acc、MME-P-score、MMMU-Acc、Nocaps-CIDEr、OKVQA-EM、POPE-F1、SQA-EM、SeedBench-Acc。
- **视频理解数据集（5个）**：ActivityNet-Score/Acc、SeedBench-Acc、VideoChatGPT-Score、NextQA-WUPS、EgoSchema-Acc。
- **基线方法**：FastV、PruMerge、VTW（即插即用）、FitPrune（需校准）、M³（需微调）。
- **核心结果（LLaVA 1.5-7B，TFLOP 比例 ~15.6%）**：
  - COCO CIDEr：Ours = 0.96 vs. FastV = 0.06 vs. VTW = 0.05；仅下降 12.7%（原始 1.10 → 0.96）。
  - POPE F1：Ours = 86.02（略超原始 85.84）vs. FastV = 32.84 vs. VTW = 25.35。
  - GQA EM：Ours = 56.85（原始 61.96，下降 5.1%）vs. FastV = 38.73（下降 37.5%）vs. VTW = 38.94。
  - MMB Acc：Ours = 59.19 vs. FastV = 20.62 vs. VTW = 21.31。
- **极端压缩场景（Fig. 1）**：TFLOP 比例 ≤ 25% 时，所有基线性能骤降，DivPrune 仍保持平滑下降趋势，显著领先。
- **视频理解结果（Tab. 2）**：LLaVA-NeXT-Video-7B，TFLOP 比例 ~14.1%，DivPrune 在 ActivityNet、SeedBench、EgoSchema 上较 FastV 提升最高 12%、较 VTW 提升最高 19%。
- **效率**：相比原始模型，GPU 显存减少 ~400MB，Prefill 时间减少 55%，E2E 延迟减少 22%；与基线相比 Prefill 时间多 6-7%（仅一次距离计算），E2E 时间缩短 1-7%。
- **最强结果**：LLaVA 1.6-7B 在 TFLOP 比例 ~10.79% 下，POPE F1 仅下降 3.4%（原始 86.38 → 82.97），MMMУ 准确率略有提升。

## 相关工作脉络
- **FastV [4]**：基于早期层注意力分数量级剪枝，训练无关但忽略 Value 信息，高压缩比下冗余严重；DivPrune 以多样性目标取代单一重要性评分。
- **PruMerge [36]**：在 Vision Encoder 侧按注意力稀疏度聚类合并 Token，支持可变剪枝比；DivPrune 不合并 Token 而是选择多样性子集，且可在 LLM 内部任意层应用。
- **VTW [21]**：主张视觉 Token 经若干 LLM 层后可完全移除，需针对任务校准最佳移除层；DivPrune 无需校准且可精确控制 TFLOP 比例。
- **FitPrune [47]**：利用校准集计算剪枝后注意力散度以制定剪枝策略，属校准类方法；DivPrune 在零校准条件下达到更优性能（如 MMB 高出 1.5%、POPE 高出 25.1%）。
- **M³ [3]**：通过微调生成嵌套多层级 Token 表示，支持动态长度选择；DivPrune 无需任何训练即可实现相近甚至更优的极端压缩表现。
- **TokenPacker [17]**：训练专用 Projector 将细粒度信息压缩进紧凑 Token；DivPrune 不修改模型架构或训练参数，通用性更强。

## 局限性与未来方向
- **距离矩阵计算开销随 Token 数平方增长**：对于超长视频（如 LLaVA-NeXT-Video 每帧 144 Token × 8 帧 = 1152 Token），预计算距离矩阵的内存和时间开销需进一步权衡。
- **仅适用于 Transformer 架构 LMMs**：方法依赖 Token 级特征向量的余弦距离，对 State Space Model（如 Mamba 类）或直接像素级输入架构的推广性未验证。
- **未探索多模态任务间的通用剪枝策略**：不同任务（如 OCR vs. 视觉推理）可能对 Token 多样性需求不同，固定策略可能在某些任务上并非最优。
- **缺乏理论保证**：MMDP 的贪心算法有近似比理论下界，但论文未给出具体界与任务性能之间的理论联系。
- **未研究剪枝后 Token 的插值/补偿机制**：直接丢弃未选中 Token 而非替换，可能导致部分语义信息永久丢失；未来可探索以代表 Token 重建的方式进一步压缩。

## 研究启发与可借鉴点
- **"多样性优先于重要性"的设计范式**：可将 Max-Min Diversity 思路迁移至其他模态（如音频 Token、3D 点云 Token）的压缩场景，或序列模型中的 Key/Value Cache 缩减。
- **一次性预计算距离矩阵的工程技巧**：在 Prefill 阶段完成所有距离计算后释放内存，不影响自回归解码；这一策略可推广至其他需要离线选择子集的场景。
- **极端压缩比的系统性评测**：本文在 TFLOP ≤ 25% 的极端设置下全面比较，揭示了注意力方法在此区间的"悬崖式"性能衰减，为后续工作提供了可靠的基准参照。
- **冗余去除的正则化效应**：实验表明适当剪枝可轻微提升 POPE 等幻觉敏感任务性能，说明去除冗余信息本身可能抑制模型过拟合；这一观察可启发稀疏表示学习与幻觉缓解的联合研究。
- **跨模型/跨任务泛化验证的完整性**：本文同时覆盖图像与视频、不同尺寸 LLaVA 系列，验证方法普适性；未来工作可沿用此评测协议以公平对比。

## 关键术语表
**Large Multimodal Models (LMMs)**：融合文本与视觉（图像/视频）输入的大型语言模型，通常由视觉编码器、投影层与预训练 LLM 组成。
**Visual Token Pruning**：在 LMM 推理过程中主动移除冗余视觉 Token 以降低计算开销的技术。
**Max-Min Diversity Problem (MMDP)**：组合优化问题，目标是从全集选出固定大小的子集，使得子集中任意两点间的最小距离最大化。
**TFLOP Ratio**：剪枝后模型相对原始模型的浮点运算量比例，用于量化计算压缩程度。
**CIDEr Score**：基于 n-gram 共识的图像描述质量评估指标，越高越好。
**POPE F1**：衡量模型视觉幻觉程度的指标，综合精确率与召回率，F1 越高表示幻觉越少。
**KV Caching**：在自回归解码中将已计算 Key/Value 矩阵缓存以避免重复计算的推理优化技术。
**Plug-and-Play**：指无需模型微调或额外校准即可直接应用的方法特性。

## 可复现要素
- **数据集**：COCO、Flickr30k、GQA、MMBench、MME、MMMU、Nocaps、OKVQA、POPE、SQA、SeedBench、ActivityNet、VideoChatGPT、NextQA、EgoSchema（均为公开数据集）。
- **代码开源**：是，论文声明代码已在 CVPR 2025 附录链接处公开（"The code is available here⋄"）。
- **模型权重**：使用 LLaVA 1.5 / 1.6 及 LLaVA-NeXT-Video 官方权重。
- **关键超参**：
  - 视觉 Token 数量：LLaVA 1.5 为 576，LLaVA 1.6 为 1728-2880（3-5倍），LLaVA-NeXT-Video 每帧 144 Token × 8 帧 = 1152 Token。
  - 距离度量：默认余弦距离，消融实验对比 l1/l2 距离。
  - 剪枝比例：通过控制目标子集大小 M̃ 实现，与 TFLOP 比例对应。
- **实验环境**：8 × V100 GPU（32GB VRAM），batch size = 1，使用 lmmsevals 包评测。
- **评估 LLM**：GPT-4o-mini（用于 GPT-assisted 评分）。
