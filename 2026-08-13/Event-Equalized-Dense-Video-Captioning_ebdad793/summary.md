---
title: "Event-Equalized-Dense-Video-Captioning"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Wu_Event-Equalized_Dense_Video_Captioning_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 07:04:04"
field: "密集视频描述与视频理解"
keywords: ["Dense Video Captioning", "Temporal Bias", "Query Initialization", "Hierarchical Clustering", "Deformable Transformer"]
innovations: ["提出EPM无训练聚类感知伪事件，首次在不依赖时间分布的前提下实现事件公平感知", "提出PEI模块将伪事件时间中心编码为decoder query初始化，显著提升短事件召回", "设计EEE模块将伪事件label嵌入拼接进Transformer编码器，缓解短事件特征被淹没问题"]
benchmarks: ["ActivityNet Captions", "YouCook2"]
---

# 论文速读：Event-Equalized Dense Video Captioning

## 一句话总结
论文提出 **E²DVC（Event-Equalized Dense Video Captioning）** 框架，利用无训练的聚合层次聚类从纯视觉特征中感知伪事件，并通过伪事件初始化模块对等分配注意力，解决密集视频描述中长期存在的时间偏差问题（对长/短事件关注度不均），在 ActivityNet Captions 和 YouCook2 上均取得 SOTA 性能。

## 研究问题与动机
- **数据分布偏差**：训练集中事件的时间轴分布极不均匀，模型倾向于学习"容易"的时间段，分布外事件被忽略或误定位。
- **事件时长偏差**：长时长事件包含更多帧特征，在自注意力中更容易压倒短时长事件，导致短事件被严重遗漏（图 1(b) 显示预测的短时长事件远少于 GT）。
- **纯时间线索失效**：现有方法均依赖时间位置编码来定位事件，但这种设计天然偏向某些时长和位置分布，需要一种不依赖时间的替代感知手段。
- **动机**：不同事件在视觉特征上具有明显差异（不同视角、背景、主体），利用视觉相似度进行初步分段可以实现对事件的"公平"对待。

## 核心贡献（创新点）
1. **发现并提出"时间偏差（temporal bias）"问题**：系统性地指出 DVC 任务中时间分布不均和时长偏差两个子问题，并论证其对模型公平性的破坏——现有工作均以此为前提但未提出针对性解决方案。
2. **无需训练的 Event Perception Module（EPM）**：首次在 DVC 中将聚合层次聚类用于纯视觉特征分段，聚类结果不受训练集时间分布影响，且短/长事件可被同等聚类到同一类别——这是 E²DVC 与其他方法最本质的区别。
3. **Pseudo-Event Initialization（PEI）模块**：将伪事件的时间中心位置编码为解码器 query 的初始化 embedding，替代随机初始化 query，使模型一开始就将注意力分配给所有伪事件——与 Conditional DETR / Anchor DETR 等目标检测 query 初始化思路在 DVC 任务中的首次迁移。
4. **Event-Enhanced Encoder（EEE）模块**：将伪事件 label embedding 拼接进帧特征送入可变形 Transformer 编码器，同时在帧-帧和帧-事件两个层次提取关系，使短事件特征在自注意力中不被淹没——区别于 PDVC / CM² 仅靠帧特征的编码设计。

## 方法详解
整体框架为 encoder-decoder + 并行 subtask heads。

**1. Event Perception Module（EPM）**
- 视频以 1 FPS 采样得帧序列 $\{v_i\}_{i=1}^F$，经预训练图像编码器（CLIP ViT-L/14、C3D 或 TSN）提取视觉特征 $X = \{x_i\}$。
- 对 $X$ 执行 **Agglomerative Hierarchical Clustering**，每次合并特征空间最近的两个簇，迭代至剩余 $N_c$ 个簇：
  $$\mathbb{C} = \{\mathbb{C}_i\}_{i=1}^{N_c} = \{\{x_i^j\}_{j=1}^{N_c^i}\}_{i=1}^{N_c}$$
- **Refinement mechanism**：由于聚类忽略时间信息，同簇帧可能时间不连续；按时间戳分裂不连续帧，并丢弃时长 $< \tau$ 的异常短簇，剩余即为 **pseudo-events**。
- 每帧分配对应的伪事件标签 $l_i$。

**2. Event-Enhanced Encoder（EEE）**
- 伪事件标签 $l_i$ 经标签编码器（可学习字典 + MLP）得 label embedding $le_i$。
- 帧特征拼接 label embedding 与位置编码（PE）：$[x_i, le_i, PE_i]$，送入 **可变形 Transformer Encoder + 多尺度卷积**，同时学习 frame-frame 和 frame-event 关系，避免短事件特征被长事件淹没。

**3. Pseudo-Event Initialization（PEI）**
- 对每个伪事件 $E_i$ 计算时间中心点：
  $$t_i = (t_{i,1} + t_{i,L_i}) / 2$$
- 对 $t_i$ 做 inverse sigmoid 后经 temporal encoder $f_t$ 得 query 初始化：
  $$q_i = f_t(\mathrm{Inv}_{Sig}(t_i))$$
- 将全部伪事件 query 与 $N_r$ 个随机 query 拼接作为 decoder query 输入，使模型显式优先关注伪事件位置。

**4. Task Heads（并行解码）**
- **Localization Head**：对每个 query 输出 $\{t_s^i, t_e^i, c^i\}$。
- **Captioning Head**：以 LSTM 为骨干，每个时间步输入 context feature、event query、前序词，生成完整句子。
- **Event Counter**：对全局 query 做 max-pooling → FC → softmax，argmax 得事件数 $N_{event}$。
- **最终置信度**（融合定位与 caption 两路）：
  $$c_i = c_i^{loc} + \frac{\mu}{M_i^{\gamma}} \sum_{t=1}^{M_i} \log(c_{i,t}^{cap})$$

**5. Loss Function**
$$L = \alpha_{gious} L_{gious} + \alpha_{cls} L_{cls} + \alpha_{ec} L_{ec} + \alpha_{cap} L_{cap}$$
其中 $L_{cap}$ 为归一化交叉熵，$L_{ec}$ 为计数分布交叉熵，通过 Hungarian matching 做二分图匹配。

## 实验与结果
- **数据集**：ActivityNet Captions（~20000 视频，均 120s，均 3.7 标注）、YouCook2（2000 烹饪视频，均 320s，均 7.7 标注）；均采用 YouTube 可用视频子集。
- **特征**：CLIP ViT-L/14、C3D（PDVC 提供）、TSN（CM² 提供）。
- **基线**：PDVC、CM²、Vid2Seq、E2ESG、MT、ECHR、UEDVC 等。
- **评估指标**：BLEU4、METEOR、CIDEr、SODA_c；定位 F1（IoU={0.3,0.5,0.7,0.9}）。

**主要结果**：
| 数据集 | 特征 | 方法 | BLEU4 | METEOR | CIDEr | 定位 F1 |
|---|---|---|---|---|---|---|
| ActivityNet | CLIP | **E²DVC** | **2.43** | **8.57** | **33.63** | **56.14** |
| ActivityNet | CLIP | PDVC* | 2.21 | 8.06 | 29.97 | — |
| ActivityNet | TSN | **E²DVC** | **2.03** | **8.02** | **29.91** | **56.42** |
| ActivityNet | C3D | **E²DVC** | **1.79** | **7.54** | **26.83** | **56.17** |
| YouCook2 | CLIP | **E²DVC** | **1.68** | **6.11** | **34.26** | **28.64** |
| YouCook2 | TSN | **E²DVC** | **1.05** | **4.76** | **23.15** | — |

- **最强提升**：ActivityNet CLIP 上 BLEU4 较 PDVC* 提升 **+9.7%**，CIDEr 提升 **+12.3%**；YouCook2 CLIP 上 CIDEr 较 PDVC* 提升 **+15.4%**，且优于所有非预训练方法。
- 定量分析（图 4、5）：E²DVC 预测的短时长事件比例更接近 GT，有效缓解了时长偏差。

**消融**：
- PEI 单独：BLEU4 +6.4%，定位 F1 +1.3%；EEE 单独：+6.4% / +0.9%；两者叠加达最优。
- $N_c = 5$、$\tau = 4$ 为最优超参。

## 相关工作脉络
1. **PDVC [48]**：端到端并行解码的 DVC 强基线，E²DVC 的核心对比对象；区别在于 PDVC 仍依赖时间位置编码初始化 query，对时长/分布偏移敏感。
2. **CM² [18]**：引入 cross-modal memory retrieval 的 SOTA；E²DVC 定位更优（F1 56.14 vs 53.71），但 CM² 使用更大预训练数据。
3. **Vid2Seq [53]**：1M 额外视频预训练的 DVC 模型；在 YouCook2 上优于 E²DVC，但属预训练方法，E²DVC 在非预训练设定下最强。
4. **DETR 系列（Conditional DETR、Anchor DETR、DAB-DETR）**：查询初始化相关；E²DVC 是首个将其思想迁移到 DVC 任务的工作。
5. **两阶段方法（BMN、DAPS 等）**：先 proposal 再描述；E²DVC 属于端到端范式，无需手工组件。
6. **Streaming Dense Video Captioning [60]**、**Video Recap [17]**：面向长视频流式/分层 captioning；E²DVC 聚焦于时间偏差纠正，方法正交可结合。

## 局限性与未来方向
- **长事件定位精度仍有偏差**：为提升 caption 质量，部分长事件会牺牲定位精度（定位损失与 caption 损失存在折衷）。
- **聚类模块虽无需训练，但超参敏感**：$N_c$ 和 $\tau$ 需逐数据集调优，且无法利用标注数据中的时间分布模式。
- **无法处理视觉相似但语义不同事件**：纯视觉聚类可能将不同事件归属同一伪事件簇。
- **未来方向**：① 引入可训练的视觉-语义联合感知模块；② 扩展至多模态（音频 + 视频）场景；③ 与流式 / 分层 DVC 框架结合处理超长视频。

## 研究启发与可借鉴点
1. **"去时间化"的初始感知思路**：用纯视觉聚类替代时间位置作为事件初筛，这一范式可迁移到其他视频理解任务（如 action localization、temporal proposal generation）中解决数据分布偏差问题。
2. **伪事件标签作为辅助监督信号**：EEE 将伪事件 label embedding 拼入 Transformer 编码层，本质是一种轻量级的结构先验注入，可在任意视觉编码器中复用。
3. **Query 初始化策略的系统化设计**：PEI 证明了将"外部感知到的候选位置"编码为 query 初始 embedding 的有效性，可直接借鉴到 Temporal Action Detection、Video Grounding 等 query-based 任务。
4. **公平性视角的评测意识**：论文通过 Figure 1、4、5 显式刻画模型在不同时长/位置区间上的召回差异，这种"公平性子集分析"可作为后续 DVC 论文的标配评测手段。
5. **轻量化与 SOTA 的平衡**：EPM 完全无参，整个额外开销极小却带来显著增益，提示在视频理解任务中"设计非可训练模块弥补数据缺陷"是一条高性价比路线。

## 关键术语表
- **Dense Video Captioning (DVC)**：在未剪辑长视频中同时定位所有事件并对每个事件生成自然语言描述的联合任务。
- **Temporal Bias（时间偏差）**：模型因训练集时间/时长分布不均而对特定类型事件关注不足的系统性偏差。
- **Agglomerative Hierarchical Clustering**：自底向上的层次聚类算法，逐步合并最近簇直至满足终止条件，无需预设簇形状或大小。
- **Pseudo-Event**：由 EPM 聚类 + 细化机制输出的潜在事件片段，作为后续编码和 query 初始化的先验。
- **PEI（Pseudo-Event Initialization）**：将伪事件的时间中心编码为 decoder query 初始 embedding 的模块。
- **EEE（Event-Enhanced Encoder）**：在 Transformer 编码器中引入伪事件 label embedding，同时建模 frame-frame 和 frame-event 关系的模块。
- **Generalized IoU Loss（GIoU）**：考虑预测与 GT 区间并集面积的边界框回归损失，比标准 IoU 更适用于时序定位。
- **SODA_c**：面向故事性（story-oriented）的 DVC 评测指标，衡量生成描述间的连贯性。

## 可复现要素
- **数据集**：ActivityNet Captions 和 YouCook2，均公开可用；论文使用 YouTube 在线子集（约少 7% 视频）。
- **代码/权重**：论文未明确提及代码开源，需联系作者获取。
- **关键超参**：帧采样 1 FPS；序列长度 F=100（ANet）/ 200（YouCook2）；$N_c = 5$；$\tau = 4$；ANet 事件查询数 10，YouCook2 查询数 100；编码器为两层 deformable transformer，multi-scale 四层级；使用 CLIP ViT-L/14 或 C3D/TSN 特征；其余超参遵循 PDVC [48]。
