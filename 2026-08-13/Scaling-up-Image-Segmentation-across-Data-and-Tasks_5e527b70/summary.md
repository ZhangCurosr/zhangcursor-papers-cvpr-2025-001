---
title: "Scaling-up-Image-Segmentation-across-Data-and-Tasks"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_Scaling_up_Image_Segmentation_across_Data_and_Tasks_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 10:53:30"
field: "通用图像分割"
keywords: ["image segmentation", "scalable model", "mixed query", "open-vocabulary", "synthetic data", "multi-task", "universal segmentation"]
innovations: ["提出混合查询机制，融合learnable与conditional query以兼顾实例与背景分割", "构建统一文本描述训练格式，实现跨数据集跨任务联合扩展", "离线生成合成mask与caption，低成本扩展训练数据规模"]
benchmarks: ["SeginW", "COCO", "ADE20K", "RefCOCOg", "LVIS", "Objects365"]
---

# 论文速读：Scaling-up-Image-Segmentation-across-Data-and-Tasks

## 一句话总结
本文提出了 MQ-Former，一种可通过混合查询（mixed query）机制实现**跨数据集与跨任务联合扩展**的图像分割统一框架。研究证实，随着训练数据规模与任务多样性的增加，模型性能持续提升，并借助低成本合成数据进一步将 SeginW 开放词汇 mask AP 从 33.2 提升至 43.2。

## 研究问题与动机
- 传统分割模型在单一任务/数据集上表现优异，但面对开放词汇、自由形式、真实场景等复杂问题时泛化能力严重不足。
- 已有统一分割模型（如 X-Decoder、DaTaSeg、OpenSeeD、OneFormer 等）要么只能扩展单一任务，要么只能扩展单一数据集，缺乏**同时跨数据与任务可扩展**的架构；尤其在实例级分割上表现不佳。
- 高质量像素级标注成本高昂（COCO 单图需数分钟），大规模数据扩展受限于人工标注瓶颈，亟需低成本且可扩展的数据增强手段。
- 不同查询策略（learnable query 与 conditional query）在 things/stuff 物体上各有优劣，缺乏统一融合机制，限制了模型在多种任务上的泛化。

## 核心贡献（创新点）
1. **提出 MQ-Former 可扩展分割架构**：首次在统一框架下实现多数据集、多任务（语义/实例/全景/前景/指代）联合训练，打破以往模型受限于特定数据集或任务的设计。
2. **引入混合查询（Mixed Query）机制**：通过双路径交叉注意力将 learnable query 与 conditional query 动态融合，实现对 instance-level 与 stuff-level 物体的自适应选择，优于 X-Decoder 单一 learnable query 在实例分割上的短板。
3. **构建低成本合成数据流水线**：利用 SAM/OFA 模型离线生成合成 mask 与对象描述文本，将检测数据集转化为分割数据，并将训练样本从 0.1M 扩展至约 2.2M，大幅降低对人工标注的依赖。
4. **系统验证数据与任务可扩展性**：在 SeginW 等开放词汇基准上，单纯扩展数据与任务即可带来稳定提升，叠加合成数据后最终 mask AP 达 43.2，较无合成数据基线提升 4.6 个点。

## 方法详解
- **整体架构**：基于 Mask DINO，包含图像编码器（Swin Transformer）、文本编码器（CLIP）、分割编码器与分割解码器。支持开放词汇输入与多任务输出。
- **混合查询设计**：查询集由 100 个 learnable query 与 300 个 conditional query 组成。在解码器每层中，通过 conditional-to-learnable（C-to-L）与 learnable-to-conditional（L-to-C）两次交叉注意力实现特征交互，避免刚性分类/任务分配。
- **统一训练格式**：所有任务统一表示为三元组 $(c_j, \mathbf{b}_j, \mathbf{m}_j)$，其中 $c_j$ 可为固定类别标签或自由文本描述，无需区分 stuff/thing，从而兼容 Objects365、Visual Genome 等缺乏统一类别定义的数据集。
- **损失函数**：
  $$\mathcal{L} = \sum_{(x_i, y_i) \in D} \sum_{(c_j, b_j, m_j) \in y_i} \left[ \mathcal{L}_c(P^c, H(c_j)) + \mathcal{L}_b(P^b, b_j) + \mathcal{L}_m(P^m, m_j) \right]$$
  其中 $\mathcal{L}_c$ 为类别 focal loss（类别嵌入与文本嵌入的点积），$\mathcal{L}_b$ 为 generalized IoU + L1，$\mathcal{L}_m$ 为 generalized dice loss。匈牙利匹配在所有查询上统一进行。
- **合成数据生成**：
  - **合成 mask**：基于 SAM 等高质量分割模型，利用 Objects365、Visual Genome 等检测数据集的 bbox 生成伪分割掩码。
  - **合成 caption**：训练 OFA-akin 模型进行对象描述生成，每个对象生成 5 条高置信度描述，训练时随机选取 1 条作为文本监督信号。

## 实验与结果
- **数据集**：训练覆盖 COCO、LVIS、ADE20K、Visual Genome、Objects365、RefCOCO/+/g、HRSOD 等多个公开数据集，总计约 2.2M 图像、57M mask 标注。
- **评估基准**：开放词汇 SeginW（25 个 in-the-wild 数据集）、RefCOCOg、Pascal Context、BDD；封闭集 COCO（PS/IS）、ADE20K（SS）。
- **关键结果**：
  - 在 SeginW 上，仅扩展数据与任务即带来 +7 点提升（33.2 → 38.6）；加入合成数据后进一步提升至 43.2（+4.6）。
  - 混合查询在各类查询策略中表现最优：在 4 数据集/4 任务设置下，COCO Mask AP 49.9（+1.3）、PANoptic PQ 56.8（+2.7）、ADE mIoU 52.1（+8.2）、SeginW AP 38.4（+6.3）。
  - 相较 X-Decoder，MQ-Former 在 COCO PS PQ 提升 1.9，IS Mask AP 提升 5.6，且在 SeginW 开放词汇上显著领先。
  - 在 RefCOCOg 指代分割上，随合成数据扩展性能稳定上升（57.8 → 60.8 → 62.6）。

## 相关工作脉络
- **X-Decoder**：采用单一 learnable query 实现多任务/数据集联合训练，但在实例分割上表现较差；本文通过混合查询弥补其在 instance-level 上的缺陷。
- **OpenSeeD**：支持开放词汇分割，但需要严格的 stuff/thing 类别标注，无法直接用于 Objects365 等无此标注的数据集；MQ-Former 通过统一文本描述消除该限制。
- **OneFormer**：通过 task token 切换完成不同任务，但缺乏跨数据集的类空间统一能力；MQ-Former 以自由文本替代硬分类，实现更灵活的跨数据集扩展。
- **DaTaSeg / OMG-Seg**：同样尝试多任务统一分割，但在实例分割性能上不及 MQ-Former，主要受限于查询设计；本文证明混合查询在兼顾 thing/stuff 上的优势。
- **Mask DINO / Mask2Former**：强单一任务模型，但架构需针对任务调整，无法直接扩展至开放词汇或多任务联合训练场景。
- **PseudoSeg / OpenSeeD 伪标签**：在线生成伪 mask 会增加训练开销；本文采用离线生成合成数据，成本低且易扩展。

## 局限性与未来方向
- 当前合成数据仍依赖于 SAM、OFA 等预训练模型的质量，在低资源类别或长尾分布上可能存在偏差。
- 模型在极端开放词汇（未见类别极多、描述模糊）下的泛化能力尚未充分验证。
- 未探索更进一步的缩放极限（如千万级图像、更多样任务组合）。
- 混合查询的计算开销略高于单一查询策略，在实际部署中需权衡精度与效率。
- 未来方向包括：更大规模合成数据 pipeline、跨模态统一表示学习、以及在机器人/自动驾驶等真实场景中的验证。

## 研究启发与可借鉴点
- **混合查询机制**可迁移至其他检测/分割任务，尤其适用于需要同时处理 instance 与 stuff 的场景。
- **统一文本描述替代硬类别分配**的思路可用于构建更多免标注依赖的通用视觉模型。
- **离线合成数据生成**（mask + caption）策略成本低、易扩展，可作为数据受限任务的常用增强手段。
- **数据与任务联合扩展**的训练范式值得在其他多任务学习（如检测+分割+描述）中推广。
- 本文的实验设计（逐步扩展数据规模与任务数量）为验证模型可扩展性提供了清晰且可复现的评估范式。

## 关键术语表
- **Mixed Query**：融合 learnable query 与 conditional query 的动态查询机制，通过双路径交叉注意力实现实例与背景物体的自适应选择。
- **Scalable Segmentation**：指模型性能随训练数据规模与任务多样性增加而持续上升的能力。
- **Synthetic Data**：利用预训练模型（如 SAM、OFA）离线生成的伪分割掩码与对象描述文本，用于扩充训练数据。
- **Open-vocabulary Segmentation**：模型能够识别训练阶段未明确见过的类别，依赖文本嵌入实现跨类别泛化。
- **Learnable Query**：从零开始训练的固定查询向量，擅长捕捉全局背景/stuff 特征。
- **Conditional Query**：由编码器输出衍生并筛选的查询，擅长定位局部实例/thing 物体。
- **SeginW**：包含 25 个真实世界数据集的开放词汇分割评测基准，用于评估模型 in-the-wild 泛化能力。
- **Generalized Dice Loss**：用于处理类别极度不平衡 segmentation 任务的损失函数，避免 majority class 主导训练。

## 可复现要素
- **数据集**：COCO、ADE20K、LVIS、Visual Genome、Objects365、RefCOCO/+/g、HRSOD 等均为公开数据集；合成数据利用 SAM/OFA 离线生成。
- **代码与权重**：论文未明确声明开源，但模型基于 Mask DINO 与 X-Decoder 架构改进，相关组件可参考开源实现。
- **关键超参**：learnable query 数量 100，conditional query 数量 300；图像编码器为 Swin Transformer，文本编码器为 CLIP；训练损失权重未在文中详细列出（以 Mask DINO 默认设置为准）。
