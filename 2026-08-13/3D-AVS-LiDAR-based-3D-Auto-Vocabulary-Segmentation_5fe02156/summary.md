---
title: "3D-AVS-LiDAR-based-3D-Auto-Vocabulary-Segmentation"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Wei_3D-AVS_LiDAR-based_3D_Auto-Vocabulary_Segmentation_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:54:17"
field: "3D开放词汇语义分割"
keywords: ["3D auto-vocabulary segmentation", "LiDAR point cloud", "open-vocabulary segmentation", "CLIP", "point captioning", "SMAP", "TPSS metric"]
innovations: ["首次提出3D点云自动词汇分割任务，动态生成场景特定词汇无需人工定义", "设计仅依赖几何信息的LiDAR点云captioner，通过SMAP模块实现多区域可控描述生成", "提出TPSS无标注评估指标，在CLIP空间中度量点-文本语义一致性避免标注偏差"]
benchmarks: ["nuScenes", "ScanNet", "ScanNet200"]
---

# 论文速读：3D-AVS-LiDAR-based-3D-Auto-Vocabulary-Segmentation

## 一句话总结
本文提出了3D-AVS，首个面向LiDAR点云的自动词汇分割框架，通过自生成的场景相关词汇替代人工预定义类别，实现了无需人工干预的3D开集分割；同时设计了仅依赖几何信息的点云captioner与无标注评估指标TPSS，在nuScenes与ScanNet200上验证了方法的语义一致性与分割精度优势。

## 研究问题与动机
1. **预定义词汇的局限性**：现有3D开词汇分割（OVS）方法依赖人工指定的文本查询或预标注类别集合，无法覆盖训练时未知的新类别，限制了在动态真实场景中的可扩展性。
2. **3D自动词汇分割的空白**：2D领域已有AutoSeg、ZeroSeg等AVS方法，但3D点云的AVS任务尚未被探索，缺乏能够自动识别点云对象并生成对应语义标签的框架。
3. **恶劣环境下图像信息的不足**：低光照、雨雾等挑战条件下，相机图像信息不可靠，而LiDAR点云的几何信息具有颜色无关性，可提供稳健的语义描述。
4. **未知词汇评估的困难**：自动生成的词汇存在同义词、层级关系与语义重叠等自然语言歧义，传统mIoU等基于固定类别的指标无法准确评估其质量。

## 核心贡献（创新点）
1. **首次提出3D自动词汇分割任务**：不同于依赖预定义词汇的OVS方法，3D-AVS针对每个输入动态生成场景特定的词汇，无需人工定义目标类别，实现了"unknown vocabulary"设置下的点云分割。
2. **设计双模态captioner框架**：结合图像captioner（使用xGen-MM）与自主研发的LiDAR点云captioner，前者提供语义多样性，后者在低光照/恶劣天气下弥补图像缺失，两者互补提升整体鲁棒性。
3. **提出SMAP模块增强点云表征多样性**：Sparse Masked Attention Pooling模块通过几何分区（室外柱坐标扇区划分、室内柱状分区）生成可控数量的点云子集特征，使point captioner能输出多个不同粒度的描述，克服了LidarCLIP仅能生成全局单一描述的局限。
4. **引入TPSS无标注评估指标**：Text-Point Semantic Similarity指标在CLIP对齐空间中计算点特征与文本嵌入的语义相似度均值，无需依赖数据集标注即可评估生成词汇的语义一致性，兼容同义词/层级关系等自然语言歧义。
5. **构建完整的"识别-解析-分割"流水线**：将场景caption生成→Caption2Tag词性解析→CLIP文本编码→点-像素特征融合→相似度分类串联为端到端可部署流程，可与现有OVS segmenter（如OpenScene）无缝集成。

## 方法详解
**整体架构**：输入为点云 $\mathbf{P} \in \mathbb{R}^{N \times 3}$ 及对应图像集合 $\mathbf{I} \in \mathbb{R}^{K \times H \times W \times 3}$，分为caption生成、文本解析、特征编码与分割四阶段。

**Scene Captioning（场景描述生成）**：
- 图像分支：采用xGen-MM（BLIP-3家族）多模态大模型，通过beam search生成多张图像的caption列表 $\mathbf{D} = \{d_{\text{im}}^{(k)}\}_{k=1}^K$。
- 点云分支：Point Captioner基于transfer learning从2D CLIP图像编码器蒸馏知识至3D backbone，仅SMAP为可训练组件，其余编码器（CLIP text encoder $h_{\text{tx}}$、高分辨率图像编码器 $h_{\text{im}}^{\text{hr}}$、CLIP对齐点编码器 $h_{\text{pt}}$）均冻结。
- 推理时Point Captioner无需相机内参，采用几何分区策略：室外点云转换为柱坐标 $(\rho_n, \varphi_n, z_n)$ 后按极角 $\varphi$ 划分为 $T$ 个扇区，生成二进制掩码 $b_n^t = \mathbb{I}\left[\frac{t}{T}2\pi \leq \varphi < \frac{t+1}{T}2\pi\right]$；室内则采用方形柱状分区。

**SMAP（Sparse Masked Attention Pooling）模块**：
- 输入：全量点云坐标 $\mathcal{C} \in \mathbb{R}^{N \times 3}$、点特征 $\mathcal{F} \in \mathbb{R}^{N \times C}$，以及 $J$ 个二进制掩码 $\mathcal{B} \in \mathbb{R}^{J \times N}$（训练时 $J=K$，推理时 $J=T$）。
- 相对位置编码后应用掩码：$\mathcal{F}' = \mathcal{B} * (\mathcal{F} + \text{PE}(\mathcal{C}, \mathcal{F}))$。
- 对每个掩码对应的子集进行零填充至相同长度后送入MHA作为Key/Value，同时对子集执行GAP作为Query，最终输出 $J$ 个 pooled features $\mathcal{F}'' \in \mathbb{R}^{J \times C}$。
- 训练时由CLIP图像编码器全局特征监督SMAP聚合结果。

**Text Parsing（Caption2Tag）**：使用spaCy提取caption中的复合名词和专有名词，经lemmatization去重后通过WordNet验证，得到 $M$ 个场景特定的tag集合 $\mathbf{L} = \{l_m\}_{m=1}^M$。

**Segmentation（分割）**：
- 文本编码：$\mathbf{E}_{\text{tx}} = h_{\text{tx}}(\mathbf{L}) \in \mathbb{R}^{M \times C}$。
- 图像特征提升：通过点-像素映射 $\varGamma: \mathbb{R}^3 \to \mathbb{R}^2$ 将可见点的像素特征 $\bar{f}_n^{\text{im}}$ 分配给对应点。
- 点特征：$\mathbf{F}_{\text{pt}} = h_{\text{pt}}(\mathbf{P})$。
- 最终分类：$\hat{l}_n = \arg\max_m \left(\max\left(\text{SIM}(f_n, e_m) \parallel \text{SIM}(\bar{f}_n^{\text{im}}, e_m)\right)\right)$，SIM为点积。

**TPSS指标**：
$$S_n = \max_m \left(\text{SIM}(f_n, e_m)\right), \quad \text{TPSS} = \text{mean}_n(S_n)$$
衡量点特征与生成词汇中最近文本嵌入的平均语义相似度，编码器对齐前提下可跨方法比较。

**LLM映射评估**：使用GPT-4o将自动生成的open-ended类别映射到数据集固定ground truth类别（扩展LAVE方法），进而计算mIoU。

## 实验与结果
**数据集**：nuScenes（16类，室外自动驾驶）、ScanNet（20类，室内）、ScanNet200（200类细粒度，室内），均使用LiDAR点云数据。

**TPSS结果（Tab. 1）**：
- nuScenes：官方标签7.39 → OpenScene扩展标签8.70 → 3D-AVS-Image 8.78 → 3D-AVS-LiDAR 8.80 → 3D-AVS（融合）9.65，较官方标签提升30.5%，较人工扩展标签提升10.9%。
- ScanNet：官方标签3.44 → 3D-AVS-Image 3.49 → 3D-AVS-LiDAR 3.71 → 3D-AVS（融合）3.78，较官方标签提升9.9%。
- 夜间/雨天场景下点云captioner表现更优（Fig. 5），验证了LiDAR模态的鲁棒性。

**mIoU结果（Tab. 2）**：
- nuScenes：3D-AVS达到36.2，超越OpenScene（30.1）、Diff2Scene（未报告）等基线，为当前SOTA。
- ScanNet：3D-AVS为40.5，略低于Diff2Scene的48.6，但作者指出这是因自动词汇映射到粗糙20类标签时的粒度损失。
- ScanNet200：3D-AVS达到14.6，为SOTA，验证了方法在细粒度场景下的有效性。

**消融与分析**：
- 模糊类别（drivable surface、terrain、manmade）在3D-AVS下获得显著增益（mIoU分别达68.2/41.4/55.4），因自动词汇提供了更精确的子类别（如building、staircase、glass door）。
- 点云captioner在夜间场景显著优于图像captioner，双模态融合效果最佳。

## 相关工作脉络
1. **Open-Vocabulary Segmentation (OVS)**：CLIP[39]开创2D图文对齐范式，ULIP[55]、CLIP2Scene[8]、OpenScene[38]等将其扩展至3D点云，但均需预定义文本查询，本文通过自动captioning消除这一限制。
2. **Auto-Vocabulary Segmentation (AVS)**：ZeroSeg[42]用DINO聚类+CLIP获得2D二值mask后由LLM生成文本；AutoSeg[70]基于BLIP embeddings聚类+解码生成名词集；CaSED[13]检索外部数据库caption。本文是3D领域首个AVS工作，与PoVo[32]（并发独立工作）不同，本文更注重与现有OVS segmenter的无缝集成。
3. **Captioning 2D/3D Data**：2D领域BLIP[26]、xGen-MM[56]已成熟；3D领域LidarCLIP[17]仅生成scene-level全局描述，本文Point Captioner通过SMAP实现多region可控描述，覆盖更丰富。
4. **AVS评估方法**：ZeroSeg依赖主观评估，AutoSeg的LAVE将生成类别映射到固定标签但易丢弃更精确的自动词汇，本文TPSS在CLIP空间中直接度量文本-点语义一致性，避免了标注偏差。
5. **CLIP-aligned 3D Encoders**：LidarCLIP[17]将稀疏点云编码为单一CLIP向量，OpenScene[38]通过点-像素投影对齐点特征与HR图像特征，本文继承CLIP对齐范式但新增点caption生成能力。
6. **Vision-Language Models (VLMs)**：CLIP[39]、BLIP[26]、xGen-MM[56]构成图像captioner基础，本文仅微调SMAP模块，其余网络冻结，体现了高效的2D-to-3D知识蒸馏思路。

## 局限性与未来方向
1. **点云captioner需额外训练**：Point Captioner的SMAP模块需通过2D-to-3D蒸馏训练，依赖标注数据（虽然仅用可见性mask监督），在缺乏对齐数据时的泛化能力有待验证。
2. **室内场景mIoU下降**：ScanNet上3D-AVS的mIoU低于基于固定词汇的方法，原因是自动生成的细粒度标签映射到20类粗粒度ground truth时产生信息损失，如何优化多粒度映射值得研究。
3. **计算开销**：同时运行图像captioner（xGen-MM）与点云captioner增加了推理时间，尤其在实时场景中可能成为瓶颈。
4. **TPSS指标的局限性**：TPSS仅衡量语义一致性而非分割质量，无法完全替代mIoU等几何精度指标，两者需结合使用。
5. **室外/室内分区策略固定**：几何分区策略（柱坐标扇区vs方形柱）依赖场景先验，对不同分布的点云（如无人机俯视、车载密集扫描）可能需要自适应分区机制。
6. **未探索跨场景词汇迁移**：当前方法为每帧独立生成词汇，未利用场景间的词汇重用或层次化词汇管理，可能存在冗余描述。

## 研究启发与可借鉴点
1. **SMAP模块的可迁移设计**：Sparse Masked Attention Pooling通过掩码分组+GAP Query+MHA的结构，将可变长子集聚合为固定数量特征，可推广至其他需要"多个局部描述"的3D理解任务（如3D实例分割、点云 grounding）。
2. **无标注评估指标的构建思路**：TPSS利用CLIP对齐空间直接度量预测类别与点特征的语义相似度，避免了人工标注依赖，该思路可迁移至任何open-vocabulary/zero-shot 3D任务的评估，尤其是词汇动态变化的场景。
3. **双模态caption互补策略**：图像caption提供丰富语义但受光照限制，点云caption提供几何鲁棒性但语义较粗，两者的融合决策机制（如置信度加权、场景自适应选择）为多模态3D感知提供了实用范式。
4. **2D-to-3D distillation via visible mask**：利用点-像素可见性mask监督3D特征的pooling操作，避免了繁琐的3D标注，该策略适用于任意需要3D对齐的预训练模型微调场景。
5. **与现有OVS框架的解耦集成**：3D-AVS将"词汇生成"与"分割"分离，可无缝接入OpenScene等现有segmenter，这种模块化设计降低了方法部署门槛，为后续研究提供了可扩展的基准平台。

## 关键术语表
**Open-Vocabulary Segmentation (OVS)**：开放词汇分割，利用预训练视觉-语言模型（如CLIP）将点/像素特征与任意文本查询对齐，实现检测训练集中未见类别的分割任务。

**Auto-Vocabulary Segmentation (AVS)**：自动词汇分割，无需人工预定义类别，从输入数据中自动识别并生成场景特定的语义词汇用于分割。

**Sparse Masked Attention Pooling (SMAP)**：稀疏掩码注意力池化模块，通过二进制掩码将点云划分为多个子集，利用GAP生成query、MHA聚合key/value，输出可控数量的点云区域特征。

**Text-Point Semantic Similarity (TPSS)**：文本-点语义相似度指标，在CLIP对齐空间中计算每个点与其最匹配生成类别的相似度均值，用于无标注评估自动生成词汇的质量。

**CLIP-aligned Encoder**：CLIP对齐编码器，将点云特征蒸馏/对齐到CLIP的视觉-语言共享隐空间中，使3D点特征能与文本嵌入直接比较。

**Point Captioner**：点云caption生成器，基于LiDAR几何信息直接生成点云场景描述文本的模块，颜色无关使其在夜间/恶劣天气下优于图像captioner。

**xGen-MM (BLIP-3)**：Meta发布的开源多模态大模型系列，本文用作图像captioner，具有灵活的架构和增强的语义覆盖能力。

**LAVE (LLM-based Auto-Vocabulary Evaluator)**：基于大语言模型的自动词汇评估器，将开放词汇映射到固定数据集类别，使mIoU等传统指标可用于AVS任务评估。

## 可复现要素
- **数据集**：nuScenes（公开）、ScanNet（公开）、ScanNet200（公开），论文使用了nuScenes的LiDAR分段标注（16类）和ScanNet/ScanNet200的室内标注。
- **代码**：论文未明确声明代码开源状态（正文及致谢未提及GitHub链接），需关注后续发布。
- **权重**：Point Captioner的SMAP模块需训练，其余编码器（CLIP text encoder、高分辨率图像编码器、CLIP对齐点编码器）均为冻结预训练权重。
- **关键超参**：室外扇区数 $T$、图像captioner使用beam search、 Caption2Tag使用spaCy词性解析与WordNet验证、评估映射使用GPT-4o（main experiments）或LAVE/SBERT（supplementary）。
- **实现细节**：点-像素映射 $\varGamma$、点云聚合0.5秒间隔、CLIP对齐点编码器来源（参考OpenScene/LidarCLIP），详见supplementary material。
