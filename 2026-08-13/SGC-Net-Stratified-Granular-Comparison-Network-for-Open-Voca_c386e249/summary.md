---
title: "SGC-Net-Stratified-Granular-Comparison-Network-for-Open-Voca"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Lin_SGC-Net_Stratified_Granular_Comparison_Network_for_Open-Vocabulary_HOI_Detection_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 09:40:05"
field: "开放词汇人体交互检测"
keywords: ["open-vocabulary HOI detection", "CLIP", "multimodal learning", "fine-grained recognition", "hierarchical classification"]
innovations: ["提出GSA模块通过高斯分布加权融合CLIP多尺度视觉特征", "设计HGC模块利用LLM生成判别性描述细粒化分类边界", "引入DGW机制和交互分数校准提升检测性能"]
benchmarks: ["HICO-DET", "V-COCO"]
---

# 论文速读：SGC-Net: Stratified Granular Comparison Network for Open-Vocabulary HOI Detection

## 一句话总结
本文针对开放词汇人体交互（OV-HOI）检测中的**特征粒度不足**和**语义相似度混淆**两大挑战，提出了分层粒度对比网络（SGC-Net），通过粒度细化自注意力（GSA）模块融合CLIP多尺度视觉特征、以及层级图文对比（HGC）模块借助LLM生成判别性描述来细粒化分类边界，在HICO-DET和V-COCO数据集上实现了SOTA性能。

---

## 研究问题与动机

1. **特征粒度缺陷（Feature Granularity Deficiency）**：现有OV-HOI方法依赖CLIP最后一层视觉特征进行文本-视觉匹配，但这些高层特征丢失了局部细节（如身体姿态、物体交互方式），难以捕捉细粒度的人体交互语义对齐。

2. **语义相似度混淆（Semantic Similarity Confusion）**：使用大语言模型（LLM）生成的细粒度描述词虽有助于区分相似交互类别，但仅靠静态描述难以精确刻画"人-物交互"三元组中人与物体之间复杂的关联关系，且固定描述无法适应不同交互场景的动态变化。

3. **细粒度语义建模困难**：现有方法（如CMD-SE、GEN-VLKT）通过CLIP文本编码器初始化分类器，但分类器主要依赖类名或固定提示词，对语义相近类别（如"hit"vs"kick"）的区分能力有限。

4. **开放词汇泛化需求**：传统HOI检测方法局限于预定义交互类别，无法泛化到训练集中未见过的物体类别和交互类型，限制了实际应用场景。

---

## 核心贡献（创新点）

1. **粒度细化自注意力模块（GSA）**：提出基于高斯分布的距离加权策略，将CLIP视觉编码器的多尺度特征进行自适应融合，捕获细粒度的人体交互局部细节，同时保留高层语义特征；与现有方法仅用最后一层特征的本质区别在于显式建模了不同网络层级的细粒度信息贡献。

2. **层级图文对比模块（HGC）**：创新性地结合CLIP与LLM，利用LLM为每个HOI类别生成判别性文本描述，并通过层次化聚类构建分类体系，从粗粒度到细粒度逐层比较，实现语义相近类别的边界细化；与传统仅依赖硬标签分类器的本质区别在于引入了动态文本描述增强判别力。

3. **距离加权高斯聚合（DGW）**：提出基于层间相对距离的高斯权重分配机制，灵活控制各层特征的重要性；与简单拼接或平均聚合的本质区别在于自适应地权衡了全局语义与局部细节的贡献比例。

4. **交互分数校准机制**：引入γ参数的交互得分校准策略（$\hat{s_i}' = \hat{s_i} \cdot \hat{c_i}^\gamma$，其中$\gamma > 1$），通过边界框置信度调整交互分数；与现有方法直接输出分类分数的本质区别在于显式建模了检测质量对交互分类的影响。

---

## 方法详解

### 整体架构
SGC-Net基于预训练的CLIP视觉编码器和文本编码器构建，包含两个核心模块：
- **粒度细化自注意力（GSA）模块**：融合多尺度视觉特征
- **层级图文对比（HGC）模块**：利用LLM增强文本描述判别性

### GSA模块设计
将CLIP ViT的$d$个block分为$S$组，通过高斯分布分配权重：

$$\alpha_l^s = \exp\left(-\frac{1}{2}\frac{(d-l)^2}{\sigma^2}\right), \quad l \in [1,d]$$

$$Z = \sum_{s=1}^{S} \alpha_s \left(\sum_{l=1}^{d} \alpha_l^s F_l\right)$$

其中$F_l$为第$l$层的特征，$\alpha_l^s$控制层内权重，$\alpha_s$控制组间权重。早期block关注细粒度局部细节，后期block关注全局语义。

最终交互特征通过交互解码器（Interaction Decoder）生成：
$$X = \text{Dec}(Q, Z)$$

其中$Q$为HOI查询向量，$Z$为聚合后的视觉特征。

### HGC模块设计
**Step 1: 文本描述生成**
利用LLM为每个HOI类别生成判别性描述，模板：
```
What features are useful to distinguish {HOI category} in a photo?
```

**Step 2: 聚类分析**
使用K-Means算法对LLM生成的描述进行分组，识别语义相似的类别簇。

**Step 3: 层次化文本特征构建**
根据簇大小动态调整分组策略，大簇采用summary-based比较，小簇采用direct比较。

**Step 4: 迭代评分机制**
定义迭代更新函数计算层次文本特征：
$$u_i^k = \mathbb{I}\left(p_i^{k+1} > p_i^k + \tau\right)$$
$$r(\pmb{x}, i) = \frac{p_i^1 + \sum_{j=2}^{M_i} p_i^j \prod_{k=1}^{j-1} u_i^k}{1 + \sum_{j=2}^{M_i} \prod_{k=1}^{j-1} u_i^k}$$

最终相似度分数：
$$s(\pmb{x}, i) = (1-\lambda)(p_i^1 + \pmb{t} \cdot \pmb{x}^T) + \lambda(r(\pmb{x}, i))$$

其中$\lambda \in [0,1]$为融合系数。

### 损失函数
训练损失由三部分组成：
$$\mathcal{L} = \lambda_b \sum_{i \in \{h,o\}} \mathcal{L}_b^i + \lambda_{iou} \sum_{i \in \{h,o\}} \mathcal{L}_{iou}^i + \lambda_{cls} \mathcal{L}_{cls}$$

其中$\mathcal{L}_b$为边界框回归损失，$\mathcal{L}_{iou}$为IoU损失，$\mathcal{L}_{cls}$为分类损失。

### 交互分数校准
最终预测分数：
$$\hat{s_i}' = \hat{s_i} \cdot \hat{c_i}^\gamma$$

其中$\gamma > 1$为校准参数。

---

## 实验与结果

### 数据集
- **HICO-DET**：包含77个物体类别、117个交互类别，共约31K张图片
- **V-COCO**：视觉COCO数据集，包含60个物体类别、29个交互类别

### 评估协议
- **Zero-Shot (ZS)**：仅使用物体类别监督训练，测试时检测未见过的交互类型
- **Generalized Zero-Shot (GS)**：同时评估见过的和未见过的类别

### 主要结果

**HICO-DET GS协议**（Table 1）：
| 方法 | Unseen | Seen | Full |
|------|--------|------|------|
| SCL | 19.07 | 30.39 | 28.08 |
| GEN-VLKT | 21.36 | 32.91 | 30.56 |
| OpenCat | 21.46 | 33.86 | 31.38 |
| HOICLIP | 23.48 | 34.47 | 32.26 |
| THID | 15.53 | 24.32 | 22.38 |
| CMD-SE | 16.70 | 23.95 | 22.35 |
| **SGC-Net** | **23.27** | **28.34** | **27.22** |

**HICO-DET ZS协议**（Table 2）：
| 方法 | Non-rare | Rare | Unseen | Full |
|------|----------|------|--------|------|
| CHOID | 10.93 | 6.63 | 2.64 | 6.64 |
| QPIC | 16.95 | 10.84 | 6.21 | 11.12 |
| GEN-VLKT | 20.91 | 10.41 | - | 10.87 |
| MP-HOI | 20.28 | 14.78 | - | 12.61 |
| THID | 17.67 | 12.82 | 10.04 | 13.26 |
| CMD-SE | 21.46 | 14.64 | 10.70 | 15.26 |
| **SGC-Net** | **23.67** | **16.55** | **12.46** | **17.20** |

**V-COCO GS协议**：
- Non-rare: 30.85 mAP
- Rare: 15.21 mAP  
- Unseen: 11.99 mAP

### 关键提升
- 在HICO-DET ZS协议Full指标上达到**17.20 mAP**，超越CMD-SE **1.94 mAP**
- 在HICO-DET GS协议Full指标上达到**27.22 mAP**，超越HOICLIP **5.04 mAP**
- 在V-COCO ZS协议Non-rare类别上达到**30.85 mAP**，超越HOICLIP **6.58 mAP**

---

## 相关工作脉络

1. **CLIP驱动的HOI检测**：HOICLIP、Open-CLIP-HOI等工作利用CLIP的零样本能力进行HOI检测，但依赖单一层级的视觉特征，缺乏细粒度建模。

2. **大语言模型增强的HOI检测**：TALL-HOI、GEN-VLKT等利用LLM生成描述增强分类器，但未解决语义相似类别的混淆问题。

3. **细粒度交互建模**：CMD-SE、Interactiveness Field等工作尝试建模细粒度语义，但仅关注类名层面的区分，未能充分利用LLM生成的丰富描述。

4. **开放词汇目标检测**：GLIP、Grounding DINO等工作将开放词汇检测技术迁移到HOI任务，但缺乏针对人体交互特性的专门设计。

5. **多尺度特征融合**：现有方法多采用简单拼接或平均聚合，本文提出的DGW机制更有效地平衡了不同层级的特征贡献。

---

## 局限性与未来方向

1. **计算开销较大**：LLM描述生成和多尺度特征融合增加了训练和推理的计算成本，可能限制在实际场景中的应用。

2. **依赖预训练模型**：方法强烈依赖CLIP和LLM的质量，在特定领域或低资源语言场景下泛化能力受限。

3. **描述生成的固定性**：当前使用固定模板生成描述，可能无法覆盖所有交互场景的多样性。

4. **未来方向**：
   - 探索更高效的特征融合策略
   - 研究自适应描述生成机制
   - 扩展至视频级别的开放词汇HOI检测

---

## 研究启发与可借鉴点

1. **多尺度特征自适应融合**：DGW机制提供了一种灵活的层间特征融合思路，可迁移到其他需要多尺度语义对齐的视觉任务中。

2. **LLM增强的分类边界细化**：HGC模块利用LLM生成判别性描述的思路，可推广到细粒度图像分类、属性识别等任务。

3. **迭代式层次化决策**：从粗到细的层次化比较策略，为处理长尾分布的开放词汇检测提供了新的视角。

4. **交互分数校准机制**：$\hat{s_i}' = \hat{s_i} \cdot \hat{c_i}^\gamma$的设计简单有效，可应用于其他需要联合优化检测和分类的任务。

---

## 关键术语表

**开放词汇人体交互检测（OV-HOI）**：识别和关联超出预定义类别集的人体-物体交互关系的任务。

**CLIP**：对比语言-图像预训练模型，通过大规模图文对训练实现跨模态语义对齐。

**粒度细化自注意力（GSA）**：通过高斯分布加权融合多尺度视觉特征，捕获细粒度交互细节的模块。

**层级图文对比（HGC）**：利用LLM生成判别性描述并通过层次化聚类构建分类体系的模块。

**距离加权高斯（DGW）**：基于层间相对距离自适应分配权重的特征聚合策略。

**零样本（ZS）协议**：训练阶段不使用目标类别的标注数据，测试时评估模型对未见类别的泛化能力。

**广义零样本（GS）协议**：同时评估模型对见过和未见类别的检测能力。

**交互分数校准**：通过边界框置信度调整交互分类分数的策略（$\hat{s_i}' = \hat{s_i} \cdot \hat{c_i}^\gamma$）。

---

## 可复现要素

- **数据集**：HICO-DET和V-COCO均为公开数据集
- **代码**：论文未提及代码开源情况
- **权重**：使用预训练的ViT-L/14 CLIP模型
- **关键超参数**：
  - 学习率：$10^{-4}$
  - batch size：64（HICO-DET）/ 32（V-COCO）
  - 训练轮数：72（HICO-DET）/ 24（V-COCO）
  - $\lambda_b = 5, \lambda_{iou} = 2, \lambda_{cls} = 5$
  - $\lambda = 0.5$（融合系数）
  - $\gamma > 1$（校准参数）
  - LLM：GPT-3.5

---
