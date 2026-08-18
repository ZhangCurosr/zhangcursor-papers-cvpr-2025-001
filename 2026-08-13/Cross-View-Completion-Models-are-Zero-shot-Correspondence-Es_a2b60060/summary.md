---
title: "Cross-View-Completion-Models-are-Zero-shot-Correspondence-Es"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/An_Cross-View_Completion_Models_are_Zero-shot_Correspondence_Estimators_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:58:53"
field: "自监督几何视觉表示学习"
keywords: ["cross-view completion", "zero-shot correspondence", "dense matching", "self-supervised learning", "cross-attention", "multi-frame depth estimation", "geometric vision"]
innovations: ["揭示CVC模型的cross-attention map是有效的零样本对应估计器", "提出ZeroCo通过reciprocity融合双向cross-attention实现SOTA零样本密集匹配", "在冻结cross-attention基础上添加轻量head，在有监督几何匹配与多帧深度估计上超越使用大规模数据的基线"]
benchmarks: ["HPatches", "ETH3D", "KITTI Eigen split", "Cityscapes (dynamic objects)"]
---

# 论文速读：Cross-View Completion Models are Zero-shot Correspondence Estimators

## 一句话总结
论文发现**Cross-View Completion（CVC）模型中的交叉注意力图（cross-attention map）本身就是一种强大的零样本密集对应估计器**，无需任何对应标注即可达到SOTA性能；在此基础上提出ZeroCo框架，并进一步设计轻量学习模块在几何匹配与多帧深度估计任务上实现有监督SOTA。

## 研究问题与动机
- **CVC的有效性机制尚未被充分理解**：CroCo-v2等CVC预训练已在多个几何任务（光流、立体深度、3D重建）上取得优异效果，但业界普遍直接复用其encoder/decoder特征，而忽略了cross-attention层本身可能蕴含更丰富的对应信息。
- **已有工作未充分利用CVC的几何潜力**：DUSt3R、MASt3R等下游方法仅使用decoder特征，未能挖掘cross-attention map中隐含的精准对应关系。
- **零样本对应估计缺乏高效且准确的方案**：传统自监督对应方法依赖大量合成数据或光度损失，而Diffusion-based方法（如DIFT）在极端几何形变下性能显著下降。
- **多帧深度估计依赖极线几何的局限性**：现有方法需精确相机位姿，对动态物体和噪声敏感，需要寻找替代的全局对应表示。

## 核心贡献（创新点）
1. **揭示了CVC与自监督密集对应学习的类比关系**：证明了CVC中的cross-attention层学习目标与传统自监督对应方法中的cost volume构建高度相似，是二者共享几何知识的根本原因。
2. **实验证明cross-attention map在对应编码上显著优于encoder/decoder特征**：通过可视化与定量评估表明，cross-attention map能更精确、锐利地捕获目标像素在源图中的对应位置，而encoder/decoder特征的相关性图则较为模糊。
3. **提出ZeroCo：一种无需训练即可使用的零样本对应估计方法**：通过交换输入对并融合pre-softmax的cross-attention map，强制双向一致性（reciprocity），在HPatches和ETH3D上取得SOTA零样本结果。
4. **设计了轻量学习增强模块（ZeroCo-flow / ZeroCo-depth）**：在冻结的cross-attention map基础上，仅添加cost aggregation与upsampling头，在几何匹配和多帧深度估计任务上均超越使用大量预训练数据的基线方法（如DUSt3R、MASt3R）。

## 方法详解

**基础架构**：基于CroCo-v2预训练模型，encoder为ViT-Large（24层），decoder为12层，每层含self-attention和cross-attention。

**Cross-attention map的计算**：在decoder的第 $l$ 层，从target特征 $D_t$ 和source特征 $D_s$ 中提取query $D_t^{l,Q}$ 和key $D_s^{l,K}$，计算pre-softmax的相似度矩阵：
$$C^l(i,j) = D_t^{l,Q}(i) \cdot D_s^{l,K}(j) / \sqrt{d}$$
经softmax后得到 $C_{\text{att}}^l$，用于将source特征warp到target空间完成图像重建。

**ZeroCo零样本对应（核心公式7）**：将原始输入对 $(I_t, I_s)$ 和交换对 $(I_s, I_t)$ 分别送入decoder，得到两组pre-softmax cross-attention map $C^l$ 和 $C_{\text{swap}}^l$，融合为最终cost volume：
$$C' = \frac{1}{L}\sum_l C^l + \left(\frac{1}{L}\sum_l C_{\text{swap}}^l\right)^T$$
最终流场 $F = \text{softargmax}(C')$，warp source至target进行评估。

**ZeroCo-flow（学习式几何匹配）**：冻结cross-attention map，在其上叠加cost aggregation模块 $\mathcal{T}_c$ 和resolution upsampling模块 $\mathcal{U}$，经reciprocity融合后由softargmax输出flow，使用标准对应回归损失训练。

**ZeroCo-depth（多帧深度估计）**：用cross-attention map替代传统的极线cost volume，结合特征描述子通过聚合模块 $\mathcal{T}_d$ 得到细化cost volume $C'$，再经DPT head输出深度图 $F_{\text{depth}}$，使用重投影与平滑损失训练。此设计无需显式相机位姿。

## 实验与结果

**数据集**：HPatches、ETH3D（几何匹配）；KITTI Eigen split、Cityscapes（深度估计）。

**零样本几何匹配**：
- HPatches-240：ZeroCo AEPE = **5.07**（场景I）至13.26（场景V），平均 **9.41**，大幅超越DIFTSD（26.14）、SD-DINO（29.19）及CroCo Encoder（47.52）、Decoder（44.63）。
- HPatches-Original：ZeroCo平均 **35.39**，对比CroCo Enc.+Dec.（153.68）。
- ETH3D：ZeroCo平均 **12.72**（rate=3~15），显著优于所有对比方法。

**学习式几何匹配（Table 3）**：
- HPatches-Original：ZeroCo-flow AEPE平均 **13.61**，超越DMP（30.64）、PDCNet+（18.91）、DiffMatch（18.84），接近GLU-Net-GOCor（20.16）。
- ETH3D：ZeroCo-flow平均 **2.88**，仅次于MASt3R（2.62），但训练数据量远少于MASt3R（MIX₁₄）。

**多帧深度估计（Table 4，KITTI Eigen）**：
- ZeroCo-depth AbsRel = **0.090**，超越ManyDepth（0.098）、DynamicDepth（0.096）、DualRefine（0.087仅用1帧）。

**动态物体深度（Table 5，Cityscapes）**：
- ZeroCo-depth AbsRel = **0.127**，优于DynamicDepth（0.143）和ManyDepth（0.169）。

**噪声鲁棒性（Table 6，Robodeth协议）**：
- 前后帧均有噪声时，ZeroCo-depth mDEE = **0.161**，mRR = **0.920**，显著优于ManyDepth（0.262/0.819）和DualRefine（0.265/0.805）。

**Ablation（Table 7）**：Reciprocity提升约0.6 AEPE；Dense zoom-in进一步提升约1.0 AEPE；归一化策略在reciprocity启用时反而有害。

## 相关工作脉络
- **CroCo / CroCo-v2**（Weinzaepfel et al.）：CVC预训练的开创性工作，本文在此基础上揭示cross-attention map的对应价值，而前作仅利用decoder特征。
- **DUSt3R / MASt3R**（Leroy et al.）：基于CVC的3D视觉方法，使用decoder特征进行几何预测；本文证明cross-attention map是更优的对应表示。
- **DIFT**（Tang et al.）：基于diffusion模型的零样本匹配；在极端几何形变下性能大幅下降，ZeroCo在此类场景显著更优。
- **SD-DINO**（Zhang et al.）：结合SD与DINO的零样本对应方法；ZeroCo在HPatches和ETH3D上全面超越。
- **GLU-Net / PDC-Net+ / DiffMatch**：有监督密集几何匹配的有代表性方法；ZeroCo-flow以极小的额外参数和学习数据量达到可比甚至更优性能。
- **ManyDepth / DualRefine / DynamicDepth**：多帧自监督深度估计；ZeroCo-depth无需极线几何约束，在动态和噪声场景下鲁棒性更强。

## 局限性与未来方向
- **Cross-attention map分辨率受限于decoder输出**：较低分辨率的cost volume影响精细匹配，虽通过upsampling缓解但仍有提升空间。
- **仅测试了CroCo-v2一个CVC模型**：对其他CVC变体（如DUSt3R架构中的cross-attention）是否同样适用尚需验证。
- **多帧深度估计仅测试了2帧**：扩展到更多帧的几何一致性和效率未充分探索。
- **零样本推理的计算开销**：需要前向传播两次（原始和交换输入）以获得双向一致性，推理耗时增加。

## 研究启发与可借鉴点
1. **Cross-attention map作为即用型cost volume**：对于任何基于Transformer的跨视图模型（不限于CVC），其cross-attention层天然编码了像素级对应关系，可直接复用而无需额外设计匹配网络。
2. **Reciprocity的简洁实现**：通过交换输入对并在pre-softmax空间融合（而非softmax之后），以极低代价实现双向一致性，这一技巧可迁移至其他视觉对应任务。
3. **冻结预训练骨干+轻量head的设计范式**：ZeroCo-flow/depth证明在高质量预训练representation上附加极简可学习模块即可达到SOTA，对资源受限场景极具参考价值。
4. **CVC与对应学习的理论桥梁**：本文建立了CVC pretraining与自监督dense correspondence之间的形式化类比，为后续研究提供了新的分析视角，可启发更多"任务间知识迁移"的工作。
5. **动态场景下的鲁棒性优势**：ZeroCo-depth无需极线约束即可处理动态物体和噪声，为自动驾驶等实际场景的深度估计提供了更实用的方案。

## 关键术语表
- **Cross-View Completion (CVC)**：一种自监督预训练任务，将MASKED图像建模扩展至两视图，用未mask的源图像重建mask的目标图像，学习跨视图几何一致性。
- **Cross-attention map**：Transformer decoder中cross-attention层的注意力权重矩阵，在本工作中被证明编码了target与source图像间像素级的对应关系。
- **ZeroCo**：本文提出的零样本对应估计方法，通过fusion双向cross-attention map实现reciprocity约束，无需任何对应标注。
- **Cost volume**：描述两幅图像所有像素对之间相似度/匹配得分的高维张量，是密集对应估计的核心中间表示。
- **Reciprocity**：对应关系的双向一致性约束，即"A中点p对应B中点q"等价于"B中点q对应A中点p"。
- **Softargmax**：对cost volume沿某一维度做加权平均操作，将离散匹配分布转化为连续的亚像素级flow/深度估计。
- **Dense correspondence**：对图像中每一个像素估计其在另一视图中对应位置的密集匹配任务，包括光流估计和立体匹配等。

## 可复现要素
- **CroCo-v2预训练权重**：论文使用公开可用的CroCo-v2权重（ViT-Large encoder + 12层decoder），来源为CroCo官方仓库。
- **数据集**：HPatches、ETH3D、KITTI Eigen split、Cityscapes均为公开数据集。
- **代码开源声明**：论文未明确声明代码是否开源（CVPR 2025常见做法为补充材料中提供），建议关注论文附带的project page。
- **关键超参**：decoder层数 $L=12$，encoder为ViT-Large（24层），cross-attention channel $d$ 未明确给出具体数值；训练数据为ImageNet-21K（CroCo-v2默认）。
- **损失函数**：CVC预训练使用图像重建损失（recon）；ZeroCo-flow使用对应回归损失；ZeroCo-depth使用重投影损失+平滑损失。
