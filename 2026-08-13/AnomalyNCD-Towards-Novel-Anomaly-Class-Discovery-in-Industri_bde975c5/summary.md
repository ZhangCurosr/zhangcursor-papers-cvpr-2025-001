---
title: "AnomalyNCD-Towards-Novel-Anomaly-Class-Discovery-in-Industri"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Huang_AnomalyNCD_Towards_Novel_Anomaly_Class_Discovery_in_Industrial_Scenarios_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:55:25"
field: "工业视觉异常检测"
keywords: ["工业异常检测", "多类异常分类", "新型类发现", "掩码引导注意力", "主元二值化", "自监督学习"]
innovations: ["提出MEBin主元二值化，通过多阈值稳定性分析提取鲁棒的异常主区域", "设计MGViT掩码引导ViT注意力，强制[CLS] token聚焦局部弱语义异常", "结合伪标签修正与面积加权区域合并，实现兼容任意AD方法的端到端异常分类"]
benchmarks: ["MVTec AD", "MTD", "AeBAD-S"]
---

# 论文速读：AnomalyNCD: Towards Novel Anomaly Class Discovery in Industrial Scenarios

## 一句话总结
本文提出AnomalyNCD，一种面向工业场景的新型异常类发现多类异常分类网络。通过主元二值化（MEBin）提取稳定的异常主区域，并结合掩码引导的ViT表示学习，解决工业异常"不显著"和"弱语义"两大挑战，与各类异常检测方法兼容，在MVTec AD和MTD数据集上显著超越现有SOTA方法。

## 研究问题与动机
1. **下游分类需求**：工业异常检测虽已成熟，但仅定位异常，缺乏细粒度异常类别识别（如裂纹、磨蚀等），难以支撑后续差异化处置策略。
2. **先验知识匮乏限制聚类方法**：现有异常聚类方法依赖冻结模型的patch特征聚合，无法学习异常专属判别特征，面对形状/外观/位置多变的同质异常时表现受限。
3. **非显著异常（Non-prominence）**：工业图像中异常区域通常不居中、占比小，背景占据主导，网络难以提取有意义的异常语义。
4. **弱语义异常（Low-semantics）**：工业异常缺乏丰富语义，预训练ViT的[CLS] token倾向于关注背景主体（如金属网格），而非局部细微缺陷。

## 核心贡献（创新点）
1. **首个基于自监督的新型异常类发现方法AnomalyNCD**：与已有工作（监督/无监督分类网络）的本质区别——无需异常标注，仅依靠检测器生成的异常图即可自动发现未见异常类别并分组。
2. **主元二值化（MEBin）稳定提取异常主区域**：与Otsu/固定阈值等方法相比，通过多阈值探索找到稳定的主元结构，显著降低误检/漏检对后续学习的干扰，且兼容任意AD方法。
3. **掩码引导表示学习（MGViT）**：通过在ViT最后几层的[CLS] token上注入掩码注意力，强制模型聚焦局部异常区域，与DINO预训练ViT自然倾向关注前景物体的行为形成本质区分。
4. **伪标签修正（PLC）+ 面积加权区域合并策略**：分别解决异常检测误检导致的伪标签污染和推理时子图像分类投票失真问题，确保端到端鲁棒性。

## 方法详解
**整体流程**：输入无标签工业图像 → MEBin提取主元异常掩码 → 裁剪异常为中心的子图像 → MGViT学习判别特征并进行类别预测 → 区域合并得到图像级分类。

1. **主元二值化（MEBin）**：
   - 给定AD方法的异常概率图$A_i \in [0,1]^{H \times W}$，定义阈值探索范围$[\mathbf{s}_{\min}, \mathbf{s}_{\max}]$，其中$\mathbf{s}_{\min} = \min_i(\max(A_i))$，$\mathbf{s}_{\max}=1$。
   - 均匀采样$\mathcal{T}$个阈值$\epsilon_j$，二值化得$M_i^j = \mathbb{1}[A_i > \epsilon_j]$，并施加腐蚀操作去除碎片。
   - 统计各阈值下连通分量数$\delta_i^j$的众数$\bar{\delta}_i$，取最长连续$\bar{\delta}_i$分量阈值区间，取其最小阈值作为最终分割阈值，生成二值掩码$\mathbf{M}_i^{\mathbf{u}}$。
   - 以最小外接正方形裁剪各主元区域，得到子图像对$\{(x_{i,k}, m_{i,k})\}$。

2. **掩码引导Vision Transformer（MGViT）**：
   - 将掩码resize到$\sqrt{N} \times \sqrt{N}$后展平为$\mathcal{M} \in \mathbb{R}^{N \times 1}$，在首部补1以对齐token数。
   - 仅在ViT最后$L_m$层的[CLS] token的self-attention中注入掩码：
     $$Attn = \text{softmax}(\text{concat}(\mathbf{Q}_{l-1}^{\text{cls}}\mathbf{K}_{l-1}^\top + \overline{\mathcal{M}}, \mathbf{Q}_{l-1}^{\text{patch}}\mathbf{K}_{l-1}^\top))\mathbf{V}_{l-1}$$
     其中$\overline{\mathcal{M}}(i) = 0$ if $M(i)>0.5$ else $-\infty$，将非异常区域的注意力权重压制为0。
   - 实验表明仅作用于[CLS] token效果最佳，保留patch token的上下文感知能力。

3. **训练策略**：
   - 采用DINO-style teacher-student双分支架构，共享MGViT和分类头，不同softmax温度。
   - 有标签子图像使用GT one-hot标签；无标签子图像由教师网络以尖锐温度$\tau_t$生成伪标签（已知类别位置置0），学生网络以平滑温度$\tau_s$学习。
   - 总损失函数：
     $$\mathcal{L} = \lambda(\mathcal{L}_{\text{rep}}^1 + \mathcal{L}_{\text{cls}}^1) + (1-\lambda)(\mathcal{L}_{\text{rep}} + \mathcal{L}_{\text{cls}}^{\mathbf{u}} + \mu\mathcal{L}_{\text{reg}}^{\mathbf{u}})$$
     含监督对比学习$\mathcal{L}_{\text{rep}}^1$、有标签分类损失$\mathcal{L}_{\text{cls}}^1$、自监督对比$\mathcal{L}_{\text{rep}}$、伪标签分类损失$\mathcal{L}_{\text{cls}}^{\mathbf{u}}$、mean-entropy正则$\mathcal{L}_{\text{reg}}^{\mathbf{u}}$。
   - **伪标签修正（PLC）**：用异常分数$s_{i,k}$校正伪标签，抑制误检样本的有害监督信号：$\hat{q}_{i,k} \leftarrow w_{i,k}\mathbf{e} + (1-w_{i,k})\hat{q}_{i,k}$，其中$w_{i,k} = \max(0.5 - s_{i,k}, 0)$。

4. **区域合并策略（推理阶段）**：
   - 基于异常区域面积加权投票，避免小面积误检区域主导最终决策：
     $$\alpha_{i,k}^{\mathbf{u}} = \frac{\exp(\sqrt{a_{i,k}^{\mathbf{u}}}/\tau_\alpha)}{\sum_k \exp(\sqrt{a_{i,k}^{\mathbf{u}}}/\tau_\alpha)}$$
     图像级logit $p_i^{\mathbf{u}} = \sum_k \alpha_{i,k}^{\mathbf{u}} p_{i,k}^{\mathbf{u}}$。

## 实验与结果
**数据集**：MVTec AD（15类，每类至少2种异常）、MTD（5种异常类）；默认标签集$\mathcal{D}^1$为AeBAD-S（去除正常类）。

**主要结果（Table 1，仅用无标签图像）**：
- AnomalyNCD + MuSc（零样本AD）vs. 最强基线AC[40]：MVTec AD上 **NMI +8.8%**（0.613）、**ARI +9.5%**（0.526）、**F₁ +10.8%**（0.712）；MTD上 **F₁ +12.8%**（0.509）、**NMI +5.7%**（0.268）、**ARI +10.8%**（0.228）。
- 显著超越NCD方法（SimGCD）：MVTec AD上 **NMI +16.1%**、**ARI +18.0%**、**F₁ +14.3%**。

**半监督设置（Table 2，额外使用有标签正常图）**：
- AnomalyNCD + CPR取得MVTec AD最佳：NMI 0.736 / ARI 0.674 / F₁ 0.805，较AC[40] semi-sup分别提升10.7%、14.9%、9.6%。
- MTD上PatchCore + AnomalyNCD较Uniformaly提升ARI 6.8%、F₁ 0.8%。

**消融结论**：
- MEBin vs. Otsu/固定阈值：自适应阈值显著降低FPR/FNR，F₁提升7.2%（Table 3）。
- MGA仅在[CLS] token上效果最优（Table 4），$L_m=9$层为最佳深度（Table 5）。
- 面积加权合并优于平均/分数加权（Table 6）。
- PLC使正常类Recall提升14.9%（Table 7）。

## 相关工作脉络
1. **异常聚类方法（AC[40], UniFormaly[28]）**：基于冻结模型的特征聚合，无自适应学习能力；AnomalyNCD通过自监督对比学习在异常区域学习判别特征，本质区别在于"学习型"vs"冻结型"表征。
2. **Novel Class Discovery（UNO[15], GCD[45], SimGCD[47], AMEND[3]）**：面向自然图像，假设主体居中且语义丰富；AnomalyNCD首次将NCD引入工业异常场景，通过MEBin+MGViT适配"非居中+弱语义"特性。
3. **二值化方法（Otsu[35], DiffuMask[48]）**：Otsu倾向过检、DiffuMask产生碎片化假阳性；MEBin通过阈值稳定性分析精准提取主元区域，专为异常分类任务设计。
4. **Vision Transformer（ViT[13], DINO[9]）**：预训练ViT的[CLS]天然关注前景物体而非局部缺陷；MGViT通过掩码注意力纠正这一偏差，将自监督视觉表示与异常定位任务对齐。
5. **对比学习方法（IIC[25], GAT-Cluster[33], SCAN[44]）**：直接对整图聚类，未利用异常定位先验；AnomalyNCD先在异常子图上学习判别特征再聚类，精度更高。

## 局限性与未来方向
1. **性能依赖上游AD质量**：AnomalyNCD的效果与所接异常检测器（AUPRO）正相关，极端情况下AD误检过多仍会影响分类。
2. **需预知新类别数量$\mathcal{C}^{\mathbf{u}}$**：遵循vanilla NCD设定，类别数需作为超参输入，对实际未知类别数场景适应性受限。
3. **仅处理单异常区域为主的情况**：MEBin聚焦主元区域，若单张图像存在多个异构异常，处理效果可能下降（附录提及可处理复合异常但未深入）。
4. **未来方向**：扩展至半监督设置（利用少量异常标注）、端到端联合优化AD与分类、处理复杂复合缺陷场景。

## 研究启发与可借鉴点
1. **主元二值化的阈值稳定性思想**：通过多阈值遍历寻找"稳定出现的连通结构"，可迁移至其他需要鲁棒阈值的分割/检测任务（如遥感目标检测、医学影像病灶提取）。
2. **掩码引导[CLS]注意力的设计**：用二值掩码直接修正ViT attention bias的思路简洁高效，可推广至其他"需要强制模型聚焦局部区域"的自监督视觉学习任务。
3. **伪标签修正（PLC）策略**：结合AD异常分数动态校正伪标签的方法，可复用于任何依赖伪标签的半监督/开放集学习框架，尤其适合高误检率场景。
4. **区域合并的面积加权策略**：针对"误检区域面积小但分数高"的工业特性设计的投票机制，为多实例学习（MIL）中的实例加权提供了新思路。
5. **与任意AD方法解耦兼容**：AnomalyNCD作为通用分类模块可即插即用，为构建模块化异常分析流水线提供了参考范式。

## 关键术语表
**Novel Class Discovery (NCD)**：一类半监督学习设定，模型在有标签基础类和无标签新类混合数据上学习，目标是发现并分类未见过的类别。
**Main Element Binarization (MEBin)**：通过多阈值遍历寻找连通分量数稳定的阈值区间，从而从异常概率图中提取最可靠的主元异常区域的二值化方法。
**Mask-Guided Attention (MGA)**：在ViT自注意力中将二值掩码转化为注意力偏置项加到[CLS] token上，强制模型聚焦掩码覆盖的异常区域。
**Pseudo Label Correction (PLC)**：利用AD异常分数动态调整无标签子图像的伪标签，降低误检区域对分类训练的负面影响。
**Region Merging**：推理时根据子图像异常区域面积加权聚合各子图像的预测 logits，得到鲁棒的图像级分类结果。
**AUPRO (Area Under PRO)**：按固定假正率（FPR）评估异常定位分割质量的指标，衡量检测器与真实mask的像素级重叠程度。
**Self-supervised Contrastive Learning**：通过对同一图像的增强视图学习一致的表征，无需类别标签即可提取判别性特征。
**Mean-Entropy Regularization**：通过最大化预测熵防止模型对无标签数据产生过度自信的虚假预测，促进类别分布均匀化。

## 可复现要素
- **数据集**：MVTec AD（公开）、MTD（公开）、AeBAD-S（公开）
- **代码**：已开源，https://github.com/HUST-SLOW/AnomalyNCD
- **关键超参**：ViT-B/8（DINO预训练），最后9层使用MGA；教师/学生温度$\tau_t < \tau_s$；损失权重$\lambda, \mu$（详见Appendix A）；MEBin阈值采样数$\mathcal{T}$；区域合并温度$\tau_\alpha$（论文未明确给出具体数值，见附录）
- **AD基线方法**：MuSc[30]（零样本）、PatchCore[38]、EfficientAD[4]、RD++[43]、PNI[2]、CPR[29]（均为一类AD方法）
