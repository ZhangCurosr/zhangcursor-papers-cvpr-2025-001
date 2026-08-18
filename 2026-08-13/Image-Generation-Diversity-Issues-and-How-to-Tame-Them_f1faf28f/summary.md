---
title: "Image-Generation-Diversity-Issues-and-How-to-Tame-Them"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Dombrowski_Image_Generation_Diversity_Issues_and_How_to_Tame_Them_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 07:11:28"
field: "生成模型评估与多样性"
keywords: ["扩散模型", "多样性评估", "图像检索", "生成模型", "测量缺口", "DiADM"]
innovations: ["提出IRS图像检索多样性指标，基于Coupon Collector框架可从小样本估计带置信区间的多样性", "揭示常用特征提取器的测量缺口问题并给出校正方法", "提出DiADM通过伪无条件特征解耦多样性与保真度提升无条件扩散模型多样性"]
benchmarks: ["ImageNet-512", "FFHQ", "ChestX-ray14", "CelebV-HQ", "Dynamic"]
---

# 论文速读：Image-Generation-Diversity-Issues-and-How-to-Tame-Them

## 一句话总结
本文提出了一种基于图像检索的多样性评估指标 IRS（Image Retrieval Score），揭示了当前扩散模型在训练后仅能覆盖训练数据约77%的多样性问题，并提出DiADM（Diversity-Aware Diffusion Models）通过伪无条件特征解耦多样性和保真度来提升生成多样性。

## 研究问题与动机
- **多样性问题难以察觉**：生成模型生成的单个样本可能看起来真实，但整体分布覆盖不足；现有指标（如FID、Precision/Recall）无法有效评估多样性。
- **现有指标存在测量缺口（Measurement Gap）**：常用特征提取器在特征空间中对真实图像严重坍缩，导致多样性被低估。
- **多样性与质量的固有冲突**：现有提升多样性的方法（如加噪声扰动）往往以牺牲图像质量为代价，缺乏泛化性。
- **条件生成的模式坍缩**：文本/类别条件扩散模型容易陷入条件模式坍缩，导致生成输出过度均匀，缺乏多样性。

## 核心贡献（创新点）
1. **提出IRS（Image Retrieval Score）**：将多样性评估建模为图像检索问题，无需超参数即可量化模型多样性，并可从少量样本中估计并给出置信区间。与FID等基于距离分布的指标本质不同，IRS直接衡量训练样本被"学习到"的比例。
2. **揭示测量缺口（Measurement Gap）**：首次系统证明所有当前常用特征提取器（Inception、DINOv2、CLIP等）在对真实图像进行检索时远低于理论期望值，说明多样性评估基线本身就存在问题。
3. **提出DiADM（Diversity-Aware Diffusion Models）**：通过伪无条件特征（pseudo-unconditional features）将多样性与保真度解耦，在不牺牲图像质量的前提下显著提升无条件扩散模型的多样性。与添加噪声扰动的方法不同，该方法可同时优化两者。

## 方法详解
- **IRS核心思想（Coupon Collector框架）**：将每个合成图像视为由一个主要训练图像成分构成，多样性等价于被至少一个合成图像"学习"的训练图像比例。
- **图像检索度量**：使用预训练特征提取器$\mathcal{F}$提取特征，对于每个合成图像$x'_t$，找到与其最近邻的真实图像$x_t$，该真实图像被定义为"learned"。IRS计算公式为$IRS_\alpha = N_{learned} / N_{train}$，其中$\alpha = N_{sample} / N_{train}$。
- **统计学估计与置信区间**：利用第二类Stirling数（Stirling's number of the second kind）计算恰好检索到$k$张不同图像的概率，通过MLE估计极限多样性$IRS_{\infty}$及上下置信区间。公式(4)-(8)给出了核心推导。
- **测量缺口校正**：在真实数据上计算$IRS_{\infty, real}$作为参考，调整后的得分$IRS_{\infty, a} = IRS_{\infty, synth} / IRS_{\infty, real}$，使结果具有直接可解释性。
- **DiADM架构**：不使用placeholder标签，而是使用Inception v3预先提取的图像特征作为输入送入扩散模型，实现"伪无条件"生成。Backbone采用XS架构[24]，多样性模块确保充分覆盖训练分布。

## 实验与结果
- **数据集**：ImageNet-512（主要）、FFHQ、ChestX-ray14、CelebV-HQ、Dynamic（医学视频）。
- **特征提取器**：BYOL、CLIP、ConvNeXt、data2vec、DINOv2、Inception、MAE、Random Inception-v3、SwAV。
- **最强多样性结果（Table 2）**：EDM-2-XXL-512达到$IRS_{\infty, a} = 0.77$，为所有SOTA模型最高，但仍远低于1.0。最小模型仅达0.46。
- **DiADM效果（Table 3）**：在固定计算预算下，DiADM相比EDM-2在多个数据集上显著提升IRS：ImageNet-512从0.09→0.15，FFHQ从0.23→1.51（超过真实数据），ChestX-ray14从0.19→1.08，CelebV-HQ从0.18→0.69，Dynamic从0.60→1.04。FID也同步改善。
- **文本到图像性别偏差（Sec 4.4）**：Deepfloyd模型在多数提示下仅达到平衡参考数据集约50%的多样性，显示明显性别偏差。

## 相关工作脉络
- **FID/Precision/Recall**[20,26]：传统生成模型评估指标，但FID需要大量采样且不可解释；Precision/Recall对数据集大小和超参数敏感，且在本文的实验中Recall最多饱和至0.8，无法区分模型差异。
- **Coverage & Density**[32]：依赖train/test集比例，密度专门设计忽略真实异常值（这对多样性评估不利），且缺乏对真实数据集的评估。
- **Vendi Score**[16]：不依赖真实参考集，但牺牲保真度且计算开销极大，不适用于大数据集。
- **噪声扰动方法**[28]：通过添加噪声提升多样性但以图像质量下降为代价，缺乏可解释性。
- **RL多样性引导**[31]：针对特定任务设计奖励函数，泛化性有限。
- **Bias Mitigation**[13,17,37]：专注于偏差缓解而非全面的分布覆盖，方法依赖特定条件类型。

## 局限性与未来方向
- **低样本量下的随机性**：当采样数较少时IRS估计不确定性较大，微小的分数差异无统计意义。
- **特征提取器限制**：未找到能在真实数据上最大化测量多样性的理想特征提取器，未来可探索任务特定的多样性评估器。
- **计算资源受限**：改进多样性的实验受限于固定计算预算（仅用基线1/10的训练时间），未充分探索更大规模的DiADM。
- **条件生成扩展**：当前DiADM仅针对无条件扩散模型，向条件生成的扩展尚待研究。
- **记忆化问题**：增加条件性可能引入新的记忆/过拟合问题，需类似文本条件化的分析。

## 研究启发与可借鉴点
1. **IRS可直接迁移到视频/3D生成评估**：论文已将IRS应用于视频数据（CelebV-HQ、Dynamic），其框架天然支持任何图像/视频生成任务。
2. **测量缺口校正思路可推广到其他度量**：任何依赖特征空间的评估指标都可能存在类似问题，可通过真实数据校准来消除系统偏差。
3. **伪无条件特征解耦多样性与保真度的范式**：DiADM的核心思想——用真实数据特征替代随机标签来同时优化质量和多样性——可迁移到其他生成模型架构（如GAN、VAE）的条件生成改进。
4. **最小样本估计思想**：IRS仅需少量样本即可给出带置信区间的多样性估计，适合快速模型评估和迭代，避免FID所需的大规模采样开销。

## 关键术语表
- **IRS (Image Retrieval Score)**：基于图像检索的多样性评估指标，衡量合成数据能检索到多少比例的训练数据。
- **Measurement Gap**：特征提取器在特征空间中对真实图像发生坍缩，导致多样性被系统性低估的现象。
- **Pseudo-Unconditional**：使用真实图像的预提取特征代替placeholder标签进行无条件生成训练的策略。
- **Coupon Collector Problem**：本文用于建模多样性评估的概率框架，等价于收集全部训练样本的问题。
- **DiADM (Diversity-Aware Diffusion Models)**：本文提出的通过伪无条件特征解耦多样性与保真度的扩散模型改进方法。
- **IRS∞ (极限多样性)**：假设无限采样情况下模型能覆盖的训练数据比例的上限估计。
- **Conditional Mode Collapse**：条件扩散模型因引导信号导致的生成输出过度趋同现象。
- **SwAV**：被论文选为最佳多样性评估特征提取器的自监督学习算法。

## 可复现要素
- **数据集**：ImageNet-512、FFHQ、ChestX-ray14、CelebV-HQ、Dynamic（论文未明确说明各数据集是否公开，ImageNet/FFHQ/ChestX-ray14均为公开数据集）。
- **代码/权重**：Python包已开源，链接：https://github.com/MischaD/beyondfid；EDM-2等预训练模型来自公开来源。
- **关键超参**：采样数50000；计算预算574 A40 GPU小时（仅为基线训练的1/10）；空间分辨率固定为512像素；视频数据限前60帧。
