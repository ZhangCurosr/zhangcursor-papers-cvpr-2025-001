---
title: "DKDM-Data-Free-Knowledge-Distillation-for-Diffusion-Models-w"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Xiang_DKDM_Data-Free_Knowledge_Distillation_for_Diffusion_Models_with_Any_Architecture_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:59:48"
field: "生成模型与模型压缩"
keywords: ["扩散模型", "知识蒸馏", "数据无关训练", "生成模型", "模型压缩", "DKDM"]
innovations: ["提出数据无关的扩散模型蒸馏新范式(DKDM)，无需访问真实数据集即可训练任意架构的扩散模型", "设计DKDM目标函数，用教师逆向分布替代真实扩散后验和先验", "提出动态迭代蒸馏方法，将知识形式定义为时间域噪声样本，效率提升至O(b)"]
benchmarks: ["CIFAR10", "CelebA 64x64", "ImageNet 32x32", "CelebA-HQ 256x256", "FFHQ 256x256"]
---

# 论文速读：DKDM-Data-Free-Knowledge-Distillation-for-Diffusion-Models-w

## 一句话总结
本文提出了**数据无关的知识蒸馏（Data-Free Knowledge Distillation, DKDM）**新范式，利用预训练扩散模型作为唯一数据源，无需访问任何真实数据集即可训练具有任意架构的新型扩散模型。该方法在多个数据集上实现了与全量数据训练相当甚至更优的生成质量。

---

## 研究问题与动机

1. **数据负担过重**：主流扩散模型训练依赖海量高质量数据，例如 Stable Diffusion 需要数十亿图像-文本对，数据采集、存储和隐私成本极高（见表1，ImageNet 相关模型训练数据达百亿级）。

2. **现有蒸馏方法受限于数据访问**：传统知识蒸馏（包括扩散模型的KD加速方法）通常仍需访问原始训练数据集，或仅能用于模型压缩/采样步数缩减，无法实现完全数据无关的训练。

3. **架构灵活性不足**：已有扩散模型蒸馏工作（如 Xie et al. [58]、BOOT [13]）通常要求学生模型初始化时继承教师模型的架构和权重，限制了学生模型的架构选择自由。

4. **直接生成合成数据不现实**：若先用教师模型生成与原始数据集同等规模的高质量合成样本再训练学生模型，将消耗巨大的计算和时间成本（生成数十亿图像不切实际），且存储开销同样巨大。

---

## 核心贡献（创新点）

1. **提出DKDM新范式**：首次系统性地探索"以预训练扩散模型为数据源、完全无需访问真实数据集"的扩散模型训练场景，突破了数据依赖的限制。

2. **设计DKDM目标函数**：通过用教师的逆向后验分布 $p_{\theta_T}(x^{t-1}|x^t)$ 替代真实扩散后验 $q(x^{t-1}|x^t, x^0)$，并借助教师模型的生成能力构造替代扩散先验 $\hat{x}^t$，使优化目标脱离对真实数据 $x^0$ 的依赖。

3. **提出动态迭代蒸馏（Dynamic Iterative Distillation）**：创新性地将知识形式定义为**时间域噪声样本**而非最终生成图像，通过单步去噪迭代和shuffle denoise机制，将批次构建的时间复杂度从 $\mathcal{O}(Tb)$ 降至 $\mathcal{O}(b)$，大幅提升了蒸馏效率。

4. **实现跨架构知识蒸馏**：方法支持CNN到ViT、ViT到CNN等任意架构间的蒸馏，突破了已有工作对学生架构的限制，并在CIFAR10上验证了CNN架构更适合做压缩扩散模型。

---

## 方法详解

### 3.2 DKDM目标函数（DKDM Objective）

**核心思想**：将标准DDPM目标中的扩散后验 $q(x^{t-1}|x^t, x^0)$ 替换为教师的逆向分布 $p_{\theta_T}(x^{t-1}|x^t)$，将扩散先验 $x^t \sim q(x^t|x^0)$ 替换为教师生成的样本 $\hat{x}^t = G_{\theta_T}(T-t)$。

**损失函数**：
$$L_{\text{DKDM}} = L'_{\text{simple}} + \lambda L'_{\text{vlb}}$$

其中：
- $L'_{\text{simple}} = \mathbb{E}_{\hat{x}^t, t}[\|\epsilon_{\theta_T}(\hat{x}^t, t) - \epsilon_{\theta_S}(\hat{x}^t, t)\|^2]$：引导学生预测噪声与教师一致
- $L'_{\text{vlb}} = \mathbb{E}_{\hat{x}^t, t}[D_{KL}(p_{\theta_T}(\hat{x}^{t-1}|\hat{x}^t) \| p_{\theta_S}(\hat{x}^{t-1}|\hat{x}^t))]$：引导学生学习教师的后验分布

**关键推导**：由于教师模型已在真实数据上充分训练，其分布 $\mathcal{D}'$ 与真实分布 $\mathcal{D}$ 高度接近，因此学生模型学习匹配教师的逆向过程，等价于间接学习真实数据的分布，无需接触真实数据 $x^0$。

### 3.3 高效知识收集（Dynamic Iterative Distillation）

**问题**：直接按DKDM目标计算需通过教师模型进行 $T-t$ 步去噪才能得到 $\hat{x}^t$，批次构建的时间复杂度为 $\mathcal{O}(Tb)$，训练极慢。

**三步渐进改进**：

1. **迭代蒸馏（Iterative Distillation）**：每轮训练仅执行单步去噪 $g_{\theta_T}(x^t, t)$，而非多步去噪。学生从教师每一去噪步骤的输出中学习，形成源源不断的数据流。

2. **Shuffle迭代蒸馏（Shuffled Iterative Distillation）**：初始批次的噪声级别 $t_i$ 统一，导致训练不稳定。引入shuffle denoise操作，对每个样本随机去噪 $t_i$ 步（$t_i \sim U[1,T]$），使批次内噪声级别服从均匀分布。

3. **动态迭代蒸馏（Dynamic Iteritive Distillation）**：为解决批次中样本配对缺乏随机性的问题（与标准训练iid假设不符），构建扩大批次集合 $\hat{B}^+_j$，大小为 $| \hat{B}^+_j | = \rho T | \hat{B}^s_j |$（$\rho$为缩放因子），从中随机采样子集进行优化，一步去噪结果回写更新扩大集。时间复杂度降至 $\mathcal{O}(b)$。

**最终损失**（Algorithm 1）：
$$L^{\star}_{\text{DKDM}} = L^{\star}_{\text{simple}} + \lambda L^{\star}_{\text{vlb}}$$
其中期望在 $(\hat{x}^t, t) \sim \hat{B}^+$ 上计算。

---

## 实验与结果

### 实验设置
- **Pixel空间**：CIFAR10 32×32、CelebA 64×64、ImageNet 32×32，教师模型均为CNN架构
- **Latent空间**：CelebA-HQ 256×256、FFHQ 256×256，教师为DDPM/LDM架构
- **评估指标**：FID↓、sFID↓、IS↑，生成50K样本，像素空间50步Improved DDPM采样，潜空间200步DDIM采样

### 主要结果

**Pixel空间（表2）**：
- CIFAR10：动态迭代蒸馏 FID=9.56，优于Data-Free Training（FID=12.06）和Data-Limited 5%（FID=10.91），IS=8.60
- CelebA：FID=7.07，优于所有baseline（Data-Free: 10.66，Data-Limited 5%: 9.64）
- ImageNet：FID=11.33，sFID=4.80，显著优于Data-Free（FID=13.20, sFID=9.56）

**Latent空间（表3）**：
- CelebA-HQ：FID=8.69，接近Data-Based Training（FID=9.09），远超Data-Free（FID=15.36）
- FFHQ：FID=11.53，同样接近Data-Based（FID=8.91），显著优于Data-Free（FID=16.32）

**跨架构蒸馏（表5）**：
- CNN教师→ViT学生：FID=17.86 vs Data-Free的63.15
- ViT教师→CNN学生：FID=13.17 vs Data-Free的44.62
- ViT教师→ViT学生：FID=17.86
- **结论**：CNN更适合作为压缩后的扩散模型

**亮点发现**：在CIFAR10的IS指标和CelebA-HQ的FID指标上，DKDM方法**超越了使用全量数据进行训练的基线**，表明从预训练教师蒸馏可能比从头训练更高效。

---

## 相关工作脉络

1. **扩散模型蒸馏加速**（如 Salimans & Ho [46]、Sauer et al. [49,50]、Meng et al. [30]）：聚焦减少采样步数，仍需原始数据训练，与本文数据无关目标不同。

2. **扩散模型压缩**（如 Yang et al. [62]、Zhang et al. [66]）：通过剪枝+蒸馏缩小模型规模，依赖真实数据集，学生架构与教师一致或更小的限制仍存在。

3. **BOOT方法**（Gu et al. [13]）：同为数据无关蒸馏，但保留教师模型的完整结构和权重，学生架构灵活性受限；本文突破此限制支持任意架构。

4. **传统数据无关蒸馏**（如 Dreaming to Distill [63]、DFKD系列[5,6,10,11,29,31,33,64,65]）：主要针对分类网络，通过优化噪声生成合成数据蒸馏；本文面向扩散模型本身的知识传递，知识形式为时间域中间表示而非最终图像。

5. **合成数据增强**（如 Azizi et al. [1]、Nguyen et al. [34]、Wu et al. [57]）：用扩散模型生成合成数据提升下游任务性能；本文直接以教师模型为数据源训练新扩散模型，无需先生成再训练的中间环节。

---

## 局限性与未来方向

1. **当前仅在图像生成任务上验证**：方法的有效性和泛化能力有待在视频生成、音频生成、3D生成等更多模态上检验。

2. **高分辨率/大规模场景的扩展性**：论文实验主要在CIFAR、CelebA、ImageNet 32×32等中等规模数据集上进行，对于百亿级数据（如Stable Diffusion训练数据）的适用性和效率优势尚待验证。

3. **超参数敏感性**：缩放因子 $\rho$ 存在最优范围（如图4b所示），过高或过低均影响性能，需要针对不同场景进行调优。

4. **教师-学生架构差异的边界**：虽支持跨架构蒸馏，但极端架构差异下（如Transformer vs U-Net）的效果和效率如何，仍需进一步研究。

5. **缺乏文本条件蒸馏实验**：论文未涉及带文本条件的扩散模型（如Stable Diffusion）的蒸馏，这是工业界更关心的场景。

---

## 研究启发与可借鉴点

1. **知识形式的重新定义**：将知识从"最终生成样本"转向"生成过程中的时间域中间表示"，为其他生成模型的知识蒸馏提供了新思路——不必生成完整数据，可直接传递生成过程的状态。

2. **动态迭代与shuffle机制的设计**：通过单步去噪+shuffle denoise+动态批次管理，解决了数据无关蒸馏的效率瓶颈。这种"在线构造+随机采样+迭代更新"的策略可迁移到其他需要避免重复采样的蒸馏场景。

3. **教师分布替代真实分布的理论合理性**：利用"已训练好的教师分布近似真实数据分布"这一假设来消除对真实数据的依赖，为类似的数据瓶颈问题（如联邦学习、隐私保护训练）提供了可行的替代方案。

4. **跨架构蒸馏的实验范式**：证明了不依赖权重初始化的架构自由蒸馏是可行的，这对团队在模型压缩和迁移学习中可能启发的设计方向是：优先关注目标函数的适配性而非架构一致性。

---

## 关键术语表

- **DKDM (Data-Free Knowledge Distillation for Diffusion Models)**：本文提出的新范式，利用预训练扩散模型作为数据源，在无真实数据集条件下训练新扩散模型。
- **DKDM Objective**：专为数据无关蒸馏设计的目标函数，用教师逆向分布替代真实扩散后验和先验，使优化脱离对原始数据的依赖。
- **Dynamic Iterative Distillation**：核心训练方法，通过shuffled denoise构建扩大批次集，随机采样子集进行蒸馏并实时更新，时间复杂度仅为 $\mathcal{O}(b)$。
- **Shuffle Denoise**：对批次内样本随机施加不同步数的去噪操作，使噪声级别服从均匀分布，增强批次多样性。
- **sFID (Structure-aware FID)**：改进的FID度量，考虑了生成样本的结构相似性，比传统FID更能反映生成质量。
- **$L'_{\text{simple}}$**：DKDM中的噪声预测损失，引导学生预测与教师一致的噪声。
- **$L'_{\text{vlb}}$**：DKDM中的变分下界损失，引导学生学习教师的后验分布。
- **Teacher-generated prior $\hat{x}^t$**：通过教师模型从纯噪声反向生成的中间表示，替代真实数据分布中的扩散先验。

---

## 可复现要素

- **代码开源**：是，GitHub仓库 https://github.com/qianlong0502/DKDM
- **数据集**：CIFAR10、CelebA 64×64、ImageNet 32×32、CelebA-HQ 256×256、FFHQ 256×256（均为公开数据集）
- **教师模型**：论文引用Ning et al. [37]的预训练模型配置，具体权重来源见Section 4.1；latent space模型基于Rombach et al. [44]的配置
- **关键超参数**：
  - $\lambda$（$L_{\text{vlb}}$权重）：论文未明确给出具体数值
  - $\rho$（批次放大因子）：消融实验中测试了不同值（图4b），最佳值约0.4附近
  - 训练迭代次数：200K（消融实验）
  - 采样步数：像素空间50步Improved DDPM，潜空间200步DDIM
  - 生成样本数：50K（用于指标计算）
- **评估工具**：ADM TensorFlow evaluation suite

---
