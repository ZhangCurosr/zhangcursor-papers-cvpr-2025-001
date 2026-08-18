---
title: "Annotation-Ambiguity-Aware-Semi-Supervised-Medical-Image-Seg"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Kumari_Annotation_Ambiguity_Aware_Semi-Supervised_Medical_Image_Segmentation_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:54:58"
field: "医学图像分析与半监督学习"
keywords: ["semi-supervised learning", "medical image segmentation", "annotation ambiguity", "pseudo-label generation", "latent distribution learning", "cross-decoder supervision"]
innovations: ["首个结合标注歧义感知与半监督学习的医学图像分割框架", "基于随机剪枝解码器的多样性伪标签生成机制，无需额外网络即可产生多种合理分割", "利用Laplace分布对伪标签后验建模以降低不可靠伪标签的过拟合风险"]
benchmarks: ["LIDC-IDRI", "ISIC 2018", "Probabilistic U-Net", "Pionono", "D-Persona", "Cross Pseudo Supervision"]
---

# 论文速读：Annotation-Ambiguity-Aware-Semi-Supervised-Medical-Image-Seg

## 一句话总结
本文提出了 **AmbiSSL**（Annotation Ambiguity Aware Semi-Supervised Learning），是首个将"标注歧义感知"与"半监督学习"结合用于医学图像分割的框架——仅用少量多标注者标注数据配合大量未标注数据，即可生成多种合理且多样化的分割掩码，降低对大规模标注数据的依赖。

## 研究问题与动机
1. **标注成本高**：医学图像像素级标注耗时耗力，难以获取大规模高质量标注数据集。
2. **单分割结果的局限**：现有方法通常输出单一确定性分割图，但真实临床场景中不同专家对同一图像可能给出不同标注（标注歧义），单一结果无法体现这种不确定性。
3. **既有方法割裂**：半监督学习（SSL）擅长利用未标注数据但只能生成单结果；歧义感知方法能生成多结果但需要多专家标注且无法利用未标注数据。
4. **临床需求迫切**：医生在诊断时需要多种可能的分割方案参考，而非单一"正确"答案。

## 核心贡献（创新点）
1. **首个统一框架**：首次将标注歧义感知与半监督学习结合，同时解决"标注稀缺"和"标注不确定性"两大问题。
2. **多样性伪标签生成（DPG）模块**：通过主干解码器的随机剪枝生成多个变体解码器，采样潜在编码后产生多样化伪标签集——与单纯使用单一模型生成伪标签的本质区别在于"显式引入多样性"。
3. **半监督潜在分布学习（SSSDL）模块**：利用真实标注和伪标签集共同构建公共潜在空间，其中未标注数据采用 Laplace 分布而非高斯分布，避免对不可靠伪标签过度自信——与标准 VAE 式潜在学习的本质区别在于"对伪标签噪声的鲁棒建模"。
4. **交叉解码器监督（CDS）模块**：让一个剪枝解码器的伪标签指导另一个解码器的训练，促进互补特征学习——与 teacher-student 或 ensemble 方法的本质区别在于"对称互教、无需额外网络"。

## 方法详解
**数据设定**：标注集 $D_l = \{(x_i^l, Y_{set}^i)\}$，未标注集 $D_u = \{x_i^u\}$，$Y_{set}^i = \{y_1, y_2, ..., y_a\}$ 为多标注者标签。

**（1）SSSDL 模块**
- 标注数据：先验分布 $D^{prior}(x^l) = \mathcal{N}(\mu_{prior}^l, \sigma_{prior}^l)$，后验分布 $D^{post}(x^l, Y_{set}) = \mathcal{N}(\mu_{post}^l, \sigma_{post}^l)$
- 未标注数据：先验为高斯，后验采用 **Laplace 分布** $\mathcal{G}(\cdot, b)$ 以降低对伪标签的过度置信（KL 散度损失对齐两者）
- 关键公式：$\mathcal{L}_{kl}^l = KL(D^{prior}(x^l) \parallel D^{post}(x^l, Y_{set}))$

**（2）DPG 模块**
- 对主干解码器最后 $n-L+1$ 层做随机剪枝变换：$\tilde{W}_k = M_k \odot W_k + (1-M_k) \odot \lambda(W_k)$
- $\lambda(W_k)$ 以概率 $q_k$ 替换为 Top-$a\%$ 权重（绝对值最大），以概率 $1-q_k$ 保留原权重
- 三个解码器（原始 + 两个剪枝版）分别在梯度关闭状态下生成伪标签集，集成两个伪标签集提升质量

**（3）CDS 模块**
- 剪枝解码器 $\phi$ 和 $\xi$ 用对方生成的伪标签进行交叉监督：$\mathcal{L}_{seg}^{u:\phi} = \mathcal{L}_{Dice}(P_\phi^{post:u}, \hat{P}_{random}^{u:\xi})$
- 对称设计，促进互补学习

**（4）总损失**
$$\mathcal{L}_{final} = \mathcal{L}_{sup} + \mu \cdot (\mathcal{L}_{unsup}^\phi + \mathcal{L}_{unsup}^\xi)$$
其中 $\mu$ 按 epoch 高斯 ramp-up，$\alpha_l=1.0, \alpha_u=0.5, \beta=0.5$。

## 实验与结果
**数据集**：
- **LIDC-IDRI**：1609 张 2D 胸 CT，214 例患者，每例 4 个标注者标记肺结节
- **ISIC 2018**：120 张皮肤镜 RGB 图像，每例 3 个标注者

**基线对比**：Prob. U-Net、CM-Global、CM-Pixel、Pionono、D-Persona（歧义分割类）；Baseline I（自训练）、Baseline II（交叉伪监督）、Baseline III（一致性正则化）

**评估指标**：GED（多样性，越低越好）、$Dice_{soft}$（一致性）、$Dice_{max}$（最大重叠）、$Dice_{match}$（一一对应匹配）

**核心结果**（LIDC-IDRI，10% 标注 / 90% 未标注）：
| 方法 | GED↓ | $Dice_{soft}$↑ | $Dice_{max}$↑ | $Dice_{match}$↑ |
|------|-------|----------------|---------------|-----------------|
| Upper Bound（100% 标注） | 0.1461 | 90.24 | 90.75 | 90.51 |
| Baseline III | 0.2153 | 87.29 | 87.92 | 87.56 |
| **Ours (AmbiSSL)** | **0.1620** | **89.86** | **90.03** | **89.84** |

- AmbiSSL 以 10% 标注 + 90% 未标注逼近 Upper Bound（100% 标注），GED 较最强 baseline 降低 **24.7%**，$Dice_{soft}$ 提升 **2.57pp**
- ISIC 数据集（20% 标注）：GED=0.2444，$Dice_{soft}$=85.87%，同样全面超越基线

## 相关工作脉络
1. **Probabilistic U-Net [19]**：用二元高斯建模标注歧义，需多专家标注且无未标注数据利用 → 本文扩展至半监督场景
2. **Pionono [27] / D-Persona [35]**：基于变分自编码器的歧义分割 → 均依赖完整多标注集，本文引入未标注数据
3. **Cross Pseudo Supervision [6]**：双网络交叉监督用于 SSL → 本文引入"随机剪枝"而非独立网络，保持参数效率
4. **FixMatch [28] / Mean Teacher [30]**：一致性正则化 SSL 方法 → 仅输出单分割，无法建模标注歧义
5. **MCF [32]**：互校正框架 → 针对有标注数据的设计，未考虑多标注者不确定性

## 局限性与未来方向
1. **性能缺口**：当前结果仍低于 100% 标注的 Upper Bound，差距约 0.4–3.3pp（$Dice_{soft}$）
2. **剪枝策略有限**：仅对最后若干层做随机剪枝，可能不足以捕捉所有语义变化
3. **未探索更多架构**：仅测试 U-Net 类骨干，Transformer 或 3D 扩展未验证
4. **作者自述方向**：未来需探索更强策略进一步缩小"半监督+歧义感知"与全标注方法的性能鸿沟

## 研究启发与可借鉴点
1. **Laplace 替代高斯建模伪标签后验**：对不可靠伪标签用重尾分布可降低过拟合风险，可迁移至其他 SSL 场景（如自然图像分割、目标检测）
2. **随机剪枝产生多样性**：无需额外网络即生成多个"视图"，相比 Dropout/Ensemble 更轻量，可用于任何需多样性的任务（如主动学习、不确定性量化）
3. **交叉监督对称设计**：CDS 模块无需 teacher-student 权重拷贝，参数效率更高，可借鉴于多模型协作学习
4. **GED + $Dice_{match}$ 双评估**：同时衡量"多样性"和"一对一匹配质量"，为歧义分割提供更全面的评测范式

## 关键术语表
**Semi-supervised Medical Image Segmentation (SSMIS)**：结合少量标注和大量未标注数据进行医学图像分割的范式。
**Annotation Ambiguity**：不同标注专家对同一医学图像给出不同分割结果的固有不确定性。
**Diverse Pseudo-Label Generation (DPG)**：通过随机剪枝解码器生成多样化伪标签集的模块。
**Semi-Supervised Latent Distribution Learning (SSSDL)**：利用标注和伪标签共同构建公共潜在空间的分布学习模块。
**Cross-Decoder Supervision (CDS)**：剪枝解码器之间互相以对方伪标签为监督信号的训练策略。
**Generalized Energy Distance (GED)**：衡量预测集与标注集差异的多样性评估指标，值越低表示多样性越好。
**Laplace Distribution for Pseudo-label Posterior**：用拉普拉斯分布替代高斯分布建模伪标签后验，以容忍噪声。
**Re-parameterization Trick**：通过可微采样使从分布中抽取潜在编码的过程支持梯度反向传播。

## 可复现要素
- **数据集**：LIDC-IDRI（公开）、ISIC 2018（公开）——均用于胸部 CT 结节分割和皮肤病变分割
- **代码/权重**：论文未明确声明开源
- **关键超参**：$\alpha_l=1.0, \alpha_u=0.5, \beta=0.5, \mu$ 高斯 ramp-up；剪枝起始层 $L=2$、保留比例 $a\%=50\%$；学习率 1e-4；Adam 优化器
- **环境**：单卡 NVIDIA A5000（24GB），模型 31.6M 参数
