---
title: "AdaptCMVC-Robust-Adaption-to-Incremental-Views-in-Continual"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_AdaptCMVC_Robust_Adaption_to_Incremental_Views_in_Continual_Multi-view_Clustering_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 06:59:30"
field: "持续学习与多视图表示学习"
keywords: ["持续多视图聚类", "域适应", "自训练", "灾难性遗忘", "噪声鲁棒", "结构对齐"]
innovations: ["从域适应视角重构CMVC问题，提出统一基线模型增量适应策略", "设计噪声鲁棒自训练框架，通过距离加权一致性损失抑制视图噪声", "提出结构对齐学习机制，通过相似度矩阵EMA更新防止灾难性遗忘"]
benchmarks: ["E-MNIST", "E-FMNIST", "Office-31", "COIL-100", "COIL-20", "PatchedMNIST"]
---

# 论文速读：AdaptCMVC: Robust Adaption to Incremental Views in Continual Multi-view Clustering

## 一句话总结
本文从域适应视角重新审视**持续多视图聚类（CMVC）**问题，提出 **AdaptCMVC** 方法，通过**噪声鲁棒的自训练框架**（加权平均教师模型 + 距离加权一致性损失）和**结构对齐学习机制**，实现模型在增量视图到来时有效适应新视图并防止灾难性遗忘，在六个基准数据集上显著优于现有 CMVC 方法。

## 研究问题与动机
- **多视图数据增量到达的现实场景**：现实中多视图/多模态数据（如医疗影像 MRI、CT、X-ray）往往分阶段收集，传统 MVC 要求同时访问所有视图，在数据隐私受限或存储受限的场景下不切实际。
- **现有 CMVC 方法的两大缺陷**：
  1. **视图间差异大**：现有方法（如 CMVC、CAC）为每个视图单独训练模型并通过移动平均更新共识矩阵，缺乏对新视图的有效适应能力，难以应对视图间的较大特征空间鸿沟。
  2. **视图噪声敏感**：部分视图含有噪声特征，直接更新共识矩阵会将噪声传播并积累。
- **灾难性遗忘**：模型仅访问当前视图数据，缺乏历史视图信息，持续适应过程中容易丢失之前视图的聚类结构知识。

## 核心贡献（创新点）
1. **域适应视角重构 CMVC**：与现有"为每个视图训练独立模型 + 后期融合"范式不同，本文提出统一的基线模型，通过持续适应策略将新视图视为目标域进行增量适配，首次将域适应思想引入 CMVC。
2. **噪声鲁棒自训练框架**：引入加权平均教师模型生成高质量目标特征，设计基于样本到聚类原型距离的距离加权一致性损失，自动降低噪声样本对模型更新的负面影响。
3. **结构对齐学习机制**：通过保持当前视图与上一视图样本间相似结构的一致性（MSE loss on similarity matrix），利用上一视图的软聚类分配矩阵作为结构先验，无需访问历史数据即可保留跨视图的全局聚类结构。
4. **系统性实验验证**：在 E-MNIST、E-FMNIST、Office-31、COIL-100、COIL-20、PatchedMNIST 六个数据集上全面评估，AdaptCMVC 在多数场景下超越 SOTA CMVC 方法（CMVC、CAC）及传统 MVC 方法。

## 方法详解
**整体架构**：采用统一的 VAE 基线模型，在首个视图上预训练，后续每个会话中通过自训练框架适应新视图。

**1. 基线模型（Base Model）**
- 使用变分自编码器（VAE）学习视图特定的噪声鲁棒表示。
- 编码器 $E_\phi$ 输入掩码版本 $\mathbf{x}_i^{v'}$，输出近似后验 $q_\phi(\mathbf{z}_i^v|\mathbf{x}_i^{v'}) = \mathcal{N}(\mu^v, \sigma^{v2})$。
- 解码器 $D_\theta$ 重构原始样本。
- 训练目标为 ELBO 损失：
$$\mathcal{L}_r = \mathbb{E}[\log p_\theta(\mathbf{x}_i^v|\mathbf{z}_i^v)] - KL(q_\phi(\mathbf{z}_i^v|\mathbf{x}_i^{v'}) \| p(\mathbf{z}_i^v))$$

**2. 噪声鲁棒自训练框架（Noise Robust Self-training）**
- **学生-教师架构**：初始化时学生模型和教师模型参数相同；$t>0$ 时教师模型参数通过 EMA 更新：$\psi'_t = \alpha \psi'_{t-1} + (1-\alpha)\psi_t$。
- **一致性损失**：
$$\mathcal{L}_c(\mathbf{z}_i^v) = -\sum_d p(\mathbf{z}_{id}^v) \log p(\mathbf{z}_{id}^{v'})$$
其中 $p(\cdot)$ 为 softmax 后的分布。
- **距离加权噪声鲁棒性**：利用上一视图的聚类原型 $\mathbf{B}^{v-1}$ 和软聚类分配 $\mathbf{H}^{v-1}$，计算样本到对应原型的距离 $l(i, b_i^{v-1}) = \|\mathbf{z}_i^v - \mathbf{h}_i^{v-1}\mathbf{B}^{v-1}\|^2$，重构损失为：
$$\mathcal{L}_c(\mathbf{z}_i^v) = -\frac{1}{(l(i, b_i^{v-1}))^2} \sum_d p(\mathbf{z}_{id}^v) \log p(\mathbf{z}_{id}^{v'})$$
距离越大（越可能是噪声样本），权重越小，降低其对模型更新的影响。

**3. 结构对齐学习（Structure Alignment Learning）**
- 防止灾难性遗忘的核心机制，通过 MSE loss 保持当前视图与上一视图的结构一致性：
$$\mathcal{L}_s = \frac{1}{n^2}\sum_i\sum_j \|\mathbf{s}_{ij}^{v-1} - \mathbf{c}_{ij}^v\|$$
其中 $\mathbf{c}_{ij}^v$ 为当前视图特征的余弦相似度，$\mathbf{s}_{ij}^{v-1}$ 为上一视图的相似结构。
- **结构矩阵初始化**：会话开始时，$\mathbf{S}^{v-1}$ 由上一视图的软聚类分配矩阵 $\mathbf{H}^{v-1}$ 初始化：同簇样本设为 1，不同簇设为 0。
- **结构矩阵动态更新**：在训练过程中通过 EMA 更新以捕获联合结构：
$$\mathbf{S}_t^{v-1} = \beta \mathbf{S}_{t-1}^{v-1} + (1-\beta)\mathbf{C}_t^v$$
- 整个过程**无需访问历史视图数据**，仅依赖存储的相似度矩阵和聚类分配。

**总损失函数**：$\mathcal{L} = \mathcal{L}_r + \mathcal{L}_c + \mathcal{L}_s$

## 实验与结果
- **数据集**：E-MNIST、E-FMNIST、Office-31、COIL-100、COIL-20、PatchedMNIST（共6个多视图基准数据集）。
- **评估指标**：Accuracy (ACC) 和 Normalized Mutual Information (NMI)。
- **基线方法**：单视图方法（Joint-VAE、β-VAE）、传统多视图聚类方法（MFLVC、CONAN、EAMC、GCFAgg、Multi-VAE）、现有 CMVC 方法（CMVC、CAC）。
- **主要结果**：
  - 在 PatchedMNIST（最多视图）上，AdaptCMVC 达到 **ACC=84.33% / NMI=55.87%**，分别较最佳 CMVC 基线提升 **+13.30** 和 **+15.17**。
  - 在 COIL-100 上，AdaptCMVC 达到 **ACC=79.29 / NMI=??**，较 CMVC 提升 **+20.71**（ACC）和 **+17.31**（NMI）。
  - 平均排名：AdaptCMVC 在 ACC 上排名第 **2.3**，NMI 上排名第 **3.5**，全面超越所有基线。
  - 与训练时可见全部视图的传统 MVC 方法 GCFAgg 相比，AdaptCMVC 在多数数据集上达到或接近其性能（PatchedMNIST 上用 ∗ 标记为超越）。
- **消融实验**：移除噪声鲁棒一致性损失或结构对齐机制均导致性能下降，验证两个核心模块的有效性。
- **灾难性遗忘分析**：最终模型在早期视图上的聚类性能等于或优于仅在该视图上训练的 Base Model，证明结构对齐机制有效保留了历史知识。
- **视图顺序鲁棒性**：在 E-MNIST 上交换视图顺序，AdaptCMVC 始终优于对比方法，结果稳定。

## 相关工作脉络
1. **传统多视图聚类（MVC）**：如 Multi-VAE、GCFAgg、CONAN 等，假设所有视图同时可用，无法处理增量场景，且需要重新训练。
2. **持续多视图聚类（CMVC）**：CMVC [58] 和 CAC [42] 为代表，采用"每视图独立模型 + 共识矩阵移动平均"的后期融合范式，缺乏对新视图的有效适应能力和抗噪声能力。
3. **域适应（Domain Adaptation）**：本文受 Continual Test-time Domain Adaptation [47] 启发，但与之本质不同：域适应通常为监督设定，本文在完全无监督的 CMVC 设定下解决视图增量适应问题。
4. **自训练/教师-学生框架**：Mean Teacher [39]、BYOL [4]、MoCo [5] 等自监督学习方法启发了本文的加权平均教师模型设计，用于生成更稳定的目标特征。
5. **持续学习防遗忘技术**：结构对齐机制与参数正则化（如 EWC）、经验回放等方法思路不同，本文通过显式对齐相似度矩阵结构来保留知识，无需访问历史数据。

## 局限性与未来方向
- **单一基线模型假设**：所有视图共享同一编码器/解码器，当视图间差异极大时，统一表示空间可能难以充分捕捉各视图的独特语义。
- **结构矩阵 EMA 更新策略**：$\beta$ 超参数控制历史结构与新结构的平衡，缺乏自适应机制，在不同数据集上可能需要手动调优。
- **仅适用于图像数据**：实验主要在视觉数据集上进行，对于图数据、文本数据等其他模态的多视图聚类，方法的适用性有待验证。
- **聚类原型依赖 K-means**：噪声鲁棒损失依赖上一视图的 K-means 聚类结果来计算样本-原型距离，若上一视图聚类质量较差，可能影响当前视图的适应效果。
- **未讨论不完整视图场景**：实际应用中可能存在某些视图缺失的情况，本文方法假设每个会话的所有样本在所有视图上均有完整特征。

## 研究启发与可借鉴点
1. **域适应视角重构持续学习问题**：将 CMVC 视为连续域适应任务，利用学生-教师自训练框架进行增量适应，这一思路可迁移至其他持续聚类或持续表示学习场景。
2. **距离加权噪声鲁棒损失设计**：利用历史聚类原型与样本距离自适应调整损失权重，是一种简洁有效的噪声抑制策略，可推广至其他持续学习中处理低质量/噪声样本的场景。
3. **结构对齐而非参数正则化**：通过相似度矩阵的 EMA 更新和 MSE 对齐来保留跨视图结构知识，避免了参数正则化方法对超参数的敏感依赖，为持续学习中的遗忘缓解提供了新思路。
4. **无数据回访的持续学习范式**：整篇方法完全无需存储或访问历史数据，仅依赖中间表示（聚类分配、相似度矩阵），在数据隐私敏感场景下具有重要实用价值。
5. **可扩展至不完整视图场景**：当前方法可自然扩展至部分视图缺失的持续聚类设定，只需对缺失视图的样本跳过对应损失项即可。

## 关键术语表
- **Continual Multi-view Clustering (CMVC)**：持续多视图聚类，指多视图数据以增量方式依次到达，模型需在仅访问当前视图数据的条件下持续聚类并防止遗忘。
- **Noise Robust Consistency Loss**：噪声鲁棒一致性损失，通过样本到历史聚类原型的距离加权一致性损失，自动降低噪声样本对模型更新的影响。
- **Structure Alignment Learning**：结构对齐学习，通过 MSE loss 对齐当前视图与历史视图的样本相似度矩阵，保留跨视图的聚类结构知识。
- **Weight-Averaged Teacher Model**：加权平均教师模型，通过指数移动平均（EMA）更新参数的教师网络，提供更稳定高质量的目标特征。
- **Catastrophic Forgetting**：灾难性遗忘，持续学习过程中模型因适应新任务而丢失已学知识的现象。
- **Exponential Moving Average (EMA)**：指数移动平均，用于平滑教师模型参数更新，公式为 $\psi'_t = \alpha \psi'_{t-1} + (1-\alpha)\psi_t$。
- **Variational Autoencoder (VAE)**：变分自编码器，本文用作基线模型骨干，通过编码-解码结构学习噪声鲁棒的隐空间表示。

## 可复现要素
- **数据集**：E-MNIST、E-FMNIST、Office-31、COIL-100、COIL-20、PatchedMNIST，均为公开数据集。
- **代码**：论文声明代码可用（"The code is available at: AdaptCMVC"），但未提供具体 URL；建议关注论文附录或作者主页获取。
- **关键超参**：训练轮数 50 epoch/session；batch size = 256；学习率 $1 \times 10^{-4}$；Adam 优化器；EMA 系数 $\alpha$、结构更新系数 $\beta$；数据增强采用随机水平翻转、ColorJitter、高斯模糊和高斯噪声。
- **实现框架**：PyTorch；编码器/解码器为卷积网络结构（参考 [13]）。
