---
title: "Handling-Spatial-Temporal-Data-Heterogeneity-for-Federated-C"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Yu_Handling_Spatial-Temporal_Data_Heterogeneity_for_Federated_Continual_Learning_via_Tail_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 07:09:49"
field: "联邦持续学习"
keywords: ["Federated Continual Learning", "Spatial-Temporal Data Heterogeneity", "Catastrophic Forgetting", "Vision Transformer", "Tail Anchor", "Feature Space Alignment"]
innovations: ["提出 Tail Anchor 机制通过冻结预训练 ViT 和可学习尾锚同时克服参数遗忘与输出遗忘", "设计最优全局原型选择策略以最低平均相似度准则维护特征空间稳定性", "首次系统揭示输入层异构导致参数遗忘和输出遗忘的双重遗忘机制"]
benchmarks: ["CIFAR-100", "ImageNet-R"]
---

# 论文速读：Handling-Spatial-Temporal-Data-Heterogeneity-for-Federated-C

## 一句话总结
本文提出 **FedTA（Federated Tail Anchor）**，利用冻结的预训练 ViT 提取稳定特征，并通过可学习的 Tail Anchor 将其与固定全局原型对齐，从而同时解决联邦持续学习（FCL）中因空间（Non-IID）和时间（任务流）数据异构引起的**参数遗忘**与**输出遗忘**。

## 研究问题与动机
- **FCL 的双重异构挑战**：传统 FL 仅考虑跨客户端的空间数据异构（Non-IID），而 FCL 还需处理同一客户端跨任务的时间数据异构，两者叠加导致更严峻的**空间-时间灾难性遗忘（ST-CF）**。
- **参数遗忘（parameter-forgetting）**：实证发现，空间-时间异构使特征提取器对同一输入提取出无关紧要的特征（如从猫的边缘提取特征），而非判别性特征。
- **输出遗忘（output-forgetting）**：在特征空间中，相同样本的特征位置随时间/空间变化逐渐偏离原始位置，导致旧知识遗忘。
- **现有方法局限**：现有 FCL 方法多依赖重放（replay）或生成伪数据来缓解遗忘，存在隐私泄露风险和高计算成本；少数同时考虑时空异构的方法未深入分析其对遗忘的影响机制。

## 核心贡献（创新点）
1. **首次从遗忘视角系统分析空间-时间异构的影响**：通过可视化与定量分析揭示了输入层异构如何导致参数遗忘与输出遗忘，这是以往 FCL 研究忽视的机制。
2. **提出 FedTA 框架，同时克服参数遗忘与输出遗忘**：利用冻结预训练 ViT 消除参数更新引起的遗忘，通过 Tail Anchor 控制特征在特征空间的位置不变性——与已有方法（依赖重放或仅调分类头）的本质区别在于**从特征空间稳定性入手**而非仅优化参数。
3. **设计选择性输入知识融合（SIKF）**：服务器从各客户端的知识库中选取最优的输入增强参数进行蒸馏融合，替代直接加权平均，提升了知识共享效率。
4. **提出最优全局原型选择（BGPS）**：以"最低平均相似度"准则为每类选定全局原型作为固定锚点，当平均相似度低于阈值时锁定原型，防止后续迭代漂移。
5. **高效且无重放的隐私友好设计**：可训练参数量仅 253,440（vs. ResNet-18 的 11,306,804），通信开销极小，且无需存储历史数据。

## 方法详解
FedTA 采用两阶段本地训练 + 服务器聚合流程：

**阶段一：Input Enhancement（输入增强）**
- 每客户端共享同一预训练 ViT（参数冻结），定义知识库 $KB_i = \{IE_1, IE_2, \ldots, IE_M\}$，其中每个 $IE$ 是可学习参数，配有对应 key $K^{ie}$。
- 输入嵌入 $E = f_e(x)$ 经 ViT 提取后，通过 key 相似度检索选出最佳 $N$ 个 $IE$，拼接形成增强嵌入：$E' = [IE_{s_1}, \dots, IE_{s_N}; E]$。
- 优化目标：$\min \mathcal{L}(\mathcal{V}_e^i(E'), y) + \lambda_1 \sum \text{dis}(K_{in}^{ie}, K_{s_i}^{ie})$，第一项为 softmax cross-entropy，第二项使选中 IE 的 key 靠近查询特征（余弦相似度）。

**阶段二：Tail Anchor（尾锚）**
- 冻结阶段一的 IE 和 key，将增强嵌入 $E'$ 再次送入冻结 ViT 得到特征 $\mathcal{F}_{out}$。
- Tail Anchor 定义为 m 个类别的键值对 $\mathcal{TA} = \{(K_1^{ta}, TA_1), \ldots, (K_m^{ta}, TA_m)\}$。通过余弦相似度为 $\mathcal{F}_{out}$ 匹配最优 key：$K_s^{ta} = \arg\min_{K^{ta}} \text{dis}(\mathcal{F}_{out}, K_i^{ta})$。
- 将匹配到的 TA 与 $\mathcal{F}_{out}$ 混合得到 $\mathcal{F}_{TA}$，训练损失：
  $$\mathcal{L}_{ta} = \mathcal{L}_{CE}(\mathcal{F}_{TA}) + \lambda_2 \mathcal{L}_{cons}(\mathcal{F}_{TA}) + \lambda_3 \text{dis}(\mathcal{F}_{TA}, K_s^{ta})$$
  其中 $\mathcal{L}_{cons}$ 为对比损失，使 $\mathcal{F}_{TA}$ 趋近全局原型 $\mathcal{G}^y$：
  $$\mathcal{L}_{cons} = -\log \frac{\exp(\mathcal{F}_{TA} \cdot \mathcal{G}^y / \tau)}{\sum_{y_a \in \mathcal{D}^t} \exp(\mathcal{F}_{TA} \cdot \mathcal{G}^{y_a} / \tau)}$$
- 训练结束后，计算各类别本地原型 $P_i^y = \frac{1}{|\mathcal{D}_a^y|}\sum \mathcal{F}_{TA}^x$，上传至服务器。

**服务器端：Selective Input Knowledge Fusion（SIKF）**
- 服务器持有小批量代理数据集 $\mathcal{D}_s$，随机选取 $KB_i$ 作为蒸馏目标，计算跨客户端增强嵌入的特征一致性：
  $$\mathcal{L}_{KD} = \frac{1}{n-1} \sum_{j \neq i} \text{MSE}(\mathcal{V}(E_i'), \mathcal{V}(E_j'))$$

**服务器端：Best Global Prototype Selection（BGPS）**
- 将各类别的本地原型排序构成 $P_G$，构建相似度邻接矩阵 $\mathcal{M}_{ij} = \text{dis}(P_G^i, P_G^j)$（同类原型间 $\mathcal{M}_{ij}=1$）。
- 为每类 y 选择平均相似度最低的原型作为全局原型：$\mathcal{G}^y = P_g^s = \arg\min_{i} \bar{\mathcal{M}}_i$。
- 若 $\bar{\mathcal{M}}_i < \text{Thr}$，则锁定该全局原型不再更新。

## 实验与结果
- **数据集**：CIFAR-100（5 客户端，每客户端 15 私有类 + 25 公共类，5 个任务×8 类）；ImageNet-R（5 客户端，每客户端 40 私有类、无公共类，更强空间异构，5 个任务×8 类）。
- **基线**：FedAvg、FedProx、FedNova（纯 FL）；FedLwF、FedViT、FedL2P、FedDualP（FL+CL）；GLFC、TARGET、MFCL、FedMGP（FCL）。
- **评估指标**：时间知识保留 $KR_t$（ Eq.11）与空间知识保留 $KR_s$（ Eq.12）。
- **主要结果**：
  - CIFAR-100 第 1 个任务：FedTA 达 **96.1%**，显著超越 FedMGP（90.2%）、GLFC（82.0%）；第 5 个任务仍保持 **89.4%**。
  - ImageNet-R 第 5 个任务：FedTA 达 **85.0%**，超越 FedMGP（75.4%）约 9.6 个百分点。
  - $KR_t$ 与 $KR_s$ 均维持在 **~98%**，而其他方法（尤其 $KR_t$）极低。
- **消融**：移除 Tail Anchor（Ours-w/o TA）后 CIFAR-100 第 1 任务降至 78.7%；移除 SIKF 或 BGPS 均有下降，验证各组件有效性。
- **可视化**：T-SNE 显示 FedTA 能有效保持同类特征相对位置不变，而其他方法特征位置严重偏移。

## 相关工作脉络
- **FedAvg/FedProx/FedNova**：经典 FL 方法处理 Non-IID，但基于静态数据假设，无法适应持续学习场景；FedTA 突破此限制。
- **FedL2P/FedDualP**：在输入侧和模型内部引入可学习参数，本地效果好但聚合后遗忘严重；FedTA 通过冻结 ViT + Tail Anchor 从根本上缓解此问题。
- **TARGET/MFCL/GLFC**：依赖重放或伪数据缓解遗忘，存在隐私风险；FedTA 无需重放，从特征空间稳定性入手。
- **FedMGP**：联邦多粒度提示学习，但仍以参数/原型为中心；FedTA 额外从"特征位置不变性"角度切入。
- **Li et al. (2024, arXiv:2404.something)**：少数同时考虑时空异构的工作，但未深入分析其如何导致遗忘；本文首次揭示 parameter-forgetting 与 output-forgetting 机制。
- **FedViT/Yang et al. (2024 survey)**：利用预训练 ViT 的思路已有探索，但本文将其与 Tail Anchor 结合并系统化解决 FCL 遗忘问题。

## 局限性与未来方向
- **Surrogate 数据量有限**：ImageNet-R 上 SIKF 效果不如直接加权平均，作者归因于代理数据不足，限制了下采样选择的质量。
- **Tail Anchor 数量敏感**：实验中 100 个 Tail Anchor 时有部分样本位置偏移，增至 500 个后改善，需根据任务调整，存在超参选择难题。
- **原型上传的隐私风险**：作者指出类专属原型可能被逆向还原原始特征，虽然随机混合可缓解，但未给出完整解决方案。
- **仅验证视觉分类任务**：方法针对 Vision Transformer + 图像分类设计，跨模态（文本、时序）或检测/分割任务的泛化性未知。
- **全局原型固定阈值的设定**：Thr 需要根据数据集和任务数量人工调节，缺乏自适应机制。

## 研究启发与可借鉴点
1. **冻结预训练 backbone + 少量可学习适配器**的思路可迁移至其他领域（如 NLP 中的 frozen LLM + prompt tuning），在保持特征稳定性的同时实现任务适应。
2. **"最低平均相似度"原型选择策略**体现了"多样性优先"原则，避免全局原型过于集中，可用于其他需要聚合多方表征的场景（如个性化 FL、多源域适配）。
3. **输入增强（IE）作为可学习 key-value 检索机制**的设计通用性强，可推广到时序数据（如可学习的 patch tokens）或图数据（可学习的节点投影向量）。
4. **两个遗忘度量（$KR_t$ 与 $KR_s$）可成为 FCL 评测标准**，建议团队在后续工作中采用，与 accuracy 指标互补。
5. **无重放的遗忘缓解**思路对隐私敏感场景（医疗、金融联邦学习）具有直接应用价值，可与团队现有的隐私保护研究结合。

## 关键术语表
- **FedTA（Federated Tail Anchor）**：本文提出的联邦持续学习框架，通过冻结预训练 ViT 和可学习尾锚协同工作，解决空间-时间数据异构引发的遗忘问题。
- **Spatial-Temporal Catastrophic Forgetting（ST-CF）**：联邦持续学习中由空间异构（Non-IID）和时间异构（任务流）共同导致的灾难性遗忘现象。
- **Parameter-forgetting**：空间-时间异构使特征提取器对同一输入提取出无关紧要的特征（如图像边缘特征），导致参数层面的遗忘。
- **Output-forgetting**：在特征空间中，相同样本的特征位置随时间/空间变化逐渐偏离原始位置，导致旧知识遗忘。
- **Tail Anchor（尾锚）**：每类一个的可学习 key-value 参数对，与冻结 ViT 输出特征混合，用于在特征空间中固定各类别的参考位置。
- **Input Enhancement（输入增强）**：可学习参数集合，拼接到 ViT 的图像嵌入之后，通过对比检索机制增强预训练模型在下游任务上的表现。
- **Selective Input Knowledge Fusion（SIKF）**：服务器端对来自不同客户端的输入增强参数进行知识蒸馏融合，替代直接平均。
- **Best Global Prototype Selection（BGPS）**：服务器根据"最低平均相似度"准则为每类选定全局原型，当相似度低于阈值时锁定原型。

## 可复现要素
- **数据集**：CIFAR-100（公开）、ImageNet-R（公开）；论文未提及是否自定义划分。
- **代码**：已开源，地址 https://github.com/SkyOfBeginning/FedTA。
- **关键超参**：Input Enhancement 数量=10，长度=10，维度=768；Tail Anchor 数量=100（敏感性分析用 500）；ViT embed dim=768；代理数据集 CIFAR-100 每类 20 样本、ImageNet-R 每类 5 样本；温度系数 $\tau$ 未具体给出（见附录）。
