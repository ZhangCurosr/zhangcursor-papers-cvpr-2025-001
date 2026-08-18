---
title: "From-Laboratory-to-Real-World-A-New-Benchmark-Towards-Privac"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Jiang_From_Laboratory_to_Real_World_A_New_Benchmark_Towards_Privacy-Preserved_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 07:06:23"
field: "可见-红外行人重识别与联邦学习交叉"
keywords: ["VI-ReID", "联邦学习", "隐私保护", "行人重识别", "域泛化", "跨模态"]
innovations: ["提出L2RW基准与CI/EI/ES三层隐私协议，首次将隐私保护纳入VI-ReID研究框架", "设计DPPT去中心化隐私保护训练方法及MRB记忆校正模块，解决模态缺失/身份缺失/域偏移三重挑战"]
benchmarks: ["SYSU-MM01", "RegDB", "LLCM"]
---

# 论文速读：From-Laboratory-to-Real-World-A-New-Benchmark-Towards-Privacy-Preserved-Visible-Infrared-Person-Re-Identification

## 一句话总结
本文提出了首个面向隐私保护的可见-红外行人重识别（VI-ReID）基准测试 L2RW，设计了三种不同隐私严格程度的协议（CI/EI/ES），并提出了去中心化隐私保护训练方法 DPPT，使模型在不泄露原始数据的前提下实现跨域泛化。

## 研究问题与动机
- 现有 VI-ReID 方法普遍依赖集中式训练，将多源数据汇聚至单一平台，忽视了真实场景中数据分散在多设备/实体、存在严格隐私与所有权约束的现实问题。
- 真实场景中相机级数据往往独立（CI）或实体级数据隔离（EI），导致模态信息不完整（单个相机可能只有可见或只有红外图像），现有双流方法无法直接使用。
- 不同实体间行人身份重叠度低，存在身份缺失问题；同时跨相机/实体的数据分布差异带来严重的域偏移（domain shift）。
- 现有领域泛化方法同样假设数据可集中访问，无法满足隐私保护需求，亟需探索隐私保护下的跨域泛化机制。

## 核心贡献（创新点）
- 提出 L2RW 基准测试与三种协议（CI/EI/ES），首次将隐私保护作为 VI-ReID 研究的核心维度，而非事后附加。
- 设计解中心化隐私保护训练方法 DPPT，是首个以联邦学习思路解决 VI-ReID 隐私问题的方法，与现有集中式工作形成本质区别。
- 提出记忆校正银行（MRB），通过局部-全局记忆对齐解决域偏移和身份缺失问题，无需共享任何原始数据。
- 构建跨域评测协议（合并多数据集，在未见实体上测试），揭示现有 SOTA 方法在真实跨域场景下的泛化瓶颈。

## 方法详解
**1. 协议设计**
- **CI（Camera Independence）**：每个相机数据完全隔离，无法获取配对可见-红外图像，存在模态缺失。
- **EI（Entity Independence）**：同一实体内数据可共享，不同实体间数据隔离，存在身份缺失。
- **ES（Entity Sharing）**：所有实体数据可共享用于训练，但测试在未见实体上进行，最宽松但仍要求跨域泛化。

**2. 架构与采样改造**
- 将双流架构改为单流，采样仅按身份随机抽取 K 张图像（可见/红外混合），消除对模态标签的依赖。
- 引入通道增强（CA）：随机选一个通道复制到其他通道，破坏纯颜色信息，缩小跨模态差异。

**3. 记忆校正银行（MRB）**
- 局部记忆：每客户端计算各类中心点 $c_k^m$，取距中心最近的前 K 个特征求平均得到 $\hat{c}_k^m$，过滤低语义样本并保证公平性。
- 全局记忆：服务器对所有客户端的非零中心取平均聚合，得到不泄露原始数据的统一代表中心 $c_i^g$。
- 记忆校正损失（$\mathcal{L}_{mrb}$）：用余弦相似度拉近本地特征与上一轮全局中心的距离，缓解域偏移和身份缺失。
- 总损失：$\mathcal{L} = \mathcal{L}_{id} + \mathcal{L}_{cir} + \lambda \mathcal{L}_{mrb}$。

**4. 聚合策略**
- 采用 FedAvg 作为默认聚合方式，按数据量加权平均上传的本地模型参数。

## 实验与结果
- **数据集**：SYSU-MM01（491 ID，6 相机）、RegDB（412 ID，2 相机）、LLCM（1064 ID，18 相机）。
- **评估指标**：rank-1、rank-10、mAP、mINP。
- **CI 协议最强结果（SYSU-MM01, FedAvg + DPPT）**：r=1 达到 **51.27%**，mAP **49.29%**，mINP **34.47%**；相较 FedAvg+DNS†（r=1: 39.60%）提升约 11.7pp。
- **EI vs ES 对比（R+L→S）**：DPPT 在 EI 下 r=1=11.27%/mAP=11.86%，优于 ES 下各 SOTA 方法（DNS r=1=11.75% 虽略高但整体 DPPT 更均衡），证明隐私隔离不牺牲泛化能力。
- **关键结论**：现有 SOTA 在跨域 unseen entity 设置下表现普遍偏低（多数 r=1 < 12%），揭示了 VI-ReID 领域泛化的严重瓶颈。

## 相关工作脉络
- **VI-ReID 集中式方法（DNS、AGW、DEEN、CAJ 等）**：依赖配对模态输入和双流架构，无法处理模态缺失问题，且假设训练/测试同分布。
- **FedAvg / FedProx / FedNova / Moon**：经典联邦学习算法可作为 CI/EI 基准，但均不能直接解决 VI-ReID 特有的模态缺失、身份缺失和域偏移三重挑战。
- **域泛化方法（对比学习、对抗学习、因果学习等）**：需要集中访问源域数据，与隐私保护目标冲突，不适用于去中心化场景。
- **通道增强方法（CA、AGW† 改造版）**：作者复现并改造了 AGW 和 DNS 去模态化版本作为强基线，仍显著落后于 DPPT。
- **定位差异**：本文首次将隐私保护纳入 VI-ReID 研究框架，并从协议设计到算法实现提供完整解决方案，填补了领域空白。

## 局限性与未来方向
- 仅验证了 FedAvg 聚合策略，FedProx/Moon/FedNova 等算法的融合潜力未充分探索。
- 跨域 unseen entity 评估下绝对精度仍然偏低（r=1 < 12%），说明真实部署仍需进一步改进。
- 实验基于公开数据集重新划分，与真实异构部署场景可能存在差距。
- 未来可探索半监督/自监督学习以缓解标签缺失，或引入更多细粒度隐私保护机制（如差分隐私）。

## 研究启发与可借鉴点
- **分层隐私协议设计思路**：CI/EI/ES 的梯度隐私控制框架可迁移至其他跨模态/跨域视觉任务（如 RGB-Thermal ReID）。
- **单流改造 + 通道增强**：将双流架构统一为单流并配合 CA 是一种低成本解决模态缺失的有效策略，可直接复用。
- **记忆校正银行（MRB）机制**：通过本地-全局中心对齐缓解域偏移和身份缺失的思想，适用于任意联邦跨域特征学习场景。
- **跨域 unseen entity 评测范式**：合并多数据集并在未见域测试的设置值得在通用 ReID 和领域泛化工作中推广。

## 关键术语表
- **VI-ReID（Visible-Infrared Person Re-Identification）**：跨可见光与红外模态的行人重识别任务。
- **L2RW（Laboratory to Real World）**：本文提出的首个面向真实隐私约束场景的 VI-ReID 基准测试。
- **CI（Camera Independence）**：相机级完全隔离的隐私协议，模拟最高隐私约束场景。
- **EI（Entity Independence）**：实体级隔离、实体内可共享的隐私协议，模拟中等隐私约束。
- **ES（Entity Sharing）**：实体间数据可共享但测试在未见实体上的协议，兼容传统集中式方法。
- **DPPT（Decentralized Privacy-Preserved Training）**：本文提出的去中心化隐私保护训练方法。
- **MRB（Memory Rectification Bank）**：通过局部-全局记忆对齐校正域偏移和身份缺失的记忆模块。
- **CA（Channel Augmentation）**：通过通道复制破坏颜色信息的增强手段，用于缩小跨模态差异。

## 可复现要素
- **数据集**：SYSU-MM01、RegDB、LLCM（均为公开数据集）。
- **代码**：已开源，GitHub 地址 https://github.com/Joey623/L2RW。
- **权重**：论文未明确说明预训练权重开源情况，需自行下载 ImageNet-1k 预训练 ResNet-50。
- **关键超参**：TOP K = 4，λ = 1.0，batch-size = 64（8 身份 × 8 图像），lr = 0.2，epochs CI=50 / EI=30，使用 OneCycleLR 调度器。
