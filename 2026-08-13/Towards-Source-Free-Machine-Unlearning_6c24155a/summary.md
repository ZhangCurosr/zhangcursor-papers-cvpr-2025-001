---
title: "Towards-Source-Free-Machine-Unlearning"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Ahmed_Towards_Source-Free_Machine_Unlearning_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 09:49:53"
field: "机器学习隐私与合规"
keywords: ["机器遗忘", "源自由学习", "零样本遗忘", "Hessian估计", "参数不可区分性", "混合线性网络", "NTK"]
innovations: ["提出首个实例级零样本线性分类器遗忘方法，无需访问剩余训练数据", "设计基于遗忘数据扰动逼近剩余数据Hessian的SDP估计机制，并提供严格误差上界", "将方法扩展至NTK混合线性框架，在ResNet-18深度网络上验证有效性"]
benchmarks: ["CIFAR-10", "CIFAR-100", "Stanford Dogs", "CalTech-256"]
---

# 论文速读：Towards-Source-Free-Machine-Unlearning

## 一句话总结
本文提出了首个面向线性分类器的**源自由（source-free）机器遗忘算法**，在不访问原始训练数据的情况下，通过用遗忘数据估计剩余数据的 Hessian 矩阵，实现了对任意类别中任意实例的零样本遗忘，并提供了严格的理论误差上界保证。

## 研究问题与动机
1. **现有方法依赖训练数据**：主流机器遗忘算法均假设在遗忘过程中可访问完整训练集或剩余数据子集，但现实中因存储成本与隐私限制，原始数据往往不可获取。
2. **已有零样本方法粒度粗糙**：Chundawat 等提出的 zero-shot 方法只能整体遗忘某个类别，无法实现跨类别的实例级精确删除。
3. **实例级零样本方法不可扩展**：Cha 等的 Adversarial 方法虽然实现了实例级遗忘，但随着遗忘样本数量增加，测试集和剩余数据集性能显著下降。
4. **缺乏形式化理论保证**：现有零样本方法均未提供遗忘完整性与数据移除效果的严格理论保证，难以在实际合规场景中部署。

## 核心贡献（创新点）
1. **首个面向线性分类器的实例级零样本遗忘方法**：与 Chundawat 等方法只能遗忘整个类别不同，本文方法可对任意类别中的任意实例进行精确遗忘。
2. **基于遗忘数据估计剩余 Hessian 的新机制**：提出通过构造半定规划（SDP）问题，仅用遗忘数据及其梯度来近似剩余数据的 Hessian 矩阵，本质区别在于不再依赖剩余训练集。
3. **通用凸损失函数框架**：算法适用于任意可微凸损失函数（包括二次损失），而不仅是特定损失设计。
4. **严格的理论保证**：给出了估计 Hessian 与真实 Hessian 之间 Frobenius 范数的误差上界（Lemma 1），以及遗忘后模型在剩余数据上梯度范数的上界（Theorem 1）。
5. **可扩展至混合线性神经网络**：将方法应用于基于 NTK 的混合线性遗忘框架（Mixed Linear Unlearning），验证了在 ResNet-18 深度网络上的有效性。

## 方法详解
- **问题设定**：给定已训练好的线性分类器参数 $w^\star$ 和遗忘数据集 $\mathcal{D}_f$，目标是求遗忘后参数 $w_{uf}$，而无需访问剩余数据集 $\mathcal{D}_r$。
- **核心思路**：基于 Guo 等人的 Newton 更新公式 $w_{uf} = w^\star + H_r^{-1}\nabla_f$，关键挑战是无法直接计算剩余数据 Hessian $H_r$。
- **Hessian 估计**：利用 Taylor 展开，在 $w^\star$ 附近生成 $m$ 个随机扰动点 $w_i = w^\star + \delta w_i$（$\delta w_i \sim \mathcal{N}(0,1)$），构造目标函数：
  $$\tilde{\Psi}(H_r) = \frac{1}{m}\sum_{i=1}^{m}\left(\frac{1}{2}(\delta w)_i^\top H_r (\delta w)_i - \nabla_f^\top(\delta w)_i - \delta\mathcal{L}_f(w_i)\right)^2$$
  核心近似：假设剩余数据与遗忘数据在扰动点的损失差值有界（$|\delta\mathcal{L}_r - \delta\mathcal{L}_f| \leq \epsilon$），用 $\delta\mathcal{L}_f$ 替代 $\delta\mathcal{L}_r$。
- **优化形式**：转化为半定规划（SDP）问题，约束 $H_r \succeq 0$（保证正半定性），求解估计 $\hat{H}_r$。
- **遗忘更新**：将 $\hat{H}_r$ 代入 Newton 步完成参数更新。
- **混合线性扩展**：对深度网络最后若干层做 NTK 一阶泰勒展开线性化，将遗忘问题转化为凸优化问题求解。

## 实验与结果
- **数据集**：CIFAR-10、CIFAR-100、Stanford Dogs、CalTech-256；使用 ResNet-18 作为特征提取器（ImageNet 预训练）。
- **评估指标**：Test Accuracy、Remaining Accuracy、Forget Accuracy、Membership Inference Attack（MIA）得分（接近 50% 表示成功遗忘）。
- **主要结果**：
  - **CIFAR-10（线性分类器，10% 遗忘数据）**：Unlearned(-) Test=70%，Retrained Test=72%，Performance Gap=2%；MIA=51.4%（接近理想的 50%）。
  - **CIFAR-100（混合线性网络，10% 遗忘数据）**：Unlearned(-) Test=61.4%，Retrained Test=63.1%，Performance Gap=0.5%。
  - **遗忘数据比例影响**：遗忘 5% 时 Performance Gap 接近 0%；15% 时 Test 差距扩大到 14%；20% 时 CIFAR-100 上 Test Gap 为 0.1%~2.4%。
  - **扰动数量影响**：250 个扰动时 Test Gap=15%；500 个时 Gap=2%；1000 个时 Gap≈0%。
  - **L2 正则化影响**：增大 $\lambda$ 可缩小 Performance Gap，与 Theorem 1 理论一致。
  - **与源自由基线对比（CIFAR-10，10% 遗忘）**：NegGrad Test=51.9%，Random Labels Test=20.6%，JiT Test=52.1%，Adversarial Test=51.5%；本文方法 Test=70%，显著优于所有基线。

## 相关工作脉络
1. **Golatkar et al. (Eternal Sunshine, CVPR 2020)**：基于梯度下降的选择性遗忘方法，但需要访问训练数据，属于有源遗忘。
2. **Chundawat et al. (Zero-shot MU, TIFS 2023)**：首个零样本遗忘方法，但只能按类别级遗忘，无法实例级控制。
3. **Guo et al. (Certified Data Removal, ICML 2020)**：提供理论保证的参数不可区分性框架，但需访问全部训练数据，非源自由。
4. **Foster et al. (JiT Unlearning, arXiv 2024)**：通过 Lipschitz 正则化和扰动样本减少梯度影响，但未提供形式化理论保证。
5. **Cha et al. (Learning to Unlearn, AAAI 2024)**：实例级零样本遗忘，使用对抗样本生成，但随遗忘样本数增加性能急剧下降，且无理论保证。
6. **Golatkar et al. (Mixed-Privacy, CVPR 2021)**：NTK 混合线性遗忘框架，需要剩余数据计算精确 Hessian，本文在其基础上解决了源自由场景下的 Hessian 估计问题。

## 局限性与未来方向
1. **理论保证仅在凸损失下严格成立**：对非凸深度网络的理论分析依赖于 NTK 线性化近似，实际性能可能与理论预测存在偏差。
2. **Hessian 矩阵存储与求逆的计算开销**：高维场景下存储完整 Hessian 不切实际，论文指出可结合对角化或 Hessian-vector product 等近似技术，但未深入展开。
3. **扰动数量需要较大才能保证精度**：实验显示 1000 个扰动才能达到较好效果，增加了计算成本。
4. **目前仅验证了图像分类任务**：在 NLP、生成模型等其他领域的泛化性有待探索。

## 研究启发与可借鉴点
1. **Hessian 估计思路可迁移**：用遗忘数据梯度+扰动逼近剩余数据 Hessian 的方法，可推广至其他缺乏源数据的隐私/合规场景（如模型卡维护、数据集版本管理）。
2. **评估框架值得借鉴**：同时报告 Test/Remaining/Forget Accuracy 与 MIA 得分，并对比 Retrained 和 Unlearned(+)（有源理想上界）两个基准，形成了完整的效果评估闭环。
3. **消融实验设计系统**：针对遗忘数据比例、扰动数量、L2 正则化三个关键超参数做了细致消融，与理论分析相互印证，增强了可信度。
4. **混合线性扩展策略**：将源自由方法无缝接入 NTK 框架，为在深度网络上应用源自由遗忘提供了可行路径，后续可在更大规模网络（如 ResNet-50、ViT）上验证。

## 关键术语表
- **Source-Free Machine Unlearning（源自由机器遗忘）**：在原始训练数据集不可访问的情况下，仅凭已训练模型和遗忘数据完成信息删除的机器遗忘场景。
- **Parameter Indistinguishability（参数不可区分性）**：衡量遗忘后模型与从剩余数据重新训练模型之间输出分布的相似程度，是验证遗忘效果的核心理论指标。
- **Retain Hessian（剩余数据 Hessian）**：剩余训练数据集对应的损失函数二阶导数矩阵，是 Newton 遗忘步的关键组件，传统方法需直接计算。
- **Mixed Linear Unlearning（混合线性遗忘）**：利用 NTK 对深度神经网络进行一阶泰勒展开线性化，将非凸遗忘问题转化为凸优化问题的方法。
- **Neural Tangent Kernel（NTK）**：描述神经网络参数微小变化对输出影响的一阶近似核函数，用于分析线性化神经网络的优化行为。
- **Membership Inference Attack（MIA）**：攻击者判断某样本是否曾属于训练集的隐私攻击方法，MIA 得分接近 50% 表示模型无法区分成员与非成员样本。
- **Semi Definite Program（SDP，半定规划）**：带有矩阵正半定约束的凸优化问题，本文用于求解估计 Hessian 的正定性约束。

## 可复现要素
- **数据集**：CIFAR-10、CIFAR-100、Stanford Dogs、CalTech-256（均为公开数据集）。
- **代码/权重**：论文未明确声明代码是否开源。
- **关键超参**：扰动数量（实验测试 250/500/1000）、L2 正则化系数 $\lambda$（实验测试 0/0.0005/0.001）、遗忘数据比例（5%/10%/15%/20%）、ResNet-18 骨干网络（ImageNet 预训练）。
- **硬件**：单张 NVIDIA RTX 3090 GPU。
