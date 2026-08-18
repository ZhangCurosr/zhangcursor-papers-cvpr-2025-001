---
title: "DELT-A-Simple-Diversity-driven-EarlyLate-Training-for-Datase"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Shen_DELT_A_Simple_Diversity-driven_EarlyLate_Training_for_Dataset_Distillation_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:59:50"
field: "数据压缩与高效学习"
keywords: ["数据集蒸馏", "批到全局匹配", "图像多样性", "EarlyLate训练", "BN正则化", "大尺度图像分类"]
innovations: ["EarlyLate训练策略：通过差异化迭代次数实现类内多样性增强", "教师模型中位概率排序的真实图像块初始化", "单次扫描拼接渐进训练，计算效率提升约39.3%"]
benchmarks: ["ImageNet-1K", "CIFAR-10", "Tiny-ImageNet", "ImageNette", "ImageNet-100"]
---

# 论文速读：DELT: A Simple Diversity-driven EarlyLate Training for Dataset Distillation

## 一句话总结
DELT 提出了一种**EarlyLate 训练策略**，通过在批次到全局匹配（batch-to-global matching）的图像蒸馏过程中让不同子批次的合成图像经历不同数量的优化迭代（从最多到最少），从而显著提升类内多样性并降低计算成本，同时在 CIFAR-10、Tiny-ImageNet 和 ImageNet-1K 等数据集上均刷新了数据集蒸馏的 SOTA 精度。

## 研究问题与动机
- **类内多样性不足**：现有批到全局匹配方法（如 SRe²L、CDA、WMDD、G-VBSM）对每个类别内的合成样本独立优化，并反复使用相同的全局监督信号（如 BN 统计量），导致同类别内生成的图像语义过于相近，多样性严重受限。
- **计算开销大**：传统 batch-to-global 方法需要对 N×T 次迭代独立优化每个样本，随着 IPC 增大计算成本急剧上升。
- **已有多样性方法各有缺陷**：G-VBSM 需依赖多个 backbone 与统计匹配，框架复杂；RDED 的拼贴方式依赖原始数据集本身的分布，未对合成内容的信息量做优化。
- **初始化质量未被充分挖掘**：现有大尺度蒸馏方法默认使用高斯噪声初始化，缺乏语义引导，影响了合成图像的最终质量和收敛效率。

## 核心贡献（创新点）
- **EarlyLate 训练策略**：将 IPC 样本拆分为若干子批次，第一个批次经历最大迭代数 MI，后续批次依次递减 RI，最后批次仅用极少迭代，形成"早训 + 晚训"的梯度覆盖，本质区别在于不额外增加匹配信号来源，而是通过差异化训练时长引入多样性。
- **教师模型驱动的真实图像块初始化**：用预训练教师模型对所有真实图像块打分排序，选取每类中位概率的图像块作为合成起点，区别于以往的高斯噪声或最高/最低概率初始化策略。
- **单次扫描的拼接渐进训练（Concatenation Training）**：不同子批次的优化通过拼接在同一批数据上完成，共享 GPU 恢复时间，无需冻结前序批次，整体迭代总量约为传统方法的 2/3，计算效率显著优于 SRe²L/CDA。
- **跨架构泛化与下游应用验证**：不仅验证了 ImageNet-1K 上的精度提升，还展示了在数据无关网络剪枝、持续学习等下游任务中的有效迁移能力。

## 方法详解

### 1. 问题设定
给定教师数据集 $\mathcal{T}$ 和目标学生合成集 $S$（每类 IPC 个样本），目标是最小化：
$$\arg \min \left( \sup \left\{ \ell(\phi_{\theta_\mathcal{T}}(x_{val}), y_{val}) - \ell(\phi_{\theta_S}(x_{val}), y_{val}) \right\}_{(x_{val}, y_{val}) \sim V} \right)$$

### 2. 教师排名器初始化（Teacher-rank Initialization）
- 使用预训练的 ResNet-18 作为教师模型，提取所有真实图像块（patches）。
- 计算每类图像块的预测概率，并按中位概率筛选作为合成图像的初始化起点。
- 动机：中等难度样本具有最大的"可提升空间"，既不过难也无法直接复用。

### 3. EarlyLate 训练机制
将目标 IPC 总数拆成 $M$ 个批次，每批次 $k$ 个图像：
- **Round 1**：第一批 $IPC_{0:k-1}$ 执行 MI 轮梯度更新。
- **Round 2**：加入第二批，第一批继续优化，第二批执行 MI−RI 轮。
- **Round M**：所有批次共同优化，最后一批仅执行 RI 轮。
- 总迭代量约等于 $N \times T - \frac{j(j-1)}{2} \text{RI}$，约为传统方法 2/3。

### 4. BatchNorm 分布正则化
沿用 SRe²L 的 BN 统计量正则项 $\mathcal{R}_{reg}$，保证生成图像的统计分布与真实数据保持一致。

### 5. 优化公式
$$\text{Round } m: \arg \min_{\mathcal{C}_{IPC_{0:mk-1}}, |\mathcal{C}|} \ell\left(\phi_{\theta_\mathcal{T}}(\tilde{x}_{IPC_{0:mk-1}}), \pmb{y}\right) + \mathcal{R}_{reg}$$

## 实验与结果

| 数据集 | 架构 | IPC | 对比基线 | DELT 准确率 | 提升幅度 |
|--------|------|-----|----------|-------------|----------|
| ImageNet-1K | ResNet-101 | 50 | RDED 61.2% | **66.1%** | **+4.9%** |
| ImageNet-1K | MobileNet-v2 | 50 | RDED 52.8% | **56.2%** | +3.4% |
| CIFAR-10 | ResNet-101 | 50 | RDED 51.6% | **54.1%** | +2.5% |
| CIFAR-10 | ResNet-18 | 1 | RDED 22.9% | **24.0%** | +1.1% |
| Tiny-ImageNet | ResNet-101 | 10 | RDED 22.9% | **42.8%** | +19.9% |

**关键结论**：
- 在 15 个配置中 **13 个达到 SOTA**，尤其在 ImageNet-1K 和 CIFAR-10 上优势明显。
- **计算效率**：ImageNet-1K 上相比 CDA/SRe²L 节省 **39.3%** 时间（RI=500），Tiny-ImageNet 节省 32%，CIFAR-10 无显著变化（数据加载占主导）。
- **类内多样性**：平均余弦相似度显著低于 SRe²L 和 CDA，可视化显示生成图像涵盖不同清晰度与风格。

## 相关工作脉络
- **SRe²L（NeurIPS 2023）**：批到全局匹配的代表性工作，利用 BN 统计量匹配；DELT 在同框架下引入差异化迭代，是正交改进。
- **CDA（TMLR 2024）**：课程式蒸馏；DELT 明确说明其 EarlyLate 与 CDA 的 Early-only（等价于 CDA+真实初始化）对比，证明多样性来源的关键性。
- **G-VBSM（CVPR 2024）**：多 backbone 统计匹配；DELT 以单 backbone 达到更高精度，复杂度更低。
- **RDED（CVPR 2024）**：拼贴式多样性增强；DELT 通过优化驱动而非拼接驱动获得更高表达能力。
- **MTT / TESLA / IDM / DATM**：批到批匹配的梯度或轨迹匹配方法；DELT 定位在于大尺度场景的解耦蒸馏方向。
- **MinimaxDiffusion（CVPR 2024）**：扩散模型生成蒸馏数据；DELT 在消融中表明其梯度优化方法优于纯扩散生成。

## 局限性与未来方向
- **IPC=1 不适用**：EarlyLate 策略要求多批次迭代差异，单样本每类无意义。
- **仍为优化密集型**：虽然相比基线节省约 1/3 计算，但整体仍需数小时甚至数十小时（ImageNet-1K IPC=50 约 17.6 小时），对消费级设备不够友好。
- **教师模型依赖**：初始化步骤需要预训练教师模型，对无预训练资源的场景构成门槛。
- **仅针对图像任务**：目前仅在 CIFAR/Tiny-ImageNet/ImageNet 系列验证，未扩展到视频、点云或跨模态蒸馏。
- **Mosaic 拼贴未采用**：作者发现 1×1 单块初始化最优，更大的拼接反而引入噪声；这说明多样性增强策略仍有权衡空间。

## 研究启发与可借鉴点
- **"差异化训练时长"作为多样性正则化**：不改变优化目标，仅通过控制每个样本的优化步数即可打破同质化，思路简洁且可迁移到其他生成优化任务。
- **教师模型排序作为数据选择先验**：用预训练模型的置信度分布指导初始化选择（中位概率最优），可推广到主动学习、课程学习等场景。
- **计算-精度联合优化的新思路**：将总迭代预算分配为递减序列，在不增加总预算的前提下换取多样性收益，为资源受限蒸馏提供了实用范式。
- **下游任务验证的价值**：除主指标外，论文还展示了剪枝、持续学习的应用，增强了工作的说服力，建议在后续研究中同样采用多任务评估。

## 关键术语表
- **数据集蒸馏（Dataset Distillation）**：将大规模原始数据集压缩为一个小型合成数据集，使得在该合成集上训练的模型在原始测试集上能达到相近的性能。
- **批到全局匹配（Batch-to-Global Matching）**：用全局统计量（如 BN 均值/方差）代替批内逐样本匹配，从而实现合成阶段不重读原始数据，适合大尺度数据集。
- **EarlyLate 训练**：将合成图像分批并以递减的优化迭代次数依次加入训练，使不同样本最终处于不同优化深度，从而产生类内多样性。
- **MI（Maximum Iteration）**：第一批合成图像经历的最大优化迭代数，决定"早训"样本的信息量上限。
- **RI（Round Iteration）**：相邻批次之间的迭代差值，控制"晚训"样本的优化深度。
- **BN 正则化（BatchNorm Regularization）**：利用真实数据的 BN 统计量约束生成图像的分布，是 SRe²L 类方法的核心组件。
- **IPC（Images Per Class）**：每个类别分配给合成数据集的样本数量，是衡量蒸馏压缩率的关键超参数。
- **数据无关网络剪枝（Data-free Network Pruning）**：无需原始训练数据即可对模型进行剪枝，蒸馏数据可作为替代训练样本。

## 可复现要素
- **数据集**：CIFAR-10、Tiny-ImageNet、ImageNet-1K、ImageNette、ImageNet-100（均为公开数据集）。
- **代码开源情况**：论文未明确声明代码仓库 URL，需联系作者或关注 VILA Lab 官方页面。
- **关键超参**：MI=4000，RI=500（或 1000），M=4~8（批次拆分数量），初始化采用 1×1 单块 patch。
- **硬件**：4× NVIDIA RTX 4090 GPU（论文声明）。
- **骨干网络**：ConvNet、ResNet-18、ResNet-101、MobileNet-V2、EfficientNet-B0、MnasNet1_3、RegNet-Y-8GF。
