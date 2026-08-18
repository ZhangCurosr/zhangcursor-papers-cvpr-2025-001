---
title: "Brain-Inspired-Spiking-Neural-Networks-for-Energy-Efficient"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Li_Brain-Inspired_Spiking_Neural_Networks_for_Energy-Efficient_Object_Detection_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:56:55"
field: "脉冲神经网络目标检测"
keywords: ["Spiking Neural Network", "Object Detection", "Energy Efficiency", "Direct Training", "Multi-scale Fusion", "Event Camera"]
innovations: ["提出全脉冲架构的MSD框架，直接训练避免ANN转换损失", "设计SCN和ONNB模块，实现高效深层特征提取", "多尺度脉冲检测框架MSDF降低能耗82.9%"]
benchmarks: ["COCO 2017", "Gen1"]
---

# 论文速读：Brain-Inspired-Spiking-Neural-Networks-for-Energy-Efficient

## 一句话总结
本文提出了一种名为 **Multi-scale Spiking Detector (MSD)** 的全新目标检测框架，通过设计脉冲卷积神经元（SCN）和全脉冲网络架构，实现了直接训练的、低能耗、高性能的多尺度 SNN 目标检测器，在 COCO 和 Gen1 数据集上均取得了优于现有 ANN-to-SNN 转换方法和其他 SNN 检测器的性能。

---

## 研究问题与动机

1. **ANN-to-SNN 转换方法的局限性**：现有主流 SNN 目标检测器（如 Spiking-Yolo、Spike Calibration）大多从 ANN 转换而来，转换过程存在性能损失，且需要大量时间步（如 Spiking-Yolo 需 3500 步）才能匹配 ANN 性能。

2. **事件数据的适应性差**：ANN-to-SNN 方法的动态设计旨在近似 ANN 的期望激活值，无法有效捕获 DVS（事件相机）数据的时空信息，难以直接处理稀疏事件数据。

3. **深层特征提取的能量效率瓶颈**：现有 SNN 在进行深层特征提取时面临梯度消失/爆炸问题，且在多尺度变换过程中非脉冲卷积操作引入了额外 MAC 运算，增加了能耗。

4. **多尺度特征利用不足**：现有 SNN 目标检测模型受限于浅层架构，对多尺度特征信息的利用率较低，难以应对复杂场景下的小目标和遮挡目标。

---

## 核心贡献（创新点）

1. **提出 MSD 框架**：设计了可直接训练、低能耗、高性能的多尺度 SNN 目标检测框架，在 COCO 数据集上达到 62.0% mAP@0.5，优于现有 ANN-to-SNN 转换方法和直接训练 SNN 方法。

2. **设计脉冲卷积神经元（SCN）**：将 LIF 脉冲神经元与卷积操作结合，作为 ONNB 的核心组件，实现了全脉冲计算路径，避免了非脉冲卷积操作的 MAC 能耗开销，相比传统 ANN-to-SNN 转换方法具有更优的深层特征提取能力。

3. **构建 Optic Nerve Nucleus Block (ONNB)**：借鉴视觉皮层神经元的树突异质性，设计了旁路传输机制（shortcut path），在保留初始信息的同时通过 SCN 转换信息为稀疏脉冲，实现高效残差学习，避免了传统 ResNet 结构中非脉冲操作的能耗问题。

4. **提出多尺度脉冲检测框架（MSDF）**：通过脉冲多尺度特征融合和脉冲解耦检测头，模拟生物对不同尺度目标的响应机制，使分类和回归分支通过脉冲驱动自适应调整特征重要性，减少了能耗并提升了多尺度检测性能。

---

## 方法详解

### 1. 脉冲卷积神经元（SCN）
SCN 结合了 LIF 神经元模型和卷积操作，核心方程如下：

**膜电位更新**：
$$V^{t+1,n+1}(i) = k_{\tau} V^{t,n+1}(i)(1-o^{t,n+1}(i)) + \sum_{j=1}^{l}\omega_{ij}^n o^{t+1,n}(j)$$

**脉冲输出**：
$$o^{t+1,n+1}(i) = f(V^{t+1,n+1}(i) - V_{th})$$

其中 $f(\cdot)$ 为阶跃函数，$V_{th}$ 为阈值。为了解决反向传播中脉冲不可导的问题，采用替代梯度（surrogate gradient）方法：

$$\frac{\partial o^{t,n}(i)}{\partial V^{t,n}(i)} = \frac{1}{a}\text{Signal}(|V^{t,n}(i) - V_{th}|)$$

**tdBN 归一化**：
$$\text{tdBN}(I^{t+1}(i)) = \lambda_i \frac{\alpha V_{th}(I^{t+1}(i) - \mu_{ci})}{\sqrt{\sigma_{ci}^2 + \epsilon}} + \beta_i$$

### 2. Optic Nerve Nucleus Block (ONNB)
ONNB 的核心设计是将 LIF 激活函数应用于每个残差和旁路路径：

**残差路径**：采用两个 SCN 串联，实现脉冲特征提取
**旁路路径**：使用 maxpooling 减少参数，并通过 SCN 将信息转换为稀疏脉冲

$$\text{SN}(X) = [\text{SCN}(\text{SCN}(X)), \text{SCN}(\text{Maxpool}(X))]$$

$$\text{ONNB}(X) = \text{Conv}[\text{Conv}(X), Sp(X), \text{SN}(Sp(X)), \cdots, \text{SN}(\cdots\text{SN}(Sp(X)))]$$

其中 $Sp(\cdot)$ 表示沿通道维度的分割操作，最终将所有分支拼接后通过 Conv 层调整通道数。

### 3. 多尺度脉冲检测框架（MSDF）
- **特征金字塔融合**：利用 ONNB 学习不同深度的脉冲响应，通过 SCN 融合相邻低层的细粒度特征
- **脉冲解耦检测头**：将脉冲引入解耦头（Decouple Head），分类和回归分支通过脉冲驱动自适应调整重要性
- **无 NMS 处理**：将神经元最终膜电位输入检测器生成不同尺度的 anchors，经 NMS-free 处理后得到最终检测结果

### 4. 能耗计算
SNN 能耗定义为：
$$E = \sum_{i=1}^{n} E_i, \quad E_i = T \times (f_r \times E_{AC} \times OP_{AC} + E_{MAC} \times OP_{MAC})$$

其中 $E_{MAC} = 4.6\text{pJ}$，$E_{AC} = 0.9\text{pJ}$（45nm 工艺，32 位浮点）。

---

## 实验与结果

### 数据集
- **MS COCO 2017**：80 类，118K 训练图，5K 验证图
- **Gen1**：事件相机数据集，304×240 分辨率，25.5 万标注框，用于行人和车辆检测

### COCO 2017 结果
| 方法 | Params (M) | Power (mJ) | mAP@0.5 | mAP@0.5:0.95 |
|------|------------|------------|---------|---------------|
| SpikeYOLO [34] | 13.2 | 23.1 | 59.2 | 42.5 |
| **MSD (Ours)** | **7.8** | **6.43** | **62.0** | **45.3** |

- MSD 参数量仅为 7.8M，能耗 6.43mJ，mAP@0.5 达 62.0%，比 prior SOTA SNN 提升 **+2.8%**
- 能耗降低 **82.9%**

### Gen1 数据集结果
| 方法 | Params (M) | Power (mJ) | mAP@0.5 | mAP@0.5:0.95 |
|------|------------|------------|---------|---------------|
| SpikeYOLO [34] | 13.2 | 11.0 | 66.0 | 38.5 |
| **MSD (Ours)** | **7.8** | **6.51** | **66.3** | **38.9** |

- MSD 在相同时间步配置（T×D = 5×1）下超越 SpikeYOLO，提升 **+0.3% mAP@0.5** 和 **+0.4% mAP@0.5:0.95**
- 参数量减少 **40.9%**，能耗降低 **40.8%**

### 消融实验（Gen1 数据集）
| 方法 | Params (M) | Power (mJ) | mAP@0.5 | mAP@0.5:0.95 |
|------|------------|------------|---------|---------------|
| Baseline | 8.8 | 38.2 | 56.8 | 37.4 |
| +ONNB | 7.8 | 11.3 | 64.3 | 34.2 |
| +MSDF | 7.7 | 20.9 | 59.1 | 41.1 |
| **MSD (完整)** | **7.8** | **6.5** | **66.3** | **38.9** |

- 引入 ONNB 后 mAP@0.5 提升 **+7.5%**，参数减少 1.0M
- 完整模型相比 Baseline 能耗降低 **82.9%**，参数减少 **11.2%**

---

## 相关工作脉络

1. **Spiking-Yolo [23]**：早期 SNN 目标检测工作，采用 ANN-to-SNN 转换，需 3500 时间步，性能受限且不适合事件数据。本文通过直接训练和全脉冲架构，显著减少时间步并提升检测精度。

2. **Spike Calibration [28]**：将时间步降至数百量级，但性能受限于基线 ANN 模型。本文提出直接训练的 SCN，避免了转换过程中的性能损失。

3. **SpikeYOLO [34]**：整数训练和脉冲驱动推理方法，但参数量和能耗较高。本文通过 ONNB 和 MSDF 设计，在更少的参数和能耗下实现更高精度。

4. **EMS-YOLO [45]**：深度直接训练 SNN，但仍存在非脉冲操作引入的 MAC 能耗。本文通过全脉冲 SC Neu ron 设计，消除了这些额外开销。

5. **MSResNet [17]**：忽略了捷径路径上的非脉冲卷积操作，导致能耗不可忽视。本文在捷径路径引入 SCN，实现真正意义上的全脉冲计算。

---

## 局限性与未来方向

1. **缺乏深度与参数的消融分析**：论文未对 ONNB 和 MSDF 在不同深度和参数配置下进行更深入的消融实验，限制了对其效果的全面评估。

2. **膜电位重置机制的影响尚不明确**：重置机制对学习动态和收敛性的影响有待进一步研究。

3. **未来方向**：作者计划将脉冲神经元扩展到更多视觉任务，并解决当前局限性。

---

## 研究启发与可借鉴点

1. **全脉冲架构设计思路**：ONNB 中通过旁路路径引入 SCN 实现纯脉冲残差学习的思想，可迁移到其他 SNN 骨干网络设计中，减少非脉冲操作的能耗。

2. **tdBN 在目标检测中的适配**：将阈值依赖批归一化（tdBN）应用于目标检测任务，可作为直接训练 SNN 的有效正则化手段。

3. **事件相机数据的 SNN 处理**：本文展示了 SNN 在处理事件相机数据上的优势，结合 Gen1 数据集的实验验证，为事件视觉任务提供了可复用的框架。

4. **脉冲驱动解耦头设计**：将脉冲引入解耦检测头，使分类和回归分支通过脉冲自适应调整特征重要性，这一思路可用于其他多任务 SNN 设计。

5. **能耗评估方法的统一**：本文采用 AC/MAC 操作数统一评估 SNN 能耗，为后续工作提供了可比对的能耗计算框架。

---

## 关键术语表

**Spiking Neural Network (SNN)**：脉冲神经网络，第三代神经网络，使用二进制脉冲信号进行神经元通信，具有异步计算和事件驱动特性，能耗低。

**Leaky Integrate-and-Fire (LIF)**：漏积分脉冲神经元模型，通过膜电位累积输入并超过阈值后产生脉冲，是 SNN 中最常用的神经元模型。

**Surrogate Gradient**：替代梯度，用于解决脉冲函数不可导问题，通过可导函数近似脉冲函数的梯度以实现反向传播。

**ANN-to-SNN Conversion**：人工神经网络到脉冲神经网络的转换方法，通过将 SNN 的平均脉冲率近似 ANN 的连续激活值来部署 SNN。

**tdBN (Threshold-dependent Batch Normalization)**：阈值依赖批归一化，同时考虑空间和时间的批归一化方法，适用于直接训练的 SNN。

**STBP (Spatio-Temporal Backpropagation)**：时空反向传播，考虑 SNN 空间和时间依赖关系的训练框架。

**ONNB (Optic Nerve Nucleus Block)**：视神经核块，受生物视觉皮层神经元异质性启发的模块，通过旁路传输和 SCN 实现高效脉冲特征提取。

**MSDF (Multi-scale Spiking Detection Framework)**：多尺度脉冲检测框架，通过脉冲多尺度特征融合和脉冲解耦检测头实现高效目标检测。

---

## 可复现要素

- **数据集**：MS COCO 2017（公开）、Gen1（公开）
- **代码开源情况**：论文未提及代码是否开源
- **权重开源情况**：论文未提及权重是否开源
- **关键超参**：
  - $V_{reset} = 0$
  - $\tau = 0.25$
  - $V_{th} = 0.5$
  - $\alpha = 1$
  - 学习率：0.01（cosine decay 至 0.0001）
  - batch size：32
  - 训练 epoch：300（COCO）、50（Gen1）
  - 输入分辨率：640×640
  - 时间步 T=5，time bin d=1（Gen1）
  - 优化器：SGD，momentum=0.9
  - Mosaic augmentation：概率 0.5
