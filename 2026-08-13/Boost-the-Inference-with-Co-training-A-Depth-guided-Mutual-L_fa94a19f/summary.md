---
title: "Boost-the-Inference-with-Co-training-A-Depth-guided-Mutual-L"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Li_Boost_the_Inference_with_Co-training_A_Depth-guided_Mutual_Learning_Framework_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:57:02"
field: "半监督医学图像分割"
keywords: ["半监督分割", "息肉分割", "RGB-D 融合", "跨模态学习", "医学图像分析"]
innovations: ["训练期跨模态互学习、推理期仅用RGB的解耦框架", "基于深度方图的难样本Patch增强方法", "输出级对比学习促进RGB与深度互补知识获取"]
benchmarks: ["Kvasir-SEG", "CVC-ClinicDB", "CVC-300", "CVC-ColonDB", "ETIS-Larib"]
---

# 论文速读：Boost-the-Inference-with-Co-training-A-Depth-guided-Mutual-Learning-Framework

## 一句话总结
本文提出 RD-Net，一种基于 mean teacher 的半监督息肉分割框架，通过引入深度输入辅助学生网络与跨模态互学习策略，在训练阶段充分利用 RGB 与深度图的互补信息，且在推理阶段仅需 RGB 图像即可达到 SOTA 性能。

## 研究问题与动机
1. **现有 RGB-D 分割方法在推理阶段依赖深度数据**：临床场景中深度图获取困难或不一致，限制了模型的部署应用。
2. **半监督医学图像分割中未充分利用空间深度信息**：现有 SSL 方法多仅依赖 RGB 图像，忽略了深度图像提供的边界与轮廓信息。
3. **跨模态特征融合复杂且难以在推理时移除深度输入**：如 ACNet、CMX 等方法依赖编码器阶段的深度拼接与融合，无法支持模态缺失场景。
4. **低标注比例下模型对困难区域的识别能力不足**：PH-Net 等基于预测熵的难样本识别方法在标注极少时置信度低，易产生误判。

## 核心贡献（创新点）
1. **提出基于 mean teacher 的半监督息肉分割框架 RD-Net，推理阶段无需深度输入**：通过引入辅助深度学生网络 Model D，在训练阶段实现跨模态互补学习，而推理时仅使用 RGB 输入的主学生网络 Model R1。
2. **设计深度引导跨模态互学习策略（Depth-guided Cross-modal Mutual Learning）**：从一致性学习（$\mathcal{L}_{con}$）、对比学习（$\mathcal{L}_{cont}$）和辅助网络指导学习（$\mathcal{L}_{ud}$）三个维度促进不同学生网络的互补知识获取。
3. **提出深度引导 Patch 增强方法（Depth-guided Patch Augmentation, DPA）**：利用深度图方差计算 patch 难度评分 $H_P$，替代基于模型预测熵的难样本识别，在少标注场景下更可靠地引导困难区域学习。

## 方法详解
**框架结构**：基于 Mean Teacher 架构，包含 Teacher 网络 Model R2（EMA 更新）、主学生网络 Model R1（RGB 输入）和辅助学生网络 Model D（深度输入），三者均采用 UNet 架构（ResNet34 backbone）。

**深度图生成**：使用 Depth Anything 模型从 RGB 图像推断单通道深度图，并通过色彩映射转换为三通道深度图像。

**监督损失**（公式 1）：
$$\mathcal{L}_{sup} = \frac{1}{2|\mathcal{D}_l|} \sum \left( \mathcal{L}_{ce}(\tilde{Y}_i^l, Y_i) + \mathcal{L}_{dice}(\tilde{Y}_i^l, Y_i) \right)$$

**无监督损失**（公式 2-3）：基于教师网络预测 $\tilde{Y}_i$ 作为伪标签，通过置信度阈值 $\gamma$ 过滤不可靠像素：
$$\mathcal{L}_u = \frac{1}{2|\mathcal{D}_u|} \sum \left( \mathcal{L}_{ce} + \mathcal{L}_{dice} \right) \cdot \mathbb{1}[\tilde{Y}_i \geq \gamma \text{ or } \tilde{Y}_i \leq 1-\gamma]$$

**一致性学习**（公式 5）：约束两个学生网络的最后层编码器特征对齐与强增强输出的预测一致性：
$$\mathcal{L}_{con} = \frac{1}{2|\mathcal{D}_u|} \sum \left( \|F_i^{r1} - F_i^d\|_2^2 + \|\mathcal{F}_{\theta_{r1}}(X_i^{S1}) - \mathcal{F}_{\theta_d}(Z_i)\|_2^2 \right)$$

**对比学习损失**（公式 6-11）：以 RGB 和深度网络的预测输出作为正样本对，利用 Dice 系数度量相似度，构建 InfoNCE 风格的对比损失 $\mathcal{L}_{cont}$，促进跨模态互补知识学习。

**辅助学生网络指导学习**（公式 12-13）：若 Model D 对某像素的预测置信度高于阈值 $\gamma$ 且比 Model R2 的预测更接近真实类别（距离更小），则用 D 的预测替换 R2 的伪标签，形成更新后的伪标签 $Y''_i$。

**深度引导 Patch 增强**（公式 14-17）：
- 难度评分：$H_P = \frac{1}{h \times w} \sum_{(i,j) \in P} \text{Var}(D_{i,j})$
- 按难度降序排列 patch，高难度 patch 优先屏蔽，结合 CutMix 操作生成增强样本。

**总损失**（公式 20）：
$$\mathcal{L}_{total} = \mathcal{L}_{rgb} + \mathcal{L}_{depth} + \mathcal{L}_{cont} + \mathcal{L}_{con}$$
其中 $\mathcal{L}_{rgb}$ 包含监督损失、DPA 损失和 $\mathcal{L}_{ud}$。

## 实验与结果
**数据集**：Kvasir-SEG（1000 张）、CVC-ClinicDB（612 张）、CVC-300、CVC-ColonDB、ETIS-Larib。评估指标为 Dice、IoU、Accuracy。

**主要结果**（Kvasir-SEG 2.5% 标注）：Dice 86.35%、IoU 80.10%、Acc 96.05%，较前作 CML 提升 Dice +3.14%、IoU +3.51%。

**主要结果**（Kvasir-SEG 10% 标注）：Dice 89.07%、IoU 84.21%、Acc 97.07%，接近全监督（Dice 89.43%）。

**泛化测试**（CVC-300/CVC-ColonDB/ETIS-Larib，1/10 标注）：在 ETIS-Larib 上 Dice 达 71.47%，较 SupOnly（43.30%）提升 +28.17%；在 CVC-ColonDB 上 Dice 达 74.29%，较 SupOnly（63.85%）提升 +10.44%。

**消融实验**（10% 标注，Kvasir-SEG）：完整模型 Dice 89.07%，对比仅监督 baseline（82.76%）提升 +6.31%；DPA 模块贡献约 +1% Dice，互学习策略贡献最大（+5.38%）。

## 相关工作脉络
1. **Mean Teacher (MT)**：本文继承其教师-学生一致性框架，扩展为双学生（RGB + Depth）架构，并引入跨模态互学习。
2. **ACNet / CMX**：前者依赖注意力机制融合双模态特征，后者通过跨模态特征校正模块融合，均需在推理时输入深度图；本文通过分离学生网络实现训练期跨模态学习、推理期免深度。
3. **Mask-Mentor**：采用 token-pixel 联合重建缩小完整/缺失模态表征差距，但模态缺失时性能下降；本文不依赖特征级融合，可天然支持推理时模态缺失。
4. **PH-Net**：基于模型预测熵评估 patch 难度，在低标注量下置信度不稳定；本文用深度方差替代，更可靠地识别困难区域。
5. **ACL-Net / SCP-Net / BCP / AD-MT**：均为半监督医学分割 SOTA 方法，本文在相同实验设置下全面超越。

## 局限性与未来方向
1. **深度图质量依赖预训练模型**：当前使用 Depth Anything 生成伪深度图，若输入图像存在严重反光或遮挡，深度估计误差可能传播至分割结果。
2. **双学生网络增加了训练计算开销**：虽然推理时无额外负担，但训练阶段需维护三个 UNet 网络，显存占用较高。
3. **实验集中于结肠息肉分割**：方法的有效性和泛化性尚未在其它类型医学图像（如 CT、MRI）或其他器官分割任务中验证。
4. **置信度阈值 $\gamma$ 为固定超参**：不同数据集和标注比例可能需要调优，缺乏自适应机制。

## 研究启发与可借鉴点
1. **"训练期多模态、推理期单模态"的解耦思路**：通过引入辅助学生网络并使用独立损失约束，可实现训练时跨模态知识迁移、推理时单一模态部署，适用于其他模态缺失场景（如超声缺失、红外缺失）。
2. **基于物理信息（深度方差）的难样本识别**：相比基于模型预测熵的方法，利用场景先验（几何结构）识别困难区域更稳定，可迁移至其他少标注医学分割任务。
3. **输出级对比学习避免特征对齐困难**：跨模态对比学习直接在预测输出层面构建正负样本对，绕过不同模态特征空间的不对齐问题，设计简洁有效。
4. **伪标签交叉修正策略**：用辅助网络的确定性预测替换主网络的不确定预测，是一种轻量级的"专家投票"机制，可推广至多教师半监督框架。

## 关键术语表
**Mean Teacher (MT)**：半监督学习经典框架，通过指数移动平均（EMA）维护教师网络权重，利用教师预测作为伪标签引导学生一致性学习。

**Cross-modal Mutual Learning**：不同模态网络之间相互学习互补信息，本文通过一致性约束、对比学习和伪标签替换三个层面实现。

**Depth-guided Patch Augmentation (DPA)**：利用深度图方差评估图像 patch 难度，高难度 patch 优先被 CutMix 操作覆盖，引导模型关注困难区域。

**Confidence Threshold ($\gamma$)**：用于过滤不可靠伪标签的置信度阈值，实验表明 $\gamma=0.85$ 在 Kvasir-SEG 上效果最佳。

**Depth Anything**：开源的单目深度估计大模型，本文用于从 RGB 图像生成与原始图像同尺寸的深度图。

## 可复现要素
- **数据集**：Kvasir-SEG、CVC-ClinicDB、CVC-300、CVC-ColonDB、ETIS-Larib（公开数据集）
- **代码**：https://github.com/pingchuan/RD-Net
- **模型权重**：论文未明确声明是否开源，代码仓库中可能包含
- **关键超参**：$\gamma=0.85$，EMA 动量 $\alpha=0.99$，初始学习率 0.001，batch size=4（标注/未标注各 2），训练 50000 次，图像尺寸 320×320，SGD 优化器，动量 0.9，权重衰减 0.00001
- **骨干网络**：UNet + ResNet34（ImageNet 预训练）
- **深度估计**：Depth Anything 预训练模型
