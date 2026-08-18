---
title: "Bridging-Past-and-Future-End-to-End-Autonomous-Driving-with"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_Bridging_Past_and_Future_End-to-End_Autonomous_Driving_with_Historical_Prediction_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:57:21"
field: "端到端自动驾驶"
keywords: ["end-to-end autonomous driving", "historical prediction", "multi-step query", "motion planning", "closed-loop evaluation", "nuScenes"]
innovations: ["将运动与规划查询重构为多步张量以支持步骤级历史交互", "提出历史 Mot2Det 融合与步级 Mot2Plan 交互模块分层利用历史信息", "在开环与闭环（NeuroNCAP）评测上同时取得 SOTA"]
benchmarks: ["nuScenes open-loop", "NeuroNCAP closed-loop"]
---

# 论文速读：Bridging-Past-and-Future-End-to-End-Autonomous-Driving-with-Historical-Prediction-and-Planning

## 一句话总结
本文提出 BridgeAD，通过将运动与规划查询重构为**多步查询**并引入 FIFO 历史记忆队列，在感知、预测、规划各阶段分层聚合历史信息，桥接过去与未来，在 nuScenes 开环与 NeuroNCAP 闭环评测上均取得 SOTA。

## 研究问题与动机
1. 现有端到端方法对历史信息的利用继承自检测范式：密集方法仅在感知模块聚合历史 BEV 特征，**忽略了运动规划对时序信息的需求**。
2. 稀疏方法通过查询稀疏记忆库交互历史信息，但**每个查询仅对应一条轨迹实例**，无法匹配运动规划需覆盖多个未来时间步的多步特性。
3. 连续驾驶场景中，缺乏对历史预测/规划信息的系统性利用，导致跨帧规划一致性差、安全关键场景下碰撞率高。

## 核心贡献（创新点）
1. **多步查询表示**：将运动/规划查询从单步扩展为多步张量 $Q_{\text{mot}} \in \mathbb{R}^{N_a \times M_{\text{mot}} \times T_{\text{mot}} \times C}$，使每个未来时间步可独立与历史信息交互，区别于 UniAD/VAD 的单步查询设计。
2. **历史 Mot2Det 融合模块**：从 FIFO 队列提取当前帧历史运动查询与对象查询做交叉注意力，将历史预测注入感知，区别于 SparseDrive 仅在规划阶段用历史。
3. **历史增强运动预测与规划模块**：通过三步注意力（交叉注意力 + 步级自注意力 + 模式级自注意力）聚合历史信息，并将历史传播至所有时间步与模式，区别于现有方法对历史信息粗糙的逐实例交互。
4. **步级 Mot2Plan 交互模块**：选择概率最高模式的历史运动查询与规划查询做步级交叉注意力，统一周围智能体预测与自车规划的时序一致性，为端到端系统首次引入该机制。
5. **开环与闭环联合评估**：在 nuScenes 开环与 NeuroNCAP 闭环安全关键场景下均验证，证明历史信息提升连续驾驶的规划连续性。

## 方法详解
1. **多步查询与 FIFO 缓存**：运动查询 $Q_{\text{mot}} \in \mathbb{R}^{N_a \times M_{\text{mot}} \times T_{\text{mot}} \times C}$，规划查询 $Q_{\text{plan}} \in \mathbb{R}^{M_{\text{plan}} \times T_{\text{plan}} \times C}$；过去 $K$ 帧存入先进先出记忆队列。
2. **历史 Mot2Det 融合**：从队列提取当前帧运动查询 $Q_{\text{m2d}} \in \mathbb{R}^{N_a \times K \times C}$，与对象查询交互：$Q_{\text{obj}} = \text{CrossAttn}(Q_{\text{obj}}, Q_{\text{m2d}})$，再经检测解码器输出分类与边界框。
3. **历史增强运动预测**：提取未来 $T_{\text{m2m}}$ 步历史运动查询 $Q_{\text{m2m}} \in \mathbb{R}^{N_a \times M_{\text{mot}} \times K \times T_{\text{m2m}} \times C}$，依次应用：
   - 交叉注意力 $\text{CrossAttn}(Q_{\text{mot}}, Q_{\text{m2m}})$
   - 步级自注意力 $\text{StepSelfAttn}(Q_{\text{mot}})$
   - 模式级自注意力 $\text{ModeSelfAttn}(Q_{\text{mot}})$
4. **历史增强规划**：类似地提取 $Q_{\text{p2p}} \in \mathbb{R}^{M_{\text{plan}} \times K \times T_{\text{p2p}} \times C}$，应用相同三步注意力。
5. **步级 Mot2Plan 交互**：按预测得分选出最优模式 $Q_{\text{mot}}^* \in \mathbb{R}^{N_a \times T_{\text{plan}} \times C}$，与规划查询做 $\text{CrossAttn}(Q_{\text{plan}}, Q_{\text{mot}}^*)$，再输出最终轨迹。
6. **损失函数**：$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{det}} + \mathcal{L}_{\text{map}} + \mathcal{L}_{\text{mot}} + \mathcal{L}_{\text{plan}}$，回归用 L1，分类用 Focal loss；多模态任务采用 winner-takes-all 策略。

## 实验与结果
- **数据集**：nuScenes（开环），NeuroNCAP 闭环仿真器。
- **开环规划（Table 1）**：BridgeAD-S L2 Avg 0.59m / Col 0.09%；BridgeAD-B L2 Avg 0.58m / Col 0.08%，优于 SparseDrive（0.61m / 0.10%）与 UniAD（0.73m / 0.61%）。
- **闭环（Table 2）**：BridgeAD-S 无后处理 Score 1.52、Col Rate 76.2%；较 SparseDrive 提升 65%，较 UniAD 碰撞率降低 12.4%。
- **检测（Table 4a）**：BridgeAD-B mAP 0.507、NDS 0.594（ResNet-101）。
- **跟踪（Table 4b）**：BridgeAD-B AMOTA 0.512、AMOTP 1.080、IDS 544。
- **预测（Table 3）**：BridgeAD-B ADE 0.60/0.70（Car/Ped）、FDE 0.96/0.98。
- **效率**：BridgeAD-S 推理延迟 157.2ms（FPS 5.0），快于 VAD（224.3ms）与 UniAD（555.6ms）。

## 相关工作脉络
1. **UniAD（CVPR 2023）**：统一查询端到端框架，但未在运动规划中系统性整合历史信息，本文在其基础上引入多步历史交互。
2. **VAD（ICCV 2023）**：向量化场景表示简化管线，规划效率优但缺乏连续驾驶中的时序记忆，本文补齐该短板。
3. **SparseDrive（arXiv 2024）**：稀疏场景表示+平行规划结构，仅感知阶段用历史，规划阶段未利用，本文将其扩展至规划全周期。
4. **BEVFormer（ECCV 2022）**：密集 BEV 时序聚合代表，历史信息仅限感知，本文提出稀疏查询层面的多步历史信息利用。
5. **ViP3D（CVPR 2023）**：联合 tracking 与 prediction，但未融入端到端规划，本文将其历史预测能力引入统一规划框架。

## 局限性与未来方向
1. 实验仅在 nuScenes 与 NeuroNCAP 上进行，**跨数据集泛化性**未验证。
2. 多步查询引入额外计算开销，在更高分辨率或更长预测 horizon 下**效率与显存压力**待进一步优化。
3. 闭环测试基于模拟器，**实车部署表现**需后续验证。
4. 历史窗口固定为 $K=3$ 帧，**更长时序依赖**的建模潜力未充分挖掘。

## 研究启发与可借鉴点
1. **多步查询范式**：将单步实例查询扩展为多步张量，可迁移至其他需多步决策的任务（如机器人控制、视频理解）。
2. **分层历史信息利用**：在不同子模块（感知/预测/规划）针对性地接入不同历史粒度，避免全量冗余，值得在 multi-task 框架中复用。
3. **步级+模式级自注意力组合**：两步自注意力实现历史信息的跨步/跨模传播，结构简洁且效果显著，可借鉴于多模态预测任务。
4. **开环+闭环联合评估**：闭环安全关键场景的定量评估补充了开环指标的不足，为端到端自动驾驶研究提供了更全面的验证范式。

## 关键术语表
**BEV（Bird's-Eye-View）**：将多视角相机图像投影至鸟瞰图空间的统一表示，常用于 3D 检测与时序聚合。  
**L2 Displacement Error**：轨迹预测评估指标，计算预测轨迹与 GT 轨迹各时间步欧氏距离的平均值。  
**NeuroNCAP**：基于 nuScenes 的 photorealistic 闭环仿真框架，提供安全关键场景用于连续驾驶评估。  
**Winner-takes-all**：多模态预测/规划策略，选取得分最高模式作为最终输出，其余丢弃。  
**AMOTA / AMOTP**：多目标跟踪评估指标，分别衡量跟踪准确性（准确率+召回率）与定位精度。  
**NDS（NuScenes Detection Score）**：综合 mAP 与校准度的检测综合评分，nuScenes 官方主要指标。  
**FIFO Memory Queue**：先进先出记忆队列，缓存过去 $K$ 帧的运动与规划查询供当前帧历史交互使用。  
**Step-Level Self-Attention**：在查询的时间步维度上施加的自注意力，实现历史信息跨时间步传播。  

## 可复现要素
- 数据集：nuScenes（公开）
- 代码：https://github.com/fudan-zvg/BridgeAD（论文已开源）
- 权重：论文未明确声明，需查看仓库 README
- 关键超参：$T_{\text{mot}}=12$，$T_{\text{plan}}=6$，$T_{\text{m2m}}=6$，$T_{\text{p2p}}=3$，$K=3$；AdamW，初始学习率 $1\times10^{-4}$，weight decay $1\times10^{-3}$；两阶段训练（感知→端到端）；8×NVIDIA RTX A6000
