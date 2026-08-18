---
title: "RoboBrain-A-Unified-Brain-Model-for-Robotic-Manipulation-fro"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Ji_RoboBrain_A_Unified_Brain_Model_for_Robotic_Manipulation_from_Abstract_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 07:48:45"
---

# 论文速读：RoboBrain-A-Unified-Brain-Model-for-Robotic-Manipulation-fro

## 一句话总结
本文提出 **RoboBrain**，一个面向机器人长周期操纵任务的多模态大语言模型，并配套构建高质量细粒度数据集 **ShareRobot**；通过两阶段多步训练与 A-LoRA/T-LoRA 模块化设计，使模型能将抽象指令转化为可执行的原子规划、物体交互区域（Affordance）与末端轨迹，在多项机器人操作基准上取得 SOTA。

## 研究问题与动机
- 现有 MLLM 在机器人场景（尤其长周期操纵任务）中暴露出三大能力缺失：任务规划（Planning）、物体功能感知（Affordance Perception）与末端轨迹预测（Trajectory Prediction）。
- 当前机器人 MLLM 多聚焦于自然语言理解或直接生成动作序列，缺乏将高层指令逐步拆解为可执行原子步骤，并输出具体空间交互区域与运动路径的机制。
- 公开机器人演示数据集（如 Open X-Embodiment）多为高层任务描述，缺乏与具体帧对齐的低层规划指令、affordance 边界框与轨迹坐标等细粒度监督信号。
- 直接将通用 MLLM 迁移至机器人领域易引发灾难性遗忘，需在数据配比、训练阶段划分与参数更新策略上进行系统性设计。

## 核心贡献（创新点）
- **提出 RoboBrain 统一框架**，首次在单个 MLLM 中联合优化规划文本生成、Affordance 区域定位与 2D 轨迹预测，实现从抽象指令到具体操作行为的闭环映射。
- **构建 ShareRobot 数据集**，从 Open X-Embodiment 中严格筛选 51,403 条高质量实例，生成 1,027,990 条规划 QA 对，并人工标注 6,522 张 affordance 框与 6,870 张轨迹关键点，覆盖 102 场景、12 种机器人形态与 107 类原子任务。
- **设计两阶段多步骤训练策略**：Phase 1 基于 LLaVA-OneVision 构建通用视觉-语言底座；Phase 2 通过 Stage 3（全量微调防遗忘）与 Stage 4（A-LoRA/T-LoRA 专项注入）实现能力升级。
- **实验验证全面领先**：在 RoboVQA、OpenEQA、ShareRobot 测试集及 AGD20K 上均 surpass 现有基线，规划 BLEU-4 超第二名 18.75，Affordance AP 达 27.1%，轨迹误差 DFD/HD/RMSE 分别下降 42.9%/94.2%/31.6%。

## 方法详解
- **模型架构**：基于 LLaVA 设计，视觉编码器采用 SigLIP，投影层为 2 层 MLP，LLM Backbone 为 Qwen2.5-7B-Instruct。包含三个子模块：规划基础模型、A-LoRA（Affordance）、T-LoRA（Trajectory）。推理时先生成详细规划文本，再拆解为子任务驱动 affordance 与轨迹生成。
- **ShareRobot 数据标注流程**：
  - *Planning*：每段演示抽取 30 帧，结合高层描述用 Gemini 分解为低层规划指令，3 名标注员人工复核；针对 RoboVQA 10 种题型设计 5 种模板，随机组合生成 QA 对，最终产出 1,027,990 对。
  - *Affordance*：筛选 6,522 张图，按指令标注人手接触区域边界框 $\{l^{(x)}, l^{(y)}, r^{(x)}, r^{(y)}\}$，人工精修确保与指令精确对齐。
  - *Trajectory*：筛选 6,870 张图，按低层指令标注末端至少 3 个 $\{x, y\}$ 关键点，人工校验一致性。
- **多阶段训练（Phase 1 & 2）**：
  - *Stage 1-2（通用基础）*：Stage 1 用 LCS-558K 训练 Projector；Stage 1.5 用 4M 高质量图文全量微调；Stage 2 用 LLaVA-OneVision-Data（3.2M 单图 + 1.6M 图视频）强化高分辨率与长视频理解。
  - *Stage 3（规划强化）*：混合 1.3M 机器人数据（RoboVQA-800K、ScanView-318K、3RScan-43K、ScanQA-25K、SQA3d-26K、ShareRobot-200K）与 1.7M 高质量通用图文，全量微调以缓解灾难性遗忘。
  - *Stage 4（技能专精）*：冻结主干，分别接入 A-LoRA（Affordance 数据）与 T-LoRA（Trajectory 数据）进行轻量专项微调。
- **轨迹预测结构化增强**：Ablation 表明，加入起点坐标（Start Points）可校正平移偏移；限制最大关键点数量（Max Points）与引入特殊 token（Spec Token & End Points
