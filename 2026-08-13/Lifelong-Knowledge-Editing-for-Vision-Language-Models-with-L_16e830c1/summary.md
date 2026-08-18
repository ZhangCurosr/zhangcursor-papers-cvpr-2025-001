---
title: "Lifelong-Knowledge-Editing-for-Vision-Language-Models-with-L"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Chen_Lifelong_Knowledge_Editing_for_Vision_Language_Models_with_Low-Rank_Mixture-of-Experts_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 07:33:31"
---

# 论文速读：Lifelong-Knowledge-Editing-for-Vision-Language-Models-with-L

## 一句话总结
本文提出了 **LiveEdit**，首个面向视觉语言模型（VLLM）的终身知识编辑框架，通过生成式低秩 MoE 专家与硬/软两级路由机制，解决了现有 LLM 编辑方法无法直接迁移至多模态场景、且难以应对持续知识更新的问题。

## 研究问题与动机
- VLLM 在部署后常因知识过时或错误产生幻觉，全量重训练成本过高，轻量级模型编辑成为刚需。
- 现有编辑工作多集中于纯 LLM，但 VLLM 引入了额外的视觉模态及复杂的跨模态交互，导致基于纯文本的 locate-then-edit 或因果中介分析难以准确定位关键参数。
- 单步编辑无法满足真实应用中持续更新的需求，而终身编辑（Lifelong Editing）在 LLM 中已有探索，但针对 VLLM 的终身编辑尚属空白。
- 视觉数据通常比文本更丰富且含噪声，直接将 LLM 的检索式或 MoE 式编辑方法平移至 VLLM 会面临视觉无关专家干扰、语义冲突等问题。

## 核心贡献（创新点）
1. **提出首个 VLLM 终身编辑框架 LiveEdit**。与现有单步 VLLM 编辑方法相比，本质区别在于支持多轮迭代更新并解耦编辑知识，避免基座模型参数累积偏移。
2. **设计生成式低秩 MoE 专家机制**。与 LEMoE 等直接训练静态专家的做法相比，本质区别在于为每次编辑独立生成专属低秩适配器，从根本上缓解了专家间的语义冲突与负载失衡。
3. **提出视觉引导硬路由与文本语义软路由协同机制**。与纯文本检索式编辑相比，本质区别在于利用提示词语义提取关键视觉特征进行跨模态粗筛，再通过双权重软路由精调，有效过滤视觉噪声。
4. **构建冻结基座的双损失联合训练范式**。与需要微调基座的单步编辑方法相比，本质区别在于将可靠性、泛化性与本地化损失统一至冻结 $f_\theta$ 的框架中，并引入对比路由损失保障多模态特征对齐。

## 方法详解
- **整体架构**：将 MoE 编辑器插入 VLLM Transformer 的高贡献层 $l_e$（实验表明 $l_e=21$ 附近效果最佳）。编辑样本 $(v_e, p_e, o_e)$ 与输入样本并行经过特征提取模块，VLLM 参数全程冻结。
- **专家生成（Expert Generator）**：在层 $l_e$ 处拼接视觉、提示词与输出表征，通过 Cross-Attention 生成低秩矩阵对 $(U_{e_t}, V_{e_t})$，作为该次编辑的专属适配器存入专家库 $\mathcal{E}_t$。
- **特征提取与硬路由**：使用特征提取器 $\hat{f}_{fe}$ 生成关键视觉特征 $\hat{\phi}_v$（受提示词引导以去除视觉噪声）和纯文本特征 $\hat{\psi}_p$。推理时，输入样本的视觉特征 $\
