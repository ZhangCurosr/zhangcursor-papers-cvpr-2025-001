---
title: "SPARC-Score-Prompting-and-Adaptive-Fusion-for-Zero-Shot-Mult"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Miller_SPARC_Score_Prompting_and_Adaptive_Fusion_for_Zero-Shot_Multi-Label_Recognition_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 10:53:12"
field: "多模态零样本/开放词汇识别"
keywords: ["zero-shot multi-label recognition", "vision-language models", "score normalization", "compound prompt", "rank fusion", "order statistics"]
innovations: ["两级标准化去除 VLM 图像级与提示级偏差", "基于 co-occurrence 的复合提示与 PCA 自适应排序融合", "发现并解释 VLM 复合提示的 OR+AND-bonus 噪声结构，论证次高得分优于最大值"]
benchmarks: ["COCO2014", "VOC2007", "NUSWIDE"]
---

# 论文速读：SPARC-Score-Prompting-and-Adaptive-Fusion-for-Zero-Shot-Multi-Label-Recognition-in-Vision-Language-Models

## 一句话总结
本文提出 SPARC，一种完全无需训练数据、不调用模型内部结构、仅将 VLM 视为黑盒的零样本多标签识别方法；通过标准化去除 VLM 分数的图像级/提示级偏差，并结合基于 co-occurrence 筛选的复合提示与 PCA 自适应排序融合，显著提升 CLIP 在 COCO/VOC/NUSWIDE 等基准上的 mAP，且可与其它零样本/训练型方法互补叠加。

## 研究问题与动机
- 零样本多标签识别要求在不依赖标注数据和模型微调的前提下，基于 VLM 分数对图像中多个类别进行鲁棒排序与判定；现有多数方法需要 prompt tuning 或访问/修改模型内部结构，通用性与部署便捷性受限。
- VLM（以 CLIP 为代表）的多类别提示得分存在系统性偏差：同一张图在不同 prompt 间出现图像级偏移，不同 prompt 之间存在提示级偏移，直接比较/融合会导致排序失真，直接影响 mAP。
- 复合提示理论上能携带共现上下文信息，但 VLM 对形如“A and B”的提示呈现近似 OR 行为：只要任一对象出现即给出高分，导致最大复合得分对“仅含关联对象 A/B 之一”的负样本也给出高置信度，产生大量误报并削弱判别性。
- 现有“最大得分优先”的直觉在多标签场景下不可靠；作者发现次高/更低位阶统计量更能削弱 OR 带来的假阳性，因此需要一套能够综合多阶统计并自适应权重的融合机制。

## 核心贡献（创新点）
1. **提出面向黑盒 VLM 的零样本多标签识别框架 SPARC**：无需训练或架构改动，仅依赖 singleton/compound 提示得分完成偏差校正与融合排序；与需 prompt-tuning 或内部访问的方法本质不同。  
2. **揭示并缓解 VLM 得分的图像级与提示级偏差**：通过两级标准化（跨 prompt 的图像内标准化、跨图像的同 prompt 标准化）显著改善类别排序稳定性；与常见仅做单维度归一化的做法相比，SPARC 同时处理两类系统性噪声源。  
3. **构建基于共现信息的复合提示并设计自适应排序融合**：用粗略 co-occurrence 阈值筛选双/三元组类别构成自然化复合 prompt；采用 PCA 最大方差方向（首主成分）对 singleton 与各阶最高分统计量做加权融合，优于简单的 max/mean 固定策略。  
4. **提供对 VLM 复合提示评分行为的噪声建模与解释**：提出 OR + 变量 AND-bonus 的半参数模型，并与纯 OR/纯 AND/加法/LUT 模型对比；解释为何“次高/弱化 max"在多标签场景中更具判别力。  
5. **验证 SPARC 的模型无关性与方法可插拔性**：在 9 种 CLIP backbone 及 COCO/VOC/NUSWIDE 上一致提升，并可叠加 TagCLIP/TaI-DPT/CoMC 等已有方法进一步增益（平均 +1.6%~+1.7%）。

## 方法详解
- **输入与符号**：图像集合 $\mathcal{I}$，类别名 $\{c_1,\dots,c_N\}$；singleton 提示得分 $s_i^t$（图像 $t$ 对类别 $i$ 的相似度），compound 提示集 $P$ 及对应得分 $s_p^t$。
- **复合提示生成**：基于从训练集估计的条件共现概率 $P$ 与阈值 $\tau_2=0.05$、$\tau_3=0.025$ 筛选 plausible 类对/类三元组；构造模板句 "A and B"、"A, B, and C"，再交由 LLM 生成为自然语言 prompt；平均每类 $\le 20$ 条 compound prompt。共现信息仅用于选择哪些复合 prompt 出现，不参与后续打分与融合。
- **图像级标准化（去图像偏差）**：
  - 对 singleton: $\tilde{s}_i^t = \frac{s_i^t - \hat{\mu}(s_{\cdot}^t)}{\hat{\sigma}(s_{\cdot}^t)}$
  - 对 compound: $\tilde{s}_p^t = \frac{s_p^t - \hat{\mu}(\check{s}_{\cdot}^t)}{\hat{\sigma}(\check{s}_{\cdot}^t)}$，其中 $\check{s}$ 为仅含单个类别名的 auxiliary prompt 得分，用于稳定统计量。
- **提示级标准化（去提示偏差）**：
  - $\bar{s}_i^t = \frac{\tilde{s}_i^t - \hat{\mu}(\tilde{s}_i^{\cdot})}{\hat{\sigma}(\tilde{s}_i^{\cdot})}$，类似对 $\bar{s}_p^t$ 操作。
  - 目的：使不同 prompt 间的得分可比，并为后续融合与 rank 统计提供稳定尺度。
- **排序融合（Rank Fusion）**：
  - 记 $r_{i,k}^t$ 为涉及类别 $i$ 的 compound prompt 中第 $k$ 高的标准化得分。
  - 构建特征向量 $[\bar{s}_i^t, r_{i,1}^t, r_{i,2}^t, \dots]$，以跨样本方差最大化为目标求权重：
    $w^{i*} = \arg\max_{w^i} \mathrm{Var}_t(w_0^i \bar{s}_i^t + \sum_k w_k^i r_{i,k}^t)$，即首主成分方向。
  - 融合得 $\tilde{\zeta}_i^t = w_0^{i*} \bar{s}_i^t + \sum_k w_k^{i*} r_{i,k}^t$，最终得分 $\zeta_i^t = \bar{s}_i^t + \tilde{\zeta}_i^t$（singleton 与融合残差相加以保留原始信号）。
- **噪声模型动机**：将复合得分建模为 $s_{i,j}^t = \theta_1^t f(y_i^t, y_j^t) + \theta_0^t + \varepsilon$；经验拟合表明 $f$ 近似 $\max(\cdot,\cdot) + \delta_{ij}\min(\cdot,\cdot)$ 的 OR+AND-bonus 形式，AND 项提供“双目标同时存在”的额外信号，而 OR 项主导则解释了为何直接取 $r_{i,1}$ 易受无关正类干扰，转而用多阶统计 + PCA 可更稳健地提取判别方向。

## 实验与结果
- **数据集**：COCO2014（40,504 张，80 类）、VOC2007（4,952 张，20 类）、NUSWIDE（107,859 张，81 类）；仅使用测试集图片，共现概率仅用于 prompt 生成。
- **Backbone**：9 种 CLIP 变体（ViT-L/14@336px、ViT-L/14、ViT-B/16、ViT-B/32、RN50x64、RN50x16、RN50x4、RN101、RN50）。
- **主要结果（vs ZSCLIP 基线）**：
  - COCO 平均提升 **+12.6%**，VOC **+8.8%**，NUSWIDE **+7.9%**；各架构均稳定提升（COCO 单模型最高 70.6，VOC 最高 90.5，NUS 最高 48.3，均优于对应基线）。
- **与局部特征/训练方法互补**：
  - 与 CLIP-DPT（免训练的局部注意力池化）叠加后仍持续提升（COCO RN50x64: 70.5→77.8；VOC: 86.6→91.1；NUS: 39.8→43.5）。
  - 与 TagCLIP 叠加平均 +1.6%（COCO 70.9→73.8，VOC 91.7→92.0）。
  - 与 TaI-DPT 叠加平均 +1.7%；与 CoMC 叠加仅降 0.3%，整体兼容。
- **消融要点**：
  - 仅做标准化即可带来 +7.7%（单 prompt）/ +8.6%（完整 pipeline）；
  - 复合 prompt 策略中，“all pairs"即带来 +1.3%，再加 co-occurrence 过滤 +0.5%，再加 triplet/descriptive +0.3%；
  - Rank Fusion：k-max 中第 2~5 阶依次优于 1st-max，mean>=1 意外有效，但自适应 max-variance 融合仍最优；合并步骤（Eq.5）不可省略；
  - 用 WaffleCLIP 式随机后缀替代复合 prompt 语义不能带来增益，证实收益来自语义而非数量/多样性。

## 相关工作脉络
- **Supervised MLR（DualCoOp、SSPA、TRM-ML 等）**：依赖标注训练与 prompt 学习；SPARC 完全不训练，属 zero-shot 黑盒范式。  
- **Unsupervised training-based（TaI-DPT、CoMC、LLM-caption 扩展等）**：利用 caption/embedding 训练提示或轻量分类器；SPARC 与之互补，可直接在其输出分数上叠化融合。  
- **Training-free 架构修改（CLIP-Surgery、TagCLIP）**：需访问 attention 结构或做 patch 级聚合；SPARC 不触碰内部结构，实现 plug-and-play。  
- **Single-label prompt enhancement**：通过 LLM 生成更丰富的描述性提示并用 max/mean 融合；SPARC 的复合提示目标是捕捉类间共现关系，而非单纯增强单类描述。  
- **定位差异**：SPARC 的核心区别于现有工作的地方在于——以“偏差校正 + 多阶统计自适应融合”为统一视角，证明次高排序比最大排序更适合 MLR，并将噪声模型分析落地为可复用的后处理流水线。

## 局限性与未来方向
- 依赖稳定的共现概率估计用于筛选复合 prompt；在极度稀疏/新类别分布场景下统计质量可能下降。  
- 当前范式仍以黑盒分数为核心，未利用区域/空间定位先验（虽可与 TagCLIP 等位置感知模块组合，但原生不内置空间聚合）。  
- 融合权重通过全局最大方差学习，对类别高度相关的复杂场景可能存在过拟合风险（论文未显式讨论）。  
- 方向：结合空间注意力/区域 proposal、扩展至视频或密集预测任务、将降噪模型扩展到更多 VLM 家族（如 BLIP、LLaVA 类多模态大模型）。

## 研究启发与可借鉴点
- **两级标准化策略**（图像内跨 prompt + 跨图像同 prompt）对多类/多 prompt 打分系统具有通用参考价值，可直接迁移到其他多标签/开放词汇检测的后处理环节。  
- **“次高优于最高”的 OR-bias 现象**提示我们在构建 multi-prompt ensemble 时应谨慎对待 argmax；引入低阶 order statistics 或屏蔽型融合可降低关联类别带来的假阳性。  
- **PCA/最大方差融合可作为零样本多源分数的通用加权方案**：在缺少 ground-truth 标注时，以跨样本方差近似替代判别信号，工程实现简单且稳定。  
- **复合 prompt 可由粗粒度统计驱动、由 LLM 语义化**的两段式生成流程，兼顾可扩展性与自然语言多样性，适用于需要快速扩展多类别上下文的场景。

## 关键术语表
- **Zero-shot Multi-Label Recognition (MLR)**：在未见标签分布的图像中，无需训练即预测多个目标类别的存在性，并以 mAP 为主要评估指标。  
- **Singleton Prompt**：仅描述单一类别的文本提示（如 "a photo of a cat"），用于获得该类的基础相似度得分。  
- **Compound Prompt**：同时包含两个及以上类别信息的提示，用于捕捉共现/上下文信号并放大判别信息。  
- **Image-level Bias**：跨 prompt 在同一张图上出现的系统性偏移，会改变同类别正负样本间的相对排序。  
- **Prompt-level Bias**：同一类别在不同 prompt 间表现出的系统性差异，影响跨 prompt 的可比性。  
- **Order Statistics (weakened max)**：将 compound 得分由高到低排序后取第 $k$ 位得分；实验发现 $k \ge 2$ 常比最大值更具判别性。  
- **OR / AND-bonus 行为**：VLM 对复合提示呈现近似逻辑 OR 的高分倾向，并在两对象同时出现时附加较小的 AND 提升项。  
- **Max-variance Rank Fusion**：对 singleton 与各阶 compound 得分做 PCA 首主成分加权融合，以最大化跨样本方差近似提取判别方向。

## 可复现要素
- **数据集**：COCO2014、VOC2007、NUSWIDE（公开）；论文仅提供测试集评测，未新增私有数据集。  
- **代码**：作者已开源，见 https://github.com/kjmillerCURIS/SPARC。  
- **模型/权重**：使用开源 CLIP 系列权重（ViT/RN 多尺度），论文未提供训练新权重。  
- **关键超参**：共现概率阈值 $\tau_2=0.05$、$\tau_3=0.025$（论文注明在所有数据集统一使用）；ViT 默认 224×224 非中心裁切；ResNet 分辨率按对应前作设置；singleton 采用 CLIP 论文 80 条模板集成；平均每类 $\le 20$ 条 compound prompt。
