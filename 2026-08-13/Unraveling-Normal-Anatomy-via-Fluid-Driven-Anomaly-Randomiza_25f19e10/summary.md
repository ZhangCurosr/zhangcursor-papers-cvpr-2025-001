---
title: "Unraveling-Normal-Anatomy-via-Fluid-Driven-Anomaly-Randomiza"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_Unraveling_Normal_Anatomy_via_Fluid-Driven_Anomaly_Randomization_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 09:52:20"
---

# 论文速读：Unraveling-Normal-Anatomy-via-Fluid-Driven-Anomaly-Randomiza

## 一句话总结
本文提出 UNA，首个无需微调即可处理健康与病理脑影像的模态无关学习框架；通过流体驱动异常随机化生成无限多样本，结合对侧对称先验与自对比学习，实现从含病变 CT/MRI 图像中重建高质量健康解剖，并直接用于无监督异常检测。

## 研究问题与动机
1. **模态与分辨率泛化瓶颈**：现有医学影像分析方法多针对特定序列（如 T1w MRI）和固定各向同性分辨率设计，临床中扫描参数、对比度、取向的差异导致模型性能急剧下降。
2. **病理场景失效**：主流模态无关模型（SynthSeg、Brain-ID 等）仅在健康数据上训练，遇到大面积病变时重建失真；PEPSI 虽兼容广泛病理，但依赖成对病理分割标注、预训练分割模型计算隐式损失，且需微调才能检测异常。
3. **病理金标准标注稀缺**：3D 医学影像的高质量病理标注成本极高，大规模带标注数据集几乎不存在，限制了对复杂病变进行数据驱动建模的能力。
4. **健康分析工具与病变影像之间存在鸿沟**：临床广泛使用的结构化分析流水线（FreeSurfer、NiftyReg 等）仅面向高分辨率健康 T1w 影像，缺乏能将病变影像“解构”回健康解剖的通用桥梁。

## 核心贡献（创新点）
1. **流体驱动异常随机化（Fluid-Driven Anomaly Randomization）**：将对流-扩散 PDE 的正向质量输运机制引入病理形态生成，仅用少量真实卒中分割作为初始条件即可生成无限多样化、符合临床真实约束的异常轮廓，并通过零 Neumann 边界条件自然限制病变不越界。
2. **模态无关的健康解剖重建框架**：首次实现无需配对病理标注、无需微调即可从任意模态（CT / T1w / T2w / FLAIR）及含病理图像中恢复高质量健康脑解剖。
3. **对侧配对本体自对比学习**：利用大脑左右半球结构近似对称的先验，构建对侧配对本体输入；结合内主体对比损失，在无病理区 Ground Truth 的情况下引导模型学习健康解剖表征。
4. **“重建即检测”的统一范式**：将病理→健康重建直接转化为体素级差异热力图
