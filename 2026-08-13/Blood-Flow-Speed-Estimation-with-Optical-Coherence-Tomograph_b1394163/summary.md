---
title: "Blood-Flow-Speed-Estimation-with-Optical-Coherence-Tomograph"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Cheng_Blood_Flow_Speed_Estimation_with_Optical_Coherence_Tomography_Angiography_Images_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:56:29"
field: "医学图像计算与生物光学成像分析"
keywords: ["OCTA", "ODT", "blood flow estimation", "pseudo label", "Adaptive Window Fusion", "Conditional Random Field Decoder", "Swin Transformer", "medical image regression"]
innovations: ["以 ODT 伪标签驱动直接从 OCTA 图像回归血流速度", "多尺度自适应窗口融合模块 AWF", "多层特征结构化平滑解码器 CRFD"]
benchmarks: ["Anesthetized Dataset", "Awake Dataset"]
---

# 论文速读：Blood-Flow-Speed-Estimation-with-Optical-Coherence-Tomograph

## 一句话总结
本文提出 OCTA-Flow，一种直接从 OCTA 血管结构图像回归像素级血流速度的深度学习方案，以有角向伪影的 ODT 数据作为伪标签进行训练，借助多尺度自适应窗口融合与 CRF 解码器实现高质量、去伪影的无创血流速度估计。

## 研究问题与动机
- 血流速度是评估血管健康与脑功能的重要指标，但在大脑皮层等复杂微血管网络中精确测量极其困难。
- 现有高分辨率方法 ODT 需要复杂硬件与信号处理，且存在明显的角度伪影：光线近垂直血管时会出现剧烈的流速跳变甚至血管断裂。
- OCTA 广泛用于血管结构分析，但并非为流速估计设计，传统上仍需配合 ODT 等专用设备获取血流信息。
- 理想的地面真值（ground truth）血流数据难以大规模获取，作者希望在不依赖昂贵专用硬件的前提下实现更实用的临床可用血流估计。

## 核心贡献（创新点）
- 提出 OCTA-Flow，直接从单幅 OCTA 图像端到端回归像素血流速度，避免 costly/复杂的 ODT 测量链路。
- 以 ODT 测量作为伪标签（pseudo labels）进行监督，有效缓解高质量 ground truth 稀缺的问题。
- 设计 Adaptive Window Fusion（AWF）模块，通过多尺寸窗口注意力提取多尺度上下文，并以内容自适应的门控权重进行细粒度空间融合。
- 引入 Conditional Random Field Decoder（CRFD），分层建模多层特征间的依赖关系，增强预测的空间一致性与平滑性。
- 构建并公开两组小鼠脑皮层 OCTA-ODT 配对数据集，分别覆盖麻醉与清醒状态，便于社区复现与对比。

## 方法详解
- 整体框架为编码器-解码器结构：输入 OCTA 图像 $I \in \mathbb{R}^{H \times W}$，通过多阶段骨干网络 $\mathcal{E}$（Swin Transformer）提取多层特征 $\mathbb{B}=\{B_0, B_1, B_2, B_3\}$。
- AWF 模块：将深层特征 $B_3$ 输入多个窗口注意力块 $\mathcal{W}_i$（窗口尺寸 $w_i \in \{3,5,7\}$），每个块包含常规窗口多头自注意力（W-MSA）与偏移窗口多头自注意力（SW-MSA），得到多尺度上下文 $M_1, M_2, M_3$；再通过卷积门 $\mathcal{G}(B_3)$ 生成空间权重矩阵并按通道拆分出 $G_1, G_2, G_3$，以逐元素相乘加权求和得到 $R_0$，实现内容感知的自适应融合。
- CRFD：由三个级联的 Hierarchical CRF（HiCRF）块 $\mathcal{H}_1, \mathcal{H}_2, \mathcal{H}_3$ 组成；每层 HiCRF 利用 Neural CRF（N-CRF）单元分别建模相邻层级骨干特征之间的关系以及高层语义特征与当前特征之间的关系，并结合上采样（bilinear）与 pixel shuffle 等操作融合后投影输出，形成从深到浅的多级细化流程。
- 解码输出：最终特征 $R_3$ 经双线性上采样得到与输入同分辨率的像素级血流速度预测 $Y$。
- 损失函数：采用尺度不变的 log 域损失（scale-invariant logarithmic loss），$g_i=\log y_i-\log \bar{y}_i$，通过 log 压缩不同管径对应的流速量纲，并对 ODT 伪影引起的极端值更具鲁棒性；具体形式为 $\mathcal{L}=\alpha\sqrt{\frac{1}{N}\sum_i g_i^2 - \frac{\lambda}{N^2}(\sum_i g_i)^2}$，文中设 $\alpha=10$、$\lambda=0.85$。
- 训练设置：50 epochs，初始学习率 2e-4 衰减至 2e-5，Adam（$\beta_1=0.9$, $\beta_2=0.999$），五折交叉验证。

## 实验与结果
- 数据集：两个体内小鼠脑皮层配对数据集，共 106 个样本（66 麻醉 + 40 清醒），覆盖不同动物生理状态；清醒数据存在明显整体运动条纹伪影。
- 评估指标：Abs Rel、RMSE（基于线性缩放后的预测进行计算）。
- 主要结果（五折 CV）：
  - Anesthetized Dataset：OCTA-Flow 达到 Abs Rel 0.353 ± 0.042、RMSE 6.278 ± 0.480，分别优于次优方法 0.009 / 0.150。
  - Awake Dataset：OCTA-Flow 达到 Abs Rel 0.318 ± 0.018、RMSE 6.037 ± 0.674，分别优于次优方法 0.041 / 0.278。
- 结论要点：OCTA-Flow 在两组数据集上均取得最优，尤其在清醒数据上相对优势更大，显示出对 ODT 角度伪影与运动伪影的有效抑制；定性结果显示预测与 OCTA 血管结构高度对齐，且符合血流动力学规律（大血管到小分支流速逐步降低）。
- 消融结论：
  - 基础 UNet-like 基线为 0.393 / 7.232；加入 AWF 与 CRFD 均有显著提升，二者结合最优（0.328 / 6.661）。
  - AWF 的多尺度窗口与自适应门控均有效；对比金字塔池化（PPM），本文方法 Abs Rel 提升 0.014、RMSE 提升 0.303。
  - CRFD 中三个 HiCRF 块逐级串联持续带来增益。

## 相关工作脉络
- 传统血流测量：Doppler Ultrasound、Phase-Contrast MRI、ODT；本文定位是以深度学习替代/补充 ODT，降低硬件门槛并抑制其系统级伪影。
- OCTA 基础：基于散斑去相关区分血管与静态组织；既往工作主要用于结构分析或需原始信号+统计模型，本文首次从通用 OCT 系统的 OCTA 图像直接回归流速。
- 深度回归方法：比较了 BFS、IEBins、NeuWin、OrdEnt、Diff Depth、ECoDepth 等主流深度回归/单目测深方法；本文强调血管结构与流场的跨尺度关联与空间一致性建模。
- 伪标签与弱监督学习：借鉴在标签不完备场景下用粗糙伪标签引导训练的思路；本文与弱监督分割类工作的差异在于伪标签为连续场且含明显伪影，需要额外平滑与去噪机制。
- CRF/结构化推理：延续 N-CRF 系列端到端可微的 CRF 思路；本文将其扩展到多尺度、多层特征的结构一致性正则。

## 局限性与未来方向
- 伪标签依赖：训练依赖 ODT 伪标签，仍会受角度伪影与系统误差影响，模型的泛化上限可能受限于伪标签质量。
- 数据规模与物种限制：目前仅在小鼠脑皮层两个条件下验证，样本量相对有限（106 对），对临床人眼/人脑的迁移性需进一步验证。
- 运动伪影仍存在痕迹：尽管方法在清醒数据上更鲁棒，OCTA 与 ODT 的运动条纹仍可见，说明极端运动情况下仍需更强的去伪影先验或预处理。
- 未引入时序信息：当前仅使用单帧/单体积投影的 OCTA 图像；引入时间序列或动态散斑信息可能进一步提升流速估计精度。
- 分辨率与位深差异：OCTA 为 8-bit，ODT 为 16-bit，位深差异增大估计难度，未来可研究更适配的表示与对齐策略。

## 研究启发与可借鉴点
- 以含伪影的物理测量作为伪标签，配合结构化先验（CRF）与多尺度注意力，是一种“用脏标签学干净输出”的可复用范式，适用于其他缺乏 ground truth 的连续场回归任务。
- AWF 的多窗口注意力 + 内容自适应门控融合，能有效建模“不同尺度结构对应不同物理量级”的问题，可迁移到具有多尺度空间先验的密度/流量/速度场估计。
- HiCRF 逐层跨尺度融合的思路，兼顾细粒度结构保持与全局一致性，适合用于需要同时保留细节与平滑输出的医学图像回归。
- 使用 log 域尺度不变损失应对多量纲、多极值伪标签的策略，对具有异方差噪声的连续值回归具有参考价值。
- 开放配对数据集的思路利于社区基准建设，后续可与本团队的方向（如医学超声/光学成像回归、跨模态伪标签训练）结合，探索人源数据或更多生理条件。

## 关键术语表
- **OCTA（Optical Coherence Tomography Angiography）**：基于散斑去相关对比的血流灌注成像技术，广泛用于血管结构可视化。
- **ODT（Optical Doppler Tomography）**：利用多普勒频移定量测量血流速度的高分辨率光学方法，但存在角度依赖与系统伪影。
- **Pseudo label**：由其他传感器或模型生成的近似标签，用于在缺乏真实标注时引导监督学习。
- **Adaptive Window Fusion（AWF）**：通过多尺寸窗口注意力提取多尺度上下文，并以输入自适应的空间门控权重进行融合的特征模块。
- **Conditional Random Field Decoder（CRFD）**：以 N-CRF 为核心、在多尺度特征间建立结构化依赖的解码模块，用于提升预测平滑性与一致性。
- **Neural CRF（N-CRF）**：将 CRF 的 unary 与 pairwise 势能用神经网络（如卷积与窗口注意力）参数化，实现端到端可训练的结构化推理。
- **Scale-invariant logarithmic loss**：在 log 域计算预测与目标差异并对全局尺度变化不敏感的回归损失，用于缓解多管径流速的尺度差异与伪标签极端值影响。
- **HiCRF（Hierarchical CRF block）**：在多层特征之间分阶段建模关系并级联输出的 CRF 模块。

## 可复现要素
- 数据集：作者构建了 Anesthetized Dataset（66 对）与 Awake Dataset（40 对），并声明代码与数据已开源。
- 代码/权重：项目页面 https://github.com/Spritea/OCTA-Flow（论文声明）。
- 关键超参：Swin Transformer 骨干；AWF 窗口尺寸 $w_1=3, w_2=5, w_3=7$；损失参数 $\alpha=10, \lambda=0.85$；训练 50 epochs；初始 LR 2e-4 衰减至 2e-5；Adam $\beta_1=0.9, \beta_2=0.999$；五折交叉验证。
- 注：具体实现细节（通道数、层数、分组策略等）以代码仓库与补充材料为准，论文正文未全部列出。
