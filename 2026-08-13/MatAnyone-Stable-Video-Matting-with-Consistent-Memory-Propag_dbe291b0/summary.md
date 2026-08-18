---
title: "MatAnyone-Stable-Video-Matting-with-Consistent-Memory-Propag"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Yang_MatAnyone_Stable_Video_Matting_with_Consistent_Memory_Propagation_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 07:38:32"
---

# 论文速读：MatAnyone-Stable-Video-Matting-with-Consistent-Memory-Propag

## 一句话总结
本文提出 MatAnyone，一种面向目标分配的视频抠图框架。通过区域自适应记忆融合机制在记忆空间中实现稳定的时序传播，并结合新构建的高质量数据集与直接利用真实分割数据的训练策略，在复杂背景与精细边界细节上均显著优于现有无辅助与掩码引导方法。

## 研究问题与动机
- **无辅助设定的内在不确定性**：仅依赖输入帧的辅助自由视频抠图（如 RVM）在复杂或背景含相似目标时极易产生语义漂移与闪烁。
- **现有掩码引导方法破坏 VOS 先验**：AdaM、FTP-VM、MaGGIe 等方法依赖子优的视频抠图数据集微调解码器，合成数据的质量缺陷会导致原本稳定的分割先验被破坏，核心区空洞与边界锯齿问题突出。
- **记忆匹配假设不适配抠图任务**：传统 VOS 记忆库仅存二值掩码，且默认所有 token 重要性相等；但视频抠图需同时保证核心区语义稳定与边界高频细节，二者对历史记忆的依赖程度不同。
- **真实高质量抠图数据稀缺**：人类标注 alpha 通道成本极高，导致模型难以在保持时序一致性的同时泛化到真实场景。

## 核心贡献（创新点）
- **区域自适应记忆融合模块**：在记忆空间内预测每帧 token 的 alpha 变化概率，动态软融合当前查询值与上一帧记忆，使核心区保留历史稳定信息、边界区聚焦当前帧细节。
- **直接针对主抠图头的分割数据监督策略**：摒弃 RVM 式的平行头训练范式，将大规模真实分割数据（COCO、SPD、YouTubeVIS）直接馈入主网络并优化 matting 输出，充分释放分割先验的语义稳定性。
- **缩放版 DDC 边界损失**：从图像合成模型严格推导原始 DDC 假设的失效条件，引入前景-背景强度差 $(F-B)$ 作为缩放因子，有效消除原版损失产生的阶梯状锯齿边缘。
- **VM800 训练集与 YoutubeMatte 评测基准**：构建两倍于 VideoMatte240K 的大规模高质量训练数据，以及细节更丰富、更具挑战性的合成与真实评测集，为领域提供可靠基础。

## 方法详解
- **整体架构**：采用类半监督 VOS 的记忆驱动范式，仅首帧需目标分割掩码。输入帧经骨干网络编码为 16 倍下采样特征，转化为 key/query 后进入一致记忆传播模块，最终经 object transformer 聚合像素级记忆生成 alpha matte。
- **Alpha Memory Bank 与融合机制**：记忆库存储连续 alpha 值而非二值 mask。引入轻量边界预测模块（3 层卷积）估计变化概率 $\hat{U}_t$，以 $|\mathbf{M}_{t-1}^{GT} - \mathbf{M}_t^{GT}| > \delta$ 为监督信号训练。融合公式为：
  $$P_t = V_t^m * U_t + V_{t-1} * (1 - U_t)$$
  其中 $V_t^m$ 为从记忆库查询的当前值，$V_{t-1}$ 为上一帧传播值，$U_t \in [0,1]$ 控制融合比例。
- **核心区监督（Core-area Supervision）**：在 Stage 2 训练中，将真实分割数据送入同一网络，直接用像素级 L1 损失 $\mathcal{L}_{core}$ 监督 matting 头的核心区输出，迫使模型学习语义稳定的全局表征。
- **缩放 DDC 边界损失**：原始 DDC 假设 $\|\alpha_i-\alpha_j\|_2 = \|I_i-I_j\|_2$，但在非纯色场景下不成立。基于合成方程 $I_i - I_j = (\alpha_i - \alpha_j)(F-B)$ 推导出边界损失：
  $$\mathcal{L}_{boundary} = \frac{1}{N}\sum_i\sum_j |(\alpha_i - \alpha_j)(\mathbf{F} - \mathbf{B}) - \|I_i - I_j\|_2|$$
  其中 $F/B$ 由小窗口内 top-k 像素均值近似，使边界优化更贴近真实光学衰减规律。
- **推理时循环精化**：可选模块，将首帧重复 $n$ 次输入，仅取第 $n$ 次输出初始化记忆库，利用序列传播特性迭代打磨首帧细节并提升对初始掩码噪声的鲁棒性。

## 实验与结果
- **数据集与基线**：合成基准 VideoMatte、YoutubeMatte；真实基准基于 25 段 YouTube 视频提取核心区 mask 作 GT。对比方法涵盖无辅助（MODNet、RVM、RVM-Large）与掩码引导（AdaM、FTP-VM、MaGGIe）。评估指标包括 MAD、MSE、Grad、Conn、dtSSD。
- **合成基准最强结果**：在 1920×1080 VideoMatte 上 MAD 降至 **4.24**、MSE **0.33**；YoutubeMatte 高分辨率上 MAD **1.99**、MSE **0.71**，在时序一致性（dtSSD）与语义精度（MAD）两项核心指标上全面领跑。
- **真实场景最强结果**：MAD **0.14**、MSE **0.10**、dtSSD **0.89**。相较次优方法 MaGGIe（MAD 1.94、MSE 1.53、dtSSD 1.63），MAD/MSE 分别提升约 **85%/93%**，dtSSD 提升约 **45%**，显著改善复杂背景下的语义稳定性与连贯性。
- **消融验证**：新数据集使 MAD 从 3.16 降至 2.55；加入 CMP 后各指标全面提升（MAD 2.55→1.85，dtSSD 1.36→1.25）；新训练策略带来跨越式提升（MAD 1.85→0.42，dtSSD 1.25→0.94）；缩放 DDC 损失对比原版可有效消除边界锯齿，视觉更自然。

## 相关工作脉络
- **无辅助视频抠图（MODNet、RVM 系列）**：无需任何额外标注，但设定内在不确定。本文转向首帧掩
