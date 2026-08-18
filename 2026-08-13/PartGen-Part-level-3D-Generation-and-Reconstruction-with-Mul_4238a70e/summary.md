---
title: "PartGen-Part-level-3D-Generation-and-Reconstruction-with-Mul"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Chen_PartGen_Part-level_3D_Generation_and_Reconstruction_with_Multi-view_Diffusion_Models_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 09:37:27"
field: "3D 生成与结构化重建"
keywords: ["part-level 3D generation", "multi-view diffusion", "3D part segmentation", "contextual completion", "structured 3D asset", "3D reconstruction"]
innovations: ["将 3D 部件分割建模为多视图颜色编码扩散采样，利用随机性隐式处理实例命名歧义", "提出上下文感知部件补全网络，以 mask 化视图 + 全局视图 + mask 三元组为条件生成完整部件视图", "统一处理文本/图像/真实 3D 扫描三种输入模态，并在不微调 RM 的前提下产出可编辑的结构化 3D 资产"]
benchmarks: ["mAP@50 (自动分割, 45.2 vs SAM2* 37.4)", "CLIP similarity (补全后重建, 0.936)", "PSNR (补全后重建, 17.16)", "mAP@50 (seeded 分割, 53.7)"]
---

# 论文速读：PartGen: Part-level 3D Generation and Reconstruction with Multi-View Diffusion Models

## 一句话总结
PartGen 利用两个微调过的多视图扩散模型，将文本/图像/真实 3D 扫描输入自动分割为语义有意义的独立部件，并对遮挡或完全不可见的部件进行**上下文感知的幻觉补全**，再经确定性重建网络输出结构化、可编辑的 3D 零件组合。

---

## 研究问题与动机
1. **结构化缺失**：现有文本/图像→3D 生成器产出的资产是单个神经场/高斯混合体，无内部部件结构，无法满足艺术家在复用、编辑、动画中的需求。
2. **部件分割的不确定性**：同一物体的"合理分件"方式因人而异，不存在单一金标准，需要建模"合理分解分布"而非单一点估计。
3. **可见性与幻觉缺失**：现有 3D 重建只建模外表面，无法恢复被遮挡的内部结构；单纯把 masked view 喂给确定性重建网络在部分/完全不可见部件上表现极差。
4. **现有分割方法的局限**：直接微调 SAM/SAM2 难以兼顾多视图一致性、部件命名歧义（instance permutation）和艺术家语义意图。

---

## 核心贡献（创新点）
1. **多视图一致的部件分割模型**：将部件分割形式化为多视图一致的着色（color-coded）概率扩散问题，避免显式部件分类树，通过随机颜色置换隐式处理实例命名问题。
2. **上下文感知的部件补全模型**：补全网络同时以 mask 化局部视图 $I \odot M$、完整视图 $I$ 和 mask $M$ 为条件，遮挡越严重越依赖全局上下文，从而"幻觉"出被遮挡部件的完整视图。
3. **面向不同输入模态的统一框架**：文本、单图或真实 3D 扫描均可接入，经同一 pipeline 输出带结构部件的 3D 资产；并自然扩展到文本驱动的部件级 3D 编辑。
4. **基于艺术家数据的训练数据构建策略**：从 140k 商业授权 3D 资产中筛选 45k 对象（含 210k 部件），过滤过于粗粒度/过细粒度部件，形成多视图渲染+深度图的分割与补全训练集。

---

## 方法详解
### 整体流程（图 2）
**输入** $y$（文本/图/3D） → 多视图渲染网格 $I \in \mathbb{R}^{3 \times 2H \times 2W}$（4 正交方位 + 20° 俯仰，排为 2×2 网格）→ $\Phi_{\text{seg}}$ 输出颜色编码的多视图分割图 $C$ → 对每部件 $k$，$\Phi_{\text{comp}}$ 以 $(I \odot M^k, I, M^k)$ 为条件生成补全视图 $\hat{J}^k$ → RM $\Psi$ 重建各部件 $\hat{\mathbf{S}}^k = \Psi(\hat{J}^k)$ → 合并得结构化 3D 资产。

### 3.2 多视图部件分割
- 将 RGB 空间量化为 $Q$ 个固定颜色 $c_1,\dots,c_Q$；对训练样本 $\mathbf{L}=(\mathbf{S}^k)_{k=1}^S$，随机排列 $\pi$ 并将部件 $\mathbf{S}^k$ 映射到颜色 $c_{\pi_k}$。
- 由深度图合成多视图分割渲染 $C \in [0,1]^{3 \times 2H \times 2W}$，以 $I$ 为条件微调多视图生成器 $\Phi$ 为 $\Phi_{\text{seg}}$：$C \sim p(C \mid \Phi_{\text{seg}}, I)$。
- 测试时多次采样 $C$，按参考颜色量化得到多种合理分件；去除像素数过少的伪部件。
- **网络实现**：VAE 编码 $I$ 后与加噪潜变量堆叠作为扩散网络输入。

### 3.3 上下文感知部件补全
- 第 $k$ 部件补全网络 $\Phi_{\text{comp}}$ 以三元组 $(I \odot M^k, I, M^k)$ 为条件，生成补全后多视图 $\hat{J}^k \sim p(\hat{J} \mid \Phi_{\text{comp}}, I \odot M^k, I, M^k)$。
- **设计直觉**：遮挡严重时 $I \odot M^k$ 信息极少，需借助完整视图 $I$ 提供全局几何/材质上下文；补全结果与其他部件"自洽"。
- **网络实现**：VAE 分别编码 $I \odot M^k$ 和 $I$（各 8 latent channel），与噪声、未编码 mask $M^k$ 堆叠为 25 通道输入到扩散模型。

### 3.4 部件重建
- 由于 $\hat{J}^k$ 已是完整多视图，直接使用预训练 RM $\Psi$（LightplaneLRM）重建：$\hat{\mathbf{S}}^k = \Psi(\hat{J}^k)$；RM 无需微调。

### 3.5 训练数据
- 来源：140k 来自商业授权的艺术家 3D 资产（GLTF 格式，含 watertight mesh 部件）。
- 过滤：剔除 <5% 体积的小部件、>10 部件或仅 1 部件的对象 → 45k 对象、210k 部件。
- 分割训练对：$\{(I_n, \mathbf{M}_n)\}$，其中 $\mathbf{M}^k$ 为部件 $k$ 在多视图下的 binary mask（$M_{i,j}^k = [\delta_{i,j}^k = \min_l \delta_{i,j}^l]$）。
- 补全训练三元组：$\{(I_{n'}, J_{n'}, M_{n'})\}$，$J^k$ 为部件 $k$ 的完整多视图渲染。

---

## 实验与结果
### 数据集与基线
- 测试：100 个 hold-out 对象（来自同一 45k 数据集）。
- 分割基线：Part123 [40]、SAM2 [67]（原始 / 微调 mask decoder / 微调多视图）、PartGen 1/5/10 次采样。
- 补全基线：无补全（$\hat{J}=I \odot M$）、去掉上下文（$\hat{J}=\mathcal{B}(I \odot M)$）、单视图补全（$\hat{J}_v=\mathcal{B}(I_v \odot M_v, I_v)$）、Oracle（GT $\hat{J}=J$）。

### 关键定量结果
**分割（Tab. 1，mAP，自动模式）**：
- Part123：mAP@50 = 11.5，mAP@75 = 7.4
- SAM2†（多视图微调）：mAP@50 = 20.3
- SAM2*（mask decoder 微调）：mAP@50 = 37.4
- **PartGen（1 sample）**：mAP@50 = **45.2**，mAP@75 = **32.9**
- **PartGen（10 samples）**：mAP@50 = **59.3**，mAP@75 = **38.5**
-  seeded 模式：PartGen（10 samples）mAP@50 = 53.7，mAP@75 = 35.4，均超 SAM2*。

**补全（Tab. 2）**：
- Oracle（用 GT $\hat{J}=J$）：CLIP = 1.0，LPIPS = 0.0，PSNR = ∞（view）；重建 CLIP = 0.957，LPIPS = 0.027，PSNR = 18.91。
- **PartGen 完整模型**：view 侧 CLIP = 0.974 / LPIPS = 0.015 / PSNR = 21.38；重建 CLIP = **0.936** / LPIPS = 0.039 / PSNR = **17.16**。
- 去掉上下文：PSNR 降至 14.83，说明全局上下文对补全至关重要。
- 单视图独立补全：PSNR 仅 13.25，证明多视图联合推理的收益。
- 无补全（直接 masked）：PSNR = 12.32。

**重组（Tab. 3）**：
- PartGen 重组后的整体重建 CLIP = 0.952，LPIPS = 0.065，PSNR = 20.33，**几乎与直接重建 $\hat{\mathbf{L}}=\Psi(I)$（CLIP 0.955 / PSNR 20.47）持平**，同时获得结构化部件。

### 应用展示
- 文本→带部件 3D（gummy bear 等强遮挡场景也能合理补全内部轮子/沙子）；图像→带部件 3D；Google Scanned Objects 真实扫描分割；文本驱动部件级形状/纹理编辑（图 7）。

---

## 相关工作脉络
1. **DreamFusion 系（SDS 3D 生成）**：本文不采用慢速 SDS 优化，而是直接在多视图扩散空间操作，后接 feed-forward RM，速度快且能自然建模分割/补全的概率分布。
2. **Instant3D / AssetGen / IM-3D 等两阶段多视图生成**：继承"先多视图再生成、再 RM 重建"架构；本文在此基础上增加**分割→补全**中间层，把单一流水线升级为部件级结构化流水线。
3. **Part123 [40]**：同样做部件感知 3D 重建，但基于单视图蒙卡重建 + 对比学习；PartGen 在自动分割 mAP 上显著领先（45.2 vs 11.5）。
4. **SAM / SAM2 系列（3D/2D 分割）**：作为强 baseline，SAM2 经微调后仍低于 PartGen，原因主要在于 SAM 是确定性模型，无法捕捉"艺术家意图的多解分布"。
5. **3D 语义分割（LERF / GARField / LangSplat / N2F2 / Uni3D）**：这些方法用 2D 特征蒸馏到 3D 神经场或 3DGS；本文完全在 2D 多视图扩散空间完成分割与补全，绕过 3D 特征融合的计算瓶颈。
6. **部件基表示（Neural Template / SPAGHETTI / PartNeRF / SALAD / NeuForm）**：通过显式模板/高斯混合表达部件；本文无需预定义部件拓扑，由扩散模型隐式建模合理分解。

---

## 局限性与未来方向
1. **部件语义标签缺失**：当前输出是颜色编码的 mask，不直接附带部件类别标签（如"车轮""把手"），需要额外 VLM 或人工标注。
2. **部件数量限制**：训练集过滤了 >10 部件的对象，对高复杂度资产（精密机械、建筑场景）的分解能力未评估。
3. **完全不可见部件的幻觉质量**：极端遮挡下补全依赖扩散先验，可能出现违反物理常识的结构（如汉堡内部出现"沙"）——这是论文主动展示的"能力"，但也是不确定性的体现。
4. **4 视角的覆盖局限**：正交 4 方位 + 20° 俯仰对复杂几何的细节仍可能不足，扩展至更多视角或稀疏视角联合推理是自然方向。
5. **未探索的部分编辑边界**：当前编辑实验仅演示了文本驱动的外观/形状修改，未见多部件间的约束编辑（如保持部件刚性连接）。

---

## 研究启发与可借鉴点
1. **"将 3D 非确定性任务转嫁为 2D 扩散采样"**：部件分割、补全的模糊性通过多次采样自然建模，启发可将其他 3D 结构化任务（如拓扑分解、装配关系推断）也用多视图扩散采样解决。
2. **随机颜色置换消解实例命名歧义**：不需要 anchor-based 或 matcher-based 的实例对齐，直接利用扩散模型的随机性"平均掉"命名问题，设计极简且有效。
3. **上下文注入策略**：补全网络以 $(I \odot M, I, M)$ 三元组为条件，"遮挡越重越依赖全局"这一设计直观、可复用到任何 inpainting / occlusion handling 任务。
4. **RM 无需微调即可重建部件**：说明好的 3D 重建先验对部件级别的输入也有通用性，后续可尝试更大 RM（如 GRM、LRM 升级版）直接对接。
5. **端到端可微的"3D→多视图渲染→分割→补全→3D"闭环**：若把渲染器与 RM 也纳入训练，可探索联合优化分割与重建的端到端方案。

---

## 关键术语表
- **Multi-view diffusion model**：以 2×2 网格形式一次性生成 4 个正交方位视图的扩散生成器，天然保证视图间 3D 一致性。
- **Contextual part completion**：同时以局部 mask 化视图、全局完整视图和 mask 为条件，对遮挡/不可见部件生成补全多视图的过程。
- **Large Reconstruction Model (LRM) / LightplaneLRM**：从 2D 多视图直接回归 3D 表示（ occupancy / 3DGS ）的前馈神经网络；本文取其预训练版本作为部件重建器。
- **Score Distillation Sampling (SDS)**：DreamFusion 的核心技术，利用 2D 扩散模型的 score 对 3D 场景进行逐次优化；本文完全不依赖 SDS，改用两阶段 feed-forward 方案。
- **Color-coded segmentation map**：将每个部件渲染为不同固定颜色的多视图 RGB 图，便于扩散模型直接输出 mask 而无需额外头。
- **Iverson bracket**：逻辑表达式 $[P]$ 当 $P$ 为真时取 1、否则取 0，用于从深度图直接构造 binary mask。

---

## 可复现要素
- **数据集**：140k 商业授权艺术家 3D 资产（作者声明来自 "commercial source licensed for AI training"）；筛选后 45k 对象、210k 部件。**非公开**。
- **代码**：论文主页 silent-chen.github.io/PartGen 提供项目页，但全文未明确开源仓库链接；需关注后续更新。
- **权重**：使用 AssetGen [73] 的多视图生成器和 LightplaneLRM [5] 作为基础；分割/补全网络微调权重未公开声明。
- **关键超参**：渲染分辨率 $H \times W$、颜色数 $Q \ge S$、多视图网格排列（2×2）、视点（4 个正交方位 + 20° 俯仰）、最小部件体积阈值 5%、最大部件数 10 —— 论文中未给出具体数值，详见 supplement。
- **基线复现**：SAM2 微调细节（学习率、epochs 等）未在主文列出；Part123 官方代码可获取。

---
