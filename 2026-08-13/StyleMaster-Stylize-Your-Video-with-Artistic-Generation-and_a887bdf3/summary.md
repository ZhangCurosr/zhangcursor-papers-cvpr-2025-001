---
title: "StyleMaster-Stylize-Your-Video-with-Artistic-Generation-and"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Ye_StyleMaster_Stylize_Your_Video_with_Artistic_Generation_and_Translation_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 09:44:22"
field: "视频生成与风格化"
keywords: ["视频风格迁移", "艺术生成", "风格提取", "motion adapter", "ControlNet", "model illusion", "对比学习"]
innovations: ["利用model illusion自动生成绝对风格一致的配对数据集", "基于文本-patch相似度过滤的内容-纹理解耦风格提取", "负系数motion adapter隐式增强视频风格化程度"]
benchmarks: ["CSD-Score", "ArtFID", "UMT-Score", "CLIP-Text", "VBench (MotionSmooth, DynamicQuality)"]
---

# 论文速读：StyleMaster-Stylize-Your-Video-with-Artistic-Generation-and-Translation

## 一句话总结
StyleMaster 提出了一种结合全局与局部风格特征的提取机制，利用 model illusion 生成绝对风格一致的对比数据集，并通过 motion adapter 和 gray tile ControlNet 实现高质量的文本驱动风格化视频生成与视频风格迁移。

## 研究问题与动机
- **风格提取不足**：现有方法（如 VideoComposer、StyleCrafter）要么过度依赖全局风格而忽略局部纹理（如梵高笔触），要么直接复制参考图所有 patch 特征导致严重的内容泄漏（content leakage）。
- **风格数据集质量缺陷**：StyleTokenizer 使用的 Style30K 数据集在同一风格组内无法保证风格一致性（如同一风格组包含写实域与动画域图像），制约了对比学习的风格提取效果。
- **视频动态退化**：仅用图像数据训练的风格化模型直接应用于视频时，会出现时间闪烁（temporal flickering）和动态范围受限的问题。
- **风格迁移缺乏精确内容控制**：现有视频风格迁移方法多依赖深度估计 ControlNet 或逐帧处理，缺乏既简洁又精确的内容引导机制，且易破坏视频语义。

## 核心贡献（创新点）
- **局部 patch 选择 + 全局投影的双层风格提取模块**：通过文本-patch 相似度过滤内容相关区域、保留纹理特征，并结合 CLIP 后的 MLP 投影提取全局风格描述，从根本上解决内容泄漏问题。
- **首次利用 model illusion 自动生成绝对风格一致的配对数据集**：基于 VisualAnagrams 的像素重排幻觉特性，以近乎零成本生成无限数量的风格配对样本，解决了现有风格数据集组内不一致的核心瓶颈。
- **Motion adapter 训练策略赋能图像到视频的无缝迁移**：在静止视频上训练的 LoRA motion adapter，推理时通过负系数（α = -1）不仅增强动态范围，还隐式推动生成结果远离真实域、强化风格化程度。
- **Gray tile ControlNet 实现简洁而精确的视频风格迁移内容控制**：去除 tile 图像中的色彩信息，避免颜色干扰风格注入，在 DomoAI 等商业方法基础上实现了更优的语义保持与风格迁移效果。

## 方法详解

### 3.1 对比数据集构建（Model Illusion）
借鉴 VisualAnagrams 的 model illusion 特性：在 T2I 模型采样过程中，复制并变换（旋转/翻转）噪声图像形成并行进程，用不同 prompt 引导噪声预测，再将预测噪声还原并叠加输出噪声。由此生成的配对图像仅为像素重排，内容不同但**共享绝对一致的风格**。自动生成无限对风格-对象配对数据（如"一只水彩狗"和"一只水彩兔子"），无需人工标注。

### 3.2 全局风格描述提取
对 CLIP 图像编码器输出不做微调（保留泛化能力），仅在 embedding 后接一个 MLP 投影模块：
$$F_i = \text{CLIP}(I).\text{image\_embed}, \quad F_{\text{global}} = \text{MLP}(F_i)$$
使用 triplet loss 进行对比学习，以配对图像中一张为 anchor、另一张为 positive、其余任意图像为 negative：
$$\mathcal{L} = \sum_{n=1}^{N} \left[ \|f(F_{i,n}^{\text{anc}}) - f(F_{i,n}^{\text{pos}})\| - \|f(F_{i,n}^{\text{anc}}) - f(F_{i,n}^{\text{neg}})\| + \alpha \right]$$
投影后全局特征与所有 patch 的相似度分布更均匀（方差更小），避免只关注特定区域的偏差。

### 3.3 局部与全局风格融合
- **局部 patch 选择**：从 CLIP 提取的 $F_p \in \mathbb{R}^{256 \times 1024}$ 中，按与文本 prompt 特征 $F_{text}$ 的相似度排序，保留相似度最低的 k=15 个 patch 作为纹理特征：
$$F_p' = \text{concat}(F_p^i \mid i \in \text{argsort}(\text{similarity}(F_p, F_{text}))[:k])$$
再用 Q-Former 结构（N 个 learnable tokens + self-attention）聚合为 $F_{texture}$。
- **双路 cross-attention 注入**：将 $F_{texture}$ 与 $F_{global}$ 拼接为 $F_{style}$，通过 adapter 机制注入 DiT：
$$F_{out} = \text{TCA}(F_{in}, F_{text}) + \text{SCA}(F_{in}, F_{style})$$

### 3.4 Motion Adapter
对时序注意力块的每个权重矩阵 $W \in \{W_Q, W_K, W_V\}$ 添加 LoRA：
$$\widetilde{W} = W + \alpha \cdot A_t^{W,\text{down}} \cdot A_t^{W,\text{up}}$$
训练时使用 still videos，$\alpha=1$ 生成静态视频，$\alpha=-1$ 产生更强动态。**更关键的是**：由于训练数据为真实世界视频，负系数使生成结果远离真实域，隐式增强风格化程度。

### 3.5 Gray Tile ControlNet
采用 $N/2$ 个 vanilla DiT blocks 作为内容引导网络，去除 tile 图像的颜色信息转为灰度图，避免色彩干扰风格注入。各 vanilla DiT block 的输出在固定间隔处叠加到对应的 style DiT block 中提供内容约束。

## 实验与结果

### 实验设置
- **基座模型**：DiT-based 视频生成模型（3D causal VAE + DiT Blocks）
- **训练数据**：10K model-illusion 配对数据（全局投影）→ still videos（motion adapter）→ Laion-Aesthetics 图像（风格调制，40K iters，batch=160/GPU）→ 20K iters（gray tile ControlNet）
- **硬件**：8×A800 GPU，约两天完成训练
- **超参**：text CFG=12.5，style CFG=6，motion adapter scale=−0.3

### 图像风格迁移（Table 1）
| 方法 | CSD-Score ↑ | ArtFID ↓ | FID ↓ |
|------|-------------|----------|-------|
| StyleID | 0.40 | 38.57 | 23.91 |
| InstantStyle | 0.32 | 42.48 | 24.59 |
| CSGO | 0.35 | 41.42 | 25.71 |
| **Ours** | **0.45** | **36.89** | **22.11** |

StyleMaster 在 CSD（+12.5% vs StyleID）、ArtFID（−4.36）、FID（−1.80）三项核心指标上均取得最优，显著提升风格相似度与感知质量。

### 风格化视频生成（Table 2）
| 方法 | CLIP-Text ↑ | UMT-Score ↑ | CSD-Score ↑ | VisualQuality ↑ | DynamicQuality ↑ | MotionSmooth ↑ |
|------|-------------|-------------|-------------|-----------------|-------------------|----------------|
| VideoComposer | 0.057 | −2.268 | 0.680 | 2.159 | 2.284 | 0.975 |
| StyleCrafter | 0.294 | 1.994 | 0.448 | 2.140 | 2.306 | 0.973 |
| **Ours** | **0.305** | **2.329** | **0.463** | **2.370** | **2.496** | **0.994** |

StyleMaster 在文本-视频对齐（CLIP-Text、UMT-Score）、视觉质量、动态质量和运动平滑度五项指标上全面领先；CSD 略低于 VideoComposer（因后者直接拷贝参考图内容导致分数虚高，牺牲了文本对齐）。

### 视频风格迁移
与商业应用 DomoAI 对比，StyleMaster 在保持视频语义完整性方面显著优于 DomoAI（后者存在语义破坏问题，见 Figure 9）。

### 消融实验（Tables 3–4）
- **风格提取模块**（Table 3）：Global Projection 将 UMT-Score 从 0.892 提升至 2.337；基于文本相似度的 patch 选择（B5 vs B4 random）同时维持风格相似度并提升文本对齐；全局+局部融合（B6）达到最佳 CSD=0.463。
- **Motion Adapter Scale**（Table 4）：α 从 0 到 −1，CSD 从 0.443 升至 0.465，DynamicDegree 从 1.371 升至 20.559；但 α≤−0.3 后 UMT-Score 和 MotionSmooth 下降，故最终选用 α=−0.3 取得视觉质量最优。
- **ControlNet 条件**（Figure 10）：Gray tile 在避免色彩干扰与提供布局引导之间取得最佳平衡，优于 Canny（细节过多、布局弱）和 RGB tile（颜色干扰导致输出偏暗）。

## 相关工作脉络
- **StyleTokenizer [24]**：微调 CLIP 进行对比学习风格提取，但 Style30K 数据集组内风格不一致；本文改进为 post-hoc MLP 投影 + 自生成配对数据集，确保绝对风格一致性。
- **InstantStyle [44] / InstantStylePlus [45]**：无训练（training-free）方法，识别特定层注入风格；但子优风格提取器导致精度不足；本文方法需训练但风格精度显著提升。
- **CSGO [51]**：基于 B-LoRA 三元组数据的全局风格提取，无法保留局部纹理；本文联合全局+局部双路特征，填补纹理细节缺失。
- **StyleCrafter [28]**：使用 Q-Former 从参考图提取风格描述，但忽视局部纹理且仅支持风格化生成不支持迁移；本文在其 Q-Former 基础上引入 patch 选择并扩展至迁移任务。
- **VideoComposer [46]**：直接注入参考图全部 token 导致严重内容泄漏，文本对齐差（UMT=-2.268）；本文通过解耦内容-风格实现兼顾。
- **StillMoving [4]**：在静止视频上训练 motion adapter 以适配任意图像定制模型；本文在此基础上利用负系数隐式增强风格化程度。

## 局限性与未来方向
- **仅依赖静态参考图像风格**：当前方法聚焦图形风格，未考虑视频中的动态风格元素（如粒子特效、运动轨迹特征）；未来需探索从参考视频中提取并迁移动态风格。
- **Gray tile ControlNet 的计算开销**：额外引入 $N/2$ 个 vanilla DiT blocks 增加推理延迟，可能在实时场景下受限。
- **Model illusion 对基础 T2I 模型的依赖**：生成配对数据集的质量受限于所用 T2I 模型的幻觉能力，不同模型间泛化性有待验证。
- **风格域覆盖有限**：当前在 Laion-Aesthetics 等通用美学数据集上训练，对高度专业化艺术风格（如特定画家笔触）的捕捉能力仍需扩展。

## 研究启发与可借鉴点
- **Model illusion 数据生成范式**：利用预训练模型的幻觉特性自动生成带绝对标签一致性的配对数据，可迁移至其他需要细粒度风格/属性解耦的研究场景（如风格分解、属性编辑）。
- **文本-patch 相似度过滤策略**：基于 prompt 与 patch 特征相似度的内容-纹理分离方法简洁有效，可直接复用于其他图文联合建模任务中的噪声抑制。
- **负系数 motion adapter 设计**：将风格化目标转化为"远离真实域"的方向性引导，这一思路可推广至其他需要隐式增强生成风格的领域（如风格化图像合成、跨域翻译）。
- **Gray tile 替代 RGB/Canny 的思路**：在风格迁移任务中去除外部引导信号的冗余通道（颜色），避免信息干扰，这一原则可适用于其他多条件联合生成场景。

## 关键术语表
- **Model Illusion**：利用预训练 T2I 模型在采样过程中对噪声图像的感知偏差，通过像素重排生成外观变化但风格一致的配对图像的技术。
- **Content Leakage**：风格迁移过程中参考图像的语义内容（如人物面部、物体形状）被错误地复制到生成结果中的现象。
- **Dual Cross-Attention (Decoupled)**：分别对文本特征和风格特征进行独立 cross-attention 计算再融合的注入机制，实现内容与风格的解耦控制。
- **CSD Score**：衡量生成结果与参考图像风格相似度的量化指标（Style Composition Distance）。
- **ArtFID**：结合风格相似性与内容保真度的感知评价指标，与人类偏好相关性较强。
- **UMT Score**：基于 Unified Multi-modal Transformer 的文本-视频对齐评估指标。
- **Gray Tile ControlNet**：去除颜色信息的灰度 tile 图像作为内容引导条件的 ControlNet 变体，避免色彩干扰风格注入。
- **Q-Former**：由 learnable query tokens 与 self-attention 构成的特征聚合结构（源自 BLIP-2），用于从离散 patch 特征中提取紧凑表示。

## 可复现要素
- **数据集**：Laion-Aesthetics（图像训练）；测试集基于 StyleCrafter 扩展（12 风格图 × 16 prompt = 192 配对）；model illusion 配对数据（10K，自生成）；论文未提及开放获取渠道。
- **代码**：论文未提及代码开源情况。
- **权重**：论文未提及预训练权重开源情况。
- **关键超参**：text CFG=12.5，style CFG=6，patch 保留数 k=15，Q-Former learnable tokens 数 N（论文未明确），motion adapter scale α=−0.3，batch size=160/GPU，A800×8 训练约 2 天。
