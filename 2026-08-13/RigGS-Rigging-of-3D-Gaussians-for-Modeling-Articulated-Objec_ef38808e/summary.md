---
title: "RigGS-Rigging-of-3D-Gaussians-for-Modeling-Articulated-Objec"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Yao_RigGS_Rigging_of_3D_Gaussians_for_Modeling_Articulated_Objects_in_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 07:47:54"
field: "动态3D重建与神经渲染"
keywords: ["3D Gaussian Splatting", "rigging", "articulated object modeling", "novel view synthesis", "skeleton extraction", "template-free animation"]
innovations: ["无模板自动骨架提取：融合运动、几何与语义信息的启发式稀疏骨架构建算法", "骨架驱动的LBS+姿态依赖细节变形：可学习skinning weights与pose-dependent detail MLP联合建模", "端到端可编辑动画生成：支持姿态编辑、运动插值与跨对象运动迁移的高保真渲染"]
benchmarks: ["D-NeRF", "DG-Mesh", "ZJU-MoCap"]
---

# 论文速读：RigGS: Rigging of 3D Gaussians for Modeling Articulated Objects in Videos

## 一句话总结
RigGS 是一种**无模板依赖的自动化 rigging 范式**，从单目视频自动提取稀疏 3D 骨架并将其绑定到 3D Gaussian 表示上，实现关节物体的**新视角合成、姿态编辑、运动插值与运动迁移**，同时支持实时高质量渲染。

## 研究问题与动机
- **现有方法依赖模板先验**：SMPL、MANO、SMAL 等参数化模型仅适用于特定类别，无法处理穿戴复杂服饰的人体、异形动物、变形玩具等多样化物体。
- **已有无模板方法存在局限**：CASA 依赖 3D 模型数据库检索；Magicpony 等方法需预定义骨架树结构；AP-NeRF / SK-GS 依赖 3D 重建质量且难以生成合理新动作。
- **神经骨（Neural Bones）语义缺失**：BANMo、BAGS 等方法虽然无需预设骨架树，但骨骼节点无语义信息，生成的骨架结构不合理，难以用于后续可控编辑。
- **动态物体 rigging 缺乏可编辑性**：多数神经渲染方法只能做新视角合成，无法将骨架与可变形几何绑定以实现 pose editing / motion transfer。

## 核心贡献（创新点）
1. **骨架感知节点控制变形场（Skeleton-aware Node-controlled Deformation）**：结合 canonical 3D Gaussian 与时空变形场，同步实现动态物体重建与候选骨架节点生成，无需任何模板先验。
2. **启发式 3D 稀疏骨架提取算法**：融合几何距离、运动轨迹相似度与 DINOv2 语义标签，从密集候选节点中构建拓扑合理的稀疏骨架树，优于依赖重建质量的 MAT 方法。
3. **骨架驱动的 LBS + 姿态依赖细节变形模型**：引入可学习的 skinning weights 修正纯几何距离绑定的不足，并设计 pose-dependent detail deformation MLP 捕获布料褶皱等细粒度形变，使新生成的动作更具物理合理性。
4. **端到端可编辑的动画生成框架**：首次在无模板条件下实现从视频到可驱动骨架模型的完整 pipeline，支持实时渲染与新动作合成。

## 方法详解
### 3.1 初始化阶段
- **Canonical 3D Gaussian 表示**：场景参数化为 $\mathcal{G} = \{\mu_i, \mathbf{q}_i, \mathbf{s}_i, \sigma_i, sh_i\}$，通过标准 3D Gaussian Splatting 渲染。
- **Skeleton-aware Node-controlled Deformation**（公式 3-7）：
  - 每个 3D Gaussian 中心 $\mu_i$ 由其最近 $k$ 个骨架感知节点 $\mathbf{c}$ 通过高斯加权 LBS 变形：$\mu_i^t = \sum_{\mathbf{c} \in \mathcal{N}(\mu_i)} w_{\mu_i,\mathbf{c}} (\tilde{\mathbf{R}}_{\mathbf{c}}^t (\mu_i - \mathbf{c}) + \mathbf{c} + \tilde{\mathbf{t}}_{\mathbf{c}}^t)$
  - 旋转通过四元数加权融合：$\mathbf{q}_i^t = (\sum w_{\mu_i,\mathbf{c}} \mathbf{r}_{\mathbf{c}}^t) \otimes \mathbf{q}_i$
  - 节点变换由 MLP $F_\Theta$ 预测，输入为节点的 positional encoding 和时间编码。
- **初始化损失**：$L_{\mathrm{init}}^t = L_{\mathrm{render}}^t + w_{\mathrm{arap}} L_{\mathrm{arap}}^t + w_{\mathrm{proj}} L_{\mathrm{proj}}^t$，其中 $L_{\mathrm{proj}}$ 基于 2D 骨架投影的 Chamfer distance。

### 3.2 由粗到细的 3D 骨架构建
- **Canonical Shape 选择**：选取节点运动轨迹均值最小化的时刻 $t^*$ 作为 canonical shape（公式 $t^* = \arg\min_t \sum_{\mathbf{c}} \|\mathbf{c}^t - \overline{\mathbf{c}}\|_2$）。
- **密集骨架构建**：
  - FPS 均匀采样候选节点
  - 构建完全图，边权 $\beta_{ij} = \sum_t \|\mathbf{c}_i^t - \mathbf{c}_j^t\| / |\mathcal{T}|$（融合位置与运动信息）
  - Prim 算法求 MST，去除冗余分叉，合并相邻 junction
- **骨架简化**：
  - 优先选取 endpoint 和 junction 作为潜在关节
  - 利用 DINOv2 语义标签确保**语义对称性**（同侧对称点保留/剔除）
  - BFS 寻找最长路径的 junction 作为根节点，建立父子关系得到稀疏骨架树 $\mathcal{T} = \{\mathcal{I}, \mathcal{A}\}$

### 3.3 骨架驱动的动态建模
- **可学习 LBS 粗变形**（公式 9-11）：
  - $\hat{\mu}_i^t = \mathbf{T}_1^t (\sum_{j=2}^{|\mathcal{I}|} \hat{\omega}_{i,j} \mathbf{T}_j^t \overline{\mu}_i)$
  - 皮肤权重 $\hat{\omega}_{i,j}$ 由几何距离 $\exp(-D^2(\mu_i, b_j)/2\nu_j^2)$ 和 MLP 学习的缩放因子 $\eta_{i,j}$ 共同决定，可纠正纯距离加权的不合理绑定
  - 各关节旋转 $\hat{\mathbf{r}}_j^t$ 由 MLP $F_\Phi$ 基于时间编码预测
- **姿态依赖的细节变形**（Pose-dependent Detail Deformation）：
  - MLP $F_\Pi(\mu_i, \{\hat{\mathbf{r}}_j^t\})$ 学习偏移量 $\delta_{i,t}$，最终位置 $\mu_i^t = \hat{\mu}_i^t + \delta_{i,t}$
  - 输入为姿态（而非时间），保证新动作生成时的细节一致性
- **最终损失**：$L^t = L_{\mathrm{render}}^t + w_{\mathrm{proj}}^t L_{\mathrm{proj}}^t + w_{\mathrm{detail}}^t L_{\mathrm{detail}}^t + w_{\mathrm{id}} L_{\mathrm{id}}^t$
  - $w_{\mathrm{proj}}^t = 10^{-3} \cdot \exp(-(L_{\mathrm{proj}}^t)^2 / 2\xi^2)$：自适应权重，抑制不准确 2D 骨架的误导
  - $L_{\mathrm{detail}}^t$：L2 正则化细节偏移，在 $t^*$ 时刻权重为 $10^3$ 以保证 canonical shape 一致性
  - $L_{\mathrm{id}}^t$：在 $t^*$ 时刻约束所有关节旋转趋向单位四元数

## 实验与结果
### 数据集
- **D-NeRF**：6 个序列（排除 Bouncing balls）
- **DG-Mesh**：5 个序列（排除 Torus2sphere）
- **ZJU-MoCap**：6 个主体（377, 386, 387, 392, 393, 394）

### 新视角合成对比（表 1, 2）
| 数据集 | 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|--------|------|--------|--------|---------|
| D-NeRF | SC-GS | 43.04 | 0.998 | 0.0066 |
| D-NeRF | **Ours** | **40.82** | **0.996** | **0.0112** |
| D-NeRF | 4D-GS | 33.25 | 0.989 | 0.0233 |
| DG-Mesh | **Ours** | **37.65** | **0.991** | **0.0169** |
| DG-Mesh | SC-GS | 38.96 | 0.993 | 0.0136 |
| ZJU-MoCap | AP-NeRF | 25.62 | 0.919 | 0.0934 |
| ZJU-MoCap | **Ours** | **33.54** | **0.975** | **0.0327** |

- **最强结果**：在 D-NeRF 上 PSNR=40.82，SSIM=0.996，LPIPS=0.0112，仅次于 SC-GS（43.04/0.998/0.0066）但大幅领先其余方法（4D-GS 33.25、AP-NeRF 30.94）
- **ZJU-MoCap 提升**：相比 AP-NeRF，PSNR 提升 **+7.92 dB**，LPIPS 降低约 65%
- **消融**：去掉 2D 投影损失导致骨架无法嵌入形体；固定权重导致骨架穿模；各组件均有贡献

### 编辑能力展示
- 成功实现**姿态编辑**、**运动插值**、**运动迁移**（Fig. 8-9）
- 相比 AP-NeRF，生成的骨架更合理、skinning weights 更平滑无关节不连续

## 相关工作脉络
1. **参数化模板方法（SMPL/MANO/SMAL）**：依赖特定类别先验，泛化性差；RigGS 完全无模板，适用于任意关节物体。
2. **BANMo / BAGS / DreaMo**（神经骨方法）：使用无语义的 learnable bones，结构不清晰难以编辑；RigGS 通过启发式算法构建具语义的稀疏骨架树。
3. **AP-NeRF [28]**：同样无模板但依赖 3D 重建表面的 Medial Axis Transform 提取骨架，对重建质量敏感；RigGS 直接从 2D 投影约束学习骨架节点，不依赖高质量 3D 网格。
4. **SK-GS [29]**：将 super-points 视为刚体部分发现骨架，属于基于体素的离散方法；RigGS 使用连续 3D Gaussian 表示，渲染质量更高且 deformation field 更光滑。
5. **SC-GS [7]**：新视角合成质量 SOTA，但使用 512 个密集控制点且变形与时间相关，无法支持可控编辑；RigGS 在略低渲染质量下换取了可编辑性。
6. **CASA [34] / Magicpony [33]**：CASA 依赖 3D 动物数据库检索，Magicpony 需预定义骨架树；RigGS 完全从零样本视频学习。

## 局限性与未来方向
- **稀疏视角与相机位姿误差敏感**：论文自述在 sparse viewpoints 和不准确 camera pose 估计下效果受限
- **过大运动场景失效**：剧烈形变超出 LBS 局部线性假设的适用范围
- **姿态无关外观未建模**：当前模型仅关注几何形变，未考虑 pose-dependent appearance变化
- **未来方向**：探索文本/图像等多模态输入进行语义驱动的骨架编辑；结合更鲁棒的 3D 重建技术处理挑战性输入。

## 研究启发与可借鉴点
1. **无模板骨架提取的启发式策略**：融合运动轨迹距离 + DINOv2 语义对称性约束的骨架简化方法，可迁移到其他 3D 结构学习任务（如机械臂标定、动物姿态估计）。
2. **pose-dependent detail deformation 设计**：将细节变形网络输入设为姿态而非时间，保证新动作的物理合理性——这一思路可用于 NeRF/Gaussian 动画生成中的细粒度形变建模。
3. **自适应投影损失权重**：$w_{\mathrm{proj}}^t$ 基于 2D 骨架质量动态调整，有效抑制 noisy supervision 的干扰，可借鉴到任何涉及 2D-3D 对齐的损失设计中。
4. **可学习 skinning weights 修正几何距离**：通过 MLP 学习 $\eta_{i,j}$ 修正纯距离加权的不足，为 LBS 绑定提供了数据驱动的正则化思路，可应用于任何基于骨骼的形变任务。
5. **与团队方向结合机会**：若团队关注动态场景理解或可控动画生成，可将 RigGS 的骨架驱动范式与 diffusion-based motion generation 结合，实现"文本描述 → 骨架动画 → 3D Gaussian 渲染"的端到端管道。

## 关键术语表
**Rigging**：将可变形几何（如网格或 3D Gaussian）绑定到动画骨架的过程，使物体能够按骨架运动而自然形变。
**Canonical 3D Gaussian Representation**：表示物体在 reference pose（通常为运动均值时刻）下的 3D Gaussian 集合，作为变形场的基础参考形态。
**Skeleton-aware Node-controlled Deformation**：用稠密节点集合控制 canonical 3D Gaussian 时空变形的模块，节点同时作为候选骨架点和变形控制点。
**Linear Blend Skinning (LBS)**：经典的基于骨架的形变方法，每个顶点的变形由其邻近骨骼的仿射变换加权blend得到。
**Skning Weights**：描述每个 3D Gaussian 受哪些骨骼影响的权重，本文通过几何距离+MLP学习得到可微分的 skinning weights。
**Chamfer Distance**：衡量两组点集之间相似度的指标，本文用于约束 3D 骨架节点投影与 2D 轮廓骨架的对齐。
**DINOv2**：Meta AI 提出的自监督视觉特征提取器，本文用于获取物体的语义标签以辅助骨架对称性优化。
**Arigid-as-possible (ARAP) Loss**：正则化项，约束变形过程中局部刚体变换尽量保持，避免非物理的过度形变。

## 可复现要素
- **数据集**：D-NeRF（公开）、DG-Mesh（公开）、ZJU-MoCap（公开）
- **代码开源**：项目主页 https://yaoyx689.github.io/RigGS.html（论文未明确声明 GitHub，需访问主页确认）
- **关键超参**：
  - MLP 层数：8 层，隐层维度 256
  - $w_{\mathrm{arap}}$：初始化阶段 ARAP 正则化权重（论文未明确数值，见 Supplement）
  - $w_{\mathrm{proj}}$：投影损失权重，初始 $10^{-3}$，后续自适应调整
  - $w_{\mathrm{detail}}^{t^*}$：canonical 时刻细节损失权重 $10^3$，其余时刻为 1
  - $\xi$：投影损失中位数的一半，用于自适应权重计算
- **训练设备**：单卡 NVIDIA RTX A6000 GPU
- **优化器**：Adam
