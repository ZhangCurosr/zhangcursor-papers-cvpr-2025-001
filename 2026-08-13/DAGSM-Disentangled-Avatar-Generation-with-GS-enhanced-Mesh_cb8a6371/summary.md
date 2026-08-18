---
title: "DAGSM-Disentangled-Avatar-Generation-with-GS-enhanced-Mesh"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Zhuang_DAGSM_Disentangled_Avatar_Generation_with_GS-enhanced_Mesh_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 07:00:03"
field: "文本驱动3D数字人生成"
keywords: ["text-to-3D human", "disentangled avatar", "Gaussian splatting", "GS-enhanced mesh", "virtual try-on", "texture refinement"]
innovations: ["提出顺序解耦的GSM生成管线，支持换装与物理仿真动画", "设计SAM-based语义过滤实现人体-衣物自动分离", "引入跨视角注意力与入射角加权去噪实现视图一致性纹理细化"]
benchmarks: ["CLIP similarity (ViT-L/14, ViTbigG-14)", "User study (visual quality, text alignment, animation realism)"]
---

# 论文速读：DAGSM-Disentangled-Avatar-Generation-with-GS-enhanced-Mesh

## 一句话总结
本文提出 DAGSM，一种从文本提示生成**解耦数字化人**的新方法，将人体与服饰分别用 GS-enhanced mesh（GSM）表示并顺序生成，从而支持换装、物理仿真动画与高质量纹理编辑。

## 研究问题与动机
1. **单模型生成的不可替换性**：现有文本驱动 3D 人体生成方法将身体与所有衣物作为一个整体优化，无法进行换装操作，且衣物紧贴身体导致动画不真实。
2. **预设拓扑限制宽松服装**：基于 SMPL-X 等固定拓扑的方法难以生成裙装等与身体拓扑差异较大的宽松衣物。
3. **SDS/RFDS 纹理质量低**：直接由 SDS 蒸馏得到的纹理常出现过平滑、颜色过饱和、跨视图不一致等问题。
4. **用户控制力弱**：缺乏对复杂服装组合、局部材质细节的精细控制能力，限制了在虚拟试穿、游戏等场景的应用。

## 核心贡献（创新点）
1. **顺序解耦生成管线**：先生成裸体 GSM，再基于身体依次生成每件衣物的 GSM；相比 DreamWaltz/HumanGaussian 的"一步整体生成"，本方法天然支持换装与独立编辑。
2. **GS-enhanced mesh（GSM）混合表示**：将 2DGS 绑定在代理网格上，几何属性随网格变形，同时用 UV 贴图存储颜色/透明度以支持直接纹理编辑；相比纯 3DGS 方法，支持物理布料仿真并获得更真实的动画。
3. **SAM-based 语义过滤实现人体-衣物分离**：通过为每个 2DGS 赋予类别属性并使用 SAM 提取语义掩码计算 MSE 损失，迭代剔除不属于衣物的噪声高斯，实现清晰的 body-garment 和 garment-garment 边界。
4. **跨视角注意力 + IAW-DE 的视图一致性纹理细化**：将视频扩散模型中的跨视角注意力引入 SD3 去噪过程，并用表面法线与相机方向的余弦相似度作为权重图进行入射角加权去噪，解决多视图纹理风格不一致与模糊问题。

## 方法详解
**总体流程（三阶段）**：身体生成 → 衣物生成 → 视图一致性细化。

**4.1 GSM 表示**
- 每个 2DGS 绑定在网格的一个三角面上，局部坐标系由三角形质心、一条边方向、法向量及其叉积定义。
- 2DGS 位置用重心坐标 $(\lambda_1, \lambda_2)$ 和法向偏移 $z$ 表示：$\hat{\mu} = \lambda_1 x_A + \lambda_2 x_B + (1-\lambda_1-\lambda_2)x_C + z\vec{n}$，旋转与缩放随网格顶点和面片面积比更新。
- 使用两张 UV 特征图 $\mathcal{U}_c$（RGB）和 $\mathcal{U}_\alpha$（透明度）存储外观信息，便于直接编辑。

**4.2 身体生成**
- 基于 SMPL-X 参数化人体网格 $\mathcal{M}$，添加可优化顶点位移 $\mathcal{D}$ 得到 $\hat{\mathcal{M}}$，在每个三角面上均匀放置 $(n^2/2)$ 个表面 2DGS。
- 几何分支优化 $\theta_1=\{\beta, \mathcal{D}\}$，以法线图 $\mathcal{N}$ 为渲染目标计算 $\mathcal{L}_{rfds}^N$。
- 颜色分支优化 $\theta_2=\{u, s, r, \mathcal{U}_c, \mathcal{U}_\alpha\}$，以 RGB 图 $\mathcal{T}_b$ 为渲染目标计算 $\mathcal{L}_{rfds}^{I_b}$，并加位置/尺度/旋转正则化：$\mathcal{L}_{\mathcal{G}_b} = \mathcal{L}_{rfds}^{I_b} + \lambda_p \mathcal{L}_p + \lambda_s \mathcal{L}_s + \lambda_r \mathcal{L}_r$。

**4.3 衣物生成**
- **初始阶段**：用未被绑定的原始 2DGS $\mathcal{G}_m$ 表示衣物，与固定的 $\mathcal{G}_b$ 合并渲染，使用 $\mathcal{L}_{rfds}^{I_a}$ 对齐文本 $T_2$，并加两点正则：(i) 高斯中心到身体网格表面的点面距离损失 $\mathcal{L}_{dis} = ||\hat{\mu}-p_m||_2$；(ii) 渲染法线图与其高斯模糊版本的平滑损失 $\mathcal{L}_{smooth} = ||G(\mathcal{T}_n)-\mathcal{T}_n||_2^2$。
- **SAM-based 过滤**：给每个 $\mathcal{G}_m$ 的高斯附加类别属性 $o$，渲染语义图 $\mathcal{T}_o$，与 SAM 掩码 $\mathcal{M}$ 计算 MSE 损失 $\mathcal{L}_{sam}$ 来优化 $o$；每 500 次迭代移除 $o<0.5$ 的高斯。总损失：$\mathcal{L}_{\mathcal{G}_m} = \mathcal{L}_{rfds}^{I_a} + \mathcal{L}_{sam} + \lambda_{dis}\mathcal{L}_{dis} + \lambda_{smooth}\mathcal{L}_{smooth}$。
- **网格提取**：对 $\mathcal{G}_m$ 的多视角深度图使用 TSDF 算法重建衣物网格，去除身体内部不可见面，简化至 10k 面片并进行 Laplacian 平滑，生成 UV 映射。
- **纹理生成**：将 2DGS $\mathcal{G}_g$ 绑定到提取的衣物网格上，分别以单衣物图 $\mathcal{T}_g$ 和全身图 $\mathcal{T}_{\dot{a}}$ 为目标计算 RFDS 损失，优化 $\mathcal{G}_g$ 的 UV 特征图。

**4.4 视图一致性细化**
- **跨视角注意力**：以规范视角 $\gamma_0$ 和前序视角 $\gamma_{i-1}$ 的特征作为 K' 和 V' 的参考，与当前视角特征 $z_{\nu_i}$ 拼接，保证多视角纹理风格一致。
- **IAW-DE（入射角加权去噪）**：构建权重图 $\mathcal{W}_i$，每个像素值为该点对应网格法线与反向相机方向的余弦相似度，经多视图聚合和温度缩放 Softmax 归一化；对高权重像素施加更多噪声和去噪迭代，使细化聚焦于"最清晰观测"的区域。损失为加权 MSE：$\mathcal{L}_{ref} = ||\mathcal{W}_i(\hat{\mathcal{V}}_i - \mathcal{V}_i)||_2^2$。

## 实验与结果
- **实现**：基于官方 2DGS 代码，RFDS loss 在 Stable Diffusion 3 上计算；渲染分辨率 1024×1024，$\lambda_p=\lambda_s=10,\lambda_r=1,\lambda_{dis}=1,\lambda_{smooth}=100$；A100 40GB。身体生成 3K 迭代（~60min），衣物原始 2DGS 优化 2K 迭代（~30min），纹理生成 2K 迭代（~20min），视图细化 16min（8 视角各 45°）。
- **基线**：DreamWaltz（单 NeRF+SDS）、HumanGaussian（3DGS+SD fine-tune）、SO-SMPL（SDS 优化传统网格，解耦但受限于固定拓扑）。
- **CLIP 相似度**（ViT-L/14）：DAGSM **28.8** vs DreamWaltz 24.2 / HumanGaussian 27.3 / SO-SMPL 25.6；（ViTbigG-14）：**45.8** vs 41.9 / 43.9 / 42.0。
- **用户研究（50 人）**：视觉质量 DAGSM 获 **74.0%** 投票，文本对齐 **76.8%**，动画真实性 **68.4%**，均大幅领先。
- **定性结论**：DAGSM 生成纹理质量更高、文本遵循更准确、动画中衣物形变更真实（如绿裙的褶皱变化）；基线存在结构问题（DreamWaltz 低分辨率/身体断裂）、意外生成物（HumanGaussian 产生篮球和运动鞋）、无法生成超出 SMPL-X 拓扑的服装（SO-SMPL 无法生成裙装）。

## 相关工作脉络
1. **DreamFusion / SDS 系列**：以 2D 扩散模型先验指导 3D 优化的开创性工作；DAGSM 沿用并扩展至 RFDS loss，但引入人体先验和 GSM 表示解决单模型局限性。
2. **HumanGaussian / DreamWaltz / DreamAvatar**：直接以 3DGS/NeRF 表征完整着装人体；DAGSM 与之本质区别在于将身体与每件衣物解耦为独立 GSM，支持换装。
3. **SO-SMPL**：早期解耦方法，通过裁剪 SMPL-X 网格区域生成衣物；受限於预设拓扑无法处理裙装等异拓扑服装，DAGSM 通过自由 2DGS 初始化 + TSDF 提取克服此限制。
4. **GaussianAvatars / SplattingAvatar**：将 3DGS 绑定网格做动画avatar重建；DAGSM 借鉴 GSM 思路但面向**文本生成**而非重建，并引入解耦衣物生成与视图一致性细化。
5. **Texfusion / Texture**：文本驱动的 3D 纹理合成；DAGSM 的视图一致性细化与之目标类似，但以多视角渲染+SA-3D  diffusion 为先验，并结合了入射角加权策略。
6. **TELA**：用多 NeRF 层叠表示身体与衣物；DAGSM 以显式网格+2DGS 替代隐式 NeRF，从而支持物理仿真动画。

## 局限性与未来方向
- **TSDF 仅重建衣物外表面**：无法恢复口袋等内部结构细节（论文自述）。
- **顺序生成效率偏低**：身体~60min + 衣物~50min + 细化~16min = 约 2h+，对实时应用不友好。
- **依赖 SAM 分类**：衣物-身体分离质量受 SAM 掩码精度影响，对半透明/重叠严重的衣物可能不够鲁棒。
- **多件衣物同时生成未深入探索**：当前实验聚焦单件上衣/下装，复杂多层穿搭（如外套+衬衫）的协同优化待验证。
- **未来方向**：加速生成、支持多层穿搭自动分割、增强半透明/复杂材质（蕾丝、毛呢）的渲染 fidelity、与虚拟试穿 pipeline 集成。

## 研究启发与可借鉴点
1. **"自由 2DGS 初筛 + TSDF 提取网格"的生成范式**：先用无拓扑约束的 2DGS 学习形状，再转为可动画的网格，可迁移至其他 3D 内容生成任务（如鞋子、包袋等配饰生成）。
2. **SAM-based 语义过滤用于组件分离**：将 SAM 作为语义标签生成器引入扩散优化循环，是一种通用且轻量的"解耦信号"，可推广至场景级 3D 生成的对象分离。
3. **跨视角注意力在扩散去噪中的引入**：受视频扩散模型启发，将已渲染视角特征拼接到 K/V 中，可有效缓解 3D 一致性问题；此技巧可复用于任意视图序列的纹理细化任务。
4. **入射角加权去噪（IAW-DE）**：基于表面法线与相机方向的余弦相似度指导去噪强度，是一种简单有效的"可信区域优先"策略，可推广至多视图立体一致性的纹理增强。
5. **UV 贴图存储 2DGS 外观**：将高斯颜色/透明度编码为 UV 特征图而非原始属性，兼顾了可编辑性与渲染效率，适合需要后期手工修图的 pipeline。

## 关键术语表
**GSM（GS-enhanced mesh）**：将 2D Gaussian Splatting 绑定在参数化网格上的混合 3D 表示，高斯几何随网格变形，外观存储在 UV 贴图中。
**RFDS loss**：基于 Rectified Flow 扩散模型的 Score Distillation Sampling 损失，用于将 2D T2I 先验蒸馏到 3D 表征优化中。
**2DGS**：将 3D 高斯压缩为 2D 定向高斯盘面的表示，更适合表面建模，减少 3DGS 的多视角不一致性。
**SAM-based filtering**：利用 Segment Anything Model 生成语义掩码，指导优化过程中剔除不属于目标组件（如衣物）的噪声高斯。
**IAW-DE（Incident-Angle-Weighted Denoising）**：根据网格表面法线与相机方向的余弦相似度构建权重图，对"正对相机"的区域施加更强去噪细化的策略。
**Cross-view attention**：在扩散模型去噪时，将规范视角和前序视角的 latent 特征拼接到当前视角的 K/V 计算中，以保持跨视角纹理风格一致。
**TSDF（Truncated Signed Distance Function）**：从多视角深度图重建封闭网格表面的经典算法，本文用于从 2DGS 提取衣物网格。
**SMPL-X**：包含手部、面部和身体的可变形参数化 3D 人体网格模型，作为身体几何先验使用。

## 可复现要素
- **数据集**：实验使用自生成样本评估，未使用公开标准数据集；动画驱动使用 AIST++ 舞蹈 pose 序列。
- **代码/权重**：论文未声明开源代码或预训练权重；使用官方 2DGS 代码 [23] 与 Stable Diffusion 3 [16]。
- **关键超参**：$\lambda_p=\lambda_s=10,\ \lambda_r=1,\ \lambda_{dis}=1,\ \lambda_{smooth}=100$；渲染分辨率 1024×1024；衣物网格简化至 10k 面片；身体生成 3K 迭代，衣物优化 2K 迭代，纹理生成 2K 迭代；视图细化 8 视角各 45° 间隔。
- **硬件**：NVIDIA A100 40GB。
