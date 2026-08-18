---
title: "CAP4D-Creating-Animatable-4D-Portrait-Avatars-with-Morphable"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Taubner_CAP4D_Creating_Animatable_4D_Portrait_Avatars_with_Morphable_Multi-View_Diffusion_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:57:38"
field: "3D/4D Generative Modeling"
keywords: ["4D Avatar Reconstruction", "Multi-View Diffusion Model", "3D Gaussian Splatting", "Portrait Animation", "Single-image Reconstruction"]
innovations: ["提出可变形多视角扩散模型(MMDM)结合随机I/O条件机制，支持任意数量参考图重建4D头像", "设计基于3D高斯溅射与U-Net变形预测的4D实时渲染流水线，实现高保真表情细节"]
benchmarks: ["NeRsemble", "FFHQ", "VFHQ"]
---

# 论文速读：CAP4D-Creating-Animatable-4D-Portrait-Avatars-with-Morphable

## 一句话总结
CAP4D 提出了一种可变形多视角扩散模型（MMDM），能够根据 1 到 100 张不等数量的参考图像，重建出具有高保真度、身份一致性的 4D（动态 3D）肖像数字人，并支持实时动画与渲染，在单图、少图和多图重建任务上均取得了 SOTA 效果。

## 研究问题与动机
- **参考图像数量受限时的重建难题**：现有的多视图立体或神经渲染方法需要数百张参考图像才能重建高质量 avatar；而仅凭单张或少数图片，传统方法难以恢复 occluded 区域（如后脑勺头发）和细微表情。
- **生成式方法视觉质量与实时性不足**：基于扩散模型的方法能从单图生成逼真头像，但推理成本高、无法实时渲染，且在 4D 动态重建与多视角一致性上仍落后于多视图方法。
- **缺乏统一的输入适配能力**：现有 4D 重建工作往往针对单一输入场景（如纯多图或多张 Monocular video），无法在单个框架内无缝支持从 1 到 100 张参考图像的输入缩放。

## 核心贡献（创新点）
1. **提出可变形多视角扩散模型（MMDM）与随机 I/O 条件机制**：通过结合 3DMM 形态参数对姿态、表情和视角进行控制，并使用随机抽样输入输出条件，实现从任意数量参考图像生成数百张一致的新视角图像。
2. **开发从生成图像蒸馏到 4D 实时 Avatar 的流水线**：利用 3D 高斯溅射（3D Gaussian splatting）表示和表达式依赖的外观 U-Net，将 MMDM 生成的高质量图像序列优化为可实时动画与渲染的动态 3D 头像。
3. **在单图、少图、多图重建任务上均取得 SOTA**：在 NeRsemble 等数据集上的自再现（self-reenactment）与跨身份再现（cross-reenactment）实验中，指标（PSNR、LPIPS、CSIM、JOD）均优于现有基线，尤其在身份保留与时间一致性上优势显著。

## 方法详解
**第一阶段：可变形多视角扩散模型（MMDM）**
- **基础架构**：基于 Stable Diffusion 2.1 改造，将 2D attention 层替换为 3D attention（空间+跨图像维度），以适应多视角生成。
- **条件输入**：除了参考图像 Latent，还拼接四张 conditioning map：
  - **3D pose map**（$\mathbf{P}$）：由 FLAME 3DMM 重建的头部几何，经位置编码后渲染得到，提供全局姿态信息。
  - **Expression deformation map**（$\mathbf{E}$）：表示相对于中性表情的顶点位移，用于引导细微表情变化。
  - **View direction map**（$\mathbf{V}$）：编码相机射线方向，确保视角一致性。
  - **Mask**（$\mathbf{B}$）：区分参考图像与生成图像，以及裁剪区域的遮挡。
- **随机 I/O 条件采样（Stochastic I/O Conditioning）**：MMDM 单次前向最多支持 4 张参考图。为突破此限制，算法在每个 diffusion timesteps 内，随机无放回地选取子集参考图像和生成图像进行批处理。这种“洗牌”机制使得模型在多个 timesteps 中逐渐将所有参考信息融入，从而支持任意数量参考图并提升生成一致性。

**第二阶段：鲁棒的 4D 数字人重建**
- **表示模型**：基于 GaussianAvatars，使用绑定在 FLAME 网格三角面上的 3D 高斯原语。改进之处包括：
  - 将网格重划分至 $128 \times 128$ 分辨率以实现像素级对齐。
  - 引入一个 **U-Net** 来预测 UV 空间中的表达式依赖变形图，从而更精细地模拟皱纹等高频细节。
  - 使用独立的上下颌网格以改善嘴部区域的几何细节。
- **优化目标**：联合使用生成的图像、参考图像、3DMM 参数及相机位姿；施加 Laplacian 正则化于变形图，以及对高斯原语的相对形变与旋转施加 $L_2$ 正则化；配合线性增加的 LPIPS 损失以提升感知质量。

## 实验与结果
- **数据集与评估**：主要在 **NeRsemble** [50] 数据集上进行 self-reenactment（使用 1/10/100 张参考图，预留 4 个视角测试），以及在 FFHQ + VFHQ 上进行 cross-reenactment 评估。
- **对比基线**：单视角方法 GAGAvatar, Portrait4D-v2, Real3D-Portrait, Voodoo3D；多视角方法 DiffusionRig, FlashAvatar, GaussianAvatars。
- **单图 Self-reenactment**：CAP4D 取得 PSNR 21.69、LPIPS 0.311、CSIM 0.633、JOD 5.672，显著优于 Real3D-Portrait（CSIM 0.457）等方法。
- **10 图 Self-reenactment**：CAP4D 在各项指标上均领先，CSIM 达 0.779，LPIPS 0.265，优于 FlashAvatar（CSIM 0.489）与 GaussianAvatars（CSIM 0.478）。
- **100 图 Self-reenactment**：CAP4D 在 LPIPS（0.257）和 CSIM（0.792）上大幅领先，PSNR 23.30，证明方法能充分利用多视图信息。
- **Cross-reenactment 用户研究**：24 名参与者评分中，CAP4D 在视觉质量、表情迁移、3D 结构、时间一致性及整体偏好上均获最高比例（视觉质量 97%、整体 96% 偏好），优于 CSIM 得分略高的 Real3D。
- **消融实验**：去除表情图（w/o expr）导致 PSNR 和 CSIM 明显下降；去除随机采样（w/o stochastic）使各指标回落；移除 U-Net 或 LPIPS 损失亦会损害重建质量，验证了各模块必要性。

## 相关工作脉络
1. **Monocular/Multi-view Avatar Reconstruction**：GaussianAvatars、FlashAvatar 等多采用多视角视频或相机阵列进行优化，CAP4D 借鉴其 3D 高斯表示，但通过扩散模型生成补充视图来克服参考图不足的局限。
2. **Single Image Avatar Reconstruction**：Voodoo3D、Real3D-Portrait 等方法直接回归 3D 参数或特征网格，易在极端姿态下失效；CAP4D 利用扩散先验生成多视图，间接规避了直接回归的不稳定性。
3. **Multi-View Diffusion Models**：CAT3D 等用于静态 3D 重建；本文将其延伸至动态 4D 头像，并结合 3DMM 控制表达，实现了视频级的时间一致性。
4. **3D Gaussian Splatting for Head Avatars**：本文继承 GaussianAvatars 的网格绑定高斯框架，但通过 U-Net 引入高分辨率变形预测，提升了表情细节的真实感。
5. **Diffusion-based Portrait Generation**：此前方法多局限于 2D 图像编辑或单帧渲染；CAP4D 将多视角扩散与 4D 重建结合，实现了端到端的 3D 一致动态头像生成。

## 局限性与未来方向
- **生成速度较慢**：单次重建需数小时（生成 840 张图像约 4 小时），难以满足交互式应用需求。
- **3DMM 建模局限**：FLAME 模型无法表达舌头运动及部分头发动态，可能导致这些区域出现伪影。
- **360 度全景重建受限**：由于缺乏大规模 360 度多身份数据集，头部背面细节生成质量仍不及正面。
- **光照建模不足**：当前外观模型未显式控制光照变化，未来可引入可调节光照模型以提升鲁棒性。

## 研究启发与可借鉴点
1. **随机 I/O 条件机制**：该思想可有效解决扩散模型输入数量有限制的瓶颈，可迁移至其他需要大量参考条件但模型容量受限的 3D/视频生成任务中。
2. **3DMM 与扩散模型的深度结合**：通过将 3D 几何先验（pose、expression maps）作为显式条件输入扩散模型，既保证了生成的结构一致性，又保留了生成模型的细节创造力，为“几何引导生成”提供了范本。
3. **4D 表示中的高频细节增强**：在 Gaussian Splatting 优化过程中引入 U-Net 预测 UV 变形图，有效解决了传统高斯点云在表情变形时细节丢失的问题，该方法可推广至其他基于点的动态场景重建。
4. **多视角蒸馏到实时渲染**：将高质量但计算昂贵的扩散生成图像作为监督信号，去训练轻量级的实时 3D 高斯表示，平衡了视觉质量与渲染效率，该范式适用于多种 3D 生成应用。

## 关键术语表
**MMDM (Morphable Multi-View Diffusion Model)**：一种结合 3D 形态模型条件与多视角扩散生成的模型，能够从少量参考图像生成多个一致的新视角图像。
**3DMM (3D Morphable Model)**：一种统计面部形状与表情模型（如 FLAME），用于参数化表示人脸的几何与表情变化。
**3D Gaussian Splatting**：一种基于 3D 高斯原语的场景表示与渲染技术，兼具体积表示的细节质量与点绘制的实时渲染效率。
**Stochastic I/O Conditioning**：在扩散模型推理时，随机采样输入参考图像和输出生成图像的子集进行条件注入，以突破单次前向的输入数量限制。
**Self-reenactment**：使用目标人物自身的视频序列驱动重建出的头像，评估头像在时间序列上保持身份一致性和表情准确性的能力。
**Cross-reenactment**：使用另一人的视频序列驱动头像，评估头像在不同驱动源下的泛化表现和 3D 结构保持能力。

## 可复现要素
- **数据集**：训练使用 VFHQ、MEAD、Ava-256、Nersemble；评估使用 NeRsemble 和 FFHQ/VFHQ。**数据集本身公开，但论文未明确声明代码开源**。
- **代码/权重**：**论文未提及代码及预训练权重是否开源**。
- **关键超参**：训练使用 AdamW，学习率 $10^{-4}$，batch size 64，共 100k 次迭代（8×H100 GPU，耗时 2 周）；生成 840 张 512×512 图像，250 DDIM 步长（4×RTX6000，耗时 4 小时）；4D 重建优化 100k 次（1×RTX6000，耗时 4 小时）。
