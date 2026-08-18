---
title: "GAF-Gaussian-Avatar-Reconstruction-from-Monocular-Videos-via"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Tang_GAF_Gaussian_Avatar_Reconstruction_from_Monocular_Videos_via_Multi-view_Diffusion_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 07:07:36"
field: "单目头像重建与3D生成"
keywords: ["3D Gaussian Splatting", "Head Avatar Reconstruction", "Multi-view Diffusion", "Monocular Video", "Normal Map Conditioning", "Score Distillation Sampling", "Parametric Head Model"]
innovations: ["提出 FLAME 法线图条件多视图头部分扩散模型，提供像素对齐视角控制", "使用迭代去噪图像作为伪监督替代单步 SDS，消除过饱和伪影", "引入潜变量上采样模块增强高斯渲染的面部细节与真实感"]
benchmarks: ["NeRSemble", "INSTA Monocular Video"]
---

# 论文速读：GAF: Gaussian Avatar Reconstruction from Monocular Videos via Multi-view Diffusion

## 一句话总结
本文提出 GAF（Gaussian Avatar Fusion），通过多视图头部分扩散先验从单目视频重建可动画的真实感 3D 高斯头像；利用 FLAME 法线图进行视角条件控制，并以迭代去噪图像作为伪监督信号约束高斯渲染，显著提升未见视角与新表情的合成质量。

## 研究问题与动机
- **单目观测严重受限**：智能手机/消费级设备录制的单目视频仅覆盖正面或有限角度，导致侧面等未观测区域在 3D 优化中欠约束，新视角渲染出现严重伪影。
- **现有单目方法缺乏补全先验**：INSTA、FlashAvatar、GA 等方法仅重建可见区域，无法合理外推缺失的面部几何与外观。
- **文本到图像扩散先验不适配**：预训练或个性化 Stable Diffusion 缺少输入图像的细粒度身份与外观信息，难以保持跨视角一致性与身份保真。
- **单步 SDS 导致过饱和**：Score Distillation Sampling 的单步采样引入随机噪声，易造成渲染外观过饱和与 3D 优化不稳定。

## 核心贡献（创新点）
- **提出多视图头部分扩散模型**：以输入图像与 FLAME 法线图 jointly 生成多视角一致的头部分图像，显著优于文本先验或单步 SDS 策略。
- **引入法线图（Normal Map）作为视角条件**：相比相机位姿 embedding 或 Plücker 射线映射，法线图提供像素对齐的归纳偏置，使新视角生成在面部对齐与结构一致性上更优（SSIM 提升 7.73%）。
- **采用迭代去噪图像作为伪 ground truth**：用 DDIM 多步去噪替代单步 SDS，消除过饱和问题，并为 3D 高斯优化提供更稳定的监督信号。
- **引入潜变量上采样模块**：在解码前将 latent 从 32×32 上采样至 64×64，使伪监督图像达到 512×512，显著提升面部细节与真实感。
- **在 NeRSemble 与真实消费级设备数据集上刷新 SOTA**：新视角合成 LPIPS 降至 0.125、SSIM 达 88.91%；新表情合成 LPIPS 降至 0.087、PSNR 达 24.12 dB。

## 方法详解
- **整体流程**：给定单目 RGB 序列 $\mathcal{T}=\{\mathbf{I}_i\}$，使用 VHAP tracker 获取每帧的 FLAME 网格 $\mathcal{M}_i$；将 3D Gaussians 绑定到 FLAME 三角面片上，通过位姿/表情参数驱动形变；优化阶段同时最小化输入视角重建损失 $\mathcal{L}_{\text{rec}}$ 与多视图扩散约束损失 $\mathcal{L}_{\text{view}}$。
- **多视图扩散模型架构**：基于 Stable Diffusion 2.1 微调，以输入图像 $\mathbf{I}_{\text{cond}}$ 的 VAE latent 通过 cross-attention 注入身份细节；目标视图法线图经 VAE 编码后与噪声 latent 在通道维拼接；使用 2D U-Net 加 3D attention 块实现跨视角信息交互。
- **法线图条件设计**：从 FLAME 网格在目标相机位姿下渲染法线图 $\mathbf{N}_{\text{tgt}}$，编码后与多视图噪声 latent 拼接；相比 pose embedding 和 ray map，法线图提供像素级几何引导，减少生成图像的错位与模糊。
- **迭代去噪伪监督生成**：每步随机采样 4 个新视角，渲染高斯图像 $\mathbf{I}^{\text{view}}$ 与法线图；将 $\mathbf{I}^{\text{view}}$ 编码为 latent $\mathbf{z}$ 并加噪至时间步 $t \sim [0.02, 0.98]$；使用 DDIM（$k=20$ 加速）多步去噪至 $\mathbf{z}_0$，解码为 4 张伪 ground truth $\hat{\mathbf{I}}^{\text{view}}$。
- **潜变量上采样**：对去噪后的 $\mathbf{z}_0$（32×32）使用预训练 SD x2 latent upsampler 上采样至 64×64，再经 10 步 DDIM 解码至 512×512，增强面部细粒度细节。
- **损失函数**：
  - 重建损失：$\mathcal{L}_{\text{img}} = 0.8\mathcal{L}_1 + 0.2\mathcal{L}_{\text{SSIM}} + 0.1\mathcal{L}_{\text{LPIPS}}$
  - 视图约束损失：$\mathcal{L}_{\text{img}}(\mathbf{I}^{\text{view}}, \hat{\mathbf{I}}^{\text{view}})$
  - 正则项：位置损失 $\mathcal{L}_{\text{pos}}$（$\lambda_{\text{pos}}=0.01$）与尺度损失 $\mathcal{L}_{\text{scale}}$（$\lambda_{\text{scale}}=1$）
  - 自适应加密：每 300 步检查梯度 > 0.0002 的 splat 进行细分，透明度 < 0.005 的删除
- **训练细节**：扩散模型在 NeRSemble 上 20,000 步、学习率 1e-4、batch size 64、8×A100 训练约 72 小时；Gaussian 优化 6,000 步，Adam 学习率分别为位置 5e-5、尺度 1.7e-2、旋转 1e-3、颜色 2.5e-3、透明度 5e-2。

## 实验与结果
- **数据集**：NeRSemble（16 视角，使用第 8 视角单目输入、其余 15 视角评估）、INSTA 单目视频、智能手机 captured 3 序列（1280×720）。
- **评估设置**：新视角合成（seen pose/expression，15 个 holdout 视角）与新表情合成（unseen pose/expression，5 个近距 holdout 视角）；NeRSemble 80% 帧训练、20% 评估。
- **NeRSemble 定量结果**：
  - 新视角：LPIPS 0.125（↓vs 次优 0.161）、PSNR 20.88、SSIM 88.91（↑vs 次优 87.63）
  - 新表情：LPIPS 0.087（↓vs 次优 0.142）、PSNR 24.12（↑vs 次优 21.63）、SSIM 90.66（↑vs 次优 90.37）
- **Monocular Video 定量结果**：L1 0.0229、LPIPS 0.090、PSNR 23.16、SSIM 89.76，全面超越 INSTA/FlashAvatar/GA/P4D-v2/GAGAvatar。
- **消融验证**：
  - 移除扩散先验：LPIPS 从 0.118 升至 0.207，PSNR 从 21.82 降至 18.47
  - 法线图 vs 位姿 embedding：L1 误差降低 0.075，SSIM 提升 7.73%
  - 迭代去噪 vs 单步 SDS：消除面部过饱和伪影
  - 潜变量上采样：PSNR 提升 0.34 dB，SSIM 提升 2.15%
- **推理速度**：单帧渲染 0.016 秒（802×550），约 62 FPS；重建耗时约 12 小时（单 A6000，32 GB 显存）。

## 相关工作脉络
- **Gaussian Avatars [57]**：将 3D Gaussians 绑定到 FLAME 网格实现可动画头像，但依赖多视角密集观测；GAF 在其基础上引入多视图扩散先验解决单目欠约束问题。
- **INSTA [105] / FlashAvatar [89]**：单目高保真头像重建方法，仅优化可见区域，未见区域出现严重伪影；GAF 通过扩散先验补全缺失区域。
- **P4D-v2 [18] / GAGAvatar [14]**：前馈网络预测表达式驱动的高斯头像，但缺乏多视角几何一致性约束，外推视角易出现身份漂移；GAF 通过迭代去噪伪监督提供跨视角一致性。
- **DreamFusion [55] / ReConfusion [88]**：使用 SDS 或迭代去噪蒸馏 2D 扩散先验到 3D；GAF 将这一思想扩展至人头场景，并引入法线图条件与潜变量上采样。
- **MVDiffusion [78] / ImageDream [85] / CAT3D [22]**：通用物体多视图扩散模型；GAF 针对人头分布微调，并以 FLAME 法线图替代通用相机条件，提升面部对齐精度。
- **HeadGAP [100] / GPHM [92]**：参数化头部分高斯模型；GAF 同样使用 FLAME 绑定，但引入扩散正则而非纯数据驱动先验。

## 局限性与未来方向
- **未显式分离材质与外观**：当前表示不支持重光照应用，未来可结合 Relightable Gaussian Codec Avatars 等方向。
- **优化耗时较长**：单头像重建约 12 小时，难以满足实时需求；未来可探索前馈大重建模型（如 LRM、GS-LRM）实现快速 4D 重建。
- **参数化头部模型表达能力有限**：FLAME 无法精细建模头发几何与动画；未来可扩展至细粒度头发高斯表示（如 GaussianHair）。
- **单目视频覆盖范围依赖**：若输入视频几乎无侧脸旋转，侧面补全仍可能不准确；需更多样化的训练数据或更强先验。

## 研究启发与可借鉴点
- **法线图作为像素对齐条件**：相比相机位姿 embedding，法线图在头部/人脸新视角合成中提供更强的几何归纳偏置，可迁移至其他参数化形体（如身体、手部）的 3D 生成任务。
- **迭代去噪伪监督替代 SDS**：多步 DDIM 去噪生成的确定性伪 ground truth 可有效缓解单步 SDS 的过饱和与梯度噪声问题，适用于任意 3D 高斯/NeRF 与 2D 扩散先验的蒸馏场景。
- **潜变量上采样增强细节**：在 latent 空间进行上采样后再解码，比直接在像素空间超分更高效且保真；可结合 SD x2/x4 upscaler 用于其他头像/角色重建任务。
- **跨视角 3D attention 保持一致性**：在多视图扩散中引入 3D attention 块进行跨视角信息交互，比单独逐视图去噪更能保证身份与外观的跨视角一致。
- **单目重建+扩散正则的范式**：先通过参数化模型（FLAME/VHAP）获取粗几何，再用扩散先验正则 3D 表示优化，这一两阶段策略可推广至身体、物体等单目重建任务。

## 关键术语表
- **3D Gaussian Splatting**：用离散 3D 高斯椭球表示场景并通过可微 rasterizer 高效渲染的技术，支持实时渲染与拓扑变化。
- **Gaussian Avatar**：将 3D Gaussians 绑定到参数化网格（如 FLAME）上，通过网格形变驱动高斯属性实现可动画头像表示。
- **FLAME**：Learning a Facial Model from Images of Expressed Subjects，一种统计 3D 头部分形变模型，提供形状、表情与姿态参数。
- **Multi-view Diffusion Model**：从单视图图像生成多视角一致图像的条件扩散模型，通常通过 3D attention 或自注意力机制保持跨视角一致性。
- **Score Distillation Sampling (SDS)**：利用预训练 2D 扩散模型的梯度对 3D 表示进行优化的损失函数，单步采样易导致过饱和与噪声。
- **Normal Map Conditioning**：将表面法线图作为空间对齐的条件输入扩散模型，提供像素级几何指导以实现更精准的新视角合成。
- **Pseudo Ground Truth**：通过扩散模型多步去噪生成的确定性图像，用作 3D 优化的监督信号以替代 noisy SDS 梯度。
- **Latent Upsampler**：在潜空间对低分辨率特征进行上采样的扩散模型模块，可在解码前增强图像细节与分辨率。

## 可复现要素
- **数据集**：NeRSemble（16 视角头部分视频，评估序列未参与扩散模型训练）；INSTA 单目视频；智能手机录制序列（1280×720）。NeRSemble 公开，其他数据集需查看原论文链接。
- **代码/权重**：项目页面 https://tangjiapeng.github.io/projects/GAF，论文未明确说明代码是否开源。
- **关键超参**：扩散模型学习率 1e-4、batch size 64、20,000 步；Gaussian 优化 6,000 步、位置 lr 5e-5、尺度 lr 1.7e-2、旋转 lr 1e-3、颜色 lr 2.5e-3、透明度 lr 5e-2；DDIM 加速系数 k=20；潜变量上采样 10 步。
- **硬件**：扩散模型训练 8×A100；重建推理单 A6000（32 GB 显存）。
