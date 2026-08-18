---
title: "Feat2GS-Probing-Visual-Foundation-Models-with-Gaussian-Splat"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Chen_Feat2GS_Probing_Visual_Foundation_Models_with_Gaussian_Splatting_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 07:05:26"
---

# 论文速读：Feat2GS-Probing-Visual-Foundation-Models-with-Gaussian-Splat

## 一句话总结
Feat2GS 提出了一种统一框架，将视觉基础模型（VFM）的 2D 特征经轻量级读出门映射为 3D Gaussian Splatting 参数，在无需 3D 真值的情况下，通过新视图合成（NVS）任务公平、稠密地探测 VFM 的几何与纹理 3D 感知能力，并基于此设计了超越 InstantSplat 的稀疏日常图像 NVS 新基线。

## 研究问题与动机
- VFM 在海量 2D 图像上预训练，但其是否真正理解 3D 世界（尤其是几何一致性与纹理还原性）缺乏统一、公平的基准评测。
- 现有 3D 探测工作主要依赖单视图 2.5D 估计（深度/法线）或双视图稀疏 2D 匹配，忽视了纹理感知，且高度依赖昂贵的 3D 标注数据，限制了评测数据的规模与场景多样性。
- NVS 仅需多视图图像即可评估，无需 3D 真值，且能让每个像素参与评测；结合免标定立体重建技术（如 DUSt3R），可处理稀疏、未校准的日常抓拍图像，实现“稠密且多样化”的 3D 能力探测。
- 3DGS 参数天然可解耦为几何（位置 $\mathbf{x}$、不透明度 $\alpha$、协方差 $\Sigma$）与纹理（球谐系数 $\mathbf{c}$），为分离评估几何与纹理感知提供了理想的参数载体。

## 核心贡献（创新点）
1. **统一的 VFM 3D 感知探测框架（Feat2GS）**：提出冻结 VFM 特征后经浅层 MLP 直接回归 3DGS 参数的 pipeline，利用 NVS 光度损失驱动优化，以 2D 指标代理 3D 能力评估，无需 3D 标签即可实现大规模跨数据集评测。
2. **几何-纹理解耦探测范式（GTA）**：设计 Geometry/Texture/All 三种探测模式，通过控制 3DGS 参数是“特征读出”还是“自由优化”，独立量化 VFM 在多视图几何一致性与纹理还原性上的差异。
3. **系统性的 VFM 预训练策略剖析**：在 7 个数据集、10 种主流 VFM 上完成全面基准测试，揭示了掩码重建（MAE 系）、点图回归（DUSt3R/MASt3R）与对比/蒸馏学习对几何与纹理感知的不同影响规律。
4. **基于特征拼接的 NVS 强基线**：基于探测发现提出三种变体（单选特征、全部特征拼接、微调 DUSt3R 特征），在稀疏日常图像 NVS 任务上全面超越当前 SOTA InstantSplat，且避免了自由优化导致的过拟合断裂伪影。

## 方法详解
- **特征读出门（Feature Readout）**：冻结的 VFM 特征图经 PCA 统一通道数、双线性插值统一分辨率后，逐像素输入 2-layer MLP（每层 256 单元，ReLU 激活）$g_\Theta$，直接回归每个像素对应的 3D Gaussian 参数：位置 $\mathbf{x} \in \mathbb{R}^3$、不透明度 $\alpha$、协方差 $\Sigma \in \mathbb{R}^{3\times3}$ 及 3 阶球谐系数 $\mathbf{c}_i \in \mathbb{R}^{48}$。
- **联合优化与光度损失**：使用免标定立体重建器 DUSt3R 初始化相机位姿 $\mathbf{T}$，随后联合优化读出门参数 $\Theta$、自由优化参数 $O^{(mode)}$ 与相机位姿，最小化渲染图像 $\mathcal{R}_v$ 与输入图像 $\mathcal{I}_v$ 的光度误差：$\min_{\Theta, O, T} \sum_{v \in N} \| \mathcal{R}_v(g_\Theta(\mathbf{f}), O, T) - \mathcal{I}_v \|$。
- **GTA 三种探测模式**：
  - **Geometry**：几何参数 $\{\mathbf{x}_i, \alpha_i, \Sigma_i\}$ 由特征读出，纹理参数 $\{\mathbf{c}_i\}$ 自由优化。
  - **Texture**：纹理参数 $\{\mathbf{c}_i\}$ 由特征读出，几何参数 $\{\mathbf{x}_i, \alpha_i, \Sigma_i\}$ 自由优化。
  - **All**：所有 3DGS 参数均由特征读出，无自由优化参数。
- **Warm Start 策略**：针对稀疏日常图像易陷入局部最优的问题，先用 DUSt3R 生成的点云 $G_{init}$ 对读出门进行预热：$\min_\Theta \| g_\Theta(\mathbf{f}) - G_{init} \|$，确保优化起点贴近真实几何结构。
- **测试时位姿优化**：在测试集上通过光度损失进一步微调测试视图位姿，确保 NVS 评估的公平性。

## 实验与结果
- **数据集与基线**：7 个多视图数据集（LLFF, DTU, DL3DV, Casual, MipNeRF360, MVImgNet, T&T），覆盖室内/室外、简单/高复杂度、不同视场范围。评测 10 种 VFM（DUSt3R, MASt3R, MiDaS, DINOv2, DINO, SAM, CLIP, RADIO, MAE, SD）及 IUVRGB 基线。
- **2D-NVS 与 3D 指标强相关**：在 DTU 上验证，NVS 的 PSNR/SSIM/LPIPS 与 3D 重建的 Accuracy/Completeness/Distances 高度相关（Tab. 4，相关矩阵接近对角线），证明用 2D 指标代理 3D 评估的可靠性。
- **几何感知 Top 模型**：
