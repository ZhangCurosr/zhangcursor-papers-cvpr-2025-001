---
title: "DeSiRe-GS-4D-Street-Gaussians-for-Static-Dynamic-Decompositi"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Peng_DeSiRe-GS_4D_Street_Gaussians_for_Static-Dynamic_Decomposition_and_Surface_Reconstruction_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 06:01:54"
field: "自动驾驶场景重建"
keywords: ["3D Gaussian Splatting", "静态-动态分解", "自监督", "城市驾驶场景", "表面重建", "时序一致性"]
innovations: ["基于外观差异的自监督运动掩码提取，无需3D标注", "2D运动先验可微蒸馏至3D Gaussian空间实现静态-动态分解", "法向与尺度联合推导及时序跨视角一致性正则化提升表面重建质量"]
benchmarks: ["Waymo Open Dataset", "KITTI Dataset"]
---

# 论文速读：DeSiRe-GS: 4D Street Gaussians for Static-Dynamic Decomposition and Surface Reconstruction for Urban Driving Scenes

## 一句话总结
本文提出 DeSiRe-GS，一种自监督的 4D 街景高斯溅射表示方法，无需 3D 边界框等额外标注，即可实现城市驾驶场景中静态-动态分解与高保真表面重建，在 Waymo 和 KITTI 数据集上超越现有自监督方法并媲美依赖显式标注的方法。

## 研究问题与动机
- **动态场景重建难题**：原始 3D Gaussian Splatting (3DGS) 时间无关的参数化无法处理自动驾驶场景中的动态物体，导致运动区域出现模糊/鬼影。
- **依赖 3D 标注的局限**：现有方法如 Street Gaussians [38]、DrivingGaussian [47]、OmniRe [6] 依赖显式 3D 边界框进行静态-动态分离，标注成本高且泛化性差。
- **稀疏观测导致的过拟合**：自动驾驶图像稀疏，3DGS 易过拟合有限观测，产生 oversized 高斯椭球，损害几何重建质量。
- **表面重建精度不足**：现有自监督方法（如 PVG [5]、S3Gaussian [15]）静态-动态分解不彻底，且无法生成贴合真实表面的物理合理高斯椭球。

## 核心贡献（创新点）
1. **基于外观差异的自监督运动掩码提取**：利用 3DGS 天然无法重构动态区域的观察，通过冻结预训练特征提取器计算渲染图与 GT 图的像素级差异，用 MLP 解码器预测运动掩码，无需 3D 标注实现动态区域检测。
2. **2D 运动先验可微蒸馏至全局 Gaussian 空间**：将第一阶段提取的 2D 运动掩码结合 Periodic Vibration Gaussian (PVG) 动态表示，通过速度正则化损失将局部帧的运动信息传播到全局 3D 高斯空间，实现可微的静态-动态分解。
3. **几何正则化联合优化表面法向与尺度**：受 2DGS 启发，引入最小尺度正则化将 3D 椭球压平成 2D 盘状，并直接从尺度向量推导法向量（而非附加独立法向量），使梯度可回传到尺度参数，提升表面重建质量。
4. **时序跨视角几何一致性约束**：针对稀疏视图导致的过拟合，提出时序空间一致性损失，利用静态区域在不同时间和视角下的深度一致性，显著增强几何感知能力。
5. **巨型高斯惩罚机制**：引入对最大尺度方向的惩罚项，抑制 oversized 高斯椭球的产生，改善表面重建结构合理性。

## 方法详解
**整体框架**：两阶段自监督优化流程（图 2）。

**Stage I — 动态掩码提取**：
- 使用冻结的预训练模型（FiT3D/DINOv2）从渲染图像 $\hat{I}$ 和 GT 图像 $I$ 中提取特征 $\hat{F}$ 和 $F$。
- 计算像素级差异：$D = (1 - \cos(\hat{F}, F))/2$，其中 $D \approx 0$ 表示静态区域，$D \approx 1$ 表示动态区域。
- 用 MLP 解码器预测动态性得分 $\delta$，训练损失：$\mathcal{L}_{dyn} = \delta \odot D$。
- 生成二值运动掩码：$M = \mathbb{I}(\delta > \varepsilon)$。
- 联合优化：$\mathcal{L}_{stage1} = M \odot \mathcal{L}_I + \mathcal{L}_{dyn}$，其中 $\mathcal{L}_I$ 为掩码加权图像渲染损失。

**Stage II — 静态-动态分解与表面重建**：
- 采用 PVG 作为动态表示基础，渲染深度图 $\mathcal{D}$、法向图 $\mathcal{N}$ 和速度图 $\mathcal{V}$。
- **速度正则化**：$\mathcal{L}_v = \mathcal{V} \odot M$，惩罚静态区域非零速度，消除噪声异常值。
- **最小尺度正则化（扁平化）**：$\mathcal{L}_s = \|\min(s_1, s_2, s_3)\|$，促使 3D 椭球压平成 2D 盘状贴合表面。
- **法向推导与监督**：$n = R \cdot \arg\min(s_1, s_2, s_3)$，直接由尺度向量推导法向，损失 $\mathcal{L}_n = \|\mathcal{N} - \hat{\mathcal{N}}\|_2$。
- **巨型高斯惩罚**：$\mathcal{L}_g = s_g \cdot \mathbb{I}(s_g > \epsilon)$，其中 $s_g = \max(s_1, s_2, s_3)$。
- **时序跨视角一致性**：将参考帧像素投影到相邻帧，计算重投影深度误差 $\mathcal{L}_{uv} = \|(u_r, v_r) - (u_{nr}, v_{nr})\|_2$。
- **深度监督**：使用 LiDAR 投影的稀疏深度图 $D_{gt}$，$\mathcal{L}_D = \|\mathcal{D} - D_{gt}\|_1$。
- 总损失：$\mathcal{L}_{stage2} = \mathcal{L}_I + \mathcal{L}_D + \mathcal{L}_n + \mathcal{L}_v + \mathcal{L}_s + \mathcal{L}_g + \mathcal{L}_{uv}$。

## 实验与结果
**数据集**：Waymo Open Dataset（使用 PVG 子集和 OmniRe 子集）、KITTI Dataset，使用前向三摄（Waymo）或左右摄（KITTI）。

**评估指标**：PSNR、SSIM、LPIPS（图像质量）；DPSNR、DSSIM（动态区域）；Depth L1（几何重建）。

**主要结果（Waymo 数据集，表 1）**：
- 图像重建 PSNR：**33.61**（SOTA，超越 PVG 的 32.46，+1.15 dB）
- 新视角合成 PSNR：**29.75**（超越 PVG 的 28.11）
- SSIM：**0.919** / **0.878**
- LPIPS：**0.204** / **0.213**
- 渲染速度：~36 FPS

**与依赖 3D 边界框方法的对比（表 2，Waymo OmniRe 子集）**：
- 自监督方法中 SOTA，PSNR(reconst) = **33.82**，超越 PVG(32.37) 和 EmerNeRF(31.93)
- 接近依赖边界框的方法：OmniRe(34.25)、StreetGS(29.08)、HUGS(28.26)
- 在无需标注的情况下达到与显式标注方法相近的性能

**消融实验（表 3）**：
- 移除 Stage I 运动掩码：PSNR 下降约 1.05 dB
- 移除 FiT3D（改用 DINOv2）：PSNR 从 35.45 降至 34.96
- 移除法向监督：PSNR 从 35.76 降至 35.24
- 移除最小/最大尺度正则化：对渲染指标影响较小，但对 3D 结构质量有显著改善
- 移除跨视角一致性：Depth L1 从 0.0713 上升至 0.1154，几何精度大幅下降

## 相关工作脉络
1. **3DGS 动态扩展**：4D-GS [35] 使用 Hexplane 编码器建模动态，但在无界街景中表现不佳；DeSiRe-GS 选择对原始 3DGS 做最小改动引入时间依赖，更适合大规模驾驶场景。
2. **依赖 3D 标注的方法**：Street Gaussians [38]、DrivingGaussian [47]、OmniRe [6]、HUGS [46] 均依赖显式 3D 边界框进行静态-动态分离，DeSiRe-GS 在无需标注的情况下达到相近甚至更优性能。
3. **自监督 Gaussian 方法**：PVG [5] 为每个 Gaussian 引入振动位置和寿命，但分解不彻底且易产生噪声速度；S3Gaussian [15] 使用 Hexplane 时空编码器，计算开销大；DeSiRe-GS 通过 2D 运动先验蒸馏实现更精准分解。
4. **神经表面重建**：2DGS [14]、SuGaR [12]、PGSR [2] 等引入几何正则化改善表面重建；DeSiRe-GS 借鉴 2DGS 思想并将法向与尺度联合优化，适配自动驾驶稀疏场景。
5. **NeRF 类自监督方法**：EmerNeRF [39] 通过场景流估计捕捉物体对应关系，SUDS [31] 使用多分支哈希表；DeSiRe-GS 作为 3DGS 方法在渲染速度（~36 FPS vs <1 FPS）和重建质量上均占优。

## 局限性与未来方向
- **稀疏 LiDAR 深度监督的局限性**：深度损失依赖于稀疏点云投影，深层结构细节仍可能不足。
- **预训练特征提取器的依赖**：运动掩码提取依赖 FiT3D/DINOv2 等冻结网络，在分布外场景可能退化。
- **两阶段训练的复杂性**：需依次完成 Stage I（30K 迭代）和 Stage II（50K 迭代），训练流程较长。
- **仅适用于城市驾驶场景**：方法针对无界户外场景设计，在室内或密集纹理场景的泛化性未验证。
- **未来方向**：可探索端到端单阶段训练、融合稠密语义分割先验、拓展至多传感器融合（如融合 RADAR 速度信息）。

## 研究启发与可借鉴点
1. **"3DGS 无法重建动态区域"的简单观察**：利用渲染图与 GT 图的外观差异直接提取运动先验，避免复杂的变形网络学习，思路简洁高效，可迁移至其他动态场景重建任务。
2. **法向从尺度向量直接推导**：将法向量与尺度参数耦合优化，替代独立附加法向量，使梯度能回传到尺度，这一设计可推广到其他基于 Gaussians 的表面重建工作。
3. **时序跨视角一致性损失**：利用静态区域的深度跨时不变性构建几何一致性约束，有效缓解稀疏视图过拟合，可结合到任意单目/多目 Gaussian 重建框架中。
4. **巨型高斯惩罚机制**：针对过拟合产生的 oversized 椭球施加尺度上限惩罚，提升 3D 结构合理性，适合任意需要几何精度的 Gaussian Splatting 应用。
5. **两阶段自监督分解策略**：先提取 2D 运动掩码再蒸馏到 3D 空间的思路，可为无需标注的动态-静态分离任务提供可复用的范式。

## 关键术语表
**3D Gaussian Splatting (3DGS)**：一种基于显式 3D 高斯椭球集合的场景表示方法，通过可微渲染实现实时高质量图像合成。
**Periodic Vibration Gaussian (PVG)**：为 3DGS 引入时间相关的位置振动和透明度衰减，用于建模动态场景的自监督表示。
**Static-Dynamic Decomposition**：将场景中静态（建筑物、道路）与动态（车辆、行人）部分分离，是自动驾驶场景理解的核心任务。
**Temporal Cross-View Consistency**：利用静态区域在不同时间和视角下深度一致的几何约束，增强多视图几何学习。
**LiDAR Point Cloud**：激光雷达获取的稀疏三维点云，常用于自动驾驶场景的深度监督和几何初始化。
**Spherical Harmonics (SH)**：用于参数化 3DGS 颜色属性的正交函数基，支持 View-dependent 颜色渲染。
**TSDF (Truncated Signed Distance Function)**：截断符号距离函数，用于表征 surfaces 的隐式几何表示。
**FiT3D**：3D-aware 微调的 2D 特征提取模型，本文用于从渲染图和 GT 图中提取判别性特征。

## 可复现要素
- **数据集**：Waymo Open Dataset [28]（公开）、KITTI Dataset [11]（公开）
- **代码开源**：https://github.com/chengweialan/DeSiRe-GS
- **关键超参**：
  - 初始化点数：$1 \times 10^6$（LiDAR 点 $6 \times 10^5$ + 随机采样 $4 \times 10^5$）
  - Stage I 训练迭代：30,000（运动解码器从 6,000 步开始训练）
  - Stage II 训练迭代：50,000（多视角一致性从 20,000 步开始）
  - 运动掩码用于速度正则化：从 30,000 步开始
  - 优化器：Adam ($\beta_1=0.9, \beta_2=0.999$)
  - 硬件：NVIDIA RTX A6000
- **特征提取器**：FiT3D [44]（首选）或 DINOv2
