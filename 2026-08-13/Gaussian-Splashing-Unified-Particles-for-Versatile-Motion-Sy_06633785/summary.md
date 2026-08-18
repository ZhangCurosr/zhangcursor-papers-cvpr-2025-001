---
title: "Gaussian-Splashing-Unified-Particles-for-Versatile-Motion-Sy"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Feng_Gaussian_Splashing_Unified_Particles_for_Versatile_Motion_Synthesis_and_Rendering_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 07:08:11"
field: "物理驱动3D场景重建与渲染"
keywords: ["3D Gaussian Splatting", "Position-Based Dynamics", "Fluid Simulation", "Physically-Based Rendering", "Solid-Fluid Interaction", "Novel View Synthesis"]
innovations: ["解耦PBD仿真粒子与3DGS渲染核的统一粒子框架", "各向异性正则化抑制大旋转形变尖刺伪影", "基于Beer-Lambert厚度的流体PBR渲染与表面张力法向估计"]
benchmarks: ["Waves crashing on cliff", "LEGO bulldozer surfing", "Deformable ficus", "Headset droplets", "Bulldozers stacking"]
---

# 论文速读：Gaussian Splashing: Unified Particles for Versatile Motion Synthesis and Rendering

## 一句话总结
本文提出**Gaussian Splashing (GSP)**，一个将3D Gaussian Splatting (3DGS) 与基于位置的动态（PBD）统一结合的新框架，首次在3DGS场景中实现了固体与流体之间的双向耦合物理交互渲染。

## 研究问题与动机
- 现有3DGS/NeRF方法主要处理静态场景或简单动态，缺乏对真实物理（流体、可变形体、刚体）的统一支持。
- 直接将PBD与3DGS结合会产生**旋转形变导致的尖刺噪声**（spiky noises）和**流体反射/折射渲染质量差**的问题。
- 传统3DGS的颜色函数仅依赖视角方向，无法表达物理真实的 specular 高光与介质透射。
- 物体位移后暴露的隐藏区域会导致3DGS重建出现**黑色斑块和脏纹理**。

## 核心贡献（创新点）
1. **统一的PBD-3DGS粒子框架**：将固体、流体、刚体统一在PBD框架下，利用二者一致的点基表示实现同构渲染与仿真。
2. **各向异性正则化损失**：通过约束高斯核缩放比例（max ratio）防止大旋转形变下的拉伸伪影，与PhysGaussian等耦合方案本质不同——本文解耦仿真粒子与渲染内核。
3. **基于PBF的表面张力与法向估计**：对近表面粒子计算法向用于specular合成；对分散液滴采用深度体积渲染反推法向。
4. **改进的流体PBR渲染**：引入Beer-Lambert定律建模厚度相关的透射颜色，配合体积光吸收与折射畸变，使流体呈现真实镜面反射与半透明效果。
5. **生成式AI Inpainting修复**：对物体位移后遗漏区域使用LaMa填补，避免3DGS的缺失重建问题。

## 方法详解
**整体流程**（Fig.2）：多视图图像 → 前景分离与网格重建（NeuS） → 粒子采样（Poisson Disk） → 3DGS训练（含PBR材料参数） → PBD动画（流体+固体） → 渲染合成（加阴影、泡沫）。

**训练阶段**（§4.1）：
- 可训练参数：位置 $\boldsymbol{x}_p$、不透明度 $\sigma_p$、协方差 $\boldsymbol{A}_p$、漫反射 $\boldsymbol{d}_p$、镜面反射 $\boldsymbol{s}_p$、粗糙度 $\rho_p$、法向 $\boldsymbol{n}_p$。
- 损失函数：$\mathcal{L} = \mathcal{L}_{\text{color}} + \lambda_n \mathcal{L}_{\text{normal}} + \lambda_a \mathcal{L}_{\text{aniso}}$，其中各向异性损失 $\mathcal{L}_{\text{aniso}} = \frac{1}{|\mathcal{P}|}\sum_p \max\left\{\frac{S^1_p}{S^2_p}-a, 0\right\}$（$S^1_p$为最大缩放，$S^2_p$为次大缩放）。
- 超参：$a=1.1, \lambda_n=0.2, \lambda_a=10$。

**固体仿真与插值**（§4.2）：
- 采用**解耦方案**：仿真用独立粒子集（Poisson Disk采样入NeuS重建网格），渲染用训练好的3DGS核。
- 通过GMLS（广义移动最小二乘）将粒子形变梯度插值到Gaussian核上更新协方差与法向：$\boldsymbol{A}_p^t = \boldsymbol{F}_p \boldsymbol{A}_p \boldsymbol{F}_p^\top$，$\boldsymbol{n}_p^t = \frac{\boldsymbol{F}_p^{-\top}\boldsymbol{n}_p}{\|\boldsymbol{F}_p^{-\top}\boldsymbol{n}_p\|}$。

**基于位置的流体（PBF）**（§4.3）：
- 密度约束：$C_i^\rho = \frac{\rho_i}{\rho_0}-1 = \sum_j \frac{m_j}{\rho_0}W(\boldsymbol{p}_i-\boldsymbol{p}_j)-1$。
- 表面检测：通过球形屏幕投影面积判定；表面法向 $n_i = \text{normalize}(-\nabla_{p_i} C_i^\rho)$。
- 面积约束最小化局部表面积；距离约束 $C_{ij}^D = \min\{0, \|\boldsymbol{p}_i-\boldsymbol{p}_j\|-d_0\}$ 促进均匀分布。
- 所有约束GPU并行化，局部三角化在独立线程完成。

**渲染**（§4.4）：
- 固体核在形变位置直接splat，配合Re-Engineered方差阴影映射（nearly-soft shadows）。
- 流体用椭球splatting：法向取自最近表面粒子； specular材料 $s_p=1, \rho_p=0.05$。
- 透射用Beer's Law：$d_p = e^{-k\tau_p} \boldsymbol{c}_p^{\text{bg}}$，其中 $\tau_p$ 为流体厚度（additive blending），折射路径畸变 $\beta \boldsymbol{n}_p$。
- 泡沫/气泡/喷雾用distinct kernel + additive splatting增强 realism。
- 最终合成：$\text{result} = \text{composite}(c^{\text{bg}}, c^{\text{fluid}})$。

**Inpainting**（§4.5）：移除被位移物体对应的所有Gaussian核，用LaMa [63] 填充像素颜色，并赋给对应第一个Hit到的Gaussian核的漫反射（specular置0，opacity置1）。

## 实验与结果
**硬件**：Intel i7-12700F / NVIDIA RTX 3090。

**消融实验**：
- 各向异性正则化：显著消除大形变下的模糊与尖刺伪影（Fig.4）。
- PBR材质：无specular时流体类似烟雾（Fig.5），加入后显著提升真实感。
- 阴影映射：近软阴影增强深度感，否则物体像扁平贴层（Fig.6）。
- Inpainting：消除黑色斑块（Fig.7）。

**演示场景**：
- **海浪拍崖**（Fig.8）：817K流体粒子 + 420K固体，展示两向耦合飞溅。
- **LEGO推土机冲浪**（Fig.10）：334,815固体 + 280,000流体核，双向耦合驱动刚体变形与振动。
- **可变形榕树**（Fig.9）：204K核，连续外部力作用下持续形变。
- **头滴水滴**（Fig.11）：64K流体，表面张力形成水滴滑落桌面形成水坑。
- **倒塌的推土机群**：6.67M核刚体碰撞堆积。

**性能**（Table 1）：最大场景（Bulldozers）6.67M核，Sim Time 4.5s/step，Render Time ~24.6×10⁻²s（固体）+ 2.1×10⁻²s（背景）。典型场景Sim < 8s/step，Render < 2.5×10⁻²s。

## 相关工作脉络
1. **PhysGaussian [72]**：首次将物理仿真嵌入3DGS，但采用耦合方案（同一套kernel同时做仿真和渲染），细薄结构采样稀疏、流体渲染依赖球谐无法产生动态高光。GSP解耦仿真粒子与渲染核，并引入PBR specular。
2. **GaussianShader [26]**：为3DGS核增加法向、漫反射/镜面反射材料参数，支持环境贴图光照。本文直接沿用其渲染管线并扩展至流体。
3. **NeuS [67]**：神经隐式表面重建，用于本文提取前景物体表面网格以进行Poisson Disk采样。
4. **Position-Based Fluids (PBF) [40]**：经典拉格朗日流体方法，本文在其基础上扩展GPU并行与表面张力模型。
5. **LaMa Inpainting [63]**：用于填补物体位移后的隐藏区域，弥补3DGS无法重建未观测区域的缺陷。
6. **SPH系列 [46,5,61]**：经典光滑粒子流体动力学，参数敏感；本文选用PBF因其更稳定且约束投影天然适配3DGS。

## 局限性与未来方向
- **PBD精度限制**：位置基方法牺牲部分物理精度换取稳定性，不适合需要精确连续介质力学的场景。
- **流体渲染非完全物理真实**：椭球splatting无法完整模拟真实光传输（如全折射、色散、焦散）。
- **训练成本较高**：多视图重建+PBR优化+流体模拟联合训练耗时较长（论文未给出具体训练时长）。
- **场景规模受限**：最大演示仅6.67M核，复杂城市级场景尚需进一步优化。
- **未来方向**：集成更复杂的本构模型（如超弹性）、支持实时交互、扩展至更多物态（可压缩气体等）。

## 研究启发与可借鉴点
1. **解耦仿真与渲染**是提升双端质量的通用策略：仿真用均匀粒子保证物理精度，渲染用稀疏/非均匀Gaussian核保证视觉质量，插值用GMLS实现平滑过渡。
2. **各向异性正则化**的思路可迁移到其他基于粒子/核的渲染方法中，防止极端形变导致的光栅化伪影。
3. **厚度依赖的Beer-Lambert透射**为3DGS渲染半透明介质提供了简洁有效的建模方式，无需显式体积光追。
4. **生成式Inpainting填补未观测区域**是处理动态场景"空洞"的有效手段，可与其他修复方法（如深度补全）结合。
5. **表面张力+法向估计**的联合设计可同时服务于流体形态与specular渲染，形成物理-外观闭环。

## 关键术语表
- **Gaussian Splashing (GSP)**：本文提出的统一框架，结合3DGS渲染与PBD物理仿真。
- **Position-Based Dynamics (PBD)**：直接迭代约束投影求解的动力学框架，数值稳定且易于并行。
- **Anisotropy Loss**：约束高斯核最大/次大缩放比例，防止大旋转形变下的尖锐伪影。
- **GMLS (Generalized Moving Least Squares)**：从离散粒子场插值连续形变梯度的鲁棒方法。
- **Position-Based Fluids (PBF)**：基于密度约束和面积约束的拉格朗日流体模拟方法。
- **Surface Tension Constraint**：通过局部表面积最小化模拟流体表面张力效应。
- **Beer-Lambert Law**：描述光在介质中吸收衰减的物理定律，用于流体厚度相关透射建模。
- **Inpainting**：利用生成AI填补图像中缺失/遮挡区域的图像修复技术。

## 可复现要素
- **数据集**：使用自采集多视图图像（论文未公开数据集链接）。
- **代码开源**：论文未提供代码开源声明。
- **权重**：未公开预训练权重。
- **关键超参**：$a=1.1, \lambda_n=0.2, \lambda_a=10$；流体specular $s_p=1, \rho_p=0.05$；PBD距离阈值 $d_0$ 未明确。
- **依赖**：PyTorch、GaussianShader [26]、NeuS [67]、LaMa [63]。
