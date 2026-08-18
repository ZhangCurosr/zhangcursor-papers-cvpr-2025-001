---
title: "UA-Pose-Uncertainty-Aware-6D-Object-Pose-Estimation-and-Onli"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Li_UA-Pose_Uncertainty-Aware_6D_Object_Pose_Estimation_and_Online_Object_Completion_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 09:53:06"
field: "6D物体位姿估计"
keywords: ["6D位姿估计", "不确定性感知", "在线对象补全", "神经SDF", "单图到3D生成"]
innovations: ["不确定性感知的混合对象表示（纹理+几何+不确定性）", "基于seen IoU的位姿置信度评估与过滤机制", "不确定性感知采样驱动的在线对象补全策略"]
benchmarks: ["YCB-Video", "YCBInEOAT", "HO3D"]
---

# 论文速读：UA-Pose-Uncertainty-Aware-6D-Object-Pose-Estimation-and-Onli

## 一句话总结
本文提出了UA-Pose，一种不确定性感知的6D物体位姿估计方法，通过融合纹理、几何与不确定性信息的混合对象表示，在仅有部分参考图像或单张RGB图像的场景下实现鲁棒的位姿估计，并支持测试期间的在线对象补全。

## 研究问题与动机
1. **部分参考场景的位姿估计挑战**：现有model-free方法（如FoundationPose）依赖完整的3D CAD模型或充分视角的参考RGBD图像，但在实际应用中往往难以获取高质量完整模型或充足参考视图。
2. **真实场景的视觉遮挡与视角限制**：动态环境中常存在物体遮挡、视角有限等问题，导致参考图像只能捕获物体外观和几何的部分区域。
3. **模型生成质量不足**：直接利用单图到3D生成的模型进行位姿估计效果不佳，因为生成模型的纹理在未见区域不够准确，仅靠几何无法保证精确位姿。
4. **不确定性量化缺失**：现有方法缺乏对部分观测模型中"已知"与"未知"区域的明确区分，无法评估位姿估计的置信度。

## 核心贡献（创新点）
1. **不确定性感知的混合对象表示**：首次将不确定性图（标记已知/未知区域）与神经SDF渲染的纹理和几何融合，形成包含三类信息的对象模型。
2. **不确定性驱动的位姿置信度评估**：提出seen IoU和uncertainty rate两个指标，用于量化位姿估计的可靠性并过滤不可靠估计。
3. **在线对象补全机制**：在测试过程中根据不确定性触发补全流程，通过新捕获的RGBD图像迭代优化不完整对象模型。
4. **不确定性感知采样策略**：设计基于最大未观测区域覆盖的采样算法，从内存池中选择最具信息量的图像用于补全训练。
5. **单图到3D生成的增强利用**：将单图生成模型作为初始模型辅助早期位姿估计，并将其渲染的RGBD作为数据增强辅助补全训练。

## 方法详解
### 整体框架
UA-Pose接受两种输入：（1）少量带已知位姿的参考RGBD图像；（2）单张未定姿RGB图像。方法包含三个核心模块：混合对象表示构建、不确定性感知位姿估计、在线对象补全。

### 混合对象表示（Hybrid Object Representation）
- **神经SDF训练**：使用双网络架构，几何网络$\varOmega: \mathbb{R}^3 \mapsto s$学习有符号距离，外观网络$\varPhi$结合中间特征$f_{\varOmega(x)}$和视角方向$d$预测颜色$c$。
- **体渲染损失**：$\mathcal{L} = w_c\mathcal{L}_c + w_e\mathcal{L}_e + w_s\mathcal{L}_s + w_{eik}\mathcal{L}_{eik}$，包含颜色损失、空间损失、近表面损失和Eikonal正则化。
- **Marching Cubes提取网格**：从隐式SDF提取显式网格$\mathcal{E}=(V,C,F)$，便于显式建模不确定性。

### 不确定性建模（Uncertainty Modeling）
- **可见性检测**：通过mesh rasterization将三维顶点投影到参考图像平面，判断每个顶点是否在参考视角可见。
- **不确定性标记**：可见顶点标记为$certain$ ($u(v_i)=0$)，不可见顶点标记为$uncertain$ ($u(v_i)=1$)。
- **置信度评估**：
  - uncertainty rate = 渲染掩码中不确定像素比例
  - seen IoU = $IoU(\neg U_\xi^{rend} \odot m_\xi^{rend}, m^{test})$，衡量已知区域与测试图像的重叠度

### 不确定性感知位姿估计
- **位姿初始化**：对首帧生成$N_v \cdot N_{in}$个假设位姿（icosphere采样视角+面内旋转）。
- **位姿精化**：复用FoundationPose的pose refinement模块，迭代优化5次（后续帧2次）。
- **位姿筛选**：丢弃uncertainty rate超过$T_u$或seen IoU低于$T_s$的假设。
- **位姿跟踪**：后续帧以前一帧估计为唯一假设，利用时序平滑性精化。

### 在线对象补全（Online Object Completion）
- **内存池管理**：按旋转测地距离阈值$T_{geo}$添加视角，按seen IoU阈值$T_{conf}$过滤低置信度估计。
- **触发条件**：当当前帧seen IoU < $T_{complete}$时触发补全。
- **不确定性感知采样**：优先选择能覆盖最多未观测区域的视角（贪心策略），限制为K个最informative帧。

### 单图到3D生成利用
- 使用InstantMesh [43]从单张RGB生成初始模型$\hat{M}$。
- 生成模型渲染的RGBD作为数据增强，补充补全训练中的未见区域几何。

## 实验与结果
### 数据集
- **YCB-Video**：日常物体的RGBD视频序列
- **YCBInEOAT**：双机械臂操作YCB物体的交互场景
- **HO3D**：人手与物体交互的复杂遮挡场景

### 评估基线
- LoFTR [30]、FS6D-DPM [10]、FoundationPose [41]
- 位姿跟踪方法：BundleTrack [36]、BundleSDF [40]

### 主要结果（YCB-Video）
| 方法 | 输入 | ADD | ADD-S | CD |
|------|------|-----|-------|-----|
| FoundationPose | 2 references | 87.4 | 94.3 | 0.57 |
| **Ours** | **2 references** | **92.8** | **96.5** | **0.53** |
| FoundationPose | single RGB | 88.9 | 93.6 | 0.67 |
| **Ours** | **single RGB** | **93.2** | **96.9** | **0.62** |

### 主要结果（YCBInEOAT）
| 方法 | 输入 | ADD | ADD-S | CD |
|------|------|-----|-------|-----|
| FoundationPose | 2 references | 68.52 | 84.80 | 0.60 |
| **Ours** | **2 references** | **89.99** | **94.35** | **0.57** |

### 主要结果（HO3D）
| 方法 | 输入 | ADD | ADD-S |
|------|------|-----|-------|
| FoundationPose | 2 references | 45.67 | 61.57 |
| **Ours** | **2 references** | **91.51** | **95.56** |

### 最强提升
- YCBInEOAT 数据集上，相比FoundationPose（2 references输入），ADD提升**21.47个百分点**（68.52→89.99），ADD-S提升**9.55个百分点**（84.80→94.35）。

## 相关工作脉络
1. **FoundationPose [41]**：统一支持model-based和model-free的6D位姿估计框架，本文作为核心基线对比，但FoundationPose需要充足参考视角，本文处理部分观测场景。
2. **BundleSDF [40]**：基于SDF的位姿跟踪方法，仅支持连续视频帧的相对位姿估计，不支持参考图像输入；本文支持外部参考并评估置信度。
3. **OnePose/OnePose++ [31, 7]**：单图位姿估计方法，使用SfM重建点云+预训练2D-3D匹配网络；本文直接在神经SDF框架中引入不确定性建模。
4. **FS6D [10]**：基于transformer的少样本位姿估计；本文通过在线补全机制逐步增强对象模型。
5. **GigaPose [23]**：探索单图生成模型用于位姿估计，但生成模型几何精度影响位姿；本文利用生成模型仅作为初始辅助，最终由真实RGBD数据精炼模型。
6. **InstantMesh [43]**：单图到3D的快速网格生成方法，本文将其作为生成初始模型的backbone。

## 局限性与未来方向
1. **依赖首帧物体分割掩码**：方法需要第一帧的准确物体mask，分割误差可能传播影响后续估计。
2. **补全触发阈值敏感**：$T_{complete}$、$T_{conf}$等阈值需要仔细调优，不同场景可能需要自适应策略。
3. **计算开销**：在线补全涉及周期性SDF重新训练，可能限制实时应用。
4. **单图生成质量依赖**：当仅使用单RGB图像时，性能受限于单图到3D生成方法的质量。
5. **未来方向**：自适应阈值学习、端对端不确定性建模、更高效的热迁移补全策略。

## 研究启发与可借鉴点
1. **不确定性可视化作为置信度代理**：将空间不确定性图与渲染结果结合评估位姿质量，思路可迁移到其他感知任务（如SLAM、重建）。
2. **热迁移在线补全机制**：测试期间增量更新对象模型的设计，适用于需要适应新对象的长期运行系统。
3. **生成模型作为初始化的双重利用**：既用作位姿估计辅助，又生成合成数据增强训练，这种"生成辅助+真实精炼"范式可推广。
4. **seen IoU指标的简洁有效性**：通过简单几何重叠度衡量估计可靠性，避免复杂的不确定性网络训练。
5. **内存池的多样性+置信度双约束**：几何距离与任务相关置信度的结合筛选策略，值得在其他增量学习场景借鉴。

## 关键术语表
**6D Object Pose Estimation**：估计物体相对于相机的6自由度刚体变换（3D旋转+3D平移）。
**Model-free Pose Estimation**：不依赖预先获取的CAD模型，通过参考图像重建对象几何进行位姿估计的方法。
**Neural Signed Distance Field (SDF)**：用神经网络隐式表示物体表面，表面定义为距离函数值为零的等值面。
**Hybrid Object Representation**：融合纹理、几何和不确定性信息的对象显式表示，用于区分已知和未知区域。
**Seen IoU**：衡量渲染掩码中"已知"区域与测试图像物体掩码的重叠度，作为位姿置信度指标。
**Online Object Completion**：在测试阶段利用新捕获图像迭代优化不完整对象模型的过程。
**Uncertainty-Aware Sampling**：优先选择能覆盖最多未观测区域的视角进行对象补全的训练策略。
**Image-to-3D Generation**：从单张RGB图像生成三维模型的技术，本文使用InstantMesh作为backbone。

## 可复现要素
- **数据集**：YCB-Video、YCBInEOAT、HO3D均公开可用
- **代码**：项目页面 https://minfenli.github.io/UA-Pose/（论文声明有开源）
- **权重**：使用FoundationPose预训练权重
- **关键超参**：姿态采样数$N_v \cdot N_{in}$、 refinement迭代次数（首帧5次/后续帧2次）、内存池大小K、距离阈值$T_{geo}$、置信度阈值$T_{conf}$、补全触发阈值$T_{complete}$、uncertainty rate阈值$T_u$、seen IoU阈值$T_s$（论文未明确给出具体数值，需查看代码或附录）
