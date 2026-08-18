---
title: "DynamicScaler-Seamless-and-Scalable-Video-Generation-for-Pan"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_DynamicScaler_Seamless_and_Scalable_Video_Generation_for_Panoramic_Scenes_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 07:31:32"
field: "全景/沉浸式视频生成"
keywords: ["全景视频生成", "扩散模型", "无微调扩展", "360° FoV", "Offset Shifting Denoiser", "Global Motion Guidance"]
innovations: ["提出无需微调的OSD移位去噪机制实现任意分辨率全景视频生成", "设计GMG分层引导策略确保全局运动一致性", "将去噪机制扩展至360°球面投影和时序循环维度"]
benchmarks: ["VBench", "Q-Align", "用户偏好研究"]
---

# 论文速读：DynamicScaler-Seamless-and-Scalable-Video-Generation-for-Pan

## 一句话总结
本文提出 DynamicScaler，一种无需微调（tuning-free）的全景动态场景生成框架，通过 Offset Shifting Denoiser（OSD）和 Global Motion Guidance（GMG）机制，实现任意分辨率、任意宽高比乃至 360° 视场角的无缝、连贯视频生成，同时保持恒定的 GPU 显存消耗。

---

## 研究问题与动机
1. **分辨率与宽高比受限**：现有视频扩散模型受固定分辨率训练限制，难以直接生成高分辨率或超宽/竖屏场景级动态内容。
2. **全景边界不连续**：全景图像左右边界在现实中代表同一子午线，但直接拼接生成的视频在边界处常出现明显割裂。
3. **360° 球面投影畸变**：等距柱状投影（ERP）将球面展开为 2:1 矩形，传统扩散模型难以处理其曲率变形与跨边界运动一致性。
4. **显存与推理效率瓶颈**：高分辨率全景视频生成易触发 OOM，且现有方法（如 4K4DGen、360DVD）依赖微调或优化过程，效率低。

---

## 核心贡献（创新点）
1. **提出 OSD 机制**：通过水平和垂直方向的移位去噪窗口，实现任意尺寸全景视频的无缝去噪，无需额外微调——与 Multi-Diffusion 等多窗口重叠方法本质不同，OSD 仅需少量重叠即可消除接缝且计算开销更低。
2. **引入 GMG 分层指导**：先以低分辨率生成粗粒度运动结构，再上采样后用 OSD 细化局部细节，解决高分辨率下多区域运动模式分离问题——区别于直接高分辨率生成或纯空间重叠策略。
3. **设计 Panoramic Projecting Denoiser**：将 ERP 球面投影映射为多个透视视角窗口进行去噪，并重新投影回球面，同时引入掩码与噪声重平衡机制处理极区重叠——与 360DVD 的隐式微调方法和 4K4DGen 的固定窗口优化本质不同。
4. **扩展至时间维度**：将 OSD 从空间域推广到时域，实现长时长（突破模型原生帧数限制）和可循环（loopable）全景视频生成——填补了现有方法仅支持短片段生成的空白。

---

## 方法详解
**整体流程（两阶段）**：
- **低分辨率阶段**：建立粗粒度运动结构；360° 设置下使用 Panoramic Projecting Denoise 初始化；常规透视设置下早期去噪步使用带重叠的 Offset Shifting。
- **超分阶段**：利用更多移位窗口细化高分辨率结果，并通过 GMG 从低分辨率视频获取全局运动引导。

**核心公式与机制**：

1. **Offset Shifting Denoiser (OSD)**：
   - 将 $W_p \times H_p$ 全景潜变量 $Z$ 划分为 $n_W \times n_H$ 个窗口，每个去噪步在水平和垂直方向移位偏移量。
   - 水平方向将潜变量视为"环"，左右边界相连（out-of-boundary 区域由另一侧对应区域填充）。
   - 公式：$Z_t = Con_{1:n_W, 1:n_H}(\Phi_\theta(t, c, Split(Z_{t-1}, i, j, t, n_W, n_H)))$

2. **Global Motion Guidance (GMG)**：
   - 先在低分辨率下运行 OSD：$Z_{LR}$，再经插值（bicubic 等）和噪声注入作为高分辨率初始值。
   - 公式：$Z_{HR^0} = \Phi_\theta^{OSD}(noise(inter(\Phi_\theta^{OSD}(Z_{LRT}))))$
   - 分层保证全局布局一致性和局部细节丰富性。

3. **360° FoV Panoramic Projecting Denoiser**：
   - 球面坐标 $(x,y,z)$ 经式 (3)(4) 映射到 ERP 平面 $(u,v)$。
   - 将 ERP 划分为 $n_\alpha \times n_\beta$ 个视角窗口，每个窗口视角为 $(f, a_{i,j,t})$，去噪后再投影回球面。
   - 极区重叠处理：维护掩码 $M_d$，对已去噪区域重新注入噪声以保持时间步一致性：
     $Z'_t[x,y] = \sqrt{\alpha_t} Z_t[x,y] + \sqrt{1-\alpha_t}\epsilon_t$（若 $M_d=1$）

4. **时序移位去噪（长视频/循环视频）**：
   - 将长视频潜变量沿时序维度切分为 $n_F$ 个片段窗口（每个窗口帧数 $\le F_\theta$），步间移位产生时序重叠。
   - 循环生成：首尾帧视为相连，窗口可跨越边界，实现无缝 loop。

---

## 实验与结果
**基线方法**：360DVD [25]、4K4DGen [18]、Scalecrafter [11]、VividDream [15]

**数据集与评估协议**：基于 VideoCrafter2（文本条件）和 VideoCrafter1（图像条件），采用 VBench [12]、CogVideoX [28] 和 Q-Align [26] 评估协议，随机选取 100 个生成视频案例评估。

**主要定量结果**（Table 1）：

| 指标 | 360DVD | DynamicScaler | 提升 |
|------|--------|---------------|------|
| CLIP-Score↑ | 0.293 | **0.302** | +3.1% |
| Image Quality↑ | 0.436 | **0.583** | +33.7% |
| Dynamic Degree↑ | 0.412 | **0.783** | +90.0% |
| Motion Smoothness↑ | 0.917 | **0.963** | +5.0% |
| Temporal Flickering↓ | 0.964 | **0.982** | +1.9% |
| Q-Align(I)↑ | 0.485 | **0.632** | +30.3% |
| Q-Align(V)↑ | 0.532 | **0.613** | +15.2% |

**用户研究**（Table 2）：20 名参与者，DynamicScaler 在图形质量、帧一致性、左右连续性、运动模式和场景丰富度五大维度上均获最高评分（同案例比较中 Graphics Quality 4.6 vs 3.3，连续性和内容分布优势显著）。

**消融实验**（Table 3）：
- w/o OSD：Image Quality 0.564，Motion Smoothness 0.948，Temporal Flickering 0.905
- w/o GMG：Image Quality 0.571，Motion Smoothness 0.961，Temporal Flickering 0.946
- Full Method：各指标最优，证明 OSD 和 GMG 均必要

**关键结论**：DynamicScaler 在恒定 VRAM 消耗下实现任意分辨率/宽高比生成，且支持无限时长和循环视频，超越现有方法。

---

## 相关工作脉络
1. **Multi-Diffusion [2] / SyncDiffusion [14]**：多窗口重叠去噪生成全景图像，但计算开销大、主要针对静态图像；本文 OSD 以最少重叠实现全景视频去噪，扩展至动态场景。
2. **360DVD [25]**：微调视频扩散模型于 ERP 空间生成 360° 视频，但分辨率低、有插值伪影；本文无需微调，先投影到透视空间去噪再映射回 ERP，保真度更高。
3. **4K4DGen [18] / VividDream [15]**：重叠区域动画生成，但固定窗口限制运动范围、依赖优化流程；本文 OSD 允许跨边界移动窗口，运动范围不受限且无需优化。
4. **Scalecrafter [11] / AnyLens [22]**：无微调高分辨率图像生成，但仅支持正方形或固定宽高比；本文支持任意宽高比和 360° 视场角，并扩展至视频。
5. **DreamScene360 [16] / LucidDreamer [5]**：基于 Gaussian Splatting 的 360° 场景生成，但局限于静态场景；本文专注于动态视频生成，兼顾时空一致性。

---

## 局限性与未来方向
- **局限性**：
  1. 基于预训练扩散模型的移位去噪策略，在极端运动场景（如高速旋转、大范围形变）下的细节保真度仍有提升空间。
  2. GMG 依赖低分辨率引导，若低分辨率阶段运动结构错误，将影响高分辨率生成质量。
  3. 360° FoV 生成在极地附近（高纬度区域）仍存在一定程度的畸变和不一致性。
- **未来方向**：
  1. 探索更高效的时序移位策略以支持超长时间视频（分钟级）生成。
  2. 结合 3D 先验或神经辐射场（NeRF）进一步提升全景视频的几何一致性。
  3. 将框架扩展至多视角/多模态条件（如深度、法线图）输入的全景视频生成。

---

## 研究启发与可借鉴点
1. **移位去噪窗口的通用性**：OSD 的"环状边界连接"设计可迁移至其他需要处理周期边界的生成任务（如纹理合成、地图生成、天空穹顶渲染）。
2. **分层运动引导策略**：GMG 的"低分辨率结构 + 高分辨率细节"两阶段范式，可推广至任意分辨率的视频/图像超分生成任务，避免端到端高分辨率训练的显存瓶颈。
3. **无微调扩展范式**：本文完全不调用目标模型的梯度，仅通过推理时的窗口移位和重投影实现扩展，这一"训练-free"思路可复用于其他需要适配新分辨率/新域的场景。
4. **噪声重平衡机制**：式 (6) 的掩码驱动噪声重注入策略，可用于处理多窗口重叠区域的不一致性，对 Patch-based 生成方法有参考价值。
5. **时序循环边界处理**：将首尾帧视为相连的循环机制，可直接应用于需要无缝循环的动画/短视频生成任务。

---

## 关键术语表
**Offset Shifting Denoiser (OSD)**：通过在水平和垂直方向逐步移位去噪窗口，实现任意尺寸全景视频的无缝、连贯去噪。
**Global Motion Guidance (GMG)**：利用低分辨率视频的全局运动结构引导高分辨率细化过程，确保大尺度运动一致性。
**Equirectangular Projection (ERP)**：将球面 360°  panorama 展开为 2:1 宽高比矩形的常用投影方式。
**Panoramic Projecting Denoiser**：将 ERP 潜变量投影为多个透视视角窗口进行去噪，再重新投影回球面的 360° 生成机制。
**Loopable Video**：首尾帧无缝衔接、可无限循环播放的动态视频序列。
**VBench**：综合视频生成质量评测基准，涵盖多个维度的自动化评估指标。
**Q-Align**：基于大语言模型的视觉质量评估工具，提供图像/视频的多维评分。
**Tuning-Free**：无需对预训练模型进行微调或参数更新，仅通过推理策略调整即可适配新任务。

---

## 可复现要素
- **数据集**：论文未明确提及训练数据集，基于 VideoCrafter1/2 预训练模型；评测数据来自公开基准（VBench/Q-Align 协议）。
- **代码**：项目页面 https://dynamicscaler.pages.dev/new，论文未明确声明 GitHub 仓库，需进一步确认。
- **权重**：依赖 VideoCrafter1（图像条件）和 VideoCrafter2（文本条件）公开权重。
- **关键超参**：论文未详细列出窗口大小、移位偏移量、去噪步数等超参数，需参考项目页面或源码。

---
