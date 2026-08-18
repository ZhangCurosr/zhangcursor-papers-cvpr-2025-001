---
title: "DiffFNO-Diffusion-Fourier-Neural-Operator"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_DiffFNO_Diffusion_Fourier_Neural_Operator_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 06:00:38"
field: "图像超分辨率"
keywords: ["超分辨率", "傅里叶神经算子", "扩散模型", "任意尺度", "神经算子", "频域方法", "ODE求解器"]
innovations: ["Weighted FNO with Mode Rebalancing for high-frequency reconstruction", "Gated Fusion Mechanism for adaptive global-local feature integration", "Adaptive Time-Step ODE solver reducing 1000 steps to 30"]
benchmarks: ["DIV2K", "Set5", "BSD100", "Urban100", "Set14"]
---

# 论文速读：DiffFNO-Diffusion-Fourier-Neural-Operator

## 一句话总结
DiffFNO 提出了一种结合加权傅里叶神经算子（WFNO）与扩散框架的任意尺度图像超分辨率方法，通过频域模式重平衡（Mode Rebalancing）、门控融合机制（GFM）和自适应时间步ODE求解器（ATS），在保持计算效率的同时实现了SOTA的重建质量。

## 研究问题与动机
- **现有FNO的频谱偏差问题**：标准FNO通过截断傅里叶模式实现计算效率，但会丢失高频成分，而超分辨率任务高度依赖高频细节重建；MLP也存在频谱偏差，倾向于学习低频函数。
- **扩散模型计算开销大**：迭代反向扩散过程需要大量采样步骤（通常1000步），推理时间成本高，难以满足实际应用需求。
- **任意尺度泛化挑战**：物理模拟数据与真实图像存在本质差异，从计算开销到高频细节恢复均面临挑战，现有方法在超出训练分布的尺度上泛化能力不足。
- **全局与局部特征融合困难**：单一算子难以同时捕捉全局结构依赖与局部精细纹理，缺乏自适应融合机制。

## 核心贡献（创新点）
1. **Weighted Fourier Neural Operator (WFNO) with Mode Rebalancing**：通过在傅里叶域引入可学习权重函数 $\mathbf{w}_l(\pmb{\xi}) = 1 + \gamma_l \cdot \|\pmb{\xi}\|^\alpha$ 放大高频分量，克服传统FNO模式截断导致的高频信息丢失问题。
2. **Gated Fusion Mechanism (GFM)**：设计空间门控机制动态融合WFNO的全局频谱特征与AttnNO的局部空间特征，替代简单拼接策略实现自适应全局-局部平衡。
3. **Adaptive Time-Step (ATS) ODE Solver**：将反向扩散重构为确定性ODE，并通过学习函数 $\phi_\psi(t)$ 自适应分配非均匀时间步，将采样步数从1000降至30步，大幅加速推理。
4. **任意尺度SOTA性能**：在DIV2K验证集及Set5、BSD100等基准上，DiffFNO在×2至×12各尺度均优于现有方法，超出训练分布的尺度（×6、×8、×12）提升达2-4 dB PSNR。

## 方法详解
**整体架构**：输入LR图像经CNN编码器（EDSR-baseline或RDN）提取特征后，并行送入WFNO与AttnNO，通过GFM融合后投影至RGB空间，再由ATS ODE求解器完成反向扩散过程生成HR图像。

**Loss函数**：采用score matching损失
$$\mathcal{L}(\theta) = \mathbb{E}_{t, \mathbf{x}_0}\left[\|s_\theta(\mathbf{x}_t, t) - \nabla_{\mathbf{x}_t}\log p_t(\mathbf{x}_t|\mathbf{x}_0)\|^2_2\right]$$

**WFNO与Mode Rebalancing**：
- 标准FNO层：$\mathbf{v}_{l+1}(\mathbf{r}) = \sigma(\mathcal{W}_l\mathbf{v}_l(\mathbf{r}) + \mathcal{K}_l\mathbf{v}_l(\mathbf{r}))$
- 频域卷积：$\mathcal{K}_l\mathbf{v}_l(\mathbf{r}) = \mathcal{F}^{-1}(\mathbf{w}_l(\pmb{\xi})\cdot\mathbf{P}_l(\pmb{\xi})\cdot\mathcal{F}[\mathbf{v}_l](\pmb{\xi}))$
- 权重函数：$\mathbf{w}_l(\pmb{\xi}) = 1 + \gamma_l\cdot\|\pmb{\xi}\|^{0.7}$，保留全部模式并对高频加权

**Gated Fusion Mechanism**：
- 门控图：$\mathbf{G} = \sigma(\text{Conv}_{1\times1}([\mathbf{v}_{\text{WFNO}}, \mathbf{v}_{\text{AttnNO}}]))$
- 融合特征：$\mathbf{v}_{\text{fused}} = \mathbf{G}\odot\mathbf{v}_{\text{WFNO}} + (1-\mathbf{G})\odot\mathbf{v}_{\text{AttnNO}}$

**前向扩散过程**：
- 采用修正的VP SDE：$d\mathbf{x}_t = -\frac{1}{2}\beta(t)(\mathbf{x}_t - \mathbf{D}\mathbf{x}_t)dt + \sqrt{\beta(t)}d\mathbf{w}$
- 线性噪声调度：$\beta(t) = \beta_{\min} + (\beta_{\max} - \beta_{\min})\cdot\frac{t}{T}$，其中$\beta_{\min}=0.1, \beta_{\max}=20$
- 下采样算子D为双三次下采样

**ATS ODE求解器**：
- 时间步映射：$t_i = \phi_\psi^{-1}(i/N)$，其中$\phi_\psi(t) = \frac{\sum_{k=1}^K \psi_k t^k}{\sum_{k=1}^K \psi_k T^k}$，$K=3$
- 使用RK4方法求解概率流ODE：$\frac{d\mathbf{x}}{dt} = f(\mathbf{x},t) - \frac{1}{2}g(t)^2 s_\theta(\mathbf{x},t)$

## 实验与结果
**数据集与评估**：
- 训练：DIV2K
- 验证：DIV2K val + Set5、Set14、BSD100、Urban100
- 尺度：×2、×3、×4（训练分布内）及×6、×8、×12（OOD）
- 指标：PSNR、SSIM

**主要结果（DIV2K验证集，EDSR编码器）**：
| 尺度 | DiffFNO PSNR | 次优方法 | 提升幅度 |
|------|-------------|---------|---------|
| ×2 | 35.72 | LMI 35.40 | +0.32 dB |
| ×4 | 30.88 | HiNOTE 30.46 | +0.42 dB |
| ×8 | 26.78 | HiNOTE 26.41 | +0.37 dB |
| ×12 | 26.48 | HiNOTE 26.23 | +0.25 dB |

**消融实验**：
- WFNO vs FNO：MR机制带来约+0.5 dB提升
- 移除AttnNO/GFM：×4尺度PSNR从30.88降至30.05
- 移除ATS：推理步数从30增至1000，时间增加约60%
- DiffFNO推理时间141ms，显著优于SRNO的147ms

**最强结果**：在Urban100 ×4尺度达34.19 dB PSNR，超越HiNOTE约0.16 dB；在复杂纹理数据集上提升尤为明显。

## 相关工作脉络
- **FNO [26]**：基础傅里叶神经算子，通过谱卷积学习函数映射，但模式截断导致高频丢失，本文通过MR机制改进。
- **SRNO [54]**：首个将NO应用于任意尺度SR的工作，使用注意力机制增强，但未解决高频重建问题，DiffFNO通过WFNO+扩散框架超越。
- **HiNOTE [35]**：层级神经算子Transformer，引入频域感知损失先验，在部分尺度表现优异但泛化性不稳定。
- **LIIF [7] / LTE [24]**：基于局部隐式函数或频率估计的方法，擅长高频纹理但难以处理非周期性细节。
- **DDIM [45] / DPM-Solver [2,33]**：确定性ODE采样方法，本文采用ODE框架但创新性地引入自适应时间步分配。
- **Meta-SR [17]**：早期任意尺度SR方法，缺乏针对高频细节的专用机制，在大尺度上性能衰退明显。

## 局限性与未来方向
- **编码器依赖**：当前使用固定CNN编码器（EDSR/RDN），未探索端到端联合优化编码与算子。
- **噪声调度设计**：线性调度$\beta(t)$虽简单有效，但非最优调度策略可能限制性能上限。
- **AttnNO轻量化**：虽与WFNO并行且共享编码器，但Galerkin注意力仍增加额外计算开销。
- **测试集多样性**：主要在自然图像上验证，医学影像、卫星遥感等特殊领域尚未充分探索。
- **未来方向**：可扩展至视频超分辨率、多模态任务；探索更高效的采样策略（如蒸馏）；研究自适应噪声调度。

## 研究启发与可借鉴点
- **频域加权策略可迁移**：Mode Rebalancing思想可应用于其他频域方法（如WaveFNO、多小波算子），解决高频偏好问题。
- **门控融合机制的通用性**：GFM的并行双分支+空间门控设计可推广至其他多模态/多尺度特征融合任务。
- **自适应时间步分配**：ATS的学习型时间步策略可复用于其他扩散加速任务，无需额外微调即可适配不同数据分布。
- **扩散+算子的结合范式**：本文证明神经算子可作为扩散score network的有效 backbone，为"算子学习+生成模型"融合提供新思路。
- **实验设计借鉴**：使用相同编码器进行公平对比、系统评估OOD尺度泛化、详尽消融验证各组件贡献，值得效仿。

## 关键术语表
**Fourier Neural Operator (FNO)**：在傅里叶域进行谱卷积的神经算子，学习函数到函数的映射，支持任意分辨率输入。

**Mode Rebalancing (MR)**：WFNO中的高频加权机制，通过学习函数放大被截断的高频傅里叶分量。

**Gated Fusion Mechanism (GFM)**：空间维度的门控融合策略，自适应平衡全局频谱特征与局部空间特征。

**Adaptive Time-Step (ATS)**：基于学习函数的ODE时间步分配策略，根据图像复杂度动态调整积分步长。

**Variance-Preserving (VP) SDE**：保持信号方差的随机微分方程，用于建模扩散过程的前向退化。

**Score Matching**：训练score network逼近数据分布梯度$\nabla\log p(\mathbf{x})$的损失函数。

**Arbitrary-Scale Super-Resolution**：支持任意连续缩放因子的超分辨率，突破固定整数倍限制。

**Galerkin Attention**：投影到有限维子空间的注意力机制，用于高效近似kernel integral。

## 可复现要素
- **数据集**：DIV2K（训练）、DIV2K val + Set5/Set14/BSD100/Urban100（评估）；公开可用
- **代码**：论文未明确提及开源声明
- **权重**：论文未提及开源
- **关键超参**：$\alpha=0.7$（可选learnable）、$\beta_{\min}=0.1$、$\beta_{\max}=20$、$K=3$（多项式基函数数）、采样步数30、RK4求解器
- **编码器**：EDSR-baseline / RDN
- **下采样算子**：双三次下采样
