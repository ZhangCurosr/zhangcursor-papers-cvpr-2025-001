---
title: "SnowMaster-Comprehensive-Real-world-Image-Desnowing-via-MLLM"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Lai_SnowMaster_Comprehensive_Real-world_Image_Desnowing_via_MLLM_with_Multi-Model_Feedback_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 09:41:44"
field: "图像去雪与退化修复"
keywords: ["single image desnowing", "multimodal large language model", "preference optimization", "semi-supervised learning", "image quality assessment", "real-world generalization"]
innovations: ["提出MMPO将多模型反馈整合至DPO框架以增强MLLM降雪感知能力", "构建RealSnow10K大规模真实雪景数据集（>10K张）", "设计四维度评估框架结合半监督伪标签更新机制提升真实场景泛化性"]
benchmarks: ["RealSnow10K", "Snow100K-realistic"]
---

# 论文速读：SnowMaster-Comprehensive-Real-world-Image-Desnowing-via-MLLM

## 一句话总结
本文提出 SnowMaster 框架，通过构建大规模真实雪景数据集 RealSnow10K、设计多模型反馈优化（MMPO）增强多模态大语言模型（MLLM）的降雪感知能力，并结合平均教师半监督框架与伪标签筛选机制，显著提升去雪模型在真实场景下的泛化性能。

## 研究问题与动机
1. **合成数据主导导致真实场景泛化差**：现有去雪模型严重依赖合成降雪数据集训练，无法捕捉真实雪景的复杂退化特征，存在显著域偏移（domain gap）。
2. **真实雪景数据集匮乏且缺乏专用评估指标**：现有真实雪景数据集规模小，且无参考评估指标（non-reference metrics）受多种因素干扰，难以准确隔离并评估降雪退化。
3. **MLLM 缺乏降雪场景专项感知能力**：虽有 MLLM（如 Q-Instruct、LLaVA）具备一定低层视觉评估能力，但未针对降雪特征进行专项微调，无法可靠评估雪量、遮挡程度等关键维度。
4. **单一 MLLM 无法覆盖多维度评估需求**：不同 MLLM 在不同评估指标上各有优劣，单一模型难以同时准确评估雪量、遮挡、纹理、背景等多个维度。

## 核心贡献（创新点）
1. **构建 RealSnow10K 大规模真实雪景数据集**：收集并筛选超过 10,000 张高质量真实雪景图像，按雪量、类别等多维度标注，是目前规模最大的真实雪景数据集。→ 与已有小规模真实数据集（如 RealSnow）相比，在规模和标注质量上形成显著优势。
2. **提出多模型反馈优化（MMPO）**：借鉴 DPO 框架，收集多个 MLLM 对同一雪景图像的多维度评估输出，由专家排序构建偏好数据集，将多模型优势整合至单个 Q-Instruct 模型。→ 区别于单模型微调或单纯 RLHF，首次将多模型反馈引入降雪评估任务。
3. **设计四维度雪景评估框架**：从雪量、物体模糊度、纹理细节、背景清晰度四个角度系统化评估去雪效果，并通过多轮对话机制与多选项 logits 加权计算 MOS 风格分数。→ 与传统单一指标（如 NIQE、BRISQUE）相比，提供语义+视觉联合评估视角。
4. **构建基于伪标签更新的半监督训练框架**：将 MMPO 微调后的 Q-Instruct 作为无监督评估器，结合 mean teacher 框架与严格的双条件伪标签更新策略，指导 NAFNet 在真实雪景数据上的训练。→ 与纯监督或无伪标签半监督方法相比，有效提升真实场景去雪性能。

## 方法详解
1. **数据集构建流程**：从互联网收集超过 10 万张雪景图像，通过 MiniCPM-V-2.6 模型进行多轮对话筛选（判断真实性、水印、雪量估计、主题分类），最终保留 10,000+ 张高质量图像，划分为 1,500 张（MMPO 微调）、1,047 张（测试）、6,406 张（半监督训练）。
2. **偏好数据集构建**：选取 1,500 张图像，用预训练去雪模型预处理后扩展至 3,000 张；针对雪量和遮挡两个指标，收集 Q-Instruct-13b、LLaVA-1.5-13b、LLaVA-1.6-13b、MiniCPM-V-2.6 四个模型的输出，由专家排序生成 6 对偏好对，共计 36,000 对偏好样本。
3. **MMPO 损失函数**：基于 DPO 重参数化，奖励函数 $r(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$；MMPO 损失为 $L_{MMPO} = -\mathbb{E}[\log \sigma(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)})]$，引导模型输出向专家偏好对齐。
4. **图像评分机制**：将四维度评估设计为五选项多选题，利用 softmax 计算各选项概率 $p_i = \frac{e^{l_i}}{\sum e^{l_j}}$，通过加权平均得到 Score，对应 MOS 计算方式。
5. **半监督训练与伪标签更新策略**：前 20k 步仅使用 Supervised Loss（基于 Snow100K 的 PSNR），之后引入 Pseudo-label Loss（L1 距离）；伪标签数据库更新需同时满足两个条件：① 至少三个教师分数高于学生分数；② 教师分数高于数据库中当前存储分数。

## 实验与结果
1. **数据集与基线**：测试集包含 RealSnow10K 的 457 张图像和 Snow100K 的 1,329 张真实雪景图像；基线包括 SMGARN、TransWeather、NAFNet、SnowFormer、T³-DiffWeather；评估指标为 BRISQUE↓、NIQE↓、PIQE↓、PAQ2-PIQ↑（越低/越高越好）。
2. **MMPO 有效性**：微调后 Q-Instruct 在雪量、遮挡、纹理、背景四个指标上准确率分别提升 +5.83%、+7.64%、+6.59%、+5.16%。
3. **去雪性能对比**：在 RealSnow10K 上，SnowMaster 取得 BRISQUE=17.546、NIQE=3.058、PIQE=29.742、PAQ2-PIQ=70.304，全面优于所有基线；在 Snow100K-realistic 上同样取得最优 BRISQUE=16.305、NIQE=2.797、PIQE=30.162。
4. **消融实验**：去除半监督学习、MLLM 模块、MMPO 均导致性能下降；多角度评估（四维度）优于单角度评估；验证了各组件必要性。

## 相关工作脉络
1. **去雪方法演进**：从传统手工特征（Pei et al.）→ CNN 多阶段网络（DesnowNet）→ Transformer/双流架构（HDCWNet、DDMSNet、SMGARN）→ 扩散模型（T³-DiffWeather），本文在 NAFNet 基础上引入 MLLM 评估指导半监督训练。
2. **图像质量评估（IQA）**：传统统计指标（SSIM、PSNR）→ 深度学习回归（NIMA、HyperIQA）→ 视觉-语言对齐（CLIP-IQA+、LIQE）→ 离散文本评分（Q-Align），本文延续 Q-Align 思路但聚焦降雪退化多维度评估。
3. **多模态大语言模型**：GPT-4o、LLaVA、MiniCPM-V 等擅长高层语义理解，Q-Instruct、Depict-QA 尝试注入低层视觉能力，本文进一步针对降雪场景进行专项微调。
4. **偏好优化方法**：RLHF（Christiano et al.）、DPO（Rafailov et al.）、Rovrm 等，本文提出 MMPO 将多模型输出整合至 DPO 框架，而非仅依赖单一模型或人工反馈。
5. **半监督/伪标签方法**：Mean teacher、Contrastive SSL 等用于水下图像恢复等工作，本文将其引入真实雪景去雪，并通过严格双条件筛选降低伪标签噪声。
6. **域适应与真实场景增强**：传统方法依赖合成数据，本文通过真实数据集+MLLM 评估筛选实现域 gap 缩小，定位区别于纯合成数据训练方法。

## 局限性与未来方向
1. **计算开销较高**：MMPO 需要多个 MLLM 同时推理收集反馈，训练和评估阶段计算成本较高。
2. **多模型反馈依赖专家标注**：偏好数据集构建需要人工专家排序，扩展至更大规模或更多模型时成本较高。
3. **训练效率待提升**：论文结论部分明确提及"computational intensity from multiple evaluations signals a need for future enhancements in training efficiency and resource use"。
4. **可能存在的幻觉问题**：MLLM 在复杂雪景评估中仍可能存在幻觉，需进一步优化其感知可靠性。

## 研究启发与可借鉴点
1. **多模型反馈整合至 DPO 框架的思路可迁移**：MMPO 将多个模型优势整合的思路可推广至其他退化类型（雨、雾、低光）的评估与恢复任务。
2. **多维度评估框架设计值得借鉴**：将雪景评估拆解为雪量、遮挡、纹理、背景四个正交维度，为其他退化评估任务提供结构化思路。
3. **伪标签双条件更新策略可复用**：严格的更新条件（多数教师一致+超越历史最优）可有效降低噪声伪标签，适用于其他半监督恢复任务。
4. **MLLM 作为无参考评估器的应用范式**：利用多轮对话+多选项 logits 加权生成连续分数，替代传统 IQA 指标，可作为通用评估器设计参考。
5. **大规模真实数据集构建流程可复现**：从互联网海量收集→MLLM 多轮对话筛选→专家抽检的分类流程，可为其他退化数据集构建提供 pipeline 参考。

## 关键术语表
**MMPO（Multi-Model Preference Optimization）**：多模型反馈优化，通过收集多个 MLLM 的输出并由专家排序构建偏好对，将多模型优势整合至单一模型的 DPO 微调过程。
**RealSnow10K**：本文构建的大规模真实雪景数据集，包含超过 10,000 张标注的真实降雪图像，是目前规模最大的真实去雪数据集。
**DPO（Direct Preference Optimization）**：直接偏好优化，无需显式奖励模型，通过将偏好学习转化为分类问题来优化策略的强化学习方法。
**Mean Teacher**：平均教师半监督学习框架，通过教师模型和学生模型的指数移动平均实现伪标签生成与训练稳定性保障。
**BRISQUE**：无参考图像质量评估指标，通过空间自然场景统计特征衡量图像质量，值越低质量越好。
**NIQE**：无参考质量评估指标，衡量图像与无失真自然图像统计分布的偏离程度，值越低质量越好。
**PIQE**：无参考评估指标，通过检测块级失真评估图像质量，值越低质量越好。
**PAQ2-PIQ**：基于深度学习的无参考评估指标，预测图像全局和局部质量，值越高质量越好。

## 可复现要素
- **数据集**：RealSnow10K 已公开（Project page: https://alexlai2860.github.io/SnowMaster）
- **代码/权重**：项目页面已提供，具体代码仓库未在论文正文明确说明
- **关键超参**：MMPO 微调 epoch=1, lr=1e-6, batchsize=64, DeepSpeed ZeRO-stage 3；去雪模型训练 lr=2e-4, batchsize=32；伪标签更新在前 20k 步仅用 supervised loss
- **硬件**：2× Nvidia A800 GPU
- **基线模型**：NAFNet（基础去雪模型）、Q-Instruct-13b、LLaVA-1.5/1.6-13b、MiniCPM-V-2.6
