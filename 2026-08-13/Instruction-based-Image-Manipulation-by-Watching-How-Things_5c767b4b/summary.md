---
title: "Instruction-based-Image-Manipulation-by-Watching-How-Things"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Cao_Instruction-based_Image_Manipulation_by_Watching_How_Things_Move_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 07:32:55"
field: "图像编辑与生成"
keywords: ["instruction-based image editing", "video frame dataset", "spatial conditioning", "non-rigid transformation", "diffusion model", "multimodal LLM"]
innovations: ["视频帧+MLLM构建真实目标编辑数据集", "空间维度条件拼接策略提升内容一致性", "兼容mask与ControlNet的灵活编辑框架"]
benchmarks: ["CLIP-D", "CLIP-Inst", "CLIP-I", "Human Preference"]
---

# 论文速读：Instruction-based-Image-Manipulation-by-Watching-How-Things

## 一句话总结
论文提出 InstructMove，一种基于指令的图像编辑模型，通过从视频中采样帧对并利用多模态大语言模型（MLLM）生成编辑指令来构建大规模真实数据集，使模型能够完成非刚性变换（姿态、表情、视角调整）等复杂编辑任务，同时保持源图像内容与身份的一致性。

## 研究问题与动机
- **现有方法依赖合成目标图像**：InstructPix2Pix、EmuEdit 等方法使用 tuning-free 技术生成目标图像，引入伪影且与源图外观偏差大，难以处理真实照片的复杂非刚性变换。
- **数据集缺乏复杂变换类型**：现有大规模编辑数据集仅支持对象叠加、风格迁移等高-level 编辑，缺少姿态调整、视角变化等真实动态变换样本。
- **零样本编辑方法效率低且鲁棒性不足**：NullTextInversion、MasaCtrl 等需要在采样过程中进行逆过程操作，速度慢且对结构变换编辑失败率高。
- **缺乏针对非刚性变换的评估基准**：现有 benchmark 主要评估风格修改、对象替换等，无法衡量模型在保持身份一致性的同时进行结构变换的能力。

## 核心贡献（创新点）
1. **视频帧 + MLLM 的数据集构建流程**：从互联网视频中采样帧对，利用 MLLM（GPT-4o、Pixtral-12B）直接分析帧对差异并生成高质量编辑指令，构建 600 万真实目标编辑数据集，与 InstructPix2Pix 等依赖合成数据的方法本质不同。
2. **空间条件策略（Spatial Conditioning）**：将源图像 latent 与噪声目标 latent 沿宽度维度拼接输入 denoising U-Net，而非传统的通道维度拼接，使模型在保持源内容一致性的同时能够进行结构性变换。
3. **兼容多种控制机制的扩展能力**：由于不修改底层 T2I 架构，模型可无缝集成 mask 引导局部编辑以及 ControlNet、T2I-Adapter 等空间控制条件，实现精确的非刚性编辑。

## 方法详解
- **数据采样与过滤**：从视频中以固定时间间隔（3-5秒）采样帧对 $(I^s, I^e)$，使用 RAFT 计算光流 $O_{I^s I^e}$，推导流动幅度 $M = \sqrt{O_x^2 + O_y^2}$，保留中等运动帧对；通过反向扭曲计算背景遮挡 mask，过滤背景变化过大的帧对。
- **指令生成**：使用 Pixtral-12B 等 MLLM 分析帧对差异，生成以动作动词开头（如 "Move", "Adjust"）的绝对表述指令，允许 MLLM 拒绝难以描述的复杂帧对。
- **空间条件训练**：通过预训练 VAE 将源图和目标图编码为 latent $z^s$ 和 $z^e$，对 $z^e$ 添加噪声得到 $z_t^e$，沿宽度维度拼接得到输入 $z_t = \text{Concat}_{width}([z^s, z_t^e])$，输入 denoising U-Net $\epsilon_\theta$ 预测噪声，裁剪右半部分计算损失：
  $$\mathcal{L}_{\text{Edit}} = \mathbb{E}_{z_t, C, t, \epsilon}[\|\epsilon - \text{Crop}_{width}(\epsilon_\theta(z_t, C, t))\|^2]$$
- **Mask 局部编辑**：推理时通过 mask $m$ 混合更新 latent 与源 latent：$z_{t-1}^* = (1-m) \cdot z_{t-1}^s + m \cdot z_{t-1}$。
- **训练配置**：在 8×A100 GPU 上以 batch size 256、学习率 $1\times10^{-4}$ 训练 100K 步，输入分辨率 512×512，DDIM 50 步采样。

## 实验与结果
- **数据集**：600 万帧对，聚焦非刚性变换、元素重排、视角变化，Table 1 显示是首个支持此类复杂编辑的真实目标大规模数据集。
- **评估基准**：自建 50 张图像的测试集，涵盖姿态、表情、视角、形状等结构变换，使用 CLIP-D（目标描述对齐）、CLIP-Inst（指令对齐）、CLIP-I（内容保真）三项指标。
- **定量结果**（Table 2）：Ours 取得 CLIP-D=0.1361↑、CLIP-Inst=0.8724↑、CLIP-I=0.9275，均优于 InstructPix2Pix（CLIP-D=0.0887, CLIP-I=0.9380）、MagicBrush（CLIP-D=0.0972）等基线。
- **人类偏好**（Table 3）：87.62% 的情况下用户选择本方法为最优。
- **消融**（Table 4）：使用 InstructPix2Pix 数据训练（SC+IP2P data）效果显著下降；改用通道条件拼接（CC+Our data）在 CLIP-I 上降至 0.8552，验证空间条件的必要性。

## 相关工作脉络
- **InstructPix2Pix [5]**：开创指令驱动图像编辑，使用 GPT-3 生成指令并通过 Prompt-to-Prompt 合成目标图像，目标是合成数据，无法处理非刚性变换。
- **EmuEdit [34]**：使用 LLaMA 2 和 Prompt-to-Prompt 生成局部 mask 配对数据，目标仍为合成图像，缺乏真实动态变换。
- **MagicBrush [42]**：手动标注小数据集（10K），目标由 DALL·E 2 生成，规模小且依赖合成。
- **Zero-shot 方法（NullTextInversion [26], MasaCtrl [7], Imagic [19]）**：基于 DDIM inversion 和注意力修改，速度慢、鲁棒性差，难以处理复杂结构变换。
- **ControlNet [43] / T2I-Adapter [27]**：空间控制条件网络，本文将其与指令编辑模型结合，实现 pose/sketch 引导的精确编辑，而 InstructPix2Pix 等不支持此扩展。
- **UltraEdit [45]**：基于合成数据的大规模指令编辑数据集（4.1M），同样缺乏真实非刚性变换样本。

## 局限性与未来方向
- **MLLM 指令质量限制**：MLLM 可能生成不准确指令或遗漏部分变换，导致模型产生意外视角偏移等偏差。
- **现实变换范围的约束**：视频帧仅捕获真实动态变化，无法处理风格迁移、对象替换等艺术性编辑，需与其他数据集结合。
- **无法精确定位特定对象**：纯文本指令在局部编辑时存在歧义，需依赖 mask 等额外控制。
- **未来方向**：改进过滤流程（引入 human-in-the-loop）、提升 MLLM 理解能力、融合多源数据集以扩展编辑类型。

## 研究启发与可借鉴点
- **视频帧作为图像编辑数据源**：视频天然保持身份一致性并提供真实动态变换，可作为高质量配对数据源，值得在其他图像生成任务中探索。
- **空间维度条件拼接策略**：沿宽度/高度拼接而非通道拼接，使参考图与噪声输入解耦，这一思路可迁移到其他条件生成任务中需要保持内容一致性同时允许结构变化的场景。
- **MLLM 直接分析图像对生成指令**：相比仅基于文本的指令生成，MLLM 直接观察源-目标图像对能捕捉更丰富的变换细节，适用于指令级视觉理解任务。
- **不自修改架构的兼容性设计**：保持基础 T2I 模型结构不变，仅改变输入条件方式，使模型可无缝集成 ControlNet/mask 等控制模块，为后续扩展提供灵活框架。

## 关键术语表
- **InstructMove**：本文提出的指令驱动图像编辑模型，基于视频帧数据和空间条件策略训练。
- **Spatial Conditioning**：将源图像 latent 与噪声目标 latent 沿宽度维度拼接的条件输入策略，区别于传统的通道维度拼接。
- **Non-rigid Transformation**：非刚性变换，指主体姿态、表情等发生的非线性形变编辑。
- **CLIP-Inst**：本文提出的新评估指标，通过 MLLM 生成输出图像指令并与原始指令计算 CLIP 距离，衡量指令遵循程度。
- **MLLM**：Multimodal Large Language Model，多模态大语言模型，用于直接分析帧对并生成编辑指令。
- **RAFT**：Recurrent All-pairs Field Transforms，用于计算帧间光流的网络。
- **Channel Conditioning**：传统条件策略，将参考图像沿通道维度与噪声拼接输入 U-Net。

## 可复现要素
- **数据集**：论文声明构建了 600 万帧对数据集，项目页提供链接（论文提及 "The project page is available here"），具体公开状态需查阅项目页。
- **代码**：论文未明确声明代码开源情况，需查阅项目页。
- **权重**：基于 Stable Diffusion V1.5 微调，论文未提及权重是否开源。
- **关键超参**：分辨率 512×512，学习率 $1\times10^{-4}$，batch size 256，100K 迭代步数，DDIM 50 步采样，8×A100 GPU。
