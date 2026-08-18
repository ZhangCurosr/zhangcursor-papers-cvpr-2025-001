---
title: "Revisiting-Backdoor-Attacks-against-Large-Vision-Language-Mo"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Liang_Revisiting_Backdoor_Attacks_against_Large_Vision-Language_Models_from_Domain_Shift_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 07:47:55"
field: "多模态大模型安全"
keywords: ["backdoor attack", "large vision-language models", "domain shift", "instruction tuning", "generalization", "MABA", "CVPR 2025"]
innovations: ["提出后门域泛化性评估维度", "揭示触发器域无关性与清洁区域竞争对泛化的影响", "设计多模态归因后门攻击MABA"]
benchmarks: ["COCO", "Flickr30K", "MIMIC-IT", "OpenFlamingo", "Blip-2", "Otter"]
---

# 论文速读：Revisiting-Backdoor-Attacks-against-Large-Vision-Language-Mo

## 一句话总结
本文首次在大视觉语言模型（LVLMs）指令微调阶段系统研究跨视觉/文本域偏移的后门攻击，提出“后门域泛化性”评估维度，并设计多模态归因后门攻击（MABA），将传统攻击的域泛化成功率提升36.4%，在仅0.2%毒化率下仍实现97%攻击成功率。

## 研究问题与动机
1. **跨域后门评估缺失**：现有LVLMs后门攻击研究多假设训练与测试数据分布一致，但实际指令微调场景中攻击者构造的训练集与用户测试集常存在显著视觉风格与文本密度偏移，导致传统后门攻击泛化性骤降。
2. **指令微调安全风险**：指令微调开放接受多源数据，攻击者可低成本注入恶意指令，但缺乏对跨域后门鲁棒性的量化评估，难以衡量真实部署威胁。
3. **触发器设计局限**：现有触发器多依赖模型参数或特定数据分布（如低频次成分、内容感知变换），在域偏移下极易失效，尚未探索域无关触发器与关键区域定位的协同机制。

## 核心贡献（创新点）
1. **提出后门域泛化性新评估维度**：首次量化跨视觉/文本域偏移时后门攻击的鲁棒性，填补LVLMs安全评估空白。
2. **揭示跨域泛化两大关键洞察**：触发器模式与数据域/模型架构的独立性提升泛化性；触发器与清洁语义区域的竞争互动中，引导模型优先预测触发器可显著增强泛化。
3. **设计多模态归因后门攻击（MABA）**：利用归因解释定位关键决策区域，注入域无关触发器（视觉：黄色椭圆；文本：罕见符号），实现跨域高效攻击。
4. **低毒化率高威胁实证**：在0.2%毒化率下All attacks achieve >97% ASR，证明极低污染即可严重威胁LVLMs安全，且Blip-2因可微调参数少而相对更鲁棒。

## 方法详解
- **域偏移数据构建**：使用Stable Diffusion生成表现主义/写实主义视觉风格偏移，结合GPT-3.5 Turbo、Qwen、LLaMA进行文本摘要/扩展，构建6类跨域指令数据集；用KS Statistic量化分布偏移程度。
- **ASR-G评估指标**：定义攻击归一化泛化率 $ASR\text{-}G = \min\left[1 + \frac{ASR_{\mathcal{D}^k} - ASR_{\mathcal{D}^t}}{\max(ASR_{\mathcal{D}^k}, ASR_{\mathcal{D}^t})}, 1\right]$，综合源域与目标域攻击成功率，值越高泛化性越强。
- **MABA触发器设计**：视觉触发器采用简单黄色椭圆（$\tau$），文本触发器使用罕见特殊符号（[ ] * { } < >），确保域无关性；通过LLM定位关键语义词插入文本触发器。
- **归因关键区域定位**：视觉侧将图像分割为区域集合，利用Chen等人归因方法计算清洁/毒化答案的掩码$m^c, m^p$，通过子模函数最大化选择关键区域$R^*$，最终掩码$m = m^c - (m^c \cap m^p)$控制触发器注入位置。
- **攻击优化目标**：微调阶段优化交叉模态融合层参数$\theta_1$，损失函数为$\theta^* = \arg\min\left[\lambda\sum_{\mathcal{D}^c}\mathcal{L}(f_\theta(q_i,x_i),y_i) + (1-\lambda)\sum_{\mathcal{D}^p}\mathcal{L}(f_\theta(\hat{q}_j,\hat{x}_j),y^p)\right]$，平衡清洁与毒化数据损失。

## 实验与结果
- **数据集与模型**：训练集使用MIMIC-IT的Image Captioning指令子集；测试集为COCO与Flickr30K； victim模型为OpenFlamingo、Blip-2、Otter。
- **基线攻击**：10种经典后门攻击，包括静态（BadNets、Blended、TextBadNets、AddSent）、内容感知（LowFrequency、WaNet、StyleBkd）、模型自适应（InputAware、GCG、DualKey）。
- **主要结果**：MABA将BadNets的Mean ASR-G从0.318提升至0.682（+36.4%）；MABA*将DualKey的ASR-G从0.638提升至0.834（+19.6%）。Blended基线ASR-G最高达0.986但触发器隐蔽性差。
- **低毒化率实验**：在0.2%、0.5%、1%毒化率下，所有攻击在COCO上ASR均超过97%，证明极低污染即可实现高成功率。
- **模型对比**：Blip-2因可微调参数较少，ASR-G显著低于OpenFlamingo和Otter，显示架构差异影响后门鲁棒性。

## 相关工作脉络
1. **Shu et al. (2023) 《On the exploitability of instruction tuning》**：聚焦静态指令攻击，本文扩展至跨域场景，揭示域偏移对攻击泛化性的量化影响。
2. **Lu et al. (2024) AnyDoor**：测试时后门攻击，本文针对指令微调阶段，考虑训练数据域偏移挑战，防御时机更早。
3. **BadCLIP等预训练后门攻击**：针对CLIP等预训练模型，本文转向成本更低、更易操作的指令微调阶段，威胁更贴近实际部署。
4. **VL-Trojan等LVLM后门研究**：未系统评估跨域泛化性，本文首次构建视觉/文本双重域偏移基准并公开评估协议。
5. **传统视觉后门（BadNets/Blended）**：依赖固定触发器，在域偏移下泛化性差；本文证明域无关触发器结合归因定位可突破此限制。
6. **GCG/DualKey等文本后门**：利用模型梯度优化触发器，但本文指出其对域偏移敏感，而罕见符号触发器更具泛化潜力。

## 局限性与未来方向
- 未深入分析不同LVLMs架构（如参数规模、融合机制）对后门泛化性的差异化影响，未来可探索架构敏感性。
- 缺乏系统防御研究，如后门检测、数据去污或鲁棒微调策略，可结合本文评估框架设计针对性防御。
- 实验局限于图像描述任务，其他多模态任务（VQA、对话生成）的跨域后门行为待验证。
- 触发器可见性（α=0.5）可能影响隐蔽性，如何在保持有效性的同时降低可检测性需进一步优化。

## 研究启发与可借鉴点
1. **域偏移数据构建范式**：Stable Diffusion+LLM可控偏移方法可迁移至其他多模态安全评估（如对抗鲁棒性、公平性测试）。
2. **触发器-清洁区域竞争洞察**：引导模型优先关注触发器的设计思路可启发防御端强化清洁语义的区域注意力，或攻击端设计更隐蔽的触发器。
3. **归因关键区域定位技术**：子模优化选择视觉关键区域的方法可扩展至其他多模态攻击（如测试时攻击）或模型解释性分析。
4. **低毒化率威胁警示**：0.2%污染即可实现97%成功率，提示实际部署需严格数据质量管控，微小污染风险亦不可忽视。
5. **ASR-G评估指标通用性**：归一化跨域成功率指标可推广至其他模型（如纯视觉/文本模型）的跨域鲁棒性评测，促进标准化比较。

## 关键术语表
- **Backdoor Domain Generalizability**：后门攻击在训练与测试数据分布不一致（跨视觉/文本域）时的有效性保持能力。
- **ASR-G**：Attack Success Rate normalized for Generalization，综合源域与目标域攻击成功率的归一化指标，值越高泛化性越强。
- **MABA（Multimodal Attribution Backdoor Attack）**：本文提出的多模态归因后门攻击，通过归因解释定位关键决策区域并注入域无关触发器。
- **Instruction Tuning**：指令微调，使用多样化指令数据对齐LVLMs输出的后训练过程，本研究主要攻击阶段。
- **Case I/II/III Attacks**：后门攻击分类，分别指静态触发、内容感知触发、模型自适应触发的攻击方法。
- **CLIP-based Similarity Shift**：基于CLIP模型计算的图像-文本相似度变化，用于分析域偏移对模型预测偏好的影响。
- **Submodular Region Selection**：通过子模函数最大化选择视觉关键区域，用于优化触发器注入位置。
- **Poisoning Rate**：毒化率，指训练数据中被植入后门样本的比例，低毒化率（如0.2%）仍可造成高威胁。

## 可复现要素
- 数据集：MIMIC-IT指令微调子集、COCO、Flickr30K；域偏移数据通过Stable Diffusion和LLM（GPT-3.5 Turbo、Qwen、LLaMA）生成。
- 代码：论文声明代码在线可用（链接见原文），但未提供具体仓库地址。
- 权重：未提及开源模型权重。
- 关键超参：α=0.5（触发器混合参数），poisoning rate=5%（主要实验），0.2%/0.5%/1%（低毒化率实验），λ在损失函数中平衡清洁与毒化数据损失（未指定具体值）。
