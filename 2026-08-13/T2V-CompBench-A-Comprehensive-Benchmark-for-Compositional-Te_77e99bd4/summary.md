---
title: "T2V-CompBench-A-Comprehensive-Benchmark-for-Compositional-Te"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Sun_T2V-CompBench_A_Comprehensive_Benchmark_for_Compositional_Text-to-video_Generation_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 09:44:21"
field: "文本到视频生成评测基准"
keywords: ["text-to-video", "compositional generation", "benchmark", "evaluation metric", "multimodal large language model", "video generation", "spatiotemporal understanding"]
innovations: ["提出首个组合式文本到视频生成基准T2V-CompBench，涵盖7大类别1400个提示", "设计MLLM-based/Detection-based/Tracking-based三类细粒度评估指标并验证与人工评估的相关性", "系统评测23个T2V模型，揭示当前模型在动态属性绑定和运动控制方面的显著不足"]
benchmarks: ["T2V-CompBench"]
---

# 论文速读：T2V-CompBench-A-Comprehensive-Benchmark-for-Compositional-Te

## 一句话总结
本文提出了**首个面向组合式文本到视频生成（compositional T2V generation）的综合基准测试 T2V-CompBench**，包含7大类别共1400个文本提示，并设计了MLLM-based、Detection-based和Tracking-based三类评估指标，系统评测了23个T2V模型的组合生成能力。

## 研究问题与动机
1. **组合式T2V生成尚未被系统探索**：现有T2V研究多聚焦于简单单对象提示的视频生成，而"将多个对象、属性、动作和运动组合到视频中"这一关键能力缺乏深入研究。
2. **现有视频基准无法评估组合性**：VBench、EvalCrafter、FETV等现有T2V基准主要针对视频质量、运动质量和单对象文本-视频对齐，缺少对多对象组合生成能力的定义与评测。
3. **传统指标无法刻画时空组合性**：IS、FVD、FID、CLIPScore等通用指标仅能评估单帧质量或粗粒度对齐，无法感知视频中对象-属性绑定、空间关系、运动方向等细粒度时空组合信息。
4. **T2I组合基准不能直接迁移**：T2I-CompBench等针对静态图像设计的基准无法处理视频中"时序动态"（temporal dynamics）的评估需求。

## 核心贡献（创新点）
1. **首个组合式T2V生成基准**：设计了7大类别（一致属性绑定、动态属性绑定、空间关系、运动绑定、动作绑定、对象交互、生成数量）共1400个文本提示，覆盖空间与时间双重维度的组合性定义。
2. **三类细粒度评估指标体系**：提出MLLM-based（Grid-LLaVA、D-LLaVA）、Detection-based（G-Dino）、Tracking-based（DOT）三类指标，针对不同类别设计专属评估方法，并通过与人工评估的相关性验证有效性。
3. **23个T2V模型的全面基准评测与深入分析**：涵盖17个开源模型和6个商业模型，揭示了各模型在不同组合类别上的能力差异与演进趋势，为社区提供了明确的差距分析。

## 方法详解

### 3.1 提示构建流程
- **词汇库构建**：分析1.67M真实用户提示（VidProM [68]），通过WordNet提取元类别，筛选高频实体对象（thing类）和活跃动词，最终收集260个物体名词、200个活跃动词、80个属性词。
- **GPT-4生成**：按类别要求使用GPT-4结合词汇库生成提示，每类别200个，其中20%为罕见/挑战性提示以确保泛化能力评估。所有提示均含至少一个活跃动词（active verb），避免生成静态视频。

### 3.2 七类别定义
| 类别 | 核心要求 | 子类别 |
|------|---------|--------|
| Consistent Attribute Binding | 两个对象各自属性在整个视频中保持一致 | 颜色、形状、纹理、人类相关属性 |
| Dynamic Attribute Binding | 对象属性随时间动态变化 | 颜色&光照变化、形状&大小变化、纹理变化、组合变化 |
| Spatial Relationships | 两个对象的空间相对位置 | 左/右、上/下、前/后（含2D/3D） |
| Motion Binding | 指定对象的移动方向 | 左移、右移、上移、下移 |
| Action Binding | 两个对象各自执行不同动作 | 常见/罕见动作、罕见对象-动作配对 |
| Object Interactions | 多对象间的物理或社交交互 | 物理交互（运动/状态改变）、社交交互 |
| Generative Numeracy | 生成指定数量的对象 | 单对象1-8个、双对象组合计数 |

### 3.3 评估指标设计

**MLLM-based指标**：
- **Grid-LLaVA（一致属性绑定、动作绑定、对象交互）**：将视频均匀采样6帧拼接为图像网格输入Image Grid，结合chain-of-thought机制与解耦问题（disentangled questions）——先让MLLM描述视频内容，再逐项评估文本-视频对齐，避免幻觉。
- **D-LLaVA（动态属性绑定）**：针对当前Video LLM在该任务上表现不佳的问题，改用Image LLM（LLaVA）逐帧评估。将提示解析为初始状态和最终状态，设计评分函数鼓励首帧对齐初始状态、末帧对齐最终状态、中间帧居中。

**Detection-based指标（G-Dino）**：
- **2D空间关系**：基于GroundingDINO检测边界框，按中心点坐标判断关系（如$|x_1 - x_2| > |y_1 - y_2|$ 且 $x_1 < x_2$ 判定为"左侧"），逐帧评分后取视频平均。
- **3D空间关系**：结合Segment Anything预测掩码、Depth Anything预测深度图，用对象内像素平均深度计算前后关系。
- **生成数量**：统计各类别检测到的对象数量，与提示数量匹配则得1分，否则0分。

**Tracking-based指标（DOT）**：
- 用GroundedSAM获取前景对象和背景掩码，用DOT [47]追踪前后景关键点，计算平均运动向量，两者的差值即为对象的真实运动方向，与提示匹配度决定得分。

### 3.4 公平性处理
为公平对比，所有模型统一抽取6帧用于MLLM评估、16帧用于检测评估、8 FPS用于追踪评估。

## 实验与结果

### 评测设置
- **模型**：23个T2V模型（17开源+6商业），包括DiT-based（CogVideoX-5B、Mochi、Latte、Open-Sora系列）、UNet-based（VideoCrafter2及其衍生模型）、商用模型（PixVerse-V3、Kling-1.0、Dreamina 1.2、Gen-3、Pika-1.0等）
- **人工评估**：每类别随机抽取15个提示×6个模型=90个生成视频，另加10+11个ground truth视频，共651个视频；AMT三名标注者打分，取平均后与自动指标计算Kendall's τ和Spearman's ρ

### 主要结果（Table 2，分数归一化至0-1）

| 类别 | 最优模型 | 得分 | 类别 | 最优模型 | 得分 |
|------|---------|------|------|---------|------|
| Consist-attr | PixVerse-V3 | **0.7060** | Spatial | Dreamina 1.2 | **0.5773** |
| Dynamic-attr | Gen-3 | **0.0687** | Motion | PixVerse-V3 | **0.2867** |
| Action | PixVerse-V3 | **0.8722** | Numeracy | PixVerse-V3 | **0.6066** |
| Interaction | PixVerse-V3 | **0.8309** | — | — | — |

- **最强模型**：PixVerse-V3在6/7类别中获得最高分，唯一未夺冠的是Dynamic-attr（Gen-3以0.0687略胜）
- **最难点**：Dynamic-attr普遍得分极低（最高仅0.0687），是当前模型的最大短板
- **演进趋势**：较新模型（如CogVideoX-5B、T2V-Turbo-V2）在运动相关类别上优于早期模型；VideoCrafter2及其衍生模型在属性绑定和对象交互上表现突出

### 指标有效性（Table 1）
所提指标与人工评估相关性显著优于传统指标：Grid-LLaVA在Consist-attr上τ=0.6373/ρ=0.7461，D-LLaVA在Dynamic-attr上τ=0.4362/ρ=0.5061，G-Dino在Spatial上τ=0.5769/ρ=0.7057，DOT在Motion上τ=0.4523/ρ=0.5366。

## 相关工作脉络

1. **T2I-CompBench [20]**：首个组合式T2I生成基准，定义了属性绑定、空间关系等类别；本文将其扩展至视频域，引入时序动态维度的评估（动态属性绑定、运动绑定等）。
2. **VBench [22] / EvalCrafter [40] / FETV [41]**：综合T2V评测基准，但侧重单对象场景的视频质量和文本对齐，缺乏对多对象组合能力的定义；本文填补这一空白。
3. **ChronoMagic-Bench [82]**：评估延时视频生成的基准，关注自然变换过程；本文额外纳入人工属性变化视频，形成更全面的组合性定义。
4. **VPEval [9] / VideoDirectorGPT [35]**：基于检测的T2I/T2V评估方法；本文将其扩展为G-Dino检测指标和DOT追踪指标，并设计了更完善的2D/3D空间关系及运动方向评估。
5. **Image Grid [24] / PLLaVA [76]**：多帧视频理解VL M方法；本文 empirically发现Image Grid优于PLLaVA，并引入chain-of-thought机制减少幻觉。
6. **TempCompass [42]**：Video LLM理解能力评测；本文借鉴其动态属性变化分类方案（颜色/形状/纹理/组合）。

## 局限性与未来方向

1. **动态属性绑定极难**：所有模型在此类别上得分极低，表明现有T2V模型缺乏时序属性演变建模能力。
2. **空间关系理解薄弱**：模型难以区分"左/右"等方位词，空间关系仍是主要瓶颈之一。
3. **运动方向控制不足**：模型难以准确生成指定方向运动，提示中虽有活跃动词但实际运动控制仍很弱。
4. **数量生成不精确**：仅在数量≤3时表现较好，较大数量（>3）生成准确率骤降。
5. **评测采样策略局限**：固定帧数抽样（6/16帧）可能无法完整捕捉长时序动态，视频时长也限制了复杂交互的生成。
6. **未来方向**：需要专门的含细粒度时序标注的训练数据，以及运动控制的专门模块设计来提升运动/交互生成能力。

## 研究启发与可借鉴点

1. **图像网格+思维链是有效的视频理解范式**：将多帧拼接为image grid输入LLaVA，再配合chain-of-thought解耦提问，能显著提升视频属性的时序评估精度，此模式可迁移至其他视频理解/评测任务。
2. **检测+追踪分离评估运动方向**：通过区分前景与背景运动向量来提取对象真实运动方向的方法，有效解耦了相机运动与对象运动，可用于视频运动质量评估的其他场景。
3. **真实用户提示驱动的词汇库构建策略**：基于大规模真实提示统计高频词汇并筛选评估友好类别，比单纯人工设计更具代表性，可借鉴到其他生成基准构建。
4. **Per-category专属最优指标选择**：不同组合类别适配不同评估范式（MLLM/Detection/Tracking），这一"因地制宜"的策略比单一指标更可靠，可为多维度评测提供方法论参考。
5. **结合LLM布局规划提升空间理解**：LVD模型通过LLM-guided layout planning在空间关系上显著优于基座ModelScope，表明引入高层语义规划模块可有效弥补扩散模型的空间推理短板。

## 关键术语表

**Compositional T2V Generation**：文本到视频的组合生成，指模型根据复杂文本提示准确组合多个对象、属性、动作和运动的能力。

**Consistent Attribute Binding**：一致属性绑定，指视频中对象所属属性（如颜色、形状）在整个视频序列中保持不变的组合能力。

**Dynamic Attribute Binding**：动态属性绑定，指对象属性随时间发生预期变化（如颜色渐变、形状变形）的时序组合能力。

**Motion Binding**：运动绑定，指模型将指定运动方向（左移/右移等）正确绑定到对应对象上的能力。

**Generative Numeracy**：生成数量能力，指模型按照文本提示生成正确数量对象的能力。

**Grid-LLaVA**：本文提出的基于图像网格的Video LLM评估指标，将视频多帧拼接为网格输入LLaVA进行组合评估。

**D-LLaVA**：本文提出的基于逐帧图像LLM的评估指标，专门用于评估动态属性绑定的时序变化一致性。

**DOT**：Dense Optical Tracking，本文提出的基于稠密光流追踪的运动方向评估指标，通过前后景运动向量差计算对象的真实运动方向。

## 可复现要素

- **数据集/提示集**：1400个提示，基于VidProM [68]词汇分析构建；论文未明确说明T2V-CompBench是否开源发布，需进一步确认
- **代码/权重**：评测代码未提及是否开源；所使用的T2V模型（如CogVideoX-5B、PixVerse-V3等）多为开源或商用API
- **关键超参**：MLLM评估帧数=6帧，检测评估帧数=16帧，追踪评估帧率=8 FPS；视频长度通常为2-5秒
- **人工评估**：AMT平台，每视频3名标注者，类别抽取15个提示×6模型
