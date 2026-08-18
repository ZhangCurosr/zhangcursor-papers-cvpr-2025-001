---
title: "Exploring-Timeline-Control-for-Facial-Motion-Generation"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Ma_Exploring_Timeline_Control_for_Facial_Motion_Generation_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 07:04:20"
field: "面部动作生成与可控视频合成"
keywords: ["Facial Motion Generation", "Timeline Control", "Diffusion Model", "Time Series Clustering", "TICC", "3DMM", "FaceVerse", "Multimodal Generation"]
innovations: ["首次提出时间线控制作为面部动作生成的精确时序控制信号", "基于TICC的高效帧级面部动作时间区间标注方法", "Base-Branch扩散架构平衡面部运动解耦精度与自然耦合"]
benchmarks: ["RealTalk", "TAS (Timeline Alignment Score)", "Macro-F1", "FID_fm", "FID_Δfm"]
---

# 论文速读：Exploring-Timeline-Control-for-Facial-Motion-Generation

## 一句话总结
本文首次提出**时间线控制（Timeline Control）**作为面部动作生成的新控制信号，通过TICC时间序列聚类高效标注面部动作的帧级时间区间，并设计了基于扩散模型的Base-Branch架构，实现与自然语言时间对齐的精确、自然面部动作生成。

## 研究问题与动机
1. **现有控制信号缺乏精确时序控制**：音频驱动方法只能生成与音频同步的动作，文本驱动方法仅能使用"then"等粗略时间副词，无法指定具体帧范围（如第10-30帧抬眉、第14-43帧微笑同时发生）。
2. **现有标注方法难以获取帧级精度**：依赖ChatGPT总结时间序列的方式对面部动作的时间动态敏感性不足；阈值方法难以处理复杂动作（如眉毛运动），且某些动作由多个描述符的相对值决定。
3. **面部动作耦合难以兼顾精度与自然性**：分离各区域生成可提升精度但会丢失自然耦合（如抬眉时的轻微头部抬起），需设计平衡机制。

## 核心贡献（创新点）
1. **首次提出劳动高效的面部动作时间区间标注方法**：利用TICC（Toeplitz Inverse Covariance-based Clustering）时间序列分析实现帧级标注，与依赖ChatGPT或手动阈值的方法本质不同。
2. **首次实现时间线控制的面部动作生成**：支持用户在多轨道时间线上精确指定每个动作的发生时段，相比AgentAvatar/InstructAvatar等文本驱动方法，时序控制粒度从粗粒度副词提升到帧级。
3. **设计了Base-Branch扩散生成网络架构**：Base网络编码全局动作耦合信息，Branch网络分别生成各面部区域运动，本质区别在于选择性解耦——既保证精度又保留必要自然耦合。
4. **首次实现ChatGPT辅助的文本到时间线转换**：支持自然语言引导的面部动作生成，用户可进一步编辑时间线实现精细控制。

## 方法详解

### 3.1 面部动作时间区间标注
**Motion Descriptor提取**：
- 使用自研blendshape检测器（因MediaPipe无法检测eyeWide、mouthFrown等），选取ARKit blendshapes构建各区域时间序列
- 眼部：eyeBlink, eyeSquint, eyeWide；眉毛：browDown, browInnerUp, browOuterUp；嘴部：mouthSmile, mouthStretch, mouthFrown
- 注视与头部姿态使用FaceVerse 3DMM系数

**TICC时间序列分析流程**：
- 将所有短视频拼接为长序列，用长度为100的"null sequence"（值为-1）分隔不同视频
- TICC同时完成分段（segmentation）和聚类（clustering），每段包含单一运动模式
- 人工检查每个簇的少量代表性样本，确定该簇代表的动作类别
- 眼部闭合、注视、头部姿态采用阈值法辅助标注

**优化超参**：最优β=5，眼区簇数=8，眉毛/嘴部簇数=9

### 3.2 基于时间线的生成模型
**问题形式化**：给定时间线控制$C = [c_i]_{i=1}^{L}$（每帧16维二进制向量），生成与自然对齐的面部运动$M = [m_l]_{l=1}^{L}$

**扩散模型损失**：
$$\mathcal{L}_{\text{denoise}} = \mathbb{E}_{t \sim \mathcal{U}[1,T], M_{(0)}, C}(\|M_{(0)} - \mathcal{G}(M_{(t)}, t, C)\|^2)$$

**Base-Branch架构设计**：
- 将面部分为三组：上脸（眼+眉+注视）、下脸（嘴+颌）、姿态及其他
- Base网络接收所有区域时间线+噪声运动，编码全局耦合信息
- 各Branch网络接收Base特征+对应区域时间线，生成解耦运动
- 因头部姿态与所有区域耦合，姿态Branch接收全部时间线

**Timeline Token机制**：
- 每帧时间线经线性投影为16维token
- 在多层Transformer Encoder中，timeline token始终使用初始token（而非前一层输出），防止时间信息被扭曲
- Motion token逐层传递，通过cross-attention学习时间控制

**Classifier-Free Guidance训练策略**：
- 每区域条件独立dropout概率0.5
- 全删概率0.1，全保留概率0.1
- 删除时条件设为-1

## 实验与结果

**数据集**：RealTalk数据集（692段真实对话视频，使用1412个视频片段约60万帧）

**标注精度**（Macro-F1）：
- 眉毛：0.90，眼部：0.91，嘴部：0.87
- 阈值法：眼部闭合0.95，姿态0.87，注视0.89
- 对比AU描述符（Macro-F1仅0.73），blendshape精度优势显著

**生成结果**：
- Timeline Alignment Score（TAS）：**0.81**
- Var=0.70（接近GT的0.73），FID_fm=12.58，FID_Δfm=0.14

**消融实验关键发现**：
- 移除Branch网络：TAS降至0.66，证明分支生成对精度的必要性
- 移除Base网络：无法生成自然耦合的细微动作（如头部微妙运动）
- 完全解耦（all decoup.）：动作僵硬且TAS仅0.69
- 保留初始timeline token每层：TAS提升0.05（0.76→0.81）
- 最优Branch层数=2，dropout概率=0.5

**用户研究**（21名参与者）：
- 92%认为标注准确
- 89%认为生成运动准确，86%认为自然

**文本引导生成**：ChatGPT可将自然语言（如"回忆往事时转头、视线转移、随后微笑"）合理转换为时间线。

## 相关工作脉络
1. **AgentAvatar [44] / InstructAvatar [49]**：文本驱动方法，仅能用时间副词粗略控制时序，本文提供帧级精确控制。
2. **Facial Action spotting方法 [9,17,48,57,59]**：仅标注macro/micro-expression区间，而非用于生成的面部动作时间线。
3. **人类运动时间线控制 [2,3,35,39]**：采用分段拼接策略，但因面部运动变化快而不适用，本文设计了专用生成架构。
4. **AutoPlait [30] / CubeMarker [22]**：时间序列分析方法，在本任务中聚类内运动不一致，效果不如TICC。
5. **Audio-driven方法 [43,54] / 3D方法 [1,7,8,11,12,24,25,34,41,52,55,65]**：仅能生成与音频同步的动作，无法独立控制时序。
6. **Rule-based方法 [5,33]**：可提供时序控制但动作不自然，本文在自然性基础上实现精确控制。

## 局限性与未来方向
1. **子类别细粒度不足**：当前将同一动作的不同变体（如大幅抬眉vs中度抬眉）合并为一类，未来可分别生成各子类别。
2. **复杂纹理相关运动未覆盖**：当前使用FaceVerse 3DMM，额头皱纹等复杂纹理运动需引入额外描述符。
3. **渲染器非本文贡献**：photorealistic渲染细节见Supplementary，可能影响最终视觉质量上限。
4. **ChatGPT生成时间线的质量依赖提示工程**：few-shot示例需人工标注，泛化到复杂描述存在局限。

## 研究启发与可借鉴点
1. **TICC时间序列聚类方法可迁移**：该标注思路可应用于其他时序动作标注任务（如手势、舞蹈动作），减少人工标注成本。
2. **Base-Branch耦合-解耦平衡设计**：选择性解耦（精度）+保留必要耦合（自然性）的架构思想可推广至多模态生成任务（如音频-视频联合生成、人体-手势联合生成）。
3. **逐层保留初始条件token机制**：防止信息在深层网络中被扭曲的策略，可应用于其他需要精确时序控制的条件生成任务。
4. **ChatGPTfew-shot转换+人工编辑管线**：结合大模型自动生成与人工精修的多模态转换管线设计，值得借鉴。
5. **与团队方向的结合机会**：若团队研究方向涉及可控视频生成/数字人，本方法的timeline control信号可与音频/文本控制结合，实现多维度精细化控制。

## 关键术语表
**Timeline Control（时间线控制）**：以多轨道时间区间为输入，精确指定各面部动作发生时段的控制信号。
**TICC（Toeplitz Inverse Covariance-based Clustering）**：基于Toeplitz逆协方差的时间序列聚类方法，可同时完成分段与聚类。
**FaceVerse 3DMM**：细粒度3D人脸可变形模型，其expression系数与ARKit blendshapes对齐。
**Base-Branch Architecture**：基础网络编码全局耦合+分支网络生成局部区域的分层生成架构。
**Macro-F1**：多分类任务的宏平均F1分数，各班级F1的算术平均。
**TAS（Timeline Alignment Score）**：时间线对齐得分，通过比较生成视频动作区间与输入时间线计算。
**Classifier-free Guidance**：训练时随机丢弃条件的扩散模型加速推理技术。
**Blendshape**：ARKit面部动作描述符，每个系数表示特定面部区域的运动强度（0-1）。

## 可复现要素
- **数据集**：RealTalk [13]，公开可用；使用1412个视频片段约60万帧
- **代码/权重**：论文未声明开源
- **关键超参**：β=5，眼区簇数=8，眉毛/嘴部簇数=9，Branch层数=2，总层数=8，dropout概率=0.5，全删概率=0.1，全保留概率=0.1
- **Motion描述符**：自研blendshape检测器（非公开），Left-side ARKit blendshapes
- **Renderer**：详见Supplementary Material
