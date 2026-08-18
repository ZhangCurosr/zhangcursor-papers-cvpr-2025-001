---
title: "Q-Bench-Video-Benchmark-the-Video-Quality-Understanding-of-L"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_Q-Bench-Video_Benchmark_the_Video_Quality_Understanding_of_LMMs_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 07:47:13"
field: "视频质量评估与多模态理解"
keywords: ["视频质量评估", "大规模多模态模型", "基准测试", "AIGC质量", "成对比较"]
innovations: ["首个专门评估LMMs视频质量理解能力的综合基准", "引入AIGC失真维度与视频对比较任务", "构建覆盖四级难度题型与四类质量维度的系统化评测框架"]
benchmarks: ["Q-Bench-Video"]
---

# 论文速读：Q-Bench-Video: Benchmark the Video Quality Understanding of LMMs

## 一句话总结
本文提出了 Q-Bench-Video，首个专门用于系统评估大规模多模态模型（LMMs）视频质量理解能力的综合基准，涵盖自然、AIGC、CG 三类视频源，以及四种质量维度（Technical、Aesthetic、Temporal、AIGC）和三种题型，测试了 17 个 LMMs 后发现当前模型在视频质量理解上显著落后于人类水平，尤其在开放式问题与 AIGC 失真识别方面存在明显短板。

## 研究问题与动机
1. **现有视频 LMM 基准聚焦语义理解，忽视质量感知**：尽管已有 Video-MME、MMBench-Video 等全面评估 LMMs 高级语义理解能力的基准，但缺乏系统性地针对视频低/中层质量理解能力的评测框架。
2. **视频质量理解在工业应用中至关重要**：在视频压缩优化、传输系统调优、高质量视频生成标准建立等实际场景中，准确理解视频质量是核心需求。
3. **AIGC 视频质量评估亟待标准化**：随着 AI 生成视频的普及，现有基准未涵盖 AIGC 特有的失真类型（如不自然纹理、人脸畸变等），导致对生成内容的质量评估能力缺失。
4. **成对视频质量比较能力未被充分评估**：在实际视频压缩性能测试和质量控制流程中，成对比较（pairwise comparison）比单视频评分更为常见，但现有基准缺乏此类任务设计。

## 核心贡献（创新点）
1. **提出首个专门针对视频质量理解的 LMM 基准 Q-Bench-Video**：与以往视频基准聚焦高层语义不同，本文首次系统性地将评测焦点转向低/中层视觉质量感知，构建了涵盖 1800 个视频、2378 个 QA 对的完整评测框架。
2. **引入 AIGC 失真维度作为第四大质量评估类别**：在传统 Technical、Aesthetic、Temporal 三维度的基础上，新增 AIGC 维度，填补了对 AI 生成内容特有失真（如 unnatural textures、uncanny faces）的评测空白，回应了视频生成领域快速发展带来的评估需求。
3. **设计视频对质量比较任务（Video Pairs Comparison）**：区别于现有基准仅评估单视频质量，本文引入成对比较任务，细分为 Joint Analysis 和 Comparative Analysis（粗粒度/细粒度），更贴近视频压缩测试与生成质量控制的实际应用场景。
4. **构建题型梯度与答案分布均衡机制**：提出 Yes-or-No、What-How、Open-ended 三级难度题型，并对 Yes-or-No 题进行 50:50 正负样本平衡标注，有效抑制 LMMs 的 yes-bias 现象，提升评测可靠性。

## 方法详解
1. **视频源采样策略**：从 LSVQ、MaxWell、WaterlooSQoE-III/IV、T2VQA-DB、VideoFeedback、LIVE-YT-Gaming 等 7 个公开视频质量数据库采样，采用基于质量标签的均匀采样（uniform sampling）确保质量分布均衡，最终选取 1000 个自然场景视频、600 个 AIGC 视频、200 个 CG 视频。
2. **三元组数据模板**：每个数据项采用元结构 tuple $(V, Q, A, C)$ 表示，其中 $V$ 可为单视频或视频对，$Q$ 为质量问题，$A$ 为候选答案集，$C$ 为正确答案。
3. **题型设计**：
   - **Yes-or-No**：二元判断，通过标注调整确保 Yes/No 比例约 50%:50%，缓解 yes-bias。
   - **What-How**：What 类识别特定失真（如"What is the most apparent distortion?"），How 类评估失真程度（如"How is the overall clarity?"）。
   - **Open-ended**：无预定义答案，要求模型自由描述质量因素及原因，如"What are the possible factors that lead to the low clarity?"
4. **质量维度分类**：
   - **Technical**：低层退化，如模糊、噪声、压缩伪影、曝光异常等。
   - **Aesthetic**：主观审美偏差，如色彩混乱、构图不佳、光线不一致等。
   - **Temporal**：时序退化，如画面抖动、闪烁、运动不一致、帧率不稳等。
   - **AIGC**：生成特有缺陷，如不自然纹理、人脸畸形、光照不一致等。
5. **单视频 vs. 视频对任务**：
   - **单视频**：分为 Global perception（整体质量）与 Referring perception（局部/特定元素质量）。
   - **视频对**：分为 Joint analysis（分析两视频共有质量特征）与 Comparative analysis（比较两视频间质量差异），后者进一步按质量标签差异分为 coarse-grain 与 fine-grain 两类。
6. **标注流程**：8 名经过培训的专家在实验室环境下完成标注，每对 QA 至少经 3 名专家交叉审核；视频对标注在两视频间插入 5 秒灰屏间隔以避免混淆。
7. **评测设置**：视频 LMMs 均匀采样 16 帧，图像 LMMs 采样 8 帧；无法适配时采用官方默认设置；Open-ended 及非选项回答采用 GPT-assisted evaluation 策略，依据准确性、完整性、相关性进行打分。

## 实验与结果
1. **测试模型**：共评测 17 个 LMMs，包括 3 个开源图像 LMM（LLaVA-Next、LLaVA-v1.5、mPLUG-Owl2）、9 个开源视频 LMM（mPLUG-Owl3、LLaVA-OneVision、InternVL-Chat、VILA1.5、PLLaVA、LLaVA-Next-Video、ST-LLM、Video-LLaVA、VideoChat2）、5 个专有 LMM（Gemini 1.5 Flash/Pro、GPT-4o mini、GPT-4o、GPT-4 Turbo）。
2. **数据集划分**：测试集 1186 条 QA，开发集 1192 条 QA，开发集答案保密。
3. **整体性能排名**：
   - 人类平均分：**81.56%**
   - 最优专有 LMM（GPT-4o）：**58.70%**，与人类差距达 **22.86%**
   - 最优开源视频 LMM（mPLUG-Owl3）：**52.39%**
   - 随机猜测基线（不含 Open-ended）：**37.79%**
4. **题型难度梯度**：Yes-or-No > What-How > Open-ended，LMMs 在 Open-ended 上性能显著下滑（人类仅下降约 9.46%，而 LMMs 降幅更陡峭），表明其在开放式质量问题推理方面能力不足。
5. **质量维度表现差异**：
   - 人类在 **AIGC 失真识别** 上表现最强（**86.21%**），LMMs 在此维度表现最弱（GPT-4o 仅 60.90%，开源模型普遍低于 54%）。
   - LMMs 在 **Aesthetic 维度** 相对优势明显（GPT-4o 达 66.23%），与其预训练中的高语义上下文对齐。
   - Technical 与 Temporal 维度 LMMs 表现相对稳定但仍有较大提升空间。
6. **单视频 vs. 视频对**：
   - 人类在视频对比较任务上达到 **87.56%**（Compare-coarse 高达 **89.11%**）。
   - LMMs 在视频对比较任务上显著优于单视频分析，尤其是粗粒度比较（GPT-4o 达 69.24%），说明成对对比为模型提供了更清晰的判别线索。
   - 细粒度比较（fine-grain）是当前 LMMs 的主要瓶颈。
7. **最强结果总结**：GPT-4o 以 **58.70%** 的整体准确率位居第一，但在所有子维度上仍显著落后于人类（最大差距在 AIGC 维度约 25 个百分点）；mPLUG-Owl3 以 **52.39%** 成为开源模型冠军，略超 GPT-4o mini（52.20%）。

## 相关工作脉络
1. **Video-MME / MMBench-Video**：现有综合性视频理解基准，聚焦高层语义任务（如动作识别、事件理解、时空推理），与本文在评测目标上形成互补——前者评估"视频在讲什么"，本文评估"视频质量如何"。
2. **FAST-VQA / COVER / VQA²**：传统深度学习视频质量评估方法，依赖手工/端到端回归预测 MOS 分数，缺乏可解释性与多维度质量分解能力；本文以 LMM 问答形式替代分数回归，提供更细粒度的质量诊断。
3. **A-Bench / Q-Bench（图像版）**：前者评估 LMMs 对 AI 生成图像的识别能力，后者评估 LMMs 在低层视觉任务（含图像质量）上的理解；本文将其拓展至视频域，并新增 AIGC 视频与时间维度。
4. **LMM-VQA / Light-VQA+**：近期将 LMM 引入 VQA 任务的前沿工作，但主要聚焦单帧或短时片段的质量打分，缺乏对 AIGC 视频、时序失真及成对比较的系统性评测设计。
5. **Ntire 2024 AIGC Challenge**：聚焦 AI 生成内容的客观质量评估算法竞赛，与本文的 LMM 主观质量理解视角不同，本文可作为补充验证 LMMs 在此任务上的潜力与局限。
6. **2AFC Prompting (Zhu et al.)**：针对图像质量评估的成对比较提示方法，本文将其思路扩展至视频域，并细化为 Joint Analysis 与 Comparative Analysis 子任务。

## 局限性与未来方向
1. **审美标注的主观性难以完全消除**：Aesthetic 维度依赖专家主观判断，不同 annotator 间可能存在评价标准偏差，影响基准的客观一致性。
2. **AIGC 模型快速演进导致基准老化风险**：随着视频生成模型持续迭代，新的失真模式可能不断涌现，现有 QA 覆盖范围可能逐渐过时，需定期更新扩充。
3. **视频对比较仅在同类源内进行**：Natural-AIGC、AIGC-CG 等跨域成对比较未被纳入，限制了基准在混合内容场景下的适用性。
4. **Open-ended 评估依赖 GPT 打分**：自动化评分虽提升了效率，但 GPT 的评分标准可能与人类专家存在偏差，需引入更精细的评估协议。
5. **未来方向**：可探索针对 AIGC 视频失真的专项训练数据构建、时序质量理解模型的针对性优化，以及将细粒度成对比较纳入后续基准迭代。

## 研究启发与可借鉴点
1. **题型设计中的 yes-bias 抑制机制**：通过调整标注比例实现 Yes/No 均衡分布的做法，可迁移至其他 LMM 评测基准中以提升评测可靠性。
2. **质量维度扩展 AIGC 类别的设计思路**：在通用视觉基准中引入"生成内容特有失真"维度，为后续 AIGC 评估基准构建提供了可复用的分类框架。
3. **成对比较任务的双层细分（coarse/fine）**：将比较任务按质量差异程度进一步划分，有助于更精准地定位模型能力边界，可在图像质量、文档质量等对比类任务中借鉴。
4. **GPT-assisted evaluation 策略**：对于无法直接量化的开放问答，采用 GPT 辅助评分的方法值得参考，但需警惕评分器自身偏差，建议结合人工校验。
5. **开源与专有模型并列评测**：同时涵盖 12 个开源与 5 个专有模型，并对比 Image LMM vs. Video LMM 的表现差异，为后续研究提供了清晰的基线参照与模型选型参考。

## 关键术语表
- **Q-Bench-Video**：首个系统性评估 LMMs 视频质量理解能力的基准，包含 1800 视频、2378 QA 对，覆盖四种质量维度与成对比较任务。
- **LMM (Large Multi-modal Model)**：大型多模态模型，指能够处理文本、图像、视频等多种模态输入的大型预训练模型，如 GPT-4o、LLaVA 系列等。
- **AIGC (AI-Generated Content)**：人工智能生成内容，本文特指由生成模型（如 Sora、Runway 等）合成的视频，其失真特征与传统采集压缩失真存在本质差异。
- **Technical Distortion**：技术失真，指由录制、编码压缩、传输等环节引入的低层视觉退化，如模糊、噪声、块效应等。
- **Aesthetic Distortion**：审美失真，指偏离预期视觉风格或艺术创作意图的主观质量缺陷，如色彩失衡、构图混乱等。
- **Temporal Distortion**：时序失真，指视频在时间维度上的质量退化，表现为抖动、闪烁、运动不连续、帧率不稳等。
- **Yes-or-No Question**：二元判断题型，要求模型对质量属性做出是/否判断，本文通过标注平衡抑制 yes-bias。
- **Video Pairs Comparison**：视频对比较任务，要求模型分析两个视频间的质量共性与差异，细分为粗粒度与细粒度比较。

## 可复现要素
- **数据集**：Q-Bench-Video 基于 LSVQ、MaxWell、WaterlooSQoE-III/IV、T2VQA-DB、VideoFeedback、LIVE-YT-Gaming 等公开数据库采样构建，视频来源可获取；QA 标注数据（开发集/测试集）论文中声明 dev 集答案公开、test 集答案保密，需向作者申请。
- **代码/权重**：项目主页已公布，链接为 https://github.com/Q-Future/Q-Bench-Video；具体评测代码与测试脚本需从该仓库获取。
- **关键超参**：视频 LMMs 统一采样 16 帧，图像 LMMs 采样 8 帧；不兼容时采用各模型官方默认设置；Open-ended 评分采用 GPT-assisted 策略（论文未详述具体 prompt 模板与温度参数，需查阅附录或代码）。
