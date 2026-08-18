---
title: "PICO-Reconstructing-3D-People-In-Contact-with-Objects"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Cseke_PICO_Reconstructing_3D_People_In_Contact_with_Objects_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 07:46:22"
field: "3D人体物体交互重建"
keywords: ["3D Human-Object Interaction", "Contact Reconstruction", "Single-image 3D Recovery", "Open-world 3D Objects", "Dataset Construction"]
innovations: ["构建首个包含人体-物体双侧3D接触对应关系的自然图像数据集PICO-db", "提出基于接触轴的快速接触转移标注方法，仅需两点击即可投影接触", "开发三阶段接触引导优化方法PICO-fit，支持开放世界物体类别的重建"]
benchmarks: ["InterCap", "DAMON", "OBJaverse-LVIS"]
---

# 论文速读：PICO-Reconstructing-3D-People-In-Contact-with-Objects

## 一句话总结
本文提出PICO框架，通过构建包含人体-物体3D接触对应关系的新数据集PICO-db，开发了一种基于渲染与比较的优化方法PICO-fit，能够从单张自然图像中同时重建3D人体姿态形状、3D物体姿态形状及其密集接触对应关系，支持任意物体类别且无需先验已知物体类型或形状。

## 研究问题与动机
1. **深度歧义与遮挡挑战**：从单视图恢复3D人机交互面临严重深度歧义和相互遮挡问题，现有方法多依赖受控场景（已知物体形状或接触），难以推广到真实自然图像。
2. **物体形状重建困难**：不同于人体有统一统计模型（SMPL），物体形状高度多样且无单一参数化模型，导致从单图鲁棒恢复3D物体形状极具挑战。
3. **接触估计局限**：现有接触推断方法要么仅在2D推断接触，要么只在人体表面推断3D接触而忽略物体，或仅在合成数据上训练泛化能力差。
4. **数据集匮乏**：缺乏同时在人体和物体上标注3D接触对应关系的大规模真实图像数据集，限制了3D HOI方法的发展。

## 核心贡献（创新点）
1. **首个自然图像-人体-物体3D接触对应数据集**：构建PICO-db数据集，首次将自然图像与人体及物体表面的密集3D接触标注配对，建立双射的人体-物体对应关系。
2. **高效接触转移标注方法**：基于PCA自动生成接触轴并通过仅需两点击的方式将DAMON的人体接触投影到物体网格，大幅降低众包标注成本。
3. **开放世界物体检索与3D重建**：利用OpenShape基础模型构建图像-3D形状联合隐空间，通过最近邻检索获取匹配物体网格，支持未见过的物体类别。
4. **三阶段接触引导优化方法**：提出PICO-fit，通过接触对应关系初始化物体位姿，再分阶段优化物体和人体的3D参数，有效解决深度歧义和穿透问题。

## 方法详解
**PICO-db构建**：
- **物体形状检索**：使用OpenShape模型将Objaverse-LVIS数据库中的3D网格嵌入联合隐空间，测试时将图像嵌入并检索余弦相似度最高的3个最近邻网格，由标注者选择最佳匹配。
- **接触轴自动生成**：对每个身体接触补丁执行PCA，取第一主成分方向作为接触轴，从均值出发沿主成分方向采样起始/终止点并投影到身体表面，通过测地线追踪生成轴。
- **手指区域处理**：对细粒度非凸区域（如手指），通过计算凸包创建带蹼的代理SMPL网格，避免轴拉伸变形。
- **众包标注流程**：每点击两个点定义接触轴起点和方向，工具实时显示投影结果并允许覆盖修正。

**PICO-fit方法**：
- **初始化阶段**：用OSXY回归器推断SMPL-X身体网格；用OpenShape检索物体网格并用GPT-4V估计初始尺度；用DECO推断3D人体接触，通过GPT-4V减少假阳性，查询PICO-db获取最近邻接触及对应物体信息。
- **阶段1-物体注册到人体**：固定人体，最小化接触损失 $\mathcal{L}_c = \frac{1}{|\mathbb{S}|} \sum_{(v_i, p_i) \in \mathbb{S}} \|v_i - p_i\|_2$ 优化物体旋转$R_o$和平移$t_o$。
- **阶段2-物体对齐到图像**：通过PyTorch3D渲染物体掩码，与SAM检测的掩码计算IoU损失$\mathcal{L}_o^m = 1 - \text{IoU}(M_o, \bar{M}_o)$；引入人体SDF计算穿透损失$\mathcal{L}_p = \sum_{v_i \in \mathcal{O}} \Omega_h(v_i)$；结合尺度损失$\mathcal{L}_o^s = \|s_o - s_o^*\|_2$，总损失$L_2 = \lambda_c \mathcal{L}_c + \lambda_p \mathcal{L}_p + \lambda_o^m \mathcal{L}_o^m + \lambda_o^s \mathcal{L}_o^s$。
- **阶段3-人体姿态细化**：仅优化躯干后到接触关节的链式参数$\theta_\mathcal{C}$，结合接触损失、穿透损失、人体掩码损失$\mathcal{L}_h^m$和姿态正则$\mathcal{L}_r = \|\theta - \theta^*\|_2$，总损失$L_3 = \lambda_c \mathcal{L}_c + \lambda_p \mathcal{L}_p + \lambda_h^m \mathcal{L}_h^m + \lambda_{\theta_c} \mathcal{L}_{\theta_c}$。

## 实验与结果
**数据集与评估**：
- 在InterCap（10类物体）和DAMON（42类物体的75张未标注图像）上评估，使用Procrustes-Aligned Chamfer Distance (PA-CD)指标。
- 通过Amazon MTurk进行感知研究，参与者判断哪种重建更真实。

**主要结果**：
- **InterCap OOD泛化**：PICO-fit（无GT接触）PA-CD$_h$达7.43cm，优于CONTHO (8.36cm)和PHOSA (10.07cm)；使用GT接触时PICO-fit*达到6.66cm，显著优于所有基线。
- **感知评估**：PICO-fit*在DAMON图像上被偏好率为62.7%-79.9%，平均比基线高74.4%的时间被认为更真实。
- **新物体类别**：成功处理沙发、香蕉、飞盘等此前方法无法处理的物体类别。
- **消融实验**：三个阶段贡献显著，完整方法PA-CD$_{h+o}$为8.36cm，相比仅阶段1+2提升0.04cm（Table 2）。

## 相关工作脉络
1. **DECO/ContactDB/ObMan**：这些工作关注接触估计但多在实验室场景或合成数据，缺乏真实自然图像的3D人体-物体接触对应关系；本文扩展至野外图像并提供双向接触标注。
2. **PHOSA**：作为最接近的基线，PHOSA使用手工类别接触约束和单尺度物体，本文通过数据库检索实现实例级尺度估计并泛化到更多物体类别。
3. **CONTHO/HDM**：回归类方法在训练分布内表现良好但在OOD场景失败；本文基于优化的方法通过接触约束实现更好的泛化。
4. **DAMON**：仅提供人体侧的3D接触标注；本文通过接触转移方法扩展为人体-物体双侧接触标注。
5. **OpenShape/Objaverse**：基础模型支持开放词汇3D理解；本文首次将其应用于HOI场景的物体检索与重建。
6. **BEHAVE/InterCap**：受控场景数据集；本文方法在InterCap上验证OOD泛化能力。

## 局限性与未来方向
1. **接触推断依赖DECO**：当前使用DECO推断人体接触存在噪声，需更鲁棒的接触估计器。
2. **检索依赖数据库规模**：物体检索性能受限于Objaverse数据库的覆盖范围和OpenShape的特征质量。
3. **手指等精细区域处理**：尽管采用凸包代理网格，手指等复杂区域的接触转移仍存在挑战。
4. **未来方向**：计划基于PICO-db训练直接接触回归器替代最近邻检索；利用Vision-Language模型扩展泛化能力；生成伪GT标签扩展训练数据。

## 研究启发与可借鉴点
1. **接触轴自动生成的设计**：通过PCA生成接触轴并将两点击转化为完整接触转移，为低成本众包3D标注提供新思路。
2. **三阶段分离优化策略**：将物体-人体联合优化分解为顺序阶段（先物体后人体），避免联合优化的鸡生蛋问题，值得类似任务借鉴。
3. **SDF穿透损失的应用**：使用完整SDF而非浅层穿透惩罚，能更好处理严重遮挡情况。
4. **基础模型+数据库检索的轻量化方案**：用OpenShape检索替代端到端物体重建，平衡了质量、速度与泛化性。

## 关键术语表
**PICO-db**：包含自然图像与人体及物体表面密集3D接触对应关系的数据集，建立双射的人体-物体对应。
**OSXY**：用于从单图回归SMPL-X身体网格的姿态形状估计器。
**OpenShape**：构建图像与3D形状联合隐空间的 Foundation Model，支持跨模态检索。
**DECO**：从图像推断3D人体接触的区域模型。
**PA-CD**：Procrustes-Aligned Chamfer Distance，用于评估3D重建质量的指标。
**Contact Axis**：通过PCA生成的接触补丁代表轴，用于接触转移到其他网格。
**Objaverse-LVIS**：大规模3D物体数据库，包含LVIS标注的物体类别。
**SDF**：Signed Distance Field，用于计算穿透损失的符号距离场表示。

## 可复现要素
- **数据集**：PICO-db（4123张图像，44类物体，627个实例）已公开；DAMON数据集可获取。
- **代码**：论文声明代码和数据可在 https://pico.is.tue.mpg.de 获取。
- **关键超参**：优化权重λ值未在正文详细列出，需在补充材料或代码中查找；物体尺度初始化使用GPT-4V。
