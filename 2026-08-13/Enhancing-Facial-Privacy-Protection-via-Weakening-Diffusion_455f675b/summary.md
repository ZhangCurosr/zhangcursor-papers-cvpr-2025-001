---
title: "Enhancing-Facial-Privacy-Protection-via-Weakening-Diffusion"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Salar_Enhancing_Facial_Privacy_Protection_via_Weakening_Diffusion_Purification_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 07:02:15"
field: "人脸识别安全与隐私保护"
keywords: ["人脸隐私保护", "扩散模型", "对抗攻击", "null-text引导", "扩散净化", "潜在扩散模型", "自注意力"]
innovations: ["学习无条件嵌入作为null-text引导以弱化扩散净化效应", "利用自注意力图对齐保持原始与生成图像的结构一致性"]
benchmarks: ["CelebA-HQ", "LADN"]
---

# 论文速读：Enhancing-Facial-Privacy-Protection-via-Weakening-Diffusion

## 一句话总结
本文提出了一种基于潜在扩散模型（LDM）的人脸隐私保护新方法，通过学习无条件嵌入（unconditional embeddings）作为null-text引导来弱化扩散净化效应，并在LDM潜在空间中直接优化对抗潜在代码，同时利用自注意力图保持结构一致性，显著提升了保护成功率（PSR）和生成图像的视觉质量。

## 研究问题与动机
1. **扩散净化效应制约保护性能**：现有扩散方法（如DiffProtect）在反向扩散过程中，对抗性修改会被模型当作高频噪声逐步去除，导致保护成功率（PSR）低下。
2. **语义潜在代码修改损害结构一致性**：DiffProtect等方法修改语义潜在代码（semantic latent code）使生成图像形状结构偏向目标人脸，导致视觉失真。
3. **化妆类方法依赖参考图像且泛化性差**：AMT-GAN、DiffAM等方法需要额外参考图像提取化妆信息，且每次更换目标身份需重新训练全模型，实用性受限。
4. **文本引导化妆方法控制精度不足**：Clip2Protect使用文本提示添加化妆，但无法精确控制化妆位置与颜色，生成效果与预期存在偏差。

## 核心贡献（创新点）
1. **学习无条件嵌入弱化扩散净化效应**：在反向扩散过程中学习per-timestamp无条件嵌入作为null-text引导，赋予模型额外学习容量以保留精细纹理和对抗修改，与DiffProtect直接修改语义潜在代码形成本质区别。
2. **自注意力图引导保持结构一致性**：通过对齐原始图像与对抗潜在代码的自注意力图（self-attention maps）来维持结构完整性，而非像DiffProtect那样约束对抗代码修改幅度（L∞范数），实现了无约束优化与结构保持的兼顾。
3. **合成目标身份解决伦理问题**：使用AI生成的合成目标身份而非真实人物进行obfuscation任务，避免了对真实个体身份的误用风险，与CLIP2Protect使用真实目标身份形成区分。
4. **端到端免参考图像的隐私保护框架**：无需参考图像、无需针对新目标重新训练，直接利用Stable Diffusion的生成能力实现transferable的对抗性人脸生成。

## 方法详解
1. **框架概述**：采用两阶段学习策略。第一阶段学习per-timestamp无条件嵌入$\{\mathcal{O}_i\}_{i=1}^t$；第二阶段冻结嵌入并优化对抗潜在代码$z_{adv}$。
2. **DDIM反演（Latent Code Initialization）**：将输入图像$x$通过DDIM inversion过程映射到时间戳$t$的噪声表示$z_t$，使用无条件嵌入$\mathcal{O}$替代文本条件，公式：$z_{t+1} = \sqrt{\bar{\alpha}_{t+1}/\bar{\alpha}_t} z_t + \sqrt{\bar{\alpha}_{t+1}}(\sqrt{1/\bar{\alpha}_{t+1}-1} - \sqrt{1/\bar{\alpha}_t-1})\epsilon_\theta(z_t, t, \emptyset)$。
3. **无条件嵌入学习**：在时间戳$t$开始，为每个时间戳学习独立的无条件嵌入，通过重构损失$\mathcal{L}_{rec} = \|z_{t-1} - \bar{z}_{t-1}\|_2^2$迫使重建图像贴近原始图像，同时保留对抗修改的容量。
4. **对抗潜在代码学习**：从$z_{adv} = z_t$出发，通过DDIM反向步骤得到$z_0'$并解码为$x^p = \mathcal{D}(z_0')$。对抗损失：$\mathcal{L}_{adv} = \frac{1}{K}\sum_{k=1}^{K}[1 - \cos(\mathcal{F}_k(x^p), \mathcal{F}_k(x^t))]$，其中$K=4$个白盒代理模型。
5. **自注意力结构保持**：计算扰动前$S(\bar{z}_t)$和扰动后$S(z_{adv})$的自注意力图，通过损失$\mathcal{L}_{str} = \|S(z_{adv}) - S(\bar{z}_t)\|_2^2$对齐结构。
6. **总损失函数**：$\mathcal{L} = \lambda_{adv}\mathcal{L}_{adv} + \mathcal{L}_{str}$，其中$\lambda_{adv} = 0.003$。

## 实验与结果
1. **数据集**：CelebA-HQ（1000张，不同身份）和LADN（332张非化妆图像）。
2. **目标模型**：IRSE50、IR152、FaceNet、MobileFace（黑盒设置，轮换使用）。
3. **基线方法**：噪声类（PGD、MI-FGSM、TI-DIM、TIP-IM）、化妆类（Adv-Makeup、AMT-GAN、CLIP2Protect、DiffAM）、扩散类（DiffProtect）。
4. **主要结果**：
   - CelebA-HQ平均PSR：ours 79.17% vs DiffAM 77.88%（提升1.29%）vs TIP-IM 50.06%（提升58.8%）
   - LADN平均PSR：ours 79.17% vs DiffAM 77.88%
   - FID：ours 15.3212显著优于DiffAM 26.1015和DiffProtect 28.2912
   - PSNR：ours 27.7223优于DiffAM 20.5260
5. **消融实验**：移除null-text引导或自注意力引导均导致PSR下降和FID恶化，验证了两模块的有效性。
6. **合成目标身份实验**：在fake target impersonation任务上，ours在IRSE50（90.2% vs 83.4%）、IR152（88.6% vs 83.6%）、Mobileface（85.0% vs 62.8%）上均优于CLIP2Protect。

## 相关工作脉络
1. **DiffProtect（2023）**：首个基于扩散模型的人脸隐私保护方法，通过修改Diff-AE的语义潜在代码生成对抗图像，但未考虑扩散净化效应，且修改语义代码会改变面部结构。本文直接修改LDM潜在代码并使用无条件嵌入弱化净化效应，从根本上解决了该问题。
2. **DiffAM（2024）**：当前化妆类SOTA，使用两个扩散模型分别进行化妆去除和对抗化妆转移，但需要参考图像且需针对新目标重新训练。本文方法无需参考图像、无需重新训练，且在PSR上略有超越。
3. **CLIP2Protect（2023）**：基于CLIP引导的文本提示化妆方法，但精细控制不足。本文采用null-text引导而非文本提示，实现了对抗修改的更精确控制。
4. **TIP-IM（2021）**：噪声类方法的代表，通过学习噪声掩码实现对抗修改，但产生可感知噪声影响图像质量。本文在像素级质量（PSNR）和感知质量（FID）上均显著优于TIP-IM。
5. **Diffusion Purification研究（Nie et al., 2022）**：揭示了扩散模型的反向过程会去除对抗噪声的"净化效应"。本文首次将此效应纳入人脸隐私保护框架，并通过无条件嵌入主动弱化净化。
6. **Null-text Inversion（Mokady et al., 2023）**：提出了通过优化无条件嵌入实现图像编辑的方法。本文借鉴该思想，但将其应用于对抗性隐私保护任务，并引入结构保持机制。

## 局限性与未来方向
1. **计算成本**：需要针对每张输入图像进行无条件嵌入学习（20次迭代）和对抗潜在代码优化（35次迭代），推理速度可能受限。
2. **黑盒评估局限**：仅使用代理模型评估transferability，实际面对未知FR模型时的鲁棒性有待验证。
3. **目标身份选择**：虽然提出了合成目标身份的方案，但合成身份的多样性及分布覆盖仍需进一步探索。
4. **伦理边界**：即使使用合成目标身份，对抗性图像生成技术的滥用风险仍需关注。

## 研究启发与可借鉴点
1. **null-text引导弱化净化效应**的策略可迁移至其他基于扩散模型的对抗攻击或隐私保护任务，如视频脱敏、医疗图像保护等。
2. **自注意力图对齐保持结构一致性**的方法可用于图像编辑、风格迁移等需要保持源图结构的生成任务，作为结构保持的替代方案。
3. **per-timestamp无条件嵌入学习**的思想可推广至其他扩散模型编辑任务，如inpainting、outpainting等，提升生成质量与可控性。
4. **合成目标身份**的伦理解决方案值得借鉴，尤其适用于需要规避真实身份使用的隐私保护应用场景。
5. **两阶段学习策略**（先学习条件再固定条件优化对抗扰动）为扩散模型的对抗性应用提供了可复用的工程范式。

## 关键术语表
- **扩散净化效应（Diffusion Purification）**：扩散模型的反向去噪过程会将对抗扰动视为高频噪声并逐步去除，从而削弱对抗攻击的效果。
- **无条件嵌入（Unconditional Embeddings）**：替代文本条件的可学习嵌入向量，作为null-text引导反向扩散过程，弱化净化效应并保留对抗修改。
- **潜在扩散模型（LDM）**：在压缩的潜在空间中进行扩散过程的模型，通过预训练的autoencoder实现高效高分辨率图像生成。
- **DDIM反演（DDIM Inversion）**：将真实图像逆向映射到扩散过程中的噪声表示，用于后续编辑或对抗性修改。
- **保护成功率（PSR）**：在固定FAR=0.01条件下，人脸保护图像成功欺骗FR模型的比例。
- **自注意力图（Self-attention Maps）**：扩散模型U-Net中捕捉图像内部空间关系的注意力图，用于保持生成图像与源图的结构一致性。
- **对抗潜在代码（Adversarial Latent Code）**：在LDM潜在空间中优化的对抗扰动，用于生成保护图像。

## 可复现要素
- **数据集**：CelebA-HQ（公开）、LADN（公开）
- **代码**：已开源，https://github.com/parham1998/Facial-Privacy-Protection
- **关键超参**：DDIM反转步数T=20，反向过程从t=3开始；无条件嵌入学习：AdamW，lr=0.1，20次迭代；对抗潜在代码优化：AdamW，lr=0.01，35次迭代；$\lambda_{adv}=0.003$；代理模型数K=4
