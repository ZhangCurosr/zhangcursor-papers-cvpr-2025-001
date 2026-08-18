---
title: "OmniGuard-Hybrid-Manipulation-Localization-via-Augmented-Ver"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_OmniGuard_Hybrid_Manipulation_Localization_via_Augmented_Versatile_Deep_Image_Watermarking_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 07:41:42"
field: "数字图像取证与水印"
keywords: ["图像篡改定位", "多功能水印", "AIGC编辑", "主动嵌入", "被动提取", "退化感知"]
innovations: ["主动嵌入+被动盲提取的混合取证框架，解耦编码与解码优化目标", "深度退化感知篡改提取器结合退化查询实现高精度鲁棒定位", "轻量级AIGC编辑模拟替代完整扩散过程，训练提速23倍"]
benchmarks: ["COCO", "UltraEdit", "Stable Diffusion Inpaint", "ControlNet Inpaint", "Instructp2p"]
---

# 论文速读：OmniGuard-Hybrid-Manipulation-Localization-via-Augmented-Ver

## 一句话总结
论文提出了OmniGuard，一种融合主动嵌入与被动盲提取的混合取证框架，通过自适应定位水印变换、深度退化感知篡改提取器和轻量级AIGC编辑模拟器，在不牺牲保真度的前提下实现了高精度的像素级篡改定位与鲁棒的版权保护。

## 研究问题与动机
- **保真度-定位精度权衡困境**：现有多功能水印（如EditGuard）依赖预定义定位水印与重建水印的残差减法生成掩码，为保证定位精度不得不牺牲容器图像质量。
- **定位水印灵活性受限**：由于解码端需已知定位水印才能提取掩码，现有方法只能使用固定水印，无法针对不同图像内容自适应选择。
- **严重退化下鲁棒性不足**：定位水印在JPEG压缩、噪声等强退化下恢复失败，且版权水印容易被全局AIGC编辑算法擦除。
- **被动方法泛化能力弱**：纯被动篡改检测方法难以应对AIGC生成/编辑产生的新型篡改痕迹。

## 核心贡献（创新点）
- **提出主动嵌入+被动盲提取的混合取证框架**：将编码端与解码端解耦，定位水印提取不再依赖先验知识，使水印网络可专注于提升保真度。
- **设计深度退化感知篡改提取器**：融合重建的定位水印伪影图与篡改图像，引入可学习退化查询$Q_{deg}$自适应选择退化类型，在严重退化下仍能高精度定位。
- **提出自适应定位水印变换**：通过可逆前向/逆向仿射变换使定位水印获得内容感知的纹理，显著降低视觉伪影同时提升保真度。
- **设计轻量级AIGC编辑模拟层**：以VAE压缩和部分移除操作替代完整的扩散去噪迭代过程，将训练耗时从Robust-Wide的3.675s/sample降至0.16s/sample。

## 方法详解
**整体框架**：由主动双水印网络（嵌入定位水印$\tilde{W}_{loc}$和版权水印$w_{cop}$生成容器图$I_{con}$）与被动退化感知篡改提取器（从重建伪影图$\hat{W}_{loc}$和接收图$I_{rec}$预测掩码$\hat{M}_{loc}$）组成。

**被动篡改提取器**：将$\hat{W}_{loc}$和$I_{rec}$分别进行patch embedding得到令牌组$T_{loc}$和$T_{rec}$；通过GAP+MLP+Sigmoid提取$I_{rec}$的特征$F_{rec}$，与退化查询$Q_{deg}$进行逐元素乘法并卷积融合，生成增强伪影令牌$\tilde{T}_{loc} = T_{loc} + \beta \cdot \text{Conv}(F_{rec} \odot Q_{deg})$；经Swin-ViT骨干网络与特征金字塔后由掩码解码器输出$\hat{M}_{loc}$。

**自适应定位水印变换**：前向变换$\tilde{W}_{loc} = W_{loc} + \phi_1(I_{ori})$，$\ H_{med} = I_{ori} \otimes \text{Exp}(\phi_2(\tilde{W}_{loc})) + \phi_3(\tilde{W}_{loc})$；逆向变换$\hat{H}_{med} = (I_{rec} - \phi_3(W'_{loc})) \otimes \text{Exp}(-\phi_2(W'_{loc}))$，$\hat{W}_{loc} = W'_{loc} - \phi_1(I_{rec})$，其中$\phi_i$为卷积层。

**轻量级AIGC编辑模拟**：以50%概率对$I_{con}$施加SD-1.5的VAE编码-解码模拟全局编辑，以50%概率施加随机掩码部分移除模拟局部篡改，梯度可直接回传。

**三阶段训练**：①仅训练版权水印网络（$\ell_{cop} = \ell_{bce}(\hat{w}_{cop}, w_{cop}) + \lambda \cdot \ell_2(I_{con}, I_{ori})$，$\lambda$从0.05渐增至27.5）；②冻结版权网络，联合训练双水印网络（$\ell_{loc} = \ell_2(\hat{W}_{loc}, W_{loc}) + 10\cdot\ell_2 + 10\cdot\ell_{GAN} + 100\cdot\ell_{LPIPS}$）；③训练篡改提取器（$\ell_{ext} = \ell_{bce}(\hat{M}_{loc}, M_{gt}) + 20\cdot\ell_{edge}$）。

## 实验与结果
- **数据集**：MIR-Flickr（版权水印训练）、DIV2K（双水印训练）、COCO测试集1000张（定位评估）、UltraEdit 1000张512×512图像（鲁棒性评估）。
- **评测基线**：被动方法MVSS-Net、CAT-Net、PSCC-Net、HiFi-Net、IML-ViT；主动方法EditGuard；鲁棒水印PIMoG、SepMark、TrustMark、Robust-Wide。
- **篡改定位**：在Stable Diffusion Inpaint、ControlNet Inpaint、Splicing三种编辑下，OmniGuard的F1-Score均超过0.95，AUC接近1.0；在含噪条件下（JPEG Q=50、高斯噪声、颜色抖动等），EditGuard的F1下降约0.2，而OmniGuard几乎无性能损失。
- **版权保护**：PSNR达42.33dB，比EditGuard高4.25dB；Instructp2p全局编辑下位准确率98.1%，略低于专门训练的Robust-Wide（99.0%）；常见退化（JPEG、颜色抖动、高斯噪声）下位准确率达95.9%-100%。
- **效率**：单样本训练迭代时间0.16s，较Robust-Wide（3.675s）提速约23倍，且可在11GB显存（NVIDIA 1080Ti）上训练。

## 相关工作脉络
- **EditGuard [66]**：采用可逆网络嵌入双水印，依赖固定定位水印与残差减法生成掩码，本文在其基础上引入被动提取器解耦编码-解码，并将退化范围扩展至AIGC编辑。
- **Robust-Wide [20]**：通过扩散去噪采样引导增强对抗AIGC编辑的鲁棒性，但需多步迭代扩散过程导致训练极慢；本文以VAE压缩+部分移除作为代理攻击，实现相近效果且训练提速23倍。
- **TrustMark [8]**：面向任意分辨率的通用鲁棒水印，不支持篡改定位；本文在其解码架构基础上扩展为支持定位与版权保护的双任务框架。
- **SepMark [52]**：首个深度可分离水印用于版权保护与deepfake检测，但仅支持样本级检测；本文实现像素级定位并同时支持AIGC编辑场景。
- **被动检测方法（MVSS-Net、HiFi-Net、IML-ViT等）**：依赖图像内部异常线索（噪声、伪影、分辨率不一致），无需预嵌入信息但泛化性有限；本文主动+被动混合策略在复杂退化下显著优于纯被动方法。

## 局限性与未来方向
- 对于Instructp2p等全局语义编辑，倾向于将整个图像标记为篡改区域，尚未有效区分语义修改与像素级篡改。
- 定位能力聚焦于像素级篡改，对AIGC生成的语义级内容替换检测能力有限。
- 自适应水印变换虽提升了保真度，但在极端内容差异下的泛化表现尚待验证。

## 研究启发与可借鉴点
- **主动嵌入+被动提取的混合范式**：将水印网络的优化目标从"精确重建"转向"提升保真度"，提取任务改为分类/分割问题，可有效解耦相互制约的优化目标。
- **退化查询机制（Degradation Query）**：借鉴Q-Former思想，通过可学习查询向量自适应融合输入图像与伪影图信息，为跨模态/多源特征融合提供了简洁有效的范式。
- **轻量级AIGC编辑模拟**：以VAE压缩替代完整扩散去噪作为训练代理攻击，在保证效果的同时大幅降低计算开销，该思路可迁移至其他对抗AIGC编辑的场景。
- **自适应内容感知嵌入**：通过可逆变换使嵌入信息获得内容匹配纹理，兼顾视觉质量与信息隐藏容量，对隐写术和水印领域具有通用参考价值。

## 关键术语表
**OmniGuard**：本文提出的混合取证水印框架，融合主动双水印嵌入与被动退化感知提取。
**退化感知篡改提取器**：基于Swin-ViT的被动网络，融合定位水印伪影与篡改图像，通过退化查询自适应预测篡改掩码。
**自适应定位水印变换**：通过可逆仿射变换将原始图像内容融入定位水印，使其纹理更自然、不易察觉。
**AIGC编辑模拟层**：以SD-1.5 VAE压缩和部分掩码移除作为全局/局部编辑的轻量级代理攻击。
**退化查询 $Q_{deg}$**：可学习的退化表示向量，通过内容感知机制自适应选择与当前退化类型匹配的查询。
**残差减法**：传统方法通过比较重建定位水印与原始定位水印的像素差来生成篡改掩码。
**Instructp2p**：基于指令驱动的通用图像编辑模型，用于模拟全局AIGC编辑场景。
**Stable Diffusion Inpaint**：基于扩散模型的局部图像修复/编辑工具。

## 可复现要素
- **数据集**：MIR-Flickr（版权水印训练）、DIV2K（双水印训练）、COCO（定位评估）、UltraEdit（鲁棒性测试）；论文未声明代码开源，未提及模型权重是否开放。
- **关键超参**：batch size 32（版权训练）/ 8（双水印训练）；初始学习率4×10⁻⁶ / 1×10⁻⁵；λ从0.05渐增至27.5；α₁=α₂=10，α₃=100；γ=20；ViT-B backbone（ImageNet-1k MAE预训练）；窗口大小14，padding 1024。
- **训练平台**：NVIDIA GTX 3090Ti服务器。
