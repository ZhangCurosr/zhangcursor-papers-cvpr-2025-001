---
title: "DnLUT-Ultra-Eficient-Color-Image-Denoising-via-Channel-Aware"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Yang_DnLUT_Ultra-Efficient_Color_Image_Denoising_via_Channel-Aware_Lookup_Tables_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 07:01:07"
---

# 论文速读：DnLUT-Ultra-Eficient-Color-Image-Denoising-via-Channel-Aware

## 一句话总结
DnLUT 提出了一种基于查找表（LUT）的超轻量级彩色图像去噪框架，通过成对通道混洗器（PCM）与 L 形旋转非重叠卷积核，在仅 ~500 KB 存储与 0.1% 能耗的约束下，实现了优于现有 LUT 方法 1 dB 以上 CPSNR 的实时去噪性能。

## 研究问题与动机
- **边缘部署瓶颈**：高性能 DNN 去噪模型（如 DnCNN、SwinIR）计算与显存开销巨大，难以部署于缺乏 GPU/TPU 的移动终端。
- **现有 LUT 方法的通道建模缺陷**：主流 LUT 基方法（如 SR-LUT）采用 2×2 空间卷积独立处理单通道，或 1×1 卷积忽略空间上下文，无法有效捕获 RGB 通道间的强相关性，导致彩色去噪时色彩失真。
- **高维 LUT 存储爆炸**：若直接使用 2×2×3 等全尺寸空间-通道卷积，LUT 维度将膨胀至约 582 TB，且传统旋转集成策略会造成像素重复访问与冗余索引。
- **资源-性能权衡缺失**：现有工作要么牺牲效率换取精度，要么过度压缩
