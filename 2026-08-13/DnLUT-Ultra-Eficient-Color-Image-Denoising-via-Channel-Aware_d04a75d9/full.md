# DnLUT: Ultra-Eficient Color Image Denoising via Channel-Aware Lookup Tables

Sidi Yang<sup>1</sup> Binxiao Huang<sup>1†</sup> Yulun Zhang<sup>2</sup> Dahai Yu<sup>3</sup> Yujiu Yang<sup>4†</sup> Ngai Wong<sup>1</sup> <sup>1</sup> The University of Hong Kong <sup>2</sup> Shanghai Jiaotong University <sup>3</sup> TCL Corporate Research <sup>4</sup> Tsinghua University

{yangsidi99, huanghx7}@connect.hku.hk, yulun100@gmail.com, dahai.yu@tcl.com yang.yujiu@sz.tsinghua.edu.cn, nwong@eee.hku.hk

## Abstract

While deep neural networks have revolutionized image denoising capabilities, their deployment on edge devices remains challenging due to substantial computational and memory requirements. To this end, we present DnLUT, an ultra-eficient lookup table-based framework that achieves high-quality color image denoising with minimal resource consumption. Our key innovation lies in two complementary components: a Pairwise Channel Mixer (PCM) that efectively captures inter-channel correlations and spatial dependencies in parallel, and a novel L-shaped convolution design that maximizes receptive field coverage while minimizing storage overhead. By converting these components into optimized lookup tables post-training, DnLUT achieves remarkable eficiency - requiring only 500KB storage and 0.1% energy consumption compared to its CNN contestant DnCNN, while delivering 20× faster inference. Extensive experiments demonstrate that DnLUT outperforms all existing LUT-based methods by over 1dB in PSNR, establishing a new state-of-the-art in resource-eficient color image denoising. The project is available at https://github. com/Stephen0808/DnLUT.

## 1. Introduction

Color images pervade our digital world - from social media and photography to scientific imaging and medical diagnostics. These images are inherently more vulnerable to noise contamination than their grayscale counterparts, not only due to their threefold data size during acquisition, storage, and transmission, but also because human perception exhibits heightened sensitivity to color distortions compared to luminance variations.

The advent of deep neural networks (DNNs) has dramatically advanced the field of color image denoising[2, 15,

![](images/15fd1829f875174be6cbfa99fe3cc4cc2a403d1d5c46e9e8e54ce9b6bf52c379.jpg)  
Figure 1. Model comparison in terms of color peak signal-to-noise ratio (CPSNR), runtime, and storage. The CPSNR and runtime are calculated on the CBSD68 dataset with Gaussian noise level � = 25 using Qualcomm Snapdragon 8 Gen2. Our method outperforms state-of-the-art LUT-based methods with the highest CPSNR, at a low storage requirements and reduced runtime. Additionally, our PCM module serves as a versatile plug-in module that enhances existing methods’ performance by over 1dB.

27, 30, 32]. State-of-the-art models leverage sophisticated architectures with numerous convolutional layers or transformer blocks, achieving unprecedented denoising quality. However, these advances come at a steep cost - the computational complexity and memory requirements of such models render them impractical for edge devices. This limitation is particularly acute given that most edge devices lack specialized hardware accelerators such as graphics processing units (GPUs) or tensor processing units (TPUs), creating a significant gap between algorithmic capabilities and practical deployability.

Recent works have explored lookup tables (LUTs) as an elegant solution for eficient image enhancement. LUTs replace complex runtime computations with simple array indexing operations through direct addressing. This approach typically involves training a neural network model, then converting it into LUTs by exhaustively mapping input-output relationships. During inference, the system performs rapid table lookups instead of costly computations, efectively leveraging DNN capabilities while circumventing hardware constraints. However, LUT-based methods face a fundamental challenge: the storage requirement grows exponentially with the input dimensionality. This has led to the adoption of 4D LUTs (using four pixel values for indexing) as a practical compromise between storage and performance.

Current LUT-based methods predominantly employ 2 × 2 convolutional kernels processed independently for each channel[10, 12, 14, 16], creating spatial-wise LUTs for single-channel processing. While this approach works well for tasks like super-resolution, it proves inadequate for color image denoising where noise typically afects all RGB channels simultaneously. Channel correlation plays a pivotal role in modeling noise characteristics, a critical factor for achieving efective restoration. Alternative approaches either use 1 × 1 convolutions with three-channel depth for RGB information (creating 3D channel-wise LUTs)[13, 25] or attempt to use larger �×�×3 kernels. However, these solutions either disrupt noise distribution patterns or lead to prohibitive storage requirements.

To address these challenges, we propose DnLUT, a novel lookup table-based framework specifically designed for efficient color image denoising. Our approach centers on two key innovations:

First, we introduce the Pairwise Channel Mixer (PCM), which simultaneously processes spatial and channel information. PCM strategically combines RGB channels into three pairs (RG, GB, BR), processing them through parallel branches using a 1×2 convolution kernel with depth 2. This design eficiently captures channel correlations while maintaining manageable storage requirements. The PCM module can also serve as a plug-in module for existing LUT-based methods, consistently improving their performance by approximately 1dB.

Second, we develop an innovative L-shaped convolution kernel that addresses the limitations of conventional rotation-based approaches. After taking the channel correlation into account, as discussed in [12, 14], there is still a need to enlarge the spatial receptive field. While previous methods like SR-LUT[10] use rotation ensemble training to expand receptive fields of spatial-wise LUT, they often introduce redundant pixel usage and significant storage overhead. Our L-shaped kernel design eliminates pixel overlap during rotations, enabling conversion to more eficient 3D LUTs while maintaining the same efective receptive field size as 4D LUTs.

Extensive experimental validation demonstrates the superiority of DnLUT over existing approaches:

• We achieve up to 1.3dB CPSNR improvement over SPFLUT[14] on standard Gaussian denoising benchmarks. For real-world denoising tasks, we outperform CBM3D[3] by nearly 5dB.

• Our method maintains exceptional eficiency, requiring only 0.1% of DnCNN[30]’s energy consumption and 500KB storage.

• The PCM module serves as a versatile performance enhancer, bringing over 1dB CPSNR improvement to existing LUT-based methods.

• Our L-shaped convolution design reduces storage requirements by 17× while preserving receptive field coverage.

These results establish DnLUT as a significant advance in practical, resource-eficient color image denoising, bridging the gap between algorithmic capability and deployability on edge devices.

## 2. Related works

## 2.1. Color Image denoising

A fundamental characteristic of color images is the strong correlation among RGB channels[5]. Research has consistently shown that processing these channels independently yields suboptimal results compared to joint processing approaches[19]. This insight has driven numerous innovations in color image denoising.

A landmark development was CBM3D[3], which operates in luminance-chrominance space $( i . e . , Y C _ { b } C _ { r } )$ The method performs BM3D[4] grouping once on the luminance channel � and leverages this grouping for collaborative filtering of chrominance channels � and �. Building on this concept, [11] introduced a radial basis function mixture for more sophisticated handling of channel correlations. Another notable approach, MC-WNNM[24], extended WNNM[7] by incorporating a weight matrix to adaptively balance the data fidelity across channels based on their noise characteristics.

The deep learning era, initiated by DnCNN[30], transformed the field of color image denoising. Subsequent approaches[2, 15, 26, 27, 32] have increasingly sophisticated architectures that process all RGB channels simultaneously through complex networks of convolution layers or transformer blocks. While these models achieve remarkable denoising quality, their substantial computational and memory requirements present significant deployment challenges, particularly for resource-constrained edge devices.

## 2.2. Replacing CNN with LUT

The challenge of deploying deep neural networks on edge devices has sparked interest in lookup table (LUT) based alternatives. Inspired by the prevalence of color LUTs in image signal processors, [28] pioneered an approach using learnable basis LUTs combined with a weight prediction network for image enhancement. [25] further developed this concept, proposing a more eficient channel-wise LUT system. However, these early channel-wise LUTs, processing single pixels per channel, proved inadequate for the complex task of denoising.

A significant breakthrough came with [10], which introduced a method to convert CNNs with limited receptive fields into spatial-wise LUTs for super-resolution. This approach trains networks on restricted pixel neighborhoods $( i . e . , 2 \times 2 )$ and converts them to LUTs by exhaustively mapping input-output relationships. During inference, the system eficiently retrieves pre-computed outputs through coordinate indexing. Follow-up works[12, 16, 17] enhanced this framework using serial LUTs to expand receptive fields, achieving substantial performance improvements.

To manage the exponential growth of LUT size with input dimensionality, these methods typically process each input channel independently. While this approach proves effective for super-resolution, where spatial information dominates, it falls short for color image denoising, which requires preserving both spatial and channel information to model noise distribution efectively. The limitations of existing channel-wise and spatial-wise LUT approaches highlight the need for a more sophisticated solution that can handle the unique challenges of color image denoising while maintaining computational eficiency.

## 3. Method

## 3.1. Preliminary

The foundation of LUT-based image restoration was established by SR-LUT[10], which introduces a constrained receptive field approach for network training $( i . e . , \ 2 \times 2 )$ Post-training, the network’s output values are systematically cached into a LUT by exhaustively traversing all possible input pixel value combinations. During inference, input patches of size 2 × 2 are converted to indices of a 4D LUT to retrieve corresponding output values. To manage storage requirements, the 4D index space is uniformly subsampled. Additionally, a rotation ensemble strategy rotates each 2 × 2 input patch four times, efectively expanding the receptive field to 3 × 3. Recent works have focused on increasing the efective receptive field through various techniques, including multiple LUTs[12, 14, 17], shift aggregations[17], and diverse kernel patterns[8, 12, 16].

Our work advances this foundation by specifically addressing the challenges of LUT-based color image denoising, a critical application for edge devices.

A notable limitation of existing LUT-based approaches is their treatment of channel depth, typically set to 1. For color images, this means each RGB channel independently accesses cached output values without inter-channel interaction. This design overlooks crucial RGB channel correlations and spatial-channel-wise degradation patterns inherent in color image denoising. While a straightforward solution might be to employ vanilla convolution kernel patterns, the exponential growth of LUT size with index dimension makes this impractical. For instance, a kernel of size $2 \times 2$ with depth 3 would require approximately 582 TB of storage, even with subsampling compression. This constraint presents a fundamental dilemma: when aiming to capture more spatial information, we have to sacrifice the richness of channel information. Moreover, while 4D spatial-wise LUTs show promise, they typically require multiple LUTs (analogous to stacked convolution layers), creating significant storage overhead, particularly problematic for L1 cache access.

Table 1. LUT size versus receptive field (RF) and channel numbers. \* means subsampled LUT size
<table><tr><td>RF</td><td>Depth</td><td>LUT Dim.</td><td>LUT Size*</td></tr><tr><td> $\overline { { 1 \times 1 } }$ </td><td>1</td><td>1D</td><td>17 B</td></tr><tr><td> $1 \times 1$ </td><td>3</td><td>3D</td><td>4.9 KB</td></tr><tr><td>2 × 2</td><td>1</td><td>4D</td><td>83.5 KB</td></tr><tr><td> $1 \times 2$ </td><td>3</td><td>6D</td><td>24.1 MB</td></tr><tr><td>2× 2</td><td>3</td><td>12D</td><td>582.6 TB</td></tr><tr><td> $k \times k$ </td><td>c</td><td> $k \times k \times c \mathbf { D }$ </td><td> $( 2 ^ { 4 } + 1 ) ^ { k \times k \times C } \mathrm { ~ B ~ }$ </td></tr></table>

## 3.2. Overview

To address these fundamental challenges, we introduce two complementary innovations: 1) a pairwise channel mixer that leverages 4D LUT eficiency while modeling spatial-channel correlations; and 2) a novel L-shaped kernel featuring inherent rotation non-overlapping properties, enabling conversion to 3D LUT while maintaining 4D LUTequivalent receptive field coverage.

We first build the DnNet for training, as shown in Fig. 2. Our proposed framework enhances traditional spatial-wise LUT capabilities through these two key modules: the pairwise channel mixer (PCM) establishes robust channelspatial correlations in three RGB channels while the Lshaped kernel expands the efective receptive field with minimal computational overhead independently in each channel. The architecture combines PCM with L-shaped convolution in two stages. The first stage, inspired by [14], generates multi-channel features from LUT groups, which are then concatenated in the fusion module. The second stage integrates both proposed modules. Importantly, all components are designed for eficient conversion to 3D or 4D LUTs during inference, collectively referred to as DnLUT.

## 3.3. Pairwise Channel Mixer

Designing efective kernel patterns for LUT-based methods presents significant challenges. Fig. 3 illustrates various kernel pattern categories. Traditional approaches have focused on single-dimension kernels due to LUT index dimension constraints, leading to two major limitations in color image denoising. Spatial-wise kernels, while capturing spatial information, neglect channel correlations, resulting in color distortion and suboptimal noise modeling. Conversely, channel-wise kernels establish strong RGB channel connections but ignore spatial relationships, leading to persistent artifacts. While combining these kernels in a cascade network might seem promising, this approach fails to process channel and spatial information simultaneously, resulting in suboptimal noise distribution modeling.

![](images/d16fceb199efd4a6ae71ea51ce1286e3c18cac99fe628e48070879c975066cab.jpg)  
Figure 2. System architecture of DnLUT: (a) The DnNet pipeline integrates pairwise channel mixers and L-shaped convolutions, with multi-scale fusion enhancing receptive field coverage. Channel dimensions are flattened for parallel processing in L-shaped operations then unfolded for PCM input. (b) Input pixels undergo four rotations (0°, 90°, 180°, 270°) during processing, with outputs averaged for enhanced results. (c) Post-training, all possible input combinations are processed through DnNet modules, with outputs cached in optimized 3D or 4D LUTs. (d) During inference, input pixels are eficiently processed through multiple LUTs, with each LUT’s outputs informing subsequent LUT indices, culminating in final denoised pixel values.

Our PCM addresses these limitations through a novel architecture for color image denoising. The module first reorganizes RGB channels into three pairwise combinations: RG, GB, and BR. These pairs are processed through parallel branches using a kernel with $1 \times 2$ spatial dimensions and depth 2 in the initial layer, followed by cascaded $1 \times 1$ convolution layers. Each convolution operation processes four pixel values to produce one channel output, enabling eficient conversion to 4D LUTs post-training. The cached LUTs (i.e., $L U T _ { R G } , L U T _ { G B } , L U T _ { G R } )$ operate in parallel, with their predictions concatenated. For an input anchor $( I _ { R } , I _ { G } , I _ { B } )$ , the output values $( V _ { R } , V _ { G } , V _ { B } )$ are computed as:

$$
\begin{array} { r } { ( V _ { R } , V _ { G } , V _ { B } ) = \mathrm { C a t } ( L U T _ { R G } [ I _ { 0 , 0 , R } ] [ I _ { 0 , 1 , R } ] [ I _ { 0 , 0 , G } ] [ I _ { 0 , 1 , G } ] , } \\ { L U T _ { G B } [ I _ { 0 , 0 , G } ] [ I _ { 0 , 1 , G } ] [ I _ { 0 , 0 , B } ] [ I _ { 0 , 1 , B } ] , } \\ { L U T _ { B R } [ I _ { 0 , 0 , B } ] [ I _ { 0 , 1 , B } ] [ I _ { 0 , 0 , R } ] [ I _ { 0 , 1 , R } ] ) , } \end{array}\tag{1}
$$

where �, � , � of each $I _ { H , W , C }$ denotes height, width, and channel index. PCM efectively balances spatial and channel-wise processing, capturing inter-channel correlations crucial for color fidelity while preserving spatial relationships through the 1 × 2 kernel design and refining features via cascaded $1 \times 1$ convolutions.

![](images/5d3f74298928a5fdc1ad8bfa3c793c65ee2903adcce75e5c2b72123d28a157da.jpg)  
(a) Spatial-wise

![](images/86632a36f3ae2424f03934424a24a12cad6f65fe80dfde8b1a626abf25fec0fe.jpg)  
(b) Channel-wise

![](images/1c3080fc5f3077b611e2207ab976d8c54a95c311f731d1f2a9f0c4af13092202.jpg)  
(c) Spatial-Channel-wise  
Figure 3. Taxonomy of kernel patterns for LUT-based methods. Dark cubes indicate rotation points, while medium-dark regions show involved pixel positions during one rotation.

## 3.4. Rotation Non-overlapping Kernel

The rotation ensemble strategy is fundamental to LUTbased methods, using four rotations (0°, 90°, 180°, 270°) to enable asymmetric convolution kernels to cover larger symmetric areas. By combining outputs from all rotations, the approach enhances performance without increasing LUT index dimensionality. As shown in Fig. 4, traditional methods like SR-LUT and MuLUT use this strategy to expand receptive fields to $3 \times 3$ and $5 \times 5$ respectively.

However, this conventional approach has two critical limitations. First, approximately half of the pixels within the receptive field are accessed multiple times during lookup operations, as illustrated in Fig. 4’s tables. This redundancy indicates ineficient kernel pattern design. Second, stacking these kernels to create multiple LUTs for expanded receptive fields results in significant 4D LUT storage overhead.

![](images/ef9e2ceab1025b50db471e64a86d05223e49cf0596bad5d62fae964d52b52f0b.jpg)  
Figure 4. Comparison of spatial-wise kernel designs. Left patterns show kernel configurations, while right tables quantify lookup frequencies during output retrieval.

To address these ineficiencies, we introduce an innovative L-shaped kernel design. During each rotation, the kernel processes two additional pixels (beyond the center pixel) without overlap, ensuring each surrounding pixel contributes exactly once to the output. This design maximizes pixel value utilization within the receptive field while enabling conversion to 3D LUTs instead of 4D LUTs, dramatically reducing storage requirements without sacrificing coverage.

## 3.5. PCM Plug-in Module

To demonstrate the versatility of our pairwise channel mixer, we developed PCM as a plug-in module compatible with existing LUT-based methods. The inference of plug-in algorithm is summarized in Alg. 1. This module seamlessly integrates with current architectures, enhancing their ability to capture channel-spatial correlations without requiring structural modifications. Compared to state-of-the-art methods like SPFLUT, the PCM plug-in adds only 12% runtime and 8% storage overhead while delivering over 1dB performance improvement.

## 4. Experiments

## 4.1. Implementation details

We implement DnLUT using the Adam optimizer with a cosine annealing learning rate schedule and mean squared error (MSE) loss function. For Gaussian denoising, we train networks for $2 \times 1 0 ^ { 5 }$ iterations with a batch size of 12, increasing to 32 for real-world denoising scenarios. Following established practices[10, 12], we uniformly sample LUTs at $2 ^ { 4 }$ intervals. To minimize indexing artifacts, final predictions utilize 4D simplex interpolation and 3D tetrahedral interpolation. We further refine cached LUTs through 2000 iterations of LUT-aware fine-tuning[12] on the training dataset.

Algorithm 1 Inference of Pairwise Channel Mixer Module   
Require: Input image � of size $H \times W \times 3$   
Require: 4D Lookup Table ��� of PCM   
Require: Padding size �   
1: Initialize output image � of size $H \times W \times 3$   
2: for $r = 0$ to 3 do   
3: Rotate image � by $r \times 9 0$ degrees to get $I _ { \mathrm { r o t } }$   
4: Pad $I _ { \mathrm { r o t } }$ with � pixels on each side to get $I _ { \mathrm { r o t } } ^ { \prime }$   
5: for $h = 0$ to $H - 1$ do   
6: for $w = 0$ to $W - 1$ do   
7: for $c = 0$ to 2 do   
8: $p _ { 1 } \gets I _ { \mathrm { r o t } } ^ { \prime } [ h , w , c ]$   
9: $p _ { 2 } \gets I _ { \mathrm { r o t } } ^ { \prime } [ h , w + 1 , c ]$   
10: $p _ { 3 }  I _ { \mathrm { r o t } } ^ { \prime } [ h , w , ( c + 1 )$ mod 3]   
11: $p _ { 4 }  I _ { \mathrm { r o t } } ^ { \prime } [ h , w + 1 , ( c + 1 )$ mod 3]   
12: $i n d e x \gets \mathrm { Q u a n t i z e } ( p _ { 1 } , p _ { 2 } , p _ { 3 } , p _ { 4 } )$   
13: ����� ← ���[�����]   
14: $O [ h , w , c ]  O [ h , w , c ] +$ Interpolate(�����)   
15: end for   
16: end for   
17: end for   
18: Rotate � back by $( 4 - r ) \times 9 0$ degrees to get ${ \cal O } _ { \mathrm { r o t } }$   
19: $O  O + O _ { \mathrm { r o t } }$   
20: end for   
21: $O  O / 4$   
22: return Output image �

## 4.2. Gaussian Color Image denoising

Following [27], we train our model using four comprehensive datasets: DIV2K[1], Flickr2K[23], BSD400[20], and WaterlooED[18]. Training data is augmented with additive Gaussian noise at three levels $( \sigma ~ \in ~ \{ 1 5 , 2 5 , 5 0 \} )$ . We evaluate performance on four standard benchmarks: CBSD68[21], Kodak24, Urban100[9], and McMaster[31], using identical noise levels.

## 4.3. Real-world Color Image denoising

To validate real-world performance, we train DnLUT on the SIDD training dataset and evaluate on both SIDD validation and DnD datasets. The SIDD dataset comprises indoor scenes captured by five smartphones under varying noise conditions, providing 320 image pairs for training and 1,280 for validation. The DnD dataset ofers 50 high-resolution images with realistic noise patterns for comprehensive evaluation.

Table 2. Quantitative comparison on Gaussian color image denoising benchmark. The table presents CPSNR(dB) values.
<table><tr><td rowspan="2">Category</td><td rowspan="2">Method</td><td colspan="3">CBSD68</td><td colspan="3">Kodak24</td><td colspan="3">Urban100</td><td colspan="3">McMaster</td></tr><tr><td>σ = 15</td><td>σ = 25</td><td>σ = 50</td><td>σ = 15</td><td>σ = 25</td><td>σ = 50</td><td>σ = 15</td><td>σ = 25</td><td> $\sigma = 5 0$ </td><td>σ = 15</td><td>σ = 25</td><td>σ = 50</td></tr><tr><td rowspan="5">LUT-based</td><td>SR-LUT[10]</td><td>29.76</td><td>26.71</td><td>22.41</td><td>30.35</td><td>27.16</td><td>22.65</td><td>29.38</td><td>26.04</td><td>21.60</td><td>31.18</td><td>28.01</td><td>23.35</td></tr><tr><td>MuLUT[12]</td><td>30.52</td><td>28.11</td><td>24.85</td><td>31.31</td><td>29.02</td><td>25.28</td><td>30.25</td><td>27.67</td><td>23.75</td><td>32.28</td><td>29.88</td><td>26.36</td></tr><tr><td>RC-LUT[16]</td><td>30.68</td><td>28.12</td><td>25.04</td><td>31.57</td><td>29.07</td><td>25.89</td><td>30.33</td><td>27.80</td><td>23.86</td><td>32.51</td><td>29.89</td><td>26.50</td></tr><tr><td>SPF-LUT[14]</td><td>30.97</td><td>28.56</td><td>25.33</td><td>31.86</td><td>29.58</td><td>26.26</td><td>30.89</td><td>28.26</td><td>24.22</td><td>31.77</td><td>30.44</td><td>26.91</td></tr><tr><td>DnLUT(Ours)</td><td>32.41</td><td>29.88</td><td>26.03</td><td>33.02</td><td>30.24</td><td>26.74</td><td>32.12</td><td>28.87</td><td>25.01</td><td>32.88</td><td>30.44</td><td>27.12</td></tr><tr><td rowspan="2">Classical</td><td>CBM3D[3]</td><td>33.52</td><td>30.71</td><td>27.38</td><td>34.28</td><td>31.68</td><td>29.02</td><td>33.93</td><td>31.36</td><td>27.93</td><td>34.06</td><td>31.66</td><td>28.51</td></tr><tr><td>MC-WNNM[24]</td><td>31.98</td><td>29.32</td><td>26.98</td><td>33.23</td><td>30.89</td><td>28.67</td><td>30.23</td><td>29.23</td><td>27.00</td><td>31.23</td><td>30.20</td><td>27.55</td></tr><tr><td rowspan="2">DNN</td><td>DnCNN[30]</td><td>33.90</td><td>31.24</td><td>27.95</td><td>34.60</td><td>32.14</td><td>28.96</td><td>32.98</td><td>30.81</td><td>27.59</td><td>33.45</td><td>31.52</td><td>28.62</td></tr><tr><td>SwinIR[15]</td><td>34.42</td><td>31.78</td><td>28.56</td><td>35.34</td><td>32.89</td><td>29.79</td><td>35.61</td><td>33.20</td><td>30.22</td><td>35.13</td><td>32.90</td><td>29.82</td></tr></table>

Table 3. Quantitative comparison on real-world color image denoising. The table presents CPSNR(dB) and SSIM values. Methods on DnD dataset are validated on the online platform. We could only get the PSNR rather than CPSNR.
<table><tr><td>Datasets</td><td>Method</td><td>SR-LUT[10]</td><td>MuLUT[12]</td><td>RC-LUT[16]</td><td>SPF-LUT[14]</td><td>DnLUT(Ours)</td><td>CBM3D[3]</td><td>MC-WNNM[24]</td><td>DnCNN[30]</td></tr><tr><td rowspan="2">SIDD</td><td>CPSNR</td><td>29.38</td><td>33.24</td><td>33.88</td><td>34.91</td><td>35.44</td><td>30.14</td><td>29.45</td><td>36.45</td></tr><tr><td>SSIM</td><td>0.634</td><td>0.830</td><td>0.839</td><td>0.865</td><td>0.875</td><td>0.702</td><td>0.682</td><td>0.900</td></tr><tr><td rowspan="2">DnD</td><td>PSNR*</td><td>33.69</td><td>35.11</td><td>35.43</td><td>36.22</td><td>36.67</td><td>33.12</td><td>32.34</td><td>37.11</td></tr><tr><td>SSIM</td><td>0.839</td><td>0.868</td><td>0.875</td><td>0.911</td><td>0.922</td><td>0.823</td><td>0.817</td><td>0.932</td></tr></table>

![](images/0199051d89fc432e0376be4682a09921237ae4100d478aedf01d955aa60d05ea.jpg)  
Figure 5. Qualitative comparison on synthetic datasets.

![](images/2ecff09cb65bbeb7428db9038d5e9d68d30473493a8031eb2b73edd0f140c9ef.jpg)  
Figure 6. Qualitative comparison on real-world datasets.

Table 4. PCM can be easily incorporated to the existing LUT-based methods. It equips the model with the ability of capturing the channel correlations which brings more than 1dB on the widely used benchmark. The noise level � of Gaussian noise is set to 25. The table presents CPSNR(dB) values.
<table><tr><td>Method</td><td>CBSD68</td><td>Kodak24</td><td>Urban100</td><td>McMaster</td><td>SIDD</td></tr><tr><td>SR-LUT[10]</td><td>26.71</td><td>27.16</td><td>26.04</td><td>28.01</td><td>29.38</td></tr><tr><td>PCM+SR-LUT</td><td>28.04(+1.33)</td><td>28.55(+1.39)</td><td>27.33(+1.29)</td><td>28.78(+0.77)</td><td>32.33(+2.96)</td></tr><tr><td>MuLUT[12] PCM+MuLUT</td><td>28.11 29.17(+1.05)</td><td>29.01 29.92(+0.91)</td><td>27.67 28.74(+1.07)</td><td>29.88 30.33(+0.44)</td><td>33.24 34.33(+1.09)</td></tr><tr><td>RC-LUT[16]</td><td>28.12</td><td>29.07</td><td>27.80</td><td>29.89</td><td>33.88</td></tr><tr><td>PCM+RC-LUT</td><td>29.22(+1.10)</td><td>30.09(+1.02)</td><td>28.77(+0.97)</td><td>30.54(+0.65)</td><td>34.67(+0.79)</td></tr><tr><td>SPF-LUT[14]</td><td>28.56</td><td>29.58</td><td>28.26</td><td>30.44</td><td>34.91</td></tr><tr><td>PCM+SPF-LUT</td><td>29.72(+1.16)</td><td>30.63(+1.05)</td><td>29.43(+1.17)</td><td>30.96(+0.52)</td><td>35.56(+0.65)</td></tr></table>

## 4.4. Quantitative results

We evaluate performance using color peak signal-to-noise ratio(CPSNR) and structural similarity index(SSIM) metrics, comparing against state-of-the-art methods across three categories: LUT-based methods (SR-LUT[10], MuLUT[12], RC-LUT[16], SPF-LUT[14]), classical approaches (CBM3D[3], MC-WNNM[24]), and DNN-based solutions (DnCNN[30], SwinIR[15]).

As shown in Tab. 2, DnLUT achieves superior performance in the LUT-based category, surpassing SPF-LUT by over 1dB in Gaussian noise removal while requiring only 17% of its storage. Results from Tab. 3 demonstrate DnLUT’s efectiveness in real-world scenarios, outperforming classical methods CBM3D[3] and MC-WNNM[24] by approximately 5dB. Though the Gaussian noise is independent across RGB channels, the network can still leverage inter-channel natural distribution in images to suppress the noise. For example, if a pixel in the R channel has a high value due to noise, but the corresponding pixels in the G and B channels do not, the network can infer that the high value is likely noise and suppress it.

## 4.5. Qualitative results

Visual comparisons in Figs. 5 & 6 illustrate DnLUT’s superior denoising capabilities. Existing LUT-based methods exhibit noticeable color distortion, particularly evident in Fig. 5’s first row. In contrast, DnLUT achieves visual quality comparable to computation-intensive methods like DnCNN and CBM3D. Fig. 6 further demonstrates DnLUT’s advantage in real-world scenarios, where both LUT-based and classical methods struggle with color fidelity and often over-smooth details. DnLUT successfully preserves color consistency while retaining fine image details.

## 4.6. Efficiency Evaluation

We conduct comprehensive eficiency analysis across three key metrics: theoretical energy cost, runtime performance, and storage requirements (Tab. 5).

Energy cost. Following AdderSR[22], we analyze multiplication and addition operations across diferent data types to estimate total energy consumption. DnLUT achieves 70% energy savings compared to SPF-LUT while delivering superior denoising quality. The eficiency gap widens dramatically when compared to DNN-based methods, with DnLUT requiring only 0.1% of DnCNN’s energy consumption.

![](images/b59eccc74731f08e178e5805fe7b6fd546f7ed02861a976d413be2e1a741ff8b.jpg)  
Figure 7. Visualization of LUT-based method with plug-in PCM.

Runtime. We implement LUT-based methods using standard JAVA API on the ANDROID platform. DnLUT maintains the rapid inference capabilities characteristic of SR-LUT[10], demonstrating clear advantages over classical and DNN-based approaches. For reference, DnCNN running on mobile CPU requires 20× more processing time. DnLUT’s performance can be further optimized through device-specific implementations, such as FPGA deployments.

Storage. DnLUT requires approximately 500 KB storage, well within typical L2-cache limits of modern chips. This eficient memory footprint enables rapid lookup operations with minimal dependency overhead.

## 4.7. Effectiveness of Plug-in PCM

Tab. 4 demonstrates PCM’s versatility as a performance enhancer for existing LUT-based methods. When integrated with current approaches, PCM consistently delivers significant improvements across all benchmark datasets. The visual comparisons in Fig. 7 show marked reduction in color distortion, particularly in smooth image regions, resulting in superior perceptual quality.

Table 5. Eficiency evaluation of three categories color image denoising methods. We use the Qualcomm Snapdragon 8 Gen 2 as the mobile platform. The energy cost is calculated on the 512 × 512 resolution color images.
<table><tr><td rowspan="2">Category</td><td rowspan="2">Method</td><td rowspan="2">Platform</td><td rowspan="2">Runtime(ms) 256×256</td><td rowspan="2">Runtime(ms) 512×512</td><td rowspan="2">Energy Cost (pJ)</td><td rowspan="2">Storage (KB)</td></tr><tr><td></td></tr><tr><td rowspan="5">LUT-based</td><td>SR-LUT[10]</td><td>Mobile</td><td>6</td><td>19</td><td>149.98M</td><td>82</td></tr><tr><td>MuLUT[12]</td><td>Mobile</td><td>21</td><td>80</td><td>899.88M</td><td>489</td></tr><tr><td>RC-LUT[6]</td><td>Mobile</td><td>18</td><td>78</td><td>612.92M</td><td>326</td></tr><tr><td>SPF-LUT[14]</td><td>Mobile</td><td>72</td><td>257</td><td>2.32G</td><td>30,178</td></tr><tr><td>DnLUT(ours)</td><td>Mobile</td><td>23</td><td>88</td><td>687.34M</td><td>518</td></tr><tr><td rowspan="2">Classical</td><td>CBM3D[29]</td><td>PC</td><td>3,189</td><td>14,343</td><td>4.82G</td><td></td></tr><tr><td>MC-WNNM[24]</td><td>PC</td><td>74,290</td><td>293,424</td><td>89.23G</td><td></td></tr><tr><td rowspan="2">DNN</td><td>DnCNN[30]</td><td>Mobile</td><td>543</td><td>1,923</td><td>542.53G</td><td>2,239</td></tr><tr><td>SwinIR[15]</td><td>Mobile</td><td>73,234</td><td>274,453</td><td>12.03T</td><td>45,499</td></tr></table>

## 4.8. Ablation study

We conduct ablation studies using Gaussian color image denoising benchmark to validate DnLUT’s key components.

## 4.8.1. Channel-spatial-wise convolution

Channel indexing[13, 25] possesses a 1 × 1 receptive field to extract channel information. We evaluate various combinations of channel-wise, spatial-wise, and our proposed channel-spatial-wise kernels using a consistent architecture comprising one channel indexing module, two L-shaped convolution modules, and one PCM component.

Results in Tab. 6 reveal that naive combination of channel-wise and spatial-wise convolutions yields suboptimal performance. Pure channel-spatial-wise convolution, while promising, struggles with limited receptive field coverage. The integration of spatial-wise and channel-spatialwise convolutions produces substantial improvements. Notably, adding channel-wise convolution ofers minimal additional benefit, as channel-spatial-wise convolution already captures comprehensive color information.

## 4.8.2. L-shaped convolution

To validate our L-shaped convolution design, we compare it against standard $2 \times 2$ spatial-wise kernels (denoted as S in Tab. 7, $\sigma ~ = ~ 1 )$ While the conventional kernel shows marginal performance advantages, our L-shaped design achieves comparable results while enabling 3D LUT conversion, reducing storage requirements by 17×. It paves the way of deploying multi-layer spatial-wise convolutions.

## 5. Conclusion

This paper presents DnLUT, a novel approach to color image denoising that bridges the gap between high-performance deep learning models and resource-constrained edge devices. Addressing the limitations of existing LUT-based methods, DnLUT introduces two key innovations. First, the Pairwise Channel Mixer (PCM) models channel and spatial correlations simultaneously by processing channel pairs, enabling comprehensive color modeling without excessive storage costs. Second, an L-shaped convolution design rethinks rotation-based receptive field expansion, eliminating redundant pixel usage and converting 4D to 3D LUTs, reducing storage by 17× while maintaining spatial coverage. These advancements allow DnLUT to outperform state-ofthe-art LUT-based methods by over 1dB in denoising quality while consuming just 0.1% of DnCNN’s energy.

Table 6. Diferent combinations of channel-wise, spatial-wise, and our channel-spatial-wise kernels. These kernels are represented by channel indexing[13], L-shaped kernel, and PCM. The table presents CPSNR(dB) values.
<table><tr><td>Channel</td><td>Spatial</td><td>Channel-spatial</td><td>CBSD68</td></tr><tr><td rowspan="5">√</td><td>√</td><td rowspan="2"></td><td>29.77</td></tr><tr><td>√</td><td>30.15</td></tr><tr><td>√</td><td></td><td>30.21</td></tr><tr><td>√</td><td>√</td><td>31.41</td></tr><tr><td>√</td><td>√</td><td>31.47</td></tr></table>

Table 7. Comparison of L-shaped kernel and $2 \times 2$ spatial-wise kernel. The table presents CPSNR(dB) values.
<table><tr><td>Method</td><td>CBSD68</td><td>Kodak</td><td>McMaster</td><td>LUT Size(KB)</td></tr><tr><td> $\overline { { \mathrm { P C M } + \mathrm { L } } }$ </td><td>26.63</td><td>27.05</td><td>27.96</td><td> $\overline { { 4 9 4 + 2 4 } }$ </td></tr><tr><td> $\mathrm { P C M } + \mathrm { S }$ </td><td>26.71</td><td>27.16</td><td>28.01</td><td> $4 9 4 + 4 0 8$ </td></tr></table>

Acknowledgements. This research was supported by: the HKU-TCL Joint Research Centre for AI, the Theme-based Research Scheme (TRS) project T45-701/22-R, and the General Research Fund (GRF) Project 17203224 from the Research Grants Council (RGC) of Hong Kong SAR. Additional support was provided by the National Key Research and Development Program of China (Grant No. 2024YFB2808903), the Shenzhen Science and Technology Program (JCYJ20220818101001004), and the Tsinghua University - Tencent Joint Laboratory for Internet Innovation Technology Research Fund.

## References

[1] Eirikur Agustsson and Radu Timofte. Ntire 2017 challenge on single image super-resolution: Dataset and study. In The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, 2017. 5

[2] Hanting Chen, Yunhe Wang, Tianyu Guo, Chang Xu, Yip ing Deng, Zhenhua Liu, Siwei Ma, Chunjing Xu, Chao Xu, and Wen Gao. Pre-trained image processing transformer. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12299–12310, 2021. 1, 2

[3] Kostadin Dabov, Alessandro Foi, Vladimir Katkovnik, and Karen Egiazarian. Color image denoising via sparse 3d collaborative filtering with grouping constraint in luminancechrominance space. In 2007 IEEE international conference on image processing, pages I–313. IEEE, 2007. 2, 6, 7

[4] Kostadin Dabov, Alessandro Foi, Vladimir Katkovnik, and Karen Egiazarian. Image denoising by sparse 3-d transformdomain collaborative filtering. IEEE Transactions on image processing, 16(8):2080–2095, 2007. 2

[5] Jingjing Dai, Oscar C Au, Lu Fang, Chao Pang, Feng Zou, and Jiali Li. Multichannel nonlocal means fusion for color image denoising. IEEE Transactions on circuits and systems for video technology, 23(11):1873–1886, 2013. 2

[6] Michaël Gharbi, Jiawen Chen, Jonathan T Barron, Samuel W Hasinof, and Frédo Durand. Deep bilateral learning for realtime image enhancement. ACM Transactions on Graphics (TOG), 36(4):1–12, 2017. 8

[7] Shuhang Gu, Lei Zhang, Wangmeng Zuo, and Xiangchu Feng. Weighted nuclear norm minimization with application to image denoising. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2862–2869, 2014. 2

[8] Binxiao Huang, Jason Chun Lok Li, Jie Ran, Boyu Li, Jiajun Zhou, Dahai Yu, and Ngai Wong. Hundred-kilobyte lookup tables for eficient single-image super-resolution. arXiv preprint arXiv:2312.06101, 2023. 3

[9] Jia-Bin Huang, Abhishek Singh, and Narendra Ahuja. Single image super-resolution from transformed self-exemplars. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5197–5206, 2015. 5

[10] Younghyun Jo and Seon Joo Kim. Practical single-image super-resolution using look-up table. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 691–700, 2021. 2, 3, 5, 6, 7, 8

[11] Stamatios Lefkimmiatis. Non-local color image denoising with convolutional neural networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3587–3596, 2017. 2

[12] Jiacheng Li, Chang Chen, Zhen Cheng, and Zhiwei Xiong. Mulut: Cooperating multiple look-up tables for eficient image super-resolution. In European Conference on Computer Vision, pages 238–256. Springer, 2022. 2, 3, 5, 6, 7, 8

[13] Jiacheng Li, Chang Chen, Zhen Cheng, and Zhiwei Xiong. Toward dnn of luts: Learning eficient image restoration with multiple look-up tables. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024. 2, 8

[14] Yinglong Li, Jiacheng Li, and Zhiwei Xiong. Look-up table compression for eficient image restoration. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pat tern Recognition, pages 26016–26025, 2024. 2, 3, 6, 7, 8

[15] Jingyun Liang, Jiezhang Cao, Guolei Sun, Kai Zhang, Luc Van Gool, and Radu Timofte. Swinir: Image restoration using swin transformer. In Proceedings ofthe IEEE/CVF international conference on computer vision, pages 1833–1844, 2021. 1, 2, 6, 7, 8

[16] Guandu Liu, Yukang Ding, Mading Li, Ming Sun, Xing Wen, and Bin Wang. Reconstructed convolution module based look-up tables for eficient image super-resolution. In Proceedings of the IEEE/CVF International Conference on Com puter Vision, pages 12217–12226, 2023. 2, 3, 6, 7

[17] Cheng Ma, Jingyi Zhang, Jie Zhou, and Jiwen Lu. Learn ing series-parallel lookup tables for eficient image superresolution. In European Conference on Computer Vision, pages 305–321. Springer, 2022. 3

[18] Kede Ma, Zhengfang Duanmu, Qingbo Wu, Zhou Wang, Hongwei Yong, Hongliang Li, and Lei Zhang. Waterloo ex ploration database: New challenges for image quality assessment models. IEEE Transactions on Image Processing, 26 (2):1004–1016, 2016. 5

[19] Julien Mairal, Michael Elad, and Guillermo Sapiro. Sparse representation for color image restoration. IEEE Transactions on image processing, 17(1):53–69, 2007. 2

[20] D. Martin, C. Fowlkes, D. Tal, and J. Malik. A database of human segmented natural images and its application to evaluating segmentation algorithms and measuring ecological statistics. In Proc. 8th Int’l Conf. Computer Vision, pages 416–423, 2001. 5

[21] David Martin, Charless Fowlkes, Doron Tal, and Jitendra Malik. A database of human segmented natural images and its application to evaluating segmentation algorithms and measuring ecological statistics. In Proceedings eighth IEEE international conference on computer vision. ICCV 2001, pages 416–423. IEEE, 2001. 5

[22] Dehua Song, Yunhe Wang, Hanting Chen, Chang Xu, Chun jing Xu, and DaCheng Tao. Addersr: Towards energy eficient image super-resolution. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 15648–15657, 2021. 7

[23] Radu Timofte, Eirikur Agustsson, Luc Van Gool, Ming Hsuan Yang, and Lei Zhang. Ntire 2017 challenge on single image super-resolution: Methods and results. In Proceedings of the IEEE conference on computer vision and pattern recognition workshops, pages 114–125, 2017. 5

[24] Jun Xu, Lei Zhang, David Zhang, and Xiangchu Feng. Multichannel weighted nuclear norm minimization for real color image denoising. In Proceedings of the IEEE international conference on computer vision, pages 1096–1104, 2017. 2, 6, 7, 8

[25] Sidi Yang, Binxiao Huang, Mingdeng Cao, Yatai Ji, Hanzhong Guo, Ngai Wong, and Yujiu Yang. Taming lookup tables for eficient image retouching. arXiv preprint arXiv:2403.19238, 2024. 2, 8

[26] Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling

Shao. Multi-stage progressive image restoration. In Proceedings ofthe IEEE/CVF conference on computer vision andpattern recognition, pages 14821–14831, 2021. 2

[27] Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang. Restormer: Eficient transformer for high-resolution image restoration. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5728–5739, 2022. 1, 2, 5

[28] Hui Zeng, Jianrui Cai, Lida Li, Zisheng Cao, and Lei Zhang. Learning image-adaptive 3d lookup tables for high performance photo enhancement in real-time. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(4):2058– 2073, 2020. 2

[29] Fengyi Zhang, Hui Zeng, Tianjun Zhang, and Lin Zhang. Clut-net: Learning adaptively compressed representations of 3dluts for lightweight image enhancement. In Proceedings of the 30th ACM International Conference on Multimedia, pages 6493–6501, 2022. 8

[30] Kai Zhang, Wangmeng Zuo, Yunjin Chen, Deyu Meng, and Lei Zhang. Beyond a gaussian denoiser: Residual learning of deep cnn for image denoising. IEEE transactions on image processing, 26(7):3142–3155, 2017. 1, 2, 6, 7, 8

[31] Lei Zhang, Xiaolin Wu, Antoni Buades, and Xin Li. Color demosaicking by local directional interpolation and nonlocal adaptive thresholding. Journal ofElectronic imaging, 20(2): 023016–023016, 2011. 5

[32] Yi Zhang, Dasong Li, Xiaoyu Shi, Dailan He, Kangning Song, Xiaogang Wang, Hongwei Qin, and Hongsheng Li. Kbnet: Kernel basis network for image restoration. arXiv preprint arXiv:2303.02881, 2023. 1, 2