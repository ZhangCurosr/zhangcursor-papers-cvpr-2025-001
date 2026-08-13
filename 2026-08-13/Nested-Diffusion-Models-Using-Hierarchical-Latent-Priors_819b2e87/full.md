This CVPR paper is the Open Access version, provided by the Computer Vision Foundation. Except for this watermark, it is identical to the accepted version; the final published version of the proceedings is available on IEEE Xplore.

# Nested Diffusion Models Using Hierarchical Latent Priors

Xiao Zhang⇤<sup>1</sup> Ruoxi Jiang⇤<sup>1</sup> Rebecca Willett<sup>1,2</sup> Michael Maire<sup>1</sup>

![](images/0fbe71cc8a1b40215420fd4e065d1d3ad61523b13cf6a0218302b71bd1c7d177.jpg)  
Figure 1. Image generation via diffusion models nested along a hierarchical semantic chain. We synthesize images using a sequence of diffusion models to generate a hierarchy of latent representations, starting from a low-dimensional semantic feature embedding and refining to a detailed image. At each hierarchical level, synthesis of a higher-dimensional latent from noise is conditioned on the more abstract latents generated at levels above. Here, each successive row visualizes resulting images when fixing latents up to some level and resampling those at subsequent levels; images (darker background) are produced by resampling only the more detailed levels of the hierarchical representation of a specific image in the preceding row (lighter background). Trained on ImageNet-1K, our multi-level generation system, free of any external conditioning (i.e., no class labels), learns a hierarchy that transitions from reflecting abstract semantic similarities to fine visua details. Our code is available.

## Abstract

We introduce nested diffusion models, an efficient and powerful hierarchical generative framework that substantially enhances the generation quality ofdiffusion models, particularlyfor images ofcomplex scenes. Our approach employs a series of diffusion models to progressively generate latent variables at different semantic levels. Each model in this series is conditioned on the output of the preceding higher-level models, culminating in image generation. Hierarchical latent variables guide the generation process along predefined semantic pathways, allowing our approach to capture intricate structural details. To construct these latent variables, we leverage a pre-trained visual encoder, which learns strong semantic visual representations, and modulate its capacity via dimensionality reduction and noise injection Across multiple datasets, our system demonstrates significant enhancements in image qualityfor both unconditional and class/text conditional generation. Moreover, our unconditional generation system substantially outperforms the baseline conditional system. These advancements incur minimal computational overhead as the more abstract levels of our hierarchy work with lower-dimensional representations.

## 1. Introduction

![](images/887597bc48cbb2a2ee01cdfb18b28e295e3a120d8c1e551702cc822a509c0aef.jpg)

![](images/2586e8153c29ba2cf551896ab707aa0381756957750342c96a3d7fd319b47417.jpg)  
Figure 2. Image generation quality when scaling our nested diffusion models on ImageNet-1K dataset. The deeper hierarchies we build lead to a slight increase in computational overhead (particularly when $L \leq 4 ) .$ , as measured by GFlops, while significantly improving the generation quality. Compared to the single-level baseline model using comparable GFlops, our 5-level unconditional system significantly improves the performance w/o classifier-free guidance (CFG) by reducing FID from 45.19 to 11.05, exceeding the class-conditional baseline of 19.74.

The modern era of computer vision opened with deep networks driving advances in representation learning: mapping images, patches, or pixels to feature vectors that encode semantic information and support a range of downstream tasks such as classification [22, 39, 60], segmentation [20, 45], and object detection [5, 16, 23]. A variety of deep generative methods have since emerged to enable the reverse mapping, from a given prior or a learned embedding, back to the space of images. GANs [17], VAEs [38, 46, 50, 61, 68], normalizing flows [1, 48, 71], and diffusion models [18, 19, 62, 77] have demonstrated capacity to synthesize complex realworld image, video, and language data [3, 44, 47]. In parallel, representation learning has advanced through development of scalable architectures and training objectives, yielding self-supervised approaches, including contrastive learning [6, 7, 9, 24], masked autoencoders (MAEs) [25], and hybrids [81], that rival supervised feature learning.

![](images/2f70cac63bfe8010478480e9a75d46c3792b264f58b40a9ef6671ea9685643fb.jpg)  
Figure 3. Nested diffusion architecture. Left: We train a sequence of diffusion models to generate a hierarchical collection of latent representations $\{ { \bf z } _ { 3 } , { \bf z } _ { 2 } , { \bf z } _ { 1 } = { \bf x } \}$ of increasing dimensionality up to an image $\mathbf { z } _ { 1 } = \mathbf { x }$ . Generated latents serve as conditional inputs (dotted lines) to diffusion models at subsequent levels, with separately parameterized noising processes, $\hat { \mathbf { z } } _ { l } \sim \mathcal { N } ( \mathbf { z } _ { l } , \sigma _ { l } ^ { 2 } \mathbf { I } )$ , controlling the information capacity of these signals. Right: A pre-trained, frozen visual encoder provides target latent representations for each level of the hierarchy. To construct these latent features, we run the encoder on patchified images, reducing patch size and applying dimensionality reduction across feature channels in order to shift focus from local details to global semantics. Upper level targets encode more abstrac semantics and, being lower-dimensional vectors, are less computationally expensive to synthesize, making hierarchical generation fast.

Although generation and representation learning may have different immediate applications, they are inherently linked. A process that synthesizes realistic images must internally capture some notion of semantics in order to produce globally coherent structure. Indeed, recent work investigating generative models reveals that they capture rich visual representations useful for downstream tasks: segmentation [4, 72], image intrinsics [14] and image recognition [40, 73]. Conversely, another branch of research demonstrates that strong visual representations can further enhance generation quality via: conditioning on clustered features [30], learning to generate visual features that serve as a conditional signal [42], or adding a self-supervised representation learning loss to a generative model [41, 76].

However, these uses of pre-trained visual encoders or feature learning objectives focus only on abstract, high-level features, and in essence may function as unsupervised substitutes for image category labels.

Images contain diverse, multi-scale structures, from local textures and edges to parts, objects, and coherent scenes. For a generative system to produce realistic images, it must model all of these aspects. Current generative models frequently struggle to accurately represent attributes such as physical properties [33] and geometric layout [59], suggesting that conventional generative training objectives are insufficient for capturing these complex visual relationships.

To address this, we anchor a generation process to a visual feature hierarchy, which provides intermediate targets to guide progressive image synthesis. Our system employs a series of diffusion models, each operating at a different level of semantic abstraction and conditioned on outputs from higher levels. We build training targets for this hierarchical generator using a pre-trained visual encoder, applied to image patches of varying scale, in order to represent visual structures ranging from local texture to global shape. As additional controls on our target feature hierarchy, we compress feature representations through dimensionality reduction and noise-based perturbation. These capacity controls are essential to prevent memorization of image details at intermediate levels, allowing us to scale our model to deep hierarchies.

Unlike traditional methods using VAE features [55] or image pyramids [18] that primarily focuses on local textures, our approach emphasizes structured, multi-scale semantic representations. Compared to the hierarchical VAE [11, 65, 68, 80] models that run generation in a hierarchical latent space but suffer training instability, we use frozen latent representations, which significantly enhances training stability and yields much better generation quality.

Figure 1 shows example output using our method for unconditional synthesis on ImageNet-1k. Our unconditional system even outperforms the conditional generation baseline, as benchmarked in Figure 2, and also achieves consistent quality improvement as the number of levels L in the hierarchy increases from 2 to 5. Figure 3 sketches the key components of our model architecture. Section 4 extends experiments to the challenging setting of text-conditioned image generation trained on COCO scenes, where our hierarchical model outperforms baseline models containing substantially more parameters and consuming significantly larger training datasets. Our contributions are as follows:

• We introduce nested diffusion models, anchoring image generation to a hierarchical feature representation. Top hierarchy levels promote consistency in global image structure, while subsequent levels refine visual details. Resampling specific levels gives tunable control over synthesis.

• Our design greatly enhances generation quality while maintaining efficiency. Our five-level hierarchical model increases computational cost, measured in GFlops, by only 25% relative to single-level diffusion, yet significantly improves quality. Relative to a baseline model requiring comparable GFlops, we decrease FID from 45.19 to 11.05 for unconditional generation and from 31.13 to 9.87 for conditional generation on ImageNet-1k.

• Our system consistently improves performance in both conditional and unconditional generation tasks as more hierarchical levels are added. Notably, on ImageNet-1k, our unconditional generation quality surpasses that of the baseline class-conditional diffusion model.

## 2. Related Work

Hierarchical models. Hierarchical variational autoencoders (HVAEs) [11, 65, 68, 80] extend the latent space of VAEs [38] to include multiple variables, and demonstrate improved generation quality. However, HVAEs are known to suffer from high variance and collapsed representations, where the top-level variables may be ignored [11, 68]. To address this issue, Luhman and Luhman [46] introduce a layer-wise scheduler and regularization to enhance stability, while Hazami et al. [21] propose a simplified architecture.

Recent work has sought to build hierarchical generative systems by freezing the latent variables and leveraging powerful generative methods such as diffusion models and autoregressive models. For example, Gu et al. [18], Ho et al. [29], Liu et al. [44] train a set of diffusion models to handle images at different resolutions, and Tian et al. [67] train a hierarchical autoregressive model to predict the residuals between tokenized representations at adjacent resolutions. However, none of these approaches involve training with hierarchical semantic latent representations.

Conditional generation. A conditional diffusion model aims to parameterize the prior as a complex joint distribution conditioned on an input, rather than using a simple Gaussian prior, significantly enhancing the model’s capacity to capture intricate data patterns. For images of complex scenes, generation conditioned on image captions [19, 34, 54] has shown notable improvements in both quality and controllability. [55, 77] extend this conditioning approach to multiple modalities, incorporating input such as segmentation, depth maps, and human joint positions. Another direction in this field is learning the conditional variable itself. Models like DiffAE [51], SODA [31], and Abstreiter et al. [2] train an encoder to produce a low-dimensional latent variable to assist the generation process; these works also demonstrate that such an encoder can learn meaningful image representations.

Generation with semantic visual representations. State-of-the-art generative models, such as diffusion and autoregressive models, can be viewed as denoising autoencoders that inherently learn meaningful data representations. Tang et al. [66], Yang and Wang [73], Zhang et al. [79] demonstrate that diffusion models capture semantic visual representations, which are directly applicable to various downstream tasks [4, 35]. Zhang and Maire [78] highlight that a discriminator in a GAN can learn useful representations. [32, 41] show that incorporating representation learning objectives into the generative framework can further enhance generation quality. Hu et al. [30], Li et al. [42], Wang et al. [70] leverage semantic representations learned by the encoder to further improve generation quality.

## 3. Method

We employ a structured approach to capture hierarchical semantic features for image generation.

## 3.1. Preliminary: Diffusion models

As a generative framework, diffusion models [27, 62, 64] consist of both a forward (diffusion) process and a backward process, each spanning over $T$ steps. Let $\mathbf { x } \in \mathbb { R } ^ { d }$ denote the original data sample. The forward process defines a sequence of latent variables $\{ \mathbf { x } ^ { ( t ) } \} _ { t = 1 } ^ { T }$ obtained by sampling from a Markov process parameterized as $q \left( \mathbf { \bar { x } } ^ { ( t ) } \mid \mathbf { \bar { x } } ^ { ( t - 1 ) } \right) \ : =$ $\mathcal { N } ( \mathbf { x } ^ { ( t ) } ; \alpha ^ { ( t ) } \mathbf { x } , \beta ^ { ( t ) } \mathbf { I } )$ , where $\alpha ^ { ( t ) }$ and $\beta ^ { ( t ) }$ are hyperparameters of the noise scheduler, ensuring that the signal-to-noise ratio (SNR) decreases as t increases.

In the backward process, the model $D _ { \theta }$ is tasked with estimating the transition probability $p ( \mathbf { x } ^ { ( t - 1 ) } | \mathbf { x } ^ { ( t ) } )$ and generating data through the process $\begin{array} { r } { \prod _ { t = 1 } ^ { T } p _ { \theta } \big ( \mathbf { x } ^ { ( t - 1 ) } | \mathbf { x } ^ { ( t ) } \big ) p \big ( \mathbf { x } ^ { ( T ) } \big ) } \end{array}$ where $p _ { \theta } ( \mathbf { x } ^ { ( t - 1 ) } | \mathbf { x } ^ { ( t ) } )$ represents the transition probability estimated by $D _ { \theta }$ . It is trained by optimizing the Variational Lower Bound (VLB) [37]:

$$
\mathcal { L } _ { \mathrm { V L B } } = \sum _ { t = 1 } ^ { T } D _ { \mathrm { K L } } ( q ( \mathbf { x } ^ { ( t - 1 ) } | \mathbf { x } ^ { ( t ) } , \mathbf { x } )  p _ { \theta } ( \mathbf { x } ^ { ( t - 1 ) } | \mathbf { x } ^ { ( t ) } ) ) .\tag{1}
$$

Here $q \left( \mathbf { x } ^ { ( t - 1 ) } | \mathbf { x } ^ { ( t ) } , \mathbf { x } \right)$ can be derived using Bayes’ rule as $q \left( \mathbf { x } ^ { ( t ) } | \mathbf { \dot { x } } ^ { ( t - 1 ) } , \mathbf { x } \right) q \left( \mathbf { \dot { x } } ^ { ( t - 1 ) } | \mathbf { x } \right) / q \left( \mathbf { x } ^ { ( t ) } | \mathbf { x } \right)$ . Minimizing the RHS of Eqn.1 can be simplified as training $D _ { \theta }$ to estimate the noise $\bar { \mathbf { \Psi } } \in \mathbb { R } ^ { d }$ sampled from $\mathcal { N } ( 0 , { \bf I } ) [ 2 8 ]$

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { d i f f u s i o n } } = \mathbb { E } _ { \epsilon ^ { ( t ) } , t } \| D _ { \theta } ( \alpha ^ { ( t ) } \mathbf { x } + \beta ^ { ( t ) } \epsilon ^ { ( t ) } , t ) - \epsilon ^ { ( t ) } \| _ { 2 } . } \end{array}
$$

## 3.2. Nested diffusion models

We propose a hierarchical generative framework with L levels, where each level is instantiated as a diffusion model $D _ { \theta _ { l } }$ . As illustrated in Figure 3, the generative model at level l produces latent variable $\mathbf { z } _ { l } \in \mathbb { R }$ with $p _ { \theta _ { l } } ( \mathbf { z } _ { l } | \mathbf { z } _ { > l } )$ , where ${ \bf z } _ { > l } : = \{ { \bf z } _ { m } \} _ { m > l }$ represents latent variables from higher levels. The feature dimension of $\mathbf { z } _ { l } \in \mathbb { R } ^ { d _ { l } }$ decreases as l increases, such that $d _ { l } > d _ { l + 1 }$ . At the shallowest level when $l = 1$ , the latent variables correspond directly to the data samples; that is, $\mathbf { z } _ { 1 } = \mathbf { x }$

Diffusion with semantic hierarchy. Our approach explicitly guides the generative process to align with a semantic hierarchy. Here, the top tier (larger l) denotes higher levels of semantic abstraction, whereas the lower tier (smaller l) represents detailed, fine-grained information. This is essential for preserving image semantic structures and producing realistic samples in generative models.

Non-Markovian generation. At each hierarchical level l, we follow the diffusion model framework and task $D _ { \theta _ { l } }$ to estimate the transition probability $p _ { \theta _ { l } } ( \mathbf { z } _ { l } ^ { ( t - 1 ) } | \mathbf { z } _ { l } ^ { ( t ) } , \mathbf { z } _ { > l } )$ At layer l, we assume a non-Markovian generation process where $D _ { \theta _ { l } }$ depends on the entire set of latent variables $\mathbf { z } _ { > l }$ estimated from the preceding hierarchies.

We optimize the hierarchal diffusion models using the following objectives:

$$
\sum _ { l = 1 } ^ { L - 1 } \sum _ { t = 1 } ^ { T } D _ { \mathrm { K L } } \left( q \left( \mathbf { z } _ { l } ^ { ( t - 1 ) } | \mathbf { z } _ { l } ^ { ( t ) } , \mathbf { z } _ { l } , \mathbf { x } \right) \Big | \Big | p _ { \theta _ { l } } \left( \mathbf { z } _ { l } ^ { ( t - 1 ) } | \mathbf { z } _ { l } ^ { ( t ) } , \mathbf { z } _ { > l } \right) \right)\tag{2}
$$

$$
+ \sum _ { t = 1 } ^ { T } D _ { \mathrm { K L } } \left( q \left( \mathbf { z } _ { L } ^ { ( t - 1 ) } | \mathbf { z } _ { L } ^ { ( t ) } , \mathbf { z } _ { L } , \mathbf { x } \right) \Big | \Big | p _ { \theta _ { l } } \left( \mathbf { z } _ { L } ^ { ( t - 1 ) } | \mathbf { z } _ { L } ^ { ( t ) } \right) \right) .
$$

Comparing to hierarchical VAEs which also includes hierarchical latent variables $\{ \mathbf { z } _ { l } \} _ { l = 1 } ^ { L } .$ , we enhance sampling capability by integrating the diffusion model and introducing an additional set of latent variables $\{ \mathbf { z } _ { l } ^ { ( t ) } \} _ { t = 0 } ^ { T }$ for each level l. This modification allows for multiple sampling steps, as opposed to the single forward pass used in hierarchical VAEs, leading to a more accurate prior estimation. This improvement is vital in hierarchical generative systems, where mismatches between the posterior and prior distributions can compound across levels, potentially degrading the quality of the generated output.

![](images/639cfa6a29899e7a9c75f04fdab4d47c719df84fbe906d6257f305d7591df6a1.jpg)  
Image  <sub>2</sub> = 0 $\sigma _ { 2 } = 0 . 5$

![](images/d4a8ed2ea1e7635c5e0c28ccbec9316d9054c8462305e0f3a6fc70534d42f8f4.jpg)  
Image  
 <sub>2</sub> = 0  
 <sub>2</sub> = 0.5  
Figure 4. Feature compression via Gaussian noise. For a twolevel hierarchical generator $( L = 2 )$ , we generate images conditioned on an oracle CLIP feature z<sub>2</sub>, inferred from input images, with feature channels reduced from 512 to 256 dimensions via SVD. Without noise $( \sigma _ { 2 } ~ = ~ 0 )$ added to $\mathbf { z } _ { 2 } .$ , the generator $D _ { \theta _ { 1 } }$ degenerates to an autoencoder that nearly reconstructs the input; adding Gaussian noise $( \sigma _ { 2 } = 0 . 5 ) \mathrm { t o } { \bf z } _ { 2 }$ limits feature information, allowing for generation of new content.

## 3.3. Hierarchical latents & progressive compression

Extraction of hierarchical features. We construct $\{ \mathbf { z } _ { l } \} _ { l = 1 } ^ { L }$ from a pre-trained visual encoder and keep those frozen during training. Most visual encoders, denoted as $\pmb { { \cal E } } ( \mathbf { x } ) = \mathbf { z }$ map an input image x to a vector z. To build a hierarchical representation, we propose running E on image patches. Let $C ( \mathbf { x } , M ) = \{ \mathbf { x } _ { m } \} _ { m = 1 } ^ { M ^ { 2 } }$ represent the cropping operation that splits the image $\mathbf { x } \in \mathbb { R } ^ { H \times W \times 3 }$ into $M ^ { 2 }$ non-overlapping square patches $\mathbf { x } _ { m } \in \mathbb { R } ^ { \frac { H } { M } \times \frac { W } { M } \times 3 }$ . We build a latent variable $\mathbf { z } _ { l } = { \pmb { E } } \left( C \left( \mathbf { x } , { \pmb { L } } - l + 1 \right) \right)$ over image patches. As patch size decreases, the visual representation transitions from capturing global structures to more localized features.

Feature compression via Gaussian noise. Our construction of $\{ \mathbf { z } _ { l } \} _ { l = 1 } ^ { L }$ contrasts with hierarchical VAEs, which use compression objectives to learn hierarchical latent variables but often face high variance issues, as noted in prior work [11, 50, 68]. While we address instability, our design presents a new challenge: representations from the visual encoder tend to be highly informative, allowing the generative model to reconstruct the input image accurately, which can cause the generator to behave like an autoencoder.

We show this effect in Figure 4: when conditioning on the oracle CLIP visual features, the diffusion model could nearly reconstruct the input with a delta distribution: $p _ { \theta _ { L } } ( \mathbf { z } _ { 0 } | \mathbf { z } _ { L } )$ ≈ $\delta ( \mathbf { z } _ { 0 } )$ , essentially “bypassing” the middle levels $p _ { \theta _ { l } } ( \mathbf { z } _ { l } | \mathbf { z } _ { l + 1 } )$ 1 in a hierarchical system. Consequently, we must reduce the information contained in z to ensure each hierarchical level contributes meaningfully to the generation process; our procedure is as follows.

Channel reduction via singular value decomposition. In our patch-based approach, the channel number of the latent variable quadratically increases as we move down the hierarchy, with $\mathbf { z } _ { l } \in \mathbb { R } ^ { ( \bar { L } - l + 1 ) ^ { 2 } \times d }$ , quickly making the information overcomplete for generation. To address this, we propose trimming the feature channels. Specifically, we apply singular value decomposition (SVD) to the encoder’s feature vector, preserving only the leading $d / ( L - l + 1 )$ channels, resulting in $\mathbf { z } _ { l } \in \mathbb { R } ^ { ( L - l + 1 ) \times d }$ , with channel linearly increasing over level of hierarchy.

Information reduction through Gaussian noise. As shown in Figure $^ { 4 , }$ channel reduction alone is insufficient to prevent the diffusion model from degrading into an autoencoder. To further increase the abstraction level, we introduce Gaussian noise to $\mathbf { z } _ { l } .$ , represented as $\hat { \mathbf { z } } _ { l } \sim$ $\mathcal { N } ( \mathbf { z } _ { l } , \sigma _ { l } ^ { 2 } \mathbf { I } )$ , where $\sigma _ { l }$ is a fixed constant based on the hierarchical level. This process limits the amount of information that can be transmitted, measured by the KL divergence $D _ { K L } \left( \mathcal { N } \left( \mathbf { z } _ { l } , \sigma _ { l } ^ { 2 } \right) , \mathcal { N } \left( \mathbf { 0 } , \mathbf { I } \right) \right)$ . A large variance $\sigma _ { l } ^ { 2 }$ substantially limits the information capacity. In our experiments, adding Gaussian noise proved essential in preserving and enhancing generation quality as the number of hierarchical levels increased. We only add Gaussian noise during training; it is not present during generation.

With these approaches, our loss function is:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { n e s t e d _ { - } d i f f u s i o n } } = } \\ & { \sum _ { l = 1 } ^ { L - 1 } \mathbb { E } _ { \hat { \mathbf { z } } _ { > l } , \epsilon _ { l } ^ { ( t ) } , t } \lVert D _ { \theta _ { l } } ( \boldsymbol { \alpha } ^ { ( t ) } \mathbf { z } _ { l } + \boldsymbol { \beta } ^ { ( t ) } \boldsymbol { \epsilon } _ { l } ^ { ( t ) } , \hat { \mathbf { z } } _ { > l } , t ) - \boldsymbol { \epsilon } _ { l } ^ { ( t ) } \rVert _ { 2 } } \\ & { ~ + \mathbb { E } _ { \epsilon _ { L } ^ { ( t ) } , t } \lVert D _ { \theta _ { L } } ( \boldsymbol { \alpha } ^ { ( t ) } \mathbf { z } _ { L } + \boldsymbol { \beta } ^ { ( t ) } \boldsymbol { \epsilon } _ { L } ^ { ( t ) } , t ) - \boldsymbol { \epsilon } _ { L } ^ { ( t ) } \rVert _ { 2 } , } \end{array}\tag{3}
$$

where $\epsilon _ { l } ^ { ( t ) } \in \mathbb { R } ^ { d _ { l } }$ denotes noise sampled at each level.

## 3.4. Diffusion with semantic consistent neighbors

Diffusion models are highly related to the mean-shift iterations [10, 12], as highlighted by the analysis [36, 63, 69]. Specifically, assuming hierarchical dependency $p ( \mathbf { z } _ { l } ^ { ( t ) } | \mathbf { z } _ { > l } ) = \mathbb { E } _ { \mathbf { z } _ { l } } p ( \mathbf { z } _ { l } ^ { ( t ) } | \mathbf { z } _ { l } , \mathbf { z } _ { > l } ) = \mathbb { E } _ { p ( \mathbf { z } _ { l } | \mathbf { z } _ { > l } ) } p ( \mathbf { z } _ { l } ^ { ( t ) } | \mathbf { z } _ { l } )$ , an optimal denoiser $\boldsymbol { D } _ { l } ^ { * }$ for Eqn. 3 can be expressed in closedform as follows [36, 63]:

$$
D _ { l } ^ { * } ( \mathbf { z } _ { l } ^ { ( t ) } | \mathbf { z } _ { > l } ) = \frac { \int _ { \mathbf { z } _ { l } } p ( \mathbf { z } _ { l } ^ { ( t ) } | \mathbf { z } _ { l } ) p ( \mathbf { z } _ { l } | \mathbf { z } _ { > l } ) \mathbf { z } _ { l } } { \int _ { \mathbf { z } _ { l } } p ( \mathbf { z } _ { l } ^ { ( t ) } | \mathbf { z } _ { l } ) p ( \mathbf { z } _ { l } | \mathbf { z } _ { > l } ) } .\tag{4}
$$

This suggests that the optimal solution is the weighted data point $\mathbf { z } _ { l }$ , based on the similarity between $\mathbf { z } _ { l }$ and $\mathbf { z } _ { l } ^ { ( t ) }$ . Therefore, the structure of $\mathbf { z } _ { l }$ significantly impacts the quality of optimal denoiser. Ideally, the neighbor of both $\mathbf { z } _ { l } ^ { ( t ) }$ and $\mathbf { z } _ { l }$ should have similar semantic structures. In Figure 5, we attempt to visualize the structure of $\mathbf { z } _ { l }$ via nearest neighbor images with CLIP features or VAE features. Unlike VAE, which focuses on low-level textures and results in unrelated neighboring images, CLIP yields semantically similar images and better generation quality in our experiments.

![](images/65e248739c86d3d54ad9995ca2ae0a1001f4d18784498453006a75058c9ac76c.jpg)  
Figure 5. Visualization of K-Nearest Neighbors (KNN) with different sources of latent features. For each input image, we display neighboring images, based on features extracted from two types of visual representations: CLIP representations, and VAE bottlenecks. Unlike the VAE, which focuses on low-level visual structures, CLIP emphasizes semantic representations, yielding more meaningful nearest neighbors. Our experiments demonstrate that running a diffusion model on a latent space with well-structured neighbors is essential for enhancing generation quality.

## 4. Experiments

We present the setup and results of our experiments, where we evaluate the performance of our nested diffusion model across various tasks. Our primary focus is to explore the model’s effectiveness in both conditional and unconditional image generation scenarios using the COCO-2014 [43] and ImageNet-1K datasets [56].

## 4.1. Experimental Setup

Nested diffusion models. We utilize U-ViT [3], a ViTbased U-Net model with an encoder-decoder architecture, for $D _ { \theta _ { l } }$ . This model employs skip connections and performs diffusion in the latent space of a pre-trained VAE, reducing the input size from $2 5 6 \times 2 5 6 \times 3$ to $3 2 \times 3 2 \times 4 .$ , which enables efficient handling of high-resolution images. We customize the network configurations to make it aligned with the standard ViT-Base model: The transformer model we use consists of 12 blocks, with the base channel dimension set to 768, and each attention block comprises 12 attention heads. We use the default diffusion scheduler, sampler, and training hyperparameters from U-ViT [3] for ImageNet-1k and COCO respectively.

To construct our nested diffusion model, we use the same network configuration for each hierarchical level. At higher hierarchical levels, there is a progressive reduction in the dimensionality of $\mathbf { z } _ { l } .$ , which leads to minimal extra computational cost even though the number of parameters increases.

To incorporate conditional features from higher levels, we simply treat $\hat { \mathbf { z } } _ { l + 1 }$ as an additional input token and append it to $\mathbf { z } _ { l }$ before feeding into the ViT. During training, we randomly (10% chance) replace the $\hat { \mathbf { z } } _ { l + 1 }$ with a learnable empty token to facilitate classifier-free guidance (CFG) [26].

<table><tr><td></td><td colspan="3">w/o CFG</td><td colspan="3">w/ CFG</td></tr><tr><td>Model</td><td> $\sigma _ { 2 } = 0 . 0$ </td><td>0.5</td><td>1.0</td><td> $\sigma _ { 2 } = 0 . 0$ </td><td>0.5</td><td>1.0</td></tr><tr><td> $\overline { { L = 1 } }$   $L = 1 ^ { * }$ </td><td>55.41 45.19</td><td>一</td><td>一</td><td>一</td><td>一</td><td>- -</td></tr><tr><td> $\overline { { L = 2 } }$ </td><td>19.32</td><td>20.66</td><td>27.40</td><td>6.59</td><td>7.19</td><td>8.69</td></tr><tr><td> $L = 3$ </td><td>20.34</td><td>19.00</td><td>23.37</td><td>6.77</td><td>6.34</td><td>6.98</td></tr><tr><td> $L = 4$ </td><td>17.67</td><td>15.14</td><td>16.27</td><td>5.79</td><td>5.54</td><td>5.89</td></tr><tr><td> $L = 5$ </td><td>19.04</td><td>11.88</td><td>11.05</td><td>7.68</td><td>5.36</td><td>5.03</td></tr></table>

(a) Unconditional image generation for ImageNet-1k, 256 256

<table><tr><td></td><td colspan="3">w/o CFG</td><td colspan="3">w/CFG</td></tr><tr><td>Model</td><td> $\sigma _ { 2 } = 0 . 0$ </td><td>0.5</td><td>1.0</td><td> $\sigma _ { 2 } = 0 . 0$ </td><td>0.5</td><td>1.0</td></tr><tr><td> $\overline { { L = 1 } }$   $L = 1 ^ { * }$ </td><td>31.13 19.74</td><td>一</td><td>-</td><td>13.75 7.18</td><td>一</td><td>- -</td></tr><tr><td> $L = 2$ </td><td>16.56</td><td>16.52</td><td>22.43</td><td>4.87</td><td>- 5.31</td><td>6.49</td></tr><tr><td> $L = 3$ </td><td>15.51</td><td>15.50</td><td>16.35</td><td>4.46</td><td>4.69</td><td>5.15</td></tr><tr><td> $L = 4$ </td><td>17.72</td><td>14.38</td><td>13.87</td><td>4.81</td><td></td><td>4.25</td></tr><tr><td> $L = 5$ </td><td>18.04</td><td>11.28</td><td>9.87</td><td>4.26</td><td>4.38 4.05</td><td>3.97</td></tr></table>

(b) Conditional image generation for ImageNet-1k, 256 256  
Table 1. We evaluate image generation quality using the Fréchet Inception Distance (FID) on the ImageNet-1k and we report the results w/o and w/ classifier free guidance (CFG). We benchmark our model across different noise levels and network depths $L .$ To determine the noise levels $\lbrace \sigma \iota \rbrace _ { l = 2 } ^ { L } ,$ , we use a top-down, greedy searching: for a model with depth L, we retain the optimal values $\{ \sigma _ { l } \} _ { l = 3 } ^ { L }$ from the shallower model $< L$ and only tune the newly added level, $\sigma _ { 2 } .$ The generation quality improves with L increases and adding Gaussian noise is crucial for better performance for deeper model. For comparison, we also provide baseline results for $L = 1 ^ { * }$ , a single-level model with increased parameters to match the GFLOPs of $L = 5$

We use the same architecture for both COCO and ImageNet-1k experiments and we train on COCO for 1000 epochs and ImageNet-1k for 200 epochs unless mentioned otherwise.

We follow the standard evaluation protocol to report the image generation quality with Fréchet inception distance (FID). For ImageNet-1k, we generate 50K images, 50 for each category, and compute the FID over the training set using the precomputed statistic provided by Dhariwal and Nichol [13]. For COCO-2014, when comparing to other approaches in Table 4, we follow previous literature to generate 30k images using text prompts from the validation set and compare the statistics against the validation set images.

Hierarchical latent variables. We build hierarchical latent variables $\{ \mathbf { z } _ { l } \} _ { l = 2 } ^ { L }$ using a pre-trained visual encoder and directly use VAE bottleneck features as our bottom level $\mathbf { z } _ { 1 }$ . For the ImageNet experiments, we extract visual features from MoCo-v3 [9] (ViT-B/16), a leading self-supervised visual representation learner. For the COCO experiments, we use CLIP [52] (ViT-B/16), a multi-modal encoder that aligns visual and textual representations.

We set the top-level latent representation to a fixed dimension: ${ \bf z } _ { L } ~ \in ~ \mathbb { R } ^ { 2 5 6 }$ , by trimming additional feature channels via singular value decomposition (SVD). Consequently, we construct hierarchical latent variables with shapes: $\mathbf { \bar { z } } _ { L } \in \mathbb { R } ^ { 1 \times 2 5 6 } , \mathbf { z } _ { L - 1 } \in \mathbb { R } ^ { 4 \times 1 2 8 } , \mathbf { z } _ { L - 2 } \in \mathbb { R } ^ { 1 6 \times 6 4 }$ and so on.

<table><tr><td>Model</td><td>Iter.</td><td>GFlops</td><td>w/CFG</td><td>w/o CFG</td></tr><tr><td>Conditional Generation</td><td></td><td></td><td></td><td></td></tr><tr><td>DiT-XL/2 [49]</td><td>7M</td><td>118.6</td><td>2.27</td><td>9.62</td></tr><tr><td>DiT-XL/2 [49]</td><td>400k</td><td>118.6</td><td>n/a</td><td>19.95</td></tr><tr><td>DiT-XL/2 + REPA [76]</td><td>400k</td><td>118.6</td><td>n/a</td><td>12.30</td></tr><tr><td>DiT-XL/2 + REPA [76]</td><td>850k</td><td>118.6</td><td>n/a</td><td>9.60</td></tr><tr><td>U-ViT (Mid)</td><td>500k</td><td>26.8</td><td>13.75</td><td>31.13</td></tr><tr><td>U-ViT (Mid*)</td><td>500k</td><td>35.1</td><td>7.18</td><td>19.74</td></tr><tr><td>Ours (L = 5)</td><td>500k</td><td>34.0</td><td>3.97</td><td>9.87</td></tr><tr><td colspan="5">Unconditional Generation</td></tr><tr><td>U-ViT (Mid)</td><td>500k</td><td>26.8</td><td>n/a</td><td>55.41</td></tr><tr><td>U-ViT (Mid*)</td><td>500k</td><td>35.1</td><td>n/a</td><td>45.19</td></tr><tr><td>Ours (L = 5)</td><td>500k</td><td>34.0</td><td>5.03</td><td>11.05</td></tr></table>

Table 2. Generation quality (FID) on ImageNet-1k 256 256. Our model significantly outperforms U-ViT (Mid) and U-ViT (Mid\*), which are $L = 1$ and L = 1 from Table 1. We show in grayscale results requiring substantially more training iterations or more computation as measured in GFlops.

<table><tr><td rowspan=1 colspan=1>γ</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0.1</td><td rowspan=1 colspan=1>0.3</td><td rowspan=1 colspan=1>0.5</td><td rowspan=1 colspan=1>∞</td></tr><tr><td rowspan=1 colspan=1> $\overline { { \mathrm { ~ w ~ } / \mathrm { ~ C F G } } }$ </td><td rowspan=1 colspan=1>5.09</td><td rowspan=1 colspan=1>5.05</td><td rowspan=1 colspan=1>5.03</td><td rowspan=1 colspan=1>5.33</td><td rowspan=1 colspan=1>11.21</td></tr><tr><td rowspan=1 colspan=1>w/o CFG</td><td rowspan=1 colspan=1>14.19</td><td rowspan=1 colspan=1>13.53</td><td rowspan=1 colspan=1>12.51</td><td rowspan=1 colspan=1>11.93</td><td rowspan=1 colspan=1>11.27</td></tr></table>

Table 3. We investigate the effects of different values of $\gamma ,$ which controls the level of noise added to $\mathbf { z } _ { l }$ during generation through the term $( t / T ) ^ { \gamma } \sigma _ { l }$ at diffusion steps t, as described in Sec. 4.2. Here, we present the generation results (FID) using a five-level nested diffusion model $( L = 5 )$ on ImageNet-1K. When $\gamma = 0 ,$ the model applies $\sigma _ { l }$ (consistent with the noise level during training) for generation, while $\gamma = \infty$ corresponds to no noise being added to $\mathbf { z } _ { l }$ during generation.

Patchification enables us to extract features at various resolutions from the encoder. However, this approach can encounter limitations when patches become too small to carry meaningful visual patterns. For cases requiring patches’ spatial resolution smaller than $6 4 \times 6 4$ , we use the feature maps from the encoder’s backbone instead. For example, with $L = 5 , \mathbf { z } _ { 2 } \in \mathbb { R } ^ { 6 4 \times 3 2 }$ is built by reducing the feature map of the encoder, with shape 14  14  768 (height width channel numbers) to $8 \times 8 \times 3 2 = 6 4 \times 3 2$ through spatial pooling and channel reduction.

Efficient training and parameter search for hierarchical models. In our design, we allow each $D _ { \theta _ { l } }$ to have a unique $\sigma _ { l } ,$ , providing flexibility, but also adding complexity to hyperparameter search. To streamline this process, we use a hierarchical local search strategy: for a L-level model, we retain the optimal noise levels $\{ \sigma _ { l } \} _ { l = 3 } ^ { L }$ from the $L - 1$ level model and search only for the newly introduced level $\sigma _ { 2 } .$ . An additional benefit of this approach is that we can directly reuse the parameters $\{ D _ { \theta _ { l } } \} _ { l = 3 } ^ { L }$ from shallower models due to the consistent model configuration, which means that we only need to train $\{ D _ { \theta _ { l } } \} _ { l = 1 } ^ { 2 }$

![](images/add25b88c068145fbc00a532e4353953751c94b3b44b003e39f03471d1d94af1.jpg)  
Figure 6. Visualization of unconditional image generation on ImageNet-1K. We present visualizations of images generated by hierarchica diffusion models containing from 2 to 5 levels, demonstrating that image quality improves as the depth of the hierarchy increases.

## 4.2. Generation with noisy hierarchical features

During training, we introduce noise to hierarchical features $\mathbf { z } _ { l }$ by sampling $\hat { \mathbf { z } } _ { l } \sim \mathcal { N } ( \mathbf { z } _ { l } , \sigma l ^ { 2 } \mathbf { I } )$ , as outlined in Section 3.3, to enhance information compression. By default, this noise is removed during generation to ensure consistent conditional signals. However, we observe that this approach can be detrimental to classifier-free guidance $( \mathrm { C F G } ) ,$ likely due to a distributional shift between training and testing. To address this, we propose a gradual decay of $\sigma _ { l }$ across diffusion steps t during generation: $\hat { \mathbf { z } } _ { l } ^ { ( t ) } \sim \mathcal { N } ( \mathbf { z } _ { l } , ( t / T ) ^ { \gamma } \cdot \boldsymbol { \sigma } _ { l } ^ { 2 } \mathbf { I } )$ , where $0 \leq$ $( t / T ) ^ { \gamma } \leq 1$ progressively reduces $\sigma _ { l } ,$ and the scalar $\gamma \geq 0$ controls the decay speed. This design shares the similar spirit of Sadat et al. [57], which adds noise to the ground truth image label to encourage diversity in the generation process. In contrast, our approach focuses on maintaining consistency between the training and testing phases.

Table 3 presents the results for various $\gamma$ values in a $L = 5$ nested diffusion model for unconditional generation on ImageNet-1K. Setting $\gamma = 0$ applies the same noise level $\sigma _ { l }$ used during training, while $\gamma = \infty$ eliminates noise during generation. In experiments with CFG, adding noise to $\mathbf { z } _ { l }$ is crucial, and our proposed scheme improves generation quality compared to the baseline of using the training noise level $( \gamma = 0 )$ . Without CFG, the best results are achieved by omitting noise from $\mathbf { z } _ { l }$ . For subsequent experiments, we use $\gamma = 0 . 3$ in CFG scenarios and $\gamma = \infty$ in non-CFG cases.

## 4.3. Benchmarking generation quality

We present our primary results for ImageNet-1k in Table 1, including the choices of model depth L, the noise level, and generation w/ or w/o CFG.

Improved performance with more hierarchy levels $L .$ Compared to the baseline model, our nested diffusion models demonstrate enhanced image quality as we deepen the hierarchical structure by increasing depth L. Specifically, our five-level generation significantly outperforms the baseline with $L = 1$ in both conditional generation $( 3 1 . 1 3  9 . 8 7 )$ and unconditional generation $( 5 5 . 4 1  1 1 . 0 5 )$ . Notably, our unconditional generator with $L = 5$ surpasses the conditional baseline generation model with $L = 1$ , achieving scores of $3 1 . 1 3  1 1 . 0 5$

We also present results using CFG, which directs the generated output toward conditional features, enhancing quality at the expense of reduced output diversity. The influence of CFG is controlled by the parameter w. Our nested diffusion models produce hierarchical representations; ideally, the top level benefits from stronger CFG to enforce semantic abstraction, while lower levels should reduce the CFG influence to promote diversity in visual details. To achieve this, we adopt a straightforward approach by specifying a set of decaying CFG weights, $\{ w _ { i } \} = [ 0 . 5 , 0 . 4 , 0 . 3 , 0 . 2 , 0 . 1 ]$ , and selecting $\{ w _ { i } \} _ { i = 1 : L }$ for our L-level nested diffusion models, with higher CFG weights assigned to the top levels. We conducted a limited hyperparameter search over the choices of CFG weights and found that this strategy yields better results than using a constant CFG weight for ImageNet conditional generation. For unconditional generation, we use a constant $w = 0 . 8 ,$ as it produces better results.

With CFG, we demonstrate that increasing the hierarchy depth L further improves image quality, with our $L = 5$ model achieving FID scores of 3.97 for conditional generation and 5.03 for unconditional generation, significantly outperforming the baseline at $L = 1 \left( 1 3 . 7 5 \right)$ .

As illustrated in Figure 2, our hierarchical models achieve computational efficiency by constructing hierarchical features with decreased spatial dimensions, thereby reducing computational expense at higher tiers. In particular, our system with $L = 5$ results in only a 27.00% increase in the computational load measured in GFlops compared to the baseline $L = 1$ , while achieving a marked decrease in FID by 68.29%, as detailed in Appendix A.

Impact of $\sigma _ { l } .$ . As described in Section 3.3, $\sigma _ { l }$ plays a key role in enabling our design to scale effectively with more hierarchy levels. It regulates the information conveyed by the conditional latent variable $\mathbf { z } _ { l } .$ , ensuring that the hierarchical model does not bypass intermediate levels. We validate this design in Table 1, where the performance gap between $\sigma _ { 2 } =$ 0 and non-zero $\sigma _ { 2 }$ widens as the model depth L increases. This can be attributed to the fact that as L grows, more feature elements are included, increasing the likelihood of the model relying primarily on the lower-level features for generation, thus neglecting higher-level features. Higher noise levels help counteract this effect: with $L = 5 ,$ , setting $\sigma _ { 2 } = 1 . 0$ achieves FID of 11.05 and 9.87, outperforming $\sigma _ { 2 } = 0$ , which yields FID of 18.04 and 19.04 for conditional and unconditional generation, respectively.

<table><tr><td>Model</td><td>FID</td><td>Training Dataset</td></tr><tr><td>Huge Model, Extra Data GLIDE [47]</td><td></td><td>DALL-E (250M)</td></tr><tr><td>DALL-E 2 [53] Imagen [58]</td><td>12.24 10.39 7.27</td><td>DALL-E (250M) Internal Data/LAION (860M)</td></tr><tr><td>Re-Imagen [8] CM3Leon-7B [75] Parti-20B [74] COCO Data Only, w/ CFG</td><td>5.25 4.88 3.22</td><td>KNN-ImageText/COCO (50M) Internal Data (350M) LAION/FIT/JFT/COCO (4.8B)</td></tr><tr><td>VQ-Diffusion [19] Friro [15] U-ViT [3]</td><td>13.86 8.97 5.42</td><td>COCO (83K) COCO (83K)</td></tr><tr><td>Ours L = 2</td><td>4.72</td><td>COCO (83K) COCO (83K)</td></tr><tr><td>Ours  $L = 3$ </td><td>5.92</td><td>COCO (83K)</td></tr><tr><td>COCO Data Only, w/o CFG</td><td></td><td></td></tr><tr><td> $\mathrm { U - V i T } \left[ 3 \right]$ </td><td></td><td></td></tr><tr><td>Ours  $L = 2$ </td><td>14.98 8.15</td><td>COCO (83K) COCO (83K)</td></tr></table>

Table 4. Comparison of text-to-image generation on COCO-2014. The upper half shows larger models trained with more data and the bottom half shows the models that are only trained on training split of COCO. When trained only on COCO, our models (with $\sigma _ { 2 } = 0 . 5 )$ outperform all the compared methods. It is worth noting that we’re better than most of the larger models, shown on the top half.

Comparison to other methods. We evaluate our models on class-conditional ImageNet $2 5 6 \times 2 5 6$ generation tasks without CFG, with results presented in Table 2. We compare against DiT [49] variants and REPA [76], which aligns diffusion representations with a pre-trained visual encoder and is trained for 400K steps. Our model with $L = 5$ significantly outperforms these baselines by a substantial margin while also requiring fewer GFlops. Remarkably, our unconditional model even surpasses their conditional version.

Experiments on COCO. In addition to ImageNet-1K, we also evaluate our model on the COCO-2014 dataset to assess performance on complex scenes. Using visual features from the CLIP ViT-B/16 model and fixed noise levels $\{ \sigma _ { l } \} = 0 . 5$ Table 4 reports results, comparing our approach with large models trained on additional data sources. Without CFG, our $L = 3$ system achieves state-of-the-art performance, surpassing most larger models. For our models, we set the CFG weight to 1, following the default in Bao et al. [3]. When generating with CFG, we find that $L = 2$ delivers the best performance, even outperforming the 7-billion-parameter model, CM3Leon. Our $L = 3$ model performs worse than $L = 2 .$ , likely due to a suboptimal CFG weight.

<table><tr><td>Features</td><td>FID↓</td><td colspan="2">KNN Acc. ↑ Top1 Top5</td></tr><tr><td>None</td><td>14.98</td><td></td><td></td></tr><tr><td>MAE [25]</td><td>10.96</td><td>27.44</td><td>45.33</td></tr><tr><td>MoCo-v3 [9]</td><td>10.59</td><td>66.57</td><td>83.09</td></tr><tr><td>CLIP [52]</td><td>6.97</td><td>73.35</td><td>91.12</td></tr><tr><td>DINO [6]</td><td>6.78</td><td>75.86</td><td>91.17</td></tr></table>

Table 5. Results on COCO text-to-image generation with different visual representations. We compare the generation quality of a 3-level nested diffusion model, where $L = 3$ and $\left\{ \sigma _ { l } \right\} = 0 . 5 ,$ using various visual encoders to construct z<sub>l</sub>. We report the results without CFG. Additionally, we report the accuracy of a KNN classifier with $K = 2 0$ on ImageNet-1K to quantify feature quality. Our results indicate that better feature quality improves generation results.

## 4.4. Generation with different visual encoders

In Section 3.4, we discuss the importance of preserving neighbor structure for the target space of diffusion models. We quantitatively validate this claim in Table 5 by presenting results from a 3-level nested diffusion model $( L = 3 )$ with fixed $\{ \sigma _ { l } \} = 0 . 5 .$ , applied to text-to-image generation on the COCO dataset. We construct $\mathbf { z } _ { l }$ using various visual encoders, including MAE, MoCo-v3, CLIP, and DINO. For a fair comparison, we use the same encoder architecture (ViT-B with a patch size of 16) and ensure that all $\mathbf { z } _ { l }$ representations have the same feature dimension. Our results show that image generation quality consistently improves with better visual representations, as measured by KNN accuracy on the ImageNet-1k dataset with $K = 2 0$

## 5. Conclusion

We introduce nested diffusion models, a novel hierarchical generative framework utilizing a succession of diffusion models to generate images starting from low-dimensional semantic feature embeddings and proceeding to detailed image refinement. Unlike conventional single-level latent models and hierarchical models that use low-level feature pyramids, each level in our model is conditional on a more abstract semantic feature hierarchy. This distinctive design improves image structure preservation and maintains global consistency, enhancing generation quality with minimal extra computational expense. We showcase the scalability of our method through a deeper unconditional system, which significantly surpasses the performance of a conditional generation baseline.

## Acknowledgements

RJ and RW gratefully acknowledge the support of the DOE grant DE-SC002223 and NSF grant DMS-2023109.

## References

[1] Rameen Abdal, Peihao Zhu, Niloy J Mitra, and Peter Wonka. Styleflow: Attribute-conditioned exploration of stylegangenerated images using conditional continuous normalizing flows. TOG, 2021.

[2] Korbinian Abstreiter, Sarthak Mittal, Stefan Bauer, Bernhard Schölkopf, and Arash Mehrjou. Diffusion-based representation learning. arXiv:2105.14257, 2021.

[3] Fan Bao, Shen Nie, Kaiwen Xue, Yue Cao, Chongxuan Li, Hang Su, and Jun Zhu. All are worth words: A vit backbone for diffusion models. In CVPR, 2023.

[4] Dmitry Baranchuk, Ivan Rubachev, Andrey Voynov, Valentin Khrulkov, and Artem Babenko. Label-efficient semantic segmentation with diffusion models. arXiv:2112.03126, 2021.

[5] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-toend object detection with transformers. In ECCV, 2020.

[6] Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In ICCV, 2021.

[7] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In ICML, 2020.

[8] Wenhu Chen, Hexiang Hu, Chitwan Saharia, and William W Cohen. Re-imagen: Retrieval-augmented text-to-image generator. arXiv:2209.14491, 2022.

[9] Xinlei Chen, Saining Xie, and Kaiming He. An empirical study of training self-supervised vision transformers. In ICCV, 2021.

[10] Yizong Cheng. Mean shift, mode seeking, and clustering. PAMI, 1995.

[11] Rewon Child. Very deep vaes generalize autoregressive models and can outperform them on images. arXiv:2011.10650, 2020.

[12] Dorin Comaniciu and Peter Meer. Mean shift analysis and applications. In ICCV, 1999.

[13] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. In NeurIPS, 2021.

[14] Xiaodan Du, Nicholas Kolkin, Greg Shakhnarovich, and Anand Bhattad. Generative models: What do they know? do they know things? let’s find out! arXiv:2311.17137, 2023.

[15] Wan-Cyuan Fan, Yen-Chun Chen, DongDong Chen, Yu Cheng, Lu Yuan, and Yu-Chiang Frank Wang. Frido: Feature pyramid diffusion for complex scene image synthesis. In AAAI, 2023.

[16] Ross B. Girshick, Jeff Donahue, Trevor Darrell, and Jitendra Malik. Region-based convolutional networks for accurate object detection and segmentation. PAMI, 2016.

[17] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and

Yoshua Bengio. Generative adversarial nets. In NeurIPS, 2014.

[18] Jiatao Gu, Shuangfei Zhai, Yizhe Zhang, Joshua M Susskind, and Navdeep Jaitly. Matryoshka diffusion models. In ICLR, 2023.

[19] Shuyang Gu, Dong Chen, Jianmin Bao, Fang Wen, Bo Zhang, Dongdong Chen, Lu Yuan, and Baining Guo. Vector quantized diffusion model for text-to-image synthesis. In CVPR, 2022.

[20] Bharath Hariharan, Pablo Andrés Arbeláez, Ross B. Girshick, and Jitendra Malik. Hypercolumns for object segmentation and fine-grained localization. In CVPR, 2015.

[21] Louay Hazami, Rayhane Mama, and Ragavan Thurairatnam. Efficientvdvae: Less is more. arXiv:2203.13751, 2022.

[22] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR, 2016.

[23] Kaiming He, Georgia Gkioxari, Piotr Dollár, and Ross B. Girshick. Mask R-CNN. In ICCV, 2017.

[24] Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In CVPR, 2020.

[25] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. Masked autoencoders are scalable vision learners. In CVPR, 2022.

[26] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv:2207.12598, 2022.

[27] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In NeurIPS, 2020.

[28] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In NeurIPS, 2020.

[29] Jonathan Ho, Chitwan Saharia, William Chan, David J Fleet, Mohammad Norouzi, and Tim Salimans. Cascaded diffusion models for high fidelity image generation. JMLR, 2022.

[30] Vincent Tao Hu, David W Zhang, Yuki M Asano, Gertjan J Burghouts, and Cees GM Snoek. Self-guided diffusion models. In CVPR, 2023.

[31] Drew A Hudson, Daniel Zoran, Mateusz Malinowski, Andrew K Lampinen, Andrew Jaegle, James L McClelland, Loic Matthey, Felix Hill, and Alexander Lerchner. Soda: Bottleneck diffusion models for representation learning. In CVPR, 2024.

[32] Ruoxi Jiang, Peter Y Lu, Elena Orlova, and Rebecca Willett. Training neural operators to preserve invariant measures of chaotic attractors. In NeurIPS, 2024.

[33] Bingyi Kang, Yang Yue, Rui Lu, Zhijie Lin, Yang Zhao, Kaixin Wang, Gao Huang, and Jiashi Feng. How far is video generation from world model: A physical law perspective. arXiv:2411.02385, 2024.

[34] Minguk Kang, Jun-Yan Zhu, Richard Zhang, Jaesik Park, Eli Shechtman, Sylvain Paris, and Taesung Park. Scaling up gans for text-to-image synthesis. In CVPR, 2023.

[35] Laurynas Karazija, Iro Laina, Andrea Vedaldi, and Christian Rupprecht. Diffusion models for zero-shot open-vocabulary segmentation. arXiv:2306.09316, 2023.

[36] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. In NeurIPS, 2022.

[37] Diederik Kingma, Tim Salimans, Ben Poole, and Jonathan Ho. Variational diffusion models. In NeurIPS, 2021.

[38] Diederik P Kingma. Auto-encoding variational bayes. arXiv:1312.6114, 2013.

[39] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton. Imagenet classification with deep convolutional neural networks. In NeurIPS, 2012.

[40] Alexander C Li, Mihir Prabhudesai, Shivam Duggal, Ellis Brown, and Deepak Pathak. Your diffusion model is secretly a zero-shot classifier. In CVPR, 2023.

[41] Tianhong Li, Huiwen Chang, Shlok Mishra, Han Zhang, Dina Katabi, and Dilip Krishnan. Mage: Masked generative encoder to unify representation learning and image synthesis. In CVPR, 2023.

[42] Tianhong Li, Dina Katabi, and Kaiming He. Selfconditioned image generation via generating representations. arXiv:2312.03701, 2023.

[43] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft COCO: Common objects in context. In ECCV, 2014.

[44] Qihao Liu, Zhanpeng Zeng, Ju He, Qihang Yu, Xiaohui Shen, and Liang-Chieh Chen. Alleviating distortion in image generation via multi-resolution diffusion models. arXiv:2406.09416, 2024.

[45] Jonathan Long, Evan Shelhamer, and Trevor Darrell. Fully convolutional networks for semantic segmentation. In CVPR, 2015.

[46] Eric Luhman and Troy Luhman. Optimizing hierarchical image VAEs for sample quality. arXiv:2210.10205, 2022.

[47] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv:2112.10741, 2021.

[48] George Papamakarios, Eric Nalisnick, Danilo Jimenez Rezende, Shakir Mohamed, and Balaji Lakshminarayanan. Normalizing flows for probabilistic modeling and inference. JMLR, 2021.

[49] William Peebles and Saining Xie. Scalable diffusion models with transformers. In ICCV, 2023.

[50] Adeel Pervez and Efstratios Gavves. Variance reduction in hierarchical variational autoencoders. 2020.

[51] Konpat Preechakul, Nattanat Chatthee, Suttisak Wizadwongsa, and Supasorn Suwajanakorn. Diffusion autoencoders: Toward a meaningful and decodable representation. In CVPR, 2022.

[52] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2021.

[53] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv:2204.06125, 2022.

[54] Scott Reed, Zeynep Akata, Xinchen Yan, Lajanugen Logeswaran, Bernt Schiele, and Honglak Lee. Generative adversarial text to image synthesis. In ICML, 2016.

[55] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022.

[56] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, et al. Imagenet large scale visual recognition challenge. IJCV, 2015.

[57] Seyedmorteza Sadat, Jakob Buhmann, Derek Bradley, Otmar Hilliges, and Romann M Weber. Cads: Unleashing the diversity of diffusion models through condition-annealed sampling. arXiv:2310.17347, 2023.

[58] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. In NeurIPS, 2022.

[59] Ayush Sarkar, Hanlin Mai, Amitabh Mahapatra, Svetlana Lazebnik, David A Forsyth, and Anand Bhattad. Shadows don’t lie and lines can’t bend! generative models don’t know projective geometry... for now. In CVPR, 2024.

[60] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. In ICLR, 2015.

[61] Casper Kaae Sønderby, Tapani Raiko, Lars Maaløe, Søren Kaae Sønderby, and Ole Winther. Ladder variational autoencoders. In NeurIPS, 2016.

[62] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv:2010.02502, 2020.

[63] Yang Song and Stefano Ermon. Improved techniques for training score-based generative models. In NeurIPS, 2020.

[64] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv:2011.13456, 2020.

[65] Yuhta Takida, Yukara Ikemiya, Takashi Shibuya, Kazuki Shimada, Woosung Choi, Chieh-Hsin Lai, Naoki Murata, Toshimitsu Uesaka, Kengo Uchida, Wei-Hsiang Liao, et al. Hq-vae: Hierarchical discrete representation learning with variational bayes. arXiv:2401.00365, 2023.

[66] Luming Tang, Menglin Jia, Qianqian Wang, Cheng Perng Phoo, and Bharath Hariharan. Emergent correspondence from image diffusion. NeurIPS, 2023.

[67] Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, and Liwei Wang. Visual autoregressive modeling: Scalable image generation via next-scale prediction. arXiv:2404.02905, 2024.

[68] Arash Vahdat and Jan Kautz. Nvae: A deep hierarchical variational autoencoder. In NeurIPS, 2020.

[69] Haochen Wang, Xiaodan Du, Jiahao Li, Raymond A Yeh, and Greg Shakhnarovich. Score jacobian chaining: Lifting pretrained 2d diffusion models for 3d generation. In CVPR, 2023.

[70] Xudong Wang, Trevor Darrell, Sai Saketh Rambhatla, Rohit Girdhar, and Ishan Misra. Instancediffusion: Instance-level control for image generation. In CVPR, 2024.

[71] Yufei Wang, Renjie Wan, Wenhan Yang, Haoliang Li, Lap-Pui Chau, and Alex Kot. Low-light image enhancement with normalizing flow. In AAAI, 2022.

[72] Jiarui Xu, Sifei Liu, Arash Vahdat, Wonmin Byeon, Xiaolong Wang, and Shalini De Mello. Open-vocabulary panoptic segmentation with text-to-image diffusion models. In CVPR, 2023.

[73] Xingyi Yang and Xinchao Wang. Diffusion model as representation learner. In ICCV, 2023.

[74] Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, et al. Scaling autoregressive models for content-rich text-to-image generation. arXiv:2206.10789, 2022.

[75] Lili Yu, Bowen Shi, Ramakanth Pasunuru, Benjamin Muller, Olga Golovneva, Tianlu Wang, Arun Babu, Binh Tang, Brian Karrer, Shelly Sheynin, et al. Scaling autoregressive multi-modal models: Pretraining and instruction tuning. arXiv:2309.02591, 2023.

[76] Sihyun Yu, Sangkyung Kwak, Huiwon Jang, Jongheon Jeong, Jonathan Huang, Jinwoo Shin, and Saining Xie. Representation alignment for generation: Training diffusion transformers is easier than you think. arXiv:2410.06940, 2024.

[77] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In ICCV, 2023.

[78] Xiao Zhang and Michael Maire. Structural adversarial objectives for self-supervised representation learning. arXiv:2310.00357, 2023.

[79] Xiao Zhang, David Yunis, and Michael Maire. Deciphering’what’and’where’visual pathways from spectral clustering of layer-distributed neural representations. In CVPR, 2024.

[80] Shengjia Zhao, Jiaming Song, and Stefano Ermon. Learning hierarchical features from generative models. arXiv:1702.08396, 2017.

[81] Jinghao Zhou, Chen Wei, Huiyu Wang, Wei Shen, Cihang Xie, Alan Yuille, and Tao Kong. ibot: Image bert pre-training with online tokenizer. arXiv:2111.07832, 2021.