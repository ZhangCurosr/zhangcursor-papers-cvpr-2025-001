# Linguistics-aware Masked Image Modeling for Self-supervised Scene Text Recognition

Yifei Zhang<sup>1,3</sup>, Chang Liu<sup>4</sup>, Jin Wei<sup>5</sup>, Xiaomeng Yang<sup>6</sup>, Yu Zhou<sup>2</sup> <sup>B</sup>, Can Ma<sup>1</sup>, Xiangyang Ji<sup>4</sup> <sup>1</sup>Institute of Information Engineering, Chinese Academy of Sciences <sup>2</sup>VCIP & TMCC & DISSec, College of Computer Science, Nankai University <sup>3</sup>School of Cyber Security, University of Chinese Academy of Sciences <sup>4</sup>Department of Automation and BNRist, Tsinghua University <sup>5</sup>Lenovo Research <sup>6</sup>College of Engineering, Northeastern University zhangyifei0115@iie.ac.cn, yzhou@nankai.edu.cn

![](images/75bf8d18f2654639a7cf1218fadc106a7702751db2c45da3fe096a02d1fed686.jpg)  
(a) Both vision and linguistics are crucial

![](images/355ba24c9765c5c8f1a8c16414ad40f14261fa0a4f3bf22b1462af22338a15f1.jpg)  
(b) Comparison of self-supervised methods

![](images/0b163d1cbe0b2cfafdc3de9b87c7eabff2b7581df924710f46df08941bbeba09.jpg)  
(c) Insufficient visual and linguistic information  
Figure 1. Illustration of our motivation. (a) Existing studies demonstrate that both visual and linguistic information are crucial for STR, as linguistic information can complement visual features. (b) Current self-supervised STR approaches, such as sequence contrastive learning (SeqCLR) and masked image modeling (MIM), primarily focus on local region alignment or rely on local visual information for reconstruction, often neglecting the integration of visual and linguistic information at a global level. To address this, our LMIM method channels linguistic information into the decoding process of MIM. (c) Attention maps reveal that SeqCLR lacks character structure information, while MIM emphasizes local regions. Our LMIM effectively captures the global context based on vision and linguistics. The red box in the input image indicates the query.

## Abstract

Text images are unique in their dual nature, encompassing both visual and linguistic information. The visual component encompasses structural and appearance-basedfeatures, while the linguistic dimension incorporates contextual and semantic elements. In scenarios with degraded visual quality, linguistic patterns serve as crucial supplements for comprehension, highlighting the necessity of integrating both aspects for robust scene text recognition (STR). Contemporary STR approaches often use language models or semantic reasoning modules to capture linguistic features, typically requiring large-scale annotated datasets. Self-supervised learning, which lacks annotations, presents

challenges in disentangling linguisticfeatures related to the global context. Typically, sequence contrastive learning emphasizes the alignment of local features, while masked image modeling (MIM) tends to exploit local structures to reconstruct visual patterns, resulting in limited linguistic knowledge. In this paper, we propose a Linguistics-aware Masked Image Modeling (LMIM) approach, which channels the linguistic information into the decoding process of MIM through a separate branch. Specifically, we design a linguistics alignment module to extract vision-independent features as linguistic guidance using inputs with different visual appearances. As features extend beyond mere visual structures, LMIM must consider the global context to achieve reconstruction. Extensive experiments on various benchmarks quantitatively demonstrate our state-ofthe-art performance, and attention visualizations qualitatively show the simultaneous capture of both visual and linguistic information. The code is available at https: //github.com/zhangyifei01/LMIM.

## 1. Introduction

Visual text plays a crucial role in various real-world applications and encompasses four key areas: text processing [37, 61, 82], text detection [12, 54, 55, 60], text recognition [45, 59, 67, 72], and text understanding [57, 58, 81, 86]. Among these, scene text recognition focuses on extracting textual content from detected text regions in images. Text images uniquely incorporate both visual and linguistic information: visual elements comprise structural and appearance-based features, while linguistic patterns encompass contextual and semantic components [51, 78]. The accuracy of text recognition is challenged by diverse backgrounds, varying lighting conditions, and multiple font styles. The integration of linguistic and visual information is essential for effective recognition, as demonstrated in Fig. 1 (a). Recent approaches have enhanced recognition accuracy by implementing language models [22, 49, 51] or semantic reasoning modules [7, 69, 78]. However, these methods typically require extensive labeled datasets, which are costly and often impractical due to specialized linguistic annotation requirements. Although synthetic data [28, 33] partially addresses this issue, a performance gap remains between models trained on synthetic versus real data [5, 34]. To bridge this gap, leveraging large-scale unannotated realworld data is of practical significance.

Self-supervised learning enables models to learn rich feature representations from real-world scenes [11, 20, 30, 38, 44, 77]. Although specialized self-supervised approaches for STR have emerged, they exhibit limitations in processing both visual and linguistic information at the global level, as illustrated in Fig. 1 (b) and (c). For example, sequence-to-sequence contrastive learning [1] and character-to-character distillation [26] focus solely on aligning local features at the fragment or character level without understanding the overall linguistic structure of the text. Meanwhile, MIM approaches, such as DiG [75] and MAERec [34], prioritize visual pattern reconstruction over coherent text, primarily utilizing local visual features while disregarding global context of characters within the same word. These limitations highlight the need for better integration of global linguistic information in self-supervised learning for STR, as seen in Fig. 1(c).

In this paper, we propose a simple yet effective Linguistics-aware Masked Image Modeling (LMIM) approach that directly addresses these challenges by concurrently capturing visual structure and linguistic information. While MIM traditionally reconstructs using local visual features, it inherently models intra-character and intercharacter associations, contributing to visual structure comprehension. Our LMIM incorporates linguistic information into the decoding process of MIM via a separate branch. Specifically, we design a linguistics alignment module that utilizes another image with identical text content but different visual appearances as input to the encoder, alongside the masked image, to extract vision-independent features as linguistic guidance. Due to the significant differences in visual appearance, the extracted features are no longer purely visual, requiring LMIM to utilize global context information for reconstruction. In this way, our method not only maintains the visual structure modeling capability of MIM through reconstruction, but also improves the global context awareness capability by leveraging linguistic information.

Extensive experiments demonstrate our method’s effectiveness, achieving state-of-the-art results on both English and Chinese benchmarks. Specifically, LMIM achieves 86.3% average accuracy on the Union14M benchmark and 97.0% on six common benchmarks using the ViT-S architecture. Additionally, to address the scarcity of public Chinese text datasets for pre-training, we collected 11 million cropped text images from the Web. Our method achieves impressive results on the Chinese benchmark as well, exhibiting a notable performance of 83.6% on Scene dataset and 82.0% on Web dataset. These results confirm that by explicitly incorporating linguistic information, LMIM ensures more accurate scene text recognition.

The contributions are summarized as follows:

• We propose a novel method called LMIM to integrate linguistic information into visual information modeling in a self-supervised manner, considering that both types of information are crucial for STR.

• Through the designed linguistics alignment module, we extract vision-independent features, compelling LMIM to exploit the global context for reconstruction.

• Given the widespread use of Chinese recognition but the lack of large-scale public datasets for pre-training, we contribute a dataset containing 11 million unlabeled Chinese text images for self-supervised STR research.

• Extensive experimental results demonstrate that LMIM achieves state-of-the-art performance on both English and Chinese benchmarks.

## 2. Related Works

## 2.1. Scene Text Recognition

STR presents distinct challenges due to its inherent duality of visual and linguistic information. Existing methods can be categorized into language-free and language-based approaches based on their utilization of language knowledge. Language-free methods rely exclusively on visual features for prediction. For instance, CRNN [59] and TRBA [4] employ CNNs for visual feature extraction and RNNs or attention mechanisms for sequence modeling. ViTSTR [3] proposes a straightforward framework that utilizes vision transformers. In addition, segmentation-based methods [25, 64] implement fully convolutional networks or specialized architectures for character-level segmentation. Languagebased methods leverage external or internal-learned language representations to assist in text recognition. The semantic reasoning network [78] incorporates a global semantic reasoning module to capture contextual information. Semantics enhanced encoder-decoder framework [51] explicitly integrates pre-trained language models to achieve semantic understanding. Subsequently, autonomous, bidirectional and iterative language modeling [22] implements iterative refinement for enhanced semantic reasoning. Visual language modeling network [69] unifies visual and linguistic capabilities within a single vision model using weakly supervised masking. Following these developments, numerous approaches have explored novel strategies for linguistic information integration [7, 18, 32, 49, 52, 70, 83].

While STR has achieved remarkable success through large-scale supervised training incorporating both visual and linguistic features, recent research has begun exploring self-supervised learning paradigms for more robust pretrained models [1, 75].

## 2.2. Self-supervised Learning

Self-supervised learning leverages the intrinsic data structure to design pretext tasks, which effectively utilizes largescale unlabeled data to capture generic representations [15, 24, 38, 44, 73, 84] These representations are important for a wide range of downstream tasks [21, 30, 39]. Recent advancements in contrastive learning (CL) and masked image modeling (MIM) have achieved significant success. The emergence of momentum contrast [30] and SimCLR [11] has spurred extensive research on CL [10, 13, 27, 68, 85, 88]. With the advent of vision transformer [17] and the inspiration of masked language modeling [14], concurrent works such as bidirectional encoder representation from image transformers [6], masked autoencoders [31], and Sim-MIM [74] have demonstrated the effectiveness of MIM. Since then, the research on MIM has dominated the selfsupervised learning community [2, 16, 19, 29, 47, 65, 71].

To enhance scene text recognition, existing research leverages the unique characteristics of text images to design self-supervised methods. Specifically, sequenceto-sequence contrastive learning (SeqCLR) [1] utilizes the sequential structure prior of data to design contrastive loss. Perceiving stroke-semantic context [40] improves SeqCLR by learning representation from low-level strokes. Character-to-character distillation [26] introduces character-level contrastive learning with a spectral character segmentation module. Similarity-aware normalization [43] decouples style and content features by reconstructing image patches guided by neighboring patches. Text-degradation invariant auto encoder [63] optimizes three pretext tasks $( i . e .$ , masking, blur and background noise) simultaneously. Dual masked autoencoder [53] decouples visual and semantic features to explicitly capture text semantics. Symmetric superimposition modeling [23] further emphasizes linguistic information by reconstructing direction-specific signals from symmetrically superimposed input. Discriminative and generative self-supervised method [75] combines CL and MIM, inspired by human learning through reading and writing. Unlike these methods, our approach seeks linguistic guidance to capture both character structures and inter-character associations.

## 3. Methodology

Although masked image modeling learns both intracharacter and inter-character patterns, it primarily focuses on visual structure information and lacks comprehensive linguistic knowledge. Our goal is to enhance the model’s ability to understand and integrate richer linguistic information.

## 3.1. Overview

Baseline. Our method is built upon MAE [31], a seminal work in the field of self-supervised learning, which consists of four core elements: masking strategy, encoder, decoder, and reconstruction target.

Given an image $\check { X ^ { \mathrm { ~ \in ~ } } } \mathbb { R } ^ { H \times W \times C }$ , where H and $W$ denote the height and width of the image and C denotes the channel, we split it into $N = H \times W / P ^ { 2 }$ non-overlapping patches $\{ x ^ { 1 } , \overset { \cdot } { x ^ { 2 } } , \cdots , x ^ { N } \}$ with the patch size of $P \times P ,$ Let $\mathcal { M }$ be the index set of masked patches, $X _ { v } = \{ x ^ { k } | k \notin \mathcal { M } \}$ denotes the set of visible patches and $X _ { m } = \{ x ^ { k } | k \in \mathcal { M } \}$ denotes the set of masked patches. The default mask ratio for MAE is 75%. Only unmasked patches are fed into the ViT encoder and the resulting feature is $\mathbb { E } ( \{ x ^ { c l s } , X _ { v } \} )$ where E is the encoder. The features fed into the decoder are the tokens of the visible patches obtained by the encoder and the mask tokens corresponding to the mask patches. The mask token is placed in the corresponding mask position. The output of the decoder is $\{ p ^ { c l { \dot { s } } } , p ^ { 1 } , p ^ { \top } , \cdots , p ^ { \dot { N } } \}$ Mean squared error (MSE) loss is applied to the masked tokens to compute the loss with respect to the corresponding original pixels. The objective is defined as

$$
\mathcal { L } _ { r e c o n } = \frac { 1 } { | | \mathcal { M } | | } \sum _ { k \in \mathcal { M } } | | p ^ { k } - t ^ { k } | | _ { 2 } ^ { 2 } ,\tag{1}
$$

where $t ^ { k }$ is the reconstruction target, MAE uses original pixels $( i . e . , x ^ { k } )$ by default.

![](images/2d432ef2fb17183824a9ab76e1a73cda0f4b4b44745c17a69779788cca091c33.jpg)  
Figure 2. Overview of our framework. Based on the dual-branch structure, the reconstruction loss and alignment loss are jointly optimized.

Pipeline. As shown in Fig. 2, our method consists of two branches, $i . e .$ , the masked reconstruction branch and the linguistic guidance branch. The input image of the linguistic guidance branch is $\hat { X }$ that contains the same linguistic information as X but different visual appearance. The two encoders share parameters. The linguistic information channels the MIM reconstruction via a specially designed decoder. We design a linguistics alignment module to extract vision-independent features as linguistic guidance. The overall loss function of LMIM is a combination of the reconstruction loss $\mathcal { L } _ { r e c o n }$ and the alignment loss $\mathcal { L } _ { a l i g n }$ , which can be formulated as

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { r e c o n } + \mathcal { L } _ { a l i g n } . } \end{array}\tag{2}
$$

The reconstruction target selects the features obtained by the encoder of MAE.

## 3.2. Linguistics-aware Masked Image Modeling

The innovation of our approach lies in integrating linguistic cues into visual information modeling, aiming to enhance the global context understanding of images. Our method introduces a novel approach that leverages visual and linguistic information concurrently, comprising two branches: the masked reconstruction branch and the linguistic guidance branch.

Guidance View Generation. In the self-supervised setting, where additional information is limited, data augmentation becomes crucial. By generating a guidance view that retains comprehensive linguistic information, we address the challenge of integrating linguistic cues within visual data. Previous research highlights the significance of data augmentation in contrastive learning for robust representations [11]. Unlike traditional methods, which risk misalignments with overly strong augmentations, our approach utilizes varied visual representations of identical textual content, ensuring linguistic coherence without explicit sequence contrastive learning.

Linguistics Alignment. To decouple visionindependent features, we exploit inputs with different visual representations to learn linguistics consistency. Specifically, we design the [cls] token to encapsulate linguistic information effectively. An alignment loss is introduced to align the [cls] token feature:

$$
\mathcal { L } _ { a l i g n } = | | f ^ { c l s } - \hat { f } ^ { c l s } | | _ { 2 } ^ { 2 } .\tag{3}
$$

This consistency ensures that linguistic information is consistently integrated across different branches, thus forcing the model to learn global linguistic information in addition to visual structures.

Linguistics-guided Reconstruction. We address the balance between mask ratio and reconstruction complexity. A high mask ratio can complicate reconstruction, while a low ratio might lead to reliance on visual shortcuts. Our default random masking strategy maximizes masking efficiency. Although block masking is feasible, it requires lower ratios and involves higher computational costs.

The linguistics-guided branch processes unmasked images, ensuring the integrity of linguistic information. The features obtained from the linguistic guidance branch are denoted as $\mathbb { E } ( \{ \hat { x } ^ { c l s } , \hat { X } \} ) = \{ \hat { f } ^ { \overline { { c } } l s } , \hat { f } ^ { 1 } , \cdots , \hat { f } ^ { N } \}$ , with $\hat { X }$ as the input image. For simplicity, we denote $\mathbb { E } ( \{ \hat { x } ^ { c l s } , \hat { X } \} )$ as $\hat { F } .$ . The sequence of features from the masked reconstruction branch, post-insertion of mask tokens, is recorded as $F .$ . To integrate linguistic information from $\hat { F }$ into $F ,$ , we design self-attention and cross-attention architecture within the decoder. The attention calculation is represented as:

$$
\begin{array} { r } { \mathbb { A } ( Q _ { F } , K _ { F } , V _ { F } ) = S o f t m a x ( Q _ { F } \cdot K _ { F } ^ { T } / \sqrt { d } ) \cdot V _ { F } , } \end{array}\tag{4}
$$

where $Q _ { F } , K _ { F }$ , and $V _ { F }$ are derived from $F$ via distinct linear layers, with $d$ as the dimension of $F .$ The decoder output is $\hat { \mathbb { A } } \big ( Q _ { \mathbb { A } ( Q _ { F } , K _ { F } , V _ { F } ) } , K _ { \hat { F } } , V _ { \hat { F } } \big )$ . The final output is used to calculate the reconstruction loss $\mathcal { L } _ { r e c o n }$

## 3.3. Discussion

Innovations and Effectiveness. STR necessitates the integration of both visual and linguistic information. In selfsupervised settings, linguistic information primarily refers to character correlations. Existing MIM can capture both intra-character and inter-character relationships, but tends to exploit local visual structures to complete reconstruction, resulting in limited global linguistics understanding. Our LMIM introduces linguistic information into the decoding process of MIM, thus effectively combining visual and linguistic information. To prevent MIM from exploiting visual structures for reconstruction, we design a linguistic alignment module to disentangle visual-independent features. This approach compels MIM to learn global context information rather than relying solely on local visual features for reconstruction. These elements collectively enable the model to effectively integrate linguistic information into visual modeling, achieving enhanced global context perception. Visualization of attention maps clearly demonstrates the advantage of our approach in simultaneously modeling visual and linguistic information, as shown in Fig. 1 (c).

Limitations. The current random masking strategy is suboptimal. Given the variable number of characters in text images, a fixed patch size masking strategy may be inadequate. Future work will explore character density-based masking strategies. Additionally, the pre-training for mask image modeling in STR is restricted to transformer architectures and is inapplicable to CNN architectures.

## 4. Experiments

## 4.1. Datasets

Pre-training Data. We conduct pre-training on real English text data (Union14M-U), and real Chinese text data (our collection). Union14M-U, a subset of Union14M [34], includes 10 million unlabeled real images. To verify the effectiveness on Chinese data, we collected 5 million images from the web and processed them to obtain 11 million unlabeled cropped Chinese text images, named Unlabeled Chinese Text Image 11M (UCTI-11M), for pre-training.

Fine-tuning Data. Annotated real data (ARD) [75] contains about 2.8 million annotated real images from TextOCR [62] and Open Image v5. Union14M-L, another subset of Union14M [34], comprises approximately 3.2 million labeled real images for fine-tuning. The Chinese benchmark [9] employs about 1.1 million images across four categories (i.e., Scene, Web, Document, and Handwriting) for fine-tuning.

Benchmarks. The six commonly used benchmarks include three regular text datasets (i.e., IIIT5K-Words (IIIT5K) [48], ICDAR2013 (IC13) [35], and Street View Text (SVT) [66]) and three irregular text datasets (i.e., ICDAR2015 (IC15) [36], SVT Perspective (SVTP) [50],

<table><tr><td colspan="4">Guide Align|Cur. M-O Art. Ctl. Sal. M-W Gen. | Avg</td></tr><tr><td>一</td><td>79.5 570.2</td><td>271.4 80.7 78.6 82.9 80.7</td><td>77.7</td></tr><tr><td>√</td><td>84.5</td><td>577.1 73.9 82.4 80.6 83.6 81.9 </td><td>80.6</td></tr><tr><td></td><td>85.077.274.6 83.3 82.2 </td><td>84.0 81.9</td><td>81.2</td></tr></table>

Table 1. Effect of linguistic guidance branch and the alignment loss.
<table><tr><td>Augment</td><td>Cur. M-O Art.</td><td>Ctl. Sal.</td><td>M-W Gen.</td><td>Avg</td></tr><tr><td>Strong</td><td>82.9 74.2 71.8</td><td>81.0 79.8</td><td>83.6 81.5</td><td>79.3</td></tr><tr><td>Weak</td><td>82.9 74.9 71.1</td><td>80.9 80.0</td><td>83.1 81.3</td><td>79.2</td></tr><tr><td>Medium</td><td>85.0 77.2 74.6</td><td>83.3 82.2</td><td>84.0 81.9</td><td>81.2</td></tr></table>

Table 2. Comparison of different levels of data augmentation.
<table><tr><td>Decoder</td><td>|Cur. M-O Art. Ctl. Sal. M-W</td><td>Gen.</td><td>Avg</td></tr><tr><td>CA-SA-FFN</td><td>82.6 74.6 73.3 83.2 81.7 84.0</td><td>81.8</td><td>|80.2</td></tr><tr><td>SA-CA-FFN</td><td>85.077.2 274.6 83.3 82.2 84.0</td><td>81.9</td><td>81.2</td></tr></table>

Table 3. Comparison of different decoder architecture designs.

CUTE80 (CUTE) [56]). Recent research identified mislabeled images and performed further verification [34, 76]. The Union14M benchmark contains approximately 0.41 million images with seven challenging scenarios: Curve, Multi-oriented, Artistic, Contextless, Salient, Multi-words, and General text. The Chinese benchmark [9] consists of 0.15 million labeled images across 4 categories.

## 4.2. Implementation Details

Pre-training Phase. The pre-training phase employs an AdamW optimizer with a cosine learning rate scheduler, utilizing a learning rate of 3e-4 and a batch size of 512. The architecture incorporates a ViT-small encoder [17] comprising 12 transformer blocks as the default configuration. The decoder uses two layers of blocks composed of selfattention and cross-attention. Input images are processed at a resolution of 32 × 128 pixels. The masking strategy implements random masking of 80% of patches with each patch size of 4 × 4. The reconstruction target utilizes features obtained by the MAERec [34] encoder with a dimension of 384. Unless otherwise specified, the model is trained for 10 epochs by default, including 1 warm-up epoch.

Fine-tuning Phase. Our baseline text recognition model comprises a ViT encoder and a transformer decoder. Following DiG [75], the decoder includes 6 transformer blocks with an embedding dimension of 512. The input image size remains 32 × 128. We apply the same data augmentation as ABINet [22] during training. We use the AdamW optimizer with a learning rate of 2e-4 and a batch size of 512. English data is fine-tuned for 10 epochs and Chinese data for 60 epochs. The maximum sequence length is 25 for English and 40 for Chinese.

<table><tr><td>Target</td><td>|Cur. M-O Art. Ctl. Sal. M-W Gen.</td><td>Avg</td></tr><tr><td>Pixel</td><td>79.6 70.1 70.6 78.8 77.0 78.8 80.0</td><td>76.4</td></tr><tr><td>Rand Feat</td><td>72.1 61.9 64.6 76.5 71.2 75.9 78.2</td><td>71.5</td></tr><tr><td>MAE Feat</td><td>85.0 77.2 74.683.3 82.2 84.0 81.9</td><td>81.2</td></tr></table>

Table 4. Comparison of different reconstruction targets.
<table><tr><td>Mask Ratio |Cur. M-O</td><td>Art. Ctl. Sal. M-W Gen.| |Avg</td></tr><tr><td>Rand 0.6 82.9 74.9</td><td>69.7 82.2 79.9 82.9 81.5 79.1</td></tr><tr><td></td><td>74.6 83.3 82.2 84.0</td></tr><tr><td>Rand 0.8 85.0 77.2</td><td>81.9 81.2 82.0 80.4</td></tr><tr><td>Rand 0.9 84.3 75.0 074.8 82.3 82.0</td><td>82.6 83.7 82.0</td></tr><tr><td>Block 0.6 84.876.875.7 83.6 80.6 Block 0.8 82.373.674.8 83.6 81.0 83.4</td><td>81.0 81.7 80.1</td></tr></table>

Table 5. Comparison of different masking strategies and ratios.

Evaluation Metrics. For English text, we evaluate the text recognizer using the WAICS (Word Accuracy Ignoring Case and Symbols) metric by default, which ignores symbols and is case-insensitive. For Chinese text, we follow existing benchmark settings to calculate sentence-level accuracy for each subset [9]. Average accuracy is calculated using the arithmetic mean (Avg) for subsets and the weighted average (W-Avg) for all instances.

## 4.3. Ablation Study

To efficiently verify the effectiveness of our method, we use two subsets of Union14-U, i.e., Book32 and OpenImages, comprising approximately 5 million images, for both pre-training and fine-tuning over 5 epochs each. The seven datasets of Union14M benchmark are abbreviated as Cur., M-O, Art., Ctl., Sal., M-W, and Gen. in the table.

Linguistic Guidance and Alignment Loss. Our method consists of two parts: reconstruction loss and alignment loss, Eq. (2). The effects of these components are shown in Tab. 1. Neither loss pertains to single-branch masked reconstruction. Optimal performance is achieved when both losses are optimized simultaneously, resulting in an average accuracy increase of 3.5% compared to when neither loss is applied. Excluding the alignment loss results in an average accuracy decrease of 0.6%. These findings underscore the importance of both components.

Guidance View Generation. We apply varying degrees of data augmentation to the original image to generate the guided view. Weak augmentation involves only geometric transformations, such as cropping and rotation. Medium augmentation adds color transformation, distortion, and perspective transformation to the basic geometric changes. Strong augmentation further adjusts parameters from medium augmentation to enhance the level. Experimental results in Tab. 2 demonstrat the importance of maintaining large visual appearance differences while preserving complete linguistic information.

Decoder Architecture. In Tab. 3, we compare decoder architectural designs. CA-SA-FFN indicates that features from the mask and guide branches are initially processed with cross-attention, followed by self-attention. Conversely, SA-CA-FNN involves self-attention on the mask branch features before interaction with the guide branch features. The experimental results suggest that performing self-attention prior to cross-attention is more effective.

Reconstruction Target. By default, we use the MAE feature as our reconstruction target. As illustrated in Tab. 4, we compare different reconstruction targets. Increased iterations or specific training strategies can minimize differences between reconstruction target when handling largescale data [41].

Mask Strategy and Ratio. Different masking strategies and ratios are crucial to mask reconstruction methods. We compare random and block masking. Unlike random masking, block masking obscures large continuous areas, thereby increasing reconstruction difficulty. As shown in Tab. 5, both masking methods are effective with appropriate mask ratios, but block masking requires fewer mask ratios (more patches need to be processed). Therefore, we default to the random masking strategy with a mask ratio of 80%.

## 4.4. Performance Comparison

Chinese Benchmark. Compared with other languages, Chinese relies more on linguistic knowledge to understand the content. To demonstrate the versatility of our method across different languages, we conducted experiments on Chinese data. As shown in Tab. 6, our method achieves state-of-the-art performance in terms of average accuracy. In particular, our method achieves a significant performance of 83.6% on the Scene dataset and 82.0% on the Web dataset, underscoring its ability to learn linguistic information and its versatility across languages.

Union14M Benchmark. In Tab. 7, the pre-trained methods (i.e., DiG, MAERec, and SSM) consistently outperform those trained from scratch. When pre-trained on Union14M-U and subsequently fine-tuned on Union14M-L, our method achieves consistent improvements across all subsets. Notably, we achieve a state-of-the-art average accuracy of 86.3% when pre-training for 20 epochs, representing a 2.0% improvement over SSM. We also experiment with fine-tuning on the ARD dataset, resulting in an average accuracy of 85.4%, with the General subset reaching 85.7%. This improvement stems from our method’s design, which integrates visual and linguistic information, enhancing the encoder’s feature representation for STR.

Common Benchmarks. We evaluate our method using corrected labels across six common benchmarks. Experimental results indicate an average accuracy of 97.0% and a weighted average accuracy of 96.7%, achieving new stateof-the-art performance, Tab. 8. Notably, higher improvements are observed on three irregular datasets, demonstrating that linguistic information offers additional benefits.

<table><tr><td>Method</td><td>Publisher</td><td>Source</td><td>Pre-train</td><td>Scene</td><td>Web</td><td>Document</td><td>Handwriting</td><td>Avg</td><td>W-Avg</td><td>Params</td></tr><tr><td>MASTER [42]</td><td>PR 21</td><td>[9]</td><td>No</td><td>62.1</td><td>53.4</td><td>82.7</td><td>18.5</td><td>54.2*</td><td>61.4*</td><td>63M</td></tr><tr><td>ABINet [22]</td><td>CVPR 21</td><td>[9]</td><td>No</td><td>60.9</td><td>51.1</td><td>91.7</td><td>13.8</td><td>54.4*</td><td>62.9*</td><td>53M</td></tr><tr><td>TransOCR [8]</td><td>CVPR 21</td><td>[9]</td><td>No</td><td>67.8</td><td>62.7</td><td>97.9</td><td>51.7</td><td>70.0*</td><td>74.8*</td><td>84M</td></tr><tr><td>SVTR-B [18]</td><td>IJCAI 22</td><td>[87]</td><td>No</td><td>71.4</td><td>64.1</td><td>99.3</td><td>50.0</td><td>71.2*</td><td>76.6*</td><td>25M</td></tr><tr><td>SVTR-L [18]</td><td>IJCAI 22</td><td>[87]</td><td>No</td><td>72.1</td><td>66.3</td><td>99.3</td><td>50.3</td><td>72.0*</td><td>77.2*</td><td>41M</td></tr><tr><td>CIRN [80]</td><td>IJCAI 23</td><td>[80]</td><td>No</td><td>73.3</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DCTC-B [87]</td><td>AAAI 24</td><td>[87]</td><td>No</td><td>72.2</td><td>67.0</td><td>99.4</td><td>50.4</td><td>72.3*</td><td>77.3*</td><td>25M</td></tr><tr><td>DCTC-L [87]</td><td>AAAI 24</td><td>[87]</td><td>No</td><td>73.9</td><td>68.5</td><td>99.4</td><td>51.0</td><td>73.2*</td><td>78.3*</td><td>41M</td></tr><tr><td>LMIM (Scratch)</td><td>CVPR 25</td><td>Ours</td><td>No</td><td>75.1</td><td>74.7</td><td>97.1</td><td>53.3</td><td>75.1</td><td>79.0</td><td>36M</td></tr><tr><td>TransOCR [8]</td><td>CVPR21</td><td>[9]</td><td>Yes</td><td>68.5</td><td>62.5</td><td>97.9</td><td>53.5</td><td>70.6*</td><td>75.3*</td><td>84M</td></tr><tr><td>CCR-CLIP [79]</td><td>ICCV 23</td><td>[79]</td><td>Yes</td><td>71.3</td><td>69.2</td><td>98.3</td><td>60.3</td><td>74.8*</td><td>78.3*</td><td></td></tr><tr><td>MaskOCR-S [46]</td><td>TMLR 24</td><td>[46]</td><td>Yes</td><td>71.4</td><td>72.5</td><td>98.8</td><td>55.6</td><td>74.6*</td><td>78.1</td><td>36M</td></tr><tr><td>MaskOCR-B [46]</td><td>TMLR 24</td><td>[46]</td><td>Yes</td><td>73.9</td><td>74.8</td><td>99.3</td><td>63.7</td><td>77.9*</td><td>80.8</td><td>100M</td></tr><tr><td>SeqCLR [1]</td><td>CVPR 21</td><td>Ours</td><td>Yes</td><td>81.7</td><td>80.5</td><td>98.5</td><td>60.3</td><td>80.3</td><td>83.8</td><td>36M</td></tr><tr><td>MIM [31]</td><td>CVPR 22</td><td>Ours</td><td>Yes</td><td>82.3</td><td>80.9</td><td>98.9</td><td>62.4</td><td>81.1</td><td>84.6</td><td>36M</td></tr><tr><td>LMIM</td><td>CVPR 25</td><td>Ours</td><td>Yes</td><td>83.6</td><td>82.0</td><td>99.1</td><td>63.9</td><td>82.2</td><td>85.5</td><td>36M</td></tr></table>

Table 6. Results on Chinese benchmark. <sup>∗</sup> means results re-calculated using those works’ original results by us.
<table><tr><td>Method</td><td>Publisher</td><td>Source PT-data</td><td></td><td>|Curve M-O Artistic Contextless Salient M-W General|Avg W-Avg|Params</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ABINet [22]</td><td>CVPR 21</td><td>[34]</td><td></td><td>75.061.5</td><td>65.3</td><td>71.1</td><td>72.9</td><td>59.1</td><td>79.4</td><td></td><td>|69.2 79.2*</td><td>37M</td></tr><tr><td>VisionLAN [69] ICCV 21</td><td></td><td>[34]</td><td></td><td>70.7 57.2</td><td>56.7</td><td>63.8</td><td>67.6</td><td>47.3</td><td>74.2</td><td></td><td>62.574.0*</td><td>33M</td></tr><tr><td>MATRN [49]</td><td>ECCV 22</td><td>[34]</td><td></td><td>80.5 64.7</td><td>71.1</td><td>74.8</td><td>79.4</td><td>67.6</td><td>77.9</td><td></td><td>74.6 77.8*</td><td>44M</td></tr><tr><td>PARSeq [7]</td><td>ECCV 22</td><td>[23]</td><td></td><td>79.8 79.2</td><td>67.4</td><td>77.4</td><td>77.0</td><td>76.9</td><td>80.6</td><td></td><td>76.9 80.5*</td><td>24M</td></tr><tr><td>DiG-S [75]</td><td>ACMMM 22 [23]</td><td></td><td>U14M-U|</td><td>85.9 83.5</td><td>77.4</td><td>82.5</td><td>84.3</td><td>84.0</td><td>83.8</td><td></td><td>83.0 83.8*</td><td>36M</td></tr><tr><td>MAERec-S [34] ICCV 23</td><td></td><td>[34]</td><td>U14M-U|</td><td>81.4 71.4</td><td>72.0</td><td>82.0</td><td>78.5</td><td>82.4</td><td>82.5</td><td></td><td>78.6 82.4*</td><td>36M</td></tr><tr><td>SSM-S [23]</td><td>IJCAI 24</td><td>[23]</td><td>U14M-U|</td><td>87.5 85.8</td><td>78.4</td><td>84.8</td><td>85.2</td><td>85.0</td><td>84.0</td><td>84.3</td><td>84.0*</td><td>36M</td></tr><tr><td>SeqCLR [1]</td><td>CVPR 21</td><td>Ours</td><td>U14M-U|</td><td>83.7 79.9</td><td>73.7</td><td>79.7</td><td>81.0</td><td>84.0</td><td>82.7</td><td>|80.7</td><td>82.7</td><td>36M</td></tr><tr><td>MIM [31]</td><td>CVPR 22</td><td>Ours</td><td>U14M-U</td><td>84.8 79.3</td><td>71.1</td><td>81.5</td><td>81.0</td><td>82.5</td><td>82.0</td><td>80.3</td><td>82.0</td><td>36M</td></tr><tr><td>LMIM‡</td><td>CVPR 25</td><td>Ours</td><td>U14M-U</td><td>89.7 84.9</td><td>79.4</td><td>81.9</td><td>86.0</td><td>82.9</td><td>85.1</td><td>84.3</td><td>85.1</td><td>36M</td></tr><tr><td>LMIM</td><td>CVPR 25</td><td>Ours</td><td>U14M-U|</td><td>87.5 84.5</td><td>79.8</td><td>84.3</td><td>86.6</td><td>86.1</td><td>84.4</td><td>84.7</td><td>84.4</td><td>36M</td></tr><tr><td> $\mathrm { L M I M _ { 2 0 e p } ^ { \ddagger } }$ </td><td>CVPR 25</td><td>Ours</td><td>U14M-U|</td><td>90.6 86.9</td><td>80.0</td><td>82.3</td><td>87.7</td><td>84.4</td><td>85.7</td><td>85.4</td><td>85.7</td><td>36M</td></tr><tr><td> $\mathrm { L M I M _ { 2 0 e p } }$ </td><td>CVPR 25</td><td>Ours</td><td>U14M-U</td><td>90.3 86.6</td><td>80.7</td><td>85.5</td><td>88.2</td><td>87.9</td><td>85.1</td><td>86.3</td><td>85.1</td><td>36M</td></tr></table>

Table 7. Results on English Union14M benchmark. Unless otherwise specified, all text recognizers are trained using real data from Union14M-L. PT-data stands for the pre-training dataset. U14M-U denotes Union14M-U. M-O indicates Multi-Oriented, while M-W indicates Multi-Words. $\mathrm { L M I M _ { 2 0 e p } }$ refers to pre-training for 20 epochs. The symbol <sup>‡</sup> represents fine-tuning on ARD.

## 4.5. Visualization

To qualitatively verify that our method has effectively learned linguistic information compared to sequence contrastive learning and masked image modeling, we visualize the attention maps for different queries, Fig. 1 (c) and Fig. 3. We use the attention map obtained from the last block (12- th block) of ViT-Small. We can see that SeqCLR’s attention at each query position is rather chaotic and cannot reflect the character structure. This is because the contrast of local sequence elements does not fully learn the underlying structure and the association between characters. In particular, we compare the attention maps obtained for different query positions of the same image in Fig. 3. The attention maps obtained from different queries of the same image and different queries of different images demonstrate that the MIM method pays more attention to local areas. This is because the visual features of local areas can be used to reconstruct the mask reconstruction process. Our LMIM not only pays attention to different characters but also shows a clear character structure, proving that the method captures both visual

<table><tr><td>Method</td><td>Publisher</td><td>Source PT-data</td><td></td><td>FT-data</td><td>IIT5K IC13 SVT IC15 SVTP CUTE|</td><td></td><td></td><td></td><td>E|Avg</td><td>W-Avg|Params</td><td></td></tr><tr><td>ABINet [22]</td><td>CVPR 21</td><td>[34]</td><td></td><td>U14M-L|</td><td>97.2</td><td>97.2 95.7 87.6</td><td>92.1</td><td>94.4</td><td>|94.0</td><td>93.9*</td><td>37M</td></tr><tr><td>VisionLAN [69]</td><td>ICCV 21</td><td>[34]</td><td></td><td>U14M-L</td><td>96.3</td><td>95.1 91.3 83.6</td><td>85.4</td><td>92.4</td><td>91.3</td><td>91.2*</td><td>33M</td></tr><tr><td>MATRN [49]</td><td>ECCV 22</td><td>[34]</td><td></td><td>U14M-L</td><td>98.2</td><td>97.9 96.9 88.2</td><td>94.1</td><td>97.9</td><td>95.5</td><td>95.0*</td><td>44M</td></tr><tr><td>PARSeq [7]</td><td>ECCV 22</td><td>[23]</td><td></td><td>U14M-L</td><td>98.0</td><td>96.8 95.2 85.2</td><td>90.5</td><td>96.5</td><td>93.8</td><td>93.5*</td><td>24M</td></tr><tr><td>MaskOCR-S [46] TMLR 24</td><td></td><td>[46]</td><td>STD&amp;URD ARD</td><td></td><td>98.0</td><td>97.8 96.9 90.2</td><td>94.9</td><td>96.2</td><td>95.6</td><td>95.4*</td><td>31M</td></tr><tr><td>DiG-S [75]</td><td>ACMMM 22 [23]</td><td></td><td>U14M-U</td><td>U14M-L</td><td>98.7</td><td>97.8 98.5 88.9</td><td>92.7</td><td>96.5</td><td>95.5</td><td>95.3*</td><td>36M</td></tr><tr><td>DiG-S [75]</td><td>ACMMM 22 [75]</td><td></td><td>STD&amp;URD ARD</td><td></td><td>97.7</td><td>97.3 96.1 88.6</td><td>91.6</td><td>96.1</td><td></td><td>94.6* 94.5*</td><td>36M</td></tr><tr><td>CCD [26]</td><td>ICCV 23</td><td>[26]</td><td>STD&amp;URD ARD</td><td></td><td>98.0</td><td>98.3 96.4 90.3</td><td>92.7</td><td>98.3</td><td></td><td>95.6* 95.4*</td><td>36M</td></tr><tr><td>MAERec-S [34]</td><td>ICCV 23</td><td>[34]</td><td>U14M-U</td><td>U14M-L</td><td>98.0</td><td>97.6 96.8 87.1</td><td>93.2</td><td>97.9</td><td>95.1</td><td>94.5*</td><td>36M</td></tr><tr><td>SSM-S [23]</td><td>IJCAI 24</td><td>[23]</td><td>U14M-U</td><td>U14M-L|</td><td>99.0</td><td>98.3 97.8 89.5</td><td>94.0</td><td>98.3</td><td>96.1</td><td>95.8*</td><td>36M</td></tr><tr><td>LMIM</td><td>CVPR 25</td><td>Ours</td><td>U14M-U</td><td>ARD</td><td>98.0</td><td>97.9 96.9 88.2</td><td>93.5</td><td>98.3</td><td>95.5</td><td>94.9</td><td>36M</td></tr><tr><td>LMIM</td><td>CVPR 25</td><td>Ours</td><td>U14M-U</td><td>U14M-L</td><td>98.5</td><td>98.0 97.7 88.7</td><td>94.0</td><td>98.6</td><td>95.9</td><td>95.3</td><td>36M</td></tr><tr><td>SeqCLR† [1]</td><td>CVPR 21</td><td>Ours</td><td>U14M-U</td><td>U14M-L</td><td>98.8</td><td>97.9 96.8 91.4</td><td>92.9</td><td>96.9</td><td>95.8</td><td>95.9</td><td>36M</td></tr><tr><td>MIM† [31]</td><td>CVPR 22</td><td>Ours</td><td>U14M-U</td><td>U14M-L</td><td>98.5</td><td>97.6 96.0 89.5</td><td>91.3</td><td>96.9</td><td>95.0</td><td>95.1</td><td>36M</td></tr><tr><td>LMIM†</td><td>CVPR 25</td><td>Ours</td><td>U14M-U</td><td>ARD</td><td>98.8</td><td>97.8 97.1 90.8</td><td>94.4</td><td>97.9</td><td>96.1</td><td>96.0</td><td>36M</td></tr><tr><td>LMIM†</td><td>CVPR 25</td><td>Ours</td><td>U14M-U</td><td>U14M-L</td><td>99.3</td><td>98.1 98.3 91.7</td><td>95.4</td><td>99.3</td><td>97.0</td><td>96.7</td><td>36M</td></tr></table>

Table 8. Results on six common benchmarks. PT-data refers to the pre-training dataset, and FT-data refers to the fine-tuning dataset. U14M-L and U14M-U are Union14M-L and Union14M-U, respectively. URD denotes the unlabeled real dataset containing 15.8 million images. STD refers to 17 million synthetic data. The symbol <sup>†</sup> denotes using corrected label.

![](images/12ecedefa9c0bdcb6d8168e804ca1521a7bc299bdfb55d2bb13a241950934c97.jpg)  
Figure 3. Visualization of attention maps. The red box in the input image refers to the query.

and linguistic information.

## 5. Conclusion

In this paper, we propose Linguistics-aware Masked Image Modeling, i.e., LMIM, a simple yet effective selfsupervised learning framework specifically designed for STR. Our approach introduces a dual-branch structure that integrates linguistic cues into visual modeling, considering both simultaneously as crucial for STR. A specially designed linguistic alignment module leverages images with varying visual appearances to disentangle visualindependent linguistic features. Unlike methods focused solely on visual structure, LMIM encourages the model to utilize global contextual information for reconstruction. Our method delivers significant improvements on various benchmarks, including English and Chinese text recognition tasks. Attention visualization qualitatively demonstrates that our method effectively combines visual and linguistic information. In future work, we will explore more suitable masking strategies tailored to character density, aiming to further optimize model performance and effectively handle diverse scene text recognition tasks.

## Acknowledgement

This work is supported by the National Natural Science Foundation of China (Grant NO 62376266 and 62406318), Key Laboratory of Ethnic Language Intelligent Analysis and Security Governance of MOE, Minzu University of China, Beijing, China.

## References

[1] Aviad Aberdam, Ron Litman, Shahar Tsiper, Oron Anschel, Ron Slossberg, Shai Mazor, R. Manmatha, and Pietro Perona. Sequence-to-sequence contrastive learning for text recognition. In CVPR, pages 15302–15312, 2021. 2, 3, 7, 8

[2] Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael G. Rabbat, Yann LeCun, and Nicolas Ballas. Self-supervised learning from images with a joint-embedding predictive architecture. In CVPR, pages 15619–15629, 2023. 3

[3] Rowel Atienza. Vision transformer for fast and efficient scene text recognition. In ICDAR, pages 319–334, 2021. 3

[4] Jeonghun Baek, Geewook Kim, Junyeop Lee, Sungrae Park, Dongyoon Han, Sangdoo Yun, Seong Joon Oh, and Hwalsuk Lee. What is wrong with scene text recognition model comparisons? dataset and model analysis. In ICCV, pages 4714–4722, 2019. 3

[5] Jeonghun Baek, Yusuke Matsui, and Kiyoharu Aizawa. What if we only use real datasets for scene text recognition? toward scene text recognition with fewer labels. In CVPR, pages 3113–3122, 2021. 2

[6] Hangbo Bao, Li Dong, Songhao Piao, and Furu Wei. BEiT: BERT pre-training of image transformers. In ICLR, 2022. 3

[7] Darwin Bautista and Rowel Atienza. Scene text recognition with permuted autoregressive sequence models. In ECCV, pages 178–196, 2022. 2, 3, 7, 8

[8] Jingye Chen, Bin Li, and Xiangyang Xue. Scene text telescope: Text-focused scene image super-resolution. In CVPR, pages 12026–12035, 2021. 7

[9] Jingye Chen, Haiyang Yu, Jianqi Ma, Mengnan Guan, Xixi Xu, Xiaocong Wang, Shaobo Qu, Bin Li, and Xiangyang Xue. Benchmarking chinese text recognition: Datasets, baselines, and an empirical study. arXiv, abs/2112.15093, 2021. 5, 6, 7

[10] Pengguang Chen, Shu Liu, and Jiaya Jia. Jigsaw clustering for unsupervised visual representation learning. In CVPR, pages 11526–11535, 2021. 3

[11] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In ICML, pages 1597–1607, 2020. 2, 3, 4

[12] Yudi Chen, Wei Wang, Yu Zhou, Fei Yang, Dongbao Yang, and Weiping Wang. Self-training for domain adaptive scene text detection. In ICPR, pages 850–857, 2020. 2

[13] Ching-Yao Chuang, Joshua Robinson, Yen-Chen Lin, Antonio Torralba, and Stefanie Jegelka. Debiased contrastive learning. In NeurIPS, pages 8765–8775, 2020. 3

[14] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: pre-training of deep bidirectional transformers for language understanding. arXiv, abs/1810.04805, 2018. 3

[15] Carl Doersch, Abhinav Gupta, and Alexei A. Efros. Unsupervised visual representation learning by context prediction. In ICCV, pages 1422–1430, 2015. 3

[16] Xiaoyi Dong, Jianmin Bao, Ting Zhang, Dongdong Chen, Weiming Zhang, Lu Yuan, Dong Chen, Fang Wen, and

Nenghai Yu. Bootstrapped masked autoencoders for vision BERT pretraining. In ECCV, pages 247–264, 2022. 3

[17] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In ICLR, 2021. 3, 5

[18] Yongkun Du, Zhineng Chen, Caiyan Jia, Xiaoting Yin, Tian lun Zheng, Chenxia Li, Yuning Du, and Yu-Gang Jiang. SVTR: scene text recognition with a single visual model. In IJCAI, pages 884–890, 2022. 3, 7

[19] Alexandre Eymael, Renaud Vandeghen, Anthony Cioppa,¨ Silvio Giancola, Bernard Ghanem, and Marc Van Droogenbroeck. Efficient image pre-training with siamese cropped masked autoencoders. In ECCV, pages 348–366, 2024. 3

[20] Bo Fang, Wenhao Wu, Chang Liu, Yu Zhou, Dongliang He, and Weiping Wang. Mamico: Macro-to-micro semantic correspondence for self-supervised video representation learn ing. In ACMMM, pages 1348–1357, 2022. 2

[21] Bo Fang, Wenhao Wu, Chang Liu, Yu Zhou, Yuxin Song, Weiping Wang, Xiangbo Shu, Xiangyang Ji, and Jingdong Wang. UATVR: uncertainty-adaptive text-video retrieval. In ICCV, pages 13677–13687, 2023. 3

[22] Shancheng Fang, Hongtao Xie, Yuxin Wang, Zhendong Mao, and Yongdong Zhang. Read like humans: Autonomous, bidirectional and iterative language modeling for scene text recognition. In CVPR, pages 7098–7107, 2021. 2, 3, 5, 7, 8

[23] Zuan Gao, Yuxin Wang, Yadong Qu, Boqiang Zhang, Zixiao Wang, Jianjun Xu, and Hongtao Xie. Self-supervised pretraining with symmetric superimposition modeling for scene text recognition. In IJCAI, pages 767–775, 2024. 3, 7, 8

[24] Spyros Gidaris, Praveer Singh, and Nikos Komodakis. Unsupervised representation learning by predicting image rota tions. In ICLR, 2018. 3

[25] Tongkun Guan, Chaochen Gu, Jingzheng Tu, Xue Yang, Qi Feng, Yudi Zhao, and Wei Shen. Self-supervised i mplicit glyph attention for text recognition. In CVPR, pages 15285– 15294, 2023. 3

[26] Tongkun Guan, Wei Shen, Xue Yang, Qi Feng, Zekun Jiang, and Xiaokang Yang. Self-supervised character-to-character distillation for text recognition. In ICCV, pages 19473– 19484, 2023. 2, 3, 8

[27] Yuanfan Guo, Minghao Xu, Jiawen Li, Bingbing Ni, Xuanyu Zhu, Zhenbang Sun, and Yi Xu. HCSC: hierarchical con trastive selective coding. In CVPR, pages 9696–9705, 2022. 3

[28] Ankush Gupta, Andrea Vedaldi, and Andrew Zisserman. Synthetic data for text localisation in natural images. In CVPR, pages 2315–2324, 2016. 2

[29] Agrim Gupta, Jiajun Wu, Jia Deng, and Fei-Fei Li. Siamese masked autoencoders. In NeurIPS, pages 40676–40693, 2023. 3

[30] Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross B. Girshick. Momentum contrast for unsupervised visual representation learning. In CVPR, pages 9726–9735, 2020. 2, 3

[31] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollar, and Ross B. Girshick. Masked autoencoders are scal-´ able vision learners. In CVPR, pages 15979–15988, 2022. 3, 7, 8

[32] Yue He, Chen Chen, Jing Zhang, Juhua Liu, Fengxiang He, Chaoyue Wang, and Bo Du. Visual semantics allow for textual reasoning better in scene text recognition. In AAAI, pages 888–896, 2022. 3

[33] Max Jaderberg, Karen Simonyan, Andrea Vedaldi, and Andrew Zisserman. Reading text in the wild with convolutional neural networks. IJCV, 116(1):1–20, 2016. 2

[34] Qing Jiang, Jiapeng Wang, Dezhi Peng, Chongyu Liu, and Lianwen Jin. Revisiting scene text recognition: A data perspective. In ICCV, pages 20543–20554, 2023. 2, 5, 7, 8

[35] Dimosthenis Karatzas, Faisal Shafait, Seiichi Uchida, Masakazu Iwamura, Lluis Gomez i Bigorda, Sergi Robles Mestre, Joan Mas, David Fernandez Mota, Jon Almaz´ an, and´ Llu´ıs-Pere de las Heras. ICDAR 2013 robust reading competition. In ICDAR, pages 1484–1493, 2013. 5

[36] Dimosthenis Karatzas, Lluis Gomez-Bigorda, Anguelos Nicolaou, and et.al. ICDAR 2015 competition on robust reading. In ICDAR, pages 1156–1160, 2015. 5

[37] Zhenhang Li, Yan Shu, Weichao Zeng, Dongbao Yang, and Yu Zhou. First creating backgrounds then rendering texts: A new paradigm for visual text blending. In ECAI, pages 346–353, 2024. 2

[38] Chang Liu, Yuan Yao, Dezhao Luo, Yu Zhou, and Qixiang Ye. Self-supervised motion perception for spatiotemporal representation learning. TNNLS, 34(12):9832–9846, 2023. 2, 3

[39] Chang Liu, Bohao Li, Mengnan Shi, Xiaozhong Chen, Qixiang Ye, and Xiangyang Ji. Explicit margin equilibrium for few-shot object detection. TNNLS, pages 1–13, 2024. 3

[40] Hao Liu, Bin Wang, Zhimin Bao, Mobai Xue, Sheng Kang, Deqiang Jiang, Yinsong Liu, and Bo Ren. Perceiving strokesemantic context: Hierarchical contrastive learning for robust scene text recognition. In AAAI, pages 1702–1710, 2022. 3

[41] Xingbin Liu, Jinghao Zhou, Tao Kong, Xianming Lin, and Rongrong Ji. Exploring target representations for masked autoencoders. In ICLR, 2024. 6

[42] Ning Lu, Wenwen Yu, Xianbiao Qi, Yihao Chen, Ping Gong, Rong Xiao, and Xiang Bai. MASTER: multi-aspect nonlocal network for scene text recognition. PR, 117:107980, 2021. 7

[43] Canjie Luo, Lianwen Jin, and Jingdong Chen. SimAN: Exploring self-supervised representation learning of scene text via similarity-aware normalization. In CVPR, pages 1029– 1038, 2022. 3

[44] Dezhao Luo, Chang Liu, Yu Zhou, Dongbao Yang, Can Ma, Qixiang Ye, and Weiping Wang. Video cloze procedure for self-supervised spatio-temporal learning. In AAAI, pages 11701–11708, 2020. 2, 3

[45] Jiahao Lyu, Wei Wang, Dongbao Yang, Jinwen Zhong, and Yu Zhou. Arbitrary reading order scene text spotter with local semantics guidance. In AAAI, 2025. 2

[46] Pengyuan Lyu, Chengquan Zhang, Shanshan Liu, Meina Qiao, Yangliu Xu, Liang Wu, Kun Yao, Junyu Han, Errui Ding, and Jingdong Wang. MaskOCR: Scene text recogni tion with masked vision-language pre-training. TMLR, 2024. 7, 8

[47] Xin Ma, Chang Liu, Chunyu Xie, Long Ye, Yafeng Deng, and Xiangyang Ji. Disjoint masking with joint distillation for efficient masked image modeling. TMM, 26:3077–3087, 2024. 3

[48] Anand Mishra, Karteek Alahari, and C. V. Jawahar. Scene text recognition using higher order language priors. In BMVC, pages 1–11, 2012. 5

[49] Byeonghu Na, Yoonsik Kim, and Sungrae Park. Multimodal text recognition networks: Interactive enhancements between visual and semantic features. In ECCV, pages 446– 463, 2022. 2, 3, 7, 8

[50] Trung Quy Phan, Palaiahnakote Shivakumara, Shangxuan Tian, and Chew Lim Tan. Recognizing text with perspective distortion in natural scenes. In ICCV, pages 569–576, 2013. 5

[51] Zhi Qiao, Yu Zhou, Dongbao Yang, Yucan Zhou, and Weip ing Wang. SEED: semantics enhanced encoder-decoder framework for scene text recognition. In CVPR, pages 13525–13534, 2020. 2, 3

[52] Zhi Qiao, Yu Zhou, Jin Wei, Wei Wang, Yuan Zhang, Ning Jiang, Hongbin Wang, and Weiping Wang. PIMNet: A par allel, iterative and mimicking network for scene text recognition. In ACMMM, pages 2046–2055, 2021. 3

[53] Zhi Qiao, Zhilong Ji, Ye Yuan, and Jinfeng Bai. Decoupling visual-semantic features learning with dual masked autoen coder for self-supervised scene text recognition. In ICDAR, pages 261–279, 2023. 3

[54] Xugong Qin, Yu Zhou, Youhui Guo, Dayan Wu, Zhihong Tian, Ning Jiang, Hongbin Wang, and Weiping Wang. Mask is all you need: Rethinking mask R-CNN for dense and arbitrary-shaped scene text detection. In ACMMM, pages 414–423, 2021. 2

[55] Xugong Qin, Pengyuan Lyu, Chengquan Zhang, Yu Zhou, Kun Yao, Peng Zhang, Hailun Lin, and Weiping Wang. Towards robust real-time scene text detection: From semantic to instance representation learning. In ACMMM, pages 2025–2034, 2023. 2

[56] Anhar Risnumawan, Palaiahnakote Shivakumara, Chee Seng Chan, and Chew Lim Tan. A robust arbitrary text detection system for natural scene images. ESWA, 41(18):8027–8048, 2014. 5

[57] Huawen Shen, Xiang Gao, Jin Wei, Liang Qiao, Yu Zhou, Qiang Li, and Zhanzhan Cheng. Divide rows and conquer cells: Towards structure recognition for large tables. In IJ-CAI, pages 1369–1377, 2023. 2

[58] Huawen Shen, Gengluo Li, Jinwen Zhong, and Yu Zhou. LDP: generalizing to multilingual visual information extraction by language decoupled pretraining. In AAAI, 2025. 2

[59] Baoguang Shi, Xiang Bai, and Cong Yao. An end-to-end trainable neural network for image-based sequence recognition and its application to scene text recognition. TPAMI, 39 (11):2298–2304, 2017. 2, 3

[60] Yan Shu, Wei Wang, Yu Zhou, Shaohui Liu, Aoting Zhang, Dongbao Yang, and Weiping Wang. Perceiving ambiguity and semantics without recognition: An efficient and effective ambiguous scene text detector. In ACMMM, pages 1851– 1862, 2023. 2

[61] Yan Shu, Weichao Zeng, Zhenhang Li, Fangmin Zhao, and Yu Zhou. Visual text meets low-level vision: A comprehensive survey on visual text processing. arXiv, abs/2402.03082, 2024. 2

[62] Amanpreet Singh, Guan Pang, Mandy Toh, Jing Huang, Wojciech Galuba, and Tal Hassner. Textocr: Towards largescale end-to-end reasoning for arbitrary-shaped scene text. In CVPR, pages 8802–8812, 2021. 5

[63] Mohamed Ali Souibgui, Sanket Biswas, Andres Mafla,´ Ali Furkan Biten, Alicia Fornes, Yousri Kessentini, Josep´ Llados, Llu´ ´ıs Gomez, and Dimosthenis Karatzas. Text-´ DIAE: A self-supervised degradation invariant autoencoder for text recognition and document enhancement. In AAAI, pages 2330–2338, 2023. 3

[64] Zhaoyi Wan, Minghang He, Haoran Chen, Xiang Bai, and Cong Yao. TextScanner: Reading characters in order for robust scene text recognition. In AAAI, pages 12120–12127, 2020. 3

[65] Haochen Wang, Kaiyou Song, Junsong Fan, Yuxi Wang, Jin Xie, and Zhaoxiang Zhang. Hard patches mining for masked image modeling. In CVPR, pages 10375–10385, 2023. 3

[66] Kai Wang, Boris Babenko, and Serge J. Belongie. Endto-end scene text recognition. In ICCV, pages 1457–1464, 2011. 5

[67] Wei Wang, Yu Zhou, Jiahao Lyu, Dayan Wu, Guoqing Zhao, Ning Jiang, and Weiping Wang. Tpsnet: Reverse thinking of thin plate splines for arbitrary shape scene text representation. In ACMMM, pages 5014–5025, 2022. 2

[68] Xinlong Wang, Rufeng Zhang, Chunhua Shen, Tao Kong, and Lei Li. Dense contrastive learning for self-supervised visual pre-training. In CVPR, pages 3024–3033, 2021. 3

[69] Yuxin Wang, Hongtao Xie, Shancheng Fang, Jing Wang, Shenggao Zhu, and Yongdong Zhang. From two to one: A new scene text recognizer with visual language modeling network. In ICCV, pages 14174–14183, 2021. 2, 3, 7, 8

[70] Zixiao Wang, Hongtao Xie, Yuxin Wang, Jianjun Xu, Boqiang Zhang, and Yongdong Zhang. Symmetrical linguistic feature distillation with CLIP for scene text recognition. In ACMMM, pages 509–518, 2023. 3

[71] Chen Wei, Haoqi Fan, Saining Xie, Chao-Yuan Wu, Alan L. Yuille, and Christoph Feichtenhofer. Masked feature prediction for self-supervised visual pre-training. In CVPR, pages 14648–14658, 2022. 3

[72] Jin Wei, Yuan Zhang, Yu Zhou, Gangyan Zeng, Zhi Qiao, Youhui Guo, Haiying Wu, Hongbin Wang, and Weiping Wang. Textblock: Towards scene text spotting without finegrained detection. In ACMMM, pages 5892–5902, 2022. 2

[73] Zhirong Wu, Yuanjun Xiong, Stella X. Yu, and Dahua Lin. Unsupervised feature learning via non-parametric instance discrimination. In CVPR, pages 3733–3742, 2018. 3

[74] Zhenda Xie, Zheng Zhang, Yue Cao, Yutong Lin, Jianmin Bao, Zhuliang Yao, Qi Dai, and Han Hu. SimMIM: a simple

framework for masked image modeling. In CVPR, pages 9643–9653, 2022. 3

[75] Mingkun Yang, Minghui Liao, Pu Lu, Jing Wang, Shenggao Zhu, Hualin Luo, Qi Tian, and Xiang Bai. Reading and writing: Discriminative and generative modeling for selfsupervised text recognition. In ACMMM, pages 4214–4223, 2022. 2, 3, 5, 7, 8

[76] Xiaomeng Yang, Zhi Qiao, Jin Wei, Dongbao Yang, and Yu Zhou. Masked and permuted implicit context learning for scene text recognition. SPL, 31:964–968, 2024. 5

[77] Yuan Yao, Chang Liu, Dezhao Luo, Yu Zhou, and Qixi ang Ye. Video playback rate perception for self-supervised spatio-temporal representation learning. In CVPR, pages 6548–6557, 2020. 2

[78] Deli Yu, Xuan Li, Chengquan Zhang, Tao Liu, Junyu Han, Jingtuo Liu, and Errui Ding. Towards accurate scene text recognition with semantic reasoning networks. In CVPR, pages 12110–12119, 2020. 2, 3

[79] Haiyang Yu, Xiaocong Wang, Bin Li, and Xiangyang Xue. Chinese text recognition with a pre-trained CLIP-like model through image-ids aligning. In ICCV, pages 11909–11918, 2023. 7

[80] Haiyang Yu, Xiaocong Wang, Bin Li, and Xiangyang Xue. Orientation-independent chinese text recognition in scene images. In IJCAI, pages 1667–1675, 2023. 7

[81] Gangyan Zeng, Yuan Zhang, Yu Zhou, Xiaomeng Yang, Ning Jiang, Guoqing Zhao, Weiping Wang, and Xu-Cheng Yin. Beyond OCR + VQA: towards end-to-end reading and reasoning for robust and accurate textvqa. PR, 138:109337, 2023. 2

[82] Weichao Zeng, Yan Shu, Zhenhang Li, Dongbao Yang, and Yu Zhou. Textctrl: Diffusion-based scene text editing with prior guidance control. In NeurIPS, pages 138569–138594, 2024. 2

[83] Xinyun Zhang, Binwu Zhu, Xufeng Yao, Qi Sun, Ruiyu Li, and Bei Yu. Context-based contrastive learning for scene text recognition. In AAAI, pages 3353–3361, 2022. 3

[84] Yifei Zhang, Chang Liu, Yu Zhou, Wei Wang, Weiping Wang, and Qixiang Ye. Progressive cluster purification for unsupervised feature learning. In ICPR, pages 8476–8483, 2020. 3

[85] Yifei Zhang, Chang Liu, Yu Zhou, Weiping Wang, Qixiang Ye, and Xiangyang Ji. Beyond instance discrimination: Relation-aware contrastive self-supervised learning. TMM, 26:4628–4640, 2024. 3

[86] Yan Zhang, Gangyan Zeng, Huawen Shen, Daiqing Wu, Yu Zhou, and Can Ma. Track the answer: Extending textvqa from image to video with spatio-temporal clues. In AAAI, 2025. 2

[87] Ziyin Zhang, Ning Lu, Minghui Liao, Yongshuai Huang, Cheng Li, Min Wang, and Wei Peng. Self-distillation regularized connectionist temporal classification loss for text recognition: A simple yet effective approach. In AAAI, pages 7441–7449, 2024. 7

[88] Mingkai Zheng, Shan You, Fei Wang, Chen Qian, Changshui Zhang, Xiaogang Wang, and Chang Xu. ReSSL: Relational self-supervised learning with weak augmentation. In NeurIPS, pages 2543–2555, 2021. 3