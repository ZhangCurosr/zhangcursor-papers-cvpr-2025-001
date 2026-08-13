# DELT: A Simple Diversity-driven EarlyLate Training for Dataset Distillation

Zhiqiang Shen<sup>\*</sup> Ammar Sherif<sup>\*</sup> Zeyuan Yin Shitong Shao

VILA Lab, MBZUAI

{zhiqiang.shen, zeyuan.yin}@mbzuai.ac.ae {ammarsherif90, 1090784053sst}@gmail.com

## Abstract

Recent advances in dataset distillation have led to solutions in two main directions. The conventional batch-to-batch matching mechanism is ideal for small-scale datasets and includes bi-level optimization methods on models and syntheses, such as FRePo, RCIG, and RaT-BPTT, as well as other methods like distribution matching, gradient matching, and weight trajectory matching. Conversely, batchto-global matching typifies decoupled methods, which are particularly advantageous for large-scale datasets. This approach has garnered substantial interest within the community, as seen in SRe<sup>2</sup>L, G-VBSM, WMDD, and CDA. A primary challenge with the second approach is the lack of diversity among syntheses within each class since samples are optimized independently and the same global supervision signals are reused across different synthetic images. In this study, we propose a new Diversity-driven EarlyLate Training (DELT) scheme to enhance the diversity of images in batch-to-global matching with less computation. Our approach is conceptually simple yet effective, it partitions predefined IPC samples into smaller subtasks and employs local optimizations to distill each subset into distributions from distinct phases, reducing the uniformity induced by the unified optimization process. These distilled images from the subtasks demonstrate effective generalization when applied to the entire task. We conduct extensive experiments on CIFAR, Tiny-ImageNet, ImageNet-1K, and its sub-datasets. Our approach outperforms the pre vious state-of-the-art by 2∼5% on average across different datasets and IPCs (images per class), increasing diversity per class by more than 5% while reducing synthesis time by up to 39.3% for enhancing the training efficiency.

## 1. Introduction

In the era of large models and large datasets, dataset distillation has emerged as a crucial strategy to enhance training efficiency and make advanced technologies more accessible and affordable for the general public. Previous approaches [3–5, 13, 17, 20, 33, 39, 40, 44] primarily employ a batch-to-batch matching technique, where information like features, gradients, and trajectories from a local original data batch are used to supervise and train a corresponding batch of generated data. The strength of this method lies in its ability to capture fine-grained information from the original data, as each batch’s supervision signals vary. However, the downside is the necessity to repeatedly input both original and generated data for each training iteration, which significantly increases memory usage and computational costs. Recently, a new decoupled method [18, 36, 37] has been proposed to separate the model training and data synthesis, also it leverages the batch-to-global matching to avoid inputting original data during distilled data generation. This solution has demonstrated great advantage on large-scale datasets like ImageNet-1K [26, 37] and ImageNet-21K [36]. However, as shown in Fig. 2, a significant limitation of this approach is the lack of diversity caused by the mechanism of synthesizing each data point individually, where supervision is repetitively applied across various synthetic images. For instance, SRe<sup>2</sup>L [37] utilizes globally-counted layer-wise running means and variances from the pre-trained model for supervising different intra-class image synthesis. This methodology results in a severely limited diversity within the same category of generated images.

![](images/24f0abd35cd79c11c6835f35484580b071e2d7f6dfdebcc2917969fe8fbff8e9.jpg)  
Figure 1. Distilling datasets to IPC<sub>N</sub> requires N ×T iterations in traditional distillation processes (left) but fewer iterations in our EarlyLate strategy (right). IPC<sub>1:N</sub> represents a set of images from 1 to N. Red shaded area is our saved computational cost.

![](images/8749f953b9c6125cf981c9b8540724a2d7a2d957affdf3963f3b7e19d594420a.jpg)

![](images/9ea21ed3c01738ab657df12fcd155c3081a0c6e3f48b5d70851cf2aa37fcdcb8.jpg)  
Figure 2. Left: Intra-class semantic cosine similarity after a pretrained ResNet-18 model on ImageNet-1K dataset, lower values are better. Right: Synthetic images from $\mathrm { S R e ^ { 2 } I }$ , CDA and our DELT.

To address this issue, a few prior studies [26, 29] have proposed to enlarge diversity within each class. For instance, G-VBSM [26] utilizes a diverse set of local-matchglobal matching signals derived from multiple backbones and statistical metrics, to achieve more precise and effective matching than the singular model. However, as the diversity of matching models grows, the overall complexity of the framework also increases thus diminishing its conciseness. RDED [29] crops each original image into multiple patches and ranks these using realism scores generated by an observer model. Then it stitches every four chosen patches from previous stage into a single new image to produce IPC-numbered distilled images for each class. RDED is efficient to combine multiple images but does not enhance or optimize the visual content on the distilled dataset, thus the diversity and richness of information are largely dependent on the distribution of the original dataset.

Our solution, termed the EarlyLate training scheme, is straightforward and also orthogonal to these prior methods: by initializing each image in the same category at a different starting point for optimization, we ensure that the final optimized results vary significantly across images. We also use teacher-ranked real image patches to initialize the synthetic images. This prevents some images from being short-optimized and ensures they provide sufficient information. As shown in Fig. 1 of the computation comparison, our approach not only enhances intra-class diversity but also dramatically reduces the computational load of the training process with 39.3% on ImageNet-1K. Specifically, while conventional training requires T optimization iterations per image or batch, in our EarlyLate scheme, the first image undergoes $T _ { 1 }$ iterations (where $T _ { 1 } = T )$ . Subsequent batches are processed with progressively fewer iterations, such as $T _ { 2 } ~ ( T _ { 2 } = T _ { 1 } - { \bf R I } ^ { 1 } )$ for the next set, and so forth. The iterations for the final batch are reduced to RI which is $1 / j$ of the standard count (where typically $j = 4$ or 8), meaning the total number of optimization iterations required is just about $2 / 3$ of prior batch-to-global matching methods, such as $\mathrm { S R e ^ { 2 } L }$ and CDA. We further visualize the average cosine similarity between each sample of 50 IPCs using the associated cluster centroid within the same class on ImageNet-1K, as shown in Fig. 2 left subfigure, our DELT illustrates a smaller similarity significantly, and also shows substantially better visual diversity than other counterparts across all classes, as in the right subfigure of Fig. 2.

We conduct extensive experiments on various datasets of CIFAR-10, Tiny-ImageNet, ImageNet-1K and its subsets. On ImageNet-1K, our proposed approach achieves 66.1% under IPC 50 with ResNet-101, outperforming previous state-of-the-art RDED by 4.9%. On small-scale datasets of CIFAR-10, our approach also obtains 2.5% and 19.2% improvement over RDED and $\mathrm { S R e ^ { 2 } I }$ using ResNet-101.

Our main contributions in this work are as follows:

• We propose a simple yet effective EarlyLate training scheme for dataset distillation to enhance intra-class diversity of synthetic images for batch-to-global matching.

• We demonstrate empirically that the proposed method can generate optimized images at different distances with a fast speed, to enlarge informativeness among generations.

• We conducted extensive experiments and ablations on various datasets across different scales to prove the effectiveness of the proposed approach.

## 2. Related Work

Dataset Distillation. Dataset distillation or condensation [33] focuses on creating a compact yet representative subset from a large original dataset. This enables more efficient model training while maintaining the ability to evaluate on the original test data distribution and achieve satisfactory performance. Previous works [3–5, 13, 17, 20, 33, 39, 40, 44] mainly designed how to better match the distribution between original data and generated data in a batch-to-batch manner, such as the distribution of features [39], gradients [40], or the model weight trajectories [3, 5]. The primary optimization method used is bi-level optimization [19, 38], which involves optimizing model parameters and updating images simultaneously. For instance, using gradient matching, the process can be formulated as to minimize the gradient distance:

![](images/8d0fadb811ed938994d65554a98b6d2ff81a51b5e6a1c98cab96ef641dfd122b.jpg)  
Figure 3. Batch–to-batch vs. batch-to-global matching in DD. $\theta _ { f }$ indicates weights are pretrained and frozen in synthesis stage.

$$
\operatorname* { m i n } _ { S \in \mathbb { R } ^ { N \times d } } D \left( \nabla _ { \theta } \ell ( S ; \theta ) , \nabla _ { \theta } \ell ( \mathcal { T } ; \theta ) \right) = D ( S , \mathcal { T } ; \theta ) ,\tag{1}
$$

where the function $D ( \cdot , \cdot )$ is defined as a distance metric such as MSE [34], θ denotes the model parameters, and $\nabla _ { \boldsymbol { \theta } } \ell ( \cdot ; \boldsymbol { \theta } )$ represents the gradient, utilizing either the original dataset $\tau$ or its synthetic version S. N is the number of d-dimensional synthetic data. During distillation, the synthetic dataset S and model θ are updated alternatively,

$$
\begin{array} { r } {  { \mathcal { S } } \gets  { \mathcal { S } } - \lambda \nabla _ { s } D (  { \mathcal { S } } , \mathcal { T } ; \theta ) , \quad \theta \gets \theta - \eta \nabla _ { \theta } \ell ( \theta ;  { \mathcal { S } } ) . } \end{array}\tag{2}
$$

where $\lambda$ and η are learning rates designated for $s$ and θ. Diversity in Dataset Distillation. Batch-to-global matching used in [7, 18, 26, 35–37] tracks the distribution of BN statistics derived from original dataset for the local batch synthetic data. However, this type of approach can easily encounter diversity issues within the same class due to the optimization objective. Fig. 3 illustrates the difference of batch-to-batch and batch-to-global matching mechanisms, where b represents a local batch in data T and S.

Moreover, for the recent advances of multi-stage dataset distillation methods, MDC [13] proposes to compress multiple condensation processes into a single one by including an adaptive subset loss on top of the basic condensation loss, so that to obtain datasets with multiple sizes. PDD [4] generates multiple small batches of synthetic images, each batch is conditioned on the accumulated data from previous batches. Unlike PDD, our current synthetic batch is independent with different operation iterations and not relevant to any previous batches. D3 [23] partitions large datasets into smaller subtasks and employs locally trained experts to distill each subset into distributions. These distilled distributions from the subtasks demonstrate effective generalization when applied to the entire task. The recently proposed LPLD [35] batches images by class, leveraging the natural independence between classes, and introduces classwise supervision for alignment.

## 3. Approach

Preliminaries. The objective of a regular dataset distillation task is to generate a compact synthetic dataset $s =$ $\{ ( \hat { \pmb x } _ { 1 } , \hat { \pmb y } _ { 1 } ) , \dotsc , \left( \hat { \pmb x } _ { | S | } , \hat { \pmb y } _ { | S | } \right) \}$ as a student dataset that captures a substantial amount of the information from a larger labeled dataset $\mathcal { T } = \{ ( \pmb { x } _ { 1 } , \pmb { y } _ { 1 } ) , . . . , \left( \pmb { x } _ { | \mathcal { T } | } , \pmb { y } _ { | \mathcal { T } | } \right) \}$ , which serves as the teacher dataset. Here, yˆ represents the soft label for the synthetic sample ${ \hat { \mathbf { x } } } .$ , and the size of S is much smaller than $\tau ,$ , yet it retains the essential information of the original dataset $\tau .$ . The learning goal using this distilled data is to train a post-validation model with parameters θ:

$$
\theta _ { S } = \underset { \theta } { \arg \operatorname* { m i n } } \mathcal { L } _ { S } ( \theta ) ,\tag{3}
$$

$$
\begin{array} { r } { \mathcal { L } _ { S } ( \pmb { \theta } ) = \mathbb { E } _ { ( \hat { \pmb { x } } , \hat { \pmb { y } } ) \in S } \left[ \ell \big ( \phi _ { \pmb { \theta } _ { S } } \big ( \hat { \pmb { x } } \big ) , \hat { \pmb { y } } ; \pmb { \theta } \big ) \right] , } \end{array}\tag{4}
$$

where ℓ is a standard loss function such as soft crossentropy and $\phi _ { \pmb { \theta } _ { \pmb { \xi } } }$ represents the model.

The primary aim of dataset distillation is to produce synthetic data that ensures minimal performance difference between models trained on the synthetic dataset $s$ and those trained on the original dataset $\tau$ using validation data V . The optimization procedure for generating S is given by:

$$
\begin{array} { c } { { \arg \operatorname* { m i n } \left( \operatorname* { s u p } \left\{ \right. \ell \left( \phi _ { \theta _ { \tau } } \left( x _ { \mathrm { v a l } } \right) , y _ { \mathrm { v a l } } \right) \right. } \right. }  \\ { { \left. \left. - \ell \left( \phi _ { \theta s } \left( x _ { v a l } \right) , y _ { v a l } \right) _ { \left( x _ { v a l } , y _ { v a l } \right) \sim V } . \right. } } \end{array}\tag{5}
$$

where $( \pmb { x } _ { v a l } , \pmb { y } _ { v a l } )$ are the sample and label pairs in the validation set of the real dataset T . The learning task then focuses on the <data, label> pairs within S, maintaining a balanced representation of distilled data across each class.

Initialization. Previous dataset distillation methods [26, 36, 37] on large-scale datasets like ImageNet-1K and 21K employ Gaussian noise by default for data initialization in the synthesis phase. However, Gaussian noise is random and lacks any semantic information. Intuitively, using real images provide a more meaningful and structured starting point, and this structured start can lead to quicker convergence during optimization because the initial data already contains useful features and patterns that are closer to the target distribution, which further enhances realism, quality, and generalization of the synthesized images. As shown in Fig. 2 right subfigure, our generated images exhibit both diversity and a high degree of realism in some cases.

Selection Criteria. Here, we introduce how to select real image patches to initialize the synthetic images. In our final syntheses, a significant fraction of our data has been subject to limited optimization iterations, making effective initialization crucial. A proper initialization also dramatically minimizes the overall computational load required for the updating on data. Prior approach [29] has demonstrated that choosing representative data patches from the original dataset without training can yield favorable performance without any additional training. Our observation, <sup>a</sup>however, underscores that applying iterative refinement to original patches can lead to markedly improved results.

![](images/aad9125cf3a15d675a1158b0dc992080a15cd5a835ee256c6543eb68e2d601f6.jpg)  
Figure 4. The proposed DELT learning procedure via a multi-round EarlyLate scheme.

As illustrated in Fig. 5, our selection criterion is based on a pretrained teacher model as a ranker, we calculate all patches’ probabilities and sort them as the initialization pool. Then, we choose a patch of images scoring closer to the per-class median as initialization for optimization. <sup>2</sup>More details regarding initialization and order can be found in Appendix. The motivation is that such images have a medium difficulty level to the teacher, so they have more room for information enhancement via distillation gradients. We further empirically validate this strategy by comparing different strategies in Table 4b.

![](images/e25fba4c6ad962b33c322e133f8c2cc5c1b8337c291bf962219f761ea9d53631.jpg)  
Figure 5. Selection criteria with a teach ranker.

Diversity-driven IPC Concatenation Training. As shown crop poolin Fig. 4, to further emphasize diversity and avoid potential distribution bias from initialization, we optimize the initialized images starting from different points. The motivation behind this design is that different data samples require varying numbers of iterations to converge as the early stopping [22] from other research domain. Importantly, as images become easier to predict with more updates by class labels, training primarily on easy data points can hinder model generalization. Therefore, our method enhances generalization by generating data samples with varying difficulty levels, acting as a regularizer by limiting the optimization process to a smaller volume of image pixel space.

Previous work [1] studies how to perform simple early stopping on different layers’ weights with progressive retraining to mitigate noisy labels. Unlike it, we are pioneering to study both early and late training when optimizing data. Moreover, we improve the efficiency of our approach by performing gradient updates in a single scan. Initially, we conduct a single gradient loop, continually introducing new Unrolled versiondata for distillation by concatenating them at different time stamps. Consequently, the M batch receives the synthetic images of all preceding batches, $I P C _ { 0 : M k - 1 } $ , as final generations. This process can be simplified as follows:

$$
I P C _ { 0 : M k - 1 } = \underbrace { \underbrace { \left[ \hat { \pmb { x } } _ { 0 } , \hat { \pmb { x } } _ { 1 } , \ldots , \hat { \pmb { x } } _ { k - 1 } , \ldots , \hat { \pmb { x } } _ { M k - 1 } \right] } _ { I P C _ { 0 : k - 1 } } } _ { \substack { \mathrm { . . . } } }\tag{6}
$$

where $\left[ \hat { \pmb { x } } _ { 0 } , \hat { \pmb { x } } _ { 1 } , \dots , \hat { \pmb { x } } _ { M k - 1 } \right]$ refers to the concatenation of generated images. M is the number of batches, k is the number of generated images in each batch. We train these different batches at different starting points, each batch goes through a completed learning phase, but the total number of iterations varies. Then, the multiple IPCs of xˆ are concatenated into a simple batch. Because of its early-late training property, we refer to this scheme as EarlyLate training. Data synthesis. Our EarlyLate optimization procedure can be formulated as a multi-stage training scheme:

$$
\operatorname { R o u n d } 1 { : } \operatorname { a r g m i n } _ { \mathcal { C } _ { \mathrm { I P C } _ { 0 : k - 1 } } , | \mathcal { C } | } \ell \left( \phi _ { \pmb { \theta } _ { \mathcal { T } } } \left( \widetilde { x } _ { \mathrm { I P C } _ { 0 : k - 1 } } \right) , \pmb { y } \right) + \mathcal { R } _ { \mathrm { r e g } }
$$

$$
\operatorname { R o u n d } \mathrm { M - 1 } { : } \operatorname * { a r g m i n } _ { \mathcal { C } _ { \mathrm { I P C } _ { 0 : M k - 1 } } , | \mathcal { C } | } \ell \left( \phi _ { \pmb { \theta } _ { \mathcal { T } } } \left( \widetilde { x } _ { \mathrm { I P C } _ { 0 : M k - 1 } } \right) , \pmb { y } \right) + \mathcal { R } _ { \mathrm { r e g } }\tag{7}
$$

where C is the target distilled dataset. The number of batches $\mathrm { ~ M ~ } > \mathrm { ~ 1 ~ } ( \mathrm { I f ~ M ~ } = \mathrm { ~ 1 ~ }$ , training will degenerate into a way without EarlyLate). This process is referred to in Fig. 4 bottom row. $\mathcal { R } _ { \mathrm { r e g } }$ is the regularization term, we also utilize the BatchNorm distribution regularization term as in SRe<sup>2</sup>L [37] to improve the quality of the generated images.

Training Procedure. Regarding concatenation training, we elaborate: Our EarlyLate enhances the diversity of the synthetic data by varying the number of iterations for different IPCs during data synthesis phase. This means the first IPC can be recovered for the largest number of iterations like 4K while the last IPC will only be recovered using 500 iterations. To make this process efficient, we share the recovery time (on the GPU) across the different IPCs via concatenation to minimize time as much as possible. Therefore the first image IPC will start recovery for a couple of iterations, and when it completes iteration 3,500 the last IPC will join it in the recovery phase to get its 500 iterations.

As illustrated in Fig. 4, our learning procedure is extremely simple using an incremental learning process: We split the total IPCs to be learned into multiple batches. The training begins with the first batch. Following a predefined number of iterations, the second batch commences its iterative training, and this process continues sequentially with subsequent batches. We define two types of optimization iterations for training: maximum iteration (MI) for the earliest batch training and round iteration (RI). MI presents the number of optimization iterations that the earliest batch goes through, i.e., the maximum number of iterations for the first batch’s gradient updating. RI represents the number of iterations used for each round in Fig. 4. It essentially indicates the iteration gap between the optimization of two adjacent batches. Batch-to-global matching algorithm [36] of Eq. 7 is utilized between each round. In our DELT, later sub-batches will join the previous sub-batches in the image recovery stage instead of freezing the earlier sub-batches.

## 4. Experiments

## 4.1. Datasets and Result Details

We first run DELT on five standard benchmark tests including CIFAR-10 (10 classes) [15], Tiny-ImageNet (200 classes) [16], ImageNet-1K (1,000 classes) [6] and it variants of ImageNette (10 classes) [8], and ImageNet-100 (100 classes) [32] with performances reported in Table 1 and Table 2. The evaluation protocol is following prior works [29, 37]. We compare DELT to six baseline dataset distillation algorithms including Matching Training Trajectories (MTT) [3], Improved Distribution Matching (IDM) [42], TrajEctory Matching with Constant Memory (TESLA) [5], Squeeze-Recover-Relabel $\mathrm { ( S R e ^ { 2 } L ) }$ [37], Difficulty-Aligned Trajectory-Matching (DATM) [11], Realistic-Diverse-Efficient Dataset Distillation (RDED) [29]. Following previous dataset distillation methods [29, 37, 40], we use ConvNet [9], ResNet-18/ResNet-101 [12], EfficientNet-B0 [30], MobileNet-V2 [25], MnasNet1 3 [31], and RegNet-Y-8GF [24], as our backbone for training or post-validation. All our experiments are conducted on 4× NVIDIA RTX 4090 GPUs.

As shown in Table 1, our approach establishes the new state-of-the-art accuracy in 13 out of 15 of the configurations on five datasets from small-scale CIFAR-10 to largescale ImageNet-1K using either relatively large backbone architecture of ResNet-101 or small MobileNet-v2, in many cases with significant margins of improvement. The results using small-scale architecture ConvNet are shown in Table 2, our approach also achieves the state-of-the-art accuracy in 7 out of 9 of the configurations on four datasets.

![](images/bad3c8995ee29ff137cf57331f47b525f3326b4836ff1081b8288e8ac71d0077.jpg)  
Figure 6. Mosaic splicing patterns on ImageNet-1K using real image patches as the initialization. In each block, the left column is the starting real image initialized samples and right is the final optimized syntheses. From top to bottom are images generated by early training and late training.

## 4.2. Cross-architecture generalization

An important characteristic of distilled datasets is their effectiveness in generalizing to novel training architectures. In this context, we assess the transferability of DELT’s distilled datasets tailored for ImageNet-1K with 10 images per class. Following previous studies [29, 37], we test our models using five distinct architectures: ResNet-18 [12], MobileNet-V2 [25], MnasNet1 3 [31], EfficientNet-B0 [30], and RegNet-Y-8GF [24]. As shown in Table 5, our proposed approach demonstrates significant better performance than other competitive methods on all these architectures.

## 4.3. Ablation Study

Mosaic splicing pattern. Mosaic stitching method [2] in RDED selects four crops from the train set as the optimal hyper-parameter, and puts the contents of the four crops into a synthetic image that is directly used for post-validation. In this work, considering that we use different difficulty levels of selection for initialization, we examine different strategies of the Mosaic splicing patterns, including 1 × 1, 2 × 2, $3 \times 3 , 4 \times 4 ,$ and 5 × 5 patches, as illustrated in Fig. 6. The ablation results are shown in Table 4a, it can be observed that 1 × 1 achieves the best accuracy.

Initialization. We examine how different initialization strategies affect final performance, including: choosing lowest probability crops, medium probability crops and highest probability crops. Our results are shown in Table 4b. Overall, the performance gap between different strategies is not significant, and selecting the medium probability crops as the initialization achieves the best accuracy.

<table><tr><td rowspan="2">Dataset</td><td rowspan="2">IPC</td><td colspan="3">ResNet-18</td><td colspan="3">ResNet-101</td><td colspan="2">MobileNet-v2</td></tr><tr><td> $\mathrm { S R e ^ { 2 } L } \left[ 3 7 \right]$ </td><td> $\mathrm { R D E D } \left[ 2 9 \right]$ </td><td> $\mathrm { D E L T } \left( \mathrm { O u r s } \right)$ </td><td> $\mathrm { S R e ^ { 2 } L } \left[ 3 7 \right]$ </td><td> $\mathrm { R D E D } \left[ 2 9 \right]$ </td><td>DELT (Ours)</td><td>RDED [29] DELT (Ours)</td><td></td></tr><tr><td rowspan="3">CIFAR-10</td><td>1</td><td> $1 6 . 6 \pm 0 . 9$ </td><td> $2 2 . 9 \pm 0 . 4$ </td><td> ${ \bf 2 4 . 0 \pm 0 . 8 }$ </td><td> $1 3 . 7 \pm 0 . 2$ </td><td> $1 8 . 7 \pm 0 . 1$ </td><td> ${ \bf 2 0 . 4 } \pm { \bf 1 . 0 }$ </td><td> $1 8 . 1 \pm 0 . 9$ </td><td> ${ \bf 2 0 . 2 \pm 0 . 4 }$ </td></tr><tr><td>10</td><td> $2 9 . 3 \pm 0 . 5$ </td><td> $3 7 . 1 \pm 0 . 3$ </td><td> ${ \bf 4 3 . 0 \pm 0 . 9 }$ </td><td> $2 4 . 3 \pm 0 . 6$ </td><td> $3 3 . 7 \pm 0 . 3$ </td><td> $3 7 . 4 \pm 1 . 2$ </td><td> $2 9 . 2 \pm 1 . 1$ </td><td> ${ \bf 2 9 . 3 \pm 0 . 3 }$ </td></tr><tr><td>50</td><td> $4 5 . 0 \pm 0 . 7$ </td><td> $6 2 . 1 \pm 0 . 1$ </td><td> ${ \bf 6 4 . 9 \pm 0 . 9 }$ </td><td> $3 4 . 9 \pm 0 . 1$ </td><td> $5 1 . 6 \pm 0 . 4$ </td><td> ${ \bf 5 4 . 1 \pm 0 . 8 }$ </td><td> $3 9 . 9 \pm 0 . 5$ </td><td> ${ \bf 4 2 . 9 \pm 2 . 2 }$ </td></tr><tr><td rowspan="3">ImageNette</td><td>1</td><td> $1 9 . 1 \pm 1 . 1$ </td><td> ${ \bf 3 5 . 8 \pm 1 . 0 }$ </td><td> $2 4 . 1 \pm 1 . 8$ </td><td> $1 5 . 8 \pm 0 . 6$ </td><td> ${ \bf 2 5 . 1 \pm 2 . 7 }$ </td><td> $1 9 . 4 \pm 1 . 7$ </td><td> $2 6 . 4 \pm 3 . 4$ </td><td> $1 9 . 1 \pm 1 . 0$ </td></tr><tr><td>10</td><td> $2 9 . 4 \pm 3 . 0$ </td><td> $6 1 . 4 \pm 0 . 4$ </td><td> ${ \bf 6 6 . 0 \pm 1 . 4 }$ </td><td> $2 3 . 4 \pm 0 . 8$ </td><td> $5 4 . 0 \pm 0 . 4$ </td><td> ${ \pm } 5 5 . 4 \pm 6 . 2$ </td><td> $5 2 . 7 \pm 6 . 6$ </td><td> ${ \bf 6 4 . 7 \pm 1 . 4 }$ </td></tr><tr><td>50</td><td> $4 0 . 9 \pm 0 . 3$ </td><td> $8 0 . 4 \pm 0 . 4$ </td><td> ${ \bf 8 8 . 2 \pm 1 . 2 }$ </td><td> $3 6 . 5 \pm 0 . 7$ </td><td> $7 5 . 0 \pm 1 . 2$ </td><td> ${ \bf 8 3 . 3 \pm 1 . 1 }$ </td><td> $8 0 . 0 \pm 0 . 0$ </td><td> ${ \bf 8 5 . 7 \pm 0 . 4 }$ </td></tr><tr><td rowspan="3">Tiny-ImageNet</td><td>1</td><td> $2 . 6 2 \pm 0 . 1$ </td><td> ${ \bf 9 . 7 \pm 0 . 4 }$ </td><td> $9 . 3 \pm 0 . 5$ </td><td> $1 . 9 \pm 0 . 1$ </td><td> $3 . 8 \pm 0 . 1$ </td><td> ${ \bf 5 . 6 \pm 1 . 0 }$ </td><td> $3 . 5 \pm 0 . 1$ </td><td> ${ \bf 3 . 5 \pm 0 . 5 }$ </td></tr><tr><td>10</td><td> $1 6 . 1 \pm 0 . 2$ </td><td> $4 1 . 9 \pm 0 . 2$ </td><td> ${ \bf 4 3 . 0 \pm 0 . 1 }$ </td><td> $1 4 . 6 \pm 1 . 1$ </td><td> $2 2 . 9 \pm 3 . 3$ </td><td> ${ \bf 4 2 . 8 \pm 0 . 9 }$ </td><td> $2 4 . 6 \pm 0 . 1$ </td><td> ${ \bf 2 6 . 5 \pm 0 . 5 }$ </td></tr><tr><td>50</td><td> $4 1 . 1 \pm 0 . 4$ </td><td> ${ \bf 5 8 . 2 \pm 0 . 1 }$ </td><td> $5 5 . 7 \pm 0 . 5$ </td><td> $4 2 . 5 \pm 0 . 2$ </td><td> $4 1 . 2 \pm 0 . 4$ </td><td> ${ \bf 5 8 . 5 \pm 0 . 3 }$ </td><td> $4 9 . 3 \pm 0 . 2$ </td><td> ${ \bf 5 1 . 3 \pm 0 . 5 }$ </td></tr><tr><td rowspan="3">ImageNet-100</td><td>10</td><td> $9 . 5 \pm 0 . 4$ </td><td> ${ \bf 3 6 . 0 \pm 0 . 3 }$ </td><td> $2 8 . 2 \pm 1 . 5$ </td><td> $6 . 4 \pm 0 . 1$ </td><td> ${ \bf 3 3 . 9 \pm 0 . 1 }$ </td><td> $2 2 . 4 \pm 3 . 3$ </td><td> ${ \bf 2 3 . 6 \pm 0 . 7 }$ </td><td> $1 5 . 8 \pm 0 . 2$ </td></tr><tr><td>50</td><td> $2 7 . 0 \pm 0 . 4$ </td><td> $6 1 . 6 \pm 0 . 1$ </td><td> ${ \bf 6 7 . 9 \pm 0 . 6 }$ </td><td> $2 5 . 7 \pm 0 . 3$ </td><td> $6 6 . 0 \pm 0 . 6$ </td><td> ${ \bf 7 0 . 8 \pm 2 . 3 }$ </td><td> $5 1 . 5 \pm 0 . 8$ </td><td> ${ \pm } 5 5 . 0 \pm 1 . 8$ </td></tr><tr><td>100</td><td></td><td> $7 4 . 5 \pm 0 . 4$ </td><td> ${ \bf 7 5 . 1 \pm 0 . 2 }$ </td><td></td><td> $7 3 . 5 \pm 0 . 8$ </td><td> ${ 7 7 . 6 \pm 1 . 8 }$ </td><td> $7 0 . 8 \pm 1 . 1$ </td><td> ${ \bf 7 6 . 7 \pm 0 . 3 }$ </td></tr><tr><td rowspan="3">ImageNet-1K</td><td>10</td><td> $2 1 . 3 \pm 0 . 6$ </td><td> $4 2 . 0 \pm 0 . 1$ </td><td> ${ \bf 4 6 . 1 \pm 0 . 4 }$ </td><td> $3 0 . 9 \pm 0 . 1$ </td><td> $4 8 . 3 \pm 1 . 0$ </td><td> ${ \bf 4 8 . 5 \pm 1 . 6 }$ </td><td> $3 3 . 1 \pm 1 . 2$ </td><td> ${ \bf 3 5 . 5 \pm 0 . 7 }$ </td></tr><tr><td>50</td><td> $4 6 . 8 \pm 0 . 2$ </td><td> $5 6 . 5 \pm 0 . 1$ </td><td> ${ \bf 5 9 . 2 \pm 0 . 4 }$ </td><td> $6 0 . 8 \pm 0 . 5$ </td><td> $6 1 . 2 \pm 0 . 4$ </td><td> ${ \bf 6 6 . 1 \pm 0 . 5 }$ </td><td> $5 2 . 8 \pm 0 . 4$ </td><td> ${ \bf 5 6 . 2 \pm 0 . 3 }$ </td></tr><tr><td>100</td><td> $5 2 . 8 \pm 0 . 3$ </td><td> $5 9 . 8 \pm 0 . 1$ </td><td> ${ \bf 6 2 . 4 \pm 0 . 2 }$ </td><td> $6 2 . 8 \pm 0 . 2$ </td><td></td><td> ${ \bf 6 7 . 6 \pm 0 . 3 }$ </td><td> $5 6 . 2 \pm 0 . 1$ </td><td> ${ \bf 5 8 . 9 \pm 0 . 3 }$ </td></tr></table>

Table 1. Comparison with SOTA dataset distillation methods using relatively large-scale backbones on five benchmarks across differen scales. MobileNet-v2 is modified to match the low resolutions of CIFAR-10 and Tiny-ImageNet following [41]. Due to the limited table space, some prior methods that are slightly weaker or comparable with RDED are not listed, such as CDA and G-VBSM. Since IPC = 1 is not applicable to use EarlyLate strategy, thus under $\mathrm { I P C } = 1$ setting, the single image in each class is optimized with a constant iteration.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">IPC</td><td colspan="6">ConvNet</td></tr><tr><td>MTT [3]</td><td>IDM [42]</td><td>TESLA [5]</td><td>DATM [11]</td><td>RDED [29]</td><td>DELT (Ours)</td></tr><tr><td rowspan="3">Tiny-ImageNet</td><td>1</td><td> $8 . 8 \pm 0 . 3$ </td><td> $1 0 . 1 \pm 0 . 2$ </td><td></td><td> ${ \bf 1 7 . 1 \pm 0 . 3 }$ </td><td> $1 2 . 0 \pm 0 . 1$ </td><td> $1 2 . 4 \pm 0 . 8$ </td></tr><tr><td>10</td><td> $2 3 . 2 \pm 0 . 2$ </td><td> $2 1 . 9 \pm 0 . 3$ </td><td></td><td> $3 1 . 1 \pm 0 . 3$ </td><td> $3 9 . 6 \pm 0 . 1$ </td><td> ${ \bf 4 0 . 0 \pm 0 . 4 }$ </td></tr><tr><td>50</td><td> $2 8 . 0 \pm 0 . 3$ </td><td> $2 7 . 7 \pm 0 . 3$ </td><td></td><td> $3 9 . 7 \pm 0 . 3$ </td><td> $4 7 . 6 \pm 0 . 2$ </td><td> ${ \bf 4 8 . 6 \pm 0 . 2 }$ </td></tr><tr><td rowspan="3">ImageNet-100</td><td>10</td><td></td><td> $1 7 . 1 \pm 0 . 6$ </td><td></td><td></td><td> ${ \bf 2 9 . 6 \pm 0 . 1 }$ </td><td> $2 4 . 7 \pm 1 . 5$ </td></tr><tr><td>50</td><td></td><td> $2 6 . 3 \pm 0 . 4$ </td><td></td><td></td><td> $5 0 . 2 \pm 0 . 2$ </td><td> ${ \bf 5 1 . 9 \pm 1 . 1 }$ </td></tr><tr><td>100</td><td></td><td></td><td></td><td></td><td> $5 8 . 6 \pm 0 . 4$ </td><td> ${ \bf 6 1 . 5 \pm 0 . 5 }$ </td></tr><tr><td rowspan="3">ImageNet-1K</td><td>1</td><td></td><td></td><td> $7 . 7 \pm 0 . 2$ </td><td></td><td> $6 . 4 \pm 0 . 1$ </td><td> ${ \bf 8 . 8 \pm 0 . 5 }$ </td></tr><tr><td>10</td><td></td><td></td><td> $1 7 . 8 \pm { 1 . 3 }$ </td><td></td><td> $2 0 . 4 \pm 0 . 1$ </td><td> ${ \bf 3 1 . 3 \pm 0 . 8 }$ </td></tr><tr><td>50</td><td>1</td><td></td><td> $2 7 . 9 \pm 1 . 2$ </td><td></td><td> $3 8 . 4 \pm 0 . 2$ </td><td> ${ \bf 4 1 . 7 \pm 0 . 1 }$ </td></tr></table>

Table 2. Comparison with SOTA dataset distillation methods using small-scale backbone architecture on three datasets. Following [3, 29, 42], Conv-4 for Tiny-ImageNet and ImageNet-1K, Conv-6 for ImageNet-100. Entries marked with “-” are missing due to scalability issue.

Optimization iterations. We examine two types of optimization iterations: maximum iteration (MI) for the earliest batch training and round iteration (RI) as the iteration gap between the two adjacent batches. As shown in Table 4c, we test MI values of 1K, 2K, and 4K, using 500 and 1K iterations for each RI. Note that when MI is set to 1K, it is not feasible to use 1K as RI. The results show that 4K (same as [36, 37]) MI and 500 RI achieves the best accuracy.

Table 4e. While the first two methods produce more realistic images, each image contains limited information. In contrast, our method achieves the best final performance.

Without real image initialization. Our EarlyLate strategy enhances the performance of 1∼2% over the initialization. Without initialization, our method improves consistently with 2.4% as shown in Table 3.

## 4.4. Computational Analysis

Early-only vs. EarlyLate. Early-only is equivalent to using constant MI to optimize each image. This will transform to baseline batch-to-global matching of CDA [36] + real image initialization. Our results in Table 4d clearly show that the EarlyLate training bring a significant improvement on final performance. More importantly, this strategy is the key factor in enhancing generation diversity.

Real image stitching vs. Minimax diffusion vs. Ours. We further compare our approach with real image stitching [29] and diffusion generation [10]. The results are presented in

For image optimization-based methods like $\mathrm { S R e ^ { 2 } L }$ and CDA, the total computational cost is calculated as $N \times T .$ where N is the MI. In our EarlyLate scheme, the first batch images undergo $T _ { 1 }$ iterations (where $T _ { 1 } = T )$ . Subsequent batches are processed with progressively fewer iterations, such as $T _ { 2 } ~ ( T _ { 2 } = T _ { 1 } - \mathrm { R I } )$ for the next set, and so forth. The iterations for the final batch are reduced to RI which is $1 / j$ of the standard count (where j = 4 or 8 in our ablation), the total number of our optimization iterations required is $\begin{array} { r } { N \times T - { \frac { j ( j - 1 ) } { 2 } } \mathbf { R I } } \end{array}$ , which is roughly $2 / 3$ of prior batch-to-global matching methods. Our real time consumptions for data generation are shown in Table 6, note that the smaller the dataset like CIFAR, the more time is spent on loading and processing the data, rather than training.

<table><tr><td>Selection criteria</td><td>Top 1 acc</td></tr><tr><td>Lowest probability</td><td>57.55</td></tr><tr><td>Medium probability</td><td>57.67</td></tr><tr><td>Highest probability</td><td>57.03</td></tr></table>

<table><tr><td>Strategy Acc.</td><td>SRe2L [37] w/o Init w/o EarlyLate 46.8</td><td>SRe2L [37] w/o Init w/ EarlyLate CDA [36] w/o Init w/o EarlyLate  $5 3 . 4 _ { ( + 6 . 6 ) }$ </td><td>53.5</td><td>CDA [36] w/o Init w/ EarlyLate  ${ \bar { \bf 5 } } 5 . { \bf 9 } _ { ( + 2 . 4 ) }$ </td></tr></table>

Table 3. Performance comparison without real image initialization on ImageNet-1K with IPC 50.

<table><tr><td># Patches</td><td>Top 1 acc</td></tr><tr><td> $1 \times 1$ </td><td>57.57</td></tr><tr><td> $2 \times 2$ </td><td>56.92</td></tr><tr><td> $3 \times 3$ </td><td>56.62</td></tr><tr><td> $4 \times 4$ </td><td>56.71</td></tr><tr><td> $5 \times 5$ </td><td>56.51</td></tr></table>

(a) Number of patches. Ablation on initializing different numbers of scoring patches. Results are from ResNet-18 on ImageNet-1K for 500 iterations to synthesize 50 IPCs. Our optimization-based method favors 1 × 1 initialized patch, and will involve inconsistency and noise using more objects.

<table><tr><td>Iterations (MI)</td><td>Round Iterations (RI) 500 1K</td></tr><tr><td>1K</td><td>44.87 43.71</td></tr><tr><td>2K</td><td>45.61 44.40</td></tr><tr><td>4K</td><td>46.42 44.66</td></tr></table>

(c) Round Iterations. Top-1 acc. of our method for IPC 10 using different round iterations with ResNet-18.  
(b) Selection criteria. Initializing 1 × 1 images selected according to teacher model’s probability

<table><tr><td>Dataset</td><td>CDA [36] + Our init.</td><td>Ours</td></tr><tr><td>ImageNet-1K</td><td>43.5</td><td>46.1</td></tr><tr><td>Tiny-ImageNet CIFAR-10</td><td>42.2 39.4</td><td>43.0 43.0</td></tr><tr><td></td><td>(d) Ablation on init. and EarlyLate under IPC 10.</td><td></td></tr><tr><td></td><td>MinimaxDiffusion [10]</td><td>Ours</td></tr><tr><td>IPC</td><td>RDED [29]</td><td></td></tr><tr><td>10 50</td><td>42.0 56.5</td><td>44.3 46.1 58.6 59.2</td></tr></table>

(e) Comparison with real and diffusion generated data.

Table 4. Ablation experiments on various aspects of our framework with ResNet-18 on ImageNet-1K.
<table><tr><td colspan="2">Recover / Validation</td><td>ResNet-18</td><td>EfficientNet-B0</td><td>MobileNet-V2</td><td>MnasNet1_3</td><td>RegNet-Y-8GF</td></tr><tr><td rowspan="5"></td><td> $\mathrm { S R e ^ { 2 } L } [ 3 7 ] ^ { \dagger }$ </td><td>41.9</td><td>41.9</td><td>33.1</td><td>39.3</td><td>51.5</td></tr><tr><td>CDA [36]</td><td>42.2</td><td>43.9</td><td>34.2</td><td>39.7</td><td>52.9</td></tr><tr><td>G-VBSM [26]</td><td>41.4</td><td>42.6</td><td>33.5</td><td>40.1</td><td>52.2</td></tr><tr><td>RDED [29]</td><td>42.3</td><td>42.8</td><td>34.4</td><td>40.0</td><td>54.8</td></tr><tr><td>Ours</td><td> $4 6 . 4 _ { ( + 4 . 1 ) }$ </td><td> ${ \bf 4 7 . 1 } _ { ( + 4 . 3 ) }$ </td><td> ${ 3 6 . 1 } _ { ( + 1 . 7 ) }$ </td><td> ${ \bf 4 0 . 7 } _ { ( + 0 . 7 ) }$ </td><td> ${ \pm } 7 . 5 _ { ( + 2 . 7 ) }$ </td></tr></table>

Table 5. Cross-architecture generalization. Results are evaluated on IPC 10. <sup>†</sup> is reproduced following CDA’s configuration.

## 4.5. Visualization of DELT

Fig. 7 illustrates a comprehensive visual comparison between randomly selected synthetic images from our distilled dataset and those from the real image patches [29], MinimaxDiffusion [10], MTT [3], IDC [14], SRe<sup>2</sup>L [37], SCDD [43], CDA [36] and G-VBSM [26] distilled data. It can be observed that the images generated by each method have their own characteristics. MinimaxDiffusion leverages the diffusion model to synthesize images which is close to the real ones. However, as in our above ablation, both real and diffusion-generated data are inferior to ours. MTT results show noticeable artifacts and distortions, the objects in all images are located in the middle of the generations, the diversity is limited. IDC results also show distorted and less recognizable dog images, but diversity is increased. $\mathrm { S R e ^ { 2 } L }$ exhibits some dog features but with significant distortions and similar simple background. SCDD shows more recognizable dog features but still the color is simple and monochromatic, the same situation happens in CDA. G-

VBSM shows more colorful patterns, possibly due to recovery from multiple different networks, but all generations are in the same pattern and the diversity is not large. Our approach’s synthetic images exhibit a higher degree of diversity, including both compressed distorted images from long-optimized initializations and clear, recognizable dog images from short-optimized initializations, a unique capability not present in other methods.

## 4.6. Application I: Data-free Network Pruning

Our distilled dataset acts as a multifunctional training tool and boosts the adaptability for diverse downstream applications. We validate its utility in the scenario of data-free network pruning [28]. Table 7 shows the applicability of our dataset in this task when pruning 50% weights, where it significantly surpasses previous methods such as $\mathrm { S R e ^ { 2 } L }$ and RDED under IPC 10 and 50.

## 4.7. Application II: Continual Learning

We examine the effectiveness of DELT generated images in the continual learning scenario. Following the setup in prior studies [37, 39], we perform 100-step class-incremental experiments on ImageNet-1K, comparing our results with the baselines G-VBSM and SRe<sup>2</sup>L. As shown in Fig. 8, our cant benefits of deploying DELT, particularly in mitigating the challenges of continual learning.

![](images/ad21d5846569afa453ea95204599abee33b98d64c2baeb34c804af357830e152.jpg)  
Figure 7. Distilled dataset visualization compared with other image optimization-based methods.

<table><tr><td rowspan="2">Method</td><td colspan="3">Dataset (hours) under same 4K iterations for all methods</td></tr><tr><td>ImageNet-1K</td><td>Tiny-ImageNet</td><td>CIFAR-10</td></tr><tr><td>G-VBSM [26]</td><td>114.1</td><td>5.5</td><td>0.195</td></tr><tr><td>SRe2L [37]</td><td>29.0</td><td>5.0</td><td>0.084</td></tr><tr><td>CDA [36]</td><td>29.0</td><td>5.0</td><td>0.084</td></tr><tr><td>Ours (RI = 1K)</td><td> ${ \bf 1 8 . 8 } _ { ( \downarrow 3 5 . 2 \% ) }$ </td><td> $3 . 6 _ { ( \downarrow 2 8 . 0 \% ) }$ </td><td> $0 . 0 8 4 _ { ( \downarrow 0 . 0 \% ) }$ </td></tr><tr><td>Ours (RI = 500)</td><td> $\mathbf { 1 7 . 6 } _ { ( \downarrow 3 9 . 3 \% ) }$ </td><td> $3 . 4 _ { ( \downarrow 3 2 . 0 \% ) }$ </td><td> $0 . 0 8 3 _ { ( \downarrow 1 . 1 \% ) }$ </td></tr></table>

Table 6. Actual computational consumption (hours under IPC 50) in data synthesis with image optimization-based methods on a single NVIDIA 4090 GPU. A total 4K iterations are used for all methods and datasets to ensure fair comparisons. “RI” represents round iterations.

<table><tr><td></td><td>SRe2L [37]</td><td>RDED [29]</td><td>Ours</td></tr><tr><td>IPC 10</td><td>12.5</td><td>13.2</td><td> $\mathbf { 1 7 . 9 } _ { ( + 4 . 7 ) }$ </td></tr><tr><td>IPC 50</td><td>31.7</td><td>42.8</td><td> ${ \pmb 4 4 . 8 } _ { ( + 2 . 0 ) }$ </td></tr></table>

Table 7. Accuracy of data-free network pruning using slimming [21] on VGG11-BN [27].

![](images/493f0ef6d97550255051bc0c6660b51542896e7b08266624119614c97c9577e6.jpg)  
Figure 8. Continual learning results.  
DELT distilled dataset significantly outperforms G-VBSM, with an average improvement of about 10% in 100-step class-incremental learning task. This highlights the signifi-

## 5. Conclusion

We have introduced a new training strategy, EarlyLate, to improve image diversity in batch-to-global matching scenarios for dataset distillation. The proposed approach organizes predefined IPC samples into smaller, manageable subtasks and utilizes local optimizations. This strategy helps in refining each subset into distributions characteristic of different phases, thereby mitigating the homogeneity typically caused by a singular optimization process. The images refined through this method exhibit robust generalization across the entire task. We have extensively evaluated this approach on CIFAR-10, Tiny-ImageNet, ImageNet-1K, and its variants. Our empirical findings indicate that our approach significantly outperforms prior state-of-the-art methods across various IPC configurations.

## Acknowledgments

This research is supported by the MBZUAI-WIS Joint Program for AI Research and the Google Research award grant.

## References

[1] Yingbin Bai, Erkun Yang, Bo Han, Yanhua Yang, Jiatong Li, Yinian Mao, Gang Niu, and Tongliang Liu. Understanding and improving early stopping for learning with noisy labels. Advances in Neural Information Processing Systems, 34:24392–24403, 2021. 4

[2] Alexey Bochkovskiy, Chien-Yao Wang, and Hong-Yuan Mark Liao. Yolov4: Optimal speed and accuracy of object detection. arXiv preprint arXiv:2004.10934, 2020. 5

[3] George Cazenavette, Tongzhou Wang, Antonio Torralba, Alexei A Efros, and Jun-Yan Zhu. Dataset distillation by matching training trajectories. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4750–4759, 2022. 1, 2, 5, 6, 7

[4] Xuxi Chen, Yu Yang, Zhangyang Wang, and Baharan Mirzasoleiman. Data distillation can be like vodka: Distilling more times for better quality. In The Twelfth International Conference on Learning Representations, 2024. 3

[5] Justin Cui, Ruochen Wang, Si Si, and Cho-Jui Hsieh. Scaling up dataset distillation to imagenet-1k with constant memory. In International Conference on Machine Learning, pages 6565–6590. PMLR, 2023. 1, 2, 5, 6

[6] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009. 5

[7] Jiawei Du, Xin Zhang, Juncheng Hu, Wenxin Huang, and Joey Tianyi Zhou. Diversity-driven synthesis: Enhancing dataset distillation through directed weight adjustment. In Advances in neural information processing systems, 2024. 3

[8] Fastai. Fastai/imagenette: A smaller subset of 10 easily classified classes from imagenet, and a little more french. 5

[9] Spyros Gidaris and Nikos Komodakis. Dynamic few-shot visual learning without forgetting. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4367–4375, 2018. 5

[10] Jianyang Gu, Saeed Vahidian, Vyacheslav Kungurtsev, Haonan Wang, Wei Jiang, Yang You, and Yiran Chen. Efficient dataset distillation via minimax diffusion. In CVPR, 2024. 6, 7

[11] Ziyao Guo, Kai Wang, George Cazenavette, Hui Li, Kaipeng Zhang, and Yang You. Towards lossless dataset distillation via difficulty-aligned trajectory matching. In The Twelfth In ternational Conference on Learning Representations, 2024. 5, 6

[12] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 770–778, 2016. 5

[13] Yang He, Lingao Xiao, Joey Tianyi Zhou, and Ivor Tsang. Multisize dataset condensation. ICLR, 2024. 1, 2, 3

[14] Jang-Hyun Kim, Jinuk Kim, Seong Joon Oh, Sangdoo Yun, Hwanjun Song, Joonhyun Jeong, Jung-Woo Ha, and Hyun Oh Song. Dataset condensation via efficient syntheticdata parameterization. In Proceedings of the 39th Interna tional Conference on Machine Learning, 2022. 7

[15] Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009. 5

[16] Ya Le and Xuan Yang. Tiny imagenet visual recognition challenge. CS 231N, 7(7):3, 2015. 5

[17] Saehyung Lee, Sanghyuk Chun, Sangwon Jung, Sangdoo Yun, and Sungroh Yoon. Dataset condensation with contrastive signals. In International Conference on Machine Learning, pages 12352–12364. PMLR, 2022. 1, 2

[18] Haoyang Liu, Tiancheng Xing, Luwei Li, Vibhu Dalal, Jingrui He, and Haohan Wang. Dataset distillation via the wasserstein metric. arXiv preprint arXiv:2311.18531, 2023. 1, 3

[19] Risheng Liu, Jiaxin Gao, Jin Zhang, Deyu Meng, and Zhouchen Lin. Investigating bi-level optimization for learning and vision from a unified perspective: A survey and beyond. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(12):10045–10067, 2021. 3

[20] Songhua Liu, Kai Wang, Xingyi Yang, Jingwen Ye, and Xinchao Wang. Dataset distillation via factorization. Advances in Neural Information Processing Systems, 35:1100–1113, 2022. 1, 2

[21] Zhuang Liu, Jianguo Li, Zhiqiang Shen, Gao Huang, Shoumeng Yan, and Changshui Zhang. Learning efficient convolutional networks through network slimming. In Proceedings of the IEEE international conference on computer vision, pages 2736–2744, 2017. 8

[22] Lutz Prechelt. Early stopping-but when? In Neural Net works: Tricks of the trade, pages 55–69. Springer, 2002. 4

[23] Tian Qin, Zhiwei Deng, and David Alvarez-Melis. Distribu tional dataset distillation with subtask decomposition. arXiv preprint arXiv:2403.00999, 2024. 3

[24] Ilija Radosavovic, Raj Prateek Kosaraju, Ross Girshick, Kaiming He, and Piotr Dollar. Designing network design´ spaces. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10428–10436, 2020. 5

[25] Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, and Liang-Chieh Chen. Mobilenetv2: Inverted residuals and linear bottlenecks. In Proceedings of the IEEE conference on computer vision and pattern recogni tion, pages 4510–4520, 2018. 5

[26] Shitong Shao, Zeyuan Yin, Muxin Zhou, Xindong Zhang, and Zhiqiang Shen. Generalized large-scale data condensation via various backbone and statistical matching. In CVPR, 2024. 1, 2, 3, 7, 8

[27] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556, 2014. 8

[28] Suraj Srinivas and R Venkatesh Babu. Data-free parameter pruning for deep neural networks. arXiv preprint arXiv:1507.06149, 2015. 7

[29] Peng Sun, Bei Shi, Daiwei Yu, and Tao Lin. On the diversity and realism of distilled dataset: An efficient dataset distillation paradigm. In CVPR, 2024. 2, 4, 5, 6, 7, 8

[30] Mingxing Tan and Quoc Le. Efficientnet: Rethinking model scaling for convolutional neural networks. In International conference on machine learning, pages 6105–6114. PMLR, 2019. 5

[31] Mingxing Tan, Bo Chen, Ruoming Pang, Vijay Vasudevan, Mark Sandler, Andrew Howard, and Quoc V Le. Mnasnet: Platform-aware neural architecture search for mobile. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2820–2828, 2019. 5

[32] Yonglong Tian, Dilip Krishnan, and Phillip Isola. Contrastive multiview coding. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XI 16, pages 776–794. Springer, 2020. 5

[33] Tongzhou Wang, Jun-Yan Zhu, Antonio Torralba, and Alexei A Efros. Dataset distillation. arXiv preprint arXiv:1811.10959, 2018. 1, 2

[34] Zhou Wang and Alan C Bovik. Mean squared error: Love it or leave it? a new look at signal fidelity measures. IEEE signal processing magazine, 26(1):98–117, 2009. 3

[35] Lingao Xiao and Yang He. Are large-scale soft labels necessary for large-scale dataset distillation? In Advances in neural information processing systems, 2024. 3

[36] Zeyuan Yin and Zhiqiang Shen. Dataset distillation via curriculum data synthesis in large data era. Transactions on Machine Learning Research. 1, 3, 5, 6, 7, 8

[37] Zeyuan Yin, Eric Xing, and Zhiqiang Shen. Squeeze, recover and relabel: Dataset condensation at imagenet scale from a new perspective. In NeurIPS, 2023. 1, 3, 4, 5, 6, 7, 8

[38] Yihua Zhang, Prashant Khanduri, Ioannis Tsaknakis, Yuguang Yao, Mingyi Hong, and Sijia Liu. An introduction to bi-level optimization: Foundations and applications in signal processing and machine learning. arXiv preprint arXiv:2308.00788, 2023. 3

[39] Bo Zhao and Hakan Bilen. Dataset condensation with distribution matching. In IEEE/CVF Winter Conference on Applications of Computer Vision, WACV 2023, Waikoloa, HI, USA, January 2-7, 2023, 2023. 1, 2, 7

[40] Bo Zhao, Konda Reddy Mopuri, and Hakan Bilen. Dataset condensation with gradient matching. arXiv preprint arXiv:2006.05929, 2020. 1, 2, 5

[41] Borui Zhao, Quan Cui, Renjie Song, Yiyu Qiu, and Jiajun Liang. Decoupled knowledge distillation. In Proceedings of the IEEE/CVF Conference on computer vision and pattern recognition, pages 11953–11962, 2022. 6

[42] Ganlong Zhao, Guanbin Li, Yipeng Qin, and Yizhou Yu. Improved distribution matching for dataset condensation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7856–7865, 2023. 5, 6

[43] Muxin Zhou, Zeyuan Yin, Shitong Shao, and Zhiqiang Shen. Self-supervised dataset distillation: A good compression is all you need. arXiv preprint arXiv:2404.07976, 2024. 7

[44] Yongchao Zhou, Ehsan Nezhadarya, and Jimmy Ba. Dataset distillation using neural feature regression. Advances in Neu ral Information Processing Systems, 35:9813–9827, 2022. 1, 2