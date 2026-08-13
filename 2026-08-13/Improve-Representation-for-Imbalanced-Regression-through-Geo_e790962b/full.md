# Improve Representation for Imbalanced Regression through Geometric Constraints

Zijian Dong<sup>1,\*</sup> Yilei Wu<sup>1,\*</sup> Chongyao Chen<sup>2,\*</sup> Yingtian Zou<sup>1</sup> Yichi Zhang<sup>1</sup> Juan Helen Zhou<sup>1,†</sup>

<sup>1</sup>National University of Singapore, Singapore <sup>2</sup> Duke University, US

{zijian.dong, yilei.wu}@u.nus.edu, helen.zhou@nus.edu.sg

## Abstract

In representation learning, uniformity refers to the uniformfeature distribution in the latent space (i.e., unit hypersphere). Previous work has shown that improving uniformity contributes to the learning of under-represented classes. However, most of the previous work focused on classification; the representation space of imbalanced regression remains unexplored. Classificationbased methods are not suitable for regression tasks because they cluster features into distinct groups without considering the continuous and ordered nature essentialfor regression. In a geometric aspect, we uniquelyfocus on ensuring uniformity in the latent spacefor imbalanced regression through two key losses: enveloping and homogeneity. The enveloping loss encourages the induced trace to uniformly occupy the surface of a hypersphere, while the homogeneity loss ensures smoothness, with representations evenly spaced at consistent intervals. Our method integrates these geometric principles into the data representations via a Surrogate-driven Representation Learning (SRL)framework. Experiments with real-world regression and operator learning tasks highlight the importance ofuniformity in imbalanced regression and validate the efficacy of our geometrybased lossfunctions. Code is available here.

## 1. Introduction

Imbalanced datasets are ubiquitous across various domains, including image recognition [30], semantic segmentation [31], and regression [25]. Previous studies have demonstrated the significance of uniform or balanced distribution of class representations for effective imbalanced classification [5, 8, 9, 12, 22, 26, 33] and imbalanced semantic segmentation [31]. In classification tasks, these representations typically form distinct clusters. However, in the context of regression, representations are expected to be continuous and ordered [24, 27], rendering the methods used for quantifying and analyzing uniformity in classification inapplicable. While the issue of deep imbalanced regression (DIR) has received considerable attention, the focus has predominantly been on training unbiased regressors, rather than on the aspect of representation learning [6, 10, 16, 19, 25]. Among the methods that do explore representation learning [24, 27], the emphasis is typically on understanding the relationship between the label space and the feature space (the representations themselves should be continuous and ordered). However, a critical aspect that remains under-explored is the interaction between data representations and the entire feature space. Specifically, how these representations distribute within the full scope of the feature space has not been examined.

![](images/1d715d01a7368a374d77b9ec47c6d3cb067257073f126e74b9c64702e3128894.jpg)  
Figure 1. 2D feature space of vanilla baseline and ours from UCI-Airfoil [2]. The vanilla feature space lacks uniformity and is dominated by samples from the Many-shot region. In contrast, our approach achieves a more uniform distribution over the feature space, improving the performance, especially in the Medium and Few-shot regions. (For visualization purposes, we curated the dataset to ensure equal partitions across the three regions.)

Uniformity in classification refers to how effectively different clusters or centroids occupy the feature space, essentially partitioning it among various classes. In regression, here we define the term “latent trace” as the pathway that the representations follow, delineating the transition from the minimum to the maximum label values. In this paper, we aim to evaluate how well a latent trace occupies the feature space? To quantify this, we approximate a tubular neighborhood around the latent trace and measure its volume relative to the entire feature space. This method gauges the effectiveness of the trace in “enveloping” the hypersphere, and we call it enveloping loss. This loss ensures that the trace shape fills the surface of the hypersphere to facilitate uniformity. In parallel, it is equally important that the points (i.e., individual data point representations) are evenly distributed along the trace. To address uniform distribution along the trace as well as smoothness, we have developed a homogeneity loss. This loss is computed based on the arc length of the trace, allowing us to effectively measure and promote an even and smooth distribution of points.

![](images/7d9fdab4f6655bcab3dc6805e4a44e7ed5efede9d79a935ef8da9971157c8e68.jpg)  
Figure 2. t-SNE visualization [20] of feature comparison. The first row corresponds to the original UCI Airfoil Dataset [2], while the second row corresponds to its curated version, with an additional few-shot region in the middle of the label range. Colored arrows point to the few-shot regions and their corresponding positions in the feature distributions. We evaluate feature distributions using: MSE Loss (Baseline), SRL without uniformity loss (w/o ${ \mathcal { L } } _ { \mathrm { e n v } } ) .$ , SRL without homogeneity loss (w/o ${ \mathcal { L } } _ { \mathrm { h o m o } } )$ , and complete SRL (ours). The baseline leads to feature collapse to many-shot regions and inadequate distinction of few-shot samples. In w/o ${ \mathcal { L } } _ { \mathrm { e n v } } ,$ features collapse into a trivial shape, not fully utilizing the feature space. In w/o $\mathcal { L } _ { \mathrm { h o m o } } .$ , features spread out along the trace. Different from the previous ones, our SRL uniformly and smoothly “fills” the feature space.

We model the uniformity in regression in two aspects: the induced trace aims to fully occupy the surface of the hypersphere (Enveloping), exhibiting smoothness with representations spaced at uniform intervals (Homogeneity). The two losses we introduce act as geometric constraints on a latent trace, implying they should not be applied to a set of representations from a single mini-batch. This is because a single batch likely does not encompass the full range of labels. To address this, we have developed a Surrogate-driven Representation Learning (SRL) scheme. It involves averaging representations of the same bins within a mini-batch to form centroids and “re-filling” missing bins by taking corresponding centroids from the previous epoch. This process results in a surrogate containing centroids for all bins, enabling the effective application of geometric loss across the complete label range. Furthermore, we introduce Imbalanced Operator Learning (IOL) as a new DIR benchmark for training models on imbalanced domain locations in function space mapping. In summary, our main contributions are four-fold:

• Geometric Losses for Uniformity in deep imbalanced regression (DIR). To the best of our knowledge, this work is the first to study representation learning in DIR. We introduce two novel loss functions, enveloping loss and homogeneity loss, to ensure uniform feature distribution for DIR.

• SRL Framework. A new framework is proposed that incorporates these geometric principles into data representations.

• Imbalanced Operator Learning (IOL). For the first time, we pioneer the task of operator learning within the realm of deep imbalanced regression, introducing an innovative task: Imbalanced Operator Learning (IOL).

• Extensive Experiments. The effectiveness of the proposed method is validated through experiments involving real-world regression and operator learning, on five datasets: AgeDB-DIR, IMDB-WIKI-DIR, STS-B-DIR, and two newly created DIR benchmarks, UCI-DIR and OL-DIR.

## 2. Related Work

Uniformity in imbalanced classification. Wang and Isola [22] identifies that uniformity is one of the key properties in contrastive representation learning. To promote uniformity in representation space for imbalanced classification, a variety of training strategies have been proposed. Kang et al. [8] decouples the training into a two-stage training of representation learning and classification. Yin et al. [26] designs a transfer learning framework for imbalanced face recognition. Kang et al. [9] combines supervised method and contrastive learning to learn a discriminative and balanced feature space. PaCo [5] and TSC [12] learn a set of class-wise balanced centers. BCL [33] balances the gradient distribution of negative classes and data distribution in mini-batch. Recent study suggests that sample-level uniform distribution may not effectively address imbalanced classification, advocating for category-level uniformity instead [1, 32]. Though progress has been made in this field, challenges persist in adapting the approach of modeling uniformity from classification to regression.

Deep imbalanced regression. With imbalanced regression data, effective learning in a continuous label space focuses on modeling the relationships among labels in the feature space [25]. Label Distribution Smoothing (LDS) [25] and DenseLoss [19] apply a Gaussian kernel to the observed label density, leading to an estimated label density distribution. Feature distribution smoothing (FDS) [25] generalizes the application of kernel smoothing from label to feature space. Ranksim [6] aims to leverage both local and global dependencies in data by aligning the order of similarities between labels and features. Balanced MSE [16] addresses the issue of imbalance in Mean Squared Error (MSE) calculations, ensuring a more balanced distribution of predictions. VIR [23] provides uncertainty for imbalanced regression. ConR [10] regularizes contrastive learning in regression by modeling global and local label similarities in feature space. RNC [27] and SupReMix [24] learn a continuous and ordered representation for regression through supervised contrastive learning. How imbalanced regression representations leverage the feature space remains under-explored.

## 3. Method

In the field of representation learning for classification, the concept of uniformity is pivotal for maximizing the use of the feature space [5, 8, 9, 12, 22, 26, 33]. This idea is based on the principle of ensuring that features from different classes are not only distinctly separated but also evenly distributed in the latent space. This uniform distribution of class centroids fosters a clear and effective decision boundary, leading to more accurate classification. However, in regression, where we deal with continuous, ordered trace [24, 27] rather than discrete clusters, the concept ofuniformity is not only more complex but essentially remains undefined.

We draw an analogy to the process of winding yarn around a ball. In this analogy, the yarn represents the latent trace, and the ball symbolizes the entirety of the available feature space. Just as the yarn must be evenly distributed across the ball’s surface to effectively cover it (without any crossing), the latent trace should strive to occupy the hypersphere of the latent space uniformly. This ensures that the model leverages the available feature space to its fullest extent, enhancing the model’s ability to capture the variability inherent in the data.

![](images/bad0bd3fe64984aa2a83b6dd29872215ee47bf193c7045360be65400ddfc2383.jpg)  
Figure 3. 2D schematic overview of two geometric losses. The arrow indicates the improvement of the loss function. Enveloping loss encourages the representations to fill the latent space, and homogeneity loss encourages the smoothness and even distribution of the representations along the trace.

Furthermore, the latent trace should be smooth and continuous, akin to the even stretching of yarn, rather than loose and disjointed. This smoothness ensures a consistent and predictable model behavior, which is crucial for the accurate prediction and interpretation of results.

We outline our method in this section. Firstly, we establish the fundamental notations and preliminaries (Section 3.1). Following this, we delve into the concept and definition of our enveloping loss (Section 3.2) and homogeneity loss (Section 3.3). Finally, we present our Surrogate-driven Representation Learning (SRL) framework, which incorporates the geometric constraints from the global image of the representations into the local range (Section 3.4). Refer to Supplementary Material 9 for the pseudo code of our method.

## 3.1. Preliminaries

A regression dataset is composed of pairs $\left( \mathbf { x } _ { i } , y _ { i } \right)$ , where $\mathbf { x } _ { i }$ represents the input and $y _ { i }$ is the corresponding continuous-value target. Denote $\mathbf { z } _ { i } = f ( \mathbf { x } _ { i } )$ as the feature representation of $\mathbf { x } _ { i } ,$ generated by a neural network $f ( \cdot )$ . The feature representation is normalized so that $\| \mathbf { z } _ { i } \| = 1$ for all i. Suppose the dataset consists of K unique bins <sup>1</sup>, we define a surrogate as a set of centroids $\mathbf { c } _ { k } .$ , where each represents a distinct bin. These centroids are computed by averaging the representations z sharing the same bin and they are normalized to $\| \mathbf { c } _ { k } \| = 1 .$ . Let l be a path: l : $[ y _ { \operatorname* { m i n } } , y _ { \operatorname* { m a x } } ] \mapsto \mathbb { R } ^ { n }$ with $\| l ( y ) \| = 1$ , such that $l ( y _ { k } ) { = } \mathbf { c } _ { k }$ . The path l is a continuous curve extended from the discrete dataset that lies on a submanifold of R<sup>n</sup>.

## 3.2. Enveloping

To maximize the use of the feature space, it is crucial for the ordered and continuous trace of regression representations to fill the entire unit hypersphere as much as possible. This is by analogy with wrapping yarn with a certain length around a ball (without any crossing), aiming to cover as much surface area as possible.

The trace of regression representations lying on a submanifold of $\mathbb { R } ^ { n }$ has a negligible hypervolume, which makes it challenging to assess its relationship with the entire hypersphere. To address this challenge, we extend the $\hbar \mathrm { { i n e } ^ { \prime \prime } }$ into a tubular neighborhood. This expansion allows us to introduce the concept of enveloping loss. Our objective with this loss function is to maximize the hypervolume of the tubular neighborhood in proportion to the total hypervolume of the hypersphere.

Denote the set of all unit vectors in $\mathbb { R } ^ { n }$ as U. Given $\epsilon \in ( 0 \mathrm { , 1 } )$ define tubular neighborhood $T ( l , \epsilon )$ of l as:

$$
T ( l , \epsilon ) = \{ { \bf z } \in \mathcal { U } \mid { \bf t } \cdot { \bf z } > \epsilon \mathrm { f o r s o m e } { \bf t } \in \mathrm { I m } ( l ) \}\tag{1}
$$

where for a function $f : A  B$ , the image is defined as Im $( f ) : = \{ f ( x ) , x \in A \}$

Then our enveloping loss is defined as:

$$
\mathcal { L } _ { \mathrm { e n v } } = - \frac { \mathrm { v o l } ( T ( l , \epsilon ) ) } { \mathrm { v o l } ( \mathcal { U } ) }\tag{2}
$$

where vol(·) returns the hypervolume of its input in the induced measure from the Euclidean space.

In practical scenarios, the trace is composed of discrete representations, which complicates the direct computation of the tubular neighborhood’s hypervolume. To navigate this challenge, we propose a continuous-to-discrete strategy. We first generate N points that are uniformly distributed across the hypersphere. We then determine the fraction of these points that fall within the neighbourhood ϵ. This fraction effectively approximates the proportion of the hypersphere covered by the tubular neighborhood with a sufficiently large N. To adapt $\mathcal { L } _ { \mathrm { e n v } }$ to discrete datasets, we re-formalize our optimization objective as:

$$
\operatorname* { m a x } \operatorname* { l i m } _ { N \to \infty } { \frac { P ( N ) } { N } }\tag{3}
$$

where

$$
P ( N ) { : = } | \{ { \bf p } _ { i } | \operatorname* { m a x } _ { y } \{ { \bf p } _ { i } \cdot l ( y ) \} > \epsilon , i \in [ N ] \} |\tag{4}
$$

assuming for each $N > 0 .$ , we can choose N evenly distributed points in U, and denote these points as $\mathbf { p } _ { i } , i \in [ N ] = 1 , . . . , N$ . For numerical application, we take $N$ to be a sufficiently large number and use the standard Monte-Carlo method [17] to approximate the evenly distributed points.

In our implementation, we did not directly define ϵ due to the non-differentiability of the binarization required to determine if $\mathbf { p } _ { i }$ is within the ϵ-tube. Instead, for each $\mathbf { p } _ { i }$ , we maximize the cosine similarity between $\mathbf { p } _ { i }$ and its closest point on the trace. In this way, we relax the step function represented by (4) to its $\mathrm { \Delta ^ { 6 6 } s o f t { 7 } }$ version, leading to smooth gradient computation.

## 3.3. Homogeneity

While the enveloping loss effectively governs the overall distribution of representations on the hypersphere, it alone may not be entirely adequate, presenting two unresolved issues. 1) The first is distribution along the trace. The enveloping loss predominantly controls the overall shape of representations on the hypersphere, yet it does not guarantee a uniform distribution along the trace. This poses a notable concern, as it may result in uneven representation density across different trace segments. 2) The second is trace smoothness. The enveloping loss could lead to a zigzag pattern of the representations, which should be avoided. Considering age estimation from facial images as an example, the progression of facial features over time is gradual. Consequently, in the corresponding latent space, we would anticipate a similar, smooth transition without abrupt changes, underlining the desirability of a smoother trace. Interestingly, these two issues can be aptly analogized to winding yarns around a ball as well. For the yarn on the ball to be smooth, it should be tightly stretched, rather than being disjointed or loosely arranged. We name the property of a trace to be smooth with representations evenly distributed along it as homogeneity.

We encourage such homogeneity property, i.e., smoothness of the trace Im(l) and uniform distribution of representations along it, by penalizing the arc length. Formally, the homogeneity loss is defined as:

$$
\mathcal { L } _ { \mathrm { h o m o } } = \int _ { y _ { \mathrm { m i n } } } ^ { y _ { \mathrm { m a x } } } \left. \left. \frac { \mathrm { d } l ( y ) } { \mathrm { d } y } \right. \right. ^ { 2 } \mathrm { d } y\tag{5}
$$

Given K different $y \mathrm { s }$ which have been ordered, the discrete format for $\mathcal { L } _ { \mathrm { { h o m o } } }$ is defined as a summation of the squared differences between adjacent points:

$$
\mathcal { L } _ { \mathrm { h o m o } } = \sum _ { k = 1 } ^ { K - 1 } \frac { \left. l \left( y _ { k + 1 } \right) - l \left( y _ { k } \right) \right. ^ { 2 } } { y _ { k + 1 } - y _ { k } }\tag{6}
$$

The use of only homogeneity loss might result in trivial solutions like representation convergence to a circle or point due to feature collapse (shown in Figure 2). The homogeneity loss should be treated as a regularization of the enveloping loss, promoting not only smoothness but also an even distribution of representations along the trace. To quantitatively define the relationship between trace arc length and these desired characteristics, we introduce Theorem 1. It demonstrates that with a given Im(l) $( l _ { \mathrm { e n v } }$ is fixed as it does not depend on the parameterization of $l ) .$ , the homogeneity loss is minimized if and only if when representations are uniformly distributed along the trace.

Theorem 1. Given an image of l, $\mathcal { L } _ { h o m o }$ attains its minimum if and only ifthe representations are uniformly distributed along the trace, i.e., $\| \nabla _ { y } l ( y ) \| = c ,$ where c is a constant.

Refer to Supplementary Material 6 for the proof.

Therefore, we formulate our geometric constraints $( \mathcal { L } _ { \mathrm { G } } )$ as a combination of enveloping and homogeneity:

![](images/aa1f33a8bbb5b5452d3eb32855acc7fbcac766208e932161dc539042f7294c38.jpg)  
Figure 4. Overview of Surrogate-driven Representation Learning (SRL). (1) Every mini-batch is encoded to the latent space. Some bins may not be present in the current batch. To address this, (2) it takes centroids corresponding to the missing bins from the previous epoch. These stored centroids are used to “re-fill” the missing bins in the current batch. (3) Average the representations for bins that appear multiple times, creating centroids for these bins. This surrogate, containing a representation for the full label range, allows for the effective application of geometric loss across all bins. (4) Loss calculation based on the surrogate. (5) Update the surrogate in memory to ensure enveloping and homogeneity. The training of the first epoch is driven by MSE loss only.

$$
\mathcal { L } _ { \mathrm { G } } = \lambda _ { e } \mathcal { L } _ { \mathrm { e n v } } + \lambda _ { h } \mathcal { L } _ { \mathrm { h o m o } }\tag{7}
$$

where $\lambda _ { e }$ and $\lambda _ { h }$ are weights for the two geometric losses. In Section $4 . 4$ , we further explore the behavior of these two geometric constraints, uncovering new insights into imbalanced regression.

## 3.4. Surrogate-driven Representation Learning (SRL)

Our geometric loss $( \mathcal { L } _ { \mathrm { G } } )$ is calculated on a surrogate instead of a mini-batch (Figure 4), as the representations from one mini-batch very likely fail to capture the global image of l, due to the randomness of batch sampling.

For illustration purposes, here we assume the original dataset has already been binned, as is the case in most DIR datasets [6, 10, 25]. Let $\mathcal { Z } = \{ \mathbf { z } _ { 1 } , \mathbf { z } _ { 2 } , . . . , \mathbf { z } _ { M } \}$ be a set of representations from a batch with batch-size M, and let $\mathcal { V } = \{ y _ { 1 } , y _ { 2 } , . . . , y _ { M } \}$ (repetitions of the label values might exist) be the corresponding labels. Define the centroid $\mathbf { c } _ { y } \in \mathcal { C }$ for $y \left( \left\| \mathbf { c } _ { y } \right\| = 1 \right)$ as:

$$
\mathbf { c } _ { y } = \frac { 1 } { | \{ \mathbf { z } _ { m } | y _ { m } = y \} | } \sum _ { \{ \mathbf { z } _ { m } | y _ { m } = y \} } \mathbf { z } _ { m }\tag{8}
$$

Suppose the whole label range is covered by a set of unique K bins $\mathcal { V } ^ { * } = \{ y _ { 1 } ^ { * } , y _ { 2 } ^ { * } , . . . , y _ { K } ^ { * } \}$ . The centroids for these K bins from the last epoch are denoted as $\mathcal { C } ^ { \prime } = \{ \mathbf { c } ^ { \prime } { } _ { y _ { 1 } ^ { * } } , \mathbf { c } ^ { \prime } { } _ { y _ { 2 } ^ { * } } , . . . , \mathbf { c } ^ { \prime } { } _ { y _ { K } ^ { * } } \}$

The surrogate $s$ is then generated by re-filling the missing centroids:

$$
S { = } \mathcal { C } \cup \{ \mathbf { c } ^ { \prime } { } _ { y _ { k } ^ { * } } | y _ { k } ^ { * } \in \mathcal { V } ^ { * } \backslash \mathcal { V } \}\tag{9}
$$

We use AdamW [13] with momentum to update parameters θ in $f ( \cdot )$ , to ensure a smooth transition of the local shape in the batch-wise representations.

At the end of each epoch $e \in ( 0 , E ]$ (excluding the first), we use the representations learned during that epoch to form a running surrogate $\hat { \boldsymbol S } ^ { e } , \boldsymbol S ^ { e + 1 }$ is formulated from the current epoch’s surrogate $S ^ { e }$ and $\hat { \boldsymbol S } ^ { e }$ with momentum. $S ^ { e + 1 }$ is employed for training in the subsequent epoch. It facilitates a gradual transition between epochs, preventing abrupt variations: $S ^ { e + 1 } \gets \alpha \cdot S ^ { e } + ( 1 - \alpha ) \cdot \hat { S } ^ { e }$

We aim for individual representations from the encoder to converge towards their respective centroids that share the same label, while simultaneously distancing them from centroids associated with different labels. To achieve this, we incorporate a contrastive loss between the individual representations and the centroids. For each representation $\mathbf { z } _ { m }$ with $y _ { m } = y $ , the centroid $\mathbf { c } _ { y }$ is considered as the positive, and the other centroids as negatives. The contrastive loss is defined as:

$$
\mathcal { L } _ { \mathrm { c o n } } = - \sum _ { m = 1 } ^ { M } \log \frac { \exp ( \sin ( \mathbf { z } _ { m } , \mathbf { c } _ { y } ) ) } { \sum _ { y ^ { * } \in \mathcal { V } ^ { * } } \exp ( \sin ( \mathbf { z } _ { m } , \mathbf { c } _ { y ^ { * } } ) ) }\tag{10}
$$

where sim(·) is the cosine similarity between two input.

The framework is trained end-to-end, the total loss used to update the parameters θ in $f ( \cdot )$ is defined as:

$$
\mathcal { L } _ { \theta } = \mathcal { L } _ { \mathrm { r e g } } + \mathcal { L } _ { \mathrm { G } } + \mathcal { L } _ { \mathrm { c o n } }\tag{11}
$$

where $\mathcal { L } _ { \mathrm { r e g } }$ is the mean squared error (MSE) loss.

## 4. Experiments

We perform extensive experiments to validate and analyze the effectiveness of SRL for deep imbalanced regression. Our regression tasks span age estimation from facial images, tabula regression, and text similarity score regression, as well as our newly established task: Imbalanced Operator Learning (IOL). This section begins by detailing the experiment setup (Section 4.1) followed by the main results (Section 4.2). The results of IOL are shown in Section 4.3 followed by the comparison with classification-based methods and hyperparameters analysis (Section 4.4).

## 4.1. Experiment Setup

Datasets. We employ three real-world regression datasets developed by Yang et al. [25], and our curated UCI-DIR from UCI Machine Learning Repository [2], to assess the effectiveness of SRL in deep imbalanced regression. Refer to Supplementary Material 7 for more dataset details.

Table 1. Results on UCI-DIR (MAE). We report the average MAE of three runs. The best results are in bold.
<table><tr><td>Datasets</td><td colspan="4">Airfoil</td><td colspan="4">Abalone</td><td colspan="4">Real Estate</td><td colspan="4">Concrete</td></tr><tr><td>Shot</td><td>All</td><td>Many</td><td>Med</td><td>Few</td><td>All</td><td>Many</td><td>Med</td><td>Few</td><td>All</td><td>Many</td><td>Med</td><td>Few</td><td>All</td><td>Many</td><td>Med</td><td>Few</td></tr><tr><td>VANILLA</td><td>5.66</td><td>5.11</td><td>5.03</td><td>6.75</td><td>4.57</td><td>0.88</td><td>2.65</td><td>7.97</td><td>0.33</td><td>0.27</td><td>0.38</td><td>0.37</td><td>7.29</td><td>5.77</td><td>6.92</td><td>9.74</td></tr><tr><td>LDS + FDS [25]</td><td>5.76</td><td>4.45</td><td>4.79</td><td>7.79</td><td>5.09</td><td>0.90</td><td>3.26</td><td>9.26</td><td>0.35</td><td>0.33</td><td>0.40</td><td>0.34</td><td>6.88</td><td>6.21</td><td>6.73</td><td>7.59</td></tr><tr><td>RankSim [6]</td><td>5.23</td><td>5.05</td><td>4.91</td><td>5.72</td><td>4.33</td><td>0.98</td><td>2.59</td><td>7.42</td><td>0.37</td><td>0.34</td><td>0.38</td><td>0.40</td><td>6.71</td><td>6.00</td><td>5.57</td><td>9.46</td></tr><tr><td>BalancedMSE [16]</td><td>5.69</td><td>4.51</td><td>5.04</td><td>7.28</td><td>5.37</td><td>2.14</td><td>2.66</td><td>9.37</td><td>0.34</td><td>0.31</td><td>0.40</td><td>0.33</td><td>7.03</td><td>4.67</td><td>6.37</td><td>9.72</td></tr><tr><td>Ordinal Entropy [29]</td><td>6.27</td><td>4.85</td><td>5.37</td><td>8.32</td><td>6.77</td><td>2.31</td><td>4.01</td><td>11.61</td><td>0.34</td><td>0.29</td><td>0.42</td><td>0.35</td><td>7.12</td><td>5.50</td><td>6.36</td><td>9.31</td></tr><tr><td>SRL (ours)</td><td>5.10</td><td>4.83</td><td>4.75</td><td>5.69</td><td>4.16</td><td>0.89</td><td>2.42</td><td>7.19</td><td>0.28</td><td>0.26</td><td>0.30</td><td>0.29</td><td>5.94</td><td>5.32</td><td>5.80</td><td>6.60</td></tr></table>

Table 2. Results on AgeDB-DIR, the best are in bold.
<table><tr><td>Metrics</td><td colspan="4">MAE↓</td><td colspan="4">GM↓</td></tr><tr><td>Shot</td><td>|All</td><td>Many</td><td>Med</td><td>Few</td><td>All</td><td>Many</td><td>Med</td><td>Few</td></tr><tr><td>VANILLA</td><td>7.67</td><td>6.66</td><td>9.30</td><td>12.61</td><td>4.85</td><td>4.17</td><td>6.51</td><td>8.98</td></tr><tr><td>LDS + FDS [25]</td><td>7.55</td><td>7.03</td><td>8.46</td><td>10.52</td><td>4.86</td><td>4.57</td><td>5.38</td><td>6.75</td></tr><tr><td>RankSim [6]</td><td>7.41</td><td>6.49</td><td>8.73</td><td>12.47</td><td>4.71</td><td>4.15</td><td>5.74</td><td>8.92</td></tr><tr><td>BalancedMSE [16]</td><td>7.98</td><td>7.58</td><td>8.65</td><td>9.93</td><td>5.01</td><td>4.83</td><td>5.46</td><td>6.30</td></tr><tr><td>Ordinal Entropy [29]</td><td>7.60</td><td>6.69</td><td>8.87</td><td>12.68</td><td>4.91</td><td>4.28</td><td>6.20</td><td>9.29</td></tr><tr><td>ConR [10]</td><td>7.41</td><td>6.51</td><td>8.81</td><td>12.04</td><td>4.70</td><td>4.13</td><td>5.91</td><td>8.59</td></tr><tr><td>SRL (ours)</td><td>7.22</td><td>6.64</td><td>8.28</td><td>9.81</td><td>4.50</td><td>4.12</td><td>5.37</td><td>6.29</td></tr></table>

Table 3. Results on IMDB-WIKI-DIR, the best are in bold.
<table><tr><td>Metrics</td><td colspan="4">MAE↓</td><td colspan="4">GM↓</td></tr><tr><td>Shot</td><td>All</td><td>Many</td><td>Med</td><td>Few</td><td>All</td><td>Many</td><td>Med</td><td>Few</td></tr><tr><td>VANILLA</td><td>8.03</td><td>7.16</td><td>15.48</td><td>26.11</td><td>4.54</td><td>4.14</td><td>10.84</td><td>18.64</td></tr><tr><td>LDS + FDS [25]</td><td>7.73</td><td>7.22</td><td>12.98</td><td>23.71</td><td>4.40</td><td>4.17</td><td>7.87</td><td>15.77</td></tr><tr><td>RankSim [6]</td><td>7.72</td><td>6.92</td><td>14.52</td><td>25.89</td><td>4.29</td><td>3.92</td><td>9.72</td><td>18.02</td></tr><tr><td>BalancedMSE [16]</td><td>8.43</td><td>7.84</td><td>13.35</td><td>23.27</td><td>4.93</td><td>4.68</td><td>7.90</td><td>15.51</td></tr><tr><td>Ordinal Entropy [29]</td><td>8.01</td><td>7.17</td><td>15.15</td><td>26.48</td><td>4.47</td><td>4.07</td><td>10.56</td><td>21.11</td></tr><tr><td>ConR [10]</td><td>7.84</td><td>7.15</td><td>14.36</td><td>25.15</td><td>4.43</td><td>4.05</td><td>9.91</td><td>18.55</td></tr><tr><td>SRL (ours)</td><td>7.69</td><td>7.08</td><td>12.65</td><td>22.78</td><td>4.28</td><td>4.03</td><td>7.28</td><td>15.25</td></tr></table>

Table 4. Results on STS-B-DIR, the best are in bold.
<table><tr><td>Metrics</td><td colspan="4">MSE↓</td><td colspan="4">Pearson correlation ↑</td></tr><tr><td>Shot</td><td>All</td><td>Many</td><td>Med</td><td>Few</td><td>All</td><td>Many</td><td>Med</td><td>Few</td></tr><tr><td>VANILLA</td><td>0.993</td><td>0.963</td><td>1.000</td><td>1.075</td><td>0.742</td><td>0.685</td><td>0.693</td><td>0.793</td></tr><tr><td>LDS + FDS [25]</td><td>0.900</td><td>0.911</td><td>0.881</td><td>0.905</td><td>0.757</td><td>0.698</td><td>0.723</td><td>0.806</td></tr><tr><td>RankSim [6]</td><td>0.889</td><td>0.907</td><td>0.874</td><td>0.757</td><td>0.763</td><td>0.708</td><td>0.692</td><td>0.842</td></tr><tr><td>BalancedMSE [16]</td><td>0.909</td><td>0.894</td><td>1.004</td><td>0.809</td><td>0.757</td><td>0.703</td><td>0.685</td><td>0.831</td></tr><tr><td>Ordinal Entropy [29]</td><td>0.943</td><td>0.902</td><td>1.161</td><td>0.812</td><td>0.750</td><td>0.702</td><td>0.679</td><td>0.767</td></tr><tr><td>SRL (ours)</td><td>0.877</td><td>0.886</td><td>0.873</td><td>0.745</td><td>0.765</td><td>0.708</td><td>0.749</td><td>0.844</td></tr></table>

• AgeDB-DIR [25]: It serves as a benchmark for estimating age from facial images, which is derived from the AgeDB dataset [15]. It contains 12,208 images for training, 2,140 images for validation, and 2,140 images for testing.

• IMDB-WIKI-DIR [25]: It is a facial image dataset for age estimation derived from the IMDB-WIKI dataset [18], which consists of face images with the corresponding age. It has 191,509 images for training, 11,022 images for validation, and 11,022 for testing.

• STS-B-DIR [25]: It is a natural language dataset formulated from STS-B dataset [3, 21], consisting of 5,249 training sentence pairs, 1,000 validation pairs, and 1,000 testing pairs. Each sentence pair is labeled with the continuous similarity score.

• UCI-DIR: To evaluate the performance of SRL on tabular data, we curated UCI Machine Learning Repository [2] to formulate UCI-DIR that includes four regression tasks (Airfoil Self-Noise, Abalone, Concrete Compressive Strength, Real estate valuation). Following the DIR setting [25], we make each regression task consist of an imbalanced training set and a balanced validation and test set.

Metrics. In line with the established settings in DIR [25], subsets in an imbalanced training set are categorized based on the number of available training samples: many-shot region (bins with > 100 training samples), medium-shot region (bins with 20 to 100 training samples), and few-shot region (bins with < 20 training samples), for the three real-world datasets. For AgeDB-DIR and IMDB-WIKI-DIR, each bin represents 1 year. In the case of STS-B-DIR, bins are segmented by 0.1. For UCI-DIR, the bins are segmented by 0.1 to 1 depending on the range of regression targets. Our evaluation metrics include mean absolute error (MAE, the lower the better) and geometric mean (GM, the lower the better) for AgeDB-DIR, IMDB-WIKI-DIR and UCI-DIR. For STS-B-DIR, we use mean squared error (MSE, the lower the better) and Pearson correlation (the higher the better). Implementation Details. For age estimation (AgeDB-DIR and IMDB-WIKI-DIR), we follow the settings from Yang et al. [25], which uses ResNet-50 [7] as a backbone network. For text similarity regression (STS-B-DIR), we follow the setting from Cer et al. [3], Yang et al. [25] that uses BiLSTM + GloVe word embeddings. For tabular regression (UCI-DIR), we use an MLP with three hidden layers (d-20-30-10-1) following the setting from Cheng et al. [4]. For all baseline methods, results were produced following provided training recipes through publicly available codebase. All experimental results, including ours and baseline methods, were obtained from a server with 8 RTX 3090 GPUs.

Baselines. We consider both DIR methods [6, 10, 25] and recent techniques proposed for general regression [16, 28, 29], in addition to VANILLA regression (MSE loss). We compare the performance of SRL with all baselines on the above four datasets. Furthermore, as SRL is orthogonal to previous DIR methods, we examine the improvement of them by adding ou geometric losses.

![](images/50bdd32c5e889e92e6728bacd8981c07a4005ba0462fe951b5c36fc075e34fac.jpg)  
Figure 5. SRL performance gain compared to VANILLA across age ranges on AgeDB-DIR. The gray histogram in the background shows the distribution of samples across age groups. SRL substantially improves the performance on the medium-shot and few-shot regions (age < 20 and > 70).

Table 5. Combine SRL with existing DIR methods (MAE)
<table><tr><td>Datasets</td><td></td><td>AgeDB (MAE)</td><td></td><td></td><td>IMDB-WIKI (MAE)</td><td></td></tr><tr><td>Shot</td><td>All</td><td>Many Med</td><td>Few</td><td>All</td><td>Many</td><td>Med</td></tr><tr><td>SRL+LDS+FDS [25]</td><td>|7.32</td><td>6.81 8.14</td><td>9.81</td><td>|7.61</td><td>7.03</td><td>12.28 21.77</td></tr><tr><td>GAINS v.s. LDS+FDS (%) SRL+RankSim [6]</td><td>3.05 3.23</td><td>3.89</td><td>6.75</td><td>1.66</td><td>2.64</td><td>5.40 8.19</td></tr><tr><td>GAINS v.s. RankSim (%)</td><td>7.29 6.57 1.62 -1.23</td><td>8.58 1.72</td><td>10.48 16.96</td><td>7.67 0.65</td><td>7.08 -1.15</td><td>12.40 22.85 14.61 11.75</td></tr><tr><td>SRL+BalancedMSE [16]</td><td>6.77</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GAINS v.s. BalancedMSE (%)</td><td>7.24 9.27 10.69</td><td>7.86 9.14</td><td>9.85 0.89</td><td>7.74 8.19</td><td>7.13</td><td>12.77 22.04</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>9.06</td><td>4.35 5.29</td></tr><tr><td>SRL+ConR [10]</td><td>7.40 6.87</td><td>8.08</td><td>10.50</td><td>7.56</td><td>7.01</td><td>12.03 21.71</td></tr><tr><td>GAINS v.s. ConR (%)</td><td>0.14 -5.53</td><td>8.39</td><td>13.80</td><td>3.68</td><td>1.96</td><td>16.23 13.68</td></tr></table>

## 4.2. Main Results

To show the effectiveness of SRL on DIR, we first benchmark SRL and baselines for tabular regression on our curated UCI-DIR with four different regression tasks (Table 1). Moreover, we evaluate our method on established DIR benchmarks [25] including age estimation on AgeDB-DIR and IMDB-WIKI-DIR (Table 2 & 3, and Figure 5), and text similarity regression on STS-B-DIR (Table 4). We evaluate the combination of SRL and previous DIR methods on AgeDB-DIR and IMDB-WIKI-DIR (Table 5). Notably, Table 1 and 4 omit results from ConR [10], as it depends on data augmentation, a technique not fully established in the domain of tabular data and natural language. Combine SRL with existing methods. Our SRL approach enhances imbalanced regression by imposing geometric constraints on feature distributions, a strategy that is orthogonal to existing methods. To illustrate this, we leverage SRL as a regularizing term in conjunction with other methods. The results of this experiment are presented in Table 5. It shows that when SRL is integrated with existing regression methods, there is improvement in performance across different regions for both datasets. This demonstrates the effectiveness and compatibility of SRL as a complementary tool in the realm of regression analysis.

## 4.3. Imbalanced Operator Learning (IOL)

We introduce a novel task for DIR called Imbalanced Operator Learning (IOL). Traditional operator learning aims to train a neural network to model the mapping between function spaces [11, 14]. However, unlike the standard approach of uniformly sampling output locations, in IOL, we intentionally adjust the sampling density within the output function’s domain to create regions with few, medium, and many regions (Figure 6).

For the linear operator, the model is trained to estimate the integral operator denoted as G:

$$
G : u ( x ) \mapsto s ( x ) = \int _ { 0 } ^ { x } u ( \tau ) d \tau , x \in [ 0 , 1 ]\tag{12}
$$

where u denotes the input function which is sampled from a Gaussian random field (GRF), and s is the target function.

For the nonlinear operator, the model is trained to learn a particular stochastic partial differential equation (PDE):

$$
\mathrm { d i v } \Big ( e ^ { b ( x ; \omega ) } \nabla u ( x ; \omega ) \Big ) = f ( x ) , x \in [ 0 , 1 ]\tag{13}
$$

where $e ^ { b ( x ; \omega ) }$ is the diffusion efficient and u(x;ω) is the target function.

Denote the domain of output function as y. For both linear and non-linear operator learning, we changed the original uniform sampling of y to three curated regions: few/medium/many. Afterward, we manually created an imbalanced training set of 10k samples and a balanced testing test of 100k samples, namely OL-DIR.

In Figure 6, we have a schematic overview of Imbalanced Operator Learning (IOL). The network is trained to model an integral operator G. The data provided to the model is $( [ u , y ] , G ( u ) ( y ) )$ ). The input consists of function u and sampled ys from the domain of $G ( u )$ . The target is $G ( u ) ( y )$ . We manipulate the distribution density of $y$ across its range to formulate few/med./many regions. Here the imbalance comes from the unequal exposure of integral interval to the model training. Refer to Supplementary Material 7.2 for more details.

Shown in Table 6, SRL consistently outperforms VANILLA and the state-of-the-art operator learning for the whole label range including all, many-shot, medium-shot, and few-shot regions. The results position SRL as the superior approach for IOL in terms of accuracy and generalizability.

## 4.4. Further Analysis

Quantification of geometric impact. We further quantified the impact of geometric constraints by comparing percentages of uniformly sampled points within few-shot regions (a measure of proportion). The results show our method significantly increases few-shot proportion (AgeDB-DIR: 1.98%→15.80%, upper bound: 23%; STS-B-DIR: 4.52% → 22.39%, upper bound: 38%), leading to improved performance (Table 7).

![](images/e7a5fe40109e4efb654ae62454c5c59fc890166b8edd4eb4c1d8eeaaad0c191c.jpg)  
Figure 6. Imbalanced Operator Learning.

Table 6. Results on OL-DIR. We report the average MAE of ten runs. The best results are bold.
<table><tr><td>Operation</td><td colspan="4">MAE(10  $^ { \cdot 3 } ) \downarrow$ </td><td colspan="4">MSE (10  $^ { \cdot 4 } ) \downarrow$ </td></tr><tr><td>Shot</td><td>All</td><td>Many</td><td>Med</td><td>Few</td><td>All</td><td>Many</td><td>Med</td><td>Few</td></tr><tr><td>Linear</td><td colspan="8"></td></tr><tr><td>VANILLA [14]</td><td>15.64</td><td>11.86</td><td>15.45</td><td>27.00</td><td>5.40</td><td>2.81</td><td>4.40</td><td>14.20</td></tr><tr><td>Ordinal Entropy [29]</td><td>10.07</td><td>9.26</td><td>9.85</td><td>13.01</td><td>2.00</td><td>1.53</td><td>1.89</td><td>3.42</td></tr><tr><td>SRL (ours)</td><td>9.18</td><td>8.32</td><td>9.47</td><td>9.33</td><td>1.98</td><td>0.98</td><td>1.72</td><td>2.67</td></tr><tr><td>Nonlinear</td><td colspan="8"></td></tr><tr><td>VANILLA [14]</td><td>11.64</td><td>9.89</td><td>11.02</td><td>19.77</td><td>9.20</td><td>4.33</td><td>7.53</td><td>24.70</td></tr><tr><td>Ordinal Entropy [29]</td><td>12.91</td><td>9.93</td><td>13.07</td><td>21.02</td><td>13.80</td><td>8.82</td><td>11.84</td><td>30.12</td></tr><tr><td>SRL (ours)</td><td>11.25</td><td>9.48</td><td>9.22</td><td>17.00</td><td>8.60</td><td>7.42</td><td>6.41</td><td>14.12</td></tr></table>

Full results with standard deviation are reported in Supplementary Material 8.10.

Table 7. Impact of geometric constraints on few-shot proportion.
<table><tr><td rowspan="2"></td><td colspan="2">Few-shot</td><td>Overall</td></tr><tr><td>Proportion</td><td>MAE</td><td>MAE</td></tr><tr><td colspan="3">AgeDB-DIR (1.10% samples, 23% label range):</td><td></td></tr><tr><td>VANILLA</td><td>1.98%</td><td>12.61</td><td>7.67</td></tr><tr><td>LDS + FDS [25]</td><td>4.95%</td><td>10.52</td><td>7.55</td></tr><tr><td>Ours</td><td>15.80%</td><td>9.81</td><td>7.22</td></tr><tr><td colspan="3">STS-B-DIR (3.49% samples, 38% label range):</td><td></td></tr><tr><td>VANILLA</td><td>4.52%</td><td>1.075</td><td>0.993</td></tr><tr><td>LDS + FDS [25]</td><td>8.13%</td><td>0.905</td><td>0.900</td></tr><tr><td>Ours</td><td>22.39%</td><td>0.877</td><td>0.745</td></tr></table>

Balancing of enveloping and homogeneity: Our proposed SRL advocates for two pivotal geometric constraints in feature distribution: enveloping and homogeneity, to effectively address imbalanced regression. These two losses are modulated by their respective coefficients, $\lambda _ { e }$ for the enveloping loss and $\lambda _ { h }$ for the homogeneity loss. Figure 7 illustrates that the omission of either constraint detrimentally impacts the performance, highlighting the importance of both of them, and it demonstrates that the best performance, as measured by MAE on the AgeDB-DIR dataset, is achieved when both coefficients $\lambda _ { e }$ and $\lambda _ { h }$ are set to $1 e ^ { - 1 }$

Ablation studies on choices of N: Table 10 (in Supplementary Material) shows that achieving optimal performance on the AgeDB-DIR and IMDB-WIKI-DIR datasets requires a sufficiently large $N ,$ as a smaller N may lead to imprecise calculation of the enveloping loss.

Ablation studies on proposed loss component: Table 11 (in Supp.) demonstrates that incorporating homogeneity, enveloping, and contrastive loss term yields superior model performance compared to using each individually.

Computational cost: As shown in Table 12 (in Supp.), the computational overhead introduced by the Surrogate-driven Representation Learning (SRL) framework is comparable to that of other imbalanced regression methods.

Impact of bin numbers: As shown in Table 13 (in Supp.), while increasing the number of bins generally leads to better model performance, the improvements become marginal beyond certain thresholds.

Limitations: Section 11 (in Supp.) examines the limitations of SRL, including its inability to handle higher-dimensional labels.

![](images/016734a9dc8f5f426620c4ae56a2fc978d1dce7dd1ec0c676f18d96a11149be3.jpg)  
Figure 7. Confusion matrix of MAE on AgeDB-DIR from different values of $\lambda _ { h }$ and $\lambda _ { e }$

## 5. Conclusion

As the first work of exploring uniformity in deep imbalanced regression, we introduce two novel loss functions - enveloping and homogeneity loss - to encourage the uniform feature distribution of an ordered and continuous trajectory. The two loss functions serve as geometric constraints which are integrated into the data representations through a Surrogate-driven Representation Learning (SRL) framework. Furthermore, we set a new benchmark in imbalanced regression: Imbalanced Operator Learning (IOL). Extensive experiments on real-world regression and operator learning demonstrate the effectiveness of our geometrically informed approach. We emphasize the significance of uniform data representation and its impact on learning performance in imbalanced regression scenarios, advocating for a more balanced and comprehensive utilization of feature spaces in regression models.

## References

[1] Mido Assran, Randall Balestriero, Quentin Duval, Florian Bordes, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, and Nicolas Ballas. The hidden uniform cluster prior in self-supervised learning. In The Eleventh International Conference on Learning Representations, 2022. 3

[2] Arthur Asuncion and David Newman. Uci machine learning repository, 2007. 1, 2, 5, 6

[3] Daniel Cer, Mona Diab, Eneko Agirre, Inigo Lopez-Gazpio, and Lucia Specia. Semeval-2017 task 1: Semantic textual similarity-multilingual and cross-lingual focused evaluation. arXiv preprint arXiv:1708.00055, 2017. 6

[4] Xin Cheng, Yuzhou Cao, Ximing Li, Bo An, and Lei Feng. Weakly supervised regression with interval targets. arXiv preprint arXiv:2306.10458, 2023. 6

[5] Jiequan Cui, Zhisheng Zhong, Shu Liu, Bei Yu, and Jiaya Jia. Parametric contrastive learning. In Proceedings ofthe IEEE/CVF international conference on computer vision, pages 715–724, 2021. 1, 3

[6] Yu Gong, Greg Mori, and Fred Tung. Ranksim: Ranking similarity regularization for deep imbalanced regression. In International Conference on Machine Learning, pages 7634–7649. PMLR, 2022. 1, 3, 5, 6, 7, 4

[7] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016. 6

[8] Bingyi Kang, Saining Xie, Marcus Rohrbach, Zhicheng Yan, Albert Gordo, Jiashi Feng, and Yannis Kalantidis. Decoupling representation and classifier for long-tailed recognition. In International Conference on Learning Representations, 2019. 1, 2, 3

[9] Bingyi Kang, Yu Li, Sa Xie, Zehuan Yuan, and Jiashi Feng. Exploring balanced feature spaces for representation learning. In International Conference on Learning Representations, 2020. 1, 3

[10] Mahsa Keramati, Lili Meng, and R David Evans. Conr: Contrastive regularizer for deep imbalanced regression. arXiv preprint arXiv:2309.06651, 2023. 1, 3, 5, 6, 7, 4

[11] Nikola Kovachki, Zongyi Li, Burigede Liu, Kamyar Azizzadenesheli, Kaushik Bhattacharya, Andrew Stuart, and Anima Anandkumar. Neural operator: Learning maps between function spaces with applications to pdes. Journal of Machine Learning Research, 24(89):1–97, 2023. 7

[12] Tianhong Li, Peng Cao, Yuan Yuan, Lijie Fan, Yuzhe Yang, Rogerio S Feris, Piotr Indyk, and Dina Katabi. Targeted supervised contrastive learning for long-tailed recognition. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6918–6928, 2022. 1, 3

[13] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2018. 5

[14] Lu Lu, Pengzhan Jin, Guofei Pang, Zhongqiang Zhang, and George Em Karniadakis. Learning nonlinear operators via deeponet based on the universal approximation theorem of operators. Nature machine intelligence, 3(3):218–229, 2021. 7, 8, 1

[15] Stylianos Moschoglou, Athanasios Papaioannou, Christos Sagonas, Jiankang Deng, Irene Kotsia, and Stefanos Zafeiriou. Agedb: the first manually collected, in-the-wild age database.

In proceedings of the IEEE conference on computer vision and pattern recognition workshops, pages 51–59, 2017. 6

[16] Jiawei Ren, Mingyuan Zhang, Cunjun Yu, and Ziwei Liu. Balanced mse for imbalanced visual regression. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7926–7935, 2022. 1, 3, 6, 7

[17] Christian P Robert, George Casella, and George Casella. Monte Carlo statistical methods. Springer, 1999. 4

[18] Rasmus Rothe, Radu Timofte, and Luc Van Gool. Deep expectation of real and apparent age from a single image without facial landmarks. International Journal of Computer Vision, 126 (2-4):144–157, 2018. 6

[19] Michael Steininger, Konstantin Kobs, Padraig Davidson, Anna Krause, and Andreas Hotho. Density-based weighting for imbalanced regression. Machine Learning, 110:2187–2211, 2021. 1, 3

[20] Laurens Van der Maaten and Geoffrey Hinton. Visualizing data using t-sne. Journal ofmachine learning research, 9(11), 2008. 2

[21] Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R Bowman. Glue: A multi-task benchmark and analysis platform for natural language understanding. arXiv preprint arXiv:1804.07461, 2018. 6

[22] Tongzhou Wang and Phillip Isola. Understanding contrastive representation learning through alignment and uniformity on the hypersphere. In International Conference on Machine Learning, pages 9929–9939. PMLR, 2020. 1, 2, 3

[23] Ziyan Wang and Hao Wang. Variational imbalanced regression: Fair uncertainty quantification via probabilistic smoothing. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. 3

[24] Yilei Wu, Zijian Dong, Chongyao Chen, Wangchunshu Zhou, and Juan Helen Zhou. Mixup your own pairs. arXiv preprin arXiv:2309.16633, 2023. 1, 3

[25] Yuzhe Yang, Kaiwen Zha, Yingcong Chen, Hao Wang, and Dina Katabi. Delving into deep imbalanced regression. In International Conference on Machine Learning, pages 11842–11851. PMLR, 2021. 1, 3, 5, 6, 7, 8, 2

[26] Xi Yin, Xiang Yu, Kihyuk Sohn, Xiaoming Liu, and Manmohan Chandraker. Feature transfer learning for face recognition with under-represented data. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 5704–5713, 2019. 1, 3

[27] Kaiwen Zha, Peng Cao, Jeany Son, Yuzhe Yang, and Dina Katabi. Rank-n-contrast: Learning continuous representations for regression. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. 1, 3

[28] Shihao Zhang, Linlin Yang, Michael Bi Mi, Xiaoxu Zheng, and Angela Yao. Improving deep regression with ordinal entropy. In The Eleventh International Conference on Learning Representations, 2022. 6

[29] Shihao Zhang, Linlin Yang, Michael Bi Mi, Xiaoxu Zheng, and Angela Yao. Improving deep regression with ordinal entropy. arXiv preprint arXiv:2301.08915, 2023. 6, 8

[30] Yifan Zhang, Bingyi Kang, Bryan Hooi, Shuicheng Yan, and Jiashi Feng. Deep long-tailed learning: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2023. 1

[31] Zhisheng Zhong, Jiequan Cui, Yibo Yang, Xiaoyang Wu, Xiaojuan Qi, Xiangyu Zhang, and Jiaya Jia. Understanding

imbalanced semantic segmentation through neural collapse. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19550–19560, 2023. 1

[32] Zhihan Zhou, Jiangchao Yao, Feng Hong, Ya Zhang, Bo Han, and Yanfeng Wang. Combating representation learning disparity with geometric harmonization. Advances in Neural Information Processing Systems, 36, 2024. 3

[33] Jianggang Zhu, Zheng Wang, Jingjing Chen, Yi-Ping Phoebe Chen, and Yu-Gang Jiang. Balanced contrastive learning for long-tailed visual recognition. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6908–6917, 2022. 1, 3