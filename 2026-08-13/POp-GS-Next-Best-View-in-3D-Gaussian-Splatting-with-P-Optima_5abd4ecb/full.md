This CVPR paper is the Open Access version, provided by the Computer Vision Foundation. Except for this watermark, it is identical to the accepted version; the final published version of the proceedings is available on IEEE Xplore.

# POp-GS: Next Best View in 3D-Gaussian Splatting with P-Optimality

Joey Wilson University of Michigan wilsoniv@umich.edu

Martin Labrie Amazon Lab 126 labrieml@amazon.com

Min Sun Amazon Lab 126 aliensunmin@gmail.com

Marcelino Almeida Amazon Lab 126 mmalmeid@amazon.com

Maani Ghaffari University of Michigan maanigj@umich.edu

Cheng-Hao Kuo Amazon Lab 126 chkuo@amazon.com

Sachit Mahajan Amazon Lab 126 msachit@amazon.com

Omid Ghasemalizadeh Amazon Lab 126 ghasemal@amazon.com

Arnab Sen Amazon Lab 126 senarnie@amazon.com

## Abstract

In this paper, we present a novel algorithm for quantifying uncertainty and information gained within 3D Gaussian Splatting (3D-GS) through P-Optimality. While 3D-GS has proven to be a useful world model with high-quality rasterizations, it does not natively quantify uncertainty or information, posing a challengefor real-world applications such as 3D-GS SLAM. We propose to quantify information gain in 3D-GS by reformulating the problem through the lens of optimal experimental design, which is a classical solution widely used in literature. By restructuring information quantification of 3D-GS through optimal experimental design, we arrive at multiple solutions, of which T-Optimality and D-Optimality perform the best quantitatively and qualitatively as measured on two popular datasets. Additionally, we propose a block diagonal covariance approximation which provides a measure ofcorrelation at the expense ofa greater computation cost.

## 1. Introduction

In order to operate in a novel environment, robots must be capable of quantifying uncertainty in their surroundings due to occlusions and unobserved regions, as well as quantifying information gained by exploring new areas. While many prior methods for map representations were built on a metric space with natural uncertainty quantification, such representations result in discretization errors of the environment [38]. Recently, 3D-Gaussian Splatting (3D-GS) was proposed as a method for high-quality novel-view synthesis [14], which captivated the attention of the robotics community due to its explicit world model representation, with the potential to function as a map [6, 23, 37, 42].

![](images/40a2375d5634f5164e25cfa1a79bf9222ced8f47a4d3b0eaa0ff129c40124351.jpg)  
(a) Uniform  
(b) FisherRF  
(d) Ground Truth  
(c) Ours  
Figure 1. We propose a novel method for calculating information gain from image in 3D Gaussian Splatting. In the above images, each method is provided a set of one hundred candidate views on scenes from the Blender dataset, and selects ten views to train a 3D-GS model on. Compared to state-of-the-art, our method more accurately estimate the information value of images to train a 3D-GS model. In this figure we demonstrate block D-Optimality, however our derivation also provides multiple solutions discussed later.

Given a set of views of a scene, 3D-GS learns to render novel views from any angle through gradient descent. 3D ellipsoids are iteratively fine-tuned, which can then be rasterized to a 2D image at novel views. Several works have shown that the 3D ellipsoids can be extended to include features beyond just color, such as the category of objects the ellipsoid represents, resulting in an improved level of semantic scene understanding [30, 41]. However, 3D-GS does not natively quantify uncertainty, which leads to issues when determining whether an image has been seen before, or when quantifying the information gained by perceiving a new view. While several works have studied this problem for neural radiance fields (NeRF) [8, 25, 26], the explicit representation of 3D-GS presents a unique challenge.

Since the conception of 3D-GS, several works have sought to quantify uncertainty directly upon a trained 3D-GS model [9, 13, 18]. Of these works, one solution which has appeared effective is relating the per-pixel gradient of the 3D-GS parameters to the information or contribution of the parameters. Notably, FisherRF [13] derived a solution for information quantification of views in 3D-GS through a diagonal approximation of Fisher Information [16]. FisherRF has since been applied to mobile manipulation [33] and active perception [21], however it does not leverage prior literature in effective functions for active perception [15, 32] or consider any correlation between parameters.

Classically, the problem of quantifying information gain from views is known as active perception in robotics literature, and has been studied extensive in probabilistic robotics applications [7, 22]. More generally, active perception can be solved through application of optimal experimental design [19], which defines solutions for identifying the most informative design parameters for an experiment in order to learn about a set of unknown parameters [10, 11, 17, 20]. One particular solution to experimental design is the use of P-Optimality, which specifies a class of solutions depending on the covariance matrix and the choice of P. P-Optimality has been successfully applied in robotics to keyframe selection, optimal loop closure, and active perception, however has not been explored in 3D-GS [3, 28, 29].

Building upon the recent work of FisherRF, which shows that a diagonal approximation of the Hessian matrix can be effective when calculating information gain in 3D-GS, we derive a general solution for the covariance matrix in 3D-GS and apply optimal experimental design techniques. Since the covariance matrix is too large to store in memory, we propose a simple diagonal approximation following FisherRF [13], as well as a block diagonal approximation which captures cross-correlation of parameters within the same ellipsoid [9]. From our general derivation, we construct and compare different p-optimality solutions, and find that D-Optimality and T-Optimality lead to significant improvements. We also find that our block diagonal approximation improves information gain quantification compared to the baselines.

To summarize, our contributions are:

1. Derive a general solution for the covariance matrix of 3D-GS, which allows application of optimal experimental design techniques.

2. Propose several novel methods for quantifying the information value of images in 3D-GS.

3. Block-diagonal approximation of 3D-GS covariance matrix which captures correlation between parameters of

the same ellipsoid.

4. Quantitative and qualitative comparison of optimal experimental design solutions on 3D-GS to each-other, and to current state-of-the-art solutions.

## 2. Related Work

In this section, we explore literature related to 3D Gaussian Splatting (3D-GS) and optimal experimental design for active perception. In this paper, we propose to leverage the theory of p-optimality from optimal experimental design literature to quantify information gain within 3D-GS.

## 2.1. 3D Gaussian Splatting

3D Gaussian Splatting (3D-GS) is a new method for novel view synthesis which models scenes through 3D ellipsoids [14], different from previous methods which model scenes implicitly [25, 34]. Each scene is modeled through thousands or millions of 3D ellipsoids where each ellipsoid contain an opacity and color modeled by spherical harmonics to capture lighting effects. 3D ellipsoids are trained through gradient descent, and at inference time are rasterized to create 2D images at candidate views through a process known as “splatting” [6].

Due to the explicit representation of scenes as 3D ellipsoids, 3D-GS has attracted a significant amount of attention from the computer vision and robotics research communities [23, 39, 42]. 3D-GS has the potential to substitute as a more expressive world model representation, and many works have explored adding additional features such as from vision-language (VL) networks to create higher levels of scene understanding [30, 41]. However, despite the name, 3D-GS does not provide a measure of uncertainty which limits applications in safety-critical or resource-constrained environments. Recently, several works have investigated uncertainty quantification and have found a promising research direction of relating Fisher Information to the explicit 3D-GS parameters [9, 13]. In particular, FisherRF developed a formulation for calculating information gain which treats the 3D-GS model as a black box, not requiring any additional training. However, FisherRF does not leverage any inter-parameter correlation when calculating the information of values to images, and does not utilize the rich literature of optimal experimental design and active perception which derive solutions for calculating information gain.

## 2.2. Active Perception

Active perception is a well-studied problem in robotics literature which seeks to identify the optimal path to improve a map. Since capturing new views in the real world requires robot traversal, a significant amount of research has focused on optimal solutions to determining which view-points are most valuable. One successful approach is the application of P-optimality [15], which defines a class of optimal experimental design solutions based on functionals of the covariance matrix which vary with the choice of an integer p. P-Optimality (P-Opt.) has been widely and successfully applied to SLAM prior to the conception of 3D-GS in keyframe selection, optimal loop closure, and active perception [3, 28, 29].

Early research in optimal decision-taking for Simultaneous Localization and Mapping problems used T-optimality due to efficient computation as the trace of the covariance matrix [24, 32], eliminating the need to compute the eigenvalues. On the other hand, recent research has focused on D-optimality as a reliable metric for an Optimal Experimental Design [4, 27–29]. As advocated in [3], D-optimality seems to be a more rewarding metric towards the fulfilling of an active mapping session. The recent success of Doptimality can also be explained by its monotonicity property in active mapping scenarios [31], which guarantees that uncertainty increases monotonically as a robot move through the scene. In this work, we propose to expand upon the formulation from FisherRF to incorporate parameter correlation and develop a more general solution which allows for application of classical optimal experimental design techniques.

## 3. Method

In this section, we introduce our method for quantifying uncertainty and information gain in 3D-Gaussian Splatting. First, we introduce preliminaries on the 3D Gaussian Splatting representation. Next, we describe our approximation of the covariance matrix for each ellipsoid, which provides a measure of uncertainty on the parameters. Finally, we detail our method for efficiently calculating the informational value of a candidate image.

## 3.1. Preliminaries: Gaussian Splatting

3D Gaussian splatting represents a scene through volumetric rendering of optimized 3D ellipsoids. The geometry of each 3D ellipsoid is parameterized by a center $\mu ,$ scale $S ,$ and rotation R, while the color contribution of each ellipsoid is defined by opacity ↵ and color c. Together, the rotation and scale define the shape of the 3D ellipsoid:

$$
\Sigma = R S S ^ { T } R ^ { T } .\tag{1}
$$

To render images, 3D ellipsoids are first splatted into 2D projections from a provided viewpoint, resulting in a 2D shape ⌃0 and location $\mu ^ { \prime }$ . Next, the contribution $\alpha _ { n } ^ { \prime }$ of each 2D ellipsoid n to pixel $x ^ { \prime }$ is calculated through a kernel as:

$$
\alpha _ { n } ^ { \prime } = \alpha _ { n } \times \exp \left( - \frac { 1 } { 2 } ( x ^ { \prime } - \mu _ { n } ^ { \prime } ) ^ { T } { \Sigma ^ { \prime } } _ { n } ^ { - 1 } ( x ^ { \prime } - \mu _ { n } ^ { \prime } ) \right) .\tag{2}
$$

Finally, 2D ellipsoids are blended into pixels through a process known as alpha compositing, which computes the color of each pixel from a depthwise sorted list of Gaussians ${ \mathcal { N } } \colon$

$$
C = \sum _ { n = 1 } ^ { N } c _ { n } \alpha _ { n } ^ { \prime } \prod _ { j = 1 } ^ { n - 1 } ( 1 - \alpha _ { j } ^ { \prime } ) .\tag{3}
$$

Parameters are optimized by comparing the rendered image from a viewpoint with the ground truth image and performing gradient descent over a weighted loss function [6, 14, 36]:

$$
\begin{array} { r } { \mathcal { L } = ( 1 - \lambda ) \mathcal { L } _ { 1 } + \lambda \mathcal { L } _ { \mathrm { D - S S I M } } , } \end{array}\tag{4}
$$

where $\lambda$ is a weighting function on the $\mathcal { L } 1$ and structural similarity loss. Once the 3D-GS model is fitted to the training data, it can render views from any perspective, however it lacks any information on when the rendering may fail. In order to quantify the amount of information the fitted Gaussian Splatting model has on each parameter, we note that the 1 loss is proportional to the partial derivative of the rendered pixel’s color with respect to the parameters of the 3D-GS model, summed over all pixels in the training set. Based on this insight, we construct our covariance matrix to capture this information.

## 3.2. Information Gain through Optimal Experimental Design

In order to quantify information gained through adding an image, we first require a measure of uncertainty, which is derived in this section. Approaching the problem from a maximum likelihood perspective, the maximum likelihood formulation aims to determine the solution variable $\pmb \theta$ that minimizes the pixel error $e = c - h ( \theta )$ across all pixels in all frames, where $h ( \cdot )$ is the 3D-GS model described in Section 3.1. Assuming that all measured pixel errors are normal zero mean, independent and identically distributed (IID), i.e, $\boldsymbol { e } \sim \mathcal { N } ( \mathbf { 0 } , \sigma _ { e } \cdot \mathbb { I } )$ , then the maximum likelihood formulation seeks an optimal solution variable $\pmb { \theta } \in \mathbb { R } ^ { l \times l }$ that maximizes the likelihood function:

$$
\begin{array} { r } { p ( \pmb { c } | \pmb { \theta } ) = \exp \left( - \frac { 1 } { 2 } \frac { e ^ { T } e } { \pmb { \sigma } _ { e } ^ { 2 } } \right) . } \end{array}\tag{5}
$$

Due to the monotonicity of the log function, maximizing the function above is equivalent to minimizing the loglikelihood function:

$$
- \sigma _ { e } ^ { 2 } \cdot \log p ( c | \theta ) = \frac { 1 } { 2 } e ^ { T } e\tag{6}
$$

Assuming that we have an estimate of the solution variables $\pmb { \theta } _ { \ast }$ , then we can expand the system’s model in the vicinity of $\pmb { \theta } _ { \ast }$ using Taylor expansion as: $h ( \theta ) \approx h ( \theta _ { * } ) +$

![](images/009621fbbce7f5f81110aac58f77483b10720a244cf1aa01e47326b7dbbad296.jpg)  
Figure 2. The Hessian matrix captures the information content of each parameter in the trained 3D-GS model, approximated as perpixel gradients over the training set of images. Since a trained 3D-GS model may contain millions of parameters, we approximate the Hessian matrix through a block or main diagonal.

$J \Delta \theta _ { : }$ , where $\Delta \pmb { \theta } \triangleq \pmb { \theta } - \pmb { \theta } ,$ and $\begin{array} { r } { J \triangleq \frac { \partial h } { \partial \theta } | _ { \theta = \pmb { \theta } _ { * } } } \end{array}$ . We can rewrite the residual function as:

$$
\begin{array} { l } { e \approx c - h ( \theta _ { * } ) - J \Delta \theta } \\ { = e _ { * } - J \Delta \theta , } \end{array}\tag{7}
$$

where the optimal residual is defined as $e _ { * } \triangleq c - h ( \theta _ { * } )$ Substituting this in Eq. 6, we have that:

$$
\frac { 1 } { 2 } e ^ { T } e \approx \frac { 1 } { 2 } e _ { * } ^ { T } e _ { * } - e _ { * } ^ { T } J \Delta \theta + \frac { 1 } { 2 } { \Delta \theta } ^ { T } J ^ { T } J { \Delta \theta } .\tag{8}
$$

In order to satisfy the first order conditions for optimality in Eq. 8, it is necessary for its first order partial derivative w.r.t. the solution variables to be zero [1]. This leads to the Gauss-Newton iterative optimization equation where $\Delta \theta$ is updated as:

$$
\Delta \pmb { \theta } = \left( J ^ { T } J \right) ^ { - 1 } J ^ { T } \pmb { e } _ { \ast } .\tag{9}
$$

Assuming that the optimal solution ✓ is unbiased, then it follows that the expected residual $\mathbb { E } [ e _ { * } ] = \mathbf { 0 }$ and $\mathbb { E } [ e _ { * } e _ { * } ^ { T } ] = \sigma _ { e } \cdot \mathbb { I }$ , leading to $\mathbb { E } [ \Delta \pmb { \theta } ] = \mathbf { 0 }$ and:

$$
\begin{array} { r l } & { \mathbb { E } [ \Delta \pmb { \theta } \Delta \pmb { \theta } ^ { T } ] = \left( \pmb { J } ^ { T } \pmb { J } \right) ^ { - 1 } \pmb { J } ^ { T } \mathbb { E } [ e _ { * } e _ { * } ^ { T } ] \pmb { J } \left( \pmb { J } ^ { T } \pmb { J } \right) ^ { - 1 } } \\ & { \quad \quad \quad = \sigma _ { e } ^ { 2 } \left( \pmb { J } ^ { T } \pmb { J } \right) ^ { - 1 } \pmb { J } ^ { T } \pmb { J } ^ { T } \pmb { J } \left( \pmb { J } ^ { T } \pmb { J } \right) ^ { - 1 } } \\ & { \quad \quad = \sigma _ { e } ^ { 2 } \left( \pmb { J } ^ { T } \pmb { J } \right) ^ { - 1 } . } \end{array}\tag{10}
$$

Without loss of generality, this work assumes<sup>1</sup> $\sigma _ { e } ~ = ~ 1$ Defining $\pmb { H } \triangleq \mathcal { \hat { I } } ^ { T } \pmb { J } \in \mathcal { \hat { \mathbb { R } } } ^ { l \times l }$ , the matrix H is known as an approximation of the Hessian, or the information matrix

![](images/8fa27c3b7a4c30f250d5dd79b1f879bac45d5410a3caf58dba2f9a812ac81675.jpg)  
Figure 3. Optimal experimental design defines functionals of the eigenvalues of the covariance matrices, each with geometric intuitions. D-Optimality approximates the volume of the covariance matrix, as shown in this figure.

for this nonlinear optimization problem. Therefore, the covariance matrix associated with the solution variables ✓ is given by the inverse of the Hessian (information) matrix.

## 3.2.1. Uncertainty Decrease due to an Added Image

In this paper, we assume that we have a set of n images to choose from (along with their respective original poses) to determine which of the images will lead to maximal uncertainty reduction among all n candidates. Therefore, our Next-Best-View formulation attempts at maximally decreasing uncertainty of the covariance matrix $\Sigma _ { i }$ that is obtained by adding the i-th image to the model, $i \in \{ 1 , \cdots , n \}$ . Note that the contents of the i-th image are not necessary for our formulation, only the pose at which we plan to take the i-th image from. This is an important feature of our solution, as it attempts to evaluate the amount of uncertainty reduction (or information increase) that can be achieved by taking an image from a new perspective without actually having the image available.

In this section, we assume that we already have an initial guess of the map $\theta _ { \ast }$ and that its associated prior Jacobian $J _ { - }$ and Hessian ${ \pmb H } _ { - } ~ = ~ { \pmb J } _ { - } ^ { T } { \pmb J } _ { - }$ have been computed. As we add one new prospective candidate image i taken from a pose $\pmb { p } _ { i }$ , it is possible to compute the Jacobian associated with the new image using prior map parameters $\theta _ { \ast }$ and the image’s pose $\pmb { p } _ { i }$ as $\begin{array} { r } { J _ { i } = \frac { \partial h } { \partial \theta } | _ { \theta _ { * } , p _ { i } } } \end{array}$ . Defining $\pmb { H } _ { i }$ as the Hessian of the problem as we add the i-th image, it can be computed as:

$$
\pmb { H } _ { i } = \pmb { H } _ { - } + \pmb { J } _ { i } ^ { T } \pmb { J } _ { i } .\tag{11}
$$

For exponential likelihoods, there is an asymptotic inverse relationship in the maximum likelihood estimator between the Hessian and covariance matrices [35]. Therefore, our goal is to maximize the information $\boldsymbol { \mathcal { T } } \left( \cdot \right)$ obtained from the i-th image, which can be found by minimizing the uncertainty $U \left( \cdot \right)$

$$
\begin{array} { r l } { \arg \operatorname* { m a x } _ { i } \mathcal { I } \left( H _ { i } \right) = \arg \operatorname* { m i n } _ { i } U \left( \Sigma _ { i } \right) } & { } \\ { = \arg \operatorname* { m i n } _ { i } U \left( H _ { i } ^ { - 1 } \right) . } \end{array}\tag{12}
$$

Table 1. Properties of P-Optimality for different values of $p .$ Note: $\lambda _ { k }$ represent the eigenvalues of the covariance matrix $\Sigma _ { i }$
<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>T-optimality</td><td rowspan=1 colspan=1>A-optimality</td><td rowspan=1 colspan=1>D-optimality</td><td rowspan=1 colspan=1>E-optimality</td></tr><tr><td rowspan=1 colspan=1>p</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>-1</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1> $\mp \infty$ </td></tr><tr><td rowspan=1 colspan=1>EquivalentFormulae</td><td rowspan=1 colspan=1> $\begin{array} { r } { \frac { 1 } { l } \mathrm { t r } \left( \Sigma _ { i } \right) = \frac { 1 } { l } \sum _ { k } ^ { l } \lambda _ { k } } \end{array}$ </td><td rowspan=1 colspan=1> $\begin{array} { r l r } {  { ( \frac { 1 } { l } \mathrm { t r } ( H _ { i } ) ) ^ { - 1 } = } } \end{array}$  $\begin{array} { r } { \left( \frac { 1 } { l } \sum _ { k } ^ { l } \lambda _ { k } ^ { - 1 } \right) ^ { - 1 } } \end{array}$ </td><td rowspan=1 colspan=1> $\sqrt [ \ i ] { | { \pmb { \Sigma } } _ { i } | } =$  $\begin{array} { r } { \exp \left( \frac { 1 } { l } \sum _ { k } ^ { l } \log \lambda _ { k } \right) } \end{array}$ </td><td rowspan=1 colspan=1> $\operatorname* { m i n } _ { \lambda _ { k } }$  $\operatorname* { m a x } _ { \lambda _ { k } }$ </td></tr><tr><td rowspan=1 colspan=1>Meaning</td><td rowspan=1 colspan=1>Average Variance</td><td rowspan=1 colspan=1>Harmonic MeanVariance</td><td rowspan=1 colspan=1>Volume of covariancehyper-ellipsoid</td><td rowspan=1 colspan=1>Single extremeeigenvalue</td></tr></table>

In this work, we rely on the Theory of Optimal Experimental Design [15, 29], which defines the P-Optimality uncertainty metric as:

$$
\begin{array} { r } { U _ { p } ( \Sigma _ { i } ) = \left( \frac { 1 } { l } \mathrm { t r a c e } \left( \Sigma _ { i } ^ { p } \right) \right) ^ { \frac { 1 } { p } } , } \end{array}\tag{13}
$$

where $p$ is an integer. Depending on the chosen value for $p ,$ the uncertainty function can have some special properties [27], as detailed in Table 1.

## 3.3. Approximating the Covariance

In practice fitted 3D-GS models may contain millions of parameters, which is intractable due to the cubic computational and quadratic memory complexity of computing the eigenvalues. Therefore, we propose two approximations of the Hessian matrix to save memory and computation.

Simple Diagonal: First, following the work of FisherRF we propose a simple diagonal approximation of the covariance matrix. FisherRF derives the approximation from a Laplace approximation [5], where the covariance matrix is approximated as the main diagonal plus a small regularizing constant  <sub>✓</sub>:

$$
\Sigma _ { \theta } \approx { \mathrm { d i a g } } ( \Sigma _ { i } ) + \lambda _ { \theta } ,\tag{14}
$$

This formulation allows for an efficient and direct comparison of our method of computing information gain with FisherRF, without consideration of correlation. Intuitively, the constant $\lambda _ { \theta }$ represents prior information for our method.

Block Diagonal: In order to capture some of the correlation between 3D-GS parameters, we propose to approximate the Hessian matrix as a block diagonal matrix where each block diagonal element contains the parameters of a single ellipsoid. Please note that a block diagonal approximation has also been proposed within 3D-GS, however the approximation was applied to the task of pruning 3D-GS models to remove redundant ellipsoids [9]. Our insight is that the parameters of the same ellipsoid are most likely to be correlated, and a block diagonal matrix can be processed in parallel on a GPU for efficient computation.

When constructing the block diagonal matrix, we note that computing partial derivatives w.r.t. each pixel leads to singularity issues since the partial derivative of each color is the value $\alpha _ { n } ^ { \prime }$ for the ellipsoid. To avoid singularity issues, we therefore compute the Hessian matrix by separately calculating partial derivatives for each channel of every pixel, instead of for every pixel for all channels. The block diagonal approximation is shown in Fig. 2.

## 3.4. Batch Selection

In practical applications it may be valuable to measure information gain over a set of candidate images, such as along a trajectory or in keyframe selection. Following FisherRF, we implement a simple approach which iteratively adds or removes the most optimal candidate image, updates the Hessian, and repeats the process without additional training. While simple, this batch information implementation does capture the redundancy of views, as the change in the Hessian is reflected when a candidate image is added.

## 4. Results

In this section, we study the effectiveness of our method on quantifying information gained from obtaining new images. Following the experimental setup of FisherRF, we quantitatively and qualitatively evaluate our method against baselines on the task of single view selection, where a 3D-GS model is trained by iteratively selecting the most informative proposal view and fitting the model to a set of candidate view. Next, we compare our method against the same baselines on batch view selection. Third, we compare the ability of each method to quantify uncertainty in novel views by studying the correlation of information gain with reconstruction metrics. Last, we perform ablation studies on the parameters of the 3D-GS model to identify the most important parameters for calculating information gain.

Baselines: We compare against the recently published FisherRF [13], which calculates information gain as:

$$
\begin{array} { r } { \mathcal { T } ( H _ { i } ) = \mathrm { t r } \left( ( J _ { i } ^ { T } J _ { i } ) H _ { - } ^ { - 1 } \right) } \end{array}\tag{15}
$$

with matrices modeled by the simple diagonal approximation. While FisherRF compared against a random and ActiveNeRF baseline [26], we omit these baselines as they performed worse than FisherRF in their experiments, and instead focus on comparing our results with FisherRF. We also add a uniform sampling baseline, which is slightly different from the implementation of random in FisherRF that we found was incorrectly implemented. Instead, our uniform sampling baseline samples views uniformly from the training set, which results in a high coverage of the views in the test set, especially if the training and testing views are similarly distributed. As one of the merits of our approach is a more general solution with optimal experimental design, we implement A, D, E, and T Optimality baselines for comparison. All baselines are compared over the peak signal-tonoise ratio (PSNR), structural similarity index (SSIM) [36], and LPIPS metrics [40].

![](images/22e8d40953eb18bbd5996f71520148b5ba1213404b3387de8d126c317049d2aa.jpg)  
(a) Uniform Sampling

![](images/ffe968bad8083ecceef2e18c8a782ee1091f124640670ba31d884db300ef696f.jpg)  
(b) FisherRF

![](images/98ff61f8077b80164949320fb3fc222b8e97930ad1fef677328dd9838bb3dff7.jpg)  
(c) D-Opt. (Block)

![](images/822e08ea8d3cdd7c3f197ee8a6db6422176ba33392b56e1aa8bd38ac94f2bafe.jpg)  
(d) Ground Truth  
Figure 4. Comparison of view selection methods on the Mip-Nerf360 dataset with 10 views. The columns are built by different methods and in order are: uniform sampling, FisherRF, Block D-GS, and the ground truth image.

Dataset: All methods are compared on two common radiance field datasets. First is the Mip-NeRF360 dataset [2], which is a real-world high-resolution dataset commonly used in novel view synthesis literature as well as by Fish erRF. Mip-NeRF360 contains nine scenes, with five outdoor scenes and four indoor, which we average performance over to obtain the final results. Following FisherRF and the original 3D-GS paper, we train all models at resolutions of 1060 1600 pixels. Additionally, the prior information constant is set to a value of $\lambda _ { \theta } ~ = ~ 1 0 ^ { - 6 }$ for all models. The Mip-NeRF360 dataset contains complex scenes, where the benefit of strong view selection models is clear. However, the dataset also contains some noisy images which are distributed at random, and can impact the results.

Therefore, following FisherRF, we also evaluate all models on the Blender dataset [25], which contains eight highfidelity objects modeled synthetically. While the scenes in this experiment are less complex, this dataset allows us to study information gain quantification without the stochasticity introduced by real-world noisy images.

Table 2. Results on Single View Selection with 10 Views on the Mip-Nerf360 Dataset.
<table><tr><td>Method</td><td>PSNR (↑)</td><td>SSIM (↑)</td><td>LPIPS (↓)</td></tr><tr><td>Uniform Sampling FisherRF</td><td>17.29 16.81</td><td>0.508 0.493</td><td>0.432 0.445</td></tr><tr><td>A-Opt. (Simple)</td><td>15.55</td><td>0.452</td><td>0.480</td></tr><tr><td>E-Opt. (Simple)</td><td>15.33</td><td>0.436</td><td>0.488</td></tr><tr><td>T-Opt. (Simple)</td><td>17.91</td><td>0.520</td><td>0.420</td></tr><tr><td>D-Opt. (Simple)</td><td>17.95</td><td>0.535</td><td>0.411</td></tr><tr><td>D-Opt. (Block)</td><td>18.15</td><td>0.548</td><td>0.401</td></tr></table>

## 4.1. Single View Selection

First, we compare all methods on single view selection, where one candidate view is selected at a time. We follow the experimental setup of FisherRF with minimal modifications, including evaluating models on the ability to select both ten views and twenty views. Note that accurate information quantification is more apparent with ten views due to the limited amount of training information.

Ten Views: For the ten view setup, each method begins with 2 training views, and is trained for 100v iterations, where v is the number of training views. The method then selects a single candidate view to add to the training set, and repeats the process until v = 10, at which point the model trains until a cumulative total of 10, 000 training steps. Qualitative examples on the Mip-Nerf360 dataset are shown in Fig. 4, and qualitative examples on the Blender dataset are shown in Fig. 1.

First, we compare models on the Mip-NeRF360 dataset, whose performance metrics can be found in Table 2. In this experiment, FisherRF is slightly outperformed by the uniform sampling method. We would like to note that uniform sampling is actually an effective method for this experimental setup, and will achieve a near optimal performance if the test and train set are similarly distributed. By reframing the approach to information gain within 3D-GS as optimal experimental design, we find that both T-Optimality and D-Optimality approaches improve significantly over the FisherRF and uniform sampling baselines. Additionally, the block diagonal approximation significantly improves the structural quality measured by SSIM and LPIPS metrics.

Table 3. Results on Single View Selection with 10 Views on the Blender Dataset.
<table><tr><td>Method</td><td>PSNR (↑)</td><td>SSIM (↑)</td><td>LPIPS (↓)</td></tr><tr><td>Uniform Sampling</td><td>23.32</td><td>0.885</td><td>0.101</td></tr><tr><td>FisherRF</td><td>24.59</td><td>0.897</td><td>0.091</td></tr><tr><td>A-Opt. (Simple)</td><td>22.39</td><td>0.876</td><td>0.116</td></tr><tr><td>E-Opt. (Simple)</td><td>21.40</td><td>0.862</td><td>0.129</td></tr><tr><td>T-Opt. (Simple)</td><td>25.40</td><td>0.908</td><td>0.080</td></tr><tr><td>D-Opt. (Simple)</td><td>25.52</td><td>0.909</td><td>0.078</td></tr><tr><td>D-Opt. (Block)</td><td>25.41</td><td>0.908</td><td>0.078</td></tr></table>

Next, we repeat the same set of experiments on the synthetic dataset, shown in Table 3. Here, we find that FisherRF performs significantly better than the uniform sampling method, which may be due to the less complex synthetic scenes, where FisherRF is able to more accurately quantify information on a single object. Similar to before, we find that both T-Optimality and D-Optimality methods outperform the baselines by a wide margin. Across all three metrics, simple and block D-Optimality perform similarly, which we expect is due to the saturation of performance as the approaches are near optimal.

Twenty Views: The twenty view experimental setup follows a very similar approach, of iteratively adding views and training for 100v iterations before selecting the next view. However, this setup begins with v = 4 training views, adds images until v = 20, and trains until a cumulative total of 21, 000 training steps. Experimental results on the MIP dataset can be found in Table 4. Similar to the ten view experiment, we find that uniform sampling slightly outperforms FisherRF while T and D Optimality achieve the highest performance with an improvement from the block diagonal approximation.

Table 4. Results on Single View Selection with 20 Views on the Mip-Nerf360 Dataset.
<table><tr><td>Method</td><td>PSNR (↑)</td><td>SSIM (↑)</td><td>LPIPS (↓)</td></tr><tr><td>Uniform Sampling FisherRF</td><td>20.86 20.89</td><td>0.616 0.608</td><td>0.408 0.416</td></tr><tr><td>A-Opt. (Simple)</td><td>18.62 19.57</td><td>0.558 0.580</td><td>0.452 0.433</td></tr><tr><td>E-Opt. (Simple) T-Opt. (Simple)</td><td>21.07</td><td>0.615</td><td>0.409</td></tr><tr><td>D-Opt. (Simple)</td><td>21.09</td><td>0.624</td><td>0.406</td></tr><tr><td>D-Opt. (Block)</td><td>21.32</td><td>0.636</td><td>0.397</td></tr></table>

## 4.2. Batch View Selection

Next, we compare all methods on batch view selection, where information gain is evaluated over several views simultaneously before training. This problem is more applicable to real-world scenarios such as information gained over a robot trajectory, or identification of the best set of views of an object.

Iterative: In the first experiment on batch view selection we follow the experimental set-up of FisherRF. The procedure is similar to single view selection, however models begin with 4 training views, are trained for 150v iterations between view selections, and select 4 views at a time until 20 views are obtained. All models are trained for a cumulative total of 10, 000 training steps. Results on the Mip-Nerf360 dataset are shown in Table 5, and similar to previous experiments demonstrate superior performance of D and T optimality. Additionally, the block diagonal approximation improves structural quality measured by SSIM and LPIPS metrics. Note that due to the iterative view selection and training, this experiment may not reward batch view diversity, motivating our next experiment.

Keyframe Selection: To further study batch view selection in a setting more similar to SLAM applications, we compare each approach on keyframe selection. All methods are provided the same pre-trained 3D-GS model and select ten keyframes without replacement from a set of views. The selected views are then used to re-train a new 3D-GS model, which is evaluated and compared as a measure of batch information quantification. Results on the Blender dataset are summarized in Table 6, demonstrating low performance of FisherRF, A Optimality and E Optimality which select similar views. Instead, T and D Optimality select informative and different views resulting in a large performance gap.

## 4.3. Correlation with Render Quality

Intuitively, we expect information gain and rasterization quality at view points to be inversely related. For instance, if a 3D-GS model has only been trained on the front side of a chair, view points from the back side would have poor renderings while providing high information gain to the 3D-GS model. Therefore, as another test of our proposed method of quantifying information gain, we create a sparsification plot [12] to study the inverse relationship between image information gain and image render quality.

Table 5. Results on Batch View Selection on Mip-Nerf360 dataset.
<table><tr><td>Method</td><td>PSNR (↑)</td><td>SSIM (↑)</td><td>LPIPS (↓)</td></tr><tr><td>Uniform Sampling</td><td>20.42</td><td>0.613</td><td>0.389</td></tr><tr><td>FisherRF</td><td>20.50</td><td>0.603</td><td>0.399</td></tr><tr><td>A-Opt. (Simple) E-Opt. (Simple)</td><td>18.14 17.88</td><td>0.553 0.535</td><td>0.428</td></tr><tr><td></td><td>20.73</td><td></td><td>0.440</td></tr><tr><td>T-Opt. (Simple)</td><td>20.86</td><td>0.611</td><td>0.391</td></tr><tr><td>D-Opt. (Simple)</td><td></td><td>0.624</td><td>0.383</td></tr><tr><td>D-Opt. (Block)</td><td>20.79</td><td>0.631</td><td>0.378</td></tr></table>

Table 6. Results on Keyframe Selection on Blender Dataset.
<table><tr><td>Method</td><td>PSNR (↑)</td><td>SSIM (↑)</td><td>LPIPS (↓)</td></tr><tr><td>Uniform Sampling</td><td>23.47</td><td>0.888</td><td>0.109</td></tr><tr><td>FisherRF</td><td>18.37</td><td>0.829</td><td>0.184</td></tr><tr><td>A-Opt. (Simple)</td><td>17.05</td><td>0.811</td><td>0.226</td></tr><tr><td>E-Opt. (Simple)</td><td>16.55</td><td>0.786</td><td>0.255</td></tr><tr><td>T-Opt. (Simple)</td><td>24.90</td><td>0.903</td><td>0.096</td></tr><tr><td>D-Opt. (Simple)</td><td>24.26</td><td>0.899</td><td>0.101</td></tr><tr><td>D-Opt. (Block)</td><td>24.53</td><td>0.902</td><td>0.099</td></tr></table>

The sparsification plot in Fig. 5 is created by first training a 3D-GS model on ten randomly selected images for 2, 000 iterations. Each method, using the same random seed and trained model, sorts candidate views by expected information gain. At decile increments, the cumulative average reconstruction quality of views is calculated and plotted for each method. Reading from left to right, the plot indicates the average reconstruction quality of the most informative views. Due to the inverse nature between image uncertainty and information gain, we would expect the most informative candidate views (left) to have low reconstruction quality.

We can see from Fig. 5 that D-Opt. (Block) has a monotomic relationship with the reconstruction quality, whereas FisherRF has difficulty with some objects. To better understand the relative performance, we also include plots of the Uniform Sampling and Oracle methods, where the Uniform Sampling method shows no correlation between the selected images and the reconstruction quality as expected. The Oracle baseline sorts the candidate views by the actual PSNR value, representing a perfect baseline, with similar performance to D-Opt. (Block).

## 4.4. Ablation Study

Last, we conclude with ablation studies on the most important parameters for information gain quantification. While we evaluated our methods with all parameters, reducing the number of parameters can improve computational efficiency. We compare the performance of D-Opt. (Block) with different parameter combinations in Table 7 on the task of single view selection with 10 views on the Blender dataset. Peculiarly, we find that the geometric parameters are important to quantifying information with few images while the spherical harmonics do not result in a significant difference, supporting the approach of PUP 3D-GS [9]. We expect that the opacity decreases performance since our approach does not capture the cross-correlation of ellipsoids. Additionally, we suspect that the spherical harmonics may be more useful with a well fitted scene which already has learned the geometric structure. Nonetheless, these results indicate that information may be evaluated with a minimal set of the geometric parameters, which can increase inference speed. Concretely, on the ship scene from the Blender dataset the memory and latency are as follows: 2.62 GB at 0.15 Hz for full block diagonal, 75.3 MB at 1.71 Hz for block diagonal without spherical harmonics, and 44.4 MB at 12.16 Hz for the simple approximation. Note that the cost of the simple approximation is the same as that of FisherRF.

![](images/f7cdca00ad7e1ecb43611074b0de5fa6fc52365a97bd2fd674e51b4d0980a66f.jpg)  
(a) Ficus

![](images/e452b3ba42c884b2f6e5943fcd1762b9dfd94de4fe089782d959b0fd34857150.jpg)  
(b) Drums  
Figure 5. Correlation of expected information gain with PSNR of candidate views on two objects in Blender dataset.

Table 7. Ablation study on D-Opt. (Block) parameters on Blender Dataset with 10 images. The parameters are sh: spherical harmonics, ↵: opacity, µ: location, R: rotation, and S: scale.
<table><tr><td>Parameters Removed</td><td>PSNR (↑)</td><td>SSIM (↑)</td><td>LPIPS (↓)</td></tr><tr><td> $\{ s h \}$ </td><td>25.52</td><td>0.9089</td><td>0.0784</td></tr><tr><td>{α}</td><td>25.56</td><td>0.9087</td><td>0.0778</td></tr><tr><td> $\{ \mu , { \bf R } , { \bf S } \}$ </td><td>25.36</td><td>0.9074</td><td>0.0793</td></tr><tr><td>∅</td><td>25.41</td><td>0.9084</td><td>0.0776</td></tr></table>

## 5. Conclusion

In this paper, we introduced a novel method for calculating the information gain from images in 3D Gaussian Splatting which builds on prior literature of P-Optimality. Information quantification for 3D-GS is an important problem for evaluating uncertainty in novel environments, selecting key frames for SLAM algorithms, and next best view applications. Our novel formulation leads to a general solution with a simple and block diagonal information matrix approximation, with computational and performance trade-offs. We demonstrate quantitatively that formulating the information quantification with T and D Optimality improves performance compared to the state of the art, supporting results from prior literature. While our method achieves significant results quantitatively and qualitatively, the simple and block diagonal approximations discard correlation between ellipsoids. For future work we would like to investigate re-formulating the problem to include inter-ellipsoid information, such as from the structural similarity loss or the opacity parameter.

## References

[1] Yaakov Bar-Shalom, X Rong Li, and Thiagalingam Kirubarajan. Estimation with Applications to Tracking and Navigation: Theory, Algorithms and Software. John Wiley & Sons, 2004. 4

[2] Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P. Srinivasan, and Peter Hedman. Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields. In Proc. IEEE Conf. Comput. Vis. Pattern Recog., pages 5460–5469, 2022. 6

[3] Henry Carrillo, Ian Reid, and Jose A. Castellanos.´ On the comparison of uncertainty criteria for active SLAM. In Proc. IEEE Int. Conf. Robot. and Automation, pages 2080–2087, 2012. 2, 3

[4] Yongbo Chen, Shoudong Huang, and Robert Fitch. Active SLAM for mobile robots with area coverage and obstacle avoidance. IEEE/ASME Trans. Mecha tronics, 25(3):1182–1192, 2020. 3

[5] Erik Daxberger, Agustinus Kristiadi, Alexander Immer, Runa Eschenhagen, Matthias Bauer, and Philipp Hennig. Laplace redux – effortless Bayesian deep learning. In Proc. Advances Neural Inform. Process. Syst. Conf., 2021. 5

[6] Ben Fei, Jingyi Xu, Rui Zhang, Qingyuan Zhou, Weidong Yang, and Ying He. 3D Gaussian Splatting as New Era: A Survey. IEEE Trans. Graph., pages 1–20, 2024. 1, 2, 3

[7] Maani Ghaffari Jadidi, Jaime Valls Miro, and Gamini Dissanayake. Gaussian processes autonomous mapping and exploration for range-sensing mobile robots. Auton. Robot., 42(2):273–290, 2018. 2

[8] Lily Goli, Cody Reading, Silvia Sellan, Alec Jacob-´ son, and Andrea Tagliasacchi. Bayes’ Rays: Uncertainty Quantification for Neural Radiance Fields. In Proc. IEEE Conf. Comput. Vis. Pattern Recog., pages 20061–20070, 2024. 2

[9] Alex Hanson, Allen Tu, Vasu Singla, Mayuka Jayawardhana, Matthias Zwicker, and Tom Goldstein. PUP 3D-GS: Principled Uncertainty Pruning for 3D Gaussian Splatting. arXiv, abs/2406.10219, 2024. 2, 5, 8

[10] Xun Huan and Youssef M. Marzouk. Simulationbased optimal Bayesian experimental design for nonlinear systems. J. of Comput. Phys., 232(1):288–317, 2013. 2

[11] Xun Huan and Youssef M. Marzouk. Gradient-based stochastic optimization methods in Bayesian experimental design. Int. J. for Uncertainty Quant., 4(6): 479–510, 2014. 2

[12] Eddy Ilg, Ozg <sup>¨</sup> un C¸ ic¸ek, Silvio Galesso, Aaron Klein, ¨ Osama Makansi, Frank Hutter, and Thomas Brox. Uncertainty Estimates and Multi-hypotheses Networks

for Optical Flow. In Proc. European Conf. Comput. Vis., pages 677–693, 2018. 8

[13] Wen Jiang, Boshu Lei, and Kostas Daniilidis. FisherRF: Active View Selection and Uncertainty Quantification for Radiance Fields using Fisher Information. In Proc. European Conf. Comput. Vis., pages 422–440, 2024. 2, 5

[14] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, and George Drettakis. 3D Gaussian¨ Splatting for Real-Time Radiance Field Rendering. IEEE Trans. Graph., 42(4), 2023. 1, 2, 3

[15] Jack Kiefer. General equivalence theory for optimum designs (approximate theory). The annals of Statistics, pages 849–879, 1974. 2, 3, 5

[16] Andreas Kirsch and Yarin Gal. Unifying Approaches in Active Learning and Active Sampling via Fisher Information and Information-Theoretic Quantities. J. Mach. Learning Res., 2022. 2

[17] S. Kullback and R. A. Leibler. On Information and Sufficiency. The Ann. Math. Stat., 22(1):79 – 86, 1951. 2

[18] Ruiqi Li and Yiu ming Cheung. Variational Multiscale Representation for Estimating Uncertainty in 3D Gaussian Splatting. In Proc. Advances Neural Inform. Process. Syst. Conf., 2024. 2

[19] D. V. Lindley. On a Measure of the Information Provided by an Experiment. The Ann. Math. Stat., 27(4): 986 – 1005, 1956. 2

[20] D. V. Lindley. On a Measure of the Information Provided by an Experiment. The Ann. Math. Stat., 27(4): 986–1005, 1956. 2

[21] Guangyi Liu, Wen Jiang, Boshu Lei, Vivek Pandey, Kostas Daniilidis, and Nader Motee. Beyond Uncertainty: Risk-Aware Active View Acquisition for Safe Robot Navigation and 3D Scene Understanding with FisherRF. arXiv, abs/2403.11396, 2024. 2

[22] Ruben Martinez-Cantin, Nando de Freitas, Eric Brochu, Jose A. Castellanos, and A. Doucet. A´ bayesian exploration-exploitation approach for optimal online sensing and planning with a visually guided mobile robot. Auton. Robot., 27:93–103, 2009. 2

[23] Hidenobu Matsuki, Riku Murai, Paul H.J. Kelly, and Andrew J. Davison. Gaussian Splatting SLAM. In Proc. IEEE Conf. Comput. Vis. Pattern Recog., pages 18039–18048, 2024. 1, 2

[24] Lyudmila Mihaylova, Tine Lefebvre, Herman Bruyninckx, Klaas Gadeyne, and Joris De Schutter. A comparison of decision making criteria and optimization methods for active robotic sensing. In Numerical Methods and Applications, pages 316–324. Springer, 2003. 3

[25] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis. In Proc. European Conf. Comput. Vis., pages 405–421, 2020. 2, 6

[26] Xuran Pan, Zihang Lai, Shiji Song, and Gao Huang. ActiveNeRF: Learning Where to See with Uncertainty Estimation. In Proc. European Conf. Comput. Vis., pages 230–246, 2022. 2, 5

[27] Julio A Placed and Jose A Castellanos. A General Re-´ lationship between Optimality Criteria and Connectivity Indices for Active Graph-SLAM. IEEE Robot. Autom. Letter., 8(2):816–823, 2022. 3, 5

[28] Julio A Placed, Juan J Gomez Rodr´ ´ıguez, Juan D Tardos, and Jos´ e A Castellanos. Explorb-slam: Ac-´ tive visual slam exploiting the pose-graph topology. In Iberian Robotics conference, pages 199–210. Springer, 2022. 2, 3

[29] Julio A. Placed, Jared Strader, Henry Carrillo, Nikolay Atanasov, Vadim Indelman, Luca Carlone, and Jose A.´ Castellanos. A Survey on Active Simultaneous Localization and Mapping: State of the Art and New Frontiers. IEEE Trans. Robot., 39:1686–1705, 2022. 2, 3, 5

[30] Minghan Qin, Wanhua Li, Jiawei Zhou, Haoqian Wang, and Hanspeter Pfister. LangSplat: 3D Language Gaussian Splatting. In Proc. IEEE Conf. Comput. Vis. Pattern Recog., pages 20051–20060, 2024. 1, 2

[31] Mar´ıa L. Rodr´ıguez-Arevalo, Jos´ e Neira, and Jos´ e A.´ Castellanos. On the Importance of Uncertainty Representation in Active SLAM. IEEE Trans. Robot., 34 (3):829–834, 2018. 3

[32] Robert Sim and Nicholas Roy. Global a-optimal robot exploration in slam. In Proc. IEEE Int. Conf. Robot. and Automation, pages 661–666. IEEE, 2005. 2, 3

[33] Matthew Strong, Boshu Lei, Aiden Swann, Wen Jiang, Kostas Daniilidis, and Monroe Kennedy III au2. Next Best Sense: Guiding Vision and Touch with FisherRF for 3D Gaussian Splatting. arXiv, abs/2410.04680, 2024. 2

[34] Matthew Tancik, Vincent Casser, Xinchen Yan, Sabeek Pradhan, Ben Mildenhall, Pratul P. Srinivasan, Jonathan T. Barron, and Henrik Kretzschmar. Block-NeRF: Scalable Large Scene Neural View Synthesis. In Proc. IEEE Conf. Comput. Vis. Pattern Recog., pages 8248–8258, 2022. 2

[35] A. W. van der Vaart. Asymptotic Statistics, chapter 4, page 35–40. Cambridge University Press, 1998. 4

[36] Zhou Wang, A.C. Bovik, H.R. Sheikh, and E.P. Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE Trans. Image Process., 13(4):600–612, 2004. 3, 6

[37] Joey Wilson, Marcelino Almeida, Min Sun, Sachit Mahajan, Maani Ghaffari, Parker Ewen, Omid Ghasemalizadeh, Cheng-Hao Kuo, and Arnie Sen. Modeling Uncertainty in 3D Gaussian Splatting through Continuous Semantic Splatting. ArXiv, abs/2411.02547, 2024. 1

[38] Joey Wilson, Yuewei Fu, Joshua Friesen, Parker Ewen, Andrew Capodieci, Paramsothy Jayakumar, Kira Barton, and Maani Ghaffari. ConvBKI: Real-Time Probabilistic Semantic Mapping Network With Quantifiable Uncertainty. IEEE Trans. Robot., 40: 4648–4667, 2024. 1

[39] Chi Yan, Delin Qu, Dan Xu, Bin Zhao, Zhigang Wang, Dong Wang, and Xuelong Li. GS-SLAM: Dense Visual SLAM with 3D Gaussian Splatting. In Proc. IEEE Conf. Comput. Vis. Pattern Recog., pages 19595–19604, 2024. 2

[40] Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. The Unreasonable Effectiveness of Deep Features as a Perceptual Metric. In Proc. IEEE Conf. Comput. Vis. Pattern Recog., pages 586–595, 2018. 6

[41] Shijie Zhou, Haoran Chang, Sicheng Jiang, Zhiwen Fan, Zehao Zhu, Dejia Xu, Pradyumna Chari, Suya You, Zhangyang Wang, and Achuta Kadambi. Feature 3dgs: Supercharging 3d gaussian splatting to enable distilled feature fields. In Proc. IEEE Conf. Comput. Vis. Pattern Recog., pages 21676–21685, 2024. 1, 2

[42] Liyuan Zhu, Yue Li, Erik Sandstrom, Shengyu Huang, ¨ Konrad Schindler, and Iro Armeni. LoopSplat: Loop Closure by Registering 3D Gaussian Splats. arXiv, abs/2408.10154, 2024. 1, 2