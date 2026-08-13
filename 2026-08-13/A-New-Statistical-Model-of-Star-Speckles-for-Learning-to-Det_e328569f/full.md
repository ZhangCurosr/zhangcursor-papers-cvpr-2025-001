# A New Statistical Model of Star Speckles for Learning to Detect and Characterize Exoplanets in Direct Imaging Observations

Theo Bodrito´ \*<sup>1</sup>, Olivier Flasseur<sup>2</sup>, Julien Mairal<sup>3</sup>, Jean Ponce<sup>1,</sup> <sup>4</sup>, Maud Langlois<sup>2</sup>, Anne-Marie Lagrange<sup>5,</sup> <sup>6</sup>

<sup>1</sup>Departement d’Informatique de l’ ´ Ecole normale sup <sup>´</sup> erieure (ENS-PSL, CNRS, Inria) ´   
<sup>2</sup>Universite Claude Bernard Lyon 1, Centre de Recherche Astrophysique de Lyon UMR 5574, ENS de Lyon, CNRS, Villeurbanne, F-69622, France <sup>3</sup>Universite Grenoble Alpes, Inria, CNRS, Grenoble INP, LJK ´ <sup>4</sup>Courant Institute and Center for Data Science, New York University   
<sup>5</sup>Laboratoire d’Etudes Spatiales et d’Instrumentation en Astrophysique, Observatoire de Paris, <sup>´</sup> Universite PSL, Sorbonne Universit´ e, Universit´ e Paris Diderot´ <sup>6</sup>Universite Grenoble Alpes, Institut de Plan ´ etologie et d’Astrophysique de Grenoble ´

## Abstract

The search for exoplanets is an active field in astronomy, with direct imaging as one of the most challenging methods due to faint exoplanet signals buried within stronger residual starlight. Successful detection requires advanced image processing to separate the exoplanet signal from this nuisance component. This paper presents a novel statistical model that captures nuisance fluctuations using a multiscale approach, leveraging problem symmetries and a joint spectral channel representation grounded in physical principles. Our model integrates into an interpretable, end-toend learnableframeworkfor simultaneous exoplanet detection and flux estimation. The proposed algorithm is evaluated against the state of the art using datasets from the SPHERE instrument operating at the Very Large Telescope (VLT). It significantly improves the precision-recall tradeoff, notably on challenging datasets that are otherwise unusable by astronomers. The proposed approach is computationally efficient, robust to varying data quality, and well suitedfor large-scale observational surveys.

## 1. Introduction

Direct imaging [30, 65] is an astronomical technique to probe the vicinity of young, nearby stars, where exoplanets and circumstellar disks—structures of dust and gas from which exoplanets can form—are found [36, 37]. Unlike indirect methods [57], which detect exoplanets via secondary effects like gravitational wobbles or transit dimming, direct imaging captures visual evidence of exoplanets and circumstellar disks by recording a direct image of their emitted flux. Analyzing this light across spectral bands provides insights into exoplanet properties (e.g., temperature, gravity), atmospheric compositions, molecular abundances, and formation processes [2, 9, 66]. While existing technology has enabled imaging of young, giant, gaseous exoplanets, next-generation ground- and space-based telescopes may soon image rocky exoplanets in habitable zones, advancing the search for life beyond our Solar System [10, 14].

![](images/6a6ef64db95be391efd11ebbc7aedd23795dcc101d994fd706caa9a44d414c23.jpg)  
Figure A. Left: typical observations y and PSF h from the SPHERE instrument in ASDI mode. The synthetic exoplanet is very bright for illustration purposes. Right: temporal slice along the vertical line.

Imaging exoplanets is challenging due to the high contrast and angular resolution required. The primary difficulty lies in the large brightness ratio (or contrast) between the host star and exoplanets; in the infrared, where gas giant exoplanets’ thermal emission is most detectable, they are typically $1 0 ^ { 5 } – 1 0 ^ { 6 }$ times fainter. Additionally, because exoplanets appear close to their host stars from Earth’s perspective, high spatial resolution is essential to separate their faint signals. To address these challenges, observatories like the VLT are equipped with specialized instruments for direct imaging (e.g., SPHERE [4]) and cutting-edge optical technologies. Adaptive optics employs a deformable mirror to obtain sharp images by correcting for atmospheric turbulence in real time [20]. Coronagraphs (optical masks) block some starlight, further improving contrast [58].

Despite advanced optical devices, direct imaging remains challenging as exoplanet signals are still dominated by a strong nuisance component with approximately 10<sup>3</sup> times higher contrast. This nuisance is mainly composed of structured speckles [24], caused by imperfections in optical corrections that allow residual starlight, unblocked by the coronagraph, to leak into images as a spatially correlated diffraction pattern. The speckle pattern is both spatially and spectrally correlated, exhibiting quasi-static behavior across exposures with minor fluctuations over time. In addition to speckles, other stochastic noise sources–such as thermal background, detector readout, and photon noise– add further contamination. Together, these factors create a non-stationary nuisance that varies in intensity and structure across the field of view, with higher intensity and correlation near the star. This nuisance, which closely resembles the point-like signals expected from exoplanets (instrumental point-spread function off the optical axis), is the primary limitation to direct imaging. Figure A illustrates these characteristics with a dataset recorded by SPHERE.

In this context, dedicated processing is crucial to separate exoplanet signals from the nuisance component [50]. Observational techniques like angular differential imaging (ADI [41]), spectral differential imaging (SDI [61]), and combined angular and spectral differential imaging (ASDI [66]) introduce diversity that aids in distinguishing exoplanet signals from the nuisance, see Sec. 3 for resulting image formation models. ADI takes advantage of Earth’s rotation by keeping the telescope pupil fixed, causing offaxis exoplanets to follow a predictable circular trajectory, while the star remains centered. This apparent motion helps separate exoplanet signals from quasi-static speckles. SDI further improves separation by capturing images across multiple spectral channels, where speckles scale quasi-linearly with wavelength while exoplanet signals stay fixed. ASDI combines ADI and SDI to produce a fourdimensional dataset (spatial, temporal, and spectral dimensions). This diversity in A(S)DI calls for advanced algorithms that can leverage these priors to optimize exoplanet detection and spectral characterization.

We propose a novel hybrid approach that combines the interpretability of a statistical framework with end-to-end learnable components, improving detection sensitivity and characterization accuracy for directly imaged exoplanets. Our main contributions include:

• A learnable architecture based on a statistical model integrating pixel correlations across spatial scales and leveraging the spatial symmetries of the nuisance component, improving both detection and characterization.

• Statistically reliable detection scores and estimates with uncertainty quantification, essential for astrophysics.

• Joint processing of 4-D datasets incorporating the ASDI forward model, shown to significantly boost detection.

## 2. Related work

Various methods have been developed to isolate faint planetary signals from stellar nuisance, broadly categorized as subtraction-, statistical-, and learning-based models [50].

Subtraction-based methods. Subtraction-based methods are among the earliest and most common techniques for exoplanet detection in direct imaging, aiming to remove quasistatic speckles that obscure faint signals. The cA(S)DI algorithms [38, 41] subtract a reference model by averaging frames and stacking residuals, enhancing the rotating exoplanet signal. TLOCI [44] and its variants (e.g., [32, 68]) optimize linear combinations of images to model speckles, while KLIP/PCA [3, 60] employs principal component analysis for low-rank subspace projection. Other approaches, such as non-negative matrix factorization [33, 49, 51, 52] and LLSG [34], decompose data into components to isolate sources, while the RSM algorithm [17–19] combines outputs from multiple methods to reduce individual biases. However, these methods often lack statistically rigorous outputs, such as interpretable detection scores and unbiased flux estimates, as they rely on heuristic image combinations rather than fully end-to-end models.

Statistical approaches. To address the previous limitations, statistical methods model the nuisance using probabilistic frameworks. The ANDROMEDA [6] and FMMF [54] approaches rely on matched filtering, simplifying the nuisance as uncorrelated Gaussian noise. Similarly, SNAP [64] estimates both the nuisance and exoplanet components jointly through maximum likelihood under the same assumption. The PACO algorithm [25, 26] improves on this by extending beyond the white noise assumption. It uses a patch-based statistical framework to model the spatial and spectral covariances of the nuisance, effectively capturing the local structure of speckles. This approach draws on techniques from computer vision, such as denoising, restoration, super-resolution, collaborative filtering [15], sparse coding [1, 40], or mixture models [71, 72].

Learning-based approaches. Machine learning methods are increasingly used in high-contrast imaging, inspired by advances in fields like photography and biomedical imaging. Early applications in exoplanet detection, such as the SVM model by [23], leverage structured high-contrast data, while the (NA)-SODINN algorithms [8, 35] frames detection as a binary classification problem, using KLIP/PCAprocessed patches with random forests or CNNs. Despite their effectiveness, SODINN struggles with high false alarm rates and complex hyperparameter tuning [7]. Generative adversarial networks (GANs) have also been applied to simulate nuisance patterns for training deep learning models [70]. Approaches like TRAP [55] and HSR [31] handle unmixing via regularized regression, modeling nuisance evolution with signal-free reference data. Deep PACO [27, 28] combines statistical modeling with deep learning, leveraging PACO’s nuisance statistical model and a CNN to detect exoplanets and refine residual mismatches.

All these algorithms are observation-dependent, building nuisance models directly from the dataset of interest. This dependency hinders detection near the host star due to (i) significant temporal fluctuations in the nuisance and (ii) residual self-subtraction, where part of the exoplanet signal is mistakenly removed. Recent observationindependent approaches address these limitations by using archival survey data to model the nuisance. Super-RDI [56] extends KLIP/PCA for large observational databases, while ConStruct [69] uses an auto-encoder to learn typical speckle patterns, and [11] employs a discriminative nuisance model. MODEL&CO [5] improves deep PACO with a deep statistically-modeled nuisance framework. Our proposed approach, ExoMILD (Exoplanet imaging by MIxture of Learnable Distributions), belongs to this new category.

## 3. Image formation models

In direct imaging, a stellar system is observed through an optical system that includes the atmosphere, telescope, and scientific instrument. The response of this system to a point source (e.g., an exoplanet) is defined by its point-spread function off the optical axis (off-axis PSF), describing how light from the source is distributed across the sensor. However, as noted in Sec. 1, residual aberrations uncorrected by adaptive optics produce a quasi-static, structured speckle pattern. To mitigate speckles through numerical processing, several observational strategies are used, as detailed next.

## 3.1. Angular Differential Imaging (ADI)

In ADI, the field derotator of the telescope is turned off causing the field of view (including any exoplanets) to rotate around the target star due to Earth’s rotation. Meanwhile, the optical system remains stationary, causing the speckles to remain fixed. This distinction helps separating exoplanet signals from speckle noise. Formally, let y in $\mathbb { R } ^ { T \times H \times W }$ denote the sequence of measurements from the observation, where T is the number of exposures and H, W represent the pixel dimensions of each exposure. Each frame $\mathbf { \mathscr { y } } _ { t }$ in $\mathbb { R } ^ { H \times W }$ can then be represented as:

$$
y _ { t } = s _ { t } + et { } { ' } \sum _ { k = 0 } \alpha ^ { ( k ) } \pmb { h } ( \pmb { x } _ { t } ^ { ( k ) } ) + \epsilon _ { t } ,\tag{1}
$$

where $\mathbf { \boldsymbol { s } } _ { t }$ in R $H \times W$ is the speckles component, K is the (unknown) number of exoplanets, $\mathbf { \pmb { x } } _ { t } ^ { ( k ) }$ is the position of exoplanet k at time $t , h ( x )$ in $\mathbb { R } ^ { H \times W }$ is the PSF model centered on position x in $\mathbb { R } ^ { 2 } , \alpha ^ { ( k ) }$ in $\mathbb { R } _ { + }$ is the flux of the exoplanet, and $\epsilon _ { t }$ in $\mathbb { R } ^ { H \times W }$ is stochastic noise. The position of exoplanet k at time t can be written as $\pmb { x } _ { t } ^ { ( k ) } \bar { \bf \Phi } = r \big ( \pmb { x } _ { 0 } ^ { ( k ) } , \phi _ { t } \big )$ where $\pmb { x } _ { 0 } ^ { ( k ) }$ in $\mathbb { R } ^ { 2 }$ is its initial position on frame ${ \mathbf { } } y _ { 0 } , \phi _ { t }$ in R is the predictable parallactic angle at time t, defined as the cumulated apparent rotation (induced by Earth’s rotation) since the start of the sequence, and $r : \mathbb { R } ^ { 2 } \times \mathbb { R } \to \mathbb { R } ^ { 2 }$ defines a rotation trajectory centered on the star.

## 3.2. Angular Spectral Differential Imaging (ASDI)

The speckles pattern induced by the star can be understood as the PSF on the optical axis (on-axis PSF) of the optical system. According to diffraction laws, this pattern scales homothetically with wavelength, creating a chromatic effect similar to chromatic aberrations in photography. In ASDI, this spectral scaling helps further disentangle exoplanet signals from speckles. In this case, each observation is denoted as y in $\mathbb { R } ^ { \tilde { C } \times T \times H \times W }$ , where C is the number of channels. The forward model (1) becomes

$$
\begin{array} { r } { \pmb { y } _ { c , t } = \beta _ { c } \mathbf { D } _ { \lambda _ { c } / \lambda _ { 0 } } \pmb { s } _ { 0 , t } + \sum _ { k = 0 } ^ { K - 1 } \alpha _ { c } ^ { ( k ) } \pmb { h } _ { c } ( \pmb { x } _ { t } ^ { ( k ) } ) + \pmb { \epsilon } _ { c , t } , } \end{array}\tag{2}
$$

where $\beta _ { c }$ in R is the amplitude of the speckles in channel $c , \mathbf { D } _ { \lambda _ { c } / \lambda _ { 0 } }$ is the homothety operator aligning channel 0 on channel c with a dilation coefficient defined as the ratio of their wavelengths, $s _ { 0 , t }$ in $\mathbb { R } ^ { H \times W }$ is speckles pattern at wavelength $\lambda _ { 0 } , \epsilon _ { c , t }$ in $\mathbb { R } ^ { H \times W }$ is the additive thermal noise. In this setting, the exoplanet flux $\alpha _ { c }$ and the off-axis PSF $h _ { c }$ now depend on channel c.

## 4. Proposed method

## 4.1. Convolutional statistical model

Local Gaussian model of speckles. Building on the PACO algorithm [25, 26], we propose to capture local spatial correlations between pixels of the nuisance term. We denote by $\mathcal { G } _ { p }$ the grid of spatial pixels, such that $| { \mathcal { G } } _ { p } | =$ $H W$ and ∀i in $[ [ 0 , H W - 1 ] ] , x _ { i } \mathrm { i n } \mathbb { R } ^ { 2 }$ . Then, we model the statistical distribution of collections of patches positioned on a spatial grid $\mathcal { G } _ { d }$ , such that $\mathcal { G } _ { d } \subset \mathcal { G } _ { p }$ , with $M : = | { \mathcal { G } } _ { d } |$ We denote by $\mathbf { \pmb { y } } _ { t } ^ { ( j ) }$ in $\mathbb { R } ^ { p } , \forall ( j , t ) \in [ [ 0 , M - 1 ] ] \times [ [ 0 , T - 1 ] ]$ the observed patch in spatial location $j$ at time t. In absence of exoplanet, i.e., when only the nuisance component is present, each collection of patches $\{ \pmb { y } _ { t } ^ { ( j ) } \} _ { t = 0 : T - 1 }$ is modeled by a multivariate Gaussian:

![](images/968d88858091c32800e195e5a43bb1940e685b8ba70e64e777d4ded6b9704836.jpg)  
Figure B. Workflow of the proposed method: it exploits both the spectral behavior of speckles and the apparent motion of exoplanets to disentangle the exoplanet signal from the nuisance component in the observations $\mathbf { \pmb { y } } .$ To achieve this, local patches of the nuisance are modeled as Gaussian distributions, leveraging problem symmetries and incorporating multiple scales. These patches are fed to our convolutional statistical model, and combined to form a detection map. Additionally, a learned object prior, represented by a UNet $f _ { \nu } ,$ is introduced to denoise this detection map produced by the statistical model. This approach results in an end-to-end learnable architecture.

![](images/76dbf8571efa5a608d4a2b8c9547dd1ec9e0eafb2568de006265fb8a2e0d4cc3.jpg)  
Figure C. Proposed convolutional statistical model: spectrally aligned speckles patches, indexed by j, with dimensions $N p$ and $C T$ samples, are first linearly projected into a lower-dimensional space of size m. In this space, the parameters of the Gaussian distribution $\widehat { \pmb { m } } _ { j }$ and $\widehat { \mathbf { C } } _ { j }$ are estimated and subsequently combined with the PSF h to compute the terms $\mathbf { \pmb { a } } ^ { ( j ) }$ and $\bar { \pmb { b } } ^ { ( j ) }$ . As detailed in Appendix ${ \mathrm { A . 3 } } ,$ , the efficient computation of $\mathbf { \pmb { a } } ^ { ( j ) }$ relies on the Cholesky decomposition of the precision matrix $\widehat { \mathbf { C } } _ { j } ^ { - }$ 1

$$
\forall ( j , t ) , \ : y _ { t } ^ { ( j ) } \sim \mathcal { N } ( m _ { j } , \mathbf { C } _ { j } ) ,\tag{3}
$$

with $m _ { j }$ in $\mathbb { R } ^ { p }$ and $\mathbf { C } _ { j }$ in $\mathbb { R } ^ { p \times p }$ the mean and covariance of the Gaussian distribution. These parameters are estimated by maximum likelihood and shrinkage of the covariance matrix. They are denoted as $\widehat { \pmb { m } } _ { j }$ and $\mathbf { \widetilde { C } } _ { j }$ in the following.  Additional details are presented in Appendix A.1.

Detection criterion at a given position. This local statistical model estimates the time-invariant flux $\widehat { \alpha }$ of an exoplanet at position $\scriptstyle { \mathbf { { \mathit { x } } } } _ { \mathrm { { 0 } } }$ in $\mathbb { R } ^ { 2 }$ in the first frame by maximizing the global likelihood:

$$
{ \widehat { \alpha } } = \arg \operatorname* { m a x } _ { \alpha } \ell ( \alpha , \mathbf { x } _ { 0 } ) .\tag{4}
$$

Assuming that collections of patches on the trajectory of an exoplanet are independent, the likelihood becomes:

$$
\ell ( \alpha , \pmb { x } _ { 0 } ) = \prod _ { t } \mathbb { P } \left( \pmb { y } _ { t } ^ { ( i _ { t } ) } - \alpha \pmb { h } ^ { ( i _ { t } ) } ( \pmb { x } _ { t } ) \vert \widehat { \pmb { m } } _ { i _ { t } } , \widehat { \mathbf { C } } _ { i _ { t } } \right)\tag{5}
$$

where ${ \pmb x } _ { t } = r ( { \pmb x } _ { 0 } , \phi _ { t } )$ is the predictive position of the exoplanet at time $t , i _ { t }$ the index of the patch centered on position $\lfloor x _ { t } \rceil$ in $\mathcal { G } _ { p } .$ , and $h ^ { ( i _ { t } ) } ( \pmb { x } _ { t } )$ in $\mathbb { R } ^ { p }$ the patch $i _ { t }$ of the off-axis PSF model centered on $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ . We propose to consider the modified likelihood:

$$
\ell ( \alpha , \pmb { x } _ { 0 } ) = \prod _ { t } \prod _ { j \in S ( \pmb { x } _ { t } ) } \mathbb { P } \left( \pmb { y } _ { t } ^ { ( j ) } - \alpha h ^ { ( j ) } ( \pmb { x } _ { t } ) | \widehat { m } _ { j } , \widehat { \mathbf { C } } _ { j } \right) ^ { w _ { j } } ,\tag{6}
$$

where $S ( { \pmb x } _ { t } )$ represents the subset of modeled distributions for patches around location $\mathbf { \nabla } _ { \mathbf { x } _ { t } } .$ , with each distribution $j$ weighted by $w _ { j }$ in $\mathbb { R } ^ { + }$ , where $\textstyle \sum w _ { j } \ = \ 1$ . Compared to $( 5 )$ , this approach is more robust, as it aggregates overlapping patch contributions at each time-step, modeling the nuisance component as a convolutional process influenced by multiple, overlapping noise sources. It is also more computationally efficient, as it does not require a dense grid $\mathcal { G } _ { d }$ for patch collections, thereby reducing the parameter estimation load. This efficiency is essential for incorporating the model into an end-to-end learning framework (see Sec. 4.3). Finally, this flexible formulation can accommodate additional correlation types (see Sec. 4.2). Solving problem (6) leads to the flux estimator α and its standard deviation $\widehat { \sigma } _ { \alpha }$ :

$$
\widehat { \alpha } = \frac { \sum _ { t } b _ { t } ( { \pmb x } _ { t } ) } { \sum _ { t } a _ { t } ( { \pmb x } _ { t } ) } , \quad \widehat { \sigma } _ { \alpha } = \frac { 1 } { \sqrt { \sum _ { t } a ( { \pmb x } _ { t } ) } } ,\tag{7}
$$

where

$$
b _ { t } ( \pmb { x } _ { t } ) = \sum _ { j \in S ( \pmb { x } _ { t } ) } w _ { j } \pmb { h } ^ { ( j ) } ( \pmb { x } _ { t } ) ^ { \top } \widehat { \mathbf { C } } _ { j } ^ { - 1 } ( \pmb { y } _ { t } ^ { ( j ) } - \widehat { \pmb { m } } _ { j } ) ,\tag{8}
$$

$$
a ( \pmb { x } _ { t } ) = \sum _ { j \in S ( \pmb { x } _ { t } ) } w _ { j } \pmb { h } ^ { ( j ) } ( \pmb { x } _ { t } ) ^ { \top } \widehat { \mathbf { C } } _ { j } ^ { - 1 } \pmb { h } ^ { ( j ) } ( \pmb { x } _ { t } ) .\tag{9}
$$

To assess the probability of an exoplanet at $\scriptstyle { \mathbf { { \mathit { x } } } } _ { 0 }$ , we use the generalized likelihood ratio test (GLRT) to statistically evaluate the parameter α. Under the null hypothesis $\mathcal { H } _ { \mathrm { 0 } }$ where $\alpha = 0$ (indicating no exoplanet), the statistics

$$
\widehat { \gamma } = \widehat { \alpha } / \widehat { \sigma } _ { \alpha } = \Big ( \sum _ { t } b _ { t } ( { \pmb x } _ { t } ) \Big ) \big / \big ( \sqrt { \sum _ { t } a ( { \pmb x } _ { t } ) } \big ) ,\tag{10}
$$

is controlled and follows a Gaussian distribution $\mathcal { N } ( 0 , 1 )$ When evaluating the probability of presence of an exoplanet, we test against the alternative hypothesis $\mathcal { H } _ { 1 } \colon \alpha > 0$ The statistics $\widehat { \gamma }$ can be directly mapped to a probability of detection. It can also be interpreted as the output signal-tonoise ratio through the statistical model, and in practice, this is the detection score used by astronomers [63]. To obtain the dense counterparts $\widehat { \gamma } , \widehat { \alpha } , \widehat { \sigma }$ in $\mathbb { R } ^ { H \times W }$ , we adopt a fast   approximation similar to [25] detailed in Appendix A.2.

Iterative procedure for characterization. Astrometry is the task of precisely estimating the flux α and the sub-pixel position $\scriptstyle { \mathbf { { \mathit { x } } } } _ { \mathrm { { 0 } } }$ of an exoplanet. This can be achieved by jointly optimizing the likelihood defined in Eq. (6). Denoting by ${ z } = [ \alpha , { \pmb x } _ { 0 } ]$ in $\mathbb { R } ^ { 3 }$ , the gradient g and Hessian matrix H of the neg-log-likelihood are decomposed as follows:

$$
g ( z ) = \sum _ { t } \sum _ { j \in S ( { \pmb x } _ { t } ) } w _ { j } { \pmb g } _ { j } ( z ) , { \pmb { \mathrm H } } ( z ) = \sum _ { t } \sum _ { j \in S ( { \pmb x } _ { t } ) } w _ { j } { \pmb { \mathrm H } } _ { j } ( z )
$$

where ${ \mathbf { \mu } } _ { { \mathbf { \mathcal { G } } } _ { j } }$ and $\mathbf { H } _ { j }$ represent the gradient and Hessian operators for each distribution $j ,$ respectively. These operators are responsible for updating the statistical parameters $m _ { j }$ and $\mathbf { C } _ { j } ,$ which are initially biased due to the presence of the exoplanet. Further details are provided in Sec. $\mathrm { A . 4 }$ . Each iteration indexed by l writes as:

$$
\pmb { z } ^ { ( l + 1 ) } = \pmb { z } ^ { ( l ) } - \mathbf { H } ( \pmb { z } ^ { ( l ) } ) ^ { - 1 } \pmb { g } ( \pmb { z } ^ { ( l ) } ) .\tag{11}
$$

## 4.2. Extensions of the statistical model: mixture of distributions

We now build on the convolutional statistical model presented in Sec. 4.1 to introduce a multi-scale statistical model of the nuisance component.

Linear projection. Instead of modeling correlations between the pixels of a patch, we propose to model correlations between projected linear features:

$$
\forall ( j , t ) , \quad \mathbf { A } y _ { t } ^ { ( j ) } \sim { \mathcal { N } } ( m _ { j } , \mathbf { C } _ { j } ) ,\tag{12}
$$

where A in $\mathbb { R } ^ { m \times p }$ is the projection matrix and $m \leq p .$ This is a generalization of the model presented in Sec. 4.1, for which $\mathbf { A } = \mathbf { I } _ { p }$ . The terms $b _ { t }$ and a introduced in Eqs. (26)- (27) become:

$$
b _ { i , t } = \sum _ { j \in S ( { \pmb x } _ { i } ) } w _ { j } h _ { j i } ^ { \top } { \bf A } _ { j } ^ { \top } \widehat { \bf C } _ { j } ^ { - 1 } \left( { \bf A } _ { j } { \pmb y } _ { t } ^ { ( j ) } - \widehat { { \pmb m } } _ { j } \right) ,\tag{13}
$$

$$
a _ { i } = \sum _ { j \in S ( \pmb { x } _ { i } ) } w _ { j } \pmb { h } _ { j i } ^ { \top } \pmb { \mathrm { A } } _ { j } ^ { \top } \widehat { \mathbf { C } } _ { j } ^ { - 1 } \mathbf { A } _ { j } \pmb { h } _ { j i } .\tag{14}
$$

The learnable projection A decorrelates the feature space, where the statistical distribution is defined, from the pixel space, enabling effective handling of long-range correlations.

Multi-scale approach. So far, we have shown how our model can combine multiple neighboring local distributions through weighted averaging of the terms $b _ { t }$ and $^ { a . }$ Extending this approach, we propose to integrate distributions across multiple spatial scales, by varying the patch sizes $p .$ We denote $\mathcal { P }$ the set of patch sizes chosen. We expand the set $S ( { \pmb x } _ { t } )$ of modeled distributions to include different scales, such that:

$$
\begin{array} { r } { S ( { \pmb x } _ { t } ) = \bigcup _ { p \in \mathcal { P } } S _ { p } ( { \pmb x } _ { t } ) , } \end{array}\tag{15}
$$

where $S _ { p } ( \pmb { x } _ { t } )$ is the subset of patches with patch size p containing $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ . The linear projection is particularly useful in controlling the dimensionality of the feature space for larger patches by setting $m \ < \ p ,$ , thereby maintaining computational efficiency.

Leveraging rotational symmetries. In direct imaging, the speckle pattern exhibits approximate central and rotational symmetries, see Fig. A. These symmetries arise from near-isotropic distortions of the wavefront and from the optical system’s near-circular symmetry. Symmetries of the speckles can be theoretically understood through a series expansion of the diffraction pattern, using increasing powers of the Fourier transform of the residual phase error [48, 53, 59]. Leveraging these symmetries is essential for constructing a more robust model of the speckles, particularly to mitigate the effects of self-subtraction. This effect is a well-known challenge in direct imaging that arises when limited parallactic rotation results in the contamination of speckle parameters by the exoplanet signal, ultimately reducing detection sensitivity [46]. To incorporate these symmetries, we propose modeling the joint distribution of patches extracted from the same location after rotating the observation by $2 \pi / N$ , also known as $N { \mathrm { - } } f o l d$ rotational symmetry:

$$
\forall ( j , t ) , \quad \mathbf { A } \big [ \mathbf { R } _ { 2 \pi n / N } ( { \boldsymbol { y } } _ { t } ) ^ { ( j ) } \big ] _ { n = 0 : N - 1 } \sim \mathcal { N } ( m _ { j } , \mathbf { C } _ { j } ) _ { j }\tag{16}
$$

where A is in $\mathbb { R } ^ { m \times N p }$ . The parameters of this joint distribution are less subject to self-subtraction as it is very unlikely to observe exoplanets simultaneously in all jointly modeled patches. We use a mixture model with $N = 1 , 2 , 4$

Joint multi-spectral modeling. In ASDI, the spectral channels are related by a homothety, as given in Eq. (2). We propose to leverage this relationship, and model:

$$
\forall ( j , c , t ) , \quad \mathbf { A } \beta _ { c , j } ^ { - 1 } \mathbf { D } _ { \lambda _ { 0 } / \lambda _ { c } } ( \pmb { y } _ { c , t } ) ^ { ( j ) } \sim \mathcal { N } ( m _ { j } , \mathbf { C } _ { j } ) .\tag{17}
$$

In this setting, the estimators of statistical parameters $\widehat { \pmb { m } } _ { j }$ and $\widehat { \mathbf { C } } _ { j }$ are less prone to self-subtraction, as the spectral diversity reduces the ambiguity between speckles and exoplanet signals. The estimator of the local amplitude $\widehat { \beta } _ { c , j }$ is computed as the standard deviation of pixel values across $\{ \mathbf { D } _ { \lambda _ { 0 } / \lambda _ { c } } ( \pmb { y } _ { c , t } ) ^ { ( j ) } \}$ t.

## 4.3. End-to-end trainable approach

Problem statement. We propose to combine our statistical model on the nuisance component with a learnable prior on the exoplanet signals. This approach can be formulated as an optimization problem:

$$
\widehat { \pmb { \alpha } } = \underset { { \pmb { \alpha } } \in \mathbb { R } ^ { H \times W } } { \arg \operatorname* { m i n } } \varphi _ { \theta } \big ( { \pmb { \alpha } } , \chi ( { \pmb { y } } ) \big ) + \psi _ { \nu } \big ( { \pmb { \alpha } } \big ) ,\tag{18}
$$

where $\varphi$ is the data-fitting term, corresponding to the negative log-likelihood of our statistical model of the nuisance component described in Sec. 4.1, and $\psi$ is a prior on exoplanet signals. We denote by $\theta , \nu$ the learnable parameters associated with these terms, and by $\chi ( y )$ the parameters estimated for each observation. In practice $\chi ( \pmb { y } ) =$ $\{ \widehat { m } _ { j } , \widehat { \mathbf { C } } _ { j } , \widehat { \beta } _ { c , j } \} , \theta = \{ \mathbf { A } _ { j } , w _ { j } \} _ { j }$ , and ν corresponds to the parameters of the neural network implementing ψ.

Detection by denoising. We propose a two-step approach to obtain the detection score corresponding to Eq. (18). First, we compute the detection score corresponding to the statistical model $\varphi ,$ which admits a fast approximation denoted by $\gamma ~ \in ~ \dot { \mathbb { R } } ^ { H \times W }$ in the following, and given by Eq. (28). We recall that under the null hypothesis, i.e., when no exoplanet is present, the elements of γ follow a Gaussian distribution $\mathcal { N } ( 0 , 1 )$ . Therefore, extracting the signals of exoplanets corresponds to denoising γ to remove this background noise. We propose to achieve this step using a neural network, such that:

$$
\widetilde { \gamma } = f _ { \nu } \left( \widehat { \gamma } \right) ,\tag{19}
$$

where $f _ { \nu }$ is the denoiser implemented by the neural network, and $\widetilde { \gamma }$ the final detection score.

Training objective. We suppose that the denoised detection score can be decomposed similarly to the GLRT form provided in Eq. (10) for the statistical model. This leads to $\widetilde { \gamma } = \widetilde { \alpha } / \widetilde { \sigma } _ { e } ,$ where $\widetilde { \alpha } , \widetilde { \sigma }$ in $\mathbb { R } ^ { H \times W }$ are the estimated flux     of the exoplanet for each pixel, and the standard deviation denoting the uncertainty associated with it. Additionally, we assume that the uncertainty $\widetilde { \pmb { \sigma } }$ remains unaffected by the neural network, as it is primarily driven by the high variability of the speckles already captured by the statistical model. Consequently, both components can be expressed as:

$$
\widetilde { \pmb { \alpha } } = f _ { \nu } ( \widehat { \pmb { \alpha } } / \widehat { \pmb { \sigma } } ) \times \widehat { \pmb { \sigma } } , \quad \widetilde { \pmb { \sigma } } = \widehat { \pmb { \sigma } } .\tag{20}
$$

This formulation yields a pixel-wise Gaussian distribution: $\mathcal { N } ( \widetilde { \pmb { \alpha } } , \widetilde { \pmb { \sigma } } )$ . The learnable parameters $\theta$ and $\nu$ of our model  are optimized by minimizing the negative log-likelihood between estimates (α, σ) and ground truth $\alpha _ { \mathrm { g t } }$

$$
\mathcal { L } ( \widetilde { \alpha } , \widetilde { \sigma } , \alpha _ { \mathrm { g t } } ) = 0 . 5 ( \widetilde { \alpha } - \alpha _ { \mathrm { g t } } ) ^ { 2 } / \widetilde { \sigma } ^ { 2 } + \log \widetilde { \sigma } .\tag{21}
$$

Model ensembling. To improve robustness, we combine outputs from multiple trained models. Specifically, given outputs $\widetilde { \alpha }$ and $\widetilde { \sigma } ,$ we can define $\widetilde { \mathbf { a } }$ and $\widetilde { b }$ such that $\widetilde { \pmb { \sigma } } =$ $1 / \sqrt { \widetilde { a } }$ and $\widetilde { \pmb { \alpha } } = \widetilde { \pmb { b } } / \widetilde { \pmb { a } }$ . For $Q$ models indexed by $q ,$ the out-  puts are aggregated as follows:

$$
\widetilde { a } _ { Q } = \sum _ { q } \widetilde { a } _ { q } / Q , \quad \widetilde { b } _ { Q } = \sum _ { q } \widetilde { b } _ { q } / Q .\tag{22}
$$

The combined detection score, $\widetilde { \gamma } _ { Q } = \widetilde { b } _ { Q } / \sqrt { \widetilde { a } _ { Q } } ,$ , represents  a weighted average of the individual detection scores.

Calibration. The trained model needs to be calibrated in order to relate the detection score to a probability of false alarm. We follow the procedure outlined in [5], and rely on a separate calibration dataset to estimate the cumulative distribution function of $\widetilde { \gamma }$ under the null hypothesis. Addi-tional details are provided in the Supplementary Material.

## 5. Experiments

## 5.1. Data, algorithms and evaluation protocol

Datasets. We use SPHERE datasets, a cutting-edge exoplanet finder instrument at the VLT [4]. Raw observations were sourced from the public data archive of the European Southern Observatory and calibrated with public tools [21, 47] from the High-Contrast Data Center. This results in science-ready 4-D datasets y (with $L = 2$ spectral channels, $T \in [ [ 1 5 , 3 0 0 ] ]$ temporal frames, and $H \times W = 1 0 2 4 ^ { 2 }$ pixels per image), along with the off-axis PSF h, wavelengths $\lambda _ { c } ,$ and parallactic angles $\phi _ { t }$ for algorithm input. Many algorithms have achieved optimal detection sensitivity far from the star, limited by photon noise [5, 28]. Thus, our analysis focuses on a smaller, star-centered region of $H = W = 2 5 6$ pixels, where detection sensitivity can still be significantly improved [5, 28].

<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=5>Spatial scales</td><td rowspan=1 colspan=3> $\mathrm { N } { = } 1$            $_ { \mathrm { N = 1 } , 2 }$         $\mathrm { N } { = } 1 , 2 , 4$ </td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=5> $8 \times 8$ </td><td rowspan=1 colspan=3> $0 . 5 5 4 \pm 0 . 0 0 5$    $0 . 5 7 1 \pm 0 . 0 0 5$    $0 . 5 7 5 \pm 0 . 0 0 5$ </td></tr><tr><td rowspan=2 colspan=1>ADI</td><td rowspan=2 colspan=2> $+ 1 6 \times<eq>+ 3 2 \times 3 2$ </td><td></td><td rowspan=1 colspan=2>1 6</eq></td><td rowspan=1 colspan=3> $0 . 5 6 1 \pm 0 . 0 0 5$    $0 . 5 7 3 \pm 0 . 0 0 4$    $0 . 5 7 9 \pm 0 . 0 0 5$ </td><td rowspan=1 colspan=1>16</td></tr><tr><td rowspan=1 colspan=3>2 × 32</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3> $0 . 5 6 7 \pm 0 . 0 0 5$    $0 . 5 7 5 \pm 0 . 0 0 5$    $0 . 5 8 0 \pm 0 . 0 0 5$ </td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=5> $+ 6 4 \times 6 4$ </td><td rowspan=1 colspan=3> $0 . 5 6 9 \pm 0 . 0 0 5$    $0 . 5 7 7 \pm 0 . 0 0 5$    $\mathbf { 0 . 5 8 1 \pm 0 . 0 0 5 }$ </td></tr><tr><td rowspan=3 colspan=1>ASDI</td><td rowspan=3 colspan=5> $8 \times 8$  $+ 1 6 \times 1 6$  $+ 3 2 \times 3 2$ </td><td rowspan=1 colspan=3> $0 . 7 1 3 \pm 0 . 0 0 5$    $0 . 7 1 9 \pm 0 . 0 0 5$    $0 . 7 2 3 \pm 0 . 0 0 5$ </td></tr><tr><td rowspan=1 colspan=3> $0 . 7 2 0 \pm 0 . 0 0 5$    $0 . 7 2 3 \pm 0 . 0 0 5$    $0 . 7 2 5 \pm 0 . 0 0 5$ </td></tr><tr><td rowspan=1 colspan=3>×32</td><td rowspan=1 colspan=3> $0 . 7 2 0 \pm 0 . 0 0 5$    $0 . 7 2 5 \pm 0 . 0 0 5$    ${ \bf 0 . 7 2 6 \pm 0 . 0 0 5 }$ </td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=5> $+ 6 4 \times 6 4$ </td><td rowspan=1 colspan=3> $0 . 7 2 0 \pm 0 . 0 0 5$    $0 . 7 2 5 \pm 0 . 0 0 4$    ${ \bf 0 . 7 2 6 \pm 0 . 0 0 4 }$ </td></tr></table>

Table A. Impact of multi-scale and N-fold rotational symmetries on detection performance (AUC) of our statistical model.

Training procedure. For training, we use 220 observations from the SHINE-F150 large survey of SPHERE [39]. For testing, we select 8 datasets representing typical diversity in observing conditions and instrumental settings (e.g., parallactic rotation amplitude $\Delta _ { \phi } ~ = ~ \vert \phi _ { T - 1 } - \phi _ { 0 } \vert )$ Five test datasets (on stars HD 159911, HD 216803, HD 206860, HD 188228, HD 102647) are used for benchmarking against state of the art methods and conducting a model ablation analysis. To evaluate performance, we simulate synthetic faint point-like sources mimicking exoplanet signatures and inject them into real data. This simulation pro cedure is common practice in direct imaging to ground the actual performance of algorithms because it is very realistic [7, 8, 16, 17, 28, 35]. Any real source indeed takes the form of the off-axis PSF, which is measured immediately before and after the main observation sequence by offsetting the star from the coronagraph. This simulation procedure is essential due to the lack of ground truth and the limited number of exoplanets (only a few dozen) detected by direct imaging to date. In addition, we rely on simulated sources for training our deep model, which is also a standard practice for learning-based techniques. Three datasets of star HR 8799, hosting three known exoplanets in the field of view [42, 43], are also used as full real data.

Baselines. For detection, we benchmark the proposed algorithms against methods from different classes described in Sect. 2. Selection criteria are (i) code availability (often limited in direct imaging), (ii) relevance, and (iii) widespread use. In this context, we include the cADI [41], KLIP/PCA [60] subtraction-based methods, as these are implemented in most data processing pipelines [13, 21, 62], have been responsible for detecting nearly all imaged exoplanets –including the most recent discoveries [45, 67]– and remain heavily utilized by astronomers. Concerning statistical methods, we focus on PACO [25, 26] that uniquely accounts for data correlations. PACO has consistently outperformed KLIP/PCA in large observational surveys [12, 22] and achieved state of the art performance on SPHERE data in a community benchmark, surpassing various subtraction-based, statistical, and learning-based approaches [7]. We also evaluate the proposed approach against MODEL&CO [5], a recent hybrid method that shown to outperform cADI, KLIP/PCA, and (deep) PACO [5]. For cADI and KLIP/PCA, we use the VIP Python package [13, 33], fine-tuning parameters (e.g., PCA modes) to optimize detection scores guided by the ground truth. PACO and MODEL&CO were processed by their authors using data-driven settings. For flux estimation, we compare only with PACO due to computational constraints. We conduct all analyses in both ADI and ASDI modes (where supported) to assess the gains from joint spectral processing.

<table><tr><td>Method</td><td>flux error (ARE)</td><td>position error (RMSE)</td></tr><tr><td>PACO</td><td>0.56</td><td>0.21</td></tr><tr><td>Proposed</td><td>0.51</td><td>0.11</td></tr></table>

Table B. Comparison of statistical models for flux estimation.

Metrics. The detection metric used is the area under the receiver operating characteristic curve (AUC), representing the true positive rate against the false discovery rate obtained by varying the detection threshold. Higher AUC values indicate better performance. This standard metric in direct imaging [7, 25, 34] captures the precision-recall tradeoff and allows fair algorithm comparisons, as a common threshold does not ensure consistent false alarm rates due to the lack of statistical grounding in some detection maps, see Sec. 1. The primary metric is the absolute relative error (ARE) between the ground truth and estimated flux, with lower values indicating better performance. We also report the root mean square error (RMSE) for sub-pixel localization of exoplanets.

## 5.2. Quantitative and qualitative evaluations

Statistical model. We evaluate the impact of leveraging multiple spatial scales and symmetries in our statistical model by testing its detection performance across configurations listed in Table A. Results show that using both symmetries and scales is crucial in ADI, where statistical parameters suffer from self-subtraction without the robustness of added spectral diversity. We then evaluate the statistical model’s performance in estimating flux and sub-pixel position (regression tasks) using the optimization process from

<table><tr><td>Modality</td><td>Method</td><td>HD159911 (54°)</td><td> $\mathrm { H D } 2 1 6 8 0 3 ( 2 3 ^ { \circ } )$ </td><td> $\mathrm { H D } 2 0 6 8 6 0 ( 1 1 ^ { \circ } )$ </td><td> $\mathrm { H D 1 8 8 2 2 8 ( 6 ^ { \circ } ) }$ </td><td> $\mathrm { H D 1 0 2 6 4 7 } \left( 2 ^ { \circ } \right)$ </td><td> $\mathrm { a v e r a g e \ A U C }$ </td></tr><tr><td rowspan="5">ADI</td><td>cADI</td><td> $0 . 2 8 8 \pm 0 . 0 1 4$ </td><td> $0 . 4 2 2 \pm 0 . 0 0 7$ </td><td> $0 . 4 8 9 \pm 0 . 0 1 0$ </td><td> $0 . 3 0 3 \pm 0 . 0 1 4$ </td><td> $0 . 3 4 3 \pm 0 . 0 0 9$ </td><td> $0 . 3 6 9 \pm 0 . 0 0 5$ </td></tr><tr><td>PCA</td><td> $0 . 6 3 4 \pm 0 . 0 1 0$ </td><td> $0 . 6 4 3 \pm 0 . 0 1 1$ </td><td> $0 . 5 0 5 \pm 0 . 0 0 9$ </td><td> $0 . 3 9 2 \pm 0 . 0 1 1$ </td><td> $0 . 2 1 8 \pm 0 . 0 1 1$ </td><td> $0 . 4 7 8 \pm 0 . 0 0 5$ </td></tr><tr><td>PACO</td><td> $0 . 6 2 9 \pm 0 . 0 0 6$ </td><td> $0 . 6 6 9 \pm 0 . 0 1 2$ </td><td> $0 . 5 7 9 \pm 0 . 0 1 5$ </td><td> $0 . 5 1 7 \pm 0 . 0 1 5$ </td><td> $0 . 2 0 7 \pm 0 . 0 1 2$ </td><td> $0 . 5 2 0 \pm 0 . 0 0 6$ </td></tr><tr><td>MODEL&amp;CO</td><td> $0 . 6 5 3 \pm 0 . 0 1 0$ </td><td> $0 . 7 3 1 \pm 0 . 0 1 1$ </td><td> $0 . 6 4 6 \pm 0 . 0 1 2$ </td><td> $\mathbf { 0 . 6 3 8 \pm 0 . 0 1 4 }$ </td><td> $\mathbf { 0 . 5 5 4 \pm 0 . 0 0 9 }$ </td><td> $\mathbf { 0 . 6 4 5 \pm 0 . 0 0 5 }$ </td></tr><tr><td>Proposed</td><td> $\mathbf { 0 . 6 7 3 \pm 0 . 0 0 9 }$ </td><td> $\mathbf { 0 . 7 4 0 \pm 0 . 0 1 0 }$ </td><td> $\mathbf { 0 . 6 6 1 \pm 0 . 0 1 2 }$ </td><td> $0 . 6 3 1 \pm 0 . 0 1 3$ </td><td> $0 . 5 1 8 \pm 0 . 0 1 2$ </td><td> $\mathbf { 0 . 6 4 5 \pm 0 . 0 0 5 }$ </td></tr><tr><td rowspan="4">ASDI</td><td>cASDI</td><td> $0 . 4 4 8 \pm 0 . 0 0 9$ </td><td> $0 . 5 3 7 \pm 0 . 0 1 3$ </td><td> $0 . 4 5 1 \pm 0 . 0 0 7$ </td><td> $0 . 3 5 2 \pm 0 . 0 1 7$ </td><td> $0 . 2 9 4 \pm 0 . 0 1 2$ </td><td> $0 . 4 1 7 \pm 0 . 0 0 6$ </td></tr><tr><td>PCA</td><td> $0 . 6 9 4 \pm 0 . 0 1 3$ </td><td> $0 . 6 9 6 \pm 0 . 0 0 7$ </td><td> $0 . 5 5 2 \pm 0 . 0 1 1$ </td><td> $0 . 3 9 8 \pm 0 . 0 1 1$ </td><td> $0 . 2 3 6 \pm 0 . 0 0 9$ </td><td> $0 . 5 1 5 \pm 0 . 0 0 5$ </td></tr><tr><td>PACO</td><td> $0 . 6 9 8 \pm 0 . 0 1 5$ </td><td> $0 . 7 6 8 \pm 0 . 0 0 8$ </td><td> $0 . 7 1 0 \pm 0 . 0 1 4$ </td><td> $0 . 7 0 0 \pm 0 . 0 0 9$ </td><td> $0 . 5 8 9 \pm 0 . 0 1 4$ </td><td> $0 . 6 9 3 \pm 0 . 0 0 5$ </td></tr><tr><td>Proposed</td><td> $\mathbf { 0 . 7 3 1 \pm 0 . 0 0 9 }$ </td><td> $\mathbf { 0 . 8 0 4 \pm 0 . 0 1 3 }$ </td><td> $\mathbf { 0 . 7 4 7 \pm 0 . 0 1 0 }$ </td><td> $\mathbf { 0 . 7 8 2 \pm 0 . 0 0 8 }$ </td><td> $\mathbf { 0 . 7 4 4 } \pm 0 . 0 0 6$ </td><td> $\mathbf { 0 . 7 6 1 \pm 0 . 0 0 4 }$ </td></tr></table>

Table C. Comparative detection scores (AUC) in A(S)DI modes. Dataset names (stars) and parallactic amplitude $\Delta _ { \phi }$ are reported on top. ASDI mode (which is better than ADI) is not supported by MODEL&CO.

![](images/18527df0ef921a06302885c7660a1e4121c6fe788f25bd4d9f02d62c5b512c29.jpg)  
Figure D. Detection maps on observations of HD 159911 star with synthetic exoplanets. The (calibrated) detection threshold is equivalent for all methods. The proposed approach here detect 1 additional source compared to the second best method (PACO ASDI).

![](images/d1a3a3d930188d2ff364ea3c1512da1c8b14df92c58c2d569d4a7c82c3426071.jpg)  
Figure E. Detection maps on 3 observations (stacked in false RGB colors) of HR 8799 star. The elliptical arcs depict the estimated (projected) orbits of three known exoplanets, with the detection results shown as red, green and blue dots for the corresponding 2016, 2018 and 2021 observations. Squares are for false alarms identified in [5], Fig. 17.

Sec. 4.1. Table B shows average errors across 1,351 synthetic exoplanets detectable by both considered methods. Our approach outperforming others in almost all cases.

End-to-end learnable model. We evaluate detection performance on 5 observations using synthetic exoplanet injections through direct models (1)-(2). For each observation, 100 cubes are generated, totaling 1,000 injected exoplanets. This process is repeated 5 times with different random seeds, and AUC scores are reported in Table C. Examples of representative detection maps are shown in Fig. K. The proposed approach matches or outperforms state of the art algorithms in ADI mode. In ASDI mode, it shows a significant boost in detection sensitivity, consistently surpassing comparative methods.

Validation on real data. We compare detection methods using 3 observations, spanning several years, of the HR

8799 star. The stacked detection maps, highlighting exoplanets’ orbital motion, are shown in Fig. E. All detection maps use a consistent unit (signal-to-noise score) and dynamic range. The proposed method maximizes detection confidence with no false alarms.

## 6. Conclusion

We propose a novel hybrid approach for exoplanet imaging that combines a multi-scale statistical model with deep learning, capturing spatial correlations in the nuisance component for simultaneous detection and flux estimation. This approach provides statistically grounded detection scores, unbiased estimates, and native uncertainty quantification. Tested on VLT/SPHERE data, it outperforms state-of-theart techniques, demonstrating efficiency and robustness across varied data qualities. Its versatility and reduced computational complexity make it ideal for large-scale surveys. The approach will be extended to handle higher spectral resolution data. Additionally, the nuisance model can be adapted for reconstructing spatially extended objects like circumstellar disks, namely birthplaces of exoplanets.

## Acknowledgments

This work was supported by the French government under management of Agence Nationale de la Recherche as part of the “France 2030” program, PR[AI]RIE-PSAI projet (reference ANR-23-IACL-0008) and MIAI 3IA Institute (reference ANR-19-P3IA-0003), and by the European Research Council (ERC) under grant agreement 101087696 (APHELEIA project). This work was also supported by the ERC under the European Union’s Horizon 2020 research and innovation programme (COBREX; grant agreement 885593), the ANR under the France 2030 program (PEPR Origins, reference ANR-22-EXOR-0016), the French National Programs (PNP and PNPS), and the Action Specifique Haute R´ esolution Angulaire (ASHRA)´ of CNRS/INSU co-funded by CNES. This work was granted access to the HPC resources of IDRIS under the allocation 2022-AD011013643 made by GENCI. JP was supported in part by the Louis Vuitton/ENS chair in artificial intelligence and a Global Distinguished Professorship at the Courant Institute of Mathematical Sciences and the Center for Data Science at New York University.

## References

[1] Michal Aharon, Michael Elad, and Alfred Bruckstein. K-SVD: An algorithm for designing overcomplete dictionaries for sparse representation. IEEE Transactions on signal processing, 54(11):4311–4322, 2006. 2

[2] France Allard, Nicole F Allard, Derek Homeier, John Kielkopf, Mark J McCaughrean, and Fernand Spiegelman. K-H2 quasi-molecular absorption detected in the T-dwarf Indi Ba. Astronomy & Astrophysics, 474(2):L21–L24, 2007. 1

[3] Adam Amara and Sascha P Quanz. Pynpoint: an image processing package for finding exoplanets. Monthly Notices of the Royal Astronomical Society, 427(2):948–955, 2012. 2

[4] Jean-Luc Beuzit, Arthur Vigan, David Mouillet, Kjetil Dohlen, Raffaele Gratton, Anthony Boccaletti, J-F Sauvage, Hans Martin Schmid, Maud Langlois, Cyril Petit, et al. SPHERE: the exoplanet imager for the Very Large Telescope. Astronomy & Astrophysics, 631:A155, 2019. 2, 6

[5] Theo Bodrito, Olivier Flasseur, Julien Mairal, Jean Ponce,´ Maud Langlois, and Anne-Marie Lagrange. Model&co: Exoplanet detection in angular differential imaging by learning across multiple observations. Monthly Notices of the Royal Astronomical Society, 534(2):1569–1596, 2024. 3, 6, 7, 8, 2

[6] F Cantalloube, D Mouillet, LM Mugnier, J Milli, Olivier Absil, CA Gomez Gonzalez, G Chauvin, J-L Beuzit, and A Cornia. Direct exoplanet detection and characterization using the andromeda method: Performance on vlt/naco data. Astronomy & Astrophysics, 582:A89, 2015. 2

[7] Faustine Cantalloube, Carlos Gomez-Gonzalez, Olivier Absil, Carles Cantero, Regis Bacher, MJ Bonse, Michael Bottom, C-H Dahlqvist, Celia Desgrange, Olivier Flasseur, et al.´ Exoplanet imaging data challenge: benchmarking the various image processing methods for exoplanet detection. In

Adaptive Optics Systems VII, pages 1027–1062. SPIE, 2020. 3, 7

[8] Carles Cantero, Olivier Absil, C-H Dahlqvist, and Marc Van Droogenbroeck. Na-sodinn: A deep learning algo rithm for exoplanet image detection based on residual noise regimes. Astronomy & Astrophysics, 680:A86, 2023. 3, 7

[9] Gilles Chabrier, Isabelle Baraffe, France Allard, and P Hauschildt. Evolutionary models for very low-mass stars and brown dwarfs with dusty atmospheres. The Astrophysical Journal, 542(1):464, 2000. 1

[10] Gael Chauvin. Direct imaging of exoplanets at the¨ era of the extremely large telescopes. arXiv preprint arXiv:1810.02031, 2018. 1

[11] Pattana Chintarungruangchai, Guey Jiang, Jun Hashimoto, Yu Komatsu, and Mihoko Konishi. A possible converter to denoise the images of exoplanet candidates through machine learning techniques. New Astronomy, 100:101997, 2023. 3

[12] A Chomez, A-M Lagrange, P Delorme, M Langlois, G Chauvin, O Flasseur, J Dallant, F Philipot, S Bergeon, D Albert, et al. Preparation for an unsupervised massive analysis of sphere high-contrast data with paco-optimization and benchmarking on 24 solar-type stars. Astronomy & Astrophysics, 675:A205, 2023. 7, 2

[13] Valentin Christiaens, Carlos Gonzalez, Ralf Farkas, Carl-Henrik Dahlqvist, Evert Nasedkin, Julien Milli, Olivier Absil, Henry Ngo, Carles Cantero, Alan Rainot, et al. Vip: A python package for high-contrast imaging. Journal of Open Source Software, 8, 2023. 7

[14] Thayne Currie, Beth Biller, Anne-Marie Lagrange, Christian Marois, Olivier Guyon, Eric Nielsen, Mickael Bonnefoy, and Robert De Rosa. Direct imaging and spectroscopy of extrasolar planets. arXiv preprint arXiv:2205.05696, 2022. 1

[15] Kostadin Dabov, Alessandro Foi, Vladimir Katkovnik, and Karen Egiazarian. Image denoising by sparse 3-d transformdomain collaborative filtering. IEEE Transactions on image processing, 16(8):2080–2095, 2007. 2

[16] Hazan Daglayan, Simon Vary, Faustine Cantalloube, P-A Absil, and Olivier Absil. Likelihood ratio map for direct exoplanet detection. In 2022 IEEE 5th International Confer ence on Image Processing Applications and Systems (IPAS), pages 1–5. IEEE, 2022. 7

[17] C-H Dahlqvist, Faustine Cantalloube, and Olivier Absil. Regime-switching model detection map for direct exoplanet detection in adi sequences. Astronomy & Astrophysics, 633: A95, 2020. 2, 7

[18] C-H Dahlqvist, Faustine Cantalloube, and Olivier Absil. Auto-rsm: An automated parameter-selection algorithm for the rsm map exoplanet detection algorithm. Astronomy & Astrophysics, 656:A54, 2021.

[19] C-H Dahlqvist, Gilles Louppe, and Olivier Absil. Improving the rsm map exoplanet detection algorithm-psf forward modelling and optimal selection of psf subtraction techniques. Astronomy & Astrophysics, 646:A49, 2021. 2

[20] Richard Davies and Markus Kasper. Adaptive optics for astronomy. Annual Review of Astronomy and Astrophysics, 50: 305–351, 2012. 2

[21] Ph Delorme, Nadege Meunier, D Albert, Eric Lagadec, H Le \` Coroller, R Galicher, D Mouillet, A Boccaletti, DINO Mesa,

J-C Meunier, et al. The SPHERE Data Center: a reference for high contrast imaging processing. arXiv preprint arXiv:1712.06948, 2017. 6, 7

[22] P Delorme, A Chomez, V Squicciarini, M Janson, O Flasseur, O Schib, R Gratton, AM Lagrange, M Langlois, L Mayer, et al. Giant planets population around b stars from the first part of the beast survey. arXiv preprint arXiv:2409.18793, 2024. 7

[23] Rob Fergus, David W Hogg, Rebecca Oppenheimer, Douglas Brenner, and Laurent Pueyo. S4: A spatial-spectral model for speckle suppression. The Astrophysical Journal, 794(2):161, 2014. 3

[24] Michael P Fitzgerald and James R Graham. Speckle statistics in adaptively corrected images. The Astrophysical Journal, 637(1):541, 2006. 2

[25] Olivier Flasseur, Lo¨ıc Denis, Eric Thi<sup>´</sup> ebaut, and Maud Lan-´ glois. Exoplanet detection in angular differential imaging by statistical learning of the nonstationary patch covariancesthe paco algorithm. Astronomy & Astrophysics, 618:A138, 2018. 2, 3, 5, 7

[26] Olivier Flasseur, Lo¨ıc Denis, Eric Thi<sup>´</sup> ebaut, and Maud Lan-´ glois. Paco asdi: an algorithm for exoplanet detection and characterization in direct imaging with integral field spectrographs. Astronomy & Astrophysics, 637:A9, 2020. 2, 3, 7

[27] Olivier Flasseur, Theo Bodrito, Julien Mairal, Jean Ponce,´ Maud Langlois, and Anne-Marie Lagrangev. Combining multi-spectral data with statistical and deep-learning models for improved exoplanet detection in direct imaging at high contrast. In 2023 31st European Signal Processing Conference (EUSIPCO), pages 1723–1727. IEEE, 2023. 3

[28] Olivier Flasseur, Theo Bodrito, Julien Mairal, Jean Ponce,´ Maud Langlois, and Anne-Marie Lagrange. deep paco: Combining statistical models with deep learning for exoplanet detection and characterization in direct imaging at high contrast. Monthly Notices of the Royal Astronomical Society, 527(1):1534–1562, 2024. 3, 7

[29] Olivier Flasseur, Eric Thiebaut, Lo´ ¨ıc Denis, and Maud Langlois. Shrinkage mmse estimators of covariances beyond the zero-mean and stationary variance assumptions. In 2024 32nd European Signal Processing Conference (EUSIPCO), pages 2727–2731, 2024. 1

[30] Katherine B Follette. An introduction to high contrast differential imaging of exoplanets and disks. Publications of the Astronomical Society of the Pacific, 135(1051):093001, 2023. 1

[31] Timothy D Gebhard, Markus J Bonse, Sascha P Quanz, and Bernhard Scholkopf. Half-sibling regression meets ex-¨ oplanet imaging: Psf modeling and subtraction using a flexible, domain knowledge-driven, causal framework. Astronomy & Astrophysics, 666:A9, 2022. 3

[32] Benjamin L Gerard and Christian Marois. Planet detection down to a few λ/d: an rsdi/tloci approach to psf subtraction. In Adaptive Optics Systems V, pages 1544–1556. SPIE, 2016. 2

[33] Carlos Alberto Gomez Gonzalez, Olivier Wertz, Olivier Absil, Valentin Christiaens, Denis Defrere, Dimitri Mawet,\` Julien Milli, Pierre-Antoine Absil, Marc Van Droogen-

broeck, Faustine Cantalloube, et al. Vip: Vortex image processing package for high-contrast direct imaging. The Astronomical Journal, 154(1):7, 2017. 2, 7

[34] CA Gomez Gonzalez, Olivier Absil, P-A Absil, Marc Van Droogenbroeck, Dimitri Mawet, and Jean Surdej. Lowrank plus sparse decomposition for exoplanet detection in direct-imaging adi sequences-the llsg algorithm. Astronomy & Astrophysics, 589:A54, 2016. 2, 7

[35] CA Gomez Gonzalez, Olivier Absil, and Marc Van Droogenbroeck. Supervised detection of exoplanets in high-contrast imaging sequences. Astronomy & Astrophysics, 613:A71, 2018. 3, 7

[36] SY Haffert, AJ Bohn, J de Boer, IAG Snellen, J Brinchmann, JH Girard, CU Keller, and R Bacon. Two accreting protoplanets around the young star PDS 70. Nature Astronomy, 3 (8):749–754, 2019. 1

[37] M Keppler, M Benisty, A Muller, Th Henning, R Van Boekel,¨ F Cantalloube, C Ginski, RG Van Holstein, A-L Maire, A Pohl, et al. Discovery of a planetary-mass companion within the gap of the transition disk around pds 70. Astronomy & Astrophysics, 617:A44, 2018. 1

[38] A-M Lagrange, D Gratadour, G Chauvin, T Fusco, D Ehrenreich, D Mouillet, G Rousset, D Rouan, F Allard, E Gen-<sup>´</sup> dron, et al. A probable giant planet imaged in the β Pictoris disk: VLT/NaCo deep L’-band imaging. Astronomy & As trophysics, 493(2):L21–L25, 2009. 2

[39] Maud Langlois, R Gratton, A-M Lagrange, P Delorme, A Boccaletti, M Bonnefoy, A-L Maire, D Mesa, G Chauvin, S Desidera, et al. The sphere infrared survey for exoplanets (shine)-ii. observations, data reduction and analysis, detection performances, and initial results. Astronomy & Astrophysics, 651:A71, 2021. 7

[40] Julien Mairal, Francis Bach, Jean Ponce, and Guillermo Sapiro. Online dictionary learning for sparse coding. In Proceedings of the 26th annual international conference on machine learning, pages 689–696, 2009. 2

[41] Christian Marois, David Lafreniere, Rene Doyon, Bruce ´ Macintosh, and Daniel Nadeau. Angular differential imaging: a powerful high-contrast imaging technique. The Astro physical Journal, 641(1):556, 2006. 2, 7

[42] Christian Marois, Bruce Macintosh, Travis Barman, B Zuckerman, Inseok Song, Jennifer Patience, David Lafreniere,\` and Rene Doyon. Direct imaging of multiple planets orbiting´ the star HR 8799. Science, 322(5906):1348–1352, 2008. 7

[43] Christian Marois, B Zuckerman, Quinn M Konopacky, Bruce Macintosh, and Travis Barman. Images of a fourth planet orbiting hr 8799. Nature, 468(7327):1080, 2010. 7

[44] Christian Marois, Carlos Correia, Raphael Galicher, Patrick Ingraham, Bruce Macintosh, Thayne Currie, and Rob De Rosa. GPI PSF subtraction with TLOCI: the next evolution in exoplanet/disk high-contrast imaging. In SPIE Astronomical Intrumentation + Telescopes, page 91480U. International Society for Optics and Photonics, 2014. 2

[45] D Mesa, R Gratton, P Kervella, M Bonavita, S Desidera, V D’Orazi, S Marino, A Zurlo, and E Rigliaco. Af lep b: The lowest-mass planet detected by coupling astrometric and direct imaging data. Astronomy & Astrophysics, 672:A93, 2023. 7

[46] J Milli, D Mouillet, A-M Lagrange, A Boccaletti, D Mawet, G Chauvin, and M Bonnefoy. Impact of angular differential imaging on circumstellar disk images. Astronomy & Astrophysics, 545:A111, 2012. 6

[47] Alexey Pavlov, Ole Moller-Nilsson, Markus Feldt, Thomas¨ Henning, Jean-Luc Beuzit, and David Mouillet. Sphere data reduction and handling system: overview, project status, and development. Advanced Software and Controlfor Astronomy II, 7019:1093–1104, 2008. 6

[48] Marshall D Perrin, Anand Sivaramakrishnan, Russell B Makidon, Ben R Oppenheimer, and James R Graham. The structure of high strehl ratio point-spread functions. The Astrophysical Journal, 596(1):702, 2003. 5

[49] Sai Krishanth PM, Ewan S Douglas, Justin Hom, Ramya M Anche, John Debes, Isabel Rebollido, and Bin B Ren. Nmfbased gpu accelerated coronagraphy pipeline. In Techniques and Instrumentation for Detection of Exoplanets XI, pages 668–679. SPIE, 2023. 2

[50] Laurent Pueyo. Direct imaging as a detection technique for exoplanets. Handbook of Exoplanets, pages 705–765, 2018. 2

[51] Bin Ren, Laurent Pueyo, Guangtun Ben Zhu, John Debes, and Gaspard Duchene. Non-negative matrix factorization:ˆ robust extraction of extended structures. The Astrophysical Journal, 852(2):104, 2018. 2

[52] Bin Ren, Laurent Pueyo, Christine Chen, Elodie Choquet, <sup>´</sup> John H Debes, Gaspard Duchene, Francˆ ¸ois Menard, and´ Marshall D Perrin. Using data imputation for signal separation in high-contrast imaging. The Astrophysical Journal, 892(2):74, 2020. 2

[53] Erez N Ribak and Szymon Gladysz. Fainter and closer: finding planets by symmetry breaking. Optics Express, 16(20): 15553–15562, 2008. 5

[54] Jean-Baptiste Ruffio, Bruce Macintosh, Jason J Wang, Laurent Pueyo, Eric L Nielsen, Robert J De Rosa, Ian Czekala, Mark S Marley, Pauline Arriaga, Vanessa P Bailey, et al. Improving and assessing planet sensitivity of the gpi exoplanet survey with a forward model matched filter. The Astrophysical Journal, 842(1):14, 2017. 2

[55] Matthias Samland, J Bouwman, DW Hogg, W Brandner, T Henning, and Markus Janson. Trap: A temporal systematics model for improved direct detection of exoplanets at small angular separations. Astronomy & Astrophysics, 646:A24, 2021. 3

[56] Aniket Sanghi, Jerry W Xuan, Jason J Wang, et al. Efficiently searching for close-in companions around young m dwarfs using a multiyear psf library. The Astronomical Journal, 168(5):215, 2024. 3

[57] Nuno C Santos. Extra-solar planets: Detection methods and results. New Astronomy Reviews, 52(2-5):154–166, 2008. 1

[58] Remi Soummer. Apodized pupil lyot coronagraphs for arbi- ´ trary telescope apertures. The Astrophysical Journal Letters, 618(2):L161, 2004. 2

[59] Remi Soummer, Andr ´ e Ferrari, Claude Aime, and Laurent ´ Jolissaint. Speckle noise and dynamic range in coronagraphic images. The Astrophysical Journal, 669(1):642, 2007. 5

[60] Remi Soummer, Laurent Pueyo, and James Larkin. Detec-´ tion and characterization of exoplanets and disks using projections on karhunen–loeve eigenimages.\` The Astrophysical Journal Letters, 755(2):L28, 2012. 2, 7

[61] William B Sparks and Holland C Ford. Imaging spectroscopy for extrasolar planet detection. The Astrophysical Journal, 578(1):543, 2002. 2

[62] Tomas Stolker, Markus J Bonse, Sascha P Quanz, Adam Amara, Gabriele Cugno, Alexander J Bohn, and Anna Boehle. Pynpoint: a modular pipeline architecture for processing and analysis of high-contrast imaging data. Astron omy & Astrophysics, 621:A59, 2019. 7

[63] Eric Thi<sup>´</sup> ebaut, Lo´ ¨ıc Denis, Laurent Mugnier, Andre Fer-´ rari, David Mary, Maud Langlois, Faustine Cantalloube, and Nicholas Devaney. Fast and robust exo-planet detection in multi-spectral, multi-temporal data. In Adaptive Optics Systems V, pages 1534–1543. SPIE, 2016. 5

[64] William Thompson and Christian Marois. Improved contrast in images of exoplanets using direct signal-to-noise ratio op timization. The Astronomical Journal, 161(5):236, 2021. 2

[65] Wesley A Traub and Ben R Oppenheimer. Direct imaging of exoplanets. Exoplanets, pages 111–156, 2010. 1

[66] Arthur Vigan, Claire Moutou, Maud Langlois, France Allard, Anthony Boccaletti, Marcel Carbillet, David Mouillet, and Isabelle Smith. Photometric characterization of exoplan ets using angular and spectral differential imaging. Monthly Notices of the Royal Astronomical Society, 407(1):71–82, 2010. 1, 2

[67] Kevin Wagner, Jordan Stone, Andrew Skemer, Steve Ertel, Ruobing Dong, Daniel Apai, Eckhart Spalding, Jarron ´ Leisenring, Michael Sitko, Kaitlin Kratter, et al. Direct images and spectroscopy of a giant protoplanet driving spiral arms in mwc 758. Nature Astronomy, 7(10):1208–1217, 2023. 7

[68] Zahed Wahhaj, Lucas A Cieza, Dimitri Mawet, Bin Yang, Hector Canovas, Jozua de Boer, Simon Casassus, Franc¸ois Menard, Matthias R Schreiber, Michael C Liu, et al. Im-´ proving signal-to-noise in the direct imaging of exoplanets and circumstellar disks with MLOCI. Astronomy & Astrophysics, 581:A24, 2015. 2

[69] Trevor N Wolf, Brandon A Jones, and Brendan P Bowler. Direct exoplanet detection using convolutional image reconstruction (construct): A new algorithm for post-processing high-contrast images. The Astronomical Journal, 167(3):92, 2024. 3

[70] Kai Hou Yip, Nikolaos Nikolaou, Piero Coronica, Angelos Tsiaras, et al. Pushing the limits of exoplanet discovery via direct imaging with deep learning. In Machine Learning and Knowledge Discovery in Databases: European Conf., ECML PKDD 2019, Wurzburg, Germany, September 16–20, 2019,¨ Proceedings, Part III, pages 322–338. Springer, 2020. 3

[71] Kai Yu, Yuanqing Lin, and John Lafferty. Learning image representations from the pixel level via hierarchical sparse coding. In CVPR 2011, pages 1713–1720. IEEE, 2011. 2

[72] Daniel Zoran and Yair Weiss. From learning models of natural image patches to whole image restoration. In 2011 international conference on computer vision, pages 479–486. IEEE, 2011. 2