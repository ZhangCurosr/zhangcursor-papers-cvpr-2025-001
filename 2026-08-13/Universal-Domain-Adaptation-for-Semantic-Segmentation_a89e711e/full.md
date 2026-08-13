# Universal Domain Adaptation for Semantic Segmentation

Seun-An Choe<sup>1</sup>

Keon-Hee Park<sup>1</sup>

Jinwoo Choi<sup>1</sup>

Gyeong-Moon Park<sup>2†</sup>

<sup>1</sup>Kyung Hee University, Yongin, Republic of Korea <sup>2</sup>Korea University, Seoul, Republic of Korea

<sup>1</sup>{dragoon0905, pgh2874, jinwoochoi}@khu.ac.kr <sup>2</sup>gm-park@korea.ac.kr

## Abstract

Unsupervised domain adaptation for semantic segmentation (UDA-SS) aims to transfer knowledge from labeled source data to unlabeled target data. However, traditional UDA-SS methods assume that category settings between source and target domains are known, which is unrealistic in real-world scenarios. This leads to performance degradation if private private classes exist. To address this limitation, we propose Universal Domain Adaptation for Semantic Segmentation (UniDA-SS), achieving robust adaptation even without prior knowledge of category settings. We define the problem in the UniDA-SS scenario as low confidence scores of common classes in the target domain, which leads to confusion with private classes. To solve this problem, we propose UniMAP: UniDA-SS with Image Matching and Prototype-based Distinction, a novel framework composed of two key components. First, Domain-Specific Prototype-based Distinction (DSPD) divides each class into two domain-specific prototypes, enabling finer separation of domain-specific features and enhancing the identification of common classes across domains. Second, Target-based Image Matching (TIM) selects a source image containing the most common-class pixels based on the target pseudo-label and pairs it in a batch to promote effective learning of common classes. We also introduce a new UniDA-SS benchmark and demonstrate through various experiments that UniMAP significantly outperforms baselines. The code is available at https://github. com/KU-VGI/UniMAP.

## 1. Introduction

Semantic segmentation is a fundamental computer vision task that predicts the class of each pixel in an image and is essential in fields like autonomous driving, medical imaging, and human-machine interaction. However, training segmentation models requires pixel-level annotations, which are costly and time-consuming. To address this, researchers have explored Unsupervised Domain Adaptation for Semantic Segmentation (UDA-SS) methods, which aim to learn domain-invariant representations from labeled synthetic data (source) to unlabeled real-world data (target).

![](images/078bc5fe29699834d4495e527c678ee3b2a6148b2034508c84e16d1eb67f4b4e.jpg)  
Figure 1. Visualization results of the UDA-SS models across different scenarios. We select MIC and BUS, which achieve the best performance in CDA-SS and ODA-SS, respectively, and visualize their results in PDA-SS and OPDA-SS. The images illustrate the performance degradation caused by the introduction of sourceprivate classes.

While UDA-SS has shown effectiveness in addressing domain shift, existing UDA-SS methods rely on the assumption that source and target categories are known in advance. This assumption is often impractical in real-world scenarios, as target labels are typically unavailable. As a result, the target domain frequently includes unseen classes that are absent in the source domain (target-private classes), or conversely, the source domain may contain classes not found in the target domain (source-private classes). This limitation can lead to negative transfer, where models incorrectly align source-private classes with the target domain, resulting in significant performance degradation. To address these challenges, we introduce a new Universal Domain Adaptation for Semantic Segmentation (UniDA-SS) scenario, enabling adaptation without prior knowledge of category configurations and classifying target samples as “unknown” if they contain target-private classes.

To understand the challenges posed by the UniDA-SS scenario, we first evaluate the performance of existing UDA-SS methods under various domain adaptation settings. Figure 1 presents qualitative results of UDA-SS methods across various scenarios. Specifically, we select MIC [15] and BUS [6] as representative models for Closed Set Domain Adaptation (CDA-SS) and Open Set Domain Adaptation (ODA-SS), respectively, and analyze their performance in Partial Domain Adaptation (PDA-SS) and Open Partial Domain Adaptation (OPDA-SS) settings. CDA-SS assumes that the source and target domains share the same set of classes, while ODA-SS contains targetprivate classes that do not exist in the source domain. PDA-SS, on the other hand, assumes that the target domain contains only a subset of the source classes. OPDA-SS extends PDA-SS by adopting the open-set characteristic of ODA-SS, where both source-private and target-private classes exist simultaneously.

These evaluations reveal that adding source-private classes in transitions from CDA to PDA and from ODA to OPDA degrades performance. In PDA, common classes like “buildings” are often misclassified as sourceprivate, while “sidewalk” regions are mistakenly predicted as “road”. Similarly, in OPDA, target-private regions are frequently confused with source-private or common classes. Most state-of-the-art UDA-SS methods depend on self-training with target pseudo-labels, heavily relying on pseudo-label confidence scores. Particularly, in ODA-SS scenarios such as BUS, confidence scores are also used to assign unknown pseudo-labels. When source-private classes are present, their feature similarity to some common classes increases, leading to a reduction in pseudolabel confidence. As a result, common classes may not be effectively learned, and if the confidence score drops below a certain threshold $( \tau _ { p } )$ , common classes are often misassigned as target-private classes. This misassignment hinders the effective learning of both common and targetprivate classes, further degrading adaptation performance.

To mitigate this issue, we propose a novel framework UniMAP, UniDA-SS with Image Matching and Prototypebased Distinction, aim to increase the confidence scores of common classes under unknown category settings. First, we introduce a Domain-Specific Prototype-based Distinction (DSPD) to distinguish between common classes and source-private classes while considering variations of common classes between the source and target domains. Unlike conventional UDA-SS methods, which treat common classes as identical across domains, DSPD assigns two prototypes per class—one for source and one for target—to learn with one class while capturing domain-specific features. This approach enables independent learning of source and target-specific features, enhancing confidence scores for target predictions. Additionally, since common class pixel embeddings will have similar relative distances to the source and target prototypes, and the private class will be relatively close to any one prototype, we can use this to distinguish between common and private classes and assign higher weights to pixel embeddings that are more likely to belong to the common classes.

Second, to increase the confidence scores of the common classes, it is crucial to increase their pixel presence during training for robust domain-invariant representation. However, source-private classes often reduce the focus on common classes, hindering effective adaptation. To address this issue, we propose Target-based Image Matching (TIM), which prioritizes source images with the highest number of common class pixels based on target pseudo-labels. TIM compares target pseudo-labels and source ground truth at the pixel level, selecting the source images that overlap the most in common classes to pair with the target image in a single batch. This matching strategy facilitates domaininvariant learning of common classes, improving performance in a variety of scenarios. We also utilize a class-wise weighting strategy based on the target class distribution to assign higher weights to rare classes to address the class imbalance problem.

We summarize our main contributions as follows:

• We introduce a new task the Universal Domain Adaptation for Semantic Segmentation (UniDA-SS) task for the first time. To address this, we propose a novel framework called UniMAP, short for UniDA-SS with Image Matching and Prototype-based Distinction.

• To enhance pseudo-label confidence in the target domain, we propose Domain-Specific Prototype-based Distinction (DSPD), a pixel-level clustering approach that utilizes domain-specific prototypes to distinguish between common and private classes.

• We propose Target-based Image Matching (TIM), which enhances domain-invariant learning by prioritizing source images rich in common class pixels.

• We demonstrate the superiority of our approach by achieving state-of-the-art performance compared to existing UDA-SS methods through extensive experiments.

## 2. Related Work

## 2.1. Semantic Segmentation.

Semantic segmentation aims to classify each pixel in an image into a specific semantic. A foundational approach, Fully

Convolutional Networks (FCNs) [21], has demonstrated impressive performance in this task. To enhance contextual understanding, subsequent works have introduced methods such as dilated convolutions [4], global pooling [20], pyramid pooling [41], and attention mechanisms $[ 4 2 , 4 5 ]$ . More recently, transformer-based methods have achieved significant performance gains [37, 43]. Despite various studies, semantic segmentation models are still vulnerable to domain shifts or category shifts. To address this issue, we propose a universal domain adaptation for semantic segmentation that handles domain shifts and category shifts.

## 2.2. Unsupervised Domain Adaptation for Semantic Segmentation.

Unsupervised Domain Adaptation (UDA) aims to leverage labeled source data to achieve high performance on unlabeled target data. Existing UDA methods for semantic segmentation can be categorized into two approaches: adversarial learning-based and self-training. Adversarial learning-based methods [3, 9, 12, 16, 25, 32, 33] use an adversarial domain classifier to learn domain-invariant features. Self-training methods [5, 13, 14, 18, 19, 23, 31, 35, 36, 40, 46, 47] assign pseudo-labels to each pixel in the target domain using confidence thresholding, and several selftraining approaches further enhance target domain performance by re-training the model with these pseudo-labels. Although UDA allows the model to be trained on the target domain without annotations, it requires prior knowledge of class overlap between the source and target domains, which limits the model’s applicability and generalizability. To overcome this limitation, we propose a universal domain adaptation approach for semantic segmentation, where the model can adapt to the target domain without requiring prior knowledge of class overlap.

## 2.3. Universal Domain Adaptation in Classification

Universal Domain Adaptation (UniDA) [38] was introduced to address various domain adaptation settings, such as closed-set, open-set, and partial domain adaptation. UniDA is a more challenging scenario because it operates without prior knowledge of the category configuration of the source and target domains. To tackle UniDA in classification tasks, prior works have focused on computing confidence scores for known classes and treating samples with lower scores as unknowns. CMU [10] proposed a thresholding function, while ROS [1] used the mean confidence score as a threshold, which results in neglecting about half of the target data as unknowns. DANCE [29] set a threshold based on the number of classes in the source domain. OVANet [28] introduced training a threshold using source samples and adapting it to the target domain. While UniDA has been extensively studied in the context of classification tasks, it remains underexplored in semantic segmentation, which requires a higher level of visual understanding due to the need for pixel-wise classification. In this work, we aim to investigate UniDA for semantic segmentation.

## 3. Method

## 3.1. Problem Formulation

In the UniDA-SS scenario, the goal is to transfer knowledge from a labeled source domain $D _ { s } = ~ \{ X _ { s } , Y _ { s } \}$ to an unlabeled target domain $D _ { t } { = } \left\{ X _ { t } \right\}$ . The model is trained on the source images $X _ { s } = \{ x _ { s } ^ { 1 } , \bar { x _ { s } ^ { 2 } } , . . . , x _ { s } ^ { i _ { s } } \}$ with the corresponding labels $Y _ { s } = \{ y _ { s } ^ { 1 } , y _ { s } ^ { 2 } , . . . , y _ { s } ^ { i _ { s } } \}$ and the target images $X _ { t } =$ $\{ \bar { x _ { t } ^ { 1 } } , x _ { t } ^ { 2 } , . . . , x _ { t } ^ { i _ { t } } \}$ , where ground-truth labels are unavailable. Each image $\bar { x _ { s } ^ { i _ { s } } } \in \mathbb { R } ^ { 3 \times \bar { H } \times W }$ and $y _ { s } ^ { i _ { s } } \in \mathbb { R } ^ { C \times H \times W }$ represent an $i _ { s } \mathrm { - t h }$ RGB image and its pixel-wise label. H and $W$ are the height and width of the image, and $C _ { s }$ and $C _ { t }$ denote the sets of classes in the source and target domains, respectively. We aim to adapt the model to perform well on $D _ { t } ,$ , even though there is no prior knowledge of class overlap between $C _ { s }$ and $C _ { t }$ given. We define $C _ { c } = C _ { s } \cap C _ { t }$ as the set of common classes, while $C _ { s } \ \backslash C _ { c }$ and $C _ { t } \ \backslash C _ { c }$ represent the classes private to each domain, respectively. To handle target-private samples in $C _ { t } \setminus C _ { s } ,$ , we classify them as “unknown” without prior knowledge of their identities. Under this formulation, UniDA-SS requires addressing two challenges: (1) to classify common classes in $C _ { c }$ correctly and (2) to detect target-private classes in $C _ { t } \ \backslash C _ { s }$

## 3.2. Baseline

We construct our UniDA-SS baseline by adopting a standard open-set self-training approach, partially following the ODA-SS formulation introduced in BUS [6]. BUS handles unknown target classes by appending an additional classification head node to predict unknown class. In our baseline, we adopt the same structural design as BUS but remove refinement components and the use of attached private class masks, resulting in a setup suitable for UniDA-SS.

In this baseline, the number of classifier heads is set to $( C _ { s } + 1 )$ , where the $( C _ { s } + 1 )$ )-th head corresponds to the unknown class. The segmentation network $f _ { \theta }$ is trained with the labeled source data using the following categorical cross-entropy loss $\mathcal { L } _ { s e g } ^ { s } \mathrm { . }$

$$
\mathcal { L } _ { s e g } ^ { s } = - \sum _ { j = 1 } ^ { H \cdot W } \sum _ { c = 1 } ^ { C _ { s } + 1 } y _ { s } ^ { ( j , c ) } \log f _ { \theta } ( x _ { s } ) ^ { ( j , c ) } ,\tag{1}
$$

where $j \in \{ 1 , 2 , . . . , H \cdot W \}$ denotes the pixel index and $c \in \{ 1 , 2 , . . . , C _ { s } + 1 \}$ denotes the class index. The baseline utilizes a teacher network $g _ { \phi }$ to generate the target pseudolabels. $g _ { \phi }$ is updated from $f _ { \theta }$ via exponential moving average $\mathbf { ( E M A ) }$ [30] with a smoothing factor α. The pseudolabel $\hat { y } _ { t p } ^ { ( j ) }$ for the $j \cdot$ -th pixel considering unknown assigned

![](images/63bbf1b96adc1dd8afb1b08604dfebc65073a077565f7673f137901ecaaea87b.jpg)  
Figure 2. Overview of our proposed method, UniMAP. The top right illustrates the main training framework. The model is optimized with three main losses: the supervised segmentation loss on the source domain $L _ { s e g } ^ { s } ,$ the pseudo-label guided loss on the target domain $L _ { s e g } ^ { t }$ using DACS [31], which is a domain mixing technique, and $L _ { p r o t o } .$ the prototype-based loss $L _ { p r o t o }$ computed in a fixed ETF space [26]. $L _ { p r o t o }$ consists of three losses, which allows the prototype to have domain-specific information. Pixel-wise weight scaling factor $w ,$ is derived based on the relative distance between source and target prototypes, assigning higher weights to common classes that align well with both prototypes. These weights are used in generating target pseudo-labels and the target loss $L _ { s e g } ^ { t }$ . On the top left is the framework of TIM. It computes the class distribution of the target pseudo-label and ranks source images based on class overlap using the similarity score $S _ { s }$ . The top-ranked source image is selected and paired with the target image in each training batch.

as follows:

$$
\hat { y } _ { t p } ^ { ( j ) } = \left\{ { c } ^ { \prime } , \qquad \mathrm { i f } \ \left( \mathrm { m a x } _ { c ^ { \prime } } g _ { \phi } ( x _ { t } ) ^ { ( j , c ^ { \prime } ) } \geq \tau _ { p } \right) \right. ,\tag{2}
$$

where $c ^ { ' } \in \{ 1 , 2 , . . . , C _ { s } \}$ denotes a known classes and $\tau _ { p }$ is a fixed threshold for assign unknown pseudo-labels. Then, we calculate the image-level reliability of the pseudo-label $q _ { t }$ as follows [31]:

$$
q _ { t } = \frac { 1 } { H \cdot W } \sum _ { j = 1 } ^ { H \cdot W } \left[ \operatorname* { m a x } _ { c ^ { \prime } } g _ { \phi } ( x _ { t } ) ^ { ( j , c ^ { \prime } ) } \geq \tau _ { t } \right] ,\tag{3}
$$

where $\tau _ { t }$ is a hyperparameter. The network $f _ { \theta }$ is trained using the pseudo-labels and the corresponding confidence estimates with the using the weighted cross-entropy loss $\mathcal { L } _ { s e g } ^ { t } \{ $

$$
\mathcal { L } _ { s e g } ^ { t } = - \sum _ { j = 1 } ^ { H \cdot W } \sum _ { c = 1 } ^ { C _ { s } + 1 } q _ { t } \cdot \hat { y } _ { t p } ^ { ( j , c ) } \log f _ { \theta } ( x _ { t } ) ^ { ( j , c ) } .\tag{4}
$$

Based on this baseline, we propose a novel framework called UniMAP, short for UniDA-SS with Image Matching and Prototype-based Distinction.

## 3.3. Domain-Specific Prototype-based Distinction

Prototype-based Learning. In conventional selftraining-based UDA-SS methods, common classes from both the source and target domains are typically treated as a unified class, assuming identical feature representations. However, in practice, common classes often exhibit domain-specific features (e.g., road appearance and texture differences between Europe and India). To address this issue, we leverage the concept from ProtoSeg [44]. ProtoSeg uses multiple non-learnable prototypes per class to represent diverse features within the pixel embedding space, adequately capturing inter-class variance. Building on this idea, we assign two prototypes per class, one for the source and one for the target. This allows the model to capture domain-specific features for each class while still learning them as a unified class, effectively enhancing the confidence scores for common classes in the target domain. To ensure that the source and target prototypes maintain a stable distance, we use a fixed Simplex Equiangular Tight Frame (ETF) [26], which guarantees equal cosine similarity and L2-norm across all prototype pairs. This structure enables consistent separation between the source and target prototypes, facilitating the learning of domain-specific features. The prototypes are defined as follows:

$$
\{ p _ { k } \} _ { k = 1 } ^ { 2 C + 1 } = \sqrt { \frac { 2 C + 1 } { 2 C } } U ( I _ { 2 C + 1 } - { \frac { 1 } { 2 C + 1 } } 1 _ { [ 2 C + 1 ] } 1 _ { [ 2 C + 1 ] } ^ { \intercal } ) ,\tag{5}
$$

Each class has a pair of prototypes $p ^ { c } \in \{ p _ { s } ^ { c } , p _ { t } ^ { c } \}$ , with an additional prototype is defined for unknown classes $p ^ { C + 1 } \in$ $\{ p _ { t } ^ { C + 1 } \}$ . We employ three prototype-based loss functions adapted from ProtoSeg for each domain $D \in \{ s , t \}$ . First, the cross entropy loss $\mathcal { L } _ { C E }$ that moves the target closer to the corresponding prototype and further away from the rest of the prototype as follows:

$$
\mathcal { L } _ { C E } ^ { D } = - l o g \frac { e x p ( i ^ { \boldsymbol { \mathsf { T } } } p _ { D } ^ { c } ) } { e x p ( i ^ { \boldsymbol { \mathsf { T } } } p _ { D } ^ { c } ) + \sum _ { c ^ { \prime } \neq c } e x p ( i ^ { \boldsymbol { \mathsf { T } } } p _ { D } ^ { c ^ { \prime } } ) } ,\tag{6}
$$

where i represents the L2-normalized pixel embedding, using the label for source pixels and the pseudo-label for target pixels to determine the corresponding class $c .$ Second, pixel-prototype contrastive learning strategy $\mathcal { L } _ { P P C }$ , which makes it closer to the corresponding prototype in the entire space and farther away from the rest as follows:

$$
\mathcal { L } _ { P P C } ^ { D } = - l o g \frac { \sum _ { p \in p ^ { c } } e x p ( i ^ { \boldsymbol { \mathsf { T } } } p ^ { c } / \tau ) } { \sum _ { p \in p ^ { c } } e x p ( i ^ { \boldsymbol { \mathsf { T } } } p ^ { c } / \tau ) + \sum _ { p ^ { - } \in P ^ { - } } e x p ( i ^ { \boldsymbol { \mathsf { T } } } p ^ { - } / \tau ) } ,
$$

where $P ^ { - }$ denotes set of prototypes excluding $p ^ { c }$ . Finally, Pixel-Prototype Distance Optimization $\mathcal { L } _ { P P D }$ makes the distance of pixel embedding and prototype closer as:

$$
\begin{array} { r } { \mathcal { L } _ { P P D } ^ { D } = ( 1 - i ^ { \boldsymbol { \mathsf { T } } } p _ { D } ^ { c } ) ^ { 2 } . } \end{array}\tag{8}
$$

Therefore, we can organize $\mathcal { L } _ { p r o t o }$ as follows:

$$
\mathcal { L } _ { p r o t o } = \mathcal { L } _ { C E } + \lambda _ { 1 } \mathcal { L } _ { P P C } + \lambda _ { 2 } \mathcal { L } _ { P P D } ,\tag{9}
$$

where $\lambda _ { 1 }$ and $\lambda _ { 2 }$ denote hyperparameters. Through the $\mathcal { L } _ { p r o t o : }$ , the model can capture domain-specific features while learning each class as a unified representation.

Prototype-based Weight Scaling. We further utilize prototypes to distinguish between common class and sourceprivate. As training progresses, common-class pixel embeddings tend to align with both source and target prototypes, whereas private-class embeddings align with only one. Thus, when an embedding is similarly close to both prototypes, it is likely to be from a common class. Based on this, we assign a pixel-wise weight scaling factor w to reflect the likelihood of a pixel belonging to a common class:

$$
w = \frac { 2 ( d _ { s } + 1 ) ( d _ { t } + 1 ) } { ( d _ { s } + 1 ) + ( d _ { t } + 1 ) } ,\tag{10}
$$

where $d _ { s } , d _ { t }$ denote cosine similarity between pixel embedding i and the source and target prototypes $p _ { s } ^ { c } , p _ { t } ^ { c }$ , respectively. The scaling factor w is then applied to Equation (11) during pseudo-label generation as follows:

$$
\mathcal { L } _ { s e g } ^ { t } = - \sum _ { j = 1 } ^ { H \cdot W } \sum _ { c = 1 } ^ { C + 1 } w \cdot q _ { t } \hat { y } _ { t p } ^ { ( j , c ) } \log f _ { \theta } ( x _ { t } ) ^ { ( j , c ) } ,\tag{11}
$$

$$
\hat { y } _ { t p } ^ { ( j ) } = \{ { { c ^ { \prime } } , \atop { C + 1 , } }   \mathrm { i f } ( \mathrm { m a x } _ { c ^ { \prime } } g _ { \phi } ( x _ { t } ) ^ { ( j , c ^ { \prime } ) } \cdot w \geq \tau _ { p } ) .\tag{12}
$$

The above method mitigates the assignment of a common class to target-private in the target pseudo-label and enhances the learning of pixels with a high probability of a common class.

## 3.4. Target-based Image Matching

To increase the confidence score of common classes, it is important to include as many common classes as possible in the training to learn domain-invariant representation. However, when source-private classes are added, the proportion of learning common classes decreases, making it difficult to learn domain-invariant representation. To solve this problem, we propose the Target-based Image Matching (TIM) method, which selects images containing as many common classes as possible from source images based on the classes appearing in the target pseudo-label. First, we calculate the proportion of each class present in the target pseudo-label $\hat { y } _ { t p }$ as follows:

$$
f _ { c } = { \frac { n _ { c } } { \sum _ { k } n _ { k } } } ,\tag{13}
$$

where $n _ { c }$ denotes the number of pixels of class $c$ in $\hat { y } _ { t p }$ Utilizing $f _ { c }$ we calculate $\hat { f } _ { c } ,$ which has a higher value for rare classes, as follows:

$$
\hat { f } _ { c } = s o f t m a x ( \frac { 1 - f _ { c } } { T } ) ,\tag{14}
$$

where $T$ denotes temperature. For each source image through $\hat { f } _ { c }$ , we measure $S _ { s }$ as follows:

$$
S _ { s } = \sum _ { c \in c ^ { * } } n _ { c } ^ { s } \hat { f } _ { c } ,\tag{15}
$$

where $n _ { c } ^ { s }$ denotes the number of pixels of class c in $y _ { s }$ and $c ^ { * }$ denotes set of overlapping classes between $y _ { s }$ and $\hat { y } _ { t p }$ . So, we select the source image with the highest $S _ { s }$ and pair it with the corresponding target image in a training batch. This approach allows us to effectively learn domaininvariant representations for common classes, which can improve performance in a variety of scenarios. It also mitigates class imbalance by prioritizing source images that contain more pixels from rare common classes, guided by class weighting based on the target class distribution.

## 4. Experiments

## 4.1. Experimental Setup

Datasets. We evaluated our method on two newly defined OPDA-SS benchmarks: Pascal-Context [24] →

<table><tr><td colspan="10">Pascal-Context → Cityscapes</td><td colspan="6"></td></tr><tr><td>Method</td><td>Road</td><td>S.walk</td><td>Build.</td><td>Wall</td><td>Fence</td><td>Veget.</td><td>Sky</td><td>Car</td><td>Truck</td><td>Bus</td><td>M.bike</td><td>Bike</td><td>Common</td><td>Private</td><td>H-score</td></tr><tr><td>UAN [38]</td><td>61.78</td><td>13.14</td><td>78.14</td><td>0.03</td><td>5.60</td><td>20.01</td><td>81.50</td><td>33.2</td><td>36.24</td><td>4.90</td><td>15.48</td><td>13.01</td><td>31.93</td><td>4.30</td><td>7.47</td></tr><tr><td>UniOT [2]</td><td>62.34</td><td>15.64</td><td>75.69</td><td>0.05</td><td>4.61</td><td>21.50</td><td>78.10</td><td>34.3</td><td>35.04</td><td>5.94</td><td>12.98</td><td>15.85</td><td>32.84</td><td>6.85</td><td>10.76</td></tr><tr><td>MLNet [22]</td><td>71.28</td><td>12.94</td><td>68.63</td><td>0.00</td><td>6.15</td><td>19.73</td><td>81.7</td><td>22.8</td><td>27.04</td><td>4.45</td><td>11.68</td><td>10.72</td><td>30.81</td><td>6.43</td><td>10.61</td></tr><tr><td>DAFormer [13]</td><td>25.29</td><td>0.00</td><td>83.44</td><td>0.09</td><td>7.69</td><td>86.94</td><td>91.68</td><td>91.59</td><td>81.80</td><td>66.18</td><td>55.66</td><td>60.49</td><td>54.24</td><td>4.43</td><td>8.20</td></tr><tr><td>HRDA [14]</td><td>62.33</td><td>0.00</td><td>77.75</td><td>0.64</td><td>30.87</td><td>80.49</td><td>83.24</td><td>88.79</td><td>70.11</td><td>58.66</td><td>9.11</td><td>21.75</td><td>51.89</td><td>8.55</td><td>14.68</td></tr><tr><td>MIC [15]</td><td>40.49</td><td>0.21</td><td>79.40</td><td>0.00</td><td>8.35</td><td>85.74</td><td>89.58</td><td>84.78</td><td>46.87</td><td>47.23</td><td>47.78</td><td>53.59</td><td>48.67</td><td>7.85</td><td>13.51</td></tr><tr><td>BUS [6]</td><td>77.90</td><td>0.01</td><td>85.26</td><td>0.00</td><td>31.16</td><td>87.12</td><td>88.43</td><td>89.94</td><td>64.51</td><td>53.71</td><td>50.22</td><td>63.40</td><td>57.64</td><td>20.38</td><td>30.11</td></tr><tr><td>UniMAP (Ours)</td><td>84.15</td><td>16.77</td><td>86.38</td><td>0.00</td><td>35.12</td><td>88.26</td><td>89.45</td><td>90.75</td><td>64.54</td><td>59.25</td><td>49.98</td><td>66.63</td><td>60.94</td><td>31.27</td><td>41.33</td></tr></table>

Table 1. Semantic segmentation performance on Pascal-Context → Cityscapes OPDA-SS benchmarks. Our method outperformed base lines in common, private, and overall performance. White columns show individual common class scores, while “Common” in gray columns represents the average performance of common classes. The best results are highlighted in bold.
<table><tr><td colspan="16">GTA5 → IDD Common Private H-score</td></tr><tr><td>Method Road S.walk Build.</td><td colspan="7">Wall Fence Pole Light Sign Veget. Sky Person Rider Car Bike</td><td colspan="6">Truck Bus M.bike</td></tr><tr><td>UAN [38]</td><td>97.38 61.33 62.24</td><td>36.27 16.41</td><td>24.11 8.96 58.29</td><td>78.82 94.15</td><td>57.06</td><td>30.09 68.98</td><td>72.92</td><td>42.66</td><td>64.93</td><td>7.85</td><td>49.20</td><td>3.14</td><td>5.92</td></tr><tr><td>UniOT [2]</td><td>96.99 41.19 63.61</td><td>34.63 18.96</td><td>28.35 3.96 54.07</td><td>72.9 92.89</td><td>53.9</td><td>32.36 81.82</td><td>72.85</td><td>63.84</td><td>63.28</td><td>5.18</td><td>51.82</td><td>7.44</td><td>13.01</td></tr><tr><td>MLNet [22]</td><td>95.59 9.87 55.53</td><td>17.26 12.14</td><td>12.69 5.81 64.13</td><td>72.69 91.57</td><td>0.00</td><td>17.92 69.59</td><td>65.65</td><td>50.35</td><td>60.76</td><td>5.39</td><td>41.58</td><td>4.23</td><td>7.68</td></tr><tr><td>DAFormer [13]</td><td>97.89 54.84 70.28</td><td>43.71 25.56</td><td>37.74 14.57 66.80</td><td>79.14 91.92</td><td>58.31</td><td>52.31 83.36</td><td>80.14</td><td>77.16</td><td>64.70</td><td>21.54</td><td>52.05</td><td>21.07</td><td>29.99</td></tr><tr><td>HRDA [14]</td><td>97.90 52.22</td><td>69.80 42.73 25.15</td><td>38.79 21.43 66.80</td><td>80.06 91.38</td><td>57.60</td><td>50.83 83.27</td><td>80.05</td><td>76.35</td><td>64.05</td><td>20.07</td><td>57.83</td><td>22.47</td><td>32.69</td></tr><tr><td>MIC [15]</td><td>95.18 39.64</td><td>67.66 43.19 23.08</td><td>36.32 17.06 65.09</td><td>85.39 94.48</td><td>53.37</td><td>57.35 79.67</td><td>81.47</td><td>65.86</td><td>65.40</td><td>20.27</td><td>56.42</td><td>24.68</td><td>34.82</td></tr><tr><td>BUS [6]</td><td>98.31 74.34</td><td>73.65 48.05 34.62</td><td>46.21 30.15 74.17</td><td>87.06 95.77</td><td>64.38</td><td>66.91 89.31</td><td>87.84 89.77</td><td></td><td>71.89</td><td>16.25</td><td>65.47</td><td>29.70</td><td>41.26</td></tr><tr><td>UniMAP (Ours)</td><td>98.13 62.50</td><td>76.12 85.74 27.48</td><td>46.56 26.07 59.63</td><td>90.44 96.31</td><td>65.87</td><td>66.85 82.83</td><td>87.08 68.33</td><td></td><td>70.27</td><td>35.45</td><td>64.08</td><td>34.78</td><td>45.51</td></tr></table>

Table 2. Semantic segmentation performance on GTA5 → IDD OPDA-SS benchmarks. Our method outperformed baselines in common, private, and overall performance. White columns show individual common class scores, while “Common” in gray columns represents the average performance of common classes. The best results are highlighted in bold.

Cityscapes [7], and GTA5 [27] → IDD [34], which we introduce to assess universal domain adaptation in more realistic settings involving both source-private and targetprivate classes. Pascal-Context → Cityscapes is a real-toreal scenario, and Pascal-Context contains both in-door and out-door, while Cityscapes only has a driving scene, so it is a scenario with a considerable amount of source-private classes. We selected 12 classes as common classes and the remaining 7 classes (“pole”, “light”, “sign”, ”terrain”, “person”, “rider”, and “train”) are treated as target-private classes. GTA5 → IDD is a synthetic-to-real scenario and GTA5 features highly detailed synthetic driving scenes set in urban cityscapes, while IDD captures real-world driving scenarios on diverse roads in India. We used 17 classes as common classes, 2 source-private class (“terrain”, “train”), and 1 target-private class (“auto-rickshaw”).

Evaluation Protocols. In the OPDA-SS setting, both common class and target-private performance are important, so we evaluate methods using H-Score, which can fully reflect them. The H-score is calculated as the harmonic mean of the common mIoU (mean Intersection-over-Union) and the target-private IoU.

Implementation Details. This method is based on BUS. We used the muli-resolution self-training strategy and training parameter used in MIC [15]. The network used a MiT-

B5 [37] encoder and was initialized with ImageNet-1k [8] pretrained. The learning rate was 6e-5 for the backbone and 6e-4 for the decoder head, with a weight decay of 0.01 and linear learning rate warm-up over 1.5k steps. EMA factor α was 0.999 and the optimizer was AdamW [17]. ImageNet feature Distance [13], DACS [31] data augmentation, Masked Image Consistency module [15], and Dilation-Erosion-based Contrastive Loss [6] were used. We also modified some of the BUS methods to suit the OPDA setting. In OpenReMix [6], we applied only Resizing Object except Attaching Private and did not use refinement through MobileSAM [39]. For rare class sampling [13], we switched from calculating a distribution based on the existing source and applying it to source sampling to applying it to target sampling based on the target pseudo-label distribution. We trained on a batch of two 512 × 512 random crops for 40k iterations. The hyperparameter are set to: $\tau _ { p } = 0 . 5 ,$ τ<sub>t</sub> = 0.968, λ<sub>1</sub> = 0.01, λ<sub>2</sub> = 0.01, τ = 0.1, and T = 0.01.

Baselines. Since there is no existing research on OPDA-SS, we performed experiments by changing the methods in different settings to suit the OPDA-SS. First, for UniDA for classification methods [2, 22, 38], we experimented by changing the backbone to a semantic segmentation model. In this case, we used the DeepLabv2 [4] segmentation network and ResNet-101 [11] as the backbone. For the CDA-SS methods [13–15], we added 1 dimension to the head dimension of the classifier to predict the target-private and

![](images/f9e4822e6318993338bdb669192f376cd9043324641efd6f3c0de24dadd37b55.jpg)  
Figure 3. Qualitative results of OPDA-SS setting. We visualize the segmentation predictions from different methods on the Cityscapes dataset. White and yellow represent target-private and source-private classes, respectively. while other colors indicate common classes (e.g., purple for “road” and pink for “sidewalk”). Compared to HRDA, MIC, and BUS, our method more accurately segments both common and target-private classes.

assigned an unknown based on the confidence score [6].   
Lastly, the ODA-SS method, BUS [6], was used as it is.

## 4.2. Comparisons with the State-of-the-Art

We compared performance on two benchmarks for OPDA-SS settings. Table 1 presents the semantic segmentation performance from Pascal-Context → Cityscapes, while Table 2 presents the performance from GTA5 → IDD. As shown in Table 1, UniMAP achieved outstanding performance in the Pascal-Context → Cityscapes benchmark. Specifically, it outperformed previous approaches by a significant margin, with improvements of approximately 3.3 for Common, 10.89 for Private, and 11.22 in H-score. These results indicate that UniMAP effectively enables the model to learn both common and private classes. Notably, UniMAP surpassed BUS, the state-of-the-art in ODA-SS, in terms of private class performance. Although our method primarily focuses on capturing knowledge of common classes, it also enhances the identification of private classes due to improved representation learning. In addition, Table 2 shows the performance comparison for the GTA5 → IDD benchmark. Our method demonstrated notable improvements in both Private and H-Score. In particular, while prior methods in CDA-SS showed inferior performance for Private and H-score, our approach led to significant gains of approximately 6.25 for Common, 10.3 for Private, and 9.69 for H-score. Although our method had a relatively lower performance than BUS in Common, it surpassed BUS in Private performance with a margin of about 5.08, ultimately leading to superior H-score results. Overall, the experimental findings demonstrate that our method delivers promising performance in OPDA-SS settings, which is critical for achieving effective UniDA-SS.

## 4.3. Qualitative Evaluation

We conducted qualitative experiments under the OPDA-SS setting. Figure 3 compared prediction maps from

<table><tr><td>UniMAP</td><td colspan="3">Pascal-Context → Cityscapes</td></tr><tr><td>DSPD</td><td>TIM Common</td><td>Private</td><td>H-Score</td></tr><tr><td rowspan="3"></td><td>53.79</td><td>26.54</td><td>36.03</td></tr><tr><td>59.46</td><td>27.97</td><td>38.04</td></tr><tr><td>56.22</td><td>29.14</td><td>38.39</td></tr><tr><td></td><td>60.94</td><td>31.27</td><td>41.33</td></tr></table>

Table 3. Ablation study of our method on Pascal-Context → Cityscapes. We evaluate the contributions of DSPD and TIM, where the baseline is BUS without private attaching and pseudolabel refinement. The best results are highlighted in bold.
<table><tr><td colspan="2">DSPD</td><td colspan="3">Pascal-Context → Cityscapes</td></tr><tr><td>w</td><td> $\mathcal { L } _ { p r o t o }$ </td><td>Common</td><td>Private</td><td>H-Score</td></tr><tr><td rowspan="3"></td><td></td><td>53.79</td><td>26.54</td><td>36.03</td></tr><tr><td></td><td>54.38</td><td>21.75</td><td>31.08</td></tr><tr><td></td><td>59.71</td><td>26.76</td><td>36.96</td></tr><tr><td></td><td></td><td>59.46</td><td>27.97</td><td>38.04</td></tr></table>

Table 4. Further ablation study of DSPD components on Pascal-Context → Cityscapes. w represents pixel-wise weight scaling factor, and $\mathcal { L } _ { \mathrm { p r o t o } }$ represents the prototype loss function. The best results are highlighted in bold.

Cityscapes against baselines, where white and yellow represent target-private and source-private classes, respectively, while other colors denote common classes. Baseline methods such as HRDA, MIC, and BUS tend to either misclassify common classes as target-private or sacrifice common class accuracy to detect target-private regions. In contrast, UniMAP successfully predicted both common and targetprivate classes. Notably, it accurately identified the “sidewalk” class (pink) in rows 2 and 3, unlike other baselines. These results indicate that UniMAP effectively balances the identification of common and target-private classes.

## 4.4. Ablation Study

Ablation Study about UniMAP. Table 3 shows the experimental results of the ablation study of the performance contribution of each component. As described in the Implementation Details section, the baseline model, derived by removing the Attaching Private and refinement pseudolabel module from the BUS, achieved an H-Score of 36.03. First, applying DSPD alone to the baseline, the H-Score improves to 38.04, increasing both Common and Private performance. This enhancement indicates that DSPD effectively captures domain-specific features, improving performance for both the common and target-private classes compared to the Baseline. Next, when only applying TIM alone to the baseline, also improves performance, achieving an H-Score of 38.39, with better Private. This result suggests that TIM successfully learns domain-invariant representations between source and target by leveraging target pseudo-labels, thus enhancing overall performance. Finally, when both DSPD and TIM are applied to the baseline, the model achieves the best performance, with an H-Score of 41.33. This demonstrates that DSPD and TIM work synergistically, enabling the model to achieve superior performance across both common and target-private classes.

<table><tr><td rowspan="2">Method</td><td colspan="8">Pascal-Context → Cityscapes</td><td rowspan="2">Common Average</td><td rowspan="2">H-Score Average</td></tr><tr><td></td><td>Open Partial Set DA</td><td></td><td></td><td>Open Set DA</td><td></td><td>Partial Set DA</td><td>Closed Set DA</td></tr><tr><td></td><td>Common</td><td>Private</td><td>H-Score</td><td>Common</td><td>Private</td><td>H-Score</td><td>Common</td><td>Common</td><td></td><td></td></tr><tr><td>DAF</td><td>54.24</td><td>4.43</td><td>8.19</td><td>44.27</td><td>12.07</td><td>18.97</td><td>35.18</td><td>46.48</td><td>44.51</td><td>12.46</td></tr><tr><td>HRDA</td><td>51.89</td><td>8.55</td><td>14.68</td><td>52.76</td><td>14.76</td><td>23.07</td><td>51.99</td><td>63.17</td><td>54.76 57.97</td><td>18.40</td></tr><tr><td>MIC BUS</td><td>48.67</td><td>7.85</td><td>13.52</td><td>60.88</td><td>23.79</td><td>34.21</td><td>58.04</td><td>65.68</td><td>59.26</td><td>21.51</td></tr><tr><td></td><td>57.64</td><td>20.38</td><td>30.11</td><td>60.67</td><td>27.05</td><td>37.42</td><td>58.54</td><td>60.24</td><td></td><td>33.57</td></tr><tr><td>UniMAP (Ours)</td><td>60.94</td><td>31.27</td><td>41.33</td><td>58.50</td><td>24.73</td><td>34.76</td><td>59.44</td><td>64.74</td><td>60.86</td><td>37.90</td></tr></table>

Table 5. Experimental results on Pascal-Context → Cityscapes for various domain adaptation scenarios. For a fair comparison, all methods used a head-expansion model. The best results are highlighted in bold.

Ablation Study about DSPD. Table 4 shows the impact of the individual components of DSPD, namely $L _ { p r o t o }$ and w on performance in the Pascal-Context → Cityscapes scenario. The $L _ { p r o t o }$ represents the pixel embedding loss in the ETF space, designed to guide pixel embeddings within a class to be closer to their respective prototypes. When only $L _ { p r o t o }$ is applied, the model achieves a Common of 59.71, a Private of 26.76, and an H-Score of 36.96. This result suggests that $L _ { p r o t o }$ alone can enhance the clustering of pixel embeddings around domain-specific prototypes, thereby improving overall performance compared to the baseline. The w, on the other hand, means a weighting mechanism based on the ETF prototype structure that estimates the common class more effectively and applies weights scaling accordingly. When only w is used, the Common drops to 54.38, and the Private score falls to 21.75, resulting in a lower H-Score of 31.08. This indicates that while w is utilized in distinguishing common classes, it is less effective without the guidance provided by $L _ { p r o t o } .$ When both $L _ { p r o t o }$ and w are combined, the model achieves the best performance, with a Common of 59.46, a Private of 27.97, and an H-Score of 38.04. This demonstrates that the two components are complementary: $L _ { p r o t o }$ enhances pixel embedding alignment with domain-specific prototypes, while w further boosts the ability to focus on common class pixels with appropriate weighting. Together, they yield a notable improvement in the overall H-Score.

Comparisons in Various Category Settings. We further compared the performance generalization ability of UniMAP across various domain adaptation settings. As shown in 5, while some existing methods achieve slightly better results in Closed Set and Open Set settings due to their specialized assumptions, UniMAP demonstrates clear advantages in Partial Set and Open Partial Set, where prior methods have not been actively explored. Notably, UniMAP achieves the highest scores, with a Common Average of 60.86 and an H-Score Average of 37.90, validating its robustness and effectiveness across varying category shift configurations. These results highlight the practicality of our framework for the real-world scenario, where category settings are often unknown.

## 5. Conclusion

In this paper, we proposed a new framework for UniDA-SS, called UniMAP. Since UniDA-SS must handle different domain configurations without prior knowledge of category settings, it is very important to identify and learn common classes across domains. To this end, UniMAP incorporates two key components: Domain-Specific Prototypebased Distinction (DSPD) and Target-based Image Matching (TIM). DSPD is used to estimate common classes from the unlabeled target domain, while TIM samples labeled source images to transfer knowledge to the target domain effectively. Experimental results show that our method improved average performance across different domain adaptation scenarios. We hope our approach sheds light on the necessity of universal domain adaptation for the semantic segmentation task.

## Acknowledgment

This research was conducted with the support of the HAN-COM InSpace Co., Ltd. (Hancom-Kyung Hee Artificial Intelligence Research Institute), and was supported by Korea Planning & Evaluation Institute of Industrial Technology (KEIT) grant funded by the Korea government (MOTIE) (RS-2024-00444344), and in part by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No. RS-2019-II190079, Artificial Intelligence Graduate School Program (Korea University)).

## References

[1] Silvia Bucci, Mohammad Reza Loghmani, and Tatiana Tommasi. On the effectiveness of image rotation for open set domain adaptation. In European conference on computer vision, pages 422–438. Springer, 2020. 3

[2] Wanxing Chang, Ye Shi, Hoang Tuan, and Jingya Wang. Unified optimal transport framework for universal domain adaptation. Advances in Neural Information Processing Systems, 35:29512–29524, 2022. 6

[3] Cheng Chen, Qi Dou, Hao Chen, Jing Qin, and Pheng-Ann Heng. Synergistic image and feature adaptation: Towards cross-modality domain adaptation for medical image segmentation. In Proceedings of the AAAI conference on artificial intelligence, pages 865–872, 2019. 3

[4] Liang-Chieh Chen, George Papandreou, Iasonas Kokkinos, Kevin Murphy, and Alan L Yuille. Deeplab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected crfs. IEEE transactions on pattern analysis and machine intelligence, 40(4):834–848, 2017. 3, 6

[5] Minghao Chen, Hongyang Xue, and Deng Cai. Domain adaptation for semantic segmentation with maximum squares loss. In Proceedings of the IEEE/CVF international conference on computer vision, pages 2090–2099, 2019. 3

[6] Seun-An Choe, Ah-Hyung Shin, Keon-Hee Park, Jinwoo Choi, and Gyeong-Moon Park. Open-set domain adaptation for semantic segmentation. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 23943–23953, 2024. 2, 3, 6, 7

[7] Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele. The cityscapes dataset for semantic urban scene understanding. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 3213–3223, 2016. 6

[8] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009. 6

[9] Liang Du, Jingang Tan, Hongye Yang, Jianfeng Feng, Xiangyang Xue, Qibao Zheng, Xiaoqing Ye, and Xiaolin Zhang. Ssf-dan: Separated semantic feature based domain adaptation network for semantic segmentation. In Proceed-

ings of the IEEE/CVF International Conference on Com puter Vision, pages 982–991, 2019. 3

[10] Bo Fu, Zhangjie Cao, Mingsheng Long, and Jianmin Wang. Learning to detect open classes for universal domain adapta tion. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XV 16, pages 567–583. Springer, 2020. 3

[11] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceed ings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016. 6

[12] Weixiang Hong, Zhenzhen Wang, Ming Yang, and Junsong Yuan. Conditional generative adversarial network for structured domain adaptation. In Proceedings of the IEEE con ference on computer vision and pattern recognition, pages 1335–1344, 2018. 3

[13] Lukas Hoyer, Dengxin Dai, and Luc Van Gool. Daformer: Improving network architectures and training strategies for domain-adaptive semantic segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9924–9935, 2022. 3, 6

[14] Lukas Hoyer, Dengxin Dai, and Luc Van Gool. Hrda: Context-aware high-resolution domain-adaptive semantic segmentation. In European Conference on Computer Vision, pages 372–391. Springer, 2022. 3, 6

[15] Lukas Hoyer, Dengxin Dai, Haoran Wang, and Luc Van Gool. Mic: Masked image consistency for contextenhanced domain adaptation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11721–11732, 2023. 2, 6

[16] Myeongjin Kim and Hyeran Byun. Learning texture invariant representation for domain adaptation of semantic segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12975– 12984, 2020. 3

[17] Xiangtai Li, Xia Li, Li Zhang, Guangliang Cheng, Jianping Shi, Zhouchen Lin, Shaohua Tan, and Yunhai Tong. Improving semantic segmentation via decoupled body and edge su pervision. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceed ings, Part XVII 16, pages 435–452. Springer, 2020. 6

[18] Yunsheng Li, Lu Yuan, and Nuno Vasconcelos. Bidirectional learning for domain adaptation of semantic segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6936–6945, 2019. 3

[19] Qing Lian, Fengmao Lv, Lixin Duan, and Boqing Gong. Constructing self-motivated pyramid curriculums for crossdomain semantic segmentation: A non-adversarial approach. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 6758–6767, 2019. 3

[20] Wei Liu, Andrew Rabinovich, and Alexander C Berg. Parsenet: Looking wider to see better. arXiv preprint arXiv:1506.04579, 2015. 3

[21] Jonathan Long, Evan Shelhamer, and Trevor Darrell. Fully convolutional networks for semantic segmentation. In Pro ceedings of the IEEE conference on computer vision and pat tern recognition, pages 3431–3440, 2015. 3

[22] Yanzuo Lu, Meng Shen, Andy J Ma, Xiaohua Xie, and Jian-Huang Lai. Mlnet: Mutual learning network with neighborhood invariance for universal domain adaptation. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 3900–3908, 2024. 6

[23] Luke Melas-Kyriazi and Arjun K Manrai. Pixmatch: Unsupervised domain adaptation via pixelwise consistency training. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12435–12445, 2021. 3

[24] Roozbeh Mottaghi, Xianjie Chen, Xiaobai Liu, Nam-Gyu Cho, Seong-Whan Lee, Sanja Fidler, Raquel Urtasun, and Alan Yuille. The role of context for object detection and semantic segmentation in the wild. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 891–898, 2014. 5

[25] Fei Pan, Inkyu Shin, Francois Rameau, Seokju Lee, and In So Kweon. Unsupervised intra-domain adaptation for semantic segmentation through self-supervision. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 3764–3773, 2020. 3

[26] Vardan Papyan, XY Han, and David L Donoho. Prevalence of neural collapse during the terminal phase of deep learning training. Proceedings of the National Academy of Sciences, 117(40):24652–24663, 2020. 4

[27] Stephan R Richter, Vibhav Vineet, Stefan Roth, and Vladlen Koltun. Playing for data: Ground truth from computer games. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part II 14, pages 102–118. Springer, 2016. 6

[28] Kuniaki Saito and Kate Saenko. Ovanet: One-vs-all network for universal domain adaptation. In Proceedings ofthe ieee/cvf international conference on computer vision, pages 9000–9009, 2021. 3

[29] Kuniaki Saito, Donghyun Kim, Stan Sclaroff, and Kate Saenko. Universal domain adaptation through self supervision. Advances in neural information processing systems, 33:16282–16292, 2020. 3

[30] Antti Tarvainen and Harri Valpola. Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results. Advances in neural information processing systems, 30, 2017. 3

[31] Wilhelm Tranheden, Viktor Olsson, Juliano Pinto, and Lennart Svensson. Dacs: Domain adaptation via crossdomain mixed sampling. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 1379–1389, 2021. 3, 4, 6

[32] Yi-Hsuan Tsai, Wei-Chih Hung, Samuel Schulter, Kihyuk Sohn, Ming-Hsuan Yang, and Manmohan Chandraker. Learning to adapt structured output space for semantic segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 7472–7481, 2018. 3

[33] Yi-Hsuan Tsai, Kihyuk Sohn, Samuel Schulter, and Manmohan Chandraker. Domain adaptation for structured output via discriminative patch representations. In Proceedings of

the IEEE/CVF international conference on computer vision, pages 1456–1465, 2019. 3

[34] Girish Varma, Anbumani Subramanian, Anoop Namboodiri, Manmohan Chandraker, and CV Jawahar. Idd: A dataset for exploring problems of autonomous navigation in unconstrained environments. In 2019 IEEE winter conference on applications of computer vision (WACV), pages 1743–1751. IEEE, 2019. 6

[35] Qin Wang, Dengxin Dai, Lukas Hoyer, Luc Van Gool, and Olga Fink. Domain adaptive semantic segmentation with self-supervised depth estimation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 8515–8525, 2021. 3

[36] Yuxi Wang, Junran Peng, and ZhaoXiang Zhang. Uncertainty-aware pseudo label refinery for domain adaptive semantic segmentation. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9092–9101, 2021. 3

[37] Enze Xie, Wenhai Wang, Zhiding Yu, Anima Anandkumar, Jose M Alvarez, and Ping Luo. Segformer: Simple and efficient design for semantic segmentation with transformers. Advances in Neural Information Processing Systems, 34:12077–12090, 2021. 3, 6

[38] Kaichao You, Mingsheng Long, Zhangjie Cao, Jianmin Wang, and Michael I Jordan. Universal domain adaptation. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 2720–2729, 2019. 3, 6

[39] Chaoning Zhang, Dongshen Han, Yu Qiao, Jung Uk Kim, Sung-Ho Bae, Seungkyu Lee, and Choong Seon Hong. Faster segment anything: Towards lightweight sam for mo bile applications. arXiv preprint arXiv:2306.14289, 2023. 6

[40] Pan Zhang, Bo Zhang, Ting Zhang, Dong Chen, Yong Wang, and Fang Wen. Prototypical pseudo label denoising and tar get structure learning for domain adaptive semantic segmen tation. In Proceedings of the IEEE/CVF conference on com puter vision and pattern recognition, pages 12414–12424, 2021. 3

[41] Hengshuang Zhao, Jianping Shi, Xiaojuan Qi, Xiaogang Wang, and Jiaya Jia. Pyramid scene parsing network. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 2881–2890, 2017. 3

[42] Hengshuang Zhao, Yi Zhang, Shu Liu, Jianping Shi, Chen Change Loy, Dahua Lin, and Jiaya Jia. Psanet: Point wise spatial attention network for scene parsing. In Proceed ings ofthe European conference on computer vision (ECCV), pages 267–283, 2018. 3

[43] Sixiao Zheng, Jiachen Lu, Hengshuang Zhao, Xiatian Zhu, Zekun Luo, Yabiao Wang, Yanwei Fu, Jianfeng Feng, Tao Xiang, Philip HS Torr, et al. Rethinking semantic segmentation from a sequence-to-sequence perspective with transformers. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6881–6890, 2021. 3

[44] Tianfei Zhou, Wenguan Wang, Ender Konukoglu, and Luc Van Gool. Rethinking semantic segmentation: A prototype view. In Proceedings of the IEEE/CVF Conference

on Computer Vision and Pattern Recognition, pages 2582– 2593, 2022. 4

[45] Zhen Zhu, Mengde Xu, Song Bai, Tengteng Huang, and Xiang Bai. Asymmetric non-local neural networks for semantic segmentation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 593–602, 2019. 3

[46] Yang Zou, Zhiding Yu, BVK Kumar, and Jinsong Wang. Unsupervised domain adaptation for semantic segmentation via class-balanced self-training. In Proceedings of the European conference on computer vision (ECCV), pages 289– 305, 2018. 3

[47] Yang Zou, Zhiding Yu, Xiaofeng Liu, BVK Kumar, and Jinsong Wang. Confidence regularized self-training. In Proceedings of the IEEE/CVF international conference on computer vision, pages 5982–5991, 2019. 3