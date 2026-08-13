# Handling Spatial-Temporal Data Heterogeneity for Federated Continual Learning via Tail Anchor

Hao Yu   
Southwestern University of   
Finance and Economics   
yuhao2033@163.com

Hanlin Gu WeBank allengu@webank.com

Xin Yang<sup>\*</sup>   
Southwestern University of   
Finance and Economics   
yangxin@swufe.edu.cn

Tianrui Li Southwest Jiaotong University trli@swjtu.edu.cn

Qiang Yang Hong Kong University of Science and Technology

Le Zhang University of Electronic Science and Technology of China lezhang@uestc.edu.cn

qyang@cse.ust.hk

Lixin Fan WeBank lixinfan@webank.com

## Abstract

Federated continual learning (FCL) allows each client to continually update its knowledge from task streams, enhancing the applicability of federated learning in realworld scenarios. However, FCL needs to address not only spatial data heterogeneity between clients but also temporal data heterogeneity between tasks. In this paper, empirical experiments demonstrate that such input-level heterogeneity significantly affects the model’s internal parameters and outputs, leading to severe spatial-temporal catastrophic forgetting of local and previous knowledge. To this end, we propose Federated Tail Anchor (FedTA) to mix trainable Tail Anchor with thefrozen outputfeatures to adjust their position in the feature space, thereby overcoming parameter-forgetting and output-forgetting. Three novel components are also included: Input Enhancementfor improving the performance of pre-trained models on downstream tasks; Selective Input Knowledge Fusion for fusion of heterogeneous local knowledge on the server; and Best Global Prototype Selection for finding the best anchor point for each class in the feature space. Extensive experiments demonstrate that FedTA not only outperforms existing FCL methods but also effectively preserves the relative positions of features. Code is available at: https://github.com/SkyOfBeginning/FedTA CVPR2025.

## 1. Introduction

Data heterogeneity across different clients (Non-IID) is one of the most important challenges in traditional Federated Learning (FL), which greatly hinders the integration of knowledge, leading to the aggregated global model underperforming on local tasks. Many studies have attempted to address this issue and have made some progress [6, 37, 45]. However, they are based on an unrealistic static assumption that the training data of all clients will remain unchanged. Federated Continual Learning (FCL) breaks the static limits by allowing clients to continually accumulate knowledge from task sequences [1, 7, 19, 38]. While FCL expands the applicability of FL in real-world scenarios, it also introduces a more challenging issue, i.e., spatial-temporal data heterogeneity. Not only is the data heterogeneous across different clients (spatial), but also the data within different tasks of the same client is heterogeneous (temporal), as shown on the left side of Fig. 1.

The most direct negative impact of spatial-temporal data heterogeneity is spatial-temporal catastrophic forgetting (ST-CF) [36], as illustrated on the middle side of Fig. 1. Catastrophic Forgetting (CF) is a term from Continual Learning (CL), used to describe the phenomenon where a deep model, after learning multiple tasks, tends to forget the knowledge of previous tasks, resulting in a decrease in accuracy [8, 13, 23]. In FCL, clients face temporal catastrophic forgetting as local models would continually learn different tasks over time. Additionally, Non-IID data would lead the aggregated global model to spatial catastrophic forgetting (i.e., a decline in the performance of the global model on local test sets). Moreover, spatial forgetting will interact with temporal forgetting, as clients use the global model as the base model to learn the next task [36].

![](images/e88c266f2186279ab0d922489597ca43bebf5015a22f8a1c99c38558ae26a677.jpg)  
Figure 1: Illustration of FCL, the negative impact of spatial-temporal data heterogeneity and the intuition of Tail Anchor.

Spatial-temporal data heterogeneity, which manifests as differences at the model input, leads to corresponding changes in model parameters and outputs. We believe this is the fundamental cause of forgetting. To be specific, as data changes over time and space, feature extractor and classification head of the model will adapt to the most recent inputs, leading to forgetting of previous and client-specific knowledge (a detailed analysis of the effect of forgetting is presented in Fig. 1(a-b) and Sec. 3.2). Fortunately, the use of pre-trained large models can effectively mitigate catastrophic forgetting at the parameter level, as they have sufficient capacity to extract features without changing internal parameters [5, 30, 32]. However, frozen pre-trained models often perform poorly in downstream tasks, making them unsuitable for direct application [11]. Additionally, they cannot handle forgetting at the output since the classification head is trainable and will fit to the most recent task data.

To fully leverage the power of pre-trained models and address spatial-temporal forgetting from both the parameter and the output aspects, we first define Tail Anchors and mix them with frozen output features to fix the position, as shown in Fig. 1(c). Based on this concept, we propose Federated Tail Anchor (FedTA), which leverages the tail anchor to keep the positions of each class invariant.

Firstly, each client shares a pre-trained Vision Transformer (ViT) [4] and a two-stage training strategy is designed to enhance the performance of pre-trained models at the input level and alleviate forgetting at the output level. By adding tail anchors to the output features, the features of samples that experience forgetting can quickly return to their original positions in the feature space, thereby avoiding forgetting caused by spatial-temporal changes. After completing local training, the server will separately process the parameters added during input enhancement by the client and the local prototype. On the one hand, we design a selective input knowledge fusion mechanism to selectively integrate the knowledge used for input enhancement from different clients; on the other hand, the server will calculate the similarity between local prototypes to form a similarity adjacency matrix. In each iteration, the server will select the local prototype with the lowest average similarity for each class as the global prototype. If the average similarity falls below a threshold, the global prototype will be fixed to prevent forgetting.

Currently, there is little research on spatial-temporal heterogeneity in FCL [6]. To our knowledge, we are the first to attempt to address both temporal and spatial data heterogeneity from the perspective of forgetting. The main contributions can be summarized as:

1. Empirical experiments are conducted to show that spatial-temporal data heterogeneity can cause significant changes in the important features extracted by the model for the same samples, and it also causes shifts in their positions within the feature space. This leads to severe ST-CF of previous knowledge and local knowledge. We refer to these two phenomena as “parameter-forgetting” and “output-forgetting”, respectively.

2. FedTA leverages a pre-trained ViT along with four novel components, aiming to prevent features from shifting their positions in the feature space due to spatialtemporal data heterogeneity. FedTA efficiently overcomes both parameter-forgetting and output-forgetting in FCL caused by spatial-temporal data heterogeneity.

3. Extensive experiments not only demonstrate the state-ofthe-art performance of FedTA but also show its remarkable ability to resist spatial-temporal forgetting. Moreover, visualization results indicate that FedTA effectively preserves the relative positions of features, preventing positional shifts due to spatial-temporal variations.

## 2. Related Work

Spatial data heterogeneity, as known as the Non-IID problem, has attracted much attention [2, 12, 28, 43]. Existing methods tackle data heterogeneity by either incorporating more effective local training or devising more comprehensive aggregation mechanisms [16].

Although these studies have made progress in overcoming spatial data heterogeneity, they are unable to cope with more realistic and dynamic scenarios where each client continually learns on their own task stream.

FCL has indeed greatly enhanced the practical value of FL in real-world scenarios, especially on the edge computing side [35, 42, 44]. It allows each client to rapidly learn knowledge from the current task without forgetting previously knowledge, thus avoiding the need to retrain from scratch and greatly saving computational resources.

In a survey paper on FCL, the authors identified a key issue that existing FCL articles have overlooked: the interaction between spatial heterogeneity and temporal heterogeneity, which leads to a unique challenge: spatial-temporal catastrophic forgetting (ST-CF) [36]. It means that models not only forget previous knowledge due to continual learning but also forget local knowledge due to federated aggregation. Existing FCL methods do not realize that spatial heterogeneity can exacerbate the temporal forgetting, so when the spatial heterogeneity becomes stronger, the performance is not as expected. Besides, effective FCL methods currently heavily rely on replaying or generating pseudo data to mitigate the effects of spatial-temporal data heterogeneity [1, 18, 20, 22, 24, 41]. However, this may pose certain privacy risks and incur high computational cost.

Only a very small portion of work has attempted to address data heterogeneity from time and space simultaneously now [15, 39]. However, none of them have delved into how this heterogeneity leads to forgetting, nor have they ensured sufficiently strong spatial-temporal heterogeneity in their experimental settings. To our best knowledge, we are the first to deeply analyze how the heterogeneity of inputs affects model parameters and outputs.

## 3. Spatial-Temporal Data Heterogeneity

## 3.1. Problem Definition

The purpose of Spatial-Temporal Data Heterogeneity is to continually integrate knowledge from different clients and different time periods. We extend the traditional FL to FCL with strong spatial-temporal data heterogeneity.

For spatial heterogeneity, given a clients (denoted as $\begin{array} { r } {  { \textsf { A } } = \ \{ A _ { 1 } , A _ { 2 } , \ldots , A _ { a } \} ) } \end{array}$ , and a central server $S ,$ each client’s data is composed of private classes $C _ { v } ^ { i }$ and public classes $C _ { p } .$ , where private classes refer to the class of data that can only be seen by the client itself. We ensure that the data of $C _ { p }$ is non-overlapping between clients. Further, we can set $| \bar { C } _ { p } | = 0$ to ensure extreme spatial heterogeneity. We run experiments with $| C _ { p } | = 0$ on Imagenet-R dataset.

For temporal heterogeneity, the task sequence of client $A _ { i }$ is denoted as $\mathcal { T } _ { i } = \{ T _ { i } ^ { 1 } , T _ { i } ^ { 2 } , . . . , T _ { i } ^ { n _ { i } } \}$ , where $n _ { i }$ represents the total number of tasks on client $A _ { i }$ . Each task consists of the same number but entirely different classes.

During the training of task $^ { r , }$ the global model on the server already possesses the knowledge of $\mathbf { \boldsymbol { T } } _ { i } ^ { 1 }$ to $T _ { i } ^ { r - 1 }$ from client $\{ A _ { i } , 1 ~ \le ~ i ~ \le ~ a \}$ . The server S then distributes it back to clients. After personalizing the received global model $\theta _ { g } ^ { r - 1 }$ , the client $A _ { i }$ continually trains it on $T _ { i } ^ { r }$ as the initial model to get the new local model $\theta _ { i } ^ { r }$ . The local model $\theta _ { i } ^ { r }$ should perform well in classifying classes of $\{ \mathcal { T } _ { 1 } ^ { 1 } \cup \mathcal { T } _ { 1 } ^ { r - \bar { 1 } } \ldots , \mathcal { T } _ { i } ^ { 1 } \cup \mathcal { T } _ { i } ^ { r } \cup \ldots , \mathcal { T } _ { a } ^ { 1 } \cup \mathcal { T } _ { a } ^ { \bar { r } - 1 } \}$

## 3.2. Negative Impact

Spatial-temporal data heterogeneity is a type of heterogeneity in model inputs. Due to the back-propagation mechanism, it would significantly affect the internal parameters of the model and the outputs [29]. It not only introduces differences between models of different clients but also causes the features output by the same sample to undergo significant changes, thereby causing spatial-temporal forgetting of previous knowledge and local knowledge.

For the feature extractor, as shown on the left side of Fig. 2, traditional centralized single-task training can accurately extract beneficial features. However, after continual learning of four other tasks, for the same image, the extracted features are completely unrelated to the cat. Similarly, for FL, the aggregated global model also fails to extract features related to the cat itself. More critically, when the model encounters spatial-temporal data heterogeneity, the crucial features extracted by the feature extraction layer are predominantly concentrated at the edges of the images, which is highly negative to classification task. We term this phenomenon as parameter-forgetting.

For the output (feature space), as shown on the right side of Fig. 2, the features extracted by the initial model have clear classification boundaries, and the features of each class are relatively concentrated in the same area. However, after CL or $\mathrm { F L } ,$ the features extracted for the same samples no longer possess clear boundaries, especially in FCL. Moreover, the positions of the features gradually deviate from their original locations with the spatial-temporal variation, leading to the forgetting of old samples. We term this phenomenon as output-forgetting.

Let’s delve even further into the effects of spatialtemporal data heterogeneity. For deep neural networks, changes at the input level directly affect model parameters and corresponding outputs, thereby causing continual variations of the feature space. For spatial data heterogeneity, the absence of a common feature space among clients makes it challenging to share heterogeneous knowledge. For temporal data heterogeneity, changes in the feature space over time lead to variations in the locations of features of the same samples. If we can address the issues mentioned above simultaneously, then spatial-temporal catastrophic forgetting will be resolved.

![](images/c02a76c0dfc63452b3172534760c83aea4c7d0ac1b79b401c67f74ec28f19a90.jpg)  
Figure 2: Illustrations of the negative impact of spatial-temporal data heterogeneity on the feature extractor and feature space [Left Side] illustrates the variation of significant features extracted by the feature extractor for the same input sample, where brighter colors indicate more important features. As spatial-temporal changes occur, the extracted features gradually shif away from “cat”, even extracting features near the image edges. [Right Side] depicts the changes in the positions of the features in the feature space after undergoing spatial-temporal transformations for the same batch.

## 3.3. Motivation

It is evident that spatial-temporal data heterogeneity leads to both parameter-forgetting and output-forgetting. Therefore, methods need to possess the following three capabilities: (1) Ensure that the model extracts nearly identical features for the same sample; (2) Fix the positions of extracted features. (3) Allow clients to have a common feature space to better utilize heterogeneous knowledge.

However, due to the training method of deep networks and the large number of parameters involved, parameter updates are uncontrollable, making it nearly impossible to mitigate parameter-forgetting. Similarly, since ensuring consistency within the parameters is impossible, it is also hard to guarantee the invariance of outputs in the feature space.

Pre-trained large models have attracted attention due to their powerful representation capabilities. There are already articles attempting to apply pre-trained ViT to overcome forgetting in CL [21, 27]. Inspired by this, we find that freezing the feature extractor of pre-trained ViT can effectively eliminate parameter-forgetting. In FCL, clients share the same pre-trained model, ensuring that they have the same knowledge/feature space, which makes knowledge transfer between clients easier. Furthermore, by mixing learnable parameters (referred to “tail anchor” in this paper) with frozen features, we can effectively control their positions in the feature space, thus addressing output-forgetting. The server selects anchor points with the lowest similarity to other classes’ anchor points as the global anchor points in the feature space. During clientside training, the tail anchor converges towards these classspecific anchor points. Therefore, we mitigate the performance degradation caused by parameter-forgetting and output-forgetting.

## 4. Methodology: FedTA

To address the spatial-temporal catastrophic forgetting, which includes parameter-forgetting in the feature extractor and output-forgetting in the feature space, we propose FedTA. Its aim is to leverage a frozen pre-trained model and cross-mix learnable parameters after the output features, ensuring that the position of features in the feature space remains fixed and unaffected by spatial-temporal changes.

The overall framework of FedTA is shown in Fig. 3. and the algorithm is summarized in Appendix A.

## 4.1. Input Enhancement

We assign each client with the same pre-trained ViT model as a foundation model. With its parameters frozen, clients learn common knowledge that operate at the input level. The purpose is to extract knowledge into a common space through the same model and enhance ViT’s performance.

Knowledge Base. We devise a knowledge base for storing and selecting the input enhancement parameters. The knowledge base of client i is defined as

$$
\begin{array} { r } { K B _ { i } = \{ I E _ { 1 } , I E _ { 2 } , . . . , I E _ { M } \} , } \end{array}\tag{1}
$$

where M is the base size and IE is a set of learnable parameters. Then, let x and $E = f _ { e } ( x )$ be the input and its corresponding embedding feature, respectively. $f _ { e } ( \cdot )$ refers to the embedding function of ViT. Denoting $\{ s _ { i } \} _ { 1 } ^ { N }$ be the indices of selected N sets, then we can modify the embedding feature as follows:

$$
E ^ { \prime } = [ I E _ { s _ { 1 } } , \dots , I E _ { s _ { N } } ; E ] , 1 \le N \le M ,\tag{2}
$$

where [;] represents concatenation along the token length dimension. Each IE has a corresponding key, denoted as

![](images/c1e35ed42e277834c4d4b53e3dcb019400f3488a08a6e0c2c677eb2f210f53cc.jpg)  
Figure 3: An overview of FedTA. Local training is a two-stage training process. The first stage involves adding input enhancement to the image embeddings to fully utilize ViT (see 1). In the second stage, the extracted features are fixed, and the corresponding tail anchor is mixed with them to adjust the similarity between classes by applying contrastive learning with global prototypes (see 2). Then, the local knowledge base of input enhancements and the local prototypes of each class are uploaded to the server, where selective input knowledge fusion for the knowledge base (see 4) and global best prototype selection for the local prototypes (see 3) are performed, respectively.

$K ^ { i e }$ , to facilitate the selection of the IE based on the similarity of keys.

Optimization for the input enhancement. Each client has a classification head used for training input enhancement parameters, denoted as $H _ { e } ^ { i }$ . At the beginning of training, it is necessary to load the pre-trained model with $H _ { e } ^ { i }$ to enable it to perform the classification task, and we denote the model with $H _ { e } ^ { i }$ as $\mathcal { V } _ { e } ^ { i }$ . Overall, the training loss function is as follows:

$$
\operatorname* { m i n } \mathcal { L } ( \mathcal { V } _ { e } ^ { i } ( E ^ { \prime } ) , y ) + \lambda _ { 1 } \sum _ { K _ { s } ^ { i e } } \mathrm { d i s } ( K _ { i n } ^ { i e } , K { s _ { i } } ^ { i e } ) ,\tag{3}
$$

where $\lambda _ { 1 }$ is a hyperparameter, $K _ { s } ^ { i e }$ and $K _ { i n } ^ { i e }$ are used to find the best $I E$ . The initial term comprises the softmax crossentropy loss to optimize selected IEs, while the subsequent term serves as a surrogate loss aimed at bringing selected keys closer to their corresponding query features. Cosine similarity is used as the distance function.

In a nutshell, IE is a set of learnable parameters that can be concatenated to the image embedding E to enhance the performance of ViT. Each IE has a corresponding key $K ^ { i e }$ . E is first processed by ViT to obtain features, which are then used as a query key $K _ { i n } ^ { i e }$ to calculate the similarity with the key of each $I E .$ . Then best $I E _ { s }$ is selected to form a enhance embedding $E ^ { \prime }$

## 4.2. Tail Anchor

Query function. Once the input enhancement parameters are well trained, they will be frozen, including their corresponding keys, until the next task training. The enhanced input embedding $E ^ { \prime }$ would be processed by the frozen ViT [4] again to get the features, denoted as $\mathcal { F } _ { o u t }$ . Then it will be used as the key to find the corresponding tail anchor based on the cosine similarity.

Tail Anchor is defined as key-value pairs for m classes: $\begin{array} { r l r } { \mathcal { T } A } & { { } = } & { \{ ( K _ { 1 } ^ { t a } , T A _ { 1 } ) , ( K _ { 2 } ^ { t \bar { a } } , T A _ { 2 } ) , \cdot . . . , ( K _ { m } ^ { t a } , T A _ { m } ) \} } \end{array}$ Specifically, $K _ { s } ^ { t a } , s \in [ m ]$ is obtained as:

$$
K _ { s } ^ { t a } = \underset { K ^ { t a } } { \operatorname { a r g m i n } } \mathrm { d i s } ( \mathcal { F } _ { o u t } , K _ { i } ^ { t a } ) ,\tag{4}
$$

where $K _ { s } ^ { t a }$ denotes the chosen tail anchor’s key, and ${ \cal { K } } ^ { t a }$ represents the set of keys for all tail anchors.

In a nutshell, the tail anchor of each class acts as additional parameters to manage the distance to a fixed position (global prototype of each class) in the feature space. Its main advantage is its relative position, which remains consistent regardless of changes in space and time. This consistency ensures that the tail anchor can effectively guide the positioning of general output features.

Optimization for the tail anchor. Once the tail anchor is chosen, it will be mixed with $\mathcal { F } _ { o u t }$ to form a new feature $\mathcal { F } _ { T A }$ . If a client has global prototypes (i.e., not the first round), then contrastive learning is utilized to unify the features across clients through the following unified representation loss function:

$$
\mathcal { L } _ { c o n s } \left( \mathcal { F } _ { T A } \right) = - \log \frac { \exp \left( \mathcal { F } _ { T A } \cdot \mathcal { G } ^ { y } / \tau \right) } { \sum _ { y _ { a } \in \mathcal { D } ^ { t } } \exp \left( \mathcal { F } _ { T A } \cdot \mathcal { G } ^ { y _ { a } / \tau } \right) } ,\tag{5}
$$

where $\mathcal { V } ^ { t }$ represents the global available classes up to task t and $\mathcal { G } ^ { y }$ represents the global prototypes of class y. τ denotes the temperature that controls the tolerance of difference between extracted features and the corresponding global prototype. The overall loss function to optimize the tail anchor can be formulated as follows:

$$
\mathcal { L } _ { t a } = \mathcal { L } _ { C E } ( \mathcal { F } _ { T A } ) + \lambda _ { 2 } \mathcal { L } _ { c o n s } \left( \mathcal { F } _ { T A } \right) + \lambda _ { 3 } \operatorname { d i s } ( \mathcal { F } _ { T A } , K _ { s } ^ { t a } ) ,\tag{6}
$$

where $\mathcal { L } _ { C E }$ is the standard cross-entropy loss, the second term is to adjust its position in the feature space through contrastive loss with the global prototypes (fixed positions). The last term is to enhance the similarity between the selected key and the query key.

Local prototypes. Once the training process of the tail anchor is done, the tail anchors will be frozen and remain unchanged. The local prototype is obtained by averaging features with tail anchors belonging to the same class, computed through

$$
P _ { i } ^ { y } = \frac { 1 } { | \mathcal { D } _ { a } ^ { y } | } \sum _ { ( x , y ) \in \mathcal { D } _ { a } ^ { y } } \mathcal { F } _ { T A } ^ { x } ,\tag{7}
$$

where $\mathcal { D } _ { a } ^ { y }$ denotes the subset of private dataset of client a of class y. Each client forms a local set of prototypes, which is then uploaded to the server. The server iteratively selects the prototype with the lowest average similarity as the global prototype for that class.

## 4.3. Selective Input Knowledge Fusion

We follow a common setting, which allows the server to possess a small-scale surrogate dataset, denoted as $\mathcal { D } _ { s } .$ $\{ x _ { s } , y _ { s } \}$ are the samples and corresponding labels from $\mathcal { D } _ { s }$ for the distillation process. Assuming the total number of KB is $n , K B _ { i }$ is randomly selected as the target for distillation. $E _ { i } ^ { \prime }$ represents the enhanced embedding formed by concatenating the $x _ { s } \mathrm { { ^ { * { s } } } }$ embedding with the IE from $\kappa B _ { i } .$ where $\mathcal { V } ( \cdot )$ denotes the ViT’s feature extraction process. Therefore, the distillation loss can be formulated as follows:

$$
\mathcal { L } _ { K D } = \frac { 1 } { n - 1 } \sum _ { \underset { j \neq i } { \sum } } ^ { n } \underset { x _ { s } \in \mathcal { D } _ { s } } { \operatorname { M S E } } ( \mathcal { V } ( E _ { i } ^ { \prime } ) , \mathcal { V } ( E _ { j } ^ { \prime } ) ) .\tag{8}
$$

## 4.4. Best Global Prototype Selection

When the server receives local prototype sets from different clients, it reorders them to form a new set $P _ { G }$ according to the class. Specifically, when two clients both have prototypes related to class $q ,$ denoted as $P _ { i } ^ { q }$ and $P _ { j } ^ { q }$ , they will be adjacent to each other in the reordered prototype set. Then, the server computes the similarity between each pair of sets in the collection, forming an adjacency matrix M. The element of M is computed through:

$$
\mathcal { M } _ { i j } = \mathrm { d i s } ( P _ { G } ^ { i } , P _ { G } ^ { j } ) , 0 < i \leq j \leq | P _ { G } | .\tag{9}
$$

Notice that if $P _ { G } ^ { i }$ and $P _ { G } ^ { j }$ belong to same class, then $\mathcal { M } _ { i j } =$ 1. In each round, the server selects the prototype with the lowest average similarity with all local prototypes as the global prototype for one class. The selection process of the global prototype $\mathcal { G } ^ { y }$ for class y can be expressed as follows:

$$
\mathcal { G } ^ { y } = P _ { g } ^ { s } = \operatornamewithlimits { a r g m i n } _ { y _ { l o w } \leq i \leq y _ { h i g h } } \bar { \mathcal { M } } _ { i } = \frac { 1 } { | P _ { G } | } \sum _ { j = 1 } ^ { | P _ { G } | } A _ { i j } ,\tag{10}
$$

where $y _ { l o w }$ and $y _ { h i g h }$ are the start index and end index of the local prototypes of class y in $P _ { G } . \ P _ { g } ^ { s }$ is the local prototype who has lowest similarity for class y. If the average similarity $\bar { \mathcal { M } } _ { i }$ falls below the threshold Thr during the iteration process, then that prototype is fixed as the global prototype for its class and will not be altered further. As a result, this global prototype will serve as a fixed anchor point for that class in the feature space.

## 5. Experiments

## 5.1. Setup

Continual Setting. To ensure that the impact of spatialtemporal data heterogeneity is adequately reflected, we partition the data as follows: Each client continually learn from a task sequence of 5 tasks, and there are 5 clients in the experimental setting. For CIFAR-100 [14], we allow each client to have access to 15 private classes exclusive to itself, resulting in 25 public classes. Thus, each client has data for a total of 40 classes, with each task consisting of only 8 classes. For ImageNet-R [9], we exacerbate spatial data heterogeneity by assigning 40 private classes to each client, with no public classes across clients. Similarly, each task consists of 8 classes. Notice that we use the Dirichlet distribution for the public classes to assign data, ensuring there is no overlap between different clients.

Surrogate Data. We follow the common setting [10, 25] where the server possesses a small surrogate dataset. For CIFAR-100, each class has only 20 samples, while for ImageNet-R, each class has only 5 samples.

Brief introductions of baseline methods and implementation details are shown in Appendix B.

## 5.2. Metrics

To verify whether the method can effectively address the challenges brought by spatial-temporal data heterogeneity, we use two new metrics from [36] to evaluate the performance of mitigating forgetting.

Definition 1. (Temporal Knowledge Retention):

$$
K R _ { t } = \frac { 1 } { a } \sum _ { i = 1 } ^ { a } \frac { A c c ( \theta _ { i } ^ { r } ; T _ { i } ^ { 0 } ) } { A c c ( \theta _ { i } ^ { 0 } ; T _ { i } ^ { 0 } ) } ,\tag{11}
$$

where $A c c ( \theta _ { i } ^ { r } ; T _ { i } ^ { 0 } )$ denotes the test accuracy of client $A _ { i } ^ { \cdot } \mathrm { s }$ local model at r-th round on the 0-th task and $A c c ( \theta _ { i } ^ { 0 } ; T _ { i } ^ { 0 } )$

Table 1: Accuracy of the aggregated global model on local test sets with 5 class-incremental tasks.
<table><tr><td rowspan="2">Algorithm</td><td rowspan="2">Type</td><td colspan="4">CIFAR-100 Task ID</td><td colspan="5">Imagenet-R Task ID</td></tr><tr><td>1 2</td><td>3</td><td></td><td>4</td><td>5</td><td>1</td><td>2 3</td><td>4</td><td>5</td></tr><tr><td>FedAvg [26]</td><td rowspan="3">FL</td><td>43.9</td><td>50.6</td><td>57.3</td><td>55.5</td><td>61.2</td><td>37.7</td><td>35.4</td><td>35.5</td><td>35.8</td><td>36.7</td></tr><tr><td>FedProx [17]</td><td>23.7</td><td>22.8</td><td>26.0</td><td>22.1</td><td>23.6</td><td>20.2</td><td>19.7</td><td>19.7</td><td>18.9</td><td>17.8</td></tr><tr><td>FedNova [31]</td><td>13.7</td><td>18.8</td><td>20.1</td><td>16.1</td><td>15.1</td><td>2.0</td><td>4.7</td><td>4.6</td><td>7.8</td><td>7.7</td></tr><tr><td rowspan="4">FedLwF [23] FedViT [4] FedL2P [32] FedDualP [33]</td><td rowspan="4">FL+CL</td><td>36.9</td><td>12.5</td><td>17.1</td><td>13.6</td><td>9.7</td><td>5.9</td><td>7.0</td><td>2.6</td><td>4.0</td><td>3.8</td></tr><tr><td>70.2</td><td>70.0</td><td>71.4</td><td>66.0</td><td>67.3</td><td>68.2</td><td>59.8</td><td>57.3</td><td>59.8</td><td>57.9</td></tr><tr><td>28.4</td><td>29.9</td><td>29.3</td><td>25.4</td><td>25.7</td><td>27.1</td><td>27.6</td><td>24.8</td><td>25.1</td><td>26.5</td></tr><tr><td>31.7</td><td>42.8</td><td>52.8</td><td>39.0</td><td>46.3</td><td>23.5</td><td>26.6</td><td>26.4</td><td>30.2</td><td>32.0</td></tr><tr><td>GLFC [3]</td><td rowspan="4">FCL</td><td>82.0</td><td>63.1</td><td>73.4</td><td>64.2</td><td>64.8</td><td>61.9</td><td>67.1</td><td>67.0</td><td>71.7</td><td>57.2</td></tr><tr><td>TARGET [41]</td><td>54.0</td><td>41.4</td><td>32.2</td><td>13.9</td><td>15.9</td><td>39.9</td><td>15.0</td><td>16.0</td><td>17.5</td><td>16.1</td></tr><tr><td>MFCL [1]</td><td>46.7</td><td>16.2</td><td>10.6</td><td>14.6</td><td>13.5</td><td>28.8</td><td>14.5</td><td>16.2</td><td>13.3</td><td>13.8</td></tr><tr><td>FedMGP [40]</td><td>90.2</td><td>85.3</td><td>90.7</td><td>89.2</td><td>82.2</td><td>77.3</td><td>76.8</td><td>78.0</td><td>75.6</td><td>75.4</td></tr><tr><td>Ours (FedTA)</td><td rowspan="4">FCL</td><td>96.1</td><td>94.0</td><td>94.6</td><td>94.4</td><td>89.4</td><td>81.5</td><td>78.8</td><td>79.2</td><td></td><td>80.6</td><td>85.0</td></tr><tr><td>Ours-w/o TA</td><td>78.7</td><td>75.5</td><td>73.4</td><td>69.3</td><td>72.3</td><td>79.6</td><td>72.3</td><td>72.3</td><td>74.1</td><td></td><td>63.2</td></tr><tr><td>Ours-w/o SIKF</td><td>90.7</td><td>88.8</td><td>91.1</td><td>91.4</td><td>89.1</td><td>80.0</td><td></td><td>80.5</td><td>81.1</td><td>82.9</td><td>81.7</td></tr><tr><td>Ours-w/o BGPS</td><td>90.8</td><td>92.5</td><td>88.4</td><td>91.4</td><td>88.6</td><td>80.5</td><td></td><td>78.7</td><td>78.3</td><td>81.7</td><td>79.1</td></tr></table>

denotes the accuracy of client $A _ { i } ^ { \cdot } \mathrm { s }$ local model at the initial round on the 0-th task.

Definition 2. (Spatial Knowledge Retention):

$$
K R _ { s } = \frac { 1 } { a } \sum _ { i = 1 } ^ { a } \frac { A c c ( \theta _ { g } ^ { r } ; T _ { i } ^ { r } ) } { A c c ( \theta _ { i } ^ { r } ; T _ { i } ^ { r } ) } ,\tag{12}
$$

where $A c c ( \theta _ { g } ^ { r } ; T _ { i } ^ { r } )$ denotes the accuracy of the global model $\theta _ { g } ^ { r }$ on the current local task $T _ { i } ^ { r }$ at client $A _ { i }$ and $A c c ( \theta _ { i } ^ { r } ; \breve { T } _ { i } ^ { r } )$ denotes the accuracy of the local model $\theta _ { i } ^ { r }$ on its current local task $T _ { i } ^ { r }$

## 5.3. Results & Ablation Study

![](images/cb5ed180ec87732b8586224f1d39432b8c69ddb9e68d4122807dd482cce7e91e.jpg)  
(a) K ${ \mathrm { R } } _ { s }$ on CIFAR-100.

![](images/28b89270db65ef5cc05b081a08106cf8261663b5e2cbfbe1fa7a4c01aea01c71.jpg)  
(b) K $R _ { t }$ on CIFAR-100.

![](images/b298db7662730b0a2bdb067b4d62c7413b8fadaa9dd41eae9b094b458097689a.jpg)  
(c) KR<sub>s</sub> on ImageNet-R.

![](images/e3731cbbe4215bab8514597af0730df125cfcb167deae35c28bae4bb0f9113a2.jpg)  
(d) KR<sub>t</sub> on ImageNet-R.  
Figure 4: Knowledge retention on different dataset.

Table 1 illustrates the average accuracy of the aggregated global model on local test sets. The performance of Fed-ViT is acceptable because all the parameters of its feature extractor are frozen, and only the classification head is involved in training and aggregation. However, it still experiences a certain degree of forgetting. FedL2P and FedDualP, which introduce trainable parameters on the input side and within the model, perform very well on the local side, achieving around 90% accuracy. However, as we concluded in Sec. 3.2, almost all trainable parameters are directly affected by the data. Consequently, after aggregation, there is significant forgetting on the local test sets.

Surprisingly, the performance of TARGET, FedLwF and MFCL, the three baseline methods that use replay data to mitigate forgetting, is extremely poor. We speculate that the large size of data $( 3 \times 2 2 4 \times 2 2 4 )$ results in the low quality of replayed pseudo-samples. Moreover, replay-based methods pose a certain risk of privacy leakage in federated learning, limiting the further application of these methods in realworld scenarios. GLFC also suffers significant performance degradation when faced with severe spatial-temporal data heterogeneity. However, its performance in Table 1 remains the best among the baseline methods.

FedTA demonstrates the superior performance in these two settings, indicating its successful mitigation of the impact of spatial data heterogeneity. Furthermore, ablation studies highlight the effectiveness of the proposed novel components, with the Tail Anchor contributing the most to the performance improvement. However, the selective input knowledge fusion at the server-side sometimes falls below the results of direct weighted averaging on ImageNet-R. We believe this is due to insufficient surrogate data, which prevents adequate selective knowledge fusion.

![](images/5eb69844a336e9ff33ea937f0a225d54fb30019f6ba65b9ed4363f4d3cd31f24.jpg)  
Figure 5: T-SNE for position changes of features corresponding to the same samples after FCL.

Fig. 4 further illustrates the impact of spatial-temporal data heterogeneity on the methods using Equ. 12 and Equ. 11. Only FedTA maintains a high level of temporal and spatial knowledge retention, both at around 98%. While other baseline methods, especially in KR<sub>t</sub>, perform extremely poorly. Tail Anchor also has been verified to play a significant role in overcoming ST-CF.

Visualization & Sensitivity Analysis. Fig. 5 illustrates that FedTA can effectively control the relative distances between features. When the number of Tail Anchors is set to 100, regardless of the quantity of Input Enhancements, the features of a portion of the data still deviate from their original positions. We speculate that this is due to the insufficient number of Tail Anchors, causing samples from the same class to match with completely dissimilar Tail Anchors. When we set the number of Tail Anchors to 500, the number of shifted points significantly decreases. The combination of 10 Input Enhancements and 500 Tail Anchors shows the most satisfactory results.

Detailed analysis is in Appendix D.

## 5.4. Privacy & Efficiency Analysis

Computational Burden. During the local training phase, clients train both the Input Enhancement and Tail Anchor components while the ViT itself remains frozen. Therefore, the number of parameters in these two components, along with the classification head, determines the training overhead of FedTA. The size of Input Enhancement is determined by the number, length and embedding dimension, which are set to 10, 10, and 768 in our setting. The size of the Tail Anchor is set to 100×768. The total size of keys is (100+10)×768. Therefore, the total number of trainable parameters amounts to 253,440. Compared to a ResNet-18 with 11,306,804 parameters, FedTA is efficient.

Communication Cost. Each client only needs to submit its own input enhancement and local prototypes to the server, with sizes of 76,800 and 768×2 per class, respectively.

Such small communication cost makes FedTA highly efficient, and also makes FedTA scalable for multi-clients.

Privacy Protection. For Input Enhancement, on the one hand, ViT is frozen, and on the other hand, due to its minimal number of parameters, it contains extremely little information. Moreover, since this method does not use replay data to alleviate forgetting, privacy protection is further strengthened. However, the uploaded local prototypes are class-specific, and employing cross-mixing might easily reveal the original features, posing a certain degree of privacy risk. If we randomly mix Tail Anchor with features, this issue will be resolved.

Please refer to Appendix E for experimental results in scenarios with 10 tasks and extra baselines.

## 6. Conclusion

This article extends the issue of data heterogeneity in static FL to the more realistic problem of spatial-temporal data heterogeneity in FCL. Empirical experiments are conducted to demonstrate that spatial-temporal data heterogeneity can cause parameter-forgetting and output-forgetting. Based on this finding, we first define the representative feature embedding of each class as the tail anchor. Then we propose FedTA by utilizing a frozen pre-trained ViT to mitigate parameter forgetting and combining Tail Anchors. Just as a ship in the ocean requires an anchor to hold its position, the features of samples also need a fixed location in the feature space to remain unaffected by the variations introduced by spatial-temporal data heterogeneity, thereby avoiding forgetting. Extensive experiments have verified the superiority of our method, and ablation studies demonstrate the effectiveness of each component, especially Tail Anchor. Visualized results demonstrate that our method effectively fixes the features’ relative positions, preventing them from being affected by spatial-temporal changes.

## Acknowledgment

This work was supported by the National Natural Science Foundation of china (No. 62476228), the Sichuan Science and Technology Program (No. 2024ZYD0180), the Graduate Representative Achievement Cultivation Project of Southwestern University of Finance and Economics (Nos. JGS2024065, JGS2024004) and the Student Study Abroad Exchange Funding Program of Southwestern University of Finance and Economics.

## References

[1] Sara Babakniya, Zalan Fabian, Chaoyang He, Mahdi Soltanolkotabi, and Salman Avestimehr. A data-free approach to mitigate catastrophic forgetting in federated class incremental learning for vision tasks. Advances in Neural Information Processing Systems, 36, 2024. 1, 3, 7

[2] Michael Crawshaw, Yajie Bao, and Mingrui Liu. Federated learning with client subsampling, data heterogeneity, and unbounded smoothness: A new algorithm and lower bounds. Advances in Neural Information Processing Systems, 36, 2024. 2

[3] Jiahua Dong, Lixu Wang, Zhen Fang, Gan Sun, Shichao Xu, Xiao Wang, and Qi Zhu. Federated class-incremental learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10164–10173, 2022. 7, 1, 3

[4] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020. 2, 5, 7, 1

[5] Beyza Ermis, Giovanni Zappella, Martin Wistuba, Aditya Rawal, and Cedric Archambeau. Continual learning with ´ transformers for image classification. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3774–3781, 2022. 2

[6] Tao Fan, Hanlin Gu, Xuemei Cao, Chee Seng Chan, Qian Chen, Yiqiang Chen, Yihui Feng, Yang Gu, Jiaxiang Geng, Bing Luo, et al. Ten challenging problems in federated foundation models. arXiv preprint arXiv:2502.12176, 2025. 1, 2

[7] Xin Gao, Xin Yang, Hao Yu, Yan Kang, and Tianrui Li. Fedprok: Trustworthy federated class-incremental learning via prototypical feature knowledge transfer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4205–4214, 2024. 1

[8] Raia Hadsell, Dushyant Rao, Andrei A Rusu, and Razvan Pascanu. Embracing change: Continual learning in deep neural networks. Trends in cognitive sciences, 24(12):1028– 1040, 2020. 1

[9] Dan Hendrycks, Steven Basart, Norman Mu, Saurav Kadavath, Frank Wang, Evan Dorundo, Rahul Desai, Tyler Zhu, Samyak Parajuli, Mike Guo, et al. The many faces of robustness: A critical analysis of out-of-distribution generalization.

In Proceedings of the IEEE/CVF international conference on computer vision, pages 8340–8349, 2021. 6, 1

[10] Wenke Huang, Mang Ye, and Bo Du. Learn from others and be yourself in heterogeneous federated learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10143–10153, 2022. 6

[11] Yan Kang, Tao Fan, Hanlin Gu, Lixin Fan, and Qiang Yang. Grounding foundation models through federated transfer learning: A general framework. arXiv preprint arXiv:2311.17431, 2023. 2

[12] Taehyeon Kim, Eric Lin, Junu Lee, Christian Lau, and Vaikkunth Mugunthan. Navigating data heterogeneity in fed erated learning: a semi-supervised federated object detection. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. 2

[13] James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, et al. Overcoming catastrophic forgetting in neural networks. Proceedings of the national academy of sciences, 114(13):3521–3526, 2017. 1

[14] Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009. 6, 1

[15] Baoxue Li, Pengyu Song, Chunhui Zhao, and Min Xie. Facing spatiotemporal heterogeneity: A unified federated continual learning framework with self-challenge rehearsal for industrial monitoring tasks. Knowledge-Based Systems, 289: 111491, 2024. 3

[16] Qinbin Li, Yiqun Diao, Quan Chen, and Bingsheng He. Fed erated learning on non-iid data silos: An experimental study. In 2022 IEEE 38th international conference on data engineering (ICDE), pages 965–978. IEEE, 2022. 2

[17] Tian Li, Anit Kumar Sahu, Manzil Zaheer, Maziar Sanjabi, Ameet Talwalkar, and Virginia Smith. Federated optimiza tion in heterogeneous networks. Proceedings of Machine learning and systems, 2:429–450, 2020. 7, 1, 3

[18] Yichen Li, Qunwei Li, Haozhao Wang, Ruixuan Li, Wenliang Zhong, and Guannan Zhang. Towards efficient replay in federated incremental learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12820–12829, 2024. 3

[19] Yichen Li, Haozhao Wang, Wenchao Xu, Tianzhe Xiao, Hong Liu, Minzhu Tu, Yuying Wang, Xin Yang, Rui Zhang, Shui Yu, Song Guo, and Ruixuan Li. Unleashing the power of continual learning on non-centralized devices: A survey, 2024. 1

[20] Yichen Li, Wenchao Xu, Haozhao Wang, Yining Qi, Ruixuan Li, and Song Guo. Sr-fdil: Synergistic replay for federated domain-incremental learning. IEEE Transactions on Parallel and Distributed Systems, 2024. 3

[21] Yujie Li, Xin Yang, Hao Wang, Xiangkun Wang, and Tianrui Li. Learning to prompt knowledge transfer for open-world continual learning. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 13700–13708, 2024. 4

[22] Yichen Li, Wenchao Xu, Haozhao Wang, Yining Qi, Jing cai Guo, and Ruixuan Li. Personalized federated domain incremental learning based on adaptive knowledge match

ing. In European Conference on Computer Vision, pages 127–144. Springer, 2025. 3

[23] Zhizhong Li and Derek Hoiem. Learning without forgetting. IEEE transactions on pattern analysis and machine intelligence, 40(12):2935–2947, 2017. 1, 7

[24] Jinglin Liang, Jin Zhong, Hanlin Gu, Zhongqi Lu, Xingxing Tang, Gang Dai, Shuangping Huang, Lixin Fan, and Qiang Yang. Diffusion-driven data replay: A novel approach to combat forgetting in federated class continual learning. In European Conference on Computer Vision, pages 303–319. Springer, 2025. 3

[25] Yuhang Ma, Zhongle Xie, Jue Wang, Ke Chen, and Lidan Shou. Continual federated learning based on knowledge distillation. In IJCAI, pages 2182–2188, 2022. 6

[26] Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. Communicationefficient learning of deep networks from decentralized data. In Artificial intelligence and statistics, pages 1273–1282. PMLR, 2017. 7, 1, 3

[27] Francesco Pelosin, Saurav Jha, Andrea Torsello, Bogdan Raducanu, and Joost van de Weijer. Towards exemplar-free continual learning in vision transformers: an account of attention, functional and weight regularization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3820–3829, 2022. 4

[28] Sara Pieri, Jose Restom, Samuel Horvath, and Hisham Cholakkal. Handling data heterogeneity via architectural design for federated visual recognition. Advances in Neural Information Processing Systems, 36:4115–4136, 2023. 2

[29] David E Rumelhart, Geoffrey E Hinton, and Ronald J Williams. Learning representations by back-propagating errors. nature, 323(6088):533–536, 1986. 3

[30] James Seale Smith, Junjiao Tian, Shaunak Halbe, Yen-Chang Hsu, and Zsolt Kira. A closer look at rehearsal-free continual learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2409–2419, 2023. 2

[31] Jianyu Wang, Qinghua Liu, Hao Liang, Gauri Joshi, and H Vincent Poor. Tackling the objective inconsistency problem in heterogeneous federated optimization. Advances in neural information processing systems, 33:7611–7623, 2020. 7, 1

[32] Zhen Wang, Liu Liu, Yiqun Duan, Yajing Kong, and Dacheng Tao. Continual learning with lifelong vision transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 171–181, 2022. 2, 7, 3

[33] Zifeng Wang, Zizhao Zhang, Sayna Ebrahimi, Ruoxi Sun, Han Zhang, Chen-Yu Lee, Xiaoqi Ren, Guolong Su, Vincent Perot, Jennifer Dy, et al. Dualprompt: Complementary prompting for rehearsal-free continual learning. In European Conference on Computer Vision, pages 631–648. Springer, 2022. 7, 1, 3

[34] Zifeng Wang, Zizhao Zhang, Chen-Yu Lee, Han Zhang, Ruoxi Sun, Xiaoqi Ren, Guolong Su, Vincent Perot, Jennifer Dy, and Tomas Pfister. Learning to prompt for continual learning. In Proceedings of the IEEE/CVF Conference on

Computer Vision and Pattern Recognition, pages 139–149, 2022. 1

[35] Zichuan Xu, Lin Wang, Weifa Liang, Qiufen Xia, Wenzheng Xu, Pan Zhou, and Omer F Rana. Age-aware data selection and aggregator placement for timely federated continual learning in mobile edge computing. IEEE Transactions on Computers, 2023. 3

[36] Xin Yang, Hao Yu, Xin Gao, Hao Wang, Junbo Zhang, and Tianrui Li. Federated continual learning via knowledge fusion: A survey. IEEE Transactions on Knowledge and Data Engineering, 2024. 1, 2, 3, 6

[37] Zhiqin Yang, Yonggang Zhang, Yu Zheng, Xinmei Tian, Hao Peng, Tongliang Liu, and Bo Han. Fedfed: Feature distillation against data heterogeneity in federated learning. Advances in Neural Information Processing Systems, 36, 2024. 1

[38] Jaehong Yoon, Wonyong Jeong, Giwoong Lee, Eunho Yang, and Sung Ju Hwang. Federated continual learning with weighted inter-client transfer. In International Conference on Machine Learning, pages 12073–12086. PMLR, 2021. 1

[39] Hao Yu, Xin Yang, Xin Gao, Yihui Feng, Hao Wang, Yan Kang, and Tianrui Li. Overcoming spatial-temporal catastrophic forgetting for federated class-incremental learning. In ACM Multimedia 2024, 2024. 3

[40] Hao Yu, Xin Yang, Xin Gao, Yan Kang, Hao Wang, Junbo Zhang, and Tianrui Li. Personalized federated continual learning via multi-granularity prompt. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pages 4023–4034, 2024. 7, 1

[41] Jie Zhang, Chen Chen, Weiming Zhuang, and Lingjuan Lyu. Target: Federated class-continual learning via exemplar-free distillation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4782–4793, 2023. 3, 7, 1

[42] Zhouyangzi Zhang, Bin Guo, Wen Sun, Yan Liu, and Zhiwen Yu. Cross-fcl: Toward a cross-edge federated contin ual learning framework in mobile edge computing systems. IEEE Transactions on Mobile Computing, 23(1):313–326, 2022. 3

[43] Yue Zhao, Meng Li, Liangzhen Lai, Naveen Suda, Damon Civin, and Vikas Chandra. Federated learning with non-iid data. arXiv preprint arXiv:1806.00582, 2018. 2

[44] Zhengyi Zhong, Weidong Bao, Ji Wang, Xiaomin Zhu, and Xiongtao Zhang. Flee: A hierarchical federated learning framework for distributed deep neural network over cloud, edge, and end device. ACM Transactions on Intelligent Sys tems and Technology (TIST), 13(5):1–24, 2022. 3

[45] Zhengyi Zhong, Ji Wang, Weidong Bao, Jingxuan Zhou, Xiaomin Zhu, and Xiongtao Zhang. Semi-hfl: semi-supervised federated learning for heterogeneous devices. Complex & Intelligent Systems, 9(2):1995–2017, 2023. 1