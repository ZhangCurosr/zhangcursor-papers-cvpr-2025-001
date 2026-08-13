# MMTL-UniAD: A Unified Framework for Multimodal and Multi-Task Learning in Assistive Driving Perception

Wenzhuo Liu<sup>1</sup> Wenshuo Wang<sup>\*,1</sup> Yicheng Qiao<sup>2</sup> Qiannan Guo<sup>2</sup> Jiayin Zhu<sup>3</sup> Pengfei Li<sup>2</sup> Zilong Chen<sup>2</sup> Huiming Yang<sup>2</sup> Zhiwei Li<sup>4</sup> Lening Wang<sup>5</sup> Tiao Tan<sup>2</sup> Huaping Liu<sup>2</sup>

<sup>1</sup>Beijing Institute of Technology, Zhuhai <sup>2</sup>Tsinghua University <sup>3</sup>HKUST(GZ) Beijing University of Chemical Technology <sup>5</sup>Beihang University

## Abstract

Advanced driver assistance systems require a comprehensive understanding ofthe driver’s mental/physical state and traffic context but existing works often neglect the potential benefits of joint learning between these tasks. This paper proposes MMTL-UniAD, a unified multi-modal multitask learning framework that simultaneously recognizes driver behavior (e.g., looking around, talking), driver emotion (e.g., anxiety, happiness), vehicle behavior (e.g., parking, turning), and traffic context (e.g., traffic jam, traffic smooth). A key challenge is avoiding negative transfer between tasks, which can impair learning performance. To address this, we introduce two key components into the framework: one is the multi-axis region attention network to extract global context-sensitive features, and the other is the dual-branch multimodal embedding to learn multimodal embeddings from both task-shared and task-specific features. The former uses a multi-attention mechanism to extract task-relevant features, mitigating negative transfer caused by task-unrelated features. The latter employs a dual-branch structure to adaptively adjust task-shared and task-specific parameters, enhancing cross-task knowledge transfer while reducing task conflicts. We assess MMTL-UniAD on the AIDE dataset, using a series of ablation studies, and show that it outperforms state-of-the-art methods across all four tasks. The code is available on https: //github.com/Wenzhuo-Liu/MMTL-UniAD.

## 1. Introduction

Over the past decade, Advanced Driver Assistance Systems (ADAS) — such as automatic emergency braking, lane keeping assist, and surround view monitoring — have significantly enhanced driving safety through the advances in monitoring driver states and surrounding traffic [6, 31, 55,

![](images/a80cd8ecc13afb0bba5a86badae310b3fe675d35d179253339f7ede0c663852a.jpg)  
Figure 1. Traffic context and driver states interaction diagram. Tasks (a), (b), (c), and (d) represent traffic context recognition, vehicle behavior recognition, driver behavior recognition, and driver emotion recognition, respectively. These tasks comprehensively demonstrate the complex and closely interconnected relationships between the driver and traffic.

66]. Despite these advancements, approximately 1.35 million fatalities occur annually in traffic accidents [42], with human drivers’ abnormal mental or physical states contributing to over 65% of these incidents [50]. Consequently, accurate identification of driver states is essential for ADAS [28, 38] but remains a complex challenge due to the intricate causal relationships between driver states and traffic context [62] (see Fig. 1). For instance, traffic congestion can induce driver anxiety, effecting driving behavior [62].

Most existing approaches to driver state and traffic context recognition focus on isolated tasks, such as driver behavior or emotion recognition [19, 29, 41, 45, 63, 64], traffic environment recognition [20, 39, 54, 67]. However, they fail to exploit the interaction between these tasks, which can limit the potential for cross-task learning [43]. In reality, these tasks are not independent when driving in real-world traffic [38, 62]. For example, lane-change behavior relies on the interaction between the current traffic context (e.g., congestion) and the driver’s emotional state [20, 57].

Multi-task learning (MTL) offers an opportunity to enhance task recognition accuracy by enabling information sharing between related tasks [8, 26]. However, current research is often hindered by negative transfer, a phenomenon where performance degrades due to information sharing between weakly related tasks. Many existing works focus on tasks with fewer differences. For instance, studies often combine similar tasks like lane detection, depth estimation, and drivable area segmentation [43, 49, 56] to improve traffic context understanding, while others focus on joint learning for driver-related tasks such as behavior, emotion, intention recognition [27, 57, 61]. However, neglecting the interaction between driver-related tasks and traffic context recognition limits the integration of driver state and environmental information, thus restricting ADAS’s ability to fully understand driving context [38, 62].

Another significant limitation of current MTL models is their reliance on single-modal inputs. Achieving satisfactory recognition performance typically requires the complementarity of multimodal data [10, 11, 18, 20, 29, 32, 33]. For instance, traffic context recognition often relies on multi-view driving scene images, while driver behavior and state recognition necessitate comprehensive in-vehicle images, fine-grained driver images, and joint data for driver posture and gesture. Although leveraging these multimodal inputs can enhance the accuracy [20, 44], most existing MTL models rely on a single input type (e.g., driving scene images or driver images) [7, 43, 49], severely limiting their practical application in ADAS.

To address these challenges, we propose a Unified Framework for Multimodal and Multi-Task Learning in Assistive Driving Perception (MMTL-UniAD). Our framework leverages multimodal data to simultaneously recognize driver behavior, emotion, traffic context, and vehicle behavior. First, inspired by [53, 69], we designed a multiaxis region attention network for multi-view images from the driving environment and the human driver. This network captures global context through horizontal-vertical attention and then uses region attention to extract interest-triggering features, selecting task-related high-level semantic information. This approach mitigates negative transfer between tasks. Additionally, we introduce a dual-branch multimodal embedding that uses soft parameter sharing [36] to adaptively adjust task-shared and task-specific parameters. This design enhances task-specific learning while promoting information sharing and positive transfer between tasks. We validate our approach on the publicly available AIDE dataset, demonstrating superior performance over state-ofthe-art methods. Our contributions are:

• We propose MMTL-UniAD, a novel framework that addresses the challenges of multimodal multi-task learning (MMTL) in assistive driving.

• We designed a multi-axis region attention network to effectively extract features from multi-view images of the driving environment and human drivers.

• We introduce a dual-branch multimodal embedding to extract task-shared and task-specific features, mitigating negative transfer caused by task differences while enhancing cross-task knowledge learning.

## 2. Related Work

## 2.1. Multi-task Learning

Recent advancements in deep neural networks have made it feasible to learn multiple tasks jointly, improving the performance of individual tasks through shared learning. A key strategy for achieving multi-task learning is the sharing of parameters across tasks. Two primary approaches for parameter sharing are hard parameter sharing and soft parameter sharing.

• Hard parameter sharing limits the sharing of parameters to the output layer of the neural network [1, 3, 14, 17, 25, 30, 58]. This method effectively reduces overfitting between tasks and improve learning efficiency. However, it lacks flexibility, as it heavily depends on the correlations between tasks. If the task has significantly different objectives or conflicts, the shared parameters may harm the performance of individual tasks [40].

• Soft parameter sharing, on the other hand, extends sharing to both output and internal layers of the network [5, 15, 59, 60]. This approach helps reduce conflicts between tasks, even when the tasks have weak correlations. By allowing more flexible adaptation, soft parameter sharing can enhance the performance and adaptability of each individual task, particularly in settings where tasks may differ in complexity or relevance.

In the context of MTL for ADAS, soft parameter sharing has become an attractive choice for improving the overall performance across multiple driving-related tasks by allowing tasks to learn shared representations while maintaining flexibility in task-specific learning.

## 2.2. Driver State Recognition

Driver state recognition is a critical component of ADAS, as understanding the mental and physical state of the driver is essential for ensuring road safety. Existing studies have leveraged various signals to infer driver states, focusing on both driver-specific and traffic-related information.

• Some approaches rely on vehicle dynamics-based data (e.g., speed, steering angle) alongside multi-view images (e.g., front-view, right-view, and left-view) of the driving scene to assess the driver’s state [19, 32, 33, 45]. These methods focus on the relationship between the vehicle’s movement and the surrounding environment but often fail to capture the driver’s internal state.

• Other studies have utilized driver behavior data, including images of the driver and physiological signals (e.g., heart rate, eye gaze) to detect emotional or cognitive states such as fatigue, stress, or distraction. [11, 29, 41, 63, 64]. These approaches focus more directly on the driver but may neglect the impact of the surrounding traffic context on driver behavior.

![](images/94e9fe09cb93c7e0e59e81c8e2289d1065f064dda22d1c276e3c6b6b29c3381c.jpg)  
Figure 2. The overall pipeline of MMTL-UniAD. MMTL-UniAD consists of two primary components: Multimodal Encoder and Dual-Branch Multimodal Embedding. The multimodal encoder is composed of a Multi-axis Regional Attention Network (MARNet) and a 3D-CNN, which are responsible for extracting features from multi-view images and driver joint, respectively. The Dual-Branch Multimodal Embeddings further integrate the multimodal features for multi-task recognition.

While these works have made progress in recognizing driver states, they tend to focus either on driver information or traffic context, without effectively combining both. This limitation becomes apparent in complex, dynamic driving situations, where both the driver’s internal state and external traffic context jointly influence driving behavior. As such, integrating driver states and traffic context remains an open challenge in ADAS research, as it requires the simultaneous consideration of both domains to provide a comprehensive understanding of the driving context.

## 3. Methods

This section presents the proposed MMTL-UniAD framework (Fig. 2), highlighting its two key components: Multi-axis Region Attention Network (MARNet) and Dual-Branch Multimodal Embedding.

## 3.1. Network Overview

The framework of MMTL-UniAD (Fig. 2) consists of two main modules: the multimodal encoder and dualbranch multimodal embedding. The former is composed of a Multi-axis Region Attention Network (MARNet) and a 3D Convolutional Neural Network (3D-CNN). Specifically, MARNet captures key features from multi-view images (i.e., front-view, right-view, left-view, inside-view, driver face, and driver body) through multi-attention mechanisms, while the 3D-CNN extracts prominent features from driver joint data (i.e., gesture and posture). The latter consists of a task-shared branch and a task-specific branch, which are used to further fuse the extracted multimodal features. This module first extracts task-shared features $\mathbf { F } _ { \mathrm { s h } }$ and taskspecific features $\mathbf { F } _ { \mathrm { s p } }$ by adaptively adjusting the parameters of these two branches, enhancing cross-task knowledge sharing while capturing unique information of tasks. Next, the two features are integrated through dynamic fusion, to obtain the recognition results $O _ { j }$ , for individual tasks j: driver behavior recognition, driver emotion recognition, traffic context recognition, and vehicle behavior recognition. The recognition process for each task is as follows:

![](images/f982f3c598d6fce0e2b995f095fd44da454c81b8fdb5a809ac2484b6b29c46c1.jpg)  
Figure 3. Diagram of different self-attention. (a) represents the most common global self-attention in images; (b) (c) (d) representing vertical attention, horizontal attention and horizontal-vertical attention respectively. Among them (d) represents the horizontalvertical attention we introduced.

$$
O _ { j } = \sigma ( w _ { j } ) L _ { j } ^ { 1 } ( { \bf F } _ { s h } ) + ( 1 - \sigma ( w _ { j } ) ) L _ { j } ^ { 2 } ( w _ { c a } ( { \bf F } _ { s p _ { j } } ) ) ,\tag{1}
$$

where $L _ { j } ^ { 1 }$ and $L _ { j } ^ { 2 }$ are fully connected layers corresponding to task $j ,$ $w _ { c a }$ represents a channel attention weighting operation that prioritizes relevant features for the task, $w _ { j }$ is a learnable weight parameter that dynamically selects the most effective information from task-shared and task-specific features, and σ denotes the Sigmoid function. This formulation enables the model to adaptively prioritize shared knowledge across tasks, while retaining the distinct characteristics necessary for task-specific performance.

## 3.2. Multimodal Encoder

## 3.2.1. Multi-axis Region Attention Network

Multi-view images of driving environment and human drivers usually contain many task-unrelated features such as billboards along the roadside and in-car decorative items. In multi-task learning, the quality of extracted features directly impacts cross-task synergy. Selecting features relevant to tasks can facilitate information complementarity between tasks in feature sharing; otherwise, would lead to a negative transfer issue. To address this challenge, we designed MARNet, as shown in Fig. 4. In this network, taskrelated features from multi-view images are extracted using the proposed horizontal-vertical attention and region attention to alleviate the negative transfer issue caused by irrelevant features across tasks.

Let $\mathbf { F } _ { o } \in \mathbb { R } ^ { H \times W \times C }$ be the input feature map, which is obtained from the multi-view images through initial convolution operations, and H, W and C are the height, width, and number of channels of the map. The horizontal-vertical attention first performs linear projections on the input feature map $\mathbf { F } _ { o }$ using three weight matrices to generate the query, key, and value, denoted as Q, K, and V, respectively. We then apply self-attention along the vertical direction of $\mathbf { F } _ { o }$ at each position $( h , w )$ to integrate features relevant (see Fig. 3 (b)), resulting in a new feature $\mathbf { F } _ { v } ,$ the computation process is as follows:

![](images/6aba15c9b45984b1165931f2eeab30b0721367ea3d4c557fbe25ce2fe7b7e692.jpg)  
Figure 4. The flowchart of the MARNet architecture, including the processes for horizontal-vertical attention and region attention.

$$
\mathbf { F } _ { v } = \sum _ { w = 1 } ^ { W } \sum _ { h = 1 } ^ { H } \sum _ { h ^ { \prime } = 1 } ^ { H } \mathrm { s o f t m a x } \left( \frac { \mathbf { Q } _ { ( h , w , \cdot ) } \mathbf { K } _ { ( h ^ { \prime } , w , \cdot ) } ^ { \top } } { \sqrt { C } } \right) \mathbf { V } _ { ( h ^ { \prime } , w , \cdot ) } ,\tag{2}
$$

where $\mathbf { Q } _ { ( h , w , \cdot ) }$ denotes the query vector at position $( h , w )$ $\mathbf { K } _ { ( h ^ { \prime } , w , \cdot ) }$ and $\mathbf { V } _ { ( h ^ { \prime } , w , \cdot ) }$ denote the key and value vector at position $( h ^ { \prime } , w )$ , respectively. Similarly, we can obtain ${ \bf { F } } _ { h }$ along the horizontal direction (see Fig. 3 (c)), the computation process is as follows:

$$
\mathbf { F } _ { h } = \sum _ { h = 1 } ^ { H } \sum _ { w = 1 } ^ { W } \sum _ { w ^ { \prime } = 1 } ^ { W } \mathrm { s o f t m a x } \left( \frac { \mathbf { F } _ { v _ { ( h , w , \cdot ) } } \mathbf { K } _ { ( h , w ^ { \prime } , \cdot ) } ^ { \top } } { \sqrt { C } } \right) \mathbf { V } _ { ( h , w ^ { \prime } , \cdot ) } .\tag{3}
$$

Next, we concatenate features $\mathbf { F } _ { h }$ and $\mathbf { F } _ { o }$ to obtain a new feature map $\mathbf { F } ^ { \prime } \in \mathbb { R } ^ { H \times W \times C }$

$$
\mathbf { F } ^ { \prime } = \operatorname { C o n c a t } ( \operatorname { C 2 D } _ { 1 * 1 } ( \mathbf { F } _ { h } ) , \mathbf { F } _ { o } ) ,\tag{4}
$$

where $\mathrm { C 2 D } _ { 1 * 1 } ( \cdot )$ denotes a 2D convolution operation with kernel size of 1, and Concat(·) denotes a concatenate operation. This operation utilizes the long-range dependencies extracted along horizontal-vertical directions of $\mathbf { F } _ { o }$ (see Fig. 3 (d)), preserving the detailed information of initial features.

Horizontal-vertical attention captures directional features and global context in multi-view images. However, in real-world scenarios, nearby road users (e.g., other vehicles and pedestrians) often appear in varying orientations, making it challenging to capture their dependencies effectively [69]. To address this limitation, we integrate MAR-Net with region attention, which enables the model to focus on the most relevant regions of the input feature map, $\mathbf { F } ^ { \prime } .$ This approach dynamically selects regions based on similarity measures, allowing the model to emphasize interesttriggering features of dynamic objects that do not adhere to a fixed direction.

Specifically, we first use region attention to partition the feature map $\mathbf { F ^ { \prime } }$ into $\textstyle { \frac { H W } { t ^ { 2 } } }$ independent regions, each of size $t \times t .$ This transforms the feature map into ${ \bf F } ^ { \prime \prime } \in  { }$ $\mathbb { R } ^ { \frac { H W } { t ^ { 2 } } \times t ^ { 2 } \times C }$ . We then apply three weight matrices to linearly project $\mathbf { F } ^ { \prime \prime }$ into the query, key, and value, denoted as Q<sup>′</sup>, K<sup>′</sup>, ${ \bf V } ^ { \prime } \in \mathbb { R } ^ { \frac { H W } { t ^ { 2 } } \times t ^ { \frac { 1 } { 2 } } \times C }$ , respectively. To improve computational efficiency, we apply global average pooling across the second dimension of both $\mathbf { Q } ^ { \prime }$ and $\mathbf { K } ^ { \prime }$ , resulting in $\mathbf { Q } ^ { \prime \prime } , \mathbf { K } ^ { \prime \prime } \in \mathbb { R } ^ { \frac { H W } { t ^ { 2 } } \times C }$ . The similarity matrix is then computed using the dot product between $\mathbf { Q } ^ { \prime \prime }$ and $\mathbf { K } ^ { \prime \prime }$ , and the k most similar regions for each region l are selected, forming the index set $\mathbf { I } _ { r } ^ { \bar { l } } \in \mathbb { R } ^ { k }$

$$
\mathbf { I } _ { r } ^ { l } = \operatorname { I n d e x } _ { k } \left( \mathbf { Q } ^ { \prime \prime } ( l ) \mathbf { K } ^ { \prime \prime \top } \right) ,\tag{5}
$$

where $\mathbf { Q } ^ { \prime \prime } ( l ) \in \mathbb { R } ^ { 1 \times C }$ is the query vector for region l, and $\mathbf { K } ^ { \prime \prime \top } \in \mathbb { R } ^ { \dot { C } \times \frac { H W } { t ^ { 2 } } }$ is the transpose of $\mathbf { K } ^ { \prime \prime }$ . Using this index, we extract the corresponding rows from $\mathbf { K } ^ { \prime }$ and $\mathbf { V } ^ { \prime }$ to form $\mathbf { K } _ { E } ^ { l } \in \mathbb { R } ^ { k \times t ^ { 2 } \times C }$ and $\mathbf { V } _ { E } ^ { l } \in \overline { { \mathbb { R } ^ { k \times t ^ { 2 } \times C } } }$ , respectively. Attention is then computed for each region l and its top k most similar regions:

$$
\mathbf F _ { x } ( l ) = \mathrm { s o f t m a x } \left( \frac { \mathbf Q ^ { \prime } ( l ) ( \mathbf K _ { E } ^ { l } ) ^ { \top } } { \sqrt { C } } \right) \mathbf V _ { E } ^ { l } ,\tag{6}
$$

where $\mathbf { F } _ { x } ( l ) \in \mathbb { R } ^ { t ^ { 2 } \times C }$ is the updated feature for region l after weighting the features via attention. After this operation, the output feature for all regions is $\mathbf { F } _ { x } \in \mathbb { R } ^ { \frac { H W } { t ^ { 2 } } \times t ^ { 2 } \times C }$ Finally, we reshape $\mathbf { F } _ { x }$ back to the dimension $( H , W , C )$ apply global average pooling, and pass the pooled result through a fully connected layer to obtain the final feature representation $\mathbf { F } _ { i } ,$ where $i \in \{ 1 , 2 , 3 , 4 , 5 , 6 \}$ corresponds to the six multi-view image inputs.

![](images/4881ddb0c3e40e629a252d12c2e2b7e727f64ff4100a745dfb4a2915dd13f5a0.jpg)  
Figure 5. Structural of Dual-Branch Multimodal Embedding.

MARNet extracts deep features from multi-view images by combining horizontal-vertical attention with region attention, yielding a rich set of effective features for subsequent multimodal feature embedding.

## 3.2.2. 3D-CNN

In the multimodal encoder, we use a 3D convolutional neural network (3D-CNN) to analyze the driver’s gestures and postures from continuous video frames. The 3D-CNN captures spatiotemporal features by applying 3D convolutional kernels across temporal and spatial dimensions. The feature representation is denoted as $\mathbf { F } _ { i }$ , where $i \in \{ 7 , 8 \}$ represents two joint inputs. The extracted features are primarily used to understand driver behavior and emotion.

## 3.3. Dual-Branch Multimodal Embedding

Balancing the synergy effect of task-shared and taskspecific features is crucial in multi-task learning. Taskshared features enable knowledge transfer across tasks, promoting generalization [1, 3]. However, disparities between tasks can result in negative transfer. Task-specific features help mitigate task conflict and reduce the risk of negative transfer. Yet, over-reliance on these features can limit information sharing, weakening the model’s ability to generalize across tasks [5, 60]. To address this, we designed a dual-branch multimodal embedding (Fig.5), which simultaneously extracts both feature types and adaptively balances their contributions based on the specific tasks at hand.

The dual-branch multimodal embedding consists of two primary components: the task-shared branch and the taskspecific branch. To address the negative transfer due to inter-task differences, the task-specific branch extracts task-specific features from multimodal inputs, as shown in Fig.5 (a). First, global average pooling is applied to the feature $\mathbf { F } _ { c } ,$ which is formed by concatenating the multimodal feature $\mathbf { F } _ { i }$ along the channel dimension $\begin{array} { r l } { ( \mathbf { F } _ { c } } & { { } = } \end{array}$ Concat $( \mathbf { F } _ { 1 } , \mathbf { F } _ { 2 } , \ldots , \mathbf { F } _ { 8 } ) )$ . Three 1D convolutions (kernel size = 3) are then used to capture local channel relationships, producing query, key, and value. Multi-head selfattention is subsequently employed to capture global interactions across channels. The global features are passed through a Sigmoid function to constrain them within the range (0, 1), and these values are used as weights to module $\mathbf { F } _ { c } ,$ resulting in task-specific features $\mathbf { F } _ { \mathrm { s p } _ { i } }$ . This design leverages the fact that different channels in $\dot { \mathbf { F } } _ { c }$ correspond to distinct modality features, such as those from MARNet and 3D-CNN, which are selectively emphasized depending on the target task (e.g., facial and body images are crucial for driver behavior and emotion recognition, as shown in the ablation study in Section 4.6.3). The channel interaction, guided by global-local attention, dynamically adjusts the importance of each modality for different tasks, alleviating interference from irrelevant modalities. The computation process of the task-specific branch $\mathbf { T } _ { \mathrm { s p } }$ is as follows:

$$
\mathbf { F } _ { \mathrm { s p } _ { j } } = \mathbf { T } _ { \mathrm { s p } } ( \mathbf { F } _ { c } ) = \mathbf { F } _ { c } { \otimes } \sigma ( \mathrm { M H A } _ { j } ( \mathrm { C } \mathrm { 1 D } _ { j } ( \mathrm { G A P } ( \mathbf { F } _ { c } ) ) , n ) ) ,\tag{7}
$$

where MH $\mathrm { A } _ { j } ( a , b )$ denotes the multi-head self-attention for task $j , \mathrm { C 1 D } _ { j } ( \cdot )$ denotes the 1D convolution, n is the number of heads, and ⊗ indicates element-wise multiplication.

To enhance task synergy while extracting task-specific features, we designed the task-shared branch, as shown in Fig.5 (b). To improve extraction efficiency, we first categorize multimodal features based on their source and structure. Features related to the traffic context (i.e., images from the left, right, and front views) are concatenated into $\mathbf { F } _ { \mathrm { s c } } .$ driver-related features (i.e., images from the inside view, driver’s face, and body) form $\mathbf { F } _ { \mathrm { d r } }$ , and joint features (i.e., posture and gesture) form $\mathbf { F } _ { \mathrm { j o } }$ . To address scale differences in multimodal data, the task-shared branch combines $\mathbf { F } _ { \mathrm { s c } }$ and $\mathbf { F } _ { \mathrm { d r } }$ , then applies convolutions with kernel sizes of 1 and 3 to capture features at different scales, producing a multi-scale feature map. The computation process f(·) is shown as follows:

$$
\begin{array} { r } { f ( \mathbf { F } _ { \mathrm { d r } } , \mathbf { F } _ { \mathrm { s c } } ) = \mathrm { C 2 D } _ { 1 * 1 } ( \mathbf { F } _ { \mathrm { d r } } + \mathbf { F } _ { \mathrm { s c } } ) + \mathrm { C 2 D } _ { 3 * 3 } ( \mathbf { F } _ { \mathrm { d r } } + \mathbf { F } _ { \mathrm { s c } } ) , } \end{array}\tag{8}
$$

Where $\mathrm { C 2 D _ { 1 * 1 } }$ and $\mathrm { C 2 D _ { 3 * 3 } }$ represent 2D convolution operations with kernel sizes of 1 and 3, respectively. The task-specific branch then dynamically integrates this feature map and passes it through a Sigmoid function to generate weights. These weights are used to merge $\mathbf { F } _ { \mathrm { s c } }$ and $\mathbf { F } _ { \mathrm { d r } }$ adaptively, resulting in the preliminary shared features $\mathbf { F } _ { \mathrm { p s } } .$ The computation process of the task-shared branch $\mathbf { T } _ { \mathrm { s h } } ( \cdot )$ is shown as follows:

$$
\begin{array} { r } { \mathbf { F } _ { \mathrm { p s } } = \mathbf { T } _ { \mathrm { s h } } ( \mathbf { F } _ { \mathrm { d r } } , \mathbf { F } _ { \mathrm { s c } } ) = \mathbf { F } _ { \mathrm { s c } } \times \sigma ( \mathbf { T } _ { \mathrm { s p } } ( f ( \mathbf { F } _ { \mathrm { d r } } , \mathbf { F } _ { \mathrm { s c } } ) ) ) \quad ( 9 ) } \\ { + \mathbf { F } _ { \mathrm { d r } } \times \big ( 1 - \sigma ( \mathbf { T } _ { \mathrm { s p } } ( f ( \mathbf { F } _ { \mathrm { d r } } , \mathbf { F } _ { \mathrm { s c } } ) ) \big ) \big ) . } \end{array}
$$

Similarly, $\mathbf { F } _ { \mathrm { p s } }$ and $\mathbf { F } _ { \mathrm { j o } }$ undergo feature extraction by the task-shared branch using the same operations, resulting in the final task-shared features ${ \bf F } _ { \mathrm { s h } }$ . The calculation is by:

$$
\mathbf { F } _ { \mathrm { s h } } = \mathbf { T } _ { \mathrm { s h } } \big ( \mathbf { F } _ { \mathrm { j o } } , \mathbf { F } _ { \mathrm { p s } } \big ) .\tag{10}
$$

Finally, the task-specific features $\mathbf { F } _ { \mathrm { s p . } }$ and the taskshared features $\mathbf { F } _ { \mathrm { s h } }$ are dynamically fused using Eq. (1) to yield the recognition result $O _ { j }$ for each task.

## 3.4. Loss Function

In MMTL-UniAD, we propose a novel loss function that integrates the individual losses of each task to optimize overall model performance. Specifically, the total loss $L _ { \mathrm { t o t a l } }$ is calculated as follows:

$$
L _ { \mathrm { t o t a l } } = \sum _ { j = 1 } ^ { m } { \mathrm { C r o s s E n t r o p y } } ( O _ { j } , \mathrm { l a b e l } _ { j } ) ,\tag{11}
$$

where $O _ { j }$ represents the recognition results for task $j ,$ label<sub>j</sub> denotes the corresponding ground truth label, CrossEntropy is the cross-entropy loss function. The number of tasks, m, is set to 4, corresponding to the four recognition tasks implemented: driver emotion recognition, driver behavior recognition, traffic context recognition, and vehicle behavior recognition.

## 4. Experiments

We conducted extensive experiments using the open-source AIDE dataset [62] to evaluate the effectiveness of our proposed MMTL-UniAD for multi-task learning. This section introduces the AIDE dataset, data preprocessing, and evaluation metrics, followed by a performance comparison of MMTL-UniAD across four tasks relative to existing algorithms. Finally, we present and analyze the results of our ablation experiment.

## 4.1. Dataset

The AIDE dataset consists of 2,898 samples with multiview, multimodal, and multi-task features. Each sample includes multi-view video data (i.e., front, right, left, inside views) and multimodal data related to the driver (i.e., images of the driver’s face and body, and joint data representing driver posture and gestures). The dataset is split into training (65%), validation(15%), and test (20%) sets. Annotations cover four tasks: driver emotion recognition, driver behavior recognition, traffic context recognition, and vehicle behavior recognition.

## 4.2. Data Preprocessing

For data preprocessing, we used bounding box coordinates to crop inside-view images, extracting refined images of the driver’s face and body. We then synchronized multi-view video data and joint data at 16 frames per second to ensure temporal alignment. Each model receives a sequence of 16 consecutive frames along with corresponding joint data for each frame, preserving temporal information during feature extraction. Additionally, data augmentation was applied to all images with a 50% probability of random horizontal and vertical flipping.

## 4.3. Evaluation Metrics

To assess the model performance, we adopted accuracy (%) as the primary metric. Furthermore, following [12, 47], we introduced a new evaluation metric mAcc (%) for multitask learning, defined as:

$$
{ \mathrm { m A c c } } = { \frac { 1 } { m } } \sum _ { j = 1 } ^ { m } \operatorname { A c c } _ { j } ,\tag{12}
$$

where m is the number of tasks in the multi-task learning, and $\operatorname { A c c } _ { j }$ is the accuracy achieved by the model on task $j .$ We use mAcc to provide a comprehensive evaluation across all tasks, ensuring a balanced assessment of model performance rather than focusing on individual tasks.

## 4.4. Implement Details

All experiments were conducted on an NVIDIA A40 GPU with a batch size of 24 for both training and testing. The stochastic gradient descent (SGD) optimizer was applied with a momentum of 0.9 and weight decay of 0.0001. The initial learning rate was set to 0.1, and the number of epochs was 125. For region attention, the t was set to 7, indicating a region size of $7 \times 7 .$ , and k was set to 4, selecting the top 4 regions with the highest similarity to the target region.

## 4.5. Comparison with the State-of-the-Art

Table 1 presents the multi-task evaluation results of our proposed MMTL-UniAD, compared to the state-of-the-art methods. Based on the feature extraction dimensions of multi-view sequential images, we categorize the comparison methods into three patterns, following [62]: 2D(using 2D models, e.g., VGG [48], ResNet[23], and CMT [21]), 2D+Timing (combining 2D models with sequence models [52]), and 3D (using 3D models, e.g., Video Swin Transformer [35], and 3D Implementations of MobileNet [24] and ShuffleNet [65]).

Our MMTL-UniAD outperforms all algorithms in these three patterns, improving the mAcc metric by 4.10%- 12.09%, and achieving the best results across all four tasks. Notably, it significantly improves recognition accuracy for driver behavior (by 4.64%) and vehicle behavior (by 3.62%), which demonstrates the superiority of our algorithm in multi-task learning.

The outstanding performance of MMTL-UniAD can be attributed to the design of its MARNet and dual-branch multimodal embedding. These components leverage multiattention mechanisms and a dual-branch structure for taskshared and task-specific feature extraction. This design enhances cross-task knowledge transfer while mitigating negative transfer caused by inter-task differences and irrelevant features. In contrast, other algorithms, although employing various feature extraction backbones for multi-task recognition, are not optimized to address multi-task learning challenges, such as negative transfer. As a result, they suffer from significant negative transfer effects between tasks, limiting improvements in recognition performance.

<table><tr><td rowspan="2">Pattern</td><td colspan="5">Backbone</td><td rowspan="2">DER</td><td colspan="2">DBR TCR</td><td rowspan="2">VBR</td><td rowspan="2">mAcc</td></tr><tr><td>Face</td><td>Body</td><td>Gesture</td><td>Posture</td><td>Scene</td><td>Acc Acc</td><td>Acc</td><td>Acc</td></tr><tr><td rowspan="11">2D</td><td>Res18 [23]</td><td>Res34 [23]</td><td>MLP+SE</td><td>MLP+SE</td><td>PP-Res18 [68]</td><td>69.05</td><td>63.87</td><td>88.01</td><td>78.16</td><td>74.77</td></tr><tr><td>Res18 [23]</td><td>Res34 [23]</td><td>MLP+SE</td><td>MLP+SE</td><td>Res34 [23]</td><td>71.26</td><td>65.35</td><td>83.74</td><td>77.12</td><td>74.37</td></tr><tr><td>Res34 [23]</td><td>Res50 [23]</td><td>MLP+SE MLP+SE</td><td></td><td>Res50 [23]</td><td>69.68</td><td>59.77</td><td>80.13</td><td>71.26</td><td>70.21</td></tr><tr><td>VGG13 [48]</td><td>VGG16 [48]</td><td>MLP+SE</td><td>MLP+SE</td><td>VGG16 [48]</td><td>70.72</td><td>63.65</td><td>82.77</td><td>77.94</td><td>73.77</td></tr><tr><td>VGG16 [48]</td><td>VGG19 [48]</td><td>MLP+SE</td><td>MLP+SE</td><td>VGG19 [48]</td><td>69.31</td><td>62.34</td><td>83.58</td><td>75.13</td><td>72.59</td></tr><tr><td>CPVT [9]</td><td>CPVT [9]</td><td>ST-GCN</td><td>ST-GCN</td><td>CPVT [9]</td><td>69.01</td><td>67.35</td><td>91.44</td><td>79.57</td><td>76.84</td></tr><tr><td>CMT [21]</td><td>CMT [21]</td><td>ST-GCN</td><td>ST-GCN</td><td>CMT [21]</td><td>68.75</td><td>68.75</td><td>93.75</td><td>81.38</td><td>78.16</td></tr><tr><td>GroupMixFormer [16]</td><td>GroupMixFormer [16]</td><td>ST-GCN</td><td>ST-GCN</td><td>GroupMixFormer [16]</td><td>66.29</td><td>67.54</td><td>92.12</td><td>77.63</td><td>75.90</td></tr><tr><td>AbSViT [34]</td><td>AbSViT [34]</td><td>ST-GCN</td><td>ST-GCN</td><td>AbSViT [34]</td><td>69.15</td><td>67.84</td><td>92.07</td><td>80.82</td><td>77.47</td></tr><tr><td>GLMDriveNet [32]</td><td>GLMDriveNet [32]</td><td>ST-GCN</td><td>ST-GCN</td><td>GLMDriveNet [32]</td><td>71.38</td><td>66.57</td><td>90.23</td><td>77.19</td><td>76.34</td></tr><tr><td rowspan="5">2D + Timing</td><td>Res18 [23]+TransE</td><td>Res34 [23]+TransE [52]</td><td>MLP+TE</td><td>MLP+TE</td><td>PP-Res18+TransE [52]</td><td>70.83</td><td>67.32</td><td>90.54</td><td>79.97</td><td>77.17</td></tr><tr><td>Res18 [23]+TransE [52]</td><td>Res34 [23]+TransE [52]</td><td>MLP+TE</td><td>MLP+TE</td><td>Res34 [23]+TransE</td><td>72.65</td><td>67.08</td><td>86.63</td><td>78.46</td><td>76.21</td></tr><tr><td>Res34 [23]+TransE [52]</td><td>Res50 [23]+TransE [52]</td><td>MLP+TE MLP+TE</td><td></td><td>Res50 [23]+TransE [52]</td><td>70.24</td><td>65.65</td><td>82.57</td><td>77.29</td><td>73.94</td></tr><tr><td>VGG13 [48]+TransE [52]</td><td>VGG16 [48]+TransE [52]</td><td>MLP+TE MLP+TE</td><td></td><td>VGG16 [48]+TransE [52]</td><td>71.12</td><td>67.15</td><td>85.13</td><td>78.58</td><td>75.50</td></tr><tr><td>VGG16 [48]+TransE [52]</td><td>VGG19 [48]+TransE [52]</td><td>MLP+TE</td><td>MLP+TE</td><td>VGG19 [48]+TransE [52]</td><td>69.46</td><td>65.48</td><td>85.74</td><td>77.91</td><td>74.65</td></tr><tr><td rowspan="11">3D</td><td>MobileNet-V1 [24]</td><td>MobileNet-V1 [24]</td><td>ST-GCN</td><td>ST-GCN</td><td>MobileNet-V1 [24]</td><td>72.23</td><td>64.20</td><td>88.34</td><td>77.83</td><td>75.65</td></tr><tr><td>MobileNet-V2 [46]</td><td>MobileNet-V2 [46]</td><td>ST-GCN</td><td>ST-GCN</td><td>MobileNet-V2 [46]</td><td>68.47</td><td>61.74</td><td>86.54</td><td>78.66</td><td>73.85</td></tr><tr><td>ShuffleNet-V1 [65]</td><td>ShuffleNet-V1 [65]</td><td>ST-GCN</td><td>ST-GCN</td><td>ShuffleNet-V1 [65]</td><td>72.41</td><td>68.97</td><td>90.64</td><td>80.79</td><td>78.20</td></tr><tr><td>ShuffleNet-V2 [37]</td><td>ShuffleNet-V2 [37]</td><td>ST-GCN</td><td>ST-GCN</td><td>ShuffleNet-V2 [37]</td><td>70.94</td><td>64.04</td><td>89.33</td><td>78.98</td><td>75.82</td></tr><tr><td>3D-Res18 [22]</td><td>3D-Res34 [22]</td><td>ST-GCN</td><td>ST-GCN</td><td>3D-Res34 [22]</td><td>70.11</td><td>66.52</td><td>88.51</td><td>81.21</td><td>76.59</td></tr><tr><td>3D-Res34 [22]</td><td>3D-Res50 [22]</td><td>ST-GCN</td><td>ST-GCN</td><td>3D-Res50 [22]</td><td>69.13</td><td>63.05</td><td>87.82</td><td>79.31</td><td>74.83</td></tr><tr><td>C3D [51]</td><td>C3D [51]</td><td>ST-GCN</td><td>ST-GCN</td><td>C3D [51]</td><td>63.05</td><td>63.95</td><td>85.41</td><td>77.01</td><td>72.36</td></tr><tr><td>I3D [4]</td><td>I3D [4]</td><td>ST-GCN</td><td>ST-GCN</td><td>I3D [4]</td><td>70.94</td><td>66.17</td><td>87.68</td><td>79.81</td><td>76.15</td></tr><tr><td>SlowFast [13]</td><td>SlowFast [13]</td><td>ST-GCN</td><td>ST-GCN</td><td>SlowFast [13]</td><td>72.38</td><td>61.58</td><td>86.86</td><td>78.33</td><td>74.79</td></tr><tr><td>TimeSFormer [2]</td><td>TimeSFormer [2]</td><td>ST-GCN</td><td>ST-GCN</td><td>TimeSFormer [2]</td><td>74.87</td><td>65.18</td><td>92.12</td><td>78.81</td><td>77.75</td></tr><tr><td>Video Swin Transformer [35] Video Swin Transformer [35]</td><td></td><td>3DCNN</td><td>3DCNN</td><td>Video Swin Transformer [35]</td><td>73.44</td><td>65.63</td><td>93.75</td><td>75.00</td><td>76.96</td></tr><tr><td colspan="2">Ours MARNet</td><td>MARNet</td><td>3DCNN</td><td>3DCNN</td><td>MARNet</td><td>76.67</td><td>73.61</td><td>93.91</td><td>85.00</td><td>82.30</td></tr></table>

Table 1. Comparison results of baselines on the AIDE dataset for all four tasks. DER is Driver Emotion Recognition, DBR is Driver Behavior Recognition, TCR is Traffic Context Recognition, VBR is Vehicle Behavior Recognition. The best results is highlighted in bold, while the second-best results are underlined. The row highlighted in indicates our proposed method. Scene denotes multi-view images (i.e., front-view, right-view, left-view, inside-view). Res: ResNet [23], MLP: multi-layer perception, SE: spatial embedding, TE temporal embedding, TransE: transformer encoder [52].

## 4.6. Ablation Experiment

We conduct a series of ablation experiments to assess the effectiveness and individual contributions of multi-task learning, multimodal data, and key components, including MARNet and the dual-branch multimodal embedding.

## 4.6.1. Ablation Studies on Multi-task Learning

To verify the necessity of multi-task learning, we perform two sets of ablation experiments.

<table><tr><td rowspan="2">Method</td><td colspan="2">Task</td><td rowspan="2">DER Acc</td><td rowspan="2">DBR</td><td rowspan="2">TCR</td><td rowspan="2">VBR Acc</td></tr><tr><td>Driver States</td><td>Traffic Context</td></tr><tr><td rowspan="2">Contrast</td><td>w/</td><td>w/o</td><td>72.22</td><td>Acc 69.35</td><td>Acc 一</td><td>一</td></tr><tr><td>w/o</td><td>w/</td><td>1</td><td>-</td><td>90.41</td><td>80.63</td></tr><tr><td>Ours</td><td>w/</td><td>w/</td><td>76.67</td><td>73.61</td><td>93.91</td><td>85.00</td></tr></table>

Table 2. Results of Multi-task Ablation Experiments for driver states and traffic context. ”w/” indicates the use of the corresponding component or method, while ”w/o” means the component or method was not used. Driver states include DER and DBR, while the traffic context includes TCR and VBR.

The first set evaluates the impact of including both driver states and traffic context tasks. Results in Table 2 show that when only the driver states-related tasks are jointly trained, without considering traffic context tasks, the accuracy of driver emotion recognition and driver behavior recognition decreases by 4.26%-4.45%. Similarly, when only traffic context-related tasks are trained, excluding driver states, the accuracy of traffic context and vehicle behavior recognition drops by 3.50%-4.37%. These results demonstrate that features learned from the driver states tasks benefit the traffic context recognition tasks, and vice versa, highlighting the advantages of joint learning for improving model accuracy and generalization.

<table><tr><td rowspan="2">Method</td><td rowspan="2">Config</td><td rowspan="2">Task</td><td colspan="4">DER DBR TCR VBR</td></tr><tr><td>Acc</td><td>Acc</td><td>Acc</td><td>Acc</td></tr><tr><td rowspan="8">Contrast</td><td>w/o</td><td>DBR &amp; TCR &amp; VBR</td><td>70.56</td><td></td><td></td><td>1</td></tr><tr><td>w/o</td><td>DER &amp; TCR &amp; VBR</td><td></td><td>68.12</td><td></td><td></td></tr><tr><td>w/o</td><td>DER &amp; DBR &amp; VBR</td><td></td><td>一</td><td>89.93</td><td></td></tr><tr><td>w/o</td><td>DER &amp; DBR &amp; TCR</td><td></td><td></td><td></td><td>78.87</td></tr><tr><td>w/</td><td>DBR &amp; TCR &amp; VBR</td><td></td><td>64.19</td><td>92.36</td><td>83.71</td></tr><tr><td>w/</td><td>DER &amp; TCR &amp; VBR</td><td>73.01</td><td></td><td>91.73</td><td>84.56</td></tr><tr><td>w/</td><td>DER &amp; DBR &amp; VBR</td><td>75.32</td><td>70.83</td><td></td><td>79.85</td></tr><tr><td>w/</td><td>DER &amp; DBR &amp; TCR</td><td>74.62</td><td>67.24</td><td>89.03</td><td>-</td></tr><tr><td>Ours</td><td>w/</td><td>Full Tasks</td><td>76.67</td><td>73.61</td><td>93.91</td><td>85.00</td></tr></table>

Table 3. Results of Multi-task Ablation Experiments.

The second set of experiments examines the interactions between different tasks. The first part trains the model on a single task, while the second part removes one task, leaving the other three. Table 3 shows that training on a single task results in a 3.98%-6.13% drop in performance across all tasks. Moreover, removing any one of the four tasks also reduce the accuracy of the remaining tasks. These findings underscore the positive impact of jointly learning all four tasks, further validating the interrelated nature of these tasks and the effectiveness of multi-task learning.

## 4.6.2. Ablation Studies on MARNet and Dual-Branch Multimodal Embedding

We conducted ablation experiments to assess the individual contributions of the key components, MARNet and dual-branch multimodal embedding, within MMTL-UniAD. Specifically, we replaced MARNet with a basic VGG network and replaced the dual-branch multimodal embedding with a simple concatenation operation. The results in Table 4 show that models without MARNet or the dual-branch multimodal embedding exhibit a significant performance drop of 5.34%-12.05% in the mAcc metric, with accuracy across all four tasks decreasing accordingly.

This degradation is attributed to the failure of the alternative components to effectively capture key features during extraction, resulting in the inclusion of task-unrelated information in the multimodal features. This not only weakens the task synergy but also exacerbates negative transfer due to task conflicts. These findings demonstrate that the inclusion of MARNet and the dual-branch multimodal embedding is essential for optimal performance, as they facilitate better feature extraction and task-specific learning, ultimately enhancing cross-task synergy and the overall effectiveness of multi-task learning.

## 4.6.3. Ablation Studies on Multimodal Data

To further investigate the contribution of each modality, we performed ablation experiments using different combinations of input modalities. The input data were divided into three groups: (i) facial and body sequential images of the driver, (ii) gesture and body joint data, and (iii) multi-view sequential images (i.e., right, left, front, inside views). Each group was used individually to train the model, and the results are presented in Table 5.

<table><tr><td rowspan="2">Method</td><td rowspan="2">MARNet</td><td rowspan="2">DBME</td><td>DER</td><td>DBR</td><td>TCR</td><td>VBR</td><td rowspan="2">mAcc</td></tr><tr><td>Acc</td><td>Acc</td><td>Acc</td><td>Acc</td></tr><tr><td rowspan="3">Contrast</td><td>w/o</td><td>w/o</td><td>62.91</td><td>60.73</td><td>82.84</td><td>74.33</td><td>70.25</td></tr><tr><td>w/</td><td>w/o</td><td>71.66</td><td>67.12</td><td>91.50</td><td>77.56</td><td>76.96</td></tr><tr><td>w/o</td><td>w/</td><td>70.35</td><td>68.45</td><td>89.22</td><td>79.12</td><td>76.79</td></tr><tr><td>Ours</td><td>w/</td><td>w/</td><td>76.67</td><td>73.61</td><td>93.91</td><td>85.00</td><td>82.30</td></tr></table>

Table 4. Ablation experiment results of MARNet and Dual-Branch Multimodal Embedding. Among them, DBME represents Dual-Branch Multimodal Embedding
<table><tr><td rowspan="2">Method</td><td colspan="3">Multimodal Data</td><td rowspan="2">DER</td><td rowspan="2">DBR</td><td rowspan="2">TCR</td><td rowspan="2">VBR Acc</td><td rowspan="2">mAcc</td></tr><tr><td>Face+Body</td><td>G+P</td><td>Scene Acc</td></tr><tr><td rowspan="3">Contrast</td><td></td><td></td><td>V</td><td>67.81</td><td>Acc 64.32</td><td>Acc 92.87</td><td>82.66</td><td>76.91</td></tr><tr><td></td><td>V</td><td></td><td>68.22</td><td>54.13</td><td>63.11</td><td>37.29</td><td>55.69</td></tr><tr><td>V</td><td></td><td></td><td>72.91</td><td>71.77</td><td>89.16</td><td>78.69</td><td>78.13</td></tr><tr><td>Ours</td><td>√</td><td>V</td><td>√</td><td>76.67</td><td>73.61</td><td>93.91</td><td>85.00</td><td>82.30</td></tr></table>

Table 5. Results of Multimodal Ablation Experiments for Four Tasks. G+P represents gesture and posture joint data.

The results show a notable decrease in both mAcc and accuracy across all tasks when only one modality was used, highlighting the importance of multimodal data for these recognition tasks. This fully demonstrates the critical role of using multimodal data in improving the performance of multi-task learning models and the need for multitask learning architectures specifically designed to integrate multimodal inputs effectively.

## 5. Conclusions

This paper proposes MMTL-UniAD, a unified multi-modal multi-task learning framework. This framework leverages multi-modal data for simultaneous recognition of driver emotions, driver behavior, traffic context, and vehicle behavior. Central to this framework is the multi-axis region attention network and dual-branch multimodal embedding, which facilitate the effective extraction of both task-shared and task-specific features. These components not only enhance cross-task synergy but also mitigate the negative transfer, leading to superior performance across all four tasks on the open-source AIDE dataset. Ablation experiments further highlight that joint learning of driver states and traffic context-related tasks enables mutual feature sharing, which significantly improves task recognition accuracy. We anticipate that MMTL-UniAD along with its core components will serve as a robust baseline for future research in multimodal multi-task learning for Advanced Driver Assistance Systems (ADAS), advancing the development of more effective and adaptable systems in this domain.

## References

[1] Ziad Al-Halah, Santhosh Kumar Ramakrishnan, and Kristen Grauman. Zero experience required: Plug & play modular transfer learning for semantic visual navigation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 17031–17041, 2022. 2, 5

[2] Gedas Bertasius, Heng Wang, and Lorenzo Torresani. Is space-time attention all you need for video understanding? In International Conference on Machine Learnin (ICML), page 4, 2021. 7

[3] Kaidi Cao, Jiaxuan You, and Jure Leskovec. Relational multi-task learning: Modeling relations between data and tasks. arXiv preprint arXiv:2303.07666, 2023. 2, 5

[4] Joao Carreira and Andrew Zisserman. Quo vadis, action recognition? a new model and the kinetics dataset. In proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 6299–6308, 2017. 7

[5] Tianlong Chen, Xuxi Chen, Xianzhi Du, Abdullah Rashwan, Fan Yang, Huizhong Chen, Zhangyang Wang, and Yeqing Li. Adamv-moe: Adaptive multi-task vision mixture-ofexperts. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 17346–17357, 2023. 2, 5

[6] Yanbo Chen, Huilong Yu, and Junqiang Xi. Fw-dbns: Feedback-weighted dynamic bayesian networks for realtime velocity prediction. IEEE Transactions on Intelligent Vehicles, 2024. 1

[7] Wonhyeok Choi, Mingyu Shin, Hyukzae Lee, Jaehoon Cho, Jaehyeon Park, and Sunghoon Im. Multi-task learning for real-time autonomous driving leveraging task-adaptive attention generator. arXiv preprint arXiv:2403.03468, 2024. 2

[8] Sauhaarda Chowdhuri, Tushar Pankaj, and Karl Zipser. Multinet: Multi-modal multi-task learning for autonomous driving. In 2019 IEEE Winter Conference on Applications of Computer Vision (WACV), pages 1496–1504. IEEE, 2019. 2

[9] Xiangxiang Chu, Zhi Tian, Bo Zhang, Xinlong Wang, and Chunhua Shen. Conditional positional encodings for vision transformers. In The Eleventh International Conference on Learning Representations, 2023. 7

[10] Jialei Cui, Jianwei Du, Wenzhuo Liu, and Zhouhui Lian. Textnerf: a novel scene-text image synthesis method based on neural radiance fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22272–22281, 2024. 2

[11] Guanglong Du, Zhiyao Wang, Boyu Gao, Shahid Mumtaz, Khamael M Abualnaja, and Cuifeng Du. A convolution bidirectional long short-term memory neural network for driver emotion recognition. IEEE Transactions on Intelligent Transportation Systems, 22(7):4570–4578, 2020. 2, 3

[12] Jiaqi Fan, Fei Wang, Hongqing Chu, Xiao Hu, Yifan Cheng, and Bingzhao Gao. Mlfnet: Multi-level fusion network for real-time semantic segmentation of autonomous driving. IEEE Transactions on Intelligent Vehicles, 8(1):756– 767, 2022. 6

[13] Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He. Slowfast networks for video recognition. In

Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 6202–6211, 2019. 7

[14] Yongqi Gan, Wenzhuo Liu, Jianwang Gan, and Guoying Zhang. A segmentation method based on boundary fracture correction for froth scale measurement. Applied Intelligence, 54(9):6959–6980, 2024. 2

[15] Min Gao, Jian-Yu Li, Chun-Hua Chen, Yun Li, Jun Zhang, and Zhi-Hui Zhan. Enhanced multi-task learning and knowl edge graph-based recommender system. IEEE Transactions on Knowledge and Data Engineering, 35(10):10281–10294, 2023. 2

[16] Chongjian Ge, Xiaohan Ding, Zhan Tong, Li Yuan, Jian gliu Wang, Yibing Song, and Ping Luo. Advancing vision transformers with group-mix attention. arXiv preprint arXiv:2311.15157, 2023. 7

[17] Golnaz Ghiasi, Barret Zoph, Ekin D Cubuk, Quoc V Le, and Tsung-Yi Lin. Multi-task self-training for learning general representations. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 8856–8865, 2021. 2

[18] Yan Gong, Jianli Lu, Jiayi Wu, and Wenzhuo Liu. Multimodal fusion technology based on vehicle information: A survey. arXiv preprint arXiv:2211.06080, 2022. 2

[19] Yan Gong, Jianli Lu, Wenzhuo Liu, Zhiwei Li, Xinmin Jiang, Xin Gao, and Xingang Wu. Sifdrivenet: Speed and image fu sion for driving behavior classification network. IEEE Transactions on Computational Social Systems, 2023. 1, 2

[20] Chenghao Guo, Haizhuang Liu, Jiansheng Chen, and Huimin Ma. Temporal information fusion network for driving behavior prediction. IEEE Transactions on Intelligent Transportation Systems, 24(9):9415–9424, 2023. 1, 2

[21] Jianyuan Guo, Kai Han, Han Wu, Yehui Tang, Xinghao Chen, Yunhe Wang, and Chang Xu. Cmt: Convolutional neural networks meet vision transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 12175–12185, 2022. 6, 7

[22] Kensho Hara, Hirokatsu Kataoka, and Yutaka Satoh. Can spatiotemporal 3d cnns retrace the history of 2d cnns and imagenet? In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 6546–6555, 2018. 7

[23] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceed ings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 770–778, 2016. 6, 7

[24] Andrew G Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco An dreetto, and Hartwig Adam. Mobilenets: Efficient convolutional neural networks for mobile vision applications. arXiv preprint arXiv:1704.04861, 2017. 6, 7

[25] Yi Huang, Wenzhuo Liu, Yaoyu Li, Lei Yang, Hanqi Jiang, Zhiwei Li, and Jun Li. Mfe-ssnet: Multi-modal fusion-based end-to-end steering angle and vehicle speed prediction network. Automotive Innovation, pages 1–14, 2024. 2

[26] Keishi Ishihara, Anssi Kanervisto, Jun Miura, and Ville Hau tamaki. Multi-task learning with attention for end-to-end autonomous driving. In Proceedings of the IEEE/CVF con

ference on computer vision and pattern recognition, pages 2902–2911, 2021. 2

[27] Inhan Kim, Hyemin Lee, Joonyeong Lee, Eunseop Lee, and Daijin Kim. Multi-task learning with future states for visionbased autonomous driving. In Proceedings ofthe Asian Conference on Computer Vision, 2020. 2

[28] Wenbo Li, Yaodong Cui, Yintao Ma, Xingxin Chen, Guofa Li, Guanzhong Zeng, Gang Guo, and Dongpu Cao. A spontaneous driver emotion facial expression (defe) dataset for intelligent vehicles: Emotions triggered by video-audio clips in driving scenarios. IEEE Transactions on Affective Computing, 14(1):747–760, 2021. 1

[29] Wenbo Li, Guanzhong Zeng, Juncheng Zhang, Yan Xu, Yang Xing, Rui Zhou, Gang Guo, Yu Shen, Dongpu Cao, and Fei-Yue Wang. Cogemonet: A cognitive-feature-augmented driver emotion recognition model for smart cockpit. IEEE Transactions on Computational Social Systems, 9(3):667– 678, 2021. 1, 2, 3

[30] Wei-Hong Li and Hakan Bilen. Knowledge distillation for multi-task learning. In Computer Vision–ECCV 2020 Workshops: Glasgow, UK, August 23–28, 2020, Proceedings, Part VI 16, pages 163–176. Springer, 2020. 2

[31] Zhiwei Li, Tingzhen Zhang, Meihua Zhou, Dandan Tang, Pengwei Zhang, Wenzhuo Liu, Qiaoning Yang, Tianyu Shen, Kunfeng Wang, and Huaping Liu. Mipd: A multi-sensory interactive perception dataset for embodied intelligent driving, 2024. 1

[32] Wenzhuo Liu, Yan Gong, Guoying Zhang, Jianli Lu, Yunlai Zhou, and Junbin Liao. Glmdrivenet: Global–local multimodal fusion driving behavior classification network. Engineering Applications of Artificial Intelligence, 129:107575, 2024. 2, 7

[33] Wenzhuo Liu, Jianli Lu, Junbin Liao, Yicheng Qiao, Guoying Zhang, Jiayin Zhu, Bozhang Xu, and Zhiwei Li. Fmdnet: Feature-attention-embedding-based multimodal-fusion driving-behavior-classification network. IEEE Transactions on Computational Social Systems, 2024. 2

[34] Yun Liu, Yu-Huan Wu, Guolei Sun, Le Zhang, Ajad Chhatkuli, and Luc Van Gool. Vision transformers with hierarchical attention. Machine Intelligence Research, pages 1–14, 2024. 7

[35] Ze Liu, Jia Ning, Yue Cao, Yixuan Wei, Zheng Zhang, Stephen Lin, and Han Hu. Video swin transformer. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 3202–3211, 2022. 6, 7

[36] Jiaqi Ma, Zhe Zhao, Xinyang Yi, Jilin Chen, Lichan Hong, and Ed H Chi. Modeling task relationships in multi-task learning with multi-gate mixture-of-experts. In Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining, pages 1930–1939, 2018. 2

[37] Ningning Ma, Xiangyu Zhang, Hai-Tao Zheng, and Jian Sun. Shufflenet v2: Practical guidelines for efficient cnn architecture design. In European Conference on Computer Vision (ECCV), pages 116–131, 2018. 7

[38] Manuel Martin, Alina Roitberg, Monica Haurilet, Matthias Horne, Simon Reiß, Michael Voit, and Rainer Stiefelhagen.

Drive&act: A multi-modal dataset for fine-grained driver behavior recognition in autonomous vehicles. In Proceedings ofthe IEEE/CVF International Conference on Computer Vi sion, pages 2801–2810, 2019. 1, 2

[39] Sujitha Martin, Sourabh Vora, Kevan Yuen, and Mohan Manubhai Trivedi. Dynamics of driver’s gaze: Explorations in behavior modeling and maneuver prediction. IEEE Transactions on Intelligent Vehicles, 3(2):141–150, 2018. 1

[40] Ishan Misra, Abhinav Shrivastava, Abhinav Gupta, and Martial Hebert. Cross-stitch networks for multi-task learning. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3994–4003, 2016. 2

[41] Luntian Mou, Yiyuan Zhao, Chao Zhou, Bahareh Nakisa, Mohammad Naim Rastgoo, Lei Ma, Tiejun Huang, Baocai Yin, Ramesh Jain, and Wen Gao. Driver emotion recognition with a hybrid attentional multimodal fusion framework. IEEE Transactions on Affective Computing, 14(4): 2970–2981, 2023. 1, 3

[42] World Health Organization. Global status report on road safety 2018. World Health Organization, 2019. 1

[43] Yeqiang Qian, John M Dolan, and Ming Yang. Dlt-net: Joint detection of drivable areas, lane lines, and traffic objects. IEEE Transactions on Intelligent Transportation Systems, 21 (11):4670–4679, 2019. 1, 2

[44] Yao Rong, Zeynep Akata, and Enkelejda Kasneci. Driver intention anticipation based on in-cabin and driving scene monitoring. In 2020 IEEE 23rd International Conference on Intelligent Transportation Systems (ITSC), pages 1–8. IEEE, 2020. 2

[45] Khaled Saleh, Mohammed Hossny, and Saeid Nahavandi. Driving behavior classification based on sensor data fusion using lstm recurrent neural networks. In 2017 IEEE 20th International Conference on Intelligent Transportation Systems (ITSC), pages 1–6. IEEE, 2017. 1, 2

[46] Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, and Liang-Chieh Chen. Mobilenetv2: Inverted residuals and linear bottlenecks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 4510–4520, 2018. 7

[47] Xiaoqiang Shi, Zhenyu Yin, Guangjie Han, Wenzhuo Liu, Li Qin, Yuanguo Bi, and Shurui Li. Bssnet: A real-time semantic segmentation network for road scenes inspired from autoencoder. IEEE Transactions on Circuits and Systemsfor Video Technology, 2023. 6

[48] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556, 2014. 6, 7

[49] Marvin Teichmann, Michael Weber, Marius Zoellner, Roberto Cipolla, and Raquel Urtasun. Multinet: Real-time joint semantic reasoning for autonomous driving. In 2018 IEEE intelligent vehicles symposium (IV), pages 1013–1020. IEEE, 2018. 2

[50] Renran Tian, Lingxi Li, Mingye Chen, Yaobin Chen, and Gerald J Witt. Studying the effects of driver distraction and traffic density on the probability of crash and near-crash events in naturalistic driving environment. IEEE Transactions on Intelligent Transportation Systems, 14(3):1547– 1555, 2013. 1

[51] Du Tran, Lubomir Bourdev, Rob Fergus, Lorenzo Torresani, and Manohar Paluri. Learning spatiotemporal features with 3d convolutional networks. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV), pages 4489–4497, 2015. 7

[52] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in Neural Information Processing Systems (NIPS), 30, 2017. 6, 7

[53] Huiyu Wang, Yukun Zhu, Bradley Green, Hartwig Adam, Alan Yuille, and Liang-Chieh Chen. Axial-deeplab: Standalone axial-attention for panoptic segmentation. In European conference on computer vision, pages 108–126. Springer, 2020. 2

[54] Xiaoyu Wang, Kangyao Huang, Xinyu Zhang, Honglin Sun, Wenzhuo Liu, Huaping Liu, Jun Li, and Pingping Lu. Path planning for air-ground robot considering modal switching point optimization. In 2023 International Conference on Unmanned Aircraft Systems (ICUAS), pages 87–94. IEEE, 2023. 1

[55] Xiaofeng Wang, Zheng Zhu, Wenbo Xu, Yunpeng Zhang, Yi Wei, Xu Chi, Yun Ye, Dalong Du, Jiwen Lu, and Xingang Wang. Openoccupancy: A large scale benchmark for surrounding semantic occupancy perception. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 17850–17859, 2023. 1

[56] Dong Wu, Man-Wen Liao, Wei-Tian Zhang, Xing-Gang Wang, Xiang Bai, Wen-Qing Cheng, and Wen-Yu Liu. Yolop: You only look once for panoptic driving perception. Machine Intelligence Research, 19(6):550–562, 2022. 2

[57] Yang Xing, Chen Lv, Dongpu Cao, and Efstathios Velenis. Multi-scale driver behavior modeling based on deep spatialtemporal representation for intelligent vehicles. Transportation research part C: emerging technologies, 130:103288, 2021. 1, 2

[58] Dan Xu, Wanli Ouyang, Xiaogang Wang, and Nicu Sebe. Pad-net: Multi-tasks guided prediction-and-distillation network for simultaneous depth estimation and scene parsing. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 675–684, 2018. 2

[59] Xiaogang Xu, Hengshuang Zhao, Vibhav Vineet, Ser-Nam Lim, and Antonio Torralba. Mtformer: Multi-task learning via transformer and cross-task reasoning. In European Conference on Computer Vision, pages 304–321. Springer, 2022. 2

[60] Yangyang Xu, Yibo Yang, and Lefei Zhang. Demt: Deformable mixer transformer for multi-task learning of dense prediction. In Proceedings of the AAAI conference on artificial intelligence, pages 3072–3080, 2023. 2, 5

[61] Yijie Xun, Jiajia Liu, and Zhenjiang Shi. Multitask learning assisted driver identity authentication and driving behavior evaluation. IEEE Transactions on Industrial Informatics, 17 (10):7093–7102, 2020. 2

[62] Dingkang Yang, Shuai Huang, Zhi Xu, Zhenpeng Li, Shunli Wang, Mingcheng Li, Yuzheng Wang, Yang Liu, Kun Yang, Zhaoyu Chen, et al. Aide: A vision-driven multi-view, multimodal, multi-tasking dataset for assistive driving perception.

In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 20459–20470, 2023. 1, 2, 6

[63] Lie Yang, Haohan Yang, Bin-Bin Hu, Yan Wang, and Chen Lv. A robust driver emotion recognition method based on high-purity feature separation. IEEE Transactions on Intelligent Transportation Systems, 2023. 1, 3

[64] Sebastian Zepf, Javier Hernandez, Alexander Schmitt, Wolfgang Minker, and Rosalind W Picard. Driver emotion recog nition for intelligent vehicles: A survey. ACM Computing Surveys (CSUR), 53(3):1–30, 2020. 1, 3

[65] Xiangyu Zhang, Xinyu Zhou, Mengxiao Lin, and Jian Sun. Shufflenet: An extremely efficient convolutional neural network for mobile devices. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 6848–6856, 2018. 6, 7

[66] Xinyu Zhang, Yan Gong, Jianli Lu, Zhiwei Li, Shixiang Li, Shu Wang, Wenzhuo Liu, Li Wang, and Jun Li. Oblique convolution: A novel convolution idea for redefining lane detection. IEEE Transactions on Intelligent Vehicles, 2023. 1

[67] Yongshuai Zhi, Zhipeng Bao, Sumin Zhang, and Rui He. Bigru based online multi-modal driving maneuvers and trajectory prediction. Proceedings ofthe institution ofmechanical engineers, part d: journal of automobile engineering, 235 (14):3431–3441, 2021. 1

[68] Bolei Zhou, Agata Lapedriza, Aditya Khosla, Aude Oliva, and Antonio Torralba. Places: A 10 million image database for scene recognition. IEEE Transactions on Pattern Analy sis and Machine Intelligence, 40(6):1452–1464, 2017. 7

[69] Lei Zhu, Xinjiang Wang, Zhanghan Ke, Wayne Zhang, and Rynson WH Lau. Biformer: Vision transformer with bi-level routing attention. In Proceedings of the IEEE/CVF con ference on computer vision and pattern recognition, pages 10323–10333, 2023. 2, 4