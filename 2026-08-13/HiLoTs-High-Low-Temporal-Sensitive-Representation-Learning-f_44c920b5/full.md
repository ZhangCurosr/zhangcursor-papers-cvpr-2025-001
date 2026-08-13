# HiLoTs: High-Low Temporal Sensitive Representation Learning for Semi-Supervised LiDAR Segmentation in Autonomous Driving

R.D. Lin<sup>1</sup>, Pengcheng Weng<sup>1</sup>, Yinqiao Wang<sup>1</sup>, Han Ding<sup>2</sup>, Jinsong Han<sup>3</sup>, Fei Wang<sup>1†</sup> <sup>1</sup> School of Software Engineering, Xi’an Jiaotong University, China <sup>2</sup> School of Computer Science and Technology, Xi’an Jiaotong University, China <sup>3</sup> College of Computer Science and Technology, Zhejiang University, China   
rdlin@stu.xjtu.edu.cn, hanjinsong@zju.edu.cn, {dinghan,feynmanw}@xjtu.edu.cn <sup>†</sup>corresponding author and project lead

https://github.com/rdlin118/HiLoTs

## Abstract

LiDAR point cloud semantic segmentation plays a crucial role in autonomous driving. In recent years, semisupervised methods have gained popularity due to their significant reduction in annotation labor and time costs. Current semi-supervised methods typically focus on point cloud spatial distribution or consider short-term temporal representations, e.g., only two adjacent frames, often overlooking the rich long-term temporal properties inherent in autonomous driving scenarios. In driving experience, we observe that nearby objects, such as roads and vehicles, remain stable while driving, whereas distant objects exhibit greater variability in category and shape. This natural phenomenon is also captured by Li-DAR, which reflects lower temporal sensitivity for nearby objects and higher sensitivity for distant ones. To leverage these characteristics, we propose HiLoTs, which learns high-temporal sensitivity and low-temporal sensitivity representations from continuous LiDAR frames. These representations are further enhanced and fused using a crossattention mechanism. Additionally, we employ a teacherstudent framework to align the representations learned by the labeled and unlabeled branches, effectively utilizing the large amounts of unlabeled data. Experimental results on the SemanticKITTI and nuScenes datasets demonstrate that our proposed HiLoTs outperforms state-of-the-art semisupervised methods, and achieves performance close to Li-DAR+Camera multimodal approaches.

## 1. Introduction

LiDAR point cloud semantic segmentation is crucial in autonomous driving for tasks such as obstacle avoidance [45], lane detection [13], localization and mapping [5, 14].

Road Sidewalk Vegetation Terrain Car Bicycle Building Fence Person  
![](images/7dab4cfe7b18e3a85b86e75340c432e36b83e8c27ab837807353cc33c6d62e4f.jpg)  
Figure 1. Different semantic classes exhibit varying degrees of sensitivity to temporal changes. Objects farther from the vehicle (e.g., vegetation, building, person, etc.) change more frequently over time, as indicated by the red box. In contrast, objects closer to the vehicle (e.g., road, sidewalk, etc.) are less sensitive to temporal changes, as shown by the blue box.

Most existing segmentation works are fully-supervised approaches [1, 11, 24, 46], which have several drawbacks. They require extensive point-wise annotations, leading to significant labor costs and time consumption. Additionally, the need for large labeled datasets limits their scalability and adaptability to new environments, highlighting the need for more efficient and scalable approaches, such as semisupervised learning (SSL) methods that can leverage unlabeled data and reduce reliance on costly annotations.

To achieve SSL LiDAR segmentation, many works leverage the spatial distribution information of point clouds [8, 23, 27]. For example, SSPC-Net [8] introduces an area partition method, constructing a super-point graph structure with both labeled and unlabeled point clouds for semi-supervised learning. DDSemi [23] addresses outlier issues in point-to-point SSL methods by employing a pointdensity-guided contrastive learning technique. In addition to utilizing the spatial distribution of point clouds, some works exploit temporal information [2, 9, 29, 33]. For instance, Aygun et al. [¨ 2] directly fuses multi-frame point clouds and inputs them into a standard encoder-decoder network to produce segmentation results. Shi et al. [33] proposes a temporal matching method, performing one-toone matching based on differences and similarities between point clouds of two consecutive frames. Although these methods demonstrate strong performance, they tend to focus on either the spatial or the temporal characteristics of point clouds, without fully integrating both aspects.

In driving experience, we observe a phenomenon: objects closer to the vehicle, such as roads and cars, tend to have stable categories and shapes as the vehicle moves, while distant objects, such as pedestrians, guardrails, plants, and buildings, exhibit significant variations in category and shape. Surprisingly, this nature is also reflected in LiDAR point cloud data, as shown in Fig. 1, where the relevant areas are highlighted. To leverage this phenomenon, we propose HiLoTs, which consists of a High Temporal Sensitivity Flow (HTSF) and a Low Temporal Sensitivity Flow (LTSF). The HTSF focuses on regions where distant objects experience significant changes in category and shape, while the LTSF focuses on nearby regions where object categories and shapes remain relatively stable. Furthermore, the features from HTSF and LTSF are fused and interact through a cross-attention mechanism. To better represent near and far objects, we convert the point cloud into cylindrical voxels using a cylindrical voxelization network [46]. To further optimize computational efficiency, we aggregate multiple spatiotemporally neighboring cylindrical voxels, enabling a more efficient computation of HTSF and LTSF.

For semi-supervised point cloud segmentation, we adopt the mainstream Mean Teacher architecture [35], which effectively leverages a small amount of labeled LiDAR frames and a large amount of unlabeled data. In HiLoTs, the labeled LiDAR frames are fed into the student network, while the unlabeled LiDAR frames are processed through the teacher network. In each iteration, a consistency loss is computed to align the predictions made by the teacher network with those from the student network. The student network gradually updates the teacher network’s parameters, a process that can be likened to the student slowly growing into the teacher. After training, the teacher network is used for the LiDAR segmentation during inference.

We evaluate HiLoTs on two widely-used autonomous driving benchmarks, SemanticKITTI and nuScenes. Extensive results show that HiLoTs outperforms the latest

LiDAR-only semi-supervised methods and achieves performance comparable to multimodal approaches such as [6, 19], which combine LiDAR and camera data. Additionally, ablation studies confirm the effectiveness of the proposed HTSF and LTSF components, aligning with our observations from driving experience. In summary, our work claims the following main contributions:

• We observe a natural but often overlooked phenomenon in driving as shown in Fig. 1. We propose HiLoTs, designed to focus on this characteristic. We believe this design could provide valuable insight for future advancements in LiDAR segmentation tasks.

• HiLoTs includes several novel techniques, such as multivoxel aggregation and temporal sensitivity embedding units, which are efficient and effective in LiDAR spatiotemporal representation learning.

• Experimental results show that HiLoTs surpasses the latest semi-supervised methods and achieves performance comparable to LiDAR+Camera multimodal methods.

## 2. Related Work

## 2.1. Semi-supervised LiDAR Segmentation

Various works utilize LiDAR to capture objects’ 3D representation since point clouds can accurately reflect the structural characteristics of objects [20, 22, 24, 25, 36]. However, most of the existing works are based on fullysupervised learning, while it is difficult to annotate point clouds due to the intensive labor and time costs. Consequently, many works have attempted to leverage SSL methods [8, 15, 18, 23, 26, 37, 41]. GPC [15] uses large amounts of unlabeled data as pseudo-label guidance to reduce the negative impact of intra-class negative pairs, achieving good results in both indoor and outdoor scenarios. LaserMix [18] deeply analyzes point cloud distribution priors of various objects and performed fusion between labeled and unlabeled point clouds. Nevertheless, continuous point clouds in autonomous driving usually contain rich temporal features. Current SSL methods mainly focus on spatial feature extraction, neglecting the inherent temporal representations of outdoor point clouds.

## 2.2. Point Cloud Representation

Mainstream outdoor point cloud representation learning can be analyzed from both spatial and temporal perspectives. For spatial representation, since 3D point clouds are unordered sets [31], it is necessary to first convert the point clouds into volumetric grids, where it can be input into neural networks as tensors. Current mainstream approaches include range mapping [1, 16], cubic-voxelization [44], pillarbased voxelization [22, 25], spherical [20] and cylindrical representation [46]. For temporal representation, commonly used methods can be divided into data-level [2, 9, 29] and feature-level [33] multi-frame fusion. However, datalevel fusion [2, 9] directly increases computational overhead, making hardware resources a primary bottleneck. Also, current feature-level fusion in semi-supervised learning [33] is based on two adjacent frames, which is insufficient to encode temporal changes compared to multiple frames. We observe that during driving, distant objects show frequent changes in both object categories and shapes over time, while closer objects exhibit more stable distribution. To fully capture this property, we propose HiLoTs.

![](images/a8c3835f2280f98163fe584b3cf565ff24dc35d5a5149c6a68ebda952e4687aa.jpg)  
Figure 2. Our segmentation model involves three stages. During voxelization, cylindrical voxelization is applied to transform unordered points into volumetric grids, followed by a spatial feature extraction backbone. Then, HiLoTs processes the labeled and unlabeled cylin drical features through a student-teacher framework. It also integrates the attention map from HiLoTs embedding unit (HEU) to produce voxel-level segmentation maps. Finally, a point-wise refinement network is utilized to obtain point-level segmentation results

## 3. Methods

Fig. 2 illustrates the overall pipeline of the proposed method. We first introduce the preliminary of the task. Next, we provide a detailed description of each component in the proposed HiLoTs. Finally, we present the loss functions for training the overall network.

Preliminary. Suppose $D = \{ D _ { i } \ | \ i = 1 , \cdot \cdot \ , N \}$ denotes the point cloud dataset, where $N$ is the total number of sequences. Each sequence $D _ { i }$ includes $t _ { i }$ point cloud frames, $\{ f _ { j } ~ \mid ~ j ~ = ~ 1 , \cdot \cdot \cdot ~ , t _ { i } \} . ~ f _ { j } ~ = ~ ( x _ { p } , y _ { p } , z _ { p } , r _ { p } ) ~ \in ~$ $\mathbb { R } ^ { P \times 4 }$ represents P point clouds in one frame, where $( x _ { p } , y _ { p } , z _ { p } )$ and $r _ { p }$ denote the Cartesian coordinates and LiDAR intensity, respectively. In the semi-supervised semantic segmentation task, given a supervised ratio of $s \% .$ $D _ { i } = [ f _ { 1 } , \cdots , f _ { t _ { i } } ]$ contains uniformly sampled s% labeled frames, and $( 1 - s ) \%$ unlabeled frames.

## 3.1. Voxelization with Cylindrical Network

Common visual modalities such as images and videos are in volumetric grids, which are efficient for neural network processing. However, point clouds are unordered sets [31]. To transform point clouds to volumetric grids for network processing, we employ a mainstream and efficient feature extraction method, namely cylindrical voxelization [46], which can adequately represent point clouds of different densities at different ranges.

The first step of cylindrical voxelization is to convert the Cartesian coordinates of each point, represented as $( x , y , z )$ , into the corresponding cylindrical coordinates $( \rho , \theta , z )$ , where $\rho = { \sqrt { x ^ { 2 } + y ^ { 2 } } }$ and $\begin{array} { r } { \theta = \arctan ( \frac { y } { x } ) } \end{array}$ represent the radial distance and the azimuth, respectively.

Next, since LiDAR captures dense point clouds in near areas and sparse point clouds in far areas, during voxelization, we set the size of cylindrical cells to increase with the distance, maintaining a balanced number of points in different cylindrical cells. Each cell contains the cylindrical coordinate $( \rho , \theta , z )$ of the original point cloud, with the remission $r _ { \epsilon } .$ For multiple points projected onto the same cell, we retain the point with the closest range. The resulting volumetric grid of each frame is $\boldsymbol { x } _ { f } \in \mathbb { R } ^ { \breve { \mathrm { R } } \times \Theta \times H \times C }$ , where C = 4 containing $( \rho , \theta , z , r _ { \epsilon } )$ , and $\mathrm { R } , \Theta$ and H represent the maximum radius, azimuth, and height, respectively.

Further, we employ 3D ResNet50 [12] to extract the initial cylindrical features for the following LiDAR segmentation, $\dot { \boldsymbol { x } } _ { f }  \boldsymbol { x } _ { f } \in \mathbb { R } ^ { M \times d }$ , where $M = \mathrm { R } \times \Theta \times H , d = 2 5 6$

## 3.2. HiLoTs Embedding Unit

We observe that nearby objects, mainly roads and vehicles, remain relatively stable in their spatial and categorical characteristics during driving, while distant objects display greater variability in both category and shape. Leveraging this observation, we propose HiLoTs Embedding Unit (HEU) that distinguishes between high temporally sensitive features for distant scenes and low temporally sensitive features for nearby scenes. By capturing these distinctions, HiLoTs can more accurately interpret dynamic scenes and better adapt to the varying spatial-temporal characteristics across different distances.

![](images/9e48c018806ec6589d277f7f0d706981461e0f332ee1dc557cdbe4b4aa56c24b.jpg)  
Figure 3. HiLoTs Embedding Unit (HEU). The distant voxel features are passed into high temporal sensitivity flow, while voxel features in closer areas undergo low temporal sensitivity flow. The output map of HEU is fused with the bottleneck feature map from the segmentation model, further passed into the decoder.

As shown in Fig. 3, HEU processes two types of initial cylindrical features, which first undergo Multi-Voxel Aggregation. Voxels located at distant ranges are directed through the High Temporal Sensitivity Flow, while those within closer ranges are processed via the Low Temporal Sensitivity Flow. The resulting feature maps from each flow are then integrated, which facilitates interaction between the high- and low-sensitivity features. These enriched features subsequently serve as inputs to the segmentation network.

## 3.2.1 Multi-Voxel Aggregation

Recent Transformer-based models [10, 21, 34, 38, 42] have demonstrated remarkable performance in various tasks such as LiDAR semantic segmentation [1, 20]. Given that Transformers serve as a unifying foundation for multimodal modeling, we also employ a Transformer-based architecture to implement our HEU model. However, the computational complexity of the attention mechanism is $O ( n ^ { \bar { 2 } } )$ ), where n represents the number of tokens. With the extensive number of voxels in outdoor scenes, applying the attention mechanism to all voxels becomes computationally prohibitive. To address this, we designed a multi-voxel aggregation method (MVA), which groups neighboring voxels into super-voxels. This approach not only significantly reduces the token count, thus alleviating computational demands, but also aggregates voxel features, enhancing the model’s performance by capturing more coherent semantic information within each super-voxel. As shown in Fig. 3, the proposed MVA includes two steps. (1) Spatial aggregation: we aggregate M cylindrical voxels into m super voxels in spatial as:

$$
F ^ { \prime } = \mathrm { M L P } _ { \theta _ { 1 } } ( \mathrm { N N G r o u p } ( F , m ) )\tag{1}
$$

where $F \in \mathbb { R } ^ { M \times d \times t }$ represents cylindrical features across t frames; $\mathrm { { N N G r o u p } ( \cdot ) }$ represents nearest neighbor grouping and $\mathrm { M L P } _ { \theta _ { 1 } }$ is a multi-layer perceptron (MLP) with parameters of $\theta _ { 1 }$ $F ^ { \prime } \in \mathbb { R } ^ { m \times \bar { d } \times t } ( m \textless M )$ is the spatially aggregated cylindrical features.

(2) Temporal fusion: we further fuse super-voxel features across temporal dimension as:

$$
V = \mathrm { M L P } _ { \theta _ { 2 } } ( \operatorname { A v g P o o l } ( F ^ { \prime } ) )\tag{2}
$$

where AvgPool(·) represents temporal average pooling and $\mathrm { M L P } _ { \theta _ { 2 } }$ is an MLP with parameters of ${ \boldsymbol { \theta } } _ { 2 } . ~ V \in \mathbb { R } ^ { m \times d }$ is the final output of the Multi-Voxel Aggregation process.

## 3.2.2 High Temporal Sensitivity Flow

The High Temporal Sensitivity Flow (HTSF) is designed to process aggregated voxels at greater distances, e.g., those within the farthest 70% of the range. This flow enables point cloud representation learning by focusing on temporally dynamic and spatially variable distant regions, where object categories and shapes are more likely to fluctuate. The process can be represented as follows:

$$
\begin{array} { l } { { V _ { i } = \mathrm { S o f t m a x } \left( \frac { \mathrm { d o t } ( V _ { i } W _ { Q } , V _ { i } W _ { K } ) } { \sqrt { d _ { k } } } \right) V _ { i } W _ { i } } } \\ { { V _ { i + 1 } = \mathrm { M L P } _ { i } ( V _ { i } ) } } \end{array}\tag{3}
$$

where $V _ { i } \in \mathbb { R } ^ { m \times d }$ represents the far-range voxel features of the i-th encoder layer, $W _ { Q } \in \mathbb { R } ^ { d \times d _ { k } }$ ， $W _ { K } \in \mathbb { R } ^ { d \times d _ { k } }$ and $W _ { V } ~ \in ~ \mathbb { R } ^ { d \times d _ { v } }$ denotes the weight parameters for Query, Key and Value matrices. $\operatorname* { d o t } ( \cdot , \cdot )$ denotes matrix multiplication. $V _ { i + 1 } \in \mathbb { R } ^ { m \times d }$ represents the output of the current encoder layer, which is also the input to the next layer.

## 3.2.3 Low Temporal Sensitivity Flow

As discussed, the point cloud distribution of closer objects, such as roads, remains relatively stable over time in LiDAR point cloud. To capture these low temporally sensitive features accurately, we design the Low Temporal Sensitivity Flow (LTSF), depicted in the lower part of Fig. 3. In this process, all distant voxels are first excluded from the voxel set, leaving only those that represent low temporal sensitivity voxel features. These selected voxels then undergo Multi-Voxel Aggregation and self-attention, similar to the HTSF process. Further, a cross-attention is applied with the attention map generated by HTSF, enabling an interaction between the high and low temporal sensitivity features.

By stacking multiple HTSF and LTSF layers, we obtain the final output of the HiLoTs Embedding Unit. To fuse the HEU output with the bottleneck encoder feature map of the segmentation network, we introduce two learnable parameters, $\alpha _ { 1 }$ and $\alpha _ { 2 } .$ , which control the feature fusion process.

$$
S = \alpha _ { 1 } \cdot { \mathrm { B o t t l e N e c k } } _ { \mathrm { E n } } ( x _ { f } ) + \alpha _ { 2 } \cdot { \mathrm { H E U } } ( F )\tag{4}
$$

where $x _ { f }$ and $F$ are described in Sec. 3.1 and Equation. 1. Finally, the decoder of the bottleneck network takes $S$ as input and outputs the voxel-level features, denoted as Bottle $\mathrm { V e c k } _ { \mathrm { D e } } ( \bar { S ) }  S \in \mathbb { R } ^ { M \times d }$ . We use Minkowski-UNet [9] as the bottleneck network for segmentation.

## 3.3. Voxel to Point Cloud Results

Since HiLoTs employs a voxelized approach to convert point clouds into volumetric grids, this inevitably leads to information loss when points from different objects are mapped to the same voxel. We first reverse voxel-level features S to point-level features, $S \to \mathbb { R } ^ { P \times d }$ , where P represents the number of points in the frame, based on the pointto-voxel mapping table from voxelization process described in Sec. 3.1. Then, we apply a point-wise refinement network [46] to output point-level segmentation results, where S are fused with the point coordinates and LiDAR intensity.

$$
\hat { y } = \mathrm { R e f i n e N e t } ( S , ( x _ { p } , y _ { p } , z _ { p } , r _ { p } ) )\tag{5}
$$

where $\hat { y } \in \mathbb { R } ^ { P \times K }$ represents K object classification confidences for $P$ points, i.e., semantic segmentation results.

## 3.4. Semi-supervised Learning and Loss Functions

Our HiLoTs model adopts Mean Teacher architecture [35] to achieve semi-supervised segmentation of point clouds. It includes two segmentation networks, namely the student and teacher network. The student network receives the labeled point clouds and uses the corresponding ground-truth for supervised training. Specifically, we use focal loss [28] as the supervised loss function, which addresses the class imbalance problem in point cloud semantic segmentation.

$$
L _ { s u p } = \mathrm { F o c a l L o s s } ( \hat { y } _ { s } , y )\tag{6}
$$

where $\hat { y } _ { s }$ represents the student network’s segmentation prediction and y represents the ground-truth labels.

In contrast, the teacher network receives unlabeled data, which are also fed into the student network. The teacher network’s loss function is the consistency loss between the two networks, represented as:

$$
L _ { c o n } = \| \hat { y } _ { s } - \hat { y } _ { t } \| _ { 2 }\tag{7}
$$

where $\hat { y } _ { t }$ denotes the prediction from the teacher network; $\lVert \cdot \rVert _ { 2 }$ represents the $\mathcal { L } _ { 2 }$ norm. After obtaining the supervised loss and consistency loss, the final loss of the model is the weighted sum of the two:

$$
L = \alpha L _ { s u p } + \beta L _ { c o n }\tag{8}
$$

where α and $\beta$ are to balance two losses. In our experiments, we set both to 1.

The weights of the teacher network are initialized by exponential moving average (EMA) [35], transferring the student network’s trained parameters to the teacher model, which can be represented as follows:

$$
W _ { t } ^ { \prime } = \gamma W _ { t - 1 } ^ { \prime } + ( 1 - \gamma ) W _ { t }\tag{9}
$$

where $W _ { t } ^ { \prime }$ and $W$ are for teacher network and student network at the time of t, respectively. The process of meanteacher semi-supervised learning is akin to a student gradually growing into a teacher, slowly absorbing knowledge and refining their understanding over time. Ultimately, we obtain the teacher network, which is then used for inference.

## 3.5. Implementation Details

HiLoTs is trained for 50,000 iterations with an early stopping strategy. AdamW optimizer [30] is used with the initial learning rate set to 1e-3. All experiments are conducted on a server with four RTX 3090 GPUs, using a batch size of 4 per GPU, resulting in a total batch size of 16. In cylindrical voxelization, we set the maximum point cloud range, azimuth, and height to (0, 50) meters, (−π, π), and (−4, 2) meters, respectively. The resolution of voxelized grids is set to (240, 180, 20). Considering both the model performance and the GPU memory cost, we set $t = 5$ as the temporal length in all experiments. The layers of both encoder and decoder of the Transformer are set to $N = 6$

During training, HiLoTs takes 1 frame (central frame) and its neighboring t − 1 frames as input. If the central frame is labeled, it is processed by the labeled branch for supervised segmentation. Otherwise, it is routed to the unlabeled branch. Regardless of label presence, all t frames pass through the HEU module. During inference, HiLoTs takes t-frame LiDAR point clouds as input and leverages the teacher network to estimate semantic segmentation for the central frame.

## 4. Experiments and Analysis

## 4.1. Datasets and Evaluation Metric

We evaluate our approach on two widely-used autonomous driving datasets: SemanticKITTI [3] and nuScenes [4]. Li-DAR point clouds from SemanticKITTI were collected with a 64-beam Velodyne HDL-64E device at 10Hz, including 19 semantic classes. Sequences 00-07 and 09-10 serve as the training set, with 19120 point cloud scenes, while sequence 08 is used as the validation set, with 4070 scenes. The nuScenes dataset, also widely used, was collected with a 32-beam Velodyne HDL-32E device at 20Hz. It contains 16 semantic classes, with the training and validation sets of 27287 and 5850 point cloud scenes, respectively.

Following prior semi-supervised segmentation works [18, 23, 26], we report the model’s performance on the validation sets of both datasets. We set the labeled ratio in {1%, 10%, 20%, 50%} and use mean Intersection-over-Union (mIoU) scores across all semantic classes as the evaluation metric.

<table><tr><td rowspan="2">Methods</td><td rowspan="2">Modality</td><td colspan="4">SemanticKITTI</td><td colspan="4">nuScenes</td></tr><tr><td>1%</td><td>10%</td><td>20%</td><td>50%</td><td>1%</td><td>10%</td><td>20%</td><td>50%</td></tr><tr><td>Cylinder3D [46] (2021)</td><td>Lidar</td><td>45.4</td><td>56.1</td><td>57.8</td><td>58.7</td><td>53.4*</td><td>63.4*</td><td>67.0*</td><td>71.9*</td></tr><tr><td>RangeViT [1] (2023)</td><td>Lidar</td><td>43.8*</td><td>53.4*</td><td>56.6*</td><td>58.8*</td><td>53.8*</td><td>64.6*</td><td>67.8*</td><td>73.1*</td></tr><tr><td>SphereFormer [20] (2023)</td><td>Lidar</td><td>41.2*</td><td>59.8*</td><td>60.6*</td><td>62.4*</td><td>49.5*</td><td>65.3*</td><td>69.2*</td><td>73.7*</td></tr><tr><td>MarS3D [29] (2023)</td><td>Lidar</td><td>44.5*</td><td>58.6*</td><td>60.2*</td><td>61.7*</td><td>51.8*</td><td>65.5*</td><td>68.4*</td><td>72.8*</td></tr><tr><td>GPC [15] (2021)</td><td>Lidar</td><td>41.8</td><td>49.9</td><td>58.8</td><td>59.9</td><td></td><td></td><td></td><td></td></tr><tr><td>PolarMix [39] (2022)</td><td>Lidar</td><td>50.1</td><td>60.9</td><td>62.0</td><td>63.8</td><td>55.6</td><td>69.6</td><td>71.0</td><td>73.8</td></tr><tr><td>LaserMix [18] (2023)</td><td>Lidar</td><td>50.6</td><td>60.0</td><td>61.9</td><td>62.3</td><td>55.3</td><td>69.9</td><td>71.8</td><td>73.2</td></tr><tr><td>LiM3D [26] (2023)</td><td>Lidar</td><td>58.4</td><td>62.2</td><td>63.1</td><td>63.6</td><td>一</td><td></td><td></td><td></td></tr><tr><td>ImageTo360 [32] (2023)</td><td>Lidar</td><td>54.5</td><td>58.6</td><td>61.4</td><td>64.2</td><td>-</td><td></td><td>-</td><td>一</td></tr><tr><td>IGNet [37] (2024)</td><td>Lidar</td><td>49.0</td><td>61.3</td><td>63.1</td><td>64.8</td><td></td><td></td><td></td><td></td></tr><tr><td>DDSemi [23] (2024)</td><td>Lidar</td><td>59.3</td><td>65.1</td><td>66.3</td><td>67.0</td><td>58.1</td><td>70.2</td><td>74.0</td><td>76.5</td></tr><tr><td>FRNet [40] (2025)</td><td>Lidar</td><td>55.8</td><td>64.8</td><td>65.2</td><td>65.4</td><td>61.2</td><td>72.2</td><td>74.6</td><td>75.4</td></tr><tr><td>CyMix+IPSL [6] (2024)</td><td>Lidar+Camera</td><td>52.8</td><td>64.8</td><td>64.9</td><td>65.9</td><td>59.1</td><td>76.0</td><td>78.7</td><td>80.5</td></tr><tr><td>LaserMix++ [19] (2024)</td><td>Lidar+Camera</td><td>63.2</td><td>67.5</td><td>67.7</td><td>68.6</td><td>65.3</td><td>75.3</td><td>75.2</td><td>76.3</td></tr><tr><td>HiLoTs (Ours)</td><td>Lidar</td><td>58.6</td><td>65.7</td><td>66.5</td><td>67.6</td><td>58.7</td><td>72.2</td><td>75.2</td><td>76.9</td></tr></table>

Table 1. Performance comparisons with current state-of-the-art. Methods highlighted in yellow represents fully-supervised methods, while blue denotes semi-supervised methods. Best scores are bolded and the second best scores are underlined. Results in ∗ are reproduced.

## 4.2. Performance Comparisons

Table 1 presents a comparison of HiLoTs with supervised methods with limited labeled data [1, 20, 29, 46], LiDARonly semi-supervised methods [15, 23, 26, 32, 37, 39, 40], and LiDAR+Camera semi-supervised methods [6, 19].

Comparisons with Fully-supervised Methods. Cylinder3D [46], RangeViT [1], SphereFormer [20], and MarS3D [29] are fully supervised methods that do not incorporate semi-supervised techniques. We re-train both methods using the same amount of labeled data as our semisupervised approach, with re-trained results marked with the <sup>∗</sup> in Table 1. HiLoTs consistently outperforms these fully supervised approaches by a notable margin. For instance, using only 1% SemanticKITTI labeled dataset, our method achieves a mIoU of 58.6, while Cylinder3D and MarS3D reach 45.4 and 44.5, respectively. This is because the mean-teacher semi-supervised strategy can effectively leverage unlabeled data to improve segmentation performance. Although involving more labeled data can gradually enhance the performance of fully supervised methods, the high labeling cost in customized point cloud segmentation tasks makes semi-supervised methods more feasible.

Comparisons with Semi-upervised Methods. In mainstream semi-supervised point cloud segmentation for autonomous driving, existing methods can be categorized into single-modal methods that rely solely on LiDAR, and multimodal methods that incorporate both LiDAR and camera.

As shown in Table 1, HiLoTs outperforms most current state-of-the-art LiDAR-only semi-supervised approaches, including recent methods such as FRNet [40] and DDSemi [23], across various labeling ratios. This demonstrates the effectiveness of the HiLoTs Embedding Unit module, described in Sec. 3.2, which leverages the distinct temporal and spatial variations of point clouds at different distances, making it highly effective for point cloud segmentation in autonomous driving.

We also compare our method with two recent LiDAR + Camera multimodal semi-supervised approaches, i.e., CyMix+IPSL [6] and LaserMix++ [19]. HiLoTs achieves performance comparable to LaserMix++ on both datasets, performing slightly better than CyMix+IPSL on SemanticKITTI dataset and slightly worse on nuScenes dataset. Overall, HiLoTs demonstrates considerable effectiveness even when compared to multimodal methods. Its advantage lies in not requiring a camera or labeled RGB data from the camera, making the system more costeffective and reducing the labeling cost.

Class-wise Performance. In terms of class-wise performance, as shown in Table 2, our method performs well on the class “parking” but shows weaker performance on the class “terrain”. This highlights HiLoTs’s capability to identify distant objects effectively. Additionally, all methods demonstrate relatively low performance on the “otherground”, “bicycle”, and “traffic-sign”, which remains a significant challenge in semi-supervised LiDAR point cloud segmentation. Future strategies could focus on improving recognition for these classes.

Segmentation Visualization. We further present typical examples of LiDAR point cloud semantic segmentation results at a 50% annotation ratio in Fig. 4, with orange boxes highlighting areas of significant differences between HiLoTs and LaserMix [18]. Compared to LaserMix, HiLoTs shows a distinct advantage in accurately segmenting distant areas with objects rapidly changing classes or shapes. This supports the design rationale behind our high and low temporal sensitivity flows, confirming the effectiveness of targeting regions with varying temporal dynamics.

<table><tr><td></td><td colspan="7">motorcycle</td><td colspan="7">parking</td><td colspan="7">vegetation</td></tr><tr><td>Methods</td><td>car</td><td>bicycle</td><td></td><td>truck</td><td>bus</td><td>person</td><td></td><td>bicyclist road</td><td></td><td></td><td>sidewalk</td><td>other-ground</td><td>building</td><td>fence</td><td></td><td>trunk</td><td>terrain</td><td>pole</td><td>traffic-sign</td><td>mIoU</td></tr><tr><td>Cylinder3D</td><td>94.1</td><td>38.8</td><td>43.9</td><td>42.1</td><td>38.2</td><td>57.3</td><td>72.0</td><td>91.4</td><td>38.1</td><td>76.4</td><td>0.4</td><td></td><td>89.7</td><td>56.0</td><td>86.4</td><td>59.7</td><td>70.6</td><td>60.5</td><td>45.5</td><td>55.9</td></tr><tr><td>RangeViT</td><td>92.0</td><td>29.4</td><td>37.0</td><td>52.6</td><td>33.6</td><td>43.5</td><td>60.4</td><td>93.9</td><td>41.5</td><td></td><td>79.9</td><td>1.4</td><td>83.8</td><td>52.1</td><td>84.3</td><td>57.7</td><td>72.4</td><td>57.2</td><td>39.7</td><td>53.4</td></tr><tr><td>GPC</td><td>94.1</td><td>1.4</td><td>53.1</td><td>47.2</td><td>28.1</td><td>43.2</td><td>0.0</td><td>93.1</td><td>28.6</td><td>77.4</td><td>0.1</td><td></td><td>87.6</td><td>39.4</td><td>87.8</td><td>56.4</td><td>77.8</td><td>60.8</td><td>50.6</td><td>48.8</td></tr><tr><td>LaserMix</td><td>95.8</td><td>13.5</td><td>52.1</td><td>71.4</td><td>50.2</td><td>66.4</td><td>55.9</td><td>93.6</td><td>48.9</td><td>81.8</td><td>0.4</td><td></td><td>91.6</td><td>66.1</td><td>88.6</td><td>66.6</td><td>75.4</td><td>65.0</td><td>49.5</td><td>59.7</td></tr><tr><td>HiLoTs</td><td>96.1</td><td>33.1</td><td>68.1</td><td>63.0</td><td>63.4</td><td>71.8</td><td>81.5</td><td>94.7</td><td>60.2</td><td>83.9</td><td>13.1</td><td></td><td>92.2</td><td>70.1</td><td>89.1</td><td>70.4</td><td>76.5</td><td>67.0</td><td>52.3</td><td>65.7</td></tr></table>

Table 2. Class-wise IoU score on the validation set of SemanticKITTI under 10% supervised ratio. Note that the class ‘motorcycle’ is omitted due to its low distribution in the validation set.

![](images/b92098ab542fd90480d9e1e194ce4a5bd00163083886ba8a6325ce5cf9a48edb.jpg)  
Figure 4. Error maps visualization (blue and red points are for correct predictions and incorrect predictions, respectively.). The left three columns are segmentation results from SemanticKITTI dataset, while the right three columns are from nuScenes. Our HiLoTs method shows a significant improvement in the area of distant objects.

## 4.3. Ablation Study

(a) HiLoTs Embedding Unit (HEU). HEU is the primary innovation of this work, focusing on both nearby regions with minimal class and shape changes and distant regions with more substantial variations. It mainly consists of the high temporal sensitivity flow (HTSF) and low temporal sensitivity flow (LTSF). Table 3(a) presents a performance comparison for segmentation by showing results with HEU entirely disabled, HTSF-only, LTSF-only, and with the complete HEU module. It shows that HTSF and LTSF significantly improve LiDAR segmentation performance. Furthermore, HTSF, which targets temporally sensitive regions with variations in both class and shape, contributes more prominently to performance enhancement than LTSF. The best results are achieved when both are combined in the HEU module.

(b) Cross-Attention and Fusion. As shown in Fig.3, when performing cross-attention between HTSF and LTSF, HTSF is used as the key (K) and value (V), while LTSF is used as the query (Q). We also test by using HTSF as Q and LTSF as Q, as well as simple addition or concatenation of HTSF and LTSF. The experimental results, reported in Table 3(b), show that performing cross-attention between high-sensitive flow and low-sensitive flow improves performance by approximately 2%. This indicates that the crossattention between high and low temporal sensitivity features allows the model to better capture both dynamic and stable object information, leading to more accurate performance.

Table 3. Ablation study on the core components of HiLoTs.
<table><tr><td rowspan="2">Structure</td><td>SemanticKITTI</td><td>nuScenes</td></tr><tr><td>10% 20% 50%</td><td>10% 20% 50%</td></tr><tr><td>None</td><td>59.2 60.3 60.9</td><td>66.8 68.4 69.2</td></tr><tr><td>HTSF</td><td>63.4 63.9 64.5</td><td>69.2 71.8 74.3</td></tr><tr><td>LTSF</td><td>62.8 63.5 64.3</td><td>68.5 71.2 73.9</td></tr><tr><td>HEU</td><td>65.7 66.5 67.6</td><td>72.2 75.2 76.9</td></tr></table>

(a) HEU components. “None” denotes no HEU is applied. “HTSF” and “LTSF” denote only applying high and low temporal sensitivity flow, respectively. “HEU” represents the original method.

<table><tr><td rowspan="2">Fusion</td><td>SemanticKITTI</td><td>nuScenes</td></tr><tr><td>10% 20% 50%</td><td>10% 20% 50%</td></tr><tr><td>Add</td><td>63.6 64.7 65.3</td><td>69.8 72.8 74.6</td></tr><tr><td>Concat</td><td>64.3 65.1 65.9</td><td>70.3 73.5 75.2</td></tr><tr><td>High as Q Low as Q 65.7 66.5</td><td>65.4 66.1 66.9</td><td>71.5 75.4 76.5 67.6 72.2 75.2 76.9</td></tr></table>

<table><tr><td rowspan="2">Sampling</td><td>SemanticKITTI</td><td>nuScenes</td></tr><tr><td>10% 20% 50%</td><td>10% 20% 50%</td></tr><tr><td>Random</td><td>63.5 64.2 64.9</td><td>69.3 72.4 73.5</td></tr><tr><td>Density</td><td>63.9 64.865.4 69.5 73.2 74.1</td><td></td></tr><tr><td></td><td>Aggregate 65.7 66.5 67.6 72.2 75.2 76.9</td><td></td></tr></table>

(c) Voxel down-sampling strategies in MVA. “Random” denotes random selection, and “Density” denotes voxels with the most point cloud density. “Aggregate” denotes our approach.

(b) Fusion in HEU. “Add” and “Concate” denote element-wise addition and concatenation, respectively. “High as Q” represents the self-attention map from HTSF serves as Query, and “Low as Q” denotes the original method.
<table><tr><td rowspan="2">Backbone</td><td>SemanticKITTI</td><td>nuScenes</td></tr><tr><td>10% 20% 50%</td><td>10% 20% 50%</td></tr><tr><td>Cubic</td><td>64.3 65.7 66.8</td><td>71.4 73.3 75.8</td></tr><tr><td>Pillar</td><td>62.4 64.1 65.2</td><td>69.4 71.5 73.2</td></tr><tr><td>Sphere</td><td>64.8 65.9 67.2</td><td>71.6 73.8 76.2</td></tr><tr><td>Cylin.</td><td>65.7 66.5 67.6</td><td>72.2 75.2 76.9</td></tr></table>

(d) Backbone networks. We test cubic, pillar, spherical, and cylindrical voxelization. All methods perform well.

<table><tr><td>EMA</td><td>SemanticKITTI 10% 20% 50% 10% 20% 50%</td><td>nuScenes</td></tr><tr><td>0.5</td><td>65.2 65.9 67.1</td><td>71.5 74.5 76.1</td></tr><tr><td>0.9</td><td>65.5 66.2 67.3 71.7 74.8 76.3</td><td></td></tr><tr><td>0.99 0.999</td><td>65.7 66.567.672.2 75.2 76.9 64.7 65.3 66.7 71.3 74.1 75.8</td><td></td></tr></table>

(e) EMA ratio. As the EMA ratio increases, segmentation performance shows a upward trend, reaching its peak at 0.99.

<table><tr><td># Comp.</td><td>mIoU Fog</td><td>Snow</td><td>Beam</td><td>Echo</td></tr><tr><td>C -</td><td>CENet 62.6 FRNet 68.7 Ours 67.8</td><td>42.7 47.6 56.2</td><td>53.6 55.8 57.1 62.5 58.0 58.5</td><td>53.4 58.1 57.9</td></tr><tr><td>C NS Ours</td><td>CENet FRNet</td><td>73.3 67.0 79.0 69.1 77.3 68.3</td><td>61.6 50.0 69.5 68.3 70.2 65.7</td><td>53.3 58.7 61.9</td></tr></table>

(f) Robustness. HiLoTs has comparable robustness with SoTA fully-supervised models in “Fog”, “Snow”, “Wet”, and “Echo” conditions.

(c) Multi-Voxel Aggregation. As described in Sec. 3.2.1, we apply multi-voxel aggregation among nearby voxels to reduce computation complexity in HiLoTs Embedding Unit. We also conduct two additional experiments to reduce the number of input voxels. The first method randomly selects m voxels, while the second method chooses the top-m voxels with the highest point cloud density. Since both methods inevitably discard information from the unselected voxels, our proposed method outperforms these two approaches by 2-3%, as reported in Table 3(c).

(d) Backbone Networks. In the voxelization step shown in Fig. 2, we can arrange LiDAR point clouds in cubic [45], pillar [22], and spherical [20] formats, and leverage corresponding backbone networks to generate feature maps. As shown in Table 3(d), all these voxelization methods perform well, indicating that HiLoTs can effectively capture LiDAR spatial characteristics at both near and far ranges if the arrangement of point clouds inherently encodes distancerelated properties. We choose the cylindrical voxelization in HiLoTs since it performs the best.

(e) EMA Ratio. The EMA update ratio is a relatively sensitive factor that impacts segmentation performance in previous semi-supervised mean-teacher architecture [18, 26]. In contrast, as shown in Table 3(e), changes in the update ratio have less impact on the performance outcomes, indicating HiLoTs exhibits a high level of stability and robustness with respect to EMA ratio.

(f) Performance on Out-of-distribution Datasets. In addition to conventional segmentation evaluation, we further investigate HiLoTs’s robustness under various data perturbations using SemanticKITTI-C and nuScenes-C datasets [17, 43], and compare it with CENet [7] and FR-Net [40]. As shown in Table 3(f), HiLoTs exhibits comparable robustness under three types of perturbations: severe weather conditions (Fog, Snow), external disturbances (Beam-missing), and internal sensor failures (Echo). These results demonstrate the effectiveness of high-low tempora sensitive representation learning in various conditions.

## 5. Conclusion and Limitation

In this paper, we propose a novel semi-supervised Li-DAR point cloud segmentation method, HiLoTs, which effectively leverages temporal dynamics through the High Temporal Sensitivity Flow and Low Temporal Sensitivity Flow. By focusing on different regions with varying temporal characteristics, HiLoTs significantly improves Li-DAR semantic segmentation performance, especially for distant objects with rapidly changing shapes and categories. Our experimental results, evaluated on the widely used SemanticKITTI and nuScenes datasets, demonstrate that HiLoTs outperforms both fully-supervised and state-of-theart semi-supervised methods under limited annotations.

Since HiLoTs is specifically designed for the autonomous driving domain, where it leverages point clouds from consecutive frames, it is not suitable for general object point cloud segmentation task, such as posed in Point-Net [31] and KPConv [36].

Acknowledgements: This work was supported by the National Natural Science Foundation of China under grant 62102307, U21A20462, 62372400, and 62372365, and Fundamental Research Funds for the Central Universities.

## References

[1] Angelika Ando, Spyros Gidaris, Andrei Bursuc, Gilles Puy, Alexandre Boulch, and Renaud Marlet. Rangevit: Towards vision transformers for 3d semantic segmentation in autonomous driving. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5240–5250, 2023. 1, 2, 4, 6

[2] Mehmet Aygun, Aljosa Osep, Mark Weber, Maxim Maxi-¨ mov, Cyrill Stachniss, Jens Behley, and Laura Leal-Taixe.´ 4d panoptic lidar segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5527–5537, 2021. 2, 3

[3] Jens Behley, Martin Garbade, Andres Milioto, Jan Quenzel, Sven Behnke, Cyrill Stachniss, and Jurgen Gall. Semantickitti: A dataset for semantic scene understanding of lidar sequences. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9297–9307, 2019. 5

[4] Holger Caesar, Varun Bankiti, Alex H. Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom. nuscenes: A multimodal dataset for autonomous driving. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020. 5

[5] Xinyuan Chang, Maixuan Xue, Xinran Liu, Zheng Pan, and Xing Wei. Driving by the rules: A benchmark for integrating traffic sign regulations into vectorized hd map. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025. 1

[6] Yujun Chen, Xin Tan, Zhizhong Zhang, Yanyun Qu, and Yuan Xie. Beyond the label itself: Latent labels enhance semi-supervised point cloud panoptic segmentation. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 1245–1253, 2024. 2, 6

[7] Hui-Xian Cheng, Xian-Feng Han, and Guo-Qiang Xiao. Cenet: Toward concise and efficient lidar semantic segmentation for autonomous driving. In 2022 IEEE international conference on multimedia and expo (ICME), pages 01–06. IEEE, 2022. 8

[8] Mingmei Cheng, Le Hui, Jin Xie, and Jian Yang. Sspc-net: Semi-supervised semantic 3d point cloud segmentation network. In Proceedings of the AAAI conference on artificial intelligence, pages 1140–1147, 2021. 1, 2

[9] Christopher Choy, JunYoung Gwak, and Silvio Savarese. 4d spatio-temporal convnets: Minkowski convolutional neural networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 3075–3084, 2019. 2, 3, 5

[10] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is

worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representa tions, 2021. 4

[11] Juncong Fei, Kunyu Peng, Philipp Heidenreich, Frank Bieder, and Christoph Stiller. Pillarsegnet: Pillar-based semantic grid map estimation using sparse lidar data. In 2021 IEEE intelligent vehicles symposium (IV), pages 838–844. IEEE, 2021. 1

[12] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceed ings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016. 3

[13] Siyuan Huang, Yichen Xie, Song-Chun Zhu, and Yixin Zhu. Spatio-temporal self-supervised representation learning for 3d point clouds. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 6535–6545, 2021. 1

[14] Alok Jhaldiyal and Navendu Chaudhary. Semantic segmentation of 3d lidar data using deep learning: a review of projection-based methods. Applied Intelligence, 53(6): 6844–6855, 2023. 1

[15] Li Jiang, Shaoshuai Shi, Zhuotao Tian, Xin Lai, Shu Liu, Chi-Wing Fu, and Jiaya Jia. Guided point contrastive learn ing for semi-supervised point cloud semantic segmentation. In Proceedings ofthe IEEE/CVF international conference on computer vision, pages 6423–6432, 2021. 2, 6

[16] Lingdong Kong, Youquan Liu, Runnan Chen, Yuexin Ma, Xinge Zhu, Yikang Li, Yuenan Hou, Yu Qiao, and Ziwei Liu. Rethinking range view representation for lidar segmentation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 228–240, 2023. 2

[17] Lingdong Kong, Youquan Liu, Xin Li, Runnan Chen, Wen wei Zhang, Jiawei Ren, Liang Pan, Kai Chen, and Ziwei Liu. Robo3d: Towards robust and reliable 3d perception against corruptions. arXiv preprint arXiv:2303.17597, 2023. 8

[18] Lingdong Kong, Jiawei Ren, Liang Pan, and Ziwei Liu. Lasermix for semi-supervised lidar semantic segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21705–21715, 2023. 2, 5, 6, 7, 8

[19] Lingdong Kong, Xiang Xu, Jiawei Ren, Wenwei Zhang, Liang Pan, Kai Chen, Wei Tsang Ooi, and Ziwei Liu. Multi modal data-efficient 3d scene understanding for autonomous driving. arXiv preprint arXiv:2405.05258, 2024. 2, 6

[20] Xin Lai, Yukang Chen, Fanbin Lu, Jianhui Liu, and Jiaya Jia. Spherical transformer for lidar-based 3d recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 17545–17555, 2023. 2, 4, 6, 8

[21] Bo Lan, Fei Wang, Lekun Xia, Fan Nai, Shiqiang Nie, Han Ding, and Jinsong Han. Bullydetect: Detecting school physical bullying with wi-fi and deep wavelet transformer. IEEE Internet ofThings Journal, 2024. 4

[22] Alex H Lang, Sourabh Vora, Holger Caesar, Lubing Zhou, Jiong Yang, and Oscar Beijbom. Pointpillars: Fast encoders for object detection from point clouds. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12697–12705, 2019. 2, 8

[23] Jianan Li and Qiulei Dong. Density-guided semi-supervised 3d semantic segmentation with dual-space hardness sampling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3260–3269, 2024. 1, 2, 5, 6

[24] Jiale Li, Hang Dai, Hao Han, and Yong Ding. Mseg3d: Multi-modal 3d semantic segmentation for autonomous driving. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 21694–21704, 2023. 1, 2

[25] Jinyu Li, Chenxu Luo, and Xiaodong Yang. Pillarnext: Rethinking network designs for 3d object detection in lidar point clouds. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 17567– 17576, 2023. 2

[26] Li Li, Hubert PH Shum, and Toby P Breckon. Less is more: Reducing task and model complexity for 3d point cloud semantic segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9361–9371, 2023. 2, 5, 6, 8

[27] Mengtian Li, Yuan Xie, Yunhang Shen, Bo Ke, Ruizhi Qiao, Bo Ren, Shaohui Lin, and Lizhuang Ma. Hybridcr: Weakly-supervised 3d point cloud semantic segmentation via hybrid contrastive regularization. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 14930–14939, 2022. 1

[28] Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollar. Focal loss for dense object detection. In´ Proceedings of the IEEE international conference on computer vision, pages 2980–2988, 2017. 5

[29] Jiahui Liu, Chirui Chang, Jianhui Liu, Xiaoyang Wu, Lan Ma, and Xiaojuan Qi. Mars3d: A plug-and-play motionaware model for semantic segmentation on multi-scan 3d point clouds. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9372– 9381, 2023. 2, 6

[30] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019. 5

[31] Charles R Qi, Hao Su, Kaichun Mo, and Leonidas J Guibas. Pointnet: Deep learning on point sets for 3d classification and segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 652–660, 2017. 2, 3, 8

[32] Laurenz Reichardt, Nikolas Ebert, and Oliver Wasenmuller.¨ 360deg from a single camera: a few-shot approach for lidar segmentation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1075–1083, 2023. 6

[33] Hanyu Shi, Jiacheng Wei, Ruibo Li, Fayao Liu, and Guosheng Lin. Weakly supervised segmentation on outdoor 4d point clouds with temporal matching and spatial graph propagation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11840– 11849, 2022. 2, 3

[34] Jingang Shi, Yusi Wang, Zitong Yu, Guanxin Li, Xiaopeng Hong, Fei Wang, and Yihong Gong. Exploiting multi-scale

parallel self-attention and local variation via dual-branch transformer-cnn structure for face super-resolution. IEEE Transactions on Multimedia, 26:2608–2620, 2023. 4

[35] Antti Tarvainen and Harri Valpola. Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results. Advances in neural information processing systems, 30, 2017. 2, 5

[36] Hugues Thomas, Charles R Qi, Jean-Emmanuel Deschaud, Beatriz Marcotegui, Franc¸ois Goulette, and Leonidas J Guibas. Kpconv: Flexible and deformable convolution for point clouds. In Proceedings of the IEEE/CVF international conference on computer vision, pages 6411–6420, 2019. 2, 8

[37] Ozan Unal, Dengxin Dai, Lukas Hoyer, Yigit Baran Can, and Luc Van Gool. 2d feature distillation for weakly-and semisupervised 3d semantic segmentation. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 7336–7345, 2024. 2, 6

[38] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017. 4

[39] Aoran Xiao, Jiaxing Huang, Dayan Guan, Kaiwen Cui, Shi jian Lu, and Ling Shao. Polarmix: A general data augmentation technique for lidar point clouds. Advances in Neural Information Processing Systems, 35:11035–11048, 2022. 6

[40] Xiang Xu, Lingdong Kong, Hui Shuai, and Qingshan Liu. Frnet: Frustum-range networks for scalable lidar segmentation. IEEE Transactions on Image Processing, 2025. 6, 8

[41] Zongyi Xu, Bo Yuan, Shanshan Zhao, Qianni Zhang, and Xinbo Gao. Hierarchical point-based active learning for semi-supervised point cloud semantic segmentation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 18098–18108, 2023. 2

[42] Kangwei Yan, Fei Wang, Bo Qian, Han Ding, Jinsong Han, and Xing Wei. Person-in-wifi 3d: End-to-end multiperson 3d pose estimation with wi-fi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 969–978, 2024. 4

[43] Xu Yan, Chaoda Zheng, Ying Xue, Zhen Li, Shuguang Cui, and Dengxin Dai. Benchmarking the robustness of lidar semantic segmentation models. International Journal of Com puter Vision, 132(7):2674–2697, 2024. 8

[44] Dongqiangzi Ye, Zixiang Zhou, Weijia Chen, Yufei Xie, Yu Wang, Panqu Wang, and Hassan Foroosh. Lidarmultinet: Towards a unified multi-task network for lidar perception. In Proceedings of the AAAI Conference on Artificial Intelli gence, pages 3231–3240, 2023. 2

[45] Yin Zhou and Oncel Tuzel. Voxelnet: End-to-end learning for point cloud based 3d object detection. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4490–4499, 2018. 1, 8

[46] Xinge Zhu, Hui Zhou, Tai Wang, Fangzhou Hong, Yuexin Ma, Wei Li, Hongsheng Li, and Dahua Lin. Cylindrical and asymmetrical 3d convolution networks for lidar segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9939–9948, 2021. 1, 2, 3, 5, 6