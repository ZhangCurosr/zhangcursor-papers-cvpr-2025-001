# Touch2Shape: Touch-Conditioned 3D Diffusion for Shape Exploration and Reconstruction

Yuanbo Wang<sup>1</sup> Zhaoxuan Zhang<sup>2</sup> Jiajin Qiu<sup>1</sup> Dilong Sun<sup>1</sup> Zhengyu Meng<sup>1</sup> Xiaopeng Wei<sup>1</sup>\* Xin Yang<sup>1</sup>\* <sup>1</sup> Key Laboratory of Social Computing and Cognitive Intelligence, Dalian University of Technology <sup>2</sup> Nanjing University of Posts and Telecommunications wangyuanbo@mail.dlut.edu.cn zhangzx@njupt.edu.cn {<sup>jiajinqiu,</sup> <sup>sundilong,</sup> <sup>mzy1</sup>}<sup>@mail.dlut.edu.cn</sup> xpwei, xinyang @dlut.edu.cn

![](images/475cfc038c942bcd3e9bd42c3852853e1bf544760249c02736942918724ff249.jpg)  
Figure 1. (1) Exploring the target object and capturing the tactile image to reconstruct the 3D shape. We trained a diffusion model to obtain a low-dimensional and compact latent vector, which is used for predicting next touch location and reconstructing the target shapes. The numbers 1 with black arrows and 2 with greeen arrows in the diagram represent two consecutive time steps. We only generate the full reconstruction at the final step. (2) As the touch exploration progresses, the reconstruction results graduall approach the ground truth. (3) Reconstruction results compared with ActiveVT [34] (under visual-tactile settings).

## Abstract

Diffusion models have made breakthroughs in 3D generation tasks. Current 3D diffusion models focus on reconstructing target shape from images or a set of partial obser vations. While excelling in global context understanding, they struggle to capture the local details of complex shapes and limited to the occlusion and lighting conditions. To overcome these limitations, we utilize tactile images to capture the local 3D information and propose a Touch2Shape model, which leverages a touch-conditioned diffusion model to explore and reconstruct the target shape from touch. For shape reconstruction, we have developed a touch embed

ding module to condition the diffusion model in creating a compact representation and a touch shape fusion module to refine the reconstructed shape. For shape exploration, we combine the diffusion model with reinforcement learning to train a policy. This involves using the generated latent vectorfrom the diffusion model to guide the touch exploration policy training through a novel reward design. Experiments validate the reconstruction quality thorough both qualitatively and quantitative analysis, and our touch exploration policyfurther boosts reconstruction performance.

## 1. Introduction

3D generation and reconstruction tasks have emerged as key focal points in the fields of computer vision and graphics, offering valuable applications in areas like autonomous driving and robot interactions characterized by environmental occlusions and camera measurement errors [5, 6, 32]. However, acquiring 3D data presents greater challenges and costs compared to 2D image and text data. This underscores the critical importance of ongoing research into 3D reconstruction and generation.

Diffusion models have garnered substantial attention in generative tasks for their innovative approaches and promising results. While their impact has been particularly notable in 2D image generation [10, 27], these models are also being applied to 3D tasks. Pioneering projects such as SDFusion [5] and DiffusionSDF [6] have demonstrated the ability to create the 3D shapes using visual images or multimodal data. However, the current focus is on 3D shape reconstruction based on predetermined partial observations. This limitation presents challenges on two fronts. Firstly, while current methods can estimate the overall shape of a target from limited data, they may overlook the local details of objects, making it challenging to generate complex shapes. This motivates further research into 3D reconstruction across new modalities. Secondly, practical scenarios present challenges where environmental factors like occlusions and varying lighting conditions impede the complete extraction of crucial local information. The complexity of real-world conditions underscores the necessity of actively exploring the target to capture information and facilitate the reconstruction process.

In this work, we employ a simulated robotic arm guided by a trained policy model to touch the target, enabling the acquisition of tactile images to reconstruct the target through touch interaction. Tactile images provide local 3D shape information, including spatial contact points and detailed shape information, enhancing the reconstruction of local details, while visual data assists in predicting overall shape characteristics. Especially, we propose Touch2Shape, a touch-conditioned diffusion model for shape exploration through continuous touch and shape reconstruction by integrating all captured tactile images (along with the optional visual image). The reason for utilizing the diffusion model is to leverage its powerful generative capabilities to assist the model in generating a latent shape when the initial data perception is limited. The latent shape is then used to guide the policy model to predict beneficial touch locations for reconstruction based on potential missing areas. To train the exploration policy, we integrate the diffusion model with reinforcement learning and design a reward function.

In contrast to previous methods, our approach not only produces superior 3D reconstruction outputs but also eliminates the need to generate the final shape at every step.

Due to the fact that the diffusion model can generate a low-dimensional and compact latent space, we only generate the full reconstruction of the target object at the final step. Extensive experiments validate the effectiveness of our method, demonstrating significant improvements in both reconstruction performance and the ability to improve reconstruction quality through touch exploration.

The main contributions of this article are as follows:

• We propose Touch2Shape, a touch-conditioned 3D diffusion model for shape exploration and reconstruction, utilizing the latent vector to guide the touch location planning and shape decoding.

• Touch2Shape utilizes contrastive touch encoder to embed touch information and shapes in a joint space.

• We propose a touch shape fusion module to optimize the reconstructed shape using touch information.

• We combine the diffusion model with reinforcement learning for shape exploration policy training and design the corresponding reward function.

## 2. Related Work

Touch-based 3D Object Reconstruction. There are a lot of works addressing 3D shape reconstruction from visual signals. The approaches in this field vary depending on the type of visual input utilized, such as single view RGB images [14], multi-view RGB images [8], and depth images [2]. Additionally, the types of 3D representations they predict also differ, encompassing aspects such as voxels [24], point clouds [13], meshes [41], and signed distance functions [26]. However, due to the high-dimensionality of the observation space and the interference of occlusions and external lighting conditions, training computer vision algorithms for object manipulation faces significant challenges. Recently, some researchers use high-resolution tactile perception such as DIGIT [22] and Gelsight sensor [45] to accomplish 3D object reconstruction. This type of perception, compared to vision-based perception, can obtain rich contact information and remains effective in the face of occlusions, transparent or reflective object materials. TouchSDF [7] predicts the local 3D shape corresponding to the tactile image and completes the 3D expression of the object by training an implicit neural function to represent the signed distance. Smith et al. [33] propose the first approach for reconstruction using both vision and touch, highlighting their complementary nature. Combining vision and touch modalities can yield richer information and improve 3D shape reconstruction. In this work, our Touch2Shape model trains a touch-conditioned diffusion model for 3D shape reconstruction, supporting settings with both tactile only and visualtactile inputs.

3D Diffusion Model. The diffusion probability model [18, 35] operates by gradually removing noise from data points to generate samples from a distribution. The diffusion model in the field of images has garnered widespread attention, for tasks such as image generation [10, 27], superresolution [30], image-editing [4].The application of the diffusion models extends well into 3D generation tasks. Some works train diffusion models to generate point clouds [37, 50], voxel grids [23, 32] and meshes [47]. Recently, SDFusion [5] encodes 3D shapes into an expressive lowdimensional latent space, which is used to train the 3D diffusion model and present the shape as a TSDF (truncated signed distance function) volume. DiffusionSDF [6] combine a 3D diffusion model with implicit expressions to represent the target 3D surface using a neural symbolic distance function and generate diverse reconstruction results using a diffusion model. These methods excel in both unconditional generation and conditional generation from partial inputs, inspiring our study of touch-based shape reconstruction tasks using the diffusion model.

![](images/b519e59a8ac62812018d64ebe48c297d0691384c1977ce8c1fc6bcebe78ce783.jpg)  
Figure 2. We pretrained (a) the shape encoder and decoder, (b) the touch CNN model that is used for touch chart prediction, and (c) the contrastive touch encoder. During the training of (d) the touch-conditioned 3D diffusion model, we kept the parameters of the pretrained modules fixed and train a touch-conditioned 3D diffusion model for touch exploration policy learning and shape reconstruction.

Shape Exploration. In this paper, the purpose of shape exploration is to actively capture continuous tactile information and enhance shape reconstruction accuracy based on these acquired tactile data. One approach to achieve this goal is to maximize the information from the captured tactile data such as Monte Carlo sampling [9] and Gaussian Process Regression [19]. In recent years, many strategies have incorporated deep learning to perform active sensing mechanisms in shape exploration. Most methods [1, 15, 29, 43, 46, 48, 49] have been developed for the active vision task by planning camera perspectives to capture multi-view images essential for 3D reconstruction. For active touch sensing, several prior works [11, 20, 25, 31, 40, 44] address the problem of shape reconstruction by estimating uncertainty for the selection of the next touch. In this paper, we combine the diffusion model with reinforcement learning to learn the exploration policy. The most relevant work is [43] and [34]. The former proposes a unified model for view planning and vision-based object reconstruction, which combines the 3D reconstruction learning and reinforcement learning. The latter proposes a mesh-based 3D shape reconstruction model where touch exploration strategies are learned over shape predictions using reinforcement learning. In contrast to existing methodologies, our study focuses on training a touch-based diffusion model to generate a latent space, obviating the need to construct a complete shape at every step. Additionally, we leverage the captured touch information to enhance the final shape output. These approaches enable our method to achieve increased reconstruction quality improvements with continuous touches.

## 3. Method

As shown in Figure 1, we train a touch-conditioned diffusion model to implicitly represent the target object information for shape exploration and reconstruction. In test stage, we gather tactile images $( T _ { 0 } , . . . , T _ { n - 1 } )$ from the target, utilizing the trained diffusion model to obtain a lowdimensional representation for predicting the subsequent touch location. The shape is also reconstructed based on all the captured tactile information.

Figure 2 illustrates the architecture of our Touch2Shape Model. Following SDFusion [5], we employ the volumetric Truncated Signed Distance Field (T-SDF) to model the distribution across 3D shapes and a 3D variant of the Vector Quantised-Variational AutoEncoder (VQ-VAE) [38] for encoding and decoding 3D shapes. Let the input shape is represented as the T-SDF volume $X \in R ^ { D \times D ^ { \bullet } \times D }$ , the encoder is $E _ { s }$ and the decoder is $D _ { s }$ , we encode the input shape into the latent vector $z$ and decode the vector to the generated shape $X ^ { \prime } { : }$

$$
z = E _ { s } ( X ) , \mathrm { a n d } X ^ { \prime } = D _ { s } ( V Q ( z ) ) ,\tag{1}
$$

where $V Q$ is the quantization step.

We pretrained the VQ-VAE model on both ABC dataset [21] and ShapeNet dataset [3]. Using the encoded latent vector, gaussian noise is added at random timestaps $t ,$ and a denoising model is trained based on the touch condition (Section 3.1). The touch charts prediction model is pretrained as [33, 34]. Contrastive learning is employed to embed touch information and shapes in a joint space. The policy model receives the denoised vector as input and is trained using reinforcement learning (Section 3.2). The generation of the final shape is optimized through tactile data (Section 3.3).

## 3.1. Touch-conditioned Diffusion Model

Given a sample z encoded by the pretrained shape encoder $E _ { s } ,$ , we learn a touch-conditioned distribution for touch based shape exploration and reconstruction. The loss function for diffusion model training is as follows:

$$
L _ { d i f f } ( t , n ) = | | E _ { \theta } ( z _ { t } , r ( t ) , C ( T _ { 0 } , . . . , T _ { n - 1 } ) ) - \epsilon _ { t } | | _ { 2 } ,\tag{2}
$$

where $\epsilon _ { t }$ is the added gaussian noise, the $E _ { \theta }$ is the denoising network, $C$ is the touch condition extraction network for n touch inputs $( T _ { 0 } , . . . , T _ { n - 1 } )$ , t is the number of diffusion timesteps and $r ( t )$ is time embedding respectively.

Touch Embedding. We pretrain a TouchCNN model as [33, 34] to predict the touch charts. We assume that we can obtain up to N tactile images, where each tactile image generates a chart as a tensor of size $M \times 4$ (each vertex contains coordinates $x , y , z ,$ , and touch status, M is the number of chart vertices). By merging all the vertices of these charts together, we create a tensor of size $N \times M \times 4$ . If we have fewer than N tactile images, the coordinates of the vertices for the remaining charts are zeroed out. Each touch chart is considered as a token. We first apply position encoding to the centroid of each chart, then perform convolution operations on the merged tensor to extract vertex features. After pooling operations and adding position embedding, we finally obtain N tokens.

Contrastive Touch Encoder. It has been confirmed in IC3D [32] that the joint encoding of both 2D and 3D information is beneficial to condition the generation of 3D shapes based on images. UniTouch [42] connects the tactile signals to other modalities, including vision, language, and sound. In this study, we propose a contrastive model for embedding touch and latent vectors in a joint space. Utilizing the latent vectors generated by the pretrained encoder

$E _ { s }$ , we establish a latent encoder to acquire the shape features. For touch features, we construct a network similar to the touch embedding model mentioned above, except for the addition of a pooling layer at the end. We utilize moco [17] for training the contrastive learning task. We set touch features as queries and shape features as keys. The objective is to pulling together with matching touch-shape pairs while pushing unmatched pairs apart. The loss function is:

$$
L _ { r l } = - l o g \frac { e ^ { q \cdot k _ { p } / \tau } } { \sum _ { i = 0 } ^ { K } e ^ { q \cdot k _ { i } / \tau } } ,\tag{3}
$$

where $q$ is the query feature, k is the key feature, $k _ { i }$ is the i-th element in the queue of size $\mathrm { K } ,$ and ε is the temperature parameter respectively.

Visual-tactile Setting. Touch2Shape model supports two modes: tactile only and visual-tactile. In the visualtactile setting, due to the ability of visual information to capture the global context, reasonable shape outputs can be obtained even when initial tactile images are limited. We initially use visual information to guide tactile exploration, predict the global structure, and refine the shape as more tactile information becomes available. The implementation involves extracting feature tokens from images using ResNet [16], combining them with touch tokens through a dropout layer, and then inputting them together into the denoising network of the Diffusion model. This process generates denoised vectors $z ^ { \prime } ,$ , which is then used for shape reconstruction and next touch location prediction.

## 3.2. Touch Shape Fusion

The touch shape fusion module is designed with two goals. Firstly, the diffusion model typically generates shapes that are globally consistent with the input conditions, but may exhibit discrepancies in local details against the input touch chart. Secondly, the encoded latent vector, being low-dimensional and compressed, risks omitting crucial high-dimensional details. To address this, we voxelize the global shape merged from all historical touch information (as Figure 2) and utilize an additional voxel encoder to capture multi-scale features. These features are fused with the different scale features generated during the decoding of the latent vector, resulting in a better shape reconstruction. The decoder originates from the pre-trained VQVAE model and is fine-tuned in touch shape fusion module training. Figure 3 depicts the network module. We represent the feature at location $( c , k , j , i )$ with a size of $C \times D \times H \times W$ is $M ( c , k , j , i )$ . Taking the first fusion block as example, the fusion feature $M _ { 1 } ( c , k , j , i )$ can be computed as follows:

$$
M _ { 1 } ( c , k , j , i ) = \frac { \lambda \cdot F _ { 3 } ^ { e } ( c , k , j , i ) \cdot e ^ { F _ { 1 } ^ { d } ( c , k , j , i ) } } { \sum _ { c ^ { \prime } = 0 } ^ { c ^ { \prime } = C - 1 } ( e ^ { F _ { 1 } ^ { d } ( c ^ { \prime } , k , j , i ) } ) } ,\tag{4}
$$

where $F _ { 1 } ^ { d }$ and $F _ { 3 } ^ { e }$ is the 3D feature maps as depicted in Figure $3 , \lambda$ is a learnable weight.

![](images/bb6b775a62f8ac36e8ecbe19bd08aa3cc1a2bb9fc59381942bf4f2b42c0b2b00.jpg)  
Figure 3. Touch shape fusion module. The black arrows indicate the flow of the shape decoder, while the red arrows represent the flow after incorporating the touch shape fusion module. To simplify, we use 2D grids to visualize the 3D feature maps.

## 3.3. Policy Training

This module aims to create potential 3D shapes utilizing the diffusion model based on the acquired tactile (and optional visual) data, which are then employed to determine the next touch location for shape exploration. Previous methods [34] required generating the final shape at each time step and encoding the shape into a latent vector, or directly predicting touch location parameters from the generated shape using a policy network. The reward setting was based on the change in Chamfer Distance (CD) [36] after two consecutive touches. In this paper, we combine the diffusion model with policy training. At each time step, we input the latent vector z of the target object, add noise through the diffusion model, and then use a touchconditioned denoising network to obtain a denoised latent vector $z ^ { \prime } .$ This vector is a low-dimensional 3D feature map. We first employ the pre-trained latent encoder in Figure 2 (c) to encode both the initial and current latent vectors of the touch-conditioned diffusion model. Subsequently, we construct an action embedding module to derive the embedding for the potential actions. Each action is identified by its positional index on a sphere consisting of 50 actions, as described in [34]. By combining these three vectors, we employ fully connected layers to predict the value associated with each action.

The complete process obviates the necessity of predicting the final high-dimensional T-SDF Volume. Instead, we utilize the shape decoder and shape fusion only at the final time step, thereby achieving a separation of shape decoder and shape exploration. For reward function setting, since the final output shape is not predicted, we design it to be the difference in the diffusion model’s loss values. The reward at step n is computed as follows:

$$
R = H ( L _ { d i f f } ( t , n - 1 ) - L _ { d i f f } ( t , n ) ) ,\tag{5}
$$

where $H ( \cdot )$ assigns a value of 1 to all values greater than or

equal to 0, and a value of 0 to those less than 0.

It can be observed that our objective is to encourage actions that help the touch-conditioned diffusion model generate a denoised vector $\mathbf { z } '$ closer to the latent vector z. It can be trained by reinforcement learning such as DQN [39]. The loss function for reinforcement learning is as follows:

$$
\begin{array} { r } { L _ { r l } = [ R + \gamma \underset { a _ { n + 1 } } { \operatorname* { m a x } } Q ( T _ { 0 } , . . . , T _ { n + 1 } , a _ { n + 1 } ) } \\ { - Q ( T _ { 0 } , . . . , T _ { n } , a _ { n } ) ] ^ { 2 } , } \end{array}\tag{6}
$$

where $Q ( \cdot )$ is the Q-value function approximated by the network, $a _ { n }$ is the action taken on the timestep n, and $\gamma$ is the discount factor.

<table><tr><td rowspan="2">Method</td><td rowspan="2">Mode</td><td colspan="2">Grasp #</td></tr><tr><td>0</td><td>1</td></tr><tr><td>VTRecon [33] ActiveVT [34]</td><td>T</td><td>25.586 24.864</td><td>9.016 8.220</td></tr><tr><td>Ours</td><td></td><td>40.283</td><td>6.794</td></tr><tr><td rowspan="2">VTRecon [33] ActiveVT [34] Ours</td><td rowspan="2">T+V</td><td>2.653</td><td>2.637</td></tr><tr><td>2.538 1.475</td><td>2.486 1.406</td></tr></table>

Table 1. Experimental results for different settings and different numbers of grasps on dataset ABC. The evaluation metric is CD (lower is better).

## 4. Experiment

In this section, we describe the experiment settings and then compare our model with the state-of-art touch-based 3D reconstruction methods and validate our policy training strategy. Through the ablation study, we validate the necessity of each module. Additional experimental details can be found in the supplementary material.

## 4.1. Experimental Settings

Datasets. We validate our model using two datasets. The first dataset utilized is derived from [33, 34], built upon the ABC dataset [21]. This dataset comprises 40,000 objects with ambiguous class definitions and diverse shapes, presenting a significant generalization hurdle. The second dataset employed originates from [7], encompassing 1650 ShapeNet [3] objects that span six categories: bowls, bottles, cameras, jars, guitars, and mugs. All the touch images and initial RGB images are rendered using the simulation environment introduced in [34].

Training Setup. We pre-trained the VQ-VAE model following SDFuison [5] and the touch chart prediction model following [33, 34]. The volume resolution is set at 64x64x64. The training process is segmented into three parts: diffusion model training, touch shape fusion module training, and policy training. The diffusion model and touch shape fusion can be trained concurrently since they do not share any components. The diffusion model was trained for 1 million iterations with an initial learning rate of 0.00001 and batch size of 12, while the touch shape fusion was trained for 250,000 iterations with an initial learning rate of 0.0001 and batch size of 8. After the diffusion model training finished, we conducted policy training in silmulation environment [34] for 200 epochs with a learning rate of 0.0003 and batch size of 16. All the network codes are implemented using the PyTorch framework [28] and trained on the GeForce RTX 4090 for approximately a week.

<table><tr><td rowspan=3 colspan=1>Category</td><td rowspan=1 colspan=9># Touches (Unseen objects and poses)</td></tr><tr><td rowspan=1 colspan=3>1</td><td rowspan=1 colspan=3>10</td><td rowspan=1 colspan=3>20</td></tr><tr><td rowspan=1 colspan=1>TouchSDF</td><td rowspan=1 colspan=1>OursT</td><td rowspan=1 colspan=1>OursTV</td><td rowspan=1 colspan=1>TouchSDF</td><td rowspan=1 colspan=1>OursT</td><td rowspan=1 colspan=1>OursTV</td><td rowspan=1 colspan=1>TouchSDF</td><td rowspan=1 colspan=1>OursT</td><td rowspan=1 colspan=1>OursTV</td></tr><tr><td rowspan=1 colspan=1>Bottle</td><td rowspan=1 colspan=1>0.113</td><td rowspan=1 colspan=1>0.110</td><td rowspan=1 colspan=1>0.036</td><td rowspan=1 colspan=1>0.082</td><td rowspan=1 colspan=1>0.042</td><td rowspan=1 colspan=1>0.034</td><td rowspan=1 colspan=1>0.047</td><td rowspan=1 colspan=1>0.041</td><td rowspan=1 colspan=1>0.033</td></tr><tr><td rowspan=1 colspan=1>Mug</td><td rowspan=1 colspan=1>0.091</td><td rowspan=1 colspan=1>0.118</td><td rowspan=1 colspan=1>0.042</td><td rowspan=1 colspan=1>0.072</td><td rowspan=1 colspan=1>0.050</td><td rowspan=1 colspan=1>0.042</td><td rowspan=1 colspan=1>0.066</td><td rowspan=1 colspan=1>0.049</td><td rowspan=1 colspan=1>0.042</td></tr><tr><td rowspan=1 colspan=1>Bowl</td><td rowspan=1 colspan=1>0.085</td><td rowspan=1 colspan=1>0.128</td><td rowspan=1 colspan=1>0.046</td><td rowspan=1 colspan=1>0.073</td><td rowspan=1 colspan=1>0.051</td><td rowspan=1 colspan=1>0.045</td><td rowspan=1 colspan=1>0.048</td><td rowspan=1 colspan=1>0.049</td><td rowspan=1 colspan=1>0.040</td></tr><tr><td rowspan=3 colspan=1>CameraGuitarJar</td><td rowspan=1 colspan=1>0.131</td><td rowspan=1 colspan=1>0.132</td><td rowspan=1 colspan=1>0.055</td><td rowspan=1 colspan=1>0.101</td><td rowspan=1 colspan=1>0.058</td><td rowspan=1 colspan=1>0.052</td><td rowspan=1 colspan=1>0.092</td><td rowspan=1 colspan=1>0.056</td><td rowspan=1 colspan=1>0.050</td></tr><tr><td rowspan=1 colspan=1>0.195</td><td rowspan=1 colspan=1>0.139</td><td rowspan=1 colspan=1>0.059</td><td rowspan=1 colspan=1>0.177</td><td rowspan=1 colspan=1>0.067</td><td rowspan=1 colspan=1>0.056</td><td rowspan=1 colspan=1>0.155</td><td rowspan=1 colspan=1>0.064</td><td rowspan=2 colspan=1>0.0450.045</td></tr><tr><td rowspan=1 colspan=1>0.164</td><td rowspan=1 colspan=1>0.122</td><td rowspan=1 colspan=1>0.048</td><td rowspan=1 colspan=1>0.136</td><td rowspan=1 colspan=1>0.059</td><td rowspan=1 colspan=1>0.046</td><td rowspan=1 colspan=1>0.071</td><td rowspan=1 colspan=1>0.055</td></tr><tr><td rowspan=1 colspan=1>Average</td><td rowspan=1 colspan=1>0.136</td><td rowspan=1 colspan=1>0.124</td><td rowspan=1 colspan=1>0.048</td><td rowspan=1 colspan=1>0.112</td><td rowspan=1 colspan=1>0.056</td><td rowspan=1 colspan=1>0.046</td><td rowspan=1 colspan=1>0.081</td><td rowspan=1 colspan=1>0.053</td><td rowspan=1 colspan=1>0.042</td></tr></table>

Table 2. Experimental results for different numbers of touches on dataset ShapeNet. Ours<sub>T</sub> and Ours<sub>TV</sub> respectively represent our methods under the tactile only and visual-tactile settings. The evaluation metric is EMD (lower is better).

Evaluation Metrics. Following previous learning-based reconstruction tasks, we use Chamfer Distance (CD) [36] and Earth Mover’s Distance (EMD) [12] for reconstruction evaluation. The former is a common 3D reconstruction metric for measuring the point-wise distance between two pointsets. The latter is a metric used to measure the dissimilarity that calculates the minimum amount of work needed to transform one point set into another, which is particularly useful in comparing the visual quality of 3D shapes.

## 4.2. Evaluation on Reconstruction Performance

Results on the ABC dataset. we evaluate our reconstruction model and compare it with [33] and [34]. The former method (we called VTRecon here) predicts a local chart at each touch site and combines them with vision signals to predict global charts via graph convolutional networks. The latter method (we called ActiveVT here) proposes an active touch sensing for 3D reconstruction method to improve the reconstruction performance. We examine the performance across 2 settings: (1) tactile only (T), only touch signals from all hand sensors are used during shape exploration, the hand has 4 fingers thus we can obtain at most 4 valid tactile images during one touch exploration. (2) touch and vision (T+V), an extension of tactile only setting with adding an initial RGB image of the target object. To calculate the Chamfer Distance (CD) for our SDF volume, we run marching cubes to get the object meshes and extract 30,000 points from each sample. We run the evaluation script 5 times and calculate the average results for report. As shown in Table 1, it is clear that when obtaining tactile image input, our method is superior to other methods. Especially on the visual-tactile 3D reconstruction task, we obtain a very low CD error, which validates the multi-modal fusion ability of our model. The visualization results of V+T settings (one grasp) are shown in Figure 4. ActiveVT generally produces poor visualizations on the mesh surfaces. For point cloud generation, ActiveVT can preserve shapes similar to the ground truth for some structurally simple objects, but for complex shapes or objects with cavities, it loses many details. In contrast, our method is capable of maintaining a complete global shape output for diverse shapes and provides a satisfactory result in local details as well.

Results on the ShapeNet dataset. TouchSDF [7] validates the reconstruction results across six categories in the ShapeNet dataset using the EMD metric. The dataset is devided into three subsets: 1,100 objects for training, 200 for validation and 350 for testing. We evaluate our model using the same dataset partition and evaluation metric. For touch sensing, we adopt the setting of TouchSDF [7], which involves capturing tactile images by poking the target object, ensuring a fair comparison. Within this framework, we are limited to obtaining a maximum of one valid tactile image per touch action. The quantitative results are reported in Table 2. Note that TouchSDF only supports reconstruction from touch but our method supports both tactile only and visual-tactile settings. It can be observed that our method did not yield ideal results initially, but as the number of touches increased, our approach gradually surpassed TouchSDF, demonstrating our method’s ability to integrate tactile information from different locations. Moreover, when combined with visual signals, our results were further enhanced, demonstrating the ability to integrate tactile and visual information to produce a better reconstruction. The visualization results on ShapeNet dataset are reported in the supplementary material.

![](images/2a4253d4b4201c6126ef45e449c7c29396f9b81881cbf33c40f5f6f2f78b5bd1.jpg)  
Figure 4. Qualitative results of ActiveVT [34] and ours. While ActiveVT struggles with visualizations and detail preservation, our method excels in maintaining global shape accuracy across diverse structures, ensuring satisfactory local details

<table><tr><td>Mode</td><td>Method</td><td>Oracle Random</td><td></td><td>Even</td><td>RL</td></tr><tr><td>T</td><td>ActiveVT Ours</td><td>16.38 4.88</td><td>25.83 8.14</td><td>24.53 7.44</td><td>23.84 6.63</td></tr><tr><td>T+V</td><td>ActiveVT Ours</td><td>77.18 75.01</td><td>90.65 88.41</td><td>90.29 87.82</td><td>89.32 86.96</td></tr></table>

Table 3. Comparison of touch exploration on dataset ABC. Numbers represent a ratio (%) between CD after 5 actions and initial CD (with zero grasp).

## 4.3. Evaluation on Policy

We evaluate our touch exploration policy model across both tactile only and visual-tactile settings. As [34], we set two baseline methods, Random and Even. The former policy selects one of the available actions at random while the latter results in uniform coverage of the target object. Furthermore, the Oracle policy is used to select the action which resulted in the best improvement, which is viewed as an upper-bound point of comparison as the true optimal policy cannot be computed in a reasonable time frame. As shown in Table 3, the ratio between CD after 5 grasps and initial CD (with zero grasp) are reported. Our reconstruction method has a higher upper-bound point of reconstruction improvement, which validates the strong touch information fusion power of our model. The evolution of the reconstructed shape with an increasing number of grasps (in the grasp only setting) is illustrated in Figure 5. We also visualize the touch points sampled on the touch charts (predicted from captured valid tactile images). Initially, due to limited information, it is challenging to determine the overall global shape. As the number of grasps increases, our method gradually identifies both the global and local geometric structures.

## 4.4. Ablation Study

We design the ablation study to further validate the necessity of our proposed reconstruction modules. The tactile images are captured through 5 random grasps. As shown in Table 4, we add our proposed modules one by one to validate that each sub-module succeeds to improve the performance. We also train a vision-conditioned diffusion model (V represents the visual only setting) with a contrastive visual encoder. The evaluation results in different modes validate that our method can effectively integrate visual and tactile information to achieve a better reconstruction performance. We also visualize the results among tactile only, visual only, and visual-tactile modalities in Figure 6 (corresponding to rows 3, 5, and 7 in Table 4)

![](images/a7d850e0726c293cda463689a2aff10637bdc01b59a4a37120b256d6867cf2a0.jpg)  
Ground truth

![](images/29313519abdbf4aaf271062534d3c72d9ae2fba5e268c6833be399a252efae4d.jpg)  
# Grasp: 0

![](images/09364e9fa351991c1a96b8bf6aeea4c01b3c0f08975658eb0a758aadcddae2f7.jpg)  
# Grasp: 1

![](images/e9e05c0a160943eed1876f27f33c9022bbf2322c63b3f020d6090b6e456424bd.jpg)  
# Grasp: 2

![](images/e285b62c6723a5bbd65c259a8e575e35a76d7e4adef9ca280e1c0dead11f1f09.jpg)  
# Grasp: 3

![](images/c4b1c2f438b4b160cb323d81e5ee80e838b166ec84825c862a55d41873ddba1f.jpg)  
# Grasp: 4

![](images/11612ea8f3732b724d9382e52f8ed3fff27991a63435221f371cf5500d09a80f.jpg)  
# Grasp: 5

Figure 5. The evolution of the reconstructed shape with an increasing number of grasps (in the tactile only setting). Initially, limited information makes determining the overall global shape challenging, but with more grasp actions, our method effectively improves the reconstruction quality. The local points sampled on predicted touch charts (from captrued valid tactile images) are painted blue.
<table><tr><td rowspan=1 colspan=1>Mode</td><td rowspan=1 colspan=1>CL</td><td rowspan=1 colspan=1>Fusion</td><td rowspan=1 colspan=1>CD↓</td></tr><tr><td rowspan=1 colspan=1>T</td><td rowspan=1 colspan=1>X√√</td><td rowspan=1 colspan=1>XX√</td><td rowspan=1 colspan=1>4.4303.2983.134</td></tr><tr><td rowspan=1 colspan=1>V</td><td rowspan=1 colspan=1>X√</td><td rowspan=1 colspan=1>XX</td><td rowspan=1 colspan=1>2.2422.068</td></tr><tr><td rowspan=1 colspan=1>T + V</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>1.304</td></tr></table>

Table 4. Ablation study results on dataset ABC. CL represents the contrastive encoder and Fusion represents the touch shape fusion module respectively.

## 5. Conclusion

In this work, we present Touch2Shape, which leverages a touch-conditioned diffusion model to explore the target object and reconstruct the 3D shape through touch interaction. The generated latent vector from the diffusion model serves as a compact shape representation, guiding the policy model to predict beneficial touch locations for reconstruction based on potential missing areas. For shape reconstruction, we created a touch embedding module to condition the diffusion model and propose a touch shape fusion module to enhance the reconstructed shape. Extensive experiments demonstrate the effectiveness of our method, showcasing notable enhancements in reconstruction quality and shape optimization through grasping exploration compared to the state-of-the-art methods.

There are also some future directions. Firstly, it’s interesting to transfer Touch2Shape from the simulation to a real robot platform for object exploration and reconstruction. Secondly, the extension of Touch2Shape to full scene reconstruction through tactile exploration presents another fascinating dimension. Lastly, the study of integrating techniques like neural rendering could offer an intriguing pathway for leveraging active touch sensing to synthesis multiview visual image.

![](images/d175d3b9a214fed6061e1b46193d3bf0a6a20570376e2d9075f91faad3530892.jpg)

![](images/af9d63568cd830171d691ece1c8f537f656ea3c8b3c86ac575717dd064a0e116.jpg)

![](images/71caaa6b2203a3114de28a7a460ec6ab61ca46b3d4601f288c744bd0037967fb.jpg)

V

![](images/a3e77916f73eb85cacadb58fdb45ea985dec09f68e7656bfdbb59933e1deb67b.jpg)  
V + T

![](images/9844b81c60b73bcfafad7a1502dadbe27fc6773e58d4f3ccd29d9d606bd38919.jpg)  
GT

Figure 6. visualization results among tactile only (T), visual only (V), and vision+touch (V+T) modalities.

## Acknowledgments

This work is supported in part by the National Key Research and Development Program of China (No. 2022ZD0210500), the National Natural Science Foundation of China under Grant 62441216/62332019, the Distinguished Young Scholars Funding of Dalian (No. 2022RJ01), and the Ningbo Major Research and Development Plan Project of China (No. 2023Z225).

## References

[1] Kumar Ashutosh, Saurabh Kumar, and Subhasis Chaudhuri. 3d-nvs: A 3d supervision approach for next view selection. In International Conference on Pattern Recognition, pages 3929–3936. IEEE, 2022. 3

[2] Xiaoxu Cai, Jianwen Lou, Jiajun Bu, Junyu Dong, Haishuai Wang, and Hui Yu. Single depth image 3d face reconstruction via domain adaptive learning. Frontiers of Computer Science, 18(1), 2024. 2

[3] Angel X Chang, Thomas Funkhouser, Leonidas Guibas, Pat Hanrahan, Qixing Huang, Zimo Li, Silvio Savarese, Manolis Savva, Shuran Song, Hao Su, et al. Shapenet: An information-rich 3d model repository. arXiv preprint arXiv:1512.03012, 2015. 4, 5, 1

[4] Shin-I Cheng, Yu-Jie Chen, Wei-Chen Chiu, Hung-Yu Tseng, and Hsin-Ying Lee. Adaptively-realistic image generation from stroke and sketch with diffusion model. In Proceedings of the IEEE/CVF Winter Conference on Applications ofComputer Vision, pages 4054–4062, 2023. 3

[5] Yen-Chi Cheng, Hsin-Ying Lee, Sergey Tulyakov, Alexander G Schwing, and Liang-Yan Gui. Sdfusion: Multimodal 3d shape completion, reconstruction, and generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4456–4465, 2023. 2, 3, 5, 1

[6] Gene Chou, Yuval Bahat, and Felix Heide. Diffusion-sdf: Conditional generative modeling of signed distance functions. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2262–2272, 2023. 2, 3

[7] Mauro Comi, Yijiong Lin, Alex Church, Alessio Tonioni, Laurence Aitchison, and Nathan F Lepora. Touchsdf: A deepsdf approach for 3d shape reconstruction using visionbased tactile sensing. IEEE Robotics and Automation Letters, 2024. 2, 5, 6, 1

[8] Amaury Dame, Victor A Prisacariu, Carl Y Ren, and Ian Reid. Dense reconstruction using 3d object shape priors. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 1288–1295, 2013. 2

[9] J Denzler and C Brown. An information theoretic approach to optimal sensor data selection for state estimation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 24(2):145–157, 2002. 3

[10] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in Neural Information Processing Systems, 34:8780–8794, 2021. 2, 3

[11] Danny Driess, Peter Englert, and Marc Toussaint. Active learning with query paths for tactile object shape exploration. In IEEE/RSJ International Conference on Intelligent Robots and Systems, pages 65–72. IEEE, 2017. 3

[12] Haoqiang Fan, Hao Su, and Leonidas J Guibas. A point set generation network for 3d object reconstruction from a single image. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 605–613, 2017. 6

[13] Haoqiang Fan, Hao Su, and Leonidas J Guibas. A point set generation network for 3d object reconstruction from a single image. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 605–613, 2017. 2

[14] Georgia Gkioxari, Jitendra Malik, and Justin Johnson. Mesh r-cnn. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 9785–9795, 2019. 2

[15] Xiaoguang Han, Zhaoxuan Zhang, Dong Du, Mingdai Yang, Jingming Yu, Pan Pan, Xin Yang, Ligang Liu, Zixiang Xiong, and Shuguang Cui. Deep reinforcement learning of volume-guided progressive view inpainting for 3d point scene completion from a single depth image. In Proceed ings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 234–243, 2019. 3

[16] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceed ings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 770–778, 2016. 4, 1

[17] Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In Proceedings of the IEEE/CVF Con ference on Computer Vision and Pattern Recognition, pages 9729–9738, 2020. 4, 1

[18] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffu sion probabilistic models. Advances in Neural Information Processing Systems, 33:6840–6851, 2020. 2

[19] Marco F Huber, Tobias Dencker, Masoud Roschani, and Jurgen Beyerer. Bayesian active object recognition via gaus- ¨ sian process regression. In International Conference on In formation Fusion, pages 1718–1725. IEEE, 2012. 3

[20] Nawid Jamali, Carlo Ciliberto, Lorenzo Rosasco, and Lorenzo Natale. Active perception: Building objects’ models using tactile exploration. In IEEE-RAS 16th International Conference on Humanoid Robots, pages 179–185. IEEE, 2016. 3

[21] Sebastian Koch, Albert Matveev, Zhongshi Jiang, Francis Williams, Alexey Artemov, Evgeny Burnaev, Marc Alexa, Denis Zorin, and Daniele Panozzo. Abc: A big cad model dataset for geometric deep learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9601–9611, 2019. 4, 5, 1

[22] Mike Lambeta, Po-Wei Chou, Stephen Tian, Brian Yang, Benjamin Maloon, Victoria Rose Most, Dave Stroud, Raymond Santos, Ahmad Byagowi, Gregg Kammerer, et al. Digit: A novel design for a low-cost compact high-resolution tactile sensor with application to in-hand manipulation. IEEE Robotics and Automation Letters, 5(3):3838–3845, 2020. 2

[23] Chieh Hubert Lin, Hsin-Ying Lee, Willi Menapace, Menglei Chai, Aliaksandr Siarohin, Ming-Hsuan Yang, and Sergey Tulyakov. Infinicity: Infinite-scale city synthesis. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 22808–22818, 2023. 3

[24] Fei Luo, Yongqiong Zhu, Yanping Fu, Huajian Zhou, Zezheng Chen, and Chunxia Xiao. Sparse rgb-d images create a real thing: A flexible voxel based 3d reconstruction pipeline for single object. Visual Informatics, 7(1):66–76, 2023. 2

[25] Takamitsu Matsubara and Kotaro Shibata. Active tactile exploration with uncertainty and travel cost for fast shape estimation of unknown objects. Robotics and Autonomous Sys tems, 91:314–326, 2017. 3

[26] Lars Mescheder, Michael Oechsle, Michael Niemeyer, Sebastian Nowozin, and Andreas Geiger. Occupancy networks: Learning 3d reconstruction in function space. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern recognition, pages 4460–4470, 2019. 2

[27] Alexander Quinn Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models. In International Conference on Machine Learning, pages 8162–8171. PMLR, 2021. 2, 3

[28] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. Advances in Neural Information Processing Systems, 32, 2019. 6

[29] SG Potapova, AV Artemov, SV Sviridov, DA Musatkina, DN Zorin, and EV Burnaev. Next best view planning via reinforcement learning for scanning of arbitrary 3d shapes. Journal of Communications Technology and Electronics, 65: 1484–1490, 2020. 3

[30] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High-resolution image¨ synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10684–10695, 2022. 3

[31] Lukas Rustler, Jens Lundell, Jan Kristof Behrens, Ville Kyrki, and Matej Hoffmann. Active visuo-haptic object shape completion. IEEE Robotics and Automation Letters, 7(2):5254–5261, 2022. 3

[32] Cristian Sbrolli, Paolo Cudrano, Matteo Frosi, and Matteo Matteucci. Ic3d: Image-conditioned 3d diffusion for shape generation. arXiv preprint arXiv:2211.10865, 2022. 2, 3, 4

[33] Edward Smith, Roberto Calandra, Adriana Romero, Georgia Gkioxari, David Meger, Jitendra Malik, and Michal Drozdzal. 3d shape reconstruction from vision and touch. Advances in Neural Information Processing Systems, 33: 14193–14206, 2020. 2, 4, 5, 6, 1

[34] Edward Smith, David Meger, Luis Pineda, Roberto Calandra, Jitendra Malik, Adriana Romero Soriano, and Michal Drozdzal. Active 3d shape reconstruction from vision and touch. Advances in Neural Information Processing Systems, 34:16064–16078, 2021. 1, 3, 4, 5, 6, 7, 2

[35] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International Conference on Machine Learning, pages 2256–2265. PMLR, 2015. 2

[36] Xingyuan Sun, Jiajun Wu, Xiuming Zhang, Zhoutong Zhang, Chengkai Zhang, Tianfan Xue, Joshua B Tenenbaum, and William T Freeman. Pix3d: Dataset and methods for single-image 3d shape modeling. In Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition, pages 2974–2983, 2018. 5, 6

[37] Arash Vahdat, Francis Williams, Zan Gojcic, Or Litany, Sanja Fidler, Karsten Kreis, et al. Lion: Latent point diffusion models for 3d shape generation. Advances in Neural Information Processing Systems, 35:10021–10039, 2022. 3

[38] Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. Advances in Neural Information Processing Systems, 30, 2017. 3, 1

[39] Hado Van Hasselt, Arthur Guez, and David Silver. Deep reinforcement learning with double q-learning. In Proceedings ofthe AAAI Conference on Artificial Intelligence, 2016. 5

[40] Boyan Wei, Xianfeng Ye, Chengjiang Long, Zhenjun Du, Bangyu Li, Baocai Yin, and Xin Yang. Discriminative active learning for robotic grasping in cluttered scene. IEEE Robotics and Automation Letters, 8(3):1858–1865, 2023. 3

[41] Chao Wen, Yinda Zhang, Zhuwen Li, and Yanwei Fu. Pixel2mesh++: Multi-view 3d mesh generation via deformation. In Proceedings of the IEEE/CVF International Confer ence on Computer Vision, pages 1042–1051, 2019. 2

[42] Fengyu Yang, Chao Feng, Ziyang Chen, Hyoungseob Park, Daniel Wang, Yiming Dou, Ziyao Zeng, Xien Chen, Rit Gangopadhyay, Andrew Owens, et al. Binding touch to everything: Learning unified multimodal tactile representations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26340–26353, 2024. 4

[43] Xin Yang, Yuanbo Wang, Yaru Wang, Baocai Yin, Qiang Zhang, Xiaopeng Wei, and Hongbo Fu. Active object reconstruction using a guided view planner. In Proceedings ofthe Twenty-Seventh International Joint Conference on Artificial Intelligence, pages 4965–4971, 2018. 3

[44] Zhengkun Yi, Roberto Calandra, Filipe Veiga, Herke van Hoof, Tucker Hermans, Yilei Zhang, and Jan Peters. Active tactile object exploration with gaussian processes. In IEEE/RSJ International Conference on Intelligent Robots and Systems, pages 4925–4930. IEEE, 2016. 3

[45] Wenzhen Yuan, Siyuan Dong, and Edward H Adelson. Gel sight: High-resolution robot tactile sensors for estimating geometry and force. Sensors, 17(12):2762, 2017. 2

[46] Rui Zeng, Wang Zhao, and Yong-Jin Liu. Pc-nbv: A point cloud based deep network for efficient next best view planning. In 2020 IEEE/RSJ International Conference on Intelligent Robots and Systems, pages 7050–7057. IEEE, 2020. 3

[47] Song-Hai Zhang, Yuan-Chen Guo, and Qing-Wen Gu. Sketch2model: View-aware 3d modeling from single freehand sketches. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6012– 6021, 2021. 3

[48] Zhaoxuan Zhang, Bo Dong, Tong Li, Felix Heide, Pieter Peers, Baocai Yin, and Xin Yang. Single depth-image 3d reflection symmetry and shape prediction. In Proceedings ofthe IEEE/CVF International Conference on Computer Vi sion, pages 8896–8906, 2023. 3

[49] Zhaoxuan Zhang, Xiaoguang Han, Bo Dong, Tong Li, Baocai Yin, and Xin Yang. Point cloud scene completion with joint color and semantic estimation from single rgb-d image. IEEE Transactions on Pattern Analysis and Machine Intelli gence, 45(9):11079–11095, 2023. 3

[50] Linqi Zhou, Yilun Du, and Jiajun Wu. 3d shape generation and completion through point-voxel diffusion. In Proceed ings of the IEEE/CVF International Conference on Com puter Vision, pages 5826–5835, 2021. 3