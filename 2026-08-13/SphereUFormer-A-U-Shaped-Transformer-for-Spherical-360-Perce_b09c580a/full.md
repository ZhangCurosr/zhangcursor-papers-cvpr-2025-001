# SphereUFormer: A U-Shaped Transformer for Spherical 360 Perception

Yaniv Benny Tel Aviv University

Lior Wolf Tel Aviv University

## Abstract

This paper proposes a novel method for omnidirectional 360<sup>◦</sup> perception. Most common previous methods relied on equirectangular projection. This representation is easily applicable to 2D operation layers but introduces distortions into the image. Other methods attempted to remove the distortions by maintaining a sphere representation but relied on complicated convolution kernels that failed to show competitive results. In this work, we introduce a transformer-based architecture that, by incorporating a novel “Spherical Local Self-Attention” and other spherically-oriented modules, successfully operates in the spherical domain and outperforms the state-of-the-art in 360<sup>◦</sup> perception benchmarks for depth estimation and semantic segmentation. Our code is available at https: //github.com/yanivbenny/sphere\_uformer.

## 1. Introduction

Monocular omnidirectional 360<sup>◦</sup> perception is an important setting, as it allows for a full field-of-view (FOV) receptive field for the underlying model. Consequently, an entire room layout can be inferred, for example. In contrast to regular images, a 360<sup>◦</sup> image has no boundaries. Instead, the image is horizontally cyclical, and converges to singular points at the vertical extreme points. This suggests that this domain requires a different treatment, and that best practices that work well with regular images might not have the same effect on 360<sup>◦</sup> images. To tackle this task, multiple topics need to be considered.

How should the data be represented? What kind of distortions and irregularities are created by this representation? How will a model process this representation? How to deal with these distortions and irregularities?

These questions have attracted researchers to construct multiple datasets [3, 4, 9, 53] where depth estimation and semantic segmentation are the main tasks, and to formulate many solutions [20, 23, 28, 32, 35, 36, 38, 42, 43, 47, 48, 51], which propose novel approaches to solving these tasks. A common pattern in previous works is that they represent the data by projecting it to a 2D plane in different ways, usually equirectangular projection [30], cube mapping [26], or patch cropping. Each approach has its advantages and disadvantages, but the common motivation is to represent the image in a grid formation. This enables grid architectures, such as Convolutional Neural Networks or Vision Transformers to easily be applied. This comes at a cost of distortions, irregularities, or limited FOV, that are caused by the selected projection. A less common alternative representation that has been proposed is to represent the image in its original sphere domain by discretizing the sphere surface in some manner [6, 14, 17, 24]. The motivation is to maintain an undistorted representation of the 360<sup>◦</sup> image. However, this representation formed architectural challenges that require novel solutions. Some previous works [8, 12, 27, 46, 49, 50] addressed these challenges in various manners, but they fail to compete with current stateof-the-art solutions that follow a grid representation.

In this work, we tackle the drawbacks of spherical representations. Specifically, we introduce SphereUFormer, which is a novel U-Shaped Transformer architecture tailored for spherical 360<sup>◦</sup> perception. By directly operating on a spherical domain without distortion-inducing projections, SphereUFormer addresses the core challenges of omnidirectional perception. Our approach builds on the strengths of transformer models, incorporating modifications that allow it to efficiently process and understand spherical data. This includes the development of spherical-specific operations such as localized self-attention mechanism that respects the geometry of the sphere, innovative up/downsampling techniques that maintain the integrity of spherical data throughout the model, and positional encoding scheme that matches the domain properties. SphereUFormer outperforms the state of the art in the field of 360<sup>◦</sup> perception, providing a robust and efficient tool for a wide range of applications that require comprehensive understanding of spherical environments. In Sec. 3 we describe this representation in detail. We then further describe a new method in Sec. 4, which utilizes this representation and performs 360<sup>◦</sup> perception on spherical representations, and in Sec. 5, we analyze and compare its results.

## 2. Related Work

Omnidirectional perception. There have been many advances, novelties, and research verticals in omnidirectional perception over the years. Initial works attempted standard convolutional networks on an equirectangular projection [54] (ERP). Although promising, the results were affected by the distortions created by the projection. To better handle these distortions, later works suggested technical adjustments to the convolution kernel such as Spherical Convolutions [37] or Distortion-Aware Convolutions [12, 39]. The latter sample the convolution features from a tangent plane. Other solutions attempted to eliminate distortions by sampling the image into smaller FOV patches [16, 28, 35, 47] followed by a post-processing stage that fuses the independent results. This, however, is at a cost of a restricted receptive field per patch and possible cutoff of crucial information, which either affects results or requires high over lap between patches. Due to the advantages and limitations of each method, [2, 42, 43] suggested a combined approach that fuses the ERP path with the individual patches at some stage of the network. Alternative approaches avoided the use of projection, and instead, operated directly on the sphere by constructing a spherical mesh. Usually either an icosphere [13, 22, 27, 46] or healpix [8, 13], and apply Graph Convolution [52], Mesh Convolution [21], or some custom convolution-like operation.

Vision Transformers. ViT’s [15] have disrupted the field of computer vision, from general-purpose architectures [34, 44] to specific purposes [7, 10, 45]. The attention mechanism [5, 29, 40] allows for dynamic pooling of features based on relevance and structural position. ViT’s benefit from an efficient large context window compared to their CNN counterparts, which allows them to more easily learn long-distance relations between features. While earlier omnidirectional perception works proposed solutions based on CNN’s, recent contributions replaced the convolution kernels with local attention of various forms [20, 36, 48] with favorable results. This work follows this direction. It utilizes the foundation of UFormer [44] and adapts it to spherical representations with custom attention layers.

Graph Attention. Graph Convolution Networks [52] (GCN) are neural networks that apply convolution-like operations on data represented as a graph. A graph convolution operation aggregates the results of local operations between a node and each of its neighbors. Over multiple layers, this allows for information to pass between distant nodes in the graph and the update of these nodes. Graph Attention Networks [41] were proposed to enhance Graph Convolution Networks, thereby improving the messagepassing capabilities of the GCN. The attention mechanism has the benefit of allowing the operations to focus on a few sets of relevant nodes and providing an efficient method to encode positional information in the graph. For Geometric Graphs [31] geometric location is encoded to inform of each node’s absolute and relative position, thereby making nodes aware of their global position and their respective neighbors’ relative position.

![](images/115e72aecddd3e813fcf98ab47a76b3668197c704cd4e1d02b910f7cef8e9bee.jpg)  
(a)

![](images/604327312505f1167ec54f481fb30e16a1d46092571694d51f90343877f734e8.jpg)  
(b)

![](images/8c365085a83574e2517bbbd5445ec305ede1ed599290f76686704c6893b0cb05.jpg)  
(c)

![](images/55d629ddd9a0853e4f8ba754df3f5814da889256a5141991079733fbdd9a9984.jpg)  
(d)  
Figure 1. Sphere Representations. From left to right: uvsphere, cubesphere, icosphere, hexasphere.

## 3. Sphere Representation

Arguably the first topic in question when attempting to solve a problem is how to formulate the data. For omnidirectional 360<sup>◦</sup> images, the data can be seen as an array of values $v _ { \phi , \theta }$ assigned to a ray pointing from a center pointof-view towards $( \theta , \phi )$ for $\theta \in [ 0 , 2 \pi ]$ being the horizontal angle and $\phi \in [ 0 , \pi ]$ the vertical. The value v can correspond to the RGB value, depth, object category, etc. By selecting to sample on finite such angle pairs, the sphere becomes discretized. The sampling method determines how the sampled data can be processed and how successful the solution will become.

The most common and straightforward is a uvsphere [17], where points are sampled in a grid with equa spacing for $( \theta , \phi )$ . See Fig. 1a. The uvsphere has a very high degree of horizontal symmetry, which is a good property. Another advantage is that it allows an unwrapping of the sphere into a 2D matrix. This unwrapping is called equirectangular projection (ERP). The downside of this representation is that the density of samples on the sphere is higher around the poles, due to the decreasing cross-section. This creates an imbalanced effective resolution over the sphere, and the projected image in 2D is distorted.

An alternative that mitigates imbalanced resolution and distortion is cube mapping (or cubesphere) [14] (Fig. 1b), which projects the sphere onto the faces of a cube. Sample resolution is controlled by deciding how many rectangles each face of the cube is subdivided into. This removes the sampling density around the poles and the sphere can be unwrapped into six 2D images. However, two of these images are not vertically oriented and require special treatment. In addition, the transition between faces is not fluent, and complex padding methods [11] can only increase the receptive field of each image to some extent. Since the images are separated, a post-processing stage needs to fuse their predictions to form a unified prediction on the sphere. One option is to operate on the cubesphere mesh instead of the 2D projection. Unfortunately, the cubesphere has irregularities at the coordinates of the cube corners and does not have a high horizontal symmetry, which makes it less optimal for this purpose.

![](images/98fe9d1ee78c77fcaf52a7555e34fe5b99115d3ada3fb9bb43df0e987acb26d1.jpg)

![](images/602b19acaac3d8d82f9e7d413aa1a0894ed3d882fc1efea5536a507305347784.jpg)  
0

![](images/5e99b40b67946716292918478998e2fbd4a17fa682987c74b3229f3b1d7db57a.jpg)  
1

![](images/19675621d25e30401df4f9ebb0aad3c6be64ecd75f910a30e16e76d0781f9245.jpg)  
2

3  
![](images/3125e0334e0b254c8477ad0faf9811fc24342e12b2190f607a242d0e678ff614.jpg)  
4

Figure 2. Icospheres of different ranks. An increase in rank is made by subdividing each triangle into 4 smaller triangles, and results in an increased resolution.
<table><tr><td>Rank: Faces:</td><td>0 1 2 3</td><td>4 5</td><td>6 7 20 80 320 1280 5120~20K~82K~328K</td></tr><tr><td></td><td></td><td>Vertices: 12 42 162642 2562 ~10K ~41K ~164K</td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr></table>

Table 1. Icosphere Properties. Number of faces (20 × 4<sup>i</sup>) and number of vertices $( 1 0 \times 4 ^ { i } + 2 )$ per subdivision rank i.

![](images/85988a19b066e30295f496d2de848945c518c8d9cc59193d66f69d7ef6ab16d5.jpg)  
(a1)

![](images/7f1ffcbcf3276026036f9dc45cc48cd7dfd930e8ae9c69ff11658cd715e79a5a.jpg)  
(b1)

![](images/32e7bbe2d398f55ea622c74b39d8a2d10d010902bea7ea731ef7216cfc15e031.jpg)  
(c1)

![](images/da41c16c853387679f1558c00025a5f6a99f018c9f93feee05241c854eb026ca.jpg)

![](images/a8a550743c2e913f156d1ac0edd892f8cb4cf93be9b3899a0693ee9536a513ec.jpg)  
(b2)

![](images/335521e08a1ad984b00ac0ba01bee3ca3db9e747b0fd056a08cea8ddf2e8e041.jpg)  
(c2)  
Figure 3. Data as Icospheres. Icosphere of rank 3 (a1-c1) and rank 6 (a2-c2). (a) RGB, (b) Depth Map, and (c) Semantic Layout.

Instead of projecting the data to a grid and then having to deal with its limitations, a more elegant approach can be formulated using the sphere representation directly. Aside from the previously mentioned options, a common way to discretize a sphere is with Geodesic (icosphere, Fig. 1c) and Goldberg (hexasphere, Fig. 1d) polyhedrons [6]. An icosphere is created by repeatedly subdividing an icosahedron into smaller triangles. A hexasphere is the dual polyhedron where vertices are swapped with face normals, and edge connections with face adjacency. These types of sphere representation have a very high degree of symmetry and have a very evenly spaced sampling on the sphere surface. Due to their duality, our work follows the icosphere mesh, and icosphere/hexasphere mode is toggled by whether data points (θ, ϕ) describe mesh vertices (hexasphere) or face normals (icosphere). Therefore we will sometimes refer to icospheres/hexaspheres by referring to the “node type” (face/vertex respectively). Other notable options for sphere sampling are Fibonacci spheres [24] and HealPix [19].

Icospheres have the highest order of symmetry, the most uniform distribution of points (perfect uniform distribution is impossible), easy setup, and an inherent up/down sampling definition through subdivision. All these reasons make them ideal for discrete spherical representation. As Fig. 2 and Tab. 1 show, in different ranks, icospheres produce sphere discretization in various resolutions. Each increase in sphere rank divides each triangle face into four smaller ones, hereby gradually increasing the sphere resolution. An example of representing data with icospheres can be seen in Fig. 3. In low resolution, the image seems “pixelated” as the independent triangle faces are distinguishable. But, with a high enough resolution, the image becomes detailed and visibly smooth. Our method uses high-resolution icospheres to represent the 360<sup>◦</sup> image, and through up and downsampling on the sphere structure, uses low-resolution spheres as hidden layers inside the model.

## 4. Method

This section introduces SphereUFormer, our proposed U-Shaped Transformer for Spherical 360<sup>◦</sup> Perception. As the name suggests, the model is influenced by UFormer [44], with significant modifications to support spherically structured data. A high-level diagram is depicted in Fig. 4. The model’s components are described in this section. For technical implementation details, see the supplementary.

The network begins with an input projection that applies linear projection from the RGB values to latent embedding vectors. The projection output is then passed into an Encoder-Decoder network with multiple levels of selfattention modules in different sphere resolutions. We call these modules “Spherical Attention Modules” (SAM). At the bottom of the network lies a bottleneck SAM module that operates on the lowest resolution. Skip connections pass from the encoder to the decoder and feed highresolution features. Finally, a linear Output Projection projects the decoder output to the output dimensions.

As described in Sec. 3, our solution features the icosphere as its spherical representation structure, as we have found it to be the most intuitive and to contain the most satisfying properties. But it is not limited to it, and with minor modifications supports any spherical structure.

Our method contains many local operations on the graph, from the down and up sampling, to the self-attention mechanism. Computing the cosine similarity or finding the neighbors of a node to generate the mapping scheme is slightly costly. This would be a problem if it was performed on each iteration, as the model would be slowed down. Luckily, since the icosphere graph is fixed, these mappings only need to be computed once in advance. Therefore there is no need to recompute it in every message massing operation.

## 4.1. Up/Down Sampling

In the spherical setting, upsampling and downsampling operations are performed by transforming a spherical graph into a new one with more or fewer nodes, respectively. This can be difficult and inefficient to compute for two arbitrary spherical polyhedrons. However, icospheres have multiple elegant and desired properties. First, an increase in resolution by one rank is the result of splitting each triangle face into four smaller ones, with one central triangle that shares its center with the large one. Second, for the same reason, each vertex in the low-resolution icosphere also exists in the high-resolution one, with new vertices created at the center of each edge connection. Third, the number of nodes/faces increases at a constant rate.

![](images/7d9bcbc2432e0c8ae80c27433d9322c998bcb04805a7739246bbb88b863057ce.jpg)  
Figure 4. The SphereUFormer architecture. A spherical representation is fed into the model. A linear input projection layer encodes the RGB values to latent embedding vectors. A sequence of SAM modules apply local self-attention on the spherical data along with downsampling layers that gradually reduce the resolution of the sphere. A sequence of SAM modules along with an upsampling layer and bypass skip connections decode the data. An output projection converts the latent embeddings to the output channel size.

All these properties make constructing up and down operations extremely straightforward and efficient. Every common pooling technique, such as max pooling, average pooling, center pooling, and interpolation, can be implemented. However, there are preferences for each setting. For downsampling, when the data nodes are represented by the faces, we have found max and center pooling to have the same overall results. When the data nodes are on the vertices (equivalent to hexasphere), max and average pooling become a little complicated, since there are both instances of five and six neighbors in the graph. Therefore, we resorted to only center pooling. For upsampling, the faces mode most simply operates with nearest upsampling, where each triangle receives the value according to the lower resolution triangle it was subdivided from. Other interpolation methods require more complicated computations and were therefore not considered. For vertices, however, interpolation is straightforward, since each new node is located precisely at the center of an existing edge. For more upscaled upsampling, Barycentric coordinates [18] interpolation can also be performed quite easily, but we did not find a reason to upscale beyond a factor of 2 at a time.

## 4.2. Global Positional Encoding

Positional encoding is a common and critical technique in Transformers [33, 40] and Vision Transformers [7, 10, 15, 34, 44, 45]. SphereUFormer utilizes positional encoding as well, but in a manner that suits spherical representations. The node position on the unit sphere is expressed using spherical coordinates (θ, ϕ). In our setting of omnidirectional perception, the data is always vertically aligned, therefore the model does not need to be vertically equivariant. However, the horizontal rotation is arbitrary, therefore it is desired to make the model horizontally equivariant to rotation and flipping. We therefore apply absolute positional encoding for the vertical position but not for the horizontal position. Horizontal position is only used in relative to another node as position bias in the Local Self-Attention described in the next section.

For the vertical position, we consider $\phi \in [ 0 , \pi ]$ . The position was encoded by applying sinusoidal encoding on the position value, then passed through an MLP with an output dimension equal to the embedding dimension. We apply the vertical global positional encoding on the projected input and on the queries and keys in each attention module. More details can be found in the supplement.

## 4.3. Spherical Local Self-Attention

The Spherical Local Self-Attention is the main component in our network that drives the SphereUFormer’s 360<sup>◦</sup> perception. A detailed diagram is shown in Fig. 5. Given an input sphere, each node $x _ { i }$ in the sphere is projected into a query $( q _ { i } ) .$ , key $( k _ { i } )$ , and value $( v _ { i } )$ . These vectors are all of the shape $( N , L , H , D _ { H } )$ , where H is the number of heads and $D _ { H }$ is the head dimension. Similar to standard practices, $D _ { H }$ is derived from the input dimension D and the number of heads. However, since the self-attention mechanism is the only spatial operation in the network, we added a “head dimension coefficient” $( C _ { h e a d } )$ hyperparameter to upscale the head dimension $D _ { H }$ as a reverse bottleneck. Thus: $D _ { H } = ( D / H ) \cdot C _ { h e a d }$ . This coefficient allows an increase in the number of attention heads without reducing the dimension of each head and without increasing the overall size of the entire network, resulting in only a minor increase in parameters. We further discuss the motivation of this in the ablation section (Sec. 5.4).

![](images/7cd3b1736ecaf65807aa323066c476d666bcd89bae574a854c89834ec1ccfd38.jpg)  
Figure 5. Spherical Local Self Attention. Attention is applied<sup>Q</sup>K<sub>V</sub> <sup>dot A</sup> (N,L,H,K) <sup>O</sup>U <sub>SAMD</sub>between each data node and its K neighbors. A learned relative<sup>o</sup>je<sup>IN K</sup>(N,L,D) (N,L,H,K,DH) <sup>OUTr</sup>oj (N,L,D)softmax <sup>w</sup>n <sup>M</sup><sub>+</sub>position bias encodes information about the neighbors’ relative po-<sup>o</sup>n i<sup>o</sup>n <sup>B B a</sup>m <sup>o</sup>wsition. In the right corner is a diagram of the enclosing block.<sup>(N,L,H,K,DH) (N,L,H\*DH)(N,L,H,DH)</sup>

![](images/4225bcc6fa5639dd1b45f55ca72118a58a20429d2e305e1bd5c4c492b539573a.jpg)  
Figure 6. Relative Positional Encondig. On the left, a center point (blue) and its first (green) and second (yellow) degree neighbors are defined by the graph structure. The relative position kernel is defined by a $7 \times 7$ grid, represented by the black nodes in the right image. The center point is positioned at the center of the grid and the neighbors are distributed across the grid according to their relative change in horizontal (∆θ) and vertical (∆ϕ) angles. Bilinear interpolation samples a learned positional encoding.

For each query, the neighboring K vectors are grouped according to the sphere geometry. Attention for each query is performed on the selected neighboring group. The number of neighbors K depends on a hyperparameter named “window coefficient” $( C _ { w i n } )$ , which dictates the N-order neighbors to collect. A window coefficient of 0 provides no neighbors, a coefficient of 1 provides only first-order neighbors, 2 also provides second-order neighbors, and so forth. Note that due to irregularities in the graph structure, the number of keys for each node is not identical as some have slightly fewer neighbors than others. This is homogenized for parallelization purposes by adding null neighbor keys and masking them.

For relative position information, a learned relative position bias is added to the attention map. Since having a learned parameter per query-key pair is both memory expensive and harmful in terms of inductive bias, we took a shared-weights approach. For each node $i \in [ 1 , L ]$ , we measure the angular difference $( \Delta \phi _ { i , k } , \Delta \theta _ { i , k } )$ for each key $k \in [ 1 , K ]$ and normalize them by the max deltas. This is only computed once during initialization. During runtime, the normalized deltas (in [−1, 1]) are used to sample learned parameters from a $7 \times 7$ window. This sampling from a non-linear 2D function acts as a cheap relative positional encoding for the query-key pairs. Fig. 6 illustrates the relative position implementation. Given the blue center points, its first-order neighbors are depicted in green, and second-order neighbors in yellow, are placed in their relative position. A 7×7 grid, represented by the black points, represents the learned weights of the grid. Each neighbor gets its positional encoding by bilinearly interpolating between the four nearby weights.

## 5. Experiments

We evaluate our method by comparing it with state-of-theart methods for depth estimation and semantic segmentation. For this purpose, we used the Stanford2D3D [4] and Structured3D [53] datasets. Both contain RGB, depth, and semantic segmentation data. We trained our model and all baselines using the training configuration of PanoFormer [36]. For more information on the training protocol please refer to the supplementary.

We compared against ERP methods [20, 36, 48], spherical methods [8, 49], and Elite360D [1], that utilizes a combination of ERP with a low-resolution icosphere. We did not compare against patch-based solutions, as the difference in compute resources makes an unfair comparison.

## 5.1. Design Decisions

Certain design decisions were made for the evaluation section. Some due to fairness considerations when comparing to the baselines, and others after testing various alternatives.

For fairness reasons, it is important to compare methods of similar size. We configured all models to have 4 downsampling stages, and a depth (number of blocks per stage) of 2 for each scale. Tab. 2 show the parameter and flop count per model. We gave PanoFormer [36], EGFormer [48], Elite360D [1] and HerRUNet[49] base embedding dim of 32, SFSS [20] 48, HealSWIN [8] 64. The resnet18 backbone was used for Elite360D. This resulted in as equal as possible resolution, parameter count, and flops that the different architectures allow. We analyzed input resolutions of 256×512 and $5 1 2 \times 1 0 2 4$ for the grid models. Resolution of the spherical models was chosen to be as close as possible to those pixel counts. This was with subdivision ranks 7 and 8 respectively. Also for fairness, none of the methods used pretrained weights. To evaluate all models fairly and on equal terms, the predictions in the various representations were projected uniformly to the sphere surface. Unlike evaluating directly on ERP, this provides an evaluation that evenly averages over the sphere surface and is not biased to extreme vertical angles.

<table><tr><td>Model</td><td>Res. (#Pixels)</td><td>Params</td><td>Flops</td></tr><tr><td>PanoFormer [36]</td><td>256×512 (131K)</td><td>14.5M</td><td>11.8G 15.6G</td></tr><tr><td>EGFormer [48] SFSS [20]</td><td>256×512 (131K) 256×512 (131K)</td><td>15.2M 15.1M 14.7M</td><td>18.9G 13.6G</td></tr><tr><td>Elite360D HealSWIN HexRUnet [49]</td><td>[1] 256×512 †(136K) [8]  $1 2 \cdot 4 ^ { 7 }$  (196K)  $1 0 \cdot 4 ^ { 7 } + 2$  (164K)</td><td>12.0M 14.0M</td><td>39.0G</td></tr><tr><td>PanoFormer [36] EGFormer [48]</td><td> $5 1 2 \times 1 0 2 4$  (524K)  $5 1 2 \times 1 0 2 4$  (524K) 15.2M</td><td>14.5M</td><td>12.4G 44.7G 65.8G</td></tr></table>

Table 2. Resolution and parameter count of baseline models. <sup>†</sup>Elite360D uses additional icosphere of rank 4 (5120 data points).
<table><tr><td>#</td><td>Rank Type</td><td> $\mathbf { C } _ { h e a d }$ </td><td> ${ \bf C } _ { w i n }$ </td><td>Res.</td><td>Params</td><td>Flops</td></tr><tr><td rowspan="5"></td><td>6 hex</td><td>1</td><td>1</td><td>41K</td><td>11.2M</td><td>2.5G</td></tr><tr><td>6</td><td>ico</td><td>1 1</td><td>82K</td><td>11.2M</td><td>4.9G</td></tr><tr><td>7</td><td>hex</td><td>1 1</td><td>164K</td><td>11.2M</td><td>9.9G</td></tr><tr><td>7</td><td>ico</td><td>1 1</td><td>328K</td><td>11.2M</td><td>19.1G</td></tr><tr><td>7</td><td>hex 2</td><td>1</td><td>164K</td><td>14.9M</td><td>13.0G</td></tr><tr><td>(1) (2)</td><td>7 hex 8 hexX</td><td>2 2</td><td>2 2</td><td>164K 655K</td><td>14.9M 14.9M</td><td>13.1G 52.7G</td></tr></table>

Table 3. Resolution (total data points) and parameter count of various configurations of our model. Selected configurations in green.

We followed these guidelines to select the sphere setting with the closest number of nodes, total parameters and flops to the baselines. These stats are reported in Tab. 2 for the existing methods, while Tab. 3 lists variations of our method concerning: sphere rank, node type, head, and window coefficients $( C _ { h e a d } , C _ { w i n } )$ . As can be seen, spheres with a rank of 6 have a very low resolution, regardless of the node type. For spheres with rank 7, we have found that the “vertex” node type (with 164K nodes) provides a resolution that is closest to the 256×512 (total of 131K pixels) resolution, and rank 8 is the closest to 512×1024. The base configuration of our method has fewer parameters and flops than the baselines. This is largely because, unlike most baselines, our method does not contain spatial layers, such as convolution kernels, outside of the attention layer. As a result, the base configuration of our method is less powerful. This is also further addressed in the ablation section (Sec. 5.4). To minimize this difference, we have found that increasing the head coefficient results in a model with slightly more parameters to match the minimum of the baselines. Increasing the window coefficient does not add any parameters and an insignificant amount of flops. Finally, we have selected two configurations for comparison with the baselines. One with rank 7 to be compared to 256×512 models, and one in rank 8 for larger resolution. These are highlighted in Tab. 3.

## 5.2. Depth Estimation

In depth estimation, the task is a dense prediction of the pixels depths. We evaluated the depth prediction up to 10 meters for Stanford2D3D and 5 meters for Structured3D. Following PanoFormer [36], we used the BerhuLoss [25] as a differentiable criterion to train the models. We then measured the Mean Absolute Error (MAE), Mean Relative Error (MRE), Root Mean Squared Error (RMSE), and $\delta _ { 1 }$ accuracy, which are standard evaluation metrics for depth estimation. Quantitative results on 256×512 are reported in Tab. 4. Our method is shown to outperform the baselines on all but one metric, where it was second. In evaluation on Stanford2D3D in 512×1024 (Tab. 5), the gap is even larger, and our method significantly improved while the baselines did not. Fig. 7 presents sample results. Evidently, our results are considerably sharper than the baselines, especially around the center, and handle distortions around the poles better. We argue that this is due to spherical representation having a better effective resolution at the center, and does not suffer from distortions. We have also found depth predictions of most baselines to have a boundary misalignment effect where the 360<sup>◦</sup> and 0<sup>◦</sup> meet, which our method has no such effect. This is further explored in the supplement.

## 5.3. Semantic Segmentation

In semantic segmentation, the task is to predict a semantic class per image pixel. Stanford2D3D has 13 categories, while Structured3D has 40. We used the standard Categorical Cross Entropy loss, where the background category was ignored. For evaluation, we measured the global pixel accuracy and the mIoU percentage, which are standard metrics for semantic segmentation. Results are depicted in Tab. 4. Tab. 5 shows additional high-resolution setup. In this task as well, our method outperformed the baselines on all metrics. Fig. 8 presents typical samples. Similarly, our semantic segmentation results present the same improvement in terms of distortion handling and improvement around the center of the image we have observed for depth estimation.

<table><tr><td rowspan="3"></td><td rowspan="3"></td><td colspan="8">Depth Estimation</td><td colspan="4">Semantic Segmentation</td></tr><tr><td colspan="4">Stanford2D3D</td><td colspan="4">Structured3D</td><td colspan="2">Stanford2D3D</td><td colspan="2">Structured3D</td></tr><tr><td>MAE↓ MRE↓ RMSE↓</td><td></td><td></td><td> $\delta _ { 1 } \uparrow$ </td><td>MAE↓ MRE↓ RMSE↓</td><td></td><td></td><td> $\delta _ { 1 } \uparrow$ </td><td>Acc.↑ mIoU↑</td><td></td><td>Acc.↑ mIoU↑</td><td></td></tr><tr><td>PanoFormer</td><td>[36]|</td><td>.174</td><td>.078</td><td>.451</td><td>92.5</td><td>.154</td><td>.051</td><td>.330</td><td>94.8</td><td>83.1</td><td>60.6</td><td>94.9</td><td>49.7</td></tr><tr><td>EGFormer</td><td>[48]</td><td>.170</td><td>.075</td><td>.456</td><td>93.1</td><td>.150</td><td>.049</td><td>.328</td><td>95.2</td><td>86.5</td><td>66.4</td><td>95.0</td><td>51.5</td></tr><tr><td>SFSS</td><td>[20]</td><td>.179</td><td>.081</td><td>.455</td><td>92.2</td><td>.155</td><td>.051</td><td>.330</td><td>95.0</td><td>86.9</td><td>68.2</td><td>95.2</td><td>51.9</td></tr><tr><td>HexRUnet</td><td>[49]]</td><td>.201</td><td>.090</td><td>.511</td><td>90.1</td><td>-</td><td>-</td><td>-</td><td>-</td><td>81.7</td><td>56.1</td><td>-</td><td>-</td></tr><tr><td>HealSWIN</td><td>[8]</td><td>.189</td><td>.084</td><td>.482</td><td>92.2</td><td>一</td><td></td><td>一</td><td>-</td><td>85.5</td><td>63.2</td><td></td><td></td></tr><tr><td>Elite360D</td><td>[1]</td><td>.169</td><td>.069</td><td>.448</td><td>93.5</td><td>.147</td><td>.046</td><td>.321</td><td>95.9</td><td>87.4</td><td>71.4</td><td>95.3</td><td>52.0</td></tr><tr><td>OURS (Tab. 3 (1))</td><td></td><td>.165</td><td>.071</td><td>.432</td><td>94.0</td><td>.142</td><td>.045</td><td>.314</td><td>96.4</td><td>88.6</td><td>72.2</td><td>95.8</td><td>53.0</td></tr></table>

Table 4. Quantitative comparison on sphere rank 7 (256×512) resolution.

![](images/d7cc5df154219640f1f38aec4fb24a76eb428a94a7d01a6616142c1be377b2ce.jpg)

![](images/5dd4ce4217878303250772cb0583c40792d7803b4b49b34bde17bf1490b59d07.jpg)

![](images/0d542b136300b5080ffaaa98f347628b5b99c3e5b4c8ce5c907395e103f43643.jpg)

![](images/7282d46fe69f515c933eeb6bb4e5da9d9628e97b6eca0caa57e13f872e42a39e.jpg)

![](images/6b2a552c9cf253e2394970dfa93067eb45e0c952930cd6845b3fd27cf6a2cd1b.jpg)

![](images/31f79ead79f54350b994c7b2f23da0e0bb3e4d5ba50937dcd7bb332f77a8da7e.jpg)

Figure 7. Depth Estimation. Top: Stanford2D3D. Bottom: Structured3D.
<table><tr><td>Model</td><td>|MAE↓ MRE↓</td><td> $\delta _ { 1 } \uparrow$ </td><td>|Acc.↑ IoU↑</td></tr><tr><td>PanoFormer Elite360D</td><td>.167 .072 .181 .077</td><td>93.7 93.2</td><td>82.4 55.6 84.5 563.3</td></tr><tr><td>OURS (Tab. 3 (2))</td><td>.147 .065</td><td>94.0</td><td>89.1 71.5</td></tr></table>

Table 5. Comparison on Stanford2D3D in rank 8 (512×1024).

## 5.4. Ablation Study

To explore the contribution of each one of the elements we have conducted several ablation studies in which multiple variants of our method are tested. These tests were conducted in the context of depth estimation on Stanford2D3D.

Positional Encoding Two main components of the method are the global absolute and relative bias position encodings. The vertical global position encoding injects vertical position while remaining rotation equivariant horizontally. The relative bias within the self-attention provides relative information. We compared various configurations that omit one or both of the encodings. Results can be seen in Tab. 6. Evidently, the absence of positional encoding, either global or relative, drastically impacts the performance of the model. Also, relative positional bias proved more critical than the global one.

Scaling Coefficients Two key modifiable scaling coefficients, the “window” $( C _ { w i n } )$ and “head” $( C _ { h e a d } )$ , affect the capacity of the attention module. The former affects the size of the attention window to incorporate further nodes in the graph. The latter increases the capacity of the attention head by increasing the head dimension. We tested the effect of these coefficients by gradually increasing them separately and together. As shown in Tab. 7. Increasing the window up until $C _ { w i n } = 2$ improves the performance. Interestingly, further increase of the window did not show an improvement. This could suggest that such large windows interfere with convergence. As expected, increasing the attention head also improves results, even up to $C _ { h e a d } = 4 .$ Unsurprisingly, this suggests that the model could benefit from an overall increase in parameters, which is beyond the scope of this work.

![](images/9387ae6f6a0d2eda9a0c0d6fe5ac93d95d4cc4e8f62441b848a65599a6d2161a.jpg)  
Figure 8. Semantic Segmentation. Top: Stanford2D3D. Bottom: Structured3D.

<table><tr><td>Variant</td><td>AbsPos</td><td>RelPos</td><td>MAE↓</td><td>MRE↓</td></tr><tr><td>No Pos. Enc.</td><td>x</td><td>x</td><td>.326</td><td>.094</td></tr><tr><td>No Rel. Pos. Enc.</td><td>√</td><td>x</td><td>.251</td><td>.091</td></tr><tr><td>No Abs. Pos. Enc.</td><td>x</td><td>√</td><td>.218</td><td>.088</td></tr><tr><td>With Pos. Enc.</td><td>V</td><td>√</td><td>.189</td><td>.077</td></tr></table>

Table 6. Ablation Study: Positional Encoding.
<table><tr><td>Variant</td><td> $C _ { h e a d }$ </td><td> $C _ { w i n }$ </td><td>MAE↓</td><td>MRE↓</td></tr><tr><td>Base variant</td><td>1</td><td>1</td><td>.189</td><td>.077</td></tr><tr><td>No window</td><td>1</td><td>0</td><td>.412</td><td>.122</td></tr><tr><td>Medium window</td><td>1</td><td>2</td><td>.175</td><td>.073</td></tr><tr><td>Large window</td><td>1</td><td>3</td><td>.180</td><td>.078</td></tr><tr><td>Medium head</td><td>2</td><td>1</td><td>.171</td><td>.075</td></tr><tr><td>Large head</td><td>4</td><td>1</td><td>.162</td><td>.067</td></tr><tr><td>Conf. (1) Tab. 3</td><td>2</td><td>2</td><td>.165</td><td>.071</td></tr></table>

Table 7. Ablation Study: Head and Window Coefficients.

## 6. Discussion and Limitations

This paper introduces a pioneering approach to omnidirectional 360<sup>◦</sup> perception, leveraging a geodesic polyhedron and a unique architecture designed for spherical computation. However, there is ample room for optimization and enhancement of its features. The potential for improvement spans various aspects of the model, from its core architecture to its computational efficiency.

One of the promising avenues for future work lies in exploring additional domains where this architecture could be applied. Beyond the realms of depth estimation and semantic segmentation in 360<sup>◦</sup> imagery, spherical data representation is also prominent in other areas such as medical imaging, where spherical structures are common, and geographical data analysis, which often deals with global-scale data that naturally fit a spherical model. The adaptability of our approach to these varied domains could pave the way for groundbreaking applications and methodologies.

While the current work has focused on supervised classification and regression tasks, extending this model to generative models presents an exciting frontier. Generative models could leverage the spherical data representation for creating highly realistic and comprehensive 3D environments among other applications. The unique challenges of generative tasks on spherical domains, such as ensuring continuity and avoiding distortions, call for accurate modeling of the type we present here.

Another area for improvement is the computational efficiency of the model, particularly regarding its GPU usage. Despite the number of parameters and flops being on par with baseline models, our approach exhibits slower per-<sub>formance (</sub>\~<sub>30% slower than PanoFormer). This is largely</sub> because the gathering operations on the sphere graph are not optimized with dedicated CUDA functions in PyTorch, leading to less efficient execution. Addressing this engineering challenge could significantly enhance the model’s appeal by reducing both the GPU footprint and runtime.

## 7. Conclusion

We have presented a novel method for omnidirectional perception that successfully utilizes spherically represented data. The method addresses the underlying spherical structure, which comes up in three different places: in the down and upsampling operators, in the positional encoding, and, above all, in the definition geometric representation and that of the neighborhoods for the local attention operator.

Our proposed method relies on the icosphere representation and utilizes its favorable properties to derive spherebased operators. Using these, we surpass the state of the art in omnidirectional perception in both depth estimation and semantic segmentation.

## Acknowledgements

This work was supported by the Tel Aviv University Center for AI and Data Science (TAD). The contribution of the first author is part of a PhD thesis at Tel Aviv University.

## References

[1] Hao Ai and Lin Wang. Elite360d: Towards efficient 360 depth estimation via semantic-and distance-aware biprojection fusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9926–9935, 2024. 5, 6, 7

[2] Hao Ai, Zidong Cao, Yan-Pei Cao, Ying Shan, and Lin Wang. Hrdfuse: Monocular 360deg depth estimation by collaboratively learning holistic-with-regional depth distributions. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13273–13282, 2023. 2

[3] Georgios Albanis, Nikolaos Zioulis, Petros Drakoulis, Vasileios Gkitsas, Vladimiros Sterzentsenko, Federico Alvarez, Dimitrios Zarpalas, and Petros Daras. Pano3d: A holistic benchmark and a solid baseline for 360° depth estimation. In 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pages 3722– 3732, 2021. 1

[4] Iro Armeni, Sasha Sax, Amir R Zamir, and Silvio Savarese. Joint 2d-3d-semantic data for indoor scene understanding. arXiv preprint arXiv:1702.01105, 2017. 1, 5

[5] Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. arXiv preprint arXiv:1409.0473, 2014. 2

[6] Herbert Busemann. The geometry of geodesics. Courier Corporation, 2012. 1, 3

[7] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-toend object detection with transformers. In European conference on computer vision, pages 213–229. Springer, 2020. 2, 4

[8] Oscar Carlsson, Jan E Gerken, Hampus Linander, Heiner Spieß, Fredrik Ohlsson, Christoffer Petersson, and Daniel Persson. Heal-swin: A vision transformer on the sphere. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6067–6077, 2024. 1, 2, 5, 6, 7

[9] Angel Chang, Angela Dai, Thomas Funkhouser, Maciej Halber, Matthias Niessner, Manolis Savva, Shuran Song, Andy Zeng, and Yinda Zhang. Matterport3d: Learning from rgbd data in indoor environments. International Conference on 3D Vision (3DV), 2017. 1

[10] Bowen Cheng, Alex Schwing, and Alexander Kirillov. Perpixel classification is not all you need for semantic segmentation. Advances in Neural Information Processing Systems, 34:17864–17875, 2021. 2, 4

[11] Hsien-Tzu Cheng, Chun-Hung Chao, Jin-Dong Dong, Hao-Kai Wen, Tyng-Luh Liu, and Min Sun. Cube padding for weakly-supervised saliency prediction in 360 videos. In Pro-

ceedings ofthe IEEE conference on computer vision andpat tern recognition, pages 1420–1429, 2018. 2

[12] Benjamin Coors, Alexandru Paul Condurache, and Andreas Geiger. Spherenet: Learning spherical representations for detection and classification in omnidirectional images. In Proceedings ofthe European conference on computer vision (ECCV), pages 518–533, 2018. 1, 2

[13] Michael Defferrard, Martino Milani, Fr¨ ed´ erick Gusset, and´ Nathanael Perraudin. DeepSphere: a graph-based spherica¨ CNN. In International Conference on Learning Representa tions, 2020. 2

[14] Aleksandar M Dimitrijevic, Martin Lambers, and Dejan´ Ranciˇ c. Comparison of spherical cube map projections used ´ in planet-sized terrain rendering. Facta Universitatis, Series: Mathematics and Informatics, 31(2):259–297, 2016. 1, 2

[15] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representa tions, 2021. 2, 4

[16] Marc Eder, Mykhailo Shvets, John Lim, and Jan-Michael Frahm. Tangent images for mitigating spherical distortion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12426–12434, 2020. 2

[17] Lance Flavell. Uv mapping. In Beginning Blender: Open Source 3D Modeling, Animation, and Game Design, pages 97–122. Springer, 2010. 1, 2

[18] Michael S Floater. Generalized barycentric coordinates and applications. Acta Numerica, 24:161–214, 2015. 4

[19] Krzysztof M Gorski, E Hivon, and BD Wandelt. Analysis´ issues for large cmb data sets. Proceedings: Evolution of Large Scale Structure–Garching, 1998. 3

[20] Suresh Guttikonda and Jason Rambach. Single frame semantic segmentation using multi-modal spherical images. In Proceedings of the IEEE/CVF Winter Conference on Appli cations of Computer Vision, pages 3222–3231, 2024. 1, 2, 5, 6, 7

[21] Shi-Min Hu, Zheng-Ning Liu, Meng-Hao Guo, Jun-Xiong Cai, Jiahui Huang, Tai-Jiang Mu, and Ralph R Martin. Subdivision-based mesh convolution networks. ACM Trans actions on Graphics (TOG), 41(3):1–16, 2022. 2

[22] Chiyu Jiang, Jingwei Huang, Karthik Kashinath, Philip Marcus, Matthias Niessner, et al. Spherical cnns on unstructured grids. International Conference on Learning Representa tions, 2019. 2

[23] Hualie Jiang, Zhe Sheng, Siyu Zhu, Zilong Dong, and Rui Huang. Unifuse: Unidirectional fusion for 360 panorama depth estimation. IEEE Robotics and Automation Letters, 6 (2):1519–1526, 2021. 1

[24] Benjamin Keinert, Matthias Innmann, Michael Sanger, and¨ Marc Stamminger. Spherical fibonacci mapping. ACM Transactions on Graphics (TOG), 34(6):1–7, 2015. 1, 3

[25] Iro Laina, Christian Rupprecht, Vasileios Belagiannis, Federico Tombari, and Nassir Navab. Deeper depth prediction

with fully convolutional residual networks. In 2016 Fourth international conference on 3D vision (3DV), pages 239– 248. IEEE, 2016. 6

[26] Martin Lambers. Survey of cube mapping methods in interactive computer graphics. The Visual Computer, 36(5): 1043–1051, 2020. 1

[27] Yeonkun Lee, Jaeseok Jeong, Jongseob Yun, Wonjune Cho, and Kuk-Jin Yoon. Spherephd: Applying cnns on a spherical polyhedron representation of 360deg images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9181–9189, 2019. 1, 2

[28] Yuyan Li, Yuliang Guo, Zhixin Yan, Xinyu Huang, Ye Duan, and Liu Ren. Omnifusion: 360 monocular depth estimation via geometry-aware fusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2801–2810, 2022. 1, 2

[29] Thang Luong, Hieu Pham, and Christopher D. Manning. Effective approaches to attention-based neural machine translation. In Proceedings ofthe 2015 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, 2015. 2

[30] Ronald Miller. An equi-rectangular map projection. Geography, pages 196–201, 1949. 1

[31] Hongbin Pei, Bingzhe Wei, Kevin Chen-Chuan Chang, Yu Lei, and Bo Yang. Geom-gcn: Geometric graph convolutional networks. International Conference on Learning Representations, 2020. 2

[32] Giovanni Pintore, Marco Agus, Eva Almansa, Jens Schneider, and Enrico Gobbetti. Slicenet: deep dense depth estimation from a single indoor panorama using a slice-based representation. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11536– 11545, 2021. 1

[33] Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. Improving language understanding by generative pre-training. 2018. 4

[34] Rene Ranftl, Alexey Bochkovskiy, and Vladlen Koltun. Vi-´ sion transformers for dense prediction. In Proceedings of the IEEE/CVF international conference on computer vision, pages 12179–12188, 2021. 2, 4

[35] Manuel Rey-Area, Mingze Yuan, and Christian Richardt. 360monodepth: High-resolution 360deg monocular depth estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3762– 3772, 2022. 1, 2

[36] Zhijie Shen, Chunyu Lin, Kang Liao, Lang Nie, Zishuo Zheng, and Yao Zhao. Panoformer: Panorama transformer for indoor 360 depth estimation. In European Conference on Computer Vision, pages 195–211. Springer, 2022. 1, 2, 5, 6, 7

[37] Yu-Chuan Su and Kristen Grauman. Learning spherical convolution for fast features from 360 imagery. Advances in Neural Information Processing Systems, 30, 2017. 2

[38] Cheng Sun, Min Sun, and Hwann-Tzong Chen. Hohonet: 360 indoor holistic understanding with latent horizontal features. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2573–2582, 2021. 1

[39] Keisuke Tateno, Nassir Navab, and Federico Tombari. Distortion-aware convolutional filters for dense prediction in panoramic images. In Proceedings ofthe European Confer ence on Computer Vision (ECCV), pages 707–722, 2018. 2

[40] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017. 2, 4

[41] Petar Velickoviˇ c, Guillem Cucurull, Arantxa Casanova,´ Adriana Romero, Pietro Lio, and Yoshua Bengio. Graph at-\` tention networks. In International Conference on Learning Representations, 2018. 2

[42] Fu-En Wang, Yu-Hsuan Yeh, Min Sun, Wei-Chen Chiu, and Yi-Hsuan Tsai. Bifuse: Monocular 360 depth estimation via bi-projection fusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 462–471, 2020. 1, 2

[43] Fu-En Wang, Yu-Hsuan Yeh, Yi-Hsuan Tsai, Wei-Chen Chiu, and Min Sun. Bifuse++: Self-supervised and efficient bi-projection fusion for 360 depth estimation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(5): 5448–5460, 2022. 1, 2

[44] Zhendong Wang, Xiaodong Cun, Jianmin Bao, Wengang Zhou, Jianzhuang Liu, and Houqiang Li. Uformer: A general u-shaped transformer for image restoration. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 17683–17693, 2022. 2, 3, 4

[45] Enze Xie, Wenhai Wang, Zhiding Yu, Anima Anandkumar, Jose M Alvarez, and Ping Luo. Segformer: Simple and efficient design for semantic segmentation with transformers. Advances in Neural Information Processing Systems, 34:12077–12090, 2021. 2, 4

[46] Qingsong Yan, Qiang Wang, Kaiyong Zhao, Bo Li, Xiaoweo Chu, and Fei Deng. Spheredepth: Panorama depth estima tion from spherical domain. In 2022 International Conference on 3D Vision (3DV), pages 1–10. IEEE, 2022. 1, 2

[47] Ilwi Yun, Hyuk-Jae Lee, and Chae Eun Rhee. Improving 360 monocular depth estimation via non-local dense prediction transformer and joint supervised and self-supervised learning. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 3224–3233, 2022. 1, 2

[48] Ilwi Yun, Chanyong Shin, Hyunku Lee, Hyuk-Jae Lee, and Chae Eun Rhee. Egformer: Equirectangular geometrybiased transformer for 360 depth estimation. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV), pages 6101–6112, 2023. 1, 2, 5, 6, 7

[49] Chao Zhang, Stephan Liwicki, William Smith, and Roberto Cipolla. Orientation-aware semantic segmentation on icosahedron spheres. In Proceedings of the IEEE/CVF Interna tional Conference on Computer Vision, pages 3533–3541, 2019. 1, 5, 6, 7

[50] Chao Zhang, Sen He, and Stephan Liwicki. A spherical ap proach to planar semantic segmentation. In BMVC, 2020. 1

[51] Jiaming Zhang, Kailun Yang, Chaoxiang Ma, Simon Reiß, Kunyu Peng, and Rainer Stiefelhagen. Bending reality: Distortion-aware transformers for adapting to panoramic se

mantic segmentation. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 16917–16927, 2022. 1

[52] Si Zhang, Hanghang Tong, Jiejun Xu, and Ross Maciejewski. Graph convolutional networks: a comprehensive review. Computational Social Networks, 6(1):1–23, 2019. 2

[53] Jia Zheng, Junfei Zhang, Jing Li, Rui Tang, Shenghua Gao, and Zihan Zhou. Structured3d: A large photo-realistic dataset for structured 3d modeling. In Proceedings of The European Conference on Computer Vision (ECCV), 2020. 1, 5

[54] Nikolaos Zioulis, Antonis Karakottas, Dimitrios Zarpalas, and Petros Daras. Omnidepth: Dense depth estimation for indoors spherical panoramas. In Proceedings of the European Conference on Computer Vision (ECCV), pages 448– 465, 2018. 2