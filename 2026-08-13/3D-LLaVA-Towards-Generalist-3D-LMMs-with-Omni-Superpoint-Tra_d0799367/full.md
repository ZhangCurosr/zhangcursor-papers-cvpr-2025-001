# 3D-LLaVA: Towards Generalist 3D LMMs with Omni Superpoint Transformer

Jiajun Deng<sup>1</sup>, Tianyu He<sup>2</sup>, Li Jiang<sup>3</sup>, Tianyu Wang<sup>4</sup>, Feras Dayoub<sup>1</sup>, Ian Reid<sup>1,4</sup> <sup>1</sup> Australian Institute for Machine Learning, The University of Adelaide   
<sup>2</sup> Microsoft Research <sup>3</sup> The Chinese University of Hong Kong, Shenzhen <sup>4</sup> Mohamed bin Zayed University of AI

## Abstract

Current 3D Large Multimodal Models (3D LMMs) have shown tremendous potential in 3D-vision-based dialogue and reasoning. However, how tofurther enhance 3D LMMs to achieve fine-grained scene understanding and facilitate flexible human-agent interaction remains a challenging problem. In this work, we introduce 3D-LLaVA, a simple yet highly powerful 3D LMM designed to act as an intelligent assistant in comprehending, reasoning, and interacting with the 3D world. Unlike existing top-performing methods that rely on complicated pipelines—such as offline multi-view feature extraction or additional task-specific heads—3D-LLaVA adopts a minimalist design with integrated architecture and only takes point clouds as input. At the core of 3D-LLaVA is a new Omni Superpoint Transformer (OST), which integrates three functionalities: (1) a visual feature selector that converts and selects visual tokens, (2) a visual prompt encoder that embeds interactive visual prompts into the visual token space, and (3) a referring mask decoder that produces 3D masks based on text description. This versatile OST is empowered by the hybrid pretraining to obtain perception priors and leveraged as the visual connector that bridges the 3D data to the LLM. After performing unified instruction tuning, our 3D-LLaVA reports impressive results on various benchmarks. The code and model will be released at https: //github.com/djiajunustc/3D-LLaVA.

## 1. Introduction

Recent advancements in Large Language Models (LLMs) [4, 43, 49, 55, 66] have reshaped the paradigm of artificial intelligence, positioning language as a universal interface for general-purpose reasoning and interaction. Building on this progress, 2D Large Multi-modal Models (LMMs) [2, 18, 39, 40, 42] have emerged, integrating images and texts to support a wide range of vision-language tasks. In a further step to extend these capabilities to 3D, 3D LMMs [11, 23] have huge potential to unlock a series of real-world applications, such as autonomous vehicles, household robots, and augmented reality, where robust reasoning, precise 3D scene comprehension, and seamless human-agent interaction are of great significance.

![](images/ee878c65f10b1ab5d0317be913c8c390d55c41193a23d777ab4eb90bb558b556.jpg)  
Figure 1. An intuitive comparison between 3D-LLaVA and other SoTA 3D LMMs (The performance of LEO on ScanQA is omitted here since its setting is different). Our 3D-LLaVA achieves the best results among the competitors on most of the benchmarks.

It is a non-trivial problem to empower 3D LMMs with these desired properties. Despite notable progress achieved with the 3D vision and language community towards 3D dialogue and reasoning, these 3D LMMs still rely on extra prompt encoder [40] or offline region proposal generation and feature extraction [23, 25] to enable interacting with both visual and textual prompts. Such extra modules and offline preprocessing result in a complex pipeline, which complicates deployment and limits accessibility.

Furthermore, an effective 3D vision and language assistant should extend beyond simply generating text output; it should also be capable of grounding open-ended language expressions within a 3D scene and accurately segmenting the corresponding 3D masks. However, current referringbased 3D point cloud segmentation methods typically align and fuse text embeddings into a specialized segmentation model without LLMs. An exception is SegPoint [21], which utilizes the reasoning capabilities of LLMs to improve referring 3D point segmentation. Nonetheless, it still depends on additional modules to achieve precise segmentation, and has not demonstrated its effectiveness in other 3D visionlanguage tasks such as VQA and captioning.

To overcome these limitations, we present 3D-LLaVA, a generalist 3D LMM that streamlines the pipeline while maintaining strong performance across diverse 3D tasks. In contrast to the prior works that assemble multiple models or extract features in an offline manner, 3D LLaVA bridges interactive 3D vision dialogue and point-level 3D scene comprehension in an integrated and shared architecture, eliminating the need for auxiliary modules and complicated steps. Particularly, as compared in Figure 1, the most distinguishing part of 3D-LLaVA is the novel visual connector, namely Omni Superpoint Transformer (OST). What makes it distinguishing includes how we use it as a shared module for multiple purposes and how we pretrain it together with the 3D scene encoder.

Specifically, existing 3D LMMs generally follow the trend of the 2D domain to leverage an MLP projector [40] or Q-Former [39] as the visual connector, both of which are single-function modules designed to transform vision features into token embeddings aligned with the language semantic space. On the contrary, OST is a versatile module built on superpoint representation that plays multiple roles in our 3D-LLaVA. Specifically, in addition to feature enhancement and projection, OST has the following functions: (1)Visual Feature Selector. OST selectively retains visual tokens, distinguishing between foreground and background superpoints. This helps highlight the informative part of the complex 3D scene and manage computational ovehead by reducing the number of tokens to be further processed by the LLM. (2) Visual Prompt Encoder. 3D-LLaVA does not involve an additional visual prompt encoder. When the user interacts with 3D-LLaVA with a visual prompt (such as a clicking point, a box, or a mask), OST plays the role of a visual prompt encoder, mapping the visual prompt to the same embedding space as the visual feature, which is then appended together with language token embeddings as the input of the LLM. (3) Mask Decoder. Instead of requiring an additional segmentation module for grounding language expressions onto 3D point clouds, OST directly generates 3D masks, keeping the model streamlined and self-contained.

Moreover, at the pretaining stage, OST is connected together with the 3D scene encoder and jointly pre-trained with the hybrid supervision of instance segmentation and 2D-to-3D knowledge distillation. Here, the 2D feature is extracted from multi-view images with the visual encoder of a 2D LMM, i.e. LLaVA-1.5 [42], and lifted to 3D by the geometric correspondence [46] between the point cloud and the pixels. Such a pretraining scheme on the one hand encompasses the perception prior to our model and takes the well-aligned 2D data as the bridge to facilitate the alignment between 3D visual embedding and language embedding.

We conduct end-to-end instruction tuning over various tasks and then benchmark our 3D-LLaVA on five popular 3D vision and language understanding datasets. As shown in Figure 1 Our method achieves the state-of-the-art performance on all of these datasets. Remarkably, we achieve 92.6% CiDEr on the competitive ScanQA dataset, improving the previous best result by absolutely 4.9% CiDEr score.

To summarize, we make three-fold contributions:

• We propose 3D-LLaVA, a generalist 3D LMM that unifies multiple tasks through the Omni Superpoint Transformer, streamlining the framework.

• We present a new perspective that a versatile visual connector can be leveraged to remove the task-specific modules added to the 3D LMM, making the model more elegant and integrated.

• We benchmark the proposed method on different datasets, demonstrating its great potential to be a powerful baseline in this field.

## 2. Related Work

3D Vision & Language Understanding. In recent years, there has been tremendous progress in understanding 3D scenes from natural language, where the language provides contextual knowledge and queries of user intentions to allow seamless interaction between humans and models. These works can be broadly categorized into four main tasks: 3D grounding [1, 7, 29, 60, 64] that localizes specific objects within the 3D scene according to the given textual queries, 3D referring segmentation [21, 27, 28, 48, 58] that predicts a point-wise mask for the described object; 3D captioning [9, 10, 13, 31, 32, 34, 61] that densely localizes objects in a 3D scene and describes them with natural language; 3D question answering [3, 44, 45] that answers given textual questions about the 3D scene.

Although achieving great success on certain tasks, the above methods fall short in generalizing across different 3D understanding tasks. Motivated by this, recent efforts have also been dedicated to designing pre-training schemes [33, 67] or unified models [6, 8] for various tasks like 3D grounding, captioning and question answering. Despite these models achieving impressive improvements in handling diverse 3D scene tasks, their reliance on taskspecific heads and limited reasoning capabilities constrain their flexibility for broader, general-purpose applications.

3D Large Multimodal Model. The huge success of Large Language Models (LLMs) [5, 15, 22, 55] has fueled the demand for a versatile interface that can handle various modalities beyond language. In response to this demand, Large Multimodal Models (LMMs) has been developed to comprehend instructions that span vision and language [2, 40, 42, 54, 63]. PointLLM [59] integrates the object-level point cloud into LLM by constructing a joint embedding space among 3D points and text, enabling explain the 3D shape with language. 3D-LLM [23] extends the 2D LMM into the 3D scene, improving the capability of 3D spatial reasoning by introducing positional embeddings and location tokens. LL3DA [11] develops a Q-Former [39] to bridge the 3D point cloud, visual prompt, and language instruction. Grounded 3D-LLM [12] involves referent tokens and employs contrastive learning to unify grounding with textual response generation. Segpoint [21] attempts to unify semantic segmentation and referring segmentation with an LLM. Agent3D-Zero [62] leverages the 2D LMM to first observe from the birds’ eye view and then selects the informative viewpoints for further zero-shot 3D scene understanding. Scene-LLM [20] lifts multi-view image features into 3D space, and follows the two-stage training scheme [40] to to perform 3D vision and language alignment. Chat-Scene [25] proposes to achieve precise object referencing and grounding by incorporating object identifiers into the 3D LMM and fusing the offline extracted 2D and 3D instance-level features.

![](images/1f604643aab32c0b215191884bcefc2691a6477f561e6ae921ff2250655a7723.jpg)  
Figure 2. An overview of 3D-LLaVA framework. Given input point cloud, language instruction, and optional visual prompt, 3D-LLaVA generates text output from LLM and produces 3D masks with Omni Superpoint Transformer (OST). The 3D feature out of the Sparse 3D U-Net is clustered into superpoint with Superpoint Pooling. Visual Sampler is a parameter-free module that samples point features corresponding to the visual prompt $X _ { P }$ . Omni Superpoint Transformer takes both superpoint feature and visual prompt feature as input, produces visual feature embedding $Z _ { V }$ and visual prompt embedding $Z _ { P }$ , followed by a projection layer $W _ { V }$ to obtain the token embedding $H _ { V }$ and $H _ { P }$ . Once the LLM outputs a special segmentation token, i.e., [SEG], the hidden state linked to [SEG] token will be sent to another projection layer $W _ { S }$ and then input as segmentation query to the frozen OST to generate segmentation masks.

## 3. Approach

The overall framework of 3D-LLaVA is illustrated in Figure 2. It is a generalist 3D LMM, capable of conducting 3D vision-centric dialogue, being interacting seamlessly with flexible visual and textual prompts, and grounding openended language description into 3D point cloud masks. In this section, we first introduce the model architecture of the 3D scene encoder (Section 3.1) and Omni Superpooint Transformer (Section 3.2). Then, in Section 3.3, we elaborate on the detail of each step in our pipeline. Finally, in Section 3.4, we introduce the training scheme.

## 3.1. 3D Scene Encoder

Given the point clouds input $\mathbf { X } _ { \mathbf { V } } \in \mathbb { R } ^ { N \times 6 }$ , where N is the number of points and the 6 channels represent the coordinates $\{ x , y , z \}$ and the color information $\{ r , g , b \}$ , we first convert the points into voxels based on their 3D coordinates. After obtaining the voxels, the Sparse 3D U-Net [16] is leveraged as the scene encoder to extract point cloud features. Sparse 3D U-Net is a U-Net-like architecture but consists of sparse convolutional layers. The output of the Sparse 3D U-Net has the same number as the input, resulting in an excessively large voxel count that is not feasible for the following steps. One option to reduce the number of points is to perform farthest point sampling [47]. However, the sampling operation inevitably causes information loss. In contrast, we follow [35, 36, 53] to implement the average pooling operation based on superpoints, which are generated with the bottom-up clustering algorithm [38]. The superpoint pooling reduces the quantity of 3D vision embeddings into hundreds or a few thousand, depending on the complexity of the 3D scene.

## 3.2. Omni Superpoint Transformer

The architecture of the proposed Omni Superpoint Transformer (OST) is shown in Figure 3 (a). Notably, the basic block of a conventional segmentation Transformer typically includes a cross-attention layer, a self-attention layer, and a feed-forward network. Here, the cross-attention layer is leveraged to abstract the information from the source feature to the object query. Although OST can perform segmentation, it is primarily composed without cross-attention layers. The superpoint features act as both queries and source features (key and value) in OST. This adjustment keeps the correspondence between the output embedding of OST and the lifted 2D feature, facilitating effective 2Dto-3D feature distillation during the pretraining phase. Additionally, to guide the superpoint queries towards relevant entities, we replace the standard self-attention layer with a distance-adaptive self-attention layer [41], which introduces a bias term based on the distances between superpoints. The pairwise attention between the i-th superpoint query and the j-th superpoint query is computed as:

$$
A t t n ( Q _ { i } , K _ { j } , V _ { j } ) = S o f t m a x ( \frac { Q _ { i } K _ { j } ^ { T } } { \sqrt { C } } - \sigma \cdot D ) V _ { j } ,\tag{1}
$$

where $Q , K , V$ is the query, key, and value of the attention module, C is the channel of the embedding, σ is a learnable parameter based on the query, and D indicates the Euler distance between the centroid of these two superpoints.

There are three heads on the top of OST: a mask head, a classification head, and an alignment head. The mask head transforms each query embedding into a mask prediction kernel, which is then applied to generate binary mask prediction by performing a dot product with the input superpoint features of OST [30, 52, 53]. The classification head predicts the category of the segmentation mask by outputting the logit of each category. The output of the alignment head is denoted as $Z _ { V }$ in Figure 2. It would be further leveraged to obtain the visual token of the LLM.

## 3.3. Details in Pipeline

Visual Feature Selection. Although superpoint pooling has reduced the query number of OST, it still results in a very long sequence if directly applied as input visual tokens of the LLM. To alleviate this issue, after obtaining $Z _ { V }$ from OST, we only keep the superpoints with the top-K objectness scores. The objections score of each superpoint query is defined as the highest score among foreground categories. Visual Prompt Encoding. A generalist 3D LMM is supposed to be interacted with both language instructions and visual prompts. Common visual prompts include a clicking point, bounding box, or binary mask. A straightforward approach to encode these prompts is to use a prompt encoder composed of several linear layers [11], designed to project the prompt (e.g., coordinates of clicking points or bounding boxes) into an embedding that aligns with the same semantic space as visual or language tokens. However, we find that this type of prompt encoder is challenging to optimize, as it lacks explicit information indicating which areas are targeted by the prompt.

![](images/db54f6495450202af3fcfeeda0a1fbcfd20b89d221f6c5be1ac716a3a5ba6ca2.jpg)  
Figure 3. An illustration of (a) the architecture of Omni Superpoint Transformer, and (b) the paradigm of the visual sampler.

In contrast, as shown in Figure 3 (b), 3D-LLaVA introduces a parameter-free visual sampler to encode the visual prompt $X _ { P }$ and reuses OST as a visual prompt encoder to generate the corresponding visual prompt embedding $Z _ { P } ,$ ensuring that the prompt is embedded in the same space as the visual features.

For a clicking point prompt, the visual sampler obtains the prompt feature through three nearest-neighbor (three-NN) interpolation [47], which first finds the three nearest points and computes the prompt feature using weighted interpolation. If the prompt is a box or mask, the visual sampler groups the points within the prompt and applies average pooling to generate the prompt feature. This prompt feature is then appended to the superpoint features and is input to OST. Here, we leverage the masked attention strategy between the superpoint feature and prompt feature. Specifically, we set the attention mask from the superpoints to the prompt as negative infinity. This prevents the prompt feature from influencing the superpoints, allowing it only to sample the relevant visual information. Similar to the visual feature embedding $Z _ { V }$ , the prompt embedding $Z _ { P }$ is out of the alignment head.

Projection. After obtaining the Top-K superpoint-based visual feature embedding $Z _ { v }$ and the visual prompt embedding $Z _ { P }$ , we apply a projection layer $W _ { V }$ to transform them into the language embedding tokens $H _ { V }$ and $H _ { P }$ . The projection layer consists of two linear layers and a GELU activation layer between the linear layers.

Instruction Formulation. We present the typical language instruction in the bottom-right part of Figure 2. There are two kinds of place holders in the instruction: “<PC>” and “<Visual Prompt>”. The text instruction except for the placeholder will be tokenized into the text token embedding $H _ { T }$ . After tokenization, we replace “<PC>” with visual token embedding $H _ { V }$ , and replace “<Visual Prompt>” with the prompt token embedding $H _ { P }$

Mask Decoding. When the instruction prompts 3D-LLaVA to perform referring segmentation [1, 7, 37], the LLM will output a [SEG] token in its text response. Once detecting this token, we extract the last hidden state of the token before the [SEG] token. This hidden state, $H _ { S } ,$ , is then fed into the projection layer $W _ { S }$ to generate the segmentation query.

In our method, we leverage the frozen OST to predict the segmentation mask of the referred object. Similar to the paradigm of using OST as the visual prompt encoder, the segmentation query is concatenated with the superpoint query to formulate the input of OST. We apply a mask attention strategy to prevent information flow from the segmentation query to the superpoints. Since the segmentation query lacks coordinate information, the bias term (from Equation 1) between this query and the superpoint queries is set to zero. The output kernel from the mask head that corresponds to the segmentation query is applied to the superpoint feature input to generate the mask prediction.

## 3.4. Training Scheme

## Stage 1: Pre-training 3D Scene Encoder and OST.

Unlike the 2D domain, which has powerful and widely recognized vision foundation models such as CLIP [50], there is currently no such 3D foundation model that can serve as a readily usable 3D visual encoder.

To this end, we pre-train the Sparse 3D U-Net and the OST by ourselves. Specifically, we adopt hybrid supervision that combines the vision-centric task, i.e., instance segmentation, and the 2D-to-3D knowledge distillation:

$$
L _ { P r e } = L _ { C l s } + L _ { M a s k } + L _ { K D } ,\tag{2}
$$

where $L _ { C l s }$ represents cross-entropy loss for multi-category classification, $L _ { M a s k }$ includes the binary cross-entropy loss and Dice loss for mask prediction, and $L _ { K D }$ denotes the knowledge distillation loss, which includes mean squared error and cosine similarity losses.

For instance segmentation, we leverage the annotation from ScanNet200 [51] as the training data. For 2D-to-3D knowledge distillation, we follow OpenScene [46] that first extracts multi-view 2D features and then lifts the 2D features into 3D points by the correspondence between 3D point clouds and 2D pixels. The lifted 2D features are pooled into each superpoint to generate the target feature. Here we leverage the visual encoder of LLaVA-1.5-7B [42], i.e., CLIP-ViT-L, to extract the teacher 2D feature.

Table 1. Dataset statistics for joint instruction tuning.
<table><tr><td>Dataset</td><td>Task</td><td>Size</td></tr><tr><td>ScanRefer Nr3D</td><td>referring segmentation referring segmentation</td><td>37K 29K</td></tr><tr><td>Multi3DRefer</td><td>referring segmentation</td><td>44K</td></tr><tr><td>ScanQA</td><td>visual question answering</td><td>30K</td></tr><tr><td>SQA3D</td><td>visual question answering</td><td>89K</td></tr><tr><td>Scan2Cap</td><td>dense captioning</td><td>37K</td></tr><tr><td>Nr3D*</td><td>dense captioning</td><td>29K</td></tr><tr><td>Total</td><td>-</td><td>295K</td></tr></table>

## Stage 2: End-to-End Instruction Tuning.

We combine various 3D vision and language understanding datasets to form our instruction-tuning data. The combined datasets include ScanRefer [7], Nr3D [1], Multi3DRefer [64], ScanQA [3], SQA3D [44], Scan2Cap [13]. The statistic of the utilized dataset is presented in Table 1. To enrich the language annotation that involves describing the object in the 3D scene, we follow [25] to use Nr3D as the complementary to the dense captioning task, which is denoted as “Nr3D\*” in this table.

The instruction tuning phase jointly optimizes 3D-LLaVA for both text generation and referring segmentation. The training objective is composed as follows:

$$
L _ { I F T } = L _ { t e x t } + 0 . 1 \times L _ { m a s k } ,\tag{3}
$$

where $L _ { t e x t }$ is the cross-entropy loss for next-token generation, $L _ { m a s k }$ represents the mask prediction loss that also consists of the binary cross-entropy loss and the Dice loss, which is the same as the pertaining stage. Here, we multiply the mask loss by a coefficient of 0.1 for balance. We always keep the Sparse 3D U-Net, OST, and the main body of LLM frozen. Only the visual projector, the SEG project, and LoRA [24] parameters adopted to the LLM are updated.

## 4. Experiments

## 4.1. Datasets and Metrics

Datasets. In this work, we conduct experiments on the 3D scans provided by ScanNet dataset [17], including 1,201 scenes for training and 312 for validation. At the pertaining stage of our 3D encoder, we leverage the mask annotation from ScanNet200 [51], which extends the original ScanNet with fine-grained categories. The language annotation leveraged in the instruction tuning has been introduced in Section 4. After instruction tuning, we validate the effectiveness of the proposed 3D-LLaVA on the following datasets: ScanQA [3] and SQA3D [44] for question answering, ScanRefer [7] and Multi3DRefer [64] for referring segmentation and Scan2Cap [13] for dense captioning.

Metrics. We follow the common practice to evaluate the quality of generated text response for ScanQA and

Table 2. Performance comparison among state-of-the-art methods. “Specialist Model” means this model can be utilized to perform 3D question answering, 3D dense captioning, or referring segmentation. “Finetuned 3D LMM” indicates the model is jointly trained and then finetuned on each dataset before evaluation. We add $\mathrm { ~ a ~ } ^ { 6 * }$ to 3D LMMs that belong to this kind. “3D LMM” includes the models that are only been trained on multiple tasks. $\mathbf { \ddot { P } C } ^ { \mathbf { \vec { \mu } } }$ means point cloud and $^ { 6 6 } \mathrm { T } ^ { \flat }$ means multi-view images. Please note that LEO [26]’s results on ScanQA is marked with a gray color and not compared to other methods, since it is in a different setting that accesses the ground truth object related to the question. The top-2 entities of each metric are marked with underline and the best one is highlighted by bolding font.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Modality</td><td rowspan="2">ScanRefer (val) mIoU↑</td><td rowspan="2">Multi3DRefer (val) mIoU↑</td><td rowspan="2">B-4↑</td><td colspan="3">ScanQA (val) M↑</td><td colspan="2">SQA3D (test) EM↑</td><td rowspan="2">C@0.5↑</td><td colspan="3">Scan2Cap (val)</td></tr><tr><td>C↑</td><td></td><td>R↑</td><td>EM-R↑</td><td></td><td>B-4@0.5↑</td><td>M@0.5↑</td><td>R@0.5↑</td></tr><tr><td>Specialist Models:</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ScanQA[3]</td><td>PC</td><td></td><td></td><td>64.9</td><td>10.1</td><td>13.1</td><td>33.3</td><td>46.6</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>3D-VLP[33]</td><td>PC</td><td></td><td></td><td>67.0</td><td>11.2</td><td>13.5</td><td>34.5</td><td></td><td></td><td>54.9</td><td>32.3</td><td>24.8</td><td>51.5</td></tr><tr><td>3D-VisTA[67]</td><td>PC</td><td></td><td></td><td>69.6</td><td>10.4</td><td>13.9</td><td>45.7</td><td>48.5</td><td></td><td>61.6</td><td>34.1</td><td>26.8</td><td>55.0</td></tr><tr><td>Scan2Cap[13]</td><td>PC</td><td></td><td></td><td></td><td></td><td></td><td></td><td>41.0</td><td></td><td>39.1</td><td>23.3</td><td>22.0</td><td>44.8</td></tr><tr><td>MORE[31]</td><td>PC</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>40.9</td><td>22.9</td><td>21.7</td><td>44.4</td></tr><tr><td>SpaCap3D[57]</td><td>PC</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>44.0</td><td>25.3</td><td>22.3</td><td>45.4</td></tr><tr><td>D3Net[8]</td><td>PC</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>46.1</td><td>30.3</td><td>24.4</td><td>51.7</td></tr><tr><td>UniT3D[14]</td><td>PC</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>46.7</td><td>27.2</td><td>21.9</td><td>46.0</td></tr><tr><td>3DJCG[6]</td><td>PC</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>49.5</td><td>31.0</td><td>24.2</td><td>50.8</td></tr><tr><td>Vote2Cap-DETR [9]</td><td>PC</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>61.8</td><td>34.5</td><td>26.2</td><td>54.4</td></tr><tr><td>TGNN [28]</td><td>PC</td><td>27.8</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>M3DRef-CLIP [64]</td><td>PC</td><td>35.7</td><td>32.6</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>X-RefSeg3D [48]</td><td>PC</td><td>29.9</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>3D-STMN [58]</td><td>PC</td><td>39.5</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Finetuned 3D LMMs:</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>3D-LLM[23]</td><td>PC+I</td><td></td><td></td><td>69.4</td><td>12.0</td><td>14.5</td><td>35.7</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Scene-LLM [20]*</td><td>PC+I</td><td></td><td></td><td>80.0</td><td>12.0</td><td>16.8</td><td>40.0</td><td>54.2</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LL3DA* [11]</td><td>PC</td><td></td><td></td><td>76.8</td><td>13.5</td><td>15.9</td><td>37.3</td><td></td><td></td><td>65.2</td><td>36.8</td><td>26.0</td><td>55.1</td></tr><tr><td>SegPoint [21]*</td><td>PC</td><td>41.7</td><td>36.1</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>3D LMMs:</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LEO [26]</td><td>PC+I</td><td></td><td></td><td>101.4</td><td>13.2</td><td>20.0</td><td>49.2</td><td>50.0</td><td>52.4</td><td>72.4</td><td>38.2</td><td>27.9</td><td>58.1</td></tr><tr><td>Scene-LLM [20]</td><td>PC+I</td><td></td><td></td><td>80.0</td><td>11.7</td><td>15.8</td><td>35.9</td><td>53.6</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Chat-Scene [25]</td><td>PC+I</td><td></td><td></td><td>87.7</td><td>14.3</td><td>18.0</td><td>41.6</td><td>54.6</td><td>57.5</td><td>77.2</td><td>36.4</td><td>28.0</td><td>58.1</td></tr><tr><td>Grounded 3D-LLM [12]</td><td>PC</td><td></td><td></td><td>72.7</td><td>13.4</td><td></td><td></td><td></td><td></td><td>70.6</td><td>35.5</td><td></td><td></td></tr><tr><td>3D-LLaVA (ours)</td><td>PC</td><td>43.3</td><td>42.7</td><td>92.6</td><td>17.1</td><td>18.4</td><td>43.1</td><td>54.5</td><td>56.6</td><td>78.8</td><td>36.9</td><td>27.1</td><td>57.7</td></tr></table>

Scan2Cap in terms of CiDEr (C) BLEU-4 (B-4), METEOR (M) and Rouge-L (R). Different from the conventional setting of ScanQA, there is a definite answer to situated question answering dataset SQA3D, therefore we leverage extract match accuracy (EM) as well as the refined version (EM-R) as the metric. For referring segmentation, we adopt the mean intersection over union (mIoU) for evaluation.

## 4.2. Implementation Details

We pre-train our 3D visual encoder on ScanNet200 for 512 epochs under the hybrid supervision of 2D-to-3D knowledge distillation and segmentation. After obtaining the 3D visual encoder, we developed our 3D-LLaVA based on the LLaVA-1.5-7B [42]. We make use of model weights of the visual projector and LLM (Vicuna-1.5-7B [43]) from the LLaVA-1.5-7B, and connect the alignment embedding out of our 3D visual encoder to the visual projector. We keep 100 superpoint features $Z _ { V }$ according to their objectness score, which are then been projected to the visual token embeddings $H _ { V }$ The instruction tuning is conducted on 8× NVIDIA RTX 3090 GPUs with the acceleration of the DeepSpeed toolkit. We adopt LoRA [24] to the LLM and keep the main body of LLM and visual encoder frozen during training. The data presented in Table 1 are leveraged to perform end-to-end training for 1 epoch. We set the batch size to 2 for each GPU and update the model weights after accumulating the gradient every 8 steps. The model is optimized with the AdamW. The Cosine Annealing schedule is leveraged to update the learning rate, with the initial learning rate set as 2e-4.

## 4.3. Comparison with SoTA Models

We compare the proposed 3D-LLaVA with other models and present the results in Table 2. The models compared in this table are divided into three groups: specialist models, Finetuned 3D LMMs, and 3D LMMs. The specialist model is designed to address a single kind of task. All of the specialist models in this table are without LLMs. The Finetuned 3D LMM is the 3D large multimodal model that is finetuned on each dataset. Such fine-tuning could improve the performance of the model on the corresponding dataset, but will affect its generalizability. The last kind, 3D LMM, is the large multimodal model that is trained on a unified dataset including various tasks. Particularly, among all the competitors, our 3D LLaVA is the only one that covers the typical text generation task (i.e., 3D dense captioning, and 3D vision question answering) and point-level understanding (i.e., 3D Referring Segmentation).

3D Referring Segmentation requires the model to output the 3D mask on the point cloud according to the users’ language expression, which validates the capability of grounding the text description on the 3D scene. We benchmark our methods and other state-of-the-art methods on both the single-target setting (Scanerfer [7]) and the various number setting (Multi3DRefer [64]).The referring text from Multi3DRefer can correspond to one, many, or even zero objects. If multiple objects are referred to in the instruction, we follow [21] to merge the masks into a single one for evaluation. When there is no corresponding to the referring expression, our 3D-LLaVA will output “Sorry, I cannot find this object.”. In this case, since there is no [SEG] token in the response, the mask decoding pipeline will not be applied, and thus all of the predicted masks will be assigned as background. As shown in the table, our 3D-LLaVA reports the best result among the competitors. Notably, our model achieves 43.3% mIoU on ScanRefer and 42.7% mIoU on Multi3DRefer, improving the previous best record of Seg-Point by absolutely 1.6% mIoU and 6.6% mIoU.

![](images/a8d8f9f74868bad718095fa3768f9d17fc708775e097bfcc86bb089129c8eef8.jpg)  
Figure 4. Different paradigms to produce visual prompt embedding. “OST”: Omni Superpoint Transformer. “P.E. Encoder”: Parameter-Free Encoder.

3D Question Answering is the task that asks the model to observe the visual information of the 3D scene and give a precise response to the user’s question involving some part of the scene. We conduct the comparison between our 3D-LLaVA and other methods on both the conventional 3D question-answering dataset ScanQA [3] and the situated question-answering dataset SQA3D [44]. As shown in Table 2, our method ranks the best for CiDEr, BLEU-4, and METEOR among the methods without accessing groundtruth information of the object relevant to the question. Remarkably, compared to Grounded 3D-LLM [12] which only uses point cloud as input, our 3D-LLaVA achieves a 19.9% improvement. When compared to the strongest competitor Chat-Scene, our 3D-LLaVA achieves 4.9% CiDEr, 2.8% BLEU-4, 0.4% METEOR, and 1.4% Rouge-L improvements on ScanQA, respectively. On SQA3D, our 3D-LLaVA reports comparable extract match accuracy as that of Chat-Scene (54.5% V.S. 54.6%). It is worth noting that Chat-Scene uses both instance-level 3D and 2D features, which rely on complicated offline preprocessing, while our 3D-LLaVA extracts superpoint features online with the OST, which is more computation-friendly.

3D Dense Captioning demands the model to describe the object and its spatial relationship to the surrounding instances within the scene. In this experiment, we follow the common practice of using the predicted mask proposals of Mask3D [52] as the visual prompt. Please note that we have not got access to the output of Mask3D in the training stage. Our OST works as a visual sampler to convert any prompts in the predefined formulation to the semantic space of visual features without the extra cost of finetuning. Results in the table show that our 3D-LLaVA also achieves the best performance in generating instance-level descriptions. This experiment further validates the effectiveness and scalability of the proposed 3D-LLaVA with OST.

Table 3. Performance comparisons for box-level 3D visual grounding. Our results are obtained by directly converting the foreground referring mask to a box, highlighted with light blue.
<table><tr><td>Methods</td><td colspan="2">ScanRefer (Box-Level) Acc@0.25 Acc@0.5</td></tr><tr><td>Specialist Models:</td><td></td><td></td></tr><tr><td>3D-VisTA [67]</td><td>50.6</td><td>45.8</td></tr><tr><td>ConcretNet [56]</td><td>50.6</td><td>46.5</td></tr><tr><td>3D LMMs:</td><td></td><td></td></tr><tr><td>3D-LLM [23]</td><td>30.3</td><td></td></tr><tr><td>Grounded 3D-LLM [12]</td><td>48.6</td><td>44.0</td></tr><tr><td>Chat-Scene [25]</td><td>55.5</td><td>50.2</td></tr><tr><td>3D-LLaVA (Ours)</td><td>51.2</td><td>40.6</td></tr></table>

Table 4. Ablation study on the paradigm to produce visual prompt embedding. The models are compared in terms of CiDEr, BLEU-4, METEOR and Rough-L on Scan2Cap [13]. The index (a), (b), and (c) in this table correspond to paradigms depicted in Figure 4. Our default setting is highlighted with light blue.
<table><tr><td rowspan="2">Visual Prompt Encoding</td><td colspan="3">Scan2Cap</td></tr><tr><td>C↑ B-4↑</td><td>M↑</td><td>R↑</td></tr><tr><td>(a) Coordinate Projection</td><td>68.7 33.9</td><td>26.7</td><td>55.1</td></tr><tr><td>(b) Pooling</td><td>76.8 36.6</td><td>26.9</td><td>57.5</td></tr><tr><td>(c) Ours with OST</td><td>78.8</td><td>36.9 27.1</td><td>57.7</td></tr></table>

## 4.4. Experimental Analysis

This section presents the experimental analysis of our 3D-LLaVA. Unless specified, the model evaluated in this section is trained with the same data and training scheme as the default setting introduced in the former sections.

Box-level 3D Visual Grounding. Even not designed for 3D referring segmentation, our 3D-LLaVA can also produce box-level grounding results. Specifically, we first apply DBSCAN algorithm [19] to the foreground mask to remove outliers, and then obtain the grounding box by considering the minimum and maximum coordinates of the mask. We compare the box-level grounding performance of our 3D-LLaVA with bot the specialist model and other 3D LMMs in Table 3. Although our model is optimized for precise binary masks, whereas competitors are trained to select best-matching proposals based on box IoU, our 3D-LLaVA achieves 51.2% accuracy when the IoU threshold is 0.25, better than most of the competitors in the table. Our performance lags behind Chat-Scene [25], but our method relies on neither an extra mask proposal generator nor the fusion of image and point cloud features.

Table 5. Quantitative comparison on the different number of visual tokens. The models are compared in terms of CiDEr and BLEU-4 on ScanQA [3] and Scan2Cap [13]. Our default setting is highlighted with light blue.
<table><tr><td></td><td colspan="2">ScanQA</td><td colspan="2">Scan2Cap</td></tr><tr><td># Visual Token</td><td>C↑</td><td>B-4↑</td><td>C↑</td><td>B-4↑</td></tr><tr><td>50</td><td>91.1</td><td>15.9</td><td>74.9</td><td>35.9</td></tr><tr><td>100</td><td>92.6</td><td>17.1</td><td>78.8</td><td>36.9</td></tr><tr><td>200</td><td>92.8</td><td>17.1</td><td>78.6</td><td>37.2</td></tr><tr><td>400</td><td>92.3</td><td>16.9</td><td>77.7</td><td>36.8</td></tr></table>

Effect of Visual Prompt Encoding. In this study, we analyze the effect of different ways to convert visual prompts into prompt embeddings (as illustrated in Figure 4). We leverage the box as the visual prompt in this experiment since the mask can be converted to a box by its boundary and the clicking point is a special case of a box without area. Among the compared paradigms, our proposed strategy to reuse OST as the visual prompt encoder, i.e. method (c), achieves the best result. On the one hand, our method avoids additional learnable parameters, which are difficult to optimize together with the LLM. On the other hand, compared to (b), appending the prompt query out of the parameter-free encoder to the superpoint queries enables deeply abstracting the superpoint features by the stack of OST encoder layers. The method (a) produces prompt embedding by applying an MLP to the box coordinates. This is because the produced prompt embedding lacks visual context, increasing the burden on the LLM in locating the corresponding region. We suppose this kind of paradigm needs more training data and training epochs to converge.

Effect of Visual Token Number. Retaining more visual tokens leads to a rapid increase in the computation complexity of LLMs. We take this experiment to explore how many visual tokens should be exploited in our 3D-LLaVA to enable an accurate understanding of the 3D scene. As shown in Table 5, increasing the token number from 50 to 100, the CiDEr on ScanQA and Scan2Cap is improved by 1.5% and 3.9%, respectively. However, further increasing the token count to 200 yields no substantial performance gains. Therefore, we set the default token number to 100.

## 4.5. Qualitative Results

In Figure 5, we showcase several visualizations of 3D-LLaVA’s performance across various 3D environments, including bedrooms, offices, and bathrooms. Our 3D-LLaVA model accurately interprets user instructions and demonstrates an ability to avoid false positives when the target object is absent from the 3D scene.

![](images/172f663efe014b139d41a7ab6c9597657548b4f1fce660ee882d5f5f8b8e5314.jpg)  
Figure 5. Visualization of 3D-LLaVA’s response on various tasks. Each of these examples includes an instruction to perform referring segmentation. Besides, the examples present the result of 3D question answering [3], 3D dense captioning [65], and situated question answering [44], respectively. When the referred object is not in the given 3D scene, the model is aware of responding with “Sorry, I cannot find this object”.

## 5. Conclusion

In this work, we introduce 3D-LLaVA, a new 3D LMM with streamlined architecture and powerful capability. The core component in 3D-LLaVA is a new visual connector, Omni Superpoint Transformer (OST), which serves as a multifunctional module in visual token selection, visual prompt encoding, and mask decoding. Therefore, taking advantage of the versatile OST, 3D-LLaVA is capable of conducting 3D vision-centric dialogue, enabling flexible interaction and grounding language expression into 3D point cloud masks with a universal architecture. Through extensive experiments, 3D-LLaVA achieves impressive results across multiple benchmarks. Although 3D-LLaVA has made significant improvements over the previous methods, 3D data is still the main obstacle in developing 3D LMMs. We regard the data collection and configuration as the next step.

## References

[1] Panos Achlioptas, Ahmed Abdelreheem, Fei Xia, Mohamed Elhoseiny, and Leonidas Guibas. Referit3d: Neural listeners for fine-grained 3d object identification in real-world scenes. In European Conference on Computer Vision, pages 422– 440. Springer, 2020. 2, 5

[2] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. Advances in neural information processing systems, 35:23716–23736, 2022. 1, 2

[3] Daichi Azuma, Taiki Miyanishi, Shuhei Kurita, and Motoaki Kawanabe. Scanqa: 3d question answering for spatial scene understanding. In proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 19129– 19139, 2022. 2, 5, 6, 7, 8

[4] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020. 1

[5] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in Neural Information Processing Systems (NeurIPS), 33:1877–1901, 2020. 2

[6] Daigang Cai, Lichen Zhao, Jing Zhang, Lu Sheng, and Dong Xu. 3djcg: A unified framework for joint dense captioning and visual grounding on 3d point clouds. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16464–16473, 2022. 2, 6

[7] Dave Zhenyu Chen, Angel X Chang, and Matthias Nießner. Scanrefer: 3d object localization in rgb-d scans using natural language. In European Conference on Computer Vision, pages 202–221. Springer, 2020. 2, 5, 7

[8] Dave Zhenyu Chen, Qirui Wu, Matthias Nießner, and Angel X Chang. D3net: A speaker-listener architecture for semi-supervised dense captioning and visual grounding in rgb-d scans. arXiv preprint arXiv:2112.01551, 2021. 2, 6

[9] Sijin Chen, Hongyuan Zhu, Xin Chen, Yinjie Lei, Gang Yu, and Tao Chen. End-to-end 3d dense captioning with vote2cap-detr. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11124– 11133, 2023. 2, 6

[10] Sijin Chen, Hongyuan Zhu, Mingsheng Li, Xin Chen, Peng Guo, Yinjie Lei, Gang Yu, Taihao Li, and Tao Chen. Vote2cap-detr++: Decoupling localization and describing for end-to-end 3d dense captioning. arXiv preprint arXiv:2309.02999, 2023. 2

[11] Sijin Chen, Xin Chen, Chi Zhang, Mingsheng Li, Gang Yu, Hao Fei, Hongyuan Zhu, Jiayuan Fan, and Tao Chen. Ll3da: Visual interactive instruction tuning for omni-3d understanding reasoning and planning. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26428–26438, 2024. 1, 3, 4, 6

[12] Yilun Chen, Shuai Yang, Haifeng Huang, Tai Wang, Ruiyuan Lyu, Runsen Xu, Dahua Lin, and Jiangmiao Pang. Grounded 3d-llm with referent tokens. arXiv preprint arXiv:2405.10370, 2024. 3, 6, 7

[13] Zhenyu Chen, Ali Gholami, Matthias Nießner, and Angel X Chang. Scan2cap: Context-aware dense captioning in rgb-d scans. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3193–3203, 2021. 2, 5, 6, 7, 8

[14] Zhenyu Chen, Ronghang Hu, Xinlei Chen, Matthias Nießner, and Angel X Chang. Unit3d: A unified transformer for 3d dense captioning and visual grounding. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 18109–18119, 2023. 6

[15] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. Palm: Scaling language modeling with pathways. Journal of Machine Learning Research, 24(240): 1–113, 2023. 2

[16] Christopher Choy, JunYoung Gwak, and Silvio Savarese. 4d spatio-temporal convnets: Minkowski convolutional neural networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 3075–3084, 2019. 3

[17] Angela Dai, Angel X Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 5828–5839, 2017. 5

[18] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. Instructblip: Towards generalpurpose vision-language models with instruction tuning, 2023. 1

[19] Martin Ester, Hans-Peter Kriegel, Jorg Sander, Xiaowei Xu,¨ et al. A density-based algorithm for discovering clusters in large spatial databases with noise. In kdd, pages 226–231, 1996. 7

[20] Rao Fu, Jingyu Liu, Xilun Chen, Yixin Nie, and Wenhan Xiong. Scene-llm: Extending language model for 3d visual understanding and reasoning. arXiv preprint arXiv:2403.11401, 2024. 3, 6

[21] Shuting He, Henghui Ding, Xudong Jiang, and Bihan Wen. Segpoint: Segment any point cloud via large language model. In European Conference on Computer Vision, pages 349–367, 2024. 1, 2, 3, 6, 7

[22] Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. Training compute-optimal large language models. arXiv preprint arXiv:2203.15556, 2022. 2

[23] Yining Hong, Haoyu Zhen, Peihao Chen, Shuhong Zheng, Yilun Du, Zhenfang Chen, and Chuang Gan. 3d-llm: Injecting the 3d world into large language models. Advances in Neural Information Processing Systems, 36:20482–20494, 2023. 1, 3, 6, 7

[24] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021. 5, 6

[25] Haifeng Huang, Yilun Chen, Zehan Wang, Rongjie Huang, Runsen Xu, Tai Wang, Luping Liu, Xize Cheng, Yang Zhao, Jiangmiao Pang, et al. Chat-scene: Bridging 3d scene and large language models with object identifiers. In The Thirtyeighth Annual Conference on Neural Information Processing Systems, 2024. 1, 3, 5, 6, 7, 8

[26] Jiangyong Huang, Silong Yong, Xiaojian Ma, Xiongkun Linghu, Puhao Li, Yan Wang, Qing Li, Song-Chun Zhu, Baoxiong Jia, and Siyuan Huang. An embodied generalist agent in 3d world. arXiv preprint arXiv:2311.12871, 2023. 6

[27] Kuan-Chih Huang, Xiangtai Li, Lu Qi, Shuicheng Yan, and Ming-Hsuan Yang. Reason3d: Searching and reasoning 3d segmentation via large language model. arXiv preprint arXiv:2405.17427, 2024. 2

[28] Pin-Hao Huang, Han-Hung Lee, Hwann-Tzong Chen, and Tyng-Luh Liu. Text-guided graph neural networks for referring 3d instance segmentation. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 1610– 1618, 2021. 2, 6

[29] Shijia Huang, Yilun Chen, Jiaya Jia, and Liwei Wang. Multiview transformer for 3d visual grounding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15524–15533, 2022. 2

[30] Jitesh Jain, Jiachen Li, Mang Tik Chiu, Ali Hassani, Nikita Orlov, and Humphrey Shi. Oneformer: One transformer to rule universal image segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2989–2998, 2023. 4

[31] Yang Jiao, Shaoxiang Chen, Zequn Jie, Jingjing Chen, Lin Ma, and Yu-Gang Jiang. More: Multi-order relation mining for dense captioning in 3d scenes. arXiv preprint arXiv:2203.05203, 2022. 2, 6

[32] Bu Jin, Yupeng Zheng, Pengfei Li, Weize Li, Yuhang Zheng, Sujie Hu, Xinyu Liu, Jinwei Zhu, Zhijie Yan, Haiyang Sun, et al. Tod3cap: Towards 3d dense captioning in outdoor scenes. In European Conference on Computer Vision, pages 367–384. Springer, 2024. 2

[33] Zhao Jin, Munawar Hayat, Yuwei Yang, Yulan Guo, and Yinjie Lei. Context-aware alignment and mutual masking for 3dlanguage pre-training. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10984–10994, 2023. 2, 6

[34] Minjung Kim, Hyung Suk Lim, Soonyoung Lee, Bumsoo Kim, and Gunhee Kim. Bi-directional contextual attention for 3d dense captioning. In European Conference on Computer Vision, pages 385–401. Springer, 2024. 2

[35] Maxim Kolodiazhnyi, Anna Vorontsova, Anton Konushin, and Danila Rukhovich. Oneformer3d: One transformer for unified point cloud segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20943–20953, 2024. 3

[36] Xin Lai, Yuhui Yuan, Ruihang Chu, Yukang Chen, Han Hu, and Jiaya Jia. Mask-attention-free transformer for 3d in-

stance segmentation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3693–3703, 2023. 3

[37] Xin Lai, Zhuotao Tian, Yukang Chen, Yanwei Li, Yuhui Yuan, Shu Liu, and Jiaya Jia. Lisa: Reasoning segmentation via large language model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9579–9589, 2024. 5

[38] Loic Landrieu and Martin Simonovsky. Large-scale point cloud semantic segmentation with superpoint graphs. In Proceedings of the IEEE conference on computer vision and pat tern recognition, pages 4558–4567, 2018. 3

[39] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. arXiv preprint arXiv:2301.12597, 2023. 1, 2, 3

[40] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. arXiv preprint arXiv:2304.08485, 2023. 1, 2, 3

[41] Haisong Liu, Yao Teng, Tao Lu, Haiguang Wang, and Limin Wang. Sparsebev: High-performance sparse 3d object de tection from multi-camera videos. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 18580–18590, 2023. 4

[42] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26296–26306, 2024. 1, 2, 5, 6

[43] LMSYS.org. Vicuna: An open-source chatbot impressing gpt-4 with 90https://lmsys.org. 1, 6

[44] Xiaojian Ma, Silong Yong, Zilong Zheng, Qing Li, Yitao Liang, Song-Chun Zhu, and Siyuan Huang. Sqa3d: Situated question answering in 3d scenes. arXiv preprint arXiv:2210.07474, 2022. 2, 5, 7, 8

[45] Maria Parelli, Alexandros Delitzas, Nikolas Hars, Georgios Vlassis, Sotirios Anagnostidis, Gregor Bachmann, and Thomas Hofmann. Clip-guided vision-language pre-training for question answering in 3d scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5606–5611, 2023. 2

[46] Songyou Peng, Kyle Genova, Chiyu Jiang, Andrea Tagliasacchi, Marc Pollefeys, Thomas Funkhouser, et al. Openscene: 3d scene understanding with open vocabularies. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 815–824, 2023. 2, 5

[47] Charles Ruizhongtai Qi, Li Yi, Hao Su, and Leonidas J Guibas. Pointnet++: Deep hierarchical feature learning on point sets in a metric space. Advances in neural information processing systems, 30, 2017. 3, 4

[48] Zhipeng Qian, Yiwei Ma, Jiayi Ji, and Xiaoshuai Sun. Xrefseg3d: Enhancing referring 3d instance segmentation via structured cross-modal graph neural networks. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 4551–4559, 2024. 2, 6

[49] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019. 1

[50] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 5

[51] David Rozenberszki, Or Litany, and Angela Dai. Languagegrounded indoor 3d semantic segmentation in the wild. In European Conference on Computer Vision, pages 125–141, 2022. 5

[52] Jonas Schult, Francis Engelmann, Alexander Hermans, Or Litany, Siyu Tang, and Bastian Leibe. Mask3d: Mask transformer for 3d semantic instance segmentation. In 2023 IEEE International Conference on Robotics and Automation (ICRA), pages 8216–8223. IEEE, 2023. 4, 7

[53] Jiahao Sun, Chunmei Qing, Junpeng Tan, and Xiangmin Xu. Superpoint transformer for 3d scene instance segmentation. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 2393–2401, 2023. 3, 4

[54] Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023. 2

[55] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023. 1, 2

[56] Ozan Unal, Christos Sakaridis, Suman Saha, and Luc Van Gool. Four ways to improve verbo-visual fusion for dense 3d visual grounding. In European Conference on Computer Vision, pages 196–213. Springer, 2024. 7

[57] Heng Wang, Chaoyi Zhang, Jianhui Yu, and Weidong Cai. Spatiality-guided transformer for 3d dense captioning on point clouds. arXiv preprint arXiv:2204.10688, 2022. 6

[58] Changli Wu, Yiwei Ma, Qi Chen, Haowei Wang, Gen Luo, Jiayi Ji, and Xiaoshuai Sun. 3d-stmn: Dependencydriven superpoint-text matching network for end-to-end 3d referring expression segmentation. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 5940– 5948, 2024. 2, 6

[59] Runsen Xu, Xiaolong Wang, Tai Wang, Yilun Chen, Jiangmiao Pang, and Dahua Lin. Pointllm: Empowering large language models to understand point clouds. arXiv preprint arXiv:2308.16911, 2023. 2

[60] Zhengyuan Yang, Songyang Zhang, Liwei Wang, and Jiebo Luo. Sat: 2d semantics assisted training for 3d visual grounding. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1856–1866, 2021. 2

[61] Zhihao Yuan, Xu Yan, Yinghong Liao, Yao Guo, Guanbin Li, Shuguang Cui, and Zhen Li. X-trans2cap: Crossmodal knowledge transfer using transformer for 3d dense captioning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8563– 8573, 2022. 2

[62] Sha Zhang, Di Huang, Jiajun Deng, Shixiang Tang, Wanli Ouyang, Tong He, and Yanyong Zhang. Agent3d-zero:

An agent for zero-shot 3d understanding. arXiv preprint arXiv:2403.11835, 2024. 3

[63] Tao Zhang, Xiangtai Li, Hao Fei, Haobo Yuan, Shengqiong Wu, Shunping Ji, Chen Change Loy, and Shuicheng Yan. Omg-llava: Bridging image-level, object-level, pixel-level reasoning and understanding. arXiv preprint arXiv:2406.19389, 2024. 2

[64] Yiming Zhang, ZeMing Gong, and Angel X Chang. Multi3drefer: Grounding text description to multiple 3d objects. In Proceedings of the IEEE/CVF International Con ference on Computer Vision, pages 15225–15236, 2023. 2, 5, 6, 7

[65] Yufeng Zhong, Long Xu, Jiebo Luo, and Lin Ma. Contextual modeling for 3d dense captioning on point clouds. arXiv preprint arXiv:2210.03925, 2022. 8

[66] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mo hamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023. 1

[67] Ziyu Zhu, Xiaojian Ma, Yixin Chen, Zhidong Deng, Siyuan Huang, and Qing Li. 3d-vista: Pre-trained transformer for 3d vision and text alignment. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2911– 2921, 2023. 2, 6, 7