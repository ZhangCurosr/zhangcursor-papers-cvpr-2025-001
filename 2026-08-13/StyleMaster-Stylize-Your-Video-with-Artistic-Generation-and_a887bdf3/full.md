# StyleMaster: Stylize Your Video with Artistic Generation and Translation

Zixuan Ye<sup>1†</sup> Huijuan Huang<sup>2\*</sup> Xintao Wang<sup>2</sup> Pengfei Wan<sup>2</sup> Di Zhang<sup>2</sup> Wenhan Luo<sup>1\*</sup> <sup>1</sup> Hong Kong University of Science and Technology <sup>2</sup> KuaiShou Technology

![](images/b3d81ea2866a247075623441407a8a3f8afddfd092473b1dd088488ecc100cd8.jpg)  
Figure 1. Our StyleMaster demonstrates superior video style transfer and stylized generation. The top section shows our method effectively applying various styles to videos, outperforming VideoComposer [46] and the combination of InstantStyle [44] with AnyV2V [22]. The bottom highlights our high-quality text-driven stylized synthesis, surpassing VideoComposer [46] and StyleCrafter [28].

## Abstract

Style control has been popular in video generation models. Existing methods often generate videos far from the given style, cause content leakage, and struggle to transfer one video to the desired style. Our first observation is that the style extraction stage matters, whereas existing methods emphasize global style but ignore local textures. In order to bring texture features while preventing content leakage, we filter content-related patches while retaining style ones based on prompt-patch similarity; for global style extraction, we generate a paired style dataset through model illusion to facilitate contrastive learning, which greatly enhances the absolute style consistency. Moreover, to fill in the image-to-video gap, we train a lightweight motion adapter on still videos, which implicitly enhances stylization extent, and enables our image-trained model to be seamlessly applied to videos. Benefited from these efforts,

our approach, StyleMaster, not only achieves significant improvement in both style resemblance and temporal coherence, but also can easily generalize to video style transfer with a gray tile ControlNet. Extensive experiments and visualizations demonstrate that StyleMaster significantly outperforms competitors, effectively generating high-quality stylized videos that align with textual content and closely resemble the style of reference images.

## 1. Introduction

Video generation [1, 7, 18, 31] has seen great success promoted by diffusion models [17, 27, 30, 33, 36, 41], which also bring great controllability. Wherein, the style control, i.e., to generate or translate a video to the same style as a given reference image, is of great interest and importance but less developed. Several instances exhibited in Fig. 2 reveal that current methods often struggle to preserve the loca textures, such as the brush strokes of a Van Gogh painting.

![](images/d953eb05c4b5d1a8420394d6c96a80b7872e786152b48997d79dd56050a1bf1f.jpg)  
Figure 2. Existing image and video stylization methods either fail in keeping local texture or suffer from content leakage. Note: \* means StyleCrafter does not support transfer, and we use text and reference style image to generate results.

Moreover, they fail to properly decouple content and style: they either focus too much on global style while losing texture details, or overuse reference features, leading to excessive copying and content leakage.

We argue that the failure comes largely from inappropriate use of global and texture features, and formulate separate remedies for both. First, to preserve the texture features, we turn to local patches for style guidance. However, directly using all image patch features from CLIP [35] can lead to content leakage. Thus, we keep patches of less similarity with text prompts, while discarding the rest. We empirically find that the selected patches can carry sufficient texture information, without bringing any associated contents. Although texture features are helpful for style representations, they are not sufficient to completely represent the style of an image without global information. One naive practice to incorporate global representation can be the global embedding from the CLIP encoder. However, it may easily invite content leakage, like that in Video-Composer [46]. One possible solution for decoupling is a contrastive learning strategy, with samples of the same style as positive, and others as negative [24]. However, existing style datasets cannot even guarantee style consistency within the group, which is sub-optimal to train a style extractor. Inspired by the illusion property demonstrated in VisualAnagrams [15], we can generate paired images where one image is a pixel-rearranged version of the other. Therefore, the content in the two images differs, while they share the same style. Unlike other manually collected and grouped datasets, we can generate a dataset of an almost infinite number of such pairs with minimal effort, while ensuring absolute style consistency within the pair. With these data pairs, we can train a strong module to extract the global-style-oriented features. In our practice, instead of fine-tuning the CLIP model, we opt to train a projection module after CLIP to ensure the generalization ability.

With the global and local features, the style information is then injected into the model in an adapter-based mechanism through the dual cross-attention strategy [52]. Since the image-only training will cause degradation in motion dynamics of videos, we adopt a motion adapter trained with still videos, inspired by StillMoving [4]. During inference, by turning the motion adapter’s ratio to negative, the motion quality is enhanced. More importantly, if the videos used for training are all in the real-world domain, the negative ratio implicitly helps to enhance the style extent by leading the generated results away from the real-world domain.

Beyond stylized generation, we explore video translation as a broader application. While existing methods focus only on stylized video generation or rely on depth-based ControlNet for content control [28, 46], we propose a simpler solution: we design a gray tile ControlNet as more accessible yet precise content guidance for video style transfer. Extensive experiments show that StyleMaster can generate high-quality videos with high style similarity to the reference image and achieve ideal translation results, significantly outperforming other competitors in several stylization tasks. Rich ablation studies are conducted to validate the effectiveness of the proposed modules. In conclusion, our contributions are threefold:

• We propose a novel style extraction module with local patch selection to overcome the content leakage in style transfer, and global projection to extract strong style cues.

• We are the first to propose using model illusion to generate datasets of paired images with absolute style consistency at nearly no cost. This not only produces accurate style-content decoupling in our approach but also benefits style-related research in the community.

• With an adopted motion adapter and gray tile Control-Net, our developed StyleMaster is capable of generating content accurately representing the given reference style in both video generation and video/image style transfer tasks, and more importantly, outperforms other methods significantly as demonstrated by the experimental results.

## 2. Related Work

## 2.1. Image Stylization

The success of generative models [7, 18, 36] has inspired customized generative models specifying object [9, 37], edge [3, 32, 53], layout [8, 49], ID [26]. Controlling generation with a specific style from a reference image has also garnered significant attention, with numerous studies exploring how to extract the style description from the reference image and how to inject the style cues [2, 6, 34, 47]. Inspired by Textural Inversion (TI) [13], some methods [10, 39, 54] optimize a specific textual embedding to represent style. Instead of relying on inversion, IP-Adapter [52] trains an image adapter to adapt T2I models to image conditions. However, it cannot decouple style and content in the reference image, resulting in severe content leakage.

To address this, InstantStyle [44, 45] identifies the layer truly impacting stylization and injects style only into the identified layer. However, due to the sub-optimal style extractor, it suffers from poor style precision. StyleTokenizer [24] fine-tunes an image encoder with a manually collected and grouped style dataset, Style30K, in a contrastive training manner. However, Style30K cannot maintain style consistency, which adversely affects the extractor. CSGO [51] creates a triplet dataset consisting of contentstyle-sample pairs generated by B-LoRA [12], achieving impressive performance. However, it can only extract global representations and fails to preserve local textures. Therefore, we propose to consider both local and global information and create a dataset with absolute style consistency, i.e., performing pixel rearrangement to form a new image, leveraging the model’s illusion property [15].

## 2.2. Video Stylization

One can achieve video style transfer by applying a frameby-frame process using an image stylization model. However, this can lead to temporal inconsistency. To address this, early deep learning methods [5, 11, 14, 20] employ optical flow constraints. Generative models have elevated this task to a new level. Given the first and/or the last frame, some Image-to-Video (I2V) methods can create stylized videos [50]. However, under this setting, users cannot specify the style with a reference style image. Some video editing methods, like AnyV2V [22], also attempt to stylize the video given an edited first frame, but it requires an image stylization model to obtain the stylized first frame. In contrast, AnimateDiff [16] extends the Text-to-Image model into a Text-to-Video model by adding a temporal module. StillMoving [4] further frees the requirement for video data by training a motion adapter with still videos, enabling easy cooperation with any image customization model [37].

Some works based on T2V models [7, 16] focus on controllable video generation. For example, VideoComposer [46] achieves multiple controls including style control. However, directly injecting all reference image tokens during training causes serious content leakage. Instead, StyleCrafter [28] adopts Q-Former to extract the style descriptions from an image. However, it ignores the local texture, resulting in sub-optimal stylization. Additionally, it focuses on stylized generation only, rather than style transfer, which is an important aspect of video stylization.

![](images/8939649d4afad2c2d29189f8c8005fd1f8b513802390eda24ed0809e1c0909f9.jpg)  
Figure 3. Comparison between Style30K with our dataset generated by model illusion. Style30K cannot ensure consistency within a style group (highlighted by the same color), while ours owns absolute consistency.

## 3. Method

In this section, we illustrate the components of our Style-Master. We first construct a contrastive dataset (Sec. 3.1) with absolute style consistency, and develop global and local style extraction methods (Sec. 3.2 and Sec. 3.3) for accurate style representation. We mitigate dynamic degradation in Sec. 3.4, and introduce a content control mechanism using gray tile guidance in Sec. 3.5.

## 3.1. Contrastive Dataset Construction

StyleTokenizer [24] collects a style dataset consisting of 30K style images, groups them into about 30 style groups, and uses this dataset to finetune the CLIP through contrastive learning. In this step, the quality of the style dataset is crucial, as it largely determines the final capability of the extractor. However, as shown in Fig. 3, the style consistency within a group cannot be guaranteed. Specifically, the first two images highlighted by yellow bounding boxes illustrate this inconsistency: one belongs to the real-world domain, while the other is from the animation domain, yet both are classified as the same style. Moreover, the process of collecting and grouping is labor-intensive. Therefore, a more efficient method to obtain style data is required.

We draw inspiration from the success of model illusion [15], which uses pretrained T2I models to create optical illusions. To be specific, given an arbitrary T2I model, during the sampling process, we copy and change the view (e.g., rotation, flip) of noisy image to form a parallel process, then use two different prompts to guide the noise prediction of the two noisy images. Then we change the predicted noise back to its original view and add the two predicted noises to form the output noise. In this way, the generated images can change appearance when pixels are rearranged. Based on this, to generate the dataset, we create two lists: one containing objects and the other containing style descriptions. We then randomly select a style and two objects to generate paired images, e.g, as shown in Fig. 4, the prompts are “a watercolor painting of a dog” and “a watercolor painting of a rabbit”. Since the paired images in model illusion are merely pixel rearrangements, we can ensure style consistency within a group. Leveraging this property, we can automatically generate an infinite amount of data with no effort.

![](images/8303b976de9fc4f106af1f5367b96a80eae48c5409c6741d83ad7783650bc190.jpg)  
Figure 4. The pipeline of our proposed StyleMaster. We first obtain patch features and image embedding of the style image from CLIP, then we select the patches sharing less similarity with text prompt as texture guidance, and use a global projection module to transform it into global style descriptions. The global projection module is trained with a contrastive dataset constructed by model illusion through contrastive learning. The style information is then injected into the model through the decoupled cross-attention. The motion adapter and gray tile ControlNet are used to enhance dynamic quality and enable content control respectively.

## 3.2. Extract Global Description

Rather than fine-tuning the entire CLIP image encoder like StyleTokenizer [24], which might compromise its generalization ability, we opt for a post-processing module to the image embedding output from CLIP, i.e.,

$$
F _ { i } = \mathrm { C L I P } ( I ) . \mathrm { i m a g e \mathrm { _ - e m b e d } } ,
$$

where I represents the style image, and $F _ { i } \in \mathbb { R } ^ { 1 \times 1 0 2 4 }$ . We use a simple MLP layer $f ( x ) = \mathrm { M L P } ( x )$ as a projection to transform the image embedding, which contains both content and style information, into only global style representation. During training, we employ a triplet loss where we treat one image from a paired set as the anchor, its counterpart as the positive sample, and any image outside this pair as the negative sample:

$$
\mathcal { L } = \sum _ { n = 1 } ^ { N } \left[ \Vert f ( F _ { i , n } ^ { \mathrm { a n c } } ) - f ( F _ { i , n } ^ { \mathrm { p o s } } ) \Vert - \Vert f ( F _ { i , n } ^ { \mathrm { a n c } } ) - f ( F _ { i , n } ^ { \mathrm { n e g } } ) \Vert + \alpha \right] ,
$$

and the margin α defines the distance between samples of different groups.

This projection allows us to preserve the generalization capabilities of pre-trained CLIP while tailoring the output to focus on style-oriented features. The MLP layer serves as a learnable transformation that distills style information from the image embedding. As illustrated in Fig. 5, we compare the global feature’s similarity with patches before and after the projection. The result shows that, compared to similarities between global feature and patch features before projection showing peak distribution, the after-projection ones are more evenly distributed. This is also supported by the smaller variance after projection. It aligns with the target of the global style description, which should represent the whole image. Therefore, we obtain the $\mathrm { g l }$ obal feature by:

![](images/182300f0919c2198f06d9c4c9c113ea4590c7264b194476521abb3752101b5a1.jpg)  
Figure 5. Similarity between the extracted global style representations among image patches. Without our global projection, the CLIP image embedding only attends to specific regions; while after the projection, the attention shows an even distribution.

$$
F _ { \mathrm { g l o b a l } } = \mathbf { M L P } ( F _ { i } ) .
$$

## 3.3. Combine Local and Global Description

However, relying solely on the global description is insufficient for obtaining optimal style representations. As illustrated in Fig. 2, while the global representation can accurately capture the overall style of the reference image, it fails to preserve local textures, such as the distinctive brushstrokes in Van Gogh paintings. To address this limitation, we consider to use the patch features extracted by CLIP:

$$
F _ { p } = \mathbf { C L I P } ( I ) . \mathsf { p a t c h \_ f e a t u r e } ,
$$

where $F _ { p } \in \mathbb { R } ^ { 2 5 6 \times 1 0 2 4 }$ . However, directly preserving all patch features would risk content leakage. Therefore, we propose a selection strategy to choose only a few patch features as the texture feature. To avoid any content leakage, we incorporate the prompt feature, and compare it with image patch features to obtain similarity scores. We further choose the patches sharing less similarity score with the texture feature, which are more likely to carry only texture instead of any content. Specifically, we choose them by

$$
F _ { p } ^ { \prime } = \operatorname { c o n c a t } ( F _ { p } ^ { i } \mid i \in \operatorname { a r g s o r t } ( \operatorname { s i m i l a r i t y } ( F _ { p } , F _ { t e x t } ) ) [ : k ] )
$$

and k is set to 15 in our method. Following StyleCrafter, we use Q-Former [23] structure to further gather features from the filtered patches. We create N learnable tokens $F _ { q u e r y } \in \mathbb { R } ^ { N \times C }$ and then concatenate it with $F _ { p } ^ { \prime }$ to perform self-attention [43]:

$$
F _ { a t t n } = \mathrm { s e l f - a t t e n t i o n } ( { \mathrm { c o n c a t } } ( F _ { q u e r y } , F _ { p } ^ { \prime } ) ) .
$$

Then we take out the first N tokens from $F _ { a t t n }$ as $F _ { t e x t u r e } ~ \in ~ \mathbb { R } ^ { N \times C }$ as the texture feature. As shown in Fig. 6, the first row demonstrates the kept patches with varying drop ratios, the patches of the face and body are gradually dropped due to higher similarity with the prompt, which includes the description of a human. Additionally, without selection, directly using all patches will pose interruptions to text alignment. For example, when the drop rate is set to 0, the texture only attends to the person on the right, serving as content guidance instead of texture guidance.

The texture feature $F _ { t e x t u r e }$ and the global style description $F _ { g l o b a l }$ are concatenated as $F _ { s t y l e }$ , to perform the dualcross-attention in an adapter manner [52], as

$$
F _ { o u t } = \mathrm { T C A } ( F _ { i n } , F _ { t e x t } ) + \mathrm { S C A } ( F _ { i n } , F _ { s t y l e } ) ,
$$

where TCA represents text cross-attention, and SCA refers to style cross-attention, the $F _ { i n }$ is the input of crossattention module, and $F _ { o u t }$ is the output.

## 3.4. Motion Adapter for Temporal and Style Quality

While the aforementioned design enables us to inject the style information into model, it would result in temporal flickering and limited range of dynamics. To address these issues, we propose a method to enhance temporal quality with minimal modifications, inspired by the success of Still-Moving [4]. It demonstrates a smooth transition from customized T2I (Text-to-Image) models to customized T2V (Text-to-Video) models by incorporating a motion adapter trained on still videos. Specifically, for each weight matrix $W \in \{ W _ { Q } , W _ { K } , W _ { V } \}$ in the temporal attention block, we train a LoRA [19] by applying the following transformation:

![](images/666b89be2457ba906789d0b6f00ca5e1445cfb0b195977c1612def018136ce5f.jpg)  
Figure 6. The selection of texture feature using similarity with prompt features. Top: the kept patches under different drop rates, showing that the dropped tokens are mainly on human-related regions (especially when the drop rate is 0.7). Bottom: the attention map of the cross-attention between texture feature and latent when the drop rate is 0 and 0.95, and their generated results.

$$
\widetilde { W } = W + \alpha \cdot A _ { t } ^ { W , \mathrm { { d o w n } } } \cdot A _ { t } ^ { W , \mathrm { { u p } } } ,
$$

where $A _ { t } ^ { W , }$ <sup>down</sup> and $A _ { t } ^ { W , \mathfrak { u p } }$ are learnable parameters of the motion adapter, trained on still videos with $\alpha = 1$ . This formulation offers flexibility in controlling the model’s behavior. Setting $\alpha = 0$ leaves the original model unchanged. Setting $\alpha = 1$ generates static videos. Setting $\alpha ~ = ~ - 1$ produces the opposite effect, transitioning from stillness to a greater dynamic range. More importantly, since we train the adapter on real-world videos, setting $\alpha = - 1$ not only increases the dynamic range but also enhances the stylization by moving further away from the real-world domain, aligning the goal of stylization.

## 3.5. Gray Tile ControlNet

To enable both stylized generation and style transfer, we incorporate content guidance into our model. Following CSGO [51] and InstantStylePlus [45], we employ a tile ControlNet as the content guidance mechanism. However, we find that the color information in the tile image may interfere with the style transfer process, as shown in Fig. 10. To address this, we remove the color information from the tile image, converting it into a grayscale image.

The gray tile ControlNet uses $N / 2$ vanilla DiT blocks, which inject the content feature into the denoising network at regular intervals. The vanilla DiT block only contains self-attention, temporal attention, text cross-attention, and FFN, and does not include specific designs like the motion adapter and style cross-attention. The output from each vanilla DiT block will be added to the corresponding style DiT block as the content guidance.

![](images/6734bad6878afd237982f75718d2b7d148322b806a963358ce31e1adbcda5507.jpg)  
Figure 7. Uncurated image style transfer results. We compare with the recent state-of-the-art methods InstantStyle [44], StyleID [10] and CSGO [51]. Best viewed in Color.

## 4. Experiments

Implementation Details. We develop a DiT-based [33] video generation model as our base model, which consists of a 3D casual VAE and several DiT Blocks as the denoising network (detail can refer to Supplement). We first train the global style extractor on 10K pairs of style data generated by the model illusion through contrastive learning. Then, we train the motion adapter with still videos for about 300 iterations with batch size 64. Next, we start to train the style modulation on image dataset, i.e., Laion-Aesthetics [38] with a batch size of 160 per GPU for 40K iterations. With the style module ready, we train the gray tile ControlNet with the image dataset, using the above setting for 20K iterations. We train our model on 8 A800 GPUs, which can be completed within two days. For the classifier-free guid ance, we use the decoupled cfg like other methods [28]. In our method, we set text cfg to 12.5 and style cfg to 6.

Evaluation Metrics. For the image style transfer task, we employ the metric CSD score [40] used in CSGO [51] to measure style similarity with the reference image, and the series of metrics used in StyleID [10] to validate the style transfer quality. Specifically, ArtFID [48] is notable for its strong correlation with human perception, as it considers both style and content fidelity. We also adopt the CFSD metric [10] to further validate the content preservation.

For the assessment of video stylization, we employ a two-fold validation. First, we utilize image metrics to perform frame-by-frame validation of image stylization. Second, following StyleCrafter [28], we evaluate motion quality. Specifically, we adopt motion smoothness metric and dynamic degree metric proposed in VBench [21] for evaluation, using the AMT [25] and RAFT [42] as the base model respectively. For the text-video alignment, we employ UMT score [29] and CLIP-Text [35] similarity as metrics.

Dataset. For video stylized generation, we base our test set on that proposed by StyleCrafter [28]. We expand this set by adding more style images and prompts, obtaining a comprehensive test set comprising 12 style images and 16 prompts, yielding a total of 192 style-prompt pairs.

<table><tr><td></td><td>StyleID [10] (CVPR&#x27;24)</td><td>InstantStyle [44] (arxiv&#x27;24)</td><td>CSGO [51] (arxiv&#x27;24)</td><td>Ours</td></tr><tr><td>CSD-Score ↑</td><td>0.40</td><td>0.32</td><td>0.35</td><td>0.45</td></tr><tr><td>ArtFID ↓</td><td>38.57</td><td>42.48</td><td>41.42</td><td>36.89</td></tr><tr><td>FID↓</td><td>23.91</td><td>24.59</td><td>25.71</td><td>22.11</td></tr><tr><td>LPIPS ↓</td><td>0.55</td><td>0.67</td><td>0.56</td><td>0.61</td></tr><tr><td>CFSD↓</td><td>1.06</td><td>1.70</td><td>5.12</td><td>2.37</td></tr></table>

Table 1. The quantitative results of image style transfer. Art-FID considers both style resemble and content preservation. CSDscore represents the similarity of style. The last two reflect content preservation. The first two metrics in bold are the most representative metrics for this evaluation.

For image style transfer, we curate a test set consisting of 8 content images paired with the aforementioned 12 style images. This leads to 96 content-style pairs, matching the test set size used in other image style transfer methods [10].

## 4.1. Image Style Transfer

We consider image style transfer as the most intuitive evaluation method for style learning, with minimal dependence on base model generation abilities or temporal factors. Therefore, we also compare our method with image stylization methods by regarding the image as a one-frame video. We choose two training-free SOTA methods StyleID [10] and InstantStyle [44], and a training-based SOTA method CSGO as our competitors. In Table 1, following CSGO [51] and StyleID [10], we demonstrate the CSD scores [40] and the content preservation metrics of the proposed method compared to recent advanced methods. Our method significantly outperforms others in the first three metrics, indicating accurate style learning from reference images. While slightly underperforming in content alignment metrics, we argue that effective style transfer requires balancing style fidelity and content retention. For example, as shown in

![](images/d1f9e1f0691578002389836dff5d1538cdfd7ea69d687326e241b5687cce78f2.jpg)  
Figure 8. Qualitative comparison of single-reference and multi-reference style-guided T2V generation. We compare with Style Crafter [28] and VideoComposer [46]. Best viewed in color.

<table><tr><td colspan="2">VideoComposer [46] (NeurIPS&#x27;24)</td><td>StyleCrafter [28] (SIGGRAPH Asia&#x27;24)</td><td>Ours</td></tr><tr><td>CLIP-Text ↑</td><td>0.057</td><td>0.294</td><td>0.305</td></tr><tr><td>UMT-Score ↑</td><td>-2.268</td><td>1.994</td><td>2.329</td></tr><tr><td>CSD-Score ↑</td><td>0.680</td><td>0.448</td><td>0.463</td></tr><tr><td>VisualQuality ↑</td><td>2.159</td><td>2.140</td><td>2.370</td></tr><tr><td>DynamicQuality ↑</td><td>2.284</td><td>2.306</td><td>2.496</td></tr><tr><td>MotionSmooth ↑</td><td>0.975</td><td>0.973</td><td>0.994</td></tr></table>

Table 2. Comparison of stylized video generation results. We compare our method with VideoComposer [46] and Style-Crafter [28]. Our method demonstrates higher style resemblance and stronger text alignment.

![](images/a04be005dd6c000ef8ddef44c807e4ede22d8ad1a1dd0937c50b5b143acd489c.jpg)  
Figure 9. Video style transfer results compared with DomoAI. Their results disrupt semantics, shown in red bounding box.

Fig. 7 (top row), the Noble style causes some loss of details due to its simple line style, which is an expected transformation to fit the style. However, we argue that effective style transfer is not solely judged by content preservation, but rather on achieving an optimal balance between style fidelity and content retention. The ArtFID metric, which aligns well with human preference, shows our method’s significant advantage. Figure 7 presents uncurated test samples, demonstrating our method’s ability to capture reference style accurately while maintaining high content preservation.

## 4.2. Stylized Video Generation

For the stylized video generation task, we use the aforementioned test set which contains 192 style-prompt pairs to generate videos. We compare our method with previous state-of-the-art methods StyleCrafter [28] and Video-Composer [46]. As shown in Table 2, our method outperforms these two methods in five metrics, which demonstrates our superiority in alignment between text and video (0.305 CLIP-Text similarity and 2.329 UMT-Score), enhanced visual and dynamic quality, and smoother motion. Our CSD score falls behind VideoComposer. The reason lies in that it directly copies the content in the reference image, therefore exhibiting a higher style score. Instead, our method implements style injection on the basis of text alignment, and achieves both high T2V alignment and high style consistency with the reference image.

The visualization results are shown in Fig. 8. Other methods either fail to accurately capture the style in the reference image or suffer from poor text alignment. For example, the generation results of VideoComposer almost show no correspondence with the given prompt, which aligns with the negative UMT score in Table 2. Although StyleCrafter demonstrates style similarity to some extent, it learns only the superficial style representations like color but not the complete style descriptions. We also compare the generation results with single/multiple reference images, both generating high-quality stylized videos.

## 4.3. Video Style Transfer

Here we compare our method with an online commercial application DomoAI<sup>\*</sup>. As shown in Fig. 9, it can achieve appealing stylization results, but they may interrupt the semantics within the video.

<table><tr><td rowspan="2"></td><td colspan="2">Global</td><td colspan="2">Texture</td><td rowspan="2">UMT Score</td><td rowspan="2">CSD Score</td></tr><tr><td>w/o GP</td><td>w/ GP</td><td>w/o S</td><td>w/S</td></tr><tr><td>B1</td><td>√</td><td></td><td rowspan="6"></td><td></td><td>0.892</td><td>0.561</td></tr><tr><td>B2</td><td>√</td><td></td><td></td><td>2.337</td><td>0.443</td></tr><tr><td>B3</td><td></td><td>√</td><td></td><td>0.771</td><td>0.534</td></tr><tr><td>B4</td><td></td><td></td><td>random</td><td>2.129</td><td>0.454</td></tr><tr><td>B5</td><td></td><td>√</td><td></td><td>2.331</td><td>0.452</td></tr><tr><td>B6</td><td>√</td><td></td><td>√</td><td>2.329</td><td>0.463</td></tr></table>

Table 3. The ablation study of the style extraction module design. GP means the global projection after CLIP image embedding, and S refers to the selection using text features. random represents selecting the same amount of tokens randomly.

## 4.4. Ablation Studies

## 4.4.1. Style Representation Extraction

To validate whether our specific designs in the style extraction module really matter, we conduct an ablation study to show their effect. The results are reported in Table 3. Com paring B1 with B2, the use of Global Project (GP) can effectively prevent content leakage caused by directly using the CLIP image embedding, with an obvious improvement in the UMT score (2.337 vs 0.892). Additionally, directly using all image patch features as texture features will bring a similar problem, reflected by the poor 0.771 UMT score in B3. However, selecting only a few of them can help alleviate the problem; i.e., randomly discarding most tokens can enhance text alignment. Furthermore, comparing B4 with B5, if we select the tokens by considering their similarity with prompts instead of a random selection, we can obtain higher text alignment while maintaining style similarity. Variant B6 demonstrates that the integration of global and local styles can further enhance style resemblance, achieving a CSD score of 0.463.

## 4.4.2. Motion Adapter

To explore the effect of motion adapter on both dynamics and style, we conduct an ablation study with different α values ranging from 0 to −1. As shown in Table 4, when the negative scale of motion adapter increases (0 → −1), the CSD score representing style similarity also increases. It verifies that, due to the training on real-world images, the negative ratio can generate results away from the real-world domain, leading to a more stylistic video. Also, the dynamic degree will increase as the scale increases. However, it damages the text alignment (UMTScore) and the motion smooth when the scale exceeds 0.3. Therefore, we choose −0.3 as our setting, which owns the best visual quality and also well achieves the dynamic degree and style similarity.

## 4.4.3. Content Control

We compare different conditions for content control during generation. We compare gray tile guidance with Canny and

<table><tr><td>MotionAdapterScale</td><td>-1</td><td>-0.5</td><td>-0.3</td><td>-0.1</td><td>0</td></tr><tr><td>CSDScore ↑</td><td>0.465</td><td>0.465</td><td>0.463</td><td>0.446</td><td>0.443</td></tr><tr><td>DynamicDegree↑</td><td>20.559</td><td>9.320</td><td>6.579</td><td>1.576</td><td>1.371</td></tr><tr><td>UMTScore↑</td><td>2.211</td><td>2.193</td><td>2.329</td><td>2.235</td><td>2.272</td></tr><tr><td>MotionSmooth↑</td><td>0.990</td><td>0.992</td><td>0.994</td><td>0.994</td><td>0.994</td></tr><tr><td>VisualQuality↑</td><td>2.259</td><td>2.263</td><td>2.370</td><td>2.279</td><td>2.278</td></tr></table>

Table 4. The effect of motion adapter on generation results. We choose −0.3 as the suitable scale.

![](images/498cc487bb348b8f2c256db3f853f7c551685eed698b373ec0006614ba009237.jpg)  
Figure 10. Ablation of different conditions of ControlNet in our method. The gray tile achieves the best performance.

RGB tile images. As shown in Fig. 10, the Canny method provides too much detailed information but less layout guidance. In contrast, the RGB tile image can provide a layout hint, leading to more precise content control. However, the color information within the guidance can interrupt the style injection, resulting in darker outputs, as shown in the third column. To alleviate this, we use the gray tile image as the condition. The results verify the improvements.

## 5. Conclusion

In this paper, we address the challenges faced by existing stylization methods, particularly in sub-optimal style extraction and the lack of video translation. To tackle these issues, we propose a novel approach that leverages both global and local style representations to achieve an ideal style descriptor. Our method involves selecting local patches with minimal content similarity to capture texture details and using a contrastive learning strategy to train a global style extractor with paired data generated through model illusion. To enhance video quality, we incorporate a motion adapter, which improves motion quality and style extent during inference. Additionally, we implement a gray tile ControlNet for more precise content guidance in video translation tasks. Beyond the implementation, our method significantly outperforms other methods in both text alignment and style resemblance.

Limitation. Current stylization methods typically rely on reference style images. However, video stylization includes more than just graphic style—it also involves dynamic elements like particle effects and motion characteristics. In future research, we aim to explore methods for extracting and transferring dynamic styles from reference videos.

## References

[1] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, et al. Video generation models as world simulators, 2024. 1

[2] Yichao Cai, Yuhang Liu, Zhen Zhang, and Javen Qinfeng Shi. Clap: Isolating content from style through contrastive learning with augmented prompts. 2024. 3

[3] John Canny. A computational approach to edge detection. IEEE Transactions on pattern analysis and machine intelligence, (6):679–698, 1986. 2

[4] Hila Chefer, Shiran Zada, Roni Paiss, Ariel Ephrat, Omer Tov, Michael Rubinstein, Lior Wolf, Tali Dekel, Tomer Michaeli, and Inbar Mosseri. Still-moving: Customized video generation without customized video data. arXiv preprint arXiv:2407.08674, 2024. 2, 3, 5

[5] Dongdong Chen, Jing Liao, Lu Yuan, Nenghai Yu, and Gang Hua. Coherent online video style transfer. In Proceedings of the IEEE International Conference on Computer Vision, pages 1105–1114, 2017. 3

[6] Dar-Yen Chen, Hamish Tennent, and Ching-Wen Hsu. Artadapter: Text-to-image style transfer using multi-level style encoder and explicit adaptation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8619–8628, 2024. 3

[7] Haoxin Chen, Menghan Xia, Yingqing He, Yong Zhang, Xiaodong Cun, Shaoshu Yang, Jinbo Xing, Yaofang Liu, Qifeng Chen, Xintao Wang, Chao Weng, and Ying Shan. Videocrafter1: Open diffusion models for high-quality video generation, 2023. 1, 2, 3

[8] Minghao Chen, Iro Laina, and Andrea Vedaldi. Training-free layout control with cross-attention guidance. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 5343–5353, 2024. 2

[9] Xi Chen, Lianghua Huang, Yu Liu, Yujun Shen, Deli Zhao, and Hengshuang Zhao. Anydoor: Zero-shot object-level image customization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6593–6602, 2024. 2

[10] Jiwoo Chung, Sangeek Hyun, and Jae-Pil Heo. Style injection in diffusion: A training-free approach for adapting largescale diffusion models for style transfer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8795–8805, 2024. 3, 6

[11] Yingying Deng, Fan Tang, Weiming Dong, Haibin Huang, Chongyang Ma, and Changsheng Xu. Arbitrary video style transfer via multi-channel correlation. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 1210– 1217, 2021. 3

[12] Yarden Frenkel, Yael Vinker, Ariel Shamir, and Daniel Cohen-Or. Implicit style-content separation using b-lora. arXiv preprint arXiv:2403.14572, 2024. 3

[13] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. An image is worth one word: Personalizing text-toimage generation using textual inversion. arXiv preprint arXiv:2208.01618, 2022. 3

[14] Wei Gao, Yijun Li, Yihang Yin, and Ming-Hsuan Yang. Fast video multi-style transfer. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 3222–3230, 2020. 3

[15] Daniel Geng, Inbum Park, and Andrew Owens. Visual anagrams: Generating multi-view optical illusions with diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24154– 24163, 2024. 2, 3

[16] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized textto-image diffusion models without specific tuning. In The Twelfth International Conference on Learning Representations. 3

[17] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 1

[18] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. arXiv preprint arXiv:2205.15868, 2022. 1, 2

[19] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021. 5

[20] Haozhi Huang, Hao Wang, Wenhan Luo, Lin Ma, Wenhao Jiang, Xiaolong Zhu, Zhifeng Li, and Wei Liu. Real-time neural style transfer for videos. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 7044– 7052, 2017. 3

[21] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21807–21818, 2024. 6

[22] Max Ku, Cong Wei, Weiming Ren, Huan Yang, and Wenhu Chen. Anyv2v: A plug-and-play framework for any videoto-video editing tasks. arXiv preprint arXiv:2403.14468, 2024. 1, 3

[23] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International conference on machine learning, pages 19730– 19742. PMLR, 2023. 5

[24] Wen Li, Muyuan Fang, Cheng Zou, Biao Gong, Ruobing Zheng, Meng Wang, Jingdong Chen, and Ming Yang. Styletokenizer: Defining image style by a single instance for controlling diffusion models. arXiv preprint arXiv:2409.02543, 2024. 2, 3, 4

[25] Zhen Li, Zuo-Liang Zhu, Ling-Hao Han, Qibin Hou, Chun-Le Guo, and Ming-Ming Cheng. Amt: All-pairs multi-field transforms for efficient frame interpolation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9801–9810, 2023. 6

[26] Zhen Li, Mingdeng Cao, Xintao Wang, Zhongang Qi, Ming-Ming Cheng, and Ying Shan. Photomaker: Customizing re-

alistic human photos via stacked id embedding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8640–8650, 2024. 2

[27] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022. 1

[28] Gongye Liu, Menghan Xia, Yong Zhang, Haoxin Chen, Jinbo Xing, Yibo Wang, Xintao Wang, Yujiu Yang, and Ying Shan. Stylecrafter: Enhancing stylized text-to-video generation with style adapter. arXiv preprint arXiv:2312.00330, 2023. 1, 2, 3, 6, 7

[29] Ye Liu, Siyuan Li, Yang Wu, Chang-Wen Chen, Ying Shan, and Xiaohu Qie. Umt: Unified multi-modal transformers for joint video moment retrieval and highlight detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3042–3051, 2022. 6

[30] Nanye Ma, Mark Goldstein, Michael S Albergo, Nicholas M Boffi, Eric Vanden-Eijnden, and Saining Xie. Sit: Exploring flow and diffusion-based generative models with scalable interpolant transformers. arXiv preprint arXiv:2401.08740, 2024. 1

[31] Xin Ma, Yaohui Wang, Gengyun Jia, Xinyuan Chen, Ziwei Liu, Yuan-Fang Li, Cunjian Chen, and Yu Qiao. Latte: Latent diffusion transformer for video generation. arXiv preprint arXiv:2401.03048, 2024. 1

[32] Chong Mou, Xintao Wang, Liangbin Xie, Yanze Wu, Jian Zhang, Zhongang Qi, and Ying Shan. T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 4296–4304, 2024. 2

[33] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4195–4205, 2023. 1, 6

[34] Tianhao Qi, Shancheng Fang, Yanze Wu, Hongtao Xie, Jiawei Liu, Lang Chen, Qian He, and Yongdong Zhang. Deadiff: An efficient stylization diffusion model with disentangled representations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8693–8702, 2024. 3

[35] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 2, 6

[36] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High-resolution image¨ synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 1, 2

[37] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 22500– 22510, 2023. 2, 3

[38] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in Neural In formation Processing Systems, 35:25278–25294, 2022. 6

[39] Kihyuk Sohn, Nataniel Ruiz, Kimin Lee, Daniel Castro Chin, Irina Blok, Huiwen Chang, Jarred Barber, Lu Jiang, Glenn Entis, Yuanzhen Li, et al. Styledrop: Text-to-image generation in any style. arXiv preprint arXiv:2306.00983, 2023. 3

[40] Gowthami Somepalli, Anubhav Gupta, Kamal Gupta, Shramay Palta, Micah Goldblum, Jonas Geiping, Abhinav Shrivastava, and Tom Goldstein. Measuring style similarity in diffusion models. arXiv preprint arXiv:2404.01292, 2024. 6

[41] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020. 1

[42] Zachary Teed and Jia Deng. Raft: Recurrent all-pairs field transforms for optical flow. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23– 28, 2020, Proceedings, Part II 16, pages 402–419. Springer, 2020. 6

[43] A Vaswani. Attention is all you need. Advances in Neural Information Processing Systems, 2017. 5

[44] Haofan Wang, Matteo Spinelli, Qixun Wang, Xu Bai, Zekui Qin, and Anthony Chen. Instantstyle: Free lunch towards style-preserving in text-to-image generation. arXiv preprint arXiv:2404.02733, 2024. 1, 3, 6

[45] Haofan Wang, Peng Xing, Renyuan Huang, Hao Ai, Qixun Wang, and Xu Bai. Instantstyle-plus: Style transfer with content-preserving in text-to-image generation. arXiv preprint arXiv:2407.00788, 2024. 3, 5

[46] Xiang Wang, Hangjie Yuan, Shiwei Zhang, Dayou Chen, Jiuniu Wang, Yingya Zhang, Yujun Shen, Deli Zhao, and Jingren Zhou. Videocomposer: Compositional video synthesis with motion controllability. Advances in Neural Information Processing Systems, 36, 2024. 1, 2, 3, 7

[47] Zhouxia Wang, Xintao Wang, Liangbin Xie, Zhongang Qi, Ying Shan, Wenping Wang, and Ping Luo. Styleadapter: A single-pass lora-free model for stylized image generation. arXiv preprint arXiv:2309.01770, 2023. 3

[48] Matthias Wright and Bjorn Ommer. Artfid: Quantitative¨ evaluation of neural style transfer. In DAGM German Con ference on Pattern Recognition, pages 560–576. Springer, 2022. 6

[49] Jinheng Xie, Yuexiang Li, Yawen Huang, Haozhe Liu, Wentian Zhang, Yefeng Zheng, and Mike Zheng Shou. Boxdiff: Text-to-image synthesis with training-free box-constrained diffusion. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7452–7461, 2023. 2

[50] Jinbo Xing, Hanyuan Liu, Menghan Xia, Yong Zhang, Xintao Wang, Ying Shan, and Tien-Tsin Wong. Tooncrafter: Generative cartoon interpolation. arXiv preprint arXiv:2405.17933, 2024. 3

[51] Peng Xing, Haofan Wang, Yanpeng Sun, Qixun Wang, Xu Bai, Hao Ai, Renyuan Huang, and Zechao Li.

Csgo: Content-style composition in text-to-image generation. arXiv preprint arXiv:2408.16766, 2024. 3, 5, 6

[52] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ipadapter: Text compatible image prompt adapter for text-toimage diffusion models. arXiv preprint arXiv:2308.06721, 2023. 2, 3, 5

[53] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3836–3847, 2023. 2

[54] Yuxin Zhang, Nisha Huang, Fan Tang, Haibin Huang, Chongyang Ma, Weiming Dong, and Changsheng Xu. Inversion-based style transfer with diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10146–10156, 2023. 3