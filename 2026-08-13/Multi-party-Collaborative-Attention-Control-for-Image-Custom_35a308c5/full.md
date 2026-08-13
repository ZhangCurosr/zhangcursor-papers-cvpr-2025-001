# Multi-party Collaborative Attention Control for Image Customization

Han Yang<sup>1,2</sup> Chuanguang Yang<sup>1∗</sup> Qiuli Wang<sup>3</sup> Zhulin An<sup>1∗</sup> Weilun Feng<sup>1,2</sup> Libo Huang<sup>1</sup> Yongjun Xu<sup>1</sup>

<sup>1</sup>Institute of Computing Technology, Chinese Academy of Sciences, Beijing, China

<sup>3</sup>Department of Radiology, The First Affiliated Hospital of Army Medical University, Chongqing, China {yanghan22s, yangchuanguang, anzhulin, huanglibo, fengweilun24s, xyj}@ict.ac.cn {wangqiuli@tmmu.edu.cn}

## Abstract

The rapid advancement of diffusion models has increased the need for customized image generation. However, current customization methods face several limitations: 1) typically accept either image or text conditions alone; 2) customization in complex visual scenarios often leads to subject leakage or confusion; 3) image-conditioned outputs tend to suffer from inconsistent backgrounds; and 4) high computational costs. To address these issues, this paper introduces Multi-party Collaborative Attention Control (MCA-Ctrl), a tuning-free method that enables high-quality image customization using both text and complex visual conditions. Specifically, MCA-Ctrl leverages two key operations within the self-attention layer to coordinate multiple parallel diffusion processes and guide the target image generation. This approach allows MCA-Ctrl to capture the content and appearance ofspecific subjects while maintaining semantic consistency with the conditional input. Additionally, to mitigate subject leakage and confusion issues common in complex visual scenarios, we introduce a Subject Localization Module that extracts precise subject and editable image layers based on user instructions. Extensive quantitative and human evaluation experiments show that MCA-Ctrl outperforms existing methods in zero-shot image customization, effectively resolving the mentioned issues.

## 1. Introduction

Recent advances in generative artificial intelligence (GenAI) have greatly enhanced text-to-image (T2I) models [8–10, 15, 23, 24, 26, 27, 33, 36], enabling them to generate realistic images from user prompts. As T2I models evolve, there has been an increasing demand for customized image creation [11, 18, 21, 28].

![](images/a5b1d70194cbcf0c7d44e7904a8d82fea10574b3810823d6b18def02a74a1973.jpg)  
Figure 1. The pipeline of MCA-Ctrl.

Image customization involves maintaining the identity and essence of a subject from a reference image while creating new representations under text or visual conditions. Traditionally, this has involved inverting the visual representation of the subject into a textual latent space and reconstructing new subject images through placeholders [11, 28]. However, this process often requires extensive fine-tuning or costly optimization for each subject. To address these challenges, certain approaches, such as IP-Adapter [35] and BLIP-Diffusion [19], have been developed to reduce training costs and enhance zero-shot performance by training a multimodal encoder and an alignment projection layer between image and text representations. BLIP-Diffusion [19] incorporates the transformed image representation into the prompt to guide image generation and editing. The series of works on IP-Adapter [35] treats the image representation as another form of prompt, employing the same cross-attention mechanism with text to introduce consistency.

However, whether subject representation is derived through inversion or a multimodal encoder, several limitations remain: (1) Lower controllability, primarily textdriven. Some works [5, 11, 16, 28, 35] are driven solely by text, which introduce uncertainties in the background, layout, and other elements. Some recent studies [12, 20] suggest using image condition to enhance control over background and custom regions. However, these approaches are often limited to single applications, focusing solely on either swapping or addition, thus restricting their applicability. (2) Subject leakage or confusion in complex visual conditions. We consider complex visual scenes to include object interactions, occlusions, multiple objects, and similarities between foreground and background. In these cases, inaccuracies in high-response regions during model generation will lead to subject leakage and confusion. (3) Poor background consistency under image conditions. (4) High fine-tuning costs for inversion-based approaches and lower subject consistency for adapter-based methods. Therefore, as shown in Figure 1, this paper seeks to explore a customization method compatible with both text and image conditions, low computational costs, and high quality.

(A) Subject Generation (text condition)  
![](images/9678bc4b6e07686af21ad2be68bfc0d51d60d396945b3aef120743a4734ddacd.jpg)  
Figure 2. Customized results from MCA-Ctrl. Without any fine-tuning or training, MCA-Ctrl can be used for text-driven subject image generation and image-driven subject image editing. Our method achieves high-quality customization across animals, people, and objects, preserving the distinctive features of specified subjects and meeting users’ specific requirements.

To achieve this goal, this paper introduces Multi-party Collaborative Attention Control (MCA-Ctrl), a tuning-free framework that enables controllable image customization under text or image conditions. Specifically, as shown in Figure 2, MCA-Ctrl can perform three types of tasks: subject generation, subject swapping, and subjet addition. The generation task is text-driven, while the swapping and addition tasks are image-driven. Built upon Stable Diffusion, MCA-Ctrl manipulates three flexible parallel diffusion processes within the self-attention layers to control the generation of the target image. These three diffusion processes are the subject diffusion process, the target image diffusion process, and the condition diffusion process, with the latter operating differently based on the form of the condition (text or image). Two distinct feature interaction operations within the self-attention layers are included: Self-Attention Global Injection (SAGI) and Self-Attention Local Query (SALQ). SALQ initiates from the target image, querying key information from the subject and conditional information. SAGI starts from the subject and conditional information, injecting the necessary visual features into the target image generation process. The combination of these two operations allows the model to maintain high consistency with both the subject and conditional information without requiring fine-tuning. To tackle subject leakage and confusion in complex visual scenarios, we introduce a Subject Localization Module (SLM) that processes multi-modal instructions. This module refines the model’s high-response regions, improving MCA-Ctrl’s image generation quality.

Our main contributions are as follows:

• We introduce MCA-Ctrl, a tuning-free method that achieves high-quality image customization under both text and image conditions, outperforming previous approaches in quantitative metrics and human evaluations.

• We propose two complementary attention control strategies that enable the generated images to maintain high consistency with both the target subject and the conditional information simultaneously.

• We present a Subject Localization Module (SLM) that corrects the high-response regions of the model in complex visual scenarios, reducing artifacts caused by feature confusion.

## 2. Related Works

## 2.1. Image Editing with Diffusion Models

Recently, the text-to-image latent diffusion models proposed enable the most advanced performance in image generation [27]. These models are trained on large-scale imagetext pairs datasets and can generate images guided by opendomain text descriptions.

Given an image-text pair $I _ { s }$ and $P _ { \mathrm { { : } } }$ , the latent diffusion model first converts $I _ { s }$ into a feature z in the latent space through an autoencoder and then, as shown in Equ.(1), Gaussian noise is progressively added to $z _ { 0 }$ through a predefined Markov chain, where $\beta _ { t }$ represents the scheduler. By converting with $\begin{array} { r } { \alpha _ { t } = \prod _ { s = 1 } ^ { t } ( 1 - \beta _ { s } ) } \end{array}$ , we can use Equ.(2) to transform $z _ { \mathrm { 0 } }$ to $z _ { t }$ at any time.

$$
q ( z _ { t } | z _ { t - 1 } ) = \mathcal { N } ( z _ { t } ; \sqrt { 1 - \beta _ { t } } z _ { t - 1 } , \beta _ { t } \mathbf { I } )\tag{1}
$$

$$
q ( z _ { t } | z _ { 0 } ) = \mathcal { N } ( z _ { t } ; \sqrt { \alpha _ { t } } z _ { 0 } ; ( 1 - \alpha _ { t } ) \mathbf { I } )\tag{2}
$$

Finally, the $z _ { t }$ is transformed into a high-resolution image $I _ { t }$ by optimizing the following objectives:

$$
\mathcal { L } ( \theta ) = \mathbb { E } _ { t \sim \mathcal { U } ( 1 , T ) , \epsilon _ { t } \sim \mathcal { N } ( 0 , \mathbf { I } ) } | | \epsilon _ { t } - \epsilon _ { \theta } ( z _ { t } , t , P ) | | ^ { 2 }\tag{3}
$$

$\epsilon _ { \theta }$ generally refers to a network with UNet architectures that interact with text prompt $P$ through cross-attention mechanisms at different resolutions. In inference, random noise is selected from the Gaussian distribution $z _ { T } \sim \mathcal { N } ( 0 , \bf { I } )$ , and the corresponding image is generated under the guidance of the given text description. Based on the text-to-image models, text-driven image editing has been proposed. These works can be roughly divided into two categories. One category, such as InstructPix2Pix [1], mainly constructs instruction-based image pair datasets $( I _ { s } , I _ { t } , P )$ to train latent diffusion models for editing purposes, where $I _ { t }$ is the ideal editing result of $I _ { s }$ under the guidance of $P .$ . The second type is to achieve image editing by controlling crossattention or self-attention, such as Prompt-to-Prompt [13], MasaCtrl [2] and so on.

When editing the real image R, we need to invert the image into the latent space to obtain the $z _ { T }$ corresponding to R [30], and then repeat the denoising process for more detailed image editing.

## 2.2. Image Customization

As image generation models advance, the demand for customization has grown. Customization involves incorporating user-provided conditions, like images or text, into generated outputs. Methods such as Textual Inversion [11] and Dreambooth [28] align the visual features of user-provided images with specific text placeholders to create custom content. However, these methods require extensive fine-tuning for each subject and offer limited control over layout and background. BLIP-Diffusion [19] and IP-Adapter [35] train a projection layer using large image-text datasets to align text and image features, enabling some zero-shot generation capabilities in the trained model. However, this still involves significant storage and training costs.

Prompt-to-Prompt [13] and MasaCtrl [2] highlight the rich semantic information embedded in cross-attention and self-attention layers, leading to new methods [7, 12, 20, 21] for incorporating custom information through attention control. Some works, like TIGIC [20] and PHOTOSWAP [12], use background-conditioned images for more complete customization. However, these methods often address single tasks, such as swapping, generation, or addition, and may struggle with subject confusion and leakage in complex visual conditions, limiting their applicability. This paper introduces a flexible multi-party collaborative control mechanism that handles all three customization tasks. Additionally, we propose a subject localization module to help the model more accurately recognize subjects in complex visual conditions, resulting in high-quality customized outputs.

## 3. Method

We propose Multi-party Collaborative Attention Control (MCA-Ctrl), a method that uses the knowledge inside the diffusion model for general image customization without fine-tuning. Its core idea is to combine the semantic information of the condition image or text prompt with the content in the subject image for a novel rendition of a specific subject. Specifically, we capture the visual appearance representation of a particular subject while preserving the spatial layout of the condition through self-attention injection and query in three parallel diffusion processes. This task is highly challenging, and most existing customization models often require extremely costly training [11, 16, 19, 28, 35].

Overall Pipeline. The overall pipeline for editing and generating by MCA-Ctrl is shown in Figure 3. MCA-Ctrl includes three diffusion processes: subject diffusion process $\begin{array} { r } { B _ { s u b } . } \end{array}$ , condition diffusion process $B _ { c o n }$ , and target diffusion process $\mathcal { B } _ { t g t } . ~ \mathcal { B } _ { s u b }$ receives the real subject image $I _ { s u b }$ and generates the diffusion initial feature $Z _ { T } ^ { s u b }$ through a DDIM inversion [30]. $B _ { c o n }$ receives the real source image $I _ { c o n }$ or the text prompt $P _ { T } .$ . As shown in Figure $3 \ ( \mathrm { A } )$ and (B), for $I _ { c o n } ,$ we get $Z _ { T } ^ { c o n }$ the same as $\begin{array} { r } { B _ { s u b } ; } \end{array}$ for $P _ { T } ,$ , we generate a random Gaussian distribution as $Z _ { T } ^ { c o n }$ $B _ { t g t }$ is a generation process that shares $Z _ { T } ^ { c o n }$ with a potential spatial layout as an initial feature to generate a target image $I _ { T }$ . At each diffusion step, we selectively perform the following operations: 1) Inject the foreground self-attention map and background self-attention map of $\boldsymbol { B } _ { s u b }$ and $B _ { c o n }$ into $B _ { t g t }$ , called Self-Attention Global Injection (SAGI). 2) $B _ { t g t }$ queries the subject appearance and background content from $\boldsymbol { B _ { s u b } }$ and $B _ { c o n }$ , called Self-Attention Local Query (SALQ). The details of SAGI and SALQ are in Section 3.2 and 3.1.

Subject Location Module. To prevent query confusion and subject feature artifacts in complex visual scenes with multiple similar objects, we introduce a Subject Location Module (SLM) to locate user-specified objects precisely. The SLM consists of an object detection model, DINO [22], and a segmentation model, SAM [17]. It processes multimodal information, such as a subject image $I _ { s u b }$ paired with textual prompts $P _ { s u b }$ and source images $I _ { c o n }$ paired with text descriptions $P _ { c o n }$ of regions to be edited. After localization and segmentation, the SLM outputs a binary subject image layer $M _ { C } ^ { s }$ and an editable image layer $M _ { S }$ . To ensure the edited region has sufficient space to blend with the background and avoid rigid transitions, we dilate $M _ { C } ^ { s }$ to

![](images/50a4373c74c010268ea36de65aeb1d8d5f990e31f54360f146bb2f5eab0565f5.jpg)  
Figure 3. Overview of the proposed MCA-Ctrl. Our method customizes images through self-attention cooperative control across three parallel diffusion processes, eliminating the need for fine-tuning. Figures (A) and (B) illustrate the inference pipeline of MCA-Ctrl unde image and text conditions, while (C) and (D) show details of self-attention local query and self-attention global injection.

$M _ { C }$ using a dilation kernel m with a size of $3 \times 3 .$

## 3.1. Self-Attention Local Query (SALQ)

From the perspective of the task, our goal is to extract the appearance features of the subject from the subject image $I _ { s u b }$ and query the background content and semantic layout from the condition $I _ { c o n }$ or $P _ { T }$ . By sharing the initial features of $B _ { c o n } .$ , the target image can basically form a spatial layout similar to $I _ { c o n }$ . Therefore, we focus on content queries from the condition. Inspired by MasaCtrl [2], the key feature K and value feature V of the self-attention layer can reflect the potential content representation of the image. Therefore, as shown in Figure 3 (C), at the denoising step t and layer $l , B _ { t g t }$ queries the foreground and background content from $\boldsymbol { B } _ { s u b }$ and $B _ { c o n }$ through the query feature $Q _ { T , t , l }$ of the self-attention layer.

Through Equ (4), we obtain the attention matrices $\mathcal { A } _ { T , C , t , l } , \mathcal { A } _ { T , S , t , l }$ of the target image to the global regions of the condition and subject image. To limit the query region and avoid confusion, we use $M _ { C }$ and $M _ { S }$ to mask the attention matrices locally, that is, to query foreground content only in the subject image and background content only in the condition. Then, according to Equ (5) and (6), we can obtain the queried foreground and background content features. Finally, we fused these two types of features through Equ (8). This operation serves two purposes: 1) $M _ { C }$ is employed to constrain the editable image region and ensure the layout consistency with the condition again; 2) Simultaneously query the foreground and background content, realizing the replacement of specific object’s appearances and enhancing the alignment of background content with the condition. $\mathcal { M F }$ stands for mask fill.

$$
\mathcal { A } _ { T , S , t , l } = \frac { Q _ { T , t , l } K _ { S , t , l } ^ { T } } { \sqrt { d } } , \mathcal { A } _ { T , C , t , l } = \frac { Q _ { T , t , l } K _ { C , t , l } ^ { T } } { \sqrt { d } }\tag{4}
$$

$$
\mathcal { F } _ { T , S , t , l } ^ { Q } = s o f t m a x ( A _ { T , S , t , l } * \mathcal { M } \mathcal { F } ( M _ { S } = 0 ) ) V _ { S , t , l }\tag{5}
$$

$$
\mathcal { F } _ { T , C , t , l } ^ { Q } = s o f t m a x ( A _ { T , C , t , l } \ast \mathcal { M } \mathcal { F } ( M _ { C } = 1 ) ) V _ { C , t , l }\tag{6}
$$

$$
\mathcal { F } _ { T , t , l } ^ { * } = M _ { C } * \mathcal { F } _ { T , C , t , l } ^ { Q } + ( 1 - M _ { C } ) * \mathcal { F } _ { T , S , t , l } ^ { Q }\tag{7}
$$

Unlike [2], we need the layout of the target image to follow the condition as closely as possible, so we recommend performing SALQ starting with the U-Net decoder in the early step.

## 3.2. Self-Attention Global Injection (SAGI)

After SALQ, we find that there are often two problems in generated images: 1) lack of authenticity in various details and 2) slight confusion with original features during the query process. We believe this is because the query process is essentially a local fusion of original and query features, inevitably leading to feature crossing and confusion. Therefore, we propose a global attention hybrid injection to enhance detail authenticity and content consistency of foreground and background.

As shown in Figure 3 (D), we first compute the attention matrices $\mathcal { A } _ { C , t , l }$ and $\mathcal { A } _ { S , t , l }$ for the condition and subject image according to Equ (8). Unlike SALQ, A here is the original attention matrix in the reconstruction of $B _ { c o n }$ and $\boldsymbol { B } _ { s u b }$ , including the mutual attention of all pixels in the image. Based on our goal, we also use $M _ { C }$ and $M _ { S }$ to filter $\mathcal { A } _ { C , t , l }$ and $\mathcal { A } _ { S , t , l }$ locally to focus on background and subject content. According to Equ (9) and (10), we can get the subject features and background features filtered by attention. Note that $\mathcal { F } _ { S , t , l } ^ { I }$ and $\mathcal { F } _ { C , t , l } ^ { I }$ here does not interact with the foreground content of the target process. We use Equ (11) to inject the subject features and background features into the target image’s diffusion process. By reconstructing the current feature output through replacement, we directly enhance foreground/background details while reducing feature confusion.

$$
\mathcal { A } _ { S , t , l } = \frac { Q _ { S , t , l } K _ { S , t , l } ^ { T } } { \sqrt { d } } , \mathcal { A } _ { C , t , l } = \frac { Q _ { C , t , l } K _ { C , t , l } ^ { T } } { \sqrt { d } }\tag{8}
$$

$$
\mathcal { F } _ { S , t , l } ^ { I } = s o f t m a x ( A _ { S , t , l } * \mathcal { M } \mathcal { F } ( M _ { S } = 0 ) ) V _ { S , t , l }\tag{9}
$$

$$
\mathcal { F } _ { C , t , l } ^ { I } = s o f t m a x ( A _ { C , t , l } * \mathcal { M } \mathcal { F } ( M _ { C } = 1 ) ) V _ { C , t , l }\tag{10}
$$

$$
\mathcal { F } _ { T , t , l } ^ { * } = M _ { C } * \mathcal { F } _ { C , t , l } ^ { I } + \left( 1 - M _ { C } \right) * \mathcal { F } _ { S , t , l } ^ { I }\tag{11}
$$

However, it should be noted that $\mathcal { F } _ { S , t , l } ^ { I }$ not only contains the content appearance but also the spatial layout information of the subject in $I _ { s u b }$ . Therefore, the location of SAGI needs to vary depending on the task. In subject editing, we want the subject image to inject content features without layout structure information into the target process, without destroying the spatial layout guided by the initial features $Z _ { T } ^ { t g t }$ and mask $M _ { C }$ . Therefore, we recommend performing SAGI in the early denoising step when the reconstructed composition of the condition and subject images has yet to generate mature spatial information. When doing subject generation, we want the subject content to be preserved completely, although some layout information is introduced. Therefore, we recommend continuously performing SAGI until later denoising steps.

## 3.3. Inference of MCA-Ctrl

The algorithm flow of image customization with image condition is shown in Algorithm 1. Assuming that the start and end steps of SAGI and SALQ are $S _ { G I } , E _ { G I } , S _ { L Q } , E _ { L Q } ,$ and the start layers are $L a y e r _ { G I }$ and Layer<sub>LQ</sub>, and the execution intervals of SAGI and SALQ do not cross.The EDIT function of Algorithm 1 at denoising step t and layer l is as follows:

$$
\mathbf { E D I T } : = \left\{ \begin{array} { l l } { S A G I , \mathrm { i f } \ S _ { G I } < t < E _ { G I } \ \mathrm { a n d } \ l > L a y e r _ { G I } } \\ { S A L Q , \mathrm { i f } \ S _ { L Q } < t < E _ { L Q } \ \mathrm { a n d } \ l > L a y e r _ { L Q } } \\ { S e l f - A t t e n t i o n ( \{ Q _ { T } , K _ { T } , V _ { T } \} ) , \ \mathrm { o t h e r w i s e } } \end{array} \right.\tag{12}
$$

Self-Attention represents the standard self-attention operation[31]. If the condition is text prompt, the acquisition of $M _ { C }$ is changed to extract from the cross attention of the corresponding step in $B _ { c o n }$ as shown in Figure 3 (B). Notably, although we present the inference of MCA-Ctrl as three parallel diffusion processes, this does not incur any additional computational cost. In the code implementation, these three parallel diffusion processes are handled as a single inference run with a batch size of 3.

Algorithm 1 The procedure of MCA-Ctrl for customization   
with image condition   
Require: A source text-image pair $( I _ { c o n } , P _ { c o n } ) ,$ a subject text  
image pair $( I _ { s u b } , P _ { s u b } ) ;$   
Ensure: a generate image $I _ { T }$   
1: $M s , M _ { C } = S L M ( ( I _ { c o n } , P _ { c o n } ) , ( I _ { s u b } , P _ { s u b } ) )$   
2: $\{ Z _ { T } ^ { c o n } , Z _ { T - 1 } ^ { c o n } , . . . , Z _ { 0 } ^ { c o n } \} = I n v e r s i o n ( I _ { c o n } )$   
3: $\{ Z _ { . T _ { . } } ^ { s u b } , Z _ { T - 1 } ^ { s u b } , . . . , Z _ { 0 } ^ { s u b } \} = I n v e r s i o n ( I _ { s u b } )$   
4: $\dot { Z } _ { T } ^ { t g t }  Z _ { T } ^ { c o n }$   
5: for $t = T , T - 1 , . . . , 1$ do   
6: $\{ Q _ { S } , K _ { S } , V _ { S } \}  \epsilon _ { \theta } ( Z _ { t } ^ { s u b } , t )$   
7: $\{ Q _ { C } , K _ { C } , V _ { C } \}  \epsilon _ { \theta } ( Z _ { t } ^ { c o n } , t )$   
8: $\{ Q _ { T } , K _ { T } , V _ { T } \} , \mathcal { F }  \epsilon _ { \theta } ( Z _ { t } ^ { t g t } , t )$   
9: $\dot { \mathcal { F } } ^ { * } \gets \bf { E D I T } ( \{ Q _ { T } , K _ { T } , V _ { T } \} , \{ Q _ { S } , K _ { S } , V _ { S } \} , \{ Q _ { C } , K _ { C } ,$   
$V _ { C } \} , M _ { S } , M _ { C } )$   
10: $\epsilon  \epsilon _ { \theta } ( Z _ { t } ^ { t g t } , t , \mathcal { F } ^ { * } )$   
11: $Z _ { t - 1 } ^ { t g t } \gets \dot { S } a m p l e ( \dot { Z } _ { t } ^ { t g t } , \epsilon )$   
12: end for   
13: return $Z _ { 0 } ^ { c o n } , Z _ { 0 } ^ { s u b } , Z _ { 0 } ^ { t g t }$

## 4. Experiment

## 4.1. Experimental Settings

Dataset. We utilize DreamBench [28] as the subject dataset, which consists of 30 subjects such as plush animals, dogs, cats, clocks, and robots. Then, we use DreamEdit-Bench [21] as the condition image dataset, providing ten editable real images for each subject in DreamBench. For subject generation, we employ 25 prompt templates from DreamBench to generate four images per prompt for model robustness assessment.

Metrics. We evaluate the images using three types of metrics: DINO [3] and CLIP-I [25] to assess image-toimage similarity, CLIP-T to evaluate image-to-text alignment, and ImageReward [32] to measure image aesthetic quality. Additionally, in subject swapping and addition tasks, we further divide DINO and CLIP-I into $\mathrm { D I N O } _ { s u b } ,$ $\mathrm { D I N O } _ { b a c k }$ , $\mathrm { C L I P - I } _ { s u b }$ , and ${ \mathrm { C L I P - I } } _ { b a c k } .$ , representing the consistency of the subject and background.

Setup. Our method utilizes the latest stable text-toimage diffusion model [27] with checkpoint v1.5. We employ DDIM deterministic inversion [30] for real image editing, converting images into initial noise maps. During sampling, we conduct 50 denoising steps of DDIM sampling with classifier-free guidance [14, 34] set to 7.5. Unless specified, SAGI is executed first, followed immediately by SALQ with no intermediate steps, meaning $S _ { L Q } = E _ { G I }$

Row#1: Subject Swapping (image condition)  
![](images/9c68db548a31ab21d3b0870e94f12eb8e586ff5c9e459f644b7af5187a4c9a9f.jpg)  
Figure 4. Qualitative result of MCA-Ctrl.  
Additionally, in all experimental validations, SAGI consistently performs better across all layers of the UNet, making $L a y e r _ { G I } ~ = ~ 1 6$ the default setting in our paper. In summary, our experiments focus on tuning four parameters: $S _ { G I }$ $E _ { G I }$ $L a y e r _ { L Q }$ , and $E _ { L Q }$ . These parameters can be adjusted for different classes to ensure more consistent editing and generation. For “Ours (Uniform)” in Table 1, we use the settings $S _ { G I } = 0$ $E _ { G I } = 2 0 $ $L a y e r _ { L Q } = 8 .$ and $E _ { L Q } = 4 8 $ . For “Ours (Uniform)” in Tables 2 and 3, we set $S _ { G I } = 0$ $E _ { G I } = 3 5 $ $L a y e r _ { L Q } = 0$ , and $E _ { L Q } = 4 8 $

## 4.2. Main Results

Main qualitative results. Figure 4 shows the qualitative editing and generated results of MCA-Ctrl. The first three rows primarily showcase subject editing performance, including subject swapping, subject addition, and subject swapping in complex visual scenes, demonstrating the high consistency and realism of MCA-Ctrl in both subject and background customization. Row#4 illustrates MCA-Ctrl’s zero-shot customization generation capabilities, achieving high-quality, consistent, and novel reproductions across objects, animals, and people. To further validate MCA-Ctrl’s editing capabilities in complex visual scenes, we categorize such scenarios into four types: Physical interactions between subjects, Similar subject and background, Occlusion, and Multiple objects. Figure 5 provides examples for each. The results show that MCA-Ctrl accurately captures the appearance of different subjects in complex scenes based on user instructions, enabling high-quality edits of specified subjects within multi-object conditions. Our model is unrestricted by manually curated datasets, allowing it to capture features from any subject in the diffusion process, with strong generalization and robustness.

Table 1. Quantitative comparisons on DreamEditBench of subject swapping. Ours (Uniform) means that all classes are tested with uniform parameters of $S _ { G I } , E _ { G I }$ $L a y e r _ { L Q }$ and $E _ { L Q } ;$ Ours (Specified) means to customize parameters for partial classes.
<table><tr><td>Methods</td><td>DINOsub↑</td><td>DINOback ↑</td><td>CLIP-Isub↑</td><td>CLIP-Iback↑</td><td>ImageReward</td></tr><tr><td>DreamBooth [28]</td><td>0.6400</td><td>0.4270</td><td>0.8110</td><td>0.7360</td><td>-1.1713</td></tr><tr><td>Customized-DiffEdit [6]</td><td>0.5100</td><td>0.7850</td><td>0.7550</td><td>0.8950</td><td>0.1375</td></tr><tr><td>DreamEditor(5) [21]</td><td>0.5640</td><td>0.6670</td><td>0.7700</td><td>0.8550</td><td>-0.5633</td></tr><tr><td>-iteration=1</td><td>0.5460</td><td>0.6640</td><td>0.7630</td><td>0.8530</td><td>-0.2731</td></tr><tr><td>BLIP-Diffusion [19]</td><td>0.6155</td><td>0.6392</td><td>0.8009</td><td>0.8248</td><td>0.2187</td></tr><tr><td>PHOTOSWAP [12]</td><td>0.6307</td><td>0.6072</td><td>0.7886</td><td>0.7977</td><td>-0.1982</td></tr><tr><td>Ours (Uniform)</td><td>0.6327±0.004</td><td>0.6684±0.004</td><td>0.7794±0.003</td><td>0.8621±0.005</td><td>0.2728±0.05</td></tr><tr><td>Ours (Specified)</td><td>0.6433±0.005</td><td>0.6782±0.002</td><td>0.8113±0.004</td><td>0.8681±0.004</td><td>0.3214±0.05</td></tr></table>

Comparison. Table 1 presents the quantitative automatic evaluation results for the subject swapping task assessed on DreamEditBench [21]. MCA-Ctrl demonstrates comparable or superior performance across all metrics relative to BLIP-Diffusion [19], DreamBooth [28] and PHOTO-SWAP [12]. Specifically, with uniform parameters, MCA-Ctrl achieves slightly higher scores than BLIP-Diffusion in $\mathrm { D I N O } _ { s u b }$ $\mathrm { D I N O } _ { b a c k }$ ${ \mathrm { C L I P - I } } _ { b a c k } ,$ CLIP-T, and ImageReward, while recording marginally lower scores than Dream-Booth in $\mathrm { D I N O } _ { s u b }$ . Upon adjusting parameters for some classes, MCA-Ctrl surpasses DreamBooth in $\mathrm { D I N O } _ { s u b }$ and ${ \mathrm { C L I P - I } } _ { s u b } ,$ thus indicating superior editing quality. As shown in Figure 7, as a training-free method, MCA-Ctrl outperforms PHOTOSWAP in capturing subject features while preserving the original layout and background content of the image. Detailed scores both before and after parameter adjustment for each subejct and the specific scheme for parameter adjustment are shown in Supplementary material. Note that, in the reported result Ours (Specified), we make only subtle adjustments to the execution steps of SAGI and the execution layers of SALQ for certain classes. Overall, these adjustments are easy to implement and not time-consuming.

![](images/982595910ad6b20c69fdb3b7df73a2474b608b2370df5d834c3f364beb06153d.jpg)

Figure 5. Editing results of MCA-Ctrl in complex visual condition.  
![](images/336e69b9e56de2440e74bf30553ba68d147b7c1363d9a13d2e324cc420f209c5.jpg)  
Figure 6. Comparison with other Subject-driven Generation Models.

Table 2. Automatic Evaluation on the DreamBench of subject generation.
<table><tr><td>Methods</td><td>DINO↑</td><td>CLIP-I↑</td><td>CLIP-T↑</td><td>ImageReward↑</td></tr><tr><td>DreamBooth [28]</td><td>0.6680</td><td>0.8430</td><td>0.3060</td><td>0.3839</td></tr><tr><td>Textual Inversion [11]</td><td>0.5690</td><td>0.7800</td><td>0.2550</td><td>-0.9788</td></tr><tr><td>Re-Imagen [4]</td><td>0.6000</td><td>0.7900</td><td>0.2700</td><td>-0.1765</td></tr><tr><td>BLIP-Diffusion [19]</td><td>0.6700</td><td>0.8250</td><td>0.3020</td><td>0.1829</td></tr><tr><td>IP-Adapter [35]</td><td>0.6504</td><td>0.8232</td><td>0.2651</td><td>-0.1782</td></tr><tr><td>FreeCustom [7]</td><td>0.6660</td><td>0.8363</td><td>0.2829</td><td>-1.1723</td></tr><tr><td>Ours (Uniform)</td><td>0.6610±0.002</td><td>0.8399±0.003</td><td>0.3022±0.002</td><td>0.3037±0.05</td></tr><tr><td>Ours (Specified)</td><td>0.6724±0.004</td><td>0.8441±0.003</td><td>0.3056±0.002</td><td>0.4132±0.06</td></tr></table>

Table 3. Human Evaluation on the DreamBench of subject-driven generation.
<table><tr><td>Methods</td><td>Backbone</td><td>Subject</td><td>Textual</td><td>Realistic↑</td><td>Overall</td></tr><tr><td>DreamBooth [28]</td><td>SD [27]</td><td>0.81</td><td>0.64</td><td>0.91</td><td>2.36</td></tr><tr><td>Textual Inversion [11]</td><td>SD [27]</td><td>0.44</td><td>0.76</td><td>0.86</td><td>2.06</td></tr><tr><td>Re-Imagen [4]</td><td>Imagen [29]</td><td>0.71</td><td>0.79</td><td>0.80</td><td>2.3</td></tr><tr><td>BLIP-Diffusion [19]</td><td>SD [27]</td><td>0.85</td><td>0.82</td><td>0.93</td><td>2.6</td></tr><tr><td>IP-Adapter [35]</td><td>SD [27]</td><td>0.85</td><td>0.84</td><td>0.94</td><td>2.63</td></tr><tr><td>FreeCustom [7]</td><td>SD [27]</td><td>0.87</td><td>0.82</td><td>0.81</td><td>2.6</td></tr><tr><td>Ours (Uniform)</td><td>SD[27]</td><td>0.88</td><td>0.84</td><td>0.85</td><td>2.57</td></tr><tr><td>Ours (Specified)</td><td>SD[27]</td><td>0.92</td><td>0.89</td><td>0.92</td><td>2.73</td></tr></table>

![](images/daed274862bc00f34d7abeac2e5d9eecf5410240dfe835b9ee4441ac705cac83.jpg)  
Figure 7. Qualitative comparison between MCA-Ctrl and PHO-TOSWAP on controllable subject editing.

Table 2 shows automatic evaluation results for the subject generation task on DreamBench. Initially, MCA-Ctrl performs better than Text Inversion, Re-Imagen, and IP-Adapter but slightly lower than DreamBooth and BLIP-

![](images/706d00d4f85e7f4e8e9436b015cb26d96b0eaf4d56c3ffb347e9e83053543046.jpg)  
Figure 8. Comparison between MCA-Ctrl and FreeCustom on character customization.

Table 4. Ablation results on DreamEditBench[21]. “reverse” means to reverse the execution order of SAGI and SALQ, executing SALQ before SAGI.
<table><tr><td>Ablation setups</td><td>DINOsub↑</td><td>DINOback↑</td><td>CLIP-Isub↑</td><td>CLIP-Iback↑</td><td>ImageReward↑</td></tr><tr><td>Ours (Uniform)</td><td>0.6327</td><td>0.6684</td><td>0.7794</td><td>0.8621</td><td>0.2728</td></tr><tr><td>- w/o SALQ</td><td>0.4238↓</td><td>0.7491↑</td><td>0.7416↓</td><td>0.8774↑</td><td>0.2454↓</td></tr><tr><td>- w/o SAGI</td><td>0.5896↓</td><td>0.6851↑</td><td>0.7746↓</td><td>0.8429↓</td><td>0.2716↓</td></tr><tr><td>- w/o mask dilation</td><td>0.5611↓</td><td>0.7319↑</td><td>0.7671↓</td><td>0.8754↑</td><td>0.2671↓</td></tr><tr><td>- w/o SLM</td><td>0.4914↓</td><td>0.8244↑</td><td>0.7532↓</td><td>0.8999↑</td><td>0.1911↓</td></tr><tr><td>- reverse</td><td>0.4585↓</td><td>0.5547↓</td><td>0.7230↓</td><td>0.8014↓</td><td>0.1076↓</td></tr></table>

![](images/4c25bc962988356674fa95781e62cc52914cc15344cf93fd74ad04437c8a57c3.jpg)  
Figure 9. Top: Quantitative ablation of $S _ { G I } , E _ { G I } , L a y e r _ { L Q }$ and $E _ { L Q } ;$ Bottom: Qualitative ablation results of SAGI and SALQ. Enlarged version please refer to Supplementary material.

Diffusion with uniform parameters. However, MCA-Ctrl with specified parameters achieves results comparable to those of BLIP-Diffusion and DreamBooth. Furthermore, Table 3 presents our human evaluation results on Dream-Bench, indicating that MCA-Ctrl demonstrates superior subject alignment and text alignment, slightly outperforming BLIP-Diffusion in overall score. As a training-free method, maintaining consistency with high-granularity subjects like character figures is quite challenging. As shown in Figure 8, FreeCustom struggles with errors in character customization, failing to accurately represent both the subject and background. In contrast, MCA-Ctrl overcomes this challenge through complementary multi-party collaborative control, achieving effective and accurate customization for character subjects.

Ablation Studies Table 4 shows the zero-shot ablation results of MCA-Ctrl on DreamBench. Figure 9 further shows quantitative and qualitative ablation of SAGI and

![](images/3f82222c41e36180a89bd7bb904df5dec1dbb15c29081034facccb417e0d2a33.jpg)  
Figure 10. Limitation of MCA-Ctrl.

SALQ related parameters. Combined with the chart, we find: a) SALQ is crucial. It guarantees the consistency of the generated image with the foreground appearance of the subject image, so it can significantly affect the $\mathrm { D I N O } _ { s u b }$ and $\mathrm { C L I P - I } _ { s u b }$ scores. b) SAGI can further improve the authenticity of the edited image in every detail and can correct the feature obfuscations caused by SALQ (the orange feature of the cat’s mouth in Figure 9), resulting in modest improvements in most metrics. c) SLM can help position the specified objects when the background of the subject image or edited image is complex to improve the confusion between the foreground and background and the quality of the generated image. d) The execution of SALQ from the self-attention mechanism of the encoder (0-7 layers) may cause image deformation since the layout is not yet formed. Starting from the low-resolution layer of the decoder (8-16 layers), it can inject subject features while maintaining the design of the source image. With the increase of the starting layer, the subject characteristics gradually weaken. e) For the subject editing, SAGI is suitable for earlier steps, emphasizing semantic information about the foreground and background at the beginning of editing. Performing too many steps may cause the layout of the generated image foreground to be too close to the subject image.

In general, although adding certain modules may reduce consistency with the source image, qualitative and quantitative results show significant improvement in consistency with the subject image, making these trade-offs acceptable.

Discussion As shown in Figure 10, through extensive validation, we found that MCA-Ctrl is constrained by the base model and encounters difficulties in certain cases: (1) when the subject image contains fine-grained features, such as text; (2) when color changes are applied, there may be issues where the color change only affects the subject’s local regions. Addressing these issues will be a focus of our future work.

## 5. Conclusion

This paper presents MCA-Ctrl, a tuning-free generation method for image customization. The model achieves highquality and high-fidelity subject-driven editing and generation through coordinated attention control among three parallel diffusion processes. In addition, MCA-Ctrl solves the feature obfuscation problem in complex visual scenes by introducing a Subject Localization Module. Many experimental results show that MCA-Ctrl performs better editing and generation than most fine-tuning models.

## 6. Acknowledgments

This work is partially supported by the National Natural Science Foundation of China under Grant Number 62476264 and 62406312, the Postdoctoral Fellowship Program and China Postdoctoral Science Foundation under Grant Number BX20240385 (China National Postdoctoral Program for Innovative Talents), the Beijing Natural Science Foundation under Grant Number 4244098, and the Science Foundation of the Chinese Academy of Sciences.

## References

[1] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18392–18402, 2023. 3

[2] Mingdeng Cao, Xintao Wang, Zhongang Qi, Ying Shan, Xiaohu Qie, and Yinqiang Zheng. Masactrl: Tuning-free mutual self-attention control for consistent image synthesis and editing. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 22560–22570, 2023. 3, 4

[3] Mathilde Caron, Hugo Touvron, Ishan Misra, Herve J´ egou,´ Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings ofthe IEEE/CVF international conference on computer vision, pages 9650–9660, 2021. 5

[4] Wenhu Chen, Hexiang Hu, Chitwan Saharia, and William W Cohen. Re-imagen: Retrieval-augmented text-to-image generator. arXiv preprint arXiv:2209.14491, 2022. 7

[5] Wenhu Chen, Hexiang Hu, Yandong Li, Nataniel Ruiz, Xuhui Jia, Ming-Wei Chang, and William W Cohen. Subject-driven text-to-image generation via apprenticeship learning. Advances in Neural Information Processing Systems, 36, 2024. 1

[6] Guillaume Couairon, Jakob Verbeek, Holger Schwenk, and Matthieu Cord. Diffedit: Diffusion-based semantic image editing with mask guidance. arXiv preprint arXiv:2210.11427, 2022. 6

[7] Ganggui Ding, Canyu Zhao, Wen Wang, Zhen Yang, Zide Liu, Hao Chen, and Chunhua Shen. Freecustom: Tuningfree customized image generation for multi-concept composition. IEEE, 2024. 3, 7

[8] Weilun Feng, Chuanguang Yang, Zhulin An, Libo Huang, Boyu Diao, Fei Wang, and Yongjun Xu. Relational diffusion distillation for efficient image generation. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 205–213, 2024. 1

[9] Weilun Feng, Haotong Qin, Chuanguang Yang, Zhulin An, Libo Huang, Boyu Diao, Fei Wang, Renshuai Tao, Yongjun Xu, and Michele Magno. Mpq-dm: Mixed precision quantization for extremely low bit diffusion models. In Proceedings ofthe AAAI Conference on Artificial Intelligence, 2025.

[10] Yutong Feng, Biao Gong, Di Chen, Yujun Shen, Yu Liu, and Jingren Zhou. Ranni: Taming text-to-image diffu-

sion for accurate instruction following. arXiv preprint arXiv:2311.17002, 2023. 1

[11] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. An image is worth one word: Personalizing text-toimage generation using textual inversion. arXiv preprint arXiv:2208.01618, 2022. 1, 3, 7

[12] Jing Gu, Yilin Wang, Nanxuan Zhao, Tsu-Jui Fu, Wei Xiong, Qing Liu, Zhifei Zhang, He Zhang, Jianming Zhang, Hyun Joon Jung, et al. Photoswap: Personalized subject swapping in images. Advances in Neural Information Processing Sys tems, 36, 2024. 1, 3, 6

[13] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt im age editing with cross attention control. arXiv preprint arXiv:2208.01626, 2022. 3

[14] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 5

[15] Hexiang Hu, Kelvin CK Chan, Yu-Chuan Su, Wenhu Chen, Yandong Li, Kihyuk Sohn, Yang Zhao, Xue Ben, Boqing Gong, William Cohen, et al. Instruct-imagen: Image generation with multi-modal instruction. arXiv preprint arXiv:2401.01952, 2024. 1

[16] Mengqi Huang, Zhendong Mao, Mingcong Liu, Qian He, and Yongdong Zhang. Realcustom: Narrowing real text word for real-time open-domain text-to-image customization. In Proceedings of the IEEE/CVF Conference on Com puter Vision and Pattern Recognition, pages 7476–7485, 2024. 1, 3

[17] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF International Con ference on Computer Vision, pages 4015–4026, 2023. 3

[18] Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu. Multi-concept customization of text-to-image diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1931–1941, 2023. 1

[19] Dongxu Li, Junnan Li, and Steven Hoi. Blip-diffusion: Pretrained subject representation for controllable text-to-image generation and editing. Advances in Neural Information Processing Systems, 36, 2024. 1, 3, 6, 7

[20] Pengzhi Li, Qiang Nie, Ying Chen, Xi Jiang, Kai Wu, Yuhuan Lin, Yong Liu, Jinlong Peng, Chengjie Wang, and Feng Zheng. Tuning-free image customization with image and text guidance. arXiv preprint arXiv:2403.12658, 2024. 1, 3

[21] Tianle Li, Max Ku, Cong Wei, and Wenhu Chen. Dreamedit: Subject-driven image editing. arXiv preprint arXiv:2306.12624, 2023. 1, 3, 5, 6, 8

[22] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. arXiv preprint arXiv:2303.05499, 2023. 3

[23] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and

Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741, 2021. 1

[24] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Muller, Joe Penna, and¨ Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023. 1

[25] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 5

[26] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1 (2):3, 2022. 1

[27] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High-resolution image¨ synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 1, 2, 5, 7

[28] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22500– 22510, 2023. 1, 3, 5, 6, 7

[29] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in neural information processing systems, 35:36479–36494, 2022. 7

[30] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020. 3, 5

[31] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017. 5

[32] Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: Learning and evaluating human preferences for textto-image generation. Advances in Neural Information Processing Systems, 36, 2024. 5

[33] Chuanguang Yang, Zhulin An, Libo Huang, Junyu Bi, Xinqiang Yu, Han Yang, Boyu Diao, and Yongjun Xu. Clip-kd: An empirical study of clip model distillation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15952–15962, 2024. 1

[34] Han Yang, Chuanguang Yang, Zhulin An, Libo Huang, and Yongjun Xu. Hsrdiff: A hierarchical self-regulation diffusion model for stochastic semantic segmentation. In Proceedings ofthe AAAI Conference on Artificial Intelligence, 2025. 5

[35] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ipadapter: Text compatible image prompt adapter for text-to-

image diffusion models. arXiv preprint arXiv:2308.06721, 2023. 1, 3, 7

[36] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3836–3847, 2023. 1