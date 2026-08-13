# From Elements to Design: A Layered Approach for Automatic Graphic Design Composition

Jiawei Lin<sup>1</sup>, Shizhao Sun<sup>2</sup>, Danqing Huang<sup>2</sup>, Ting Liu<sup>1</sup>, Ji Li<sup>2</sup>, Jiang Bian<sup>2</sup> <sup>1</sup>Xi’an Jiaotong University, <sup>2</sup>Microsoft Research

kylelin@stu.xjtu.edu.cn, tingliu@mail.xjtu.edu.cn, {shizsu, dahua, jili5, jiang.bian}@microsoft.com

![](images/f8bcd53c802d1aa04aa0782c4a10801434d572571846e9051a1c36cc6222326f.jpg)  
(c) Generated Graphic Designs  
Figure 1. (a) Given a set of multimodal elements as input, our approach automatically composes them into a cohesive, balanced, and aesthetically pleasing graphic design. (b) Since a holistic design can be divided into different layers according to element semantics, we achieve the design composition task in a layer-by-layer manner. (c) Our approach is able to craft high-quality design pieces.

## Abstract

In this work, we investigate automatic design composition from multimodal graphic elements. Although recent studies have developed various generative models for graphic design, they usually face the following limitations: they onlyfocus on certain subtasks and arefarfrom achieving the design composition task; they do not consider the hierarchical information ofgraphic designs during the generation process. To tackle these issues, we introduce the layered design principle into Large Multimodal Models (LMMs) and propose a novel approach, called LaDeCo, to accomplish this challenging task. Specifically, LaDeCo first performs layer planning for a given element set, dividing the input elements into different semantic layers according to their contents. Based on the planning results, it subsequently predicts element attributes that control the design composition in a layer-wise manner, and includes the rendered image of previously generated layers into the context. With this insightful design, LaDeCo decomposes the difficult task into smaller manageable steps, making the genera-

tion process smoother and clearer. The experimental results demonstrate the effectiveness of LaDeCo in design composition. Furthermore, we show that LaDeCo enables some interesting applications in graphic design, such as resolution adjustment, design decoration, design variation, etc. In addition, it even outperforms the specialized models in some design subtasks without any task-specific training.

## 1. Introduction

Graphic design is an artistic discipline dedicated to creating visual content that attracts attention and communicates messages effectively. Creating visually appealing designs today relies on human designers with both artistic creativity and technical expertise to skillfully integrate multimodal graphic elements like images, headlines, decorative embellishments, and so on. This is a complex and timeconsuming process that requires careful consideration of many aspects. For example, as shown in Figure 1a, it is important to ensure that the main object (i.e., pizza) is not obscured by other elements. For readability, there should be sufficient contrast between the text and the underlay. Additionally, designers also need to adjust the element sizes to make the design balanced. In this work, we refer to the challenging process of composing a set of elements into a holistic design as design composition (see Figure 1a).

To ease the burden on human designers, recently, there has been a growing interest in developing generative models to streamline this process. Most existing work focuses on certain typical subtasks of design composition. For example, some previous approaches investigate content-aware layout generation [7, 8, 28, 34, 38], which aims to automatically arrange graphic elements on a given canvas while ensuring that the main object remains unobstructed. Although these methods are capable of creating high-quality layouts, they typically only consider the background image content while overlooking the content of other elements. In addition, they do not predict text-related attributes during the layout generation process, limiting their ability to produce fully integrated designs. Another popular subtask is called typography generation [5, 11, 13, 27, 30]. Its goal is to generate font, color, size, and other attributes for text elements, enhancing both aesthetics and readability. However, it ignores the visual elements in graphic designs. Together, all these studies fall short of holistic design creation. Consequently, users have to manually integrate models of different functions to achieve design composition, which brings high costs and unnecessary obstacles.

To be best of our knowledge, FlexDM [11] is the only attempt towards automatic design composition. By representing graphic designs as a flat combination of element attributes, FlexDM formulates various design tasks as masked field prediction problems, including the design composition task. While the flat representation provides practicality and versatility, it has a notable limitation: it overlooks the inherent hierarchical structure within graphic designs. This hierarchical structure arises because human designers usually follow the layered design principle. Specifically, it involves arranging elements in separate semantic layers, starting with a background and then gradually adding underlays, images, texts and small embellishments (see Figure 1b). Such layered principle brings two main benefits. First, each preceding layer provides a strong foundation for designing subsequent layers, aiding cohesion. Second, grouping similar elements by layer clarifies the design process and enhances workflow efficiency.

Based on the insight, we propose LaDeCo (see Figure 2), a layered design composition method. To support the layered mechanism, we develop a layer planning module. Specifically, we carefully design the task prompts and leverage GPT-4o [1] to predict the semantic label for each input element by considering its content. Elements sharing the same label are placed on the same layer, thus reaching layer planning. Subsequently, LaDeCo divides the generation process into several steps according to the layer planning results. Considering that input elements are inherently multimodal and that design composition can be formulated by sequentially predicting attributes for each element, we choose Large Multimodal Models (LMMs) [2, 3, 23, 31, 39] as the backbone. At each step, LMMs are asked to predict element attributes within a single layer. After each step, the previously generated layers will be rendered as an intermediate design image and fed back into the LMMs, providing contextual information for following layer prediction.

Thanks to the novel design, LaDeCo is able to decompose the challenging task into smaller manageable steps. At each step, the model only focuses on design composition of the current layer, making the process more accessible. Additionally, by rendering intermediate designs and adding them into the context, the model can better generate subsequent layers based on previous layers. The layerwise generation also offers flexibility that allows LaDeCo to support certain subtasks of design composition without any taskspecific training, as shown in Section 4.5.

To validate the effectiveness of our approach, we conduct experiments on a publicly available dataset Crello [33] to compare it with the state-of-the-art baselines. Both quantitative and qualitative experimental results show that LaDeCo significantly outperforms the baseline models in design composition. Through ablation studies, we demonstrate the effectiveness of the layer planning module and layered design composition introduced in LaDeCo. Besides, we show that LaDeCo enables some interesting applications such as resolution adjustment, design decoration, and design variation, further showcasing the practicality and versatility of LaDeCo. In addition, LaDeCo even surpasses the specialized models in two subtasks: content-aware layout generation and typography generation.

## 2. Related Work

Graphic Design Composition. There are three aspects of relevant research on graphic design composition: contentaware layout generation, typography generation, and the holistic design composition. We’ll introduce them below.

Content-Aware Layout Generation [4, 7, 8, 22, 28, 34]. It is an emerging research topic that studies layout generation conditioned on a given canvas. LayoutPrompter [22] proposes a RAG-based approach to retrieve training samples with the most similar saliency bounding boxes as prompts, and achieves content-aware layout generation via in-context learning. PosterLlama [28] introduces a LLM-based twostage training method for this task. It first keeps the backbone parameters fixed and trains the adapter, and then finetunes the backbone to generate layouts in the HTML format. PosterLLaVA [34] further improves practicality by supporting more user requirements (e.g., element relationships). Compared to our work, these existing methods do not consider element content, nor predict text attributes, and thus cannot create a complete design.

![](images/02927ee05fd834d95be99251be1ce6037f0a369a61c70703744a8172427e3b73.jpg)  
Figure 2. Illustration of our proposed LaDeCo. First, it utilizes GPT-4o [1] to annotate the semantic labels for input elements. The layer structure is obtained from the predictions. Then LaDeCo fine-tunes LMMs to achieve layered design composition. After generating each layer, the intermediate designs will be rendered as images and fed back into LMMs to guide subsequent layer generation.

Typography Generation [5, 11–13, 27]. This task studies visually compelling, harmonious text rendering on a graphic design. Some relevant work [5, 27] regards text styles as images and formulates typography generation as text image generation in the pixel space. For editability, some other work directly predicts editable text attributes (e.g., font type, color, size) and utilizes the renderer to visualize stylized text elements. FlexDM [11] achieves this by masked field prediction, generating all attributes in a single pass. COLEs [12, 13] propose Typography-LMM to autoregressively predict text attributes based on the input canvas. However, these methods still can not craft holistic graphic designs since they do not consider the visual elements.

Design Composition. FlexDM [11] accomplishes design composition by simultaneously masking and predicting element attributes. However, it treats all input elements equally and does not incorporate the critical layered information during the generation process. Another relevant work is VLC [29], which adopts an image-vector dual diffusion model for design generation. One of its issues is that it represents text elements as visual modalities, and uses the ground truth values for all text attributes. Besides, it also does not take into account hierarchical information. In summary, we are the first to introduce layered design composition and develop a novel approach based on it.

Other Topics in Graphic Design. Earlier, some studies investigate content-agnostic layout generation. To meet diverse user needs, previous methods have considered various input conditions, such as element attributes [10, 14, 15, 17, 20, 22, 36], element relationships [15, 18], textual descriptions [21, 22], and more. However, the contentagnostic nature makes them unable to adapt to the input elements, in which element deformations often occur. Recently, there is an interesting topic studying design content generation [12, 13, 16]. We emphasize that we focus on design composition from user-provided elements rather than generate the element content as presented in their work.

Large Multimodal Models (LMMs). LMMs have shown strong capabilities in understanding multimodal contexts and generating plausible responses across diverse domains [1, 2, 23, 25, 35, 39]. In this work, we leverage them to effectively achieve design composition from multimodal input elements. We introduce the layered design principle in our approach, enabling LMMs to tackle different semantic layers iteratively instead of generating all layers at once. To the best of our knowledge, we are the first work to adopt LMMs for layered design generation. Furthermore, our layerwise generation method also reflects the idea of chain-of thought (CoT) reasoning [32], which has been proven to be effective in enhancing reasoning performance [26, 37].

## 3. LaDeCo

## 3.1. Problem Formulation

The input and output of design composition are a set of multimodal elements and their attributes, respectively. After predicting the attributes, we can adopt an off-the-shelf renderer to render the input elements into a graphic design, thereby achieving design composition. In this work, input elements are categorized into two modalities: image modality and text modality. For image modality elements, the attributes are four bounding box parameters, i.e., left and top coordinates, element width and height. For text modality elements, we consider eight more attributes, namely angle, font, font size, color, text alignment, capitalization, letter spacing, and line height. We empirically find that these attributes are sufficient to describe a high-quality design.

## 3.2. Method Overview

In this work, we take inspiration from the layered design principle to decompose a holistic graphic design into different layers and progressively create these layers to reach the complete design, making the design composition process smoother and clearer. Here, a layer is a collection of graphic elements with same semantic labels. To be more specific, our method includes two key techniques, namely a layer planning module and a layered design composition process. The layer planning module is responsible for categorizing input elements into pre-defined layers. Then, in the layered design composition process, our approach predicts element attributes in a layerwise manner and gathers them together to get the complete attributes.

## 3.3. Layer Planning

The very first step here is to determine a reasonable layer structure. By examining numerous completed design pieces and consulting experienced designers, in this work, we consider 5 design layers, namely background, underlay, logo/image, text, and embellishment layers in the placement order (see Figure 1b). By sequentially rendering these layers on the empty canvas $G _ { 0 }$ , we obtain $G _ { 1 }$ through $G _ { 5 }$ (see Figure 4), where $G _ { 1 }$ represents only the background layer, $G _ { 2 }$ includes the background layer plus the underlay layer, and so forth, with $G _ { 5 }$ representing the complete, finalized design. Notably, the layer structure is not limited to our proposed one. There is flexibility to add or remove some as long as it is reasonable.

Although the publicly available dataset does not contain any layer information for the elements, we find it is feasible to infer from element content. For example, a solid colored rectangular box might be an underlay, and a element with a star is an embellishment. Therefore, we formulate layer planning as an element content understanding problem and leverage pre-trained LMMs to resolve it. In the implementation, we employ GPT-4o [1] to automatically generate input element labels (see Figure 2a), thereby achieving layer planning. To guide GPT-4o effectively, we carefully craft prompts, clearly defining the problem, describing the characteristics of each element label, and requesting the model to output the most appropriate one for input elements. For training samples where ground truth designs are available, we also include the design images and some metadata (e.g., the canvas size, element size) into the model input to enhance the prediction accuracy. For full details of the prompt text, please refer to the supplementary materials.

## 3.4. Layered Design Composition

Our main idea here is to generate element attributes in a layerwise manner from background to embellishment layers, according to the layer planning results in Section 3.3. The rendered images of previous layers are incorporated in the context for the prediction of the subsequent layers. It is noteworthy that this process requires strong understanding capabilities of element content. For example, as shown in the row 3, column 1 of Figure 3, the text element with an email address should be placed at the bottom of the canvas, and the barbershop logo should be placed at the top. Meanwhile, the understanding of intermediate results is also critical. For example, in the same figure, the model should avoid blocking the persons by understanding the intermediate rendered image. To this end, we resort to LMMs, which possess remarkable understanding capabilities, to model the layered design process for multimodal design elements. As shown in Figure 2b, in the i-th design layer, LMMs predict the attribute set $Y _ { i }$ for current layer’s elements $X _ { i }$ . The preceding layers are rendered as an image $G _ { i - 1 }$ and reintroduced into the LMMs to be part of the contextual input, guiding the generation of $Y _ { i }$ . Next, we will detail the representation of $X _ { i }$ and $Y _ { i } ,$ as well as the model architecture.

Representation of $X _ { i }$ and $Y _ { i } .$ In general, we represent $X _ { i }$ and $Y _ { i }$ by concatenating the element content and attributes of current layer, respectively. For background, underlay, logo/image, and embellishment layers, the inputs are all visual elements. Hence, $X _ { i }$ is a combination of element images (i.e., pixel values). For the text layer, we concatenate its text content together to construct $X _ { i }$ (see Figure 2b). In terms of $Y _ { i } ,$ , we follow the trend of structured representation in existing approaches [12, 22, 28], and serialize the element attributes into JSON strings, as implemented in Open-COLE [12]. Notably, within each layer, the elements are randomly shuffled to prevent information leakage $( \mathrm { e . g . }$ ., the top-down placement order). In the design layers that do not have any elements, we represent their inputs as null and outputs as an empty JSON string {}. Please refer to the supplementary materials for an example of the input-output representation.

Model Architecture. The model consists of three components: a vision encoder, a projector and the LMM backbone. The vision encoder is responsible for encoding the element images and intermediate designs, generating image embeddings. The projector then projects these embeddings to match the hidden state dimension required by the backbone. Finally, the backbone is used to model the joint distribution across layers, ensuring cohesion in the layered design process. To reduce computational complexity, a 2D average pooling operation is applied to the output of vision encoder to compress the image tokens effectively.

## 3.5. Training and Inference

Training. Ground truth attributes for training samples are available, allowing us to pre-render and cache the intermediate canvas states $\{ G _ { i } \} _ { i = 1 } ^ { 5 }$ based on the layer planning re-

<table><tr><td rowspan="2">Methods</td><td colspan="4">LLaVA-OV Scores</td><td rowspan="2">(v)</td><td rowspan="2">Val</td><td rowspan="2">Ove</td><td rowspan="2">Ali</td><td rowspan="2">Undl</td><td rowspan="2">Unds</td></tr><tr><td>(i)</td><td>(ii)</td><td>(iii)</td><td>(iv)</td></tr><tr><td>FlexDM [11]</td><td>5.34</td><td>5.29</td><td>5.41</td><td>5.09</td><td>4.54</td><td>0.8757</td><td>0.3242</td><td>0.0016</td><td>0.7286</td><td>0.7298</td></tr><tr><td>GPT-4o [1]</td><td>6.53</td><td>6.49</td><td>6.60</td><td>6.27</td><td>5.69</td><td>0.9968</td><td>0.0595</td><td>0.0001</td><td>0.3780</td><td>0.5708</td></tr><tr><td>LaDeCo (Ours)</td><td>8.08</td><td>7.92</td><td>8.00</td><td>7.82</td><td>6.98</td><td>0.9365</td><td>0.0865</td><td>0.0013</td><td>0.6922</td><td>0.6580</td></tr><tr><td>GT</td><td>8.35</td><td>8.21</td><td>8.30</td><td>8.01</td><td>7.26</td><td>0.9265</td><td>0.0768</td><td>0.0015</td><td>0.6848</td><td>0.6732</td></tr></table>

Table 1. Quantitative comparison on the design composition task. LLaVA-OV evaluation includes the following aspects: (i) design and layout, (ii) content relevance, (iii) typography and color, (vi) graphics and images, and (v) innovation and originality. The score closest to the one calculated from real data (denoted as GT) is highlighted in bold, indicating the best performance among different methods.

sults (see Figure 4). During training, we fine-tune the model by minimizing the negative log-likelihood of $Y _ { i }$ across all layers:

$$
\mathcal { L } = - \sum _ { i = 1 } ^ { 5 } \log P ( Y _ { i } | Y _ { < i } , X _ { \le i } , G _ { < i } ) .
$$

Inference. At inference time, LaDeCo iteratively generates design layers from $G _ { 1 }$ to $G _ { 5 } ,$ , thereby achieving the goal of design composition. Compared to generating the whole attributes at once, LaDeCo adds only about a 20% increase in rendering time to obtain the intermediate designs, making it an effective and efficient method.

Notably, LaDeCo offers remarkable flexibility at inference. It can handle other design subtasks without any taskspecific training. For example, when the ground truth background layer $( G _ { 1 } )$ is provided, and the model is tasked with generating $G _ { 2 }$ through $G _ { 5 } ,$ it can effectively achieve content-aware layout generation. Similarly, when $G _ { 1 }$ to $G _ { 3 }$ are given as ground truth, and the model is asked to generate $G _ { 4 } ,$ , it performs typography generation.

## 4. Experiments

## 4.1. Setup

Datasets. We conduct experiments on the publicly available Crello-v4 [33] dataset, which includes 23,421 graphic designs sourced from VistaCreate <sup>1</sup>. Since Crello has provided separate rendered pixel images for all elements, we can conveniently build the training inputs as described in Section 3.4. Besides, based on the pixel images as well as the layer planning results obtained in Section 3.3, we can easily render the intermediate designs for training samples through the renderer <sup>2</sup> developed in OpenCOLE [12]. We adopt the same data splits of Crello-v4, dividing the dataset into 19,095 training, 1,951 validation, and 2,375 test samples. To enhance training efficiency, we filter out training samples with more than 25 elements (a total of 938 examples). In addition, we gather a large-scale commercial dataset similar to Crello to study the effect of dataset size on performance. We refer to it as LargeCrello. LargeCrello dataset contains a total of 109,235 samples. We also filter out its samples exceeding 25 elements and manually verify that LargeCrello has no overlap with the test set of Crello. Implementation Details. We choose Llama-3.1-8B <sup>3</sup>, one of the most advanced open-source large language models (LLMs), as the model backbone. The vision encoder is initialized from the CLIP ViT-L/14 model <sup>4</sup>, and the projector is structured as a two-layer MLP using GELU [6] activation functions. To enable efficient training, we utilize the LoRA technique [9] on the backbone, jointly optimizing LoRA parameters and the projector while keeping the vision encoder parameters fixed. We conduct training on four A100-80G GPUs with a global batch size of 128, and employ AdamW [24] to optimize for about 7K iterations with a learning rate of 2e-4. As for the hyper-parameters, the rank number of LoRA is set to 32, the rank alpha is 64, and the token number of an input image is set to 5 (1 cls token, 2 × 2 compressed tokens). At inference, we set the sampling temperature to 0.7 and Top-p (nucleus sampling) to 0.95, balancing diversity and quality for generated designs. Baselines. To demonstrate the effectiveness of LaDeCo, we compare it with existing methods, i.e., FlexDM [11] and GPT-4o [1]. FlexDM originally conducts experiments on Crello-v2, which has different dataset splits of Crello-v4. We re-train it on the upgraded v4 dataset. To accommodate design composition, we mask the position and text attribute fields, and predict them simultaneously. In the GPT-4o baseline, we sequentially concatenate element contents in a manner similar to our approach, and prompt GPT-4o to generate their attributes one by one. For the categorica attributes (e.g., font), we provide all options in the context. Evaluation Metrics. We follow previous work to prepare the evaluation metrics. (1) Overall metrics. Following COLEs [12, 13], we introduce a robust proxy model for comprehensive evaluation. Specifically, we use the LLaVA-OV-7B model [19] to evaluate quality across five aspects: design and layout, content relevance, typography and color, graphics and images, innovation and originality. We use the same prompts as presented in COLE [13]. (2) Geometryrelated metrics. These metrics focus purely on the geometric attributes of elements without considering their content, including element validity (Val), overlap (Ove), alignment (Ali), and underlay effectiveness (Und<sub>l</sub>, Und<sub>s</sub>) [8, 22, 28]. For each metric, the closer it is to the one calculated from real data (denoted as GT in the table), the better. Note that higher or lower values alone are not indicative of better performance. For example, a model could always put all the elements along the left side of the canvas to achieve a low alignment (Ali) score, but such an arrangement would not produce meaningful and high-quality designs.

![](images/f5ab7250aabdd1df2fab9df061ee70e0b6e6de42a6d1505e4a258385b87b097e.jpg)

Figure 3. Qualitative comparison. We also show the ground truth designs for these samples. Please zoom in for a better view.  
![](images/400b02281f5ccee61ed4e5622092d3f16906845a2b8b302c9b71ccd65abcf5e5.jpg)  
Figure 4. The rendered results of different layers from LaDeCo.

## 4.2. Quantitative Evaluation

Table 1 shows quantitative results. LaDeCo significantly outperforms the baseline models in overall metrics (i.e., LLaVA-OV scores), indicating the superiority of LaDeCo in design composition. Particularly, LaDeCo achieves very good scores in design and layout as well as typography and color. This suggests that LaDeCo excels at layout generation and nuanced attribute prediction, both of which are critical for design creation. In terms of geometry-related metrics, LaDeCo achieves closet scores to the one calculated on real data (denoted as GT) on most metrics, showcasing its strong capabilities to model real-world data. In contrast, baseline models often struggle with some metrics. For example, FlexDM exhibits a serious overlap issue, while GPT-4o has low underlay effectiveness.

## 4.3. Qualitative Evaluation

We show rendered generated designs for each method and the corresponding ground truth designs in Figure 3. The results demonstrate that LaDeCo is proficient in composing input elements into high-quality, visually pleasing and balanced designs. On the contrast, the baselines all suffer from some serious problems, such as composition failure (FlexDM, column 2, 6), poor readability (FlexDM, column

<table><tr><td rowspan="2">Settings</td><td colspan="4">LLaVA-OV Scores</td><td rowspan="2">(v)</td><td rowspan="2">Val</td><td rowspan="2">Ove</td><td rowspan="2">Ali</td><td rowspan="2">Undl</td><td rowspan="2">Unds</td></tr><tr><td>(i)</td><td>(ii)</td><td>(iii)</td><td>(iv)</td></tr><tr><td>Llama-3.1-8B (rank 16)</td><td>8.03</td><td>7.89</td><td>8.00</td><td>7.75</td><td>6.90</td><td>0.9347</td><td>0.0796</td><td>0.0012</td><td>0.6900</td><td>0.6564</td></tr><tr><td>Llama-3.1-8B (rank 64)</td><td>8.10</td><td>7.94</td><td>8.04</td><td>7.83</td><td>6.98</td><td>0.9352</td><td>0.0787</td><td>0.0013</td><td>0.7084</td><td>0.6715</td></tr><tr><td>llava-v1.5-7b (rank 32)</td><td>8.00</td><td>7.86</td><td>8.02</td><td>7.78</td><td>6.90</td><td>0.9403</td><td>0.0940</td><td>0.0015</td><td>0.6703</td><td>0.6208</td></tr><tr><td>Llama-3.1-8B-Instruct (rank 32)</td><td>8.08</td><td>7.89</td><td>8.03</td><td>7.82</td><td>6.99</td><td>0.9388</td><td>0.0804</td><td>0.0015</td><td>0.6867</td><td>0.6640</td></tr><tr><td>w/o LP, w/o LDC (rank 32)</td><td>7.23</td><td>7.12</td><td>7.28</td><td>6.99</td><td>6.29</td><td>0.9325</td><td>0.0954</td><td>0.0013</td><td>0.6194</td><td>0.5875</td></tr><tr><td>w/ LP, w/o LDC (rank 32)</td><td>7.84</td><td>7.67</td><td>7.78</td><td>7.56</td><td>6.66</td><td>0.9389</td><td>0.0843</td><td>0.0013</td><td>0.6568</td><td>0.6242</td></tr><tr><td>Llama-3.1-8B* (rank 32)</td><td>8.22</td><td>8.06</td><td>8.22</td><td>7.94</td><td>7.09</td><td>0.9335</td><td>0.1029</td><td>0.0005</td><td>0.7321</td><td>0.7116</td></tr><tr><td>Llama-3.1-8B (rank 32)</td><td>8.08</td><td>7.92</td><td>8.00</td><td>7.82</td><td>6.98</td><td>0.9365</td><td>0.0865</td><td>0.0013</td><td>0.6922</td><td>0.6580</td></tr><tr><td>GT</td><td>8.35</td><td>8.21</td><td>8.30</td><td>8.01</td><td>7.26</td><td>0.9265</td><td>0.0768</td><td>0.0015</td><td>0.6848</td><td>0.6732</td></tr></table>

Table 2. Ablation studies. Our investigation covers four aspects (from top to bottom): (1) the rank number in LoRA, (2) the base model, (3) the key techniques in LaDeCo, where LP denotes layer planning , and LDC represents layered design composition, (4) dataset size. The model with \* to is trained on the combined Crello and LargeCrello datasets, while the models without \* are trained on Crello only.

![](images/05035a56fb5438045731bc6d55b4def70854eb136d563d7a8e7bf01ac0a0a05b.jpg)

Figure 5. LaDeCo combines the same input elements into designs with different canvas sizes.  
![](images/511ae91ea7d1552876f34504e743a9b53a720f2e2dc5b63bed992c533291d4af.jpg)  
Figure 6. LaDeCo adds embellishment elements to an existing design to achieve a more appealing design

1) and imbalance (GPT-4o, column 1, 3, 4). Additionally, LaDeCo can accurately capture relationships between elements. For instance, text elements are precisely positioned on underlay elements (Ours, column 1, 4, 5, 7), and embellishment elements contribute meaningfully to decorate the designs (Ours, column 2, 6).

We further shows the rendered results of different layers generated by LaDeCo in Figure 4. These results demonstrate that with the proposed layer planning module, LaDeCo can generate meaningful layers.

Besides, LaDeCo also enables some interesting applications. Figure 5 shows that LaDeCo can achieve design composition at different canvas sizes (called resolution adjustment). The predicted attributes will be adjusted to suit the given canvas size. Figure 6 presents that LaDeCo can add embellishment elements to an existing design to make it more pleasing (called design decoration). Figure 7 shows that, given the same input elements, LaDeCo can compose them to create diverse designs, providing the users with multiple choices (called design variation).

![](images/78ca617eb220fc1153b71d018ebafb294554c34beff0723b3e989f20d38ec5a8.jpg)  
Figure 7. LaDeCo creates diverse designs with the same elements.

## 4.4. Ablation Studies

Table 2 shows the results of ablation studies. (1) Rank number in LoRA: When the rank number varies from 16 to 32, and then to 64, the quantitative metrics show little variation, indicating the robustness of LaDeCo with respect to the amount of training parameters. (2) Base model: We adopt two other pre-trained LLMs, llava-v1.5-7b and Llama-3.1- 8B-Instruct, as the backbone. Similar to previous ablation study, our LaDeCo is robust to the choice of base model. (3) Key techniques in LaDeCo: We consider two settings. First, we remove the layered design composition (LDC) and use the model to predict all element attributes sequentially without taking the intermediate rendered layers as input (denoted as w/ LP, w/o LDC (rank 32) in the table). This leads to drops in five overall metrics and two underlay effectiveness metrics, indicating the importance of LDC in design composition. Second, we continue to remove layer planning (LP) module and construct the input-output representation using a random ordered design elements (denoted as w/o LP, w/o LDC (rank 32) in the table). The aforementioned metrics show a significant decline, which indicates that arranging elements following the hierarchical structure does play a critical role in design composition. (4) Dataset size: When the model is trained on the combined datasets, most quantitative metrics are considerably improved and get closer to ground truth values. The results suggest that LaDeCo is a scalable algorithm with respect to dataset size.

<table><tr><td>Methods</td><td>Val</td><td>Ove</td><td>Ali</td><td>Undl</td><td>Unds</td><td>Uti</td><td>Occ</td><td>Rea</td></tr><tr><td>PosterLLaVa [34]</td><td>0.9269</td><td>0.0685</td><td>0.0011</td><td>0.7879</td><td>0.7375</td><td>0.4199</td><td>0.1936</td><td>0.0747</td></tr><tr><td>PosterLlama [28]</td><td>0.8701</td><td>0.0868</td><td>0.0014</td><td>0.8483</td><td>0.7798</td><td>0.4115</td><td>0.1772</td><td>0.0694</td></tr><tr><td>LaDeCo (Ours)</td><td>0.9340</td><td>0.0805</td><td>0.0016</td><td>0.6851</td><td>0.6540</td><td>0.4414</td><td>0.1835</td><td>0.0768</td></tr><tr><td>GT</td><td>0.9265</td><td>0.0768</td><td>0.0015</td><td>0.6848</td><td>0.6732</td><td>0.4737</td><td>0.1628</td><td>0.0709</td></tr></table>

Table 3. Quantitative results on the content-aware layout generation subtask. The score closest to the one calculated from real data (denoted as GT) is highlighted in bold, indicating the best performance among different methods.

PosterLlama  
PosterLLaVA  
Ours  
![](images/4a7e5121ffcbc9d9cffc5f035df377b540054726a164b536d2eef22bc2ea4d97.jpg)  
GT  
Figure 8. Qualitative comparison on the content-aware layout generation task. The yellow, red, green, pink boxes represent underlay, image, text, and embellishment elements, respectively.

## 4.5. Comparison with Task-specific Baselines

We further compare LaDeCo with task-specific baselines, which focus on handling sub-tasks of design composition.

Content-aware layout generation. We consider two state-of-the-art baselines specialized for this subtask, including PosterLLava [34] and PosterLlama [28]. We retrain them on the Crello dataset for a fair comparison. Besides geometry-related metrics, we further adopt metrics in DS-GAN [8] to assess content-related quality based on the given canvas, i.e., canvas utility (Uti), occlusion (Occ), and text readability (Rea). Table 3 shows quantitative results. Compared to specialized methods, LaDeCo achieves best performance on most metrics. Figure 8 shows qualitative results. LaDeCo excels in generating layouts that prevent blocking the main content within the given canvas.

FlexDM  
OpenCOLE  
Ours  
GT  
![](images/ca8584e212dce5dfb172ee9e81680af915e7195b3ce00a2bb781bd16b11b88e7.jpg)  
Figure 9. Qualitative comparison on typography generation.

Typography generation. We consider FlexDM [11] and OpenCOLE [12] as baselines. For FlexDM, we only mask the text attribute fields, leaving the position fields accessible. For OpenCOLE, which has already been trained on Crello-v4, we leverage the released Typography LMM model <sup>5</sup> for comparison. Figure 9 shows the resutls. LaDeCo takes into consideration various aspects such as text layout, aesthetics, and readability during typography generation process. In contrast, the baseline models suffer from text overlap and poor readability.

## 5. Conclusion

In this work, we introduce LaDeCo for design composition. The main idea is integrating the inherent hierarchical structure, which emerges from the layered principle during the practical design process, into LMMs. Specifically, a layer planning module categorizes input elements into different layers based on their contents, and a layered design composition process predicts the attributes that controls the composition by using the rendered previous layers as context. In the future, we plan to expand our model to accommodate more user requirements, e.g., textual descriptions. We also plan to explore the integration of our design composition model with image generation models for element content creation, aiming to achieve end-to-end design generation.

## References

[1] https://openai.com/index/hello-gpt-4o/. [Accessed 04-11-2024]. 2, 3, 4, 5, 11

[2] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023. 2, 3

[3] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24185–24198, 2024. 2

[4] Yutao Cheng, Zhao Zhang, Maoke Yang, Hui Nie, Chunyuan Li, Xinglong Wu, and Jie Shao. Graphic design with large multimodal model. arXiv preprint arXiv:2404.14368, 2024. 2

[5] Yifan Gao, Jinpeng Lin, Min Zhou, Chuanbin Liu, Hongtao Xie, Tiezheng Ge, and Yuning Jiang. Textpainter: Multimodal text image generation with visual-harmony and textcomprehension for poster design. In Proceedings of the 31st ACM International Conference on Multimedia, pages 7236– 7246, 2023. 2, 3

[6] Dan Hendrycks and Kevin Gimpel. Gaussian error linear units (gelus). arXiv preprint arXiv:1606.08415, 2016. 5

[7] Daichi Horita, Naoto Inoue, Kotaro Kikuchi, Kota Yamaguchi, and Kiyoharu Aizawa. Retrieval-augmented layout transformer for content-aware layout generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 67–76, 2024. 2

[8] Hsiao Yuan Hsu, Xiangteng He, Yuxin Peng, Hao Kong, and Qing Zhang. Posterlayout: A new benchmark and approach for content-aware visual-textual presentation layout. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6018–6026, 2023. 2, 6, 8

[9] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021. 5

[10] Naoto Inoue, Kotaro Kikuchi, Edgar Simo-Serra, Mayu Otani, and Kota Yamaguchi. LayoutDM: Discrete Diffusion Model for Controllable Layout Generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10167–10176, 2023. 3

[11] Naoto Inoue, Kotaro Kikuchi, Edgar Simo-Serra, Mayu Otani, and Kota Yamaguchi. Towards flexible multi-modal document models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14287–14296, 2023. 2, 3, 5, 8

[12] Naoto Inoue, Kento Masui, Wataru Shimoda, and Kota Yamaguchi. Opencole: Towards reproducible automatic graphic design generation. arXiv preprint arXiv:2406.08232, 2024. 3, 4, 5, 8

[13] Peidong Jia, Chenxuan Li, Zeyu Liu, Yichao Shen, Xingru Chen, Yuhui Yuan, Yinglin Zheng, Dong Chen, Ji Li, Xiaodong Xie, et al. Cole: A hierarchical generation frame-

work for graphic design. arXiv preprint arXiv:2311.16974, 2023. 2, 3, 5, 6

[14] Zhaoyun Jiang, Jiaqi Guo, Shizhao Sun, Huayu Deng, Zhongkai Wu, Vuksan Mijovic, Zijiang James Yang, Jian Guang Lou, and Dongmei Zhang. Layoutformer++: Conditional graphic layout generation via constraint serializa tion and decoding space restriction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18403–18412, 2023. 3

[15] Kotaro Kikuchi, Edgar Simo-Serra, Mayu Otani, and Kota Yamaguchi. Constrained graphic layout generation via latent optimization. In Proceedings of the 29th ACM International Conference on Multimedia, pages 88–96, 2021. 3

[16] Kotaro Kikuchi, Naoto Inoue, Mayu Otani, Edgar Simo-Serra, and Kota Yamaguchi. Multimodal markup document models for graphic design completion. arXiv preprint arXiv:2409.19051, 2024. 3

[17] Xiang Kong, Lu Jiang, Huiwen Chang, Han Zhang, Yuan Hao, Haifeng Gong, and Irfan Essa. Blt: Bidirectional layout transformer for controllable layout generation. In European Conference on Computer Vision, pages 474–490. Springer, 2022. 3

[18] Hsin-Ying Lee, Lu Jiang, Irfan Essa, Phuong B Le, Haifeng Gong, Ming-Hsuan Yang, and Weilong Yang. Neural design network: Graphic layout generation with constraints. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part III 16, pages 491–506. Springer, 2020. 3

[19] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024. 5

[20] Jianan Li, Jimei Yang, Jianming Zhang, Chang Liu, Christina Wang, and Tingfa Xu. Attribute-conditioned layout gan for automatic graphic design. IEEE Transactions on Visualization and Computer Graphics, 27(10):4039–4048, 2020. 3

[21] Jiawei Lin, Jiaqi Guo, Shizhao Sun, Weijiang Xu, Ting Liu, Jian-Guang Lou, and Dongmei Zhang. A parse-then-place approach for generating graphic layouts from textual descriptions. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 23622–23631, 2023. 3

[22] Jiawei Lin, Jiaqi Guo, Shizhao Sun, Zijiang Yang, Jian Guang Lou, and Dongmei Zhang. Layoutprompter: Awaken the design ability of large language models. Advances in Neural Information Processing Systems, 36, 2024. 2, 3, 4, 6

[23] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36, 2024. 2, 3

[24] I Loshchilov. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. 5

[25] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan. Video-chatgpt: Towards detailed video understanding via large vision and language models. arXiv preprint arXiv:2306.05424, 2023. 3

[26] Chancharik Mitra, Brandon Huang, Trevor Darrell, and Roei Herzig. Compositional chain-of-thought prompting for large

multimodal models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14420–14431, 2024. 3

[27] KhayTze Peong, Seiichi Uchida, and Daichi Haraguchi. Typographic text generation with off-the-shelf diffusion model. In International Conference on Document Analysis and Recognition, pages 52–69. Springer, 2024. 2, 3

[28] Jaejung Seol, Seojun Kim, and Jaejun Yoo. Posterllama: Bridging design ability of langauge model to contents-aware layout generation. arXiv preprint arXiv:2404.00995, 2024. 2, 4, 6, 8

[29] Mohammad Amin Shabani, Zhaowen Wang, Difan Liu, Nanxuan Zhao, Jimei Yang, and Yasutaka Furukawa. Visual layout composer: Image-vector dual diffusion model for design layout generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9222–9231, 2024. 3

[30] Wataru Shimoda, Daichi Haraguchi, Seiichi Uchida, and Kota Yamaguchi. Towards diverse and consistent typography generation. In Proceedings of the IEEE/CVF Winter Conference on Applications ofComputer Vision, pages 7296–7305, 2024. 2

[31] Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023. 2

[32] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022. 3

[33] Kota Yamaguchi. Canvasvae: Learning to generate vector graphic documents. ICCV, 2021. 2, 5

[34] Tao Yang, Yingmin Luo, Zhongang Qi, Yang Wu, Ying Shan, and Chang Wen Chen. Posterllava: Constructing a unified multi-modal layout generator with llm. arXiv preprint arXiv:2406.02884, 2024. 2, 8

[35] Jiabo Ye, Anwen Hu, Haiyang Xu, Qinghao Ye, Ming Yan, Yuhao Dan, Chenlin Zhao, Guohai Xu, Chenliang Li, Junfeng Tian, et al. mplug-docowl: Modularized multimodal large language model for document understanding. arXiv preprint arXiv:2307.02499, 2023. 3

[36] Junyi Zhang, Jiaqi Guo, Shizhao Sun, Jian-Guang Lou, and Dongmei Zhang. Layoutdiffusion: Improving graphic layout generation by discrete diffusion probabilistic models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7226–7236, 2023. 3

understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023. 2, 3

[37] Zhuosheng Zhang, Aston Zhang, Mu Li, Hai Zhao, George Karypis, and Alex Smola. Multimodal chain-ofthought reasoning in language models. arXiv preprint arXiv:2302.00923, 2023. 3

[38] Min Zhou, Chenchen Xu, Ye Ma, Tiezheng Ge, Yuning Jiang, and Weiwei Xu. Composition-aware graphic layout gan for visual-textual presentation designs. arXiv preprint arXiv:2205.00303, 2022. 2

[39] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language