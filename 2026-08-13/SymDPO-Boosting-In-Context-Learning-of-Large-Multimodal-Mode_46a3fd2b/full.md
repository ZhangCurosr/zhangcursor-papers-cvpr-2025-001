# SymDPO: Boosting In-Context Learning of Large Multimodal Models with Symbol Demonstration Direct Preference Optimization

Hongrui Jia<sup>1,†</sup> Chaoya Jiang<sup>1,†</sup> Haiyang Xu<sup>2,∗</sup> Wei Ye<sup>1,∗</sup> Mengfan Dong<sup>1</sup> Ming Yan<sup>2</sup> Ji Zhang<sup>2</sup> Fei Huang<sup>2</sup> Shikun Zhang<sup>1</sup>

<sup>1</sup> National Engineering Research Center for Software Engineering, Peking University <sup>2</sup> Alibaba Group

{jiahongrui, jiangchaoya, wye, zhangsk}@pku.edu.cn, {shuofeng.xhy, fei.huang}@alibaba-inc.com

## Abstract

As language models continue to scale, Large Language Models (LLMs) have exhibited emerging capabilities in In-Context Learning (ICL), enabling them to solve language tasks by prefixing a few in-context demonstrations (ICDs) as context. Inspired by these advancements, researchers have extended these techniques to develop Large Multimodal Models (LMMs) with ICL capabilities. However, existing LMMs face a critical issue: they often fail to effectively leverage the visual context in multimodal demonstrations and instead simply follow textual patterns. This indicates that LMMs do not achieve effective alignment between multimodal demonstrations and model outputs. To address this problem, we propose Symbol Demonstration Direct Preference Optimization (SymDPO). Specifically, SymDPO aims to break the traditional paradigm ofconstructing multimodal demonstrations by using random symbols to replace text answers within instances. This forces the model to carefully understand the demonstration images and establish a relationship between the images and the symbols to answer questions correctly. We validate the effectiveness of this method on multiple benchmarks, demonstrating that with SymDPO, LMMs can more effectively understand the multimodal context within examples and utilize this knowledge to answer questions better. Code is available at https://github.com/APiaoG/SymDPO.

## 1. Introduction

The rapid advancement of Large Language Models (LLMs) [3, 7, 10, 41, 47] has brought remarkable improvements in their In-Context Learning (ICL) capabilities [15]. By leveraging In-Context Demonstrations (ICDs), a small set of exemplars provided as context, these models achieve impressive performance on various language tasks. This breakthrough in Natural Language Processing (NLP) has catalyzed research efforts to extend similar contextual learning capabilities to Large Multimodal Models (LMMs) [1, 4, 22, 23, 26, 29, 32, 33, 54–56]. The ultimate goal is to enable LMMs to effectively learn from a limited number of image-text pairs for specific tasks without parameter updates, thereby achieving few-shot learning in the multimodal domain.

![](images/865499fa53bc0674e68993e39f77e4a99e5e95070c7ce2bd8fed0a6117a3eecf.jpg)

![](images/2ff31dc410d84e43ae719192695c2075ba9b52286ad7e1307635436b0cd50347.jpg)  
Figure 1. In subfigure (a), an example of visual context overlook is illustrated using OpenFlamingo as a case study. Here, Open-Flamingo [6] erroneously generates a response by solely following the textual cues in the demonstration, leading to an inaccurate answer. After applying SymDPO to enhance alignment, Open-Flamingo with SymDPO successfully corrects its response, accurately addressing the question. Subfigure (b) further demonstrates that for OpenFlamingo (OF), replacing images in the demonstration with blank placeholders (OF w/ blank) or omitting images altogether (OF w/o image) surprisingly yields even better performance than the original setup. This result suggests a substantial model dependency on textual context over visual information.

To enhance the ICL capabilities of LMMs, prior works [4, 6, 26, 27, 29] have explored two primary approaches: The first approach [4, 6, 33, 46] involves pretraining LMMs on massive-scale interleaved image-text data collected from the internet. The second approach [17, 24, 29] focuses on constructing specialized instruction-tuning datasets with numerous ICD examples. However, despite these efforts, recent studies [9, 11] have shown that both approaches still face a significant limitation, which we term Visual Context Overlook. This phenomenon manifests as the LMMs persistently struggle to effectively incorporate visual context from multimodal demonstrations, exhibiting a strong bias towards textual pattern matching. As illustrated in Figure 1 (a), when presented with multimodal examples, LMMs tend to generate responses by following textual patterns in the context while failing to properly utilize the critical visual information, leading to inaccurate responses. Additionally, as shown in Figure 1 (b), substituting images in ICDs with blank images or even removing them altogether does not affect model performance, further underscoring the limited role of visual information in the current alignment process of LMMs.

This issue highlights a core limitation in LMMs’ ability to follow instructions from multimodal demonstrations within ICL scenarios accurately. Recently, Direct Preference Optimization (DPO) [43], a human preference-aligned reinforcement learning technique applied during the posttraining phase, has been widely adopted to enhance LMMs instruction-following capabilities [33, 36, 42, 49], offering a promising direction for addressing visual context overlook. However, current DPO methods exhibit two key limitations within ICL scenarios: 1. Insufficient Mechanisms for Multimodal In-Context Instruction Following: Current DPO methods [42, 49] are largely optimized for general instruction-following tasks and lack the specialized mechanisms necessary to enhance LMMs’ comprehension and adherence to the combined visual and textual information characteristic of multimodal demonstrations in ICL settings. 2. Challenges in Preference Data Collection for Visual Context Dependency: In typical Visual Question Answering (VQA) tasks, many questions can be effectively answered based on text alone, without needing information from multimodal in-context demonstrations (ICDs). This reliance on textual cues creates a significant barrier to collecting reliable preference data for multimodal learning. Specifically, it complicates the distinction between “accepted” and “rejected” answers, as models may default to simple text pattern matching.

To overcome these limitations, we introduce SymDPO (Symbol Demonstration Direct Preference Optimization), a novel method specifically designed to compel LMMs to depend on both visual and textual inputs in ICDs by establishing a mapping between visual information and symbolic responses. SymDPO replaces traditional textual answers in ICDs with semantically neutral or mismatched symbolic text strings—specific characters or strings that have no semantic relevance to the visual context. This symbolic substitution compels the LMM to construct a mapping between visual elements and these symbolic strings, effectively linking the image content to a symbolic representation of the answer. As a result, the model can only generate correct responses by thoroughly interpreting the visual content within ICDs, as there is no relevant meaning in the symbolic text alone to support a response. This configuration makes visual information essential for understanding and responding accurately, ensuring that correct answers derive from a combined understanding of both image content and symbolic text. SymDPO thus redefines the model’s reliance on visual context, reinforcing the multimodal comprehension required for accurate response generation in visuallydependent scenarios. Our contributions are as follows:

• We propose a novel symbolic preference optimization method, SymDPO, that compels LMMs to leverage multimodal information effectively, advancing their ability to integrate visual and textual cues within ICDs.

• We design and implement a pipeline that generates symbolic preference data, replacing textual answers with contextually mismatched symbols to enforce symbolic alignment with visual context.

• Through comprehensive experiments across multiple LMM architectures, we demonstrate consistent improvements in performance on various benchmarks, verifying SymDPO’s effectiveness in addressing visual context overlook and enhancing multimodal understanding.

## 2. Related Work

## 2.1. In-Context Learning

In LLMs, prompt engineering handles specific tasks without constant fine-tuning. A variant of this method, ICL, enhances these abilities by generating prompts that incorporate multiple demonstrations [10, 15, 30, 31, 40]. ICL has already shown superior performance and broad applicability across many tasks [16, 18, 19, 38, 39] and can be easily modified for downstream tasks [2, 50, 52]. As LLMs continue to improve, more researchers are adapting them to the multimodal domain [34, 59, 61]. Leveraging the robust inference capabilities of LLMs, some LMMs[1, 4, 21, 26, 29, 32, 33, 54–56] have begun to display ICL capabilities, like Flamingo [4] and IDEFICS [26]. These models have notably improved their ICL abilities by incorporating multiple samples as contextual information during the training process. However, Multimodal ICL has some limitations: LMMs pay much more attention to the textual pattern instead of image information. Previous studies [53, 59] have primarily focused on the method of constructing context to mitigate the issue, overlooking the inherent characteristics of LMMs themselves, resulting in ineffective outcomes.

![](images/29f338f9011d2270f038f883f2b448753ddc032641da1eec0b45f877fc313111.jpg)  
Figure 2. Comparison of General DPO and SymDPO Formats: General DPO relies solely on standard text for Questions, Answers, Chosen, and Rejected Answers, focusing on text-based training. In contrast, SymDPO replaces textual Answers with symbolized text to boost multimodal understanding, requiring models to interpret both visual and symbolized cues. This approach strengthens the model’s ability to reason and decide in complex multimodal contexts.

## 2.2. Reinforcement Learning from Human Feedback (RLHF)

Reinforcement Learning from Human Feedback (RLHF) [41] has become a pivotal technique in guiding model responses to align more closely with human expectations. This approach fine-tunes models based on preference data obtained from human evaluators, facilitating the development of more intuitive and contextually accurate responses [25, 28]. Although RLHF has proven effective in aligning language models (LMs) with human values, its complexity and resource demands have prompted the search for alternatives. RAFT [14] chooses the best training samples through an existing reward model, while RRHF[58] uses a simpler ranking loss, maintaining the efficiency of PPO [44]. In contrast, DPO [43] directly optimizes LMs using a preference-based loss function, demonstrating improved training stability compared to traditional RLHF.

## 3. Method

To address the limitations of LMMs in leveraging image information within ICL, we propose SymDPO. As shown in Figure 2, SymDPO replaces the original answers in the context with unrelated words. This alteration prevents the model from depending solely on text patterns to infer answers. Instead, the model is driven to derive the correct answer by recognizing patterns that emerge jointly from the images and text in the context. In Section 3.1, we will discuss the construction of a dataset tailored for SymDPO to facilitate this balanced learning approach. Section 3.2 will elaborate on the specific steps and mechanisms of the SymDPO algorithm.

## 3.1. SymDPO Data Construction

The construction of the SymDPO data follows a three-step process. First, we gather and structure a QA dataset in an In-Context Learning (ICL) format. Based on this ICL dataset, we proceed to build a standard DPO dataset. Finally, we introduce the concept of SymDPO, expanding upon the general DPO data foundation to create the SymDPO dataset with enhanced multimodal challenges for LMMs.

Construct In-Context dataset: First, we collect imagequestion-answer triplets from VQA datasets such as GQA, VQAv2, and image classification datasets like ImageNet. These data points are then reorganized into an In-Context Learning (ICL) format. Questions that target similar task types, such as binary yes/no questions, or those related to image object categories, attributes, relationships, or quantities, are grouped together. For example, all questions in one group may revolve around the concept of ”narrow” versus ”wide.” Next, for each category of questions, we construct data in the ICL format: $D _ { 1 } , D _ { 2 } , \ldots , D _ { N } , F .$ , where $D _ { i } = \{ I _ { i } , Q _ { i } , A _ { i } \} , i \in \{ 1 , 2 , \ldots , N \}$ and $F = \{ \hat { I } , \hat { Q } , \hat { A } \}$ Here, $D _ { i }$ represents the i-th demonstration, and $F$ denotes the final question-answer pair. We ensure that the demonstrations contain at least two different answers, with at least one answer matching the answer A<sup>ˆ</sup> in F. Building on this structure, we proceed to construct the DPO data required for training.

Construct Original DPO Dataset: For each constructed ICL dataset, we treat the original answer A<sup>ˆ</sup> from the final QA round as the positive label. Then, we select a distinctly different answer $A _ { j }$ (where $A _ { j } \neq { \hat { A } } )$ from the same category of questions as the negative label, ensuring that this answer is not one of the previous answers. For instance, in the case of color-related questions, we may randomly choose another distinct answer like ”The logo is $\mathrm { b l a c k } ^ { \prime }$ as the negative label. With this, a single DPO data point is constructed. Construct SymDPO Dataset: The SymDPO dataset is designed to increase the difficulty of answering questions, requiring Large Multimodal Models (LMMs) to fully comprehend the combined visual and textual information within the In-Context Demonstrations (ICDs) to accurately respond. Illustrated in the Figure 2, we constructed five distinct data configurations based on the standard DPO dataset to further enhance the model’s comprehension capabilities.

In the SymDPO dataset, we employ a more challenging approach: firstly, all answers in the demonstrations are replaced with semantically meaningless symbols. This transformation can be expressed as:

$$
\dot { D } _ { i } = \{ I _ { i } , Q _ { i } , S _ { i } \} , \quad i \in \{ 1 , 2 , \ldots , N \}\tag{1}
$$

where $S _ { i }$ represents a symbol unrelated to the actual answer, effectively stripping away semantic information to prevent the model from deducing the answer solely through simple textual patterns. This design compels the model to rely on a combination of visual and textual information within the ICDs to respond accurately within a multimodal environment.

Furthermore, a unique demonstration $D _ { k }$ is designated within the ICDs, in which the symbolic answer $S _ { k }$ aligns with the answer to the final question-answer pair $F =$ $\{ \hat { I } , \hat { Q } , \hat { A } \}$ . For this setup, the chosen answer for $F$ is set as $S _ { k }$ , while the rejected answer can be another unchosen answer, such as a different answer $A _ { j }$ from the same question type or another symbolic answer $S _ { j }$ , provided it satisfies the following conditions:

$$
A _ { j } \neq { \hat { A } } \quad { \mathrm { a n d } } \quad S _ { j } \neq S _ { k }\tag{2}
$$

For example, in the second data configuration of SymDPO in Figure 2, we replace ”narrow” and ”wide” with the symbols ”rhondda” and ”odwyer”. In this scenario, even if the model can reason independently of the ICD, it must still interpret the overall semantics of these symbols within the ICD to answer correctly. This approach ensures that the model needs to understand the implicit meaning of the symbols deeply, rather than relying solely on isolated textual or visual information.

Finally, we add further complexity by introducing additional configurations, such as whether to erase the question in the ICD or whether the Rejected Answer should be semantically relevant. These adjustments lead to five different types of SymDPO data, maintaining a certain proportion of representation across all types in the final dataset to maximize diversity.

## 3.2. SymDPO

In the previous section, we introduced the data construction process of SymDPO. In this section, we will explain the principles behind SymDPO. Training LMMs with SymDPO can make LMMs pay more attention to visual information. Aligning preferences in Large Multimodal Models (LMMs) involves aligning the model’s preferences with human preferences, typically employing techniques such as RLHF[41] (Reinforcement Learning from Human Feedback) and RLAIF[8] (Reinforcement Learning from AI Feedback). Considering a dataset ${ \mathcal { D } } _ { S } .$ , each entry includes an input $x = \{ q , I , C \}$ , a chosen response $y _ { w }$ and a rejected response $y _ { l } .$ , while q represents the question, I represent images and C represents the context. $\mathcal { D } _ { S }$ can be represented as $\mathcal { D } _ { S } = \{ x , y _ { w } , y _ { l } \}$

Upon receiving the input x, a LMM produces an output y, to which a reward $r ( x , y )$ is allocated. The reward model assesses both chosen (high $r ( x , y ) )$ and rejected (low $r ( x , y ) )$ samples. Meanwhile, to avoid overfitting on the dataset $\mathcal { D } _ { S }$ , preference alignment in LMMs incorporates a Kullback-Leibler (KL) divergence loss $D _ { K L }$ , which normalizes the disparity between the model’s policy $\pi _ { \boldsymbol { \theta } } ( y | \boldsymbol { x } )$ and the reference model’s policy $\pi _ { r e f } ( y | x )$ . The goal is to maximize this:

$$
\begin{array} { r } { \operatorname* { m a x } _ { \theta } \left[ \mathbb { E } _ { x \sim \mathcal { D } _ { S } , ( y ) \sim \pi _ { \theta } ( y | x ) } [ r ( x , y ) ] - \beta \cdot D _ { K L } ( \pi _ { \theta } ( y | x ) | | \pi _ { r e f } ( y | x ) ) \right] } \end{array}\tag{3}
$$

Here, θ denote the parameters of the LMM, $\pi _ { \theta }$ denote the policy of the LMM, $\pi _ { \boldsymbol { \theta } } ( y | \boldsymbol { x } )$ denote the distribution of the LMM and the hyperparameter $\beta$ controls the impact of the KL divergence within the optimization target. The reference model is the model’s state prior to preference alignment.

To enhance the preference alignment target, the Direct Preference Optimization (DPO) method is utilized. The DPO method is efficient, stable, and does not require fitting a reward model. Our method, SymDPO, is based on the classical DPO algorithm. The SymDPO objective is formally defined as follows:

$$
\begin{array} { l } { { \displaystyle { \mathcal { L } } _ { S } ( \pi _ { \theta } ; \pi _ { r e f } ) = } } \\ { { \displaystyle - { \mathbb { E } } _ { ( x , y _ { w } , y _ { l } ) \sim { \mathcal { D } } _ { S } } l o g \sigma ( \beta l o g \frac { \pi _ { \theta } \left( y _ { w } | x \right) } { \pi _ { r e f } \left( y _ { w } | x \right) } - \beta l o g \frac { \pi _ { \theta } \left( y _ { l } | x \right) } { \pi _ { r e f } \left( y _ { l } | x \right) } ) } } \end{array}\tag{4}
$$

where $\sigma$ is the logistic function.

## 4. Experiment

## 4.1. Experiment Setting

Implementation Details: We instantiate our model with Open-Flamingo [6] and IDEFICS [42], constructing the process taking approximately 1 hour.

<table><tr><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>Shots</td><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>COCO Caption(CIDEr)</td><td rowspan=1 colspan=1>Flickr-30K(CIDEr)</td><td rowspan=1 colspan=1>VQAv2(Acc)</td><td rowspan=1 colspan=4>OK-VQA(Acc)</td><td rowspan=1 colspan=1>TextVQA(Acc)</td></tr><tr><td rowspan=14 colspan=1>OF-3B (I)</td><td rowspan=5 colspan=1>4</td><td rowspan=5 colspan=1>Base+ SymDPO+ General DPO+ Video DPO+ MIA-DPO</td><td rowspan=5 colspan=1>82.7 $\mathbf { 8 7 . 4 } _ { + 4 . 7 }$  $8 3 . 5 _ { + 0 . 8 }$  $8 2 . 5 _ { - 0 . 2 }$  $8 4 . 7 _ { + 2 . 0 }$ </td><td rowspan=2 colspan=1>59.1 ${ \bf 6 1 . 2 } _ { + 2 . 1 }$ </td><td rowspan=1 colspan=1>45.7</td><td rowspan=1 colspan=4>30.6</td><td rowspan=2 colspan=1>28.1 $2 8 . 3 _ { + 0 . 2 }$ </td></tr><tr><td rowspan=1 colspan=1> ${ \bf 4 6 . 2 } _ { + 0 . 5 }$ </td><td rowspan=1 colspan=2> $3 1 . 6 _ { + 1 . 0 }$ </td><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=1> $6 0 . 0 _ { + 0 . 9 }$ </td><td rowspan=1 colspan=1> $4 6 . 0 _ { + 0 . 3 }$ </td><td rowspan=2 colspan=4> $3 0 . 7 _ { + 0 . 1 }$  $3 0 . 3 _ { - 0 . 3 }$ </td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=1> $2 8 . 2 _ { + 0 . 1 }$ </td></tr><tr><td rowspan=2 colspan=1> $5 9 . 5 _ { + 0 . 4 }$  $6 0 . 8 _ { + 1 . 7 }$ </td><td rowspan=1 colspan=1> $4 5 . 5 _ { - 0 . 2 }$ </td><td rowspan=1 colspan=1> $2 8 . 4 _ { + 0 . 3 }$ </td></tr><tr><td rowspan=1 colspan=1> $4 6 . 1 _ { + 0 . 4 }$ </td><td rowspan=1 colspan=4> $3 0 . 4 _ { - 0 . 2 }$ </td><td rowspan=1 colspan=1> $2 { \bf 8 . 5 } _ { + 0 . 4 }$ </td></tr><tr><td rowspan=4 colspan=1>8</td><td rowspan=4 colspan=1>Base+ SymDPO+ General DPO+ Video DPO+ MIA-DPO</td><td rowspan=4 colspan=1>87.8 ${ \bf 9 1 . 2 } _ { + 3 . 4 }$  $8 8 . 4 _ { + 0 . 6 }$  $8 7 . 3 _ { - 0 . 5 }$  $8 9 . 0 _ { + 1 . 2 }$ </td><td rowspan=1 colspan=1> $\overline { { 6 0 . 7 } }$  ${ \bf 6 5 . 3 _ { + 4 . 6 } }$ </td><td rowspan=1 colspan=1>45.9 $\mathbf { 4 6 . 5 _ { + 0 . 6 } }$ </td><td rowspan=1 colspan=4>31.5 $3 2 . 7 _ { + 1 . 2 }$ </td><td rowspan=1 colspan=1>29.1 $\mathbf { 2 9 . 8 _ { + 0 . 7 } }$ </td></tr><tr><td rowspan=3 colspan=1> $6 1 . 5 _ { + 0 . 8 }$  $6 0 . 6 _ { - 0 . 1 }$  $6 2 . 5 _ { + 1 . 8 }$ </td><td rowspan=1 colspan=1> $4 6 . 1 _ { + 0 . 2 }$ </td><td rowspan=2 colspan=4> $3 1 . 3 _ { - 0 . 2 }$ </td><td rowspan=2 colspan=1> $2 9 . 1 _ { + 0 . 0 }$  $2 8 . 7 _ { - 0 . 4 }$ </td></tr><tr><td rowspan=1 colspan=1> $4 6 . 0 _ { + 0 . 1 }$ </td><td rowspan=1 colspan=3>31.4-0.1</td></tr><tr><td rowspan=1 colspan=1> $4 6 . 3 _ { + 0 . 4 }$ </td><td rowspan=1 colspan=4> $3 1 . 2 _ { - 0 . 3 }$ </td><td rowspan=1 colspan=1> $2 9 . 3 _ { + 0 . 2 }$ </td></tr><tr><td rowspan=5 colspan=1>16</td><td rowspan=5 colspan=1>Base+ SymDPO+ General DPO+ Video DPO+ MIA-DPO</td><td rowspan=2 colspan=1> $\overline { { 9 1 . 9 } }$  $\mathbf { 9 3 . 4 } _ { + 1 . 5 }$ </td><td rowspan=2 colspan=1> $\overline { { 6 3 . 0 } }$  ${ \bf 6 6 . 1 } _ { + 3 . 1 }$ </td><td rowspan=1 colspan=1> $\overline { { 4 5 . 8 } }$ </td><td rowspan=2 colspan=4>30.7 $3 2 . 8 _ { + 1 . 9 }$ </td><td rowspan=2 colspan=1>29.1 $\mathbf { 2 9 . 6 } _ { + 0 . 5 }$ </td></tr><tr><td rowspan=1 colspan=1> $4 6 . 5 _ { + 0 . 7 }$ </td></tr><tr><td rowspan=3 colspan=1> $9 2 . 0 _ { + 0 . 1 }$  $9 1 . 8 _ { - 0 . 1 }$  $9 2 . 5 _ { + 0 . 6 }$ </td><td rowspan=1 colspan=1> $6 2 . 7 _ { - 0 . 3 }$ </td><td rowspan=1 colspan=1> $4 6 . 0 _ { + 0 . 2 }$ </td><td rowspan=1 colspan=4> $3 0 . 5 _ { - 0 . 2 }$ </td><td rowspan=1 colspan=1> $2 9 . 0 _ { - 0 . 1 }$ </td></tr><tr><td rowspan=1 colspan=1> $6 2 . 8 _ { - 0 . 2 }$ </td><td rowspan=1 colspan=1> $4 5 . 9 _ { + 0 . 1 }$ </td><td rowspan=2 colspan=4> $3 0 . 9 _ { + 0 . 2 }$  $3 1 . 1 _ { + 0 . 4 }$ </td><td rowspan=2 colspan=1> $2 9 . 2 _ { + 0 . 1 }$  $2 9 . 4 _ { + 0 . 3 }$ </td></tr><tr><td rowspan=1 colspan=1> $6 3 . 2 _ { + 0 . 2 }$ </td><td rowspan=1 colspan=1> $4 6 . 1 _ { + 0 . 3 }$ </td></tr><tr><td rowspan=14 colspan=1>OF-9B</td><td rowspan=5 colspan=1>4</td><td rowspan=5 colspan=1>Base+ SymDPO+ General DPO+ Video DPO+ MIA-DPO</td><td rowspan=2 colspan=1> $8 9 . 0$  $\mathbf { 9 3 . 8 _ { + 4 . 8 } }$ </td><td rowspan=2 colspan=1> $6 5 . 8$  $\mathbf { 6 9 . 4 } _ { + 3 . 6 }$ </td><td rowspan=2 colspan=1>54.8 $\pmb { 5 6 . 8 } _ { + 2 . 0 }$ </td><td rowspan=1 colspan=4> $4 0 . 1$ </td><td rowspan=1 colspan=1> $2 8 . 2$ </td></tr><tr><td rowspan=1 colspan=4> ${ \bf 4 1 . 0 _ { + 0 . 9 } }$ </td><td rowspan=1 colspan=1> $2 { \bf 8 . 8 } _ { + 0 . 6 }$ </td></tr><tr><td rowspan=3 colspan=1> $8 9 . 2 _ { + 0 . 2 }$  $8 8 . 7 _ { - 0 . 3 }$  $8 8 . 6 _ { - 0 . 4 }$ </td><td rowspan=1 colspan=1> $6 6 . 4 _ { + 0 . 6 }$ </td><td rowspan=1 colspan=1> $5 5 . 2 _ { + 0 . 4 }$ </td><td rowspan=1 colspan=4> $4 0 . 3 _ { + 0 . 2 }$ </td><td rowspan=1 colspan=1> $2 8 . 5 _ { + 0 . 3 }$ </td></tr><tr><td rowspan=2 colspan=1> $6 5 . 7 _ { - 0 . 1 }$  $6 7 . 5 _ { + 1 . 7 }$ </td><td rowspan=1 colspan=1> $5 4 . 7 _ { - 0 . 1 }$ </td><td rowspan=2 colspan=4> $4 0 . 5 _ { + 0 . 4 }$  $4 0 . 7 _ { + 0 . 6 }$ </td><td rowspan=2 colspan=1> $2 8 . 7 _ { + 0 . 5 }$  $2 8 . 9 _ { + 0 . 7 }$ </td></tr><tr><td rowspan=1 colspan=1> $5 5 . 2 _ { + 0 . 4 }$ </td></tr><tr><td rowspan=5 colspan=1>8</td><td rowspan=5 colspan=1>Base+ SymDPO+ General DPO+ Video DPO+ MIA-DPO</td><td rowspan=5 colspan=1> $\overline { { 9 6 . 3 } }$  $\mathbf { 1 0 2 . 5 } _ { + 6 . 2 }$  $9 6 . 5 _ { + 0 . 2 }$  $9 5 . 7 _ { - 0 . 6 }$  $9 7 . 0 _ { + 0 . 7 }$ </td><td rowspan=2 colspan=1> $\overline { { 6 2 . 9 } }$  ${ \bf 6 7 . 3 _ { + 4 . 4 } }$ </td><td rowspan=1 colspan=1> $\overline { { 5 4 . 8 } }$ </td><td rowspan=2 colspan=4> $\overline { { 4 1 . 1 } }$ </td><td rowspan=2 colspan=1> $\overline { { 2 9 . 1 } }$  ${ \bf 3 0 . 1 _ { + 1 . 0 } }$ </td></tr><tr><td rowspan=1 colspan=1> $\mathbf { 5 5 . 6 } _ { + 0 . 8 }$ </td><td rowspan=1 colspan=2>42.3+1.2</td></tr><tr><td rowspan=1 colspan=1> $6 2 . 9 _ { + 0 . 0 }$ </td><td rowspan=1 colspan=1> $5 5 . 0 _ { + 0 . 2 }$ </td><td rowspan=1 colspan=2> $4 1 . 5 _ { + 0 . 4 }$ </td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=1> $2 9 . 3 _ { + 0 . 2 }$ </td></tr><tr><td rowspan=2 colspan=1> $6 2 . 8 _ { - 0 . 1 }$  $6 3 . 5 _ { + 0 . 6 }$ </td><td rowspan=1 colspan=1> $5 5 . 1 _ { + 0 . 3 }$ </td><td rowspan=2 colspan=4> $4 0 . 2 _ { - 0 . 9 }$  $4 0 . 2 _ { - 0 . 9 }$ </td><td rowspan=2 colspan=1> $2 9 . 0 _ { - 0 . 1 }$  $2 9 . 7 _ { + 0 . 6 }$ </td></tr><tr><td rowspan=1 colspan=1> $5 5 . 3 _ { + 0 . 5 }$ </td></tr><tr><td rowspan=4 colspan=1>16</td><td rowspan=4 colspan=1>Base+ SymDPO+ General DPO+ Video DPO+ MIA-DPO</td><td rowspan=4 colspan=1> $\overline { { 9 8 . 8 } }$  $\mathbf { 1 0 4 . 3 } _ { + 5 . 5 }$  $9 8 . 9 _ { + 0 . 1 }$  $9 8 . 2 _ { - 0 . 6 }$  $9 8 . 5 _ { - 0 . 3 }$ </td><td rowspan=1 colspan=1> $\overline { { 6 2 . 8 } }$  $\mathbf { 6 4 . 9 } _ { + 2 . 1 }$ </td><td rowspan=1 colspan=1>54.3 $5 5 . 7 _ { + 1 . 4 }$ </td><td rowspan=2 colspan=4>42.7 $4 4 . 5 _ { + 1 . 8 }$ </td><td rowspan=2 colspan=1>27.3 ${ \bf 2 8 . 1 _ { + 0 . 8 } }$  $2 7 . 5 _ { + 0 . 2 }$ </td></tr><tr><td rowspan=1 colspan=1> $6 3 . 0 _ { + 0 . 2 }$ </td><td rowspan=1 colspan=1> $5 4 . 5 _ { + 0 . 2 }$ </td><td rowspan=2 colspan=4> $4 2 . 7 _ { + 0 . 0 }$ </td></tr><tr><td rowspan=1 colspan=1> $6 2 . 2 _ { - 0 . 6 }$ </td><td rowspan=1 colspan=1> $5 4 . 6 _ { + 0 . 3 }$ </td><td rowspan=1 colspan=1> $2 6 . 7 _ { - 0 . 6 }$ </td></tr><tr><td rowspan=1 colspan=1> $6 2 . 9 _ { + 0 . 1 }$ </td><td rowspan=1 colspan=1> $5 4 . 8 _ { + 0 . 5 }$ </td><td rowspan=1 colspan=4> $4 3 . 1 _ { + 0 . 4 }$ </td><td rowspan=1 colspan=1> $2 6 . 9 _ { - 0 . 4 }$ </td></tr><tr><td rowspan=17 colspan=1>IDEFICS-9B</td><td rowspan=7 colspan=1>4</td><td rowspan=7 colspan=1>Base+ SymDPO+ General DPO+ Video DPO+ MIA-DPO</td><td rowspan=7 colspan=1>93.0 $\mathbf { 9 6 . 5 } _ { + 3 . 5 }$  $9 3 . 2 _ { + 0 . 2 }$  $9 3 . 5 _ { + 0 . 5 }$  $9 3 . 7 _ { + 0 . 7 }$ </td><td rowspan=6 colspan=1> $5 9 . 7$  ${ \bf 6 4 . 0 } _ { + 4 . 3 }$  $6 0 . 2 _ { + 0 . 5 }$  $5 9 . 6 _ { - 0 . 1 }$ </td><td rowspan=1 colspan=1> $5 5 . 4$ </td><td rowspan=1 colspan=4> $4 5 . 4$ </td><td rowspan=2 colspan=1>27.6</td></tr><tr><td rowspan=1 colspan=1> ${ \bf 5 6 . 1 _ { + 0 . 7 } }$ </td><td rowspan=1 colspan=4> ${ \bf 4 7 . 2 } _ { + 1 . 8 }$ </td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>28.6+1.0</td></tr><tr><td rowspan=3 colspan=1> $5 5 . 6 _ { + 0 . 2 }$ </td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=2 colspan=4> $4 5 . 9 _ { + 0 . 5 }$ </td><td></td></tr><tr><td rowspan=3 colspan=1> $2 7 . 8 _ { + 0 . 2 }$  $2 8 . 0 _ { + 0 . 4 }$  $2 8 . 2 _ { + 0 . 6 }$ </td></tr><tr><td rowspan=1 colspan=1> $5 5 . 7 _ { + 0 . 3 }$ </td><td rowspan=1 colspan=4>45.8+0.4</td></tr><tr><td rowspan=1 colspan=1> $6 1 . 5 _ { + 1 . 8 }$ </td><td rowspan=1 colspan=1> $5 5 . 9 _ { + 0 . 5 }$ </td><td rowspan=1 colspan=4> $4 6 . 3 _ { + 0 . 9 }$ </td></tr><tr><td rowspan=5 colspan=1>8</td><td rowspan=5 colspan=1>Base+ SymDPO+ General DPO+ Video DPO+ MIA-DPO</td><td rowspan=5 colspan=1> $\overline { { 9 7 . 0 } }$  $\mathbf { 1 0 3 . 8 _ { + 6 . 8 } }$  $9 7 . 2 _ { + 0 . 2 }$  $9 7 . 5 _ { + 0 . 5 }$  $9 7 . 7 _ { + 0 . 7 }$ </td><td rowspan=5 colspan=1> $\overline { { 6 1 . 9 } }$  ${ \bf 6 6 . 1 _ { + 4 . 2 } }$  $6 2 . 0 _ { + 0 . 1 }$  $6 2 . 2 _ { + 0 . 3 }$  $6 2 . 5 _ { + 0 . 6 }$ </td><td rowspan=1 colspan=1> $5 6 . 4$ </td><td rowspan=4 colspan=4> $\overline { { 4 7 . 7 } }$  $\mathbf { 4 9 . 5 } _ { + 1 . 8 }$  $4 8 . 1 _ { + 0 . 4 }$  $4 7 . 3 _ { - 0 . 4 }$ </td><td rowspan=4 colspan=1> $\overline { { 2 7 . 5 } }$  $2 { \bf 8 . 5 } _ { + 1 . 0 }$  $2 7 . 7 _ { + 0 . 2 }$  $2 7 . 9 _ { + 0 . 4 }$ </td></tr><tr><td rowspan=1 colspan=1> $\bar { 5 } 7 . 2 _ { + 0 . 8 }$ </td></tr><tr><td rowspan=1 colspan=1> $5 6 . 6 _ { + 0 . 2 }$ </td></tr><tr><td rowspan=1 colspan=1> $5 6 . 2 _ { - 0 . 2 }$ </td></tr><tr><td rowspan=1 colspan=1> $5 6 . 9 _ { + 0 . 5 }$ </td><td rowspan=1 colspan=4> $4 8 . 3 _ { + 0 . 6 }$ </td><td rowspan=1 colspan=1> $2 8 . 1 _ { + 0 . 6 }$ </td></tr><tr><td rowspan=5 colspan=1>16</td><td rowspan=2 colspan=1>Base+ SymDPO</td><td rowspan=2 colspan=1> $\overline { { 9 9 . 7 } }$  $\mathbf { 1 0 7 . 9 } _ { + 8 . 2 }$ </td><td rowspan=2 colspan=1> $6 4 . 5$  ${ \bf 6 9 . 3 } _ { + 4 . 8 }$ </td><td rowspan=1 colspan=1> $\overline { { 5 7 . 0 } }$ </td><td rowspan=1 colspan=4> $\overline { { 4 8 . 4 } }$ </td><td rowspan=2 colspan=1> $\overline { { 2 7 . 9 } }$  $\mathbf { 2 9 . 3 _ { + 1 . 4 } }$ </td></tr><tr><td rowspan=1 colspan=1> ${ \pmb { 5 8 . 2 } } _ { + 1 . 2 }$ </td><td rowspan=1 colspan=4> $\mathbf { 5 0 . 6 } _ { + 2 . 2 }$ </td></tr><tr><td rowspan=3 colspan=1>+ General DPO $+ \mathrm { V i d e o } \mathrm { D P O }$  $+ \mathbf { M } \mathbf { A } { \cdot } \mathbf { D } \mathbf { P } 0$ </td><td rowspan=3 colspan=1> $9 9 . 6 _ { - 0 . 1 }$  $9 9 . 4 _ { - 0 . 3 }$  $9 9 . 8 _ { + 0 . 1 }$ </td><td rowspan=1 colspan=1> $6 4 . 7 _ { + 0 . 2 }$ </td><td rowspan=1 colspan=1> $5 7 . 2 _ { + 0 . 2 }$ </td><td rowspan=1 colspan=4> $4 3 . 8 _ { - 4 . 6 }$ </td><td rowspan=1 colspan=1> $2 8 . 8 _ { + 0 . 9 }$ </td></tr><tr><td rowspan=2 colspan=1> $6 3 . 9 _ { - 0 . 6 }$  $6 4 . 0 _ { - 0 . 5 }$ </td><td rowspan=1 colspan=1> $5 7 . 3 _ { + 0 . 3 }$ </td><td rowspan=2 colspan=4> $4 8 . 3 _ { - 0 . 1 }$  $4 8 . 2 _ { - 0 . 2 }$ </td><td rowspan=2 colspan=1> $2 8 . 0 _ { + 0 . 1 }$  $2 8 . 2 _ { + 0 . 3 }$ </td></tr><tr><td rowspan=1 colspan=1> $5 7 . 5 _ { + 0 . 5 }$ </td></tr></table>

Table 1. Comparison of Different DPO Methods: Performance of m SymDPO dataset from the VQAv2 [5], GQA [20], and ImageNet [13] training sets, amassing a total of 872,000 data items. From this collection, a subset of 10,000 samples is randomly selected for training. To enhance data qual-  
els with various DPO techniques across benchmarks and shot counts. ity, we apply GPT-4v to the selected samples. In the posttraining phase, we employ linear annealing to adjust the learning rate, initializing it at 5e-6. This task is executed on 8 NVIDIA A100 GPUs, with the complete post-training

![](images/a049ae1f4911983c093d0bd7aabde414c56ec87487d177ed745f7941476c8d3d.jpg)

![](images/a6b80b43d43277b46b97cfc4c210ce6bede697a471edf1e30cf39e52d0e09525.jpg)

![](images/368ec7e84db9aad74aa7febdf24b7379099d88e27f931dae18b270a321e72f71.jpg)  
Figure 3. Comparison of Symbol Tuning, General DPO, and SymDPO Methods: We optimized OF 3b using three different methods: Symbol Tuning, General DPO, and SymDPO, resulting in three distinct variants. The performance of these variants was visualized using line charts, showcasing the results across four-shot (4, 8, 16, 32) settings on the COCO, VQAv2, and OK-VQA benchmarks.

Benchmark: Our model is evaluated in alignment with Flamingo on image captioning benchmarks, speci fically COCO Caption [12] and Flickr 30K [57], as well as on three question-answering (QA) benchmarks: VQA v2 [5], OK-VQA [37], and TextVQA [45]. For the image captioning tasks, we report CIDEr scores [48] as evaluation metrics, while for QA tasks, we use accuracy (Acc) as the metric.

Baseline: We compare SymDPO against two different DPO optimization approaches:

• Video DPO [60] - Proposed by LLaVA-Hound-DPO, this approach utilizes a video-specific DPO dataset to improve the model’s understanding of video data.

• MIA-DPO [35] - Designed for multi-image scenarios, this method aims to mitigate hallucinations in LMMs by optimizing in multi-image settings.

## 4.2. Main Results

As illustrated in Table 1, we evaluated the performance of SymDPO, Video DPO, and MIA-DPO on Open-Flamingo (OF) and IDEFICS 9B across five different benchmarks. The results reveal that SymDPO consistently enhances performance across all benchmarks for both OF and IDEFICS, demonstrating the efficacy of SymDPO. In contrast, Video DPO showed no notable improvement, while MIA-DPO yielded only marginal gains. We attribute these outcomes to the specific design focuses of Video DPO and MIA-DPO: Video DPO is primarily oriented toward semantic alignment and optimization for video data, whereas MIA-DPO targets alignment for generic multi-image instructions. Neither approach explicitly addresses the instruction alignment in incontext scenarios, a key focus of SymDPO. We interpret this as a result of the expanded contextual knowledge composed of both visual and textual elements, allowing an LMM finetuned with SymDPO to better integrate and leverage this combined knowledge, thus achieving greater performance gains.

## 4.3. Ablation Study

## 4.3.1. Effectiveness of SymDPO

To further assess the effectiveness of SymDPO, we conducted several ablation experiments.

General DPO vs. SymDPO: As outlined in Method Subsection 3.1, the General DPO approach employs a standard DPO dataset for optimization, without replacing answers with symbols as SymDPO does. As shown in Table 1 and Figure 3, we observed that models optimized with General DPO (i.e., Open-Flamingo (OF) and IDEFICS) exhibit significantly lower performance improvements compared to those optimized with SymDPO. This result substantiates the advantage of the symbolic answer replacement strategy within SymDPO, affirming its effectiveness.

Visual Context Overlook Investigation: To determine whether SymDPO’s enhancements arise from addressing the visual context overlook issue in large multimodal models, we conducted additional tests. Specifically, we modified the demonstration data by either replacing images with blank placeholders (”w/ blank”) or omitting images altogether (”w/o image”). We then evaluated the performance of OF and OF+SymDPO (”OF-s”) under these modified conditions. The results, displayed in Figure 4, reveal that the performance of OF-SymDPO significantly declines when images are removed, suggesting that the model’s advantage derives from its comprehensive understanding of both visual and textual information in in-context demonstrations, rather than relying solely on textual data. This further emphasizes SymDPO’s capability in leveraging the integrated visual-textual knowledge, enabling a more robust and contextually aware model.

Symbol DPO VS. Symbol Finetuning: To validate the advantages of DPO-based optimization, we conducted additional experiments following the preference data collection methodology in Method Subsection 3.1. After collecting the preference dataset, we constructed a multimodal symbolic fine-tuning dataset, inspired by the Symbol Tuning approach [51]. In this setup, we used the chosen answer as the target label for an autoregressive generation task during model fine-tuning, producing the variant OF3B + SymTune. The experimental results, as shown in Table 3, indicate that SymTune does not achieve satisfactory outcomes; notably, its performance on captioning tasks even declines. In contrast, the performance gains of OF3B + SymDPO are substantial across all benchmarks. We attribute this difference to the following key factors: The SymTune approach relies on symbolic fine-tuning where the model learns to predict the chosen answer directly in an autoregressive manner. However, this approach may not fully exploit preference data’s structured feedback, resulting in limited guidance for multimodal alignment.

![](images/b5028523fe2f5b346a8a219102168f46f947f9c40f880bc88c34f7ee2fd24f77.jpg)  
Figure 4. Impact of Visual Context Removal on OF and OF+SymDPO Performance.

<table><tr><td>Model</td><td>Shots</td><td>Method</td><td>COCO Caption (CIDEr)</td><td>Flickr-30K (CIDEr)</td><td>VQAv2 (Acc)</td><td>OK-VQA (Acc)</td><td>TextVQA (Acc)</td></tr><tr><td rowspan="5">OF-3B (I)</td><td>4</td><td>Base + RICES + SymDPO + SymDPO &amp; RICES</td><td>82.7  $9 0 . 5 _ { + 7 . 8 }$   $8 7 . 4 _ { + 4 . 7 }$   $9 3 . 5 _ { + 1 0 . 8 }$ </td><td>59.1  $5 3 . 9 _ { - 5 . 2 }$   $6 1 . 2 _ { + 2 . 1 }$   ${ \bf 6 2 . 0 _ { + 2 . 9 } }$ </td><td>45.7  $4 5 . 3 _ { - 0 . 4 }$   $4 5 . 8 _ { + 0 . 1 }$ </td><td>30.6  $3 1 . 4 _ { + 0 . 8 }$   $3 1 . 6 _ { + 1 . 0 }$ </td><td>28.1  $2 8 . 9 _ { + 0 . 8 }$   $2 8 . 3 _ { + 0 . 2 }$   ${ \bf 2 9 . 1 _ { + 1 . 0 } }$ </td></tr><tr><td>8</td><td>Base + RICES + SymDPO + SymDPO &amp; RICES</td><td>87.8  $9 6 . 8 _ { + 9 . 0 }$   $9 1 . 2 _ { + 3 . 4 }$ </td><td>60.7  $5 8 . 6 _ { - 2 . 1 }$   $6 5 . 3 _ { + 4 . 6 }$ </td><td> $\mathbf { 4 6 . 6 _ { \mathrm { + 0 . 9 } } }$  45.9  $4 6 . 1 _ { + 0 . 2 }$   $4 6 . 5 _ { + 0 . 6 }$ </td><td> $3 3 . 4 _ { + 2 . 8 }$  31.5  $3 2 . 8 _ { + 1 . 3 }$   $3 2 . 7 _ { + 1 . 2 }$ </td><td>29.1  $2 8 . 8 _ { - 0 . 3 }$   $2 9 . 8 _ { + 0 . 7 }$ </td></tr><tr><td rowspan="4">16</td><td>Base</td><td> $9 8 . 4 _ { + 1 0 . 6 }$  91.9</td><td> ${ \bf 6 8 . 2 _ { + 7 . 5 } }$  63.0</td><td> ${ \bf 4 7 . 2 _ { + 1 . 3 } }$  45.8</td><td> $3 4 . 3 _ { + 2 . 8 }$  30.7</td><td> $3 1 . 7 _ { + 2 . 6 }$ </td></tr><tr><td>+ RICES</td><td></td><td></td><td></td><td></td><td>29.1</td></tr><tr><td>+ SymDPO</td><td> $1 0 1 . 1 _ { + 9 . 2 }$   $9 3 . 4 _ { + 1 . 5 }$ </td><td> $6 1 . 5 _ { - 1 . 5 }$   $6 4 . 6 _ { + 1 . 6 }$ </td><td> $4 6 . 6 _ { + 0 . 8 }$   $4 6 . 5 _ { + 0 . 7 }$ </td><td> $3 3 . 9 _ { + 3 . 2 }$ </td><td> $2 8 . 8 _ { - 0 . 3 }$ </td></tr><tr><td>+ SymDPO &amp; RICES</td><td> $\mathbf { 1 0 6 . 8 _ { \div 1 4 . 9 } }$ </td><td> ${ \bf 6 6 . 5 _ { + 3 . 5 } }$ </td><td> ${ \bf 4 7 . 2 _ { + 1 . 4 } }$ </td><td> $3 2 . 8 _ { + 1 . 9 }$   $3 5 . 1 _ { + 4 . 4 }$ </td><td> $2 9 . 6 _ { + 0 . 5 }$   $2 9 . 8 _ { + 0 . 7 }$ </td></tr></table>

Table 2. Performance comparison of the OF-3B (I) model using RICES and SymDPO across various datasets and shot counts.

demos. We then re-evaluated the performance of Open-Flamingo 3B (OF) and $\mathrm { O F } + \mathrm { S y m D P O }$ across different benchmarks. As shown in Table 2, introducing RICES leads to a more pronounced improvement in the SymDPO model. This finding further highlights that incorporating SymDPO enables LMMs to leverage the integrated semantics and knowledge present in the demo’s visual-textual content more effectively.

## 4.3.2. Effect of Different Demonstration Selection Strategies

In the ICL scenario, the choice of demonstration examples (demos) can significantly influence the reasoning performance of large multimodal models. However, prior research has also noted that, due to the issue of visual context overlooks, varying the demo selection does not markedly impact LMM performance. To validate the effectiveness of SymDPO, we employed the RICES (Retrieval In-Context Example Selection) method, as used in Flamingo, to select

## 4.3.3. SymDPO and General DPO Integration: Impact on Task Performance

As depicted in Figure 5, we investigated the effects of integrating SymDPO and General DPO at varying proportions on task-specific performance throughout the alignment phase of DPO optimization. Symbolic data was incrementally introduced into the OF optimization process at ratios of 0%, 30%, 50%, 70%, and 100%, and model performance was subsequently evaluated across benchmark datasets. The experimental results presented in Figure 5 indicate that model performance improves with increasing proportions of symbolic data, particularly within questionanswering (QA) tasks. However, an exclusive reliance on SymDPO data does not yield optimal performance. Our findings show that a 70% symbolic data ratio achieves peak effectiveness; whereas a 100% symbolic data ratio is more effective for the OK-VQA task, suggesting task-specific dependencies on the symbolic data ratio.

![](images/c06ef4ea152c3e3a90882c41ee61808cfe5c90111a97a5dc058eabe641f8209b.jpg)

![](images/8908c904ac51e30e90f636299cb8d1d6c2d9ab38a4cd5517de6d9502ad1b9058.jpg)

![](images/41709e3069c6bcdbb03c98a0caab317a8d0d7318f2f7ae51b089322e275db3b3.jpg)  
Figure 5. Comparison of the Impact of General DPO and SymDPO on LMMs with Varying Data Proportions in the Preference Dataset

![](images/713cdb843355ac5afe8cb0674a5156102dfeb852838da9609d9eeba9453595d1.jpg)  
Figure 6. Example Visualization of OpenFlamingo 3B and OpenFlamingo 3B + SymDPO in ICL 2-Shot Setting.

## 4.4. Case Study

For further quantitative analysis of our method’s effectiveness, we visualized several diverse In-Context Learning (ICL) scenarios. As illustrated in Figure 6, OF+SymDPO consistently yields accurate answers by interpreting the semantic context within demonstrations, whereas OF alone often misinterprets the task, relying predominantly on proximate textual information rather than fully understanding the demonstration content. In the first case, a demonstration provides the names of two pets: a dog named ”Cheems” and a cat named ”Duanwu.” When the model is shown an image of the dog and asked to identify its name, OF+SymDPO accurately answers ”Cheems,” whereas OF responds incorrectly with ”Duanwu,” influenced by nearby textual cues without integrating the visual context. This pattern recurs across other cases, indicating that SymDPO effectively addresses the ”visual context overlook” issue in Large Multimodal Models (LMMs). By doing so, SymDPO enables these models to comprehend and utilize both visual and textual information from demonstrations in a more holistic manner.

## 5. Conclusion

This work presented SymDPO, a symbolic preference optimization method designed to tackle the visual context overlooked in LMMs. By enforcing reliance on both visual and textual cues in In-Context Demonstrations, SymDPO effectively reduces LMMs’ tendency toward textual pattern matching. Experiments confirm that SymDPO improves multimodal comprehension by compelling models to integrate visual context meaningfully, leading to consistent performance gains across benchmarks. In sum, SymDPO provides a robust approach to enhancing multimodal learning, marking a step toward more contextually aware LMMs.

## 6. Acknowledgement

This work is supported by the National Natural Science Foundation of China (NO.623B2007) and CCF-Zhipu Large Model Innovation Fund(NO.CCF-Zhipu202415).

## References

[1] Gpt-4v(ision) system card. 2023. 1, 2

[2] Jacob D. Abernethy, Alekh Agarwal, Teodor Vanislavov Marinov, and Manfred K. Warmuth. A mechanism for sample-efficient in-context learning for sparse retrieval tasks. ArXiv, abs/2305.17040, 2023. 2

[3] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023. 1

[4] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. Advances in neural information processing systems, 35:23716–23736, 2022. 1, 2

[5] Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C Lawrence Zitnick, and Devi Parikh. Vqa: Visual question answering. In Proceedings ofthe IEEE international conference on computer vision, pages 2425– 2433, 2015. 5, 6

[6] Anas Awadalla, Irena Gao, Josh Gardner, Jack Hessel, Yusuf Hanafy, Wanrong Zhu, Kalyani Marathe, Yonatan Bitton, Samir Yitzhak Gadre, Shiori Sagawa, Jenia Jitsev, Simon Kornblith, Pang Wei Koh, Gabriel Ilharco, Mitchell Wortsman, and Ludwig Schmidt. Openflamingo: An open-source framework for training large autoregressive vision-language models. ArXiv, abs/2308.01390, 2023. 1, 2, 4

[7] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023. 1

[8] Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, et al. Constitutional ai: Harmlessness from ai feedback. arXiv preprint arXiv:2212.08073, 2022. 4

[9] Folco Bertini Baldassini, Mustafa Shukor, Matthieu Cord, Laure Soulier, and Benjamin Piwowarski. What makes multimodal in-context learning work? 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pages 1539–1550, 2024. 2

[10] Tom B Brown. Language models are few-shot learners. arXiv preprint arXiv:2005.14165, 2020. 1, 2

[11] Shuo Chen, Zhen Han, Bailan He, Mark Buckley, Philip Torr, Volker Tresp, and Jindong Gu. Understanding and improving in-context learning on vision-language models. ArXiv, abs/2311.18021, 2023. 2

[12] Xinlei Chen, Hao Fang, Tsung-Yi Lin, Ramakrishna Vedantam, Saurabh Gupta, Piotr Dollar, and C Lawrence Zitnick.´ Microsoft coco captions: Data collection and evaluation server. arXiv preprint arXiv:1504.00325, 2015. 6

[13] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009. 5

[14] Hanze Dong, Wei Xiong, Deepanshu Goyal, Yihan Zhang, Winnie Chow, Rui Pan, Shizhe Diao, Jipeng Zhang, Kashun Shum, and Tong Zhang. Raft: Reward ranked finetuning for generative foundation model alignment. arXiv preprint arXiv:2304.06767, 2023. 3

[15] Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Jingyuan Ma, Rui Li, Heming Xia, Jingjing Xu, Zhiyong Wu, Tianyu Liu, et al. A survey on in-context learning. arXiv preprint arXiv:2301.00234, 2022. 1, 2

[16] Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Jingyuan Ma, Rui Li, Heming Xia, Jingjing Xu, Zhiyong Wu, Tianyu Liu, et al. A survey on in-context learning. arXiv preprint arXiv:2301.00234, 2022. 2

[17] Sivan Doveh, Shaked Perek, Muhammad Jehanzeb Mirza, Amit Alfassy, Assaf Arbelle, Shimon Ullman, and Leonid Karlinsky. Towards multimodal in-context learning for vision & language models. ArXiv, abs/2403.12736, 2024. 2

[18] Yaru Hao, Yutao Sun, Li Dong, Zhixiong Han, Yuxian Gu, and Furu Wei. Structured prompting: Scaling in-context learning to 1,000 examples. arXiv preprint arXiv:2212.06713, 2022. 2

[19] Or Honovich, Uri Shaham, Samuel R Bowman, and Omer Levy. Instruction induction: From few examples to natural language task descriptions. arXiv preprint arXiv:2205.10782, 2022. 2

[20] Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6700–6709, 2019. 5

[21] Chaoya Jiang, Jia Hongrui, Haiyang Xu, Wei Ye, Mengfan Dong, Ming Yan, Ji Zhang, Fei Huang, and Shikun Zhang. Maven: An effective multi-granularity hybrid visual encod ing framework for multimodal large language model. arXiv preprint arXiv:2408.12321, 2024. 2

[22] Chaoya Jiang, Hongrui Jia, Mengfan Dong, Wei Ye, Haiyang Xu, Ming Yan, Ji Zhang, and Shikun Zhang. Hal-eval: A universal and fine-grained hallucination evaluation framework for large vision language models. In Proceedings ofthe 32nd ACM International Conference on Multimedia, pages 525– 534, 2024. 1

[23] Chaoya Jiang, Haiyang Xu, Mengfan Dong, Jiaxing Chen, Wei Ye, Ming Yan, Qinghao Ye, Ji Zhang, Fei Huang, and Shikun Zhang. Hallucination augmented contrastive learning for multimodal large language model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 27036–27046, 2024. 1

[24] Dongfu Jiang, Xuan He, Huaye Zeng, Cong Wei, Max Ku, Qian Liu, and Wenhu Chen. Mantis: Interleaved multi-image

instruction tuning. arXiv preprint arXiv:2405.01483, 2024. 2

[25] Timo Kaufmann, Paul Weng, Viktor Bengs, and Eyke Hullermeier. A survey of reinforcement learning from hu-¨ man feedback. ArXiv, abs/2312.14925, 2023. 3

[26] Hugo Laurenc¸on, Lucile Saulnier, Leo Tronchon, Stas Bek-´ man, Amanpreet Singh, Anton Lozhkov, Thomas Wang, Siddharth Karamcheti, Alexander Rush, Douwe Kiela, et al. Obelics: An open web-scale filtered dataset of interleaved image-text documents. Advances in Neural Information Processing Systems, 36, 2024. 1, 2

[27] Hugo Laurenc¸on, Leo Tronchon, Matthieu Cord, and Victor´ Sanh. What matters when building vision-language models? arXiv preprint arXiv:2405.02246, 2024. 2

[28] Harrison Lee, Samrat Phatale, Hassan Mansoor, Kellie Lu, Thomas Mesnard, Colton Bishop, Victor Carbune, and Abhinav Rastogi. Rlaif vs. rlhf: Scaling reinforcement learning from human feedback with ai feedback. In International Conference on Machine Learning, 2023. 3

[29] Bo Li, Yuanhan Zhang, Liangyu Chen, Jinghao Wang, Fanyi Pu, Jingkang Yang, Chunyuan Li, and Ziwei Liu. Mimicit: Multi-modal in-context instruction tuning. arXiv preprint arXiv:2306.05425, 2023. 1, 2

[30] Bo Li, Yuanhan Zhang, Liangyu Chen, Jinghao Wang, Jingkang Yang, and Ziwei Liu. Otter: A multi-modal model with in-context instruction tuning. ArXiv, abs/2305.03726, 2023. 2

[31] Haokun Liu, Derek Tam, Mohammed Muqeeth, Jay Mohta, Tenghao Huang, Mohit Bansal, and Colin Raffel. Few-shot parameter-efficient fine-tuning is better and cheaper than incontext learning. ArXiv, abs/2205.05638, 2022. 2

[32] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning, 2023. 1, 2

[33] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, 2024. 1, 2

[34] Weihao Liu, Fangyu Lei, Tongxu Luo, Jiahe Lei, Shizhu He, Jun Zhao, and Kang Liu. Mmhqa-icl: Multimodal in-context learning for hybrid question answering over text, tables and images. ArXiv, abs/2309.04790, 2023. 2

[35] Ziyu Liu, Yuhang Zang, Xiaoyi Dong, Pan Zhang, Yuhang Cao, Haodong Duan, Conghui He, Yuanjun Xiong, Dahua Lin, and Jiaqi Wang. Mia-dpo: Multi-image augmented direct preference optimization for large vision-language models. arXiv preprint arXiv:2410.17637, 2024. 6

[36] Ziyu Liu, Yuhang Zang, Xiao wen Dong, Pan Zhang, Yuhang Cao, Haodong Duan, Conghui He, Yuanjun Xiong, Dahua Lin, and Jiaqi Wang. Mia-dpo: Multi-image augmented direct preference optimization for large vision-language models. 2024. 2

[37] Kenneth Marino, Mohammad Rastegari, Ali Farhadi, and Roozbeh Mottaghi. Ok-vqa: A visual question answering benchmark requiring external knowledge. In Proceedings of the IEEE/cvf conference on computer vision and pattern recognition, pages 3195–3204, 2019. 6

[38] Sewon Min, Mike Lewis, Hannaneh Hajishirzi, and Luke Zettlemoyer. Noisy channel language model prompt-

ing for few-shot text classification. arXiv preprint arXiv:2108.04106, 2021. 2

[39] Marius Mosbach, Tiago Pimentel, Shauli Ravfogel, Dietrich Klakow, and Yanai Elazar. Few-shot fine-tuning vs. incontext learning: A fair comparison and evaluation. arXiv preprint arXiv:2305.16938, 2023. 2

[40] Catherine Olsson, Nelson Elhage, Neel Nanda, Nicholas Joseph, Nova Dassarma, Tom Henighan, Benjamin Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Dawn Drain, Deep Ganguli, Zac Hatfield-Dodds, Danny Hernandez, Scott Johnston, Andy Jones, John Kernion, Liane Lovitt, Kamal Ndousse, Dario Amodei, Tom B. Brown, Jack Clark, Jared Kaplan, Sam McCandlish, and Christopher Olah. In-context learning and induction heads. ArXiv, abs/2209.11895, 2022. 2

[41] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Car roll Wainwright, Pamela Mishkin, Chong Zhang, Sandhin Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730– 27744, 2022. 1, 3, 4

[42] Renjie Pi, Tianyang Han, Wei Xiong, Jipeng Zhang, Run tao Liu, Rui Pan, and Tong Zhang. Strengthening multi modal large language model with bootstrapped preference optimization. ArXiv, abs/2403.08730, 2024. 2, 4

[43] Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. ArXiv, abs/2305.18290, 2023. 2, 3

[44] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. ArXiv, abs/1707.06347, 2017. 3

[45] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 8317–8326, 2019. 6

[46] Quan Sun, Yufeng Cui, Xiaosong Zhang, Fan Zhang, Qiying Yu, Zhengxiong Luo, Yueze Wang, Yongming Rao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. Generative multimodal models are in-context learners. 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 14398–14409, 2023. 2

[47] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothee Lacroix, Baptiste´ Roziere, Naman Goyal, Eric Hambro, Faisal Azhar, et al.\` Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. 1

[48] Ramakrishna Vedantam, C Lawrence Zitnick, and Devi Parikh. Cider: Consensus-based image description evaluation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4566–4575, 2015. 6

[49] Fei Wang, Wenxuan Zhou, James Y. Huang, Nan Xu, Sheng Zhang, Hoifung Poon, and Muhao Chen. mdpo: Conditional preference optimization for multimodal large language mod els. ArXiv, abs/2406.11839, 2024. 2

[50] Yifan Wang, Qingyan Guo, Xinzhe Ni, Chufan Shi, Lemao Liu, Haiyun Jiang, and Yujiu Yang. Hint-enhanced

in-context learning wakes large language models up for knowledge-intensive tasks. ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 10276–10280, 2023. 2

[51] Jerry Wei, Le Hou, Andrew Lampinen, Xiangning Chen, Da Huang, Yi Tay, Xinyun Chen, Yifeng Lu, Denny Zhou, Tengyu Ma, et al. Symbol tuning improves in-context learning in language models. arXiv preprint arXiv:2305.08298, 2023. 6

[52] Noam Wies, Yoav Levine, and Amnon Shashua. The learnability of in-context learning. ArXiv, abs/2303.07895, 2023. 2

[53] Xu Yang, Yingzhe Peng, Haoxuan Ma, Shuo Xu, Chi Zhang, Yucheng Han, and Hanwang Zhang. Lever lm: Configuring in-context sequence to lever large vision language models. arXiv e-prints, pages arXiv–2312, 2023. 2

[54] Jiabo Ye, Haiyang Xu, Haowei Liu, Anwen Hu, Ming Yan, Qi Qian, Ji Zhang, Fei Huang, and Jingren Zhou. mplugowl3: Towards long image-sequence understanding in multimodal large language models. arXiv e-prints, pages arXiv– 2408, 2024. 1, 2

[55] Qinghao Ye, Haiyang Xu, Guohai Xu, Jiabo Ye, Ming Yan, Yiyang Zhou, Junyang Wang, Anwen Hu, Pengcheng Shi, Yaya Shi, et al. mplug-owl: Modularization empowers large language models with multimodality. arXiv preprint arXiv:2304.14178, 2023.

[56] Qinghao Ye, Haiyang Xu, Jiabo Ye, Ming Yan, Anwen Hu, Haowei Liu, Qi Qian, Ji Zhang, and Fei Huang. mplug-owl2: Revolutionizing multi-modal large language model with modality collaboration. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13040–13051, 2024. 1, 2

[57] Peter Young, Alice Lai, Micah Hodosh, and Julia Hockenmaier. From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions. Transactions of the Association for Computational Linguistics, 2:67–78, 2014. 6

[58] Zheng Yuan, Hongyi Yuan, Chuanqi Tan, Wei Wang, Songfang Huang, and Fei Huang. Rrhf: Rank responses to align language models with human feedback without tears. arXiv preprint arXiv:2304.05302, 2023. 3

[59] Chaoyi Zhang, Kevin Qinghong Lin, Zhengyuan Yang, Jianfeng Wang, Linjie Li, Chung-Ching Lin, Zicheng Liu, and Lijuan Wang. Mm-narrator: Narrating long-form videos with multimodal in-context learning. 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 13647–13657, 2023. 2

[60] Ruohong Zhang, Liangke Gui, Zhiqing Sun, Yihao Feng, Keyang Xu, Yuanhan Zhang, Di Fu, Chunyuan Li, Alexander Hauptmann, Yonatan Bisk, et al. Direct preference optimization of video large multimodal models from language model reward. arXiv preprint arXiv:2404.01258, 2024. 6

[61] Haozhe Zhao, Zefan Cai, Shuzheng Si, Xiaojian Ma, Kaikai An, Liang Chen, Zixuan Liu, Sheng Wang, Wenjuan Han, and Baobao Chang. Mmicl: Empowering vision-language model with multi-modal in-context learning. ArXiv, abs/2309.07915, 2023. 2