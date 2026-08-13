# Hybrid-Level Instruction Injection for Video Token Compression in Multi-modal Large Language Models

Zhihang Liu¹, Chen-Wei Xie2, Pandeng Li1,2† Liming Zhao2, Longxiang Tang3, Yun Zheng2, Chuanbin Liu¹, Hongtao Xie1 1 University of Science and Technology of China 2 Tongyi Lab, Alibaba Group 3 Tsinghua University {1iuzhihang, lpd}@mail.ustc.edu.cn, htxie@ustc.edu.cn

## Abstract

Recent Multi-modal Large Language Models (MLLMs) have been challenged by the computational overhead resulting from massive video frames, often alleviated through compression strategies. However, the visual content is not equally contributed to user instructions, existing strategies (e.g., average pool) inevitably lead to the loss of potentially useful information. To tackle this, we propose the Hybridlevel Instruction Injection Strategy for Conditional Token Compression in MLLMs (HICom), utilizing the instruction as a condition to guide the compression from both local and global levels. This encourages the compression to retain the maximum amount of user-focused information while reducing visual tokens to minimize computational burden. Specifically, the instruction condition is injected into the grouped visual tokens at the local level and the learnable tokens at the global level, and we conduct the attention mechanism to complete the conditional compression. From the hybridlevel compression, the instruction-relevant visual parts are highlighted while the temporal-spatial structure is also preserved for easier understanding of LLMs. To further unleash the potential of HICom, we introduce a new conditional pre-training stage with our proposed dataset HICom-248K. Experiments show that our HICom can obtain distinguished video understanding ability with fewer tokens, increasing the performance by 2.43% average on three multiple-choice QA benchmarks and saving 78.8% tokens compared with the SOTA method. The code is available at https://github.com/lntzm/HICom.

## 1. Introduction

Multi-modal Large Language Models [2, 14, 22, 31, 32, 74] (MLLMs) have gained significant improvements in multimodal understanding by integrating visual information into the LLMs, beating various expert methods on downstream tasks [12, 13, 25, 26, 47, 61, 65, 70]. While initially focused on image understanding, more research [30, 34, 36, 40, 56, 57] has shifted towards challenging video tasks. Most MLLMs handle videos by treating them as sequences of images, sampling frames, and concatenating visual tokens from these frames [6, 20, 42, 52, 66].

![](images/81aa4996a593e0126a800ad104cd709e4c76c48b4212b5f74e0e3cf5962991a9.jpg)  
Figure 1. An example of the video understanding task, and the comparison between the unconditional compression and our proposed conditional compression with hybrid-level instruction injection. We inject instruction at both local and global levels, guiding the compression to retain the maximum amount of user-focused information and minimize the computational burden.

Compared to images, videos comprise multiple frames, resulting in more visual tokens and thus higher computational costs. To make the computation affordable, early methods [23, 30, 34] often employ sparse temporal sampling strategies, leading to significant loss of temporal information. Consequently, current MLLMs focus on compressing visual tokens to achieve a trade-off between computational costs and video understanding ability. Mainstream MLLMs [20, 59, 71] use a spatial pooling strategy within each frame, leveraging capabilities gained from image data. Some approaches [19, 43, 67] use Q-Former [22] or Resampler [1] to compress the spatial-temporal information into a fixed number of visual tokens. Other methods employ convolution [7], clustering [17], or memory mechanisms [16, 46, 68] to compress visual tokens in both spatial and temporal dimensions. However, these compression methods are unconditional, lacking explicit guidance, as illustrated in Fig. 1 (b), making it challenging to achieve high compression ratios with minimal information loss. Consequently, user-relevant visual content may be overlooked during such unconditional compression.

Therefore, we focus on introducing the instruction information as a condition to perform targeted compression, as it is rarely explored in previous work. As shown in Fig. 1 (c), conditional compression uses explicit instructional guidance to select user-focused information for retention, thereby enhancing efficiency. To thoroughly explore the conditional compression, we propose Hybridlevel Instruction Injection Strategy for Conditional Token Compression in MLLMs (HICom). Observing how humans process complex visual information, they usually adopt a coarse-to-fine approach, first considering the relative parts in the global coarse impression and then seeking detailed information locally in the video. Motivated by this human perception process, HICom conducts the compression at both local and global levels, retaining the maximum amount of instruction-relevant information while preserving the temporal-spatial structure with fewer tokens, thus achieving a better balance between the computational burden and video comprehension.

Specifically, at the local level, we divide the sampled frame features into several groups and compress the tokens of each group into one token based on the attention with the injected instruction condition. Different from Q-Former [22] and Resampler [1], the grouped local attention can explicitly maintain the spatial-temporal structural characteristics of the video while highlighting the relevant parts within each group. At the global level, we inject the instruction condition into a small number of learnable tokens and then conduct the attention with the flattened frame features without grouping. The global compression focuses more on searching instruction-relevant information in the whole video and thus can be expressed by a small number of tokens as an auxiliary. To further unleash the potential of HICom, we introduce a new conditional pre-training stage between the alignment and instruction tuning and construct a new dataset HICom-248K of 248K video clips with high-quality instruction-followed descriptions. The proposed new stage continues to pre-train the parameters of condition injection, further improving the performance. Experiments show that our HICom can achieve state-of-the-art performance on 5 benchmarks with significantly fewer tokens. (e.g., increasing by 2.43% average on three multiplechoice QA benchmarks and saving 78.8% tokens compared with LLaVA-Video-7B [72]).

Our main contributions are listed as follows:

• We propose a conditional compression method HICom for MLLMs, achieving effective video understanding while significantly reducing the computational burden.

• We conduct hybrid-level instruction injection to achieve the conditional compression, and introduce a new conditional pre-training stage with our constructed HICom-248K dataset to further unleash the potential of the conditional compressing.

• Experiments on various benchmarks show the proposed HICom can achieve better performance with fewer visual tokens, providing valuable insights for MLLMs.

## 2. Related Work

Multi-modal large language models. Based on the powerful capabilities of LLMs among various linguistic tasks, many works try to extend the understanding abilities to the computer vision area. Flamingo [1] and BLIP series [8, 22] successfully explore the usage of Resampler and Q-Former to bridge the visual tokens to language models. LLaVA series [31–33] use a simple MLP as the visual projector to translate visual tokens to the LLM's embedding space, achieving huge success. Recently, more researchers have focused on extending image tasks to video tasks. The biggest challenge for video tasks is to design an effective method to achieve the trade-off between computational costs and video understanding ability. Early methods [23, 30, 34] sparsely sample frames, lossing too much temporal information. Mainstream methods [20, 59, 71] use a simple spatial pooling strategy to reduce the number of visual tokens and get a huge success. [19, 67] employ Q-Former to compress the spatial-temporal tokens into a fixed number. Chat-Univi [17] uses DPC-KNN algorithm to cluster dynamic visual tokens. VideoLLaMA2 [7] utilizes convolution both in temporal and spatial dimensions to downsample tokens. Flash-VStream [68] introduces a learnable memory mechanism to compress online video streaming. All of them try their best to observe more visual information. However, their unconditional compression without explicit guidance inherently leads to the loss of instructionrelevant information. Different from them, we conduct conditional compression by injecting instruction at hybrid levels, only emphasizing the visual parts related to the instruction and allowing the irrelevant parts to be discarded, thus getting a better representation for each question.

Text-based visual representation for video LLMs. There are some methods using instruction information during the visual encoding. [53, 63] sample the frames based on the CLIP [41] similarity or a localization model using the instruction, [29, 50] make the frame selection strategy trainable. However, these methods only focus on how to select the correct frames in a rough manner, failing to guide the token compression with the key instruction information. LLaMA-VID [28] follows a similar instruction-guided idea, as it introduces the instruction to interact with visual features. However, it roughly simplifies the interacted but uncompressed result into a single token only as auxiliary information for each frame, destroying the spatial structure and failing to discuss the guiding role of the instruction for conditional compression. Therefore, it is only designed for long videos and the performance is limited. To fully explore conditional compression, we propose to inject the instruction at hybrid levels, exploring a better way for conditional compression to alleviate the information loss while maintaining the performance.

![](images/e877d0bf4224ca78f7438bb111d9786ba3e2a32ec979dfdb73701e186b045a65.jpg)  
Figure 2. The framework of our proposed HICom. We propose the hybrid-level instruction injection to conditionally compress video tokens in MLLMs. We extract instruction-relevant information within each grouped sub-region at the local level, and extract it to a fixed number of tokens at the global level. The instruction condition is injected into the attention process to guide the compression.

## 3. Method

## 3.1. Instruction-Guided Conditional Compression

Fig. 2 shows the framework of our proposed HICom, which focuses on the design between the vision encoder and the LLM to conduct the conditional compression. We inject the instruction condition into the compression process at both local and global levels for better information extraction with spatial-temporal structure maintained.

Local-Level Compression. Previous works have demonstrated the importance of the spatial inductive bias when processing visual representations for image MLLMs [27, 49]. Inspired by them, we divide the tokens into several groups and perform conditional compression within each group to preserve the temporal-spatial structure of video features. Specifically, given the encoded video frame features $V ~ \in ~ \hat { \mathbb { R } } ^ { T \times H \times W \times \hat { D } }$ , and the downsampling ratio $( \alpha _ { T } , \alpha _ { H } , \alpha _ { W } )$ , we divide V into $N _ { T } { \cdot } N _ { H } { \cdot } N _ { W }$ groups, where $N _ { i } = \lceil i / \alpha _ { i } \rceil , i \in \{ T , H , W \}$ . Therefore, there are αT·αH·αw tokens in each group, which can be formulated as $V ^ { t , h , w } ~ \in$ RαT×αH×αw×D, where $0 ~ \le ~ \{ t , h , w \} ~ \le$ $\{ N _ { T } , N _ { H } , N _ { W } \}$ We first pool $V ^ { t , h , w }$ into one token $\mathring { V } _ { p } ^ { t , h , w } \in \mathbb { R } ^ { 1 \times \bar { D } }$ and inject the instruction condition $C \left( i . e . \right.$ the text feature of instructions) into $V _ { p } ^ { t , h , w }$ . Then the local attention within each group can be calculated by:

$$
\begin{array} { r } { \boldsymbol { Z } _ { l } ^ { t , h , w } = \mathrm { A t t n } \left[ \operatorname { I n j } ( \boldsymbol { V } _ { p } ^ { t , h , w } , \boldsymbol { C } ) , \boldsymbol { V } ^ { t , h , w } , \boldsymbol { V } ^ { t , h , w } \right] , } \end{array}\tag{1}
$$

where Inj(·) represents the condition injection, Attn(·) donates the attention mechanism [9] and the inputs of Attn(·) are query, key, and value, respectively. Therefore, the visual features within each group are compressed into only one token, preserving the temporal-spatial structure while conditionally highlighting the instruction-relevant parts. Finally, we concatenate $Z _ { l } ^ { t , { \breve { h } } , w }$ as the local compressed results $\pmb { Z } _ { l } \ \in \ \mathbb { R } ^ { N _ { T } \times N _ { H } \times N _ { W } \times \mathring { D } }$ , and use an MLP layer to project $Z _ { l }$ into the LLM's embedding space.

Global-Level Compression. Though the temporal-spatial structure can be maintained at the local level, the attention is forced to focus on small sub-regions. However, each subregion may contain information that is not equally valid for the question, and some sub-regions are even quite irrelevant. Therefore, we also implement conditional compression at the global level to highlight the most relevant parts within the whole video. Specifically, we initialize a small set of learnable tokens $\bar { \mathbf { L } } \in \mathbb { R } ^ { N _ { L } \times D }$ , where $N _ { L }$ donates the number of tokens, and then inject the instruction condition into them. Instead of grouping the video frame features, we apply 3D position embedding Pos(·) for them and directly flatten them at global-level compression. Then the global compressed tokens $\boldsymbol { Z } _ { g } \in \mathbb { R } ^ { N _ { L } \times \tilde { D } }$ can be calculated by:

![](images/9672e88ee808a1081bc5f9bc273770017052d4bdb607a87c5295537204352e2a.jpg)  
Figure 3. We introduce a new guidance pre-training stage and implement three-stage training for conditional compression.

$$
Z _ { g } = \mathrm { A t t n } \left[ \mathrm { I n j } ( L , C ) , V + \mathrm { P o s } ( V ) , V \right] .\tag{2}
$$

An MLP layer is also utilized to project $Z _ { g }$ into the LLM's embedding space. And we finally concatenate $Z _ { l }$ and $Z _ { g }$ together for further understanding of LLM.

Instruction Condition Injection. In order to fulfill the guiding role of the instruction condition, we explore three different types of injection modules, and define them direct injection, coarse injection, and fine injection, respectively. Note the text encoder can obtain both pooled tokens $C _ { p } \in \mathbb { R } ^ { 1 \times D }$ (i.e., global text embedding) and finegrained tokens $C _ { f } \in \mathbb { R } ^ { L \times D } ( i . e .$ , token-level text embedding). Given the pooled token $C _ { p } ,$ the direct injection uses an MLP to translate the single token into the injected output without any interaction with $V _ { p } ^ { t , h , w }$ or L. We employ coarse injection by the adaptive layer norm [39]. Specifically, we use an MLP to regress the scale and shift from $C _ { p } ,$ then add to the visual input $V _ { p } ^ { t , h , w }$ or learnable tokens L, which we represent them as A, after the layer norm. The process can be formulated as:

$$
\operatorname { I n j } ( A , C ) = \operatorname { L N } ( A ) \cdot \operatorname { s c a l e } ( C _ { p } ) + \operatorname { s h i f t } ( C _ { p } ) ,\tag{3}
$$

Table 1. The statistical information of our constructed HICom-248K dataset.  
![](images/8d548cd4887c9c69e4fcfc23fc5905b0b115629a15221ba34467bd0d6d63deb7.jpg)  
Figure 4. The visualization of data source (left) and video length (right) of our constructed HICom-248K dataset.

where LN(·) is the layer norm. Given the fine-grained tokens $C _ { f }$ as condition C, we conduct the fine injection via the cross attention, which can be formulated as:

$$
\operatorname { I n j } ( A , C ) = \operatorname { A t t n } \left( \operatorname { L N } ( A ) , C _ { f } , C _ { f } \right) .\tag{4}
$$

As the direct injection directly translates the condition to the query of the attention in Eq. (1) and Eq. (2), the token length limits that it is only suitable for local compression as the local branch compresses each group into only one token. On the contrary, the coarse and fine injections are more flexible to fit various situations as the condition is added into A. We conduct the experiments on the three types of injection and finally choose direct injection for local compression and coarse injection for global compression.

Conditional Pre-training Stage. The existing mainstream methods follow the pipeline of alignment first and then instruction tuning [7, 31, 32, 68], and previous text-based methods [28, 53] also follow this pipeline. However, the image-caption pairs cannot provide valid instruction information in the alignment stage, direct instruction tuning with modules that are not adequately aligned will confuse the guiding process, thus harming the performance. Therefore, we propose a new conditional pre-training stage between the alignment and the instruction tuning to make the training process easier, as shown in Fig. 3. In this stage, we use our constructed instruction-followed descriptions to pre-train the compression module with condition injection, fulfilling the guiding role of the instruction.

## 3.2. Dataset Construction

To pre-train the condition injection in the compression module, we construct a new instruction-followed description dataset HICom-248K. Different from common instruction tuning datasets, HICom-248K focuses on providing only one data type, i.e., the description of the visual content related to the instruction, and does not include other rich types of instruction data (e.g., reasoning, counting, etc).

Table 2. Performance comparison between our HICom and other SOTA methods on multiple-choice QA video benchmarks. † means we use the result reproduced by VideoLLaMA2 [7], \* indicates we reproduce the results ourselves using the official checkpoint and inference code provided by authors. § donates we inference with a new length of frames trained by sampling 32 frames.
<table><tr><td rowspan="2">Methods</td><td rowspan="2">LLM Size # Frames # Tokens</td><td rowspan="2"></td><td rowspan="2"></td><td colspan="4">VideoMME w/o sub.</td><td colspan="4">VideoMME w/ sub.</td><td rowspan="2">MV- Bench</td><td rowspan="2">Ego- Schema</td></tr><tr><td>short</td><td>mid</td><td>long</td><td>all</td><td>short</td><td>mid</td><td>long</td><td>all</td></tr><tr><td>Video-LLaVA [30]</td><td>7B</td><td>8</td><td>2048</td><td>45.3</td><td>38.0</td><td>36.2</td><td>39.9</td><td>46.1</td><td>40.7</td><td>38.1</td><td>41.6</td><td>43.0</td><td></td></tr><tr><td>VideoChat2-Mistral [24]</td><td>7B</td><td>16</td><td>1536</td><td>48.3</td><td>36.3</td><td>35.0</td><td>39.5</td><td>52.8</td><td>39.4</td><td>39.2</td><td>43.8</td><td>60.4</td><td></td></tr><tr><td>LLaMA-VID [28]</td><td>7B</td><td>1fps</td><td>2tps</td><td></td><td></td><td></td><td>25.9†</td><td></td><td></td><td></td><td></td><td>41.9†</td><td>38.5†</td></tr><tr><td>Chat-Univi-1.5 [17]</td><td>7B</td><td>64</td><td>448</td><td>45.7</td><td>40.3</td><td>35.8</td><td>40.6</td><td>51.2</td><td>44.6</td><td>41.8</td><td>45.9</td><td>45.9</td><td></td></tr><tr><td>PLLaVA [59]</td><td>7B</td><td>16</td><td>2304</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>46.6</td><td></td></tr><tr><td>LLaVA-Next-Video [71]</td><td>7B</td><td>32</td><td>4608</td><td>49.9*</td><td>41.4*</td><td>37.0*</td><td>42.8*</td><td></td><td></td><td></td><td></td><td>46.5†</td><td>43.9†</td></tr><tr><td>LongVA [69]</td><td>7B</td><td>128</td><td>18432</td><td>61.1</td><td>50.4</td><td>46.2</td><td>52.6</td><td>61.6</td><td>53.6</td><td>47.6</td><td>54.3</td><td></td><td></td></tr><tr><td>VideoLLaMA2 [7]</td><td>7B</td><td>16</td><td>1152</td><td>56.0</td><td>45.4</td><td>42.1</td><td>47.9</td><td></td><td></td><td></td><td></td><td>54.6</td><td>51.7</td></tr><tr><td>Tarsier [51]</td><td>7B</td><td>16</td><td>2304</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>62.6</td><td>56.0</td></tr><tr><td>VITA [11]</td><td>8×7B</td><td>16</td><td>4096</td><td>65.9</td><td>52.9</td><td>48.6</td><td>55.8</td><td>70.4</td><td>56.2</td><td>50.9</td><td>59.2</td><td></td><td></td></tr><tr><td>LLaVA-One Vision [20]</td><td>7B</td><td>32</td><td>6272</td><td>70.1*</td><td>56.4*</td><td>48.9*</td><td>58.5*</td><td>75.8*</td><td>58.4*</td><td>51.6*</td><td>61.9*</td><td>56.7</td><td>60.1</td></tr><tr><td>LLaVA-Video [72]</td><td>7B</td><td>32</td><td>6272</td><td>73.9*</td><td>57.3*</td><td>50.4*</td><td>60.6*</td><td>76.6*</td><td>60.3*</td><td>51.7*</td><td>62.9*</td><td>58.6</td><td>57.3</td></tr><tr><td>HICom (Ours)</td><td>1.5B</td><td>32</td><td>680</td><td>61.7</td><td>51.3</td><td>43.9</td><td>52.3</td><td>65.1</td><td>54.7</td><td>47.3</td><td>55.7</td><td>56.4</td><td>50.1</td></tr><tr><td>HICom (Ours)</td><td>7B</td><td>32</td><td>680</td><td>65.7</td><td>57.1</td><td>51.4</td><td>58.1</td><td>68.4</td><td>58.9</td><td>53.1</td><td>60.1</td><td>64.1</td><td>60.5</td></tr><tr><td>HICom (Ours)§</td><td>7B</td><td>64</td><td>1328</td><td>67.6</td><td>58.0</td><td>51.2</td><td>58.9</td><td>71.8</td><td>63.7</td><td>54.5</td><td>63.3</td><td>64.7</td><td>62.2</td></tr><tr><td>HICom (Ours)§</td><td>7B</td><td>128</td><td>2624</td><td>69.0</td><td>60.2</td><td>51.5</td><td>60.3</td><td>72.7</td><td>64.6</td><td>57.3</td><td>64.8</td><td>64.0</td><td>62.3</td></tr></table>

Data Collection. To ensure the diversity and quality of our videos, we collect videos from public datasets Panda-70M [5] and Ego4D [15]. Panda-70M contains various open-domain videos from YouTube, and Ego4D collects many high-resolution ego-centric human activity videos. Notably, we use the original untrimmed videos to cut clips ourselves since the provided split clips of both datasets are too short, greatly decreasing the content complexity. To balance the number of different types of videos, we pre-define 29 categories [10, 72] using natural language (e.g. A video about cooking activity.) and use InternVideo2 [54] to extract both the video features and the pre-defined sentence features. Based on the similarities between them, we select 1,500 videos for each category and then randomly select additional 10,000 videos from the others to ensure diversity.

Video Processing. The untrimmed videos are too long for our pre-training, and also severely affect the quality of captions and annotations by existing SOTA models. Therefore, we use PySceneDetect to split each video into shorter ones with a much looser threshold than Panda-70M. As the split clips can sometimes be redundant with repeat semantics, we further extract keyframes for each video and only select the split clips containing the keyframes as our final results. Specifically, inspired by ShareGPT4Video [4], we calculate the CLIP score for densely sampled frames within a video, and select the less similar frames as keyframes. Finally, we further filter out the video clips shorter than 5 seconds and longer than 120 seconds.

Instruction-Followed Description Generation. For the processed video clips, we use Qwen2-VL-72B-Instruct [52] to systematically generate the instruction-followed description with the CoT technique [55]. Specifically, we first generate the detailed caption of each input video, then take both the video content and caption as input to generate the three instruction-answer pairs for each video. Different from the normal instruction datasets, we require Qwen2-VL-72B-Instruct to meet the following two rules: 1) The instructions should refer to the specific visual information, such as the people or the objects in the video; 2) The instructions should lead to a descriptive response and the answers should describe the object mentioned in the instruction in detail. Finally, we filter the generated results by discarding answers containing invalid information, such as the phrase "not provide", or "not describe".

Dataset Statistics. Tab. 1 and Fig. 4 shows some statistics about our constructed dataset. We collect 248K video clips in total from Panda-70M and Ego4D, the average of the video lengths is 25.67 seconds and the distribution is shown in Fig. 4. With the help of Qwen2-VL-72B-Instruct, we generate 739K instruction-followed descriptions, providing sufficient data for our guidance pre-training.

## 4. Experiments

## 4.1. Experimental Setup

Implementation Details. We use SigLIP [64] (so400mpatch14-384) as our vision encoder and text encoder, choose Qwen2.5 series [48] as our LLMs, and randomly initialize the compressor. The vision encoder keeps frozen at all stages, the LLM is frozen at the two pre-train stages, and is fine-tuned at the instruction tuning stage. We follow LLaVA-OneVision [20] to choose our training configurations, the global batch size is set to 512 at the alignment stage, 256 at the conditional pre-train and instruction tuning stage. We use the learning rate of 1e-3 at the alignment stage, 1e-5 at the instruction tuning stage. At our proposed conditional pre-training stage, we use 1e-3 for the condition injection sub-module and 1e-4 for other parameters in the compressing module. We train 1 epoch for all stages. More implementation details can be found in Appendix A. Datasets. We use LLaVA-558K [32] image-caption pairs for our alignment stage, the constructed 248K HICom-Pretrain instruction-followed descriptions for conditional pre-train stage. Inspired by [51, 72], we collect 2.68M video instruction data for instruction tuning, including 1.6M from LLaVA-Video [72], 292K from VideoChat2-IT [23] subset, 255K from M4-Instruct-Data [21], 11K from Charades [45], 114K from NTU RGB+D [44], 122K from TVQA [18], and 290K from MiT [38]. We evaluate our model on 5 video benchmarks, including 3 multiplechoice benchmarks VideoMME [10], MVBench [24], EgoSchema [37], and 2 open-ended benchmarks ActivityNet [3], VideoChatGPT Bench [36].

Table 3. Performance of our HICom and other SOTA methods on open-ended QA video benchmarks. All the compared models are 7B. †, \*, and § keep the same meaning with Tab. 2. # F., # T. are short for # Frames, # Tokens, ANet is short for ActivityNet, LLaVA-NV, LLaVA-OV are short for LLaVA-Next-Video, and LLaVA-OneVision. The format we report ANet is Acc/Score.
<table><tr><td rowspan="2">Methods</td><td rowspan="2">#F. #T.</td><td rowspan="2"></td><td rowspan="2">ANet</td><td colspan="6">VideoChatGPT Bench</td></tr><tr><td>CI</td><td>DO</td><td>CU</td><td>TU</td><td>CO</td><td>Avg</td></tr><tr><td>Video-LLaVA [30]</td><td>8</td><td>2048</td><td>45.3/3.3</td><td></td><td></td><td>1</td><td></td><td></td><td></td></tr><tr><td>VideoChat2 [24]</td><td>16</td><td>1536</td><td>49.1/3.3</td><td>3.02</td><td>2.88</td><td>3.51</td><td>2.66</td><td>2.81</td><td>2.98</td></tr><tr><td>Chat-Univi [17]</td><td>64</td><td>448</td><td>45.8/3.2</td><td>2.89</td><td>2.91</td><td>3.46</td><td>2.89</td><td>2.81</td><td>2.99</td></tr><tr><td>LLaMA-VID [28]</td><td></td><td>1fps 2tps</td><td>47.4/3.3</td><td>2.96</td><td>3.00</td><td>3.53</td><td>2.46</td><td>2.51</td><td>2.89</td></tr><tr><td>LongVLM [58]</td><td>100</td><td>305</td><td>47.6/3.3</td><td>2.76</td><td>2.86</td><td>3.34</td><td>2.39</td><td>3.11</td><td>2.89</td></tr><tr><td>ST-LLM [35]</td><td>16</td><td>512</td><td>50.9/3.3</td><td>3.23</td><td>33.05</td><td>3.74</td><td>2.93</td><td>2.81</td><td>3.15</td></tr><tr><td>PLLaVA [59]</td><td>16</td><td>2304</td><td>56.3/3.5</td><td>3.21</td><td>2.86</td><td>3.62</td><td>2.33</td><td>2.93</td><td>2.99</td></tr><tr><td>LLaVA-NV [71]</td><td>32</td><td>4608</td><td>53.5/3.2</td><td>3.39</td><td>3.29</td><td>3.92</td><td>2.6</td><td>3.12</td><td>3.26</td></tr><tr><td>LongVA [69]</td><td></td><td>128 18432</td><td>-/2.8</td><td>3.05</td><td>3.09</td><td>3.77</td><td>2.44</td><td>3.64</td><td>3.20</td></tr><tr><td>VideoLLaMA2 [7]</td><td></td><td>16 1152</td><td>50.2/3.3</td><td>3.16</td><td>3.08</td><td>3.69</td><td>2.56</td><td>3.14</td><td>3.13</td></tr><tr><td>Tarsier [51]</td><td></td><td>16 2304</td><td>59.5/3.6</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SF-LLaVA [60]</td><td>50</td><td>3680</td><td>56.3/3.4</td><td>3.09</td><td>2.70</td><td>3.57</td><td>2.52</td><td>3.35</td><td>3.04</td></tr><tr><td>LLaVA-OV [20]</td><td>32</td><td>6272</td><td>56.6/3.6*</td><td>3.45*</td><td>3.00*</td><td>3.71*</td><td>2.68*</td><td>3.14*</td><td>3.20*</td></tr><tr><td>HICom(Ours)-1.5B</td><td>32</td><td>680</td><td>53.0/3.5</td><td>3.09</td><td>2.67</td><td>3.40</td><td>2.41</td><td>2.98</td><td>2.91</td></tr><tr><td>HICom(Ours)-7B</td><td>32</td><td>680</td><td>58.3/3.7</td><td>3.29</td><td>2.85</td><td>3.59</td><td>2.67</td><td>3.22</td><td>3.12</td></tr><tr><td>HICom(Ours)-7B§</td><td>64</td><td>1328</td><td>59.4/3.7</td><td>3.32</td><td>2.92</td><td>3.65</td><td>2.74</td><td>3.35</td><td>3.20</td></tr><tr><td>HICom(Ours)-7B§</td><td></td><td>128 2624</td><td>59.5/3.7</td><td>3.33</td><td>2.86</td><td>3.65</td><td>2.74</td><td>3.32</td><td>3.18</td></tr></table>

## 4.2. Main Results

Multiple-choice benchmarks. In Tab. 2, we compare the performance of our HICom with different SOTA models on three multiple-choice video benchmarks. We sample 32 frames with 680 tokens to train all our models and sample different frames for inference. Our HICom-7B with 2624 tokens inference achieves the best performance on all three benchmarks. Compared to LLaVA-Video with 6272 tokens, our HICom with only 1328 tokens can obtain better performance, increasing the performance by 2.43% average and saving 78.8% tokens, HICom with 2624 tokens gains 3% averagely when saving 58.2% tokens, significantly lowering the computational burden. Notably, LLaVA-Video uses numerous additional image data, and it also unfreezes the vision encoder during training, which can both increase performance. It is also noteworthy that though our HICom performs a little worse than LLaVA-Video on VideoMME short videos (less than 2 minutes), we beat LLaVA-Video on both VideoMME medium (4-15 minutes) and long videos (30-60 minutes). We argue that conditional compression is more helpful for long videos. For short videos, the sampled frames are dense enough for unconditional compression, baselines with enough information would naturally perform powerfully. But for long videos, the sampled frames are more sparse with less redundant, our HICom can preserve the maximum amount of instruct-relevant information while it is easily suppressed in LLaVA-Video.

Table 4. The ablation study on our proposed HICom, $\alpha _ { ( T , H , W ) }$ donates the downsampling ratio of each dimension, # Tok. donates the number of tokens input to LLMs. 8f means we sample 8 frames for training. The conditional compression combining both local and global achieves the best.
<table><tr><td rowspan="2">Methods</td><td rowspan="2"> $\alpha _ { ( T , H , W ) }$ </td><td rowspan="2"># Tok.</td><td colspan="2">VideoMME w/o sub.</td><td rowspan="2">MV-</td><td rowspan="2">Ego- BenchSchema</td></tr><tr><td>short mid long</td><td>all</td></tr><tr><td colspan="7">Unconditional Compression</td></tr><tr><td>avg pool</td><td>(1,3,3)</td><td>2592</td><td>39.3 34.9 33.035.7</td><td></td><td>44.8</td><td>43.2</td></tr><tr><td>8f, avg pool avg pool</td><td>(1,3,3) (4,3,3)</td><td>648 648</td><td>36.334.4 32.034.3 36.3 33.3 32.2 34.0</td><td></td><td>43.6 43.7</td><td>42.7 41.9</td></tr><tr><td>local</td><td>(4,3,3)</td><td>648</td><td>36.7 34.4 32.0 34.4</td><td></td><td>43.7</td><td>42.7</td></tr><tr><td>global local+global</td><td>None</td><td>32</td><td>30.031.6 28.930.1</td><td></td><td>34.6</td><td>33.6</td></tr><tr><td></td><td>(4,3,3)</td><td>680</td><td>37.1 34.8 32.334.7</td><td></td><td>44.1</td><td>42.4</td></tr><tr><td colspan="7">Conditional Compression</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>local</td><td>(4,3,3)</td><td>648</td><td>38.8 36.1 33.1 36.0</td><td></td><td>44.0</td><td>43.2</td></tr><tr><td>global</td><td>None</td><td>32</td><td>30.8 30.3 29.8 30.3</td><td></td><td>35.5</td><td>34.0</td></tr><tr><td>local+global</td><td>(4,3,3)</td><td>680</td><td>39.0 36.7 34.2 36.6</td><td></td><td>45.0</td><td>43.5</td></tr></table>

![](images/634ac3445940d3761caed8a8017fede5c9b44ff98c8a394fd434dc16f4ee6f90.jpg)  
Figure 5. The ablation study on different compressing ratios. The figures show the performance on VideoMME-Short (upper left), MVBench (upper right), EgoSchema (lower left), and the inference time of 7B LLM (lower right).

Table 5. The ablation study on the conditional pre-training stage with the constructed HICom-248K dataset.
<table><tr><td rowspan="2">Methods</td><td rowspan="2">Conditional Pre-trained</td><td colspan="4">VideoMME w/o subs.</td><td rowspan="2">MV- Bench</td><td rowspan="2">Ego- Schema</td></tr><tr><td>short mid</td><td></td><td>long</td><td>all</td></tr><tr><td rowspan="2">avg pool</td><td>x</td><td>36.6</td><td>32.4</td><td>33.2</td><td>34.1</td><td>42.7</td><td>40.6</td></tr><tr><td>√</td><td>36.3</td><td>33.3</td><td>32.2</td><td>34.0</td><td>43.7</td><td>41.9</td></tr><tr><td rowspan="2">w/o inj.</td><td>x</td><td>39.4</td><td>33.7</td><td>33.1</td><td>35.4</td><td>42.8</td><td>41.64</td></tr><tr><td>√</td><td>37.1</td><td>34.8</td><td>32.3</td><td>34.7</td><td>44.1</td><td>42.4</td></tr><tr><td rowspan="2">w/inj.</td><td>x</td><td>38.8</td><td>35.1</td><td>34.1</td><td>36.0</td><td>44.0</td><td>41.6</td></tr><tr><td>√</td><td>39.0</td><td>36.7</td><td>34.2</td><td>36.6</td><td>45.0</td><td>43.5</td></tr></table>

Open-ended benchmarks. We also compare the performance of different models on two open-ended video benchmarks, as shown in Tab. 3. We evaluate all our results via GPT-3.5-Turbo-0125. Our HICom also achieves comparable results on both benchmarks, as arriving the SOTA on ActivityNet and getting the second best on the VideoChat-GPT Bench. Our HICom with 1328 tokens is on par with Tarsier with much more training data, and performs similarly to LLaVA-OneVision with much more training data and much more visual tokens, demonstrating the effectiveness of our compression.

Generalization Ability on Video Length. Due to the design of the local-level compression, we can easily extend the lengths of sampled frames at the inference stage for our trained model with 32 frames, rather than re-training it As we increase the number of sampled frames, the metrics of our HICom also improve across all these benchmarks. For example, the performance on short and medium videos of VideoMME grows fast from 32 frames to 128 frames at our high compressing ratio, as more useful frames are sampled. The performance on long videos of VideoMME without subtitles changes slightly, and we argue this may caused by the extremely long videos (30-60 minutes). Both 32 frames and 128 frames are too sparse for them. Our HICom can pay attention to the most relevant parts based on the sparsely sampled frames, thus changing slightly.

## 4.3. Ablation Studies

To save time costs, we choose VideoChat2-IT [23] 896K video data as our instruction tuning data and Qwen2.5- 0.5B [48] as our LLM to conduct ablation studies, and extract 32 frames for training unless otherwise specified. This requires less than 28 hours training on 8 A800 GPUs.

Component Analysis. The key components of our HICom are local- and global- level compression. Tab. 4 shows the results of them in the status of both unconditional (i.e., without condition injection) and conditional compression (i.e., with condition injection). Note we use $\hat { V } _ { p } ^ { t , h , w }$ as query of Eq. (1) in unconditional compression for direct injection. With the same number of input tokens, the unconditional local compression can achieve comparable performance with both spatial pooling (line 2) and temporalspatial pooling (line 3). Due to the lack of temporal-spatial inductive bias and too few tokens, the global compression significantly harms the performance, and local+global performs only slightly better than local as the additional information of the global branch is limited without explicit guidance. Compared with unconditional compression, the local and global conditional compression increase by 0.8% and 0.5%, respectively. The local+global achieves the best, increasing by 1.9% on VideoMME and 1.3% average on three benchmarks, also increasing by 0.63% compared with local conditional compression. As the global branch can provide instruction-relevant information from a global sight, combining both is better. It is noteworthy that our conditional compression with 680 tokens can beat the average pooling with 2592 tokens on all benchmarks, reducing 73.8% visual tokens and increasing by 0.47% on the performance.

Table 6. The ablation study on different types of injection.
<table><tr><td rowspan="2">Local</td><td rowspan="2">Global</td><td colspan="4">VideoMME w/o sub.</td><td rowspan="2">MV- Bench</td><td rowspan="2">Ego- Schema</td></tr><tr><td>short</td><td>mid</td><td>long</td><td>all</td></tr><tr><td>direct</td><td>coarse</td><td>39.0</td><td>36.7</td><td>34.2</td><td>36.6</td><td>45.0</td><td>43.5</td></tr><tr><td>direct</td><td>fine</td><td>39.3</td><td>35.3</td><td>34.3</td><td>36.3</td><td>44.1</td><td>43.5</td></tr><tr><td>coarse</td><td>coarse</td><td>39.8</td><td>34.8</td><td>35.0</td><td>36.5</td><td>44.1</td><td>42.7</td></tr><tr><td>coarse</td><td>fine</td><td>38.4</td><td>36.7</td><td>34.2</td><td>36.4</td><td>44.3</td><td>43.5</td></tr><tr><td>fine</td><td>coarse</td><td>39.1</td><td>34.7</td><td>34.2</td><td>36.0</td><td>44.2</td><td>42.7</td></tr><tr><td>fine</td><td>fine</td><td>38.8</td><td>34.9</td><td>34.2</td><td>36.0</td><td>43.6</td><td>42.0</td></tr></table>

Compressing Ratio. We explore the effectiveness of the compressing ratio on both unconditional and conditional compression in Fig. 5. Our unconditional compression performs similarly with average pooling on three benchmarks at different compressing ratios, and the conditional compression performs much better than them. It is noteworthy that the speed of performance decreasing of the conditional compression is also lower than the unconditional compression as the compressing ratio becomes higher. This shows instruction-relevant information remains with the guidance, demonstrating the superiority. In the lower right sub-figure of Fig. 5, the compressing ratio of (4,3,3) saves 57.68% inference time on Qwen2.5-7B than (1,3,3) with only 2.06% performance loss on Qwen2.5-0.5B. Higher compressing ratios (4,9,9) and (8,9,9) can only save a little more time with much higher performance loss. Therefore, we choose (4,3,3) as our final compressing ratio.

Conditional Pre-training Stage. To evaluate the effectiveness of our proposed conditional pre-training stage, we conduct experiments on whether to use this stage or not on both unconditional and conditional compression in Tab. 5. For average pooling and our unconditional compression, the conditional pre-training stage is also effective on MVBench and EgoSchema since more data is trained, but is hard to be effective on VideoMME. For the conditional compression, the conditional pre-training stage is effective on all benchmarks, increasing by 1.17% averagely, demonstrating the effectiveness of not only the constructed data, but also the newly proposed stage.

![](images/095c620f80b05ab839c49a2b3be8fecb9c24fa9de238763026857856a44c6174.jpg)  
Figure 6. The visualization of the attention map at both the local and global level in the situation of unconditional (w/o inj.) and conditional (w/ inj.) compression. We indicate the instruction-relevant parts with the red bounding box for easier reading.

Types of Injection. We propose three types of instruction condition injection in Sec. 3.1, direct injection is designed for local compression only, while coarse and fine injection fit both. We conduct the ablation study on them in Tab. 6. Coarse injection performs better than fine injection on all three benchmarks, and both of them perform better than unconditional compression. We argue the reason may be that the pooled instruction feature is more in line with the CLIP training, and the attention of fine injection is harder to converge than the MLP of coarse injection.

## 4.4. Qualitative Analysis

To qualitatively figure out how our HICom realizes the conditional compression, we visualize the attention map of the compression module in Fig. 6. It clearly shows where the model pays attention to based on the instructions. Since we conduct the local-level compression within the subregion of each group, the attention map of the local level is more evenly distributed throughout the video than the global level. Compared to unconditional compression, the attention map of conditional compression is more related to the instruction-relevant visual parts. As shown in Fig. 6, the attention maps at both levels successfully focus on the candle holders in the first example, and successfully on the Christmas tree, the apples, candles, and berries in the second example, which means the model will give more weights of these parts during compressing. We also notice that the global-level compression tends to forget some detailed visual information (e.g., the candle holders in the first and third frame of the first example, the Christmas tree in the second frame of the second example), while the locallevel compression is better to capture these details due to group-limited attention. This shows that hybrid-level compression provides different sights, thus helping the model better understand the videos.

## 5. Conclusion

In this paper, we address the challenge of achieving efficient video understanding in multi-modal large language models (MLLMs) through our proposed HICom. Our approach effectively balances the trade-off between computational costs and information retention by integrating user instructions into the compression process. This method not only preserves the temporal-spatial structure of video data but also emphasizes instruction-relevant content through a dual-level compression strategy. Furthermore, by introducing a conditional pre-training stage with proposed HICom-248K dataset, targeted pre-training enhances the efficacy of instruction-injected schema. Our experiments demonstrate the effectiveness, as HICom can maintain distinguished video understanding ability with much fewer tokens. HICom shows a good generalization ability to extend the number of frames during inference, but the fixed frame sampling strategy during training still limits the ability to understand long videos. In the future, we will try to sample frames based on fps and add more long videos for training to further increase the ability. We will also explore the potential of conditional compression on images, especially high-resolution images, making our HICom more powerful.

## Acknowledgment

This work is supported by the National Key Research and Development Program of China (2022YFB3104700), the National Nature Science Foundation of China (62425114, 62121002, U23B2028, 62232006). We acknowledge the support of Alibaba Group, the GPU cluster built by MCC Lab of Information Science and Technology Institution, USTC, and USTC super-computing center for providing computational resources for this project.

## References

[1] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. NeurIPS, 35: 23716–23736, 2022. 2

[2] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. arXiv preprint arXiv:2308.12966, 2023. 1

[3] Fabian Caba Heilbron, Victor Escorcia, Bernard Ghanem, and Juan Carlos Niebles. Activitynet: A large-scale video benchmark for human activity understanding. In CVPR, pages 961–970, 2015. 6

[4] Lin Chen, Xilin Wei, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Bin Lin, Zhenyu Tang, et al. Sharegpt4video: Improving video understanding and generation with better captions. arXiv preprint arXiv:2406.04325, 2024. 5

[5] Tsai-Shien Chen, Aliaksandr Siarohin, Willi Menapace, Ekaterina Deyneka, Hsiang-wei Chao, Byung Eun Jeon, Yuwei Fang, Hsin-Ying Lee, Jian Ren, Ming-Hsuan Yang, et al. Panda-70m: Captioning 70m videos with multiple cross-modality teachers. In CVPR, pages 13320–13331, 2024.5,3

[6] Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. arXiv preprint arXiv:2404.16821, 2024. 1

[7] Zesen Cheng, Sicong Leng, Hang Zhang, Yifei Xin, Xin Li, Guanzheng Chen, Yongxin Zhu, Wenqi Zhang, Ziyang Luo, Deli Zhao, et al. Videollama 2: Advancing spatialtemporal modeling and audio understanding in video-llms. arXiv preprint arXiv:2406.07476, 2024. 2, 4, 5, 6

[8] Wenliang Dai, Junnan Li, DONGXU LI, Anthony Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale N Fung, and Steven Hoi. Instructblip: Towards general-purpose vision-language models with instruction tuning. In NeurIPS, pages 49250–49267, 2023. 2

[9] Alexey Dosovitskiy. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020. 3

[10] Chaoyou Fu, Yuhan Dai, Yondong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen,

Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. arXiv preprint arXiv:2405.21075, 2024. 5, 6, 3

[11] Chaoyou Fu, Haojia Lin, Zuwei Long, Yunhang Shen, Meng Zhao, Yifan Zhang, Xiong Wang, Di Yin, Long Ma, Xiawu Zheng, et al. Vita: Towards open-source interactive omni multimodal 1lm. arXiv preprint arXiv:2408.05211, 2024. 5

[12] Zuan Gao, Yuxin Wang, Yadong Qu, Boqiang Zhang, Zixiao Wang, Jianjun Xu, and Hongtao Xie. Self-supervised pretraining with symmetric superimposition modeling for scene text recognition. arXiv preprint arXiv:2405.05841, 2024. 1

[13] Jiannan Ge, Hongtao Xie, Pandeng Li, Lingxi Xie, Shaobo Min, and Yongdong Zhang. Towards discriminative feature generation for generalized zero-shot learning. IEEE Transactions on Multimedia, 2024. 1

[14] Jiannan Ge, Lingxi Xie, Hongtao Xie, Pandeng Li, Xiaopeng Zhang, Yongdong Zhang, and Qi Tian. Alignzeg: Mitigating objective misalignment for zero-shot semantic segmentation. In ECCV, pages 142–161. Springer, 2024. 1

[15] Kristen Grauman, Andrew Westbury, Eugene Byrne, Zachary Chavis, Antonino Furnari, Rohit Girdhar, Jackson Hamburger, Hao Jiang, Miao Liu, Xingyu Liu, et al. Ego4d: Around the world in 3,000 hours of egocentric video. In CVPR, pages 18995–19012, 2022. 5, 3

[16] Bo He, Hengduo Li, Young Kyun Jang, Menglin Jia, Xuefei Cao, Ashish Shah, Abhinav Shrivastava, and Ser-Nam Lim. Ma-lmm: Memory-augmented large multimodal model for long-term video understanding. In CVPR, pages 13504– 13514, 2024.2

[17] Peng Jin, Ryuichi Takanobu, Wancai Zhang, Xiaochun Cao, and Li Yuan. Chat-univi: Unified visual representation empowers large language models with image and video understanding. In CVPR, pages 13700–13710, 2024. 2, 5, 6

[18] Jie Lei, Licheng Yu, Mohit Bansal, and Tamara L Berg. Tvqa: Localized, compositional video question answering arXiv preprint arXiv:1809.01696, 2018. 6

[19] Bo Li, Yuanhan Zhang, Liangyu Chen, Jinghao Wang, Jingkang Yang, and Ziwei Liu. Otter: A multi-modal model with in-context instruction tuning. arXiv preprint arXiv:2305.03726, 2023. 2

[20] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024. 1, 2, 5, 6

[21] Feng Li, Renrui Zhang, Hao Zhang, Yuanhan Zhang, Bo Li, Wei Li, Zejun Ma, and Chunyuan Li. Llava-next-interleave: Tackling multi-image, video, and 3d in large multimodal models. arXiv preprint arXiv:2407.07895, 2024. 6

[22] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In ICML, pages 19730–19742. PMLR, 2023. 1, 2

[23] KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. Videochat: Chat-centric video understanding. arXiv preprint arXiv:2305.06355, 2023. 1, 2, 6, 7

[24] Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, et al.

Mvbench: A comprehensive multi-modal video understanding benchmark. In CVPR, pages 22195–22206, 2024. 5, 6

[25] Pandeng Li, Chen-Wei Xie, Hongtao Xie, Liming Zhao, Lei Zhang, Yun Zheng, Deli Zhao, and Yongdong Zhang. Momentdiff: Generative video moment retrieval from random to real. NeurIPS, 36:65948–65966, 2023. 1

[26] Pandeng Li, Chen-Wei Xie, Liming Zhao, Hongtao Xie, Jiannan Ge, Yun Zheng, Deli Zhao, and Yongdong Zhang. Progressive spatio-temporal prototype matching for textvideo retrieval. In ICCV, pages 4100–4110, 2023. 1

[27] Wentong Li, Yuqian Yuan, Jian Liu, Dongqi Tang, Song Wang, Jianke Zhu, and Lei Zhang. Tokenpacker: Efficient visual projector for multimodal 1lm. arXiv preprint arXiv:2407.02392, 2024. 3

[28] Yanwei Li, Chengyao Wang, and Jiaya Jia. Llama-vid: An image is worth 2 tokens in large language models. In ECCV, pages 323–340. Springer, 2024. 3, 4, 5, 6

[29] Jianxin Liang, Xiaojun Meng, Yueqian Wang, Chang Liu, Qun Liu, and Dongyan Zhao. End-to-end video question answering with frame scoring mechanisms and adaptive sampling. arXiv preprint arXiv:2407.15047, 2024. 3

[30] Bin Lin, Yang Ye, Bin Zhu, Jiaxi Cui, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection. arXiv preprint arXiv:2311.10122, 2023. 1, 2, 5, 6

[31] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. NeurIPS, 36, 2023. 1, 2, 4

[32] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In CVPR, pages 26296–26306, 2024. 1, 4, 6

[33] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, 2024. 2, 1

[34] Ruyang Liu, Chen Li, Yixiao Ge, Thomas H Li, Ying Shan, and Ge Li. Bt-adapter: Video conversation is feasible without video instruction tuning. In CVPR, pages 13658–13667, 2024. 1,2

[35] Ruyang Liu, Chen Li, Haoran Tang, Yixiao Ge, Ying Shan, and Ge Li. St-llm: Large language models are effective temporal learners. In European Conference on Computer Vision, pages 1–18. Springer, 2024. 6

[36] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan. Video-chatgpt: Towards detailed video understanding via large vision and language models. In ACL, 2024.1,6

[37] Karttikeya Mangalam, Raiymbek Akshulakov, and Jitendra Malik. Egoschema: A diagnostic benchmark for very longform video language understanding. NeurIPS, 36:46212– 46244, 2023.6

[38] Mathew Monfort, Alex Andonian, Bolei Zhou, Kandan Ramakrishnan, Sarah Adel Bargal, Tom Yan, Lisa Brown, Quanfu Fan, Dan Gutfreund, Carl Vondrick, et al. Moments in time dataset: one million videos for event understanding. IEEE TPAMI, 42(2):502–508, 2019. 6

[39] William Peebles and Saining Xie. Scalable diffusion models with transformers. In ICCV, pages 4195–4205, 2023. 4

[40] Tianyuan Qu, Longxiang Tang, Bohao Peng, Senqiao Yang, Bei Yu, and Jiaya Jia. Does your vision-language model get lost in the long video sampling dilemma?, 2025. 1

[41] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, pages 8748–8763. PMLR, 2021. 3

[42] Machel Reid, Nikolay Savinov, Denis Teplyashin, Dmitry Lepikhin, Timothy Lillicrap, Jean-baptiste Alayrac, Radu Soricut, Angeliki Lazaridou, Orhan Firat, Julian Schrittwieser, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024. 1

[43] Shuhuai Ren, Linli Yao, Shicheng Li, Xu Sun, and Lu Hou. Timechat: A time-sensitive multimodal large language model for long video understanding. In CVPR, pages 14313– 14323, 2024.2

[44] Amir Shahroudy, Jun Liu, Tian-Tsong Ng, and Gang Wang. Ntu rgb+d: A large scale dataset for 3d human activity analysis. In CVPR, pages 1010–1019, 2016. 6

[45] Gunnar A. Sigurdsson, Gül Varol, Xiaolong Wang, Ivan Laptev, Ali Farhadi, and Abhinav Gupta. Hollywood in homes: Crowdsourcing data collection for activity understanding. ArXiv e-prints, 2016. 6

[46] Enxin Song, Wenhao Chai, Guanhong Wang, Yucheng Zhang, Haoyang Zhou, Feiyang Wu, Haozhe Chi, Xun Guo, Tian Ye, Yanting Zhang, et al. Moviechat: From dense token to sparse memory for long video understanding. In CVPR, pages 18221–18232, 2024. 2

[47] Longxiang Tang, Zhuotao Tian, Kai Li, Chunming He, Hantao Zhou, Hengshuang Zhao, Xiu Li, and Jiaya Jia. Mind the interference: Retaining pre-trained knowledge in parameter efficient continual learning of vision-language models. In ECCV, pages 346–365. Springer, 2024. 1

[48] Qwen Team. Qwen2.5: A party of foundation models, 2024. 5,7, 1

[49] Shengbang Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Manoj Middepogu, Sai Charitha Akula, Jihan Yang, Shusheng Yang, Adithya Iyer, Xichen Pan, et al. Cambrian-1: A fully open, vision-centric exploration of multimodal 1lms. arXiv preprint arXiv:2406.16860, 2024. 3

[50] Haibo Wang, Chenghang Lai, Yixuan Sun, and Weifeng Ge. Weakly supervised gaussian contrastive grounding with large multimodal models for video question answering. arXiv preprint arXiv:2401.10711, 2024. 3

[51] Jiawei Wang, Liping Yuan, Yuchen Zhang, and Haomiao Sun. Tarsier: Recipes for training and evaluating large video description models. arXiv preprint arXiv:2407.00634, 2024. 5,6

[52] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model's perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024. 1, 5, 3

[53] Yizhou Wang, Ruiyi Zhang, Haoliang Wang, Uttaran Bhattacharya, Yun Fu, and Gang Wu. Vaquita: Enhancing align-

ment in llm-assisted video understanding. arXiv preprint arXiv:2312.02310, 2023. 3, 4

[54] Yi Wang, Kunchang Li, Xinhao Li, Jiashuo Yu, Yinan He, Guo Chen, Baoqi Pei, Rongkun Zheng, Jilan Xu, Zun Wang, et al. Internvideo2: Scaling video foundation models for multimodal video understanding. arXiv preprint arXiv:2403.15377, 2024. 5, 3

[55] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. NeurIPS, 35:24824–24837, 2022. 5

[56] Yujie Wei, Shiwei Zhang, Zhiwu Qing, Hangjie Yuan, Zhiheng Liu, Yu Liu, Yingya Zhang, Jingren Zhou, and Hongming Shan. Dreamvideo: Composing your dream videos with customized subject and motion. In CVPR, pages 6537– 6549, 2024.1

[57] Yujie Wei, Shiwei Zhang, Hangjie Yuan, Xiang Wang, Haonan Qiu, Rui Zhao, Yutong Feng, Feng Liu, Zhizhong Huang, Jiaxin Ye, et al. Dreamvideo-2: Zero-shot subjectdriven video customization with precise motion control. arXiv preprint arXiv:2410.13830, 2024. 1

[58] Yuetian Weng, Mingfei Han, Haoyu He, Xiaojun Chang, and Bohan Zhuang. Longvlm: Efficient long video understanding via large language models. In ECCV, pages 453–470. Springer, 2024. 6

[59] Lin Xu, Yilin Zhao, Daquan Zhou, Zhijie Lin, See Kiong Ng, and Jiashi Feng. Pllava: Parameter-free llava extension from images to videos for video dense captioning. arXiv preprint arXiv:2404.16994, 2024. 2, 5, 6

[60] Mingze Xu, Mingfei Gao, Zhe Gan, Hong-You Chen, Zhengfeng Lai, Haiming Gang, Kai Kang, and Afshin Dehghan. Slowfast-llava: A strong training-free baseline for video large language models. arXiv preprint arXiv:2407.15841, 2024. 6

[61] Zhipei Xu, Xuanyu Zhang, Runyi Li, Zecheng Tang, Qing Huang, and Jian Zhang. Fakeshield: Explainable image forgery detection and localization via multi-modal large language models. arXiv preprint arXiv:2410.02761, 2024. 1

[62] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024. 1

[63] Shoubin Yu, Jaemin Cho, Prateek Yadav, and Mohit Bansal. Self-chained image-language model for video localization and question answering. NeurIPS, 36, 2023. 3

[64] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In ICCV, pages 11975–11986, 2023. 5, 1

[65] Boqiang Zhang, Hongtao Xie, Zuan Gao, and Yuxin Wang. Choose what you need: Disentangled representation learning for scene text recognition removal and editing. In CVPR, pages 28358–28368, 2024. 1

[66] Boqiang Zhang, Kehan Li, Zesen Cheng, Zhiqiang Hu, Yuqian Yuan, Guanzheng Chen, Sicong Leng, Yuming Jiang, Hang Zhang, Xin Li, et al. Videollama 3: Frontier multimodal foundation models for image and video understanding. arXiv preprint arXiv:2501.13106, 2025. 1

[67] Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding. arXiv preprint arXiv:2306.02858, 2023. 2

[68] Haoji Zhang, Yiqin Wang, Yansong Tang, Yong Liu, Jiashi Feng, Jifeng Dai, and Xiaojie Jin. Flash-vstream: Memorybased real-time understanding for long video streams. arXiv preprint arXiv:2406.08085, 2024. 2, 4

[69] Peiyuan Zhang, Kaichen Zhang, Bo Li, Guangtao Zeng, Jingkang Yang, Yuanhan Zhang, Ziyue Wang, Haoran Tan, Chunyuan Li, and Ziwei Liu. Long context transfer from language to vision. arXiv preprint arXiv:2406.16852, 2024. 5,6

[70] Xuanyu Zhang, Runyi Li, Jiwen Yu, Youmin Xu, Weiqi Li, and Jian Zhang. Editguard: Versatile image watermarking for tamper localization and copyright protection. In CVPR, pages 11964–11974, 2024. 1

[71] Yuanhan Zhang, Bo Li, haotian Liu, Yong jae Lee, Liangke Gui, Di Fu, Jiashi Feng, Ziwei Liu, and Chunyuan Li. Llavanext: A strong zero-shot video understanding model, 2024. 2,5,6

[72] Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. Video instruction tuning with synthetic data. arXiv preprint arXiv:2410.02713, 2024. 2, 5, 6,3

[73] Junjie Zhou, Yan Shu, Bo Zhao, Boya Wu, Shitao Xiao, Xi Yang, Yongping Xiong, Bo Zhang, Tiejun Huang, and Zheng Liu. Mlvu: A comprehensive benchmark for multi-task long video understanding. arXiv preprint arXiv:2406.04264, 2024.6

[74] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023. 1