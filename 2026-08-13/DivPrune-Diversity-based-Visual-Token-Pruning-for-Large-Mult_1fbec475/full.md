# DivPrune: Diversity-based Visual Token Pruning for Large Multimodal Models

Saeed Ranjbar Alvar, Gursimran Singh<sup>†</sup>, Mohammad Akbari<sup>†</sup>, Yong Zhang

Huawei Technologies Canada Co., Ltd.

{saeed.ranjbar.alvar1, gursimran.singh1, mohammad.akbari, yong.zhang3}@huawei.com

## Abstract

Large Multimodal Models (LMMs) have emerged as powerful models capable of understanding various data modalities, including text, images, and videos. LMMs encode both text and visual data into tokens that are then combined and processed by an integrated Large Language Model (LLM). Including visual tokens substantially increases the total token count, often by thousands. The increased input length for LLM significantly raises the complexity ofinference, resulting in high latency in LMMs. To address this issue, token pruning methods, which remove part of the visual tokens, are proposed. The existing token pruning methods either require extensive calibration and fine-tuning or rely on suboptimal importance metrics which results in increased redundancy among the retained tokens. In this paper, we first formulate token pruning as Max-Min Diversity Problem (MMDP) where the goal is to select a subset such that the diversity among the selected tokens is maximized. Then, we solve the MMDP to obtain the selected subset and prune the rest. The proposed method, DivPrune, reduces redundancy and achieves the highest diversity of the selected tokens. By ensuring high diversity, the selected tokens better represent the original tokens, enabling effective performance even at high pruning ratios without requiring fine-tuning. Extensive experiments with various LMMs show that DivPrune achieves state-of-the-art accuracy over 16 image- and video-language datasets. Additionally, DivPrune reduces both the end-to-end latency and GPU memory usagefor the tested models. The code is available here<sup>⋄</sup>.

## 1. Introduction

Following the success of Large Language Models (LLMs) in language understanding [1, 6, 40], Large Multimodal Models (LMMs) [19, 22, 23, 52] have emerged to handle diverse data types like images and video, by leveraging the foundational capabilities of LLMs. Typically, LMMs encode text and visual modalities into tokens, also known as embeddings. These tokens are then combined and processed by an integrated LLM. The inclusion of visual tokens significantly increases the total number of tokens, often adding thousands to the combined set. Since the running time and memory requirements scale quadratically with input size [7, 8, 15, 38], the addition of visual tokens can substantially raise the running time for LMMs. Hence, many of these models often struggle to meet the demands of lowlatency applications, particularly in resource-constrained environments [46].

![](images/ba882c43fba1ab05722de8b5926821174d09bcf0f71607835fd041f0cfa15662.jpg)  
Figure 1. Comparison of different visual token pruning methods across various pruning ratios for LLaVA 1.5-7B. The y-axis is the performance averaged on COCO (CIDEr), OKVQA (Acc), POPE (F1), and MMBench (Acc). The x-axis is the TFLOP ratio of the model after token pruning compared to the original model before pruning. The proposed method significantly outperforms all baselines. Note that, unlike other methods, FitPrune uses an additional calibration step to prune tokens.

Previous research [4, 36, 47] has demonstrated that there is a high degree of redundancy in the visual information processed by LMMs. As a result, visual token pruning has emerged as a promising solution to address the computational complexity challenges faced by LMMs. Specifically, previous research has demonstrated that reducing the number of visual tokens by 50% [4] to 95% [36] can significantly enhance the inference speed of LMMs.

While promising, token pruning methods have certain shortcomings. For example, the works in [3, 17, 21, 47] require calibration or finetuning for each model which is costly and time-consuming to implement. FastV [4] and PruMerge [36] use attention scores to identify less important tokens for pruning. However, it is shown that using attention scores is not optimal, as some important tokens are overlooked [21]. Additionally, attention-based pruning tends to retain tokens that are similar to each other, leading to redundancy. At high compression ratio, this redundancy prevents the selection of a sufficient number of unique tokens to accurately represent the original tokens. In line with this observation, our findings indicate that pruning a large portion of visual tokens using these methods, without subsequent fine-tuning, results in a significant drop in the performance of LMMs across various tasks (Fig. 1).

To address the above-mentioned issues, we formulate token pruning as a Max-Min Diversity Problem (MMDP) [35]. In an MMDP, the objective is to select a subset of elements such that the diversity among them is maximized. We apply this concept to token pruning, which we call DivPrune, aiming to maximize the diversity of the selected tokens by increasing the minimum distance between them. By ensuring high diversity, DivPrune captures a broader range of visual tokens, making it inherently more robust compared to attention-based methods that focus only on token importance scores. Increasing the diversity also helps ensure that the selected tokens better represent the original set of tokens, enabling effective performance even at high pruning ratios without the need for fine-tuning.

DivPrune also offers practical advantages that make it a highly useful solution in real-world scenarios. DivPrune is a plug-and-play solution that can be used without requiring offline optimization with a calibration set, or fine-tuning of the model, which are often time-consuming and computationally expensive. DivPrune is applicable to LMMs with any LLM architecture and vision encoder. Additionally, DivPrune is compatible with inference optimization techniques, such as KV caching, resulting in practical speedup in real-world applications. In summary, our major contributions are as follows:

• We introduce DivPrune, a token pruning method based on MMDP that maximizes diversity among visual tokens, effectively reducing redundancy and ensuring a highly representative subset.

• DivPrune is a training-free, calibration-data-free, plugand-play solution that can be seamlessly integrated with off-the-shelf LMMs.

• We conduct evaluations using 16 datasets on imageand video-language models with image and video understanding tasks. DivPrune achieves state-of-the-art performance, with noticeable gains under extreme pruning (i.e., ratio ≥ 80%).

• DivPrune reduces GPU memory usage and inference latency while maintaining comparable accuracy compared to the original model across most datasets.

## 2. Related Works

## 2.1. Large Multimodal Models (LMMs)

LMMs handle diverse data types, including text, audio, image, and, video [5, 19, 22, 23, 30, 39, 52]. This work focuses on open-source LMMs that support language and visual inputs. These LMMs can be categorized into two types: image-based and video-based LMMs. The imagebased LMMs [22, 23] address image-language understanding tasks, like image captioning, visual question answering, and image reasoning. On the other hand, video-based LMMs are geared towards video understanding [19, 52] tasks, like video captioning, video summarization, and video question answering.

## 2.2. Efficient LMMs

Several techniques are proposed to improve inference efficiency specifically for LMMs. The first technique is to change the model architecture in LMMs. For example, [33] proposed to replace transformer-based LLMs with Mamba model [12]. [49, 53] retrained LMMs with small scale LLMs to improve their efficiency. [45] used knowledge distillation to train a small LMM. In addition to changing the architecture, it is shown in [37] that skipping some blocks or layers within LMMs can improve the inference speed without damaging the model’s performance. Furthermore, efficient decoding techniques such as speculative decoding are proposed to make LMM inference more efficient [11].

## 2.3. Visual Token Pruning

Visual token pruning methods are proposed to reduce the inference complexity for LMMs. The first group of methods uses attention scores to prune tokens [4, 36]. PruMerge [36] introduces a token pruning method for the vision encoder where the visual tokens are clustered and merged based on their attention sparsity. In addition, FastV [4] prunes tokens within a specific layer of the LLM based on the magnitude of attention scores in an earlier layer. It is shown that pruning tokens based on attention scores are not optimal [13, 21], especially at higher pruning ratios.

Calibration-based methods offer another line of work, where pruning layers and/or ratios are determined by analyzing the LLM outputs for a calibration dataset [21, 47]. For example, FitPrune [47] calculates a pruning recipe based on the observed attention divergence before and after pruning. VTW [21] argues that visual tokens can be entirely removed after a certain layer within LLM. The layer to remove the visual tokens is chosen using a calibration dataset. These methods rely on calibration datasets and require custom calibration for each LMM, which can be costly and cumbersome for new models.

![](images/b5781acd5ff1a3d43a89fc686b0ad742c64e0b21890bb3f70537d612e988dd90.jpg)  
Figure 2. An overview of the LMM architecture, with DivPrune applied to visual tokens. The blocks on the right-hand side illustrate the steps of the method.

Some previous works proposed token pruning with the need for fine-tuning. $\bar { { \bf M } ^ { 3 } } \bar { { \bf \Lambda } } [ 3 ]$ applies model fine-tuning to produce nested visual token representations at multiple granularities, allowing users to select token lengths dynamically during inference. In [17], a projector layer trained using a large-scale dataset is proposed that packs finer detailed information into compact token representations. These methods need significant computational resources for training, limiting their use across various scenarios.

## 3. Proposed Method

In this section, we briefly discuss how LMMs work. Then, the token pruning problem is defined, followed by a detailed presentation of the proposed method.

## 3.1. Large Multimodal Models (LMMs)

An LMM typically processes a pair of inputs, denoted as $( T , V )$ , where $T$ is the text input and $V$ is the visual input such as image or video. The text input is mapped to N textual tokens $\mathbf { E _ { t } } = \{ t _ { 1 } , \ldots , t _ { N } \}$ using a text encoder. Similarly, the visual input is processed by a corresponding vision encoder. Specifically, it takes visual information $V$ as input and outputs image features, that are further converted to M (generally $M \gg N )$ vision tokens $\mathbf { E _ { v } } = \{ v _ { 1 } , \dots , v _ { M } \}$ using a projector layer $( \mathrm { F i g . ~ } 2 )$

The textual tokens and visual tokens are then combined to be fed to an LLM to generate the prediction in an autoregressive manner. Specifically, $\hat { N }$ output tokens ${ \textbf { Y } } =$ $\{ y _ { 1 } , \dotsc , y _ { \hat { N } } \}$ are generated as follows:

$$
P ( y _ { 1 } , . . . , y _ { \hat { N } } \mid \mathbf { E _ { t } } , \mathbf { E _ { v } } ) = \prod _ { i = 1 } ^ { \hat { N } } P ( y _ { i } \mid y _ { < i } , \mathbf { E _ { t } } , \mathbf { E _ { v } } ) ,\tag{1}
$$

where $P ( \ u )$ is the conditional probability obtained at the

output of the LLM.

## 3.2. Token Pruning

Reducing the number of input tokens in an integrated LLM within LMMs helps to lower memory usage and inference latency. Since visual tokens tend to have more redundancy, they are generally selected for pruning.

In this context, the problem of token pruning can be defined as follows: given a set of visual tokens $\mathbf { E _ { v } }$ with $| { \bf E } _ { \bf v } | = M$ and the subset size $\tilde { M } ( \tilde { M } < M )$ , the goal is to select a subset, $\tilde { \mathbf { E } } _ { \mathbf { v } } .$ , while preserving key information necessary for accurate predictions. To mathematically formulate the token pruning problem, we define a mapping function $f ,$ which maps the original set of visual tokens, $\mathbf { E _ { v } } ,$ to a subset, $\tilde { \mathbf { E } } _ { \mathbf { v } } \overset {  } { = } \{ \tilde { v } _ { 1 } , \dots , \tilde { v } _ { \tilde { M } } \}$ , where $| \tilde { \mathbf { E } } _ { \mathbf { v } } | = \tilde { M }$ . The objective is to identify a mapping function $f$ that minimizes the difference in the model’s output before and after pruning while ensuring the reduced set still captures the essential information from the original set:

$$
\begin{array} { r l r } & { } & { \mathrm { F i n d : } \quad f : { \bf E } _ { v }  \tilde { { \bf E } } _ { \bf v } } \\ & { } & { \mathrm { O b j e c t i v e : } \quad \underset { f } { \operatorname* { m i n } } \mathcal { L } \big ( \mathcal { P } , \tilde { \mathcal { P } } \big ) } \\ & { } & { \mathrm { S u b j e c t : } \quad \mathrm { | \tilde { \bf E } _ { v } | = \tilde { M } , } } \end{array}\tag{2}
$$

where $\begin{array} { r l r } { \mathcal { P } } & { = } & { P ( y _ { 1 } , . . . , y _ { \hat { N } } \quad | \quad \mathbf { E _ { t } } , \mathbf { E _ { v } } ) } \end{array}$ and $\begin{array} { r l } { \tilde { \mathcal P } } & { { } = } \end{array}$ $P ( y _ { 1 } , . . . , y _ { \hat { N } } \mid \mathbf { E _ { t } } , f ( \mathbf { E _ { v } } ) )$ . Here, L represents a loss function that measures the difference in the model’s output with and without pruning, and $\tilde { M }$ indicates the number of retained tokens. Next, we propose a novel diversity-based solution for the introduced token pruning problem.

## 3.3. DivPrune: Method Overview

We proposed a diversity-based token pruning method by reformulating the problem in (2) to select a subset of M<sup>˜</sup> elements that maximizes the diversity, thereby reducing redundancy. Specifically, we define token pruning as Max–Min Diversity Problem (MMDP) [32] where the goal is to find the set $\tilde { \mathbf { E } } _ { \mathbf { v } }$ among all possible sets with M<sup>˜</sup> samples in $\mathbf { E _ { v } }$ that has the maximum minimum distance between its elements. So, MMDP is defined as:

$$
\mathrm { F i n d ~ } \tilde { \mathbf { E } } _ { \mathbf { v } } = \arg \operatorname* { m a x } \left[ \operatorname* { m i n } _ { \gamma , \omega \in S } \left( d ( \gamma , \omega ) \right) : \forall S \subset \mathbf { E } _ { \mathbf { v } } \right] ,\tag{3}
$$

where $S$ is an arbitrary set in $\mathbf { E _ { v } }$ with $\tilde { M }$ elements and $( \gamma , \omega )$ are arbitrary elements in $S .$ The distance is measured by $d ( . , . )$ which is defined using the cosine distance as follows:

$$
d ( \gamma , \omega ) = 1 - \frac { \gamma \cdot \omega } { \| \gamma \| \| \omega \| } .\tag{4}
$$

A solution for the MMDP problem in (3) is a subset of $\mathbf { E _ { v } }$ that maximizes diversity by minimizing redundancy between elements. In the literature, several solutions including exact and heuristic methods are proposed to solve the

Algorithm 1: Proposed Token Pruning Method   
1 M<sup>˜</sup> : subset size; $\mathbf { E _ { v } } \colon$ visual tokens; $\tilde { \mathbf { E } } _ { \mathbf { v } } \mathrm { : }$ selected subset   
2 Initialize $\tilde { \bf E } _ { \bf v } = [ ]$ and $\mathbf { R } = \mathbf { E } _ { \mathbf { v } }$   
3 // First stage: add the first token   
4 $D = [ ]$ initialize the distance array   
5 for i in R do   
6 $d _ { m i n } = + i n f$   
7 for $j$ in R do   
8 $\mathbf { I f } \left( i \neq j \ \& \ d ( i , j ) \leq d _ { m i n } \right)$ then   
$d _ { m i n } = d ( i , j )$   
9 Add $d _ { m i n }$ to $D$   
10 $k = \mathbf { R } [ \mathrm { a r g }$ max (D)]   
11 move k from R to $\tilde { \mathbf { E } } _ { \mathbf { v } }$   
12 // Second stage: iteratively add the subsequent tokens   
13 while $| \tilde { \mathbf { E } } _ { \mathbf { v } } | < \tilde { M }$ do   
14 $D = [ ]$ initialize the distance array   
15 for i in R do   
16 $d _ { m i n } = + i n f$   
17 for $j$ in $\tilde { \mathbf { E } } _ { \mathbf { v } }$ do   
18 If $\cdot d ( i , j ) \leq d _ { m i n }$ then $d _ { m i n } = d ( i , j )$   
19 Add $d _ { m i n }$ to D   
20 $k = \mathbf { R }$ [arg max (D)]   
21 move k from R to $\tilde { \mathbf { E } } _ { \mathbf { v } }$   
22 Return $\tilde { \mathbf { E } } _ { \mathbf { v } }$

MMDP problem [29, 35]. Since the number of tokens is generally limited (e.g., 576 in LLaVA 1.5 [22]) and the solvers are not generally designed for GPU acceleration, we obtain exact solution for the problem. Notably, the overhead of the selection process using GPU is negligible compared to the computations within the LLM. Detailed steps of the proposed method is summarized in Algorithm 1. Once the selected tokens are identified, the remaining visual tokens are discarded. The selected tokens along with the textual tokens are passed to the LLM.

As shown in Algorithm 1, the proposed method has two stages after the initialization. The selected subset, $\tilde { \mathbf { E } } _ { \mathbf { v } }$ , is initialized as empty, and the candidate list R is initialized with all the visual tokens. In the first stage, the first token of the selected subset is chosen based on the pairwise distance between the tokens of the candidate list. Then, the chosen token is moved from the candidate list to the selected list. In the second stage, similar to the first stage, the pairwise distance of the tokens in $\tilde { \mathbf { E } } _ { \mathbf { v } }$ and the tokens in R is used to add samples to $\tilde { \mathbf { E } } _ { \mathbf { v } }$ iteratively. Finally, once the number of tokens in $\tilde { \mathbf { E } } _ { \mathbf { v } }$ reaches the specified subset size, the selection procedure is terminated and the $\tilde { \mathbf { E } } _ { \mathbf { v } }$ is returned. To avoid repeated distance calculations over iterations a distance matrix is initially calculated by one matrix multiplication.

The proposed method can also be applied to the features (i.e., hidden states) in the intermediate layers of the LLM. In this case, our method is not applied to the visual tokens, but to the features corresponding to the visual tokens obtained from a decoder layer to select a subset before feeding them to the subsequent layers. In either case, our method obtains the highest diversity for the selected elements. Ablation studies are provided in the next section to analyze the effect of pruning different elements at different layers.

## 4. Experiments

In this section, we present a comprehensive analysis comparing the performance of our method and previous works across various settings, tasks, and datasets. Insights into the proposed method are also provided through illustrative examples. Moreover, the efficiency of DivPrune along with ablation study are provided.

## 4.1. Experimental Settings

Baselines and Models: We consider five baselines, namely, FastV [4], PruMerge [36], VTW [21], FitPrune [47] and ${ \bf { M } } ^ { 3 }$ [3]. Among these, we consider FastV, PruMerge, and VTW as our main competitors as they are plug-and-play and do not rely on any further costly finetuning or calibration process. However, for the sake of completeness, we also report performance comparison with respect to one finetuning-based $( \mathbf { M } ^ { 3 } )$ and one calibration-based (FitPrune) methods. Note that, VTW, by default, requires calibration to determine the best layer for a given task. However, doing that does not allow us to set a specific TFLOP ratio, complicating the comparison. Hence, whenever required we disable the calibration of VTW to select the layer that matches the FLOP requirement of a particular experiment.

We test DivPrune and the baselines with popular LMMs namely LLaVA 1.5-7B [22]<sup>1</sup>, LLaVA 1.5-13B [22]<sup>2</sup> LLaVA $1 . 6 \mathrm { - } 7 \mathbf { B } ^ { 3 }$ (also known as LLaVA-NeXT [23]), and LLaVA-NeXT-Video-7B $[ 5 2 ] ^ { 4 }$ to demonstrate the generality of DivPrune. For each tested model and task, we report only the relevant subset of baseline that is applicable to that specific model and task, alongside our results.

All the tested LMMs used CLIP vision encoder [34]. LLaVA 1.5 model uses 576 visual tokens to represent images. LLaVA 1.6 converts each image into a varying number of patches, resulting in 3-5 times more visual tokens compared to LLaVA 1.5. LLaVA-NeXT-Video uses 144 tokens to process each frame. For all the experiments with LLaVA-NeXT-Video we used a total of 8 frames resulting in 1152 tokens for the processed frames.

Datasets, Tasks, and Metrics: We selected a comprehensive set of common tasks and datasets aimed at multimodal reasoning and understanding. Specifically, we chose 11 image-language and 5 video-language datasets.

These datasets encompass a wide range of tasks, including captioning, multiple-choice Question Answering (QA), and open-ended QA based on text and image/video inputs. Consistent with prior works, CIDEr score [42] is used for evaluating captioning tasks, and Exact Match (EM), Accuracy (Acc), F1, Perception Score (P-score) [9] and GPTassisted [10] score are used for QA tasks. Furthermore, Wu-Palmer similarity (WUPS) score [43] and GPT-assisted score [10] is used for open-ended QA. For all task performance metrics used in this paper, higher values indicate better performance. For the reported time and memory, lower values indicate better results. Further details regarding the datasets, tasks, and metrics are provided in the supplementary material.

Following the earlier works in [4, 21, 47], we report the computational requirement, measured in TFLOPs, for DivPrune and the baselines. Various configurations including different pruning ratios at different layers are examined to obtain different working TFLOPs for our method and the baselines. The reported TFLOP ratio is the TFLOP of the model with pruned tokens relative to the original model’s TFLOP with no pruning. This ratio is estimated as [4]:

$$
\frac { K \times ( 4 \mu d ^ { 2 } - 2 \mu ^ { 2 } d + 2 \mu d m ) + ( T - K ) \times ( 4 \tilde { \mu } d ^ { 2 } - 2 \tilde { \mu } ^ { 2 } d + 2 \tilde { \mu } d m ) } { T \times ( 4 \mu d ^ { 2 } - 2 \mu ^ { 2 } d + 2 \mu d m ) } ,\tag{5}
$$

where T is the total transformer-based decoder layers. $\mu = N + M$ is the total sequence length before pruning, $\tilde { \mu } ~ = ~ N + \tilde { M }$ is the sequence length after pruning, d is the hidden state size of the layer, and m is the intermediate size of feed-forward network module. Depending on the TFLOP ratio requirement set by a particular experiment, we adjust the pruning hyperparameters of all baselines to match that requirement. However, some baselines do not support fine-grained adjustments like our approach does. In these cases, we choose the smallest available TFLOP ratio that exceeds the requirement set by an experiment, which might give these baselines a slight advantage over our method.

We used 8 × V100 GPUs with 32GB VRAM for all the experiments in this paper. Additionally, we used the lmmsevals package [51] for running these benchmarks for all the baselines and models. All results are obtained with a batch size of 1. For the metrics that require ChatGPT API access, the model is set to “gpt-4o-mini”.

## 4.2. Insights

We provide visualizations comparing DivPrune with importance-based token pruning methods using LLaVA 1.5- 7B and the SeedBench dataset [16]. Detailed analysis across different models and datasets is provided in the following subsections.

The visual tokens in LLaVa 1.5 model are 4096- dimensional vectors. The t-SNE method [41] is utilized to project the visual tokens in $\mathbf { E _ { v } }$ from a high dimensional to a 2D space. The corresponding visualization for a sample input data is shown in Fig. 3-(a) using light Pruple points. Then, DivPrune is applied to select 10% of the visual tokens (i.e., pruning 90%). Additionally, FastV, as an importance-based token pruning method, which utilizes attention scores, is employed to prune with the same ratio. The selected subsets using DivPrune and FastV are shown with different markers in Fig. 3-(a). More examples are provided in the supplementary materials.

As the example in Fig. 3-(a) shows, the proposed method selects points from all the clusters that appeared in the projected space whereas FastV does not choose any samples from the upper cluster. So, our method achieves a better representation of the original points by including samples from all clusters. In addition, the FastV method selects many tokens that are very close to each other which increases redundancy among the selected set. On the other hand, our method reduces redundancy by pruning the closely similar tokens.

![](images/b2c3e83b625413d3d5709e9af899b614794976c1f3832f199c030a947f84b0eb.jpg)  
(a)

![](images/a1c09d475b368ac6de65ffe2824821260d7516224c77d677111058dc8519ab37.jpg)  
(b)  
Figure 3. (a) t-SNE visualization of visual tokens for the original model, our method, and FastV. (b) Histogram of the Max-Min distance between the selected tokens over the SeedBench dataset.

<table><tr><td></td><td>Method</td><td>TFLOP (ratio %)</td><td>COCO CIDEr</td><td>Flickr CIDEr</td><td>GQA EM</td><td>MMB Acc</td><td>MME P-score</td><td>MMMU Acc</td><td>Nocaps CIDEr</td><td>OKVQA EM</td><td>POPE F1</td><td>SQA EM</td><td>SEEDB Acc</td></tr><tr><td rowspan="9">I .7B</td><td>Original</td><td>3.228 (100.00)</td><td>1.10</td><td>0.75</td><td>61.96</td><td>64.09</td><td>1506</td><td>36.44</td><td>1.06</td><td>53.39</td><td>85.84</td><td>69.41</td><td>66.17</td></tr><tr><td>VTW [21]</td><td>0.603 (18.46)</td><td>0.05</td><td>0.03</td><td>38.94</td><td>21.31</td><td>681</td><td>32.60</td><td>0.03</td><td>18.64</td><td>25.35</td><td>65.29</td><td>36.13</td></tr><tr><td>FastV [4]</td><td>0.514 (15.69)</td><td>0.06</td><td>0.03</td><td>38.73</td><td>20.62</td><td>696</td><td>32.00</td><td>0.04</td><td>18.32</td><td>32.84</td><td>65.15</td><td>35.69</td></tr><tr><td>Ours</td><td>0.512 (15.63)</td><td>0.96</td><td>0.62</td><td>56.85</td><td>59.19</td><td>1328</td><td>35.89</td><td>0.92</td><td>46.98</td><td>86.02</td><td>68.27</td><td>59.47</td></tr><tr><td>PruMerge [36]</td><td>Variable</td><td>0.77</td><td>0.50</td><td>51.30</td><td>54.47</td><td>1259</td><td>35.11</td><td>0.73</td><td>41.74</td><td>66.89</td><td>68.91</td><td>53.26</td></tr><tr><td>Ours*</td><td>Variable</td><td>0.91</td><td>0.56</td><td>55.25</td><td>58.16</td><td>1330</td><td>35.44</td><td>0.87</td><td>44.38</td><td>83.06</td><td>67.87</td><td>57.88</td></tr><tr><td>FitPrune[47]</td><td>0.513 (15.65)</td><td>0.90</td><td>0.56</td><td>52.39</td><td>57.65</td><td>1197</td><td>36.00</td><td>0.86</td><td>42.53</td><td>60.89</td><td>68.02</td><td>54.84</td></tr><tr><td> $\mathbf { M } ^ { 3 \bullet } \left[ 3 \right]$ </td><td>0.512 (15.63)</td><td>1.00</td><td>0.67</td><td>60.81</td><td>65.81</td><td>1391</td><td>31.80</td><td>0.95</td><td>55.12</td><td>86.33</td><td>64.65</td><td>64.93</td></tr><tr><td>PruMerge-LoRA•</td><td>Variable</td><td>0.96</td><td>0.63</td><td>55.96</td><td>59.88</td><td>1334</td><td>34.89</td><td>0.90</td><td>47.99</td><td>77.13</td><td>68.32</td><td>57.93</td></tr><tr><td></td><td>Original</td><td>6.281 (100.00)</td><td>1.16</td><td>0.80</td><td>63.33</td><td>68.64</td><td>1522</td><td>35.67</td><td>1.09</td><td>58.28</td><td>85.99</td><td>72.88</td><td>66.82</td></tr><tr><td></td><td>VTW [21]</td><td>1.030 (16.16)</td><td>0.08</td><td>0.05</td><td>39.71</td><td>21.91</td><td>622</td><td>32.10</td><td>0.05</td><td>22.49</td><td>0.40</td><td>66.24</td><td>38.59</td></tr><tr><td>I-3B</td><td>FastV [4]</td><td>1.003 (15.73)</td><td>0.38</td><td>0.18</td><td>44.98</td><td>37.80</td><td>942</td><td>35.11</td><td>0.33</td><td>32.14</td><td>30.02</td><td>69.96</td><td>44.95</td></tr><tr><td></td><td>Ours</td><td>1.002 (15.71)</td><td>1.00</td><td>0.66</td><td>57.29</td><td>63.40</td><td>1407</td><td>34.89</td><td>0.95</td><td>53.29</td><td>83.43</td><td>72.34</td><td>62.04</td></tr><tr><td></td><td> $\overline { { \mathrm { P r u M e r g e } ^ { \triangle } \ [ 3 6 ] } }$ </td><td>Variable</td><td>0.80</td><td>0.53</td><td>52.01</td><td>58.93</td><td>1256</td><td>36.56</td><td>0.77</td><td>49.15</td><td>64.36</td><td>72.53</td><td>56.10</td></tr><tr><td></td><td> $\mathbf { O u r s } ^ { * }$ </td><td>Variable</td><td>0.94</td><td>0.59</td><td>56.09</td><td>61.77</td><td>1344</td><td>34.89</td><td>0.91</td><td>50.86</td><td>79.60</td><td>71.34</td><td>60.00</td></tr><tr><td></td><td>Original</td><td>11.849 (100.00)</td><td>1.00</td><td>0.68</td><td>64.28</td><td>67.01</td><td>1520</td><td>36.44</td><td>0.88</td><td>44.20</td><td>86.38</td><td>70.15</td><td>70.16</td></tr><tr><td></td><td>VTW [21]</td><td>1.318 (11.23)</td><td>0.06</td><td>0.03</td><td>38.62</td><td>19.76</td><td>606</td><td>31.30</td><td>0.03</td><td>8.66</td><td>7.13</td><td>65.74</td><td>37.48</td></tr><tr><td>I-7B</td><td>FastV [4]</td><td>1.327 (11.30)</td><td>0.06</td><td>0.03</td><td>38.79</td><td>20.36</td><td>619</td><td>32.56</td><td>0.04</td><td>8.80</td><td>7.78</td><td>65.49</td><td>37.62</td></tr><tr><td></td><td>Ours</td><td>1.266 (10.79)</td><td>0.89</td><td>0.61</td><td>58.69</td><td>63.49</td><td>1362</td><td>37.11</td><td>0.76</td><td>41.92</td><td>82.97</td><td>68.57</td><td>64.11</td></tr><tr><td></td><td> $\overline { { { \bf M } } } ^ { 3 \bullet } [ 3 ]$ </td><td>1.266 (10.79)</td><td>1.01</td><td>0.67</td><td>62.97</td><td>69.16</td><td>1490</td><td>35.00</td><td>0.85</td><td>57.49</td><td>87.44</td><td>69.51</td><td>68.49</td></tr></table>

Table 1. Comparison results of our method and different baselines on image-language understanding datasets. •: Finetuning is used, △: Calibration dataset is used. Ours<sup>∗</sup>: Our method matching the PruMerge selection ratio.

In addition, the max-min distance (Eq. 3) for the selected subset of tokens is computed using 1000 randomly data samples from the SeedBench dataset and the histogram of the computed values is shown in Fig. 3-(b). As the plot indicates, the proposed method selects a subset where samples have a higher minimum pair-wise distance compared to the FastV method. Hence, our method achieves higher diversity among the selected tokens that have less redundancy compared to the ones chosen using FastV. We analyze the effect of the reduced diversity on task performance in the following sections.

## 4.3. Image-Language Understanding

In this section, we compare DivPrune against baselines across various image-language understanding tasks, including open- and closed-ended QA, visual reasoning, and image captioning. Specifically, ScienceQA-IMG (SQA) [25], POPE [18], MME [9], MMB [24], GQA [14], MMMU [50], Flicker30k [31], SeedBench (SEEDB) [16], Nocaps [2], OKVQA [28], and COCO-2017 [20] are used.

In the first experiment, summarized in Tab. 1, we analyze an extreme compression scenario for three imagebased LMMs by fixing the TFLOP ratio at approximately 15%, wherever the baseline allows configuration to a fixed TFLOP ratio. Since PruMerge does not allow fixing the TFLOP ratio, we configure our approach (Ours\*) to match the variable pruning corresponding to PruMerge for a fair comparison. In the top section of the table, we compare the results of various baselines for LLaVa 1.5-7B. Specifically, the baselines supporting LLaVA 1.5 are grouped into three categories: plug-and-play methods, those with a variable TFLOP ratio, and those requiring a calibration dataset or involving fine-tuning the LMMs. Among the plug-andplay methods, which are the focus of this work, our approach significantly outperforms both the VTW and FastV baselines across all datasets. This result holds despite using lower TFLOPs, clearly demonstrating the advantage of our method in this scenario. For instance, when DivPrune is used, the performance of LLaVA 1.5-7b decreases by 5.1% on the GQA dataset and 4.9% on the MMB dataset. In contrast, the VTW and FastV methods result in performance drops of at least 23.0% and 42.8% on these datasets, respectively. The performance gap between DivPrune and the baseline methods is even more pronounced in image captioning tasks. For example, the CIDEr score on the COCO dataset drops by approximately 95% with VTW and FastV, but only by 12.7% with DivPrune. Additionally, DivPrune, compared to the original model, shows less than a 2% performance drop on the MMMU and SQA datasets and slightly enhances the original model’s performance on the POPE dataset while reducing the TLOP ratio by 84.4%. It is shown that removing redundant tokens in some datasets can improve the original model’s performance [4].

Next, in the variable scenario, the pruning ratio is determined dynamically. To ensure a fair comparison, we matched the pruning ratio with that of the PruMerge baseline, assuming the average sequence length for calculating the average TFLOPs across each dataset. As indicated by the results, our approach consistently outperforms PruMerge across all benchmarks, except one. Further, for the baseline with calibration, we observe that our approach outperforms the FitPrune approach on nearly all datasets by up to 25.1%, despite not using any calibration dataset. Finally, compared to baselines involving fine-tuning, our method achieves comparable or superior performance without requiring any fine-tuning.

<table><tr><td></td><td>TFLOPs (ratio %)</td><td>ActivityNet Score/Acc</td><td>SeedBench Acc</td><td>VChatGPT Score</td><td>NextQA WUPS</td><td>EgoSch. Acc</td><td>Max GPU mem (GB)</td><td>Prefill Time (sec)</td><td>E2E Latency (sec)</td></tr><tr><td>Original</td><td>6.539 (100)</td><td>2.67 / 48.10</td><td>38.7</td><td>2.16</td><td>26.05</td><td>41.8</td><td>14.06</td><td>0.330</td><td>4.37</td></tr><tr><td>VTW [21]</td><td>1.124 (16.97)</td><td>1.61 / 26.84</td><td>29.39</td><td>1.19</td><td>18.66</td><td>25.42</td><td>13.63</td><td>0.150</td><td>3.43</td></tr><tr><td>FastV [4]</td><td>0.943 (14.20)</td><td>1.95 / 33.91</td><td>32.98</td><td>1.44</td><td>22.51</td><td>29.14</td><td>13.57</td><td>0.150</td><td>3.63</td></tr><tr><td>Ours</td><td>0.937 (14.10)</td><td>2.56 / 45.90</td><td>37.00</td><td>1.92</td><td>24.48</td><td>39.76</td><td>13.51</td><td>0.161</td><td>3.39</td></tr></table>

Table 2. Comparison results of our method and baselines on LLaVA-NeXT-Video-7B across video-language understanding datasets.

The above experiment is repeated with LLaVa 1.5-13B model and the results are shown in the middle part of Tab. 1. The baselines that support this model are FastV, VTW, and PruMerge. As shown in the table, DivPrune outperforms the corresponding baselines in both plug-and-play and variable scenarios almost on all the tested datasets. For example, on the POPE dataset, DivPrune outperforms VTW, FastV, and PruMerge with F1 score improvements of 83%, 53.4%, and 15.2%, respectively. Additionally, on the MMB dataset, DivPrune achieves higher accuracy rates of 41.5%, 25.6%, and 2.8% compared to VTW, FastV, and PruMerge, respectively. This demonstrates that DivPrune generalizes effectively across models with varying numbers of parameters.

In the bottom part of Tab. 1, the results corresponding to LLaVA 1.6-7B model are shown. We used the same pruning ratio as for LLava 1.5. However, the lower TFLOP ratio is due to the large number of visual tokens in LLaVA 1.6. The results indicate that the performance of the model drops significantly when baseline pruning methods are applied. For example, the F1 score on the POPE dataset drops by 79% with the baselines as compared to the original model, whereas the drop with DivPrune is only 3.4%. DivPrune also maintains competitive performance compared to the original model across various datasets. Specifically, DivPrune shows only 3.5%, 2.3%,3.4%,1.6% drop in accuracy compared to the original model on the MMB, OKVQA, POPE, and SQA datasets, respectively, while reducing the TFLOP by 89%. The results also demonstrate that pruning visual tokens with DivPrune enhances the original model’s performance on the MMMU task. These results show that DivPrune generalizes across different models. Qualitative examples as well as results with additional datasets are provided in the supplementary materials.

Furthermore, we show the comparison of different baselines and our method across various TFLOP ratios. We plot the results in Fig. 1 where the y-axis represents average performance on four datasets, namely, COCO (CIDEr), OKVQA (Acc), POPE (F1), and MMBench (Acc). The range of the performance metric for all datasets is between

0 and 1, except for the CIDEr metric, which has a maximum reported value of 1.10. On the x-axis, we only show the high compression scenario (TFLOP ratio ≤ 45%). As shown in the figure, our method significantly outperforms all the baselines, particularly in high compression scenarios $( \mathrm { T F L O P } \leq 2 5 \% )$ . Further, we notice a steep drop in performance of all baselines as the TFLOP ratio → 10, while our method falls more gracefully. This results in an increasing performance gap between our approach and the baselines at extreme compression levels. For higher TFLOP ratios almost all converge toward the original performance, with Fit-Prune slightly outperforming our approach by an insignificant margin. It is important to note that, unlike our method, FitPrune relies on a calibration dataset to prune tokens.

## 4.4. Video-Language Understanding

In this section, LLaVA-NeXT-Video-7B [23], a video-based LMM is used to analyze the performance of the proposed method on various video-language understanding tasks. Specifically, we evaluate DivPrune using five datasets, namely, ActivityNet [48], SeedBench [16], VideoChatGPT (temporal) [26], NextQA [44], and EgoSchema [27]. FastV and VTW methods are chosen as the baselines. We tested DivPrune using the same pruning ratio as in the image understanding experiments. However, due to the higher number of visual tokens in the LLaVA-NeXT-Video model, this pruning ratio results in lower TFLOPs ratio. For the baselines, we match their TFLOPs with ours by selecting the smallest available TFLOP ratio that exceeds the TFLOPs of our method. The results for the original model, DivPrune, and the baselines are given in Tab. 2. As shown in the table, DivPrune outperforms both FastV and VTW by a significant margin. Specifically, DivPrune achieves upto 12% higher accuracy than FastV and upto 19% better than VTW on Video QA datasets including ActivityNet, SeedBench, and EgoSchema. DivPrune also outperforms both baselines on open-ended QA such as VideoChatGPT and NextQA by achieving higher GPT-assisted and WUPS scores.

Furthermore, our method achieves performance that is highly competitive compared to the original model without pruning despite using only 14.1% of the original model’s TFLOPs. This demonstrates the robustness of DivPrune, as it effectively generalizes to video LMMs. Notably, the performance gap between DivPrune and the original model without pruning narrows as the number of visual tokens increases, indicating that DivPrune is more effective for the models with larger visual contexts.

<table><tr><td></td><td>TFLOP (ratio %)</td><td>MMB Acc</td><td>MMMU Acc</td><td>POPE F1</td><td>SQA EM</td><td>Avg</td></tr><tr><td>Layer 0 (Ours)</td><td>19.61</td><td>59.19</td><td>35.89</td><td>86.02</td><td>68.27</td><td>62.34</td></tr><tr><td>Layer 1</td><td>19.65</td><td>59.02</td><td>34.89</td><td>80.67</td><td>67.18</td><td>60.44</td></tr><tr><td>Layer 2</td><td>19.70</td><td>54.90</td><td>34.22</td><td>69.27</td><td>69.56</td><td>56.99</td></tr><tr><td>Layer 3</td><td>19.80</td><td>23.97</td><td>32.67</td><td>31.82</td><td>65.94</td><td>38.60</td></tr></table>

Table 3. Ablation study on applying DivPrune at different layers.

## 4.5. Efficiency Analysis

In this section, we analyze the efficiency of the proposed method using memory usage (i.e., max allocated memory), prefill time, and end-to-end latency (E2E). For this experiment, VideoChatGPT dataset with 499 samples is used to obtain the average time and memory usage for LLaVA-NeXT-Video-7B model. The results are summarized on the right side of Tab. 2. The obtained results are compared against the original model, as well as the FastV and VTW baselines. As shown in the table, our approach requires approximately 400MB less memory than the original model, with memory usage comparable to the baselines. In terms of prefill and E2E time, our approach is about 55% and 22% faster, respectively, compared to the original model. When compared to the baselines, our prefill time is approximately 6-7% longer, while the E2E time is 1-7% shorter. The slight increase in prefill time for our method compared to the baselines is due to the distance calculations (See Section 3.3), which are performed only once during the prefill stage. In contrast, for baselines, the corresponding calculations for token pruning need to be done at each decoding step, resulting in longer E2E time.

## 4.6. Ablation Study

In this section, we conduct an ablation study to analyze the impact of modifying various core components of our method. The ablation experiments are conducted with the LLaVA 1.5-7B model. First, we show the effect of pruning tokens inside the LLM in Tab. 3 using 5 datasets. By default, in our method, visual tokens are pruned before being passed to the first decoder layer in the LLM, which we refer to as ’Layer 0’. We also tested ’Layer 1’ where the first layer is processed without pruning and the pruning is performed afterward. We further extended this approach by allowing tokens to pass through the first few layers unpruned and then pruning them after specific layers. As shown in the table, for a fixed TFLOP ratio of 19.61%, pruning done by our method at layer 0 achieves higher task accuracies compared to pruning at layers 1, 2, and 3 of the LLM.

Furthermore, in Tab. 4, we provide an analysis of using alternative diversity measures for token pruning. The first three rows show the impact of choosing different distance measures to quantify the similarity among tokens. It can be seen that all three similarity measures, cosine, $\ell _ { 1 }$ , and $\ell _ { 2 }$ perform comparably, with cosine (default setting) performing slightly better. This suggests that the choice of similarity measure does not significantly impact DivPrune’s overall performance.

<table><tr><td></td><td>TFLOP (ratio %)</td><td>MMB Acc</td><td>MMMU Acc</td><td>POPE F1</td><td>SQA EM</td><td>Avg</td></tr><tr><td>Cosine (Ours)</td><td>19.61</td><td>59.19</td><td>35.89</td><td>86.02</td><td>68.27</td><td>62.34</td></tr><tr><td> $\ell _ { 1 }$ </td><td>19.61</td><td>59.71</td><td>34.67</td><td>85.40</td><td>67.97</td><td>61.94</td></tr><tr><td> $\ell _ { 2 }$ </td><td>19.61</td><td>59.97</td><td>35.00</td><td>85.64</td><td>68.27</td><td>62.22</td></tr><tr><td>Random</td><td>19.61</td><td>52.66</td><td>34.56</td><td>72.78</td><td>66.63</td><td>56.66</td></tr><tr><td>Min-Max</td><td>19.61</td><td>38.57</td><td>33.11</td><td>49.26</td><td>65.20</td><td>46.53</td></tr></table>

Table 4. Ablation on using various diversity measures.

The last two rows in Tab. 4 show the effect of choosing alternative strategies of token selection other than the proposed Max-Min diversity-based solution (3). We tested random pruning as well as the Min-Max strategy where the maximum distance between the selected samples is minimized. The Min-Max strategy enforces high redundancy among the selected samples, resulting in reduced diversity. As results in the bottom part of Tab. 4 reveal that any deviation from our proposed selection strategy results in suboptimal performance. Specifically, the Min-Max strategy performs the worst, showing approximately 15.8% lower performance compared to ours. This decline is due to the Min-Max approach selecting tokens that are highly similar to each other, resulting in less diversity among the selected visual tokens. Random selection provides some degree of diversity, but it performs 5.6% worse than the proposed method because it cannot guarantee maximum diversity. This proves that redundancy of visual tokens leads to poor performance and diversity maximization is needed for optimal performance, corroborating the utility and need of the proposed diversity maximization in Eq. (3).

## 5. Conclusion

In this paper, we proposed a token pruning method based on a max-min diversity problem, called DivPrune. In the proposed method, maximum diversity is achieved among the selected tokens, resulting in reduced redundancy. By ensuring high diversity, the selected tokens provide a more representative subset of the original tokens, enabling effective performance even at high pruning ratios without requiring fine-tuning. Extensive experiments were conducted with multiple LMMs on image and video understanding tasks across 16 datasets. The results show that DivPrune achieves state-of-the-art accuracy on the tested datasets. DivPrune generalizes well to different model sizes and architectures, while also improving memory consumption and end-to-end latency for the tested LMMs.

## References

[1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023. 1

[2] Harsh Agrawal, Karan Desai, Yufei Wang, Xinlei Chen, Rishabh Jain, Mark Johnson, Dhruv Batra, Devi Parikh, Stefan Lee, and Peter Anderson. nocaps: novel object captioning at scale. In Proceedings of the IEEE International Conference on Computer Vision, pages 8948–8957, 2019. 6

[3] Mu Cai, Jianwei Yang, Jianfeng Gao, and Yong Jae Lee. Matryoshka multimodal models. Proceedings of the International Conference on Learning Representation, 2025. 2, 3, 4, 6

[4] Liang Chen, Haozhe Zhao, Tianyu Liu, Shuai Bai, Junyang Lin, Chang Zhou, and Baobao Chang. An image is worth 1/2 tokens after layer 2: Plug-and-play inference acceleration for large vision-language models. In European Conference on Computer Vision (ECCV), 2024. 1, 2, 4, 5, 6, 7

[5] Zesen Cheng, Sicong Leng, Hang Zhang, Yifei Xin, Xin Li, Guanzheng Chen, Yongxin Zhu, Wenqi Zhang, Ziyang Luo, Deli Zhao, et al. Videollama 2: Advancing spatialtemporal modeling and audio understanding in video-llms. arXiv preprint arXiv:2406.07476, 2024. 2

[6] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E Gonzalez, et al. Vicuna: An open-source chatbot impressing gpt-4 with 90%\* chatgpt quality. See https://vicuna. lmsys. org (accessed 14 April 2023), 2(3):6, 2023. 1

[7] Krzysztof Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamas Sarlos, Peter Hawkins, Jared Davis, Afroz Mohiuddin, Lukasz Kaiser, et al. Rethinking attention with performers. arXiv preprint arXiv:2009.14794, 2020. 1

[8] Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Re. Flashattention: Fast and memory-efficient exact at-´ tention with io-awareness. Advances in Neural Information Processing Systems, 35:16344–16359, 2022. 1

[9] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, et al. Mme: A comprehensive evaluation benchmark for multimodal large language models. arXiv preprint arXiv:2306.13394, 2023. 5, 6

[10] Jinlan Fu, See-Kiong Ng, Zhengbao Jiang, and Pengfei Liu. Gptscore: Evaluate as you desire, 2023. 5

[11] Mukul Gagrani, Raghavv Goel, Wonseok Jeon, Junyoung Park, Mingu Lee, and Christopher Lott. On speculative decoding for multimodal large language models. arXiv preprint arXiv:2404.08856, 2024. 2

[12] Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752, 2023. 2

[13] Zhiyu Guo, Hidetaka Kamigaito, and Taro Watanabe. Attention score is not all you need for token importance indicator

in kv cache reduction: Value also matters. arXiv preprint arXiv:2406.12335, 2024. 2

[14] Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. Conference on Computer Vision and Pattern Recognition (CVPR), 2019. 6

[15] Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and Franc¸ois Fleuret. Transformers are rnns: Fast autoregressive transformers with linear attention. In International confer ence on machine learning, pages 5156–5165. PMLR, 2020. 1

[16] Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yix iao Ge, and Ying Shan. Seed-bench: Benchmarking multimodal llms with generative comprehension. arXiv preprint arXiv:2307.16125, 2023. 5, 6, 7

[17] Wentong Li, Yuqian Yuan, Jian Liu, Dongqi Tang, Song Wang, Jianke Zhu, and Lei Zhang. Tokenpacker: Efficient visual projector for multimodal llm. arXiv preprint arXiv:2407.02392, 2024. 2, 3

[18] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. arXiv preprint arXiv:2305.10355, 2023. 6

[19] Bin Lin, Yang Ye, Bin Zhu, Jiaxi Cui, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection. CoRR, abs/2311.10122, 2023. 1, 2

[20] Tsung-Yi Lin, Michael Maire, Serge Belongie, Lubomir Bourdev, Ross Girshick, James Hays, Pietro Perona, Deva Ramanan, C. Lawrence Zitnick, and Piotr Dollar. Microsoft´ coco: Common objects in context, 2015. 6

[21] Zhihang Lin, Mingbao Lin, Luxi Lin, and Rongrong Ji. Boosting multimodal large language models with visual tokens withdrawal for rapid inference. arXiv preprint arXiv:2405.05803, 2024. 2, 4, 5, 6, 7

[22] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In NeurIPS, 2023. 1, 2, 4

[23] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Im proved reasoning, ocr, and world knowledge, 2024. 1, 2, 4, 7

[24] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? In European Conference on Computer Vision, pages 216–233. Springer, 2025. 6

[25] Pan Lu, Swaroop Mishra, Tony Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering, 2022. 6

[26] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan. Video-chatgpt: Towards detailed video understanding via large vision and language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL 2024), 2024. 7

[27] Karttikeya Mangalam, Raiymbek Akshulakov, and Jitendra Malik. Egoschema: A diagnostic benchmark for very long form video language understanding, 2023. 7

[28] Kenneth Marino, Mohammad Rastegari, Ali Farhadi, and Roozbeh Mottaghi. Ok-vqa: A visual question answering benchmark requiring external knowledge. In Conference on Computer Vision and Pattern Recognition (CVPR), 2019. 6

[29] Rafael Mart´ı, Anna Mart´ınez-Gavara, Sergio Perez-Pel´ o, and´ Jesus S´ anchez-Oro. A review on discrete diversity and dis-´ persion maximization from an or perspective. European Journal ofOperational Research, 299(3):795–813, 2022. 4

[30] OpenAI. Hello gpt-4o, 2024. https://openai.com/ index/hello-gpt-4o/ [Accessed: (Nov 2024)]. 2

[31] Bryan A. Plummer, Liwei Wang, Christopher M. Cervantes, Juan C. Caicedo, Julia Hockenmaier, and Svetlana Lazebnik. Flickr30k entities: Collecting region-to-phrase correspondences for richer image-to-sentence models. IJCV, 123 (1):74–93, 2017. 6

[32] Daniel Cosmin Porumbel, Jin-Kao Hao, and Fred Glover. A simple and effective algorithm for the maxmin diversity problem. Annals of Operations Research, 186:275–293, 2011. 3

[33] Yanyuan Qiao, Zheng Yu, Longteng Guo, Sihan Chen, Zijia Zhao, Mingzhen Sun, Qi Wu, and Jing Liu. Vl-mamba: Exploring state space models for multimodal learning. arXiv preprint arXiv:2403.13600, 2024. 2

[34] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 4

[35] Mauricio GC Resende, Rafael Mart´ı, Micael Gallego, and Abraham Duarte. Grasp and path relinking for the max–min diversity problem. Computers & Operations Research, 37 (3):498–508, 2010. 2, 4

[36] Yuzhang Shang, Mu Cai, Bingxin Xu, Yong Jae Lee, and Yan Yan. Llava-prumerge: Adaptive token reduction for efficient large multimodal models. arXiv preprint arXiv:2403.15388, 2024. 1, 2, 4, 6

[37] Mustafa Shukor and Matthieu Cord. Skipping computations in multimodal llms. arXiv preprint arXiv:2410.09454, 2024. 2

[38] Sainbayar Sukhbaatar, Edouard Grave, Piotr Bojanowski, and Armand Joulin. Adaptive attention span in transformers. arXiv preprint arXiv:1905.07799, 2019. 1

[39] Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024. 2

[40] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothee Lacroix, Baptiste´ Roziere, Naman Goyal, Eric Hambro, Faisal Azhar, et al.\` Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. 1

[41] Laurens Van der Maaten and Geoffrey Hinton. Visualizing data using t-sne. Journal of machine learning research, 9 (11), 2008. 5

[42] Ramakrishna Vedantam, C. Lawrence Zitnick, and Devi Parikh. Cider: Consensus-based image description evaluation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015. 5

[43] Zhibiao Wu and Martha Palmer. Verb semantics and lexical selection. arXiv preprint cmp-lg/9406033, 1994. 5

[44] Junbin Xiao, Xindi Shang, Angela Yao, and Tat-Seng Chua. Next-qa: Next phase of question-answering to explaining temporal actions. In Proceedings of the IEEE/CVF Confer ence on Computer Vision and Pattern Recognition (CVPR), pages 9777–9786, 2021. 7

[45] Shilin Xu, Xiangtai Li, Haobo Yuan, Lu Qi, Yunhai Tong, and Ming-Hsuan Yang. Llavadi: What matters for multimodal large language models distillation. arXiv preprint arXiv:2407.19409, 2024. 2

[46] Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, et al. Minicpm-v: A gpt-4v level mllm on your phone. arXiv preprint arXiv:2408.01800, 2024. 1

[47] Weihao Ye, Qiong Wu, Wenhao Lin, and Yiyi Zhou. Fit and prune: Fast and training-free visual token pruning for multi-modal large language models. arXiv preprint arXiv:2409.10197, 2024. 1, 2, 4, 5, 6

[48] Zhou Yu, Dejing Xu, Jun Yu, Ting Yu, Zhou Zhao, Yueting Zhuang, and Dacheng Tao. Activitynet-qa: A dataset for understanding complex web videos via question answering. In AAAI, pages 9127–9134, 2019. 7

[49] Zhengqing Yuan, Zhaoxu Li, Weiran Huang, Yanfang Ye, and Lichao Sun. Tinygpt-v: Efficient multimodal large language model via small backbones. arXiv preprint arXiv:2312.16862, 2023. 2

[50] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of CVPR, 2024. 6

[51] Kaichen Zhang, Bo Li, Peiyuan Zhang, Fanyi Pu, Joshua Adrian Cahyono, Kairui Hu, Shuai Liu, Yuanhan Zhang, Jingkang Yang, Chunyuan Li, and Ziwei Liu. Lmmseval: Reality check on the evaluation of large multimodal models, 2024. 5

[52] Yuanhan Zhang, Bo Li, haotian Liu, Yong jae Lee, Liangke Gui, Di Fu, Jiashi Feng, Ziwei Liu, and Chunyuan Li. Llavanext: A strong zero-shot video understanding model, 2024. 1, 2, 4

[53] Baichuan Zhou, Ying Hu, Xi Weng, Junlong Jia, Jie Luo, Xien Liu, Ji Wu, and Lei Huang. Tinyllava: A framework of small-scale large multimodal models. arXiv preprint arXiv:2402.14289, 2024. 2