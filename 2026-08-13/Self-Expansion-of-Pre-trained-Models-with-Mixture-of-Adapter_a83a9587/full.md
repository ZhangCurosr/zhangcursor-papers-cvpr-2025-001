# Self-Expansion of Pre-trained Models with Mixture of Adapters for Continual Learning

Huiyi Wang<sup>1,2</sup>, Haodong Lu<sup>1</sup>, Lina Yao<sup>2,1</sup>, Dong Gong<sup>1</sup>\* <sup>1</sup>University of New South Wales, <sup>2</sup>CSIRO’s Data61

{huiyi.wang, haodong.lu, dong.gong}@unsw.edu.au; lina.yao@data61.csiro.au

## Abstract

Continual learning (CL) aims to continually accumulate knowledgefrom a non-stationary data stream without catastrophicforgetting oflearned knowledge, requiring a balance between stability and adaptability. Relying on the generalizable representation in pre-trained models (PTMs), PTMbased CL methods perform effective continual adaptation on downstream tasks by adding learnable adapters or prompts upon the frozen PTMs. However, many existing PTM-based CL methods use restricted adaptation on afixed set ofthese modules to avoidforgetting, sufferingfrom limited CL ability. Periodically adding task-specific modules results in linear model growth rate and impaired knowledge reuse. We propose Self-Expansion of pre-trained models with Modularized Adaptation (SEMA), a novel approach to enhance the control ofstability-plasticity balance in PTM-based CL. SEMA automatically decides to reuse or add adapter modules on demand in CL, depending on whether significant distribution shift that cannot be handled is detected at different representation levels. We design modular adapter consisting ofafunctional adapter and a representation descriptor. The representation descriptors are trained as a distribution shift indicator and used to trigger self-expansion signals. For better composing the adapters, an expandable weighting router is learnedjointlyfor mixture ofadapter outputs. SEMA enables better knowledge reuse and sub-linear expansion rate. Extensive experiments demonstrate the effectiveness of the proposed self-expansion method, achieving state-of-the-art performance compared to PTM-based CL methods without memory rehearsal. Code is available at https://github.com/huiyiwang01/SEMA-CL.

## 1. Introduction

With the development of deep neural networks, deep learning models have achieved significant success in various fields, such as computer vision [15, 24]. However, real-world scenarios often present learning tasks in a dynamic data stream with non-stationary distributions [50]. Considering the need for efficient model updating and restricted budgets on storage and computation [35], it is not guaranteed to store all the historical data and repeatedly re-train the model. Continual learning (CL) is investigated to learn incrementally and accumulate knowledge efficiently from the non-stationary data stream without catastrophicforgetting [46, 54] of previously learned knowledge [14, 59, 65, 71]. It requires CL approaches to achieve a balance between knowledge expansion (i.e., plasticity) and knowledge retention (i.e., stability) [22, 55, 71]. Many CL approaches have been studied to tackle the challenge relying on different strategies, such as experience replay (ER) [7, 8, 77], regularization on parameters or representations [6, 39, 77], and architectures with modularization or isolation [55, 66, 70, 75, 78].

Given the progress in the pre-trained models (PTMs) with reliable representation, recent works explore the potential of using PTMs, such as Vision Transformer (ViT) [15], as the starting point of CL, unlike the “training-fromscratch” paradigm. PTM-based CL approaches [73, 74] usually keep the PTMs frozen to enable stable representation and alleviate forgetting. The PTMs are continually adapted to downstream tasks through parameter-efficient fine-tuning with newly expanded parameters as prompts and/or adapters [13, 51, 68, 73, 74, 83, 90, 91]. On the other hand, some methods enable continual fine-tuning of PTMs on real-world downstream tasks arriving in a streaming manner. Many PTM-based CL approaches mainly add and learn a fixed set/pool of prompts [33, 93] or adapters [9] shared by all downstream tasks in the stream [51, 73, 74, 90]. To alleviate forgetting caused by the interference on the newly added parameters, they restrict the parameter updating only on the first task seen in stream [51, 90] or use various regularization on the shared parameters [73, 74]. Their continual adaptation potentials are limited by the fixed and static size of prompt and adapter parameters. Some recent methods expand the PTMs with task-specific parameters to produce input-conditioned prompts [68] or ensemble of adapters [92].

![](images/50e5ae903ab5d10dc94573f3feb379db0c7b81cfa75e2783e14777d3cb603667.jpg)  
Figure 1. An example of the self-expansion process. (a) The PTM (i.e., ViT) with L transformer layers at the initial point of CL. (b) The first session adaptation – at Task 1, a modular adapter and a (dummy) router is added and trained in each transformer layer. (c) The modula adapters and routers added in the previous step (Task 1) are frozen to alleviate forgetting. When Task 2 arrives, only the representation descriptor in the L-th layer detects feature distribution shift (with novel patterns) and generates expansion signal. A new module is added and trained in the L-th layer, with the router expanded and updated. (d) At Task 3, new adapter is added at L 1-th layer after the expansion signal is firstly generated. In this demo example, the expansion is triggered and produced again in the L-th layer, following the expansion in the L  1-th layer. If a task does not trigger expansion signal in any layer (implying no significantly different pattern), expansion would not happen, and existing adapters would be reused. More discussions are in Appendix A.1.

The task-specifically added modules can help reduce the interference but cause a linearly-scaled model size (w.r.t. number of tasks) and restrained knowledge sharing and reuse.

Considering that the PTM and the newly added parameters in expansion can provide a stable representation and a knowledge extension mechanism for CL, respectively, we focus on how to further enhance the control of the stabilityplasticity balance during continual expansion. Although task-specific expansion of PTMs [68, 92] directly reduces the cross-task conflicts, it causes undesired linear scaling of model size and may impair knowledge transfer/reuse [55, 65, 70]. To address these issues, we propose SEMA, a CL approach with Self-Expansion of pre-trained models with Modularized Adaptation. It automatically expands PTMs with modularized adapters on demand and continually learns them to accommodate the distribution shifts without overwriting previously learned knowledge. Unlike existing methods that expand PTMs with a pre-defined fixed-size pool [51, 74, 83, 90] or task-specific components [68, 73, 92], we design modularized adapters to enable SEMA automatically decide when and where (i.e., which layer) to expand the PTM (i.e., a pre-trained ViT) on demand for tackling new requirements with sufficient and flexible plasticity, as shown in Fig. 1. The model continually learns how to compose the learned adapters. With the enhanced knowledge transfer and reuse, SEMA can thus perform better by only expanding the parameter size sub-linearly.

We introduce modular/modularized adapters that can be identified and reused to solve new tasks, selectively adding and learning a subset of new adapters for unseen knowledge. Specifically, we design the modular adapter as a pair of a functional adapter and a representation descriptor (RD). The functional adapters produce specific feature representations to adapt to the different requirements of different tasks. The RDs are jointly trained to capture thefeature distribution relevant to the coupled adapter at corresponding layers, serving as indicators of distribution shift at the representation level of intermediate layers. SEMA can use the representation descriptors for self-expansion – a new modular adapter is added at a specific layer if and only if all the representation descriptors indicate the input feature as a unseen pattern; otherwise, the existing frozen adapters are reused, resulting in sub-linear expansion. They can be implemented as a model with density estimation or novelty detection ability, such as autoencoder (AE) [27] or variational autoencoder (VAE) [38]. The module expansion at each layer can happen flexibly to supplement existing representation space, leading to sufficient plasticity. The on-demand expansion strategy strengthens the knowledge transfer and reuse, compared to the task-specific expansion [68, 92]. For example, cat images and dog images have more shared features than food images; the SEMA model trained only on cat images tends to expand more new adapters when training on food images than on dog images. To effectively compose the adapters, we design an expandable weighting router to produce layerwise weighted mixture of the adapters in a form of mixture of experts (MoE), which are expanded and learned in the self-expansion process. Despite the RDs may be used for adapter assignment by hard selection, the learned soft mixture router can perform more effectively (Appendix C.3). We summarize our contributions as follows:

• We propose a novel continual learning approach via selfexpansion of PTMs with modularized adapters, i.e. SEMA. In CL, it automatically determines the expansion necessity and location for new adapters, adding them at specific layers to accommodate new patterns in samples. The model enhances the control of stability-plasticity trade-off through adapter reuse and flexible expansion performed only on demand. SEMA enables sub-linear expansion and operates without the need for rehearsal.

• To achieve SEMA, we introduce modular adapters comprising a functional adapter and a representation descriptor. The representation descriptor maintains the distribution of pertinent input features, serving as a local novel pattern detector for expansion during training. The expandable weighting router is maintained simultaneously for composing the adapters via weighted mixture.

• Extensive experiments are conducted to validate the effectiveness and analyze the behavior of the proposed method, which demonstrates the model’s ability on alleviating forgetting and knowledge transfer as well as the plausibility of the automated process.

## 2. Related Work

Continual Learning (CL). The mainstream taxonomy classifies continual learning methods into three categories: replay-based methods, regularization-based methods and architecture-based methods [14, 71]. Replay-based methods aim to alleviate catastrophic forgetting by retaining a memory buffer to store the information from old tasks for future replay [6, 8, 48, 59]. With simple intuition and effectiveness in preventing forgetting, these methods are limited by the size of the memory buffer and may also raise privacy concerns. An alternative approach is to implicitly maintain a generative model for producing pseudo-samples with similar distribution to old classes [11, 37, 60, 61, 67]. Regularization-based methods penalize significant changes to important parameters for seen tasks [2, 4, 39, 53, 84, 85], or consolidate the knowledge learnt from previous tasks with knowledge distillation [28, 41, 46, 88]. Instead of using all available parameters for all tasks, architecture-based methods allocate a subset of parameters dedicated to each task, which can be performed with task masking [36, 49, 66, 75] or dynamic architecture [3, 31, 43, 44, 55, 70, 78–81]. These methods tend to achieve optimal performance with less forgetting as isolating parameters and growing capacity for novel tasks reduce task interference during training, however, they are mostly restricted to simple applications due to the complex model design.

Parameter-Efficient Fine-Tuning (PEFT). Parameterefficient fine-tuning methods train a small set of additional parameters rather than the entire pre-trained model, which reduces the demands placed upon computational resources. Prompt tuning modifies input tokens/prefixes via learnable prompts [33, 45]. LoRA [30] injects low-rank matrices to approximate weight updates and avoids additional inference latency via re-parameterization, which has been further utilized as experts with mixture modeling in recent works [16, 21, 72, 76]. Adapters introduced by [29], along with its variants [9, 34], insert lightweight learnable modules into the transformer. To enhance the efficacy of adapter learning, [23] investigates different insertion forms, and [12, 57, 63] explores the potential of adapter compositions.

PTM-based CL. Recent works adopt PTMs, such as ViT and CLIP, as the backbone in the CL system to exploit its robust representational ability and enable further adaptation on downstream tasks [32, 62, 87, 89]. PTM can serve as a feature extractor for prototypes, which can be used for classification with distance measurement [51, 52, 56, 90]. PEFT techniques are also widely used to adapt PTMs in CL, including adaptation and prompting. L2P [74] and Dual-Prompt [73] apply a pool of prompts in CL through visual prompt tuning [33]. The prompt learning process is further improved by [68] with an attention mechanism and inputconditioned weights. ConvPrompt [62] adds parameter per task using linguistic knowledge from a large language model. Similar to prompt tuning in $\mathrm { C L } ,$ , some works also explore the use of a fixed set of adapters [13, 17, 82] or task-oriented expansion [47, 92] for better transfer of ViT to downstream CL tasks. [19] builds a unified framework incorporating both prompt and adapter-based methods. [10] adds experts in the pre-training of large language models (LLMs).

## 3. Methodology

## 3.1. Problem Definition

Continual learning constructs a scenario where the model is required to learn from sequentially arriving tasks [14]. Consider a sequence of T tasks $( \mathcal { D } ^ { 1 } , \mathbf { \bar { \mathcal { D } } } ^ { 2 } , . . . , \mathbf { \bar { \mathcal { D } } } ^ { T } )$ with distribution shift, where $\mathcal { D } ^ { t } = \{ ( x _ { i } ^ { t } , y _ { i } ^ { t } ) \} _ { i = 1 } ^ { n _ { t } }$ is the dataset containing $n _ { t }$ data samples for the t-th task. Only the training samples from $\mathcal { D } ^ { t }$ are accessible while seeing the t-th task [74], if without additional ER process [8]. In a typical class-incremental learning (CIL) scenario [14], the classes in different tasks are non-overlapping, specifically, with the label space of the t-th task denoted by $Y _ { t } , Y _ { t } \cap Y _ { t ^ { \prime } } = \emptyset$ for $t \neq t ^ { \prime }$ . Let $F _ { \theta } : X  Y$ (with X and Y denoting the domain of input and label) be a model parameterized with ✓. The goal of CL is to learn one model $F _ { \theta }$ that can minimize the objective on each task t in the stream: $\mathbb { E } _ { ( x , y ) \in D ^ { t } } { \mathcal { L } } _ { \mathrm { C E } } ( F _ { \theta } ( x ) , y )$ where $\mathcal { L } _ { \mathrm { C E } } ( \cdot , \cdot )$ denotes the cross entropy loss in CIL.

![](images/839341fb6773fa01b71c99ea0d31f74c421f33229adf295db51e6534cca08add.jpg)  
Figure 2. Overview of the model architecture. (a) shows the structure of expandable adapter modules with adapters, RDs and router. (b) shows the scenario where expansion is triggered by representations with distribution different to previous tasks, estimated by RD. RDs are trained to align with the feature distribution of the corresponding task via only $\mathcal { L } _ { \mathrm { R D } }$ , unaffected by gradients from the classification loss. (c) shows the scenario where incoming distribution can be handled by previously added modules, resulting in no expansion and adapter reuse

## 3.2. Overview

We propose a PTM-based CL approach (i.e., SEMA) with a self-expansion mechanism to automatically add modularized adapters at arbitrary layers of the PTM (i.e., a pre-trained ViT with frozen parameters) on demand for handling automatically detected novel patterns in CL task stream, as shown in Fig. 1 and 2. The proposed method simultaneously learns a weighted mixture router for composing the adapters for different inputs. The design enhances the balance of knowledge transfer/reuse and plasticity for handling novelty, with only sub-linear expansion rate [5, 55].

To achieve the modularized design of SEMA, we introduce the modular adapters containing a pair of functional adapter $f _ { \phi } ( \cdot )$ and representation descriptor $g _ { \varphi } ( \cdot )$ , as defined in Sec. 3.3. Each added functional adapter works as a branch of a specific layer of the pre-trained transformer; and the representation descriptor indicates the feature distribution that can be handled by the paired $f _ { \phi } ( \cdot )$ . In CL, when new tasks arrive, $g _ { \varphi } ( \cdot ) \cdot \mathrm { \mathbf { s } }$ of the already-added adapters are used to detect novel feature patterns layer-by-layer. Only when novel pattern $( i . e .$ , representation-level distribution shifts) are detected, new adapters, $i . e . ,$ , pairs of $( f _ { \phi } ( \cdot ) , g _ { \varphi } ( \cdot ) )$ ), are added and trained. After trained sufficiently, the adapters are kept frozen to alleviate forgetting and can be reused in future tasks. The details of the self-expansion strategy are in Sec. 3.6. At each layer of the PTM, an expandable weighting router is continually maintained and updated for composing the adapters via weighted mixture, as introduced in Sec. 3.4. When no adapters are added, the existing frozen adapters are retrieved and reused.

## 3.3. Representation-Aware Modular Adapter

The modular adapter $( f _ { \phi } ( \cdot ) , g _ { \varphi } ( \cdot ) )$ is designed as a pair of functional adapter $f _ { \phi } ( \cdot )$ and a representation descriptor $g _ { \varphi } ( \cdot )$ , which enables the module to be aware of the distribution of the local representation. One or more adapters can be added at arbitrary blocks/layers of the transformer.

Functional adapter. In a (pre-trained) ViT, there are L layers of transformer blocks, where each of them mainly contains a multi-head self-attention (MHSA) module and a multi-layer perceptron (MLP) module [15], as shown in Fig. 2. We keep all the parameters in the ViT frozen and perform adaptation through the learnable parameters in the continually added adapters. As a commonly used solution [9, 90], the functional adapter with learnable parameters is added as a side branch of the MLP in any layer of the ViT.

Let $\mathbf { x } ^ { l } \in \mathbb { R } ^ { d }$ denote the feature input of the MLP at l-th layer/block of ViT. In the proposed method, there can be different numbers $( i . e . , K ^ { l } )$ of adapters added at each layer through the self-expansion process. The k-th functional adapter at l-th layer is denoted as $f _ { \phi _ { k } ^ { l } } \left( \cdot \right)$ . Each $f _ { \phi _ { k } ^ { l } } \left( \cdot \right)$ takes $\mathbf { x } ^ { l }$ as input to bridge the representation gap between the pretrained model and the downstream tasks. By default, we implement $f _ { \phi _ { k } ^ { l } } \left( \cdot \right)$ as a lightweight adapter [9] containing a down-projection layer with parameters $\mathbf { W } _ { \mathrm { d o w n } , k } ^ { l } \in \mathbb { R } ^ { d \times r }$ , an up-projection layer with parameters $\mathbf { W } _ { \mathbf { u p } , k } ^ { l } \in \mathbb { R } ^ { r \times d }$ , and a non-linear ReLU activation [1] in between. By taking $\mathbf { x } ^ { l }$ as input, the output of each functional adapter is formulated as

$$
f _ { \phi _ { k } ^ { l } } ( \mathbf { x } ^ { l } ) = \mathrm { R e L U } ( \mathbf { x } ^ { l } \cdot \mathbf { W } _ { \mathrm { d o w n } , k } ^ { l } ) \cdot \mathbf { W } _ { \mathrm { u p } , k } ^ { l } ,\tag{1}
$$

where $\phi _ { k } ^ { l } \equiv \{ \mathbf { W } _ { \mathrm { u p } , k } ^ { l } , \mathbf { W } _ { \mathrm { d o w n } , k } ^ { l } \}$ and $\mathbf { x } ^ { l }$ is treated as row vector for notation simplicity. If there is only one adapter at the l-th layer $( i . e . , K ^ { l } = 1 )$ , the output representation of the MLP is adjusted as $\mathbf { x } _ { \mathrm { o u t } } ^ { l } = \mathbf { M } \mathbf { L } \mathbf { P } ( \mathbf { x } ^ { l } ) + f _ { \phi _ { r } ^ { l } } ( \mathbf { x } ^ { l } )$ . SEMA can continually expand the model with more than one adapters if needed. The number of adapters at each layer is automatically determined on demand, with a rate that is sub-linear w.r.t. number of tasks. Although similar adapter formulation have been used to handle CL, they only perform adaptation on the first task using only one adapter [51, 90] or periodically expand the PTM using task-specific adapters linearly [92]. In addition to Eq. 1, the functional adapters can also be implemented as other forms, such as LoRA [30], as discussed in Sec. 4.3.

Representation descriptor. The representation descriptor (RD) $g _ { \varphi _ { k } ^ { l } } ( \cdot )$ is paired with the functional descriptor $f _ { \phi _ { k } ^ { l } } \left( \cdot \right)$ to capture the characteristics of the local representation. It is designed and trained to indicate what kind of input representation can be handled by the corresponding functional adapter at each specific layer. Representation descriptors can be implemented as any model with density estimation or novelty detection ability. For simplicity, we implement them as AE [27], containing an encoder and a decoder. When a new pair of modular adapter is added at layer l, the RD $g _ { \varphi _ { k } ^ { l } } ( \cdot )$ is trained by minimizing the reconstruction loss on all the features fed to $f _ { \phi _ { k } ^ { l } } ( \cdot ) , i . e . , \mathcal { X } _ { k } ^ { l . }$

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { R D } , k } ^ { l } ( x ) = \sum _ { \mathbf { x } \in \mathcal { X } _ { k } ^ { l } } | | \mathbf { x } - g _ { \varphi _ { k } ^ { l } } ( \mathbf { x } ) | | _ { 2 } ^ { 2 } . } \end{array}\tag{2}
$$

In our expansion strategy (in Sec. 3.6), when a new task t arrives, at each l-th layer, if all existing RDs detect significantly novel distributions (based on the z-score of reconstruction errors), the expansion signal is triggered. $f _ { \phi _ { k } ^ { l } } \left( \cdot \right)$ and $g _ { \varphi _ { k } ^ { l } } \left( \cdot \right)$ are trained on this task t and then kept frozen in the future. $\mathcal { X } _ { k } ^ { l }$ represents the input feature $\mathbf { x } ^ { l }$ of all the samples in this new expansion-triggering task t.

## 3.4. Expandable Weighting Router for Mixture Usage of Adapters

By definition, the representation descriptor can be used to compose the adapters, as in similar modular networks. However, it heavily relies on the statistics of similar inputs in a batch [55] and can be unreliable for individual inputs. We thus directly maintain and learn an expandable weighting router for a weighted mixture of the functional adapters.

For any l-th layer with $K ^ { l }$ adapters, the routing function is defined as $h _ { \psi ^ { l } } ( \cdot ) : \mathbb { R } ^ { d }  \mathbb { R } ^ { K ^ { l } }$ . Similar to [16], we implement $h _ { \psi ^ { l } } ( \cdot )$ as a linear mapping function followed by a softmax operation $\mathbf { w } ^ { l } = h _ { \psi ^ { l } } ( \mathbf { \bar { x } } ^ { l } ) \equiv \mathrm { s o f t m a x } ( \mathbf { x } ^ { l } \cdot \mathbf { W } _ { \mathrm { m i x } } ^ { l } )$ where $\mathbf { W } _ { \operatorname* { m i x } } ^ { l } \in \mathbb { R } ^ { d \times K ^ { l } }$ is the parameter of $\psi ^ { l }$ . As shown in Fig. 2, the weights $\mathbf { w } ^ { l } \in \mathbb { R } ^ { K ^ { l } }$ can produce the mixture of the added functional adapters to produce the output representation of the MLP in the transformer:

$$
\mathbf { x } _ { \mathrm { o u t } } ^ { l } = \mathbf { M L P } ( \mathbf { x } _ { } ^ { l } ) + et { } { ' } \sum _ { k = 1 } ^ { K ^ { l } } w _ { k } ^ { l } \cdot f _ { \phi _ { k } ^ { l } } ( \mathbf { x } _ { } ^ { l } ) .\tag{3}
$$

When new adapter is added at any layer $l ,$ the router $h _ { \psi ^ { l } } ( \cdot )$ $i . e . , \textbf { W } _ { \operatorname* { m i x } } ^ { l } .$ , is expanded for producing weights with one more dimension. The expanded router is trained together with the added adapters. While expanding the router, the parameters corresponding to the existing adapters remain frozen and only the newly added ones (i.e., a newly added column in $\mathbf { W } _ { \operatorname* { m i x } } ^ { i } )$ are trained. This approach, similar to the common practice for training classification heads in CL [47, 68], controls and restricts forgetting in the expandable router (shown in Fig. 5), though it cannot fully eliminate it.

## 3.5. Continual Learning Objective of SEMA

In SEMA, the model $F _ { \theta } ( \cdot )$ for solving the tasks consists of learnable parameters from the functional adapters and router with learnable parameters, $i . e . , \left\{ \phi _ { k } ^ { l } \right\}$ and $\{ \psi ^ { l } \}$ . The learnable parameters are dynamically added and learned. The representation descriptors are learned jointly for maintaining a state of the local representation. The overall objective in SEMA optimizes all these parameters:

$$
\begin{array} { r l } & { \underset { \{ \phi _ { k } ^ { l } \} , \{ \psi ^ { l } \} , \{ \varphi _ { k } ^ { l } \} } { \operatorname* { m i n } } \sum _ { t = 1 } ^ { T } \mathbb { E } _ { ( x , y ) \in D ^ { t } } \left[ \mathcal { L } _ { \mathrm { C E } } ( F _ { \{ \phi _ { k } ^ { l } \} , \{ \psi ^ { l } \} } ( x ) , y ) \right. } \\ & { \qquad \left. + \sum _ { l = 1 } ^ { L } \sum _ { k = 1 } ^ { K ^ { l } } \mathcal { L } _ { \mathrm { R D } , k } ^ { l } ( x ; \varphi _ { k } ^ { l } ) \right] . } \end{array}\tag{4}
$$

Learning of modular adapters is executed only when new modules are added. The learned modules are kept frozen to prevent forgetting. Optimization of RDs can be parallel to other parameters. If no module is added in a specific task due to no significant pattern being identified by RDs, the existing modules can be reused without training.

## 3.6. Self-Expansion Strategy

The RDs provide the capacity to decide when and where to expand the model. We designed a more specific strategy to achieve the reliable self-expansion in the CL task stream.

Task-oriented expansion. The expansion may occur at any time as new samples are seen during training. To incorporate the task identification prior knowledge in CL, especially CIL, we improve parameter efficiency and expansion stability with task-oriented expansion. We restrict the addition to at most one adapter per layer for each task. When a new task t arrives, the method scans all samples in the first epoch to decide whether to expand the model. If the expansion signal is triggered, only one adapter is added and then trained for the whole task; otherwise, the task t data can reuse learned modules and the learning process moves to the next task.

z-score based expansion signal. When scanning through the new task data, an expansion signal at layer l is triggered when significantly new patterns are identified. It reflects that a $\mathbf { x } ^ { l }$ is out of the scope of all RDs, i.e., reconstruction error is high with each $g _ { \varphi _ { k } ^ { l } } \left( \mathbf { x } \right) \left[ 2 0 \right]$ , as illustrated in Fig. 4. However, it is impractical to directly use reconstruction error due to the perturbation and heterogeneous characteristics of each task and adapter. We thus compute and maintain the running statistics $\mu _ { k } ^ { l }$ and standard deviation $\sigma _ { k } ^ { l }$ of reconstruction error on all relevant inputs used in training. Given any $x ^ { l }$ in the scanning process for a future task, the z-score corresponding to each existing RD can be calculated as $z _ { k } ^ { l } = ( r _ { k } ^ { l } - \mu _ { k } ^ { l } ) / \sigma _ { k } ^ { l }$ with $r _ { k } ^ { l }$ as reconstruction error. If all $z _ { k } ^ { l } \mathrm { \Delta } ^ { , } \mathrm { s }$ for $k = 1 , . . . , K ^ { l }$ are larger than a threshold, the expansion signal is triggered. Considering that the z-score has normalized out perturbation and scale, the process can be very robust to the threshold setting, as shown in Sec. 4.3.

<table><tr><td rowspan="2">Method</td><td colspan="2">CIFAR-100</td><td colspan="2">5-Task IN-R</td><td colspan="2">10-Task IN-R</td><td colspan="2">20-Task IN-R</td><td colspan="2">ImageNet-A</td><td colspan="2">VTAB</td></tr><tr><td>A</td><td>AN</td><td>A</td><td>AN</td><td>A</td><td>AN</td><td>A</td><td>AN</td><td>A</td><td>AN</td><td>Ã</td><td> $\mathcal { A } _ { N }$ </td></tr><tr><td>FT Adapter</td><td>47.88</td><td>30.9</td><td>53.91</td><td>41.23</td><td>45.31</td><td>30.93</td><td>38.51</td><td>24.22</td><td>29.78</td><td>17.64</td><td>59.98</td><td>43.50</td></tr><tr><td>L2P</td><td>84.77</td><td>77.87</td><td>77.40</td><td>73.59</td><td>66.97</td><td>62.72</td><td>70.67</td><td>62.90</td><td>47.16</td><td>38.48</td><td>81.19</td><td>80.83</td></tr><tr><td>DualPrompt</td><td>86.60</td><td>80.43</td><td>76.39</td><td>72.29</td><td>72.83</td><td>66.75</td><td>62.33</td><td>61.97</td><td>59.54</td><td>50.23</td><td>82.89</td><td>79.79</td></tr><tr><td>CODA-P</td><td>91.55</td><td>86.11</td><td>81.63</td><td>76.98</td><td>81.11</td><td>75.25</td><td>75.00</td><td>70.02</td><td>47.29</td><td>35.02</td><td>79.88</td><td>81.58</td></tr><tr><td>SimpleCIL</td><td>82.31</td><td>76.21</td><td>65.83</td><td>61.31</td><td>67.09</td><td>61.35</td><td>67.59</td><td>61.35</td><td>60.05</td><td>49.24</td><td>85.29</td><td>83.61</td></tr><tr><td>ADAM</td><td>90.55</td><td>85.62</td><td>79.91</td><td>74.25</td><td>79.11</td><td>73.15</td><td>75.84</td><td>69.10</td><td>60.15</td><td>49.24</td><td>85.29</td><td>83.61</td></tr><tr><td>InfLoRA</td><td>90.51</td><td>85.05</td><td>78.58</td><td>72.58</td><td>81.39</td><td>75.32</td><td>78.87</td><td>72.60</td><td>59.71</td><td>46.21</td><td>88.90</td><td>87.63</td></tr><tr><td>SEMA</td><td>91.37</td><td>86.98</td><td>84.75</td><td>79.78</td><td>83.56</td><td>78.00</td><td>81.75</td><td>74.53</td><td>64.53</td><td>53.32</td><td>91.26</td><td>89.64</td></tr></table>

Table 1. Comparison with ViT-based CL methods in CIL. All models adopt ViT-B/16-IN1K as the backbone.

![](images/ac957c3992ff370d63e8561897a909094dd757b20213b999952f768ddf254c55.jpg)

![](images/97d6db69423a9a9468be4d7bdb4d3dadda0bc47e4435e78ff3be34762162dff1.jpg)

![](images/a7dbe6d922da7ae98e799cbc807b784a0ac06c4f3b38d39096c01b76166701f7.jpg)

![](images/03047eeffd3d108eb6ea6e30aea99de5b677122ff8db66383e03b7db54349392.jpg)  
Figure 3. Incremental performance of different methods on class-incremental learning benchmarks.

Multi-layer expansion. We facilitate self-expansion across multiple layers through distinct decision processes. Upon encountering a new task, self-expansion operations are executed sequentially from shallow layers to deeper layers. As new adapters are introduced at shallow levels, training ensures representations are aligned accordingly. Subsequently, the model determines whether to continue expanding into subsequent layers. The adaptable multi-layer expansion facilitates the accommodation of various distribution shifts and enables flexible inter-class knowledge sharing [18, 42].

## 4. Experiments

## 4.1. Setting and Implementation Details

Datasets. Experiments are conducted on common datasets used for pre-trained ViT-based CIL: CIFAR-100 [40], ImageNet-R (IN-R) [25], ImageNet-A [26] and VTAB [86]. Baselines. We validate our method by comparing with PTMbased rehearsal-free CL approaches using similar backbone (e.g., ViT) and methodology, including fully fine-tuning of the adapter, L2P [74], DualPrompt [73], CODA-P [68], SimpleCIL [90], ADAM with Adapter [90] and InfLoRA [47]. Training details. We use the commonly used ViT-B/16 model [15] weights pre-trained on ImageNet-1K [64] as the PTM weights. We also conducted experiments with other pre-trained weights and left discussions in Appendix C.1. The batch size is set to 32. SGD is used as the optimizer with the initial learning rate set to 0.005 and 0.01 for adapters and RDs, respectively, decaying with cosine annealing. The hidden dimension of adapter is 16. In experiments, by default, we enable self-expansion in the last three transformer layers for simplicity without losing generality.

## 4.2. Experimental Results

We validate the proposed method by comparing with previous related state-of-the-art methods and reporting the average accuracy of all tasks $\mathcal { A } _ { N }$ [7] and average incremental accuracy <sup>¯</sup> [59] metrics in Tab. 1. It shows that our method performs better than other related methods in terms of the average accuracy at the last step $\mathcal { A } _ { N }$ , which reflects the final goal of CL. Fig. 3 shows the variation in accuracy during the continual learning process.It shows the consistently superior performance of SEMA in the process. Although most previous approaches exhibit strong performance on CIFAR-100, the proposed methods shows more improvements on datasets containing adversarial samples similar to those found in ImageNet, due to its better stability-plasticity balance.

## 4.3. Ablation Studies and Analyses

Ablation studies on module expansion and adapter composing. We conduct ablation studies to demonstrate the effectiveness of the self-expansion process and investigate the influence of different adapter composing strategies, with the results reported in Tab. 2. We first conduct an experiment by removing the self-expansion process and only keeping the first-session adaptation (No Exp.), which is similar to ADAM [90] with slight difference on implementation. The

results show that the self-expansion can work reliably to continually improve the adaptation results.
<table><tr><td>Method</td><td colspan="2">ImageNet-A Ã AN</td><td colspan="2">VTAB À  $\mathcal { A } _ { N }$ </td></tr><tr><td>SEMA</td><td>64.53</td><td>53.32</td><td>91.26</td><td>89.64</td></tr><tr><td>No Exp.</td><td>61.20</td><td>49.90</td><td>86.21</td><td>83.66</td></tr><tr><td>Avg. W.</td><td>56.88</td><td>44.31</td><td>90.84</td><td>89.14</td></tr><tr><td>Rand. W.</td><td>62.95</td><td>49.77</td><td>88.87</td><td>85.17</td></tr><tr><td>Top-1 Sel.</td><td>62.00</td><td>50.56</td><td>90.83</td><td>88.61</td></tr><tr><td>Rand. Sel.</td><td>61.70</td><td>50.36</td><td>90.82</td><td>88.51</td></tr><tr><td>Top-1 Sel. Inf.</td><td>61.96</td><td>50.36</td><td>90.95</td><td>88.84</td></tr></table>

Table 2. Ablation studies on adapter expansion and composing.

To demonstrate the benefits of the weighted mixture routing, we investigate several variants of SEMA with different adapter composing strategies. Firstly, we study two variants with a soft mixture of adapters relying average weighting (Avg. W.) and random weighting (Rand. W.), respectively. Tab. 2 shows that the expandable weighting router learns an effective weighting function. We further study the variants that perform routing by selecting only a single adapter indicated by the highest value from the learned weighting router (Top-1 Sel.) or through random drawing (Rand. Sel.). Additionally, we evaluate SEMA trained with mixture routing, using an inference strategy that selects only the adapter with the highest weight (Top-1 Sel. Inf.). The results show that the weighted soft mixture of the learned adapters works more effectively by encouraging the better usage of the learned adapters. More experiments about adapter composing using representation descriptor are in Appendix C.3.

Analysis on dynamic expansion process. To demonstrate how the representation descriptors are learned and how they work for self-expansion in CL, we visualize the reconstruction error of each AE-based RD corresponding to each sample seen during training, i.e., their representation features at specific layer, in Fig. 4. For more intuitive visualization and simplified experiment, in this analysis, we restrict the automatic self-expansion only to the last layer of transformer. The analysis is conducted on VTAB dataset. In this case shown in Fig. 4, the reconstruction error of each RD decreases and converges after training on the corresponding task, after the RD is added for handling this task. When a new task arrives, the reconstruction errors for the existing RDs are calculated and used to detect novelty. The expansion signal is generated when significantly high reconstruction errors (scaled as z-scores) are detected from all the previous RDs (in Task 2 and 3). In Task 4 and 5, all samples can be well covered by at least one previous RD, which implies no significant distribution shift is detected and results in no expansion. Note that the z-score (i.e., a normalized version of reconstruction error) is used for expansion in SEMA.

Analysis on adapter usage. Fig. 5 demonstrates the average adapter usage of each task from VTAB. This analysis is produced by restricting self-expansion to the last layer, as in

![](images/ade9b4d0ca65ec96d5668731d9c228e3e3c78b39362ed207501d72650100e203.jpg)  
Figure 4. Reconstruction error during training to show the dynamic expansion process. Expansion occurs for Tasks 1, 2, and 3, while no expansion is triggered for Tasks 4 and 5 due to no detected distribution shift.

![](images/6247147c60ff8b98d7a033bbc46a5c2f6ca8bdc6d955d6b4daf84293c25d2da8.jpg)  
Figure 5. Visualization of adapter usage on VTAB. Adapters 1, 2, and 3 are added and trained on Tasks 1, 2, and 3, respectively. Tasks 4 and 5 primarily reuse Adapters 1 and 3 due to similar feature distributions with Tasks 1 and 3.

Fig. 4. Self-expansion is automatically produced for Task 1, 2 and 3. For tasks that triggered expansion, the adapters used are primarily those they were trained with, as shown in the figure. Task 4 and 5 share a similar selection pattern with the tasks they are similar with (Task 1 and 3 respectively), showing that added adapters are effectively reused for new tasks. More details are in Appendix C.3.

Study of expansion threshold. We investigate the impact of the expansion threshold on accuracy and the number of added adapters using ImageNet-A and VTAB. Firstly, the results in Fig. 6 show that the proposed method is not sensitive to the setting of the threshold, benefiting from the z-score-based expansion signal. Fig. 6b and 6d show how the threshold influences the number of added adapters (at each layer), displaying trends consistent with those in Fig. 6a and 6c. Fig. 6a and 6b show that a smaller expansion threshold leads to more frequent expansion, which could boost the performance at some level through more parameters. A threshold that is too large (e.g., values over 1.5) minimizes the chance for expansion, which may lead to insufficient adaptation. In SEMA, a proper expansion threshold within a wide range can lead to a balance between the performance gain and the parameter size.

Analysis of multi-layer expansion. In Fig. 7, we explore the effects on accuracy by implementing expansion across varying numbers of layers, ranging from the last 2 layers (#11-#12) to the last 4 layers (#9-#12). Intuitively, allowing expansion in deeper layers enables better adaptation to different tasks. However, as shown in Fig. 7b and Fig. 7d, permitting expansion in early transformer layers also increases the overall number of added adapters, without a significant boost in performance as earlier layers tend to behave similarly despite distribution shifts. Also, enforcing addition of too many adapters may cause difficulty in training, especially in early transformer layers.

![](images/d15e10aebd7093dd87b648e1f48f5172091fadd048dbfae0d1d857dcc90a6a3c.jpg)  
(a) Accuracy

![](images/93d0409c059edc2211fec02d605a289b6843ec63833b898d386b1ddd7a504e77.jpg)  
(b) Num. of adapters

![](images/2014cf332b4bff609127c358459de6179917e9b4b0bd307ddd9ba9d3888b3c8e.jpg)  
(c) Accuracy

![](images/d41379d9a82b65608c31a743484935ab242348a5b121f53db8929a53223522e8.jpg)  
(d) Num. of adapters

Figure 6. Analysis of the impact of expansion threshold with (a)(b) ImageNet-A and (c)(d) VTAB. (a) and (c) show that SEMA can produce good accuracy stably with slight variation w.r.t. varying expansion threshold. (b) and (d) report how the number of added adapters (on the specific Transformer layers #10, #11, #12) changes with the varying threshold values, corresponding to (a) and (c), respectively. The proposed method is insensitive to the threshold. Adding more adapters may lead to higher accuracy, a proper threshold can achieve a balance between performance and model size.  
![](images/ef203f934115cd491a928955f15eed711506dc1fbdc329e4d90644e16f6a37d1.jpg)  
(a) Accuracy

![](images/29b22fe216543f5adf1737b2a61b3e955298fe1cf49f44c57d8054be6b3e1dab.jpg)  
(b) Num. of adapters

![](images/2353e45ab6c74fcdfd98c5dfda9860a2edf08bb45495346b58d5002e6df598a4.jpg)  
(c) Accuracy

![](images/895d6a6a19ed7e1fd81a895dfa628307a4444c849231426b457280c07c8be035.jpg)  
(d) Num. of adapters  
Figure 7. Analysis of the effect of multi-layer expansion, with (a)(b) ImageNet-A and (c)(d) VTAB. By enabling automatic selfexpansion on multiple transformer layers, SEMA can achieve better performance than restricting that on a single layer.

<table><tr><td>Method</td><td colspan="2">ImageNet-A À AN</td><td colspan="2">VTAB À  $\mathcal { A } _ { N }$ </td></tr><tr><td>Adapter[9]</td><td>64.53</td><td>53.32</td><td>91.26</td><td>89.64</td></tr><tr><td>LoRA[30]</td><td>63.50</td><td>52.67</td><td>91.85</td><td>88.53</td></tr><tr><td>Convpass[34]</td><td>63.48</td><td>51.74</td><td>90.68</td><td>88.62</td></tr></table>

Table 3. Different adapter variants.

![](images/0fbe7fff7d53c44f8b393dfefb2293230c0fc400d7f92b8b17fe18030c543c98.jpg)  
Figure 8. Analysis on added parameters (in Millions) during model deployment on ImageNet-A.

Ablation studies on adapter variants. Apart from Adapter [9], we extend our evaluation to other variants, namely LoRA [30] and Convpass [34]. As shown in Tab. 3, our proposed approach is robust to the choice of adapter methods, showing the broad applicability and effectiveness of our dynamic expansion strategy across different adapter methods.

Sub-linear growth of parameters. In Fig. 8, instead of expanding w.r.t. number of tasks, SEMA adds parameters at a sub-linear rate, showing the efficiency of the self-expansion mechanism. Further analysis is provided in Appendix C.2.

## 5. Conclusion

In this paper, we propose a novel self-expandable modularized adaptation approach for continual learning. SEMA learns to reuse and add modules in an automated manner without memory rehearsal. We incorporate an efficient expansion strategy with detection for feature distribution shifts in different layers of transformer-based models, successfully mitigating the forgetting problem of jointly using the fixed set of parameters. Experimental results demonstrate the outstanding performance of SEMA to datasets with different levels of distribution shifts.

Limitations and future work. We perform the task-oriented expansion at most once per layer for each task considering the CIL characteristics and parameter efficiency. The design can be more flexible to enable fully online dynamic expansion, which could open possibility in better adaptation for data with intra-task diversity and enable online CL. Moreover, the expansion of SEMA is based on the distribution shift detection ability from RDs, which could be further enhanced by elevating the optimization of RDs and expansion protocol to a meta level with a closed loop.

## References

[1] Abien Fred Agarap. Deep learning using rectified linear units (relu). arxiv 2018. arXiv preprint arXiv:1803.08375, 1803. 4

[2] Hongjoon Ahn, Sungmin Cha, Donggyu Lee, and Taesup Moon. Uncertainty-based continual learning with adaptive regularization. Advances in neural information processing systems, 32, 2019. 3

[3] Rahaf Aljundi, Punarjay Chakravarty, and Tinne Tuytelaars. Expert gate: Lifelong learning with a network of experts. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 3366–3375, 2017. 3

[4] Rahaf Aljundi, Francesca Babiloni, Mohamed Elhoseiny, Marcus Rohrbach, and Tinne Tuytelaars. Memory aware synapses: Learning what (not) to forget. In Proceedings of the European conference on computer vision (ECCV), pages 139–154, 2018. 3

[5] Jacob Andreas, Marcus Rohrbach, Trevor Darrell, and Dan Klein. Neural module networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 39–48, 2016. 4

[6] Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, and Simone Calderara. Dark experience for general continual learning: a strong, simple baseline. Advances in neural information processing systems, 33:15920–15930, 2020. 1, 3

[7] Arslan Chaudhry, Marc’Aurelio Ranzato, Marcus Rohrbach, and Mohamed Elhoseiny. Efficient lifelong learning with A-GEM. In 7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019. OpenReview.net, 2019. 1, 6

[8] Arslan Chaudhry, Marcus Rohrbach, Mohamed Elhoseiny, Thalaiyasingam Ajanthan, Puneet K Dokania, Philip HS Torr, and Marc’Aurelio Ranzato. On tiny episodic memories in continual learning. arXiv preprint arXiv:1902.10486, 2019. 1, 3

[9] Shoufa Chen, Chongjian Ge, Zhan Tong, Jiangliu Wang, Yibing Song, Jue Wang, and Ping Luo. Adaptformer: Adapting vision transformers for scalable visual recognition. Advances in Neural Information Processing Systems, 35:16664–16678, 2022. 1, 3, 4, 8

[10] Wuyang Chen, Yanqi Zhou, Nan Du, Yanping Huang, James Laudon, Zhifeng Chen, and Claire Cui. Lifelong language pretraining with distribution-specialized experts. In International Conference on Machine Learning, pages 5383–5395. PMLR, 2023. 3

[11] WU Chenshen, L Herranz, LIU Xialei, et al. Memory replay gans: Learning to generate images from new categories without forgetting [c]. In The 32nd International Conference on Neural Information Processing Systems, Montréal, Canada, pages 5966–5976, 2018. 3

[12] Alexandra Chronopoulou, Matthew E. Peters, Alexander Fraser, and Jesse Dodge. Adaptersoup: Weight averaging to improve generalization of pretrained language models. In Findings of the Association for Computational Linguistics: EACL 2023, Dubrovnik, Croatia, May 2-6, 2023, pages 2009– 2018. Association for Computational Linguistics, 2023. 3

[13] Yawen Cui, Zitong Yu, Rizhao Cai, Xun Wang, Alex C Kot, and Li Liu. Generalized few-shot continual learning with contrastive mixture of adapters. arXiv preprint arXiv:2302.05936, 2023. 1, 3

[14] Matthias De Lange, Rahaf Aljundi, Marc Masana, Sarah Parisot, Xu Jia, Aleš Leonardis, Gregory Slabaugh, and Tinne Tuytelaars. A continual learning survey: Defying forgetting in classification tasks. IEEE transactions on pattern analysis and machine intelligence, 44(7):3366–3385, 2021. 1, 3

[15] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021. 1, 4, 6

[16] Shihan Dou, Enyu Zhou, Yan Liu, Songyang Gao, Jun Zhao, Wei Shen, Yuhao Zhou, Zhiheng Xi, Xiao Wang, Xiaoran Fan, et al. Loramoe: Revolutionizing mixture of experts for maintaining world knowledge in language model alignment. arXiv preprint arXiv:2312.09979, 2023. 3, 5

[17] Beyza Ermis, Giovanni Zappella, Martin Wistuba, Aditya Rawal, and Cedric Archambeau. Memory efficient continual learning with transformers. Advances in Neural Information Processing Systems, 35:10629–10642, 2022. 3

[18] Chongyang Gao, Kezhen Chen, Jinmeng Rao, Baochen Sun, Ruibo Liu, Daiyi Peng, Yawen Zhang, Xiaoyuan Guo, Jie Yang, and VS Subrahmanian. Higher layers need more lora experts, 2024. 6

[19] Qiankun Gao, Chen Zhao, Yifan Sun, Teng Xi, Gang Zhang, Bernard Ghanem, and Jian Zhang. A unified continual learning framework with general parameter-efficient tuning. arXiv preprint arXiv:2303.10070, 2023. 3

[20] Dong Gong, Lingqiao Liu, Vuong Le, Budhaditya Saha, Moussa Reda Mansour, Svetha Venkatesh, and Anton van den Hengel. Memorizing normality to detect anomaly: Memoryaugmented deep autoencoder for unsupervised anomaly detection. In Proceedings ofthe IEEE/CVF international conference on computer vision, pages 1705–1714, 2019. 5

[21] Yunhao Gou, Zhili Liu, Kai Chen, Lanqing Hong, Hang Xu, Aoxue Li, Dit-Yan Yeung, James T Kwok, and Yu Zhang. Mixture of cluster-conditional lora experts for visionlanguage instruction tuning. arXiv preprint arXiv:2312.12379, 2023. 3

[22] Raia Hadsell, Dushyant Rao, Andrei A Rusu, and Razvan Pascanu. Embracing change: Continual learning in deep neural networks. Trends in cognitive sciences, 24(12):1028– 1040, 2020. 1

[23] Junxian He, Chunting Zhou, Xuezhe Ma, Taylor Berg-Kirkpatrick, and Graham Neubig. Towards a unified view of parameter-efficient transfer learning. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net, 2022. 3

[24] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016. 1

[25] Dan Hendrycks, Steven Basart, Norman Mu, Saurav Kadavath, Frank Wang, Evan Dorundo, Rahul Desai, Tyler Zhu, Samyak Parajuli, Mike Guo, et al. The many faces of robustness: A critical analysis of out-of-distribution generalization. In ICCV, pages 8340–8349, 2021. 6

[26] Dan Hendrycks, Kevin Zhao, Steven Basart, Jacob Steinhardt, and Dawn Song. Natural adversarial examples. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15262–15271, 2021. 6

[27] Geoffrey E Hinton and Ruslan R Salakhutdinov. Reducing the dimensionality of data with neural networks. science, 313 (5786):504–507, 2006. 2, 5

[28] Saihui Hou, Xinyu Pan, Chen Change Loy, Zilei Wang, and Dahua Lin. Lifelong learning via progressive distillation and retrospection. In Proceedings ofthe European Conference on Computer Vision (ECCV), pages 437–452, 2018. 3

[29] Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin De Laroussilhe, Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly. Parameter-efficient transfer learning for nlp. In International Conference on Machine Learning, pages 2790–2799. PMLR, 2019. 3

[30] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021. 3, 5, 8

[31] Ching-Yi Hung, Cheng-Hao Tu, Cheng-En Wu, Chien-Hung Chen, Yi-Ming Chan, and Chu-Song Chen. Compacting, picking and growing for unforgetting continual learning. Advances in Neural Information Processing Systems, 32, 2019. 3

[32] Saurav Jha, Dong Gong, and Lina Yao. CLAP4CLIP: Continual learning with probabilistic finetuning for vision-language models. In Thirty-eighth Conference on Neural Information Processing Systems, 2024. 3

[33] Menglin Jia, Luming Tang, Bor-Chun Chen, Claire Cardie, Serge Belongie, Bharath Hariharan, and Ser-Nam Lim. Visual prompt tuning. In European Conference on Computer Vision, pages 709–727. Springer, 2022. 1, 3

[34] Shibo Jie and Zhi-Hong Deng. Convolutional bypasses are better vision transformer adapters. arXiv preprint arXiv:2207.07039, 2022. 3, 8

[35] Daniel Justus, John Brennan, Stephen Bonner, and Andrew Stephen McGough. Predicting the computational cost of deep learning models. In 2018 IEEE international conference on big data (Big Data), pages 3873–3882. IEEE, 2018. 1

[36] Zixuan Ke, Bing Liu, Nianzu Ma, Hu Xu, and Lei Shu. Achieving forgetting prevention and knowledge transfer in continual learning. Advances in Neural Information Processing Systems, 34:22443–22456, 2021. 3

[37] Ronald Kemker and Christopher Kanan. Fearnet: Braininspired model for incremental learning. arXiv preprint arXiv:1711.10563, 2017. 3

[38] Diederik P. Kingma and Max Welling. Auto-encoding variational bayes. In 2nd International Conference on Learning Representations, ICLR 2014, Banff, AB, Canada, April 14-16, 2014, Conference Track Proceedings, 2014. 2

[39] James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A Rusu, Kieran Milan,

John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, et al. Overcoming catastrophic forgetting in neural networks. Proceedings of the national academy of sciences, 114(13): 3521–3526, 2017. 1, 3

[40] A. Krizhevsky and G. Hinton. Learning multiple layers of features from tiny images. Master’s thesis, Department of Computer Science, University ofToronto, 2009. 6

[41] Kibok Lee, Kimin Lee, Jinwoo Shin, and Honglak Lee. Overcoming catastrophic forgetting with unlabeled data in the wild. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 312–321, 2019. 3

[42] Yoonho Lee, Annie S. Chen, Fahim Tajwar, Ananya Kumar, Huaxiu Yao, Percy Liang, and Chelsea Finn. Surgical finetuning improves adaptation to distribution shifts. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023. 6

[43] Xilai Li, Yingbo Zhou, Tianfu Wu, Richard Socher, and Caiming Xiong. Learn to grow: A continual structure learning framework for overcoming catastrophic forgetting. In Proceedings of the 36th International Conference on Machine Learning, ICML 2019, 9-15 June 2019, Long Beach, California, USA, pages 3925–3934. PMLR, 2019. 3

[44] Xilai Li, Yingbo Zhou, Tianfu Wu, Richard Socher, and Caiming Xiong. Learn to grow: A continual structure learning framework for overcoming catastrophic forgetting. In International Conference on Machine Learning, pages 3925–3934. PMLR, 2019. 3

[45] Xiang Lisa Li and Percy Liang. Prefix-tuning: Optimizing continuous prompts for generation. arXiv preprint arXiv:2101.00190, 2021. 3

[46] Zhizhong Li and Derek Hoiem. Learning without forgetting. IEEE transactions on pattern analysis and machine intelligence, 40(12):2935–2947, 2017. 1, 3

[47] Yan-Shuo Liang and Wu-Jun Li. Inflora: Interference-free low-rank adaptation for continual learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 23638–23647, 2024. 3, 5, 6

[48] David Lopez-Paz and Marc’Aurelio Ranzato. Gradient episodic memory for continual learning. Advances in neural information processing systems, 30, 2017. 3

[49] Arun Mallya, Dillon Davis, and Svetlana Lazebnik. Piggyback: Adapting a single network to multiple tasks by learning to mask weights. In Proceedings of the European conference on computer vision (ECCV), pages 67–82, 2018. 3

[50] Michael McCloskey and Neal J Cohen. Catastrophic interference in connectionist networks: The sequential learning problem. In Psychology of learning and motivation, pages 109–165. Elsevier, 1989. 1

[51] Mark McDonnell, Dong Gong, Amin Parvaneh, Ehsan Abbasnejad, and Anton van den Hengel. RanPAC: Random projections and pre-trained models for continual learning. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. 1, 2, 3, 5, 6

[52] Lu Mi, Hao Wang, Yonglong Tian, Hao He, and Nir N Shavit. Training-free uncertainty estimation for dense regression: Sensitivity as a surrogate. In Proceedings ofthe AAAI Con-

ference on Artificial Intelligence, pages 10042–10050, 2022. 3

[53] Cuong V Nguyen, Yingzhen Li, Thang D Bui, and Richard E Turner. Variational continual learning. arXiv preprint arXiv:1710.10628, 2017. 3

[54] Cuong V Nguyen, Alessandro Achille, Michael Lam, Tal Hassner, Vijay Mahadevan, and Stefano Soatto. Toward understanding catastrophic forgetting in continual learning. arXiv preprint arXiv:1908.01091, 2019. 1

[55] Oleksiy Ostapenko, Pau Rodriguez, Massimo Caccia, and Laurent Charlin. Continual learning via local module composition. Advances in Neural Information Processing Systems, 34:30298–30312, 2021. 1, 2, 3, 4, 5

[56] Francesco Pelosin. Simpler is better: off-the-shelf continual learning through pretrained backbones. arXiv preprint arXiv:2205.01586, 2022. 3

[57] Jonas Pfeiffer, Aishwarya Kamath, Andreas Rücklé, Kyunghyun Cho, and Iryna Gurevych. Adapterfusion: Nondestructive task composition for transfer learning. In Proceedings of the 16th Conference of the European Chapter of the Associationfor Computational Linguistics: Main Volume, EACL 2021, Online, April 19 - 23, 2021, pages 487–503. Association for Computational Linguistics, 2021. 3

[58] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In Proceedings of the 38th International Conference on Machine Learning, pages 8748–8763. PMLR, 2021. 7

[59] Sylvestre-Alvise Rebuffi, Alexander Kolesnikov, Georg Sperl, and Christoph H Lampert. icarl: Incremental classifier and representation learning. In Proceedings of the IEEE conference on Computer Vision and Pattern Recognition, pages 2001–2010, 2017. 1, 3, 6

[60] Matthew Riemer, Tim Klinger, Djallel Bouneffouf, and Michele Franceschini. Scalable recollections for continual lifelong learning. In Proceedings of the AAAI conference on artificial intelligence, pages 1352–1359, 2019. 3

[61] Mohammad Rostami, Soheil Kolouri, and Praveen K Pilly. Complementary learning for overcoming catastrophic forgetting using experience replay. arXiv preprint arXiv:1903.04566, 2019. 3

[62] Anurag Roy, Riddhiman Moulick, Vinay K Verma, Saptarshi Ghosh, and Abir Das. Convolutional prompting meets language models for continual learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 23616–23626, 2024. 3

[63] Andreas Rücklé, Gregor Geigle, Max Glockner, Tilman Beck, Jonas Pfeiffer, Nils Reimers, and Iryna Gurevych. Adapterdrop: On the efficiency of adapters in transformers. In Proceedings ofthe 2021 Conference on Empirical Methods in Natural Language Processing, EMNLP 2021, Virtual Event / Punta Cana, Dominican Republic, 7-11 November, 2021, pages 7930–7946. Association for Computational Linguistics, 2021. 3

[64] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy,

Aditya Khosla, Michael Bernstein, et al. Imagenet large scale visual recognition challenge. International journal of computer vision, 115:211–252, 2015. 6

[65] Jonathan Schwarz, Wojciech Czarnecki, Jelena Luketina, Agnieszka Grabska-Barwinska, Yee Whye Teh, Razvan Pascanu, and Raia Hadsell. Progress & compress: A scalable framework for continual learning. In International conference on machine learning, pages 4528–4537. PMLR, 2018. 1, 2

[66] Joan Serra, Didac Suris, Marius Miron, and Alexandros Karatzoglou. Overcoming catastrophic forgetting with hard attention to the task. In International conference on machine learning, pages 4548–4557. PMLR, 2018. 1, 3

[67] Hanul Shin, Jung Kwon Lee, Jaehong Kim, and Jiwon Kim. Continual learning with deep generative replay. Advances in neural information processing systems, 30, 2017. 3

[68] James Seale Smith, Leonid Karlinsky, Vyshnavi Gutta, Paola Cascante-Bonilla, Donghyun Kim, Assaf Arbelle, Rameswar Panda, Rogerio Feris, and Zsolt Kira. Coda-prompt: Continual decomposed attention-based prompting for rehearsal-free continual learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11909–11919, 2023. 1, 2, 3, 5, 6

[69] Hai-Long Sun, Da-Wei Zhou, Han-Jia Ye, and De-Chuan Zhan. Pilot: A pre-trained model-based continual learning toolbox. arXiv preprint arXiv:2309.07117, 2023. 1

[70] Tom Veniat, Ludovic Denoyer, and Marc’Aurelio Ranzato. Efficient continual learning with modular networks and taskdriven priors. arXiv preprint arXiv:2012.12631, 2020. 1, 2, 3

[71] Liyuan Wang, Xingxing Zhang, Hang Su, and Jun Zhu. A comprehensive survey of continual learning: Theory, method and application. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024. 1, 3

[72] Yaqing Wang, Sahaj Agarwal, Subhabrata Mukherjee, Xiaodong Liu, Jing Gao, Ahmed Hassan Awadallah, and Jianfeng Gao. AdaMix: Mixture-of-adaptations for parameterefficient model tuning. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 5744–5760, Abu Dhabi, United Arab Emirates, 2022. Association for Computational Linguistics. 3

[73] Zifeng Wang, Zizhao Zhang, Sayna Ebrahimi, Ruoxi Sun, Han Zhang, Chen-Yu Lee, Xiaoqi Ren, Guolong Su, Vincent Perot, Jennifer Dy, et al. Dualprompt: Complementary prompting for rehearsal-free continual learning. In European Conference on Computer Vision, pages 631–648. Springer, 2022. 1, 2, 3, 6

[74] Zifeng Wang, Zizhao Zhang, Chen-Yu Lee, Han Zhang, Ruoxi Sun, Xiaoqi Ren, Guolong Su, Vincent Perot, Jennifer Dy, and Tomas Pfister. Learning to prompt for continual learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 139–149, 2022. 1, 2, 3, 6

[75] Mitchell Wortsman, Vivek Ramanujan, Rosanne Liu, Aniruddha Kembhavi, Mohammad Rastegari, Jason Yosinski, and Ali Farhadi. Supermasks in superposition. Advances in Neural Information Processing Systems, 33:15173–15184, 2020. 1, 3

[76] Xun Wu, Shaohan Huang, and Furu Wei. MoLE: Mixture of loRA experts. In The Twelfth International Conference on Learning Representations, 2024. 3

[77] Qingsen Yan, Dong Gong, Yuhang Liu, Anton van den Hengel, and Javen Qinfeng Shi. Learning bayesian sparse networks with full experience replay for continual learning. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 109–118, 2022. 1

[78] Shipeng Yan, Jiangwei Xie, and Xuming He. Der: Dynamically expandable representation for class incremental learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3014–3023, 2021. 1, 3

[79] Fei Ye and Adrian G Bors. Task-free continual learning via online discrepancy distance learning. Advances in Neural Information Processing Systems, 35:23675–23688, 2022.

[80] Fei Ye and Adrian G Bors. Self-evolved dynamic expansion model for task-free continual learning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 22102–22112, 2023.

[81] Jaehong Yoon, Eunho Yang, Jeongtae Lee, and Sung Ju Hwang. Lifelong learning with dynamically expandable networks. arXiv preprint arXiv:1708.01547, 2017. 3

[82] Jiazuo Yu, Yunzhi Zhuge, Lu Zhang, Ping Hu, Dong Wang, Huchuan Lu, and You He. Boosting continual learning of vision-language models via mixture-of-experts adapters. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 23219–23230, 2024. 3

[83] Jiazuo Yu, Yunzhi Zhuge, Lu Zhang, Dong Wang, Huchuan Lu, and You He. Boosting continual learning of visionlanguage models via mixture-of-experts adapters. In CVPR, 2024. 1, 2

[84] Friedemann Zenke, Ben Poole, and Surya Ganguli. Continual learning through synaptic intelligence. In International conference on machine learning, pages 3987–3995. PMLR, 2017. 3

[85] Chen Zeno, Itay Golan, Elad Hoffer, and Daniel Soudry. Task agnostic continual learning using online variational bayes. arXiv preprint arXiv:1803.10123, 2018. 3

[86] Xiaohua Zhai, Joan Puigcerver, Alexander Kolesnikov, Pierre Ruyssen, Carlos Riquelme, Mario Lucic, Josip Djolonga, Andre Susano Pinto, Maxim Neumann, Alexey Dosovitskiy, et al. A large-scale study of representation learning with the visual task adaptation benchmark. arXiv preprint arXiv:1910.04867, 2019. 6

[87] Gengwei Zhang, Liyuan Wang, Guoliang Kang, Ling Chen, and Yunchao Wei. Slca: Slow learner with classifier alignment for continual learning on a pre-trained model. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 19148–19158, 2023. 3

[88] Junting Zhang, Jie Zhang, Shalini Ghosh, Dawei Li, Serafettin Tasci, Larry Heck, Heming Zhang, and C-C Jay Kuo. Class-incremental learning via deep model consolidation. In Proceedings of the IEEE/CVF Winter Conference on Applications ofComputer Vision, pages 1131–1140, 2020. 3

[89] Zangwei Zheng, Mingyuan Ma, Kai Wang, Ziheng Qin, Xiangyu Yue, and Yang You. Preventing zero-shot transfer degradation in continual learning of vision-language models.

In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 19125–19136, 2023. 3

[90] Da-Wei Zhou, Han-Jia Ye, De-Chuan Zhan, and Ziwei Liu. Revisiting class-incremental learning with pre-trained models: Generalizability and adaptivity are all you need, 2023. 1, 2, 3, 4, 5, 6

[91] Da-Wei Zhou, Yuanhan Zhang, Jingyi Ning, Han-Jia Ye, De-Chuan Zhan, and Ziwei Liu. Learning without forgetting for vision-language models, 2023. 1

[92] Da-Wei Zhou, Hai-Long Sun, Han-Jia Ye, and De-Chuan Zhan. Expandable subspace ensemble for pre-trained modelbased class-incremental learning. In CVPR, 2024. 1, 2, 3, 5, 6

[93] Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu. Learning to prompt for vision-language models. International Journal ofComputer Vision, 130(9):2337–2348, 2022. 1