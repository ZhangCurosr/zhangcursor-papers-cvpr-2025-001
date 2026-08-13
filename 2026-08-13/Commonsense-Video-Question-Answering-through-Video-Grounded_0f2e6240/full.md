# Commonsense Video Question Answering through Video-Grounded Entailment Tree Reasoning

Huabin Liu<sup>1,3\*</sup>, Filip Ilievski<sup>2</sup>, Cees G. M. Snoek<sup>3</sup> <sup>1</sup> Shanghai Jiao Tong University <sup>2</sup> Vrije Universiteit Amsterdam <sup>3</sup> University of Amsterdam huabinliu@sjtu.edu.cn, f.ilievski@vu.nl, c.g.m.snoek@uva.nl

## Abstract

This paper proposes the first video-grounded entailment tree reasoning method for commonsense video question answering (VQA). Despite the remarkable progress of large visual-language models (VLMs), there are growing concerns that they learn spurious correlations between videos and likely answers, reinforced by their black-box nature and remaining benchmarking biases. Our method explicitly grounds VQA tasks to video fragments in four steps: entailment tree construction, video-language entailment verification, tree reasoning, and dynamic tree expansion. A vital benefit of the method is its generalizability to current video- and image-based VLMs across reasoning types. To support fair evaluation, we devise a de-biasing procedure based on large-language models that rewrites VQA benchmark answer sets to enforce model reasoning. Systematic experiments on existing and de-biased benchmarks highlight the impact of our method components across bench marks, VLMs, and reasoning types.

## 1. Introduction

This paper proposes a video-grounded reasoning method for commonsense video question answering (VQA). VQA has a long tradition in computer vision [11, 16, 25, 26], with remarkable recent progress obtained through videoand image-language models [10, 12, 14, 17, 33] (throughout this paper collectively referred to as vision-language models, or VLMs). Yet, there are growing concerns that their improved performance is based on learning shortcut associations between videos and likely answers, as opposed to reasoning [27]. Such concerns are reinforced by the black-box nature of these models [12, 14], which prohibits a deeper understanding of their decision-making process.

We are inspired by recent work in natural language processing, where entailment trees have emerged as a mechanism to explicitly analyze answer candidates, using LLMs to recursively decompose a candidate into hypotheses and natural language inference formalisms to evaluate the hypotheses [4]. Entailment trees provide an explicit reasoning chain that explains the model’s decision-making process and enables verification of each step, thus addressing concerns about shortcut learning. Recently, Sanders et al. [19] have devised a mechanism to apply entailment trees to videos. However, their work assumes video transcripts are explicitly provided to evaluate answers, thus avoiding the complexity of grounding hypotheses into video content. In this paper, we propose the first video-grounded entailment tree reasoning method for commonsense VQA.

![](images/065f69728badb2f217939016addb1f9168d765dcea52246b73425e77de339749.jpg)  
Figure 1. Given a video questioning answering task, our framework performs explicit reasoning over an entailment tree, where answer options are transformed into statements. These statements are then recursively decomposed and verified based on videogrounded evidence relevant to the question.

Our method explicitly grounds VQA tasks to video fragments in four steps: (i) entailment tree construction, (ii) video-language entailment verification, (iii) tree reasoning, and (iv) dynamic tree expansion. As shown in Fig. 1, given a video and a multiple-choice question, we generate a state ment for each answer candidate that acts as a first-level hypothesis. We decompose each statement iteratively, aiming to produce sub-statements that can be confidently verified in the video. The video is itself decomposed into partitions, consisting of sets of frames. Verifying each statement is then a matter of aligning it to a video partition. A vital benefit of the method is its generalizability to current video and image-based VLMs across reasoning types, including temporal and causal. To demonstrate its video reasoning ability, we develop an answer-set de-bias procedure supported by an LLM that ensures that VQA benchmarks [11, 26] are adequate for reasoning in videos without relying on spurious correlations. Our experiments show that our video-grounded entailment tree method consistently improves video- and image-based baselines on both the existing and de-biased benchmarks. Moreover, it performs on par with, and often better than, state-of-the-art video-based VLMs while leveraging 257× fewer parameters. Further ablations show that the method benefits from considering both textual and video information and that its performance is especially strong on causal and temporal questions.

## 2. Related work

Video question answering. Recent research has shown that while video-based VLMs can achieve state-of-the-art performance, their answers are sensitive to object size, positioning, and speed [1, 28]. Moreover, when answering temporal and spatial questions, VLMs rely on textual biases to “guess” answers rather than performing genuine understanding and reasoning over visual-text information [32].

To improve the robustness and interpretability of VLMs, one line of research enriches VLMs with visual grounding functionality during QA, which enables VLMs to localize relevant video moments [18, 27, 30] or key frames [15, 22] to support answers. However, while these methods localize visual evidence, the process by which VLMs use it to deduce answers remains opaque. Another approach leverages external LLMs as reasoners or agents to enhance interpretability in textual modality. For instance, LLoVi [31] converts VQA to a text-based QA task via video captioning, then prompts an LLM to provide answers. Similarly, VideoAgent [21] uses an LLM to recursively determine if the current frames can answer the given question based on their textual descriptions. However, these methods heavily rely on the reasoning capabilities of LLMs. Like VLMs, the LLM reasoning process remains a black box, and hallucinations are common. Recently, TV-trees [19] attempted to perform explicit reasoning over both visual and textual modalities using a neuro-symbolic system. However, their work assumes video transcripts are explicitly provided to evaluate answers, thus avoiding the complexity of ground ing hypotheses into video content. Instead, we contribute a general framework for explicit reasoning in commonsense VQA, fueled by a grounding component that aligns question components with video fragments.

Beyond methodology, some works focus on providing fair and comprehensive VLM evaluations in VQA tasks by creating new benchmarks [3, 6, 9]. These benchmarks contain videos with diverse scenarios and durations, with carefully crafted questions and options designed to prevent textual shortcuts that VLMs might exploit. Video-specific questions (e.g., compositional action reasoning) [1], which require insights beyond textual associations, are included to test commonsense reasoning in VLMs. Addressing concerns of remaining biases in such benchmarks [6], we contribute an LLM-based answer-set de-biasing procedure to ensure that VQA benchmarks [11, 26] are adequate to evaluate reasoning in videos rather than spurious correlations.

Systematic language reasoning. As LLMs demonstrate great potential in reasoning, there has been considerable interest in using LLMs to generate systematic explanations to support their answers. The series of Chain-of-Thought prompting [2, 23, 29] encourages LLMs to think stepby-step to perform explicit multi-hop reasoning, providing free-form reasoning steps before arriving at an answer. However, such implicit explanations are not grounded in external knowledge or evidence, which may lead to unverifiable and unfaithful reasoning. Since the development of EntailmentBank [4], research has increasingly focused on constructing explanation trees [20, 24] and graphs [8], encouraging models to generate step-wise entailment proofs of a statement using a set of supporting facts. Entailer [20] introduced this systematic explanation framework into languagebased multiple-choice QA, performing explicit reasoning by generating entailment trees grounded in the model’s internal beliefs. REFLEX [8] extends the entailment tree to form a belief graph for QA models, aiming to address consistency issues by intervening in the intermediate reasoning steps. Instead of grounding facts in predefined rules or model beliefs, NELLIE [24] adopts Prolog-based inference engines and external natural language corpora to build entailment trees as explainable reasoning for multiple-choice QA tasks. While such techniques for natural language processing inspire our framework, we generalize entailment trees to VQA, contributing a novel grounding method that aligns entailment trees with video fragments.

## 3. Video-grounded entailment tree reasoning

This paper devises a novel explainable framework for grounded commonsense VQA. It derives the answers through systematic reasoning over video-text information with entailment trees. Specifically, in the entailment tree (Fig. 2a), each candidate answer is decomposed into statements that entail the answer, explaining why each answer could be plausible. These statements are then grounded in relevant visual evidence from the video to prove or refute them (Fig. 2b). While entailment trees in natural language processing are constructed based on a model’s internal knowledge or corpora [8, 20, 24], we ground entailment trees into video fragments. Finally, backtracking through the entailment tree leads to systematic reasoning over the statements (Fig. 3). Thus, answers can be deduced by a systematic structure with explicit reasoning paths and explanations rather than relying on opaque, black-box models.

## 3.1. Entailment tree construction

Initial statement generation. Given a question and its answer candidates, we first convert each question-answer pair into a declarative sentence that preserves the semantic meaning of the original QA pair. As a result, an N-way multiple-choice $\mathrm { Q A }$ problem produces a set of statements, denoted as $\mathcal { D } { = } d _ { 1 } , \ldots , d _ { N }$ . For example, the two-way question “What did the boy in white do after hefirst took the balloon? (A) resting on a chair (B) carries it toward the hula hoop” is transformed into: $\mathcal { D } : \{ d _ { 1 } = { ^ { \circ } T h e }$ boy in white resting on a chair after first taking the balloon. $\ v { r } , d _ { 2 } = \ v { r } ^ { * } T h e$ boy in white carries it toward the hula hoop after first taking the balloon.” }. Thus, selecting the best answer equals identifying the correct statement for a given video.

Recursive statement decomposition. For each initial statement in D, we generate two sub-statements as proofs that support the statement: Statement ⇐ Sub-statement1, Sub-statement2. The statement is True if and only if both its sub-statements are proved to be True, i.e., the sub-statements entail the statement. Proving the original statements is thus translated into proving two simpler sub-statements. This procedure is recursive: the sub-statements can be further decomposed into further sub-statements that entail them. Therefore, to construct an entailment tree, we recursively decompose these sub-statements as new statements in the next tree layer until reaching the maximum depth or meeting the stop criterion. Fig. 2(a) presents an example of entailment tree generation.

We leverage LLM prompting for both the initial statement generation and the statement decomposition, as these are linguistic tasks (see implementation details).

## 3.2. Video-language entailment verification

Given the entailment tree, the framework then verifies language statements based on the grounded video content as evidence. Specifically, each statement in the entailment tree must be proven or refuted by analyzing the video. A straightforward solution is to encode the whole video to collect information that can be used to verify the statement. However, the critical visual evidence that accurately verifies a statement tends to exist in a local moment instead of the whole video. Therefore, we develop a novel video grounding that guides the verification process to the moments with relevant visual evidence.

Question-aware video captioning. Given a video, we convert its visual information into detailed textual information. Specifically, we input video frames into a VLM-based captioner Cap(·) to obtain a caption $c _ { i } { = } { \mathsf { C a p } } ( f _ { i } )$ for each frame. However, captioning frames individually can overlook essential details or introduce irrelevant information for VQA. In commonsense VQA, questions often focus on specific facts already observed in the video. For example, a typical temporal reasoning question is “What happened before/after Event-A?” where Event-A refers to a fact statement about an event in the video. The fact referenced by the question can be leveraged to guide video understanding. To this end, we first extract the anchor fact indicated by the question and provide it to Cap(·) as prior knowledge, encouraging the generation of relevant captions. Moreover, captions from all previous frames are also provided for each current frame to ensure $\mathrm { C a p } ( \cdot )$ captures the temporal context from the past. This process is formulated as:

$$
c _ { i } = \mathsf { C a p } ( f _ { i } \mid F , ( c _ { 1 } , \cdot \cdot \cdot , c _ { i - 1 } ) ) ,\tag{1}
$$

where F indicates the fact statement.

Video evidence grounding. For commonsense VQA, depending on how the question reasons around the fact statement, the necessary evidence for answers can be gathered from specific video moments. For instance, in the case of temporal reasoning (e.g., before or after questions), the answer should be inferred from moments occurring either before or after the time of the relevant fact. Following this intuition, we design a two-step evidence-grounding strategy to localize the critical moments for answering.

First, given the frame-wise captions, we retrieve a keyframe deemed most relevant to the fact statement, which we refer to as the anchor frame. A straightforward retrieval approach would involve comparing each $c _ { i }$ with the fact description using specific metrics to identify the anchor frame. However, we enhance retrieval accuracy by adopting a structured semantic retrieval strategy. Specifically, the textual descriptions of each frame and fact statement are converted into structured triplets. These triplets capture the attributes and relationships of objects in each frame through structured semantics. As shown in Fig. 2(a), rather than directly comparing raw textual descriptions of frames and fact statements, we use these triplets for retrieval. Inspired by the success of using LLMs for retrieval tasks, we prompt an LLM to conduct anchor frame retrieval using the triplets of the fact statement as the query. The LLM then identifies and returns the most relevant frame ID, i.e., its timestamp.

$$
t _ { \mathrm { a n c h o r } } = \mathsf { R t v } ( c _ { i } , F ) ,\tag{2}
$$

where $t _ { \mathrm { a n c h o r } }$ is the time stamp of the anchor frame, Rtv(·) denotes the retrieval process. Second, we determine the final moment where we should look centered on the $t _ { \mathrm { a n c h o r } } ,$ to incorporate the temporal relations present in the question. Therefore, based on the anchor frame, the navigation for the moment is selected from “look ahead, look behind, look around” by considering the question:

![](images/c89d3bc844cfcb5cb231c94b71eaad3cfc2509fcb2ce6c8467fae7ab7ae02e75.jpg)  
Figure 2. Overview of our framework. (a) The generation of the entailment tree, where statements are recursively decomposed until the tree reaches its max depth or meets the stop criterion. (b) The process of video-language entailment verification: the input video is first converted into textual descriptions. Each caption is then parsed into structured semantics. Given the fact statement as a query, we retrieve the anchor frame. Then, based on the temporal or causal navigation indicated by questions, the visual evidence moment can be grounded.

$$
\mathcal { M } = \mathrm { G n d } ( t _ { \mathrm { a n c h o r } } | \mathcal { Q } ) ,\tag{3}
$$

where Gnd(·) is the grounding process and M denotes the grounded continuous interval in the video. Then, frames are resampled from the video within M and used as visual evidence proving or refuting entailment tree statements.

Visual-text statement prover. Given grounded visual evidence M of the video, statements are estimated to be true or false. Specifically, we employ a VLM as the statement prover, denoted as Prv(·), to evaluate each statement within the tree by probing VLM’s internal belief on this statement. Each statement will be transformed into a binary QA task, with possible options being True or False. We then directly probe the $\mathrm { { P r v } ( \cdot ) }$ with the binary QA prompt and use the next token prediction probabilities of the words to elicit the model’s belief. We normalize the prediction logits of the two options to get the confidence score of that statement. The above process is formulated as:

$$
s _ { d } = \mathtt { P r v } ( \mathcal { M } , h ) , s \in [ 0 , 1 ] ,\tag{4}
$$

![](images/b2093c34ea8ba4b4c86700a33650dc10005de57b17138ba1c7fee88ffe360fa3.jpg)  
Figure 3. Illustration of dynamic tree generation and backtrace. In Step-3, when the proof score of the left statement calculated from its child nodes is less than its direct score $( 0 . 6 3 < 0 . 8 )$ , its decomposition is pruned and stops.

where M is the grounded moment and h denotes the statement that needs to be verified.

## 3.3. Dynamic entailment tree expansion

So far, we have performed statement decomposition recursively to construct an entailment tree with pre-defined depth. However, not all statements need to be verified recursively, especially those easily determined to be true or false by VLMs. Moreover, as the depth increases, some statement sentences are atomic and directly verifiable. Thus, to improve the efficiency of the reasoning process, we further adopt a strategy to expand the entailment tree dynamically. Specifically, each statement d is tied with two confidence scores provided by the $\mathbb { P } \mathbb { r } \mathbb { v } ( \cdot ) ;$

![](images/a59d48c827708cf53d72a9e082f01ab25fe08bd2d8e0e2b01f7cdd82f4a066e0.jpg)  
Figure 4. Illustration of commonsense bias in video question answering. The example is selected from the NExT-QA dataset.

(1) The direct score $s _ { d } ,$ which indicates the belief of Prv(·) model in d.

(2) The proof score $s _ { p } ,$ denoting how confidently the model can prove $d ,$ is calculated by multiplying the scores of its direct sub-statements.

For a statement d, the goal of decomposition is to establish a more reliable and convincing proof path than merely evaluating whether d is true by VLMs. If the decompositionbased reasoning can prove d with higher confidence than its direct score, the overall confidence for statement d should increase. Otherwise, the decomposition should be disregarded. Thus, in the dynamic tree expansion, if decomposition does not enhance a statement’s score, it is pruned, and that statement node becomes a leaf in the entailment tree. Fig. 3 presents a toy example. This criterion ensures that only beneficial decompositions are retained, significantly enhancing the tree reasoning process’s efficiency.

## 3.4. Reasoning over the entailment tree

Finally, we perform a backtrace through the entailment tree to calculate the confidence score of each top statement. Specifically, the final score for each statement is produced by comparing its direct score $s _ { d }$ and proof score $s _ { p } ,$ i.e., $s { = } m a x ( s _ { d } , s _ { p } )$ during backtrace (as shown in Fig. 3). The overall framework selects the answer corresponding to the statement at the top layer with top-scoring proof.

## 4. De-biasing commonsense VQA answer sets

To demonstrate the reasoning ability of video-grounded entailment trees, it is essential to evaluate using commonsense VQA benchmarks that enforce model reasoning. Recent work [13, 18, 27] has provided evidence that shortcuts are present in VQA datasets which enables VLMs to solve these tasks based on textual associations rather than videogrounded reasoning. While VQA benchmarks increasingly focus on commonsense reasoning skills, such as temporal (e.g. after, before) or causal (how, why, what if) relationships in video content, reasoning shortcuts affect the validity of their evaluation. This is illustrated in Fig. 4 (top), where the correct answer (D) is much more relevant to the question and also aligns best with real-world expectations. Consequently, a VLM (VideoLLaVA [14] used in this example) can answer this question correctly by leveraging such associations and without analyzing the video content. Meanwhile, replacing the answer set distractors with other commonsensical answer candidates, as illustrated in Fig. 4 (bottom), makes this task challenging for VLMs. Here, a VLM switches its answer incorrectly to option (C), which confirms the impact of commonsense associations and the lack of grounded reasoning by these models.

![](images/db6aa10cfcca95ececb52c35bd87b3eeb04348accde03b20213cd20c36fec651.jpg)  
Figure 5. Prompt used for rewriting answers on NExT-QA.

To this end, we devise a de-biasing procedure that mitigates reasoning shortcuts in commonsense VQA answer sets. Our de-biasing procedure transforms multiple-choice VQA benchmarks (e.g., NExT-QA) by rewriting their answer distractors while keeping their question and groundtruth answer intact. We prompt an LLM (LLaMA-3) to implement the rewriting procedure for each original QA set. Fig. 5 shows the detailed prompt we used for LLaMA-3 on NExT-QA dataset. This procedure ensures that (1) the answers cannot be easily derived from the QA set associations and (2) the answer remains consistent with the original QA pair. Thus, our procedure enables the scalable construction of de-biased QA sets by leveraging the commonsense associations in LLMs. The next section, focusing on experimental evaluation, analyzes the application of de-biasing to various datasets and its impact on the performance of VLMs, with and without entailment tree reasoning.

## 5. Experiments

## 5.1. Experimental setup

Datasets. We test our framework on three VQA benchmarks: (1) NExT-QA [26], a VQA benchmark for causal and temporal reasoning. (2) IntentQA [11], which focuses on video intent reasoning from both causal and temporal aspects. (3) Video-MME [6], which is a recently proposed comprehensive evaluation benchmark for video analysis; we use its “short-term” split (video length < 2 mins) and 4 question types (temporal, spatial, action, and object reasoning) highly related to commonsense reasoning are selected. Evaluation. We report model performances on our rewritten test set for each dataset and its original test set. We evaluate our framework on all datasets under the multiplechoice QA setting, using a standard accuracy metric.

Baselines. Our baselines represent three categories:

• Video-based VLMs: Video-based VLMs are widely used for VQA tasks, so we include Video-LLaVA [14], VideoChat2 [12], and VideoLLaMA [33]. To test the effectiveness of our framework, we integrate these VLMs by replacing our Prv(·) with specific VLM models.

• Image-based VLMs: We include BLIP-2 [10] and LLaVA-1.5 [17] as Image-LLM baselines.

• State-of-the-art VQA approaches: Recent works in VQA, such as VideoTree [22], VideoAgent [21], and LLoVi [31], are included as strong baselines.

Implementation details. Our entire framework is trainingand human annotation-free. We use LLaMA-3-8B [5] to handle basic functionalities, including (1) converting original QA into declarative statements, (2) statement decomposition, (3) structured semantic extraction and retrieval, and (4) guiding evidence grounding. Detailed prompts for each functionality are provided in the supplementary material. For frame-wise captioning across all datasets, we use LLaVA-1.5 [17] as our default captioner. When comparing with state-of-the-art VQA methods (e.g., VideoAgent), we follow their setup by replacing the captioner with the stronger CogAgent [7] model for fair comparison. When integrating our framework with VLMs, the VLM itself serves as the Prover, Prv(·). For video-LLMs, frames are uniformly sampled from the grounded video moment to meet their input requirements (VideoChat2: 16 frames; VideoLLaVA & VideoLLaMA: 8 frames). For image-language models like LLaVA, we sample 8 frames from the grounded moment and process them individually; the final confidence score is obtained by averaging scores across frames. During dynamic tree generation, we also set a max depth of 5 for the overall entailment tree to improve the efficiency.

## 5.2. Main results

Benefit for image and video-based VLMs. Tab. 1 summarizes the results of our method compared to baselines. Integrating entailment tree reasoning brings consistent improvement across the video- and image-based VLMs for all datasets (1-4% on average). This finding includes the recently proposed benchmark VideoMME, which poses much more challenging videos and questions. The benefits are particularly regular for temporal reasoning (improvement in 14 out of 15 cases), which illustrates how our explicit reasoning process enhances temporal commonsense QA. Image-based VLMs, which initially lack temporal modeling capabilities, perform poorly when directly applied to video QA tasks. However, our framework provides a significant performance boost for these models up to 8% for LLAVA-1.5. By reasoning over multiple sub-problems rather than tackling the entire complex question simultaneously, our reasoning method makes the task more manageable for both video- and image-based VLMs.

Results on de-biased QA sets. As shown in Tab. 2, all models experience considerable performance drops on the de-biased set of the same VQA dataset, which aligns with our observation that current VLMs often rely on textual bias in commonsense reasoning tasks. Notably, videobased VLMs show an 8%-10% decrease in the de-biased set even though the question and the correct answer remain unchanged. In contrast, our proposed framework, which derives answers through an explicit reasoning process based on specific visual evidence, demonstrates much greater robustness on the de-biased set. The improvement brought by our framework on the de-biased set is even higher than on the original test sets. In turn, our framework compensates for the performance loss of the VLMs on the de-biased set. This analysis underscores our framework’s potential to mitigate textual bias in commonsense reasoning. Furthermore, the performance differences between the original and debiased QA sets highlight VQA benchmarks’ limitations in evaluating VLMs’ true reasoning abilities.

Comparison with state-of-the-art. Tab. 3 compares our framework’s results on the de-biased sets to state-of-the-art VQA approaches. The table shows that, next to the consistent benefit our framework provides to various VLMs, it is also competitive with state-of-the-art VQA methods. When applying an advanced captioner and reasoner that aligns with VideoAgent and VideoTree, our framework yields new state-of-the-art results in some cases. In particular, our framework performs best on temporal reasoning questions for all three benchmarks and outperforms all methods on the IntentQA dataset. Importantly, our method reaches such competitive performance despite using 257× fewer parameters for its reasoning compared to state-of-the-art methods.

## 5.3. Ablation studies

Ablation experiments are conducted using VideoLLaVA’s baseline with our framework. Results are reported on the test set of the NExT-QA dataset.

Impact of LLMs for statement decomposition. Entailment generation in our framework relies on prompting an external LLM to recursively decompose statements (cf. Sec. 3.1), which is crucial in guiding reasoning paths. Consequently, we tested various LLMs for entailment tree generation, including open-source models (LLama-3 and Mistral) of different sizes and proprietary LLMs (GPT-4 and Gemini-1.5). The results are summarized in Tab. 4. As expected, the proprietary model GPT-4, known for its strong step-by-step reasoning capabilities, delivers the best performance across all settings. Scaling up LLaMA-3 to 70B offers improvements over the 8B model, though with a notable increase in inference time. As the overall performance difference between models is within 1%, we select LLaMA-3-8B as the default for integrating our framework into VLMs due to its free availability and efficiency.

<table><tr><td colspan="2">Model</td><td colspan="2">NExT-QA</td><td colspan="2">IntentQA</td><td colspan="3">VideoMME</td><td rowspan="3">Avg 34.0</td></tr><tr><td rowspan="4">Image-based VLMs</td><td>BLIP-2 [10]</td><td>Temporal 38.3</td><td>Causal</td><td>Temporal Causal</td><td>Temporal</td><td>Spatial</td><td>Action</td><td>Object</td></tr><tr><td></td><td></td><td>36.1</td><td>43.8 48.6</td><td>25.4</td><td>26.9</td><td>24.2</td><td>28.6 28.8</td></tr><tr><td>+Ours</td><td>45.3</td><td>41.8</td><td>48.9 52.5</td><td>30.5</td><td>27.4</td><td>27.7</td><td>37.9</td></tr><tr><td>LLaVA-1.5 [17]</td><td>37.8</td><td>40.7</td><td>45.8 50.0</td><td>31.1</td><td>33.6</td><td>27.7</td><td>29.8</td><td>37.1</td></tr><tr><td rowspan="6">Video-based VLMs</td><td>+Ours</td><td>45.6</td><td>47.9</td><td>48.4 54.7</td><td>36.7</td><td>36.8</td><td>31.4</td><td>30.7</td><td>41.5</td></tr><tr><td>VideoChat2 [12]</td><td>56.9</td><td>62.1</td><td>60.4 63.2</td><td>50.3</td><td>52.4</td><td>49.5</td><td>50.1</td><td>55.6</td></tr><tr><td>+Ours</td><td>57.8</td><td>61.6</td><td>62.3 63.8</td><td>52.8</td><td>51.5</td><td>51.3</td><td>50.0</td><td>56.4</td></tr><tr><td>VideoLLaVA [14]</td><td>56.0</td><td>60.4</td><td>53.9 60.7</td><td>47.6</td><td>44.3</td><td>46.2</td><td>49.7</td><td>52.3</td></tr><tr><td>+Ours</td><td>58.3</td><td>62.7</td><td>57.6 61.8</td><td>48.3</td><td>46.1</td><td>49.8</td><td>50.3</td><td>54.2</td></tr><tr><td>VideoLLaMA [33] +Ours</td><td>55.4 58.1</td><td>60.2 60.4</td><td>55.1 56.7 54.5 58.9</td><td>44.8 47.4</td><td>47.2 47.8</td><td>44.7 48.6</td><td>49.3 49.1</td><td>51.7 53.1</td></tr></table>

Table 1. Impact on image and video-based VLMs on the original NExT-QA, IntentQA, and VideoMME test sets. Our framework increases accuracy of all video- and image-based VLMs by 1-4% on average across all data partitions. Temporal and action partitions benefit most.

<table><tr><td colspan="2">Model</td><td rowspan="2">BLIP-2</td><td rowspan="2">+Ours</td><td rowspan="2">LLaVA</td><td rowspan="2">+Ours</td><td rowspan="2">Video Chat2</td><td rowspan="2">+Ours</td><td rowspan="2">Video LLaVA</td><td rowspan="2">Video +Ours LLaMA</td><td rowspan="2">+Ours</td></tr><tr><td></td><td></td></tr><tr><td rowspan="2">NExT-QA</td><td>Original Rewritten</td><td>37.2 33.5</td><td>43.6</td><td>39.3</td><td>46.8</td><td>59.5 45.4</td><td>59.7 49.0</td><td>58.2 51.1</td><td>60.5</td><td>57.8 59.3</td></tr><tr><td></td><td></td><td>39.8</td><td>34.8</td><td>44.9</td><td></td><td></td><td>55.4</td><td>41.4 55.9</td><td>47.0</td></tr><tr><td rowspan="2">Intent-QA</td><td>Original</td><td>46.2</td><td>50.7</td><td>47.9</td><td>51.6</td><td>61.8</td><td>63.1 55.7</td><td>57.3</td><td>59.7</td><td>56.7</td></tr><tr><td>Rewritten</td><td>38.2</td><td>45.5</td><td>42.7</td><td>48.6</td><td>52.6</td><td>50.5</td><td>54.7</td><td>46.3</td><td>50.0</td></tr><tr><td colspan="2">Avg</td><td>38.8</td><td>44.9</td><td>41.2</td><td>48.0</td><td>54.8</td><td>56.9</td><td>54.3</td><td>57.6</td><td>50.4 53.3</td></tr></table>

Table 2. Results on de-biased QA sets. Video-based VLMs show significant decreases in the rewritten de-biased set. In contrast, our framework demonstrates much greater robustness on the rewritten set.

Ablation on grounding components. Next, we test the effectiveness of each component in our grounding module (Sec. 3.2). The results, summarized in Tab. 5, indicate that both fact-conditional captioning and structureguided retrieval enhance overall performance by improving grounding accuracy. However, using only structureguided retrieval results in a slight performance drop, possibly because Cap(·) introduces irrelevant semantic information that doesn’t align with the question’s focus, and the structured representation can make identifying anchor frames more challenging. In contrast, fact-conditional captioning alone yields substantial improvement, demonstrating that this straightforward approach can yield an effective and more controllable textual description for videos by conditioning on prior knowledge or relevant facts.

Impact of length of video frames. We further ablate the impact of input video frame length in our framework to determine the optimal number of frame-wise captions to generate per video. The results, summarized in Tab. 6, show that ideal performance is achieved only when sampling a sufficient number of frames (at least 16 for NExT-QA). When fewer frames are used (e.g., 4 or 8), key anchor frames may be missed, reducing the accuracy of grounded visual evidence. Additionally, while increasing the frame count to 32 yields the best performance, it also increases the calls required for Cap(·) to generate frame-wise captions. Balancing efficiency with performance gains, we set 24 frames as the default in our implementation.

Effectiveness of evidence grounding. Our method grounds relevant video fragments to support statements in the entailment tree (Sec. 3.2). To validate its effectiveness, we compare it with two other variations as sources of visual evidence: (1) without evidence grounding, using the full video as evidence, and (2) upper-bound results: manually annotated temporal boundaries provided in the NExT-GQA dataset, indicating where the QA models should focus when producing correct answers. The results are shown in Tab. 7. Compared to the baseline, our video-grounded method provides consistent improvements across the original and debiased sets. The improvement is more apparent in the debiased set, where the answer options are more semantically similar and require more precise, discriminative visual evidence. Using the ground-truth fragment can further boost our approach, suggesting that enhancing grounding accuracy could further improve our framework.

<table><tr><td rowspan="2">Method</td><td rowspan="2">Model Reasoner</td><td colspan="2">NExT-QA*</td><td colspan="2">IntentQA*</td><td colspan="4">VideoMME</td></tr><tr><td>Temporal</td><td>Causal</td><td>Temporal</td><td>Causal</td><td>Temporal</td><td>Spatial</td><td>Action</td><td>Object</td></tr><tr><td>VideoAgent [21]</td><td>GPT-4 (1.8T)</td><td>58.2</td><td>66.6</td><td>60.4</td><td>61.0</td><td></td><td></td><td></td><td></td></tr><tr><td>VideoTree [22]</td><td>GPT-4 (1.8T)</td><td>60.2</td><td>66.4</td><td>56.7</td><td>60.1</td><td>55.7</td><td>54.3</td><td>54.2</td><td>52.6</td></tr><tr><td>LLoVi [31]</td><td>GPT-4 (1.8T)</td><td>53.1</td><td>60.8</td><td>58.6</td><td>61.8</td><td>52.2</td><td>55.3</td><td>51.8</td><td>50.8</td></tr><tr><td>Ours</td><td>VideoLLAVA (7B)</td><td>60.8</td><td>65.9</td><td>61.0</td><td>62.6</td><td>55.9</td><td>53.8</td><td>54.0</td><td>50.8</td></tr></table>

Table 3. Comparison with state-of-the-art. Results for NExT-QA and IntentQA are reported under the de-biased set (the results on the original sets are similar; we provide them in the Appendix). The ‘Reasoner” in these approaches is similar to the “Prover” in our framework The captioner for all methods is CogAgent [7]. Despite other methods relying on much stronger reasoning models, our approach yields competitive performance (four state-of-the-art results) and high parameter efficiency (257× fewer than GPT-4 reasoners).

<table><tr><td rowspan="2">Model class</td><td rowspan="2">LLM</td><td colspan="2">NExT-QA</td></tr><tr><td>Original</td><td>Rewritten</td></tr><tr><td rowspan="3">Open-source</td><td>Mistral-7B</td><td>60.0</td><td>55.6</td></tr><tr><td>LLaMA-3-8B</td><td>60.5</td><td>55.4</td></tr><tr><td>LLaMA-3-70B</td><td>61.3</td><td>55.9</td></tr><tr><td rowspan="2">Proprietary</td><td>Gemini-1.5-Pro</td><td>61.1</td><td>55.2</td></tr><tr><td>GPT-4</td><td>61.6</td><td>56.1</td></tr></table>

Table 4. Impact of LLMs for statement decomposition. The opensource and proprietary models are ordered ascendingly by size. Larger models, especially GPT-4, are best at decomposition, but the smaller models (e.g., LlaMa-3-8B) come close.

<table><tr><td colspan="2">Components</td><td colspan="2">NExT-QA</td></tr><tr><td>Fact-conditioned captioning</td><td>Structure-based retrieval</td><td>Original</td><td>Rewritten</td></tr><tr><td rowspan="3">√</td><td></td><td>56.2</td><td>49.6</td></tr><tr><td></td><td>59.5</td><td>53.3</td></tr><tr><td>√</td><td>58.4</td><td>52.7</td></tr><tr><td>√</td><td>√</td><td>60.5</td><td>55.4</td></tr></table>

Table 5. Ablation on grounding components, showing that both fact-conditional captioning and structure-guided retrieval enhance overall performance by improving grounding accuracy.

<table><tr><td rowspan="2">Acc (NExT-QA)</td><td colspan="5">Frame number</td></tr><tr><td>4</td><td>8</td><td>16</td><td>24</td><td>32</td></tr><tr><td>Original</td><td>57.7</td><td>59.3</td><td>60.5</td><td>60.7</td><td>61.0</td></tr><tr><td>Rewritten</td><td>51.9</td><td>53.4</td><td>55.4</td><td>55.4</td><td>55.7</td></tr></table>

Table 6. Impact of video frame amount. Strong performance requires a sufficiently high frame number (over 16 for NExT-QA).

Effectiveness of dynamic tree expansion. The depth of the entailment tree determines the granularity of reasoning (Sec. 3.3). This ablation analyzes how tree depth impacts overall performance and compares a fixed-depth approach with our dynamic tree generation strategy. Increasing the depth of reasoning yields significant improvements, as complex, long statements are broken down into concise sub-statements that VLMs can understand more effectively. However, extending reasoning beyond the 4th layer offers diminishing returns; for NExT-QA, the original statements’ complexity constrains the task, and some 5th-layer substatements become overly simplistic and less effective for reasoning. This finding highlights the necessity of our dynamic strategy. Applying the dynamic tree expansion strategy, we can see that the performance outperforms the fixeddepth paradigm. In the meantime, the dynamic strategy increases the reasoning efficiency over the entailment tree, more details about efficiency comparison can be found in our supplementary material.

<table><tr><td colspan="2">Video fragment</td><td>Full</td><td>Grounded (ours)</td><td>GT</td></tr><tr><td rowspan="2">NExT-QA</td><td>Original</td><td>58.3</td><td>60.5</td><td>61.8</td></tr><tr><td>Rewritten</td><td>51.7</td><td>55.4</td><td>56.9</td></tr></table>

Table 7. Effectiveness of evidence grounding. Our videogrounded method yields clear improvement over using the full video. More precise grounding can further enhance our accuracy.

<table><tr><td rowspan="2">Strategy</td><td rowspan="2"></td><td colspan="4">Static (Depth=)</td><td rowspan="2">Dynamic</td></tr><tr><td>2</td><td>3</td><td>4</td><td>5</td></tr><tr><td rowspan="2">NExT-QA</td><td>Original</td><td>58.8</td><td>59.2</td><td>60.2</td><td>60.3</td><td>60.5</td></tr><tr><td>Rewritten</td><td>52.0</td><td>53.4</td><td>55.6</td><td>55.3</td><td>55.4</td></tr></table>

Table 8. Effectiveness of dynamic tree expansion. It yields superior accuracy while increasing reasoning efficiency.

## 6. Conclusion

This paper proposed the first video-grounded entailment tree framework for VQA. Moreover, we also contributed a de-biasing procedure to avoid spurious correlations during evaluation and applied it to enhance representative benchmarks. Extensive experiments with five video- and imagebased VLMs demonstrate consistent benefits of our method on these benchmarks. Besides, our proposed framework performs on par with state-of-the-art video reasoning methods despite using 257× fewer parameters. While de-biasing hurts VLM accuracy, our framework regains the accuracy losses and is competitive with state-of-the-art VQA methods.

## References

[1] Piyush Bagad, Makarand Tapaswi, and Cees GM Snoek. Test of time: Instilling video-language models with a sense of time. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2503–2516, 2023. 2

[2] Maciej Besta, Nils Blach, Ales Kubicek, Robert Gerstenberger, Michal Podstawski, Lukas Gianinazzi, Joanna Gajda, Tomasz Lehmann, Hubert Niewiadomski, Piotr Nyczyk, et al. Graph of thoughts: Solving elaborate problems with large language models. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 17682–17690, 2024. 2

[3] Tieyuan Chen, Huabin Liu, Tianyao He, Yihang Chen, Chaofan Gan, Xiao Ma, Cheng Zhong, Yang Zhang, Yingxue Wang, Hui Lin, et al. Mecd: Unlocking multievent causal discovery in video reasoning. arXiv preprint arXiv:2409.17647, 2024. 2

[4] Bhavana Dalvi, Peter Jansen, Oyvind Tafjord, Zhengnan Xie, Hannah Smith, Leighanna Pipatanangkura, and Peter Clark. Explaining answers with entailment trees. arXiv preprint arXiv:2104.08661, 2021. 1, 2

[5] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024. 6

[6] Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. arXiv preprint arXiv:2405.21075, 2024. 2, 6

[7] Wenyi Hong, Weihan Wang, Qingsong Lv, Jiazheng Xu, Wenmeng Yu, Junhui Ji, Yan Wang, Zihan Wang, Yuxiao Dong, Ming Ding, et al. Cogagent: A visual language model for gui agents. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14281– 14290, 2024. 6, 8

[8] Nora Kassner, Oyvind Tafjord, Ashish Sabharwal, Kyle Richardson, Hinrich Schuetze, and Peter Clark. Language models with rationality. arXiv preprint arXiv:2305.14250, 2023. 2, 3

[9] Muhammad Uzair Khattak, Muhammad Ferjad Naeem, Jameel Hassan, Muzammal Naseer, Federico Tombari, Fahad Shahbaz Khan, and Salman Khan. Complex video reasoning and robustness evaluation suite for video-lmms. arXiv preprint arXiv:2405.03690, 2024. 2

[10] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International conference on machine learning, pages 19730– 19742. PMLR, 2023. 1, 6, 7

[11] Jiapeng Li, Ping Wei, Wenjuan Han, and Lifeng Fan. Intentqa: Context-aware video intent reasoning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 11963–11974, 2023. 1, 2, 6

[12] Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, et al.

Mvbench: A comprehensive multi-modal video understanding benchmark. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22195– 22206, 2024. 1, 6, 7

[13] Yicong Li, Xiang Wang, Junbin Xiao, Wei Ji, and Tat-Seng Chua. Invariant grounding for video question answering. In Proceedings of the IEEE/CVF Conference on Computer Vi sion and Pattern Recognition, pages 2928–2937, 2022. 5

[14] Bin Lin, Yang Ye, Bin Zhu, Jiaxi Cui, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection. arXiv preprint arXiv:2311.10122, 2023. 1, 5, 6, 7

[15] Huabin Liu, Weixian Lv, John See, and Weiyao Lin. Taskadaptive spatial-temporal video sampler for few-shot action recognition. In Proceedings of the 30th ACM International Conference on Multimedia, pages 6230–6240, 2022. 2

[16] Huabin Liu, Weiyao Lin, Tieyuan Chen, Yuxi Li, Shuyuan Li, and John See. Few-shot action recognition via intraand inter-video information maximization. arXiv preprint arXiv:2305.06114, 2023. 1

[17] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26296–26306, 2024. 1, 6, 7

[18] Huabin Liu, Xiao Ma, Cheng Zhong, Yang Zhang, and Weiyao Lin. Timecraft: Navigate weakly-supervised tempo ral grounded video question answering via bi-directional reasoning. In European Conference on Computer Vision, pages 92–107. Springer, 2025. 2, 5

[19] Kate Sanders, Nathaniel Weir, and Benjamin Van Durme. Tv-trees: Multimodal entailment trees for neuro-symbolic video reasoning. arXiv preprint arXiv:2402.19467, 2024. 1, 2

[20] Oyvind Tafjord, Bhavana Dalvi Mishra, and Peter Clark. Entailer: Answering questions with faithful and truthful chains of reasoning. arXiv preprint arXiv:2210.12217, 2022. 2, 3

[21] Xiaohan Wang, Yuhui Zhang, Orr Zohar, and Serena Yeung-Levy. Videoagent: Long-form video understanding with large language model as agent. arXiv preprint arXiv:2403.10517, 2024. 2, 6, 8

[22] Ziyang Wang, Shoubin Yu, Elias Stengel-Eskin, Jaehong Yoon, Feng Cheng, Gedas Bertasius, and Mohit Bansal. Videotree: Adaptive tree-based video representation for llm reasoning on long videos. arXiv preprint arXiv:2405.19209, 2024. 2, 6, 8

[23] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022. 2

[24] Nathaniel Weir and Benjamin Van Durme. Dynamic gener ation of grounded logical explanations in a neuro-symbolic expert system. arXiv preprint arXiv:2209.07662, 2022. 2, 3

[25] Hao Wu, Huabin Liu, Yu Qiao, and Xiao Sun. Dibs: Enhanc ing dense video captioning with unlabeled videos via pseudo boundary enrichment and online refinement. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 18699–18708, 2024. 1

[26] Junbin Xiao, Xindi Shang, Angela Yao, and Tat-Seng Chua. Next-qa: Next phase of question-answering to explaining temporal actions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9777–9786, 2021. 1, 2, 6

[27] Junbin Xiao, Angela Yao, Yicong Li, and Tat-Seng Chua. Can i trust your answer? visually grounded video question answering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13204– 13214, 2024. 1, 2, 5

[28] Yu Yang, Besmira Nushi, Hamid Palangi, and Baharan Mirzasoleiman. Mitigating spurious correlations in multimodal models during fine-tuning. In International Conference on Machine Learning, pages 39365–39379. PMLR, 2023. 2

[29] Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. Advances in Neural Information Processing Systems, 36, 2024. 2

[30] Shoubin Yu, Jaemin Cho, Prateek Yadav, and Mohit Bansal. Self-chained image-language model for video localization and question answering. Advances in Neural Information Processing Systems, 36, 2024. 2

[31] Ce Zhang, Taixi Lu, Md Mohaiminul Islam, Ziyang Wang, Shoubin Yu, Mohit Bansal, and Gedas Bertasius. A simple llm framework for long-range video question-answering. arXiv preprint arXiv:2312.17235, 2023. 2, 6, 8

[32] Gengyuan Zhang, Yurui Zhang, Kerui Zhang, and Volker Tresp. Can vision-language models be a good guesser? exploring vlms for times and location reasoning. In Proceedings ofthe IEEE/CVF Winter Conference on Applications of Computer Vision, pages 636–645, 2024. 2

[33] Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding. arXiv preprint arXiv:2306.02858, 2023. 1, 6, 7