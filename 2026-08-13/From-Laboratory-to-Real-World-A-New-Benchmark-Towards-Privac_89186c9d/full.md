# From Laboratory to Real World: A New Benchmark Towards Privacy-Preserved Visible-Infrared Person Re-Identification

Yan Jiang<sup>1,2</sup>, Hao Yu<sup>2</sup>, Xu Cheng<sup>1\*</sup>, Haoyu Chen<sup>2</sup>, Zhaodong Sun<sup>1,2</sup>, Guoying Zhao<sup>2</sup> <sup>1</sup> School of Computer Science, Nanjing University of Information Science and Technology <sup>2</sup> Center for Machine Vision and Signal Analysis, University of Oulu

{jiangyan,xcheng,zhaodong.sun}@nuist.edu.cn {hao.2.yu,chen.haoyu,guoying.zhao}@oulu.fi https://github.com/Joey623/L2RW

## Abstract

Aiming to match pedestrian images captured under varying lighting conditions, visible-infrared person reidentification (VI-ReID) has drawn intensive research attention and achievedpromising results. However, in real-world surveillance contexts, data is distributed across multiple devices/entities, raising privacy and ownership concerns that make existing centralized training impracticalfor VI-ReID. To tackle these challenges, we propose L2RW, a benchmark that brings VI-ReID closer to real-world applications. The rationale of L2RW is that integrating decentralized training into VI-ReID can address privacy concerns in scenarios with limited data-sharing regulation. Specifically, we design protocols and corresponding algorithms for different privacy sensitivity levels. In our new benchmark, we ensure the model training is done in the conditions that: 1) data from each camera remains completely isolated, or 2) different data entities (e.g., data controllers of a certain region) can selectively share the data. In this way, we simulate scenarios with strict privacy constraints which is closer to real-world conditions. Intensive experiments with various server-side federated algorithms are conducted, showing the feasibility of decentralized VI-ReID training. Notably, when evaluated in unseen domains (i.e., new data entities), our L2RW, trained with isolated data (privacy-preserved), achieves performance comparable to SOTAs trained with shared data (privacy-unrestricted). We hope this work offers a novel research entry for deploying VI-ReID that fits real-world scenarios and can benefit the community.

## 1. Introduction

The growing need for dependable pedestrian identification across all hours has made visible-infrared person reidentification (VI-ReID) a key technology [38, 46, 49].

![](images/0a803750b754167c6c27eaa046bc96cd62f6ec568777c80c08b01b67217674ef.jpg)  
Figure 1. Illustration of existing setting and the proposed L2RW setting. (a) The existing setting relies on centralized training, allowing unrestricted data sharing between cameras or entities, which brings serious privacy concerns. (b) Unlike this, our L2RW framework introduces camera-level and entity-level privacy regulation, simulating real-world privacy-preserved VI-ReID scenes.

It solves the challenge of identifying individuals in different spectral modalities, such as visible during the day and infrared at night. However, although existing methods [30, 39–42, 47, 48, 51] achieve encouraging advancements, their deployment in real-world scenarios remains limited.

In our view, the current experimental setting of the VI-ReID in the field is idealized and impedes its advancement to real-world applications. Specifically, current approaches [3, 15, 17, 25, 27, 31, 40, 41, 52] rely predominantly on centralized training, where data from multiple sources is aggregated on a central platform. While beneficial for learning modality-shared representations, they neglect practical constraints, especially in privacy and ownership concerns. In real-world scenarios, data is stored in a distributed manner, surveillance images are subject to strict privacy restrictions, etc. This centralized training not only complicates data sharing but also increases the risk of data leakage, bringing serious privacy concerns in real-world deployment.

Based on the above observation, we argue that it is essential to establish a benchmark that protects data privacy while still enabling effective VI-ReID. Therefore, we propose L2RW (Laboratory to Real World), a benchmark designed to simulate real-world VI-ReID scenarios with privacy preservation at its core. In L2RW, privacy is not an afterthought but a fundamental consideration. However, in reality, determining which data should be protected for privacy remains a challenging issue. For example, 1) pedestrian images captured by surveillance cameras are typically independent and without direct data sharing across cameras. 2) The entity, e.g., the data owner of a certain region, has the right to access all data. When different entities collaborate, should their data be isolated or kept shared?

This complexity brings us to a natural question in designing the L2RW benchmark for real-world VI-ReID: How can we define a proper andfeasible privacyframework to adapt to different privacy levels? Thus, we design three protocols, i.e., Camera Independence (CI), Entity Independence (EI), and Entity Sharing (ES), for flexible privacy level control. For CI, all camera data remains isolated, simulating the highest privacy level where no data sharing occurs across every single camera as shown in the camera level of Fig. 1(b). Regarding a more relaxed privacy level in which all entity data is isolated while data can be shared inside each data entity, we propose the EI protocol as shown in the entity level of Fig. 1(b). When all data entities can be shared for centralized training, we name this protocol ES as shown in Fig. 1(a), which is the commonly used protocol in previous Re-ID methods. In the ES protocol, existing VI-ReID methods [14, 26, 43, 44, 49] can be seamlessly applied. However, note that these algorithms are designed for a closed-world setting, where training and testing are conducted on the same entity. To avoid this, in our ES protocol, we train on multiple shared entities but directly test on an unseen entity.

Evidently, existing VI-ReID methods cannot be directly applied under these privacy-preserving protocols (CI and EI) due to many technical obstacles: 1) In CI, a single camera may consist of only visible, only infrared, or both modalities, this modality mismatch between different cameras makes the training extremely challenging, leading to a modality incomplete issue. 2) In EI, since pedestrians vary across different entities, this non-sharing approach introduces the identity missing issue. 3) In both the CI and EI protocols, differences in data distribution across cameras or entities inevitably bring the domain shift issue.

To address the above challenges and make the L2RW benchmark actionable, we provide a novel algorithm customized for this task, named Decentralized Privacy-Preserved Training (DPPT) as a baseline solution. Inspired by the substantial success of federated learning [10, 11, 13, 18, 19, 23, 37], our DPPT employs decentralized training to ensure privacy protection. To solve the modality incomplete issue, DPPT converts the existing two-stream architecture into a one-stream architecture and modifies the current sampling way to eliminate the need for modality distinction. To handle the identity missing and domain shift issues, we design a memory rectification bank (MRB) that selects the top K features closest to the center point within each class as representatives. By averaging these top K features, we correct the center point to better reflect the current dominant class distribution. Then, MRB averages the dominant centers across clients and stores them in a global memory, which serves as a fair convergence point. This enables different clients to optimize towards the same goal. Generally, the contributions of this paper are summarized as follows:

• We propose a benchmark named L2RW to simulate privacy-preserving real-world VI-ReID scenarios. Within the L2RW benchmark, we design three protocols, i.e., CI (Camera Independence), EI (Entity Independence) and ES (Entity Sharing). This simulates scenarios with different privacy constraints, bringing VI-ReID evaluation closer to real-world conditions.

• We analyze the challenges encountered under the CI and EI protocols and propose a unified method named DPPT, which is the first work that handles privacy concerns for VI-ReID in a decentralized manner.

• In our L2RW benchmark, unlike existing methods that validate solely on a single dataset, we merge several existing datasets and conduct the evaluation in a cross-domain manner to simulate the real-world scenarios.

• Extensive experiments on three public VI-ReID datasets confirm the feasibility of decentralized training in L2RW, with our method achieving significant improvement on various federated learning baselines under CI. We also show that DPPT achieves performance under EI comparable to that of ES.

## 2. Related Work

## 2.1. Visible-Infrared Person ReID

VI-ReID aims to match visible and infrared images of the same pedestrian captured by non-overlapped cameras. Many studies have emerged recently, achieving remarkable progress. For instance, Zhang et al. [49] explored diverse modality-shared features to mine significant cross-modality patterns. Jiang et al. [14] designed domain shifting (DNS) that augments modality-specific and modality-shared representation, thereby regulating the model to concentrate on consistency between modalities. Ren et al. proposed a novel implicit discriminative knowledge learning (IKDL) network to discover identity-aware salient information for aligning visible and infrared modalities. Some auxiliarybased methods [6, 48, 50] have also been developed to achieve modality alignment with the help of generated auxiliary modalities. Feng et al. [6] proposed a shape-guided diverse feature learning framework (SGIEL) that employs the body shape as the auxiliary information for modality alignment. Yu et al. [48] proposed a modality unifying network (MUN) that constructed a robust auxiliary modality by intra- and inter-modality learners.

However, all the above methods are achieved under idealized lab settings, which neglect privacy constraints and heterogeneous environments pose significant challenges in reality. More importantly, these methods require paired visible and infrared images, making privacy-preserved implementation infeasible, especially under CI, where modality information is unknown. In addition, existing works neglect the research on unseen environments, struggling to generalize effectively in real-world scenes. Therefore, we propose L2RW, offering a novel practical and privacy-preserving benchmark for bringing VI-ReID closer to real world.

## 2.2. Federated Learning

Federated Learning is a technology that enables multiple devices to collaborative learning without sharing private data. The pioneering work, FedAvg [23], averages the gradients of locally trained models and redistributes them to local clients for further training. Based on FedAvg, Li et al. [19] proposed FedProx, introducing a proximal term to ensure the convergence of the network in federated environments. Subsequently, Li et al. [18] designed MOON, which utilizes the modality representation similarity to correct the local training of clients. These methods brought new insights to the community, inspiring numerous impressive works that solve data heterogeneity [12, 28, 29, 34] and model heterogeneity [10, 22, 45] in federated learning. This progress provides solid technical support for our L2RW.

However, despite these advancements, existing federated learning methods cannot be directly applied to VI-ReID as they cannot handle specific issues such as modality incomplete, identity missing, and domain shift encountered in real-world scenarios.

## 2.3. Domain Generalization

Domain generalization (DG) uses multiple seen domains to train a model that generalizes well to unseen domains. This technology aims to address the key issue of current deep learning methods, which heavily rely on the assumption that training and testing sets are independently and identically distributed. In practice, new information often arrives in continuous portions, leading to a domain gap between newly acquired data and the original data. This gap makes it challenging for the trained model to perform well on unseen data. Numerous works have emerged to extract domaininvariant features via contrastive loss [2, 7, 9], adversarial learning [4, 16, 20], causal learning [1, 21, 33], etc. Despite their success, these methods require centralized data from source domains and assume that data within the same domain share the same distribution. These not only limit the applicability of these methods in decentralized settings but also increase the risk of privacy leakage.

Conversely, our designed EI protocol in the L2RW benchmark eliminates the need to share source domain data, addressing privacy concerns. This also shows the feasibility of privacy domain generalization in VI-ReID and shows its potential for secure and effective deployment in reality.

## 3. Methodology

In this section, we describe the proposed L2RW benchmark in detail. In brief, Sec. 3.1 will present the overview of the L2RW. Then, we elaborate on the three protocols we designed, which are grounded in real-world scenarios: camera independence (CI) in Sec. 3.2.1, entity independence (EC) and entity sharing (ES) in Sec. 3.2.2. Finally, we introduce our proposed decentralized privacy-preserved training (DPPT) in Sec. 3.3.

## 3.1. Overview

The overall pipeline of the L2RW benchmark is shown in Fig. 2(a). Let $D _ { m } = \{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N _ { m } }$ denote the m-th local private data, where $x _ { i }$ and $y _ { i }$ denote the pedestrian images and corresponding label, and $N _ { m }$ is the local data scale. The overall training set is denoted as $D = \{ D _ { 1 } , D _ { 2 } , . . . , D _ { n } \}$ . In the realistic VI-ReID, each private dataset $D _ { m }$ can represent the data collected by a single camera or the surveillance data from within a specific entity. The overall process of the L2RW can be roughly summarized into five steps. ⃝1 Local Training: Each private data is trained by a local model $\theta ^ { l }$ (client). ⃝2 Upload: Each client model weight is uploaded to the server. ⃝3 Aggregation: The uploaded local model weights are aggregated to obtain the global model weight θ<sup>g</sup>. ⃝4 Download: The global model weight θ<sup>g</sup> is downloaded by the clients for next local training. By repeating step ⃝1 to ⃝4 until convergence, execute ⃝5 Test: test seen domains or unseen domains according to the protocol.

## 3.2. Benchmark Protocols

Considering real-world conditions, we design three protocols, i.e., camera independence (CI), entity independence (EI) and entity sharing (ES), to simulate different privacyrestricted scenarios.

## 3.2.1 Protocol 1: Camera Independence (CI)

CI: all camera data remains isolated, which simulates the most strict setting where no data sharing occurs across every single device.

It solves the concerns about data privacy, ownership regulations, or security policies in reality. However, existing VI-ReID methods cannot be directly applied under CI due to the modality incomplete issue. Specifically, current VI-ReID methods require paired visible and infrared images of the same identity as input during training, which is achieved through the PK strategy. P identities with K visible and K infrared images are randomly sampled from the training set $( { \mathrm { i . e . , m i n i - b a t c h { = } 2 } } \times P \times K )$ . These methods typically use a dual-stream architecture to extract modality-specific features in the shallow layers and modality-shared features in the deeper layers, relying on modality information as a prerequisite. Consequently, these requirements make existing VI-ReID methods incompatible with the CI protocol due to the challenges posed by modality incomplete.

![](images/40d46740c0103f752db961e387a21eab801dbff1b439bb04d18b6753164de8a3.jpg)  
Figure 2. Protocols and proposed method. (a) are the protocols of camera independence (CI) and entity independence (EI) for privacy preserved VI-ReID. Another our proposed protocol entity sharing (ES) is omitted in the figure, cause its training way is the same as tha used in existing VI-ReID methods. (b) is the local training of our proposed decentralized privacy-preserved training (DPPT).

To solve these issues, we convert the two-stream structure into one-stream architecture and only sample images according to identity rather than identity and modality. Specifically, we randomly sample K images from P identities in the m-th local private data $D _ { m } ,$ , where the K images can be visible, infrared, or a mixed of both. This eliminates the reliance on the paired visible and infrared images of the same subject as the input, thereby addressing the modality incomplete. In addition, we also employ channel augmentation (CA) [43] to destroy the color information by randomly selecting one channel (R, G, or B) to replace the other channels, which reduces the cross-modality gap across clients.

## 3.2.2 Protocol 2&3: Entity Independence and Entity Sharing (EI&ES)

In real-world scenarios, it is often feasible for an entity (data controller) to access all data within its designated region or with other data entities, which is a more relaxed privacy regulation compared to CI. Therefore, based on varying levels of mutual trust and privacy agreements, we categorize entity-level privacy regulation into two protocols: Entity

Independence (EI) and Entity Sharing (ES).

EI: Data from different entities is kept isolated during training, and the model is trained using a decentralized way. Then evaluate the trained model on an unseen entity to assess its generalization ability.

EI ensures that data sharing across entities is entirely avoided, aligning with real-world privacy requirements and ensuring data security. It also guarantees that knowledge is still transferred across entities without exposing raw data, providing a practical and secure solution for VI-ReID in real-world deployments.

ES: Data from different entities is shared during training, and the model is evaluated on an unseen entity to assess its generalization ability.

In such an ES protocol, existing VI-ReID methods can be seamlessly applied. However, note that these methods’ evaluating protocols are often designed on the same dataset, where the data distribution is assumed to be consistent. Meanwhile, in our provided ES protocol, we merge several existing datasets and conduct the evaluation in a cross-domain manner to simulate real-world scenarios. To the best of our knowledge, this has not been explored and remains an open issue.

We reproduce several VI-ReID methods [14, 26, 43, 44, 49] under this protocol, which will be discussed in Sec. 4.3. Additionally, we provide a baseline using Empirical Risk Minimization (ERM) [36]. It merges data from all training domains to learn domain-invariant knowledge and can be formulated as follows:

$$
\mathcal { L } _ { E R M } = \operatorname* { m i n } _ { f , g } \frac { 1 } { \left| \mathcal { V } \right| } \sum _ { V _ { i } \in \mathcal { V } } \mathcal { R } ^ { V _ { i } } ( f , g ) ,\tag{1}
$$

where $f$ and $g$ are the encoder and classifier, respectively. $V _ { i }$ is the i-th entity, and V is a set containing all entities. The empirical risk function $\mathcal { R } ^ { V _ { i } } ( f , g )$ for a given entity $V _ { i }$ is defined by:

$$
\begin{array} { r } { \mathcal { R } ^ { V _ { i } } ( f , g ) \triangleq \mathbb { E } _ { ( x _ { j } , y _ { j } ) \sim V _ { i } } \mathcal { L } ( ( x _ { j } ; f , g ) , y _ { j } ) , } \end{array}\tag{2}
$$

where our baseline loss function $\mathcal L ( \cdot , \cdot )$ includes the identity loss and circle loss. It is worth noting that this protocol is fully compatible with existing methods designed for laboratory settings, as it does not change their training paradigm. The loss function $\mathcal L ( \cdot , \cdot )$ can be replaced with the specific loss functions employed by those methods, ensuring seamless integration with their frameworks. Our protocol provides a fair and comprehensive evaluation of the generalization capability of existing VI-ReID approaches.

## 3.3. DPPT Method

To address the challenges in CI and EI, we propose decentralized privacy-preserved training (DPPT). DPPT consists of five steps aimed at addressing the challenges encountered in these protocols that cannot be effectively resolved through existing VI-ReID methods:

We first outline the challenges encountered under CI and EI, which cannot be handled by existing VI-ReID methods:

• Domain shift: $P _ { i } ( x | y ) \neq P _ { j } ( x | y )$ . Pedestrian images on different clients will exhibit the non-iid (Independent and identically distributed) issue, where the corresponding feature distributions are different across different clients.

• Modality incomplete: For an individual client, pedestrian images can fall into three scenarios: only visible, only infrared, or both visible and infrared.

• Identity missing: $P _ { i } ( y ) \neq P _ { j } ( y )$ . The subject is hard to appear in all cameras or entities.

Therefore, we propose the DPPT, which consists of five steps that we will introduce one by one.

⃝1 Local Training: The process of client local training is shown in Fig. 2(b). The private data are fed into the encoder and get the l<sub>2</sub>-normalized feature embedding z and logits l. The identity loss $\mathcal { L } _ { i d }$ and circle loss $\mathcal { L } _ { c i r }$ are employed as the baseline loss functions (provided in the supplementary material) to ensure the learned features and logits are identity-related.

However, this training objective design cannot address the domain shift and identity missing, as the model tends to overfit the current private data in the client training and forget previously learned knowledge after download. Driven by this, we propose a memory rectification bank (MRB) that enables different clients to optimize towards the same goal. MRB. Specifically, we design a local memory $\mathcal { M } ^ { l }$ to store $l _ { 2 } .$ -normalized feature embedding z in each client. We first calculate the mean vector $c _ { k }$ that store features belonging to the same class via Eq. 3:

$$
c _ { k } ^ { m } = \frac { 1 } { \left| S _ { k } ^ { m } \right| } \sum _ { ( z _ { i } ^ { m } , y _ { i } ^ { m } ) \in S _ { k } ^ { m } } z _ { i } ^ { m } ,\tag{3}
$$

where $S _ { k } ^ { m }$ denotes the set of samples annotated with class k in the m-th client. The local memory can be denoted as $\mathcal { M } _ { m } ^ { l } = \{ c _ { 1 } ^ { m } , c _ { 2 } ^ { m } , . . . , c _ { { I } _ { m } } ^ { m } \}$ . Each center can be viewed as a representative of each identity, containing client-specific information. By aggregating the center from different clients, a representative encompassing all clients can be obtained without leaking the original data. However, the modality and number of images for each pedestrian within a camera are unknown, and the centers for different individuals are generated from varying sample numbers, which undermines fairness between identities. Additionally, images affected by occlusions or environmental interference contain lower semantic information, which can distort the center calculation. Therefore, our MRB adjusts the centers by averaging the top K closest feature embeddings to the center, effectively filtering out low-information features and providing a fairer representation for each identity. It is defined as:

$$
\hat { c } _ { k } ^ { m } = \left\{ \begin{array} { l l } { c _ { k } ^ { m } , } & { \mathrm { i f ~ } | S _ { k } ^ { m } | < K , } \\ { \displaystyle \frac { 1 } { K } \sum \operatorname* { m i n } _ { K } d ( z _ { i } ^ { m } , c _ { k } ^ { m } ) , } & { \mathrm { i f ~ } | S _ { k } ^ { m } | \geq K , } \end{array} \right.\tag{4}
$$

where $d ( \cdot , \cdot )$ is the Euclidean distance and the local memory is updated as $\mathcal { M } _ { m } ^ { l } = \{ \hat { c } _ { 1 } ^ { m } , \hat { c } _ { 2 } ^ { m } , . . . , \hat { c } _ { I _ { m } } ^ { m } \} , I _ { m }$ is the identity number in client m. The local memory is uploaded to the server and aggregated through a global memory $\mathcal { M } ^ { g }$ , as defined below:

$$
\mathcal { M } ^ { g } = \left\{ c _ { i } ^ { g } = \frac { \sum _ { m = 1 } ^ { n } \mathbb { 1 } _ { \{ \hat { c } _ { i } ^ { m } \neq 0 \} } \cdot \hat { c } _ { i } ^ { m } } { \sum _ { m = 1 } ^ { n } \mathbb { 1 } _ { \{ \hat { c } _ { i } ^ { m } \neq 0 \} } } \Bigg | i = 1 , 2 , \dots , I \right\} ,\tag{5}
$$

where n is the number of clients. I is the total identity number, and $I _ { m } \leq I$ , thereby handling the identity missing issue. 1 denotes the indicator function. The global memory contains information from different domains while ensuring no data leakage. In the next round of local training, we use the memory rectification bank loss ${ \mathcal { L } } _ { m r b }$ to guide the optimization of the locally updated memory towards the global memory, thereby relieving domain shift. The $\mathcal { L } _ { m r b }$ is defined as follows:

$$
\mathcal { L } _ { m r b } = \frac { 1 } { N _ { m } } \sum _ { i = 1 } ^ { N _ { m } } ( 1 - \frac { \langle z _ { i } ^ { m } \rangle ^ { ( t ) } \cdot \langle c _ { y _ { i } } ^ { g } \rangle ^ { ( t - 1 ) } } { | | \langle z _ { i } ^ { m } \rangle ^ { ( t ) } | | \times | | \langle c _ { y _ { i } } ^ { g } \rangle ^ { ( t - 1 ) } | | } ) ,\tag{6}
$$

where $\langle z _ { i } ^ { m } \rangle ^ { ( t ) }$ is the i-th l2-normalized feature embedding in the t-th epoch on the m-th client. $\langle c _ { u _ { i } } ^ { g } \rangle ^ { ( t - 1 ) }$ is the global center of identity $y _ { i }$ in the last epoch $t - 1$ , which contains the knowledge from the client m and the other clients without leaking raw data. $\mathcal { L } _ { m r b }$ pulls together the local memory and the global memory, addressing the domain shift and identity missing. The total training loss of the $m - t h$ client can be written as:

Algorithm 1: DPPT   
Input : The number of clients $n ,$ initial global   
model parameters $\theta ^ { g }$ , private local model   
parameters $\{ \theta _ { 1 } ^ { l } , \theta _ { 2 } ^ { l } , . . . , \overline { { { \theta _ { n } ^ { l } } } } \}$ , local private   
datasets $\{ D _ { 1 } , D _ { 2 } , . . . , D _ { n } \}$ , and the number   
of training epochs E.   
Output: The global model parameters $\theta ^ { g }$ for test.   
for $e = 1$ to E do   
for $m = 1 , 2 , \ldots , n$ do   
L $\theta _ { m } ^ { l } , \mathcal { M } _ { m } ^ { l }$ ← ClientTraining $( \theta ^ { g } , D _ { m } , { \mathcal { M } } ^ { g } ) ;$   
$/ /$ Upload and aggregate the   
local models   
$\begin{array} { r } { \theta ^ { g }  \sum _ { i = 1 } ^ { n } \frac { | D _ { i } | } { \sum _ { j = 1 } ^ { n } | D _ { j } | } \theta _ { i } ^ { l } } \end{array}$ ;   
$/ /$ Upload and aggregate the   
local memory banks   
$\begin{array} { r } { \mathcal { M } ^ { g }  \{ c _ { i } ^ { g } = \frac { \sum _ { m = 1 } ^ { n } \mathbb { 1 } _ { \{ \hat { c } _ { i } ^ { m } \neq 0 \} } \cdot \hat { c } _ { i } ^ { m } } { \sum _ { m = 1 } ^ { n } \mathbb { 1 } _ { \{ \hat { c } _ { i } ^ { m } \neq 0 \} } } | i = 1 , 2 , . . . , I \} ; } \end{array}$   
ClientTraining $( \theta ^ { g } , D _ { m } , { \mathcal { M } } ^ { g } ) \colon$   
$\theta _ { m } ^ { l }  \theta ^ { g } \ / /$ Download the global model   
to the local model   
for $( x _ { i } , y _ { i } ) \in D _ { m }$ do   
$z _ { i } = f _ { m } ( x _ { i } ) , l _ { i } = g _ { m } ( z _ { i } ) ;$   
$\mathcal { L } _ { c i r }  ( z _ { i } , y _ { i } ) , \mathcal { L } _ { i d }  ( l _ { i } , y _ { i } )$   
$\mathcal { M } _ { m } ^ { l } \gets z _ { i } \mathrm { ~ v i a ~ E q . } ( 3 , 4 ) ;$   
$\mathcal { L } _ { m r b } \gets ( \mathcal { M } _ { m } ^ { l } , \mathcal { M } ^ { g } )$ via Eq.(6);   
$\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { i d } + \mathcal { L } _ { c i r } + \lambda \mathcal { L } _ { m r b } ; } \end{array}$   
$\theta _ { m } ^ { l } \gets \theta _ { m } ^ { l } - \eta \nabla { \mathcal { L } } ;$   
return $\theta _ { m } ^ { l } , \mathcal { M } _ { m } ^ { l } ;$

$$
\mathcal { L } _ { a l l } = \mathcal { L } _ { i d } + \mathcal { L } _ { c i r } + \lambda \mathcal { L } _ { m r b } ,\tag{7}
$$

where $\lambda$ is the loss balance factor.

⃝2 Upload: The client model parameters $\theta ^ { l }$ are upload to the server, denoted by $\{ \theta _ { 1 } ^ { l } , . . . , \theta _ { n } ^ { l } \}$

⃝3 Aggregation: The uploaded local model parameters are aggregated to obtain the global model parameters $\theta ^ { g }$ :

$$
\theta ^ { g } = \sum _ { i = 1 } ^ { n } \frac { N _ { i } } { N _ { t o t a l } } \theta _ { i } ^ { l } ,\tag{8}
$$

where $\scriptstyle N _ { t o t a l } = \sum _ { i = 1 } ^ { n } N _ { i }$ is the total number of samples across all clients.

⃝4 Download: The global model parameters $\theta ^ { g }$ is downloaded by the clients for next local training.

⃝5 Test: After training, test seen domains or unseen domains according to the CI or EI protocols. The overall process is shown in Algorithm 1.

## 4. Experiment

## 4.1. Experimental Settings

Datasets. We conduct experiments on three widely-used VI-ReID datasets to evaluate our proposed method, i.e., SYSU-MM01 [38], RegDB [24], and LLCM [49].

SYSU-MM01 contains 44,754 images of 491 identities captured by 4 visible and 2 infrared cameras, including 29,033 visible images and 15,715 infrared images. RegDB contains 412 identities collected by 1 visible and 1 infrared camera. Each identity has 10 visible and 10 infrared images. LLCM contains 46,767 images of 1,064 identities captured by 9 visible and infrared cameras, with the training and testing sets split at an approximate ratio of 2:1.

Evaluation Protocols. Following existing VI-ReID settings [49], we adopt the rank-k matching accuracy, mean Average Precision (mAP), and mean Inverse Negative Precision (mINP) [44] as evaluation metrics. The protocol in terms of CI is defined in Sec. 3.2.1, and the protocols of EI and ES are defined in Sec. 3.2.2. Following [14, 38, 43, 44, 49], all datasets are used in infrared-tovisible mode for clarity, where infrared images serve as the query set, and visible images as the gallery. All the reported results are repeated ten timesfor afair comparison.

## 4.2. Implementation Details

We adopt the ResNet-50 [8] pretrained on the ImageNet-1k [5] as our backbone. During the training state, all the input images are resized to 288 × 144. The batch-size is set to 64, where randomly sampling 8 identities with 8 images. For the optimization, the Stochastic Gradient Descent (SGD) optimizer is adopted, with a weight decay of $5 e ^ { - 4 }$ and momentum of 0.9. The initial learning rate is set to 0.2, and the OneCycleLR scheduler [32] is adopted. The epochs for CI and EC experiments are set to 50 and 30, respectively. The FedAvg [23] is employed as our default. The TOP K = 4 and λ = 1 are decided by ablations on settings.

For CI, each client model is trained using data from a single camera within the training set of a given dataset. The global model is then evaluated on the testing set of the same dataset. For EI, we have two client models. Two datasets (entities) are used to train the two client models, respectively, and another new entity is used to test the global model. For ES, which is similar to EI, two entities are merged to train a model, and the model is tested on another new entity, which is not learned during training.

## 4.3. Qualitative Analysis

Evaluation on the CI Protocol. The results are reported on Tab. 1. It is noted that existing VI-ReID methods require paired visible and infrared images as input, making them unsuitable for addressing the modality incomplete challenge under the CI protocol. Therefore, we report our results using four classic federated learning algorithms: Fed-Prox [19], Fednova [37], Moon [18], and FedAvg [23], each adopting different aggregation strategies. For these methods, the setting of local training is the same, conducting ResNet-50 supervised by $\mathcal { L } _ { i d }$ and $\mathcal { L } _ { c i r }$ . The comparison methods after + only modify local training stage, with all other steps remaining consistent. The compared method only changes the local training stage. It is obvious that our proposed DPPT achieves consistent improvements across the board. Additionally, we modify two VI-ReID methods, i.e., AGW [44] and DNS [14], by converting its twostream architecture into a single-stream structure, adopting our proposed sampling approach, and removing its modality information (denoted by AGW<sup>†</sup> and DNS<sup>†</sup>). They remain a significant performance gap compared with our DPPT.

Table 1. Evaluation of our DPPT under CI protocol. Note that existing VI-ReID methods cannot directly be applied to the CI protocol as theirframeworks are designedfor data-shared learning. Thus, we implemented four classic federated learning algorithms as baselines to verify the efficacy of our method: FedProx [19], Fednova [37], Moon [18], and FedAvg [23]. Evaluating metrics rank-1(%), rank-10(%), mAP(%), and mINP(%) are reported. AGW<sup>†</sup> and DNS<sup>†</sup> are the reproduced VI-ReID method that removes the modality information.
<table><tr><td rowspan="2">Methods</td><td colspan="4">SYSU-MM01 [38]</td><td colspan="4">RegDB [24]</td><td colspan="4">LLCM [49]</td></tr><tr><td>r=1 ↑</td><td>r=10 ↑</td><td>mAP↑</td><td>mINP ↑</td><td>r=1 ↑</td><td>r=10↑</td><td>mAP ↑</td><td>mINP ↑</td><td>r=1 ↑</td><td>r=10↑</td><td>mAP↑</td><td>mINP ↑</td></tr><tr><td>FedProx [19]</td><td>25.90</td><td>68.83</td><td>27.48</td><td>17.18</td><td>25.62</td><td>44.19</td><td>25.96</td><td>17.80</td><td>23.72</td><td>53.95</td><td>30.59</td><td>27.69</td></tr><tr><td>+AGW† [44]</td><td>21.50</td><td>62.59</td><td>23.07</td><td>13.89</td><td>20.82</td><td>38.94</td><td>21.57</td><td>14.35</td><td>24.65</td><td>55.98</td><td>31.62</td><td>28.45</td></tr><tr><td>+DNS† [14]</td><td>36.11</td><td>78.29</td><td>35.22</td><td>22.02</td><td>46.75</td><td>69.27</td><td>43.06</td><td>28.99</td><td>26.35</td><td>57.00</td><td>32.88</td><td>29.54</td></tr><tr><td>+DPPT (Ours)</td><td>38.16</td><td>81.54</td><td>38.15</td><td>25.17</td><td>51.33</td><td>71.43</td><td>48.93</td><td>36.15</td><td>27.46</td><td>59.33</td><td>34.60</td><td>31.44</td></tr><tr><td>Fednova [37]</td><td>29.15</td><td>74.00</td><td>31.34</td><td>20.77</td><td>20.50</td><td>35.00</td><td>22.15</td><td>16.41</td><td></td><td></td><td></td><td></td></tr><tr><td>+AGW† [44]</td><td>22.00</td><td>64.15</td><td>23.52</td><td>13.54</td><td>13.83</td><td>25.47</td><td>15.92</td><td>12.21</td><td>1</td><td>-</td><td>-</td><td></td></tr><tr><td>+DNS† [14]</td><td>40.79</td><td>83.14</td><td>41.01</td><td>27.87</td><td>46.70</td><td>68.15</td><td>43.40</td><td>29.92</td><td>1</td><td></td><td></td><td></td></tr><tr><td>+DPPT (Ours)</td><td>50.37</td><td>88.92</td><td>48.67</td><td>33.65</td><td>59.67</td><td>78.36</td><td>55.17</td><td>41.12</td><td></td><td></td><td></td><td></td></tr><tr><td>Moon [18]</td><td>26.88</td><td>71.65</td><td>29.54</td><td>19.55</td><td>19.96</td><td>34.19</td><td>21.54</td><td>15.93</td><td>25.02</td><td>57.58</td><td>32.34</td><td>29.25</td></tr><tr><td>+AGW† [44]</td><td>20.79</td><td>62.57</td><td>22.44</td><td>12.80</td><td>11.19</td><td>20.56</td><td>13.44</td><td>10.46</td><td>23.60</td><td>55.98</td><td>31.62</td><td>28.45</td></tr><tr><td>+DNS† [14]</td><td>38.18</td><td>81.31</td><td>38.94</td><td>26.53</td><td>43.60</td><td>65.46</td><td>40.33</td><td>27.87</td><td>29.55</td><td>60.82</td><td>36.64</td><td>33.52</td></tr><tr><td>+DPPT (Ours)</td><td>46.78</td><td>87.40</td><td>45.99</td><td>31.74</td><td>53.22</td><td>73.01</td><td>49.27</td><td>35.46</td><td>32.66</td><td>64.83</td><td>39.78</td><td>36.41</td></tr><tr><td>FedAvg [23]</td><td>27.51</td><td>72.26</td><td>29.98</td><td>19.79</td><td>19.07</td><td>32.95</td><td>21.05</td><td>15.61</td><td>26.24</td><td>59.31</td><td>33.52</td><td>30.24</td></tr><tr><td>+AGW† [44]</td><td>21.65</td><td>63.13</td><td>23.25</td><td>13.45</td><td>14.17</td><td>23.84</td><td>15.89</td><td>12.10</td><td>24.31</td><td>54.51</td><td>30.66</td><td>27.19</td></tr><tr><td>+DNS† [14]</td><td>39.60</td><td>81.96</td><td>40.09</td><td>27.64</td><td>48.48</td><td>69.76</td><td>45.30</td><td>31.51</td><td>30.79</td><td>62.29</td><td>37.81</td><td>34.66</td></tr><tr><td>+DPPT (Ours)</td><td>51.27</td><td>88.55</td><td>49.29</td><td>34.47</td><td>59.85</td><td>77.58</td><td>55.58</td><td>41.70</td><td>34.69</td><td>67.20</td><td>41.91</td><td>38.48</td></tr></table>

Table 2. Evaluation of our DPPT under EI and ES protocols. The upper part of the table is under ES, while the lower part is under EI. The underlined and bold indicate the best results for both protocols, respectively. B is the baseline using ERM with $\mathcal { L } _ { i d }$ and $\mathcal { L } _ { c i r }$ under ES, B<sup>†</sup> denote the baseline using FedAvg supervised by $\mathcal { L } _ { i d }$ and $\mathcal { L } _ { c i r }$ under EI. We use R, L, and S to denote RegDB, LLCM, and SYSU-MM01 datasets, respectively. The left of → indicates seen entities and the right is the unseen entity.
<table><tr><td rowspan="2">Methods</td><td rowspan="2">Param.</td><td rowspan="2">FLOPs</td><td colspan="4">R [24] + L [49] → S [38]</td><td colspan="4">L [49] + S [38] → R [24]</td><td colspan="4">R [24] + S [38] → L [49]</td></tr><tr><td>r=1 ↑</td><td>r=10 ↑</td><td>mAP↑</td><td>mINP ↑</td><td>r=1↑</td><td>r=10↑</td><td>mAP↑</td><td>mINP↑</td><td>r=1 ↑</td><td>r=10 ↑</td><td>mAP↑</td><td>mINP ↑</td></tr><tr><td>B</td><td>23.50</td><td>10.34</td><td>8.63</td><td>36.26</td><td>9.57</td><td>4.01</td><td>17.08</td><td>34.32</td><td>17.69</td><td>11.15</td><td>8.74</td><td>26.83</td><td>12.23</td><td>9.78</td></tr><tr><td>LBA [26]</td><td>23.55</td><td>10.36</td><td>8.09</td><td>34.63</td><td>9.55</td><td>4.29</td><td>12.01</td><td>28.65</td><td>11.69</td><td>6.01</td><td>8.38</td><td>26.80</td><td>12.20</td><td>9.90</td></tr><tr><td>AGW [44]</td><td>23.55</td><td>10.36</td><td>9.59</td><td>38.28</td><td>10.42</td><td>4.45</td><td>13.79</td><td>29.76</td><td>14.05</td><td>8.25</td><td>9.08</td><td>26.77</td><td>12.37</td><td>9.80</td></tr><tr><td>DEEN [49]</td><td>41.23</td><td>27.70</td><td>9.48</td><td>38.39</td><td>10.03</td><td>3.72</td><td>19.42</td><td>37.59</td><td>19.44</td><td>12.33</td><td>10.50</td><td>29.41</td><td>14.09</td><td>11.48</td></tr><tr><td>CAJ [43]</td><td>23.55</td><td>10.36</td><td>10.90</td><td>40.57</td><td>11.18</td><td>4.39</td><td>16.55</td><td>37.23</td><td>17.40</td><td>10.74</td><td>11.35</td><td>31.17</td><td>15.03</td><td>12.10</td></tr><tr><td>DNS [14]</td><td>25.45</td><td>10.36</td><td>11.75</td><td>42.36</td><td>11.77</td><td>4.56</td><td>18.87</td><td>37.79</td><td>18.36</td><td>10.56</td><td>10.14</td><td>28.47</td><td>13.55</td><td>10.94</td></tr><tr><td>B†</td><td>23.50</td><td>5.17</td><td>9.72</td><td>38.21</td><td>10.74</td><td>4.75</td><td>15.37</td><td>29.95</td><td>16.77</td><td>10.90</td><td>8.97</td><td>26.81</td><td>12.85</td><td>10.62</td></tr><tr><td>B†+CA</td><td>23.50</td><td>5.17</td><td>10.10</td><td>39.34</td><td>10.73</td><td>4.19</td><td>17.61</td><td>33.58</td><td>17.73</td><td>11.31</td><td>14.40</td><td>35.87</td><td>18.89</td><td>16.30</td></tr><tr><td>DPPT (Ours)</td><td>23.50</td><td>5.17</td><td>11.27</td><td>41.20</td><td>11.86</td><td>5.34</td><td>21.54</td><td>40.37</td><td>20.72</td><td>12.78</td><td>14.63</td><td>37.03</td><td>19.15</td><td>16.44</td></tr></table>

Evaluation on ES and EI Protocols. We first replicate several recent VI-ReID methods [14, 26, 43, 44, 49] to evaluate their generalization ability under the ES protocol, shown in Tab. 2. While most methods slightly outperform our baseline B (ERM with $\mathcal { L } _ { i d }$ and $\mathcal { L } _ { c i r } )$ , the differences are minimal, and overall rank-1 accuracy remains low. This suggests that current VI-ReID methods still have limited capability in handling unseen environments. We encourage researchers to explore the bottlenecks that limit the generalization ability of existing methods, bringing VI-ReID closer to real world and advancing the community. Moreover, we report our method under the EI protocol. Remarkably, our DPPT achieves comparable or even superior performance under EI compared to methods evaluated under ES. Moreover, The baseline results under the EI protocol (B<sup>†</sup>) are similar to those under the ES protocol (B), indicating that decentralized training does not significantly impact the model’s ability to generalize to unseen environments.

## 4.4. Ablation Study

Effectiveness of Each Component. To evaluate the contribution of each component, we conduct a series of ablation experiments on SYSU-MM01 [38] and LLCM [49] datasets Under CI protocol. The results are shown in Tab. 3. Baseline denotes that we employ the FedAvg [23] to train the ResNet-50 and supervised by $\mathcal { L } _ { i d }$ and $\mathcal { L } _ { c i r }$ . Compared with the grayscale, CA brings a superior improvement. This is because CA preserves original information by randomly mapping one channel to the other two, whereas grayscale discards valuable color information crucial for identity discrimination. Subsequently, our proposed $\mathcal { L } _ { m r b }$ further improves the performance. In addition, to explore the role of cosine similarity in Eq. 6, we substituted it with Euclidean distance (edu) and a combination of both (mixed), as detailed in the supplementary material. The ablation study under EI is shown in Tab. 2, showing consistent improvement by adding each component.

![](images/17e4f073abdfdec86417f9a4592b5947d20cec5e00dad96a94f470ff5b1429bc.jpg)

![](images/0d861467c2778d05ddca70996d50166bd6e3c5c38d4a6ef41de587381c8fd1ff.jpg)

![](images/b6562c746552093dda3948c12bec4bece18cbc54301fd2d48fd62cc3ebf59f77.jpg)

![](images/4979efa8761a4da4e39ab1f6598864e4b7271d58c8c40024ffaf970b6acdbfff.jpg)

(a) original  
![](images/4cb659e237fe1f1393cfb00364fecf341afcf7f57097b397d1281b2d93662d99.jpg)

![](images/cc5e42b42c5435353e93ab6e176c82f06037d264c4cbb9ebce3d9f7a3dc66745.jpg)

(b) baseline  
![](images/28a71d0c68616c0f75d2af2611838c475feb62cb3cff40c97fbf6bc47941b656.jpg)  
(c) baseline+CA

![](images/5add682358ba73a16ffd9c3a3c2cccee7ab1201c6119f48180fe38e74b05524c.jpg)  
(d) baseline+CA+MRB (Ours)  
Figure 3. t-SNE and intra-inter distances visualization of each component. The circles and triangles denote the visible and infrared modality, respectively. Different colors denote different identities.

Table 3. Ablation studies on SYSU-MM01 and LLCM datasets. The light purple is our final choice setting.
<table><tr><td rowspan="2">Settings</td><td colspan="3">SYSU-MM01 [38]</td><td colspan="3">LLCM [49]</td></tr><tr><td>r=1↑</td><td>mAP↑</td><td>mINP ↑</td><td>r=1 ↑</td><td>mAP ↑</td><td>mINP ↑</td></tr><tr><td>B</td><td>27.51</td><td>29.98</td><td>19.79</td><td>26.24</td><td>33.52</td><td>30.24</td></tr><tr><td>B+CA B+gray</td><td>39.47</td><td>39.78</td><td>26.95</td><td>30.47</td><td>37.30</td><td>34.09</td></tr><tr><td> $\mathbf { B } { + } \mathbf { C } \mathbf { A } { + } \mathcal { L } _ { m r b } ( \cos )$ </td><td>35.25 51.27</td><td>36.16 49.29</td><td>24.03 34.47</td><td>29.33 34.69</td><td>36.29 41.91</td><td>33.06 38.48</td></tr><tr><td> $\mathbf { B } { + } \mathbf { C } \mathbf { A } { + } \mathcal { L } _ { m r b } ( \mathbf { e d u } )$ </td><td>43.79</td><td>44.00</td><td>30.77</td><td>33.45</td><td>40.62</td><td>37.35</td></tr><tr><td> $ { \mathbf { B } } { + }  { \mathbf { C } }  { \mathbf { A } } { + }  { \mathcal { L } } _ { m r b } (  { \mathbf { m i x e d } } )$ </td><td>47.82</td><td>47.55</td><td>33.94</td><td>34.41</td><td>41.56</td><td>38.09</td></tr></table>

![](images/19d1ce82c4bf4a9ccd8008cce77c544eee58f50f70975511e3ec8e776f06e5b4.jpg)

Visualization. To demonstrate the effectiveness of each component, we visualize the distribution of learned features by t-SNE [35] and intra-class and inter-class distances, as shown in Fig. 3. Compared with (a) which only uses ResNet-50 pretrained on the ImageNet-1k, the baseline improves the discriminative of identity within the modality but still exhibits a significant cross-modality gap. Subsequently, the integration of CA effectively bridges this huge discrepancy, pushing away the intra-class and inter-class distances. Despite encouraging progress, the feature distribution map shows that the similarity between different identities is relatively close, making it hard to distinguish. Finally, with the help of MRB, the modality gap is further reduced. The reason can be attributed to that MRB handles the domain shift and identity missing well, thus achieving extremely smaller intra-class and giant inter-class distances, facilitating easier distinction between identity samples.

![](images/d39bc1afb0ff3e304bacb0f7f63a2615e93336d4cd5206526891eb3512a8be38.jpg)  
Figure 4. The effect of hyperparameter λ and K on SYSU-MM01.

Hyper-Parameter Ablation. The hyper-parameter ablation of λ and Top K is as shown in Fig. 4(a) and (b). The λ is set to 1.0 and the K is set to 4.

## 5. Conclusion

This paper presents L2RW, the first privacy-preserved benchmark for VI-ReID, aims at bringing VI-ReID closer to real-world applications. To simulate privacy constraints, we introduce three protocols: camera independence (CI), entity independence (EI) and entity sharing (ES). We also propose the DPPT method, which addresses challenges under both protocols without compromising data privacy. Extensive experiments validate the feasibility of privacypreserved VI-ReID and demonstrate the effectiveness of our method which achieves significant improvement under privacy protection conditions. Our results further show that decentralized training does not significantly affect generalization. We hope this work inspires future research in privacy-preserved VI-ReID and advances the field.

## 6. Acknowledgement

This work was supported by the National Natural Science Foundation of China (Grant No. U22B2062), the Research Council of Finland (former Academy of Finland) Academy Professor project EmotionAI (grants 336116, 345122, 359854), the University of Oulu & Research Council of Finland Profi 7 (grant 352788), EU HORIZON-MSCA-SE-2022 project ACMod (grant 101130271), the Finnish Doctoral Program Network in Artificial Intelligence, AI-DOC (decision number VN/3137/2024-OKM-6), and the Startup Foundation for Introducing Talent of NUIST.

## References

[1] Kartik Ahuja, Ethan Caballero, Dinghuai Zhang, Jean-Christophe Gagnon-Audet, Yoshua Bengio, Ioannis Mitliagkas, and Irina Rish. Invariance principle meets information bottleneck for out-of-distribution generalization. Advances in Neural Information Processing Systems, 34: 3438–3450, 2021. 3

[2] Aristotelis Ballas and Christos Diou. Multi-scale and multilayer contrastive learning for domain generalization. IEEE Transactions on Artificial Intelligence, 2024. 3

[3] Cuiqun Chen, Mang Ye, and Ding Jiang. Towards modalityagnostic person re-identification with descriptive query. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15128–15137, 2023. 1

[4] Aveen Dayal, Vimal KB, Linga Reddy Cenkeramaddi, C Mohan, Abhinav Kumar, and Vineeth N Balasubramanian. Madg: margin-based adversarial learning for domain generalization. Advances in Neural Information Processing Systems, 36, 2024. 3

[5] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009. 6

[6] Jiawei Feng, Ancong Wu, and Wei-Shi Zheng. Shape-erased feature learning for visible-infrared person re-identification. In CVPR, pages 22752–22761, 2023. 2

[7] Yutong Feng, Jianwen Jiang, Mingqian Tang, Rong Jin, and Yue Gao. Rethinking supervised pre-training for better downstream transferring. arXiv preprint arXiv:2110.06014, 2021. 3

[8] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 770–778, 2016. 6

[9] Jiajun Hu, Lei Qi, Jian Zhang, and Yinghuan Shi. Domain generalization via inter-domain alignment and intra-domain expansion. Pattern Recognition, 146:110029, 2024. 3

[10] Wenke Huang, Mang Ye, and Bo Du. Learn from others and be yourself in heterogeneous federated learning. In CVPR, pages 10143–10153, 2022. 2, 3

[11] Wenke Huang, Mang Ye, Zekun Shi, and Bo Du. Generalizable heterogeneous federated cross-correlation and instance

similarity learning. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2023. 2

[12] Wenke Huang, Mang Ye, Zekun Shi, He Li, and Bo Du. Rethinking federated learning with domain shift: A prototype view. In CVPR, pages 16312–16322. IEEE, 2023. 3

[13] Wenke Huang, Mang Ye, Zekun Shi, Guancheng Wan, He Li, Bo Du, and Qiang Yang. Federated learning for generalization, robustness, fairness: A survey and benchmark. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024. 2

[14] Yan Jiang, Xu Cheng, Hao Yu, Xingyu Liu, Haoyu Chen, and Guoying Zhao. Domain shifting: A generalized solution for heterogeneous cross-modality person re-identification. In European Conference on Computer Vision, pages 289–306. Springer, 2025. 2, 4, 6, 7

[15] Yan Jiang, Xu Cheng, Hao Yu, Xingyu Liu, Haoyu Chen, and Guoying Zhao. Dsaf: Dual space alignment framework for visible-infrared person re-identification. IEEE Transactions on Multimedia, 2025. 1

[16] Haoliang Li, Sinno Jialin Pan, Shiqi Wang, and Alex C Kot. Domain generalization with adversarial feature learning. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5400–5409, 2018. 3

[17] He Li, Mang Ye, Ming Zhang, and Bo Du. All in one framework for multimodal re-identification in the wild. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 17459–17469, 2024. 1

[18] Qinbin Li, Bingsheng He, and Dawn Song. Modelcontrastive federated learning. In CVPR, pages 10713– 10722, 2021. 2, 3, 7

[19] Tian Li, Anit Kumar Sahu, Manzil Zaheer, Maziar Sanjabi, Ameet Talwalkar, and Virginia Smith. Federated optimization in heterogeneous networks. Proceedings of Machine learning and systems, 2:429–450, 2020. 2, 3, 7

[20] Ya Li, Xinmei Tian, Mingming Gong, Yajing Liu, Tongliang Liu, Kun Zhang, and Dacheng Tao. Deep domain generalization via conditional invariant adversarial networks. In Proceedings ofthe European conference on computer vision (ECCV), pages 624–639, 2018. 3

[21] Divyat Mahajan, Shruti Tople, and Amit Sharma. Domain generalization using causal matching. In International conference on machine learning, pages 7313–7324. PMLR, 2021. 3

[22] Disha Makhija, Xing Han, Nhat Ho, and Joydeep Ghosh. Architecture agnostic federated learning for neural networks. In International Conference on Machine Learning, pages 14860–14870. PMLR, 2022. 3

[23] Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. Communicationefficient learning of deep networks from decentralized data. In Artificial intelligence and statistics, pages 1273–1282. PMLR, 2017. 2, 3, 6, 7

[24] Dat Tien Nguyen, Hyung Gil Hong, Ki Wan Kim, and Kang Ryoung Park. Person recognition system based on a combination of body images from visible light and thermal cameras. Sensors, 17(3):605, 2017. 6, 7

[25] Honghu Pan, Wenjie Pei, Xin Li, and Zhenyu He. Unified conditional image generation for visible-infrared person reidentification. IEEE Transactions on Information Forensics and Security, 2024. 1

[26] Hyunjong Park, Sanghoon Lee, Junghyup Lee, and Bumsub Ham. Learning by aligning: Visible-infrared person reidentification using cross-modal correspondences. In Proceedings ofthe IEEE/CVF international conference on computer vision, pages 12046–12055, 2021. 2, 4, 7

[27] Zhihao Qian, Yutian Lin, and Bo Du. Visible–infrared person re-identification via patch-mixed cross-modality learning. Pattern Recognition, 157:110873, 2025. 1

[28] Liangqiong Qu, Yuyin Zhou, Paul Pu Liang, Yingda Xia, Feifei Wang, Ehsan Adeli, Li Fei-Fei, and Daniel Rubin. Rethinking architecture design for tackling data heterogeneity in federated learning. In CVPR, pages 10061–10071, 2022. 3

[29] Zhe Qu, Xingyu Li, Rui Duan, Yao Liu, Bo Tang, and Zhuo Lu. Generalized federated learning via sharpness aware minimization. In International conference on machine learning, pages 18250–18280. PMLR, 2022. 3

[30] Kaijie Ren and Lei Zhang. Implicit discriminative knowledge learning for visible-infrared person re-identification. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 393–402, 2024. 1

[31] Jiangming Shi, Xiangbo Yin, Yeyun Chen, Yachao Zhang, Zhizhong Zhang, Yuan Xie, and Yanyun Qu. Multimemory matching for unsupervised visible-infrared person re-identification. arXiv preprint arXiv:2401.06825, 2024. 1

[32] Leslie N Smith and Nicholay Topin. Super-convergence: Very fast training of neural networks using large learning rates. In Artificial intelligence and machine learning for multi-domain operations applications, pages 369–386. SPIE, 2019. 6

[33] Shuzhou Sun, Shuaifeng Zhi, Qing Liao, Janne Heikkila,¨ and Li Liu. Unbiased scene graph generation via two-stage causal modeling. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(10):12562–12580, 2023. 3

[34] Yue Tan, Guodong Long, Lu Liu, Tianyi Zhou, Qinghua Lu, Jing Jiang, and Chengqi Zhang. Fedproto: Federated prototype learning across heterogeneous clients. In AAAI, pages 8432–8440, 2022. 3

[35] Laurens Van der Maaten and Geoffrey Hinton. Visualizing data using t-sne. Journal of machine learning research, 9 (11), 2008. 8

[36] Vladimir Vapnik. Principles of risk minimization for learning theory. Advances in neural information processing systems, 4, 1991. 4

[37] Jianyu Wang, Qinghua Liu, Hao Liang, Gauri Joshi, and H Vincent Poor. Tackling the objective inconsistency problem in heterogeneous federated optimization. Advances in neural information processing systems, 33:7611–7623, 2020. 2, 7

[38] Ancong Wu, Wei-Shi Zheng, Hong-Xing Yu, Shaogang Gong, and Jianhuang Lai. Rgb-infrared cross-modality person re-identification. In Proceedings of the IEEE international conference on computer vision, pages 5380–5389, 2017. 1, 6, 7, 8

[39] Jianbing Wu, Hong Liu, Yuxin Su, Wei Shi, and Hao Tang. Learning concordant attention via target-aware alignment for visible-infrared person re-identification. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 11122–11131, 2023. 1

[40] Zesen Wu and Mang Ye. Unsupervised visible-infrared person re-identification via progressive graph matching and alternate learning. In CVPR, pages 9548–9558, 2023. 1

[41] Bin Yang, Jun Chen, and Mang Ye. Towards grand unified representation learning for unsupervised visible-infrared person re-identification. In ICCV, pages 11069–11079, 2023. 1

[42] Bin Yang, Jun Chen, and Mang Ye. Shallow-deep collaborative learning for unsupervised visible-infrared person reidentification. In CVPR, pages 16870–16879, 2024. 1

[43] Mang Ye, Weijian Ruan, Bo Du, and Mike Zheng Shou. Channel augmented joint learning for visible-infrared recognition. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 13567–13576, 2021. 2, 4, 6, 7

[44] Mang Ye, Jianbing Shen, Gaojie Lin, Tao Xiang, Ling Shao, and Steven CH Hoi. Deep learning for person reidentification: A survey and outlook. IEEE transactions on pattern analysis and machine intelligence, 44(6):2872–2893, 2021. 2, 4, 6, 7

[45] Mang Ye, Xiuwen Fang, Bo Du, Pong C Yuen, and Dacheng Tao. Heterogeneous federated learning: State-of-the-art and research challenges. ACM Computing Surveys, 56(3):1–44, 2023. 3

[46] Mang Ye, Zesen Wu, Cuiqun Chen, and Bo Du. Channel augmentation for visible-infrared re-identification. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2023. 1

[47] Hao Yu, Xu Cheng, and Wei Peng. Toplight: Lightweight neural networks with task-oriented pretraining for visibleinfrared recognition. In CVPR, pages 3541–3550, 2023. 1

[48] Hao Yu, Xu Cheng, Wei Peng, Weihao Liu, and Guoying Zhao. Modality unifying network for visible-infrared person re-identification. In ICCV, pages 11185–11195, 2023. 1, 2, 3

[49] Yukang Zhang and Hanzi Wang. Diverse embedding expansion network and low-light cross-modality benchmark for visible-infrared person re-identification. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2153–2162, 2023. 1, 2, 4, 6, 7, 8

[50] Yukang Zhang, Yan Yan, Yang Lu, and Hanzi Wang. Towards a unified middle modality learning for visible-infrared person re-identification. In Proceedings of the 29th ACM international conference on multimedia, pages 788–796, 2021. 2

[51] Yukang Zhang, Yang Lu, Yan Yan, Hanzi Wang, and Xuelong Li. Frequency domain nuances mining for visible-infrared person re-identification. arXiv preprint arXiv:2401.02162, 2024. 1

[52] Jianqing Zhu, Hanxiao Wu, Yutao Chen, Heng Xu, Yuqing Fu, Huanqiang Zeng, Liu Liu, and Zhen Lei. Cross-modal group-relation optimization for visible–infrared person reidentification. Neural Networks, 179:106576, 2024. 1