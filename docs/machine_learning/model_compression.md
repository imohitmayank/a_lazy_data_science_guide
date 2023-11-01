## Introduction

- Large Deep Learning models provide enhanced accuracy for majority of tasks, but they compromise with speed. This could be a deal breaker for industrial use case, where sometimes hundreds or thousands of inferences needs to be made in short amount of time. In fact we have two problems to solve, and let's take a detour and explore what can be done from the AIOps perspective *(not involving Data Scientists at all)*,

  1. **Slower model** *leading to high inference time*. To solve this issue we can,
    - **Use more powerful compute:** just use newer and faster machine! 😎 
    - **Use GPU or TPU compute:** use GPU computes *(if only using CPU till now)* to enhance Deep learning models performance due to faster matrix juggling by GPUs.
  2. **High demand** *leading to a lots of inference request in short amount of time.* To solve this issue we can,
    - **Multi Threading**: if your system performs some I/O calls or DB calls or 3rd party API polling, multi threading could be the way to improve inference speed. Basically, while one thread is busy perfoming asyncronous task, another task can take over and start running. 
    - **Multi Processing**: we can utilize multiple cores of a compute to create multiple processes, each working independently of each other. 
    !!! Note
        Multi-processing is a memory (or GPU) consuming task. This is because if you create 4 workers (4 processes), OS will create 4 copies of your model! For a 1GB model, 4GB will be consumed just to keep the system up and running.
    - **Multi Instances**: if you are using K8, just create multiple nodes of the code. It's like creating multiple copies of the code and running them independently on different computes. 
    !!! Note
        Multi Instances is a costly task, as each new replication needs a new compute which will increase the cost.

- All of the above proposed solutions have one major flaw -- it will cost you money 💰 and that too on a recurrent basis if you plan to keep the model running for months. On top of it, all of these are not "Data Scientist" way of tacking the problem 😊 Can we do something more scientific? 🤔 
- Enter **model compression** techniques, where the intuition is to reduce the size of the model which will inherently increase the speed of the inference *(or training)*. And do this with minimal compromise on the accuracy!

!!! Note
    Please understand that the main assumption before applying model compression techniques is that to create a good smaller model, it is more optimal *(in terms of either cost, time, data or accuracy)* to utilize an existing good bigger model, rather than training the smaller model from stratch on the same data. 

    From another perspective, people might argue that if they have sufficient compute and data, why not just train a smaller model from stratch? A counter argument could be that for the same data, a bigger model will most likely provide better accuracy, so using a bigger model to distil a smaller model might induce some additional knowledge from the teacher model or at the least drastically reduce the learning time.

## Types of Model compression

- At a high level, there are following types of model compression. We will go through them one by one.
  - **Knowledge Distillation**: in these methods, we distil the learned information (or knowledge) from one neural network model *(generally larger)* to another model *(generally smaller)*.
  - **Parameter Pruning and Sharing**: in these methods, we remove the non-essential parameters from a neural network with minimal to no effect on the overall performance. 

## Knowledge Distillation

<figure markdown> 
    ![](../imgs/dl_mc_kd_cover.png){ width="500" }
    <figcaption>Teacher with a student. *Source: DallE*</figcaption>
</figure>

- The main idea in Knowledge Distillation (KD) is to make a smaller model *(student model)* mimic the larger model *(teacher model)*. This could lead to student model having competitive or sometimes even superior performance than teacher model.
- For example consider this scenario -- suppose you have a medium sized (>300MBs) model that is quite accurate (>80%) but slow (700-800ms per inference on CPU). While we can apply any of the above non-scientific solutions, it will not be cost effective. With KD if we can distil the model's knowledge to a much smaller one (say, ~100MBs) with minimum compromise on accuracy (>78%) we can greatly reduce the inference time (2x to 3x) and make the complete solution scalable. 

    | Metric | Teacher Model | Student Model |
    | ------ | ------ | ----- |
    | Size class | Medium | Small |
    | Size | >300MBs | ~100MBs |
    | Accuracy | >80% | >78% |
    | Inference time | 800ms | ~300ms |

!!! Note
    Remember there will always be a compromise between speed and accuracy. As you decrease the size of model *(by model compression techniques)* the accuracy will also drop. The science here is to make sure that decrease is not too drastic. And this relationship must be kept in mind before making the choice to do KD. For example, in certain use case related to medical domain, accuracy is of upmost importance and developers should be aware of the risks.

- To better understand the different ways of performing KD, we should understand two things - (1) knowledge (2) distillation schemes. Let's understand them one by one.

### Types of Knowledge

- Before we start to distil knowledge, we should first define and understand what is meant by "knowledge" in context of neural network and K -- is it the prediction OR the parameters learned OR activations for one input OR maybe multiple inputs? Once we know this, we can try to teach student model to mimic that particular characterisitics of the teacher model. Based on this intuition, let's categorize knowledge and respective distillation techniques.

<figure markdown> 
    ![](../imgs/dl_mc_dl_kd_knowledges.png)
    <figcaption>Different types of knowledge in Deep teacher network. *[1]*</figcaption>
</figure>

#### Response Based Knowledge

- Here, we define the final layer output of the teacher model as the knowledge, so the intuition is to train a student model that will mimic the final prediction of the teacher model. For example, for a cat vs dog image classification model, if the teacher model classifies an image as 'Cat', a good student model should also learn from the same classification and vice-versa.
- Now the final predictions could also be of multiple types - logits *(the model output)*, soft targets *(the class probabilities)* or hard targets *(the class enums)* ([refer](deep_learning_terms.md#logits-soft-and-hard-targets)). Developers can select any of the prediction types, but **usually soft targets are preferred**, as they contain more information than hard target and are not as specific or architecture dependent as logits.
- Technically, we first predict the responses of student and teacher model on a sample, then compute the distillation loss on the difference between the logits *(or other prediction)* values generated by both. The distillation loss is given by $L_R(z_t, z_s)$ where $L_R()$ denotes the divergence loss and $z_t$ and $z_s$ denotes logits of teacher and student models respectively. In case of soft targets, we compute the probability of logits of one class wrt to other classes using softmax function, 

  $$p(z_i,T) = \frac{exp(z_i/T)}{\sum{}_j exp(z_j/T)}$$

  and the distillation loss becomes $L_R(p(z_t,T),p(z_s,T))$ which often employs Kullback-Leibler divergence loss.
<!-- - In terms of final loss, researchers have also expermiented with complimenting distillation loss with student loss which is nothing but cross-entropy loss between the ground truth label and soft logits of the student model. -->

#### Feature Based Knowledge

- Here, instead of the repsonse, we directly match the feature activations of the teacher and student model.

$$LF=Φt(ft(x)),Φs(fs(x))$$

#### Relation Based Knowledge



### Distillation schemas

#### Online Distillation
#### Offline Distillation

## Parameter Pruning [TODO]

## References

[1] [J Gou et al. - Knowledge Distillation: A Survey](https://arxiv.org/abs/2006.05525)