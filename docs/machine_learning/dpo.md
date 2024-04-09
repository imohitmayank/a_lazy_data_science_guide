## Introduction

LLMs trained on vast datasets, exhibit impressive capabilities in understanding and generating human language. However, ensuring that these models align with human preferences and produce desirable outputs remains a significant challenge. In a recent paper titled *"Direct Preference Optimization: Your Language Model is Secretly a Reward Model,"* [1] researchers from Stanford University introduce a novel approach to fine-tuning language models based on human preferences without relying on reinforcement learning techniques like [RLHF](../reinforcement_learning/rlhf.md)

While RLHF is still one of the primary methods for training language models to align with human preferences, it has several limitations, including high computational costs *(you need multiple copies of the model for finetuning)*, complex reward modeling *(new to train an accurate reward model)*, and challenges in reward shaping *(RL is quite infamous for this problem)*. DPO was developed to address these limitations and provide a more efficient and effective alternative for training language models based on human preferences.

## The Technique

The DPO pipeline consists of two steps, (1) SFT finetuning of the language model on the required dataset *(which is mostly instruction dataset)*, and then (2) DPO fine-tuning on the preference dataset. The preference dataset consists of *(input, output)* generation examples pairs, where each output has an associated score that signifies its preference in comparison with other output for the same input. The DPO algorithm optimizes the language model to generate the preferred output by minimizing a modified version of binary cross entropy objective between the model's output and the preferred output.

<figure markdown> 
    ![](../imgs/ml_dpo_1.png)
    <figcaption>Source: [1]</figcaption>
</figure>

DPO algorithm implicitly optimizes the same objectives as RLHF but is simpler to implement and train. DPO does this by incorporating two factors into its policy, 

1. DPO relies on a preference model like [Bradley-Terry model](./interview_questions.md#what-is-bradley-terry-model-and-how-is-it-used-in-machine-learning), to express the human preference probability in terms of the optimal policy and the reference policy. This allows for the optimization of the language model based on human preferences without the need for an explicit reward model. 
2. It also has a KL divergence constraint that ensures the policy *(trained model)* remains close to the reference policy *(original model)*. This is required to ensure that the model in training does not deviate a lot from original model under the influence of preferences data.

The DPO policy objective is derived from the optimal solution of the KL-constrained reward maximization objective in the context of reinforcement learning. The DPO policy objective is formulated as below:

$$ 
L_{DPO}(π_θ; π_{ref}) = -E_{(x, y_w, y_l)∼D} \left[ \log \sigma \left( \beta \log \frac{πθ(y_w | x)}{πref(y_w | x)} - \beta \log \frac{πθ(y_l | x)}{πref(y_l | x)} \right) \right] 
$$

Here, $π_θ$ represents the language model, $π_{ref}$ represents the reference model, $D$ represents the preferences dataset, in which $x$ represents the input, $y_w$ and $y_l$ represent the winning (preferred) and losing (undesired) output, respectively. The objective is derived from the optimal solution of the KL-constrained reward maximization objective under the Bradley-Terry preference model, which depends on the difference of rewards between two completions. The DPO policy objective is a maximum likelihood objective that enables the optimization of the language model based on human preferences, without the need for an explicit reward model.


--- 
# TODO

### Key Contributions and Findings
The main contribution of DPO lies in its ability to train language models from preferences effectively. Through a series of experiments, the researchers demonstrate that DPO is as effective as existing methods, including reinforcement learning-based approaches, in tasks such as sentiment modulation, summarization, and dialogue. Importantly, DPO outperforms traditional methods in controlling sentiment and response quality while being computationally lightweight and easier to implement.

### Generalization and Performance
One of the notable strengths of DPO is its ability to generalize well to new input distributions. The algorithm's performance remains robust even when faced with distribution shifts, outperforming traditional reinforcement learning methods like proximal policy optimization (PPO) in various scenarios. Additionally, DPO shows promising results in tasks such as text summarization and dialogue generation, indicating its versatility and effectiveness across different text generation applications.

### Conclusion
In conclusion, the Direct Preference Optimization (DPO) algorithm offers a promising avenue for fine-tuning language models based on human preferences without the need for reinforcement learning. By simplifying the preference learning process and enhancing the model's controllability, DPO represents a significant advancement in the field of natural language processing. The research findings underscore the effectiveness and efficiency of DPO in training language models to align with human preferences, paving the way for safer, more performant, and controllable AI systems.

## Reference
[1] [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)