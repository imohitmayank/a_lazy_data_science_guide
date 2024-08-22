## Introduction

[Refer LLama3.1 paper]

Large Language Models (LLMs) are a class of deep learning models which are designed to model a language by creating probability distribution over the sequence of tokens *(that can be words)*. They are often used for tasks like text generation, completion, summarisation, question answering, etc. The recent surge in such models is due to their ability to perform these tasks at a high level of accuracy. They have been trained on a large corpus of text and have learned the structure of language, which is why they can predict the next word with a high accuracy.

The recent advancements in LLMs have become possible due to the success of transformer models which introduced the concept of self-attention mechanism. This mechanism allows the model to capture the relationships between different parts of the input sequence. This has led to significant improvement in the performance of LLMs. They have shown great success in a variety of tasks like text generation, machine translation, and question answering.

The recent surge in the demand for LLMs has led to the development of several large language models like GPT-3, LLaMA-1 and LLaMA-2, etc. These models have been trained on a large corpus of text and have learned the structure of language. They have become a powerful tool for natural language processing tasks and have shown great success in a variety of tasks like text generation, machine translation, and question answering.

The recent advancements in LLMs have led to the development of several large language models like GPT-3, LLaMA-1 and LLaMA-2, etc. These models have been trained on a large corpus of text and have learned the structure of language. They have become a powerful tool for natural language processing tasks and have shown great success in a variety of tasks like text generation, machine translation, and question answering.

## Stages of Training LLMs

The process of training a Large Language Model (LLM) consists of two distinct stages. 

During the pre-training phase, the model is trained on a large corpus of text to predict the next word in a sequence. This learning process enables the model to understand the structure of language and acquire knowledge about the world. The pre-training stage consists of four main components: text data curation, model architecture development, efficient pre-training techniques, and the creation of a pre-training recipe.

After the pre-training phase, the model undergoes post-training. This stage focuses on aligning the model with human feedback. The model is fine-tuned with instruction tuning data and Direct Preference Optimization (DPO) to improve its ability to follow instructions. Additionally, new capabilities like tool-use are integrated, leading to enhancements in areas like coding and reasoning. Lastly, safety mitigations are put in place.

To summarize, the training of a Large Language Model involves two stages: pre-training and post-training. Pre-training focuses on learning the structure of language from a large corpus of text, while post-training aims to align the model with human feedback and integrate new capabilities.

- **Language model pre-training.** We start by converting a large, multilingual text corpus to discrete tokens and pre-training a large language model (LLM) on the resulting data to perform next-token prediction. In the language model pre-training stage, the model learns the structure of language and obtains large amounts of knowledge about the world from the text it is “reading”.  This is further divided into,
  
  - (1) the curation and filtering of a large-scale training corpus,
  - (2) the development of model architecture and corresponding scaling laws for determining model size,
  - (3) the development of techniques for eﬃcient pre-training at large scale, and
  - (4) the development of a pre-training recipe.
  
- **Language model post-training.** The pre-trained language model has a rich understanding of language but it does not yet follow instructions or behave in the way we would expect an assistant to. We align the model with human feedback in several rounds, each of which involves supervised finetuning (SFT) on instruction tuning data and Direct Preference Optimization (DPO; Rafailov et al., 2024). At this post-training2 stage, we also integrate new capabilities, such as tool-use, and observe strong improvements in other areas, such as coding and reasoning. Finally, safety mitigations are also incorporated.

## Pre-Training

The pre-training stage is divided into four main components: data curation, model architecture design, efficient pre-training techniques, and the creation of a pre-training recipe.

### Data Curation

To train a Large Language Model (LLM), we first need to collect a large amount of text data.

The data collection for language model pre-training involves curation and filtering of web data to create a diverse and high-quality dataset. This dataset is then processed and optimized to extract high-quality text. The dataset is filtered to remove domains that contain personal information, harmful content, or adult content. Additionally, the dataset is de-duplicated to remove duplicate lines and documents. Heuristics and model-based quality filters are used to remove low-quality documents and excessive repetitions.

The data mix for pre-training is determined by classifying the types of information in the web data and using scaling law experiments to determine the best data mix. The data mix contains roughly 50% of tokens corresponding to general knowledge, 25% of mathematical and reasoning tokens, 17% code tokens, and 8% multilingual tokens.

Annealing is used to improve the performance of pre-trained models on key benchmarks by upsampling high-quality code and mathematical data. The efficacy of annealing is evaluated on the GSM8k (Cobbe et al., 2021) and MATH (Hendrycks et al., 2021b) training sets in annealing.

The data quality of small domain-specific datasets is judged using annealing, which enables us to assess the true few-shot learning capabilities and out-of-domain generalization of Llama 3.

In summary, language model pre-training involves curation and filtering of text data, developing a model architecture and scaling laws, creating efficient pre-training techniques, and creating a pre-training recipe. The post-training stage involves aligning the model with human feedback, integrating new capabilities, and mitigating safety risks. The data collection stage involves curation and optimization of web data, de-duplication, heuristics, and model-based quality filters. The data mix is determined by classifying the types of information and using scaling law experiments. Annealing is used to improve the performance of pre-trained models. The data quality of small domain-specific datasets is judged using annealing.

- Pre-Training Data (p.4)
    - Web Data Curation (p.4)
    - Determining the Data Mix (p.6)
    - Annealing Data (p.6)
- Model Architecture (p.6)
    - Scaling Laws (p.7)
- Infrastructure, Scaling, and Effi... (p.8)
    - Training Infrastructure (p.8)
    - Parallelism for Model Scaling (p.10)
    - Collective Communication (p.12)
    - Reliability and Operational C... (p.12)
- Training Recipe (p.14)
    - Initial Pre-Training (p.14)
    - Long Context Pre-Training (p.14)
    - Annealing (p.14)

## Post-Training

- Modeling (p.15)
    - Chat Dialog Format (p.15)
    - Reward Modeling (p.16)
    - Supervised Finetuning (p.16)
    - Direct Preference Optimizati... (p.16)
    - Model Averaging (p.16)
    - Iterative Rounds (p.16)
- Post-training Data (p.17)
    - Preference Data (p.17)
    - SFT Data (p.17)
    - Data Processing and Quality... (p.18)

**Capabilities** (p.19)

- Code (p.19)
- Multilinguality (p.22)
- Math and Reasoning (p.23)
- Long Context (p.24)
- Tool Use (p.24)
- Factuality (p.26)
- Steerability (p.27)