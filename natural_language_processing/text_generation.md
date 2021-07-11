Text generation
========================

## Introduction

- Text generation is an interesting task in NLP, where the intention is to generate text when provided with some prompt as input. Usually, we apply some form of Sequence-to-Sequence model (Seq2Seq) for this task. They are called language models, as they can be used to predict the next word based on the previous sentences. The recent surge in interest in this field is due to two main reasons, (1) the availability of several high performance pre-trained models, and (2) it's very easy to transform a large variety of NLP based tasks into the text-in text-out type of problem. Beacuse of this, it becomes easy to solve several such problems using a single language model.
- The Seq2Seq model consists of two main modules - 
    - **Encoder**: which takes text as input and encodes it to a compact vector representation, 
    - **Decoder**: which takes the compact vector representation as input and generates text as output. 
- Conventionally, the models were trained from scratch for a specific task, which requires a lot of training data. Recently, as NLP has deviated more towards fine-tuning methodology, we have a number of pre-trained models which either out-of-the-box work very well or can be fine-tuned for the specific task.
- Some of the conventional models were RNN, GRU and, LSTM. Whereas recent pre-trained models include Transformers, GPT-{1, 2, 3}, GPT-{Neo, J}, T5, etc.
- Text generation task is a very specific but also a very generic task because we can formulate a lot of NLP tasks in the form of text generation. For example, *(not a complete list)*
    - **Language translation** English to Hindi translation, or any language for that matter, is just a text in and text out task
    - **Summarization** takes the complete text dump as input and generates crisp informative sentences.
    - **Question answering** question is taken as input and answer is the output. We can even include some context as input on closed-domain QA tasks.
    - **Sentiment analysis** is a classification task where we can provide the text as input and train the model to generate sentiment tag as output
    - **Rating prediction** where we have to rate the writing style, kind of regression problem, but still can be formulated as text in number out
- As obvious from the examples mentioned above, it is possible to formulate a lot of problems as a text-in-text-out task, and it could even expand to classification and regression types of tasks. Some example tasks that can be performed using T5 is shown below, 

```{figure} /imgs/t5_example.gif
---
height: 200px
---
T5 text-to-text framework examples. See: [Google Blog](https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html) {cite}`raffel2020exploring`
```

<!-- ## Recent language models

**TODO**

### Prompt engineering

**TODO**

### Text generation strategies

**TODO** -->
## Analysis

### Comparing models (basic details)

- A brief comparison table of the different models mentioned above is as follows,

| models  | type         | year | pre-trained? |         parameters         |
|---------|--------------|------|--------------|:--------------------------:|
| RNN     | conventional | -    | no           |       17k (one layer)      |
| LSTM    | conventional | 1997 | no           | 71k (one layer)            |
| GRU     | conventional | 2014 | no           | 30-40k (one layer)         |
| GPT-2   | recent       | 2019 | yes          | 117M, 345M, 774M, 1.5B     |
| GPT-Neo | recent       | 2021 | yes          | 125M, 1.2B, 2.7B           |
| T5      | recent       | 2020 | yes          | 60M, 220M, 770M, 2.8B, 11B |

### Comparing models (fine-tuning performance)

- A more detailed fine-tuning performance of the recent TG models for sentiment detection was performed. While the analysis is for a specific task, the process remains the same for any NLP problem that can be transformed in the form of text generation. 
- The following recent language models were discussed in the article, 
    - **GPT-2**: It is the second iteration of the original series of language models released by OpenAI. In fact, this series of GPT models made the language model famous! GPT stands for "Generative Pre-trained Transformer", and currently we have 3 versions of the model (v1, v2 and v3). Out of these only GPT-1 and GPT-2 are open-sourced, and hence we will pick the latest version for our experiment. On the technical side, the architecture of GPT-2 is made up of the decoder part of the Transformer architecture.
    - **GPT-Neo**: This model was released by  EleutherAI to counter the GPT-3 model which was not open-sourced. The architecture is quite similar to GPT-3, but training was done on [The Pile](https://pile.eleuther.ai/), an 825 GB sized text dataset.
    - **T5**: stands for "Text-to-Text Transfer Transformer" and was Google's answer to the world for open source language models. T5 paper showcase that using the complete encoder-decoder architecture (of the transformer) is better than only using the decoder (as done by the GPT series), hence they stay true to the original transformer architecture.

- The **results** from the analysis are as follows,

| model   | trial   0 | trial   1 | trial   2 | average |
|---------|:---------:|:---------:|:---------:|:-------:|
| GPT-Neo |   0.824   |   0.7893  |   0.808   |  0.8071 |
|   GPT-2 |   0.8398  |   0.808   |   0.806   |  0.8179 |
|      T5 |   0.8214  |   0.7976  |   0.804   |  0.8077 |

- **Conclusion**
    - *"While GPT-2 may have won this round, the result table does show the prowess of text generation models on whole. All of them performed very well on the sentiment detection task, and all it took was a few epochs of training. Even if this experiment was done for a single task, I hope this helps to show how easy it is to use the TG models for completely new tasks. In a way, if we can transform the NLP problem into that of text generation, rest assured the pre-trained model will not fail, well at least not drastically :) This makes them the perfect baseline if not the state-of-the-art for many tasks."*
## Additional materials

- How to generate text: using different decoding methods for language generation with Transformers - [Link](https://huggingface.co/blog/how-to-generate)

## References

```{bibliography}
:filter: docname in docnames
```