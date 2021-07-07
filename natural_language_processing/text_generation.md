Text generation
========================

- Text generation is an interesting task in NLP, where the intention is to generate text when provided with some prompt as input. Usually, we apply some form of Seq2Seq model for this task.
- The Seq2Seq model consists of two main modules - 
    - **Encoder**: which takes text as input and encodes it to a compact vector representation, 
    - **Decoder**: which takes the compact vector representation as input and generates text as output. 
- Conventionally, the models were trained from scratch for a specific task, which requires a lot of training data. Recently, as NLP has deviated more towards fine-tuning methodology, we have a number of pre-trained models which either out-of-the-box work very well or can be fine-tuned for the specific task.
- Some of the conventional models were RNN, GRU and, LSTM. Whereas recent pre-trained models include Transformers, GPT-{1, 2, 3}, GPT-Neo, T5, etc.
- Text generation task is a very specific but also a very generic task, because we can formulate a lot of NLP tasks in the form of text generation. For example, *(not a complete list)*
    - **Language translation** English to Hindi translation, or any language for that matter, is just a text in and text out task
    - **Summarization** takes the complete text dump as input and generates crisp informative sentences.
    - **Question answering** question is taken as input and answer is the output. We can even include some context as input on closed-domain QA tasks.
    - **Sentiment analysis** is a classification task where we can provide the text as input and train the model to generate sentiment tag as output
    - **Rating prediction** where we have to rate the writing style, kind of regression problem, but still can be formulated as text in number out
- As obvious from the examples mentioned above, it is possible to formulate a lot of problems as a text-in-text-out task, and it could even expand to classification and regression type of tasks. Some examples tasks which can be performed using T5 is shown below, 

```{figure} /imgs/t5_example.gif
---
height: 200px
---
T5 text-to-text framework examples. See: [Google Blog](https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html)
```

<!-- {cite}`raffel2020exploring` -->

- A brief comparison table of the different model mentioned above is as follows,

| models  | type         | year | pre-trained? |         parameters         |
|---------|--------------|------|--------------|:--------------------------:|
| RNN     | conventional | -    | no           |       17k (one layer)      |
| LSTM    | conventional | 1997 | no           | 71k (one layer)            |
| GRU     | conventional | 2014 | no           | 30-40k (one layer)         |
| GPT-2   | recent       | 2019 | yes          | 117M, 345M, 774M, 1.5B     |
| GPT-Neo | recent       | 2021 | yes          | 125M, 1.2B, 2.7B           |
| T5      | recent       | 2020 | yes          | 60M, 220M, 770M, 2.8B, 11B |

