 
- Here are some questions and their answers to make you ready for your next interview. Best of luck :wave:

!!! Question ""
    === "Question"
        #### While training deep learning models, why do we prefer training on mini-batch rather than on individual sample?

    === "Answer"

        First, the gradient of the loss over a mini-batch is an estimate of the gradient over the training set, whose quality improves as the batch size increases. Second, computation over a batch can be much more efficient than `m` computations for individual examples, due to the parallelism afforded by the modern computing platforms. [Ref](https://arxiv.org/abs/1502.03167v3)

!!! Question ""
    === "Question"
        #### What are the benefits of using Batch Normalizattion?

    === "Answer"

        Batch Normalization also has a beneficial effect on the gradient flow through the network, by reducing the dependence of gradients on the scale of the parameters or of their initial values. This allows us to use much higher learning rates without the risk of divergence. Furthermore, batch normalization regularizes the model and reduces the need for Dropout (Srivastava et al., 2014). Finally, Batch Normalization makes it possible to use saturating nonlinearities by preventing the network from getting stuck in the saturated modes. [Ref](https://arxiv.org/abs/1502.03167v3)

!!! Question ""
    === "Question"
        #### What is weight tying in language model?

    === "Answer"

        Weight-tying is where you have a language model and use the same weight matrix for the input-to-embedding layer (the input embedding) and the hidden-to-softmax layer (the output embedding). The idea is that these two matrices contain essentially the same information, each having a row per word in the vocabulary. [Ref](https://tomroth.com.au/weight_tying/)


!!! Question ""
    === "Question"
        #### Is BERT a Text Generation model?

    === "Answer"

        Short answer is no. BERT is not a text generation model or a language model because the probability of the predicting a token in masked input is dependent on the context of the token. This context is bidirectional, hence the model is not able to predict the next token in the sequence accurately with only one directional context *(as expected for language model)*.


!!! Question ""
    === "Question"
        #### What is Entropy *(information theory)*?

    === "Answer"

        Entropy is a measurement of uncertainty of a system. Intuitively, it is the amount of information needed to remove uncertainty from the system. The entropy of a probability distribution `p` for various states of a system can be computed as: $-\sum_{i}^{} (p_i \log p_i)$

!!! Question ""
    === "Question"
        #### What is so special about the special tokens used in different LM tokenizers?

    === "Answer"

        Special tokens are called special because they are added for a certain purpose and are independent of the input. For example, in BERT we have `[CLS]` token that is added at the start of every input sentence and `[SEP]` is a special separator token. Similarly in GPT2, `<|endoftext|>` is special token to denote end of sentence. Users can create their own special token based on their specific use case and train them during finetuning. [Refer cronoik's answer in SO](https://stackoverflow.com/questions/71679626/what-is-so-special-about-special-tokens)


!!! Question ""
    === "Question"
        #### What are Attention Masks?

    === "Answer"

        Attention masks are the token level boolean identifiers used to differentiate between important and not important tokens in the input. One use case is during batch training, where a batch with text of different lengths can be created by adding paddings to shorter texts. The padding tokens can be identified using 0 in attention mask and the original input tokens can be marked as 1. [Refer blog @ lukesalamone.com](https://lukesalamone.github.io/posts/what-are-attention-masks/)

        !!! Note
        We can use a special token for padding. For example in BERT it can be `[PAD]` token and in GPT-2 we can use `<|endoftext|>` token.

!!! Question ""
    === "Question"
        #### What is the difference between Attention and Self-Attention?

    === "Answer"

        Self-attention (SA) is applied within one component, so the input is from one component only. One example is the encoder block in Transformers, where SA is used, and it takes the tokens from only the sentence as the input. Attention on the other hand can be used to connect two components or modality of data. Decoder in Transformer has Attention layer that connects the encoder and decoder data together. [Refer StackExchange QA](https://datascience.stackexchange.com/questions/49468/whats-the-difference-between-attention-vs-self-attention-what-problems-does-ea)

!!! Question ""
    === "Question"
        #### What is the difference between Logits, Soft and Hard targets?

    === "Answer"

        - Let us understand each of the terms one by one. For better understanding, let's take a dog vs cat image classification as an example. 
          - **Logits** are the un-normalized output of the model. In our cat vs dog example, logits will be, say, `10.1` for cat and `5.6` for dog for an image with cat. [Refer this SE question]((https://datascience.stackexchange.com/questions/31041/what-does-logits-in-machine-learning-mean)).
          - **Soft target**: are normalized logits by applying a [linear function](https://stats.stackexchange.com/questions/163695/non-linearity-before-final-softmax-layer-in-a-convolutional-neural-network). In our example, if we use softmax to the logits we get `0.99` for cat and `0.1` for dog.
          - **Hard targets**: are the encoding of the soft targets. In our example, as the model predicted (here correctly) the image as of cat, the hard targets be `1` for cat and `0` for dog.

        ``` mermaid
        graph LR
            A[Logits] -- normalization --> B[Soft Targets]
            B -- encoding --> C[Hard Targets]
        ```