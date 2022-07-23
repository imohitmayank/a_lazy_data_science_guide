 
- One of the most obvious reason for failing an interview is not knowing the answer to the questions. While there are other factors at play, like how confident you sound, your behavior, attitude and even the mood of the interviewer, knowledge of the Data science field is well within the scope of this book - hence something I can help you with. So here are some questions to make you ready for your upcoming interview.  

!!! Question ""
    === "Question"
        While training deep learning models, why do we prefer training on mini-batch rather than on individual sample?

    === "Answer"

        First, the gradient of the loss over a mini-batch is an estimate of the gradient over the training set, whose quality improves as the batch size increases. Second, computation over a batch can be much more efficient than `m` computations for individual examples, due to the parallelism afforded by the modern computing platforms. [Ref](https://arxiv.org/abs/1502.03167v3)

!!! Question ""
    === "Question"
        What are the benefits of using Batch Normalizattion?

    === "Answer"

        Batch Normalization also has a beneficial effect on the gradient flow through the network, by reducing the dependence of gradients on the scale of the parameters or of their initial values. This allows us to use much higher learning rates without the risk of divergence. Furthermore, batch normalization regularizes the model and reduces the need for Dropout (Srivastava et al., 2014). Finally, Batch Normalization makes it possible to use saturating nonlinearities by preventing the network from getting stuck in the saturated modes. [Ref](https://arxiv.org/abs/1502.03167v3)

!!! Question ""
    === "Question"
        What is weight tying in language model?

    === "Answer"

        Weight-tying is where you have a language model and use the same weight matrix for the input-to-embedding layer (the input embedding) and the hidden-to-softmax layer (the output embedding). The idea is that these two matrices contain essentially the same information, each having a row per word in the vocabulary. [Ref](https://tomroth.com.au/weight_tying/)