 
- Here are some tricky but interesting questions asked during Data Science interviews. 

!!! Question ""
    === "Question"
        While training deep learning models, why do we prefer training on mini-batch rather than on individual sample?

    === "Answer"

        First, the gradient of the loss over a mini-batch is an estimate of the gradient over the training set, whose quality improves as the batch size increases. Second, computation over a batch can be much more efficient than `m` computations for individual examples, due to the parallelism afforded by the modern computing platforms. [Ref](https://arxiv.org/abs/1502.03167v3)

!!! Question ""
    === "Question"
        What are the benefits of using Batch Normalizattion?

    === "Answer"

        Batch Normalization also has a beneficial effect on the gradient flow through the network, by reducing the dependence of gradients on the scale of the parameters or of their initial values. This allows us to use much higher learning rates with- out the risk of divergence. Furthermore, batch normal- ization regularizes the model and reduces the need for Dropout (Srivastava et al., 2014). Finally, Batch Normal- ization makes it possible to use saturating nonlinearities by preventing the network from getting stuck in the satu- rated modes. [Ref](https://arxiv.org/abs/1502.03167v3)

        
