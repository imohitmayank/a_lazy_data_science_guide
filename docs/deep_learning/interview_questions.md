 
- Here are some questions and their answers to make you ready for your next interview. Best of luck :wave:

!!! Question ""
    === "Question"
        #### What is deep learning and how is it different from traditional machine learning?

    === "Answer"
        
        Deep learning is a subfield of machine learning that uses neural networks with many layers, called deep neural networks, to learn and make predictions. It is different from traditional machine learning in that it can automatically learn hierarchical representations of the data and doesn't rely heavily on feature engineering.
        
!!! Question ""
    === "Question"
        #### How does backpropagation work in a neural network?

    === "Answer" 
        
        Backpropagation is an algorithm used to train neural networks. It starts by propagating the input forward through the network, calculating the output. Then it compares the output to the desired output and calculates the error. The error is then propagated backwards through the network, adjusting the weights in the network so as to minimize the error. This process is repeated multiple times until the error is minimized.

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
        #### What is Entropy *(information theory)*?

    === "Answer"

        Entropy is a measurement of uncertainty of a system. Intuitively, it is the amount of information needed to remove uncertainty from the system. The entropy of a probability distribution `p` for various states of a system can be computed as: $-\sum_{i}^{} (p_i \log p_i)$

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

!!! Question ""
    === "Question"
        #### How do you handle overfitting in deep learning models?

    === "Answer"

        - Overfitting occurs when a model becomes too complex and starts to fit the noise in the training data, rather than the underlying pattern. There are several ways to handle overfitting in deep learning models, including:
          - **Regularization techniques** such as L1 and L2 regularization, which add a penalty term to the loss function to discourage large weights
          - **Early stopping**, where training is stopped before the model has a chance to fully fit the noise in the training data
          - **Dropout**, which randomly drops out a certain percentage of neurons during training to prevent them from co-adapting and becoming too specialized
          - Adding **more data** to the training set


!!! Question ""
    === "Question"
        ####  Can you explain the concept of convolutional neural networks (CNN)?

    === "Answer"
        
        A convolutional neural network (CNN) is a type of neural network that is primarily used for learning image and video patterns. CNNs are designed to automatically and adaptively learn spatial hierarchies of features from input data. They use a variation of multi-layer perceptrons, designed to require minimal preprocessing. Instead of hand-engineered features, CNNs learn features from data using a process called convolution.

!!! Question ""
    === "Question"
        #### How do you handle missing data in deep learning?

    === "Answer"
        
        - Missing data can be handled in several ways, including:
          - Removing the rows or columns with missing data
          - Interpolation or imputation of missing values
          - Using a technique called masking, which allows the model to ignore missing values when making predictions

!!! Question ""
    === "Question"
        #### Can you explain the concept of transfer learning in deep learning?

    === "Answer"
        
        Transfer learning is a technique where a model trained on one task is used as a starting point for a model on a second, related task. This allows the model to take advantage of the features learned from the first task and apply them to the second task, which can lead to faster training and better performance. This can be done by using a pre-trained model as a feature extractor or fine-tuning the pre-trained model on new data.

<!-- !!! Question ""
    === "Question"
        #### What is the difference between batch normalization and layer normalization?

    === "Answer"
        
        Batch normalization normalizes the activations of a layer for each mini-batch during training, where as Layer normalization normalizes the activations of a layer for the whole dataset during training. Layer normalization is typically used in recurrent neural networks (RNNs) where the normalization is performed across the feature dimension, while batch normalization is typically used in feedforward neural networks, where the normalization is performed across the mini-batch dimension. -->

!!! Question ""
    === "Question"
        #### What is Gradient Descent in deep learning?

    === "Answer"
        
        Gradient Descent is an optimization algorithm used to minimize the loss function of a neural network. It works by updating the weights of the network in the opposite direction of the gradient of the loss function with respect to the weights. The magnitude of the update is determined by the learning rate. There are several variants of gradient descent, such as batch gradient descent, stochastic gradient descent, and mini-batch gradient descent.

!!! Question ""
    === "Question"
        #### Please explain what is Dropout in deep learning?

    === "Answer"
        
        Dropout is a regularization technique used in deep learning to prevent overfitting. It works by randomly dropping out a certain percentage of neurons during training, effectively reducing the capacity of the network. This forces the network to learn multiple independent representations of the data, making it less prone to overfitting.

!!! Question ""
    === "Question"
        #### What are Autoencoder?

    === "Answer"
        
        An autoencoder is a type of neural network that is trained to reconstruct its input. It has an encoder part that compresses the input into a lower-dimensional representation called the bottleneck or latent code, and a decoder part that reconstructs the input from the latent code. Autoencoders can be used for tasks such as dimensionality reduction, anomaly detection and generative modelling.

!!! Question ""
    === "Question"
        #### Can you explain the concept of attention mechanism in deep learning?

    === "Answer"
        
        Attention mechanism is a way to weight different parts of the input in a neural network, giving more importance to certain features than others. It is commonly used in tasks such as machine translation, where the model needs to focus on different parts of the input sentence at different times. Attention mechanisms can be implemented in various ways, such as additive attention, dot-product attention, and multi-head attention.


!!! Question ""
    === "Question"
        #### What are Generative Adversarial Networks (GANs)?

    === "Answer"
        
        Generative Adversarial Networks (GANs) are a type of generative model that consists of two parts, a generator and a discriminator. The generator is trained to generate new data that is similar to the data it was trained on, while the discriminator is trained to distinguish the generated data from the real data. The two parts are trained together in a game-theoretic manner, where the generator tries to generate data that can fool the discriminator, and the discriminator tries to correctly identify the generated data.

!!! Question ""
    === "Question"
        #### Can you explain the concept of Memory Networks in deep learning?

    === "Answer"
        
        Memory networks are a type of neural network architecture that allow the model to access and manipulate an external memory matrix, which can be used to store information that is relevant to the task. This allows the model to reason about the past and use this information to make predictions about the future. Memory networks have been used in tasks such as language understanding and question answering. [Refer this](https://arxiv.org/abs/1410.3916) for more details.

!!! Question ""
    === "Question"
        #### Explain Capsule Networks in deep learning?

    === "Answer"
        
        Capsule networks are a type of neural network architecture that aims to overcome the limitations of traditional convolutional neural networks (CNNs) by using a new type of layer called a capsule. A capsule contains multiple neurons that work together to represent an object or part of an object, and the activities of the neurons are used to represent the properties of the object such as position, size and orientation. Capsule networks have been used in tasks such as image classification and object detection.

!!! Question ""
    === "Question"
        #### Can you explain the concept of generative models in deep learning?

    === "Answer"
        
        Generative models are a type of deep learning model that can generate new data that is similar to the data it was trained on. These models are trained on a dataset and learn the underlying probability distribution of the data, allowing them to generate new, unseen data that fits that distribution. Examples of generative models include Generative Adversarial Networks (GANs) and Variational Autoencoders (VAEs).

!!! Question ""
    === "Question"
        #### What is the concept of adversarial training in deep learning?

    === "Answer"
        
        Adversarial training is a technique used to improve the robustness of deep learning models by generating adversarial examples and using them to train the model. Adversarial examples are inputs that are slightly perturbed in such a way as to cause the model to make a mistake. By training the model on these examples, it becomes more robust to similar perturbations in the real world.

!!! Question ""
    === "Question"
        #### What is weight initialization in deep learning?

    === "Answer"
        
        Weight initialization is the process of setting the initial values for the weights in a neural network. The initial values of the weights can have a big impact on the network's performance and training time. There are several methods to initialize weights, including random initialization, Glorot initialization, and He initialization. Each of these methods have different properties and are more suitable for different types of problems.

!!! Question ""
    === "Question"
        #### Explain data augmentation?

    === "Answer"
        
        Data augmentation is a technique used to increase the amount of data available for training a deep learning model. This is done by creating new training examples by applying various random transformations to the original data, such as random cropping, flipping, or rotation. Data augmentation can be a powerful tool to prevent overfitting and improve the generalization performance of a mode.