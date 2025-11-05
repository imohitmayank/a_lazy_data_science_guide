- Here are some questions and their answers to make you ready for your next interview. Best of luck :wave:

!!! Question ""
    === "Question"
        #### What is Deep learning and how is it different from traditional Machine learning?

    === "Answer"
        
        Deep learning is a subfield of machine learning that uses neural networks with many layers, called deep neural networks, to learn and make predictions. It is different from traditional machine learning in that it can automatically learn hierarchical representations of the data and doesn't rely heavily on feature engineering.
        
!!! Question ""
    === "Question"
        #### What is Dummy Variable Trap in ML?

    === "Answer"
        
        The dummy variable trap is a situation in which a set of variables are perfectly correlated with each other, making it impossible to estimate the parameters of a linear regression model. This occurs when one or more of the dummy variables (one-hot encoded variables) are perfectly correlated with the constant term, which is a column of ones in the design matrix.
        
        For example, consider a dataset with a categorical variable with three levels: red, blue, and green. If we create three dummy variables for this variable, we might end up with a situation where the sum of the dummy variables is always equal to 1 for each observation. This would make it impossible to estimate the parameters of the linear regression model, as the design matrix would be singular.
        
        To avoid the dummy variable trap, we can drop one of the dummy variables. This will ensure that the dummy variables are not perfectly correlated with each other, and the design matrix will be invertible.
        
        !!! note
            If using regularizing, then don't drop a level as it biases your model in favor of the variable you dropped. Refer [Damien Martin's Blog](https://kiwidamien.github.io/are-you-getting-burned-by-one-hot-encoding.html)
        

!!! Question ""
    === "Question"
        #### How does back-propagation work in a neural network?

    === "Answer" 
        
        Backpropagation is an algorithm used to train neural networks. It starts by propagating the input forward through the network, calculating the output. Then it compares the output to the desired output and calculates the error. The error is then propagated backwards through the network, adjusting the weights in the network so as to minimize the error. This process is repeated multiple times until the error is minimized.

!!! Question ""
    === "Question"
        #### While training deep learning models, why do we prefer training on mini-batch rather than on individual sample?

    === "Answer"

        First, the gradient of the loss over a mini-batch is an estimate of the gradient over the training set, whose quality improves as the batch size increases. Second, computation over a batch can be much more efficient than `m` computations for individual examples, due to the parallelism afforded by the modern computing platforms. [Ref](https://arxiv.org/abs/1502.03167v3)


!!! Question ""
    === "Question"
        #### What is Layer Normalization?

    === "Answer"

        Layer normalization is a technique used in deep learning to normalize the activations (outputs) of a neural network layer for each individual data sample. It works by computing the mean and variance of all features (neurons) in a layer for a single input, and then normalizing these values so that they have a standard distribution (zero mean and unit variance). This helps stabilize and accelerate the training process, making the model less sensitive to changes in the scale of the inputs and more robust to different batch sizes.

        - **How it works**: For each input sample, calculate the mean and variance across all features in a layer, then subtract the mean and divide by the standard deviation for each feature.

        - **Where it's used**: Especially useful in models like RNNs and transformers, and in any scenario where batch sizes are small or variable.

        - **Key benefit**: Works the same way during both training and inference, and does not depend on the batch size.
        

!!! Question ""
    === "Question"
        #### What is Batch Normalization?

    === "Answer"

    
        Batch normalization is a normalization technique that normalizes each feature across all samples in a mini-batch. This means, for each feature (e.g., each neuron in a layer), the mean and variance are computed across the entire batch, and each feature value is normalized using these batch statistics. [Refer](https://arxiv.org/abs/1502.03167v3)

        - **How it works**: For each feature, calculate the mean and variance across the current mini-batch, then normalize each value by subtracting the batch mean and dividing by the batch standard deviation.

        - **Where it's used**: Commonly used in convolutional neural networks (CNNs) and feedforward networks, especially with large and consistent batch sizes.

        - **Key benefit**: Helps accelerate training, improve generalization, and allows for higher learning rates.
        

!!! Question ""
    === "Question"
        #### What is Entropy *(information theory)*?

    === "Answer"

        Entropy is a measurement of uncertainty of a system. Intuitively, it is the amount of information needed to remove uncertainty from the system. The entropy of a probability distribution `p` for various states of a system can be computed as: $-\sum_{i}^{} (p_i \log p_i)$

!!! Question ""
    === "Question"
        #### Even though Sigmoid function is non-linear, why is Logistic regression considered a linear classifier?

    === "Answer"

        Logistic regression is called a linear classifier because it computes a linear combination of the input features $(z = w^T x + b)$, then applies the sigmoid function to output a probability. The decision boundary is defined by $w^T x + b = 0$, which is a linear equation—so the separation between classes is linear in the feature space, even though the output is passed through a non-linear sigmoid. [Refer](https://stats.stackexchange.com/questions/93569/why-is-logistic-regression-a-linear-classifier)

!!! Question ""
    === "Question"
        #### What is the difference between Logits, Soft and Hard targets?

    === "Answer"

        - Let us understand each of the terms one by one. For better understanding, let's take a dog vs cat image classification as an example. 
          - **Logits** are the un-normalized output of the model. In our cat vs dog example, logits will be, say, `10.1` for cat and `5.6` for dog for an image with cat. [Refer this SE question]((https://datascience.stackexchange.com/questions/31041/what-does-logits-in-machine-learning-mean)).
          - **Soft target**: are normalized logits by applying a [function](https://stats.stackexchange.com/questions/163695/non-linearity-before-final-softmax-layer-in-a-convolutional-neural-network). In our example, if we use softmax to the logits we get `0.99` for cat and `0.1` for dog.
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
        #### Explain Regularization and different types of regularization techniques.

    === "Answer"

        Regularization is a set of techniques used in machine learning to reduce overfitting and improve a model's ability to generalize to new, unseen data. Overfitting happens when a model learns not only the underlying patterns in the training data but also the noise, making it perform poorly on new data. Regularization addresses this by adding a penalty to the model's loss function, discouraging overly complex models and large parameter values.

        **Why use regularization?**

        - Prevents overfitting by discouraging complex models
        - Improves generalization to new data
        - Encourages simpler, more robust models

        **How does it work?**
        Regularization modifies the loss function by adding a penalty term based on the model's weights:
        
        $$
        \text{Loss} = \text{Original Loss} + \lambda \times \text{Penalty}
        $$

        where $\lambda$ controls the strength of the penalty.

        **Common regularization techniques:**

        - **L1 Regularization (Lasso):** Adds the sum of the absolute values of the weights. Can shrink some weights to exactly zero, effectively performing feature selection.
        - **L2 Regularization (Ridge):** Adds the sum of the squared values of the weights. Shrinks weights toward zero but rarely makes them exactly zero.
        - **Elastic Net:** Combines L1 and L2 penalties, balancing between feature selection and coefficient shrinkage.

        | Technique      | Penalty Type           | Effect on Weights         | Typical Use Case                      |
        |---------------|-----------------------|---------------------------|---------------------------------------|
        | L1 (Lasso)    | Sum of absolute values| Many weights set to zero  | Feature selection, sparse models      |
        | L2 (Ridge)    | Sum of squares        | Weights shrink toward zero| General shrinkage, no feature removal |
        | Elastic Net   | L1 + L2 combination   | Mix of both above         | Both shrinkage and feature selection  |

        **In summary:** Regularization is essential for building robust machine learning models. L1 and L2 are the most common forms, each adding different types of penalties to control model complexity and improve generalizability.

!!! Question ""
    === "Question"
        ####  Explain the concept of temperature in deep learning?

    === "Answer"
        
        In deep learning, the concept of "temperature" is often associated with the Softmax function and is used to control the degree of confidence or uncertainty in the model's predictions. It's primarily applied in the context of classification tasks, such as image recognition or natural language processing, where the model assigns probabilities to different classes.

        The Softmax function is used to convert raw model scores or logits into a probability distribution over the classes. Each class is assigned a probability score, and the class with the highest probability is typically selected as the predicted class.

        The Softmax function is defined as follows for a class "i":

        $$P(i) = \frac{e^{z_i / \tau}}{\sum_{j} e^{z_j / \tau}}$$

        Where:

        - \(P(i)\) is the probability of class "i."
        - \(z_i\) is the raw score or logit for class "i."
        - \(\tau\), known as the "temperature," is a positive scalar parameter.

        The temperature parameter, \(\tau\), affects the shape of the probability distribution. When \(\tau\) is high, the distribution becomes "soft," meaning that the probabilities are more evenly spread among the classes. A lower temperature results in a "harder" distribution, with one or a few classes having much higher probabilities.

        Here's how temperature impacts the Softmax function:
        
        - High \(\tau\): The model is more uncertain, and the probability distribution is more uniform, which can be useful when exploring diverse options or when dealing with noisy data.
        - Low \(\tau\): The model becomes more confident, and the predicted class will have a much higher probability. This is useful when you want to make decisive predictions.

        Temperature allows you to control the trade-off between exploration and exploitation in the model's predictions. It's a hyperparameter that can be adjusted during training or inference to achieve the desired level of certainty in the model's output, depending on the specific requirements of your application.

        Here is a good online [tool](https://artefact2.github.io/llm-sampling/index.xhtml) to learn about the impact of temperature and other parameters on output generation. 


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
        #### What is Gradient Checkpointing and how does it help in training deep neural networks?

    === "Answer"
        
        Gradient checkpointing is a memory optimization technique that helps save GPU memory during the training of deep neural networks by trading computation time for memory usage.

        **How it helps:**

        - Normally, when you train a neural network, the computer saves the outputs (called activations) of many layers during the forward pass, because these are needed later to calculate gradients during the backward pass.
        - Saving all these activations takes up a lot of GPU memory.
        - Gradient checkpointing changes this by only saving activations at certain points (called checkpoints) and **not saving all the intermediate activations**.
        - When the backward pass runs, the missing activations are **recomputed on demand** by rerunning parts of the forward pass.
        - This way, the training uses much less memory because fewer activations are saved, but it takes a bit more time since some calculations are repeated.

        **When it helps:**

        - When your model is very large or your GPU memory is limited, and you want to train bigger or deeper models than your memory would normally allow.
        - When you want to increase the batch size (the number of samples processed at once) but memory runs out.
        - It is especially useful for big transformer models like BERT or GPT, where activations take a lot of memory.

        **Tradeoff:**

        - It saves **significant memory (about 50-60%)**.
        - But it increases computation time by roughly **15-25%** because of the recomputation.
        - In many cases, this tradeoff lets you train bigger models or batches on hardware that otherwise wouldn't be able to handle them.

        So in simple terms, gradient checkpointing lets you "pause and forget" some intermediate calculations during training to save memory and "replay" them later when needed, trading off some extra time for a lot less memory usage.

!!! Question ""
    === "Question"
        #### What is Representation learning?

    === "Answer"

        Representation learning is the fundamental concept in AI that denotes the power of the system to learn multiple levels of feature representation with increasing abstraction i.e. learning representations of data. These representations are stored inside the neurons and are used to make predictions and make decisions. 

!!! Question ""
    === "Question"
        #### Explain Label smoothing.

    === "Answer"

        Label smoothing is a technique used in machine learning to prevent the model from becoming over-confident (overfitting). The smoothing is done by adding a small amount of noise to the labels of the training data, which makes the model less likely to overfit to the training data. Technically it generates soft labels by applying a weighted average between the uniform distribution and the hard label. Refer [Paper 1](https://arxiv.org/pdf/1906.02629.pdf) or
        [Paper 2](https://arxiv.org/pdf/2011.12562.pdf)

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
        
!!! Question ""
    === "Question"
        #### What is the different between Standardization and Normalization?

    === "Answer"

        Normalization is the process of scaling the data to a common scale. It is also known as Min-Max Scaling where the final range could be [0, 1] or [-1,1] or something else. $X_{new} = (X - X_{min})/(X_{max} - X_{min})$ Standardization is the process of scaling the data to have zero mean and unit variance. $X_{new} = (X - mean)/Std$

!!! Question ""
    === "Question"
        #### Is it possible that during ML training, both validation (or test) loss and accuracy, are increasing?

    === "Answer"

        Accuracy and loss are not necessarily exactly (inversely) correlated, as loss measures a difference between raw prediction (float) and class (0 or 1), while accuracy measures the difference between thresholded prediction (0 or 1) and class. So if raw predictions change, loss changes but accuracy is more "resilient" as predictions need to go over/under a threshold to actually change accuracy. [Soltius's answer on SE](https://stats.stackexchange.com/questions/282160/how-is-it-possible-that-validation-loss-is-increasing-while-validation-accuracy)


!!! Question ""
    === "Question"
        #### Is K-means clustering algorithm guaranteed to converge with unique result?

    === "Answer"

        K-means clustering algorithm is guaranteed to converge but the final result may vary based on the centroid initialisation. This is why it is suggested to try multiple initialization strategies and pick the one with best clustering. The convergence is guaranteed as the sum of squared distances between each point and its centroid strictly decreases over each iteration. Also the practical run time of k-means is basically linear. Refer [Visualizing K Means Clustering - Naftali Harris](https://www.naftaliharris.com/blog/visualizing-k-means-clustering/)

!!! Question ""
    === "Question"
        #### In K-means clustering, is it possible that a centroid has no data points assigned to it?

    === "Answer"

        Yes it is possible, imagine a centroid placed in middle of ring of other centroids. Several implementations either removes that centroid or random;y replace it somewhere else in the data space. Refer [Visualizing K Means Clustering - Naftali Harris](https://www.naftaliharris.com/blog/visualizing-k-means-clustering/)

!!! Question ""
    === "Question"
        #### What is entropy in information theory?

    === "Answer"

        Entropy is a measure of the amount of uncertainty or randomness in a system. It is often used in information theory and statistical mechanics to describe the unpredictability of a system or the amount of information required to describe it. It's formula is, $\mathrm {H} (X):=-\sum _{x\in {\mathcal {X}}}p(x)\log p(x)=\mathbb {E} [-\log p(X)]$

        Here is an [excellent video](https://www.youtube.com/watch?v=ErfnhcEV1O8) from Aurelien Geron, explaining the topic.

!!! Question ""
    === "Question"
        #### What is the difference between supervised and unsupervised learning?

    === "Answer"
        
        Supervised learning uses labeled data to train a model to make predictions, while unsupervised learning uses unlabeled data to find patterns or structure in the data.

!!! Question ""
    === "Question"
        #### How do you evaluate the performance of a machine learning model?

    === "Answer"
        
        One common method is to split the data into a training set and a test set, and use metrics such as accuracy, precision, recall, and F1 score to evaluate the model's performance on the test set.

!!! Question ""
    === "Question"
        #### What is overfitting in machine learning and how can it be prevented?

    === "Answer"
        
        Overfitting occurs when a model is too complex and is able to fit the noise in the training data, leading to poor performance on new, unseen data. To prevent overfitting, methods such as cross-validation, regularization, and early stopping can be used.

!!! Question ""
    === "Question"
        #### What is the difference between a decision tree and random forest?

    === "Answer"
        
        A decision tree is a single tree model that makes a prediction by traversing the tree from the root to a leaf node, while a random forest is an ensemble of decision trees, where the final prediction is made by averaging the predictions of all the trees in the forest.

!!! Question ""
    === "Question"
        #### What is the Bias-Variance trade-off in machine learning?

    === "Answer"
        
        The bias-variance trade-off is a key concept in machine learning that describes the balance between two types of errors affecting a model's ability to generalize:

        - **Bias**: Error from overly simplistic assumptions in the model. High bias can cause underfitting—poor performance on both training and test data.
        - **Variance**: Error from excessive sensitivity to small fluctuations in the training set. High variance can cause overfitting—good performance on training data but poor generalization to new data.

        As model complexity increases, bias decreases but variance increases. The goal is to find a balance: a model complex enough to capture patterns (low bias) but not so complex that it overfits (low variance).

        | Model Complexity | Bias           | Variance      | Generalization      |
        |------------------|----------------|---------------|---------------------|
        | Too Simple       | High (underfit)| Low           | Poor                |
        | Too Complex      | Low            | High (overfit)| Poor                |
        | Just Right       | Low/Moderate   | Low/Moderate  | Good                |

        **Managing the trade-off:**

        - Use regularization to penalize complexity and reduce overfitting.
        - Use cross-validation to estimate model performance on unseen data.
        - Increasing training data can help reduce variance.

        **In summary:** The bias-variance trade-off is about finding the sweet spot where your model is neither too simple nor too complex, so it generalizes well to new data.

!!! Question ""
    === "Question"
        #### What is the difference between batch and online learning?

    === "Answer"
        
        Batch learning is a type of machine learning where the model is trained on a fixed dataset and the parameters are updated after processing the entire dataset. In contrast, online learning is a type of machine learning where the model is trained on a continuous stream of data and the parameters are updated incrementally after processing each example.

!!! Question ""
    === "Question"
        #### What is the difference between a decision boundary and a decision surface in machine learning?

    === "Answer"
        
        A decision boundary is a boundary that separates different classes in a dataset, it can be represented by a line or a hyperplane in a two-dimensional or multi-dimensional space respectively. A decision surface is a generalization of decision boundary, it's a surface that separates different classes in a dataset, it can be represented by a surface in a multi-dimensional space. In simple words, a decision boundary is a one-dimensional representation of a decision surface.

!!! Question ""
    === "Question"
        #### What is the use of principal component analysis (PCA) in machine learning?

    === "Answer"
        
        Principal component analysis (PCA) is a technique used to reduce the dimensionality of a dataset by identifying the most important features, called principal components. PCA finds a new set of uncorrelated features, called principal components, that can explain most of the variance in the original data. This can be useful for visualizing high-dimensional data, reducing noise, and improving the performance of machine learning models.


!!! Question ""
    === "Question"
        #### What is the use of the Random Forest algorithm in machine learning?

    === "Answer"
        
        Random Forest is an ensemble learning technique that combines multiple decision trees to improve the performance and stability of the model. It works by creating multiple decision trees using a random subset of the features and training data, and then averaging the predictions of all the trees to make a final prediction. Random Forest algorithm is often used for classification and regression problems, it's robust to outliers, missing values, and irrelevant features, and it can also be used for feature selection and feature importance analysis.


!!! Question ""
    === "Question"
        #### What is the difference between a generative model and a discriminative model?

    === "Answer"
        
        A generative model learns the probability distribution of the data and can generate new samples from it, while a discriminative model learns the boundary between different classes and make predictions based on it. Generative models are used for tasks such as density estimation, anomaly detection, and data generation, while discriminative models are used for tasks such as classification and regression.


!!! Question ""
    === "Question"
        #### What is the difference between an autoencoder and a variational autoencoder?

    === "Answer"
        
        An autoencoder is a type of neural network that learns to encode and decode input data, it can be used to reduce the dimensionality of the data and learn a compact representation of it. A variational autoencoder (VAE) is a type of autoencoder that learns a probabilistic encoding of the input data, it generates new samples from the learned distribution. VAE can be used for tasks such as image generation and anomaly detection.

!!! Question ""
    === "Question"
        #### What is Expectation-Maximization (EM) algorithm?

    === "Answer"
        
        The Expectation-Maximization (EM) algorithm is a method for finding maximum likelihood estimates in incomplete data problems, where some of the data is missing or hidden. EM works by iteratively refining estimates of the missing data and the parameters of the model, until it converges to a local maximum of the likelihood function. It can be used for tasks such as clustering, image segmentation, and missing data imputation.

!!! Question ""
    === "Question"
        #### What is the difference between L1 and L2 regularization in machine learning?

    === "Answer"
        
        L1 and L2 regularization are methods used to prevent overfitting in machine learning models by adding a penalty term to the loss function. L1 regularization adds the absolute value of the weights to the loss function, while L2 regularization adds the square of the weights. L1 regularization leads to sparse models where some weights will be zero, while L2 regularization leads to models where all weights are small.

!!! Question ""
    === "Question"
        #### Explain Support Vector Machine (SVM).

    === "Answer"
        
        Support Vector Machine (SVM) is a supervised learning algorithm that can be used for classification and regression tasks. SVM works by finding the hyperplane that maximally separates the different classes in a dataset and then uses this hyperplane to make predictions. SVM is particularly useful when the data is linearly separable, it's also robust to high-dimensional data and it can be used with kernel functions to solve non-linearly separable problems.

!!! Question ""
    === "Question"
        #### What is the use of the k-nearest neighbors (k-NN) algorithm?

    === "Answer"
        
        k-nearest neighbors (k-NN) is a type of instance-based learning algorithm that can be used for classification and regression tasks. The algorithm works by finding the k training examples that are closest to a new input and using the majority class or average value of those examples to make a prediction. k-NN is a simple and efficient algorithm that can be used for tasks such as image classification, anomaly detection, and recommendation systems.

!!! Question ""
    === "Question"
        #### What is the use of the Random Sampling method for feature selection in machine learning?

    === "Answer"
        
        Random Sampling is a method for feature selection that involves randomly selecting a subset of features from the dataset and evaluating the performance of a model trained on that subset. The subset of features that result in the best performance are then chosen for further analysis or use in a final model. This method can be useful when the number of features is large and there is no prior knowledge of which features are most relevant.

!!! Question ""
    === "Question"
        #### Explain Bagging method in ensemble learning?

    === "Answer"
        
        Bagging (Bootstrap Aggregating) is a method for ensemble learning that involves training multiple models on different subsets of the data and then combining the predictions of those models. The subsets of data are created by randomly sampling the original data with replacement, this method helps to reduce the variance of the model and increase the robustness of the predictions. Bagging is commonly used with decision trees and can be implemented using Random Forest algorithm.

!!! Question ""
    === "Question"
        #### Explain AdaBoost method in ensemble learning?

    === "Answer"
        
        AdaBoost (Adaptive Boosting) is a method for ensemble learning that involves training multiple models on different subsets of the data and then combining the predictions of those models. The subsets of data are created by giving more weight to the examples that are misclassified by the previous models, this method helps to increase the accuracy of the model. AdaBoost is commonly used with decision trees and can be used with any type of base classifier.

!!! Question ""
    === "Question"
        #### Explain Gradient Boosting method in ensemble learning?

    === "Answer"
        
        Gradient Boosting is a method for ensemble learning that involves training multiple models in a sequential manner, where each model tries to correct the mistakes of the previous model. The method uses gradient descent to minimize the loss function, this method is commonly used with decision trees and it can be used with any type of base classifier. Gradient Boosting is a powerful method that can achieve state-of-the-art performance in many machine learning tasks.

!!! Question ""
    === "Question"
        #### Explain XGBoost method in ensemble learning?

    === "Answer"
        
        XGBoost (Extreme Gradient Boosting) is a specific implementation of the Gradient Boosting method that uses a more efficient tree-based model and a number of techniques to speed up the training process and reduce overfitting. XGBoost is commonly used in machine learning competitions and it's one of the most popular libraries used for gradient boosting. It's used for classification and regression problems.

!!! Question ""
    === "Question"
        #### What is `group_size` in context of Quantization?

    === "Answer"
        
        Group size is a parameter used in the quantization process that determines the number of weights or activations *(imagine weights in a row of matrix)* that are quantized together. A smaller group size can lead to better quantization accuracy, but it can also increase the memory and computational requirements of the model. Group size is an important hyperparameter that needs to be tuned to achieve the best trade-off between accuracy and efficiency. Note, the default groupsize for a GPTQ is 1024. [Refer this interesting Reddit discussion](https://www.reddit.com/r/LocalLLaMA/comments/12rtg82/what_is_group_size_128_and_why_do_30b_models_give/?rdt=46348)

!!! Question ""
    === "Question"
        #### What is EMA (Exponential Moving Average) in context of deep learning?

    === "Answer"

        EMA is a technique used in deep learning to stabilize the training process and improve the generalization performance of the model. It works by maintaining a moving average of the model's parameters during training, which helps to smooth out the noise in the gradients and prevent the model from overfitting to the training data. Usually during training, the model's parameters are updated using the gradients of the loss function, but the EMA parameters are updated using a weighted average of the current parameters and the previous EMA parameters, as shown below:

        $$
        \text{ema_param} = \text{ema_param} * \text{decay} + \text{param} * (1 - \text{decay})
        $$

        One more advantage of EMA model is that it can be used to resume the training. Some models provide both the model parameters and EMA parameters.

!!! Question ""
    === "Question"
        #### What is Bradley-Terry model? And how is it used in machine learning?

    === "Answer"

        Bradley-Terry model is a probability model that can be used to model the outcome of a pairwise comparison between two items. It is commonly used in sports analytics to rank teams or players based on their performance in head-to-head matches. Refer: [Wikipedia](https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model)

        In case of ML, the model is a popular choice for modeling human preferences in the context of training language models. It stipulates that the probability of a human preferring one completion over another can be expressed as a ratio of exponentials of the latent reward associated with each completion. Specifically, the human preference distribution 
        
        $$ 
        p^*(y_1 \succ y_2 | x) = \frac{exp(r^*(x, y_1))}{exp(r^*(x, y_1)) + exp(r^*(x, y_2))} 
        $$

        This model is commonly used to capture human preferences to provide a framework for understanding and incorporating human feedback into the training of language models.

!!! Question ""
    === "Question"
        #### What is Rejection Sampling in Machine Learning?

    === "Answer"
        Rejection sampling is a method to generate samples from a complex target distribution (like a hard-to-sample probability curve) by using a simpler "proposal" distribution you can easily sample from (e.g., a uniform or normal distribution). 
        
        Here's how it works: you first pick a proposal distribution that covers the target's range. Then, you repeatedly draw samples from this simpler distribution and "accept" or "reject" each sample based on a quality check—if a random number (from 0 to 1) is less than the ratio of the target's density to the proposal's density (scaled by a constant), you keep the sample; otherwise, you discard it. This process ensures the accepted samples match the target distribution. It's like filtering out bad candidates until you're left with samples that fit your desired pattern. While simple to implement, it becomes inefficient for high-dimensional data or if the proposal distribution doesn't closely match the target shape.

