
- Here are some questions and their answers to make you ready for your next interview. Best of luck :wave:

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
        
        The Bias-Variance trade-off refers to the trade-off between a model's ability to fit the training data well (low bias) and its ability to generalize well to new, unseen data (low variance). A model with high bias will underfit the training data, while a model with high variance will overfit the training data.

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