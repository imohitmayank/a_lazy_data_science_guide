!!! warning
    This page is still under progress. Please check back after some time or [contact me](mailto:mohitmayank1@gmail.com)

## Introduction

- Loss functions are the "ideal objectives" that neural networks (NN) tries to optimize. In fact, they are the mathematical personification of what we want to achieve with the NN. As the name suggests, it is a function that takes input and compute a loss value that determines how further away the current model is from the ideal model for that example. In an ideal world, we would expect the loss value to be 0, but in reality it could get very close to 0 and sometimes even be high enough so that we terminate training to handle overfitting.
- We also have cost functions that is nothing but aggrgation of the loss functions over a batch or complete dataset. The cost function is the function that we use in practice to optimize the model.

!!! Hint
    Loss functions --> loss on one example

    Cost functions --> loss on entire dataset or a batch

## Types of Loss functions

- Selecting a loss function for your NN depends a lot on your use case and even the type of data you are working with. For example, in case of regression, you can use MSE loss function. In case of classification, you can use Cross Entropy loss function. 
- Here, we will go through some examples of loss functions.

### MAE (L1 loss)

- Mean Absolute Error (MAE) loss function is used to calculate regression loss. Due to the presence of absolute term instead of square, it is more robust to outliers. [It is differentiable](https://stats.stackexchange.com/questions/312737/mean-absolute-error-mae-derivative) at all points expect when predicted target value equals true target value. The actual formula is shown below,

$${\displaystyle \mathrm {MAE_{loss}}(i) ={\left|y_{i}-\hat {y_{i}}\right|}}$$

$${\displaystyle \mathrm {MAE_{cost}} ={\frac {1}{n}{\sum _{i=1}^{n} \mathrm{MAE_loss}(i)}}}$$

### MSE (L2 loss)

- Mean Squared Error (MSE) loss function is used to measure the difference between the predicted value and the actual value. Basically it is the mean squared euclidean distance. It is most widely used loss function for regression tasks and representation (embedding) similarity task. The actual formuale is shown below, 

$${\displaystyle \operatorname {MSE_loss}(i) = (y_{i}-{\hat {y_{i}}})^{2}}$$

$${\displaystyle \operatorname {MSE_cost} ={\frac {1}{n}}\sum _{i=1}^{n}\operatorname {MSE_loss}(i)}$$

- The MSE cost function is less resistant to outliers since the loss function squares big mistakes in order to punish the model. As a result, if the data is prone to many outliers, you shouldn't utilise L2 loss.

### Cross entropy loss

- Cross entropy loss is used for classification tasks. It is a simplication of Kullback–Leibler divergence that is used to compute the difference between two probability distributions *(here the model's prediction and true one)*. For binary classification the formula is shown below, ($y$ is the actual class and $\hat{y}$ is the predicted class)

$${\displaystyle \operatorname {CrossEntropy_loss}(i) = -(y_i \log(\hat{y_i})+(1-y_i) \log(1-\hat{y_i}))}$$

$${\displaystyle \operatorname {CrossEntropy_cost} ={\frac {1}{n}}\sum _{i=1}^{n}\operatorname {CrossEntropy_loss}(i)}$$

- Let's go through the different possibilities, 
  - if $y_i=1$, 
    - the loss function reduces to only the left part i.e. $-y_i \log(\hat{y_i})$
    - now to have a small loss, model would want the $\log(\hat{y_i})$ to be large *(bcoz of negative sign)*. 
    - for this, model will make $\hat{y_i}$ large *(ideally 1)*
  - if $y_i=0$, 
    - the loss function reduces to only the right part i.e. $-(1-y_i) \log(1-\hat{y_i})$
    - now to have a small loss, model would want the $\log(1 - \hat{y_i})$ to be large *(bcoz of negative sign)*. 
    - for this, model will make $\hat{y_i}$ small *(ideally 0)*

!!! Note
    For both the cases we want to make sure that $\hat{y_i}$ is bounded between 0 and 1. For this, usually the output of model is passed through some activation function like sigmoid.

    Additionally, some people may argue why not use a simple $(\hat{y}-y)^2$ as the loss function for classification task. The major problem is that with this loss function the optimization becomes non convex hence the model will not converge.

<iframe src="https://www.desmos.com/calculator/lfquh1ib5d?embed" width="500" height="500" style="border: 1px solid #ccc" frameborder=0></iframe>

- Above we can see loss function graph plotted for both the cases,
  - Red line is for case when $y_i=1$. The line is plotted for $x=-y_i \log(\hat{y_i})$. As we want the loss to be zero, it is only possible for $y_i=1$.
  - Blue line is for case when $y_i=0$. The line is plotted for $x=-(1-y_i) \log(1-\hat{y_i})$. As we want the loss to be zero, it is only possible for $y_i=0$.

!!! Hint
    Refer [this excellent video](https://www.coursera.org/lecture/neural-networks-deep-learning/logistic-regression-cost-function-yWaRd) from AndrewNg on CrossEntropyLoss for more details.

<!-- ### Triplet loss -->


## References

[1] [Triplet Loss and Online Triplet Mining in TensorFlow](https://omoindrot.github.io/triplet-loss)

[2] [The 7 Most Common Machine Learning Loss Functions](https://builtin.com/machine-learning/common-loss-functions)