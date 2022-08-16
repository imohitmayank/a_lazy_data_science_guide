!!! warning
    This page is still under progress. Please check back after some time or [contact me](mailto:mohitmayank1@gmail.com)

## Introduction

- Loss functions are the "objectives" that neural networks (NN) tries to optimize. In fact, they are the mathematical personification of what we want to achieve with the NN. As the name suggests, it is a function that takes input *(in batches)* and compute a loss value that determines how further away the current model is from the ideal model. In an ideal world, we would expect the loss value to be 0, but in reality it could get very close to 0 and sometimes even be high enough when we terminate training to handle overfitting.
- Selecting a loss function for your NN depends a lot on your use case and even the type of data you are working with. For example, in case of regression, you can use MSE loss function. In case of classification, you can use Cross Entropy loss function. Here, we will go through some examples of loss functions.

### MAE (L1 loss)

- Mean Absolute Error (MAE) is another loss function used to calculate regresion loss. Due to the presence of absolute term instead of square, it is more robust to outliers. [It is differentiable](https://stats.stackexchange.com/questions/312737/mean-absolute-error-mae-derivative) at all points expect when predicted target value equals true target value. The actual formula is shown below,

$${\displaystyle \mathrm {MAE} ={\frac {\sum _{i=1}^{n}\left|y_{i}-\hat {y_{i}}\right|}{n}}}$$

### MSE (L2 loss)

- Mean Squared Error (MSE) loss function is used to measure the difference between the predicted value and the actual value. Basically it is the mean squared euclidean distance. It is most widely used loss function for regression tasks and representation (embedding) similarity task. The actual formuale is shown below, 

$${\displaystyle \operatorname {MSE} ={\frac {1}{n}}\sum _{i=1}^{n}\left(y_{i}-{\hat {y_{i}}}\right)^{2}}$$

- The MSE cost function is less resistant to outliers since the loss function squares big mistakes in order to punish the model. As a result, if the data is prone to many outliers, you shouldn't utilise it.

### Cross entropy loss

### Triplet loss


## References

[1] [Triplet Loss and Online Triplet Mining in TensorFlow](https://omoindrot.github.io/triplet-loss)

[2] [The 7 Most Common Machine Learning Loss Functions](https://builtin.com/machine-learning/common-loss-functions)