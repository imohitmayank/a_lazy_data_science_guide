Deep Learning Terms
=========================

Some of the most widely used terms in Deep Learning are shared below, 

## Logits, Soft and Hard targets

- Let us understand each of the terms one by one. For better understanding, let's take a dog vs cat image classification as an example. 
  - **Logits** are the un-normalized output of the model. In our cat vs dog example, logits will be, say, `10.1` for cat and `5.6` for dog for an image with cat. [Refer this SE question]((https://datascience.stackexchange.com/questions/31041/what-does-logits-in-machine-learning-mean)).
  - **Soft target**: are normalized logits by apply a [linear function](https://stats.stackexchange.com/questions/163695/non-linearity-before-final-softmax-layer-in-a-convolutional-neural-network). In our example, if we apply softmax to the logits we get `0.99` for cat and `0.1` for dog.
  - **Hard targets**: are the encoding of the soft targets. In our example, as the model predicted (here correctly) the image as of cat, the hard targets be `1` for cat and `0` for dog.

``` mermaid
graph LR
  A[Logits] -- normalization --> B[Soft Targets]
  B -- encoding --> C[Hard Targets]
```