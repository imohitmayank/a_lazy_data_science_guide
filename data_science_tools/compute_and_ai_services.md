# Compute and AI Services

- Gone are the days when we needed high end local devices to do literally anything. Currently there are plethora of services available online *(and many of them are free!)* that provide not only compute to use as you feel, but also generic AI services. 

- Just like a mechanic should know about all the tools at his disposal, a data scientist should be aware of different ready-made and possibly free services available. You can quote me on this, *"Never pay for what you can do for free, and never build something which has already been built"* 😎

- With this fortune cookie quote in mind, let's look into some of the famous compute and AI services.

## CaaS: Compute as a Service

- In this section we will cover some of the famous *(and with some hint of free)* platforms that provide compute-as-a-service (CaaS). These CaaS sometimes could be plain simple virtual machines, sometime they can be a cluster of nodes, while in other cases they can also be jupyter like coding environment. Let's go through some of the examples.

### Google Colab

#### Introduction

TODO

#### Google Colab Snippets

##### Run tensorboard to visualize embeddings

- Taken from: [how-to-use-tensorboard-embedding-projector](https://stackoverflow.com/questions/40849116/how-to-use-tensorboard-embedding-projector)

```{code-block}
---
lineno-start: 1
emphasize-lines: 13, 14
---
import numpy as np
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
from torch.utils.tensorboard import SummaryWriter

vectors = np.array([[0,0,1], [0,1,0], [1,0,0], [1,1,1]])
metadata = ['001', '010', '100', '111']  # labels
writer = SummaryWriter()
writer.add_embedding(vectors, metadata)
writer.close()

%load_ext tensorboard
%tensorboard --logdir=runs
```

### Kaggle

TODO

### DeepNote

TODO

## MLaaS: Machine Learning as a Service

- In this section we will cover some of the famous platforms that provide Machine learning-as-a-Service (MLaaS). These MLaaS take care of infrastructure related aspect of data holding, data preparing, model training and model deployment. On top of this, they provide a repository of classical ML algorithms that can be leverage to create data science solutions. The idea is to make data science as a plug and play solution creation activity, as they take care of most of the engineering aspect. Let's go through some of the examples.

### Amazon ML

TODO

### Azure AI

TODO

### Google AI Platform

TODO
