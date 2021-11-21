# Compute and AI Services

- Gone are the days when we needed to buy high end devices to do literally anything. Currently there are plethora of services available online *(and many of them are free!)* that provide not only compute to use as you feel, but also generic AI services. 

- Let's look into some of the famous and widely used compute and AI services.

## CaaS: Compute as a Service

In this section we will cover some of the famous *(and with some hint of free)* platforms that provide compute-as-a-service (CaaS). These CaaS sometimes could be plain simple virtual machines, sometime they can be a cluster of nodes, while in other cases they can also be jupyter like coding environment. Let's go through some of the examples.

### Google Colab

#### Introduction

- Colaboratory or ["Colab"](https://colab.research.google.com/signup) in short, is a browser based jupyter notebook environment that is available for free. It requires no installation and even provides access to free GPU and TPU. 
- The main disadvantages of Colab is that you cannot run long-running jobs (limit to max 12 hrs), GPU is subject to availability and in case of consistent usage of Colab, it might take longer to get GPU access.
- Google provides Pro and Pro+ options which are paid subscriptions to Colab (10$ and 50$ per month, respectively). While it provides longer background execution time and better compute (among others), they do not guarantee GPU and TPU access all the time. Remember, Colab is not an alternative to a full-blown cloud computing environment. It's just a place to test out some ideas quickly.

#### Google Colab Snippets

##### Run tensorboard to visualize embeddings

- Taken from: [how-to-use-tensorboard-embedding-projector](https://stackoverflow.com/questions/40849116/how-to-use-tensorboard-embedding-projector)

``` python linenums="1"
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

##### Connect with Google Drive and access files

- This code will prompt you to provide authorization to access your Google Drive.

``` python linenums="1"
from google.colab import drive
drive.mount('/content/drive')
```

### Kaggle

Coming soon!

### DeepNote

Coming soon!

## MLaaS: Machine Learning as a Service

In this section we will cover some of the famous platforms that provide Machine learning-as-a-Service (MLaaS). These MLaaS take care of infrastructure related aspect of data holding, data preparing, model training and model deployment. On top of this, they provide a repository of classical ML algorithms that can be leveraged to create data science solutions. The idea is to make data science as a plug and play solution creation activity, as they take care of most of the engineering aspect. Let's go through some of the examples.

### AWS Sagemaker (Amazon)

- [AWS Sagemaker](https://aws.amazon.com/sagemaker/) is a cloud-based servies that helps data scientists with the complete lifecycle of data science project.
- They have specialised tools that cover following stages of data science projects, 
  - **Prepare**: It's the pre-processing step of the project. Some of the important services are "*Gound Truth*" that is used for data labeling/annotation and "*Feature Store*" that is used to provide consistence data transformation across teams and services like training and deployment.
  - **Build**: It's where an Data Scientists spends most of his time coding. "*Studio Notebooks*" provides jupyter notebooks that can be used to perform quick ideation check and build the model.
  - **Train & Tune**: It's where you can efficiently train and debug your models. "*Automatic Model Training*" can be used for hyper-parameter tuning of the model i.e. finding the best parameters that provides highest accuracy. "*Experiments*" can be used to run and track multiple experiments, its an absolute must if your projects requires multiple runs to find the best architecture or parameters. 
  - **Deploy & Manage**: The final stage, where you deploy your model for the rest of the world to use. "*One-Click Deployment*" can be used to efficiently deploy your model to the cloud. "*Model Monitor*" can be used to manage your model, like deleting, updating, and so on.
 
<figure markdown> 
        ![](../imgs/aws_sagemaker.png)
        <figcaption>Services provided by [AWS Sagemaker](https://aws.amazon.com/sagemaker/).</figcaption>
        </figure>

- AWS charges a premium for providing all of these features under a single umbrella. For a more detailed pricing information, you can estimate the cost using [this](https://aws.amazon.com/sagemaker/pricing/).

!!! hint
    As AWS Sagemaker is a costly affair, several DS teams try to find workarounds. Some of them are like using spot instances for training as they are cheaper & using AWS Lambda for deploying small models. 

### Azure ML Services (Microsoft)

Coming soon!

### Google AI Platform

Coming soon!