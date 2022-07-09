## Introduction

- Deep Learning models provide enhanced accuracy for majority of tasks, but they compromises with speed. This is highly relevant for industrial use case, where somtimes hundreds and thousands of inferences needs to be made in short amount of time. 
- From enginnering perspective, we could explore a couple of solutions such as,
  - **Multi Threading**: if your system performs some I/O calls or DB calls or 3rd party API polling, multi threading could be the way to improve inference speed. Basically, while one thread is busy perfoming asyncronous task, another task can take over and start running. 
  - **Multi Processing**: we can utilize multiple cores of a compute to create multiple processes, each working independently of each other. Note, it is memory (or GPU) consuming task, as if you create 4 workers (4 processes), OS will create 4 copies of your model! For a 1GB model, 4GB will be consumed just to keep the system up and running.
  - **Replication**: if you are using K8, just create multiple nodes of the code. It's like creating multiple copies of the code and running them independently on different computes. Note, this is cost consuming task, as each new replication needs a new compute which will increase the cost.
- All of the above proposed solutions have one major flaw -- while they can help with handling large number of inference call, they cannot improve the speed of each inference call. Because in all of the cases, we are using the same big and slow model. 
- And that's where model compression techniques comes into the picture, where the intuition is to enhance the speed of the inference (or training) with minimal compromise on the accuracy!

## Types of Model compression

- At a high level, there are following types of model compression. We will go through them one by one.
  - **Knowledge Distillation**: in these methods, we distill the learned information (or knowledge) from one neural network model (mainly larger) to another (smaller)
  - **Parameter Pruning and Sharing**: in these methods, we remove the non-essential parameters from a neural network with minimal to no effect on the overall performance. 

## Knowledge Distillation

- 