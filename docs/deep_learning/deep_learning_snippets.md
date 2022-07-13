Deep Learning Snippets
=========================

Sharing some of the most widely used and still not *famous* Deep Learning snippets. :smile:

## Callbacks

- Callbacks are the hooks that you can attach to your deep learning training or validation process.
- It can be used to affect the training process from simple logging metric to even terminating the training in case special conditions are met.
- Below is an example of `EarlyStopping` and `ModelCheckpoint` callbacks.

=== "Keras"
    ``` python linenums="1"
    # fit the model
    history = model.fit(train_data_gen, # training data generator
    #                     .... # put usual code here
                        callbacks=[checkpoint, earlystopping]
                    )
    ```

## Mean pooling

- References this [stackoverflow answer](https://stackoverflow.com/questions/36428323/lstm-followed-by-mean-pooling/64630846#64630846).

=== "Keras"
``` python linenums="1"
# import
import numpy as np
import keras

# create sample data
A=np.array([[1,2,3],[4,5,6],[0,0,0],[0,0,0],[0,0,0]])
B=np.array([[1,3,0],[4,0,0],[0,0,1],[0,0,0],[0,0,0]])
C=np.array([A,B]).astype("float32")
# expected answer (for temporal mean)
np.mean(C, axis=1)

"""
The output is
array([[1. , 1.4, 1.8],
       [1. , 0.6, 0.2]], dtype=float32)
Now using AveragePooling1D,
"""

model = keras.models.Sequential(
        tf.keras.layers.AveragePooling1D(pool_size=5)
        )
model.predict(C)

"""
The output is,
array([[[1. , 1.4, 1.8]],
       [[1. , 0.6, 0.2]]], dtype=float32)
"""
```

- Some points to consider,
  - The `pool_size` should be equal to the step/timesteps size of the recurrent layer.
  - The shape of the output is (`batch_size`, `downsampled_steps`, `features`), which contains one additional `downsampled_steps` dimension. This will be always 1 if you set the `pool_size` equal to timestep size in recurrent layer.

## Dataset and Dataloader

- Dataset can be downloaded from [Kaggle](Dataset: https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).

=== "PyTorch"
``` python linenums="1"
# init the train and test dataset
train_dataset = IMDBDataset(X_train, y_train)
test_dataset = IMDBDataset(X_test, y_test)
# create the dataloader
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)
```

## Freeze Layers

- Example on how to freeze certain layers while training

=== "PyTorch lightning"
``` python linenums="1"
{code-block} python
# Before defining the optimizer, we need to freeze the layers
# In pytorch lightning, as optimizer is defined in configure_optimizers, we freeze layers there.
def configure_optimizers(self):
    # iterate through the layers and freeze the one with certain name (here all BERT models)
    # note: the name of layer depends on the varibale name
    for name, param in self.named_parameters():
        if 'BERTModel' in name:
            param.requires_grad = False
    # only pass the non-frozen paramters to optimizer
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=1e-3)
    # return optimizer
    return optimizer
```

## Check for GPU availability 

- We need GPUs for deep learning, and before we start training or inference it's a good idea to check if GPU is availbale on the system or not. 
- The most basic way to check for GPUs (if it's a NVIDIA one) is to run `nvidia-smi` command. It will return a detailed output with driver's version, cuda version and the processes using GPU. [Refer this](https://medium.com/analytics-vidhya/explained-output-of-nvidia-smi-utility-fc4fbee3b124) for more details on individual components.


``` shell
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 435.21       Driver Version: 435.21       CUDA Version: 10.1     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce MX110       Off  | 00000000:01:00.0 Off |                  N/A |
| N/A   43C    P0    N/A /  N/A |    164MiB /  2004MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0      6348      G   /usr/lib/xorg                                 53MiB |
|    0     13360      G   ...BBBBBaxsxsuxbssxsxs --shared-files         28MiB |
+-----------------------------------------------------------------------------+
```

- You can even use deep learning frameworks like Pytorch to check for the GPU availbaility. In act, this is where you will most probably use them.

``` python linenums="1"
# import 
import torch
# checks
torch.cuda.is_available()
## Output: True
torch.cuda.device_count()
## Output: 1
torch.cuda.current_device()
## Output: 0
torch.cuda.device(0)
## Output: <torch.cuda.device at 0x7efce0b03be0>
torch.cuda.get_device_name(0)
## Output: 'GeForce MX110'
```