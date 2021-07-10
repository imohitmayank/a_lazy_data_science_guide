Neural Network Design
=========================

- To solve a problem using Neural networks, it is important to understand the science or art of NN design.
- Here we will discuss some interesting topic which could he helpful in the quest to create the ultimate NN for your task.

## Callbacks

- Callbacks are the hooks that you can attach to your deep learning training or validation process.
- It can be used to affect the training process from simple logging metric to even terminating the training in case special conditions are met.
- Below is an example of `EarlyStopping` and `ModelCheckpoint` callbacks.

````{tabbed} Keras
```{code-block} python
---
lineno-start: 1
---
# import
import keras

# Step 1: Defining the callbacks
#------------------------------
# define early stopping callback
earlystopping = keras.callbacks.EarlyStopping(monitor='loss', patience=10)

# define model checkpoint callback
checkpoint = keras.callbacks.ModelCheckpoint("checkpoint_model_{epoch:02d}.hdf5",
        monitor='loss', verbose=1, save_best_only=False, mode='auto', period=10)

# Step 2: Assigning the callback
#------------------------------
# fit the model
history = model.fit(train_data_gen, # training data generator
#                     .... # put usual code here
                    callbacks=[checkpoint, earlystopping]
                   )
```
````

## Mean pooling

- References this [stackoverflow answer](https://stackoverflow.com/questions/36428323/lstm-followed-by-mean-pooling/64630846#64630846).

````{tabbed} Keras
```{code-block} python
---
lineno-start: 1
---
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
````

Some points to consider,
- The `pool_size` should be equal to the step/timesteps size of the recurrent layer.
- The shape of the output is (`batch_size`, `downsampled_steps`, `features`), which contains one additional `downsampled_steps` dimension. This will be always 1 if you set the `pool_size` equal to timestep size in recurrent layer.

## Dataset and Dataloader

- Dataset can be downloaded from [Kaggle](Dataset: https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).

````{tabbed} PyTorch
```{code-block} python
# Import
# -----------
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Load and prepare IMDB dataset
#-------------------------------
# load
df = pd.read_csv("imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv")

# divide into test and train
X_train, X_test, y_train, y_test = train_test_split(df['review'].tolist(), df['sentiment'].tolist(), shuffle=True,
                                                    test_size=0.33, random_state=42, stratify=df['sentiment'])

# Dataset
# -----------
# define dataset class which takes care of the dataset preparation before passing to model.
# Class takes all data at once (__init__) and define functions to fetch one data at a time (__getitem__)
class IMDBDataset(Dataset):
    def __init__(self, sentences, labels, max_length=150):
        'constructor'
        # var
        self.sentences = sentences
        self.labels = [['positive', 'negative'].index(x) for x in labels]
        self.max_length = max_length
        # tokenizer
        self.tokenizer = ... # some tokenizer

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.sentences)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        sentence = self.sentences[index]
        label = self.labels[index]
        # Load data and get label
        X = self.tokenizer(sentence, ...) # tokenize one data sample
        y = label
        # return
        return X, y

# Init and Dataloader
# ---------------------
# init the train and test dataset
train_dataset = IMDBDataset(X_train, y_train)
test_dataset = IMDBDataset(X_test, y_test)
# create the dataloader
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)
```
````

## Freeze Layers

- Example on how to freeze certain layers while training

````{tabbed} PyTorch lightning
```{code-block} python
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
````
