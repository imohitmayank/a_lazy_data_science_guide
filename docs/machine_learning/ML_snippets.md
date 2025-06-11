Machine Learning / Deep Learning Snippets
========================

Sharing some of the most widely used and arguably not *so famous* Machine Learning snippets 😉

## Feature importance

- Feature importance calculation is an important technique to identify the  features which "helps" with the downstream classification or regression tasks. 

- Sklearn provides several options to infer the importance of a feature. Most importantly, many model automatically computed the importane and store it in `model.feature_importances_`, after you call `.fit()` 

- As an example, lets take the text based classification task and try to infer the following, 
  - **Part 1:** First use `CountVectorizer` for feature engineering and `ExtraTreesClassifier` for classification. 
  - **Part 2:** Show the top N features.
  - **Part 3:** Show evidence of a feature (by value count over different class labels)

- Following dataset based assumtions have been made,  
  - We assume `x_train` and `y_train` contains the a list of sentences and labels repectively.
  - We assume a pandas dataframe of name `train_df` is present which contains `x_train` and `y_train` as columns with name `title` and `label` respectively. 

``` python linenums="1"
# import
import random
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import CountVectorizer

# PART 1: Train the model
# -----------------------
# variables
max_features = 10000

# get features
vectorizer = CountVectorizer(max_features=max_features)
features = vectorizer.fit_transform(x_train)

# model
model = ExtraTreesClassifier(random_state=1)
model.fit(features, y_train)


# PART 2: View top features
# -----------------------

top_n = 10 # no of top features to extract 
feature_imp_indices = model.feature_importances_.argsort()[-top_n:][::-1]
feature_importance = pd.DataFrame({'score': model.feature_importances_[feature_imp_indices], 
                                  'feature': np.array(vectorizer.get_feature_names())[feature_imp_indices],
                                  'indices': feature_imp_indices})
feature_importance # a pandas dataframe of top_n features


# PART 3: View individual feature's evidence
# -----------------------
index = 2282 # the feature's index 
# the label's distribution if this word is present in sentence
x_train.iloc[np.where(features[:, index].todense() >= 1)[0]]['label'].value_counts()
```

## Cross validation

- Cross validation is a technique in which at each iteration you create different split of train and dev data. At each such iteration, we train he model on the train split and validate on the remaining split. This way, event with small training data, we can perform multiple fold of validation.
- If you repeat this operation (for $N$ iterations) over the complete data such that (1) each data point belonged to the dev split at most once, (2) each data point belonged to train split $N-1$ times - its cross-validation.   
- I have used Stratified K-Folds cross-validator, you can use any function from the complete list mentioned here - [Sklearn Model selection](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection)

``` python linenums="1"
# import =======
from sklearn.model_selection import StratifiedKFold

# code =============
# split the dataset into K fold test
def split_dataset(dataset, return_fold=0, n_splits=3, shuffle=True, random_state=1):
    """
    dataset: pandas dataframe
    return_fold: the fold out of `n_split` fold, which is to be returned
    n_splits: # cross fold
    """
    # defined the KFOld function
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    # defined the dataset
    X = dataset
    y = dataset['class'] # label/class

    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        if return_fold == i:
            return dataset.loc[train_index], dataset.loc[test_index]

# example call
if __name__ == '__main__':
    # read the dataset
    df = pd.read_csv("....")
    # get one specific fold out of
    train, test = split_dataset(dataset=df, return_fold=0, n_splits=3)
    # run for all folds
    for fold in range(n_splits):
        train, test = split_dataset(dataset=df, return_fold=fold, n_splits=n_splits)
        # <perform actions here...>
```

## Hyper-parameter tuning

- Below is an example of hyperparameter tuning for SVR regression algorithm. There we specify the search space i.e. the list of algorithm parameters to try, and for each parameter combination perform a 5 fold CV test. Refer for more details - [Sklearn Hyperparameter tuning](https://scikit-learn.org/stable/modules/grid_search.html) and [Sklearn SVR Algorithm](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)

``` python linenums="1"
# import =======
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error

# DATA LOAD ============
train_data = ...  # load the features and target on which to train

# SEARCH SPACE ============
search_space = [{'kernel': ['poly', 'rbf', 'sigmoid'],
               'C': [1, 10, 100], 'epsilon': [10, 1, 0.1, 0.2, 0.01]}]

# TUNING ============
scorer = make_scorer(mean_squared_error, greater_is_better=False)
svr_gs = GridSearchCV(SVR(), search_space, cv = 5, scoring=scorer, verbose=10, n_jobs=None)
svr_gs.fit(train_data['features'], train_data['target'])


# PRINT RESULT ============
parameter_result = []
print("Grid scores on training set:")
means = svr_gs.cv_results_['mean_test_score']
stds = svr_gs.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, svr_gs.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"% (mean, std * 2, params))
    parameter_result.append({'mean': abs(mean), 'std': std, **params})

# SELECT BEST PARAMETERS ============
# select the settings with smallest loss
parameter_result = pd.DataFrame(parameter_result)
parameter_result = parameter_result.sort_values(by=['mean'])
best_settings = parameter_result.head(1).to_dict(orient='records')[0]

# FIT WITH BEST PARAMETERS ============
SVRModel = SVR(C=best_settings['C'],
            epsilon=best_settings['epsilon'],
            kernel= best_settings['kernel'])
SVRModel.fit(train_data['features'], train_data['target'])
```

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

- We need GPUs for deep learning, and before we start training or inference it's a good idea to check if GPU is available on the system or not. 
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

- You can even use deep learning frameworks like Pytorch to check for the GPU availability. In fact, this is where you will most probably use them.

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

## Monitor GPU usage

- If you want to continuously monitor the GPU usage, you can use `watch -n 2 nvidia-smi --id=0` command. This will refresh the `nvidia-smi` output every 2 second.

## HuggingFace Tokenizer

- Tokenizer is a pre-processing step that converts the text into a sequence of tokens. [HuggingFace tokenizer](https://huggingface.co/docs/transformers/main_classes/tokenizer) is a wrapper around the [tokenizers library](https://github.com/huggingface/tokenizers), that contains multiple base algorithms for fast tokenization.

``` python linenums="1"

# import
from transformers import AutoTokenizer

# load a tokenizer (use the model of your choice)
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# create an dummy text
text = "Hello my Name is Mohit"

# this will tokenize the text and return a list of tokens
tokenizer.tokenize(text)
# Output: ['hello', 'my', 'name', 'is', 'mo', '##hit']

# this will tokenize the text and return a list of token ids
tokenizer.encode(text)
# Output: [101, 7592, 2026, 2171, 2003, 9587, 16584, 102]

# this will return the decoded text (from token ids)
tokenizer.decode(tokenizer.encode(text))
# Output: [CLS] hello my name is mohit [SEP]

# get the token and id details in key value pair
vocabulary = tokenizer.get_vocab()
# length of vocab here is 30522
# vocabulary['hello'] returns 7592
```

## Explore Model

- You can use the `summary` method to check the model's architecture. This will show the layers, their output shape and the number of parameters in each layer.

=== "Keras"
    ``` python linenums="1"
    # import
    from keras.models import Sequential
    from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

    # create a model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax))

    # print the model summary
    model.summary()
    ```

=== "PyTorch"
    ``` python linenums="1"
    # import
    import torch
    import torch.nn as nn

    # create a model

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.conv3 = nn.Conv2d(64, 64, 3, 1)
            self.fc1 = nn.Linear(1024, 64)
            self.fc2 = nn.Linear(64, 10)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2, 2)
            x = F.relu(self.conv3(x))
            x = x.view(-1, 1024)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)
    
    # create an instance of the model
    model = Net()
    # print the model summary
    print(model)
    ```

- To check the named parameters of the model and their dtypes, you can use the following code,

=== "PyTorch"
    ``` python linenums="1"
    print(f"Total number of names params: {len(list(model.named_parameters()))}")
    print("They are - ")
    for name, param in model.named_parameters():
        print(name, param.dtype)
    ```
<!-- ## Tensor operations

- Tensors are the building blocks of any Deep Learning project. Here, let's go through some common tensor operations,

=== "Pytorch"

    ``` python

    # create a tensor

    # remove all 1 sized dimensions
    tensor_array.squeeze() # Input: (1, 1, 192), Output: (, 192)

    # from numpy array to tensor

    # from tensor to numpy array
    tensor_array.detach().numpy()
    ```

!!! Note
    `tensor.detach()` and `torch.no_grad()` both are used to defined logic for which grad should not be computed. While `detach` is specifically for a tensor, `no_grad` turns off `required_grad` temporarily for any operations within the `with` block. *(Refer [StackOverflow Question](https://stackoverflow.com/questions/56816241/difference-between-detach-and-with-torch-nograd-in-pytorch))* -->