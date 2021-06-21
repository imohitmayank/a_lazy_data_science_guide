LSTM, GRU and RNN
========================
----------

## Introduction

- LSTM, GRU or RNN are a type of recurrent layers. They were the SotA before the transformers based models.

## Code

### Using BiLSTM for regression

```{code-block} python
---
lineno-start: 1
---
"""Sample code for recurrent layer based models
The model code is at line 42; rest are fillers and prerequisities
"""

# ---- Imports ----
import keras

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional, Embedding

# ---- Data loading ----
# contains a list or series of sentences
# train = ...
# test =  ...
# combined = train['text'] + test['text']

# ---- Data processing ----
# set max vocab size
vocab = 10000
# create tokenizer instances  
tokenizer = Tokenizer(num_words=vocab, oov_token=0)
# fit the tokenizer on text sequences
tokenizer.fit_on_texts(combined)
# tokenize the complete data
sequence_combined = tokenizer.texts_to_sequences(combined)
# get the max len
max_len = max([len(x) for x in sequence_combined])
# tokenize the train data
train_sequences = tokenizer.texts_to_sequences(train['text'])
# add padding to the data
padded_train_sequence = pad_sequences(train_sequences, maxlen=max_len, dtype='int32', padding='pre', truncating='pre', value=0)

# ---- Model ----
model = Sequential()
# encoder
model.add(keras.Input(shape=(padded_train_sequence.shape[1], ))) # input layer
model.add(Embedding(vocab, 256)) # embedding layer
model.add(Bidirectional(LSTM(256))) # biLSTM layer
# decoder
model.add(Dense(256, kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(8, kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='linear'))
# summary
model.summary()

# ---- Compile and Train ----
# callbacks
earlystopping = keras.callbacks.EarlyStopping(monitor='loss', patience=3)
# compile
model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])
# fit
history = model.fit(padded_train_sequence, train['y'], epochs=100, batch_size=16, verbose=2, callbacks=[earlystopping])

# ---- Prediction ----
# prepare testing data
test_sequences = tokenizer.texts_to_sequences(test['text'])
test_padded_sequences = pad_sequences(test_sequences, maxlen=max_len, dtype='int32', padding='pre',truncating='pre', value=0)
# perform prediction
y_pred = model.predict(test_padded_sequences)
```

## Additional materials
- A practical guide to RNN and LSTM in Keras {cite}`practical_guide_to_rnn_and_lstm`
- Guide to Custom Recurrent Modeling in Keras {cite}`Guide_to_custom_recurrent_modeling`


```{bibliography}
:filter: docname in docnames
```
