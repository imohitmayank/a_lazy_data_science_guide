LSTM, GRU and RNN
========================

## Introduction

- LSTM, GRU or RNN are a type of recurrent layers. They were the state-of-the-art neural network models for text related applications before the transformers based models. They can be used to process any sequential data like timeseries, text, audio, etc. RNN is the simpler of the lot, where as LSTM and GRU are the "gated" (and so a little complex) version of RNN. These gates helps LSTM and GRU to mitigate some of the problems of RNN like exploding gradient. 
- The most basic unit in these layers is a "cell", which is repeated in a layer - equal to the number of token size. For example, if we want to process a text with 150 words (with word level tokenization), we need to perceptually attach 150 cells one after the another. Hence, each cell process one word and passes on the representation of the word (hidden state value) to the next cell in line.
- Starting with RNN, the flow of data and hidden state inside the RNN cell is shown below.

```{figure} /imgs/nlp_rnn.png
---
height: 300px
---
Cell of RNN layer. {cite}`practical_guide_to_rnn_and_lstm`
```

- As shown in the figure, `x^t` is the input token at time `t`, `h^{t-1}` is the hidden state output from the last step, `y^t` and `h^t` are two notations for the hidden state output of time `t`. The formulation of an RNN cell is as follow, 

$$
h^{t}=g\left(W_{h h} h^{t-1}+W_{h x} x^{t}+b_{h}\right) \\
y^{t}=h^{t}
$$

- LSTM cells on the other hand is more complicated and is shown below.

```{figure} /imgs/nlp_lstm.png
---
height: 200px
---
LSTM cell. {cite}`lstm_understanding`
```

```{note}
Python libraries take liberty in modifying the architecture of the RNN and LSTM cells. For more details about how these cells are implemented in Keras, check out {cite}`practical_guide_to_rnn_and_lstm`. 
```

- Finally, a GRU cell looks as follow, 

```{figure} /imgs/nlp_gru.png
---
height: 200px
---
GRU cell with formulae. {cite}`lstm_understanding`
```

- While RNN is the most basic recurrent layer, LSTM and GRU are the de facto baseline for any text related application. There are lots of debate over which one is better, but the answer is usually fuzzy and it all comes down to the data and use case. In terms of tunable parameter size the order is as follow - `RNN < GRU < LSTM`. That said, in terms of learing power the order is -  `RNN < GRU = LSTM`. 
- Go for GRU if you want to reduce the tunable parameters but keep the learning power relatively similar. That said, do not forget to experiment wth LSTM, as it may suprise you once in a while.

## Code

### Using BiLSTM (for regression)

```{code-block} python
---
lineno-start: 1
---
"""Sample code for recurrent layer based models
The model code is at line 38-41; rest are fillers and prerequisities
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
- Understanding LSTM Networks {cite}`lstm_understanding`

```{bibliography}
:filter: docname in docnames
```
