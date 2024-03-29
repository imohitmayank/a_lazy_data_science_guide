BERT
========================

## Introduction

- **BERT** stands for **B**idirectional **E**ncoder **R**epresentations from **T**ransformers. (*devlin2019bert*)
- Basically, it is a modification of Transformers (*vaswani2017attention*), where we just keep the encoder part and discard the decoder part.

<figure markdown> 
    ![](../imgs/nlp_transformers.png)
    <figcaption>Transformer architecture. BERT is the left part i.e. encoder part. (*vaswani2017attention*)</figcaption>
</figure>

- At the time of release, it obtained state-of-the-art results on eleven natural language processing tasks. To quote the paper, "_[paper pushed] the GLUE score to 80.5% (7.7% point absolute improvement), MultiNLI accuracy to 86.7% (4.6% absolute improvement), SQuAD v1.1 question answering Test F1 to 93.2 (1.5 point absolute improvement) and SQuAD v2.0 Test F1 to 83.1 (5.1 point absolute improvement)._"
- The major motivation behind BERT is to handle the limitation of the existing language models which are unidirectional in nature. This means that they only consider text left to right for sentence level inference. BERT on the other hand, allows tokens to attend to both sides in self-attention layer. This is one of the major reason for it high performance.

<figure markdown> 
        ![](../imgs/nlp_bert_elmo_gpt.png)
        <figcaption>Differences in pre-training model architectures. BERT uses a bidirectional Transformer. OpenAI GPT uses a left-to-right Transformer. ELMo uses the concatenation of independently trained left-to-right and right-toleft LSTMs to generate features for downstream tasks. Among the three, only BERT representations are jointly conditioned on both left and right context in all layers. In addition to the architecture differences, BERT and OpenAI GPT are fine-tuning approaches, while ELMo is a feature-based approach.. (*devlin2019bert*)</figcaption>
        </figure>

- The most fascinating feature of BERT is that it is super easy to use it for a large number of NLP tasks. The idea is to take the pretrained BERT model and later fine tune it for the specific task. The pre-trained model is trained on a large corpus in a unsupervised manner, hence the model learns the generic representations of the tokens from large corpus of text. This makes it easy to later fine tune it for any other NLP task, as the model comes pretrained with large context about the language, grammar and semantic representations.

<figure markdown> 
        ![](../imgs/nlp_bert_applications.png)
        <figcaption>Illustrations of Fine-tuning BERT on Different Tasks. (*devlin2019bert*)</figcaption>
        </figure>

- Training BERT is an interesting paradigm in itself. The original paper proposed two unsupervised methods for training,
  1. **Masked LM (MLM)**: Where some percentage (15%) of the input tokens are masked at random, and then the model tries to predict those masked tokens. They created a special token `[MASK]` for this purpose.
  2. **Next Sentence Prediction (NSP)**: Where two sentences A and B are chosen such that, 50% of the time B is the actual next sentence that follows A (labelled as `IsNext`), and 50% of the time it is a random sentence from the corpus (labelled as `NotNext`). The model is trained to predict if the second sentences follow the first or not.

## Analysis

### BERT output and finetuning (unsupervised)

- An analysis on the selection of suitable BERT output and the advantage of fine-tuning *(unsupervised learning on unlabeled data on tasks like MLM)* the model was done. The report provides following performance table comparing different experiments. Complete article [here](https://towardsdatascience.com/tips-and-tricks-for-your-bert-based-applications-359c6b697f8e).

| Exp no | Model name                       | F1 macro score | Accuracy |
|--------|----------------------------------|----------------|----------|
| 1      | Pooler output                    | 64.6%          | 68.4%    |
| 2      | Last hidden state                | 86.7%          | 87.5%    |
| 3      | Fine-tuned and Pooler output     | 87.5%          | 88.1%    |
| 4      | Fine-tuned and last hidden state | 79.8%          | 81.3%    |

- It also answers following questions,
  - **Should I only use CLS token or all token's output for sentence representation?**  Well, it depends. From the experiments, it seems if you are fine-tuning the model, using the pooler output will be better. But if there is no fine-tuning, the last hidden state output is much better. Personally, I will prefer the last hidden state output, as it provides comparative result without any additional compute expensive fine-tuning. 
  - **Will fine-tuning the model beforehand increase the accuracy?** A definite yes! Exp 3 and 4 reports higher score than Exp 1 and 2. So if you have the time and resource (which ironically is not usually the case), go for fine-tuning!

### Is BERT a Text Generation model?

- Short answer is no. BERT is not a text generation model or a language model because the probability of the predicting a token in masked input is dependent on the context of the token. This context is bidirectional, hence the model is not able to predict the next token in the sequence accurately with only one directional context *(as expected for language model)*.
- Several analysis were done on the text generation prowess of BERT model. One such analysis is presented [in this paper](https://arxiv.org/abs/1902.04094?context=cs). Here the authors presents BERT as markov random field language model. Then after some errors were pointed out wrt paper, the authors corrected the claim and suggested BERT is a non-equilibrium language model ([here](https://sites.google.com/site/deepernn/home/blog/amistakeinwangchoberthasamouthanditmustspeakbertasamarkovrandomfieldlanguagemodel))

!!! Tip
    Do you know that during mask prediction, BERT model predicts some tokens for `[PAD]` tokens as well. This is true for sentences that are smaller than the max length of the model and hence require padding. In a sense, this is kind of text generation, where you just provide the sentence and the model predicts the next token till the max length. But as expected the prediction is not that accurate.


### BERT for sentence representation?

- One question usually asked is that - "Can we use BERT to generate meaningful sentence representations?" The answer is "No". Don't get me wrong, while it is possible to use BERT to generate sentence representations, but the key word here is "meaningful". One of the way to do this is to pass one sentence to the model and get the representation for fixed `[CLS]` token as sentence representation. But as shown in [2], this common practice yields bad sentence embedding, often even worse than Glove embeddings *(which was introduced in 2014)*!
- The major problem here is the pre-training strategy used to train BERT. While it is good for downstream tasks like classification, it's not that good for generating generic representations. This is because for correct sentence representation, we want the embeddings of similar sentences closer to each other and dissimilar sentences to be further apart. And this is not what happens during BERT pretraining. To cater to this issue, we will have to further finetune the model. And in fact, this is where BERT shines again, as with minimal training *(sometimes even for 20 mins with <1000 samples)* we can expect good results.
- One of the ways to finetune for sentence represenration is to use triplet loss. For this, we prepare a dataset with a combination of `(anchor, positive, negative)` sentences. Here anchor is the base sentence, positive is the sentence that is similar to the anchor sentence, and negative is the sentence that is dissimilar to the anchor sentence. The model is trained to "bring" the representation of `(anchor, positive)` closer and `(anchor, negative)` apart. The loss is defined below, where $s_*$ is the respective sentence representation and $\epsilon$ is the margin.

$$triplet loss = max( \parallel s_a - s_p \parallel - \parallel s_a - s_n \parallel + \epsilon , 0)$$

## Code

### Pretrained BERT for Sentiment Classification

- The code contains the `Dataset` and `Dataloader` as well, which can be referred for any fine tuning task.  
- Download dataset from [IMDB 50k review](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

``` python linenums="1"
# helper
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# for BERT model
from transformers import BertTokenizer, BertModel
from transformers import BertForSequenceClassification

# for DL stuff
import torch
import pytorch_lightning as pl
from torch.nn import Softmax, Linear
from torchmetrics import Accuracy, F1
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.loggers import WandbLogger


model_name = 'bert-base-uncased'

# load dataset and sample 10k reviews.
df = pd.read_csv("../input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv")
df = df.sample(10000, random_state=1)

# divide into test and train
X_train, X_test, y_train, y_test = \
          train_test_split(df['review'].tolist(), df['sentiment'].tolist(),
          shuffle=True, test_size=0.33, random_state=1, stratify=df['sentiment'])

# define dataset with load and prep functions. Pass all the data at a time.
def squz(x, dim=0):
    return torch.squeeze(x, dim)

class IMDBDataset(Dataset):
    def __init__(self, sentences, labels, max_length=512, model_name='bert-base-uncased'):
        # var
        self.sentences = sentences
        self.labels = [['positive', 'negative'].index(x) for x in labels]
        self.max_length = max_length
        # tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        # Select sample
        sentence = self.sentences[index]
        label = self.labels[index]
        # Load data and get label
        X = self.tokenizer(sentence, padding="max_length", truncation=True,
                        max_length=self.max_length, return_tensors="pt")
        X = {key: squz(value) for key, value in X.items()}
        y = label
        # return
        return X, y

# init the train and test dataset
train_dataset = IMDBDataset(X_train, y_train)
test_dataset = IMDBDataset(X_test, y_test)
# create the dataloader
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)

# define BERT model
class BERT_pooler_output(pl.LightningModule):
    def __init__(self, model_name='bert-base-uncased'):
        super().__init__()
        # model and layers
        self.BERTModel = BertModel.from_pretrained(model_name)
        self.linear1 = Linear(768, 128)    
        self.linear2 = Linear(128, 2)
        self.softmax = Softmax(dim=1)
        self.relu = torch.nn.ReLU()
        # loss
        self.criterion = torch.nn.CrossEntropyLoss()
        # performance
        self.accuracy = Accuracy()
        self.f1 = F1(num_classes=2, average='macro')

    def forward(self, x):
        # pass input to BERTmodel
        input_ids, attention_mask = x['input_ids'], x['attention_mask']
        bert_output = self.BERTModel(input_ids, attention_mask=attention_mask)
        output = bert_output.pooler_output     
        output = self.relu(self.linear1(output))
        output = self.linear2(output)
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self.forward(x)
        loss = self.criterion(x_hat, y)
        acc = self.accuracy(x_hat.argmax(dim=1), y)
        f1 = self.f1(x_hat.argmax(dim=1), y)
        self.log('loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('f1', f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self.forward(x)
        acc = self.accuracy(x_hat.argmax(dim=1), y)
        f1 = self.f1(x_hat.argmax(dim=1), y)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_f1', f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        # freezing the params of BERT model
        for name, param in self.named_parameters():
            if 'BERTModel' in name:
                param.requires_grad = False
        # define the optimizer
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()),
                                                                    lr=1e-3)
        return optimizer


# init model
model = BERT_pooler_output()
# init trainer
trainer = pl.Trainer(gpus=1, max_epochs=3)
# train the model
trainer.fit(model, train_dataloader, test_dataloader)
```

### Fine tuning the BERT model

- Fine tuning could include training BERT on one or many of the proposed unsupervised tasks.
- Here, we will train the BERT on MLM (Masked language modeling) task.
- This includes masking some tokens of input and BERT predicting the token based on the context tokens.
- Referenced from [this video of James Briggs](https://youtu.be/R6hcxMMOrPE).

``` python linenums="1"
# IMPORT =========
import pandas as pd
from tqdm import tqdm
import numpy as np
import numpy as np

# for deep learning
import torch
import torch.nn as nn
import torch.optim as optim

# load BERT model
from transformers import AdamW
from transformers import BertTokenizer, BertForMaskedLM

# MODEL LOAD =========
model_path = "bert-base-uncased" # if local copy is not present
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForMaskedLM.from_pretrained(model_path)


# DATA PREP 1 =========
df = pd.read_csv("file_with_text.csv")

# tokenize
inputs = tokenizer(df['review'].tolist(), return_tensors='pt', max_length=512,
                   truncation=True, padding='max_length')
inputs['labels'] = inputs.input_ids.detach().clone()

# create random array of floats with equal dimensions to input_ids tensor
rand = torch.rand(inputs.input_ids.shape)
# create mask array - mask tokens except special tokens like CLS and SEP
mask_ratio = 0.3
mask_arr = (rand < mask_ratio) * (inputs.input_ids != 101) * \
           (inputs.input_ids != 102) * (inputs.input_ids != 0)

# get the indices where to add mask
selection = []
for i in range(inputs.input_ids.shape[0]):
    selection.append(
        torch.flatten(mask_arr[i].nonzero()).tolist()
    )

# add the mask
for i in range(inputs.input_ids.shape[0]):
    inputs.input_ids[i, selection[i]] = 103

# DATA PREP 2 - DATALOADER =========
# define dataset class
class IMDBDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids)

# create instance
dataset = IMDBDataset(inputs)
loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

# PRE_TRAIN =============
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# and move our model over to the selected device
model.to(device)
# activate training mode
model.train()

# initialize optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# TRAIN =====================
epochs = 20
for epoch in range(epochs):
    # setup loop with TQDM and dataloader
    loop = tqdm(loader, leave=True)
    for batch in loop:
        # initialize calculated gradients (from prev step)
        optimizer.zero_grad()
        # pull all tensor batches required for training
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        # process
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        # extract loss
        loss = outputs.loss
        # calculate loss for every parameter that needs grad update
        loss.backward()
        # update parameters
        optimizer.step()
        # print relevant info to progress bar
        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())

# SAVE MODEL =====================
model.save_pretrained("bert_finetuned_on_text/")
tokenizer.save_pretrained("bert_finetuned_on_text/")
```

### BERT output for sentence level inference

- BERT provides `pooler_output` and `last_hidden_state` as two potential "_representations_" for sentence level inference.
- `pooler_output` is the embedding of the `[CLS]` special token. In many cases it is considered as a valid representation of the complete sentence.

``` python linenums="1"
BERTModel = BertModel.from_pretrained('bert-base-uncased')
bert_output = BERTModel(input_ids, attention_mask=attention_mask)
output = bert_output.pooler_output
```

- `last_hidden_state` contains the embeddings of all tokens in the sentence from the last hidden state. We can apply permutation invariant methods (like max, mean or sum) to aggregate the embeddings into a single sentence representation.

``` python linenums="1"
BERTModel = BertModel.from_pretrained('bert-base-uncased')
bert_output = BERTModel(input_ids, attention_mask=attention_mask)
output = squeeze(torch.matmul(attention_mask.type(torch.float32).view(-1, 1, 512), bert_output['last_hidden_state']), 1)
```

!!! Tip
    Consider finetuning the BERT model *(triplet loss)* further to generate meaningful sentence representation, as pretrained BERT model is even worse than Glove embeddings [2]. For more details look at [this analysis](BERT.md#bert-for-sentence-representation) or use [S-BERT package](https://www.sbert.net/examples/training/sts/README.html) to finetune.
## References

[1] [Jay Alammar's blog "The Illustrated BERT, ELMo, and co. (How NLP Cracked Transfer Learning)"](the_illustrated_bert*(https://jalammar.github.io/illustrated-bert/))

[2] [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)
