!!! warning
    This page is still under progress. Please check back after some time or [contact me](mailto:mohitmayank1@gmail.com)

## Introduction

- Paraphrasing is a NLP task of reformatting the input text considering a set of objectives. The objectives could be,
  - **Adequecy:** *is the meaning of sentence preserved?* It can be measured by using a NLI model that could determine if the paraphrase is entailment of the original sentence or not. 
  - **Fluency:** *is the paraphrase fluent?* It can be measured by using fluency classification models.
  - **Diversity:** *how much different paraphrase is from original sentence?* It can be measured by computing [text similarity](text_similarity.md) between the original sentence and paraphrase. Lower the text similarity score, higher the diversity. We can use edit based algorithms like Levenshtein. 
  - **Tonality:** *has the tone of the parapharse changed?* It can be measured with tone detection models.
  - **Formality:** *has the writing style of the parapharse changed?* It can be measured with formality detection models.
  - **Length:** *has the paraphrase become more concise or detailed?* It can be measured by simple word or token based tokenizers. 

!!! Note
    The objectives could be one or multiple. Also, they could be applied while training or inference. Once way to combine existing models with objectives it was not trained on, is to perform multiple generations and pick the one with highest score in terms of objective metrics. 

- While we will go through the programmer way of performing Paraphrasing, here are some of the free tools *(limited)* available online for Paraphrasing -- [Quillbot](https://quillbot.com/), [Paraphraser.io](https://www.paraphraser.io/), [Rephrase.Info](https://www.rephrase.info/), [Outwrite](https://www.outwrite.com/), [Grammarly](https://app.grammarly.com/), etc.

## Code

### Parrot Paraphraser

- Usually a Seq2Seq or specifically large language models are either directly used or finetuned to perform Paraphrasing. This is because LLM have good with text generation and Paraphrasing can be easily converted to text generation task. 
- Parrot [2] is a Python package that use finetuned T5 model to perform Paraphrasing. Let's first see how to use the pacakge, 

``` python linenums="1"
# taken from Parrot Readme -- https://github.com/PrithivirajDamodaran/Parrot_Paraphraser
# import
from parrot import Parrot
import torch
import warnings
warnings.filterwarnings("ignore")

#Init models (make sure you init ONLY once if you integrate this to your code)
parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5")

phrases = ["Can you recommend some upscale restaurants in Newyork?",
           "What are the famous places we should not miss in Russia?"]

for phrase in phrases:
    para_phrases = parrot.augment(input_phrase=phrase, use_gpu=False)
    for para_phrase in para_phrases:
        print(para_phrase)
```

- Btw they also provide advanced set of options to tune the objective we discussed before. For this you only need to modify the parameters for the augment function. Example is shown below, 

``` python linenums="1"
para_phrases = parrot.augment(input_phrase=phrase,
                               use_gpu=False,
                               diversity_ranker="levenshtein",
                               do_diverse=False, 
                               max_return_phrases = 10, 
                               max_length=32, 
                               adequacy_threshold = 0.99, 
                               fluency_threshold = 0.90)
```

- If you want to directly use the finetuned model, try this

``` python linenums="1"
# install packages
!pip install transformers
!pip install -q sentencepiece

# import
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# load the tokenizers and model
tokenizer = AutoTokenizer.from_pretrained("prithivida/parrot_paraphraser_on_T5")
model = AutoModelForSeq2SeqLM.from_pretrained("prithivida/parrot_paraphraser_on_T5")

# for a phrase get the tokenised input ids
input_ids = tokenizer("paraphrase: Can I call you after I am done with this thing I am working on?", 
                     return_tensors="pt").input_ids
# use the input ids to generte output
outputs = model.generate(input_ids, max_new_tokens=10, do_sample=False, num_beams=1, length_penalty=5)
# decode the output token ids to text
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
## Output --> 
## Can I call you after I've finished this
```

### Finetuning T5 as Paraphraser

- Any LLM can be used for Paraphrase generation by zero-shot for conservating accuracy. If you want to better result, finetune it on datasets like [PAWS](https://huggingface.co/datasets/paws), [MSRP](https://huggingface.co/datasets/HHousen/msrp), etc. A more detailed list of dataset is presented [here](https://www.sbert.net/examples/training/paraphrases/README.html). Here we will try to finetune [T5](T5.md), 

``` python linenums="1"
# install
!pip install -q simplet5
!pip install -q datasets

# import
import pandas as pd
from simplet5 import SimpleT5
from datasets import load_dataset

# load datasets
msrp = load_dataset("HHousen/msrp")
paws = load_dataset("paws", 'labeled_final')

# prepare dataset
def clean_msrp_paws_dataset(data):
    df = pd.DataFrame(data)
    df = df[df['label']==1]
    df['source_text'] = f'Paraphrase: ' + df['sentence1']
    return df

# clean both train and test data
train_msrp_data = clean_msrp_paws_dataset(msrp['train'])
test_msrp_data = clean_msrp_paws_dataset(msrp['test'])

# clean_msrp_paws_dataset
train_paws_data = clean_msrp_paws_dataset(paws['train'])
test_paws_data = clean_msrp_paws_dataset(paws['test'])
validation_paws_data = clean_msrp_paws_dataset(paws['validation'])

# combine the individual splits of datasets
msrp_dataset = pd.concat([train_msrp_data, test_msrp_data])
paws_dataset = pd.concat([train_paws_data, test_paws_data, validation_paws_data])

# combine the datasets
df1 = msrp_dataset[['source_text', 'sentence2']]
df1 = df1.rename(columns={'source_text':'source_text', 'sentence2': 'target_text'})
df2 = paws_dataset[['source_text', 'sentence2']]
df2 = df2.rename(columns={'source_text':'source_text', 'sentence2': 'target_text'})
train_data = pd.concat([df1, df2])

# Train
# load model
model = SimpleT5()
model.from_pretrained(model_type="t5", model_name="t5-small")

# train model
model.train(train_df=train_data,
            eval_df=train_data.head(100),  # dummy eval, in reality keep some held-out samples as validation/test
            source_max_token_len=300, 
            target_max_token_len=200, 
            batch_size=4, 
            max_epochs=20, 
            outputdir = "outputs",
            use_gpu=True
            )

# Inference
# last_epoch_model = "/content/outputs/simplet5-epoch-1-train-loss-1.5314-val-loss-1.2911" # put the name here
# model.load_model("t5", last_epoch_model, use_gpu=True)
# model.predict("Paraphrase: He is  going to USA to visit his friend")
```

## References

[1] [Paraphrase Generation: A Survey of the State of the Art](https://aclanthology.org/2021.emnlp-main.414.pdf)

[2] [Parrot Paraphraser](https://github.com/PrithivirajDamodaran/Parrot_Paraphraser)

