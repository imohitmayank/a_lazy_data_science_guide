Relation Extraction
===================


## Introduction

- Relation extraction (RE) is the process of identifying the relationships between entities in a text. Entities could be of multiple types such as person, location, organization, etc and they can be identified using Named Enitity Recognition (NER). 
- Let's understand RE with an example: `"Ram is the son of Shyam and Shyam is the son of Radhe"`. Here the entities are identified as: "Ram", "Shyam" and "Radhe". The relations could be `(Ram, son of, Shyam)`, `(Shyam, son of, Radhe)` and `(Ram, grandson of, Radhe)`.

## Code

### Using OpenNRE

- [OpenNRE](https://github.com/thunlp/OpenNRE) is an open source tool for relationship extraction. OpenNRE makes it easy to extract relationships from text. It is as simple as writing a few lines of code. 
- One example from their github repository is as follows, 

``` python linenums="1"
# import opennre
import opennre
# load the model
model = opennre.get_model('wiki80_cnn_softmax')
# infer for a text
model.infer({'text': 'He was the son of Máel Dúin mac Máele Fithrich, and grandson of the high king Áed Uaridnach (died 612).', 'h': {'pos': (18, 46)}, 't': {'pos': (78, 91)}})
# Output: ('father', 0.5108704566955566)
```

- At the time of writing they had following models available:


| model_name   | description  | 
|---------|:---------:|
|wiki80_cnn_softmax | trained on wiki80 dataset with a CNN encoder.|
|wiki80_bert_softmax | trained on wiki80 dataset with a BERT encoder.|
|wiki80_bertentity_softmax | trained on wiki80 dataset with a BERT encoder (using entity representation concatenation).|
|tacred_bert_softmax | trained on TACRED dataset with a BERT encoder.|
|tacred_bertentity_softmax | trained on TACRED dataset with a BERT encoder (using entity representation concatenation).|