Named entity recognition
========================

## Introduction

- Named entity recognition (NER) is the process of identifying entities in the unstructured text, where entities could be objects, people, locations, organizations, etc. 
- NER's most basic building block consists of pair of `entity_type` and `entity_value`. Consider the following example,

```
## Statement
My name is Mohit, and I am from India. 
I am a Data Scientist and I will be leaving for my office around 9 AM.

## Entities
[{
    'entity_type': 'PERSON',
    'entity_value': 'Mohit',
},
{
    'entity_type': 'LOCATION',
    'entity_value': 'India',
}, 
{
    'entity_type': 'TIME',
    'entity_value': '9 AM',
}]
```

- The process of extracting entities could be done in two ways.
  - **Heuristic**: by identifying the entities based on the rules.
  - **Semantic**: by identifying the entities based on the semantics and context.

- Heuristic based approach is suited only for simple entities for which approximate rules can be created. Take for example EMAILID, PHONE_NUMBER, WEBSITE, etc. It should be easy enough to create regular expressions for such cases and hence heuristic approach could be applied. We can also apply part of speech based rules to extract certain entities.
- On the other hand, the Semantic approach is required where the cardinality of the entities is high and the context is required to extract the entities. For example, NAME, LOCATION, DESIGNATION, etc. For these cases, we usually train neural network based models that learn to consider the context and extract the entities.

!!! Note
    A good approach to creating your NER solution would be to segregate your entities into simple and complex, and then create either a heuristic or a semantic based solution or a combination of both. In short, it is not always suitable to directly go to fancy NN based semantic approaches - it could be unnecessary overkill.

- Remember the entity types are not set in stone and we can even train new models or finetune existing models on our own custom entities. 
- For this, in the Semantic-based approach, it's a good idea to finetune the existing model rather than to train a new one as it will require far fewer data.
- The amount of data required to finetune model depends on how similar the custom entities are with the existing entities. Consider the following cases, 
  - The model is pretrained to detect PERSON and now you want to finetune it to detect MALE_NAME and FEMALE_NAME. As this is just a lower granularity on the existing PERSON entity, a mere ~200 examples *(for each new entity type)* could give you good results.
  - On the other hand, if you now want to finetune a completely new entity like OBJECTIONS_TYPE, you may need ~500 examples. 

!!! Note
    Another thing to consider is the length of `entity_value`. With an increase in `entity_value` you may require more examples to get good accuracy results.

## Code

- There are lots of Python-based packages that provide open source NER models. Some of these packages are [Spacy](https://spacy.io/), [NLTK](https://www.nltk.org/), [Flair](https://github.com/flairNLP/flair), etc. While packages provide an easy interface to the NER models or rules, we can even load and use external open-source NER models. 

### Using Spacy NER model for Inference

- Spacy comes with several pre-trained models that can be selected based on the use case. For this example, we will use the Transformer model available with [Spacy Transformers](https://spacy.io/universe/project/spacy-transformers).

``` python linenums="1"
# install spacy-transformers and transformers model
!pip install spacy-transformers
!python3 -m spacy download en_core_web_trf

# import the spacy library
import spacy

# load the model
nlp = spacy.load("en_core_web_trf")
# set the text
text = "My name is Mohit, and I am from India. I am a Data Scientist and I will be leaving for my office around 9 AM."
# create spacy doc by passing the text
doc = nlp(text)
# print all of the identified entities
for ent in doc.ents:
  print(f"Type: {ent.label_} -- Value: {ent.text}")

# Output:
# Type: PERSON -- Value: Mohit
# Type: GPE -- Value: India
# Type: TIME -- Value: around 9 AM
```

- We can even display the results in a much more intuitive and fancy way by,
  
``` python linenums="1"
# use displacy to render the result
spacy.displacy.render(doc, style='ent', jupyter=True, options={'distance': 90})
```

<figure markdown> 
    ![](../imgs/ner_example.png)
    <figcaption>NER result for the above example</figcaption>
</figure>


## Additional materials

- To train Spacy NER model on a custom dataset: [Spacy v3 Custom NER](https://towardsdatascience.com/using-spacy-3-0-to-build-a-custom-ner-model-c9256bea098)
- [Named-Entity evaluation metrics based on entity-level](https://www.davidsbatista.net/blog/2018/05/09/Named_Entity_Evaluation/)
