GPT models
==========

## Introduction

- GPT stands for "Generative Pre-trained Transformer". It is an autoregressive language model which is based on the decoder block of the Transformer architecture.

<figure markdown> 
        ![](../imgs/nlp_transformers.png)
        <figcaption>Transformer architecture. Left part is the encoder, right part is the decoder. GPT is made up of the right part i.e. decoder part. (*vaswani2017attention*)</figcaption>
        </figure>

- The idea for the model is similar to any text generation model i.e. it takes some prompt as input and generates text as output. But the caveat is that, GPT model's tunable parameter ranges from 100 million to 175 billion, which leads to the model learning much more than basic language syntax information or word related contextual information. In fact, it has been shown that GPT models store additional real-world information as well, and there has been interesting recent research about how much knowledge you can pack into the parameters (*roberts2020knowledge*).
- GPT models are also famous because they can be easily applied to many downstream NLP tasks. This is because they have been shown to be very good in few-shot leaning, i.e. they are able to perform tasks for which they are not even trained by only providing a few examples!
- This lead to a new interesting paradigm of prompt engineering, where creating crisp prompt could lead to good results. This means that, instead of playing around with training, just by modifying the input prompt to the model, we can improve the accuracy. Ofcourse, for better accuracy, it is always preferred to fine-tune the model. Example of a sample prompt is provided below

```md
## prompt input - this can be passed to a GPT based model
Below are some examples for sentiment detection of movie reviews.

Review: I am sad that the hero died.
Sentiment: Negative

Review: The ending was perfect.
Sentiment: Positive

Review: The plot was not so good!
Sentiment: 

## The model should predict the sentiment for the last review.
```

!!! tip
    Copy the prompt from above and try it @ [GPT-Neo 2.7B model](https://huggingface.co/EleutherAI/gpt-neo-2.7B). You should get "Negative" as output! We just created a Sentiment detection module without a single training epoch!

## Analysis

### Comparing GPT models (basic details)

- There are two famous series of GPT models, 
    - **GPT-{1,2,3}:** the original series released by [OpenAI](https://en.wikipedia.org/wiki/OpenAI), a San Francisco-based artificial intelligence research laboratory. It includes GPT-1 (*radford2018improving, GPT-2 radford2019language, GPT-3 brown2020language*)
    - **GPT-{Neo, J}:** the open source series released by [EleutherAI](https://www.eleuther.ai/). For GPT-Neo, the architecture is quite similar to GPT-3, but training was done on [The Pile](https://pile.eleuther.ai/), an 825 GB sized text dataset.
- Details of the models are as follows, *([details](https://huggingface.co/transformers/pretrained_models.html))*

| models  | released by | year | open-source |       model size       |
| ------- | ----------- | ---- | ----------- | :--------------------: |
| GPT-1   | OpenAI      | 2018 | yes         |          110M          |
| GPT-2   | OpenAI      | 2019 | yes         | 117M, 345M, 774M, 1.5B |
| GPT-3   | OpenAI      | 2020 | no          |          175B          |
| GPT-Neo | EleutherAI  | 2021 | yes         |    125M, 1.3B, 2.7B    |
| GPT-J   | EleutherAI  | 2021 | yes         |           6B           |

## Code

- The most recent open-source models from OpenAI and EleutherAI are GPT-2 and GPT-Neo, respectively. And as they share nearly the same architecture, the majority of the code for inference or training, or fine-tuning remains the same. Hence for brevity's sake, code for GPT-2 will be shared, but I will point out changes required to make it work for GPT-Neo model as well.

### Inference of GPT-2 pre-trained model

- For a simple inference, we will load the pre-trained GPT-2 model and use it for a dummy sentiment detection task (using the prompt shared above).
- To make this code work for GPT-Neo, 
    - import `GPTNeoForCausalLM` at line 2
    - replace line 5 with `model_name = "EleutherAI/gpt-neo-2.7B"` *(choose from any of the available sized models)*
    - use `GPTNeoForCausalLM` in place of `GPT2LMHeadModel` at line 9

``` python linenums="1"
# import
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# model name
model_name = "gpt2"

# load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name).cuda()

# create prompt
prompt = """
Below are some examples for sentiment detection of movie reviews.

Review: I am sad that the hero died.
Sentiment: Negative

Review: The ending was perfect.
Sentiment: Positive

Review: The plot was not so good!
Sentiment:"""

# generate tokens
generated = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

# perform prediction 
sample_outputs = model.generate(generated, do_sample=False, top_k=50, max_length=512, top_p=0.90, 
        temperature=0, num_return_sequences=0)

# decode the predicted tokens into texts
predicted_text = tokenizer.decode(sample_outputs[0], skip_special_tokens=True)
print(predicted_text)
```

### Finetuning GPT-2 (for sentiment classification)

- Tweet sentiment data can be downloaded from [here](https://www.kaggle.com/kazanova/sentiment140)
- We add the special tokens at line 72, so that the model learns the start and end of the prompt. This will be helpful later on during the testing phase, as we don't want the model to keep on writing the next word, but it should know when to stop the process. This can be done by setting the `eos_token` and training the model to predict the same.
- We also define how to process the training data inside `data_collator` on line 91. The first two elements within the collator are `input_ids `- the tokenized prompt and `attention_mask `- a simple 1/0 vector which denote which part of the tokenized vector is prompt and which part is the padding. The last part is quite interesting, where we pass the input data as the label instead of just the sentiment labels. This is because we are training a language model, hence we want the model to learn the pattern of the prompt and not just sentiment class. In a sense, the model learns to predict the words of the input tweet + sentiment structured in the prompt, and in the process learn the sentiment detection task.

``` python linenums="1"
-

# set the model to eval mode
_ = model.eval()

# run model inference on all test data
original_label, predicted_label, original_text, predicted_text = [], [], [], []
map_label = {0:'negative', 4: 'positive'}
# iter over all of the test data
for text, label in tqdm(zip(test_dataset[0], test_dataset[1])):
    # create prompt (in compliance with the one used during training)
    prompt = f'<|startoftext|>Tweet: {text}\nSentiment:'
    # generate tokens
    generated = tokenizer(f"{prompt}", return_tensors="pt").input_ids.cuda()
    # perform prediction
    sample_outputs = model.generate(generated, do_sample=False, top_k=50, max_length=512, top_p=0.90, 
            temperature=0, num_return_sequences=0)
    # decode the predicted tokens into texts
    predicted_text  = tokenizer.decode(sample_outputs[0], skip_special_tokens=True)
    # extract the predicted sentiment
    try:
        pred_sentiment = re.findall("\nSentiment: (.*)", predicted_text)[-1]
    except:
        pred_sentiment = "None"
    # append results
    original_label.append(map_label[label])
    predicted_label.append(pred_sentiment)
    original_text.append(text)
    predicted_text.append(pred_text)

# transform result into dataframe
df = pd.DataFrame({'original_text': original_text, 'predicted_label': predicted_label, 
                    'original_label': original_label, 'predicted_text': predicted_text})

# predict the accuracy
print(f1_score(original_label, predicted_label, average='macro'))
```

## Additional materials

- The Illustrated GPT-2 (Visualizing Transformer Language Models) - [Link](https://jalammar.github.io/illustrated-gpt2/)




--8<-- "includes/abbreviations.md"