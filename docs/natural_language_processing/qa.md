!!! warning
    This page is still under progress. Please check back after some time or [contact me](mailto:mohitmayank1@gmail.com)

## Introduction

- As the name suggests, Question Answering is a NLP task of finding answer given the question and context *(optional)*. There are two varieties based on type of inputs, 
  - **Open-domain QA**: where **context is not provided**. The expectation is that model has *internalised* knowledge within its parameters and should be able to answer the question.

    ``` mermaid
    graph LR
    A("Who is the author of Lazy Data Scientist?") -- Question --> B(QA model)
    B -- Answer --> C("Mohit")
    style B stroke:#f66,stroke-width:2px,stroke-dasharray: 5 5
    ```

  - **Closed-domain QA**: where **context is provided**. The expectation is that model should find the answer from the context.

    ``` mermaid
    graph LR
    A("Who is the author of Lazy Data Scientist?") -- Question --> B(QA model)
    D("Hi, my name is Mohit. I am the author of Lazy Data Scientist!") -- Context --> B(QA model)
    B -- Answer --> C("Mohit")
    style B stroke:#f66,stroke-width:2px,stroke-dasharray: 5 5
    ```

- Answers could be also be of two types, 
  - **Short form Answers**: where the answer is brief and to the point. In above example, the answers are short form. Majority of the Closed domain QA generate short form answers as they follow extractive approach of finding answer.
  - **Long form Answers**: where the answer is descriptive, standalone and may also contain additional details. For the above example, long form answer could be `Mohit is the author of Lazy Data Scientist`. 

!!! Note
    We can use additional models like LLM (GPTs, T5, etc) on top of QA system to convert short form answers to long form. Existing model will require finetuning with the Question and Short form answer as input and Long form answer as output.

  
<!-- ## Datasets

### SQuaRD -->


## Metrics

- There are two main metrics preferred to analyze the performance of Question Answering models. They are, 

### Exact Match

- As the name suggests, for each question we compare the golden answer with the predicted answer. If the two answers are exactly similar (`y_pred == y_true`) then `exact_match_example = 1` else `exact_match_example = 0`.

### F1 score

- It's a possibility that the predicted answer includes the important parts of the golden answer, but due to the nature of exact match, the score is still 0. Let's understand it with an example, here ideally the score should be high *(if not 1)*, but exact match will give 0.
  - Question: When can we meet?
  - Golden answer: We can meet around 5 PM.
  - Predicted answer: 5 PM.
  
- For such cases we can perform partial match. Below is the example, 
``` python linenums="1"
# vars to begin with
predicted_answer = 'Hi, My name is Mohit'
golden_answer = 'My name is Mohit'

# tokenize
predicted_words = set(predicted_answer.lower().split())
golden_words = set(golden_answer.lower().split())

# find common words
common_words = predicted_words & golden_words

# compute metrics
recall = common_words / predicted_words
precision = common_words / golden_words
F1 = 2 * precision * recall / (precision + recall)
```

!!! Note
    In case one question has multiple independent answers, we can compare each golden answer for the example against the prediction and pick the one with highest score. The overall accuracy is then the average of the individual example level score. This logic can be applied to both the metrics discussed above.


<!-- ## Code -->

## References

[1] [Evaluating QA: Metrics, Predictions, and the Null Response](https://qa.fastforwardlabs.com/no%20answer/null%20threshold/bert/distilbert/exact%20match/f1/robust%20predictions/2020/06/09/Evaluating_BERT_on_SQuAD.html)

[2] [SQuAD Explorer](https://rajpurkar.github.io/SQuAD-explorer/)

[3] [How to Build an Open-Domain Question Answering System?](https://lilianweng.github.io/posts/2020-10-29-odqa/)