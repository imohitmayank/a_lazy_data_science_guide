- Here are some questions and their answers to make you ready for your next interview. Best of luck :wave:

!!! Question ""

    === "Question"
        #### What are the different types of reasoning tasks in NLP?

    === "Answer"

        - **Arithmetic Reasoning:** Arithmetic reasoning is the ability of an NLP system to perform mathematical operations on numerical data. This can include basic arithmetic operations such as addition, subtraction, multiplication, and division as well as more complex operations such as algebraic equations and calculus.
        - **Commonsense Reasoning:** Commonsense reasoning refers to the ability of an NLP system to make deductions based on the knowledge and information that is commonly understood by humans. This includes understanding social norms, cultural contexts, and everyday life experiences. *([StrategyQA](https://allenai.org/data/strategyqa) is a sample dataset that contains True/False questions like "Did Aristotle use a laptop?")*
        - **Symbolic Reasoning:** Symbolic reasoning involves the ability of an NLP system to manipulate and reason about symbolic representations of information, such as words, phrases, and sentences. This includes tasks such as parsing, string operations, semantic role labeling and entity recognition. *(Last Letter Concatenation is a sample dataset with questions like "Take the last letters of the words in 'Lady Gaga' and concatenate them")*
        - **Logic Reasoning:** Logic reasoning refers to the ability of an NLP system to perform logical deductions based on formal rules of inference. This can include tasks such as identifying logical fallacies, determining the validity of arguments, and drawing conclusions based on deductive reasoning. *(Date understanding is a sample dataset with questions like "Today is Christmas Eve 1937, what is the date tomorrow in MM/DD/YYYY?")*

!!! Question ""

    === "Question"
        #### How much VRAM is required to load a 7B LLM?

    === "Answer"
        
        In full precision (float32), the model stores each parameter in 32 bits or 4 bytes. Therefore, for inference only, loading a 7B model requires 28 billion bytes, which is equivalent to 28 GB of GPU memory. If we use half precision, each parameter is stored in 16 bits, or 2 bytes, resulting in a requirement of 14 GB for inference. Additionally, there are more efficient algorithms like 8 bit and 4 bit, where with 4 bits (half a byte) per parameter, only 3.5 GB of memory is needed for inference.

        For training, the memory requirement depends on the optimizer used. If the regular AdamW optimizer is employed, it necessitates 8 bytes per parameter since it not only stores the parameters but also their gradients and second-order gradients. Therefore, for a 7 billion parameter model, it would require 56 GB of GPU memory. Alternatively, using AdaFactor, which requires 4 bytes per parameter, the memory requirement decreases to 28 GB. There are also optimizers like bitsandbytes, such as the 8-bit AdamW, which only need 2 bytes per parameter, resulting in a 14 GB GPU memory requirement. [Source](https://discuss.huggingface.co/t/llama-7b-gpu-memory-requirement/34323/8)

        [Read for more details](https://huggingface.co/docs/transformers/perf_train_gpu_one#anatomy-of-models-memory)


!!! Question ""

    === "Question"
        #### What are word embeddings in NLP?

    === "Answer"
        
        Word embeddings are a type of representation for words in NLP. They are a dense vector representation of a word, learned from the data using techniques such as word2vec or GloVe. The embeddings capture the semantics of the words, meaning that words with similar meanings will have similar vectors. Word embeddings are used as input in many NLP tasks such as language translation, text classification, and text generation.

!!! Question ""
    === "Question"
        #### What is Sentence Encoding?

    === "Answer"
        
        Sentence encoding is the process of converting a sentence into a fixed-length vector representation, also known as sentence embeddings. This is done by using techniques such as bag-of-words, TF-IDF, or BERT-based models. Sentence encodings can be used as input in various NLP tasks such as text classification, text generation, and text similarity. Several algorithms first tokenize the sentence in words or tokens, compute thir embedding and then aggregate them (min, max, mean, etc) to get the sentence embedding.

!!! Question ""
    === "Question"
        #### Explain the concept of attention mechanism in NLP?

    === "Answer"
        
        Attention mechanism is a way to weight different parts of the input in a neural network, giving more importance to certain parts of the input than others. It is commonly used in NLP tasks such as machine translation, where the model needs to focus on different parts of the input sentence at different times. Attention mechanisms can be implemented in various ways, such as additive attention ($𝑄+𝐾$) and dot-product attention ($𝑄𝐾^{𝑇}$)

!!! Question ""
    === "Question"
        #### What are transformer models in NLP?

    === "Answer"
        
        Transformer models are a type of neural network architecture that have been successful in various NLP tasks such as language translation and language understanding. They were introduced in the transformer paper and use self-attention mechanism to weigh the different parts of the input. This allows the model to efficiently process long input sequences and handle the dependencies between the words. [Refer](transformer.md) for more details.

!!! Question ""
    === "Question"
        #### Can you explain the concept of Named Entity Recognition (NER) in NLP?

    === "Answer"
        
        Named Entity Recognition (NER) is a subtask of information extraction that seeks to locate and classify named entities in text into predefined categories such as person names, organizations, locations, medical codes, time expressions, quantities, monetary values, percentages, etc. NER systems can be rule-based or based on machine learning, and are used in a wide range of applications such as information retrieval, question answering and text summarization.

!!! Question ""
    === "Question"
        #### Explain Part-of-Speech (POS) tagging in NLP?

    === "Answer"
        
        Part-of-Speech (POS) tagging is the process of marking each word in a text with its corresponding POS tag. This is a fundamental step in many NLP tasks such as parsing, text summarization, and information extraction. POS tagging can be rule-based or based on machine learning, and is typically done using algorithms such as Hidden Markov Models (HMMs) or Conditional Random Fields (CRFs).


!!! Question ""
    === "Question"
        #### Can you explain the concept of Language Modeling in NLP?

    === "Answer"
        
        Language modeling is the task of predicting the next word in a sentence, given the previous words. This is done by training a model on a large dataset of text, which learns the probability distribution of the words in the language. Language models are used in a wide range of NLP tasks such as machine translation, text generation, and speech recognition.

!!! Question ""
    === "Question"
        #### Can you explain the concept of Text Summarization?

    === "Answer"
        
        Text summarization is the task of generating a shorter version of a text that retains the most important information. There are two main types of text summarization: extractive and abstractive. Extractive summarization selects important sentences or phrases from the text to form the summary, while abstractive summarization generates new text that captures the meaning of the original text.

!!! Question ""
    === "Question"
        #### What is Sentiment Analysis?

    === "Answer"
        
        Sentiment analysis is the task of determining the sentiment or emotion of a piece of text. This is typically done by classifying the text as positive, negative, or neutral. Sentiment analysis can be done using a variety of techniques such as rule-based systems, machine learning, and deep learning. It is used in a wide range of applications such as customer feedback analysis and social media analysis.

!!! Question ""
    === "Question"
        #### Can you explain the concept of Dependency Parsing?

    === "Answer"
        
        Dependency parsing is the task of analyzing the grammatical structure of a sentence, identifying the dependencies between the words. This is done by creating a dependency parse tree, which represents the grammatical relationships between the words in a sentence. Dependency parsing is a fundamental step in many NLP tasks such as machine translation, text summarization, and information extraction.

!!! Question ""
    === "Question"
        #### Explain the Coreference Resolution task in NLP?

    === "Answer"
        
        Coreference resolution is the task of identifying when different expressions in a text refer to the same entity. This is done by analyzing the text and identifying when two or more expressions have the same referent. Coreference resolution is a fundamental step in many NLP tasks such as machine translation, text summarization, and information extraction. In this example text, *"Mohit lives in Pune and he works as a Data Scientist"*, the co-reference resolution will identify "Mohit" and "he" as belonging to the same entity.


!!! Question ""
    === "Question"
        #### Explain Stemming and Lemmatization in NLP?

    === "Answer"
        
        - Stemming is the process of reducing inflected (or sometimes derived) words to their word stem, base or root form. This is done by using a stemmer algorithm which removes the suffixes or prefixes from the word. The goal of stemming is to reduce the dimensionality of the text data, grouping together the inflected forms of a word so they can be analyzed as a single term, which can be useful for tasks such as text classification and information retrieval.
        - Lemmatization is the process of reducing a word to its base form, which is called the lemma. This is done by using a lemmatizer algorithm which takes into consideration the context and the part of speech of the word. The goal of lemmatization is to reduce the dimensionality of the text data and group together the different forms of a word so they can be analyzed as a single term, which can be useful for tasks such as text classification and information retrieval.

        !!! Note
            An obvious difference is that Lemmatization consider the grammar of the sentence while Stemming only consider the word. 


!!! Question ""
    === "Question"
        #### What is Text Classification?

    === "Answer"
        
        - Text classification is the task of assigning predefined categories or labels to a given text. This is done by training a model on a labeled dataset of text, which learns to predict the label of new text. Text classification is used in a wide range of applications such as sentiment analysis, spam detection, and topic classification. There are multiple types of text classification such as, 

          - **Binary classification:** where there only two classes (ex: positive vs negative)
          - **Multi-class classification**: where there are more than 2 classes (ex: positive, negative and neutral)
          - **Multi-label classification**: when there are two or more classes and each text can have more than 1 class/label assigned to them (ex: single text can have some positive phrase and negative phrase)


!!! Question ""
    === "Question"
        #### What are Dialogue Systems in NLP?

    === "Answer"
        
        A Dialogue system, also known as a conversational agent or chatbot, is a computer program that can interact with human users through natural language. It can understand the user's input, generate appropriate responses, and carry out tasks such as answering questions, booking a flight, or making a reservation. Dialogue systems can be rule-based, or based on machine learning and deep learning, and can be integrated into various platforms such as smartphones, websites, and messaging apps.

!!! Question ""
    === "Question"
        #### Please explain the concept of Text Generation?

    === "Answer"
        
        Text generation is the task of generating new text that is similar to the text it was trained on. This is done by training a model on a large dataset of text, which learns the probability distribution of the words in the language. Text generation can be used for various applications such as chatbot, text completion and summarization. [Refer](text_generation.md) for more details.

!!! Question ""
    === "Question"
        #### Can you explain the concept of Text Similarity in NLP?

    === "Answer"
        
        Text Similarity is the task of determining how similar two pieces of text are to each other. This is done by comparing the text using various similarity measures such as cosine similarity, Jaccard similarity, or Levenshtein distance. Text similarity can be used in a wide range of applications such as plagiarism detection and text retrieval. [Refer](text_similarity.md) for more details.

!!! Question ""
    === "Question"
        #### Please explain Text Clustering?

    === "Answer"
        
        Text Clustering is the process of grouping similar texts together. This is done by using clustering algorithms such as K-means, Hierarchical Clustering, or DBSCAN. Text clustering can be used in a wide range of applications such as topic modeling, sentiment analysis, and text summarization. This is usually a two step process, first the text is converted to a representation (usually by text embedding algorithms) and then a clustering algorithm is used to create clusters.

!!! Question ""
    === "Question"
        #### What is Named Entity Disambiguation (NED)?

    === "Answer"
        
        Named Entity Disambiguation (NED) is the task of determining which entity (from a database) a mention (from text or doc) refers to, from a set of candidate entities. This is done by using techniques such as string matching, co-reference resolution, or graph-based methods. NED is important for tasks such as information extraction and knowledge base population. For example, NED will process a wikipedia page and map "Mohit M.", "M. Mayank", "Mohit" and similar named entities with "Mohit Mayank" present in the database. 


!!! Question ""
    === "Question"
        #### What is the difference between a feedforward neural network and a recurrent neural network?

    === "Answer"

        A feedforward neural network is a type of neural network in which the data flows in one direction, from input to output. There are no loops or connections that allow information to flow in a cyclical manner. On the other hand, a recurrent neural network (RNN) is a type of neural network that allows information to flow in a cyclical manner, with connections between neurons allowing information to be passed from one step of the process to the next. RNNs are useful for tasks such as language modeling and speech recognition, where the current input is dependent on the previous inputs.

!!! Question ""
    === "Question"
        #### Is BERT a Text Generation model?

    === "Answer"

        Short answer is no. BERT is not a text generation model or a language model because the probability of the predicting a token in masked input is dependent on the context of the token. This context is bidirectional, hence the model is not able to predict the next token in the sequence accurately with only one directional context *(as expected for language model)*.

!!! Question ""
    === "Question"
        #### What is weight tying in language model?

    === "Answer"

        Weight-tying is where you have a language model and use the same weight matrix for the input-to-embedding layer (the input embedding) and the hidden-to-softmax layer (the output embedding). The idea is that these two matrices contain essentially the same information, each having a row per word in the vocabulary. [Ref](https://tomroth.com.au/weight_tying/)
        
!!! Question ""
    === "Question"
        #### What is so special about the special tokens used in different LM tokenizers?

    === "Answer"

        Special tokens are called special because they are added for a certain purpose and are independent of the input. For example, in BERT we have `[CLS]` token that is added at the start of every input sentence and `[SEP]` is a special separator token. Similarly in GPT2, `<|endoftext|>` is special token to denote end of sentence. Users can create their own special token based on their specific use case and train them during finetuning. [Refer cronoik's answer in SO](https://stackoverflow.com/questions/71679626/what-is-so-special-about-special-tokens)


!!! Question ""
    === "Question"
        #### What are Attention Masks?

    === "Answer"

        Attention masks are the token level boolean identifiers used to differentiate between important and not important tokens in the input. One use case is during batch training, where a batch with text of different lengths can be created by adding paddings to shorter texts. The padding tokens can be identified using 0 in attention mask and the original input tokens can be marked as 1. [Refer blog @ lukesalamone.com](https://lukesalamone.github.io/posts/what-are-attention-masks/)

        !!! Note
        We can use a special token for padding. For example in BERT it can be `[PAD]` token and in GPT-2 we can use `<|endoftext|>` token.

!!! Question ""
    === "Question"
        #### What is the difference between Attention and Self-Attention?

    === "Answer"

        Self-attention (SA) is applied within one component, so the input is from one component only. One example is the encoder block in Transformers, where SA is used, and it takes the tokens from only the sentence as the input. Attention on the other hand can be used to connect two components or modality of data. Decoder in Transformer has Attention layer that connects the encoder and decoder data together. [Refer StackExchange QA](https://datascience.stackexchange.com/questions/49468/whats-the-difference-between-attention-vs-self-attention-what-problems-does-ea)


!!! Question ""
    === "Question"
        #### What is the difference between Bi-encoder and Cross-encoder?

    === "Answer"

        In the context of NLP, the difference between Bi-encoders and Cross-encoders lies in how they handle and process input sentences and the kind of outputs they produce. [Refer SBert](https://www.sbert.net/examples/applications/cross-encoder/README.html)

        1. **Bi-Encoders:**
           - **Process:** Bi-encoders process two sentences (A and B) independently through a transformer network like BERT. Each sentence is input separately, resulting in two separate sentence embeddings (u and v).
           - **Output:** The output is a pair of sentence embeddings, which can be compared using measures like cosine similarity to determine their relationship or similarity.
           - **Usage:** These are particularly useful when you want to pre-compute sentence embeddings for a large collection of text, as you can then quickly compare new sentences to the entire collection using the embeddings. They are efficient for tasks that require comparing multiple sentences against one another.

        2. **Cross-Encoders:**
           - **Process:** Cross-encoders take a pair of sentences as a single input and pass them simultaneously to the transformer network. The model takes into account the interaction between the sentences directly during the encoding process.
           - **Output:** Instead of separate embeddings, a Cross-encoder outputs a single score or value, typically between 0 and 1, representing the relationship or similarity of the sentence pair. It does not produce separate sentence embeddings.
           - **Usage:** Cross-encoders are generally more accurate for tasks like semantic similarity or relevance ranking because they consider the interaction between the sentence pair directly. However, they are less efficient compared to Bi-encoders for large-scale tasks because each sentence pair must be processed together and in real-time, making it hard to pre-compute and store representations.
  

!!! Question ""
    === "Question"
        #### What is the Fertility and Parity of the tokenizer?

    === "Answer"
        Fertility and parity are metrics used to evaluate a tokenizer's performance. 

        - **Fertility** is defined as the average number of tokens required to represent a word or document. It's calculated by dividing the number of tokens in a dataset by the number of words in the dataset. Higher fertility scores indicate weaker compression capabilities of the tokenizer.

        - **Parity** assesses how fairly a tokenizer treats equivalent sentences in different languages. A tokenizer achieves parity for language A with respect to language B if the ratio of the number of tokens in language A to the number of tokens in language B is approximately 1. This is calculated by performing tokenization on a translation dataset where each sentence in language A is translated into language B and vice versa. 
        


