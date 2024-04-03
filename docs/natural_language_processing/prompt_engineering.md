# Prompt Engineering

## Introduction

Prompt engineering involves crafting well-defined and strategically designed input queries to elicit desired responses from AI systems. It serves as a bridge between human intention and machine understanding, enabling AI models to provide more accurate and contextually relevant outputs. As AI applications continue to proliferate across various domains, mastering the art of prompt engineering has become essential for both developers and users.  What makes prompt engineering more tempting is that it does not require any finetuning of the model but nevertheless, it can enhance the model accuracy substantially! In this article, we will explore different key strategies for crafting effective prompts that can enhance AI model capabilities.

## Types of Prompts

Before getting started with prompt techniques, let’s discuss the main types of prompts,

### System Prompts

System prompts are like global settings that are applied once to set the mood and intention of the AI model’s subsequent generations in the same chat. These prompts are carefully crafted by developers to guide the AI system toward specific outputs that align with the intended use case. ChatGPT UI’s custom instruction is a good example of a system prompt, as whatever you mention there is applicable to every chat. Users can provide details to format output in a certain style (like JSON), provide details about themselves so that the responses are personalized, set the tone of the generation, define privacy and ethics details, and much more! An example is shown below

```ts
System Prompt:
You are a helpful AI Assistant. Help users by replying to their queries and make 
sure the responses are polite. Do not hallucinate and say "I don't know" if required.
```

### User Prompts

User prompts are generated on the fly by users and are designed to elicit specific responses to their queries. Unlike system prompts, user prompts are not pre-defined and can vary widely in structure and content. These are more transactional in nature, and are usally present after system prompt and could be multiple in count.

```sql
System Prompt:
You are a helpful AI Assistant. Help users in replying to their queries and make 
sure the responses are polite. Do not hallucinate and say "I don't know" if required.

User Prompt:
What is your name?
```

### Assistant Output

These are AI’s generated output to the System and previous user prompts. In complex use cases, developers can modify this as an example to the AI model to highlight the kind of result expected from the model. 

```sql
System Prompt:
You are a helpful AI Assistant. Help users in replying to their queries and make 
sure the responses are polite. Do not hallucinate and say "I don't know" if required.

User Prompt:
What is your name?

Assistant:
I am an AI Assistant that can help you with your queries. Please let me know your questions!
```

## Prompt Strategies

### Zero-Shot Prompts

Zero-shot prompts are a fundamental technique in prompt engineering, allowing AI models to provide meaningful responses without any specific training examples. With zero-shot prompts, developers and users can harness the model's innate knowledge and reasoning capabilities to answer questions or complete tasks, even if the model has never seen similar queries before. When using a zero-shot prompt, formulate your question or request as clearly and concisely as possible. You can even provide some context if that helps, overall avoid ambiguity to help the model understand your intention accurately.

```sql
Example 1 - Clear instructions
################
User Prompt:
Translate the following English text to French: "Hello, how are you?"

Example 2 - Provide context
################
User Prompt:
Calculate the total revenue for our company in the last quarter, given the following financial data: [insert data].
```

Note, deciding which data should goes where *(system or user prompt)* depends on experimenting how it works for a specific model but a general thumb rule is to keep the constant details in system prompt and dynamic details in user prompt. In the first example above, we can also have following prompts,

```sql
Example 1 - Clear instructions with System prompt
################
System prompt: 
You are a Translator GPT. Given a sentence in English, translate it into French.

User prompt:
"Hello, how are you?"
```

### Few-Shot Prompts

While zero-shot prompts are fundamental, there are situations where you may need to provide a bit more guidance to the AI model. In such cases, you can use few-shot prompts that involve providing a small number of examples or demonstrations to help the model understand the desired task or context. Developers can use this approach to further guide the AI model's responses. One example of 2-shot prompt is shown below, 

```sql
System prompt: 
You are a Translator GPT. Given a sentence in English, translate it into French. 
Examples are shared below,

English: "Hello, how are you?"
French: "Bonjour, comment ça va ?"

English: "I am learning French."
French: "J'apprends le français."

User Prompt:
English: "Please pass the salt."
French: 
```

Note, the number of examples to be included (n-shot) is highly experimental. The objective should be to keep the example count as less as possible *(otherwise the token size and cost will increase)* while making sure the accuracy is not impacted. So the prompt design should be done incrementally, i.e. keep adding more examples if the accuracy is below expectations. Also, make sure to add diverse examples and do not add exact or even semantically similar examples as latest LLMs are quite “smart” enough to learn from few examples.

### Few-shot Chain-of-Thought Prompt

Few shot CoT Prompting was introduced in [1] and the idea is that generating a chain of thought, i.e. a series of intermediate reasoning steps, can significantly improves the ability of large language models to perform complex reasoning. Experiments shows that chain-of-thought prompting improves performance on a range of arithmetic, commonsense, and symbolic reasoning tasks. Basically it is clubbed with few shot prompt where the examples are provided in CoT format. Example is shown in the below image, 

<figure markdown> 
    ![](../imgs/nlp_pe_fhcot.png)
    <figcaption>Example inputs and outputs for Standard 1-shot prompt and CoT prompts</figcaption>
</figure>


### Zero-shot Chain-of-Thought Prompt

Zero shot variant of CoT was introduced in [2] and it can significantly increase the accuracy of Zero shot prompts, and all you need to do is to add “Let’s think step by step.” 😜. Btw additional post-processing is required on the output to extract the correct result, which can either be done by creating regex scripts or calling LLMs again to extract the answer.

!!! Note
    Few-shot prompts should always give better result than Zero-shot, but the former requires additional token consumption which will increase the cost. To mitigate this, developers can experiment with Zero-shot CoT technique and if the result accuracy is acceptable, it might end up reducing the overall cost.

<figure markdown> 
    ![](../imgs/nlp_pe_zscot.png)
    <figcaption>Example inputs and outputs of GPT-3 with (a) standard Few-shot, (b) Few-shot-CoT, (c) standard Zero-shot, and (d) Zero-shot-CoT</figcaption>
</figure>

### Self-consistency

Self-consistency [3] is based on the idea that there are multiple ways to solve a complex problem i.e. if multiple reasoning paths are leading to the same output, it is highly probable that it is a correct answer. In their own words, *"...we hypothesize that correct reasoning processes, even if they are diverse, tend to
have greater agreement in their final answer than incorrect processes."*. The self-consistency method consists of three steps:

1. prompt a language model using chain-of-thought (CoT) prompting;
2. replace the “greedy decode” in CoT prompting by sampling from the language model’s decoder to generate a diverse set of reasoning paths; and 
3. marginalize out the reasoning paths and aggregate by choosing the most consistent answer in the final answer set.

The authors of the paper performed extensive empirical evaluations to shows that self-consistency boosts the performance of chain-of-thought prompting on a range of popular arithmetic and commonsense reasoning
benchmarks, including GSM8K (+17.9%), SVAMP (+11.0%), AQuA (+12.2%), StrategyQA (+6.4%) and ARC-challenge (+3.9%).

<figure markdown> 
    ![](../imgs/nlp_pe_sc.png)
    <figcaption>CoT vs Self-consistency prompting example</figcaption>
</figure>

### Tree-of-Thoughts

Tree-of-Thoughts (ToT) [4] is based on the idea that to solve any complex problem we need to (a) explore multiple reasoning paths *(branches in a graph)*, and (b) perform planning i.e. lookahead or even backtrack on the paths if required. ToT frames any problem as a search over a tree, where each node is a state `s = [x, z1···i]` representing a partial solution with the input and the sequence of thoughts so far. A specific instantiation of ToT involves answering four questions: 

1. How to decompose the intermediate process into thought steps -- depending on different problems, a thought could be a couple of words (Crosswords), a line of equation (Game of 24), or a whole paragraph of writing plan (Creative Writing). In general, a thought should be “small” enough so that LMs can generate promising and diverse samples.
2. How to generate potential thoughts from each state -- again it depends on the problem, so for Creative writing we can sample thoughts from a CoT prompt and for Game of 24 and Crosswords we can propose thoughts sequentially using propose prompt.
3. How to heuristically evaluate states -- this can be done automatically by either asking the model to generate a value *(score between 1 to 10 or classification of sure/likely/impossible)* or voting on different results.
4. What search algorithm to use -- authors propose Breadth-first search (BFS) and Depth-first Search (DFS) and left more complex search algorithms like A* for future works.

<figure markdown> 
    ![](../imgs/nlp_pe_tot.png)
    <figcaption>Schematic illustrating various approaches to problem solving with LLMs. Each rectangle box represents a thought, which is a coherent language sequence that serves as an intermediate step toward problem solving</figcaption>
</figure>

### Retrieval Augmented Generation (RAG)

In all of the previous approaches, the result was generated entirely by the LLMs without any external intervention. This leverages the knowledge stored within the neural networks of the LLMs *(read, weights in the network)*. This poses issues like hallucinations *(this happens when model is not sure what to say, especially due to lack of knowledge)* and factual inaccuracies *(lack of knowledge leads to model lying)*. To mitigate these issues, we can "connect" LLMs with external data source *(vector database, wikipedia, google, etc)* so that true, diverse and dynamic data can be fetched from these sources and LLM can do what they are best suited for - reasoning on the provided data to format the final result. This is the fundamental idea behind Retrieval Augmented Generation (RAG).

One example of such system is [Sankshep](https://www.sankshep.co.in/) *(by yours truly :sunglasses:)* that provides ChatGPT-powered assistant to summarize and talk to Arxiv research papers. Here, if you ask a question regarding the paper, Sankshep refer the content of the paper to be better aware and provide factually correct results.

<figure markdown> 
    ![](../imgs/nlp_pe_sankshep.png)
    <figcaption>Sankshep.co.in built considering Retrieval Augmented Generation (RAG)</figcaption>
</figure>

### ReAct

ReAct [5] combines the external knowledge of RAG with the planning and reasoning notion of ToT. As per the paper, 

> A unique feature of human intelligence is the ability to seamlessly combine task-oriented actions with verbal reasoning. Consider the example of cooking up a dish in the kitchen. Between any two specific actions, we may reason in language in order to track progress (“now that everything is cut, I should heat up the pot of water”), to handle exceptions or adjust the plan according to the situation (“I don’t have salt, so let me use soy sauce and pepper instead”), and to realize when external information is needed (“how do I prepare dough? Let me search on the Internet”).

The important point of the above quote (and in fact the paper) is the intention to combine two powerful abilities of LLMs — reasoning (e.g. chain-of-thought prompting) and acting (e.g. action plan generation). While the former helps with improving the accuracy of an individual task, the latter provides the LLM power to perform multiple tasks. The idea is quite simple — ask LLM a question (input) and let it “plan” what to do next (action) by reasoning on the input (thoughts). It can even propose values to the action (action input). Once we perform the action and get an output (observation) the LLM can again reason (thought) to propose the next “plan” (action). In a way, we keep the LLM busy in this loop, until it terminates and we get the result we wanted. To summarize we iterate over “input —> thoughts —> action —> action input —> observation —> thoughts”. For practical details, please refer [Creating GPT-driven Applications using LangChain](https://mohitmayank.com/blog/openai-gpt-apps-langchain)

<figure markdown> 
    ![](../imgs/nlp_pe_react.png)
    <figcaption>LLM reasoning and planning using ReAct technique</figcaption>
</figure>

### EmotionPrompt

<figure markdown> 
    ![](../imgs/nlp_pe_emotionprompt.png)
    <figcaption>EmotionPrompt impact on different LLMs</figcaption>
</figure>

- Emotion plays a vital role in our daily lives, but LLMs are usually considered deprived of it. Researchers test that theory using EmotionPrompt [7] that involves combining the original prompt with emotional stimuli to improve LLMs' performance. Researchers conducted two experiments, (1) automatic experiments across 45 tasks using a range of LLMs, such as Flan-T5-Large, Vicuna, Llama 2, BLOOM, ChatGPT, and GPT-4. (2) human studies involving 106 participants. And both the cases shows improvements across performance, truthfulness, and responsibility metrics.

!!! Warning
    The above statements does not, in any way, implies that LLMs really understand emotions OR we have acheived AGI OR doom of humanity is near :smile:

- On the topic of creating EmotionPrompt, researchers tested out 11 different prompts created in consideration of 3 different psychological domains:
  - **Self-monitoring**: is a concept in social psychology and refers to the process by which individuals regulate and control their behavior in response to social situations. Example of a prompt - "This is very important to my career".
  - **Social Cognitive theory**: wherein self-efficacy *(individual's belief in their capacity to act in the ways necessary to rech specific goal)* was applied to build up confidence and emphasize the goal. Example of a prompt - "believe in your abilities".
  - **Cognitive Emotion Regulation**: suggests that people lacking emotion regulation skills are more likely to engage in compulsive behavior and techniques like reappraisal can help. According to this theory, researchers aimed to stimulate the reappraisal skills of LLMs by incorporating pivotal terms, such as “sure” and “take another look”. Example of a prompt - "Are you sure?".

<figure markdown> 
    ![](../imgs/nlp_pe_emotionprompt_examples.png)
    <figcaption>EmotionPrompt impact on different LLMs</figcaption>
</figure>

- Researchers [7] conducted a segregated examination to assess the effectiveness of various emotional stimuli on two different benchmarks, Instruction Induction and BIG-Bench. They used six Large Language Models (LLMs) for each emotional stimulus and calculated the average performance for both human-designed and APE-generated prompts. Key findings revealed that EP02 was the most effective stimuli for Instruction Induction, while EP06 performed best in BIG-Bench, highlighting the influence of task complexity and type on the choice of emotional stimuli. The study suggests that different stimuli can activate LLMs' capabilities differently, making them more suitable for specific tasks, thus emphasizing the need for task-specific emotional stimuli in interactions with LLMs.

## Jailbreaking

In the world of technology, jailbreaking refers to bypassing security measures on a device to grant users more control and potentially unlock functionalities not intended by the developers. Similarly, jailbreaking large language models (LLMs) aims to circumvent the safeguards put in place to ensure safe and ethical outputs. These safeguards might prevent the LLM from answering certain questions or generating certain content. Do note, the definition of "safe and ethical" content is a grey area and subject to believes of developers of the AI models, hence jailbreaking sometimes helps to circumvent such scenarios.

!!! Warning
    This section is shared for educational purposes only and should not be used to bypass legal guidelines. Always ensure that AI systems are used responsibly.

### Many-Shot Jailbreaking

Multi-shot jailbreaking [8] technique exploits the growing context window of LLMs. This window allows the model to consider a larger chunk of text preceding a prompt. Multi-shot jailbreaking involves feeding the LLM a series of fabricated dialogues where a user successfully elicits unsafe information from an AI assistant. By conditioning the LLM on these examples, it essentially learns to disregard its safety protocols when presented with a similar query later on. This technique has proven effective against various LLMs, highlighting the ongoing challenge of securing these powerful language models.

<figure markdown> 
    ![](../imgs/nlp_pe_multishotjailbreaking.png)
    <figcaption> Example of Many-shot jailbreaking [8]</figcaption>
</figure>

In accordance with "In-context learning", the model "learns" from the context of the conversation and hence, the more shots you provide, the more the model "learns" about the context and hence, the more chance of it bypassing the safety protocols.

<figure markdown> 
    ![](../imgs/nlp_pe_multishotjailbreaking_2.png)
    <figcaption>As the number of shots increases beyond a certain number, so does the percentage of harmful responses to target prompts related to violent or hateful statements, deception, discrimination, and regulated content (e.g. drug- or gambling-related statements). The model used for this demonstration is Claude 2.0. [8]</figcaption>
</figure>

## Conclusion

Prompt engineering is a crucial skill for leveraging the capabilities of LLMs effectively. By understanding the different types of prompts and employing strategies such as zero-shot prompts, few-shot prompts, etc, developers and users can harness the power of AI to achieve more accurate and contextually relevant responses. As AI technologies continue to evolve, mastering prompt engineering will remain an essential tool for unlocking the full potential of AI systems across various domains.

## References

[1] [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)

[2] [Large Language Models are Zero-Shot Reasoners](https://arxiv.org/pdf/2205.11916.pdf)

[3] [Self-consistency improves chain of thought reasoning in language models](https://arxiv.org/pdf/2203.11171.pdf)

[4] [Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/pdf/2305.10601.pdf)

[5] [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)

[6] [Prompting Guide](https://www.promptingguide.ai/)

[7] [Large Language Models Understand and Can Be Enhanced by Emotional Stimuli](https://www.sankshep.co.in/PDFViewer/https%3A%2F%2Farxiv.org%2Fpdf%2F2307.11760.pdf)

[8] [Many-shot jailbreaking](https://www.anthropic.com/research/many-shot-jailbreaking)