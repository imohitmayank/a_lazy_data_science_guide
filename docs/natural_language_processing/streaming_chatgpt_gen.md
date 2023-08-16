# Streaming ChatGPT Generations

## Introduction

- ChatGPT is an auto-regressive Large Language Model. That means its output is generated token by token in a sequential fashion, where each token could be a combination of characters. Under normal circumstances *(and popular coding practices)*, we access the ChatGPT model via an API which takes an input prompt, generate the output and then returns it. While it may sound okay, there is one problem -- model returns the output when the complete generation is done! This means if you want the model to generate long outputs *(or even if your prompt is big due to few-shots examples or lengthy system prompts)*, you can expect a delay of several seconds before you receive the output.
- This is not okay for user-facing applications where your users are patiently waiting for the output. Thats why [ChatGPT UI](https://chat.openai.com) gives the output in streaming fashion. Here, you see characters or words printing on your screen one after the another, rather than showing the complete output at once. This creates a perception of model writing your output as human does, and even though the delay in generating the complete output will be the same, the flow becomes more enduring. 
- Behind the scene, the ChatGPT API is using Server Side Events (SSE) i.e. media stream events to return each token as and when they are generated. SSE is like an intermediate approach between normal HTTP request *(server returns one result per request)* and websocket *(server and client can send multiple requests and results)*. In SSE, while client sends one request, server can return multiple results. 
- In this article, we will try to replicate the ChatGPT streaming output behavior by creating a python application *(FastAPI server)* that could acts as a middleman between OpenAI and Frontend. In this case, OpenAI returns the outputs to our Python server at token level and the server passes it along to its client in the same manner. Let's get started! 

## Code

### OpenAI Streaming

- If you know how to use OpenAI package to generate ChatGPT output *(which in itself is quite easy)*, then getting the streaming output nothing but adding one more param (`stream=True`) in `openai.ChatCompletion.create`. Below is the code that you can easily copy paste and start using right now,

```python linenums="1"
# import 
import openai

# set the keys
openai.api_key = ".." # provide the key here
openai.organization = ".." # provide the key here

# create the ChatCompletion.create obj
completion = openai.ChatCompletion.create(
  stream=True, # the magic happens here
  model="gpt-3.5-turbo-0301",
  messages=[
    {"role": "system", "content": "You are a mails assistant. Given an email, write a proper reply."},
    {"role": "user", "content": """Mail: "We are all set. Thanks -Cam"
    """}
  ]
)

# print in streaming fashion
for chunk in completion:
    print(chunk.choices[0].delta.get("content", ""), end="", flush=True)
```

- In the above code, just by adding the `stream=True` OpenAI package is able to take care of all the hardwork and we get a completion generator. In Python, you just need to iterate over it to get the result at token level and as and when they are available. For example, if you time it with and without the `stream=True` param, you will notice the difference in output and as well as in time. While without the param the output could be available after a couple of seconds, with the param the first token will be available within a second, and the subsequent tokens after the previous one with short gap.
- To simulate the streaming output, we use print statement with `end=""` instead of default `end="\n"` so that to ignore the newline after each iteration. We also use `flush=True` so that print statment does not wait for its buffer to get full before printing on terminal.

### OpenAI Streaming App (using FastAPI)

- Now that we have the OpenAI related part done, let's focus on creating FastAPI App and expose the OpenAI wrapper as an event stream service. Below is the code, 

```python linenums="1"
# Filename: api.py
# import
import openai
from pydantic import BaseModel
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# define app 
app = FastAPI()

# add CORS policy
origins = [
    "*",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set your OpenAI API key here
openai.api_key = "..."
openai.organization = "..."

# Create BaseModel class
class Message(BaseModel):
    prompt: str

@app.post("/chat")
def chat_socket(msg: Message):
    """
    Generates a response using the OpenAI Chat API in streaming fashion.

    Returns:
        A string representing the generated response.
    """
    # ChatGPT streaming response function
    def generate_openai_response():
        """
        Generates a response using the OpenAI Chat API in streaming fashion.

        Returns:
            A string representing the generated response.
        """
        completion = openai.ChatCompletion.create(
            stream=True,
            model="gpt-3.5-turbo-0301",
            messages=[
                {"role": "system", "content": "You are a bot, given the user's input, reply appropriately"},
                {"role": "user", "content": msg.prompt}]
        )
        # print in streaming fashion
        for chunk in completion:
            yield chunk.choices[0].delta.get("content", "")

    # return 
    return StreamingResponse(generate_openai_response(), media_type='text/event-stream')

# welcome page
@app.get("/")
async def home():
    return {"message": "Welcome to the ChatGPT FastAPI App!"}


# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

- To run the code, you need to just hit `python api.py` and our endpoint is available at `http://0.0.0.0:8000/chat` endpoint!

### Client for FastAPI OpenAI Streaming App

- Once you have the server running, let's see how we can hit it from any other service. Here I am showing the example of a Python Client.

```python linenums="1"
# import 
import json
import requests

def query_api(url):
    # query the url
    response = requests.post(url, json={"prompt": "Tell me a joke!"}, stream=True)

    # parse the response
    for chunk in response.iter_content(chunk_size=None):
        if chunk:  # filter out keep-alive new chunks
            print(chunk.decode('utf-8'), end="", flush=True)

# Example usage
query_api("http://0.0.0.0:8000/chat")
```


## References

- [OpenAI Codebook - Examples](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_stream_completions.ipynb)