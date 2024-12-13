<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Chat Templates

## Introduction

An increasingly common use case for LLMs is **chat**. In a chat context, rather than continuing a single string
of text (as is the case with a standard language model), the model instead continues a conversation that consists
of one or more **messages**, each of which includes a **role**, like "user" or "assistant", as well as message text.

Much like tokenization, different models expect very different input formats for chat. This is the reason we added
**chat templates** as a feature. Chat templates are part of the tokenizer. They specify how to convert conversations, 
represented as lists of messages, into a single tokenizable string in the format that the model expects. 

Let's make this concrete with a quick example using the `mistralai/Mistral-7B-Instruct-v0.1` model:

```python
>>> from transformers import AutoTokenizer
>>> tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

>>> chat = [
...   {"role": "user", "content": "Hello, how are you?"},
...   {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
...   {"role": "user", "content": "I'd like to show off how chat templating works!"},
... ]

>>> tokenizer.apply_chat_template(chat, tokenize=False)
"<s>[INST] Hello, how are you? [/INST]I'm doing great. How can I help you today?</s> [INST] I'd like to show off how chat templating works! [/INST]"
```

Notice how the tokenizer has added the control tokens [INST] and [/INST] to indicate the start and end of 
user messages (but not assistant messages!), and the entire chat is condensed into a single string. 
If we use `tokenize=True`, which is the default setting, that string will also be tokenized for us.

Now, try the same code, but swap in the `HuggingFaceH4/zephyr-7b-beta` model instead, and you should get:

```text
<|user|>
Hello, how are you?</s>
<|assistant|>
I'm doing great. How can I help you today?</s>
<|user|>
I'd like to show off how chat templating works!</s>
```

Both Zephyr and Mistral-Instruct were fine-tuned from the same base model, `Mistral-7B-v0.1`. However, they were trained
with totally different chat formats. Without chat templates, you would have to write manual formatting code for each
model, and it's very easy to make minor errors that hurt performance! Chat templates handle the details of formatting 
for you, allowing you to write universal code that works for any model.


## How do I use chat templates?

As you can see in the example above, chat templates are easy to use. Simply build a list of messages, with `role`
and `content` keys, and then pass it to the [`~PreTrainedTokenizer.apply_chat_template`] method. Once you do that,
you'll get output that's ready to go! When using chat templates as input for model generation, it's also a good idea
to use `add_generation_prompt=True` to add a [generation prompt](#what-are-generation-prompts). 

Here's an example of preparing input for `model.generate()`, using `Zephyr` again:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "HuggingFaceH4/zephyr-7b-beta"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint)  # You may want to use bfloat16 and/or move to GPU here

messages = [
    {
        "role": "system",
        "content": "You are a friendly chatbot who always responds in the style of a pirate",
    },
    {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
 ]
tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
print(tokenizer.decode(tokenized_chat[0]))
```
This will yield a string in the input format that Zephyr expects. 
```text
<|system|>
You are a friendly chatbot who always responds in the style of a pirate</s> 
<|user|>
How many helicopters can a human eat in one sitting?</s> 
<|assistant|>
```

Now that our input is formatted correctly for Zephyr, we can use the model to generate a response to the user's question:

```python
outputs = model.generate(tokenized_chat, max_new_tokens=128) 
print(tokenizer.decode(outputs[0]))
```

This will yield:

```text
<|system|>
You are a friendly chatbot who always responds in the style of a pirate</s> 
<|user|>
How many helicopters can a human eat in one sitting?</s> 
<|assistant|>
Matey, I'm afraid I must inform ye that humans cannot eat helicopters. Helicopters are not food, they are flying machines. Food is meant to be eaten, like a hearty plate o' grog, a savory bowl o' stew, or a delicious loaf o' bread. But helicopters, they be for transportin' and movin' around, not for eatin'. So, I'd say none, me hearties. None at all.
```

Arr, 'twas easy after all!

## Is there an automated pipeline for chat?

Yes, there is! Our text generation pipelines support chat inputs, which makes it easy to use chat models. In the past,
we used to use a dedicated "ConversationalPipeline" class, but this has now been deprecated and its functionality
has been merged into the [`TextGenerationPipeline`]. Let's try the `Zephyr` example again, but this time using 
a pipeline:

```python
from transformers import pipeline

pipe = pipeline("text-generation", "HuggingFaceH4/zephyr-7b-beta")
messages = [
    {
        "role": "system",
        "content": "You are a friendly chatbot who always responds in the style of a pirate",
    },
    {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
]
print(pipe(messages, max_new_tokens=128)[0]['generated_text'][-1])  # Print the assistant's response
```

```text
{'role': 'assistant', 'content': "Matey, I'm afraid I must inform ye that humans cannot eat helicopters. Helicopters are not food, they are flying machines. Food is meant to be eaten, like a hearty plate o' grog, a savory bowl o' stew, or a delicious loaf o' bread. But helicopters, they be for transportin' and movin' around, not for eatin'. So, I'd say none, me hearties. None at all."}
```

The pipeline will take care of all the details of tokenization and calling `apply_chat_template` for you -
once the model has a chat template, all you need to do is initialize the pipeline and pass it the list of messages!

## What are "generation prompts"?

You may have noticed that the `apply_chat_template` method has an `add_generation_prompt` argument. This argument tells
the template to add tokens that indicate the start of a bot response. For example, consider the following chat:

```python
messages = [
    {"role": "user", "content": "Hi there!"},
    {"role": "assistant", "content": "Nice to meet you!"},
    {"role": "user", "content": "Can I ask a question?"}
]
```

Here's what this will look like without a generation prompt, for a model that uses standard "ChatML" formatting:

```python
tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
"""<|im_start|>user
Hi there!<|im_end|>
<|im_start|>assistant
Nice to meet you!<|im_end|>
<|im_start|>user
Can I ask a question?<|im_end|>
"""
```

And here's what it looks like **with** a generation prompt:

```python
tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
"""<|im_start|>user
Hi there!<|im_end|>
<|im_start|>assistant
Nice to meet you!<|im_end|>
<|im_start|>user
Can I ask a question?<|im_end|>
<|im_start|>assistant
"""
```

Note that this time, we've added the tokens that indicate the start of a bot response. This ensures that when the model
generates text it will write a bot response instead of doing something unexpected, like continuing the user's 
message. Remember, chat models are still just language models - they're trained to continue text, and chat is just a 
special kind of text to them! You need to guide them with appropriate control tokens, so they know what they're 
supposed to be doing.

Not all models require generation prompts. Some models, like LLaMA, don't have any
special tokens before bot responses. In these cases, the `add_generation_prompt` argument will have no effect. The exact
effect that `add_generation_prompt` has will depend on the template being used.

## What does "continue_final_message" do?

When passing a list of messages to `apply_chat_template` or `TextGenerationPipeline`, you can choose
to format the chat so the model will continue the final message in the chat instead of starting a new one. This is done
by removing any end-of-sequence tokens that indicate the end of the final message, so that the model will simply
extend the final message when it begins to generate text. This is useful for "prefilling" the model's response. 

Here's an example:

```python
chat = [
    {"role": "user", "content": "Can you format the answer in JSON?"},
    {"role": "assistant", "content": '{"name": "'},
]

formatted_chat = tokenizer.apply_chat_template(chat, tokenize=True, return_dict=True, continue_final_message=True)
model.generate(**formatted_chat)
```

The model will generate text that continues the JSON string, rather than starting a new message. This approach
can be very useful for improving the accuracy of the model's instruction-following when you know how you want
it to start its replies.

Because `add_generation_prompt` adds the tokens that start a new message, and `continue_final_message` removes any
end-of-message tokens from the final message, it does not make sense to use them together. As a result, you'll
get an error if you try!

<Tip>

The default behaviour of `TextGenerationPipeline` is to set `add_generation_prompt=True` so that it starts a new
message. However, if the final message in the input chat has the "assistant" role, it will assume that this message is 
a prefill and switch to `continue_final_message=True` instead, because most models do not support multiple 
consecutive assistant messages. You can override this behaviour by explicitly passing the `continue_final_message` 
argument when calling the pipeline.

</Tip>

## Can I use chat templates in training?

Yes! This is a good way to ensure that the chat template matches the tokens the model sees during training.
We recommend that you apply the chat template as a preprocessing step for your dataset. After this, you
can simply continue like any other language model training task. When training, you should usually set 
`add_generation_prompt=False`, because the added tokens to prompt an assistant response will not be helpful during 
training. Let's see an example:

```python
from transformers import AutoTokenizer
from datasets import Dataset

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")

chat1 = [
    {"role": "user", "content": "Which is bigger, the moon or the sun?"},
    {"role": "assistant", "content": "The sun."}
]
chat2 = [
    {"role": "user", "content": "Which is bigger, a virus or a bacterium?"},
    {"role": "assistant", "content": "A bacterium."}
]

dataset = Dataset.from_dict({"chat": [chat1, chat2]})
dataset = dataset.map(lambda x: {"formatted_chat": tokenizer.apply_chat_template(x["chat"], tokenize=False, add_generation_prompt=False)})
print(dataset['formatted_chat'][0])
```
And we get:
```text
<|user|>
Which is bigger, the moon or the sun?</s>
<|assistant|>
The sun.</s>
```

From here, just continue training like you would with a standard language modelling task, using the `formatted_chat` column.

<Tip>

By default, some tokenizers add special tokens like `<bos>` and `<eos>` to text they tokenize. Chat templates should 
already include all the special tokens they need, and so additional special tokens will often be incorrect or 
duplicated, which will hurt model performance.

Therefore, if you format text with `apply_chat_template(tokenize=False)`, you should set the argument
`add_special_tokens=False` when you tokenize that text later. If you use `apply_chat_template(tokenize=True)`, you don't need to worry about this!

</Tip>

## Advanced: Extra inputs to chat templates

The only argument that `apply_chat_template` requires is `messages`. However, you can pass any keyword
argument to `apply_chat_template` and it will be accessible inside the template. This gives you a lot of freedom to use
chat templates for many things. There are no restrictions on the names or the format of these arguments - you can pass
strings, lists, dicts or whatever else you want. 

That said, there are some common use-cases for these extra arguments,
such as passing tools for function calling, or documents for retrieval-augmented generation. In these common cases,
we have some opinionated recommendations about what the names and formats of these arguments should be, which are
described in the sections below. We encourage model authors to make their chat templates compatible with this format,
to make it easy to transfer tool-calling code between models.

## Advanced: Tool use / function calling

"Tool use" LLMs can choose to call functions as external tools before generating an answer. When passing tools
to a tool-use model, you can simply pass a list of functions to the `tools` argument:

```python
import datetime

def current_time():
    """Get the current local time as a string."""
    return str(datetime.now())

def multiply(a: float, b: float):
    """
    A function that multiplies two numbers
    
    Args:
        a: The first number to multiply
        b: The second number to multiply
    """
    return a * b

tools = [current_time, multiply]

model_input = tokenizer.apply_chat_template(
    messages,
    tools=tools
)
```

In order for this to work correctly, you should write your functions in the format above, so that they can be parsed
correctly as tools. Specifically, you should follow these rules:

- The function should have a descriptive name
- Every argument must have a type hint
- The function must have a docstring in the standard Google style (in other words, an initial function description  
  followed by an `Args:` block that describes the arguments, unless the function does not have any arguments. 
- Do not include types in the `Args:` block. In other words, write `a: The first number to multiply`, not
  `a (int): The first number to multiply`. Type hints should go in the function header instead.
- The function can have a return type and a `Returns:` block in the docstring. However, these are optional
  because most tool-use models ignore them.

### Passing tool results to the model

The sample code above is enough to list the available tools for your model, but what happens if it wants to actually use
one? If that happens, you should:

1. Parse the model's output to get the tool name(s) and arguments.
2. Add the model's tool call(s) to the conversation.
3. Call the corresponding function(s) with those arguments.
4. Add the result(s) to the conversation

### A complete tool use example

Let's walk through a tool use example, step by step. For this example, we will use an 8B `Hermes-2-Pro` model,
as it is one of the highest-performing tool-use models in its size category at the time of writing. If you have the
memory, you can consider using a larger model instead like [Command-R](https://huggingface.co/CohereForAI/c4ai-command-r-v01)
or [Mixtral-8x22B](https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1), both of which also support tool use
and offer even stronger performance.

First, let's load our model and tokenizer:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "NousResearch/Hermes-2-Pro-Llama-3-8B"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.bfloat16, device_map="auto")
```

Next, let's define a list of tools:

```python
def get_current_temperature(location: str, unit: str) -> float:
    """
    Get the current temperature at a location.
    
    Args:
        location: The location to get the temperature for, in the format "City, Country"
        unit: The unit to return the temperature in. (choices: ["celsius", "fahrenheit"])
    Returns:
        The current temperature at the specified location in the specified units, as a float.
    """
    return 22.  # A real function should probably actually get the temperature!

def get_current_wind_speed(location: str) -> float:
    """
    Get the current wind speed in km/h at a given location.
    
    Args:
        location: The location to get the temperature for, in the format "City, Country"
    Returns:
        The current wind speed at the given location in km/h, as a float.
    """
    return 6.  # A real function should probably actually get the wind speed!

tools = [get_current_temperature, get_current_wind_speed]
```

Now, let's set up a conversation for our bot:

```python
messages = [
  {"role": "system", "content": "You are a bot that responds to weather queries. You should reply with the unit used in the queried location."},
  {"role": "user", "content": "Hey, what's the temperature in Paris right now?"}
]
```

Now, let's apply the chat template and generate a response:

```python
inputs = tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, return_dict=True, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}
out = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(out[0][len(inputs["input_ids"][0]):]))
```

And we get:

```text
<tool_call>
{"arguments": {"location": "Paris, France", "unit": "celsius"}, "name": "get_current_temperature"}
</tool_call><|im_end|>
```

The model has called the function with valid arguments, in the format requested by the function docstring. It has
inferred that we're most likely referring to the Paris in France, and it remembered that, as the home of SI units,
the temperature in France should certainly be displayed in Celsius.

<Tip>

The output format above is specific to the `Hermes-2-Pro` model we're using in this example. Other models may emit different
tool call formats, and you may need to do some manual parsing at this step. For example, `Llama-3.1` models will emit
slightly different JSON, with `parameters` instead of `arguments`. Regardless of the format the model outputs, you 
should add the tool call to the conversation in the format below, with `tool_calls`, `function` and `arguments` keys. 

</Tip>

Next, let's append the model's tool call to the conversation.

```python
tool_call = {"name": "get_current_temperature", "arguments": {"location": "Paris, France", "unit": "celsius"}}
messages.append({"role": "assistant", "tool_calls": [{"type": "function", "function": tool_call}]})
```

<Tip warning={true}>

If you're familiar with the OpenAI API, you should pay attention to an important difference here - the `tool_call` is
a dict, but in the OpenAI API it's a JSON string. Passing a string may cause errors or strange model behaviour!

</Tip>

Now that we've added the tool call to the conversation, we can call the function and append the result to the
conversation. Since we're just using a dummy function for this example that always returns 22.0, we can just append 
that result directly.

```python
messages.append({"role": "tool", "name": "get_current_temperature", "content": "22.0"})
```

<Tip>

Some model architectures, notably Mistral/Mixtral, also require a `tool_call_id` here, which should be
9 randomly-generated alphanumeric characters, and assigned to the `id` key of the tool call
dictionary. The same key should also be assigned to the `tool_call_id` key of the tool response dictionary below, so 
that tool calls can be matched to tool responses. So, for Mistral/Mixtral models, the code above would be:

```python
tool_call_id = "9Ae3bDc2F"  # Random ID, 9 alphanumeric characters
tool_call = {"name": "get_current_temperature", "arguments": {"location": "Paris, France", "unit": "celsius"}}
messages.append({"role": "assistant", "tool_calls": [{"type": "function", "id": tool_call_id, "function": tool_call}]})
```

and

```python
messages.append({"role": "tool", "tool_call_id": tool_call_id, "name": "get_current_temperature", "content": "22.0"})
```

</Tip>

Finally, let's let the assistant read the function outputs and continue chatting with the user:

```python
inputs = tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, return_dict=True, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}
out = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(out[0][len(inputs["input_ids"][0]):]))
```

And we get:

```text
The current temperature in Paris, France is 22.0 ° Celsius.<|im_end|>
```

Although this was a simple demo with dummy tools and a single call, the same technique works with 
multiple real tools and longer conversations. This can be a powerful way to extend the capabilities of conversational
agents with real-time information, computational tools like calculators, or access to large databases.

### Understanding tool schemas

Each function you pass to the `tools` argument of `apply_chat_template` is converted into a 
[JSON schema](https://json-schema.org/learn/getting-started-step-by-step). These schemas
are then passed to the model chat template. In other words, tool-use models do not see your functions directly, and they
never see the actual code inside them. What they care about is the function **definitions** and the **arguments** they
need to pass to them - they care about what the tools do and how to use them, not how they work! It is up to you
to read their outputs, detect if they have requested to use a tool, pass their arguments to the tool function, and
return the response in the chat.

Generating JSON schemas to pass to the template should be automatic and invisible as long as your functions
follow the specification above, but if you encounter problems, or you simply want more control over the conversion, 
you can handle the conversion manually. Here is an example of a manual schema conversion.

```python
from transformers.utils import get_json_schema

def multiply(a: float, b: float):
    """
    A function that multiplies two numbers
    
    Args:
        a: The first number to multiply
        b: The second number to multiply
    """
    return a * b

schema = get_json_schema(multiply)
print(schema)
```

This will yield:

```json
{
  "type": "function", 
  "function": {
    "name": "multiply", 
    "description": "A function that multiplies two numbers", 
    "parameters": {
      "type": "object", 
      "properties": {
        "a": {
          "type": "number", 
          "description": "The first number to multiply"
        }, 
        "b": {
          "type": "number",
          "description": "The second number to multiply"
        }
      }, 
      "required": ["a", "b"]
    }
  }
}
```

If you wish, you can edit these schemas, or even write them from scratch yourself without using `get_json_schema` at 
all. JSON schemas can be passed directly to the `tools` argument of 
`apply_chat_template` - this gives you a lot of power to define precise schemas for more complex functions. Be careful,
though - the more complex your schemas, the more likely the model is to get confused when dealing with them! We 
recommend simple function signatures where possible, keeping arguments (and especially complex, nested arguments) 
to a minimum.

Here is an example of defining schemas by hand, and passing them directly to `apply_chat_template`:

```python
# A simple function that takes no arguments
current_time = {
  "type": "function", 
  "function": {
    "name": "current_time",
    "description": "Get the current local time as a string.",
    "parameters": {
      'type': 'object',
      'properties': {}
    }
  }
}

# A more complete function that takes two numerical arguments
multiply = {
  'type': 'function',
  'function': {
    'name': 'multiply',
    'description': 'A function that multiplies two numbers', 
    'parameters': {
      'type': 'object', 
      'properties': {
        'a': {
          'type': 'number',
          'description': 'The first number to multiply'
        }, 
        'b': {
          'type': 'number', 'description': 'The second number to multiply'
        }
      }, 
      'required': ['a', 'b']
    }
  }
}

model_input = tokenizer.apply_chat_template(
    messages,
    tools = [current_time, multiply]
)
```

## Advanced: Retrieval-augmented generation

"Retrieval-augmented generation" or "RAG" LLMs can search a corpus of documents for information before responding
to a query. This allows models to vastly expand their knowledge base beyond their limited context size. Our 
recommendation for RAG models is that their template
should accept a `documents` argument. This should be a list of documents, where each "document"
is a single dict with `title` and `contents` keys, both of which are strings. Because this format is much simpler
than the JSON schemas used for tools, no helper functions are necessary.

Here's an example of a RAG template in action:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the model and tokenizer
model_id = "CohereForAI/c4ai-command-r-v01-4bit"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
device = model.device # Get the device the model is loaded on

# Define conversation input
conversation = [
    {"role": "user", "content": "What has Man always dreamed of?"}
]

# Define documents for retrieval-based generation
documents = [
    {
        "title": "The Moon: Our Age-Old Foe", 
        "text": "Man has always dreamed of destroying the moon. In this essay, I shall..."
    },
    {
        "title": "The Sun: Our Age-Old Friend",
        "text": "Although often underappreciated, the sun provides several notable benefits..."
    }
]

# Tokenize conversation and documents using a RAG template, returning PyTorch tensors.
input_ids = tokenizer.apply_chat_template(
    conversation=conversation,
    documents=documents,
    chat_template="rag",
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt").to(device)

# Generate a response 
gen_tokens = model.generate(
    input_ids,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.3,
    )

# Decode and print the generated text along with generation prompt
gen_text = tokenizer.decode(gen_tokens[0])
print(gen_text)
```

<Tip>

The `documents` input for retrieval-augmented generation is not widely supported, and many models have chat templates which simply ignore this input.

To verify if a model supports the `documents` input, you can read its model card, or `print(tokenizer.chat_template)` to see if the `documents` key is used anywhere.

One model class that does support it, though, is Cohere's [Command-R](https://huggingface.co/CohereForAI/c4ai-command-r-08-2024) and [Command-R+](https://huggingface.co/CohereForAI/c4ai-command-r-plus-08-2024), through their `rag` chat template. You can see additional examples of grounded generation using this feature in their model cards.

</Tip>



## Advanced: How do chat templates work?

The chat template for a model is stored on the `tokenizer.chat_template` attribute. If no chat template is set, the
default template for that model class is used instead. Let's take a look at a `Zephyr` chat template, though note this
one is a little simplified from the actual one!

```
{%- for message in messages %}
    {{- '<|' + message['role'] + '|>\n' }}
    {{- message['content'] + eos_token }}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|assistant|>\n' }}
{%- endif %}
```

If you've never seen one of these before, this is a [Jinja template](https://jinja.palletsprojects.com/en/3.1.x/templates/).
Jinja is a templating language that allows you to write simple code that generates text. In many ways, the code and
syntax resembles Python. In pure Python, this template would look something like this:

```python
for message in messages:
    print(f'<|{message["role"]}|>')
    print(message['content'] + eos_token)
if add_generation_prompt:
    print('<|assistant|>')
```

Effectively, the template does three things:
1. For each message, print the role enclosed in `<|` and `|>`, like `<|user|>` or `<|assistant|>`.
2. Next, print the content of the message, followed by the end-of-sequence token.
3. Finally, if `add_generation_prompt` is set, print the assistant token, so that the model knows to start generating
   an assistant response.

This is a pretty simple template but Jinja gives you a lot of flexibility to do more complex things! Let's see a Jinja
template that can format inputs similarly to the way LLaMA formats them (note that the real LLaMA template includes 
handling for default system messages and slightly different system message handling in general - don't use this one 
in your actual code!)

```
{%- for message in messages %}
    {%- if message['role'] == 'user' %}
        {{- bos_token + '[INST] ' + message['content'] + ' [/INST]' }}
    {%- elif message['role'] == 'system' %}
        {{- '<<SYS>>\\n' + message['content'] + '\\n<</SYS>>\\n\\n' }}
    {%- elif message['role'] == 'assistant' %}
        {{- ' '  + message['content'] + ' ' + eos_token }}
    {%- endif %}
{%- endfor %}
```

Hopefully if you stare at this for a little bit you can see what this template is doing - it adds specific tokens like
`[INST]` and `[/INST]` based on the role of each message. User, assistant and system messages are clearly
distinguishable to the model because of the tokens they're wrapped in.

## Advanced: Adding and editing chat templates

### How do I create a chat template?

Simple, just write a jinja template and set `tokenizer.chat_template`. You may find it easier to start with an 
existing template from another model and simply edit it for your needs! For example, we could take the LLaMA template
above and add "[ASST]" and "[/ASST]" to assistant messages:

```
{%- for message in messages %}
    {%- if message['role'] == 'user' %}
        {{- bos_token + '[INST] ' + message['content'].strip() + ' [/INST]' }}
    {%- elif message['role'] == 'system' %}
        {{- '<<SYS>>\\n' + message['content'].strip() + '\\n<</SYS>>\\n\\n' }}
    {%- elif message['role'] == 'assistant' %}
        {{- '[ASST] '  + message['content'] + ' [/ASST]' + eos_token }}
    {%- endif %}
{%- endfor %}
```

Now, simply set the `tokenizer.chat_template` attribute. Next time you use [`~PreTrainedTokenizer.apply_chat_template`], it will
use your new template! This attribute will be saved in the `tokenizer_config.json` file, so you can use
[`~utils.PushToHubMixin.push_to_hub`] to upload your new template to the Hub and make sure everyone's using the right
template for your model!

```python
template = tokenizer.chat_template
template = template.replace("SYS", "SYSTEM")  # Change the system token
tokenizer.chat_template = template  # Set the new template
tokenizer.push_to_hub("model_name")  # Upload your new template to the Hub!
```

The method [`~PreTrainedTokenizer.apply_chat_template`] which uses your chat template is called by the [`TextGenerationPipeline`] class, so 
once you set the correct chat template, your model will automatically become compatible with [`TextGenerationPipeline`].

<Tip>
If you're fine-tuning a model for chat, in addition to setting a chat template, you should probably add any new chat
control tokens as special tokens in the tokenizer. Special tokens are never split, 
ensuring that your control tokens are always handled as single tokens rather than being tokenized in pieces. You 
should also set the tokenizer's `eos_token` attribute to the token that marks the end of assistant generations in your
template. This will ensure that text generation tools can correctly figure out when to stop generating text.
</Tip>


### Why do some models have multiple templates?

Some models use different templates for different use cases. For example, they might use one template for normal chat
and another for tool-use, or retrieval-augmented generation. In these cases, `tokenizer.chat_template` is a dictionary.
This can cause some confusion, and where possible, we recommend using a single template for all use-cases. You can use
Jinja statements like `if tools is defined` and `{% macro %}` definitions to easily wrap multiple code paths in a
single template.

When a tokenizer has multiple templates, `tokenizer.chat_template` will be a `dict`, where each key is the name
of a template. The `apply_chat_template` method has special handling for certain template names: Specifically, it will
look for a template named `default` in most cases, and will raise an error if it can't find one. However, if a template
named `tool_use` exists when the user has passed a `tools` argument, it will use that instead. To access templates
with other names, pass the name of the template you want to the `chat_template` argument of
`apply_chat_template()`.

We find that this can be a bit confusing for users, though - so if you're writing a template yourself, we recommend
trying to put it all in a single template where possible!

### What template should I use?

When setting the template for a model that's already been trained for chat, you should ensure that the template
exactly matches the message formatting that the model saw during training, or else you will probably experience
performance degradation. This is true even if you're training the model further - you will probably get the best 
performance if you keep the chat tokens constant. This is very analogous to tokenization - you generally get the
best performance for inference or fine-tuning when you precisely match the tokenization used during training.

If you're training a model from scratch, or fine-tuning a base language model for chat, on the other hand,
you have a lot of freedom to choose an appropriate template! LLMs are smart enough to learn to handle lots of different
input formats. One popular choice is the `ChatML` format, and this is a good, flexible choice for many use-cases. 
It looks like this:

```
{%- for message in messages %}
    {{- '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}
{%- endfor %}
```

If you like this one, here it is in one-liner form, ready to copy into your code. The one-liner also includes
handy support for [generation prompts](#what-are-generation-prompts), but note that it doesn't add BOS or EOS tokens!
If your model expects those, they won't be added automatically by `apply_chat_template` - in other words, the
text will be tokenized with `add_special_tokens=False`. This is to avoid potential conflicts between the template and
the `add_special_tokens` logic. If your model expects special tokens, make sure to add them to the template!

```python
tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
```

This template wraps each message in `<|im_start|>` and `<|im_end|>` tokens, and simply writes the role as a string, which
allows for flexibility in the roles you train with. The output looks like this:

```text
<|im_start|>system
You are a helpful chatbot that will do its best not to say anything so stupid that people tweet about it.<|im_end|>
<|im_start|>user
How are you?<|im_end|>
<|im_start|>assistant
I'm doing great!<|im_end|>
```

The "user", "system" and "assistant" roles are the standard for chat, and we recommend using them when it makes sense,
particularly if you want your model to operate well with [`TextGenerationPipeline`]. However, you are not limited
to these roles - templating is extremely flexible, and any string can be a role.

### I want to add some chat templates! How should I get started?

If you have any chat models, you should set their `tokenizer.chat_template` attribute and test it using
[`~PreTrainedTokenizer.apply_chat_template`], then push the updated tokenizer to the Hub. This applies even if you're
not the model owner - if you're using a model with an empty chat template, or one that's still using the default class
template, please open a [pull request](https://huggingface.co/docs/hub/repositories-pull-requests-discussions) to the model repository so that this attribute can be set properly!

Once the attribute is set, that's it, you're done! `tokenizer.apply_chat_template` will now work correctly for that
model, which means it is also automatically supported in places like `TextGenerationPipeline`!

By ensuring that models have this attribute, we can make sure that the whole community gets to use the full power of
open-source models. Formatting mismatches have been haunting the field and silently harming performance for too long - 
it's time to put an end to them!

## Advanced: Template writing tips

<Tip>

The easiest way to get started with writing Jinja templates is to take a look at some existing ones. You can use
`print(tokenizer.chat_template)` for any chat model to see what template it's using. In general, models that support tool use have 
much more complex templates than other models - so when you're just getting started, they're probably a bad example
to learn from! You can also take a look at the 
[Jinja documentation](https://jinja.palletsprojects.com/en/3.1.x/templates/#synopsis) for details
of general Jinja formatting and syntax.

</Tip>

Jinja templates in `transformers` are identical to Jinja templates elsewhere. The main thing to know is that 
the conversation history will be accessible inside your template as a variable called `messages`.  
You will be able to access `messages` in your template just like you can in Python, which means you can loop over 
it with `{% for message in messages %}` or access individual messages with `{{ messages[0] }}`, for example.

You can also use the following tips to write clean, efficient Jinja templates:

### Trimming whitespace

By default, Jinja will print any whitespace that comes before or after a block. This can be a problem for chat
templates, which generally want to be very precise with whitespace! To avoid this, we strongly recommend writing
your templates like this:

```
{%- for message in messages %}
    {{- message['role'] + message['content'] }}
{%- endfor %}
```

rather than like this:

```
{% for message in messages %}
    {{ message['role'] + message['content'] }}
{% endfor %}
```

Adding `-` will strip any whitespace that comes before the block. The second example looks innocent, but the newline
and indentation may end up being included in the output, which is probably not what you want!

### Special variables

Inside your template, you will have access several special variables. The most important of these is `messages`, 
which contains the chat history as a list of message dicts. However, there are several others. Not every
variable will be used in every template. The most common other variables are:

- `tools` contains a list of tools in JSON schema format. Will be `None` or undefined if no tools are passed.
- `documents` contains a list of documents in the format `{"title": "Title", "contents": "Contents"}`, used for retrieval-augmented generation. Will be `None` or undefined if no documents are passed.
- `add_generation_prompt` is a bool that is `True` if the user has requested a generation prompt, and `False` otherwise. If this is set, your template should add the header for an assistant message to the end of the conversation. If your model doesn't have a specific header for assistant messages, you can ignore this flag.
- **Special tokens** like `bos_token` and `eos_token`. These are extracted from `tokenizer.special_tokens_map`. The exact tokens available inside each template will differ depending on the parent tokenizer.

<Tip>

You can actually pass any `kwarg` to `apply_chat_template`, and it will be accessible inside the template as a variable. In general,
we recommend trying to stick to the core variables above, as it will make your model harder to use if users have
to write custom code to pass model-specific `kwargs`. However, we're aware that this field moves quickly, so if you
have a new use-case that doesn't fit in the core API, feel free to use a new `kwarg` for it! If a new `kwarg`
becomes common we may promote it into the core API and create a standard, documented format for it.

</Tip>

### Callable functions

There is also a short list of callable functions available to you inside your templates. These are:

- `raise_exception(msg)`: Raises a `TemplateException`. This is useful for debugging, and for telling users when they're
doing something that your template doesn't support.
- `strftime_now(format_str)`: Equivalent to `datetime.now().strftime(format_str)` in Python. This is used for getting
the current date/time in a specific format, which is sometimes included in system messages.

### Compatibility with non-Python Jinja

There are multiple implementations of Jinja in various languages. They generally have the same syntax,
but a key difference is that when you're writing a template in Python you can use Python methods, such as
`.lower()` on strings or `.items()` on dicts. This will break if someone tries to use your template on a non-Python
implementation of Jinja. Non-Python implementations are particularly common in deployment environments, where JS
and Rust are very popular. 

Don't panic, though! There are a few easy changes you can make to your templates to ensure they're compatible across
all implementations of Jinja:

- Replace Python methods with Jinja filters. These usually have the same name, for example `string.lower()` becomes
  `string|lower`, and `dict.items()` becomes `dict|items`. One notable change is that `string.strip()` becomes `string|trim`.
  See the [list of built-in filters](https://jinja.palletsprojects.com/en/3.1.x/templates/#builtin-filters)
  in the Jinja documentation for more.
- Replace `True`, `False` and `None`, which are Python-specific, with `true`, `false` and `none`.
- Directly rendering a dict or list may give different results in other implementations (for example, string entries
  might change from single-quoted to double-quoted). Adding the `tojson` filter can help to ensure consistency here.

### Writing generation prompts

We mentioned above that `add_generation_prompt` is a special variable that will be accessible inside your template,
and is controlled by the user setting the `add_generation_prompt` flag. If your model expects a header for
assistant messages, then your template must support adding the header when `add_generation_prompt` is set.

Here is an example of a template that formats messages ChatML-style, with generation prompt support:

```text
{{- bos_token }}
{%- for message in messages %}
    {{- '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}
```

The exact content of the assistant header will depend on your specific model, but it should always be **the string
that represents the start of an assistant message**, so that if the user applies your template with 
`add_generation_prompt=True` and then generates text, the model will write an assistant response. Also note that some
models do not need a generation prompt, because assistant messages always begin immediately after user messages. 
This is particularly common for LLaMA and Mistral models, where assistant messages begin immediately after the `[/INST]`
token that ends user messages. In these cases, the template can ignore the `add_generation_prompt` flag.

Generation prompts are important! If your model requires a generation prompt but it is not set in the template, then
model generations will likely be severely degraded, or the model may display unusual behaviour like continuing 
the final user message! 

### Writing and debugging larger templates

When this feature was introduced, most templates were quite small, the Jinja equivalent of a "one-liner" script. 
However, with new models and features like tool-use and RAG, some templates can be 100 lines long or more. When
writing templates like these, it's a good idea to write them in a separate file, using a text editor. You can easily 
extract a chat template to a file:

```python
open("template.jinja", "w").write(tokenizer.chat_template)
```

Or load the edited template back into the tokenizer:

```python
tokenizer.chat_template = open("template.jinja").read()
```

As an added bonus, when you write a long, multi-line template in a separate file, line numbers in that file will
exactly correspond to line numbers in template parsing or execution errors. This will make it much easier to
identify the source of issues.

### Writing templates for tools

Although chat templates do not enforce a specific API for tools (or for anything, really), we recommend 
template authors try to stick to a standard API where possible. The whole point of chat templates is to allow code
to be transferable across models, so deviating from the standard tools API means users will have to write
custom code to use tools with your model. Sometimes it's unavoidable, but often with clever templating you can
make the standard API work!

Below, we'll list the elements of the standard API, and give tips on writing templates that will work well with it.

#### Tool definitions

Your template should expect that the variable `tools` will either be null (if no tools are passed), or is a list 
of JSON schema dicts. Our chat template methods allow users to pass tools as either JSON schema or Python functions, but when
functions are passed, we automatically generate JSON schema and pass that to your template. As a result, the 
`tools` variable that your template receives will always be a list of JSON schema. Here is
a sample tool JSON schema:

```json
{
  "type": "function", 
  "function": {
    "name": "multiply", 
    "description": "A function that multiplies two numbers", 
    "parameters": {
      "type": "object", 
      "properties": {
        "a": {
          "type": "number", 
          "description": "The first number to multiply"
        }, 
        "b": {
          "type": "number",
          "description": "The second number to multiply"
        }
      }, 
      "required": ["a", "b"]
    }
  }
}
```

And here is some example code for handling tools in your chat template. Remember, this is just an example for a
specific format - your model will probably need different formatting!

```text
{%- if tools %}
    {%- for tool in tools %}
        {{- '<tool>' + tool['function']['name'] + '\n' }}
        {%- for argument in tool['function']['parameters']['properties'] %}
            {{- argument + ': ' + tool['function']['parameters']['properties'][argument]['description'] + '\n' }}
        {%- endfor %}
        {{- '\n</tool>' }}
    {%- endif %}
{%- endif %}
```

The specific tokens and tool descriptions your template renders should of course be chosen to match the ones your model
was trained with. There is no requirement that your **model** understands JSON schema input, only that your template can translate
JSON schema into your model's format. For example, [Command-R](https://huggingface.co/CohereForAI/c4ai-command-r-plus-08-2024) 
was trained with tools defined using Python function headers, but the Command-R tool template accepts JSON schema, 
converts types internally and renders the input tools as Python headers. You can do a lot with templates!

#### Tool calls

Tool calls, if present, will be a list attached to a message with the "assistant" role. Note that `tool_calls` is 
always a list, even though most tool-calling models only support single tool calls at a time, which means
the list will usually only have a single element. Here is a sample message dict containing a tool call:

```json
{
  "role": "assistant",
  "tool_calls": [
    {
      "type": "function",
      "function": {
        "name": "multiply",
        "arguments": {
          "a": 5,
          "b": 6
        }
      }
    }
  ]
}
```

And a common pattern for handling them would be something like this:

```text
{%- if message['role'] == 'assistant' and 'tool_calls' in message %}
    {%- for tool_call in message['tool_calls'] %}
            {{- '<tool_call>' + tool_call['function']['name'] + '\n' + tool_call['function']['arguments']|tojson + '\n</tool_call>' }}
        {%- endif %}
    {%- endfor %}
{%- endif %}
```

Again, you should render the tool call with the formatting and special tokens that your model expects.

#### Tool responses

Tool responses have a simple format: They are a message dict with the "tool" role, a "name" key giving the name
of the called function, and a "content" key containing the result of the tool call. Here is a sample tool response:

```json
{
  "role": "tool",
  "name": "multiply",
  "content": "30"
}
```

You don't need to use all of the keys in the tool response. For example, if your model doesn't expect the function
name to be included in the tool response, then rendering it can be as simple as:

```text
{%- if message['role'] == 'tool' %}
    {{- "<tool_result>" + message['content'] + "</tool_result>" }}
{%- endif %}
```

Again, remember that the actual formatting and special tokens are model-specific - you should take a lot of care
to ensure that tokens, whitespace and everything else exactly match the format your model was trained with!
