<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Tools and RAG

The [`~PreTrainedTokenizerBase.apply_chat_template`] method supports virtually any additional argument types - strings, lists, dicts - besides the chat message. This makes it possible to use chat templates for many use cases.

This guide will demonstrate how to use chat templates with tools and retrieval-augmented generation (RAG).

## Tools

Tools are functions a large language model (LLM) can call to perform specific tasks. It is a powerful way to extend the capabilities of conversational agents with real-time information, computational tools, or access to large databases.

Follow the rules below when creating a tool.

1. The function should have a descriptive name.
2. The function arguments must have a type hint in the function header (don't include in the `Args` block).
3. The function must have a [Google-style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) docstring.
4. The function can have a return type and `Returns` block, but these are optional because most tool use models ignore them.

An example tool to get temperature and wind speed is shown below.

```py
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

Load a model and tokenizer that supports tool use like [NousResearch/Hermes-2-Pro-Llama-3-8B](https://hf.co/NousResearch/Hermes-2-Pro-Llama-3-8B), but you can also consider a larger model like [Command-R](./model_doc/cohere) and [Mixtral-8x22B](./model_doc/mixtral) if your hardware can support it.

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained( "NousResearch/Hermes-2-Pro-Llama-3-8B")
tokenizer = AutoTokenizer.from_pretrained( "NousResearch/Hermes-2-Pro-Llama-3-8B")
model = AutoModelForCausalLM.from_pretrained( "NousResearch/Hermes-2-Pro-Llama-3-8B", torch_dtype=torch.bfloat16, device_map="auto")
```

Create a chat message.

```py
messages = [
  {"role": "system", "content": "You are a bot that responds to weather queries. You should reply with the unit used in the queried location."},
  {"role": "user", "content": "Hey, what's the temperature in Paris right now?"}
]
```

Use [`~PreTrainedTokenizerBase.apply_chat_template`] on the messages and pass the list of tools to the `tools` parameter. Then you can pass the inputs to the model for generation.

```py
inputs = tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, return_dict=True, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}
outputs = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):]))
```

```txt
<tool_call>
{"arguments": {"location": "Paris, France", "unit": "celsius"}, "name": "get_current_temperature"}
</tool_call><|im_end|>
```

The chat model called the `get_current_temperature` tool with the correct parameters from the docstring. It inferred France as the location based on Paris, and that it should use Celsius for the units of temperature. 

Now append the `get_current_temperature` function and these arguments to the chat message as `tool_call`. The `tool_call` dictionary should be provided to the `assistant` role instead of the `system` or `user`.

> [!WARNING]
> The OpenAI API uses a JSON string as its `tool_call` format. This may cause errors or strange model behavior if used in Transformers, which expects a dict.

<hfoptions id="tool-call">
<hfoption id="Llama">

```py
tool_call = {"name": "get_current_temperature", "arguments": {"location": "Paris, France", "unit": "celsius"}}
messages.append({"role": "assistant", "tool_calls": [{"type": "function", "function": tool_call}]})
```

Allow the assistant to read the function outputs and chat with the user.

```py
inputs = tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, return_dict=True, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}
out = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(out[0][len(inputs["input_ids"][0]):]))
```

```txt
The temperature in Paris, France right now is approximately 12°C (53.6°F).<|im_end|>
```

</hfoption>
<hfoption id="Mistral/Mixtral">

For [Mistral](./model_doc/mistral) and [Mixtral](./model_doc/mixtral) models, you need an additional `tool_call_id`. The `tool_call_id` is 9 randomly generated alphanumeric characters assigned to the `id` key in the `tool_call` dictionary.

```py
tool_call_id = "9Ae3bDc2F"
tool_call = {"name": "get_current_temperature", "arguments": {"location": "Paris, France", "unit": "celsius"}}
messages.append({"role": "assistant", "tool_calls": [{"type": "function", "id": tool_call_id, "function": tool_call}]})
```

```py
inputs = tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, return_dict=True, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}
out = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(out[0][len(inputs["input_ids"][0]):]))
```

</hfoption>
</hfoptions>

### Schema

[`~PreTrainedTokenizerBase.apply_chat_template`] converts functions into a [JSON schema](https://json-schema.org/learn/getting-started-step-by-step) which is passed to the model chat template. A LLM never sees the code inside the function. In other words, a LLM doesn't care how the model works technically, it only cares about function **definition** and **arguments**.

The JSON schema is automatically generated behind the scenes as long as your function follows the rules listed earlier above. But you can use [get_json_schema](https://github.com/huggingface/transformers/blob/14561209291255e51c55260306c7d00c159381a5/src/transformers/utils/chat_template_utils.py#L205) to manually convert a schema for more visibility or debugging.

```py
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

You can edit the schema or write one entirely from scratch. This gives you a lot of flexibility to definie precise schemas for more complex functions.

> [!WARNING]
> Try keeping your function signatures simple and the arguments to a minimum. These are easier for a model to understand and use than complex functions for example with nested arguments.

The example below demonstrates writing a schema manually and then passing it to [`~PreTrainedTokenizerBase.apply_chat_template`].

```py
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

## Retrieval-augmented generation (RAG)

Retrieval-augmented generation (RAG) models enhance a models existing knowledge by allowing it to search documents for additional information before returning a query. For RAG models, add a `documents` parameter to [`~PreTrainedTokenizerBase.apply_chat_template`]. This `documents` parameter should be a list of documents, and each document should be a single dict with `title` and `content` keys.

> [!TIP]
> The `documents` parameter for RAG isn't widely supported and many models have chat templates that ignore `documents`. Verify if a model supports `documents` by reading its model card or executing `print(tokenizer.chat_template)` to see if the `documents` key is present. [Command-R](https://hf.co/CohereForAI/c4ai-command-r-08-2024) and [Command-R+](https://hf.co/CohereForAI/c4ai-command-r-plus-08-2024) both support `documents` in their RAG chat templates.

Create a list of documents to pass to the model.

```py
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
```

Set `chat_template="rag"` in [`~PreTrainedTokenizerBase.apply_chat_template`] and generate a response.

```py
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("CohereForAI/c4ai-command-r-v01-4bit")
model = AutoModelForCausalLM.from_pretrained("CohereForAI/c4ai-command-r-v01-4bit", device_map="auto")
device = model.device # Get the device the model is loaded on

# Define conversation input
conversation = [
    {"role": "user", "content": "What has Man always dreamed of?"}
]

input_ids = tokenizer.apply_chat_template(
    conversation=conversation,
    documents=documents,
    chat_template="rag",
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt").to(device)

# Generate a response 
generated_tokens = model.generate(
    input_ids,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.3,
    )

# Decode and print the generated text along with generation prompt
generated_text = tokenizer.decode(generated_tokens[0])
print(generated_text)
```
