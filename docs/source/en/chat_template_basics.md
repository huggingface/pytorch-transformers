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

# Getting Started with Chat Templates for Text LLMs

An increasingly common use case for LLMs is **chat**. In a chat context, rather than continuing a single string
of text (as is the case with a standard language model), the model instead continues a conversation that consists
of one or more **messages**, each of which includes a **role**, like "user" or "assistant", as well as message text.

Much like tokenization, different models expect very different input formats for chat. This is the reason we added
**chat templates** as a feature. Chat templates are part of the tokenizer for text-only LLMs or processor for multimodal LLMs. They specify how to convert conversations, represented as lists of messages, into a single tokenizable string in the format that the model expects. 

We'll explore the basic usage of chat templates with text-only LLMs in this page. For detailed guidance on multimodal models, we have a dedicated [documentation oage for multimodal models](./chat_template_multimodal), which covers how to work with image, video and audio inputs in your templates.

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
and `content` keys, and then pass it to the [`~PreTrainedTokenizer.apply_chat_template`] or [`~ProcessorMixin.apply_chat_template`] method
depending on what type of model you are using. Once you do that,
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

