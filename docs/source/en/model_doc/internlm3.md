<!--Copyright 2024 The InternLM Team and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# InternLM3

## Introduction

InternLM3 has open-sourced an 8-billion parameter instruction model, InternLM3-8B-Instruct, designed for general-purpose usage and advanced reasoning. This model has the following characteristics:

- Enhanced performance at reduced cost: State-of-the-art performance on reasoning and knowledge-intensive tasks surpass models like Llama3.1-8B and Qwen2.5-7B. Remarkably, InternLM3 is trained on only 4 trillion high-quality tokens, saving more than 75% of the training cost compared to other LLMs of similar scale.

- Deep thinking capability: InternLM3 supports both the deep thinking mode for solving complicated reasoning tasks via the long chain-of-thought and the normal response mode for fluent user interactions.

## Usage tips

Model weights can be requested [here](https://huggingface.co/internlm/internlm3-8b-instruct).

In the following, we demonstrate how to use `InternLM3-8B-Instruct` for the inference.

```python
>>> from transformers import AutoModelForCausalLM, AutoTokenizer
>>> device = "cuda" # the device to load the model onto

>>> tokenizer = transformers.InternLM3Tokenizer.from_pretrained("internlm/internlm3-8b-instruct")
>>> model = transformers.InternLM3ForCausalLM.from_pretrained("internlm/internlm3-8b-instruct", device_map="auto")
>>> model = model.eval()

>>> system_prompt = """You are an AI assistant whose name is InternLM (书生·浦语).
    - InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
    - InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文."""
>>> messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "Please tell me five scenic spots in Shanghai"},
]
>>> tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
>>> generated_ids = model.generate(tokenized_chat, max_new_tokens=1024, temperature=1, repetition_penalty=1.005, top_k=40, top_p=0.8)

>>> generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(tokenized_chat, generated_ids)
]
>>> prompt = tokenizer.batch_decode(tokenized_chat)[0]
>>> print(prompt)
>>> response = tokenizer.batch_decode(generated_ids)[0]
>>> print(response)
```

## InternLM3Config

[[autodoc]] InternLM3Config

## InternLM3Tokenizer

[[autodoc]] InternLM3Tokenizer
    - save_vocabulary

## InternLM3Model

[[autodoc]] InternLM3Model
    - forward

## InternLM3ForCausalLM

[[autodoc]] InternLM3ForCausalLM
    - forward