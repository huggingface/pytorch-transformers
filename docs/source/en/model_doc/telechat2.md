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

# TeleChat2

## Overview

The TeleChat2 model was proposed in [TELECHAT TECHNICAL REPORT](https://arxiv.org/pdf/2401.03804) by TeleAI.

The abstract from the paper is the following:
*TeleChat is a series of large language models, offering decoder-based language models in various sizes (3B, 7B, and 12B). For each size, we provide both the base pretrained model and the fine-tuned chat model aligned with human preferences. TeleChat leverages a Transformer architecture with features such as SwiGLU activation, advanced attention mechanisms (QKV bias, group query attention), and support for sliding window attention. The models are optimized for bilingual proficiency (English and Chinese) and include an enhanced tokenizer adaptable to diverse natural languages and coding formats.*

The original code for telechat2 can be found [here](https://huggingface.co/Tele-AI/TeleChat2-7B).
## Tips
In the following, we demonstrate how to use `TeleChat2-7B` for inference. The example below shows how to use `apply_chat_template` with the ChatML format for dialog.

```python
>>> from transformers import AutoModelForCausalLM, AutoTokenizer
>>> device = "cuda" # the device to load the model onto

>>> model = AutoModelForCausalLM.from_pretrained("Tele-AI/TeleChat2-7B", device_map="auto")
>>> tokenizer = AutoTokenizer.from_pretrained("Tele-AI/TeleChat2-7B")

>>> prompt = "Give me a short introduction to large language model."

>>> messages = [{"role": "user", "content": prompt}]

>>> text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

>>> model_inputs = tokenizer([text], return_tensors="pt").to(device)

>>> generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512, do_sample=True)

>>> generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]

>>> response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

## TeleChat2Config

[[autodoc]] TeleChat2Config


## TeleChat2Model

[[autodoc]] TeleChat2Model
    - forward

## TeleChat2ForCausalLM

[[autodoc]] TeleChat2ForCausalLM
    - forward

## TeleChat2ForSequenceClassification

[[autodoc]] TeleChat2ForSequenceClassification
    - forward

## TeleChat2ForTokenClassification

[[autodoc]] TeleChat2ForTokenClassification
    - forward

## TeleChat2ForQuestionAnswering

[[autodoc]] TeleChat2ForQuestionAnswering
    - forward
