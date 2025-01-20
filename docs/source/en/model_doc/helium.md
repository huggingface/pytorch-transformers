<!--Copyright 2024 Kyutai and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Helium


## Overview

Helium was proposed in [Announcing Helium-1 Preview](https://kyutai.org/2025/01/13/helium.html) by the Kyutai Team.


Helium-1 preview is a lightweight language model with 2B parameters, targeting edge and mobile devices.
It supports the following languages: English, French, German, Italian, Portuguese, Spanish.

- **Developed by:** Kyutai
- **Model type:** Large Language Model
- **Language(s) (NLP):** English, French, German, Italian, Portuguese, Spanish
- **License:** CC-BY 4.0




## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

#### Testing Data

<!-- This should link to a Dataset Card if possible. -->

The model was evaluated on MMLU, TriviaQA, NaturalQuestions, ARC Easy & Challenge, Open Book QA, Common Sense QA, 
Physical Interaction QA, Social Interaction QA, HellaSwag, WinoGrande, Multilingual Knowledge QA, FLORES 200.

#### Metrics

<!-- These are the evaluation metrics being used, ideally with a description of why. -->

We report accuracy on MMLU, ARC, OBQA, CSQA, PIQA, SIQA, HellaSwag, WinoGrande.
We report exact match on TriviaQA, NQ and MKQA.
We report BLEU on FLORES.

### English Results

| Benchmark | Helium-1 Preview | HF SmolLM2 (1.7B) | Gemma-2 (2.6B) | Llama-3.2 (3B) | Qwen2.5 (1.5B) |
|--------------|--------|--------|--------|--------|--------|
| | | | | | |
| MMLU | 51.2 | 50.4 | 53.1 | 56.6 | 61.0 |
| NQ   | 17.3 | 15.1 | 17.7 | 22.0 | 13.1 |
| TQA  | 47.9 | 45.4 | 49.9 | 53.6 | 35.9 |
| ARC E | 80.9 | 81.8 | 81.1 | 84.6 | 89.7 |
| ARC C | 62.7 | 64.7 | 66.0 | 69.0 | 77.2 |
| OBQA | 63.8 | 61.4 | 64.6 | 68.4 | 73.8 |
| CSQA | 65.6 | 59.0 | 64.4 | 65.4 | 72.4 |
| PIQA | 77.4 | 77.7 | 79.8 | 78.9 | 76.0 |
| SIQA | 64.4 | 57.5 | 61.9 | 63.8 | 68.7 |
| HS | 69.7 | 73.2 | 74.7 | 76.9 | 67.5 |
| WG | 66.5 | 65.6 | 71.2 | 72.0 | 64.8 |
| | | | | | |
| Average | 60.7 | 59.3 | 62.2 | 64.7 | 63.6 |

#### Multilingual Results

| Language | Benchmark | Helium-1 Preview | HF SmolLM2 (1.7B) | Gemma-2 (2.6B) | Llama-3.2 (3B) | Qwen2.5 (1.5B) |
|-----|--------------|--------|--------|--------|--------|--------|
| | | | | | | |
|German| MMLU | 45.6 | 35.3 | 45.0 | 47.5 | 49.5 |
|| ARC C | 56.7 | 38.4 | 54.7 | 58.3 | 60.2 |
|| HS | 53.5 | 33.9 | 53.4 | 53.7 | 42.8 |
|| MKQA | 16.1 | 7.1 | 18.9 | 20.2 | 10.4 |
| | | | | | | |
|Spanish| MMLU | 46.5 | 38.9 | 46.2 | 49.6 | 52.8 |
|| ARC C | 58.3 | 43.2 | 58.8 | 60.0 | 68.1 |
|| HS | 58.6 | 40.8 | 60.5 | 61.1 | 51.4 |
|| MKQA | 16.0 | 7.9 | 18.5 | 20.6 | 10.6 |


## Technical Specifications

### Model Architecture and Objective

| Hyperparameter | Value |
|--------------|--------|
| Layers | 24 |
| Heads  | 20 |
| Model dimension | 2560 |
| MLP dimension | 7040 |
| Context size | 4096 |
| Theta RoPE | 100,000 |

Tips:

- This model was contributed by [Laurent Mazare](https://huggingface.co/lmz)

  
## Usage tips

`Helium` can be found on the [Huggingface Hub](https://huggingface.co/collections/kyutai/helium-1-preview)

In the following, we demonstrate how to use `helium-1-preview` for the inference. 

```python
>>> from transformers import AutoModelForCausalLM, AutoTokenizer
>>> device = "cuda" # the device to load the model onto

>>> model = AutoModelForCausalLM.from_pretrained("helium-1-preview", device_map="auto")
>>> tokenizer = AutoTokenizer.from_pretrained("helium-1-preview")

>>> prompt = "Give me a short introduction to large language model."

>>> messages = [{"role": "user", "content": prompt}]

>>> text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

>>> model_inputs = tokenizer([text], return_tensors="pt").to(device)

>>> generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512, do_sample=True)

>>> generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]

>>> response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

## HeliumConfig

[[autodoc]] HeliumConfig

## HeliumModel

[[autodoc]] HeliumModel
    - forward

## HeliumForCausalLM

[[autodoc]] HeliumForCausalLM
    - forward

## HeliumForSequenceClassification

[[autodoc]] HeliumForSequenceClassification
    - forward

## HeliumForTokenClassification

[[autodoc]] HeliumForTokenClassification
    - forward
