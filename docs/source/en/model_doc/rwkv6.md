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

# RWKV6

## Overview

The RWKV6 model was proposed in [this repo](https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v5)

It suggests a tweak in the traditional Transformer attention to make it linear. This way, the model can be used as recurrent network: passing inputs for timestamp 0 and timestamp 1 together is the same as passing inputs at timestamp 0, then inputs at timestamp 1 along with the state of timestamp 0 (see example below).

This can be more efficient than a regular Transformer and can deal with sentence of any length (even if the model uses a fixed context length for training).

The original code can be found [here](https://github.com/BlinkDL/RWKV-LM).

## Usage example

```py
import torch
from transformers import AutoTokenizer, Rwkv6Config, Rwkv6Model, Rwkv6Tokenizer

model = Rwkv6Model.from_pretrained("RWKV/v6-Finch-1B6-HF")
tokenizer = Rwkv6Tokenizer.from_pretrained("RWKV/v6-Finch-1B6-HF")

inputs = tokenizer("This is an example.", return_tensors="pt")
# Feed everything to the model
outputs = model(inputs["input_ids"])
output_whole = outputs.last_hidden_state

outputs = model(inputs["input_ids"][:, :2])
output_one = outputs.last_hidden_state

# Using the state computed on the first inputs, we will get the same output
outputs = model(inputs["input_ids"][:, 2:], state=outputs.state)
output_two = outputs.last_hidden_state

torch.allclose(torch.cat([output_one, output_two], dim=1), output_whole, atol=1e-5)
```

If you want to make sure the model stops generating when `'\n\n'` is detected, we recommend using the following stopping criteria:

```python 
from transformers import StoppingCriteria

class RwkvStoppingCriteria(StoppingCriteria):
    def __init__(self, eos_sequence = [187,187], eos_token_id = 537):
        self.eos_sequence = eos_sequence
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        last_2_ids = input_ids[:,-2:].tolist()
        return self.eos_sequence in last_2_ids


output = model.generate(inputs["input_ids"], max_new_tokens=64, stopping_criteria = [RwkvStoppingCriteria()])
```

## Rwkv6Config

[[autodoc]] Rwkv6Config

## Rwkv6Model

[[autodoc]] Rwkv6Model
    - forward

## RwkvLMHeadModel

[[autodoc]] Rwkv6ForCausalLM
    - forward

## Rwkv6Tokenizer

[[autodoc]] Rwkv6Tokenizer
    - vocab_size
    - get_vocab
    - _tokenize
    - _convert_token_to_id
    - _convert_id_to_token
    - convert_tokens_to_string
    - save_vocabulary
    - build_inputs_with_special_tokens
    - get_special_tokens_mask 