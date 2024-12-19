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

# Bamba


## Overview

Bamba-9B is a decoder-only language model based on the [Mamba-2](https://github.com/state-spaces/mamba) architecture and is designed to handle a wide range of text generation tasks. It is trained from scratch using a two-stage training approach. In the first stage, the model is trained on 2 trillion tokens from the Dolma v1.7 dataset. In the second stage, it undergoes additional training on 200 billion tokens, leveraging a carefully curated blend of high-quality data to further refine its performance and enhance output quality.

Checkout all Bamba-9B model checkpoints [here](https://github.com/foundation-model-stack/bamba).

## BambaConfig

| Model            | Params       | # Layers | Hidden Dim. | Attention Heads | GQA | KV Heads | Context Length |  Tied Embeddings |
|-------------------|--------------|----------|-------------|-----------------|-----|----------|----------------|------------------|
| Bamba  | 9B (9.78B)   | 32       | 4096        | 32              | Yes | 8        | 4096           | True |

[[autodoc]] BambaConfig

<!---
## Usage Tips

Tips: 

- The architecture is based on Mamba-2 models.

## BambaModel

[[autodoc]] BambaModel
    - forward
-->

## BambaForCausalLM

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("ibm-fms/Bamba-9B")
tokenizer = AutoTokenizer.from_pretrained("ibm-fms/Bamba-9B")

message = ["Mamba is a snake with following properties  "]
inputs = tokenizer(message, return_tensors='pt', return_token_type_ids=False)
response = model.generate(**inputs, max_new_tokens=64)
print(tokenizer.batch_decode(response, skip_special_tokens=True)[0])
```

[[autodoc]] BambaForCausalLM
    - forward

This HF implementation is contributed by [ani300](https://github.com/ani300) and [fabianlim](https://github.com/fabianlim). 
