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
# Zamba2

Zamba2 is a large language model (LLM) trained by Zyphra, and made available under an Apache 2.0 license. Please see the [Zyphra Hugging Face](https://huggingface.co/collections/zyphra/) repository for model weights.

This model was contributed by [pglo](https://huggingface.co/pglo).


## Model details

Zamba2-1.2B, Zamba2-2.7B and Zamba2-7B are hybrid models combining state-space models (Specifically [Mamba](https://github.com/state-spaces/mamba)) and transformer, and were trained using next-token prediction. Zamba2 uses shared transformer layers after every 6 mamba blocks. It uses the [Mistral v0.1 tokenizer](https://huggingface.co/mistralai/Mistral-7B-v0.1). We came to this architecture after a series of ablations at small scales. Zamba2-1.2B, Zamba2-2.7B and Zamba2-7B were pre-trained on 2T and 3T tokens, respectively.

<img src=https://github.com/user-attachments/assets/c2cff209-b901-483c-87aa-774b82a0769f width=30% height=40% />

## Quick start


### Presequities

Zamba2 requires you use `transformers` version 4.46.0 or higher:
```bash
pip install transformers>=4.46.0
```

## Inference

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("Zyphra/Zamba2-7B")
model = AutoModelForCausalLM.from_pretrained("Zyphra/Zamba2-7B", device_map="cuda", torch_dtype=torch.bfloat16)

input_text = "What factors contributed to the fall of the Roman Empire?"
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```


## Model card

The model cards can be found at:
* [Zamba2-1.2B](https://huggingface.co/Zyphra/Zamba2-1.2B)
* [Zamba2-2.7B](https://huggingface.co/Zyphra/Zamba2-2.7B)
* [Zamba2-7B](https://huggingface.co/Zyphra/Zamba2-7B)


## Issues
For issues with model output, or community discussion, please use the Hugging Face community [forum](https://huggingface.co/Zyphra/Zamba2-7B/discussions)


## License

The model weights are open-sourced via an Apache 2.0 license.


## Zamba2Config

[[autodoc]] Zamba2Config


## Zamba2Model

[[autodoc]] Zamba2Model
    - forward


## Zamba2ForCausalLM

[[autodoc]] Zamba2ForCausalLM
    - forward


## Zamba2ForSequenceClassification

[[autodoc]] transformers.Zamba2ForSequenceClassification
    - forward
