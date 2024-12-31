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

# HIGGS

[HIGGS](https://arxiv.org/abs/2411.17525) is a zero-shot quantization algorithm that combines Hadamard preprocessing with MSE-Optimal quantization grids to achieve lower quantization error and state-of-the-art performance.

Runtime support for HIGGS is implemented through the [FLUTE](https://github.com/HanGuo97/flute) library. Only the 70B and 405B variants of Llama 3 and Llama 3.0, and the 8B and 27B variants of Gemma 2 are currently supported. HIGGS also doesn't support quantized training and backward passes in general at the moment.

Run the command below to install FLUTE.

<hfoptions id="install">
<hfoption id="CUDA 12.1">

```bash
pip install flute-kernel
```

</hfoption>
<hfoption id="CUDA 11.8">

```bash
pip install flute-kernel -i https://flute-ai.github.io/whl/cu118
```

</hfoption>
</hfoptions>

Create a [`HiggsConfig`] with the number of bits to quantize a model to.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, HiggsConfig

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-9b-it",
    quantization_config=HiggsConfig(bits=4),
    device_map="auto",
)
```

> [!TIP]
> Find models pre-quantized with HIGGS in the official ISTA-DASLab [collection](https://huggingface.co/collections/ISTA-DASLab/higgs-675308e432fd56b7f6dab94e).

## torch.compile

HIGGS is fully compatible with [torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html).

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, HiggsConfig

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-9b-it",
    quantization_config=HiggsConfig(bits=4),
    device_map="auto",
)

model = torch.compile(model)
```

Refer to the table below for a benchmark of forward passes/sec for Llama-3.1-8B-Instruct on a RTX4090.

| Batch Size | BF16 (with `torch.compile`) | HIGGS 4bit (without `torch.compile`) | HIGGS 4bit (with `torch.compile`) |
|------------|-----------------------------|----------------------------------|-----------------------------------|
| 1          | 59                          | 41                               | 124                               |
| 4          | 57                          | 42                               | 123                               |
| 16         | 56                          | 41                               | 120                               |
