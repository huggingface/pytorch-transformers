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

HIGGS is a 0-shot quantization algorithm that combines Hadamard preprocessing with MSE-Optimal quantization grids to achieve lower quantization error and SOTA performance. You can find more information in the paper [arxiv.org/abs/2411.17525](https://arxiv.org/abs/2411.17525).

Runtime support for HIGGS is implemented through [FLUTE](https://arxiv.org/abs/2407.10960), and its [library](https://github.com/HanGuo97/flute).

## Quantization Example

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, HiggsConfig

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-9b-it",
    quantization_config=HiggsConfig(bits=4),
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")

tokenizer.decode(model.generate(
    **tokenizer("Hi,", return_tensors="pt").to(model.device),
    temperature=0.5,
    top_p=0.80,
)[0])
```

## Pre-quantized models

Some pre-quantized models can be found in the [official collection](https://huggingface.co/collections/ISTA-DASLab/higgs-675308e432fd56b7f6dab94e) on Hugging Face Hub.

## Current Limitations

**Architectures**

Currently, FLUTE, and HIGGS by extension, **only support Llama 3 and 3.0 of 8B, 70B and 405B parameters, as well as Gemma-2 9B and 27B**. We're working on allowing to run more diverse models as well as allow arbitrary models by modifying the FLUTE compilation procedure.

**torch.compile**

HIGGS is fully compatible with `torch.compile`. Compiling `model.forward`, as described [here](../perf_torch_compile.md), here're the speedups it provides on RTX 4090 for `Llama-3.1-8B-Instruct` (forward passes/sec):

| Batch Size | BF16 (With `torch.compile`) | HIGGS 4bit (No `torch.compile`) | HIGGS 4bit (With `torch.compile`) |
|------------|-----------------------------|----------------------------------|-----------------------------------|
| 1          | 59                          | 41                               | 124                               |
| 4          | 57                          | 42                               | 123                               |
| 16         | 56                          | 41                               | 120                               |


**Quantized training**

Currently, HIGGS doesn't support quantized training (and backward passes in general). We're working on adding support for it.