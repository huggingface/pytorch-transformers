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

# AQLM

Additive Quantization of Language Models ([AQLM](https://arxiv.org/abs/2401.06118)) quantizes multiple weights together and takes advantage of interdependencies between them. AQLM represents groups of 8-16 weights as a sum of multiple vector codes.

AQLM also supports fine-tuning with [LoRA](https://huggingface.co/docs/peft/package_reference/lora) with the [PEFT](https://huggingface.co/docs/peft) library, and is fully compatible with [torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) for even faster inference and training.

Run the command below to install the AQLM library with kernel support for both GPU and CPU inference and training. AQLM only works with Python 3.10+.

```bash
pip install aqlm[gpu,cpu]
```

Load an AQLM-quantized model with [`~PreTrainedModel.from_pretrained`].

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

quantized_model = AutoModelForCausalLM.from_pretrained(
    "ISTA-DASLab/Mixtral-8x7b-AQLM-2Bit-1x16-hf",
    torch_dtype="auto", 
    device_map="auto"
)
```

## Configurations

AQLM quantization setups vary mainly in the number of codebooks used, as well as codebook sizes in bits. The most popular setups and supported inference kernels are shown below.

| Kernel | Number of codebooks | Codebook size, bits | Notation | Accuracy | Speedup     | Fast GPU inference | Fast CPU inference |
|---|---------------------|---------------------|----------|-------------|-------------|--------------------|--------------------|
| Triton | K                   | N                  | KxN     | -        | Up to ~0.7x | ✅                  | ❌                  |
| CUDA | 1                   | 16                  | 1x16     | Best        | Up to ~1.3x | ✅                  | ❌                  |
| CUDA | 2                   | 8                   | 2x8      | OK          | Up to ~3.0x | ✅                  | ❌                  |
| Numba | K                   | 8                   | Kx8      | Good        | Up to ~4.0x | ❌                  | ✅                  |

## Resources

Run the AQLM demo [notebook](https://colab.research.google.com/drive/1-xZmBRXT5Fm3Ghn4Mwa2KRypORXb855X?usp=sharing) for more examples of how to quantize a model, push a quantized model to the Hub, and more.

For more example demo notebooks, visit the AQLM [repository](https://github.com/Vahe1994/AQLM).
