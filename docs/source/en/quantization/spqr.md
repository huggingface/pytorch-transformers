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

# SpQR

[SpQR](https://github.com/Vahe1994/SpQR) quantization algorithm involves a 16x16 tiled bi-level group 3-bit quantization structure, with sparse outliers as detailed in [SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression](https://arxiv.org/abs/2306.03078).

To SpQR-quantize a model, refer to the [Vahe1994/SpQR](https://github.com/Vahe1994/SpQR) repository.

Load a pre-SpQR-quantized model in [`~PreTrainedModel.from_pretrained`].

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

quantized_model = AutoModelForCausalLM.from_pretrained(
    "elvircrn/Llama-2-7b-SPQR-3Bit-16x16-red_pajama-hf",
    torch_dtype=torch.half,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("elvircrn/Llama-2-7b-SPQR-3Bit-16x16-red_pajama-hf")
```
