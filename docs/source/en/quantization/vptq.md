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

# VPTQ

[Vector Post-Training Quantization (VPTQ)](https://github.com/microsoft/VPTQ) is a Post-Training Quantization (PTQ) method that leverages vector quantization to quantize LLMs at an extremely low bit-width (<2-bit). VPTQ can compress a 70B, even a 405B model, to 1-2 bits without retraining and still maintain a high-degree of accuracy. It is a lightweight quantization algorithm that takes ~17 hours to quantize a 405B model. VPTQ features agile quantization inference with low decoding overhead and high throughput and Time To First Token (TTFT).

Run the command below to install VPTQ which provides efficient kernels for inference on NVIDIA and AMD GPUs.

```bash
pip install vptq
```

The [VPTQ-community](https://huggingface.co/VPTQ-community) provides a collection of VPTQ-quantized models. The model name contains information about its bitwidth (excluding cookbook, parameter, and padding overhead). Consider the [Meta-Llama-3.1-70B-Instruct-v8-k65536-256-woft] model as an example.

- The model name is Meta-Llama-3.1-70B-Instruct.
- The number of centroids is given by 65536 (2^16).
- The number of residual centroids is given by 256 (2^8).

The equivalent bit-width calculation is given by the following.

- index: log2(65536) = 16 / 8 = 2-bits
- residual index: log2(256) = 8 / 8 = 1-bit
- total bit-width: 2 + 1 = 3-bits

From here, estimate the model size by multiplying 70B * 3-bits / 8-bits/byte for a total of 26.25GB.

Load a VPTQ quantized model with [`~PreTrainedModel.from_pretrained`].

```py
from transformers import AutoTokenizer, AutoModelForCausalLM

quantized_model = AutoModelForCausalLM.from_pretrained(
    "VPTQ-community/Meta-Llama-3.1-70B-Instruct-v16-k65536-65536-woft",
    torch_dtype="auto", 
    device_map="auto"
)
```

To quantize your own model, refer to the [VPTQ Quantization Algorithm Tutorial](https://github.com/microsoft/VPTQ/blob/algorithm/algorithm.md) tutorial.

## Benchmarks

VPTQ achieves better accuracy and higher throughput with lower quantization overhead across models of different sizes. The following experimental results are for reference only; VPTQ can achieve better outcomes under reasonable parameters, especially in terms of model accuracy and inference speed.

| Model       | bitwidth | W2↓  | C4↓  | AvgQA↑ | tok/s↑ | mem(GB) | cost/h↓ |
| ----------- | -------- | ---- | ---- | ------ | ------ | ------- | ------- |
| LLaMA-2 7B  | 2.02     | 6.13 | 8.07 | 58.2   | 39.9   | 2.28    | 2       |
|             | 2.26     | 5.95 | 7.87 | 59.4   | 35.7   | 2.48    | 3.1     |
| LLaMA-2 13B | 2.02     | 5.32 | 7.15 | 62.4   | 26.9   | 4.03    | 3.2     |
|             | 2.18     | 5.28 | 7.04 | 63.1   | 18.5   | 4.31    | 3.6     |
| LLaMA-2 70B | 2.07     | 3.93 | 5.72 | 68.6   | 9.7    | 19.54   | 19      |
|             | 2.11     | 3.92 | 5.71 | 68.7   | 9.7    | 20.01   | 19      |

## Resources

See an example demo of VPTQ on the VPTQ Online Demo [Space](https://huggingface.co/spaces/microsoft/VPTQ) or try running the VPTQ inference [notebook](https://colab.research.google.com/github/microsoft/VPTQ/blob/main/notebooks/vptq_example.ipynb).

For more information, read the VPTQ [paper](https://arxiv.org/pdf/2409.17066).
