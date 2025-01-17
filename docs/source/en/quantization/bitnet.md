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

# BitNet

[BitNet](https://arxiv.org/abs/2402.17764) replaces traditional linear layers in Multi-Head Attention and feed-forward networks with specialized BitLinear layers. The BitLinear layers quantize the weights using ternary precision (with values of -1, 0, and 1) and quantize the activations to 8-bit precision.

<figure style="text-align: center;">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/1.58llm_extreme_quantization/bitlinear.png" alt="Alt Text" />
  <figcaption>The architecture of BitNet with BitLinear layers.</figcaption>
</figure>

BitNet models can't be quantized on the fly. They need to be quantized during pretraining or fine-tuning because it is a Quantization-Aware Training (QAT) technique. During training, the weights are quantized to ternary values with symmetric per tensor quantization.

1. Compute the average of the absolute values of the weight matrix and use as a scale.
2. Divide the weights by the scale, round the values, constrain them between -1 and 1, and rescale them to continue in full precision.
3. Activations are quantized to a specified bit-width (8-bit) using [absmax](https://arxiv.org/pdf/2208.07339) quantization (symmetric per channel quantization). This involves scaling the activations into a range of [−128,127].

Refer to this [PR](https://github.com/huggingface/nanotron/pull/180) to pretrain or fine-tune a 1.58-bit model with [Nanotron](https://github.com/huggingface/nanotron). For fine-tuning, convert a model from the Hugging Face to Nanotron format. Find the conversion steps in this [PR](https://github.com/huggingface/nanotron/pull/174).

Load a BitNet quantized model with [`~PreTrainedModel.from_pretrained`].

```py
from transformers import AutoModelForCausalLM
path = "/path/to/model"
model = AutoModelForCausalLM.from_pretrained(path, device_map="auto")
```

## Kernels

`@torch.compile` is used to unpack the weights and perform the forward pass. It’s very straightforward to implement and delivers significant speed improvements. Additional optimized kernels will be integrated in future versions.

## Resources

Read [Fine-tuning LLMs to 1.58bit: extreme quantization made easy](https://huggingface.co/blog/1_58_llm_extreme_quantization) to learn more about how BitNet models are trained and fine-tuned.
