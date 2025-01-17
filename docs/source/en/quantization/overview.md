<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Quantization

Quantization techniques focus on representing data with less information while also trying to not lose too much accuracy. This often means converting a data type to represent the same information with fewer bits. For example, if your model weights are stored as 32-bit floating points and they're quantized to 16-bit floating points, this halves the model size which makes it easier to store and reduces memory-usage. Lower precision can also speedup inference because it takes less time to perform calculations with fewer bits.

<Tip>

Interested in adding a new quantization method to Transformers? Read the [HfQuantizer](./contribute) guide to learn how!

</Tip>

<Tip>

If you are new to the quantization field, we recommend you to check out these beginner-friendly courses about quantization in collaboration with DeepLearning.AI:

* [Quantization Fundamentals with Hugging Face](https://www.deeplearning.ai/short-courses/quantization-fundamentals-with-hugging-face/)
* [Quantization in Depth](https://www.deeplearning.ai/short-courses/quantization-in-depth/)

</Tip>

## When to use what?

The community has developed many quantization methods for various use cases. With Transformers, you can run any of these integrated methods depending on your use case because each method has their own pros and cons.

For example, some quantization methods require calibrating the model with a dataset for more accurate and "extreme" compression (up to 1-2 bits quantization), while other methods work out of the box with on-the-fly quantization.

Another parameter to consider is compatibility with your target device. Do you want to quantize on a CPU, GPU, or Apple silicon?

In short, supporting a wide range of quantization methods allows you to pick the best quantization method for your specific use case.

Use the table below to help you decide which quantization method to use.

| Quantization Method                           | On the fly quantization | CPU             | CUDA GPU | ROCm GPU  | Metal (Apple Silicon)              | Intel GPU       | Torch compile() | Bits          | PEFT Fine Tuning | Serializable with 🤗Transformers | 🤗Transformers Support  | Link to library                             |
|-----------------------------------------------|----------------------|-----------------|----------|-----------|------------------------------------|-----------------|-----------------|---------------|------------------|-----------------------------|-------------------------|---------------------------------------------|
| [AQLM](./aqlm.md)                             | 🔴                   | 🟢              |     🟢     | 🔴        | 🔴                                 | 🔴              | 🟢              | 1/2         | 🟢               | 🟢                          | 🟢                      | https://github.com/Vahe1994/AQLM            |
| [AWQ](./awq.md)                               | 🔴                   | 🟢              | 🟢        | 🟢        | 🔴                                 | 🟢              | ?               | 4             | 🟢               | 🟢                          | 🟢                      | https://github.com/casper-hansen/AutoAWQ    |
| [bitsandbytes](./bitsandbytes.md)             | 🟢                   | 🟡 <sub>1</sub> |     🟢     | 🟡 <sub>1</sub> | 🔴 <sub>2</sub>                    | 🟡 <sub>1</sub> | 🔴 <sub>1</sub> | 4/8         | 🟢               | 🟢                          | 🟢                      | https://github.com/bitsandbytes-foundation/bitsandbytes |
| [compressed-tensors](./compressed_tensors.md) | 🔴                   | 🟢              |     🟢     | 🟢        | 🔴                                 | 🔴              | 🔴              | 1/8         | 🟢               | 🟢                          | 🟢                      | https://github.com/neuralmagic/compressed-tensors |
| [EETQ](./eetq.md)                             | 🟢                   | 🔴              | 🟢        | 🔴        | 🔴                                 | 🔴              | ?               | 8             | 🟢               | 🟢                          | 🟢                      | https://github.com/NetEase-FuXi/EETQ        |
| [GGUF / GGML (llama.cpp)](../gguf.md)         | 🟢                   | 🟢              | 🟢        | 🔴        | 🟢                                 | 🔴              | 🔴              | 1/8         | 🔴               | [See Notes](../gguf.md)     | [See Notes](../gguf.md) | https://github.com/ggerganov/llama.cpp      |
| [GPTQModel](./gptq.md)                        | 🔴                   | 🟢 <sub>3</sub> | 🟢        | 🟢        | 🟢                                 | 🟢 <sub>4</sub> | 🔴              | 2/3/4/8 | 🟢               | 🟢                          | 🟢                      | https://github.com/ModelCloud/GPTQModel        |
| [AutoGPTQ](./gptq.md)                         | 🔴                   | 🔴              | 🟢        | 🟢        | 🔴                                 | 🔴              | 🔴              | 2/3/4/8 | 🟢               | 🟢                          | 🟢                      | https://github.com/AutoGPTQ/AutoGPTQ        |
| [HIGGS](./higgs.md)                           | 🟢                   | 🔴              | 🟢        | 🔴        | 🔴                                 | 🔴              | 🟢              | 2/4         | 🔴               | 🟢                          | 🟢                      | https://github.com/HanGuo97/flute           |       
| [HQQ](./hqq.md)                               | 🟢                   | 🟢              | 🟢        | 🔴        | 🔴                                 | 🔴              | 🟢              | 1/8         | 🟢               | 🔴                          | 🟢                      | https://github.com/mobiusml/hqq/            |
| [optimum-quanto](./quanto.md)                 | 🟢                   | 🟢              | 🟢        | 🔴        | 🟢                                 | 🔴              | 🟢              | 2/4/8     | 🔴               | 🔴                          | 🟢                      | https://github.com/huggingface/optimum-quanto       |
| [FBGEMM_FP8](./fbgemm_fp8.md)                 | 🟢                   | 🔴              | 🟢        | 🔴        | 🔴                                 | 🔴              | 🔴              | 8             | 🔴               | 🟢                          | 🟢                      | https://github.com/pytorch/FBGEMM       |
| [torchao](./torchao.md)                       | 🟢                   |                 | 🟢        | 🔴        | 🟡 <sub>5</sub> | 🔴              |                 | 4/8         |                  | 🟢🔴                        | 🟢                      | https://github.com/pytorch/ao       |
| [VPTQ](./vptq.md)                             | 🔴                   | 🔴              |     🟢     | 🟡        | 🔴                                 | 🔴              | 🟢              | 1/8         | 🔴               | 🟢                          | 🟢                      | https://github.com/microsoft/VPTQ            |

<Tip>
  
**1:** bitsandbytes is being refactored to support multiple backends beyond CUDA. Currently, ROCm (AMD GPU) and Intel CPU implementations are mature, with Intel XPU in progress and Apple Silicon support expected by Q4/Q1. For installation instructions and the latest backend updates, visit [this link](https://huggingface.co/docs/bitsandbytes/main/en/installation#multi-backend). Check out [these docs](https://huggingface.co/docs/bitsandbytes/main/en/non_cuda_backends) for more details and feedback links.

</Tip>

<Tip>

**2:** bitsandbytes is seeking contributors to help develop and lead the Apple Silicon backend. Interested? Contact them directly via their repo. Stipends may be available through sponsorships.

</Tip>

<Tip>

**3:** GPTQModel[CPU] supports 4-bit via IPEX on Intel/AMD and full bit range via Torch on Intel/AMD/Apple Silicon.

</Tip>

<Tip>

**4:** GPTQModel[Intel GPU] via IPEX only supports 4-bit for Intel Datacenter Max/Arc GPUs.

</Tip>

<Tip>

**5:** torchao only supports int4 weight on Metal (Apple Silicon).

</Tip>

