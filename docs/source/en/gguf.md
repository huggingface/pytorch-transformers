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

# GGUF and interaction with Transformers

The GGUF file format is used to store models for inference with [GGML](https://github.com/ggerganov/ggml) and other 
libraries that depend on it, like the very popular [llama.cpp](https://github.com/ggerganov/llama.cpp) or 
[whisper.cpp](https://github.com/ggerganov/whisper.cpp).

It is a file format [supported by the Hugging Face Hub](https://huggingface.co/docs/hub/en/gguf) with features 
allowing for quick inspection of tensors and metadata within the file.

This file format is designed as a "single-file-format" where a single file usually contains both the configuration
attributes, the tokenizer vocabulary and other attributes, as well as all tensors to be loaded in the model. These
files come in different formats according to the quantization type of the file. We briefly go over some of them
[here](https://huggingface.co/docs/hub/en/gguf#quantization-types).

## Support within Transformers

We have added the ability to load `gguf` files within `transformers` in order to offer further training/fine-tuning
capabilities to gguf models, before converting back those models to `gguf` to use within the `ggml` ecosystem. When
loading a model, we first dequantize it to fp32, before loading the weights to be used in PyTorch.

> [!NOTE]
> The support is still very exploratory and we welcome contributions in order to solidify it across quantization types
> and model architectures.

For now, here are the supported model architectures and quantization types:

### Supported quantization types

The initial supported quantization types are decided according to the popular quantized files that have been shared
on the Hub.

- F32
- F16
- BF16
- Q4_0
- Q4_1
- Q5_0
- Q5_1
- Q8_0
- Q2_K
- Q3_K
- Q4_K
- Q5_K
- Q6_K
- IQ1_S
- IQ1_M
- IQ2_XXS
- IQ2_XS
- IQ2_S
- IQ3_XXS
- IQ3_S
- IQ4_XS
- IQ4_NL

> [!NOTE]
> To support gguf dequantization, `gguf>=0.10.0` installation is required.

### Supported model architectures

For now the supported model architectures are the architectures that have been very popular on the Hub, namely:

- LLaMa
- Mistral
- Qwen2
- Qwen2Moe
- Phi3
- Bloom
- Falcon
- StableLM
- GPT2
- Starcoder2
- T5
- Mamba
- Nemotron
- Gemma2

## Example usage

In order to load `gguf` files in `transformers`, you should specify the `gguf_file` argument to the `from_pretrained`
methods of both tokenizers and models. Here is how one would load a tokenizer and a model, which can be loaded
from the exact same file:

```py
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
filename = "tinyllama-1.1b-chat-v1.0.Q6_K.gguf"

tokenizer = AutoTokenizer.from_pretrained(model_id, gguf_file=filename)
model = AutoModelForCausalLM.from_pretrained(model_id, gguf_file=filename)
```

Now you have access to the full, unquantized version of the model in the PyTorch ecosystem, where you can combine it
with a plethora of other tools.

In order to convert back to a `gguf` file, we recommend using the 
[`convert-hf-to-gguf.py` file](https://github.com/ggerganov/llama.cpp/blob/master/convert_hf_to_gguf.py) from llama.cpp.

Here's how you would complete the script above to save the model and export it back to `gguf`:

```py
tokenizer.save_pretrained('directory')
model.save_pretrained('directory')

!python ${path_to_llama_cpp}/convert-hf-to-gguf.py ${directory}
```
