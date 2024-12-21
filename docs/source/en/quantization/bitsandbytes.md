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

# bitsandbytes

[bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) features the LLM.int8 and QLoRA quantization to enable accessible large language model inference and training.

LLM.int8 matrix multiplication, or 8-bit quantization, is based on vector-wise quantization to quantize most of the weights to 8-bits and treating outliers with 16-bit matrix multiplication to reduce their degradative effect on model accuracy.

QLoRA, or 4-bit quantization, compresses a model even further to 4-bits and inserts a small set of trainable low-rank adaptation (LoRA) weights to allowing training.

Run the command below to install bitsandbytes.

```bash
pip install --upgrade transformers accelerate bitsandbytes
```

Quantize a model by passing a [`BitsAndBytesConfig`] to [`~PreTrainedModel.from_pretrained`]. This works for any model in any modality, as long as it supports [Accelerate]https://huggingface.co/docs/accelerate/index() and contains [torch.nn.Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) layers.

<hfoptions id="bnb">
<hfoption id="8-bit">

Quantizing a model in 8-bit halves the memory-usage, and for large models, set `device_map="auto"` to efficiently distribute the weights across all available GPUs.

```py
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

model_8bit = AutoModelForCausalLM.from_pretrained(
    "bigscience/bloom-1b7", 
    quantization_config=quantization_config
)
```

By default, all other modules such as [torch.nn.LayerNorm](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html) are converted to `torch.float16`. You can change the data type of these modules with the `torch_dtype` parameter.. Setting `torch_dtype="auto"` loads the model in the data type defined in a model's `config.json` file.

```py
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

model_8bit = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-350m", 
    quantization_config=quantization_config, 
    torch_dtype="auto"
)
model_8bit.model.decoder.layers[-1].final_layer_norm.weight.dtype
```

Once a model is quantized to 8-bit, you can't push the quantized weights to the Hub unless you're using the latest version of Transformers and bitsandbytes. If you have the latest versions, then you can push the 8-bit model to the Hub with [`~PreTrainedModel.push_to_hub`]. The quantization config.json file is pushed first, followed by the quantized model weights.

```py
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

model = AutoModelForCausalLM.from_pretrained(
    "bigscience/bloom-560m", 
    quantization_config=quantization_config
)
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")

model.push_to_hub("bloom-560m-8bit")
```

</hfoption>
<hfoption id="4-bit">

Quantizing a model in 4-bit reduces your memory-usage by 4x, and for large models, set `device_map="auto"` to efficiently distribute the weights across all available GPUs.

```py
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_4bit=True)

model_4bit = AutoModelForCausalLM.from_pretrained(
    "bigscience/bloom-1b7",
    quantization_config=quantization_config
)
```

By default, all other modules such as [torch.nn.LayerNorm](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html) are converted to `torch.float16`. You can change the data type of these modules with the `torch_dtype` parameter.. Setting `torch_dtype="auto"` loads the model in the data type defined in a model's `config.json` file.

```py
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_4bit=True)

model_4bit = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-350m",
    quantization_config=quantization_config, 
    torch_dtype="auto"
)
model_4bit.model.decoder.layers[-1].final_layer_norm.weight.dtype
```

Make sure you have the latest bitsandbytes version so you can serialize 4-bit models and push them to the Hub with [`~PreTrainedModel.push_to_hub`]. Use [`~PreTrainedModel.save_pretrained`] to save the 4-bit model locally.  

</hfoption>
</hfoptions>

> [!WARNING]
> 8 and 4-bit training is only supported for training *extra* parameters.

Check your memory footprint with `get_memory_footprint`.

```py
print(model.get_memory_footprint())
```

Load quantized models with [`~PreTrainedModel.from_pretrained`] without a `quantization_config`.

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{your_username}/bloom-560m-8bit", device_map="auto")
```

## LLM.int8

This section explores some of the specific features of 8-bit quantization, such as offloading, outlier thresholds, skipping module conversion, and finetuning.

### Offloading

8-bit models can offload weights between the CPU and GPU to fit very large models into memory. The weights dispatched to the CPU are stored in **float32** and aren't converted to 8-bit. For example, enable offloading for [bigscience/bloom-1b7](https://huggingface.co/bigscience/bloom-1b7) through [`BitsAndBytesConfig`].

```py
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)
```

Design a custom device map to fit everything on your GPU except for the `lm_head`, which is dispatched to the CPU.

```py
device_map = {
    "transformer.word_embeddings": 0,
    "transformer.word_embeddings_layernorm": 0,
    "lm_head": "cpu",
    "transformer.h": 0,
    "transformer.ln_f": 0,
}
```

Now load your model with the custom `device_map` and `quantization_config`.

```py
model_8bit = AutoModelForCausalLM.from_pretrained(
    "bigscience/bloom-1b7",
    torch_dtype="auto",
    device_map=device_map,
    quantization_config=quantization_config,
)
```

### Outlier threshold

An "outlier" is a hidden state value greater than a certain threshold, and these values are computed in fp16. While the values are usually normally distributed ([-3.5, 3.5]), this distribution can be very different for large models ([-60, 6] or [6, 60]). 8-bit quantization works well for values ~5, but beyond that, there is a significant performance penalty. A good default threshold value is 6, but a lower threshold may be needed for more unstable models (small models or finetuning).

To find the best threshold for your model, experiment with the `llm_int8_threshold` parameter in [`BitsAndBytesConfig`].

```py
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

model_id = "bigscience/bloom-1b7"

quantization_config = BitsAndBytesConfig(
    llm_int8_threshold=10,
)

model_8bit = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map=device_map,
    quantization_config=quantization_config,
)
```

### Skip module conversion

For some models, like [Jukebox](model_doc/jukebox), you don't need to quantize every module to 8-bit because it can actually cause instability. With Jukebox, there are several `lm_head` modules that should be skipped using the `llm_int8_skip_modules` parameter in [`BitsAndBytesConfig`].

```py
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_id = "bigscience/bloom-1b7"

quantization_config = BitsAndBytesConfig(
    llm_int8_skip_modules=["lm_head"],
)

model_8bit = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
    quantization_config=quantization_config,
)
```

### Finetuning

The [PEFT](https://github.com/huggingface/peft) library supports fine-tuning large models like [flan-t5-large](https://huggingface.co/google/flan-t5-large) and [facebook/opt-6.7b](https://huggingface.co/facebook/opt-6.7b) with 8-bit quantization. You don't need to pass the `device_map` parameter for training because it automatically loads your model on a GPU. However, you can still customize the device map with the `device_map` parameter (`device_map="auto"` should only be used for inference).

## QLoRA

This section explores some of the specific features of 4-bit quantization, such as changing the compute data type, the Normal Float 4 (NF4) data type, and nested quantization.

### Compute data type

Change the data type from float32 (the default value) to bf16 in [`BitsAndBytesConfig`] to speedup computation.

```py
import torch
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
```

### Normal Float 4 (NF4)

NF4 is a 4-bit data type from the [QLoRA](https://hf.co/papers/2305.14314) paper, adapted for weights initialized from a normal distribution. You should use NF4 for training 4-bit base models.

```py
from transformers import BitsAndBytesConfig

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
)

model_nf4 = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", quantization_config=nf4_config)
```

For inference, the `bnb_4bit_quant_type` does not have a huge impact on performance. However, to remain consistent with the model weights, you should use the `bnb_4bit_compute_dtype` and `torch_dtype` values.

### Nested quantization

Nested quantization can save additional memory at no additional performance cost. This feature performs a second quantization of the already quantized weights to save an additional 0.4 bits/parameter. For example, with nested quantization, you can finetune a [Llama-13b](https://huggingface.co/meta-llama/Llama-2-13b) model on a 16GB NVIDIA T4 GPU with a sequence length of 1024, a batch size of 1, and enable gradient accumulation with 4 steps.

```py
from transformers import BitsAndBytesConfig

double_quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
)

model_double_quant = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b", torch_dtype="auto", quantization_config=double_quant_config)
```

## Dequantizing bitsandbytes models

Once quantized, you can [`~PreTrainedModel.dequantize`] a model to the original precision but this may result in some quality loss. Make sure you have enough GPU memory to fit the dequantized model.

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m", BitsAndBytesConfig(load_in_4bit=True))
model.dequantize()
```

## Resources

Learn more about the details of 8-bit quantization in [A Gentle Introduction to 8-bit Matrix Multiplication for transformers at scale using Hugging Face Transformers, Accelerate and bitsandbytes](https://huggingface.co/blog/hf-bitsandbytes-integration).

Try 4-bit quantization in this [notebook](https://colab.research.google.com/drive/1ge2F1QSK8Q7h0hn3YKuBCOAS0bK8E0wf) and learn more about it's details in [Making LLMs even more accessible with bitsandbytes, 4-bit quantization and QLoRA](https://huggingface.co/blog/4bit-transformers-bitsandbytes).
