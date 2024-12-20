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

# compressed-tensors

[compressed-tensors](https://github.com/neuralmagic/compressed-tensors) extends [safetensors](https://github.com/huggingface/safetensors) files to compressed tensor data types to provide a unified checkpoint format for storing and loading various quantization and sparsity formats such dense, int-quantized (int8), float-quantized (fp8), and pack-quantized (int4 or int8 weight-quantized packed into int32).

compressed-tensors supports fine-tuning with [PEFT](https://huggingface.co/docs/peft) and includes the following features as well.

- fp8, int4, int8 weight and activation precisions.
- Quantization scales and zero-points strategies for [tensor, channel, group, block, token](https://github.com/neuralmagic/compressed-tensors/blob/83b2e7a969d70606421a76b9a3d112646077c8de/src/compressed_tensors/quantization/quant_args.py#L43-L52).
- Dynamic per-token activation quantization (or any static strategy).
- Weight sparsity (unstructured or semi-structured like 2:4) can be composed with quantization for extreme compression.
- Quantization of arbitrary modules, not just [nn.Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) modules.
- Targeted support for specific modules by name or class.

Install compressed-tensors from [PyPI](https://pypi.org/project/compressed-tensors) to get the latest stable release (recommended) or install it from source to get the latest features.

<hfoptions id="install">
<hfoption id="PyPI">

```bash
pip install compressed-tensors
```

</hfoption>
<hfoption id="source code>

```bash
git clone https://github.com/neuralmagic/compressed-tensors
cd compressed-tensors
pip install -e .
```

</hfoption>
</hfoptions>

Only models that have already been quantized can be loaded at the moment, and once a model is loaded, it cannot be saved. To quantize a model into the compressed-tensors format, see [llm-compressor](https://github.com/vllm-project/llm-compressor). Alternatively, models can be created independently and serizlied with a compressed-tensors config.

```python
from transformers import AutoModelForCausalLM

ct_model = AutoModelForCausalLM.from_pretrained("nm-testing/Meta-Llama-3.1-8B-Instruct-FP8-hf", device_map="auto")

# measure memory usage
mem_params = sum([param.nelement()*param.element_size() for param in ct_model.parameters()])
print(f"{mem/2**30:.4f} GB")
# 8.4575 GB
```

## Model checkpoint

compressed-tensor models are defined through its configuration entry. The following example is taken from the [nm-testing/Meta-Llama-3.1-8B-Instruct-FP8-hf](https://huggingface.co/nm-testing/Meta-Llama-3.1-8B-Instruct-FP8-hf/blob/main/config.json) `config.json` file.

There are a lot of entries to allow for flexible expression both during and after compression, but the entries for loading and inference can be simplified to focus on just a few key entries.

```yaml
"quantization_config": {
  "config_groups": {
    "group_0": {
      "input_activations": {
        "num_bits": 8,
        "strategy": "tensor",
        "type": "float"
      },
      "targets": ["Linear"],
      "weights": {
        "num_bits": 8,
        "strategy": "tensor",
        "type": "float"
      }
    }
  },
  "format": "naive-quantized",
  "ignore": ["lm_head"],
  "quant_method": "compressed-tensors",
  "quantization_status": "frozen"
},
```

The config file specifies the quantization of a config group (`group_0`), which includes weight and activation quantization to fp8 with a static per-tensor strategy. The `lm_head` module is unquantized as shown in the `ignore` key.

For a more detailed look at the model weights, use the [safetensors viewer](https://huggingface.co/nm-testing/Meta-Llama-3.1-8B-Instruct-FP8-hf?show_file_info=model.safetensors.index.json) on the model card to see the quantized weights, input scale, and weight scale for all [nn.Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) modules.

| Tensors | Shape |	Precision |
| ------- | ----- | --------- |
model.layers.0.input_layernorm.weight	| [4 096]	| BF16 
model.layers.0.mlp.down_proj.input_scale	| [1]	| BF16 
model.layers.0.mlp.down_proj.weight	| [4 096, 14 336] |	F8_E4M3 
model.layers.0.mlp.down_proj.weight_scale |	[1]	| BF16 
model.layers.0.mlp.gate_proj.input_scale |	[1]	| BF16 
model.layers.0.mlp.gate_proj.weight	| [14 336, 4 096]	| F8_E4M3 
model.layers.0.mlp.gate_proj.weight_scale	| [1] |	BF16 
model.layers.0.mlp.up_proj.input_scale|	[1]	|BF16 
model.layers.0.mlp.up_proj.weight |	[14 336, 4 096]	| F8_E4M3 
model.layers.0.mlp.up_proj.weight_scale | [1]	| BF16 
model.layers.0.post_attention_layernorm.weight |	[4 096]	|BF16 
model.layers.0.self_attn.k_proj.input_scale |	[1]	|  BF16
model.layers.0.self_attn.k_proj.weight |	[1 024, 4 096]|	F8_E4M3
model.layers.0.self_attn.k_proj.weight_scale |[1]	| BF16 
model.layers.0.self_attn.o_proj.input_scale	| [1]	| BF16
model.layers.0.self_attn.o_proj.weight | [4 096, 4 096]	| F8_E4M3 
model.layers.0.self_attn.o_proj.weight_scale | [1]	| BF16 
model.layers.0.self_attn.q_proj.input_scale	| [1]	| BF16 
model.layers.0.self_attn.q_proj.weight | [4 096, 4 096]	| F8_E4M3 
model.layers.0.self_attn.q_proj.weight_scale |	[1] | BF16 
model.layers.0.self_attn.v_proj.input_scale	| [1] | BF16 
model.layers.0.self_attn.v_proj.weight |	[1 024, 4 096]	| F8_E4M3 
model.layers.0.self_attn.v_proj.weight_scale |	[1] |	BF16 

When loading a compressed-tensors model with the [`~quantizers.HFQuantizer`] integration, all the [nn.Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) modules specified in the quantization config are replaced by [CompressedLinear](https://github.com/neuralmagic/compressed-tensors/blob/975cb223b19fcac2b98a4271d17668462d4d6e1d/src/compressed_tensors/linear/compressed_linear.py#L30) modules that manage the compressed weights and forward pass for inference. The `lm_head` module is still kept as an unquantized nn.Linear module.

```python
from transformers import AutoModelForCausalLM

ct_model = AutoModelForCausalLM.from_pretrained("nm-testing/Meta-Llama-3.1-8B-Instruct-FP8-hf")
print(ct_model)
"""
LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(128256, 4096)
    (layers): ModuleList(
      (0-31): 32 x LlamaDecoderLayer(
        (self_attn): LlamaSdpaAttention(
          (q_proj): CompressedLinear(
            in_features=4096, out_features=4096, bias=False
            (input_observer): MovingAverageMinMaxObserver()
            (weight_observer): MovingAverageMinMaxObserver()
          )
          (k_proj): CompressedLinear(
            in_features=4096, out_features=1024, bias=False
            (input_observer): MovingAverageMinMaxObserver()
            (weight_observer): MovingAverageMinMaxObserver()
          )
          (v_proj): CompressedLinear(
            in_features=4096, out_features=1024, bias=False
            (input_observer): MovingAverageMinMaxObserver()
            (weight_observer): MovingAverageMinMaxObserver()
          )
          (o_proj): CompressedLinear(
            in_features=4096, out_features=4096, bias=False
            (input_observer): MovingAverageMinMaxObserver()
            (weight_observer): MovingAverageMinMaxObserver()
          )
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): CompressedLinear(
            in_features=4096, out_features=14336, bias=False
            (input_observer): MovingAverageMinMaxObserver()
            (weight_observer): MovingAverageMinMaxObserver()
          )
          (up_proj): CompressedLinear(
            in_features=4096, out_features=14336, bias=False
            (input_observer): MovingAverageMinMaxObserver()
            (weight_observer): MovingAverageMinMaxObserver()
          )
          (down_proj): CompressedLinear(
            in_features=14336, out_features=4096, bias=False
            (input_observer): MovingAverageMinMaxObserver()
            (weight_observer): MovingAverageMinMaxObserver()
          )
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm((4096,), eps=1e-05)
        (post_attention_layernorm): LlamaRMSNorm((4096,), eps=1e-05)
      )
    )
    (norm): LlamaRMSNorm((4096,), eps=1e-05)
    (rotary_emb): LlamaRotaryEmbedding()
  )
  (lm_head): Linear(in_features=4096, out_features=128256, bias=False)
)
"""
```
