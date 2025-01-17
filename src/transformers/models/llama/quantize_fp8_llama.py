# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from transformers import FbgemmFp8Config, LlamaForCausalLM


modules_to_not_convert = []

# As defined by Meta, we don't quantize the first and last layers as well as the lm_head. Also, we don't quantize the self_attn layers.
modules_to_not_convert.append("model.layers.0")
modules_to_not_convert.append("model.layers.125")
modules_to_not_convert.append("lm_head")
for layer_i in range(1, 125):
    modules_to_not_convert.append(f"model.layers.{layer_i}.self_attn")

quantization_config = FbgemmFp8Config(modules_to_not_convert=modules_to_not_convert)
model_name = "meta-llama/Llama-3.1-405B"

model = LlamaForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map="auto", quantization_config=quantization_config
)

model.save_pretrained(f"{model_name}-FP8")
