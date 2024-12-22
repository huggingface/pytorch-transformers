# Copyright 2024 The HuggingFace Team. All rights reserved.
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
"VPTQ (Vector Post-Training Quantization) integration file"

import torch.nn as nn
from accelerate import init_empty_weights
from vptq import VQuantLinear


def replace_with_vptq_linear(
    model,
    quantization_config=None,
    modules_to_not_convert=None,
    current_key_name=None,
    has_been_replaced=False,
):
    """
    Public method that recursively replaces the Linear layers of the given model with VPTQ quantized layers.
    `accelerate` is needed to use this method. Returns the converted model and a boolean that indicates if the
    conversion has been successfull or not.

    Args:
        model (`torch.nn.Module`):
            The model to convert, can be any `torch.nn.Module` instance.
        quantization_config (`VptqConfig`):
            The quantization config object that contains the quantization parameters.
        modules_to_not_convert (`List[`str`]`, *optional*, defaults to `["lm_head"]`):
            Names of the modules to not convert in `VQuantLinear`. In practice we keep the `lm_head` in full precision
            for numerical stability reasons.
        current_key_name (`list`, *optional*):
            A list that contains the current key name. This is used for recursion and should not be passed by the user.
        has_been_replaced (`bool`, *optional*):
            A boolean that indicates if the conversion has been successful or not. This is used for recursion and
            should not be passed by the user.
    """

    modules_to_not_convert = ["lm_head"] if not modules_to_not_convert else modules_to_not_convert

    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []
        current_key_name.append(name)
        layer_name = ".".join(current_key_name)
        shared_layer_config = quantization_config.shared_layer_config
        config_for_layers = quantization_config.config_for_layers

        if (
            isinstance(module, nn.Linear)
            and layer_name not in modules_to_not_convert
            and ((layer_name in config_for_layers) or (current_key_name[-1] in shared_layer_config))
        ):
            layer_params = config_for_layers.get(layer_name, None) or shared_layer_config.get(
                current_key_name[-1], None
            )

            with init_empty_weights():
                in_features = module.in_features
                out_features = module.out_features

                model._modules[name] = VQuantLinear(
                    in_features,
                    out_features,
                    vector_lens=layer_params["vector_lens"],
                    num_centroids=layer_params["num_centroids"],
                    num_res_centroids=layer_params["num_res_centroids"],
                    group_num=layer_params["group_num"],
                    group_size=layer_params["group_size"],
                    outlier_size=layer_params["outlier_size"],
                    indices_as_float=layer_params["indices_as_float"],
                    enable_norm=layer_params["enable_norm"],
                    enable_perm=layer_params["enable_perm"],
                    is_indice_packed=True,
                    enable_proxy_error=False,
                    bias=module.bias is not None,
                )
                has_been_replaced = True

                # Force requires grad to False to avoid unexpected errors
                model._modules[name].requires_grad_(False)
        if len(list(module.children())) > 0:
            _, has_been_replaced = replace_with_vptq_linear(
                module,
                quantization_config=quantization_config,
                modules_to_not_convert=modules_to_not_convert,
                current_key_name=current_key_name,
                has_been_replaced=has_been_replaced,
            )
        # Remove the last key for recursion
        current_key_name.pop(-1)
    return model, has_been_replaced
