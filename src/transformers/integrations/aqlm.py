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
"AQLM (Additive Quantization of Language Model) integration file"

from ..utils import ACCELERATE_MIN_VERSION, is_accelerate_available, is_aqlm_available, is_torch_available


if is_torch_available():
    import torch.nn as nn


def replace_with_aqlm_linear(
    model,
    quantization_config=None,
    modules_to_not_convert=None,
    current_key_name=None,
    has_been_replaced=False,
):
    """
    Public method that recursively replaces the Linear layers of the given model with AQLM quantized layers.
    `accelerate` is needed to use this method. Returns the converted model and a boolean that indicates if the
    conversion has been successfull or not.

    Args:
        model (`torch.nn.Module`):
            The model to convert, can be any `torch.nn.Module` instance.
        quantization_config (`AqlmConfig`):
            The quantization config object that contains the quantization parameters.
        modules_to_not_convert (`list[str]`, *optional*):
            A list of nn.Linear weights to not convert. If a parameter path is in the list (e.g. `lm_head.weight`), the corresponding module will not be
            converted.
        current_key_name (`list`, *optional*):
            A list that contains the current key name. This is used for recursion and should not be passed by the user.
        has_been_replaced (`bool`, *optional*):
            A boolean that indicates if the conversion has been successful or not. This is used for recursion and
            should not be passed by the user.
    """
    if not is_aqlm_available():
        raise ValueError("AQLM is not available. Please install it with `pip install aqlm[cpu,gpu]`")

    if not is_accelerate_available():
        raise ValueError(
            f"AQLM requires Accelerate to be installed: `pip install 'accelerate>={ACCELERATE_MIN_VERSION}'`"
        )

    if modules_to_not_convert is None:
        modules_to_not_convert = []

    from accelerate import init_empty_weights
    from aqlm import QuantizedLinear

    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []
        current_key_name.append(name)

        if isinstance(module, nn.Linear):
            # Check if the current key is not in the `modules_to_not_convert`
            if ".".join(current_key_name) + ".weight" not in modules_to_not_convert:
                with init_empty_weights():
                    in_features = module.in_features
                    out_features = module.out_features

                    model._modules[name] = QuantizedLinear(
                        in_features,
                        out_features,
                        bias=module.bias is not None,
                        in_group_size=quantization_config.in_group_size,
                        out_group_size=quantization_config.out_group_size,
                        num_codebooks=quantization_config.num_codebooks,
                        nbits_per_codebook=quantization_config.nbits_per_codebook,
                    )
                    has_been_replaced = True

                    # Store the module class in case we need to transpose the weight later
                    model._modules[name].source_cls = type(module)
                    # Force requires grad to False to avoid unexpected errors
                    model._modules[name].requires_grad_(False)
        if len(list(module.children())) > 0:
            _, has_been_replaced = replace_with_aqlm_linear(
                module,
                quantization_config=quantization_config,
                modules_to_not_convert=modules_to_not_convert,
                current_key_name=current_key_name,
                has_been_replaced=has_been_replaced,
            )
        # Remove the last key for recursion
        current_key_name.pop(-1)
    return model, has_been_replaced
