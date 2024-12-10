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

from ..utils import is_optimum_quanto_available, is_torch_available, logging


if is_torch_available():
    import torch

logger = logging.get_logger(__name__)


def replace_with_quanto_layers(
    model,
    quantization_config=None,
    modules_to_not_convert=None,
    current_key_name=None,
    has_been_replaced=False,
):
    """
    Public method that recursively replaces the Linear layers of the given model with Quanto quantized layers.
    Returns the converted model and a boolean that indicates if the conversion has been successfull or not.

    Args:
        model (`torch.nn.Module`):
            The model to convert, can be any `torch.nn.Module` instance.
        quantization_config (`AqlmConfig`, defaults to `None`):
            The quantization config object that contains the quantization parameters.
        modules_to_not_convert (`list`, *optional*, defaults to `None`):
            A list of modules to not convert. If a module name is in the list (e.g. `lm_head`), it will not be
            converted.
        current_key_name (`list`, *optional*, defaults to `None`):
            A list that contains the current key name. This is used for recursion and should not be passed by the user.
        has_been_replaced (`bool`, *optional*, defaults to `None`):
            A boolean that indicates if the conversion has been successful or not. This is used for recursion and
            should not be passed by the user.
    """
    from accelerate import init_empty_weights

    if is_optimum_quanto_available():
        from optimum.quanto import QLayerNorm, QLinear, qfloat8, qint2, qint4, qint8

    w_mapping = {"float8": qfloat8, "int8": qint8, "int4": qint4, "int2": qint2}
    a_mapping = {None: None, "float8": qfloat8, "int8": qint8}

    if modules_to_not_convert is None:
        modules_to_not_convert = []

    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []
        current_key_name.append(name)

        if not any(key in ".".join(current_key_name) for key in modules_to_not_convert):
            with init_empty_weights():
                if isinstance(module, torch.nn.Linear):
                    model._modules[name] = QLinear(
                        in_features=module.in_features,
                        out_features=module.out_features,
                        bias=module.bias is not None,
                        dtype=module.weight.dtype,
                        weights=w_mapping[quantization_config.weights],
                        activations=a_mapping[quantization_config.activations],
                    )
                    model._modules[name].requires_grad_(False)
                    has_been_replaced = True
                elif isinstance(module, torch.nn.LayerNorm):
                    if quantization_config.activations is not None:
                        model._modules[name] = QLayerNorm(
                            module.normalized_shape,
                            module.eps,
                            module.elementwise_affine,
                            module.bias is not None,
                            activations=a_mapping[quantization_config.activations],
                        )
                        has_been_replaced = True
        if len(list(module.children())) > 0:
            _, has_been_replaced = replace_with_quanto_layers(
                module,
                quantization_config=quantization_config,
                modules_to_not_convert=modules_to_not_convert,
                current_key_name=current_key_name,
                has_been_replaced=has_been_replaced,
            )
        # Remove the last key for recursion
        current_key_name.pop(-1)
    return model, has_been_replaced
