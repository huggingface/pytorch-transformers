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
import importlib
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from packaging import version

from .base import HfQuantizer
from .quantizers_utils import get_module_from_name


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel

from ..utils import (
    is_accelerate_available,
    is_optimum_quanto_available,
    is_torch_available,
    logging,
)
from ..utils.quantization_config import QuantoConfig


if is_torch_available():
    import torch

logger = logging.get_logger(__name__)


class QuantoHfQuantizer(HfQuantizer):
    """
    Quantizer for the quanto library
    """

    required_packages = ["quanto", "accelerate"]
    requires_parameters_quantization = True
    requires_calibration = False

    def __init__(self, quantization_config: QuantoConfig, **kwargs):
        super().__init__(quantization_config, **kwargs)
        self.post_init()

    def post_init(self):
        r"""
        Safety checker
        """
        if self.quantization_config.activations is not None and not self.pre_quantized:
            raise ValueError(
                "We don't support quantizing the activations with transformers library."
                "Use quanto library for more complex use cases such as activations quantization, calibration and quantization aware training."
            )

    def validate_environment(self, *args, **kwargs):
        if not is_optimum_quanto_available():
            raise ImportError(
                "Loading an optimum-quanto quantized model requires optimum-quanto library (`pip install optimum-quanto`)"
            )
        if not is_accelerate_available():
            raise ImportError(
                "Loading an optimum-quanto quantized model requires accelerate library (`pip install accelerate`)"
            )

    def update_device_map(self, device_map):
        if device_map is None:
            device_map = {"": "cpu"}
            logger.info(
                "The device_map was not initialized. "
                "Setting device_map to {'':'cpu'}. "
                "If you want to use the model for inference, please set device_map ='auto'"
            )
        return device_map

    def update_torch_dtype(self, torch_dtype: "torch.dtype") -> "torch.dtype":
        if torch_dtype is None:
            logger.info("You did not specify `torch_dtype` in `from_pretrained`. Setting it to `torch.float32`.")
            torch_dtype = torch.float32
        return torch_dtype

    def update_missing_keys(self, model, missing_keys: List[str], prefix: str) -> List[str]:
        if is_optimum_quanto_available():
            from optimum.quanto import QModuleMixin

        not_missing_keys = []
        for name, module in model.named_modules():
            if isinstance(module, QModuleMixin):
                for missing in missing_keys:
                    if (
                        (name in missing or name in f"{prefix}.{missing}")
                        and not missing.endswith(".weight")
                        and not missing.endswith(".bias")
                    ):
                        not_missing_keys.append(missing)
        return [k for k in missing_keys if k not in not_missing_keys]

    def check_quantized_param(
        self,
        model: "PreTrainedModel",
        param_value: "torch.Tensor",
        param_name: str,
        state_dict: Dict[str, Any],
        **kwargs,
    ) -> bool:
        """
        Check if a parameter needs to be quantized.
        """
        if is_optimum_quanto_available():
            from optimum.quanto import QModuleMixin

        device_map = kwargs.get("device_map", None)
        param_device = kwargs.get("param_device", None)
        # we don't quantize the model if the module is going to be offloaded to the cpu
        if device_map is not None and param_device is not None:
            device_map_values = set(device_map.values())
            if param_device == "cpu" and len(device_map_values) > 1:
                if not (device_map_values == {"cpu"} or device_map_values == {"cpu", "disk"}):
                    return False

        module, tensor_name = get_module_from_name(model, param_name)
        # We only quantize the weights and the bias is not quantized.
        if isinstance(module, QModuleMixin) and "weight" in tensor_name:
            # if the weights are quantized, don't need to recreate it again with `create_quantized_param`
            return not module.frozen
        else:
            return False

    def adjust_max_memory(self, max_memory: Dict[str, Union[int, str]]) -> Dict[str, Union[int, str]]:
        max_memory = {key: val * 0.90 for key, val in max_memory.items()}
        return max_memory

    def create_quantized_param(
        self,
        model: "PreTrainedModel",
        param_value: "torch.Tensor",
        param_name: str,
        target_device: "torch.device",
        *args,
        **kwargs,
    ):
        """
        Create the quantized parameter by calling .freeze() after setting it to the module.
        """
        from accelerate.utils import set_module_tensor_to_device

        set_module_tensor_to_device(model, param_name, target_device, param_value)
        module, _ = get_module_from_name(model, param_name)
        module.freeze()
        module.weight.requires_grad = False

    def adjust_target_dtype(self, target_dtype: "torch.dtype") -> "torch.dtype":
        if version.parse(importlib.metadata.version("accelerate")) > version.parse("0.27.0"):
            from accelerate.utils import CustomDtype

            mapping = {
                "int8": torch.int8,
                "float8": CustomDtype.FP8,
                "int4": CustomDtype.INT4,
                "int2": CustomDtype.INT2,
            }
            target_dtype = mapping[self.quantization_config.weights]
            return target_dtype
        else:
            raise ValueError(
                "You are using `device_map='auto'` on an optimum-quanto quantized model. To automatically compute"
                " the appropriate device map, you should upgrade your `accelerate` library,"
                "`pip install --upgrade accelerate` or install it from source."
            )

    def _process_model_before_weight_loading(
        self, model: "PreTrainedModel", keep_in_fp32_modules: List[str] = [], **kwargs
    ):
        from ..integrations import get_keys_to_not_convert, replace_with_quanto_layers

        # We keep some modules such as the lm_head in their original dtype for numerical stability reasons
        if self.quantization_config.modules_to_not_convert is None:
            self.modules_to_not_convert = get_keys_to_not_convert(model)
        else:
            self.modules_to_not_convert = self.quantization_config.modules_to_not_convert

        if not isinstance(self.modules_to_not_convert, list):
            self.modules_to_not_convert = [self.modules_to_not_convert]

        self.modules_to_not_convert.extend(keep_in_fp32_modules)

        model, _ = replace_with_quanto_layers(
            model, modules_to_not_convert=self.modules_to_not_convert, quantization_config=self.quantization_config
        )
        model.config.quantization_config = self.quantization_config

    def _process_model_after_weight_loading(self, model):
        return model

    @property
    def is_trainable(self, model: Optional["PreTrainedModel"] = None):
        return True

    def is_serializable(self, safe_serialization=None):
        return False
