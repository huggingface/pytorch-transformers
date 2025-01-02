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
from functools import cached_property
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from packaging import version

from .base import HfQuantizer
from .quantizers_utils import get_module_from_name


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel

from ..utils import (
    ACCELERATE_MIN_VERSION,
    is_accelerate_available,
    is_bitsandbytes_available,
    is_torch_available,
    is_torch_npu_available,
    is_torch_xpu_available,
    logging,
)


if is_torch_available():
    import torch

    from ..pytorch_utils import Conv1D

logger = logging.get_logger(__name__)


class Bnb4BitHfQuantizer(HfQuantizer):
    """
    4-bit quantization from bitsandbytes.py quantization method:
        before loading: converts transformer layers into Linear4bit during loading: load 16bit weight and pass to the
        layer object after: quantizes individual weights in Linear4bit into 4bit at the first .cuda() call
        saving:
            from state dict, as usual; saves weights and `quant_state` components
        loading:
            need to locate `quant_state` components and pass to Param4bit constructor
    """

    use_keep_in_fp32_modules = True
    requires_parameters_quantization = True
    requires_calibration = False

    required_packages = ["bitsandbytes", "accelerate"]

    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)

        if self.quantization_config.llm_int8_skip_modules is not None:
            self.modules_to_not_convert = self.quantization_config.llm_int8_skip_modules

    def validate_environment(self, *args, **kwargs):
        if not is_accelerate_available():
            raise ImportError(
                f"Using `bitsandbytes` 4-bit quantization requires Accelerate: `pip install 'accelerate>={ACCELERATE_MIN_VERSION}'`"
            )
        if not is_bitsandbytes_available():
            raise ImportError(
                "Using `bitsandbytes` 4-bit quantization requires the latest version of bitsandbytes: `pip install -U bitsandbytes`"
            )

        from ..integrations import validate_bnb_backend_availability
        from ..utils import is_bitsandbytes_multi_backend_available

        bnb_multibackend_is_enabled = is_bitsandbytes_multi_backend_available()
        validate_bnb_backend_availability(raise_exception=True)

        if kwargs.get("from_tf", False) or kwargs.get("from_flax", False):
            raise ValueError(
                "Converting into 4-bit or 8-bit weights from tf/flax weights is currently not supported, please make"
                " sure the weights are in PyTorch format."
            )

        device_map = kwargs.get("device_map", None)
        if (
            device_map is not None
            and isinstance(device_map, dict)
            and not self.quantization_config.llm_int8_enable_fp32_cpu_offload
        ):
            device_map_without_lm_head = {
                key: device_map[key] for key in device_map.keys() if key not in self.modules_to_not_convert
            }
            if set(device_map.values()) == {"cpu"} and bnb_multibackend_is_enabled:
                pass
            elif "cpu" in device_map_without_lm_head.values() or "disk" in device_map_without_lm_head.values():
                raise ValueError(
                    "Some modules are dispatched on the CPU or the disk. Make sure you have enough GPU RAM to fit the "
                    "quantized model. If you want to dispatch the model on the CPU or the disk while keeping these modules "
                    "in 32-bit, you need to set `llm_int8_enable_fp32_cpu_offload=True` and pass a custom `device_map` to "
                    "`from_pretrained`. Check "
                    "https://huggingface.co/docs/transformers/main/en/main_classes/quantization#offload-between-cpu-and-gpu "
                    "for more details. "
                )

        if version.parse(importlib.metadata.version("bitsandbytes")) < version.parse("0.39.0"):
            raise ValueError(
                "You have a version of `bitsandbytes` that is not compatible with 4bit inference and training"
                " make sure you have the latest version of `bitsandbytes` installed"
            )

    def adjust_target_dtype(self, target_dtype: "torch.dtype") -> "torch.dtype":
        if version.parse(importlib.metadata.version("accelerate")) > version.parse("0.19.0"):
            from accelerate.utils import CustomDtype

            if target_dtype != torch.int8:
                logger.info("target_dtype {target_dtype} is replaced by `CustomDtype.INT4` for 4-bit BnB quantization")
            return CustomDtype.INT4
        else:
            raise ValueError(
                "You are using `device_map='auto'` on a 4bit loaded version of the model. To automatically compute"
                " the appropriate device map, you should upgrade your `accelerate` library,"
                "`pip install --upgrade accelerate` or install it from source to support fp4 auto device map"
                "calculation. You may encounter unexpected behavior, or pass your own device map"
            )

    def check_quantized_param(
        self,
        model: "PreTrainedModel",
        param_value: "torch.Tensor",
        param_name: str,
        state_dict: Dict[str, Any],
        **kwargs,
    ) -> bool:
        import bitsandbytes as bnb

        module, tensor_name = get_module_from_name(model, param_name)
        if isinstance(module._parameters.get(tensor_name, None), bnb.nn.Params4bit):
            # Add here check for loaded components' dtypes once serialization is implemented
            return True
        elif isinstance(module, bnb.nn.Linear4bit) and tensor_name == "bias":
            # bias could be loaded by regular set_module_tensor_to_device() from accelerate,
            # but it would wrongly use uninitialized weight there.
            return True
        else:
            return False

    def create_quantized_param(
        self,
        model: "PreTrainedModel",
        param_value: "torch.Tensor",
        param_name: str,
        target_device: "torch.device",
        state_dict: Dict[str, Any],
        unexpected_keys: Optional[List[str]] = None,
    ):
        """
        combines logic from _load_state_dict_into_meta_model and .integrations.bitsandbytes.py::set_module_quantized_tensor_to_device()
        """
        import bitsandbytes as bnb

        module, tensor_name = get_module_from_name(model, param_name)

        if tensor_name not in module._parameters:
            raise ValueError(f"{module} does not have a parameter or a buffer named {tensor_name}.")

        old_value = getattr(module, tensor_name)

        # `torch.Tensor.to(<int num>)` is not supported by `torch_npu` (see this [issue](https://github.com/Ascend/pytorch/issues/16)).
        if isinstance(target_device, int) and is_torch_npu_available():
            target_device = f"npu:{target_device}"
        if tensor_name == "bias":
            if param_value is None:
                new_value = old_value.to(target_device)
            else:
                new_value = param_value.to(target_device)

            new_value = torch.nn.Parameter(new_value, requires_grad=old_value.requires_grad)
            module._parameters[tensor_name] = new_value
            return

        if not isinstance(module._parameters[tensor_name], bnb.nn.Params4bit):
            raise ValueError("this function only loads `Linear4bit components`")
        if (
            old_value.device == torch.device("meta")
            and target_device not in ["meta", torch.device("meta")]
            and param_value is None
        ):
            raise ValueError(f"{tensor_name} is on the meta device, we need a `value` to put in on {target_device}.")

        # construct `new_value` for the module._parameters[tensor_name]:
        if self.pre_quantized:
            # 4bit loading. Collecting components for restoring quantized weight
            # This can be expanded to make a universal call for any quantized weight loading

            if not self.is_serializable:
                raise ValueError(
                    "Detected int4 weights but the version of bitsandbytes is not compatible with int4 serialization. "
                    "Make sure to download the latest `bitsandbytes` version. `pip install --upgrade bitsandbytes`."
                )

            if (param_name + ".quant_state.bitsandbytes__fp4" not in state_dict) and (
                param_name + ".quant_state.bitsandbytes__nf4" not in state_dict
            ):
                raise ValueError(
                    f"Supplied state dict for {param_name} does not contain `bitsandbytes__*` and possibly other `quantized_stats` components."
                )

            quantized_stats = {}
            for k, v in state_dict.items():
                if param_name + "." in k:
                    quantized_stats[k] = v
                    if unexpected_keys is not None and k in unexpected_keys:
                        unexpected_keys.remove(k)

            param_kwargs = {}
            if self.is_bnb_supports_quant_storage_module:
                param_kwargs["module"] = module

            new_value = bnb.nn.Params4bit.from_prequantized(
                data=param_value,
                quantized_stats=quantized_stats,
                requires_grad=False,
                device=target_device,
                **param_kwargs,
            )
        else:
            new_value = param_value.to("cpu")

            # Support models using `Conv1D` in place of `nn.Linear` (e.g. openai-community/gpt2) by transposing the weight matrix prior to quantization.
            # Since weights are saved in the correct "orientation", we skip transposing when loading.
            if issubclass(module.source_cls, Conv1D):
                new_value = new_value.T

            kwargs = old_value.__dict__
            new_value = bnb.nn.Params4bit(new_value, requires_grad=False, **kwargs).to(target_device)

        module._parameters[tensor_name] = new_value

    # Copied from transformers.quantizers.quantizer_bnb_8bit.Bnb8BitHfQuantizer.adjust_max_memory
    def adjust_max_memory(self, max_memory: Dict[str, Union[int, str]]) -> Dict[str, Union[int, str]]:
        # need more space for buffers that are created during quantization
        max_memory = {key: val * 0.90 for key, val in max_memory.items()}
        return max_memory

    # Copied from transformers.quantizers.quantizer_bnb_8bit.Bnb8BitHfQuantizer.update_torch_dtype
    def update_torch_dtype(self, torch_dtype: "torch.dtype") -> "torch.dtype":
        if torch_dtype is None:
            # We force the `dtype` to be float16, this is a requirement from `bitsandbytes`
            logger.info(
                "Overriding torch_dtype=%s with `torch_dtype=torch.float16` due to "
                "requirements of `bitsandbytes` to enable model loading in 8-bit or 4-bit. "
                "Pass your own torch_dtype to specify the dtype of the remaining non-linear layers or pass"
                " torch_dtype=torch.float16 to remove this warning.",
                torch_dtype,
            )
            torch_dtype = torch.float16
        return torch_dtype

    def update_device_map(self, device_map):
        if device_map is None:
            if torch.cuda.is_available():
                device_map = {"": torch.cuda.current_device()}
            elif is_torch_npu_available():
                device_map = {"": f"npu:{torch.npu.current_device()}"}
            elif is_torch_xpu_available():
                device_map = {"": f"xpu:{torch.xpu.current_device()}"}
            else:
                device_map = {"": "cpu"}
            logger.info(
                "The device_map was not initialized. "
                f"Setting device_map to {device_map}. "
                "If you want to use the model for inference, please set device_map ='auto' "
            )
        return device_map

    # Copied from transformers.quantizers.quantizer_bnb_8bit.Bnb8BitHfQuantizer._process_model_before_weight_loading
    def _process_model_before_weight_loading(
        self,
        model: "PreTrainedModel",
        device_map,
        keep_in_fp32_modules: List[str] = [],
        **kwargs,
    ):
        from ..integrations import get_keys_to_not_convert, replace_with_bnb_linear

        llm_int8_enable_fp32_cpu_offload = self.quantization_config.llm_int8_enable_fp32_cpu_offload

        # We keep some modules such as the lm_head in their original dtype for numerical stability reasons
        if self.quantization_config.llm_int8_skip_modules is None:
            self.modules_to_not_convert = get_keys_to_not_convert(model)
        else:
            self.modules_to_not_convert = self.quantization_config.llm_int8_skip_modules

        if not isinstance(self.modules_to_not_convert, list):
            self.modules_to_not_convert = [self.modules_to_not_convert]

        self.modules_to_not_convert.extend(keep_in_fp32_modules)

        # Extend `self.modules_to_not_convert` to keys that are supposed to be offloaded to `cpu` or `disk`
        if isinstance(device_map, dict) and len(device_map.keys()) > 1:
            keys_on_cpu = [key for key, value in device_map.items() if value in ["disk", "cpu"]]

            if len(keys_on_cpu) > 0 and not llm_int8_enable_fp32_cpu_offload:
                raise ValueError(
                    "If you want to offload some keys to `cpu` or `disk`, you need to set "
                    "`llm_int8_enable_fp32_cpu_offload=True`. Note that these modules will not be "
                    " converted to 8-bit but kept in 32-bit."
                )
            self.modules_to_not_convert.extend(keys_on_cpu)

        model = replace_with_bnb_linear(
            model, modules_to_not_convert=self.modules_to_not_convert, quantization_config=self.quantization_config
        )
        # TODO: consider bringing replace_with_bnb_linear() code from ..integrations/bitsandbyter.py to here

        model.config.quantization_config = self.quantization_config

    # Copied from transformers.quantizers.quantizer_bnb_8bit.Bnb8BitHfQuantizer._process_model_after_weight_loading with 8bit->4bit
    def _process_model_after_weight_loading(self, model: "PreTrainedModel", **kwargs):
        model.is_loaded_in_4bit = True
        model.is_4bit_serializable = self.is_serializable()
        return model

    def is_serializable(self, safe_serialization=None):
        _is_4bit_serializable = version.parse(importlib.metadata.version("bitsandbytes")) >= version.parse("0.41.3")

        if not _is_4bit_serializable:
            logger.warning(
                "You are calling `save_pretrained` to a 4-bit converted model, but your `bitsandbytes` version doesn't support it. "
                "If you want to save 4-bit models, make sure to have `bitsandbytes>=0.41.3` installed."
            )
            return False

        return True

    @cached_property
    def is_bnb_supports_quant_storage_module(self) -> bool:
        """
        determines if the current version of bitsandbytes supports
        the `module` parameter in `Params4bit.from_prequantized`
        :return:
        """
        return version.parse(importlib.metadata.version("bitsandbytes")) >= version.parse("0.43.3")

    @property
    def is_trainable(self) -> bool:
        return True

    def _dequantize(self, model):
        from ..integrations import dequantize_and_replace

        model = dequantize_and_replace(
            model, self.modules_to_not_convert, quantization_config=self.quantization_config
        )
        return model
