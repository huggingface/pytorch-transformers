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
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .base import HfQuantizer


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel

from ..utils import is_accelerate_available, is_eetq_available, is_torch_available, logging
from .quantizers_utils import get_module_from_name


if is_torch_available():
    import torch


logger = logging.get_logger(__name__)


class EetqHfQuantizer(HfQuantizer):
    """
    8-bit quantization from EETQ quantization method:
        before loading: converts transformer layers into W8A16Linear during loading: load 16bit weight and pass to the
        layer object after: quantizes individual weights in Linear8bitLt into 8bit at first .cuda() call
    """

    requires_parameters_quantization = True
    requires_calibration = False

    required_packages = ["eetq", "accelerate"]

    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)
        self.quantization_config = quantization_config

    def validate_environment(self, *args, **kwargs):
        if not is_eetq_available():
            raise ImportError(
                "Using `eetq` 8-bit quantization requires eetq."
                "Please install the latest version of eetq from : https://github.com/NetEase-FuXi/EETQ"
            )

        try:
            import eetq  # noqa: F401
        except ImportError as exc:
            if "shard_checkpoint" in str(exc):
                # EETQ 1.0.0 is currently broken with the latest transformers because it tries to import the removed
                # shard_checkpoint function, see https://github.com/NetEase-FuXi/EETQ/issues/34.
                # TODO: Update message once eetq releases a fix
                raise ImportError(
                    "You are using a version of EETQ that is incompatible with the current transformers version. "
                    "Either downgrade transformers to <= v4.46.3 or, if available, upgrade EETQ to > v1.0.0."
                ) from exc
            else:
                raise

        if not is_accelerate_available():
            raise ImportError("Loading an EETQ quantized model requires accelerate (`pip install accelerate`)")

        if kwargs.get("from_tf", False) or kwargs.get("from_flax", False):
            raise ValueError(
                "Converting into 8-bit weights from tf/flax weights is currently not supported, please make"
                " sure the weights are in PyTorch format."
            )

        if not torch.cuda.is_available():
            raise RuntimeError("No GPU found. A GPU is needed for quantization.")

        device_map = kwargs.get("device_map", None)
        if device_map is None:
            logger.warning_once(
                "You have loaded an EETQ model on CPU and have a CUDA device available, make sure to set "
                "your model on a GPU device in order to run your model."
            )
        elif device_map is not None:
            if isinstance(device_map, dict) and ("cpu" in device_map.values() or "disk" in device_map.values()):
                raise ValueError(
                    "You are attempting to load an EETQ model with a device_map that contains a CPU or disk device."
                    " This is not supported. Please remove the CPU or disk device from the device_map."
                )

    def update_torch_dtype(self, torch_dtype: "torch.dtype") -> "torch.dtype":
        if torch_dtype is None:
            torch_dtype = torch.float16
            logger.info(
                "Overriding torch_dtype=%s with `torch_dtype=torch.float16` due to "
                "requirements of `eetq` to enable model loading in 8-bit. "
                "Pass your own torch_dtype to specify the dtype of the remaining non-linear layers or pass"
                " torch_dtype=torch.float16 to remove this warning.",
                torch_dtype,
            )
        elif torch_dtype != torch.float16:
            logger.info("We suggest you to set `torch_dtype=torch.float16` for better efficiency with EETQ.")
        return torch_dtype

    def check_quantized_param(
        self,
        model: "PreTrainedModel",
        param_value: "torch.Tensor",
        param_name: str,
        state_dict: Dict[str, Any],
        **kwargs,
    ):
        from eetq import EetqLinear

        module, tensor_name = get_module_from_name(model, param_name)

        if isinstance(module, EetqLinear):
            if self.pre_quantized or tensor_name == "bias":
                if tensor_name == "weight" and param_value.dtype != torch.int8:
                    raise ValueError("Expect quantized weights but got an unquantized weight")
                return False
            else:
                if tensor_name == "weight_scale":
                    raise ValueError("Expect unquantized weights but got a quantized weight_scale")
                return True
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
        quantizes weights into qweight and weight_scales
        """
        from eetq import quantize_and_preprocess_weights

        module, tensor_name = get_module_from_name(model, param_name)
        new_value, weight_scale = quantize_and_preprocess_weights(param_value)

        module._buffers[tensor_name] = new_value.to(target_device)
        module.register("weight_scales", weight_scale.to(target_device))

    def _process_model_after_weight_loading(self, model: "PreTrainedModel", **kwargs):
        return model

    def _process_model_before_weight_loading(
        self,
        model: "PreTrainedModel",
        device_map,
        keep_in_fp32_modules: List[str] = [],
        **kwargs,
    ):
        from ..integrations import get_keys_to_not_convert, replace_with_eetq_linear

        self.modules_to_not_convert = get_keys_to_not_convert(model)

        if self.quantization_config.modules_to_not_convert is not None:
            self.modules_to_not_convert.extend(self.quantization_config.modules_to_not_convert)

        model = replace_with_eetq_linear(
            model,
            modules_to_not_convert=self.modules_to_not_convert,
            quantization_config=self.quantization_config,
            pre_quantized=self.pre_quantized,
        )

        model.config.quantization_config = self.quantization_config

    def is_serializable(self, safe_serialization=None):
        return True

    @property
    def is_trainable(self) -> bool:
        return True
