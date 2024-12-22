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
from typing import TYPE_CHECKING, Optional

from .base import HfQuantizer


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel

from ..utils import is_accelerate_available, is_torch_available, is_vptq_available, logging
from ..utils.quantization_config import QuantizationConfigMixin


if is_torch_available():
    import torch

logger = logging.get_logger(__name__)


class VptqHfQuantizer(HfQuantizer):
    """
    Quantizer of the VPTQ method. Enables the loading of prequantized models.
    """

    requires_calibration = True
    required_packages = ["vptq"]

    def __init__(self, quantization_config: QuantizationConfigMixin, **kwargs):
        super().__init__(quantization_config, **kwargs)
        self.quantization_config = quantization_config

    def validate_environment(self, *args, **kwargs):
        if not is_accelerate_available():
            raise ImportError("Using `vptq` quantization requires Accelerate: `pip install accelerate`")

        if not is_vptq_available():
            raise ImportError("Using `vptq` quantization requires VPTQ>=0.0.4: `pip install -U vptq`")

    def update_torch_dtype(self, torch_dtype: "torch.dtype") -> "torch.dtype":
        if torch_dtype is None:
            if torch.cuda.is_available():
                torch_dtype = torch.float16
                logger.info(
                    "CUDA available. Assuming VPTQ inference on GPU and loading the model in `torch.float16`. To overwrite it, set `torch_dtype` manually."
                )
            else:
                import vptq

                device_availability = getattr(vptq, "device_availability", lambda device: False)
                if device_availability("cpu") is True:
                    raise RuntimeError("No GPU found. Please wait for the next release of VPTQ to use CPU inference")
                torch_dtype = torch.float32
                logger.info("No GPU found. Assuming VPTQ inference on CPU and loading the model in `torch.float32`.")
        return torch_dtype

    def _process_model_before_weight_loading(
        self,
        model: "PreTrainedModel",
        **kwargs,
    ):
        """
        we don't have param like modules_to_not_convert to indicate which layers should not be quantized
        because `quantization_config` include the layers that should be quantized
        """
        from ..integrations import replace_with_vptq_linear

        modules_to_not_convert = kwargs.get("modules_to_not_convert", []) + (
            self.quantization_config.modules_to_not_convert or []
        )

        replace_with_vptq_linear(
            model,
            quantization_config=self.quantization_config,
            modules_to_not_convert=modules_to_not_convert,
        )
        model.config.quantization_config = self.quantization_config

    def _process_model_after_weight_loading(self, model: "PreTrainedModel", **kwargs):
        return model

    @property
    def is_trainable(self, model: Optional["PreTrainedModel"] = None):
        return False

    def is_serializable(self, safe_serialization=None):
        return True
