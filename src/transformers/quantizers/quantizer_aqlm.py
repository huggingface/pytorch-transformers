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
from typing import TYPE_CHECKING, Optional

from packaging import version

from .base import HfQuantizer


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel

from ..integrations import replace_with_aqlm_linear
from ..utils import is_accelerate_available, is_aqlm_available, is_torch_available, logging
from ..utils.quantization_config import QuantizationConfigMixin


if is_torch_available():
    import torch

logger = logging.get_logger(__name__)


class AqlmHfQuantizer(HfQuantizer):
    """
    Quantizer of the AQLM method. Enables the loading of prequantized models.
    """

    requires_calibration = True
    required_packages = ["aqlm"]
    optimum_quantizer = None

    def __init__(self, quantization_config: QuantizationConfigMixin, **kwargs):
        super().__init__(quantization_config, **kwargs)
        self.quantization_config = quantization_config

    def validate_environment(self, *args, **kwargs):
        if not is_accelerate_available():
            raise ImportError("Using `aqlm` quantization requires Accelerate: `pip install accelerate`")

        if not is_aqlm_available():
            raise ImportError("Using `aqlm` quantization requires AQLM: `pip install aqlm[gpu,cpu]`")

    def update_torch_dtype(self, torch_dtype: "torch.dtype") -> "torch.dtype":
        if torch_dtype is None:
            if torch.cuda.is_available():
                torch_dtype = torch.float16
                logger.info(
                    "CUDA available. Assuming AQLM inference on GPU and loading the model in `torch.float16`. To overwrite it, set `torch_dtype` manually."
                )
            else:
                torch_dtype = torch.float32
                logger.info(
                    "CUDA is unavailable. Assuming AQLM inference on CPU and loading the model in `torch.float32`. To overwrite it, set `torch_dtype` manually."
                )
        return torch_dtype

    def _process_model_before_weight_loading(
        self,
        model: "PreTrainedModel",
        **kwargs,
    ):
        replace_with_aqlm_linear(
            model,
            quantization_config=self.quantization_config,
            modules_to_not_convert=self.quantization_config.modules_to_not_convert,
        )
        model.config.quantization_config = self.quantization_config

    def _process_model_after_weight_loading(self, model: "PreTrainedModel", **kwargs):
        return model

    @property
    def is_trainable(self, model: Optional["PreTrainedModel"] = None):
        aqlm_supports_training = version.parse(importlib.metadata.version("aqlm")) >= version.parse("1.0.2")
        if aqlm_supports_training:
            return True
        else:
            logger.warning(
                f"Currently installed `aqlm` version ({importlib.metadata.version('aqlm')}) doesn't support training. If you wish to train a quantized model, please update `aqlm` with `pip install aqlm>=1.0.2`"
            )
            return False

    def is_serializable(self, safe_serialization=None):
        return True
