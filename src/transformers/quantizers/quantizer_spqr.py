# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/lic enses/LICENSE-2.0
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

from ..integrations import replace_with_spqr_linear
from ..utils import is_accelerate_available, is_spqr_available, is_torch_available, logging
from ..utils.quantization_config import QuantizationConfigMixin


if is_torch_available():
    import torch

logger = logging.get_logger(__name__)


class SpQRHfQuantizer(HfQuantizer):
    """
    Quantizer of the SpQR method. Enables the loading of prequantized models.
    """

    def __init__(self, quantization_config: QuantizationConfigMixin, **kwargs):
        super().__init__(quantization_config, **kwargs)
        self.quantization_config = quantization_config

    def validate_environment(self, *args, **kwargs):
        if not torch.cuda.is_available():
            raise RuntimeError("GPU is required to run SpQR quantized model.")

        if not is_accelerate_available():
            raise ImportError("Using `spqr` quantization requires Accelerate: `pip install accelerate`")

        if not is_spqr_available():
            raise ImportError("Using `spqr` quantization requires SpQR: `pip install spqr_quant[gpu]`")

    def update_torch_dtype(self, torch_dtype: "torch.dtype") -> "torch.dtype":
        if torch_dtype is None:
            torch_dtype = torch.float16
            logger.info("Assuming SpQR inference on GPU and loading the model in `torch.float16`.")
        elif torch_dtype != torch.float16:
            raise ValueError(
                "You cannot use any type other than torch.float16 for SpQR. Please either leave it None or set it to"
                "torch.float16 explicitly."
            )
        return torch_dtype

    def _process_model_before_weight_loading(
        self,
        model: "PreTrainedModel",
        **kwargs,
    ):
        replace_with_spqr_linear(
            model,
            quantization_config=self.quantization_config,
            modules_to_not_convert=self.quantization_config.modules_to_not_convert,
        )
        model.config.quantization_config = self.quantization_config

    def _process_model_after_weight_loading(self, model: "PreTrainedModel", **kwargs):
        return model

    @property
    def is_trainable(self, model: Optional["PreTrainedModel"] = None):
        return False

    def is_serializable(self, safe_serialization=None):
        return True
