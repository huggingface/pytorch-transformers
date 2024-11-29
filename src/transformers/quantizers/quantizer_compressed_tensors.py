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

from ..utils import is_compressed_tensors_available, is_torch_available, logging
from ..utils.quantization_config import QuantizationConfigMixin
from .base import HfQuantizer


if is_torch_available():
    import torch

logger = logging.get_logger(__name__)


class CompressedTensorsHfQuantizer(HfQuantizer):
    """
    Quantizer for the compressed_tensors package.  Loads and restores models to
    quantized state with compressed_tensors
    """

    requires_calibration = True
    required_packages = ["compressed_tensors"]

    def __init__(self, quantization_config: QuantizationConfigMixin, **kwargs):
        super().__init__(quantization_config, **kwargs)

        from compressed_tensors.compressors import ModelCompressor

        self.compressor = ModelCompressor.from_compression_config(quantization_config)

    def validate_environment(self, *args, **kwargs):
        if not is_compressed_tensors_available():
            raise ImportError(
                "Using `compressed_tensors` quantized models requires the compressed-tensors library: "
                "`pip install compressed-tensors`"
            )
        if not is_torch_available():
            # torch already should be installed as part of compressed tensors
            raise ImportError("torch is required for using compressed-tensors quantization")

    def update_torch_dtype(self, torch_dtype: "torch.dtype") -> "torch.dtype":
        if torch_dtype is None:
            logger.info("Loading model using torch.float16 for compressed-tensors quantization")
            torch_dtype = torch.float16
        elif torch_dtype != torch.float16:
            logger.info(
                "We suggest you to set `torch_dtype=torch.float16` for better efficiency with compressed_tensors."
            )
        return torch_dtype

    def _process_model_before_weight_loading(self, model, **kwargs):
        from compressed_tensors.quantization import apply_quantization_config

        ct_quantization_config = self.compressor.quantization_config
        apply_quantization_config(model, ct_quantization_config, run_compressed=True)

    def _process_model_after_weight_loading(self, model, **kwargs) -> None:
        pass

    @property
    def is_trainable(self) -> bool:
        """Models quantized using compressed tensors can be finetuned"""
        return True

    @property
    def is_qat_trainable(self) -> bool:
        """Loaded Models can carry out quantization aware training"""
        return True

    def is_serializable(self, safe_serialization=None) -> bool:
        """Models quantized using compressed tensors can be saved to disk"""
        return True
