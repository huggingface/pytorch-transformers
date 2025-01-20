# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
import math
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ...generation import GenerationMixin
from ...generation.logits_process import (
    AlternatingCodebooksLogitsProcessor,
    StyleTextToSpeech2EosPrioritizerLogitsProcessor,
    SuppressTokensLogitsProcessor,
)
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import CausalLMOutputWithPast, MaskedLMOutput
from ...modeling_utils import PreTrainedModel, get_parameter_device
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_accelerate_available,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
)
from ..auto import AutoModel
from .configuration_style_text_to_speech_2 import (
    StyleTextToSpeech2CoarseConfig,
    StyleTextToSpeech2Config,
    StyleTextToSpeech2FineConfig,
    StyleTextToSpeech2SemanticConfig,
    StyleTextToSpeech2SubModelConfig,
)
from .generation_configuration_style_text_to_speech_2 import (
    StyleTextToSpeech2CoarseGenerationConfig,
    StyleTextToSpeech2FineGenerationConfig,
    StyleTextToSpeech2SemanticGenerationConfig,
)


if is_flash_attn_2_available():
    from ...modeling_flash_attention_utils import _flash_attention_forward


logger = logging.get_logger(__name__)


_CHECKPOINT_FOR_DOC = "hexgrad/Kokoro-82M"
_CONFIG_FOR_DOC = "StyleTextToSpeech2Config"
