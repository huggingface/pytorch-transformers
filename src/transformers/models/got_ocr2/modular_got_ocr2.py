# coding=utf-8
# Copyright 2024 HuggingFace Inc. team. All rights reserved.
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

import torch.nn as nn

from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM, Qwen2Model
from transformers.models.sam.modeling_sam import SamVisionEncoder


class GotOcr2Config(Qwen2Config):
    pass


class GotOcr2VisionEncoder(SamVisionEncoder):
    pass


class GotOcr2Model(Qwen2Model):
    def __init__(self, config: GotOcr2Config):
        super().__init__(config)

        self.vision_tower_high = GotOcr2VisionEncoder

        self.mm_projector_vary = nn.Linear(1024, 1024)


class GotOcr2ForCausalLM(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = GotOcr2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
