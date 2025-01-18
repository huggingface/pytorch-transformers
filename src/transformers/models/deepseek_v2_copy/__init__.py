# Copyright 2024 DeepSeek-AI and the HuggingFace Inc. team. All rights reserved.
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
from typing import TYPE_CHECKING

from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_sentencepiece_available,
    is_tokenizers_available,
    is_torch_available,
)


_import_structure = {
    "configuration_deepseek_v2": ["DeepseekV2Config"],
}

try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_deepseek_v2"] = ["DeepseekV2Tokenizer"]

try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_deepseek_v2_fast"] = ["DeepseekV2TokenizerFast"]

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_deepseek_v2"] = [
        "DeepseekV2ForCausalLM",
        "DeepseekV2Model",
        "DeepseekV2PreTrainedModel",
        "DeepseekV2ForSequenceClassification",
        "DeepseekV2ForQuestionAnswering",
        "DeepseekV2ForTokenClassification",
    ]

try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_flax_deepseek_v2"] = [
        "FlaxDeepseekV2ForCausalLM",
        "FlaxDeepseekV2Model",
        "FlaxDeepseekV2PreTrainedModel",
    ]


if TYPE_CHECKING:
    from .configuration_deepseek_v2 import DeepseekV2Config

    try:
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_deepseek_v2 import DeepseekV2Tokenizer

    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_deepseek_v2_fast import DeepseekV2TokenizerFast

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_deepseek_v2 import (
            DeepseekV2ForCausalLM,
            DeepseekV2ForQuestionAnswering,
            DeepseekV2ForSequenceClassification,
            DeepseekV2ForTokenClassification,
            DeepseekV2Model,
            DeepseekV2PreTrainedModel,
        )

    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_flax_deepseek_v2 import (
            FlaxDeepseekV2ForCausalLM,
            FlaxDeepseekV2Model,
            FlaxDeepseekV2PreTrainedModel,
        )


else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
