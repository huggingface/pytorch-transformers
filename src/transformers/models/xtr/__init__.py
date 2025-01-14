# Copyright 2020 The HuggingFace Team. All rights reserved.
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
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)


_import_structure = {"configuration_t5": ["T5Config", "T5OnnxConfig"]}

try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_t5"] = ["T5Tokenizer"]

try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_t5_fast"] = ["T5TokenizerFast"]

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_t5"] = [
        "T5EncoderModel",
        "T5ForConditionalGeneration",
        "T5Model",
        "T5PreTrainedModel",
        "load_tf_weights_in_t5",
        "T5ForQuestionAnswering",
        "T5ForSequenceClassification",
        "T5ForTokenClassification",
    ]

try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_tf_t5"] = [
        "TFT5EncoderModel",
        "TFT5ForConditionalGeneration",
        "TFT5Model",
        "TFT5PreTrainedModel",
    ]

try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_flax_t5"] = [
        "FlaxT5EncoderModel",
        "FlaxT5ForConditionalGeneration",
        "FlaxT5Model",
        "FlaxT5PreTrainedModel",
    ]


if TYPE_CHECKING:
    from .configuration_t5 import T5Config, T5OnnxConfig

    try:
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_t5 import T5Tokenizer

    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_t5_fast import T5TokenizerFast

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_t5 import (
            T5EncoderModel,
            T5ForConditionalGeneration,
            T5ForQuestionAnswering,
            T5ForSequenceClassification,
            T5ForTokenClassification,
            T5Model,
            T5PreTrainedModel,
            load_tf_weights_in_t5,
        )

    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_tf_t5 import (
            TFT5EncoderModel,
            TFT5ForConditionalGeneration,
            TFT5Model,
            TFT5PreTrainedModel,
        )

    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_flax_t5 import (
            FlaxT5EncoderModel,
            FlaxT5ForConditionalGeneration,
            FlaxT5Model,
            FlaxT5PreTrainedModel,
        )


else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
