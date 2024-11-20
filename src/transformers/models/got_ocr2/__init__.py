# Copyright 2024 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
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
    is_tokenizers_available,
    is_torch_available,
    is_vision_available,
)


_import_structure = {
    "configuration_got_ocr2": ["GotOcr2Config", "GotOcr2VisionConfig"],
    "processing_got_ocr2": ["GotOcr2Processor"],
}
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["image_processing_got_ocr2"] = ["GotOcr2ImageProcessor"]


try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_got_ocr2"] = [
        "GotOcr2ForConditionalGeneration",
        "GotOcr2Model",
        "GotOcr2PreTrainedModel",
    ]


if TYPE_CHECKING:
    from .configuration_got_ocr2 import GotOcr2Config, GotOcr2VisionConfig
    from .processing_got_ocr2 import GotOcr2Processor

    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .image_processing_got_ocr2 import GotOcr2ImageProcessor
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_got_ocr2 import (
            GotOcr2ForConditionalGeneration,
            GotOcr2Model,
            GotOcr2PreTrainedModel,
        )


else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
