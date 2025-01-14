# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the S-Lab License, Version 1.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/sczhou/ProPainter/blob/main/LICENSE
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
    is_torch_available,
    is_torchvision_available,
)


_import_structure = {
    "configuration_propainter": ["ProPainterConfig"],
    "processing_propainter": ["ProPainterProcessor"],
}

try:
    if not is_torchvision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["video_processing_propainter"] = ["ProPainterVideoProcessor"]

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_propainter"] = [
        "ProPainterModel",
        "ProPainterPreTrainedModel",
    ]

if TYPE_CHECKING:
    from .configuration_propainter import ProPainterConfig
    from .processing_propainter import ProPainterProcessor

    try:
        if not is_torchvision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .video_processing_propainter import ProPainterVideoProcessor

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_propainter import (
            ProPainterModel,
            ProPainterPreTrainedModel,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
