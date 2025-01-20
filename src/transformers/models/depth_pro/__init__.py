# Copyright 2024 The HuggingFace Team. All rights reserved.
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

from ...file_utils import _LazyModule, is_torch_available, is_vision_available
from ...utils import OptionalDependencyNotAvailable


_import_structure = {"configuration_depth_pro": ["DepthProConfig"]}

try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["image_processing_depth_pro"] = ["DepthProImageProcessor"]
    _import_structure["image_processing_depth_pro_fast"] = ["DepthProImageProcessorFast"]

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_depth_pro"] = [
        "DepthProForDepthEstimation",
        "DepthProModel",
        "DepthProPreTrainedModel",
    ]


if TYPE_CHECKING:
    from .configuration_depth_pro import DepthProConfig

    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .image_processing_depth_pro import DepthProImageProcessor
        from .image_processing_depth_pro_fast import DepthProImageProcessorFast

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_depth_pro import (
            DepthProForDepthEstimation,
            DepthProModel,
            DepthProPreTrainedModel,
        )


else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
