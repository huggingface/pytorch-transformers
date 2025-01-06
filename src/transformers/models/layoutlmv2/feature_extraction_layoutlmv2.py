# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
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
"""
Feature extractor class for LayoutLMv2.
"""

import warnings

from ...utils import logging
from .image_processing_layoutlmv2 import LayoutLMv2ImageProcessor


logger = logging.get_logger(__name__)


class LayoutLMv2FeatureExtractor(LayoutLMv2ImageProcessor):
    def __init__(self, *args, **kwargs) -> None:
        warnings.warn(
            "The class LayoutLMv2FeatureExtractor is deprecated and will be removed in version 5 of Transformers."
            " Please use LayoutLMv2ImageProcessor instead.",
            FutureWarning,
        )
        super().__init__(*args, **kwargs)


__all__ = ["LayoutLMv2FeatureExtractor"]
