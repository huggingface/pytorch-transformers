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

import unittest
import requests

from transformers import (
    Florence2Config,
    Florence2VisionConfig,
)
from transformers.testing_utils import require_torch, require_vision, slow, torch_device
from transformers.utils import is_torch_available, is_vision_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    ModelTesterMixin,
)


if is_torch_available():
    import torch

    from transformers import (
        Florence2ForConditionalGeneration,
        Florence2VisionModel,
        Florence2VisionModelWithProjection,
    )
else:
    torch = {}

if is_vision_available():
    from PIL import Image

    from transformers import Florence2Processor

MODEL_ID = "microsoft/Florence-2-base-ft"


class Florence2ForConditionalGenerationTester:
    def __init__(self, parent):
        pass

    def prepare_config_and_inputs_for_common(self):
        pass


@require_torch
class Florence2ForConditionalGenerationTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (Florence2ForConditionalGeneration,) if is_torch_available() else ()

    def setUp(self):
        self.model_tester = Florence2ForConditionalGenerationTester(self)
        self.config_tester = ConfigTester(self, config_class=Florence2Config)

    @slow
    def test_model_from_pretrained(self):
        model = Florence2ForConditionalGeneration.from_pretrained(MODEL_ID)
        self.assertIsNotNone(model)


class Florence2VisionModelTester:
    def __init__(self, parent):
        pass

    def prepare_config_and_inputs_for_common(self):
        pass


@require_torch
class Florence2VisionModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (Florence2VisionModel,) if is_torch_available() else ()

    def setUp(self):
        self.model_tester = Florence2VisionModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Florence2VisionConfig)

    @slow
    def test_model_from_pretrained(self):
        model = Florence2VisionModel.from_pretrained(MODEL_ID)
        self.assertIsNotNone(model)


class Florence2VisionModelWithProjectionTester:
    def __init__(self, parent):
        pass

    def prepare_config_and_inputs_for_common(self):
        pass


@require_torch
class Florence2VisionModelWithProjectionTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (Florence2VisionModelWithProjection,) if is_torch_available() else ()

    def setUp(self):
        self.model_tester = Florence2VisionModelWithProjectionTester(self)
        self.config_tester = ConfigTester(self, config_class=Florence2VisionConfig)

    @slow
    def test_model_from_pretrained(self):
        model = Florence2VisionModelWithProjection.from_pretrained(MODEL_ID)
        self.assertIsNotNone(model)


def prepare_img():
    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


@require_vision
@require_torch
class Florence2ForConditionalGenerationIntegrationTest(unittest.TestCase):
    @slow
    def test_inference(self):
        model = Florence2ForConditionalGeneration.from_pretrained(MODEL_ID).to(torch_device)
        processor = Florence2Processor.from_pretrained(MODEL_ID)

        img = prepare_img()
        inputs = processor(img, return_tensors="pt").to(torch_device)

        with torch.no_grad():
            outputs = model(**inputs)

        # TODO: write test condition
        self.assertIsNotNone(outputs)


@require_vision
@require_torch
class Florence2VisionModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference(self):
        model = Florence2VisionModel.from_pretrained(MODEL_ID).to(torch_device)
        processor = Florence2Processor.from_pretrained(MODEL_ID)

        img = prepare_img()
        inputs = processor(img, return_tensors="pt").to(torch_device)

        with torch.no_grad():
            outputs = model(**inputs)

        # TODO: write test condition
        self.assertIsNotNone(outputs)


@require_vision
@require_torch
class Florence2VisionModelWithProjectionIntegrationTest(unittest.TestCase):
    @slow
    def test_inference(self):
        model = Florence2VisionModelWithProjection.from_pretrained(MODEL_ID).to(torch_device)
        processor = Florence2Processor.from_pretrained(MODEL_ID)

        img = prepare_img()
        inputs = processor(img, return_tensors="pt").to(torch_device)

        with torch.no_grad():
            outputs = model(**inputs)

        # TODO: write test condition
        self.assertIsNotNone(outputs)

