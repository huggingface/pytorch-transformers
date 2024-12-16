# coding=utf-8
# Copyright 2024 HuggingFace Inc.
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

import tempfile
import unittest

import numpy as np

from transformers.testing_utils import require_torch, require_torchvision, require_vision
from transformers.utils import is_torch_available, is_vision_available


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image

    from transformers import TimmWrapperConfig, TimmWrapperImageProcessor


@require_torch
@require_vision
@require_torchvision
class TimmWrapperImageProcessingTest(unittest.TestCase):
    image_processing_class = TimmWrapperImageProcessor if is_vision_available() else None

    def setUp(self):
        super().setUp()
        self.temp_dir = tempfile.TemporaryDirectory()
        config = TimmWrapperConfig.from_pretrained("timm/resnet18.a1_in1k")
        config.save_pretrained(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_load_from_hub(self):
        image_processor = TimmWrapperImageProcessor.from_pretrained("timm/resnet18.a1_in1k")
        self.assertIsInstance(image_processor, TimmWrapperImageProcessor)

    def test_load_from_local_dir(self):
        image_processor = TimmWrapperImageProcessor.from_pretrained(self.temp_dir.name)
        self.assertIsInstance(image_processor, TimmWrapperImageProcessor)

    def test_image_processor_properties(self):
        image_processor = TimmWrapperImageProcessor.from_pretrained(self.temp_dir.name)
        self.assertTrue(hasattr(image_processor, "data_config"))
        self.assertTrue(hasattr(image_processor, "val_transforms"))
        self.assertTrue(hasattr(image_processor, "train_transforms"))

    def test_image_processor_call_numpy(self):
        image_processor = TimmWrapperImageProcessor.from_pretrained(self.temp_dir.name)

        single_image = np.random.randint(256, size=(256, 256, 3), dtype=np.uint8)
        batch_images = [single_image, single_image, single_image]

        # single image
        pixel_values = image_processor(single_image).pixel_values
        self.assertEqual(pixel_values.shape, (1, 3, 224, 224))

        # batch images
        pixel_values = image_processor(batch_images).pixel_values
        self.assertEqual(pixel_values.shape, (3, 3, 224, 224))

    def test_image_processor_call_pil(self):
        image_processor = TimmWrapperImageProcessor.from_pretrained(self.temp_dir.name)

        single_image = Image.fromarray(np.random.randint(256, size=(256, 256, 3), dtype=np.uint8))
        batch_images = [single_image, single_image, single_image]

        # single image
        pixel_values = image_processor(single_image).pixel_values
        self.assertEqual(pixel_values.shape, (1, 3, 224, 224))

        # batch images
        pixel_values = image_processor(batch_images).pixel_values
        self.assertEqual(pixel_values.shape, (3, 3, 224, 224))

    def test_image_processor_call_tensor(self):
        image_processor = TimmWrapperImageProcessor.from_pretrained(self.temp_dir.name)

        single_image = torch.from_numpy(np.random.randint(256, size=(3, 256, 256), dtype=np.uint8)).float()
        batch_images = [single_image, single_image, single_image]

        # single image
        pixel_values = image_processor(single_image).pixel_values
        self.assertEqual(pixel_values.shape, (1, 3, 224, 224))

        # batch images
        pixel_values = image_processor(batch_images).pixel_values
        self.assertEqual(pixel_values.shape, (3, 3, 224, 224))
