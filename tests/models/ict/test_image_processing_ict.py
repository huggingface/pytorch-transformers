# coding=utf-8
# Copyright 2023 HuggingFace Inc.
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

import json
import os
import tempfile
import unittest

import numpy as np
from datasets import load_dataset


from transformers.testing_utils import check_json_file_has_correct_format, require_torch, require_vision, slow
from transformers.utils import is_torch_available, is_vision_available

from ...test_image_processing_common import ImageProcessingSavingTestMixin, prepare_image_inputs


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image

    from transformers import IctImageProcessor


class IctImageProcessingTester(unittest.TestCase):
    def __init__(
        self,
        parent,
        batch_size=7,
        num_channels=3,
        image_size=18,
        min_resolution=30,
        max_resolution=400,
        do_resize=True,
        size=None,
        do_normalize=True,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
    ):
        size = size if size is not None else {"height": 18, "width": 18}
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_resize = do_resize
        self.size = size
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std

    def prepare_image_processor_dict(self):
        return {
            # here we create 2 clusters for the sake of simplicity
            "clusters": np.asarray([[241., 212., 177.], [ 50., 125., 197.]]),
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "do_normalize": self.do_normalize,
            "do_resize": self.do_resize,
            "size": self.size,
        }


@require_torch
@require_vision
class IctImageProcessingTest(ImageProcessingSavingTestMixin, unittest.TestCase):
    image_processing_class = IctImageProcessor if is_vision_available() else None

    def setUp(self):
        self.image_processor_tester = IctImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        image_processing = self.image_processing_class(**self.image_processor_dict)
        self.assertTrue(hasattr(image_processing, "clusters"))
        self.assertTrue(hasattr(image_processing, "image_mean"))
        self.assertTrue(hasattr(image_processing, "image_std"))
        self.assertTrue(hasattr(image_processing, "do_normalize"))
        self.assertTrue(hasattr(image_processing, "do_resize"))
        self.assertTrue(hasattr(image_processing, "size"))

    def test_image_processor_from_dict_with_kwargs(self):
        image_processor = self.image_processing_class.from_dict(self.image_processor_dict)
        self.assertEqual(image_processor.size, {"height": 18, "width": 18})

        image_processor = self.image_processing_class.from_dict(self.image_processor_dict, size=42)
        self.assertEqual(image_processor.size, {"height": 42, "width": 42})

    def test_image_processor_to_json_file(self):
        image_processor_first = self.image_processing_class(**self.image_processor_dict)

        with tempfile.TemporaryDirectory() as tmpdirname:
            json_file_path = os.path.join(tmpdirname, "image_processor.json")
            image_processor_first.to_json_file(json_file_path)
            image_processor_second = self.image_processing_class.from_json_file(json_file_path).to_dict()

        image_processor_first = image_processor_first.to_dict()
        for key, value in image_processor_first.items():
            if key == "clusters":
                self.assertTrue(np.array_equal(value, image_processor_second[key]))
            else:
                self.assertEqual(image_processor_first[key], value)

    def test_image_processor_to_json_string(self):
        image_processor = self.image_processing_class(**self.image_processor_dict)
        obj = json.loads(image_processor.to_json_string())
        for key, value in self.image_processor_dict.items():
            if key == "clusters":
                self.assertTrue(np.array_equal(value, obj[key]))
            else:
                self.assertEqual(obj[key], value)
                
    def test_image_processor_from_and_save_pretrained(self):
        image_processor_first = self.image_processing_class(**self.image_processor_dict)

        with tempfile.TemporaryDirectory() as tmpdirname:
            saved_file = image_processor_first.save_pretrained(tmpdirname)[0]
            check_json_file_has_correct_format(saved_file)
            image_processor_second = self.image_processing_class.from_pretrained(tmpdirname).to_dict()

        image_processor_first = image_processor_first.to_dict()
        for key, value in image_processor_first.items():
            if key == "clusters":
                self.assertTrue(np.array_equal(value, image_processor_second[key]))
            else:
                self.assertEqual(image_processor_first[key], value)

    def test_batch_feature(self):
        pass

    def test_call_pil(self):
        # Initialize image_processing
        image_processing = self.image_processing_class(**self.image_processor_dict)
        # create random PIL images
        image_inputs = prepare_image_inputs(self.image_processor_tester, equal_resolution=False)
        for image in image_inputs:
            self.assertIsInstance(image, Image.Image)

        # Test not batched input
        encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values
        self.assertEqual(
            encoded_images.shape,
            (
                1,
                self.image_processor_tester.size["height"] * self.image_processor_tester.size["width"],
            ),
        )
        # Test batched
        encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
        self.assertEqual(
            encoded_images.shape,
            (
                self.image_processor_tester.batch_size,
                self.image_processor_tester.size["height"] * self.image_processor_tester.size["width"],
            ),
        )

    def test_call_numpy(self):
        # Initialize image_processing
        image_processing = self.image_processing_class(**self.image_processor_dict)
        # create random numpy tensors
        image_inputs = prepare_image_inputs(self.image_processor_tester, equal_resolution=False, numpify=True)
        for image in image_inputs:
            self.assertIsInstance(image, np.ndarray)

        # Test not batched input
        encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values
        self.assertEqual(
            encoded_images.shape,
            (
                1,
                self.image_processor_tester.size["height"] * self.image_processor_tester.size["width"],
            ),
        )

        # Test batched
        encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
        self.assertEqual(
            encoded_images.shape,
            (
                self.image_processor_tester.batch_size,
                self.image_processor_tester.size["height"] * self.image_processor_tester.size["width"],
            ),
        )

    def test_call_pytorch(self):
        # Initialize image_processing
        image_processing = self.image_processing_class(**self.image_processor_dict)
        # create random PyTorch tensors
        image_inputs = prepare_image_inputs(self.image_processor_tester, equal_resolution=False, torchify=True)
        for image in image_inputs:
            self.assertIsInstance(image, torch.Tensor)

        # Test not batched input
        encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values
        self.assertEqual(
            encoded_images.shape,
            (
                1,
                self.image_processor_tester.size["height"] * self.image_processor_tester.size["width"],
            ),
        )

        # Test batched
        encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
        self.assertEqual(
            encoded_images.shape,
            (
                self.image_processor_tester.batch_size,
                self.image_processor_tester.size["height"] * self.image_processor_tester.size["width"],
            ),
        )

def prepare_images():
    dataset = load_dataset("hf-internal-testing/fixtures_image_utils", split="test")

    image1 = Image.open(dataset[4]["file"])
    image2 = Image.open(dataset[5]["file"])

    images = [image1, image2]

    return images

@require_vision
@require_torch
class IctImageProcessorIntegrationTest(unittest.TestCase):
    @slow
    def test_image(self):
        image_processing = IctImageProcessor.from_pretrained("sheonhan/ict-imagenet-256")

        images = prepare_images()

        # test non-batched
        encoding = image_processing(images[0], return_tensors="pt")

        self.assertIsInstance(encoding.pixel_values, torch.LongTensor)
        self.assertEqual(encoding.pixel_values.shape, (1, 1024))

        expected_slice = [291, 145, 48]
        self.assertEqual(encoding.pixel_values[0, :3].tolist(), expected_slice)

        # test batched
        encoding = image_processing(images, return_tensors="pt")

        self.assertIsInstance(encoding.pixel_values, torch.LongTensor)
        self.assertEqual(encoding.pixel_values.shape, (2, 1024))

        expected_slice = [228, 315, 375]
        self.assertEqual(encoding.pixel_values[1, -3:].tolist(), expected_slice)
