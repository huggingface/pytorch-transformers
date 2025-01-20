# coding=utf-8
# Copyright 2022 HuggingFace Inc.
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

import numpy as np
from datasets import load_dataset

from transformers.file_utils import is_torch_available, is_vision_available
from transformers.testing_utils import require_torch, require_vision

from ...test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image

    from transformers import DPTImageProcessor


class DPTImageProcessingTester(unittest.TestCase):
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
        do_reduce_labels=False,
    ):
        super().__init__()
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
        self.do_reduce_labels = do_reduce_labels

    def prepare_image_processor_dict(self):
        return {
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "do_normalize": self.do_normalize,
            "do_resize": self.do_resize,
            "size": self.size,
            "do_reduce_labels": self.do_reduce_labels,
        }

    def expected_output_image_shape(self, images):
        return self.num_channels, self.size["height"], self.size["width"]

    def prepare_image_inputs(self, equal_resolution=False, numpify=False, torchify=False):
        return prepare_image_inputs(
            batch_size=self.batch_size,
            num_channels=self.num_channels,
            min_resolution=self.min_resolution,
            max_resolution=self.max_resolution,
            equal_resolution=equal_resolution,
            numpify=numpify,
            torchify=torchify,
        )


# Copied from transformers.tests.models.beit.test_image_processing_beit.prepare_semantic_single_inputs
def prepare_semantic_single_inputs():
    dataset = load_dataset("hf-internal-testing/fixtures_ade20k", split="test", trust_remote_code=True)

    image = Image.open(dataset[0]["file"])
    map = Image.open(dataset[1]["file"])

    return image, map


# Copied from transformers.tests.models.beit.test_image_processing_beit.prepare_semantic_batch_inputs
def prepare_semantic_batch_inputs():
    ds = load_dataset("hf-internal-testing/fixtures_ade20k", split="test", trust_remote_code=True)

    image1 = Image.open(ds[0]["file"])
    map1 = Image.open(ds[1]["file"])
    image2 = Image.open(ds[2]["file"])
    map2 = Image.open(ds[3]["file"])

    return [image1, image2], [map1, map2]


@require_torch
@require_vision
class DPTImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_class = DPTImageProcessor if is_vision_available() else None

    def setUp(self):
        super().setUp()
        self.image_processor_tester = DPTImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        image_processing = self.image_processing_class(**self.image_processor_dict)
        self.assertTrue(hasattr(image_processing, "image_mean"))
        self.assertTrue(hasattr(image_processing, "image_std"))
        self.assertTrue(hasattr(image_processing, "do_normalize"))
        self.assertTrue(hasattr(image_processing, "do_resize"))
        self.assertTrue(hasattr(image_processing, "size"))
        self.assertTrue(hasattr(image_processing, "do_rescale"))
        self.assertTrue(hasattr(image_processing, "rescale_factor"))
        self.assertTrue(hasattr(image_processing, "do_pad"))
        self.assertTrue(hasattr(image_processing, "size_divisor"))
        self.assertTrue(hasattr(image_processing, "do_reduce_labels"))

    def test_image_processor_from_dict_with_kwargs(self):
        image_processor = self.image_processing_class.from_dict(self.image_processor_dict)
        self.assertEqual(image_processor.size, {"height": 18, "width": 18})

        image_processor = self.image_processing_class.from_dict(self.image_processor_dict, size=42)
        self.assertEqual(image_processor.size, {"height": 42, "width": 42})

    def test_padding(self):
        image_processing = self.image_processing_class(**self.image_processor_dict)
        image = np.random.randn(3, 249, 491)

        # test individual method
        image = image_processing.pad_image(image, size_divisor=4)
        self.assertTrue(image.shape[1] % 4 == 0)
        self.assertTrue(image.shape[2] % 4 == 0)

        # test by calling
        pixel_values = image_processing.preprocess(
            image, do_rescale=False, do_resize=False, do_pad=True, size_divisor=4, return_tensors="pt"
        ).pixel_values
        self.assertTrue(pixel_values.shape[2] % 4 == 0)
        self.assertTrue(pixel_values.shape[3] % 4 == 0)

    def test_keep_aspect_ratio(self):
        size = {"height": 512, "width": 512}
        image_processor = DPTImageProcessor(size=size, keep_aspect_ratio=True, ensure_multiple_of=32)

        image = np.zeros((489, 640, 3))

        pixel_values = image_processor(image, return_tensors="pt").pixel_values

        self.assertEqual(list(pixel_values.shape), [1, 3, 512, 672])

    # Copied from transformers.tests.models.beit.test_image_processing_beit.BeitImageProcessingTest.test_call_segmentation_maps
    def test_call_segmentation_maps(self):
        # Initialize image_processor
        image_processor = self.image_processing_class(**self.image_processor_dict)
        # create random PyTorch tensors
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, torchify=True)
        maps = []
        for image in image_inputs:
            self.assertIsInstance(image, torch.Tensor)
            maps.append(torch.zeros(image.shape[-2:]).long())

        # Test not batched input
        encoding = image_processor(image_inputs[0], maps[0], return_tensors="pt")
        self.assertEqual(
            encoding["pixel_values"].shape,
            (
                1,
                self.image_processor_tester.num_channels,
                self.image_processor_tester.size["height"],
                self.image_processor_tester.size["width"],
            ),
        )
        self.assertEqual(
            encoding["labels"].shape,
            (
                1,
                self.image_processor_tester.size["height"],
                self.image_processor_tester.size["width"],
            ),
        )
        self.assertEqual(encoding["labels"].dtype, torch.long)
        self.assertTrue(encoding["labels"].min().item() >= 0)
        self.assertTrue(encoding["labels"].max().item() <= 255)

        # Test batched
        encoding = image_processor(image_inputs, maps, return_tensors="pt")
        self.assertEqual(
            encoding["pixel_values"].shape,
            (
                self.image_processor_tester.batch_size,
                self.image_processor_tester.num_channels,
                self.image_processor_tester.size["height"],
                self.image_processor_tester.size["width"],
            ),
        )
        self.assertEqual(
            encoding["labels"].shape,
            (
                self.image_processor_tester.batch_size,
                self.image_processor_tester.size["height"],
                self.image_processor_tester.size["width"],
            ),
        )
        self.assertEqual(encoding["labels"].dtype, torch.long)
        self.assertTrue(encoding["labels"].min().item() >= 0)
        self.assertTrue(encoding["labels"].max().item() <= 255)

        # Test not batched input (PIL images)
        image, segmentation_map = prepare_semantic_single_inputs()

        encoding = image_processor(image, segmentation_map, return_tensors="pt")
        self.assertEqual(
            encoding["pixel_values"].shape,
            (
                1,
                self.image_processor_tester.num_channels,
                self.image_processor_tester.size["height"],
                self.image_processor_tester.size["width"],
            ),
        )
        self.assertEqual(
            encoding["labels"].shape,
            (
                1,
                self.image_processor_tester.size["height"],
                self.image_processor_tester.size["width"],
            ),
        )
        self.assertEqual(encoding["labels"].dtype, torch.long)
        self.assertTrue(encoding["labels"].min().item() >= 0)
        self.assertTrue(encoding["labels"].max().item() <= 255)

        # Test batched input (PIL images)
        images, segmentation_maps = prepare_semantic_batch_inputs()

        encoding = image_processor(images, segmentation_maps, return_tensors="pt")
        self.assertEqual(
            encoding["pixel_values"].shape,
            (
                2,
                self.image_processor_tester.num_channels,
                self.image_processor_tester.size["height"],
                self.image_processor_tester.size["width"],
            ),
        )
        self.assertEqual(
            encoding["labels"].shape,
            (
                2,
                self.image_processor_tester.size["height"],
                self.image_processor_tester.size["width"],
            ),
        )
        self.assertEqual(encoding["labels"].dtype, torch.long)
        self.assertTrue(encoding["labels"].min().item() >= 0)
        self.assertTrue(encoding["labels"].max().item() <= 255)

    # Copied from transformers.tests.models.beit.test_image_processing_beit.BeitImageProcessingTest.test_reduce_labels
    def test_reduce_labels(self):
        # Initialize image_processor
        image_processor = self.image_processing_class(**self.image_processor_dict)

        # ADE20k has 150 classes, and the background is included, so labels should be between 0 and 150
        image, map = prepare_semantic_single_inputs()
        encoding = image_processor(image, map, return_tensors="pt")
        self.assertTrue(encoding["labels"].min().item() >= 0)
        self.assertTrue(encoding["labels"].max().item() <= 150)

        image_processor.do_reduce_labels = True
        encoding = image_processor(image, map, return_tensors="pt")
        self.assertTrue(encoding["labels"].min().item() >= 0)
        self.assertTrue(encoding["labels"].max().item() <= 255)
