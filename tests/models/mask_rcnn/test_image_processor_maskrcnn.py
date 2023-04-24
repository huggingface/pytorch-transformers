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
import unittest

import numpy as np

from transformers.testing_utils import require_torch, require_vision, slow
from transformers.utils import is_torch_available, is_vision_available

from ...test_image_processing_common import ImageProcessingSavingTestMixin, prepare_image_inputs


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image

    from transformers import MaskRCNNImageProcessor


class MaskRCNNImageProcessingTester(unittest.TestCase):
    def __init__(
        self,
        parent,
        batch_size=7,
        num_channels=3,
        min_resolution=30,
        max_resolution=400,
        do_resize=True,
        size=None,
        do_rescale=True,
        rescale_factor=1 / 255,
        do_normalize=True,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        do_pad=True,
    ):
        # by setting size["longest_edge"] > max_resolution we're effectively not testing this :p
        size = size if size is not None else {"shortest_edge": 18, "longest_edge": 1333}
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_resize = do_resize
        self.size = size
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.do_pad = do_pad

    def prepare_image_processor_dict(self):
        return {
            "do_resize": self.do_resize,
            "size": self.size,
            "do_rescale": self.do_rescale,
            "rescale_factor": self.rescale_factor,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "do_pad": self.do_pad,
        }

    def get_expected_values(self, image_inputs, batched=False):
        """
        This function computes the expected height and width when providing images to MaskRCNNImageProcessor,
        assuming do_resize is set to True with a scalar size.
        """
        if not batched:
            image = image_inputs[0]
            if isinstance(image, Image.Image):
                w, h = image.size
            else:
                h, w = image.shape[1], image.shape[2]
            if w < h:
                expected_height = int(self.size["shortest_edge"] * h / w)
                expected_width = self.size["shortest_edge"]
            elif w > h:
                expected_height = self.size["shortest_edge"]
                expected_width = int(self.size["shortest_edge"] * w / h)
            else:
                expected_height = self.size["shortest_edge"]
                expected_width = self.size["shortest_edge"]

        else:
            expected_values = []
            for image in image_inputs:
                expected_height, expected_width = self.get_expected_values([image])
                expected_values.append((expected_height, expected_width))
            expected_height = max(expected_values, key=lambda item: item[0])[0]
            expected_width = max(expected_values, key=lambda item: item[1])[1]

        return expected_height, expected_width


@require_torch
@require_vision
class MaskRCNNImageProcessingTest(ImageProcessingSavingTestMixin, unittest.TestCase):
    image_processing_class = MaskRCNNImageProcessor if is_vision_available() else None

    def setUp(self):
        self.image_processor_tester = MaskRCNNImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        image_processing = self.image_processing_class(**self.image_processor_dict)
        self.assertTrue(hasattr(image_processing, "image_mean"))
        self.assertTrue(hasattr(image_processing, "image_std"))
        self.assertTrue(hasattr(image_processing, "do_normalize"))
        self.assertTrue(hasattr(image_processing, "do_rescale"))
        self.assertTrue(hasattr(image_processing, "rescale_factor"))
        self.assertTrue(hasattr(image_processing, "do_resize"))
        self.assertTrue(hasattr(image_processing, "size"))
        self.assertTrue(hasattr(image_processing, "do_pad"))

    def test_image_processor_from_dict_with_kwargs(self):
        image_processor = self.image_processing_class.from_dict(self.image_processor_dict)
        self.assertEqual(image_processor.size, {"shortest_edge": 18, "longest_edge": 1333})
        self.assertEqual(image_processor.do_pad, True)

        image_processor = self.image_processing_class.from_dict(self.image_processor_dict, size=42, max_size=84)
        self.assertEqual(image_processor.size, {"shortest_edge": 42, "longest_edge": 84})
        self.assertEqual(image_processor.do_pad, False)

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

        expected_height, expected_width = self.image_processor_tester.get_expected_values(image_inputs)

        self.assertEqual(
            encoded_images.shape,
            (1, self.image_processor_tester.num_channels, expected_height, expected_width),
        )

        # Test batched
        expected_height, expected_width = self.image_processor_tester.get_expected_values(image_inputs, batched=True)

        encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
        self.assertEqual(
            encoded_images.shape,
            (
                self.image_processor_tester.batch_size,
                self.image_processor_tester.num_channels,
                expected_height,
                expected_width,
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

        expected_height, expected_width = self.image_processor_tester.get_expected_values(image_inputs)

        self.assertEqual(
            encoded_images.shape,
            (1, self.image_processor_tester.num_channels, expected_height, expected_width),
        )

        # Test batched
        encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values

        expected_height, expected_width = self.image_processor_tester.get_expected_values(image_inputs, batched=True)

        self.assertEqual(
            encoded_images.shape,
            (
                self.image_processor_tester.batch_size,
                self.image_processor_tester.num_channels,
                expected_height,
                expected_width,
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

        expected_height, expected_width = self.image_processor_tester.get_expected_values(image_inputs)

        self.assertEqual(
            encoded_images.shape,
            (1, self.image_processor_tester.num_channels, expected_height, expected_width),
        )

        # Test batched
        encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values

        expected_height, expected_width = self.image_processor_tester.get_expected_values(image_inputs, batched=True)

        self.assertEqual(
            encoded_images.shape,
            (
                self.image_processor_tester.batch_size,
                self.image_processor_tester.num_channels,
                expected_height,
                expected_width,
            ),
        )

    @slow
    def test_call_pytorch_with_coco_detection_annotations(self):
        # prepare image and target
        image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
        with open("./tests/fixtures/tests_samples/COCO/coco_annotations.txt", "r") as f:
            target = json.loads(f.read())

        target = {"image_id": 39769, "annotations": target}

        # encode them
        # TODO use from_pretrained here
        image_processing = MaskRCNNImageProcessor()
        encoding = image_processing(images=image, annotations=target, return_tensors="pt")

        # verify pixel values
        expected_shape = torch.Size([1, 3, 800, 1066])
        self.assertEqual(encoding["pixel_values"].shape, expected_shape)

        expected_slice = torch.tensor([0.2796, 0.3138, 0.3481])
        self.assertTrue(torch.allclose(encoding["pixel_values"][0, 0, 0, :3], expected_slice, atol=1e-4))

        # verify area
        expected_area = torch.tensor([5887.9600, 11250.2061, 489353.8438, 837122.7500, 147967.5156, 165732.3438])
        self.assertTrue(torch.allclose(encoding["labels"][0]["area"], expected_area))
        # verify boxes
        expected_boxes_shape = torch.Size([6, 4])
        self.assertEqual(encoding["labels"][0]["boxes"].shape, expected_boxes_shape)
        expected_boxes_slice = torch.tensor([554.4699, 132.6167, 618.8463, 309.8000])
        self.assertTrue(torch.allclose(encoding["labels"][0]["boxes"][0], expected_boxes_slice, atol=1e-3))
        # verify image_id
        expected_image_id = torch.tensor([39769])
        self.assertTrue(torch.allclose(encoding["labels"][0]["image_id"], expected_image_id))
        # verify is_crowd
        expected_is_crowd = torch.tensor([0, 0, 0, 0, 0, 0])
        self.assertTrue(torch.allclose(encoding["labels"][0]["iscrowd"], expected_is_crowd))
        # verify class_labels
        expected_class_labels = torch.tensor([75, 75, 63, 65, 17, 17])
        self.assertTrue(torch.allclose(encoding["labels"][0]["class_labels"], expected_class_labels))
        # verify orig_size
        expected_orig_size = torch.tensor([480, 640])
        self.assertTrue(torch.allclose(encoding["labels"][0]["orig_size"], expected_orig_size))
        # verify size
        expected_size = torch.tensor([800, 1066])
        self.assertTrue(torch.allclose(encoding["labels"][0]["size"], expected_size))
