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

import numpy as np

from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_vision_available

from ...test_image_processing_common import (
    ImageProcessingTestMixin,
    prepare_image_inputs,
)


if is_torch_available():
    import torch

    from transformers.models.superpoint.modeling_superpoint import SuperPointKeypointDescriptionOutput

if is_vision_available():
    from transformers import SuperPointImageProcessor


class SuperPointImageProcessingTester:
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
    ):
        size = size if size is not None else {"height": 480, "width": 640}
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_resize = do_resize
        self.size = size

    def prepare_image_processor_dict(self):
        return {
            "do_resize": self.do_resize,
            "size": self.size,
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

    def prepare_keypoint_detection_output(self, pixel_values):
        max_number_keypoints = 50
        batch_size = len(pixel_values)
        mask = torch.zeros((batch_size, max_number_keypoints))
        keypoints = torch.zeros((batch_size, max_number_keypoints, 2))
        scores = torch.zeros((batch_size, max_number_keypoints))
        descriptors = torch.zeros((batch_size, max_number_keypoints, 16))
        for i in range(batch_size):
            random_number_keypoints = np.random.randint(0, max_number_keypoints)
            mask[i, :random_number_keypoints] = 1
            keypoints[i, :random_number_keypoints] = torch.rand((random_number_keypoints, 2))
            scores[i, :random_number_keypoints] = torch.rand((random_number_keypoints,))
            descriptors[i, :random_number_keypoints] = torch.rand((random_number_keypoints, 16))
        return SuperPointKeypointDescriptionOutput(
            loss=None, keypoints=keypoints, scores=scores, descriptors=descriptors, mask=mask, hidden_states=None
        )


@require_torch
@require_vision
class SuperPointImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_class = SuperPointImageProcessor if is_vision_available() else None

    def setUp(self) -> None:
        super().setUp()
        self.image_processor_tester = SuperPointImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processing(self):
        image_processing = self.image_processing_class(**self.image_processor_dict)
        self.assertTrue(hasattr(image_processing, "do_resize"))
        self.assertTrue(hasattr(image_processing, "size"))
        self.assertTrue(hasattr(image_processing, "do_rescale"))
        self.assertTrue(hasattr(image_processing, "rescale_factor"))

    def test_image_processor_from_dict_with_kwargs(self):
        image_processor = self.image_processing_class.from_dict(self.image_processor_dict)
        self.assertEqual(image_processor.size, {"height": 480, "width": 640})

        image_processor = self.image_processing_class.from_dict(
            self.image_processor_dict, size={"height": 42, "width": 42}
        )
        self.assertEqual(image_processor.size, {"height": 42, "width": 42})

    @unittest.skip(reason="SuperPointImageProcessor is always supposed to return a grayscaled image")
    def test_call_numpy_4_channels(self):
        pass

    def test_input_image_properly_converted_to_grayscale(self):
        image_processor = self.image_processing_class.from_dict(self.image_processor_dict)
        image_inputs = self.image_processor_tester.prepare_image_inputs()
        pre_processed_images = image_processor.preprocess(image_inputs)
        for image in pre_processed_images["pixel_values"]:
            self.assertTrue(np.all(image[0, ...] == image[1, ...]) and np.all(image[1, ...] == image[2, ...]))

    @require_torch
    def test_post_processing_keypoint_detection(self):
        image_processor = self.image_processing_class.from_dict(self.image_processor_dict)
        image_inputs = self.image_processor_tester.prepare_image_inputs()
        pre_processed_images = image_processor.preprocess(image_inputs, return_tensors="pt")
        outputs = self.image_processor_tester.prepare_keypoint_detection_output(**pre_processed_images)

        def check_post_processed_output(post_processed_output, image_size):
            for post_processed_output, image_size in zip(post_processed_output, image_size):
                self.assertTrue("keypoints" in post_processed_output)
                self.assertTrue("descriptors" in post_processed_output)
                self.assertTrue("scores" in post_processed_output)
                keypoints = post_processed_output["keypoints"]
                all_below_image_size = torch.all(keypoints[:, 0] <= image_size[1]) and torch.all(
                    keypoints[:, 1] <= image_size[0]
                )
                all_above_zero = torch.all(keypoints[:, 0] >= 0) and torch.all(keypoints[:, 1] >= 0)
                self.assertTrue(all_below_image_size)
                self.assertTrue(all_above_zero)

        tuple_image_sizes = [(image.size[0], image.size[1]) for image in image_inputs]
        tuple_post_processed_outputs = image_processor.post_process_keypoint_detection(outputs, tuple_image_sizes)

        check_post_processed_output(tuple_post_processed_outputs, tuple_image_sizes)

        tensor_image_sizes = torch.tensor([image.size for image in image_inputs]).flip(1)
        tensor_post_processed_outputs = image_processor.post_process_keypoint_detection(outputs, tensor_image_sizes)

        check_post_processed_output(tensor_post_processed_outputs, tensor_image_sizes)
