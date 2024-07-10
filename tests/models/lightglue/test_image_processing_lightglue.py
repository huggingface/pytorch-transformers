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

from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_vision_available

from ...test_image_processing_common import (
    ImageProcessingTestMixin,
    prepare_image_inputs,
)


if is_torch_available():
    import numpy as np

if is_vision_available():
    from transformers import LightGlueImageProcessor


class LightGlueImageProcessingTester(unittest.TestCase):
    def __init__(
        self,
        parent,
        batch_size=6,
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
        return 2, self.num_channels, self.size["height"], self.size["width"]

    def prepare_image_inputs(self, equal_resolution=False, numpify=False, torchify=False, pairs=True, batch_size=None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        image_inputs = prepare_image_inputs(
            batch_size=batch_size,
            num_channels=self.num_channels,
            min_resolution=self.min_resolution,
            max_resolution=self.max_resolution,
            equal_resolution=equal_resolution,
            numpify=numpify,
            torchify=torchify,
        )
        if pairs:
            image_inputs = [image_inputs[i : i + 2] for i in range(0, len(image_inputs), 2)]
        return image_inputs


@require_torch
@require_vision
class LightGlueImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_class = LightGlueImageProcessor if is_vision_available() else None

    def setUp(self) -> None:
        super().setUp()
        self.image_processor_tester = LightGlueImageProcessingTester(self)

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

    def test_input_images_properly_paired(self):
        image_processor = self.image_processing_class.from_dict(self.image_processor_dict)
        image_inputs = self.image_processor_tester.prepare_image_inputs()
        pre_processed_images = image_processor.preprocess(image_inputs, return_tensors="np")
        self.assertEqual(len(pre_processed_images["pixel_values"].shape), 5)
        self.assertEqual(pre_processed_images["pixel_values"].shape[1], 2)

    def test_input_not_paired_images_raises_error(self):
        image_processor = self.image_processing_class.from_dict(self.image_processor_dict)
        image_inputs = self.image_processor_tester.prepare_image_inputs(pairs=False)
        with self.assertRaises(ValueError):
            image_processor.preprocess(image_inputs[0])

    def test_input_image_properly_converted_to_grayscale(self):
        image_processor = self.image_processing_class.from_dict(self.image_processor_dict)
        image_inputs = self.image_processor_tester.prepare_image_inputs()
        pre_processed_images = image_processor.preprocess(image_inputs)
        for image_pair in pre_processed_images["pixel_values"]:
            for image in image_pair:
                self.assertTrue(np.all(image[0, ...] == image[1, ...]) and np.all(image[1, ...] == image[2, ...]))

    def test_call_numpy(self):
        # Test overwritten because LightGlueImageProcessor combines images by pair to feed it into LightGlue

        # Initialize image_processing
        image_processing = self.image_processing_class(**self.image_processor_dict)
        # create random numpy tensors
        image_pairs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, numpify=True)
        for image_pair in image_pairs:
            self.assertEqual(len(image_pair), 2)

        expected_batch_size = int(self.image_processor_tester.batch_size / 2)

        # Test with 2 images
        encoded_images = image_processing(image_pairs[0], return_tensors="pt").pixel_values
        expected_output_image_shape = self.image_processor_tester.expected_output_image_shape(image_pairs[0])
        self.assertEqual(tuple(encoded_images.shape), (1, *expected_output_image_shape))

        # Test with list of pairs
        encoded_images = image_processing(image_pairs, return_tensors="pt").pixel_values
        expected_output_image_shape = self.image_processor_tester.expected_output_image_shape(image_pairs)
        self.assertEqual(tuple(encoded_images.shape), (expected_batch_size, *expected_output_image_shape))

        # Test without paired images
        image_pairs = self.image_processor_tester.prepare_image_inputs(
            equal_resolution=False, numpify=True, pairs=False
        )
        with self.assertRaises(ValueError):
            image_processing(image_pairs, return_tensors="pt").pixel_values

    def test_call_pil(self):
        # Test overwritten because LightGlueImageProcessor combines images by pair to feed it into LightGlue

        # Initialize image_processing
        image_processing = self.image_processing_class(**self.image_processor_dict)
        # create random PIL images
        image_pairs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False)
        for image_pair in image_pairs:
            self.assertEqual(len(image_pair), 2)

        expected_batch_size = int(self.image_processor_tester.batch_size / 2)

        # Test with 2 images
        encoded_images = image_processing(image_pairs[0], return_tensors="pt").pixel_values
        expected_output_image_shape = self.image_processor_tester.expected_output_image_shape(image_pairs[0])
        self.assertEqual(tuple(encoded_images.shape), (1, *expected_output_image_shape))

        # Test with list of pairs
        encoded_images = image_processing(image_pairs, return_tensors="pt").pixel_values
        expected_output_image_shape = self.image_processor_tester.expected_output_image_shape(image_pairs)
        self.assertEqual(tuple(encoded_images.shape), (expected_batch_size, *expected_output_image_shape))

        # Test without paired images
        image_pairs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, pairs=False)
        with self.assertRaises(ValueError):
            image_processing(image_pairs, return_tensors="pt").pixel_values

    def test_call_pytorch(self):
        # Test overwritten because LightGlueImageProcessor combines images by pair to feed it into LightGlue

        # Initialize image_processing
        image_processing = self.image_processing_class(**self.image_processor_dict)
        # create random PyTorch tensors
        image_pairs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, torchify=True)
        for image_pair in image_pairs:
            self.assertEqual(len(image_pair), 2)

        expected_batch_size = int(self.image_processor_tester.batch_size / 2)

        # Test with 2 images
        encoded_images = image_processing(image_pairs[0], return_tensors="pt").pixel_values
        expected_output_image_shape = self.image_processor_tester.expected_output_image_shape(image_pairs[0])
        self.assertEqual(tuple(encoded_images.shape), (1, *expected_output_image_shape))

        # Test with list of pairs
        encoded_images = image_processing(image_pairs, return_tensors="pt").pixel_values
        expected_output_image_shape = self.image_processor_tester.expected_output_image_shape(image_pairs)
        self.assertEqual(tuple(encoded_images.shape), (expected_batch_size, *expected_output_image_shape))

        # Test without paired images
        image_pairs = self.image_processor_tester.prepare_image_inputs(
            equal_resolution=False, torchify=True, pairs=False
        )
        with self.assertRaises(ValueError):
            image_processing(image_pairs, return_tensors="pt").pixel_values

    def test_image_processor_with_list_of_two_images(self):
        image_processing = self.image_processing_class(**self.image_processor_dict)

        image_pairs = self.image_processor_tester.prepare_image_inputs(
            equal_resolution=False, numpify=True, batch_size=2, pairs=False
        )
        self.assertEqual(len(image_pairs), 2)
        self.assertTrue(isinstance(image_pairs[0], np.ndarray))
        self.assertTrue(isinstance(image_pairs[1], np.ndarray))

        expected_batch_size = 1
        encoded_images = image_processing(image_pairs, return_tensors="pt").pixel_values
        expected_output_image_shape = self.image_processor_tester.expected_output_image_shape(image_pairs[0])
        self.assertEqual(tuple(encoded_images.shape), (expected_batch_size, *expected_output_image_shape))

    def test_image_processor_padding(self):
        custom_image_processor_dict = self.image_processor_dict
        custom_image_processor_dict["do_resize"] = False
        image_processing = self.image_processing_class(**custom_image_processor_dict)
        image_pairs = self.image_processor_tester.prepare_image_inputs(
            equal_resolution=False, numpify=True, batch_size=2, pairs=False
        )
        encoded_images = image_processing(image_pairs, return_tensors="pt").pixel_values
        max_height = max(image.shape[0] for image in image_pairs)
        max_width = max(image.shape[1] for image in image_pairs)
        expected_output_image_shape = (2, 3, max_height, max_width)
        self.assertEqual(tuple(encoded_images.shape), (1, *expected_output_image_shape))
