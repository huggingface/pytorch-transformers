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
    import torch

    from transformers.models.lightglue.modeling_lightglue import LightGlueKeypointMatchingOutput

if is_vision_available():
    from transformers import LightGlueImageProcessor


class LightGlueImageProcessingTester(unittest.TestCase):
    # Copied from tests.models.superglue.test_image_processing_superglue.SuperGlueImageProcessingTester.__init__
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
        do_grayscale=True,
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
        self.do_grayscale = do_grayscale

    # Copied from tests.models.superglue.test_image_processing_superglue.SuperGlueImageProcessingTester.prepare_image_processor_dict
    def prepare_image_processor_dict(self):
        return {
            "do_resize": self.do_resize,
            "size": self.size,
            "do_grayscale": self.do_grayscale,
        }

    # Copied from tests.models.superglue.test_image_processing_superglue.SuperGlueImageProcessingTester.expected_output_image_shape
    def expected_output_image_shape(self, images):
        return 2, self.num_channels, self.size["height"], self.size["width"]

    # Copied from tests.models.superglue.test_image_processing_superglue.SuperGlueImageProcessingTester.prepare_image_inputs
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

    def prepare_keypoint_matching_output(self, pixel_values):
        max_number_keypoints = 50
        batch_size = len(pixel_values)
        mask = torch.zeros((batch_size, 2, max_number_keypoints), dtype=torch.int)
        keypoints = torch.zeros((batch_size, 2, max_number_keypoints, 2))
        matches = torch.full((batch_size, 2, max_number_keypoints), -1, dtype=torch.int)
        scores = torch.zeros((batch_size, 2, max_number_keypoints))
        prune = torch.zeros((batch_size, 2, max_number_keypoints), dtype=torch.int)
        for i in range(batch_size):
            random_number_keypoints0 = np.random.randint(10, max_number_keypoints)
            random_number_keypoints1 = np.random.randint(10, max_number_keypoints)
            random_number_matches = np.random.randint(5, min(random_number_keypoints0, random_number_keypoints1))
            mask[i, 0, :random_number_keypoints0] = 1
            mask[i, 1, :random_number_keypoints1] = 1
            keypoints[i, 0, :random_number_keypoints0] = torch.rand((random_number_keypoints0, 2))
            keypoints[i, 1, :random_number_keypoints1] = torch.rand((random_number_keypoints1, 2))
            random_matches_indices0 = torch.randperm(random_number_keypoints1, dtype=torch.int)[:random_number_matches]
            random_matches_indices1 = torch.randperm(random_number_keypoints0, dtype=torch.int)[:random_number_matches]
            matches[i, 0, random_matches_indices1] = random_matches_indices0
            matches[i, 1, random_matches_indices0] = random_matches_indices1
            scores[i, 0, random_matches_indices1] = torch.rand((random_number_matches,))
            scores[i, 1, random_matches_indices0] = torch.rand((random_number_matches,))
        return LightGlueKeypointMatchingOutput(
            mask=mask, keypoints=keypoints, matches=matches, matching_scores=scores, prune=prune
        )


@require_torch
@require_vision
# Copied from tests.models.superglue.test_image_processing_superglue.SuperGlueImageProcessingTest with SuperGlue->LightGlue
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
        self.assertTrue(hasattr(image_processing, "do_grayscale"))

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

    def test_number_and_format_of_images_in_input(self):
        image_processor = self.image_processing_class.from_dict(self.image_processor_dict)

        # Cases where the number of images and the format of lists in the input is correct
        image_input = self.image_processor_tester.prepare_image_inputs(pairs=False, batch_size=2)
        image_processed = image_processor.preprocess(image_input, return_tensors="pt")
        self.assertEqual((1, 2, 3, 480, 640), tuple(image_processed["pixel_values"].shape))

        image_input = self.image_processor_tester.prepare_image_inputs(pairs=True, batch_size=2)
        image_processed = image_processor.preprocess(image_input, return_tensors="pt")
        self.assertEqual((1, 2, 3, 480, 640), tuple(image_processed["pixel_values"].shape))

        image_input = self.image_processor_tester.prepare_image_inputs(pairs=True, batch_size=4)
        image_processed = image_processor.preprocess(image_input, return_tensors="pt")
        self.assertEqual((2, 2, 3, 480, 640), tuple(image_processed["pixel_values"].shape))

        image_input = self.image_processor_tester.prepare_image_inputs(pairs=True, batch_size=6)
        image_processed = image_processor.preprocess(image_input, return_tensors="pt")
        self.assertEqual((3, 2, 3, 480, 640), tuple(image_processed["pixel_values"].shape))

        # Cases where the number of images or the format of lists in the input is incorrect
        ## List of 4 images
        image_input = self.image_processor_tester.prepare_image_inputs(pairs=False, batch_size=4)
        with self.assertRaises(ValueError) as cm:
            image_processor.preprocess(image_input, return_tensors="pt")
        self.assertEqual(ValueError, cm.exception.__class__)

        ## List of 3 images
        image_input = self.image_processor_tester.prepare_image_inputs(pairs=False, batch_size=3)
        with self.assertRaises(ValueError) as cm:
            image_processor.preprocess(image_input, return_tensors="pt")
        self.assertEqual(ValueError, cm.exception.__class__)

        ## List of 2 pairs and 1 image
        image_input = self.image_processor_tester.prepare_image_inputs(pairs=True, batch_size=3)
        with self.assertRaises(ValueError) as cm:
            image_processor.preprocess(image_input, return_tensors="pt")
        self.assertEqual(ValueError, cm.exception.__class__)

    def test_image_shape_in_input(self):
        def random_array(size):
            return np.random.randint(255, size=size)

        def random_tensor(size):
            return torch.rand(size)

        image_processor = self.image_processing_class.from_dict(self.image_processor_dict)

        # Cases where the shape of the images in the input is correct
        ## 2 images with shape (3, 100, 200)
        image_input = [random_array((3, 100, 200)), random_array((3, 100, 200))]
        image_processed = image_processor.preprocess(image_input, return_tensors="pt")
        self.assertEqual((1, 2, 3, 480, 640), tuple(image_processed["pixel_values"].shape))

        ## 1 array with shape (2, 3, 100, 200)
        image_input = random_array((2, 3, 100, 200))
        image_processed = image_processor.preprocess(image_input, return_tensors="pt")
        self.assertEqual((1, 2, 3, 480, 640), tuple(image_processed["pixel_values"].shape))

        ## 1 list of 2 images with shape (3, 100, 200)
        image_input = [[random_array((3, 100, 200)), random_array((3, 100, 200))]]
        image_processed = image_processor.preprocess(image_input, return_tensors="pt")
        self.assertEqual((1, 2, 3, 480, 640), tuple(image_processed["pixel_values"].shape))

        ## 1 list of 1 array with shape (2, 3, 100, 200)
        image_input = [random_array((2, 3, 100, 200))]
        image_processed = image_processor.preprocess(image_input, return_tensors="pt")
        self.assertEqual((1, 2, 3, 480, 640), tuple(image_processed["pixel_values"].shape))

        ## 1 list of 3 array with shape (2, 3, 100, 200)
        image_input = [random_array((2, 3, 100, 200)), random_array((2, 3, 100, 200)), random_array((2, 3, 100, 200))]
        image_processed = image_processor.preprocess(image_input, return_tensors="pt")
        self.assertEqual((3, 2, 3, 480, 640), tuple(image_processed["pixel_values"].shape))

        ## 1 array with shape (1, 2, 3, 100, 200)
        image_input = random_array((1, 2, 3, 100, 200))
        image_processed = image_processor.preprocess(image_input, return_tensors="pt")
        self.assertEqual((1, 2, 3, 480, 640), tuple(image_processed["pixel_values"].shape))

        ## 1 array with shape (3, 2, 3, 100, 200)
        image_input = random_array((3, 2, 3, 100, 200))
        image_processed = image_processor.preprocess(image_input, return_tensors="pt")
        self.assertEqual((3, 2, 3, 480, 640), tuple(image_processed["pixel_values"].shape))

        ## 1 list of 2 tensors with shape (3, 100, 200)
        image_input = [random_tensor((3, 100, 200)), random_tensor((3, 100, 200))]
        image_processed = image_processor.preprocess(image_input, return_tensors="pt")
        self.assertEqual((1, 2, 3, 480, 640), tuple(image_processed["pixel_values"].shape))

        ## 1 tensor with shape (2, 3, 100, 200)
        image_input = random_tensor((2, 3, 100, 200))
        image_processed = image_processor.preprocess(image_input, return_tensors="pt")
        self.assertEqual((1, 2, 3, 480, 640), tuple(image_processed["pixel_values"].shape))

        ## 1 tensor with shape (1, 2, 3, 100, 200)
        image_input = random_tensor((1, 2, 3, 100, 200))
        image_processed = image_processor.preprocess(image_input, return_tensors="pt")
        self.assertEqual((1, 2, 3, 480, 640), tuple(image_processed["pixel_values"].shape))

        ## 1 tensor with shape (2, 2, 3, 100, 200)
        image_input = random_tensor((2, 2, 3, 100, 200))
        image_processed = image_processor.preprocess(image_input, return_tensors="pt")
        self.assertEqual((2, 2, 3, 480, 640), tuple(image_processed["pixel_values"].shape))

        # Cases where the shape of the images in the input is incorrect
        ## 1 image with shape (3, 100, 200)
        image_input = random_array((3, 100, 200))
        with self.assertRaises(ValueError) as cm:
            image_processor.preprocess(image_input, return_tensors="pt")
        self.assertEqual(ValueError, cm.exception.__class__)

        ## 1 list of 1 image with shape (3, 100, 200)
        image_input = [random_array((3, 100, 200))]
        with self.assertRaises(ValueError) as cm:
            image_processor.preprocess(image_input, return_tensors="pt")
        self.assertEqual(ValueError, cm.exception.__class__)

        ## 1 array with shape (1, 3, 100, 200)
        image_input = random_array((1, 3, 100, 200))
        with self.assertRaises(ValueError) as cm:
            image_processor.preprocess(image_input, return_tensors="pt")
        self.assertEqual(ValueError, cm.exception.__class__)

        ## 1 list of a list of 1 image (3, 100, 200)
        image_input = [[random_array((3, 100, 200))]]
        with self.assertRaises(ValueError) as cm:
            image_processor.preprocess(image_input, return_tensors="pt")
        self.assertEqual(ValueError, cm.exception.__class__)

        ## 1 list of 2 lists of 1 image (3, 100, 200)
        image_input = [[random_array((3, 100, 200))], [random_array((3, 100, 200))]]
        with self.assertRaises(ValueError) as cm:
            image_processor.preprocess(image_input, return_tensors="pt")
        self.assertEqual(ValueError, cm.exception.__class__)

        ## 1 list of 2 arrays with shape (1, 3, 100, 200)
        image_input = [random_array((1, 3, 100, 200)), random_array((1, 3, 100, 200))]
        with self.assertRaises(ValueError) as cm:
            image_processor.preprocess(image_input, return_tensors="pt")
        self.assertEqual(ValueError, cm.exception.__class__)

        ## 1 array with shape (1, 1, 3, 100, 200)
        image_input = random_array((1, 1, 3, 100, 200))
        with self.assertRaises(ValueError) as cm:
            image_processor.preprocess(image_input, return_tensors="pt")
        self.assertEqual(ValueError, cm.exception.__class__)

        ## 1 tensor with shape (3, 100, 200)
        image_input = random_tensor((3, 100, 200))
        with self.assertRaises(ValueError) as cm:
            image_processor.preprocess(image_input, return_tensors="pt")
        self.assertEqual(ValueError, cm.exception.__class__)

        ## 1 tensor with shape (1, 3, 100, 200)
        image_input = random_tensor((1, 3, 100, 200))
        with self.assertRaises(ValueError) as cm:
            image_processor.preprocess(image_input, return_tensors="pt")
        self.assertEqual(ValueError, cm.exception.__class__)

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

    @require_torch
    def test_post_processing_keypoint_matching(self):
        image_processor = self.image_processing_class.from_dict(self.image_processor_dict)
        image_inputs = self.image_processor_tester.prepare_image_inputs()
        pre_processed_images = image_processor.preprocess(image_inputs, return_tensors="pt")
        outputs = self.image_processor_tester.prepare_keypoint_matching_output(**pre_processed_images)

        def check_post_processed_output(post_processed_output, image_pair_size):
            for post_processed_output, (image_size0, image_size1) in zip(post_processed_output, image_pair_size):
                self.assertTrue("keypoints0" in post_processed_output)
                self.assertTrue("keypoints1" in post_processed_output)
                self.assertTrue("matching_scores" in post_processed_output)
                keypoints0 = post_processed_output["keypoints0"]
                keypoints1 = post_processed_output["keypoints1"]
                all_below_image_size0 = torch.all(keypoints0[:, 0] <= image_size0[1]) and torch.all(
                    keypoints0[:, 1] <= image_size0[0]
                )
                all_below_image_size1 = torch.all(keypoints1[:, 0] <= image_size1[1]) and torch.all(
                    keypoints1[:, 1] <= image_size1[0]
                )
                all_above_zero0 = torch.all(keypoints0[:, 0] >= 0) and torch.all(keypoints0[:, 1] >= 0)
                all_above_zero1 = torch.all(keypoints0[:, 0] >= 0) and torch.all(keypoints0[:, 1] >= 0)
                self.assertTrue(all_below_image_size0)
                self.assertTrue(all_below_image_size1)
                self.assertTrue(all_above_zero0)
                self.assertTrue(all_above_zero1)
                all_scores_different_from_minus_one = torch.all(post_processed_output["matching_scores"] != -1)
                self.assertTrue(all_scores_different_from_minus_one)

        tuple_image_sizes = [
            ((image_pair[0].size[0], image_pair[0].size[1]), (image_pair[1].size[0], image_pair[1].size[1]))
            for image_pair in image_inputs
        ]
        tuple_post_processed_outputs = image_processor.post_process_keypoint_matching(outputs, tuple_image_sizes)

        check_post_processed_output(tuple_post_processed_outputs, tuple_image_sizes)

        tensor_image_sizes = torch.tensor(
            [(image_pair[0].size, image_pair[1].size) for image_pair in image_inputs]
        ).flip(2)
        tensor_post_processed_outputs = image_processor.post_process_keypoint_matching(outputs, tensor_image_sizes)

        check_post_processed_output(tensor_post_processed_outputs, tensor_image_sizes)
