# coding=utf-8
# Copyright 2021 HuggingFace Inc.
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

import codecs
import os
import tempfile
import unittest
from io import BytesIO
from typing import Optional

import numpy as np
import pytest
import requests
from huggingface_hub.file_download import hf_hub_url, http_get
from requests import ConnectTimeout, ReadTimeout

from tests.pipelines.test_pipelines_document_question_answering import INVOICE_URL
from transformers import is_torch_available, is_vision_available
from transformers.image_utils import (
    ChannelDimension,
    get_channel_dimension_axis,
    make_batched_videos,
    make_flat_list_of_images,
    make_list_of_images,
    make_nested_list_of_images,
)
from transformers.testing_utils import is_flaky, require_torch, require_vision


if is_torch_available():
    import torch

if is_vision_available():
    import PIL.Image

    from transformers import ImageFeatureExtractionMixin
    from transformers.image_utils import get_image_size, infer_channel_dimension_format, load_image


def get_image_from_hub_dataset(dataset_id: str, filename: str, revision: Optional[str] = None) -> "PIL.Image.Image":
    url = hf_hub_url(dataset_id, filename, repo_type="dataset", revision=revision)
    return PIL.Image.open(BytesIO(requests.get(url).content))


def get_random_image(height, width):
    random_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return PIL.Image.fromarray(random_array)


@require_vision
class ImageFeatureExtractionTester(unittest.TestCase):
    def test_conversion_image_to_array(self):
        feature_extractor = ImageFeatureExtractionMixin()
        image = get_random_image(16, 32)

        # Conversion with defaults (rescale + channel first)
        array1 = feature_extractor.to_numpy_array(image)
        self.assertTrue(array1.dtype, np.float32)
        self.assertEqual(array1.shape, (3, 16, 32))

        # Conversion with rescale and not channel first
        array2 = feature_extractor.to_numpy_array(image, channel_first=False)
        self.assertTrue(array2.dtype, np.float32)
        self.assertEqual(array2.shape, (16, 32, 3))
        self.assertTrue(np.array_equal(array1, array2.transpose(2, 0, 1)))

        # Conversion with no rescale and channel first
        array3 = feature_extractor.to_numpy_array(image, rescale=False)
        self.assertTrue(array3.dtype, np.uint8)
        self.assertEqual(array3.shape, (3, 16, 32))
        self.assertTrue(np.array_equal(array1, array3.astype(np.float32) * (1 / 255.0)))

        # Conversion with no rescale and not channel first
        array4 = feature_extractor.to_numpy_array(image, rescale=False, channel_first=False)
        self.assertTrue(array4.dtype, np.uint8)
        self.assertEqual(array4.shape, (16, 32, 3))
        self.assertTrue(np.array_equal(array2, array4.astype(np.float32) * (1 / 255.0)))

    def test_conversion_array_to_array(self):
        feature_extractor = ImageFeatureExtractionMixin()
        array = np.random.randint(0, 256, (16, 32, 3), dtype=np.uint8)

        # By default, rescale (for an array of ints) and channel permute
        array1 = feature_extractor.to_numpy_array(array)
        self.assertTrue(array1.dtype, np.float32)
        self.assertEqual(array1.shape, (3, 16, 32))
        self.assertTrue(np.array_equal(array1, array.transpose(2, 0, 1).astype(np.float32) * (1 / 255.0)))

        # Same with no permute
        array2 = feature_extractor.to_numpy_array(array, channel_first=False)
        self.assertTrue(array2.dtype, np.float32)
        self.assertEqual(array2.shape, (16, 32, 3))
        self.assertTrue(np.array_equal(array2, array.astype(np.float32) * (1 / 255.0)))

        # Force rescale to False
        array3 = feature_extractor.to_numpy_array(array, rescale=False)
        self.assertTrue(array3.dtype, np.uint8)
        self.assertEqual(array3.shape, (3, 16, 32))
        self.assertTrue(np.array_equal(array3, array.transpose(2, 0, 1)))

        # Force rescale to False and no channel permute
        array4 = feature_extractor.to_numpy_array(array, rescale=False, channel_first=False)
        self.assertTrue(array4.dtype, np.uint8)
        self.assertEqual(array4.shape, (16, 32, 3))
        self.assertTrue(np.array_equal(array4, array))

        # Now test the default rescale for a float array (defaults to False)
        array5 = feature_extractor.to_numpy_array(array2)
        self.assertTrue(array5.dtype, np.float32)
        self.assertEqual(array5.shape, (3, 16, 32))
        self.assertTrue(np.array_equal(array5, array1))

    def test_make_list_of_images_pil(self):
        # Test a single image is converted to a list of 1 image
        pil_image = get_random_image(16, 32)
        images_list = make_list_of_images(pil_image)
        self.assertIsInstance(images_list, list)
        self.assertEqual(len(images_list), 1)
        self.assertIsInstance(images_list[0], PIL.Image.Image)

        # Test a list of images is not modified
        images = [get_random_image(16, 32) for _ in range(4)]
        images_list = make_list_of_images(images)
        self.assertIsInstance(images_list, list)
        self.assertEqual(len(images_list), 4)
        self.assertIsInstance(images_list[0], PIL.Image.Image)

    def test_make_list_of_images_numpy(self):
        # Test a single image is converted to a list of 1 image
        images = np.random.randint(0, 256, (16, 32, 3))
        images_list = make_list_of_images(images)
        self.assertEqual(len(images_list), 1)
        self.assertTrue(np.array_equal(images_list[0], images))
        self.assertIsInstance(images_list, list)

        # Test a batch of images is converted to a list of images
        images = np.random.randint(0, 256, (4, 16, 32, 3))
        images_list = make_list_of_images(images)
        self.assertEqual(len(images_list), 4)
        self.assertTrue(np.array_equal(images_list[0], images[0]))
        self.assertIsInstance(images_list, list)

        # Test a list of images is not modified
        images = [np.random.randint(0, 256, (16, 32, 3)) for _ in range(4)]
        images_list = make_list_of_images(images)
        self.assertEqual(len(images_list), 4)
        self.assertTrue(np.array_equal(images_list[0], images[0]))
        self.assertIsInstance(images_list, list)

        # Test batched masks with no channel dimension are converted to a list of masks
        masks = np.random.randint(0, 2, (4, 16, 32))
        masks_list = make_list_of_images(masks, expected_ndims=2)
        self.assertEqual(len(masks_list), 4)
        self.assertTrue(np.array_equal(masks_list[0], masks[0]))
        self.assertIsInstance(masks_list, list)

    @require_torch
    def test_make_list_of_images_torch(self):
        # Test a single image is converted to a list of 1 image
        images = torch.randint(0, 256, (16, 32, 3))
        images_list = make_list_of_images(images)
        self.assertEqual(len(images_list), 1)
        self.assertTrue(np.array_equal(images_list[0], images))
        self.assertIsInstance(images_list, list)

        # Test a batch of images is converted to a list of images
        images = torch.randint(0, 256, (4, 16, 32, 3))
        images_list = make_list_of_images(images)
        self.assertEqual(len(images_list), 4)
        self.assertTrue(np.array_equal(images_list[0], images[0]))
        self.assertIsInstance(images_list, list)

        # Test a list of images is left unchanged
        images = [torch.randint(0, 256, (16, 32, 3)) for _ in range(4)]
        images_list = make_list_of_images(images)
        self.assertEqual(len(images_list), 4)
        self.assertTrue(np.array_equal(images_list[0], images[0]))
        self.assertIsInstance(images_list, list)

    def test_make_flat_list_of_images_pil(self):
        # Test a single image is converted to a list of 1 image
        pil_image = get_random_image(16, 32)
        images_list = make_flat_list_of_images(pil_image)
        self.assertIsInstance(images_list, list)
        self.assertEqual(len(images_list), 1)
        self.assertIsInstance(images_list[0], PIL.Image.Image)

        # Test a list of images is not modified
        images = [get_random_image(16, 32) for _ in range(4)]
        images_list = make_flat_list_of_images(images)
        self.assertIsInstance(images_list, list)
        self.assertEqual(len(images_list), 4)
        self.assertIsInstance(images_list[0], PIL.Image.Image)

        # Test a nested list of images is flattened
        images = [[get_random_image(16, 32) for _ in range(2)] for _ in range(2)]
        images_list = make_flat_list_of_images(images)
        self.assertIsInstance(images_list, list)
        self.assertEqual(len(images_list), 4)
        self.assertIsInstance(images_list[0], PIL.Image.Image)

    def test_make_flat_list_of_images_numpy(self):
        # Test a single image is converted to a list of 1 image
        images = np.random.randint(0, 256, (16, 32, 3))
        images_list = make_flat_list_of_images(images)
        self.assertEqual(len(images_list), 1)
        self.assertTrue(np.array_equal(images_list[0], images))
        self.assertIsInstance(images_list, list)

        # Test a 4d array of images is changed to a list of images
        images = np.random.randint(0, 256, (4, 16, 32, 3))
        images_list = make_flat_list_of_images(images)
        self.assertEqual(len(images_list), 4)
        self.assertIsInstance(images_list, list)
        self.assertIsInstance(images_list[0], np.ndarray)
        self.assertTrue(np.array_equal(images_list[0], images[0]))

        # Test a list of images is not modified
        images = [np.random.randint(0, 256, (16, 32, 3)) for _ in range(4)]
        images_list = make_flat_list_of_images(images)
        self.assertEqual(len(images_list), 4)
        self.assertTrue(np.array_equal(images_list[0], images[0]))
        self.assertIsInstance(images_list, list)

        # Test list of 4d array images is flattened
        images = [np.random.randint(0, 256, (4, 16, 32, 3)) for _ in range(2)]
        images_list = make_flat_list_of_images(images)
        self.assertEqual(len(images_list), 8)
        self.assertTrue(np.array_equal(images_list[0], images[0][0]))
        self.assertIsInstance(images_list, list)
        self.assertIsInstance(images_list[0], np.ndarray)

        # Test nested list of images is flattened
        images = [[np.random.randint(0, 256, (16, 32, 3)) for _ in range(2)] for _ in range(2)]
        images_list = make_flat_list_of_images(images)
        self.assertEqual(len(images_list), 4)
        self.assertTrue(np.array_equal(images_list[0], images[0][0]))
        self.assertIsInstance(images_list, list)

    @require_torch
    def test_make_flat_list_of_images_torch(self):
        # Test a single image is converted to a list of 1 image
        images = torch.randint(0, 256, (16, 32, 3))
        images_list = make_flat_list_of_images(images)
        self.assertEqual(len(images_list), 1)
        self.assertTrue(np.array_equal(images_list[0], images))
        self.assertIsInstance(images_list, list)

        # Test a 4d tensors of images is changed to a list of images
        images = torch.randint(0, 256, (4, 16, 32, 3))
        images_list = make_flat_list_of_images(images)
        self.assertEqual(len(images_list), 4)
        self.assertIsInstance(images_list, list)
        self.assertIsInstance(images_list[0], torch.Tensor)
        self.assertTrue(np.array_equal(images_list[0], images[0]))

        # Test a list of images is not modified
        images = [torch.randint(0, 256, (16, 32, 3)) for _ in range(4)]
        images_list = make_flat_list_of_images(images)
        self.assertEqual(len(images_list), 4)
        self.assertTrue(np.array_equal(images_list[0], images[0]))
        self.assertIsInstance(images_list, list)

        # Test list of 4d tensors of imagess is flattened
        images = [torch.randint(0, 256, (4, 16, 32, 3)) for _ in range(2)]
        images_list = make_flat_list_of_images(images)
        self.assertEqual(len(images_list), 8)
        self.assertTrue(np.array_equal(images_list[0], images[0][0]))
        self.assertIsInstance(images_list, list)
        self.assertIsInstance(images_list[0], torch.Tensor)

        # Test nested list of images is flattened
        images = [[torch.randint(0, 256, (16, 32, 3)) for _ in range(2)] for _ in range(2)]
        images_list = make_flat_list_of_images(images)
        self.assertEqual(len(images_list), 4)
        self.assertTrue(np.array_equal(images_list[0], images[0][0]))
        self.assertIsInstance(images_list, list)

    def test_make_nested_list_of_images_pil(self):
        # Test a single image is converted to a nested list of 1 image
        pil_image = get_random_image(16, 32)
        images_list = make_nested_list_of_images(pil_image)
        self.assertIsInstance(images_list[0], list)
        self.assertEqual(len(images_list[0]), 1)
        self.assertIsInstance(images_list[0][0], PIL.Image.Image)

        # Test a list of images is converted to a nested list of images
        images = [get_random_image(16, 32) for _ in range(4)]
        images_list = make_nested_list_of_images(images)
        self.assertIsInstance(images_list[0], list)
        self.assertEqual(len(images_list), 1)
        self.assertEqual(len(images_list[0]), 4)
        self.assertIsInstance(images_list[0][0], PIL.Image.Image)

        # Test a nested list of images is not modified
        images = [[get_random_image(16, 32) for _ in range(2)] for _ in range(2)]
        images_list = make_nested_list_of_images(images)
        self.assertIsInstance(images_list[0], list)
        self.assertEqual(len(images_list), 2)
        self.assertEqual(len(images_list[0]), 2)
        self.assertIsInstance(images_list[0][0], PIL.Image.Image)

    def test_make_nested_list_of_images_numpy(self):
        # Test a single image is converted to a nested list of 1 image
        images = np.random.randint(0, 256, (16, 32, 3))
        images_list = make_nested_list_of_images(images)
        self.assertIsInstance(images_list[0], list)
        self.assertEqual(len(images_list), 1)
        self.assertTrue(np.array_equal(images_list[0][0], images))

        # Test a 4d array of images is converted to a nested list of images
        images = np.random.randint(0, 256, (4, 16, 32, 3))
        images_list = make_nested_list_of_images(images)
        self.assertIsInstance(images_list[0], list)
        self.assertIsInstance(images_list[0][0], np.ndarray)
        self.assertEqual(len(images_list), 1)
        self.assertEqual(len(images_list[0]), 4)
        self.assertTrue(np.array_equal(images_list[0][0], images[0]))

        # Test a list of images is converted to a nested list of images
        images = [np.random.randint(0, 256, (16, 32, 3)) for _ in range(4)]
        images_list = make_nested_list_of_images(images)
        self.assertIsInstance(images_list[0], list)
        self.assertEqual(len(images_list), 1)
        self.assertEqual(len(images_list[0]), 4)
        self.assertTrue(np.array_equal(images_list[0][0], images[0]))

        # Test a nested list of images is left unchanged
        images = [[np.random.randint(0, 256, (16, 32, 3)) for _ in range(2)] for _ in range(2)]
        images_list = make_nested_list_of_images(images)
        self.assertIsInstance(images_list[0], list)
        self.assertEqual(len(images_list), 2)
        self.assertEqual(len(images_list[0]), 2)
        self.assertTrue(np.array_equal(images_list[0][0], images[0][0]))

        # Test a list of 4d array images is converted to a nested list of images
        images = [np.random.randint(0, 256, (4, 16, 32, 3)) for _ in range(2)]
        images_list = make_nested_list_of_images(images)
        self.assertIsInstance(images_list[0], list)
        self.assertIsInstance(images_list[0][0], np.ndarray)
        self.assertEqual(len(images_list), 2)
        self.assertEqual(len(images_list[0]), 4)
        self.assertTrue(np.array_equal(images_list[0][0], images[0][0]))

    @require_torch
    def test_make_nested_list_of_images_torch(self):
        # Test a single image is converted to a nested list of 1 image
        images = torch.randint(0, 256, (16, 32, 3))
        images_list = make_nested_list_of_images(images)
        self.assertIsInstance(images_list[0], list)
        self.assertEqual(len(images_list[0]), 1)
        self.assertTrue(np.array_equal(images_list[0][0], images))

        # Test a 4d tensor of images is converted to a nested list of images
        images = torch.randint(0, 256, (4, 16, 32, 3))
        images_list = make_nested_list_of_images(images)
        self.assertIsInstance(images_list[0], list)
        self.assertIsInstance(images_list[0][0], torch.Tensor)
        self.assertEqual(len(images_list), 1)
        self.assertEqual(len(images_list[0]), 4)
        self.assertTrue(np.array_equal(images_list[0][0], images[0]))

        # Test a list of images is converted to a nested list of images
        images = [torch.randint(0, 256, (16, 32, 3)) for _ in range(4)]
        images_list = make_nested_list_of_images(images)
        self.assertIsInstance(images_list[0], list)
        self.assertEqual(len(images_list), 1)
        self.assertEqual(len(images_list[0]), 4)
        self.assertTrue(np.array_equal(images_list[0][0], images[0]))

        # Test a nested list of images is left unchanged
        images = [[torch.randint(0, 256, (16, 32, 3)) for _ in range(2)] for _ in range(2)]
        images_list = make_nested_list_of_images(images)
        self.assertIsInstance(images_list[0], list)
        self.assertEqual(len(images_list), 2)
        self.assertEqual(len(images_list[0]), 2)
        self.assertTrue(np.array_equal(images_list[0][0], images[0][0]))

        # Test a list of 4d tensor images is converted to a nested list of images
        images = [torch.randint(0, 256, (4, 16, 32, 3)) for _ in range(2)]
        images_list = make_nested_list_of_images(images)
        self.assertIsInstance(images_list[0], list)
        self.assertIsInstance(images_list[0][0], torch.Tensor)
        self.assertEqual(len(images_list), 2)
        self.assertEqual(len(images_list[0]), 4)
        self.assertTrue(np.array_equal(images_list[0][0], images[0][0]))

    def test_make_batched_videos_pil(self):
        # Test a single image is converted to a list of 1 video with 1 frame
        pil_image = get_random_image(16, 32)
        videos_list = make_batched_videos(pil_image)
        self.assertIsInstance(videos_list[0], list)
        self.assertEqual(len(videos_list[0]), 1)
        self.assertIsInstance(videos_list[0][0], PIL.Image.Image)

        # Test a list of images is converted to a list of 1 video
        images = [get_random_image(16, 32) for _ in range(4)]
        videos_list = make_batched_videos(images)
        self.assertIsInstance(videos_list[0], list)
        self.assertEqual(len(videos_list), 1)
        self.assertEqual(len(videos_list[0]), 4)
        self.assertIsInstance(videos_list[0][0], PIL.Image.Image)

        # Test a nested list of images is not modified
        images = [[get_random_image(16, 32) for _ in range(2)] for _ in range(2)]
        videos_list = make_nested_list_of_images(images)
        self.assertIsInstance(videos_list[0], list)
        self.assertEqual(len(videos_list), 2)
        self.assertEqual(len(videos_list[0]), 2)
        self.assertIsInstance(videos_list[0][0], PIL.Image.Image)

    def test_make_batched_videos_numpy(self):
        # Test a single image is converted to a list of 1 video with 1 frame
        images = np.random.randint(0, 256, (16, 32, 3))
        videos_list = make_nested_list_of_images(images)
        self.assertIsInstance(videos_list[0], list)
        self.assertEqual(len(videos_list), 1)
        self.assertTrue(np.array_equal(videos_list[0][0], images))

        # Test a 4d array of images is converted to a a list of 1 video
        images = np.random.randint(0, 256, (4, 16, 32, 3))
        videos_list = make_nested_list_of_images(images)
        self.assertIsInstance(videos_list[0], list)
        self.assertIsInstance(videos_list[0][0], np.ndarray)
        self.assertEqual(len(videos_list), 1)
        self.assertEqual(len(videos_list[0]), 4)
        self.assertTrue(np.array_equal(videos_list[0][0], images[0]))

        # Test a list of images is converted to a list of videos
        images = [np.random.randint(0, 256, (16, 32, 3)) for _ in range(4)]
        videos_list = make_nested_list_of_images(images)
        self.assertIsInstance(videos_list[0], list)
        self.assertEqual(len(videos_list), 1)
        self.assertEqual(len(videos_list[0]), 4)
        self.assertTrue(np.array_equal(videos_list[0][0], images[0]))

        # Test a nested list of images is left unchanged
        images = [[np.random.randint(0, 256, (16, 32, 3)) for _ in range(2)] for _ in range(2)]
        videos_list = make_nested_list_of_images(images)
        self.assertIsInstance(videos_list[0], list)
        self.assertEqual(len(videos_list), 2)
        self.assertEqual(len(videos_list[0]), 2)
        self.assertTrue(np.array_equal(videos_list[0][0], images[0][0]))

        # Test a list of 4d array images is converted to a list of videos
        images = [np.random.randint(0, 256, (4, 16, 32, 3)) for _ in range(2)]
        videos_list = make_nested_list_of_images(images)
        self.assertIsInstance(videos_list[0], list)
        self.assertIsInstance(videos_list[0][0], np.ndarray)
        self.assertEqual(len(videos_list), 2)
        self.assertEqual(len(videos_list[0]), 4)
        self.assertTrue(np.array_equal(videos_list[0][0], images[0][0]))

    @require_torch
    def test_make_batched_videos_torch(self):
        # Test a single image is converted to a list of 1 video with 1 frame
        images = torch.randint(0, 256, (16, 32, 3))
        videos_list = make_nested_list_of_images(images)
        self.assertIsInstance(videos_list[0], list)
        self.assertEqual(len(videos_list[0]), 1)
        self.assertTrue(np.array_equal(videos_list[0][0], images))

        # Test a 4d tensor of images is converted to a list of 1 video
        images = torch.randint(0, 256, (4, 16, 32, 3))
        videos_list = make_nested_list_of_images(images)
        self.assertIsInstance(videos_list[0], list)
        self.assertIsInstance(videos_list[0][0], torch.Tensor)
        self.assertEqual(len(videos_list), 1)
        self.assertEqual(len(videos_list[0]), 4)
        self.assertTrue(np.array_equal(videos_list[0][0], images[0]))

        # Test a list of images is converted to a list of videos
        images = [torch.randint(0, 256, (16, 32, 3)) for _ in range(4)]
        videos_list = make_nested_list_of_images(images)
        self.assertIsInstance(videos_list[0], list)
        self.assertEqual(len(videos_list), 1)
        self.assertEqual(len(videos_list[0]), 4)
        self.assertTrue(np.array_equal(videos_list[0][0], images[0]))

        # Test a nested list of images is left unchanged
        images = [[torch.randint(0, 256, (16, 32, 3)) for _ in range(2)] for _ in range(2)]
        videos_list = make_nested_list_of_images(images)
        self.assertIsInstance(videos_list[0], list)
        self.assertEqual(len(videos_list), 2)
        self.assertEqual(len(videos_list[0]), 2)
        self.assertTrue(np.array_equal(videos_list[0][0], images[0][0]))

        # Test a list of 4d tensor images is converted to a list of videos
        images = [torch.randint(0, 256, (4, 16, 32, 3)) for _ in range(2)]
        videos_list = make_nested_list_of_images(images)
        self.assertIsInstance(videos_list[0], list)
        self.assertIsInstance(videos_list[0][0], torch.Tensor)
        self.assertEqual(len(videos_list), 2)
        self.assertEqual(len(videos_list[0]), 4)
        self.assertTrue(np.array_equal(videos_list[0][0], images[0][0]))

    @require_torch
    def test_conversion_torch_to_array(self):
        feature_extractor = ImageFeatureExtractionMixin()
        tensor = torch.randint(0, 256, (16, 32, 3))
        array = tensor.numpy()

        # By default, rescale (for a tensor of ints) and channel permute
        array1 = feature_extractor.to_numpy_array(array)
        self.assertTrue(array1.dtype, np.float32)
        self.assertEqual(array1.shape, (3, 16, 32))
        self.assertTrue(np.array_equal(array1, array.transpose(2, 0, 1).astype(np.float32) * (1 / 255.0)))

        # Same with no permute
        array2 = feature_extractor.to_numpy_array(array, channel_first=False)
        self.assertTrue(array2.dtype, np.float32)
        self.assertEqual(array2.shape, (16, 32, 3))
        self.assertTrue(np.array_equal(array2, array.astype(np.float32) * (1 / 255.0)))

        # Force rescale to False
        array3 = feature_extractor.to_numpy_array(array, rescale=False)
        self.assertTrue(array3.dtype, np.uint8)
        self.assertEqual(array3.shape, (3, 16, 32))
        self.assertTrue(np.array_equal(array3, array.transpose(2, 0, 1)))

        # Force rescale to False and no channel permute
        array4 = feature_extractor.to_numpy_array(array, rescale=False, channel_first=False)
        self.assertTrue(array4.dtype, np.uint8)
        self.assertEqual(array4.shape, (16, 32, 3))
        self.assertTrue(np.array_equal(array4, array))

        # Now test the default rescale for a float tensor (defaults to False)
        array5 = feature_extractor.to_numpy_array(array2)
        self.assertTrue(array5.dtype, np.float32)
        self.assertEqual(array5.shape, (3, 16, 32))
        self.assertTrue(np.array_equal(array5, array1))

    def test_conversion_image_to_image(self):
        feature_extractor = ImageFeatureExtractionMixin()
        image = get_random_image(16, 32)

        # On an image, `to_pil_image1` is a noop.
        image1 = feature_extractor.to_pil_image(image)
        self.assertTrue(isinstance(image, PIL.Image.Image))
        self.assertTrue(np.array_equal(np.array(image), np.array(image1)))

    def test_conversion_array_to_image(self):
        feature_extractor = ImageFeatureExtractionMixin()
        array = np.random.randint(0, 256, (16, 32, 3), dtype=np.uint8)

        # By default, no rescale (for an array of ints)
        image1 = feature_extractor.to_pil_image(array)
        self.assertTrue(isinstance(image1, PIL.Image.Image))
        self.assertTrue(np.array_equal(np.array(image1), array))

        # If the array is channel-first, proper reordering of the channels is done.
        image2 = feature_extractor.to_pil_image(array.transpose(2, 0, 1))
        self.assertTrue(isinstance(image2, PIL.Image.Image))
        self.assertTrue(np.array_equal(np.array(image2), array))

        # If the array has floating type, it's rescaled by default.
        image3 = feature_extractor.to_pil_image(array.astype(np.float32) * (1 / 255.0))
        self.assertTrue(isinstance(image3, PIL.Image.Image))
        self.assertTrue(np.array_equal(np.array(image3), array))

        # You can override the default to rescale.
        image4 = feature_extractor.to_pil_image(array.astype(np.float32), rescale=False)
        self.assertTrue(isinstance(image4, PIL.Image.Image))
        self.assertTrue(np.array_equal(np.array(image4), array))

        # And with floats + channel first.
        image5 = feature_extractor.to_pil_image(array.transpose(2, 0, 1).astype(np.float32) * (1 / 255.0))
        self.assertTrue(isinstance(image5, PIL.Image.Image))
        self.assertTrue(np.array_equal(np.array(image5), array))

    @require_torch
    def test_conversion_tensor_to_image(self):
        feature_extractor = ImageFeatureExtractionMixin()
        tensor = torch.randint(0, 256, (16, 32, 3))
        array = tensor.numpy()

        # By default, no rescale (for a tensor of ints)
        image1 = feature_extractor.to_pil_image(tensor)
        self.assertTrue(isinstance(image1, PIL.Image.Image))
        self.assertTrue(np.array_equal(np.array(image1), array))

        # If the tensor is channel-first, proper reordering of the channels is done.
        image2 = feature_extractor.to_pil_image(tensor.permute(2, 0, 1))
        self.assertTrue(isinstance(image2, PIL.Image.Image))
        self.assertTrue(np.array_equal(np.array(image2), array))

        # If the tensor has floating type, it's rescaled by default.
        image3 = feature_extractor.to_pil_image(tensor.float() / 255.0)
        self.assertTrue(isinstance(image3, PIL.Image.Image))
        self.assertTrue(np.array_equal(np.array(image3), array))

        # You can override the default to rescale.
        image4 = feature_extractor.to_pil_image(tensor.float(), rescale=False)
        self.assertTrue(isinstance(image4, PIL.Image.Image))
        self.assertTrue(np.array_equal(np.array(image4), array))

        # And with floats + channel first.
        image5 = feature_extractor.to_pil_image(tensor.permute(2, 0, 1).float() * (1 / 255.0))
        self.assertTrue(isinstance(image5, PIL.Image.Image))
        self.assertTrue(np.array_equal(np.array(image5), array))

    def test_resize_image_and_array(self):
        feature_extractor = ImageFeatureExtractionMixin()
        image = get_random_image(16, 32)
        array = np.array(image)

        # Size can be an int or a tuple of ints.
        resized_image = feature_extractor.resize(image, 8)
        self.assertTrue(isinstance(resized_image, PIL.Image.Image))
        self.assertEqual(resized_image.size, (8, 8))

        resized_image1 = feature_extractor.resize(image, (8, 16))
        self.assertTrue(isinstance(resized_image1, PIL.Image.Image))
        self.assertEqual(resized_image1.size, (8, 16))

        # Passing an array converts it to a PIL Image.
        resized_image2 = feature_extractor.resize(array, 8)
        self.assertTrue(isinstance(resized_image2, PIL.Image.Image))
        self.assertEqual(resized_image2.size, (8, 8))
        self.assertTrue(np.array_equal(np.array(resized_image), np.array(resized_image2)))

        resized_image3 = feature_extractor.resize(image, (8, 16))
        self.assertTrue(isinstance(resized_image3, PIL.Image.Image))
        self.assertEqual(resized_image3.size, (8, 16))
        self.assertTrue(np.array_equal(np.array(resized_image1), np.array(resized_image3)))

    def test_resize_image_and_array_non_default_to_square(self):
        feature_extractor = ImageFeatureExtractionMixin()

        heights_widths = [
            # height, width
            # square image
            (28, 28),
            (27, 27),
            # rectangular image: h < w
            (28, 34),
            (29, 35),
            # rectangular image: h > w
            (34, 28),
            (35, 29),
        ]

        # single integer or single integer in tuple/list
        sizes = [22, 27, 28, 36, [22], (27,)]

        for (height, width), size in zip(heights_widths, sizes):
            for max_size in (None, 37, 1000):
                image = get_random_image(height, width)
                array = np.array(image)

                size = size[0] if isinstance(size, (list, tuple)) else size
                # Size can be an int or a tuple of ints.
                # If size is an int, smaller edge of the image will be matched to this number.
                # i.e, if height > width, then image will be rescaled to (size * height / width, size).
                if height < width:
                    exp_w, exp_h = (int(size * width / height), size)
                    if max_size is not None and max_size < exp_w:
                        exp_w, exp_h = max_size, int(max_size * exp_h / exp_w)
                elif width < height:
                    exp_w, exp_h = (size, int(size * height / width))
                    if max_size is not None and max_size < exp_h:
                        exp_w, exp_h = int(max_size * exp_w / exp_h), max_size
                else:
                    exp_w, exp_h = (size, size)
                    if max_size is not None and max_size < size:
                        exp_w, exp_h = max_size, max_size

                resized_image = feature_extractor.resize(image, size=size, default_to_square=False, max_size=max_size)
                self.assertTrue(isinstance(resized_image, PIL.Image.Image))
                self.assertEqual(resized_image.size, (exp_w, exp_h))

                # Passing an array converts it to a PIL Image.
                resized_image2 = feature_extractor.resize(array, size=size, default_to_square=False, max_size=max_size)
                self.assertTrue(isinstance(resized_image2, PIL.Image.Image))
                self.assertEqual(resized_image2.size, (exp_w, exp_h))
                self.assertTrue(np.array_equal(np.array(resized_image), np.array(resized_image2)))

    @require_torch
    def test_resize_tensor(self):
        feature_extractor = ImageFeatureExtractionMixin()
        tensor = torch.randint(0, 256, (16, 32, 3))
        array = tensor.numpy()

        # Size can be an int or a tuple of ints.
        resized_image = feature_extractor.resize(tensor, 8)
        self.assertTrue(isinstance(resized_image, PIL.Image.Image))
        self.assertEqual(resized_image.size, (8, 8))

        resized_image1 = feature_extractor.resize(tensor, (8, 16))
        self.assertTrue(isinstance(resized_image1, PIL.Image.Image))
        self.assertEqual(resized_image1.size, (8, 16))

        # Check we get the same results as with NumPy arrays.
        resized_image2 = feature_extractor.resize(array, 8)
        self.assertTrue(np.array_equal(np.array(resized_image), np.array(resized_image2)))

        resized_image3 = feature_extractor.resize(array, (8, 16))
        self.assertTrue(np.array_equal(np.array(resized_image1), np.array(resized_image3)))

    def test_normalize_image(self):
        feature_extractor = ImageFeatureExtractionMixin()
        image = get_random_image(16, 32)
        array = np.array(image)
        mean = [0.1, 0.5, 0.9]
        std = [0.2, 0.4, 0.6]

        # PIL Image are converted to NumPy arrays for the normalization
        normalized_image = feature_extractor.normalize(image, mean, std)
        self.assertTrue(isinstance(normalized_image, np.ndarray))
        self.assertEqual(normalized_image.shape, (3, 16, 32))

        # During the conversion rescale and channel first will be applied.
        expected = array.transpose(2, 0, 1).astype(np.float32) * (1 / 255.0)
        np_mean = np.array(mean).astype(np.float32)[:, None, None]
        np_std = np.array(std).astype(np.float32)[:, None, None]
        expected = (expected - np_mean) / np_std
        self.assertTrue(np.array_equal(normalized_image, expected))

    def test_normalize_array(self):
        feature_extractor = ImageFeatureExtractionMixin()
        array = np.random.random((16, 32, 3))
        mean = [0.1, 0.5, 0.9]
        std = [0.2, 0.4, 0.6]

        # mean and std can be passed as lists or NumPy arrays.
        expected = (array - np.array(mean)) / np.array(std)
        normalized_array = feature_extractor.normalize(array, mean, std)
        self.assertTrue(np.array_equal(normalized_array, expected))

        normalized_array = feature_extractor.normalize(array, np.array(mean), np.array(std))
        self.assertTrue(np.array_equal(normalized_array, expected))

        # Normalize will detect automatically if channel first or channel last is used.
        array = np.random.random((3, 16, 32))
        expected = (array - np.array(mean)[:, None, None]) / np.array(std)[:, None, None]
        normalized_array = feature_extractor.normalize(array, mean, std)
        self.assertTrue(np.array_equal(normalized_array, expected))

        normalized_array = feature_extractor.normalize(array, np.array(mean), np.array(std))
        self.assertTrue(np.array_equal(normalized_array, expected))

    @require_torch
    def test_normalize_tensor(self):
        feature_extractor = ImageFeatureExtractionMixin()
        tensor = torch.rand(16, 32, 3)
        mean = [0.1, 0.5, 0.9]
        std = [0.2, 0.4, 0.6]

        # mean and std can be passed as lists or tensors.
        expected = (tensor - torch.tensor(mean)) / torch.tensor(std)
        normalized_tensor = feature_extractor.normalize(tensor, mean, std)
        self.assertTrue(torch.equal(normalized_tensor, expected))

        normalized_tensor = feature_extractor.normalize(tensor, torch.tensor(mean), torch.tensor(std))
        self.assertTrue(torch.equal(normalized_tensor, expected))

        # Normalize will detect automatically if channel first or channel last is used.
        tensor = torch.rand(3, 16, 32)
        expected = (tensor - torch.tensor(mean)[:, None, None]) / torch.tensor(std)[:, None, None]
        normalized_tensor = feature_extractor.normalize(tensor, mean, std)
        self.assertTrue(torch.equal(normalized_tensor, expected))

        normalized_tensor = feature_extractor.normalize(tensor, torch.tensor(mean), torch.tensor(std))
        self.assertTrue(torch.equal(normalized_tensor, expected))

    def test_center_crop_image(self):
        feature_extractor = ImageFeatureExtractionMixin()
        image = get_random_image(16, 32)

        # Test various crop sizes: bigger on all dimensions, on one of the dimensions only and on both dimensions.
        crop_sizes = [8, (8, 64), 20, (32, 64)]
        for size in crop_sizes:
            cropped_image = feature_extractor.center_crop(image, size)
            self.assertTrue(isinstance(cropped_image, PIL.Image.Image))

            # PIL Image.size is transposed compared to NumPy or PyTorch (width first instead of height first).
            expected_size = (size, size) if isinstance(size, int) else (size[1], size[0])
            self.assertEqual(cropped_image.size, expected_size)

    def test_center_crop_array(self):
        feature_extractor = ImageFeatureExtractionMixin()
        image = get_random_image(16, 32)
        array = feature_extractor.to_numpy_array(image)

        # Test various crop sizes: bigger on all dimensions, on one of the dimensions only and on both dimensions.
        crop_sizes = [8, (8, 64), 20, (32, 64)]
        for size in crop_sizes:
            cropped_array = feature_extractor.center_crop(array, size)
            self.assertTrue(isinstance(cropped_array, np.ndarray))

            expected_size = (size, size) if isinstance(size, int) else size
            self.assertEqual(cropped_array.shape[-2:], expected_size)

            # Check result is consistent with PIL.Image.crop
            cropped_image = feature_extractor.center_crop(image, size)
            self.assertTrue(np.array_equal(cropped_array, feature_extractor.to_numpy_array(cropped_image)))

    @require_torch
    def test_center_crop_tensor(self):
        feature_extractor = ImageFeatureExtractionMixin()
        image = get_random_image(16, 32)
        array = feature_extractor.to_numpy_array(image)
        tensor = torch.tensor(array)

        # Test various crop sizes: bigger on all dimensions, on one of the dimensions only and on both dimensions.
        crop_sizes = [8, (8, 64), 20, (32, 64)]
        for size in crop_sizes:
            cropped_tensor = feature_extractor.center_crop(tensor, size)
            self.assertTrue(isinstance(cropped_tensor, torch.Tensor))

            expected_size = (size, size) if isinstance(size, int) else size
            self.assertEqual(cropped_tensor.shape[-2:], expected_size)

            # Check result is consistent with PIL.Image.crop
            cropped_image = feature_extractor.center_crop(image, size)
            self.assertTrue(torch.equal(cropped_tensor, torch.tensor(feature_extractor.to_numpy_array(cropped_image))))


@require_vision
class LoadImageTester(unittest.TestCase):
    def test_load_img_url(self):
        img = load_image(INVOICE_URL)
        img_arr = np.array(img)

        self.assertEqual(img_arr.shape, (1061, 750, 3))

    @is_flaky()
    def test_load_img_url_timeout(self):
        with self.assertRaises((ReadTimeout, ConnectTimeout)):
            load_image(INVOICE_URL, timeout=0.001)

    def test_load_img_local(self):
        img = load_image("./tests/fixtures/tests_samples/COCO/000000039769.png")
        img_arr = np.array(img)

        self.assertEqual(
            img_arr.shape,
            (480, 640, 3),
        )

    def test_load_img_base64_prefix(self):
        try:
            tmp_file = tempfile.NamedTemporaryFile(delete=False).name
            with open(tmp_file, "wb") as f:
                http_get(
                    "https://huggingface.co/datasets/hf-internal-testing/dummy-base64-images/raw/main/image_0.txt", f
                )

            with open(tmp_file, encoding="utf-8") as b64:
                img = load_image(b64.read())
                img_arr = np.array(img)

        finally:
            os.remove(tmp_file)

        self.assertEqual(img_arr.shape, (64, 32, 3))

    def test_load_img_base64(self):
        try:
            tmp_file = tempfile.NamedTemporaryFile(delete=False).name
            with open(tmp_file, "wb") as f:
                http_get(
                    "https://huggingface.co/datasets/hf-internal-testing/dummy-base64-images/raw/main/image_1.txt", f
                )

            with open(tmp_file, encoding="utf-8") as b64:
                img = load_image(b64.read())
                img_arr = np.array(img)

        finally:
            os.remove(tmp_file)

        self.assertEqual(img_arr.shape, (64, 32, 3))

    def test_load_img_base64_encoded_bytes(self):
        try:
            tmp_file = tempfile.NamedTemporaryFile(delete=False).name
            with open(tmp_file, "wb") as f:
                http_get(
                    "https://huggingface.co/datasets/hf-internal-testing/dummy-base64-images/raw/main/image_2.txt", f
                )

            with codecs.open(tmp_file, encoding="unicode_escape") as b64:
                img = load_image(b64.read())
                img_arr = np.array(img)

        finally:
            os.remove(tmp_file)

        self.assertEqual(img_arr.shape, (256, 256, 3))

    def test_load_img_rgba(self):
        # we use revision="refs/pr/1" until the PR is merged
        # https://hf.co/datasets/hf-internal-testing/fixtures_image_utils/discussions/1
        img = get_image_from_hub_dataset(
            "hf-internal-testing/fixtures_image_utils", "0-test-lena.png", revision="refs/pr/1"
        )

        img = load_image(img)  # img with mode RGBA
        img_arr = np.array(img)

        self.assertEqual(
            img_arr.shape,
            (512, 512, 3),
        )

    def test_load_img_la(self):
        # we use revision="refs/pr/1" until the PR is merged
        # https://hf.co/datasets/hf-internal-testing/fixtures_image_utils/discussions/1
        img = get_image_from_hub_dataset(
            "hf-internal-testing/fixtures_image_utils", "1-test-parrots.png", revision="refs/pr/1"
        )

        img = load_image(img)  # img with mode LA
        img_arr = np.array(img)

        self.assertEqual(
            img_arr.shape,
            (512, 768, 3),
        )

    def test_load_img_l(self):
        # we use revision="refs/pr/1" until the PR is merged
        # https://hf.co/datasets/hf-internal-testing/fixtures_image_utils/discussions/1
        img = get_image_from_hub_dataset(
            "hf-internal-testing/fixtures_image_utils", "2-test-tree.png", revision="refs/pr/1"
        )

        img = load_image(img)  # img with mode L
        img_arr = np.array(img)

        self.assertEqual(
            img_arr.shape,
            (381, 225, 3),
        )

    def test_load_img_exif_transpose(self):
        # we use revision="refs/pr/1" until the PR is merged
        # https://hf.co/datasets/hf-internal-testing/fixtures_image_utils/discussions/1

        img_without_exif_transpose = get_image_from_hub_dataset(
            "hf-internal-testing/fixtures_image_utils", "3-test-cat-rotated.jpg", revision="refs/pr/1"
        )
        img_arr_without_exif_transpose = np.array(img_without_exif_transpose)

        self.assertEqual(
            img_arr_without_exif_transpose.shape,
            (333, 500, 3),
        )

        img_with_exif_transpose = load_image(img_without_exif_transpose)
        img_arr_with_exif_transpose = np.array(img_with_exif_transpose)

        self.assertEqual(
            img_arr_with_exif_transpose.shape,
            (500, 333, 3),
        )


class UtilFunctionTester(unittest.TestCase):
    def test_get_image_size(self):
        # Test we can infer the size and channel dimension of an image.
        image = np.random.randint(0, 256, (32, 64, 3))
        self.assertEqual(get_image_size(image), (32, 64))

        image = np.random.randint(0, 256, (3, 32, 64))
        self.assertEqual(get_image_size(image), (32, 64))

        # Test the channel dimension can be overriden
        image = np.random.randint(0, 256, (3, 32, 64))
        self.assertEqual(get_image_size(image, channel_dim=ChannelDimension.LAST), (3, 32))

    def test_infer_channel_dimension(self):
        # Test we fail with invalid input
        with pytest.raises(ValueError):
            infer_channel_dimension_format(np.random.randint(0, 256, (10, 10)))

        with pytest.raises(ValueError):
            infer_channel_dimension_format(np.random.randint(0, 256, (10, 10, 10, 10, 10)))

        # Test we fail if neither first not last dimension is of size 3 or 1
        with pytest.raises(ValueError):
            infer_channel_dimension_format(np.random.randint(0, 256, (10, 1, 50)))

        # But if we explicitly set one of the number of channels to 50 it works
        inferred_dim = infer_channel_dimension_format(np.random.randint(0, 256, (10, 1, 50)), num_channels=50)
        self.assertEqual(inferred_dim, ChannelDimension.LAST)

        # Test we correctly identify the channel dimension
        image = np.random.randint(0, 256, (3, 4, 5))
        inferred_dim = infer_channel_dimension_format(image)
        self.assertEqual(inferred_dim, ChannelDimension.FIRST)

        image = np.random.randint(0, 256, (1, 4, 5))
        inferred_dim = infer_channel_dimension_format(image)
        self.assertEqual(inferred_dim, ChannelDimension.FIRST)

        image = np.random.randint(0, 256, (4, 5, 3))
        inferred_dim = infer_channel_dimension_format(image)
        self.assertEqual(inferred_dim, ChannelDimension.LAST)

        image = np.random.randint(0, 256, (4, 5, 1))
        inferred_dim = infer_channel_dimension_format(image)
        self.assertEqual(inferred_dim, ChannelDimension.LAST)

        # We can take a batched array of images and find the dimension
        image = np.random.randint(0, 256, (1, 3, 4, 5))
        inferred_dim = infer_channel_dimension_format(image)
        self.assertEqual(inferred_dim, ChannelDimension.FIRST)

    def test_get_channel_dimension_axis(self):
        # Test we correctly identify the channel dimension
        image = np.random.randint(0, 256, (3, 4, 5))
        inferred_axis = get_channel_dimension_axis(image)
        self.assertEqual(inferred_axis, 0)

        image = np.random.randint(0, 256, (1, 4, 5))
        inferred_axis = get_channel_dimension_axis(image)
        self.assertEqual(inferred_axis, 0)

        image = np.random.randint(0, 256, (4, 5, 3))
        inferred_axis = get_channel_dimension_axis(image)
        self.assertEqual(inferred_axis, 2)

        image = np.random.randint(0, 256, (4, 5, 1))
        inferred_axis = get_channel_dimension_axis(image)
        self.assertEqual(inferred_axis, 2)

        # We can take a batched array of images and find the dimension
        image = np.random.randint(0, 256, (1, 3, 4, 5))
        inferred_axis = get_channel_dimension_axis(image)
        self.assertEqual(inferred_axis, 1)
