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

import unittest

import numpy as np

from transformers.image_utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_vision_available

from ...test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image

    from transformers import LlavaNextVideoImageProcessor


class LlavaNextVideoProcessingTester:
    def __init__(
        self,
        parent,
        batch_size=5,
        num_channels=3,
        image_size=18,
        min_resolution=30,
        max_resolution=80,
        do_resize=True,
        size=None,
        do_center_crop=True,
        crop_size=None,
        do_normalize=True,
        image_mean=OPENAI_CLIP_MEAN,
        image_std=OPENAI_CLIP_STD,
        do_convert_rgb=True,
    ):
        size = size if size is not None else {"shortest_edge": 20}
        crop_size = crop_size if crop_size is not None else {"height": 18, "width": 18}
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_resize = do_resize
        self.size = size
        self.do_center_crop = do_center_crop
        self.crop_size = crop_size
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.do_convert_rgb = do_convert_rgb

    def prepare_image_processor_dict(self):
        return {
            "do_resize": self.do_resize,
            "size": self.size,
            "do_center_crop": self.do_center_crop,
            "crop_size": self.crop_size,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "do_convert_rgb": self.do_convert_rgb,
        }

    # Copied from tests.models.clip.test_image_processing_clip.CLIPImageProcessingTester.expected_output_image_shape
    def expected_output_image_shape(self, images):
        return self.num_channels, self.crop_size["height"], self.crop_size["width"]

    # Copied from tests.models.clip.test_image_processing_clip.CLIPImageProcessingTester.prepare_image_inputs
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

    def prepare_video_inputs(self, equal_resolution=False, numpify=False, torchify=False):
        images = prepare_image_inputs(
            batch_size=self.batch_size,
            num_channels=self.num_channels,
            min_resolution=self.min_resolution,
            max_resolution=self.max_resolution,
            equal_resolution=equal_resolution,
            numpify=numpify,
            torchify=torchify,
        )

        # let's simply copy the frames to fake a long video-clip
        if numpify or torchify:
            videos = []
            for image in images:
                if numpify:
                    video = image[None, ...].repeat(8, 0)
                else:
                    video = image[None, ...].repeat(8, 1, 1, 1)
                videos.append(video)
        else:
            videos = []
            for pil_image in images:
                videos.append([pil_image] * 8)

        return videos


@require_torch
@require_vision
class LlavaNextVideoProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_class = LlavaNextVideoImageProcessor if is_vision_available() else None

    def setUp(self):
        super().setUp()
        self.image_processor_tester = LlavaNextVideoProcessingTester(self)

    @property
    # Copied from tests.models.clip.test_image_processing_clip.CLIPImageProcessingTest.image_processor_dict
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        image_processing = self.image_processing_class(**self.image_processor_dict)
        self.assertTrue(hasattr(image_processing, "do_resize"))
        self.assertTrue(hasattr(image_processing, "size"))
        self.assertTrue(hasattr(image_processing, "do_center_crop"))
        self.assertTrue(hasattr(image_processing, "center_crop"))
        self.assertTrue(hasattr(image_processing, "do_normalize"))
        self.assertTrue(hasattr(image_processing, "image_mean"))
        self.assertTrue(hasattr(image_processing, "image_std"))
        self.assertTrue(hasattr(image_processing, "do_convert_rgb"))

    # Copied from tests.models.clip.test_image_processing_clip.CLIPImageProcessingTest.test_image_processor_from_dict_with_kwargs
    def test_image_processor_from_dict_with_kwargs(self):
        image_processor = self.image_processing_class.from_dict(self.image_processor_dict)
        self.assertEqual(image_processor.size, {"shortest_edge": 20})
        self.assertEqual(image_processor.crop_size, {"height": 18, "width": 18})

        image_processor = self.image_processing_class.from_dict(self.image_processor_dict, size=42, crop_size=84)
        self.assertEqual(image_processor.size, {"shortest_edge": 42})
        self.assertEqual(image_processor.crop_size, {"height": 84, "width": 84})

    def test_call_pil(self):
        # Initialize image_processing
        image_processing = self.image_processing_class(**self.image_processor_dict)
        # create random numpy tensors
        video_inputs = self.image_processor_tester.prepare_video_inputs(equal_resolution=True)
        for video in video_inputs:
            self.assertIsInstance(video[0], Image.Image)

        # Test not batched input (pass as `videos` arg to test that ImageProcessor can handle videos in absence of images!)
        encoded_videos = image_processing(images=video_inputs[0], return_tensors="pt").pixel_values_videos
        expected_output_video_shape = (1, 8, 3, 18, 18)
        self.assertEqual(tuple(encoded_videos.shape), expected_output_video_shape)

        # Test batched
        encoded_videos = image_processing(images=video_inputs, return_tensors="pt").pixel_values_videos
        expected_output_video_shape = (5, 8, 3, 18, 18)
        self.assertEqual(tuple(encoded_videos.shape), expected_output_video_shape)

    def test_call_numpy(self):
        # Initialize image_processing
        image_processing = self.image_processing_class(**self.image_processor_dict)
        # create random numpy tensors
        video_inputs = self.image_processor_tester.prepare_video_inputs(equal_resolution=True, numpify=True)
        for video in video_inputs:
            self.assertIsInstance(video, np.ndarray)

        # Test not batched input (pass as `videos` arg to test that ImageProcessor can handle videos in absence of images!)
        encoded_videos = image_processing(images=video_inputs[0], return_tensors="pt").pixel_values_videos
        expected_output_video_shape = (1, 8, 3, 18, 18)
        self.assertEqual(tuple(encoded_videos.shape), expected_output_video_shape)

        # Test batched
        encoded_videos = image_processing(images=video_inputs, return_tensors="pt").pixel_values_videos
        expected_output_video_shape = (5, 8, 3, 18, 18)
        self.assertEqual(tuple(encoded_videos.shape), expected_output_video_shape)

    def test_call_pytorch(self):
        # Initialize image_processing
        image_processing = self.image_processing_class(**self.image_processor_dict)
        # create random PyTorch tensors
        video_inputs = self.image_processor_tester.prepare_video_inputs(equal_resolution=True, torchify=True)
        for video in video_inputs:
            self.assertIsInstance(video, torch.Tensor)

        # Test not batched input
        encoded_videos = image_processing(images=video_inputs[0], return_tensors="pt").pixel_values_videos
        expected_output_video_shape = (1, 8, 3, 18, 18)
        self.assertEqual(tuple(encoded_videos.shape), expected_output_video_shape)

        # Test batched
        encoded_videos = image_processing(images=video_inputs, return_tensors="pt").pixel_values_videos
        expected_output_video_shape = (5, 8, 3, 18, 18)
        self.assertEqual(tuple(encoded_videos.shape), expected_output_video_shape)

    @unittest.skip("LlavaNextVideoImageProcessor doesn't treat 4 channel PIL and numpy consistently yet")
    def test_call_numpy_4_channels(self):
        pass
