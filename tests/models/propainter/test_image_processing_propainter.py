# coding=utf-8
# Copyright 2024 HuggingFace Inc.
#
# Licensed under the S-Lab License, Version 1.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/sczhou/ProPainter/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import unittest
import warnings

import numpy as np

from transformers.image_utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_vision_available

from ...test_image_processing_common import (
    ImageProcessingTestMixin,
    prepare_video_inputs,
)


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image

    from transformers import ProPainterVideoProcessor


class ProPainterImageProcessingTester(unittest.TestCase):
    def __init__(
        self,
        parent,
        batch_size=5,
        num_channels=3,
        image_size=64,
        num_frames=10,
        min_resolution=30,
        max_resolution=80,
        do_resize=True,
        size=None,
        do_center_crop=True,
        crop_size=None,
        do_normalize=True,
        image_mean=OPENAI_CLIP_MEAN,
        image_std=OPENAI_CLIP_STD,
    ):
        super().__init__()
        size = size if size is not None else {"shortest_edge": 20}
        crop_size = crop_size if crop_size is not None else {"height": 64, "width": 64}
        outpainting_size = {"height": 64, "width": 72}
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.num_frames = num_frames
        self.image_size = image_size
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_resize = do_resize
        self.size = size
        self.do_center_crop = do_center_crop
        self.crop_size = crop_size
        self.outpainting_size = outpainting_size
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std

    def prepare_image_processor_dict(self):
        return {
            "do_resize": self.do_resize,
            "size": self.size,
            "do_center_crop": self.do_center_crop,
            "crop_size": self.crop_size,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
        }

    def expected_output_image_shape(self, images):
        return (
            self.num_frames,
            self.num_channels,
            self.crop_size["height"],
            self.crop_size["width"],
        )

    def expected_output_image_shape_outpainting(self, images):
        return (
            self.num_frames,
            self.num_channels,
            self.outpainting_size["height"],
            self.outpainting_size["width"],
        )

    def prepare_video_inputs(self, equal_resolution=False, numpify=False, torchify=False):
        return prepare_video_inputs(
            batch_size=self.batch_size,
            num_channels=self.num_channels,
            num_frames=self.num_frames,
            min_resolution=self.min_resolution,
            max_resolution=self.max_resolution,
            equal_resolution=equal_resolution,
            numpify=numpify,
            torchify=torchify,
        )


@require_torch
@require_vision
class ProPainterImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    video_processing_class = ProPainterVideoProcessor if is_vision_available() else None

    # Copied from tests.models.video_llava.test_image_processing_video_llava.VideoLlavaImageProcessingTest.setUp with VideoLlava->ProPainter
    def setUp(self):
        super().setUp()
        self.image_processor_tester = ProPainterImageProcessingTester(self)

    @property
    # Copied from tests.models.video_llava.test_image_processing_video_llava.VideoLlavaImageProcessingTest.image_processor_dict
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_video_processor_properties(self):
        image_processing = self.video_processing_class(**self.image_processor_dict)
        self.assertTrue(hasattr(image_processing, "do_resize"))
        self.assertTrue(hasattr(image_processing, "size"))
        self.assertTrue(hasattr(image_processing, "do_center_crop"))
        self.assertTrue(hasattr(image_processing, "center_crop"))
        self.assertTrue(hasattr(image_processing, "do_normalize"))
        self.assertTrue(hasattr(image_processing, "image_mean"))
        self.assertTrue(hasattr(image_processing, "image_std"))

    def test_image_processor_from_dict_with_kwargs(self):
        image_processor = self.video_processing_class.from_dict(self.image_processor_dict)
        self.assertEqual(image_processor.size, {"shortest_edge": 20})
        self.assertEqual(image_processor.crop_size, {"height": 64, "width": 64})

        image_processor = self.video_processing_class.from_dict(self.image_processor_dict, size=42, crop_size=84)
        self.assertEqual(image_processor.size, {"shortest_edge": 42})
        self.assertEqual(image_processor.crop_size, {"height": 84, "width": 84})

    def test_call_pil_video(self):
        # Initialize video_processing
        video_processing = self.video_processing_class(**self.image_processor_dict)

        # the inputs come in list of lists batched format
        video_inputs = self.image_processor_tester.prepare_video_inputs(equal_resolution=False)
        for video in video_inputs:
            self.assertIsInstance(video, list)
            self.assertIsInstance(video[0], Image.Image)

        mask_inputs = [[frame.point(lambda p: 1 if p >= 128 else 0) for frame in video] for video in video_inputs]
        for mask in mask_inputs:
            self.assertIsInstance(mask, list)
            self.assertIsInstance(mask[0], Image.Image)

        # Test not batched input (video inpainting)
        encoded_videos = video_processing(
            video_inputs[0], masks=mask_inputs[0], return_tensors="pt"
        ).pixel_values_videos
        expected_output_video_shape = (1, 10, 3, 64, 64)
        self.assertEqual(tuple(encoded_videos.shape), expected_output_video_shape)

        # Test batched  (video inpainting)
        encoded_videos = video_processing(video_inputs, masks=mask_inputs, return_tensors="pt").pixel_values_videos
        expected_output_video_shape = (5, 10, 3, 64, 64)
        self.assertEqual(tuple(encoded_videos.shape), expected_output_video_shape)

        # Test not batched input (video outpainting)
        encoded_videos = video_processing(
            video_inputs[0],
            masks=mask_inputs[0],
            video_painting_mode="video_outpainting",
            scale_size=(1.0, 1.2),
            return_tensors="pt",
        ).pixel_values_videos
        expected_output_video_shape = (1, 10, 3, 64, 72)
        self.assertEqual(tuple(encoded_videos.shape), expected_output_video_shape)

        # Test batched  (video outpainting)
        encoded_videos = video_processing(
            video_inputs,
            masks=mask_inputs,
            video_painting_mode="video_outpainting",
            scale_size=(1.0, 1.2),
            return_tensors="pt",
        ).pixel_values_videos
        expected_output_video_shape = (5, 10, 3, 64, 72)
        self.assertEqual(tuple(encoded_videos.shape), expected_output_video_shape)

    def test_call_numpy_video(self):
        # Initialize video_processing
        video_processing = self.video_processing_class(**self.image_processor_dict)

        # create random numpy tensors
        video_inputs = self.image_processor_tester.prepare_video_inputs(equal_resolution=False, numpify=True)
        for video in video_inputs:
            self.assertIsInstance(video, list)
            self.assertIsInstance(video[0], np.ndarray)

        mask_inputs = [[np.where(frame > 128, 1, 0) for frame in video] for video in video_inputs]
        for mask in mask_inputs:
            self.assertIsInstance(mask, list)
            self.assertIsInstance(mask[0], np.ndarray)

        # Test not batched input (video inpainting)
        encoded_images = video_processing(
            video_inputs[0], masks=mask_inputs[0], return_tensors="pt"
        ).pixel_values_videos
        expected_output_image_shape = (1, 10, 3, 64, 64)
        self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

        # Test batched (video inpainting)
        encoded_images = video_processing(video_inputs, masks=mask_inputs, return_tensors="pt").pixel_values_videos
        expected_output_image_shape = (5, 10, 3, 64, 64)
        self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

        # Test not batched input (video outpainting)
        encoded_videos = video_processing(
            video_inputs[0],
            masks=mask_inputs[0],
            video_painting_mode="video_outpainting",
            scale_size=(1.0, 1.2),
            return_tensors="pt",
        ).pixel_values_videos
        expected_output_video_shape = (1, 10, 3, 64, 72)
        self.assertEqual(tuple(encoded_videos.shape), expected_output_video_shape)

        # Test batched  (video outpainting)
        encoded_videos = video_processing(
            video_inputs,
            masks=mask_inputs,
            video_painting_mode="video_outpainting",
            scale_size=(1.0, 1.2),
            return_tensors="pt",
        ).pixel_values_videos
        expected_output_video_shape = (5, 10, 3, 64, 72)
        self.assertEqual(tuple(encoded_videos.shape), expected_output_video_shape)

    def test_call_pytorch_video(self):
        # Initialize video_processing
        video_processing = self.video_processing_class(**self.image_processor_dict)

        # create random PyTorch tensors
        video_inputs = self.image_processor_tester.prepare_video_inputs(equal_resolution=False, torchify=True)
        for video in video_inputs:
            self.assertIsInstance(video, list)
            self.assertIsInstance(video[0], torch.Tensor)

        mask_inputs = [
            [torch.where(frame > 128, torch.tensor(1), torch.tensor(0)) for frame in video] for video in video_inputs
        ]
        for mask in mask_inputs:
            self.assertIsInstance(mask, list)
            self.assertIsInstance(mask[0], torch.Tensor)

        # Test not batched input (video inpainting)
        encoded_images = video_processing(
            video_inputs[0], masks=mask_inputs[0], return_tensors="pt"
        ).pixel_values_videos
        expected_output_image_shape = (1, 10, 3, 64, 64)
        self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

        # Test batched (video inpainting)
        encoded_images = video_processing(video_inputs, masks=mask_inputs, return_tensors="pt").pixel_values_videos
        expected_output_image_shape = (5, 10, 3, 64, 64)
        self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)

        # Test not batched input (video outpainting)
        encoded_videos = video_processing(
            video_inputs[0],
            masks=mask_inputs[0],
            video_painting_mode="video_outpainting",
            scale_size=(1.0, 1.2),
            return_tensors="pt",
        ).pixel_values_videos
        expected_output_video_shape = (1, 10, 3, 64, 72)
        self.assertEqual(tuple(encoded_videos.shape), expected_output_video_shape)

        # Test batched  (video outpainting)
        encoded_videos = video_processing(
            video_inputs,
            masks=mask_inputs,
            video_painting_mode="video_outpainting",
            scale_size=(1.0, 1.2),
            return_tensors="pt",
        ).pixel_values_videos
        expected_output_video_shape = (5, 10, 3, 64, 72)
        self.assertEqual(tuple(encoded_videos.shape), expected_output_video_shape)

    def test_call_numpy_4_channels(self):
        # Test that can process images which have an arbitrary number of channels
        # Initialize video_processing
        video_processor = self.video_processing_class(**self.image_processor_dict)

        # create random numpy tensors
        self.image_processor_tester.num_channels = 4
        video_inputs = self.image_processor_tester.prepare_video_inputs(equal_resolution=False, numpify=True)

        mask_inputs = [[np.where(frame > 128, 1, 0) for frame in video] for video in video_inputs]

        # Test not batched input (video inpainting)
        encoded_images = video_processor(
            video_inputs[0],
            masks=mask_inputs[0],
            return_tensors="pt",
            input_data_format="channels_first",
            image_mean=0,
            image_std=1,
        ).pixel_values_videos
        expected_output_image_shape = self.image_processor_tester.expected_output_image_shape([video_inputs[0]])
        self.assertEqual(tuple(encoded_images.shape), (1, *expected_output_image_shape))

        # Test batched (video inpainting)
        encoded_images = video_processor(
            video_inputs,
            masks=mask_inputs,
            return_tensors="pt",
            input_data_format="channels_first",
            image_mean=0,
            image_std=1,
        ).pixel_values_videos
        expected_output_image_shape = self.image_processor_tester.expected_output_image_shape(video_inputs)
        self.assertEqual(
            tuple(encoded_images.shape),
            (self.image_processor_tester.batch_size, *expected_output_image_shape),
        )

        # Test not batched input (video outpainting)
        encoded_images = video_processor(
            video_inputs[0],
            masks=mask_inputs[0],
            video_painting_mode="video_outpainting",
            scale_size=(1.0, 1.2),
            return_tensors="pt",
            input_data_format="channels_first",
            image_mean=0,
            image_std=1,
        ).pixel_values_videos
        expected_output_image_shape = self.image_processor_tester.expected_output_image_shape_outpainting(
            [video_inputs[0]]
        )
        self.assertEqual(tuple(encoded_images.shape), (1, *expected_output_image_shape))

        # Test batched (video outpainting)
        encoded_images = video_processor(
            video_inputs,
            masks=mask_inputs,
            video_painting_mode="video_outpainting",
            scale_size=(1.0, 1.2),
            return_tensors="pt",
            input_data_format="channels_first",
            image_mean=0,
            image_std=1,
        ).pixel_values_videos
        expected_output_image_shape = self.image_processor_tester.expected_output_image_shape_outpainting(video_inputs)
        self.assertEqual(
            tuple(encoded_images.shape),
            (self.image_processor_tester.batch_size, *expected_output_image_shape),
        )

    def test_image_processor_preprocess_arguments(self):
        is_tested = False

        for image_processing_class in self.image_processor_list:
            image_processor = image_processing_class(**self.image_processor_dict)

            # validation done by _valid_processor_keys attribute
            if hasattr(image_processor, "_valid_processor_keys") and hasattr(image_processor, "preprocess"):
                preprocess_parameter_names = inspect.getfullargspec(image_processor.preprocess).args
                preprocess_parameter_names.remove("self")
                preprocess_parameter_names.sort()
                valid_processor_keys = image_processor._valid_processor_keys
                valid_processor_keys.sort()
                self.assertEqual(preprocess_parameter_names, valid_processor_keys)
                is_tested = True

            # validation done by @filter_out_non_signature_kwargs decorator
            if hasattr(image_processor.preprocess, "_filter_out_non_signature_kwargs"):
                if hasattr(self.image_processor_tester, "prepare_image_inputs"):
                    inputs = self.image_processor_tester.prepare_image_inputs()
                elif hasattr(self.image_processor_tester, "prepare_video_inputs"):
                    inputs = self.image_processor_tester.prepare_video_inputs()
                else:
                    self.skipTest(reason="No valid input preparation method found")

                mask_inputs = [[frame.point(lambda p: 1 if p >= 128 else 0) for frame in video] for video in inputs]
                with warnings.catch_warnings(record=True) as raised_warnings:
                    warnings.simplefilter("always")
                    image_processor(inputs, masks=mask_inputs, extra_argument=True)

                messages = " ".join([str(w.message) for w in raised_warnings])
                self.assertGreaterEqual(len(raised_warnings), 1)
                self.assertIn("extra_argument", messages)
                is_tested = True

        if not is_tested:
            self.skipTest(reason="No validation found for `preprocess` method")
