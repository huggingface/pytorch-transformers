# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
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

import shutil
import tempfile
import unittest

import pytest

from transformers import AutoProcessor, Qwen2Tokenizer
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from transformers import Qwen2_5_VLImageProcessor, Qwen2_5_VLProcessor


@require_vision
@require_torch
class Qwen2_5_VLProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = Qwen2_5_VLProcessor

    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()
        processor = Qwen2_5_VLProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", patch_size=4)
        processor.save_pretrained(self.tmpdirname)

    def get_tokenizer(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).tokenizer

    def get_image_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).image_processor

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def test_save_load_pretrained_default(self):
        tokenizer = self.get_tokenizer()
        image_processor = self.get_image_processor()

        processor = Qwen2_5_VLProcessor(tokenizer=tokenizer, image_processor=image_processor)
        processor.save_pretrained(self.tmpdirname)
        processor = Qwen2_5_VLProcessor.from_pretrained(self.tmpdirname, use_fast=False)

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer.get_vocab())
        self.assertEqual(processor.image_processor.to_json_string(), image_processor.to_json_string())
        self.assertIsInstance(processor.tokenizer, Qwen2Tokenizer)
        self.assertIsInstance(processor.image_processor, Qwen2_5_VLImageProcessor)

    def test_image_processor(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = Qwen2_5_VLProcessor(tokenizer=tokenizer, image_processor=image_processor)

        image_input = self.prepare_image_inputs()

        input_image_proc = image_processor(image_input, return_tensors="np")
        input_processor = processor(images=image_input, text="dummy", return_tensors="np")

        for key in input_image_proc.keys():
            self.assertAlmostEqual(input_image_proc[key].sum(), input_processor[key].sum(), delta=1e-2)

    def test_processor(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = Qwen2_5_VLProcessor(tokenizer=tokenizer, image_processor=image_processor)

        input_str = "lower newer"
        image_input = self.prepare_image_inputs()
        inputs = processor(text=input_str, images=image_input)

        self.assertListEqual(
            list(inputs.keys()),
            ["input_ids", "attention_mask", "pixel_values", "image_grid_thw"],
        )

        # test if it raises when no input is passed
        with pytest.raises(ValueError):
            processor()

        # test if it raises when no text is passed
        with pytest.raises(TypeError):
            processor(images=image_input)

    def test_model_input_names(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = Qwen2_5_VLProcessor(tokenizer=tokenizer, image_processor=image_processor)

        input_str = "lower newer"
        image_input = self.prepare_image_inputs()
        video_inputs = self.prepare_video_inputs()

        inputs = processor(text=input_str, images=image_input, videos=video_inputs)

        self.assertListEqual(list(inputs.keys()), processor.model_input_names)
