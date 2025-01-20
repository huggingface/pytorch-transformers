# coding=utf-8
# Copyright 2025 HuggingFace Inc.
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
import shutil
import tempfile
import unittest

from transformers import AutoProcessor, LlamaTokenizerFast, LlavaNextVideoProcessor
from transformers.testing_utils import require_av, require_torch, require_vision
from transformers.utils import is_torch_available, is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from transformers import LlavaNextImageProcessor, LlavaNextVideoImageProcessor

if is_torch_available:
    import torch


@require_vision
class LlavaNextVideoProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = LlavaNextVideoProcessor

    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()
        image_processor = LlavaNextImageProcessor()
        video_processor = LlavaNextVideoImageProcessor()
        tokenizer = LlamaTokenizerFast.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf")
        processor_kwargs = self.prepare_processor_dict()

        processor = LlavaNextVideoProcessor(
            video_processor=video_processor, image_processor=image_processor, tokenizer=tokenizer, **processor_kwargs
        )
        processor.save_pretrained(self.tmpdirname)

    def get_tokenizer(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).tokenizer

    def get_image_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).image_processor

    def get_video_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).video_processor

    def prepare_processor_dict(self):
        return {
            "chat_template": "dummy_template",
            "num_additional_image_tokens": 6,
            "patch_size": 4,
            "vision_feature_select_strategy": "default",
        }

    def test_processor_to_json_string(self):
        processor = self.get_processor()
        obj = json.loads(processor.to_json_string())
        for key, value in self.prepare_processor_dict().items():
            # chat_tempalate are tested as a separate test because they are saved in separate files
            if key != "chat_template":
                self.assertEqual(obj[key], value)
                self.assertEqual(getattr(processor, key, None), value)

    # Copied from tests.models.llava.test_processor_llava.LlavaProcessorTest.test_chat_template_is_saved
    def test_chat_template_is_saved(self):
        processor_loaded = self.processor_class.from_pretrained(self.tmpdirname)
        processor_dict_loaded = json.loads(processor_loaded.to_json_string())
        # chat templates aren't serialized to json in processors
        self.assertFalse("chat_template" in processor_dict_loaded.keys())

        # they have to be saved as separate file and loaded back from that file
        # so we check if the same template is loaded
        processor_dict = self.prepare_processor_dict()
        self.assertTrue(processor_loaded.chat_template == processor_dict.get("chat_template", None))

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def test_chat_template(self):
        processor = AutoProcessor.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf")
        expected_prompt = "USER: <image>\nWhat is shown in this image? ASSISTANT:"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]

        formatted_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        self.assertEqual(expected_prompt, formatted_prompt)

    @require_av
    def test_chat_template_dict(self):
        processor = AutoProcessor.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video"},
                    {"type": "text", "text": "What is shown in this video?"},
                ],
            },
        ]

        formatted_prompt_tokenized = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_tensors=None
        )
        expected_output = [[1, 3148, 1001, 29901, 29871, 32000, 13, 5618, 338, 4318, 297, 445, 4863, 29973, 319, 1799, 9047, 13566, 29901]]  # fmt: skip
        self.assertListEqual(expected_output, formatted_prompt_tokenized)

        out_dict = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True)
        self.assertListEqual(list(out_dict.keys()), ["input_ids", "attention_mask"])

        # add image URL for return dict
        messages[0]["content"][0] = {
            "type": "video",
            "url": "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/720/Big_Buck_Bunny_720_10s_10MB.mp4",
        }
        out_dict_with_video = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True
        )
        self.assertListEqual(list(out_dict_with_video.keys()), ["input_ids", "attention_mask", "pixel_values_videos"])

    @require_torch
    @require_av
    def test_chat_template_dict_torch(self):
        processor = AutoProcessor.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf")
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "url": "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/720/Big_Buck_Bunny_720_10s_10MB.mp4",
                    },
                    {"type": "text", "text": "What is shown in this video?"},
                ],
            },
        ]

        out_dict_tensors = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        self.assertListEqual(list(out_dict_tensors.keys()), ["input_ids", "attention_mask", "pixel_values_videos"])
        self.assertTrue(isinstance(out_dict_tensors["input_ids"], torch.Tensor))
