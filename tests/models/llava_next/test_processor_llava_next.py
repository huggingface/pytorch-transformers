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
import json
import tempfile
import unittest

import torch

from transformers import AutoProcessor, LlamaTokenizerFast, LlavaNextProcessor
from transformers.testing_utils import (
    require_vision,
)
from transformers.utils import is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from transformers import LlavaNextImageProcessor


@require_vision
class LlavaNextProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = LlavaNextProcessor

    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()

        image_processor = LlavaNextImageProcessor()
        tokenizer = LlamaTokenizerFast.from_pretrained("huggyllama/llama-7b")
        processor_kwargs = self.prepare_processor_dict()
        processor = LlavaNextProcessor(image_processor, tokenizer, **processor_kwargs)
        processor.save_pretrained(self.tmpdirname)

    def get_tokenizer(self, **kwargs):
        return LlavaNextProcessor.from_pretrained(self.tmpdirname, **kwargs).tokenizer

    def get_image_processor(self, **kwargs):
        return LlavaNextProcessor.from_pretrained(self.tmpdirname, **kwargs).image_processor

    def prepare_processor_dict(self):
        return {"chat_template": "dummy_template", "patch_size": 3, "vision_feature_select_strategy": "default"}

    @unittest.skip(
        "Skip because the model has no processor kwargs except for chat template and"
        "chat template is saved as a separate file. Stop skipping this test when the processor"
        "has new kwargs saved in config file."
    )
    def test_processor_to_json_string(self):
        pass

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

    def test_chat_template(self):
        processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-vicuna-7b-hf")
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

    def test_image_token_filling(self):
        processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-vicuna-7b-hf")
        processor.patch_size = 14
        processor.vision_feature_select_strategy = "default"
        # Important to check with non square image
        image = torch.randint(0, 2, (3, 500, 316))
        expected_image_tokens = 1526
        image_token_index = 32000

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]
        inputs = processor(
            text=[processor.apply_chat_template(messages)],
            images=[image],
            return_tensors="pt",
        )
        image_tokens = (inputs["input_ids"] == image_token_index).sum().item()
        self.assertEqual(expected_image_tokens, image_tokens)
