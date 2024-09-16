# Copyright 2021 The HuggingFace Team. All rights reserved.
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

from transformers import AutoTokenizer, LlamaTokenizerFast, LlavaProcessor
from transformers.testing_utils import require_vision
from transformers.utils import is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from transformers import CLIPImageProcessor


@require_vision
class LlavaProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = LlavaProcessor

    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()

        image_processor = CLIPImageProcessor()
        tokenizer = LlamaTokenizerFast.from_pretrained("huggyllama/llama-7b")
        processor_kwargs = self.prepare_processor_dict()
        processor = LlavaProcessor(image_processor, tokenizer, **processor_kwargs)
        processor.save_pretrained(self.tmpdirname)

    def get_tokenizer(self, **kwargs):
        return LlavaProcessor.from_pretrained(self.tmpdirname, **kwargs).tokenizer

    def get_image_processor(self, **kwargs):
        return LlavaProcessor.from_pretrained(self.tmpdirname, **kwargs).image_processor

    def prepare_processor_dict(self):
        return {"chat_template": "dummy_template"}

    def test_processor_to_json_string(self):
        processor = self.get_processor()
        obj = json.loads(processor.to_json_string())
        for key, value in self.prepare_processor_dict().items():
            # chat templates are popped from dict
            if key != "chat_template":
                self.assertEqual(obj[key], value)
                self.assertEqual(getattr(processor, key, None), value)

    def test_chat_template_is_saved(self):
        processor_loaded = self.processor_class.from_pretrained(self.tmpdirname)
        processor_dict = self.prepare_processor_dict()
        self.assertTrue(processor_loaded.chat_template == processor_dict.get("chat_template", None))

    def test_can_load_various_tokenizers(self):
        for checkpoint in ["Intel/llava-gemma-2b", "llava-hf/llava-1.5-7b-hf"]:
            processor = LlavaProcessor.from_pretrained(checkpoint)
            tokenizer = AutoTokenizer.from_pretrained(checkpoint)
            self.assertEqual(processor.tokenizer.__class__, tokenizer.__class__)

    def test_chat_template(self):
        processor = LlavaProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
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
