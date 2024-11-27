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
import shutil
import tempfile
import unittest

from transformers import AutoProcessor, AutoTokenizer, LlamaTokenizerFast, MolmoProcessor
from transformers.testing_utils import require_vision
from transformers.utils import is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from transformers import MolmoImageProcessor


@require_vision
class MolmoProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = MolmoProcessor

    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()

        image_processor = MolmoImageProcessor(do_center_crop=False)
        extra_special_tokens = {
            "image_token": "<image>",
            "boi_token": "<im_patch>",
            "eoi_token": "<im_start>",
            "im_patch_token": "<im_end>",
            "im_col_token": "<im_col>",
        }
        tokenizer = LlamaTokenizerFast.from_pretrained(
            "huggyllama/llama-7b", extra_special_tokens=extra_special_tokens
        )
        processor_kwargs = self.prepare_processor_dict()
        processor = MolmoProcessor(image_processor, tokenizer, **processor_kwargs)
        processor.save_pretrained(self.tmpdirname)

    def get_tokenizer(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).tokenizer

    def get_image_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).image_processor

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def prepare_processor_dict(self):
        return {"chat_template": "dummy_template"}

    @unittest.skip(
        "Skip because the model has no processor kwargs except for chat template and"
        "chat template is saved as a separate file. Stop skipping this test when the processor"
        "has new kwargs saved in config file."
    )
    def test_processor_to_json_string(self):
        pass

    def test_chat_template_is_saved(self):
        processor_loaded = self.processor_class.from_pretrained(self.tmpdirname)
        processor_dict_loaded = json.loads(processor_loaded.to_json_string())
        # chat templates aren't serialized to json in processors
        self.assertFalse("chat_template" in processor_dict_loaded.keys())

        # they have to be saved as separate file and loaded back from that file
        # so we check if the same template is loaded
        processor_dict = self.prepare_processor_dict()
        self.assertTrue(processor_loaded.chat_template == processor_dict.get("chat_template", None))

    def test_can_load_various_tokenizers(self):
        for checkpoint in ["Intel/molmo-gemma-2b", "allenai/Molmo-7B-D-0924"]:
            processor = MolmoProcessor.from_pretrained(checkpoint)
            tokenizer = AutoTokenizer.from_pretrained(checkpoint)
            self.assertEqual(processor.tokenizer.__class__, tokenizer.__class__)

    def test_chat_template(self):
        processor = MolmoProcessor.from_pretrained("allenai/Molmo-7B-D-0924-hf")
        expected_prompt = "User: <image> What is shown in this image? Assistant:"

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
