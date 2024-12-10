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

import shutil
import tempfile
import unittest
from io import BytesIO
from typing import Optional

import numpy as np
import requests

from transformers import AriaProcessor
from transformers.models.auto.processing_auto import AutoProcessor
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from PIL import Image


@require_torch
@require_vision
class AriaProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = AriaProcessor

    @classmethod
    def setUpClass(cls):
        cls.tmpdirname = tempfile.mkdtemp()
        processor = AriaProcessor.from_pretrained("m-ric/Aria_hf_2", image_seq_len=2)
        processor.save_pretrained(cls.tmpdirname)
        cls.image1 = Image.open(
            BytesIO(
                requests.get(
                    "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
                ).content
            )
        )
        cls.image2 = Image.open(
            BytesIO(requests.get("https://cdn.britannica.com/59/94459-050-DBA42467/Skyline-Chicago.jpg").content)
        )
        cls.image3 = Image.open(
            BytesIO(
                requests.get(
                    "https://thumbs.dreamstime.com/b/golden-gate-bridge-san-francisco-purple-flowers-california-echium-candicans-36805947.jpg"
                ).content
            )
        )
        cls.bos_token = "<|im_start|>"
        cls.eos_token = "<|im_end|>"

        cls.image_token = processor.tokenizer.image_token
        cls.fake_image_token = "o"
        cls.global_img_token = "<|img|>"

        cls.bos_token_id = processor.tokenizer.convert_tokens_to_ids(cls.bos_token)
        cls.eos_token_id = processor.tokenizer.convert_tokens_to_ids(cls.eos_token)

        cls.image_token_id = processor.tokenizer.convert_tokens_to_ids(cls.image_token)
        cls.fake_image_token_id = processor.tokenizer.convert_tokens_to_ids(cls.fake_image_token)
        cls.global_img_tokens_id = processor.tokenizer(cls.global_img_token, add_special_tokens=False)["input_ids"]
        cls.padding_token_id = processor.tokenizer.pad_token_id
        cls.image_seq_len = 256

    def get_tokenizer(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).tokenizer

    def get_image_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).image_processor

    def get_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdirname)

    def test_kwargs_overrides_default_image_processor_kwargs(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor_components["image_processor"] = self.get_component(
            "image_processor", do_rescale=True, rescale_factor=1
        )
        processor_components["tokenizer"] = self.get_component("tokenizer", max_length=117, padding="max_length")

        processor = self.processor_class(**processor_components)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = self.prepare_text_inputs()
        image_input = self.prepare_image_inputs()

        inputs = processor(text=input_str, images=image_input, return_tensors="pt")
        self.assertLessEqual(inputs[self.images_input_name][0][0].mean(), 0)

    def test_process_interleaved_images_prompts_image_splitting(self):
        processor = self.get_processor()
        processor.image_processor.split_image = True

        # Test that a single image is processed correctly
        inputs = processor(images=self.image1, text="Ok<|img|>", images_kwargs={"split_image": True})
        self.assertEqual(np.array(inputs["pixel_values"]).shape, (2, 3, 980, 980))
        self.assertEqual(np.array(inputs["pixel_mask"]).shape, (2, 980, 980))

    def test_process_interleaved_images_prompts_no_image_splitting(self):
        processor = self.get_processor()
        processor.image_processor.split_image = False

        # Test that a single image is processed correctly
        inputs = processor(images=self.image1, text="Ok<|img|>")
        image1_expected_size = (980, 980)
        self.assertEqual(np.array(inputs["pixel_values"]).shape, (1, 3, *image1_expected_size))
        self.assertEqual(np.array(inputs["pixel_mask"]).shape, (1, *image1_expected_size))
        # fmt: on

        # Test a single sample with image and text
        image_str = "<|img|>"
        text_str = "In this image, we see"
        text = image_str + text_str
        inputs = processor(text=text, images=self.image1)

        # fmt: off
        tokenized_sentence = processor.tokenizer(text_str, add_special_tokens=False)

        expected_input_ids = [[self.image_token_id] * self.image_seq_len + tokenized_sentence["input_ids"]]
        # self.assertEqual(len(inputs["input_ids"]), len(expected_input_ids))

        self.assertEqual(inputs["input_ids"], expected_input_ids)
        self.assertEqual(inputs["attention_mask"], [[1] * len(expected_input_ids[0])])
        self.assertEqual(np.array(inputs["pixel_values"]).shape, (1, 3, *image1_expected_size))
        self.assertEqual(np.array(inputs["pixel_mask"]).shape, (1, *image1_expected_size))
        # fmt: on

        # Test that batch is correctly processed
        image_str = "<|img|>"
        text_str_1 = "In this image, we see"
        text_str_2 = "In this image, we see"

        text = [
            image_str + text_str_1,
            image_str + image_str + text_str_2,
        ]
        images = [[self.image1], [self.image2, self.image3]]

        inputs = processor(text=text, images=images, padding=True)

        # fmt: off
        tokenized_sentence_1 = processor.tokenizer(text_str_1, add_special_tokens=False)
        tokenized_sentence_2 = processor.tokenizer(text_str_2, add_special_tokens=False)

        image_tokens = [self.image_token_id] * self.image_seq_len
        expected_input_ids_1 = image_tokens + tokenized_sentence_1["input_ids"]
        expected_input_ids_2 = 2 * image_tokens + tokenized_sentence_2["input_ids"]

        # Pad the first input to match the second input
        pad_len = len(expected_input_ids_2) - len(expected_input_ids_1)

        expected_attention_mask = [[0] * pad_len + [1] * len(expected_input_ids_1), [1] * (len(expected_input_ids_2))]

        self.assertEqual(
            inputs["attention_mask"],
            expected_attention_mask
        )
        self.assertEqual(np.array(inputs['pixel_values']).shape, (3, 3, 980, 980))
        self.assertEqual(np.array(inputs['pixel_mask']).shape, (3, 980, 980))
        # fmt: on

    def test_non_nested_images_with_batched_text(self):
        processor = self.get_processor()
        processor.image_processor.do_image_splitting = False

        image_str = "<|img|>"
        text_str_1 = "In this image, we see"
        text_str_2 = "In this image, we see"

        text = [
            image_str + text_str_1,
            image_str + image_str + text_str_2,
        ]
        images = [self.image1, self.image2, self.image3]

        inputs = processor(text=text, images=images, padding=True)

        self.assertEqual(np.array(inputs["pixel_values"]).shape, (3, 3, 980, 980))
        self.assertEqual(np.array(inputs["pixel_mask"]).shape, (3, 980, 980))

    def test_apply_chat_template(self):
        # Message contains content which a mix of lists with images and image urls and string
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What do these images show?"},
                    {"type": "image"},
                    {"type": "image"},
                    "What do these images show?",
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "The first image shows the statue of Liberty in New York. The second image picture depicts Idefix, the dog of Obelix in Asterix and Obelix.",
                    }
                ],
            },
            {"role": "user", "content": [{"type": "text", "text": "And who is that?"}]},
        ]
        processor = self.get_processor()
        # Make short sequence length to test that the fake tokens are added correctly
        rendered = processor.apply_chat_template(messages, add_generation_prompt=True)
        print(rendered)

        expected_rendered = """<|im_start|>user
What do these images show?<fim_prefix><|img|><fim_suffix><fim_prefix><|img|><fim_suffix><|im_end|>
<|im_start|>assistant
The first image shows the statue of Liberty in New York. The second image picture depicts Idefix, the dog of Obelix in Asterix and Obelix.<|im_end|>
<|im_start|>user
And who is that?<|im_end|>
<|im_start|>assistant
"""
        self.assertEqual(rendered, expected_rendered)

    # Override as AriaProcessor needs image tokens in prompts
    def prepare_text_inputs(self, batch_size: Optional[int] = None):
        if batch_size is None:
            return "lower newer <|img|>"

        if batch_size < 1:
            raise ValueError("batch_size must be greater than 0")

        if batch_size == 1:
            return ["lower newer <|img|>"]
        return ["lower newer <|img|>", "<|img|> upper older longer string"] + ["<|img|> lower newer"] * (
            batch_size - 2
        )

    # Override tests as inputs_ids padded dimension is the second one but not the last one
    @require_vision
    @require_torch
    def test_kwargs_overrides_default_tokenizer_kwargs(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        image_processor = self.get_component("image_processor")
        tokenizer = self.get_component("tokenizer", max_length=30)

        processor = self.processor_class(tokenizer=tokenizer, image_processor=image_processor)
        self.skip_processor_without_typed_kwargs(processor)
        input_str = self.prepare_text_inputs()
        image_input = self.prepare_image_inputs()

        inputs = processor(text=input_str, images=image_input, return_tensors="pt", max_length=30)
        self.assertEqual(len(inputs["input_ids"][0]), 30)

    @require_torch
    @require_vision
    def test_structured_kwargs_nested(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        image_processor = self.get_component("image_processor")
        tokenizer = self.get_component("tokenizer")

        processor = self.processor_class(tokenizer=tokenizer, image_processor=image_processor)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = self.prepare_text_inputs()
        image_input = self.prepare_image_inputs()

        # Define the kwargs for each modality
        inputs = processor(
            text=input_str,
            images=image_input,
            common_kwargs={"return_tensors": "pt"},
            images_kwargs={"max_image_size": 980},
            text_kwargs={"padding": "max_length", "max_length": 120, "truncation": "longest_first"},
        )
        self.skip_processor_without_typed_kwargs(processor)

        self.assertEqual(inputs["pixel_values"].shape[3], 980)

        self.assertEqual(len(inputs["input_ids"][0]), 120)

    @require_torch
    @require_vision
    def test_structured_kwargs_nested_from_dict(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")

        image_processor = self.get_component("image_processor")
        tokenizer = self.get_component("tokenizer")

        processor = self.processor_class(tokenizer=tokenizer, image_processor=image_processor)
        self.skip_processor_without_typed_kwargs(processor)
        input_str = self.prepare_text_inputs()
        image_input = self.prepare_image_inputs()

        # Define the kwargs for each modality
        all_kwargs = {
            "common_kwargs": {"return_tensors": "pt"},
            "images_kwargs": {"max_image_size": 980},
            "text_kwargs": {"padding": "max_length", "max_length": 120, "truncation": "longest_first"},
        }

        inputs = processor(text=input_str, images=image_input, **all_kwargs)
        self.assertEqual(inputs["pixel_values"].shape[3], 980)
        self.assertEqual(len(inputs["input_ids"][0]), 120)

    @require_vision
    @require_torch
    def test_tokenizer_defaults_preserved_by_kwargs(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        image_processor = self.get_component("image_processor")
        tokenizer = self.get_component("tokenizer", max_length=30)

        processor = self.processor_class(tokenizer=tokenizer, image_processor=image_processor)
        self.skip_processor_without_typed_kwargs(processor)
        input_str = self.prepare_text_inputs()
        image_input = self.prepare_image_inputs()

        inputs = processor(text=input_str, images=image_input, return_tensors="pt")
        self.assertEqual(len(inputs["input_ids"][0]), 30)

    @require_torch
    @require_vision
    def test_unstructured_kwargs_batched(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        image_processor = self.get_component("image_processor")
        tokenizer = self.get_component("tokenizer")

        processor = self.processor_class(tokenizer=tokenizer, image_processor=image_processor)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = self.prepare_text_inputs(batch_size=2)
        image_input = self.prepare_image_inputs(batch_size=2)
        inputs = processor(
            text=input_str,
            images=image_input,
            return_tensors="pt",
            padding="longest",
            max_length=76,
            truncation=True,
            max_image_size=980,
        )

        self.assertEqual(inputs["pixel_values"].shape[1], 3)
        self.assertEqual(inputs["pixel_values"].shape[3], 980)
        self.assertEqual(len(inputs["input_ids"][0]), 76)

    @require_torch
    @require_vision
    def test_unstructured_kwargs(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        image_processor = self.get_component("image_processor")
        tokenizer = self.get_component("tokenizer")

        processor = self.processor_class(tokenizer=tokenizer, image_processor=image_processor)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = self.prepare_text_inputs()
        image_input = self.prepare_image_inputs()
        inputs = processor(
            text=input_str,
            images=image_input,
            return_tensors="pt",
            max_image_size=980,
            padding="max_length",
            max_length=120,
            truncation="longest_first",
        )

        self.assertEqual(inputs["pixel_values"].shape[3], 980)
        self.assertEqual(len(inputs["input_ids"][0]), 120)
