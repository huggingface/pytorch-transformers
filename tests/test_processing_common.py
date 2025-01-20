# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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


import inspect
import json
import random
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

from transformers.models.auto.processing_auto import processor_class_from_name
from transformers.processing_utils import Unpack
from transformers.testing_utils import (
    check_json_file_has_correct_format,
    require_torch,
    require_vision,
)
from transformers.utils import is_vision_available


global_rng = random.Random()

if is_vision_available():
    from PIL import Image


def prepare_image_inputs():
    """This function prepares a list of PIL images"""
    image_inputs = [np.random.randint(255, size=(3, 30, 400), dtype=np.uint8)]
    image_inputs = [Image.fromarray(np.moveaxis(x, 0, -1)) for x in image_inputs]
    return image_inputs


# Copied from tests.models.whisper.test_feature_extraction_whisper.floats_list
def floats_list(shape, scale=1.0, rng=None, name=None):
    """Creates a random float32 tensor"""
    if rng is None:
        rng = global_rng

    values = []
    for batch_idx in range(shape[0]):
        values.append([])
        for _ in range(shape[1]):
            values[-1].append(rng.random() * scale)

    return values


@require_torch
@require_vision
class ProcessorTesterMixin:
    processor_class = None
    text_input_name = "input_ids"
    images_input_name = "pixel_values"
    videos_input_name = "pixel_values_videos"
    audio_input_name = "input_features"

    def prepare_processor_dict(self):
        return {}

    def get_component(self, attribute, **kwargs):
        assert attribute in self.processor_class.attributes
        component_class_name = getattr(self.processor_class, f"{attribute}_class")
        if isinstance(component_class_name, tuple):
            component_class_name = component_class_name[0]

        component_class = processor_class_from_name(component_class_name)
        component = component_class.from_pretrained(self.tmpdirname, **kwargs)  # noqa
        if "tokenizer" in attribute and not component.pad_token:
            component.pad_token = "[TEST_PAD]"
            if component.pad_token_id is None:
                component.pad_token_id = 0

        return component

    def prepare_components(self):
        components = {}
        for attribute in self.processor_class.attributes:
            component = self.get_component(attribute)
            components[attribute] = component

        return components

    def get_processor(self):
        components = self.prepare_components()
        processor = self.processor_class(**components, **self.prepare_processor_dict())
        return processor

    def prepare_text_inputs(self, batch_size: Optional[int] = None):
        if batch_size is None:
            return "lower newer"

        if batch_size < 1:
            raise ValueError("batch_size must be greater than 0")

        if batch_size == 1:
            return ["lower newer"]
        return ["lower newer", "upper older longer string"] + ["lower newer"] * (batch_size - 2)

    @require_vision
    def prepare_image_inputs(self, batch_size: Optional[int] = None):
        """This function prepares a list of PIL images for testing"""
        if batch_size is None:
            return prepare_image_inputs()[0]
        if batch_size < 1:
            raise ValueError("batch_size must be greater than 0")
        return prepare_image_inputs() * batch_size

    @require_vision
    def prepare_video_inputs(self):
        """This function prepares a list of numpy videos."""
        video_input = [np.random.randint(255, size=(3, 30, 400), dtype=np.uint8)] * 8
        image_inputs = [video_input] * 3  # batch-size=3
        return image_inputs

    def test_processor_to_json_string(self):
        processor = self.get_processor()
        obj = json.loads(processor.to_json_string())
        for key, value in self.prepare_processor_dict().items():
            self.assertEqual(obj[key], value)
            self.assertEqual(getattr(processor, key, None), value)

    def test_processor_from_and_save_pretrained(self):
        processor_first = self.get_processor()

        with tempfile.TemporaryDirectory() as tmpdirname:
            saved_files = processor_first.save_pretrained(tmpdirname)
            if len(saved_files) > 0:
                check_json_file_has_correct_format(saved_files[0])
                processor_second = self.processor_class.from_pretrained(tmpdirname)

                self.assertEqual(processor_second.to_dict(), processor_first.to_dict())

                for attribute in processor_first.attributes:
                    attribute_first = getattr(processor_first, attribute)
                    attribute_second = getattr(processor_second, attribute)

                    # tokenizer repr contains model-path from where we loaded
                    if "tokenizer" not in attribute:
                        self.assertEqual(repr(attribute_first), repr(attribute_second))

    # These kwargs-related tests ensure that processors are correctly instantiated.
    # they need to be applied only if an image_processor exists.

    def skip_processor_without_typed_kwargs(self, processor):
        # TODO this signature check is to test only uniformized processors.
        # Once all are updated, remove it.
        is_kwargs_typed_dict = False
        call_signature = inspect.signature(processor.__call__)
        for param in call_signature.parameters.values():
            if param.kind == param.VAR_KEYWORD and param.annotation != param.empty:
                is_kwargs_typed_dict = (
                    hasattr(param.annotation, "__origin__") and param.annotation.__origin__ == Unpack
                )
        if not is_kwargs_typed_dict:
            self.skipTest(f"{self.processor_class} doesn't have typed kwargs.")

    def test_tokenizer_defaults_preserved_by_kwargs(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor_components["tokenizer"] = self.get_component("tokenizer", max_length=117, padding="max_length")
        processor_kwargs = self.prepare_processor_dict()

        processor = self.processor_class(**processor_components, **processor_kwargs)
        self.skip_processor_without_typed_kwargs(processor)
        input_str = self.prepare_text_inputs()
        image_input = self.prepare_image_inputs()
        inputs = processor(text=input_str, images=image_input, return_tensors="pt")
        self.assertEqual(inputs[self.text_input_name].shape[-1], 117)

    def test_image_processor_defaults_preserved_by_image_kwargs(self):
        """
        We use do_rescale=True, rescale_factor=-1 to ensure that image_processor kwargs are preserved in the processor.
        We then check that the mean of the pixel_values is less than or equal to 0 after processing.
        Since the original pixel_values are in [0, 255], this is a good indicator that the rescale_factor is indeed applied.
        """
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor_components["image_processor"] = self.get_component(
            "image_processor", do_rescale=True, rescale_factor=-1
        )
        processor_components["tokenizer"] = self.get_component("tokenizer", max_length=117, padding="max_length")
        processor_kwargs = self.prepare_processor_dict()

        processor = self.processor_class(**processor_components, **processor_kwargs)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = self.prepare_text_inputs()
        image_input = self.prepare_image_inputs()

        inputs = processor(text=input_str, images=image_input, return_tensors="pt")
        self.assertLessEqual(inputs[self.images_input_name][0][0].mean(), 0)

    def test_kwargs_overrides_default_tokenizer_kwargs(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor_components["tokenizer"] = self.get_component("tokenizer", padding="longest")
        processor_kwargs = self.prepare_processor_dict()

        processor = self.processor_class(**processor_components, **processor_kwargs)
        self.skip_processor_without_typed_kwargs(processor)
        input_str = self.prepare_text_inputs()
        image_input = self.prepare_image_inputs()
        inputs = processor(
            text=input_str, images=image_input, return_tensors="pt", max_length=112, padding="max_length"
        )
        self.assertEqual(inputs[self.text_input_name].shape[-1], 112)

    def test_kwargs_overrides_default_image_processor_kwargs(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor_components["image_processor"] = self.get_component(
            "image_processor", do_rescale=True, rescale_factor=1
        )
        processor_components["tokenizer"] = self.get_component("tokenizer", max_length=117, padding="max_length")
        processor_kwargs = self.prepare_processor_dict()

        processor = self.processor_class(**processor_components, **processor_kwargs)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = self.prepare_text_inputs()
        image_input = self.prepare_image_inputs()

        inputs = processor(text=input_str, images=image_input, do_rescale=True, rescale_factor=-1, return_tensors="pt")
        self.assertLessEqual(inputs[self.images_input_name][0][0].mean(), 0)

    def test_unstructured_kwargs(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor_kwargs = self.prepare_processor_dict()
        processor = self.processor_class(**processor_components, **processor_kwargs)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = self.prepare_text_inputs()
        image_input = self.prepare_image_inputs()
        inputs = processor(
            text=input_str,
            images=image_input,
            return_tensors="pt",
            do_rescale=True,
            rescale_factor=-1,
            padding="max_length",
            max_length=76,
        )

        self.assertLessEqual(inputs[self.images_input_name][0][0].mean(), 0)
        self.assertEqual(inputs[self.text_input_name].shape[-1], 76)

    def test_unstructured_kwargs_batched(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor_kwargs = self.prepare_processor_dict()
        processor = self.processor_class(**processor_components, **processor_kwargs)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = self.prepare_text_inputs(batch_size=2)
        image_input = self.prepare_image_inputs(batch_size=2)
        inputs = processor(
            text=input_str,
            images=image_input,
            return_tensors="pt",
            do_rescale=True,
            rescale_factor=-1,
            padding="longest",
            max_length=76,
        )

        self.assertLessEqual(inputs[self.images_input_name][0][0].mean(), 0)
        self.assertTrue(
            len(inputs[self.text_input_name][0]) == len(inputs[self.text_input_name][1])
            and len(inputs[self.text_input_name][1]) < 76
        )

    def test_doubly_passed_kwargs(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor_kwargs = self.prepare_processor_dict()
        processor = self.processor_class(**processor_components, **processor_kwargs)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = [self.prepare_text_inputs()]
        image_input = self.prepare_image_inputs()
        with self.assertRaises(ValueError):
            _ = processor(
                text=input_str,
                images=image_input,
                images_kwargs={"do_rescale": True, "rescale_factor": -1},
                do_rescale=True,
                return_tensors="pt",
            )

    def test_structured_kwargs_nested(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor_kwargs = self.prepare_processor_dict()
        processor = self.processor_class(**processor_components, **processor_kwargs)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = self.prepare_text_inputs()
        image_input = self.prepare_image_inputs()

        # Define the kwargs for each modality
        all_kwargs = {
            "common_kwargs": {"return_tensors": "pt"},
            "images_kwargs": {"do_rescale": True, "rescale_factor": -1},
            "text_kwargs": {"padding": "max_length", "max_length": 76},
        }

        inputs = processor(text=input_str, images=image_input, **all_kwargs)
        self.skip_processor_without_typed_kwargs(processor)

        self.assertLessEqual(inputs[self.images_input_name][0][0].mean(), 0)
        self.assertEqual(inputs[self.text_input_name].shape[-1], 76)

    def test_structured_kwargs_nested_from_dict(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor_kwargs = self.prepare_processor_dict()
        processor = self.processor_class(**processor_components, **processor_kwargs)
        self.skip_processor_without_typed_kwargs(processor)
        input_str = self.prepare_text_inputs()
        image_input = self.prepare_image_inputs()

        # Define the kwargs for each modality
        all_kwargs = {
            "common_kwargs": {"return_tensors": "pt"},
            "images_kwargs": {"do_rescale": True, "rescale_factor": -1},
            "text_kwargs": {"padding": "max_length", "max_length": 76},
        }

        inputs = processor(text=input_str, images=image_input, **all_kwargs)
        self.assertLessEqual(inputs[self.images_input_name][0][0].mean(), 0)
        self.assertEqual(inputs[self.text_input_name].shape[-1], 76)

    # text + audio kwargs testing
    @require_torch
    def test_tokenizer_defaults_preserved_by_kwargs_audio(self):
        if "feature_extractor" not in self.processor_class.attributes:
            self.skipTest(f"feature_extractor attribute not present in {self.processor_class}")

        feature_extractor = self.get_component("feature_extractor")
        tokenizer = self.get_component("tokenizer", max_length=300, padding="max_length")
        processor_kwargs = self.prepare_processor_dict()

        processor = self.processor_class(tokenizer=tokenizer, feature_extractor=feature_extractor, **processor_kwargs)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = "lower newer"
        raw_speech = floats_list((3, 1000))
        raw_speech = [np.asarray(audio) for audio in raw_speech]
        inputs = processor(text=input_str, audio=raw_speech, return_tensors="pt")
        self.assertEqual(len(inputs[self.text_input_name][0]), 300)

    @require_torch
    def test_kwargs_overrides_default_tokenizer_kwargs_audio(self):
        if "feature_extractor" not in self.processor_class.attributes:
            self.skipTest(f"feature_extractor attribute not present in {self.processor_class}")

        feature_extractor = self.get_component("feature_extractor")
        tokenizer = self.get_component("tokenizer", max_length=117)
        processor_kwargs = self.prepare_processor_dict()

        processor = self.processor_class(tokenizer=tokenizer, feature_extractor=feature_extractor, **processor_kwargs)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = "lower newer"
        raw_speech = floats_list((3, 1000))
        raw_speech = [np.asarray(audio) for audio in raw_speech]
        inputs = processor(text=input_str, audio=raw_speech, return_tensors="pt", max_length=300, padding="max_length")

        self.assertEqual(len(inputs[self.text_input_name][0]), 300)

    @require_torch
    def test_unstructured_kwargs_audio(self):
        if "feature_extractor" not in self.processor_class.attributes:
            self.skipTest(f"feature_extractor attribute not present in {self.processor_class}")

        feature_extractor = self.get_component("feature_extractor")
        tokenizer = self.get_component("tokenizer")
        processor_kwargs = self.prepare_processor_dict()

        processor = self.processor_class(tokenizer=tokenizer, feature_extractor=feature_extractor, **processor_kwargs)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = "lower newer"
        raw_speech = floats_list((3, 1000))
        raw_speech = [np.asarray(audio) for audio in raw_speech]
        inputs = processor(text=input_str, audio=raw_speech, return_tensors="pt", max_length=300, padding="max_length")

        self.assertEqual(len(inputs[self.text_input_name][0]), 300)

    @require_torch
    def test_doubly_passed_kwargs_audio(self):
        if "feature_extractor" not in self.processor_class.attributes:
            self.skipTest(f"feature_extractor attribute not present in {self.processor_class}")

        feature_extractor = self.get_component("feature_extractor")
        tokenizer = self.get_component("tokenizer")
        processor_kwargs = self.prepare_processor_dict()

        processor = self.processor_class(tokenizer=tokenizer, feature_extractor=feature_extractor, **processor_kwargs)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = ["lower newer"]
        raw_speech = floats_list((3, 1000))
        raw_speech = [np.asarray(audio) for audio in raw_speech]
        with self.assertRaises(ValueError):
            _ = processor(
                text=input_str,
                audio=raw_speech,
                text_kwargs={"padding": "max_length"},
                padding="max_length",
            )

    @require_torch
    @require_vision
    def test_structured_kwargs_audio_nested(self):
        if "feature_extractor" not in self.processor_class.attributes:
            self.skipTest(f"feature_extractor attribute not present in {self.processor_class}")

        feature_extractor = self.get_component("feature_extractor")
        tokenizer = self.get_component("tokenizer", max_length=117)
        processor_kwargs = self.prepare_processor_dict()

        processor = self.processor_class(tokenizer=tokenizer, feature_extractor=feature_extractor, **processor_kwargs)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = ["lower newer"]
        raw_speech = floats_list((3, 1000))
        raw_speech = [np.asarray(audio) for audio in raw_speech]

        # Define the kwargs for each modality
        all_kwargs = {
            "common_kwargs": {"return_tensors": "pt"},
            "text_kwargs": {"padding": "max_length", "max_length": 76},
            "audio_kwargs": {"padding": "max_length", "max_length": 300},
        }

        inputs = processor(text=input_str, audio=raw_speech, **all_kwargs)
        self.assertEqual(len(inputs[self.text_input_name][0]), 76)

    # TODO (molbap) use the same structure of attribute kwargs for other tests to avoid duplication
    def test_overlapping_text_image_kwargs_handling(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")

        processor_components = self.prepare_components()
        processor = self.processor_class(**processor_components)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = self.prepare_text_inputs()
        image_input = self.prepare_image_inputs()

        with self.assertRaises(ValueError):
            _ = processor(
                text=input_str,
                images=image_input,
                return_tensors="pt",
                padding="max_length",
                text_kwargs={"padding": "do_not_pad"},
            )

    def test_overlapping_text_audio_kwargs_handling(self):
        """
        Checks that `padding`, or any other overlap arg between audio extractor and tokenizer
        is be passed to only text and ignored for audio for BC purposes
        """
        if "feature_extractor" not in self.processor_class.attributes:
            self.skipTest(f"feature_extractor attribute not present in {self.processor_class}")

        feature_extractor = self.get_component("feature_extractor")
        tokenizer = self.get_component("tokenizer")
        processor_kwargs = self.prepare_processor_dict()

        processor = self.processor_class(tokenizer=tokenizer, feature_extractor=feature_extractor, **processor_kwargs)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = ["lower newer"]
        audio_lengths = [4000, 8000, 16000, 32000]
        raw_speech = [np.asarray(audio)[:length] for audio, length in zip(floats_list((3, 32_000)), audio_lengths)]

        # padding = True should not raise an error and will if the audio processor popped its value to None
        _ = processor(text=input_str, audio=raw_speech, padding=True, return_tensors="pt")

    def test_prepare_and_validate_optional_call_args(self):
        processor = self.get_processor()
        optional_call_args_name = getattr(processor, "optional_call_args", [])
        num_optional_call_args = len(optional_call_args_name)
        if num_optional_call_args == 0:
            self.skipTest("No optional call args")
        # test all optional call args are given
        optional_call_args = processor.prepare_and_validate_optional_call_args(
            *(f"optional_{i}" for i in range(num_optional_call_args))
        )
        self.assertEqual(
            optional_call_args, {arg_name: f"optional_{i}" for i, arg_name in enumerate(optional_call_args_name)}
        )
        # test only one optional call arg is given
        optional_call_args = processor.prepare_and_validate_optional_call_args("optional_1")
        self.assertEqual(optional_call_args, {optional_call_args_name[0]: "optional_1"})
        # test no optional call arg is given
        optional_call_args = processor.prepare_and_validate_optional_call_args()
        self.assertEqual(optional_call_args, {})
        # test too many optional call args are given
        with self.assertRaises(ValueError):
            processor.prepare_and_validate_optional_call_args(
                *(f"optional_{i}" for i in range(num_optional_call_args + 1))
            )

    def test_chat_template_save_loading(self):
        processor = self.get_processor()
        existing_tokenizer_template = getattr(processor.tokenizer, "chat_template", None)
        processor.chat_template = "test template"
        with tempfile.TemporaryDirectory() as tmpdirname:
            processor.save_pretrained(tmpdirname)
            self.assertTrue(Path(tmpdirname, "chat_template.json").is_file())
            self.assertFalse(Path(tmpdirname, "chat_template.jinja").is_file())
            reloaded_processor = self.processor_class.from_pretrained(tmpdirname)
            self.assertEqual(processor.chat_template, reloaded_processor.chat_template)
            # When we don't use single-file chat template saving, processor and tokenizer chat templates
            # should remain separate
            self.assertEqual(getattr(reloaded_processor.tokenizer, "chat_template", None), existing_tokenizer_template)

        with tempfile.TemporaryDirectory() as tmpdirname:
            processor.save_pretrained(tmpdirname, save_raw_chat_template=True)
            self.assertTrue(Path(tmpdirname, "chat_template.jinja").is_file())
            self.assertFalse(Path(tmpdirname, "chat_template.json").is_file())
            reloaded_processor = self.processor_class.from_pretrained(tmpdirname)
            self.assertEqual(processor.chat_template, reloaded_processor.chat_template)
            # When we save as single files, tokenizers and processors share a chat template, which means
            # the reloaded tokenizer should get the chat template as well
            self.assertEqual(reloaded_processor.chat_template, reloaded_processor.tokenizer.chat_template)
