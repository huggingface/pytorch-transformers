import shutil
import tempfile
import unittest

import torch

from transformers import GemmaTokenizer
from transformers.models.colpali.processing_colpali import ColPaliProcessor
from transformers.testing_utils import get_tests_dir, require_torch, require_vision
from transformers.utils import is_vision_available
from transformers.utils.dummy_vision_objects import SiglipImageProcessor

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from transformers import (
        ColPaliProcessor,
        PaliGemmaProcessor,
        SiglipImageProcessor,
    )

SAMPLE_VOCAB = get_tests_dir("fixtures/test_sentencepiece.model")


@require_vision
class ColPaliProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = ColPaliProcessor

    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()
        image_processor = SiglipImageProcessor.from_pretrained("google/siglip-so400m-patch14-384")
        image_processor.image_seq_length = 0
        tokenizer = GemmaTokenizer(SAMPLE_VOCAB, keep_accents=True)
        processor = PaliGemmaProcessor(image_processor=image_processor, tokenizer=tokenizer)
        processor.save_pretrained(self.tmpdirname)

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    @require_torch
    @require_vision
    def test_process_images(self):
        # Processor configuration
        image_input = self.prepare_image_inputs()
        image_processor = self.get_component("image_processor")
        tokenizer = self.get_component("tokenizer", max_length=112, padding="max_length")
        image_processor.image_seq_length = 14

        # Get the processor
        processor = self.processor_class(
            tokenizer=tokenizer,
            image_processor=image_processor,
        )

        # Process the image
        batch_feature = processor.process_images(images=image_input, return_tensors="pt")

        # Assertions
        self.assertIn("pixel_values", batch_feature)
        self.assertEqual(batch_feature["pixel_values"].shape, torch.Size([1, 3, 384, 384]))

    @require_torch
    @require_vision
    def test_process_queries(self):
        # Inputs
        queries = [
            "Is attention really all you need?",
            "Are Benjamin, Antoine, Merve, and Jo best friends?",
        ]

        # Processor configuration
        image_processor = self.get_component("image_processor")
        tokenizer = self.get_component("tokenizer", max_length=112, padding="max_length")
        image_processor.image_seq_length = 14

        # Get the processor
        processor = self.processor_class(
            tokenizer=tokenizer,
            image_processor=image_processor,
        )

        # Process the image
        batch_feature = processor.process_queries(text=queries, return_tensors="pt")

        # Assertions
        self.assertIn("input_ids", batch_feature)
        self.assertIsInstance(batch_feature["input_ids"], torch.Tensor)
        self.assertEqual(batch_feature["input_ids"].shape[0], len(queries))

        # The following tests are overwritten as ColPaliProcessor can only take one of images or text as input at a time

    def test_tokenizer_defaults_preserved_by_kwargs(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor_components["tokenizer"] = self.get_component("tokenizer", max_length=117, padding="max_length")

        processor = self.processor_class(**processor_components)
        self.skip_processor_without_typed_kwargs(processor)
        input_str = self.prepare_text_inputs()
        inputs = processor(text=input_str, return_tensors="pt")
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

        processor = self.processor_class(**processor_components)
        self.skip_processor_without_typed_kwargs(processor)

        image_input = self.prepare_image_inputs()

        inputs = processor(images=image_input, return_tensors="pt")
        self.assertLessEqual(inputs[self.images_input_name][0][0].mean(), 0)

    def test_kwargs_overrides_default_tokenizer_kwargs(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor_components["tokenizer"] = self.get_component("tokenizer", padding="longest")

        processor = self.processor_class(**processor_components)
        self.skip_processor_without_typed_kwargs(processor)
        input_str = self.prepare_text_inputs()
        inputs = processor(text=input_str, return_tensors="pt", max_length=112, padding="max_length")
        self.assertEqual(inputs[self.text_input_name].shape[-1], 112)

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

        image_input = self.prepare_image_inputs()

        inputs = processor(images=image_input, do_rescale=True, rescale_factor=-1, return_tensors="pt")
        self.assertLessEqual(inputs[self.images_input_name][0][0].mean(), 0)

    def test_unstructured_kwargs(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor = self.processor_class(**processor_components)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = self.prepare_text_inputs()
        inputs = processor(
            text=input_str,
            return_tensors="pt",
            do_rescale=True,
            rescale_factor=-1,
            padding="max_length",
            max_length=76,
        )

        self.assertEqual(inputs[self.text_input_name].shape[-1], 76)

    def test_unstructured_kwargs_batched(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor = self.processor_class(**processor_components)
        self.skip_processor_without_typed_kwargs(processor)

        image_input = self.prepare_image_inputs(batch_size=2)
        inputs = processor(
            images=image_input,
            return_tensors="pt",
            do_rescale=True,
            rescale_factor=-1,
            padding="longest",
            max_length=76,
        )

        self.assertLessEqual(inputs[self.images_input_name][0][0].mean(), 0)

    def test_doubly_passed_kwargs(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor = self.processor_class(**processor_components)
        self.skip_processor_without_typed_kwargs(processor)

        image_input = self.prepare_image_inputs()
        with self.assertRaises(ValueError):
            _ = processor(
                images=image_input,
                images_kwargs={"do_rescale": True, "rescale_factor": -1},
                do_rescale=True,
                return_tensors="pt",
            )

    def test_structured_kwargs_nested(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor = self.processor_class(**processor_components)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = self.prepare_text_inputs()

        # Define the kwargs for each modality
        all_kwargs = {
            "common_kwargs": {"return_tensors": "pt"},
            "images_kwargs": {"do_rescale": True, "rescale_factor": -1},
            "text_kwargs": {"padding": "max_length", "max_length": 76},
        }

        inputs = processor(text=input_str, **all_kwargs)
        self.skip_processor_without_typed_kwargs(processor)

        self.assertEqual(inputs[self.text_input_name].shape[-1], 76)

    def test_structured_kwargs_nested_from_dict(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor = self.processor_class(**processor_components)
        self.skip_processor_without_typed_kwargs(processor)
        image_input = self.prepare_image_inputs()

        # Define the kwargs for each modality
        all_kwargs = {
            "common_kwargs": {"return_tensors": "pt"},
            "images_kwargs": {"do_rescale": True, "rescale_factor": -1},
            "text_kwargs": {"padding": "max_length", "max_length": 76},
        }

        inputs = processor(images=image_input, **all_kwargs)
        self.assertEqual(inputs[self.text_input_name].shape[-1], 76)
