import os
import shutil
import tempfile
import unittest

import pytest

from transformers.models.xlm_roberta.tokenization_xlm_roberta import VOCAB_FILES_NAMES
from transformers.testing_utils import (
    require_sentencepiece,
    require_tokenizers,
    require_vision,
)
from transformers.utils import is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from transformers import TrOCRProcessor, ViTImageProcessor, XLMRobertaTokenizerFast


@require_sentencepiece
@require_tokenizers
@require_vision
class TrOCRProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    text_input_name = "labels"
    processor_class = TrOCRProcessor

    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()

        vocab_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "want", "##want", "##ed", "wa", "un", "runn", "##ing", ",", "low", "lowest"]  # fmt: skip
        self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        with open(self.vocab_file, "w", encoding="utf-8") as vocab_writer:
            vocab_writer.write("".join([x + "\n" for x in vocab_tokens]))

        image_processor = ViTImageProcessor.from_pretrained("hf-internal-testing/tiny-random-vit")
        tokenizer = XLMRobertaTokenizerFast.from_pretrained("FacebookAI/xlm-roberta-base")
        processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
        processor.save_pretrained(self.tmpdirname)

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def get_tokenizer(self, **kwargs):
        return XLMRobertaTokenizerFast.from_pretrained(self.tmpdirname, **kwargs)

    def get_image_processor(self, **kwargs):
        return ViTImageProcessor.from_pretrained(self.tmpdirname, **kwargs)

    def test_save_load_pretrained_default(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()
        processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)

        processor.save_pretrained(self.tmpdirname)
        processor = TrOCRProcessor.from_pretrained(self.tmpdirname)

        self.assertIsInstance(processor.tokenizer, XLMRobertaTokenizerFast)
        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer.get_vocab())
        self.assertIsInstance(processor.image_processor, ViTImageProcessor)
        self.assertEqual(processor.image_processor.to_json_string(), image_processor.to_json_string())

    def test_save_load_pretrained_additional_features(self):
        processor = TrOCRProcessor(tokenizer=self.get_tokenizer(), image_processor=self.get_image_processor())
        processor.save_pretrained(self.tmpdirname)
        tokenizer_add_kwargs = self.get_tokenizer(bos_token="(BOS)", eos_token="(EOS)")
        image_processor_add_kwargs = self.get_image_processor(do_normalize=False, padding_value=1.0)

        processor = TrOCRProcessor.from_pretrained(
            self.tmpdirname, bos_token="(BOS)", eos_token="(EOS)", do_normalize=False, padding_value=1.0
        )

        self.assertIsInstance(processor.tokenizer, XLMRobertaTokenizerFast)
        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer_add_kwargs.get_vocab())

        self.assertEqual(processor.image_processor.to_json_string(), image_processor_add_kwargs.to_json_string())
        self.assertIsInstance(processor.image_processor, ViTImageProcessor)

    def test_image_processor(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()
        processor = TrOCRProcessor(tokenizer=tokenizer, image_processor=image_processor)
        image_input = self.prepare_image_inputs()

        input_feat_extract = image_processor(image_input, return_tensors="np")
        input_processor = processor(images=image_input, return_tensors="np")

        for key in input_feat_extract.keys():
            self.assertAlmostEqual(input_feat_extract[key].sum(), input_processor[key].sum(), delta=1e-2)

    def test_tokenizer(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()
        processor = TrOCRProcessor(tokenizer=tokenizer, image_processor=image_processor)
        input_str = "lower newer"

        encoded_processor = processor(text=input_str)
        encoded_tok = tokenizer(input_str)

        for key in encoded_tok.keys():
            self.assertListEqual(encoded_tok[key], encoded_processor[key])

    def test_processor_text(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()
        processor = TrOCRProcessor(tokenizer=tokenizer, image_processor=image_processor)
        input_str = "lower newer"
        image_input = self.prepare_image_inputs()

        inputs = processor(text=input_str, images=image_input)

        self.assertListEqual(list(inputs.keys()), ["pixel_values", "labels"])

        # test if it raises when no input is passed
        with pytest.raises(ValueError):
            processor()

    def test_tokenizer_decode(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()
        processor = TrOCRProcessor(tokenizer=tokenizer, image_processor=image_processor)
        predicted_ids = [[1, 4, 5, 8, 1, 0, 8], [3, 4, 3, 1, 1, 8, 9]]

        decoded_processor = processor.batch_decode(predicted_ids)
        decoded_tok = tokenizer.batch_decode(predicted_ids)

        self.assertListEqual(decoded_tok, decoded_processor)
