# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch GotOcr2 model."""

import unittest

from transformers import (
    AutoProcessor,
    GotOcr2Config,
    is_torch_available,
    is_vision_available,
)
from transformers.testing_utils import cleanup, require_torch, slow, torch_device

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, _config_zero_init, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        GotOcr2ForConditionalGeneration,
    )


if is_vision_available():
    from transformers.image_utils import load_image


class GotOcr2VisionText2TextModelTester:
    def __init__(
        self,
        parent,
        batch_size=3,
        seq_length=7,
        num_channels=3,
        ignore_index=-100,
        image_size=64,
        bos_token_id=0,
        eos_token_id=0,
        pad_token_id=0,
        image_token_id=1,
        hidden_act="silu",
        hidden_size=128,
        vocab_size=99,
        intermediate_size=37,
        max_position_embeddings=512,
        max_window_layers=3,
        model_type="got_ocr2",
        num_attention_heads=4,
        num_hidden_layers=4,
        num_key_value_heads=2,
        rope_theta=10000,
        tie_word_embeddings=True,
        is_training=True,
        vision_config={
            "num_hidden_layers": 2,
            "output_channels": 64,
            "hidden_act": "quick_gelu",
            "hidden_size": 32,
            "mlp_ratio": 4,
            "num_attention_heads": 4,
            "patch_size": 2,
            "image_size": 64,
        },
        rope_scaling={"type": "mrope", "mrope_section": [2, 1, 1]},
    ):
        self.parent = parent
        self.ignore_index = ignore_index
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.image_token_id = image_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.max_window_layers = max_window_layers
        self.model_type = model_type
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.rope_theta = rope_theta
        self.tie_word_embeddings = tie_word_embeddings
        self.vision_config = vision_config
        self.rope_scaling = rope_scaling
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.is_training = is_training
        self.vocab_size = vocab_size
        self.num_image_tokens = 64
        self.seq_length = seq_length + self.num_image_tokens

    def get_config(self):
        return GotOcr2Config(
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            hidden_act=self.hidden_act,
            max_position_embeddings=self.max_position_embeddings,
            vision_config=self.vision_config,
            model_type=self.model_type,
            max_window_layers=self.max_window_layers,
            rope_scaling=self.rope_scaling,
            tie_word_embeddings=self.tie_word_embeddings,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            image_token_id=self.image_token_id,
            vocab_size=self.vocab_size,
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])

        return config, pixel_values

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=torch_device)

        # input_ids[:, -1] = self.pad_token_id
        input_ids[input_ids == self.image_token_id] = self.pad_token_id
        input_ids[:, : self.num_image_tokens] = self.image_token_id

        inputs_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict

    def create_and_check_model_fp16_forward(self, config, input_ids, pixel_values, attention_mask):
        model = GotOcr2ForConditionalGeneration(config=config)
        model.to(torch_device)
        model.half()
        model.eval()
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values.to(torch.bfloat16),
            return_dict=True,
        )["logits"]
        self.parent.assertFalse(torch.isnan(logits).any().item())

    def create_and_check_model_fp16_autocast_forward(self, config, input_ids, pixel_values, attention_mask):
        config.torch_dtype = torch.float16
        model = GotOcr2ForConditionalGeneration(config=config)
        model.to(torch_device)
        model.eval()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values.to(torch.bfloat16),
                return_dict=True,
            )["logits"]
        self.parent.assertFalse(torch.isnan(logits).any().item())


@require_torch
class GotOcr2ModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (GotOcr2ForConditionalGeneration,) if is_torch_available() else ()
    all_generative_model_classes = (GotOcr2ForConditionalGeneration,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "image-to-text": GotOcr2ForConditionalGeneration,
            "image-text-to-text": GotOcr2ForConditionalGeneration,
        }
        if is_torch_available()
        else {}
    )
    test_headmasking = False
    test_pruning = False

    def setUp(self):
        self.model_tester = GotOcr2VisionText2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=GotOcr2Config, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.assertIn(
                        ((param.data.mean() * 1e9).round() / 1e9).item(),
                        [0.0, 1.0],
                        msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                    )

    @unittest.skip(
        reason="VLMs can't generate from inputs embeds and pixels. This can be tested as part of bacbone LM, no need to run the test for VLMs"
    )
    def test_generate_from_inputs_embeds_with_static_cache(self):
        pass

    @unittest.skip(
        reason="GotOcr2's language backbone is Qwen2 which uses GQA so the KV cache is a non standard format"
    )
    def test_past_key_values_format(self):
        pass

    @unittest.skip(
        reason="GotOcr2 needs a dynamic control flow to pass pixel values to the forward function only in the first generation step"
    )
    def test_generate_compile_1_end_to_end(self):
        pass


@require_torch
class GotOcr2IntegrationTest(unittest.TestCase):
    def setUp(self):
        self.processor = AutoProcessor.from_pretrained("/home/ubuntu/models_implem/GotOcr2")

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @slow
    def test_small_model_integration_test_got_ocr_stop_strings(self):
        model_id = "/home/ubuntu/models_implem/GotOcr2"
        model = GotOcr2ForConditionalGeneration.from_pretrained(model_id)
        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/fixtures_ocr/resolve/main/iam_picture.jpeg"
        )

        inputs = self.processor(image, return_tensors="pt")
        generate_ids = model.generate(
            **inputs,
            do_sample=False,
            num_beams=1,
            tokenizer=self.processor.tokenizer,
            stop_strings="<|im_end|>",
            max_new_tokens=4096,
        )
        decoded_output = self.processor.decode(
            generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        expected_output = "industre"
        self.assertEqual(decoded_output, expected_output)

    @slow
    def test_small_model_integration_test_got_ocr_format(self):
        model_id = "/home/ubuntu/models_implem/GotOcr2"
        model = GotOcr2ForConditionalGeneration.from_pretrained(model_id)
        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/fixtures_got_ocr/resolve/main/image_ocr.jpg"
        )

        inputs = self.processor(image, return_tensors="pt", format=True)
        generate_ids = model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=4)
        decoded_output = self.processor.decode(
            generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        expected_output = "\\title{\nR'"
        self.assertEqual(decoded_output, expected_output)

    @slow
    def test_small_model_integration_test_got_ocr_fine_grained(self):
        model_id = "/home/ubuntu/models_implem/GotOcr2"
        model = GotOcr2ForConditionalGeneration.from_pretrained(model_id)
        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/fixtures_got_ocr/resolve/main/multi_box.png"
        )

        inputs = self.processor(image, return_tensors="pt", color="green")
        generate_ids = model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=4)
        decoded_output = self.processor.decode(
            generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        expected_output = "You should keep in"
        self.assertEqual(decoded_output, expected_output)

    @slow
    def test_small_model_integration_test_got_ocr_crop_to_patches(self):
        model_id = "/home/ubuntu/models_implem/GotOcr2"
        model = GotOcr2ForConditionalGeneration.from_pretrained(model_id)
        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/fixtures_got_ocr/resolve/main/one_column.png"
        )

        inputs = self.processor(image, return_tensors="pt", crop_to_patches=True)
        generate_ids = model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=4)
        decoded_output = self.processor.decode(
            generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        expected_output = "on developing architectural improvements"
        self.assertEqual(decoded_output, expected_output)

    @slow
    def test_small_model_integration_test_got_ocr_multi_pages(self):
        model_id = "/home/ubuntu/models_implem/GotOcr2"
        model = GotOcr2ForConditionalGeneration.from_pretrained(model_id)
        image1 = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/fixtures_got_ocr/resolve/main/one_column.png"
        )
        image2 = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/fixtures_got_ocr/resolve/main/multi_box.png"
        )

        inputs = self.processor([image1, image2], return_tensors="pt", multi_page=True)
        generate_ids = model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=4)
        decoded_output = self.processor.decode(
            generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        expected_output = "on developing architectural improvements"
        self.assertEqual(decoded_output, expected_output)

    @slow
    def test_small_model_integration_test_got_ocr_batched(self):
        model_id = "/home/ubuntu/models_implem/GotOcr2"
        model = GotOcr2ForConditionalGeneration.from_pretrained(model_id)
        image1 = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/fixtures_got_ocr/resolve/main/multi_box.png"
        )
        image2 = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/fixtures_got_ocr/resolve/main/image_ocr.jpg"
        )

        inputs = self.processor([image1, image2], return_tensors="pt")
        generate_ids = model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=4)
        decoded_output = self.processor.batch_decode(
            generate_ids[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        expected_output = ["Reducing the number", "R&D QUALITY"]
        self.assertEqual(decoded_output, expected_output)
