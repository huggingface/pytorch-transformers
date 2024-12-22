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
"""Testing suite for the PyTorch PaliGemma model."""

import unittest

import requests

from transformers import (
    PaliGemmaConfig,
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
    is_torch_available,
    is_vision_available,
)
from transformers.testing_utils import (
    cleanup,
    require_read_token,
    require_torch,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor


if is_torch_available():
    import torch


if is_vision_available():
    from PIL import Image


class PaliGemmaVisionText2TextModelTester:
    def __init__(
        self,
        parent,
        ignore_index=-100,
        image_token_index=0,
        projector_hidden_act="gelu",
        seq_length=25,
        vision_feature_select_strategy="default",
        vision_feature_layer=-1,
        projection_dim=32,
        text_config={
            "model_type": "gemma",
            "seq_length": 128,
            "is_training": True,
            # "use_input_mask": True,
            "use_token_type_ids": False,
            "use_labels": True,
            "vocab_size": 99,
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "intermediate_size": 37,
            "hidden_activation": "gelu_pytorch_tanh",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 16,
            "type_sequence_label_size": 2,
            "initializer_range": 0.02,
            "num_labels": 3,
            "num_choices": 4,
            "pad_token_id": 1,
        },
        is_training=True,
        vision_config={
            "use_labels": True,
            "image_size": 20,
            "patch_size": 5,
            "num_image_tokens": 4,
            "num_channels": 3,
            "is_training": True,
            "hidden_size": 32,
            "projection_dim": 32,
            "num_key_value_heads": 1,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 37,
            "dropout": 0.1,
            "attention_dropout": 0.1,
            "initializer_range": 0.02,
        },
        use_cache=False,
    ):
        self.parent = parent
        self.ignore_index = ignore_index
        # `image_token_index` is set to 0 to pass "resize_embeddings" test, do not modify
        self.image_token_index = image_token_index
        self.projector_hidden_act = projector_hidden_act
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.vision_feature_layer = vision_feature_layer
        self.text_config = text_config
        self.vision_config = vision_config
        self.seq_length = seq_length
        self.projection_dim = projection_dim
        self.pad_token_id = text_config["pad_token_id"]

        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.vocab_size = text_config["vocab_size"]
        self.hidden_size = text_config["hidden_size"]
        self.num_attention_heads = text_config["num_attention_heads"]
        self.is_training = is_training

        self.batch_size = 3
        self.num_channels = vision_config["num_channels"]
        self.image_size = vision_config["image_size"]
        self.encoder_seq_length = seq_length
        self.use_cache = use_cache

    def get_config(self):
        return PaliGemmaConfig(
            text_config=self.text_config,
            vision_config=self.vision_config,
            ignore_index=self.ignore_index,
            image_token_index=self.image_token_index,
            projector_hidden_act=self.projector_hidden_act,
            projection_dim=self.projection_dim,
            vision_feature_select_strategy=self.vision_feature_select_strategy,
            vision_feature_layer=self.vision_feature_layer,
        )

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor(
            [
                self.batch_size,
                self.vision_config["num_channels"],
                self.vision_config["image_size"],
                self.vision_config["image_size"],
            ]
        )
        config = self.get_config()

        return config, pixel_values

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        input_ids = ids_tensor([self.batch_size, self.seq_length], config.text_config.vocab_size - 1) + 1
        attention_mask = input_ids.ne(self.pad_token_id).to(torch_device)

        # set the 16 first tokens to be image, and ensure that no other tokens are image tokens
        # do not change this unless you modified image size or patch size
        input_ids[input_ids == config.image_token_index] = self.pad_token_id
        input_ids[:, :16] = config.image_token_index
        inputs_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids,
            "token_type_ids": torch.zeros_like(input_ids),
        }
        return config, inputs_dict


@require_torch
class PaliGemmaForConditionalGenerationModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    """
    Model tester for `PaliGemmaForConditionalGeneration`.
    """

    all_model_classes = (PaliGemmaForConditionalGeneration,) if is_torch_available() else ()
    all_generative_model_classes = (PaliGemmaForConditionalGeneration,) if is_torch_available() else ()
    pipeline_model_mapping = {"image-text-to-text": PaliGemmaForConditionalGeneration}
    fx_compatible = False
    test_pruning = False
    test_torchscript = False
    test_head_masking = False
    _is_composite = True

    def setUp(self):
        self.model_tester = PaliGemmaVisionText2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=PaliGemmaConfig, has_text_modality=False)

    # overwrite inputs_embeds tests because we need to delete "pixel values" for LVLMs
    def test_inputs_embeds(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            inputs = self._prepare_for_class(inputs_dict, model_class)

            input_ids = inputs["input_ids"]
            del inputs["input_ids"]
            del inputs["pixel_values"]

            wte = model.get_input_embeddings()
            inputs["inputs_embeds"] = wte(input_ids)

            with torch.no_grad():
                model(**inputs)

    # overwrite inputs_embeds tests because we need to delete "pixel values" for LVLMs
    # while some other models require pixel_values to be present
    def test_inputs_embeds_matches_input_ids(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            inputs = self._prepare_for_class(inputs_dict, model_class)
            input_ids = inputs["input_ids"]
            del inputs["input_ids"]
            del inputs["pixel_values"]

            inputs_embeds = model.get_input_embeddings()(input_ids)

            with torch.no_grad():
                out_ids = model(input_ids=input_ids, **inputs)[0]
                out_embeds = model(inputs_embeds=inputs_embeds, **inputs)[0]
            self.assertTrue(torch.allclose(out_embeds, out_ids))

    # Copied from tests.models.llava.test_modeling_llava.LlavaForConditionalGenerationModelTest.test_mismatching_num_image_tokens
    def test_mismatching_num_image_tokens(self):
        """
        Tests that VLMs through an error with explicit message saying what is wrong
        when number of images don't match number of image tokens in the text.
        Also we need to test multi-image cases when one prompr has multiple image tokens.
        """
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device)
            _ = model(**input_dict)  # successfull forward with no modifications

            # remove one image but leave the image token in text
            input_dict["pixel_values"] = input_dict["pixel_values"][-1:, ...]
            with self.assertRaises(ValueError):
                _ = model(**input_dict)

            # simulate multi-image case by concatenating inputs where each has exactly one image/image-token
            input_ids = input_dict["input_ids"][:1]
            pixel_values = input_dict["pixel_values"][:1]
            input_ids = torch.cat([input_ids, input_ids], dim=0)

            # one image and two image tokens raise an error
            with self.assertRaises(ValueError):
                _ = model(input_ids=input_ids, pixel_values=pixel_values)

            # two images and two image tokens don't raise an error
            pixel_values = torch.cat([pixel_values, pixel_values], dim=0)
            _ = model(input_ids=input_ids, pixel_values=pixel_values)

    @unittest.skip(
        reason="This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(
        reason="This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(
        reason="This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip(reason="Some undefined behavior encountered with test versions of this model. Skip for now.")
    def test_cpu_offload(self):
        pass

    @unittest.skip(reason="Some undefined behavior encountered with test versions of this model. Skip for now.")
    def test_disk_offload_bin(self):
        pass

    @unittest.skip(reason="Some undefined behavior encountered with test versions of this model. Skip for now.")
    def test_disk_offload_safetensors(self):
        pass

    @unittest.skip(reason="Some undefined behavior encountered with test versions of this model. Skip for now.")
    def test_model_parallelism(self):
        pass

    @unittest.skip(
        reason="PaliGemmma's SigLip encoder uses the same initialization scheme as the Flax original implementation"
    )
    def test_initialization(self):
        pass

    # TODO extend valid outputs to include this test @Molbap
    @unittest.skip(reason="PaliGemma has currently one output format.")
    def test_model_outputs_equivalence(self):
        pass

    # TODO fix the loss = nan in the testing configuration chosen @Molbap
    @unittest.skip(reason="Edge case giving loss nan values in testing configuration.")
    def test_determinism(self):
        pass

    @unittest.skip(reason="PaliGemma does not use feedforward chunking.")
    def test_feed_forward_chunking(self):
        pass

    @unittest.skip(reason="PaliGemma does not support low_cpu_mem_usage.")
    def test_save_load_low_cpu_mem_usage(self):
        pass

    @unittest.skip(reason="PaliGemma does not support low_cpu_mem_usage.")
    def test_save_load_low_cpu_mem_usage_checkpoints(self):
        pass

    @unittest.skip(reason="PaliGemma does not support low_cpu_mem_usage.")
    def test_save_load_low_cpu_mem_usage_no_safetensors(self):
        pass

    @unittest.skip(
        reason="VLMs doen't accept inputs embeds and pixel values at the same time. So if the test passed for bacbone LM, it passes for VLM also"
    )
    def test_generate_from_inputs_embeds_with_static_cache(self):
        pass

    @unittest.skip("FlashAttention only support fp16 and bf16 data type")
    def test_flash_attn_2_fp32_ln(self):
        pass

    @unittest.skip(
        "VLMs need lots of steps to prepare images/mask correctly to get pad-free inputs. Can be tested as part of LLM test"
    )
    def test_flash_attention_2_padding_matches_padding_free_with_position_ids(self):
        pass

    # TODO (joao, raushan): fix me -- the problem is in `cache_position[0] == 0`, i.e. dynamic control flow
    @unittest.skip("PaliGemma is not compatible with end-to-end generation compilation")
    def test_generate_compile_fullgraph(self):
        pass


@slow
@require_torch
@require_read_token
class PaliGemmaForConditionalGenerationIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.processor = PaliGemmaProcessor.from_pretrained("google/paligemma-3b-pt-224")

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def test_small_model_integration_test(self):
        # Let' s make sure we test the preprocessing to replace what is used
        model_id = "google/paligemma-3b-pt-224"
        model = PaliGemmaForConditionalGeneration.from_pretrained(model_id)
        prompt = ""
        image_file = (
            "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/cow_beach_1.png"
        )
        raw_image = Image.open(requests.get(image_file, stream=True).raw)
        inputs = self.processor(images=raw_image, text=prompt, return_tensors="pt")
        EXPECTED_INPUT_IDS = torch.tensor([[257152] * 256 + [2, 108]])
        self.assertTrue(torch.equal(inputs["input_ids"], EXPECTED_INPUT_IDS))

        output = model.generate(**inputs, max_new_tokens=20)
        EXPECTED_DECODED_TEXT = "\ncow on the beach"  # fmt: skip

        self.assertEqual(
            self.processor.decode(output[0], skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    def test_small_model_integration_test_multiimage(self):
        model_id = "google/paligemma-3b-ft-nlvr2-448"  # checkpoint tuned for multiple images
        model = PaliGemmaForConditionalGeneration.from_pretrained(model_id)
        processor = PaliGemmaProcessor.from_pretrained(model_id)
        prompt = "answer en There is no snowman in any of the images. Is this true or false?"
        stop_sign_image = Image.open(
            requests.get("https://www.ilankelman.org/stopsigns/australia.jpg", stream=True).raw
        )
        snow_image = Image.open(
            requests.get(
                "https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/snowman.jpg", stream=True
            ).raw
        )

        inputs = processor(text=prompt, images=[[snow_image, snow_image]], return_tensors="pt")

        output = model.generate(**inputs, max_new_tokens=20)
        EXPECTED_DECODED_TEXT = "answer en There is no snowman in any of the images. Is this true or false?\nFalse"

        self.assertEqual(
            self.processor.decode(output[0], skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

        # try another prompt with two different image this time
        prompt = "answer en There is exactly one snowman. Is this true or false?"
        inputs = processor(text=prompt, images=[[snow_image, stop_sign_image]], return_tensors="pt")
        output = model.generate(**inputs, max_new_tokens=20)
        EXPECTED_DECODED_TEXT = "answer en There is exactly one snowman. Is this true or false?\nTrue"
        self.assertEqual(
            self.processor.decode(output[0], skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    def test_small_model_integration_test_paligemma_VQA(self):
        # Let' s make sure we test the preprocessing to replace what is used
        model_id = "google/paligemma-3b-pt-224"
        model = PaliGemmaForConditionalGeneration.from_pretrained(model_id)
        prompt = "answer en Where is the cow standing?"
        image_file = (
            "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/cow_beach_1.png"
        )
        raw_image = Image.open(requests.get(image_file, stream=True).raw)
        inputs = self.processor(images=raw_image, text=prompt, return_tensors="pt").to(torch.float16)

        output = model.generate(**inputs, max_new_tokens=900, do_sample=False)
        EXPECTED_DECODED_TEXT = "answer en Where is the cow standing?\nbeach"  # fmt: skip

        self.assertEqual(
            self.processor.decode(output[0], skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    def test_small_model_integration_test_paligemma_empty_prompt(self):
        # Let' s make sure we test the preprocessing to replace what is used
        model_id = "google/paligemma-3b-pt-224"
        model = PaliGemmaForConditionalGeneration.from_pretrained(model_id)

        prompt = ""
        image_file = (
            "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/cow_beach_1.png"
        )
        raw_image = Image.open(requests.get(image_file, stream=True).raw)
        inputs = self.processor(images=raw_image, text=prompt, return_tensors="pt").to(torch.float16)

        output = model.generate(**inputs, max_new_tokens=900, do_sample=False)
        EXPECTED_DECODED_TEXT = "\ncow on the beach"  # fmt: skip

        self.assertEqual(
            self.processor.decode(output[0], skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    def test_small_model_integration_test_paligemma_batched(self):
        # Let' s make sure we test the preprocessing to replace what is used
        model_id = "google/paligemma-3b-pt-224"

        model = PaliGemmaForConditionalGeneration.from_pretrained(model_id)

        prompts = [
            "answer en Where is the cow standing?",
            "",
        ]
        image1 = Image.open(
            requests.get(
                "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/cow_beach_1.png",
                stream=True,
            ).raw
        )
        image2 = image1

        inputs = self.processor(images=[image1, image2], text=prompts, return_tensors="pt", padding=True)

        output = model.generate(**inputs, max_new_tokens=20)

        EXPECTED_DECODED_TEXT = ["answer en Where is the cow standing?\nbeach", "\ncow on the beach"]  # fmt: skip

        self.assertEqual(self.processor.batch_decode(output, skip_special_tokens=True), EXPECTED_DECODED_TEXT)

    def test_small_model_integration_test_paligemma_batched_bf16(self):
        # Let' s make sure we test the preprocessing to replace what is used
        model_id = "google/paligemma-3b-pt-224"
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id, revision="bfloat16", torch_dtype=torch.bfloat16
        ).to(torch_device)
        # The first batch is longer in terms of text, the second will be padded.
        prompts = [
            "answer en Where is the cow standing?",
            "",
        ]
        image1 = Image.open(
            requests.get(
                "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/cow_beach_1.png",
                stream=True,
            ).raw
        )
        image2 = image1

        inputs = (
            self.processor(images=[image1, image2], text=prompts, return_tensors="pt", padding=True)
            .to(torch.bfloat16)
            .to(torch_device)
        )
        output = model.generate(**inputs, max_new_tokens=20)

        EXPECTED_DECODED_TEXT = ["answer en Where is the cow standing?\nbeach", "\ncow on the beach"]  # fmt: skip
        self.assertEqual(self.processor.batch_decode(output, skip_special_tokens=True), EXPECTED_DECODED_TEXT)

    def test_small_model_integration_test_paligemma_batched_f16(self):
        # Let' s make sure we test the preprocessing to replace what is used
        model_id = "google/paligemma-3b-pt-224"
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id, revision="float16", torch_dtype=torch.float16
        ).to(torch_device)
        # The first batch is longer in terms of text, the second will be padded.
        prompts = [
            "answer en Where is the cow standing?",
            "",
        ]
        image1 = Image.open(
            requests.get(
                "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/cow_beach_1.png",
                stream=True,
            ).raw
        )
        image2 = image1

        inputs = (
            self.processor(images=[image1, image2], text=prompts, return_tensors="pt", padding=True)
            .to(torch.float16)
            .to(torch_device)
        )

        output = model.generate(**inputs, max_new_tokens=20)

        EXPECTED_DECODED_TEXT = ["answer en Where is the cow standing?\nbeach", "\ncow on the beach"]  # fmt: skip
        self.assertEqual(self.processor.batch_decode(output, skip_special_tokens=True), EXPECTED_DECODED_TEXT)

    def test_integration_detection_bug(self):
        # this is a reproducer of https://github.com/huggingface/transformers/issues/31425 where not enough context
        # impacted negatively segmentation generations.
        model_id = "google/paligemma-3b-pt-224"
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id, revision="bfloat16", torch_dtype=torch.bfloat16
        ).to(torch_device)
        prompt = ("detect shoe",)

        image = Image.open(
            requests.get(
                "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/shoe.png",
                stream=True,
            ).raw
        )

        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(torch.bfloat16).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20)

        EXPECTED_DECODED_TEXT = "detect shoe\n<loc0051><loc0309><loc0708><loc0646> shoe"  # fmt: skip
        self.assertEqual(self.processor.decode(output[0], skip_special_tokens=True), EXPECTED_DECODED_TEXT)

    def test_paligemma_index_error_bug(self):
        # This is a reproducer of https://github.com/huggingface/transformers/pull/28032 and makes sure it does not happen anymore
        # Please refer to that PR, or specifically https://github.com/huggingface/transformers/pull/28032#issuecomment-1860650043 for
        # more details
        model_id = "google/paligemma-3b-pt-224"
        model = PaliGemmaForConditionalGeneration.from_pretrained(model_id)

        # Simulate a super long prompt
        prompt = "\n" * 200
        image_file = (
            "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/cow_beach_1.png"
        )

        raw_image = Image.open(requests.get(image_file, stream=True).raw)
        inputs = self.processor(
            images=raw_image,
            text=prompt,
            return_tensors="pt",
        ).to(torch.float16)

        # Make sure that `generate` works
        _ = model.generate(**inputs, max_new_tokens=20)

    def test_paligemma_finetuning_with_suffixes_bf16(self):
        # this is a supplementary test to ensure paligemma fine-tuning that relies on token_type_ids is robust to future changes
        model_id = "google/paligemma-3b-pt-224"
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id, revision="bfloat16", torch_dtype=torch.bfloat16
        ).to(torch_device)
        # The first batch is longer in terms of text, the second will be padded.
        prompts = [
            "answer en Where is the cow standing?",
            "",
        ]

        suffixes = ["beach", "cow standing on the beach"]
        image1 = Image.open(
            requests.get(
                "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/cow_beach_1.png",
                stream=True,
            ).raw
        )
        image2 = image1

        inputs = (
            self.processor(images=[image1, image2], text=prompts, suffix=suffixes, return_tensors="pt", padding=True)
            .to(torch.bfloat16)
            .to(torch_device)
        )

        expected_labels = torch.tensor(
            [266 * [-100] + [54901, 1], 262 * [-100] + [14706, 9980, 611, 573, 8318, 1]]
        ).to(torch_device)

        assert torch.equal(inputs["labels"], expected_labels)

        expected_token_type_ids = torch.tensor([266 * [0] + 2 * [1], 262 * [0] + 6 * [1]]).to(torch_device)

        assert torch.equal(inputs["token_type_ids"], expected_token_type_ids)

        output = model(**inputs)

        # check that loss does not error out
        _ = output.loss
