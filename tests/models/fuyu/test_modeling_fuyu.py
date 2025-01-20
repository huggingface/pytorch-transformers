# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Fuyu model."""

import io
import unittest

import pytest
import requests
from parameterized import parameterized

from transformers import FuyuConfig, is_torch_available, is_vision_available
from transformers.testing_utils import require_torch, require_torch_accelerator, slow, torch_device
from transformers.utils import cached_property

from ...generation.test_utils import GenerationTesterMixin
from ...test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin


if is_vision_available():
    from PIL import Image


if is_torch_available() and is_vision_available():
    from transformers import FuyuProcessor


if is_torch_available():
    import torch

    from transformers import FuyuForCausalLM


class FuyuModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        image_size=30,
        patch_size=15,
        num_channels=3,
        is_training=True,
        use_input_mask=True,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        type_sequence_label_size=2,
        initializer_range=0.02,
        num_labels=3,
        num_choices=4,
        pad_token_id=0,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.pad_token_id = pad_token_id
        self.scope = scope

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        sequence_labels = None
        token_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)

        config = self.get_config()

        return config, input_ids, input_mask, sequence_labels, token_labels

    def get_config(self):
        return FuyuConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            is_decoder=False,
            initializer_range=self.initializer_range,
            pad_token_id=self.pad_token_id,
        )

    def create_and_check_model(
        self,
        config,
        input_ids,
        input_mask,
        sequence_labels,
        token_labels,
    ):
        model = FuyuForCausalLM(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_model_as_decoder(
        self,
        config,
        input_ids,
        input_mask,
        sequence_labels,
        token_labels,
        encoder_hidden_states,
        encoder_attention_mask,
    ):
        config.add_cross_attention = True
        model = FuyuForCausalLM(config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        result = model(
            input_ids,
            attention_mask=input_mask,
            encoder_hidden_states=encoder_hidden_states,
        )
        result = model(input_ids, attention_mask=input_mask)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_for_causal_lm(
        self,
        config,
        input_ids,
        input_mask,
        sequence_labels,
        token_labels,
        encoder_hidden_states,
        encoder_attention_mask,
    ):
        model = FuyuForCausalLM(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_decoder_model_past_large_inputs(
        self,
        config,
        input_ids,
        input_mask,
        sequence_labels,
        token_labels,
        encoder_hidden_states,
        encoder_attention_mask,
    ):
        config.is_decoder = True
        config.add_cross_attention = True
        model = FuyuForCausalLM(config=config)
        model.to(torch_device)
        model.eval()

        # first forward pass
        outputs = model(
            input_ids,
            attention_mask=input_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values

        # create hypothetical multiple next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 3), config.vocab_size)
        next_mask = ids_tensor((self.batch_size, 3), vocab_size=2)

        # append to next input_ids and
        next_input_ids = torch.cat([input_ids, next_tokens], dim=-1)
        next_attention_mask = torch.cat([input_mask, next_mask], dim=-1)

        output_from_no_past = model(
            next_input_ids,
            attention_mask=next_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_hidden_states=True,
        )["hidden_states"][0]
        output_from_past = model(
            next_tokens,
            attention_mask=next_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            output_hidden_states=True,
        )["hidden_states"][0]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, -3:, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, :, random_slice_idx].detach()

        self.parent.assertTrue(output_from_past_slice.shape[1] == next_tokens.shape[1])

        # test that outputs are equal for slice
        self.parent.assertTrue(torch.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-3))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            input_mask,
            sequence_labels,
            token_labels,
        ) = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_torch
class FuyuModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (FuyuForCausalLM,) if is_torch_available() else ()
    all_generative_model_classes = (FuyuForCausalLM,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {"text-generation": FuyuForCausalLM, "image-text-to-text": FuyuForCausalLM} if is_torch_available() else {}
    )

    test_head_masking = False
    test_pruning = False
    test_cpu_offload = False
    test_disk_offload = False
    test_model_parallel = False

    def setUp(self):
        self.model_tester = FuyuModelTester(self)

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

    @pytest.mark.generate
    @parameterized.expand([("random",), ("same",)])
    @unittest.skip("Fuyu doesn't support assisted generation due to the need to crop/extend image patches indices")
    def test_assisted_decoding_matches_greedy_search(self):
        pass

    @unittest.skip("Fuyu doesn't support assisted generation due to the need to crop/extend image patches indices")
    def test_assisted_decoding_sample(self):
        pass

    # TODO: Fix me (once this model gets more usage)
    @unittest.skip(reason="Does not work on the tiny model.")
    def test_disk_offload_bin(self):
        super().test_disk_offload()

    # TODO: Fix me (once this model gets more usage)
    @unittest.skip(reason="Does not work on the tiny model.")
    def test_disk_offload_safetensors(self):
        super().test_disk_offload()

    # TODO: Fix me (once this model gets more usage)
    @unittest.skip(reason="Does not work on the tiny model.")
    def test_model_parallelism(self):
        super().test_model_parallelism()


@slow
@require_torch_accelerator
class FuyuModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_processor(self):
        return FuyuProcessor.from_pretrained("adept/fuyu-8b")

    @cached_property
    def default_model(self):
        return FuyuForCausalLM.from_pretrained("adept/fuyu-8b")

    def test_greedy_generation(self):
        processor = self.default_processor
        model = self.default_model

        url = "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/bus.png"
        image = Image.open(io.BytesIO(requests.get(url).content))

        text_prompt_coco_captioning = "Generate a coco-style caption.\n"

        inputs = processor(images=image, text=text_prompt_coco_captioning, return_tensors="pt")
        generated_ids = model.generate(**inputs, max_new_tokens=10)

        # take the last 8 tokens (in order to skip special \n\x04 characters) and decode them
        generated_text = processor.batch_decode(generated_ids[:, -8:], skip_special_tokens=True)[0]
        self.assertEqual(generated_text, "A blue bus parked on the side of a road.")


"""
    @slow
    @require_torch_accelerator
    def test_model_8b_chat_greedy_generation_bus_color(self):
        EXPECTED_TEXT_COMPLETION = "The bus is blue.\n|ENDOFTEXT|"
        text_prompt_bus_color = "What color is the bus?\n"
        model_inputs_bus_color = self.processor(text=text_prompt_bus_color, images=self.bus_image_pil)

        generated_tokens = self.model.generate(**model_inputs_bus_color, max_new_tokens=10)
        text = self.processor.tokenizer.batch_decode(generated_tokens)
        end_sequence = text[0].split("\x04")[1]
        clean_sequence = (
            end_sequence[: end_sequence.find("|ENDOFTEXT|") + len("|ENDOFTEXT|")]
            if "|ENDOFTEXT|" in end_sequence
            else end_sequence
        )
        self.assertEqual(EXPECTED_TEXT_COMPLETION, clean_sequence)

    @slow
    @require_torch_accelerator
    def test_model_8b_chat_greedy_generation_chart_vqa(self):
        EXPECTED_TEXT_TOKENS = ["The","life expectancy","at","birth","of male","s in","","20","18","is","","80",".","7",".","\n","|ENDOFTEXT|",]  # fmt: skip
        expected_text_completion = " ".join(EXPECTED_TEXT_TOKENS)  # TODO make sure the end string matches

        text_prompt_chart_vqa = "What is the highest life expectancy at birth of male?\n"

        chart_image_url = (
            "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/chart.png"
        )
        chart_image_pil = Image.open(io.BytesIO(requests.get(chart_image_url).content))

        model_inputs_chart_vqa = self.processor(text=text_prompt_chart_vqa, images=chart_image_pil)
        generated_tokens = self.model.generate(**model_inputs_chart_vqa, max_new_tokens=10)
        text = self.processor.tokenizer.batch_decode(generated_tokens)
        end_sequence = text[0].split("\x04")[1]
        clean_sequence = (
            end_sequence[: end_sequence.find("|ENDOFTEXT|") + len("|ENDOFTEXT|")]
            if "|ENDOFTEXT|" in end_sequence
            else end_sequence
        )
        self.assertEqual(expected_text_completion, clean_sequence)

    @slow
    @require_torch_accelerator
    def test_model_8b_chat_greedy_generation_bounding_box(self):
        EXPECTED_TEXT_COMPLETION = "\x00194213202244\x01|ENDOFTEXT|"
        text_prompt_bbox = "When presented with a box, perform OCR to extract text contained within it. If provided with text, generate the corresponding bounding box.\\nWilliams"  # noqa: E231

        bbox_image_url = "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/bbox_sample_image.png"
        bbox_image_pil = Image.open(io.BytesIO(requests.get(bbox_image_url).content))

        model_inputs_bbox = self.processor(text=text_prompt_bbox, images=bbox_image_pil)
        generated_tokens = self.model.generate(**model_inputs_bbox, max_new_tokens=10)
        text = self.processor.tokenizer.batch_decode(generated_tokens)
        end_sequence = text[0].split("\x04")[1]
        clean_sequence = (
            end_sequence[: end_sequence.find("|ENDOFTEXT|") + len("|ENDOFTEXT|")]
            if "|ENDOFTEXT|" in end_sequence
            else end_sequence
        )
        self.assertEqual(EXPECTED_TEXT_COMPLETION, clean_sequence)
"""
