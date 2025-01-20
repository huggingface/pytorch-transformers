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
"""Testing suite for the PyTorch emu3 model."""

import unittest

import numpy as np
import pytest
import requests
from huggingface_hub import hf_hub_download
from parameterized import parameterized

from transformers import Emu3Config, Emu3TextConfig, is_torch_available, is_vision_available, set_seed
from transformers.testing_utils import (
    require_bitsandbytes,
    require_torch,
    require_torch_large_gpu,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_vision_available():
    from PIL import Image

if is_torch_available():
    import torch

    from transformers import (
        Emu3ForCausalLM,
        Emu3ForConditionalGeneration,
        Emu3Processor,
        Emu3TextModel,
    )


class Emu3Text2TextModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=False,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=37,
        max_position_embeddings=512,
        initializer_range=0.02,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        attention_mask = input_ids.ne(1).to(torch_device)

        config = self.get_config()

        return config, input_ids, attention_mask

    def get_config(self):
        return Emu3TextConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            intermediate_size=self.intermediate_size,
            max_position_embeddings=self.max_position_embeddings,
            is_decoder=False,
            initializer_range=self.initializer_range,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            attention_mask,
        ) = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": attention_mask}
        return config, inputs_dict


@require_torch
class Emu3Text2TextModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (Emu3ForCausalLM,) if is_torch_available() else ()
    all_generative_model_classes = (Emu3ForCausalLM,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "text-generation": Emu3ForCausalLM,
        }
        if is_torch_available()
        else {}
    )
    test_headmasking = False
    test_pruning = False
    fx_compatible = False

    def setUp(self):
        self.model_tester = Emu3Text2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Emu3TextConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    @parameterized.expand([("linear",), ("dynamic",)])
    def test_model_rope_scaling(self, scaling_type):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        short_input = ids_tensor([1, 10], config.vocab_size)
        long_input = ids_tensor([1, int(config.max_position_embeddings * 1.5)], config.vocab_size)

        set_seed(42)  # Fixed seed at init time so the two models get the same random weights
        original_model = Emu3TextModel(config)
        original_model.to(torch_device)
        original_model.eval()
        original_short_output = original_model(short_input).last_hidden_state
        original_long_output = original_model(long_input).last_hidden_state

        set_seed(42)  # Fixed seed at init time so the two models get the same random weights
        config.rope_scaling = {"type": scaling_type, "factor": 10.0}
        scaled_model = Emu3TextModel(config)
        scaled_model.to(torch_device)
        scaled_model.eval()
        scaled_short_output = scaled_model(short_input).last_hidden_state
        scaled_long_output = scaled_model(long_input).last_hidden_state

        # Dynamic scaling does not change the RoPE embeddings until it receives an input longer than the original
        # maximum sequence length, so the outputs for the short input should match.
        if scaling_type == "dynamic":
            self.assertTrue(torch.allclose(original_short_output, scaled_short_output, atol=1e-5))
        else:
            self.assertFalse(torch.allclose(original_short_output, scaled_short_output, atol=1e-5))

        # The output should be different for long inputs
        self.assertFalse(torch.allclose(original_long_output, scaled_long_output, atol=1e-5))

    @unittest.skip("Doesn't work, tensors are not almost same")  # TODO raushan fixme
    def test_custom_4d_attention_mask(self):
        pass


class Emu3Vision2TextModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=False,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=37,
        max_position_embeddings=512,
        initializer_range=0.02,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        image_token_id=3,
        image_size=30,
        codebook_size=20,
        temporal_downsample_factor=1,
        base_channels=32,
        vq_channel_multiplier=[1, 1],
        image_seq_length=100,
        vq_img_token_start_id=3,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.is_training = is_training
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.image_token_id = image_token_id
        self.image_size = image_size
        self.codebook_size = codebook_size
        self.temporal_downsample_factor = temporal_downsample_factor
        self.vq_channel_multiplier = vq_channel_multiplier
        self.vq_img_token_start_id = vq_img_token_start_id
        self.base_channels = base_channels
        self.seq_length = seq_length + image_seq_length
        self.image_seq_length = image_seq_length

    def prepare_config_and_inputs(self):
        config = self.get_config()

        input_ids = ids_tensor([self.batch_size, self.seq_length], config.text_config.vocab_size)
        attention_mask = input_ids.ne(1).to(torch_device)
        input_ids[input_ids == self.image_token_id] = self.pad_token_id
        input_ids[:, : self.image_seq_length] = self.image_token_id

        pixel_values = floats_tensor(
            [
                self.batch_size,
                3,
                self.image_size,
                self.image_size,
            ]
        )
        image_sizes = [[self.image_size, self.image_size]] * self.batch_size
        image_sizes = torch.tensor(image_sizes, device=torch_device, dtype=torch.int64)

        return config, input_ids, attention_mask, pixel_values, image_sizes

    def get_config(self):
        # create dummy vocab map for image2bpe mapping if it needs remapping
        # we assume that vocab size is big enough to account for `codebook_size` amount of
        # image tokens somewhere at the beginning of total vocab size

        vocab_map = {i: chr(i) for i in range(self.vocab_size)}
        start = self.vq_img_token_start_id
        end = self.vq_img_token_start_id + self.codebook_size
        for i in range(start, end):
            # dummy str for each token, anything that fits pattern "<|visual token XXXXXX|>"
            vocab_map[i] = f"<|visual token{i:06d}|>"

        # add tokens that have to be in the vocab, we'll retrieve their ids later in modeling code
        vocab_map[self.image_token_id] = "<image>"
        vocab_map[self.image_token_id + 1] = "<|extra_200|>"
        vocab_map = {v: k for k, v in vocab_map.items()}

        text_config = Emu3TextConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            intermediate_size=self.intermediate_size,
            max_position_embeddings=self.max_position_embeddings,
            initializer_range=self.initializer_range,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
        )

        vq_config = {
            "codebook_size": self.codebook_size,
            "temporal_downsample_factor": self.temporal_downsample_factor,
            "base_channels": self.base_channels,
            "channel_multiplier": self.vq_channel_multiplier,
            "hidden_size": self.base_channels,
        }
        return Emu3Config(text_config=text_config, vq_config=vq_config, vocabulary_map=vocab_map)

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            attention_mask,
            pixel_values,
            image_sizes,
        ) = config_and_inputs
        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_sizes": image_sizes,
        }
        return config, inputs_dict


@require_torch
class Emu3Vision2TextModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (Emu3ForConditionalGeneration,) if is_torch_available() else ()
    all_generative_model_classes = (Emu3ForConditionalGeneration,) if is_torch_available() else ()
    pipeline_model_mapping = {}
    test_headmasking = False
    test_pruning = False
    fx_compatible = False

    def setUp(self):
        self.model_tester = Emu3Vision2TextModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=Emu3Config, has_text_modality=False, common_properties=["vocabulary_map"]
        )

    def test_config(self):
        self.config_tester.run_common_tests()

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

    @unittest.skip(
        "Emu3 has a VQ module that uses `weight.data` directly in forward which prevent offloding on that module"
    )
    def test_disk_offload_safetensors(self):
        pass

    @unittest.skip(
        "Emu3 has a VQ module that uses `weight.data` directly in forward which prevent offloding on that module"
    )
    def test_disk_offload_bin(self):
        pass

    @unittest.skip(
        "Emu3 has a VQ module that uses `weight.data` directly in forward which prevent offloding on that module"
    )
    def test_cpu_offload(self):
        pass

    @unittest.skip("Doesn't work, tensors are not almost same")  # TODO raushan fixme
    def test_custom_4d_attention_mask(self):
        pass

    @unittest.skip("VQ-VAE module doesn't initialize weights properly")
    def test_initialization(self):
        pass

    @pytest.mark.generate
    @unittest.skip("Emu3 has dynamic control flow in vision backbone")
    def test_generate_with_static_cache(self):
        pass


@require_torch
class Emu3IntegrationTest(unittest.TestCase):
    @slow
    @require_bitsandbytes
    def test_model_generation(self):
        model = Emu3ForConditionalGeneration.from_pretrained("BAAI/Emu3-Chat-hf", load_in_4bit=True)
        processor = Emu3Processor.from_pretrained("BAAI/Emu3-Chat-hf")

        image = Image.open(requests.get("https://picsum.photos/id/237/200/200", stream=True).raw)
        prompt = "USER: <image>Describe what do you see here and tell me about the history behind it? ASSISTANT:"

        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device, torch.float16)

        # greedy generation outputs
        EXPECTED_TEXT_COMPLETION = ['USER: 64*64Describe what do you see here and tell me about the history behind it? ASSISTANT: The image captures a moment of tranquility with a black Labrador Retriever resting on a wooden floor. The dog, with its glossy black coat, is lying down with its front legs stretched out in']  # fmt: skip
        generated_ids = model.generate(**inputs, max_new_tokens=40, do_sample=False)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, text)

    @slow
    @require_bitsandbytes
    @require_torch_large_gpu
    def test_model_generation_batched(self):
        model = Emu3ForConditionalGeneration.from_pretrained("BAAI/Emu3-Chat-hf", load_in_4bit=True)
        processor = Emu3Processor.from_pretrained("BAAI/Emu3-Chat-hf")
        processor.tokenizer.padding_side = "left"

        image = Image.open(requests.get("https://picsum.photos/id/237/50/50", stream=True).raw)
        image_2 = Image.open(requests.get("https://picsum.photos/id/247/50/50", stream=True).raw)
        prompts = [
            "USER: <image>Describe what do you see here? ASSISTANT:",
            "USER: <image>What can you say about the image? ASSISTANT:",
        ]

        inputs = processor(images=[image, image_2], text=prompts, padding=True, return_tensors="pt").to(
            model.device, torch.float16
        )

        # greedy generation outputs
        EXPECTED_TEXT_COMPLETION = [
            "USER: 64*64Describe what do you see here? ASSISTANT: The image depicts a black panther in a crouched position. The panther's body is elongated and curved, with its head lowered and ears pointed forward, suggesting alertness or focus.",
            'USER: 64*64What can you say about the image? ASSISTANT: The image depicts a serene natural landscape. The foreground consists of a grassy area with some patches of bare earth. The middle ground shows a steep, reddish-brown cliff, which could be a'
        ]  # fmt: skip
        generated_ids = model.generate(**inputs, max_new_tokens=40, do_sample=False)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, text)

    @slow
    @require_bitsandbytes
    @require_torch_large_gpu
    def test_model_generation_multi_image(self):
        model = Emu3ForConditionalGeneration.from_pretrained("BAAI/Emu3-Chat-hf", load_in_4bit=True)
        processor = Emu3Processor.from_pretrained("BAAI/Emu3-Chat-hf")

        image = Image.open(requests.get("https://picsum.photos/id/237/50/50", stream=True).raw)
        image_2 = Image.open(requests.get("https://picsum.photos/id/247/50/50", stream=True).raw)
        prompt = "USER: <image><image>What do these two images have in common? ASSISTANT:"

        inputs = processor(images=[image, image_2], text=prompt, return_tensors="pt").to(model.device, torch.float16)

        # greedy generation outputs
        EXPECTED_TEXT_COMPLETION = ["USER: 64*6464*64What do these two images have in common? ASSISTANT: Both images feature a black animal, but they are not the same animal. The top image shows a close-up of a black cow's head, while the bottom image depicts a black cow in a natural"]  # fmt: skip
        generated_ids = model.generate(**inputs, max_new_tokens=40, do_sample=False)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, text)

    @slow
    @require_bitsandbytes
    @require_torch_large_gpu
    def test_model_generate_images(self):
        model = Emu3ForConditionalGeneration.from_pretrained("BAAI/Emu3-Gen-hf", load_in_4bit=True)
        processor = Emu3Processor.from_pretrained("BAAI/Emu3-Gen-hf")

        inputs = processor(
            text=["a portrait of young girl. masterpiece, film grained, best quality."],
            padding=True,
            return_tensors="pt",
            return_for_image_generation=True,
            image_area=1600,
        ).to(model.device)
        self.assertTrue(inputs.input_ids.shape[1] == 21)

        image_sizes = inputs.pop("image_sizes")
        HEIGHT, WIDTH = image_sizes[0]
        VISUAL_TOKENS = model.vocabulary_mapping.image_tokens

        def prefix_allowed_tokens_fn(batch_id, input_ids):
            height, width = HEIGHT, WIDTH
            visual_tokens = VISUAL_TOKENS
            image_wrapper_token_id = torch.tensor([processor.tokenizer.image_wrapper_token_id], device=model.device)
            eoi_token_id = torch.tensor([processor.tokenizer.eoi_token_id], device=model.device)
            eos_token_id = torch.tensor([processor.tokenizer.eos_token_id], device=model.device)
            pad_token_id = torch.tensor([processor.tokenizer.pad_token_id], device=model.device)
            eof_token_id = torch.tensor([processor.tokenizer.eof_token_id], device=model.device)
            eol_token_id = processor.tokenizer.encode("<|extra_200|>", return_tensors="pt")[0]

            position = torch.nonzero(input_ids == image_wrapper_token_id, as_tuple=True)[0][0]
            offset = input_ids.shape[0] - position
            if offset % (width + 1) == 0:
                return (eol_token_id,)
            elif offset == (width + 1) * height + 1:
                return (eof_token_id,)
            elif offset == (width + 1) * height + 2:
                return (eoi_token_id,)
            elif offset == (width + 1) * height + 3:
                return (eos_token_id,)
            elif offset > (width + 1) * height + 3:
                return (pad_token_id,)
            else:
                return visual_tokens

        out = model.generate(
            **inputs,
            max_new_tokens=200,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            do_sample=False,
        )
        self.assertTrue(out.shape[1] == 54)

        image = model.decode_image_tokens(out[:, inputs.input_ids.shape[1] :], height=HEIGHT, width=WIDTH)
        images = processor.postprocess(list(image.float()), return_tensors="np")
        self.assertTrue(images["pixel_values"].shape == (3, 40, 40))
        self.assertTrue(isinstance(images["pixel_values"], np.ndarray))

        filepath = hf_hub_download(
            repo_id="raushan-testing-hf/images_test",
            filename="emu3_image.npy",
            repo_type="dataset",
        )
        original_pixels = np.load(filepath)
        self.assertTrue(np.allclose(original_pixels, images["pixel_values"]))
