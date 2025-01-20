# coding=utf-8
# Copyright 2023 Microsoft Research and The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch KOSMOS-2 model."""

import copy
import inspect
import os
import tempfile
import unittest

import numpy as np
import pytest
import requests
from parameterized import parameterized

from transformers import AutoModelForImageTextToText, AutoProcessor, Kosmos2Config
from transformers.models.kosmos2.configuration_kosmos2 import Kosmos2TextConfig, Kosmos2VisionConfig
from transformers.testing_utils import (
    IS_ROCM_SYSTEM,
    require_torch,
    require_vision,
    slow,
    torch_device,
)
from transformers.utils import (
    is_torch_available,
    is_vision_available,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    ModelTesterMixin,
    _config_zero_init,
    floats_tensor,
    ids_tensor,
    random_attention_mask,
)
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import Kosmos2ForConditionalGeneration, Kosmos2Model


if is_vision_available():
    from PIL import Image


class Kosmos2VisionModelTester:
    def __init__(
        self,
        parent,
        batch_size=12,
        image_size=32,
        patch_size=4,
        num_channels=3,
        is_training=True,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        dropout=0.1,
        attention_dropout=0.1,
        initializer_range=1e-10,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.is_training = is_training
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.scope = scope

        # in ViT, the seq length equals the number of patches + 1 (we add 1 for the [CLS] token)
        num_patches = (image_size // patch_size) ** 2
        self.seq_length = num_patches + 1

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        config = self.get_config()

        return config, pixel_values

    def get_config(self):
        return Kosmos2VisionConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            initializer_range=self.initializer_range,
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


class Kosmos2TextModelTester:
    def __init__(
        self,
        parent,
        batch_size=12,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        dropout=0.1,
        attention_dropout=0.1,
        max_position_embeddings=512,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.max_position_embeddings = max_position_embeddings
        self.scope = scope

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        if input_mask is not None:
            batch_size, seq_length = input_mask.shape
            rnd_start_indices = np.random.randint(1, seq_length - 1, size=(batch_size,))
            for batch_idx, start_index in enumerate(rnd_start_indices):
                input_mask[batch_idx, :start_index] = 1
                input_mask[batch_idx, start_index:] = 0

        config = self.get_config()

        return config, input_ids, input_mask

    def get_config(self):
        return Kosmos2TextConfig(
            vocab_size=self.vocab_size,
            embed_dim=self.hidden_size,
            layers=self.num_hidden_layers,
            attention_heads=self.num_attention_heads,
            ffn_dim=self.intermediate_size,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            max_position_embeddings=self.max_position_embeddings,
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, input_mask = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


class Kosmos2ModelTester:
    def __init__(self, parent, text_kwargs=None, vision_kwargs=None, latent_query_num=3, is_training=True):
        if text_kwargs is None:
            text_kwargs = {}
        if vision_kwargs is None:
            vision_kwargs = {}

        self.parent = parent
        self.text_model_tester = Kosmos2TextModelTester(parent, **text_kwargs)
        self.vision_model_tester = Kosmos2VisionModelTester(parent, **vision_kwargs)
        self.batch_size = self.text_model_tester.batch_size  # need bs for batching_equivalence test
        self.seq_length = self.text_model_tester.seq_length
        self.latent_query_num = latent_query_num
        self.is_training = is_training

    def prepare_config_and_inputs(self):
        text_config, input_ids, attention_mask = self.text_model_tester.prepare_config_and_inputs()
        vision_config, pixel_values = self.vision_model_tester.prepare_config_and_inputs()

        # build `image_embeds_position_mask`
        image_embeds_position_mask = torch.zeros_like(input_ids)
        image_embeds_position_mask[:, 1 : 1 + self.latent_query_num :] = 1

        config = self.get_config()

        return config, input_ids, attention_mask, image_embeds_position_mask, pixel_values

    def get_config(self):
        return Kosmos2Config(
            self.text_model_tester.get_config().to_dict(),
            self.vision_model_tester.get_config().to_dict(),
            latent_query_num=self.latent_query_num,
        )

    def create_and_check_model(self, config, input_ids, attention_mask, image_embeds_position_mask, pixel_values):
        model = Kosmos2Model(config).to(torch_device).eval()
        with torch.no_grad():
            result = model(pixel_values, input_ids, image_embeds_position_mask, attention_mask)
        self.parent.assertEqual(
            result.last_hidden_state.shape,
            (self.text_model_tester.batch_size, self.text_model_tester.seq_length, self.text_model_tester.hidden_size),
        )
        self.parent.assertEqual(
            result.image_embeds.shape,
            (self.text_model_tester.batch_size, self.latent_query_num, self.text_model_tester.hidden_size),
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, attention_mask, image_embeds_position_mask, pixel_values = config_and_inputs
        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image_embeds_position_mask": image_embeds_position_mask,
            "pixel_values": pixel_values,
        }
        return config, inputs_dict


@require_torch
class Kosmos2ModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (Kosmos2Model, Kosmos2ForConditionalGeneration) if is_torch_available() else ()
    all_generative_model_classes = (Kosmos2ForConditionalGeneration,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": Kosmos2Model,
            "image-to-text": Kosmos2ForConditionalGeneration,
            "image-text-to-text": Kosmos2ForConditionalGeneration,
        }
        if is_torch_available()
        else {}
    )
    fx_compatible = False
    test_head_masking = False
    test_pruning = False
    test_resize_embeddings = False
    test_attention_outputs = False
    _is_composite = True

    # TODO: `image-to-text` pipeline for this model needs Processor.
    # TODO: Tiny model needs fixing for `image-text-to-text` (latent_query_num=3 not compatible with num_image_tokens=64).
    def is_pipeline_test_to_skip(
        self,
        pipeline_test_case_name,
        config_class,
        model_architecture,
        tokenizer_name,
        image_processor_name,
        feature_extractor_name,
        processor_name,
    ):
        return (
            pipeline_test_case_name == "ImageToTextPipelineTests"
            or pipeline_test_case_name == "ImageTextToTextPipelineTests"
        )

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = copy.deepcopy(inputs_dict)

        if return_labels:
            if model_class.__name__ == "Kosmos2ForConditionalGeneration":
                inputs_dict["labels"] = torch.zeros(
                    (self.model_tester.text_model_tester.batch_size, self.model_tester.text_model_tester.seq_length),
                    dtype=torch.long,
                    device=torch_device,
                )

        return inputs_dict

    def setUp(self):
        self.model_tester = Kosmos2ModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=Kosmos2Config, has_text_modality=False, common_properties=["latent_query_num"]
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    # overwrite from common to skip `image_to_text_projection.latent_query`
    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if name == "image_to_text_projection.latent_query":
                        # The original code use ` nn.Parameter(torch.randn(...))` for which this test won't pass.
                        continue
                    self.assertIn(
                        ((param.data.mean() * 1e9).round() / 1e9).item(),
                        [0.0, 1.0],
                        msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                    )

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["pixel_values"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_load_save_without_tied_weights(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        config.text_config.tie_word_embeddings = False
        for model_class in self.all_model_classes:
            model = model_class(config)
            with tempfile.TemporaryDirectory() as d:
                model.save_pretrained(d)

                model_reloaded, infos = model_class.from_pretrained(d, output_loading_info=True)
                # Checking the state dicts are correct
                reloaded_state = model_reloaded.state_dict()
                for k, v in model.state_dict().items():
                    self.assertIn(k, reloaded_state, f"Key {k} is missing from reloaded")
                    torch.testing.assert_close(
                        v, reloaded_state[k], msg=lambda x: f"{model_class.__name__}: Tensor {k}: {x}"
                    )
                # Checking there was no complain of missing weights
                self.assertEqual(infos["missing_keys"], [])

    # overwrite from common in order to use `self.model_tester.text_model_tester.num_hidden_layers`
    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.hidden_states

            expected_num_layers = getattr(
                self.model_tester,
                "expected_num_hidden_layers",
                self.model_tester.text_model_tester.num_hidden_layers + 1,
            )
            self.assertEqual(len(hidden_states), expected_num_layers)

            seq_length = self.model_tester.text_model_tester.seq_length

            self.assertListEqual(
                list(hidden_states[0].shape[-2:]),
                [seq_length, self.model_tester.text_model_tester.hidden_size],
            )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True

            check_hidden_states_output(inputs_dict, config, model_class)

    # overwrite from common in order to use `config.text_config.vocab_size` instead of `config.vocab_size`
    def test_tie_model_weights(self):
        if not self.test_torchscript:
            self.skipTest(reason="test_torchscript is set to False")

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def check_same_values(layer_1, layer_2):
            equal = True
            for p1, p2 in zip(layer_1.weight, layer_2.weight):
                if p1.data.ne(p2.data).sum() > 0:
                    equal = False
            return equal

        for model_class in self.all_model_classes:
            config.torchscript = True
            model_not_tied = model_class(config)
            if model_not_tied.get_output_embeddings() is None:
                continue

            config_tied = copy.deepcopy(config)
            config_tied.torchscript = False
            model_tied = model_class(config_tied)
            params_tied = list(model_tied.parameters())
            # Check that the embedding layer and decoding layer are the same in size and in value
            # self.assertTrue(check_same_values(embeddings, decoding))

            # # Check that after modification, they remain the same.
            # embeddings.weight.data.div_(2)
            # # Check that the embedding layer and decoding layer are the same in size and in value
            # self.assertTrue(embeddings.weight.shape, decoding.weight.shape)
            # self.assertTrue(check_same_values(embeddings, decoding))

            # # Check that after modification, they remain the same.
            # decoding.weight.data.div_(4)
            # # Check that the embedding layer and decoding layer are the same in size and in value
            # self.assertTrue(embeddings.weight.shape, decoding.weight.shape)
            # self.assertTrue(check_same_values(embeddings, decoding))

            # Check that after resize they remain tied.
            model_tied.resize_token_embeddings(config.text_config.vocab_size + 10)
            params_tied_2 = list(model_tied.parameters())
            self.assertEqual(len(params_tied_2), len(params_tied))

            # decoding.weight.data.mul_(20)
            # # Check that the embedding layer and decoding layer are the same in size and in value
            # self.assertTrue(model.transformer.wte.weight.shape, model.lm_head.weight.shape)
            # self.assertTrue(check_same_values(model.transformer.wte, model.lm_head))

    @pytest.mark.generate
    @parameterized.expand([("greedy", 1), ("beam search", 2)])
    @unittest.skip(
        "KOSMOS-2 doesn't support inputs embeds. The test isn't skipped by checking input args because KOSMOS-2 has `generate()` overwritten"
    )
    def test_generate_from_inputs_embeds(self):
        pass

    @pytest.mark.generate
    def test_left_padding_compatibility(self):
        # Overwrite because Kosmos-2 need to padd pixel values and pad image-attn-mask

        def _prepare_model_kwargs(input_ids, attention_mask, pad_size, signature):
            model_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if "position_ids" in signature:
                position_ids = torch.cumsum(attention_mask, dim=-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                model_kwargs["position_ids"] = position_ids
            if "cache_position" in signature:
                cache_position = torch.arange(input_ids.shape[-1], device=torch_device)
                model_kwargs["cache_position"] = cache_position
            if "image_embeds_position_mask" in signature:
                image_embeds_position_mask = torch.zeros_like(input_ids)
                image_embeds_position_mask[:, (pad_size + 1) : pad_size + 1 + self.model_tester.latent_query_num] = 1
                model_kwargs["image_embeds_position_mask"] = image_embeds_position_mask
            return model_kwargs

        for model_class in self.all_generative_model_classes:
            config, inputs_dict = self.prepare_config_and_inputs_for_generate()
            input_ids = inputs_dict["input_ids"]
            pixel_values = inputs_dict["pixel_values"]
            attention_mask = inputs_dict.get("attention_mask")
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)

            model = model_class(config).to(torch_device).eval()
            signature = inspect.signature(model.forward).parameters.keys()

            # no cache as some models require special cache classes to be init outside forward
            model.generation_config.use_cache = False

            # Without padding
            model_kwargs = _prepare_model_kwargs(input_ids, attention_mask, pad_size=0, signature=signature)
            next_logits_wo_padding = model(**model_kwargs, pixel_values=pixel_values).logits[:, -1, :]

            # With left-padding (length 32)
            # can hardcode pad_token to be 0 as we'll do attn masking anyway
            pad_token_id = (
                config.get_text_config().pad_token_id if config.get_text_config().pad_token_id is not None else 0
            )
            pad_size = (input_ids.shape[0], 32)
            padding = torch.ones(pad_size, dtype=input_ids.dtype, device=torch_device) * pad_token_id
            padded_input_ids = torch.cat((padding, input_ids), dim=1)
            padded_attention_mask = torch.cat((torch.zeros_like(padding), attention_mask), dim=1)
            model_kwargs = _prepare_model_kwargs(
                padded_input_ids, padded_attention_mask, pad_size=32, signature=signature
            )
            next_logits_with_padding = model(**model_kwargs, pixel_values=pixel_values).logits[:, -1, :]

            # They should result in very similar logits
            self.assertTrue(torch.allclose(next_logits_wo_padding, next_logits_with_padding, atol=1e-3))

    @slow
    def test_model_from_pretrained(self):
        model_name = "microsoft/kosmos-2-patch14-224"
        model = Kosmos2Model.from_pretrained(model_name)
        self.assertIsNotNone(model)

    def _create_and_check_torchscript(self, config, inputs_dict):
        if not self.test_torchscript:
            self.skipTest(reason="test_torchscript is set to False")

        configs_no_init = _config_zero_init(config)  # To be sure we have no Nan
        configs_no_init.torchscript = True
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            model.to(torch_device)
            model.eval()
            inputs = self._prepare_for_class(inputs_dict, model_class)

            main_input_name = model_class.main_input_name

            try:
                main_input = inputs[main_input_name]
                model(main_input, inputs["input_ids"], inputs["image_embeds_position_mask"])
                traced_model = torch.jit.trace(
                    model, (main_input, inputs["input_ids"], inputs["image_embeds_position_mask"])
                )
            except RuntimeError:
                self.fail("Couldn't trace module.")

            with tempfile.TemporaryDirectory() as tmp_dir_name:
                pt_file_name = os.path.join(tmp_dir_name, "traced_model.pt")

                try:
                    torch.jit.save(traced_model, pt_file_name)
                except Exception:
                    self.fail("Couldn't save module.")

                try:
                    loaded_model = torch.jit.load(pt_file_name)
                except Exception:
                    self.fail("Couldn't load module.")

            model.to(torch_device)
            model.eval()

            loaded_model.to(torch_device)
            loaded_model.eval()

            model_state_dict = model.state_dict()
            loaded_model_state_dict = loaded_model.state_dict()

            non_persistent_buffers = {}
            for key in loaded_model_state_dict.keys():
                if key not in model_state_dict.keys():
                    non_persistent_buffers[key] = loaded_model_state_dict[key]

            loaded_model_state_dict = {
                key: value for key, value in loaded_model_state_dict.items() if key not in non_persistent_buffers
            }

            self.assertEqual(set(model_state_dict.keys()), set(loaded_model_state_dict.keys()))

            model_buffers = list(model.buffers())
            for non_persistent_buffer in non_persistent_buffers.values():
                found_buffer = False
                for i, model_buffer in enumerate(model_buffers):
                    if torch.equal(non_persistent_buffer, model_buffer):
                        found_buffer = True
                        break

                self.assertTrue(found_buffer)
                model_buffers.pop(i)

            models_equal = True
            for layer_name, p1 in model_state_dict.items():
                if layer_name in loaded_model_state_dict:
                    p2 = loaded_model_state_dict[layer_name]
                    if p1.data.ne(p2.data).sum() > 0:
                        models_equal = False

            self.assertTrue(models_equal)

            # Avoid memory leak. Without this, each call increase RAM usage by ~20MB.
            # (Even with this call, there are still memory leak by ~0.04MB)
            self.clear_torch_jit_class_registry()


# We will verify our results on an image of cute cats
def prepare_img():
    url = "https://huggingface.co/hf-internal-testing/Kosmos2-test-image/resolve/main/demo.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


@require_vision
@require_torch
@slow
class Kosmos2ModelIntegrationTest(unittest.TestCase):
    def run_example(self, prompt, image, model, processor):
        inputs = processor(text=prompt, images=image, return_tensors="pt", padding=True).to(torch_device)

        generation_outputs = model.generate(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_embeds=None,
            image_embeds_position_mask=inputs["image_embeds_position_mask"],
            use_cache=True,
            max_new_tokens=128,
            output_scores=True,
            return_dict_in_generate=True,
        )

        scores = generation_outputs.scores
        generated_ids = generation_outputs.sequences
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        # Specify `cleanup_and_extract=False` in order to see the raw model generation.
        processed_text = [processor.post_process_generation(x, cleanup_and_extract=False) for x in generated_text]
        # By default, the generated  text is cleanup and the entities are extracted.
        final_text_with_entities = [processor.post_process_generation(x) for x in generated_text]

        return scores, generated_ids, generated_text, processed_text, final_text_with_entities

    def test_snowman_image_captioning(self):
        url = "https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/snowman.png"

        image = Image.open(requests.get(url, stream=True).raw)
        image.save("new_image.jpg")
        image = Image.open("new_image.jpg")

        model = AutoModelForImageTextToText.from_pretrained("microsoft/kosmos-2-patch14-224").to(torch_device)
        processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")

        prompt = "<grounding>An image of"
        scores, generated_ids, generated_text, processed_text, final_text_with_entities = self.run_example(
            prompt, image, model, processor
        )
        processed_text = processed_text[0]
        final_text, entities = final_text_with_entities[0]

        atol = 1e-4 if IS_ROCM_SYSTEM else 1e-5

        np.testing.assert_allclose(
            torch.concat(scores[1:4])[:3, :3].to("cpu").numpy(),
            np.array(
                [
                    [-1.5672581195831299, -5.007406711578369, 4.36448860168457],
                    [-2.147017002105713, -4.966302871704102, 4.592559337615967],
                    [-0.9352350831031799, -4.688288688659668, 6.240612983703613],
                ]
            ),
            atol=atol,
        )
        np.testing.assert_allclose(
            torch.concat(scores[-3:])[-3:, -3:].to("cpu").numpy(),
            np.array(
                [
                    [2.9916205406188965, 2.481820583343506, 4.646594524383545],
                    [-2.8381078243255615, -2.9687185287475586, -2.6926779747009277],
                    [-2.8909168243408203, -3.2228589057922363, -1.7056822776794434],
                ]
            ),
            atol=1e-5,
        )

        # fmt: off
        EXPECTED_IDS = [
           [
                0, 64003, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
                55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 64004, 64012, 712, 1648, 9, 64007, 10, 43867, 64008,
                64009, 64057, 64876, 64010, 5950, 597, 32, 64007, 10, 646, 64008, 64009, 64018, 64924, 64010, 4, 2
           ]
        ]
        # fmt: on
        self.assertListEqual(generated_ids.to("cpu").numpy().tolist(), EXPECTED_IDS)

        EXPECTED_PROCESSED_TEXT = (
            "<grounding> An image of<phrase> a snowman</phrase><object><patch_index_0044><patch_index_0863></object> "
            "warming himself by<phrase> a fire</phrase><object><patch_index_0005><patch_index_0911></object>."
        )
        self.assertEqual(processed_text, EXPECTED_PROCESSED_TEXT)

        self.assertEqual(final_text, "An image of a snowman warming himself by a fire.")

        EXPECTED_ENTITIES = [
            ("a snowman", (12, 21), [(0.390625, 0.046875, 0.984375, 0.828125)]),
            ("a fire", (41, 47), [(0.171875, 0.015625, 0.484375, 0.890625)]),
        ]
        self.assertListEqual(entities, EXPECTED_ENTITIES)

        # test with the detail caption generation

        prompt = "<grounding>Describe this image in detail:"
        scores, generated_ids, generated_text, processed_text, final_text_with_entities = self.run_example(
            prompt, image, model, processor
        )
        processed_text = processed_text[0]
        final_text, entities = final_text_with_entities[0]

        np.testing.assert_allclose(
            torch.concat(scores[1:4])[:3, :3].to("cpu").numpy(),
            np.array(
                [
                    [-0.9093570113182068, -4.578373908996582, 5.96360969543457],
                    [2.452126979827881, -4.090598106384277, 8.738677024841309],
                    [-0.7624598741531372, -4.771658897399902, 6.576295852661133],
                ]
            ),
            atol=atol,
        )
        np.testing.assert_allclose(
            torch.concat(scores[-3:])[-3:, -3:].to("cpu").numpy(),
            np.array(
                [
                    [-1.673659086227417, -2.162452220916748, -1.95430588722229],
                    [-2.006824493408203, -2.2038745880126953, -1.24686861038208],
                    [-3.2783470153808594, -2.814181089401245, -1.390632152557373],
                ]
            ),
            atol=1e-5,
        )

        # fmt: off
        EXPECTED_IDS_LONG = [
            [
                0, 64003, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
                55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 64004, 64012, 34645, 247, 38, 1648, 12, 3391, 55,
                24, 1648, 1338, 10, 43867, 1280, 32, 64007, 10, 30879, 64008, 64009, 64018, 65020, 64010, 12, 5, 1842,
                4, 71, 17, 1679, 64007, 10, 3958, 64008, 64009, 64061, 64263, 64010, 6, 64007, 15719, 64008, 64009,
                64253, 64617, 64010, 6, 8, 64007, 9626, 64008, 64009, 64413, 64545, 64010, 6, 23, 64007, 10, 4363,
                64008, 64009, 64623, 64885, 64010, 2255, 8, 64007, 10, 3486, 64008, 64009, 64809, 65036, 64010, 1560,
                2255, 4, 24, 43867, 1684, 7, 27, 3774, 5, 10356, 9, 5, 646, 6, 8, 22, 1684, 7, 30, 10, 2007, 8, 16239,
                4337, 4, 2
            ]
        ]
        # fmt: on
        self.assertListEqual(generated_ids.to("cpu").numpy().tolist(), EXPECTED_IDS_LONG)

        EXPECTED_PROCESSED_TEXT_LONG = (
            "<grounding> Describe this image in detail: The image features a snowman sitting by<phrase> a campfire"
            "</phrase><object><patch_index_0005><patch_index_1007></object> in the snow. He is wearing<phrase> a hat"
            "</phrase><object><patch_index_0048><patch_index_0250></object>,<phrase> scarf</phrase><object>"
            "<patch_index_0240><patch_index_0604></object>, and<phrase> gloves</phrase><object><patch_index_0400>"
            "<patch_index_0532></object>, with<phrase> a pot</phrase><object><patch_index_0610><patch_index_0872>"
            "</object> nearby and<phrase> a cup</phrase><object><patch_index_0796><patch_index_1023></object> placed "
            "nearby. The snowman appears to be enjoying the warmth of the fire, and it appears to have a warm and cozy "
            "atmosphere."
        )
        self.assertEqual(processed_text, EXPECTED_PROCESSED_TEXT_LONG)

        EXPECTED_FINAL_TEXT_LONG = (
            "Describe this image in detail: The image features a snowman sitting by a campfire in the snow. He is "
            "wearing a hat, scarf, and gloves, with a pot nearby and a cup placed nearby. The snowman appears to be "
            "enjoying the warmth of the fire, and it appears to have a warm and cozy atmosphere."
        )
        self.assertEqual(final_text, EXPECTED_FINAL_TEXT_LONG)

        EXPECTED_ENTITIES_LONG = [
            ("a campfire", (71, 81), [(0.171875, 0.015625, 0.484375, 0.984375)]),
            ("a hat", (109, 114), [(0.515625, 0.046875, 0.828125, 0.234375)]),
            ("scarf", (116, 121), [(0.515625, 0.234375, 0.890625, 0.578125)]),
            ("gloves", (127, 133), [(0.515625, 0.390625, 0.640625, 0.515625)]),
            ("a pot", (140, 145), [(0.078125, 0.609375, 0.265625, 0.859375)]),
            ("a cup", (157, 162), [(0.890625, 0.765625, 0.984375, 0.984375)]),
        ]
        self.assertListEqual(entities, EXPECTED_ENTITIES_LONG)

    def test_snowman_image_captioning_batch(self):
        url = "https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/snowman.png"

        image = Image.open(requests.get(url, stream=True).raw)
        image.save("new_image.jpg")
        image = Image.open("new_image.jpg")

        model = AutoModelForImageTextToText.from_pretrained("microsoft/kosmos-2-patch14-224").to(torch_device)

        prompt = ["<grounding>Describe this image in detail:", "<grounding>An image of"]

        # left padding
        processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224", padding_side="left")

        scores, generated_ids, generated_text, processed_text, final_text_with_entities = self.run_example(
            prompt, [image] * len(prompt), model, processor
        )
        all_final_text = [x[0] for x in final_text_with_entities]
        all_entities = [x[1] for x in final_text_with_entities]

        # left padding gives identical results as non-padding
        EXPECTED_PROCESSED_TEXT_0 = (
            "<grounding> Describe this image in detail: The image features a snowman sitting by<phrase> a campfire"
            "</phrase><object><patch_index_0005><patch_index_1007></object> in the snow. He is wearing<phrase> a hat"
            "</phrase><object><patch_index_0048><patch_index_0250></object>,<phrase> scarf</phrase><object>"
            "<patch_index_0240><patch_index_0604></object>, and<phrase> gloves</phrase><object><patch_index_0400>"
            "<patch_index_0532></object>, with<phrase> a pot</phrase><object><patch_index_0610><patch_index_0872>"
            "</object> nearby and<phrase> a cup</phrase><object><patch_index_0796><patch_index_1023></object> placed "
            "nearby. The snowman appears to be enjoying the warmth of the fire, and it appears to have a warm and cozy "
            "atmosphere."
        )
        EXPECTED_PROCESSED_TEXT_1 = (
            "<grounding> An image of<phrase> a snowman</phrase><object><patch_index_0044><patch_index_0863></object> "
            "warming himself by<phrase> a fire</phrase><object><patch_index_0005><patch_index_0911></object>."
        )
        self.assertListEqual(processed_text, [EXPECTED_PROCESSED_TEXT_0, EXPECTED_PROCESSED_TEXT_1])

        EXPECTED_FINAL_TEXT_0 = (
            "Describe this image in detail: The image features a snowman sitting by a campfire in the snow. He is "
            "wearing a hat, scarf, and gloves, with a pot nearby and a cup placed nearby. The snowman appears to be "
            "enjoying the warmth of the fire, and it appears to have a warm and cozy atmosphere."
        )
        EXPECTED_FINAL_TEXT_1 = "An image of a snowman warming himself by a fire."
        self.assertListEqual(all_final_text, [EXPECTED_FINAL_TEXT_0, EXPECTED_FINAL_TEXT_1])

        EXPECTED_ENTITIES_0 = [
            ("a campfire", (71, 81), [(0.171875, 0.015625, 0.484375, 0.984375)]),
            ("a hat", (109, 114), [(0.515625, 0.046875, 0.828125, 0.234375)]),
            ("scarf", (116, 121), [(0.515625, 0.234375, 0.890625, 0.578125)]),
            ("gloves", (127, 133), [(0.515625, 0.390625, 0.640625, 0.515625)]),
            ("a pot", (140, 145), [(0.078125, 0.609375, 0.265625, 0.859375)]),
            ("a cup", (157, 162), [(0.890625, 0.765625, 0.984375, 0.984375)]),
        ]
        EXPECTED_ENTITIES_1 = [
            ("a snowman", (12, 21), [(0.390625, 0.046875, 0.984375, 0.828125)]),
            ("a fire", (41, 47), [(0.171875, 0.015625, 0.484375, 0.890625)]),
        ]
        self.assertListEqual(all_entities, [EXPECTED_ENTITIES_0, EXPECTED_ENTITIES_1])

        # right padding
        processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")

        scores, generated_ids, generated_text, processed_text, final_text_with_entities = self.run_example(
            prompt, [image] * len(prompt), model, processor
        )
        all_final_text = [x[0] for x in final_text_with_entities]
        all_entities = [x[1] for x in final_text_with_entities]

        # For right padding, only the non-padded sequences will give the same results as non-padding
        self.assertEqual(processed_text[0], EXPECTED_PROCESSED_TEXT_0)
        self.assertEqual(all_final_text[0], EXPECTED_FINAL_TEXT_0)
        self.assertListEqual(all_entities[0], EXPECTED_ENTITIES_0)

    @slow
    def test_inference_interpolate_pos_encoding(self):
        # ViT models have an `interpolate_pos_encoding` argument in their forward method,
        # allowing to interpolate the pre-trained position embeddings in order to use
        # the model on higher resolutions. The DINO model by Facebook AI leverages this
        # to visualize self-attention on higher resolution images.
        model = Kosmos2Model.from_pretrained("microsoft/kosmos-2-patch14-224").to(torch_device)

        processor = AutoProcessor.from_pretrained(
            "microsoft/kosmos-2-patch14-224", size={"shortest_edge": 180}, crop_size={"height": 180, "width": 180}
        )

        image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
        inputs = processor(text="what's in the image", images=image, return_tensors="pt").to(torch_device)

        # interpolate_pos_encodiung false should return value error
        with self.assertRaises(ValueError, msg="doesn't match model"):
            with torch.no_grad():
                model(**inputs, interpolate_pos_encoding=False)

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs, interpolate_pos_encoding=True)

        # verify the logits
        expected_shape = torch.Size((1, 145, 1024))

        self.assertEqual(outputs.vision_model_output.last_hidden_state.shape, expected_shape)

        expected_slice = torch.tensor(
            [[0.9148, -1.4148, 3.8040], [3.3443, 1.9478, 0.2080], [1.6604, 2.8184, -0.3618]]
        ).to(torch_device)

        self.assertTrue(
            torch.allclose(outputs.vision_model_output.last_hidden_state[0, :3, :3], expected_slice, atol=1e-1)
        )
