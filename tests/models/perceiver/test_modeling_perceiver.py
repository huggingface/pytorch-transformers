# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Perceiver model."""

import copy
import inspect
import math
import tempfile
import unittest
import warnings
from typing import Dict, List, Tuple

import numpy as np
from datasets import load_dataset

from transformers import PerceiverConfig
from transformers.testing_utils import (
    IS_ROCM_SYSTEM,
    require_torch,
    require_torch_multi_gpu,
    require_vision,
    slow,
    torch_device,
)
from transformers.utils import is_torch_available, is_vision_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch
    from torch import nn

    from transformers import (
        PerceiverForImageClassificationConvProcessing,
        PerceiverForImageClassificationFourier,
        PerceiverForImageClassificationLearned,
        PerceiverForMaskedLM,
        PerceiverForMultimodalAutoencoding,
        PerceiverForOpticalFlow,
        PerceiverForSequenceClassification,
        PerceiverModel,
        PerceiverTokenizer,
    )
    from transformers.models.auto.modeling_auto import (
        MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES,
        MODEL_FOR_MASKED_LM_MAPPING_NAMES,
        MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES,
        MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES,
        MODEL_MAPPING_NAMES,
    )


if is_vision_available():
    from PIL import Image

    from transformers import PerceiverImageProcessor


class PerceiverModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        num_channels=3,
        image_size=32,
        train_size=[20, 20],
        num_frames=5,
        audio_samples_per_frame=200,
        samples_per_patch=20,
        nchunks=20,
        num_latents=10,
        d_latents=20,
        d_model=64,
        num_blocks=1,
        num_self_attends_per_block=2,
        num_self_attention_heads=1,
        num_cross_attention_heads=1,
        self_attention_widening_factor=4,
        cross_attention_widening_factor=4,
        is_training=True,
        use_input_mask=True,
        use_labels=True,
        vocab_size=99,
        hidden_act="gelu",
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        max_position_embeddings=7,
        num_labels=3,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.num_channels = num_channels
        self.image_size = image_size
        self.train_size = train_size
        self.num_frames = num_frames
        self.audio_samples_per_frame = audio_samples_per_frame
        self.samples_per_patch = samples_per_patch
        self.nchunks = nchunks
        self.num_latents = num_latents
        self.d_latents = d_latents
        self.d_model = d_model
        self.num_blocks = num_blocks
        self.num_self_attends_per_block = num_self_attends_per_block
        self.num_self_attention_heads = num_self_attention_heads
        self.num_cross_attention_heads = num_cross_attention_heads
        self.self_attention_widening_factor = self_attention_widening_factor
        self.cross_attention_widening_factor = cross_attention_widening_factor
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_act = hidden_act
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.scope = scope
        # set subsampling for multimodal model (take first chunk)
        image_chunk_size = np.prod((self.num_frames, self.image_size, self.image_size)) // self.nchunks
        audio_chunk_size = self.num_frames * self.audio_samples_per_frame // self.samples_per_patch // self.nchunks
        self.subsampling = {
            "image": torch.arange(0, image_chunk_size),
            "audio": torch.arange(0, audio_chunk_size),
            "label": None,
        }

    def prepare_config_and_inputs(self, model_class=None):
        config = self.get_config()

        input_mask = None
        sequence_labels = None
        token_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.num_labels)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)

        if model_class is None or model_class.__name__ == "PerceiverModel":
            inputs = floats_tensor([self.batch_size, self.seq_length, config.d_model], scale=1.0)
            return config, inputs, input_mask, sequence_labels, token_labels
        elif model_class.__name__ in ["PerceiverForMaskedLM", "PerceiverForSequenceClassification"]:
            inputs = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
            # input mask is only relevant for text inputs
            if self.use_input_mask:
                input_mask = random_attention_mask([self.batch_size, self.seq_length])
        elif model_class.__name__ == "PerceiverForImageClassificationLearned":
            inputs = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        elif model_class.__name__ == "PerceiverForImageClassificationFourier":
            inputs = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        elif model_class.__name__ == "PerceiverForImageClassificationConvProcessing":
            inputs = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        elif model_class.__name__ == "PerceiverForOpticalFlow":
            inputs = floats_tensor([self.batch_size, 2, 27, self.train_size[0], self.train_size[1]])
        elif model_class.__name__ == "PerceiverForMultimodalAutoencoding":
            images = torch.randn(
                (self.batch_size, self.num_frames, self.num_channels, self.image_size, self.image_size),
                device=torch_device,
            )
            audio = torch.randn(
                (self.batch_size, self.num_frames * self.audio_samples_per_frame, 1), device=torch_device
            )
            inputs = {
                "image": images,
                "audio": audio,
                "label": torch.zeros((self.batch_size, self.num_labels), device=torch_device),
            }
        else:
            raise ValueError(f"Model class {model_class} not supported")

        return config, inputs, input_mask, sequence_labels, token_labels

    def get_config(self):
        return PerceiverConfig(
            num_latents=self.num_latents,
            d_latents=self.d_latents,
            d_model=self.d_model,
            qk_channels=self.d_latents,
            v_channels=self.d_latents,
            num_blocks=self.num_blocks,
            num_self_attends_per_block=self.num_self_attends_per_block,
            num_self_attention_heads=self.num_self_attention_heads,
            num_cross_attention_heads=self.num_cross_attention_heads,
            self_attention_widening_factor=self.self_attention_widening_factor,
            cross_attention_widening_factor=self.cross_attention_widening_factor,
            vocab_size=self.vocab_size,
            hidden_act=self.hidden_act,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            initializer_range=self.initializer_range,
            max_position_embeddings=self.max_position_embeddings,
            image_size=self.image_size,
            train_size=self.train_size,
            num_frames=self.num_frames,
            audio_samples_per_frame=self.audio_samples_per_frame,
            samples_per_patch=self.samples_per_patch,
            num_labels=self.num_labels,
            output_num_channels=32,
            _label_trainable_num_channels=16,
        )

    def get_pipeline_config(self):
        config = self.get_config()
        # Byte level vocab
        config.vocab_size = 261
        config.max_position_embeddings = 40
        return config

    def create_and_check_for_masked_lm(self, config, inputs, input_mask, sequence_labels, token_labels):
        model = PerceiverForMaskedLM(config=config)
        model.to(torch_device)
        model.eval()
        result = model(inputs, attention_mask=input_mask, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_for_sequence_classification(self, config, inputs, input_mask, sequence_labels, token_labels):
        model = PerceiverForSequenceClassification(config=config)
        model.to(torch_device)
        model.eval()
        result = model(inputs, attention_mask=input_mask, labels=sequence_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def create_and_check_for_image_classification_learned(
        self, config, inputs, input_mask, sequence_labels, token_labels
    ):
        model = PerceiverForImageClassificationLearned(config=config)
        model.to(torch_device)
        model.eval()
        result = model(inputs, attention_mask=input_mask, labels=sequence_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def create_and_check_for_image_classification_fourier(
        self, config, inputs, input_mask, sequence_labels, token_labels
    ):
        model = PerceiverForImageClassificationFourier(config=config)
        model.to(torch_device)
        model.eval()
        result = model(inputs, attention_mask=input_mask, labels=sequence_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def create_and_check_for_image_classification_conv(
        self, config, inputs, input_mask, sequence_labels, token_labels
    ):
        model = PerceiverForImageClassificationConvProcessing(config=config)
        model.to(torch_device)
        model.eval()
        result = model(inputs, attention_mask=input_mask, labels=sequence_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, inputs, input_mask, sequence_labels, token_labels = config_and_inputs
        inputs_dict = {"inputs": inputs, "attention_mask": input_mask}
        return config, inputs_dict

    def prepare_config_and_inputs_for_model_class(self, model_class):
        config_and_inputs = self.prepare_config_and_inputs(model_class)
        config, inputs, input_mask, sequence_labels, token_labels = config_and_inputs
        inputs_dict = {"inputs": inputs, "attention_mask": input_mask}

        return config, inputs_dict


@require_torch
class PerceiverModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            PerceiverModel,
            PerceiverForMaskedLM,
            PerceiverForImageClassificationLearned,
            PerceiverForImageClassificationConvProcessing,
            PerceiverForImageClassificationFourier,
            PerceiverForOpticalFlow,
            PerceiverForMultimodalAutoencoding,
            PerceiverForSequenceClassification,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": PerceiverModel,
            "fill-mask": PerceiverForMaskedLM,
            "image-classification": (
                PerceiverForImageClassificationConvProcessing,
                PerceiverForImageClassificationFourier,
                PerceiverForImageClassificationLearned,
            ),
            "text-classification": PerceiverForSequenceClassification,
            "zero-shot": PerceiverForSequenceClassification,
        }
        if is_torch_available()
        else {}
    )
    test_pruning = False
    test_head_masking = False
    test_torchscript = False

    maxDiff = None

    def setUp(self):
        self.model_tester = PerceiverModelTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=PerceiverConfig,
            hidden_size=37,
            common_properties=["d_model", "num_self_attention_heads", "num_cross_attention_heads"],
        )

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = copy.deepcopy(inputs_dict)

        if model_class.__name__ == "PerceiverForMultimodalAutoencoding":
            inputs_dict["subsampled_output_points"] = self.model_tester.subsampling

        if return_labels:
            if model_class.__name__ in [
                *MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES.values(),
                "PerceiverForImageClassificationLearned",
                "PerceiverForImageClassificationFourier",
                "PerceiverForImageClassificationConvProcessing",
                *MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES.values(),
            ]:
                inputs_dict["labels"] = torch.zeros(
                    self.model_tester.batch_size, dtype=torch.long, device=torch_device
                )
            elif model_class.__name__ in [
                *MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES.values(),
                *MODEL_FOR_MASKED_LM_MAPPING_NAMES.values(),
            ]:
                inputs_dict["labels"] = torch.zeros(
                    (self.model_tester.batch_size, self.model_tester.seq_length), dtype=torch.long, device=torch_device
                )
        return inputs_dict

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_for_masked_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs(model_class=PerceiverForMaskedLM)
        self.model_tester.create_and_check_for_masked_lm(*config_and_inputs)

    def test_for_sequence_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs(model_class=PerceiverForSequenceClassification)
        self.model_tester.create_and_check_for_sequence_classification(*config_and_inputs)

    def test_for_image_classification_learned(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs(
            model_class=PerceiverForImageClassificationLearned
        )
        self.model_tester.create_and_check_for_image_classification_learned(*config_and_inputs)

    def test_for_image_classification_fourier(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs(
            model_class=PerceiverForImageClassificationFourier
        )
        self.model_tester.create_and_check_for_image_classification_fourier(*config_and_inputs)

    def test_for_image_classification_conv(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs(
            model_class=PerceiverForImageClassificationConvProcessing
        )
        self.model_tester.create_and_check_for_image_classification_conv(*config_and_inputs)

    def test_model_get_set_embeddings(self):
        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_model_class(model_class)
            model = model_class(config)
            # we overwrite this, as the embeddings of Perceiver are an instance of nn.Parameter
            # and Perceiver doesn't support get_output_embeddings
            self.assertIsInstance(model.get_input_embeddings(), (nn.Parameter))

    def test_training(self):
        if not self.model_tester.is_training:
            self.skipTest(reason="model_tester.is_training is set to False")

        for model_class in self.all_model_classes:
            if model_class.__name__ in [
                *MODEL_MAPPING_NAMES.values(),
                "PerceiverForOpticalFlow",
                "PerceiverForMultimodalAutoencoding",
            ]:
                continue

            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_model_class(model_class)
            config.return_dict = True

            model = model_class(config)
            model.to(torch_device)
            model.train()
            inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            loss = model(**inputs).loss
            loss.backward()

    def test_forward_signature(self):
        for model_class in self.all_model_classes:
            config, _ = self.model_tester.prepare_config_and_inputs_for_model_class(model_class)

            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["inputs"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_determinism(self):
        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_model_class(model_class)

            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                inputs_dict = self._prepare_for_class(inputs_dict, model_class)
                first = model(**inputs_dict)[0]
                second = model(**inputs_dict)[0]

            if model_class.__name__ == "PerceiverForMultimodalAutoencoding":
                # model outputs a dictionary with logits per modality, let's verify each modality
                for modality in first.keys():
                    out_1 = first[modality].cpu().numpy()
                    out_2 = second[modality].cpu().numpy()
                    out_1 = out_1[~np.isnan(out_1)]
                    out_2 = out_2[~np.isnan(out_2)]
                    max_diff = np.amax(np.abs(out_1 - out_2))
                    self.assertLessEqual(max_diff, 1e-5)
            else:
                out_1 = first.cpu().numpy()
                out_2 = second.cpu().numpy()
                out_1 = out_1[~np.isnan(out_1)]
                out_2 = out_2[~np.isnan(out_2)]
                max_diff = np.amax(np.abs(out_1 - out_2))
                self.assertLessEqual(max_diff, 1e-5)

    def test_attention_outputs(self):
        seq_len = getattr(self.model_tester, "num_latents", None)

        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_model_class(model_class)
            config.return_dict = True

            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            config.return_dict = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            self_attentions = outputs.attentions
            cross_attentions = outputs.cross_attentions

            # check expected number of attentions depending on model class
            expected_num_self_attentions = self.model_tester.num_blocks * self.model_tester.num_self_attends_per_block
            if model.__class__.__name__ == "PerceiverModel":
                # we expect to have 2 cross-attentions, namely one in the PerceiverEncoder, and one in PerceiverBasicDecoder
                expected_num_cross_attentions = 1
            else:
                # we expect to have 2 cross-attentions, namely one in the PerceiverEncoder, and one in PerceiverBasicDecoder
                expected_num_cross_attentions = 2
            self.assertEqual(len(self_attentions), expected_num_self_attentions)
            self.assertEqual(len(cross_attentions), expected_num_cross_attentions)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            self_attentions = outputs.attentions
            cross_attentions = outputs.cross_attentions
            self.assertEqual(len(self_attentions), expected_num_self_attentions)
            self.assertEqual(len(cross_attentions), expected_num_cross_attentions)

            self.assertListEqual(
                list(self_attentions[0].shape[-3:]),
                [self.model_tester.num_self_attention_heads, seq_len, seq_len],
            )
            out_len = len(outputs)

            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            self.assertEqual(out_len + 1, len(outputs))

            self_attentions = outputs.attentions

            self.assertEqual(len(self_attentions), expected_num_self_attentions)
            self.assertListEqual(
                list(self_attentions[0].shape[-3:]),
                [self.model_tester.num_self_attention_heads, seq_len, seq_len],
            )

    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.hidden_states

            expected_num_layers = self.model_tester.num_blocks * self.model_tester.num_self_attends_per_block + 1
            self.assertEqual(len(hidden_states), expected_num_layers)

            seq_length = self.model_tester.num_latents

            self.assertListEqual(
                list(hidden_states[0].shape[-2:]),
                [seq_length, self.model_tester.d_latents],
            )

        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_model_class(model_class)

            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True

            check_hidden_states_output(inputs_dict, config, model_class)

    def test_model_outputs_equivalence(self):
        def set_nan_tensor_to_zero(t):
            t[t != t] = 0
            return t

        def check_equivalence(model, tuple_inputs, dict_inputs, additional_kwargs={}):
            with torch.no_grad():
                tuple_output = model(**tuple_inputs, return_dict=False, **additional_kwargs)
                dict_output = model(**dict_inputs, return_dict=True, **additional_kwargs).to_tuple()

                def recursive_check(tuple_object, dict_object):
                    if isinstance(tuple_object, (List, Tuple)):
                        for tuple_iterable_value, dict_iterable_value in zip(tuple_object, dict_object):
                            recursive_check(tuple_iterable_value, dict_iterable_value)
                    elif isinstance(tuple_object, Dict):
                        for tuple_iterable_value, dict_iterable_value in zip(
                            tuple_object.values(), dict_object.values()
                        ):
                            recursive_check(tuple_iterable_value, dict_iterable_value)
                    elif tuple_object is None:
                        return
                    else:
                        self.assertTrue(
                            torch.allclose(
                                set_nan_tensor_to_zero(tuple_object), set_nan_tensor_to_zero(dict_object), atol=1e-5
                            ),
                            msg=(
                                "Tuple and dict output are not equal. Difference:"
                                f" {torch.max(torch.abs(tuple_object - dict_object))}. Tuple has `nan`:"
                                f" {torch.isnan(tuple_object).any()} and `inf`: {torch.isinf(tuple_object)}. Dict has"
                                f" `nan`: {torch.isnan(dict_object).any()} and `inf`: {torch.isinf(dict_object)}."
                            ),
                        )

                recursive_check(tuple_output, dict_output)

        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_model_class(model_class)

            model = model_class(config)
            model.to(torch_device)
            model.eval()

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class)
            check_equivalence(model, tuple_inputs, dict_inputs)

            if model_class.__name__ not in ["PerceiverForOpticalFlow", "PerceiverForMultimodalAutoencoding"]:
                # optical flow + multimodal models don't support training for now
                tuple_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
                dict_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
                check_equivalence(model, tuple_inputs, dict_inputs)

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class)

            check_equivalence(model, tuple_inputs, dict_inputs, {"output_hidden_states": True})

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class)
            check_equivalence(model, tuple_inputs, dict_inputs, {"output_attentions": True})

            if model_class.__name__ not in ["PerceiverForOpticalFlow", "PerceiverForMultimodalAutoencoding"]:
                # optical flow + multimodal models don't support training for now
                tuple_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
                dict_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
                check_equivalence(model, tuple_inputs, dict_inputs, {"output_hidden_states": True})

            if model_class.__name__ not in ["PerceiverForOpticalFlow", "PerceiverForMultimodalAutoencoding"]:
                # optical flow + multimodal models don't support training for now
                tuple_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
                dict_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
                check_equivalence(model, tuple_inputs, dict_inputs, {"output_attentions": True})

            if model_class.__name__ not in ["PerceiverForOpticalFlow", "PerceiverForMultimodalAutoencoding"]:
                # optical flow + multimodal models don't support training for now
                tuple_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
                dict_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
                check_equivalence(
                    model, tuple_inputs, dict_inputs, {"output_hidden_states": True, "output_attentions": True}
                )

    def test_retain_grad_hidden_states_attentions(self):
        # no need to test all models as different heads yield the same functionality
        model_class = PerceiverForMaskedLM
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_model_class(model_class)
        config.output_hidden_states = True
        config.output_attentions = True

        model = model_class(config)
        model.to(torch_device)

        inputs = self._prepare_for_class(inputs_dict, model_class)

        outputs = model(**inputs)

        output = outputs[0]

        # Encoder-only model
        hidden_states = outputs.hidden_states[0]
        attentions = outputs.attentions[0]

        hidden_states.retain_grad()
        attentions.retain_grad()

        output.flatten()[0].backward(retain_graph=True)

        self.assertIsNotNone(hidden_states.grad)
        self.assertIsNotNone(attentions.grad)

    def test_feed_forward_chunking(self):
        for model_class in self.all_model_classes:
            original_config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_model_class(model_class)
            torch.manual_seed(0)
            config = copy.deepcopy(original_config)
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            hidden_states_no_chunk = model(**self._prepare_for_class(inputs_dict, model_class))[0]

            torch.manual_seed(0)
            config.chunk_size_feed_forward = 1
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            hidden_states_with_chunk = model(**self._prepare_for_class(inputs_dict, model_class))[0]
            if model_class.__name__ == "PerceiverForMultimodalAutoencoding":
                # model outputs a dictionary with logits for each modality
                for modality in hidden_states_no_chunk.keys():
                    self.assertTrue(
                        torch.allclose(hidden_states_no_chunk[modality], hidden_states_with_chunk[modality], atol=1e-3)
                    )
            else:
                torch.testing.assert_close(hidden_states_no_chunk, hidden_states_with_chunk, rtol=1e-3, atol=1e-3)

    def test_save_load(self):
        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_model_class(model_class)

            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            if model_class.__name__ == "PerceiverForMultimodalAutoencoding":
                for modality in outputs[0].keys():
                    out_2 = outputs[0][modality].cpu().numpy()
                    out_2[np.isnan(out_2)] = 0

                    with tempfile.TemporaryDirectory() as tmpdirname:
                        model.save_pretrained(tmpdirname)
                        model = model_class.from_pretrained(tmpdirname)
                        model.to(torch_device)
                        with torch.no_grad():
                            after_outputs = model(**self._prepare_for_class(inputs_dict, model_class))

                        # Make sure we don't have nans
                        out_1 = after_outputs[0][modality].cpu().numpy()
                        out_1[np.isnan(out_1)] = 0
                        max_diff = np.amax(np.abs(out_1 - out_2))
                        self.assertLessEqual(max_diff, 1e-5)

            else:
                out_2 = outputs[0].cpu().numpy()
                out_2[np.isnan(out_2)] = 0

                with tempfile.TemporaryDirectory() as tmpdirname:
                    model.save_pretrained(tmpdirname)
                    model = model_class.from_pretrained(tmpdirname)
                    model.to(torch_device)
                    with torch.no_grad():
                        after_outputs = model(**self._prepare_for_class(inputs_dict, model_class))

                    # Make sure we don't have nans
                    out_1 = after_outputs[0].cpu().numpy()
                    out_1[np.isnan(out_1)] = 0
                    max_diff = np.amax(np.abs(out_1 - out_2))
                    self.assertLessEqual(max_diff, 1e-5)

    def test_correct_missing_keys(self):
        if not self.test_missing_keys:
            self.skipTest(reason="test_missing_keys is set to False")
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            # most Perceiver models don't have a typical head like is the case with BERT
            if model_class.__name__ in [
                "PerceiverForOpticalFlow",
                "PerceiverForMultimodalAutoencoding",
                *MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES.values(),
                "PerceiverForImageClassificationLearned",
                "PerceiverForImageClassificationFourier",
                "PerceiverForImageClassificationConvProcessing",
                *MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES.values(),
            ]:
                continue

            model = model_class(config)
            base_model_prefix = model.base_model_prefix

            if hasattr(model, base_model_prefix):
                with tempfile.TemporaryDirectory() as temp_dir_name:
                    model.base_model.save_pretrained(temp_dir_name)
                    model, loading_info = model_class.from_pretrained(temp_dir_name, output_loading_info=True)
                    with self.subTest(msg=f"Missing keys for {model.__class__.__name__}"):
                        self.assertGreater(len(loading_info["missing_keys"]), 0)

    def test_problem_types(self):
        problem_types = [
            {"title": "multi_label_classification", "num_labels": 2, "dtype": torch.float},
            {"title": "single_label_classification", "num_labels": 1, "dtype": torch.long},
            {"title": "regression", "num_labels": 1, "dtype": torch.float},
        ]

        for model_class in self.all_model_classes:
            if model_class.__name__ not in MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES.values():
                continue

            config, inputs, input_mask, _, _ = self.model_tester.prepare_config_and_inputs(model_class=model_class)
            inputs_dict = {"inputs": inputs, "attention_mask": input_mask}

            for problem_type in problem_types:
                with self.subTest(msg=f"Testing {model_class} with {problem_type['title']}"):
                    config.problem_type = problem_type["title"]
                    config.num_labels = problem_type["num_labels"]

                    model = model_class(config)
                    model.to(torch_device)
                    model.train()

                    inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)

                    if problem_type["num_labels"] > 1:
                        inputs["labels"] = inputs["labels"].unsqueeze(1).repeat(1, problem_type["num_labels"])

                    inputs["labels"] = inputs["labels"].to(problem_type["dtype"])

                    # This tests that we do not trigger the warning form PyTorch "Using a target size that is different
                    # to the input size. This will likely lead to incorrect results due to broadcasting. Please ensure
                    # they have the same size." which is a symptom something in wrong for the regression problem.
                    # See https://github.com/huggingface/transformers/issues/11780
                    with warnings.catch_warnings(record=True) as warning_list:
                        loss = model(**inputs).loss
                    for w in warning_list:
                        if "Using a target size that is different to the input size" in str(w.message):
                            raise ValueError(
                                f"Something is going wrong in the regression problem: intercepted {w.message}"
                            )

                    loss.backward()

    @require_torch_multi_gpu
    @unittest.skip(
        reason=(
            "Perceiver does not work with data parallel (DP) because of a bug in PyTorch:"
            " https://github.com/pytorch/pytorch/issues/36035"
        )
    )
    def test_multi_gpu_data_parallel_forward(self):
        pass

    @unittest.skip(reason="Perceiver models don't have a typical head like is the case with BERT")
    def test_save_load_fast_init_from_base(self):
        pass

    @unittest.skip(reason="Perceiver models don't have a typical head like is the case with BERT")
    def test_save_load_fast_init_to_base(self):
        pass

    @unittest.skip(reason="Perceiver doesn't support resize_token_embeddings")
    def test_resize_tokens_embeddings(self):
        pass

    @unittest.skip(reason="Perceiver doesn't support resize_token_embeddings")
    def test_resize_embeddings_untied(self):
        pass

    @unittest.skip(reason="Perceiver doesn't support inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="Perceiver doesn't support the AutoModel API")
    def test_load_with_mismatched_shapes(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        model_name = "deepmind/language-perceiver"
        model = PerceiverModel.from_pretrained(model_name)
        self.assertIsNotNone(model)


# We will verify our results on an image of cute cats
def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image


# Helper functions for optical flow integration test
def prepare_optical_flow_images():
    dataset = load_dataset("hf-internal-testing/fixtures_sintel", split="test", trust_remote_code=True)
    image1 = Image.open(dataset[0]["file"]).convert("RGB")
    image2 = Image.open(dataset[0]["file"]).convert("RGB")

    return image1, image2


def normalize(img):
    return img / 255.0 * 2 - 1


def extract_image_patches(x, kernel, stride=1, dilation=1):
    # Do TF 'SAME' Padding
    b, c, h, w = x.shape
    h2 = math.ceil(h / stride)
    w2 = math.ceil(w / stride)
    pad_row = (h2 - 1) * stride + (kernel - 1) * dilation + 1 - h
    pad_col = (w2 - 1) * stride + (kernel - 1) * dilation + 1 - w
    x = torch.nn.functional.pad(x, (pad_row // 2, pad_row - pad_row // 2, pad_col // 2, pad_col - pad_col // 2))

    # Extract patches
    patches = x.unfold(2, kernel, stride).unfold(3, kernel, stride)
    patches = patches.permute(0, 4, 5, 1, 2, 3).contiguous()

    return patches.view(b, -1, patches.shape[-2], patches.shape[-1])


@require_torch
@require_vision
class PerceiverModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference_masked_lm(self):
        tokenizer = PerceiverTokenizer.from_pretrained("deepmind/language-perceiver")
        model = PerceiverForMaskedLM.from_pretrained("deepmind/language-perceiver")
        model.to(torch_device)

        # prepare inputs
        text = "This is an incomplete sentence where some words are missing."
        encoding = tokenizer(text, padding="max_length", return_tensors="pt")

        # mask " missing.".
        encoding.input_ids[0, 52:61] = tokenizer.mask_token_id
        inputs, input_mask = encoding.input_ids.to(torch_device), encoding.attention_mask.to(torch_device)

        # forward pass
        with torch.no_grad():
            outputs = model(inputs=inputs, attention_mask=input_mask)
        logits = outputs.logits

        # verify logits
        expected_shape = torch.Size((1, tokenizer.model_max_length, len(tokenizer)))
        self.assertEqual(logits.shape, expected_shape)

        expected_slice = torch.tensor(
            [[-10.8609, -10.7651, -10.9187], [-12.1689, -11.9389, -12.1479], [-12.1518, -11.9707, -12.2073]],
            device=torch_device,
        )

        torch.testing.assert_close(logits[0, :3, :3], expected_slice, rtol=1e-4, atol=1e-4)

        expected_greedy_predictions = [38, 115, 111, 121, 121, 111, 116, 109, 52]
        masked_tokens_predictions = logits[0, 52:61].argmax(dim=-1).tolist()
        self.assertListEqual(expected_greedy_predictions, masked_tokens_predictions)

    @slow
    def test_inference_image_classification(self):
        image_processor = PerceiverImageProcessor()
        model = PerceiverForImageClassificationLearned.from_pretrained("deepmind/vision-perceiver-learned")
        model.to(torch_device)

        # prepare inputs
        image = prepare_img()
        inputs = image_processor(image, return_tensors="pt").pixel_values.to(torch_device)
        input_mask = None

        # forward pass
        with torch.no_grad():
            outputs = model(inputs=inputs, attention_mask=input_mask)
        logits = outputs.logits

        # verify logits
        expected_shape = torch.Size((1, model.config.num_labels))
        self.assertEqual(logits.shape, expected_shape)

        expected_slice = torch.tensor([-1.1652, -0.1992, -0.7520], device=torch_device)

        atol = 1e-3 if IS_ROCM_SYSTEM else 1e-4
        torch.testing.assert_close(logits[0, :3], expected_slice, rtol=atol, atol=atol)

    @slow
    def test_inference_image_classification_fourier(self):
        image_processor = PerceiverImageProcessor()
        model = PerceiverForImageClassificationFourier.from_pretrained("deepmind/vision-perceiver-fourier")
        model.to(torch_device)

        # prepare inputs
        image = prepare_img()
        inputs = image_processor(image, return_tensors="pt").pixel_values.to(torch_device)
        input_mask = None

        # forward pass
        with torch.no_grad():
            outputs = model(inputs=inputs, attention_mask=input_mask)
        logits = outputs.logits

        # verify logits
        expected_shape = torch.Size((1, model.config.num_labels))
        self.assertEqual(logits.shape, expected_shape)

        expected_slice = torch.tensor([-1.1295, -0.2832, 0.3226], device=torch_device)

        torch.testing.assert_close(logits[0, :3], expected_slice, rtol=1e-4, atol=1e-4)

    @slow
    def test_inference_image_classification_conv(self):
        image_processor = PerceiverImageProcessor()
        model = PerceiverForImageClassificationConvProcessing.from_pretrained("deepmind/vision-perceiver-conv")
        model.to(torch_device)

        # prepare inputs
        image = prepare_img()
        inputs = image_processor(image, return_tensors="pt").pixel_values.to(torch_device)
        input_mask = None

        # forward pass
        with torch.no_grad():
            outputs = model(inputs=inputs, attention_mask=input_mask)
        logits = outputs.logits

        # verify logits
        expected_shape = torch.Size((1, model.config.num_labels))
        self.assertEqual(logits.shape, expected_shape)

        expected_slice = torch.tensor([-1.1186, 0.0554, 0.0897], device=torch_device)

        torch.testing.assert_close(logits[0, :3], expected_slice, rtol=1e-4, atol=1e-4)

    @slow
    def test_inference_optical_flow(self):
        model = PerceiverForOpticalFlow.from_pretrained("deepmind/optical-flow-perceiver")
        model.to(torch_device)

        # prepare inputs
        image1, image2 = prepare_optical_flow_images()
        img1 = normalize(np.array(image1))
        img2 = normalize(np.array(image1))

        # stack images
        img1 = torch.tensor(np.moveaxis(img1, -1, 0))
        img2 = torch.tensor(np.moveaxis(img2, -1, 0))
        images = torch.stack([img1, img2], dim=0)

        # extract 3x3 patches
        patch_size = model.config.train_size

        inputs = images[..., : patch_size[0], : patch_size[1]].unsqueeze(0)
        batch_size, _, C, H, W = inputs.shape
        patches = extract_image_patches(inputs.view(batch_size * 2, C, H, W), kernel=3)
        _, C, H, W = patches.shape
        patches = patches.view(batch_size, -1, C, H, W).float()

        # forward pass
        with torch.no_grad():
            outputs = model(inputs=patches.to(torch_device))
        logits = outputs.logits

        # verify logits
        expected_shape = torch.Size((1, 368, 496, 2))
        self.assertEqual(logits.shape, expected_shape)

        expected_slice = torch.tensor(
            [
                [[0.0025, -0.0050], [0.0025, -0.0049], [0.0025, -0.0048]],
                [[0.0026, -0.0049], [0.0026, -0.0048], [0.0026, -0.0047]],
                [[0.0026, -0.0049], [0.0026, -0.0048], [0.0026, -0.0046]],
            ],
            device=torch_device,
        )

        torch.testing.assert_close(logits[0, :3, :3, :3], expected_slice, rtol=1e-4, atol=1e-4)

    @slow
    def test_inference_interpolate_pos_encoding(self):
        image_processor = PerceiverImageProcessor(size={"height": 384, "width": 384})
        model = PerceiverForImageClassificationLearned.from_pretrained("deepmind/vision-perceiver-learned")
        model.to(torch_device)

        # prepare inputs
        image = prepare_img()
        inputs = image_processor(image, return_tensors="pt").pixel_values.to(torch_device)
        input_mask = None

        # forward pass
        with torch.no_grad():
            outputs = model(inputs=inputs, attention_mask=input_mask, interpolate_pos_encoding=True)
        logits = outputs.logits

        # verify logits
        expected_shape = torch.Size((1, model.config.num_labels))
        self.assertEqual(logits.shape, expected_shape)
