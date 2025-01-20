# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch MaskFormer Swin model."""

import collections
import unittest
from typing import Dict, List, Tuple

from transformers import MaskFormerSwinConfig
from transformers.testing_utils import require_torch, require_torch_multi_gpu, torch_device
from transformers.utils import is_torch_available

from ...test_backbone_common import BackboneTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch
    from torch import nn

    from transformers import MaskFormerSwinBackbone
    from transformers.models.maskformer import MaskFormerSwinModel


class MaskFormerSwinModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        image_size=32,
        patch_size=2,
        num_channels=3,
        embed_dim=16,
        depths=[1, 2, 1],
        num_heads=[2, 2, 4],
        window_size=2,
        mlp_ratio=2.0,
        qkv_bias=True,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        drop_path_rate=0.1,
        hidden_act="gelu",
        use_absolute_embeddings=False,
        patch_norm=True,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        is_training=True,
        scope=None,
        use_labels=True,
        type_sequence_label_size=10,
        encoder_stride=8,
        out_features=["stage1", "stage2", "stage3"],
        out_indices=[1, 2, 3],
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.drop_path_rate = drop_path_rate
        self.hidden_act = hidden_act
        self.use_absolute_embeddings = use_absolute_embeddings
        self.patch_norm = patch_norm
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.is_training = is_training
        self.scope = scope
        self.use_labels = use_labels
        self.type_sequence_label_size = type_sequence_label_size
        self.encoder_stride = encoder_stride
        self.out_features = out_features
        self.out_indices = out_indices

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])

        labels = None
        if self.use_labels:
            labels = ids_tensor([self.batch_size], self.type_sequence_label_size)

        config = self.get_config()

        return config, pixel_values, labels

    def get_config(self):
        return MaskFormerSwinConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            embed_dim=self.embed_dim,
            depths=self.depths,
            num_heads=self.num_heads,
            window_size=self.window_size,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=self.qkv_bias,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            drop_path_rate=self.drop_path_rate,
            hidden_act=self.hidden_act,
            use_absolute_embeddings=self.use_absolute_embeddings,
            path_norm=self.patch_norm,
            layer_norm_eps=self.layer_norm_eps,
            initializer_range=self.initializer_range,
            encoder_stride=self.encoder_stride,
            out_features=self.out_features,
            out_indices=self.out_indices,
        )

    def create_and_check_model(self, config, pixel_values, labels):
        model = MaskFormerSwinModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)

        expected_seq_len = ((config.image_size // config.patch_size) ** 2) // (4 ** (len(config.depths) - 1))
        expected_dim = int(config.embed_dim * 2 ** (len(config.depths) - 1))

        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, expected_seq_len, expected_dim))

    def create_and_check_backbone(self, config, pixel_values, labels):
        model = MaskFormerSwinBackbone(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)

        # verify feature maps
        self.parent.assertEqual(len(result.feature_maps), len(config.out_features))
        self.parent.assertListEqual(list(result.feature_maps[0].shape), [13, 16, 16, 16])

        # verify channels
        self.parent.assertEqual(len(model.channels), len(config.out_features))
        self.parent.assertListEqual(model.channels, [16, 32, 64])

        # verify ValueError
        with self.parent.assertRaises(ValueError):
            config.out_features = ["stem"]
            model = MaskFormerSwinBackbone(config=config)

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values, labels = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class MaskFormerSwinModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            MaskFormerSwinModel,
            MaskFormerSwinBackbone,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = {"feature-extraction": MaskFormerSwinModel} if is_torch_available() else {}
    fx_compatible = False
    test_torchscript = False
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    test_torch_exportable = True

    def setUp(self):
        self.model_tester = MaskFormerSwinModelTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=MaskFormerSwinConfig,
            has_text_modality=False,
            embed_dim=37,
            common_properties=["image_size", "patch_size", "num_channels"],
        )

    @require_torch_multi_gpu
    @unittest.skip(
        reason=(
            "`MaskFormerSwinModel` outputs `hidden_states_spatial_dimensions` which doesn't work well with"
            " `nn.DataParallel`"
        )
    )
    def test_multi_gpu_data_parallel_forward(self):
        pass

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_backbone(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_backbone(*config_and_inputs)

    @unittest.skip(reason="Swin does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="Swin does not support feedforward chunking")
    def test_feed_forward_chunking(self):
        pass

    def test_model_get_set_embeddings(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), (nn.Module))
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, nn.Linear))

    @unittest.skip(reason="MaskFormerSwin is only used as backbone and doesn't support output_attentions")
    def test_attention_outputs(self):
        pass

    @unittest.skip(reason="MaskFormerSwin is only used as an internal backbone")
    def test_save_load_fast_init_to_base(self):
        pass

    def check_hidden_states_output(self, inputs_dict, config, model_class, image_size):
        model = model_class(config)
        model.to(torch_device)
        model.eval()

        with torch.no_grad():
            outputs = model(**self._prepare_for_class(inputs_dict, model_class))

        hidden_states = outputs.hidden_states

        expected_num_layers = getattr(
            self.model_tester, "expected_num_hidden_layers", len(self.model_tester.depths) + 1
        )
        self.assertEqual(len(hidden_states), expected_num_layers)

        # Swin has a different seq_length
        patch_size = (
            config.patch_size
            if isinstance(config.patch_size, collections.abc.Iterable)
            else (config.patch_size, config.patch_size)
        )

        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])

        self.assertListEqual(
            list(hidden_states[0].shape[-2:]),
            [num_patches, self.model_tester.embed_dim],
        )

    def test_hidden_states_output(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        image_size = (
            self.model_tester.image_size
            if isinstance(self.model_tester.image_size, collections.abc.Iterable)
            else (self.model_tester.image_size, self.model_tester.image_size)
        )

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            self.check_hidden_states_output(inputs_dict, config, model_class, image_size)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True

            self.check_hidden_states_output(inputs_dict, config, model_class, image_size)

    def test_hidden_states_output_with_padding(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.patch_size = 3

        image_size = (
            self.model_tester.image_size
            if isinstance(self.model_tester.image_size, collections.abc.Iterable)
            else (self.model_tester.image_size, self.model_tester.image_size)
        )
        patch_size = (
            config.patch_size
            if isinstance(config.patch_size, collections.abc.Iterable)
            else (config.patch_size, config.patch_size)
        )

        padded_height = image_size[0] + patch_size[0] - (image_size[0] % patch_size[0])
        padded_width = image_size[1] + patch_size[1] - (image_size[1] % patch_size[1])

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            self.check_hidden_states_output(inputs_dict, config, model_class, (padded_height, padded_width))

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True
            self.check_hidden_states_output(inputs_dict, config, model_class, (padded_height, padded_width))

    @unittest.skip(reason="MaskFormerSwin doesn't have pretrained checkpoints")
    def test_model_from_pretrained(self):
        pass

    @unittest.skip(reason="This will be fixed once MaskFormerSwin is replaced by native Swin")
    def test_initialization(self):
        pass

    @unittest.skip(reason="This will be fixed once MaskFormerSwin is replaced by native Swin")
    def test_gradient_checkpointing_backward_compatibility(self):
        pass

    def test_model_outputs_equivalence(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

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
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class)
            check_equivalence(model, tuple_inputs, dict_inputs)

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            check_equivalence(model, tuple_inputs, dict_inputs)

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class)
            check_equivalence(model, tuple_inputs, dict_inputs, {"output_hidden_states": True})

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            check_equivalence(model, tuple_inputs, dict_inputs, {"output_hidden_states": True})


@require_torch
class MaskFormerSwinBackboneTest(unittest.TestCase, BackboneTesterMixin):
    all_model_classes = (MaskFormerSwinBackbone,) if is_torch_available() else ()
    config_class = MaskFormerSwinConfig

    def setUp(self):
        self.model_tester = MaskFormerSwinModelTester(self)

    # Overriding as returned hidden states are tuples of tensors instead of a single tensor
    def test_backbone_outputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        batch_size = inputs_dict["pixel_values"].shape[0]

        for backbone_class in self.all_model_classes:
            backbone = backbone_class(config)
            backbone.to(torch_device)
            backbone.eval()

            outputs = backbone(**inputs_dict)

            # Test default outputs and verify feature maps
            self.assertIsInstance(outputs.feature_maps, tuple)
            self.assertTrue(len(outputs.feature_maps) == len(backbone.channels))
            for feature_map, n_channels in zip(outputs.feature_maps, backbone.channels):
                self.assertTrue(feature_map.shape[:2], (batch_size, n_channels))
            self.assertIsNone(outputs.hidden_states)
            self.assertIsNone(outputs.attentions)

            # Test output_hidden_states=True
            outputs = backbone(**inputs_dict, output_hidden_states=True)
            self.assertIsNotNone(outputs.hidden_states)
            self.assertTrue(len(outputs.hidden_states), len(backbone.stage_names))
            # We skip the stem layer
            for hidden_states, n_channels in zip(outputs.hidden_states[1:], backbone.channels):
                for hidden_state in hidden_states:
                    # Hidden states are in the format (batch_size, (height * width), n_channels)
                    h_batch_size, _, h_n_channels = hidden_state.shape
                    self.assertTrue((h_batch_size, h_n_channels), (batch_size, n_channels))

            # Test output_attentions=True
            if self.has_attentions:
                outputs = backbone(**inputs_dict, output_attentions=True)
                self.assertIsNotNone(outputs.attentions)
