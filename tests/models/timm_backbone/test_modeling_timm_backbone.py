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

import copy
import inspect
import unittest

from transformers import AutoBackbone
from transformers.testing_utils import is_flaky, require_timm, require_torch, torch_device
from transformers.utils.import_utils import is_torch_available

from ...test_backbone_common import BackboneTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor


if is_torch_available():
    import torch

    from transformers import TimmBackbone, TimmBackboneConfig

from ...test_pipeline_mixin import PipelineTesterMixin


class TimmBackboneModelTester:
    def __init__(
        self,
        parent,
        out_indices=None,
        out_features=None,
        stage_names=None,
        backbone="resnet18",
        batch_size=3,
        image_size=32,
        num_channels=3,
        is_training=True,
        use_pretrained_backbone=True,
    ):
        self.parent = parent
        self.out_indices = out_indices if out_indices is not None else [4]
        self.stage_names = stage_names
        self.out_features = out_features
        self.backbone = backbone
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_channels = num_channels
        self.use_pretrained_backbone = use_pretrained_backbone
        self.is_training = is_training

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        config = self.get_config()

        return config, pixel_values

    def get_config(self):
        return TimmBackboneConfig(
            image_size=self.image_size,
            num_channels=self.num_channels,
            out_features=self.out_features,
            out_indices=self.out_indices,
            stage_names=self.stage_names,
            use_pretrained_backbone=self.use_pretrained_backbone,
            backbone=self.backbone,
        )

    def create_and_check_model(self, config, pixel_values):
        model = TimmBackbone(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(pixel_values)
        self.parent.assertEqual(
            result.feature_map[-1].shape,
            (self.batch_size, model.channels[-1], 14, 14),
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
@require_timm
class TimmBackboneModelTest(ModelTesterMixin, BackboneTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (TimmBackbone,) if is_torch_available() else ()
    pipeline_model_mapping = {"feature-extraction": TimmBackbone} if is_torch_available() else {}
    test_resize_embeddings = False
    test_head_masking = False
    test_pruning = False
    has_attentions = False

    def setUp(self):
        # self.config_class = PretrainedConfig
        self.config_class = TimmBackboneConfig
        self.model_tester = TimmBackboneModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=self.config_class, has_text_modality=False, common_properties=["num_channels"]
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    @is_flaky(description="`TimmBackbone` has no `_init_weights`. Timm's way of weight init. seems to give larger magnitude in the intermediate values during `forward`.")
    def test_batching_equivalence(self):
        super().test_batching_equivalence()

    def test_timm_transformer_backbone_equivalence(self):
        timm_checkpoint = "resnet18"
        transformers_checkpoint = "microsoft/resnet-18"

        timm_model = AutoBackbone.from_pretrained(timm_checkpoint, use_timm_backbone=True)
        transformers_model = AutoBackbone.from_pretrained(transformers_checkpoint)

        self.assertEqual(len(timm_model.out_features), len(transformers_model.out_features))
        self.assertEqual(len(timm_model.stage_names), len(transformers_model.stage_names))
        self.assertEqual(timm_model.channels, transformers_model.channels)
        # Out indices are set to the last layer by default. For timm models, we don't know
        # the number of layers in advance, so we set it to (-1,), whereas for transformers
        # models, we set it to [len(stage_names) - 1] (kept for backward compatibility).
        self.assertEqual(timm_model.out_indices, [-1])
        self.assertEqual(transformers_model.out_indices, [len(timm_model.stage_names) - 1])

        timm_model = AutoBackbone.from_pretrained(timm_checkpoint, use_timm_backbone=True, out_indices=[1, 2, 3])
        transformers_model = AutoBackbone.from_pretrained(transformers_checkpoint, out_indices=[1, 2, 3])

        self.assertEqual(timm_model.out_indices, transformers_model.out_indices)
        self.assertEqual(len(timm_model.out_features), len(transformers_model.out_features))
        self.assertEqual(timm_model.channels, transformers_model.channels)

    @unittest.skip(reason="TimmBackbone doesn't support feed forward chunking")
    def test_feed_forward_chunking(self):
        pass

    @unittest.skip(reason="TimmBackbone doesn't have num_hidden_layers attribute")
    def test_hidden_states_output(self):
        pass

    @unittest.skip(reason="TimmBackbone initialization is managed on the timm side")
    def test_initialization(self):
        pass

    @unittest.skip(reason="TimmBackbone models doesn't have inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="TimmBackbone models doesn't have inputs_embeds")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="TimmBackbone model cannot be created without specifying a backbone checkpoint")
    def test_from_pretrained_no_checkpoint(self):
        pass

    @unittest.skip(reason="Only checkpoints on timm can be loaded into TimmBackbone")
    def test_save_load(self):
        pass

    @unittest.skip(reason="No support for low_cpu_mem_usage=True.")
    def test_save_load_low_cpu_mem_usage(self):
        pass

    @unittest.skip(reason="No support for low_cpu_mem_usage=True.")
    def test_save_load_low_cpu_mem_usage_checkpoints(self):
        pass

    @unittest.skip(reason="No support for low_cpu_mem_usage=True.")
    def test_save_load_low_cpu_mem_usage_no_safetensors(self):
        pass

    @unittest.skip(reason="model weights aren't tied in TimmBackbone.")
    def test_tie_model_weights(self):
        pass

    @unittest.skip(reason="model weights aren't tied in TimmBackbone.")
    def test_tied_model_weights_key_ignore(self):
        pass

    @unittest.skip(reason="Only checkpoints on timm can be loaded into TimmBackbone")
    def test_load_save_without_tied_weights(self):
        pass

    @unittest.skip(reason="Only checkpoints on timm can be loaded into TimmBackbone")
    def test_model_weights_reload_no_missing_tied_weights(self):
        pass

    @unittest.skip(reason="TimmBackbone doesn't have hidden size info in its configuration.")
    def test_channels(self):
        pass

    @unittest.skip(reason="TimmBackbone doesn't support output_attentions.")
    def test_torchscript_output_attentions(self):
        pass

    @unittest.skip(reason="Safetensors is not supported by timm.")
    def test_can_use_safetensors(self):
        pass

    @unittest.skip(reason="Need to use a timm backbone and there is no tiny model available.")
    def test_model_is_small(self):
        pass

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["pixel_values"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_retain_grad_hidden_states_attentions(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.output_hidden_states = True
        config.output_attentions = self.has_attentions

        # no need to test all models as different heads yield the same functionality
        model_class = self.all_model_classes[0]
        model = model_class(config)
        model.to(torch_device)

        inputs = self._prepare_for_class(inputs_dict, model_class)
        outputs = model(**inputs)
        output = outputs[0][-1]

        # Encoder-/Decoder-only models
        hidden_states = outputs.hidden_states[0]
        hidden_states.retain_grad()

        if self.has_attentions:
            attentions = outputs.attentions[0]
            attentions.retain_grad()

        output.flatten()[0].backward(retain_graph=True)

        self.assertIsNotNone(hidden_states.grad)

        if self.has_attentions:
            self.assertIsNotNone(attentions.grad)

    # TimmBackbone config doesn't have out_features attribute
    def test_create_from_modified_config(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            result = model(**inputs_dict)

            self.assertEqual(len(result.feature_maps), len(config.out_indices))
            self.assertEqual(len(model.channels), len(config.out_indices))

            # Check output of last stage is taken if out_features=None, out_indices=None
            modified_config = copy.deepcopy(config)
            modified_config.out_indices = None
            model = model_class(modified_config)
            model.to(torch_device)
            model.eval()
            result = model(**inputs_dict)

            self.assertEqual(len(result.feature_maps), 1)
            self.assertEqual(len(model.channels), 1)

            # Check backbone can be initialized with fresh weights
            modified_config = copy.deepcopy(config)
            modified_config.use_pretrained_backbone = False
            model = model_class(modified_config)
            model.to(torch_device)
            model.eval()
            result = model(**inputs_dict)
