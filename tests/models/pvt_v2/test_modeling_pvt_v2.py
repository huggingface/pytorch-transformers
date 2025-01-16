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
"""Testing suite for the PyTorch PvtV2 model."""

import inspect
import tempfile
import unittest

from transformers import PvtV2Backbone, PvtV2Config, is_torch_available, is_vision_available
from transformers.models.auto.modeling_auto import MODEL_MAPPING_NAMES
from transformers.testing_utils import (
    require_accelerate,
    require_torch,
    require_torch_accelerator,
    require_torch_fp16,
    slow,
    torch_device,
)

from ...test_backbone_common import BackboneTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import AutoImageProcessor, PvtV2ForImageClassification, PvtV2Model


if is_vision_available():
    from PIL import Image


class PvtV2ConfigTester(ConfigTester):
    def run_common_tests(self):
        config = self.config_class(**self.inputs_dict)
        self.parent.assertTrue(hasattr(config, "hidden_sizes"))
        self.parent.assertTrue(hasattr(config, "num_encoder_blocks"))


class PvtV2ModelTester(ModelTesterMixin):
    def __init__(
        self,
        parent,
        batch_size=13,
        image_size=None,
        num_channels=3,
        num_encoder_blocks=4,
        depths=[2, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        hidden_sizes=[16, 32, 64, 128],
        downsampling_rates=[1, 4, 8, 16],
        num_attention_heads=[1, 2, 4, 8],
        out_indices=[0, 1, 2, 3],
        is_training=True,
        use_labels=True,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        num_labels=3,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = 64 if image_size is None else image_size
        self.num_channels = num_channels
        self.num_encoder_blocks = num_encoder_blocks
        self.sr_ratios = sr_ratios
        self.depths = depths
        self.hidden_sizes = hidden_sizes
        self.downsampling_rates = downsampling_rates
        self.num_attention_heads = num_attention_heads
        self.is_training = is_training
        self.use_labels = use_labels
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.out_indices = out_indices
        self.num_labels = num_labels
        self.scope = scope

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])

        labels = None
        if self.use_labels:
            labels = ids_tensor([self.batch_size, self.image_size, self.image_size], self.num_labels)

        config = self.get_config()
        return config, pixel_values, labels

    def get_config(self):
        return PvtV2Config(
            image_size=self.image_size,
            num_channels=self.num_channels,
            num_encoder_blocks=self.num_encoder_blocks,
            depths=self.depths,
            sr_ratios=self.sr_ratios,
            hidden_sizes=self.hidden_sizes,
            num_attention_heads=self.num_attention_heads,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            initializer_range=self.initializer_range,
            out_indices=self.out_indices,
        )

    def create_and_check_model(self, config, pixel_values, labels):
        model = PvtV2Model(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        self.parent.assertIsNotNone(result.last_hidden_state)

    def create_and_check_backbone(self, config, pixel_values, labels):
        model = PvtV2Backbone(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)

        # verify feature maps
        self.parent.assertEqual(len(result.feature_maps), len(config.out_features))
        self.parent.assertListEqual(list(result.feature_maps[0].shape), [self.batch_size, self.hidden_sizes[1], 4, 4])

        # verify channels
        self.parent.assertEqual(len(model.channels), len(config.out_features))
        self.parent.assertListEqual(model.channels, config.hidden_sizes[1:])

        # verify backbone works with out_features=None
        config.out_features = None
        model = PvtV2Backbone(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)

        # verify feature maps
        self.parent.assertEqual(len(result.feature_maps), 1)
        self.parent.assertListEqual(list(result.feature_maps[0].shape), [self.batch_size, self.hidden_sizes[-1], 1, 1])

        # verify channels
        self.parent.assertEqual(len(model.channels), 1)
        self.parent.assertListEqual(model.channels, [config.hidden_sizes[-1]])

    def create_and_check_for_image_classification(self, config, pixel_values, labels):
        config.num_labels = self.num_labels
        model = PvtV2ForImageClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values, labels=labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

        # test greyscale images
        config.num_channels = 1
        model = PvtV2ForImageClassification(config)
        model.to(torch_device)
        model.eval()

        pixel_values = floats_tensor([self.batch_size, 1, self.image_size, self.image_size])
        result = model(pixel_values)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.type_sequence_label_size))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values, labels = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


# We will verify our results on an image of cute cats
def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image


@require_torch
class PvtV2ModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (PvtV2Model, PvtV2ForImageClassification) if is_torch_available() else ()
    pipeline_model_mapping = (
        {"feature-extraction": PvtV2Model, "image-classification": PvtV2ForImageClassification}
        if is_torch_available()
        else {}
    )

    test_head_masking = False
    test_pruning = False
    test_resize_embeddings = False
    test_torchscript = False
    has_attentions = False

    def setUp(self):
        self.model_tester = PvtV2ModelTester(self)
        self.config_tester = PvtV2ConfigTester(self, config_class=PvtV2Config)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(reason="Pvt-V2 does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="Pvt-V2 does not have get_input_embeddings method and get_output_embeddings methods")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="This architecture does not work with using reentrant.")
    def test_training_gradient_checkpointing(self):
        # Scenario - 1 default behaviour
        self.check_training_gradient_checkpointing()

    @unittest.skip(reason="This architecture does not work with using reentrant.")
    def test_training_gradient_checkpointing_use_reentrant(self):
        # Scenario - 2 with `use_reentrant=True` - this is the default value that is used in pytorch's
        # torch.utils.checkpoint.checkpoint
        self.check_training_gradient_checkpointing(gradient_checkpointing_kwargs={"use_reentrant": True})

    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config=config)
            for name, param in model.named_parameters():
                self.assertTrue(
                    -1.0 <= ((param.data.mean() * 1e9).round() / 1e9).item() <= 1.0,
                    msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                )

    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.hidden_states

            expected_num_layers = len(self.model_tester.depths)
            self.assertEqual(len(hidden_states), expected_num_layers)

            # verify the first hidden states (first block)
            self.assertListEqual(
                list(hidden_states[0].shape[-3:]),
                [
                    self.model_tester.hidden_sizes[self.model_tester.out_indices[0]],
                    self.model_tester.image_size // 2 ** (2 + self.model_tester.out_indices[0]),
                    self.model_tester.image_size // 2 ** (2 + self.model_tester.out_indices[0]),
                ],
            )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True

            check_hidden_states_output(inputs_dict, config, model_class)

    def test_training(self):
        if not self.model_tester.is_training:
            self.skipTest(reason="model_tester.is_training is set to False")

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True

        for model_class in self.all_model_classes:
            if model_class.__name__ in MODEL_MAPPING_NAMES.values():
                continue
            model = model_class(config)
            model.to(torch_device)
            model.train()
            inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            loss = model(**inputs).loss
            loss.backward()

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["pixel_values"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    @slow
    def test_model_from_pretrained(self):
        model_name = "OpenGVLab/pvt_v2_b0"
        model = PvtV2Model.from_pretrained(model_name)
        self.assertIsNotNone(model)


@require_torch
class PvtV2ModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference_image_classification(self):
        # only resize + normalize
        image_processor = AutoImageProcessor.from_pretrained("OpenGVLab/pvt_v2_b0")
        model = PvtV2ForImageClassification.from_pretrained("OpenGVLab/pvt_v2_b0").to(torch_device).eval()

        image = prepare_img()
        encoded_inputs = image_processor(images=image, return_tensors="pt")
        pixel_values = encoded_inputs.pixel_values.to(torch_device)

        with torch.no_grad():
            outputs = model(pixel_values)

        expected_shape = torch.Size((1, model.config.num_labels))
        self.assertEqual(outputs.logits.shape, expected_shape)

        expected_slice = torch.tensor([-1.4192, -1.9158, -0.9702]).to(torch_device)

        torch.testing.assert_close(outputs.logits[0, :3], expected_slice, rtol=1e-4, atol=1e-4)

    @slow
    def test_inference_model(self):
        model = PvtV2Model.from_pretrained("OpenGVLab/pvt_v2_b0").to(torch_device).eval()

        image_processor = AutoImageProcessor.from_pretrained("OpenGVLab/pvt_v2_b0")
        image = prepare_img()
        inputs = image_processor(images=image, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(torch_device)

        # forward pass
        with torch.no_grad():
            outputs = model(pixel_values)

        # verify the logits
        expected_shape = torch.Size((1, 50, 512))
        self.assertEqual(outputs.last_hidden_state.shape, expected_shape)

        expected_slice = torch.tensor(
            [[-0.3086, 1.0402, 1.1816], [-0.2880, 0.5781, 0.6124], [0.1480, 0.6129, -0.0590]]
        ).to(torch_device)

        torch.testing.assert_close(outputs.last_hidden_state[0, :3, :3], expected_slice, rtol=1e-4, atol=1e-4)

    @slow
    @require_accelerate
    @require_torch_accelerator
    @require_torch_fp16
    def test_inference_fp16(self):
        r"""
        A small test to make sure that inference work in half precision without any problem.
        """
        model = PvtV2ForImageClassification.from_pretrained("OpenGVLab/pvt_v2_b0", torch_dtype=torch.float16)
        model.to(torch_device)
        image_processor = AutoImageProcessor.from_pretrained("OpenGVLab/pvt_v2_b0")

        image = prepare_img()
        inputs = image_processor(images=image, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(torch_device, dtype=torch.float16)

        # forward pass to make sure inference works in fp16
        with torch.no_grad():
            _ = model(pixel_values)


@require_torch
class PvtV2BackboneTest(BackboneTesterMixin, unittest.TestCase):
    all_model_classes = (PvtV2Backbone,) if is_torch_available() else ()
    has_attentions = False
    config_class = PvtV2Config

    def test_config(self):
        config_class = self.config_class

        # test default config
        config = config_class()
        self.assertIsNotNone(config)
        num_stages = len(config.depths) if hasattr(config, "depths") else config.num_hidden_layers
        expected_stage_names = [f"stage{idx}" for idx in range(1, num_stages + 1)]
        self.assertEqual(config.stage_names, expected_stage_names)
        self.assertTrue(set(config.out_features).issubset(set(config.stage_names)))

        # Test out_features and out_indices are correctly set
        # out_features and out_indices both None
        config = config_class(out_features=None, out_indices=None)
        self.assertEqual(config.out_features, [config.stage_names[-1]])
        self.assertEqual(config.out_indices, [len(config.stage_names) - 1])

        # out_features and out_indices both set
        config = config_class(out_features=["stage1", "stage2"], out_indices=[0, 1])
        self.assertEqual(config.out_features, ["stage1", "stage2"])
        self.assertEqual(config.out_indices, [0, 1])

        # Only out_features set
        config = config_class(out_features=["stage2", "stage4"])
        self.assertEqual(config.out_features, ["stage2", "stage4"])
        self.assertEqual(config.out_indices, [1, 3])

        # Only out_indices set
        config = config_class(out_indices=[0, 2])
        self.assertEqual(config.out_features, [config.stage_names[0], config.stage_names[2]])
        self.assertEqual(config.out_indices, [0, 2])

        # Error raised when out_indices do not correspond to out_features
        with self.assertRaises(ValueError):
            config = config_class(out_features=["stage1", "stage2"], out_indices=[0, 2])

    def test_config_save_pretrained(self):
        config_class = self.config_class
        config_first = config_class(out_indices=[0, 1, 2, 3])

        with tempfile.TemporaryDirectory() as tmpdirname:
            config_first.save_pretrained(tmpdirname)
            config_second = self.config_class.from_pretrained(tmpdirname)

        # Fix issue where type switches in the saving process
        if isinstance(config_second.image_size, list):
            config_second.image_size = tuple(config_second.image_size)

        self.assertEqual(config_second.to_dict(), config_first.to_dict())

    def setUp(self):
        self.model_tester = PvtV2ModelTester(self)
