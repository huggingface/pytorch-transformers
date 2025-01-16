# coding=utf-8
# Copyright 2023 The Intel Team Authors, The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch TVP model."""

import unittest

from transformers import ResNetConfig, TimmBackboneConfig, TvpConfig
from transformers.testing_utils import require_timm, require_torch, require_vision, torch_device
from transformers.utils import cached_property, is_torch_available, is_vision_available

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

    from transformers import TvpForVideoGrounding, TvpModel

if is_vision_available():
    from PIL import Image

    from transformers import TvpImageProcessor


# Copied from test.models.videomae.test_modeling_videomae.VideoMAEModelTester with VideoMAE->TVP
class TVPModelTester:
    def __init__(
        self,
        parent,
        batch_size=1,
        seq_length=2,
        alpha=1.0,
        beta=0.1,
        visual_prompter_type="framepad",
        visual_prompter_apply="replace",
        num_frames=2,
        max_img_size=448,
        visual_prompt_size=96,
        vocab_size=100,
        hidden_size=32,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=30,
        max_grid_col_position_embeddings=30,
        max_grid_row_position_embeddings=30,
        hidden_dropout_prob=0.1,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
        initializer_range=0.02,
        pad_token_id=0,
        type_vocab_size=2,
        attention_probs_dropout_prob=0.1,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.input_id_length = seq_length
        self.seq_length = seq_length + 10 + 784  # include text prompt length and visual input length
        self.alpha = alpha
        self.beta = beta
        self.visual_prompter_type = visual_prompter_type
        self.visual_prompter_apply = visual_prompter_apply
        self.num_frames = num_frames
        self.max_img_size = max_img_size
        self.visual_prompt_size = visual_prompt_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.max_grid_col_position_embeddings = max_grid_col_position_embeddings
        self.max_grid_row_position_embeddings = max_grid_row_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.pad_token_id = pad_token_id
        self.type_vocab_size = type_vocab_size
        self.is_training = False
        self.num_channels = 3

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.input_id_length], self.vocab_size)
        attention_mask = random_attention_mask([self.batch_size, self.input_id_length])
        pixel_values = floats_tensor(
            [self.batch_size, self.num_frames, self.num_channels, self.max_img_size, self.max_img_size]
        )

        config = self.get_config()

        return (config, input_ids, pixel_values, attention_mask)

    def get_config(self):
        resnet_config = ResNetConfig(
            num_channels=3,
            embeddings_size=64,
            hidden_sizes=[64, 128],
            depths=[2, 2],
            hidden_act="relu",
            out_features=["stage2"],
            out_indices=[2],
        )
        return TvpConfig(
            backbone_config=resnet_config,
            backbone=None,
            alpha=self.alpha,
            beta=self.beta,
            visual_prompter_type=self.visual_prompter_type,
            visual_prompter_apply=self.visual_prompter_apply,
            num_frames=self.num_frames,
            max_img_size=self.max_img_size,
            visual_prompt_size=self.visual_prompt_size,
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            max_grid_col_position_embeddings=self.max_grid_col_position_embeddings,
            max_grid_row_position_embeddings=self.max_grid_row_position_embeddings,
            layer_norm_eps=self.layer_norm_eps,
            initializer_range=self.initializer_range,
            pad_token_id=self.pad_token_id,
            type_vocab_size=self.type_vocab_size,
        )

    def create_and_check_model(self, config, input_ids, pixel_values, attention_mask):
        model = TvpModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, pixel_values, attention_mask)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, pixel_values, attention_mask = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "pixel_values": pixel_values, "attention_mask": attention_mask}
        return config, inputs_dict


@require_torch
class TVPModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as TVP does not use, inputs_embeds.
    The seq_length in TVP contain textual and visual inputs, and prompt.
    """

    all_model_classes = (TvpModel, TvpForVideoGrounding) if is_torch_available() else ()
    pipeline_model_mapping = (
        {"feature-extraction": TvpModel, "temporal-video-grounding": TvpForVideoGrounding}
        if is_torch_available()
        else {}
    )

    # TODO: Enable this once this model gets more usage
    test_torchscript = False

    def setUp(self):
        self.model_tester = TVPModelTester(self)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(reason="TVP does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="TVPModel does not have input/output embeddings")
    def test_model_get_set_embeddings(self):
        pass

    # override as the `logit_scale` parameter initilization is different for TVP
    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    # params are randomly initialized.
                    self.assertAlmostEqual(
                        param.data.mean().item(),
                        0.0,
                        delta=1.0,
                        msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                    )

    @require_timm
    def test_backbone_selection(self):
        def _validate_backbone_init():
            for model_class in self.all_model_classes:
                model = model_class(config)
                model.to(torch_device)
                model.eval()

                # Confirm out_indices propogated to backbone
                if model.__class__.__name__ == "TvpModel":
                    self.assertEqual(len(model.vision_model.backbone.out_indices), 2)
                elif model.__class__.__name__ == "TvpForVideoGrounding":
                    self.assertEqual(len(model.model.vision_model.backbone.out_indices), 2)

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        # Force load_backbone path
        config.is_hybrid = False

        # We load through configs, as the modeling file assumes config.backbone_config is always set
        config.use_pretrained_backbone = False
        config.backbone_kwargs = None

        # Load a timm backbone
        # We hack adding hidden_sizes to the config to test the backbone loading
        backbone_config = TimmBackboneConfig("resnet18", out_indices=[-2, -1], hidden_sizes=[64, 128])
        config.backbone_config = backbone_config
        _validate_backbone_init()

        # Load a HF backbone
        backbone_config = ResNetConfig.from_pretrained("facebook/dinov2-small", out_indices=[-2, -1])
        config.backbone_config = backbone_config
        _validate_backbone_init()


# We will verify our results on an image of cute cats
def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image


@require_vision
@require_torch
class TvpModelIntegrationTests(unittest.TestCase):
    @cached_property
    def default_image_processor(self):
        return TvpImageProcessor.from_pretrained("Jiqing/tiny-random-tvp")

    def test_inference_no_head(self):
        model = TvpModel.from_pretrained("Jiqing/tiny-random-tvp").to(torch_device)

        image_processor = self.default_image_processor
        image = prepare_img()
        encoding = image_processor(images=image, return_tensors="pt")
        input_ids = torch.tensor([[1, 2]])
        attention_mask = torch.tensor([[1, 1]])
        encoding.update({"input_ids": input_ids, "attention_mask": attention_mask})
        encoding.to(torch_device)

        with torch.no_grad():
            outputs = model(**encoding)

        expected_shape = torch.Size((1, 796, 128))
        assert outputs.last_hidden_state.shape == expected_shape
        expected_slice = torch.tensor(
            [[-0.4902, -0.4121, -1.7872], [-0.2184, 2.1211, -0.9371], [0.1180, 0.5003, -0.1727]]
        ).to(torch_device)
        torch.testing.assert_close(outputs.last_hidden_state[0, :3, :3], expected_slice, rtol=1e-4, atol=1e-4)

    def test_inference_with_head(self):
        model = TvpForVideoGrounding.from_pretrained("Jiqing/tiny-random-tvp").to(torch_device)

        image_processor = self.default_image_processor
        image = prepare_img()
        encoding = image_processor(images=image, return_tensors="pt")
        input_ids = torch.tensor([[1, 2]])
        attention_mask = torch.tensor([[1, 1]])
        encoding.update({"input_ids": input_ids, "attention_mask": attention_mask})
        encoding.to(torch_device)

        with torch.no_grad():
            outputs = model(**encoding)

        expected_shape = torch.Size((1, 2))
        assert outputs.logits.shape == expected_shape
        expected_slice = torch.tensor([[0.5061, 0.4988]]).to(torch_device)
        torch.testing.assert_close(outputs.logits, expected_slice, rtol=1e-4, atol=1e-4)

    def test_interpolate_inference_no_head(self):
        model = TvpModel.from_pretrained("Jiqing/tiny-random-tvp").to(torch_device)

        image_processor = self.default_image_processor
        image = prepare_img()  # 480X640
        encoding = image_processor(
            images=image, return_tensors="pt", do_resize=False, do_pad=False, do_center_crop=False
        )
        input_ids = torch.tensor([[1, 2]])
        attention_mask = torch.tensor([[1, 1]])
        encoding.update({"input_ids": input_ids, "attention_mask": attention_mask})
        encoding.to(torch_device)

        with torch.no_grad():
            outputs = model(**encoding, interpolate_pos_encoding=True)

        expected_shape = torch.Size((1, 1212, 128))
        assert outputs.last_hidden_state.shape == expected_shape

    def test_interpolate_inference_with_head(self):
        model = TvpForVideoGrounding.from_pretrained("Jiqing/tiny-random-tvp").to(torch_device)

        image_processor = self.default_image_processor
        image = prepare_img()  # 480X640
        encoding = image_processor(
            images=image, return_tensors="pt", do_resize=False, do_pad=False, do_center_crop=False
        )
        input_ids = torch.tensor([[1, 2]])
        attention_mask = torch.tensor([[1, 1]])
        encoding.update({"input_ids": input_ids, "attention_mask": attention_mask})
        encoding.to(torch_device)

        with torch.no_grad():
            outputs = model(**encoding, interpolate_pos_encoding=True, output_hidden_states=True)

        expected_shape = torch.Size((1, 1212, 128))
        assert outputs.hidden_states[-1].shape == expected_shape
