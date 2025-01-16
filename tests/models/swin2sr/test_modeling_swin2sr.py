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
"""Testing suite for the PyTorch Swin2SR model."""

import unittest

from transformers import Swin2SRConfig
from transformers.testing_utils import require_torch, require_vision, slow, torch_device
from transformers.utils import is_torch_available, is_vision_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, _config_zero_init, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch
    from torch import nn

    from transformers import Swin2SRForImageSuperResolution, Swin2SRModel

if is_vision_available():
    from PIL import Image

    from transformers import Swin2SRImageProcessor


class Swin2SRModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        image_size=32,
        patch_size=1,
        num_channels=3,
        num_channels_out=1,
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
        use_labels=False,
        upscale=2,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_channels_out = num_channels_out
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
        self.upscale = upscale

        # here we set some attributes to make tests pass
        self.num_hidden_layers = len(depths)
        self.hidden_size = embed_dim
        self.seq_length = (image_size // patch_size) ** 2

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])

        labels = None
        if self.use_labels:
            labels = ids_tensor([self.batch_size], self.type_sequence_label_size)

        config = self.get_config()

        return config, pixel_values, labels

    def get_config(self):
        return Swin2SRConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            num_channels_out=self.num_channels_out,
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
            upscale=self.upscale,
        )

    def create_and_check_model(self, config, pixel_values, labels):
        model = Swin2SRModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)

        self.parent.assertEqual(
            result.last_hidden_state.shape, (self.batch_size, self.embed_dim, self.image_size, self.image_size)
        )

    def create_and_check_for_image_super_resolution(self, config, pixel_values, labels):
        model = Swin2SRForImageSuperResolution(config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)

        expected_image_size = self.image_size * self.upscale
        self.parent.assertEqual(
            result.reconstruction.shape,
            (self.batch_size, self.num_channels_out, expected_image_size, expected_image_size),
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values, labels = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class Swin2SRModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (Swin2SRModel, Swin2SRForImageSuperResolution) if is_torch_available() else ()
    pipeline_model_mapping = (
        {"image-feature-extraction": Swin2SRModel, "image-to-image": Swin2SRForImageSuperResolution}
        if is_torch_available()
        else {}
    )

    fx_compatible = False
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    test_torchscript = False

    def setUp(self):
        self.model_tester = Swin2SRModelTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=Swin2SRConfig,
            embed_dim=37,
            has_text_modality=False,
            common_properties=["image_size", "patch_size", "num_channels"],
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_for_image_super_resolution(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_image_super_resolution(*config_and_inputs)

    # TODO: check if this works again for PyTorch 2.x.y
    @unittest.skip(reason="Got `CUDA error: misaligned address` with PyTorch 2.0.0.")
    def test_multi_gpu_data_parallel_forward(self):
        pass

    @unittest.skip(reason="Swin2SR does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="Swin2SR does not support training yet")
    def test_training(self):
        pass

    @unittest.skip(reason="Swin2SR does not support training yet")
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

    def test_model_get_set_embeddings(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), (nn.Module))
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, nn.Linear))

    @slow
    def test_model_from_pretrained(self):
        model_name = "caidas/swin2SR-classical-sr-x2-64"
        model = Swin2SRModel.from_pretrained(model_name)
        self.assertIsNotNone(model)

    # overwriting because of `logit_scale` parameter
    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                if "logit_scale" in name:
                    continue
                if param.requires_grad:
                    self.assertIn(
                        ((param.data.mean() * 1e9).round() / 1e9).item(),
                        [0.0, 1.0],
                        msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                    )

    def test_attention_outputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            config.return_dict = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            expected_num_attentions = len(self.model_tester.depths)
            self.assertEqual(len(attentions), expected_num_attentions)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            window_size_squared = config.window_size**2
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertEqual(len(attentions), expected_num_attentions)

            self.assertListEqual(
                list(attentions[0].shape[-3:]),
                [self.model_tester.num_heads[0], window_size_squared, window_size_squared],
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

            self.assertEqual(len(self_attentions), expected_num_attentions)

            self.assertListEqual(
                list(self_attentions[0].shape[-3:]),
                [self.model_tester.num_heads[0], window_size_squared, window_size_squared],
            )


@require_vision
@require_torch
@slow
class Swin2SRModelIntegrationTest(unittest.TestCase):
    def test_inference_image_super_resolution_head(self):
        processor = Swin2SRImageProcessor()
        model = Swin2SRForImageSuperResolution.from_pretrained("caidas/swin2SR-classical-sr-x2-64").to(torch_device)

        image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
        inputs = processor(images=image, return_tensors="pt").to(torch_device)

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # verify the logits
        expected_shape = torch.Size([1, 3, 976, 1296])
        self.assertEqual(outputs.reconstruction.shape, expected_shape)
        expected_slice = torch.tensor(
            [[0.5458, 0.5546, 0.5638], [0.5526, 0.5565, 0.5651], [0.5396, 0.5426, 0.5621]]
        ).to(torch_device)
        torch.testing.assert_close(outputs.reconstruction[0, 0, :3, :3], expected_slice, rtol=1e-4, atol=1e-4)

    def test_inference_fp16(self):
        processor = Swin2SRImageProcessor()
        model = Swin2SRForImageSuperResolution.from_pretrained(
            "caidas/swin2SR-classical-sr-x2-64", torch_dtype=torch.float16
        ).to(torch_device)

        image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
        inputs = processor(images=image, return_tensors="pt").to(model.dtype).to(torch_device)

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # verify the logits
        expected_shape = torch.Size([1, 3, 976, 1296])
        self.assertEqual(outputs.reconstruction.shape, expected_shape)
        expected_slice = torch.tensor(
            [[0.5454, 0.5542, 0.5640], [0.5518, 0.5562, 0.5649], [0.5391, 0.5425, 0.5620]], dtype=model.dtype
        ).to(torch_device)
        torch.testing.assert_close(outputs.reconstruction[0, 0, :3, :3], expected_slice, rtol=1e-4, atol=1e-4)
