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
"""Testing suite for the PyTorch ViTMAE model."""

import math
import tempfile
import unittest

import numpy as np

from transformers import ViTMAEConfig
from transformers.testing_utils import require_torch, require_vision, slow, torch_device
from transformers.utils import cached_property, is_torch_available, is_vision_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch
    from torch import nn

    from transformers import ViTMAEForPreTraining, ViTMAEModel


if is_vision_available():
    from PIL import Image

    from transformers import ViTImageProcessor


class ViTMAEModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        image_size=30,
        patch_size=2,
        num_channels=3,
        is_training=True,
        use_labels=True,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        type_sequence_label_size=10,
        initializer_range=0.02,
        num_labels=3,
        scope=None,
        mask_ratio=0.5,
        attn_implementation="eager",
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.is_training = is_training
        self.use_labels = use_labels
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.mask_ratio = mask_ratio
        self.scope = scope
        self.attn_implementation = attn_implementation

        # in ViTMAE, the expected sequence length = (num_patches + 1) * (1 - config.mask_ratio), rounded above
        # (we add 1 for the [CLS] token)
        num_patches = (image_size // patch_size) ** 2
        self.seq_length = int(math.ceil((1 - mask_ratio) * (num_patches + 1)))
        self.mask_ratio = mask_ratio
        self.num_masks = int(mask_ratio * self.seq_length)
        self.mask_length = num_patches

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])

        labels = None
        if self.use_labels:
            labels = ids_tensor([self.batch_size], self.type_sequence_label_size)

        config = self.get_config()

        return config, pixel_values, labels

    def get_config(self):
        return ViTMAEConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            is_decoder=False,
            initializer_range=self.initializer_range,
            mask_ratio=self.mask_ratio,
            decoder_hidden_size=self.hidden_size,
            decoder_intermediate_size=self.intermediate_size,
            decoder_num_attention_heads=self.num_attention_heads,
            decoder_num_hidden_layers=self.num_hidden_layers,
            attn_implementation=self.attn_implementation,
        )

    def create_and_check_model(self, config, pixel_values, labels):
        model = ViTMAEModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_for_pretraining(self, config, pixel_values, labels):
        model = ViTMAEForPreTraining(config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        num_patches = (self.image_size // self.patch_size) ** 2
        expected_num_channels = self.patch_size**2 * self.num_channels
        self.parent.assertEqual(result.logits.shape, (self.batch_size, num_patches, expected_num_channels))

        # test greyscale images
        config.num_channels = 1
        model = ViTMAEForPreTraining(config)
        model.to(torch_device)
        model.eval()
        pixel_values = floats_tensor([self.batch_size, 1, self.image_size, self.image_size])
        result = model(pixel_values)
        expected_num_channels = self.patch_size**2
        self.parent.assertEqual(result.logits.shape, (self.batch_size, num_patches, expected_num_channels))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values, labels = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class ViTMAEModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as ViTMAE does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (ViTMAEModel, ViTMAEForPreTraining) if is_torch_available() else ()
    pipeline_model_mapping = {"image-feature-extraction": ViTMAEModel} if is_torch_available() else {}

    test_pruning = False
    test_torchscript = False
    test_resize_embeddings = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = ViTMAEModelTester(self)
        self.config_tester = ConfigTester(self, config_class=ViTMAEConfig, has_text_modality=False, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="ViTMAE does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    def test_model_get_set_embeddings(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), (nn.Module))
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, nn.Linear))

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_pretraining(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_pretraining(*config_and_inputs)

    # overwrite from common since ViTMAEForPretraining has random masking, we need to fix the noise
    # to generate masks during test
    def check_pt_tf_models(self, tf_model, pt_model, pt_inputs_dict):
        # make masks reproducible
        np.random.seed(2)

        num_patches = int((pt_model.config.image_size // pt_model.config.patch_size) ** 2)
        noise = np.random.uniform(size=(self.model_tester.batch_size, num_patches))
        pt_noise = torch.from_numpy(noise)

        # Add `noise` argument.
        # PT inputs will be prepared in `super().check_pt_tf_models()` with this added `noise` argument
        pt_inputs_dict["noise"] = pt_noise

        super().check_pt_tf_models(tf_model, pt_model, pt_inputs_dict)

    def test_save_load(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            # make random mask reproducible
            torch.manual_seed(2)
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            out_2 = outputs[0].cpu().numpy()
            out_2[np.isnan(out_2)] = 0

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model = model_class.from_pretrained(tmpdirname)
                model.to(torch_device)
                # make random mask reproducible
                torch.manual_seed(2)
                with torch.no_grad():
                    after_outputs = model(**self._prepare_for_class(inputs_dict, model_class))

                # Make sure we don't have nans
                out_1 = after_outputs[0].cpu().numpy()
                out_1[np.isnan(out_1)] = 0
                max_diff = np.amax(np.abs(out_1 - out_2))
                self.assertLessEqual(max_diff, 1e-5)

    @unittest.skip(
        reason="""ViTMAE returns a random mask + ids_restore in each forward pass. See test_save_load
    to get deterministic results."""
    )
    def test_determinism(self):
        pass

    @unittest.skip(
        reason="""ViTMAE returns a random mask + ids_restore in each forward pass. See test_save_load
    to get deterministic results."""
    )
    def test_save_load_fast_init_from_base(self):
        pass

    @unittest.skip(
        reason="""ViTMAE returns a random mask + ids_restore in each forward pass. See test_save_load
    to get deterministic results."""
    )
    def test_save_load_fast_init_to_base(self):
        pass

    @unittest.skip(reason="""ViTMAE returns a random mask + ids_restore in each forward pass. See test_save_load""")
    def test_model_outputs_equivalence(self):
        pass

    @unittest.skip(reason="ViTMAE returns a random mask + ids_restore in each forward pass")
    def test_batching_equivalence(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        model_name = "google/vit-base-patch16-224"
        model = ViTMAEModel.from_pretrained(model_name)
        self.assertIsNotNone(model)


# We will verify our results on an image of cute cats
def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image


@require_torch
@require_vision
class ViTMAEModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_image_processor(self):
        return ViTImageProcessor.from_pretrained("facebook/vit-mae-base")

    @cached_property
    def default_model(self):
        return ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base").to(torch_device)

    @slow
    def test_inference_for_pretraining(self):
        # make random mask reproducible across the PT and TF model
        np.random.seed(2)

        model = self.default_model

        image_processor = self.default_image_processor
        image = prepare_img()
        inputs = image_processor(images=image, return_tensors="pt").to(torch_device)

        # prepare a noise vector that will be also used for testing the TF model
        # (this way we can ensure that the PT and TF models operate on the same inputs)
        vit_mae_config = ViTMAEConfig()
        num_patches = int((vit_mae_config.image_size // vit_mae_config.patch_size) ** 2)
        noise = torch.from_numpy(np.random.uniform(size=(1, num_patches))).to(device=torch_device)

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs, noise=noise)

        # verify the logits
        expected_shape = torch.Size((1, 196, 768))
        self.assertEqual(outputs.logits.shape, expected_shape)

        expected_slice = torch.tensor(
            [[-0.0548, -1.7023, -0.9325], [0.3721, -0.5670, -0.2233], [0.8235, -1.3878, -0.3524]]
        )

        torch.testing.assert_close(outputs.logits[0, :3, :3], expected_slice.to(torch_device), rtol=1e-4, atol=1e-4)

    @slow
    def test_inference_interpolate_pos_encoding(self):
        # ViTMAE models have an `interpolate_pos_encoding` argument in their forward method,
        # allowing to interpolate the pre-trained position embeddings in order to use
        # the model on higher resolutions. The DINO model by Facebook AI leverages this
        # to visualize self-attention on higher resolution images.

        # make random mask reproducible across the PT and TF model
        np.random.seed(2)

        model = self.default_model

        image_processor = self.default_image_processor
        image = prepare_img()
        inputs = image_processor(images=image, return_tensors="pt", do_resize=False).to(torch_device)

        # prepare a noise vector that will be also used for testing the TF model
        # (this way we can ensure that the PT and TF models operate on the same inputs)
        vit_mae_config = ViTMAEConfig()
        num_patches = (image.height // vit_mae_config.patch_size) * (image.width // vit_mae_config.patch_size)
        noise = torch.from_numpy(np.random.uniform(size=(1, num_patches))).to(device=torch_device)

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs, noise=noise, interpolate_pos_encoding=True)

        # verify the logits
        expected_shape = torch.Size((1, 1200, 768))
        self.assertEqual(outputs.logits.shape, expected_shape)

    @slow
    def test_inference_interpolate_pos_encoding_custom_sizes(self):
        # Ensure custom sizes are correctly handled when interpolating the position embeddings

        # make random mask reproducible across the PT and TF model
        np.random.seed(2)

        model = self.default_model
        image_processor = self.default_image_processor

        image = prepare_img()
        inputs = image_processor(images=image, return_tensors="pt", size={"height": 256, "width": 256}).to(
            torch_device
        )

        # forward pass
        with torch.no_grad():
            outputs = model(
                **inputs,
                interpolate_pos_encoding=True,
            )

        # verify the logits
        expected_shape = torch.Size((1, 256, 768))
        self.assertEqual(outputs.logits.shape, expected_shape)
