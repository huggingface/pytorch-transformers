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
"""Testing suite for the PyTorch VitPose model."""

import inspect
import unittest

import requests

from transformers import VitPoseBackboneConfig, VitPoseConfig
from transformers.testing_utils import require_torch, require_vision, slow, torch_device
from transformers.utils import cached_property, is_torch_available, is_vision_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor


if is_torch_available():
    import torch

    from transformers import VitPoseForPoseEstimation


if is_vision_available():
    from PIL import Image

    from transformers import VitPoseImageProcessor


class VitPoseModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        image_size=[16 * 8, 12 * 8],
        patch_size=[8, 8],
        num_channels=3,
        is_training=True,
        use_labels=True,
        hidden_size=32,
        num_hidden_layers=5,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        type_sequence_label_size=10,
        initializer_range=0.02,
        num_labels=2,
        scale_factor=4,
        out_indices=[-1],
        scope=None,
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
        self.num_labels = num_labels
        self.scale_factor = scale_factor
        self.out_indices = out_indices
        self.scope = scope

        # in VitPose, the seq length equals the number of patches
        num_patches = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])
        self.seq_length = num_patches

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size[0], self.image_size[1]])

        labels = None
        if self.use_labels:
            labels = ids_tensor([self.batch_size], self.type_sequence_label_size)

        config = self.get_config()

        return config, pixel_values, labels

    def get_config(self):
        return VitPoseConfig(
            backbone_config=self.get_backbone_config(),
        )

    def get_backbone_config(self):
        return VitPoseBackboneConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            num_hidden_layers=self.num_hidden_layers,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_attention_heads=self.num_attention_heads,
            hidden_act=self.hidden_act,
            out_indices=self.out_indices,
        )

    def create_and_check_for_pose_estimation(self, config, pixel_values, labels):
        model = VitPoseForPoseEstimation(config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)

        expected_height = (self.image_size[0] // self.patch_size[0]) * self.scale_factor
        expected_width = (self.image_size[1] // self.patch_size[1]) * self.scale_factor

        self.parent.assertEqual(
            result.heatmaps.shape, (self.batch_size, self.num_labels, expected_height, expected_width)
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            pixel_values,
            labels,
        ) = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class VitPoseModelTest(ModelTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as VitPose does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (VitPoseForPoseEstimation,) if is_torch_available() else ()
    fx_compatible = False

    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = VitPoseModelTester(self)
        self.config_tester = ConfigTester(self, config_class=VitPoseConfig, has_text_modality=False, hidden_size=37)

    def test_config(self):
        self.config_tester.create_and_test_config_to_json_string()
        self.config_tester.create_and_test_config_to_json_file()
        self.config_tester.create_and_test_config_from_and_save_pretrained()
        self.config_tester.create_and_test_config_with_num_labels()
        self.config_tester.check_config_can_be_init_without_params()
        self.config_tester.check_config_arguments_init()

    @unittest.skip(reason="VitPose does not support input and output embeddings")
    def test_model_common_attributes(self):
        pass

    @unittest.skip(reason="VitPose does not support input and output embeddings")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="VitPose does not support input and output embeddings")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="VitPose does not support training yet")
    def test_training(self):
        pass

    @unittest.skip(reason="VitPose does not support training yet")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(reason="VitPose does not support training yet")
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(reason="VitPose does not support training yet")
    def test_training_gradient_checkpointing_use_reentrant_false(self):
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

    def test_for_pose_estimation(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_pose_estimation(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        model_name = "usyd-community/vitpose-base-simple"
        model = VitPoseForPoseEstimation.from_pretrained(model_name)
        self.assertIsNotNone(model)


# We will verify our results on an image of people in house
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000000139.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    return image


@require_torch
@require_vision
class VitPoseModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_image_processor(self):
        return (
            VitPoseImageProcessor.from_pretrained("usyd-community/vitpose-base-simple")
            if is_vision_available()
            else None
        )

    @slow
    def test_inference_pose_estimation(self):
        image_processor = self.default_image_processor
        model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple")
        model.to(torch_device)
        model.eval()

        image = prepare_img()
        boxes = [[[412.8, 157.61, 53.05, 138.01], [384.43, 172.21, 15.12, 35.74]]]

        inputs = image_processor(images=image, boxes=boxes, return_tensors="pt").to(torch_device)

        with torch.no_grad():
            outputs = model(**inputs)
        heatmaps = outputs.heatmaps

        assert heatmaps.shape == (2, 17, 64, 48)

        expected_slice = torch.tensor(
            [
                [9.9330e-06, 9.9330e-06, 9.9330e-06],
                [9.9330e-06, 9.9330e-06, 9.9330e-06],
                [9.9330e-06, 9.9330e-06, 9.9330e-06],
            ]
        ).to(torch_device)

        assert torch.allclose(heatmaps[0, 0, :3, :3], expected_slice, atol=1e-4)

        pose_results = image_processor.post_process_pose_estimation(outputs, boxes=boxes)[0]

        expected_bbox = torch.tensor([391.9900, 190.0800, 391.1575, 189.3034])
        expected_keypoints = torch.tensor(
            [
                [3.9813e02, 1.8184e02],
                [3.9828e02, 1.7981e02],
                [3.9596e02, 1.7948e02],
            ]
        )
        expected_scores = torch.tensor([8.7529e-01, 8.4315e-01, 9.2678e-01])

        self.assertEqual(len(pose_results), 2)
        self.assertTrue(torch.allclose(pose_results[1]["bbox"].cpu(), expected_bbox, atol=1e-4))
        self.assertTrue(torch.allclose(pose_results[1]["keypoints"][:3].cpu(), expected_keypoints, atol=1e-2))
        self.assertTrue(torch.allclose(pose_results[1]["scores"][:3].cpu(), expected_scores, atol=1e-4))

    @slow
    def test_batched_inference(self):
        image_processor = self.default_image_processor
        model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple")
        model.to(torch_device)
        model.eval()

        image = prepare_img()
        boxes = [
            [[412.8, 157.61, 53.05, 138.01], [384.43, 172.21, 15.12, 35.74]],
            [[412.8, 157.61, 53.05, 138.01], [384.43, 172.21, 15.12, 35.74]],
        ]

        inputs = image_processor(images=[image, image], boxes=boxes, return_tensors="pt").to(torch_device)

        with torch.no_grad():
            outputs = model(**inputs)
        heatmaps = outputs.heatmaps

        assert heatmaps.shape == (4, 17, 64, 48)

        expected_slice = torch.tensor(
            [
                [9.9330e-06, 9.9330e-06, 9.9330e-06],
                [9.9330e-06, 9.9330e-06, 9.9330e-06],
                [9.9330e-06, 9.9330e-06, 9.9330e-06],
            ]
        ).to(torch_device)

        assert torch.allclose(heatmaps[0, 0, :3, :3], expected_slice, atol=1e-4)

        pose_results = image_processor.post_process_pose_estimation(outputs, boxes=boxes)
        print(pose_results)

        expected_bbox = torch.tensor([391.9900, 190.0800, 391.1575, 189.3034])
        expected_keypoints = torch.tensor(
            [
                [3.9813e02, 1.8184e02],
                [3.9828e02, 1.7981e02],
                [3.9596e02, 1.7948e02],
            ]
        )
        expected_scores = torch.tensor([8.7529e-01, 8.4315e-01, 9.2678e-01])

        self.assertEqual(len(pose_results), 2)
        self.assertEqual(len(pose_results[0]), 2)
        self.assertTrue(torch.allclose(pose_results[0][1]["bbox"].cpu(), expected_bbox, atol=1e-4))
        self.assertTrue(torch.allclose(pose_results[0][1]["keypoints"][:3].cpu(), expected_keypoints, atol=1e-2))
        self.assertTrue(torch.allclose(pose_results[0][1]["scores"][:3].cpu(), expected_scores, atol=1e-4))
