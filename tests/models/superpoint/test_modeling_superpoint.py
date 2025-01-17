# Copyright 2024 The HuggingFace Team. All rights reserved.
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
import inspect
import unittest
from typing import List

from transformers.models.superpoint.configuration_superpoint import SuperPointConfig
from transformers.testing_utils import is_flaky, require_torch, require_vision, slow, torch_device
from transformers.utils import cached_property, is_torch_available, is_vision_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor


if is_torch_available():
    import torch

    from transformers import (
        SuperPointForKeypointDetection,
    )

if is_vision_available():
    from PIL import Image

    from transformers import AutoImageProcessor


class SuperPointModelTester:
    def __init__(
        self,
        parent,
        batch_size=3,
        image_width=80,
        image_height=60,
        encoder_hidden_sizes: List[int] = [32, 32, 64, 64],
        decoder_hidden_size: int = 128,
        keypoint_decoder_dim: int = 65,
        descriptor_decoder_dim: int = 128,
        keypoint_threshold: float = 0.005,
        max_keypoints: int = -1,
        nms_radius: int = 4,
        border_removal_distance: int = 4,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_width = image_width
        self.image_height = image_height

        self.encoder_hidden_sizes = encoder_hidden_sizes
        self.decoder_hidden_size = decoder_hidden_size
        self.keypoint_decoder_dim = keypoint_decoder_dim
        self.descriptor_decoder_dim = descriptor_decoder_dim
        self.keypoint_threshold = keypoint_threshold
        self.max_keypoints = max_keypoints
        self.nms_radius = nms_radius
        self.border_removal_distance = border_removal_distance

    def prepare_config_and_inputs(self):
        # SuperPoint expects a grayscale image as input
        pixel_values = floats_tensor([self.batch_size, 3, self.image_height, self.image_width])
        config = self.get_config()
        return config, pixel_values

    def get_config(self):
        return SuperPointConfig(
            encoder_hidden_sizes=self.encoder_hidden_sizes,
            decoder_hidden_size=self.decoder_hidden_size,
            keypoint_decoder_dim=self.keypoint_decoder_dim,
            descriptor_decoder_dim=self.descriptor_decoder_dim,
            keypoint_threshold=self.keypoint_threshold,
            max_keypoints=self.max_keypoints,
            nms_radius=self.nms_radius,
            border_removal_distance=self.border_removal_distance,
        )

    def create_and_check_keypoint_detection(self, config, pixel_values):
        model = SuperPointForKeypointDetection(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        self.parent.assertEqual(result.keypoints.shape[0], self.batch_size)
        self.parent.assertEqual(result.keypoints.shape[-1], 2)

        result = model(pixel_values, output_hidden_states=True)
        self.parent.assertEqual(
            result.hidden_states[-1].shape,
            (
                self.batch_size,
                self.encoder_hidden_sizes[-1],
                self.image_height // 8,
                self.image_width // 8,
            ),
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class SuperPointModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (SuperPointForKeypointDetection,) if is_torch_available() else ()
    all_generative_model_classes = () if is_torch_available() else ()

    fx_compatible = False
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    has_attentions = False
    from_pretrained_id = "magic-leap-community/superpoint"

    def setUp(self):
        self.model_tester = SuperPointModelTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=SuperPointConfig,
            has_text_modality=False,
            hidden_size=37,
            common_properties=["encoder_hidden_sizes", "decoder_hidden_size"],
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    @is_flaky(description="The `indices` computed with `topk()` in `top_k_keypoints` is not stable.")
    def test_batching_equivalence(self):
        super().test_batching_equivalence()

    @unittest.skip(reason="SuperPointForKeypointDetection does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="SuperPointForKeypointDetection does not support input and output embeddings")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="SuperPointForKeypointDetection does not use feedforward chunking")
    def test_feed_forward_chunking(self):
        pass

    @unittest.skip(reason="SuperPointForKeypointDetection does not support training")
    def test_training(self):
        pass

    @unittest.skip(reason="SuperPointForKeypointDetection does not support training")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(reason="SuperPointForKeypointDetection does not support training")
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(reason="SuperPointForKeypointDetection does not support training")
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip(reason="SuperPoint does not output any loss term in the forward pass")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    def test_keypoint_detection(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_keypoint_detection(*config_and_inputs)

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["pixel_values"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.hidden_states

            # SuperPoint's feature maps are of shape (batch_size, num_channels, width, height)
            for i, conv_layer_size in enumerate(self.model_tester.encoder_hidden_sizes[:-1]):
                self.assertListEqual(
                    list(hidden_states[i].shape[-3:]),
                    [
                        conv_layer_size,
                        self.model_tester.image_height // (2 ** (i + 1)),
                        self.model_tester.image_width // (2 ** (i + 1)),
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

    @slow
    def test_model_from_pretrained(self):
        model = SuperPointForKeypointDetection.from_pretrained(self.from_pretrained_id)
        self.assertIsNotNone(model)

    def test_forward_labels_should_be_none(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                model_inputs = self._prepare_for_class(inputs_dict, model_class)
                # Provide an arbitrary sized Tensor as labels to model inputs
                model_inputs["labels"] = torch.rand((128, 128))

                with self.assertRaises(ValueError) as cm:
                    model(**model_inputs)
                self.assertEqual(ValueError, cm.exception.__class__)


def prepare_imgs():
    image1 = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    image2 = Image.open("./tests/fixtures/tests_samples/COCO/000000004016.png")
    return [image1, image2]


@require_torch
@require_vision
class SuperPointModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_image_processor(self):
        return AutoImageProcessor.from_pretrained("magic-leap-community/superpoint") if is_vision_available() else None

    @slow
    def test_inference(self):
        model = SuperPointForKeypointDetection.from_pretrained("magic-leap-community/superpoint").to(torch_device)
        preprocessor = self.default_image_processor
        images = prepare_imgs()
        inputs = preprocessor(images=images, return_tensors="pt").to(torch_device)
        with torch.no_grad():
            outputs = model(**inputs)
        expected_number_keypoints_image0 = 568
        expected_number_keypoints_image1 = 830
        expected_max_number_keypoints = max(expected_number_keypoints_image0, expected_number_keypoints_image1)
        expected_keypoints_shape = torch.Size((len(images), expected_max_number_keypoints, 2))
        expected_scores_shape = torch.Size(
            (
                len(images),
                expected_max_number_keypoints,
            )
        )
        expected_descriptors_shape = torch.Size((len(images), expected_max_number_keypoints, 256))
        # Check output shapes
        self.assertEqual(outputs.keypoints.shape, expected_keypoints_shape)
        self.assertEqual(outputs.scores.shape, expected_scores_shape)
        self.assertEqual(outputs.descriptors.shape, expected_descriptors_shape)
        expected_keypoints_image0_values = torch.tensor([[0.75, 0.0188], [0.7719, 0.0188], [0.7641, 0.0333]]).to(
            torch_device
        )
        expected_scores_image0_values = torch.tensor(
            [0.0064, 0.0139, 0.0591, 0.0727, 0.5170, 0.0175, 0.1526, 0.2057, 0.0335]
        ).to(torch_device)
        expected_descriptors_image0_value = torch.tensor(-0.1095).to(torch_device)
        predicted_keypoints_image0_values = outputs.keypoints[0, :3]
        predicted_scores_image0_values = outputs.scores[0, :9]
        predicted_descriptors_image0_value = outputs.descriptors[0, 0, 0]
        # Check output values
        self.assertTrue(
            torch.allclose(
                predicted_keypoints_image0_values,
                expected_keypoints_image0_values,
                atol=1e-4,
            )
        )
        self.assertTrue(torch.allclose(predicted_scores_image0_values, expected_scores_image0_values, atol=1e-4))
        self.assertTrue(
            torch.allclose(
                predicted_descriptors_image0_value,
                expected_descriptors_image0_value,
                atol=1e-4,
            )
        )
        # Check mask values
        self.assertTrue(outputs.mask[0, expected_number_keypoints_image0 - 1].item() == 1)
        self.assertTrue(outputs.mask[0, expected_number_keypoints_image0].item() == 0)
        self.assertTrue(torch.all(outputs.mask[0, : expected_number_keypoints_image0 - 1]))
        self.assertTrue(torch.all(torch.logical_not(outputs.mask[0, expected_number_keypoints_image0:])))
        self.assertTrue(torch.all(outputs.mask[1]))
