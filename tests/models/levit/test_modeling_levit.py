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
"""Testing suite for the PyTorch LeViT model."""

import unittest
import warnings
from math import ceil, floor

from transformers import LevitConfig
from transformers.file_utils import cached_property, is_torch_available, is_vision_available
from transformers.testing_utils import require_torch, require_vision, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        LevitForImageClassification,
        LevitForImageClassificationWithTeacher,
        LevitModel,
    )
    from transformers.models.auto.modeling_auto import (
        MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES,
        MODEL_MAPPING_NAMES,
    )


if is_vision_available():
    from PIL import Image

    from transformers import LevitImageProcessor


class LevitConfigTester(ConfigTester):
    def create_and_test_config_common_properties(self):
        config = self.config_class(**self.inputs_dict)
        self.parent.assertTrue(hasattr(config, "hidden_sizes"))
        self.parent.assertTrue(hasattr(config, "num_attention_heads"))


class LevitModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        image_size=64,
        num_channels=3,
        kernel_size=3,
        stride=2,
        padding=1,
        patch_size=16,
        hidden_sizes=[16, 32, 48],
        num_attention_heads=[1, 2, 3],
        depths=[2, 3, 4],
        key_dim=[8, 8, 8],
        drop_path_rate=0,
        mlp_ratio=[2, 2, 2],
        attention_ratio=[2, 2, 2],
        initializer_range=0.02,
        is_training=True,
        use_labels=True,
        num_labels=2,  # Check
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.hidden_sizes = hidden_sizes
        self.num_attention_heads = num_attention_heads
        self.depths = depths
        self.key_dim = key_dim
        self.drop_path_rate = drop_path_rate
        self.patch_size = patch_size
        self.attention_ratio = attention_ratio
        self.mlp_ratio = mlp_ratio
        self.initializer_range = initializer_range
        self.down_ops = [
            ["Subsample", key_dim[0], hidden_sizes[0] // key_dim[0], 4, 2, 2],
            ["Subsample", key_dim[0], hidden_sizes[1] // key_dim[0], 4, 2, 2],
        ]
        self.is_training = is_training
        self.use_labels = use_labels
        self.num_labels = num_labels
        self.initializer_range = initializer_range

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])

        labels = None
        if self.use_labels:
            labels = ids_tensor([self.batch_size], self.num_labels)

        config = self.get_config()
        return config, pixel_values, labels

    def get_config(self):
        return LevitConfig(
            image_size=self.image_size,
            num_channels=self.num_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            patch_size=self.patch_size,
            hidden_sizes=self.hidden_sizes,
            num_attention_heads=self.num_attention_heads,
            depths=self.depths,
            key_dim=self.key_dim,
            drop_path_rate=self.drop_path_rate,
            mlp_ratio=self.mlp_ratio,
            attention_ratio=self.attention_ratio,
            initializer_range=self.initializer_range,
            down_ops=self.down_ops,
        )

    def create_and_check_model(self, config, pixel_values, labels):
        model = LevitModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        image_size = (self.image_size, self.image_size)
        height, width = image_size[0], image_size[1]
        for _ in range(4):
            height = floor(((height + 2 * self.padding - self.kernel_size) / self.stride) + 1)
            width = floor(((width + 2 * self.padding - self.kernel_size) / self.stride) + 1)
        self.parent.assertEqual(
            result.last_hidden_state.shape,
            (self.batch_size, ceil(height / 4) * ceil(width / 4), self.hidden_sizes[-1]),
        )

    def create_and_check_for_image_classification(self, config, pixel_values, labels):
        config.num_labels = self.num_labels
        model = LevitForImageClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values, labels=labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values, labels = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class LevitModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as Levit does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (
        (LevitModel, LevitForImageClassification, LevitForImageClassificationWithTeacher)
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "image-feature-extraction": LevitModel,
            "image-classification": (LevitForImageClassification, LevitForImageClassificationWithTeacher),
        }
        if is_torch_available()
        else {}
    )

    test_pruning = False
    test_torchscript = False
    test_resize_embeddings = False
    test_head_masking = False
    has_attentions = False

    def setUp(self):
        self.model_tester = LevitModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=LevitConfig, has_text_modality=False, common_properties=["image_size", "num_channels"]
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="Levit does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="Levit does not support input and output embeddings")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="Levit does not output attentions")
    def test_attention_outputs(self):
        pass

    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.hidden_states

            expected_num_layers = len(self.model_tester.depths) + 1
            self.assertEqual(len(hidden_states), expected_num_layers)

            image_size = (self.model_tester.image_size, self.model_tester.image_size)
            height, width = image_size[0], image_size[1]
            for _ in range(4):
                height = floor(
                    (
                        (height + 2 * self.model_tester.padding - self.model_tester.kernel_size)
                        / self.model_tester.stride
                    )
                    + 1
                )
                width = floor(
                    (
                        (width + 2 * self.model_tester.padding - self.model_tester.kernel_size)
                        / self.model_tester.stride
                    )
                    + 1
                )
            # verify the first hidden states (first block)
            self.assertListEqual(
                list(hidden_states[0].shape[-2:]),
                [
                    height * width,
                    self.model_tester.hidden_sizes[0],
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

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)

        if return_labels:
            if model_class.__name__ == "LevitForImageClassificationWithTeacher":
                del inputs_dict["labels"]

        return inputs_dict

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_image_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_image_classification(*config_and_inputs)

    # special case for LevitForImageClassificationWithTeacher model
    def test_training(self):
        if not self.model_tester.is_training:
            self.skipTest(reason="model_tester.is_training is set to False")

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True

        for model_class in self.all_model_classes:
            # LevitForImageClassificationWithTeacher supports inference-only
            if (
                model_class.__name__ in MODEL_MAPPING_NAMES.values()
                or model_class.__name__ == "LevitForImageClassificationWithTeacher"
            ):
                continue
            model = model_class(config)
            model.to(torch_device)
            model.train()
            inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            loss = model(**inputs).loss
            loss.backward()

    def test_training_gradient_checkpointing(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        if not self.model_tester.is_training:
            self.skipTest(reason="model_tester.is_training is set to False")

        config.use_cache = False
        config.return_dict = True

        for model_class in self.all_model_classes:
            if model_class.__name__ in MODEL_MAPPING_NAMES.values() or not model_class.supports_gradient_checkpointing:
                continue
            # LevitForImageClassificationWithTeacher supports inference-only
            if model_class.__name__ == "LevitForImageClassificationWithTeacher":
                continue
            model = model_class(config)
            model.gradient_checkpointing_enable()
            model.to(torch_device)
            model.train()
            inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            loss = model(**inputs).loss
            loss.backward()

    def test_problem_types(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        problem_types = [
            {"title": "multi_label_classification", "num_labels": 2, "dtype": torch.float},
            {"title": "single_label_classification", "num_labels": 1, "dtype": torch.long},
            {"title": "regression", "num_labels": 1, "dtype": torch.float},
        ]

        for model_class in self.all_model_classes:
            if (
                model_class.__name__
                not in [
                    *MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES.values(),
                ]
                or model_class.__name__ == "LevitForImageClassificationWithTeacher"
            ):
                continue

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

    @slow
    def test_model_from_pretrained(self):
        model_name = "facebook/levit-128S"
        model = LevitModel.from_pretrained(model_name)
        self.assertIsNotNone(model)


# We will verify our results on an image of cute cats
def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image


@require_torch
@require_vision
class LevitModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_image_processor(self):
        return LevitImageProcessor.from_pretrained("facebook/levit-128S")

    @slow
    def test_inference_image_classification_head(self):
        model = LevitForImageClassificationWithTeacher.from_pretrained("facebook/levit-128S").to(torch_device)

        image_processor = self.default_image_processor
        image = prepare_img()
        inputs = image_processor(images=image, return_tensors="pt").to(torch_device)

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # verify the logits
        expected_shape = torch.Size((1, 1000))
        self.assertEqual(outputs.logits.shape, expected_shape)

        expected_slice = torch.tensor([1.0448, -0.3745, -1.8317]).to(torch_device)

        torch.testing.assert_close(outputs.logits[0, :3], expected_slice, rtol=1e-4, atol=1e-4)
