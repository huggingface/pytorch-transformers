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
"""Testing suite for the PyTorch Grounding DINO model."""

import collections
import inspect
import math
import re
import unittest

from transformers import (
    GroundingDinoConfig,
    SwinConfig,
    is_torch_available,
    is_vision_available,
)
from transformers.file_utils import cached_property
from transformers.testing_utils import (
    require_timm,
    require_torch,
    require_torch_accelerator,
    require_vision,
    slow,
    torch_device,
)

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, _config_zero_init, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import GroundingDinoForObjectDetection, GroundingDinoModel
    from transformers.pytorch_utils import id_tensor_storage


if is_vision_available():
    from PIL import Image

    from transformers import AutoProcessor


class GroundingDinoModelTester:
    def __init__(
        self,
        parent,
        batch_size=4,
        is_training=True,
        use_labels=True,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=4,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        num_queries=2,
        num_channels=3,
        image_size=98,
        n_targets=8,
        num_labels=3,
        num_feature_levels=4,
        encoder_n_points=2,
        decoder_n_points=6,
        max_text_len=7,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.is_training = is_training
        self.use_labels = use_labels
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.num_queries = num_queries
        self.num_channels = num_channels
        self.image_size = image_size
        self.n_targets = n_targets
        self.num_labels = num_labels
        self.num_feature_levels = num_feature_levels
        self.encoder_n_points = encoder_n_points
        self.decoder_n_points = decoder_n_points
        self.max_text_len = max_text_len

        # we also set the expected seq length for both encoder and decoder
        self.encoder_seq_length_vision = (
            math.ceil(self.image_size / 8) ** 2
            + math.ceil(self.image_size / 16) ** 2
            + math.ceil(self.image_size / 32) ** 2
            + math.ceil(self.image_size / 64) ** 2
        )

        self.encoder_seq_length_text = self.max_text_len

        self.decoder_seq_length = self.num_queries

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        pixel_mask = torch.ones([self.batch_size, self.image_size, self.image_size], device=torch_device)

        input_ids = ids_tensor([self.batch_size, self.max_text_len], self.num_labels)

        labels = None
        if self.use_labels:
            # labels is a list of Dict (each Dict being the labels for a given example in the batch)
            labels = []
            for i in range(self.batch_size):
                target = {}
                target["class_labels"] = torch.randint(
                    high=self.num_labels, size=(self.n_targets,), device=torch_device
                )
                target["boxes"] = torch.rand(self.n_targets, 4, device=torch_device)
                target["masks"] = torch.rand(self.n_targets, self.image_size, self.image_size, device=torch_device)
                labels.append(target)

        config = self.get_config()
        return config, pixel_values, pixel_mask, input_ids, labels

    def get_config(self):
        swin_config = SwinConfig(
            window_size=7,
            embed_dim=8,
            depths=[1, 1, 1, 1],
            num_heads=[1, 1, 1, 1],
            image_size=self.image_size,
            out_features=["stage2", "stage3", "stage4"],
            out_indices=[2, 3, 4],
        )
        text_backbone = {
            "hidden_size": 8,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "intermediate_size": 8,
            "max_position_embeddings": 8,
            "model_type": "bert",
        }
        return GroundingDinoConfig(
            d_model=self.hidden_size,
            encoder_layers=self.num_hidden_layers,
            decoder_layers=self.num_hidden_layers,
            encoder_attention_heads=self.num_attention_heads,
            decoder_attention_heads=self.num_attention_heads,
            encoder_ffn_dim=self.intermediate_size,
            decoder_ffn_dim=self.intermediate_size,
            dropout=self.hidden_dropout_prob,
            attention_dropout=self.attention_probs_dropout_prob,
            num_queries=self.num_queries,
            num_labels=self.num_labels,
            num_feature_levels=self.num_feature_levels,
            encoder_n_points=self.encoder_n_points,
            decoder_n_points=self.decoder_n_points,
            use_timm_backbone=False,
            backbone_config=swin_config,
            max_text_len=self.max_text_len,
            text_config=text_backbone,
        )

    def prepare_config_and_inputs_for_common(self):
        config, pixel_values, pixel_mask, input_ids, labels = self.prepare_config_and_inputs()
        inputs_dict = {"pixel_values": pixel_values, "pixel_mask": pixel_mask, "input_ids": input_ids}
        return config, inputs_dict

    def create_and_check_model(self, config, pixel_values, pixel_mask, input_ids, labels):
        model = GroundingDinoModel(config=config)
        model.to(torch_device)
        model.eval()

        result = model(pixel_values=pixel_values, pixel_mask=pixel_mask, input_ids=input_ids)

        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.num_queries, self.hidden_size))

    def create_and_check_object_detection_head_model(self, config, pixel_values, pixel_mask, input_ids, labels):
        model = GroundingDinoForObjectDetection(config=config)
        model.to(torch_device)
        model.eval()

        result = model(pixel_values=pixel_values, pixel_mask=pixel_mask, input_ids=input_ids)

        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_queries, config.max_text_len))
        self.parent.assertEqual(result.pred_boxes.shape, (self.batch_size, self.num_queries, 4))

        result = model(pixel_values=pixel_values, pixel_mask=pixel_mask, input_ids=input_ids, labels=labels)

        self.parent.assertEqual(result.loss.shape, ())
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_queries, config.max_text_len))
        self.parent.assertEqual(result.pred_boxes.shape, (self.batch_size, self.num_queries, 4))


@require_torch
class GroundingDinoModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (GroundingDinoModel, GroundingDinoForObjectDetection) if is_torch_available() else ()
    is_encoder_decoder = True
    test_torchscript = False
    test_pruning = False
    test_head_masking = False
    test_missing_keys = False
    pipeline_model_mapping = (
        {"image-feature-extraction": GroundingDinoModel, "zero-shot-object-detection": GroundingDinoForObjectDetection}
        if is_torch_available()
        else {}
    )

    # special case for head models
    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)

        if return_labels:
            if model_class.__name__ == "GroundingDinoForObjectDetection":
                labels = []
                for i in range(self.model_tester.batch_size):
                    target = {}
                    target["class_labels"] = torch.ones(
                        size=(self.model_tester.n_targets,), device=torch_device, dtype=torch.long
                    )
                    target["boxes"] = torch.ones(
                        self.model_tester.n_targets, 4, device=torch_device, dtype=torch.float
                    )
                    target["masks"] = torch.ones(
                        self.model_tester.n_targets,
                        self.model_tester.image_size,
                        self.model_tester.image_size,
                        device=torch_device,
                        dtype=torch.float,
                    )
                    labels.append(target)
                inputs_dict["labels"] = labels

        return inputs_dict

    def setUp(self):
        self.model_tester = GroundingDinoModelTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=GroundingDinoConfig,
            has_text_modality=False,
            common_properties=["d_model", "encoder_attention_heads", "decoder_attention_heads"],
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_object_detection_head_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_object_detection_head_model(*config_and_inputs)

    @unittest.skip(reason="Grounding DINO does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="Grounding DINO does not have a get_input_embeddings method")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="Grounding DINO does not use token embeddings")
    def test_resize_tokens_embeddings(self):
        pass

    @unittest.skip(reason="Feed forward chunking is not implemented")
    def test_feed_forward_chunking(self):
        pass

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
            attentions = outputs.encoder_attentions[-1]
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.encoder_attentions[-1]
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            self.assertListEqual(
                list(attentions[0].shape[-3:]),
                [
                    self.model_tester.num_attention_heads,
                    self.model_tester.num_feature_levels,
                    self.model_tester.encoder_n_points,
                ],
            )
            out_len = len(outputs)

            correct_outlen = 10

            # loss is at first position
            if "labels" in inputs_dict:
                correct_outlen += 1  # loss is added to beginning
            # Object Detection model returns pred_logits and pred_boxes
            if model_class.__name__ == "GroundingDinoForObjectDetection":
                correct_outlen += 2

            self.assertEqual(out_len, correct_outlen)

            # decoder attentions
            decoder_attentions = outputs.decoder_attentions[0]
            self.assertIsInstance(decoder_attentions, (list, tuple))
            self.assertEqual(len(decoder_attentions), self.model_tester.num_hidden_layers)
            self.assertListEqual(
                list(decoder_attentions[0].shape[-3:]),
                [self.model_tester.num_attention_heads, self.model_tester.num_queries, self.model_tester.num_queries],
            )

            # cross attentions
            cross_attentions = outputs.decoder_attentions[-1]
            self.assertIsInstance(cross_attentions, (list, tuple))
            self.assertEqual(len(cross_attentions), self.model_tester.num_hidden_layers)
            self.assertListEqual(
                list(cross_attentions[0].shape[-3:]),
                [
                    self.model_tester.num_attention_heads,
                    self.model_tester.num_feature_levels,
                    self.model_tester.decoder_n_points,
                ],
            )

            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            self.assertEqual(out_len + 3, len(outputs))

            self_attentions = outputs.encoder_attentions[-1]

            self.assertEqual(len(self_attentions), self.model_tester.num_hidden_layers)
            self.assertListEqual(
                list(self_attentions[0].shape[-3:]),
                [
                    self.model_tester.num_attention_heads,
                    self.model_tester.num_feature_levels,
                    self.model_tester.encoder_n_points,
                ],
            )

    # overwrite since hidden_states are called encoder_text_hidden_states
    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.encoder_vision_hidden_states

            expected_num_layers = getattr(
                self.model_tester, "expected_num_hidden_layers", self.model_tester.num_hidden_layers + 1
            )
            self.assertEqual(len(hidden_states), expected_num_layers)

            seq_len = self.model_tester.encoder_seq_length_vision

            self.assertListEqual(
                list(hidden_states[0].shape[-2:]),
                [seq_len, self.model_tester.hidden_size],
            )

            hidden_states = outputs.encoder_text_hidden_states

            self.assertEqual(len(hidden_states), expected_num_layers)

            seq_len = self.model_tester.encoder_seq_length_text

            self.assertListEqual(
                list(hidden_states[0].shape[-2:]),
                [seq_len, self.model_tester.hidden_size],
            )

            hidden_states = outputs.decoder_hidden_states

            self.assertIsInstance(hidden_states, (list, tuple))
            self.assertEqual(len(hidden_states), expected_num_layers)
            seq_len = getattr(self.model_tester, "seq_length", None)
            decoder_seq_length = getattr(self.model_tester, "decoder_seq_length", seq_len)

            self.assertListEqual(
                list(hidden_states[0].shape[-2:]),
                [decoder_seq_length, self.model_tester.hidden_size],
            )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True

            check_hidden_states_output(inputs_dict, config, model_class)

    # removed retain_grad and grad on decoder_hidden_states, as queries don't require grad
    def test_retain_grad_hidden_states_attentions(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.output_hidden_states = True
        config.output_attentions = True

        # no need to test all models as different heads yield the same functionality
        model_class = self.all_model_classes[0]
        model = model_class(config)
        model.to(torch_device)

        inputs = self._prepare_for_class(inputs_dict, model_class)

        outputs = model(**inputs)

        output = outputs[0]

        encoder_hidden_states = outputs.encoder_vision_hidden_states[0]
        encoder_attentions = outputs.encoder_attentions[0][0]
        encoder_hidden_states.retain_grad()
        encoder_attentions.retain_grad()

        cross_attentions = outputs.decoder_attentions[-1][0]
        cross_attentions.retain_grad()

        output.flatten()[0].backward(retain_graph=True)

        self.assertIsNotNone(encoder_hidden_states.grad)
        self.assertIsNotNone(encoder_attentions.grad)
        self.assertIsNotNone(cross_attentions.grad)

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["pixel_values", "input_ids"]
            self.assertListEqual(arg_names[: len(expected_arg_names)], expected_arg_names)

    def test_different_timm_backbone(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        # let's pick a random timm backbone
        config.backbone = "tf_mobilenetv3_small_075"
        config.use_timm_backbone = True
        config.backbone_config = None
        config.backbone_kwargs = {"in_chans": 3, "out_indices": (2, 3, 4)}

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            if model_class.__name__ == "GroundingDinoForObjectDetection":
                expected_shape = (
                    self.model_tester.batch_size,
                    self.model_tester.num_queries,
                    config.max_text_len,
                )
                self.assertEqual(outputs.logits.shape, expected_shape)

            self.assertTrue(outputs)

    @require_timm
    def test_hf_backbone(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        # Load a pretrained HF checkpoint as backbone
        config.backbone = "microsoft/resnet-18"
        config.backbone_config = None
        config.use_timm_backbone = False
        config.use_pretrained_backbone = True
        config.backbone_kwargs = {"out_indices": [2, 3, 4]}

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            if model_class.__name__ == "GroundingDinoForObjectDetection":
                expected_shape = (
                    self.model_tester.batch_size,
                    self.model_tester.num_queries,
                    config.max_text_len,
                )
                self.assertEqual(outputs.logits.shape, expected_shape)

            self.assertTrue(outputs)

    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if (
                        "level_embed" in name
                        or "sampling_offsets.bias" in name
                        or "text_param" in name
                        or "vision_param" in name
                        or "value_proj" in name
                        or "output_proj" in name
                        or "reference_points" in name
                    ):
                        continue
                    self.assertIn(
                        ((param.data.mean() * 1e9).round() / 1e9).item(),
                        [0.0, 1.0],
                        msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                    )

    # Copied from tests.models.deformable_detr.test_modeling_deformable_detr.DeformableDetrModelTest.test_two_stage_training with DeformableDetr->GroundingDino
    def test_two_stage_training(self):
        model_class = GroundingDinoForObjectDetection
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True
        config.two_stage = True
        config.auxiliary_loss = True
        config.with_box_refine = True

        model = model_class(config)
        model.to(torch_device)
        model.train()
        inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
        loss = model(**inputs).loss
        loss.backward()

    def test_tied_weights_keys(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        config.tie_word_embeddings = True
        for model_class in self.all_model_classes:
            model_tied = model_class(config)

            ptrs = collections.defaultdict(list)
            for name, tensor in model_tied.state_dict().items():
                ptrs[id_tensor_storage(tensor)].append(name)

            # These are all the pointers of shared tensors.
            tied_params = [names for _, names in ptrs.items() if len(names) > 1]

            tied_weight_keys = model_tied._tied_weights_keys if model_tied._tied_weights_keys is not None else []
            # Detect we get a hit for each key
            for key in tied_weight_keys:
                if not any(re.search(key, p) for group in tied_params for p in group):
                    raise ValueError(f"{key} is not a tied weight key for {model_class}.")

            # Removed tied weights found from tied params -> there should only be one left after
            for key in tied_weight_keys:
                for i in range(len(tied_params)):
                    tied_params[i] = [p for p in tied_params[i] if re.search(key, p) is None]

            # GroundingDino when sharing weights also uses the shared ones in GroundingDinoDecoder
            # Therefore, differently from DeformableDetr, we expect the group lens to be 2
            # one for self.bbox_embed in GroundingDinoForObejectDetection and another one
            # in the decoder
            tied_params = [group for group in tied_params if len(group) > 2]
            self.assertListEqual(
                tied_params,
                [],
                f"Missing `_tied_weights_keys` for {model_class}: add all of {tied_params} except one.",
            )


# We will verify our results on an image of cute cats
def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image


def prepare_text():
    text = "a cat."
    return text


@require_timm
@require_vision
@slow
class GroundingDinoModelIntegrationTests(unittest.TestCase):
    @cached_property
    def default_processor(self):
        return AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny") if is_vision_available() else None

    def test_inference_object_detection_head(self):
        model = GroundingDinoForObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny").to(torch_device)

        processor = self.default_processor
        image = prepare_img()
        text = prepare_text()
        encoding = processor(images=image, text=text, return_tensors="pt").to(torch_device)

        with torch.no_grad():
            outputs = model(**encoding)

        expected_shape_logits = torch.Size((1, model.config.num_queries, model.config.d_model))
        self.assertEqual(outputs.logits.shape, expected_shape_logits)

        expected_boxes = torch.tensor(
            [[0.7674, 0.4136, 0.4572], [0.2566, 0.5463, 0.4760], [0.2585, 0.5442, 0.4641]]
        ).to(torch_device)
        expected_logits = torch.tensor(
            [[-4.8913, -0.1900, -0.2161], [-4.9653, -0.3719, -0.3950], [-5.9599, -3.3765, -3.3104]]
        ).to(torch_device)

        torch.testing.assert_close(outputs.logits[0, :3, :3], expected_logits, rtol=1e-3, atol=1e-3)

        expected_shape_boxes = torch.Size((1, model.config.num_queries, 4))
        self.assertEqual(outputs.pred_boxes.shape, expected_shape_boxes)
        torch.testing.assert_close(outputs.pred_boxes[0, :3, :3], expected_boxes, rtol=1e-4, atol=1e-4)

        # verify postprocessing
        results = processor.image_processor.post_process_object_detection(
            outputs, threshold=0.35, target_sizes=[image.size[::-1]]
        )[0]
        expected_scores = torch.tensor([0.4526, 0.4082]).to(torch_device)
        expected_slice_boxes = torch.tensor([344.8143, 23.1796, 637.4004, 373.8295]).to(torch_device)

        self.assertEqual(len(results["scores"]), 2)
        torch.testing.assert_close(results["scores"], expected_scores, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(results["boxes"][0, :], expected_slice_boxes, rtol=1e-2, atol=1e-2)

        # verify grounded postprocessing
        expected_labels = ["a cat", "a cat"]
        results = processor.post_process_grounded_object_detection(
            outputs=outputs,
            input_ids=encoding.input_ids,
            box_threshold=0.35,
            text_threshold=0.3,
            target_sizes=[image.size[::-1]],
        )[0]

        torch.testing.assert_close(results["scores"], expected_scores, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(results["boxes"][0, :], expected_slice_boxes, rtol=1e-2, atol=1e-2)
        self.assertListEqual(results["labels"], expected_labels)

    @require_torch_accelerator
    def test_inference_object_detection_head_equivalence_cpu_gpu(self):
        processor = self.default_processor
        image = prepare_img()
        text = prepare_text()
        encoding = processor(images=image, text=text, return_tensors="pt")

        # 1. run model on CPU
        model = GroundingDinoForObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny")

        with torch.no_grad():
            cpu_outputs = model(**encoding)

        # 2. run model on GPU
        model.to(torch_device)
        encoding = encoding.to(torch_device)
        with torch.no_grad():
            gpu_outputs = model(**encoding)

        # 3. assert equivalence
        for key in cpu_outputs.keys():
            torch.testing.assert_close(cpu_outputs[key], gpu_outputs[key].cpu(), rtol=1e-3, atol=1e-3)

        expected_logits = torch.tensor(
            [[-4.8915, -0.1900, -0.2161], [-4.9658, -0.3716, -0.3948], [-5.9596, -3.3763, -3.3103]]
        )
        torch.testing.assert_close(cpu_outputs.logits[0, :3, :3], expected_logits, rtol=1e-3, atol=1e-3)

        # assert postprocessing
        results_cpu = processor.image_processor.post_process_object_detection(
            cpu_outputs, threshold=0.35, target_sizes=[image.size[::-1]]
        )[0]

        result_gpu = processor.image_processor.post_process_object_detection(
            gpu_outputs, threshold=0.35, target_sizes=[image.size[::-1]]
        )[0]

        torch.testing.assert_close(results_cpu["scores"], result_gpu["scores"].cpu(), rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(results_cpu["boxes"], result_gpu["boxes"].cpu(), rtol=1e-3, atol=1e-3)

    def test_cross_attention_mask(self):
        model = GroundingDinoForObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny").to(torch_device)

        processor = self.default_processor
        image = prepare_img()
        text1 = "a cat."
        text2 = "a remote control."
        text_batched = [text1, text2]

        encoding1 = processor(images=image, text=text1, return_tensors="pt").to(torch_device)
        encoding2 = processor(images=image, text=text2, return_tensors="pt").to(torch_device)
        # If we batch the text and cross attention masking is working the batched result should be equal to
        # The singe text result
        encoding_batched = processor(
            images=[image] * len(text_batched), text=text_batched, padding="longest", return_tensors="pt"
        ).to(torch_device)

        with torch.no_grad():
            outputs1 = model(**encoding1)
            outputs2 = model(**encoding2)
            outputs_batched = model(**encoding_batched)

        torch.testing.assert_close(outputs1.logits, outputs_batched.logits[:1], rtol=1e-3, atol=1e-3)
        # For some reason 12 elements are > 1e-3, but the rest are fine
        torch.testing.assert_close(outputs2.logits, outputs_batched.logits[1:], rtol=1.8e-3, atol=1.8e-3)
