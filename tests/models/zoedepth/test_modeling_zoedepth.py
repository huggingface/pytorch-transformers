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
"""Testing suite for the PyTorch ZoeDepth model."""

import unittest

import numpy as np

from transformers import Dinov2Config, ZoeDepthConfig
from transformers.file_utils import is_torch_available, is_vision_available
from transformers.testing_utils import require_torch, require_vision, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import ZoeDepthForDepthEstimation


if is_vision_available():
    from PIL import Image

    from transformers import ZoeDepthImageProcessor


class ZoeDepthModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        num_channels=3,
        image_size=32,
        patch_size=16,
        use_labels=True,
        num_labels=3,
        is_training=True,
        hidden_size=4,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=8,
        out_features=["stage1", "stage2"],
        apply_layernorm=False,
        reshape_hidden_states=False,
        neck_hidden_sizes=[2, 2],
        fusion_hidden_size=6,
        bottleneck_features=6,
        num_out_features=[6, 6, 6, 6],
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.out_features = out_features
        self.apply_layernorm = apply_layernorm
        self.reshape_hidden_states = reshape_hidden_states
        self.use_labels = use_labels
        self.num_labels = num_labels
        self.is_training = is_training
        self.neck_hidden_sizes = neck_hidden_sizes
        self.fusion_hidden_size = fusion_hidden_size
        self.bottleneck_features = bottleneck_features
        self.num_out_features = num_out_features
        # ZoeDepth's sequence length
        self.seq_length = (self.image_size // self.patch_size) ** 2 + 1

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])

        labels = None
        if self.use_labels:
            labels = ids_tensor([self.batch_size, self.image_size, self.image_size], self.num_labels)

        config = self.get_config()

        return config, pixel_values, labels

    def get_config(self):
        return ZoeDepthConfig(
            backbone_config=self.get_backbone_config(),
            backbone=None,
            neck_hidden_sizes=self.neck_hidden_sizes,
            fusion_hidden_size=self.fusion_hidden_size,
            bottleneck_features=self.bottleneck_features,
            num_out_features=self.num_out_features,
        )

    def get_backbone_config(self):
        return Dinov2Config(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            is_training=self.is_training,
            out_features=self.out_features,
            reshape_hidden_states=self.reshape_hidden_states,
        )

    def create_and_check_for_depth_estimation(self, config, pixel_values, labels):
        config.num_labels = self.num_labels
        model = ZoeDepthForDepthEstimation(config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        self.parent.assertEqual(result.predicted_depth.shape, (self.batch_size, self.image_size, self.image_size))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values, labels = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class ZoeDepthModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as ZoeDepth does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (ZoeDepthForDepthEstimation,) if is_torch_available() else ()
    pipeline_model_mapping = {"depth-estimation": ZoeDepthForDepthEstimation} if is_torch_available() else {}

    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = ZoeDepthModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=ZoeDepthConfig, has_text_modality=False, hidden_size=37, common_properties=[]
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="ZoeDepth with AutoBackbone does not have a base model and hence no input_embeddings")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="ZoeDepth with AutoBackbone does not have a base model and hence no input_embeddings")
    def test_model_get_set_embeddings(self):
        pass

    def test_for_depth_estimation(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_depth_estimation(*config_and_inputs)

    @unittest.skip(reason="ZoeDepth with AutoBackbone does not have a base model and hence no input_embeddings")
    def test_model_common_attributes(self):
        pass

    @unittest.skip(reason="ZoeDepth with AutoBackbone does not have a base model")
    def test_save_load_fast_init_from_base(self):
        pass

    @unittest.skip(reason="ZoeDepth with AutoBackbone does not have a base model")
    def test_save_load_fast_init_to_base(self):
        pass

    @unittest.skip(reason="ZoeDepth does not support training yet")
    def test_training(self):
        pass

    @unittest.skip(reason="ZoeDepth does not support training yet")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(reason="ZoeDepth does not support training yet")
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(reason="ZoeDepth does not support training yet")
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        model_name = "Intel/zoedepth-nyu"
        model = ZoeDepthForDepthEstimation.from_pretrained(model_name)
        self.assertIsNotNone(model)


# We will verify our results on an image of cute cats
def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image


@require_torch
@require_vision
@slow
class ZoeDepthModelIntegrationTest(unittest.TestCase):
    expected_slice_post_processing = {
        (False, False): [
            [[1.1348238, 1.1193453, 1.130562], [1.1754476, 1.1613507, 1.1701596], [1.2287744, 1.2101802, 1.2148322]],
            [[2.7170, 2.6550, 2.6839], [2.9827, 2.9438, 2.9587], [3.2340, 3.1817, 3.1602]],
        ],
        (False, True): [
            [[1.0610938, 1.1042216, 1.1429265], [1.1099341, 1.148696, 1.1817775], [1.1656011, 1.1988826, 1.2268101]],
            [[2.5848, 2.7391, 2.8694], [2.7882, 2.9872, 3.1244], [2.9436, 3.1812, 3.3188]],
        ],
        (True, False): [
            [[1.8382794, 1.8380532, 1.8375976], [1.848761, 1.8485023, 1.8479986], [1.8571457, 1.8568444, 1.8562847]],
            [[6.2030, 6.1902, 6.1777], [6.2303, 6.2176, 6.2053], [6.2561, 6.2436, 6.2312]],
        ],
        (True, True): [
            [[1.8306141, 1.8305621, 1.8303483], [1.8410318, 1.8409299, 1.8406585], [1.8492792, 1.8491366, 1.8488203]],
            [[6.2616, 6.2520, 6.2435], [6.2845, 6.2751, 6.2667], [6.3065, 6.2972, 6.2887]],
        ],
    }  # (pad, flip)

    def test_inference_depth_estimation(self):
        image_processor = ZoeDepthImageProcessor.from_pretrained("Intel/zoedepth-nyu")
        model = ZoeDepthForDepthEstimation.from_pretrained("Intel/zoedepth-nyu").to(torch_device)

        image = prepare_img()
        inputs = image_processor(images=image, return_tensors="pt").to(torch_device)

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth

        # verify the predicted depth
        expected_shape = torch.Size((1, 384, 512))
        self.assertEqual(predicted_depth.shape, expected_shape)

        expected_slice = torch.tensor(
            [[1.0020, 1.0219, 1.0389], [1.0349, 1.0816, 1.1000], [1.0576, 1.1094, 1.1249]],
        ).to(torch_device)

        torch.testing.assert_close(outputs.predicted_depth[0, :3, :3], expected_slice, rtol=1e-4, atol=1e-4)

    def test_inference_depth_estimation_multiple_heads(self):
        image_processor = ZoeDepthImageProcessor.from_pretrained("Intel/zoedepth-nyu-kitti")
        model = ZoeDepthForDepthEstimation.from_pretrained("Intel/zoedepth-nyu-kitti").to(torch_device)

        image = prepare_img()
        inputs = image_processor(images=image, return_tensors="pt").to(torch_device)

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth

        # verify the predicted depth
        expected_shape = torch.Size((1, 384, 512))
        self.assertEqual(predicted_depth.shape, expected_shape)

        expected_slice = torch.tensor(
            [[1.1571, 1.1438, 1.1783], [1.2163, 1.2036, 1.2320], [1.2688, 1.2461, 1.2734]],
        ).to(torch_device)

        torch.testing.assert_close(outputs.predicted_depth[0, :3, :3], expected_slice, rtol=1e-4, atol=1e-4)

    def check_target_size(
        self,
        image_processor,
        pad_input,
        images,
        outputs,
        raw_outputs,
        raw_outputs_flipped=None,
    ):
        outputs_large = image_processor.post_process_depth_estimation(
            raw_outputs,
            [img.size[::-1] for img in images],
            outputs_flipped=raw_outputs_flipped,
            target_sizes=[tuple(np.array(img.size[::-1]) * 2) for img in images],
            do_remove_padding=pad_input,
        )

        for img, out, out_l in zip(images, outputs, outputs_large):
            out = out["predicted_depth"]
            out_l = out_l["predicted_depth"]
            out_l_reduced = torch.nn.functional.interpolate(
                out_l.unsqueeze(0).unsqueeze(1), size=img.size[::-1], mode="bicubic", align_corners=False
            )
            self.assertTrue((np.array(out_l.shape)[::-1] == np.array(img.size) * 2).all())
            torch.testing.assert_close(out, out_l_reduced, rtol=2e-2)

    def check_post_processing_test(self, image_processor, images, model, pad_input=True, flip_aug=True):
        inputs = image_processor(images=images, return_tensors="pt", do_pad=pad_input).to(torch_device)

        with torch.no_grad():
            raw_outputs = model(**inputs)
            raw_outputs_flipped = None
            if flip_aug:
                raw_outputs_flipped = model(pixel_values=torch.flip(inputs.pixel_values, dims=[3]))

        outputs = image_processor.post_process_depth_estimation(
            raw_outputs,
            [img.size[::-1] for img in images],
            outputs_flipped=raw_outputs_flipped,
            do_remove_padding=pad_input,
        )

        expected_slices = torch.tensor(self.expected_slice_post_processing[pad_input, flip_aug]).to(torch_device)
        for img, out, expected_slice in zip(images, outputs, expected_slices):
            out = out["predicted_depth"]
            self.assertTrue(img.size == out.shape[::-1])
            torch.testing.assert_close(expected_slice, out[:3, :3], rtol=1e-3)

        self.check_target_size(image_processor, pad_input, images, outputs, raw_outputs, raw_outputs_flipped)

    def test_post_processing_depth_estimation_post_processing_nopad_noflip(self):
        images = [prepare_img(), Image.open("./tests/fixtures/tests_samples/COCO/000000004016.png")]
        image_processor = ZoeDepthImageProcessor.from_pretrained("Intel/zoedepth-nyu-kitti", keep_aspect_ratio=False)
        model = ZoeDepthForDepthEstimation.from_pretrained("Intel/zoedepth-nyu-kitti").to(torch_device)

        self.check_post_processing_test(image_processor, images, model, pad_input=False, flip_aug=False)

    def test_inference_depth_estimation_post_processing_nopad_flip(self):
        images = [prepare_img(), Image.open("./tests/fixtures/tests_samples/COCO/000000004016.png")]
        image_processor = ZoeDepthImageProcessor.from_pretrained("Intel/zoedepth-nyu-kitti", keep_aspect_ratio=False)
        model = ZoeDepthForDepthEstimation.from_pretrained("Intel/zoedepth-nyu-kitti").to(torch_device)

        self.check_post_processing_test(image_processor, images, model, pad_input=False, flip_aug=True)

    def test_inference_depth_estimation_post_processing_pad_noflip(self):
        images = [prepare_img(), Image.open("./tests/fixtures/tests_samples/COCO/000000004016.png")]
        image_processor = ZoeDepthImageProcessor.from_pretrained("Intel/zoedepth-nyu-kitti", keep_aspect_ratio=False)
        model = ZoeDepthForDepthEstimation.from_pretrained("Intel/zoedepth-nyu-kitti").to(torch_device)

        self.check_post_processing_test(image_processor, images, model, pad_input=True, flip_aug=False)

    def test_inference_depth_estimation_post_processing_pad_flip(self):
        images = [prepare_img(), Image.open("./tests/fixtures/tests_samples/COCO/000000004016.png")]
        image_processor = ZoeDepthImageProcessor.from_pretrained("Intel/zoedepth-nyu-kitti", keep_aspect_ratio=False)
        model = ZoeDepthForDepthEstimation.from_pretrained("Intel/zoedepth-nyu-kitti").to(torch_device)

        self.check_post_processing_test(image_processor, images, model, pad_input=True, flip_aug=True)
