# coding=utf-8
# Copyright 2023 The Intel Labs Team Authors, The Microsoft Research Team Authors and HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch BridgeTower model."""

import tempfile
import unittest

import numpy as np

from transformers import (
    BridgeTowerConfig,
    BridgeTowerTextConfig,
    BridgeTowerVisionConfig,
    is_torch_available,
    is_vision_available,
)
from transformers.testing_utils import require_torch, require_vision, slow, torch_device
from transformers.utils import cached_property

from ...test_configuration_common import ConfigTester
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

    from transformers import (
        BridgeTowerForContrastiveLearning,
        BridgeTowerForImageAndTextRetrieval,
        BridgeTowerForMaskedLM,
        BridgeTowerModel,
    )

if is_vision_available():
    from PIL import Image

    from transformers import BridgeTowerProcessor


class BridgeTowerTextModelTester:
    def __init__(
        self,
        parent,
        hidden_act="gelu",
        hidden_size=64,
        initializer_factor=1,
        layer_norm_eps=1e-05,
        num_attention_heads=4,
        num_hidden_layers=2,
        intermediate_size=128,
        tie_word_embeddings=False,
        output_hidden_states=False,
    ):
        self.parent = parent
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.initializer_factor = initializer_factor
        self.layer_norm_eps = layer_norm_eps
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.tie_word_embeddings = tie_word_embeddings
        self.vocab_size = 99
        self.seq_length = 4
        self.batch_size = 1
        self.is_training = False
        self.output_hidden_states = output_hidden_states

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        attention_mask = random_attention_mask([self.batch_size, self.seq_length])

        config = self.get_config()

        return config, input_ids, attention_mask

    def get_config(self):
        return BridgeTowerTextConfig(
            hidden_act=self.hidden_act,
            hidden_size=self.hidden_size,
            initializer_factor=self.initializer_factor,
            layer_norm_eps=self.layer_norm_eps,
            num_attention_heads=self.num_attention_heads,
            num_hidden_layers=self.num_hidden_layers,
            intermediate_size=self.intermediate_size,
            tie_word_embeddings=self.tie_word_embeddings,
            output_hidden_states=self.output_hidden_states,
            vocab_size=self.vocab_size,
        )


class BridgeTowerImageModelTester:
    def __init__(
        self,
        parent,
        hidden_size=64,
        initializer_factor=1,
        layer_norm_eps=1e-05,
        num_hidden_layers=2,
        init_layernorm_from_vision_encoder=False,
        output_hidden_states=False,
        image_size=64,
    ):
        self.parent = parent
        self.hidden_size = hidden_size
        self.initializer_factor = initializer_factor
        self.layer_norm_eps = layer_norm_eps
        self.num_hidden_layers = num_hidden_layers
        self.init_layernorm_from_vision_encoder = init_layernorm_from_vision_encoder
        self.num_channels = 3
        self.num_image_features = 17
        self.batch_size = 1
        self.image_size = image_size
        self.is_training = False
        self.output_hidden_states = output_hidden_states

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        pixel_mask = random_attention_mask([self.batch_size, self.image_size, self.image_size])
        config = self.get_config()

        return config, pixel_values, pixel_mask

    def get_config(self):
        return BridgeTowerVisionConfig(
            hidden_size=self.hidden_size,
            initializer_factor=self.initializer_factor,
            layer_norm_eps=self.layer_norm_eps,
            num_hidden_layers=self.num_hidden_layers,
            init_layernorm_from_vision_encoder=self.init_layernorm_from_vision_encoder,
            num_channels=self.num_channels,
            num_image_features=self.num_image_features,
            batch_size=self.batch_size,
            image_size=self.image_size,
            is_training=self.is_training,
            output_hidden_states=self.output_hidden_states,
        )


class BridgeTowerModelTester:
    def __init__(
        self,
        parent,
        text_kwargs=None,
        vision_kwargs=None,
        share_cross_modal_transformer_layers=True,
        share_link_tower_layers=False,
        link_tower_type="add",
        init_layernorm_from_vision_encoder=False,
        contrastive_hidden_size=512,
        logit_scale_init_value=2.6592,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
    ):
        if text_kwargs is None:
            text_kwargs = {}
        if vision_kwargs is None:
            vision_kwargs = {}

        self.parent = parent
        self.text_model_tester = BridgeTowerTextModelTester(parent, **text_kwargs)
        self.vision_model_tester = BridgeTowerImageModelTester(parent, **vision_kwargs)

        self.share_cross_modal_transformer_layers = share_cross_modal_transformer_layers
        self.share_link_tower_layers = share_link_tower_layers
        self.link_tower_type = link_tower_type
        self.init_layernorm_from_vision_encoder = init_layernorm_from_vision_encoder
        self.contrastive_hidden_size = contrastive_hidden_size
        self.logit_scale_init_value = logit_scale_init_value

        self.batch_size = 1
        self.expected_num_hidden_layers = 8
        self.is_training = False

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size

    def prepare_config_and_inputs(self):
        text_config, input_ids, attention_mask = self.text_model_tester.prepare_config_and_inputs()
        vision_config, pixel_values, pixel_mask = self.vision_model_tester.prepare_config_and_inputs()

        config = self.get_config()

        return (config, input_ids, attention_mask, pixel_values, pixel_mask)

    def get_config(self):
        return BridgeTowerConfig.from_text_vision_configs(
            text_config=self.text_model_tester.get_config(),
            vision_config=self.vision_model_tester.get_config(),
            share_cross_modal_transformer_layers=self.share_cross_modal_transformer_layers,
            share_link_tower_layers=self.share_link_tower_layers,
            link_tower_type=self.link_tower_type,
            init_layernorm_from_vision_encoder=self.init_layernorm_from_vision_encoder,
            contrastive_hidden_size=self.contrastive_hidden_size,
            logit_scale_init_value=self.logit_scale_init_value,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
        )

    def create_and_check_model(
        self,
        config,
        input_ids,
        attention_mask,
        pixel_values,
        pixel_mask,
    ):
        model = BridgeTowerModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, pixel_values=pixel_values, pixel_mask=pixel_mask)
        result = model(input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
        self.parent.assertEqual(
            result["text_features"].shape,
            (self.batch_size, self.text_model_tester.seq_length, self.text_model_tester.hidden_size),
        )
        self.parent.assertEqual(
            result["image_features"].shape,
            (self.batch_size, self.vision_model_tester.num_image_features, self.vision_model_tester.hidden_size),
        )
        self.parent.assertEqual(
            result["pooler_output"].shape,
            (self.batch_size, self.text_model_tester.hidden_size + self.vision_model_tester.hidden_size),
        )

    def create_and_check_for_image_and_text_retrieval(
        self,
        config,
        input_ids,
        attention_mask,
        pixel_values,
        pixel_mask,
    ):
        bridgetower_itm_output_last_dimension = 2

        model = BridgeTowerForImageAndTextRetrieval(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, pixel_values=pixel_values, pixel_mask=pixel_mask)
        result = model(input_ids, attention_mask=attention_mask, pixel_values=pixel_values)

        self.parent.assertEqual(result.logits.shape, (self.batch_size, bridgetower_itm_output_last_dimension))

    def create_and_check_for_masked_language_modeling(
        self,
        config,
        input_ids,
        attention_mask,
        pixel_values,
        pixel_mask,
    ):
        model = BridgeTowerForMaskedLM(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, pixel_values=pixel_values, pixel_mask=pixel_mask)
        result = model(input_ids, attention_mask=attention_mask, pixel_values=pixel_values)

        self.parent.assertEqual(
            result.logits.shape,
            (self.batch_size, self.text_model_tester.seq_length, self.text_model_tester.vocab_size),
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (config, input_ids, attention_mask, pixel_values, pixel_mask) = config_and_inputs
        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "pixel_mask": pixel_mask,
        }
        return config, inputs_dict


@require_torch
class BridgeTowerModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            BridgeTowerModel,
            BridgeTowerForImageAndTextRetrieval,
            BridgeTowerForMaskedLM,
            BridgeTowerForContrastiveLearning,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = {"feature-extraction": BridgeTowerModel} if is_torch_available() else {}

    is_training = False
    test_headmasking = False
    test_pruning = False
    test_torchscript = False
    test_resize_embeddings = False
    has_attentions = False

    @unittest.skip(reason="Does not work on the tiny model as we keep hitting edge cases.")
    def test_cpu_offload(self):
        pass

    @unittest.skip(reason="Does not work on the tiny model as we keep hitting edge cases.")
    def test_disk_offload(self):
        pass

    @unittest.skip(reason="Does not work on the tiny model as we keep hitting edge cases.")
    def test_model_parallelism(self):
        pass

    # function to extract meaningful tensor from output per different model_class
    def extract_output(self, outputs, model_class):
        return outputs["pooler_output"] if model_class == "BridgeTowerModel" else outputs["logits"]

    def setUp(self):
        self.model_tester = BridgeTowerModelTester(self)
        self.config_tester = ConfigTester(self, config_class=BridgeTowerConfig, hidden_size=37, vocab_size=99)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_image_and_text_retrieval(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_image_and_text_retrieval(*config_and_inputs)

    def test_for_masked_language_modeling(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_masked_language_modeling(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        model_name = "BridgeTower/bridgetower-base"
        model = BridgeTowerModel.from_pretrained(model_name)
        self.assertIsNotNone(model)

    @slow
    def test_save_load_fast_init_from_base(self):
        # Override as it is a slow test on this model
        super().test_save_load_fast_init_from_base()

    # Override as extracting meaningful tensor from output is different for BridgeTower
    def test_save_load(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**input_dict)

            out_2 = self.extract_output(outputs, model_class.__name__)
            out_2 = out_2.cpu().numpy()
            out_2[np.isnan(out_2)] = 0

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model = model_class.from_pretrained(tmpdirname)
                model.to(torch_device)
                with torch.no_grad():
                    after_outputs = model(**input_dict)

                # Make sure we don't have nans
                out_1 = self.extract_output(after_outputs, model_class.__name__)
                out_1 = out_1.cpu().numpy()
                out_1[np.isnan(out_1)] = 0
                max_diff = np.amax(np.abs(out_1 - out_2))
                self.assertLessEqual(max_diff, 1e-5)

    # Override this as `hidden states output` is different for BridgeTower
    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states_text, hidden_states_vision, hidden_states_cross = (
                outputs.encoder_hidden_states if config.is_encoder_decoder else outputs.hidden_states
            )

            expected_num_layers = self.model_tester.expected_num_hidden_layers
            self.assertEqual(
                sum((len(hidden_states_text), len(hidden_states_vision), len(hidden_states_cross))),
                expected_num_layers,
            )

            seq_length = self.model_tester.text_model_tester.seq_length
            num_image_features = self.model_tester.vision_model_tester.num_image_features

            self.assertListEqual(
                list(hidden_states_text[0].shape[-2:]),
                [seq_length, self.model_tester.text_model_tester.hidden_size],
            )
            self.assertListEqual(
                list(hidden_states_vision[0].shape),
                [num_image_features, 1, self.model_tester.vision_model_tester.hidden_size],
            )
            self.assertListEqual(
                list(hidden_states_cross[0][0].shape[-2:]),
                [seq_length, self.model_tester.text_model_tester.hidden_size],
            )
            self.assertListEqual(
                list(hidden_states_cross[0][1].shape[-2:]),
                [num_image_features, self.model_tester.vision_model_tester.hidden_size],
            )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True
            check_hidden_states_output(inputs_dict, config, model_class)

    # Override as `hidden states output` is different for BridgeTower
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

        output = outputs[0]

        # Encoder-/Decoder-only models
        hidden_states = outputs.hidden_states[0][0]
        hidden_states.retain_grad()

        if self.has_attentions:
            attentions = outputs.attentions[0][0]
            attentions.retain_grad()

        output.flatten()[0].backward(retain_graph=True)

        self.assertIsNotNone(hidden_states.grad)

        if self.has_attentions:
            self.assertIsNotNone(attentions.grad)

    # override as the `logit_scale` parameter initilization is different for BRIDGE TOWER
    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if name == "logit_scale":
                        self.assertAlmostEqual(
                            param.data.item(),
                            config.logit_scale_init_value,
                            delta=1e-3,
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )
                    else:
                        self.assertIn(
                            ((param.data.mean() * 1e9).round() / 1e9).item(),
                            [0.0, 1.0],
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )

    @unittest.skip(reason="""Bridge Tower does not have input/output embeddings. So this test is not applicable.""")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="""Bridge Tower does not have input/output embeddings. Thus this test is not applicable.""")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="Bridge Tower does not use inputs_embeds")
    def test_inputs_embeds_matches_input_ids(self):
        pass


# We will verify our results on an image of cute cats
def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image


@require_torch
@require_vision
class BridgeTowerModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_processor(self):
        return (
            BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-base-itm-mlm")
            if is_vision_available()
            else None
        )

    @slow
    def test_image_and_text_retrieval(self):
        model = BridgeTowerForImageAndTextRetrieval.from_pretrained("BridgeTower/bridgetower-base-itm-mlm").to(
            torch_device
        )
        model.eval()
        processor = self.default_processor
        image = prepare_img()
        text = "a bunch of cats laying on a tower."
        inputs = processor(image, text, return_tensors="pt").to(torch_device)

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # verify the logits
        expected_shape = torch.Size([1, 2])
        self.assertEqual(outputs.logits.shape, expected_shape)
        self.assertTrue(outputs.logits[0, 1].item() > outputs.logits[0, 0].item())

        # verify loss
        inputs["labels"] = torch.ones(1, dtype=torch.long, device=torch_device)
        inputs = inputs.to(torch_device)
        with torch.no_grad():
            outputs = model(**inputs)
        self.assertAlmostEqual(outputs.loss.item(), 0.5108, places=4)

    @slow
    def test_masked_language_modeling(self):
        model = BridgeTowerForMaskedLM.from_pretrained("BridgeTower/bridgetower-base-itm-mlm").to(torch_device)
        model.eval()
        processor = self.default_processor
        image = prepare_img()
        text = "a bunch of <mask> laying on a tower."
        inputs = processor(image, text, return_tensors="pt").to(torch_device)

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # verify the logits
        expected_shape = torch.Size([1, 11, 50265])
        self.assertEqual(outputs.logits.shape, expected_shape)

        # verify predicted word
        predicted_id = outputs.logits.argmax(dim=-1).squeeze(0).tolist()[4]
        self.assertTrue(processor.decode([predicted_id]) == " cats")

        # verify loss
        inputs["labels"] = inputs["input_ids"].clone()
        inputs = inputs.to(torch_device)
        with torch.no_grad():
            outputs = model(**inputs)
        self.assertAlmostEqual(outputs.loss.item(), 5.7373, places=4)

    @slow
    def test_constrastive_learning(self):
        model = BridgeTowerForContrastiveLearning.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-itc").to(
            torch_device
        )
        model.eval()
        processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-itc")
        image = prepare_img()
        text = "a bunch of cats laying on a tower."
        inputs = processor(image, text, padding=True, return_tensors="pt").to(torch_device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, return_loss=True)

        # verify the logits
        expected_shape = torch.Size([1, 3, 512])
        self.assertEqual(outputs.logits.shape, expected_shape)


@slow
@require_torch
class BridgeTowerModelTrainingTest(unittest.TestCase):
    all_training_supported_model_classes = (
        (BridgeTowerForImageAndTextRetrieval, BridgeTowerForMaskedLM, BridgeTowerForContrastiveLearning)
        if is_torch_available()
        else ()
    )

    def setUp(self):
        self.model_tester = BridgeTowerModelTester(self)
        self.config_tester = ConfigTester(self, config_class=BridgeTowerConfig, hidden_size=37, vocab_size=99)

    def _prepare_inputs_for_training(self, model_class):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        if model_class == BridgeTowerForMaskedLM:
            inputs_dict["labels"] = inputs_dict["input_ids"]
        elif model_class == BridgeTowerForImageAndTextRetrieval:
            inputs_dict["labels"] = ids_tensor([1], 2)
        elif model_class == BridgeTowerForContrastiveLearning:
            inputs_dict["return_loss"] = True
        return config, inputs_dict

    def _get_non_used_layer_names(self, model_class):
        non_used_layer_names = ["text_model.pooler"]
        if model_class == BridgeTowerForMaskedLM:
            non_used_layer_names = non_used_layer_names + [
                # This number `1` actually depends on the number of layers in `cross_modal_image_layers` (by minus 1)
                "cross_modal_image_layers.1",
                "cross_modal_image_pooler",
                "cross_modal_text_pooler",
            ]
        return non_used_layer_names

    def _is_layer_used(self, model_class, layer_name):
        non_used_layer_names = self._get_non_used_layer_names(model_class)
        for non_used_layer_name in non_used_layer_names:
            if non_used_layer_name in layer_name:
                return False
        return True

    def test_training(self):
        for model_class in self.all_training_supported_model_classes:
            config, inputs_dict = self._prepare_inputs_for_training(model_class)
            model = model_class(config)
            model.to(torch_device)
            model.train()

            loss = model(**inputs_dict).loss
            loss.backward()

            # verify the gradients of used layers' weight are not None
            for name, param in model.named_parameters():
                if self._is_layer_used(model_class, name):
                    self.assertIsNotNone(param.grad, f"Gradients should not be None - got {param.grad} for {name}")

    @slow
    def test_inference_interpolate_pos_encoding(self):
        # ViT models have an `interpolate_pos_encoding` argument in their forward method,
        # allowing to interpolate the pre-trained position embeddings in order to use
        # the model on higher resolutions. The DINO model by Facebook AI leverages this
        # to visualize self-attention on higher resolution images.
        model_name = "BridgeTower/bridgetower-base"
        model = BridgeTowerModel.from_pretrained(model_name).to(torch_device)

        image_processor = BridgeTowerProcessor.from_pretrained(model_name, size={"shortest_edge": 180})

        image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
        inputs = image_processor(text="what's in the image", images=image, return_tensors="pt").to(torch_device)

        # interpolate_pos_encodiung false should return value error
        with self.assertRaises(ValueError, msg="doesn't match model"):
            with torch.no_grad():
                model(**inputs, interpolate_pos_encoding=False)

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs, interpolate_pos_encoding=True)

        # verify the logits
        expected_shape = torch.Size((1, 122, 768))

        self.assertEqual(outputs.image_features.shape, expected_shape)

        expected_slice = torch.tensor(
            [[-0.6518, 0.4978, -0.4544], [-2.6672, -0.0843, -0.4210], [-2.4510, -0.1002, -0.3458]]
        ).to(torch_device)

        torch.testing.assert_close(outputs.image_features[0, :3, :3], expected_slice, rtol=1e-4, atol=1e-4)
