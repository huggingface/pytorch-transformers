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
"""Testing suite for the PyTorch ViLT model."""

import unittest

from datasets import load_dataset
from packaging import version

from transformers import ViltConfig, is_torch_available, is_vision_available
from transformers.testing_utils import require_torch, require_vision, slow, torch_device
from transformers.utils import cached_property

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        ViltForImageAndTextRetrieval,
        ViltForImagesAndTextClassification,
        ViltForMaskedLM,
        ViltForQuestionAnswering,
        ViltForTokenClassification,
        ViltModel,
    )
    from transformers.models.auto.modeling_auto import MODEL_MAPPING_NAMES

if is_vision_available():
    import PIL
    from PIL import Image

    from transformers import ViltProcessor


class ViltModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        image_size=30,
        patch_size=2,
        num_channels=3,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=True,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        type_sequence_label_size=2,
        initializer_range=0.02,
        num_labels=3,
        scope=None,
        modality_type_vocab_size=2,
        add_multiple_images=False,
        num_images=-1,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_token_type_ids = use_token_type_ids
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.scope = scope
        self.modality_type_vocab_size = modality_type_vocab_size
        self.add_multiple_images = add_multiple_images
        self.num_images = num_images
        # we set the expected sequence length (which is used in several tests)
        # this is equal to the seq length of the text tokens + number of image patches + 1 for the CLS token
        self.expected_seq_len = self.seq_length + (self.image_size // self.patch_size) ** 2 + 1

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        if self.add_multiple_images:
            pixel_values = floats_tensor([self.batch_size, 2, self.num_channels, self.image_size, self.image_size])
        else:
            pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)

        if self.use_labels:
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)

        config = self.get_config()

        return (config, input_ids, token_type_ids, input_mask, pixel_values, token_labels)

    def get_config(self):
        return ViltConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            is_decoder=False,
            initializer_range=self.initializer_range,
            num_labels=self.num_labels,
            modality_type_vocab_size=self.modality_type_vocab_size,
            num_images=self.num_images,
        )

    def create_and_check_model(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        pixel_values,
        token_labels,
    ):
        model = ViltModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, pixel_values=pixel_values)
        result = model(input_ids, token_type_ids=token_type_ids, pixel_values=pixel_values)
        result = model(input_ids, pixel_values=pixel_values)
        self.parent.assertEqual(
            result.last_hidden_state.shape, (self.batch_size, self.expected_seq_len, self.hidden_size)
        )

    def create_and_check_for_token_classification(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        pixel_values,
        token_labels,
    ):
        model = ViltForTokenClassification(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, pixel_values=pixel_values)
        result = model(input_ids, token_type_ids=token_type_ids, pixel_values=pixel_values)
        result = model(input_ids, pixel_values=pixel_values)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.num_labels))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            token_type_ids,
            input_mask,
            pixel_values,
            token_labels,
        ) = config_and_inputs
        inputs_dict = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": input_mask,
            "pixel_values": pixel_values,
        }
        return config, inputs_dict

    def prepare_pixel_values(self):
        return floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])


@require_torch
class ViltModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            ViltModel,
            ViltForQuestionAnswering,
            ViltForImageAndTextRetrieval,
            ViltForMaskedLM,
            ViltForTokenClassification,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {"image-feature-extraction": ViltModel, "visual-question-answering": ViltForQuestionAnswering}
        if is_torch_available()
        else {}
    )
    test_pruning = False
    test_headmasking = False
    test_torchscript = False
    model_split_percents = [0.5, 0.8, 0.9]

    # ViltForMaskedLM, ViltForQuestionAnswering and ViltForImagesAndTextClassification require special treatment
    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)

        if return_labels:
            if model_class.__name__ == "ViltForQuestionAnswering":
                inputs_dict["labels"] = torch.zeros(
                    self.model_tester.batch_size, self.model_tester.num_labels, device=torch_device
                )
            elif model_class.__name__ in ["ViltForMaskedLM", "ViltForTokenClassification"]:
                inputs_dict["labels"] = torch.zeros(
                    (self.model_tester.batch_size, self.model_tester.seq_length), dtype=torch.long, device=torch_device
                )
            elif model_class.__name__ == "ViltForImagesAndTextClassification":
                inputs_dict["labels"] = torch.zeros(
                    self.model_tester.batch_size, dtype=torch.long, device=torch_device
                )

        return inputs_dict

    def setUp(self):
        self.model_tester = ViltModelTester(self)
        self.config_tester = ConfigTester(self, config_class=ViltConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_token_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_token_classification(*config_and_inputs)

    def test_training(self):
        if not self.model_tester.is_training:
            self.skipTest(reason="model_tester.is_training is set to False.")

        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            config.return_dict = True

            if model_class.__name__ == "ViltForImagesAndTextClassification":
                config.modality_type_vocab_size = 3

            # ViltForImageAndTextRetrieval doesn't support training for now
            if model_class.__name__ in [*MODEL_MAPPING_NAMES.values(), "ViltForImageAndTextRetrieval"]:
                continue

            model = model_class(config)
            model.to(torch_device)
            model.train()
            inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            for k, v in inputs.items():
                print(k, v.shape)
            loss = model(**inputs).loss
            loss.backward()

    def test_training_gradient_checkpointing(self):
        if not self.model_tester.is_training:
            self.skipTest(reason="model_tester.is_training is set to False.")

        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            config.use_cache = False
            config.return_dict = True

            # ViltForImageAndTextRetrieval doesn't support training for now
            if (
                model_class.__name__ in [*MODEL_MAPPING_NAMES.values(), "ViltForImageAndTextRetrieval"]
                or not model_class.supports_gradient_checkpointing
            ):
                continue

            model = model_class(config)
            model.to(torch_device)
            model.gradient_checkpointing_enable()
            model.train()
            inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            loss = model(**inputs).loss
            loss.backward()

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

    @unittest.skip(
        reason="""VilT samples image tokens from a multinomial distribution, resulting in not deterministic
                            hidden states"""
    )
    def test_save_load(self):
        pass

    @unittest.skip(
        reason="""VilT samples image tokens from a multinomial distribution, resulting in not deterministic
                            hidden states"""
    )
    def test_determinism(self):
        pass

    @unittest.skip(
        "VilT samples image tokens from a multinomial distribution, resulting in not deterministic hidden states"
    )
    def test_batching_equivalence(self):
        pass

    @unittest.skip(
        reason="""VilT samples image tokens from a multinomial distribution, resulting in not deterministic
                            hidden states"""
    )
    def test_model_outputs_equivalence(self):
        pass

    @unittest.skip(
        reason="""VilT samples image tokens from a multinomial distribution, resulting in not deterministic
                            hidden states. Cannot test equivalence on logit level"""
    )
    def test_inputs_embeds_matches_input_ids(self):
        pass

    def test_attention_outputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True

        seq_len = getattr(self.model_tester, "expected_seq_len", None)

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
            if model_class.__name__ == "ViltForImagesAndTextClassification":
                # attentions are a list of length num_images
                # each element contains the attentions of a particular image index
                self.assertEqual(len(attentions), self.model_tester.num_images)
                self.assertEqual(len(attentions[0]), self.model_tester.num_hidden_layers)
            else:
                self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            if model_class.__name__ == "ViltForImagesAndTextClassification":
                # attentions are a list of length num_images
                # each element contains the attentions of a particular image index
                self.assertEqual(len(attentions), self.model_tester.num_images)
                self.assertEqual(len(attentions[0]), self.model_tester.num_hidden_layers)
            else:
                self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            if model_class.__name__ == "ViltForImagesAndTextClassification":
                self.assertListEqual(
                    list(attentions[0][0].shape[-3:]),
                    [self.model_tester.num_attention_heads, seq_len, seq_len],
                )
            else:
                self.assertListEqual(
                    list(attentions[0].shape[-3:]),
                    [self.model_tester.num_attention_heads, seq_len, seq_len],
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

            self_attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions

            if model_class.__name__ == "ViltForImagesAndTextClassification":
                self.assertEqual(len(self_attentions), self.model_tester.num_images)
                self.assertEqual(len(self_attentions[0]), self.model_tester.num_hidden_layers)
                self.assertListEqual(
                    list(self_attentions[0][0].shape[-3:]),
                    [self.model_tester.num_attention_heads, seq_len, seq_len],
                )
            else:
                self.assertEqual(len(self_attentions), self.model_tester.num_hidden_layers)
                self.assertListEqual(
                    list(self_attentions[0].shape[-3:]),
                    [self.model_tester.num_attention_heads, seq_len, seq_len],
                )

    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.encoder_hidden_states if config.is_encoder_decoder else outputs.hidden_states

            expected_num_layers = getattr(
                self.model_tester, "expected_num_hidden_layers", self.model_tester.num_hidden_layers + 1
            )
            if model_class.__name__ == "ViltForImagesAndTextClassification":
                # hidden_states are a list of length num_images
                # each element contains the hidden states of a particular image index
                self.assertEqual(len(hidden_states), self.model_tester.num_images)
                self.assertEqual(len(hidden_states[0]), expected_num_layers)
            else:
                self.assertEqual(len(hidden_states), expected_num_layers)

            seq_length = self.model_tester.expected_seq_len

            if model_class.__name__ == "ViltForImagesAndTextClassification":
                self.assertListEqual(
                    list(hidden_states[0][0].shape[-2:]),
                    [seq_length, self.model_tester.hidden_size],
                )
            else:
                self.assertListEqual(
                    list(hidden_states[0].shape[-2:]),
                    [seq_length, self.model_tester.hidden_size],
                )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            print("Model class:", model_class)
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True

            check_hidden_states_output(inputs_dict, config, model_class)

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

        # Encoder-/Decoder-only models
        hidden_states = outputs.hidden_states[0]
        attentions = outputs.attentions[0]

        if model_class.__name__ == "ViltForImagesAndTextClassification":
            # hidden_states are a list of length num_images
            # each element contains the hidden states of a particular image index
            hidden_states[0].retain_grad()
            attentions[0].retain_grad()
        else:
            hidden_states.retain_grad()
            attentions.retain_grad()

        output.flatten()[0].backward(retain_graph=True)

        if model_class.__name__ == "ViltForImagesAndTextClassification":
            # hidden_states are a list of length num_images
            # each element contains the hidden states of a particular image index
            self.assertIsNotNone(hidden_states[0].grad)
            self.assertIsNotNone(attentions[0].grad)
        else:
            self.assertIsNotNone(hidden_states.grad)
            self.assertIsNotNone(attentions.grad)

    @slow
    def test_model_from_pretrained(self):
        model_name = "dandelin/vilt-b32-mlm"
        model = ViltModel.from_pretrained(model_name)
        self.assertIsNotNone(model)


@require_torch
class ViltForImagesAndTextClassificationModelTest(ViltModelTest, unittest.TestCase):
    all_model_classes = (ViltForImagesAndTextClassification,) if is_torch_available() else ()

    def setUp(self):
        self.model_tester = ViltModelTester(self, modality_type_vocab_size=3, add_multiple_images=True, num_images=2)
        self.config_tester = ConfigTester(self, config_class=ViltConfig, hidden_size=37)

    @unittest.skip(reason="We only test the model that takes in multiple images")
    def test_model(self):
        pass

    @unittest.skip(reason="We only test the model that takes in multiple images")
    def test_for_token_classification(self):
        pass


# We will verify our results on an image of cute cats
def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image


@require_torch
@require_vision
class ViltModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_processor(self):
        return ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa") if is_vision_available() else None

    @slow
    def test_inference_masked_lm(self):
        model = ViltForMaskedLM.from_pretrained("dandelin/vilt-b32-mlm").to(torch_device)

        processor = self.default_processor
        image = prepare_img()
        text = "a bunch of [MASK] laying on a [MASK]."
        inputs = processor(image, text, return_tensors="pt").to(torch_device)

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # verify the logits
        expected_shape = torch.Size([1, 11, 30522])
        self.assertEqual(outputs.logits.shape, expected_shape)

        expected_slice = torch.tensor([-12.5061, -12.5123, -12.5174]).to(torch_device)
        torch.testing.assert_close(outputs.logits[0, 0, :3], expected_slice, rtol=1e-4, atol=1e-4)

        # verify masked token prediction equals "cats"
        predicted_id = outputs.logits[0, 4, :].argmax(-1).item()
        assert processor.decode([predicted_id]) == "cats"

    @slow
    def test_inference_visual_question_answering(self):
        model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa").to(torch_device)

        processor = self.default_processor
        image = prepare_img()
        text = "How many cats are there?"
        inputs = processor(image, text, return_tensors="pt").to(torch_device)

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # verify the logits
        expected_shape = torch.Size((1, 3129))
        self.assertEqual(outputs.logits.shape, expected_shape)

        expected_slice = torch.tensor([-15.9495, -18.1472, -10.3041]).to(torch_device)

        torch.testing.assert_close(outputs.logits[0, :3], expected_slice, rtol=1e-4, atol=1e-4)

        # compute loss
        vqa_labels = [[2, 3, 155, 800]]
        vqa_scores = [[1.0, 0.3, 0.3, 0.3]]
        labels = torch.zeros(1, model.config.num_labels).to(torch_device)

        for i, (labels_example, scores_example) in enumerate(zip(vqa_labels, vqa_scores)):
            for l, s in zip(labels_example, scores_example):
                labels[i, l] = s

        # forward pass
        outputs = model(**inputs, labels=labels)

        # verify we have a positive loss
        self.assertTrue(outputs.loss > 0)

    @slow
    def test_inference_natural_language_visual_reasoning(self):
        model = ViltForImagesAndTextClassification.from_pretrained("dandelin/vilt-b32-finetuned-nlvr2").to(
            torch_device
        )

        processor = self.default_processor

        dataset = load_dataset("hf-internal-testing/fixtures_nlvr2", split="test", trust_remote_code=True)
        image1 = Image.open(dataset[0]["file"]).convert("RGB")
        image2 = Image.open(dataset[1]["file"]).convert("RGB")

        text = (
            "The left image contains twice the number of dogs as the right image, and at least two dogs in total are"
            " standing."
        )
        encoding_1 = processor(image1, text, return_tensors="pt")
        encoding_2 = processor(image2, text, return_tensors="pt")

        pixel_values = torch.stack([encoding_1.pixel_values, encoding_2.pixel_values], dim=1)

        # forward pass
        outputs = model(
            input_ids=encoding_1.input_ids.to(torch_device),
            pixel_values=pixel_values.to(torch_device),
        )

        # verify the logits
        expected_shape = torch.Size([1, 2])
        self.assertEqual(outputs.logits.shape, expected_shape)

        is_pillow_less_than_9 = version.parse(PIL.__version__) < version.parse("9.0.0")

        if is_pillow_less_than_9:
            expected_slice = torch.tensor(
                [-2.4013, 2.9342],
                device=torch_device,
            )
        else:
            expected_slice = torch.tensor(
                [-2.3713, 2.9168],
                device=torch_device,
            )

        torch.testing.assert_close(outputs.logits[0, :3], expected_slice, rtol=1e-4, atol=1e-4)
