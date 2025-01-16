# coding=utf-8
# Copyright 2020 HuggingFace Inc. team.
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


import unittest

from transformers import FunnelConfig, FunnelTokenizer, is_torch_available
from transformers.models.auto import get_values
from transformers.testing_utils import require_sentencepiece, require_tokenizers, require_torch, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        MODEL_FOR_PRETRAINING_MAPPING,
        FunnelBaseModel,
        FunnelForMaskedLM,
        FunnelForMultipleChoice,
        FunnelForPreTraining,
        FunnelForQuestionAnswering,
        FunnelForSequenceClassification,
        FunnelForTokenClassification,
        FunnelModel,
    )


class FunnelModelTester:
    """You can also import this e.g, from .test_modeling_funnel import FunnelModelTester"""

    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=True,
        use_labels=True,
        vocab_size=99,
        block_sizes=[1, 1, 2],
        num_decoder_layers=1,
        d_model=32,
        n_head=4,
        d_head=8,
        d_inner=37,
        hidden_act="gelu_new",
        hidden_dropout=0.1,
        attention_dropout=0.1,
        activation_dropout=0.0,
        max_position_embeddings=512,
        type_vocab_size=3,
        initializer_std=0.02,  # Set to a smaller value, so we can keep the small error threshold (1e-5) in the test
        num_labels=3,
        num_choices=4,
        scope=None,
        base=False,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_token_type_ids = use_token_type_ids
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.block_sizes = block_sizes
        self.num_decoder_layers = num_decoder_layers
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head
        self.d_inner = d_inner
        self.hidden_act = hidden_act
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = 2
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.scope = scope
        self.initializer_std = initializer_std

        # Used in the tests to check the size of the first attention layer
        self.num_attention_heads = n_head
        # Used in the tests to check the size of the first hidden state
        self.hidden_size = self.d_model
        # Used in the tests to check the number of output hidden states/attentions
        self.num_hidden_layers = sum(self.block_sizes) + (0 if base else self.num_decoder_layers)
        # FunnelModel adds two hidden layers: input embeddings and the sum of the upsampled encoder hidden state with
        # the last hidden state of the first block (which is the first hidden state of the decoder).
        if not base:
            self.expected_num_hidden_layers = self.num_hidden_layers + 2

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = ids_tensor([self.batch_size, self.seq_length], vocab_size=2)

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)

        sequence_labels = None
        token_labels = None
        choice_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)
            fake_token_labels = ids_tensor([self.batch_size, self.seq_length], 1)

        config = self.get_config()

        return (
            config,
            input_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
            fake_token_labels,
        )

    def get_config(self):
        return FunnelConfig(
            vocab_size=self.vocab_size,
            block_sizes=self.block_sizes,
            num_decoder_layers=self.num_decoder_layers,
            d_model=self.d_model,
            n_head=self.n_head,
            d_head=self.d_head,
            d_inner=self.d_inner,
            hidden_act=self.hidden_act,
            hidden_dropout=self.hidden_dropout,
            attention_dropout=self.attention_dropout,
            activation_dropout=self.activation_dropout,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            initializer_std=self.initializer_std,
        )

    def create_and_check_model(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
        fake_token_labels,
    ):
        model = FunnelModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)
        result = model(input_ids, token_type_ids=token_type_ids)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.d_model))

        model.config.truncate_seq = False
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.d_model))

        model.config.separate_cls = False
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.d_model))

    def create_and_check_base_model(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
        fake_token_labels,
    ):
        model = FunnelBaseModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)
        result = model(input_ids, token_type_ids=token_type_ids)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, 2, self.d_model))

        model.config.truncate_seq = False
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, 3, self.d_model))

        model.config.separate_cls = False
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, 2, self.d_model))

    def create_and_check_for_pretraining(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
        fake_token_labels,
    ):
        config.num_labels = self.num_labels
        model = FunnelForPreTraining(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, labels=fake_token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length))

    def create_and_check_for_masked_lm(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
        fake_token_labels,
    ):
        model = FunnelForMaskedLM(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_for_sequence_classification(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
        fake_token_labels,
    ):
        config.num_labels = self.num_labels
        model = FunnelForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, labels=sequence_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def create_and_check_for_multiple_choice(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
        fake_token_labels,
    ):
        config.num_choices = self.num_choices
        model = FunnelForMultipleChoice(config=config)
        model.to(torch_device)
        model.eval()
        multiple_choice_inputs_ids = input_ids.unsqueeze(1).expand(-1, self.num_choices, -1).contiguous()
        multiple_choice_token_type_ids = token_type_ids.unsqueeze(1).expand(-1, self.num_choices, -1).contiguous()
        multiple_choice_input_mask = input_mask.unsqueeze(1).expand(-1, self.num_choices, -1).contiguous()
        result = model(
            multiple_choice_inputs_ids,
            attention_mask=multiple_choice_input_mask,
            token_type_ids=multiple_choice_token_type_ids,
            labels=choice_labels,
        )
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_choices))

    def create_and_check_for_token_classification(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
        fake_token_labels,
    ):
        config.num_labels = self.num_labels
        model = FunnelForTokenClassification(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.num_labels))

    def create_and_check_for_question_answering(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
        fake_token_labels,
    ):
        model = FunnelForQuestionAnswering(config=config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            start_positions=sequence_labels,
            end_positions=sequence_labels,
        )
        self.parent.assertEqual(result.start_logits.shape, (self.batch_size, self.seq_length))
        self.parent.assertEqual(result.end_logits.shape, (self.batch_size, self.seq_length))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
            fake_token_labels,
        ) = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_torch
class FunnelModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    test_head_masking = False
    test_pruning = False
    all_model_classes = (
        (
            FunnelModel,
            FunnelForMaskedLM,
            FunnelForPreTraining,
            FunnelForQuestionAnswering,
            FunnelForTokenClassification,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": (FunnelBaseModel, FunnelModel),
            "fill-mask": FunnelForMaskedLM,
            "question-answering": FunnelForQuestionAnswering,
            "text-classification": FunnelForSequenceClassification,
            "token-classification": FunnelForTokenClassification,
            "zero-shot": FunnelForSequenceClassification,
        }
        if is_torch_available()
        else {}
    )

    # special case for ForPreTraining model
    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)

        if return_labels:
            if model_class in get_values(MODEL_FOR_PRETRAINING_MAPPING):
                inputs_dict["labels"] = torch.zeros(
                    (self.model_tester.batch_size, self.model_tester.seq_length), dtype=torch.long, device=torch_device
                )
        return inputs_dict

    def setUp(self):
        self.model_tester = FunnelModelTester(self)
        self.config_tester = ConfigTester(self, config_class=FunnelConfig)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_pretraining(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_pretraining(*config_and_inputs)

    def test_for_masked_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_masked_lm(*config_and_inputs)

    def test_for_token_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_token_classification(*config_and_inputs)

    def test_for_question_answering(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_question_answering(*config_and_inputs)

    # overwrite from test_modeling_common
    def _mock_init_weights(self, module):
        if hasattr(module, "weight") and module.weight is not None:
            module.weight.data.fill_(3)
        if hasattr(module, "bias") and module.bias is not None:
            module.bias.data.fill_(3)

        for param in ["r_w_bias", "r_r_bias", "r_kernel", "r_s_bias", "seg_embed"]:
            if hasattr(module, param) and getattr(module, param) is not None:
                weight = getattr(module, param)
                weight.data.fill_(3)


@require_torch
class FunnelBaseModelTest(ModelTesterMixin, unittest.TestCase):
    test_head_masking = False
    test_pruning = False
    all_model_classes = (
        (FunnelBaseModel, FunnelForMultipleChoice, FunnelForSequenceClassification) if is_torch_available() else ()
    )

    def setUp(self):
        self.model_tester = FunnelModelTester(self, base=True)
        self.config_tester = ConfigTester(self, config_class=FunnelConfig)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_base_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_base_model(*config_and_inputs)

    def test_for_sequence_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_sequence_classification(*config_and_inputs)

    def test_for_multiple_choice(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_multiple_choice(*config_and_inputs)

    # overwrite from test_modeling_common
    def test_training(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True

        for model_class in self.all_model_classes:
            if model_class.__name__ == "FunnelBaseModel":
                continue
            model = model_class(config)
            model.to(torch_device)
            model.train()
            inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            loss = model(**inputs).loss
            loss.backward()

    # overwrite from test_modeling_common
    def _mock_init_weights(self, module):
        if hasattr(module, "weight") and module.weight is not None:
            module.weight.data.fill_(3)
        if hasattr(module, "bias") and module.bias is not None:
            module.bias.data.fill_(3)

        for param in ["r_w_bias", "r_r_bias", "r_kernel", "r_s_bias", "seg_embed"]:
            if hasattr(module, param) and getattr(module, param) is not None:
                weight = getattr(module, param)
                weight.data.fill_(3)


@require_torch
@require_sentencepiece
@require_tokenizers
class FunnelModelIntegrationTest(unittest.TestCase):
    def test_inference_tiny_model(self):
        batch_size = 13
        sequence_length = 7
        input_ids = torch.arange(0, batch_size * sequence_length).long().reshape(batch_size, sequence_length)
        lengths = [0, 1, 2, 3, 4, 5, 6, 4, 1, 3, 5, 0, 1]
        token_type_ids = torch.tensor([[2] + [0] * a + [1] * (sequence_length - a - 1) for a in lengths])

        model = FunnelModel.from_pretrained("sgugger/funnel-random-tiny")
        output = model(input_ids, token_type_ids=token_type_ids)[0].abs()

        expected_output_sum = torch.tensor(2344.8352)
        expected_output_mean = torch.tensor(0.8052)
        torch.testing.assert_close(output.sum(), expected_output_sum, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(output.mean(), expected_output_mean, rtol=1e-4, atol=1e-4)

        attention_mask = torch.tensor([[1] * 7, [1] * 4 + [0] * 3] * 6 + [[0, 1, 1, 0, 0, 1, 1]])
        output = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0].abs()

        expected_output_sum = torch.tensor(2343.8425)
        expected_output_mean = torch.tensor(0.8049)
        torch.testing.assert_close(output.sum(), expected_output_sum, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(output.mean(), expected_output_mean, rtol=1e-4, atol=1e-4)

    @slow
    def test_inference_model(self):
        tokenizer = FunnelTokenizer.from_pretrained("huggingface/funnel-small")
        model = FunnelModel.from_pretrained("huggingface/funnel-small")
        inputs = tokenizer("Hello! I am the Funnel Transformer model.", return_tensors="pt")
        output = model(**inputs)[0]

        expected_output_sum = torch.tensor(235.7246)
        expected_output_mean = torch.tensor(0.0256)
        torch.testing.assert_close(output.sum(), expected_output_sum, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(output.mean(), expected_output_mean, rtol=1e-4, atol=1e-4)
