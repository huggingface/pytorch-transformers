# coding=utf-8
# Copyright 2020 The HuggingFace Team. All rights reserved.
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


import copy
import unittest

from transformers import IBertConfig, is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch
    from torch import nn

    from transformers import (
        IBertForMaskedLM,
        IBertForMultipleChoice,
        IBertForQuestionAnswering,
        IBertForSequenceClassification,
        IBertForTokenClassification,
        IBertModel,
    )
    from transformers.models.ibert.modeling_ibert import (
        IBertEmbeddings,
        IntGELU,
        IntLayerNorm,
        IntSoftmax,
        QuantAct,
        QuantEmbedding,
        QuantLinear,
        create_position_ids_from_input_ids,
    )


class IBertModelTester:
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
        num_choices=4,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
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
        self.num_choices = num_choices
        self.scope = scope

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

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

        config = self.get_config()

        return config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels

    def get_config(self):
        return IBertConfig(
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
            initializer_range=self.initializer_range,
            quant_mode=True,
        )

    def get_pipeline_config(self):
        config = self.get_config()
        config.vocab_size = 300
        return config

    def create_and_check_model(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = IBertModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)
        result = model(input_ids, token_type_ids=token_type_ids)
        result = model(input_ids)

        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size, self.hidden_size))

    def create_and_check_for_masked_lm(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = IBertForMaskedLM(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_for_token_classification(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.num_labels = self.num_labels
        model = IBertForTokenClassification(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.num_labels))

    def create_and_check_for_multiple_choice(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.num_choices = self.num_choices
        model = IBertForMultipleChoice(config=config)
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

    def create_and_check_for_question_answering(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = IBertForQuestionAnswering(config=config)
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
        ) = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_torch
class IBertModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    test_pruning = False
    test_torchscript = False
    test_head_masking = False
    test_resize_embeddings = False

    all_model_classes = (
        (
            IBertForMaskedLM,
            IBertModel,
            IBertForSequenceClassification,
            IBertForTokenClassification,
            IBertForMultipleChoice,
            IBertForQuestionAnswering,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": IBertModel,
            "fill-mask": IBertForMaskedLM,
            "question-answering": IBertForQuestionAnswering,
            "text-classification": IBertForSequenceClassification,
            "token-classification": IBertForTokenClassification,
            "zero-shot": IBertForSequenceClassification,
        }
        if is_torch_available()
        else {}
    )

    def setUp(self):
        self.model_tester = IBertModelTester(self)
        self.config_tester = ConfigTester(self, config_class=IBertConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_various_embeddings(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        # I-BERT only supports absolute embedding
        for type in ["absolute"]:
            config_and_inputs[0].position_embedding_type = type
            self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_masked_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_masked_lm(*config_and_inputs)

    def test_for_token_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_token_classification(*config_and_inputs)

    def test_for_multiple_choice(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_multiple_choice(*config_and_inputs)

    def test_for_question_answering(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_question_answering(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        model_name = "kssteven/ibert-roberta-base"
        model = IBertModel.from_pretrained(model_name)
        self.assertIsNotNone(model)

    def test_create_position_ids_respects_padding_index(self):
        """This is a regression test for https://github.com/huggingface/transformers/issues/1761

        The position ids should be masked with the embedding object's padding index. Therefore, the
        first available non-padding position index is IBertEmbeddings.padding_idx + 1
        """
        config = self.model_tester.prepare_config_and_inputs()[0]
        model = IBertEmbeddings(config=config)

        input_ids = torch.as_tensor([[12, 31, 13, model.padding_idx]])
        expected_positions = torch.as_tensor(
            [[0 + model.padding_idx + 1, 1 + model.padding_idx + 1, 2 + model.padding_idx + 1, model.padding_idx]]
        )

        position_ids = create_position_ids_from_input_ids(input_ids, model.padding_idx)
        self.assertEqual(position_ids.shape, expected_positions.shape)
        self.assertTrue(torch.all(torch.eq(position_ids, expected_positions)))

    def test_create_position_ids_from_inputs_embeds(self):
        """This is a regression test for https://github.com/huggingface/transformers/issues/1761
        The position ids should be masked with the embedding object's padding index. Therefore, the
        first available non-padding position index is IBertEmbeddings.padding_idx + 1
        """
        config = self.model_tester.prepare_config_and_inputs()[0]
        embeddings = IBertEmbeddings(config=config)

        inputs_embeds = torch.empty(2, 4, 30)
        expected_single_positions = [
            0 + embeddings.padding_idx + 1,
            1 + embeddings.padding_idx + 1,
            2 + embeddings.padding_idx + 1,
            3 + embeddings.padding_idx + 1,
        ]
        expected_positions = torch.as_tensor([expected_single_positions, expected_single_positions])
        position_ids = embeddings.create_position_ids_from_inputs_embeds(inputs_embeds)
        self.assertEqual(position_ids.shape, expected_positions.shape)
        self.assertTrue(torch.all(torch.eq(position_ids, expected_positions)))

    # Override
    def test_model_get_set_embeddings(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), QuantEmbedding)
            model.set_input_embeddings(nn.Embedding(10, 10))
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, nn.Linear))

    # Override
    def test_feed_forward_chunking(self):
        pass  # I-BERT does not support chunking

    # Override
    def test_inputs_embeds(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            inputs = copy.deepcopy(self._prepare_for_class(inputs_dict, model_class))

            if not self.is_encoder_decoder:
                input_ids = inputs["input_ids"]
                del inputs["input_ids"]
            else:
                encoder_input_ids = inputs["input_ids"]
                decoder_input_ids = inputs.get("decoder_input_ids", encoder_input_ids)
                del inputs["input_ids"]
                inputs.pop("decoder_input_ids", None)

            wte = model.get_input_embeddings()
            if not self.is_encoder_decoder:
                embed, embed_scaling_factor = wte(input_ids)
                inputs["inputs_embeds"] = embed
            else:
                inputs["inputs_embeds"] = wte(encoder_input_ids)
                inputs["decoder_inputs_embeds"] = wte(decoder_input_ids)

            with torch.no_grad():
                model(**inputs)[0]

    @unittest.skip(reason="ibert overrides scaling to None if inputs_embeds")
    def test_inputs_embeds_matches_input_ids(self):
        pass


@require_torch
class IBertModelIntegrationTest(unittest.TestCase):
    def test_quant_embedding(self):
        weight_bit = 8
        embedding = QuantEmbedding(2, 4, quant_mode=True, weight_bit=weight_bit)
        embedding_weight = torch.tensor([[-1.0, -2.0, -3.0, -4.0], [5.0, 6.0, 7.0, 8.0]])
        embedding.weight = nn.Parameter(embedding_weight)

        expected_scaling_factor = embedding_weight.abs().max() / (2 ** (weight_bit - 1) - 1)
        x, x_scaling_factor = embedding(torch.tensor(0))
        y, y_scaling_factor = embedding(torch.tensor(1))

        # scaling factor should follow the symmetric quantization rule
        torch.testing.assert_close(x_scaling_factor, expected_scaling_factor, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(x_scaling_factor, expected_scaling_factor, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(y_scaling_factor, expected_scaling_factor, rtol=1e-4, atol=1e-4)

        # quantization error should not exceed the scaling factor
        torch.testing.assert_close(x, embedding_weight[0], rtol=expected_scaling_factor, atol=expected_scaling_factor)
        torch.testing.assert_close(y, embedding_weight[1], rtol=expected_scaling_factor, atol=expected_scaling_factor)

    def test_quant_act(self):
        def _test_range():
            act = QuantAct(activation_bit, act_range_momentum, quant_mode=True)

            # First pass
            x = torch.tensor([[-1.0, -2.0, -3.0, -4.0], [5.0, 6.0, 7.0, 8.0]])
            x_scaling_factor = torch.tensor(1.0)
            y, y_scaling_factor = act(x, x_scaling_factor)
            y_int = y / y_scaling_factor

            # After the first pass, x_min and x_max should be initialized with x.min() and x.max()
            expected_x_min, expected_x_max = x.min(), x.max()
            torch.testing.assert_close(act.x_min, expected_x_min, rtol=1e-4, atol=1e-4)
            torch.testing.assert_close(act.x_max, expected_x_max, rtol=1e-4, atol=1e-4)

            # scaling factor should follow the symmetric quantization rule
            expected_range = torch.max(expected_x_min.abs(), expected_x_max.abs())
            expected_scaling_factor = expected_range / (2 ** (activation_bit - 1) - 1)
            torch.testing.assert_close(y_scaling_factor, expected_scaling_factor, rtol=1e-4, atol=1e-4)

            # quantization error should not exceed the scaling factor
            torch.testing.assert_close(x, y, rtol=expected_scaling_factor, atol=expected_scaling_factor)

            # output should be integer
            torch.testing.assert_close(y_int, y_int.round(), rtol=1e-4, atol=1e-4)

            # Second Pass
            x = torch.tensor([[-1.0, -2.0, -3.0, -4.0], [5.0, 6.0, 7.0, 8.0]]) * 2
            x_scaling_factor = torch.tensor(1.0)
            y, y_scaling_factor = act(x, x_scaling_factor)
            y_int = y / y_scaling_factor

            # From the second pass, x_min and x_max should be updated with moving average
            expected_x_min = expected_x_min * act_range_momentum + x.min() * (1 - act_range_momentum)
            expected_x_max = expected_x_max * act_range_momentum + x.max() * (1 - act_range_momentum)
            torch.testing.assert_close(act.x_min, expected_x_min, rtol=1e-4, atol=1e-4)
            torch.testing.assert_close(act.x_max, expected_x_max, rtol=1e-4, atol=1e-4)

            # scaling factor should follow the symmetric quantization rule
            expected_range = torch.max(expected_x_min.abs(), expected_x_max.abs())
            expected_scaling_factor = expected_range / (2 ** (activation_bit - 1) - 1)
            torch.testing.assert_close(y_scaling_factor, expected_scaling_factor, rtol=1e-4, atol=1e-4)

            # quantization error should not exceed the scaling factor
            x = x.clamp(min=-expected_range, max=expected_range)
            torch.testing.assert_close(x, y, rtol=expected_scaling_factor, atol=expected_scaling_factor)

            # output should be integer
            torch.testing.assert_close(y_int, y_int.round(), rtol=1e-4, atol=1e-4)

            # Third pass, with eval()
            act.eval()
            x = torch.tensor([[-1.0, -2.0, -3.0, -4.0], [5.0, 6.0, 7.0, 8.0]]) * 3

            # In eval mode, min/max and scaling factor must be fixed
            torch.testing.assert_close(act.x_min, expected_x_min, rtol=1e-4, atol=1e-4)
            torch.testing.assert_close(act.x_max, expected_x_max, rtol=1e-4, atol=1e-4)
            torch.testing.assert_close(y_scaling_factor, expected_scaling_factor, rtol=1e-4, atol=1e-4)

        def _test_identity():
            # test if identity and identity_scaling_factor are given
            # should add the input values
            act = QuantAct(activation_bit, act_range_momentum, quant_mode=True)
            x = torch.tensor([[-1.0, -2.0, -3.0, -4.0], [5.0, 6.0, 7.0, 8.0]])
            y = torch.tensor([[6.0, -7.0, 1.0, -2.0], [3.0, -4.0, -8.0, 5.0]])
            x_scaling_factor = torch.tensor(1.0)
            y_scaling_factor = torch.tensor(0.5)
            z, z_scaling_factor = act(x, x_scaling_factor, y, y_scaling_factor)
            z_int = z / z_scaling_factor
            torch.testing.assert_close(x + y, z, rtol=0.1, atol=0.1)
            torch.testing.assert_close(z_int, z_int.round(), rtol=1e-4, atol=1e-4)

        activation_bit = 8
        act_range_momentum = 0.95
        _test_range()
        _test_identity()

    def test_quant_linear(self):
        def _test(per_channel):
            linear_q = QuantLinear(2, 4, quant_mode=True, per_channel=per_channel, weight_bit=weight_bit)
            linear_dq = QuantLinear(2, 4, quant_mode=False, per_channel=per_channel, weight_bit=weight_bit)
            linear_weight = torch.tensor([[-1.0, 2.0, 3.0, -4.0], [5.0, -6.0, -7.0, 8.0]]).T
            linear_q.weight = nn.Parameter(linear_weight)
            linear_dq.weight = nn.Parameter(linear_weight)

            q, q_scaling_factor = linear_q(x, x_scaling_factor)
            q_int = q / q_scaling_factor
            dq, dq_scaling_factor = linear_dq(x, x_scaling_factor)

            if per_channel:
                q_max = linear_weight.abs().max(dim=1).values
            else:
                q_max = linear_weight.abs().max()
            expected_scaling_factor = q_max / (2 ** (weight_bit - 1) - 1)

            # scaling factor should follow the symmetric quantization rule
            torch.testing.assert_close(linear_q.fc_scaling_factor, expected_scaling_factor, rtol=1e-4, atol=1e-4)

            # output of the normal linear layer and the quantized linear layer should be similar
            torch.testing.assert_close(q, dq, rtol=0.5, atol=0.5)

            # output of the quantized linear layer should be integer
            torch.testing.assert_close(q_int, q_int.round(), rtol=1e-4, atol=1e-4)

        weight_bit = 8
        x = torch.tensor([[2.0, -5.0], [-3.0, 4.0]])
        x_scaling_factor = torch.tensor([1.0])
        _test(True)
        _test(False)

    def test_int_gelu(self):
        gelu_q = IntGELU(quant_mode=True)
        gelu_dq = nn.GELU()

        x_int = torch.arange(-10000, 10001, 1)
        x_scaling_factor = torch.tensor(0.001)
        x = x_int * x_scaling_factor

        q, q_scaling_factor = gelu_q(x, x_scaling_factor)
        q_int = q / q_scaling_factor
        dq = gelu_dq(x)

        # output of the normal GELU and the quantized GELU should be similar
        torch.testing.assert_close(q, dq, rtol=0.5, atol=0.5)

        # output of the quantized GELU layer should be integer
        torch.testing.assert_close(q_int, q_int.round(), rtol=1e-4, atol=1e-4)

    def test_force_dequant_gelu(self):
        x_int = torch.arange(-10000, 10001, 1)
        x_scaling_factor = torch.tensor(0.001)
        x = x_int * x_scaling_factor

        gelu_dq = IntGELU(quant_mode=False)
        gelu_fdqs_dict = {
            True: [
                IntGELU(quant_mode=True, force_dequant="nonlinear"),
                IntGELU(quant_mode=True, force_dequant="gelu"),
            ],
            False: [
                IntGELU(quant_mode=True, force_dequant="none"),
                IntGELU(quant_mode=True, force_dequant="softmax"),
                IntGELU(quant_mode=True, force_dequant="layernorm"),
            ],
        }

        dq, dq_scaling_factor = gelu_dq(x, x_scaling_factor)
        for label, gelu_fdqs in gelu_fdqs_dict.items():
            for gelu_fdq in gelu_fdqs:
                q, q_scaling_factor = gelu_fdq(x, x_scaling_factor)
                if label:
                    torch.testing.assert_close(q, dq, rtol=1e-4, atol=1e-4)
                else:
                    self.assertFalse(torch.allclose(q, dq, atol=1e-4))

    def test_int_softmax(self):
        output_bit = 8
        softmax_q = IntSoftmax(output_bit, quant_mode=True)
        softmax_dq = nn.Softmax()

        def _test(array):
            x_int = torch.tensor(array)
            x_scaling_factor = torch.tensor(0.1)
            x = x_int * x_scaling_factor

            q, q_scaling_factor = softmax_q(x, x_scaling_factor)
            q_int = q / q_scaling_factor
            dq = softmax_dq(x)

            # output of the normal Softmax and the quantized Softmax should be similar
            torch.testing.assert_close(q, dq, rtol=0.5, atol=0.5)

            # output of the quantized GELU layer should be integer
            torch.testing.assert_close(q_int, q_int.round(), rtol=1e-4, atol=1e-4)

            # Output of the quantize Softmax should not exceed the output_bit
            self.assertTrue(q.abs().max() < 2**output_bit)

        array = [[i + j for j in range(10)] for i in range(-10, 10)]
        _test(array)
        array = [[i + j for j in range(50)] for i in range(-10, 10)]
        _test(array)
        array = [[i + 100 * j for j in range(2)] for i in range(-10, 10)]
        _test(array)

    def test_force_dequant_softmax(self):
        output_bit = 8
        array = [[i + j for j in range(10)] for i in range(-10, 10)]
        x_int = torch.tensor(array)
        x_scaling_factor = torch.tensor(0.1)
        x = x_int * x_scaling_factor

        softmax_dq = IntSoftmax(output_bit, quant_mode=False)
        softmax_fdqs_dict = {
            True: [
                IntSoftmax(output_bit, quant_mode=True, force_dequant="nonlinear"),
                IntSoftmax(output_bit, quant_mode=True, force_dequant="softmax"),
            ],
            False: [
                IntSoftmax(output_bit, quant_mode=True, force_dequant="none"),
                IntSoftmax(output_bit, quant_mode=True, force_dequant="gelu"),
                IntSoftmax(output_bit, quant_mode=True, force_dequant="layernorm"),
            ],
        }

        dq, dq_scaling_factor = softmax_dq(x, x_scaling_factor)
        for label, softmax_fdqs in softmax_fdqs_dict.items():
            for softmax_fdq in softmax_fdqs:
                q, q_scaling_factor = softmax_fdq(x, x_scaling_factor)
                if label:
                    torch.testing.assert_close(q, dq, rtol=1e-4, atol=1e-4)
                else:
                    self.assertFalse(torch.allclose(q, dq, atol=1e-4))

    def test_int_layernorm(self):
        output_bit = 8

        # some random matrix
        array = [[[i * j * j + j for j in range(5, 15)]] for i in range(-10, 10)]
        x_int = torch.tensor(array)
        x_scaling_factor = torch.tensor(0.1)
        x = x_int * x_scaling_factor

        ln_q = IntLayerNorm(x.shape[1:], 1e-5, quant_mode=True, output_bit=output_bit)
        ln_dq = nn.LayerNorm(x.shape[1:], 1e-5)

        ln_q.weight = nn.Parameter(torch.ones(x.shape[1:]))
        ln_q.bias = nn.Parameter(torch.ones(x.shape[1:]))
        ln_dq.weight = nn.Parameter(torch.ones(x.shape[1:]))
        ln_dq.bias = nn.Parameter(torch.ones(x.shape[1:]))

        q, q_scaling_factor = ln_q(x, x_scaling_factor)
        q_int = q / q_scaling_factor
        dq = ln_dq(x)

        # output of the normal LN and the quantized LN should be similar
        torch.testing.assert_close(q, dq, rtol=0.5, atol=0.5)

        # output of the quantized GELU layer should be integer
        torch.testing.assert_close(q_int, q_int.round(), rtol=1e-4, atol=1e-4)

    def test_force_dequant_layernorm(self):
        output_bit = 8
        array = [[[i * j * j + j for j in range(5, 15)]] for i in range(-10, 10)]
        x_int = torch.tensor(array)
        x_scaling_factor = torch.tensor(0.1)
        x = x_int * x_scaling_factor

        ln_dq = IntLayerNorm(x.shape[1:], 1e-5, quant_mode=False, output_bit=output_bit)
        ln_fdqs_dict = {
            True: [
                IntLayerNorm(x.shape[1:], 1e-5, quant_mode=True, output_bit=output_bit, force_dequant="nonlinear"),
                IntLayerNorm(x.shape[1:], 1e-5, quant_mode=True, output_bit=output_bit, force_dequant="layernorm"),
            ],
            False: [
                IntLayerNorm(x.shape[1:], 1e-5, quant_mode=True, output_bit=output_bit, force_dequant="none"),
                IntLayerNorm(x.shape[1:], 1e-5, quant_mode=True, output_bit=output_bit, force_dequant="gelu"),
                IntLayerNorm(x.shape[1:], 1e-5, quant_mode=True, output_bit=output_bit, force_dequant="softmax"),
            ],
        }

        ln_dq.weight = nn.Parameter(torch.ones(x.shape[1:]))
        ln_dq.bias = nn.Parameter(torch.ones(x.shape[1:]))
        dq, dq_scaling_factor = ln_dq(x, x_scaling_factor)
        for label, ln_fdqs in ln_fdqs_dict.items():
            for ln_fdq in ln_fdqs:
                ln_fdq.weight = nn.Parameter(torch.ones(x.shape[1:]))
                ln_fdq.bias = nn.Parameter(torch.ones(x.shape[1:]))
                q, q_scaling_factor = ln_fdq(x, x_scaling_factor)
                if label:
                    torch.testing.assert_close(q, dq, rtol=1e-4, atol=1e-4)
                else:
                    self.assertFalse(torch.allclose(q, dq, atol=1e-4))

    def quantize(self, model):
        # Helper function that quantizes the given model
        # Recursively convert all the `quant_mode` attributes as `True`
        if hasattr(model, "quant_mode"):
            model.quant_mode = True
        elif isinstance(model, nn.Sequential):
            for n, m in model.named_children():
                self.quantize(m)
        elif isinstance(model, nn.ModuleList):
            for n in model:
                self.quantize(n)
        else:
            for attr in dir(model):
                mod = getattr(model, attr)
                if isinstance(mod, nn.Module) and mod != model:
                    self.quantize(mod)

    @slow
    def test_inference_masked_lm(self):
        # I-BERT should be "equivalent" to RoBERTa if not quantized
        # Test coped from `test_modeling_roberta.py`
        model = IBertForMaskedLM.from_pretrained("kssteven/ibert-roberta-base")
        input_ids = torch.tensor([[0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2]])
        output = model(input_ids)[0]
        expected_shape = torch.Size((1, 11, 50265))
        self.assertEqual(output.shape, expected_shape)
        expected_slice = torch.tensor(
            [[[33.8802, -4.3103, 22.7761], [4.6539, -2.8098, 13.6253], [1.8228, -3.6898, 8.8600]]]
        )
        torch.testing.assert_close(output[:, :3, :3], expected_slice, rtol=1e-4, atol=1e-4)

        # I-BERT should be "similar" to RoBERTa if quantized
        self.quantize(model)
        output = model(input_ids)[0]
        self.assertEqual(output.shape, expected_shape)
        torch.testing.assert_close(output[:, :3, :3], expected_slice, rtol=0.1, atol=0.1)

    @slow
    def test_inference_classification_head(self):
        # I-BERT should be "equivalent" to RoBERTa if not quantized
        # Test coped from `test_modeling_roberta.py`
        model = IBertForSequenceClassification.from_pretrained("kssteven/ibert-roberta-large-mnli")
        input_ids = torch.tensor([[0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2]])
        output = model(input_ids)[0]
        expected_shape = torch.Size((1, 3))
        self.assertEqual(output.shape, expected_shape)
        expected_tensor = torch.tensor([[-0.9469, 0.3913, 0.5118]])
        torch.testing.assert_close(output, expected_tensor, rtol=1e-4, atol=1e-4)

        # I-BERT should be "similar" to RoBERTa if quantized
        self.quantize(model)
        output = model(input_ids)[0]
        self.assertEqual(output.shape, expected_shape)
        torch.testing.assert_close(output, expected_tensor, rtol=0.1, atol=0.1)
