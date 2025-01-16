# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
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

from transformers import XLMRobertaTokenizer, is_torch_available
from transformers.testing_utils import require_sentencepiece, require_tokenizers, require_torch, slow, torch_device

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        XmodConfig,
        XmodForCausalLM,
        XmodForMaskedLM,
        XmodForMultipleChoice,
        XmodForQuestionAnswering,
        XmodForSequenceClassification,
        XmodForTokenClassification,
        XmodModel,
    )
    from transformers.models.xmod.modeling_xmod import XmodEmbeddings, create_position_ids_from_input_ids


class XmodModelTester:
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
        return XmodConfig(
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
            default_language="en_XX",
        )

    def prepare_config_and_inputs_for_decoder(self):
        (
            config,
            input_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = self.prepare_config_and_inputs()

        config.is_decoder = True
        encoder_hidden_states = floats_tensor([self.batch_size, self.seq_length, self.hidden_size])
        encoder_attention_mask = ids_tensor([self.batch_size, self.seq_length], vocab_size=2)

        return (
            config,
            input_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
            encoder_hidden_states,
            encoder_attention_mask,
        )

    def create_and_check_model(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = XmodModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)
        result = model(input_ids, token_type_ids=token_type_ids)
        result = model(input_ids)

        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size, self.hidden_size))

    def create_and_check_model_as_decoder(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
        encoder_hidden_states,
        encoder_attention_mask,
    ):
        config.add_cross_attention = True
        model = XmodModel(config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        result = model(
            input_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            encoder_hidden_states=encoder_hidden_states,
        )
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size, self.hidden_size))

    def create_and_check_for_causal_lm(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
        encoder_hidden_states,
        encoder_attention_mask,
    ):
        model = XmodForCausalLM(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_decoder_model_past_large_inputs(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
        encoder_hidden_states,
        encoder_attention_mask,
    ):
        config.is_decoder = True
        config.add_cross_attention = True
        model = XmodForCausalLM(config=config).to(torch_device).eval()

        # make sure that ids don't start with pad token
        mask = input_ids.ne(config.pad_token_id).long()
        input_ids = input_ids * mask

        # first forward pass
        outputs = model(
            input_ids,
            attention_mask=input_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values

        # create hypothetical multiple next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 3), config.vocab_size)

        # make sure that ids don't start with pad token
        mask = next_tokens.ne(config.pad_token_id).long()
        next_tokens = next_tokens * mask
        next_mask = ids_tensor((self.batch_size, 3), vocab_size=2)

        # append to next input_ids and
        next_input_ids = torch.cat([input_ids, next_tokens], dim=-1)
        next_attention_mask = torch.cat([input_mask, next_mask], dim=-1)

        output_from_no_past = model(
            next_input_ids,
            attention_mask=next_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_hidden_states=True,
        )["hidden_states"][0]
        output_from_past = model(
            next_tokens,
            attention_mask=next_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            output_hidden_states=True,
        )["hidden_states"][0]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, -3:, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, :, random_slice_idx].detach()

        self.parent.assertTrue(output_from_past_slice.shape[1] == next_tokens.shape[1])

        # test that outputs are equal for slice
        self.parent.assertTrue(torch.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-3))

    def create_and_check_for_masked_lm(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = XmodForMaskedLM(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_for_token_classification(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.num_labels = self.num_labels
        model = XmodForTokenClassification(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.num_labels))

    def create_and_check_for_multiple_choice(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.num_choices = self.num_choices
        model = XmodForMultipleChoice(config=config)
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
        model = XmodForQuestionAnswering(config=config)
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
class XmodModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            XmodForCausalLM,
            XmodForMaskedLM,
            XmodModel,
            XmodForSequenceClassification,
            XmodForTokenClassification,
            XmodForMultipleChoice,
            XmodForQuestionAnswering,
        )
        if is_torch_available()
        else ()
    )
    all_generative_model_classes = (XmodForCausalLM,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": XmodModel,
            "fill-mask": XmodForMaskedLM,
            "question-answering": XmodForQuestionAnswering,
            "text-classification": XmodForSequenceClassification,
            "text-generation": XmodForCausalLM,
            "token-classification": XmodForTokenClassification,
            "zero-shot": XmodForSequenceClassification,
        }
        if is_torch_available()
        else {}
    )

    # TODO: Fix the failed tests
    def is_pipeline_test_to_skip(
        self,
        pipeline_test_case_name,
        config_class,
        model_architecture,
        tokenizer_name,
        image_processor_name,
        feature_extractor_name,
        processor_name,
    ):
        if pipeline_test_case_name == "QAPipelineTests" and not tokenizer_name.endswith("Fast"):
            return True

        return False

    def setUp(self):
        self.model_tester = XmodModelTester(self)
        self.config_tester = ConfigTester(self, config_class=XmodConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_various_embeddings(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        for type in ["absolute", "relative_key", "relative_key_query"]:
            config_and_inputs[0].position_embedding_type = type
            self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_as_decoder(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_decoder()
        self.model_tester.create_and_check_model_as_decoder(*config_and_inputs)

    def test_model_as_decoder_with_default_input_mask(self):
        # This regression test was failing with PyTorch < 1.3
        (
            config,
            input_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
            encoder_hidden_states,
            encoder_attention_mask,
        ) = self.model_tester.prepare_config_and_inputs_for_decoder()

        input_mask = None

        self.model_tester.create_and_check_model_as_decoder(
            config,
            input_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
            encoder_hidden_states,
            encoder_attention_mask,
        )

    def test_for_causal_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_decoder()
        self.model_tester.create_and_check_for_causal_lm(*config_and_inputs)

    def test_decoder_model_past_with_large_inputs(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_decoder()
        self.model_tester.create_and_check_decoder_model_past_large_inputs(*config_and_inputs)

    def test_decoder_model_past_with_large_inputs_relative_pos_emb(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_decoder()
        config_and_inputs[0].position_embedding_type = "relative_key"
        self.model_tester.create_and_check_decoder_model_past_large_inputs(*config_and_inputs)

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

    def test_create_position_ids_respects_padding_index(self):
        """This is a regression test for https://github.com/huggingface/transformers/issues/1761

        The position ids should be masked with the embedding object's padding index. Therefore, the
        first available non-padding position index is XmodEmbeddings.padding_idx + 1
        """
        config = self.model_tester.prepare_config_and_inputs()[0]
        model = XmodEmbeddings(config=config)

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
        first available non-padding position index is XmodEmbeddings.padding_idx + 1
        """
        config = self.model_tester.prepare_config_and_inputs()[0]
        embeddings = XmodEmbeddings(config=config)

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

    def test_set_default_language(self):
        config = self.model_tester.prepare_config_and_inputs()[0]
        model = XmodForMaskedLM(config=config)
        model.set_default_language("en_XX")
        self.assertEqual(model.config.default_language, "en_XX")
        with self.assertRaises(ValueError):
            model.set_default_language("xx_XX")

    def test_freeze_embeddings_and_language_adapters(self):
        config = self.model_tester.prepare_config_and_inputs()[0]
        model = XmodForMaskedLM(config=config)
        num_trainable_params_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model.freeze_embeddings_and_language_adapters()
        num_trainable_params_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.assertLess(num_trainable_params_after, num_trainable_params_before)


@require_sentencepiece
@require_tokenizers
@require_torch
class XmodModelIntegrationTest(unittest.TestCase):
    @slow
    def test_xmod_base(self):
        model = XmodModel.from_pretrained("facebook/xmod-base")

        # language en_XX
        model.set_default_language("en_XX")
        input_ids = torch.tensor([[0, 581, 10269, 83, 99942, 136, 60742, 23, 70, 80583, 18276, 2]])
        # The dog is cute and lives in the garden house
        expected_output_shape = torch.Size((1, 12, 768))  # batch_size, sequence_length, embedding_vector_dim
        expected_output_values_last_dim = torch.tensor(
            [[-0.2394, -0.0036, 0.1252, -0.0087, 0.1325, 0.0580, -0.2049, -0.1978, -0.1223, 0.0648, -0.2599, -0.3724]]
        )
        output = model(input_ids)["last_hidden_state"].detach()
        self.assertEqual(output.shape, expected_output_shape)
        # compare the actual values for a slice of last dim
        torch.testing.assert_close(output[:, :, -1], expected_output_values_last_dim, rtol=1e-3, atol=1e-3)

        # language de_DE
        model.set_default_language("de_DE")
        input_ids = torch.tensor([[0, 1310, 49083, 443, 269, 71, 5486, 165, 60429, 660, 23, 2315, 58761, 18391, 5, 2]])
        # Der Hund ist niedlich und wohnt in einem Gartenhaus.
        expected_output_shape = torch.Size((1, 16, 768))  # batch_size, sequence_length, embedding_vector_dim
        # fmt: off
        expected_output_values_last_dim = torch.tensor(
            [[0.0162, 0.0075, -0.1882, 0.2335, -0.0952, -0.3994, -0.0317, -0.1174, 0.0177, 0.4280, -0.0240, -0.2138,
              0.0785, -0.1045, -0.2811, -0.3220]]
        )
        # fmt: on
        output = model(input_ids)["last_hidden_state"].detach()
        self.assertEqual(output.shape, expected_output_shape)
        # compare the actual values for a slice of last dim
        torch.testing.assert_close(output[:, :, -1], expected_output_values_last_dim, rtol=1e-3, atol=1e-3)

    @slow
    def test_xmod_large_prenorm(self):
        model = XmodModel.from_pretrained("facebook/xmod-large-prenorm")

        # language en_XX
        model.set_default_language("en_XX")
        input_ids = torch.tensor([[0, 581, 10269, 83, 99942, 136, 60742, 23, 70, 80583, 18276, 2]])
        # The dog is cute and lives in the garden house
        expected_output_shape = torch.Size((1, 12, 1024))  # batch_size, sequence_length, embedding_vector_dim
        # fmt: off
        expected_output_values_last_dim = torch.tensor(
            [[-0.0121, -0.0194, -0.0240, -0.0160, -0.0205, -0.0159, -0.0243, -0.0206, -0.0161, -0.0335, -0.0196,
              -0.0141]]
        )
        # fmt: on
        output = model(input_ids)["last_hidden_state"].detach()
        self.assertEqual(output.shape, expected_output_shape)
        # compare the actual values for a slice of last dim
        torch.testing.assert_close(output[:, :, -1], expected_output_values_last_dim, rtol=1e-3, atol=1e-3)

        # language de_DE
        model.set_default_language("de_DE")
        input_ids = torch.tensor([[0, 1310, 49083, 443, 269, 71, 5486, 165, 60429, 660, 23, 2315, 58761, 18391, 5, 2]])
        # Der Hund ist niedlich und wohnt in einem Gartenhaus.
        expected_output_shape = torch.Size((1, 16, 1024))  # batch_size, sequence_length, embedding_vector_dim
        # fmt: off
        expected_output_values_last_dim = torch.tensor(
            [[-0.0120, -0.0262, -0.0253, -0.0112, -0.0128, -0.0164, -0.0080, -0.0081, -0.0192, -0.0117, -0.0170,
              -0.0120, -0.0210, -0.0173, -0.0078, -0.0122]]
        )
        # fmt: on
        output = model(input_ids)["last_hidden_state"].detach()
        self.assertEqual(output.shape, expected_output_shape)
        # compare the actual values for a slice of last dim
        torch.testing.assert_close(output[:, :, -1], expected_output_values_last_dim, rtol=1e-3, atol=1e-3)

    @slow
    def test_multilingual_batch(self):
        model = XmodModel.from_pretrained("facebook/xmod-base")
        # fmt: off
        input_ids = torch.tensor([
            [0, 581, 10269, 83, 99942, 136, 60742, 23, 70, 80583, 18276, 2],
            [0, 1310, 49083, 443, 269, 71, 5486, 165, 60429, 660, 23, 2],
            [0, 1310, 49083, 443, 269, 71, 5486, 165, 60429, 660, 23, 2],
            [0, 581, 10269, 83, 99942, 136, 60742, 23, 70, 80583, 18276, 2],
        ])
        # fmt: on
        lang_ids = torch.LongTensor([0, 8, 8, 0])
        expected_output_shape = torch.Size((4, 12, 768))  # batch_size, sequence_length, embedding_vector_dim
        # fmt: off
        expected_output_values_last_dim = torch.tensor([
            [-0.2394, -0.0036, 0.1252, -0.0087, 0.1325, 0.0580, -0.2049, -0.1978, -0.1223, 0.0648, -0.2599, -0.3724],
            [-0.2668, -0.0235, -0.1739, 0.2266, -0.0901, -0.3482, 0.0105, -0.1915, 0.0397, 0.3822, 0.1836, -0.3407],
            [-0.2668, -0.0235, -0.1739, 0.2266, -0.0901, -0.3482, 0.0105, -0.1915, 0.0397, 0.3822, 0.1836, -0.3407],
            [-0.2394, -0.0036, 0.1252, -0.0087, 0.1325, 0.0580, -0.2049, -0.1978, -0.1223, 0.0648, -0.2599, -0.3724],
        ])
        # fmt: on
        output = model(input_ids, lang_ids=lang_ids)["last_hidden_state"].detach()
        self.assertEqual(output.shape, expected_output_shape)
        # compare the actual values for a slice of last dim
        torch.testing.assert_close(output[:, :, -1], expected_output_values_last_dim, rtol=1e-3, atol=1e-3)

    @slow
    def test_end_to_end_mask_fill(self):
        tokenizer = XLMRobertaTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
        model = XmodForMaskedLM.from_pretrained("facebook/xmod-base", default_language="en_XX")
        model.to(torch_device)

        sentences = [
            "Hello, my dog is a little <mask>.",
            "Hi <mask>!",
        ]

        inputs = tokenizer(sentences, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(torch_device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=inputs["attention_mask"].to(torch_device),
        )
        probs = outputs.logits.softmax(dim=-1)
        _, predictions = probs.topk(1)
        predictions = predictions.squeeze(-1)

        inputs_non_padded = tokenizer(sentences[0], return_tensors="pt").input_ids.to(torch_device)
        output_non_padded = model(input_ids=inputs_non_padded)
        probs_non_padded = output_non_padded.logits.softmax(dim=-1)
        _, predictions_non_padded = probs_non_padded.topk(1)
        predictions_non_padded = predictions_non_padded.squeeze(-1)

        inputs_padded = tokenizer(sentences[1], return_tensors="pt").input_ids.to(torch_device)
        output_padded = model(input_ids=inputs_padded)
        probs_padded = output_padded.logits.softmax(dim=-1)
        _, predictions_padded = probs_padded.topk(1)
        predictions_padded = predictions_padded.squeeze(-1)

        batch_out_sentence = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        non_padded_sentence = tokenizer.decode(predictions_non_padded[0], skip_special_tokens=True)
        padded_sentence = tokenizer.decode(predictions_padded[0], skip_special_tokens=True)

        expected_output_sentence = [
            "Hello, my dog is a little girl.",
            "Hi everyone!",
        ]
        self.assertListEqual(expected_output_sentence, batch_out_sentence)
        self.assertListEqual(batch_out_sentence, [non_padded_sentence, padded_sentence])
