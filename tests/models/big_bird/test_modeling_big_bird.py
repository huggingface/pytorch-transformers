# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch BigBird model."""

import unittest

from transformers import BigBirdConfig, is_torch_available
from transformers.models.auto import get_values
from transformers.models.big_bird.tokenization_big_bird import BigBirdTokenizer
from transformers.testing_utils import require_torch, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        MODEL_FOR_PRETRAINING_MAPPING,
        BigBirdForCausalLM,
        BigBirdForMaskedLM,
        BigBirdForMultipleChoice,
        BigBirdForPreTraining,
        BigBirdForQuestionAnswering,
        BigBirdForSequenceClassification,
        BigBirdForTokenClassification,
        BigBirdModel,
    )


class BigBirdModelTester:
    def __init__(
        self,
        parent,
        batch_size=7,
        seq_length=128,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=True,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu_new",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=256,
        type_vocab_size=16,
        type_sequence_label_size=2,
        initializer_range=0.02,
        num_labels=3,
        num_choices=4,
        attention_type="block_sparse",
        use_bias=True,
        rescale_embeddings=False,
        block_size=8,
        num_rand_blocks=3,
        position_embedding_type="absolute",
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

        self.attention_type = attention_type
        self.use_bias = use_bias
        self.rescale_embeddings = rescale_embeddings
        self.block_size = block_size
        self.num_rand_blocks = num_rand_blocks
        self.position_embedding_type = position_embedding_type

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
        return BigBirdConfig(
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
            is_encoder_decoder=False,
            initializer_range=self.initializer_range,
            attention_type=self.attention_type,
            use_bias=self.use_bias,
            rescale_embeddings=self.rescale_embeddings,
            block_size=self.block_size,
            num_random_blocks=self.num_rand_blocks,
            position_embedding_type=self.position_embedding_type,
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
        model = BigBirdModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)
        result = model(input_ids, token_type_ids=token_type_ids)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_for_pretraining(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = BigBirdForPreTraining(config=config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            labels=token_labels,
            next_sentence_label=sequence_labels,
        )
        self.parent.assertEqual(result.prediction_logits.shape, (self.batch_size, self.seq_length, self.vocab_size))
        self.parent.assertEqual(result.seq_relationship_logits.shape, (self.batch_size, config.num_labels))

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
        model = BigBirdModel(config)
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
        model = BigBirdForCausalLM(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_for_masked_lm(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = BigBirdForMaskedLM(config=config)
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
        model = BigBirdForCausalLM(config=config)
        model.to(torch_device)
        model.eval()

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

    def create_and_check_for_question_answering(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = BigBirdForQuestionAnswering(config=config)
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

    def create_and_check_for_sequence_classification(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.num_labels = self.num_labels
        model = BigBirdForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, labels=sequence_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def create_and_check_for_token_classification(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.num_labels = self.num_labels
        model = BigBirdForTokenClassification(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.num_labels))

    def create_and_check_for_multiple_choice(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.num_choices = self.num_choices
        model = BigBirdForMultipleChoice(config=config)
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

    def create_and_check_for_auto_padding(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        model = BigBirdModel(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_for_change_to_full_attn(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        model = BigBirdModel(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))
        # the config should not be changed
        self.parent.assertTrue(model.config.attention_type == "block_sparse")


@require_torch
class BigBirdModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    # head masking & pruning is currently not supported for big bird
    test_head_masking = False
    test_pruning = False

    # torchscript should be possible, but takes prohibitively long to test.
    # Also torchscript is not an important feature to have in the beginning.
    test_torchscript = False

    all_model_classes = (
        (
            BigBirdModel,
            BigBirdForPreTraining,
            BigBirdForMaskedLM,
            BigBirdForCausalLM,
            BigBirdForMultipleChoice,
            BigBirdForQuestionAnswering,
            BigBirdForSequenceClassification,
            BigBirdForTokenClassification,
        )
        if is_torch_available()
        else ()
    )
    all_generative_model_classes = (BigBirdForCausalLM,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": BigBirdModel,
            "fill-mask": BigBirdForMaskedLM,
            "question-answering": BigBirdForQuestionAnswering,
            "text-classification": BigBirdForSequenceClassification,
            "text-generation": BigBirdForCausalLM,
            "token-classification": BigBirdForTokenClassification,
            "zero-shot": BigBirdForSequenceClassification,
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
                inputs_dict["next_sentence_label"] = torch.zeros(
                    self.model_tester.batch_size, dtype=torch.long, device=torch_device
                )
        return inputs_dict

    def setUp(self):
        self.model_tester = BigBirdModelTester(self)
        self.config_tester = ConfigTester(self, config_class=BigBirdConfig, hidden_size=37)

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

    def test_for_multiple_choice(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_multiple_choice(*config_and_inputs)

    def test_decoder_model_past_with_large_inputs(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_decoder()
        self.model_tester.create_and_check_decoder_model_past_large_inputs(*config_and_inputs)

    def test_for_question_answering(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_question_answering(*config_and_inputs)

    def test_for_sequence_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_sequence_classification(*config_and_inputs)

    def test_for_token_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_token_classification(*config_and_inputs)

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

    def test_retain_grad_hidden_states_attentions(self):
        # bigbird cannot keep gradients in attentions when `attention_type=block_sparse`

        if self.model_tester.attention_type == "original_full":
            super().test_retain_grad_hidden_states_attentions()

    @slow
    def test_model_from_pretrained(self):
        model_name = "google/bigbird-roberta-base"
        model = BigBirdForPreTraining.from_pretrained(model_name)
        self.assertIsNotNone(model)

    def test_model_various_attn_type(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        for type in ["original_full", "block_sparse"]:
            config_and_inputs[0].attention_type = type
            self.model_tester.create_and_check_model(*config_and_inputs)

    def test_fast_integration(self):
        # fmt: off
        input_ids = torch.tensor(
            [[6, 117, 33, 36, 70, 22, 63, 31, 71, 72, 88, 58, 109, 49, 48, 116, 92, 6, 19, 95, 118, 100, 80, 111, 93, 2, 31, 84, 26, 5, 6, 82, 46, 96, 109, 4, 39, 19, 109, 13, 92, 31, 36, 90, 111, 18, 75, 6, 56, 74, 16, 42, 56, 92, 69, 108, 127, 81, 82, 41, 106, 19, 44, 24, 82, 121, 120, 65, 36, 26, 72, 13, 36, 98, 43, 64, 8, 53, 100, 92, 51, 122, 66, 17, 61, 50, 104, 127, 26, 35, 94, 23, 110, 71, 80, 67, 109, 111, 44, 19, 51, 41, 86, 71, 76, 44, 18, 68, 44, 77, 107, 81, 98, 126, 100, 2, 49, 98, 84, 39, 23, 98, 52, 46, 10, 82, 121, 73],[6, 117, 33, 36, 70, 22, 63, 31, 71, 72, 88, 58, 109, 49, 48, 116, 92, 6, 19, 95, 118, 100, 80, 111, 93, 2, 31, 84, 26, 5, 6, 82, 46, 96, 109, 4, 39, 19, 109, 13, 92, 31, 36, 90, 111, 18, 75, 6, 56, 74, 16, 42, 56, 92, 69, 108, 127, 81, 82, 41, 106, 19, 44, 24, 82, 121, 120, 65, 36, 26, 72, 13, 36, 98, 43, 64, 8, 53, 100, 92, 51, 12, 66, 17, 61, 50, 104, 127, 26, 35, 94, 23, 110, 71, 80, 67, 109, 111, 44, 19, 51, 41, 86, 71, 76, 28, 18, 68, 44, 77, 107, 81, 98, 126, 100, 2, 49, 18, 84, 39, 23, 98, 52, 46, 10, 82, 121, 73]],  # noqa: E231
            dtype=torch.long,
            device=torch_device,
        )
        # fmt: on
        input_ids = input_ids % self.model_tester.vocab_size
        input_ids[1] = input_ids[1] - 1

        attention_mask = torch.ones((input_ids.shape), device=torch_device)
        attention_mask[:, :-10] = 0

        config, _, _, _, _, _, _ = self.model_tester.prepare_config_and_inputs()
        torch.manual_seed(0)
        model = BigBirdModel(config).eval().to(torch_device)

        with torch.no_grad():
            hidden_states = model(input_ids, attention_mask=attention_mask).last_hidden_state
            self.assertTrue(
                torch.allclose(
                    hidden_states[0, 0, :5],
                    torch.tensor([1.4825, 0.0774, 0.8226, -0.2962, -0.9593], device=torch_device),
                    atol=1e-3,
                )
            )

    def test_auto_padding(self):
        self.model_tester.seq_length = 241
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_auto_padding(*config_and_inputs)

    def test_for_change_to_full_attn(self):
        self.model_tester.seq_length = 9
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_change_to_full_attn(*config_and_inputs)

    @unittest.skip(
        reason="This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing(self):
        pass

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

    # overwrite from common in order to skip the check on `attentions`
    def check_pt_flax_outputs(self, fx_outputs, pt_outputs, model_class, tol=1e-5, name="outputs", attributes=None):
        # `bigbird_block_sparse_attention` in `FlaxBigBird` returns `attention_probs = None`, while in PyTorch version,
        # an effort was done to return `attention_probs` (yet to be verified).
        if name.startswith("outputs.attentions"):
            return
        else:
            super().check_pt_flax_outputs(fx_outputs, pt_outputs, model_class, tol, name, attributes)


@require_torch
@slow
class BigBirdModelIntegrationTest(unittest.TestCase):
    # we can have this true once block_sparse attn_probs works accurately
    test_attention_probs = False

    def _get_dummy_input_ids(self):
        # fmt: off
        ids = torch.tensor(
            [[6, 117, 33, 36, 70, 22, 63, 31, 71, 72, 88, 58, 109, 49, 48, 116, 92, 6, 19, 95, 118, 100, 80, 111, 93, 2, 31, 84, 26, 5, 6, 82, 46, 96, 109, 4, 39, 19, 109, 13, 92, 31, 36, 90, 111, 18, 75, 6, 56, 74, 16, 42, 56, 92, 69, 108, 127, 81, 82, 41, 106, 19, 44, 24, 82, 121, 120, 65, 36, 26, 72, 13, 36, 98, 43, 64, 8, 53, 100, 92, 51, 122, 66, 17, 61, 50, 104, 127, 26, 35, 94, 23, 110, 71, 80, 67, 109, 111, 44, 19, 51, 41, 86, 71, 76, 44, 18, 68, 44, 77, 107, 81, 98, 126, 100, 2, 49, 98, 84, 39, 23, 98, 52, 46, 10, 82, 121, 73]],  # noqa: E231
            dtype=torch.long,
            device=torch_device,
        )
        # fmt: on
        return ids

    def test_inference_block_sparse_pretraining(self):
        model = BigBirdForPreTraining.from_pretrained("google/bigbird-roberta-base", attention_type="block_sparse")
        model.to(torch_device)

        input_ids = torch.tensor([[20920, 232, 328, 1437] * 1024], dtype=torch.long, device=torch_device)
        with torch.no_grad():
            outputs = model(input_ids)
        prediction_logits = outputs.prediction_logits
        seq_relationship_logits = outputs.seq_relationship_logits

        self.assertEqual(prediction_logits.shape, torch.Size((1, 4096, 50358)))
        self.assertEqual(seq_relationship_logits.shape, torch.Size((1, 2)))

        expected_prediction_logits_slice = torch.tensor(
            [
                [-0.5583, 0.0475, -0.2508, 7.4423],
                [0.7409, 1.4460, -0.7593, 7.7010],
                [1.9150, 3.1395, 5.8840, 9.3498],
                [-0.1854, -1.4640, -2.2052, 3.7968],
            ],
            device=torch_device,
        )

        torch.testing.assert_close(
            prediction_logits[0, 128:132, 128:132], expected_prediction_logits_slice, rtol=1e-4, atol=1e-4
        )

        expected_seq_relationship_logits = torch.tensor([[46.9465, 47.9517]], device=torch_device)
        torch.testing.assert_close(seq_relationship_logits, expected_seq_relationship_logits, rtol=1e-4, atol=1e-4)

    def test_inference_full_pretraining(self):
        model = BigBirdForPreTraining.from_pretrained("google/bigbird-roberta-base", attention_type="original_full")
        model.to(torch_device)

        input_ids = torch.tensor([[20920, 232, 328, 1437] * 512], dtype=torch.long, device=torch_device)
        with torch.no_grad():
            outputs = model(input_ids)
        prediction_logits = outputs.prediction_logits
        seq_relationship_logits = outputs.seq_relationship_logits

        self.assertEqual(prediction_logits.shape, torch.Size((1, 512 * 4, 50358)))
        self.assertEqual(seq_relationship_logits.shape, torch.Size((1, 2)))

        expected_prediction_logits_slice = torch.tensor(
            [
                [0.1499, -1.1217, 0.1990, 8.4499],
                [-2.7757, -3.0687, -4.8577, 7.5156],
                [1.5446, 0.1982, 4.3016, 10.4281],
                [-1.3705, -4.0130, -3.9629, 5.1526],
            ],
            device=torch_device,
        )
        torch.testing.assert_close(
            prediction_logits[0, 128:132, 128:132], expected_prediction_logits_slice, rtol=1e-4, atol=1e-4
        )

        expected_seq_relationship_logits = torch.tensor([[41.4503, 41.2406]], device=torch_device)
        torch.testing.assert_close(seq_relationship_logits, expected_seq_relationship_logits, rtol=1e-4, atol=1e-4)

    def test_block_sparse_attention_probs(self):
        """
        Asserting if outputted attention matrix is similar to hard coded attention matrix
        """

        if not self.test_attention_probs:
            self.skipTest("test_attention_probs is set to False")

        model = BigBirdModel.from_pretrained(
            "google/bigbird-roberta-base", attention_type="block_sparse", num_random_blocks=3, block_size=16
        )
        model.to(torch_device)
        model.eval()
        config = model.config

        input_ids = self._get_dummy_input_ids()

        hidden_states = model.embeddings(input_ids)

        batch_size, seqlen, _ = hidden_states.size()
        attn_mask = torch.ones(batch_size, seqlen, device=torch_device, dtype=torch.float)
        to_seq_length = from_seq_length = seqlen
        from_block_size = to_block_size = config.block_size

        blocked_mask, band_mask, from_mask, to_mask = model.create_masks_for_block_sparse_attn(
            attn_mask, config.block_size
        )
        from_blocked_mask = to_blocked_mask = blocked_mask

        for i in range(config.num_hidden_layers):
            pointer = model.encoder.layer[i].attention.self

            query_layer = pointer.transpose_for_scores(pointer.query(hidden_states))
            key_layer = pointer.transpose_for_scores(pointer.key(hidden_states))
            value_layer = pointer.transpose_for_scores(pointer.value(hidden_states))

            context_layer, attention_probs = pointer.bigbird_block_sparse_attention(
                query_layer,
                key_layer,
                value_layer,
                band_mask,
                from_mask,
                to_mask,
                from_blocked_mask,
                to_blocked_mask,
                pointer.num_attention_heads,
                pointer.num_random_blocks,
                pointer.attention_head_size,
                from_block_size,
                to_block_size,
                batch_size,
                from_seq_length,
                to_seq_length,
                seed=pointer.seed,
                plan_from_length=None,
                plan_num_rand_blocks=None,
                output_attentions=True,
            )

            context_layer = context_layer.contiguous().view(batch_size, from_seq_length, -1)
            cl = torch.einsum("bhqk,bhkd->bhqd", attention_probs, value_layer)
            cl = cl.view(context_layer.size())

            torch.testing.assert_close(context_layer, cl, rtol=0.001, atol=0.001)

    def test_block_sparse_context_layer(self):
        model = BigBirdModel.from_pretrained(
            "google/bigbird-roberta-base", attention_type="block_sparse", num_random_blocks=3, block_size=16
        )
        model.to(torch_device)
        model.eval()
        config = model.config

        input_ids = self._get_dummy_input_ids()
        dummy_hidden_states = model.embeddings(input_ids)

        attn_mask = torch.ones_like(input_ids, device=torch_device)
        blocked_mask, band_mask, from_mask, to_mask = model.create_masks_for_block_sparse_attn(
            attn_mask, config.block_size
        )

        targeted_cl = torch.tensor(
            [
                [0.1870, 1.5248, 0.2333, -0.0483, -0.0952, 1.8359, -0.0142, 0.1239, 0.0083, -0.0045],
                [-0.0601, 0.1243, 0.1329, -0.1524, 0.2347, 0.0894, -0.2248, -0.2461, -0.0645, -0.0109],
                [-0.0418, 0.1463, 0.1290, -0.1638, 0.2489, 0.0799, -0.2341, -0.2406, -0.0524, 0.0106],
                [0.1859, 1.5182, 0.2324, -0.0473, -0.0952, 1.8295, -0.0148, 0.1242, 0.0080, -0.0045],
                [0.1879, 1.5300, 0.2334, -0.0480, -0.0967, 1.8428, -0.0137, 0.1256, 0.0087, -0.0050],
                [0.1852, 1.5149, 0.2330, -0.0492, -0.0936, 1.8236, -0.0154, 0.1210, 0.0080, -0.0048],
                [0.1857, 1.5186, 0.2331, -0.0484, -0.0940, 1.8285, -0.0148, 0.1224, 0.0077, -0.0045],
                [0.1884, 1.5336, 0.2334, -0.0469, -0.0974, 1.8477, -0.0132, 0.1266, 0.0085, -0.0046],
                [0.1881, 1.5308, 0.2334, -0.0479, -0.0969, 1.8438, -0.0136, 0.1258, 0.0088, -0.0050],
                [0.1849, 1.5143, 0.2329, -0.0491, -0.0930, 1.8230, -0.0156, 0.1209, 0.0074, -0.0047],
                [0.1878, 1.5299, 0.2333, -0.0472, -0.0967, 1.8434, -0.0137, 0.1257, 0.0084, -0.0048],
                [0.1873, 1.5260, 0.2333, -0.0478, -0.0961, 1.8383, -0.0142, 0.1245, 0.0083, -0.0048],
                [0.1849, 1.5145, 0.2327, -0.0491, -0.0935, 1.8237, -0.0156, 0.1215, 0.0083, -0.0046],
                [0.1866, 1.5232, 0.2332, -0.0488, -0.0950, 1.8342, -0.0143, 0.1237, 0.0084, -0.0047],
            ],
            device=torch_device,
        )

        context_layer = model.encoder.layer[0].attention.self(
            dummy_hidden_states,
            band_mask=band_mask,
            from_mask=from_mask,
            to_mask=to_mask,
            from_blocked_mask=blocked_mask,
            to_blocked_mask=blocked_mask,
        )
        context_layer = context_layer[0]

        self.assertEqual(context_layer.shape, torch.Size((1, 128, 768)))
        torch.testing.assert_close(context_layer[0, 64:78, 300:310], targeted_cl, rtol=0.0001, atol=0.0001)

    def test_tokenizer_inference(self):
        tokenizer = BigBirdTokenizer.from_pretrained("google/bigbird-roberta-base")
        model = BigBirdModel.from_pretrained(
            "google/bigbird-roberta-base", attention_type="block_sparse", num_random_blocks=3, block_size=16
        )
        model.to(torch_device)

        text = [
            "Transformer-based models are unable to process long sequences due to their self-attention operation,"
            " which scales quadratically with the sequence length. To address this limitation, we introduce the"
            " Longformer with an attention mechanism that scales linearly with sequence length, making it easy to"
            " process documents of thousands of tokens or longer. Longformer’s attention mechanism is a drop-in"
            " replacement for the standard self-attention and combines a local windowed attention with a task"
            " motivated global attention. Following prior work on long-sequence transformers, we evaluate Longformer"
            " on character-level language modeling and achieve state-of-the-art results on text8 and enwik8. In"
            " contrast to most prior work, we also pretrain Longformer and finetune it on a variety of downstream"
            " tasks. Our pretrained Longformer consistently outperforms RoBERTa on long document tasks and sets new"
            " state-of-the-art results on WikiHop and TriviaQA."
        ]
        inputs = tokenizer(text)

        for k in inputs:
            inputs[k] = torch.tensor(inputs[k], device=torch_device, dtype=torch.long)

        prediction = model(**inputs)
        prediction = prediction[0]

        self.assertEqual(prediction.shape, torch.Size((1, 199, 768)))

        expected_prediction = torch.tensor(
            [
                [0.1887, -0.0474, 0.2604, 0.1453],
                [0.0651, 0.1999, 0.1797, 0.1161],
                [0.2833, -0.3036, 0.6910, 0.1123],
                [0.2836, -0.4644, -0.0111, 0.1530],
                [0.3919, -0.2823, 0.4192, 0.1687],
                [0.2168, -0.1956, 0.4050, 0.0925],
                [0.2597, -0.0884, 0.1258, 0.1119],
                [0.1127, -0.1203, 0.1924, 0.2859],
                [0.1362, -0.1315, 0.2693, 0.1027],
                [-0.3169, -0.2266, 0.4419, 0.6740],
                [0.2366, -0.1452, 0.2589, 0.0579],
                [0.0358, -0.2021, 0.3112, -0.1392],
            ],
            device=torch_device,
        )

        torch.testing.assert_close(prediction[0, 52:64, 320:324], expected_prediction, rtol=1e-4, atol=1e-4)

    def test_inference_question_answering(self):
        tokenizer = BigBirdTokenizer.from_pretrained("google/bigbird-base-trivia-itc")
        model = BigBirdForQuestionAnswering.from_pretrained(
            "google/bigbird-base-trivia-itc", attention_type="block_sparse", block_size=16, num_random_blocks=3
        )
        model.to(torch_device)

        context = (
            "The BigBird model was proposed in Big Bird: Transformers for Longer Sequences by Zaheer, Manzil and"
            " Guruganesh, Guru and Dubey, Kumar Avinava and Ainslie, Joshua and Alberti, Chris and Ontanon, Santiago"
            " and Pham, Philip and Ravula, Anirudh and Wang, Qifan and Yang, Li and others. BigBird, is a"
            " sparse-attention based transformer which extends Transformer based models, such as BERT to much longer"
            " sequences. In addition to sparse attention, BigBird also applies global attention as well as random"
            " attention to the input sequence. Theoretically, it has been shown that applying sparse, global, and"
            " random attention approximates full attention, while being computationally much more efficient for longer"
            " sequences. As a consequence of the capability to handle longer context, BigBird has shown improved"
            " performance on various long document NLP tasks, such as question answering and summarization, compared"
            " to BERT or RoBERTa."
        )

        question = [
            "Which is better for longer sequences- BigBird or BERT?",
            "What is the benefit of using BigBird over BERT?",
        ]
        inputs = tokenizer(
            question,
            [context, context],
            padding=True,
            return_tensors="pt",
            add_special_tokens=True,
            max_length=256,
            truncation=True,
        )

        inputs = {k: v.to(torch_device) for k, v in inputs.items()}

        start_logits, end_logits = model(**inputs).to_tuple()

        # fmt: off
        target_start_logits = torch.tensor(
            [[-8.5622, -9.6209, -14.3351, -8.7032, -11.8596, -7.7446, -9.6730, -13.6063, -8.9651, -11.7417, -8.2641, -8.7056, -13.4116, -5.6600, -8.8316, -10.4148, -12.2180, -7.7979, -12.5274, -6.0685, -10.3373, -11.3128, -6.6456, -14.4030, -6.8292, -14.5383, -11.5638, -6.3326, 11.5293, -1.8434, -10.0013, -7.6150], [-10.7384, -13.1179, -10.1837, -13.7700, -10.0186, -11.7335, -13.3411, -10.0188, -13.4235, -9.9381, -10.4252, -13.1281, -8.2022, -10.4326, -11.5542, -14.1549, -10.7546, -13.4691, -8.2744, -11.4324, -13.3773, -9.8284, -14.5825, -8.7471, -14.7050, -8.0364, -11.3627, -6.4638, -11.7031, -14.3446, -9.9425, -8.0088]], # noqa: E231
            device=torch_device,
        )

        target_end_logits = torch.tensor(
            [[-12.1736, -8.8487, -14.8877, -11.6713, -15.1165, -12.2396, -7.6828, -15.4153, -12.2528, -14.3671, -12.3596, -7.4272, -14.9615, -13.6356, -11.7939, -9.9767, -14.8112, -8.9567, -15.8798, -11.5291, -9.4249, -14.7544, -7.9387, -16.2789, -8.9702, -15.3111, -11.5585, -7.9992, -4.1127, 10.3209, -8.3926, -10.2005], [-11.1375, -15.4027, -12.6861, -16.9884, -13.7093, -10.3560, -15.7228, -12.9290, -15.8519, -13.7953, -10.2460, -15.7198, -14.2078, -12.8477, -11.4861, -16.1017, -11.8900, -16.4488, -13.2959, -10.3980, -15.4874, -10.3539, -16.8263, -10.9973, -17.0344, -9.2751, -10.1196, -13.8907, -12.1025, -13.0628, -12.8530, -13.8173]], # noqa: E321
            device=torch_device,
        )
        # fmt: on

        torch.testing.assert_close(start_logits[:, 64:96], target_start_logits, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(end_logits[:, 64:96], target_end_logits, rtol=1e-4, atol=1e-4)

        input_ids = inputs["input_ids"].tolist()
        answer = [
            input_ids[i][torch.argmax(start_logits, dim=-1)[i] : torch.argmax(end_logits, dim=-1)[i] + 1]
            for i in range(len(input_ids))
        ]
        answer = tokenizer.batch_decode(answer)

        self.assertTrue(answer == ["BigBird", "global attention"])

    def test_fill_mask(self):
        tokenizer = BigBirdTokenizer.from_pretrained("google/bigbird-roberta-base")
        model = BigBirdForMaskedLM.from_pretrained("google/bigbird-roberta-base")
        model.to(torch_device)

        input_ids = tokenizer("The goal of life is [MASK] .", return_tensors="pt").input_ids.to(torch_device)
        logits = model(input_ids).logits

        # [MASK] is token at 6th position
        pred_token = tokenizer.decode(torch.argmax(logits[0, 6:7], axis=-1))
        self.assertEqual(pred_token, "happiness")

    def test_auto_padding(self):
        model = BigBirdModel.from_pretrained(
            "google/bigbird-roberta-base", attention_type="block_sparse", num_random_blocks=3, block_size=16
        )
        model.to(torch_device)
        model.eval()

        input_ids = torch.tensor([200 * [10] + 40 * [2] + [1]], device=torch_device, dtype=torch.long)
        with torch.no_grad():
            output = model(input_ids).to_tuple()[0]

        # fmt: off
        target = torch.tensor(
            [[-0.129420, -0.164740, 0.042422, -0.336030, 0.094379, 0.033794, 0.384590, 0.229660, -0.196500, 0.108020], [-0.000154, -0.168800, 0.165820, -0.313670, 0.101240, 0.035145, 0.381880, 0.213730, -0.201080, 0.077443], [0.053754, -0.166350, 0.225520, -0.272900, 0.119670, 0.019987, 0.348670, 0.199190, -0.181600, 0.084640], [0.063636, -0.187110, 0.237010, -0.297380, 0.126300, 0.020025, 0.268490, 0.191820, -0.192300, 0.035077], [0.073893, -0.184790, 0.188870, -0.297860, 0.134280, 0.028972, 0.174650, 0.186890, -0.180530, 0.006851], [0.005253, -0.169360, 0.123100, -0.302550, 0.126930, 0.024188, 0.133410, 0.200600, -0.168210, -0.001006], [-0.093336, -0.175370, -0.004768, -0.333170, 0.114330, 0.034168, 0.120960, 0.203570, -0.162810, -0.005757], [-0.160210, -0.169310, -0.049064, -0.331950, 0.115730, 0.027062, 0.143600, 0.205310, -0.144580, 0.026746], [-0.193200, -0.156820, -0.079422, -0.351600, 0.106450, 0.032174, 0.245690, 0.210250, -0.173480, 0.043914], [-0.167980, -0.153050, -0.059764, -0.357890,0.103910, 0.031481, 0.334190, 0.208960,-0.178180, 0.072165], [-0.136990, -0.156950, -0.012099, -0.353140,0.096996, 0.025864, 0.376340, 0.216050, -0.171820, 0.089963], [-0.041143, -0.167060, 0.079754, -0.353220, 0.093247, 0.019867, 0.385810, 0.214340, -0.191800, 0.065946],[0.040373, -0.158610, 0.152570, -0.312930, 0.110590, 0.012282, 0.345270, 0.204040, -0.176500, 0.064972], [0.043762, -0.166450, 0.179500, -0.317930, 0.117280, -0.004040, 0.304490, 0.201380, -0.182780, 0.044000]], # noqa: E231
            device=torch_device,
        )
        # fmt: on

        self.assertEqual(output.shape, torch.Size((1, 241, 768)))
        torch.testing.assert_close(output[0, 64:78, 300:310], target, rtol=0.0001, atol=0.0001)
