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
import os
import tempfile
import unittest

import pytest

from transformers import DistilBertConfig, is_torch_available
from transformers.testing_utils import require_flash_attn, require_torch, require_torch_accelerator, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        AutoTokenizer,
        DistilBertForMaskedLM,
        DistilBertForMultipleChoice,
        DistilBertForQuestionAnswering,
        DistilBertForSequenceClassification,
        DistilBertForTokenClassification,
        DistilBertModel,
    )
    from transformers.models.distilbert.modeling_distilbert import _create_sinusoidal_embeddings
    from transformers.pytorch_utils import is_torch_greater_or_equal_than_2_4


class DistilBertModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=False,
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

        sequence_labels = None
        token_labels = None
        choice_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = self.get_config()

        return config, input_ids, input_mask, sequence_labels, token_labels, choice_labels

    def get_config(self):
        return DistilBertConfig(
            vocab_size=self.vocab_size,
            dim=self.hidden_size,
            n_layers=self.num_hidden_layers,
            n_heads=self.num_attention_heads,
            hidden_dim=self.intermediate_size,
            hidden_act=self.hidden_act,
            dropout=self.hidden_dropout_prob,
            attention_dropout=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            initializer_range=self.initializer_range,
        )

    def create_and_check_distilbert_model(
        self, config, input_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = DistilBertModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, input_mask)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_distilbert_for_masked_lm(
        self, config, input_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = DistilBertForMaskedLM(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_distilbert_for_question_answering(
        self, config, input_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = DistilBertForQuestionAnswering(config=config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids, attention_mask=input_mask, start_positions=sequence_labels, end_positions=sequence_labels
        )
        self.parent.assertEqual(result.start_logits.shape, (self.batch_size, self.seq_length))
        self.parent.assertEqual(result.end_logits.shape, (self.batch_size, self.seq_length))

    def create_and_check_distilbert_for_sequence_classification(
        self, config, input_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.num_labels = self.num_labels
        model = DistilBertForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=sequence_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def create_and_check_distilbert_for_token_classification(
        self, config, input_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.num_labels = self.num_labels
        model = DistilBertForTokenClassification(config=config)
        model.to(torch_device)
        model.eval()

        result = model(input_ids, attention_mask=input_mask, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.num_labels))

    def create_and_check_distilbert_for_multiple_choice(
        self, config, input_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.num_choices = self.num_choices
        model = DistilBertForMultipleChoice(config=config)
        model.to(torch_device)
        model.eval()
        multiple_choice_inputs_ids = input_ids.unsqueeze(1).expand(-1, self.num_choices, -1).contiguous()
        multiple_choice_input_mask = input_mask.unsqueeze(1).expand(-1, self.num_choices, -1).contiguous()
        result = model(
            multiple_choice_inputs_ids,
            attention_mask=multiple_choice_input_mask,
            labels=choice_labels,
        )
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_choices))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (config, input_ids, input_mask, sequence_labels, token_labels, choice_labels) = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_torch
class DistilBertModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            DistilBertModel,
            DistilBertForMaskedLM,
            DistilBertForMultipleChoice,
            DistilBertForQuestionAnswering,
            DistilBertForSequenceClassification,
            DistilBertForTokenClassification,
        )
        if is_torch_available()
        else None
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": DistilBertModel,
            "fill-mask": DistilBertForMaskedLM,
            "question-answering": DistilBertForQuestionAnswering,
            "text-classification": DistilBertForSequenceClassification,
            "token-classification": DistilBertForTokenClassification,
            "zero-shot": DistilBertForSequenceClassification,
        }
        if is_torch_available()
        else {}
    )
    fx_compatible = True
    test_pruning = True
    test_resize_embeddings = True
    test_resize_position_embeddings = True

    def setUp(self):
        self.model_tester = DistilBertModelTester(self)
        self.config_tester = ConfigTester(self, config_class=DistilBertConfig, dim=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_distilbert_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_distilbert_model(*config_and_inputs)

    def test_distilbert_model_with_sinusoidal_encodings(self):
        config = DistilBertConfig(sinusoidal_pos_embds=True)
        model = DistilBertModel(config=config)
        sinusoidal_pos_embds = torch.empty((config.max_position_embeddings, config.dim), dtype=torch.float32)
        _create_sinusoidal_embeddings(config.max_position_embeddings, config.dim, sinusoidal_pos_embds)
        self.model_tester.parent.assertTrue(
            torch.equal(model.embeddings.position_embeddings.weight, sinusoidal_pos_embds)
        )

    def test_for_masked_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_distilbert_for_masked_lm(*config_and_inputs)

    def test_for_question_answering(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_distilbert_for_question_answering(*config_and_inputs)

    def test_for_sequence_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_distilbert_for_sequence_classification(*config_and_inputs)

    def test_for_token_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_distilbert_for_token_classification(*config_and_inputs)

    def test_for_multiple_choice(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_distilbert_for_multiple_choice(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        model_name = "distilbert-base-uncased"
        model = DistilBertModel.from_pretrained(model_name)
        self.assertIsNotNone(model)

    @slow
    @require_torch_accelerator
    def test_torchscript_device_change(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            # BertForMultipleChoice behaves incorrectly in JIT environments.
            if model_class == DistilBertForMultipleChoice:
                self.skipTest(reason="DistilBertForMultipleChoice behaves incorrectly in JIT environments.")

            config.torchscript = True
            model = model_class(config=config)

            inputs_dict = self._prepare_for_class(inputs_dict, model_class)
            traced_model = torch.jit.trace(
                model, (inputs_dict["input_ids"].to("cpu"), inputs_dict["attention_mask"].to("cpu"))
            )

            with tempfile.TemporaryDirectory() as tmp:
                torch.jit.save(traced_model, os.path.join(tmp, "traced_model.pt"))
                loaded = torch.jit.load(os.path.join(tmp, "traced_model.pt"), map_location=torch_device)
                loaded(inputs_dict["input_ids"].to(torch_device), inputs_dict["attention_mask"].to(torch_device))

    # Because DistilBertForMultipleChoice requires inputs with different shapes we need to override this test.
    @require_flash_attn
    @require_torch_accelerator
    @pytest.mark.flash_attn_test
    @slow
    def test_flash_attn_2_inference_equivalence(self):
        import torch

        for model_class in self.all_model_classes:
            dummy_input = torch.LongTensor(
                [
                    [1, 2, 3, 4],
                    [1, 2, 8, 9],
                    [1, 2, 11, 12],
                    [1, 2, 13, 14],
                ]
            ).to(torch_device)
            dummy_attention_mask = torch.LongTensor(
                [
                    [0, 1, 1, 1],
                    [0, 1, 1, 1],
                    [0, 1, 1, 1],
                    [0, 1, 1, 1],
                ]
            ).to(torch_device)

            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model_fa = model_class.from_pretrained(
                    tmpdirname, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
                )
                model_fa.to(torch_device)

                model = model_class.from_pretrained(tmpdirname, torch_dtype=torch.bfloat16)
                model.to(torch_device)

                logits = model(dummy_input, output_hidden_states=True).hidden_states[-1]
                logits_fa = model_fa(dummy_input, output_hidden_states=True).hidden_states[-1]

                torch.testing.assert_close(logits_fa, logits, rtol=4e-2, atol=4e-2)

                output_fa = model_fa(dummy_input, attention_mask=dummy_attention_mask, output_hidden_states=True)
                logits_fa = output_fa.hidden_states[-1]

                output = model(dummy_input, attention_mask=dummy_attention_mask, output_hidden_states=True)
                logits = output.hidden_states[-1]

                torch.testing.assert_close(logits_fa[1:], logits[1:], rtol=4e-2, atol=4e-2)

    # Because DistilBertForMultipleChoice requires inputs with different shapes we need to override this test.
    @require_flash_attn
    @require_torch_accelerator
    @pytest.mark.flash_attn_test
    @slow
    def test_flash_attn_2_inference_equivalence_right_padding(self):
        import torch

        for model_class in self.all_model_classes:
            dummy_input = torch.LongTensor(
                [
                    [1, 2, 3, 4],
                    [1, 2, 8, 9],
                    [1, 2, 11, 12],
                    [1, 2, 13, 14],
                ]
            ).to(torch_device)
            dummy_attention_mask = torch.LongTensor(
                [
                    [0, 1, 1, 1],
                    [0, 1, 1, 1],
                    [0, 1, 1, 1],
                    [0, 1, 1, 1],
                ]
            ).to(torch_device)

            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model_fa = model_class.from_pretrained(
                    tmpdirname, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
                )
                model_fa.to(torch_device)

                model = model_class.from_pretrained(
                    tmpdirname,
                    torch_dtype=torch.bfloat16,
                )
                model.to(torch_device)

                logits = model(dummy_input, output_hidden_states=True).hidden_states[-1]
                logits_fa = model_fa(dummy_input, output_hidden_states=True).hidden_states[-1]

                torch.testing.assert_close(logits_fa, logits, rtol=4e-2, atol=4e-2)

                output_fa = model_fa(dummy_input, attention_mask=dummy_attention_mask, output_hidden_states=True)
                logits_fa = output_fa.hidden_states[-1]

                output = model(dummy_input, attention_mask=dummy_attention_mask, output_hidden_states=True)
                logits = output.hidden_states[-1]

                torch.testing.assert_close(logits_fa[:-1], logits[:-1], rtol=4e-2, atol=4e-2)


@require_torch
class DistilBertModelIntergrationTest(unittest.TestCase):
    @slow
    def test_inference_no_head_absolute_embedding(self):
        model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        input_ids = torch.tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])
        attention_mask = torch.tensor([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        with torch.no_grad():
            output = model(input_ids, attention_mask=attention_mask)[0]
        expected_shape = torch.Size((1, 11, 768))
        self.assertEqual(output.shape, expected_shape)
        expected_slice = torch.tensor(
            [[[-0.1639, 0.3299, 0.1648], [-0.1746, 0.3289, 0.1710], [-0.1884, 0.3357, 0.1810]]]
        )

        torch.testing.assert_close(output[:, 1:4, 1:4], expected_slice, rtol=1e-4, atol=1e-4)

    @slow
    def test_export(self):
        if not is_torch_greater_or_equal_than_2_4:
            self.skipTest(reason="This test requires torch >= 2.4 to run.")

        distilbert_model = "distilbert-base-uncased"
        device = "cpu"
        attn_implementation = "sdpa"
        max_length = 64

        tokenizer = AutoTokenizer.from_pretrained(distilbert_model)
        inputs = tokenizer(
            f"Paris is the {tokenizer.mask_token} of France.",
            return_tensors="pt",
            padding="max_length",
            max_length=max_length,
        )

        model = DistilBertForMaskedLM.from_pretrained(
            distilbert_model,
            device_map=device,
            attn_implementation=attn_implementation,
        )

        logits = model(**inputs).logits
        eager_predicted_mask = tokenizer.decode(logits[0, 4].topk(5).indices)
        self.assertEqual(
            eager_predicted_mask.split(),
            ["capital", "birthplace", "northernmost", "centre", "southernmost"],
        )

        exported_program = torch.export.export(
            model,
            args=(inputs["input_ids"],),
            kwargs={"attention_mask": inputs["attention_mask"]},
            strict=True,
        )

        result = exported_program.module().forward(inputs["input_ids"], inputs["attention_mask"])
        exported_predicted_mask = tokenizer.decode(result.logits[0, 4].topk(5).indices)
        self.assertEqual(eager_predicted_mask, exported_predicted_mask)
