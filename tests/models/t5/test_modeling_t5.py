# coding=utf-8
# Copyright 2018 Google T5 Authors and HuggingFace Inc. team.
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
import os
import pickle
import tempfile
import unittest

from transformers import T5Config, is_torch_available
from transformers.models.auto.modeling_auto import MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES
from transformers.testing_utils import (
    require_accelerate,
    require_sentencepiece,
    require_tokenizers,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)
from transformers.utils import cached_property, is_torch_fx_available

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, _config_zero_init, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_fx_available():
    from transformers.utils.fx import symbolic_trace


if is_torch_available():
    import torch
    import torch.nn.functional as F

    from transformers import (
        AutoTokenizer,
        ByT5Tokenizer,
        T5EncoderModel,
        T5ForConditionalGeneration,
        T5ForQuestionAnswering,
        T5ForSequenceClassification,
        T5ForTokenClassification,
        T5Model,
        T5Tokenizer,
    )


class T5ModelTester:
    def __init__(
        self,
        parent,
        vocab_size=99,
        batch_size=13,
        encoder_seq_length=7,
        decoder_seq_length=7,
        # For common tests
        is_training=True,
        use_attention_mask=True,
        use_labels=True,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        d_ff=37,
        relative_attention_num_buckets=8,
        dropout_rate=0.1,
        initializer_factor=0.002,
        eos_token_id=1,
        pad_token_id=0,
        decoder_start_token_id=0,
        scope=None,
        decoder_layers=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.encoder_seq_length = encoder_seq_length
        self.decoder_seq_length = decoder_seq_length
        # For common tests
        self.seq_length = self.decoder_seq_length
        self.is_training = is_training
        self.use_attention_mask = use_attention_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.d_ff = d_ff
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.dropout_rate = dropout_rate
        self.initializer_factor = initializer_factor
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.scope = None
        self.decoder_layers = decoder_layers

    def get_large_model_config(self):
        return T5Config.from_pretrained("google-t5/t5-base")

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.encoder_seq_length], self.vocab_size).clamp(2)
        input_ids[:, -1] = self.eos_token_id  # Eos Token
        decoder_input_ids = ids_tensor([self.batch_size, self.decoder_seq_length], self.vocab_size)

        attention_mask = None
        decoder_attention_mask = None
        if self.use_attention_mask:
            attention_mask = ids_tensor([self.batch_size, self.encoder_seq_length], vocab_size=2)
            decoder_attention_mask = ids_tensor([self.batch_size, self.decoder_seq_length], vocab_size=2)

        lm_labels = None
        if self.use_labels:
            lm_labels = ids_tensor([self.batch_size, self.decoder_seq_length], self.vocab_size)

        config = self.get_config()

        return (
            config,
            input_ids,
            decoder_input_ids,
            attention_mask,
            decoder_attention_mask,
            lm_labels,
        )

    def get_pipeline_config(self):
        return T5Config(
            vocab_size=166,  # t5 forces 100 extra tokens
            d_model=self.hidden_size,
            d_ff=self.d_ff,
            d_kv=self.hidden_size // self.num_attention_heads,
            num_layers=self.num_hidden_layers,
            num_decoder_layers=self.decoder_layers,
            num_heads=self.num_attention_heads,
            relative_attention_num_buckets=self.relative_attention_num_buckets,
            dropout_rate=self.dropout_rate,
            initializer_factor=self.initializer_factor,
            eos_token_id=self.eos_token_id,
            bos_token_id=self.pad_token_id,
            pad_token_id=self.pad_token_id,
            decoder_start_token_id=self.decoder_start_token_id,
        )

    def get_config(self):
        return T5Config(
            vocab_size=self.vocab_size,
            d_model=self.hidden_size,
            d_ff=self.d_ff,
            d_kv=self.hidden_size // self.num_attention_heads,
            num_layers=self.num_hidden_layers,
            num_decoder_layers=self.decoder_layers,
            num_heads=self.num_attention_heads,
            relative_attention_num_buckets=self.relative_attention_num_buckets,
            dropout_rate=self.dropout_rate,
            initializer_factor=self.initializer_factor,
            eos_token_id=self.eos_token_id,
            bos_token_id=self.pad_token_id,
            pad_token_id=self.pad_token_id,
            decoder_start_token_id=self.decoder_start_token_id,
        )

    def check_prepare_lm_labels_via_shift_left(
        self,
        config,
        input_ids,
        decoder_input_ids,
        attention_mask,
        decoder_attention_mask,
        lm_labels,
    ):
        model = T5Model(config=config)
        model.to(torch_device)
        model.eval()

        # make sure that lm_labels are correctly padded from the right
        lm_labels.masked_fill_((lm_labels == self.decoder_start_token_id), self.eos_token_id)

        # add casaul pad token mask
        triangular_mask = torch.tril(lm_labels.new_ones(lm_labels.shape)).logical_not()
        lm_labels.masked_fill_(triangular_mask, self.pad_token_id)
        decoder_input_ids = model._shift_right(lm_labels)

        for i, (decoder_input_ids_slice, lm_labels_slice) in enumerate(zip(decoder_input_ids, lm_labels)):
            # first item
            self.parent.assertEqual(decoder_input_ids_slice[0].item(), self.decoder_start_token_id)
            if i < decoder_input_ids_slice.shape[-1]:
                if i < decoder_input_ids.shape[-1] - 1:
                    # items before diagonal
                    self.parent.assertListEqual(
                        decoder_input_ids_slice[1 : i + 1].tolist(), lm_labels_slice[:i].tolist()
                    )
                # pad items after diagonal
                if i < decoder_input_ids.shape[-1] - 2:
                    self.parent.assertListEqual(
                        decoder_input_ids_slice[i + 2 :].tolist(), lm_labels_slice[i + 1 : -1].tolist()
                    )
            else:
                # all items after square
                self.parent.assertListEqual(decoder_input_ids_slice[1:].tolist(), lm_labels_slice[:-1].tolist())

    def create_and_check_model(
        self,
        config,
        input_ids,
        decoder_input_ids,
        attention_mask,
        decoder_attention_mask,
        lm_labels,
    ):
        model = T5Model(config=config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
        )
        result = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
        decoder_output = result.last_hidden_state
        decoder_past = result.past_key_values
        encoder_output = result.encoder_last_hidden_state

        self.parent.assertEqual(encoder_output.size(), (self.batch_size, self.encoder_seq_length, self.hidden_size))
        self.parent.assertEqual(decoder_output.size(), (self.batch_size, self.decoder_seq_length, self.hidden_size))
        # There should be `num_layers` key value embeddings stored in decoder_past
        self.parent.assertEqual(len(decoder_past), config.num_layers)
        # There should be a self attn key, a self attn value, a cross attn key and a cross attn value stored in each decoder_past tuple
        self.parent.assertEqual(len(decoder_past[0]), 4)

    def create_and_check_with_lm_head(
        self,
        config,
        input_ids,
        decoder_input_ids,
        attention_mask,
        decoder_attention_mask,
        lm_labels,
    ):
        model = T5ForConditionalGeneration(config=config).to(torch_device).eval()
        outputs = model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=lm_labels,
        )
        self.parent.assertEqual(len(outputs), 4)
        self.parent.assertEqual(outputs["logits"].size(), (self.batch_size, self.decoder_seq_length, self.vocab_size))
        self.parent.assertEqual(outputs["loss"].size(), ())

    def create_and_check_with_sequence_classification_head(
        self,
        config,
        input_ids,
        decoder_input_ids,
        attention_mask,
        decoder_attention_mask,
        lm_labels,
    ):
        labels = torch.tensor([1] * self.batch_size, dtype=torch.long, device=torch_device)
        model = T5ForSequenceClassification(config=config).to(torch_device).eval()
        outputs = model(
            input_ids=input_ids,
            decoder_input_ids=input_ids,
            labels=labels,
        )
        # self.parent.assertEqual(len(outputs), 4)
        self.parent.assertEqual(outputs["logits"].size(), (self.batch_size, config.num_labels))
        self.parent.assertEqual(outputs["loss"].size(), ())

    def create_and_check_decoder_model_past(
        self,
        config,
        input_ids,
        decoder_input_ids,
        attention_mask,
        decoder_attention_mask,
        lm_labels,
    ):
        model = T5Model(config=config).get_decoder().to(torch_device).eval()
        # first forward pass
        outputs = model(input_ids, use_cache=True)
        outputs_use_cache_conf = model(input_ids)
        outputs_no_past = model(input_ids, use_cache=False)

        self.parent.assertTrue(len(outputs) == len(outputs_use_cache_conf))
        self.parent.assertTrue(len(outputs) == len(outputs_no_past) + 1)

        output, past_key_values = outputs.to_tuple()

        # create hypothetical next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 1), config.vocab_size)

        # append to next input_ids and
        next_input_ids = torch.cat([input_ids, next_tokens], dim=-1)

        output_from_no_past = model(next_input_ids)["last_hidden_state"]
        output_from_past = model(next_tokens, past_key_values=past_key_values)["last_hidden_state"]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, -1, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, 0, random_slice_idx].detach()

        # test that outputs are equal for slice
        self.parent.assertTrue(torch.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-3))

    def create_and_check_decoder_model_attention_mask_past(
        self,
        config,
        input_ids,
        decoder_input_ids,
        attention_mask,
        decoder_attention_mask,
        lm_labels,
    ):
        model = T5Model(config=config).get_decoder()
        model.to(torch_device)
        model.eval()

        # create attention mask
        attn_mask = torch.ones(input_ids.shape, dtype=torch.long, device=torch_device)

        half_seq_length = input_ids.shape[-1] // 2
        attn_mask[:, half_seq_length:] = 0

        # first forward pass
        output, past_key_values = model(input_ids, attention_mask=attn_mask, use_cache=True).to_tuple()

        # create hypothetical next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 1), config.vocab_size)

        # change a random masked slice from input_ids
        random_seq_idx_to_change = ids_tensor((1,), half_seq_length).item() + 1
        random_other_next_tokens = ids_tensor((self.batch_size, 1), config.vocab_size).squeeze(-1)
        input_ids[:, -random_seq_idx_to_change] = random_other_next_tokens

        # append to next input_ids and attn_mask
        next_input_ids = torch.cat([input_ids, next_tokens], dim=-1)
        attn_mask = torch.cat(
            [attn_mask, torch.ones((attn_mask.shape[0], 1), dtype=torch.long, device=torch_device)],
            dim=1,
        )

        # get two different outputs
        output_from_no_past = model(next_input_ids, attention_mask=attn_mask)["last_hidden_state"]
        output_from_past = model(next_tokens, past_key_values=past_key_values, attention_mask=attn_mask)[
            "last_hidden_state"
        ]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, -1, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, 0, random_slice_idx].detach()

        # test that outputs are equal for slice
        self.parent.assertTrue(torch.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-3))

    def create_and_check_decoder_model_past_large_inputs(
        self,
        config,
        input_ids,
        decoder_input_ids,
        attention_mask,
        decoder_attention_mask,
        lm_labels,
    ):
        model = T5Model(config=config).get_decoder().to(torch_device).eval()
        # first forward pass
        outputs = model(input_ids, attention_mask=attention_mask, use_cache=True)

        output, past_key_values = outputs.to_tuple()

        # create hypothetical multiple next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 3), config.vocab_size)
        next_mask = ids_tensor((self.batch_size, 3), vocab_size=2)

        # append to next input_ids and
        next_input_ids = torch.cat([input_ids, next_tokens], dim=-1)
        next_attention_mask = torch.cat([attention_mask, next_mask], dim=-1)

        output_from_no_past = model(next_input_ids, attention_mask=next_attention_mask)["last_hidden_state"]
        output_from_past = model(next_tokens, attention_mask=next_attention_mask, past_key_values=past_key_values)[
            "last_hidden_state"
        ]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, -3:, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, :, random_slice_idx].detach()

        self.parent.assertTrue(output_from_past_slice.shape[1] == next_tokens.shape[1])

        # test that outputs are equal for slice
        self.parent.assertTrue(torch.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-3))

    def create_and_check_generate_with_past_key_values(
        self,
        config,
        input_ids,
        decoder_input_ids,
        attention_mask,
        decoder_attention_mask,
        lm_labels,
    ):
        model = T5ForConditionalGeneration(config=config).to(torch_device).eval()
        torch.manual_seed(0)
        output_without_past_cache = model.generate(
            input_ids[:1], num_beams=2, max_length=5, do_sample=True, use_cache=False
        )
        torch.manual_seed(0)
        output_with_past_cache = model.generate(input_ids[:1], num_beams=2, max_length=5, do_sample=True)
        self.parent.assertTrue(torch.all(output_with_past_cache == output_without_past_cache))

    def create_and_check_model_fp16_forward(
        self,
        config,
        input_ids,
        decoder_input_ids,
        attention_mask,
        decoder_attention_mask,
        lm_labels,
    ):
        model = T5Model(config=config).to(torch_device).half().eval()
        output = model(input_ids, decoder_input_ids=input_ids, attention_mask=attention_mask)["last_hidden_state"]
        self.parent.assertFalse(torch.isnan(output).any().item())

    def create_and_check_encoder_decoder_shared_weights(
        self,
        config,
        input_ids,
        decoder_input_ids,
        attention_mask,
        decoder_attention_mask,
        lm_labels,
    ):
        for model_class in [T5Model, T5ForConditionalGeneration]:
            torch.manual_seed(0)
            model = model_class(config=config).to(torch_device).eval()
            # load state dict copies weights but does not tie them
            model.encoder.load_state_dict(model.decoder.state_dict(), strict=False)

            torch.manual_seed(0)
            tied_config = copy.deepcopy(config)
            tied_config.tie_encoder_decoder = True
            tied_model = model_class(config=tied_config).to(torch_device).eval()

            model_result = model(
                input_ids=input_ids,
                decoder_input_ids=decoder_input_ids,
                attention_mask=attention_mask,
                decoder_attention_mask=decoder_attention_mask,
            )

            tied_model_result = tied_model(
                input_ids=input_ids,
                decoder_input_ids=decoder_input_ids,
                attention_mask=attention_mask,
                decoder_attention_mask=decoder_attention_mask,
            )

            # check that models has less parameters
            self.parent.assertLess(
                sum(p.numel() for p in tied_model.parameters()), sum(p.numel() for p in model.parameters())
            )
            random_slice_idx = ids_tensor((1,), model_result[0].shape[-1]).item()

            # check that outputs are equal
            self.parent.assertTrue(
                torch.allclose(
                    model_result[0][0, :, random_slice_idx], tied_model_result[0][0, :, random_slice_idx], atol=1e-4
                )
            )

            # check that outputs after saving and loading are equal
            with tempfile.TemporaryDirectory() as tmpdirname:
                tied_model.save_pretrained(tmpdirname)
                tied_model = model_class.from_pretrained(tmpdirname)
                tied_model.to(torch_device)
                tied_model.eval()

                # check that models has less parameters
                self.parent.assertLess(
                    sum(p.numel() for p in tied_model.parameters()), sum(p.numel() for p in model.parameters())
                )
                random_slice_idx = ids_tensor((1,), model_result[0].shape[-1]).item()

                tied_model_result = tied_model(
                    input_ids=input_ids,
                    decoder_input_ids=decoder_input_ids,
                    attention_mask=attention_mask,
                    decoder_attention_mask=decoder_attention_mask,
                )

                # check that outputs are equal
                self.parent.assertTrue(
                    torch.allclose(
                        model_result[0][0, :, random_slice_idx],
                        tied_model_result[0][0, :, random_slice_idx],
                        atol=1e-4,
                    )
                )

    def check_resize_embeddings_t5_v1_1(
        self,
        config,
    ):
        prev_vocab_size = config.vocab_size

        config.tie_word_embeddings = False
        model = T5ForConditionalGeneration(config=config).to(torch_device).eval()
        model.resize_token_embeddings(prev_vocab_size - 10)

        self.parent.assertEqual(model.get_input_embeddings().weight.shape[0], prev_vocab_size - 10)
        self.parent.assertEqual(model.get_output_embeddings().weight.shape[0], prev_vocab_size - 10)
        self.parent.assertEqual(model.config.vocab_size, prev_vocab_size - 10)

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            decoder_input_ids,
            attention_mask,
            decoder_attention_mask,
            lm_labels,
        ) = config_and_inputs

        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "use_cache": False,
        }
        return config, inputs_dict


@require_torch
class T5ModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (T5Model, T5ForConditionalGeneration, T5ForSequenceClassification, T5ForQuestionAnswering)
        if is_torch_available()
        else ()
    )
    all_generative_model_classes = (T5ForConditionalGeneration,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": T5Model,
            "question-answering": T5ForQuestionAnswering,
            "summarization": T5ForConditionalGeneration,
            "text-classification": T5ForSequenceClassification,
            "text2text-generation": T5ForConditionalGeneration,
            "translation": T5ForConditionalGeneration,
            "zero-shot": T5ForSequenceClassification,
        }
        if is_torch_available()
        else {}
    )
    all_parallelizable_model_classes = (T5Model, T5ForConditionalGeneration) if is_torch_available() else ()
    fx_compatible = True
    test_pruning = False
    test_resize_embeddings = True
    test_model_parallel = True
    is_encoder_decoder = True
    # The small T5 model needs higher percentages for CPU/MP tests
    model_split_percents = [0.5, 0.8, 0.9]

    def setUp(self):
        self.model_tester = T5ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=T5Config, d_model=37)

    # `QAPipelineTests` is not working well with slow tokenizers (for some models) and we don't want to touch the file
    # `src/transformers/data/processors/squad.py` (where this test fails for this model)
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
        if tokenizer_name is None:
            return True
        if pipeline_test_case_name == "QAPipelineTests" and not tokenizer_name.endswith("Fast"):
            return True

        return False

    def _create_and_check_torch_fx_tracing(self, config, inputs_dict, output_loss=False):
        if not is_torch_fx_available() or not self.fx_compatible:
            self.skipTest(reason="torch.fx is not available or not compatible with this model")

        configs_no_init = _config_zero_init(config)  # To be sure we have no Nan
        configs_no_init.return_dict = False

        for model_class in self.all_model_classes:
            if model_class.__name__ == "T5ForSequenceClassification":
                continue
            model = model_class(config=configs_no_init)
            model.to(torch_device)
            model.eval()
            inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=output_loss)

            try:
                if model.config.is_encoder_decoder:
                    model.config.use_cache = False  # FSTM still requires this hack -> FSTM should probably be refactored similar to BART afterward
                    labels = inputs.get("labels", None)
                    input_names = [
                        "attention_mask",
                        "decoder_attention_mask",
                        "decoder_input_ids",
                        "input_features",
                        "input_ids",
                        "input_values",
                    ]
                    if labels is not None:
                        input_names.append("labels")
                    filtered_inputs = {k: v for (k, v) in inputs.items() if k in input_names}
                    input_names = list(filtered_inputs.keys())
                    model_output = model(**filtered_inputs)
                    traced_model = symbolic_trace(model, input_names)
                    traced_output = traced_model(**filtered_inputs)
                else:
                    input_names = [
                        "attention_mask",
                        "bbox",
                        "input_features",
                        "input_ids",
                        "input_values",
                        "pixel_values",
                        "token_type_ids",
                        "visual_feats",
                        "visual_pos",
                    ]
                    labels = inputs.get("labels", None)
                    start_positions = inputs.get("start_positions", None)
                    end_positions = inputs.get("end_positions", None)
                    if labels is not None:
                        input_names.append("labels")
                    if start_positions is not None:
                        input_names.append("start_positions")
                    if end_positions is not None:
                        input_names.append("end_positions")
                    filtered_inputs = {k: v for (k, v) in inputs.items() if k in input_names}
                    input_names = list(filtered_inputs.keys())
                    if model.__class__.__name__ in set(MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES.values()) and (
                        not hasattr(model.config, "problem_type") or model.config.problem_type is None
                    ):
                        model.config.problem_type = "single_label_classification"
                    traced_model = symbolic_trace(model, input_names)
                    traced_output = traced_model(**filtered_inputs)
                    model_output = model(**filtered_inputs)

            except Exception as e:
                self.fail(f"Couldn't trace module: {e}")

            def flatten_output(output):
                flatten = []
                for x in output:
                    if isinstance(x, (tuple, list)):
                        flatten += flatten_output(x)
                    elif not isinstance(x, torch.Tensor):
                        continue
                    else:
                        flatten.append(x)
                return flatten

            model_output = flatten_output(model_output)
            traced_output = flatten_output(traced_output)
            num_outputs = len(model_output)

            for i in range(num_outputs):
                self.assertTrue(
                    torch.allclose(model_output[i], traced_output[i]),
                    f"traced {i}th output doesn't match model {i}th output for {model_class}",
                )

            # Test that the model can be serialized and restored properly
            with tempfile.TemporaryDirectory() as tmp_dir_name:
                pkl_file_name = os.path.join(tmp_dir_name, "model.pkl")
                try:
                    with open(pkl_file_name, "wb") as f:
                        pickle.dump(traced_model, f)
                    with open(pkl_file_name, "rb") as f:
                        loaded = pickle.load(f)
                except Exception as e:
                    self.fail(f"Couldn't serialize / deserialize the traced model: {e}")

                loaded_output = loaded(**filtered_inputs)
                loaded_output = flatten_output(loaded_output)

                for i in range(num_outputs):
                    self.assertTrue(
                        torch.allclose(model_output[i], loaded_output[i]),
                        f"serialized model {i}th output doesn't match model {i}th output for {model_class}",
                    )

            # Avoid memory leak. Without this, each call increase RAM usage by ~20MB.
            # (Even with this call, there are still memory leak by ~0.04MB)
            self.clear_torch_jit_class_registry()

    # overwrite because T5 doesn't accept position ids as input and expects `decoder_input_ids`
    def test_custom_4d_attention_mask(self):
        for model_class in self.all_generative_model_classes:
            config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config).to(device=torch_device, dtype=torch.float32)

            (
                input_ids,
                _,
                input_ids_shared_prefix,
                mask_shared_prefix,
                _,
            ) = self._get_custom_4d_mask_test_data()

            logits = model.forward(
                decoder_input_ids=input_ids,
                input_ids=input_dict["input_ids"][:3],
            ).logits
            # logits.shape == torch.Size([3, 4, ...])

            logits_shared_prefix = model(
                input_ids=input_dict["input_ids"][:1],
                decoder_input_ids=input_ids_shared_prefix,
                decoder_attention_mask=mask_shared_prefix,
            )[0]
            # logits_shared_prefix.shape == torch.Size([1, 6, ...])

            out_last_tokens = logits[:, -1, :]  # last tokens in each batch line
            out_shared_prefix_last_tokens = logits_shared_prefix[0, -3:, :]  # last three tokens

            # comparing softmax-normalized logits:
            normalized_0 = F.softmax(out_last_tokens)
            normalized_1 = F.softmax(out_shared_prefix_last_tokens)
            torch.testing.assert_close(normalized_0, normalized_1, rtol=1e-3, atol=1e-4)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_shift_right(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_prepare_lm_labels_via_shift_left(*config_and_inputs)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_v1_1(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        # check that gated gelu feed forward and different word embeddings work
        config = config_and_inputs[0]
        config.tie_word_embeddings = False
        config.feed_forward_proj = "gated-gelu"
        self.model_tester.create_and_check_model(config, *config_and_inputs[1:])

    # T5ForSequenceClassification does not support inputs_embeds
    def test_inputs_embeds(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in (T5Model, T5ForConditionalGeneration, T5ForQuestionAnswering):
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
                inputs["inputs_embeds"] = wte(input_ids)
            else:
                inputs["inputs_embeds"] = wte(encoder_input_ids)
                inputs["decoder_inputs_embeds"] = wte(decoder_input_ids)

            with torch.no_grad():
                model(**inputs)[0]

    def test_config_and_model_silu_gated(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        config = config_and_inputs[0]
        config.feed_forward_proj = "gated-silu"
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_with_lm_head(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_with_lm_head(*config_and_inputs)

    def test_with_sequence_classification_head(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_with_sequence_classification_head(*config_and_inputs)

    def test_decoder_model_past(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_decoder_model_past(*config_and_inputs)

    def test_decoder_model_past_with_attn_mask(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_decoder_model_attention_mask_past(*config_and_inputs)

    def test_decoder_model_past_with_3d_attn_mask(self):
        (
            config,
            input_ids,
            decoder_input_ids,
            attention_mask,
            decoder_attention_mask,
            lm_labels,
        ) = self.model_tester.prepare_config_and_inputs()

        attention_mask = ids_tensor(
            [self.model_tester.batch_size, self.model_tester.encoder_seq_length, self.model_tester.encoder_seq_length],
            vocab_size=2,
        )
        decoder_attention_mask = ids_tensor(
            [self.model_tester.batch_size, self.model_tester.decoder_seq_length, self.model_tester.decoder_seq_length],
            vocab_size=2,
        )

        self.model_tester.create_and_check_decoder_model_attention_mask_past(
            config,
            input_ids,
            decoder_input_ids,
            attention_mask,
            decoder_attention_mask,
            lm_labels,
        )

    def test_decoder_model_past_with_large_inputs(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_decoder_model_past_large_inputs(*config_and_inputs)

    def test_generate_with_past_key_values(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_generate_with_past_key_values(*config_and_inputs)

    def test_encoder_decoder_shared_weights(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_encoder_decoder_shared_weights(*config_and_inputs)

    @unittest.skipIf(torch_device == "cpu", "Cant do half precision")
    def test_model_fp16_forward(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_fp16_forward(*config_and_inputs)

    def test_v1_1_resize_embeddings(self):
        config = self.model_tester.prepare_config_and_inputs()[0]
        self.model_tester.check_resize_embeddings_t5_v1_1(config)

    @slow
    def test_model_from_pretrained(self):
        model_name = "google-t5/t5-small"
        model = T5Model.from_pretrained(model_name)
        self.assertIsNotNone(model)

    @unittest.skip(reason="Test has a segmentation fault on torch 1.8.0")
    def test_export_to_onnx(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        model = T5Model(config_and_inputs[0]).to(torch_device)
        with tempfile.TemporaryDirectory() as tmpdirname:
            torch.onnx.export(
                model,
                (config_and_inputs[1], config_and_inputs[3], config_and_inputs[2]),
                f"{tmpdirname}/t5_test.onnx",
                export_params=True,
                opset_version=9,
                input_names=["input_ids", "decoder_input_ids"],
            )

    def test_generate_with_head_masking(self):
        attention_names = ["encoder_attentions", "decoder_attentions", "cross_attentions"]
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        config = config_and_inputs[0]
        max_length = config_and_inputs[1].shape[-1] + 3
        model = T5ForConditionalGeneration(config).eval()
        model.to(torch_device)

        head_masking = {
            "head_mask": torch.zeros(config.num_layers, config.num_heads, device=torch_device),
            "decoder_head_mask": torch.zeros(config.num_decoder_layers, config.num_heads, device=torch_device),
            "cross_attn_head_mask": torch.zeros(config.num_decoder_layers, config.num_heads, device=torch_device),
        }

        for attn_name, (name, mask) in zip(attention_names, head_masking.items()):
            head_masks = {name: mask}
            # Explicitly pass decoder_head_mask as it is required from T5 model when head_mask specified
            if name == "head_mask":
                head_masks["decoder_head_mask"] = torch.ones(
                    config.num_decoder_layers, config.num_heads, device=torch_device
                )

            out = model.generate(
                config_and_inputs[1],
                num_beams=1,
                max_length=max_length,
                output_attentions=True,
                return_dict_in_generate=True,
                **head_masks,
            )
            # We check the state of decoder_attentions and cross_attentions just from the last step
            attn_weights = out[attn_name] if attn_name == attention_names[0] else out[attn_name][-1]
            self.assertEqual(sum([w.sum().item() for w in attn_weights]), 0.0)


class T5EncoderOnlyModelTester:
    def __init__(
        self,
        parent,
        vocab_size=99,
        batch_size=13,
        encoder_seq_length=7,
        # For common tests
        use_attention_mask=True,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        d_ff=37,
        relative_attention_num_buckets=8,
        is_training=False,
        dropout_rate=0.1,
        initializer_factor=0.002,
        is_encoder_decoder=False,
        eos_token_id=1,
        pad_token_id=0,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.encoder_seq_length = encoder_seq_length
        # For common tests
        self.seq_length = self.encoder_seq_length
        self.use_attention_mask = use_attention_mask
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.d_ff = d_ff
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.dropout_rate = dropout_rate
        self.initializer_factor = initializer_factor
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.is_encoder_decoder = is_encoder_decoder
        self.scope = None
        self.is_training = is_training

    def get_large_model_config(self):
        return T5Config.from_pretrained("google-t5/t5-base")

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.encoder_seq_length], self.vocab_size)

        attention_mask = None
        if self.use_attention_mask:
            attention_mask = ids_tensor([self.batch_size, self.encoder_seq_length], vocab_size=2)

        config = T5Config(
            vocab_size=self.vocab_size,
            d_model=self.hidden_size,
            d_ff=self.d_ff,
            d_kv=self.hidden_size // self.num_attention_heads,
            num_layers=self.num_hidden_layers,
            num_heads=self.num_attention_heads,
            relative_attention_num_buckets=self.relative_attention_num_buckets,
            dropout_rate=self.dropout_rate,
            initializer_factor=self.initializer_factor,
            eos_token_id=self.eos_token_id,
            bos_token_id=self.pad_token_id,
            pad_token_id=self.pad_token_id,
            is_encoder_decoder=self.is_encoder_decoder,
        )

        return (
            config,
            input_ids,
            attention_mask,
        )

    def create_and_check_model(
        self,
        config,
        input_ids,
        attention_mask,
    ):
        model = T5EncoderModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        result = model(input_ids=input_ids)
        encoder_output = result.last_hidden_state

        self.parent.assertEqual(encoder_output.size(), (self.batch_size, self.encoder_seq_length, self.hidden_size))

    def create_and_check_model_fp16_forward(
        self,
        config,
        input_ids,
        attention_mask,
    ):
        model = T5EncoderModel(config=config).to(torch_device).half().eval()
        output = model(input_ids, attention_mask=attention_mask)["last_hidden_state"]
        self.parent.assertFalse(torch.isnan(output).any().item())

    def create_and_check_with_token_classification_head(
        self,
        config,
        input_ids,
        attention_mask,
    ):
        labels = torch.tensor([1] * self.seq_length * self.batch_size, dtype=torch.long, device=torch_device)
        model = T5ForTokenClassification(config=config).to(torch_device).eval()
        outputs = model(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )
        self.parent.assertEqual(outputs["logits"].size(), (self.batch_size, self.seq_length, config.num_labels))
        self.parent.assertEqual(outputs["loss"].size(), ())

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            attention_mask,
        ) = config_and_inputs

        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


class T5EncoderOnlyModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (T5EncoderModel, T5ForTokenClassification) if is_torch_available() else ()
    test_pruning = False
    test_resize_embeddings = False
    test_model_parallel = True
    pipeline_model_mapping = (
        {
            "token-classification": T5ForTokenClassification,
        }
        if is_torch_available()
        else {}
    )
    all_parallelizable_model_classes = (T5EncoderModel,) if is_torch_available() else ()

    def setUp(self):
        self.model_tester = T5EncoderOnlyModelTester(self)
        self.config_tester = ConfigTester(self, config_class=T5Config, d_model=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skipIf(torch_device == "cpu", "Cant do half precision")
    def test_model_fp16_forward(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_fp16_forward(*config_and_inputs)

    def test_with_token_classification_head(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_with_token_classification_head(*config_and_inputs)

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
        if tokenizer_name is None:
            return True

        # `T5EncoderOnlyModelTest` is not working well with slow tokenizers (for some models) and we don't want to touch the file
        # `src/transformers/data/processors/squad.py` (where this test fails for this model)
        if pipeline_test_case_name == "TokenClassificationPipelineTests" and not tokenizer_name.endswith("Fast"):
            return True

        return False


def use_task_specific_params(model, task):
    model.config.update(model.config.task_specific_params[task])


@require_torch
@require_accelerate
@require_tokenizers
@slow
class T5ModelFp16Tests(unittest.TestCase):
    def test_fp16_fp32_conversion(self):
        r"""
        A test to check whether the argument `keep_in_fp32_modules` correctly does its job
        """
        orig_import = __import__
        accelerate_mock = unittest.mock.Mock()

        # mock import of accelerate
        def import_accelerate_mock(name, *args, **kwargs):
            if name == "accelerate":
                if accelerate_available:
                    return accelerate_mock
                else:
                    raise ImportError
            return orig_import(name, *args, **kwargs)

        # Load without using `accelerate`
        with unittest.mock.patch("builtins.__import__", side_effect=import_accelerate_mock):
            accelerate_available = False

            model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small", torch_dtype=torch.float16)
            self.assertTrue(model.decoder.block[0].layer[2].DenseReluDense.wo.weight.dtype == torch.float32)
            self.assertTrue(model.decoder.block[0].layer[2].DenseReluDense.wi.weight.dtype == torch.float16)

            # Load without in bf16
            model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small", torch_dtype=torch.bfloat16)
            self.assertTrue(model.decoder.block[0].layer[2].DenseReluDense.wo.weight.dtype == torch.bfloat16)
            self.assertTrue(model.decoder.block[0].layer[2].DenseReluDense.wi.weight.dtype == torch.bfloat16)

        # Load using `accelerate` in bf16
        model = T5ForConditionalGeneration.from_pretrained(
            "google-t5/t5-small", torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.assertTrue(model.decoder.block[0].layer[2].DenseReluDense.wo.weight.dtype == torch.bfloat16)
        self.assertTrue(model.decoder.block[0].layer[2].DenseReluDense.wi.weight.dtype == torch.bfloat16)

        # Load using `accelerate` in bf16
        model = T5ForConditionalGeneration.from_pretrained(
            "google-t5/t5-small", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
        )
        self.assertTrue(model.decoder.block[0].layer[2].DenseReluDense.wo.weight.dtype == torch.bfloat16)
        self.assertTrue(model.decoder.block[0].layer[2].DenseReluDense.wi.weight.dtype == torch.bfloat16)

        # Load without using `accelerate`
        model = T5ForConditionalGeneration.from_pretrained(
            "google-t5/t5-small", torch_dtype=torch.float16, low_cpu_mem_usage=True
        )
        self.assertTrue(model.decoder.block[0].layer[2].DenseReluDense.wo.weight.dtype == torch.float32)
        self.assertTrue(model.decoder.block[0].layer[2].DenseReluDense.wi.weight.dtype == torch.float16)

        # Load using `accelerate`
        model = T5ForConditionalGeneration.from_pretrained(
            "google-t5/t5-small", torch_dtype=torch.float16, device_map="auto"
        )
        self.assertTrue(model.decoder.block[0].layer[2].DenseReluDense.wo.weight.dtype == torch.float32)
        self.assertTrue(model.decoder.block[0].layer[2].DenseReluDense.wi.weight.dtype == torch.float16)


@require_torch
@require_sentencepiece
@require_tokenizers
class T5ModelIntegrationTests(unittest.TestCase):
    @cached_property
    def model(self):
        return T5ForConditionalGeneration.from_pretrained("google-t5/t5-base").to(torch_device)

    @cached_property
    def tokenizer(self):
        return T5Tokenizer.from_pretrained("google-t5/t5-base")

    @slow
    def test_torch_quant(self):
        r"""
        Test that a simple `torch.quantization.quantize_dynamic` call works on a T5 model.
        """
        model_name = "google/flan-t5-small"
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        input_text = "Answer the following yes/no question by reasoning step-by-step. Can you write a whole Haiku in a single tweet?"
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids
        _ = model.generate(input_ids)

    @slow
    def test_small_generation(self):
        model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small").to(torch_device)
        model.config.max_length = 8
        model.config.num_beams = 1
        model.config.do_sample = False
        tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")

        input_ids = tokenizer("summarize: Hello there", return_tensors="pt").input_ids.to(torch_device)

        sequences = model.generate(input_ids)

        output_str = tokenizer.batch_decode(sequences, skip_special_tokens=True)[0]
        self.assertTrue(output_str == "Hello there!")

    @slow
    def test_small_integration_test(self):
        """
        For comparision run:
        >>> import t5  # pip install t5==0.7.1
        >>> from t5.data.sentencepiece_vocabulary import SentencePieceVocabulary

        >>> path_to_mtf_small_t5_checkpoint = '<fill_in>'
        >>> path_to_mtf_small_spm_model_path = '<fill_in>'
        >>> t5_model = t5.models.MtfModel(model_dir=path_to_mtf_small_t5_checkpoint, batch_size=1, tpu=None)
        >>> vocab = SentencePieceVocabulary(path_to_mtf_small_spm_model_path, extra_ids=100)
        >>> score = t5_model.score(inputs=["Hello there"], targets=["Hi I am"], vocabulary=vocab)
        """

        model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small").to(torch_device)
        tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")

        input_ids = tokenizer("Hello there", return_tensors="pt").input_ids
        labels = tokenizer("Hi I am", return_tensors="pt").input_ids

        loss = model(input_ids.to(torch_device), labels=labels.to(torch_device)).loss
        mtf_score = -(labels.shape[-1] * loss.item())

        EXPECTED_SCORE = -19.0845
        self.assertTrue(abs(mtf_score - EXPECTED_SCORE) < 1e-4)

    @slow
    def test_small_v1_1_integration_test(self):
        """
        For comparision run:
        >>> import t5  # pip install t5==0.7.1
        >>> from t5.data.sentencepiece_vocabulary import SentencePieceVocabulary

        >>> path_to_mtf_small_t5_v1_1_checkpoint = '<fill_in>'
        >>> path_to_mtf_small_spm_model_path = '<fill_in>'
        >>> t5_model = t5.models.MtfModel(model_dir=path_to_mtf_small_t5_v1_1_checkpoint, batch_size=1, tpu=None)
        >>> vocab = SentencePieceVocabulary(path_to_mtf_small_spm_model_path, extra_ids=100)
        >>> score = t5_model.score(inputs=["Hello there"], targets=["Hi I am"], vocabulary=vocab)
        """

        model = T5ForConditionalGeneration.from_pretrained("google/t5-v1_1-small").to(torch_device)
        tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-small")

        input_ids = tokenizer("Hello there", return_tensors="pt").input_ids
        labels = tokenizer("Hi I am", return_tensors="pt").input_ids

        loss = model(input_ids.to(torch_device), labels=labels.to(torch_device)).loss
        mtf_score = -(labels.shape[-1] * loss.item())

        EXPECTED_SCORE = -59.0293
        self.assertTrue(abs(mtf_score - EXPECTED_SCORE) < 1e-4)

    @slow
    def test_small_byt5_integration_test(self):
        """
        For comparision run:
        >>> import t5  # pip install t5==0.9.1

        >>> path_to_byt5_small_checkpoint = '<fill_in>'
        >>> t5_model = t5.models.MtfModel(model_dir=path_to_tf_checkpoint, batch_size=1, tpu=None)
        >>> vocab = t5.data.ByteVocabulary()
        >>> score = t5_model.score(inputs=["Hello there"], targets=["Hi I am"], vocabulary=vocab)
        """

        model = T5ForConditionalGeneration.from_pretrained("google/byt5-small").to(torch_device)
        tokenizer = ByT5Tokenizer.from_pretrained("google/byt5-small")

        input_ids = tokenizer("Hello there", return_tensors="pt").input_ids
        labels = tokenizer("Hi I am", return_tensors="pt").input_ids

        loss = model(input_ids.to(torch_device), labels=labels.to(torch_device)).loss
        mtf_score = -(labels.shape[-1] * loss.item())

        EXPECTED_SCORE = -60.7397
        self.assertTrue(abs(mtf_score - EXPECTED_SCORE) < 1e-4)

    @slow
    def test_summarization(self):
        model = self.model
        tok = self.tokenizer

        FRANCE_ARTICLE = (  # @noqa
            "Marseille, France (CNN)The French prosecutor leading an investigation into the crash of Germanwings"
            " Flight 9525 insisted Wednesday that he was not aware of any video footage from on board the plane."
            ' Marseille prosecutor Brice Robin told CNN that "so far no videos were used in the crash investigation."'
            ' He added, "A person who has such a video needs to immediately give it to the investigators." Robin\'s'
            " comments follow claims by two magazines, German daily Bild and French Paris Match, of a cell phone video"
            " showing the harrowing final seconds from on board Germanwings Flight 9525 as it crashed into the French"
            " Alps. All 150 on board were killed. Paris Match and Bild reported that the video was recovered from a"
            " phone at the wreckage site. The two publications described the supposed video, but did not post it on"
            " their websites. The publications said that they watched the video, which was found by a source close to"
            " the investigation. \"One can hear cries of 'My God' in several languages,\" Paris Match reported."
            ' "Metallic banging can also be heard more than three times, perhaps of the pilot trying to open the'
            " cockpit door with a heavy object.  Towards the end, after a heavy shake, stronger than the others, the"
            ' screaming intensifies. Then nothing." "It is a very disturbing scene," said Julian Reichelt,'
            " editor-in-chief of Bild online. An official with France's accident investigation agency, the BEA, said"
            " the agency is not aware of any such video. Lt. Col. Jean-Marc Menichini, a French Gendarmerie spokesman"
            " in charge of communications on rescue efforts around the Germanwings crash site, told CNN that the"
            ' reports were "completely wrong" and "unwarranted." Cell phones have been collected at the site, he said,'
            ' but that they "hadn\'t been exploited yet." Menichini said he believed the cell phones would need to be'
            " sent to the Criminal Research Institute in Rosny sous-Bois, near Paris, in order to be analyzed by"
            " specialized technicians working hand-in-hand with investigators. But none of the cell phones found so"
            " far have been sent to the institute, Menichini said. Asked whether staff involved in the search could"
            ' have leaked a memory card to the media, Menichini answered with a categorical "no." Reichelt told "Erin'
            ' Burnett: Outfront" that he had watched the video and stood by the report, saying Bild and Paris Match'
            ' are "very confident" that the clip is real. He noted that investigators only revealed they\'d recovered'
            ' cell phones from the crash site after Bild and Paris Match published their reports. "That is something'
            " we did not know before. ... Overall we can say many things of the investigation weren't revealed by the"
            ' investigation at the beginning," he said. What was mental state of Germanwings co-pilot? German airline'
            " Lufthansa confirmed Tuesday that co-pilot Andreas Lubitz had battled depression years before he took the"
            " controls of Germanwings Flight 9525, which he's accused of deliberately crashing last week in the"
            ' French Alps. Lubitz told his Lufthansa flight training school in 2009 that he had a "previous episode of'
            ' severe depression," the airline said Tuesday. Email correspondence between Lubitz and the school'
            " discovered in an internal investigation, Lufthansa said, included medical documents he submitted in"
            " connection with resuming his flight training. The announcement indicates that Lufthansa, the parent"
            " company of Germanwings, knew of Lubitz's battle with depression, allowed him to continue training and"
            " ultimately put him in the cockpit. Lufthansa, whose CEO Carsten Spohr previously said Lubitz was 100%"
            ' fit to fly, described its statement Tuesday as a "swift and seamless clarification" and said it was'
            " sharing the information and documents -- including training and medical records -- with public"
            " prosecutors. Spohr traveled to the crash site Wednesday, where recovery teams have been working for the"
            " past week to recover human remains and plane debris scattered across a steep mountainside. He saw the"
            " crisis center set up in Seyne-les-Alpes, laid a wreath in the village of Le Vernet, closer to the crash"
            " site, where grieving families have left flowers at a simple stone memorial. Menichini told CNN late"
            " Tuesday that no visible human remains were left at the site but recovery teams would keep searching."
            " French President Francois Hollande, speaking Tuesday, said that it should be possible to identify all"
            " the victims using DNA analysis by the end of the week, sooner than authorities had previously suggested."
            " In the meantime, the recovery of the victims' personal belongings will start Wednesday, Menichini said."
            " Among those personal belongings could be more cell phones belonging to the 144 passengers and six crew"
            " on board. Check out the latest from our correspondents . The details about Lubitz's correspondence with"
            " the flight school during his training were among several developments as investigators continued to"
            " delve into what caused the crash and Lubitz's possible motive for downing the jet. A Lufthansa"
            " spokesperson told CNN on Tuesday that Lubitz had a valid medical certificate, had passed all his"
            ' examinations and "held all the licenses required." Earlier, a spokesman for the prosecutor\'s office in'
            " Dusseldorf, Christoph Kumpa, said medical records reveal Lubitz suffered from suicidal tendencies at"
            " some point before his aviation career and underwent psychotherapy before he got his pilot's license."
            " Kumpa emphasized there's no evidence suggesting Lubitz was suicidal or acting aggressively before the"
            " crash. Investigators are looking into whether Lubitz feared his medical condition would cause him to"
            " lose his pilot's license, a European government official briefed on the investigation told CNN on"
            ' Tuesday. While flying was "a big part of his life," the source said, it\'s only one theory being'
            " considered. Another source, a law enforcement official briefed on the investigation, also told CNN that"
            " authorities believe the primary motive for Lubitz to bring down the plane was that he feared he would"
            " not be allowed to fly because of his medical problems. Lubitz's girlfriend told investigators he had"
            " seen an eye doctor and a neuropsychologist, both of whom deemed him unfit to work recently and concluded"
            " he had psychological issues, the European government official said. But no matter what details emerge"
            " about his previous mental health struggles, there's more to the story, said Brian Russell, a forensic"
            ' psychologist. "Psychology can explain why somebody would turn rage inward on themselves about the fact'
            " that maybe they weren't going to keep doing their job and they're upset about that and so they're"
            ' suicidal," he said. "But there is no mental illness that explains why somebody then feels entitled to'
            " also take that rage and turn it outward on 149 other people who had nothing to do with the person's"
            ' problems." Germanwings crash compensation: What we know . Who was the captain of Germanwings Flight'
            " 9525? CNN's Margot Haddad reported from Marseille and Pamela Brown from Dusseldorf, while Laura"
            " Smith-Spark wrote from London. CNN's Frederik Pleitgen, Pamela Boykoff, Antonia Mortensen, Sandrine"
            " Amiel and Anna-Maja Rappard contributed to this report."
        )
        SHORTER_ARTICLE = (
            "(CNN)The Palestinian Authority officially became the 123rd member of the International Criminal Court on"
            " Wednesday, a step that gives the court jurisdiction over alleged crimes in Palestinian territories. The"
            " formal accession was marked with a ceremony at The Hague, in the Netherlands, where the court is based."
            " The Palestinians signed the ICC's founding Rome Statute in January, when they also accepted its"
            ' jurisdiction over alleged crimes committed "in the occupied Palestinian territory, including East'
            ' Jerusalem, since June 13, 2014." Later that month, the ICC opened a preliminary examination into the'
            " situation in Palestinian territories, paving the way for possible war crimes investigations against"
            " Israelis. As members of the court, Palestinians may be subject to counter-charges as well. Israel and"
            " the United States, neither of which is an ICC member, opposed the Palestinians' efforts to join the"
            " body. But Palestinian Foreign Minister Riad al-Malki, speaking at Wednesday's ceremony, said it was a"
            ' move toward greater justice. "As Palestine formally becomes a State Party to the Rome Statute today, the'
            ' world is also a step closer to ending a long era of impunity and injustice," he said, according to an'
            ' ICC news release. "Indeed, today brings us closer to our shared goals of justice and peace." Judge'
            " Kuniko Ozaki, a vice president of the ICC, said acceding to the treaty was just the first step for the"
            ' Palestinians. "As the Rome Statute today enters into force for the State of Palestine, Palestine'
            " acquires all the rights as well as responsibilities that come with being a State Party to the Statute."
            ' These are substantive commitments, which cannot be taken lightly," she said. Rights group Human Rights'
            ' Watch welcomed the development. "Governments seeking to penalize Palestine for joining the ICC should'
            " immediately end their pressure, and countries that support universal acceptance of the court's treaty"
            ' should speak out to welcome its membership," said Balkees Jarrah, international justice counsel for the'
            " group. \"What's objectionable is the attempts to undermine international justice, not Palestine's"
            ' decision to join a treaty to which over 100 countries around the world are members." In January, when'
            " the preliminary ICC examination was opened, Israeli Prime Minister Benjamin Netanyahu described it as an"
            ' outrage, saying the court was overstepping its boundaries. The United States also said it "strongly"'
            " disagreed with the court's decision. \"As we have said repeatedly, we do not believe that Palestine is a"
            ' state and therefore we do not believe that it is eligible to join the ICC," the State Department said in'
            ' a statement. It urged the warring sides to resolve their differences through direct negotiations. "We'
            ' will continue to oppose actions against Israel at the ICC as counterproductive to the cause of peace,"'
            " it said. But the ICC begs to differ with the definition of a state for its purposes and refers to the"
            ' territories as "Palestine." While a preliminary examination is not a formal investigation, it allows the'
            " court to review evidence and determine whether to investigate suspects on both sides. Prosecutor Fatou"
            ' Bensouda said her office would "conduct its analysis in full independence and impartiality." The war'
            " between Israel and Hamas militants in Gaza last summer left more than 2,000 people dead. The inquiry"
            " will include alleged war crimes committed since June. The International Criminal Court was set up in"
            " 2002 to prosecute genocide, crimes against humanity and war crimes. CNN's Vasco Cotovio, Kareem Khadder"
            " and Faith Karimi contributed to this report."
        )
        IRAN_ARTICLE = (
            "(CNN)The United States and its negotiating partners reached a very strong framework agreement with Iran"
            " in Lausanne, Switzerland, on Thursday that limits Iran's nuclear program in such a way as to effectively"
            " block it from building a nuclear weapon. Expect pushback anyway, if the recent past is any harbinger."
            " Just last month, in an attempt to head off such an agreement, House Speaker John Boehner invited Israeli"
            " Prime Minister Benjamin Netanyahu to preemptively blast it before Congress, and 47 senators sent a"
            " letter to the Iranian leadership warning them away from a deal. The debate that has already begun since"
            " the announcement of the new framework will likely result in more heat than light. It will not be helped"
            " by the gathering swirl of dubious assumptions and doubtful assertions. Let us address some of these: ."
            " The most misleading assertion, despite universal rejection by experts, is that the negotiations'"
            " objective at the outset was the total elimination of any nuclear program in Iran. That is the position"
            " of Netanyahu and his acolytes in the U.S. Congress. But that is not and never was the objective. If it"
            " had been, there would have been no Iranian team at the negotiating table. Rather, the objective has"
            " always been to structure an agreement or series of agreements so that Iran could not covertly develop a"
            " nuclear arsenal before the United States and its allies could respond. The new framework has exceeded"
            " expectations in achieving that goal. It would reduce Iran's low-enriched uranium stockpile, cut by"
            " two-thirds its number of installed centrifuges and implement a rigorous inspection regime. Another"
            " dubious assumption of opponents is that the Iranian nuclear program is a covert weapons program. Despite"
            " sharp accusations by some in the United States and its allies, Iran denies having such a program, and"
            " U.S. intelligence contends that Iran has not yet made the decision to build a nuclear weapon. Iran's"
            " continued cooperation with International Atomic Energy Agency inspections is further evidence on this"
            " point, and we'll know even more about Iran's program in the coming months and years because of the deal."
            " In fact, the inspections provisions that are part of this agreement are designed to protect against any"
            " covert action by the Iranians. What's more, the rhetoric of some members of Congress has implied that"
            " the negotiations have been between only the United States and Iran (i.e., the 47 senators' letter"
            " warning that a deal might be killed by Congress or a future president). This of course is not the case."
            " The talks were between Iran and the five permanent members of the U.N. Security Council (United States,"
            " United Kingdom, France, China and Russia) plus Germany, dubbed the P5+1. While the United States has"
            " played a leading role in the effort, it negotiated the terms alongside its partners. If the agreement"
            " reached by the P5+1 is rejected by Congress, it could result in an unraveling of the sanctions on Iran"
            " and threaten NATO cohesion in other areas. Another questionable assertion is that this agreement"
            " contains a sunset clause, after which Iran will be free to do as it pleases. Again, this is not the"
            " case. Some of the restrictions on Iran's nuclear activities, such as uranium enrichment, will be eased"
            " or eliminated over time, as long as 15 years. But most importantly, the framework agreement includes"
            " Iran's ratification of the Additional Protocol, which allows IAEA inspectors expanded access to nuclear"
            " sites both declared and nondeclared. This provision will be permanent. It does not sunset. Thus, going"
            " forward, if Iran decides to enrich uranium to weapons-grade levels, monitors will be able to detect such"
            " a move in a matter of days and alert the U.N. Security Council. Many in Congress have said that the"
            ' agreement should be a formal treaty requiring the Senate to "advise and consent." But the issue is not'
            " suited for a treaty. Treaties impose equivalent obligations on all signatories. For example, the New"
            " START treaty limits Russia and the United States to 1,550 deployed strategic warheads. But any agreement"
            " with Iran will not be so balanced.  The restrictions and obligations in the final framework agreement"
            " will be imposed almost exclusively on Iran. The P5+1 are obligated only to ease and eventually remove"
            " most but not all economic sanctions, which were imposed as leverage to gain this final deal. Finally"
            " some insist that any agreement must address Iranian missile programs, human rights violations or support"
            " for Hamas or Hezbollah.  As important as these issues are, and they must indeed be addressed, they are"
            " unrelated to the most important aim of a nuclear deal: preventing a nuclear Iran.  To include them in"
            " the negotiations would be a poison pill. This agreement should be judged on its merits and on how it"
            " affects the security of our negotiating partners and allies, including Israel. Those judgments should be"
            " fact-based, not based on questionable assertions or dubious assumptions."
        )
        ARTICLE_SUBWAY = (
            "New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County, New York. A"
            " year later, she got married again in Westchester County, but to a different man and without divorcing"
            " her first husband.  Only 18 days after that marriage, she got hitched yet again. Then, Barrientos"
            ' declared "I do" five more times, sometimes only within two weeks of each other. In 2010, she married'
            " once more, this time in the Bronx. In an application for a marriage license, she stated it was her"
            ' "first and only" marriage. Barrientos, now 39, is facing two criminal counts of "offering a false'
            ' instrument for filing in the first degree," referring to her false statements on the 2010 marriage'
            " license application, according to court documents. Prosecutors said the marriages were part of an"
            " immigration scam. On Friday, she pleaded not guilty at State Supreme Court in the Bronx, according to"
            " her attorney, Christopher Wright, who declined to comment further. After leaving court, Barrientos was"
            " arrested and charged with theft of service and criminal trespass for allegedly sneaking into the New"
            " York subway through an emergency exit, said Detective Annette Markowski, a police spokeswoman. In total,"
            " Barrientos has been married 10 times, with nine of her marriages occurring between 1999 and 2002.  All"
            " occurred either in Westchester County, Long Island, New Jersey or the Bronx. She is believed to still be"
            " married to four men, and at one time, she was married to eight men at once, prosecutors say. Prosecutors"
            " said the immigration scam involved some of her husbands, who filed for permanent residence status"
            " shortly after the marriages.  Any divorces happened only after such filings were approved. It was"
            " unclear whether any of the men will be prosecuted. The case was referred to the Bronx District"
            " Attorney's Office by Immigration and Customs Enforcement and the Department of Homeland Security's"
            ' Investigation Division. Seven of the men are from so-called "red-flagged" countries, including Egypt,'
            " Turkey, Georgia, Pakistan and Mali. Her eighth husband, Rashid Rajput, was deported in 2006 to his"
            " native Pakistan after an investigation by the Joint Terrorism Task Force. If convicted, Barrientos faces"
            " up to four years in prison.  Her next court appearance is scheduled for May 18."
        )

        expected_summaries = [
            'prosecutor: "so far no videos were used in the crash investigation" two magazines claim to have found a'
            " cell phone video of the final seconds . \"one can hear cries of 'My God' in several languages,\" one"
            " magazine says .",
            "the formal accession was marked by a ceremony at The Hague, in the Netherlands . the ICC opened a"
            " preliminary examination into the situation in the occupied Palestinian territory . as members of the"
            " court, Palestinians may be subject to counter-charges as well .",
            "the u.s. and its negotiating partners reached a very strong framework agreement with Iran . aaron miller:"
            " the debate that has already begun since the announcement of the new framework will likely result in more"
            " heat than light . the deal would reduce Iran's low-enriched uranium stockpile, cut centrifuges and"
            " implement a rigorous inspection regime .",
            "prosecutors say the marriages were part of an immigration scam . if convicted, barrientos faces two"
            ' criminal counts of "offering a false instrument for filing in the first degree" she has been married 10'
            " times, with nine of her marriages occurring between 1999 and 2002 .",
        ]

        use_task_specific_params(model, "summarization")

        dct = tok(
            [model.config.prefix + x for x in [FRANCE_ARTICLE, SHORTER_ARTICLE, IRAN_ARTICLE, ARTICLE_SUBWAY]],
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(torch_device)
        self.assertEqual(512, dct["input_ids"].shape[1])

        hypotheses_batch = model.generate(
            **dct,
            num_beams=4,
            length_penalty=2.0,
            max_length=142,
            min_length=56,
            no_repeat_ngram_size=3,
            do_sample=False,
            early_stopping=True,
        )

        decoded = tok.batch_decode(hypotheses_batch, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        self.assertListEqual(
            expected_summaries,
            decoded,
        )

    @slow
    def test_translation_en_to_de(self):
        model = self.model
        tok = self.tokenizer
        use_task_specific_params(model, "translation_en_to_de")

        en_text = '"Luigi often said to me that he never wanted the brothers to end up in court", she wrote.'
        expected_translation = (
            '"Luigi sagte mir oft, dass er nie wollte, dass die Brüder am Gericht sitzen", schrieb sie.'
        )

        input_ids = tok.encode(model.config.prefix + en_text, return_tensors="pt")
        input_ids = input_ids.to(torch_device)
        output = model.generate(input_ids)
        translation = tok.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        self.assertEqual(translation, expected_translation)

    @slow
    def test_translation_en_to_fr(self):
        model = self.model  # google-t5/t5-base
        tok = self.tokenizer
        use_task_specific_params(model, "translation_en_to_fr")

        en_text = (
            ' This image section from an infrared recording by the Spitzer telescope shows a "family portrait" of'
            " countless generations of stars: the oldest stars are seen as blue dots. "
        )

        input_ids = tok.encode(model.config.prefix + en_text, return_tensors="pt")
        input_ids = input_ids.to(torch_device)

        output = model.generate(
            input_ids=input_ids,
            num_beams=4,
            length_penalty=2.0,
            max_length=100,
            no_repeat_ngram_size=3,
            do_sample=False,
            early_stopping=True,
        )
        translation = tok.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        new_truncated_translation = (
            "Cette section d'images provenant de l'enregistrement infrarouge effectué par le télescope Spitzer montre "
            "un "
            "« portrait familial » de générations innombrables d’étoiles : les plus anciennes sont observées "
            "sous forme "
            "de points bleus."
        )

        self.assertEqual(translation, new_truncated_translation)

    @slow
    def test_translation_en_to_ro(self):
        model = self.model
        tok = self.tokenizer
        use_task_specific_params(model, "translation_en_to_ro")
        en_text = "Taco Bell said it plans to add 2,000 locations in the US by 2022."
        expected_translation = "Taco Bell a declarat că intenţionează să adauge 2 000 de locaţii în SUA până în 2022."

        inputs = tok(model.config.prefix + en_text, return_tensors="pt").to(torch_device)
        output = model.generate(**inputs)
        translation = tok.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        self.assertEqual(translation, expected_translation)

    @slow
    def test_contrastive_search_t5(self):
        article = (
            " New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County, New York. A"
            " year later, she got married again in Westchester County, but to a different man and without divorcing"
            " her first husband.  Only 18 days after that marriage, she got hitched yet again. Then, Barrientos"
            ' declared "I do" five more times, sometimes only within two weeks of each other. In 2010, she married'
            " once more, this time in the Bronx. In an application for a marriage license, she stated it was her"
            ' "first and only" marriage. Barrientos, now 39, is facing two criminal counts of "offering a false'
            ' instrument for filing in the first degree," referring to her false statements on the 2010 marriage'
            " license application, according to court documents. Prosecutors said the marriages were part of an"
            " immigration scam. On Friday, she pleaded not guilty at State Supreme Court in the Bronx, according to"
            " her attorney, Christopher Wright, who declined to comment further. After leaving court, Barrientos was"
            " arrested and charged with theft of service and criminal trespass for allegedly sneaking into the New"
            " York subway through an emergency exit, said Detective Annette Markowski, a police spokeswoman. In total,"
            " Barrientos has been married 10 times, with nine of her marriages occurring between 1999 and 2002.  All"
            " occurred either in Westchester County, Long Island, New Jersey or the Bronx. She is believed to still be"
            " married to four men, and at one time, she was married to eight men at once, prosecutors say. Prosecutors"
            " said the immigration scam involved some of her husbands, who filed for permanent residence status"
            " shortly after the marriages.  Any divorces happened only after such filings were approved. It was"
            " unclear whether any of the men will be prosecuted. The case was referred to the Bronx District"
            " Attorney's Office by Immigration and Customs Enforcement and the Department of Homeland Security's"
            ' Investigation Division. Seven of the men are from so-called "red-flagged" countries, including Egypt,'
            " Turkey, Georgia, Pakistan and Mali. Her eighth husband, Rashid Rajput, was deported in 2006 to his"
            " native Pakistan after an investigation by the Joint Terrorism Task Force. If convicted, Barrientos faces"
            " up to four years in prison.  Her next court appearance is scheduled for May 18."
        )
        article = "summarize: " + article.strip()
        t5_tokenizer = AutoTokenizer.from_pretrained("flax-community/t5-base-cnn-dm")
        t5_model = T5ForConditionalGeneration.from_pretrained("flax-community/t5-base-cnn-dm").to(torch_device)
        input_ids = t5_tokenizer(
            article, add_special_tokens=False, truncation=True, max_length=512, return_tensors="pt"
        ).input_ids.to(torch_device)

        outputs = t5_model.generate(input_ids, penalty_alpha=0.5, top_k=5, max_length=64)
        generated_text = t5_tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # TODO: @arthur?
        # PR #31938 caused regression on this test which was fixed by PR #34089
        self.assertListEqual(
            generated_text,
            [
                "Liana Barrientos has been married 10 times, nine of them in the Bronx . Her husbands filed for "
                "permanent residence after the marriages, prosecutors say ."
            ],
        )

    @slow
    @require_torch_accelerator
    def test_compile_static_cache(self):
        NUM_TOKENS_TO_GENERATE = 40
        EXPECTED_TEXT_COMPLETION = [
            "theory of relativity states that 1) the speed of light is constant in all inertial reference frames. the laws of physics are the same for all inertial reference frames.",
            "ketchup is my favorite condiment.",
        ]

        prompts = [
            "summarize: Simply put, the theory of relativity states that 1) the speed of light is constant in all inertial "
            "reference frames, and 2) the laws of physics are the same for all inertial reference frames.\nThe "
            "theory of relativity is not hard to grasp.",
            "summarize: My favorite all time favorite condiment is ketchup. I love it on everything. I love it on my eggs, "
            "my fries, my chicken, my burgers, my hot dogs, my sandwiches, my salads, my pizza.",
        ]
        model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small").to(torch_device)
        tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

        # Dynamic Cache
        generated_ids = model.generate(**inputs, max_new_tokens=NUM_TOKENS_TO_GENERATE, do_sample=False)
        dynamic_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, dynamic_text)

        # Static Cache
        generated_ids = model.generate(
            **inputs, max_new_tokens=NUM_TOKENS_TO_GENERATE, do_sample=False, cache_implementation="static"
        )
        static_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, static_text)

        # Static Cache + compile
        model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)
        generated_ids = model.generate(
            **inputs, max_new_tokens=NUM_TOKENS_TO_GENERATE, do_sample=False, cache_implementation="static"
        )
        static_compiled_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, static_compiled_text)

    @slow
    @require_torch_accelerator
    def test_compile_static_cache_encoder(self):
        prompts = [
            "summarize: Simply put, the theory of relativity states that 1) the speed of light is constant in all inertial "
            "reference frames, and 2) the laws of physics are the same for all inertial reference frames.\nThe "
            "theory of relativity is not hard to grasp.",
            "summarize: My favorite all time favorite condiment is ketchup. I love it on everything. I love it on my eggs, "
            "my fries, my chicken, my burgers, my hot dogs, my sandwiches, my salads, my pizza.",
        ]
        model = T5EncoderModel.from_pretrained("google-t5/t5-small").to(torch_device)
        tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

        logits = model(**inputs)

        model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)
        logits_compiled = model(**inputs)
        torch.testing.assert_close(logits[0][:, -3:, -3], logits_compiled[0][:, -3:, -3], rtol=1e-5, atol=1e-5)


@require_torch
class TestAsymmetricT5(unittest.TestCase):
    def build_model_and_check_forward_pass(self, **kwargs):
        tester = T5ModelTester(self, **kwargs)
        config, *inputs = tester.prepare_config_and_inputs()
        (
            input_ids,
            decoder_input_ids,
            attention_mask,
            decoder_attention_mask,
            lm_labels,
        ) = inputs
        model = T5ForConditionalGeneration(config=config).to(torch_device).eval()
        outputs = model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=lm_labels,
        )
        # outputs = model(*inputs)
        assert len(outputs) == 4
        assert outputs["logits"].size() == (tester.batch_size, tester.decoder_seq_length, tester.vocab_size)
        assert outputs["loss"].size() == ()
        return model

    def test_small_decoder(self):
        # num_hidden_layers is passed to T5Config as num_layers
        model = self.build_model_and_check_forward_pass(decoder_layers=1, num_hidden_layers=2)
        assert len(model.encoder.block) == 2
        assert len(model.decoder.block) == 1

    def test_defaulting_to_symmetry(self):
        # num_hidden_layers is passed to T5Config as num_layers
        model = self.build_model_and_check_forward_pass(num_hidden_layers=2)
        assert len(model.decoder.block) == len(model.encoder.block) == 2
