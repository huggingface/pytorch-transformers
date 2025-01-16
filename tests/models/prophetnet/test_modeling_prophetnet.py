# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team, The Microsoft Research team.
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
import tempfile
import unittest

from transformers import ProphetNetConfig, is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        ProphetNetDecoder,
        ProphetNetEncoder,
        ProphetNetForCausalLM,
        ProphetNetForConditionalGeneration,
        ProphetNetModel,
        ProphetNetTokenizer,
    )
    from transformers.modeling_outputs import BaseModelOutput


class ProphetNetModelTester:
    def __init__(
        self,
        parent,
        vocab_size=99,
        batch_size=13,
        hidden_size=16,
        encoder_seq_length=7,
        decoder_seq_length=9,
        # For common tests
        is_training=True,
        use_attention_mask=True,
        use_labels=True,
        decoder_start_token_id=0,
        encoder_ffn_dim=32,
        num_encoder_layers=2,
        num_encoder_attention_heads=4,
        decoder_ffn_dim=32,
        num_decoder_layers=2,
        num_decoder_attention_heads=4,
        max_position_embeddings=30,
        is_encoder_decoder=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        ngram=2,
        num_buckets=32,
        relative_max_distance=128,
        disable_ngram_loss=False,
        scope=None,
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
        self.num_hidden_layers = num_decoder_layers
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.decoder_ffn_dim = decoder_ffn_dim
        self.encoder_ffn_dim = encoder_ffn_dim
        self.num_attention_heads = num_decoder_attention_heads
        self.num_encoder_attention_heads = num_encoder_attention_heads
        self.num_decoder_attention_heads = num_decoder_attention_heads
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.ngram = ngram
        self.num_buckets = num_buckets
        self.relative_max_distance = relative_max_distance
        self.disable_ngram_loss = disable_ngram_loss
        self.max_position_embeddings = max_position_embeddings
        self.is_encoder_decoder = is_encoder_decoder

        self.scope = None
        self.decoder_key_length = decoder_seq_length
        self.base_model_out_len = 7
        self.num_hidden_states_types = 3  # encoder, decoder_main, decoder_ngram
        self.decoder_attention_idx = 2

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.encoder_seq_length], self.vocab_size)
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

    def get_config(self):
        return ProphetNetConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            decoder_ffn_dim=self.decoder_ffn_dim,
            encoder_ffn_dim=self.encoder_ffn_dim,
            num_encoder_attention_heads=self.num_encoder_attention_heads,
            num_decoder_attention_heads=self.num_decoder_attention_heads,
            eos_token_id=self.eos_token_id,
            bos_token_id=self.bos_token_id,
            pad_token_id=self.pad_token_id,
            decoder_start_token_id=self.decoder_start_token_id,
            ngram=self.ngram,
            num_buckets=self.num_buckets,
            relative_max_distance=self.relative_max_distance,
            disable_ngram_loss=self.disable_ngram_loss,
            max_position_embeddings=self.max_position_embeddings,
            is_encoder_decoder=self.is_encoder_decoder,
        )

    def prepare_config_and_inputs_for_decoder(self):
        (
            config,
            input_ids,
            decoder_input_ids,
            attention_mask,
            decoder_attention_mask,
            lm_labels,
        ) = self.prepare_config_and_inputs()

        encoder_hidden_states = floats_tensor([self.batch_size, self.encoder_seq_length, self.hidden_size])
        encoder_attention_mask = ids_tensor([self.batch_size, self.encoder_seq_length], vocab_size=2)

        return (
            config,
            decoder_input_ids,
            decoder_attention_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            lm_labels,
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
        model = ProphetNetModel(config=config)
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
        model = ProphetNetModel(config=config)
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
        self.parent.assertEqual(len(decoder_past), config.num_decoder_layers)
        # There should be a self attn key, a self attn value, a cross attn key and a cross attn value stored in each decoder_past tuple
        self.parent.assertEqual(len(decoder_past[0]), 4)  # cross-attention + uni-directional self-attention

    def create_and_check_with_lm_head(
        self,
        config,
        input_ids,
        decoder_input_ids,
        attention_mask,
        decoder_attention_mask,
        lm_labels,
    ):
        model = ProphetNetForConditionalGeneration(config=config).to(torch_device).eval()
        outputs = model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=lm_labels,
        )
        self.parent.assertEqual(len(outputs), 5)
        self.parent.assertEqual(outputs["logits"].size(), (self.batch_size, self.decoder_seq_length, self.vocab_size))
        self.parent.assertEqual(outputs["loss"].size(), ())

    def create_and_check_causal_lm_decoder(
        self,
        config,
        input_ids,
        decoder_input_ids,
        attention_mask,
        decoder_attention_mask,
        lm_labels,
    ):
        model = ProphetNetForCausalLM(config=config).to(torch_device).eval()
        outputs = model(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            labels=lm_labels,
        )
        self.parent.assertEqual(len(outputs), 4)
        self.parent.assertEqual(outputs["logits"].size(), (self.batch_size, self.decoder_seq_length, self.vocab_size))
        self.parent.assertEqual(outputs["loss"].size(), ())

    def create_and_check_generate_with_past_key_value_states(
        self,
        config,
        input_ids,
        decoder_input_ids,
        attention_mask,
        decoder_attention_mask,
        lm_labels,
    ):
        model = ProphetNetForConditionalGeneration(config=config).to(torch_device).eval()
        torch.manual_seed(0)
        output_without_past_cache = model.generate(
            input_ids[:1], num_beams=2, max_length=5, do_sample=True, use_cache=False
        )
        torch.manual_seed(0)
        output_with_past_cache = model.generate(input_ids[:1], num_beams=2, max_length=5, do_sample=True)
        self.parent.assertTrue(torch.all(output_with_past_cache == output_without_past_cache))

    def create_and_check_decoder_generate_with_past_key_value_states(
        self,
        config,
        input_ids,
        decoder_input_ids,
        attention_mask,
        decoder_attention_mask,
        lm_labels,
    ):
        model = ProphetNetForCausalLM(config=config).to(torch_device).eval()
        torch.manual_seed(0)
        output_without_past_cache = model.generate(
            input_ids[:1], num_beams=2, max_length=10, do_sample=True, use_cache=False
        )
        torch.manual_seed(0)
        output_with_past_cache = model.generate(input_ids[:1], num_beams=2, max_length=10, do_sample=True)
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
        model = ProphetNetModel(config=config).to(torch_device).half().eval()
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
        for model_class in [ProphetNetModel, ProphetNetForConditionalGeneration]:
            torch.manual_seed(0)
            model = model_class(config=config).to(torch_device).eval()
            # load state dict copies weights but does not tie them

            if model_class == ProphetNetForConditionalGeneration:
                model.prophetnet.encoder.load_state_dict(model.prophetnet.decoder.state_dict(), strict=False)
            else:
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

    def check_fast_integration(
        self,
        config,
        *args,
    ):
        input_ids = torch.tensor([[7, 4, 78, 0, 24, 52, 43]], device=torch_device, dtype=torch.long)
        decoder_input_ids = torch.tensor([[12, 62, 25, 11, 47, 15, 14]], device=torch_device, dtype=torch.long)
        attention_mask = torch.tensor([[1, 1, 1, 0, 1, 0, 0]], device=torch_device, dtype=torch.long)
        decoder_attention_mask = torch.tensor([[1, 1, 1, 0, 0, 1, 0]], device=torch_device, dtype=torch.long)
        lm_labels = torch.tensor([[62, 25, 11, 47, 15, 14, 24]], device=torch_device, dtype=torch.long)
        torch.manual_seed(0)
        config.ngram = 4
        model = ProphetNetForConditionalGeneration(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(
                input_ids=input_ids,
                decoder_input_ids=decoder_input_ids,
                attention_mask=attention_mask,
                decoder_attention_mask=decoder_attention_mask,
                labels=lm_labels,
            )
        self.parent.assertTrue(torch.allclose(result.loss, torch.tensor(4.5892, device=torch_device), atol=1e-3))

        expected_logit_slice = torch.tensor(
            [-0.0184, 0.0758, -0.0543, -0.0093, 0.0050, -0.0660, -0.1453], device=torch_device
        )
        self.parent.assertTrue(torch.allclose(result.logits[0, :, 1], expected_logit_slice, atol=1e-3))

    def check_model_with_attn_mask(self, config, input_ids, decoder_input_ids, *args):
        model = ProphetNetModel(config=config)
        model.to(torch_device)
        model.eval()

        outputs_no_mask = model(input_ids=input_ids[:, :5], decoder_input_ids=decoder_input_ids[:, :5])
        attention_mask = torch.ones_like(input_ids)
        decoder_attention_mask = torch.ones_like(decoder_input_ids)

        attention_mask[:, 5:] = 0

        outputs_with_mask = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
        )

        # check encoder
        self.parent.assertTrue(
            torch.allclose(
                outputs_no_mask.encoder_last_hidden_state[0, :, 0],
                outputs_with_mask.encoder_last_hidden_state[0, :5, 0],
                atol=1e-3,
            )
        )

        # check decoder
        # main stream
        self.parent.assertTrue(
            torch.allclose(
                outputs_no_mask.last_hidden_state[0, :, 0], outputs_with_mask.last_hidden_state[0, :5, 0], atol=1e-3
            )
        )
        # predict stream
        self.parent.assertTrue(
            torch.allclose(
                outputs_no_mask.last_hidden_state_ngram[0, :5, 0],
                outputs_with_mask.last_hidden_state_ngram[0, :5, 0],
                atol=1e-2,
            )
        )

    def check_causal_lm_from_pretrained(
        self, config, input_ids, decoder_input_ids, attention_mask, decoder_attention_mask, *args
    ):
        model = ProphetNetForConditionalGeneration(config).to(torch_device).eval()

        with tempfile.TemporaryDirectory() as tmp_dirname:
            model.save_pretrained(tmp_dirname)
            decoder = ProphetNetForCausalLM.from_pretrained(tmp_dirname).to(torch_device)

        encoder_hidden_states = model.prophetnet.encoder(input_ids).last_hidden_state

        model_outputs = model(
            encoder_outputs=BaseModelOutput(last_hidden_state=encoder_hidden_states),
            decoder_input_ids=decoder_input_ids,
        )
        dec_outputs = decoder(encoder_hidden_states=encoder_hidden_states, input_ids=decoder_input_ids)

        self.parent.assertTrue(
            torch.allclose(
                model_outputs.logits[0, :5],
                dec_outputs.logits[0, :5],
                atol=1e-3,
            )
        )

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


class ProphetNetStandaloneDecoderModelTester:
    def __init__(
        self,
        parent,
        vocab_size=99,
        batch_size=13,
        hidden_size=16,
        encoder_seq_length=7,
        decoder_seq_length=7,
        # For common tests
        is_training=True,
        is_decoder=True,
        use_attention_mask=True,
        add_cross_attention=False,
        use_cache=False,
        use_labels=True,
        decoder_start_token_id=0,
        encoder_ffn_dim=32,
        num_encoder_layers=2,
        num_encoder_attention_heads=4,
        decoder_ffn_dim=32,
        num_decoder_layers=2,
        num_decoder_attention_heads=4,
        max_position_embeddings=30,
        is_encoder_decoder=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        ngram=2,
        num_buckets=32,
        relative_max_distance=128,
        disable_ngram_loss=False,
        scope=None,
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
        self.num_hidden_layers = num_decoder_layers
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.decoder_ffn_dim = decoder_ffn_dim
        self.encoder_ffn_dim = encoder_ffn_dim
        self.num_attention_heads = num_decoder_attention_heads
        self.num_encoder_attention_heads = num_encoder_attention_heads
        self.num_decoder_attention_heads = num_decoder_attention_heads
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.ngram = ngram
        self.num_buckets = num_buckets
        self.relative_max_distance = relative_max_distance
        self.use_cache = use_cache
        self.disable_ngram_loss = disable_ngram_loss
        self.max_position_embeddings = max_position_embeddings
        self.add_cross_attention = add_cross_attention
        self.is_encoder_decoder = is_encoder_decoder

        self.scope = None
        self.decoder_key_length = decoder_seq_length
        self.base_model_out_len = 2
        self.num_hidden_states_types = 2  # decoder_main, decoder_ngram
        self.decoder_attention_idx = 1

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.encoder_seq_length], self.vocab_size)

        attention_mask = None
        if self.use_attention_mask:
            attention_mask = ids_tensor([self.batch_size, self.encoder_seq_length], vocab_size=2)

        lm_labels = None
        if self.use_labels:
            lm_labels = ids_tensor([self.batch_size, self.encoder_seq_length], self.vocab_size)

        config = ProphetNetConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            decoder_ffn_dim=self.decoder_ffn_dim,
            encoder_ffn_dim=self.encoder_ffn_dim,
            num_encoder_attention_heads=self.num_encoder_attention_heads,
            num_decoder_attention_heads=self.num_decoder_attention_heads,
            eos_token_id=self.eos_token_id,
            bos_token_id=self.bos_token_id,
            use_cache=self.use_cache,
            pad_token_id=self.pad_token_id,
            decoder_start_token_id=self.decoder_start_token_id,
            ngram=self.ngram,
            num_buckets=self.num_buckets,
            relative_max_distance=self.relative_max_distance,
            disable_ngram_loss=self.disable_ngram_loss,
            max_position_embeddings=self.max_position_embeddings,
            add_cross_attention=self.add_cross_attention,
            is_encoder_decoder=self.is_encoder_decoder,
        )

        return (
            config,
            input_ids,
            attention_mask,
            lm_labels,
        )

    def prepare_config_and_inputs_for_decoder(self):
        (
            config,
            input_ids,
            attention_mask,
            lm_labels,
        ) = self.prepare_config_and_inputs()

        encoder_hidden_states = floats_tensor([self.batch_size, self.encoder_seq_length, self.hidden_size])
        encoder_attention_mask = ids_tensor([self.batch_size, self.encoder_seq_length], vocab_size=2)

        return (
            config,
            input_ids,
            attention_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            lm_labels,
        )

    def create_and_check_decoder_model_past(
        self,
        config,
        input_ids,
        attention_mask,
        lm_labels,
    ):
        config.use_cache = True
        model = ProphetNetDecoder(config=config).to(torch_device).eval()
        # first forward pass
        outputs = model(input_ids, use_cache=True)
        outputs_use_cache_conf = model(input_ids)
        outputs_no_past = model(input_ids, use_cache=False)

        self.parent.assertTrue(len(outputs) == len(outputs_use_cache_conf))
        self.parent.assertTrue(len(outputs) == len(outputs_no_past) + 1)

        past_key_values = outputs["past_key_values"]

        # create hypothetical next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 1), config.vocab_size)

        # append to next input_ids and
        next_input_ids = torch.cat([input_ids, next_tokens], dim=-1)

        output_from_no_past = model(next_input_ids)["last_hidden_state"]
        output_from_past = model(next_tokens, past_key_values=past_key_values)["last_hidden_state"]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, next_input_ids.shape[-1] - 1, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, 0, random_slice_idx].detach()

        # test that outputs are equal for slice
        assert torch.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-3)

    def create_and_check_decoder_model_attention_mask_past(
        self,
        config,
        input_ids,
        attention_mask,
        lm_labels,
    ):
        model = ProphetNetDecoder(config=config).to(torch_device).eval()

        # create attention mask
        attn_mask = torch.ones(input_ids.shape, dtype=torch.long, device=torch_device)

        half_seq_length = input_ids.shape[-1] // 2
        attn_mask[:, half_seq_length:] = 0

        # first forward pass
        past_key_values = model(input_ids, attention_mask=attn_mask, use_cache=True)["past_key_values"]

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
        output_from_no_past = model(next_input_ids)["last_hidden_state"]
        output_from_past = model(next_tokens, past_key_values=past_key_values)["last_hidden_state"]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, next_input_ids.shape[-1] - 1, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, 0, random_slice_idx].detach()

        # test that outputs are equal for slice
        assert torch.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-2)

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            attention_mask,
            lm_labels,
        ) = config_and_inputs

        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


class ProphetNetStandaloneEncoderModelTester:
    def __init__(
        self,
        parent,
        vocab_size=99,
        batch_size=13,
        hidden_size=16,
        encoder_seq_length=7,
        decoder_seq_length=7,
        # For common tests
        is_training=True,
        is_decoder=False,
        use_attention_mask=True,
        add_cross_attention=False,
        use_cache=False,
        use_labels=True,
        decoder_start_token_id=0,
        encoder_ffn_dim=32,
        num_encoder_layers=2,
        num_encoder_attention_heads=4,
        decoder_ffn_dim=32,
        num_decoder_layers=2,
        num_decoder_attention_heads=4,
        max_position_embeddings=30,
        is_encoder_decoder=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        num_buckets=32,
        relative_max_distance=128,
        disable_ngram_loss=False,
        scope=None,
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
        self.num_hidden_layers = num_decoder_layers
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.decoder_ffn_dim = decoder_ffn_dim
        self.encoder_ffn_dim = encoder_ffn_dim
        self.num_attention_heads = num_decoder_attention_heads
        self.num_encoder_attention_heads = num_encoder_attention_heads
        self.num_decoder_attention_heads = num_decoder_attention_heads
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.num_buckets = num_buckets
        self.relative_max_distance = relative_max_distance
        self.use_cache = use_cache
        self.disable_ngram_loss = disable_ngram_loss
        self.max_position_embeddings = max_position_embeddings
        self.add_cross_attention = add_cross_attention
        self.is_encoder_decoder = is_encoder_decoder

        self.scope = None
        self.decoder_key_length = decoder_seq_length
        self.base_model_out_len = 1
        self.num_hidden_states_types = 1
        self.decoder_attention_idx = 1

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.encoder_seq_length], self.vocab_size)

        attention_mask = None
        if self.use_attention_mask:
            attention_mask = ids_tensor([self.batch_size, self.encoder_seq_length], vocab_size=2)

        config = ProphetNetConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            decoder_ffn_dim=self.decoder_ffn_dim,
            encoder_ffn_dim=self.encoder_ffn_dim,
            num_encoder_attention_heads=self.num_encoder_attention_heads,
            num_decoder_attention_heads=self.num_decoder_attention_heads,
            eos_token_id=self.eos_token_id,
            bos_token_id=self.bos_token_id,
            use_cache=self.use_cache,
            pad_token_id=self.pad_token_id,
            decoder_start_token_id=self.decoder_start_token_id,
            num_buckets=self.num_buckets,
            relative_max_distance=self.relative_max_distance,
            disable_ngram_loss=self.disable_ngram_loss,
            max_position_embeddings=self.max_position_embeddings,
            add_cross_attention=self.add_cross_attention,
            is_encoder_decoder=self.is_encoder_decoder,
        )

        return (
            config,
            input_ids,
            attention_mask,
        )

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


@require_torch
class ProphetNetModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (ProphetNetModel, ProphetNetForConditionalGeneration) if is_torch_available() else ()
    all_generative_model_classes = (ProphetNetForConditionalGeneration,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": ProphetNetModel,
            "summarization": ProphetNetForConditionalGeneration,
            "text-generation": ProphetNetForCausalLM,
            "text2text-generation": ProphetNetForConditionalGeneration,
            "translation": ProphetNetForConditionalGeneration,
        }
        if is_torch_available()
        else {}
    )
    test_pruning = False
    test_resize_embeddings = False
    is_encoder_decoder = True

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
        if pipeline_test_case_name == "TextGenerationPipelineTests":
            # Get `ValueError: AttributeError: 'NoneType' object has no attribute 'new_ones'` or `AssertionError`.
            # `ProphetNetConfig` was never used in pipeline tests: cannot create a simple
            # tokenizer.
            return True

        return False

    def setUp(self):
        self.model_tester = ProphetNetModelTester(self)
        self.config_tester = ConfigTester(self, config_class=ProphetNetConfig)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_lm_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_with_lm_head(*config_and_inputs)

    def test_only_decoder_causal_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_causal_lm_decoder(*config_and_inputs)

    def test_fast_integration(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_fast_integration(*config_and_inputs)

    def test_shared_weights(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_encoder_decoder_shared_weights(*config_and_inputs)

    def test_shift_labels_via_shift_left(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_prepare_lm_labels_via_shift_left(*config_and_inputs)

    @unittest.skip(reason="Flaky test with no simple resolution. TODO Fix me @patrickvonplaten")
    def test_decoder_model_generate(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_generate_with_past_key_value_states(*config_and_inputs)

    def test_encoder_decoder_model_generate(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_decoder_generate_with_past_key_value_states(*config_and_inputs)

    def test_attn_mask_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_model_with_attn_mask(*config_and_inputs)

    def test_config_save(self):
        config = self.model_tester.prepare_config_and_inputs()[0]
        config.add_cross_attention = False
        with tempfile.TemporaryDirectory() as tmp_dirname:
            config.save_pretrained(tmp_dirname)
            config = ProphetNetConfig.from_pretrained(tmp_dirname)

        self.assertFalse(config.add_cross_attention)

    def test_causal_lm_from_pretrained(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_causal_lm_from_pretrained(*config_and_inputs)

    @unittest.skipIf(torch_device == "cpu", "Cant do half precision")
    def test_fp16_forward(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_fp16_forward(*config_and_inputs)

    # methods overwrite method in `test_modeling_common.py`
    def test_attention_outputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        seq_len = getattr(self.model_tester, "seq_length", None)
        decoder_seq_length = getattr(self.model_tester, "decoder_seq_length", seq_len)
        encoder_seq_length = getattr(self.model_tester, "encoder_seq_length", seq_len)
        decoder_key_length = getattr(self.model_tester, "decoder_key_length", decoder_seq_length)
        encoder_key_length = getattr(self.model_tester, "key_length", encoder_seq_length)
        chunk_length = getattr(self.model_tester, "chunk_length", None)
        if chunk_length is not None and hasattr(self.model_tester, "num_hashes"):
            encoder_seq_length = encoder_seq_length * self.model_tester.num_hashes

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            if chunk_length is not None:
                self.assertListEqual(
                    list(attentions[0].shape[-4:]),
                    [self.model_tester.num_attention_heads, encoder_seq_length, chunk_length, encoder_key_length],
                )
            else:
                self.assertListEqual(
                    list(attentions[0].shape[-3:]),
                    [self.model_tester.num_attention_heads, encoder_seq_length, encoder_key_length],
                )
            out_len = len(outputs)

            correct_outlen = 7

            # loss is at first position
            if "labels" in inputs_dict:
                correct_outlen += 1  # loss is added to beginning

            self.assertEqual(out_len, correct_outlen)

            # decoder attentions
            decoder_attentions = outputs.decoder_attentions
            self.assertIsInstance(decoder_attentions, (list, tuple))
            self.assertEqual(len(decoder_attentions), self.model_tester.num_hidden_layers)
            self.assertListEqual(
                list(decoder_attentions[0].shape[-3:]),
                [self.model_tester.num_attention_heads, decoder_seq_length, decoder_key_length],
            )

            # cross attentions
            cross_attentions = outputs.cross_attentions
            self.assertIsInstance(cross_attentions, (list, tuple))
            self.assertEqual(len(cross_attentions), self.model_tester.num_hidden_layers)
            self.assertListEqual(
                list(cross_attentions[0].shape[-3:]),
                [
                    self.model_tester.num_attention_heads,
                    (self.model_tester.ngram + 1) * decoder_seq_length,
                    encoder_key_length,
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

            if hasattr(self.model_tester, "num_hidden_states_types"):
                added_hidden_states = self.model_tester.num_hidden_states_types
            elif self.is_encoder_decoder:
                added_hidden_states = 2
            else:
                added_hidden_states = 1
            self.assertEqual(out_len + added_hidden_states, len(outputs))

            self_attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions

            self.assertEqual(len(self_attentions), self.model_tester.num_hidden_layers)
            if chunk_length is not None:
                self.assertListEqual(
                    list(self_attentions[0].shape[-4:]),
                    [self.model_tester.num_attention_heads, encoder_seq_length, chunk_length, encoder_key_length],
                )
            else:
                self.assertListEqual(
                    list(self_attentions[0].shape[-3:]),
                    [self.model_tester.num_attention_heads, encoder_seq_length, encoder_key_length],
                )

    def test_retain_grad_hidden_states_attentions(self):
        # decoder cannot keep gradients
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

        encoder_hidden_states = outputs.encoder_hidden_states[0]
        encoder_attentions = outputs.encoder_attentions[0]
        encoder_hidden_states.retain_grad()
        encoder_attentions.retain_grad()

        output.flatten()[0].backward(retain_graph=True)

        self.assertIsNotNone(encoder_hidden_states.grad)
        self.assertIsNotNone(encoder_attentions.grad)

    @unittest.skip(reason="Generating with head_masking has not been implemented for ProphetNet models yet.")
    def test_generate_with_head_masking(self):
        pass


@require_torch
class ProphetNetStandaloneDecoderModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    all_model_classes = (ProphetNetDecoder, ProphetNetForCausalLM) if is_torch_available() else ()
    all_generative_model_classes = (ProphetNetForCausalLM,) if is_torch_available() else ()
    test_pruning = False

    test_resize_embeddings = False
    is_encoder_decoder = False

    def setUp(self):
        self.model_tester = ProphetNetStandaloneDecoderModelTester(self, is_training=False)
        self.config_tester = ConfigTester(self, config_class=ProphetNetConfig)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_decoder_model_past(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_decoder_model_past(*config_and_inputs)

    def test_decoder_model_attn_mask_past(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_decoder_model_attention_mask_past(*config_and_inputs)

    @unittest.skip(reason="Decoder cannot keep gradients")
    def test_retain_grad_hidden_states_attentions(self):
        return


@require_torch
class ProphetNetStandaloneEncoderModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (ProphetNetEncoder,) if is_torch_available() else ()
    test_pruning = False

    test_resize_embeddings = False
    is_encoder_decoder = False

    def setUp(self):
        self.model_tester = ProphetNetStandaloneEncoderModelTester(self, is_training=False)
        self.config_tester = ConfigTester(self, config_class=ProphetNetConfig)

    def test_config(self):
        self.config_tester.run_common_tests()


@require_torch
class ProphetNetModelIntegrationTest(unittest.TestCase):
    @slow
    def test_pretrained_checkpoint_hidden_states(self):
        model = ProphetNetForConditionalGeneration.from_pretrained("microsoft/prophetnet-large-uncased")
        model.to(torch_device)

        # encoder-decoder outputs
        encoder_ids = torch.tensor(
            [
                [
                    2871,
                    102,
                    2048,
                    3176,
                    2780,
                    1997,
                    2871,
                    26727,
                    2169,
                    2097,
                    12673,
                    1996,
                    8457,
                    2006,
                    2049,
                    8240,
                    2859,
                    2799,
                    1012,
                    2023,
                    6512,
                    2038,
                    2174,
                    13977,
                    2195,
                    25962,
                    1012,
                    102,
                ]
            ]
        ).to(torch_device)

        decoder_prev_ids = torch.tensor([[102, 2129, 2116, 2372, 2024, 2006, 2169, 1997, 2122, 2048, 2780, 1029]]).to(
            torch_device
        )
        output = model(
            input_ids=encoder_ids,
            attention_mask=None,
            encoder_outputs=None,
            decoder_input_ids=decoder_prev_ids,
        )
        output_predited_logits = output[0]
        expected_shape = torch.Size((1, 12, 30522))
        self.assertEqual(output_predited_logits.shape, expected_shape)
        expected_slice = torch.tensor(
            [[[-7.7729, -8.0343, -8.26001], [-7.74213, -7.8629, -8.6000], [-7.7328, -7.8269, -8.5264]]]
        ).to(torch_device)
        #        torch.testing.assert_close(output_predited_logits[:, :3, :3], expected_slice, rtol=1e-4, atol=1e-4)
        assert torch.allclose(output_predited_logits[:, :3, :3], expected_slice, atol=1e-4)

        # encoder outputs
        encoder_outputs = model.prophetnet.encoder(encoder_ids)[0]
        expected_encoder_outputs_slice = torch.tensor(
            [[[-0.2526, -0.1951, -0.2185], [-0.8923, 0.2992, -0.4623], [-0.4585, 0.0165, -0.6652]]]
        ).to(torch_device)
        expected_shape_encoder = torch.Size((1, 28, 1024))
        self.assertEqual(encoder_outputs.shape, expected_shape_encoder)
        #        torch.testing.assert_close(encoder_outputs[:, :3, :3], expected_encoder_outputs_slice, rtol=1e-4, atol=1e-4)
        assert torch.allclose(encoder_outputs[:, :3, :3], expected_encoder_outputs_slice, atol=1e-4)

        # decoder outputs
        decoder_outputs = model.prophetnet.decoder(decoder_prev_ids, encoder_hidden_states=encoder_outputs)
        predicting_streams = decoder_outputs[1].view(1, model.config.ngram, 12, -1)
        predicting_streams_logits = model.lm_head(predicting_streams)
        next_first_stream_logits = predicting_streams_logits[:, 0]
        #        torch.testing.assert_close(next_first_stream_logits[:, :3, :3], expected_slice, rtol=1e-4, atol=1e-4)
        assert torch.allclose(next_first_stream_logits[:, :3, :3], expected_slice, atol=1e-4)

    @slow
    def test_cnndm_inference(self):
        model = ProphetNetForConditionalGeneration.from_pretrained("microsoft/prophetnet-large-uncased-cnndm")
        model.config.max_length = 512
        model.to(torch_device)

        tokenizer = ProphetNetTokenizer.from_pretrained("microsoft/prophetnet-large-uncased-cnndm")

        ARTICLE_TO_SUMMARIZE = (
            "USTC was founded in Beijing by the Chinese Academy of Sciences (CAS) in September 1958. The Director of"
            " CAS, Mr. Guo Moruo was appointed the first president of USTC. USTC's founding mission was to develop a"
            " high-level science and technology workforce, as deemed critical for development of China's economy,"
            ' defense, and science and technology education. The establishment was hailed as "A Major Event in the'
            ' History of Chinese Education and Science." CAS has supported USTC by combining most of its institutes'
            " with the departments of the university. USTC is listed in the top 16 national key universities, becoming"
            " the youngest national key university.".lower()
        )
        input_ids = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=511, return_tensors="pt").input_ids

        input_ids = input_ids.to(torch_device)

        summary_ids = model.generate(
            input_ids, num_beams=4, length_penalty=1.0, no_repeat_ngram_size=3, early_stopping=True
        )
        EXPECTED_SUMMARIZE_512 = (
            "us ##tc was founded by the chinese academy of sciences ( cas ) in 1958 . [X_SEP] us ##tc is listed in the"
            " top 16 national key universities ."
        )
        generated_titles = [
            " ".join(tokenizer.convert_ids_to_tokens(g, skip_special_tokens=True)) for g in summary_ids
        ]
        self.assertListEqual(
            [EXPECTED_SUMMARIZE_512],
            generated_titles,
        )
        input_ids = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=99, return_tensors="pt").input_ids
        input_ids = input_ids.to(torch_device)
        # actually 98 tokens are used. max_length=100 contains bos and eos.
        summary_ids = model.generate(
            input_ids, num_beams=4, length_penalty=1.0, no_repeat_ngram_size=3, early_stopping=True
        )
        EXPECTED_SUMMARIZE_100 = (
            r"us ##tc was founded in beijing by the chinese academy of sciences ( cas ) in 1958 . [X_SEP] us ##tc "
            "'"
            " s founding mission was to develop a high - level science and technology workforce . [X_SEP]"
            ' establishment hailed as " a major event in the history of chinese education and science "'
        )
        generated_titles = [
            " ".join(tokenizer.convert_ids_to_tokens(g, skip_special_tokens=True)) for g in summary_ids
        ]
        self.assertListEqual(
            [EXPECTED_SUMMARIZE_100],
            generated_titles,
        )

    @slow
    def test_question_gen_inference(self):
        model = ProphetNetForConditionalGeneration.from_pretrained("microsoft/prophetnet-large-uncased-squad-qg")
        model.to(torch_device)

        tokenizer = ProphetNetTokenizer.from_pretrained("microsoft/prophetnet-large-uncased-squad-qg")

        INPUTS = [
            "Bill Gates [SEP] Microsoft was founded by Bill Gates and Paul Allen on April 4, 1975.",
            "1975 [SEP] Microsoft was founded by Bill Gates and Paul Allen on April 4, 1975.",
            "April 4, 1975 [SEP] Microsoft was founded by Bill Gates and Paul Allen on April 4, 1975.",
        ]

        input_ids = tokenizer(INPUTS, truncation=True, padding=True, return_tensors="pt").input_ids
        input_ids = input_ids.to(torch_device)

        gen_output = model.generate(input_ids, num_beams=5, early_stopping=True)
        generated_questions = tokenizer.batch_decode(gen_output, skip_special_tokens=True)

        EXPECTED_QUESTIONS = [
            "along with paul allen, who founded microsoft?",
            "what year was microsoft founded?",
            "when was microsoft founded?",
        ]

        self.assertListEqual(
            EXPECTED_QUESTIONS,
            generated_questions,
        )
