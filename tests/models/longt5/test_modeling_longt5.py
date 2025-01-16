# coding=utf-8
# Copyright 2022 Google LongT5 Authors and HuggingFace Inc. team.
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

from transformers import LongT5Config, is_torch_available
from transformers.models.auto import get_values
from transformers.testing_utils import require_sentencepiece, require_tokenizers, require_torch, slow, torch_device
from transformers.utils import cached_property

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch
    import torch.nn.functional as F

    from transformers import (
        MODEL_FOR_QUESTION_ANSWERING_MAPPING,
        AutoTokenizer,
        LongT5EncoderModel,
        LongT5ForConditionalGeneration,
        LongT5Model,
    )


class LongT5ModelTester:
    def __init__(
        self,
        parent,
        vocab_size=99,
        batch_size=13,
        encoder_seq_length=7,
        decoder_seq_length=9,
        local_radius=5,
        encoder_attention_type="local",
        global_block_size=3,
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
        large_model_config_path="google/long-t5-local-large",
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.encoder_seq_length = encoder_seq_length
        self.decoder_seq_length = decoder_seq_length
        self.local_radius = local_radius
        self.block_len = local_radius + 1
        self.encoder_attention_type = encoder_attention_type
        self.global_block_size = global_block_size
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
        self.large_model_config_path = large_model_config_path

    def get_large_model_config(self):
        return LongT5Config.from_pretrained(self.large_model_config_path)

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

    def get_pipeline_config(self):
        return LongT5Config(
            vocab_size=166,  # longt5 forces 100 extra tokens
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
            local_radius=self.local_radius,
            encoder_attention_type=self.encoder_attention_type,
            global_block_size=self.global_block_size,
        )

    def get_config(self):
        return LongT5Config(
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
            local_radius=self.local_radius,
            encoder_attention_type=self.encoder_attention_type,
            global_block_size=self.global_block_size,
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
        model = LongT5Model(config=config)
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
        model = LongT5Model(config=config)
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
        model = LongT5ForConditionalGeneration(config=config).to(torch_device).eval()
        outputs = model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=lm_labels,
        )
        self.parent.assertEqual(len(outputs), 4)
        self.parent.assertEqual(outputs["logits"].size(), (self.batch_size, self.decoder_seq_length, self.vocab_size))
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
        model = LongT5Model(config=config).get_decoder().to(torch_device).eval()
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
        model = LongT5Model(config=config).get_decoder()
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
        model = LongT5Model(config=config).get_decoder().to(torch_device).eval()
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
        model = LongT5ForConditionalGeneration(config=config).to(torch_device).eval()
        torch.manual_seed(0)
        output_without_past_cache = model.generate(
            input_ids[:1], num_beams=2, max_length=5, do_sample=True, use_cache=False
        )
        torch.manual_seed(0)
        output_with_past_cache = model.generate(input_ids[:1], num_beams=2, max_length=5, do_sample=True)
        self.parent.assertTrue(torch.all(output_with_past_cache == output_without_past_cache))

    def create_and_check_encoder_decoder_shared_weights(
        self,
        config,
        input_ids,
        decoder_input_ids,
        attention_mask,
        decoder_attention_mask,
        lm_labels,
    ):
        for model_class in [LongT5Model, LongT5ForConditionalGeneration]:
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
class LongT5ModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (LongT5Model, LongT5ForConditionalGeneration) if is_torch_available() else ()
    all_generative_model_classes = (LongT5ForConditionalGeneration,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": LongT5Model,
            "summarization": LongT5ForConditionalGeneration,
            "text2text-generation": LongT5ForConditionalGeneration,
            "translation": LongT5ForConditionalGeneration,
        }
        if is_torch_available()
        else {}
    )
    fx_compatible = False
    test_pruning = False
    test_torchscript = True
    test_resize_embeddings = True
    test_model_parallel = False
    is_encoder_decoder = True

    def setUp(self):
        self.model_tester = LongT5ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=LongT5Config, d_model=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_shift_right(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_prepare_lm_labels_via_shift_left(*config_and_inputs)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_with_lm_head(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_with_lm_head(*config_and_inputs)

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

    def test_decoder_model_past_with_large_inputs(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_decoder_model_past_large_inputs(*config_and_inputs)

    def test_generate_with_past_key_values(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_generate_with_past_key_values(*config_and_inputs)

    def test_encoder_decoder_shared_weights(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_encoder_decoder_shared_weights(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        model_name = "google/long-t5-local-base"
        model = LongT5Model.from_pretrained(model_name)
        self.assertIsNotNone(model)

    @slow
    def test_export_to_onnx(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        model = LongT5Model(config_and_inputs[0]).to(torch_device)
        with tempfile.TemporaryDirectory() as tmpdirname:
            torch.onnx.export(
                model,
                (config_and_inputs[1], config_and_inputs[3], config_and_inputs[2]),
                f"{tmpdirname}/longt5_test.onnx",
                export_params=True,
                opset_version=14,
                input_names=["input_ids", "decoder_input_ids"],
            )

    def test_generate_with_head_masking(self):
        attention_names = ["encoder_attentions", "decoder_attentions", "cross_attentions"]
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        config = config_and_inputs[0]
        max_length = config_and_inputs[1].shape[-1] + 3
        model = LongT5ForConditionalGeneration(config).eval()
        model.to(torch_device)

        head_masking = {
            "head_mask": torch.zeros(config.num_layers, config.num_heads, device=torch_device),
            "decoder_head_mask": torch.zeros(config.num_decoder_layers, config.num_heads, device=torch_device),
            "cross_attn_head_mask": torch.zeros(config.num_decoder_layers, config.num_heads, device=torch_device),
        }

        for attn_name, (name, mask) in zip(attention_names, head_masking.items()):
            head_masks = {name: mask}
            # Explicitly pass decoder_head_mask as it is required from LONGT5 model when head_mask specified
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

    def test_attention_outputs(self):
        if not self.has_attentions:
            self.skipTest(reason="has_attentions is set to False")

        else:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            config.return_dict = True

            seq_len = getattr(self.model_tester, "seq_length", None)
            decoder_seq_length = getattr(self.model_tester, "decoder_seq_length", seq_len)
            encoder_seq_length = getattr(self.model_tester, "encoder_seq_length", seq_len)
            decoder_key_length = getattr(self.model_tester, "decoder_key_length", decoder_seq_length)
            encoder_key_length = getattr(self.model_tester, "key_length", encoder_seq_length)
            chunk_length = getattr(self.model_tester, "chunk_length", None)
            block_len = getattr(self.model_tester, "block_len", None)

            if chunk_length is not None and hasattr(self.model_tester, "num_hashes"):
                encoder_seq_length = encoder_seq_length * self.model_tester.num_hashes

            for model_class in self.all_model_classes:
                inputs_dict["output_attentions"] = True
                inputs_dict["output_hidden_states"] = False
                config.return_dict = True
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

                self.assertListEqual(
                    list(attentions[0].shape[-3:]),
                    [self.model_tester.num_attention_heads, block_len, 3 * block_len],
                )
                out_len = len(outputs)

                if self.is_encoder_decoder:
                    correct_outlen = 5

                    # loss is at first position
                    if "labels" in inputs_dict:
                        correct_outlen += 1  # loss is added to beginning
                    # Question Answering model returns start_logits and end_logits
                    if model_class in get_values(MODEL_FOR_QUESTION_ANSWERING_MAPPING):
                        correct_outlen += 1  # start_logits and end_logits instead of only 1 output
                    if "past_key_values" in outputs:
                        correct_outlen += 1  # past_key_values have been returned

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
                            decoder_seq_length,
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
                self.assertListEqual(
                    list(self_attentions[0].shape[-3:]),
                    [self.model_tester.num_attention_heads, block_len, 3 * block_len],
                )

    def _check_encoder_attention_for_generate(self, attentions, batch_size, config, seq_length):
        block_len = getattr(self.model_tester, "block_len", None)
        encoder_expected_shape = (batch_size, 2, config.num_attention_heads, block_len, 3 * block_len)
        self.assertIsInstance(attentions, tuple)
        self.assertListEqual(
            [layer_attentions.shape for layer_attentions in attentions],
            [encoder_expected_shape] * len(attentions),
        )

    @unittest.skip(
        reason="This architecure has tied weights by default and there is no way to remove it, check: https://github.com/huggingface/transformers/pull/31771#issuecomment-2210915245"
    )
    def test_load_save_without_tied_weights(self):
        pass


@require_torch
class LongT5TGlobalModelTest(LongT5ModelTest):
    def setUp(self):
        self.model_tester = LongT5ModelTester(
            self, encoder_attention_type="transient-global", large_model_config_path="google/long-t5-tglobal-large"
        )
        self.config_tester = ConfigTester(self, config_class=LongT5Config, d_model=37)

    def test_attention_outputs(self):
        if not self.has_attentions:
            self.skipTest(reason="has_attentions is set to False")

        else:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            config.return_dict = True

            seq_len = getattr(self.model_tester, "seq_length", None)
            decoder_seq_length = getattr(self.model_tester, "decoder_seq_length", seq_len)
            encoder_seq_length = getattr(self.model_tester, "encoder_seq_length", seq_len)
            decoder_key_length = getattr(self.model_tester, "decoder_key_length", decoder_seq_length)
            encoder_key_length = getattr(self.model_tester, "key_length", encoder_seq_length)
            chunk_length = getattr(self.model_tester, "chunk_length", None)
            block_len = getattr(self.model_tester, "block_len", None)
            global_block_size = getattr(self.model_tester, "global_block_size", None)
            global_seq_len = encoder_seq_length // global_block_size

            if chunk_length is not None and hasattr(self.model_tester, "num_hashes"):
                encoder_seq_length = encoder_seq_length * self.model_tester.num_hashes

            for model_class in self.all_model_classes:
                inputs_dict["output_attentions"] = True
                inputs_dict["output_hidden_states"] = False
                config.return_dict = True
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

                self.assertListEqual(
                    list(attentions[0].shape[-3:]),
                    [self.model_tester.num_attention_heads, block_len, 3 * block_len + global_seq_len],
                )
                out_len = len(outputs)

                if self.is_encoder_decoder:
                    correct_outlen = 5

                    # loss is at first position
                    if "labels" in inputs_dict:
                        correct_outlen += 1  # loss is added to beginning
                    # Question Answering model returns start_logits and end_logits
                    if model_class in get_values(MODEL_FOR_QUESTION_ANSWERING_MAPPING):
                        correct_outlen += 1  # start_logits and end_logits instead of only 1 output
                    if "past_key_values" in outputs:
                        correct_outlen += 1  # past_key_values have been returned

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
                            decoder_seq_length,
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
                self.assertListEqual(
                    list(self_attentions[0].shape[-3:]),
                    [self.model_tester.num_attention_heads, block_len, 3 * block_len + global_seq_len],
                )

    def _check_encoder_attention_for_generate(self, attentions, batch_size, config, seq_length):
        block_len = getattr(self.model_tester, "block_len", None)
        global_block_size = getattr(self.model_tester, "global_block_size", None)
        global_seq_length = seq_length // global_block_size
        encoder_expected_shape = (
            batch_size,
            2,
            config.num_attention_heads,
            block_len,
            3 * block_len + global_seq_length,
        )
        self.assertIsInstance(attentions, tuple)
        self.assertListEqual(
            [layer_attentions.shape for layer_attentions in attentions],
            [encoder_expected_shape] * len(attentions),
        )


class LongT5EncoderOnlyModelTester:
    def __init__(
        self,
        parent,
        vocab_size=99,
        batch_size=13,
        encoder_seq_length=7,
        local_radius=5,
        encoder_attention_type="local",
        global_block_size=3,
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
        large_model_config_path="google/long-t5-local-large",
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.encoder_seq_length = encoder_seq_length
        self.local_radius = local_radius
        self.block_len = local_radius + 1
        self.encoder_attention_type = encoder_attention_type
        self.global_block_size = global_block_size
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
        self.large_model_config_path = large_model_config_path

    def get_large_model_config(self):
        return LongT5Config.from_pretrained(self.large_model_config_path)

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.encoder_seq_length], self.vocab_size)

        attention_mask = None
        if self.use_attention_mask:
            attention_mask = ids_tensor([self.batch_size, self.encoder_seq_length], vocab_size=2)

        config = LongT5Config(
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
            local_radius=self.local_radius,
            encoder_attention_type=self.encoder_attention_type,
            global_block_size=self.global_block_size,
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
        model = LongT5EncoderModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        result = model(input_ids=input_ids)
        encoder_output = result.last_hidden_state

        self.parent.assertEqual(encoder_output.size(), (self.batch_size, self.encoder_seq_length, self.hidden_size))

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


class LongT5EncoderOnlyModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (LongT5EncoderModel,) if is_torch_available() else ()
    test_pruning = False
    test_torchscript = True
    test_resize_embeddings = False
    test_model_parallel = False

    def setUp(self):
        self.model_tester = LongT5EncoderOnlyModelTester(self)
        self.config_tester = ConfigTester(self, config_class=LongT5Config, d_model=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_attention_outputs(self):
        if not self.has_attentions:
            self.skipTest(reason="has_attentions is set to False")

        else:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            config.return_dict = True

            block_len = getattr(self.model_tester, "block_len", 4)

            for model_class in self.all_model_classes:
                inputs_dict["output_attentions"] = True
                inputs_dict["output_hidden_states"] = False
                config.return_dict = True
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

                self.assertListEqual(
                    list(attentions[0].shape[-3:]),
                    [self.model_tester.num_attention_heads, block_len, 3 * block_len],
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

                if hasattr(self.model_tester, "num_hidden_states_types"):
                    added_hidden_states = self.model_tester.num_hidden_states_types
                elif self.is_encoder_decoder:
                    added_hidden_states = 2
                else:
                    added_hidden_states = 1
                self.assertEqual(out_len + added_hidden_states, len(outputs))

                self_attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions

                self.assertEqual(len(self_attentions), self.model_tester.num_hidden_layers)
                self.assertListEqual(
                    list(self_attentions[0].shape[-3:]),
                    [self.model_tester.num_attention_heads, block_len, 3 * block_len],
                )

    @unittest.skip(
        reason="This architecure has tied weights by default and there is no way to remove it, check: https://github.com/huggingface/transformers/pull/31771#issuecomment-2210915245"
    )
    def test_load_save_without_tied_weights(self):
        pass


class LongT5EncoderOnlyTGlobalModelTest(LongT5EncoderOnlyModelTest):
    def setUp(self):
        self.model_tester = LongT5EncoderOnlyModelTester(
            self, encoder_attention_type="transient-global", large_model_config_path="google/long-t5-tglobal-large"
        )
        self.config_tester = ConfigTester(self, config_class=LongT5Config, d_model=37)

    def test_attention_outputs(self):
        if not self.has_attentions:
            self.skipTest(reason="has_attentions is set to False")

        else:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            config.return_dict = True

            block_len = getattr(self.model_tester, "block_len", None)
            seq_len = getattr(self.model_tester, "seq_length", None)
            global_block_size = getattr(self.model_tester, "global_block_size", 4)
            global_seq_len = seq_len // global_block_size

            for model_class in self.all_model_classes:
                inputs_dict["output_attentions"] = True
                inputs_dict["output_hidden_states"] = False
                config.return_dict = True
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

                self.assertListEqual(
                    list(attentions[0].shape[-3:]),
                    [self.model_tester.num_attention_heads, block_len, 3 * block_len + global_seq_len],
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

                if hasattr(self.model_tester, "num_hidden_states_types"):
                    added_hidden_states = self.model_tester.num_hidden_states_types
                elif self.is_encoder_decoder:
                    added_hidden_states = 2
                else:
                    added_hidden_states = 1
                self.assertEqual(out_len + added_hidden_states, len(outputs))

                self_attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions

                self.assertEqual(len(self_attentions), self.model_tester.num_hidden_layers)
                self.assertListEqual(
                    list(self_attentions[0].shape[-3:]),
                    [self.model_tester.num_attention_heads, block_len, 3 * block_len + global_seq_len],
                )


def use_task_specific_params(model, task):
    model.config.update(model.config.task_specific_params[task])


@require_torch
@require_sentencepiece
@require_tokenizers
class LongT5ModelIntegrationTests(unittest.TestCase):
    @cached_property
    def model(self):
        return LongT5ForConditionalGeneration.from_pretrained("Stancld/longt5-tglobal-large-16384-pubmed-3k_steps").to(
            torch_device
        )

    @cached_property
    def tokenizer(self):
        return AutoTokenizer.from_pretrained("Stancld/longt5-tglobal-large-16384-pubmed-3k_steps")

    def expected_summary(self):
        return [
            "background : coronary artery disease ( cad ) is the emerging cause of morbidity and mortality in"
            " developing world . it provides an excellent resolution for visualization of the coronaryarteries for"
            " catheter - based or operating interventions . although the association of this technique with major"
            " complications such as mortality is highly uncommon , it is frequently associated with various cardiac"
            " and noncardiac complications.materials and methods : in aortic stenosis , we aimed to report the"
            " diagnostic performance of 128-slice computed tomography coronary angiogram in 50 patients undergoing for"
            " major noncoron ary cardiac surgery referred"
        ]

    @slow
    def test_summarization(self):
        model = self.model
        tok = self.tokenizer

        ARTICLE = """coronary artery disease ( cad ) is the emerging cause of morbidity and mortality in developing world . \n it provides an excellent resolution for visualization of the coronary arteries for catheter - based or operating interventions . \n
            although the association of this technique with major complications such as mortality is highly uncommon , it is frequently associated with various cardiac and noncardiac complications . computed tomography ( ct ) coronary angiography is
            a promising technique for the evaluation of cad noninvasively . \n it assesses disease within the coronary artery and provides qualitative and quantitative information about nonobstructive atherosclerotic plaque burden within the vessel
            wall . \n thus , ct angiography - based disease evaluation may provide clinically more significant information than conventional angiography . the introduction of multi - slice computed tomography ( msct ) technology such as 64-slice , 12
            8-slice , 256-slice , and now 320-slice msct has produced a high diagnostic accuracy of ct coronary angiography . \n it has consistently showed to have a very high negative predictive value ( well above 90% ) in ruling out patients with s
            ignificant cad defined as coronary luminal stenosis of > 50% . \n the american college of cardiology / american heart association recommends that coronary angiography should be performed before valve surgery in men aged > 40 years , women
            aged > 35 years with coronary risk factors and in postmenopausal women . \n the prevalence of cad in patients undergoing valve replacement is 2040% in developed countries . in the previous studies , \n the incidence of angiographically p
            roven cad in acquired valvular diseases has been shown to vary widely from 9% to 41% . in aortic stenosis , \n we aimed to report the diagnostic performance of 128-slice ct coronary angiography in 50 patients undergoing for major noncoron
            ary cardiac surgery referred for diagnostic invasive coronary angiography to assess the extent and severity of coronary stenosis . \n during january 2013 to december 2014 , we enrolled fifty major noncoronary cardiac surgery patients sche
            duled for invasive coronary angiography who fulfilled the following inclusion criteria of age 40 years , having low or intermediate probability of cad , left ventricular ejection fraction ( lvef ) > 35% , and patient giving informed conse
            nt for undergoing msct and conventional coronary angiography . \n those having any contraindication for contrast injection , lvef < 35% , high pretest probability of cad , and hemodynamic instability were excluded from the study . \n pati
            ents with heart rates of > 70 bpm received ( unless they had known overt heart failure or electrocardiogram ( ecg ) atrioventricular conduction abnormalities ) a single oral dose of 100 mg metoprolol 45 min before the scan . \n patients w
            ith heart rates of > 80 bpm received an additional oral dose of metoprolol if not contraindicated . \n all patients were scanned with a 128-slice ct scanner ( siemens , somatom definition as ) equipped with a new feature in msct technolog
            y , so - called z - axis flying - focus technology . \n the central 32 detector rows acquire 0.6-mm slices , and the flying - focus spot switches back and forth between 2 z positions between each reading . \n two slices per detector row a
            re acquired , which results in a higher oversampling rate in the z - axis , thereby reducing artifacts related to the spiral acquisition and improving spatial resolution down to 0.4 mm . \n a bolus of 6580 ml contrast material ( omnipaque
            ) was injected through an arm vein at a flow rate of 5 ml / s . \n a bolus tracking technique was used to synchronize the arrival of contrast in the coronary arteries with the initiation of the scan . to monitor the arrival of contrast m
            aterial , \n axial scans were obtained at the level of the ascending aorta with a delay of 10 s after the start of the contrast injection . \n the scan was automatically started when a threshold of 150 hounsfield units was reached in a re
            gion of interest positioned in the ascending aorta . \n images were reconstructed with ecg gating to obtain optimal , motion - free image quality . \n all scans were performed within 2 weeks of the msct coronary diagnostic angiogram . a s
            ingle observer unaware of the multi - slice ct results identified coronary lesion as a single vessel , double vessel , or triple vessel disease . \n all lesion , regardless of size , were included for comparison with ct coronary angiograp
            hy . \n lesions were classified as having nonsignificant disease ( luminal irregularities or < 50% stenosis ) or as having significant stenosis . \n stenosis was evaluated in two orthogonal views and classified as significant if the mean
            lumen diameter reduction was 50% using a validated quantitative coronary angiography ( qca ) . \n all scans were analyzed independently by a radiologist and a cardiologist who were unaware of the results of conventional coronary angiograp
            hy . \n total calcium scores of all patients were calculated with dedicated software and expressed as agatston scores . \n the agatston score is a commonly used scoring method that calculates the total amount of calcium on the basis of th
            e number , areas , and peak hounsfield units of the detected calcified lesions . \n all available coronary segments were visually scored for the presence of > 50% considered as significant stenosis . \n maximum intensity projections were
            used to identify coronary lesions and ( curved ) multiplanar reconstructions to classify lesions as significant or nonsignificant . \n data were analyzed using statistical system spss version 20 software ( chicago , il , usa ) . \n the di
            agnostic performance of ct coronary angiography for the detection of significant lesions in coronary arteries with qca as the standard of reference is presented as sensitivity , specificity , positive and negative predictive values , and
            positive and negative likelihood ratios with the corresponding exact 95% of confidence interval ( cis ) . \n comparison between ct and conventional coronary angiography was performed on the two level vessel by vessel ( no or any disease p
            er vessel ) , and patient by patient ( no or any disease per patient ) . \n all scans were performed within 2 weeks of the msct coronary diagnostic angiogram . a single observer unaware of the multi - slice ct results identified coronary
            lesion as a single vessel , double vessel , or triple vessel disease . \n all lesion , regardless of size , were included for comparison with ct coronary angiography . \n lesions were classified as having nonsignificant disease ( luminal
            irregularities or < 50% stenosis ) or as having significant stenosis . \n stenosis was evaluated in two orthogonal views and classified as significant if the mean lumen diameter reduction was 50% using a validated quantitative coronary an
            giography ( qca ) . \n all scans were analyzed independently by a radiologist and a cardiologist who were unaware of the results of conventional coronary angiography . \n total calcium scores of all patients were calculated with dedicated
            software and expressed as agatston scores . \n the agatston score is a commonly used scoring method that calculates the total amount of calcium on the basis of the number , areas , and peak hounsfield units of the detected calcified lesi
            ons . \n all available coronary segments were visually scored for the presence of > 50% considered as significant stenosis . \n maximum intensity projections were used to identify coronary lesions and ( curved ) multiplanar reconstruction
            s to classify lesions as significant or nonsignificant . \n data were analyzed using statistical system spss version 20 software ( chicago , il , usa ) . \n the diagnostic performance of ct coronary angiography for the detection of signif
            icant lesions in coronary arteries with qca as the standard of reference is presented as sensitivity , specificity , positive and negative predictive values , and positive and negative likelihood ratios with the corresponding exact 95% of
            confidence interval ( cis ) . \n comparison between ct and conventional coronary angiography was performed on the two level vessel by vessel ( no or any disease per vessel ) , and patient by patient ( no or any disease per patient ) . \n
            in this study , 29 ( 58% ) subjects were female , and 21 ( 42% ) were male showing an average age of 50.36  8.39 years . \n of fifty patients 24 ( 48% ) , 13 ( 26% ) , eight ( 16% ) , and five ( 10% ) underwent mitral valve replacement ,
            double valve replacement ( dvr ) , aortic valve replacement , and other surgeries , respectively . \n high distribution of cad risk factors such as hypertension ( 24% ) , smoking ( 22% ) , and dyslipidemia ( 18% ) was observed in the stu
            dy group . \n the mean creatinine level was 0.766  0.17 and average dye used in conventional angiography was 48.5  26.6 whereas for ct angiography it was 72.8  6.32 . \n average radiation dose in conventional coronary angiography and msct
            coronary angiography was 5.2 msv and 9.2 msv , respectively . \n the majority of the patients had sinus rhythm ( 68% ) , whereas atrial fibrillation was found in 32% of the subjects . \n patients included in the study had low to intermed
            iate probability of cad . in this study , three patients had complications after conventional angiography . \n complications were of local site hematoma , acute kidney injury managed conservatively , and acute heart failure . \n a patient
            who developed hematoma was obese female patients with body mass index > 30 kg / m . \n the patient suffered from pseudoaneurysm , had hospitalized for 9 days , which leads to increased morbidity and cost of hospital stay . \n the diagnos
            tic accuracy of ct coronary angiography was evaluated regarding true positive , true negative values and is presented in table 1 . the overall sensitivity and \n specificity of ct angiography technique was 100% ( 95% ci : 39.76%100% ) and
            91.30% ( 95% ci : 79.21%97.58% ) , respectively [ table 2 ] . \n the positive predictive value ( 50% ; 95% ci : 15.70%84.30% ) and negative predictive value ( 100% ; 95% ci : 91.59%100% ) of ct angiography were also fairly high in these
            patients . \n recent reports from multiple studies demonstrated that recent - generation msct scanners showed promise for noninvasive detection of coronary stenosis however , until now no studies were found regarding the clinical efficacy
            or prognostic value of 128-slice ct coronary angiography versus conventional invasive coronary angiography in the diagnosis of patients planned for major noncoronary surgeries such as dvr , bentall , atrial septal defect closure , etc .
            in our study , we reported 8% cad prevalence in patients planned for major noncoronary cardiac surgery . \n we performed conventional and msct coronary angiography in all patients and the results showed that ct coronary angiography with i
            nvasive coronary angiography as the reference standard had a considerably high sensitivity ( 100% ) and specificity ( 95.65% ) . \n the health economic model using invasive coronary angiography as the reference standard showed that at a p
            retest probability of cad of 70% or lower , ct coronary angiography resulted in lower cost per patient with a true positive diagnosis . at a pretest probability of cad of 70% or higher , invasive coronary angiography was associated with a
            lower cost per patient with a true positive diagnosis . in our study population , \n two patients developed local site complications in the form of hematoma and pseudoaneurysm after conventional angiography . \n hence , msct coronary ang
            iography will be more favorable in female obese patients with intermediate likelihood of cad . \n hence , msct coronary angiography will be cost - effective in patients of valvular heart diseases . \n however , ct angiography suffers from
            a drawback that average amount of dye used in msct coronary angiography were 72.8  6.32 ml which is higher than average amount of dye required for conventional angiography ( 48.6  26.6 ml ) . \n hence , the use of ct coronary angiography
            could not be used in patients with known renal dysfunction , where reduction of contrast dye load is highly advocated . \n our results show that 128-slice ct coronary angiography is a reliable technique to detect coronary stenosis in pat
            ients planned for noncoronary cardiac surgery . \n although there has been important technological progress in the development of ct coronary angiography , its clinical application remains limited . \n a study wth large numbers of patient
            s is required for the recommendation of only ct coronary angiography for the coronary evaluation in major non - cardiac surgeries . \n mehta institute of cardiology and research center ( affiliated to bj medical college , ahmedabad , guja
            rat , india ) . \n u.n . mehta institute of cardiology and research center ( affiliated to bj medical college , ahmedabad , gujarat , india ) . \n """

        dct = tok(
            [ARTICLE],
            max_length=1024,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(torch_device)

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
            self.expected_summary(),
            decoded,
        )

    @slow
    def test_inference_hidden_states(self):
        model = self.model

        input_ids = torch.tensor(
            [[100, 19, 3, 9, 7142, 1200, 145, 8, 1252, 14145, 2034, 812, 5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
            dtype=torch.long,
            device=torch_device,
        )
        decoder_input_ids = torch.tensor(
            [[100, 19, 3, 9, 7142, 1200, 145, 8, 1252, 14145, 2034, 812, 5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
            dtype=torch.long,
            device=torch_device,
        )
        attention_mask = torch.tensor(
            [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
            dtype=torch.long,
            device=torch_device,
        )

        output = model(
            input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, output_hidden_states=True
        )

        # check if encoder_outputs match
        expected_output_slice = torch.tensor([0.0629, -0.1294, -0.0089, 0.0772, 0.0663], device=torch_device)
        torch.testing.assert_close(
            output.encoder_hidden_states[-1][0, 0, :5], expected_output_slice, rtol=1e-4, atol=1e-4
        )

        # check if logits match
        expected_output_slice = torch.tensor([5.5231, 6.1058, 3.1766, 8.2391, -5.9453], device=torch_device)
        torch.testing.assert_close(output.logits[0, 0, :5], expected_output_slice, rtol=1e-4, atol=1e-4)
