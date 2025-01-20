# coding=utf-8
# Copyright 2021, The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch MBART model."""

import copy
import tempfile
import unittest

from transformers import MBartConfig, is_torch_available
from transformers.testing_utils import (
    require_sentencepiece,
    require_tokenizers,
    require_torch,
    require_torch_fp16,
    slow,
    torch_device,
)
from transformers.utils import cached_property

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        AutoTokenizer,
        BatchEncoding,
        MBartForCausalLM,
        MBartForConditionalGeneration,
        MBartForQuestionAnswering,
        MBartForSequenceClassification,
        MBartModel,
    )
    from transformers.models.mbart.modeling_mbart import MBartDecoder, MBartEncoder


def prepare_mbart_inputs_dict(
    config,
    input_ids,
    decoder_input_ids,
    attention_mask=None,
    decoder_attention_mask=None,
    head_mask=None,
    decoder_head_mask=None,
    cross_attn_head_mask=None,
):
    if attention_mask is None:
        attention_mask = input_ids.ne(config.pad_token_id)
    if decoder_attention_mask is None:
        decoder_attention_mask = decoder_input_ids.ne(config.pad_token_id)
    if head_mask is None:
        head_mask = torch.ones(config.encoder_layers, config.encoder_attention_heads, device=torch_device)
    if decoder_head_mask is None:
        decoder_head_mask = torch.ones(config.decoder_layers, config.decoder_attention_heads, device=torch_device)
    if cross_attn_head_mask is None:
        cross_attn_head_mask = torch.ones(config.decoder_layers, config.decoder_attention_heads, device=torch_device)
    return {
        "input_ids": input_ids,
        "decoder_input_ids": decoder_input_ids,
        "attention_mask": attention_mask,
        "decoder_attention_mask": attention_mask,
        "head_mask": head_mask,
        "decoder_head_mask": decoder_head_mask,
        "cross_attn_head_mask": cross_attn_head_mask,
    }


class MBartModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_labels=False,
        vocab_size=99,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=4,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=100,
        eos_token_id=2,
        pad_token_id=1,
        bos_token_id=0,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
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
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size).clamp(
            3,
        )
        input_ids[:, -1] = self.eos_token_id  # Eos Token

        decoder_input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        config = self.get_config()
        inputs_dict = prepare_mbart_inputs_dict(config, input_ids, decoder_input_ids)
        return config, inputs_dict

    def get_config(self):
        return MBartConfig(
            vocab_size=self.vocab_size,
            d_model=self.hidden_size,
            encoder_layers=self.num_hidden_layers,
            decoder_layers=self.num_hidden_layers,
            encoder_attention_heads=self.num_attention_heads,
            decoder_attention_heads=self.num_attention_heads,
            encoder_ffn_dim=self.intermediate_size,
            decoder_ffn_dim=self.intermediate_size,
            dropout=self.hidden_dropout_prob,
            attention_dropout=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            eos_token_id=self.eos_token_id,
            bos_token_id=self.bos_token_id,
            pad_token_id=self.pad_token_id,
        )

    def prepare_config_and_inputs_for_common(self):
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict

    def create_and_check_decoder_model_past_large_inputs(self, config, inputs_dict):
        model = MBartModel(config=config).get_decoder().to(torch_device).eval()
        input_ids = inputs_dict["input_ids"]
        attention_mask = inputs_dict["attention_mask"]
        head_mask = inputs_dict["head_mask"]

        # first forward pass
        outputs = model(input_ids, attention_mask=attention_mask, head_mask=head_mask, use_cache=True)

        output, past_key_values = outputs.to_tuple()

        # create hypothetical multiple next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 3), config.vocab_size)
        next_attn_mask = ids_tensor((self.batch_size, 3), 2)

        # append to next input_ids and
        next_input_ids = torch.cat([input_ids, next_tokens], dim=-1)
        next_attention_mask = torch.cat([attention_mask, next_attn_mask], dim=-1)

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

    def check_encoder_decoder_model_standalone(self, config, inputs_dict):
        model = MBartModel(config=config).to(torch_device).eval()
        outputs = model(**inputs_dict)

        encoder_last_hidden_state = outputs.encoder_last_hidden_state
        last_hidden_state = outputs.last_hidden_state

        with tempfile.TemporaryDirectory() as tmpdirname:
            encoder = model.get_encoder()
            encoder.save_pretrained(tmpdirname)
            encoder = MBartEncoder.from_pretrained(tmpdirname).to(torch_device)

        encoder_last_hidden_state_2 = encoder(inputs_dict["input_ids"], attention_mask=inputs_dict["attention_mask"])[
            0
        ]

        self.parent.assertTrue((encoder_last_hidden_state_2 - encoder_last_hidden_state).abs().max().item() < 1e-3)

        with tempfile.TemporaryDirectory() as tmpdirname:
            decoder = model.get_decoder()
            decoder.save_pretrained(tmpdirname)
            decoder = MBartDecoder.from_pretrained(tmpdirname).to(torch_device)

        last_hidden_state_2 = decoder(
            input_ids=inputs_dict["decoder_input_ids"],
            attention_mask=inputs_dict["decoder_attention_mask"],
            encoder_hidden_states=encoder_last_hidden_state,
            encoder_attention_mask=inputs_dict["attention_mask"],
        )[0]

        self.parent.assertTrue((last_hidden_state_2 - last_hidden_state).abs().max().item() < 1e-3)


@require_torch
class MBartModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (MBartModel, MBartForConditionalGeneration, MBartForSequenceClassification, MBartForQuestionAnswering)
        if is_torch_available()
        else ()
    )
    all_generative_model_classes = (MBartForConditionalGeneration,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": MBartModel,
            "fill-mask": MBartForConditionalGeneration,
            "question-answering": MBartForQuestionAnswering,
            "summarization": MBartForConditionalGeneration,
            "text-classification": MBartForSequenceClassification,
            "text-generation": MBartForCausalLM,
            "text2text-generation": MBartForConditionalGeneration,
            "translation": MBartForConditionalGeneration,
            "zero-shot": MBartForSequenceClassification,
        }
        if is_torch_available()
        else {}
    )
    is_encoder_decoder = True
    fx_compatible = False  # Fix me Michael
    test_pruning = False
    test_missing_keys = False

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
        self.model_tester = MBartModelTester(self)
        self.config_tester = ConfigTester(self, config_class=MBartConfig)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_save_load_strict(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs()
        for model_class in self.all_model_classes:
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model2, info = model_class.from_pretrained(tmpdirname, output_loading_info=True)
            self.assertEqual(info["missing_keys"], [])

    def test_decoder_model_past_with_large_inputs(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_decoder_model_past_large_inputs(*config_and_inputs)

    def test_encoder_decoder_model_standalone(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_common()
        self.model_tester.check_encoder_decoder_model_standalone(*config_and_inputs)

    # MBartForSequenceClassification does not support inputs_embeds
    def test_inputs_embeds(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in (MBartModel, MBartForConditionalGeneration, MBartForQuestionAnswering):
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

    @require_torch_fp16
    def test_generate_fp16(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs()
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        model = MBartForConditionalGeneration(config).eval().to(torch_device)
        model.half()
        model.generate(input_ids, attention_mask=attention_mask)
        model.generate(num_beams=4, do_sample=True, early_stopping=False, num_return_sequences=3)

    def test_ensure_weights_are_shared(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs()

        config.tie_word_embeddings = True
        model = MBartForConditionalGeneration(config)

        # MBart shares four weights.
        # Not an issue to not have these correctly tied for torch.load, but it is an issue for safetensors.
        self.assertEqual(
            len(
                {
                    model.get_output_embeddings().weight.data_ptr(),
                    model.get_input_embeddings().weight.data_ptr(),
                    model.base_model.decoder.embed_tokens.weight.data_ptr(),
                    model.base_model.encoder.embed_tokens.weight.data_ptr(),
                }
            ),
            1,
        )

        config.tie_word_embeddings = False
        model = MBartForConditionalGeneration(config)

        # MBart shares four weights.
        # Not an issue to not have these correctly tied for torch.load, but it is an issue for safetensors.
        self.assertEqual(
            len(
                {
                    model.get_output_embeddings().weight.data_ptr(),
                    model.get_input_embeddings().weight.data_ptr(),
                    model.base_model.decoder.embed_tokens.weight.data_ptr(),
                    model.base_model.encoder.embed_tokens.weight.data_ptr(),
                }
            ),
            2,
        )

    @unittest.skip(
        reason="This architecure has tied weights by default and there is no way to remove it, check: https://github.com/huggingface/transformers/pull/31771#issuecomment-2210915245"
    )
    def test_load_save_without_tied_weights(self):
        pass

    def test_resize_embeddings_persists_embeddings_type(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs()

        config.scale_embedding = True
        model = MBartForConditionalGeneration(config)
        old_type = type(model.model.decoder.embed_tokens)

        model.resize_token_embeddings(new_num_tokens=config.vocab_size)

        new_type = type(model.model.decoder.embed_tokens)
        self.assertIs(old_type, new_type)


def assert_tensors_close(a, b, atol=1e-12, prefix=""):
    """If tensors have different shapes, different values or a and b are not both tensors, raise a nice Assertion error."""
    if a is None and b is None:
        return True
    try:
        if torch.allclose(a, b, atol=atol):
            return True
        raise
    except Exception:
        pct_different = (torch.gt((a - b).abs(), atol)).float().mean().item()
        if a.numel() > 100:
            msg = f"tensor values are {pct_different:.1%} percent different."
        else:
            msg = f"{a} != {b}"
        if prefix:
            msg = prefix + ": " + msg
        raise AssertionError(msg)


def _long_tensor(tok_lst):
    return torch.tensor(tok_lst, dtype=torch.long, device=torch_device)


@require_torch
@require_sentencepiece
@require_tokenizers
class AbstractSeq2SeqIntegrationTest(unittest.TestCase):
    maxDiff = 1000  # longer string compare tracebacks
    checkpoint_name = None

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.checkpoint_name, use_fast=False)
        return cls

    @cached_property
    def model(self):
        """Only load the model if needed."""
        model = MBartForConditionalGeneration.from_pretrained(self.checkpoint_name).to(torch_device)
        if "cuda" in torch_device:
            model = model.half()
        return model


@require_torch
@require_sentencepiece
@require_tokenizers
class MBartEnroIntegrationTest(AbstractSeq2SeqIntegrationTest):
    checkpoint_name = "facebook/mbart-large-en-ro"
    src_text = [
        " UN Chief Says There Is No Military Solution in Syria",
        """ Secretary-General Ban Ki-moon says his response to Russia's stepped up military support for Syria is that "there is no military solution" to the nearly five-year conflict and more weapons will only worsen the violence and misery for millions of people.""",
    ]
    tgt_text = [
        "Şeful ONU declară că nu există o soluţie militară în Siria",
        "Secretarul General Ban Ki-moon declară că răspunsul său la intensificarea sprijinului militar al Rusiei"
        ' pentru Siria este că "nu există o soluţie militară" la conflictul de aproape cinci ani şi că noi arme nu vor'
        " face decât să înrăutăţească violenţa şi mizeria pentru milioane de oameni.",
    ]
    expected_src_tokens = [8274, 127873, 25916, 7, 8622, 2071, 438, 67485, 53, 187895, 23, 51712, 2, 250004]

    @slow
    def test_enro_generate_one(self):
        batch: BatchEncoding = self.tokenizer(
            ["UN Chief Says There Is No Military Solution in Syria"], return_tensors="pt"
        ).to(torch_device)
        translated_tokens = self.model.generate(**batch)
        decoded = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        self.assertEqual(self.tgt_text[0], decoded[0])
        # self.assertEqual(self.tgt_text[1], decoded[1])

    @slow
    def test_enro_generate_batch(self):
        batch: BatchEncoding = self.tokenizer(self.src_text, return_tensors="pt", padding=True, truncation=True).to(
            torch_device
        )
        translated_tokens = self.model.generate(**batch)
        decoded = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        assert self.tgt_text == decoded

    def test_mbart_enro_config(self):
        mbart_models = ["facebook/mbart-large-en-ro"]
        expected = {"scale_embedding": True, "output_past": True}
        for name in mbart_models:
            config = MBartConfig.from_pretrained(name)
            for k, v in expected.items():
                try:
                    self.assertEqual(v, getattr(config, k))
                except AssertionError as e:
                    e.args += (name, k)
                    raise

    def test_mbart_fast_forward(self):
        config = MBartConfig(
            vocab_size=99,
            d_model=24,
            encoder_layers=2,
            decoder_layers=2,
            encoder_attention_heads=2,
            decoder_attention_heads=2,
            encoder_ffn_dim=32,
            decoder_ffn_dim=32,
            max_position_embeddings=48,
            add_final_layer_norm=True,
        )
        lm_model = MBartForConditionalGeneration(config).to(torch_device)
        context = torch.tensor(
            [[71, 82, 18, 33, 46, 91, 2], [68, 34, 26, 58, 30, 2, 1]], device=torch_device, dtype=torch.long
        )
        summary = torch.tensor([[82, 71, 82, 18, 2], [58, 68, 2, 1, 1]], device=torch_device, dtype=torch.long)
        result = lm_model(input_ids=context, decoder_input_ids=summary, labels=summary)
        expected_shape = (*summary.shape, config.vocab_size)
        self.assertEqual(result.logits.shape, expected_shape)


@require_torch
@require_sentencepiece
@require_tokenizers
class MBartCC25IntegrationTest(AbstractSeq2SeqIntegrationTest):
    checkpoint_name = "facebook/mbart-large-cc25"
    src_text = [
        " UN Chief Says There Is No Military Solution in Syria",
        " I ate lunch twice yesterday",
    ]
    tgt_text = ["Şeful ONU declară că nu există o soluţie militară în Siria", "to be padded"]

    @unittest.skip(reason="This test is broken, still generates english")
    def test_cc25_generate(self):
        inputs = self.tokenizer([self.src_text[0]], return_tensors="pt").to(torch_device)
        translated_tokens = self.model.generate(
            input_ids=inputs["input_ids"].to(torch_device),
            decoder_start_token_id=self.tokenizer.lang_code_to_id["ro_RO"],
        )
        decoded = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        self.assertEqual(self.tgt_text[0], decoded[0])

    @slow
    def test_fill_mask(self):
        inputs = self.tokenizer(["One of the best <mask> I ever read!"], return_tensors="pt").to(torch_device)
        outputs = self.model.generate(
            inputs["input_ids"], decoder_start_token_id=self.tokenizer.lang_code_to_id["en_XX"], num_beams=1
        )
        prediction: str = self.tokenizer.batch_decode(
            outputs, clean_up_tokenization_spaces=True, skip_special_tokens=True
        )[0]
        self.assertEqual(prediction, "of the best books I ever read!")


class MBartStandaloneDecoderModelTester:
    def __init__(
        self,
        parent,
        vocab_size=99,
        batch_size=13,
        d_model=16,
        decoder_seq_length=7,
        is_training=True,
        is_decoder=True,
        use_attention_mask=True,
        use_cache=False,
        use_labels=True,
        decoder_start_token_id=2,
        decoder_ffn_dim=32,
        decoder_layers=2,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        max_position_embeddings=50,
        is_encoder_decoder=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.decoder_seq_length = decoder_seq_length
        # For common tests
        self.seq_length = self.decoder_seq_length
        self.is_training = is_training
        self.use_attention_mask = use_attention_mask
        self.use_labels = use_labels

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.hidden_size = d_model
        self.num_hidden_layers = decoder_layers
        self.decoder_layers = decoder_layers
        self.decoder_ffn_dim = decoder_ffn_dim
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_attention_heads = decoder_attention_heads
        self.num_attention_heads = decoder_attention_heads
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.use_cache = use_cache
        self.max_position_embeddings = max_position_embeddings
        self.is_encoder_decoder = is_encoder_decoder

        self.scope = None
        self.decoder_key_length = decoder_seq_length
        self.base_model_out_len = 2
        self.decoder_attention_idx = 1

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.decoder_seq_length], self.vocab_size)

        attention_mask = None
        if self.use_attention_mask:
            attention_mask = ids_tensor([self.batch_size, self.decoder_seq_length], vocab_size=2)

        lm_labels = None
        if self.use_labels:
            lm_labels = ids_tensor([self.batch_size, self.decoder_seq_length], self.vocab_size)

        config = MBartConfig(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            decoder_layers=self.decoder_layers,
            decoder_ffn_dim=self.decoder_ffn_dim,
            encoder_attention_heads=self.encoder_attention_heads,
            decoder_attention_heads=self.decoder_attention_heads,
            eos_token_id=self.eos_token_id,
            bos_token_id=self.bos_token_id,
            use_cache=self.use_cache,
            pad_token_id=self.pad_token_id,
            decoder_start_token_id=self.decoder_start_token_id,
            max_position_embeddings=self.max_position_embeddings,
            is_encoder_decoder=self.is_encoder_decoder,
        )

        return (
            config,
            input_ids,
            attention_mask,
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
        model = MBartDecoder(config=config).to(torch_device).eval()
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
        model = MBartDecoder(config=config).to(torch_device).eval()

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
        output_from_no_past = model(next_input_ids, attention_mask=attn_mask)["last_hidden_state"]
        output_from_past = model(next_tokens, attention_mask=attn_mask, past_key_values=past_key_values)[
            "last_hidden_state"
        ]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, next_input_ids.shape[-1] - 1, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, 0, random_slice_idx].detach()

        # test that outputs are equal for slice
        assert torch.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-3)

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


@require_torch
class MBartStandaloneDecoderModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    all_model_classes = (MBartDecoder, MBartForCausalLM) if is_torch_available() else ()
    all_generative_model_classes = (MBartForCausalLM,) if is_torch_available() else ()
    test_pruning = False
    is_encoder_decoder = False

    def setUp(
        self,
    ):
        self.model_tester = MBartStandaloneDecoderModelTester(self, is_training=False)
        self.config_tester = ConfigTester(self, config_class=MBartConfig)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_decoder_model_past(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_decoder_model_past(*config_and_inputs)

    def test_decoder_model_attn_mask_past(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_decoder_model_attention_mask_past(*config_and_inputs)

    @unittest.skip(reason="Decoder cannot retain gradients")
    def test_retain_grad_hidden_states_attentions(self):
        return

    # maybe related to https://github.com/pytorch/pytorch/issues/133166
    @unittest.skip(reason="It fails only when `dynamic=True` with {'tensor' object has no attribute '__pow__'}")
    def test_sdpa_can_compile_dynamic(self):
        pass
