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
"""Testing suite for the PyTorch BART model."""

import copy
import tempfile
import unittest

import timeout_decorator  # noqa

from transformers import BartConfig, is_torch_available
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
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        AutoModelForSequenceClassification,
        BartForCausalLM,
        BartForConditionalGeneration,
        BartForQuestionAnswering,
        BartForSequenceClassification,
        BartModel,
        BartTokenizer,
        pipeline,
    )
    from transformers.models.bart.modeling_bart import BartDecoder, BartEncoder, shift_tokens_right


def prepare_bart_inputs_dict(
    config,
    input_ids,
    decoder_input_ids=None,
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


class BartModelTester:
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
        max_position_embeddings=20,
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
        inputs_dict = prepare_bart_inputs_dict(config, input_ids, decoder_input_ids)
        return config, inputs_dict

    def get_config(self):
        return BartConfig(
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

    def get_pipeline_config(self):
        config = self.get_config()
        config.max_position_embeddings = 100
        config.vocab_size = 300
        return config

    def prepare_config_and_inputs_for_common(self):
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict

    def create_and_check_decoder_model_past_large_inputs(self, config, inputs_dict):
        model = BartModel(config=config).get_decoder().to(torch_device).eval()
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
        model = BartModel(config=config).to(torch_device).eval()
        outputs = model(**inputs_dict)

        encoder_last_hidden_state = outputs.encoder_last_hidden_state
        last_hidden_state = outputs.last_hidden_state

        with tempfile.TemporaryDirectory() as tmpdirname:
            encoder = model.get_encoder()
            encoder.save_pretrained(tmpdirname)
            encoder = BartEncoder.from_pretrained(tmpdirname).to(torch_device)

        encoder_last_hidden_state_2 = encoder(inputs_dict["input_ids"], attention_mask=inputs_dict["attention_mask"])[
            0
        ]

        self.parent.assertTrue((encoder_last_hidden_state_2 - encoder_last_hidden_state).abs().max().item() < 1e-3)

        with tempfile.TemporaryDirectory() as tmpdirname:
            decoder = model.get_decoder()
            decoder.save_pretrained(tmpdirname)
            decoder = BartDecoder.from_pretrained(tmpdirname).to(torch_device)

        last_hidden_state_2 = decoder(
            input_ids=inputs_dict["decoder_input_ids"],
            attention_mask=inputs_dict["decoder_attention_mask"],
            encoder_hidden_states=encoder_last_hidden_state,
            encoder_attention_mask=inputs_dict["attention_mask"],
        )[0]

        self.parent.assertTrue((last_hidden_state_2 - last_hidden_state).abs().max().item() < 1e-3)


@require_torch
class BartHeadTests(unittest.TestCase):
    vocab_size = 99

    def _get_config_and_data(self):
        input_ids = torch.tensor(
            [
                [71, 82, 18, 33, 46, 91, 2],
                [68, 34, 26, 58, 30, 82, 2],
                [5, 97, 17, 39, 94, 40, 2],
                [76, 83, 94, 25, 70, 78, 2],
                [87, 59, 41, 35, 48, 66, 2],
                [55, 13, 16, 58, 5, 2, 1],  # note padding
                [64, 27, 31, 51, 12, 75, 2],
                [52, 64, 86, 17, 83, 39, 2],
                [48, 61, 9, 24, 71, 82, 2],
                [26, 1, 60, 48, 22, 13, 2],
                [21, 5, 62, 28, 14, 76, 2],
                [45, 98, 37, 86, 59, 48, 2],
                [70, 70, 50, 9, 28, 0, 2],
            ],
            dtype=torch.long,
            device=torch_device,
        )

        batch_size = input_ids.shape[0]
        config = BartConfig(
            vocab_size=self.vocab_size,
            d_model=24,
            encoder_layers=2,
            decoder_layers=2,
            encoder_attention_heads=2,
            decoder_attention_heads=2,
            encoder_ffn_dim=32,
            decoder_ffn_dim=32,
            max_position_embeddings=48,
            eos_token_id=2,
            pad_token_id=1,
            bos_token_id=0,
        )
        return config, input_ids, batch_size

    def test_sequence_classification_forward(self):
        config, input_ids, batch_size = self._get_config_and_data()
        labels = _long_tensor([2] * batch_size).to(torch_device)
        model = BartForSequenceClassification(config)
        model.to(torch_device)
        outputs = model(input_ids=input_ids, decoder_input_ids=input_ids, labels=labels)
        expected_shape = torch.Size((batch_size, config.num_labels))
        self.assertEqual(outputs["logits"].shape, expected_shape)
        self.assertIsInstance(outputs["loss"].item(), float)

    def test_question_answering_forward(self):
        config, input_ids, batch_size = self._get_config_and_data()
        sequence_labels = ids_tensor([batch_size], 2).to(torch_device)
        model = BartForQuestionAnswering(config)
        model.to(torch_device)
        outputs = model(
            input_ids=input_ids,
            start_positions=sequence_labels,
            end_positions=sequence_labels,
        )

        self.assertEqual(outputs["start_logits"].shape, input_ids.shape)
        self.assertEqual(outputs["end_logits"].shape, input_ids.shape)
        self.assertIsInstance(outputs["loss"].item(), float)

    @timeout_decorator.timeout(1)
    def test_lm_forward(self):
        config, input_ids, batch_size = self._get_config_and_data()
        lm_labels = ids_tensor([batch_size, input_ids.shape[1]], self.vocab_size).to(torch_device)
        lm_model = BartForConditionalGeneration(config)
        lm_model.to(torch_device)
        outputs = lm_model(input_ids=input_ids, labels=lm_labels)
        expected_shape = (batch_size, input_ids.shape[1], config.vocab_size)
        self.assertEqual(outputs["logits"].shape, expected_shape)
        self.assertIsInstance(outputs["loss"].item(), float)

    def test_lm_uneven_forward(self):
        config = BartConfig(
            vocab_size=self.vocab_size,
            d_model=14,
            encoder_layers=2,
            decoder_layers=2,
            encoder_attention_heads=2,
            decoder_attention_heads=2,
            encoder_ffn_dim=8,
            decoder_ffn_dim=8,
            max_position_embeddings=48,
        )
        lm_model = BartForConditionalGeneration(config).to(torch_device)
        context = torch.tensor(
            [[71, 82, 18, 33, 46, 91, 2], [68, 34, 26, 58, 30, 2, 1]], device=torch_device, dtype=torch.long
        )
        summary = torch.tensor([[82, 71, 82, 18, 2], [58, 68, 2, 1, 1]], device=torch_device, dtype=torch.long)
        outputs = lm_model(input_ids=context, decoder_input_ids=summary, labels=summary)
        expected_shape = (*summary.shape, config.vocab_size)
        self.assertEqual(outputs["logits"].shape, expected_shape)

    def test_generate_beam_search(self):
        input_ids = torch.tensor([[71, 82, 2], [68, 34, 2]], device=torch_device, dtype=torch.long)
        config = BartConfig(
            vocab_size=self.vocab_size,
            d_model=24,
            encoder_layers=2,
            decoder_layers=2,
            encoder_attention_heads=2,
            decoder_attention_heads=2,
            encoder_ffn_dim=32,
            decoder_ffn_dim=32,
            max_position_embeddings=48,
            eos_token_id=2,
            pad_token_id=1,
            bos_token_id=0,
        )
        lm_model = BartForConditionalGeneration(config).to(torch_device)
        lm_model.eval()

        max_length = 5
        generated_ids = lm_model.generate(
            input_ids.clone(),
            do_sample=True,
            num_return_sequences=1,
            num_beams=2,
            no_repeat_ngram_size=3,
            max_length=max_length,
        )
        self.assertEqual(generated_ids.shape, (input_ids.shape[0], max_length))

    def test_shift_tokens_right(self):
        input_ids = torch.tensor([[71, 82, 18, 33, 2, 1, 1], [68, 34, 26, 58, 30, 82, 2]], dtype=torch.long)
        shifted = shift_tokens_right(input_ids, 1, 2)
        n_pad_before = input_ids.eq(1).float().sum()
        n_pad_after = shifted.eq(1).float().sum()
        self.assertEqual(shifted.shape, input_ids.shape)
        self.assertEqual(n_pad_after, n_pad_before - 1)
        self.assertTrue(torch.eq(shifted[:, 0], 2).all())

    @slow
    def test_tokenization(self):
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
        examples = [" Hello world", " DomDramg"]  # need leading spaces for equality
        fairseq_results = [
            torch.tensor([0, 20920, 232, 2]),
            torch.tensor([0, 11349, 495, 4040, 571, 2]),
        ]
        for ex, desired_result in zip(examples, fairseq_results):
            bart_toks = tokenizer.encode(ex, return_tensors="pt").squeeze()
            assert_tensors_close(desired_result.long(), bart_toks, prefix=ex)

    @require_torch_fp16
    def test_generate_fp16(self):
        config, input_ids, batch_size = self._get_config_and_data()
        attention_mask = input_ids.ne(1).to(torch_device)
        model = BartForConditionalGeneration(config).eval().to(torch_device)
        model.half()
        model.generate(input_ids, attention_mask=attention_mask)
        model.generate(num_beams=4, do_sample=True, early_stopping=False, num_return_sequences=3)

    def test_dummy_inputs(self):
        config, *_ = self._get_config_and_data()
        model = BartForConditionalGeneration(config).eval().to(torch_device)
        model(**model.dummy_inputs)

    def test_resize_tokens_embeddings_more(self):
        config, input_ids, _ = self._get_config_and_data()

        def _get_embs(m):
            return (m.get_input_embeddings().weight.data.clone(), m.get_output_embeddings().weight.data.clone())

        model = BartForConditionalGeneration(config).eval().to(torch_device)
        input, output = _get_embs(model)
        self.assertTrue(torch.eq(input, output).all())
        new_vocab_size = 45
        model.resize_token_embeddings(new_vocab_size)
        input_new, output_new = _get_embs(model)
        self.assertEqual(input_new.shape, (new_vocab_size, config.d_model))
        self.assertEqual(output_new.shape, (new_vocab_size, config.d_model))
        self.assertTrue(torch.eq(input_new, output_new).all())


@require_torch
class BartModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (BartModel, BartForConditionalGeneration, BartForSequenceClassification, BartForQuestionAnswering)
        if is_torch_available()
        else ()
    )
    all_generative_model_classes = (BartForConditionalGeneration,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": BartModel,
            "fill-mask": BartForConditionalGeneration,
            "question-answering": BartForQuestionAnswering,
            "summarization": BartForConditionalGeneration,
            "text-classification": BartForSequenceClassification,
            "text-generation": BartForCausalLM,
            "text2text-generation": BartForConditionalGeneration,
            "translation": BartForConditionalGeneration,
            "zero-shot": BartForSequenceClassification,
        }
        if is_torch_available()
        else {}
    )
    is_encoder_decoder = True
    fx_compatible = False  # Fix me Michael
    test_pruning = False

    def setUp(self):
        self.model_tester = BartModelTester(self)
        self.config_tester = ConfigTester(self, config_class=BartConfig)

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

    # BartForSequenceClassification does not support inputs_embeds
    def test_inputs_embeds(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in (BartModel, BartForConditionalGeneration, BartForQuestionAnswering):
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
        model = BartForConditionalGeneration(config).eval().to(torch_device)
        model.half()
        model.generate(input_ids, attention_mask=attention_mask)
        model.generate(num_beams=4, do_sample=True, early_stopping=False, num_return_sequences=3)

    @unittest.skip(
        reason="This architecure has tied weights by default and there is no way to remove it, check: https://github.com/huggingface/transformers/pull/31771#issuecomment-2210915245"
    )
    def test_load_save_without_tied_weights(self):
        pass

    def test_resize_embeddings_persists_embeddings_type(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs()

        config.scale_embedding = True
        model = BartForConditionalGeneration(config)
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
@slow
class FastIntegrationTests(unittest.TestCase):
    """These tests are useful for debugging since they operate on a model with 1 encoder layer and 1 decoder layer."""

    @cached_property
    def tok(self):
        return BartTokenizer.from_pretrained("facebook/bart-large")

    @cached_property
    def xsum_1_1_model(self):
        return BartForConditionalGeneration.from_pretrained("sshleifer/distilbart-xsum-1-1")

    def test_xsum_1_1_generation(self):
        hf = self.xsum_1_1_model
        tok = self.tok
        ARTICLE = (
            "The Palestinian Authority officially became the 123rd member of the International Criminal Court on"
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
            " 2002 to prosecute genocide, crimes against humanity and war crimes."
        )
        EXPECTED = (
            " The International Criminal Court (ICC) has announced that it has been announced by the International"
            " Criminal court."
        )

        dct = tok(ARTICLE, return_tensors="pt")
        generated_ids = hf.generate(**dct, num_beams=4)
        result = tok.batch_decode(generated_ids, skip_special_tokens=True)[0]
        assert EXPECTED == result

    def test_xsum_1_1_batch_generation(self):
        # test batch

        batch = self.tok(
            [
                "The Palestinian Authority officially became the 123rd member of the International Criminal Court on"
                " Wednesday, a step that gives the court jurisdiction over alleged crimes in Palestinian territories."
                " The formal accession was marked with a ceremony at The Hague, in the Netherlands, where the court is"
                " based. The Palestinians signed the ICC's founding Rome Statute in January, when they also accepted"
                ' its jurisdiction over alleged crimes committed "in the occupied Palestinian territory, including'
                ' East Jerusalem, since June 13, 2014." Later that month, the ICC opened a preliminary examination'
                " into the situation in Palestinian territories, paving the way for possible war crimes investigations"
                " against Israelis. As members of the court, Palestinians may be subject to counter-charges as well."
                " Israel and the United States, neither of which is an ICC member, opposed the Palestinians' efforts"
                " to join the body. But Palestinian Foreign Minister Riad al-Malki, speaking at Wednesday's ceremony,"
                ' said it was a move toward greater justice. "As Palestine formally becomes a State Party to the Rome'
                ' Statute today, the world is also a step closer to ending a long era of impunity and injustice," he'
                ' said, according to an ICC news release. "Indeed, today brings us closer to our shared goals of'
                ' justice and peace." Judge Kuniko Ozaki, a vice president of the ICC, said acceding to the treaty was'
                ' just the first step for the Palestinians. "As the Rome Statute today enters into force for the State'
                " of Palestine, Palestine acquires all the rights as well as responsibilities that come with being a"
                ' State Party to the Statute. These are substantive commitments, which cannot be taken lightly," she'
                ' said. Rights group Human Rights Watch welcomed the development. "Governments seeking to penalize'
                " Palestine for joining the ICC should immediately end their pressure, and countries that support"
                " universal acceptance of the court's treaty should speak out to welcome its membership,\" said"
                " Balkees Jarrah, international justice counsel for the group. \"What's objectionable is the attempts"
                " to undermine international justice, not Palestine's decision to join a treaty to which over 100"
                ' countries around the world are members." In January, when the preliminary ICC examination was'
                " opened, Israeli Prime Minister Benjamin Netanyahu described it as an outrage, saying the court was"
                ' overstepping its boundaries. The United States also said it "strongly" disagreed with the court\'s'
                ' decision. "As we have said repeatedly, we do not believe that Palestine is a state and therefore we'
                ' do not believe that it is eligible to join the ICC," the State Department said in a statement. It'
                ' urged the warring sides to resolve their differences through direct negotiations. "We will continue'
                ' to oppose actions against Israel at the ICC as counterproductive to the cause of peace," it said.'
                " But the ICC begs to differ with the definition of a state for its purposes and refers to the"
                ' territories as "Palestine." While a preliminary examination is not a formal investigation, it allows'
                " the court to review evidence and determine whether to investigate suspects on both sides. Prosecutor"
                ' Fatou Bensouda said her office would "conduct its analysis in full independence and impartiality."'
                " The war between Israel and Hamas militants in Gaza last summer left more than 2,000 people dead. The"
                " inquiry will include alleged war crimes committed since June. The International Criminal Court was"
                " set up in 2002 to prosecute genocide, crimes against humanity and war crimes.",
                "The French prosecutor leading an investigation into the crash of Germanwings Flight 9525 insisted"
                " Wednesday that he was not aware of any video footage from on board the plane. Marseille prosecutor"
                ' Brice Robin told CNN that "so far no videos were used in the crash investigation." He added, "A'
                " person who has such a video needs to immediately give it to the investigators.\" Robin's comments"
                " follow claims by two magazines, German daily Bild and French Paris Match, of a cell phone video"
                " showing the harrowing final seconds from on board Germanwings Flight 9525 as it crashed into the"
                " French Alps. All 150 on board were killed. Paris Match and Bild reported that the video was"
                " recovered from a phone at the wreckage site. The two publications described the supposed video, but"
                " did not post it on their websites. The publications said that they watched the video, which was"
                " found by a source close to the investigation. \"One can hear cries of 'My God' in several"
                ' languages," Paris Match reported. "Metallic banging can also be heard more than three times, perhaps'
                " of the pilot trying to open the cockpit door with a heavy object.  Towards the end, after a heavy"
                ' shake, stronger than the others, the screaming intensifies. Then nothing." "It is a very disturbing'
                " scene,\" said Julian Reichelt, editor-in-chief of Bild online. An official with France's accident"
                " investigation agency, the BEA, said the agency is not aware of any such video. Lt. Col. Jean-Marc"
                " Menichini, a French Gendarmerie spokesman in charge of communications on rescue efforts around the"
                ' Germanwings crash site, told CNN that the reports were "completely wrong" and "unwarranted." Cell'
                ' phones have been collected at the site, he said, but that they "hadn\'t been exploited yet."'
                " Menichini said he believed the cell phones would need to be sent to the Criminal Research Institute"
                " in Rosny sous-Bois, near Paris, in order to be analyzed by specialized technicians working"
                " hand-in-hand with investigators. But none of the cell phones found so far have been sent to the"
                " institute, Menichini said. Asked whether staff involved in the search could have leaked a memory"
                ' card to the media, Menichini answered with a categorical "no." Reichelt told "Erin Burnett:'
                ' Outfront" that he had watched the video and stood by the report, saying Bild and Paris Match are'
                ' "very confident" that the clip is real. He noted that investigators only revealed they\'d recovered'
                ' cell phones from the crash site after Bild and Paris Match published their reports. "That is'
                " something we did not know before. ... Overall we can say many things of the investigation weren't"
                ' revealed by the investigation at the beginning," he said. What was mental state of Germanwings'
                " co-pilot? German airline Lufthansa confirmed Tuesday that co-pilot Andreas Lubitz had battled"
                " depression years before he took the controls of Germanwings Flight 9525, which he's accused of"
                " deliberately crashing last week in the French Alps. Lubitz told his Lufthansa flight training school"
                ' in 2009 that he had a "previous episode of severe depression," the airline said Tuesday. Email'
                " correspondence between Lubitz and the school discovered in an internal investigation, Lufthansa"
                " said, included medical documents he submitted in connection with resuming his flight training. The"
                " announcement indicates that Lufthansa, the parent company of Germanwings, knew of Lubitz's battle"
                " with depression, allowed him to continue training and ultimately put him in the cockpit. Lufthansa,"
                " whose CEO Carsten Spohr previously said Lubitz was 100% fit to fly, described its statement Tuesday"
                ' as a "swift and seamless clarification" and said it was sharing the information and documents --'
                " including training and medical records -- with public prosecutors. Spohr traveled to the crash site"
                " Wednesday, where recovery teams have been working for the past week to recover human remains and"
                " plane debris scattered across a steep mountainside. He saw the crisis center set up in"
                " Seyne-les-Alpes, laid a wreath in the village of Le Vernet, closer to the crash site, where grieving"
                " families have left flowers at a simple stone memorial. Menichini told CNN late Tuesday that no"
                " visible human remains were left at the site but recovery teams would keep searching. French"
                " President Francois Hollande, speaking Tuesday, said that it should be possible to identify all the"
                " victims using DNA analysis by the end of the week, sooner than authorities had previously suggested."
                " In the meantime, the recovery of the victims' personal belongings will start Wednesday, Menichini"
                " said. Among those personal belongings could be more cell phones belonging to the 144 passengers and"
                " six crew on board. Check out the latest from our correspondents . The details about Lubitz's"
                " correspondence with the flight school during his training were among several developments as"
                " investigators continued to delve into what caused the crash and Lubitz's possible motive for"
                " downing the jet. A Lufthansa spokesperson told CNN on Tuesday that Lubitz had a valid medical"
                ' certificate, had passed all his examinations and "held all the licenses required." Earlier, a'
                " spokesman for the prosecutor's office in Dusseldorf, Christoph Kumpa, said medical records reveal"
                " Lubitz suffered from suicidal tendencies at some point before his aviation career and underwent"
                " psychotherapy before he got his pilot's license. Kumpa emphasized there's no evidence suggesting"
                " Lubitz was suicidal or acting aggressively before the crash. Investigators are looking into whether"
                " Lubitz feared his medical condition would cause him to lose his pilot's license, a European"
                ' government official briefed on the investigation told CNN on Tuesday. While flying was "a big part'
                " of his life,\" the source said, it's only one theory being considered. Another source, a law"
                " enforcement official briefed on the investigation, also told CNN that authorities believe the"
                " primary motive for Lubitz to bring down the plane was that he feared he would not be allowed to fly"
                " because of his medical problems. Lubitz's girlfriend told investigators he had seen an eye doctor"
                " and a neuropsychologist, both of whom deemed him unfit to work recently and concluded he had"
                " psychological issues, the European government official said. But no matter what details emerge about"
                " his previous mental health struggles, there's more to the story, said Brian Russell, a forensic"
                ' psychologist. "Psychology can explain why somebody would turn rage inward on themselves about the'
                " fact that maybe they weren't going to keep doing their job and they're upset about that and so"
                ' they\'re suicidal," he said. "But there is no mental illness that explains why somebody then feels'
                " entitled to also take that rage and turn it outward on 149 other people who had nothing to do with"
                " the person's problems.\" Germanwings crash compensation: What we know . Who was the captain of"
                " Germanwings Flight 9525? CNN's Margot Haddad reported from Marseille and Pamela Brown from"
                " Dusseldorf, while Laura Smith-Spark wrote from London. CNN's Frederik Pleitgen, Pamela Boykoff,"
                " Antonia Mortensen, Sandrine Amiel and Anna-Maja Rappard contributed to this report.",
            ],
            return_tensors="pt",
            padding="longest",
            truncation=True,
        )
        generated_ids = self.xsum_1_1_model.generate(**batch, num_beams=4)
        result = self.tok.batch_decode(generated_ids, skip_special_tokens=True)
        assert (
            result[0]
            == " The International Criminal Court (ICC) has announced that it has been announced by the International"
            " Criminal court."
        )
        assert (
            result[1]
            == " An investigation into the crash that killed at least 10 people in the French capital has been"
            " released by the French police investigating the crash."
        )

    def test_encoder_equiv(self):
        # test batch

        batch = self.tok(
            [
                "The Palestinian Authority officially became the 123rd member of the International Criminal Court on"
                " Wednesday, a step that gives the court jurisdiction over alleged crimes in Palestinian territories."
                " The formal accession was marked with a ceremony at The Hague, in the Netherlands, where the court is"
                " based. The Palestinians signed the ICC's founding Rome Statute in January, when they also accepted"
                ' its jurisdiction over alleged crimes committed "in the occupied Palestinian territory, including'
                ' East Jerusalem, since June 13, 2014." Later that month, the ICC opened a preliminary examination'
                " into the situation in Palestinian territories, paving the way for possible war crimes investigations"
                " against Israelis. As members of the court, Palestinians may be subject to counter-charges as well."
                " Israel and the United States, neither of which is an ICC member, opposed the Palestinians' efforts"
                " to join the body. But Palestinian Foreign Minister Riad al-Malki, speaking at Wednesday's ceremony,"
                ' said it was a move toward greater justice. "As Palestine formally becomes a State Party to the Rome'
                ' Statute today, the world is also a step closer to ending a long era of impunity and injustice," he'
                ' said, according to an ICC news release. "Indeed, today brings us closer to our shared goals of'
                ' justice and peace." Judge Kuniko Ozaki, a vice president of the ICC, said acceding to the treaty was'
                ' just the first step for the Palestinians. "As the Rome Statute today enters into force for the State'
                " of Palestine, Palestine acquires all the rights as well as responsibilities that come with being a"
                ' State Party to the Statute. These are substantive commitments, which cannot be taken lightly," she'
                ' said. Rights group Human Rights Watch welcomed the development. "Governments seeking to penalize'
                " Palestine for joining the ICC should immediately end their pressure, and countries that support"
                " universal acceptance of the court's treaty should speak out to welcome its membership,\" said"
                " Balkees Jarrah, international justice counsel for the group. \"What's objectionable is the attempts"
                " to undermine international justice, not Palestine's decision to join a treaty to which over 100"
                ' countries around the world are members." In January, when the preliminary ICC examination was'
                " opened, Israeli Prime Minister Benjamin Netanyahu described it as an outrage, saying the court was"
                ' overstepping its boundaries. The United States also said it "strongly" disagreed with the court\'s'
                ' decision. "As we have said repeatedly, we do not believe that Palestine is a state and therefore we'
                ' do not believe that it is eligible to join the ICC," the State Department said in a statement. It'
                ' urged the warring sides to resolve their differences through direct negotiations. "We will continue'
                ' to oppose actions against Israel at the ICC as counterproductive to the cause of peace," it said.'
                " But the ICC begs to differ with the definition of a state for its purposes and refers to the"
                ' territories as "Palestine." While a preliminary examination is not a formal investigation, it allows'
                " the court to review evidence and determine whether to investigate suspects on both sides. Prosecutor"
                ' Fatou Bensouda said her office would "conduct its analysis in full independence and impartiality."'
                " The war between Israel and Hamas militants in Gaza last summer left more than 2,000 people dead. The"
                " inquiry will include alleged war crimes committed since June. The International Criminal Court was"
                " set up in 2002 to prosecute genocide, crimes against humanity and war crimes.",
                "The French prosecutor leading an investigation into the crash of Germanwings Flight 9525 insisted"
                " Wednesday that he was not aware of any video footage from on board the plane. Marseille prosecutor"
                ' Brice Robin told CNN that "so far no videos were used in the crash investigation." He added, "A'
                " person who has such a video needs to immediately give it to the investigators.\" Robin's comments"
                " follow claims by two magazines, German daily Bild and French Paris Match, of a cell phone video"
                " showing the harrowing final seconds from on board Germanwings Flight 9525 as it crashed into the"
                " French Alps. All 150 on board were killed. Paris Match and Bild reported that the video was"
                " recovered from a phone at the wreckage site. The two publications described the supposed video, but"
                " did not post it on their websites. The publications said that they watched the video, which was"
                " found by a source close to the investigation. \"One can hear cries of 'My God' in several"
                ' languages," Paris Match reported. "Metallic banging can also be heard more than three times, perhaps'
                " of the pilot trying to open the cockpit door with a heavy object.  Towards the end, after a heavy"
                ' shake, stronger than the others, the screaming intensifies. Then nothing." "It is a very disturbing'
                " scene,\" said Julian Reichelt, editor-in-chief of Bild online. An official with France's accident"
                " investigation agency, the BEA, said the agency is not aware of any such video. Lt. Col. Jean-Marc"
                " Menichini, a French Gendarmerie spokesman in charge of communications on rescue efforts around the"
                ' Germanwings crash site, told CNN that the reports were "completely wrong" and "unwarranted." Cell'
                ' phones have been collected at the site, he said, but that they "hadn\'t been exploited yet."'
                " Menichini said he believed the cell phones would need to be sent to the Criminal Research Institute"
                " in Rosny sous-Bois, near Paris, in order to be analyzed by specialized technicians working"
                " hand-in-hand with investigators. But none of the cell phones found so far have been sent to the"
                " institute, Menichini said. Asked whether staff involved in the search could have leaked a memory"
                ' card to the media, Menichini answered with a categorical "no." Reichelt told "Erin Burnett:'
                ' Outfront" that he had watched the video and stood by the report, saying Bild and Paris Match are'
                ' "very confident" that the clip is real. He noted that investigators only revealed they\'d recovered'
                ' cell phones from the crash site after Bild and Paris Match published their reports. "That is'
                " something we did not know before. ... Overall we can say many things of the investigation weren't"
                ' revealed by the investigation at the beginning," he said. What was mental state of Germanwings'
                " co-pilot? German airline Lufthansa confirmed Tuesday that co-pilot Andreas Lubitz had battled"
                " depression years before he took the controls of Germanwings Flight 9525, which he's accused of"
                " deliberately crashing last week in the French Alps. Lubitz told his Lufthansa flight training school"
                ' in 2009 that he had a "previous episode of severe depression," the airline said Tuesday. Email'
                " correspondence between Lubitz and the school discovered in an internal investigation, Lufthansa"
                " said, included medical documents he submitted in connection with resuming his flight training. The"
                " announcement indicates that Lufthansa, the parent company of Germanwings, knew of Lubitz's battle"
                " with depression, allowed him to continue training and ultimately put him in the cockpit. Lufthansa,"
                " whose CEO Carsten Spohr previously said Lubitz was 100% fit to fly, described its statement Tuesday"
                ' as a "swift and seamless clarification" and said it was sharing the information and documents --'
                " including training and medical records -- with public prosecutors. Spohr traveled to the crash site"
                " Wednesday, where recovery teams have been working for the past week to recover human remains and"
                " plane debris scattered across a steep mountainside. He saw the crisis center set up in"
                " Seyne-les-Alpes, laid a wreath in the village of Le Vernet, closer to the crash site, where grieving"
                " families have left flowers at a simple stone memorial. Menichini told CNN late Tuesday that no"
                " visible human remains were left at the site but recovery teams would keep searching. French"
                " President Francois Hollande, speaking Tuesday, said that it should be possible to identify all the"
                " victims using DNA analysis by the end of the week, sooner than authorities had previously suggested."
                " In the meantime, the recovery of the victims' personal belongings will start Wednesday, Menichini"
                " said. Among those personal belongings could be more cell phones belonging to the 144 passengers and"
                " six crew on board. Check out the latest from our correspondents . The details about Lubitz's"
                " correspondence with the flight school during his training were among several developments as"
                " investigators continued to delve into what caused the crash and Lubitz's possible motive for"
                " downing the jet. A Lufthansa spokesperson told CNN on Tuesday that Lubitz had a valid medical"
                ' certificate, had passed all his examinations and "held all the licenses required." Earlier, a'
                " spokesman for the prosecutor's office in Dusseldorf, Christoph Kumpa, said medical records reveal"
                " Lubitz suffered from suicidal tendencies at some point before his aviation career and underwent"
                " psychotherapy before he got his pilot's license. Kumpa emphasized there's no evidence suggesting"
                " Lubitz was suicidal or acting aggressively before the crash. Investigators are looking into whether"
                " Lubitz feared his medical condition would cause him to lose his pilot's license, a European"
                ' government official briefed on the investigation told CNN on Tuesday. While flying was "a big part'
                " of his life,\" the source said, it's only one theory being considered. Another source, a law"
                " enforcement official briefed on the investigation, also told CNN that authorities believe the"
                " primary motive for Lubitz to bring down the plane was that he feared he would not be allowed to fly"
                " because of his medical problems. Lubitz's girlfriend told investigators he had seen an eye doctor"
                " and a neuropsychologist, both of whom deemed him unfit to work recently and concluded he had"
                " psychological issues, the European government official said. But no matter what details emerge about"
                " his previous mental health struggles, there's more to the story, said Brian Russell, a forensic"
                ' psychologist. "Psychology can explain why somebody would turn rage inward on themselves about the'
                " fact that maybe they weren't going to keep doing their job and they're upset about that and so"
                ' they\'re suicidal," he said. "But there is no mental illness that explains why somebody then feels'
                " entitled to also take that rage and turn it outward on 149 other people who had nothing to do with"
                " the person's problems.\" Germanwings crash compensation: What we know . Who was the captain of"
                " Germanwings Flight 9525? CNN's Margot Haddad reported from Marseille and Pamela Brown from"
                " Dusseldorf, while Laura Smith-Spark wrote from London. CNN's Frederik Pleitgen, Pamela Boykoff,"
                " Antonia Mortensen, Sandrine Amiel and Anna-Maja Rappard contributed to this report.",
            ],
            return_tensors="pt",
            padding="longest",
            truncation=True,
        )
        features = self.xsum_1_1_model.get_encoder()(**batch).last_hidden_state
        expected = [[-0.0828, -0.0251, -0.0674], [0.1277, 0.3311, -0.0255], [0.2613, -0.0840, -0.2763]]
        assert_tensors_close(features[0, :3, :3], torch.tensor(expected), atol=1e-3)


@require_torch
@require_sentencepiece
@require_tokenizers
class BartModelIntegrationTests(unittest.TestCase):
    @cached_property
    def default_tokenizer(self):
        return BartTokenizer.from_pretrained("facebook/bart-large")

    @slow
    def test_inference_no_head(self):
        model = BartModel.from_pretrained("facebook/bart-large").to(torch_device)
        input_ids = _long_tensor([[0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2]])
        attention_mask = input_ids.ne(model.config.pad_token_id)
        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        expected_shape = torch.Size((1, 11, 1024))
        self.assertEqual(output.shape, expected_shape)
        expected_slice = torch.tensor(
            [[0.7144, 0.8143, -1.2813], [0.7144, 0.8143, -1.2813], [-0.0467, 2.5911, -2.1845]], device=torch_device
        )
        torch.testing.assert_close(output[:, :3, :3], expected_slice, rtol=1e-3, atol=1e-3)

    @slow
    def test_base_mask_filling(self):
        pbase = pipeline(task="fill-mask", model="facebook/bart-base")
        src_text = [" I went to the <mask>."]
        results = [x["token_str"] for x in pbase(src_text)]
        assert " bathroom" in results

    @slow
    def test_large_mask_filling(self):
        plarge = pipeline(task="fill-mask", model="facebook/bart-large")
        src_text = [" I went to the <mask>."]
        results = [x["token_str"] for x in plarge(src_text)]
        expected_results = [" bathroom", " gym", " wrong", " movies", " hospital"]
        self.assertListEqual(results, expected_results)

    @slow
    def test_mnli_inference(self):
        example_b = [0, 31414, 232, 328, 740, 1140, 69, 46078, 1588, 2, 1]
        input_ids = _long_tensor([[0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2], example_b])

        model = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli").to(
            torch_device
        )  # eval called in from_pre
        attention_mask = input_ids.ne(model.config.pad_token_id)
        # Test that model hasn't changed
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        batched_logits = outputs.logits
        expected_shape = torch.Size((2, 3))
        self.assertEqual(batched_logits.shape, expected_shape)
        expected_slice = torch.tensor([[0.1907, 1.4342, -1.0289]], device=torch_device)
        logits_arr = batched_logits[0].detach()

        # Test that padding does not change results
        input_ids_no_pad = _long_tensor([example_b[:-1]])
        attention_mask_no_pad = input_ids_no_pad.ne(model.config.pad_token_id)

        with torch.no_grad():
            logits2 = model(input_ids=input_ids_no_pad, attention_mask=attention_mask_no_pad).logits.squeeze()
        assert_tensors_close(batched_logits[1], logits2, atol=1e-3)
        assert_tensors_close(expected_slice, logits_arr, atol=1e-3)

    @slow
    def test_xsum_summarization_same_as_fairseq(self):
        model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-xsum").to(torch_device)
        tok = self.default_tokenizer

        PGE_ARTICLE = """ PG&E stated it scheduled the blackouts in response to forecasts for high winds amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."""

        EXPECTED_SUMMARY = (
            "California's largest power company has begun shutting off electricity to thousands of customers in the"
            " state."
        )
        dct = tok.batch_encode_plus(
            [PGE_ARTICLE],
            max_length=1024,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(torch_device)

        hypotheses_batch = model.generate(
            input_ids=dct["input_ids"],
            attention_mask=dct["attention_mask"],
            num_beams=2,
            max_length=62,
            min_length=11,
            length_penalty=1.0,
            no_repeat_ngram_size=3,
            early_stopping=True,
            decoder_start_token_id=model.config.eos_token_id,
        )

        decoded = tok.batch_decode(
            hypotheses_batch,
            skip_special_tokens=True,
        )
        self.assertEqual(EXPECTED_SUMMARY, decoded[0])

    def test_xsum_config_generation_params(self):
        config = BartConfig.from_pretrained("facebook/bart-large-xsum")
        expected_params = {"num_beams": 6, "do_sample": False, "early_stopping": True, "length_penalty": 1.0}
        config_params = {k: getattr(config, k, "MISSING") for k, v in expected_params.items()}
        self.assertDictEqual(expected_params, config_params)

    @slow
    def test_cnn_summarization_same_as_fairseq(self):
        hf = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn").to(torch_device)
        tok = BartTokenizer.from_pretrained("facebook/bart-large")

        FRANCE_ARTICLE = (  # @noq
            " Marseille, France (CNN)The French prosecutor leading an investigation into the crash of Germanwings"
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
            " (CNN)The Palestinian Authority officially became the 123rd member of the International Criminal Court on"
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

        # The below article tests that we don't add any hypotheses outside of the top n_beams
        IRAN_ARTICLE = (
            " (CNN)The United States and its negotiating partners reached a very strong framework agreement with Iran"
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

        dct = tok.batch_encode_plus(
            [FRANCE_ARTICLE, SHORTER_ARTICLE, IRAN_ARTICLE, ARTICLE_SUBWAY],
            max_length=1024,
            padding="max_length",
            truncation_strategy="only_first",
            truncation=True,
            return_tensors="pt",
        )

        self.assertEqual(1024, dct["input_ids"].shape[1])
        hypotheses_batch = hf.generate(
            input_ids=dct["input_ids"].to(torch_device),
            attention_mask=dct["attention_mask"].to(torch_device),
            num_beams=2,
        )
        assert hypotheses_batch[:, 1].eq(0).all().item()

        EXPECTED = [
            "A French prosecutor says he is not aware of any video footage from on board the plane. Two German "
            "magazines claim to have found a cell phone video showing the crash. The publications say they watched "
            "the video, which was found by a source close to the investigation. All 150 on board Germanwings Flight "
            "9525 were killed.",
            "Palestinian Authority becomes 123rd member of the International Criminal Court. The move gives the court "
            "jurisdiction over alleged crimes in Palestinian territories. Israel and the United States opposed the "
            "Palestinians' efforts to join the body. But Palestinian Foreign Minister Riad al-Malki said it was a "
            "move toward greater justice.",
            "U.S. and its negotiating partners reached a strong framework agreement with Iran. Peter Bergen: The "
            "debate that has already begun will likely result in more heat than light. He says critics have made "
            "dubious assumptions and doubtful assertions. Bergen says the goal was to block Iran from building a "
            "nuclear weapon.",
            "Liana Barrientos, 39, has been married 10 times, sometimes within two weeks of each other. Prosecutors "
            "say the marriages were part of an immigration scam. She pleaded not guilty at State Supreme Court in the "
            "Bronx on Friday. If convicted, she faces up to four years in prison.",
        ]

        generated_summaries = tok.batch_decode(
            hypotheses_batch.tolist(), clean_up_tokenization_spaces=True, skip_special_tokens=True
        )
        assert generated_summaries == EXPECTED

    @slow
    def test_contrastive_search_bart(self):
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
        bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn").to(torch_device)
        input_ids = bart_tokenizer(
            article, add_special_tokens=False, truncation=True, max_length=512, return_tensors="pt"
        ).input_ids.to(torch_device)

        outputs = bart_model.generate(input_ids, penalty_alpha=0.5, top_k=5, max_length=64, num_beams=1)
        generated_text = bart_tokenizer.batch_decode(outputs, skip_special_tokens=True)

        self.assertListEqual(
            generated_text,
            [
                "Liana Barrientos, 39, pleaded not guilty to charges related to false marriage statements. "
                "Prosecutors say she married at least 10 times, sometimes within two weeks of each other. She is "
                "accused of being part of an immigration scam to get permanent residency. If convicted, she faces up "
                "to four years in"
            ],
        )

    @slow
    def test_decoder_attention_mask(self):
        model = BartForConditionalGeneration.from_pretrained("facebook/bart-large", forced_bos_token_id=0).to(
            torch_device
        )
        tokenizer = self.default_tokenizer
        sentence = "UN Chief Says There Is No <mask> in Syria"
        input_ids = tokenizer(sentence, return_tensors="pt").input_ids.to(torch_device)
        padding_size = 3
        decoder_input_ids = torch.tensor(
            [
                [model.config.decoder_start_token_id]
                + padding_size * [model.config.pad_token_id]
                + [model.config.bos_token_id]
            ],
            dtype=torch.long,
            device=torch_device,
        )
        decoder_attention_mask = torch.where(decoder_input_ids == model.config.pad_token_id, 0, 1).to(torch_device)
        generated_ids = model.generate(
            input_ids=input_ids,
            use_cache=False,
            max_new_tokens=20,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
        )
        generated_sentence = tokenizer.batch_decode(generated_ids)[0]
        expected_sentence = "</s><pad><pad><pad><s>UN Chief Says There Is No Plan B for Peace in Syria</s>"
        self.assertEqual(generated_sentence, expected_sentence)


class BartStandaloneDecoderModelTester:
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
        max_position_embeddings=30,
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

        config = BartConfig(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            encoder_layers=self.decoder_layers,
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

    def prepare_config_and_inputs_for_decoder(self):
        (
            config,
            input_ids,
            attention_mask,
            lm_labels,
        ) = self.prepare_config_and_inputs()

        encoder_hidden_states = floats_tensor([self.batch_size, self.decoder_seq_length, self.hidden_size])
        encoder_attention_mask = ids_tensor([self.batch_size, self.decoder_seq_length], vocab_size=2)

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
        model = BartDecoder(config=config).to(torch_device).eval()
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
        model = BartDecoder(config=config).to(torch_device).eval()

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
class BartStandaloneDecoderModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    all_model_classes = (BartDecoder, BartForCausalLM) if is_torch_available() else ()
    all_generative_model_classes = (BartForCausalLM,) if is_torch_available() else ()
    fx_comptatible = True
    test_pruning = False
    is_encoder_decoder = False
    test_missing_keys = False

    def setUp(
        self,
    ):
        self.model_tester = BartStandaloneDecoderModelTester(self, is_training=False)
        self.config_tester = ConfigTester(self, config_class=BartConfig)

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

    @unittest.skip
    def test_save_load_fast_init_from_base(self):
        pass
