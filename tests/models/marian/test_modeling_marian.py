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
"""Testing suite for the PyTorch Marian model."""

import tempfile
import unittest

from huggingface_hub.hf_api import list_models

from transformers import MarianConfig, is_torch_available
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
        AutoConfig,
        AutoModelWithLMHead,
        AutoTokenizer,
        MarianModel,
        MarianMTModel,
        TranslationPipeline,
    )
    from transformers.models.marian.convert_marian_to_pytorch import (
        ORG_NAME,
        convert_hf_name_to_opus_name,
        convert_opus_name_to_hf_name,
    )
    from transformers.models.marian.modeling_marian import (
        MarianDecoder,
        MarianEncoder,
        MarianForCausalLM,
        shift_tokens_right,
    )


def prepare_marian_inputs_dict(
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


class MarianModelTester:
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
        decoder_start_token_id=3,
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
        self.decoder_start_token_id = decoder_start_token_id

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size).clamp(
            3,
        )
        input_ids[:, -1] = self.eos_token_id  # Eos Token

        decoder_input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        config = self.get_config()
        inputs_dict = prepare_marian_inputs_dict(config, input_ids, decoder_input_ids)
        return config, inputs_dict

    def get_config(self):
        return MarianConfig(
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
            decoder_start_token_id=self.decoder_start_token_id,
        )

    def prepare_config_and_inputs_for_common(self):
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict

    def create_and_check_decoder_model_past_large_inputs(self, config, inputs_dict):
        model = MarianModel(config=config).get_decoder().to(torch_device).eval()
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
        model = MarianModel(config=config).to(torch_device).eval()
        outputs = model(**inputs_dict)

        encoder_last_hidden_state = outputs.encoder_last_hidden_state
        last_hidden_state = outputs.last_hidden_state

        with tempfile.TemporaryDirectory() as tmpdirname:
            encoder = model.get_encoder()
            encoder.save_pretrained(tmpdirname)
            encoder = MarianEncoder.from_pretrained(tmpdirname).to(torch_device)

        encoder_last_hidden_state_2 = encoder(inputs_dict["input_ids"], attention_mask=inputs_dict["attention_mask"])[
            0
        ]

        self.parent.assertTrue((encoder_last_hidden_state_2 - encoder_last_hidden_state).abs().max().item() < 1e-3)

        with tempfile.TemporaryDirectory() as tmpdirname:
            decoder = model.get_decoder()
            decoder.save_pretrained(tmpdirname)
            decoder = MarianDecoder.from_pretrained(tmpdirname).to(torch_device)

        last_hidden_state_2 = decoder(
            input_ids=inputs_dict["decoder_input_ids"],
            attention_mask=inputs_dict["decoder_attention_mask"],
            encoder_hidden_states=encoder_last_hidden_state,
            encoder_attention_mask=inputs_dict["attention_mask"],
        )[0]

        self.parent.assertTrue((last_hidden_state_2 - last_hidden_state).abs().max().item() < 1e-3)


@require_torch
class MarianModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (MarianModel, MarianMTModel) if is_torch_available() else ()
    all_generative_model_classes = (MarianMTModel,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": MarianModel,
            "summarization": MarianMTModel,
            "text-generation": MarianForCausalLM,
            "text2text-generation": MarianMTModel,
            "translation": MarianMTModel,
        }
        if is_torch_available()
        else {}
    )
    is_encoder_decoder = True
    fx_compatible = True
    test_pruning = False
    test_missing_keys = False

    def setUp(self):
        self.model_tester = MarianModelTester(self)
        self.config_tester = ConfigTester(self, config_class=MarianConfig)

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

    @require_torch_fp16
    def test_generate_fp16(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs()
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        model = MarianMTModel(config).eval().to(torch_device)
        model.half()
        model.generate(input_ids, attention_mask=attention_mask)
        model.generate(num_beams=4, do_sample=True, early_stopping=False, num_return_sequences=3)

    def test_share_encoder_decoder_embeddings(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs()

        # check if embeddings are shared by default
        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIs(model.get_encoder().embed_tokens, model.get_decoder().embed_tokens)
            self.assertIs(model.get_encoder().embed_tokens.weight, model.get_decoder().embed_tokens.weight)

        # check if embeddings are not shared when config.share_encoder_decoder_embeddings = False
        config.share_encoder_decoder_embeddings = False
        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsNot(model.get_encoder().embed_tokens, model.get_decoder().embed_tokens)
            self.assertIsNot(model.get_encoder().embed_tokens.weight, model.get_decoder().embed_tokens.weight)

        # check if a model with shared embeddings can be saved and loaded with share_encoder_decoder_embeddings = False
        config, _ = self.model_tester.prepare_config_and_inputs()
        for model_class in self.all_model_classes:
            model = model_class(config)
            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model = model_class.from_pretrained(tmpdirname, share_encoder_decoder_embeddings=False)
                self.assertIsNot(model.get_encoder().embed_tokens, model.get_decoder().embed_tokens)
                self.assertIsNot(model.get_encoder().embed_tokens.weight, model.get_decoder().embed_tokens.weight)

    def test_resize_decoder_token_embeddings(self):
        config, _ = self.model_tester.prepare_config_and_inputs()

        # check if resize_decoder_token_embeddings raises an error when embeddings are shared
        for model_class in self.all_model_classes:
            model = model_class(config)
            with self.assertRaises(ValueError):
                model.resize_decoder_token_embeddings(config.vocab_size + 1)

        # check if decoder embeddings are resized when config.share_encoder_decoder_embeddings = False
        config.share_encoder_decoder_embeddings = False
        for model_class in self.all_model_classes:
            model = model_class(config)
            model.resize_decoder_token_embeddings(config.vocab_size + 1)
            self.assertEqual(model.get_decoder().embed_tokens.weight.shape, (config.vocab_size + 1, config.d_model))

        # check if lm_head is also resized
        config, _ = self.model_tester.prepare_config_and_inputs()
        config.share_encoder_decoder_embeddings = False
        model = MarianMTModel(config)
        model.resize_decoder_token_embeddings(config.vocab_size + 1)
        self.assertEqual(model.lm_head.weight.shape, (config.vocab_size + 1, config.d_model))

    @unittest.skip
    def test_tie_word_embeddings_decoder(self):
        pass

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

    @unittest.skip(reason="No support for low_cpu_mem_usage=True.")
    def test_save_load_low_cpu_mem_usage(self):
        pass

    @unittest.skip(reason="No support for low_cpu_mem_usage=True.")
    def test_save_load_low_cpu_mem_usage_checkpoints(self):
        pass

    @unittest.skip(reason="No support for low_cpu_mem_usage=True.")
    def test_save_load_low_cpu_mem_usage_no_safetensors(self):
        pass


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


class ModelManagementTests(unittest.TestCase):
    @slow
    @require_torch
    def test_model_names(self):
        model_list = list_models()
        model_ids = [x.id for x in model_list if x.id.startswith(ORG_NAME)]
        bad_model_ids = [mid for mid in model_ids if "+" in model_ids]
        self.assertListEqual([], bad_model_ids)
        self.assertGreater(len(model_ids), 500)


@require_torch
@require_sentencepiece
@require_tokenizers
class MarianIntegrationTest(unittest.TestCase):
    src = "en"
    tgt = "de"
    src_text = [
        "I am a small frog.",
        "Now I can forget the 100 words of german that I know.",
        "Tom asked his teacher for advice.",
        "That's how I would do it.",
        "Tom really admired Mary's courage.",
        "Turn around and close your eyes.",
    ]
    expected_text = [
        "Ich bin ein kleiner Frosch.",
        "Jetzt kann ich die 100 Wörter des Deutschen vergessen, die ich kenne.",
        "Tom bat seinen Lehrer um Rat.",
        "So würde ich das machen.",
        "Tom bewunderte Marias Mut wirklich.",
        "Drehen Sie sich um und schließen Sie die Augen.",
    ]
    # ^^ actual C++ output differs slightly: (1) des Deutschen removed, (2) ""-> "O", (3) tun -> machen

    @classmethod
    def setUpClass(cls) -> None:
        cls.model_name = f"Helsinki-NLP/opus-mt-{cls.src}-{cls.tgt}"
        return cls

    @cached_property
    def tokenizer(self):
        return AutoTokenizer.from_pretrained(self.model_name)

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @cached_property
    def model(self):
        model: MarianMTModel = AutoModelWithLMHead.from_pretrained(self.model_name).to(torch_device)
        c = model.config
        self.assertListEqual(c.bad_words_ids, [[c.pad_token_id]])
        self.assertEqual(c.max_length, 512)
        self.assertEqual(c.decoder_start_token_id, c.pad_token_id)

        if torch_device == "cuda":
            return model.half()
        else:
            return model

    def _assert_generated_batch_equal_expected(self, **tokenizer_kwargs):
        generated_words = self.translate_src_text(**tokenizer_kwargs)
        self.assertListEqual(self.expected_text, generated_words)

    def translate_src_text(self, **tokenizer_kwargs):
        model_inputs = self.tokenizer(self.src_text, padding=True, return_tensors="pt", **tokenizer_kwargs).to(
            torch_device
        )
        self.assertEqual(self.model.device, model_inputs.input_ids.device)
        generated_ids = self.model.generate(
            model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            num_beams=2,
            max_length=128,
            renormalize_logits=True,  # Marian should always renormalize its logits. See #25459
        )
        generated_words = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return generated_words


@require_sentencepiece
@require_tokenizers
class TestMarian_EN_DE_More(MarianIntegrationTest):
    @slow
    def test_forward(self):
        src, tgt = ["I am a small frog"], ["Ich bin ein kleiner Frosch."]
        expected_ids = [38, 121, 14, 697, 38848, 0]

        model_inputs = self.tokenizer(src, text_target=tgt, return_tensors="pt").to(torch_device)

        self.assertListEqual(expected_ids, model_inputs.input_ids[0].tolist())

        desired_keys = {
            "input_ids",
            "attention_mask",
            "labels",
        }
        self.assertSetEqual(desired_keys, set(model_inputs.keys()))
        model_inputs["decoder_input_ids"] = shift_tokens_right(
            model_inputs.labels, self.tokenizer.pad_token_id, self.model.config.decoder_start_token_id
        )
        model_inputs["return_dict"] = True
        model_inputs["use_cache"] = False
        with torch.no_grad():
            outputs = self.model(**model_inputs)
        max_indices = outputs.logits.argmax(-1)
        self.tokenizer.batch_decode(max_indices)

    def test_unk_support(self):
        t = self.tokenizer
        ids = t(["||"], return_tensors="pt").to(torch_device).input_ids[0].tolist()
        expected = [t.unk_token_id, t.unk_token_id, t.eos_token_id]
        self.assertEqual(expected, ids)

    def test_pad_not_split(self):
        input_ids_w_pad = self.tokenizer(["I am a small frog <pad>"], return_tensors="pt").input_ids[0].tolist()
        expected_w_pad = [38, 121, 14, 697, 38848, self.tokenizer.pad_token_id, 0]  # pad
        self.assertListEqual(expected_w_pad, input_ids_w_pad)

    @slow
    def test_batch_generation_en_de(self):
        self._assert_generated_batch_equal_expected()

    def test_auto_config(self):
        config = AutoConfig.from_pretrained(self.model_name)
        self.assertIsInstance(config, MarianConfig)


@require_sentencepiece
@require_tokenizers
class TestMarian_EN_FR(MarianIntegrationTest):
    src = "en"
    tgt = "fr"
    src_text = [
        "I am a small frog.",
        "Now I can forget the 100 words of german that I know.",
    ]
    expected_text = [
        "Je suis une petite grenouille.",
        "Maintenant, je peux oublier les 100 mots d'allemand que je connais.",
    ]

    @slow
    def test_batch_generation_en_fr(self):
        self._assert_generated_batch_equal_expected()


@require_sentencepiece
@require_tokenizers
class TestMarian_FR_EN(MarianIntegrationTest):
    src = "fr"
    tgt = "en"
    src_text = [
        "Donnez moi le micro.",
        "Tom et Mary étaient assis à une table.",  # Accents
    ]
    expected_text = [
        "Give me the microphone.",
        "Tom and Mary were sitting at a table.",
    ]

    @slow
    def test_batch_generation_fr_en(self):
        self._assert_generated_batch_equal_expected()


@require_sentencepiece
@require_tokenizers
class TestMarian_RU_FR(MarianIntegrationTest):
    src = "ru"
    tgt = "fr"
    src_text = ["Он показал мне рукопись своей новой пьесы."]
    expected_text = ["Il m'a montré le manuscrit de sa nouvelle pièce."]

    @slow
    def test_batch_generation_ru_fr(self):
        self._assert_generated_batch_equal_expected()


@require_sentencepiece
@require_tokenizers
class TestMarian_MT_EN(MarianIntegrationTest):
    """Cover low resource/high perplexity setting. This breaks without adjust_logits_generation overwritten"""

    src = "mt"
    tgt = "en"
    src_text = ["Billi messu b'mod ġentili, Ġesù fejjaq raġel li kien milqut bil - marda kerha tal - ġdiem."]
    expected_text = ["Touching gently, Jesus healed a man who was affected by the sad disease of leprosy."]

    @slow
    def test_batch_generation_mt_en(self):
        self._assert_generated_batch_equal_expected()


@require_sentencepiece
@require_tokenizers
class TestMarian_en_zh(MarianIntegrationTest):
    src = "en"
    tgt = "zh"
    src_text = ["My name is Wolfgang and I live in Berlin"]
    expected_text = ["我叫沃尔夫冈 我住在柏林"]

    @slow
    def test_batch_generation_eng_zho(self):
        self._assert_generated_batch_equal_expected()


@require_sentencepiece
@require_tokenizers
class TestMarian_en_ROMANCE(MarianIntegrationTest):
    """Multilingual on target side."""

    src = "en"
    tgt = "ROMANCE"
    src_text = [
        ">>fr<< Don't spend so much time watching TV.",
        ">>pt<< Your message has been sent.",
        ">>es<< He's two years older than me.",
    ]
    expected_text = [
        "Ne passez pas autant de temps à regarder la télé.",
        "A sua mensagem foi enviada.",
        "Es dos años más viejo que yo.",
    ]

    @slow
    def test_batch_generation_en_ROMANCE_multi(self):
        self._assert_generated_batch_equal_expected()

    @slow
    @require_torch
    def test_pipeline(self):
        pipeline = TranslationPipeline(self.model, self.tokenizer, framework="pt", device=torch_device)
        output = pipeline(self.src_text)
        self.assertEqual(self.expected_text, [x["translation_text"] for x in output])


@require_sentencepiece
@require_tokenizers
class TestMarian_FI_EN_V2(MarianIntegrationTest):
    src = "fi"
    tgt = "en"
    src_text = [
        "minä tykkään kirjojen lukemisesta",
        "Pidän jalkapallon katsomisesta",
    ]
    expected_text = ["I like to read books", "I like watching football"]

    @classmethod
    def setUpClass(cls) -> None:
        cls.model_name = "hf-internal-testing/test-opus-tatoeba-fi-en-v2"
        return cls

    @slow
    def test_batch_generation_fi_en(self):
        self._assert_generated_batch_equal_expected()


@require_torch
class TestConversionUtils(unittest.TestCase):
    def test_renaming_multilingual(self):
        old_names = [
            "opus-mt-cmn+cn+yue+ze_zh+zh_cn+zh_CN+zh_HK+zh_tw+zh_TW+zh_yue+zhs+zht+zh-fi",
            "opus-mt-cmn+cn-fi",  # no group
            "opus-mt-en-de",  # standard name
            "opus-mt-en-de",  # standard name
        ]
        expected = ["opus-mt-ZH-fi", "opus-mt-cmn_cn-fi", "opus-mt-en-de", "opus-mt-en-de"]
        self.assertListEqual(expected, [convert_opus_name_to_hf_name(x) for x in old_names])

    def test_undoing_renaming(self):
        hf_names = ["opus-mt-ZH-fi", "opus-mt-cmn_cn-fi", "opus-mt-en-de", "opus-mt-en-de"]
        converted_opus_names = [convert_hf_name_to_opus_name(x) for x in hf_names]
        expected_opus_names = [
            "cmn+cn+yue+ze_zh+zh_cn+zh_CN+zh_HK+zh_tw+zh_TW+zh_yue+zhs+zht+zh-fi",
            "cmn+cn-fi",
            "en-de",  # standard name
            "en-de",
        ]
        self.assertListEqual(expected_opus_names, converted_opus_names)


class MarianStandaloneDecoderModelTester:
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
        max_position_embeddings=100,
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

        config = MarianConfig(
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
        model = MarianDecoder(config=config).to(torch_device).eval()
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
        model = MarianDecoder(config=config).to(torch_device).eval()

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
class MarianStandaloneDecoderModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    all_model_classes = (MarianDecoder, MarianForCausalLM) if is_torch_available() else ()
    all_generative_model_classes = (MarianForCausalLM,) if is_torch_available() else ()
    test_pruning = False
    is_encoder_decoder = False

    def setUp(
        self,
    ):
        self.model_tester = MarianStandaloneDecoderModelTester(self, is_training=False)
        self.config_tester = ConfigTester(self, config_class=MarianConfig)

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
