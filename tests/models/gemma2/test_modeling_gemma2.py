# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Gemma2 model."""

import unittest

from packaging import version
from parameterized import parameterized
from pytest import mark

from transformers import AutoModelForCausalLM, AutoTokenizer, Gemma2Config, HybridCache, is_torch_available, pipeline
from transformers.generation.configuration_utils import GenerationConfig
from transformers.testing_utils import (
    require_flash_attn,
    require_read_token,
    require_torch,
    require_torch_gpu,
    slow,
    tooslow,
    torch_device,
)

from ...models.gemma.test_modeling_gemma import GemmaModelTest, GemmaModelTester
from ...test_configuration_common import ConfigTester


if is_torch_available():
    import torch

    from transformers import (
        Gemma2ForCausalLM,
        Gemma2ForSequenceClassification,
        Gemma2ForTokenClassification,
        Gemma2Model,
    )


class Gemma2ModelTester(GemmaModelTester):
    if is_torch_available():
        config_class = Gemma2Config
        model_class = Gemma2Model
        for_causal_lm_class = Gemma2ForCausalLM
        for_sequence_class = Gemma2ForSequenceClassification
        for_token_class = Gemma2ForTokenClassification


@require_torch
class Gemma2ModelTest(GemmaModelTest, unittest.TestCase):
    all_model_classes = (
        (Gemma2Model, Gemma2ForCausalLM, Gemma2ForSequenceClassification, Gemma2ForTokenClassification)
        if is_torch_available()
        else ()
    )
    all_generative_model_classes = (Gemma2ForCausalLM,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": Gemma2Model,
            "text-classification": Gemma2ForSequenceClassification,
            "token-classification": Gemma2ForTokenClassification,
            "text-generation": Gemma2ForCausalLM,
            "zero-shot": Gemma2ForSequenceClassification,
        }
        if is_torch_available()
        else {}
    )
    test_headmasking = False
    test_pruning = False
    _is_stateful = True
    model_split_percents = [0.5, 0.6]

    def setUp(self):
        self.model_tester = Gemma2ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Gemma2Config, hidden_size=37)

    @unittest.skip("Failing because of unique cache (HybridCache)")
    def test_model_outputs_equivalence(self, **kwargs):
        pass

    @unittest.skip("Gemma2's forcefully disables sdpa due to softcapping")
    def test_sdpa_can_dispatch_non_composite_models(self):
        pass

    @parameterized.expand([("float16",), ("bfloat16",), ("float32",)])
    @unittest.skip("Gemma2's eager attn/sdpa attn outputs are expected to be different")
    def test_eager_matches_sdpa_inference(self):
        pass

    @unittest.skip("Gemma2's eager attn/sdpa attn outputs are expected to be different")
    def test_eager_matches_sdpa_generate(self):
        pass

    @parameterized.expand([("random",), ("same",)])
    @unittest.skip("Gemma2 has HybridCache which is not compatible with assisted decoding")
    def test_assisted_decoding_matches_greedy_search(self, assistant_type):
        pass

    @unittest.skip("Gemma2 has HybridCache which is not compatible with assisted decoding")
    def test_prompt_lookup_decoding_matches_greedy_search(self, assistant_type):
        pass

    @unittest.skip("Gemma2 has HybridCache which is not compatible with assisted decoding")
    def test_assisted_decoding_sample(self):
        pass

    @unittest.skip("Gemma2 has HybridCache which is not compatible with dola decoding")
    def test_dola_decoding_sample(self):
        pass

    @parameterized.expand([(1, False), (1, True), (4, False)])
    @unittest.skip("Gemma2 has HybridCache and doesn't support old tuple format at all")
    def test_new_cache_format(self, num_beams, do_sample):
        pass

    @unittest.skip("Gemma2 has HybridCache and doesn't support continue from past kv")
    def test_generate_continue_from_past_key_values(self):
        pass

    @unittest.skip("Gemma2 has HybridCache and doesn't support low_memory generation")
    def test_beam_search_low_memory(self):
        pass

    @unittest.skip("Gemma2 has HybridCache and doesn't support contrastive generation")
    def test_contrastive_generate(self):
        pass

    @unittest.skip("Gemma2 has HybridCache and doesn't support contrastive generation")
    def test_contrastive_generate_dict_outputs_use_cache(self):
        pass

    @unittest.skip("Gemma2 has HybridCache and doesn't support contrastive generation")
    def test_contrastive_generate_low_memory(self):
        pass

    @unittest.skip("Gemma2 has HybridCache and doesn't support StaticCache. Though it could, it shouldn't support.")
    def test_generate_with_static_cache(self):
        pass

    @unittest.skip("Gemma2 has HybridCache and doesn't support StaticCache. Though it could, it shouldn't support.")
    def test_generate_from_inputs_embeds_with_static_cache(self):
        pass

    # overwrite because HybridCache has fixed length for key/values
    def _check_attentions_for_generate(
        self, batch_size, attentions, min_length, max_length, config, use_cache=False, num_beam_groups=1
    ):
        self.assertIsInstance(attentions, tuple)
        self.assertListEqual(
            [isinstance(iter_attentions, tuple) for iter_attentions in attentions], [True] * len(attentions)
        )
        self.assertEqual(len(attentions), (max_length - min_length) * num_beam_groups)

        for idx, iter_attentions in enumerate(attentions):
            tgt_len = min_length + idx if not use_cache else 1
            src_len = min_length + idx if not use_cache else max_length

            expected_shape = (
                batch_size * num_beam_groups,
                config.num_attention_heads,
                tgt_len,
                src_len,
            )
            # check attn size
            self.assertListEqual(
                [layer_attention.shape for layer_attention in iter_attentions], [expected_shape] * len(iter_attentions)
            )

    # overwrite because HybridCache has fixed length for key/values
    def _check_past_key_values_for_generate(self, batch_size, past_key_values, seq_length, config, num_beam_groups=1):
        self.assertIsInstance(past_key_values, HybridCache)

        # check shape key, value (batch, head, max_seq_length, head_features)
        head_dim = config.head_dim if hasattr(config, "head_dim") else config.hidden_size // config.num_attention_heads
        num_key_value_heads = (
            config.num_attention_heads
            if getattr(config, "num_key_value_heads", None) is None
            else config.num_key_value_heads
        )
        num_hidden_layers = config.num_hidden_layers

        # we should get `max_length` in shape, not `max_length - embeds_length`
        # `+1` because the test in Mixin subtracts 1 which is needed for tuple cache
        static_cache_shape = (batch_size, num_key_value_heads, seq_length + 1, head_dim)
        static_layers = [layer_idx for layer_idx, boolean in enumerate(past_key_values.is_sliding) if not boolean]
        self.assertTrue(len(past_key_values.key_cache) == num_hidden_layers)
        self.assertTrue(past_key_values.key_cache[static_layers[0]].shape == static_cache_shape)

    @unittest.skip("Gemma2's eager attn/sdpa attn outputs are expected to be different")
    def test_sdpa_equivalence(self):
        pass


@slow
@require_torch_gpu
class Gemma2IntegrationTest(unittest.TestCase):
    input_text = ["Hello I am doing", "Hi today"]
    # This variable is used to determine which CUDA device are we using for our runners (A10 or T4)
    # Depending on the hardware we get different logits / generations
    cuda_compute_capability_major_version = None

    @classmethod
    def setUpClass(cls):
        if is_torch_available() and torch.cuda.is_available():
            # 8 is for A100 / A10 and 7 for T4
            cls.cuda_compute_capability_major_version = torch.cuda.get_device_capability()[0]

    @tooslow
    @require_read_token
    def test_model_9b_bf16(self):
        model_id = "google/gemma-2-9b"
        EXPECTED_TEXTS = [
            "<bos>Hello I am doing a project on the 1918 flu pandemic and I am trying to find out how many",
            "<pad><pad><bos>Hi today I'm going to be talking about the history of the United States. The United States of America",
        ]

        model = AutoModelForCausalLM.from_pretrained(
            model_id, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16, attn_implementation="eager"
        ).to(torch_device)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=False)

        self.assertEqual(output_text, EXPECTED_TEXTS)

    @tooslow
    @require_read_token
    def test_model_9b_fp16(self):
        model_id = "google/gemma-2-9b"
        EXPECTED_TEXTS = [
            "<bos>Hello I am doing a project on the 1918 flu pandemic and I am trying to find out how many",
            "<pad><pad><bos>Hi today I'm going to be talking about the history of the United States. The United States of America",
        ]

        model = AutoModelForCausalLM.from_pretrained(
            model_id, low_cpu_mem_usage=True, torch_dtype=torch.float16, attn_implementation="eager"
        ).to(torch_device)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=False)

        self.assertEqual(output_text, EXPECTED_TEXTS)

    @require_read_token
    @tooslow
    def test_model_9b_pipeline_bf16(self):
        # See https://github.com/huggingface/transformers/pull/31747 -- pipeline was broken for Gemma2 before this PR
        model_id = "google/gemma-2-9b"
        # EXPECTED_TEXTS should match the same non-pipeline test, minus the special tokens
        EXPECTED_TEXTS = [
            "Hello I am doing a project on the 1918 flu pandemic and I am trying to find out how many",
            "Hi today I'm going to be talking about the history of the United States. The United States of America",
        ]

        model = AutoModelForCausalLM.from_pretrained(
            model_id, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16, attn_implementation="flex_attention"
        ).to(torch_device)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

        output = pipe(self.input_text, max_new_tokens=20, do_sample=False, padding=True)

        self.assertEqual(output[0][0]["generated_text"], EXPECTED_TEXTS[0])
        self.assertEqual(output[1][0]["generated_text"], EXPECTED_TEXTS[1])

    @require_read_token
    def test_model_2b_pipeline_bf16_flex_attention(self):
        # See https://github.com/huggingface/transformers/pull/31747 -- pipeline was broken for Gemma2 before this PR
        model_id = "google/gemma-2-2b"
        # EXPECTED_TEXTS should match the same non-pipeline test, minus the special tokens
        EXPECTED_TEXTS = [
            "Hello I am doing a project on the 1960s and I am trying to find out what the average",
            "Hi today I'm going to be talking about the 10 best anime of all time.\n\n1",
        ]

        model = AutoModelForCausalLM.from_pretrained(
            model_id, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16, attn_implementation="flex_attention"
        ).to(torch_device)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

        output = pipe(self.input_text, max_new_tokens=20, do_sample=False, padding=True)

        self.assertEqual(output[0][0]["generated_text"], EXPECTED_TEXTS[0])
        self.assertEqual(output[1][0]["generated_text"], EXPECTED_TEXTS[1])

    @require_read_token
    @require_flash_attn
    @require_torch_gpu
    @mark.flash_attn_test
    @slow
    @tooslow
    def test_model_9b_flash_attn(self):
        # See https://github.com/huggingface/transformers/issues/31953 --- flash attn was generating garbage for gemma2, especially in long context
        model_id = "google/gemma-2-9b"
        EXPECTED_TEXTS = [
            '<bos>Hello I am doing a project on the 1918 flu pandemic and I am trying to find out how many people died in the United States. I have found a few sites that say 500,000 but I am not sure if that is correct. I have also found a site that says 675,000 but I am not sure if that is correct either. I am trying to find out how many people died in the United States. I have found a few',
            "<pad><pad><bos>Hi today I'm going to be talking about the history of the United States. The United States of America is a country in North America. It is the third largest country in the world by total area and the third most populous country with over 320 million people. The United States is a federal republic consisting of 50 states and a federal district. The 48 contiguous states and the district of Columbia are in central North America between Canada and Mexico. The state of Alaska is in the"
        ]  # fmt: skip

        model = AutoModelForCausalLM.from_pretrained(
            model_id, attn_implementation="flash_attention_2", torch_dtype="float16"
        ).to(torch_device)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=100, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=False)

        self.assertEqual(output_text, EXPECTED_TEXTS)

    @slow
    @require_read_token
    def test_export_static_cache(self):
        if version.parse(torch.__version__) < version.parse("2.5.0"):
            self.skipTest(reason="This test requires torch >= 2.5 to run.")

        from transformers.integrations.executorch import (
            TorchExportableModuleWithStaticCache,
            convert_and_export_with_cache,
        )

        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b", pad_token="</s>", padding_side="right")
        EXPECTED_TEXT_COMPLETION = [
            "Hello I am doing a project for my school and I need to know how to make a program that will take a number",
        ]
        max_generation_length = tokenizer(EXPECTED_TEXT_COMPLETION, return_tensors="pt", padding=True)[
            "input_ids"
        ].shape[-1]

        # Load model
        device = "cpu"
        dtype = torch.bfloat16
        cache_implementation = "static"
        attn_implementation = "sdpa"
        batch_size = 1
        model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-2-2b",
            device_map=device,
            torch_dtype=dtype,
            attn_implementation=attn_implementation,
            generation_config=GenerationConfig(
                use_cache=True,
                cache_implementation=cache_implementation,
                max_length=max_generation_length,
                cache_config={
                    "batch_size": batch_size,
                    "max_cache_len": max_generation_length,
                },
            ),
        )

        prompts = ["Hello I am doing"]
        prompt_tokens = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        prompt_token_ids = prompt_tokens["input_ids"]
        max_new_tokens = max_generation_length - prompt_token_ids.shape[-1]

        # Static Cache + export
        exported_program = convert_and_export_with_cache(model)
        ep_generated_ids = TorchExportableModuleWithStaticCache.generate(
            exported_program=exported_program, prompt_token_ids=prompt_token_ids, max_new_tokens=max_new_tokens
        )
        ep_generated_text = tokenizer.batch_decode(ep_generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, ep_generated_text)

    @require_read_token
    @tooslow
    def test_model_9b_bf16_flex_attention(self):
        model_id = "google/gemma-2-9b"
        EXPECTED_TEXTS = [
            "<bos>Hello I am doing a project on the 1918 flu pandemic and I am trying to find out how many",
            "<pad><pad><bos>Hi today I'm going to be talking about the history of the United States. The United States of America",
        ]

        model = AutoModelForCausalLM.from_pretrained(
            model_id, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16, attn_implementation="flex_attention"
        ).to(torch_device)
        assert model.config._attn_implementation == "flex_attention"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=False)

        self.assertEqual(output_text, EXPECTED_TEXTS)
