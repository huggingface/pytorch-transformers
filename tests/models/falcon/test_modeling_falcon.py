# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Falcon model."""

import unittest

from parameterized import parameterized

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    FalconConfig,
    is_torch_available,
    set_seed,
)
from transformers.testing_utils import (
    require_bitsandbytes,
    require_torch,
    require_torch_sdpa,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        FalconForCausalLM,
        FalconForQuestionAnswering,
        FalconForSequenceClassification,
        FalconForTokenClassification,
        FalconModel,
    )
    from transformers.models.falcon.modeling_falcon import (
        FalconRotaryEmbedding,
    )


class FalconModelTester:
    def __init__(
        self,
        parent,
        batch_size=3,
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

        token_type_ids = None

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
        return FalconConfig(
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
            is_decoder=False,
            initializer_range=self.initializer_range,
            pad_token_id=1,
            new_decoder_architecture=True,
        )

    def create_and_check_model(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = FalconModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

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
        model = FalconModel(config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        result = model(
            input_ids,
            attention_mask=input_mask,
            encoder_hidden_states=encoder_hidden_states,
        )
        result = model(input_ids, attention_mask=input_mask)
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
        model = FalconForCausalLM(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=token_labels)
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
        model = FalconForCausalLM(config=config)
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
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_torch
class FalconModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            FalconModel,
            FalconForCausalLM,
            FalconForSequenceClassification,
            FalconForTokenClassification,
            FalconForQuestionAnswering,
        )
        if is_torch_available()
        else ()
    )
    all_generative_model_classes = (FalconForCausalLM,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": FalconModel,
            "question-answering": FalconForQuestionAnswering,
            "text-classification": FalconForSequenceClassification,
            "text-generation": FalconForCausalLM,
            "token-classification": FalconForTokenClassification,
            "zero-shot": FalconForSequenceClassification,
        }
        if is_torch_available()
        else {}
    )
    test_headmasking = False
    test_pruning = False

    # TODO (ydshieh): Check this. See https://app.circleci.com/pipelines/github/huggingface/transformers/79245/workflows/9490ef58-79c2-410d-8f51-e3495156cf9c/jobs/1012146
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
        return True

    def setUp(self):
        self.model_tester = FalconModelTester(self)
        self.config_tester = ConfigTester(self, config_class=FalconConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_position_embedding_types(self):
        config, *inputs = self.model_tester.prepare_config_and_inputs()
        for alibi in [True, False]:
            config.alibi = alibi
            self.model_tester.create_and_check_model(config, *inputs)

    def test_falcon_sequence_classification_model(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        sequence_labels = ids_tensor([self.model_tester.batch_size], self.model_tester.type_sequence_label_size)
        model = FalconForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
        self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))

    def test_falcon_sequence_classification_model_for_single_label(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        config.problem_type = "single_label_classification"
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        sequence_labels = ids_tensor([self.model_tester.batch_size], self.model_tester.type_sequence_label_size)
        model = FalconForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
        self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))

    def test_falcon_sequence_classification_model_for_multi_label(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        config.problem_type = "multi_label_classification"
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        sequence_labels = ids_tensor(
            [self.model_tester.batch_size, config.num_labels], self.model_tester.type_sequence_label_size
        ).to(torch.float)
        model = FalconForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
        self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))

    def test_past_key_values_format(self):
        # Falcon can have different numbers of KV-heads than the number of query heads, so we need
        # to override this test to use the right head counts.
        for model_class in self.all_generative_model_classes:
            config, inputs = self.model_tester.prepare_config_and_inputs_for_common()

            # If it doesn't support cache, pass the test
            if not hasattr(config, "use_cache"):
                self.skipTest(reason="Model does not support cache")

            model = model_class(config).to(torch_device)
            if "use_cache" not in inputs:
                inputs["use_cache"] = True
            outputs = model(**inputs)

            # If "past_key_values" is not returned, pass the test (e.g. RWKV uses a different cache name and format)
            if "past_key_values" not in outputs:
                self.skipTest(reason="Model does not return past_key_values")

            num_hidden_layers = (
                getattr(config, "decoder_layers", None)
                or getattr(config, "num_decoder_layers", None)
                or config.num_hidden_layers
            )
            num_attention_heads = getattr(config, "num_kv_heads", config.num_attention_heads)
            embed_dim = getattr(config, "d_model", config.hidden_size)
            per_head_embed_dim = embed_dim // num_attention_heads

            past_kv = outputs["past_key_values"]
            self.assertEqual(len(past_kv), num_hidden_layers)

            batch_size, seq_length = inputs["input_ids"].shape
            for i in range(num_hidden_layers):
                if config.new_decoder_architecture:
                    num_attention_heads = config.num_attention_heads
                elif config.multi_query:
                    num_attention_heads = 1
                self.assertEqual(len(past_kv[0]), 2)  # K V for the decoder = 2
                self.assertEqual(
                    past_kv[i][0].shape, (batch_size, num_attention_heads, seq_length, per_head_embed_dim)
                )
                self.assertEqual(
                    past_kv[i][1].shape, (batch_size, num_attention_heads, seq_length, per_head_embed_dim)
                )

    @parameterized.expand([("linear",), ("dynamic",)])
    # Copied from tests.models.llama.test_modeling_llama.LlamaModelTest.test_model_rope_scaling_from_config with Llama->Falcon
    def test_model_rope_scaling_from_config(self, scaling_type):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        short_input = ids_tensor([1, 10], config.vocab_size)
        long_input = ids_tensor([1, int(config.max_position_embeddings * 1.5)], config.vocab_size)

        set_seed(42)  # Fixed seed at init time so the two models get the same random weights
        original_model = FalconModel(config)
        original_model.to(torch_device)
        original_model.eval()
        original_short_output = original_model(short_input).last_hidden_state
        original_long_output = original_model(long_input).last_hidden_state

        set_seed(42)  # Fixed seed at init time so the two models get the same random weights
        config.rope_scaling = {"type": scaling_type, "factor": 10.0}
        scaled_model = FalconModel(config)
        scaled_model.to(torch_device)
        scaled_model.eval()
        scaled_short_output = scaled_model(short_input).last_hidden_state
        scaled_long_output = scaled_model(long_input).last_hidden_state

        # Dynamic scaling does not change the RoPE embeddings until it receives an input longer than the original
        # maximum sequence length, so the outputs for the short input should match.
        if scaling_type == "dynamic":
            torch.testing.assert_close(original_short_output, scaled_short_output, rtol=1e-5, atol=1e-5)
        else:
            self.assertFalse(torch.allclose(original_short_output, scaled_short_output, atol=1e-5))

        # The output should be different for long inputs
        self.assertFalse(torch.allclose(original_long_output, scaled_long_output, atol=1e-5))

    # Copied from tests.models.gpt_neox.test_modeling_gpt_neox.GPTNeoXModelTest.test_model_rope_scaling with GPTNeoX->Falcon
    def test_model_rope_scaling(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        scaling_factor = 10
        short_input_length = 10
        long_input_length = int(config.max_position_embeddings * 1.5)

        # Inputs
        x = torch.randn(1, dtype=torch.float32, device=torch_device)  # used exlusively to get the dtype and the device
        position_ids_short = torch.arange(short_input_length, dtype=torch.long, device=torch_device)
        position_ids_short = position_ids_short.unsqueeze(0)
        position_ids_long = torch.arange(long_input_length, dtype=torch.long, device=torch_device)
        position_ids_long = position_ids_long.unsqueeze(0)

        # Sanity check original RoPE
        original_rope = FalconRotaryEmbedding(config).to(torch_device)
        original_cos_short, original_sin_short = original_rope(x, position_ids_short)
        original_cos_long, original_sin_long = original_rope(x, position_ids_long)
        torch.testing.assert_close(original_cos_short, original_cos_long[:, :short_input_length, :])
        torch.testing.assert_close(original_sin_short, original_sin_long[:, :short_input_length, :])

        # Sanity check linear RoPE scaling
        # New position "x" should match original position with index "x/scaling_factor"
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
        linear_scaling_rope = FalconRotaryEmbedding(config).to(torch_device)
        linear_cos_short, linear_sin_short = linear_scaling_rope(x, position_ids_short)
        linear_cos_long, linear_sin_long = linear_scaling_rope(x, position_ids_long)
        torch.testing.assert_close(linear_cos_short, linear_cos_long[:, :short_input_length, :])
        torch.testing.assert_close(linear_sin_short, linear_sin_long[:, :short_input_length, :])
        for new_position in range(0, long_input_length, scaling_factor):
            original_position = int(new_position // scaling_factor)
            torch.testing.assert_close(linear_cos_long[:, new_position, :], original_cos_long[:, original_position, :])
            torch.testing.assert_close(linear_sin_long[:, new_position, :], original_sin_long[:, original_position, :])

        # Sanity check Dynamic NTK RoPE scaling
        # Scaling should only be observed after a long input is fed. We can observe that the frequencies increase
        # with scaling_factor (or that `inv_freq` decreases)
        config.rope_scaling = {"type": "dynamic", "factor": scaling_factor}
        ntk_scaling_rope = FalconRotaryEmbedding(config).to(torch_device)
        ntk_cos_short, ntk_sin_short = ntk_scaling_rope(x, position_ids_short)
        ntk_cos_long, ntk_sin_long = ntk_scaling_rope(x, position_ids_long)
        torch.testing.assert_close(ntk_cos_short, original_cos_short)
        torch.testing.assert_close(ntk_sin_short, original_sin_short)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(ntk_cos_long, original_cos_long)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(ntk_sin_long, original_sin_long)
        self.assertTrue((ntk_scaling_rope.inv_freq <= original_rope.inv_freq).all())


@require_torch
class FalconLanguageGenerationTest(unittest.TestCase):
    @slow
    def test_lm_generate_falcon(self):
        tokenizer = AutoTokenizer.from_pretrained("Rocketknight1/falcon-rw-1b")
        model = FalconForCausalLM.from_pretrained("Rocketknight1/falcon-rw-1b")
        model.eval()
        model.to(torch_device)
        inputs = tokenizer("My favorite food is", return_tensors="pt").to(torch_device)

        EXPECTED_OUTPUT = (
            "My favorite food is pizza. I love it so much that I have a pizza party every year for my birthday."
        )

        output_ids = model.generate(**inputs, do_sample=False, max_new_tokens=19)
        output_str = tokenizer.batch_decode(output_ids)[0]

        self.assertEqual(output_str, EXPECTED_OUTPUT)

    @slow
    @require_bitsandbytes
    def test_lm_generate_falcon_11b(self):
        tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-11B", padding_side="left")
        model = FalconForCausalLM.from_pretrained(
            "tiiuae/falcon-11B", device_map={"": torch_device}, load_in_8bit=True
        )
        model.eval()
        inputs = tokenizer(
            "Two roads diverged in a yellow wood,", return_tensors="pt", return_token_type_ids=False
        ).to(torch_device)

        EXPECTED_OUTPUT = "Two roads diverged in a yellow wood,\nAnd sorry I could not travel both\n"

        output_ids = model.generate(**inputs, do_sample=False, max_new_tokens=9)
        output_str = tokenizer.batch_decode(output_ids)[0]

        self.assertEqual(output_str, EXPECTED_OUTPUT)

    @slow
    def test_lm_generation_big_models(self):
        # The big models are way too big for the CI, so we use tiny random models that resemble their
        # architectures but with much smaller and fewer layers
        for repo in ["Rocketknight1/tiny-random-falcon-7b", "Rocketknight1/tiny-random-falcon-40b"]:
            tokenizer = AutoTokenizer.from_pretrained(repo)
            model = FalconForCausalLM.from_pretrained(repo)
            model.eval()
            model.to(torch_device)
            inputs = tokenizer("My favorite food is", return_tensors="pt").to(torch_device)

            # We just test that these run without errors - the models are randomly initialized
            # and so the actual text outputs will be garbage
            model.generate(**inputs, do_sample=False, max_new_tokens=4)
            model.generate(**inputs, do_sample=True, max_new_tokens=4)
            model.generate(**inputs, num_beams=2, max_new_tokens=4)

    @slow
    def test_lm_generation_use_cache(self):
        # The big models are way too big for the CI, so we use tiny random models that resemble their
        # architectures but with much smaller and fewer layers
        with torch.no_grad():
            for repo in [
                "Rocketknight1/falcon-rw-1b",
                "Rocketknight1/tiny-random-falcon-7b",
                "Rocketknight1/tiny-random-falcon-40b",
            ]:
                tokenizer = AutoTokenizer.from_pretrained(repo)
                model = FalconForCausalLM.from_pretrained(repo)
                model.eval()
                model.to(device=torch_device)
                inputs = tokenizer("My favorite food is", return_tensors="pt").to(torch_device)

                # Test results are the same with and without cache
                outputs_no_cache = model.generate(**inputs, do_sample=False, max_new_tokens=20, use_cache=False)
                outputs_cache = model.generate(**inputs, do_sample=False, max_new_tokens=20, use_cache=True)
                self.assertTrue((outputs_cache - outputs_no_cache).sum().item() == 0)

    @require_bitsandbytes
    @slow
    def test_batched_generation(self):
        tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b", padding_side="left")
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            "tiiuae/falcon-7b",
            device_map={"": torch_device},
            load_in_4bit=True,
        )

        test_text = "A sequence: 1, 2"  # should generate the rest of the sequence

        unpadded_inputs = tokenizer([test_text], return_tensors="pt").to("cuda:0")
        unpadded_gen_out = model.generate(**unpadded_inputs, max_new_tokens=20)
        unpadded_gen_text = tokenizer.batch_decode(unpadded_gen_out, skip_special_tokens=True)

        dummy_text = "This is a longer text " * 2  # forces left-padding on `test_text`
        padded_inputs = tokenizer([test_text, dummy_text], return_tensors="pt", padding=True).to("cuda:0")
        padded_gen_out = model.generate(**padded_inputs, max_new_tokens=20)
        padded_gen_text = tokenizer.batch_decode(padded_gen_out, skip_special_tokens=True)

        expected_output = "A sequence: 1, 2, 3, 4, 5, 6, 7, 8, "
        self.assertLess(unpadded_inputs.input_ids.shape[-1], padded_inputs.input_ids.shape[-1])  # left-padding exists
        self.assertEqual(unpadded_gen_text[0], expected_output)
        self.assertEqual(padded_gen_text[0], expected_output)

    @slow
    @require_torch_sdpa
    def test_falcon_alibi_sdpa_matches_eager(self):
        input_ids = torch.randint(0, 1000, (5, 20))

        config = FalconConfig(
            vocab_size=1000,
            hidden_size=64,
            num_hidden_layers=3,
            num_attention_heads=4,
            new_decoder_architecture=True,
            alibi=True,
        )

        falcon = FalconForCausalLM(config)
        falcon = falcon.eval()

        with torch.no_grad():
            # output_attentions=True dispatches to eager path
            falcon_output_eager = falcon(input_ids, output_attentions=True)[0]
            falcon_output_sdpa = falcon(input_ids)[0]

        torch.testing.assert_close(falcon_output_eager, falcon_output_sdpa, rtol=1e-3, atol=1e-3)
