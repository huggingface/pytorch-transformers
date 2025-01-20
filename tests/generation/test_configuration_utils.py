# coding=utf-8
# Copyright 2022 The HuggingFace Team Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a clone of the License at
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
import tempfile
import unittest
import warnings

from huggingface_hub import HfFolder, create_pull_request
from parameterized import parameterized

from transformers import AutoConfig, GenerationConfig, WatermarkingConfig, is_torch_available


if is_torch_available():
    import torch

from transformers.generation import (
    ClassifierFreeGuidanceLogitsProcessor,
    EncoderNoRepeatNGramLogitsProcessor,
    EncoderRepetitionPenaltyLogitsProcessor,
    EpsilonLogitsWarper,
    EtaLogitsWarper,
    ExponentialDecayLengthPenalty,
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    GenerationMode,
    HammingDiversityLogitsProcessor,
    MinLengthLogitsProcessor,
    MinNewTokensLengthLogitsProcessor,
    MinPLogitsWarper,
    NoBadWordsLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    PrefixConstrainedLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    SequenceBiasLogitsProcessor,
    SuppressTokensAtBeginLogitsProcessor,
    SuppressTokensLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    TypicalLogitsWarper,
    UnbatchedClassifierFreeGuidanceLogitsProcessor,
    WatermarkLogitsProcessor,
)
from transformers.testing_utils import TOKEN, TemporaryHubRepo, is_staging_test, torch_device


class GenerationConfigTest(unittest.TestCase):
    @parameterized.expand([(None,), ("foo.json",)])
    def test_save_load_config(self, config_name):
        config = GenerationConfig(
            do_sample=True,
            temperature=0.7,
            length_penalty=1.0,
            bad_words_ids=[[1, 2, 3], [4, 5]],
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            config.save_pretrained(tmp_dir, config_name=config_name)
            loaded_config = GenerationConfig.from_pretrained(tmp_dir, config_name=config_name)

        # Checks parameters that were specified
        self.assertEqual(loaded_config.do_sample, True)
        self.assertEqual(loaded_config.temperature, 0.7)
        self.assertEqual(loaded_config.length_penalty, 1.0)
        self.assertEqual(loaded_config.bad_words_ids, [[1, 2, 3], [4, 5]])

        # Checks parameters that were not specified (defaults)
        self.assertEqual(loaded_config.top_k, 50)
        self.assertEqual(loaded_config.max_length, 20)
        self.assertEqual(loaded_config.max_time, None)

    def test_from_model_config(self):
        model_config = AutoConfig.from_pretrained("openai-community/gpt2")
        generation_config_from_model = GenerationConfig.from_model_config(model_config)
        default_generation_config = GenerationConfig()

        # The generation config has loaded a few non-default parameters from the model config
        self.assertNotEqual(generation_config_from_model, default_generation_config)

        # One of those parameters is eos_token_id -- check if it matches
        self.assertNotEqual(generation_config_from_model.eos_token_id, default_generation_config.eos_token_id)
        self.assertEqual(generation_config_from_model.eos_token_id, model_config.eos_token_id)

    def test_update(self):
        generation_config = GenerationConfig()
        update_kwargs = {
            "max_new_tokens": 1024,
            "foo": "bar",
        }
        update_kwargs_copy = copy.deepcopy(update_kwargs)
        unused_kwargs = generation_config.update(**update_kwargs)

        # update_kwargs was not modified (no side effects)
        self.assertEqual(update_kwargs, update_kwargs_copy)

        # update_kwargs was used to update the config on valid attributes
        self.assertEqual(generation_config.max_new_tokens, 1024)

        # `.update()` returns a dictionary of unused kwargs
        self.assertEqual(unused_kwargs, {"foo": "bar"})

    # TODO: @Arthur and/or @Joao
    # FAILED tests/generation/test_configuration_utils.py::GenerationConfigTest::test_initialize_new_kwargs - AttributeError: 'GenerationConfig' object has no attribute 'get_text_config'
    # See: https://app.circleci.com/pipelines/github/huggingface/transformers/104831/workflows/e5e61514-51b7-4c8c-bba7-3c4d2986956e/jobs/1394252
    @unittest.skip("failed with `'GenerationConfig' object has no attribute 'get_text_config'`")
    def test_initialize_new_kwargs(self):
        generation_config = GenerationConfig()
        generation_config.foo = "bar"

        with tempfile.TemporaryDirectory("test-generation-config") as tmp_dir:
            generation_config.save_pretrained(tmp_dir)

            new_config = GenerationConfig.from_pretrained(tmp_dir)
        # update_kwargs was used to update the config on valid attributes
        self.assertEqual(new_config.foo, "bar")

        generation_config = GenerationConfig.from_model_config(new_config)
        assert not hasattr(generation_config, "foo")  # no new kwargs should be initialized if from config

    def test_kwarg_init(self):
        """Tests that we can overwrite attributes at `from_pretrained` time."""
        default_config = GenerationConfig()
        self.assertEqual(default_config.temperature, 1.0)
        self.assertEqual(default_config.do_sample, False)
        self.assertEqual(default_config.num_beams, 1)

        config = GenerationConfig(
            do_sample=True,
            temperature=0.7,
            length_penalty=1.0,
            bad_words_ids=[[1, 2, 3], [4, 5]],
        )
        self.assertEqual(config.temperature, 0.7)
        self.assertEqual(config.do_sample, True)
        self.assertEqual(config.num_beams, 1)

        with tempfile.TemporaryDirectory() as tmp_dir:
            config.save_pretrained(tmp_dir)
            loaded_config = GenerationConfig.from_pretrained(tmp_dir, temperature=1.0)

        self.assertEqual(loaded_config.temperature, 1.0)
        self.assertEqual(loaded_config.do_sample, True)
        self.assertEqual(loaded_config.num_beams, 1)  # default value

    def test_validate(self):
        """
        Tests that the `validate` method is working as expected. Note that `validate` is called at initialization time
        """
        # A correct configuration will not throw any warning
        with warnings.catch_warnings(record=True) as captured_warnings:
            GenerationConfig()
        self.assertEqual(len(captured_warnings), 0)

        # Inconsequent but technically wrong configuration will throw a warning (e.g. setting sampling
        # parameters with `do_sample=False`). May be escalated to an error in the future.
        with warnings.catch_warnings(record=True) as captured_warnings:
            GenerationConfig(do_sample=False, temperature=0.5)
        self.assertEqual(len(captured_warnings), 1)

        with warnings.catch_warnings(record=True) as captured_warnings:
            GenerationConfig(return_dict_in_generate=False, output_scores=True)
        self.assertEqual(len(captured_warnings), 1)

        # Expanding on the case above, we can update a bad configuration to get rid of the warning. Ideally,
        # that is done by unsetting the parameter (i.e. setting it to None)
        generation_config_bad_temperature = GenerationConfig(do_sample=False, temperature=0.5)
        with warnings.catch_warnings(record=True) as captured_warnings:
            # BAD - 0.9 means it is still set, we should warn
            generation_config_bad_temperature.update(temperature=0.9)
        self.assertEqual(len(captured_warnings), 1)
        generation_config_bad_temperature = GenerationConfig(do_sample=False, temperature=0.5)
        with warnings.catch_warnings(record=True) as captured_warnings:
            # CORNER CASE - 1.0 is the default, we can't detect whether it is set by the user or not, we shouldn't warn
            generation_config_bad_temperature.update(temperature=1.0)
        self.assertEqual(len(captured_warnings), 0)
        generation_config_bad_temperature = GenerationConfig(do_sample=False, temperature=0.5)
        with warnings.catch_warnings(record=True) as captured_warnings:
            # OK - None means it is unset, nothing to warn about
            generation_config_bad_temperature.update(temperature=None)
        self.assertEqual(len(captured_warnings), 0)

        # Impossible sets of contraints/parameters will raise an exception
        with self.assertRaises(ValueError):
            GenerationConfig(do_sample=False, num_beams=1, num_return_sequences=2)
        with self.assertRaises(ValueError):
            # dummy constraint
            GenerationConfig(do_sample=True, num_beams=2, constraints=["dummy"])
        with self.assertRaises(ValueError):
            GenerationConfig(do_sample=True, num_beams=2, force_words_ids=[[[1, 2, 3]]])

        # Passing `generate()`-only flags to `validate` will raise an exception
        with self.assertRaises(ValueError):
            GenerationConfig(logits_processor="foo")

        # Model-specific parameters will NOT raise an exception or a warning
        with warnings.catch_warnings(record=True) as captured_warnings:
            GenerationConfig(foo="bar")
        self.assertEqual(len(captured_warnings), 0)

    def test_refuse_to_save(self):
        """Tests that we refuse to save a generation config that fails validation."""

        # setting the temperature alone is invalid, as we also need to set do_sample to True -> throws a warning that
        # is caught, doesn't save, and raises an exception
        config = GenerationConfig()
        config.temperature = 0.5
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(ValueError) as exc:
                config.save_pretrained(tmp_dir)
            self.assertTrue("Fix these issues to save the configuration." in str(exc.exception))
            self.assertTrue(len(os.listdir(tmp_dir)) == 0)

        # greedy decoding throws an exception if we try to return multiple sequences -> throws an exception that is
        # caught, doesn't save, and raises a warning
        config = GenerationConfig()
        config.num_return_sequences = 2
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(ValueError) as exc:
                config.save_pretrained(tmp_dir)
            self.assertTrue("Fix these issues to save the configuration." in str(exc.exception))
            self.assertTrue(len(os.listdir(tmp_dir)) == 0)

        # final check: no warnings/exceptions thrown if it is correct, and file is saved
        config = GenerationConfig()
        with tempfile.TemporaryDirectory() as tmp_dir:
            with warnings.catch_warnings(record=True) as captured_warnings:
                config.save_pretrained(tmp_dir)
            self.assertEqual(len(captured_warnings), 0)
            self.assertTrue(len(os.listdir(tmp_dir)) == 1)

    def test_generation_mode(self):
        """Tests that the `get_generation_mode` method is working as expected."""
        config = GenerationConfig()
        self.assertEqual(config.get_generation_mode(), GenerationMode.GREEDY_SEARCH)

        config = GenerationConfig(do_sample=True)
        self.assertEqual(config.get_generation_mode(), GenerationMode.SAMPLE)

        config = GenerationConfig(num_beams=2)
        self.assertEqual(config.get_generation_mode(), GenerationMode.BEAM_SEARCH)

        config = GenerationConfig(top_k=10, do_sample=False, penalty_alpha=0.6)
        self.assertEqual(config.get_generation_mode(), GenerationMode.CONTRASTIVE_SEARCH)

        config = GenerationConfig()
        self.assertEqual(config.get_generation_mode(assistant_model="foo"), GenerationMode.ASSISTED_GENERATION)

    def test_static_cache_without_cache_config(self):
        """Regression test for #35026 -- static cache should work without a cache config."""
        config = GenerationConfig(cache_implementation="static")
        self.assertEqual(config.cache_implementation, "static")
        self.assertEqual(config.cache_config, None)


class GenerationConfigSerializationTest(unittest.TestCase):
    def test_serialize_generation_sequence_bias(self):
        """Tests that GenerationConfig is serialized and SequenceBiasLogitsProcessor is initialized with sequence_bias parameter"""
        generation_config = GenerationConfig()
        sequence_bias = [[[45, 67], -0.6], [[89], 1.2]]
        generation_config.sequence_bias = sequence_bias
        with tempfile.TemporaryDirectory("test-generation-config") as tmp_dir:
            generation_config.save_pretrained(tmp_dir)
            new_config = GenerationConfig.from_pretrained(tmp_dir)
        self.assertSequenceEqual(new_config.sequence_bias, sequence_bias)

        expected_sequence_bias = {(45, 67): -0.6, (89,): 1.2}
        bias_logits_processor = SequenceBiasLogitsProcessor(new_config.sequence_bias)
        self.assertDictEqual(bias_logits_processor.sequence_bias, expected_sequence_bias)

    def test_serialize_generation_min_length_eos_token(self):
        """Tests that GenerationConfig is serialized and MinLengthLogitsProcessor is initialized with min_length and eos_token_id"""
        eos_token_id = 0
        min_length = 10

        generation_config = GenerationConfig(min_length=min_length, eos_token_id=eos_token_id)
        with tempfile.TemporaryDirectory("test-generation-config") as tmp_dir:
            generation_config.save_pretrained(tmp_dir)
            new_config = GenerationConfig.from_pretrained(tmp_dir)
        self.assertEqual(new_config.min_length, min_length)
        self.assertEqual(new_config.eos_token_id, eos_token_id)

        min_dist_processor = MinLengthLogitsProcessor(
            min_length=new_config.min_length, eos_token_id=new_config.eos_token_id
        )
        self.assertEqual(min_dist_processor.min_length, min_length)
        self.assertEqual(min_dist_processor.eos_token_id, eos_token_id)

    def test_serialize_generation_min_new_tokens(self):
        """Tests that GenerationConfig is serialized and MinNewTokensLengthLogitsProcessor is initialized with min_new_tokens"""
        eos_token_id = 0
        min_new_tokens = 5
        prompt_length_to_skip = 2

        generation_config = GenerationConfig(min_new_tokens=min_new_tokens)
        with tempfile.TemporaryDirectory("test-generation-config") as tmp_dir:
            generation_config.save_pretrained(tmp_dir)
            new_config = GenerationConfig.from_pretrained(tmp_dir)
        self.assertEqual(new_config.min_new_tokens, min_new_tokens)

        min_new_tokens_processor = MinNewTokensLengthLogitsProcessor(
            prompt_length_to_skip=prompt_length_to_skip,
            min_new_tokens=new_config.min_new_tokens,
            eos_token_id=eos_token_id,
        )
        self.assertEqual(min_new_tokens_processor.min_new_tokens, min_new_tokens)

    def test_serialize_generation_temperature(self):
        """Tests that GenerationConfig is serialized and TemperatureLogitsWarper is initialized with temperature"""
        temperature = 2.0

        generation_config = GenerationConfig(temperature=temperature, do_sample=True)
        with tempfile.TemporaryDirectory("test-generation-config") as tmp_dir:
            generation_config.save_pretrained(tmp_dir)
            new_config = GenerationConfig.from_pretrained(tmp_dir)
        self.assertEqual(new_config.temperature, temperature)

        temperature_logits_warper = TemperatureLogitsWarper(temperature=new_config.temperature)
        self.assertEqual(temperature_logits_warper.temperature, temperature)

    def test_serialize_generation_repetition_penalty(self):
        """Tests that GenerationConfig is serialized and RepetitionPenaltyLogitsProcessor is initialized with repetition_penalty"""
        penalty = 2.0

        generation_config = GenerationConfig(repetition_penalty=penalty)
        with tempfile.TemporaryDirectory("test-generation-config") as tmp_dir:
            generation_config.save_pretrained(tmp_dir)
            new_config = GenerationConfig.from_pretrained(tmp_dir)
        self.assertEqual(new_config.repetition_penalty, penalty)

        rep_penalty_proc = RepetitionPenaltyLogitsProcessor(penalty=new_config.repetition_penalty)
        self.assertEqual(rep_penalty_proc.penalty, penalty)

    def test_serialize_generation_encoder_repetition_penalty(self):
        """Tests that GenerationConfig is serialized and EncoderRepetitionPenaltyLogitsProcessor is initialized with penalty and input_ids"""
        penalty = 2.0
        input_ids = torch.tensor([[0, 1], [5, 0]], device=torch_device, dtype=torch.long)

        generation_config = GenerationConfig(encoder_repetition_penalty=penalty)
        with tempfile.TemporaryDirectory("test-generation-config") as tmp_dir:
            generation_config.save_pretrained(tmp_dir)
            new_config = GenerationConfig.from_pretrained(tmp_dir)
        self.assertEqual(new_config.encoder_repetition_penalty, penalty)

        rep_penalty_proc = EncoderRepetitionPenaltyLogitsProcessor(
            penalty=new_config.encoder_repetition_penalty, encoder_input_ids=input_ids
        )
        self.assertEqual(rep_penalty_proc.penalty, 1 / penalty)
        torch.testing.assert_close(rep_penalty_proc.encoder_input_ids, input_ids)

    def test_serialize_generation_top_p(self):
        """Tests that GenerationConfig is serialized and TopPLogitsWarper is initialized with top_p"""
        top_p = 0.8

        generation_config = GenerationConfig(top_p=top_p, do_sample=True)
        with tempfile.TemporaryDirectory("test-generation-config") as tmp_dir:
            generation_config.save_pretrained(tmp_dir)
            new_config = GenerationConfig.from_pretrained(tmp_dir)
        self.assertEqual(new_config.top_p, top_p)

        rep_penalty_proc = TopPLogitsWarper(top_p=new_config.top_p)
        self.assertEqual(rep_penalty_proc.top_p, top_p)

    def test_serialize_generation_top_k(self):
        """Tests that GenerationConfig is serialized and TopKLogitsWarper is initialized with top_k"""
        top_k = 2

        generation_config = GenerationConfig(top_k=top_k, do_sample=True)
        with tempfile.TemporaryDirectory("test-generation-config") as tmp_dir:
            generation_config.save_pretrained(tmp_dir)
            new_config = GenerationConfig.from_pretrained(tmp_dir)
        self.assertEqual(new_config.top_k, top_k)

        top_k_logits_wrap = TopKLogitsWarper(top_k=new_config.top_k)
        self.assertEqual(top_k_logits_wrap.top_k, top_k)

    def test_serialize_generation_min_p(self):
        """Tests that GenerationConfig is serialized and MinPLogitsWarper is initialized with min_p"""
        min_p = 0.8

        generation_config = GenerationConfig(min_p=min_p, do_sample=True)
        with tempfile.TemporaryDirectory("test-generation-config") as tmp_dir:
            generation_config.save_pretrained(tmp_dir)
            new_config = GenerationConfig.from_pretrained(tmp_dir)
        self.assertEqual(new_config.min_p, min_p)

        min_k_logits_wrap = MinPLogitsWarper(min_p=new_config.min_p)
        self.assertEqual(min_k_logits_wrap.min_p, min_p)

    def test_serialize_generation_typical_p(self):
        """Tests that GenerationConfig is serialized and TypicalLogitsWarper is initialized with mass"""
        mass = 0.8

        generation_config = GenerationConfig(typical_p=mass, do_sample=True)
        with tempfile.TemporaryDirectory("test-generation-config") as tmp_dir:
            generation_config.save_pretrained(tmp_dir)
            new_config = GenerationConfig.from_pretrained(tmp_dir)
        self.assertEqual(new_config.typical_p, mass)

        typical_p_logits_wrap = TypicalLogitsWarper(mass=new_config.typical_p)
        self.assertEqual(typical_p_logits_wrap.mass, mass)

    def test_serialize_generation_epsilon_cutoff(self):
        """Tests that GenerationConfig is serialized and EpsilonLogitsWarper is initialized with epsilon"""
        epsilon = 0.8

        generation_config = GenerationConfig(epsilon_cutoff=epsilon, do_sample=True)
        with tempfile.TemporaryDirectory("test-generation-config") as tmp_dir:
            generation_config.save_pretrained(tmp_dir)
            new_config = GenerationConfig.from_pretrained(tmp_dir)
        self.assertEqual(new_config.epsilon_cutoff, epsilon)

        epsilon_logits_wrap = EpsilonLogitsWarper(epsilon=new_config.epsilon_cutoff)
        self.assertEqual(epsilon_logits_wrap.epsilon, epsilon)

    def test_serialize_generation_eta_cutoff(self):
        """Tests that GenerationConfig is serialized and EtaLogitsWarper is initialized with epsilon"""
        epsilon = 0.8

        generation_config = GenerationConfig(eta_cutoff=epsilon, do_sample=True)
        with tempfile.TemporaryDirectory("test-generation-config") as tmp_dir:
            generation_config.save_pretrained(tmp_dir)
            new_config = GenerationConfig.from_pretrained(tmp_dir)
        self.assertEqual(new_config.eta_cutoff, epsilon)

        eta_logits_wrap = EtaLogitsWarper(epsilon=new_config.eta_cutoff)
        self.assertEqual(eta_logits_wrap.epsilon, epsilon)

    def test_serialize_generation_ngram_size(self):
        """Tests that GenerationConfig is serialized and NoRepeatNGramLogitsProcessor is initialized with ngram_size"""
        ngram_size = 2

        generation_config = GenerationConfig(no_repeat_ngram_size=ngram_size, do_sample=True)
        with tempfile.TemporaryDirectory("test-generation-config") as tmp_dir:
            generation_config.save_pretrained(tmp_dir)
            new_config = GenerationConfig.from_pretrained(tmp_dir)
        self.assertEqual(new_config.no_repeat_ngram_size, ngram_size)

        no_repeat_ngram_proc = NoRepeatNGramLogitsProcessor(ngram_size=new_config.no_repeat_ngram_size)
        self.assertEqual(no_repeat_ngram_proc.ngram_size, ngram_size)

    def test_serialize_generation_encoder_ngram_size(self):
        """Tests that GenerationConfig is serialized and EncoderNoRepeatNGramLogitsProcessor is initialized with ngram_size"""
        ngram_size = 2
        input_ids = torch.tensor([[0, 1], [5, 0]], device=torch_device, dtype=torch.long)

        generation_config = GenerationConfig(encoder_no_repeat_ngram_size=ngram_size, do_sample=True)
        with tempfile.TemporaryDirectory("test-generation-config") as tmp_dir:
            generation_config.save_pretrained(tmp_dir)
            new_config = GenerationConfig.from_pretrained(tmp_dir)
        self.assertEqual(new_config.encoder_no_repeat_ngram_size, ngram_size)

        encoder_no_repeat_ngram_proc = EncoderNoRepeatNGramLogitsProcessor(
            encoder_ngram_size=new_config.encoder_no_repeat_ngram_size, encoder_input_ids=input_ids
        )
        self.assertEqual(encoder_no_repeat_ngram_proc.ngram_size, ngram_size)

    def test_serialize_generation_bad_words_ids(self):
        """Tests that GenerationConfig is serialized and NoBadWordsLogitsProcessor is initialized with bad_words_ids"""
        bad_word_tokens = [[1], [4], [1, 0], [0, 1, 2], [1, 3, 1, 3]]

        generation_config = GenerationConfig(bad_words_ids=bad_word_tokens)
        with tempfile.TemporaryDirectory("test-generation-config") as tmp_dir:
            generation_config.save_pretrained(tmp_dir)
            new_config = GenerationConfig.from_pretrained(tmp_dir)
        self.assertSequenceEqual(new_config.bad_words_ids, bad_word_tokens)

        no_bad_words_dist_proc = NoBadWordsLogitsProcessor(bad_words_ids=new_config.bad_words_ids)
        self.assertSequenceEqual(no_bad_words_dist_proc.bad_word_ids, bad_word_tokens)

    def test_serialize_generation_num_beams(self):
        """Tests that GenerationConfig is serialized and PrefixConstrainedLogitsProcessor is initialized with num_beams"""
        num_beams = 1

        def prefix_allowed_tokens_fn(batch_id, inputs_ids):
            return [[0, 1], [2, 3]][batch_id]

        generation_config = GenerationConfig(num_beams=num_beams)
        with tempfile.TemporaryDirectory("test-generation-config") as tmp_dir:
            generation_config.save_pretrained(tmp_dir)
            new_config = GenerationConfig.from_pretrained(tmp_dir)
        self.assertEqual(new_config.num_beams, num_beams)

        prefix_constrained_logits_proc = PrefixConstrainedLogitsProcessor(
            prefix_allowed_tokens_fn, num_beams=new_config.num_beams
        )
        self.assertEqual(prefix_constrained_logits_proc._num_beams, num_beams)

    def test_serialize_generation_diversity_penalty_and_num_bean_groups(self):
        """Tests that GenerationConfig is serialized and HammingDiversityLogitsProcessor is initialized with diversity_penalty_and_num_bean_groups"""
        num_beams = 2
        num_beam_groups = 2
        diversity_penalty = 1.0

        generation_config = GenerationConfig(
            num_beams=num_beams, diversity_penalty=diversity_penalty, num_beam_groups=num_beam_groups
        )
        with tempfile.TemporaryDirectory("test-generation-config") as tmp_dir:
            generation_config.save_pretrained(tmp_dir)
            new_config = GenerationConfig.from_pretrained(tmp_dir)
        self.assertEqual(new_config.num_beams, num_beams)
        self.assertEqual(new_config.diversity_penalty, diversity_penalty)
        self.assertEqual(new_config.num_beam_groups, num_beam_groups)

        diversity_logits_processor = HammingDiversityLogitsProcessor(
            diversity_penalty=new_config.diversity_penalty,
            num_beams=new_config.num_beams,
            num_beam_groups=new_config.num_beam_groups,
        )
        self.assertEqual(diversity_logits_processor._num_beams, num_beams)
        self.assertEqual(diversity_logits_processor._diversity_penalty, diversity_penalty)
        self.assertEqual(diversity_logits_processor._num_sub_beams, num_beams // num_beam_groups)

    def test_serialize_generation_bos_token_id(self):
        """Tests that GenerationConfig is serialized and ForcedBOSTokenLogitsProcessor is initialized with bos_token_id"""
        bos_token_id = 0

        generation_config = GenerationConfig(bos_token_id=bos_token_id)
        with tempfile.TemporaryDirectory("test-generation-config") as tmp_dir:
            generation_config.save_pretrained(tmp_dir)
            new_config = GenerationConfig.from_pretrained(tmp_dir)
        self.assertEqual(new_config.bos_token_id, bos_token_id)

        logits_processor = ForcedBOSTokenLogitsProcessor(bos_token_id=new_config.bos_token_id)
        self.assertEqual(logits_processor.bos_token_id, bos_token_id)

    def test_serialize_generation_eos_token_id(self):
        """Tests that GenerationConfig is serialized and ForcedEOSTokenLogitsProcessor is initialized with eos_token_id"""
        eos_token_id = 0
        max_length = 5

        generation_config = GenerationConfig(eos_token_id=eos_token_id)
        with tempfile.TemporaryDirectory("test-generation-config") as tmp_dir:
            generation_config.save_pretrained(tmp_dir)
            new_config = GenerationConfig.from_pretrained(tmp_dir)
        self.assertEqual(new_config.eos_token_id, eos_token_id)

        logits_processor = ForcedEOSTokenLogitsProcessor(
            max_length=max_length, eos_token_id=new_config.eos_token_id, device=torch_device
        )
        self.assertEqual(logits_processor.eos_token_id, eos_token_id)

    def test_serialize_generation_exponential_decay_length_penalty(self):
        """Tests that GenerationConfig is serialized and ExponentialDecayLengthPenalty is initialized with regulation_start and regulation_factor"""
        eos_token_id = 0
        penalty_start = 5
        penalty_factor = 1.1
        input_ids_seq_length = 10
        exponential_decay_length_penalty = (penalty_start, penalty_factor)

        generation_config = GenerationConfig(exponential_decay_length_penalty=exponential_decay_length_penalty)
        with tempfile.TemporaryDirectory("test-generation-config") as tmp_dir:
            generation_config.save_pretrained(tmp_dir)
            new_config = GenerationConfig.from_pretrained(tmp_dir)
        self.assertEqual(new_config.exponential_decay_length_penalty, [penalty_start, penalty_factor])

        exponential_decay_processor = ExponentialDecayLengthPenalty(
            exponential_decay_length_penalty=new_config.exponential_decay_length_penalty,
            eos_token_id=eos_token_id,
            input_ids_seq_length=input_ids_seq_length,
        )
        self.assertEqual(
            exponential_decay_processor.regulation_start, exponential_decay_length_penalty[0] + input_ids_seq_length
        )
        self.assertEqual(exponential_decay_processor.regulation_factor, exponential_decay_length_penalty[1])

    def test_serialize_generation_begin_suppress_tokens(self):
        """Tests that GenerationConfig is serialized and SuppressTokensAtBeginLogitsProcessor is initialized with begin_suppress_token and begin_index"""

        begin_suppress_tokens = [220, 50256]
        begin_index = 0
        generation_config = GenerationConfig(begin_suppress_tokens=begin_suppress_tokens)
        with tempfile.TemporaryDirectory("test-generation-config") as tmp_dir:
            generation_config.save_pretrained(tmp_dir)
            new_config = GenerationConfig.from_pretrained(tmp_dir)
        self.assertSequenceEqual(new_config.begin_suppress_tokens, begin_suppress_tokens)

        suppress_processor = SuppressTokensAtBeginLogitsProcessor(
            begin_suppress_tokens=new_config.begin_suppress_tokens, begin_index=begin_index
        )
        self.assertSequenceEqual(suppress_processor.begin_suppress_tokens, begin_suppress_tokens)
        self.assertEqual(suppress_processor.begin_index, begin_index)

    def test_serialize_generation_suppress_tokens(self):
        """Tests that GenerationConfig is serialized and SuppressTokensLogitsProcessor is initialized with suppress_token"""
        suppress_tokens = [220, 50256]

        generation_config = GenerationConfig(suppress_tokens=suppress_tokens)
        with tempfile.TemporaryDirectory("test-generation-config") as tmp_dir:
            generation_config.save_pretrained(tmp_dir)
            new_config = GenerationConfig.from_pretrained(tmp_dir)
        self.assertSequenceEqual(new_config.suppress_tokens, suppress_tokens)

        suppress_processor = SuppressTokensLogitsProcessor(suppress_tokens=new_config.suppress_tokens)
        self.assertSequenceEqual(suppress_processor.suppress_tokens, suppress_tokens)

    def test_serialize_generation_guidance_scale(self):
        """Tests that GenerationConfig is serialized and ClassifierFreeGuidanceLogitsProcessor is initialized with guidance_scale"""
        guidance_scale = 2.0
        generation_config = GenerationConfig(guidance_scale=guidance_scale)
        with tempfile.TemporaryDirectory("test-generation-config") as tmp_dir:
            generation_config.save_pretrained(tmp_dir)
            new_config = GenerationConfig.from_pretrained(tmp_dir)
        self.assertEqual(new_config.guidance_scale, guidance_scale)

        classifier_processor = ClassifierFreeGuidanceLogitsProcessor(guidance_scale=new_config.guidance_scale)
        self.assertEqual(classifier_processor.guidance_scale, guidance_scale)

    def test_serialize_generation_guidance_scale_unbatched(self):
        """Tests that GenerationConfig is serialized and UnbatchedClassifierFreeGuidanceLogitsProcessor is initialized with guidance_scale"""
        guidance_scale = 2.0

        input_ids = torch.LongTensor([[0]])

        generation_config = GenerationConfig(guidance_scale=guidance_scale)
        with tempfile.TemporaryDirectory("test-generation-config") as tmp_dir:
            generation_config.save_pretrained(tmp_dir)
            new_config = GenerationConfig.from_pretrained(tmp_dir)
        self.assertEqual(new_config.guidance_scale, guidance_scale)

        cfg = UnbatchedClassifierFreeGuidanceLogitsProcessor(new_config.guidance_scale, {}, input_ids)
        self.assertEqual(cfg.guidance_scale, guidance_scale)

    def test_serialize_generation_watermarking_config(self):
        """Tests that GenerationConfig is serialized and WatermarkLogitsProcessor is initialized with WatermarkingConfig parameters"""

        vocab_size = 20
        bias = 2.0
        greenlist_ratio = 0.5
        hashing_key = 10
        seeding_scheme = "lefthash"
        context_width = 10
        watermarking_config = WatermarkingConfig(
            bias=bias,
            greenlist_ratio=greenlist_ratio,
            hashing_key=hashing_key,
            seeding_scheme=seeding_scheme,
            context_width=context_width,
        )
        generation_config = GenerationConfig(watermarking_config=watermarking_config)

        with tempfile.TemporaryDirectory("test-generation-config") as tmp_dir:
            generation_config.save_pretrained(tmp_dir)
            new_config = GenerationConfig.from_pretrained(tmp_dir)
        self.assertEqual(new_config.watermarking_config.bias, bias)
        self.assertEqual(new_config.watermarking_config.greenlist_ratio, greenlist_ratio)
        self.assertEqual(new_config.watermarking_config.hashing_key, hashing_key)
        self.assertEqual(new_config.watermarking_config.seeding_scheme, seeding_scheme)
        self.assertEqual(new_config.watermarking_config.context_width, context_width)

        watermark = WatermarkLogitsProcessor(
            vocab_size=vocab_size,
            device=torch_device,
            greenlist_ratio=new_config.watermarking_config.greenlist_ratio,
            bias=new_config.watermarking_config.bias,
            hashing_key=new_config.watermarking_config.hashing_key,
            seeding_scheme=new_config.watermarking_config.seeding_scheme,
            context_width=new_config.watermarking_config.context_width,
        )
        self.assertEqual(watermark.bias, bias)
        self.assertEqual(watermark.greenlist_size, int(vocab_size * greenlist_ratio))
        self.assertEqual(watermark.hash_key, hashing_key)
        self.assertEqual(watermark.seeding_scheme, seeding_scheme)
        self.assertEqual(watermark.context_width, context_width)


@is_staging_test
class ConfigPushToHubTester(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._token = TOKEN
        HfFolder.save_token(TOKEN)

    def test_push_to_hub(self):
        with TemporaryHubRepo(token=self._token) as tmp_repo:
            config = GenerationConfig(
                do_sample=True,
                temperature=0.7,
                length_penalty=1.0,
            )
            config.push_to_hub(tmp_repo.repo_id, token=self._token)

            new_config = GenerationConfig.from_pretrained(tmp_repo.repo_id)
            for k, v in config.to_dict().items():
                if k != "transformers_version":
                    self.assertEqual(v, getattr(new_config, k))

    def test_push_to_hub_via_save_pretrained(self):
        with TemporaryHubRepo(token=self._token) as tmp_repo:
            config = GenerationConfig(
                do_sample=True,
                temperature=0.7,
                length_penalty=1.0,
            )
            # Push to hub via save_pretrained
            with tempfile.TemporaryDirectory() as tmp_dir:
                config.save_pretrained(tmp_dir, repo_id=tmp_repo.repo_id, push_to_hub=True, token=self._token)

            new_config = GenerationConfig.from_pretrained(tmp_repo.repo_id)
            for k, v in config.to_dict().items():
                if k != "transformers_version":
                    self.assertEqual(v, getattr(new_config, k))

    def test_push_to_hub_in_organization(self):
        with TemporaryHubRepo(namespace="valid_org", token=self._token) as tmp_repo:
            config = GenerationConfig(
                do_sample=True,
                temperature=0.7,
                length_penalty=1.0,
            )
            config.push_to_hub(tmp_repo.repo_id, token=self._token)

            new_config = GenerationConfig.from_pretrained(tmp_repo.repo_id)
            for k, v in config.to_dict().items():
                if k != "transformers_version":
                    self.assertEqual(v, getattr(new_config, k))

    def test_push_to_hub_in_organization_via_save_pretrained(self):
        with TemporaryHubRepo(namespace="valid_org", token=self._token) as tmp_repo:
            config = GenerationConfig(
                do_sample=True,
                temperature=0.7,
                length_penalty=1.0,
            )
            # Push to hub via save_pretrained
            with tempfile.TemporaryDirectory() as tmp_dir:
                config.save_pretrained(tmp_dir, repo_id=tmp_repo.repo_id, push_to_hub=True, token=self._token)

            new_config = GenerationConfig.from_pretrained(tmp_repo.repo_id)
            for k, v in config.to_dict().items():
                if k != "transformers_version":
                    self.assertEqual(v, getattr(new_config, k))

    def test_push_to_hub_on_pr_revision(self):
        with TemporaryHubRepo(token=self._token) as tmp_repo:
            # create a PR
            pr = create_pull_request(repo_id=tmp_repo.repo_id, title="Test PR", token=self._token)
            revision = f"refs/pr/{pr.num}"

            # push to PR ref
            config = GenerationConfig(
                do_sample=True,
                temperature=0.7,
                length_penalty=1.0,
            )
            config.push_to_hub(tmp_repo.repo_id, token=self._token, revision=revision)

            # load from PR ref
            new_config = GenerationConfig.from_pretrained(tmp_repo.repo_id, revision=revision)
            for k, v in config.to_dict().items():
                if k != "transformers_version":
                    self.assertEqual(v, getattr(new_config, k))
