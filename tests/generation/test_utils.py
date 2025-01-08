# coding=utf-8
# Copyright 2020 The HuggingFace Team Inc.
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


import collections
import copy
import gc
import inspect
import tempfile
import unittest
import warnings

import numpy as np
import pytest
from parameterized import parameterized

from transformers import AutoConfig, is_torch_available, pipeline, set_seed
from transformers.testing_utils import (
    is_flaky,
    require_accelerate,
    require_flash_attn,
    require_optimum_quanto,
    require_torch,
    require_torch_gpu,
    require_torch_multi_accelerator,
    require_torch_multi_gpu,
    require_torch_sdpa,
    set_config_for_less_flaky_test,
    set_model_for_less_flaky_test,
    set_model_tester_for_less_flaky_test,
    slow,
    torch_device,
)

from ..test_modeling_common import floats_tensor, ids_tensor
from .test_framework_agnostic import GenerationIntegrationTestsMixin


if is_torch_available():
    import torch
    import torch.nn.functional as F

    from transformers import (
        AutoModelForCausalLM,
        AutoModelForSeq2SeqLM,
        AutoModelForSpeechSeq2Seq,
        AutoModelForVision2Seq,
        AutoProcessor,
        AutoTokenizer,
        BartForCausalLM,
        BartForConditionalGeneration,
        BartTokenizer,
        GPT2LMHeadModel,
        GPT2Tokenizer,
        ImageGPTForCausalImageModeling,
        SpeechEncoderDecoderModel,
        T5ForConditionalGeneration,
    )
    from transformers.cache_utils import DynamicCache, EncoderDecoderCache, QuantoQuantizedCache, StaticCache
    from transformers.generation import (
        BeamSampleDecoderOnlyOutput,
        BeamSampleEncoderDecoderOutput,
        BeamSearchDecoderOnlyOutput,
        BeamSearchEncoderDecoderOutput,
        DisjunctiveConstraint,
        GenerateBeamDecoderOnlyOutput,
        GenerateBeamEncoderDecoderOutput,
        GenerateDecoderOnlyOutput,
        GenerateEncoderDecoderOutput,
        GenerationConfig,
        GreedySearchDecoderOnlyOutput,
        GreedySearchEncoderDecoderOutput,
        LogitsProcessorList,
        MaxLengthCriteria,
        MinLengthLogitsProcessor,
        PhrasalConstraint,
        PromptLookupCandidateGenerator,
        SampleDecoderOnlyOutput,
        SampleEncoderDecoderOutput,
        StoppingCriteria,
        StoppingCriteriaList,
        SynthIDTextWatermarkingConfig,
        WatermarkDetector,
        WatermarkingConfig,
    )
    from transformers.generation.candidate_generator import (
        AssistedCandidateGenerator,
        AssistedCandidateGeneratorDifferentTokenizers,
    )
    from transformers.generation.utils import _speculative_sampling

from unittest.mock import patch

from transformers.utils import is_sklearn_available


class GenerationTesterMixin:
    input_name = "input_ids"
    model_tester = None
    all_generative_model_classes = ()
    max_new_tokens = 3

    def prepare_config_and_inputs_for_generate(self, batch_size=2):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        # We don't want a few model inputs in our model input dictionary for generation tests
        input_keys_to_ignore = [
            # we don't want to mask attention heads
            "head_mask",
            "decoder_head_mask",
            "cross_attn_head_mask",
            # we don't want encoder-decoder models to start from filled decoder ids
            "decoder_input_ids",
            "decoder_attention_mask",
            # we'll set cache use in each test differently
            "use_cache",
            # Ignore labels if it is in the input dict
            "labels",
            # model-specific exceptions should overload/overwrite this function
        ]
        filtered_inputs_dict = {
            k: v[:batch_size, ...] if isinstance(v, torch.Tensor) else v
            for k, v in inputs_dict.items()
            if k not in input_keys_to_ignore
        }

        # It is important set `eos_token_id` to `None` to avoid early stopping (would break for length-based checks)
        text_gen_config = config.get_text_config(decoder=True)
        if text_gen_config.eos_token_id is not None and text_gen_config.pad_token_id is None:
            text_gen_config.pad_token_id = (
                text_gen_config.eos_token_id
                if isinstance(text_gen_config.eos_token_id, int)
                else text_gen_config.eos_token_id[0]
            )
        text_gen_config.eos_token_id = None
        text_gen_config.forced_eos_token_id = None

        return config, filtered_inputs_dict

    def _check_similar_generate_outputs(self, output_1, output_2, atol=1e-5, rtol=1e-5):
        """
        Checks whether a pair of generate outputs are similar. Two `generate` call outputs are considered similar in
        the following siturations:
        1. The sequences are the same
        2. The sequences are different, but the scores up to (and including) the first mismatch are nearly identical
        """
        # scores doesn't include data regarding decoder input tokens
        decoder_input_length = output_1.sequences.shape[1] - len(output_1.scores)
        output_matches = output_1.sequences == output_2.sequences
        has_matching_outputs = output_matches.all()
        has_matching_scores = None
        if not has_matching_outputs:
            for batch_idx in range(output_1.sequences.shape[0]):
                batch_matches = output_matches[batch_idx]
                if batch_matches.all():
                    continue
                first_mismatch_idx = batch_matches.int().argmin()  # gets the index of the first False
                first_mismatch_idx -= decoder_input_length
                output_1_first_mismatch_scores = output_1.scores[first_mismatch_idx][batch_idx]
                output_2_first_mismatch_scores = output_2.scores[first_mismatch_idx][batch_idx]
                has_matching_scores = torch.allclose(
                    output_1_first_mismatch_scores, output_2_first_mismatch_scores, rtol=atol, atol=rtol
                )
                if not has_matching_scores:
                    break
        self.assertTrue(has_matching_outputs or has_matching_scores)

    def _get_logits_processor_kwargs(self, do_sample=False, config=None):
        logits_processor_kwargs = {
            "bad_words_ids": [[1, 0]],
            "repetition_penalty": 1.2,
            "remove_invalid_values": True,
        }
        if do_sample:
            logits_processor_kwargs.update(
                {
                    "top_k": 10,
                    "top_p": 0.7,
                    "temperature": 0.7,
                }
            )
        # TODO (joao, raushan): see this comment for a long-term fix
        # https://github.com/huggingface/transformers/pull/33593#issuecomment-2361824264)
        # This is a band-aid for VLM models, to ensure they don't generate image/video tokens which would cause them
        # to crash. On pretrained models this isn't a risk, as they are trained to not generate these tokens.
        if config is not None:
            for key in [
                "image_token_index",
                "image_token_id",
                "video_token_index",
                "video_token_id",
                "vision_start_token_id",
            ]:
                token_index = getattr(config, key, None)
                if token_index is not None and token_index < config.get_text_config().vocab_size:
                    logits_processor_kwargs["bad_words_ids"].append([token_index])

        return logits_processor_kwargs

    def _get_beam_kwargs(self, num_return_sequences=1):
        beam_kwargs = {
            "early_stopping": False,
            "length_penalty": 2.0,
            "num_beams": 2,
            "num_return_sequences": num_return_sequences,
        }
        return beam_kwargs

    def _get_diverse_beam_kwargs(self, num_return_sequences=1):
        beam_kwargs = {
            "early_stopping": False,
            "length_penalty": 2.0,
            "num_beams": 2,
            "num_return_sequences": num_return_sequences,
            "num_beam_groups": 2,  # one beam per group
            "diversity_penalty": 2.0,
        }
        return beam_kwargs

    def _get_constrained_beam_kwargs(self, num_return_sequences=1):
        beam_kwargs = {
            "early_stopping": False,
            "length_penalty": 2.0,
            "num_beams": num_return_sequences * 4,
            "num_return_sequences": num_return_sequences,
        }
        return beam_kwargs

    def _greedy_generate(
        self,
        model,
        inputs_dict,
        output_scores=False,
        output_logits=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict_in_generate=False,
        use_cache=True,
    ):
        logits_processor_kwargs = self._get_logits_processor_kwargs(do_sample=False, config=model.config)
        output_generate = model.generate(
            do_sample=False,
            num_beams=1,
            max_new_tokens=self.max_new_tokens,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_scores=output_scores,
            output_logits=output_logits,
            return_dict_in_generate=return_dict_in_generate,
            use_cache=use_cache,
            **logits_processor_kwargs,
            **inputs_dict,
        )

        return output_generate

    def _sample_generate(
        self,
        model,
        inputs_dict,
        num_return_sequences,
        output_scores=False,
        output_logits=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict_in_generate=False,
        use_cache=True,
    ):
        torch.manual_seed(0)
        logits_processor_kwargs = self._get_logits_processor_kwargs(do_sample=True, config=model.config)
        output_generate = model.generate(
            do_sample=True,
            num_beams=1,
            max_new_tokens=self.max_new_tokens,
            num_return_sequences=num_return_sequences,
            output_scores=output_scores,
            output_logits=output_logits,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict_in_generate=return_dict_in_generate,
            use_cache=use_cache,
            **logits_processor_kwargs,
            **inputs_dict,
        )

        return output_generate

    def _beam_search_generate(
        self,
        model,
        inputs_dict,
        beam_kwargs,
        output_scores=False,
        output_logits=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict_in_generate=False,
        use_cache=True,
    ):
        logits_processor_kwargs = self._get_logits_processor_kwargs(do_sample=False, config=model.config)
        output_generate = model.generate(
            do_sample=False,
            max_new_tokens=self.max_new_tokens,
            output_scores=output_scores,
            output_logits=output_logits,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict_in_generate=return_dict_in_generate,
            use_cache=use_cache,
            **beam_kwargs,
            **logits_processor_kwargs,
            **inputs_dict,
        )

        return output_generate

    def _beam_sample_generate(
        self,
        model,
        inputs_dict,
        beam_kwargs,
        output_scores=False,
        output_logits=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict_in_generate=False,
        use_cache=True,
    ):
        torch.manual_seed(0)
        logits_processor_kwargs = self._get_logits_processor_kwargs(do_sample=True, config=model.config)
        output_generate = model.generate(
            do_sample=True,
            max_new_tokens=self.max_new_tokens,
            output_scores=output_scores,
            output_logits=output_logits,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict_in_generate=return_dict_in_generate,
            use_cache=use_cache,
            **beam_kwargs,
            **logits_processor_kwargs,
            **inputs_dict,
        )

        return output_generate

    def _group_beam_search_generate(
        self,
        model,
        inputs_dict,
        beam_kwargs,
        output_scores=False,
        output_logits=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict_in_generate=False,
        use_cache=True,
    ):
        logits_processor_kwargs = self._get_logits_processor_kwargs(do_sample=False, config=model.config)
        output_generate = model.generate(
            do_sample=False,
            max_new_tokens=self.max_new_tokens,
            output_scores=output_scores,
            output_logits=output_logits,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict_in_generate=return_dict_in_generate,
            use_cache=use_cache,
            **beam_kwargs,
            **logits_processor_kwargs,
            **inputs_dict,
        )

        return output_generate

    def _constrained_beam_search_generate(
        self,
        model,
        inputs_dict,
        constraints,
        beam_kwargs,
        output_scores=False,
        output_logits=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict_in_generate=False,
        use_cache=True,
    ):
        logits_processor_kwargs = self._get_logits_processor_kwargs(do_sample=False, config=model.config)
        output_generate = model.generate(
            do_sample=False,
            max_new_tokens=self.max_new_tokens,
            output_scores=output_scores,
            output_logits=output_logits,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict_in_generate=return_dict_in_generate,
            constraints=constraints,
            use_cache=use_cache,
            **beam_kwargs,
            **logits_processor_kwargs,
            **inputs_dict,
        )

        return output_generate

    def _contrastive_generate(
        self,
        model,
        inputs_dict,
        output_scores=False,
        output_logits=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict_in_generate=False,
        use_cache=True,
    ):
        contrastive_search_kwargs = {
            "penalty_alpha": 0.6,
            "top_k": 5,
        }

        logits_processor_kwargs = self._get_logits_processor_kwargs(do_sample=False, config=model.config)
        output_generate = model.generate(
            do_sample=False,
            num_beams=1,
            max_new_tokens=self.max_new_tokens,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_scores=output_scores,
            output_logits=output_logits,
            return_dict_in_generate=return_dict_in_generate,
            use_cache=use_cache,
            **logits_processor_kwargs,
            **contrastive_search_kwargs,
            **inputs_dict,
        )

        return output_generate

    @pytest.mark.generate
    def test_greedy_generate(self):
        for model_class in self.all_generative_model_classes:
            config, inputs_dict = self.prepare_config_and_inputs_for_generate()

            model = model_class(config).to(torch_device).eval()
            output_generate = self._greedy_generate(model=model, inputs_dict=inputs_dict)

            if model.config.is_encoder_decoder:
                self.assertTrue(output_generate.shape[-1] == self.max_new_tokens + 1)
            else:
                self.assertTrue(output_generate.shape[-1] == self.max_new_tokens + inputs_dict["input_ids"].shape[-1])

    @pytest.mark.generate
    def test_greedy_generate_dict_outputs(self):
        for model_class in self.all_generative_model_classes:
            config, inputs_dict = self.prepare_config_and_inputs_for_generate()

            model = model_class(config).to(torch_device).eval()
            output_generate = self._greedy_generate(
                model=model,
                inputs_dict=inputs_dict,
                output_scores=True,
                output_logits=True,
                output_hidden_states=True,
                output_attentions=self.has_attentions,
                return_dict_in_generate=True,
                use_cache=False,
            )

            if model.config.is_encoder_decoder:
                self.assertTrue(output_generate.sequences.shape[-1] == self.max_new_tokens + 1)
                self.assertIsInstance(output_generate, GenerateEncoderDecoderOutput)
                # Retrocompatibility check
                self.assertIsInstance(output_generate, GreedySearchEncoderDecoderOutput)
            else:
                self.assertTrue(
                    output_generate.sequences.shape[-1] == self.max_new_tokens + inputs_dict["input_ids"].shape[-1]
                )
                self.assertIsInstance(output_generate, GenerateDecoderOnlyOutput)
                # Retrocompatibility check
                self.assertIsInstance(output_generate, GreedySearchDecoderOnlyOutput)

            self._check_outputs(output_generate, model.config)

    @pytest.mark.generate
    def test_greedy_generate_dict_outputs_use_cache(self):
        for model_class in self.all_generative_model_classes:
            config, inputs_dict = self.prepare_config_and_inputs_for_generate()

            if not hasattr(config, "use_cache"):
                self.skipTest(reason=f"{model_class.__name__} doesn't support caching")
            if any(model_name in model_class.__name__.lower() for model_name in ["rwkv"]):
                self.skipTest(reason="Won't fix: model with non-standard dictionary output shapes")

            config.is_decoder = True
            model = model_class(config).to(torch_device).eval()
            output_generate = self._greedy_generate(
                model=model,
                inputs_dict=inputs_dict,
                output_scores=True,
                output_logits=True,
                output_hidden_states=True,
                output_attentions=self.has_attentions,
                return_dict_in_generate=True,
                use_cache=True,  # Enable cache
            )

            if model.config.is_encoder_decoder:
                self.assertTrue(output_generate.sequences.shape[-1] == self.max_new_tokens + 1)
            else:
                self.assertTrue(
                    output_generate.sequences.shape[-1] == self.max_new_tokens + inputs_dict["input_ids"].shape[-1]
                )

            self._check_outputs(output_generate, model.config, use_cache=True)

    @pytest.mark.generate
    def test_sample_generate(self):
        for model_class in self.all_generative_model_classes:
            config, inputs_dict = self.prepare_config_and_inputs_for_generate()

            model = model_class(config).to(torch_device).eval()
            output_generate = self._sample_generate(model=model, inputs_dict=inputs_dict, num_return_sequences=1)

            if model.config.is_encoder_decoder:
                self.assertTrue(output_generate.shape[-1] == self.max_new_tokens + 1)
            else:
                self.assertTrue(output_generate.shape[-1] == self.max_new_tokens + inputs_dict["input_ids"].shape[-1])

    @pytest.mark.generate
    def test_sample_generate_dict_output(self):
        for model_class in self.all_generative_model_classes:
            config, inputs_dict = self.prepare_config_and_inputs_for_generate()

            model = model_class(config).to(torch_device).eval()
            output_generate = self._sample_generate(
                model=model,
                inputs_dict=inputs_dict,
                num_return_sequences=2,
                output_scores=True,
                output_logits=True,
                output_hidden_states=True,
                output_attentions=self.has_attentions,
                return_dict_in_generate=True,
                use_cache=False,
            )

            if model.config.is_encoder_decoder:
                self.assertTrue(output_generate.sequences.shape[-1] == self.max_new_tokens + 1)
                self.assertIsInstance(output_generate, GenerateEncoderDecoderOutput)
                # Retrocompatibility check
                self.assertIsInstance(output_generate, SampleEncoderDecoderOutput)
            else:
                self.assertTrue(
                    output_generate.sequences.shape[-1] == self.max_new_tokens + inputs_dict["input_ids"].shape[-1]
                )
                self.assertIsInstance(output_generate, GenerateDecoderOnlyOutput)
                # Retrocompatibility check
                self.assertIsInstance(output_generate, SampleDecoderOnlyOutput)

            self._check_outputs(output_generate, model.config, num_return_sequences=2)

    @pytest.mark.generate
    def test_beam_search_generate(self):
        for model_class in self.all_generative_model_classes:
            config, inputs_dict = self.prepare_config_and_inputs_for_generate()

            model = model_class(config).to(torch_device).eval()

            beam_kwargs = self._get_beam_kwargs()
            output_generate = self._beam_search_generate(model=model, inputs_dict=inputs_dict, beam_kwargs=beam_kwargs)

            if model.config.is_encoder_decoder:
                self.assertTrue(output_generate.shape[-1] == self.max_new_tokens + 1)
            else:
                self.assertTrue(output_generate.shape[-1] == self.max_new_tokens + inputs_dict["input_ids"].shape[-1])

    @pytest.mark.generate
    def test_beam_search_generate_dict_output(self):
        for model_class in self.all_generative_model_classes:
            config, inputs_dict = self.prepare_config_and_inputs_for_generate()

            model = model_class(config).to(torch_device).eval()
            beam_kwargs = self._get_beam_kwargs()
            output_generate = self._beam_search_generate(
                model=model,
                inputs_dict=inputs_dict,
                beam_kwargs=beam_kwargs,
                output_scores=True,
                output_logits=True,
                output_hidden_states=True,
                output_attentions=self.has_attentions,
                return_dict_in_generate=True,
                use_cache=False,
            )
            if model.config.is_encoder_decoder:
                self.assertTrue(output_generate.sequences.shape[-1] == self.max_new_tokens + 1)
                self.assertIsInstance(output_generate, GenerateBeamEncoderDecoderOutput)
                # Retrocompatibility check
                self.assertIsInstance(output_generate, BeamSearchEncoderDecoderOutput)
            else:
                self.assertTrue(
                    output_generate.sequences.shape[-1] == self.max_new_tokens + inputs_dict["input_ids"].shape[-1]
                )
                self.assertIsInstance(output_generate, GenerateBeamDecoderOnlyOutput)
                # Retrocompatibility check
                self.assertIsInstance(output_generate, BeamSearchDecoderOnlyOutput)

            self._check_outputs(
                output_generate,
                model.config,
                num_return_sequences=beam_kwargs["num_return_sequences"],
                num_beams=beam_kwargs["num_beams"],
            )

    @pytest.mark.generate
    def test_beam_search_generate_dict_outputs_use_cache(self):
        for model_class in self.all_generative_model_classes:
            config, inputs_dict = self.prepare_config_and_inputs_for_generate()

            if not hasattr(config, "use_cache"):
                self.skipTest(reason=f"{model_class.__name__} doesn't support caching")
            if any(model_name in model_class.__name__.lower() for model_name in ["rwkv"]):
                self.skipTest(reason="Won't fix: model with non-standard dictionary output shapes")

            model = model_class(config).to(torch_device).eval()
            beam_kwargs = self._get_beam_kwargs()

            config.is_decoder = True
            model = model_class(config).to(torch_device).eval()
            output_generate = self._beam_search_generate(
                model=model,
                inputs_dict=inputs_dict,
                beam_kwargs=beam_kwargs,
                output_scores=True,
                output_logits=True,
                output_hidden_states=True,
                output_attentions=self.has_attentions,
                return_dict_in_generate=True,
                use_cache=True,  # Enable cache
            )

            if model.config.is_encoder_decoder:
                self.assertTrue(output_generate.sequences.shape[-1] == self.max_new_tokens + 1)
            else:
                self.assertTrue(
                    output_generate.sequences.shape[-1] == self.max_new_tokens + inputs_dict["input_ids"].shape[-1]
                )

            self._check_outputs(
                output_generate,
                model.config,
                use_cache=True,
                num_return_sequences=beam_kwargs["num_return_sequences"],
                num_beams=beam_kwargs["num_beams"],
            )

    @require_accelerate
    @require_torch_multi_accelerator
    @pytest.mark.generate
    def test_model_parallel_beam_search(self):
        for model_class in self.all_generative_model_classes:
            if "xpu" in torch_device:
                return unittest.skip(reason="device_map='auto' does not work with XPU devices")

            if model_class._no_split_modules is None:
                continue

            config, inputs_dict = self.prepare_config_and_inputs_for_generate()

            model = model_class(config).eval()
            with tempfile.TemporaryDirectory() as tmp_dir:
                model.cpu().save_pretrained(tmp_dir)
                new_model = model_class.from_pretrained(tmp_dir, device_map="auto")

                new_model.generate(
                    max_new_tokens=self.max_new_tokens,
                    num_beams=2,
                    **inputs_dict,
                )

    @pytest.mark.generate
    def test_beam_sample_generate(self):
        for model_class in self.all_generative_model_classes:
            config, inputs_dict = self.prepare_config_and_inputs_for_generate()

            model = model_class(config).to(torch_device).eval()
            beam_kwargs = self._get_beam_kwargs()
            output_generate = self._beam_sample_generate(
                model=model,
                inputs_dict=inputs_dict,
                beam_kwargs=beam_kwargs,
            )

            if model.config.is_encoder_decoder:
                self.assertTrue(output_generate.shape[-1] == self.max_new_tokens + 1)
            else:
                self.assertTrue(output_generate.shape[-1] == self.max_new_tokens + inputs_dict["input_ids"].shape[-1])

    @pytest.mark.generate
    def test_beam_sample_generate_dict_output(self):
        for model_class in self.all_generative_model_classes:
            config, inputs_dict = self.prepare_config_and_inputs_for_generate()

            model = model_class(config).to(torch_device).eval()
            beam_kwargs = self._get_beam_kwargs()

            output_generate = self._beam_sample_generate(
                model=model,
                inputs_dict=inputs_dict,
                beam_kwargs=beam_kwargs,
                output_scores=True,
                output_logits=True,
                output_hidden_states=True,
                output_attentions=self.has_attentions,
                return_dict_in_generate=True,
                use_cache=False,
            )

            if model.config.is_encoder_decoder:
                self.assertTrue(output_generate.sequences.shape[-1] == self.max_new_tokens + 1)
                self.assertIsInstance(output_generate, GenerateBeamEncoderDecoderOutput)
                # Retrocompatibility check
                self.assertIsInstance(output_generate, BeamSampleEncoderDecoderOutput)
            else:
                self.assertTrue(
                    output_generate.sequences.shape[-1] == self.max_new_tokens + inputs_dict["input_ids"].shape[-1]
                )
                self.assertIsInstance(output_generate, GenerateBeamDecoderOnlyOutput)
                # Retrocompatibility check
                self.assertIsInstance(output_generate, BeamSampleDecoderOnlyOutput)

            self._check_outputs(
                output_generate,
                model.config,
                num_return_sequences=beam_kwargs["num_return_sequences"],
                num_beams=beam_kwargs["num_beams"],
            )

    @pytest.mark.generate
    def test_generate_without_input_ids(self):
        config, _ = self.prepare_config_and_inputs_for_generate()

        # if no bos token id => cannot generate from None
        if config.bos_token_id is None:
            self.skipTest(reason="bos_token_id is None")

        # hack in case they are equal, otherwise the attn mask will be [0]
        if config.bos_token_id == config.pad_token_id:
            config.pad_token_id = None

        for model_class in self.all_generative_model_classes:
            model = model_class(config).to(torch_device)
            model.eval()

            output_ids_generate = model.generate(
                do_sample=False, max_new_tokens=self.max_new_tokens, remove_invalid_values=True
            )
            self.assertIsNotNone(output_ids_generate)

    @pytest.mark.generate
    def test_group_beam_search_generate(self):
        for model_class in self.all_generative_model_classes:
            config, inputs_dict = self.prepare_config_and_inputs_for_generate()

            model = model_class(config).to(torch_device).eval()
            # check `generate()` and `group_beam_search()` are equal
            beam_kwargs = self._get_diverse_beam_kwargs()
            output_generate = self._group_beam_search_generate(
                model=model,
                inputs_dict=inputs_dict,
                beam_kwargs=beam_kwargs,
            )
            if model.config.is_encoder_decoder:
                self.assertTrue(output_generate.shape[-1] == self.max_new_tokens + 1)
            else:
                self.assertTrue(output_generate.shape[-1] == self.max_new_tokens + inputs_dict["input_ids"].shape[-1])

            # check `group_beam_search` for higher than 1 `num_return_sequences`
            num_return_sequences = 2
            beam_kwargs = self._get_diverse_beam_kwargs(num_return_sequences=num_return_sequences)
            output_generate = self._group_beam_search_generate(
                model=model,
                inputs_dict=inputs_dict,
                beam_kwargs=beam_kwargs,
            )
            if model.config.is_encoder_decoder:
                self.assertTrue(output_generate.shape[-1] == self.max_new_tokens + 1)
            else:
                self.assertTrue(output_generate.shape[-1] == self.max_new_tokens + inputs_dict["input_ids"].shape[-1])

    @pytest.mark.generate
    def test_group_beam_search_generate_dict_output(self):
        for model_class in self.all_generative_model_classes:
            config, inputs_dict = self.prepare_config_and_inputs_for_generate()

            model = model_class(config).to(torch_device).eval()
            beam_kwargs = self._get_diverse_beam_kwargs()
            output_generate = self._group_beam_search_generate(
                model=model,
                inputs_dict=inputs_dict,
                beam_kwargs=beam_kwargs,
                output_scores=True,
                output_logits=True,
                output_hidden_states=True,
                output_attentions=self.has_attentions,
                return_dict_in_generate=True,
                use_cache=False,
            )
            if model.config.is_encoder_decoder:
                self.assertTrue(output_generate.sequences.shape[-1] == self.max_new_tokens + 1)
                self.assertIsInstance(output_generate, GenerateBeamEncoderDecoderOutput)
                # Retrocompatibility check
                self.assertIsInstance(output_generate, BeamSearchEncoderDecoderOutput)
            else:
                self.assertTrue(
                    output_generate.sequences.shape[-1] == self.max_new_tokens + inputs_dict["input_ids"].shape[-1]
                )
                self.assertIsInstance(output_generate, GenerateBeamDecoderOnlyOutput)
                # Retrocompatibility check
                self.assertIsInstance(output_generate, BeamSearchDecoderOnlyOutput)

            self._check_outputs(
                output_generate,
                model.config,
                num_return_sequences=beam_kwargs["num_return_sequences"],
                num_beams=beam_kwargs["num_beams"],
            )

    # TODO: @gante check why it is flaky
    @is_flaky()
    @pytest.mark.generate
    def test_constrained_beam_search_generate(self):
        for model_class in self.all_generative_model_classes:
            config, inputs_dict = self.prepare_config_and_inputs_for_generate()

            model = model_class(config).to(torch_device).eval()

            # Sample constraints
            min_id = 3
            max_id = config.get_text_config(decoder=True).vocab_size

            force_tokens = torch.randint(min_id, max_id, (1, 2)).tolist()[0]
            constraints = [
                PhrasalConstraint(force_tokens),
            ]

            beam_kwargs = self._get_constrained_beam_kwargs()
            output_generate = self._constrained_beam_search_generate(
                model=model,
                inputs_dict=inputs_dict,
                constraints=constraints,
                beam_kwargs=beam_kwargs,
            )

            if model.config.is_encoder_decoder:
                self.assertTrue(output_generate.shape[-1] == self.max_new_tokens + 1)
            else:
                self.assertTrue(output_generate.shape[-1] == self.max_new_tokens + inputs_dict["input_ids"].shape[-1])

            for generation_output in output_generate:
                self._check_sequence_inside_sequence(force_tokens, generation_output)

            # check`constrained_beam_search` for higher than 1 `num_return_sequences`
            # Sample constraints
            force_tokens = torch.randint(min_id, max_id, (1, 2)).tolist()[0]
            constraints = [
                PhrasalConstraint(force_tokens),
            ]

            beam_kwargs = self._get_constrained_beam_kwargs(num_return_sequences=2)

            output_generate = self._constrained_beam_search_generate(
                model=model,
                inputs_dict=inputs_dict,
                constraints=constraints,
                beam_kwargs=beam_kwargs,
            )

            if model.config.is_encoder_decoder:
                self.assertTrue(output_generate.shape[-1] == self.max_new_tokens + 1)
            else:
                self.assertTrue(output_generate.shape[-1] == self.max_new_tokens + inputs_dict["input_ids"].shape[-1])

            for generation_output in output_generate:
                self._check_sequence_inside_sequence(force_tokens, generation_output)

    @pytest.mark.generate
    def test_constrained_beam_search_generate_dict_output(self):
        for model_class in self.all_generative_model_classes:
            config, inputs_dict = self.prepare_config_and_inputs_for_generate()

            model = model_class(config).to(torch_device).eval()

            # Sample constraints
            min_id = 3
            max_id = model.config.get_text_config(decoder=True).vocab_size
            force_tokens = torch.randint(min_id, max_id, (1, 2)).tolist()[0]
            constraints = [
                PhrasalConstraint(force_tokens),
            ]

            beam_kwargs = self._get_constrained_beam_kwargs()
            output_generate = self._constrained_beam_search_generate(
                model=model,
                inputs_dict=inputs_dict,
                constraints=constraints,
                beam_kwargs=beam_kwargs,
                output_scores=True,
                output_logits=True,
                output_hidden_states=True,
                output_attentions=self.has_attentions,
                return_dict_in_generate=True,
                use_cache=False,
            )

            if model.config.is_encoder_decoder:
                self.assertTrue(output_generate.sequences.shape[-1] == self.max_new_tokens + 1)
                self.assertIsInstance(output_generate, GenerateBeamEncoderDecoderOutput)
                # Retrocompatibility check
                self.assertIsInstance(output_generate, BeamSearchEncoderDecoderOutput)
            else:
                self.assertTrue(
                    output_generate.sequences.shape[-1] == self.max_new_tokens + inputs_dict["input_ids"].shape[-1]
                )
                self.assertIsInstance(output_generate, GenerateBeamDecoderOnlyOutput)
                # Retrocompatibility check
                self.assertIsInstance(output_generate, BeamSearchDecoderOnlyOutput)

            self._check_outputs(
                output_generate,
                model.config,
                num_return_sequences=beam_kwargs["num_return_sequences"],
                num_beams=beam_kwargs["num_beams"],
            )

    @pytest.mark.generate
    def test_contrastive_generate(self):
        for model_class in self.all_generative_model_classes:
            if model_class._is_stateful:
                self.skipTest(reason="Stateful models don't support contrastive search generation")

            # won't fix: FSMT and Reformer have a different cache variable type (and format).
            if any(model_name in model_class.__name__.lower() for model_name in ["fsmt", "reformer"]):
                self.skipTest(reason="Won't fix: old model with different cache format")

            config, inputs_dict = self.prepare_config_and_inputs_for_generate()

            # NOTE: contrastive search only works with cache on at the moment.
            if not hasattr(config, "use_cache"):
                self.skipTest(reason=f"{model_class.__name__} doesn't support caching")
            config.is_decoder = True

            # test old generation output for backwards compatibility
            model = model_class(config).to(torch_device).eval()
            output_generate = self._contrastive_generate(
                model=model,
                inputs_dict=inputs_dict,
                use_cache=True,  # Enable cache
            )
            if model.config.is_encoder_decoder:
                self.assertTrue(output_generate.shape[-1] == self.max_new_tokens + 1)
            else:
                self.assertTrue(output_generate.shape[-1] == self.max_new_tokens + inputs_dict["input_ids"].shape[-1])

    @pytest.mark.generate
    def test_contrastive_generate_dict_outputs_use_cache(self):
        for model_class in self.all_generative_model_classes:
            if model_class._is_stateful:
                self.skipTest(reason="Stateful models don't support contrastive search generation")

            # won't fix: FSMT and Reformer have a different cache variable type (and format).
            if any(model_name in model_class.__name__.lower() for model_name in ["fsmt", "reformer"]):
                self.skipTest(reason="Won't fix: old model with different cache format")

            config, inputs_dict = self.prepare_config_and_inputs_for_generate()

            # NOTE: contrastive search only works with cache on at the moment.
            if not hasattr(config, "use_cache"):
                self.skipTest(reason=f"{model_class.__name__} doesn't support caching")
            config.is_decoder = True

            model = model_class(config).to(torch_device).eval()
            output_generate = self._contrastive_generate(
                model=model,
                inputs_dict=inputs_dict,
                output_scores=True,
                output_logits=True,
                output_hidden_states=True,
                output_attentions=self.has_attentions,
                return_dict_in_generate=True,
                use_cache=True,  # Enable cache
            )

            if model.config.is_encoder_decoder:
                self.assertTrue(output_generate.sequences.shape[-1] == self.max_new_tokens + 1)
            else:
                self.assertTrue(
                    output_generate.sequences.shape[-1] == self.max_new_tokens + inputs_dict["input_ids"].shape[-1]
                )

            self._check_outputs(output_generate, model.config, use_cache=True)

    @pytest.mark.generate
    def test_contrastive_generate_low_memory(self):
        # Check that choosing 'low_memory' does not change the model output
        for model_class in self.all_generative_model_classes:
            if model_class._is_stateful:
                self.skipTest(reason="Stateful models don't support contrastive search generation")

            if any(model_name in model_class.__name__.lower() for model_name in ["fsmt", "reformer", "speech2text"]):
                self.skipTest(reason="Won't fix: old model with different cache format")
            if any(model_name in model_class.__name__.lower() for model_name in ["gptbigcode"]):
                self.skipTest(reason="TODO: fix me")

            config, inputs_dict = self.prepare_config_and_inputs_for_generate(batch_size=1)

            # NOTE: contrastive search only works with cache on at the moment.
            if not hasattr(config, "use_cache"):
                self.skipTest(reason=f"{model_class.__name__} doesn't support caching")

            config.is_decoder = True

            # test output equality of low versus high memory
            model = model_class(config).to(torch_device).eval()

            low_output = model.generate(
                top_k=4,
                penalty_alpha=0.6,
                low_memory=True,
                max_new_tokens=self.max_new_tokens,
                **inputs_dict,
                use_cache=True,
            )

            high_output = model.generate(
                top_k=4,
                penalty_alpha=0.6,
                low_memory=False,
                max_new_tokens=self.max_new_tokens,
                **inputs_dict,
                use_cache=True,
            )
            self.assertListEqual(low_output.tolist(), high_output.tolist())

    @pytest.mark.generate
    def test_beam_search_low_memory(self):
        # Check that choosing 'low_memory' does not change the model output
        for model_class in self.all_generative_model_classes:
            if model_class._is_stateful:
                self.skipTest(reason="May fix in the future: need custom cache handling")
            if any(model_name in model_class.__name__.lower() for model_name in ["fsmt", "reformer"]):
                self.skipTest(reason="Won't fix: old model with different cache format")
            if any(
                model_name in model_class.__name__.lower()
                for model_name in [
                    "ctrl",
                    "gptbigcode",
                    "transo_xl",
                    "xlnet",
                    "cpm",
                    "jamba",
                ]
            ):
                self.skipTest(reason="May fix in the future: need model-specific fixes")

            config, inputs_dict = self.prepare_config_and_inputs_for_generate()
            # batch_size=1 is ok, but batch_size>1 will cause non-identical output

            config.use_cache = True
            config.is_decoder = True

            # test output equality of low versus high memory
            model = model_class(config).to(torch_device).eval()

            low_output = model.generate(
                **inputs_dict,
                max_new_tokens=8,
                num_beams=5,
                early_stopping=True,
                low_memory=True,
                use_cache=True,
            )

            high_output = model.generate(
                **inputs_dict,
                max_new_tokens=8,
                num_beams=5,
                early_stopping=True,
                low_memory=False,
                use_cache=True,
            )
            self.assertListEqual(low_output.tolist(), high_output.tolist())

    @pytest.mark.generate
    @parameterized.expand([("random",), ("same",)])
    def test_assisted_decoding_matches_greedy_search(self, assistant_type):
        # This test ensures that the assisted generation does not introduce output changes over greedy search.
        # See https://github.com/huggingface/transformers/issues/25420#issuecomment-1775317535 for more info.
        # NOTE: It breaks the pattern in the tests above, for multiple reasons:
        # - assisted_decoding, contrarily to the other methods, can't be called on its own (e.g. needs to
        # prepare the assistant encoder outputs in the main generate body);
        # - assisted_decoding does not support `use_cache = False`
        # - assisted_decoding does not support `batch_size > 1`

        for model_class in self.all_generative_model_classes:
            if model_class._is_stateful:
                self.skipTest(reason="Stateful models don't support assisted generation")
            if any(model_name in model_class.__name__.lower() for model_name in ["fsmt", "reformer"]):
                self.skipTest(reason="Won't fix: old model with different cache format")
            if any(
                model_name in model_class.__name__.lower()
                for model_name in [
                    "bigbirdpegasus",
                    "led",
                    "mega",
                    "moshi",
                    "speech2text",
                    "git",
                    "prophetnet",
                    "seamlessm4t",
                    "clvp",
                ]
            ):
                self.skipTest(reason="May fix in the future: need model-specific fixes")

            # enable cache
            config, inputs_dict = self.prepare_config_and_inputs_for_generate(batch_size=1)

            # NOTE: assisted generation only works with cache on at the moment.
            if not hasattr(config, "use_cache"):
                self.skipTest(reason=f"{model_class.__name__} doesn't support caching")

            config.is_decoder = True
            model = model_class(config).to(torch_device).eval()
            # Sets assisted generation arguments such that:
            # a) no EOS is generated, to ensure generation doesn't break early
            # b) the assistant model always generates two tokens when it is called, to ensure the input preparation of
            #    the assistant model is correct
            # c) there are at least two forward passes in the main model, to ensure the input preparation of
            #    the main model is correct
            generation_kwargs = {
                "eos_token_id": -1,  # see a)
                "max_new_tokens": 4,  # see c)
                "num_beams": 1,
                "do_sample": False,
                "output_scores": True,
                "output_logits": True,
                "output_hidden_states": True,
                "output_attentions": self.has_attentions,
                "return_dict_in_generate": True,
                "use_cache": True,
            }
            output_greedy = model.generate(**generation_kwargs, **inputs_dict)

            # test with the same assistant model or randomly init one
            # in the first case all candidate tokens are accepted, in the second none is accepted
            # case when some are accepted and some not is hard to reproduce, so let's hope this catches most errors :)
            if assistant_type == "random":
                assistant_model = model_class(config).to(torch_device).eval()
            else:
                assistant_model = model
            assistant_model.generation_config.num_assistant_tokens = 2  # see b)
            assistant_model.generation_config.num_assistant_tokens_schedule = "constant"  # see b)
            generation_kwargs.update({"assistant_model": assistant_model})
            output_assisted = model.generate(**generation_kwargs, **inputs_dict)

            # The two outputs must match and their shape must be as expected
            self._check_similar_generate_outputs(output_greedy, output_assisted)
            for output in (output_greedy, output_assisted):
                self._check_outputs(output, model.config, use_cache=True)

    @pytest.mark.generate
    def test_prompt_lookup_decoding_matches_greedy_search(self):
        # This test ensures that the prompt lookup generation does not introduce output changes over greedy search.
        # This test is mostly a copy of test_assisted_decoding_matches_greedy_search

        for model_class in self.all_generative_model_classes:
            if model_class._is_stateful:
                self.skipTest(reason="Stateful models don't support assisted generation")
            if any(model_name in model_class.__name__.lower() for model_name in ["fsmt", "reformer"]):
                self.skipTest(reason="Won't fix: old model with different cache format")
            if any(
                model_name in model_class.__name__.lower()
                for model_name in [
                    "bigbirdpegasus",
                    "led",
                    "mega",
                    "moshi",
                    "speech2text",
                    "git",
                    "prophetnet",
                    "seamlessm4t",
                    "clvp",
                    "fuyu",
                ]
            ):
                self.skipTest(reason="May fix in the future: need model-specific fixes")

            # enable cache
            config, inputs_dict = self.prepare_config_and_inputs_for_generate(batch_size=1)

            # NOTE: assisted generation only works with cache on at the moment.
            if not hasattr(config, "use_cache"):
                self.skipTest(reason=f"{model_class.__name__} doesn't support caching")

            config.is_decoder = True
            model = model_class(config).to(torch_device).eval()
            # Sets assisted generation arguments such that:
            # a) no EOS is generated, to ensure generation doesn't break early
            # b) the prompt lookup tries to give the model 2 tokens, to ensure the input preparation of
            #    prompt lookup is correct
            # c) there are at least two forward passes in the main model, to ensure the input preparation of
            #    the main model is correct
            generation_kwargs = {
                "eos_token_id": -1,  # see a)
                "max_new_tokens": 4,  # see c)
                "num_beams": 1,
                "do_sample": False,
                "output_scores": True,
                "output_logits": True,
                "output_hidden_states": True,
                "output_attentions": self.has_attentions,
                "return_dict_in_generate": True,
                "use_cache": True,
            }

            output_greedy = model.generate(**generation_kwargs, **inputs_dict)

            generation_kwargs.update({"prompt_lookup_num_tokens": 2})  # see b)
            output_prompt_lookup = model.generate(**generation_kwargs, **inputs_dict)

            # The two outputs must match and their shape must be as expected
            self._check_similar_generate_outputs(output_greedy, output_prompt_lookup)
            for output in (output_greedy, output_prompt_lookup):
                self._check_outputs(output, model.config, use_cache=True)

    @pytest.mark.generate
    def test_dola_decoding_sample(self):
        # TODO (joao): investigate skips, try to reduce incompatibilities
        for model_class in self.all_generative_model_classes:
            if model_class._is_stateful:
                self.skipTest(reason="Stateful models don't support DoLa decoding")

            if any(model_name in model_class.__name__.lower() for model_name in ["reformer"]):
                self.skipTest("Skip Reformer as the lm_head input size is 2 * hidden size, adopted from Rev Nets.")

            if any(model_name in model_class.__name__.lower() for model_name in ["marian", "mbart", "pegasus"]):
                self.skipTest("DoLa is not supported for models that don't return layerwise hidden states")

            if any(model_name == model_class.__name__ for model_name in ["LlavaNextVideoForConditionalGeneration"]):
                self.skipTest(f"DoLa is failing for {model_class.__name__}")

            # enable cache if the model is not openai-gpt, xlnet, cpm, or xlm
            config, inputs_dict = self.prepare_config_and_inputs_for_generate()

            # Encoder-decoder models are not supported
            if config.is_encoder_decoder:
                self.skipTest("DoLa is not supported for encoder-decoder models")
            config.is_decoder = True
            model = model_class(config).to(torch_device).eval()

            if model.get_output_embeddings() is None:
                self.skipTest("DoLa is not supported for models that don't have output embeddings")

            logits_processor_kwargs = self._get_logits_processor_kwargs(do_sample=True, config=model.config)

            # Sets dola generation arguments such that:
            # a) no EOS is generated, to ensure generation doesn't break early
            # b) there are at least two forward passes in the main model, to ensure the input preparation of
            #    the main model is correct
            generation_kwargs = {
                "eos_token_id": -1,  # see a)
                "max_new_tokens": 4,  # see b)
                "num_beams": 1,
                "do_sample": True,
                "output_scores": True,
                "output_logits": True,
                "output_hidden_states": True,
                "output_attentions": self.has_attentions,
                "return_dict_in_generate": True,
                "use_cache": getattr(config, "use_cache", False),  # Some models don't support the cache
                "dola_layers": "low",
            }
            output_dola = model.generate(**generation_kwargs, **logits_processor_kwargs, **inputs_dict)
            self._check_outputs(output_dola, model.config, use_cache=getattr(config, "use_cache", False))

    @pytest.mark.generate
    def test_assisted_decoding_sample(self):
        # In this test we don't check assisted vs non-assisted output -- seeded assisted decoding with sample will not
        # match sample for the same seed, as the forward pass does not return the exact same logits (due to matmul with
        # different shapes, see https://github.com/huggingface/transformers/issues/25420#issuecomment-1775317535).
        for model_class in self.all_generative_model_classes:
            if model_class._is_stateful:
                self.skipTest(reason="Stateful models don't support assisted generation")
            if any(model_name in model_class.__name__.lower() for model_name in ["fsmt", "reformer"]):
                self.skipTest(reason="Won't fix: old model with different cache format")
            if any(
                model_name in model_class.__name__.lower()
                for model_name in [
                    "bigbirdpegasus",
                    "led",
                    "mega",
                    "moshi",
                    "speech2text",
                    "git",
                    "prophetnet",
                    "seamlessm4t",
                    "clvp",
                ]
            ):
                self.skipTest(reason="May fix in the future: need model-specific fixes")

            # enable cache
            config, inputs_dict = self.prepare_config_and_inputs_for_generate(batch_size=1)

            # NOTE: assisted generation only works with cache on at the moment.
            if not hasattr(config, "use_cache"):
                self.skipTest(reason=f"{model_class.__name__} doesn't support caching")

            config.is_decoder = True
            model = model_class(config).to(torch_device).eval()
            # Sets assisted generation arguments such that:
            # a) no EOS is generated, to ensure generation doesn't break early
            # b) the assistant model always generates two tokens when it is called, to ensure the input preparation of
            #    the assistant model is correct
            # c) there are at least two forward passes in the main model, to ensure the input preparation of
            #    the main model is correct
            assistant_model = model
            assistant_model.generation_config.num_assistant_tokens = 2  # see b)
            assistant_model.generation_config.num_assistant_tokens_schedule = "constant"  # see b)
            generation_kwargs = {
                "eos_token_id": -1,  # see a)
                "max_new_tokens": 4,  # see c)
                "num_beams": 1,
                "do_sample": True,
                "assistant_model": assistant_model,
                "output_scores": True,
                "output_logits": True,
                "output_hidden_states": True,
                "output_attentions": self.has_attentions,
                "return_dict_in_generate": True,
                "use_cache": True,
            }
            output_assisted = model.generate(**generation_kwargs, **inputs_dict)

            self._check_outputs(output_assisted, config, use_cache=True)

    @pytest.mark.generate
    def test_prompt_lookup_decoding_stops_at_eos(self):
        # This test ensures that the prompt lookup generation stops at eos token and does not suggest more tokens
        # (see https://github.com/huggingface/transformers/pull/31301)

        # The main idea is to have an ngram (unigram in our case) that is repeated twice in the input ids.
        # First time at the very end, so input ends with the unigrams, and second any arbitrary location.
        # Also, we need an EOS token which will be injected just after the arbitrary located ngram.
        # We verify that PLD will not copy and propose candidated that contain an EOS token, even if there are overlapping ngrams
        # in input ids. Otherwise a proposed EOS along with the trailing (ngrams-1) tokens might be accepted by the target model.
        # That seems as if the model "generated" and EOS but didn't stop from user's perspective

        input_ids = torch.randint(1, 50, (1, 10), device=torch_device)  # generate inputs in range from 1-50
        arbitrary_ngram = 51  # this is the arbitrary ngram, specifically chosen OOV to prevent flaky tests
        input_ids[:, 3] = arbitrary_ngram  # set pre-eos to arbitrary_ngram which is for sure not present in inputs
        input_ids[:, -1] = arbitrary_ngram  # put arbitrary_ngram in the end for the necessary match to happen

        eos_token_id = torch.tensor([0], device=torch_device)
        input_ids[:, 4] = eos_token_id  # inject eos-token-id in input ids so that it is located after arbitrary_ngram

        # init cand geenerator with max_matching_ngram_size=1 to match per-token
        candidate_generator = PromptLookupCandidateGenerator(
            eos_token_id=eos_token_id, num_output_tokens=4, max_matching_ngram_size=1
        )
        output_prompt_lookup = candidate_generator.get_candidates(input_ids)[0]

        # PLD shouldn't propose any new tokens based on eos-match
        self.assertTrue(output_prompt_lookup.shape[-1] == 10)

    @pytest.mark.generate
    def test_generate_with_head_masking(self):
        """Test designed for encoder-decoder models to ensure the attention head masking is used."""
        attention_names = ["encoder_attentions", "decoder_attentions", "cross_attentions"]
        for model_class in self.all_generative_model_classes:
            config, inputs_dict = self.prepare_config_and_inputs_for_generate()
            text_config = config.get_text_config()

            # We want to test only encoder-decoder models
            if not text_config.is_encoder_decoder:
                continue
            model = model_class(config).to(torch_device)

            head_masking = {
                "head_mask": torch.zeros(
                    text_config.encoder_layers, text_config.encoder_attention_heads, device=torch_device
                ),
                "decoder_head_mask": torch.zeros(
                    text_config.decoder_layers, text_config.decoder_attention_heads, device=torch_device
                ),
                "cross_attn_head_mask": torch.zeros(
                    text_config.decoder_layers, text_config.decoder_attention_heads, device=torch_device
                ),
            }

            signature = inspect.signature(model.forward)
            # We want to test only models where encoder/decoder head masking is implemented
            if not set(head_masking.keys()) < {*signature.parameters.keys()}:
                continue

            for attn_name, (name, mask) in zip(attention_names, head_masking.items()):
                out = model.generate(
                    num_beams=1,
                    output_attentions=self.has_attentions,
                    return_dict_in_generate=True,
                    remove_invalid_values=True,
                    **{name: mask},
                    **inputs_dict,
                )
                # We check the state of decoder_attentions and cross_attentions just from the last step
                attn_weights = out[attn_name] if attn_name == attention_names[0] else out[attn_name][-1]
                self.assertEqual(sum([w.sum().item() for w in attn_weights]), 0.0)

    @pytest.mark.generate
    def test_left_padding_compatibility(self):
        # NOTE: left-padding results in small numerical differences. This is expected.
        # See https://github.com/huggingface/transformers/issues/25420#issuecomment-1775317535

        # First, filter out models that don't support left padding
        # - The model must have generative capabilities
        if len(self.all_generative_model_classes) == 0:
            self.skipTest(reason="No generative architecture available for this model.")

        # - The model must support padding
        if not self.has_attentions:
            self.skipTest(reason="This model doesn't support padding.")

        # - The model must be a decoder-only architecture (encoder-based architectures use right-padding)
        decoder_only_classes = []
        for model_class in self.all_generative_model_classes:
            config, _ = self.prepare_config_and_inputs_for_generate()
            if config.is_encoder_decoder:
                continue
            else:
                decoder_only_classes.append(model_class)
        if len(decoder_only_classes) == 0:
            self.skipTest(reason="No decoder-only architecture available for this model.")

        # - Decoder-only architectures derived from encoder-decoder models could support it in theory, but we haven't
        #   added support for it yet. We skip these models for now.
        has_encoder_attributes = any(
            attr_name
            for attr_name in config.to_dict().keys()
            if attr_name.startswith("encoder") and attr_name != "encoder_no_repeat_ngram_size"
        )
        if has_encoder_attributes:
            self.skipTest(
                reason="The decoder-only derived from encoder-decoder models are not expected to support left-padding."
            )

        # Then, test left-padding
        def _prepare_model_kwargs(input_ids, attention_mask, signature):
            model_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if "position_ids" in signature:
                position_ids = torch.cumsum(attention_mask, dim=-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                model_kwargs["position_ids"] = position_ids
            if "cache_position" in signature:
                cache_position = torch.arange(input_ids.shape[-1], device=torch_device)
                model_kwargs["cache_position"] = cache_position
            return model_kwargs

        for model_class in decoder_only_classes:
            config, inputs_dict = self.prepare_config_and_inputs_for_generate()
            input_ids = inputs_dict["input_ids"]
            attention_mask = inputs_dict.get("attention_mask")
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)

            model = model_class(config).to(torch_device).eval()
            signature = inspect.signature(model.forward).parameters.keys()

            # no cache as some models require special cache classes to be init outside forward
            model.generation_config.use_cache = False

            # Without padding
            model_kwargs = _prepare_model_kwargs(input_ids, attention_mask, signature)
            next_logits_wo_padding = model(**model_kwargs).logits[:, -1, :]

            # With left-padding (length 32)
            # can hardcode pad_token to be 0 as we'll do attn masking anyway
            pad_token_id = (
                config.get_text_config().pad_token_id if config.get_text_config().pad_token_id is not None else 0
            )
            pad_size = (input_ids.shape[0], 32)
            padding = torch.ones(pad_size, dtype=input_ids.dtype, device=torch_device) * pad_token_id
            padded_input_ids = torch.cat((padding, input_ids), dim=1)
            padded_attention_mask = torch.cat((torch.zeros_like(padding), attention_mask), dim=1)
            model_kwargs = _prepare_model_kwargs(padded_input_ids, padded_attention_mask, signature)
            next_logits_with_padding = model(**model_kwargs).logits[:, -1, :]

            # They should result in very similar logits
            torch.testing.assert_close(next_logits_wo_padding, next_logits_with_padding, atol=1e-5, rtol=1e-5)

    @pytest.mark.generate
    def test_past_key_values_format(self):
        # Test that the KV cache is formatted correctly. Exceptions need to explicitly overwrite this test. Having a
        # standard KV cache format is important for a consistent API (and for advanced generation methods).
        for model_class in self.all_generative_model_classes:
            config, inputs = self.model_tester.prepare_config_and_inputs_for_common()

            # If it doesn't support cache, pass the test
            if not hasattr(config, "use_cache"):
                self.skipTest(reason=f"{model_class.__name__} doesn't support caching")

            model = model_class(config).to(torch_device)
            if "use_cache" not in inputs:
                inputs["use_cache"] = True
            outputs = model(**inputs)

            # If "past_key_values" is not returned, pass the test (e.g. RWKV uses a different cache name and format)
            if "past_key_values" not in outputs:
                self.skipTest(reason="This model doesn't return `past_key_values`")

            text_config = config.get_text_config()
            num_hidden_layers = (
                getattr(text_config, "decoder_layers", None)
                or getattr(text_config, "num_decoder_layers", None)
                or text_config.num_hidden_layers
            )
            num_attention_heads = getattr(text_config, "decoder_attention_heads", text_config.num_attention_heads)
            embed_dim = getattr(text_config, "d_model", text_config.hidden_size)
            per_head_embed_dim = embed_dim // num_attention_heads

            # some models have diffent num-head for query vs key/value so we need to assign correct value
            # BUT only after `per_head_embed_dim` is set
            num_attention_heads = (
                text_config.num_key_value_heads
                if getattr(text_config, "num_key_value_heads", None) is not None
                else num_attention_heads
            )

            past_kv = outputs["past_key_values"]
            self.assertEqual(len(past_kv), num_hidden_layers)

            # Encoder-Decoder checks
            if config.is_encoder_decoder:
                encoder_num_attention_heads = config.encoder_attention_heads
                encoder_per_head_embed_dim = embed_dim // encoder_num_attention_heads
                batch_size, seq_length = inputs["decoder_input_ids"].shape
                for i in range(num_hidden_layers):
                    self.assertEqual(len(past_kv[i]), 4)  # K V for the decoder + K V for the encoder = 4
                    self.assertEqual(
                        past_kv[i][0].shape, (batch_size, num_attention_heads, seq_length, per_head_embed_dim)
                    )
                    self.assertEqual(
                        past_kv[i][1].shape, (batch_size, num_attention_heads, seq_length, per_head_embed_dim)
                    )
                    # The sequence length for the encoder K V depends on the model. Since it is not manipulated in
                    # autoregressive generation, I'm keeping the test general and not checking the 3rd dim
                    self.assertEqual(
                        (past_kv[i][2].shape[0], past_kv[i][2].shape[1], past_kv[i][2].shape[3]),
                        (batch_size, encoder_num_attention_heads, encoder_per_head_embed_dim),
                    )
                    self.assertEqual(
                        (past_kv[i][3].shape[0], past_kv[i][3].shape[1], past_kv[i][3].shape[3]),
                        (batch_size, encoder_num_attention_heads, encoder_per_head_embed_dim),
                    )

            # Decoder-only checks
            else:
                # TODO: this line is only needed because of imagegpt, where "pixel_values" = "input_ids". Fix the
                # tests in imagegpt such that `prepare_config_and_inputs_for_common` returns the later (and the other
                # tests use it)
                key = "input_ids" if "input_ids" in inputs else "pixel_values"
                batch_size, seq_length = inputs[key].shape
                for i in range(num_hidden_layers):
                    self.assertEqual(len(past_kv[0]), 2)  # K V for the decoder = 2
                    self.assertEqual(
                        past_kv[i][0].shape, (batch_size, num_attention_heads, seq_length, per_head_embed_dim)
                    )
                    self.assertEqual(
                        past_kv[i][1].shape, (batch_size, num_attention_heads, seq_length, per_head_embed_dim)
                    )

    @pytest.mark.generate
    @parameterized.expand([("greedy", 1), ("beam search", 2)])
    def test_generate_from_inputs_embeds(self, _, num_beams):
        """Tests that we can generate from `inputs_embeds` instead of `input_ids` in LLMs, VLMs, etc"""
        # When supported, tests that the decoder model can generate from `inputs_embeds` instead of `input_ids`
        # if fails, you should probably update the `prepare_inputs_for_generation` function
        for model_class in self.all_generative_model_classes:
            config, inputs_dict = self.prepare_config_and_inputs_for_generate()

            # This test is for decoder-only models (encoder-decoder models have native input embeddings support in the
            # decoder)
            if config.is_encoder_decoder:
                continue
            config.is_decoder = True

            # Skip models without explicit support
            model = model_class(config).to(torch_device).eval()
            if "inputs_embeds" not in inspect.signature(model.prepare_inputs_for_generation).parameters.keys():
                continue

            # There are a few exception patterns in this test:
            # 1 - Some models can't generate without `input_ids`, when `inputs_embeds` are passed
            requires_inputs_ids = any(model_name in model_class.__name__.lower() for model_name in ["idefics"])
            # 2 - Complex `inputs_embeds` computation, i.e. the correct computation of inputs embeds is more complex
            # than calling the embedding layer with `input_ids`. Subcases of this exception:
            #   2.A - Ignore `scale_embedding`, if the model supports it (it is controlled by a model-dependent flag)
            if hasattr(config, "scale_embedding"):
                config.scale_embedding = False
            #   2.B - Some VLMs assume `inputs_embeds` and `pixel_values` are mutually exclusive AND fall in the
            #   exception above (complex `inputs_embeds` computation). Popping `pixel_values` allow us to run the
            #   checks without adding test complexity. Ditto for `pixel_values_videos` and `pixel_values_images`
            pixel_values_is_mutually_exclusive = any(
                model_name in model_class.__name__.lower()
                for model_name in ["llava", "idefics2", "idefics3", "mllama", "paligemma"]
            )
            if pixel_values_is_mutually_exclusive:
                inputs_dict.pop("pixel_values", None)
                inputs_dict.pop("pixel_values_videos", None)
                inputs_dict.pop("pixel_values_images", None)
            #   2.C - No easy fix, let's skip the check that compares the outputs from `input_ids` and `inputs_embeds`
            has_complex_embeds_computation = any(
                model_name in model_class.__name__.lower() for model_name in ["moshi", "qwen2vl"]
            )
            # 3 - `inputs_dict` doesn't contain `attention_mask`. When `attention_mask` is not passed to generate,
            # we infer it from `input_ids`. The last test case will fail if there is a pad token in the original input.
            missing_attention_mask = "attention_mask" not in inputs_dict

            # Traditional way of generating text
            input_ids = inputs_dict.pop("input_ids")
            generation_kwargs = {
                "return_dict_in_generate": True,
                "output_scores": True,
                "num_beams": num_beams,
                "do_sample": False,
                "max_new_tokens": 5,
                "min_new_tokens": 5,  # generate exactly 5 tokens
            }
            outputs_from_ids = model.generate(input_ids, **generation_kwargs, **inputs_dict)
            self.assertEqual(outputs_from_ids.sequences.shape, (input_ids.shape[0], input_ids.shape[1] + 5))

            # Same thing, but from input embeddings (`input_ids` is passed so the prompt is present in the output).
            # The output of the two calls should be the same.
            inputs_embeds = model.get_input_embeddings()(input_ids)
            outputs_from_embeds = model.generate(
                input_ids, inputs_embeds=inputs_embeds, **generation_kwargs, **inputs_dict
            )
            if not has_complex_embeds_computation:
                self._check_similar_generate_outputs(outputs_from_ids, outputs_from_embeds)

            # If we pass different inputs_embeds, we should get different outputs (the output text may be the
            # same, but the logits will almost surely be different)
            random_embeds = torch.rand_like(inputs_embeds)
            outputs_from_rand_embeds = model.generate(
                input_ids, inputs_embeds=random_embeds, **generation_kwargs, **inputs_dict
            )
            for i in range(len(outputs_from_rand_embeds.scores)):
                self.assertFalse(torch.allclose(outputs_from_embeds.scores[i], outputs_from_rand_embeds.scores[i]))

            # input_ids is not a required input on most models -- if we don't pass it, the newly generated tokens will
            # be the same
            if not (requires_inputs_ids or missing_attention_mask):
                outputs_from_embeds_wo_ids = model.generate(
                    inputs_embeds=inputs_embeds, **generation_kwargs, **inputs_dict
                )
                outputs_from_embeds.sequences = outputs_from_embeds.sequences[:, inputs_embeds.shape[1] :]
                self._check_similar_generate_outputs(outputs_from_embeds_wo_ids, outputs_from_embeds)

    @pytest.mark.generate
    def test_generate_from_inputs_embeds_with_static_cache(self):
        """
        Test that StaticCache can generate from inputs_embeds and calculates max_cache_length
        correctly in `generate()`. We force the model to not stop generation until max-length is reached
        to verify that the cache length is indeed set correctly and we don't run out of index when slicing the cache.
        """
        for model_class in self.all_generative_model_classes:
            if not model_class._supports_static_cache:
                self.skipTest(reason="This model does not support the static cache format")

            config, inputs_dict = self.prepare_config_and_inputs_for_generate()

            if config.is_encoder_decoder:
                self.skipTest(reason="This model is encoder-decoder and has Encoder-Decoder Cache")

            model = model_class(config).to(torch_device).eval()
            if "inputs_embeds" not in inspect.signature(model.prepare_inputs_for_generation).parameters.keys():
                self.skipTest(reason="This model does not support `inputs_embeds` in generation")

            input_ids = inputs_dict.pop("input_ids")

            model.config.use_cache = True
            model.config.is_decoder = True
            batch_size = input_ids.shape[0]
            max_cache_len = 30

            # here we force to not stop at eos and go until max-length
            model.generation_config.eos_token_id = model.config.get_text_config().eos_token_id = -1
            generation_kwargs = {
                "max_length": max_cache_len,
                "cache_implementation": "static",
                "return_dict_in_generate": True,  # Required to return `past_key_values`
            }

            text_config = model.config.get_text_config()
            head_dim = (
                text_config.head_dim
                if hasattr(text_config, "head_dim")
                else text_config.hidden_size // text_config.num_attention_heads
            )
            num_key_value_heads = (
                text_config.num_attention_heads
                if getattr(text_config, "num_key_value_heads", None) is None
                else text_config.num_key_value_heads
            )
            num_hidden_layers = text_config.num_hidden_layers

            inputs_embeds = model.get_input_embeddings()(input_ids)
            max_cache_len += inputs_embeds.shape[1]
            outputs = model.generate(inputs_embeds=inputs_embeds, **generation_kwargs, **inputs_dict)

            # we should get `max_length` in shape, not `max_length - embeds_length`
            cache_shape = (batch_size, num_key_value_heads, max_cache_len, head_dim)
            self.assertTrue(isinstance(outputs.past_key_values, StaticCache))
            self.assertTrue(len(outputs.past_key_values.key_cache) == num_hidden_layers)
            self.assertTrue(outputs.past_key_values.key_cache[0].shape == cache_shape)

    @pytest.mark.generate
    def test_generate_continue_from_past_key_values(self):
        # Tests that we can continue generating from past key values, returned from a previous `generate` call
        for model_class in self.all_generative_model_classes:
            if any(model_name in model_class.__name__.lower() for model_name in ["imagegpt"]):
                self.skipTest(reason="Won't fix: old model with unique inputs/caches/other")
            if any(model_name in model_class.__name__.lower() for model_name in ["umt5"]):
                self.skipTest(reason="TODO: needs modeling or test input preparation fixes for compatibility")

            config, inputs = self.model_tester.prepare_config_and_inputs_for_common()

            if not hasattr(config, "use_cache"):
                self.skipTest(reason=f"{model_class.__name__} doesn't support caching")

            # Let's make it always:
            # 1. use cache (for obvious reasons)
            # 2. generate to max length (which can be achieved by setting the eos token to an invalid value), which
            #    would make the test flaky (e.g. EOS is generated on iteration 1 on both generations, but the
            #    continuation would force it to generate beyond an EOS token)
            # 3. ignore `token_type_ids` for simplicity
            # 4. ignore `forced_eos_token_id`, which requires further manipulation of the continuation inputs and is
            #    active by default on some models
            # 5. ignore `encoder_no_repeat_ngram_size`, which is set by default in some encoder-decoder models. When
            #    we use their decoder as a stand-alone model, `encoder_no_repeat_ngram_size` actually prevents
            #    repetition exclusively from the prompt. This test relies on comparing one call vs 2 calls
            #    with cache, what is considered a prompt is different in the two cases.

            if "token_type_ids" in inputs:
                del inputs["token_type_ids"]

            model = model_class(config).to(torch_device)
            model.eval()
            model.generation_config.pad_token_id = model.generation_config.eos_token_id = -1
            model.generation_config.forced_eos_token_id = None
            model.generation_config.encoder_no_repeat_ngram_size = 0
            model.generation_config.use_cache = True

            # If "past_key_values" is not returned, skip the test (e.g. RWKV uses a different cache name and format)
            outputs = model(**inputs)
            if "past_key_values" not in outputs:
                self.skipTest(reason="This model doesn't return `past_key_values`")

            # Traditional way of generating text, with `return_dict_in_generate` to return the past key values
            outputs = model.generate(**inputs, do_sample=False, max_new_tokens=4, return_dict_in_generate=True)

            # Let's generate again, but passing the past key values in between (3 + 1 = 4 tokens). Note that the
            # inputs may need to be tweaked across `generate` calls (like the attention mask).
            outputs_cached = model.generate(**inputs, do_sample=False, max_new_tokens=3, return_dict_in_generate=True)

            # Continue from the tokens generated above, preparing the inputs accordingly
            inputs["past_key_values"] = outputs_cached.past_key_values
            new_attention_len = outputs_cached.sequences.shape[-1]
            if config.is_encoder_decoder:
                inputs["decoder_input_ids"] = outputs_cached.sequences
                if "decoder_attention_mask" in inputs:
                    inputs["decoder_attention_mask"] = torch.nn.functional.pad(
                        inputs["decoder_attention_mask"],
                        (0, new_attention_len - inputs["decoder_attention_mask"].shape[1]),
                        mode="constant",
                        value=1,
                    )
            else:
                inputs["input_ids"] = outputs_cached.sequences
                if "attention_mask" in inputs:
                    inputs["attention_mask"] = torch.nn.functional.pad(
                        inputs["attention_mask"],
                        (0, new_attention_len - inputs["attention_mask"].shape[1]),
                        mode="constant",
                        value=1,
                    )
            outputs_cached = model.generate(**inputs, do_sample=False, max_new_tokens=1, return_dict_in_generate=True)

            # The two sets of generated text and past kv should be equal to each other
            self.assertListEqual(outputs.sequences.tolist(), outputs_cached.sequences.tolist())
            for layer_idx in range(len(outputs_cached.past_key_values)):
                for kv_idx in range(len(outputs_cached.past_key_values[layer_idx])):
                    self.assertTrue(
                        torch.allclose(
                            outputs.past_key_values[layer_idx][kv_idx],
                            outputs_cached.past_key_values[layer_idx][kv_idx],
                        )
                    )

    @parameterized.expand([(1, False), (1, True), (4, False)])
    @pytest.mark.generate
    def test_new_cache_format(self, num_beams, do_sample):
        # Tests that generating with the new format is exactly the same as the legacy one (for models that support it).
        # 👉 tests with and without beam search so that we can test with and without cache reordering.
        # 👉 tests with and without sampling so we can cover the most common use cases.
        for model_class in self.all_generative_model_classes:
            if not model_class._supports_cache_class:
                self.skipTest(reason="This model does not support the new cache format")

            config, inputs_dict = self.prepare_config_and_inputs_for_generate()

            model = model_class(config).to(torch_device).eval()
            generation_kwargs = {
                "max_new_tokens": 5,
                "do_sample": do_sample,
                "num_beams": num_beams,
                "num_return_sequences": num_beams,
                "return_dict_in_generate": True,  # Required to return `past_key_values`
                "use_cache": True,
            }

            # Sets seed before calling `generate` for the case with do_sample=True
            seed = torch.randint(0, 1000000, (1,)).item()
            set_seed(seed)
            legacy_results = model.generate(**generation_kwargs, **inputs_dict)
            set_seed(seed)
            if config.is_encoder_decoder:
                cache_cls = EncoderDecoderCache
                past_key_values = cache_cls(DynamicCache(), DynamicCache())
            else:
                cache_cls = DynamicCache
                past_key_values = cache_cls()

            new_results = model.generate(past_key_values=past_key_values, **generation_kwargs, **inputs_dict)

            # The two sets of generated sequences must match, despite the cache format between forward passes being
            # different
            self.assertListEqual(legacy_results.sequences.tolist(), new_results.sequences.tolist())
            self.assertTrue(isinstance(legacy_results.past_key_values, tuple))
            self.assertTrue(isinstance(new_results.past_key_values, cache_cls))

            # The contents of the two caches, when converted to the same format (in both directions!), must match
            legacy_cache = legacy_results.past_key_values
            new_cache_converted = new_results.past_key_values.to_legacy_cache()
            for layer_idx in range(len(legacy_cache)):
                for kv_idx in range(len(legacy_cache[layer_idx])):
                    # TODO: @raushan, please look into this for new cache format
                    if legacy_cache[layer_idx][kv_idx] != []:
                        self.assertTrue(
                            torch.allclose(
                                legacy_cache[layer_idx][kv_idx],
                                new_cache_converted[layer_idx][kv_idx],
                            )
                        )

            new_cache = new_results.past_key_values
            legacy_cache_converted = cache_cls.from_legacy_cache(legacy_results.past_key_values)
            for layer_idx in range(len(new_cache)):
                for kv_idx in range(len(new_cache[layer_idx])):
                    # TODO: @raushan, please look into this for new cache format
                    if new_cache[layer_idx][kv_idx] != []:
                        self.assertTrue(
                            torch.allclose(
                                new_cache[layer_idx][kv_idx],
                                legacy_cache_converted[layer_idx][kv_idx],
                            )
                        )

    @parameterized.expand([("offloaded",)])  # ("offloaded_static",) TODO: @raushan fixme in some models (eg T5)
    @require_torch_gpu
    @pytest.mark.generate
    def test_offloaded_cache_implementation(self, cache_implementation):
        """Tests we can generate by indicating `cache_implementation` for each possible cache class"""
        for model_class in self.all_generative_model_classes:
            if not model_class._supports_cache_class:
                self.skipTest(reason="This model does not support the new cache format")

            config, inputs_dict = self.prepare_config_and_inputs_for_generate()

            model = model_class(config).to(torch_device).eval()
            generation_kwargs = {
                "max_new_tokens": 5,
                "use_cache": True,
                "cache_implementation": cache_implementation,
            }

            legacy_results = model.generate(**generation_kwargs, **inputs_dict)

            # Most cache classes have their own tests except for some that are tested here
            # The ones here do not need special treatment when passing `cache_implementation`
            # and are not bound to specific models only
            new_results = model.generate(**generation_kwargs, **inputs_dict)
            self.assertListEqual(legacy_results.tolist(), new_results.tolist())

    @pytest.mark.generate
    def test_generate_with_static_cache(self):
        """
        Tests that generating with static cache give almost same results as with dynamic cache, and the output cache
        has the expected shapes
        """
        set_model_tester_for_less_flaky_test(self)
        for model_class in self.all_generative_model_classes:
            if not model_class._supports_static_cache:
                self.skipTest(reason="This model does not support the static cache format")

            config, inputs_dict = self.prepare_config_and_inputs_for_generate()
            set_config_for_less_flaky_test(config)
            main_input = inputs_dict[model_class.main_input_name]

            if config.is_encoder_decoder:
                self.skipTest(reason="This model is encoder-decoder and has Encoder-Decoder Cache")

            config.is_decoder = True
            batch_size = main_input.shape[0]
            seq_length = main_input.shape[-1]
            max_new_tokens = 20

            for dtype in (torch.float32, torch.float16):
                model = model_class(config).to(torch_device).to(dtype).eval()
                set_model_for_less_flaky_test(model)

                generation_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    "return_dict_in_generate": True,  # Required to return `past_key_values`
                    "output_scores": True,
                    "use_cache": True,
                }

                static_cache_generation = model.generate(
                    **generation_kwargs, **inputs_dict, cache_implementation="static"
                )

                # Check 1: The cache shapes must match the expected shapes
                max_cache_len = seq_length + max_new_tokens
                text_config = config.text_config if hasattr(config, "text_config") else config
                head_dim = (
                    text_config.head_dim
                    if hasattr(text_config, "head_dim")
                    else text_config.hidden_size // text_config.num_attention_heads
                )
                num_key_value_heads = (
                    text_config.num_attention_heads
                    if getattr(text_config, "num_key_value_heads", None) is None
                    else text_config.num_key_value_heads
                )
                num_hidden_layers = text_config.num_hidden_layers
                cache_shape = (batch_size, num_key_value_heads, max_cache_len, head_dim)
                self.assertTrue(isinstance(static_cache_generation.past_key_values, StaticCache))
                self.assertTrue(len(static_cache_generation.past_key_values.key_cache) == num_hidden_layers)
                self.assertTrue(static_cache_generation.past_key_values.key_cache[0].shape == cache_shape)

                # Check 2: The outputs must be similar to the case with dynamic cache
                dynamic_cache_generation = model.generate(**generation_kwargs, **inputs_dict)
                self._check_similar_generate_outputs(dynamic_cache_generation, static_cache_generation)

    @require_optimum_quanto
    @pytest.mark.generate
    def test_generate_with_quant_cache(self):
        for model_class in self.all_generative_model_classes:
            if not model_class._supports_quantized_cache:
                self.skipTest(reason="This model does not support the quantized cache format")

            config, inputs_dict = self.prepare_config_and_inputs_for_generate()
            config.is_decoder = True

            model = model_class(config).to(torch_device).eval()
            generation_kwargs = {
                "max_new_tokens": 5,
                "cache_implementation": "quantized",
                # careful with group size, should be divisor of model's hidden size
                "cache_config": {"backend": "quanto", "nbits": 2, "q_group_size": 8, "residual_length": 128},
                "return_dict_in_generate": True,  # Required to return `past_key_values`
                "use_cache": True,
            }

            results = model.generate(**generation_kwargs, **inputs_dict)
            self.assertTrue(isinstance(results.past_key_values, QuantoQuantizedCache))

            # passing past key values of different type should raise Error
            with self.assertRaises(ValueError):
                model.generate(past_key_valyes=DynamicCache(), **generation_kwargs, **inputs_dict)

            # setting incorrect cache_config args should raise an Error, i.e. nbits=60 does not make sense
            generation_kwargs["cache_config"] = {"nbits": 60, "q_group_size": 8, "residual_length": 128}
            with self.assertRaises(ValueError):
                model.generate(**generation_kwargs, **inputs_dict)

    @parameterized.expand(
        [
            ("forward_only", False),  # TODO (@joao): a few models failing. After fixed, this should not be "@slow"
            ("end_to_end", True),  # TODO (@joao): end-to-end compilation is broken with torch 2.5+, explore and fix
        ]
    )
    @pytest.mark.generate
    @require_torch_gpu
    @slow
    def test_generate_compile(self, _, end_to_end):
        """
        Tests that `.generate` is compatible with torch.compile without graph breaks, keeping the same results. Tests
        end-to-end compilation and forward pass compilation only.
        ⚠️ Runs two sequential generations to ensure the cache doesn't get stuck after the first compiled run! ⚠️
        """
        for model_class in self.all_generative_model_classes:
            if not model_class._supports_static_cache:
                self.skipTest("This model doesn't support static cache")

            # TODO (joao) -- fix and enable me :)
            if end_to_end and any(model_name in model_class.__name__.lower() for model_name in ["whisper"]):
                self.skipTest("whisper model end-to-end generate compile not yet supported")

            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            # TODO (joao) -- fix and enable me :)
            if end_to_end and config.is_encoder_decoder:
                self.skipTest("Encoder-decoder model end-to-end generate compile not yet supported")

            model = model_class(config).to(torch_device)
            model.eval()  # otherwise `self.training` is `True` -- this flag is used at attn mask creation time

            input_ids = inputs_dict["input_ids"].to(torch_device)
            # creates two sets of *different* inputs with the same shape
            half_batch_size = input_ids.shape[0] // 2
            input_ids_sets = [input_ids[:half_batch_size, :], input_ids[half_batch_size : half_batch_size * 2, :]]
            self.assertTrue(input_ids_sets[0].shape == input_ids_sets[1].shape)

            generation_kwargs = {
                "do_sample": False,
                "max_new_tokens": 10,
                "return_dict_in_generate": True,
                "output_scores": True,
            }
            # end-to-end works best with dynamic cache, forward compilation works best with static cache
            if not end_to_end:
                generation_kwargs["cache_implementation"] = "static"

            # get eager + dynamic cache results for future comparison
            dynamic_outputs = []
            for model_inputs in input_ids_sets:
                dynamic_outputs.append(model.generate(model_inputs, **generation_kwargs))

            # get compiled results
            generation_config = copy.deepcopy(model.generation_config)
            generation_config.update(**generation_kwargs)
            torch.compiler.reset()
            if end_to_end:
                model.generate = torch.compile(model.generate, fullgraph=True, mode="reduce-overhead")
            else:
                model.forward = torch.compile(model.forward, fullgraph=True, mode="reduce-overhead")

            compiled_outputs = []
            for model_inputs in input_ids_sets:
                compiled_outputs.append(model.generate(model_inputs, generation_config=generation_config))

            for dynamic_result, compiled_result in zip(dynamic_outputs, compiled_outputs):
                self._check_similar_generate_outputs(dynamic_result, compiled_result)

    @pytest.mark.generate
    def test_generate_methods_with_num_logits_to_keep(self):
        for model_class in self.all_generative_model_classes:
            if "num_logits_to_keep" not in set(inspect.signature(model_class.forward).parameters.keys()):
                self.skipTest(reason="This model does not support `num_logits_to_keep` argument.")

            config, inputs_dict = self.prepare_config_and_inputs_for_generate()
            config.use_cache = True
            config.is_decoder = True

            model = model_class(config).to(torch_device).eval()
            # All generation methods (except assisted decoding) rely on always extracting the last token logits of the
            # full logits matrix, so testing out only greedy search and assisted decoding is enough (if it works,
            # other methods will work as well)
            generation_kwargs = {
                "max_new_tokens": 10,
                "do_sample": False,
            }

            # Setting num_logits_to_keep at 0 keeps all logits (old behavior)
            with_all_logits = model.generate(**generation_kwargs, **inputs_dict, num_logits_to_keep=0)
            # By default, num_logits_to_keep is automatically set to 1 if not provided (new behavior)
            without_all_logits = model.generate(**inputs_dict, **generation_kwargs)
            self.assertEqual(with_all_logits.tolist(), without_all_logits.tolist())

    @pytest.mark.generate
    def test_assisted_decoding_with_num_logits_to_keep(self):
        for model_class in self.all_generative_model_classes:
            if "num_logits_to_keep" not in set(inspect.signature(model_class.forward).parameters.keys()):
                self.skipTest(reason="This model does not support `num_logits_to_keep` argument.")
            if model_class._is_stateful:
                self.skipTest(reason="Stateful models don't support assisted generation")

            config, inputs_dict = self.prepare_config_and_inputs_for_generate(batch_size=1)
            # NOTE: assisted generation only works with cache on at the moment.
            if not hasattr(config, "use_cache"):
                self.skipTest(reason=f"{model_class.__name__} doesn't support caching")
            config.use_cache = True
            config.is_decoder = True

            model = model_class(config).to(torch_device).eval()
            assistant_model = model
            # All generation methods (except assisted decoding) rely on always extracting the last token logits of the
            # full logits matrix, so testing out only greedy search and assisted decoding is enough (if it works,
            # other methods will work as well)
            generation_kwargs = {
                "max_new_tokens": 10,
                "do_sample": False,
                "assistant_model": assistant_model,
                "return_dict_in_generate": True,
                "output_scores": True,
            }

            # Setting num_logits_to_keep at 0 keeps all logits (old behavior)
            with_all_logits = model.generate(**generation_kwargs, **inputs_dict, num_logits_to_keep=0)
            # By default, num_logits_to_keep is automatically set to 1 if not provided (new behavior)
            without_all_logits = model.generate(**inputs_dict, **generation_kwargs)

            self._check_similar_generate_outputs(with_all_logits, without_all_logits)

    @pytest.mark.generate
    def test_inherits_generation_mixin(self):
        """
        Tests that the model class directly inherits `GenerationMixin`, as opposed to relying on `PreTrainedModel`
        to inherit it.
        """
        for model_class in self.all_generative_model_classes:
            self.assertTrue("GenerationMixin" in str(model_class.__bases__))

    def _test_attention_implementation(self, attn_implementation):
        """
        Compares the output of generate with the eager attention implementation against other implementations.
        NOTE: despite the test logic being the same, different implementations actually need diferent decorators, hence
        this separate function.
        """
        max_new_tokens = 30
        support_flag = {
            "sdpa": "_supports_sdpa",
            "flash_attention_2": "_supports_flash_attn_2",
        }

        for model_class in self.all_generative_model_classes:
            if not getattr(model_class, support_flag[attn_implementation]):
                self.skipTest(f"{model_class.__name__} does not support `attn_implementation={attn_implementation}`")

            config, original_inputs_dict = self.prepare_config_and_inputs_for_generate()
            inputs_dict = {}
            for input_name, input_data in original_inputs_dict.items():
                if isinstance(input_data, torch.Tensor) and input_data.dtype in [torch.float32, torch.bfloat16]:
                    inputs_dict[input_name] = input_data.to(torch.float16)
                else:
                    inputs_dict[input_name] = input_data
            main_input = inputs_dict[model_class.main_input_name]

            # make sure that all models have enough positions for generation
            if hasattr(config, "max_position_embeddings"):
                config.max_position_embeddings = max_new_tokens + main_input.shape[1] + 1

            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                del model
                gc.collect()

                generate_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": False,
                    "return_dict_in_generate": True,
                    "output_scores": True,
                    "use_cache": True,
                }

                model_eager = model_class.from_pretrained(
                    tmpdirname,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    attn_implementation="eager",
                ).to(torch_device)
                res_eager = model_eager.generate(**inputs_dict, **generate_kwargs)
                del model_eager
                gc.collect()

                model_attn = model_class.from_pretrained(
                    tmpdirname,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    attn_implementation=attn_implementation,
                ).to(torch_device)
                res_attn = model_attn.generate(**inputs_dict, **generate_kwargs)
                del model_attn
                gc.collect()

                self._check_similar_generate_outputs(res_eager, res_attn, atol=1e-3, rtol=1e-3)

    @pytest.mark.generate
    @require_torch_sdpa
    @slow
    def test_eager_matches_sdpa_generate(self):
        """Tests that generate has equivalent outputs with SDPA and eager attention implementations."""
        self._test_attention_implementation("sdpa")

    @pytest.mark.flash_attn_test
    @require_flash_attn
    @require_torch_gpu
    @slow
    def test_eager_matches_fa2_generate(self):
        """Tests that generate has equivalent outputs with FA2 and eager attention implementations."""
        # TODO (@joao @raushan) -- this test is failing the output checks on most models, investigate. After fixing,
        # check whether we still need the overwrites
        self._test_attention_implementation("flash_attention_2")

    def _check_outputs(self, output, config, use_cache=False, num_return_sequences=1, num_beams=1):
        input_batch_size = int(output.sequences.shape[0] / num_return_sequences)
        internal_batch_size = (
            input_batch_size * num_beams if num_beams > 1 else input_batch_size * num_return_sequences
        )

        seq_length = getattr(self.model_tester, "seq_length", None)
        seq_length = getattr(self.model_tester, "encoder_seq_length", seq_length)
        seq_length = getattr(self.model_tester, "text_seq_length", seq_length)

        config = config.text_config if hasattr(config, "text_config") else config

        gen_len = (
            output.sequences.shape[-1] - 1 if config.is_encoder_decoder else output.sequences.shape[-1] - seq_length
        )

        # in some models we subsample the sequence length in inner layers
        if hasattr(self.model_tester, "get_subsampled_output_lengths"):
            seq_length = self.model_tester.get_subsampled_output_lengths(seq_length)

        # scores
        self._check_scores(internal_batch_size, output.scores, length=gen_len, config=config)

        # unprocessed logits
        self._check_logits(internal_batch_size, output.logits, config=config)

        # Attentions
        if self.has_attentions:
            if config.is_encoder_decoder:
                # encoder
                self._check_encoder_attention_for_generate(
                    output.encoder_attentions, input_batch_size, config, seq_length
                )
                # decoder
                self._check_attentions_for_generate(
                    internal_batch_size,
                    output.decoder_attentions,
                    min_length=1,
                    max_length=output.sequences.shape[-1],
                    config=config,
                    use_cache=use_cache,
                )
            else:
                # if use_cache first input is equal to no use_cache, so skip here
                attentions = output.attentions if not use_cache else output.attentions[1:]
                min_length = seq_length if not use_cache else seq_length + 1
                self._check_attentions_for_generate(
                    internal_batch_size,
                    attentions=attentions,
                    min_length=min_length,
                    max_length=output.sequences.shape[-1],
                    config=config,
                    use_cache=use_cache,
                )

        # Hidden States
        if config.is_encoder_decoder:
            # encoder
            self._check_encoder_hidden_states_for_generate(
                output.encoder_hidden_states, input_batch_size, config, seq_length
            )

            # decoder
            self._check_hidden_states_for_generate(
                internal_batch_size,
                output.decoder_hidden_states,
                min_length=1,
                max_length=output.sequences.shape[-1],
                config=config,
                use_cache=use_cache,
            )
        else:
            # if use_cache first input is equal to no use_cache, so skip here
            hidden_states = output.hidden_states if not use_cache else output.hidden_states[1:]
            min_length = seq_length if not use_cache else seq_length + 1
            self._check_hidden_states_for_generate(
                internal_batch_size,
                hidden_states,
                min_length=min_length,
                max_length=output.sequences.shape[-1],
                config=config,
                use_cache=use_cache,
            )

        # Past Key Value States -- a few notes here:
        # 1. Its inner sequence length is with respect to the inputs of the latest forward pass, hence the "-1"
        # 2. We ignore models that have unique cache structures (e.g. mamba) or are in need of refatoring to match the
        #    standard cache format (e.g.gptbigcode )
        models_without_standard_cache = (
            "bamba",
            "ctrl",
            "fsmt",
            "gptbigcode",
            "mega",
            "reformer",
            "jamba",
            "mamba",
            "xlnet",
            "zamba",
        )
        has_standard_cache = not any(
            model_name in config.__class__.__name__.lower() for model_name in models_without_standard_cache
        )
        if has_standard_cache:
            if use_cache:
                past_key_values = output.past_key_values
                past_sequence_length = output.sequences.shape[-1] - 1
                self._check_past_key_values_for_generate(
                    internal_batch_size,
                    past_key_values,
                    seq_length=past_sequence_length,
                    config=config,
                )
            elif use_cache is False:
                self.assertTrue(output.past_key_values is None)

    def _check_scores(self, batch_size, scores, length, config):
        vocab_size = config.get_text_config(decoder=True).vocab_size
        expected_shape = (batch_size, vocab_size)
        self.assertIsInstance(scores, tuple)
        self.assertEqual(len(scores), length)
        self.assertListEqual([iter_scores.shape for iter_scores in scores], [expected_shape] * len(scores))

    def _check_logits(self, batch_size, scores, config):
        vocab_size = config.get_text_config(decoder=True).vocab_size
        self.assertIsInstance(scores, tuple)
        self.assertListEqual([iter_scores.shape[0] for iter_scores in scores], [batch_size] * len(scores))
        # vocabulary difference equal to one (imagegptmodel?) or zero (all other models)
        vocab_diff = vocab_size - scores[0].shape[-1]
        self.assertTrue(vocab_diff in [0, 1])
        self.assertListEqual([vocab_size - score.shape[-1] for score in scores], [vocab_diff] * len(scores))

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
            src_len = min_length + idx

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

    def _check_encoder_attention_for_generate(self, attentions, batch_size, config, seq_length):
        encoder_expected_shape = (batch_size, config.num_attention_heads, seq_length, seq_length)
        self.assertIsInstance(attentions, tuple)
        self.assertListEqual(
            [layer_attentions.shape for layer_attentions in attentions],
            [encoder_expected_shape] * len(attentions),
        )

    def _check_hidden_states_for_generate(
        self, batch_size, hidden_states, min_length, max_length, config, use_cache=False, num_beam_groups=1
    ):
        self.assertIsInstance(hidden_states, tuple)
        self.assertListEqual(
            [isinstance(iter_hidden_states, tuple) for iter_hidden_states in hidden_states],
            [True] * len(hidden_states),
        )
        self.assertEqual(len(hidden_states), (max_length - min_length) * num_beam_groups)

        for idx, iter_hidden_states in enumerate(hidden_states):
            seq_len = min_length + idx if not use_cache else 1
            expected_shape = (batch_size * num_beam_groups, seq_len, config.hidden_size)
            # check hidden size
            self.assertListEqual(
                [layer_hidden_states.shape for layer_hidden_states in iter_hidden_states],
                [expected_shape] * len(iter_hidden_states),
            )

    def _check_encoder_hidden_states_for_generate(self, hidden_states, batch_size, config, seq_length):
        encoder_expected_shape = (batch_size, seq_length, config.hidden_size)
        self.assertIsInstance(hidden_states, tuple)
        self.assertListEqual(
            [layer_hidden_states.shape for layer_hidden_states in hidden_states],
            [encoder_expected_shape] * len(hidden_states),
        )

    def _check_past_key_values_for_generate(self, batch_size, past_key_values, seq_length, config, num_beam_groups=1):
        self.assertIsInstance(past_key_values, tuple)
        self.assertListEqual(
            [isinstance(iter_past_key_values, tuple) for iter_past_key_values in past_key_values],
            [True] * len(past_key_values),
        )

        # (batch, head, seq_length, head_features)
        expected_shape = (
            batch_size * num_beam_groups,
            config.num_key_value_heads if hasattr(config, "num_key_value_heads") else config.num_attention_heads,
            seq_length,
            config.hidden_size // config.num_attention_heads,
        )
        # check shape key, value
        self.assertListEqual(
            [layer_past_key_values[0].shape for layer_past_key_values in past_key_values],
            [expected_shape] * len(past_key_values),
        )
        self.assertListEqual(
            [layer_past_key_values[1].shape for layer_past_key_values in past_key_values],
            [expected_shape] * len(past_key_values),
        )

    def _check_sequence_inside_sequence(self, tensor_1, tensor_2):
        # check if tensor_1 inside tensor_2 or tensor_2 inside tensor_1.
        # set to same device. we don't care what device.

        if not isinstance(tensor_1, list):
            tensor_1 = tensor_1.cpu().tolist()
        if not isinstance(tensor_2, list):
            tensor_2 = tensor_2.cpu().tolist()

        in_order = len(tensor_1) <= len(tensor_2)
        longer = tensor_2 if in_order else tensor_1
        shorter = tensor_1 if in_order else tensor_2

        flag = False
        chunk_size = len(shorter)
        for chunk_idx in range(len(longer) - chunk_size + 1):
            subseq = longer[chunk_idx : chunk_idx + chunk_size]
            if subseq == shorter:
                flag = True
                break

        self.assertTrue(flag)


@require_torch
class UtilsFunctionsTest(unittest.TestCase):
    def test_speculative_sampling(self):
        # assume vocab size 10, input length 5 + 3 generated candidates
        candidate_input_ids = torch.tensor([[8, 0, 3, 9, 8, 1, 4, 5]])  # input tokens
        candidate_logits = torch.tensor(
            [
                [
                    [-10.0, 10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0],  # generated 1
                    [-10.0, -10.0, -10.0, -10.0, 10.0, -10.0, -10.0, -10.0, -10.0, -10.0],  # generated 4
                    [-10.0, -10.0, -10.0, -10.0, -10.0, 10.0, -10.0, -10.0, -10.0, -10.0],  # generated 5
                ]
            ]
        )
        candidate_length = 3
        inf = float("inf")
        new_logits = torch.tensor(
            [
                [
                    [-10.0, 10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0],  # accepts 1
                    [-10.0, -10.0, -10.0, -10.0, 10.0, -10.0, -10.0, -10.0, -10.0, -10.0],  # accepts 4
                    [-inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, 10.0, -inf],  # rejects 5, accepts 8
                    [-10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0],  # N/A
                ]
            ]
        )
        last_assistant_token_is_eos = False
        validated_tokens, n_matches = _speculative_sampling(
            candidate_input_ids,
            candidate_logits,
            candidate_length,
            new_logits,
            last_assistant_token_is_eos,
        )
        self.assertTrue(n_matches.item() == 2)
        self.assertTrue(validated_tokens.tolist()[0] == [1, 4, 8])

    def test_speculative_sampling_target_distribution(self):
        """
        Asserts that the target distribution is preserved.
        Should help with catching issues like #32867.
        """
        # assume vocab size 10, input length 5 + 3 generated candidates
        candidate_input_ids = torch.tensor([[8, 0, 3, 9, 8, 1, 4, 5]])  # input tokens
        candidate_logits = torch.tensor(
            [
                [
                    [-10.0, 10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0],  # generated 1
                    [-10.0, -10.0, -10.0, -10.0, 10.0, -10.0, -10.0, -10.0, -10.0, -10.0],  # generated 4
                    [-10.0, -10.0, -10.0, -10.0, -10.0, 10.0, -10.0, -10.0, -10.0, -10.0],  # generated 5
                ]
            ]
        )
        candidate_length = 3
        inf = float("inf")
        new_logits = torch.tensor(
            [
                [
                    # accepts 1:
                    [-inf, 10.0, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf],
                    # accepts 4:
                    [-inf, -inf, -inf, -inf, 10.0, -inf, -inf, -inf, -inf, -inf],
                    # most likely to be 1 or 8, less likely to be 3, then 7, and should never be any other value:
                    [-inf, 2.0, -inf, 1.0, -inf, -inf, -inf, -0.01, 2.0, -inf],
                    # N/A:
                    [-inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf],
                ]
            ]
        )
        last_assistant_token_is_eos = False
        last_validated_token = []
        for _ in range(10_000):
            validated_tokens, n_matches = _speculative_sampling(
                candidate_input_ids,
                candidate_logits,
                candidate_length,
                new_logits,
                last_assistant_token_is_eos,
            )
            self.assertTrue(n_matches.item() == 2)
            self.assertTrue(validated_tokens.tolist()[0][0] == 1)
            self.assertTrue(validated_tokens.tolist()[0][1] == 4)
            self.assertTrue(validated_tokens.tolist()[0][2] in [1, 3, 7, 8])
            last_validated_token.append(validated_tokens.tolist()[0][2])
        # check that the most likely tokens are selected more often than the less likely ones
        last_token_counts = collections.Counter(last_validated_token)
        self.assertTrue(last_token_counts[1] > last_token_counts[3] > last_token_counts[7] > 0)
        self.assertTrue(last_token_counts[8] > last_token_counts[3])


@pytest.mark.generate
@require_torch
class GenerationIntegrationTests(unittest.TestCase, GenerationIntegrationTestsMixin):
    # setting framework_dependent_parameters needs to be gated, just like its contents' imports
    if is_torch_available():
        framework_dependent_parameters = {
            "AutoModelForCausalLM": AutoModelForCausalLM,
            "AutoModelForSpeechSeq2Seq": AutoModelForSpeechSeq2Seq,
            "AutoModelForSeq2SeqLM": AutoModelForSeq2SeqLM,
            "AutoModelForVision2Seq": AutoModelForVision2Seq,
            "LogitsProcessorList": LogitsProcessorList,
            "MinLengthLogitsProcessor": MinLengthLogitsProcessor,
            "create_tensor_fn": torch.tensor,
            "floats_tensor": floats_tensor,
            "return_tensors": "pt",
        }

    @slow
    def test_diverse_beam_search(self):
        # PT-only test: TF doesn't have a diverse beam search implementation
        article = """Justin Timberlake and Jessica Biel, welcome to parenthood.
        The celebrity couple announced the arrival of their son, Silas Randall Timberlake, in statements to People.
        "Silas was the middle name of Timberlake's maternal grandfather Bill Bomar, who died in 2012, while Randall is the musician's own middle name, as well as his father's first," People reports.
        The couple announced the pregnancy in January, with an Instagram post. It is the first baby for both."""

        bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn").to(torch_device)
        input_ids = bart_tokenizer(article, return_tensors="pt").input_ids.to(torch_device)

        outputs = bart_model.generate(
            input_ids,
            num_beams=4,
            num_return_sequences=2,
            num_beam_groups=4,
            diversity_penalty=2.0,
            remove_invalid_values=True,
        )

        generated_text = bart_tokenizer.batch_decode(outputs, skip_special_tokens=True)

        self.assertListEqual(
            generated_text,
            [
                "The couple announced the birth of their son, Silas Randall Timberlake, in a statement. Silas was the"
                " middle name of Timberlake's maternal grandfather Bill Bomar. Randall is the musician's own middle"
                " name, as well as his father's first. It is the first baby for both of them.",
                "Justin Timberlake and Jessica Biel have a son. The baby is named Silas Randall Timberlake. It is the"
                " first child for both. The couple announced the pregnancy in January. The name Silas is the middle"
                " name of Timberlake's maternal grandfather. It's also his own middle name.",
            ],
        )

    def test_max_length_if_input_embeds(self):
        # PT-only test: TF doesn't have StoppingCriteria
        article = "Today a dragon flew over Paris."
        model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2").to(torch_device)
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        input_ids = tokenizer(article, return_tensors="pt").input_ids.to(torch_device)
        inputs_embeds = model.get_input_embeddings()(input_ids)

        max_length = 20
        input_len = input_ids.shape[-1]
        out_gen = model.generate(input_ids=input_ids, max_length=max_length)
        out_gen_embeds = model.generate(inputs_embeds=inputs_embeds, max_length=max_length)
        self.assertEqual(out_gen.shape[-1], input_len + out_gen_embeds.shape[-1])

    def test_min_length_if_input_embeds(self):
        # PT-only test: TF doesn't have StoppingCriteria
        article = "Today a dragon flew over Paris."
        model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2").to(torch_device)
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        input_ids = tokenizer(article, return_tensors="pt").input_ids.to(torch_device)
        inputs_embeds = model.get_input_embeddings()(input_ids)

        min_length = 10
        input_len = input_ids.shape[-1]
        out_gen = model.generate(input_ids=input_ids, min_length=min_length)
        out_gen_embeds = model.generate(inputs_embeds=inputs_embeds, min_length=min_length)
        self.assertEqual(out_gen.shape[-1], input_len + out_gen_embeds.shape[-1])

    def test_custom_stopping_criteria_overload_error(self):
        # PT-only test: TF doesn't have StoppingCriteria
        article = """Justin Timberlake and Jessica Biel, welcome to parenthood."""
        bart_tokenizer = BartTokenizer.from_pretrained("sshleifer/bart-tiny-random")
        bart_model = BartForConditionalGeneration.from_pretrained("sshleifer/bart-tiny-random").to(torch_device)

        input_ids = bart_tokenizer(article, return_tensors="pt").input_ids.to(torch_device)
        stopping_criteria = StoppingCriteriaList()
        stopping_criteria.append(MaxLengthCriteria(max_length=42))
        with self.assertRaises(ValueError):
            bart_model.generate(input_ids, stopping_criteria=stopping_criteria)
        with self.assertRaises(ValueError):
            bart_model.generate(input_ids, stopping_criteria=stopping_criteria, max_length=32)

    def test_custom_stopping_criteria(self):
        # PT-only test: TF doesn't have StoppingCriteria
        article = """Justin Timberlake and Jessica Biel, welcome to parenthood."""
        bart_tokenizer = BartTokenizer.from_pretrained("sshleifer/bart-tiny-random")
        bart_model = BartForConditionalGeneration.from_pretrained("sshleifer/bart-tiny-random").to(torch_device)
        input_ids = bart_tokenizer(article, return_tensors="pt").input_ids.to(torch_device)

        class DummyCriteria(StoppingCriteria):
            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
                return input_ids.shape[-1] >= 20

        stopping_criteria = StoppingCriteriaList()
        stopping_criteria.append(DummyCriteria())

        self.assertEqual(
            list(bart_model.generate(input_ids, stopping_criteria=stopping_criteria, max_length=22).shape),
            [1, 20],
        )
        self.assertEqual(
            list(bart_model.generate(input_ids, stopping_criteria=stopping_criteria, max_length=18).shape),
            [1, 18],
        )

    # TODO (joao): replace `stop_sequence` in the pipeline by the more recent `generate` functionality
    def test_stop_sequence_stopping_criteria(self):
        # PT-only test: TF doesn't have StoppingCriteria
        prompt = """Hello I believe in"""
        generator = pipeline("text-generation", model="hf-internal-testing/tiny-random-bart")
        output = generator(prompt)
        self.assertEqual(
            output,
            [{"generated_text": ("Hello I believe in we we we we we we we we we")}],
        )

        output = generator(prompt, stop_sequence=" we")
        self.assertEqual(output, [{"generated_text": "Hello I believe in we"}])

    def test_generate_non_nlp_input_ids_as_kwarg(self):
        # PT-only test: AFAIK there's no non-NLP model architecture in TF that supports `input_ids` as its only input
        model = ImageGPTForCausalImageModeling.from_pretrained(
            "hf-internal-testing/tiny-random-imagegpt", max_length=10
        ).to(torch_device)
        input_ids = ids_tensor((3, 5), vocab_size=10)

        output_sequences_kwargs = model.generate(input_ids=input_ids).cpu()
        output_sequences = model.generate(input_ids).cpu()

        self.assertListEqual(output_sequences.tolist(), output_sequences_kwargs.tolist())
        self.assertEqual(output_sequences.shape, (3, 10))

    def test_generate_input_values_as_encoder_kwarg(self):
        # PT-only test: AFAIK there's no generate-capable architecture in TF that supports `input_values` as its input
        input_values = floats_tensor((2, 250))
        model = SpeechEncoderDecoderModel.from_pretrained("hf-internal-testing/tiny-random-speech-encoder-decoder")
        model = model.to(torch_device)
        output_sequences_kwargs = model.generate(input_values=input_values, max_length=5).cpu()
        output_sequences = model.generate(input_values, max_length=5).cpu()

        self.assertListEqual(output_sequences.tolist(), output_sequences_kwargs.tolist())
        self.assertEqual(output_sequences.shape, (2, 5))

    def test_transition_scores_group_beam_search_encoder_decoder(self):
        # PT-only test: TF doesn't have group beam search
        articles = [
            "Justin Timberlake and Jessica Biel, welcome to parenthood.",
            "Michael Phelps is arguably the most decorated Olympian of all time.",
        ]
        tokenizer = BartTokenizer.from_pretrained("hf-internal-testing/tiny-random-bart")
        model = BartForConditionalGeneration.from_pretrained(
            "hf-internal-testing/tiny-random-bart",
            max_length=10,
            num_beams=2,
            num_beam_groups=2,
            num_return_sequences=2,
            diversity_penalty=1.0,
            eos_token_id=None,
            return_dict_in_generate=True,
            output_scores=True,
            length_penalty=0.0,
        )
        model = model.to(torch_device)

        input_ids = tokenizer(articles, return_tensors="pt", padding=True).input_ids.to(torch_device)
        outputs = model.generate(input_ids=input_ids)

        transition_scores = model.compute_transition_scores(outputs.sequences, outputs.scores, outputs.beam_indices)
        transition_scores_sum = transition_scores.sum(-1)

        self.assertTrue(torch.allclose(transition_scores_sum, outputs.sequences_scores, atol=1e-3))

    def test_beam_search_low_memory(self):
        tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
        model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model_inputs = tokenizer("I", return_tensors="pt")["input_ids"]

        low_output = model.generate(model_inputs, max_new_tokens=40, num_beams=5, early_stopping=True, low_memory=True)

        high_output = model.generate(
            model_inputs, max_new_tokens=40, num_beams=5, early_stopping=True, low_memory=False
        )
        self.assertListEqual(low_output.tolist(), high_output.tolist())

    @slow
    def test_green_red_watermark_generation(self):
        model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2").to(torch_device)
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model_inputs = tokenizer("I will be", return_tensors="pt").to(torch_device)
        input_len = model_inputs["input_ids"].shape[-1]

        # generation should work with both input types: WatermarkingConfig or Dict, so let's check it here :)
        watermark_config = WatermarkingConfig(bias=2.5, seeding_scheme="selfhash")
        _ = model.generate(**model_inputs, watermarking_config=watermark_config, do_sample=False, max_length=15)

        # We will not check watermarked text, since we check it in `logits_processors` tests
        # Checking if generated ids are as expected fails on different hardware
        args = {
            "bias": 2.0,
            "context_width": 1,
            "seeding_scheme": "selfhash",
            "greenlist_ratio": 0.25,
            "hashing_key": 15485863,
        }
        output = model.generate(**model_inputs, do_sample=False, max_length=15)
        output_selfhash = model.generate(**model_inputs, watermarking_config=args, do_sample=False, max_length=15)

        # Check that the detector is detecting watermarked text
        detector = WatermarkDetector(model_config=model.config, device=torch_device, watermarking_config=args)
        detection_out_watermarked = detector(output_selfhash[:, input_len:], return_dict=True)
        detection_out = detector(output[:, input_len:], return_dict=True)

        self.assertListEqual(detection_out_watermarked.prediction.tolist(), [True])
        self.assertListEqual(detection_out.prediction.tolist(), [False])

    """Check the mean bias inserted by the watermarking algorithm."""

    @slow
    def test_synthid_text_watermark_generation_mean_expected_bias(self):
        model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2").to(torch_device)
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model_inputs = tokenizer("I will be", return_tensors="pt").to(torch_device)
        input_len = 5
        batch_size = 200

        # generation should work with both input types: WatermarkingConfig or Dict, so let's check it here :)
        watermark_config = SynthIDTextWatermarkingConfig(keys=[10, 20], ngram_len=5, debug_mode=True)
        logits_processor = watermark_config.construct_processor(model.config.vocab_size, torch_device)
        mean_g_values_repeats = []
        for _ in range(40):
            input_ids = torch.zeros(
                (batch_size, input_len),
                dtype=torch.int64,
                device=torch_device,
            )
            model_inputs = {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids, device=torch_device),
            }
            output = model.generate(
                **model_inputs, watermarking_config=watermark_config, do_sample=True, max_length=500, top_k=1000
            )
            g_values = logits_processor.compute_g_values(input_ids=output[:, input_len:])
            context_repetition_mask = logits_processor.compute_context_repetition_mask(
                input_ids=output[:, input_len:],
            ).unsqueeze(dim=2)

            mean_g_values = torch.masked.mean(
                g_values,
                mask=context_repetition_mask,
                dim=0,
                keepdim=True,
                dtype=torch.float64,
            )
            mean_g_values_repeats.append(mean_g_values)

        mean_g_values = torch.concat(mean_g_values_repeats, dim=0).mean(dim=0)
        expected_mean_g_value = logits_processor.expected_mean_g_value(
            vocab_size=model.config.vocab_size,
        )
        atol = 0.03
        is_close = torch.isclose(
            mean_g_values,
            torch.tensor(expected_mean_g_value, dtype=torch.float64),
            atol=atol,
            rtol=0,
        )
        self.assertTrue(torch.all(is_close))

    @slow
    def test_beam_search_example_integration(self):
        # PT-only test: TF doesn't have a BeamSearchScorer
        # exactly the example provided in the docstrings of beam search, which previously
        # failed after directly copying from it. Refer to PR #15555
        tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
        model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-base")

        encoder_input_str = "translate English to German: How old are you?"
        encoder_input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids

        # lets run beam search using 3 beams
        num_beams = 3
        # define decoder start token ids
        input_ids = torch.ones((1, 1), device=model.device, dtype=torch.long)
        input_ids = input_ids * model.config.decoder_start_token_id

        # add encoder_outputs to model keyword arguments
        model_kwargs = {"encoder_outputs": model.get_encoder()(encoder_input_ids, return_dict=True)}

        outputs = model.generate(
            input_ids, num_beams=num_beams, min_length=5, eos_token_id=model.config.eos_token_id, **model_kwargs
        )
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        self.assertListEqual(outputs, ["Wie alt bist du?"])

    @slow
    def test_constrained_beam_search(self):
        # PT-only test: TF doesn't have constrained beam search
        model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").to(torch_device)
        tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")

        force_tokens = tokenizer("scared", add_prefix_space=True, add_special_tokens=False).input_ids
        force_tokens_2 = tokenizer("big weapons", add_prefix_space=True, add_special_tokens=False).input_ids

        constraints = [
            PhrasalConstraint(force_tokens),
            PhrasalConstraint(force_tokens_2),
        ]

        starting_text = ["The soldiers were not prepared and"]

        input_ids = tokenizer(starting_text, return_tensors="pt").input_ids.to(torch_device)

        outputs = model.generate(
            input_ids,
            constraints=constraints,
            num_beams=10,
            num_return_sequences=1,
            no_repeat_ngram_size=1,
            max_length=30,
            remove_invalid_values=True,
        )

        generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        self.assertListEqual(
            generated_text,
            [
                "The soldiers were not prepared and didn't know what to do. They had no idea how they would react if"
                " the enemy attacked them, big weapons scared"
            ],
        )

    @slow
    def test_constrained_beam_search_mixed(self):
        # PT-only test: TF doesn't have constrained beam search
        model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").to(torch_device)
        tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")

        force_phrase = tokenizer("scared", add_prefix_space=True, add_special_tokens=False).input_ids
        flexible_phrases = tokenizer(
            ["scream", "screams", "screaming", "screamed"], add_prefix_space=True, add_special_tokens=False
        ).input_ids

        constraints = [
            PhrasalConstraint(force_phrase),
            DisjunctiveConstraint(flexible_phrases),
        ]

        starting_text = ["The soldiers", "The child"]

        input_ids = tokenizer(starting_text, return_tensors="pt").input_ids.to(torch_device)

        outputs = model.generate(
            input_ids,
            constraints=constraints,
            num_beams=10,
            num_return_sequences=1,
            no_repeat_ngram_size=1,
            # max_length=20,
            remove_invalid_values=True,
        )

        generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        self.assertListEqual(
            generated_text,
            [
                "The soldiers, who had been stationed at the base for more than a year before being evacuated"
                " screaming scared",
                "The child was taken to a local hospital where he died.\n 'I don't think screaming scared",
            ],
        )

    @slow
    def test_constrained_beam_search_mixed_mixin(self):
        # PT-only test: TF doesn't have constrained beam search
        model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").to(torch_device)
        tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")

        force_word = "scared"
        force_flexible = ["scream", "screams", "screaming", "screamed"]

        force_words_ids = [
            tokenizer([force_word], add_prefix_space=True, add_special_tokens=False).input_ids,
            tokenizer(force_flexible, add_prefix_space=True, add_special_tokens=False).input_ids,
        ]

        starting_text = ["The soldiers", "The child"]

        input_ids = tokenizer(starting_text, return_tensors="pt").input_ids.to(torch_device)

        outputs = model.generate(
            input_ids,
            force_words_ids=force_words_ids,
            num_beams=10,
            num_return_sequences=1,
            no_repeat_ngram_size=1,
            remove_invalid_values=True,
        )

        generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        self.assertListEqual(
            generated_text,
            [
                "The soldiers, who had been stationed at the base for more than a year before being evacuated"
                " screaming scared",
                "The child was taken to a local hospital where he died.\n 'I don't think screaming scared",
            ],
        )

    @slow
    def test_cfg_mixin(self):
        model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").to(torch_device)
        tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")

        input = tokenizer(["The dragon flew over Paris,"], return_tensors="pt", return_attention_mask=True)
        input["input_ids"] = input["input_ids"].to(torch_device)
        input["attention_mask"] = input["attention_mask"].to(torch_device)

        outputs = model.generate(**input, max_new_tokens=32, guidance_scale=1.5)
        generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        self.assertListEqual(
            generated_text,
            [
                "The dragon flew over Paris, landing in the Rue de la Bastille. The crowd was so excited "
                'that they had to leave the city.\n\n"We\'re going to Paris!"\n'
            ],
        )

        neg = tokenizer(["France,"], return_tensors="pt", return_attention_mask=True)
        neg["input_ids"] = neg["input_ids"].to(torch_device)
        neg["attention_mask"] = neg["attention_mask"].to(torch_device)
        outputs = model.generate(
            **input,
            max_new_tokens=32,
            guidance_scale=1.5,
            negative_prompt_ids=neg["input_ids"],
            negative_prompt_attention_mask=neg["attention_mask"],
        )
        generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        self.assertListEqual(
            generated_text,
            [
                'The dragon flew over Paris, landing on the pavement.\n\n"Paris!"\n\n"Paris!"\n\n"'
                'Paris!"\n\n"Paris!"\n\n"Paris!"\n\n'
            ],
        )

    @slow
    def test_constrained_beam_search_example_translation_mixin(self):
        # PT-only test: TF doesn't have constrained beam search
        tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
        model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-base")

        encoder_input_str = "translate English to German: How old are you?"
        force_words = ["sind"]

        input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids
        force_words_ids = tokenizer(force_words, add_special_tokens=False).input_ids

        outputs = model.generate(
            input_ids,
            force_words_ids=force_words_ids,
            num_beams=10,
            num_return_sequences=1,
            no_repeat_ngram_size=1,
            remove_invalid_values=True,
        )

        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        self.assertListEqual(outputs, ["Wie alt sind Sie?"])

    @slow
    def test_constrained_beam_search_example_integration(self):
        # PT-only test: TF doesn't have constrained beam search
        tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
        model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-base")

        encoder_input_str = "translate English to German: How old are you?"
        encoder_input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids

        # lets run beam search using 5 beams
        num_beams = 5
        # define decoder start token ids
        input_ids = torch.ones((1, 1), device=model.device, dtype=torch.long)
        input_ids = input_ids * model.config.decoder_start_token_id

        # add encoder_outputs to model keyword arguments
        model_kwargs = {"encoder_outputs": model.get_encoder()(encoder_input_ids, return_dict=True)}

        constraint_str = "sind"
        constraint_token_ids = tokenizer.encode(constraint_str)[:-1]  # remove eos token

        outputs = model.generate(
            input_ids,
            num_beams=num_beams,
            force_words_ids=[constraint_token_ids],
            min_length=5,
            eos_token_id=model.config.eos_token_id,
            **model_kwargs,
        )
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        self.assertListEqual(outputs, ["Wie alt sind Sie?"])

    @slow
    def test_per_row_stopping_criteria(self):
        text = [
            "They completed the challenging puzzle, revealing the hidden",
            "Today a dragon flew over France",
            "The aroma of freshly baked pizza filled the kitchen",
        ]
        stop_strings = ["secrets"]

        model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2").to(torch_device)
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        tokenizer.padding_side = "left"
        tokenizer.pad_token_id = tokenizer.eos_token_id
        input_ids = tokenizer(text, return_tensors="pt", padding="longest", add_special_tokens=False).input_ids.to(
            torch_device
        )

        # normal generation with one stopping criteria
        out = model.generate(input_ids, max_length=15)
        out_text = tokenizer.batch_decode(out)
        expected_out = [
            "They completed the challenging puzzle, revealing the hidden secrets of the world.\n",
            "<|endoftext|><|endoftext|><|endoftext|>Today a dragon flew over France and the French government was forced",
            "The aroma of freshly baked pizza filled the kitchen with a sense of freshness",
        ]
        self.assertListEqual(out_text, expected_out)

        # generation should stop at "secrets" for first batch only, filling the rest with eos tokens
        out = model.generate(input_ids, max_length=15, stop_strings=stop_strings, tokenizer=tokenizer)
        out_text = tokenizer.batch_decode(out)
        expected_out = [
            "They completed the challenging puzzle, revealing the hidden secrets<|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|>",
            "<|endoftext|><|endoftext|><|endoftext|>Today a dragon flew over France and the French government was forced",
            "The aroma of freshly baked pizza filled the kitchen with a sense of freshness",
        ]
        self.assertListEqual(out_text, expected_out)

    def test_constrained_beam_search_mixin_type_checks(self):
        # PT-only test: TF doesn't have constrained beam search
        tokenizer = AutoTokenizer.from_pretrained("patrickvonplaten/t5-tiny-random")
        model = AutoModelForSeq2SeqLM.from_pretrained("patrickvonplaten/t5-tiny-random")

        encoder_input_str = "translate English to German: How old are you?"
        input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids

        with self.assertRaises(ValueError):
            force_words = ["sind"]
            force_words_ids = tokenizer(force_words, return_tensors="pt").input_ids
            model.generate(
                input_ids,
                force_words_ids=force_words_ids,
                num_beams=10,
                num_return_sequences=1,
                no_repeat_ngram_size=1,
                remove_invalid_values=True,
            )

        with self.assertRaises(ValueError):
            force_words = ["sind"]
            force_words_ids = [tokenizer(force_words, return_tensors="pt").input_ids]
            model.generate(
                input_ids,
                force_words_ids=force_words_ids,
                num_beams=10,
                num_return_sequences=1,
                no_repeat_ngram_size=1,
                remove_invalid_values=True,
            )

        with self.assertRaises(ValueError):
            model.generate(input_ids, force_words_ids=[])

        with self.assertRaises(ValueError):
            model.generate(input_ids, force_words_ids=[[-1]])

        with self.assertRaises(ValueError):
            model.generate(input_ids, force_words_ids=[[[-1]]])

    def test_batched_decoder_start_id(self):
        # PT-only test: TF doesn't support batched_decoder_start_id
        articles = [
            "Justin Timberlake and Jessica Biel, welcome to parenthood.",
            "Michael Phelps is arguably the most decorated Olympian of all time.",
        ]
        bart_tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-bart")
        bart_model = BartForConditionalGeneration.from_pretrained("hf-internal-testing/tiny-random-bart").to(
            torch_device
        )
        input_ids = bart_tokenizer(articles, return_tensors="pt", padding=True).input_ids.to(torch_device)
        decoder_start_token_id = bart_model.generation_config.decoder_start_token_id
        decoder_start_token_id_batch = [decoder_start_token_id] * input_ids.shape[0]

        outputs = bart_model.generate(input_ids, decoder_start_token_id=decoder_start_token_id)

        outputs_batched_ids = bart_model.generate(input_ids, decoder_start_token_id=decoder_start_token_id_batch)

        self.assertListEqual(outputs.tolist(), outputs_batched_ids.tolist())

    def test_decoder_start_id_from_config(self):
        # Refer to: (#30899)
        articles = [
            "Justin Timberlake and Jessica Biel, welcome to parenthood.",
            "Michael Phelps is arguably the most decorated Olympian of all time.",
        ]
        bart_tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-bart")
        bart_model = BartForConditionalGeneration.from_pretrained("hf-internal-testing/tiny-random-bart").to(
            torch_device
        )
        input_ids = bart_tokenizer(articles, return_tensors="pt", padding=True).input_ids.to(torch_device)
        decoder_start_token_id = bart_model.generation_config.decoder_start_token_id

        # we should be able to take `decoder_start_token_id` from model's generation config if user passes a `GenerationConfig` type
        outputs = bart_model.generate(input_ids, generation_config=GenerationConfig(do_sample=False))

        # If the generatoin config has no `decoder_start_token_id` or `bos_token_id`, we will raise an error unless user passes it in config
        bart_model.generation_config.decoder_start_token_id = None
        bart_model.generation_config.bos_token_id = None
        outputs_with_user_id = bart_model.generate(
            input_ids,
            generation_config=GenerationConfig(do_sample=False, decoder_start_token_id=decoder_start_token_id),
        )

        self.assertListEqual(outputs.tolist(), outputs_with_user_id.tolist())

        with self.assertRaises(ValueError):
            outputs = bart_model.generate(input_ids, generation_config=GenerationConfig(do_sample=False))

    def test_contrastive_search_batched(self):
        # PT-only test: TF doesn't have constrained beam search
        # Tests that contrastive search works with batched inputs (i.e. has the same output as for non-batched inputs)
        articles = ["Foo", "Bar Baz"]
        tokenizer = BartTokenizer.from_pretrained("hf-internal-testing/tiny-random-bart")
        model = BartForConditionalGeneration.from_pretrained("hf-internal-testing/tiny-random-bart").to(torch_device)

        model.config.eos_token_id = None
        input_ids_batched = tokenizer(articles, padding=True, return_tensors="pt").input_ids.to(torch_device)
        input_ids = tokenizer(articles[1], return_tensors="pt").input_ids.to(torch_device)

        output_sequences_batched = model.generate(
            input_ids=input_ids_batched, penalty_alpha=0.6, top_k=4, return_dict_in_generate=True, output_scores=True
        )
        output_sequences = model.generate(
            input_ids=input_ids, penalty_alpha=0.6, top_k=4, return_dict_in_generate=True, output_scores=True
        )

        batched_out = tokenizer.decode(output_sequences_batched.sequences[1], skip_special_tokens=True)
        out = tokenizer.decode(output_sequences.sequences[0], skip_special_tokens=True)
        self.assertEqual(batched_out, out)

        # output_sequences_batched.scores[0][1] -> 1st set of logits, 2nd sequence
        max_score_diff = (output_sequences_batched.scores[0][1] - output_sequences.scores[0][0]).abs().max()
        self.assertTrue(max_score_diff < 1e-5)

    def test_logits_processor_not_inplace(self):
        # PT-only test: TF fixes were not made
        article = "Today a dragon flew over Paris."
        model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2").to(torch_device)
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        input_ids = tokenizer(article, return_tensors="pt").input_ids.to(torch_device)

        out = model.generate(input_ids, output_logits=True, output_scores=True, return_dict_in_generate=True)
        out_with_temp = model.generate(
            input_ids,
            temperature=0.5,
            do_sample=True,
            output_logits=True,
            output_scores=True,
            return_dict_in_generate=True,
        )

        # if no logits processor is used, scores == logits. Otherwise, the processor has to modify the scores
        self.assertListEqual(out.logits[-1].tolist(), out.scores[-1].tolist())
        self.assertNotEqual(out_with_temp.logits[-1].tolist(), out_with_temp.scores[-1].tolist())

    def test_eos_token_id_int_and_list_top_k_top_sampling(self):
        # Has TF equivalent: this test relies on random sampling
        generation_kwargs = {
            "do_sample": True,
            "num_beams": 1,
            "top_p": 0.7,
            "top_k": 10,
            "temperature": 0.7,
        }
        expectation = 20

        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        text = """Hello, my dog is cute and"""
        tokens = tokenizer(text, return_tensors="pt").to(torch_device)
        model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2").to(torch_device)

        # Only some seeds will work both on CPU/GPU for a fixed `expectation` value.
        # The selected seed is not guaranteed to work on all torch versions.
        torch.manual_seed(1)
        eos_token_id = 846
        generated_tokens = model.generate(**tokens, eos_token_id=eos_token_id, **generation_kwargs)
        self.assertTrue(expectation == len(generated_tokens[0]))

        torch.manual_seed(1)
        eos_token_id = [846, 198]
        generated_tokens = model.generate(**tokens, eos_token_id=eos_token_id, **generation_kwargs)
        self.assertTrue(expectation == len(generated_tokens[0]))

    def test_model_kwarg_encoder_signature_filtering(self):
        # Has TF equivalent: ample use of framework-specific code
        bart_tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-bart")
        article = """Hugging Face is a technology company based in New York and Paris."""
        input_ids = bart_tokenizer(article, return_tensors="pt").input_ids.to(torch_device)
        bart_model = BartForConditionalGeneration.from_pretrained("hf-internal-testing/tiny-random-bart").to(
            torch_device
        )
        output = bart_model.generate(input_ids).cpu().numpy()

        # Let's create a fake model that has a different signature. In particular, this fake model accepts "foo" as an
        # argument. Because "foo" is not in the encoder signature and doesn't start with "decoder_", it will be part of
        # the encoder kwargs prior to signature filtering, which would lead to an exception. But filtering kicks in and
        # saves the day.
        class FakeBart(BartForConditionalGeneration):
            def forward(self, input_ids, foo=None, **kwargs):
                return super().forward(input_ids, **kwargs)

        bart_model = FakeBart.from_pretrained("hf-internal-testing/tiny-random-bart").to(torch_device)
        fake_output = bart_model.generate(input_ids, foo="bar").cpu().numpy()
        self.assertTrue(np.array_equal(output, fake_output))

        # Encoder signature filtering only kicks in if it doesn't accept wildcard kwargs. The following test will fail
        # because it doesn't do signature filtering.
        class FakeEncoder(bart_model.model.encoder.__class__):
            def forward(self, input_ids, **kwargs):
                return super().forward(input_ids, **kwargs)

        fake_encoder = FakeEncoder(bart_model.config, bart_model.model.shared).to(torch_device)
        bart_model.model.encoder = fake_encoder

        # Normal generation still works (the output will be different because the encoder weights are different)
        fake_output = bart_model.generate(input_ids).cpu().numpy()
        with self.assertRaises(TypeError):
            # FakeEncoder.forward() accepts **kwargs -> no filtering -> type error due to unexpected input "foo"
            bart_model.generate(input_ids, foo="bar")

    def test_default_max_length_warning(self):
        model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2").to(torch_device)
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        model.generation_config.pad_token_id = tokenizer.eos_token_id

        text = "Hello world"
        tokenized_inputs = tokenizer([text], return_tensors="pt")
        input_ids = tokenized_inputs.input_ids.to(torch_device)

        # Default generation config value of 20 -> emits warning
        with self.assertWarns(UserWarning):
            model.generate(input_ids)

        # Explicitly setting max_length to 20 -> no warning
        with warnings.catch_warnings(record=True) as warning_list:
            model.generate(input_ids, max_length=20)
            self.assertEqual(len(warning_list), 0)

        # Generation config max_length != 20 -> no warning
        with warnings.catch_warnings(record=True) as warning_list:
            # generation_config is modified -> legacy mode is disabled = generation_config takes precedence
            model.generation_config.max_length = 10
            model.generate(input_ids)
            self.assertEqual(len(warning_list), 0)

    def test_length_warning_assisted_generation(self):
        # PT-only test: TF doesn't support assisted decoding yet.
        model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2").to(torch_device)
        assistant = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2").to(torch_device)
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        model.generation_config.pad_token_id = tokenizer.eos_token_id
        assistant.generation_config.pad_token_id = tokenizer.eos_token_id

        text = "Hello world"
        tokenized_inputs = tokenizer([text], return_tensors="pt")
        input_ids = tokenized_inputs.input_ids.to(torch_device)

        # This should not raise any warning that min length is not feasible in candidate generation
        with warnings.catch_warnings(record=True) as warning_list:
            model.generate(
                input_ids,
                assistant_model=assistant,
                min_new_tokens=10,
                max_length=20,
            )
            self.assertEqual(len(warning_list), 0)

    def test_default_assisted_generation(self):
        # Initialize the GenerationConfig object
        config = GenerationConfig()

        # Check the default values
        self.assertEqual(config.num_assistant_tokens, 20)
        self.assertEqual(config.num_assistant_tokens_schedule, "constant")
        self.assertEqual(config.assistant_confidence_threshold, 0.4)
        self.assertEqual(config.is_assistant, False)

    def test_generated_length_assisted_generation(self):
        # PT-only test: TF doesn't support assisted decoding yet.
        model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2").to(torch_device)
        assistant = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2").to(torch_device)
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        model.generation_config.pad_token_id = tokenizer.eos_token_id
        assistant.generation_config.pad_token_id = tokenizer.eos_token_id

        text = "Hello world"
        tokenized_inputs = tokenizer([text], return_tensors="pt")
        input_ids = tokenized_inputs.input_ids.to(torch_device)
        input_length = input_ids.shape[-1]

        out = model.generate(
            input_ids,
            assistant_model=assistant,
            min_new_tokens=10,
            max_new_tokens=20,
        )
        self.assertTrue((10 + input_length) <= out.shape[-1] <= (20 + input_length))

        out = model.generate(
            input_ids,
            assistant_model=assistant,
            min_new_tokens=10,
        )
        self.assertTrue((input_length + 10) <= out.shape[-1] <= 20)

    def test_model_kwarg_assisted_decoding_decoder_only(self):
        # PT-only test: TF doesn't support assisted decoding yet.
        model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2").to(torch_device)
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        model.generation_config.pad_token_id = tokenizer.eos_token_id

        text = "Hello world"
        tokenized_inputs = tokenizer([text], return_tensors="pt")
        input_ids = tokenized_inputs.input_ids.to(torch_device)

        # Traditional way of generating text
        outputs_normal = model.generate(input_ids)
        self.assertEqual(outputs_normal.shape, (1, 20))

        # Should be different with token_type_ids
        outputs_tti = model.generate(
            input_ids,
            token_type_ids=torch.zeros(input_ids.shape, dtype=torch.long).to(torch_device),
        )
        with self.assertRaises(AssertionError):
            self.assertListEqual(outputs_tti.tolist(), outputs_normal.tolist())

        # Assistant model
        assistant = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2").to(torch_device)
        assistant.config.pad_token_id = tokenizer.eos_token_id

        # If assisted generation passes model_kwargs correctly, should be same as previous
        outputs_assisted = model.generate(
            input_ids,
            token_type_ids=torch.zeros(input_ids.shape, dtype=torch.long).to(torch_device),
            assistant_model=assistant,
        )
        self.assertListEqual(outputs_assisted.tolist(), outputs_tti.tolist())

    def test_model_kwarg_assisted_decoding_encoder_decoder(self):
        """
        Tests that the following scenario is compatible with assisted generation:
        1. encoder-decoder main model
        2. encoder-decoder assistant model
        3. both have a custom input
        (e.g. Whisper)
        """

        # PT-only test: TF doesn't support assisted decoding yet.
        # Bart subclass with a kwarg that distorts the output
        class FakeBart(BartForConditionalGeneration):
            def forward(self, input_ids, past_key_values, foo=False, **kwargs):
                outs = super().forward(input_ids, past_key_values=past_key_values, **kwargs)
                if foo:
                    outs["logits"][:, :, :] = 0.0
                return outs

            def prepare_inputs_for_generation(self, *args, foo=False, encoder_outputs=None, **kwargs):
                kwargs["encoder_outputs"] = encoder_outputs
                inputs = super().prepare_inputs_for_generation(*args, **kwargs)
                inputs["foo"] = foo
                return inputs

        model = FakeBart.from_pretrained("hf-internal-testing/tiny-random-BartForConditionalGeneration").to(
            torch_device
        )
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-BartForConditionalGeneration")

        text = "Hello world"
        tokenized_inputs = tokenizer([text], return_tensors="pt")
        input_ids = tokenized_inputs.input_ids.to(torch_device)

        # Traditional way of generating text
        outputs_normal = model.generate(input_ids)
        self.assertEqual(outputs_normal.shape, (1, 20))

        # Should be different with foo
        outputs_foo = model.generate(input_ids, foo=True)
        with self.assertRaises(AssertionError):
            self.assertListEqual(outputs_foo.tolist(), outputs_normal.tolist())

        # Assistant model
        assistant = FakeBart.from_pretrained("hf-internal-testing/tiny-random-BartForConditionalGeneration").to(
            torch_device
        )

        # If assisted generation passes model_kwargs correctly, should be same as previous
        outputs_assisted = model.generate(
            input_ids,
            foo=True,
            assistant_model=assistant,
        )
        self.assertListEqual(outputs_assisted.tolist(), outputs_foo.tolist())

        # Check that passing encoder_outputs directly also works as expected
        encoder_outputs = assistant.get_encoder()(input_ids)

        outputs_assisted = model.generate(
            foo=True,
            assistant_model=assistant,
            encoder_outputs=encoder_outputs,
            assistant_encoder_outputs=encoder_outputs,
        )
        self.assertListEqual(outputs_assisted.tolist(), outputs_foo.tolist())

    def test_assisted_decoding_encoder_decoder_shared_encoder(self):
        """
        Tests that the following scenario is compatible with assisted generation:
        1. encoder-decoder main model
        2. decoder-only assistant model
        3. both have a custom input
        (e.g. DistilWhisper)
        """

        # PT-only test: TF doesn't support assisted decoding yet.
        # Bart subclass with a kwarg called foo that distorts the output
        class FakeBartSeq2Seq(BartForConditionalGeneration):
            def forward(self, input_ids, foo=False, **kwargs):
                outs = super().forward(input_ids, **kwargs)
                if foo:
                    outs["logits"][:, :, :] = 0.0
                return outs

            def prepare_inputs_for_generation(self, *args, foo=False, encoder_outputs=None, **kwargs):
                kwargs["encoder_outputs"] = encoder_outputs
                inputs = super().prepare_inputs_for_generation(*args, **kwargs)
                inputs["foo"] = foo
                return inputs

        class FakeBartCausalLM(BartForCausalLM):
            def forward(self, input_ids, attention_mask, past_key_values, foo=False, **kwargs):
                outs = super().forward(input_ids, attention_mask, past_key_values=past_key_values, **kwargs)
                if foo:
                    outs["logits"][:, :, :] = 0.0
                return outs

            def prepare_inputs_for_generation(self, *args, foo=False, encoder_outputs=None, **kwargs):
                kwargs["encoder_outputs"] = encoder_outputs
                inputs = super().prepare_inputs_for_generation(*args, **kwargs)
                inputs["foo"] = foo
                return inputs

        model = FakeBartSeq2Seq.from_pretrained("hf-internal-testing/tiny-random-BartForConditionalGeneration").to(
            torch_device
        )
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-BartForConditionalGeneration")

        text = "Hello world"
        tokenized_inputs = tokenizer([text], return_tensors="pt")
        input_ids = tokenized_inputs.input_ids.to(torch_device)

        # Traditional way of generating text
        outputs_normal = model.generate(input_ids)
        self.assertEqual(outputs_normal.shape, (1, 20))

        # Should be different with foo
        outputs_foo = model.generate(input_ids, foo=True)
        with self.assertRaises(AssertionError):
            self.assertListEqual(outputs_foo.tolist(), outputs_normal.tolist())

        # Assistant model
        assistant = FakeBartCausalLM.from_pretrained(
            "hf-internal-testing/tiny-random-BartForConditionalGeneration"
        ).to(torch_device)

        # If assisted generation passes model_kwargs correctly, should be same as previous
        outputs_assisted = model.generate(
            input_ids,
            foo=True,
            assistant_model=assistant,
        )
        self.assertListEqual(outputs_assisted.tolist(), outputs_foo.tolist())

        # Check that passing encoder_outputs directly also works as expected
        encoder_outputs = model.get_encoder()(input_ids)

        outputs_assisted = model.generate(
            foo=True,
            assistant_model=assistant,
            encoder_outputs=encoder_outputs,
        )
        self.assertListEqual(outputs_assisted.tolist(), outputs_foo.tolist())

    def test_assisted_decoding_num_assistant_tokens_heuristic_schedule(self):
        # This test ensures that the assisted generation num_assistant_tokens 'heuristic' schedule works properly.

        prompt = "Alice and Bob"
        checkpoint = "EleutherAI/pythia-160m-deduped"
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        inputs = tokenizer(prompt, return_tensors="pt")

        model = AutoModelForCausalLM.from_pretrained(checkpoint)

        assistant_model = model
        assistant_model.generation_config.num_assistant_tokens = 5
        assistant_model.generation_config.num_assistant_tokens_schedule = "heuristic"
        generation_kwargs = {
            "eos_token_id": -1,
            "max_new_tokens": 5,
            "do_sample": False,
            "assistant_model": assistant_model,
        }
        model.generate(**inputs, **generation_kwargs)
        # update_candidate_strategy is called only once and therefore, assistant_model.generation_config.num_assistant_tokens should be either 4 or 7
        self.assertTrue(assistant_model.generation_config.num_assistant_tokens in (4, 7))

    def test_assisted_decoding_num_assistant_tokens_heuristic_transient_schedule(self):
        # This test ensures that the assisted generation num_assistant_tokens 'heuristic' schedule works properly.

        prompt = "Alice and Bob"
        checkpoint = "EleutherAI/pythia-160m-deduped"
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        inputs = tokenizer(prompt, return_tensors="pt")

        model = AutoModelForCausalLM.from_pretrained(checkpoint)

        assistant_model = model
        assistant_model.generation_config.num_assistant_tokens = 5
        assistant_model.generation_config.num_assistant_tokens_schedule = "heuristic_transient"
        generation_kwargs = {
            "eos_token_id": -1,
            "max_new_tokens": 5,
            "do_sample": False,
            "assistant_model": assistant_model,
        }
        model.generate(**inputs, **generation_kwargs)
        # update_candidate_strategy is called once but assistant_model.generation_config.num_assistant_tokens should stay 5
        self.assertEqual(assistant_model.generation_config.num_assistant_tokens, 5)

    @slow
    def test_validate_assistant(self):
        # Generate a random sample:
        inputs = np.random.rand(160000)

        # Load a main encoder-decoder model:
        model_id = "openai/whisper-large-v2"
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        model.to(torch_device)

        # process the input:
        features = processor(inputs, return_tensors="pt").to(torch_device)

        # Load an encoder-decoder assistant with same encoder as the main model:
        assistant_distil_model_id = "distil-whisper/distil-large-v2"
        assistant_seq_to_seq = AutoModelForSpeechSeq2Seq.from_pretrained(
            assistant_distil_model_id,
            use_safetensors=True,
        ).to(torch_device)
        self.assertTrue(model.generate(**features, assistant_model=assistant_seq_to_seq).sum())

        # Load its decoder only version:
        assistant_causal_lm = AutoModelForCausalLM.from_pretrained(
            assistant_distil_model_id,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        ).to(torch_device)
        self.assertTrue(model.generate(**features, assistant_model=assistant_causal_lm).sum())

        # Load an encoder-decoder assistant with a different encoder than the main model:
        assistant_distil_model_id = "openai/whisper-tiny"
        assistant_seq_to_seq = AutoModelForSpeechSeq2Seq.from_pretrained(
            assistant_distil_model_id,
            use_safetensors=True,
        ).to(torch_device)
        self.assertTrue(model.generate(**features, assistant_model=assistant_seq_to_seq).sum())

        # Load its decoder only version:
        assistant_causal_lm = AutoModelForCausalLM.from_pretrained(
            assistant_distil_model_id,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        ).to(torch_device)
        # It will raise an error as the encoder of the main and assistant model are not compatible:
        with self.assertRaises(ValueError):
            model.generate(**features, assistant_model=assistant_causal_lm)

        # Load an encoder-decoder model with a different tokenizer than the main model:
        assistant_distil_model_id = "hf-internal-testing/tiny-random-SeamlessM4Tv2ForSpeechToText"
        assistant_seq_to_seq = AutoModelForSpeechSeq2Seq.from_pretrained(
            assistant_distil_model_id,
        ).to(torch_device)
        # This should raise an error as the main and assistant model don't use the same tokenizer:
        with self.assertRaises(ValueError):
            model.generate(**features, assistant_model=assistant_seq_to_seq)

    def test_compare_unprocessed_logit_scores(self):
        # Get unprocessed logit scores back from model generate function.
        # Assert that unprocessed logits from generate() are same as those from modal eval()

        # tell model to generate text and return unprocessed/unwarped logit scores
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        text = "generate yes or no: "
        input_ids = tokenizer([text], return_tensors="pt").input_ids.to(torch_device)

        model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2").to(torch_device)

        with torch.no_grad():
            # Get logits for the next token from fwd pass
            logits_fwd = model(input_ids).logits[:, -1, :][0]

        # Get logits for the next token from generate function
        outputs = model.generate(
            input_ids=input_ids,
            return_dict_in_generate=True,
            output_logits=True,
            max_new_tokens=1,
            do_sample=True,
        )
        logits_gen = outputs.logits[0][0]

        # assert that unprocessed logits from generate() are same as those from modal eval()
        self.assertListEqual(logits_fwd.tolist(), logits_gen.tolist())

    def test_return_unprocessed_logit_scores(self):
        # tell model to generate text and return unprocessed/unwarped logit scores
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        text = "generate yes or no: "
        input_ids = tokenizer([text], return_tensors="pt").input_ids.to(torch_device)
        model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2").to(torch_device)

        outputs = model.generate(
            input_ids=input_ids, return_dict_in_generate=True, output_logits=True, max_new_tokens=3
        )

        # perform dummy check if unpreprocessed logits make sense.
        # do preselection on high probabilities; find scores of y and n tokens
        probs_all = torch.nn.functional.softmax(outputs.logits[2][0], dim=-1)
        indices = torch.argwhere(probs_all > 0.001)
        indices = indices[:, -1]
        tokens_max = tokenizer.batch_decode(indices, skip_special_tokens=True)
        probs_max = probs_all[probs_all > 0.001]

        self.assertTrue(len(indices) >= 2)
        next_token_dict = {str(t): p for t, p in zip(tokens_max, probs_max)}
        self.assertTrue("n" in next_token_dict)
        self.assertTrue("y" in next_token_dict)
        y_prob = next_token_dict["y"]
        n_prob = next_token_dict["n"]

        self.assertTrue(y_prob > 0.001 and n_prob > 0.001)
        self.assertTrue(y_prob <= 1.0 and n_prob <= 1.0)

    @slow
    @require_torch_multi_gpu
    def test_assisted_decoding_in_different_gpu(self):
        # PT-only test: TF doesn't support assisted decoding yet.
        model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-MistralForCausalLM").to("cuda:0")
        assistant = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-MistralForCausalLM").to(
            "cuda:1"
        )
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-MistralForCausalLM")
        model.config.pad_token_id = tokenizer.eos_token_id
        assistant.config.pad_token_id = tokenizer.eos_token_id

        text = "Hello world"
        tokenized_inputs = tokenizer([text], return_tensors="pt")
        input_ids = tokenized_inputs.input_ids.to(torch_device)
        input_length = input_ids.shape[-1]

        out = model.generate(
            input_ids,
            assistant_model=assistant,
            max_new_tokens=20,
        )
        self.assertTrue(input_length <= out.shape[-1] <= input_length + 20)

    @slow
    @require_torch_gpu
    def test_assisted_decoding_model_in_gpu_assistant_in_cpu(self):
        # PT-only test: TF doesn't support assisted decoding yet.
        model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-MistralForCausalLM").to("cuda")
        assistant = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-MistralForCausalLM").to(
            "cpu"
        )
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-MistralForCausalLM")
        model.config.pad_token_id = tokenizer.eos_token_id
        assistant.config.pad_token_id = tokenizer.eos_token_id

        text = "Hello world"
        tokenized_inputs = tokenizer([text], return_tensors="pt")
        input_ids = tokenized_inputs.input_ids.to(torch_device)
        input_length = input_ids.shape[-1]

        out = model.generate(
            input_ids,
            assistant_model=assistant,
            max_new_tokens=20,
        )
        self.assertTrue(input_length <= out.shape[-1] <= input_length + 20)

    def test_special_tokens_fall_back_to_model_default(self):
        # PT-only test: TF doesn't support assisted decoding yet.
        model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-MistralForCausalLM").to(
            torch_device
        )
        test_bos_id = 50

        # Sanity-check: the model has a BOS token set, and the first generated token is a BOS token
        gen_output = model.generate()
        self.assertTrue(model.generation_config.bos_token_id is not None)
        self.assertTrue(model.generation_config.bos_token_id == gen_output[0, 0])

        # If we pass a generation config **with** a BOS token, `generate` will use it
        generation_config = GenerationConfig(bos_token_id=test_bos_id)
        gen_output = model.generate(generation_config=generation_config)
        self.assertFalse(model.generation_config.bos_token_id == gen_output[0, 0])
        self.assertTrue(generation_config.bos_token_id == gen_output[0, 0])
        self.assertTrue(test_bos_id == gen_output[0, 0])

        # If we pass a generation config **without** a BOS token, `generate` will fetch the BOS token from
        # `model.generation_config`
        generation_config = GenerationConfig(bos_token_id=None)
        gen_output = model.generate(generation_config=generation_config)
        self.assertTrue(model.generation_config.bos_token_id == gen_output[0, 0])
        self.assertFalse(test_bos_id == gen_output[0, 0])
        self.assertTrue(generation_config.bos_token_id is None)

        # Changing `model.generation_config` will affect fallback behavior
        model.generation_config.bos_token_id = test_bos_id
        gen_output = model.generate(generation_config=generation_config)
        self.assertTrue(model.generation_config.bos_token_id == gen_output[0, 0])
        self.assertTrue(test_bos_id == gen_output[0, 0])
        self.assertTrue(generation_config.bos_token_id is None)

    def test_speculative_decoding_equals_regular_decoding(self):
        draft_name = "double7/vicuna-68m"
        target_name = "Qwen/Qwen2-0.5B-Instruct"

        draft_model = AutoModelForCausalLM.from_pretrained(draft_name)
        target_model = AutoModelForCausalLM.from_pretrained(target_name)

        assistant_tokenizer = AutoTokenizer.from_pretrained(draft_name)
        target_tokenizer = AutoTokenizer.from_pretrained(target_name)

        prompt_size = torch.randint(low=20, high=100, size=(1,))
        max_new_tokens = torch.randint(low=10, high=50, size=(1,))
        input_ids = (torch.rand(1, prompt_size[0]) * 100).to(int) + 50

        max_new_tokens_item = max_new_tokens[0].item()
        expected_out = target_model.generate(input_ids, do_sample=False, max_new_tokens=max_new_tokens_item)
        predicted_out = target_model.generate(
            input_ids,
            do_sample=False,
            max_new_tokens=max_new_tokens_item,
            assistant_model=draft_model,
            tokenizer=target_tokenizer,
            assistant_tokenizer=assistant_tokenizer,
        )

        self.assertEqual(expected_out.shape, predicted_out.shape)
        self.assertTrue((expected_out == predicted_out).all().item())

    @pytest.mark.generate
    @require_torch_multi_gpu
    def test_generate_with_static_cache_multi_gpu(self):
        """
        Tests if the static cache has been set correctly and if generate works correctly when we are using multi-gpus.
        """
        # need to split manually as auto doesn't work well with unbalanced model
        device_map = {"model.embed_tokens": 0, "model.layers.0": 0, "model.layers.1": 1, "model.norm": 1, "lm_head": 0}
        model = AutoModelForCausalLM.from_pretrained(
            "hf-internal-testing/tiny-random-MistralForCausalLM", device_map=device_map
        )
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-MistralForCausalLM")

        text = "Hello world"
        tokenized_inputs = tokenizer([text], return_tensors="pt")
        input_ids = tokenized_inputs.input_ids.to(torch_device)

        generation_kwargs = {
            "max_new_tokens": 20,
            "cache_implementation": "static",
            "return_dict_in_generate": True,  # Required to return `past_key_values`
        }

        results = model.generate(input_ids, **generation_kwargs)
        self.assertTrue(isinstance(results.past_key_values, StaticCache))

        # check device of each layer
        key_cache_0 = results.past_key_values.key_cache[0]
        value_cache_0 = results.past_key_values.value_cache[0]
        self.assertTrue(key_cache_0.device == value_cache_0.device == torch.device(0))

        key_cache_1 = results.past_key_values.key_cache[1]
        value_cache_1 = results.past_key_values.value_cache[1]
        self.assertTrue(key_cache_1.device == value_cache_1.device == torch.device(1))

    @pytest.mark.generate
    @require_torch_multi_gpu
    def test_init_static_cache_multi_gpu(self):
        """
        Tests if the static cache has been set correctly when we initialize it manually in a multi-gpu setup.
        """
        # need to split manually as auto doesn't work well with unbalanced model
        device_map = {"model.embed_tokens": 0, "model.layers.0": 0, "model.layers.1": 1, "model.norm": 1, "lm_head": 0}
        model = AutoModelForCausalLM.from_pretrained(
            "hf-internal-testing/tiny-random-MistralForCausalLM", device_map=device_map
        )
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-MistralForCausalLM")

        text = "Hello world"
        tokenized_inputs = tokenizer([text], return_tensors="pt")
        input_ids = tokenized_inputs.input_ids.to(torch_device)

        generation_kwargs = {
            "max_new_tokens": 20,
            "return_dict_in_generate": True,  # Required to return `past_key_values`
        }

        # TODO: We need to raise a warning in case the cache is not set correctly
        # with self.assertRaisesRegex(ValueError, "If you are manually initializing the cache"):
        #     past_key_values = StaticCache(
        #         config=model.config, batch_size=1, max_cache_len=30, device=torch_device, dtype=model.dtype
        #     )
        #     results = model.generate(input_ids, past_key_values=past_key_values, **generation_kwargs)

        # deduced from the device_map : layer 0 on device 0 and layer 1 on device 1
        layer_device_map = {0: 0, 1: 1}
        past_key_values = StaticCache(
            config=model.config,
            batch_size=1,
            max_cache_len=30,
            device=torch_device,
            dtype=model.dtype,
            layer_device_map=layer_device_map,
        )
        results = model.generate(input_ids, past_key_values=past_key_values, **generation_kwargs)

        # check device of each layer
        key_cache_0 = results.past_key_values.key_cache[0]
        value_cache_0 = results.past_key_values.value_cache[0]
        self.assertTrue(key_cache_0.device == value_cache_0.device == torch.device(0))

        key_cache_1 = results.past_key_values.key_cache[1]
        value_cache_1 = results.past_key_values.value_cache[1]
        self.assertTrue(key_cache_1.device == value_cache_1.device == torch.device(1))

    @slow
    def test_padding_input_contrastive_search_gpt2(self):
        # Load the pre-trained GPT-2 model and tokenizer
        model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
        model.to(torch_device)
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2", clean_up_tokenization_spaces=True)

        # Set the tokenizer to left-pad the sequences
        tokenizer.padding_side = "left"

        # Define the PAD token as the EOS token
        tokenizer.pad_token = tokenizer.eos_token
        model.generation_config.pad_token_id = model.generation_config.eos_token_id

        # Define the input prompt
        prompt_text = "The whispered legends of the haunted mansion spoke"

        # Tokenize the input prompt
        encoded_prompt = tokenizer(prompt_text, return_tensors="pt", padding=True)
        input_ids = encoded_prompt.input_ids.to(torch_device)
        attention_mask = encoded_prompt.attention_mask.to(torch_device)

        # Define the contrastive search params
        penalty_alpha = 0.6
        top_k = 4

        # Define the padding length to add to the input IDs and attention mask
        padding_length = 10

        # Generate text without padding
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            penalty_alpha=penalty_alpha,
            top_k=top_k,
            max_new_tokens=64,
        )
        generated_text_no_padding = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Pad the input IDs and attention mask on the left
        padded_input_ids = F.pad(
            input_ids, (padding_length, 0), "constant", value=model.generation_config.pad_token_id
        )
        padded_attention_mask = F.pad(attention_mask, (padding_length, 0), "constant", value=0)

        # Generate text with padded inputs
        outputs_with_padding = model.generate(
            input_ids=padded_input_ids,
            attention_mask=padded_attention_mask,
            do_sample=False,
            penalty_alpha=penalty_alpha,
            top_k=top_k,
            max_new_tokens=64,
        )
        generated_text_with_padding = tokenizer.decode(outputs_with_padding[0], skip_special_tokens=True)

        # Assert that the generated texts are identical for padded and non-padded inputs
        self.assertEqual(generated_text_no_padding, generated_text_with_padding)
        self.assertEqual(
            generated_text_with_padding,
            'The whispered legends of the haunted mansion spoke of the "souls of the dead" who were "falling '
            'out of the sky" and "falling into the sea."\n\nThe ghostly apparitions were said to have been '
            'created by the spirits of the dead, who were "falling out of the sky" and "falling into the sea',
        )

    @slow
    def test_padding_input_contrastive_search_t5(self):
        # Load the pre-trained T5 model and tokenizer
        model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")
        model.to(torch_device)
        tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small", clean_up_tokenization_spaces=True)

        # Define the input prompt
        prompt_text = "translate English to German: I need to finish this task before the end of the day."

        # Tokenize the input prompt
        encoded_prompt = tokenizer(prompt_text, return_tensors="pt")
        input_ids = encoded_prompt.input_ids.to(torch_device)
        attention_mask = encoded_prompt.attention_mask.to(torch_device)

        # Define the decoder prompt
        decoder_prompt_text = "Ich muss diese Aufgabe"
        encoded_decoder_prompt = tokenizer(decoder_prompt_text, add_special_tokens=False, return_tensors="pt")
        decoder_input_ids = encoded_decoder_prompt.input_ids.to(torch_device)
        decoder_attention_mask = encoded_decoder_prompt.attention_mask.to(torch_device)

        # Define the contrastive search params
        penalty_alpha = 0.6
        top_k = 4

        # Generate text without padding
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            do_sample=False,
            penalty_alpha=penalty_alpha,
            top_k=top_k,
            max_new_tokens=64,
        )
        generated_text_no_padding = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Define the padding length to add to the input IDs and attention mask
        padding_length = 10

        # Pad the decoder input IDs and attention mask on the left
        padded_decoder_input_ids = F.pad(
            decoder_input_ids, (padding_length, 0), "constant", value=model.generation_config.pad_token_id
        )
        padded_decoder_attention_mask = F.pad(decoder_attention_mask, (padding_length, 0), "constant", value=0)
        # Since the decoder_start_token_id is the same as the pad_token_id,
        # the last padded token represents the decoder start token.
        # Set the attention mask for the decoder_start_token_id to True (1).
        padded_decoder_attention_mask[:, padding_length - 1] = 1
        # Generate text with padded inputs
        outputs_with_padding = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=padded_decoder_input_ids,
            decoder_attention_mask=padded_decoder_attention_mask,
            do_sample=False,
            penalty_alpha=penalty_alpha,
            top_k=top_k,
            max_new_tokens=64,
        )
        generated_text_with_padding = tokenizer.decode(outputs_with_padding[0], skip_special_tokens=True)

        # Assert that the generated texts are identical for padded and non-padded inputs
        self.assertEqual(generated_text_no_padding, generated_text_with_padding)
        self.assertEqual(generated_text_no_padding, "Ich muss diese Aufgabe vor Ende des Tages beenden.")

    def test_prepare_inputs_for_generation_decoder_llm(self):
        """Tests GenerationMixin.prepare_inputs_for_generation against expected usage with decoder-only llms."""

        config = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-LlamaForCausalLM")
        model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-LlamaForCausalLM")
        model = model.to(torch_device)

        # 1. Sanity check: the model's `prepare_inputs_for_generation` comes from `GenerationMixin`
        self.assertTrue("GenerationMixin" in str(model.prepare_inputs_for_generation))

        # 2. If we pass input ids by themselves, we should get back the same input ids
        input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]]).to(torch_device)
        model_inputs = model.prepare_inputs_for_generation(input_ids)
        self.assertTrue(torch.all(model_inputs["input_ids"] == input_ids))

        # 3. If we pass the attention mask too, we will get back the attention mask and position ids built from it
        attention_mask = torch.tensor([[1, 1, 1], [1, 1, 1]]).to(torch_device)
        model_inputs = model.prepare_inputs_for_generation(input_ids, attention_mask=attention_mask)
        self.assertTrue(torch.all(model_inputs["attention_mask"] == attention_mask))
        self.assertTrue(model_inputs["position_ids"].shape == input_ids.shape)

        # 4. `use_cache` (and other kwargs) are forwarded
        self.assertFalse("use_cache" in model_inputs)  # From the previous input, there is no `use_cache`
        model_inputs = model.prepare_inputs_for_generation(input_ids, use_cache=True, foo="bar")
        self.assertTrue(model_inputs["use_cache"] is True)
        self.assertTrue(model_inputs["foo"] == "bar")

        # 5. When we pass a cache, we discard data related to already seen tokens in some tensors. We are now also
        # forced to pass a correctly prepared `cache_positions` to slice the data accordingly.
        init_input_ids = input_ids[:, :2]
        dynamic_cache = DynamicCache()
        dynamic_cache = model(init_input_ids, past_key_values=dynamic_cache).past_key_values
        with self.assertRaises(AttributeError):  # past_key_values + no cache_position -> exception
            model_inputs = model.prepare_inputs_for_generation(input_ids, past_key_values=dynamic_cache)

        cache_position = torch.arange(input_ids.shape[-1], dtype=torch.long).to(torch_device)
        cache_position = cache_position[dynamic_cache.get_seq_length() :]
        model_inputs = model.prepare_inputs_for_generation(
            input_ids, past_key_values=dynamic_cache, cache_position=cache_position, attention_mask=attention_mask
        )
        self.assertTrue("past_key_values" in model_inputs)
        self.assertTrue(torch.all(model_inputs["cache_position"] == cache_position))
        self.assertTrue(model_inputs["input_ids"].shape[-1] == 1)  # 1 = 3 fed tokens - 2 tokens in the cache
        self.assertTrue(model_inputs["position_ids"].shape[-1] == 1)
        self.assertTrue(model_inputs["attention_mask"].shape[-1] == 3)  # we still need the full attention mask!

        # 6. If we pass a `static_cache`, the attention mask will be prepared as a static shape 4D mask
        max_cache_len = 10
        batch_size = 2
        query_length = input_ids.shape[-1] - init_input_ids.shape[-1]
        static_cache = StaticCache(
            config=config, batch_size=batch_size, max_cache_len=max_cache_len, device=torch_device, dtype=torch.float32
        )
        static_cache = model(init_input_ids, past_key_values=static_cache).past_key_values
        model_inputs = model.prepare_inputs_for_generation(
            input_ids, past_key_values=static_cache, cache_position=cache_position, attention_mask=attention_mask
        )
        self.assertTrue("past_key_values" in model_inputs)
        self.assertTrue(list(model_inputs["attention_mask"].shape) == [batch_size, 1, query_length, max_cache_len])

        # 7. We can also pass `inputs_embeds` as the embedded prompt. Because `generate` will append its result to
        # `input_ids` and the models will only accept one of the two inputs (`input_ids` or `inputs_embeds`), we
        # a) must use the cache b) must expect `input_ids` after the prompt is processed
        init_inputs_embeds = model.get_input_embeddings()(init_input_ids)
        init_cache_positions = torch.arange(init_input_ids.shape[-1], dtype=torch.long).to(torch_device)
        empty_cache = DynamicCache()

        # Prompt processing
        model_inputs = model.prepare_inputs_for_generation(
            init_input_ids,
            past_key_values=empty_cache,
            inputs_embeds=init_inputs_embeds,
            cache_position=init_cache_positions,
        )
        self.assertTrue(model_inputs["input_ids"] is None)
        self.assertTrue(model_inputs["inputs_embeds"] is not None)

        # After prompt processing
        model_inputs = model.prepare_inputs_for_generation(
            input_ids, past_key_values=dynamic_cache, inputs_embeds=init_inputs_embeds, cache_position=cache_position
        )
        self.assertTrue(model_inputs["input_ids"] is not None)
        self.assertTrue(model_inputs["inputs_embeds"] is None)

    def test_prepare_inputs_for_generation_encoder_decoder_llm(self):
        """
        Same as `test_prepare_inputs_for_generation_decoder_llm` but for encoder-decoder models. Main difference: we
        should look for `decoder_input_ids`, instead of `input_ids`.
        """
        model = AutoModelForSeq2SeqLM.from_pretrained("hf-internal-testing/tiny-random-t5")
        model = model.to(torch_device)

        # 1. Sanity check: the model's `prepare_inputs_for_generation` comes from `GenerationMixin`
        self.assertTrue("GenerationMixin" in str(model.prepare_inputs_for_generation))

        # 2. If we pass input ids by themselves, we should get back the same input ids -- with the encoder-decoder key
        decoder_input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]]).to(torch_device)
        model_inputs = model.prepare_inputs_for_generation(decoder_input_ids)
        self.assertTrue(torch.all(model_inputs["decoder_input_ids"] == decoder_input_ids))

        # 3. If we pass the attention mask too, we will get back the attention mask. Encoder-decoder models usually
        # don't use `position_ids`
        decoder_attention_mask = torch.tensor([[1, 1, 1], [1, 1, 1]]).to(torch_device)
        model_inputs = model.prepare_inputs_for_generation(
            decoder_input_ids, decoder_attention_mask=decoder_attention_mask
        )
        self.assertTrue(torch.all(model_inputs["decoder_attention_mask"] == decoder_attention_mask))
        self.assertTrue("position_ids" not in model_inputs)

        # 4. `use_cache` (and other kwargs, like the encoder outputs) are forwarded
        self.assertFalse("use_cache" in model_inputs)  # From the previous input, there is no `use_cache`
        model_inputs = model.prepare_inputs_for_generation(decoder_input_ids, use_cache=True, encoder_outputs="foo")
        self.assertTrue(model_inputs["use_cache"] is True)
        self.assertTrue(model_inputs["encoder_outputs"] == "foo")
        # See the decoder-only test for more corner cases. The code is the same, so we don't repeat it here.

    def test_generate_compile_fullgraph_tiny(self):
        """
        Tests that we can call end-to-end generation with a tiny model (i.e. doesn't crash)
        NOTE: this test is quite slow (~20s on a consumer desktop), but it is important that we keep it as part of the
        non-slow tests to prevent regressions!
        """
        model = AutoModelForCausalLM.from_pretrained(
            "hf-internal-testing/tiny-random-LlamaForCausalLM", torch_dtype=torch.bfloat16, device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-LlamaForCausalLM")

        # compile generate
        compiled_generate = torch.compile(model.generate, fullgraph=True, mode="reduce-overhead")

        # compiled generate does NOT accept parameterization except a) model inputs b) a generation config
        generation_config = copy.deepcopy(model.generation_config)
        generation_config.pad_token_id = model.config.eos_token_id

        model_inputs = tokenizer(["Write a poem about the market crashing in summer"], return_tensors="pt")
        model_inputs = model_inputs.to(model.device)
        gen_out = compiled_generate(**model_inputs, generation_config=generation_config)
        self.assertTrue(gen_out.shape[1] > model_inputs["input_ids"].shape[1])  # some text was generated

    def test_assisted_generation_early_exit(self):
        """
        Tests that assisted generation with early exit works as expected. Under the hood, this has complex cache
        manipulation, which will cause the test to fail if something goes wrong there.
        """
        expected_output = "Alice and Bob are playing a game of poker. Alice has a pair of 8s and Bob has a pair"

        prompt = "Alice and Bob"
        checkpoint = "facebook/layerskip-llama3.2-1B"

        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        inputs = tokenizer(prompt, return_tensors="pt").to(torch_device)

        model = AutoModelForCausalLM.from_pretrained(checkpoint).to(torch_device)
        original_outputs = model.generate(**inputs, do_sample=False, max_new_tokens=20)
        original_decoded = tokenizer.batch_decode(original_outputs, skip_special_tokens=True)
        self.assertEqual(original_decoded, [expected_output])

        outputs_assisted = model.generate(**inputs, assistant_early_exit=4, do_sample=False, max_new_tokens=20)
        decoded_assisted = tokenizer.batch_decode(outputs_assisted, skip_special_tokens=True)
        self.assertEqual(decoded_assisted, [expected_output])


@require_torch
class TokenHealingTestCase(unittest.TestCase):
    @parameterized.expand(
        [
            ("url", 'The link is <a href="http:', 'The link is <a href="http://'),
            # aggressive_healing: "http" shouldn't be replaced with "https"
            ("aggressive_healing", 'The link is <a href="http', 'The link is <a href="http'),
            ("trailing_whitespace", "I read a book about ", "I read a book about"),
            ("nothing_to_heal", "I read a book about", "I read a book about"),
            ("single_token", "I", "I"),
            ("empty_prompt", "", ""),
        ]
    )
    def test_prompts(self, name, input, expected):
        model_name_or_path = "distilbert/distilgpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        completion_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map="auto",
            trust_remote_code=False,
            revision="main",
            use_cache=True,
        )

        """
        tokenizer.pad_token value can be empty but it is required in the latter codes
        so assigned it here with eos_token
		"""
        tokenizer.pad_token = tokenizer.eos_token

        input_ids = tokenizer(input, return_tensors="pt").input_ids.to(completion_model.device)

        healed_ids = completion_model.heal_tokens(input_ids, tokenizer=tokenizer)
        predicted = tokenizer.decode(healed_ids[0], skip_special_tokens=True)

        self.assertEqual(predicted, expected)

    def test_generate_from_inputs_embeds_with_bos_token_id_is_none(self):
        article = "Today a dragon flew over Paris."
        model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2").to(torch_device)
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        input_ids = tokenizer(article, return_tensors="pt").input_ids.to(torch_device)
        inputs_embeds = model.get_input_embeddings()(input_ids)

        model.generate(inputs_embeds=inputs_embeds, max_length=20, bos_token_id=None)

        # bos_token_id is required when no input ids nor inputs_embeds is passed
        with self.assertRaises(ValueError):
            model.generate(max_length=20, bos_token_id=None)


class TestAssistedCandidateGeneratorDifferentTokenizers(unittest.TestCase):
    def test_no_intersection(self):
        prompt = np.array([[1, 2, 3]])
        prompt_plus_new_tokens = np.array([[4, 5, 6]])
        result = AssistedCandidateGeneratorDifferentTokenizers._get_tokens_diag(prompt, prompt_plus_new_tokens)
        self.assertEqual(result, (None, None, None))

    def test_complete_overlap(self):
        prompt = np.array([[1, 2, 3]])
        prompt_plus_new_tokens = np.array([[1, 2, 3, 4, 5]])
        discrep_length, new_tokens_only, discrep_only = AssistedCandidateGeneratorDifferentTokenizers._get_tokens_diag(
            prompt, prompt_plus_new_tokens
        )
        self.assertEqual(discrep_length, 0)
        np.testing.assert_array_equal(new_tokens_only, np.array([[4, 5]]))
        np.testing.assert_array_equal(discrep_only, np.array([[]]))

    def test_partial_overlap(self):
        prompt = np.array([[1, 2, 3]])
        prompt_plus_new_tokens = np.array([[2, 3, 4, 5]])
        discrep_length, new_tokens_only, discrep_only = AssistedCandidateGeneratorDifferentTokenizers._get_tokens_diag(
            prompt, prompt_plus_new_tokens
        )
        self.assertEqual(discrep_length, 0)
        np.testing.assert_array_equal(new_tokens_only, np.array([[4, 5]]))
        np.testing.assert_array_equal(discrep_only, np.array([[]]))

    def test_no_new_tokens(self):
        prompt = np.array([[1, 2, 3]])
        prompt_plus_new_tokens = np.array([[1, 2, 3]])
        discrep_length, new_tokens_only, discrep_only = AssistedCandidateGeneratorDifferentTokenizers._get_tokens_diag(
            prompt, prompt_plus_new_tokens
        )
        self.assertEqual(discrep_length, 0)
        np.testing.assert_array_equal(new_tokens_only, np.array([[]]))
        np.testing.assert_array_equal(discrep_only, np.array([[]]))


class TestAssistedCandidateGeneratorUpdateStrategy(unittest.TestCase):
    def setUp(self):
        checkpoint = "EleutherAI/pythia-160m-deduped"
        self.assistant_model = AutoModelForCausalLM.from_pretrained(checkpoint)
        self.assistant_model.generation_config.assistant_confidence_threshold = 0.4
        self.model_kwargs = {}
        self.input_ids = torch.randint(1, 10, (1, 9))
        self.candidate_generator = AssistedCandidateGenerator(
            input_ids=self.input_ids,
            assistant_model=self.assistant_model,
            generation_config=self.assistant_model.generation_config,
            model_kwargs=self.model_kwargs,
        )
        self.candidate_generator.probs = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        self.original_probs = self.candidate_generator.probs
        self.original_threshold = self.assistant_model.generation_config.assistant_confidence_threshold

    def assert_no_sklearn(self):
        with patch("transformers.utils.import_utils._sklearn_available", False):
            self.candidate_generator.update_candidate_strategy(self.input_ids, None, self.num_matches)
            self.assertEqual(self.candidate_generator.matches, self.original_matches)
            self.assertEqual(self.candidate_generator.probs, self.original_probs)
            self.assertEqual(
                self.assistant_model.generation_config.assistant_confidence_threshold, self.original_threshold
            )

    @parameterized.expand([(is_sklearn_available(),), (False,)])
    def test_update_candidate_strategy_no_matches_short(self, sklearn_available):
        print("test_update_candidate_strategy_no_matches_short")
        self.original_matches = []
        self.candidate_generator.matches = self.original_matches
        self.num_matches = 0

        if sklearn_available:
            self.candidate_generator.update_candidate_strategy(self.input_ids, None, self.num_matches)
            self.assertEqual(self.candidate_generator.matches, [0])
            self.assertEqual(self.candidate_generator.probs, [0.9])
            self.assertEqual(self.assistant_model.generation_config.assistant_confidence_threshold, 0.4)
        else:
            self.assert_no_sklearn()

    @parameterized.expand([(is_sklearn_available(),), (False,)])
    def test_update_candidate_strategy_with_mix_matches_3(self, sklearn_available):
        self.original_matches = [1, 0, 1, 0, 1]
        self.candidate_generator.matches = self.original_matches
        self.num_matches = 3
        if sklearn_available:
            self.candidate_generator.update_candidate_strategy(self.input_ids, None, self.num_matches)
            self.assertEqual(self.candidate_generator.matches, [1, 0, 1, 0, 1, 1, 1, 1, 0])
            self.assertEqual(self.candidate_generator.probs, [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
            self.assertEqual(self.assistant_model.generation_config.assistant_confidence_threshold, 0.2)
        else:
            self.assert_no_sklearn()

    @parameterized.expand([(is_sklearn_available(),), (False,)])
    def test_update_candidate_strategy_with_matches_4(self, sklearn_available):
        self.original_matches = [1, 1, 1, 1, 1]
        self.candidate_generator.matches = self.original_matches
        self.num_matches = 4
        if sklearn_available:
            self.candidate_generator.update_candidate_strategy(self.input_ids, None, self.num_matches)
            self.assertEqual(self.candidate_generator.matches, [1, 1, 1, 1, 1, 1, 1, 1, 1])
            self.assertEqual(self.candidate_generator.probs, [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
            self.assertEqual(self.assistant_model.generation_config.assistant_confidence_threshold, 0.4)
        else:
            self.assert_no_sklearn()

    @parameterized.expand([(is_sklearn_available(),), (False,)])
    def test_update_candidate_strategy_with_matches_3(self, sklearn_available):
        self.original_matches = [1, 1, 1, 1, 1]
        self.candidate_generator.matches = self.original_matches
        self.num_matches = 3
        if sklearn_available:
            self.candidate_generator.update_candidate_strategy(self.input_ids, None, self.num_matches)
            self.assertEqual(self.candidate_generator.matches, [1, 1, 1, 1, 1, 1, 1, 1, 0])
            self.assertEqual(self.candidate_generator.probs, [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
            self.assertEqual(self.assistant_model.generation_config.assistant_confidence_threshold, 0.2)
        else:
            self.assert_no_sklearn()

    @parameterized.expand([(is_sklearn_available(),), (False,)])
    def test_update_candidate_strategy_with_matches_2(self, sklearn_available):
        self.original_matches = [1, 1, 1, 1, 1]
        self.candidate_generator.matches = self.original_matches
        self.num_matches = 2
        if sklearn_available:
            self.candidate_generator.update_candidate_strategy(self.input_ids, None, self.num_matches)
            self.assertEqual(self.candidate_generator.matches, [1, 1, 1, 1, 1, 1, 1, 0])
            self.assertEqual(self.candidate_generator.probs, [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
            self.assertEqual(self.assistant_model.generation_config.assistant_confidence_threshold, 0.3)
        else:
            self.assert_no_sklearn()

    @parameterized.expand([(is_sklearn_available(),), (False,)])
    def test_update_candidate_strategy_with_matches_1(self, sklearn_available):
        self.original_matches = [1, 1, 1, 1, 1]
        self.candidate_generator.matches = self.original_matches
        self.num_matches = 1
        if sklearn_available:
            self.candidate_generator.update_candidate_strategy(self.input_ids, None, self.num_matches)
            self.assertEqual(self.candidate_generator.matches, [1, 1, 1, 1, 1, 1, 0])
            self.assertEqual(self.candidate_generator.probs, [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
            self.assertEqual(self.assistant_model.generation_config.assistant_confidence_threshold, 0.4)
        else:
            self.assert_no_sklearn()
