# coding=utf-8
# Copyright 2021 The Google AI Flax Team Authors, and The HuggingFace Inc. team.
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
import inspect
import warnings
from functools import partial
from typing import Any, Dict, Optional, Union

import flax
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from ..models.auto import (
    FLAX_MODEL_FOR_CAUSAL_LM_MAPPING,
    FLAX_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
    FLAX_MODEL_FOR_VISION_2_SEQ_MAPPING,
)
from ..utils import ModelOutput, logging
from .configuration_utils import GenerationConfig
from .flax_logits_process import (
    FlaxForcedBOSTokenLogitsProcessor,
    FlaxForcedEOSTokenLogitsProcessor,
    FlaxForceTokensLogitsProcessor,
    FlaxLogitsProcessorList,
    FlaxMinLengthLogitsProcessor,
    FlaxNoRepeatNGramLogitsProcessor,
    FlaxSuppressTokensAtBeginLogitsProcessor,
    FlaxSuppressTokensLogitsProcessor,
    FlaxTemperatureLogitsWarper,
    FlaxTopKLogitsWarper,
    FlaxTopPLogitsWarper,
)


logger = logging.get_logger(__name__)


@flax.struct.dataclass
class FlaxGreedySearchOutput(ModelOutput):
    """
    Flax Base class for outputs of decoder-only generation models using greedy search.


    Args:
        sequences (`jnp.ndarray` of shape `(batch_size, max_length)`):
            The generated sequences.
    """

    sequences: jnp.ndarray = None


@flax.struct.dataclass
class FlaxSampleOutput(ModelOutput):
    """
    Flax Base class for outputs of decoder-only generation models using sampling.


    Args:
        sequences (`jnp.ndarray` of shape `(batch_size, max_length)`):
            The generated sequences.
    """

    sequences: jnp.ndarray = None


@flax.struct.dataclass
class FlaxBeamSearchOutput(ModelOutput):
    """
    Flax Base class for outputs of decoder-only generation models using greedy search.


    Args:
        sequences (`jnp.ndarray` of shape `(batch_size, max_length)`):
            The generated sequences.
        scores (`jnp.ndarray` of shape `(batch_size,)`):
            The scores (log probabilities) of the generated sequences.
    """

    sequences: jnp.ndarray = None
    scores: jnp.ndarray = None


@flax.struct.dataclass
class GreedyState:
    cur_len: jnp.ndarray
    sequences: jnp.ndarray
    running_token: jnp.ndarray
    is_sent_finished: jnp.ndarray
    model_kwargs: Dict[str, jnp.ndarray]


@flax.struct.dataclass
class SampleState:
    cur_len: jnp.ndarray
    sequences: jnp.ndarray
    running_token: jnp.ndarray
    is_sent_finished: jnp.ndarray
    prng_key: jnp.ndarray
    model_kwargs: Dict[str, jnp.ndarray]


@flax.struct.dataclass
class BeamSearchState:
    cur_len: jnp.ndarray
    running_sequences: jnp.ndarray
    running_scores: jnp.ndarray
    sequences: jnp.ndarray
    scores: jnp.ndarray
    is_sent_finished: jnp.ndarray
    model_kwargs: Dict[str, jnp.ndarray]


class FlaxGenerationMixin:
    """
    A class containing all functions for auto-regressive text generation, to be used as a mixin in
    [`FlaxPreTrainedModel`].

    The class exposes [`~generation.FlaxGenerationMixin.generate`], which can be used for:
            - *greedy decoding* by calling [`~generation.FlaxGenerationMixin._greedy_search`] if `num_beams=1` and
              `do_sample=False`
            - *multinomial sampling* by calling [`~generation.FlaxGenerationMixin._sample`] if `num_beams=1` and
              `do_sample=True`
            - *beam-search decoding* by calling [`~generation.FlaxGenerationMixin._beam_search`] if `num_beams>1` and
              `do_sample=False`

    You do not need to call any of the above methods directly. Pass custom parameter values to 'generate' instead. To
    learn more about decoding strategies refer to the [text generation strategies guide](../generation_strategies).
    """

    def prepare_inputs_for_generation(self, *args, **kwargs):
        raise NotImplementedError(
            "A model class needs to define a `prepare_inputs_for_generation` method in order to use `generate`."
        )

    @staticmethod
    def _run_loop_in_debug(cond_fn, body_fn, init_state):
        """
        Run generation in untraced mode. This should only be used for debugging purposes.
        """
        state = init_state
        while cond_fn(state):
            state = body_fn(state)
        return state

    def _prepare_encoder_decoder_kwargs_for_generation(self, input_ids, params, model_kwargs):
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not (argument.startswith("decoder_") or argument.startswith("cross_attn"))
        }
        model_kwargs["encoder_outputs"] = self.encode(input_ids, params=params, return_dict=True, **encoder_kwargs)
        return model_kwargs

    def _prepare_decoder_input_ids_for_generation(
        self,
        batch_size: int,
        decoder_start_token_id: int = None,
        bos_token_id: int = None,
        model_kwargs: Optional[Dict[str, jnp.ndarray]] = None,
    ) -> jnp.ndarray:
        if model_kwargs is not None and "decoder_input_ids" in model_kwargs:
            # Only use this arg if not None, otherwise just remove from model_kwargs
            decoder_input_ids = model_kwargs.pop("decoder_input_ids")
            if decoder_input_ids is not None:
                return decoder_input_ids
        decoder_start_token_id = self._get_decoder_start_token_id(decoder_start_token_id, bos_token_id)
        return jnp.array(decoder_start_token_id, dtype="i4").reshape(1, -1).repeat(batch_size, axis=0)

    def _get_decoder_start_token_id(self, decoder_start_token_id: int = None, bos_token_id: int = None) -> int:
        # retrieve decoder_start_token_id for encoder-decoder models
        # fall back to bos_token_id if necessary
        decoder_start_token_id = (
            decoder_start_token_id
            if decoder_start_token_id is not None
            else self.generation_config.decoder_start_token_id
        )
        bos_token_id = bos_token_id if bos_token_id is not None else self.generation_config.bos_token_id
        if decoder_start_token_id is not None:
            return decoder_start_token_id
        elif (
            hasattr(self.config, "decoder")
            and hasattr(self.config.decoder, "decoder_start_token_id")
            and self.config.decoder.decoder_start_token_id is not None
        ):
            return self.config.decoder.decoder_start_token_id
        elif bos_token_id is not None:
            return bos_token_id
        elif (
            hasattr(self.config, "decoder")
            and hasattr(self.config.decoder, "bos_token_id")
            and self.config.decoder.bos_token_id is not None
        ):
            return self.config.decoder.bos_token_id
        raise ValueError(
            "`decoder_start_token_id` or `bos_token_id` has to be defined for encoder-decoder generation."
        )

    @staticmethod
    def _expand_to_num_beams(tensor, num_beams):
        return jnp.broadcast_to(tensor[:, None], (tensor.shape[0], num_beams) + tensor.shape[1:])

    def _adapt_logits_for_beam_search(self, logits):
        """
        This function can be overwritten in the specific modeling_flax_<model-name>.py classes to allow for custom beam
        search behavior. Note that the only model that overwrites this method is [`~transformes.FlaxMarianMTModel`].
        """
        return logits

    def _validate_model_class(self):
        """
        Confirms that the model class is compatible with generation. If not, raises an exception that points to the
        right class to use.
        """
        if not self.can_generate():
            generate_compatible_mappings = [
                FLAX_MODEL_FOR_CAUSAL_LM_MAPPING,
                FLAX_MODEL_FOR_VISION_2_SEQ_MAPPING,
                FLAX_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
            ]
            generate_compatible_classes = set()
            for model_mapping in generate_compatible_mappings:
                supported_models = model_mapping.get(type(self.config), default=None)
                if supported_models is not None:
                    generate_compatible_classes.add(supported_models.__name__)
            exception_message = (
                f"The current model class ({self.__class__.__name__}) is not compatible with `.generate()`, as "
                "it doesn't have a language model head."
            )
            if generate_compatible_classes:
                exception_message += f" Please use one of the following classes instead: {generate_compatible_classes}"
            raise TypeError(exception_message)

    def _validate_model_kwargs(self, model_kwargs: Dict[str, Any]):
        """Validates model kwargs for generation. Generate argument typos will also be caught here."""
        unused_model_args = []
        model_args = set(inspect.signature(self.prepare_inputs_for_generation).parameters)
        # `kwargs`/`model_kwargs` is often used to handle optional forward pass inputs like `attention_mask`. If
        # `prepare_inputs_for_generation` doesn't accept them, then a stricter check can be made ;)
        if "kwargs" in model_args or "model_kwargs" in model_args:
            model_args |= set(inspect.signature(self.__call__).parameters)
        for key, value in model_kwargs.items():
            if value is not None and key not in model_args:
                unused_model_args.append(key)

        if unused_model_args:
            raise ValueError(
                f"The following `model_kwargs` are not used by the model: {unused_model_args} (note: typos in the"
                " generate arguments will also show up in this list)"
            )

    def generate(
        self,
        input_ids: jnp.ndarray,
        generation_config: Optional[GenerationConfig] = None,
        prng_key: Optional[jnp.ndarray] = None,
        trace: bool = True,
        params: Optional[Dict[str, jnp.ndarray]] = None,
        logits_processor: Optional[FlaxLogitsProcessorList] = None,
        **kwargs,
    ):
        r"""
        Generates sequences of token ids for models with a language modeling head.

        Parameters:
            input_ids (`jnp.ndarray` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            generation_config (`~generation.GenerationConfig`, *optional*):
                The generation configuration to be used as base parametrization for the generation call. `**kwargs`
                passed to generate matching the attributes of `generation_config` will override them. If
                `generation_config` is not provided, the default will be used, which had the following loading
                priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
                configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
                default values, whose documentation should be checked to parameterize generation.
            trace (`bool`, *optional*, defaults to `True`):
                Whether to trace generation. Setting `trace=False` should only be used for debugging and will lead to a
                considerably slower runtime.
            params (`Dict[str, jnp.ndarray]`, *optional*):
                Optionally the model parameters can be passed. Can be useful for parallelized generation.
            logits_processor (`FlaxLogitsProcessorList `, *optional*):
                Custom logits processors that complement the default logits processors built from arguments and
                generation config. If a logit processor is passed that is already created with the arguments or a
                generation config an error is thrown. This feature is intended for advanced users.
            kwargs (`Dict[str, Any]`, *optional*):
                Ad hoc parametrization of `generate_config` and/or additional model-specific kwargs that will be
                forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder
                specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.

        Return:
            [`~utils.ModelOutput`].

        """
        # Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        self._validate_model_class()

        # priority: `generation_config` argument > `model.generation_config` (the default generation config)
        if generation_config is None:
            # legacy: users may modify the model configuration to control generation. To trigger this legacy behavior,
            # two conditions must be met
            # 1) the generation config must have been created from the model config (`_from_model_config` field);
            # 2) the generation config must have seen no modification since its creation (the hash is the same).
            if self.generation_config._from_model_config and self.generation_config._original_object_hash == hash(
                self.generation_config
            ):
                new_generation_config = GenerationConfig.from_model_config(self.config)
                if new_generation_config != self.generation_config:
                    warnings.warn(
                        "You have modified the pretrained model configuration to control generation. This is a"
                        " deprecated strategy to control generation and will be removed soon, in a future version."
                        " Please use and modify the model generation configuration (see"
                        " https://huggingface.co/docs/transformers/generation_strategies#default-text-generation-configuration )"
                    )
                    self.generation_config = new_generation_config
            generation_config = self.generation_config

        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(**kwargs)  # All unused kwargs must be model kwargs
        self._validate_model_kwargs(model_kwargs.copy())

        logits_processor = logits_processor if logits_processor is not None else FlaxLogitsProcessorList()

        # set init values
        prng_key = prng_key if prng_key is not None else jax.random.PRNGKey(0)

        if generation_config.pad_token_id is None and generation_config.eos_token_id is not None:
            if model_kwargs.get("attention_mask") is None:
                logger.warning(
                    "The attention mask and the pad token id were not set. As a consequence, you may observe "
                    "unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results."
                )
            eos_token_id = generation_config.eos_token_id
            if isinstance(eos_token_id, list):
                eos_token_id = eos_token_id[0]
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
            generation_config.pad_token_id = eos_token_id

        if generation_config.decoder_start_token_id is None and self.config.is_encoder_decoder:
            raise ValueError("`decoder_start_token_id` has to be defined for encoder-decoder generation.")

        # decoder-only models should use left-padding for generation (can't be checked with `trace=True`)
        if not self.config.is_encoder_decoder and not trace:
            if (
                generation_config.pad_token_id is not None
                and jnp.sum(input_ids[:, -1] == generation_config.pad_token_id) > 0
            ):
                logger.warning(
                    "A decoder-only architecture is being used, but right-padding was detected! For correct "
                    "generation results, please set `padding_side='left'` when initializing the tokenizer."
                )

        batch_size = input_ids.shape[0]

        if self.config.is_encoder_decoder:
            # add encoder_outputs to model_kwargs
            if model_kwargs.get("encoder_outputs") is None:
                model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(input_ids, params, model_kwargs)
            # prepare decoder_input_ids for generation
            input_ids = self._prepare_decoder_input_ids_for_generation(
                batch_size,
                decoder_start_token_id=generation_config.decoder_start_token_id,
                bos_token_id=generation_config.bos_token_id,
                model_kwargs=model_kwargs,
            )

        # Prepare `max_length` depending on other stopping criteria.
        input_ids_seq_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        if has_default_max_length and generation_config.max_new_tokens is None and generation_config.max_length == 20:
            # 20 is the default max_length of the generation config
            warnings.warn(
                f"Using the model-agnostic default `max_length` (={generation_config.max_length}) "
                "to control the generation length.  recommend setting `max_new_tokens` to control the maximum length of the generation.",
                UserWarning,
            )
        elif generation_config.max_new_tokens is not None:
            if not has_default_max_length and generation_config.max_length is not None:
                logger.warning(
                    f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                    f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                    "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
                )
            generation_config.max_length = generation_config.max_new_tokens + input_ids_seq_length
        else:  # by default let's always generate 10 new tokens
            if generation_config.max_length == GenerationConfig().max_length:
                generation_config.max_length = generation_config.max_length + input_ids_seq_length
                max_position_embeddings = getattr(self.config, "max_position_embeddings", None)
                if max_position_embeddings is not None:
                    generation_config.max_length = min(generation_config.max_length, max_position_embeddings)

        if generation_config.min_length is not None and generation_config.min_length > generation_config.max_length:
            raise ValueError(
                f"Unfeasable length constraints: the minimum length ({generation_config.min_length}) is larger than"
                f" the maximum length ({generation_config.max_length})"
            )
        if input_ids_seq_length >= generation_config.max_length:
            input_ids_string = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
            logger.warning(
                f"Input length of {input_ids_string} is {input_ids_seq_length}, but `max_length` is set to"
                f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
                " increasing`max_new_tokens`."
            )

        logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_seq_length,
            logits_processor=logits_processor,
        )

        if not generation_config.do_sample and generation_config.num_beams == 1:
            return self._greedy_search(
                input_ids,
                generation_config.max_length,
                generation_config.pad_token_id,
                generation_config.eos_token_id,
                logits_processor=logits_processor,
                trace=trace,
                params=params,
                model_kwargs=model_kwargs,
            )
        elif generation_config.do_sample and generation_config.num_beams == 1:
            logits_warper = self._get_logits_warper(generation_config=generation_config)
            return self._sample(
                input_ids,
                generation_config.max_length,
                generation_config.pad_token_id,
                generation_config.eos_token_id,
                prng_key,
                logits_warper=logits_warper,
                logits_processor=logits_processor,
                trace=trace,
                params=params,
                model_kwargs=model_kwargs,
            )
        elif not generation_config.do_sample and generation_config.num_beams > 1:
            # broadcast input_ids & encoder_outputs
            input_ids = self._expand_to_num_beams(input_ids, num_beams=generation_config.num_beams)

            if "encoder_outputs" in model_kwargs:
                model_kwargs["encoder_outputs"]["last_hidden_state"] = self._expand_to_num_beams(
                    model_kwargs["encoder_outputs"]["last_hidden_state"], num_beams=generation_config.num_beams
                )

            for kwarg in ["attention_mask", "decoder_attention_mask"]:
                if kwarg in model_kwargs:
                    model_kwargs[kwarg] = self._expand_to_num_beams(
                        model_kwargs[kwarg], num_beams=generation_config.num_beams
                    )

            return self._beam_search(
                input_ids,
                generation_config.max_length,
                generation_config.pad_token_id,
                generation_config.eos_token_id,
                length_penalty=generation_config.length_penalty,
                early_stopping=generation_config.early_stopping,
                logits_processor=logits_processor,
                trace=trace,
                params=params,
                num_return_sequences=generation_config.num_return_sequences,
                model_kwargs=model_kwargs,
            )
        else:
            raise NotImplementedError("`Beam sampling is currently not implemented.")

    def _get_logits_warper(self, generation_config: GenerationConfig) -> FlaxLogitsProcessorList:
        """
        This class returns a [`FlaxLogitsProcessorList`] list object that contains all relevant [`FlaxLogitsWarper`]
        instances used for multinomial sampling.
        """
        warpers = FlaxLogitsProcessorList()

        if generation_config.temperature is not None and generation_config.temperature != 1.0:
            warpers.append(FlaxTemperatureLogitsWarper(generation_config.temperature))
        if generation_config.top_k is not None and generation_config.top_k != 0:
            warpers.append(FlaxTopKLogitsWarper(top_k=generation_config.top_k, min_tokens_to_keep=1))
        if generation_config.top_p is not None and generation_config.top_p < 1.0:
            warpers.append(FlaxTopPLogitsWarper(top_p=generation_config.top_p, min_tokens_to_keep=1))

        return warpers

    def _get_logits_processor(
        self,
        generation_config: GenerationConfig,
        input_ids_seq_length: int,
        logits_processor: Optional[FlaxLogitsProcessorList],
    ) -> FlaxLogitsProcessorList:
        """
        This class returns a [`FlaxLogitsProcessorList`] list object that contains all relevant [`FlaxLogitsProcessor`]
        instances used to modify the scores of the language model head.
        """
        processors = FlaxLogitsProcessorList()

        if (
            generation_config.min_length is not None
            and generation_config.eos_token_id is not None
            and generation_config.min_length > -1
        ):
            processors.append(
                FlaxMinLengthLogitsProcessor(generation_config.min_length, generation_config.eos_token_id)
            )
        if generation_config.forced_bos_token_id is not None:
            processors.append(FlaxForcedBOSTokenLogitsProcessor(generation_config.forced_bos_token_id))
        if generation_config.forced_eos_token_id is not None:
            processors.append(
                FlaxForcedEOSTokenLogitsProcessor(generation_config.max_length, generation_config.forced_eos_token_id)
            )
        if generation_config.suppress_tokens is not None:
            processors.append(FlaxSuppressTokensLogitsProcessor(generation_config.suppress_tokens))
        if generation_config.begin_suppress_tokens is not None:
            begin_index = input_ids_seq_length
            begin_index = (
                begin_index
                if (input_ids_seq_length > 1 or generation_config.forced_bos_token_id is None)
                else begin_index + 1
            )
            if generation_config.forced_decoder_ids is not None and len(generation_config.forced_decoder_ids) > 0:
                # generation starts after the last token that is forced
                begin_index += generation_config.forced_decoder_ids[-1][0]
            processors.append(
                FlaxSuppressTokensAtBeginLogitsProcessor(generation_config.begin_suppress_tokens, begin_index)
            )
        if generation_config.forced_decoder_ids is not None:
            forced_decoder_ids = [
                [input_ids_seq_length + i[0] - 1, i[1]] for i in generation_config.forced_decoder_ids
            ]
            processors.append(FlaxForceTokensLogitsProcessor(forced_decoder_ids))
        if generation_config.no_repeat_ngram_size is not None and generation_config.no_repeat_ngram_size > 0:
            processors.append(FlaxNoRepeatNGramLogitsProcessor(generation_config.no_repeat_ngram_size))
        processors = self._merge_criteria_processor_list(processors, logits_processor)

        return processors

    def _merge_criteria_processor_list(
        self,
        default_list: FlaxLogitsProcessorList,
        custom_list: FlaxLogitsProcessorList,
    ) -> FlaxLogitsProcessorList:
        if len(custom_list) == 0:
            return default_list
        for default in default_list:
            for custom in custom_list:
                if type(custom) is type(default):
                    object_type = "logits processor"
                    raise ValueError(
                        f"A custom {object_type} of type {type(custom)} with values {custom} has been passed to"
                        f" `generate`, but it has already been created with the values {default}. {default} has been"
                        " created by passing the corresponding arguments to generate or by the model's config default"
                        f" values. If you just want to change the default values of {object_type} consider passing"
                        f" them as arguments to `generate` instead of using a custom {object_type}."
                    )
        default_list.extend(custom_list)
        return default_list

    def _greedy_search(
        self,
        input_ids: None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        logits_processor: Optional[FlaxLogitsProcessorList] = None,
        trace: bool = True,
        params: Optional[Dict[str, jnp.ndarray]] = None,
        model_kwargs: Optional[Dict[str, jnp.ndarray]] = None,
    ):
        # init values
        max_length = max_length if max_length is not None else self.generation_config.max_length
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id

        batch_size, cur_len = input_ids.shape

        eos_token_id = jnp.array(eos_token_id, dtype=jnp.int32 if eos_token_id is not None else None)
        pad_token_id = jnp.array(pad_token_id, dtype=jnp.int32)
        cur_len = jnp.array(cur_len)

        # per batch-item holding current token in loop.
        sequences = jnp.full((batch_size, max_length), pad_token_id, dtype=jnp.int32)
        sequences = lax.dynamic_update_slice(sequences, input_ids, (0, 0))

        # per batch-item state bit indicating if sentence has finished.
        is_sent_finished = jnp.zeros((batch_size,), dtype=jnp.bool_)

        # For Seq2Seq generation, we only need to use the decoder instead of the whole model in generation loop
        # and pass it the `encoder_outputs`, which are part of the `model_kwargs`.
        model = self.decode if self.config.is_encoder_decoder else self
        # initialize model specific kwargs
        model_kwargs = self.prepare_inputs_for_generation(input_ids, max_length, **model_kwargs)

        # initialize state
        state = GreedyState(
            cur_len=cur_len,
            sequences=sequences,
            running_token=input_ids,
            is_sent_finished=is_sent_finished,
            model_kwargs=model_kwargs,
        )

        def greedy_search_cond_fn(state):
            """state termination condition fn."""
            has_reached_max_length = state.cur_len == max_length
            all_sequence_finished = jnp.all(state.is_sent_finished)
            finish_generation = jnp.logical_or(has_reached_max_length, all_sequence_finished)
            return ~finish_generation

        def greedy_search_body_fn(state):
            """state update fn."""
            model_outputs = model(state.running_token, params=params, **state.model_kwargs)
            logits = model_outputs.logits[:, -1]

            # apply min_length, ...
            logits = logits_processor(state.sequences, logits, state.cur_len)

            next_token = jnp.argmax(logits, axis=-1)

            next_token = next_token * ~state.is_sent_finished + pad_token_id * state.is_sent_finished
            next_is_sent_finished = state.is_sent_finished | (next_token == eos_token_id)
            next_token = next_token[:, None]

            next_sequences = lax.dynamic_update_slice(state.sequences, next_token, (0, state.cur_len))
            next_model_kwargs = self.update_inputs_for_generation(model_outputs, state.model_kwargs)
            return GreedyState(
                cur_len=state.cur_len + 1,
                sequences=next_sequences,
                running_token=next_token,
                is_sent_finished=next_is_sent_finished,
                model_kwargs=next_model_kwargs,
            )

        # The very first prompt often has sequence length > 1, so run outside of `lax.while_loop` to comply with TPU
        if input_ids.shape[1] > 1:
            state = greedy_search_body_fn(state)

        if not trace:
            state = self._run_loop_in_debug(greedy_search_cond_fn, greedy_search_body_fn, state)
        else:
            state = lax.while_loop(greedy_search_cond_fn, greedy_search_body_fn, state)

        return FlaxGreedySearchOutput(sequences=state.sequences)

    def _sample(
        self,
        input_ids: None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        prng_key: Optional[jnp.ndarray] = None,
        logits_processor: Optional[FlaxLogitsProcessorList] = None,
        logits_warper: Optional[FlaxLogitsProcessorList] = None,
        trace: bool = True,
        params: Optional[Dict[str, jnp.ndarray]] = None,
        model_kwargs: Optional[Dict[str, jnp.ndarray]] = None,
    ):
        # init values
        max_length = max_length if max_length is not None else self.generation_config.max_length
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        prng_key = prng_key if prng_key is not None else jax.random.PRNGKey(0)

        batch_size, cur_len = input_ids.shape

        eos_token_id = jnp.array(eos_token_id, dtype=jnp.int32 if eos_token_id is not None else None)
        pad_token_id = jnp.array(pad_token_id, dtype=jnp.int32)
        cur_len = jnp.array(cur_len)

        # per batch-item holding current token in loop.
        sequences = jnp.full((batch_size, max_length), pad_token_id, dtype=jnp.int32)
        sequences = lax.dynamic_update_slice(sequences, input_ids, (0, 0))

        # per batch-item state bit indicating if sentence has finished.
        is_sent_finished = jnp.zeros((batch_size,), dtype=jnp.bool_)

        # For Seq2Seq generation, we only need to use the decoder instead of the whole model in generation loop
        # and pass it the `encoder_outputs`, which are part of the `model_kwargs`.
        model = self.decode if self.config.is_encoder_decoder else self

        # initialize model specific kwargs
        model_kwargs = self.prepare_inputs_for_generation(input_ids, max_length, **model_kwargs)

        # initialize state
        state = SampleState(
            cur_len=cur_len,
            sequences=sequences,
            running_token=input_ids,
            is_sent_finished=is_sent_finished,
            prng_key=prng_key,
            model_kwargs=model_kwargs,
        )

        def sample_search_cond_fn(state):
            """state termination condition fn."""
            has_reached_max_length = state.cur_len == max_length
            all_sequence_finished = jnp.all(state.is_sent_finished)
            finish_generation = jnp.logical_or(has_reached_max_length, all_sequence_finished)
            return ~finish_generation

        def sample_search_body_fn(state):
            """state update fn."""
            prng_key, prng_key_next = jax.random.split(state.prng_key)
            model_outputs = model(state.running_token, params=params, **state.model_kwargs)

            logits = model_outputs.logits[:, -1]

            # apply min_length, ...
            logits = logits_processor(state.sequences, logits, state.cur_len)
            # apply top_p, top_k, temperature
            logits = logits_warper(logits, logits, state.cur_len)

            next_token = jax.random.categorical(prng_key, logits, axis=-1)

            next_token = next_token * ~state.is_sent_finished + pad_token_id * state.is_sent_finished
            next_is_sent_finished = state.is_sent_finished | (next_token == eos_token_id)
            next_token = next_token[:, None]

            next_sequences = lax.dynamic_update_slice(state.sequences, next_token, (0, state.cur_len))
            next_model_kwargs = self.update_inputs_for_generation(model_outputs, state.model_kwargs)

            return SampleState(
                cur_len=state.cur_len + 1,
                sequences=next_sequences,
                running_token=next_token,
                is_sent_finished=next_is_sent_finished,
                model_kwargs=next_model_kwargs,
                prng_key=prng_key_next,
            )

        # The very first prompt often has sequence length > 1, so run outside of `lax.while_loop` to comply with TPU
        if input_ids.shape[1] > 1:
            state = sample_search_body_fn(state)

        if not trace:
            state = self._run_loop_in_debug(sample_search_cond_fn, sample_search_body_fn, state)
        else:
            state = lax.while_loop(sample_search_cond_fn, sample_search_body_fn, state)

        return FlaxSampleOutput(sequences=state.sequences)

    def _beam_search(
        self,
        input_ids: None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        length_penalty: Optional[float] = None,
        early_stopping: Optional[Union[bool, str]] = None,
        logits_processor: Optional[FlaxLogitsProcessorList] = None,
        trace: bool = True,
        params: Optional[Dict[str, jnp.ndarray]] = None,
        num_return_sequences: Optional[int] = None,
        model_kwargs: Optional[Dict[str, jnp.ndarray]] = None,
    ):
        """
        This beam search function is heavily inspired by Flax's official example:
        https://github.com/google/flax/blob/main/examples/wmt/decode.py
        """

        def flatten_beam_dim(tensor):
            """Flattens the first two dimensions of a non-scalar array."""
            # ignore scalars (e.g. cache index)
            if tensor.ndim == 0:
                return tensor
            return tensor.reshape((tensor.shape[0] * tensor.shape[1],) + tensor.shape[2:])

        def unflatten_beam_dim(tensor, batch_size, num_beams):
            """Unflattens the first, flat batch*beam dimension of a non-scalar array."""
            # ignore scalars (e.g. cache index)
            if tensor.ndim == 0:
                return tensor
            return tensor.reshape((batch_size, num_beams) + tensor.shape[1:])

        def gather_beams(nested, beam_indices, batch_size, new_num_beams):
            """
            Gathers the beam slices indexed by beam_indices into new beam array.
            """
            batch_indices = jnp.reshape(
                jnp.arange(batch_size * new_num_beams) // new_num_beams, (batch_size, new_num_beams)
            )

            def gather_fn(tensor):
                # ignore scalars (e.g. cache index)
                if tensor.ndim == 0:
                    return tensor
                else:
                    return tensor[batch_indices, beam_indices]

            return jax.tree_util.tree_map(gather_fn, nested)

        # init values
        max_length = max_length if max_length is not None else self.generation_config.max_length
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        length_penalty = length_penalty if length_penalty is not None else self.generation_config.length_penalty
        early_stopping = early_stopping if early_stopping is not None else self.generation_config.early_stopping
        num_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.generation_config.num_return_sequences
        )

        batch_size, num_beams, cur_len = input_ids.shape

        eos_token_id = jnp.array(eos_token_id, dtype=jnp.int32 if eos_token_id is not None else None)
        pad_token_id = jnp.array(pad_token_id, dtype=jnp.int32)
        cur_len = jnp.array(cur_len)

        # record the prompt length of decoder
        decoder_prompt_len = input_ids.shape[-1]

        # per batch,beam-item holding current token in loop.
        sequences = jnp.full((batch_size, num_beams, max_length), pad_token_id, dtype=jnp.int32)
        running_sequences = jnp.full((batch_size, num_beams, max_length), pad_token_id, dtype=jnp.int32)
        running_sequences = lax.dynamic_update_slice(sequences, input_ids, (0, 0, 0))

        # per batch,beam-item state bit indicating if sentence has finished.
        is_sent_finished = jnp.zeros((batch_size, num_beams), dtype=jnp.bool_)

        # per batch,beam-item score, logprobs
        running_scores = jnp.tile(jnp.array([0.0] + [np.array(-1.0e7)] * (num_beams - 1)), [batch_size, 1])
        scores = jnp.ones((batch_size, num_beams)) * np.array(-1.0e7)

        # For Seq2Seq generation, we only need to use the decoder instead of the whole model in generation loop
        # and pass it the `encoder_outputs`, which are part of the `model_kwargs`.
        model = self.decode if self.config.is_encoder_decoder else self

        # flatten beam dim
        if "encoder_outputs" in model_kwargs:
            model_kwargs["encoder_outputs"]["last_hidden_state"] = flatten_beam_dim(
                model_kwargs["encoder_outputs"]["last_hidden_state"]
            )
        for kwarg in ["attention_mask", "decoder_attention_mask"]:
            if kwarg in model_kwargs:
                model_kwargs[kwarg] = flatten_beam_dim(model_kwargs[kwarg])

        # initialize model specific kwargs
        model_kwargs = self.prepare_inputs_for_generation(flatten_beam_dim(input_ids), max_length, **model_kwargs)

        # initialize state
        state = BeamSearchState(
            cur_len=cur_len,
            running_sequences=running_sequences,
            running_scores=running_scores,
            sequences=sequences,
            scores=scores,
            is_sent_finished=is_sent_finished,
            model_kwargs=model_kwargs,
        )

        def beam_search_cond_fn(state):
            """beam search state termination condition fn."""

            # 1. is less than max length?
            not_max_length_yet = state.cur_len < max_length

            # 2. can the new beams still improve?
            # early_stopping == False -> apply heuristic = always get the best score from `cur_len`. See the discussion
            # below for more details.
            # https://github.com/huggingface/transformers/pull/20901#issuecomment-1369845565
            # early_stopping == "never" -> compute the best score from max_length or cur_len, depending on the sign of
            #   length_penalty. Positive length_penalty favors longer sequences, thus we use max_length there.
            if early_stopping == "never" and length_penalty > 0.0:
                best_running_score = state.running_scores[:, :1] / (
                    (max_length - decoder_prompt_len) ** length_penalty
                )
            else:
                best_running_score = state.running_scores[:, :1] / (
                    (state.cur_len - decoder_prompt_len) ** length_penalty
                )
            worst_finished_score = jnp.where(
                state.is_sent_finished, jnp.min(state.scores, axis=1, keepdims=True), np.array(-1.0e7)
            )
            improvement_still_possible = jnp.any(best_running_score > worst_finished_score)

            # 3. is there still a beam that has not finished?
            still_open_beam = ~(jnp.all(state.is_sent_finished) & (early_stopping is True))

            return not_max_length_yet & still_open_beam & improvement_still_possible

        def beam_search_body_fn(state, input_ids_length=1):
            """beam search state update fn."""
            # 1. Forward current tokens
            # Collect the current position slice along length to feed the fast
            # autoregressive decoder model.  Flatten the beam dimension into batch
            # dimension for feeding into the model.
            # unflatten beam dimension
            # Unflatten beam dimension in attention cache arrays
            input_token = flatten_beam_dim(
                lax.dynamic_slice(
                    state.running_sequences,
                    (0, 0, state.cur_len - input_ids_length),
                    (batch_size, num_beams, input_ids_length),
                )
            )
            model_outputs = model(input_token, params=params, **state.model_kwargs)

            logits = unflatten_beam_dim(model_outputs.logits[:, -1], batch_size, num_beams)
            cache = jax.tree_util.tree_map(
                lambda tensor: unflatten_beam_dim(tensor, batch_size, num_beams), model_outputs.past_key_values
            )

            # adapt logits for FlaxMarianMTModel
            logits = self._adapt_logits_for_beam_search(logits)

            # 2. Compute log probs
            # get log probabilities from logits,
            # process logits with processors (*e.g.* min_length, ...), and
            # add new logprobs to existing running logprobs scores.
            log_probs = jax.nn.log_softmax(logits)
            log_probs = logits_processor(
                flatten_beam_dim(state.running_sequences), flatten_beam_dim(log_probs), state.cur_len
            )
            log_probs = unflatten_beam_dim(log_probs, batch_size, num_beams)
            log_probs = log_probs + jnp.expand_dims(state.running_scores, axis=2)
            vocab_size = log_probs.shape[2]
            log_probs = log_probs.reshape((batch_size, num_beams * vocab_size))

            # 3. Retrieve top-K
            # Each item in batch has num_beams * vocab_size candidate sequences.
            # For each item, get the top 2*k candidates with the highest log-
            # probabilities. We gather the top 2*K beams here so that even if the best
            # K sequences reach EOS simultaneously, we have another K sequences
            # remaining to continue the live beam search.
            # Gather the top 2*K scores from _all_ beams.
            # Gather 2*k top beams.
            # Recover the beam index by floor division.
            # Recover token id by modulo division and expand Id array for broadcasting.
            # Update sequences for the 2*K top-k new sequences.
            beams_to_keep = 2 * num_beams
            topk_log_probs, topk_indices = lax.top_k(log_probs, k=beams_to_keep)
            topk_beam_indices = topk_indices // vocab_size
            topk_running_sequences = gather_beams(
                state.running_sequences, topk_beam_indices, batch_size, beams_to_keep
            )
            topk_ids = jnp.expand_dims(topk_indices % vocab_size, axis=2)
            topk_sequences = lax.dynamic_update_slice(topk_running_sequences, topk_ids, (0, 0, state.cur_len))

            # 4. Check which sequences have ended
            # Update current sequences:
            # Did any of these sequences reach an end marker?
            # To prevent these just finished sequences from being added to the current sequences
            # set of active beam search sequences, set their log probs to a very large
            # negative value.
            did_topk_just_finished = topk_sequences[:, :, state.cur_len] == eos_token_id
            running_topk_log_probs = topk_log_probs + did_topk_just_finished * np.array(-1.0e7)
            # 5. Get running sequences scores for next
            # Determine the top k beam indices (from top 2*k beams) from log probs
            # and gather top k beams (from top 2*k beams).
            next_topk_indices = lax.top_k(running_topk_log_probs, k=num_beams)[1]
            next_running_sequences, next_running_scores = gather_beams(
                [topk_sequences, running_topk_log_probs], next_topk_indices, batch_size, num_beams
            )

            # 6. Process topk logits
            # Further process log probs:
            # - add length penalty
            # - make sure no scores can be added anymore if beam is full
            # - make sure still running sequences cannot be chosen as finalized beam
            topk_log_probs = topk_log_probs / ((state.cur_len + 1 - decoder_prompt_len) ** length_penalty)
            beams_in_batch_are_full = jnp.broadcast_to(
                state.is_sent_finished.all(axis=-1, keepdims=True), did_topk_just_finished.shape
            ) & (early_stopping is True)
            add_penalty = ~did_topk_just_finished | beams_in_batch_are_full
            topk_log_probs += add_penalty * np.array(-1.0e7)

            # 7. Get scores, sequences, is sentence finished for next.
            # Combine sequences, scores, and flags along the beam dimension and compare
            # new finished sequence scores to existing finished scores and select the
            # best from the new set of beams
            merged_sequences = jnp.concatenate([state.sequences, topk_sequences], axis=1)
            merged_scores = jnp.concatenate([state.scores, topk_log_probs], axis=1)
            merged_is_sent_finished = jnp.concatenate([state.is_sent_finished, did_topk_just_finished], axis=1)
            topk_merged_indices = lax.top_k(merged_scores, k=num_beams)[1]
            next_sequences, next_scores, next_is_sent_finished = gather_beams(
                [merged_sequences, merged_scores, merged_is_sent_finished], topk_merged_indices, batch_size, num_beams
            )

            # 8. Update model kwargs.
            # Determine the top k beam indices from the original set of all beams.
            # With these, gather the top k beam-associated caches.
            next_running_indices = gather_beams(topk_beam_indices, next_topk_indices, batch_size, num_beams)
            next_cache = gather_beams(cache, next_running_indices, batch_size, num_beams)
            model_outputs["past_key_values"] = jax.tree_util.tree_map(lambda x: flatten_beam_dim(x), next_cache)
            next_model_kwargs = self.update_inputs_for_generation(model_outputs, state.model_kwargs)

            return BeamSearchState(
                cur_len=state.cur_len + 1,
                running_scores=next_running_scores,
                running_sequences=next_running_sequences,
                scores=next_scores,
                sequences=next_sequences,
                is_sent_finished=next_is_sent_finished,
                model_kwargs=next_model_kwargs,
            )

        # Always run first iteration outside of `lax.while_loop` to avoid calling `beam_search_cond_fn`
        # when `state.cur_len` equals `decoder_prompt_len`. This also helps to comply with TPU when
        # the very first prompt has sequence length > 1.
        state = partial(beam_search_body_fn, input_ids_length=input_ids.shape[-1])(state)

        if not trace:
            state = self._run_loop_in_debug(beam_search_cond_fn, beam_search_body_fn, state)
        else:
            state = lax.while_loop(beam_search_cond_fn, beam_search_body_fn, state)

        # Account for the edge-case where there are no finished sequences for a
        # particular batch item. If so, return running sequences for that batch item.
        none_finished = jnp.any(state.is_sent_finished, axis=1)
        sequences = jnp.where(none_finished[:, None, None], state.sequences, state.running_sequences)
        scores = jnp.where(none_finished[:, None], state.scores, state.running_scores)

        # Take best beams for each batch (the score is sorted in descending order)
        sequences = flatten_beam_dim(sequences[:, :num_return_sequences, :])
        scores = flatten_beam_dim(scores[:, :num_return_sequences])

        return FlaxBeamSearchOutput(sequences=sequences, scores=scores)
