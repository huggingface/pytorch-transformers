# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team
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

import inspect

import jax
import jax.lax as lax
import jax.numpy as jnp
from jax.experimental import sparse

from ..utils import add_start_docstrings
from ..utils.logging import get_logger


logger = get_logger(__name__)


LOGITS_PROCESSOR_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`jnp.ndarray` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`PreTrainedTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        scores (`jnp.ndarray` of shape `(batch_size, config.vocab_size)`):
            Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
            search or log softmax for each vocabulary token when using beam search
        kwargs (`Dict[str, Any]`, *optional*):
            Additional logits processor specific kwargs.

    Return:
        `jnp.ndarray` of shape `(batch_size, config.vocab_size)`: The processed prediction scores.

"""


class FlaxLogitsProcessor:
    """Abstract base class for all logit processors that can be applied during generation."""

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray) -> jnp.ndarray:
        """Flax method for processing logits."""
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )


class FlaxLogitsWarper:
    """Abstract base class for all logit warpers that can be applied during generation with multinomial sampling."""

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray) -> jnp.ndarray:
        """Flax method for warping logits."""
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )


class FlaxLogitsProcessorList(list):
    """
    This class can be used to create a list of [`FlaxLogitsProcessor`] or [`FlaxLogitsWarper`] to subsequently process
    a `scores` input tensor. This class inherits from list and adds a specific *__call__* method to apply each
    [`FlaxLogitsProcessor`] or [`FlaxLogitsWarper`] to the inputs.
    """

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int, **kwargs) -> jnp.ndarray:
        for processor in self:
            function_args = inspect.signature(processor.__call__).parameters
            if len(function_args) > 3:
                if not all(arg in kwargs for arg in list(function_args.keys())[2:]):
                    raise ValueError(
                        f"Make sure that all the required parameters: {list(function_args.keys())} for "
                        f"{processor.__class__} are passed to the logits processor."
                    )
                scores = processor(input_ids, scores, cur_len, **kwargs)
            else:
                scores = processor(input_ids, scores, cur_len)
        return scores


class FlaxTemperatureLogitsWarper(FlaxLogitsWarper):
    r"""
    [`FlaxLogitsWarper`] for temperature (exponential scaling output probability distribution).

    Args:
        temperature (`float`):
            The value used to module the logits distribution.
    """

    def __init__(self, temperature: float):
        if not isinstance(temperature, float) or not (temperature > 0):
            raise ValueError(f"`temperature` has to be a strictly positive float, but is {temperature}")

        self.temperature = temperature

    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int) -> jnp.ndarray:
        scores = scores / self.temperature
        return scores


class FlaxTopPLogitsWarper(FlaxLogitsWarper):
    """
    [`FlaxLogitsWarper`] that performs top-p, i.e. restricting to top tokens summing to prob_cut_off <= prob_cut_off.

    Args:
        top_p (`float`):
            If set to < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or
            higher are kept for generation.
        filter_value (`float`, *optional*, defaults to -inf):
            All filtered values will be set to this float value.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered.
    """

    def __init__(self, top_p: float, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        if not isinstance(top_p, float) or (top_p < 0 or top_p > 1.0):
            raise ValueError(f"`top_p` has to be a float > 0 and < 1, but is {top_p}")
        if not isinstance(min_tokens_to_keep, int) or (min_tokens_to_keep < 1):
            raise ValueError(f"`min_tokens_to_keep` has to be a positive integer, but is {min_tokens_to_keep}")

        self.top_p = top_p
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int) -> jnp.ndarray:
        topk_scores, topk_indices = lax.top_k(scores, scores.shape[-1])

        mask_scores = jnp.full_like(scores, self.filter_value)
        cumulative_probs = jax.nn.softmax(topk_scores, axis=-1).cumsum(axis=-1)
        score_mask = cumulative_probs < self.top_p

        # include the token that is higher than top_p as well
        score_mask = jnp.roll(score_mask, 1)
        score_mask |= score_mask.at[:, 0].set(True)

        # min tokens to keep
        score_mask = score_mask.at[:, : self.min_tokens_to_keep].set(True)

        topk_next_scores = jnp.where(score_mask, topk_scores, mask_scores)
        next_scores = jax.lax.sort_key_val(topk_indices, topk_next_scores)[-1]

        return next_scores


class FlaxTopKLogitsWarper(FlaxLogitsWarper):
    r"""
    [`FlaxLogitsWarper`] that performs top-k, i.e. restricting to the k highest probability elements.

    Args:
        top_k (`int`):
            The number of highest probability vocabulary tokens to keep for top-k-filtering.
        filter_value (`float`, *optional*, defaults to -inf):
            All filtered values will be set to this float value.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered.
    """

    def __init__(self, top_k: int, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"`top_k` has to be a strictly positive integer, but is {top_k}")

        self.top_k = max(top_k, min_tokens_to_keep)
        self.filter_value = filter_value

    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int) -> jnp.ndarray:
        batch_size, vocab_size = scores.shape
        next_scores_flat = jnp.full(batch_size * vocab_size, self.filter_value)

        topk = min(self.top_k, scores.shape[-1])  # Safety check
        topk_scores, topk_indices = lax.top_k(scores, topk)
        shift = jnp.broadcast_to((jnp.arange(batch_size) * vocab_size)[:, None], (batch_size, topk)).flatten()
        topk_scores_flat = topk_scores.flatten()
        topk_indices_flat = topk_indices.flatten() + shift

        next_scores_flat = next_scores_flat.at[topk_indices_flat].set(topk_scores_flat)
        next_scores = next_scores_flat.reshape(batch_size, vocab_size)
        return next_scores


class FlaxForcedBOSTokenLogitsProcessor(FlaxLogitsProcessor):
    r"""
    [`FlaxLogitsProcessor`] that enforces the specified token as the first generated token.

    Args:
        bos_token_id (`int`):
            The id of the token to force as the first generated token.
    """

    def __init__(self, bos_token_id: int):
        self.bos_token_id = bos_token_id

    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int) -> jnp.ndarray:
        new_scores = jnp.full(scores.shape, -float("inf"))

        apply_penalty = 1 - jnp.bool_(cur_len - 1)

        scores = jnp.where(apply_penalty, new_scores.at[:, self.bos_token_id].set(0), scores)

        return scores


class FlaxForcedEOSTokenLogitsProcessor(FlaxLogitsProcessor):
    r"""
    [`FlaxLogitsProcessor`] that enforces the specified token as the last generated token when `max_length` is reached.

    Args:
        max_length (`int`):
            The maximum length of the sequence to be generated.
        eos_token_id (`int`):
            The id of the token to force as the last generated token when `max_length` is reached.
    """

    def __init__(self, max_length: int, eos_token_id: int):
        self.max_length = max_length
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int) -> jnp.ndarray:
        new_scores = jnp.full(scores.shape, -float("inf"))

        apply_penalty = 1 - jnp.bool_(cur_len - self.max_length + 1)

        scores = jnp.where(apply_penalty, new_scores.at[:, self.eos_token_id].set(0), scores)

        return scores


class FlaxMinLengthLogitsProcessor(FlaxLogitsProcessor):
    r"""
    [`FlaxLogitsProcessor`] enforcing a min-length by setting EOS probability to 0.

    Args:
        min_length (`int`):
            The minimum length below which the score of `eos_token_id` is set to `-float("Inf")`.
        eos_token_id (`int`):
            The id of the *end-of-sequence* token.
    """

    def __init__(self, min_length: int, eos_token_id: int):
        if not isinstance(min_length, int) or min_length < 0:
            raise ValueError(f"`min_length` has to be a positive integer, but is {min_length}")

        if not isinstance(eos_token_id, int) or eos_token_id < 0:
            raise ValueError(f"`eos_token_id` has to be a positive integer, but is {eos_token_id}")

        self.min_length = min_length
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int) -> jnp.ndarray:
        # create boolean flag to decide if min length penalty should be applied
        apply_penalty = 1 - jnp.clip(cur_len - self.min_length, 0, 1)

        scores = jnp.where(apply_penalty, scores.at[:, self.eos_token_id].set(-float("inf")), scores)

        return scores


class FlaxSuppressTokensAtBeginLogitsProcessor(FlaxLogitsProcessor):
    r"""
    [`FlaxLogitsProcessor`] supressing a list of tokens as soon as the `generate` function starts generating using
    `begin_index` tokens. This should ensure that the tokens defined by `begin_suppress_tokens` are not sampled at the
    beginning of the generation.

    Args:
        begin_suppress_tokens (`List[int]`):
            Tokens to not sample.
        begin_index (`int`):
            Index where the tokens are suppressed.
    """

    def __init__(self, begin_suppress_tokens, begin_index):
        self.begin_suppress_tokens = list(begin_suppress_tokens)
        self.begin_index = begin_index

    def __call__(self, input_ids, scores, cur_len: int):
        apply_penalty = 1 - jnp.bool_(cur_len - self.begin_index)

        scores = jnp.where(apply_penalty, scores.at[:, self.begin_suppress_tokens].set(-float("inf")), scores)

        return scores


class FlaxSuppressTokensLogitsProcessor(FlaxLogitsProcessor):
    r"""
    [`FlaxLogitsProcessor`] suppressing a list of tokens at each decoding step. The processor will set their log probs
    to be `-inf` so they are not sampled.

    Args:
        suppress_tokens (`list`):
            Tokens to not sample.
    """

    def __init__(self, suppress_tokens: list):
        self.suppress_tokens = list(suppress_tokens)

    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int) -> jnp.ndarray:
        scores = scores.at[..., self.suppress_tokens].set(-float("inf"))

        return scores


class FlaxForceTokensLogitsProcessor(FlaxLogitsProcessor):
    r"""
    [`FlaxLogitsProcessor`] that takes a list of pairs of integers which indicates a mapping from generation indices to
    token indices that will be forced before sampling. The processor will set their log probs to 0 and all other tokens
    to `-inf` so that they are sampled at their corresponding index.

    Args:
        force_token_map (`list`):
            Map giving token ids and indices where they will be forced to be sampled.
    """

    def __init__(self, force_token_map):
        force_token_map = dict(force_token_map)
        # Converts the dictionary of format {index: token} containing the tokens to be forced to an array, where the
        # index of the array corresponds to the index of the token to be forced, for XLA compatibility.
        # Indexes without forced tokens will have a negative value.
        force_token_array = jnp.ones((max(force_token_map.keys()) + 1), dtype=jnp.int32) * -1
        for index, token in force_token_map.items():
            if token is not None:
                force_token_array = force_token_array.at[index].set(token)
        self.force_token_array = jnp.int32(force_token_array)

    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int) -> jnp.ndarray:
        def _force_token(generation_idx):
            batch_size = scores.shape[0]
            current_token = self.force_token_array[generation_idx]

            new_scores = jnp.ones_like(scores, dtype=scores.dtype) * -float("inf")
            updates = jnp.zeros((batch_size, 1), dtype=scores.dtype)
            new_scores = lax.dynamic_update_slice(new_scores, updates, (0, current_token))
            return new_scores

        scores = lax.cond(
            cur_len >= self.force_token_array.shape[0],
            # If the current length is geq than the length of force_token_array, the processor does nothing.
            lambda: scores,
            # Otherwise, it may force a certain token.
            lambda: lax.cond(
                self.force_token_array[cur_len] >= 0,
                # Only valid (positive) tokens are forced
                lambda: _force_token(cur_len),
                # Otherwise, the processor does nothing.
                lambda: scores,
            ),
        )
        return scores


class FlaxWhisperTimeStampLogitsProcessor(FlaxLogitsProcessor):
    r"""
    Whisper specific Processor. This processor can be used to force a list of tokens. The processor will set their log
    probs to `inf` so that they are sampled at their corresponding index.

    Args:
        generate_config (`GenerateConfig`):
            The generate config used to generate the output. The following parameters are required:
                eos_token_id (`int`, *optional*, defaults to 50257):
                    The id of the *end-of-sequence* token.
                no_timestamps_token_id (`int`, *optional*, defaults to 50363):
                    The id of the `"<|notimestamps|>"` token.
                max_initial_timestamp_index (`int`, *optional*, defaults to 1):
                    Used to set the maximum value of the initial timestamp. This is used to prevent the model from
                    predicting timestamps that are too far in the future.
    """

    def __init__(self, generate_config, model_config, decoder_input_length):
        self.eos_token_id = generate_config.eos_token_id
        self.no_timestamps_token_id = generate_config.no_timestamps_token_id
        self.timestamp_begin = generate_config.no_timestamps_token_id + 1

        self.begin_index = decoder_input_length + 1

        if generate_config.is_multilingual:
            # room for language token and task token
            self.begin_index += 2
        if hasattr(generate_config, "max_initial_timestamp_index"):
            self.max_initial_timestamp_index = generate_config.max_initial_timestamp_index
        else:
            self.max_initial_timestamp_index = model_config.vocab_size
        if self.max_initial_timestamp_index is None:
            self.max_initial_timestamp_index = model_config.vocab_size

    def __call__(self, input_ids, scores, cur_len):
        # suppress <|notimestamps|> which is handled by without_timestamps
        scores = scores.at[:, self.no_timestamps_token_id].set(-float("inf"))

        def handle_pairs(input_ids_k, scores_k):
            last_was_timestamp = jnp.where((cur_len - self.begin_index) >= 1, True, False)
            last_was_timestamp = jnp.where(
                input_ids_k[cur_len - 1] >= self.timestamp_begin,
                True and last_was_timestamp,
                False,
            )

            penultimate_was_timestamp = jnp.where((cur_len - self.begin_index) < 2, True, False)
            penultimate_was_timestamp = jnp.where(
                input_ids_k[cur_len - 2] >= self.timestamp_begin,
                True,
                penultimate_was_timestamp,
            )

            return jnp.where(
                last_was_timestamp,
                jnp.where(
                    penultimate_was_timestamp > 0,
                    scores_k.at[self.timestamp_begin :].set(-float("inf")),
                    scores_k.at[: self.eos_token_id].set(-float("inf")),
                ),
                scores_k,
            )

        scores = jax.vmap(handle_pairs)(input_ids, scores)

        apply_max_initial_timestamp = jnp.where(cur_len == self.begin_index, True, False)
        apply_max_initial_timestamp = jnp.where(
            self.max_initial_timestamp_index is not None,
            True and apply_max_initial_timestamp,
            False,
        )

        last_allowed = self.timestamp_begin + self.max_initial_timestamp_index

        scores = jnp.where(
            apply_max_initial_timestamp,
            scores.at[:, last_allowed + 1 :].set(-float("inf")),
            scores,
        )

        # if sum of probability over timestamps is above any other token, sample timestamp
        logprobs = jax.nn.log_softmax(scores, axis=-1)

        def handle_cumulative_probs(logprobs_k, scores_k):
            timestamp_logprob = jax.nn.logsumexp(logprobs_k[self.timestamp_begin :], axis=-1)
            max_text_token_logprob = jnp.max(logprobs_k[: self.timestamp_begin])
            return jnp.where(
                timestamp_logprob > max_text_token_logprob,
                scores_k.at[: self.timestamp_begin].set(-float("inf")),
                scores_k,
            )

        scores = jax.vmap(handle_cumulative_probs)(logprobs, scores)

        return scores


class FlaxNoRepeatNGramLogitsProcessor(FlaxLogitsProcessor):
    r"""
    [`FlaxLogitsProcessor`] that enforces no repetition of n-grams. See
    [Fairseq](https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345).

    Args:
        ngram_size (`int`):
            All ngrams of size `ngram_size` can only occur once.
    """

    def __init__(self, ngram_size: int):
        if not isinstance(ngram_size, int) or ngram_size <= 0:
            raise ValueError(f"`ngram_size` has to be a strictly positive integer, but is {ngram_size}")
        self.ngram_size = ngram_size

    def get_previous_ngrams(self, input_ids: jnp.ndarray, vocab_size: int, cur_len: int):
        """
        get a matrix of size (batch_size,) + (vocab_size,)*n (for n-grams) that
        represent the n-grams that occurred previously.
        The BCOO representation allow to store only the few non-zero entries, instead of the full (huge) matrix
        """
        batch_size, seq_len = input_ids.shape
        # number of n-grams in the whole sequence
        seq_ngrams = seq_len - (self.ngram_size - 1)
        # number of n-grams in the currently generated sequence
        cur_ngrams = cur_len - (self.ngram_size - 1)

        def body_fun(i, val):
            b = i % batch_size
            pos = i // batch_size
            return val.at[i].set(
                jnp.array(
                    [
                        b,
                    ]
                    + [jnp.array(input_ids)[b, pos + j] for j in range(self.ngram_size)]
                )
            )

        shape = (batch_size * seq_ngrams, self.ngram_size + 1)
        all_update_indices = jax.lax.fori_loop(
            0, batch_size * cur_ngrams, body_fun, jnp.zeros(shape, dtype=input_ids.dtype)
        )

        # ignore the n-grams not yet generated
        data = (jnp.arange(batch_size * seq_ngrams) < batch_size * cur_ngrams).astype("float32")

        return sparse.BCOO((data, all_update_indices), shape=(batch_size,) + (vocab_size,) * self.ngram_size)

    def get_banned_tokens_mask(self, latest_tokens: jnp.ndarray, previous_ngrams) -> jnp.ndarray:
        """
        Determines which tokens must be banned given latest tokens and the previously seen
        ngrams.
        """

        @sparse.sparsify
        @jax.vmap
        def inner_fn(latest_tokens, previous_ngrams):
            return previous_ngrams[tuple(latest_tokens)]

        return sparse.bcoo_todense(inner_fn(latest_tokens, previous_ngrams))

    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int) -> jnp.ndarray:
        def true_fn():
            _, vocab_size = scores.shape
            # store the previously seen n-grams
            previous_ngrams = self.get_previous_ngrams(input_ids, vocab_size, cur_len)

            # get the n-1 last tokens that prefix the n-gram being generated
            latest_tokens = jnp.zeros((input_ids.shape[0], self.ngram_size - 1), dtype=input_ids.dtype)
            latest_tokens = jax.lax.dynamic_update_slice(
                latest_tokens,
                jax.lax.dynamic_slice(
                    input_ids, (0, cur_len - (self.ngram_size - 1)), (input_ids.shape[0], (self.ngram_size - 1))
                ),
                (0, 0),
            )

            # compute the banned tokens, ie all the tokens that when added to the latest tokens lead to a n-gram that was previously generated
            banned_tokens_indices_mask = self.get_banned_tokens_mask(latest_tokens, previous_ngrams).astype("bool")
            return jnp.where(banned_tokens_indices_mask, -float("inf"), scores)

        output = jax.lax.cond((cur_len >= self.ngram_size - 1), true_fn, lambda: scores)
        return output
