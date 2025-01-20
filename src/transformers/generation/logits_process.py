# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team and Google DeepMind.
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
import math
from typing import Callable, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch

from ..pytorch_utils import isin_mps_friendly
from ..utils import add_start_docstrings
from ..utils.logging import get_logger


logger = get_logger(__name__)


LOGITS_PROCESSOR_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
        scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`):
            Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
            search or log softmax for each vocabulary token when using beam search

    Return:
        `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`: The processed prediction scores.

"""


class LogitsProcessor:
    """Abstract base class for all logit processors that can be applied during generation."""

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )


class LogitsProcessorList(list):
    """
    This class can be used to create a list of [`LogitsProcessor`] to subsequently process a `scores` input tensor.
    This class inherits from list and adds a specific *__call__* method to apply each [`LogitsProcessor`] to the
    inputs.
    """

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
            scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`):
                Prediction scores of a language modeling head. These can be logits for each vocabulary when not using
                beam search or log softmax for each vocabulary token when using beam search
            kwargs (`Dict[str, Any]`, *optional*):
                Additional kwargs that are specific to a logits processor.

        Return:
            `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`:
                The processed prediction scores.

        """
        for processor in self:
            function_args = inspect.signature(processor.__call__).parameters
            if len(function_args) > 2:
                if not all(arg in kwargs for arg in list(function_args.keys())[2:]):
                    raise ValueError(
                        f"Make sure that all the required parameters: {list(function_args.keys())} for "
                        f"{processor.__class__} are passed to the logits processor."
                    )
                scores = processor(input_ids, scores, **kwargs)
            else:
                scores = processor(input_ids, scores)

        return scores


class MinLengthLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] enforcing a min-length by setting EOS probability to 0. Note that, for decoder-only models
    like most LLMs, the length includes the prompt.

    Args:
        min_length (`int`):
            The minimum length below which the score of `eos_token_id` is set to `-float("Inf")`.
        eos_token_id (`Union[int, List[int], torch.Tensor]`):
            The id(s) of the *end-of-sequence* token.
        device (`str`, *optional*, defaults to `"cpu"`):
            The device to allocate the tensors.

    Examples:

    ```python
    >>> from transformers import AutoModelForCausalLM, AutoTokenizer

    >>> tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-560m")
    >>> model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-560m")

    >>> inputs = tokenizer("A number:", return_tensors="pt")
    >>> gen_out = model.generate(**inputs)
    >>> print(tokenizer.batch_decode(gen_out, skip_special_tokens=True)[0])
    A number: one

    >>> # setting `min_length` to a value smaller than the uncontrolled output length has no impact
    >>> gen_out = model.generate(**inputs, min_length=3)
    >>> print(tokenizer.batch_decode(gen_out, skip_special_tokens=True)[0])
    A number: one

    >>> # setting a larger `min_length` will force the model to generate beyond its natural ending point, which is not
    >>> # necessarily incorrect
    >>> gen_out = model.generate(**inputs, min_length=10)
    >>> print(tokenizer.batch_decode(gen_out, skip_special_tokens=True)[0])
    A number: one thousand, nine hundred and ninety-four
    ```
    """

    def __init__(self, min_length: int, eos_token_id: Union[int, List[int], torch.Tensor], device: str = "cpu"):
        if not isinstance(min_length, int) or min_length < 0:
            raise ValueError(f"`min_length` has to be a non-negative integer, but is {min_length}")

        if not isinstance(eos_token_id, torch.Tensor):
            if isinstance(eos_token_id, int):
                eos_token_id = [eos_token_id]
            eos_token_id = torch.tensor(eos_token_id, device=device)

        self.min_length = min_length
        self.eos_token_id = eos_token_id

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        vocab_tensor = torch.arange(scores.shape[-1], device=scores.device)
        eos_token_mask = isin_mps_friendly(vocab_tensor, self.eos_token_id)
        scores_processed = scores.clone()
        if input_ids.shape[-1] < self.min_length:
            scores_processed = torch.where(eos_token_mask, -math.inf, scores)
        return scores_processed


class MinNewTokensLengthLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] enforcing a min-length of new tokens by setting EOS (End-Of-Sequence) token probability to 0.
    Contrarily to [`MinLengthLogitsProcessor`], this processor ignores the prompt.

    Args:
        prompt_length_to_skip (`int`):
            The input tokens length. Not a valid argument when used with `generate` as it will automatically assign the
            input length.
        min_new_tokens (`int`):
            The minimum *new* tokens length below which the score of `eos_token_id` is set to `-float("Inf")`.
        eos_token_id (`Union[int, List[int], torch.Tensor]`):
            The id(s) of the *end-of-sequence* token.
        device (`str`, *optional*, defaults to `"cpu"`):
            The device to allocate the tensors.

    Examples:

    ```python
    >>> from transformers import AutoModelForCausalLM, AutoTokenizer

    >>> tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-560m")
    >>> model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-560m")

    >>> inputs = tokenizer(["A number:"], return_tensors="pt")
    >>> gen_out = model.generate(**inputs)
    >>> print(tokenizer.batch_decode(gen_out, skip_special_tokens=True)[0])
    A number: one

    >>> # setting `min_new_tokens` will force the model to generate beyond its natural ending point, which is not
    >>> # necessarily incorrect
    >>> gen_out = model.generate(**inputs, min_new_tokens=2)
    >>> print(tokenizer.batch_decode(gen_out, skip_special_tokens=True)[0])
    A number: one thousand
    ```
    """

    def __init__(
        self,
        prompt_length_to_skip: int,
        min_new_tokens: int,
        eos_token_id: Union[int, List[int], torch.Tensor],
        device: str = "cpu",
    ):
        for arg_name, arg_value in [
            ("prompt_length_to_skip", prompt_length_to_skip),
            ("min_new_tokens", min_new_tokens),
        ]:
            if not isinstance(arg_value, int) or arg_value < 0:
                raise ValueError(f"`{arg_name}` has to be a positive integer, but is {arg_value}")

        if not isinstance(eos_token_id, torch.Tensor):
            if isinstance(eos_token_id, int):
                eos_token_id = [eos_token_id]
            eos_token_id = torch.tensor(eos_token_id, device=device)

        self.prompt_length_to_skip = prompt_length_to_skip
        self.min_new_tokens = min_new_tokens
        self.eos_token_id = eos_token_id

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        new_tokens_length = input_ids.shape[-1] - self.prompt_length_to_skip
        scores_processed = scores.clone()
        vocab_tensor = torch.arange(scores.shape[-1], device=scores.device)
        eos_token_mask = isin_mps_friendly(vocab_tensor, self.eos_token_id)
        if new_tokens_length < self.min_new_tokens:
            scores_processed = torch.where(eos_token_mask, -math.inf, scores)

        return scores_processed


class TemperatureLogitsWarper(LogitsProcessor):
    r"""
    [`LogitsProcessor`] for temperature (exponential scaling output probability distribution), which effectively means
    that it can control the randomness of the predicted tokens. Often used together with [`TopPLogitsWarper`] and
    [`TopKLogitsWarper`].

    <Tip>

    Make sure that `do_sample=True` is included in the `generate` arguments otherwise the temperature value won't have
    any effect.

    </Tip>

    Args:
        temperature (`float`):
            Strictly positive float value used to modulate the logits distribution. A value smaller than `1` decreases
            randomness (and vice versa), with `0` being equivalent to shifting all probability mass to the most likely
            token.

    Examples:

    ```python
    >>> import torch
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

    >>> set_seed(0)  # for reproducibility

    >>> tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    >>> model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    >>> model.config.pad_token_id = model.config.eos_token_id
    >>> inputs = tokenizer(["Hugging Face Company is"], return_tensors="pt")

    >>> # With temperature=1.0, the default, we consistently get random outputs due to random sampling.
    >>> generate_kwargs = {"max_new_tokens": 10, "do_sample": True, "temperature": 1.0, "num_return_sequences": 2}
    >>> outputs = model.generate(**inputs, **generate_kwargs)
    >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    ['Hugging Face Company is one of these companies that is going to take a',
    "Hugging Face Company is a brand created by Brian A. O'Neil"]

    >>> # However, with temperature close to 0, it approximates greedy decoding strategies (invariant)
    >>> generate_kwargs["temperature"] = 0.0001
    >>> outputs = model.generate(**inputs, **generate_kwargs)
    >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    ['Hugging Face Company is a company that has been around for over 20 years',
    'Hugging Face Company is a company that has been around for over 20 years']
    ```
    """

    def __init__(self, temperature: float):
        if not isinstance(temperature, float) or not (temperature > 0):
            except_msg = (
                f"`temperature` (={temperature}) has to be a strictly positive float, otherwise your next token "
                "scores will be invalid."
            )
            if isinstance(temperature, float) and temperature == 0.0:
                except_msg += " If you're looking for greedy decoding strategies, set `do_sample=False`."
            raise ValueError(except_msg)

        self.temperature = temperature

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores_processed = scores / self.temperature
        return scores_processed


class RepetitionPenaltyLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that prevents the repetition of previous tokens through a penalty. This penalty is applied at
    most once per token. Note that, for decoder-only models like most LLMs, the considered tokens include the prompt.

    In the original [paper](https://arxiv.org/pdf/1909.05858.pdf), the authors suggest the use of a penalty of around
    1.2 to achieve a good balance between truthful generation and lack of repetition. To penalize and reduce
    repetition, use `penalty` values above 1.0, where a higher value penalizes more strongly. To reward and encourage
    repetition, use `penalty` values between 0.0 and 1.0, where a lower value rewards more strongly.

    Args:
        penalty (`float`):
            The parameter for repetition penalty. 1.0 means no penalty. Above 1.0 penalizes previously generated
            tokens. Between 0.0 and 1.0 rewards previously generated tokens.

    Examples:

    ```py
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM

    >>> # Initializing the model and tokenizer for it
    >>> model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
    >>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
    >>> inputs = tokenizer(["I'm not going to"], return_tensors="pt")

    >>> # This shows a normal generate without any specific parameters
    >>> summary_ids = model.generate(**inputs)
    >>> print(tokenizer.batch_decode(summary_ids, skip_special_tokens=True)[0])
    I'm not going to be able to do that. I'm going to be able to do that

    >>> # This generates a penalty for repeated tokens
    >>> penalized_ids = model.generate(**inputs, repetition_penalty=1.1)
    >>> print(tokenizer.batch_decode(penalized_ids, skip_special_tokens=True)[0])
    I'm not going to be able to do that. I'll just have to go out and play
    ```
    """

    def __init__(self, penalty: float):
        if not isinstance(penalty, float) or not (penalty > 0):
            raise ValueError(f"`penalty` has to be a strictly positive float, but is {penalty}")

        self.penalty = penalty

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        score = torch.gather(scores, 1, input_ids)

        # if score < 0 then repetition penalty has to be multiplied to reduce the token probabilities
        score = torch.where(score < 0, score * self.penalty, score / self.penalty)

        scores_processed = scores.scatter(1, input_ids, score)
        return scores_processed


class EncoderRepetitionPenaltyLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that works similarly to [`RepetitionPenaltyLogitsProcessor`], but with an *inverse* penalty
    that is applied to the tokens present in the prompt. In other words, a penalty above 1.0 increases the odds of
    selecting tokens that were present in the prompt.

    It was designed to avoid hallucination in input-grounded tasks, like summarization. Although originally intended
    for encoder-decoder models, it can also be used with decoder-only models like LLMs.

    Args:
        penalty (`float`):
            The parameter for repetition penalty. 1.0 means no penalty. Above 1.0 rewards prompt tokens. Between 0.0
            and 1.0 penalizes prompt tokens.
        encoder_input_ids (`torch.LongTensor`):
            The encoder_input_ids that should be repeated within the decoder ids.

    Examples:

    ```python
    >>> from transformers import AutoModelForCausalLM, AutoTokenizer

    >>> tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-560m")
    >>> model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-560m")

    >>> inputs = tokenizer(["Alice and Bob. The third member's name was"], return_tensors="pt")
    >>> gen_out = model.generate(**inputs)
    >>> print(tokenizer.batch_decode(gen_out, skip_special_tokens=True)[0])
    Alice and Bob. The third member's name was not mentioned.

    >>> # With the `encoder_repetition_penalty` argument we can trigger this logits processor in `generate`, which can
    >>> # promote the use of prompt tokens ("Bob" in this example)
    >>> gen_out = model.generate(**inputs, encoder_repetition_penalty=1.2)
    >>> print(tokenizer.batch_decode(gen_out, skip_special_tokens=True)[0])
    Alice and Bob. The third member's name was Bob. The third member's name was Bob.
    ```
    """

    def __init__(self, penalty: float, encoder_input_ids: torch.LongTensor):
        if not isinstance(penalty, float) or not (penalty > 0):
            raise ValueError(f"`penalty` has to be a strictly positive float, but is {penalty}")

        self.penalty = 1 / penalty
        self.encoder_input_ids = encoder_input_ids

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        score = torch.gather(scores, 1, self.encoder_input_ids)

        # if score < 0 then hallucination penalty has to be multiplied to increase the token probabilities
        score = torch.where(score < 0, score * self.penalty, score / self.penalty)

        scores_processed = scores.scatter(1, self.encoder_input_ids, score)
        return scores_processed


class TopPLogitsWarper(LogitsProcessor):
    """
    [`LogitsProcessor`] that performs top-p, i.e. restricting to top tokens summing to prob_cut_off <= prob_cut_off.
    Often used together with [`TemperatureLogitsWarper`] and [`TopKLogitsWarper`].

    Args:
        top_p (`float`):
            If set to < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or
            higher are kept for generation.
        filter_value (`float`, *optional*, defaults to -inf):
            All filtered values will be set to this float value.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered.

    Examples:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

    >>> set_seed(1)
    >>> model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
    >>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")

    >>> inputs = tokenizer("A sequence: 1, 2", return_tensors="pt")

    >>> # With sampling, the output is unexpected -- sometimes too unexpected.
    >>> outputs = model.generate(**inputs, do_sample=True)
    >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    A sequence: 1, 2, 3 | < 4 (left-hand pointer) ;
    <BLANKLINE>
    <BLANKLINE>

    >>> # With `top_p` sampling, the output gets restricted to high-probability tokens.
    >>> # Pro tip: In practice, LLMs use `top_p` in the 0.9-0.95 range.
    >>> outputs = model.generate(**inputs, do_sample=True, top_p=0.1)
    >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    A sequence: 1, 2, 3, 4, 5, 6, 7, 8, 9
    ```
    """

    def __init__(self, top_p: float, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        top_p = float(top_p)
        if top_p < 0 or top_p > 1.0:
            raise ValueError(f"`top_p` has to be a float > 0 and < 1, but is {top_p}")
        if not isinstance(min_tokens_to_keep, int) or (min_tokens_to_keep < 1):
            raise ValueError(f"`min_tokens_to_keep` has to be a positive integer, but is {min_tokens_to_keep}")

        self.top_p = top_p
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        sorted_logits, sorted_indices = torch.sort(scores, descending=False)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs <= (1 - self.top_p)
        # Keep at least min_tokens_to_keep
        sorted_indices_to_remove[..., -self.min_tokens_to_keep :] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        scores_processed = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores_processed


class TopKLogitsWarper(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that performs top-k, i.e. restricting to the k highest probability elements. Often used
    together with [`TemperatureLogitsWarper`] and [`TopPLogitsWarper`].

    Args:
        top_k (`int`):
            The number of highest probability vocabulary tokens to keep for top-k-filtering.
        filter_value (`float`, *optional*, defaults to -inf):
            All filtered values will be set to this float value.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered.

    Examples:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

    >>> set_seed(1)
    >>> model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
    >>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")

    >>> inputs = tokenizer("A sequence: A, B, C, D", return_tensors="pt")

    >>> # With sampling, the output is unexpected -- sometimes too unexpected.
    >>> outputs = model.generate(**inputs, do_sample=True)
    >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    A sequence: A, B, C, D, E — S — O, P — R

    >>> # With `top_k` sampling, the output gets restricted the k most likely tokens.
    >>> # Pro tip: In practice, LLMs use `top_k` in the 5-50 range.
    >>> outputs = model.generate(**inputs, do_sample=True, top_k=2)
    >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    A sequence: A, B, C, D, E, F, G, H, I
    ```
    """

    def __init__(self, top_k: int, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"`top_k` has to be a strictly positive integer, but is {top_k}")

        self.top_k = max(top_k, min_tokens_to_keep)
        self.filter_value = filter_value

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        top_k = min(self.top_k, scores.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
        scores_processed = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores_processed


class MinPLogitsWarper(LogitsProcessor):
    """
    [`LogitsProcessor`] that performs min-p, i.e. keeps all tokens that are above a minimum probability, scaled by the
    probability of the most likely token. As a result, the filter becomes more agressive in the presence of
    high-probability tokens, which is a sign of a confident output that we shouldn't deviate from.

    Often used together with [`TemperatureLogitsWarper`]. Used as an alternative to [`TopPLogitsWarper`] and
    [`TopKLogitsWarper`].

    Created by @menhguin and @kalomaze (github handles). Code adapted from [this external PR](https://github.com/oobabooga/text-generation-webui/pull/4449/files)

    Args:
        min_p (`float`):
            Minimum token probability, which will be scaled by the probability of the most likely token. It must be a
            value between 0 and 1. Typical values are in the 0.01-0.2 range, comparably selective as setting `top_p` in
            the 0.99-0.8 range (use the opposite of normal `top_p` values).
        filter_value (`float`, *optional*, defaults to -inf):
            All filtered values will be set to this float value.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered.

    Examples:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

    >>> set_seed(1)
    >>> model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
    >>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")

    >>> inputs = tokenizer("A sequence: 1, 2", return_tensors="pt")

    >>> # With sampling, the output is unexpected -- sometimes too unexpected.
    >>> outputs = model.generate(**inputs, do_sample=True)
    >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    A sequence: 1, 2, 3 | < 4 (left-hand pointer) ;
    <BLANKLINE>
    <BLANKLINE>

    >>> # With `min_p` sampling, the output gets restricted to high-probability tokens.
    >>> # Pro tip: In practice, LLMs use `min_p` in the 0.01-0.2 range.
    >>> outputs = model.generate(**inputs, do_sample=True, min_p=0.1)
    >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    A sequence: 1, 2, 3, 4, 5, 6, 7, 8, 9
    ```
    """

    def __init__(self, min_p: float, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        if not (0 <= min_p <= 1.0):
            raise ValueError(f"`min_p` has to be a float in the [0, 1] interval, but is {min_p}")
        if not isinstance(min_tokens_to_keep, int) or (min_tokens_to_keep < 1):
            raise ValueError(f"`min_tokens_to_keep` has to be a positive integer, but is {min_tokens_to_keep}")

        self.min_p = min_p
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Convert logits to probabilities
        probs = torch.softmax(scores, dim=-1)
        # Get the probability of the top token for each sequence in the batch
        top_probs, _ = probs.max(dim=-1, keepdim=True)
        # Calculate the actual min_p threshold by scaling min_p with the top token's probability
        scaled_min_p = self.min_p * top_probs
        # Create a mask for tokens that have a probability less than the scaled min_p
        tokens_to_remove = probs < scaled_min_p

        sorted_indices = torch.argsort(scores, descending=True, dim=-1)
        sorted_indices_to_remove = torch.gather(tokens_to_remove, dim=-1, index=sorted_indices)
        # Keep at least min_tokens_to_keep
        sorted_indices_to_remove[..., : self.min_tokens_to_keep] = False

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        scores_processed = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores_processed


class TypicalLogitsWarper(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that performs typical decoding. Inspired on how humans use language, it prioritizes tokens
    whose log probability is close to the entropy of the token probability distribution. This means that the most
    likely tokens may be discarded in the process.

    See [Typical Decoding for Natural Language Generation](https://arxiv.org/abs/2202.00666) for more information.

    Args:
        mass (`float`, *optional*, defaults to 0.9):
            Value of typical_p between 0 and 1 inclusive, defaults to 0.9.
        filter_value (`float`, *optional*, defaults to -inf):
            All filtered values will be set to this float value.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered.

    Examples:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

    >>> model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-560m")
    >>> tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-560m")

    >>> inputs = tokenizer("1, 2, 3", return_tensors="pt")

    >>> # We can see that greedy decoding produces a sequence of numbers
    >>> outputs = model.generate(**inputs)
    >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,

    >>> # For this particular seed, we can see that sampling produces nearly the same low-information (= low entropy)
    >>> # sequence
    >>> set_seed(18)
    >>> outputs = model.generate(**inputs, do_sample=True)
    >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    1, 2, 3, 4, 5, 6, 7, 8, 9 and 10

    >>> # With `typical_p` set, the most obvious sequence is no longer produced, which may be good for your problem
    >>> set_seed(18)
    >>> outputs = model.generate(
    ...     **inputs, do_sample=True, typical_p=0.1, return_dict_in_generate=True, output_scores=True
    ... )
    >>> print(tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)[0])
    1, 2, 3 and 5

    >>> # We can see that the token corresponding to "4" (token 934) in the second position, the most likely token
    >>> # as seen with greedy decoding, was entirely blocked out
    >>> print(outputs.scores[1][0, 934])
    tensor(-inf)
    ```
    """

    def __init__(self, mass: float = 0.9, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        mass = float(mass)
        if not (mass > 0 and mass < 1):
            raise ValueError(f"`typical_p` has to be a float > 0 and < 1, but is {mass}")
        if not isinstance(min_tokens_to_keep, int) or (min_tokens_to_keep < 1):
            raise ValueError(f"`min_tokens_to_keep` has to be a positive integer, but is {min_tokens_to_keep}")

        self.filter_value = filter_value
        self.mass = mass
        self.min_tokens_to_keep = min_tokens_to_keep

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # calculate entropy
        normalized = torch.nn.functional.log_softmax(scores, dim=-1)
        p = torch.exp(normalized)
        ent = -(normalized * p).nansum(-1, keepdim=True)

        # shift and sort
        shifted_scores = torch.abs((-normalized) - ent)
        sorted_scores, sorted_indices = torch.sort(shifted_scores, descending=False)
        sorted_logits = scores.gather(-1, sorted_indices)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        # Remove tokens with cumulative mass above the threshold
        last_ind = (cumulative_probs < self.mass).sum(dim=1)
        last_ind.clamp_(max=sorted_scores.shape[-1] - 1)
        sorted_indices_to_remove = sorted_scores > sorted_scores.gather(1, last_ind.view(-1, 1))
        sorted_indices_to_remove[..., : self.min_tokens_to_keep] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)

        scores_processed = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores_processed


class EpsilonLogitsWarper(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that performs epsilon-sampling, i.e. restricting to tokens with `prob >= epsilon`. Takes the
    largest min_tokens_to_keep tokens if no tokens satisfy this constraint. See [Truncation Sampling as Language Model
    Desmoothing](https://arxiv.org/abs/2210.15191) for more information.

    Args:
        epsilon (`float`):
            If set to > 0, only the most tokens with probabilities `epsilon` or higher are kept for generation.
        filter_value (`float`, *optional*, defaults to -inf):
            All filtered values will be set to this float value.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered.

    Examples:
    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

    >>> set_seed(1)
    >>> model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
    >>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")

    >>> inputs = tokenizer("A sequence: 1, 2", return_tensors="pt")

    >>> # With sampling, the output is unexpected -- sometimes too unexpected.
    >>> outputs = model.generate(**inputs, do_sample=True)
    >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    A sequence: 1, 2, 3 | < 4 (left-hand pointer) ;
    <BLANKLINE>
    <BLANKLINE>

    >>> # With epsilon sampling, the output gets restricted to high-probability tokens. Note that this is similar to
    >>> # Top P sampling, which restricts tokens based on their cumulative probability.
    >>> # Pro tip: The paper recomends using `epsilon_cutoff` values between 3e-4 and 9e-4
    >>> outputs = model.generate(**inputs, do_sample=True, epsilon_cutoff=0.1)
    >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    A sequence: 1, 2, 3, 4, 5, 6, 7, 8, 9
    ```
    """

    def __init__(self, epsilon: float, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        epsilon = float(epsilon)
        if epsilon <= 0 or epsilon >= 1:
            raise ValueError(f"`epsilon_cutoff` has to be a float > 0 and < 1, but is {epsilon}")

        min_tokens_to_keep = int(min_tokens_to_keep)
        if min_tokens_to_keep < 1:
            raise ValueError(
                f"`min_tokens_to_keep` has to be a strictly positive integer, but is {min_tokens_to_keep}"
            )

        self.epsilon = epsilon
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Determine which indices to remove
        probabilities = scores.softmax(dim=-1)
        indices_to_remove = probabilities < self.epsilon

        # Keep the words with the 'min_tokens_to_keep'-highest probabilities
        top_k = min(self.min_tokens_to_keep, scores.size(-1))  # Safety check
        indices_to_remove = indices_to_remove & (scores < torch.topk(scores, top_k)[0][..., -1, None])

        scores_processed = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores_processed


class EtaLogitsWarper(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that performs eta-sampling, a technique to filter out tokens with probabilities below a dynamic
    cutoff value, `eta`, which is calculated based on a combination of the hyperparameter `epsilon` and the entropy of
    the token probabilities, i.e. `eta := min(epsilon, sqrt(epsilon * e^-entropy(probabilities)))`. Takes the largest
    min_tokens_to_keep tokens if no tokens satisfy this constraint. It addresses the issue of poor quality in long
    samples of text generated by neural language models leading to more coherent and fluent text. See [Truncation
    Sampling as Language Model Desmoothing](https://arxiv.org/abs/2210.15191) for more information. Note: `do_sample`
    must be set to `True` for this `LogitsProcessor` to work.


    Args:
        epsilon (`float`):
            A float value in the range (0, 1). Hyperparameter used to calculate the dynamic cutoff value, `eta`. The
            suggested values from the paper ranges from 3e-4 to 4e-3 depending on the size of the model.
        filter_value (`float`, *optional*, defaults to -inf):
            All values that are found to be below the dynamic cutoff value, `eta`, are set to this float value. This
            parameter is useful when logits need to be modified for very low probability tokens that should be excluded
            from generation entirely.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Specifies the minimum number of tokens that must be kept for generation, regardless of their probabilities.
            For example, if `min_tokens_to_keep` is set to 1, at least one token will always be kept for generation,
            even if all tokens have probabilities below the cutoff `eta`.
        device (`str`, *optional*, defaults to `"cpu"`):
            The device to allocate the tensors.

    Examples:
    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

    >>> set_seed(1)
    >>> model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
    >>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")

    >>> inputs = tokenizer("A sequence: 1, 2", return_tensors="pt")

    >>> # With sampling, the output is unexpected -- sometimes too unexpected.
    >>> outputs = model.generate(**inputs, do_sample=True)
    >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    A sequence: 1, 2, 3 | < 4 (left-hand pointer) ;
    <BLANKLINE>
    <BLANKLINE>

    >>> # With eta sampling, the output gets restricted to high-probability tokens. You can see it as a dynamic form of
    >>> # epsilon sampling that adapts its cutoff probability based on the entropy (high entropy = lower cutoff).
    >>> # Pro tip: The paper recomends using `eta_cutoff` values between 3e-4 to 4e-3
    >>> outputs = model.generate(**inputs, do_sample=True, eta_cutoff=0.1)
    >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    A sequence: 1, 2, 3, 4, 5, 6, 7, 8, 9
    ```
    """

    def __init__(
        self, epsilon: float, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1, device: str = "cpu"
    ):
        epsilon = float(epsilon)
        if epsilon <= 0 or epsilon >= 1:
            raise ValueError(f"`eta_cutoff` has to be a float > 0 and < 1, but is {epsilon}")

        min_tokens_to_keep = int(min_tokens_to_keep)
        if min_tokens_to_keep < 1:
            raise ValueError(
                f"`min_tokens_to_keep` has to be a strictly positive integer, but is {min_tokens_to_keep}"
            )

        self.epsilon = torch.tensor(epsilon, device=device)
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        probabilities = scores.softmax(dim=-1)
        entropy = torch.distributions.Categorical(logits=scores).entropy()
        eta = torch.min(self.epsilon, torch.sqrt(self.epsilon) * torch.exp(-entropy))[..., None]
        indices_to_remove = probabilities < eta

        # Keep the words with the 'min_tokens_to_keep'-highest probabilities
        top_k = min(self.min_tokens_to_keep, scores.size(-1))  # Safety check
        indices_to_remove = indices_to_remove & (scores < torch.topk(scores, top_k)[0][..., -1, None])

        scores_processed = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores_processed


def _get_ngrams(ngram_size: int, prev_input_ids: torch.Tensor, num_hypos: int):
    """
    Assume ngram_size=2 and prev_input_ids=tensor([[40, 2883, 2712, 4346]]). The output of generated ngrams look like
    this {(40,): [2883], (2883,): [2712], (2712,): [4346]}.

    Args:
        ngram_size (`int`):
            The number sequential tokens taken as a group which may only occur once before being banned.
        prev_input_ids (`torch.Tensor`):
           Generated token ids for the current hypothesis.
        num_hypos (`int`):
            The number of hypotheses for which n-grams need to be generated.

    Returns:
        generated_ngrams (`dict`):
            Dictionary of generated ngrams.
    """
    # Initialize an empty list of dictionaries, one for each hypothesis (index) in the range of num_hypos
    generated_ngrams = [{} for _ in range(num_hypos)]
    for idx in range(num_hypos):
        gen_tokens = prev_input_ids[idx].tolist()
        generated_ngram = generated_ngrams[idx]
        # Loop through each n-gram of size ngram_size in the list of tokens (gen_tokens)
        for ngram in zip(*[gen_tokens[i:] for i in range(ngram_size)]):
            prev_ngram_tuple = tuple(ngram[:-1])
            generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]
    return generated_ngrams


def _get_generated_ngrams(banned_ngrams, prev_input_ids, ngram_size, cur_len):
    """
    Determines the banned tokens for the current hypothesis based on previously generated n-grams.

    Args:
        banned_ngrams (`dict`):
            A dictionary containing previously generated n-grams for each hypothesis.
        prev_input_ids (`torch.Tensor`):
            Generated token ids for the current hypothesis.
        ngram_size (`int`):
            The number sequential tokens taken as a group which may only occur once before being banned.
        cur_len (`int`):
            The current length of the token sequences for which the n-grams are being checked.

    Returns:
        List of tokens that are banned.
    """
    # Before decoding the next token, prevent decoding of ngrams that have already appeared
    start_idx = cur_len + 1 - ngram_size
    ngram_idx = tuple(prev_input_ids[start_idx:cur_len].tolist())
    return banned_ngrams.get(ngram_idx, [])


def _calc_banned_ngram_tokens(
    ngram_size: int, prev_input_ids: torch.Tensor, num_hypos: int, cur_len: int
) -> List[Iterable[int]]:
    """Copied from fairseq for no_repeat_ngram in beam_search"""
    if cur_len + 1 < ngram_size:
        # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
        return [[] for _ in range(num_hypos)]
    generated_ngrams = _get_ngrams(ngram_size, prev_input_ids, num_hypos)
    banned_tokens = [
        _get_generated_ngrams(generated_ngrams[hypo_idx], prev_input_ids[hypo_idx], ngram_size, cur_len)
        for hypo_idx in range(num_hypos)
    ]
    return banned_tokens


class NoRepeatNGramLogitsProcessor(LogitsProcessor):
    r"""
    N-grams are groups of "n" consecutive words, characters, or tokens taken from a sequence of text. Given the
    sentence: "She runs fast", the bi-grams (n=2) would be ("she", "runs") and ("runs", "fast"). In text generation,
    avoiding repetitions of word sequences provides a more diverse output. This [`LogitsProcessor`] enforces no
    repetition of n-grams by setting the scores of banned tokens to negative infinity which eliminates those tokens
    from consideration when further processing the scores. Note that, for decoder-only models like most LLMs, the
    prompt is also considered to obtain the n-grams.
    [Fairseq](https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345).

    <Tip>

    Use n-gram penalties with care. For instance, penalizing 2-grams (bigrams) in an article about the city of New York
    might lead to undesirable outcomes where the city's name appears only once in the entire text.
    [Reference](https://huggingface.co/blog/how-to-generate)

    </Tip>

    Args:
        ngram_size (`int`):
            All ngrams of size `ngram_size` can only occur once.

    Examples:

    ```py
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM

    >>> model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
    >>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
    >>> inputs = tokenizer(["Today I"], return_tensors="pt")

    >>> output = model.generate(**inputs)
    >>> print(tokenizer.decode(output[0], skip_special_tokens=True))
    Today I’m not sure if I’m going to be able to do it.

    >>> # Now let's add ngram size using `no_repeat_ngram_size`. This stops the repetitions ("I’m") in the output.
    >>> output = model.generate(**inputs, no_repeat_ngram_size=2)
    >>> print(tokenizer.decode(output[0], skip_special_tokens=True))
    Today I’m not sure if I can get a better understanding of the nature of this issue
    ```
    """

    def __init__(self, ngram_size: int):
        if not isinstance(ngram_size, int) or ngram_size <= 0:
            raise ValueError(f"`ngram_size` has to be a strictly positive integer, but is {ngram_size}")
        self.ngram_size = ngram_size

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        num_batch_hypotheses = scores.shape[0]
        cur_len = input_ids.shape[-1]
        scores_processed = scores.clone()
        banned_batch_tokens = _calc_banned_ngram_tokens(self.ngram_size, input_ids, num_batch_hypotheses, cur_len)
        for i, banned_tokens in enumerate(banned_batch_tokens):
            scores_processed[i, banned_tokens] = -float("inf")

        return scores_processed


class EncoderNoRepeatNGramLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that works similarly to [`NoRepeatNGramLogitsProcessor`], but applied exclusively to prevent
    the repetition of n-grams present in the prompt.

    It was designed to promote chattiness in a language model, by preventing the generation of n-grams present in
    previous conversation rounds.

    Args:
        encoder_ngram_size (`int`):
            All ngrams of size `ngram_size` can only occur within the encoder input ids.
        encoder_input_ids (`int`):
            The encoder_input_ids that should not be repeated within the decoder ids.

    Examples:

    ```py
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM

    >>> model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-560m")
    >>> tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-560m")

    >>> inputs = tokenizer("Alice: I love cats. What do you love?\nBob:", return_tensors="pt")

    >>> # With greedy decoding, we see Bob repeating Alice's opinion. If Bob was a chatbot, it would be a poor one.
    >>> outputs = model.generate(**inputs)
    >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    Alice: I love cats. What do you love?
    Bob: I love cats. What do you

    >>> # With this logits processor, we can prevent Bob from repeating Alice's opinion.
    >>> outputs = model.generate(**inputs, encoder_no_repeat_ngram_size=2)
    >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    Alice: I love cats. What do you love?
    Bob: My cats are very cute.
    ```
    """

    def __init__(self, encoder_ngram_size: int, encoder_input_ids: torch.LongTensor):
        if not isinstance(encoder_ngram_size, int) or encoder_ngram_size <= 0:
            raise ValueError(
                f"`encoder_ngram_size` has to be a strictly positive integer, but is {encoder_ngram_size}"
            )
        self.ngram_size = encoder_ngram_size
        if len(encoder_input_ids.shape) == 1:
            encoder_input_ids = encoder_input_ids.unsqueeze(0)
        self.batch_size = encoder_input_ids.shape[0]
        self.generated_ngrams = _get_ngrams(encoder_ngram_size, encoder_input_ids, self.batch_size)

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # B x num_beams
        num_hypos = scores.shape[0]
        num_beams = num_hypos // self.batch_size
        cur_len = input_ids.shape[-1]
        scores_processed = scores.clone()
        banned_batch_tokens = [
            _get_generated_ngrams(
                self.generated_ngrams[hypo_idx // num_beams], input_ids[hypo_idx], self.ngram_size, cur_len
            )
            for hypo_idx in range(num_hypos)
        ]

        for i, banned_tokens in enumerate(banned_batch_tokens):
            scores_processed[i, banned_tokens] = -float("inf")

        return scores_processed


class SequenceBiasLogitsProcessor(LogitsProcessor):
    """
    [`LogitsProcessor`] that applies an additive bias on sequences. The bias is applied to the last token of a sequence
    when the next generated token can complete it. Consequently, to take the most of biasing sequences with more than
    one token, consider using beam methods (to gracefully work around partially completed sequences that have a
    negative bias) and applying the bias to their prefixes (to ensure the bias is applied earlier).

    <Tip>

    In order to get the token ids of the sequences that you want to bias, make sure to set `add_prefix_space=True` when
    initializing the tokenizer, and use `tokenizer(bad_words, add_special_tokens=False).input_ids`. The
    `add_prefix_space` argument is only supported for some slow tokenizers, as fast tokenizers' prefixing behaviours
    come from `pre tokenizers`. Read more [here](https://huggingface.co/docs/tokenizers/api/pre-tokenizers).

    </Tip>

    Args:
        sequence_bias (`List[List[Union[List[int], float]]]`):
            List of lists that maps a sequence of tokens to its bias term (e.g. `[[[10, 45], -2.0],
            [[64], -7.5]]`). Positive biases increase the odds of the
            sequence being selected, while negative biases do the opposite. If a sequence has a length of 1, its bias
            will always be applied. Otherwise, the bias will only be applied if the sequence in question is about to be
            completed (in the token selection step after this processor is applied).

    Examples:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM

    >>> model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    >>> tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    >>> inputs = tokenizer(["The full name of Donald is Donald"], return_tensors="pt")

    >>> summary_ids = model.generate(inputs["input_ids"], max_new_tokens=4)
    >>> print(tokenizer.batch_decode(summary_ids, skip_special_tokens=True)[0])
    The full name of Donald is Donald J. Trump Jr

    >>> # Now let's control generation through a bias. Please note that the tokenizer is initialized differently!
    >>> tokenizer_with_prefix_space = AutoTokenizer.from_pretrained("openai-community/gpt2", add_prefix_space=True)


    >>> def get_tokens(word):
    ...     return tokenizer_with_prefix_space([word], add_special_tokens=False).input_ids[0]


    >>> # If we add a negative bias without beam search, it may become "stuck" in a prefix without good continuations
    >>> sequence_bias = [get_tokens("Trump"), -10.0]
    >>> biased_ids = model.generate(inputs["input_ids"], max_new_tokens=4, sequence_bias=sequence_bias)
    >>> print(tokenizer.batch_decode(biased_ids, skip_special_tokens=True)[0])
    The full name of Donald is Donald J. Donald,

    >>> biased_ids = model.generate(inputs["input_ids"], max_new_tokens=4, num_beams=4, sequence_bias=sequence_bias)
    >>> print(tokenizer.batch_decode(biased_ids, skip_special_tokens=True)[0])
    The full name of Donald is Donald Rumsfeld,

    >>> # We can also add a positive bias to nudge the model towards specific tokens or continuations
    >>> sequence_bias = [get_tokens("Donald Duck"), 10.0]
    >>> biased_ids = model.generate(inputs["input_ids"], max_new_tokens=4, num_beams=4, sequence_bias=sequence_bias)
    >>> print(tokenizer.batch_decode(biased_ids, skip_special_tokens=True)[0])
    The full name of Donald is Donald Duck.
    ```
    """

    def __init__(self, sequence_bias: List[List[Union[List[int], float]]]):
        self.sequence_bias = sequence_bias
        self._validate_arguments()
        self._convert_list_arguments_into_dict()

        # Bias variables that will be populated on the first call (for retrocompatibility purposes, the vocabulary size
        # is infered in the first usage, which inhibits initializing here)
        self.length_1_bias = None
        self.prepared_bias_variables = False

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 1 - Prepares the bias tensors. This is only needed the first time the logit processor is called.
        if not self.prepared_bias_variables:
            self._prepare_bias_variables(scores)

        # 2 - prepares an empty bias to add
        bias = torch.zeros_like(scores)

        # 3 - include the bias from length = 1
        bias += self.length_1_bias

        # 4 - include the bias from length > 1, after determining which biased sequences may be completed.
        for sequence_ids, sequence_bias in self.sequence_bias.items():
            if len(sequence_ids) == 1:  # the sequence is of length 1, already applied
                continue
            if len(sequence_ids) > input_ids.shape[1]:  # the sequence is longer than the context, ignore
                continue
            prefix_length = len(sequence_ids) - 1
            last_token = sequence_ids[-1]
            matching_rows = torch.eq(
                input_ids[:, -prefix_length:],
                torch.tensor(sequence_ids[:-1], dtype=input_ids.dtype, device=input_ids.device),
            ).prod(dim=1)
            bias[:, last_token] += torch.where(
                matching_rows.bool(),
                torch.tensor(sequence_bias, device=input_ids.device),
                torch.tensor(0.0, device=input_ids.device),
            )

        # 5 - apply the bias to the scores
        scores_processed = scores + bias
        return scores_processed

    def _prepare_bias_variables(self, scores: torch.FloatTensor):
        vocabulary_size = scores.shape[-1]

        # Check biased tokens out of bounds
        invalid_biases = []
        for sequence_ids in self.sequence_bias:
            for token_id in sequence_ids:
                if token_id >= vocabulary_size:
                    invalid_biases.append(token_id)
        if len(invalid_biases) > 0:
            raise ValueError(
                f"The model vocabulary size is {vocabulary_size}, but the following tokens were being biased: "
                f"{invalid_biases}"
            )

        # Precompute the bias tensors to be applied. Sequences of length 1 are kept separately, as they can be applied
        # with simpler logic.
        self.length_1_bias = torch.zeros((vocabulary_size,), dtype=torch.float).to(scores.device)
        for sequence_ids, bias in self.sequence_bias.items():
            if len(sequence_ids) == 1:
                self.length_1_bias[sequence_ids[-1]] = bias

        self.prepared_bias_variables = True

    def _validate_arguments(self):
        sequence_bias = self.sequence_bias
        if not isinstance(sequence_bias, dict) and not isinstance(sequence_bias, list) or len(sequence_bias) == 0:
            raise ValueError(
                f"`sequence_bias` has to be a non-empty dictionary, or non-empty list of lists but is {sequence_bias}."
            )
        if isinstance(sequence_bias, dict) and any(
            not isinstance(sequence_ids, tuple) for sequence_ids in sequence_bias.keys()
        ):
            raise ValueError(f"`sequence_bias` has to be a dict with tuples as keys, but is {sequence_bias}.")
        if isinstance(sequence_bias, dict) and any(
            any((not isinstance(token_id, (int, np.integer)) or token_id < 0) for token_id in sequence_ids)
            or len(sequence_ids) == 0
            for sequence_ids in sequence_bias.keys()
        ):
            raise ValueError(
                f"Each key in `sequence_bias` has to be a non-empty tuple of positive integers, but is "
                f"{sequence_bias}."
            )

        def all_token_bias_pairs_are_valid(sequence):
            return (
                isinstance(sequence[0], list)
                and all(isinstance(token_id, (int, np.integer)) and token_id > 0 for token_id in sequence[0])
                and isinstance(sequence[1], float)
            )

        if isinstance(sequence_bias, list) and any(
            (not all_token_bias_pairs_are_valid(sequence)) or len(sequence) == 0 for sequence in sequence_bias
        ):
            raise ValueError(
                f"Each element in `sequence_bias` has to be a non-empty list of lists of positive integers and float, but is "
                f"{sequence_bias}."
            )
        if isinstance(sequence_bias, dict) and any(not isinstance(bias, float) for bias in sequence_bias.values()):
            raise ValueError(f"`sequence_bias` has to be a dict with floats as values, but is {sequence_bias}.")

    def _convert_list_arguments_into_dict(self):
        """BC: we used to accept `dict{tuple of tokens: float}` directly, now we expect a list"""
        if isinstance(self.sequence_bias, list):
            temp_sequence = self.sequence_bias
            self.sequence_bias = {tuple(sublist[0]): sublist[1] for sublist in temp_sequence}


class NoBadWordsLogitsProcessor(SequenceBiasLogitsProcessor):
    """
    [`LogitsProcessor`] that enforces that specified sequences will never be selected.

    <Tip>

    In order to get the token ids of the words that should not appear in the generated text, make sure to set
    `add_prefix_space=True` when initializing the tokenizer, and use `tokenizer(bad_words,
    add_special_tokens=False).input_ids`. The `add_prefix_space` argument is only supported for some slow tokenizers,
    as fast tokenizers' prefixing behaviours come from `pre tokenizers`. Read more
    [here](https://huggingface.co/docs/tokenizers/api/pre-tokenizers).

    </Tip>

    Args:
        bad_words_ids (`List[List[int]]`):
            List of list of token ids that are not allowed to be generated.
        eos_token_id (`Union[int, List[int], torch.Tensor]`, *optional*):
            The id(s) of the *end-of-sequence* token.

    Examples:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM

    >>> model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    >>> tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    >>> inputs = tokenizer(["In a word, the cake is a"], return_tensors="pt")

    >>> output_ids = model.generate(inputs["input_ids"], max_new_tokens=5, pad_token_id=tokenizer.eos_token_id)
    >>> print(tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0])
    In a word, the cake is a bit of a mess.

    >>> # Now let's take the bad words out. Please note that the tokenizer is initialized differently
    >>> tokenizer_with_prefix_space = AutoTokenizer.from_pretrained("openai-community/gpt2", add_prefix_space=True)


    >>> def get_tokens_as_list(word_list):
    ...     "Converts a sequence of words into a list of tokens"
    ...     tokens_list = []
    ...     for word in word_list:
    ...         tokenized_word = tokenizer_with_prefix_space([word], add_special_tokens=False).input_ids[0]
    ...         tokens_list.append(tokenized_word)
    ...     return tokens_list


    >>> bad_words_ids = get_tokens_as_list(word_list=["mess"])
    >>> output_ids = model.generate(
    ...     inputs["input_ids"], max_new_tokens=5, bad_words_ids=bad_words_ids, pad_token_id=tokenizer.eos_token_id
    ... )
    >>> print(tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0])
    In a word, the cake is a bit of a surprise.
    ```
    """

    def __init__(
        self, bad_words_ids: List[List[int]], eos_token_id: Optional[Union[int, List[int], torch.Tensor]] = None
    ):
        self.bad_word_ids = bad_words_ids
        self._validate_arguments()

        # Filter EOS token from bad_words_ids
        if eos_token_id is not None:
            if not isinstance(eos_token_id, torch.Tensor):
                if isinstance(eos_token_id, int):
                    eos_token_id = [eos_token_id]
                eos_token_id = torch.tensor(eos_token_id)

            bad_words_ids = list(
                filter(lambda bad_token_seq: all(bad_token_seq != [i] for i in eos_token_id), bad_words_ids)
            )

        # Forbidding a sequence is equivalent to setting its bias to -inf
        sequence_bias = {tuple(sequence): float("-inf") for sequence in bad_words_ids}
        super().__init__(sequence_bias=sequence_bias)

    def _validate_arguments(self):
        bad_words_ids = self.bad_word_ids
        if not isinstance(bad_words_ids, list) or len(bad_words_ids) == 0:
            raise ValueError(f"`bad_words_ids` has to be a non-empty list, but is {bad_words_ids}.")
        if any(not isinstance(bad_word_ids, list) for bad_word_ids in bad_words_ids):
            raise ValueError(f"`bad_words_ids` has to be a list of lists, but is {bad_words_ids}.")
        if any(
            any((not isinstance(token_id, (int, np.integer)) or token_id < 0) for token_id in bad_word_ids)
            for bad_word_ids in bad_words_ids
        ):
            raise ValueError(
                f"Each list in `bad_words_ids` has to be a list of positive integers, but is {bad_words_ids}."
            )


class PrefixConstrainedLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that enforces constrained generation and is useful for prefix-conditioned constrained
    generation. See [Autoregressive Entity Retrieval](https://arxiv.org/abs/2010.00904) for more information.

    Args:
        prefix_allowed_tokens_fn (`Callable[[int, torch.Tensor], List[int]]`):
            This function constraints the beam search to allowed tokens only at each step. This function takes 2
            arguments `inputs_ids` and the batch ID `batch_id`. It has to return a list with the allowed tokens for the
            next generation step conditioned on the previously generated tokens `inputs_ids` and the batch ID
            `batch_id`.

    Examples:

    ```py
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM

    >>> model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-560m")
    >>> tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-560m")

    >>> inputs = tokenizer("Alice and Bob", return_tensors="pt")

    >>> # By default, it continues generating according to the model's logits
    >>> outputs = model.generate(**inputs, max_new_tokens=5)
    >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    Alice and Bob are friends

    >>> # We can contrain it with `prefix_allowed_tokens_fn` to force a certain behavior based on a prefix.
    >>> # For instance, we can force an entire entity to be generated when its beginning is detected.
    >>> entity = tokenizer(" Bob Marley", return_tensors="pt").input_ids[0]  # 3 tokens
    >>> def prefix_allowed_tokens_fn(batch_id, input_ids):
    ...     '''
    ...     Attempts to generate 'Bob Marley' when 'Bob' is detected.
    ...     In this case, `batch_id` is not used, but you can set rules for each batch member.
    ...     '''
    ...     if input_ids[-1] == entity[0]:
    ...         return [entity[1].item()]
    ...     elif input_ids[-2] == entity[0] and input_ids[-1] == entity[1]:
    ...         return [entity[2].item()]
    ...     return list(range(tokenizer.vocab_size))  # If no match, allow all tokens

    >>> outputs = model.generate(**inputs, max_new_tokens=5, prefix_allowed_tokens_fn=prefix_allowed_tokens_fn)
    >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    Alice and Bob Marley
    ```
    """

    def __init__(self, prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]], num_beams: int):
        self._prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
        self._num_beams = num_beams

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.full_like(scores, -math.inf)
        for batch_id, beam_sent in enumerate(input_ids.view(-1, self._num_beams, input_ids.shape[-1])):
            for beam_id, sent in enumerate(beam_sent):
                prefix_allowed_tokens = self._prefix_allowed_tokens_fn(batch_id, sent)
                if len(prefix_allowed_tokens) == 0:
                    raise ValueError(
                        f"`prefix_allowed_tokens_fn` returned an empty list for batch ID {batch_id}."
                        f"This means that the constraint is unsatisfiable. Please check your implementation"
                        f"of `prefix_allowed_tokens_fn` "
                    )
                mask[batch_id * self._num_beams + beam_id, prefix_allowed_tokens] = 0

        scores_processed = scores + mask
        return scores_processed


class HammingDiversityLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that enforces diverse beam search.

    Note that this logits processor is only effective for [`PreTrainedModel.group_beam_search`]. See [Diverse Beam
    Search: Decoding Diverse Solutions from Neural Sequence Models](https://arxiv.org/pdf/1610.02424.pdf) for more
    details.

    Traditional beam search often generates very similar sequences across different beams.
    `HammingDiversityLogitsProcessor` addresses this by penalizing beams that generate tokens already chosen by other
    beams in the same time step.

    Args:
        diversity_penalty (`float`):
            This value is subtracted from a beam's score if it generates a token same as any beam from other group at a
            particular time. A higher `diversity_penalty` will enforce greater diversity among the beams. Adjusting
            this value can help strike a balance between diversity and natural likelihood.
        num_beams (`int`):
            Number of beams for beam search. 1 means no beam search.
        num_beam_groups (`int`):
            Number of groups to divide `num_beams` into in order to ensure diversity among different groups of beams.
            [this paper](https://arxiv.org/pdf/1610.02424.pdf) for more details.

    Examples:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    >>> import torch

    >>> # Initialize the model and tokenizer
    >>> tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
    >>> model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-base")

    >>> # A long text about the solar system
    >>> text = (
    ...     "The Solar System is a gravitationally bound system comprising the Sun and the objects that orbit it, "
    ...     "either directly or indirectly. Of the objects that orbit the Sun directly, the largest are the eight "
    ...     "planets, with the remainder being smaller objects, such as the five dwarf planets and small Solar System "
    ...     "bodies. The Solar System formed 4.6 billion years ago from the gravitational collapse of a giant "
    ...     "interstellar molecular cloud."
    ... )
    >>> inputs = tokenizer("summarize: " + text, return_tensors="pt")

    >>> # Generate diverse summary
    >>> outputs_diverse = model.generate(
    ...     **inputs,
    ...     num_beam_groups=2,
    ...     diversity_penalty=10.0,
    ...     max_length=100,
    ...     num_beams=4,
    ...     num_return_sequences=2,
    ... )
    >>> summaries_diverse = tokenizer.batch_decode(outputs_diverse, skip_special_tokens=True)

    >>> # Generate non-diverse summary
    >>> outputs_non_diverse = model.generate(
    ...     **inputs,
    ...     max_length=100,
    ...     num_beams=4,
    ...     num_return_sequences=2,
    ... )
    >>> summary_non_diverse = tokenizer.batch_decode(outputs_non_diverse, skip_special_tokens=True)

    >>> # With `diversity_penalty`, the resulting beams are much more diverse
    >>> print(summary_non_diverse)
    ['the solar system formed 4.6 billion years ago from the collapse of a giant interstellar molecular cloud. of the objects that orbit the Sun directly, the largest are the eight planets.',
    'the Solar System formed 4.6 billion years ago from the collapse of a giant interstellar molecular cloud. of the objects that orbit the Sun directly, the largest are the eight planets.']

    >>> print(summaries_diverse)
    ['the solar system formed 4.6 billion years ago from the collapse of a giant interstellar molecular cloud. of the objects that orbit the Sun directly, the largest are the eight planets.',
    'the solar system formed 4.6 billion years ago from the collapse of a giant interstellar molecular cloud. of the objects that orbit the Sun directly, the largest are the eight planets. the rest of the objects are smaller objects, such as the five dwarf planets and small solar system bodies.']
    ```
    """

    def __init__(self, diversity_penalty: float, num_beams: int, num_beam_groups: int):
        if not isinstance(diversity_penalty, float) or (not diversity_penalty > 0.0):
            raise ValueError("`diversity_penalty` should be a float strictly larger than 0.")
        self._diversity_penalty = diversity_penalty
        if not isinstance(num_beams, int) or num_beams < 2:
            raise ValueError("`num_beams` should be an integer strictly larger than 1.")
        self._num_beams = num_beams
        if not isinstance(num_beam_groups, int) or num_beam_groups < 2:
            raise ValueError("`num_beam_groups` should be an integer strictly larger than 1.")
        if num_beam_groups > num_beams:
            raise ValueError("`beam_groups` has to be smaller or equal to `num_beams`.")
        self._num_sub_beams = num_beams // num_beam_groups

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        current_tokens: torch.LongTensor,
        beam_group_idx: int,
    ) -> torch.FloatTensor:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
            scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`):
                Prediction scores of a language modeling head. These can be logits for each vocabulary when not using
                beam search or log softmax for each vocabulary token when using beam search
            current_tokens (`torch.LongTensor` of shape `(batch_size)`):
                Indices of input sequence tokens in the vocabulary, corresponding to the tokens selected by the other
                beam groups in the current generation step.
            beam_group_idx (`int`):
                The index of the beam group currently being processed.

        Return:
            `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`:
                The processed prediction scores.
        """
        # hamming diversity: penalise using same token in current group which was used in previous groups at
        # the same time step
        batch_size = current_tokens.shape[0] // self._num_beams
        group_start_idx = beam_group_idx * self._num_sub_beams
        group_end_idx = min(group_start_idx + self._num_sub_beams, self._num_beams)
        group_size = group_end_idx - group_start_idx
        vocab_size = scores.shape[-1]

        if group_start_idx == 0:
            return scores

        scores_processed = scores.clone()
        for batch_idx in range(batch_size):
            # predicted tokens of last time step of previous groups
            previous_group_tokens = current_tokens[
                batch_idx * self._num_beams : batch_idx * self._num_beams + group_start_idx
            ]
            token_frequency = torch.bincount(previous_group_tokens, minlength=vocab_size).to(scores.device)
            scores_processed[batch_idx * group_size : (batch_idx + 1) * group_size] -= (
                self._diversity_penalty * token_frequency
            )

        return scores_processed


class ForcedBOSTokenLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that enforces the specified token as the first generated token. Used with encoder-decoder
    models.

    Args:
        bos_token_id (`int`):
            The id of the token to force as the first generated token.

    Examples:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    >>> model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    >>> tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

    >>> inputs = tokenizer("Translate from English to German: I love cats.", return_tensors="pt")

    >>> # By default, it continues generating according to the model's logits
    >>> outputs = model.generate(**inputs, max_new_tokens=10)
    >>> print(tokenizer.batch_decode(outputs)[0])
    <pad> Ich liebe Kitty.</s>

    >>> # We can use `forced_bos_token_id` to force the start of generation with an encoder-decoder model
    >>> # (including forcing it to end straight away with an EOS token)
    >>> outputs = model.generate(**inputs, max_new_tokens=10, forced_bos_token_id=tokenizer.eos_token_id)
    >>> print(tokenizer.batch_decode(outputs)[0])
    <pad></s>
    ```
    """

    def __init__(self, bos_token_id: int):
        self.bos_token_id = bos_token_id

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        cur_len = input_ids.shape[-1]
        scores_processed = scores
        if cur_len == 1:
            scores_processed = torch.full_like(scores, -math.inf)
            scores_processed[:, self.bos_token_id] = 0
        return scores_processed


class ForcedEOSTokenLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that enforces the specified token as the last generated token when `max_length` is reached.

    Args:
        max_length (`int`):
            The maximum length of the sequence to be generated.
        eos_token_id (`Union[int, List[int], torch.Tensor]`):
            The id(s) of the *end-of-sequence* token.
        device (`str`, *optional*, defaults to `"cpu"`):
            The device to allocate the tensors.

    Examples:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM

    >>> model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
    >>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")

    >>> inputs = tokenizer("A sequence: 1, 2, 3", return_tensors="pt")

    >>> # By default, it continues generating according to the model's logits
    >>> outputs = model.generate(**inputs, max_new_tokens=10)
    >>> print(tokenizer.batch_decode(outputs)[0])
    A sequence: 1, 2, 3, 4, 5, 6, 7, 8

    >>> # `forced_eos_token_id` ensures the generation ends with a EOS token
    >>> outputs = model.generate(**inputs, max_new_tokens=10, forced_eos_token_id=tokenizer.eos_token_id)
    >>> print(tokenizer.batch_decode(outputs)[0])
    A sequence: 1, 2, 3, 4, 5, 6, 7,<|endoftext|>
    ```
    """

    def __init__(self, max_length: int, eos_token_id: Union[int, List[int], torch.Tensor], device: str = "cpu"):
        self.max_length = max_length

        if not isinstance(eos_token_id, torch.Tensor):
            if isinstance(eos_token_id, int):
                eos_token_id = [eos_token_id]
            eos_token_id = torch.tensor(eos_token_id, device=device)
        self.eos_token_id = eos_token_id

        if torch.is_floating_point(eos_token_id) or (eos_token_id < 0).any():
            raise ValueError(f"`eos_token_id` has to be a list of positive integers, but is {eos_token_id}")

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        cur_len = input_ids.shape[-1]
        scores_processed = scores
        if cur_len == self.max_length - 1:
            scores_processed = torch.full_like(scores, -math.inf)
            scores_processed[:, self.eos_token_id] = 0
        return scores_processed


class InfNanRemoveLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that removes all `nan` and `inf` values to avoid the generation method to fail. Note that using
    the logits processor should only be used if necessary since it can slow down the generation method.

    This logits processor has no `generate` example, as there shouldn't be a correct combination of flags that warrants
    its use.
    """

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # set all nan values to 0.0
        scores_processed = torch.where(scores != scores, 0.0, scores)

        # set all +/-inf values to max/min possible value
        scores_processed = torch.where(scores == float("inf"), torch.finfo(scores.dtype).max, scores_processed)
        scores_processed = torch.where(scores == -float("inf"), torch.finfo(scores.dtype).min, scores_processed)

        return scores_processed


class ExponentialDecayLengthPenalty(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that exponentially increases the score of the `eos_token_id` after `start_index` has been
    reached. This allows generating shorter sequences without having a hard cutoff, allowing the `eos_token` to be
    predicted in a meaningful position.

    Args:
        exponential_decay_length_penalty (`tuple(int, float)`):
            This tuple shall consist of: `(start_index, decay_factor)` where `start_index` indicates where penalty
            starts and `decay_factor` represents the factor of exponential decay
        eos_token_id (`Union[int, List[int], torch.Tensor]`):
            The id(s) of the *end-of-sequence* token.
        input_ids_seq_length (`int`):
            The length of the input sequence.

    Examples:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

    >>> model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    >>> tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

    >>> text = "Just wanted to let you know, I"
    >>> inputs = tokenizer(text, return_tensors="pt")

    >>> # Let's consider that we want short sentences, so we limit `max_length=30`. However, we observe that the answer
    >>> # tends to end abruptly.
    >>> set_seed(1)
    >>> outputs = model.generate(**inputs, do_sample=True, temperature=0.9, max_length=30, pad_token_id=50256)
    >>> print(tokenizer.batch_decode(outputs)[0])
    Just wanted to let you know, I received a link to an ebook, the book How To Start A Social Network which was
    published in 2010. Although

    >>> # To promote the appearance of the EOS token at the right time, we add the `exponential_decay_length_penalty =
    >>> # (start_index, decay_factor)`. Instead of cutting at max_tokens, the output comes to an end before and usually
    >>> # with more meaning. What happens is that starting from `start_index` the EOS token score will be increased
    >>> # by `decay_factor` exponentially. However, if you set a high decay factor, you may also end up with abruptly
    >>> # ending sequences.
    >>> set_seed(1)
    >>> outputs = model.generate(
    ...     **inputs,
    ...     do_sample=True,
    ...     temperature=0.9,
    ...     max_length=30,
    ...     pad_token_id=50256,
    ...     exponential_decay_length_penalty=(15, 1.6),
    ... )
    >>> print(tokenizer.batch_decode(outputs)[0])
    Just wanted to let you know, I received a link to an ebook, the book How To Start A Social Network
    which<|endoftext|>

    >>> # With a small decay factor, you will have a higher chance of getting a meaningful sequence.
    >>> set_seed(1)
    >>> outputs = model.generate(
    ...     **inputs,
    ...     do_sample=True,
    ...     temperature=0.9,
    ...     max_length=30,
    ...     pad_token_id=50256,
    ...     exponential_decay_length_penalty=(15, 1.01),
    ... )
    >>> print(tokenizer.batch_decode(outputs)[0])
    Just wanted to let you know, I received a link to an ebook, the book How To Start A Social Network which was
    published in 2010.<|endoftext|>
    ```
    """

    def __init__(
        self,
        exponential_decay_length_penalty: Tuple[int, float],
        eos_token_id: Union[int, List[int], torch.Tensor],
        input_ids_seq_length: int,
    ):
        self.regulation_start = exponential_decay_length_penalty[0] + input_ids_seq_length
        self.regulation_factor = exponential_decay_length_penalty[1]

        if not isinstance(eos_token_id, torch.Tensor):
            if isinstance(eos_token_id, int):
                eos_token_id = [eos_token_id]
            eos_token_id = torch.tensor(eos_token_id)
        self.eos_token_id = eos_token_id

        if torch.is_floating_point(eos_token_id) or (eos_token_id < 0).any():
            raise ValueError(f"`eos_token_id` has to be a list of positive integers, but is {eos_token_id}")

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        cur_len = input_ids.shape[-1]
        self.eos_token_id = self.eos_token_id.to(scores.device)
        penalties = torch.zeros_like(scores)
        scores_processed = scores
        if cur_len > self.regulation_start:
            penalty_idx = cur_len - self.regulation_start
            # To support negative logits we compute the penalty of the absolute value and add to the original logit
            penalty = torch.abs(scores[:, self.eos_token_id]) * (pow(self.regulation_factor, penalty_idx) - 1)
            penalties[:, self.eos_token_id] = penalty
            scores_processed = scores + penalties
        return scores_processed


class LogitNormalization(LogitsProcessor):
    r"""
    [`LogitsProcessor`] for normalizing the scores using log-softmax. It's important to normalize
    the scores during beam search, after applying the logits processors or warpers, since the search algorithm used in
    this library doesn't do it (it only does it before, but they may need re-normalization) but it still supposes that
    the scores are normalized when comparing the hypotheses.

    Examples:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM
    >>> import torch

    >>> model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
    >>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")

    >>> inputs = tokenizer("A sequence: 1, 2, 3", return_tensors="pt")

    >>> # By default, the scores are not normalized -- the sum of their exponentials is NOT a normalized probability
    >>> # distribution, summing to 1
    >>> outputs = model.generate(**inputs, return_dict_in_generate=True, output_scores=True)
    >>> print(torch.allclose(torch.sum(torch.exp(outputs.scores[-1])), torch.Tensor((1.000,)), rtol=1e-4))
    False

    >>> # Normalizing them may have a positive impact on beam methods, or when using the scores on your application
    >>> outputs = model.generate(**inputs, renormalize_logits=True, return_dict_in_generate=True, output_scores=True)
    >>> print(torch.allclose(torch.sum(torch.exp(outputs.scores[-1])), torch.Tensor((1.000,)), rtol=1e-4))
    True
    ```
    """

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores_processed = scores.log_softmax(dim=-1)
        return scores_processed


class SuppressTokensAtBeginLogitsProcessor(LogitsProcessor):
    r"""
    [`SuppressTokensAtBeginLogitsProcessor`] supresses a list of tokens as soon as the `generate` function starts
    generating using `begin_index` tokens. This should ensure that the tokens defined by `begin_suppress_tokens` are
    not generated at the beginning. Originally created for
    [Whisper](https://huggingface.co/docs/transformers/model_doc/whisper).

    Examples:

    ```python
    >>> from transformers import AutoProcessor, WhisperForConditionalGeneration
    >>> from datasets import load_dataset

    >>> processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")
    >>> model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
    >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    >>> inputs = processor(ds[0]["audio"]["array"], return_tensors="pt")

    >>> # Whisper has `begin_suppress_tokens` set by default (= `[220, 50256]`). 50256 is the EOS token, so this means
    >>> # it can't generate and EOS token in the first iteration, but it can in the others.
    >>> outputs = model.generate(**inputs, return_dict_in_generate=True, output_scores=True)
    >>> print(outputs.scores[0][0, 50256])
    tensor(-inf)
    >>> print(outputs.scores[-1][0, 50256])  # in other places we can see some probability mass for EOS
    tensor(29.9010)

    >>> # If we disable `begin_suppress_tokens`, we can generate EOS in the first iteration.
    >>> outputs = model.generate(
    ...     **inputs, return_dict_in_generate=True, output_scores=True, begin_suppress_tokens=None
    ... )
    >>> print(outputs.scores[0][0, 50256])
    tensor(11.2027)
    ```
    """

    def __init__(self, begin_suppress_tokens, begin_index, device: str = "cpu"):
        self.begin_suppress_tokens = torch.tensor(list(begin_suppress_tokens), device=device)
        self.begin_index = begin_index

    def set_begin_index(self, begin_index):
        self.begin_index = begin_index

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        vocab_tensor = torch.arange(scores.shape[-1], device=scores.device)
        suppress_token_mask = isin_mps_friendly(vocab_tensor, self.begin_suppress_tokens)
        scores_processed = scores
        if input_ids.shape[-1] == self.begin_index:
            scores_processed = torch.where(suppress_token_mask, -float("inf"), scores)

        return scores_processed


class SuppressTokensLogitsProcessor(LogitsProcessor):
    r"""
    This processor can be used to suppress a list of tokens. The processor will set their log probs to `-inf` so
    that they are not generated. Originally created for
    [Whisper](https://huggingface.co/docs/transformers/model_doc/whisper).

    Examples:

    ```python
    >>> from transformers import AutoProcessor, WhisperForConditionalGeneration
    >>> from datasets import load_dataset

    >>> processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")
    >>> model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
    >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    >>> inputs = processor(ds[0]["audio"]["array"], return_tensors="pt")

    >>> # Whisper has a long list of suppressed tokens. For instance, in this case, the token 1 is suppressed by default.
    >>> outputs = model.generate(**inputs, return_dict_in_generate=True, output_scores=True)
    >>> print(outputs.scores[1][0, 1])  # 1 (and not 0) is the first freely generated token
    tensor(-inf)

    >>> # If we disable `suppress_tokens`, we can generate it.
    >>> outputs = model.generate(**inputs, return_dict_in_generate=True, output_scores=True, suppress_tokens=None)
    >>> print(outputs.scores[1][0, 1])
    tensor(6.0678)
    ```
    """

    def __init__(self, suppress_tokens, device: str = "cpu"):
        self.suppress_tokens = torch.tensor(list(suppress_tokens), device=device)

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        vocab_tensor = torch.arange(scores.shape[-1], device=scores.device)
        suppress_token_mask = isin_mps_friendly(vocab_tensor, self.suppress_tokens)
        scores = torch.where(suppress_token_mask, -float("inf"), scores)
        return scores


class WhisperTimeStampLogitsProcessor(LogitsProcessor):
    r"""

    [`LogitsProcessor`] that modifies the logits for the generation of timestamps in the transcription. When the input
    tokens are at a specific threshold, the processor sets the scores to negative infinity. The processor makes sure
    that timestamp tokens appear in pairs, by masking out the logits that would break this pairing pattern. This is
    done to maintain the consistency and structure of generated timestamps. It also ensures that when the predicted
    probability of sampling any of the timestamp token is greater than any individual non-timestamp token, those
    non-timestamp logits are set to negative infinity. This is done to ensure the generation of timestamps over other
    potential tokens.


    See [the paper](https://arxiv.org/abs/2212.04356) for more information.

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
        begin_index (`Optional`, *optional*): Token index of the first token that is generated by the model.
        _detect_timestamp_from_logprob (`bool`, *optional*): Whether timestamps can be predicted from logprobs over all timestamps.

    Examples:
    ``` python
    >>> import torch
    >>> from transformers import AutoProcessor, WhisperForConditionalGeneration, GenerationConfig
    >>> from datasets import load_dataset

    >>> processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")
    >>> model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
    >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    >>> inputs = processor(ds[3]["audio"]["array"], return_tensors="pt")
    >>> input_features = inputs.input_features

    >>> #Displaying timestamps
    >>> generated_ids = model.generate(inputs=input_features, return_timestamps=True)
    >>> transcription = processor.batch_decode(generated_ids, decode_with_timestamps=True)[0]
    >>> print("Transcription:", transcription)
    Transcription: <|startoftranscript|><|0.00|> He has grave doubts whether Sir Frederick Layton's work is really Greek after all, and can<|6.44|><|6.44|> discover in it but little of rocky Ithaca.<|9.44|><|endoftext|>


    >>> #No timestamps & change EOS:
    >>> #This allows the user to select a specific token to terminate the sequence on, in this case it's the word "can"(460)
    >>> model.generation_config.eos_token_id = 460
    >>> generated_ids = model.generate(inputs=input_features,return_timestamps=False)
    >>> transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    >>> print("Transcription:", transcription)
    Transcription:  He has grave doubts whether Sir Frederick Layton's work is really Greek after all and can
    ```
    """

    def __init__(
        self,
        generate_config,
        begin_index: Optional[int] = None,
        _detect_timestamp_from_logprob: Optional[bool] = None,
    ):  # support for the kwargs
        self.no_timestamps_token_id = generate_config.no_timestamps_token_id
        self.timestamp_begin = generate_config.no_timestamps_token_id + 1
        self.eos_token_id = generate_config.eos_token_id or generate_config.bos_token_id

        # this variable is mostly just used for testing
        self._detect_timestamp_from_logprob = (
            _detect_timestamp_from_logprob
            if _detect_timestamp_from_logprob is not None
            else getattr(generate_config, "_detect_timestamp_from_logprob", True)
        )

        num_forced_ids = (
            len(generate_config.forced_decoder_ids) if generate_config.forced_decoder_ids is not None else 0
        )
        self.begin_index = begin_index or (num_forced_ids + 1)

        self.max_initial_timestamp_index = getattr(generate_config, "max_initial_timestamp_index", None)
        # TODO(Patrick): Make sure that official models have max_initial_timestamp_index set to 50
        # self.max_initial_timestamp_index = 50

    def set_begin_index(self, begin_index):
        self.begin_index = begin_index

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # suppress <|notimestamps|> which is handled by without_timestamps
        scores_processed = scores.clone()
        scores_processed[:, self.no_timestamps_token_id] = -float("inf")

        # timestamps have to appear in pairs, except directly before eos_token; mask logits accordingly
        for k in range(input_ids.shape[0]):
            sampled_tokens = input_ids[k, self.begin_index :]
            seq = list(sampled_tokens.tolist())

            last_was_timestamp = len(seq) >= 1 and seq[-1] >= self.timestamp_begin
            penultimate_was_timestamp = len(seq) < 2 or seq[-2] >= self.timestamp_begin

            if last_was_timestamp:
                if penultimate_was_timestamp:  # has to be non-timestamp
                    scores_processed[k, self.timestamp_begin :] = -float("inf")
                else:  # cannot be normal text tokens
                    scores_processed[k, : self.eos_token_id] = -float("inf")

            timestamps = sampled_tokens[sampled_tokens.ge(self.timestamp_begin)]
            if timestamps.numel() > 0:
                # `timestamps` shouldn't decrease; forbid timestamp tokens smaller than the last
                # The following lines of code are copied from: https://github.com/openai/whisper/pull/914/files#r1137085090
                if last_was_timestamp and not penultimate_was_timestamp:
                    timestamp_last = timestamps[-1]
                else:
                    # Avoid to emit <|0.00|> again
                    timestamp_last = timestamps[-1] + 1

                scores_processed[k, self.timestamp_begin : timestamp_last] = -float("inf")

        # apply the `max_initial_timestamp` option
        if input_ids.shape[1] == self.begin_index:
            scores_processed[:, : self.timestamp_begin] = -float("inf")

            if self.max_initial_timestamp_index is not None:
                last_allowed = self.timestamp_begin + self.max_initial_timestamp_index
                scores_processed[:, last_allowed + 1 :] = -float("inf")

        # if sum of probability over timestamps is above any other token, sample timestamp
        logprobs = torch.nn.functional.log_softmax(scores_processed.float(), dim=-1)
        for k in range(input_ids.shape[0]):
            timestamp_logprob = logprobs[k, self.timestamp_begin :].logsumexp(dim=-1)
            max_text_token_logprob = logprobs[k, : self.timestamp_begin].max()
            if timestamp_logprob > max_text_token_logprob and self._detect_timestamp_from_logprob:
                scores_processed[k, : self.timestamp_begin] = -float("inf")

        return scores_processed


class WhisperNoSpeechDetection(LogitsProcessor):
    r"""This processor can be used to detect silence when using Whisper. It should take as input unprocessed logits to follow the original implementation"""

    def __init__(self, no_speech_token: int, begin_index: int, scores_is_logprobs: bool = False):
        self.no_speech_token = no_speech_token
        # offset between <start-of-transcription> token, <SOT>, in paper and first generated token
        # is equal to the position of the first generated token index
        self.start_of_trans_offset = begin_index

        # `self.begin_index` is a running value that is changed on the fly
        self.begin_index = begin_index
        self._no_speech_prob = [0.0]
        self.is_scores_logprobs = scores_is_logprobs

        # overwritten dynamically
        self.model = None
        self.inputs = None

    def set_model(self, model):
        self.model = model

    def set_inputs(self, inputs):
        self.inputs = {**self.model.prepare_inputs_for_generation(**inputs), **inputs}
        self.inputs["input_features"] = self.inputs.pop("inputs")

    @property
    def no_speech_prob(self):
        return self._no_speech_prob

    def set_begin_index(self, begin_index):
        self.begin_index = begin_index

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        is_scores_logprobs = self.is_scores_logprobs

        if input_ids.shape[1] == self.begin_index:
            if self.start_of_trans_offset > 1:
                with torch.no_grad():
                    logits = self.model(**self.inputs).logits

                no_speech_index = self.begin_index - self.start_of_trans_offset
                no_speech_scores = logits[:, no_speech_index]
                is_scores_logprobs = False
            else:
                no_speech_scores = scores

            if is_scores_logprobs:
                probs = no_speech_scores.exp()
            else:
                probs = no_speech_scores.float().softmax(dim=-1)

            self._no_speech_prob = probs[:, self.no_speech_token]

        return scores


class ClassifierFreeGuidanceLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] for classifier free guidance (CFG). The scores are split over the batch dimension,
    where the first half correspond to the conditional logits (predicted from the input prompt) and the second half
    correspond to the unconditional logits (predicted from an empty or 'null' prompt). The processor computes a
    weighted average across the conditional and unconditional logits, parameterised by the `guidance_scale`.

    See [the paper](https://arxiv.org/abs/2306.05284) for more information.

    <Tip warning={true}>

    This logits processor is exclusively compatible with
    [MusicGen](https://huggingface.co/docs/transformers/main/en/model_doc/musicgen)

    </Tip>

    Args:
        guidance_scale (float):
            The guidance scale for classifier free guidance (CFG). CFG is enabled by setting `guidance_scale > 1`.
            Higher guidance scale encourages the model to generate samples that are more closely linked to the input
            prompt, usually at the expense of poorer quality.

    Examples:

    ```python
    >>> from transformers import AutoProcessor, MusicgenForConditionalGeneration

    >>> processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    >>> model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

    >>> inputs = processor(
    ...     text=["80s pop track with bassy drums and synth", "90s rock song with loud guitars and heavy drums"],
    ...     padding=True,
    ...     return_tensors="pt",
    ... )
    >>> audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=256)
    ```
    """

    def __init__(self, guidance_scale):
        if guidance_scale > 1:
            self.guidance_scale = guidance_scale
        else:
            raise ValueError(
                "Require guidance scale >1 to use the classifier free guidance processor, got guidance scale "
                f"{guidance_scale}."
            )

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # simple check to make sure we have compatible batch sizes between our
        # logits scores (cond + uncond) and input ids (cond only)
        if scores.shape[0] != 2 * input_ids.shape[0]:
            raise ValueError(
                f"Logits should have twice the batch size of the input ids, the first half of batches corresponding to "
                f"the conditional inputs, and the second half of batches corresponding to the unconditional inputs. Got "
                f"batch size {scores.shape[0]} for the logits and {input_ids.shape[0]} for the input ids."
            )
        unguided_bsz = scores.shape[0] // 2
        cond_logits, uncond_logits = scores.split(unguided_bsz, dim=0)
        scores_processed = uncond_logits + (cond_logits - uncond_logits) * self.guidance_scale
        return scores_processed


class AlternatingCodebooksLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] enforcing alternated generation between the two codebooks of Bark.

    <Tip warning={true}>

    This logits processor is exclusively compatible with
    [Bark](https://huggingface.co/docs/transformers/en/model_doc/bark)'s fine submodel. See the model documentation
    for examples.

    </Tip>

    Args:
        input_start_len (`int`):
            The length of the initial input sequence.
        semantic_vocab_size (`int`):
            Vocabulary size of the semantic part, i.e number of tokens associated to the semantic vocabulary.
        codebook_size (`int`):
            Number of tokens associated to the codebook.
    """

    def __init__(self, input_start_len: int, semantic_vocab_size: int, codebook_size: int):
        if not isinstance(input_start_len, int) or input_start_len < 0:
            raise ValueError(f"`input_starting_length` has to be a non-negative integer, but is {input_start_len}")

        self.input_start_len = input_start_len
        self.semantic_vocab_size = semantic_vocab_size
        self.codebook_size = codebook_size

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        curr_len = input_ids.shape[-1]

        # even -> first codebook, odd -> second codebook
        is_first_codebook = ((curr_len - self.input_start_len) % 2) == 0

        scores_processed = scores.clone()
        if is_first_codebook:
            scores_processed[:, : self.semantic_vocab_size] = -float("inf")
            scores_processed[:, self.semantic_vocab_size + self.codebook_size :] = -float("inf")
        else:
            scores_processed[:, : self.semantic_vocab_size + self.codebook_size] = -float("inf")

        return scores_processed


class UnbatchedClassifierFreeGuidanceLogitsProcessor(LogitsProcessor):
    r"""
    Logits processor for Classifier-Free Guidance (CFG). The processors computes a weighted average across scores
    from prompt conditional and prompt unconditional (or negative) logits, parameterized by the `guidance_scale`.
    The unconditional scores are computed internally by prompting `model` with the `unconditional_ids` branch.

    See [the paper](https://arxiv.org/abs/2306.17806) for more information.

    Args:
        guidance_scale (`float`):
            The guidance scale for classifier free guidance (CFG). CFG is enabled by setting `guidance_scale != 1`.
            Higher guidance scale encourages the model to generate samples that are more closely linked to the input
            prompt, usually at the expense of poorer quality. A value smaller than 1 has the opposite effect, while
            making the negative prompt provided with negative_prompt_ids (if any) act as a positive prompt.
        model (`PreTrainedModel`):
            The model computing the unconditional scores. Supposedly the same as the one computing the conditional
            scores. Both models must use the same tokenizer.
        unconditional_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of input sequence tokens in the vocabulary for the unconditional branch. If unset, will default to
            the last token of the prompt.
        unconditional_attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Attention mask for unconditional_ids.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether to cache key/values during the negative prompt forward pass.


    Examples:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM

    >>> model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    >>> tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    >>> inputs = tokenizer(["Today, a dragon flew over Paris, France,"], return_tensors="pt")
    >>> out = model.generate(inputs["input_ids"], guidance_scale=1.5)
    >>> tokenizer.batch_decode(out, skip_special_tokens=True)[0]
    'Today, a dragon flew over Paris, France, killing at least 50 people and injuring more than 100'

    >>> # with a negative prompt
    >>> neg_inputs = tokenizer(["A very happy event happened,"], return_tensors="pt")
    >>> out = model.generate(inputs["input_ids"], guidance_scale=2, negative_prompt_ids=neg_inputs["input_ids"])
    >>> tokenizer.batch_decode(out, skip_special_tokens=True)[0]
    'Today, a dragon flew over Paris, France, killing at least 130 people. French media reported that'

    >>> # with a positive prompt
    >>> neg_inputs = tokenizer(["A very happy event happened,"], return_tensors="pt")
    >>> out = model.generate(inputs["input_ids"], guidance_scale=0, negative_prompt_ids=neg_inputs["input_ids"])
    >>> tokenizer.batch_decode(out, skip_special_tokens=True)[0]
    "Today, a dragon flew over Paris, France, and I'm very happy to be here. I"
    ```
    """

    def __init__(
        self,
        guidance_scale: float,
        model,
        unconditional_ids: Optional[torch.LongTensor] = None,
        unconditional_attention_mask: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = True,
    ):
        self.guidance_scale = guidance_scale
        self.model = model
        self.unconditional_context = {
            "input_ids": unconditional_ids,
            "attention_mask": unconditional_attention_mask,
            "use_cache": use_cache,
            "past_key_values": None,
            "first_pass": True,
        }

    def get_unconditional_logits(self, input_ids):
        if self.unconditional_context["first_pass"]:
            if self.unconditional_context["input_ids"] is None:
                self.unconditional_context["input_ids"] = input_ids[:, -1:]
            if self.unconditional_context["attention_mask"] is None:
                self.unconditional_context["attention_mask"] = torch.ones_like(
                    self.unconditional_context["input_ids"], dtype=torch.long
                )
            input_ids = self.unconditional_context["input_ids"]
            attention_mask = self.unconditional_context["attention_mask"]
            self.unconditional_context["first_pass"] = False
        else:
            attention_mask = torch.cat(
                [
                    self.unconditional_context["attention_mask"],
                    torch.ones_like(input_ids[:, -1:], dtype=torch.long),
                ],
                dim=1,
            )
            if not self.unconditional_context["use_cache"]:
                input_ids = torch.cat([self.unconditional_context["input_ids"], input_ids[:, -1:]], dim=1)
            else:
                input_ids = input_ids[:, -1:]
            self.unconditional_context["input_ids"] = input_ids
            self.unconditional_context["attention_mask"] = attention_mask

        out = self.model(
            input_ids,
            attention_mask=attention_mask,
            use_cache=self.unconditional_context["use_cache"],
            past_key_values=self.unconditional_context["past_key_values"],
        )
        self.unconditional_context["past_key_values"] = out.get("past_key_values", None)

        return out.logits

    def __call__(self, input_ids, scores):
        scores = torch.nn.functional.log_softmax(scores, dim=-1)
        if self.guidance_scale == 1:
            return scores

        logits = self.get_unconditional_logits(input_ids)

        unconditional_logits = torch.nn.functional.log_softmax(logits[:, -1], dim=-1)
        scores_processed = self.guidance_scale * (scores - unconditional_logits) + unconditional_logits
        return scores_processed


class BarkEosPrioritizerLogitsProcessor(LogitsProcessor):
    r"""This processor ensures that the EOS token is selected if its probability is greater than the `min_eos_p`.

    <Tip warning={true}>

    This logits processor is exclusively compatible with
    [Bark](https://huggingface.co/docs/transformers/en/model_doc/bark). See the model documentation for examples.

    </Tip>

    Args:
        eos_token_id (`Union[int, List[int], torch.Tensor]`):
            The id(s) of the *end-of-sequence* token.
        min_eos_p (`float`, *optional*):
            Minimum end of speech threshold.
    """

    def __init__(self, eos_token_id: Union[int, List[int], torch.Tensor], min_eos_p: float, device: str = "cpu"):
        if not isinstance(eos_token_id, torch.Tensor):
            if isinstance(eos_token_id, int):
                eos_token_id = [eos_token_id]
            eos_token_id = torch.tensor(eos_token_id, device=device)
        self.eos_token_id = eos_token_id

        if torch.is_floating_point(eos_token_id) or (eos_token_id < 0).any():
            raise ValueError(f"`eos_token_id` has to be a list of positive integers, but is {eos_token_id}")

        if min_eos_p is not None and min_eos_p <= 0:
            raise ValueError(f"`min_eos_p` has to be a positive float, but is {min_eos_p}")
        self.min_eos_p = min_eos_p

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores_processed = scores
        if self.min_eos_p:
            probs = torch.nn.functional.softmax(scores.float(), dim=-1)
            # create scores full of -inf except for the eos_token_id
            early_stop_scores = torch.ones_like(scores) * -float("inf")
            early_stop_scores[:, self.eos_token_id] = scores[:, self.eos_token_id]

            do_early_stop = probs[:, self.eos_token_id] > self.min_eos_p
            do_early_stop = torch.any(do_early_stop, dim=1, keepdim=True)
            scores_processed = torch.where(do_early_stop, early_stop_scores, scores)

        return scores_processed


class WatermarkLogitsProcessor(LogitsProcessor):
    r"""
    Logits processor for watermarking generated text. The processor modifies model output scores by adding a small bias to
    randomized set of "green" tokens before generating the next token. "Green" tokens selection process depends on the
    `seeding_scheme` used. The code was based on the [original repo](https://github.com/jwkirchenbauer/lm-watermarking/tree/main).

    The text generated by this `LogitsProcessor` can be detected using `WatermarkDetector`. See [`~WatermarkDetector.__call__`] for details,

    See [the paper](https://arxiv.org/abs/2306.04634) for more information.

    Args:
        vocab_size (`int`):
            The model tokenizer's vocab_size. Used to calculate "green" tokens ratio.
        device (`str`):
            The device where model is allocated.
        greenlist_ratio (`float`, optional, *optional*, defaults to 0.25):
            The ratio of "green" tokens used to the vocabulary size. Defaults to 0.25.
        bias (`float`, optional, *optional*, defaults to 2.0):
            The bias added to the selected "green" tokens' logits. Consider lowering the
            `bias` if the text generation quality degrades. Recommended values are in the
            range of [0.5, 2.0]. Defaults to 2.0.
        hashing_key (`int`, optional, *optional*, defaults to 15485863):
            Key used for hashing. If you deploy this watermark, we advise using another private key.
            Defaults to 15485863 (the millionth prime).
        seeding_scheme (`str`, optional, *optional*, defaults to `"lefthash"`):
            The seeding scheme used for selecting "green" tokens. Accepts values:
                - "lefthash" (default): "green" tokens selection depend on the last token (Algorithm 2 from paper)
                - "selfhash": "green" tokens selection depends on the current token itself (Algorithm 3 from paper)
                    The downside of this scheme is that it considers all possible next tokens and can be slower than "lefthash".
            The context length of previous tokens to use in seeding. Higher context length makes watermarking more robust.
        context_width (`int`, *optional*, defaults to 1):
            The number of previous tokens to use when setting the seed.

    Examples:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM, WatermarkingConfig

    >>> model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    >>> tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    >>> inputs = tokenizer(["Alice and Bob are"], return_tensors="pt")

    >>> # normal generation
    >>> out = model.generate(inputs["input_ids"], max_length=20, do_sample=False)
    >>> tokenizer.batch_decode(out, skip_special_tokens=True)[0]
    'Alice and Bob are both in the same room.\n\n"I\'m not sure if you\'re'

    >>> # watermarked generation
    >>> watermarking_config = WatermarkingConfig(bias=2.5, context_width=2, seeding_scheme="selfhash")
    >>> out = model.generate(inputs["input_ids"], watermarking_config=watermarking_config, max_length=20, do_sample=False)
    >>> tokenizer.batch_decode(out, skip_special_tokens=True)[0]
    'Alice and Bob are both still alive and well and the story is pretty much a one-hour adventure'

    >>> # to detect watermarked text use the WatermarkDetector class
    >>> from transformers import WatermarkDetector
    >>> detector = WatermarkDetector(model_config=model.config, device="cpu", watermarking_config= watermarking_config)
    >>> detection_preds = detector(out)
    >>> detection_preds
    array([ True])
    ```
    """

    def __init__(
        self,
        vocab_size,
        device,
        greenlist_ratio: float = 0.25,
        bias: float = 2.0,
        hashing_key: int = 15485863,
        seeding_scheme: str = "lefthash",
        context_width: int = 1,
    ):
        if seeding_scheme not in ["selfhash", "lefthash"]:
            raise ValueError(f"seeding_scheme has to be one of [`selfhash`, `lefthash`], but found {seeding_scheme}")
        if greenlist_ratio >= 1.0 or greenlist_ratio <= 0.0:
            raise ValueError(
                f"greenlist_ratio has be in range between 0.0 and 1.0, exclusively. but found {greenlist_ratio}"
            )

        self.vocab_size = vocab_size
        self.greenlist_size = int(self.vocab_size * greenlist_ratio)
        self.bias = bias
        self.seeding_scheme = seeding_scheme
        self.rng = torch.Generator(device=device)
        self.hash_key = hashing_key
        self.context_width = context_width

        self.rng.manual_seed(hashing_key)
        self.table_size = 1_000_003
        self.fixed_table = torch.randperm(self.table_size, generator=self.rng, device=device)

    def set_seed(self, input_seq: torch.LongTensor):
        input_seq = input_seq[-self.context_width :]
        if self.seeding_scheme == "selfhash":
            a = self.fixed_table[input_seq % self.table_size] + 1
            b = self.fixed_table[input_seq[-1] % self.table_size] + 1
            seed = (self.hash_key * a * b).min().item()
        else:
            seed = self.hash_key * input_seq[-1].item()
        self.rng.manual_seed(seed % (2**64 - 1))

    def _get_greenlist_ids(self, input_seq: torch.LongTensor) -> torch.LongTensor:
        self.set_seed(input_seq)
        vocab_permutation = torch.randperm(self.vocab_size, device=input_seq.device, generator=self.rng)
        greenlist_ids = vocab_permutation[: self.greenlist_size]
        return greenlist_ids

    def _score_rejection_sampling(self, input_seq: torch.LongTensor, scores: torch.FloatTensor) -> torch.LongTensor:
        """
        Generate greenlist based on current candidate next token. Reject and move on if necessary.
        Runs for a fixed number of steps only for efficiency, since the methods is not batched.
        """
        final_greenlist = []
        _, greedy_predictions = scores.sort(dim=-1, descending=True)

        # 40 is an arbitrary number chosen to save compute and not run for long (taken from orig repo)
        for i in range(40):
            greenlist_ids = self._get_greenlist_ids(torch.cat([input_seq, greedy_predictions[i, None]], dim=-1))
            if greedy_predictions[i] in greenlist_ids:
                final_greenlist.append(greedy_predictions[i])
        return torch.tensor(final_greenlist, device=input_seq.device)

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.shape[-1] < self.context_width:
            logger.warning(
                f"`input_ids` should have at least `{self.context_width}` tokens but has {input_ids.shape[-1]}. "
                "The seeding will be skipped for this generation step!"
            )
            return scores

        scores_processed = scores.clone()
        for b_idx, input_seq in enumerate(input_ids):
            if self.seeding_scheme == "selfhash":
                greenlist_ids = self._score_rejection_sampling(input_seq, scores[b_idx])
            else:
                greenlist_ids = self._get_greenlist_ids(input_seq)
            scores_processed[b_idx, greenlist_ids] = scores_processed[b_idx, greenlist_ids] + self.bias

        return scores_processed


class SynthIDTextWatermarkState:
    """SynthID watermarking state."""

    def __init__(
        self,
        batch_size: int,
        ngram_len: int,
        context_history_size: int,
        device: torch.device,
    ):
        """Initializes the state.

        Args:
            batch_size (`int`): Batch size.
            ngram_len (`int`): Ngram length.
            context_history_size (`int`): Size of the tensor to keep track of seen contexts.
            device (`int`): Device to use.
        """
        self.context = torch.zeros(
            (batch_size, ngram_len - 1),
            dtype=torch.int64,
            device=device,
        )
        self.context_history = torch.zeros(
            (batch_size, context_history_size),
            dtype=torch.int64,
            device=device,
        )
        self.num_calls = 0


class SynthIDTextWatermarkLogitsProcessor(LogitsProcessor):
    r"""
    Logits processor that implements watermarking techniques for text generation models.
    This class facilitates the application of SynthID text watermarking, a method for embedding imperceptible signals
    into generated text to aid in detecting synthetic content. It operates by subtly manipulating the probabilities of
    token selection during text generation in a manner that can be reliably recovered later for verification.

    Key Features:
    * **State Management:** Maintains internal state to track token sequences and generate watermarking keys
    dynamically.

    * **Key Generation:** Computes hashes based on token sequences and watermarking parameters to create unique keys
    for each position.

    * **G-Value Sampling:** Employs a pre-computed sampling table to sample watermarking values (g-values) based on
    the generated keys.

    * **Score Adjustment:** Applies calculated g-values to modify token probabilities during generation, embedding the
    watermark.

    * **Context Repetition Handling:** Incorporates logic to avoid watermarking tokens in repeated contexts,
    preserving naturalness.

    * **EOS Token Masking:** Supports masking end-of-sentence tokens to prevent their inclusion in watermarking
    calculations.

    * **Utility Functions:** Provides functions to compute g-values directly, check for context repetition, create
    EOS token masks, and estimate expected mean g-values.

    Refer to paper url: https://www.nature.com/articles/s41586-024-08025-4 for more details around this.

    Args:
        ngram_len (`int`):
            Ngram length.
        keys (`List[int]`):
            A sequence of watermarking keys, one for each depth.
        sampling_table_size (`int`):
            Size of the sampling table.
        sampling_table_seed (`int`):
            Random seed to generate the sampling table.
        context_history_size (`int`):
            Size of the tensor to keep track of seen contexts.
        device (`torch.device`):
            Device to use.
        skip_first_ngram_calls (`bool`, *optional*, defaults to `False`):
            Whether to skip first ngram calls.
        debug_mode (`bool`, optional, *optional*, defaults to `False`):
            Logits are modified to uniform one got before watermarking modification is applied. This is to test the
            implementation.

    Examples:
    ```python
    >>> from transformers import AutoModelForCausalLM, AutoTokenizer, SynthIDTextWatermarkingConfig

    >>> tokenizer = AutoTokenizer.from_pretrained('google/gemma-2-2b', padding_side="left")
    >>> model = AutoModelForCausalLM.from_pretrained('google/gemma-2-2b')

    >>> # SynthID Text configuration
    >>> watermarking_config = SynthIDTextWatermarkingConfig(
    ...     keys=[654, 400, 836, 123, 340, 443, 597, 160, 57],
    ...     ngram_len=5,
    ... )

    >>> # Generation with watermarking
    >>> tokenized_prompts = tokenizer(["Once upon a time, "], return_tensors="pt", padding=True)
    >>> output_sequences = model.generate(
    ...     **tokenized_prompts, watermarking_config=watermarking_config, do_sample=True, max_new_tokens=10
    ... )
    >>> watermarked_text = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
    ```
    """

    def __init__(
        self,
        ngram_len: int,
        keys: List[int],
        sampling_table_size: int,
        sampling_table_seed: int,
        context_history_size: int,
        device: torch.device,
        skip_first_ngram_calls: bool = False,
        debug_mode: bool = False,
    ):
        self.ngram_len = ngram_len
        self.keys = torch.tensor(keys, device=device)

        generator = torch.Generator(device=device).manual_seed(sampling_table_seed)
        # A random sampling table is pre-computed and modulo table size is applied to map from a hash of ngram keys to
        # g values, this is similar to the hashtable implementation used in
        # https://github.com/facebookresearch/three_bricks. We note that the hashing employed in this repository is
        # different from that used to watermark the Gemini App, and hence the detectors trained based on the
        # hashing in this repository will not transfer to text generated by the Gemini App.
        self.sampling_table = torch.randint(
            low=0,
            high=2,
            size=(sampling_table_size,),
            generator=generator,
            device=device,
        )
        self.context_history_size = context_history_size
        self.device = device
        self.state = None
        self.skip_first_ngram_calls = skip_first_ngram_calls
        self.debug_mode = debug_mode

    def _init_state(self, batch_size: int):
        """Initializes the state."""
        self.state = SynthIDTextWatermarkState(
            batch_size=batch_size,
            ngram_len=self.ngram_len,
            context_history_size=self.context_history_size,
            device=self.device,
        )

    def update_scores(self, scores: torch.FloatTensor, g_values: torch.FloatTensor) -> torch.FloatTensor:
        """Updates scores using the g values.

        We assume that the scores are in the log space.
        Args:
            scores (`torch.FloatTensor`): Scores (batch_size, vocab_size).
            g_values (`torch.FloatTensor`): G valus (batch_size, vocab_size, depth).

        Returns:
            Updated scores (batch_size, vocab_size).
        """
        _, _, depth = g_values.shape

        probs = torch.softmax(scores, dim=1)

        for i in range(depth):
            g_values_at_depth = g_values[:, :, i]
            g_mass_at_depth = (g_values_at_depth * probs).sum(axis=1, keepdims=True)
            probs = probs * (1 + g_values_at_depth - g_mass_at_depth)

        log_probs = torch.log(probs)
        log_probs = torch.where(torch.isfinite(log_probs), log_probs, torch.finfo(log_probs.dtype).min)
        return log_probs

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self._check_input_ids_shape(input_ids)
        batch_size, vocab_size = scores.shape

        if self.debug_mode:
            scores = torch.ones_like(scores)

        # Currently indices is just a arange to compute watermarking on the desnse logits.
        all_indices = torch.stack([torch.arange(vocab_size, device=self.device) for _ in range(batch_size)])

        if self.state is None:
            # Initialize watermarking state if it does not exist.
            self._init_state(batch_size)
        else:
            # Append last input id (which is the input id added in last call) to the
            # previous context so we have the context to be used for current
            # watermarking.
            self.state.context = torch.concat(
                (self.state.context, input_ids[:, -1:]),
                dim=1,
            )
            self.state.context = self.state.context[:, 1:]

        if self.state is None:
            raise ValueError("self.state can't be None! Call `self._init_state` to initialize the state.")

        self.state.num_calls += 1

        # Don't watermark the first ngram_len - 1 tokens if set.
        if self.skip_first_ngram_calls and self.state.num_calls < self.ngram_len:
            return scores

        # 2. Generate random keys for each ngram key combination.
        ngram_keys, hash_result_with_just_context = self._compute_keys(self.state.context, all_indices)
        # ngram_keys shape [batch_size, top_k, depth]

        # 3. Sample g values.
        g_values = self.sample_g_values(ngram_keys)
        # g_values shape [batch_size, top_k, depth]

        # 4. Modify scores.
        updated_scores = self.update_scores(scores, g_values)
        # updated scores shape [batch_size, top_k]

        # 5. Check if the current watermarking context was previously used, if yes skip watermarking.
        hash_result_with_just_context = hash_result_with_just_context[:, None]
        is_repeated_context = (self.state.context_history == hash_result_with_just_context).any(
            dim=1,
            keepdim=True,
        )
        self.state.context_history = torch.concat(
            (hash_result_with_just_context, self.state.context_history),
            dim=1,
        )[:, :-1]

        updated_watermarked_scores = torch.where(
            is_repeated_context,
            input=scores,
            other=updated_scores,
        )
        return updated_watermarked_scores

    def accumulate_hash(
        self,
        current_hash: torch.LongTensor,
        data: torch.LongTensor,
        multiplier: int = 6364136223846793005,
        increment: int = 1,
    ) -> torch.LongTensor:
        """
        Accumulate hash of data on current hash.

        Method uses adapted linear congruential generator with newlib/musl parameters.

        This function has following property -
        f(x, data[T]) = f(f(x, data[:T - 1]), data[T])

        This function expects current_hash.shape and data.shape[:-1] to
        match/broadcastable.

        Args:
            current_hash (`torch.LongTensor`):
                (shape,)
            data (`torch.LongTensor`):
                (shape, tensor_len)
            multiplier (`int`, optional, *optional*, defaults to 6364136223846793005):
                multiplier of linear congruential generator
            increment (`int`, optional, *optional*, defaults to 1):
                increment of linear congruential generator

        Returns:
            updated hash (shape,)
        """
        for i in range(data.shape[-1]):
            current_hash = torch.add(current_hash, data[..., i])
            current_hash = torch.mul(current_hash, multiplier)
            current_hash = torch.add(current_hash, increment)
        return current_hash

    def compute_ngram_keys(self, ngrams: torch.LongTensor) -> torch.LongTensor:
        """Computes random keys for each ngram and depth.

        Args:
            ngrams (`torch.LongTensor`):
                Ngrams (batch_size, num_ngrams, ngram_len).

        Returns:
            ngram keys (batch_size, num_ngrams, depth).
        """
        if len(ngrams.shape) != 3:
            raise ValueError(
                "Ngrams should be of shape (batch_size, num_ngrams, ngram_len), but" f" is {ngrams.shape}"
            )
        if ngrams.shape[2] != self.ngram_len:
            raise ValueError(
                "Ngrams should be of shape (batch_size, num_ngrams, ngram_len),"
                f" where ngram_len is {self.ngram_len}, but is {ngrams.shape}"
            )
        batch_size, _, _ = ngrams.shape

        hash_result = torch.ones(batch_size, device=self.device, dtype=torch.long)
        # hash_result shape [batch_size,]
        # ngrams shape [batch_size, num_ngrams, ngram_len]
        hash_result = torch.vmap(self.accumulate_hash, in_dims=(None, 1), out_dims=1)(hash_result, ngrams)
        # hash_result shape [batch_size, num_ngrams]

        keys = self.keys[None, None, :, None]
        # hash_result shape [batch_size, num_ngrams]
        # keys shape [1, 1, depth, 1]
        hash_result = torch.vmap(self.accumulate_hash, in_dims=(None, 2), out_dims=2)(hash_result, keys)
        # hash_result shape [batch_size, num_ngrams, depth]

        return hash_result

    def _compute_keys(
        self, n_minus_1_grams: torch.LongTensor, indices: torch.LongTensor
    ) -> Tuple[torch.LongTensor, torch.LongTensor]:
        """Computes random keys for each ngram and depth.

        Args:
            n_minus_1_grams (`torch.LongTensor`):
                Ngrams (batch_size, ngram_len - 1).
            indices (`torch.LongTensor`):
                indices of the continuations (batch_size, num_indices)

        Returns:
            Ngram keys (batch_size, num_indices, depth).
        """
        batch_size, _ = n_minus_1_grams.shape

        hash_result = torch.ones(batch_size, device=self.device, dtype=torch.long)
        # First hash n_minus_1 gram, for each batch entry we have a single
        # n_minus_1 gram context.
        # hash_result shape [batch_size]
        # n_minus_1_gram shape [batch_size, ngram_len - 1]
        hash_result_with_just_context = self.accumulate_hash(hash_result, n_minus_1_grams)
        # hash_result shape [batch_size,]
        # Indices is of shape [batch_size, num_indices], so we make it
        # [batch_size, num_indices, 1] so we can vmap over num_indices dim.
        hash_result = torch.vmap(self.accumulate_hash, in_dims=(None, 1), out_dims=1)(
            hash_result_with_just_context, indices[:, :, None]
        )
        # hash_result shape [batch_size, num_indices]
        # Basically we have a hash for each batch entry and each indices
        # Now we add watermarking keys to this hash.
        # keys are of shape [depth,]
        # We add batch, num_indices and data dimension to this making it
        # [1, 1, depth, 1].
        # So we can vmap over the depth dimension for compute_hash
        keys = self.keys[None, None, :, None]
        hash_result = torch.vmap(self.accumulate_hash, in_dims=(None, 2), out_dims=2)(hash_result, keys)
        # hash_result shape should be [batch_size, num_indices, depth]
        return hash_result, hash_result_with_just_context

    def sample_g_values(self, ngram_keys: torch.LongTensor) -> torch.LongTensor:
        """
        Samples g values from Bernoulli distribution.

        It is not possible to pass random keys in a vectorized way in torch. Instead
        we pre-compute a random sampling table, and use apply modulo table size to
        map from ngram keys (int64) to g values.

        Args:
            ngram_keys (`torch.LongTensor`):
                Random keys (batch_size, num_ngrams, depth).

        Returns:
            G values (batch_size, num_ngrams, depth).
        """
        (sampling_table_size,) = self.sampling_table.shape
        sampling_table = self.sampling_table.reshape((1, 1, sampling_table_size))
        ngram_keys = ngram_keys % sampling_table_size
        return torch.take_along_dim(sampling_table, indices=ngram_keys, dim=2)

    def _check_input_ids_shape(self, input_ids: torch.LongTensor):
        """Checks the shape of input ids."""
        if len(input_ids.shape) != 2:
            raise ValueError("Input ids should be of shape (batch_size, input_len), but is" f" {input_ids.shape}")

    def compute_g_values(self, input_ids: torch.LongTensor) -> torch.LongTensor:
        """
        Computes g values for each ngram from the given sequence of tokens.

        Args:
            input_ids (`torch.LongTensor`):
                Input token ids (batch_size, input_len).

        Returns:
            G values (batch_size, input_len - (ngram_len - 1), depth).
        """
        self._check_input_ids_shape(input_ids)
        ngrams = input_ids.unfold(dimension=1, size=self.ngram_len, step=1)
        ngram_keys = self.compute_ngram_keys(ngrams)
        return self.sample_g_values(ngram_keys)

    def compute_context_repetition_mask(self, input_ids: torch.LongTensor) -> torch.LongTensor:
        """
        Computes repetition mask.

        0 and 1 stand for repeated and not repeated context n-1 grams respectively.

        Args:
            input_ids (`torch.LongTensor`):
                Input token ids (batch_size, input_len).

        Returns:
            Repetitions mask (batch_size, input_len - (ngram_len - 1)).
        """
        self._check_input_ids_shape(input_ids)
        batch_size, _ = input_ids.shape
        state = SynthIDTextWatermarkState(
            batch_size=batch_size,
            ngram_len=self.ngram_len,
            context_history_size=self.context_history_size,
            device=self.device,
        )
        contexts = input_ids[:, :-1].unfold(
            dimension=1,
            size=self.ngram_len - 1,
            step=1,
        )
        _, num_contexts, _ = contexts.shape

        are_repeated_contexts = []
        for i in range(num_contexts):
            context = contexts[:, i, :]
            hash_result = torch.ones(batch_size, device=self.device, dtype=torch.long)
            context_hash = self.accumulate_hash(hash_result, context)[:, None]
            is_repeated_context = (state.context_history == context_hash).any(
                dim=1,
                keepdim=True,
            )
            are_repeated_contexts.append(is_repeated_context)
            state.context_history = torch.concat(
                (context_hash, state.context_history),
                dim=1,
            )[:, :-1]
        are_repeated_contexts = torch.concat(are_repeated_contexts, dim=1)

        return torch.logical_not(are_repeated_contexts)

    def compute_eos_token_mask(self, input_ids: torch.LongTensor, eos_token_id: int) -> torch.LongTensor:
        """
        Computes repetitions mask.

        1 stands for ngrams that don't contain EOS tokens and vice versa.

        Args:
            input_ids (`torch.LongTensor`):
                Input token ids (batch_size, input_len).
            eos_token_id (`int`):
                EOS token ID.

        Returns:
            EOS token mask (batch_size, input_len).
        """
        self._check_input_ids_shape(input_ids)
        noneos_masks = []
        all_eos_equated = input_ids == eos_token_id
        for eos_equated in all_eos_equated:
            nonzero_idx = torch.nonzero(eos_equated)
            noneos_mask = torch.ones_like(eos_equated)
            if nonzero_idx.shape[0] != 0:
                noneos_mask[nonzero_idx[0][0] :] = 0
            noneos_masks.append(noneos_mask)
        return torch.stack(noneos_masks, dim=0)

    def expected_mean_g_value(self, vocab_size: int, coinflip_prob: float = 0.5) -> float:
        """
        Compute expected mean g-value after watermarking, assuming uniform LM dist.

        This is the theoretical expected value for single-layer watermarking.

        Args:
            vocab_size (`int`):
                The size of the vocabulary.
            coinflip_prob arg_name (`float`, *optional*, defaults to 0.5):
                Probability of 1 in boolean prf.

        Returns:
            The expected mean g-value for watermarked text.
        """
        return coinflip_prob + coinflip_prob * (1 - coinflip_prob) * (1 - (1 / vocab_size))
