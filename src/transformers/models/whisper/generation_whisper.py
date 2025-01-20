# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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
import math
import warnings
import zlib
from typing import Callable, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from transformers.cache_utils import EncoderDecoderCache

from ...generation import GenerationConfig, GenerationMixin
from ...generation.logits_process import (
    LogitsProcessorList,
    SuppressTokensAtBeginLogitsProcessor,
    SuppressTokensLogitsProcessor,
    WhisperNoSpeechDetection,
    WhisperTimeStampLogitsProcessor,
)
from ...generation.stopping_criteria import StoppingCriteriaList
from ...modeling_outputs import BaseModelOutput
from ...utils import logging
from .tokenization_whisper import TASK_IDS, TO_LANGUAGE_CODE


logger = logging.get_logger(__name__)


def _median_filter(inputs: torch.Tensor, filter_width: int) -> torch.Tensor:
    """
    Applies a median filter of width `filter_width` along the last dimension of the input.

    The `inputs` tensor is assumed to be 3- or 4-dimensional.
    """
    if filter_width <= 0 or filter_width % 2 != 1:
        raise ValueError("`filter_width` should be an odd number")

    pad_width = filter_width // 2
    if inputs.shape[-1] <= pad_width:
        return inputs

    # Pad the left and right edges.
    inputs = nn.functional.pad(inputs, (pad_width, pad_width, 0, 0), mode="reflect")

    # sort() is faster than torch.median (https://github.com/pytorch/pytorch/issues/51450)
    result = inputs.unfold(-1, filter_width, 1).sort()[0][..., pad_width]
    return result


def _dynamic_time_warping(matrix: np.ndarray):
    """
    Measures similarity between two temporal sequences: the input audio and the output tokens. Used to generate
    token-level timestamps.
    """
    output_length, input_length = matrix.shape
    cost = np.ones((output_length + 1, input_length + 1), dtype=np.float32) * np.inf
    trace = -np.ones((output_length + 1, input_length + 1), dtype=np.float32)

    cost[0, 0] = 0
    for j in range(1, input_length + 1):
        for i in range(1, output_length + 1):
            c0 = cost[i - 1, j - 1]
            c1 = cost[i - 1, j]
            c2 = cost[i, j - 1]

            if c0 < c1 and c0 < c2:
                c, t = c0, 0
            elif c1 < c0 and c1 < c2:
                c, t = c1, 1
            else:
                c, t = c2, 2

            cost[i, j] = matrix[i - 1, j - 1] + c
            trace[i, j] = t

    # backtrace
    i = trace.shape[0] - 1
    j = trace.shape[1] - 1
    trace[0, :] = 2
    trace[:, 0] = 1

    text_indices = []
    time_indices = []
    while i > 0 or j > 0:
        text_indices.append(i - 1)
        time_indices.append(j - 1)
        if trace[i, j] == 0:
            i -= 1
            j -= 1
        elif trace[i, j] == 1:
            i -= 1
        elif trace[i, j] == 2:
            j -= 1
        else:
            raise RuntimeError(
                f"Internal error in dynamic time warping. Unexpected trace[{i}, {j}]. Please file a bug report."
            )

    text_indices = np.array(text_indices)[::-1]
    time_indices = np.array(time_indices)[::-1]
    return text_indices, time_indices


def _get_attr_from_logit_processors(logits_processor, logit_processor_class, attribute_name):
    if logits_processor is not None:
        logit_processor = next((cls for cls in logits_processor if isinstance(cls, logit_processor_class)), None)
        if logit_processor:
            return getattr(logit_processor, attribute_name, None)
    return None


def _pad_to_max_length(
    current_segments,
    pad_token_id,
    device,
    padding_side="right",
    padding="longest",
    bos_token_tensor=None,
    cut_off_length=None,
    return_token_timestamps=False,
    force_unique_generate_call=False,
):
    max_total_length = 0
    sequences = []
    token_timestamps_list = []

    if padding_side not in ["right", "left"]:
        raise ValueError(f"`padding_side` must be either 'right' or 'left', not {padding_side}")

    if padding not in ["longest", "max_length"]:
        raise ValueError(f"`padding` must be either 'longest' or 'max_length', not {padding}")
    elif padding == "max_length" and cut_off_length is None:
        raise ValueError("`cut_off_length` must be specified when `padding='max_length'`")

    if force_unique_generate_call:
        sequences_list = []
        timestamps_list = []
        for segments in current_segments:
            result = segments[0]["result"]
            sequences_list.append(result if isinstance(result, torch.Tensor) else result["sequences"])
            if return_token_timestamps:
                timestamps_list.append(result["token_timestamps"])

        sequences = torch.stack(sequences_list, dim=0)
        if return_token_timestamps:
            token_timestamps = torch.stack(timestamps_list, dim=0)
            return sequences, token_timestamps
        return sequences

    for current_segment_list in current_segments:
        if current_segment_list is not None and len([d["tokens"] for d in current_segment_list]) > 0:
            sequence = torch.cat([d["tokens"] for d in current_segment_list], dim=-1)
            if return_token_timestamps:
                token_timestamps = torch.cat(
                    [d["result"]["token_timestamps"][d["idxs"][0] : d["idxs"][1]] for d in current_segment_list],
                    dim=-1,
                )

            if cut_off_length is not None:
                sequence = sequence[-cut_off_length:]
                if return_token_timestamps:
                    token_timestamps = token_timestamps[-cut_off_length:]

            if bos_token_tensor is not None:
                sequence = torch.cat([bos_token_tensor, sequence])
                if return_token_timestamps:
                    token_timestamps = torch.cat(
                        [torch.ones_like(bos_token_tensor, device=device) * 0.0, token_timestamps]
                    )
            sequences.append(sequence)
            if return_token_timestamps:
                token_timestamps_list.append(token_timestamps)
            max_total_length = max(max_total_length, len(sequences[-1]))
        elif bos_token_tensor is not None:
            sequences.append(bos_token_tensor)
            if return_token_timestamps:
                token_timestamps_list.append(torch.ones_like(bos_token_tensor, device=device) * 0.0)
        else:
            sequences.append(torch.tensor([], device=device))
            if return_token_timestamps:
                token_timestamps_list.append(torch.tensor([], device=device))

    max_total_length = cut_off_length + 1 if padding == "max_length" else max_total_length
    for i in range(len(current_segments)):
        pad_length = max_total_length - len(sequences[i])
        pad = (0, pad_length) if padding_side == "right" else (pad_length, 0)

        sequences[i] = F.pad(sequences[i], pad=pad, value=pad_token_id)
        if return_token_timestamps:
            token_timestamps_list[i] = F.pad(
                token_timestamps_list[i],
                pad=pad,
                value=token_timestamps_list[i][-1] if len(token_timestamps_list[i]) > 0 else 0.0,
            )

    sequences = torch.stack(sequences, dim=0)

    if return_token_timestamps:
        token_timestamps = torch.stack(token_timestamps_list, dim=0)
        return sequences, token_timestamps
    else:
        return sequences


class WhisperGenerationMixin(GenerationMixin):
    def _extract_token_timestamps(
        self, generate_outputs, alignment_heads, time_precision=0.02, num_frames=None, num_input_ids=None
    ):
        """
        Calculates token-level timestamps using the encoder-decoder cross-attentions and dynamic time-warping (DTW) to
        map each output token to a position in the input audio. If `num_frames` is specified, the encoder-decoder
        cross-attentions will be cropped before applying DTW.

        Returns:
            tensor containing the timestamps in seconds for each predicted token
        """
        # Create a list with `decoder_layers` elements, each a tensor of shape
        # (batch size, attention_heads, output length, input length).
        cross_attentions = []
        for i in range(self.config.decoder_layers):
            cross_attentions.append(torch.cat([x[i] for x in generate_outputs.cross_attentions], dim=2))

        # Select specific cross-attention layers and heads. This is a tensor
        # of shape (batch size, num selected, output length, input length).
        weights = torch.stack([cross_attentions[l][:, h] for l, h in alignment_heads])
        weights = weights.permute([1, 0, 2, 3])

        weight_length = None

        if "beam_indices" in generate_outputs:
            # If beam search has been used, the output sequences may have been generated for more timesteps than their sequence_lengths
            # since the beam search strategy chooses the most probable sequences at the end of the search.
            # In that case, the cross_attentions weights are too long and we have to make sure that they have the right output_length
            weight_length = (generate_outputs.beam_indices != -1).sum(-1).max()
            weight_length = weight_length if num_input_ids is None else weight_length + num_input_ids

            # beam search takes `decoder_input_ids` into account in the `beam_indices` length
            # but forgot to shift the beam_indices by the number of `decoder_input_ids`
            beam_indices = torch.zeros_like(generate_outputs.beam_indices[:, :weight_length])
            # we actually shif the beam indices here
            beam_indices[:, num_input_ids:] = generate_outputs.beam_indices[:, : weight_length - num_input_ids]

            weights = weights[:, :, :weight_length]

            # If beam index is still -1, it means that the associated token id is EOS
            # We need to replace the index with 0 since index_select gives an error if any of the indexes is -1.
            beam_indices = beam_indices.masked_fill(beam_indices == -1, 0)

            # Select the cross attention from the right beam for each output sequences
            weights = torch.stack(
                [
                    torch.index_select(weights[:, :, i, :], dim=0, index=beam_indices[:, i])
                    for i in range(beam_indices.shape[1])
                ],
                dim=2,
            )

        # make sure timestamps are as long as weights
        input_length = weight_length or cross_attentions[0].shape[2]
        batch_size = generate_outputs.sequences.shape[0]
        timestamps = torch.zeros(
            (batch_size, input_length + 1), dtype=torch.float32, device=generate_outputs.sequences.device
        )

        if num_frames is not None:
            # two cases:
            # 1. num_frames is the same for each sample -> compute the DTW matrix for each sample in parallel
            # 2. num_frames is different, compute the DTW matrix for each sample sequentially

            # we're using np.unique because num_frames can be int/list/tuple
            if isinstance(num_frames, int):
                weights = weights[..., : num_frames // 2]

            elif isinstance(num_frames, (list, tuple, np.ndarray)) and len(np.unique(num_frames)) == 1:
                weights = weights[..., : num_frames[0] // 2]

            elif isinstance(num_frames, (torch.Tensor)) and len(torch.unique(num_frames)) == 1:
                weights = weights[..., : num_frames[0] // 2]

            else:
                # num_frames is of shape (batch_size,) whereas batch_size is truely batch_size*num_return_sequences
                repeat_time = batch_size if isinstance(num_frames, int) else batch_size // len(num_frames)
                num_frames = num_frames.cpu() if isinstance(num_frames, (torch.Tensor)) else num_frames
                num_frames = np.repeat(num_frames, repeat_time)

        if num_frames is None or isinstance(num_frames, int):
            # Normalize and smoothen the weights.
            std = torch.std(weights, dim=-2, keepdim=True, unbiased=False)
            mean = torch.mean(weights, dim=-2, keepdim=True)
            weights = (weights - mean) / std
            weights = _median_filter(weights, self.config.median_filter_width)

            # Average the different cross-attention heads.
            weights = weights.mean(dim=1)

        # Perform dynamic time warping on each element of the batch.
        for batch_idx in range(batch_size):
            if num_frames is not None and isinstance(num_frames, (tuple, list, np.ndarray, torch.Tensor)):
                matrix = weights[batch_idx, ..., : num_frames[batch_idx] // 2]

                # Normalize and smoothen the weights.
                std = torch.std(matrix, dim=-2, keepdim=True, unbiased=False)
                mean = torch.mean(matrix, dim=-2, keepdim=True)
                matrix = (matrix - mean) / std
                matrix = _median_filter(matrix, self.config.median_filter_width)

                # Average the different cross-attention heads.
                matrix = matrix.mean(dim=0)
            else:
                matrix = weights[batch_idx]

            text_indices, time_indices = _dynamic_time_warping(-matrix.cpu().double().numpy())
            jumps = np.pad(np.diff(text_indices), (1, 0), constant_values=1).astype(bool)
            jump_times = time_indices[jumps] * time_precision
            timestamps[batch_idx, 1:] = torch.tensor(jump_times)

        return timestamps

    def generate(
        self,
        input_features: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: bool = False,
        return_timestamps: Optional[bool] = None,
        task: Optional[str] = None,
        language: Optional[Union[str, List[str]]] = None,
        is_multilingual: Optional[bool] = None,
        prompt_ids: Optional[torch.Tensor] = None,
        prompt_condition_type: Optional[str] = None,  # first-segment, all-segments
        condition_on_prev_tokens: Optional[bool] = None,
        temperature: Optional[Union[float, Tuple[float, ...]]] = None,
        compression_ratio_threshold: Optional[float] = None,
        logprob_threshold: Optional[float] = None,
        no_speech_threshold: Optional[float] = None,
        num_segment_frames: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
        time_precision: float = 0.02,
        time_precision_features: float = 0.01,
        return_token_timestamps: Optional[bool] = None,
        return_segments: bool = False,
        return_dict_in_generate: Optional[bool] = None,
        force_unique_generate_call: Optional[bool] = None,
        **kwargs,
    ):
        """
        Transcribes or translates log-mel input features to a sequence of auto-regressively generated token ids.

        <Tip warning={true}>

        Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
        model's default generation configuration. You can override any `generation_config` by passing the corresponding
        parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`.

        For an overview of generation strategies and code examples, check out the [following
        guide](./generation_strategies).

        </Tip>

        Parameters:
            input_features (`torch.Tensor` of shape `(batch_size, feature_size, sequence_length)`, *optional*):
                Float values of log-mel features extracted from the raw speech waveform. The raw speech waveform can be obtained by
                loading a `.flac` or `.wav` audio file into an array of type `List[float]` or a `numpy.ndarray`, *e.g.* via
                the soundfile library (`pip install soundfile`). To prepare the array into `input_features`, the
                [`AutoFeatureExtractor`] should be used for extracting the mel features, padding and conversion into a
                tensor of type `torch.FloatTensor`. See [`~WhisperFeatureExtractor.__call__`] for details.
            generation_config ([`~generation.GenerationConfig`], *optional*):
                The generation configuration to be used as base parametrization for the generation call. `**kwargs`
                passed to generate matching the attributes of `generation_config` will override them. If
                `generation_config` is not provided, the default will be used, which had the following loading
                priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
                configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
                default values, whose documentation should be checked to parameterize generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                Custom logits processors that complement the default logits processors built from arguments and
                generation config. If a logit processor is passed that is already created with the arguments or a
                generation config an error is thrown. This feature is intended for advanced users.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                Custom stopping criteria that complement the default stopping criteria built from arguments and a
                generation config. If a stopping criteria is passed that is already created with the arguments or a
                generation config an error is thrown. This feature is intended for advanced users.
            prefix_allowed_tokens_fn (`Callable[[int, torch.Tensor], List[int]]`, *optional*):
                If provided, this function constraints the beam search to allowed tokens only at each step. If not
                provided no constraint is applied. This function takes 2 arguments: the batch ID `batch_id` and
                `input_ids`. It has to return a list with the allowed tokens for the next generation step conditioned
                on the batch ID `batch_id` and the previously generated tokens `inputs_ids`. This argument is useful
                for constrained generation conditioned on the prefix, as described in [Autoregressive Entity
                Retrieval](https://arxiv.org/abs/2010.00904).
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed to avoid deadlocking with
                `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).
            return_timestamps (`bool`, *optional*):
                Whether to return the timestamps with the text. This enables the `WhisperTimestampsLogitsProcessor`.
            task (`str`, *optional*):
                Task to use for generation, either "translate" or "transcribe". The `model.config.forced_decoder_ids`
                will be updated accordingly.
            language (`str` or list of `str`, *optional*):
                Language token to use for generation, can be either in the form of `<|en|>`, `en` or `english`. For
                batched generation, a list of language tokens can be passed. You can find all the possible language
                tokens in the `model.generation_config.lang_to_id` dictionary.
            is_multilingual (`bool`, *optional*):
                Whether or not the model is multilingual.
            prompt_ids (`torch.Tensor`, *optional*):
                Rank-1 tensor of token IDs created by passing text to [`~WhisperProcessor.get_prompt_ids`] that is
                provided as a prompt to each chunk. This can be used to provide or "prompt-engineer" a context for
                transcription, e.g. custom vocabularies or proper nouns to make it more likely to predict those words
                correctly. It cannot be used in conjunction with `decoder_start_token_id` as it overwrites this value.
            prompt_condition_type (`str`, *optional*):
                Only relevant for long-form transcription. Condition type of `prompt_ids`. 'first-segment' means only the first segment is conditioned on `prompt_ids`. 'all-segments' means each segment is conditioned on `prompt_ids`. Make sure to enable `condition_on_prev_tokens` for 'all-segments'.
                Defaults to 'first-segment'. For short-term transcription only 'first-segment' is possible.
            condition_on_prev_tokens (`bool`, *optional*):
                Only relevant for long-form transcription. Whether to condition each segment on the previous segment.
                As shown in the [the Whisper paper](https://cdn.openai.com/papers/whisper.pdf), this can help to improve
                performance.
            temperature (`float` or list of `float`, *optional*):
                The temperature to be used for generation. Passing a single `float` value and `do_sample=True` activates
                generation using sampling. For long-form transcription, temperature fallback can be activated by passing
                a list of float values such as (0.0, 0.2, 0.4, 0.6, 0.8, 1.0). As shown in the [the Whisper paper](https://cdn.openai.com/papers/whisper.pdf), this can help to improve
                performance.
            compression_ratio_threshold (`float`, *optional*):
                Only relevant for long-form transcription. If defined, the zlib compression rate of each segment will be computed. If the compression rate of
                a segment is higher than `compression_ratio_threshold`, temperature fallback is activated: the generated segment is discarded and the generation is
                repeated using a higher temperature. The intuition behind this feature is that segments with very high compression rates
                suffer from a lot of repetition. The unwanted repetition can be reduced by injecting more randomness by increasing the temperature. If `compression_ratio_threshold` is defined
                make sure that `temperature` is a list of values. A common value for `compression_ratio_threshold` is 1.35.
                As shown in the [the Whisper paper](https://cdn.openai.com/papers/whisper.pdf), this can help to improve
                performance.
            logprob_threshold (`float`, *optional*):
                Only relevant for long-form transcription. If defined, the average log-probability of each segment will be computed. If the log-probability of
                a given segment is lower than `logprob_threshold`, temperature fallback is activated: the generated segment is discarded and the generation is
                repeated using a higher temperature. The intuition behind this feature is that segments of low log-probability
                can be improved by injecting more randomness by increasing the temperature. If `logprob_threshold` is defined
                make sure that `temperature` is a list of values. A common value for `logprob_threshold` is -1.0.
                As shown in the [the Whisper paper](https://cdn.openai.com/papers/whisper.pdf), this can help to improve
                performance.
            no_speech_threshold (`float`, *optional*):
                Only relevant for long-form transcription. If defined, the "no-speech" token combined with the `logprob_threshold`
                is used to determine whether a segment contains only silence. In this case, the transcription for this segment
                is skipped.
                As shown in the [the Whisper paper](https://cdn.openai.com/papers/whisper.pdf), this can help to improve
                performance.
            num_segment_frames (`int`, *optional*):
                The number of frames a single segment is made of. If not defined, `num_segment_frames` defaults to the model's stride
                times the maximum input length.
            attention_mask (`torch.Tensor`, *optional*):
                `attention_mask` needs to be passed when doing long-form transcription using a batch size > 1.
            time_precision (`int`, *optional*, defaults to 0.02):
                The duration of output token in seconds. *E.g.* 0.02 means that a generated token on average accounts
                for 20 ms.
            time_precision_features (`int`, *optional*, defaults to 0.01):
                The duration represented by a feature frame in seconds.
            return_token_timestamps (`bool`, *optional*):
                Whether to return token-level timestamps with the text. This can be used with or without the
                `return_timestamps` option. To get word-level timestamps, use the tokenizer to group the tokens into
                words.
            return_segments (`bool`, *optional*, defaults to `False`):
                Whether to additionally return a list of all segments. Note that this option can only be enabled
                when doing long-form transcription.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~utils.ModelOutput`] instead of just returning the generated tokens.
                Note that when doing long-form transcription, `return_dict_in_generate` can only be enabled when
                `return_segments` is set True. In this case the generation outputs of each segment is added to each
                segment.
            force_unique_generate_call (`bool`, *optional*):
                Whether to force a unique call to the underlying GenerationMixin's [`~generation.GenerationMixin.generate`] method. This is useful for assisted decoding and testing purposes to ensure
                that only one call to [`~generation.GenerationMixin.generate`] is made and therefore decoder input token ids and eos token ids are returned.
            kwargs (`Dict[str, Any]`, *optional*):
                Ad hoc parametrization of `generate_config` and/or additional model-specific kwargs that will be
                forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder
                specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.
        Return:
            [`~utils.ModelOutput`] or `Dict[str, Any]` or `torch.LongTensor`:

                A:
                - [`~utils.ModelOutput`] when `return_dict_in_generate=True` and (`return_timestamps=False` or `force_unique_generate_call=True`), including the decoder input ids and end of sequence id.
                - `Dict[str, Any]` when (`return_dict_in_generate=True` and `return_timestamps=True`) or `return_segments=True` or `return_token_timestamps=True`.
                - `torch.LongTensor` in all other cases, excluding the decoder input ids and end of sequence id.

                The possible [`~utils.ModelOutput`] types are:
                - [`~generation.GenerateEncoderDecoderOutput`]
                - [`~generation.GenerateBeamEncoderDecoderOutput`]

                `segments` is a list of lists (one list per batch element) of `segment`.
                A `segment` is a dictionary with keys `start`, `end`, `tokens`, `idxs`, and `result`.
                - `start`: the start timestamp of the segment.
                - `end`: the end timestamp of the segment.
                - `tokens`: the tokens of the segment, excluding the decoder input ids and end of sequence id.
                - `idxs`: the start (included) and end (excluded) indices of the `tokens` of the segment in the underlying call to GenerationMixin's [`~generation.GenerationMixin.generate`] (present in `result`).
                - `result`: the result of the underlying call to GenerationMixin's [`~generation.GenerationMixin.generate`].

                When `return_timestamps=True`, `return_dict_in_generate=True` applies to each call of the underlying GenerationMixin's [`~generation.GenerationMixin.generate`], with outputs stored in `result` of each `segment`.

        Example:

        - *Longform transcription*: To transcribe or translate audios longer than 30 seconds, process the audio files without truncation and pass all mel features at once to generate. It is necessary to set `return_timestamps=True`.
        Indeed, long-form transcription uses a sequential algorithm based on timestamps predictions, with heuristics like compression ratio threshold, log probability threshold and temperature fallback. This algorithm is described in the [the Whisper original paper](https://cdn.openai.com/papers/whisper.pdf), section *3.8. Long-form Transcription*.

        ```python
        >>> import torch
        >>> from transformers import AutoProcessor, WhisperForConditionalGeneration
        >>> from datasets import load_dataset, Audio

        >>> processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")
        >>> model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
        >>> model.cuda()  # doctest: +IGNORE_RESULT

        >>> # load audios > 30 seconds
        >>> ds = load_dataset("distil-whisper/meanwhile", "default")["test"]
        >>> # resample to 16kHz
        >>> ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        >>> # take first 8 audios and retrieve array
        >>> audio = ds[:8]["audio"]
        >>> audio = [x["array"] for x in audio]

        >>> # make sure to NOT truncate the input audio, to return the `attention_mask` and to pad to the longest audio
        >>> inputs = processor(audio, return_tensors="pt", truncation=False, padding="longest", return_attention_mask=True, sampling_rate=16_000)
        >>> inputs = inputs.to("cuda", torch.float32)

        >>> # transcribe audio to ids
        >>> generated_ids = model.generate(**inputs)

        >>> transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)
        >>> transcription[0]
        " Folks, if you watch the show, you know, I spent a lot of time right over there. Patiently and astutely scrutinizing the boxwood and mahogany chest set of the day's biggest stories developing the central headline pawns, definitely maneuvering an oso topical night to F6, fainting a classic Sicilian, nade door variation on the news, all the while seeing eight moves deep and patiently marshalling the latest press releases into a fisher's shows in Lip Nitsky attack that culminates in the elegant lethal slow-played, all-passant checkmate that is my nightly monologue. But sometimes, sometimes, folks, I. CHEERING AND APPLAUSE Sometimes I startle away, cubside down in the monkey bars of a condemned playground on a super fun site. Get all hept up on goofballs. Rummage that were discarded tag bag of defective toys. Yank out a fist bowl of disembodied doll limbs, toss them on a stained kid's place mat from a defunct dennies. set up a table inside a rusty cargo container down by the Wharf and challenged toothless drifters to the godless bughouse blitz of tournament that is my segment. Meanwhile."
        ```

        - *Shortform transcription*: If passed mel input features are <= 30 seconds, there are two possibilities:
            - `return_timestamps=False`: the whole audio will be transcribed with a single call to GenerationMixin's [`~generation.GenerationMixin.generate`].
            - `return_timestamps=True`: the audio will be transcribed using the same logic as long-form transcription.

        ```python
        >>> import torch
        >>> from transformers import AutoProcessor, WhisperForConditionalGeneration
        >>> from datasets import load_dataset

        >>> processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")
        >>> model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

        >>> inputs = processor(ds[0]["audio"]["array"], return_tensors="pt")
        >>> input_features = inputs.input_features

        >>> generated_ids = model.generate(inputs=input_features)

        >>> transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        >>> transcription
        ' Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.'
        ```

        """
        # 0. deprecate old inputs
        if "inputs" in kwargs:
            input_features = kwargs.pop("inputs")
            warnings.warn(
                "The input name `inputs` is deprecated. Please make sure to use `input_features` instead.",
                FutureWarning,
            )

        # 1. prepare generation config
        generation_config, kwargs = self._prepare_generation_config(generation_config, **kwargs)

        # 2. set global generate variables
        input_stride = self.model.encoder.conv1.stride[0] * self.model.encoder.conv2.stride[0]
        num_segment_frames = input_stride * self.config.max_source_positions
        batch_size, total_input_frames = self._retrieve_total_input_frames(
            input_features=input_features, input_stride=input_stride, kwargs=kwargs
        )
        is_shortform = total_input_frames <= num_segment_frames

        # 3. Make sure generation config is correctly set
        # Make sure the generation config is correctly set depending on whether timestamps are to be returned or not
        return_dict_in_generate = self._set_return_outputs(
            return_dict_in_generate=return_dict_in_generate,
            return_token_timestamps=return_token_timestamps,
            logprob_threshold=logprob_threshold,
            generation_config=generation_config,
        )
        timestamp_begin = self._set_return_timestamps(
            return_timestamps=return_timestamps, is_shortform=is_shortform, generation_config=generation_config
        )
        self._set_language_and_task(
            language=language, task=task, is_multilingual=is_multilingual, generation_config=generation_config
        )
        self._set_num_frames(
            return_token_timestamps=return_token_timestamps, generation_config=generation_config, kwargs=kwargs
        )
        self._set_thresholds_and_condition(
            generation_config=generation_config,
            logprob_threshold=logprob_threshold,
            compression_ratio_threshold=compression_ratio_threshold,
            no_speech_threshold=no_speech_threshold,
            condition_on_prev_tokens=condition_on_prev_tokens,
        )
        self._set_prompt_condition_type(
            generation_config=generation_config,
            prompt_condition_type=prompt_condition_type,
        )

        # pass self.config for backward compatibility
        init_tokens = self._retrieve_init_tokens(
            input_features,
            batch_size=batch_size,
            generation_config=generation_config,
            config=self.config,
            num_segment_frames=num_segment_frames,
            kwargs=kwargs,
        )
        # passing `decoder_input_ids` is deprecated - the only exception is for assisted generation
        # where the input ids are handled explicitly by the generate method
        self._check_decoder_input_ids(kwargs=kwargs)

        # 3. Retrieve logits processors
        device = kwargs["encoder_outputs"][0].device if "encoder_outputs" in kwargs else input_features.device
        begin_index = init_tokens.shape[1]
        num_beams = kwargs.get(
            "num_beams",
            generation_config.num_beams
            if hasattr(generation_config, "num_beams") and generation_config.num_beams is not None
            else 1,
        )
        if "assistant_model" in kwargs:
            # speculative decoding: the model should be able to return eos token
            generation_config.begin_suppress_tokens = None

        logits_processor = self._retrieve_logit_processors(
            generation_config=generation_config,
            logits_processor=logits_processor,
            begin_index=begin_index,  # begin index is index of first generated decoder token
            num_beams=num_beams,
            device=device,
        )

        # 4 Set and retrieve global generation variables
        self._set_condition_on_prev_tokens(
            condition_on_prev_tokens=condition_on_prev_tokens, generation_config=generation_config
        )

        temperatures = [temperature] if not isinstance(temperature, (list, tuple)) else temperature
        temperature = temperatures[0]

        max_frames, seek = self._retrieve_max_frames_and_seek(
            batch_size=batch_size,
            attention_mask=attention_mask,
            total_input_frames=total_input_frames,
            is_shortform=is_shortform,
        )

        # 5 Prepare running variables, list for generation
        num_return_sequences = generation_config.num_return_sequences
        (
            batch_idx_map,
            cur_bsz,
            input_features,
            seek,
            max_frames,
            init_tokens,
            do_condition_on_prev_tokens,
        ) = self._expand_variables_for_generation(
            input_features=input_features,
            seek=seek,
            max_frames=max_frames,
            init_tokens=init_tokens,
            batch_size=batch_size,
            condition_on_prev_tokens=condition_on_prev_tokens,
            generation_config=generation_config,
        )

        current_segments = self._prepare_segments(
            prompt_ids=prompt_ids,
            batch_size=cur_bsz,
            generation_config=generation_config,
        )
        # 5bis speculative decoding: ensure the assistant model does only one call to generate and therefore returns decoder input token ids and eos token id
        # we set a flag in the generation config to force the model to make only one call to generate and return the decoder input token ids and eos token id
        if "assistant_model" in kwargs:
            assistant_model = kwargs["assistant_model"]
            assistant_model.generation_config.force_unique_generate_call = True

        if force_unique_generate_call is None:
            if hasattr(generation_config, "force_unique_generate_call"):
                force_unique_generate_call = generation_config.force_unique_generate_call
            elif hasattr(self.generation_config, "force_unique_generate_call"):
                force_unique_generate_call = self.generation_config.force_unique_generate_call
            else:
                force_unique_generate_call = False

        # 6 Transcribe audio until we reach the end of all input audios
        while (seek < max_frames).any():
            # 6.1 NOTE: When in longform transcription mode and batch size > 1 we need to dynamically reduce the batch size during the loop
            # in case one audio finished earlier than another one. Thus, we need to keep a table of "previous-index-2-current-index" in order
            # to know which original audio is being decoded
            # Set updated index map, duration of previously decoded chunks and number of max frames of current decoding chunk
            input_features, cur_bsz, batch_idx_map = self._maybe_reduce_batch(
                input_features=input_features,
                seek=seek,
                max_frames=max_frames,
                cur_bsz=cur_bsz,
                batch_idx_map=batch_idx_map,
            )
            time_offset = (
                seek.to(torch.float32 if device.type == "mps" else torch.float64) * time_precision / input_stride
            )
            seek_num_frames = (max_frames - seek).clamp(max=num_segment_frames)

            # 6.2 cut out next 30s segment from input features
            segment_input = self._get_input_segment(
                input_features=input_features,
                seek=seek,
                seek_num_frames=seek_num_frames,
                num_segment_frames=num_segment_frames,
                cur_bsz=cur_bsz,
                batch_idx_map=batch_idx_map,
            )

            # 6.3 prepare decoder input ids
            suppress_tokens = _get_attr_from_logit_processors(
                logits_processor, SuppressTokensLogitsProcessor, "suppress_tokens"
            )

            decoder_input_ids, kwargs = self._prepare_decoder_input_ids(
                cur_bsz=cur_bsz,
                init_tokens=init_tokens,
                current_segments=current_segments,
                batch_idx_map=batch_idx_map,
                do_condition_on_prev_tokens=do_condition_on_prev_tokens,
                prompt_ids=prompt_ids,
                generation_config=generation_config,
                config=self.config,
                device=init_tokens.device,
                suppress_tokens=suppress_tokens,
                timestamp_begin=timestamp_begin,
                kwargs=kwargs,
            )

            # 6.4 set max new tokens or max length
            self._set_max_new_tokens_and_length(
                config=self.config,
                decoder_input_ids=decoder_input_ids,
                generation_config=generation_config,
            )

            # 6.5 Set current `begin_index` for all logit processors
            if logits_processor is not None:
                for proc in logits_processor:
                    if hasattr(proc, "set_begin_index"):
                        proc.set_begin_index(decoder_input_ids.shape[-1])

            # 6.6 Run generate with fallback
            (
                seek_sequences,
                seek_outputs,
                should_skip,
                do_condition_on_prev_tokens,
                model_output_type,
            ) = self.generate_with_fallback(
                segment_input=segment_input,
                decoder_input_ids=decoder_input_ids,
                cur_bsz=cur_bsz,
                batch_idx_map=batch_idx_map,
                seek=seek,
                num_segment_frames=num_segment_frames,
                max_frames=max_frames,
                temperatures=temperatures,
                generation_config=generation_config,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                synced_gpus=synced_gpus,
                return_token_timestamps=return_token_timestamps,
                do_condition_on_prev_tokens=do_condition_on_prev_tokens,
                is_shortform=is_shortform,
                batch_size=batch_size,
                attention_mask=attention_mask,
                kwargs=kwargs,
            )

            # 6.7 In every generated sequence, split by timestamp tokens and extract segments
            for i, seek_sequence in enumerate(seek_sequences):
                prev_i = batch_idx_map[i]

                if should_skip[i]:
                    seek[prev_i] += seek_num_frames[prev_i]
                    continue

                segments, segment_offset = self._retrieve_segment(
                    seek_sequence=seek_sequence,
                    seek_outputs=seek_outputs,
                    time_offset=time_offset,
                    timestamp_begin=timestamp_begin,
                    seek_num_frames=seek_num_frames,
                    time_precision=time_precision,
                    time_precision_features=time_precision_features,
                    input_stride=input_stride,
                    prev_idx=prev_i,
                    idx=i,
                    return_token_timestamps=return_token_timestamps,
                    decoder_input_ids=decoder_input_ids,
                )

                seek[prev_i] += segment_offset

                current_segments[prev_i] += segments

            if force_unique_generate_call:
                break

        # 7. Once all segments are added to the list of all segments, called `current_segments`, we extract the predicted
        # output tokens from the list of dicts. If we use batch size > 1, we make sure to pad the output
        final_segments = (
            [x[1:] for x in current_segments]
            if (prompt_ids is not None and generation_config.prompt_condition_type == "first-segment")
            else current_segments
        )

        # if return_dict_in_generate=True and we forced a unique call to generate or return_timestamps=False, meaning we are sure only one call to generate has been made,
        # -> we can return a ModelOutput
        # otherwise, return_dict_in_generate is applied in the 'result' of each segment in final_segments
        if (
            return_dict_in_generate
            and generation_config.return_dict_in_generate
            and (force_unique_generate_call or not return_timestamps)
        ):
            # only one call to generate_with_fallback, we can return a ModelOutput
            outputs = self._stack_split_outputs(seek_outputs, model_output_type, self.device, kwargs)
            if num_return_sequences > 1:
                if hasattr(outputs, "encoder_attentions") and outputs.encoder_attentions is not None:
                    outputs.encoder_attentions = tuple(
                        outputs.encoder_attentions[i][::num_return_sequences]
                        for i in range(len(outputs.encoder_attentions))
                    )
                if hasattr(outputs, "encoder_hidden_states") and outputs.encoder_hidden_states is not None:
                    outputs.encoder_hidden_states = tuple(
                        outputs.encoder_hidden_states[i][::num_return_sequences]
                        for i in range(len(outputs.encoder_hidden_states))
                    )
            return outputs

        padded_outputs = _pad_to_max_length(
            current_segments=final_segments,
            pad_token_id=generation_config.pad_token_id,
            device=self.device,
            padding_side="right",
            return_token_timestamps=return_token_timestamps,
            force_unique_generate_call=force_unique_generate_call,
        )

        if return_dict_in_generate and generation_config.return_dict_in_generate:
            logger.warning_once(
                "You have passed `return_dict_in_generate=True` and `return_timestamps=True`, this automatically sets `return_segments=True` to access the resuls of the underlying calls to GenerationMixin's generate in the returned `segments`."
            )
            return_segments = True
        elif not return_segments and not return_token_timestamps:
            return padded_outputs

        if return_token_timestamps:
            sequences, token_timestamps = padded_outputs
            outputs = {
                "sequences": sequences,
                "token_timestamps": token_timestamps,
            }
        else:
            sequences = padded_outputs
            outputs = {
                "sequences": sequences,
            }

        if return_segments:
            outputs["segments"] = final_segments

        return outputs

    def generate_with_fallback(
        self,
        segment_input,
        decoder_input_ids,
        cur_bsz,
        batch_idx_map,
        seek,
        num_segment_frames,
        max_frames,
        temperatures,
        generation_config,
        logits_processor,
        stopping_criteria,
        prefix_allowed_tokens_fn,
        synced_gpus,
        return_token_timestamps,
        do_condition_on_prev_tokens,
        is_shortform,
        batch_size,
        attention_mask,
        kwargs,
    ):
        kwargs = copy.copy(kwargs)

        # 6.6 Batch generate current chunk
        seek_sequence_list = [None for _ in range(cur_bsz)]
        seek_outputs_list = [None for _ in range(cur_bsz)]
        needs_fallback = [False for _ in range(cur_bsz)]
        should_skip = [False for _ in range(cur_bsz)]
        fallback_index_map = list(range(cur_bsz))
        if generation_config.no_speech_threshold is not None:
            self._setup_no_speech_detection(logits_processor, segment_input, decoder_input_ids, kwargs)

        for fallback_idx, temperature in enumerate(temperatures):
            generation_config.do_sample = temperature is not None and temperature > 0.0
            generation_config.temperature = temperature if generation_config.do_sample else 1.0
            if generation_config.do_sample:
                generation_config.num_beams = 1

            generate_kwargs = copy.copy(kwargs)
            for key in ["do_sample", "temperature", "num_beams"]:
                if key in generate_kwargs:
                    del generate_kwargs[key]

            cur_bsz = decoder_input_ids.shape[0]
            if generation_config.cache_implementation == "static" and cur_bsz < batch_size:
                segment_input = F.pad(segment_input, (0, 0, 0, 0, 0, batch_size - cur_bsz), value=0)
                decoder_input_ids = F.pad(
                    decoder_input_ids, (0, 0, 0, batch_size - cur_bsz), value=generation_config.pad_token_id
                )
                if generate_kwargs.get("decoder_attention_mask") is not None:
                    generate_kwargs["decoder_attention_mask"] = F.pad(
                        generate_kwargs["decoder_attention_mask"], (0, 0, 0, batch_size - cur_bsz), value=True
                    )
                if generate_kwargs.get("encoder_outputs") is not None:
                    generate_kwargs["encoder_outputs"] = F.pad(
                        generate_kwargs["encoder_outputs"], (0, 0, 0, 0, 0, batch_size - cur_bsz), value=0
                    )

            seek_outputs = super().generate(
                segment_input,
                generation_config=generation_config,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                synced_gpus=synced_gpus,
                decoder_input_ids=decoder_input_ids,
                attention_mask=attention_mask,
                **generate_kwargs,
            )

            model_output_type = type(seek_outputs)

            # post-process sequence tokens and outputs to be in list form
            seek_sequences, seek_outputs = self._postprocess_outputs(
                seek_outputs=seek_outputs,
                decoder_input_ids=decoder_input_ids,
                return_token_timestamps=return_token_timestamps,
                generation_config=generation_config,
                is_shortform=is_shortform,
            )

            if cur_bsz < batch_size:
                seek_sequences = seek_sequences[:cur_bsz]
                seek_outputs = seek_outputs[:cur_bsz]

            # 6.7 Extract cut sequences from every sequence and check if fallback should be applied
            # Loop over each decoded audio individually as each decoding can be of a different length
            new_fallback_index_map = []
            new_segment_input = []
            new_decoder_input_ids = []
            new_decoder_attention_mask = []

            for i, seek_sequence in enumerate(seek_sequences):
                # remove all padding tokens, except for the eos token
                if seek_sequence[-1] == generation_config.pad_token_id:
                    num_paddings = (seek_sequence == generation_config.pad_token_id).sum()
                    if generation_config.pad_token_id == generation_config.eos_token_id:
                        # we do not remove the eos token id since it is needed for avg logprob calculation in _need_fallback
                        num_paddings -= 1
                    if num_paddings != 0:
                        seek_sequence = seek_sequence[:-num_paddings]

                # check which sequences in batch need fallback & which should be skipped
                needs_fallback[i], should_skip[i] = self._need_fallback(
                    seek_sequence,
                    seek_outputs,
                    i,
                    logits_processor,
                    generation_config,
                    self.config.vocab_size,
                    temperature,
                )

                # remove eos token
                if seek_sequence[-1] == generation_config.eos_token_id:
                    seek_sequence = seek_sequence[:-1]

                seek_sequence_list[fallback_index_map[i]] = seek_sequence
                seek_outputs_list[fallback_index_map[i]] = seek_outputs[i]
                is_low_temperature = temperature is None or temperature < 0.5
                do_condition_on_prev_tokens[fallback_index_map[i]] = (
                    generation_config.condition_on_prev_tokens and is_low_temperature
                )

                if needs_fallback[i]:
                    new_fallback_index_map.append(fallback_index_map[i])
                    new_segment_input.append(segment_input[i])
                    new_decoder_input_ids.append(decoder_input_ids[i])
                    if "decoder_attention_mask" in kwargs:
                        new_decoder_attention_mask.append(kwargs["decoder_attention_mask"][i])

            fallback_index_map = new_fallback_index_map

            # if no sequence needs to be run with temperature fallback, we're finished
            if len(fallback_index_map) == 0 or fallback_idx == len(temperatures) - 1:
                seek_sequences = seek_sequence_list
                seek_outputs = seek_outputs_list
                break

            # if we're still in the loop, make sure that decoder_input_ids and segment inputs are tensors
            decoder_input_ids = torch.stack(new_decoder_input_ids)
            segment_input = torch.stack(new_segment_input)
            if "decoder_attention_mask" in kwargs:
                kwargs["decoder_attention_mask"] = torch.stack(new_decoder_attention_mask)

        return seek_sequences, seek_outputs, should_skip, do_condition_on_prev_tokens, model_output_type

    @staticmethod
    def _prepare_segments(prompt_ids, batch_size, generation_config):
        if prompt_ids is not None and generation_config.prompt_condition_type == "first-segment":
            prev_sot_token_id = getattr(generation_config, "prev_sot_token_id", None)
            prompt_ids = prompt_ids[1:] if prompt_ids[0] == prev_sot_token_id else prompt_ids
            current_segments = [[{"tokens": prompt_ids}] for _ in range(batch_size)]
        else:
            current_segments = [[] for _ in range(batch_size)]

        return current_segments

    def _postprocess_outputs(
        self,
        seek_outputs,
        decoder_input_ids,
        return_token_timestamps,
        generation_config,
        is_shortform,
    ):
        # remove all previously passed decoder input ids
        # should happen only if it is the first generated segment
        start_idx = decoder_input_ids.shape[-1]

        if isinstance(seek_outputs, torch.Tensor):
            return seek_outputs[:, start_idx:], seek_outputs

        if return_token_timestamps and hasattr(generation_config, "alignment_heads"):
            num_frames = getattr(generation_config, "num_frames", None)
            seek_outputs["token_timestamps"] = self._extract_token_timestamps(
                seek_outputs,
                generation_config.alignment_heads,
                num_frames=num_frames,
                num_input_ids=decoder_input_ids.shape[-1],
            )

        def split_by_batch_index(values, key, batch_idx, is_shortform, beam_indices=None):
            if beam_indices is not None and key == "scores":
                return [v[beam_idx].cpu() for (v, beam_idx) in zip(values, beam_indices[batch_idx][: len(values)])]
            if key in ["scores", "encoder_attentions", "encoder_hidden_states", "logits"]:
                return [v[batch_idx].cpu() for v in values]
            if key in ["decoder_attentions", "decoder_hidden_states", "cross_attentions"]:
                return tuple(tuple(w[batch_idx][None].cpu() for w in v) for v in values)
            elif key == "past_key_values":
                if not is_shortform:
                    # we don't save `past_key_values` as this is too costly for longform
                    return None
                elif isinstance(values, EncoderDecoderCache):
                    all_past_key_values = []
                    for layer_idx in range(self.config.decoder_layers):
                        layer_past_key_values = []
                        for cache_cls in [values.self_attention_cache, values.cross_attention_cache]:
                            for v in [cache_cls.key_cache, cache_cls.value_cache]:
                                layer_past_key_values.append(v[layer_idx][batch_idx][None].cpu())
                        all_past_key_values.append(tuple(layer_past_key_values))
                    return tuple(all_past_key_values)
                else:
                    all_past_key_values = []
                    for v in range(len(values)):
                        layer_past_key_values = []
                        for w in values[v]:
                            if len(w) != 0:
                                layer_past_key_values.append(w[batch_idx][None].cpu())
                            else:
                                layer_past_key_values.append(w)
                        all_past_key_values.append(tuple(layer_past_key_values))
                    return tuple(all_past_key_values)

            return values[batch_idx].cpu()

        sequence_tokens = seek_outputs["sequences"][:, start_idx:]
        seek_outputs = [
            {
                k: split_by_batch_index(v, k, i, is_shortform, beam_indices=seek_outputs.get("beam_indices"))
                for k, v in seek_outputs.items()
            }
            for i in range(sequence_tokens.shape[0])
        ]

        return sequence_tokens, seek_outputs

    def _stack_split_outputs(self, seek_outputs, model_output_type, device, kwargs):
        # Stack back seek_outputs tensors after splitting them with the split_by_batch_index method
        outputs = {}
        for key in seek_outputs[0].keys():
            if key in ["sequences", "beam_indices", "token_timestamps"]:
                outputs[key] = torch.stack([v[key] for v in seek_outputs], dim=0).to(device)
            elif key in ["scores", "encoder_attentions", "encoder_hidden_states", "logits"]:
                outputs[key] = tuple(
                    torch.stack([v[key][i] for v in seek_outputs]).to(device) for i in range(len(seek_outputs[0][key]))
                )
            elif key == "sequences_scores":
                outputs[key] = torch.stack([v[key] for v in seek_outputs], dim=0).to(device)
            elif key in ["decoder_attentions", "decoder_hidden_states", "cross_attentions"]:
                outputs[key] = tuple(
                    tuple(
                        torch.stack([v[key][i][j] for v in seek_outputs]).squeeze(1).to(device)
                        for j in range(len(seek_outputs[0][key][0]))
                    )
                    for i in range(len(seek_outputs[0][key]))
                )
            elif key == "past_key_values":
                past_key_value_type = kwargs.get("past_key_values")
                if seek_outputs[0][key] is not None:
                    outputs[key] = tuple(
                        tuple(
                            torch.stack([v[key][i][j] for v in seek_outputs]).squeeze(1).to(device)
                            for j in range(len(seek_outputs[0][key][0]))
                        )
                        for i in range(len(seek_outputs[0][key]))
                    )
                    if past_key_value_type is not None and isinstance(past_key_value_type, EncoderDecoderCache):
                        outputs[key] = past_key_value_type.from_legacy_cache(outputs[key])
                else:
                    outputs[key] = None

        token_timestamps = outputs.get("token_timestamps", None)
        if token_timestamps is not None:
            model_output_type = dict

        return model_output_type(**outputs)

    def _need_fallback(
        self,
        seek_sequence,
        seek_outputs,
        index,
        logits_processor,
        generation_config,
        vocab_size,
        temperature,
    ):
        needs_fallback = False
        should_skip = False
        if generation_config.compression_ratio_threshold is not None:
            compression_ratio = self._retrieve_compression_ratio(seek_sequence, vocab_size)

            if compression_ratio > generation_config.compression_ratio_threshold:
                needs_fallback = True

        if generation_config.logprob_threshold is not None:
            if hasattr(seek_outputs[0], "sequences_scores"):
                logprobs = [s["sequences_scores"] for s in seek_outputs][index]
            else:
                scores = seek_outputs[index]["scores"]
                logprobs = self._retrieve_avg_logprobs(
                    scores,
                    seek_sequence,
                    temperature,
                )

            if logprobs < generation_config.logprob_threshold:
                needs_fallback = True

        if generation_config.no_speech_threshold is not None:
            no_speech_prob = _get_attr_from_logit_processors(
                logits_processor, WhisperNoSpeechDetection, "no_speech_prob"
            )

            if (
                logprobs < generation_config.logprob_threshold
                and no_speech_prob[index] > generation_config.no_speech_threshold
            ):
                needs_fallback = False
                should_skip = True

        return needs_fallback, should_skip

    def _expand_variables_for_generation(
        self, input_features, seek, max_frames, init_tokens, batch_size, condition_on_prev_tokens, generation_config
    ):
        if generation_config.num_return_sequences is not None and generation_config.num_return_sequences > 1:
            batch_idx_map = list(range(batch_size * generation_config.num_return_sequences))
            cur_bsz = len(batch_idx_map)
            do_condition_on_prev_tokens = [condition_on_prev_tokens for _ in range(len(batch_idx_map))]
            input_features = input_features.repeat_interleave(generation_config.num_return_sequences, dim=0)
            seek = seek.repeat_interleave(generation_config.num_return_sequences, dim=0)
            max_frames = max_frames.repeat_interleave(generation_config.num_return_sequences, dim=0)
            init_tokens = init_tokens.repeat_interleave(generation_config.num_return_sequences, dim=0)
            generation_config.num_return_sequences = 1
        else:
            cur_bsz = batch_size
            batch_idx_map = list(range(cur_bsz))
            do_condition_on_prev_tokens = [condition_on_prev_tokens for _ in range(cur_bsz)]

        return (
            batch_idx_map,
            cur_bsz,
            input_features,
            seek,
            max_frames,
            init_tokens,
            do_condition_on_prev_tokens,
        )

    @staticmethod
    def _setup_no_speech_detection(logits_processor, segment_input, decoder_input_ids, kwargs):
        set_inputs = _get_attr_from_logit_processors(logits_processor, WhisperNoSpeechDetection, "set_inputs")
        extra_kwargs = {k: v for k, v in kwargs.items() if torch.is_tensor(v)}
        set_inputs({"inputs": segment_input, "decoder_input_ids": decoder_input_ids, **extra_kwargs})

    @staticmethod
    def _retrieve_total_input_frames(input_features, input_stride, kwargs):
        if input_features is not None:
            return input_features.shape[0], input_features.shape[-1]

        if "encoder_outputs" in kwargs:
            encoder_outputs_shape = (
                kwargs["encoder_outputs"][0].shape
                if isinstance(kwargs["encoder_outputs"], BaseModelOutput)
                else kwargs["encoder_outputs"].shape
            )
            return encoder_outputs_shape[0], encoder_outputs_shape[1] * input_stride

        raise ValueError("Make sure to provide either `input_features` or `encoder_outputs` to `generate`.")

    @staticmethod
    def _maybe_warn_unused_inputs(
        condition_on_prev_tokens,
        temperature,
        compression_ratio_threshold,
        logprob_threshold,
        no_speech_threshold,
        total_input_frames,
    ):
        warning_prefix = (
            f"Audio input consists of only {total_input_frames}. "
            "Short-form transcription is activated."
            "{}, but will be ignored."
        )
        if condition_on_prev_tokens is not None:
            logger.warning(warning_prefix.format(f"condition_on_prev_tokens is set to {condition_on_prev_tokens}"))

        if compression_ratio_threshold is not None:
            logger.warning(
                warning_prefix.format(f"compression_ratio_threshold is set to {compression_ratio_threshold}")
            )

        if logprob_threshold is not None:
            logger.warning(warning_prefix.format(f"logprob_threshold is set to {logprob_threshold}"))

        if no_speech_threshold is not None:
            logger.warning(warning_prefix.format(f"no_speech_threshold is set to {no_speech_threshold}"))

    @staticmethod
    def _set_return_outputs(return_dict_in_generate, return_token_timestamps, logprob_threshold, generation_config):
        if return_dict_in_generate is None:
            return_dict_in_generate = generation_config.return_dict_in_generate
        else:
            generation_config.return_dict_in_generate = return_dict_in_generate

        generation_config.return_token_timestamps = return_token_timestamps
        if return_token_timestamps:
            generation_config.return_dict_in_generate = True
            generation_config.output_attentions = True
            generation_config.output_scores = True

        if logprob_threshold is not None:
            generation_config.return_dict_in_generate = True
            generation_config.output_scores = True

        return return_dict_in_generate

    def _set_return_timestamps(self, return_timestamps, is_shortform, generation_config):
        if return_timestamps is None and hasattr(generation_config, "return_timestamps"):
            return_timestamps = generation_config.return_timestamps

        if not is_shortform:
            if return_timestamps is False:
                raise ValueError(
                    "You have passed more than 3000 mel input features (> 30 seconds) which automatically enables long-form generation which "
                    "requires the model to predict timestamp tokens. Please either pass `return_timestamps=True` or make sure to pass no more than 3000 mel input features."
                )

            logger.info("Setting `return_timestamps=True` for long-form generation.")
            return_timestamps = True

        if return_timestamps and not hasattr(generation_config, "no_timestamps_token_id"):
            raise ValueError(
                "You are trying to return timestamps, but the generation config is not properly set. "
                "Make sure to initialize the generation config with the correct attributes that are needed such as `no_timestamps_token_id`. "
                "For more details on how to generate the approtiate config, refer to https://github.com/huggingface/transformers/issues/21878#issuecomment-1451902363"
            )

        generation_config.return_timestamps = return_timestamps

        if hasattr(generation_config, "no_timestamps_token_id"):
            timestamp_begin = generation_config.no_timestamps_token_id + 1
        else:
            # BC for models missing the `no_timestamps_token_id` in the generation config when generating short-form with no timestamps
            # We set the timestamp begin token larger than the vocab size, such that the timestamp condition is never met in the decoding loop
            timestamp_begin = self.config.vocab_size + 1

        return timestamp_begin

    @staticmethod
    def _set_language_and_task(language, task, is_multilingual, generation_config):
        if is_multilingual is not None:
            if not hasattr(generation_config, "is_multilingual"):
                raise ValueError(
                    "The generation config is outdated and is thus not compatible with the `is_multilingual` argument "
                    "to `generate`. Please update the generation config as per the instructions "
                    "https://github.com/huggingface/transformers/issues/25084#issuecomment-1664398224"
                )
            generation_config.is_multilingual = is_multilingual

        if hasattr(generation_config, "is_multilingual") and not generation_config.is_multilingual:
            if task is not None or language is not None:
                raise ValueError(
                    "Cannot specify `task` or `language` for an English-only model. If the model is intended to be "
                    "multilingual, pass `is_multilingual=True` to generate, or update the generation config."
                )

        if language is not None:
            if not hasattr(generation_config, "lang_to_id"):
                raise ValueError(
                    "The generation config is outdated and is thus not compatible with the `language` argument "
                    "to `generate`. Either set the language using the `forced_decoder_ids` in the model config, "
                    "or update the generation config as per the instructions https://github.com/huggingface/transformers/issues/25084#issuecomment-1664398224"
                )
            generation_config.language = language

        if task is not None:
            if not hasattr(generation_config, "task_to_id"):
                raise ValueError(
                    "The generation config is outdated and is thus not compatible with the `task` argument "
                    "to `generate`. Either set the task using the `forced_decoder_ids` in the model config, "
                    "or update the generation config as per the instructions https://github.com/huggingface/transformers/issues/25084#issuecomment-1664398224"
                )
            generation_config.task = task

    def _retrieve_init_tokens(self, input_features, batch_size, generation_config, config, num_segment_frames, kwargs):
        def replace_or_add(lst: List[int], num: int, itr: Iterator[int]):
            """short function to replace num with a itr in lst"""
            found = any(i in lst for i in itr)
            if found:
                lst = [num if i in itr else i for i in lst]
            else:
                lst.append(num)
            return lst

        def language_to_id(language: str) -> int:
            language = language.lower()
            if language in generation_config.lang_to_id.keys():
                language_token = language
            elif language in TO_LANGUAGE_CODE.keys():
                language_token = f"<|{TO_LANGUAGE_CODE[language]}|>"
            elif language in TO_LANGUAGE_CODE.values():
                language_token = f"<|{language}|>"
            else:
                is_language_code = len(language) == 2
                raise ValueError(
                    f"Unsupported language: {language}. Language should be one of:"
                    f" {list(TO_LANGUAGE_CODE.values()) if is_language_code else list(TO_LANGUAGE_CODE.keys())}."
                )
            if language_token not in generation_config.lang_to_id:
                raise ValueError(
                    f"{language_token} is not supported by this specific model as it is not in the `generation_config.lang_to_id`."
                    "(You should just add it to the generation config)"
                )

            return generation_config.lang_to_id[language_token]

        task = getattr(generation_config, "task", None)
        language = getattr(generation_config, "language", None)

        forced_decoder_ids = generation_config.forced_decoder_ids
        if forced_decoder_ids is not None:
            if language is None and task is None and forced_decoder_ids[0][1] is None:
                logger.warning_once(
                    "Due to a bug fix in https://github.com/huggingface/transformers/pull/28687 transcription using a multilingual Whisper will default to language detection followed by transcription instead of translation to English."
                    "This might be a breaking change for your use case. If you want to instead always translate your audio to English, make sure to pass `language='en'`."
                )
        elif hasattr(config, "forced_decoder_ids") and config.forced_decoder_ids is not None:
            forced_decoder_ids = config.forced_decoder_ids

        if forced_decoder_ids is not None and task is not None:
            logger.warning_once(
                f"You have passed task={task}, but also have set `forced_decoder_ids` to {forced_decoder_ids} which creates a conflict. `forced_decoder_ids` will be ignored in favor of task={task}."
            )
            forced_decoder_ids = None
        elif forced_decoder_ids is not None and language is not None:
            logger.warning_once(
                f"You have passed language={language}, but also have set `forced_decoder_ids` to {forced_decoder_ids} which creates a conflict. `forced_decoder_ids` will be ignored in favor of language={language}."
            )
            forced_decoder_ids = None

        init_tokens = [generation_config.decoder_start_token_id]
        if forced_decoder_ids is not None and forced_decoder_ids[0][0] == 1:
            i = 1
            while len(forced_decoder_ids) > 0 and forced_decoder_ids[0][0] == i:
                init_tokens += [forced_decoder_ids[0][1]]
                forced_decoder_ids = forced_decoder_ids[1:]
                i += 1

            if len(forced_decoder_ids) > 0:
                raise ValueError(
                    f"You are using token ids in `forced_decoder_ids` that do not seem to correctly follow the prompt pattern of Whisper. Make sure that {forced_decoder_ids} has an entry for all indices >= 1 and < {forced_decoder_ids[0][0]}.",
                )

        # from v4.39 the forced decoder ids are always None in favour of decoder input ids
        generation_config.forced_decoder_ids = None

        is_lang_id_undefined = len(init_tokens) <= 1 or (len(init_tokens) > 1 and init_tokens[1] is None)

        # Make sure language is a list of strings of the correct length
        if isinstance(language, (list, tuple)):
            if any(l is None for l in language):
                raise TypeError(
                    "Expected `language` to be `None`, a single string (e.g. `'en'`), or a list of strings with length equal to the batch size (e.g. `('en', 'fr')` for a batch size of 2). Got a list containing `None`."
                )
            if len(language) != batch_size:
                raise ValueError(
                    "When passing a list of languages, the length of the list must match the batch size. "
                    f"Expected length of {batch_size}, but got {len(language)} languages."
                )
            languages = language
        elif language is None:
            # Language will be detected for each item in batch
            languages = [None] * batch_size
        else:
            languages = [language]  # Use a length-1 list now, broadcast later

        # Separate init_tokens for each language
        init_tokens = [copy.copy(init_tokens) for _ in languages]

        # Update init_tokens with languages
        lang_ids = None
        if language is not None:
            lang_ids = [language_to_id(l) for l in languages]
        elif hasattr(generation_config, "lang_to_id") and is_lang_id_undefined:
            # language is not defined or intentially set to `None` to trigger language detection
            lang_ids = self.detect_language(
                input_features=input_features,
                encoder_outputs=kwargs.get("encoder_outputs", None),
                generation_config=generation_config,
                num_segment_frames=num_segment_frames,
            ).tolist()
        if lang_ids is not None:
            # append or replace lang_ids to init_tokens
            for i in range(len(init_tokens)):
                if len(init_tokens[i]) > 1:
                    init_tokens[i][1] = lang_ids[i]
                else:
                    init_tokens[i].append(lang_ids[i])
        del languages

        # Update init_tokens with task
        for i in range(len(init_tokens)):
            if task is not None:
                if task in TASK_IDS:
                    init_tokens[i].append(generation_config.task_to_id[generation_config.task])
                    task_id = generation_config.task_to_id[generation_config.task]

                    # if task is defined it'll overwrite task ids that might have already been defined via the generation_config
                    replace_or_add(init_tokens[i], task_id, generation_config.task_to_id.values())
                else:
                    raise ValueError(f"The `{task}`task is not supported. The task should be one of `{TASK_IDS}`")
            elif language is not None and hasattr(generation_config, "task_to_id"):
                # if language is defined, but no task id is in `init_tokens`, default to transcribe
                if not any(ti in init_tokens[i] for ti in generation_config.task_to_id.values()):
                    init_tokens[i].append(generation_config.task_to_id["transcribe"])

            if (
                not generation_config.return_timestamps
                and hasattr(generation_config, "no_timestamps_token_id")
                and init_tokens[i][-1] != generation_config.no_timestamps_token_id
            ):
                init_tokens[i].append(generation_config.no_timestamps_token_id)
            elif (
                generation_config.return_timestamps and init_tokens[i][-1] == generation_config.no_timestamps_token_id
            ):
                logger.info(
                    "<|notimestamps|> prompt token is removed from generation_config since `return_timestamps` is set to `'True'`."
                )
                init_tokens[i] = init_tokens[i][:-1]

            # let's make sure we don't pass `None` tokens as prompt tokens
            init_tokens[i] = [t for t in init_tokens[i] if t is not None]

        return torch.as_tensor(init_tokens, dtype=torch.long, device=self.device).expand(batch_size, -1)

    def detect_language(
        self,
        input_features: Optional[torch.FloatTensor] = None,
        encoder_outputs: Optional[Union[torch.FloatTensor, BaseModelOutput]] = None,
        generation_config: Optional[GenerationConfig] = None,
        num_segment_frames: int = 3000,
    ) -> torch.Tensor:
        """
        Detects language from log-mel input features or encoder_outputs

        Parameters:
            input_features (`torch.Tensor` of shape `(batch_size, feature_size, sequence_length)`, *optional*):
                Float values of log-mel features extracted from the raw speech waveform. The raw speech waveform can be obtained by
                loading a `.flac` or `.wav` audio file into an array of type `List[float]` or a `numpy.ndarray`, *e.g.* via
                the soundfile library (`pip install soundfile`). To prepare the array into `input_features`, the
                [`AutoFeatureExtractor`] should be used for extracting the mel features, padding and conversion into a
                tensor of type `torch.FloatTensor`. See [`~WhisperFeatureExtractor.__call__`] for details.
            encoder_outputs (`tuple(tuple(torch.FloatTensor)`, *optional*):
                Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
                `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of
                hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
            generation_config (`~generation.GenerationConfig`, *optional*):
                The generation configuration to be used as base parametrization for the generation call. `**kwargs`
                passed to generate matching the attributes of `generation_config` will override them. If
                `generation_config` is not provided, the default will be used, which had the following loading
                priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
                configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
                default values, whose documentation should be checked to parameterize generation.
            num_segment_frames (`int`, *optional*, defaults to 3000):
                The number of log-mel frames the model expects

        Return:
            A `torch.LongTensor` representing the detected language ids.
        """
        if input_features is None and encoder_outputs is None:
            raise ValueError("You have to specify either `input_features` or `encoder_outputs`")
        elif input_features is not None and encoder_outputs is not None:
            raise ValueError("Make sure to specificy only one of `input_features` or `encoder_outputs` - not both!")
        elif input_features is not None:
            inputs = {"input_features": input_features[:, :, :num_segment_frames]}
            batch_size = input_features.shape[0]
        elif encoder_outputs is not None:
            inputs = {"encoder_outputs": encoder_outputs}
            batch_size = (
                encoder_outputs[0].shape[0] if isinstance(encoder_outputs, BaseModelOutput) else encoder_outputs[0]
            )

        generation_config = generation_config or self.generation_config
        decoder_input_ids = (
            torch.ones((batch_size, 1), device=self.device, dtype=torch.long)
            * generation_config.decoder_start_token_id
        )

        with torch.no_grad():
            logits = self(**inputs, decoder_input_ids=decoder_input_ids, use_cache=False).logits[:, -1]

        non_lang_mask = torch.ones_like(logits[0], dtype=torch.bool)
        non_lang_mask[list(generation_config.lang_to_id.values())] = False

        logits[:, non_lang_mask] = -np.inf

        lang_ids = logits.argmax(-1)

        return lang_ids

    @staticmethod
    def _check_decoder_input_ids(kwargs):
        decoder_input_ids = kwargs.get("decoder_input_ids", None)
        assistant_model = kwargs.get("assistant_model", None)
        if decoder_input_ids is not None and assistant_model is not None:
            raise ValueError(
                "Passing `decoder_input_ids` is deprecated. Consider passing `prompt_ids` instead.",
            )

    @staticmethod
    def _set_num_frames(return_token_timestamps, generation_config, kwargs):
        if return_token_timestamps:
            if getattr(generation_config, "task", None) == "translate":
                logger.warning("Token-level timestamps may not be reliable for task 'translate'.")
            if not hasattr(generation_config, "alignment_heads"):
                raise ValueError(
                    "Model generation config has no `alignment_heads`, token-level timestamps not available. "
                    "See https://gist.github.com/hollance/42e32852f24243b748ae6bc1f985b13a on how to add this property to the generation config."
                )
            generation_config.num_frames = kwargs.pop("num_frames", None)

    @staticmethod
    def _set_thresholds_and_condition(
        generation_config,
        logprob_threshold,
        compression_ratio_threshold,
        no_speech_threshold,
        condition_on_prev_tokens,
    ):
        generation_config.logprob_threshold = (
            logprob_threshold
            if logprob_threshold is not None
            else getattr(generation_config, "logprob_threshold", None)
        )
        generation_config.compression_ratio_threshold = (
            compression_ratio_threshold
            if compression_ratio_threshold is not None
            else getattr(generation_config, "compression_ratio_threshold", None)
        )
        generation_config.no_speech_threshold = (
            no_speech_threshold
            if no_speech_threshold is not None
            else getattr(generation_config, "no_speech_threshold", None)
        )
        generation_config.condition_on_prev_tokens = (
            condition_on_prev_tokens
            if condition_on_prev_tokens is not None
            else getattr(generation_config, "condition_on_prev_tokens", None)
        )

    @staticmethod
    def _set_prompt_condition_type(generation_config, prompt_condition_type):
        allowed_cond_types = ["first-segment", "all-segments"]

        # default to "first-segment"
        prompt_condition_type = prompt_condition_type or allowed_cond_types[0]

        if prompt_condition_type not in allowed_cond_types:
            raise ValueError(
                f"`prompt_condition_type={prompt_condition_type} does not exist. Make sure to set `prompt_condition_type` to one of {', '.join(allowed_cond_types)}"
            )

        if generation_config.condition_on_prev_tokens is not True and prompt_condition_type == "all-segments":
            raise ValueError(
                "Make sure to set `condition_on_prev_tokens=True` when setting `prompt_condition_type='all-segments'`."
            )

        generation_config.prompt_condition_type = prompt_condition_type

    @staticmethod
    def _set_condition_on_prev_tokens(condition_on_prev_tokens, generation_config):
        condition_on_prev_tokens = (
            condition_on_prev_tokens
            if condition_on_prev_tokens is not None
            else getattr(generation_config, "condition_on_prev_tokens", False)
        )
        generation_config.condition_on_prev_tokens = condition_on_prev_tokens

    @staticmethod
    def _retrieve_max_frames_and_seek(batch_size, attention_mask, total_input_frames, is_shortform):
        if batch_size > 1 and not is_shortform and attention_mask is None:
            raise ValueError(
                "When doing batched long-form audio transcription, make sure to pass an `attention_mask`. You can retrieve the `attention_mask` by doing `processor(audio, ..., return_attention_mask=True)` "
            )
        elif batch_size > 1 and not is_shortform:
            max_frames = attention_mask.sum(-1).cpu().to(torch.long)
            seek = torch.zeros((batch_size,), dtype=torch.long)
        else:
            max_frames = torch.ones((batch_size,), dtype=torch.long) * total_input_frames
            seek = torch.zeros((batch_size,), dtype=torch.long)

        return max_frames, seek

    def _retrieve_logit_processors(self, generation_config, logits_processor, begin_index, num_beams, device):
        if generation_config.return_timestamps is True:
            timestamp_processor = WhisperTimeStampLogitsProcessor(generation_config, begin_index=begin_index)
            logits_processor = (
                [timestamp_processor] if logits_processor is None else [timestamp_processor] + logits_processor
            )

        if generation_config.suppress_tokens is not None:
            suppress_tokens_processor = SuppressTokensLogitsProcessor(generation_config.suppress_tokens, device=device)
            logits_processor = (
                [suppress_tokens_processor]
                if logits_processor is None
                else [suppress_tokens_processor] + logits_processor
            )
            generation_config.suppress_tokens = None

        if generation_config.begin_suppress_tokens is not None:
            begin_suppress_processor = SuppressTokensAtBeginLogitsProcessor(
                generation_config.begin_suppress_tokens, begin_index=begin_index, device=device
            )
            logits_processor = (
                [begin_suppress_processor]
                if logits_processor is None
                else [begin_suppress_processor] + logits_processor
            )
            generation_config.begin_suppress_tokens = None

        if generation_config.no_speech_threshold is not None:
            no_speech_detector = WhisperNoSpeechDetection(
                no_speech_token=generation_config.no_timestamps_token_id - 1,
                begin_index=begin_index,
                scores_is_logprobs=num_beams > 1,
            )
            logits_processor = (
                [no_speech_detector] if logits_processor is None else [no_speech_detector] + logits_processor
            )
            no_speech_detector.set_model(self)

        return logits_processor

    @staticmethod
    def _maybe_reduce_batch(input_features, seek, max_frames, cur_bsz, batch_idx_map):
        prev_bsz = cur_bsz
        new_batch_idx_map = []
        for i in range(prev_bsz):
            prev_i = batch_idx_map[i]
            if seek[prev_i] >= max_frames[prev_i]:
                cut_index = i + (cur_bsz - prev_bsz)
                cur_bsz -= 1
                input_features = torch.cat([input_features[:cut_index], input_features[cut_index + 1 :]], dim=0)
            else:
                # cut out index that goes away
                new_batch_idx_map.append(prev_i)

        return input_features, cur_bsz, new_batch_idx_map

    @staticmethod
    def _get_input_segment(input_features, seek, seek_num_frames, num_segment_frames, cur_bsz, batch_idx_map):
        if input_features is None:
            return None

        segment_input = []
        for i in range(cur_bsz):
            prev_i = batch_idx_map[i]
            segment_input_slice = input_features[i : i + 1, :, seek[prev_i] : seek[prev_i] + seek_num_frames[prev_i]]

            if segment_input_slice.shape[-1] < num_segment_frames:
                # pad to 3000 if necessary
                segment_input_slice = F.pad(
                    segment_input_slice, pad=(0, num_segment_frames - segment_input_slice.shape[-1])
                )

            segment_input.append(segment_input_slice)

        segment_input = torch.cat(segment_input, dim=0)

        return segment_input

    @staticmethod
    def _prepare_decoder_input_ids(
        cur_bsz,
        init_tokens,
        current_segments,
        batch_idx_map,
        do_condition_on_prev_tokens,
        prompt_ids,
        generation_config,
        config,
        device,
        suppress_tokens,
        timestamp_begin,
        kwargs,
    ):
        if "decoder_input_ids" in kwargs:
            decoder_input_ids = kwargs.pop("decoder_input_ids")

            return decoder_input_ids, kwargs

        cut_off_length = config.max_target_positions // 2 - 1

        decoder_input_ids = init_tokens[batch_idx_map]

        prev_start_of_text = getattr(generation_config, "prev_sot_token_id", None)
        if prev_start_of_text is None:
            prev_start_of_text = suppress_tokens[-2] if suppress_tokens is not None else None

        if any(do_condition_on_prev_tokens) and len(current_segments[0]) > 0:
            # according to https://github.com/openai/whisper/blob/e58f28804528831904c3b6f2c0e473f346223433/whisper/decoding.py#L609
            active_segments = [current_segments[i] if do_condition_on_prev_tokens[i] else None for i in batch_idx_map]

            for segments in active_segments:
                for seg in segments:
                    if len(seg["tokens"]) > 2 and seg["tokens"][-2] >= timestamp_begin:
                        # the segment finishes with two timestamp tokens
                        # we need to ignore the last timestamp token
                        # see https://github.com/huggingface/transformers/pull/34537
                        seg["tokens"] = seg["tokens"][:-1]

            if prompt_ids is not None and generation_config.prompt_condition_type == "all-segments":
                prev_ids = prompt_ids
            else:
                one_tensor = torch.ones((cur_bsz, 1), device=device, dtype=torch.long)
                prev_ids = prev_start_of_text * one_tensor[0] if prev_start_of_text is not None else None

            padding = "max_length" if generation_config.cache_implementation == "static" else "longest"

            prev_tokens = _pad_to_max_length(
                active_segments,
                generation_config.pad_token_id,
                device=device,
                padding_side="left",
                padding=padding,
                bos_token_tensor=prev_ids,
                cut_off_length=cut_off_length,
            )
            decoder_input_ids = torch.cat([prev_tokens, decoder_input_ids], dim=-1)

            kwargs["decoder_attention_mask"] = decoder_input_ids != generation_config.pad_token_id
        elif prompt_ids is not None:
            prev_tokens = prompt_ids[None].repeat(decoder_input_ids.shape[0], 1)
            decoder_input_ids = torch.cat([prev_tokens, decoder_input_ids], dim=-1)
            # make sure `"decoder_attention_mask"` is not passed to forward
            kwargs.pop("decoder_attention_mask", None)
        else:
            # make sure `"decoder_attention_mask"` is not passed to forward
            kwargs.pop("decoder_attention_mask", None)

        return decoder_input_ids, kwargs

    def _set_max_new_tokens_and_length(self, config, decoder_input_ids, generation_config):
        max_new_tokens = generation_config.max_new_tokens if generation_config.max_new_tokens is not None else 0
        if max_new_tokens + decoder_input_ids.shape[-1] > self.config.max_target_positions:
            raise ValueError(
                f"The length of `decoder_input_ids`, including special start tokens, prompt tokens, and previous tokens, is {decoder_input_ids.shape[-1]}, "
                f" and `max_new_tokens` is {max_new_tokens}. Thus, the combined length of "
                f"`decoder_input_ids` and `max_new_tokens` is: {max_new_tokens + decoder_input_ids.shape[-1]}. This exceeds the "
                f"`max_target_positions` of the Whisper model: {self.config.max_target_positions}. "
                "You should either reduce the length of your prompt, or reduce the value of `max_new_tokens`, "
                f"so that their combined length is less than {self.config.max_target_positions}."
            )

        num_initial_tokens = min(config.max_target_positions // 2 - 1, decoder_input_ids.shape[-1] - 1)

        # Make sure we don't get larger than `max_length`
        if generation_config.max_length is not None and generation_config.max_new_tokens is None:
            max_length = min(generation_config.max_length + num_initial_tokens, config.max_target_positions)
            logger.info(
                f"Increase max_length from {generation_config.max_length} to {max_length} since input is conditioned on previous segment."
            )
        elif (
            generation_config.max_new_tokens is not None
            and generation_config.max_new_tokens + decoder_input_ids.shape[-1] > config.max_target_positions
        ):
            max_new_tokens = config.max_target_positions - decoder_input_ids.shape[-1]
            generation_config.max_new_tokens = max_new_tokens

    @staticmethod
    def _retrieve_compression_ratio(tokens, vocab_size):
        """Compute byte length of zlib compressed token bytes vs. byte length of raw token bytes"""
        length = int(math.log2(vocab_size) / 8) + 1
        token_bytes = b"".join([t.to_bytes(length, "little") for t in tokens.tolist()])
        compression_ratio = len(token_bytes) / len(zlib.compress(token_bytes))

        return compression_ratio

    @staticmethod
    def _retrieve_avg_logprobs(scores, tokens, temperature):
        rescale_temperature = temperature if temperature > 0.0 else 1
        scores = torch.stack(scores).to(tokens.device)

        if scores.shape[0] > tokens.shape[0]:
            scores = scores[: tokens.shape[0]]
        else:
            tokens = tokens[-scores.shape[0] :]

        logprobs = F.log_softmax((scores * rescale_temperature).float(), dim=-1).to(scores.dtype)

        # retrieve logprob of selected tokens and sum
        # don't remove the eos token logprob! it counts in avg_logprob calculation in the original implementation
        sum_logprobs = sum(logprobs[i][tokens[i]] for i in range(logprobs.shape[0]))

        avg_logprobs = sum_logprobs / len(tokens)
        return avg_logprobs

    @staticmethod
    def _retrieve_segment(
        seek_sequence,
        seek_outputs,
        time_offset,
        timestamp_begin,
        seek_num_frames,
        time_precision,
        time_precision_features,
        input_stride,
        prev_idx,
        idx,
        return_token_timestamps,
        decoder_input_ids,
    ):
        # find the predicted "end of segment" predictions of Whisper
        # "end of segment" predictions occur whenever Whisper predicts a timestamp token
        timestamp_tokens: torch.Tensor = seek_sequence.ge(timestamp_begin)
        single_timestamp_ending = timestamp_tokens[-2:].tolist() == [False, True]
        timestamp_segment_indices = torch.where(timestamp_tokens[:-1] & timestamp_tokens[1:])[0]
        timestamp_segment_indices.add_(1)
        token_timestamps = seek_outputs[idx]["token_timestamps"] if return_token_timestamps else []
        idx_offset = decoder_input_ids.shape[-1]
        device = seek_sequence.device

        # If whisper predicted a "end of segment" via a timestep token, let's go ever each
        # "end of segment" prediction and slice the decoding into segments accordingly
        if len(timestamp_segment_indices) > 0:
            # if the output contains two consecutive timestamp tokens
            slices = timestamp_segment_indices.tolist()
            segments = []
            if single_timestamp_ending:
                slices.append(len(seek_sequence))
            else:
                # we want to include the last timestamp token in the last segment to know it was no single ending
                slices[-1] += 1

            last_slice = 0
            # Add each segment to list of all segments
            for i, current_slice in enumerate(slices):
                is_last_slice = i == len(slices) - 1
                sliced_tokens = seek_sequence[last_slice:current_slice]
                start_timestamp_pos = sliced_tokens[0] - timestamp_begin
                idx_sliced_tokens = -1 if not is_last_slice or single_timestamp_ending else -2
                end_timestamp_pos = sliced_tokens[idx_sliced_tokens] - timestamp_begin
                segments.append(
                    {
                        "start": time_offset[prev_idx]
                        + start_timestamp_pos.to(torch.float32 if device.type == "mps" else torch.float64)
                        * time_precision,
                        "end": time_offset[prev_idx]
                        + end_timestamp_pos.to(torch.float32 if device.type == "mps" else torch.float64)
                        * time_precision,
                        "tokens": sliced_tokens,
                        "idxs": (idx_offset + last_slice, idx_offset + current_slice),
                        "result": seek_outputs[idx],
                    }
                )
                if return_token_timestamps:
                    segments[-1]["token_timestamps"] = (
                        token_timestamps[idx_offset + last_slice : idx_offset + current_slice] + time_offset[prev_idx]
                    )
                last_slice = current_slice

            if single_timestamp_ending:
                # single timestamp at the end means no speech after the last timestamp.
                segment_offset = seek_num_frames[prev_idx]
            else:
                # otherwise, ignore the unfinished segment and seek to the last timestamp
                # here we throw away all predictions after the last predicted "end of segment"
                # since we are cutting right in the middle of an audio
                last_timestamp_pos = seek_sequence[last_slice - 2].item() - timestamp_begin
                segment_offset = last_timestamp_pos * input_stride
        else:
            # If whisper does not predict any "end of segment" token, then
            # the whole decoding is considered a segment and we add it to the list of segments
            timestamps = seek_sequence[timestamp_tokens.nonzero().flatten()]
            last_timestamp_pos = int(seek_num_frames[prev_idx] * time_precision_features / time_precision)
            if timestamps.numel() > 0 and timestamps[-1] != timestamp_begin:
                # no consecutive timestamps but it has a timestamp; use the last one.
                last_timestamp_pos = (timestamps[-1] - timestamp_begin).to(
                    torch.float32 if device.type == "mps" else torch.float64
                )
            segments = [
                {
                    "start": time_offset[prev_idx],
                    "end": time_offset[prev_idx] + last_timestamp_pos * time_precision,
                    "tokens": seek_sequence,
                    "idxs": (idx_offset, idx_offset + len(seek_sequence)),
                    "result": seek_outputs[idx],
                }
            ]
            if return_token_timestamps:
                segments[-1]["token_timestamps"] = (
                    token_timestamps[idx_offset : idx_offset + len(seek_sequence)] + time_offset[prev_idx]
                )
            segment_offset = seek_num_frames[prev_idx]

        return segments, segment_offset
