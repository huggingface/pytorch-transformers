# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
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
"""
Speech processor class for Wav2Vec2
"""

import os
import warnings
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from multiprocessing import Pool, get_context, get_start_method
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Union

import numpy as np

from ...processing_utils import ProcessorMixin
from ...utils import ModelOutput, logging, requires_backends


logger = logging.get_logger(__name__)


if TYPE_CHECKING:
    from pyctcdecode import BeamSearchDecoderCTC

    from ...feature_extraction_utils import FeatureExtractionMixin
    from ...tokenization_utils import PreTrainedTokenizerBase


ListOfDict = List[Dict[str, Union[int, str]]]


@dataclass
class Wav2Vec2DecoderWithLMOutput(ModelOutput):
    """
    Output type of [`Wav2Vec2DecoderWithLM`], with transcription.

    Args:
        text (list of `str` or `str`):
            Decoded logits in text from. Usually the speech transcription.
        logit_score (list of `float` or `float`):
            Total logit score of the beams associated with produced text.
        lm_score (list of `float`):
            Fused lm_score of the beams associated with produced text.
        word_offsets (list of `List[Dict[str, Union[int, str]]]` or `List[Dict[str, Union[int, str]]]`):
            Offsets of the decoded words. In combination with sampling rate and model downsampling rate word offsets
            can be used to compute time stamps for each word.
    """

    text: Union[List[List[str]], List[str], str]
    logit_score: Union[List[List[float]], List[float], float] = None
    lm_score: Union[List[List[float]], List[float], float] = None
    word_offsets: Union[List[List[ListOfDict]], List[ListOfDict], ListOfDict] = None


class Wav2Vec2ProcessorWithLM(ProcessorMixin):
    r"""
    Constructs a Wav2Vec2 processor which wraps a Wav2Vec2 feature extractor, a Wav2Vec2 CTC tokenizer and a decoder
    with language model support into a single processor for language model boosted speech recognition decoding.

    Args:
        feature_extractor ([`Wav2Vec2FeatureExtractor`] or [`SeamlessM4TFeatureExtractor`]):
            An instance of [`Wav2Vec2FeatureExtractor`] or [`SeamlessM4TFeatureExtractor`]. The feature extractor is a required input.
        tokenizer ([`Wav2Vec2CTCTokenizer`]):
            An instance of [`Wav2Vec2CTCTokenizer`]. The tokenizer is a required input.
        decoder (`pyctcdecode.BeamSearchDecoderCTC`):
            An instance of [`pyctcdecode.BeamSearchDecoderCTC`]. The decoder is a required input.
    """

    feature_extractor_class = "AutoFeatureExtractor"
    tokenizer_class = "Wav2Vec2CTCTokenizer"

    def __init__(
        self,
        feature_extractor: "FeatureExtractionMixin",
        tokenizer: "PreTrainedTokenizerBase",
        decoder: "BeamSearchDecoderCTC",
    ):
        from pyctcdecode import BeamSearchDecoderCTC

        super().__init__(feature_extractor, tokenizer)
        if not isinstance(decoder, BeamSearchDecoderCTC):
            raise TypeError(f"`decoder` has to be of type {BeamSearchDecoderCTC.__class__}, but is {type(decoder)}")

        if feature_extractor.__class__.__name__ not in ["Wav2Vec2FeatureExtractor", "SeamlessM4TFeatureExtractor"]:
            raise ValueError(
                f"`feature_extractor` has to be of type `Wav2Vec2FeatureExtractor` or `SeamlessM4TFeatureExtractor`, but is {type(feature_extractor)}"
            )

        # make sure that decoder's alphabet and tokenizer's vocab match in content
        missing_decoder_tokens = self.get_missing_alphabet_tokens(decoder, tokenizer)
        if len(missing_decoder_tokens) > 0:
            raise ValueError(
                f"The tokens {missing_decoder_tokens} are defined in the tokenizer's "
                "vocabulary, but not in the decoder's alphabet. "
                f"Make sure to include {missing_decoder_tokens} in the decoder's alphabet."
            )

        self.decoder = decoder
        self.current_processor = self.feature_extractor
        self._in_target_context_manager = False

    def save_pretrained(self, save_directory):
        super().save_pretrained(save_directory)
        self.decoder.save_to_dir(save_directory)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        r"""
        Instantiate a [`Wav2Vec2ProcessorWithLM`] from a pretrained Wav2Vec2 processor.

        <Tip>

        This class method is simply calling the feature extractor's
        [`~feature_extraction_utils.FeatureExtractionMixin.from_pretrained`], Wav2Vec2CTCTokenizer's
        [`~tokenization_utils_base.PreTrainedTokenizerBase.from_pretrained`], and
        [`pyctcdecode.BeamSearchDecoderCTC.load_from_hf_hub`].

        Please refer to the docstrings of the methods above for more information.

        </Tip>

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                This can be either:

                - a string, the *model id* of a pretrained feature_extractor hosted inside a model repo on
                  huggingface.co.
                - a path to a *directory* containing a feature extractor file saved using the
                  [`~SequenceFeatureExtractor.save_pretrained`] method, e.g., `./my_model_directory/`.
                - a path or url to a saved feature extractor JSON *file*, e.g.,
                  `./my_model_directory/preprocessor_config.json`.
            **kwargs
                Additional keyword arguments passed along to both [`SequenceFeatureExtractor`] and
                [`PreTrainedTokenizer`]
        """
        requires_backends(cls, "pyctcdecode")
        from pyctcdecode import BeamSearchDecoderCTC

        feature_extractor, tokenizer = super()._get_arguments_from_pretrained(pretrained_model_name_or_path, **kwargs)

        if os.path.isdir(pretrained_model_name_or_path) or os.path.isfile(pretrained_model_name_or_path):
            unigram_encoding = kwargs.get("unigram_encoding", "utf-8")
            decoder = BeamSearchDecoderCTC.load_from_dir(pretrained_model_name_or_path, unigram_encoding)
        else:
            # BeamSearchDecoderCTC has no auto class
            kwargs.pop("_from_auto", None)
            # snapshot_download has no `trust_remote_code` flag
            kwargs.pop("trust_remote_code", None)

            # make sure that only relevant filenames are downloaded
            language_model_filenames = os.path.join(BeamSearchDecoderCTC._LANGUAGE_MODEL_SERIALIZED_DIRECTORY, "*")
            alphabet_filename = BeamSearchDecoderCTC._ALPHABET_SERIALIZED_FILENAME
            allow_patterns = [language_model_filenames, alphabet_filename]

            decoder = BeamSearchDecoderCTC.load_from_hf_hub(
                pretrained_model_name_or_path, allow_patterns=allow_patterns, **kwargs
            )

        # set language model attributes
        for attribute in ["alpha", "beta", "unk_score_offset", "score_boundary"]:
            value = kwargs.pop(attribute, None)

            if value is not None:
                cls._set_language_model_attribute(decoder, attribute, value)

        # make sure that decoder's alphabet and tokenizer's vocab match in content
        missing_decoder_tokens = cls.get_missing_alphabet_tokens(decoder, tokenizer)
        if len(missing_decoder_tokens) > 0:
            raise ValueError(
                f"The tokens {missing_decoder_tokens} are defined in the tokenizer's "
                "vocabulary, but not in the decoder's alphabet. "
                f"Make sure to include {missing_decoder_tokens} in the decoder's alphabet."
            )

        return cls(feature_extractor=feature_extractor, tokenizer=tokenizer, decoder=decoder)

    @staticmethod
    def _set_language_model_attribute(decoder: "BeamSearchDecoderCTC", attribute: str, value: float):
        setattr(decoder.model_container[decoder._model_key], attribute, value)

    @property
    def language_model(self):
        return self.decoder.model_container[self.decoder._model_key]

    @staticmethod
    def get_missing_alphabet_tokens(decoder, tokenizer):
        from pyctcdecode.alphabet import BLANK_TOKEN_PTN, UNK_TOKEN, UNK_TOKEN_PTN

        # we need to make sure that all of the tokenizer's except the special tokens
        # are present in the decoder's alphabet. Retrieve missing alphabet token
        # from decoder
        tokenizer_vocab_list = list(tokenizer.get_vocab().keys())

        # replace special tokens
        for i, token in enumerate(tokenizer_vocab_list):
            if BLANK_TOKEN_PTN.match(token):
                tokenizer_vocab_list[i] = ""
            if token == tokenizer.word_delimiter_token:
                tokenizer_vocab_list[i] = " "
            if UNK_TOKEN_PTN.match(token):
                tokenizer_vocab_list[i] = UNK_TOKEN

        # are any of the extra tokens no special tokenizer tokens?
        missing_tokens = set(tokenizer_vocab_list) - set(decoder._alphabet.labels)

        return missing_tokens

    def __call__(self, *args, **kwargs):
        """
        When used in normal mode, this method forwards all its arguments to the feature extractor's
        [`~FeatureExtractionMixin.__call__`] and returns its output. If used in the context
        [`~Wav2Vec2ProcessorWithLM.as_target_processor`] this method forwards all its arguments to
        Wav2Vec2CTCTokenizer's [`~Wav2Vec2CTCTokenizer.__call__`]. Please refer to the docstring of the above two
        methods for more information.
        """
        # For backward compatibility
        if self._in_target_context_manager:
            return self.current_processor(*args, **kwargs)

        if "raw_speech" in kwargs:
            warnings.warn("Using `raw_speech` as a keyword argument is deprecated. Use `audio` instead.")
            audio = kwargs.pop("raw_speech")
        else:
            audio = kwargs.pop("audio", None)
        sampling_rate = kwargs.pop("sampling_rate", None)
        text = kwargs.pop("text", None)
        if len(args) > 0:
            audio = args[0]
            args = args[1:]

        if audio is None and text is None:
            raise ValueError("You need to specify either an `audio` or `text` input to process.")

        if audio is not None:
            inputs = self.feature_extractor(audio, *args, sampling_rate=sampling_rate, **kwargs)
        if text is not None:
            encodings = self.tokenizer(text, **kwargs)

        if text is None:
            return inputs
        elif audio is None:
            return encodings
        else:
            inputs["labels"] = encodings["input_ids"]
            return inputs

    def pad(self, *args, **kwargs):
        """
        When used in normal mode, this method forwards all its arguments to the feature extractor's
        [`~FeatureExtractionMixin.pad`] and returns its output. If used in the context
        [`~Wav2Vec2ProcessorWithLM.as_target_processor`] this method forwards all its arguments to
        Wav2Vec2CTCTokenizer's [`~Wav2Vec2CTCTokenizer.pad`]. Please refer to the docstring of the above two methods
        for more information.
        """
        # For backward compatibility
        if self._in_target_context_manager:
            return self.current_processor.pad(*args, **kwargs)

        input_features = kwargs.pop("input_features", None)
        labels = kwargs.pop("labels", None)
        if len(args) > 0:
            input_features = args[0]
            args = args[1:]

        if input_features is not None:
            input_features = self.feature_extractor.pad(input_features, *args, **kwargs)
        if labels is not None:
            labels = self.tokenizer.pad(labels, **kwargs)

        if labels is None:
            return input_features
        elif input_features is None:
            return labels
        else:
            input_features["labels"] = labels["input_ids"]
            return input_features

    def batch_decode(
        self,
        logits: np.ndarray,
        pool: Optional[Pool] = None,
        num_processes: Optional[int] = None,
        beam_width: Optional[int] = None,
        beam_prune_logp: Optional[float] = None,
        token_min_logp: Optional[float] = None,
        hotwords: Optional[Iterable[str]] = None,
        hotword_weight: Optional[float] = None,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        unk_score_offset: Optional[float] = None,
        lm_score_boundary: Optional[bool] = None,
        output_word_offsets: bool = False,
        n_best: int = 1,
    ):
        """
        Batch decode output logits to audio transcription with language model support.

        <Tip>

        This function makes use of Python's multiprocessing. Currently, multiprocessing is available only on Unix
        systems (see this [issue](https://github.com/kensho-technologies/pyctcdecode/issues/65)).

        If you are decoding multiple batches, consider creating a `Pool` and passing it to `batch_decode`. Otherwise,
        `batch_decode` will be very slow since it will create a fresh `Pool` for each call. See usage example below.

        </Tip>

        Args:
            logits (`np.ndarray`):
                The logits output vector of the model representing the log probabilities for each token.
            pool (`multiprocessing.Pool`, *optional*):
                An optional user-managed pool. If not set, one will be automatically created and closed. The pool
                should be instantiated *after* `Wav2Vec2ProcessorWithLM`. Otherwise, the LM won't be available to the
                pool's sub-processes.

                <Tip>

                Currently, only pools created with a 'fork' context can be used. If a 'spawn' pool is passed, it will
                be ignored and sequential decoding will be used instead.

                </Tip>

            num_processes (`int`, *optional*):
                If `pool` is not set, number of processes on which the function should be parallelized over. Defaults
                to the number of available CPUs.
            beam_width (`int`, *optional*):
                Maximum number of beams at each step in decoding. Defaults to pyctcdecode's DEFAULT_BEAM_WIDTH.
            beam_prune_logp (`int`, *optional*):
                Beams that are much worse than best beam will be pruned Defaults to pyctcdecode's DEFAULT_PRUNE_LOGP.
            token_min_logp (`int`, *optional*):
                Tokens below this logp are skipped unless they are argmax of frame Defaults to pyctcdecode's
                DEFAULT_MIN_TOKEN_LOGP.
            hotwords (`List[str]`, *optional*):
                List of words with extra importance, can be OOV for LM
            hotword_weight (`int`, *optional*):
                Weight factor for hotword importance Defaults to pyctcdecode's DEFAULT_HOTWORD_WEIGHT.
            alpha (`float`, *optional*):
                Weight for language model during shallow fusion
            beta (`float`, *optional*):
                Weight for length score adjustment of during scoring
            unk_score_offset (`float`, *optional*):
                Amount of log score offset for unknown tokens
            lm_score_boundary (`bool`, *optional*):
                Whether to have kenlm respect boundaries when scoring
            output_word_offsets (`bool`, *optional*, defaults to `False`):
                Whether or not to output word offsets. Word offsets can be used in combination with the sampling rate
                and model downsampling rate to compute the time-stamps of transcribed words.
            n_best (`int`, *optional*, defaults to `1`):
                Number of best hypotheses to return. If `n_best` is greater than 1, the returned `text` will be a list
                of lists of strings, `logit_score` will be a list of lists of floats, and `lm_score` will be a list of
                lists of floats, where the length of the outer list will correspond to the batch size and the length of
                the inner list will correspond to the number of returned hypotheses . The value should be >= 1.

                <Tip>

                Please take a look at the Example of [`~Wav2Vec2ProcessorWithLM.decode`] to better understand how to
                make use of `output_word_offsets`. [`~Wav2Vec2ProcessorWithLM.batch_decode`] works the same way with
                batched output.

                </Tip>

        Returns:
            [`~models.wav2vec2.Wav2Vec2DecoderWithLMOutput`].

        Example:
            See [Decoding multiple audios](#decoding-multiple-audios).
        """

        from pyctcdecode.constants import (
            DEFAULT_BEAM_WIDTH,
            DEFAULT_HOTWORD_WEIGHT,
            DEFAULT_MIN_TOKEN_LOGP,
            DEFAULT_PRUNE_LOGP,
        )

        # set defaults
        beam_width = beam_width if beam_width is not None else DEFAULT_BEAM_WIDTH
        beam_prune_logp = beam_prune_logp if beam_prune_logp is not None else DEFAULT_PRUNE_LOGP
        token_min_logp = token_min_logp if token_min_logp is not None else DEFAULT_MIN_TOKEN_LOGP
        hotword_weight = hotword_weight if hotword_weight is not None else DEFAULT_HOTWORD_WEIGHT

        # reset params at every forward call. It's just a `set` method in pyctcdecode
        self.decoder.reset_params(
            alpha=alpha, beta=beta, unk_score_offset=unk_score_offset, lm_score_boundary=lm_score_boundary
        )

        # create multiprocessing pool and list numpy arrays
        # filter out logits padding
        logits_list = [array[(array != -100.0).all(axis=-1)] for array in logits]

        # create a pool if necessary while also using it as a context manager to close itself
        if pool is None:
            # fork is safe to use only on Unix, see "Contexts and start methods" section on
            # multiprocessing's docs (https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods)
            default_context = get_start_method()

            if default_context == "fork":
                cm = pool = get_context().Pool(num_processes)
            else:
                logger.warning(
                    "Parallel batch decoding is not currently supported in this platform. "
                    "Falling back to sequential decoding."
                )
                cm = nullcontext()
        else:
            # pool is managed by the user, so we don't need to close it
            cm = nullcontext()

            if num_processes is not None:
                logger.warning(
                    "Parameter `num_process` was passed, but it will be ignored since `pool` was also specified."
                )

        # pyctcdecode
        with cm:
            decoded_beams = self.decoder.decode_beams_batch(
                pool=pool,
                logits_list=logits_list,
                beam_width=beam_width,
                beam_prune_logp=beam_prune_logp,
                token_min_logp=token_min_logp,
                hotwords=hotwords,
                hotword_weight=hotword_weight,
            )

        # extract text and scores
        batch_texts, logit_scores, lm_scores, word_offsets = [], [], [], []

        for d in decoded_beams:
            batch_texts.append([beam[0] for beam in d])
            logit_scores.append([beam[-2] for beam in d])
            lm_scores.append([beam[-1] for beam in d])

            # word_offsets.append([{"word": t[0], "start_offset": t[1][0], "end_offset": t[1][1]} for t in d[0][1]])

            word_offsets.append(
                [
                    [
                        {"word": word, "start_offset": start_offset, "end_offset": end_offset}
                        for word, (start_offset, end_offset) in beam[1]
                    ]
                    for beam in d
                ]
            )

        word_offsets = word_offsets if output_word_offsets else None

        if n_best == 1:
            return Wav2Vec2DecoderWithLMOutput(
                text=[hyps[0] for hyps in batch_texts],
                logit_score=[hyps[0] for hyps in logit_scores],
                lm_score=[hyps[0] for hyps in lm_scores],
                word_offsets=[hyps[0] for hyps in word_offsets] if word_offsets is not None else None,
            )
        else:
            return Wav2Vec2DecoderWithLMOutput(
                text=[hyps[:n_best] for hyps in batch_texts],
                logit_score=[hyps[:n_best] for hyps in logit_scores],
                lm_score=[hyps[:n_best] for hyps in lm_scores],
                word_offsets=[hyps[:n_best] for hyps in word_offsets] if word_offsets is not None else None,
            )

    def decode(
        self,
        logits: np.ndarray,
        beam_width: Optional[int] = None,
        beam_prune_logp: Optional[float] = None,
        token_min_logp: Optional[float] = None,
        hotwords: Optional[Iterable[str]] = None,
        hotword_weight: Optional[float] = None,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        unk_score_offset: Optional[float] = None,
        lm_score_boundary: Optional[bool] = None,
        output_word_offsets: bool = False,
        n_best: int = 1,
    ):
        """
        Decode output logits to audio transcription with language model support.

        Args:
            logits (`np.ndarray`):
                The logits output vector of the model representing the log probabilities for each token.
            beam_width (`int`, *optional*):
                Maximum number of beams at each step in decoding. Defaults to pyctcdecode's DEFAULT_BEAM_WIDTH.
            beam_prune_logp (`int`, *optional*):
                A threshold to prune beams with log-probs less than best_beam_logp + beam_prune_logp. The value should
                be <= 0. Defaults to pyctcdecode's DEFAULT_PRUNE_LOGP.
            token_min_logp (`int`, *optional*):
                Tokens with log-probs below token_min_logp are skipped unless they are have the maximum log-prob for an
                utterance. Defaults to pyctcdecode's DEFAULT_MIN_TOKEN_LOGP.
            hotwords (`List[str]`, *optional*):
                List of words with extra importance which can be missing from the LM's vocabulary, e.g. ["huggingface"]
            hotword_weight (`int`, *optional*):
                Weight multiplier that boosts hotword scores. Defaults to pyctcdecode's DEFAULT_HOTWORD_WEIGHT.
            alpha (`float`, *optional*):
                Weight for language model during shallow fusion
            beta (`float`, *optional*):
                Weight for length score adjustment of during scoring
            unk_score_offset (`float`, *optional*):
                Amount of log score offset for unknown tokens
            lm_score_boundary (`bool`, *optional*):
                Whether to have kenlm respect boundaries when scoring
            output_word_offsets (`bool`, *optional*, defaults to `False`):
                Whether or not to output word offsets. Word offsets can be used in combination with the sampling rate
                and model downsampling rate to compute the time-stamps of transcribed words.
            n_best (`int`, *optional*, defaults to `1`):
                Number of best hypotheses to return. If `n_best` is greater than 1, the returned `text` will be a list
                of strings, `logit_score` will be a list of floats, and `lm_score` will be a list of floats, where the
                length of these lists will correspond to the number of returned hypotheses. The value should be >= 1.

                <Tip>

                Please take a look at the example below to better understand how to make use of `output_word_offsets`.

                </Tip>

        Returns:
            [`~models.wav2vec2.Wav2Vec2DecoderWithLMOutput`].

        Example:

        ```python
        >>> # Let's see how to retrieve time steps for a model
        >>> from transformers import AutoTokenizer, AutoProcessor, AutoModelForCTC
        >>> from datasets import load_dataset
        >>> import datasets
        >>> import torch

        >>> # import model, feature extractor, tokenizer
        >>> model = AutoModelForCTC.from_pretrained("patrickvonplaten/wav2vec2-base-100h-with-lm")
        >>> processor = AutoProcessor.from_pretrained("patrickvonplaten/wav2vec2-base-100h-with-lm")

        >>> # load first sample of English common_voice
        >>> dataset = load_dataset("mozilla-foundation/common_voice_11_0", "en", split="train", streaming=True, trust_remote_code=True)
        >>> dataset = dataset.cast_column("audio", datasets.Audio(sampling_rate=16_000))
        >>> dataset_iter = iter(dataset)
        >>> sample = next(dataset_iter)

        >>> # forward sample through model to get greedily predicted transcription ids
        >>> input_values = processor(sample["audio"]["array"], return_tensors="pt").input_values
        >>> with torch.no_grad():
        ...     logits = model(input_values).logits[0].cpu().numpy()

        >>> # retrieve word stamps (analogous commands for `output_char_offsets`)
        >>> outputs = processor.decode(logits, output_word_offsets=True)
        >>> # compute `time_offset` in seconds as product of downsampling ratio and sampling_rate
        >>> time_offset = model.config.inputs_to_logits_ratio / processor.feature_extractor.sampling_rate

        >>> word_offsets = [
        ...     {
        ...         "word": d["word"],
        ...         "start_time": round(d["start_offset"] * time_offset, 2),
        ...         "end_time": round(d["end_offset"] * time_offset, 2),
        ...     }
        ...     for d in outputs.word_offsets
        ... ]
        >>> # compare word offsets with audio `en_train_0/common_voice_en_19121553.mp3` online on the dataset viewer:
        >>> # https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0/viewer/en
        >>> word_offsets[:4]
        [{'word': 'THE', 'start_time': 0.68, 'end_time': 0.78}, {'word': 'TRACK', 'start_time': 0.88, 'end_time': 1.1}, {'word': 'APPEARS', 'start_time': 1.18, 'end_time': 1.66}, {'word': 'ON', 'start_time': 1.86, 'end_time': 1.92}]
        ```"""

        from pyctcdecode.constants import (
            DEFAULT_BEAM_WIDTH,
            DEFAULT_HOTWORD_WEIGHT,
            DEFAULT_MIN_TOKEN_LOGP,
            DEFAULT_PRUNE_LOGP,
        )

        # set defaults
        beam_width = beam_width if beam_width is not None else DEFAULT_BEAM_WIDTH
        beam_prune_logp = beam_prune_logp if beam_prune_logp is not None else DEFAULT_PRUNE_LOGP
        token_min_logp = token_min_logp if token_min_logp is not None else DEFAULT_MIN_TOKEN_LOGP
        hotword_weight = hotword_weight if hotword_weight is not None else DEFAULT_HOTWORD_WEIGHT

        # reset params at every forward call. It's just a `set` method in pyctcdecode
        self.decoder.reset_params(
            alpha=alpha, beta=beta, unk_score_offset=unk_score_offset, lm_score_boundary=lm_score_boundary
        )

        # pyctcdecode
        decoded_beams = self.decoder.decode_beams(
            logits,
            beam_width=beam_width,
            beam_prune_logp=beam_prune_logp,
            token_min_logp=token_min_logp,
            hotwords=hotwords,
            hotword_weight=hotword_weight,
        )

        word_offsets = None
        if output_word_offsets:
            word_offsets = [
                [
                    {"word": word, "start_offset": start_offset, "end_offset": end_offset}
                    for word, (start_offset, end_offset) in beam[2]
                ]
                for beam in decoded_beams
            ]
        logit_scores = [beam[-2] for beam in decoded_beams]

        lm_scores = [beam[-1] for beam in decoded_beams]

        hypotheses = [beam[0] for beam in decoded_beams]

        if n_best > len(decoded_beams):
            logger.info(
                "N-best size is larger than the number of generated hypotheses, all hypotheses will be returned."
            )

        if n_best == 1:
            return Wav2Vec2DecoderWithLMOutput(
                text=hypotheses[0],
                logit_score=logit_scores[0],
                lm_score=lm_scores[0],
                word_offsets=word_offsets[0] if word_offsets is not None else None,
            )
        else:
            return Wav2Vec2DecoderWithLMOutput(
                text=hypotheses[:n_best],
                logit_score=logit_scores[:n_best],
                lm_score=lm_scores[:n_best],
                word_offsets=word_offsets[:n_best] if word_offsets is not None else None,
            )

    @contextmanager
    def as_target_processor(self):
        """
        Temporarily sets the processor for processing the target. Useful for encoding the labels when fine-tuning
        Wav2Vec2.
        """
        warnings.warn(
            "`as_target_processor` is deprecated and will be removed in v5 of Transformers. You can process your "
            "labels by using the argument `text` of the regular `__call__` method (either in the same call as "
            "your audio inputs, or in a separate call."
        )
        self._in_target_context_manager = True
        self.current_processor = self.tokenizer
        yield
        self.current_processor = self.feature_extractor
        self._in_target_context_manager = False


__all__ = ["Wav2Vec2ProcessorWithLM"]
