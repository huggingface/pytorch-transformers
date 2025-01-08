# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
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
Feature extractor class for SeamlessM4T
"""

from typing import List, Optional, Union

import numpy as np

from ...utils import is_torch_available


if is_torch_available():
    import torch

from ...audio_utils import mel_filter_bank, spectrogram, window_function
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import PaddingStrategy, TensorType, logging


logger = logging.get_logger(__name__)


class SeamlessM4TFeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs a SeamlessM4T feature extractor.

    This feature extractor inherits from [`SequenceFeatureExtractor`] which contains most of the main methods. Users
    should refer to this superclass for more information regarding those methods.

    This class extracts mel-filter bank features from raw speech.

    Args:
        feature_size (`int`, *optional*, defaults to 80):
            The feature dimension of the extracted features.
        sampling_rate (`int`, *optional*, defaults to 16000):
            The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
        num_mel_bins (`int`, *optional*, defaults to 80):
            Number of Mel-frequency bins.
        padding_value (`float`, *optional*, defaults to 0.0):
            The value that is used to fill the padding vectors.
        stride (`int`, *optional*, defaults to 2):
            Stride used to reshape audios from shape (batch_size,num_frames,num_mel_bins) to
            (batch_size,num_frames//stride,num_mel_bins*stride).
    """

    model_input_names = ["input_features", "attention_mask"]

    def __init__(
        self,
        feature_size=80,
        sampling_rate=16000,
        num_mel_bins=80,
        padding_value=0.0,
        stride=2,
        **kwargs,
    ):
        self.num_mel_bins = num_mel_bins
        self.return_attention_mask = True
        self.stride = stride

        mel_filters = mel_filter_bank(
            num_frequency_bins=256,
            num_mel_filters=self.num_mel_bins,
            min_frequency=20,
            max_frequency=sampling_rate // 2,
            sampling_rate=sampling_rate,
            norm=None,
            mel_scale="kaldi",
            triangularize_in_mel_space=True,
        )

        self.mel_filters = np.pad(mel_filters, ((0, 1), (0, 0)))
        self.window = window_function(400, "povey", periodic=False)

        super().__init__(feature_size=feature_size, sampling_rate=sampling_rate, padding_value=padding_value, **kwargs)

    @staticmethod
    # Copied from transformers.models.wav2vec2.feature_extraction_wav2vec2.Wav2Vec2FeatureExtractor.zero_mean_unit_var_norm
    def zero_mean_unit_var_norm(
        input_values: List[np.ndarray], attention_mask: List[np.ndarray], padding_value: float = 0.0
    ) -> List[np.ndarray]:
        """
        Every array in the list is normalized to have zero mean and unit variance
        """
        if attention_mask is not None:
            attention_mask = np.array(attention_mask, np.int32)
            normed_input_values = []

            for vector, length in zip(input_values, attention_mask.sum(-1)):
                normed_slice = (vector - vector[:length].mean()) / np.sqrt(vector[:length].var() + 1e-7)
                if length < normed_slice.shape[0]:
                    normed_slice[length:] = padding_value

                normed_input_values.append(normed_slice)
        else:
            normed_input_values = [(x - x.mean()) / np.sqrt(x.var() + 1e-7) for x in input_values]

        return normed_input_values

    def _extract_fbank_features(
        self,
        waveform: np.ndarray,
    ) -> np.ndarray:
        """
        Get mel-filter bank features using TorchAudio. Note that TorchAudio requires 16-bit signed integers as inputs
        and hence the waveform should not be normalized before feature extraction.
        """
        # by default, it extracts the left channel if stereo
        if len(waveform.shape) == 2:
            waveform = waveform[0]

        waveform = np.squeeze(waveform) * (2**15)  # Kaldi compliance: 16-bit signed integers
        features = spectrogram(
            waveform,
            self.window,
            frame_length=400,
            hop_length=160,
            fft_length=512,
            power=2.0,
            center=False,
            preemphasis=0.97,
            mel_filters=self.mel_filters,
            log_mel="log",
            mel_floor=1.192092955078125e-07,
            remove_dc_offset=True,
        ).T
        return features

    def __call__(
        self,
        raw_speech: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]],
        padding: Union[bool, str, PaddingStrategy] = True,
        pad_to_multiple_of: Optional[int] = 2,
        max_length: Optional[int] = None,
        truncation: bool = False,
        return_tensors: Optional[Union[str, TensorType]] = None,
        sampling_rate: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        do_normalize_per_mel_bins: Optional[bool] = True,
        **kwargs,
    ) -> BatchFeature:
        """
        Main method to featurize and prepare for the model one or several sequence(s).

        Args:
            raw_speech (`np.ndarray`, `torch.Tensor`, `List[float]`, `List[np.ndarray]`, `List[torch.Tensor]`,
            `List[List[float]]`, `List[List[List[float]]]`):
                The sequence or batch of sequences to be padded. Each sequence can be a numpy array,
                a torch tensor, a list of float values, a list of numpy arrays, a list of torch tensors,
                a list of list of float values or a list of a list of list of float values.
                If `raw_speech` is a one-dimensional `np.ndarray`, `torch.Tensor` or a `List[float]`, `raw_speech` is
                considered a single-channel, single-sample sound. In all other cases, the first dimension of
                `raw_speech`, whether from an `np.ndarray`, a `torch.Tensor` or a `List[...]`,
                corresponds to the number of samples in the batch, and the number of channels
                (i.e. mono or stereo character) is derived from the other dimensions
                (1D -> single-channel waveform batches; 2D-> stereo-channel waveform batches).
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding
                index) among:

                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            pad_to_multiple_of (`int`, *optional*, defaults to 2):
                If set will pad the sequence to a multiple of the provided value.

                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                `>= 7.5` (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128.
            max_length (`int`, *optional*):
                Maximum length of the returned list and optionally padding length (see above).
            truncation (`bool`):
                Activates truncation to cut input sequences longer than *max_length* to *max_length*.
            return_attention_mask (`bool`, *optional*):
                Whether to return the attention mask. If left to the default, will return the attention mask according
                to the specific feature_extractor's default.

                [What are attention masks?](../glossary#attention-mask)

                <Tip>

                For SeamlessM4T models, `attention_mask` should always be passed for batched inference, to avoid subtle
                bugs.

                </Tip>

            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
            sampling_rate (`int`, *optional*):
                The sampling rate at which the `raw_speech` input was sampled. It is strongly recommended to pass
                `sampling_rate` at the forward call to prevent silent errors.
            do_normalize_per_mel_bins (`bool`, *optional*, defaults to `True`):
                Whether or not to zero-mean unit-variance normalize the input per mel-channel.
            kwargs (*optional*):
                Remaining dictionary of keyword arguments that will be passed to the tokenizer or the feature
                extractor.
        """
        if sampling_rate is not None:
            if sampling_rate != self.sampling_rate:
                raise ValueError(
                    f"The model corresponding to this feature extractor: {self} was trained using a sampling rate of"
                    f" {self.sampling_rate}. Please make sure that the provided `raw_speech` input was sampled with"
                    f" {self.sampling_rate} and not {sampling_rate}."
                )
        else:
            logger.warning(
                "It is strongly recommended to pass the `sampling_rate` argument to this function. "
                "Failing to do so can result in silent errors that might be hard to debug."
            )

        return_attention_mask = (
            return_attention_mask if return_attention_mask is not None else self.return_attention_mask
        )

        is_batched_numpy = isinstance(raw_speech, np.ndarray) and len(raw_speech.shape) > 1
        if is_batched_numpy and len(raw_speech.shape) > 3:
            raise ValueError(f"Only mono-channel or stereo-channel audio is supported for input to {self}")

        acceptable_types = (
            (torch.Tensor, np.ndarray, tuple, list) if is_torch_available() else (np.ndarray, tuple, list)
        )
        is_batched = is_batched_numpy or (
            isinstance(raw_speech, (list, tuple)) and (isinstance(raw_speech[0], acceptable_types))
        )

        if is_batched:
            raw_speech = [np.asarray(speech, dtype=np.float32) for speech in raw_speech]
        elif not is_batched and not isinstance(raw_speech, np.ndarray):
            raw_speech = np.asarray(raw_speech, dtype=np.float32)
        elif isinstance(raw_speech, np.ndarray) and raw_speech.dtype is np.dtype(np.float64):
            raw_speech = raw_speech.astype(np.float32)

        # always return batch
        if not is_batched:
            raw_speech = [raw_speech]

        # extract fbank features
        features = [self._extract_fbank_features(waveform) for waveform in raw_speech]

        if do_normalize_per_mel_bins:
            # torch defaults to ddof=1, and numpy defaults to ddof=0
            features = [
                (x - np.expand_dims(x.mean(0), 0)) / np.sqrt(np.expand_dims(x.var(0, ddof=1), 0) + 1e-7)
                for x in features
            ]

        # convert into correct format for padding
        encoded_inputs = BatchFeature({"input_features": features})

        padded_inputs = self.pad(
            encoded_inputs,
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors="np",
        )

        # SeamlessM4T needs to process extracted features
        input_features = padded_inputs.get("input_features")
        attention_mask = padded_inputs.pop("attention_mask")

        batch_size, num_frames, num_channels = input_features.shape

        remainder = num_frames % self.stride
        if remainder != 0:
            input_features = input_features[:, : num_frames - remainder, :]
            attention_mask = attention_mask[:, : num_frames - remainder]

        input_features = np.reshape(
            input_features, (batch_size, num_frames // self.stride, num_channels * self.stride)
        )

        indices = np.arange(0, num_frames - remainder)
        attention_mask = attention_mask[:, indices % self.stride == 1]

        padded_inputs["input_features"] = input_features
        if return_attention_mask:
            padded_inputs["attention_mask"] = attention_mask

        if return_tensors is not None:
            padded_inputs = padded_inputs.convert_to_tensors(return_tensors)

        return padded_inputs


__all__ = ["SeamlessM4TFeatureExtractor"]
