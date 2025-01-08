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
Feature extractor class for CLVP
"""

from typing import List, Optional, Union

import numpy as np

from ...audio_utils import mel_filter_bank, spectrogram, window_function
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import TensorType, logging


logger = logging.get_logger(__name__)


class ClvpFeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs a CLVP feature extractor.

    This feature extractor inherits from [`~feature_extraction_sequence_utils.SequenceFeatureExtractor`] which contains
    most of the main methods. Users should refer to this superclass for more information regarding those methods.

    This class extracts log-mel-spectrogram features from raw speech using a custom numpy implementation of the `Short
    Time Fourier Transform` which should match pytorch's `torch.stft` equivalent.

    Args:
        feature_size (`int`, *optional*, defaults to 80):
            The feature dimension of the extracted features.
        sampling_rate (`int`, *optional*, defaults to 22050):
            The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
        default_audio_length (`int`, *optional*, defaults to 6):
            The default length of raw audio in seconds. If `max_length` is not set during `__call__` then it will
            automatically be set to default_audio_length * `self.sampling_rate`.
        hop_length (`int`, *optional*, defaults to 256):
            Length of the overlaping windows for the STFT used to obtain the Mel Frequency coefficients.
        chunk_length (`int`, *optional*, defaults to 30):
            The maximum number of chuncks of `sampling_rate` samples used to trim and pad longer or shorter audio
            sequences.
        n_fft (`int`, *optional*, defaults to 1024):
            Size of the Fourier transform.
        padding_value (`float`, *optional*, defaults to 0.0):
            Padding value used to pad the audio. Should correspond to silences.
        mel_norms (`list` of length `feature_size`, *optional*):
            If `mel_norms` is provided then it will be used to normalize the log-mel spectrograms along each
            mel-filter.
        return_attention_mask (`bool`, *optional*, defaults to `False`):
            Whether to return the attention mask. If left to the default, it will return the attention mask.

            [What are attention masks?](../glossary#attention-mask)
    """

    model_input_names = ["input_features", "attention_mask"]

    def __init__(
        self,
        feature_size=80,
        sampling_rate=22050,
        default_audio_length=6,
        hop_length=256,
        chunk_length=30,
        n_fft=1024,
        padding_value=0.0,
        mel_norms=None,
        return_attention_mask=False,  # pad inputs to max length with silence token (zero) and no attention mask
        **kwargs,
    ):
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            return_attention_mask=return_attention_mask,
            **kwargs,
        )
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.chunk_length = chunk_length
        self.n_samples = chunk_length * sampling_rate
        self.nb_max_frames = self.n_samples // hop_length
        self.sampling_rate = sampling_rate
        self.default_audio_length = default_audio_length
        self.mel_norms = mel_norms
        self.mel_filters = mel_filter_bank(
            num_frequency_bins=1 + (n_fft // 2),
            num_mel_filters=feature_size,
            min_frequency=0.0,
            max_frequency=8000.0,
            sampling_rate=sampling_rate,
            norm="slaney",
            mel_scale="htk",
        )

    def _np_extract_fbank_features(self, waveform: np.array) -> np.ndarray:
        """
        This method first computes the log-mel spectrogram of the provided audio then applies normalization along the
        each mel-filterbank, if `mel_norms` is provided.
        """
        log_spec = spectrogram(
            waveform,
            window_function(self.n_fft, "hann"),
            frame_length=self.n_fft,
            hop_length=self.hop_length,
            power=2.0,
            mel_filters=self.mel_filters,
            log_mel=None,
        )

        log_spec = np.log(np.clip(log_spec, a_min=1e-5, a_max=None))

        if self.mel_norms is not None:
            log_spec = log_spec / np.array(self.mel_norms)[:, None]

        return log_spec

    def __call__(
        self,
        raw_speech: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]],
        sampling_rate: Optional[int] = None,
        truncation: bool = True,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_attention_mask: Optional[bool] = True,
        padding: Optional[str] = "max_length",
        max_length: Optional[int] = None,
        **kwargs,
    ) -> BatchFeature:
        """
        `ClvpFeatureExtractor` is used to extract various voice specific properties such as the pitch and tone of the
        voice, speaking speed, and even speaking defects like a lisp or stuttering from a sample voice or `raw_speech`.

        First the voice is padded or truncated in a way such that it becomes a waveform of `self.default_audio_length`
        seconds long and then the log-mel spectrogram is extracted from it.

        Args:
            raw_speech (`np.ndarray`, `List[float]`, `List[np.ndarray]`, `List[List[float]]`):
                The sequence or batch of sequences to be padded. Each sequence can be a numpy array, a list of float
                values, a list of numpy arrays or a list of list of float values. Must be mono channel audio, not
                stereo, i.e. single float per timestep.
            sampling_rate (`int`, *optional*):
                The sampling rate at which the `raw_speech` input was sampled. It is strongly recommended to pass
                `sampling_rate` at the forward call to prevent silent errors and allow automatic speech recognition
                pipeline.
            truncation (`bool`, *optional*, default to `True`):
                Activates truncation to cut input sequences longer than *max_length* to *max_length*.
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the sequence to a multiple of the provided value.

                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                `>= 7.5` (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128.
            return_attention_mask (`bool`, *optional*, defaults to `True`):
                Whether to return the attention mask. If left to the default, it will return the attention mask.

                [What are attention masks?](../glossary#attention-mask)
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
            padding_value (`float`, *optional*, defaults to 0.0):
                The value that is used to fill the padding values / vectors.
            max_length (`int`, *optional*):
                The maximum input length of the inputs.
        """

        if sampling_rate is not None:
            if sampling_rate != self.sampling_rate:
                raise ValueError(
                    f"The model corresponding to this feature extractor: {self.__class__.__name__} was trained using a"
                    f" sampling rate of {self.sampling_rate}. Please make sure that the provided `raw_speech` input"
                    f" was sampled with {self.sampling_rate} and not {sampling_rate}."
                )
        else:
            logger.warning(
                "It is strongly recommended to pass the `sampling_rate` argument to this function. "
                "Failing to do so can result in silent errors that might be hard to debug."
            )

        is_batched_numpy = isinstance(raw_speech, np.ndarray) and len(raw_speech.shape) > 1
        if is_batched_numpy and len(raw_speech.shape) > 2:
            raise ValueError(f"Only mono-channel audio is supported for input to {self}")
        is_batched = is_batched_numpy or (
            isinstance(raw_speech, (list, tuple)) and (isinstance(raw_speech[0], (np.ndarray, tuple, list)))
        )

        if is_batched:
            raw_speech = [np.asarray([speech], dtype=np.float32).T for speech in raw_speech]
        elif not is_batched and not isinstance(raw_speech, np.ndarray):
            raw_speech = np.asarray(raw_speech, dtype=np.float32)
        elif isinstance(raw_speech, np.ndarray) and raw_speech.dtype is np.dtype(np.float64):
            raw_speech = raw_speech.astype(np.float32)

        # always return batch
        if not is_batched:
            raw_speech = [np.asarray([raw_speech]).T]

        batched_speech = BatchFeature({"input_features": raw_speech})

        max_length = self.default_audio_length * self.sampling_rate if max_length is None else max_length

        padded_inputs = self.pad(
            batched_speech,
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
        )

        # make sure list is in array format
        input_features = padded_inputs.get("input_features").transpose(2, 0, 1)

        input_features = [
            self._np_extract_fbank_features(waveform).astype(np.float32) for waveform in input_features[0]
        ]

        if isinstance(input_features[0], List):
            padded_inputs["input_features"] = [np.asarray(feature) for feature in input_features]
        else:
            padded_inputs["input_features"] = input_features

        return padded_inputs.convert_to_tensors(return_tensors)


__all__ = ["ClvpFeatureExtractor"]
