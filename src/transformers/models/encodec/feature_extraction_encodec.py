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
"""Feature extractor class for EnCodec."""

from typing import List, Optional, Union

import numpy as np

from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import PaddingStrategy, TensorType, logging


logger = logging.get_logger(__name__)


class EncodecFeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs an EnCodec feature extractor.

    This feature extractor inherits from [`~feature_extraction_sequence_utils.SequenceFeatureExtractor`] which contains
    most of the main methods. Users should refer to this superclass for more information regarding those methods.

    Instantiating a feature extractor with the defaults will yield a similar configuration to that of the
    [facebook/encodec_24khz](https://huggingface.co/facebook/encodec_24khz) architecture.

    Args:
        feature_size (`int`, *optional*, defaults to 1):
            The feature dimension of the extracted features. Use 1 for mono, 2 for stereo.
        sampling_rate (`int`, *optional*, defaults to 24000):
            The sampling rate at which the audio waveform should be digitalized expressed in hertz (Hz).
        padding_value (`float`, *optional*, defaults to 0.0):
            The value that is used to fill the padding values.
        chunk_length_s (`float`, *optional*):
            If defined the audio is pre-processed into chunks of lengths `chunk_length_s` and then encoded.
        overlap (`float`, *optional*):
            Defines the overlap between each chunk. It is used to compute the `chunk_stride` using the following
            formulae : `int((1.0 - self.overlap) * self.chunk_length)`.
    """

    model_input_names = ["input_values", "padding_mask"]

    def __init__(
        self,
        feature_size: int = 1,
        sampling_rate: int = 24000,
        padding_value: float = 0.0,
        chunk_length_s: float = None,
        overlap: float = None,
        **kwargs,
    ):
        super().__init__(feature_size=feature_size, sampling_rate=sampling_rate, padding_value=padding_value, **kwargs)
        self.chunk_length_s = chunk_length_s
        self.overlap = overlap

    # This is a property because you might want to change the chunk_length_s on the fly
    @property
    def chunk_length(self) -> Optional[int]:
        if self.chunk_length_s is None:
            return None
        else:
            return int(self.chunk_length_s * self.sampling_rate)

    # This is a property because you might want to change the chunk_length_s on the fly
    @property
    def chunk_stride(self) -> Optional[int]:
        if self.chunk_length_s is None or self.overlap is None:
            return None
        else:
            return max(1, int((1.0 - self.overlap) * self.chunk_length))

    def __call__(
        self,
        raw_audio: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]],
        padding: Optional[Union[bool, str, PaddingStrategy]] = None,
        truncation: Optional[bool] = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        sampling_rate: Optional[int] = None,
    ) -> BatchFeature:
        """
        Main method to featurize and prepare for the model one or several sequence(s).

        Args:
            raw_audio (`np.ndarray`, `List[float]`, `List[np.ndarray]`, `List[List[float]]`):
                The sequence or batch of sequences to be processed. Each sequence can be a numpy array, a list of float
                values, a list of numpy arrays or a list of list of float values. The numpy array must be of shape
                `(num_samples,)` for mono audio (`feature_size = 1`), or `(2, num_samples)` for stereo audio
                (`feature_size = 2`).
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding
                index) among:

                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            truncation (`bool`, *optional*, defaults to `False`):
                Activates truncation to cut input sequences longer than `max_length` to `max_length`.
            max_length (`int`, *optional*):
                Maximum length of the returned list and optionally padding length (see above).
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
            sampling_rate (`int`, *optional*):
                The sampling rate at which the `audio` input was sampled. It is strongly recommended to pass
                `sampling_rate` at the forward call to prevent silent errors.
        """
        if sampling_rate is not None:
            if sampling_rate != self.sampling_rate:
                raise ValueError(
                    f"The model corresponding to this feature extractor: {self} was trained using a sampling rate of"
                    f" {self.sampling_rate}. Please make sure that the provided audio input was sampled with"
                    f" {self.sampling_rate} and not {sampling_rate}."
                )
        else:
            logger.warning(
                "It is strongly recommended to pass the `sampling_rate` argument to this function. "
                "Failing to do so can result in silent errors that might be hard to debug."
            )

        if padding and truncation:
            raise ValueError("Both padding and truncation were set. Make sure you only set one.")
        elif padding is None:
            # by default let's pad the inputs
            padding = True

        is_batched = bool(
            isinstance(raw_audio, (list, tuple)) and (isinstance(raw_audio[0], (np.ndarray, tuple, list)))
        )

        if is_batched:
            raw_audio = [np.asarray(audio, dtype=np.float32).T for audio in raw_audio]
        elif not is_batched and not isinstance(raw_audio, np.ndarray):
            raw_audio = np.asarray(raw_audio, dtype=np.float32)
        elif isinstance(raw_audio, np.ndarray) and raw_audio.dtype is np.dtype(np.float64):
            raw_audio = raw_audio.astype(np.float32)

        # always return batch
        if not is_batched:
            raw_audio = [np.asarray(raw_audio).T]

        # verify inputs are valid
        for idx, example in enumerate(raw_audio):
            if example.ndim > 2:
                raise ValueError(f"Expected input shape (channels, length) but got shape {example.shape}")
            if self.feature_size == 1 and example.ndim != 1:
                raise ValueError(f"Expected mono audio but example has {example.shape[-1]} channels")
            if self.feature_size == 2 and example.shape[-1] != 2:
                raise ValueError(f"Expected stereo audio but example has {example.shape[-1]} channels")

        padded_inputs = None
        input_values = BatchFeature({"input_values": raw_audio})
        if self.chunk_stride is not None and self.chunk_length is not None and max_length is None:
            if truncation:
                max_length = min(array.shape[0] for array in raw_audio)
                nb_step = int(np.floor(max_length / self.chunk_stride))
                max_length = (nb_step - 1) * self.chunk_stride + self.chunk_length
            elif padding:
                max_length = max(array.shape[0] for array in raw_audio)
                nb_step = int(np.ceil(max_length / self.chunk_stride))
                max_length = (nb_step - 1) * self.chunk_stride + self.chunk_length
                padding = "max_length"
            else:
                padded_inputs = input_values

        # normal padding on batch
        if padded_inputs is None:
            padded_inputs = self.pad(
                input_values,
                max_length=max_length,
                truncation=truncation,
                padding=padding,
                return_attention_mask=padding,
            )
            if padding:
                padded_inputs["padding_mask"] = padded_inputs.pop("attention_mask")

        input_values = []
        for example in padded_inputs.pop("input_values"):
            if self.feature_size == 1:
                example = example[..., None]
            input_values.append(example.T)

        padded_inputs["input_values"] = input_values
        if return_tensors is not None:
            padded_inputs = padded_inputs.convert_to_tensors(return_tensors)

        return padded_inputs


__all__ = ["EncodecFeatureExtractor"]
