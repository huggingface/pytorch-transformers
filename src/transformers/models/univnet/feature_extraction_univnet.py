# Copyright 2023 The HuggingFace Team. All rights reserved.
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
"""Feature extractor class for UnivNetModel."""

from typing import Any, Dict, List, Optional, Union

import numpy as np

from ...audio_utils import mel_filter_bank, optimal_fft_length, spectrogram, window_function
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import PaddingStrategy, TensorType, logging


logger = logging.get_logger(__name__)


class UnivNetFeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs a UnivNet feature extractor.

    This class extracts log-mel-filter bank features from raw speech using the short time Fourier Transform (STFT). The
    STFT implementation follows that of TacoTron 2 and Hifi-GAN.

    This feature extractor inherits from [`~feature_extraction_sequence_utils.SequenceFeatureExtractor`] which contains
    most of the main methods. Users should refer to this superclass for more information regarding those methods.

    Args:
        feature_size (`int`, *optional*, defaults to 1):
            The feature dimension of the extracted features.
        sampling_rate (`int`, *optional*, defaults to 24000):
            The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
        padding_value (`float`, *optional*, defaults to 0.0):
            The value to pad with when applying the padding strategy defined by the `padding` argument to
            [`UnivNetFeatureExtractor.__call__`]. Should correspond to audio silence. The `pad_end` argument to
            `__call__` will also use this padding value.
        do_normalize (`bool`, *optional*, defaults to `False`):
            Whether to perform Tacotron 2 normalization on the input. Normalizing can help to significantly improve the
            performance for some models.
        num_mel_bins (`int`, *optional*, defaults to 100):
            The number of mel-frequency bins in the extracted spectrogram features. This should match
            `UnivNetModel.config.num_mel_bins`.
        hop_length (`int`, *optional*, defaults to 256):
            The direct number of samples between sliding windows. Otherwise referred to as "shift" in many papers. Note
            that this is different from other audio feature extractors such as [`SpeechT5FeatureExtractor`] which take
            the `hop_length` in ms.
        win_length (`int`, *optional*, defaults to 1024):
            The direct number of samples for each sliding window. Note that this is different from other audio feature
            extractors such as [`SpeechT5FeatureExtractor`] which take the `win_length` in ms.
        win_function (`str`, *optional*, defaults to `"hann_window"`):
            Name for the window function used for windowing, must be accessible via `torch.{win_function}`
        filter_length (`int`, *optional*, defaults to 1024):
            The number of FFT components to use. If `None`, this is determined using
            `transformers.audio_utils.optimal_fft_length`.
        max_length_s (`int`, *optional*, defaults to 10):
            The maximum input lenght of the model in seconds. This is used to pad the audio.
        fmin (`float`, *optional*, defaults to 0.0):
            Minimum mel frequency in Hz.
        fmax (`float`, *optional*):
            Maximum mel frequency in Hz. If not set, defaults to `sampling_rate / 2`.
        mel_floor (`float`, *optional*, defaults to 1e-09):
            Minimum value of mel frequency banks. Note that the way [`UnivNetFeatureExtractor`] uses `mel_floor` is
            different than in [`transformers.audio_utils.spectrogram`].
        center (`bool`, *optional*, defaults to `False`):
            Whether to pad the waveform so that frame `t` is centered around time `t * hop_length`. If `False`, frame
            `t` will start at time `t * hop_length`.
        compression_factor (`float`, *optional*, defaults to 1.0):
            The multiplicative compression factor for dynamic range compression during spectral normalization.
        compression_clip_val (`float`, *optional*, defaults to 1e-05):
            The clip value applied to the waveform before applying dynamic range compression during spectral
            normalization.
        normalize_min (`float`, *optional*, defaults to -11.512925148010254):
            The min value used for Tacotron 2-style linear normalization. The default is the original value from the
            Tacotron 2 implementation.
        normalize_max (`float`, *optional*, defaults to 2.3143386840820312):
            The max value used for Tacotron 2-style linear normalization. The default is the original value from the
            Tacotron 2 implementation.
        model_in_channels (`int`, *optional*, defaults to 64):
            The number of input channels to the [`UnivNetModel`] model. This should match
            `UnivNetModel.config.model_in_channels`.
        pad_end_length (`int`, *optional*, defaults to 10):
            If padding the end of each waveform, the number of spectrogram frames worth of samples to append. The
            number of appended samples will be `pad_end_length * hop_length`.
        return_attention_mask (`bool`, *optional*, defaults to `True`):
            Whether or not [`~UnivNetFeatureExtractor.__call__`] should return `attention_mask`.
    """

    model_input_names = ["input_features", "noise_sequence", "padding_mask"]

    def __init__(
        self,
        feature_size: int = 1,
        sampling_rate: int = 24000,
        padding_value: float = 0.0,
        do_normalize: bool = False,
        num_mel_bins: int = 100,
        hop_length: int = 256,
        win_length: int = 1024,
        win_function: str = "hann_window",
        filter_length: Optional[int] = 1024,
        max_length_s: int = 10,
        fmin: float = 0.0,
        fmax: Optional[float] = None,
        mel_floor: float = 1e-9,
        center: bool = False,
        compression_factor: float = 1.0,
        compression_clip_val: float = 1e-5,
        normalize_min: float = -11.512925148010254,
        normalize_max: float = 2.3143386840820312,
        model_in_channels: int = 64,
        pad_end_length: int = 10,
        return_attention_mask=True,
        **kwargs,
    ):
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            return_attention_mask=return_attention_mask,
            **kwargs,
        )

        self.do_normalize = do_normalize

        self.num_mel_bins = num_mel_bins
        self.hop_length = hop_length
        self.win_length = win_length
        self.win_function = win_function
        self.filter_length = filter_length
        self.fmin = fmin
        if fmax is None:
            # Follows the librosa.filters.mel implementation
            fmax = float(sampling_rate) / 2
        self.fmax = fmax
        self.mel_floor = mel_floor

        self.max_length_s = max_length_s
        self.num_max_samples = max_length_s * sampling_rate

        if self.filter_length is None:
            self.n_fft = optimal_fft_length(self.win_length)
        else:
            self.n_fft = self.filter_length
        self.n_freqs = (self.n_fft // 2) + 1

        self.window = window_function(window_length=self.win_length, name=self.win_function, periodic=True)

        self.mel_filters = mel_filter_bank(
            num_frequency_bins=self.n_freqs,
            num_mel_filters=self.num_mel_bins,
            min_frequency=self.fmin,
            max_frequency=self.fmax,
            sampling_rate=self.sampling_rate,
            norm="slaney",
            mel_scale="slaney",
        )

        self.center = center
        self.compression_factor = compression_factor
        self.compression_clip_val = compression_clip_val
        self.normalize_min = normalize_min
        self.normalize_max = normalize_max
        self.model_in_channels = model_in_channels
        self.pad_end_length = pad_end_length

    def normalize(self, spectrogram):
        return 2 * ((spectrogram - self.normalize_min) / (self.normalize_max - self.normalize_min)) - 1

    def denormalize(self, spectrogram):
        return self.normalize_min + (self.normalize_max - self.normalize_min) * ((spectrogram + 1) / 2)

    def mel_spectrogram(self, waveform: np.ndarray) -> np.ndarray:
        """
        Calculates log MEL spectrograms from a batch of waveforms. Note that the input waveform(s) will be padded by
        `int(self.n_fft - self.hop_length) / 2` on both sides using the `reflect` padding mode.

        Args:
            waveform (`np.ndarray` of shape `(length,)`):
                The input waveform. This must be a single real-valued, mono waveform.

        Returns:
            `numpy.ndarray`: Array containing a log-mel spectrogram of shape `(num_frames, num_mel_bins)`.
        """
        # Do custom padding based on the official MelGAN and Hifi-GAN implementations
        # See https://github.com/maum-ai/univnet/blob/9bb2b54838bb6d7ce767131cc7b8b61198bc7558/utils/stft.py#L84-L86
        waveform = np.pad(
            waveform,
            (int((self.n_fft - self.hop_length) / 2), int((self.n_fft - self.hop_length) / 2)),
            mode="reflect",
        )

        # Get the complex spectrogram.
        # Note: waveform must be unbatched currently due to the implementation of spectrogram(...).
        complex_spectrogram = spectrogram(
            waveform,
            window=self.window,
            frame_length=self.n_fft,
            hop_length=self.hop_length,
            fft_length=self.n_fft,
            power=None,
            center=self.center,
            mel_filters=None,
            mel_floor=None,
        )

        # Apply the MEL filter bank and MEL floor manually since UnivNet uses a slightly different implementation
        amplitude_spectrogram = np.sqrt(
            np.real(complex_spectrogram) ** 2 + np.imag(complex_spectrogram) ** 2 + self.mel_floor
        )
        mel_spectrogram = np.matmul(self.mel_filters.T, amplitude_spectrogram)

        # Perform spectral normalization to get the log mel spectrogram.
        log_mel_spectrogram = np.log(
            np.clip(mel_spectrogram, a_min=self.compression_clip_val, a_max=None) * self.compression_factor
        )

        # Return spectrogram with num_mel_bins last
        return log_mel_spectrogram.T

    def generate_noise(
        self,
        noise_length: int,
        generator: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Generates a random noise sequence of standard Gaussian noise for use in the `noise_sequence` argument of
        [`UnivNetModel.forward`].

        Args:
            spectrogram_length (`int`):
                The length (dim 0) of the generated noise.
            model_in_channels (`int`, *optional*, defaults to `None`):
                The number of features (dim 1) of the generated noise. This should correspond to the
                `model_in_channels` of the [`UnivNetGan`] model. If not set, this will default to
                `self.config.model_in_channels`.
            generator (`numpy.random.Generator`, *optional*, defaults to `None`)
                An optional `numpy.random.Generator` random number generator to control noise generation. If not set, a
                new generator with fresh entropy will be created.

        Returns:
            `numpy.ndarray`: Array containing random standard Gaussian noise of shape `(noise_length,
            model_in_channels)`.
        """
        if generator is None:
            generator = np.random.default_rng()

        noise_shape = (noise_length, self.model_in_channels)
        noise = generator.standard_normal(noise_shape, dtype=np.float32)

        return noise

    def batch_decode(self, waveforms, waveform_lengths=None) -> List[np.ndarray]:
        r"""
        Removes padding from generated audio after running [`UnivNetModel.forward`]. This returns a ragged list of 1D
        audio waveform arrays and not a single tensor/array because in general the waveforms will have different
        lengths after removing padding.

        Args:
            waveforms (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
                The batched output waveforms from the [`UnivNetModel`].
            waveform_lengths (`torch.FloatTensor` of shape `(batch_size,)`, *optional*):
                The batched lengths of each waveform before padding.

        Returns:
            `List[np.ndarray]`: A ragged list of 1D waveform arrays with padding removed.
        """
        # Collapse the batched waveform tensor to a list of 1D audio waveforms
        waveforms = [waveform.detach().clone().cpu().numpy() for waveform in waveforms]

        if waveform_lengths is not None:
            waveforms = [waveform[: waveform_lengths[i]] for i, waveform in enumerate(waveforms)]

        return waveforms

    def __call__(
        self,
        raw_speech: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]],
        sampling_rate: Optional[int] = None,
        padding: Union[bool, str, PaddingStrategy] = True,
        max_length: Optional[int] = None,
        truncation: bool = True,
        pad_to_multiple_of: Optional[int] = None,
        return_noise: bool = True,
        generator: Optional[np.random.Generator] = None,
        pad_end: bool = False,
        pad_length: Optional[int] = None,
        do_normalize: Optional[str] = None,
        return_attention_mask: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
    ) -> BatchFeature:
        """
        Main method to featurize and prepare for the model one or several sequence(s).

        Args:
            raw_speech (`np.ndarray`, `List[float]`, `List[np.ndarray]`, `List[List[float]]`):
                The sequence or batch of sequences to be padded. Each sequence can be a numpy array, a list of float
                values, a list of numpy arrays or a list of list of float values. Must be mono channel audio, not
                stereo, i.e. single float per timestep.
            sampling_rate (`int`, *optional*):
                The sampling rate at which the `raw_speech` input was sampled. It is strongly recommended to pass
                `sampling_rate` at the forward call to prevent silent errors and allow automatic speech recognition
                pipeline.
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
                Select a strategy to pad the input `raw_speech` waveforms (according to the model's padding side and
                padding index) among:

                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).

                If `pad_end = True`, that padding will occur before the `padding` strategy is applied.
            max_length (`int`, *optional*):
                Maximum length of the returned list and optionally padding length (see above).
            truncation (`bool`, *optional*, defaults to `True`):
                Activates truncation to cut input sequences longer than `max_length` to `max_length`.
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the sequence to a multiple of the provided value.

                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                `>= 7.5` (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128.
            return_noise (`bool`, *optional*, defaults to `True`):
                Whether to generate and return a noise waveform for use in [`UnivNetModel.forward`].
            generator (`numpy.random.Generator`, *optional*, defaults to `None`):
                An optional `numpy.random.Generator` random number generator to use when generating noise.
            pad_end (`bool`, *optional*, defaults to `False`):
                Whether to pad the end of each waveform with silence. This can help reduce artifacts at the end of the
                generated audio sample; see https://github.com/seungwonpark/melgan/issues/8 for more details. This
                padding will be done before the padding strategy specified in `padding` is performed.
            pad_length (`int`, *optional*, defaults to `None`):
                If padding the end of each waveform, the length of the padding in spectrogram frames. If not set, this
                will default to `self.config.pad_end_length`.
            do_normalize (`bool`, *optional*):
                Whether to perform Tacotron 2 normalization on the input. Normalizing can help to significantly improve
                the performance for some models. If not set, this will default to `self.config.do_normalize`.
            return_attention_mask (`bool`, *optional*):
                Whether to return the attention mask. If left to the default, will return the attention mask according
                to the specific feature_extractor's default.

                [What are attention masks?](../glossary#attention-mask)

            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.np.array` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
        """
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize

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
            raw_speech = [np.asarray(speech, dtype=np.float32) for speech in raw_speech]
        elif not is_batched and not isinstance(raw_speech, np.ndarray):
            raw_speech = np.asarray(raw_speech, dtype=np.float32)
        elif isinstance(raw_speech, np.ndarray) and raw_speech.dtype is np.dtype(np.float64):
            raw_speech = raw_speech.astype(np.float32)

        # always return batch
        if not is_batched:
            raw_speech = [np.asarray(raw_speech, dtype=np.float32)]

        # Pad end to reduce artifacts
        if pad_end:
            pad_length = pad_length if pad_length is not None else self.pad_end_length
            raw_speech = [
                np.pad(waveform, (0, pad_length * self.hop_length), constant_values=self.padding_value)
                for waveform in raw_speech
            ]

        batched_speech = BatchFeature({"input_features": raw_speech})

        padded_inputs = self.pad(
            batched_speech,
            padding=padding,
            max_length=max_length if max_length is not None else self.num_max_samples,
            truncation=truncation,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
        )

        # make sure list is in array format
        # input_features = padded_inputs.get("input_features").transpose(2, 0, 1)
        input_features = padded_inputs.get("input_features")

        mel_spectrograms = [self.mel_spectrogram(waveform) for waveform in input_features]

        if isinstance(input_features[0], List):
            batched_speech["input_features"] = [np.asarray(mel, dtype=np.float32) for mel in mel_spectrograms]
        else:
            batched_speech["input_features"] = [mel.astype(np.float32) for mel in mel_spectrograms]

        # convert attention_mask to correct format
        attention_mask = padded_inputs.get("attention_mask")
        if attention_mask is not None:
            batched_speech["padding_mask"] = [np.asarray(array, dtype=np.int32) for array in attention_mask]

        if return_noise:
            noise = [
                self.generate_noise(spectrogram.shape[0], generator)
                for spectrogram in batched_speech["input_features"]
            ]
            batched_speech["noise_sequence"] = noise

        if do_normalize:
            batched_speech["input_features"] = [
                self.normalize(spectrogram) for spectrogram in batched_speech["input_features"]
            ]

        if return_tensors is not None:
            batched_speech = batched_speech.convert_to_tensors(return_tensors)

        return batched_speech

    def to_dict(self) -> Dict[str, Any]:
        output = super().to_dict()

        # Don't serialize these as they are derived from the other properties.
        names = ["window", "mel_filters", "n_fft", "n_freqs", "num_max_samples"]
        for name in names:
            if name in output:
                del output[name]

        return output


__all__ = ["UnivNetFeatureExtractor"]
