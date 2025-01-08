# Copyright 2021 The HuggingFace Team. All rights reserved.
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
import subprocess
from typing import Union

import numpy as np
import requests

from ..utils import add_end_docstrings, is_torch_available, is_torchaudio_available, logging
from .base import Pipeline, build_pipeline_init_args


if is_torch_available():
    from ..models.auto.modeling_auto import MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES

logger = logging.get_logger(__name__)


def ffmpeg_read(bpayload: bytes, sampling_rate: int) -> np.array:
    """
    Helper function to read an audio file through ffmpeg.
    """
    ar = f"{sampling_rate}"
    ac = "1"
    format_for_conversion = "f32le"
    ffmpeg_command = [
        "ffmpeg",
        "-i",
        "pipe:0",
        "-ac",
        ac,
        "-ar",
        ar,
        "-f",
        format_for_conversion,
        "-hide_banner",
        "-loglevel",
        "quiet",
        "pipe:1",
    ]

    try:
        ffmpeg_process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    except FileNotFoundError:
        raise ValueError("ffmpeg was not found but is required to load audio files from filename")
    output_stream = ffmpeg_process.communicate(bpayload)
    out_bytes = output_stream[0]

    audio = np.frombuffer(out_bytes, np.float32)
    if audio.shape[0] == 0:
        raise ValueError("Malformed soundfile")
    return audio


@add_end_docstrings(build_pipeline_init_args(has_feature_extractor=True))
class AudioClassificationPipeline(Pipeline):
    """
    Audio classification pipeline using any `AutoModelForAudioClassification`. This pipeline predicts the class of a
    raw waveform or an audio file. In case of an audio file, ffmpeg should be installed to support multiple audio
    formats.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> classifier = pipeline(model="superb/wav2vec2-base-superb-ks")
    >>> classifier("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac")
    [{'score': 0.997, 'label': '_unknown_'}, {'score': 0.002, 'label': 'left'}, {'score': 0.0, 'label': 'yes'}, {'score': 0.0, 'label': 'down'}, {'score': 0.0, 'label': 'stop'}]
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)


    This pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"audio-classification"`.

    See the list of available models on
    [huggingface.co/models](https://huggingface.co/models?filter=audio-classification).
    """

    def __init__(self, *args, **kwargs):
        # Default, might be overriden by the model.config.
        kwargs["top_k"] = kwargs.get("top_k", 5)
        super().__init__(*args, **kwargs)

        if self.framework != "pt":
            raise ValueError(f"The {self.__class__} is only available in PyTorch.")

        self.check_model_type(MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES)

    def __call__(
        self,
        inputs: Union[np.ndarray, bytes, str],
        **kwargs,
    ):
        """
        Classify the sequence(s) given as inputs. See the [`AutomaticSpeechRecognitionPipeline`] documentation for more
        information.

        Args:
            inputs (`np.ndarray` or `bytes` or `str` or `dict`):
                The inputs is either :
                    - `str` that is the filename of the audio file, the file will be read at the correct sampling rate
                      to get the waveform using *ffmpeg*. This requires *ffmpeg* to be installed on the system.
                    - `bytes` it is supposed to be the content of an audio file and is interpreted by *ffmpeg* in the
                      same way.
                    - (`np.ndarray` of shape (n, ) of type `np.float32` or `np.float64`)
                        Raw audio at the correct sampling rate (no further check will be done)
                    - `dict` form can be used to pass raw audio sampled at arbitrary `sampling_rate` and let this
                      pipeline do the resampling. The dict must be either be in the format `{"sampling_rate": int,
                      "raw": np.array}`, or `{"sampling_rate": int, "array": np.array}`, where the key `"raw"` or
                      `"array"` is used to denote the raw audio waveform.
            top_k (`int`, *optional*, defaults to None):
                The number of top labels that will be returned by the pipeline. If the provided number is `None` or
                higher than the number of labels available in the model configuration, it will default to the number of
                labels.
            function_to_apply(`str`, *optional*, defaults to "softmax"):
                The function to apply to the model output. By default, the pipeline will apply the softmax function to
                the output of the model. Valid options: ["softmax", "sigmoid", "none"]. Note that passing Python's
                built-in `None` will default to "softmax", so you need to pass the string "none" to disable any
                post-processing.

        Return:
            A list of `dict` with the following keys:

            - **label** (`str`) -- The label predicted.
            - **score** (`float`) -- The corresponding probability.
        """
        return super().__call__(inputs, **kwargs)

    def _sanitize_parameters(self, top_k=None, function_to_apply=None, **kwargs):
        # No parameters on this pipeline right now
        postprocess_params = {}
        if top_k is not None:
            if top_k > self.model.config.num_labels:
                top_k = self.model.config.num_labels
            postprocess_params["top_k"] = top_k
        if function_to_apply is not None:
            if function_to_apply not in ["softmax", "sigmoid", "none"]:
                raise ValueError(
                    f"Invalid value for `function_to_apply`: {function_to_apply}. "
                    "Valid options are ['softmax', 'sigmoid', 'none']"
                )
            postprocess_params["function_to_apply"] = function_to_apply
        else:
            postprocess_params["function_to_apply"] = "softmax"
        return {}, {}, postprocess_params

    def preprocess(self, inputs):
        if isinstance(inputs, str):
            if inputs.startswith("http://") or inputs.startswith("https://"):
                # We need to actually check for a real protocol, otherwise it's impossible to use a local file
                # like http_huggingface_co.png
                inputs = requests.get(inputs).content
            else:
                with open(inputs, "rb") as f:
                    inputs = f.read()

        if isinstance(inputs, bytes):
            inputs = ffmpeg_read(inputs, self.feature_extractor.sampling_rate)

        if isinstance(inputs, dict):
            # Accepting `"array"` which is the key defined in `datasets` for
            # better integration
            if not ("sampling_rate" in inputs and ("raw" in inputs or "array" in inputs)):
                raise ValueError(
                    "When passing a dictionary to AudioClassificationPipeline, the dict needs to contain a "
                    '"raw" key containing the numpy array representing the audio and a "sampling_rate" key, '
                    "containing the sampling_rate associated with that array"
                )

            _inputs = inputs.pop("raw", None)
            if _inputs is None:
                # Remove path which will not be used from `datasets`.
                inputs.pop("path", None)
                _inputs = inputs.pop("array", None)
            in_sampling_rate = inputs.pop("sampling_rate")
            inputs = _inputs
            if in_sampling_rate != self.feature_extractor.sampling_rate:
                import torch

                if is_torchaudio_available():
                    from torchaudio import functional as F
                else:
                    raise ImportError(
                        "torchaudio is required to resample audio samples in AudioClassificationPipeline. "
                        "The torchaudio package can be installed through: `pip install torchaudio`."
                    )

                inputs = F.resample(
                    torch.from_numpy(inputs), in_sampling_rate, self.feature_extractor.sampling_rate
                ).numpy()

        if not isinstance(inputs, np.ndarray):
            raise TypeError("We expect a numpy ndarray as input")
        if len(inputs.shape) != 1:
            raise ValueError("We expect a single channel audio input for AudioClassificationPipeline")

        processed = self.feature_extractor(
            inputs, sampling_rate=self.feature_extractor.sampling_rate, return_tensors="pt"
        )
        return processed

    def _forward(self, model_inputs):
        model_outputs = self.model(**model_inputs)
        return model_outputs

    def postprocess(self, model_outputs, top_k=5, function_to_apply="softmax"):
        if function_to_apply == "softmax":
            probs = model_outputs.logits[0].softmax(-1)
        elif function_to_apply == "sigmoid":
            probs = model_outputs.logits[0].sigmoid()
        else:
            probs = model_outputs.logits[0]
        scores, ids = probs.topk(top_k)

        scores = scores.tolist()
        ids = ids.tolist()

        labels = [{"score": score, "label": self.model.config.id2label[_id]} for score, _id in zip(scores, ids)]

        return labels
