<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Feature extractors

Feature extractors preprocess audio data into the correct format for a given model. It takes the raw audio signal and converts it into a tensor that can be fed to a model. The tensor shape depends on the model, but the feature extractor will correctly preprocess the audio data for you given the model you're using. Feature extractors also include methods for padding, truncation, and resampling.

Call [`~AutoFeatureExtractor.from_pretrained`] to load a feature extractor and its preprocessor configuration from the Hugging Face [Hub](https://hf.co/models) or local directory. The feature extractor and preprocessor configuration is saved in a [preprocessor_config.json](https://hf.co/openai/whisper-tiny/blob/main/preprocessor_config.json) file.

Pass the audio signal, typically stored in `array`, to the feature extractor and set the `sampling_rate` parameter to the pretrained audio models sampling rate. It is important the sampling rate of the audio data matches the sampling rate of the data a pretrained audio model was trained on.

```py
from transformers import AutoFeatureExtractor

feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
processed_sample = feature_extractor(dataset[0]["audio"]["array"], sampling_rate=16000)
processed_sample
{'input_values': [array([ 9.4472744e-05,  3.0777880e-03, -2.8888427e-03, ...,
       -2.8888427e-03,  9.4472744e-05,  9.4472744e-05], dtype=float32)]}
```

The feature extractor returns an input, `input_values`, that is ready for the model to consume.

This guide walks you through the feature extractor classes and how to preprocess audio data.

## Feature extractor classes

Transformers feature extractors inherit from the base [`SequenceFeatureExtractor`] class which subclasses [`FeatureExtractionMixin`].

- [`SequenceFeatureExtractor`] provides a method to [`~SequenceFeatureExtractor.pad`] sequences to a certain length to avoid uneven sequence lengths.
- [`FeatureExtractionMixin`] provides [`~FeatureExtractionMixin.from_pretrained`] and [`~FeatureExtractionMixin.save_pretrained`] to load and save a feature extractor.

There are two ways you can load a feature extractor, [`AutoFeatureExtractor`] and a model-specific feature extractor class.

<hfoptions id="feature-extractor-classes">
<hfoption id="AutoFeatureExtractor">

The [AutoClass](./model_doc/auto) API automatically loads the correct feature extractor for a given model.

Use [`~AutoFeatureExtractor.from_pretrained`] to load a feature extractor.

```py
from transformers import AutoFeatureExtractor

feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-tiny")
```

</hfoption>
<hfoption id="model-specific feature extractor">

Every pretrained audio model has a specific associated feature extractor for correctly processing audio data. When you load a feature extractor, it retrieves the feature extractors configuration (feature size, chunk length, etc.) from [preprocessor_config.json](https://hf.co/openai/whisper-tiny/blob/main/preprocessor_config.json).

A feature extractor can be loaded directly from its model-specific class.

```py
from transformers import WhisperFeatureExtractor

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")
```

</hfoption>
</hfoptions>

## Preprocess

A feature extractor expects the input as a PyTorch tensor of a certain shape. The exact input shape can vary depending on the specific audio model you're using.

For example, [Whisper](https://huggingface.co/docs/transformers/model_doc/whisper) expects `input_features` to be a tensor of shape `(batch_size, feature_size, sequence_length)` but [Wav2Vec2](https://hf.co/docs/transformers/model_doc/wav2vec2) expects `input_values` to be a tensor of shape `(batch_size, sequence_length)`.

The feature extractor generates the correct input shape for whichever audio model you're using.

A feature extractor also sets the sampling rate (the number of audio signal values taken per second) of the audio files. The sampling rate of your audio data must match the sampling rate of the dataset a pretrained model was trained on. This value is typically given in the model card.

Load a dataset and feature extractor with [`~FeatureExtractionMixin.from_pretrained`].

```py
from datasets import load_dataset, Audio
from transformers import AutoFeatureExtractor

dataset = load_dataset("PolyAI/minds14", name="en-US", split="train")
feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
```

Check out the first example from the dataset and access the `audio` column which contains `array`, the raw audio signal.

```py
dataset[0]["audio"]["array"]
array([ 0.        ,  0.00024414, -0.00024414, ..., -0.00024414,
        0.        ,  0.        ])
```

The feature extractor preprocesses `array` into the expected input format for a given audio model. Use the `sampling_rate` parameter to set the appropriate sampling rate.

```py
processed_dataset = feature_extractor(dataset[0]["audio"]["array"], sampling_rate=16000)
processed_dataset
{'input_values': [array([ 9.4472744e-05,  3.0777880e-03, -2.8888427e-03, ...,
       -2.8888427e-03,  9.4472744e-05,  9.4472744e-05], dtype=float32)]}
```

### Padding

Audio sequence lengths that are different is an issue because Transformers expects all sequences to have the same lengths so they can be batched. Uneven sequence lengths can't be batched.

```py
dataset[0]["audio"]["array"].shape
(86699,)

dataset[1]["audio"]["array"].shape
(53248,)
```

Padding adds a special *padding token* to ensure all sequences have the same length. The feature extractor adds a `0` - interpreted as silence - to `array` to pad it. Set `padding=True` to pad sequences to the longest sequence length in the batch.

```py
def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=16000,
        padding=True,
    )
    return inputs

processed_dataset = preprocess_function(dataset[:5])
processed_dataset["input_values"][0].shape
(86699,)

processed_dataset["input_values"][1].shape
(86699,)
```

### Truncation

Models can only process sequences up to a certain length before crashing.

Truncation is a strategy for removing excess tokens from a sequence to ensure it doesn't exceed the maximum length. Set `truncation=True` to truncate a sequence to the length in the `max_length` parameter.

```py
def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=16000,
        max_length=50000,
        truncation=True,
    )
    return inputs

processed_dataset = preprocess_function(dataset[:5])
processed_dataset["input_values"][0].shape
(50000,)

processed_dataset["input_values"][1].shape
(50000,)
```

### Resampling

The [Datasets](https://hf.co/docs/datasets/index) library can also resample audio data to match an audio models expected sampling rate. This method resamples the audio data on the fly when they're loaded which can be faster than resampling the entire dataset in-place.

The audio dataset you've been working on has a sampling rate of 8kHz and the pretrained model expects 16kHz.

```py
dataset[0]["audio"]
{'path': '/root/.cache/huggingface/datasets/downloads/extracted/f507fdca7f475d961f5bb7093bcc9d544f16f8cab8608e772a2ed4fbeb4d6f50/en-US~JOINT_ACCOUNT/602ba55abb1e6d0fbce92065.wav',
 'array': array([ 0.        ,  0.00024414, -0.00024414, ..., -0.00024414,
         0.        ,  0.        ]),
 'sampling_rate': 8000}
```

Call [`~datasets.Dataset.cast_column`] on the `audio` column to upsample the sampling rate to 16kHz.

```py
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
```

When you load the dataset sample, it is now resampled to 16kHz.

```py
dataset[0]["audio"]
{'path': '/root/.cache/huggingface/datasets/downloads/extracted/f507fdca7f475d961f5bb7093bcc9d544f16f8cab8608e772a2ed4fbeb4d6f50/en-US~JOINT_ACCOUNT/602ba55abb1e6d0fbce92065.wav',
 'array': array([ 1.70562416e-05,  2.18727451e-04,  2.28099874e-04, ...,
         3.43842403e-05, -5.96364771e-06, -1.76846661e-05]),
 'sampling_rate': 16000}
```
