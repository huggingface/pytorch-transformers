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

# Processors

Multimodal models require a preprocessor capable of handling inputs that combine more than one modality. Depending on the input modality, a processor needs to convert text into an array of tensors, images into pixel values, and audio into an array with tensors with the correct sampling rate.

For example, [PaliGemma](./model_doc/paligemma) is a vision-language model that uses the [SigLIP](./model_doc/siglip) image processor and the [Llama](./model_doc/llama) tokenizer. A [`ProcessorMixin`] class wraps both of these preprocessor types, providing a single and unified processor class for a multimodal model.

To load a processor, call [`~ProcessorMixin.from_pretrained`]. Pass the input type to the processor to generate the expected model inputs, the input ids and pixel values.

```py
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from PIL import Image
import requests

processor = AutoProcessor.from_pretrained("google/paligemma-3b-pt-224")

prompt = "answer en Where is the cow standing?"
url = "https://huggingface.co/gv-hf/PaliGemma-test-224px-hf/resolve/main/cow_beach_1.png"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=prompt, images=image, return_tensors="pt")
inputs
```

This guide describes the processor class and how to preprocess multimodal inputs.

## Processor classes

All processors inherit from the [`ProcessorMixin`] class which provides methods like [`~ProcessorMixin.from_pretrained`], [`~ProcessorMixin.save_pretrained`], and [`~ProcessorMixin.push_to_hub`] for loading, saving, and sharing processors to the Hub repsectively.

There are two ways to load a processor, with an [`AutoProcessor`] and with a model-specific processor class.

<hfoptions id="processor-class">
<hfoption id="AutoProcessor">

The [AutoClass](./model_doc/auto) API provides a simple interface to load processors without directly specifying the specific model class it belongs to.

Use [`~AutoProcessor.from_pretrained`] to load a processor.

```py
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("google/paligemma-3b-pt-224")
```

</hfoption>
<hfoption id="model-specific processor">

Processors are also associated with a specific pretrained multimodal model class. You can load a processor directly from the model class with [`~ProcessorMixin.from_pretrained`].

```py
from transformers import WhisperProcessor

processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
```

You could also separately load the two preprocessor types, [`WhisperTokenizerFast`] and [`WhisperFeatureExtractor`].

```py
from transformers import WhisperTokenizerFast, WhisperFeatureExtractor, WhisperProcessor

tokenizer = WhisperTokenizerFast.from_pretrained("openai/whisper-tiny")
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")
processor = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
```

</hfoption>
</hfoptions>

## Preprocess

Processors preprocess multimodal inputs into the expected Transformers format. There are a couple combinations of input modalities that a processor can handle such as text and audio or text and image.

Automatic speech recognition (ASR) tasks require a processor that can handle text and audio inputs. Load a dataset and take a look at the `audio` and `text` columns (you can remove the other columns which aren't needed).

```py
from datasets import load_dataset

dataset = load_dataset("lj_speech", split="train")
dataset = dataset.map(remove_columns=["file", "id", "normalized_text"])
dataset[0]["audio"]
{'array': array([-7.3242188e-04, -7.6293945e-04, -6.4086914e-04, ...,
         7.3242188e-04,  2.1362305e-04,  6.1035156e-05], dtype=float32),
 'path': '/root/.cache/huggingface/datasets/downloads/extracted/917ece08c95cf0c4115e45294e3cd0dee724a1165b7fc11798369308a465bd26/LJSpeech-1.1/wavs/LJ001-0001.wav',
 'sampling_rate': 22050}

dataset[0]["text"]
'Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition'
```

Remember to resample the sampling rate to match the model's requirements.

```py
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
```

Load a processor and pass the audio `array` to the `audio` parameter and pass the `text` column to the `text` parameter.

```py
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("openai/whisper-tiny")

def prepare_dataset(example):
    audio = example["audio"]
    example.update(processor(audio=audio["array"], text=example["text"], sampling_rate=16000))
    return example
```

Apply the `prepare_dataset` function to the dataset to preprocess it. The processor returns the `input_features` for the `audio` column and `labels` for the text column.

```py
prepare_dataset(dataset[0])
```
