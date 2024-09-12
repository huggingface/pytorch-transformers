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

# Pipeline

The [`Pipeline`] is a simple but powerful inference API that is readily available for a variety of machine learning tasks with any model from the Hugging Face [Hub](https://hf.co/models). Tailor the [`Pipeline`] to your task with certain task specific parameters, such as adding timestamps to an automatic speech recognition (ASR) pipeline for transcribing meeting notes. [`Pipeline`] supports GPUs, Apple silicon, and half-precision weights to accelerate inference and save memory.

<Youtube id=tiZFewofSLM/>

Transformers has two pipeline classes, a generic [`Pipeline`] and many individual task-specific pipelines like [`TextGenerationPipeline`] or [`VisualQuestionAnsweringPipeline`]. Load these individual pipelines by setting the task identifier in the `task` parameter in [`Pipeline`]. You can find the task identifier for each pipeline in their API documentation.

Each task is configured to use a default pretrained model and preprocessor, but this can be overriden with the `model` parameter if you want to use a different model.

For example, to use the [`TextGenerationPipeline`] with [Gemma 2](./model_doc/gemma2), set `task="text-generation"` and `model="google/gemma-2-2b"`.

```py
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="google/gemma-2-2b")
pipeline("the secret to baking a really good cake is ")
[{'generated_text': 'the secret to baking a really good cake is 1. the right ingredients 2. the'}]
```

When you have more than one input, pass them as a list.

```py
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="google/gemma-2-2b", device="cuda")
pipeline(["the secret to baking a really good cake is ", "a baguette is "])
[[{'generated_text': 'the secret to baking a really good cake is 1. the right ingredients 2. the'}],
 [{'generated_text': 'a baguette is 100% bread.\n\na baguette is 100%'}]]
```

This guide will introduce you to the [`Pipeline`], demonstrate its features, and show you how to configure its various parameters.

## Tasks

[`Pipeline`] is compatible with many machine learning tasks across different modalities. You just need to pass an appropriate input to the pipeline and it will handle the rest.

Here are some examples of how to use [`Pipeline`] for different tasks and modalities.

<hfoption id="tasks">
<hfoption id="summarization">

```py
from transformers import pipeline

pipeline = pipeline(task="summarization", model="google/pegasus-billsum")
pipeline("Section was formerly set out as section 44 of this title. As originally enacted, this section contained two further provisions that 'nothing in this act shall be construed as in any wise affecting the grant of lands made to the State of California by virtue of the act entitled 'An act authorizing a grant to the State of California of the Yosemite Valley, and of the land' embracing the Mariposa Big-Tree Grove, approved June thirtieth, eighteen hundred and sixty-four; or as affecting any bona-fide entry of land made within the limits above described under any law of the United States prior to the approval of this act.' The first quoted provision was omitted from the Code because the land, granted to the state of California pursuant to the Act cite, was receded to the United States. Resolution June 11, 1906, No. 27, accepted the recession.")
[{'summary_text': 'Instructs the Secretary of the Interior to convey to the State of California all right, title, and interest of the United States in and to specified lands which are located within the Yosemite and Mariposa National Forests, California.'}]
```

</hfoption>
<hfoption id="automatic speech recognition">

```py
from transformers import pipeline

pipeline = pipeline(task="automatic-speech-recognition", model="openai/whisper-large-v3")
pipeline("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
{'text': ' I have a dream that one day this nation will rise up and live out the true meaning of its creed.'}
```

</hfoption>
<hfoption id="image classification">

```py
from transformers import pipeline

pipeline = pipeline(task="image-classification", model="google/vit-base-patch16-224")
pipeline(images="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg")
[{'label': 'lynx, catamount', 'score': 0.43350091576576233},
 {'label': 'cougar, puma, catamount, mountain lion, painter, panther, Felis concolor',
  'score': 0.034796204417943954},
 {'label': 'snow leopard, ounce, Panthera uncia',
  'score': 0.03240183740854263},
 {'label': 'Egyptian cat', 'score': 0.02394474856555462},
 {'label': 'tiger cat', 'score': 0.02288915030658245}]
```

</hfoption>
<hfoption id="visual question answering">

```py
from transformers import pipeline

pipeline = pipeline(task="visual-question-answering", model="Salesforce/blip-vqa-base")
pipeline(
    image="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-few-shot.jpg",
    question="What is in the image?",
)
[{'answer': 'statue of liberty'}]
```

</hfoption>
</hfoptions>

## Parameters

At a minimum, a [`Pipeline`] only requires a task identifier, model, and the appropriate input. But there are many parameters available to configure the pipeline with, from task-specific parameters to optimizing performance.

This section will walk you through some of the more important parameters.

### Device

[`Pipeline`] is compatible with many hardware types, including GPUs, CPUs, Apple silicon, and more. This is configured with the `device` parameter. By default, [`Pipeline`] runs on a CPU which is given by `device=-1`.

<hfoptions id="device">
<hfoption id="GPU">

To run [`Pipeline`] on a GPU, set `device` to the associated CUDA device id. For example, `device=0` runs on the first GPU.

```py
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="google/gemma-2-2b", device=0)
pipeline("the secret to baking a really good cake is ")
```

You could also let [Accelerate](https://hf.co/docs/accelerate/index), a library for distributed training, automatically choose how to load and store the model weights on the appropriate device. This is especially useful if you have multiple devices. Accelerate loads and stores the model weights on the fastest device first, and then moves the weights to other devices (CPU, hard drive) as needed. Set `device_map="auto"` to let Accelerate choose the device.

> [!TIP]
> Make sure you have [Accelerate](https://hf.co/docs/accelerate/basic_tutorials/install) installed.
>
> ```py
> !pip install -U accelerate
> ```

```py
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="google/gemma-2-2b", device_map="auto")
pipeline("the secret to baking a really good cake is ")
```

</hfoption>
<hfoption id="Apple silicon">

To run [`Pipeline`] on Apple silicon, set `device="mps"`.

```py
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="google/gemma-2-2b", device="mps")
pipeline("the secret to baking a really good cake is ")
```

</hfoption>
</hfoptions>

### Batch inference

[`Pipeline`] can also process batches of inputs with the `batch_size` parameter. Batch inference may improve speed, especially on a GPU, but it isn't guaranteed to. Other variables such as hardware, data, and the model itself can affect whether batch inference improves speed. For this reason, batch inference is disabled by default.

In this example, when there are 4 inputs and `batch_size` is set to 2, [`Pipeline`] passes a batch of 2 inputs to the model at a time.

```py
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="google/gemma-2-2b", device="cuda", batch_size=2)
pipeline(["the secret to baking a really good cake is", "a baguette is", "paris is the", "hotdogs are"])
[[{'generated_text': 'the secret to baking a really good cake is to use a good cake mix.\n\ni’'}],
 [{'generated_text': 'a baguette is'}],
 [{'generated_text': 'paris is the most beautiful city in the world.\n\ni’ve been to paris 3'}],
 [{'generated_text': 'hotdogs are a staple of the american diet. they are a great source of protein and can'}]]
```

Another good use case for batch inference is when you stream data in [`Pipeline`].

```py
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
import datasets

# KeyDataset is a utility that returns the item in the dict returned by the dataset
dataset = datasets.load_dataset("imdb", name="plain_text", split="unsupervised")
pipeline = pipeline(task="text-classification", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english", device="cuda")
for out in pipeline(KeyDataset(dataset, "text"), batch_size=8, truncation="only_first"):
    print(out)
```

Here are some general rules of thumb for determining whether batch inference can help improve performance.

1. The only way to know for sure is to measure performance on your model, data, and hardware.
2. Don't batch inference if you're constrained by latency (a live inference product for example).
3. Don't batch inference if you're using a CPU.
4. Don't batch inference if you don't know the `sequence_length` of your data. Measure performance, iteratively add to `sequence_length`, and include out-of-memory (OOM) checks to recover from failures.
5. Do batch inference if your `sequence_length` is regular, and keep pushing it until you reach an OOM error. The larger the GPU, the more likely batch inference is to be beneficial.
6. Do make sure you can handle OOM errors if you decide to do batch inference.

### Task-specific parameters

The [`Pipeline`] accepts any parameters that are supported by each individual task pipeline. Make sure to check out each individual task pipeline to see what type of parameters are available. If you can't find a parameter that would be useful for your use case, please feel free to open a GitHub [issue](https://github.com/huggingface/transformers/issues/new?assignees=&labels=feature&template=feature-request.yml) to request it!

Here are some examples of enabling these task-specific parameters in [`Pipeline`].

<hfoptions id="task-specific-parameters">
<hfoption id="automatic speech recognition">

The [`AutomaticSpeechRecognitionPipeline.__call__`] method has a `return_timestamps` parameter that returns when each word was spoken by setting it to `"word"`. This parameter can be passed along to [`Pipeline`].

```py
from transformers import pipeline

pipeline = pipeline(task="automatic-speech-recognition", model="openai/whisper-large-v3")
pipeline(audio="https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac", return_timestamp="word")
{'text': ' I have a dream that one day this nation will rise up and live out the true meaning of its creed.',
 'chunks': [{'text': ' I', 'timestamp': (0.0, 1.1)},
  {'text': ' have', 'timestamp': (1.1, 1.44)},
  {'text': ' a', 'timestamp': (1.44, 1.62)},
  {'text': ' dream', 'timestamp': (1.62, 1.92)},
  {'text': ' that', 'timestamp': (1.92, 3.7)},
  {'text': ' one', 'timestamp': (3.7, 3.88)},
  {'text': ' day', 'timestamp': (3.88, 4.24)},
  {'text': ' this', 'timestamp': (4.24, 5.82)},
  {'text': ' nation', 'timestamp': (5.82, 6.78)},
  {'text': ' will', 'timestamp': (6.78, 7.36)},
  {'text': ' rise', 'timestamp': (7.36, 7.88)},
  {'text': ' up', 'timestamp': (7.88, 8.46)},
  {'text': ' and', 'timestamp': (8.46, 9.2)},
  {'text': ' live', 'timestamp': (9.2, 10.34)},
  {'text': ' out', 'timestamp': (10.34, 10.58)},
  {'text': ' the', 'timestamp': (10.58, 10.8)},
  {'text': ' true', 'timestamp': (10.8, 11.04)},
  {'text': ' meaning', 'timestamp': (11.04, 11.4)},
  {'text': ' of', 'timestamp': (11.4, 11.64)},
  {'text': ' its', 'timestamp': (11.64, 11.8)},
  {'text': ' creed.', 'timestamp': (11.8, 12.3)}]}
```

</hfoption>
<hfoption id="text generation">

The [`~TextGenerationPipeline.__call__`] method has a `return_full_text` parameter that determines whether to return the full text or only the generated text. Set it to `False` to only return the generated text.

[`~TextGenerationPipeline.__call__`] also additional keyword arguments from the [`~GenerationMixin.generate`] method, which itself takes generation configuration parameters from [`GenerationConfig`]. To return more than one generated sequence, set `num_return_sequences` to a value greater than 1. Pass this parameter to [`Pipeline`].

```py
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="openai-community/gpt2")
pipeline("the secret to baking a good cake is", num_return_sequences=4, return_full_text=False)
[{'generated_text': ' how easy it is for me to do it with my hands. You must not go nuts, or the cake is going to fall out.'},
 {'generated_text': ' to prepare the cake before baking. The key is to find the right type of icing to use and that icing makes an amazing frosting cake.\n\nFor a good icing cake, we give you the basics'},
 {'generated_text': " to remember to soak it in enough water and don't worry about it sticking to the wall. In the meantime, you could remove the top of the cake and let it dry out with a paper towel.\n"},
 {'generated_text': ' the best time to turn off the oven and let it stand 30 minutes. After 30 minutes, stir and bake a cake in a pan until fully moist.\n\nRemove the cake from the heat for about 12'}]
```

</hfoption>
</hfoptions>

## Chunk batching

There are some instances where you need to process data in chunks.

- for some data types, a single input (for example, a really long audio file) may need to be chunked into multiple parts before it can be processed
- for some tasks, like zero-shot classification or question answering, a single input may need multiple forward passes which can cause issues with the `batch_size` parameter

The [`ChunkPipeline`] class is designed to handle these use cases. Both pipeline classes are used in the same way, but since [`ChunkPipeline`] can automatically handle batching you don't need to worry about the number of forward passes your inputs trigger. Instead, you can optimize `batch_size` independently of the inputs.

Here is how it differs from a regular [`Pipeline`].

<hfoptions id="chunk-batching">
<hfoption id="ChunkPipeline">

```py
all_model_outputs = []
for preprocessed in pipeline.preprocess(inputs):
    model_outputs = pipeline.model_forward(preprocessed)
    all_model_outputs.append(model_outputs)
outputs =pipeline.postprocess(all_model_outputs)
```

</hfoption>
<hfoption id="Pipeline">

```py
preprocessed = pipeline.preprocess(inputs)
model_outputs = pipeline.forward(preprocessed)
outputs = pipeline.postprocess(model_outputs)
```

</hfoption>
</hfoptions>

## Large datasets

For inference on large datasets, you can iterate directly over the dataset. This avoids immediately allocating memory for the entire dataset, and you don't need to worry about creating batches yourself. As mentioned in the [Batch inference](#batch-inference) section, you can try using the `batch_size` parameter to see if it improves performance.

```py
from transformers.pipelines.pt_utils import KeyDataset
from transformers import pipeline
from datasets import load_dataset

dataset = datasets.load_dataset("imdb", name="plain_text", split="unsupervised")
pipeline = pipeline(task="text-classification", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english", device="cuda")
for out in pipeline(KeyDataset(dataset, "text"), batch_size=8, truncation="only_first"):
    print(out)
```

Other ways to run inference on large datasets with [`Pipeline`] include using an iterator or generator.

```py
def data():
    for i in range(1000):
        yield f"My example {i}"

pipeline = pipeline(model="openai-community/gpt2", device=0)
generated_characters = 0
for out in pipeline(data()):
    generated_characters += len(out[0]["generated_text"])
```

## Large models

[Accelerate](https://hf.co/docs/accelerate/index) enables a couple of optimizations for running large models with [`Pipeline`]. Make sure Accelerate is installed first.

```py
!pip install -U accelerate
```

As mentioned in the [Device](#device) section, the `device_map="auto"` setting is useful for automatically distributing the model across the fastest devices (GPUs) first before dispatching to other slower devices if available (CPU, hard drive).

[`Pipeline`] supports half-precision weights, torch.float16, which can be significantly faster and save memory. Performance loss is negligible for most models, especially for larger models. If your hardware supports it, you can enable torch.bfloat16 instead for more range.

> [!TIP]
> Inputs are internally converted to torch.float16, and it only works for models with a PyTorch backend.

Lastly, [`Pipeline`] also accepts quantized models to really reduce memory usage even further. Make sure you have the [bitsandbytes](https://hf.co/docs/bitsandbytes/installation) library installed first, and then add `load_in_8bit=True` to `model_kwargs` in the pipeline.

```py
import torch
from transformers import pipeline, BitsAndBytesConfig

pipeline = pipeline(model="google/gemma-7b", torch_dtype=torch.bfloat16, device_map="auto", model_kwargs={"quantization_config": BitsAndBytesConfig(load_in_8bit=True)})
pipeline("the secret to baking a good cake is ")
[{'generated_text': 'the secret to baking a good cake is 1. the right ingredients 2. the right'}]
```
