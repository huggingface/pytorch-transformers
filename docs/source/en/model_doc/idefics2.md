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

# Idefics2

## Overview

The Idefics2 model was proposed in [What matters when building vision-language models?](https://arxiv.org/abs/2405.02246) by Léo Tronchon, Hugo Laurencon, Victor Sanh. The accompanying blog post can be found [here](https://huggingface.co/blog/idefics2).

Idefics2 is an open multimodal model that accepts arbitrary sequences of image and text inputs and produces text
outputs. The model can answer questions about images, describe visual content, create stories grounded on multiple
images, or simply behave as a pure language model without visual inputs. It improves upon IDEFICS-1, notably on
document understanding, OCR, or visual reasoning. Idefics2 is lightweight (8 billion parameters) and treats
images in their native aspect ratio and resolution, which allows for varying inference efficiency.

The abstract from the paper is the following:

*The growing interest in vision-language models (VLMs) has been driven by improvements in large language models and vision transformers. Despite the abundance of literature on this subject, we observe that critical decisions regarding the design of VLMs are often not justified. We argue that these unsupported decisions impede progress in the field by making it difficult to identify which choices improve model performance. To address this issue, we conduct extensive experiments around pre-trained models, architecture choice, data, and training methods. Our consolidation of findings includes the development of Idefics2, an efficient foundational VLM of 8 billion parameters. Idefics2 achieves state-of-the-art performance within its size category across various multimodal benchmarks, and is often on par with models four times its size. We release the model (base, instructed, and chat) along with the datasets created for its training.*

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/idefics2_architecture.png"
alt="drawing" width="600"/>

<small> Idefics2 architecture. Taken from the <a href="https://arxiv.org/abs/2405.02246">original paper.</a> </small>

This model was contributed by [amyeroberts](https://huggingface.co/amyeroberts).
The original code can be found [here](https://huggingface.co/HuggingFaceM4/idefics2).

## Usage tips

- Each sample can contain multiple images, and the number of images can vary between samples. The processor will pad the inputs to the maximum number of images in a batch for input to the model.
- The processor has a `do_image_splitting` option. If `True`, each input image will be split into 4 sub-images, and concatenated with the original to form 5 images. This is useful for increasing model performance. Make sure `processor.image_processor.do_image_splitting` is set to `False` if the model was not trained with this option.
- `text` passed to the processor should have the `<image>` tokens where the images should be inserted. And `<end_of_utterance>` at the end of each utterance if the text is a chat message.
- The processor has its own `apply_chat_template` method to convert chat messages to text that can then be passed as `text` to the processor.

Example of how to use the processor on chat messages:

```python
import requests
from PIL import Image
from transformers import Idefics2Processor, Idefics2ForConditionalGeneration
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

url_1 = "http://images.cocodataset.org/val2017/000000039769.jpg"
url_2 = "http://images.cocodataset.org/val2017/000000219578.jpg"

image_1 = Image.open(requests.get(url_1, stream=True).raw)
image_2 = Image.open(requests.get(url_2, stream=True).raw)
images = [image_1, image_2]

messages = [{
    "role": "user",
    "content": [
        {"type": "text", "text": "What’s the difference between these two images?"},
        {"type": "image"},
        {"type": "image"},
    ],
}]

processor = Idefics2Processor.from_pretrained("HuggingFaceM4/idefics2-8b")
model = Idefics2ForConditionalGeneration.from_pretrained("HuggingFaceM4/idefics2-8b")
model.to(device)

# at inference time, one needs to pass `add_generation_prompt=True` in order to make sure the model completes the prompt
text = processor.apply_chat_template(messages, add_generation_prompt=True)
print(text)
# 'User: What’s the difference between these two images?<image><image><end_of_utterance>\nAssistant:'

inputs = processor(images=images, text=text, return_tensors="pt").to(device)

generated_text = model.generate(**inputs, max_new_tokens=500)
generated_text = processor.batch_decode(generated_text, skip_special_tokens=True)[0]
print("Generated text:", generated_text)
```

- During training, it's important to determine which tokens the model should not learn. For Idefics2, this typically comes down to the image and padding tokens. This means that one can create the labels as follows:

```python
import requests
from PIL import Image
from transformers import Idefics2Processor, Idefics2ForConditionalGeneration
import torch

url_1 = "http://images.cocodataset.org/val2017/000000039769.jpg"
url_2 = "http://images.cocodataset.org/val2017/000000219578.jpg"

image_1 = Image.open(requests.get(url_1, stream=True).raw)
image_2 = Image.open(requests.get(url_2, stream=True).raw)
images = [image_1, image_2]

messages = [{
    "role": "user",
    "content": [
        {"type": "text", "text": "What’s the difference between these two images?"},
        {"type": "image"},
        {"type": "image"},
    ],
},
{
    "role": "assistant",
    "content": [
        {"type": "text", "text": "The difference is that one image is about dogs and the other one about cats."},
    ],
}]

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = Idefics2Processor.from_pretrained("HuggingFaceM4/idefics2-8b")
model = Idefics2ForConditionalGeneration.from_pretrained("HuggingFaceM4/idefics2-8b")
model.to(device)

text = processor.apply_chat_template(messages, add_generation_prompt=False)
inputs = processor(images=images, text=text, return_tensors="pt").to(device)

labels = inputs.input_ids.clone()
labels[labels == processor.tokenizer.pad_token_id] = -100
labels[labels == model.config.image_token_id] = -100

inputs["labels"] = labels

outputs = model(**inputs)
loss = outputs.loss
loss.backward()
```

Do note that when training Idefics2 on multi-turn conversations between a user and an assistant, one typically also sets all the tokens corresponding to the user messages to -100.

## Model optimizations: Flash Attention

The code snippets above showcase inference without any optimization tricks. However, one can drastically speed up the model by leveraging [Flash Attention](../perf_train_gpu_one#flash-attention-2), which is a faster implementation of the attention mechanism used inside the model.

First, make sure to install the latest version of Flash Attention 2 to include the sliding window attention feature.

```bash
pip install -U flash-attn --no-build-isolation
```

Make also sure that you have a hardware that is compatible with Flash-Attention 2. Read more about it in the official documentation of the [flash attention repository](https://github.com/Dao-AILab/flash-attention). Make also sure to load your model in half-precision (e.g. `torch.float16`)

To load and run a model using Flash Attention-2, simply change the code snippet above with the following change:

```diff
model = Idefics2ForConditionalGeneration.from_pretrained(
    "HuggingFaceM4/idefics2-8b",
+    torch_dtype=torch.float16,    
+    attn_implementation="flash_attention_2",
).to(device)
```

## Shrinking down Idefics2 using quantization

As the Idefics2 model has 8 billion parameters, that would require about 16GB of GPU RAM in half precision (float16), since each parameter is stored in 2 bytes. However, one can shrink down the size of the model using [quantization](../quantization.md). If the model is quantized to 4 bits (or half a byte per parameter), that requires only about 3.5GB of RAM.

Quantizing a model is as simple as passing a `quantization_config` to the model. One can change the code snippet above with the changes below. We'll leverage the BitsAndyBytes quantization (but refer to [this page](../quantization.md) for other quantization methods):

```diff
+ from transformers import BitsAndBytesConfig

+ quantization_config = BitsAndBytesConfig(
+    load_in_4bit=True,
+    bnb_4bit_quant_type="nf4",
+    bnb_4bit_use_double_quant=True,
+    bnb_4bit_compute_dtype=torch.float16
+ )
model = Idefics2ForConditionalGeneration.from_pretrained(
    "HuggingFaceM4/idefics2-8b",
+    torch_dtype=torch.float16,    
+    quantization_config=quantization_config,
).to(device)
```

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with Idefics2. If you're interested in submitting a resource to be included here, please feel free to open a Pull Request and we'll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

- A notebook on how to fine-tune Idefics2 on a custom dataset using the [Trainer](../main_classes/trainer.md) can be found [here](https://colab.research.google.com/drive/1NtcTgRbSBKN7pYD3Vdx1j9m8pt3fhFDB?usp=sharing). It supports both full fine-tuning as well as (quantized) LoRa.
- A script regarding how to fine-tune Idefics2 using the TRL library can be found [here](https://gist.github.com/edbeeching/228652fc6c2b29a1641be5a5778223cb).
- Demo notebook regarding fine-tuning Idefics2 for JSON extraction use cases can be found [here](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/Idefics2). 🌎

## Idefics2Config

[[autodoc]] Idefics2Config


## Idefics2Model

[[autodoc]] Idefics2Model
    - forward


## Idefics2ForConditionalGeneration

[[autodoc]] Idefics2ForConditionalGeneration
    - forward


## Idefics2ImageProcessor
[[autodoc]] Idefics2ImageProcessor
    - preprocess


## Idefics2Processor
[[autodoc]] Idefics2Processor
    - __call__