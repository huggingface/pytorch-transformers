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

# Pixtral

## Overview

The Pixtral model was released by the Mistral AI team in a [blog post](https://mistral.ai/news/pixtral-12b/). Pixtral is a multimodal version of [Mistral](mistral), incorporating a 400 million parameter vision encoder trained from scratch.

The intro from the blog says the following:

*Pixtral is trained to understand both natural images and documents, achieving 52.5% on the MMMU reasoning benchmark, surpassing a number of larger models. The model shows strong abilities in tasks such as chart and figure understanding, document question answering, multimodal reasoning and instruction following. Pixtral is able to ingest images at their natural resolution and aspect ratio, giving the user flexibility on the number of tokens used to process an image. Pixtral is also able to process any number of images in its long context window of 128K tokens. Unlike previous open-source models, Pixtral does not compromise on text benchmark performance to excel in multimodal tasks.*

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/pixtral_architecture.webp"
alt="drawing" width="600"/>

<small> Pixtral architecture. Taken from the <a href="https://mistral.ai/news/pixtral-12b/">blog post.</a> </small>

Tips:

- Pixtral is a multimodal model, taking images and text as input, and producing text as output.
- This model follows the [Llava](llava) architecture. The model uses [`PixtralVisionModel`] for its vision encoder, and [`MistralForCausalLM`] for its language decoder.
- The main contribution is the 2d ROPE (rotary position embeddings) on the images, and support for arbitrary image sizes (the images are not padded together nor are they resized).
- Similar to [Llava](llava), the model internally replaces the `[IMG]` token placeholders by image embeddings from the vision encoder. The format for one or multiple prompts is the following:
```
"<s>[INST][IMG]\nWhat are the things I should be cautious about when I visit this place?[/INST]"
```
Then, the processor will replace each `[IMG]` token with a number of `[IMG]` tokens that depend on the height and the width of each image. Each *row* of the image is separated by an `[IMG_BREAK]` token, and each image is separated by an `[IMG_END]` token. It's advised to use the `apply_chat_template` method of the processor, which takes care of all of this. See the [usage section](#usage) for more info.

This model was contributed by [amyeroberts](https://huggingface.co/amyeroberts) and [ArthurZ](https://huggingface.co/ArthurZ). The original code can be found [here](https://github.com/vllm-project/vllm/pull/8377).

## Usage

At inference time, it's advised to use the processor's `apply_chat_template` method, which correctly formats the prompt for the model:

```python
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image

model_id = "mistral-community/pixtral-12b"
processor = AutoProcessor.from_pretrained(model_id)
model = LlavaForConditionalGeneration.from_pretrained(model_id).to("cuda")

url_dog = "https://picsum.photos/id/237/200/300"
url_mountain = "https://picsum.photos/seed/picsum/200/300"

chat = [
    {
      "role": "user", "content": [
        {"type": "text", "content": "Can this animal"}, 
        {"type": "image"}, 
        {"type": "text", "content": "live here?"}, 
        {"type": "image"}
      ]
    }
]

prompt = processor.apply_chat_template(chat)
inputs = processor(text=prompt, images=[url_dog, url_mountain], return_tensors="pt").to(model.device)
generate_ids = model.generate(**inputs, max_new_tokens=500)
output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
```

## PixtralVisionConfig

[[autodoc]] PixtralVisionConfig

## PixtralVisionModel

[[autodoc]] PixtralVisionModel
    - forward

## PixtralImageProcessor

[[autodoc]] PixtralImageProcessor
    - preprocess

## PixtralImageProcessorFast

[[autodoc]] PixtralImageProcessorFast
    - preprocess

## PixtralProcessor

[[autodoc]] PixtralProcessor
