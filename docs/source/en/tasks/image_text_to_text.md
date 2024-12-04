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

# Image-text-to-text

[[open-in-colab]]

Image-text-to-text models, also known as vision language models (VLMs), are language models that take an image input. These models can tackle various tasks, from visual question answering to image segmentation. This task shares many similarities with image-to-text, but with some overlapping use cases like image captioning. Image-to-text models only take image inputs and often accomplish a specific task, whereas VLMs take open-ended text and image inputs and are more generalist models.

In this guide, we provide a brief overview of VLMs and show how to use them with Transformers for inference.

To begin with, there are multiple types of VLMs:
- base models used for fine-tuning
- chat fine-tuned models for conversation
- instruction fine-tuned models

This guide focuses on inference with an instruction-tuned model.

Let's begin installing the dependencies.

```bash
pip install -q transformers accelerate flash_attn
```

Let's initialize the model and the processor.

```python
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

device = torch.device("cuda")
model = AutoModelForImageTextToText.from_pretrained(
    "HuggingFaceM4/idefics2-8b",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
).to(device)

processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b")
```

This model has a [chat template](./chat_templating) that helps user parse chat outputs. Moreover, the model can also accept multiple images as input in a single conversation or message. We will now prepare the inputs.

The image inputs look like the following.

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.png" alt="Two cats sitting on a net"/>
</div>

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg" alt="A bee on a pink flower"/>
</div>


```python
from PIL import Image
import requests

img_urls =["https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.png",
           "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"]
images = [Image.open(requests.get(img_urls[0], stream=True).raw),
          Image.open(requests.get(img_urls[1], stream=True).raw)]
```

Below is an example of the chat template. We can feed conversation turns and the last message as an input by appending it at the end of the template.


```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "What do we see in this image?"},
        ]
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "In this image we can see two cats on the nets."},
        ]
    },
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "And how about this image?"},
        ]
    },
]
```

We will now call the processors' [`~ProcessorMixin.apply_chat_template`] method to preprocess its output along with the image inputs.

```python
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=[images[0], images[1]], return_tensors="pt").to(device)
```

We can now pass the preprocessed inputs to the model.

```python
with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=500)
generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

print(generated_texts)
## ['User: What do we see in this image? \nAssistant: In this image we can see two cats on the nets. \nUser: And how about this image? \nAssistant: In this image we can see flowers, plants and insect.']
```

## Pipeline

The fastest way to get started is to use the [`Pipeline`] API. Specify the `"image-text-to-text"` task and the model you want to use.

```python
from transformers import pipeline
pipe = pipeline("image-text-to-text", model="llava-hf/llava-interleave-qwen-0.5b-hf")
```

The example below uses chat templates to format the text inputs.

```python
messages = [
     {
         "role": "user",
         "content": [
             {
                 "type": "image",
                 "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",
             },
             {"type": "text", "text": "Describe this image."},
         ],
     },
     {
         "role": "assistant",
         "content": [
             {"type": "text", "text": "There's a pink flower"},
         ],
     },
 ]
```

Pass the chat template formatted text and image to [`Pipeline`] and set `return_full_text=False` to remove the input from the generated output.

```python
outputs = pipe(text=messages, max_new_tokens=20, return_full_text=False)
outputs[0]["generated_text"]
#  with a yellow center in the foreground. The flower is surrounded by red and white flowers with green stems
```

## Streaming

We can use [text streaming](./generation_strategies#streaming) for a better generation experience. Transformers supports streaming with the [`TextStreamer`] or [`TextIteratorStreamer`] classes. We will use the [`TextIteratorStreamer`] with IDEFICS-8B.

Assume we have an application that keeps chat history and takes in the new user input. We will preprocess the inputs as usual and initialize [`TextIteratorStreamer`] to handle the generation in a separate thread. This allows you to stream the generated text tokens in real-time. Any generation arguments can be passed to [`TextIteratorStreamer`].


```python
import time
from transformers import TextIteratorStreamer
from threading import Thread

def model_inference(
    user_prompt,
    chat_history,
    max_new_tokens,
    images
):
    user_prompt = {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": user_prompt},
        ]
    }
    chat_history.append(user_prompt)
    streamer = TextIteratorStreamer(
        processor.tokenizer,
        skip_prompt=True,
        timeout=5.0,
    )

    generation_args = {
        "max_new_tokens": max_new_tokens,
        "streamer": streamer,
        "do_sample": False
    }

    # add_generation_prompt=True makes model generate bot response
    prompt = processor.apply_chat_template(chat_history, add_generation_prompt=True)
    inputs = processor(
        text=prompt,
        images=images,
        return_tensors="pt",
    ).to(device)
    generation_args.update(inputs)

    thread = Thread(
        target=model.generate,
        kwargs=generation_args,
    )
    thread.start()

    acc_text = ""
    for text_token in streamer:
        time.sleep(0.04)
        acc_text += text_token
        if acc_text.endswith("<end_of_utterance>"):
            acc_text = acc_text[:-18]
        yield acc_text

    thread.join()
```

Now let's call the `model_inference` function we created and stream the values.

```python
generator = model_inference(
    user_prompt="And what is in this image?",
    chat_history=messages[:2],
    max_new_tokens=100,
    images=images
)

for value in generator:
  print(value)

# In
# In this
# In this image ...
```

## Fit models in smaller hardware

VLMs are often large and need to be optimized to fit on smaller hardware. Transformers supports many model quantization libraries, and here we will only show int8 quantization with [Quanto](./quantization/quanto#quanto). int8 quantization offers memory improvements up to 75 percent (if all weights are quantized). However it is no free lunch, since 8-bit is not a CUDA-native precision, the weights are quantized back and forth on the fly, which adds up to latency.

First, install dependencies.

```bash
pip install -U quanto bitsandbytes
```

To quantize a model during loading, we need to first create [`QuantoConfig`]. Then load the model as usual, but pass `quantization_config` during model initialization.

```python
from transformers import AutoModelForImageTextToText, QuantoConfig

model_id = "HuggingFaceM4/idefics2-8b"
quantization_config = QuantoConfig(weights="int8")
quantized_model = AutoModelForImageTextToText.from_pretrained(
    model_id, device_map="cuda", quantization_config=quantization_config
)
```

And that's it, we can use the model the same way with no changes.

## Further Reading

Here are some more resources for the image-text-to-text task.

- [Image-text-to-text task page](https://huggingface.co/tasks/image-text-to-text) covers model types, use cases, datasets, and more.
- [Vision Language Models Explained](https://huggingface.co/blog/vlms) is a blog post that covers everything about vision language models and supervised fine-tuning using [TRL](https://huggingface.co/docs/trl/en/index).
