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

# Multimodal Chat Templates for Vision and Audio LLMs

In this section, we'll explore how to use chat templates with multimodal models, enabling your templates to handle a variety of inputs such as text, images, and audio. Multimodal models provide richer, more interactive experiences, and understanding how to effectively combine these inputs within your templates is key. We’ll walk through how to work with different modalities, configure your templates for optimal performance, and tackle common challenges along the way.

Just like with text-only LLMs, multimodal models expect a chat with **messages** each of which includes a **role** and **content**. However for multimodal models chat templates are a part of the [Processor](./main_cllasses/processors). Let's see how we can format our prompts when there are images or videos in the input along with text.


## How to use chat templates with image inputs

For models such as [LLaVA](https://huggingface.co/llava-hf) the prompts can be formatted as below. Notice that the only difference from text-only models is that we need to pass as well a placeholder for input images. TO accomodate for extra modalities, each **content** will be a list containing either a text or an image **type**.

Let's make this concrete with a quick example using the `llava-hf/llava-onevision-qwen2-0.5b-ov-hf` model:

```python
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

model_id = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
processor = AutoProcessor.from_pretrained(model_id)

messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a friendly chatbot who always responds in the style of a pirate"}],
    },
    {
      "role": "user",
      "content": [
            {"type": "image"},
            {"type": "text", "text": "What are these?"},
        ],
    },
]

formatted_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
print(formatted_prompt)
```

This yields a string in LLaVA's expected input format with many `<image>` tokens prepended before the text.
```text
'<|im_start|>system 
<|im_start|>system 
You are a friendly chatbot who always responds in the style of a pirate<|im_end|><|im_start|>user <image>
What are these?<|im_end|>
```


### Passing Images as Paths or URLs for Seamless Integration

To incorporate images into your chat templates, you can pass them as file paths or URLs. This method automatically loads the image, processes it, and prepares the necessary pixel values to create ready-to-use inputs for the model. This approach simplifies the integration of images, enabling seamless multimodal functionality.

Let's see how it works with an example using the same model as above. This time we'll indicate an image URL with `"url"` key in the message's **content** and ask the chat template to `tokenize` and `return_dict`. Currently the following sources are supported to load images: "base64", "url", and "path .

```python
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

model_id = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
model = LlavaOnevisionForConditionalGeneration.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a friendly chatbot who always responds in the style of a pirate"}],
    },
    {
      "role": "user",
      "content": [
            {"type": "image", "url": "http://images.cocodataset.org/val2017/000000039769.jpg"},
            {"type": "text", "text": "What are these?"},
        ],
    },
]

processed_chat = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
print(processed_chat.keys())
```

This yields a dictionary with inputs processed and ready to be further passed into [`~GenerationMixin.generate`] to generate text.
```text
dict_keys(["input_ids", "attention_mask", "pixel_values", "image_sizes"])
```


## How to use chat templates with video inputs

Some vision models support videos as inputs as well as images. The message format is very similar to the image-only models with tiny differences to handle loading videos from a URL. We can continue using the same model as before since it supports videos.

Here's an example of how to set up a conversation with video inputs. Notice the extra `kwargs` passed to `processor.apply_chat_template()`. The key parameter here is `num_frames`, which controls how many frames to sample uniformly from the video. Each model checkpoint has a maximum frame count it was trained with, and exceeding this limit can significantly impact generation quality. So, it’s important to choose a frame count that fits both the model's capacity and your computational resources. If you don't specify `num_frames`, the entire video will be loaded without any frame sampling.

You also have the option to choose a specific framework to load the video, depending on your preferences or needs. Currently, we support `decord`, `pyav` (the default), `opencv`, and `torchvision`. For this example, we’ll use `decord`, as it's a bit faster than `pyav`.


<Tip>

Note that if you are trying to load a video from URL, you can decode the video only with `pyav` or `decord` as backend.

</Tip>


```python
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

model_id = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
model = LlavaOnevisionForConditionalGeneration.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a friendly chatbot who always responds in the style of a pirate"}],
    },
    {
      "role": "user",
      "content": [
            {"type": "video", "url": "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/720/Big_Buck_Bunny_720_10s_10MB.mp4"},
            {"type": "text", "text": "What do you see in this video?"},
        ],
    },
]

processed_chat = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    num_frames=32,
    video_load_backend="decord",
)
print(processed_chat.keys())
```



## How to use chat templates with audio inputs

To be supported soon. Stay tuned! 🤗 



## Is there also a pipeline for multimodal conversations?

Yes, similarly to text generation pipeline we support an [`ImageTextToTextPipeline`]. Currently it accepts only images as inputs but we are planning to add support for video inputs in the future. The pipeline supports chat inputs in the same format as we have seen above. Apart from that the pipeline will accept chats in OpenAI format. However note, that OpenAI format in supported exclusively within the pipeline to make inference easier and more accessible. 

Here is how OpenAI conversation format looks like:

```python
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "What is in this image?",
            },
            {
                "type": "image_url",
                "image_url": {"url": f"http://images.cocodataset.org/val2017/000000039769.jpg"},
            },
        ],
    }
]
```

## Best Practices for Multimodal Template Configuration


To add a custom chat template for your multimodal LLM, simply create your template using [Jinja](https://jinja.palletsprojects.com/en/3.1.x/templates/) and set it with `processor.chat_template`. If you're new to writing chat templates or need some tips, check out our [tutorial here](./chat_template_advanced) for helpful guidance.

In some cases, you may want your template to handle a **list of content** from multiple modalities, while still supporting a plain string for text-only inference. Here's an example of how you can achieve that, using the [Llama-Vision](https://huggingface.co/collections/meta-llama/metas-llama-32-multimodal-models-675bfd70e574a62dd0e4059b) chat template.


```
{% for message in messages %}
{% if loop.index0 == 0 %}{{ bos_token }}{% endif %}
{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' }}
{% if message['content'] is string %}
{{ message['content'] }}
{% else %}
{% for content in message['content'] %}
{% if content['type'] == 'image' %}
{{ '<|image|>' }}
{% elif content['type'] == 'text' %}
{{ content['text'] }}
{% endif %}
{% endfor %}
{% endif %}
{{ '<|eot_id|>' }}
{% endfor %}
{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}
```
