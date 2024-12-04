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

# Video-text-to-text

[[open-in-colab]]

Video-text-to-text models, also known as video language models or vision language models with video input, are language models that take a video input. These models can tackle various tasks, from video question answering to video captioning. 

These models have nearly the same architecture as [image-text-to-text](../image_text_to_text.md) models except for some changes to accept video data, since video data is essentially image frames with temporal dependencies. Some image-text-to-text models take in multiple images, but this alone is inadequate for a model to accept videos. Moreover, video-text-to-text models are often trained with all vision modalities. Each example might have videos, multiple videos, images and multiple images. Some of these models can also take interleaved inputs. For example, you can refer to a specific video inside a string of text by adding a video token in text like "What is happening in this video? `<video>`". 

In this guide, we provide a brief overview of video LMs and show how to use them with Transformers for inference.

To begin with, there are multiple types of video LMs:
- base models used for fine-tuning
- chat fine-tuned models for conversation
- instruction fine-tuned models

This guide focuses on inference with an instruction-tuned model, [llava-hf/llava-interleave-qwen-7b-hf](https://huggingface.co/llava-hf/llava-interleave-qwen-7b-hf) which can take in interleaved data. Alternatively, you can try [llava-interleave-qwen-0.5b-hf](https://huggingface.co/llava-hf/llava-interleave-qwen-0.5b-hf) if your hardware doesn't allow running a 7B model.

Let's begin installing the dependencies.

```bash
pip install -q transformers accelerate flash_attn 
```

Let's initialize the model and the processor. 

```python
from transformers import LlavaProcessor, LlavaForConditionalGeneration
import torch
model_id = "llava-hf/llava-interleave-qwen-0.5b-hf"

processor = LlavaProcessor.from_pretrained(model_id)

model = LlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16)
model.to("cuda") # can also be xpu, mps, npu etc. depending on your hardware accelerator
```

Some models directly consume the `<video>` token, and others accept `<image>` tokens equal to the number of sampled frames. This model handles videos in the latter fashion. We will write a simple utility to handle image tokens, and another utility to get a video from a url and sample frames from it. 

```python
import uuid
import requests
import cv2
from PIL import Image

def replace_video_with_images(text, frames):
  return text.replace("<video>", "<image>" * frames)

def sample_frames(url, num_frames):

    response = requests.get(url)
    path_id = str(uuid.uuid4())

    path = f"./{path_id}.mp4" 

    with open(path, "wb") as f:
      f.write(response.content)

    video = cv2.VideoCapture(path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = total_frames // num_frames
    frames = []
    for i in range(total_frames):
        ret, frame = video.read()
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not ret:
            continue
        if i % interval == 0:
            frames.append(pil_img)
    video.release()
    return frames[:num_frames]
```

Let's get our inputs. We will sample frames and concatenate them.

```python
video_1 = "https://huggingface.co/spaces/merve/llava-interleave/resolve/main/cats_1.mp4"
video_2 = "https://huggingface.co/spaces/merve/llava-interleave/resolve/main/cats_2.mp4"

video_1 = sample_frames(video_1, 6)
video_2 = sample_frames(video_2, 6)

videos = video_1 + video_2

videos

# [<PIL.Image.Image image mode=RGB size=1920x1080>,
# <PIL.Image.Image image mode=RGB size=1920x1080>,
# <PIL.Image.Image image mode=RGB size=1920x1080>, ...]
```

Both videos have cats.

<div class="container">
  <div class="video-container">
    <video width="400" controls>
      <source src="https://huggingface.co/spaces/merve/llava-interleave/resolve/main/cats_1.mp4" type="video/mp4">
    </video>
  </div>

  <div class="video-container">
    <video width="400" controls>
      <source src="https://huggingface.co/spaces/merve/llava-interleave/resolve/main/cats_2.mp4" type="video/mp4">
    </video>
  </div>
</div>

Now we can preprocess the inputs.

This model has a prompt template that looks like following. First, we'll put all the sampled frames into one list. Since we have eight frames in each video, we will insert 12 `<image>` tokens to our prompt. Add `assistant` at the end of the prompt to trigger the model to give answers. Then we can preprocess.

```python
user_prompt = "Are these two cats in these two videos doing the same thing?"
toks = "<image>" * 12
prompt = "<|im_start|>user"+ toks + f"\n{user_prompt}<|im_end|><|im_start|>assistant"
inputs = processor(text=prompt, images=videos, return_tensors="pt").to(model.device, model.dtype)
```

We can now call [`~GenerationMixin.generate`] for inference. The model outputs the question in our input and answer, so we only take the text after the prompt and `assistant` part from the model output. 

```python
output = model.generate(**inputs, max_new_tokens=100, do_sample=False)
print(processor.decode(output[0][2:], skip_special_tokens=True)[len(user_prompt)+10:])

# The first cat is shown in a relaxed state, with its eyes closed and a content expression, while the second cat is shown in a more active state, with its mouth open wide, possibly in a yawn or a vocalization.


```

And voila! 

To learn more about chat templates and token streaming for video-text-to-text models, refer to the [image-text-to-text](../image_text_to_text) task guide because these models work similarly.