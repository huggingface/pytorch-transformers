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

# 영상 기반 텍스트 변환(Video-text-to-text) [[videotexttotext]]

[[open-in-colab]]

영상-텍스트-텍스트 모델, 영상 언어 모델 또는 영상 입력을 사용하는 비전 언어 모델로도 알려져 있으며, 영상 입력을 받는 언어 모델입니다. 이러한 모델은 영상에 대한 질의응답에서 영상 캡셔닝에 이르기까지 다양한 작업을 수행할 수 있습니다.

이 모델들은 [이미지 기반 텍스트 변환(image-text-to-text)](../image_text_to_text.md) 모델과 거의 동일한 아키텍처를 가지지만 영상 데이터를 수용하기 위한 일부 변경 사항이 있습니다. 영상 데이터는 기본적으로 시간적 의존성을 가진 이미지 프레임이기 때문입니다. 일부 이미지 기반 텍스트 변환(image-text-to-text) 모델은 여러 이미지를 입력으로 받을 수 있지만, 이것만으로는 영상을 수용하기에 충분하지 않습니다. 또한 영상 기반 텍스트 변환 모델은 종종 모든 비전 모달리티로 학습됩니다. 각 예시에는 영상, 여러 영상, 이미지 및 여러 이미지가 포함될 수 있습니다. 일부 모델은 교차 입력을 받을 수도 있습니다. 예를 들어, 텍스트 내에 영상 토큰을 추가하여 특정 영상을 참조할 수 있습니다. 예: "이 영상에서 무슨 일이 벌어지고 있나요? `<video>`".

이 가이드에서는 영상 언어 모델에 대한 간략한 개요를 제공하고 🤗 Transformers로 추론하는 방법을 보여줍니다.

먼저, 영상 언어 모델에는 여러 유형이 있습니다:
- 미세 조정에 사용되는 기본 모델
- 대화를 위한 채팅 미세 조정 모델
- 명령어에 맞춘 미세 조정 모델

이 가이드는 명령어에 맞춘 모델인 [llava-hf/llava-interleave-qwen-7b-hf](https://huggingface.co/llava-hf/llava-interleave-qwen-7b-hf)를 사용하여 추론하는 것에 중점을 둡니다. 이 모델은 교차된 데이터를 받을 수 있습니다. 하드웨어가 7B 모델을 실행할 수 없다면 [llava-interleave-qwen-0.5b-hf](https://huggingface.co/llava-hf/llava-interleave-qwen-0.5b-hf)를 시도할 수 있습니다.

이제 종속성을 설치해 봅시다.

```bash
pip install -q transformers accelerate flash_attn 
```

모델과 프로세서를 초기화해 보겠습니다.

```python
from transformers import LlavaProcessor, LlavaForConditionalGeneration
import torch
model_id = "llava-hf/llava-interleave-qwen-0.5b-hf"

processor = LlavaProcessor.from_pretrained(model_id)

model = LlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16)
model.to("cuda")
```

일부 모델은 `<video>` 토큰을 직접 사용하고, 다른 모델은 샘플링된 프레임 수에 맞는 `<image>` 토큰을 받습니다. 이 모델은 후자의 방식으로 비디오를 처리합니다. 우리는 이미지 토큰을 처리하는 간단한 도구(utility)와 URL에서 비디오를 가져와 프레임을 샘플링하는 또 다른 도구(utility)를 작성할 것입니다.

```python
import uuid
import requests
import cv2

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
    return frames
```

이제 입력을 받아 보겠습니다. 우리는 프레임을 샘플링하고 그것을 연결할 것입니다.

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

두 영상 모두 고양이를 포함하고 있습니다.

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

이제 입력을 전처리할 수 있습니다.

이 모델에는 다음과 같은 프롬프트 양식이 있습니다. 먼저, 샘플링된 모든 프레임을 하나의 리스트에 넣을 것입니다. 각 영상에는 8개의 프레임이 있으므로 프롬프트에 12개의 `<image>` 토큰을 삽입할 것입니다. 프롬프트 끝에 `assistant`를 추가하여 모델이 응답을 하도록 합니다. 그런 다음 전처리를 진행할 수 있습니다.

```python
user_prompt = "Are these two cats in these two videos doing the same thing?" # 두 영상 속 고양이들이 같은 행동을 하고 있나요?
toks = "<image>" * 12
prompt = "<|im_start|>user"+ toks + f"\n{user_prompt}<|im_end|><|im_start|>assistant"
inputs = processor(prompt, images=videos).to(model.device, model.dtype)
```

이제 [`~GenerationMixin.generate`]를 호출하여 추론할 수 있습니다. 모델은 입력의 질문과 답변을 출력하므로, 프롬프트와 `assistant` 부분 이후의 텍스트만 모델 출력에서 가져옵니다.

```python
output = model.generate(**inputs, max_new_tokens=100, do_sample=False)
print(processor.decode(output[0][2:], skip_special_tokens=True)[len(user_prompt)+10:])

# The first cat is shown in a relaxed state, with its eyes closed and a content expression, while the second cat is shown in a more active state, with its mouth open wide, possibly in a yawn or a vocalization. (첫 번째 고양이는 눈을 감고 만족스러운 표정을 지으며 편안한 상태를 보여주고 있으며, 두 번째 고양이는 입을 크게 벌리고 있어 하품을 하거나 소리를 내는 것처럼 더 활발한 상태를 보여줍니다.)


```

그리고 보세요!

영상 기반 텍스트 변환 모델을 위한 채팅 양식과 토큰 스트리밍에 대해 더 배우고 싶다면, 이 모델들이 유사하게 작동하므로 [이미지 기반 텍스트 변환(image-text-to-text)](../image_text_to_text) 작업 가이드를 참조하세요.