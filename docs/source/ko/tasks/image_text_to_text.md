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

# Image-text-to-text[[image-text-to-text]]

[[open-in-colab]]

비전 언어 모델이라고도 불리는 이미지-텍스트-텍스트 모델은, 이미지를 입력 받는 언어 모델입니다. 이러한 모델은 시각적 질의 응답부터 이미지 분할에 이르기까지 다양한 작업을 처리할 수 있습니다. 이 작업은 이미지에서 텍스트로 변환하는 모델과 유사할뿐더러, 이미지 캡셔닝과 같은 중복된 사용 사례도 있습니다. 이미지-텍스트로 모델은 이미지 입력만 받아 특정 작업을 수행하는 경우가 많지만, 비전 언어 모델은 개방형 텍스트와 이미지 입력을 모두 받는 더욱 범용적인 모델입니다.

이 가이드에서는 비전 언어 모델에 대해 간략히 살펴보고, 이를 Transformers와 함께 사용하여 추론하는 방법을 보여줍니다.

먼저, 비전 언어 모델에는 다양한 종류가 있습니다:
- 미세 조정에 사용되는 기본 모델
- 대화를 위한 채팅 미세 조정 모델
- 명령어에 맞춘 미세 조정 모델

해당 가이드에서는 명령어에 맞춘 미세 조정 모델을 사용하여 추론하는 것을 중점적으로 다룹니다.

이제 필요한 라이브러리를 설치해 봅시다.

```bash
pip install -q transformers accelerate flash_attn 
```

모델과 프로세서를 초기화해 보겠습니다. 

```python
from transformers import AutoProcessor, Idefics2ForConditionalGeneration
import torch

device = torch.device("cuda")
model = Idefics2ForConditionalGeneration.from_pretrained(
    "HuggingFaceM4/idefics2-8b",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
).to(device)

processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b")
```

이 모델은 사용자에게 채팅 출력을 파싱하는데 도움이 되는 [chat template](./chat_templating)을 가지고 있습니다. 또한, 해당 모델은 단일 대화나 메시지에서 여러 이미지를 입력으로 받을 수도 있습니다. 이제 입력을 준비하겠습니다.

이미지 입력은 다음과 같습니다.

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.png" alt="그물 위에 앉아있는 두 고양이"/>
</div>

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg" alt="분홍색 꽃 위에 있는 벌"/>
</div>


```python
from PIL import Image
import requests

img_urls =["https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.png",
           "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"]
images = [Image.open(requests.get(img_urls[0], stream=True).raw),
          Image.open(requests.get(img_urls[1], stream=True).raw)]
```

아래는 채팅 템플릿의 예시입니다. 대화의 턴과 마지막 메시지를 템플릿 끝에 추가하여 입력으로 제공할 수 있습니다.

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "What do we see in this image?"}, # 이 이미지에서 무엇이 보이시나요?
        ]
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "In this image we can see two cats on the nets."}, # 이 이미지에서는 그물 위에 있는 두 마리의 고양이를 볼 수 있습니다. 
        ]
    },
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "And how about this image?"}, # 그렇다면 이 이미지에선 무엇이 보이시나요?
        ]
    },       
]
```

프로세서의 [`~ProcessorMixin.apply_chat_template`] 메소드를 호출하여 이미지 입력과 함께 출력을 출력을 전처리하겠습니다.

```python
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=[images[0], images[1]], return_tensors="pt").to(device)
```

이제 전처리된 입력을 모델에 넘겨주도록 하겠습니다.

```python
with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=500)
generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

print(generated_texts)
## ['User: What do we see in this image? \nAssistant: In this image we can see two cats on the nets. \nUser: And how about this image? \nAssistant: In this image we can see flowers, plants and insect.']

## ['User: 이 이미지에서 무엇이 보이시나요? \nAssistant: 이 이미지에서는 그물 위에 있는 두 마리의 고양이를 볼 수 있습니다. \nUser: 그렇다면 이 이미지에서는 무엇이 보이시나요? \nAssistant: 이 이미지에서는 꽃, 풀 그리고 곤충을 볼 수 있습니다.']
```

## Streaming [[streaming]]

더 나은 생성을 위해 [text streaming](./generation_strategies#streaming)을 사용할 수 있습니다. Transformers는 [`TextStreamer`] 또는 [`TextIteratorStreamer`] 클래스를 통해 스트리밍을 지원합니다. 여기서는 IDEFICS-8B와 함께 [`TextIteratorStreamer`]을 사용할 것입니다.

채팅 기록을 저장하고 새로운 사용자 입력을 받는 애플리케이션이 있다고 가정해봅시다. 평소대로 입력을 전처리 하고 [`TextIteratorStreamer`]를 초기화 하여 별도의 스레드에서 생성 과정을 처리합니다. 이를 통해 생성된 텍스트 토큰을 실시간으로 스트리밍할 수 있습니다. 생성과 관련된 인수는  [`TextIteratorStreamer`]에 전달할 수 있습니다.


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

    # add_generation_prompt=True 옵션을 사용하면 모델이 봇 응답을 생성하도록 만듭니다.
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

이제 우리가 생성한 `model_inference` 함수를 호출하고 값을 스트리밍해 봅시다.

```python
generator = model_inference(
    user_prompt="And what is in this image?",
    chat_history=messages,
    max_new_tokens=100,
    images=images
)

for value in generator:
  print(value)

# In
# In this
# In this image ...
```

## Fit models in smaller hardware [[fit-models-in-smaller-hardware]]

비전 언어 모델은 대부분 크기 때문에 더 작은 하드웨어에 맞추기 위해 최적화가 필요합니다. Transformers 여러 모델 양자화 라이브러리를 지원하며, 여기서는 [Quanto](./quantization/quanto#quanto)을 사용한 int8 양자화만을 다룹니다. int8 양자화를 사용하면 모든 가중치가 양자화될 경우 최대 75%의 메모리를 절감할 수 있습니다. 하지만 8비트는 CUDA에서 기본 정밀도가 아니기 때문에, 가중치가 실시간으로 양자화되면서 시간이 오래걸릴 수 있기 때문에, 이것은 완전한 해결책은 아닙니다.

먼저 필요한 라이브러리를 설치합니다.

```bash
pip install -U quanto bitsandbytes
```

모델을 로드하는 동안 양자화 하려면, 먼저 [`QuantoConfig`]를 생성해야 합니다. 그런 다음 평소와 같이 모델을 로드하되, 모델 초기화 시 `quantization_config`를 전달해야여 양자화를 수행해야 합니다.

```python
from transformers import Idefics2ForConditionalGeneration, AutoTokenizer, QuantoConfig

model_id = "HuggingFaceM4/idefics2-8b"
quantization_config = QuantoConfig(weights="int8")
quantized_model = Idefics2ForConditionalGeneration.from_pretrained(model_id, device_map="cuda", quantization_config=quantization_config)
```

이제 끝났습니다. 동일한 방식으로 아무런 변경 없이 모델을 사용할 수 있습니다. 

## Further Reading [[further-reading]]

다음은 이미지 기반 텍스트 변환 작업에 대한 몇 가지 추가 자료입니다.

- [Image-text-to-text task page](https://huggingface.co/tasks/image-text-to-text) 모델 종류, 사용 사례, 데이터셋 등의 내용을 다룹니다.
- [Vision Language Models Explained](https://huggingface.co/blog/vlms) 비전 언어 모델과 [TRL](https://huggingface.co/docs/trl/en/index)을 사용한 지도 학습 미세 조정에 관한 모든 내용을 다루는 블로그 포스트입니다.
