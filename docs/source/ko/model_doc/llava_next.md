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

# LLaVA-NeXT [[llava-next]]

## 개요 [[overview]]

LLaVA-NeXT 모델은 Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, Yong Jae Lee에 의해 제안된 [LLaVA-NeXT: Improved reasoning, OCR, and world knowledge](https://llava-vl.github.io/blog/2024-01-30-llava-next/)에서 소개되었습니다. LLaVA-1.6 이라고도 불리는 LLaVA-NeXT는 입력 이미지 해상도를 높이고 개선된 시각적 지시 튜닝 데이터 세트로 훈련하여 [LLaVa](llava)의 OCR 및 상식 추론 능력을 향상시켰습니다.

블로그의 소개는 다음과 같습니다:

*2023년 10월, 우리는 간단하고 효율적인 디자인과 함께 12개의 데이터 세트로 구성된 벤치마크에서 뛰어난 성능을 보이는 LLaVA-1.5를 공개했습니다. 공개 이후 대형 멀티모달 모델(LMM)의 데이터, 모델 및 기능에 대한 종합적인 연구의 기반이 되었으며, 다양한 새로운 응용 프로그램을 가능하게 했습니다.

오늘, 우리는 추론, OCR 및 세계 지식이 향상된 LLaVA-NeXT를 발표하게 되어 기쁩니다. LLaVA-NeXT는 여러 벤치마크에서 Gemini Pro를 능가합니다.

LLaVA-1.5와 비교하여, LLaVA-NeXT는 몇 가지 개선 사항이 있습니다:

입력 이미지 해상도를 4배 늘렸습니다. 이를 통해 더 많은 시각적 세부 사항을 파악할 수 있습니다. 세 가지 가로세로 비율을 제공하며, 최대 672x672, 336x1344, 1344x336 해상도를 지원합니다. 
개선된 시각적 지시 튜닝 데이터 조합으로 더 나은 시각적 추론 및 OCR 기능을 제공합니다. 
다양한 응용 프로그램을 다루는 더 많은 시나리오에서 더 나은 시각적 대화를 제공합니다. 더 향상된 세계 지식과 논리적 추론 능력도 갖추고 있습니다. 
SGLang을 통한 효율적인 배포와 추론이 가능합니다. 
성능 향상과 더불어, LLaVA-NeXT는 LLaVA-1.5의 간결한 디자인과 데이터 효율성을 유지합니다. LLaVA-1.5의 사전 학습된 커넥터를 재사용하며, 여전히 100만 개 미만의 시각적 지시 튜닝 샘플을 사용합니다. 가장 큰 34B 모델은 32개의 A100 GPU로 약 1일 만에 학습을 완료합니다.*

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/llava_next_overview.png"
alt="drawing" width="600"/>

<small> LLaVa-NeXT는 입력 이미지의 다양한 패치를 인코딩하여 더 높은 입력 해상도를 통합합니다. <a href="https://arxiv.org/abs/2310.03744">원본 논문</a>에서 가져왔습니다. </small>

이 모델은 [nielsr](https://huggingface.co/nielsr)가 기여하였습니다.
원본 코드는 [여기](https://github.com/haotian-liu/LLaVA/tree/main)에서 찾을 수 있습니다.

## 사용 팁 [[usage-tips]]

- 배치 생성을 계산할 때 더 정확한 결과를 얻기 위해 사용자들에게 `padding_side="left"`를 사용할 것을 권장합니다. 생성하기 전에 단순히 `processor.tokenizer.padding_side = "left"`를 호출하면 됩니다.

<Tip warning={true}>

- Llava-Next는 이미지에 대해 서로 다른 수의 패치를 사용하므로, 입력을 처리할 때의 패딩 외에도 모델링 코드 내에서 입력을 패딩해야 합니다. 모델이 `eval()` 모드일 경우 기본 설정은 "left-padding"이며, 그렇지 않으면 "right-padding"입니다.

</Tip>


- 각 체크포인트는 사용된 대형 언어 모델(LLM)에 따라 특정 프롬프트 형식을 사용하여 훈련되었음을 유의해야 합니다. 당신은 프롬프트를 올바르게 형식화하기 위해 프로세서의 `apply_chat_template`를 사용할 수 있습니다. 이를 위해서는 대화 기록을 구성해야 하며, 단순한 문자열을 전달하면 프롬프트가 형식화되지 않습니다. 채팅 템플릿의 대화 기록의 각 메시지는 "role"과 "content" 키를 가진 딕셔너리입니다. "content"는 "text"와 "image" 모달리티에 대한 딕셔너리의 리스트여야 합니다. 아래는 이를 수행하는 방법과 각 체크포인트에서 허용되는 형식의 목록입니다.

우리는 [llava-v1.6-mistral-7b-hf](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf)와 텍스트 및 이미지의 대화 기록을 사용할 것입니다. 각 content 필드는 다음과 같이 딕셔너리의 리스트여야 합니다:

```python
from transformers import LlavaNextProcessor

processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "What’s shown in this image?"},
        ],
    },
    {
        "role": "assistant",
        "content": [{"type": "text", "text": "This image shows a red stop sign."},]
    },
    {

        "role": "user",
        "content": [
            {"type": "text", "text": "Describe the image in more details."},
        ],
    },
]

text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

# 템플릿은 단순히 프롬프트의 형식을 맞춰줄 뿐입니다. 여전히 프롬프트를 토큰화하고 이미지의 픽셀 값을 직접 얻어야 합니다.

print(text_prompt)
>>> "[INST] <image>\nWhat's shown in this image? [/INST] This image shows a red stop sign. [INST] Describe the image in more details. [/INST]"
```

- 만약 직접 대화 프롬프트를 구성하고 싶다면, 아래는 가능한 형식의 목록입니다.
[llava-v1.6-mistral-7b-hf](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf) 에는 다음 형식이 필요합니다:
```bash
"[INST] <image>\nWhat is shown in this image? [/INST]"
```

[llava-v1.6-vicuna-7b-hf](https://huggingface.co/llava-hf/llava-v1.6-vicuna-7b-hf) 와 [llava-v1.6-vicuna-13b-hf](https://huggingface.co/llava-hf/llava-v1.6-vicuna-13b-hf) 에는 다음 형식이 필요합니다:
```bash
"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\nWhat is shown in this image? ASSISTANT:"
```

[llava-v1.6-34b-hf](https://huggingface.co/llava-hf/llava-v1.6-34b-hf) 에는 다음 형식이 필요합니다:
```bash
"<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\nWhat is shown in this image?<|im_end|><|im_start|>assistant\n"
```

[llama3-llava-next-8b-hf](https://huggingface.co/llava-hf/llava-next-8b-hf) 에는 다음 형식이 필요합니다:

```bash
"<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.<|eot_id|><|start_header_id|><|start_header_id|>user<|end_header_id|>\n\n<image>\nWhat is shown in this image?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
```

[llava-next-72b-hf](https://huggingface.co/llava-hf/llava-next-72b-hf) and [llava-next-110b-hf](https://huggingface.co/llava-hf/llava-next-110b-hf) 에는 다음 형식이 필요합니다:

```bash
"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\nWhat is shown in this image?<|im_end|>\n<|im_start|>assistant\n"
```

## 사용 예시 [[usage-example]]

### 단일 이미지 추론 [[single-image-inference]]

다음은 모델을 로드하고 반정밀도로 추론을 수행하는 방법입니다(`torch.float16`):

```python
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests

processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True) 
model.to("cuda:0")

# 적절한 프롬프트 템플릿을 사용하여 이미지와 텍스트 프롬프트를 준비합니다
url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
image = Image.open(requests.get(url, stream=True).raw)

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "What is shown in this image?"},
        ],
    },
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
inputs = processor(prompt, image, return_tensors="pt").to("cuda:0")

# 자기회귀적으로 프롬프트를 완성합니다
output = model.generate(**inputs, max_new_tokens=100)

print(processor.decode(output[0], skip_special_tokens=True))
```

### 다중 이미지 추론 [[multi-image-inference]]

LLaVa-Next는 입력으로 여러 이미지를 사용하여 추론을 수행할 수 있습니다. 이미지들은 동일한 프롬프트에 속하거나 (배치 추론 시) 서로 다른 프롬프트에 속할 수 있습니다. 다음은 그 방법입니다:

```python
import requests
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaNextForConditionalGeneration

# 반정밀도로 모델을 로드합니다
model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, device_map="auto")
processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

# 세 장의 다른 이미지를 얻습니다
url = "https://www.ilankelman.org/stopsigns/australia.jpg"
image_stop = Image.open(requests.get(url, stream=True).raw)

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image_cats = Image.open(requests.get(url, stream=True).raw)

url = "https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/snowman.jpg"
image_snowman = Image.open(requests.get(url, stream=True).raw)

# 두 개의 프롬프트로 구성된 배치를 준비합니다. 첫 번째 프롬프트는 여러 턴의 대화이고, 두 번째는 그렇지 않습니다
conversation_1 = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "What is shown in this image?"},
            ],
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "There is a red stop sign in the image."},
            ],
    },
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "What about this image? How many cats do you see?"},
            ],
    },
]

conversation_2 = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "What is shown in this image?"},
            ],
    },
]

prompt_1 = processor.apply_chat_template(conversation_1, add_generation_prompt=True)
prompt_2 = processor.apply_chat_template(conversation_2, add_generation_prompt=True)
prompts = [prompt_1, prompt_2]

# 텍스트 프롬프트에서 사용해야 할 순서대로 이미지를 간단히 입력할 수 있습니다
# 각 "<image>" 토큰은 하나의 이미지를 사용하며, 다음 "<image>" 토큰에는 그다음 이미지가 할당됩니다
inputs = processor(text=prompts, images=[image_stop, image_cats, image_snowman], padding=True, return_tensors="pt").to(model.device)

# 텍스트를 생성합니다
generate_ids = model.generate(**inputs, max_new_tokens=30)
processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
```

## 모델 최적화 [[model-optimization]]

### Bitsandbytes를 사용한 양자화[[quantization-using-bitsandbytes]]

모델은 8비트 또는 4비트로 로드할 수 있어, 원래 모델의 성능을 유지하면서도 메모리 요구 사항을 크게 줄일 수 있습니다. 먼저 `pip install bitsandbytes`를 사용하여 bitsandbytes를 설치하고, CUDA 호환 GPU 장치에 접근할 수 있는지 확인하세요. 위의 코드 스니펫을 다음과 같이 변경하면 됩니다:

```python
from transformers import LlavaNextForConditionalGeneration, BitsAndBytesConfig

# 모델을 어떻게 양자화할지 지정합니다
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", quantization_config=quantization_config, device_map="auto")
```

### Flash-Attention 2를 사용하여 생성 속도를 더욱 높이기 [[use-flash-attention-2-to-further-speed-up-generation]]

먼저 flash-attn을 설치해야 합니다. 해당 패키지 설치에 대해서는 [Flash Attention의 원본 리포지토리](https://github.com/Dao-AILab/flash-attention)를 참고하세요. 위의 코드 스니펫을 다음과 같이 변경하면 됩니다:

```python
from transformers import LlavaNextForConditionalGeneration

model = LlavaNextForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True,
    use_flash_attention_2=True
).to(0)
```

## LlavaNextConfig [[transformers.LlavaNextConfig]]

[[autodoc]] LlavaNextConfig

## LlavaNextImageProcessor [[transformers.LlavaNextImageProcessor]]

[[autodoc]] LlavaNextImageProcessor
    - preprocess

## LlavaNextProcessor [[transformers.LlavaNextProcessor]]

[[autodoc]] LlavaNextProcessor

## LlavaNextForConditionalGeneration [[transformers.LlavaNextForConditionalGeneration]]

[[autodoc]] LlavaNextForConditionalGeneration
    - forward
