# AltCLIP

## 개요[[overview]]

AltCLIP 모델은 Zhongzhi Chen, Guang Liu, Bo-Wen Zhang, Fulong Ye, Qinghong Yang, Ledell Wu의 [AltCLIP: Altering the Language Encoder in CLIP for Extended Language Capabilities](https://arxiv.org/abs/2211.06679v2) 논문에서 제안되었습니다. AltCLIP(CLIP의 언어 인코더를 변경하여 언어 기능 확장)은 다양한 이미지-텍스트 및 텍스트-텍스트 쌍으로 훈련된 신경망입니다. CLIP의 텍스트 인코더를 사전 훈련된 다국어 텍스트 인코더 XLM-R로 교체하여, 거의 모든 작업에서 CLIP과 유사한 성능을 얻을 수 있었으며, 원래 CLIP의 다국어 이해와 같은 기능도 확장되었습니다.

논문의 초록은 다음과 같습니다:

*본 연구에서는 강력한 이중 언어 멀티모달 표현 모델을 훈련하는 개념적으로 간단하고 효과적인 방법을 제시합니다. OpenAI에서 출시한 사전 훈련된 멀티모달 표현 모델 CLIP에서 시작하여, 그 텍스트 인코더를 사전 훈련된 다국어 텍스트 인코더 XLM-R로 교체하고, 교사 학습과 대조 학습으로 구성된 2단계 훈련 스키마를 통해 언어와 이미지 표현을 정렬했습니다. 우리는 광범위한 작업 평가를 통해 우리의 방법을 검증했습니다. ImageNet-CN, Flicker30k-CN, COCO-CN을 포함한 여러 작업에서 새로운 최고 성능을 달성했으며, 거의 모든 작업에서 CLIP과 유사한 성능을 얻었습니다. 이는 CLIP의 텍스트 인코더를 단순히 변경하여 다국어 이해와 같은 확장 기능을 얻을 수 있음을 시사합니다.*

이 모델은 [jongjyh](https://huggingface.co/jongjyh)에 의해 기여되었습니다.

## 사용 팁과 예제[[usage-tips-and-example]]

AltCLIP의 사용법은 CLIP과 매우 유사하며, 차이점은 텍스트 인코더에 있습니다. 일반적인 어텐션 대신 양방향 어텐션을 사용하며, XLM-R의 [CLS] 토큰을 사용하여 텍스트 임베딩을 나타냅니다.

AltCLIP은 멀티모달 비전 및 언어 모델입니다. 이미지와 텍스트 간의 유사성 계산 및 제로샷 이미지 분류에 사용할 수 있습니다. AltCLIP은 ViT와 같은 트랜스포머를 사용하여 시각적 특징을 얻고, 양방향 언어 모델을 사용하여 텍스트 특징을 얻습니다. 이후 텍스트와 시각적 특징 모두 동일한 차원의 잠재 공간으로 투사됩니다. 투사된 이미지와 텍스트 특징 간의 내적을 유사도 점수로 사용합니다.

이미지를 트랜스포머 인코더에 입력하기 위해, 각 이미지를 일정한 크기의 겹치지 않는 패치 시퀀스로 분할한 뒤, 이를 선형 임베딩합니다. 전체 이미지를 나타내기 위해 [CLS] 토큰이 추가됩니다. 저자들은 절대 위치 임베딩도 추가하여 결과 벡터 시퀀스를 표준 트랜스포머 인코더에 입력합니다. [`CLIPImageProcessor`]는 모델을 위해 이미지를 크기 조정하고 정규화하는 데 사용할 수 있습니다.

[`AltCLIPProcessor`]는 [`CLIPImageProcessor`]와 [`XLMRobertaTokenizer`]를 하나의 인스턴스로 묶어 텍스트를 인코딩하고 이미지를 준비합니다. 다음 예제는 [`AltCLIPProcessor`]와 [`AltCLIPModel`]을 사용하여 이미지와 텍스트 간의 유사성 점수를 얻는 방법을 보여줍니다.
```python
>>> from PIL import Image
>>> import requests

>>> from transformers import AltCLIPModel, AltCLIPProcessor

>>> model = AltCLIPModel.from_pretrained("BAAI/AltCLIP")
>>> processor = AltCLIPProcessor.from_pretrained("BAAI/AltCLIP")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

>>> outputs = model(**inputs)
>>> logits_per_image = outputs.logits_per_image  # 이미지-텍스트 유사도 점수
>>> probs = logits_per_image.softmax(dim=1)  # 라벨 마다 확률을 얻기 위해 softmax 적용
```
<Tip>

이 모델은 `CLIPModel`을 기반으로 하므로, 원래 CLIP처럼 사용할 수 있습니다.

</Tip>

## AltCLIPConfig

[[autodoc]] AltCLIPConfig
    - from_text_vision_configs

## AltCLIPTextConfig

[[autodoc]] AltCLIPTextConfig

## AltCLIPVisionConfig

[[autodoc]] AltCLIPVisionConfig

## AltCLIPProcessor

[[autodoc]] AltCLIPProcessor

## AltCLIPModel

[[autodoc]] AltCLIPModel
    - forward
    - get_text_features
    - get_image_features

## AltCLIPTextModel

[[autodoc]] AltCLIPTextModel
    - forward

## AltCLIPVisionModel

[[autodoc]] AltCLIPVisionModel
    - forward
