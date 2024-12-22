<!---
Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/transformers-logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/transformers-logo-light.svg">
    <img alt="Hugging Face Transformers Library" src="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/transformers-logo-light.svg" width="352" height="59" style="max-width: 100%;">
  </picture>
  <br/>
  <br/>
</p>

<p align="center">
    <a href="https://circleci.com/gh/huggingface/transformers"><img alt="Build" src="https://img.shields.io/circleci/build/github/huggingface/transformers/main"></a>
    <a href="https://github.com/huggingface/transformers/blob/main/LICENSE"><img alt="GitHub" src="https://img.shields.io/github/license/huggingface/transformers.svg?color=blue"></a>
    <a href="https://huggingface.co/docs/transformers/index"><img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/docs/transformers/index.svg?down_color=red&down_message=offline&up_message=online"></a>
    <a href="https://github.com/huggingface/transformers/releases"><img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/transformers.svg"></a>
    <a href="https://github.com/huggingface/transformers/blob/main/CODE_OF_CONDUCT.md"><img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg"></a>
    <a href="https://zenodo.org/badge/latestdoi/155220641"><img src="https://zenodo.org/badge/155220641.svg" alt="DOI"></a>
</p>

<h4 align="center">
    <p>
        <a href="https://github.com/huggingface/transformers/">English</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_zh-hans.md">简体中文</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_zh-hant.md">繁體中文</a> |
        <b>한국어</b> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_es.md">Español</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ja.md">日本語</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_hd.md">हिन्दी</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ru.md">Русский</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_pt-br.md">Рortuguês</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_te.md">తెలుగు</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_fr.md">Français</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_de.md">Deutsch</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_vi.md">Tiếng Việt</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ar.md">العربية</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ur.md">اردو</a> |
    </p>
</h4>

<h3 align="center">
    <p> Jax, Pytorch, TensorFlow를 위한 최첨단 머신러닝</p>
</h3>

<h3 align="center">
    <a href="https://hf.co/course"><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/course_banner.png"></a>
</h3>

🤗 Transformers는 텍스트, 비전, 오디오와 같은 다양한 분야에서 여러 과제를 수행하는 수천 개의 사전 학습된 모델을 제공합니다.

제공되는 모델을 통해 다음 과제를 수행할 수 있습니다.
- 📝 텍스트: 100개 이상의 언어들로, 텍스트 분류, 정보 추출, 질문 답변, 요약, 번역 및 문장 생성
- 🖼️ 이미지: 이미지 분류(Image Classification), 객체 탐지(Object Detection) 및 분할(Segmentation)
- 🗣️ 오디오: 음성 인식(Speech Recognition) 및 오디오 분류(Audio Classification)

Transformer의 모델은 표를 통한 질의응답(Table QA), 광학 문자 인식(Optical Character Recognition), 스캔 한 문서에서 정보 추출, 비디오 분류 및 시각적 질의응답과 같은 **여러 분야가 결합된** 과제 또한 수행할 수 있습니다.

🤗 Transformers는 이러한 사전학습 모델을 빠르게 다운로드해 특정 텍스트에 사용하고, 원하는 데이터로 fine-tuning해 커뮤니티나 우리의 [모델 허브](https://huggingface.co/models)에 공유할 수 있도록 API를 제공합니다. 또한, 모델 구조를 정의하는 각 파이썬 모듈은 완전히 독립적이여서 연구 실험을 위해 손쉽게 수정할 수 있습니다.

🤗 Transformers는 가장 유명한 3개의 딥러닝 라이브러리를 지원합니다. 이들은 서로 완벽히 연동됩니다 — [Jax](https://jax.readthedocs.io/en/latest/), [PyTorch](https://pytorch.org/), [TensorFlow](https://www.tensorflow.org/). 간단하게 이 라이브러리 중 하나로 모델을 학습하고, 또 다른 라이브러리로 추론을 위해 모델을 불러올 수 있습니다.

## 온라인 데모

대부분의 모델을 [모델 허브](https://huggingface.co/models) 페이지에서 바로 테스트해 볼 수 있습니다. 공개 및 비공개 모델을 위한 [비공개 모델 호스팅, 버전 관리, 추론 API](https://huggingface.co/pricing)도 제공합니다.

아래 몇 가지 예시가 있습니다:  

자연어 처리:   
- [BERT로 마스킹된 단어 완성하기](https://huggingface.co/google-bert/bert-base-uncased?text=Paris+is+the+%5BMASK%5D+of+France)
- [Electra를 이용한 개체명 인식](https://huggingface.co/dbmdz/electra-large-discriminator-finetuned-conll03-english?text=My+name+is+Sarah+and+I+live+in+London+city)
- [GPT-2로 텍스트 생성하기](https://huggingface.co/openai-community/gpt2?text=A+long+time+ago%2C+)
- [RoBERTa로 자연어 추론하기](https://huggingface.co/FacebookAI/roberta-large-mnli?text=The+dog+was+lost.+Nobody+lost+any+animal)
- [BART를 이용한 요약](https://huggingface.co/facebook/bart-large-cnn?text=The+tower+is+324+metres+%281%2C063+ft%29+tall%2C+about+the+same+height+as+an+81-storey+building%2C+and+the+tallest+structure+in+Paris.+Its+base+is+square%2C+measuring+125+metres+%28410+ft%29+on+each+side.+During+its+construction%2C+the+Eiffel+Tower+surpassed+the+Washington+Monument+to+become+the+tallest+man-made+structure+in+the+world%2C+a+title+it+held+for+41+years+until+the+Chrysler+Building+in+New+York+City+was+finished+in+1930.+It+was+the+first+structure+to+reach+a+height+of+300+metres.+Due+to+the+addition+of+a+broadcasting+aerial+at+the+top+of+the+tower+in+1957%2C+it+is+now+taller+than+the+Chrysler+Building+by+5.2+metres+%2817+ft%29.+Excluding+transmitters%2C+the+Eiffel+Tower+is+the+second+tallest+free-standing+structure+in+France+after+the+Millau+Viaduct)
- [DistilBERT를 이용한 질문 답변](https://huggingface.co/distilbert/distilbert-base-uncased-distilled-squad?text=Which+name+is+also+used+to+describe+the+Amazon+rainforest+in+English%3F&context=The+Amazon+rainforest+%28Portuguese%3A+Floresta+Amaz%C3%B4nica+or+Amaz%C3%B4nia%3B+Spanish%3A+Selva+Amaz%C3%B3nica%2C+Amazon%C3%ADa+or+usually+Amazonia%3B+French%3A+For%C3%AAt+amazonienne%3B+Dutch%3A+Amazoneregenwoud%29%2C+also+known+in+English+as+Amazonia+or+the+Amazon+Jungle%2C+is+a+moist+broadleaf+forest+that+covers+most+of+the+Amazon+basin+of+South+America.+This+basin+encompasses+7%2C000%2C000+square+kilometres+%282%2C700%2C000+sq+mi%29%2C+of+which+5%2C500%2C000+square+kilometres+%282%2C100%2C000+sq+mi%29+are+covered+by+the+rainforest.+This+region+includes+territory+belonging+to+nine+nations.+The+majority+of+the+forest+is+contained+within+Brazil%2C+with+60%25+of+the+rainforest%2C+followed+by+Peru+with+13%25%2C+Colombia+with+10%25%2C+and+with+minor+amounts+in+Venezuela%2C+Ecuador%2C+Bolivia%2C+Guyana%2C+Suriname+and+French+Guiana.+States+or+departments+in+four+nations+contain+%22Amazonas%22+in+their+names.+The+Amazon+represents+over+half+of+the+planet%27s+remaining+rainforests%2C+and+comprises+the+largest+and+most+biodiverse+tract+of+tropical+rainforest+in+the+world%2C+with+an+estimated+390+billion+individual+trees+divided+into+16%2C000+species)
- [T5로 번역하기](https://huggingface.co/google-t5/t5-base?text=My+name+is+Wolfgang+and+I+live+in+Berlin)

컴퓨터 비전:
- [ViT와 함께하는 이미지 분류](https://huggingface.co/google/vit-base-patch16-224)
- [DETR로 객체 탐지하기](https://huggingface.co/facebook/detr-resnet-50)
- [SegFormer로 의미적 분할(semantic segmentation)하기](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512)
- [Mask2Former로 판옵틱 분할(panoptic segmentation)하기](https://huggingface.co/facebook/mask2former-swin-large-coco-panoptic)
- [Depth Anything으로 깊이 추정(depth estimation)하기](https://huggingface.co/docs/transformers/main/model_doc/depth_anything)
- [VideoMAE와 함께하는 비디오 분류](https://huggingface.co/docs/transformers/model_doc/videomae)
- [OneFormer로 유니버설 분할(universal segmentation)하기](https://huggingface.co/shi-labs/oneformer_ade20k_dinat_large)

오디오:
- [Whisper와 함께하는 자동 음성 인식](https://huggingface.co/openai/whisper-large-v3)
- [Wav2Vec2로 키워드 검출(keyword spotting)하기](https://huggingface.co/superb/wav2vec2-base-superb-ks)
- [Audio Spectrogram Transformer로 오디오 분류하기](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593)

멀티 모달(Multimodal Task):
- [TAPAS로 표 안에서 질문 답변하기](https://huggingface.co/google/tapas-base-finetuned-wtq)
- [ViLT와 함께하는 시각적 질의응답](https://huggingface.co/dandelin/vilt-b32-finetuned-vqa)
- [LLaVa로 이미지에 설명 넣기](https://huggingface.co/llava-hf/llava-1.5-7b-hf)
- [SigLIP와 함께하는 제로 샷(zero-shot) 이미지 분류](https://huggingface.co/google/siglip-so400m-patch14-384)
- [LayoutLM으로 문서 안에서 질문 답변하기](https://huggingface.co/impira/layoutlm-document-qa)
- [X-CLIP과 함께하는 제로 샷(zero-shot) 비디오 분류](https://huggingface.co/docs/transformers/model_doc/xclip)
- [OWLv2로 진행하는 제로 샷(zero-shot) 객체 탐지](https://huggingface.co/docs/transformers/en/model_doc/owlv2)
- [CLIPSeg로 진행하는 제로 샷(zero-shot) 이미지 분할](https://huggingface.co/docs/transformers/model_doc/clipseg)
- [SAM과 함께하는 자동 마스크 생성](https://huggingface.co/docs/transformers/model_doc/sam)

**[Transformer와 글쓰기](https://transformer.huggingface.co)** 는 이 저장소의 텍스트 생성 능력에 관한 Hugging Face 팀의 공식 데모입니다.

## Transformers를 사용한 100개의 프로젝트

Transformers는 사전 학습된 모델들을 이용하는 도구를 넘어 Transformers와 함께 빌드 된 프로젝트 및 Hugging Face Hub를 위한 하나의 커뮤니티입니다. 우리는 Transformers를 통해 개발자, 연구자, 학생, 교수, 엔지니어 및 모든 이들이 꿈을 품은 프로젝트(Dream Project)를 빌드 할 수 있길 바랍니다.

Transformers에 달린 100,000개의 별을 축하하기 위해, 우리는 커뮤니티를 주목하고자 Transformers를 품고 빌드 된 100개의 어마어마한 프로젝트들을 선별하여 [awesome-transformers](https://github.com/huggingface/transformers/blob/main/awesome-transformers.md) 페이지에 나열하였습니다.

만일 소유한 혹은 사용하고 계신 프로젝트가 이 리스트에 꼭 등재되어야 한다고 믿으신다면, PR을 열고 추가하여 주세요!

## 조직 안에서 AI 사용에 대해 진지하게 고민 중이신가요? Hugging Face Enterprise Hub을 통해 더 빨리 구축해 보세요.

<a target="_blank" href="https://huggingface.co/enterprise">
    <img alt="Hugging Face Enterprise Hub" src="https://github.com/user-attachments/assets/247fb16d-d251-4583-96c4-d3d76dda4925">
</a><br>

## 퀵 투어

주어진 입력(텍스트, 이미지, 오디오, ...)에 바로 모델을 사용할 수 있도록, 우리는 `pipeline` API를 제공합니다. Pipeline은 사전학습 모델과 그 모델을 학습할 때 적용한 전처리 방식을 하나로 합칩니다. 다음은 긍정적인 텍스트와 부정적인 텍스트를 분류하기 위해 pipeline을 사용한 간단한 예시입니다:

```python
>>> from transformers import pipeline

# 감정 분석 파이프라인을 할당하세요
>>> classifier = pipeline('sentiment-analysis')
>>> classifier('We are very happy to introduce pipeline to the transformers repository.')
[{'label': 'POSITIVE', 'score': 0.9996980428695679}]
```

코드의 두 번째 줄은 pipeline이 사용하는 사전학습 모델을 다운로드하고 캐시로 저장합니다. 세 번째 줄에선 그 모델이 주어진 텍스트를 평가합니다. 여기서 모델은 99.97%의 확률로 텍스트가 긍정적이라고 평가했습니다.

자연어 처리(NLP) 뿐만 아니라 컴퓨터 비전, 발화(Speech) 과제들을 사전 학습된 `pipeline`으로 바로 수행할 수 있습니다. 예를 들어, 사진에서 손쉽게 객체들을 탐지할 수 있습니다.:

``` python
>>> import requests
>>> from PIL import Image
>>> from transformers import pipeline

# 귀여운 고양이가 있는 이미지를 다운로드하세요
>>> url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png"
>>> image_data = requests.get(url, stream=True).raw
>>> image = Image.open(image_data)

# 객체 감지를 위한 파이프라인을 할당하세요
>>> object_detector = pipeline('object-detection')
>>> object_detector(image)
[{'score': 0.9982201457023621,
  'label': 'remote',
  'box': {'xmin': 40, 'ymin': 70, 'xmax': 175, 'ymax': 117}},
 {'score': 0.9960021376609802,
  'label': 'remote',
  'box': {'xmin': 333, 'ymin': 72, 'xmax': 368, 'ymax': 187}},
 {'score': 0.9954745173454285,
  'label': 'couch',
  'box': {'xmin': 0, 'ymin': 1, 'xmax': 639, 'ymax': 473}},
 {'score': 0.9988006353378296,
  'label': 'cat',
  'box': {'xmin': 13, 'ymin': 52, 'xmax': 314, 'ymax': 470}},
 {'score': 0.9986783862113953,
  'label': 'cat',
  'box': {'xmin': 345, 'ymin': 23, 'xmax': 640, 'ymax': 368}}]
```
위와 같이, 우리는 이미지에서 탐지된 객체들에 대하여 객체를 감싸는 박스와 확률 리스트를 얻을 수 있습니다. 왼쪽이 원본 이미지이며 오른쪽은 해당 이미지에 탐지된 결과를 표시하였습니다.
<h3 align="center">
    <a><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png" width="400"></a>
    <a><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample_post_processed.png" width="400"></a>
</h3>

[이 튜토리얼](https://huggingface.co/docs/transformers/ko/task_summary)에서 `pipeline` API가 지원하는 다양한 과제를 확인할 수 있습니다.

코드 3줄로 원하는 과제에 맞게 사전학습 모델을 다운로드 받고 사용할 수 있습니다. 다음은 PyTorch 버전입니다:
```python
>>> from transformers import AutoTokenizer, AutoModel

>>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
>>> model = AutoModel.from_pretrained("google-bert/bert-base-uncased")

>>> inputs = tokenizer("Hello world!", return_tensors="pt")
>>> outputs = model(**inputs)
```
다음은 TensorFlow 버전입니다:
```python
>>> from transformers import AutoTokenizer, TFAutoModel

>>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
>>> model = TFAutoModel.from_pretrained("google-bert/bert-base-uncased")

>>> inputs = tokenizer("Hello world!", return_tensors="tf")
>>> outputs = model(**inputs)
```

토크나이저는 사전학습 모델의 모든 전처리를 책임집니다. 그리고 (위의 예시처럼) 1개의 스트링이나 리스트도 처리할 수 있습니다. 토크나이저는 딕셔너리를 반환하는데, 이는 다운스트림 코드에 사용하거나 언패킹 연산자 ** 를 이용해 모델에 바로 전달할 수도 있습니다.

모델 자체는 일반적으로 사용되는 [Pytorch `nn.Module`](https://pytorch.org/docs/stable/nn.html#torch.nn.Module)이나 [TensorFlow `tf.keras.Model`](https://www.tensorflow.org/api_docs/python/tf/keras/Model)입니다. [이 튜토리얼](https://huggingface.co/docs/transformers/ko/training)은 이러한 모델을 표준적인 PyTorch나 TensorFlow 학습 과정에서 사용하는 방법, 또는 새로운 데이터로 파인 튜닝하기 위해 `Trainer` API를 사용하는 방법을 설명해 줍니다.

## 왜 transformers를 사용해야 할까요?

1. 손쉽게 사용할 수 있는 최첨단 모델:
    - 자연어 이해(NLU)와 생성(NLG), 컴퓨터 비전, 오디오 과제에서 뛰어난 성능을 보입니다.
    - 교육자와 실무자에게 진입 장벽이 낮습니다.
    - 3개의 클래스만 배우면 바로 사용할 수 있습니다.
    - 하나의 API로 모든 사전학습 모델을 사용할 수 있습니다.

1. 더 적은 계산 비용, 더 적은 탄소 발자국:
    - 연구자들은 모델을 계속 다시 학습시키는 대신 학습된 모델을 공유할 수 있습니다.
    - 실무자들은 학습에 필요한 시간과 비용을 절약할 수 있습니다.
    - 모든 분야를 통틀어서 400,000개 이상의 사전 학습된 모델이 있는 수십 개의 아키텍처.

1. 모델의 각 생애주기에 적합한 프레임워크:
    - 코드 3줄로 최첨단 모델을 학습하세요.
    - 목적에 알맞게 모델을 TF2.0/Pytorch/Jax 프레임 워크 중 하나로 이동시키세요.
    - 학습, 평가, 공개 등 각 단계에 맞는 프레임워크를 원하는대로 선택하세요.

1. 필요한 대로 모델이나 예시를 커스터마이즈하세요:
    - 우리는 저자가 공개한 결과를 재현하기 위해 각 모델 구조의 예시를 제공합니다.
    - 모델 내부 구조는 가능한 일관적으로 공개되어 있습니다.
    - 빠른 실험을 위해 모델 파일은 라이브러리와 독립적으로 사용될 수 있습니다.

## 왜 transformers를 사용하지 말아야 할까요?

- 이 라이브러리는 신경망 블록을 만들기 위한 모듈이 아닙니다. 연구자들이 여러 파일을 살펴보지 않고 바로 각 모델을 사용할 수 있도록, 모델 파일 코드의 추상화 수준을 적정하게 유지했습니다.
- 학습 API는 모든 모델에 적용할 수 있도록 만들어지진 않았지만, 라이브러리가 제공하는 모델들에 적용할 수 있도록 최적화되었습니다. 일반적인 머신 러닝을 위해선, 다른 라이브러리를 사용하세요(예를 들면, [Accelerate](https://huggingface.co/docs/accelerate/index)).
- 가능한 많은 사용 예시를 보여드리고 싶어서, [예시 폴더](https://github.com/huggingface/transformers/tree/main/examples)의 스크립트를 준비했습니다. 이 스크립트들을 수정 없이 특정한 문제에 바로 적용하지 못할 수 있습니다. 필요에 맞게 일부 코드를 수정해야 할 수 있습니다.

## 설치

### pip로 설치하기

이 저장소는 Python 3.9+, Flax 0.4.1+, PyTorch 2.0+, TensorFlow 2.6+에서 테스트 되었습니다.

[가상 환경](https://docs.python.org/3/library/venv.html)에 🤗 Transformers를 설치하세요. Python 가상 환경에 익숙하지 않다면, [사용자 가이드](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)를 확인하세요.

우선, 사용할 Python 버전으로 가상 환경을 만들고 실행하세요.

그 다음, Flax, PyTorch, TensorFlow 중 적어도 하나는 설치해야 합니다.
플랫폼에 맞는 설치 명령어를 확인하기 위해 [TensorFlow 설치 페이지](https://www.tensorflow.org/install/), [PyTorch 설치 페이지](https://pytorch.org/get-started/locally/#start-locally), [Flax 설치 페이지](https://github.com/google/flax#quick-install)를 확인하세요.

이들 중 적어도 하나가 설치되었다면, 🤗 Transformers는 다음과 같이 pip을 이용해 설치할 수 있습니다:

```bash
pip install transformers
```

예시들을 체험해보고 싶거나, 최최최첨단 코드를 원하거나, 새로운 버전이 나올 때까지 기다릴 수 없다면 [라이브러리를 소스에서 바로 설치](https://huggingface.co/docs/transformers/ko/installation#install-from-source)하셔야 합니다.

### conda로 설치하기

🤗 Transformers는 다음과 같이 conda로 설치할 수 있습니다:

```shell script
conda install conda-forge::transformers
```

> **_노트:_** `huggingface` 채널에서 `transformers`를 설치하는 것은 사용이 중단되었습니다.

Flax, PyTorch, TensorFlow 설치 페이지에서 이들을 conda로 설치하는 방법을 확인하세요.

> **_노트:_** 윈도우 환경에서 캐싱의 이점을 위해 개발자 모드를 활성화할 수 있습니다. 만약 여러분에게 있어서 선택이 아닌 필수라면 [이 이슈](https://github.com/huggingface/huggingface_hub/issues/1062)를 통해 알려주세요.

## 모델 구조

**🤗 Transformers가 제공하는 [모든 모델 체크포인트](https://huggingface.co/models)** 는 huggingface.co [모델 허브](https://huggingface.co/models)에 완벽히 연동되어 있습니다. [개인](https://huggingface.co/users)과 [기관](https://huggingface.co/organizations)이 모델 허브에 직접 업로드할 수 있습니다.

현재 사용 가능한 모델 체크포인트의 개수: ![](https://img.shields.io/endpoint?url=https://huggingface.co/api/shields/models&color=brightgreen)

🤗 Transformers는 다음 모델들을 제공합니다: 각 모델의 요약은 [여기](https://huggingface.co/docs/transformers/ko/model_summary)서 확인하세요.

각 모델이 Flax, PyTorch, TensorFlow으로 구현되었는지 또는 🤗 Tokenizers 라이브러리가 지원하는 토크나이저를 사용하는지 확인하려면, [이 표](https://huggingface.co/docs/transformers/ko/index#supported-framework)를 확인하세요.

이 구현은 여러 데이터로 검증되었고 (예시 스크립트를 참고하세요) 오리지널 구현의 성능과 같아야 합니다. [도큐먼트](https://github.com/huggingface/transformers/tree/main/examples)의 Examples 섹션에서 성능에 대한 자세한 설명을 확인할 수 있습니다.

## 더 알아보기

| 섹션 | 설명 |
|-|-|
| [도큐먼트](https://huggingface.co/transformers/ko/) | 전체 API 도큐먼트와 튜토리얼 |
| [과제 요약](https://huggingface.co/docs/transformers/ko/task_summary) | 🤗 Transformers가 지원하는 과제들 |
| [전처리 튜토리얼](https://huggingface.co/docs/transformers/ko/preprocessing) | `Tokenizer` 클래스를 이용해 모델을 위한 데이터 준비하기 |
| [학습과 파인 튜닝](https://huggingface.co/docs/transformers/ko/training) | 🤗 Transformers가 제공하는 모델 PyTorch/TensorFlow 학습 과정과 `Trainer` API에서 사용하기 |
| [퀵 투어: 파인 튜닝/사용 스크립트](https://github.com/huggingface/transformers/tree/main/examples) | 다양한 과제에서 모델을 파인 튜닝하는 예시 스크립트 |
| [모델 공유 및 업로드](https://huggingface.co/docs/transformers/ko/model_sharing) | 커뮤니티에 파인 튜닝된 모델을 업로드 및 공유하기 |

## 인용

🤗 Transformers 라이브러리를 인용하고 싶다면, 이 [논문](https://www.aclweb.org/anthology/2020.emnlp-demos.6/)을 인용해 주세요:
```bibtex
@inproceedings{wolf-etal-2020-transformers,
    title = "Transformers: State-of-the-Art Natural Language Processing",
    author = "Thomas Wolf and Lysandre Debut and Victor Sanh and Julien Chaumond and Clement Delangue and Anthony Moi and Pierric Cistac and Tim Rault and Rémi Louf and Morgan Funtowicz and Joe Davison and Sam Shleifer and Patrick von Platen and Clara Ma and Yacine Jernite and Julien Plu and Canwen Xu and Teven Le Scao and Sylvain Gugger and Mariama Drame and Quentin Lhoest and Alexander M. Rush",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
    month = oct,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-demos.6",
    pages = "38--45"
}
```
