<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# 둘러보기 [[quick-tour]]

[[open-in-colab]]

🤗 Transformers를 시작해보세요! 개발해본 적이 없더라도 쉽게 읽을 수 있도록 쓰인 이 글은 [`pipeline`](./main_classes/pipelines)을 사용하여 추론하고, 사전학습된 모델과 전처리기를 [AutoClass](./model_doc/auto)로 로드하고, PyTorch 또는 TensorFlow로 모델을 빠르게 학습시키는 방법을 소개해 드릴 것입니다. 본 가이드에서 소개되는 개념을 (특히 초보자의 관점으로) 더 친절하게 접하고 싶다면, 튜토리얼이나 [코스](https://huggingface.co/course/chapter1/1)를 참조하기를 권장합니다.

시작하기 전에 필요한 라이브러리가 모두 설치되어 있는지 확인하세요:

```bash
!pip install transformers datasets evaluate accelerate
```

또한 선호하는 머신 러닝 프레임워크를 설치해야 합니다:

<frameworkcontent>
<pt>

```bash
pip install torch
```
</pt>
<tf>

```bash
pip install tensorflow
```
</tf>
</frameworkcontent>

## 파이프라인 [[pipeline]]

<Youtube id="tiZFewofSLM"/>

[`pipeline`](./main_classes/pipelines)은 사전 훈련된 모델로 추론하기에 가장 쉽고 빠른 방법입니다. [`pipeline`]은 여러 모달리티에서 다양한 과업을 쉽게 처리할 수 있으며, 아래 표에 표시된 몇 가지 과업을 기본적으로 지원합니다:

<Tip>

사용 가능한 작업의 전체 목록은 [Pipelines API 참조](./main_classes/pipelines)를 확인하세요.

</Tip>

| **태스크**      | **설명**                                                             | **모달리티**     | **파이프라인 ID**                             |
|-----------------|----------------------------------------------------------------------|------------------|-----------------------------------------------|
| 텍스트 분류      | 텍스트에 알맞은 레이블 붙이기                                         | 자연어 처리(NLP) | pipeline(task="sentiment-analysis")           |
| 텍스트 생성      | 주어진 문자열 입력과 이어지는 텍스트 생성하기                       | 자연어 처리(NLP) | pipeline(task="text-generation")              |
| 개체명 인식      | 문자열의 각 토큰마다 알맞은 레이블 붙이기 (인물, 조직, 장소 등등)     | 자연어 처리(NLP) | pipeline(task="ner")                          |
| 질의응답         | 주어진 문맥과 질문에 따라 올바른 대답하기                           | 자연어 처리(NLP) | pipeline(task="question-answering")           |
| 빈칸 채우기      | 문자열의 빈칸에 알맞은 토큰 맞추기                                  | 자연어 처리(NLP) | pipeline(task="fill-mask")                    |
| 요약             | 텍스트나 문서를 요약하기                                            | 자연어 처리(NLP) | pipeline(task="summarization")                |
| 번역             | 텍스트를 한 언어에서 다른 언어로 번역하기                           | 자연어 처리(NLP) | pipeline(task="translation")                  |
| 이미지 분류      | 이미지에 알맞은 레이블 붙이기                                         | 컴퓨터 비전(CV)  | pipeline(task="image-classification")         |
| 이미지 분할      | 이미지의 픽셀마다 레이블 붙이기(시맨틱, 파놉틱 및 인스턴스 분할 포함) | 컴퓨터 비전(CV)  | pipeline(task="image-segmentation")           |
| 객체 탐지        | 이미지 속 객체의 경계 상자를 그리고 클래스를 예측하기               | 컴퓨터 비전(CV)  | pipeline(task="object-detection")             |
| 오디오 분류      | 오디오 파일에 알맞은 레이블 붙이기                                    | 오디오           | pipeline(task="audio-classification")         |
| 자동 음성 인식   | 오디오 파일 속 음성을 텍스트로 바꾸기                               | 오디오           | pipeline(task="automatic-speech-recognition") |
| 시각 질의응답    | 주어진 이미지와 질문에 대해 올바르게 대답하기                       | 멀티모달         | pipeline(task="vqa")                          |
| 문서 질의응답    | 주어진 문서와 질문에 대해 올바르게 대답하기                         | 멀티모달         | pipeline(task="document-question-answering")  |
| 이미지 캡션 달기 | 주어진 이미지의 캡션 생성하기                                       | 멀티모달         | pipeline(task="image-to-text")                |

먼저 [`pipeline`]의 인스턴스를 생성하고 사용할 작업을 지정합니다. 이 가이드에서는 감정 분석을 위해 [`pipeline`]을 사용하는 예제를 보여드리겠습니다:

```py
>>> from transformers import pipeline

>>> classifier = pipeline("sentiment-analysis")
```

[`pipeline`]은 감정 분석을 위한 [사전 훈련된 모델](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english)과 토크나이저를 자동으로 다운로드하고 캐시합니다. 이제 `classifier`를 대상 텍스트에 사용할 수 있습니다:

```py
>>> classifier("We are very happy to show you the 🤗 Transformers library.")
[{'label': 'POSITIVE', 'score': 0.9998}]
```

만약 입력이 여러 개 있는 경우, 입력을 리스트로 [`pipeline`]에 전달하여, 사전 훈련된 모델의 출력을 딕셔너리로 이루어진 리스트 형태로 받을 수 있습니다:

```py
>>> results = classifier(["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."])
>>> for result in results:
...     print(f"label: {result['label']}, with score: {round(result['score'], 4)}")
label: POSITIVE, with score: 0.9998
label: NEGATIVE, with score: 0.5309
```

[`pipeline`]은 주어진 과업에 관계없이 데이터셋 전부를 순회할 수도 있습니다. 이 예제에서는 자동 음성 인식을 과업으로 선택해 보겠습니다:

```py
>>> import torch
>>> from transformers import pipeline

>>> speech_recognizer = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")
```

데이터셋을 로드할 차례입니다. (자세한 내용은 🤗 Datasets [시작하기](https://huggingface.co/docs/datasets/quickstart#audio)을 참조하세요) 여기에서는 [MInDS-14](https://huggingface.co/datasets/PolyAI/minds14) 데이터셋을 로드하겠습니다:

```py
>>> from datasets import load_dataset, Audio

>>> dataset = load_dataset("PolyAI/minds14", name="en-US", split="train")  # doctest: +IGNORE_RESULT
```

데이터셋의 샘플링 레이트가 기존 모델인 [`facebook/wav2vec2-base-960h`](https://huggingface.co/facebook/wav2vec2-base-960h)의 훈련 당시 샘플링 레이트와 일치하는지 확인해야 합니다:

```py
>>> dataset = dataset.cast_column("audio", Audio(sampling_rate=speech_recognizer.feature_extractor.sampling_rate))
```

`"audio"` 열을 호출하면 자동으로 오디오 파일을 가져와서 리샘플링합니다. 첫 4개 샘플에서 원시 웨이브폼 배열을 추출하고 파이프라인에 리스트로 전달하세요:

```py
>>> result = speech_recognizer(dataset[:4]["audio"])
>>> print([d["text"] for d in result])
['I WOULD LIKE TO SET UP A JOINT ACCOUNT WITH MY PARTNER HOW DO I PROCEED WITH DOING THAT', "FONDERING HOW I'D SET UP A JOIN TO HELL T WITH MY WIFE AND WHERE THE AP MIGHT BE", "I I'D LIKE TOY SET UP A JOINT ACCOUNT WITH MY PARTNER I'M NOT SEEING THE OPTION TO DO IT ON THE APSO I CALLED IN TO GET SOME HELP CAN I JUST DO IT OVER THE PHONE WITH YOU AND GIVE YOU THE INFORMATION OR SHOULD I DO IT IN THE AP AN I'M MISSING SOMETHING UQUETTE HAD PREFERRED TO JUST DO IT OVER THE PHONE OF POSSIBLE THINGS", 'HOW DO I FURN A JOINA COUT']
```

음성이나 비전과 같이 입력이 큰 대규모 데이터셋의 경우, 모든 입력을 메모리에 로드하려면 리스트 대신 제너레이터 형태로 전달해야 합니다. 자세한 내용은 [Pipelines API 참조](./main_classes/pipelines)를 확인하세요.

### 파이프라인에서 다른 모델과 토크나이저 사용하기 [[use-another-model-and-tokenizer-in-the-pipeline]]

[`pipeline`]은 [Hub](https://huggingface.co/models)의 모든 모델을 사용할 수 있기 때문에, [`pipeline`]을 다른 용도에 맞게 쉽게 수정할 수 있습니다. 예를 들어, 프랑스어 텍스트를 처리할 수 있는 모델을 사용하기 위해선 Hub의 태그를 사용하여 적절한 모델을 필터링하면 됩니다. 필터링된 결과의 상위 항목으로는 프랑스어 텍스트에 사용할 수 있는 다국어 [BERT 모델](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment)이 반환됩니다:

```py
>>> model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
```

<frameworkcontent>
<pt>
[`AutoModelForSequenceClassification`]과 [`AutoTokenizer`]를 사용하여 사전 훈련된 모델과 관련된 토크나이저를 로드하세요 (다음 섹션에서 [`AutoClass`]에 대해 더 자세히 알아보겠습니다):

```py
>>> from transformers import AutoTokenizer, AutoModelForSequenceClassification

>>> model = AutoModelForSequenceClassification.from_pretrained(model_name)
>>> tokenizer = AutoTokenizer.from_pretrained(model_name)
```
</pt>
<tf>
[`TFAutoModelForSequenceClassification`]과 [`AutoTokenizer`]를 사용하여 사전 훈련된 모델과 관련된 토크나이저를 로드하세요 (다음 섹션에서 [`TFAutoClass`]에 대해 더 자세히 알아보겠습니다):

```py
>>> from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

>>> model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
>>> tokenizer = AutoTokenizer.from_pretrained(model_name)
```
</tf>
</frameworkcontent>

[`pipeline`]에서 모델과 토크나이저를 지정하면, 이제 `classifier`를 프랑스어 텍스트에 적용할 수 있습니다:

```py
>>> classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
>>> classifier("Nous sommes très heureux de vous présenter la bibliothèque 🤗 Transformers.")
[{'label': '5 stars', 'score': 0.7273}]
```

마땅한 모델을 찾을 수 없는 경우 데이터를 기반으로 사전 훈련된 모델을 미세조정해야 합니다. 미세조정 방법에 대한 자세한 내용은 [미세조정 튜토리얼](./training)을 참조하세요. 사전 훈련된 모델을 미세조정한 후에는 모델을 Hub의 커뮤니티와 공유하여 머신러닝 민주화에 기여해주세요! 🤗

## AutoClass [[autoclass]]

<Youtube id="AhChOFRegn4"/>

[`AutoModelForSequenceClassification`]과 [`AutoTokenizer`] 클래스는 위에서 다룬 [`pipeline`]의 기능을 구현하는 데 사용됩니다. [AutoClass](./model_doc/auto)는 사전 훈련된 모델의 아키텍처를 이름이나 경로에서 자동으로 가져오는 '바로가기'입니다. 과업에 적합한 `AutoClass`를 선택하고 해당 전처리 클래스를 선택하기만 하면 됩니다.

이전 섹션의 예제로 돌아가서 [`pipeline`]의 결과를 `AutoClass`를 활용해 복제하는 방법을 살펴보겠습니다.

### AutoTokenizer [[autotokenizer]]

토크나이저는 텍스트를 모델의 입력으로 사용하기 위해 숫자 배열 형태로 전처리하는 역할을 담당합니다. 토큰화 과정에는 단어를 어디에서 끊을지, 어느 수준까지 나눌지와 같은 여러 규칙들이 있습니다 (토큰화에 대한 자세한 내용은 [토크나이저 요약](./tokenizer_summary)을 참조하세요). 가장 중요한 점은 모델이 사전 훈련된 모델과 동일한 토큰화 규칙을 사용하도록 동일한 모델 이름으로 토크나이저를 인스턴스화해야 한다는 것입니다.

[`AutoTokenizer`]로 토크나이저를 로드하세요:

```py
>>> from transformers import AutoTokenizer

>>> model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
>>> tokenizer = AutoTokenizer.from_pretrained(model_name)
```

텍스트를 토크나이저에 전달하세요:

```py
>>> encoding = tokenizer("We are very happy to show you the 🤗 Transformers library.")
>>> print(encoding)
{'input_ids': [101, 11312, 10320, 12495, 19308, 10114, 11391, 10855, 10103, 100, 58263, 13299, 119, 102],
 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
```

토크나이저는 다음을 포함한 딕셔너리를 반환합니다:

* [input_ids](./glossary#input-ids): 토큰의 숫자 표현.
* [attention_mask](.glossary#attention-mask): 어떤 토큰에 주의를 기울여야 하는지를 나타냅니다.

토크나이저는 입력을 리스트 형태로도 받을 수 있으며, 텍스트를 패딩하고 잘라내어 일정한 길이의 묶음을 반환할 수도 있습니다:

<frameworkcontent>
<pt>

```py
>>> pt_batch = tokenizer(
...     ["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."],
...     padding=True,
...     truncation=True,
...     max_length=512,
...     return_tensors="pt",
... )
```
</pt>
<tf>

```py
>>> tf_batch = tokenizer(
...     ["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."],
...     padding=True,
...     truncation=True,
...     max_length=512,
...     return_tensors="tf",
... )
```
</tf>
</frameworkcontent>

<Tip>

[전처리](./preprocessing) 튜토리얼을 참조하시면 토큰화에 대한 자세한 설명과 함께 이미지, 오디오와 멀티모달 입력을 전처리하기 위한 [`AutoImageProcessor`]와 [`AutoFeatureExtractor`], [`AutoProcessor`]의 사용방법도 알 수 있습니다.

</Tip>

### AutoModel [[automodel]]

<frameworkcontent>
<pt>
🤗 Transformers는 사전 훈련된 인스턴스를 간단하고 통합된 방법으로 로드할 수 있습니다. 즉, [`AutoTokenizer`]처럼 [`AutoModel`]을 로드할 수 있습니다. 유일한 차이점은 과업에 알맞은 [`AutoModel`]을 선택해야 한다는 점입니다. 텍스트 (또는 시퀀스) 분류의 경우 [`AutoModelForSequenceClassification`]을 로드해야 합니다:

```py
>>> from transformers import AutoModelForSequenceClassification

>>> model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
>>> pt_model = AutoModelForSequenceClassification.from_pretrained(model_name)
```

<Tip>

[`AutoModel`] 클래스에서 지원하는 과업에 대해서는 [과업 요약](./task_summary)을 참조하세요.

</Tip>

이제 전처리된 입력 묶음을 직접 모델에 전달해야 합니다. 아래처럼 `**`를 앞에 붙여 딕셔너리를 풀어주면 됩니다:

```py
>>> pt_outputs = pt_model(**pt_batch)
```

모델의 최종 활성화 함수 출력은 `logits` 속성에 담겨있습니다. `logits`에 softmax 함수를 적용하여 확률을 얻을 수 있습니다:

```py
>>> from torch import nn

>>> pt_predictions = nn.functional.softmax(pt_outputs.logits, dim=-1)
>>> print(pt_predictions)
tensor([[0.0021, 0.0018, 0.0115, 0.2121, 0.7725],
        [0.2084, 0.1826, 0.1969, 0.1755, 0.2365]], grad_fn=<SoftmaxBackward0>)
```
</pt>
<tf>
🤗 Transformers는 사전 훈련된 인스턴스를 간단하고 통합된 방법으로 로드할 수 있습니다. 즉, [`AutoTokenizer`]처럼 [`TFAutoModel`]을 로드할 수 있습니다. 유일한 차이점은 과업에 알맞은 [`TFAutoModel`]을 선택해야 한다는 점입니다. 텍스트 (또는 시퀀스) 분류의 경우 [`TFAutoModelForSequenceClassification`]을 로드해야 합니다:

```py
>>> from transformers import TFAutoModelForSequenceClassification

>>> model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
>>> tf_model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
```

<Tip>

[`AutoModel`] 클래스에서 지원하는 과업에 대해서는 [과업 요약](./task_summary)을 참조하세요.

</Tip>

이제 전처리된 입력 묶음을 직접 모델에 전달해야 합니다. 아래처럼 그대로 텐서를 전달하면 됩니다:

```py
>>> tf_outputs = tf_model(tf_batch)
```

모델의 최종 활성화 함수 출력은 `logits` 속성에 담겨있습니다. `logits`에 softmax 함수를 적용하여 확률을 얻을 수 있습니다:

```py
>>> import tensorflow as tf

>>> tf_predictions = tf.nn.softmax(tf_outputs.logits, axis=-1)
>>> tf_predictions  # doctest: +IGNORE_RESULT
```
</tf>
</frameworkcontent>

<Tip>

모든 🤗 Transformers 모델(PyTorch 또는 TensorFlow)은 (softmax와 같은) 최종 활성화 함수 *이전에* 텐서를 출력합니다. 왜냐하면 최종 활성화 함수의 출력은 종종 손실 함수 출력과 결합되기 때문입니다. 모델 출력은 특수한 데이터 클래스이므로 IDE에서 자동 완성됩니다. 모델 출력은 튜플이나 딕셔너리처럼 동작하며 (정수, 슬라이스 또는 문자열로 인덱싱 가능), None인 속성은 무시됩니다.

</Tip>

### 모델 저장하기 [[save-a-model]]

<frameworkcontent>
<pt>
미세조정된 모델을 토크나이저와 함께 저장하려면 [`PreTrainedModel.save_pretrained`]를 사용하세요:

```py
>>> pt_save_directory = "./pt_save_pretrained"
>>> tokenizer.save_pretrained(pt_save_directory)  # doctest: +IGNORE_RESULT
>>> pt_model.save_pretrained(pt_save_directory)
```

모델을 다시 사용하려면 [`PreTrainedModel.from_pretrained`]로 모델을 다시 로드하세요:

```py
>>> pt_model = AutoModelForSequenceClassification.from_pretrained("./pt_save_pretrained")
```
</pt>
<tf>
미세조정된 모델을 토크나이저와 함께 저장하려면 [`TFPreTrainedModel.save_pretrained`]를 사용하세요:

```py
>>> tf_save_directory = "./tf_save_pretrained"
>>> tokenizer.save_pretrained(tf_save_directory)  # doctest: +IGNORE_RESULT
>>> tf_model.save_pretrained(tf_save_directory)
```

모델을 다시 사용하려면 [`TFPreTrainedModel.from_pretrained`]로 모델을 다시 로드하세요:

```py
>>> tf_model = TFAutoModelForSequenceClassification.from_pretrained("./tf_save_pretrained")
```
</tf>
</frameworkcontent>

🤗 Transformers의 멋진 기능 중 하나는 모델을 PyTorch 또는 TensorFlow 모델로 저장해뒀다가 다른 프레임워크로 다시 로드할 수 있는 점입니다. `from_pt` 또는 `from_tf` 매개변수를 사용하여 모델을 한 프레임워크에서 다른 프레임워크로 변환할 수 있습니다:

<frameworkcontent>
<pt>

```py
>>> from transformers import AutoModel

>>> tokenizer = AutoTokenizer.from_pretrained(pt_save_directory)
>>> pt_model = AutoModelForSequenceClassification.from_pretrained(pt_save_directory, from_pt=True)
```
</pt>
<tf>

```py
>>> from transformers import TFAutoModel

>>> tokenizer = AutoTokenizer.from_pretrained(tf_save_directory)
>>> tf_model = TFAutoModelForSequenceClassification.from_pretrained(tf_save_directory, from_tf=True)
```
</tf>
</frameworkcontent>

## 커스텀 모델 구축하기 [[custom-model-builds]]

모델의 구성 클래스를 수정하여 모델의 구조를 바꿀 수 있습니다. (은닉층이나 어텐션 헤드의 수와 같은) 모델의 속성은 구성에서 지정되기 때문입니다. 커스텀 구성 클래스로 모델을 만들면 처음부터 시작해야 합니다. 모델 속성은 무작위로 초기화되므로 의미 있는 결과를 얻으려면 먼저 모델을 훈련시켜야 합니다.

먼저 [`AutoConfig`]를 가져오고 수정하고 싶은 사전학습된 모델을 로드하세요. [`AutoConfig.from_pretrained`] 내부에서 (어텐션 헤드 수와 같이) 변경하려는 속성를 지정할 수 있습니다:

```py
>>> from transformers import AutoConfig

>>> my_config = AutoConfig.from_pretrained("distilbert/distilbert-base-uncased", n_heads=12)
```

<frameworkcontent>
<pt>
[`AutoModel.from_config`]를 사용하여 바꾼 구성대로 모델을 생성하세요:

```py
>>> from transformers import AutoModel

>>> my_model = AutoModel.from_config(my_config)
```
</pt>
<tf>
[`TFAutoModel.from_config`]를 사용하여 바꾼 구성대로 모델을 생성하세요:

```py
>>> from transformers import TFAutoModel

>>> my_model = TFAutoModel.from_config(my_config)
```
</tf>
</frameworkcontent>

커스텀 구성에 대한 자세한 내용은 [커스텀 아키텍처 만들기](./create_a_model) 가이드를 확인하세요.

## Trainer - PyTorch에 최적화된 훈련 루프 [[trainer-a-pytorch-optimized-training-loop]]

모든 모델은 [`torch.nn.Module`](https://pytorch.org/docs/stable/nn.html#torch.nn.Module)이므로 일반적인 훈련 루프에서 사용할 수 있습니다. 직접 훈련 루프를 작성할 수도 있지만, 🤗 Transformers는 PyTorch를 위한 [`Trainer`] 클래스를 제공합니다. 이 클래스에는 기본 훈련 루프가 포함되어 있으며 분산 훈련, 혼합 정밀도 등과 같은 기능을 추가로 제공합니다.

과업에 따라 다르지만 일반적으로 [`Trainer`]에 다음 매개변수를 전달합니다:

1. [`PreTrainedModel`] 또는 [`torch.nn.Module`](https://pytorch.org/docs/stable/nn.html#torch.nn.Module)로 시작합니다:

   ```py
   >>> from transformers import AutoModelForSequenceClassification

   >>> model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased")
   ```

2. [`TrainingArguments`]는 학습률, 배치 크기, 훈련할 에포크 수와 같은 모델 하이퍼파라미터를 포함합니다. 훈련 인자를 지정하지 않으면 기본값이 사용됩니다:

   ```py
   >>> from transformers import TrainingArguments

   >>> training_args = TrainingArguments(
   ...     output_dir="path/to/save/folder/",
   ...     learning_rate=2e-5,
   ...     per_device_train_batch_size=8,
   ...     per_device_eval_batch_size=8,
   ...     num_train_epochs=2,
   ... )
   ```

3. 토크나이저, 이미지 프로세서, 특징 추출기(feature extractor) 또는 프로세서와 전처리 클래스를 로드하세요:

   ```py
   >>> from transformers import AutoTokenizer

   >>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
   ```

4. 데이터셋을 로드하세요:

   ```py
   >>> from datasets import load_dataset

   >>> dataset = load_dataset("rotten_tomatoes")  # doctest: +IGNORE_RESULT
   ```

5. 데이터셋을 토큰화하는 함수를 생성하세요:

   ```py
   >>> def tokenize_dataset(dataset):
   ...     return tokenizer(dataset["text"])
   ```

   그리고 [`~datasets.Dataset.map`]로 데이터셋 전체에 적용하세요:

   ```py
   >>> dataset = dataset.map(tokenize_dataset, batched=True)
   ```

6. [`DataCollatorWithPadding`]을 사용하여 데이터셋의 표본 묶음을 만드세요:

   ```py
   >>> from transformers import DataCollatorWithPadding

   >>> data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
   ```

이제 위의 모든 클래스를 [`Trainer`]로 모으세요:

```py
>>> from transformers import Trainer

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=dataset["train"],
...     eval_dataset=dataset["test"],
...     processing_class=tokenizer,
...     data_collator=data_collator,
... )  # doctest: +SKIP
```

준비가 되었으면 [`~Trainer.train`]을 호출하여 훈련을 시작하세요:

```py
>>> trainer.train()  # doctest: +SKIP
```

<Tip>

번역이나 요약과 같이 시퀀스-시퀀스 모델을 사용하는 과업에는 [`Seq2SeqTrainer`] 및 [`Seq2SeqTrainingArguments`] 클래스를 사용하세요.

</Tip>

[`Trainer`] 내의 메서드를 서브클래스화하여 훈련 루프를 바꿀 수도 있습니다. 이러면 손실 함수, 옵티마이저, 스케줄러와 같은 기능 또한 바꿀 수 있게 됩니다. 변경 가능한 메소드에 대해서는 [`Trainer`] 문서를 참고하세요.

훈련 루프를 수정하는 다른 방법은 [Callbacks](./main_classes/callback)를 사용하는 것입니다. Callbacks로 다른 라이브러리와 통합하고, 훈련 루프를 체크하여 진행 상황을 보고받거나, 훈련을 조기에 중단할 수 있습니다. Callbacks은 훈련 루프 자체를 바꾸지는 않습니다. 손실 함수와 같은 것을 바꾸려면 [`Trainer`]를 서브클래스화해야 합니다.

## TensorFlow로 훈련시키기 [[train-with-tensorflow]]

모든 모델은 [`tf.keras.Model`](https://www.tensorflow.org/api_docs/python/tf/keras/Model)이므로 [Keras](https://keras.io/) API를 통해 TensorFlow에서 훈련시킬 수 있습니다. 🤗 Transformers는 데이터셋을 쉽게 `tf.data.Dataset` 형태로 쉽게 로드할 수 있는 [`~TFPreTrainedModel.prepare_tf_dataset`] 메소드를 제공하기 때문에, Keras의 [`compile`](https://keras.io/api/models/model_training_apis/#compile-method) 및 [`fit`](https://keras.io/api/models/model_training_apis/#fit-method) 메소드로 바로 훈련을 시작할 수 있습니다.

1. [`TFPreTrainedModel`] 또는 [`tf.keras.Model`](https://www.tensorflow.org/api_docs/python/tf/keras/Model)로 시작합니다:

   ```py
   >>> from transformers import TFAutoModelForSequenceClassification

   >>> model = TFAutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased")
   ```

2. 토크나이저, 이미지 프로세서, 특징 추출기(feature extractor) 또는 프로세서와 같은 전처리 클래스를 로드하세요:

   ```py
   >>> from transformers import AutoTokenizer

   >>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
   ```

3. 데이터셋을 토큰화하는 함수를 생성하세요:

   ```py
   >>> def tokenize_dataset(dataset):
   ...     return tokenizer(dataset["text"])  # doctest: +SKIP
   ```

4. [`~datasets.Dataset.map`]을 사용하여 전체 데이터셋에 토큰화 함수를 적용하고, 데이터셋과 토크나이저를 [`~TFPreTrainedModel.prepare_tf_dataset`]에 전달하세요. 배치 크기를 변경하거나 데이터셋을 섞을 수도 있습니다:

   ```py
   >>> dataset = dataset.map(tokenize_dataset)  # doctest: +SKIP
   >>> tf_dataset = model.prepare_tf_dataset(
   ...     dataset["train"], batch_size=16, shuffle=True, tokenizer=tokenizer
   ... )  # doctest: +SKIP
   ```

5. 준비되었으면 `compile` 및 `fit`를 호출하여 훈련을 시작하세요. 🤗 Transformers의 모든 모델은 과업과 관련된 기본 손실 함수를 가지고 있으므로 명시적으로 지정하지 않아도 됩니다:

   ```py
   >>> from tensorflow.keras.optimizers import Adam

   >>> model.compile(optimizer=Adam(3e-5))  # No loss argument!
   >>> model.fit(tf_dataset)  # doctest: +SKIP
   ```

## 다음 단계는 무엇인가요? [[whats-next]]

🤗 Transformers 둘러보기를 모두 읽으셨다면, 가이드를 살펴보고 더 구체적인 것을 수행하는 방법을 알아보세요. 이를테면 커스텀 모델 구축하는 방법, 과업에 알맞게 모델을 미세조정하는 방법, 스크립트로 모델 훈련하는 방법 등이 있습니다. 🤗 Transformers 핵심 개념에 대해 더 알아보려면 커피 한 잔 들고 개념 가이드를 살펴보세요!
