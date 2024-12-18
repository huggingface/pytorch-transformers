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

# 오디오 분류[[audio_classification]]

[[open-in-colab]]

<Youtube id="KWwzcmG98Ds"/>

오디오 분류는 텍스트와 마찬가지로 입력 데이터에 클래스 레이블 출력을 할당합니다. 유일한 차이점은 텍스트 입력 대신 원시 오디오 파형이 있다는 것입니다. 오디오 분류의 실제 적용 분야에는 화자의 의도 파악, 언어 분류, 소리로 동물 종을 식별하는 것 등이 있습니다.

이 문서에서 방법을 알아보겠습니다:

1. [MInDS-14](https://huggingface.co/datasets/PolyAI/minds14) 데이터 세트를 [Wav2Vec2](https://huggingface.co/facebook/wav2vec2-base)로 미세 조정하여 화자의 의도를 분류합니다.
2. 추론에 미세 조정된 모델을 사용하세요.

<Tip>

이 작업과 호환되는 모든 아키텍처와 체크포인트를 보려면 [작업 페이지](https://huggingface.co/tasks/audio-classification)를 확인하는 것이 좋습니다.

</Tip>

시작하기 전에 필요한 라이브러리가 모두 설치되어 있는지 확인하세요:

```bash
pip install transformers datasets evaluate
```

모델을 업로드하고 커뮤니티와 공유할 수 있도록 허깅페이스 계정에 로그인하는 것이 좋습니다. 메시지가 표시되면 토큰을 입력하여 로그인합니다:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## MInDS-14 데이터셋 불러오기[[load_minds_14_dataset]]

먼저 🤗 Datasets 라이브러리에서 MinDS-14 데이터 세트를 가져옵니다:

```py
>>> from datasets import load_dataset, Audio

>>> minds = load_dataset("PolyAI/minds14", name="en-US", split="train")
```

데이터 세트의 `train` 분할을 [`~datasets.Dataset.train_test_split`] 메소드를 사용하여 더 작은 훈련 및 테스트 집합으로 분할합니다. 이렇게 하면 전체 데이터 세트에 더 많은 시간을 소비하기 전에 모든 것이 작동하는지 실험하고 확인할 수 있습니다.

```py
>>> minds = minds.train_test_split(test_size=0.2)
```

이제 데이터 집합을 살펴볼게요:

```py
>>> minds
DatasetDict({
    train: Dataset({
        features: ['path', 'audio', 'transcription', 'english_transcription', 'intent_class', 'lang_id'],
        num_rows: 450
    })
    test: Dataset({
        features: ['path', 'audio', 'transcription', 'english_transcription', 'intent_class', 'lang_id'],
        num_rows: 113
    })
})
```

데이터 세트에는 `lang_id` 및 `english_transcription`과 같은 유용한 정보가 많이 포함되어 있지만 이 가이드에서는 `audio` 및 `intent_class`에 중점을 둘 것입니다. 다른 열은 [`~datasets.Dataset.remove_columns`] 메소드를 사용하여 제거합니다:

```py
>>> minds = minds.remove_columns(["path", "transcription", "english_transcription", "lang_id"])
```

예시를 살펴보겠습니다:

```py
>>> minds["train"][0]
{'audio': {'array': array([ 0.        ,  0.        ,  0.        , ..., -0.00048828,
         -0.00024414, -0.00024414], dtype=float32),
  'path': '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~APP_ERROR/602b9a5fbb1e6d0fbce91f52.wav',
  'sampling_rate': 8000},
 'intent_class': 2}
```

두 개의 필드가 있습니다:

- `audio`: 오디오 파일을 가져오고 리샘플링하기 위해 호출해야 하는 음성 신호의 1차원 `배열`입니다.
- `intent_class`: 화자의 의도에 대한 클래스 ID를 나타냅니다.

모델이 레이블 ID에서 레이블 이름을 쉽게 가져올 수 있도록 레이블 이름을 정수로 매핑하는 사전을 만들거나 그 반대로 매핑하는 사전을 만듭니다:

```py
>>> labels = minds["train"].features["intent_class"].names
>>> label2id, id2label = dict(), dict()
>>> for i, label in enumerate(labels):
...     label2id[label] = str(i)
...     id2label[str(i)] = label
```

이제 레이블 ID를 레이블 이름으로 변환할 수 있습니다:

```py
>>> id2label[str(2)]
'app_error'
```

## 전처리[[preprocess]]

다음 단계는 오디오 신호를 처리하기 위해 Wav2Vec2 특징 추출기를 가져오는 것입니다:

```py
>>> from transformers import AutoFeatureExtractor

>>> feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
```

MinDS-14 데이터 세트의 샘플링 속도는 8khz이므로(이 정보는 [데이터세트 카드](https://huggingface.co/datasets/PolyAI/minds14)에서 확인할 수 있습니다), 사전 훈련된 Wav2Vec2 모델을 사용하려면 데이터 세트를 16kHz로 리샘플링해야 합니다:

```py
>>> minds = minds.cast_column("audio", Audio(sampling_rate=16_000))
>>> minds["train"][0]
{'audio': {'array': array([ 2.2098757e-05,  4.6582241e-05, -2.2803260e-05, ...,
         -2.8419291e-04, -2.3305941e-04, -1.1425107e-04], dtype=float32),
  'path': '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~APP_ERROR/602b9a5fbb1e6d0fbce91f52.wav',
  'sampling_rate': 16000},
 'intent_class': 2}
```

이제 전처리 함수를 만듭니다:

1. 가져올 `오디오` 열을 호출하고 필요한 경우 오디오 파일을 리샘플링합니다.
2. 오디오 파일의 샘플링 속도가 모델에 사전 훈련된 오디오 데이터의 샘플링 속도와 일치하는지 확인합니다. 이 정보는 Wav2Vec2 [모델 카드](https://huggingface.co/facebook/wav2vec2-base)에서 확인할 수 있습니다.
3. 긴 입력이 잘리지 않고 일괄 처리되도록 최대 입력 길이를 설정합니다.

```py
>>> def preprocess_function(examples):
...     audio_arrays = [x["array"] for x in examples["audio"]]
...     inputs = feature_extractor(
...         audio_arrays, sampling_rate=feature_extractor.sampling_rate, max_length=16000, truncation=True
...     )
...     return inputs
```

전체 데이터 세트에 전처리 기능을 적용하려면 🤗 Datasets [`~datasets.Dataset.map`] 함수를 사용합니다. `batched=True`를 설정하여 데이터 집합의 여러 요소를 한 번에 처리하면 `map`의 속도를 높일 수 있습니다. 필요하지 않은 열을 제거하고 `intent_class`의 이름을 모델이 예상하는 이름인 `label`로 변경합니다:

```py
>>> encoded_minds = minds.map(preprocess_function, remove_columns="audio", batched=True)
>>> encoded_minds = encoded_minds.rename_column("intent_class", "label")
```

## 평가하기[[evaluate]]

훈련 중에 메트릭을 포함하면 모델의 성능을 평가하는 데 도움이 되는 경우가 많습니다. 🤗 [Evaluate](https://huggingface.co/docs/evaluate/index) 라이브러리를 사용하여 평가 방법을 빠르게 가져올 수 있습니다. 이 작업에서는 [accuracy(정확도)](https://huggingface.co/spaces/evaluate-metric/accuracy) 메트릭을 가져옵니다(메트릭을 가져오고 계산하는 방법에 대한 자세한 내용은 🤗 Evalutate [빠른 둘러보기](https://huggingface.co/docs/evaluate/a_quick_tour) 참조하세요):

```py
>>> import evaluate

>>> accuracy = evaluate.load("accuracy")
```

그런 다음 예측과 레이블을 [`~evaluate.EvaluationModule.compute`]에 전달하여 정확도를 계산하는 함수를 만듭니다:

```py
>>> import numpy as np


>>> def compute_metrics(eval_pred):
...     predictions = np.argmax(eval_pred.predictions, axis=1)
...     return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)
```

이제 `compute_metrics` 함수를 사용할 준비가 되었으며, 트레이닝을 설정할 때 이 함수를 사용합니다.

## 훈련[[train]]

<frameworkcontent>
<pt>
<Tip>

[`Trainer`]로 모델을 미세 조정하는 데 익숙하지 않다면 기본 튜토리얼 [여기](../training#train-with-pytorch-trainer)을 살펴보세요!

</Tip>

이제 모델 훈련을 시작할 준비가 되었습니다! [`AutoModelForAudioClassification`]을 이용해서 Wav2Vec2를 불러옵니다. 예상되는 레이블 수와 레이블 매핑을 지정합니다:

```py
>>> from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer

>>> num_labels = len(id2label)
>>> model = AutoModelForAudioClassification.from_pretrained(
...     "facebook/wav2vec2-base", num_labels=num_labels, label2id=label2id, id2label=id2label
... )
```

이제 세 단계만 남았습니다:

1. 훈련 하이퍼파라미터를 [`TrainingArguments`]에 정의합니다. 유일한 필수 매개변수는 모델을 저장할 위치를 지정하는 `output_dir`입니다. `push_to_hub = True`를 설정하여 이 모델을 허브로 푸시합니다(모델을 업로드하려면 허깅 페이스에 로그인해야 합니다). 각 에폭이 끝날 때마다 [`Trainer`]가 정확도를 평가하고 훈련 체크포인트를 저장합니다.
2. 모델, 데이터 세트, 토크나이저, 데이터 콜레이터, `compute_metrics` 함수와 함께 훈련 인자를 [`Trainer`]에 전달합니다.
3. [`~Trainer.train`]을 호출하여 모델을 미세 조정합니다.


```py
>>> training_args = TrainingArguments(
...     output_dir="my_awesome_mind_model",
...     eval_strategy="epoch",
...     save_strategy="epoch",
...     learning_rate=3e-5,
...     per_device_train_batch_size=32,
...     gradient_accumulation_steps=4,
...     per_device_eval_batch_size=32,
...     num_train_epochs=10,
...     warmup_ratio=0.1,
...     logging_steps=10,
...     load_best_model_at_end=True,
...     metric_for_best_model="accuracy",
...     push_to_hub=True,
... )

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=encoded_minds["train"],
...     eval_dataset=encoded_minds["test"],
...     processing_class=feature_extractor,
...     compute_metrics=compute_metrics,
... )

>>> trainer.train()
```

훈련이 완료되면 모든 사람이 모델을 사용할 수 있도록 [`~transformers.Trainer.push_to_hub`] 메소드를 사용하여 모델을 허브에 공유하세요:

```py
>>> trainer.push_to_hub()
```
</pt>
</frameworkcontent>

<Tip>

For a more in-depth example of how to finetune a model for audio classification, take a look at the corresponding [PyTorch notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/audio_classification.ipynb).

</Tip>

## 추론[[inference]]

이제 모델을 미세 조정했으니 추론에 사용할 수 있습니다!

추론을 실행할 오디오 파일을 가져옵니다. 필요한 경우 오디오 파일의 샘플링 속도를 모델의 샘플링 속도와 일치하도록 리샘플링하는 것을 잊지 마세요!

```py
>>> from datasets import load_dataset, Audio

>>> dataset = load_dataset("PolyAI/minds14", name="en-US", split="train")
>>> dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
>>> sampling_rate = dataset.features["audio"].sampling_rate
>>> audio_file = dataset[0]["audio"]["path"]
```

추론을 위해 미세 조정한 모델을 시험해 보는 가장 간단한 방법은 [`pipeline`]에서 사용하는 것입니다. 모델을 사용하여 오디오 분류를 위한 `pipeline`을 인스턴스화하고 오디오 파일을 전달합니다:

```py
>>> from transformers import pipeline

>>> classifier = pipeline("audio-classification", model="stevhliu/my_awesome_minds_model")
>>> classifier(audio_file)
[
    {'score': 0.09766869246959686, 'label': 'cash_deposit'},
    {'score': 0.07998877018690109, 'label': 'app_error'},
    {'score': 0.0781070664525032, 'label': 'joint_account'},
    {'score': 0.07667109370231628, 'label': 'pay_bill'},
    {'score': 0.0755252093076706, 'label': 'balance'}
]
```

원하는 경우 `pipeline`의 결과를 수동으로 복제할 수도 있습니다:

<frameworkcontent>
<pt>
특징 추출기를 가져와서 오디오 파일을 전처리하고 `입력`을 PyTorch 텐서로 반환합니다:

```py
>>> from transformers import AutoFeatureExtractor

>>> feature_extractor = AutoFeatureExtractor.from_pretrained("stevhliu/my_awesome_minds_model")
>>> inputs = feature_extractor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
```

모델에 입력을 전달하고 로짓을 반환합니다:

```py
>>> from transformers import AutoModelForAudioClassification

>>> model = AutoModelForAudioClassification.from_pretrained("stevhliu/my_awesome_minds_model")
>>> with torch.no_grad():
...     logits = model(**inputs).logits
```

확률이 가장 높은 클래스를 가져온 다음 모델의 `id2label` 매핑을 사용하여 이를 레이블로 변환합니다:

```py
>>> import torch

>>> predicted_class_ids = torch.argmax(logits).item()
>>> predicted_label = model.config.id2label[predicted_class_ids]
>>> predicted_label
'cash_deposit'
```
</pt>
</frameworkcontent>
