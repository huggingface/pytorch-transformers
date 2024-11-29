<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# 인코더-디코더 모델[[Encoder Decoder Models]]

## 개요[[Overview]]

[`EncoderDecoderModel`]은 사전 학습된 자동 인코딩(autoencoding) 모델을 인코더로, 사전 학습된 자가 회귀(autoregressive) 모델을 디코더로 활용하여 시퀀스-투-시퀀스(sequence-to-sequence) 모델을 초기화하는 데 이용됩니다.

사전 학습된 체크포인트를 활용해 시퀀스-투-시퀀스 모델을 초기화하는 것이 시퀀스 생성(sequence generation) 작업에 효과적이라는 점이 Sascha Rothe, Shashi Narayan, Aliaksei Severyn의 논문 [Leveraging Pre-trained Checkpoints for Sequence Generation Tasks](https://arxiv.org/abs/1907.12461)에서 입증되었습니다.

[`EncoderDecoderModel`]이 학습/미세 조정된 후에는 다른 모델과 마찬가지로 저장/불러오기가 가능합니다. 자세한 사용법은 예제를 참고하세요.

이 아키텍처의 한 가지 응용 사례는 두 개의 사전 학습된 [`BertModel`]을 각각 인코더와 디코더로 활용하여 요약 모델(summarization model)을 구축하는 것입니다. 이는 Yang Liu와 Mirella Lapata의 논문 [Text Summarization with Pretrained Encoders](https://arxiv.org/abs/1908.08345)에서 제시된 바 있습니다.

## 모델 설정에서 `EncoderDecoderModel`을 무작위 초기화하기[[Randomly initializing `EncoderDecoderModel` from model configurations.]]

[`EncoderDecoderModel`]은 인코더와 디코더 설정(config)을 기반으로 무작위 초기화를 할 수 있습니다. 아래 예시는 [`BertModel`] 설정을 인코더로, 기본 [`BertForCausalLM`] 설정을 디코더로 사용하는 방법을 보여줍니다.

```python
>>> from transformers import BertConfig, EncoderDecoderConfig, EncoderDecoderModel

>>> config_encoder = BertConfig()
>>> config_decoder = BertConfig()

>>> config = EncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)
>>> model = EncoderDecoderModel(config=config)
```

## 사전 학습된 인코더와 디코더로 `EncoderDecoderModel` 초기화하기[[Initialising `EncoderDecoderModel` from a pretrained encoder and a pretrained decoder.]]

[`EncoderDecoderModel`]은 사전 학습된 인코더 체크포인트와 사전 학습된 디코더 체크포인트를 사용해 초기화할 수 있습니다. BERT와 같은 모든 사전 학습된 자동 인코딩(auto-encoding) 모델은 인코더로 활용할 수 있으며, GPT2와 같은 자가 회귀(autoregressive) 모델이나 BART의 디코더와 같이 사전 학습된 시퀀스-투-시퀀스 디코더 모델을 디코더로 사용할 수 있습니다. 디코더로 선택한 아키텍처에 따라 교차 어텐션(cross-attention) 레이어가 무작위로 초기화될 수 있습니다. 사전 학습된 인코더와 디코더 체크포인트를 이용해 [`EncoderDecoderModel`]을 초기화하려면, 모델을 다운스트림 작업에 대해 미세 조정(fine-tuning)해야 합니다. 이에 대한 자세한 내용은 [the *Warm-starting-encoder-decoder blog post*](https://huggingface.co/blog/warm-starting-encoder-decoder)에 설명되어 있습니다. 이 작업을 위해 `EncoderDecoderModel` 클래스는 [`EncoderDecoderModel.from_encoder_decoder_pretrained`] 메서드를 제공합니다.


```python
>>> from transformers import EncoderDecoderModel, BertTokenizer

>>> tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
>>> model = EncoderDecoderModel.from_encoder_decoder_pretrained("google-bert/bert-base-uncased", "google-bert/bert-base-uncased")
```

## 기존 `EncoderDecoderModel` 체크포인트 불러오기 및 추론하기[[Loading an existing `EncoderDecoderModel` checkpoint and perform inference.]]

`EncoderDecoderModel` 클래스의 미세 조정(fine-tuned)된 체크포인트를 불러오려면, Transformers의 다른 모델 아키텍처와 마찬가지로 [`EncoderDecoderModel`]에서 제공하는 `from_pretrained(...)`를 사용할 수 있습니다.

추론을 수행하려면 [`generate`] 메서드를 활용하여 텍스트를 자동 회귀(autoregressive) 방식으로 생성할 수 있습니다. 이 메서드는 탐욕 디코딩(greedy decoding), 빔 서치(beam search), 다항 샘플링(multinomial sampling) 등 다양한 디코딩 방식을 지원합니다.

```python
>>> from transformers import AutoTokenizer, EncoderDecoderModel

>>> # 미세 조정된 seq2seq 모델과 대응하는 토크나이저 가져오기
>>> model = EncoderDecoderModel.from_pretrained("patrickvonplaten/bert2bert_cnn_daily_mail")
>>> tokenizer = AutoTokenizer.from_pretrained("patrickvonplaten/bert2bert_cnn_daily_mail")

>>> # let's perform inference on a long piece of text
>>> ARTICLE_TO_SUMMARIZE = (
...     "PG&E stated it scheduled the blackouts in response to forecasts for high winds "
...     "amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were "
...     "scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."
... )
>>> input_ids = tokenizer(ARTICLE_TO_SUMMARIZE, return_tensors="pt").input_ids

>>> # 자기회귀적으로 요약 생성 (기본적으로 그리디 디코딩 사용)
>>> generated_ids = model.generate(input_ids)
>>> generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
>>> print(generated_text)
nearly 800 thousand customers were affected by the shutoffs. the aim is to reduce the risk of wildfires. nearly 800, 000 customers were expected to be affected by high winds amid dry conditions. pg & e said it scheduled the blackouts to last through at least midday tomorrow.
```

## `TFEncoderDecoderModel`에 Pytorch 체크포인트 불러오기[[Loading a PyTorch checkpoint into `TFEncoderDecoderModel`.]]

[`TFEncoderDecoderModel.from_pretrained`] 메서드는 현재 Pytorch 체크포인트를 사용한 모델 초기화를 지원하지 않습니다. 이 메서드에 `from_pt=True`를 전달하면 예외(exception)가 발생합니다. 특정 인코더-디코더 모델에 대한 Pytorch 체크포인트만 존재하는 경우, 다음과 같은 해결 방법을 사용할 수 있습니다:

```python
>>> # 파이토치 체크포인트에서 로드하는 해결 방법
>>> from transformers import EncoderDecoderModel, TFEncoderDecoderModel

>>> _model = EncoderDecoderModel.from_pretrained("patrickvonplaten/bert2bert-cnn_dailymail-fp16")

>>> _model.encoder.save_pretrained("./encoder")
>>> _model.decoder.save_pretrained("./decoder")

>>> model = TFEncoderDecoderModel.from_encoder_decoder_pretrained(
...     "./encoder", "./decoder", encoder_from_pt=True, decoder_from_pt=True
... )
>>> # 이 부분은 특정 모델의 구체적인 세부사항을 복사할 때에만 사용합니다.
>>> model.config = _model.config
```

## 학습[[Training]]

모델이 생성된 후에는 BART, T5 또는 기타 인코더-디코더 모델과 유사한 방식으로 미세 조정(fine-tuning)할 수 있습니다.
보시다시피, 손실(loss)을 계산하려면 단 2개의 입력만 필요합니다: `input_ids`(입력 시퀀스를 인코딩한 `input_ids`)와 `labels`(목표 시퀀스를 인코딩한 `input_ids`).

```python
>>> from transformers import BertTokenizer, EncoderDecoderModel

>>> tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
>>> model = EncoderDecoderModel.from_encoder_decoder_pretrained("google-bert/bert-base-uncased", "google-bert/bert-base-uncased")

>>> model.config.decoder_start_token_id = tokenizer.cls_token_id
>>> model.config.pad_token_id = tokenizer.pad_token_id

>>> input_ids = tokenizer(
...     "The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side.During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was  finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft).Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct.",
...     return_tensors="pt",
... ).input_ids

>>> labels = tokenizer(
...     "the eiffel tower surpassed the washington monument to become the tallest structure in the world. it was the first structure to reach a height of 300 metres in paris in 1930. it is now taller than the chrysler building by 5. 2 metres ( 17 ft ) and is the second tallest free - standing structure in paris.",
...     return_tensors="pt",
... ).input_ids

>>> # forward 함수가 자동으로 적합한 decoder_input_ids를 생성합니다.
>>> loss = model(input_ids=input_ids, labels=labels).loss
```
훈련에 대한 자세한 내용은 [colab](https://colab.research.google.com/drive/1WIk2bxglElfZewOHboPFNj8H44_VAyKE?usp=sharing#scrollTo=ZwQIEhKOrJpl) 노트북을 참조하세요. 

이 모델은 [thomwolf](https://github.com/thomwolf)가 기여했으며, 이 모델에 대한 TensorFlow 및 Flax 버전은 [ydshieh](https://github.com/ydshieh)가 기여했습니다.


## EncoderDecoderConfig

[[autodoc]] EncoderDecoderConfig

<frameworkcontent>
<pt>

## EncoderDecoderModel

[[autodoc]] EncoderDecoderModel
    - forward
    - from_encoder_decoder_pretrained

</pt>
<tf>

## TFEncoderDecoderModel

[[autodoc]] TFEncoderDecoderModel
    - call
    - from_encoder_decoder_pretrained

</tf>
<jax>

## FlaxEncoderDecoderModel

[[autodoc]] FlaxEncoderDecoderModel
    - __call__
    - from_encoder_decoder_pretrained

</jax>
</frameworkcontent>
