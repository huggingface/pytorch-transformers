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

# Audio classification

[[open-in-colab]]

<Youtube id="KWwzcmG98Ds"/>


音声分類では、テキストと同様に、入力データから出力されたクラス ラベルを割り当てます。唯一の違いは、テキスト入力の代わりに生のオーディオ波形があることです。音声分類の実際的な応用例には、話者の意図、言語分類、さらには音による動物の種類の識別などがあります。

このガイドでは、次の方法を説明します。

1. [MInDS-14](https://huggingface.co/datasets/PolyAI/minds14) データセットで [Wav2Vec2](https://huggingface.co/facebook/wav2vec2-base) を微調整して話者の意図を分類します。
2. 微調整したモデルを推論に使用します。

<Tip>

このタスクと互換性のあるすべてのアーキテクチャとチェックポイントを確認するには、[タスクページ](https://huggingface.co/tasks/audio-classification) を確認することをお勧めします。

</Tip>

```bash
pip install transformers datasets evaluate
```

モデルをアップロードしてコミュニティと共有できるように、Hugging Face アカウントにログインすることをお勧めします。プロンプトが表示されたら、トークンを入力してログインします。

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## Load MInDS-14 dataset

まず、🤗 データセット ライブラリから MInDS-14 データセットをロードします。

```py
>>> from datasets import load_dataset, Audio

>>> minds = load_dataset("PolyAI/minds14", name="en-US", split="train")
```

[`~datasets.Dataset.train_test_split`] メソッドを使用して、データセットの `train` をより小さなトレインとテスト セットに分割します。これにより、完全なデータセットにさらに時間を費やす前に、実験してすべてが機能することを確認する機会が得られます。

```py
>>> minds = minds.train_test_split(test_size=0.2)
```

次に、データセットを見てみましょう。

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


データセットには`lang_id`や`english_transcription`などの多くの有用な情報が含まれていますが、このガイドでは`audio`と`intent_class`に焦点を当てます。 [`~datasets.Dataset.remove_columns`] メソッドを使用して他の列を削除します。

```py
>>> minds = minds.remove_columns(["path", "transcription", "english_transcription", "lang_id"])
```

ここで例を見てみましょう。

```py
>>> minds["train"][0]
{'audio': {'array': array([ 0.        ,  0.        ,  0.        , ..., -0.00048828,
         -0.00024414, -0.00024414], dtype=float32),
  'path': '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~APP_ERROR/602b9a5fbb1e6d0fbce91f52.wav',
  'sampling_rate': 8000},
 'intent_class': 2}
```

次の 2 つのフィールドがあります。

- `audio`: 音声ファイルをロードしてリサンプリングするために呼び出す必要がある音声信号の 1 次元の `array`。
- `intent_class`: スピーカーのインテントのクラス ID を表します。

モデルがラベル ID からラベル名を取得しやすくするために、ラベル名を整数に、またはその逆にマップする辞書を作成します。

```py
>>> labels = minds["train"].features["intent_class"].names
>>> label2id, id2label = dict(), dict()
>>> for i, label in enumerate(labels):
...     label2id[label] = str(i)
...     id2label[str(i)] = label
```

これで、ラベル ID をラベル名に変換できるようになりました。

```py
>>> id2label[str(2)]
'app_error'
```

## Preprocess

次のステップでは、Wav2Vec2 特徴抽出プログラムをロードしてオーディオ信号を処理します。

```py
>>> from transformers import AutoFeatureExtractor

>>> feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
```

MInDS-14 データセットのサンプリング レートは 8khz です (この情報は [データセット カード](https://huggingface.co/datasets/PolyAI/minds14) で確認できます)。つまり、データセットを再サンプリングする必要があります。事前トレーニングされた Wav2Vec2 モデルを使用するには、16kHz に設定します。

```py
>>> minds = minds.cast_column("audio", Audio(sampling_rate=16_000))
>>> minds["train"][0]
{'audio': {'array': array([ 2.2098757e-05,  4.6582241e-05, -2.2803260e-05, ...,
         -2.8419291e-04, -2.3305941e-04, -1.1425107e-04], dtype=float32),
  'path': '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~APP_ERROR/602b9a5fbb1e6d0fbce91f52.wav',
  'sampling_rate': 16000},
 'intent_class': 2}
```

次に、次の前処理関数を作成します。

1. `audio`列を呼び出してロードし、必要に応じてオーディオ ファイルをリサンプリングします。
2. オーディオ ファイルのサンプリング レートが、モデルが事前トレーニングされたオーディオ データのサンプリング レートと一致するかどうかを確認します。この情報は、Wav2Vec2 [モデル カード](https://huggingface.co/facebook/wav2vec2-base) で見つけることができます。
3. 入力の最大長を設定して、長い入力を切り捨てずにバッチ処理します。

```py
>>> def preprocess_function(examples):
...     audio_arrays = [x["array"] for x in examples["audio"]]
...     inputs = feature_extractor(
...         audio_arrays, sampling_rate=feature_extractor.sampling_rate, max_length=16000, truncation=True
...     )
...     return inputs
```

データセット全体に前処理関数を適用するには、🤗 Datasets [`~datasets.Dataset.map`] 関数を使用します。 `batched=True` を設定してデータセットの複数の要素を一度に処理することで、`map` を高速化できます。不要な列を削除し、`intent_class` の名前を `label` に変更します。これはモデルが期待する名前であるためです。

```py
>>> encoded_minds = minds.map(preprocess_function, remove_columns="audio", batched=True)
>>> encoded_minds = encoded_minds.rename_column("intent_class", "label")
```
## Evaluate

トレーニング中にメトリクスを含めると、多くの場合、モデルのパフォーマンスを評価するのに役立ちます。 🤗 [Evaluate](https://huggingface.co/docs/evaluate/index) ライブラリを使用して、評価メソッドをすばやくロードできます。このタスクでは、[accuracy](https://huggingface.co/spaces/evaluate-metric/accuracy) メトリクスを読み込みます (🤗 Evaluate [クイック ツアー](https://huggingface.co/docs/evaluate/a_quick_tour) を参照してください) メトリクスの読み込みと計算方法の詳細については、次を参照してください。

```py
>>> import evaluate

>>> accuracy = evaluate.load("accuracy")
```

次に、予測とラベルを [`~evaluate.EvaluationModule.compute`] に渡して精度を計算する関数を作成します。

```py
>>> import numpy as np


>>> def compute_metrics(eval_pred):
...     predictions = np.argmax(eval_pred.predictions, axis=1)
...     return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)
```

これで`compute_metrics`関数の準備が整いました。トレーニングをセットアップするときにこの関数に戻ります。

## Train

<frameworkcontent>
<pt>
<Tip>

[`Trainer`] を使用したモデルの微調整に慣れていない場合は、[こちら](../training#train-with-pytorch-trainer) の基本的なチュートリアルをご覧ください。

</Tip>

これでモデルのトレーニングを開始する準備が整いました。 [`AutoModelForAudioClassification`] を使用して、予期されるラベルの数とラベル マッピングを使用して Wav2Vec2 を読み込みます。

```py
>>> from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer

>>> num_labels = len(id2label)
>>> model = AutoModelForAudioClassification.from_pretrained(
...     "facebook/wav2vec2-base", num_labels=num_labels, label2id=label2id, id2label=id2label
... )
```

この時点で残っている手順は次の 3 つだけです。

1. [`TrainingArguments`] でトレーニング ハイパーパラメータを定義します。唯一の必須パラメータは、モデルの保存場所を指定する `output_dir` です。 `push_to_hub=True`を設定して、このモデルをハブにプッシュします (モデルをアップロードするには、Hugging Face にサインインする必要があります)。各エポックの終了時に、[`トレーナー`] は精度を評価し、トレーニング チェックポイントを保存します。
2. トレーニング引数を、モデル、データセット、トークナイザー、データ照合器、および `compute_metrics` 関数とともに [`Trainer`] に渡します。
3. [`~Trainer.train`] を呼び出してモデルを微調整します。

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

トレーニングが完了したら、 [`~transformers.Trainer.push_to_hub`] メソッドを使用してモデルをハブに共有し、誰もがモデルを使用できるようにします。

```py
>>> trainer.push_to_hub()
```
</pt>
</frameworkcontent>

<Tip>

音声分類用のモデルを微調整する方法の詳細な例については、対応する [PyTorch notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/audio_classification.ipynb).

</Tip>

## Inference

モデルを微調整したので、それを推論に使用できるようになりました。

推論を実行したい音声ファイルをロードします。必要に応じて、オーディオ ファイルのサンプリング レートをモデルのサンプリング レートと一致するようにリサンプリングすることを忘れないでください。

```py
>>> from datasets import load_dataset, Audio

>>> dataset = load_dataset("PolyAI/minds14", name="en-US", split="train")
>>> dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
>>> sampling_rate = dataset.features["audio"].sampling_rate
>>> audio_file = dataset[0]["audio"]["path"]
```

推論用に微調整されたモデルを試す最も簡単な方法は、それを [`pipeline`] で使用することです。モデルを使用して音声分類用の`pipeline`をインスタンス化し、それに音声ファイルを渡します。

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

必要に応じて、`pipeline` の結果を手動で複製することもできます。

<frameworkcontent>
<pt>

特徴抽出器をロードしてオーディオ ファイルを前処理し、`input`を PyTorch テンソルとして返します。

```py
>>> from transformers import AutoFeatureExtractor

>>> feature_extractor = AutoFeatureExtractor.from_pretrained("stevhliu/my_awesome_minds_model")
>>> inputs = feature_extractor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
```

入力をモデルに渡し、ロジットを返します。

```py
>>> from transformers import AutoModelForAudioClassification

>>> model = AutoModelForAudioClassification.from_pretrained("stevhliu/my_awesome_minds_model")
>>> with torch.no_grad():
...     logits = model(**inputs).logits
```

最も高い確率でクラスを取得し、モデルの `id2label` マッピングを使用してそれをラベルに変換します。

```py
>>> import torch

>>> predicted_class_ids = torch.argmax(logits).item()
>>> predicted_label = model.config.id2label[predicted_class_ids]
>>> predicted_label
'cash_deposit'
```
</pt>
</frameworkcontent>
