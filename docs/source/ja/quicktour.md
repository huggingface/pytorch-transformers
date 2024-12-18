<!--
Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ このファイルはMarkdown形式ですが、Hugging Faceのドキュメントビルダー向けに特定の構文を含んでいるため、
通常のMarkdownビューアーで正しく表示されないことに注意してください。
-->

# Quick tour

[[open-in-colab]]

🤗 Transformersを使い始めましょう！ 開発者であろうと、日常的なユーザーであろうと、このクイックツアーは
初めて始めるのを支援し、[`pipeline`]を使った推論方法、[AutoClass](./model_doc/auto)で事前学習済みモデルとプリプロセッサをロードする方法、
そしてPyTorchまたはTensorFlowで素早くモデルをトレーニングする方法を示します。 初心者の場合、ここで紹介されたコンセプトの詳細な説明を提供する
チュートリアルまたは[コース](https://huggingface.co/course/chapter1/1)を次に参照することをお勧めします。

始める前に、必要なライブラリがすべてインストールされていることを確認してください：

```bash
!pip install transformers datasets evaluate accelerate
```

あなたはまた、好きな機械学習フレームワークをインストールする必要があります:

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

## Pipeline

<Youtube id="tiZFewofSLM"/>

[`pipeline`] は、事前学習済みモデルを推論に最も簡単で高速な方法です。
[`pipeline`] を使用することで、さまざまなモダリティにわたる多くのタスクに対して即座に使用できます。
いくつかのタスクは以下の表に示されています：

<Tip>

使用可能なタスクの完全な一覧については、[pipeline API リファレンス](./main_classes/pipelines)を確認してください。

</Tip>

| **タスク**                    | **説明**                                                                                                     | **モダリティ**   | **パイプライン識別子**                        |
|------------------------------|--------------------------------------------------------------------------------------------------------------|-----------------|-----------------------------------------------|
| テキスト分類               | テキストのシーケンスにラベルを割り当てる                                                                        | NLP             | pipeline(task="sentiment-analysis")           |
| テキスト生成               | プロンプトを指定してテキストを生成する                                                                          | NLP             | pipeline(task="text-generation")              |
| 要約                       | テキストまたはドキュメントの要約を生成する                                                                      | NLP             | pipeline(task="summarization")                |
| 画像分類                   | 画像にラベルを割り当てる                                                                                      | コンピュータビジョン | pipeline(task="image-classification")         |
| 画像セグメンテーション     | 画像の各個別のピクセルにラベルを割り当てる（セマンティック、パノプティック、およびインスタンスセグメンテーションをサポート） | コンピュータビジョン | pipeline(task="image-segmentation")           |
| オブジェクト検出           | 画像内のオブジェクトの境界ボックスとクラスを予測する                                                          | コンピュータビジョン | pipeline(task="object-detection")             |
| オーディオ分類             | オーディオデータにラベルを割り当てる                                                                           | オーディオ       | pipeline(task="audio-classification")         |
| 自動音声認識             | 音声をテキストに変換する                                                                                     | オーディオ       | pipeline(task="automatic-speech-recognition") |
| ビジュアルクエスチョン応答 | 画像と質問が与えられた場合に、画像に関する質問に回答する                                                       | マルチモーダル  | pipeline(task="vqa")                          |
| ドキュメントクエスチョン応答 | ドキュメントと質問が与えられた場合に、ドキュメントに関する質問に回答する                                     | マルチモーダル  | pipeline(task="document-question-answering")  |
| 画像キャプショニング       | 与えられた画像にキャプションを生成する                                                                         | マルチモーダル  | pipeline(task="image-to-text")                |

まず、[`pipeline`] のインスタンスを作成し、使用したいタスクを指定します。
このガイドでは、センチメント分析のために [`pipeline`] を使用する例を示します：

```python
>>> from transformers import pipeline

>>> classifier = pipeline("sentiment-analysis")
```

[`pipeline`]は、感情分析のためのデフォルトの[事前学習済みモデル](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english)とトークナイザをダウンロードしてキャッシュし、使用できるようになります。
これで、`classifier`を対象のテキストに使用できます：

```python
>>> classifier("私たちは🤗 Transformersライブラリをお見せできてとても嬉しいです。")
[{'label': 'POSITIVE', 'score': 0.9998}]
```

複数の入力がある場合は、[`pipeline`]に入力をリストとして渡して、辞書のリストを返します：

```py
>>> results = classifier(["🤗 Transformersライブラリをご紹介できて非常に嬉しいです。", "嫌いにならないでほしいです。"])
>>> for result in results:
...     print(f"label: {result['label']}, スコア: {round(result['score'], 4)}")
label: POSITIVE, スコア: 0.9998
label: NEGATIVE, スコア: 0.5309
```

[`pipeline`]は、任意のタスクに対してデータセット全体を繰り返し処理することもできます。この例では、自動音声認識をタスクとして選びましょう：

```python
>>> import torch
>>> from transformers import pipeline

>>> speech_recognizer = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")
```

オーディオデータセットをロードします（詳細については🤗 Datasets [クイックスタート](https://huggingface.co/docs/datasets/quickstart#audio)を参照してください）。
たとえば、[MInDS-14](https://huggingface.co/datasets/PolyAI/minds14)データセットをロードします：

```python
>>> from datasets import load_dataset, Audio

>>> dataset = load_dataset("PolyAI/minds14", name="en-US", split="train")  # doctest: +IGNORE_RESULT
```

データセットのサンプリングレートが[`facebook/wav2vec2-base-960h`](https://huggingface.co/facebook/wav2vec2-base-960h)がトレーニングされたサンプリングレートと一致することを確認してください：

```py
>>> dataset = dataset.cast_column("audio", Audio(sampling_rate=speech_recognizer.feature_extractor.sampling_rate))
```

"audio"列を呼び出すと、オーディオファイルは自動的にロードされ、リサンプリングされます。最初の4つのサンプルから生の波形配列を抽出し、それをパイプラインにリストとして渡します。

```py
>>> result = speech_recognizer(dataset[:4]["audio"])
>>> print([d["text"] for d in result])
['I WOULD LIKE TO SET UP A JOINT ACCOUNT WITH MY PARTNER HOW DO I PROCEED WITH DOING THAT', "FONDERING HOW I'D SET UP A JOIN TO HELL T WITH MY WIFE AND WHERE THE AP MIGHT BE", "I I'D LIKE TOY SET UP A JOINT ACCOUNT WITH MY PARTNER I'M NOT SEEING THE OPTION TO DO IT ON THE APSO I CALLED IN TO GET SOME HELP CAN I JUST DO IT OVER THE PHONE WITH YOU AND GIVE YOU THE INFORMATION OR SHOULD I DO IT IN THE AP AN I'M MISSING SOMETHING UQUETTE HAD PREFERRED TO JUST DO IT OVER THE PHONE OF POSSIBLE THINGS", 'HOW DO I FURN A JOINA COUT']
```

大規模なデータセットで、入力が大きい場合（音声や画像など）、すべての入力をメモリに読み込む代わりに、リストではなくジェネレータを渡すことがお勧めです。詳細については[パイプラインAPIリファレンス](./main_classes/pipelines)を参照してください。

### Use another model and tokenizer in the pipeline

[`pipeline`]は[Hub](https://huggingface.co/models)からの任意のモデルを収容でき、他のユースケースに[`pipeline`]を適応させることが容易です。たとえば、フランス語のテキストを処理できるモデルが必要な場合、Hubのタグを使用して適切なモデルをフィルタリングできます。トップのフィルタリングされた結果は、フランス語のテキストに使用できる感情分析用に調整された多言語の[BERTモデル](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment)を返します：

```py
>>> model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
```

<frameworkcontent>
<pt>
[`AutoModelForSequenceClassification`]と[`AutoTokenizer`]を使用して事前学習済みモデルとそれに関連するトークナイザをロードします（次のセクションで`AutoClass`について詳しく説明します）：

```python
>>> from transformers import AutoTokenizer, AutoModelForSequenceClassification

>>> model = AutoModelForSequenceClassification.from_pretrained(model_name)
>>> tokenizer = AutoTokenizer.from_pretrained(model_name)
```

</pt>
<tf>
以下のコードは、[`TFAutoModelForSequenceClassification`]および[`AutoTokenizer`]を使用して、事前学習済みモデルとその関連するトークナイザをロードする方法を示しています（`TFAutoClass`については次のセクションで詳しく説明します）：

```python
>>> from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

>>> model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
>>> tokenizer = AutoTokenizer.from_pretrained(model_name)
```
</tf>
</frameworkcontent>

指定したモデルとトークナイザを[`pipeline`]に設定し、今度はフランス語のテキストに`classifier`を適用できます：

```py
>>> classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
>>> classifier("Nous sommes très heureux de vous présenter la bibliothèque 🤗 Transformers.")
[{'label': '5 stars', 'score': 0.7273}]
```

もし、あなたのユースケースに適したモデルが見つからない場合、事前学習済みモデルをあなたのデータでファインチューニングする必要があります。
ファインチューニングの方法については、[ファインチューニングのチュートリアル](./training)をご覧ください。
最後に、ファインチューニングした事前学習済みモデルを共有し、コミュニティと共有ハブで共有することを検討してください。これにより、機械学習を民主化する手助けができます！ 🤗

## AutoClass

<Youtube id="AhChOFRegn4"/>

[`AutoModelForSequenceClassification`] および [`AutoTokenizer`] クラスは、上記で使用した [`pipeline`] を駆動するために協力して動作します。
[AutoClass](./model_doc/auto) は、事前学習済みモデルのアーキテクチャをその名前またはパスから自動的に取得するショートカットです。
適切な `AutoClass` を選択し、それに関連する前処理クラスを選択するだけで済みます。

前のセクションからの例に戻り、`AutoClass` を使用して [`pipeline`] の結果を再現する方法を見てみましょう。

### AutoTokenizer

トークナイザはテキストをモデルの入力として使用できる数値の配列に前処理する役割を果たします。
トークナイゼーションプロセスには、単語をどのように分割するかや、単語をどのレベルで分割するかといった多くのルールがあります
（トークナイゼーションについての詳細は [トークナイザサマリー](./tokenizer_summary) をご覧ください）。
最も重要なことは、モデルが事前学習済みになったときと同じトークナイゼーションルールを使用するために、同じモデル名でトークナイザをインスタンス化する必要があることです。

[`AutoTokenizer`] を使用してトークナイザをロードします：

```python
>>> from transformers import AutoTokenizer

>>> model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
>>> tokenizer = AutoTokenizer.from_pretrained(model_name)
```

Pass your text to the tokenizer:

```python
>>> encoding = tokenizer("私たちは🤗 Transformersライブラリをお見せできてとても嬉しいです。")
>>> print(encoding)
{'input_ids': [101, 11312, 10320, 12495, 19308, 10114, 11391, 10855, 10103, 100, 58263, 13299, 119, 102],
 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
```

トークナイザは、次の情報を含む辞書を返します：

- [input_ids](./glossary#input-ids): トークンの数値表現。
- [attention_mask](.glossary#attention-mask): どのトークンにアテンションを向けるかを示します。

トークナイザはまた、入力のリストを受け入れ、一様な長さのバッチを返すためにテキストをパディングおよび切り詰めることができます。

<frameworkcontent>
<pt>

```py
>>> pt_batch = tokenizer(
...     ["🤗 Transformersライブラリをお見せできて非常に嬉しいです。", "嫌いではないことを願っています。"],
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

[前処理](./preprocessing)チュートリアルをご覧いただき、トークナイゼーションの詳細や、[`AutoImageProcessor`]、[`AutoFeatureExtractor`]、[`AutoProcessor`]を使用して画像、オーディオ、およびマルチモーダル入力を前処理する方法について詳しく説明されているページもご覧ください。

</Tip>

### AutoModel

<frameworkcontent>
<pt>
🤗 Transformersは事前学習済みインスタンスを簡単に統一的にロードする方法を提供します。
これは、[`AutoTokenizer`]をロードするのと同じように[`AutoModel`]をロードできることを意味します。
タスクに適した[`AutoModel`]を選択する以外の違いはありません。
テキスト（またはシーケンス）分類の場合、[`AutoModelForSequenceClassification`]をロードする必要があります：

```py
>>> from transformers import AutoModelForSequenceClassification

>>> model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
>>> pt_model = AutoModelForSequenceClassification.from_pretrained(model_name)
```

<Tip>

[`AutoModel`]クラスでサポートされているタスクに関する詳細については、[タスクの概要](./task_summary)を参照してください。

</Tip>

今、前処理済みのバッチを直接モデルに渡します。辞書を展開するだけで、`**`を追加する必要があります：

```python
>>> pt_outputs = pt_model(**pt_batch)
```

モデルは、`logits`属性に最終的なアクティベーションを出力します。 `logits`にsoftmax関数を適用して確率を取得します：

```py
>>> from torch import nn

>>> pt_predictions = nn.functional.softmax(pt_outputs.logits, dim=-1)
>>> print(pt_predictions)
tensor([[0.0021, 0.0018, 0.0115, 0.2121, 0.7725],
        [0.2084, 0.1826, 0.1969, 0.1755, 0.2365]], grad_fn=<SoftmaxBackward0>)
```

</pt>
<tf>
🤗 Transformersは事前学習済みインスタンスをロードするためのシンプルで統一された方法を提供します。
これは、[`TFAutoModel`]を[`AutoTokenizer`]をロードするのと同じようにロードできることを意味します。
唯一の違いは、タスクに適した[`TFAutoModel`]を選択することです。
テキスト（またはシーケンス）分類の場合、[`TFAutoModelForSequenceClassification`]をロードする必要があります：

```py
>>> from transformers import TFAutoModelForSequenceClassification

>>> model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
>>> tf_model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
```

<Tip>

詳細については、[`AutoModel`]クラスでサポートされているタスクに関する情報は、[タスクの概要](./task_summary)を参照してください。

</Tip>

次に、前処理済みのバッチを直接モデルに渡します。テンソルをそのまま渡すことができます：

```python
>>> tf_outputs = tf_model(tf_batch)
```

モデルは`logits`属性に最終的なアクティベーションを出力します。`logits`にソフトマックス関数を適用して確率を取得します：

```python
>>> import tensorflow as tf

>>> tf_predictions = tf.nn.softmax(tf_outputs.logits, axis=-1)
>>> tf_predictions  # doctest: +IGNORE_RESULT
```

</tf>
</frameworkcontent>

<Tip>

🤗 Transformersのすべてのモデル（PyTorchまたはTensorFlow）は、最終的な活性化関数（softmaxなど）*前*のテンソルを出力します。
最終的な活性化関数は、しばしば損失と結合されているためです。モデルの出力は特別なデータクラスであり、その属性はIDEで自動補完されます。
モデルの出力は、タプルまたは辞書のように動作します（整数、スライス、または文字列でインデックスを付けることができます）。
この場合、Noneである属性は無視されます。

</Tip>

### Save a Model

<frameworkcontent>
<pt>
モデルをファインチューニングしたら、[`PreTrainedModel.save_pretrained`]を使用してトークナイザと共に保存できます：

```py
>>> pt_save_directory = "./pt_save_pretrained"
>>> tokenizer.save_pretrained(pt_save_directory)  # doctest: +IGNORE_RESULT
>>> pt_model.save_pretrained(pt_save_directory)
```

再びモデルを使用する準備ができたら、[`PreTrainedModel.from_pretrained`]を使用して再度ロードします：

```py
>>> pt_model = AutoModelForSequenceClassification.from_pretrained("./pt_save_pretrained")
```

</pt>
<tf>
モデルをファインチューニングしたら、そのトークナイザを使用してモデルを保存できます。[`TFPreTrainedModel.save_pretrained`]を使用します：

```py
>>> tf_save_directory = "./tf_save_pretrained"
>>> tokenizer.save_pretrained(tf_save_directory)  # doctest: +IGNORE_RESULT
>>> tf_model.save_pretrained(tf_save_directory)
```

モデルを再度使用する準備ができたら、[`TFPreTrainedModel.from_pretrained`]を使用して再度ロードします：

```py
>>> tf_model = TFAutoModelForSequenceClassification.from_pretrained("./tf_save_pretrained")
```

</tf>
</frameworkcontent>

🤗 Transformersの特に素晴らしい機能の一つは、モデルを保存し、それをPyTorchモデルまたはTensorFlowモデルとして再ロードできることです。 `from_pt`または`from_tf`パラメータを使用してモデルをフレームワーク間で変換できます：

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

## Custom model builds

モデルを構築方法を変更するには、モデルの設定クラスを変更できます。設定はモデルの属性を指定します。例えば、隠れ層の数やアテンションヘッドの数などがこれに含まれます。カスタム設定クラスからモデルを初期化する際には、ゼロから始めます。モデルの属性はランダムに初期化され、有意義な結果を得るためにモデルをトレーニングする必要があります。

最初に[`AutoConfig`]をインポートし、変更したい事前学習済みモデルをロードします。[`AutoConfig.from_pretrained`]内で、変更したい属性（例：アテンションヘッドの数）を指定できます：

```python
>>> from transformers import AutoConfig

>>> my_config = AutoConfig.from_pretrained("distilbert/distilbert-base-uncased", n_heads=12)
```

<frameworkcontent>
<pt>
[`AutoModel.from_config`]を使用してカスタム設定からモデルを作成します：

```python
>>> from transformers import AutoModel

>>> my_model = AutoModel.from_config(my_config)
```

</pt>
<tf>
カスタム構成からモデルを作成するには、[`TFAutoModel.from_config`]を使用します：

```py
>>> from transformers import TFAutoModel

>>> my_model = TFAutoModel.from_config(my_config)
```

</tf>
</frameworkcontent>

[カスタムアーキテクチャを作成](./create_a_model)ガイドを参照して、カスタム構成の詳細情報を確認してください。

## Trainer - PyTorch向けの最適化されたトレーニングループ

すべてのモデルは標準の[`torch.nn.Module`](https://pytorch.org/docs/stable/nn.html#torch.nn.Module)であるため、通常のトレーニングループで使用できます。
独自のトレーニングループを作成できますが、🤗 TransformersはPyTorch向けに[`Trainer`]クラスを提供しており、基本的なトレーニングループに加えて、
分散トレーニング、混合精度などの機能の追加を行っています。

タスクに応じて、通常は[`Trainer`]に以下のパラメータを渡します：

1. [`PreTrainedModel`]または[`torch.nn.Module`](https://pytorch.org/docs/stable/nn.html#torch.nn.Module)から始めます：

    ```py
    >>> from transformers import AutoModelForSequenceClassification

    >>> model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased")
    ```

2. [`TrainingArguments`]には、変更できるモデルのハイパーパラメータが含まれており、学習率、バッチサイズ、トレーニングエポック数などが変更できます。指定しない場合、デフォルト値が使用されます：

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

3. トークナイザ、画像プロセッサ、特徴量抽出器、またはプロセッサのような前処理クラスをロードします：

   ```py
   >>> from transformers import AutoTokenizer

   >>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
    ```

4. データセットをロードする:

   ```py
   >>> from datasets import load_dataset

   >>> dataset = load_dataset("rotten_tomatoes")  # doctest: +IGNORE_RESULT
    ```

5. データセットをトークン化するための関数を作成します：

   ```python
   >>> def tokenize_dataset(dataset):
   ...     return tokenizer(dataset["text"])
   ```

    その後、[`~datasets.Dataset.map`]を使用してデータセット全体に適用します：

    ```python
    >>> dataset = dataset.map(tokenize_dataset, batched=True)
    ```

6. データセットからの例のバッチを作成するための [`DataCollatorWithPadding`]：

   ```py
   >>> from transformers import DataCollatorWithPadding

   >>> data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
   ```

次に、これらのクラスを[`Trainer`]にまとめます：

```python
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

訓練を開始する準備ができたら、[`~Trainer.train`]を呼び出してトレーニングを開始します：

```py
>>> trainer.train()  # doctest: +SKIP
```

<Tip>

翻訳や要約など、シーケンス間モデルを使用するタスクには、代わりに[`Seq2SeqTrainer`]と[`Seq2SeqTrainingArguments`]クラスを使用してください。

</Tip>

[`Trainer`]内のメソッドをサブクラス化することで、トレーニングループの動作をカスタマイズできます。これにより、損失関数、オプティマイザ、スケジューラなどの機能をカスタマイズできます。サブクラス化できるメソッドの一覧については、[`Trainer`]リファレンスをご覧ください。

トレーニングループをカスタマイズする別の方法は、[Callbacks](./main_classes/callback)を使用することです。コールバックを使用して他のライブラリと統合し、トレーニングループを監視して進捗状況を報告したり、トレーニングを早期に停止したりできます。コールバックはトレーニングループ自体には何も変更を加えません。損失関数などのカスタマイズを行う場合は、[`Trainer`]をサブクラス化する必要があります。

## Train with TensorFlow

すべてのモデルは標準の[`tf.keras.Model`](https://www.tensorflow.org/api_docs/python/tf/keras/Model)であるため、[Keras](https://keras.io/) APIを使用してTensorFlowでトレーニングできます。
🤗 Transformersは、データセットを`tf.data.Dataset`として簡単にロードできるようにする[`~TFPreTrainedModel.prepare_tf_dataset`]メソッドを提供しており、Kerasの[`compile`](https://keras.io/api/models/model_training_apis/#compile-method)および[`fit`](https://keras.io/api/models/model_training_apis/#fit-method)メソッドを使用してトレーニングをすぐに開始できます。

1. [`TFPreTrainedModel`]または[`tf.keras.Model`](https://www.tensorflow.org/api_docs/python/tf/keras/Model)から始めます：

   ```py
   >>> from transformers import TFAutoModelForSequenceClassification

   >>> model = TFAutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased")
   ```

2. トークナイザ、画像プロセッサ、特徴量抽出器、またはプロセッサのような前処理クラスをロードします：

   ```py
   >>> from transformers import AutoTokenizer

   >>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
   ```

3. データセットをトークナイズするための関数を作成します：

   ```python
   >>> def tokenize_dataset(dataset):
   ...     return tokenizer(dataset["text"])  # doctest: +SKIP
   ```

4. [`~datasets.Dataset.map`]を使用してデータセット全体にトークナイザを適用し、データセットとトークナイザを[`~TFPreTrainedModel.prepare_tf_dataset`]に渡します。バッチサイズを変更し、データセットをシャッフルすることもできます。

   ```python
   >>> dataset = dataset.map(tokenize_dataset)  # doctest: +SKIP
   >>> tf_dataset = model.prepare_tf_dataset(
   ...     dataset["train"], batch_size=16, shuffle=True, tokenizer=tokenizer
   ... )  # doctest: +SKIP
   ```

5. 準備ができたら、`compile`と`fit`を呼び出してトレーニングを開始できます。 Transformersモデルはすべてデフォルトのタスクに関連する損失関数を持っているため、指定しない限り、損失関数を指定する必要はありません。

   ```python
   >>> from tensorflow.keras.optimizers import Adam

   >>> model.compile(optimizer=Adam(3e-5))  # 損失関数の引数は不要です！
   >>> model.fit(tf
   ```

## What's next?

🤗 Transformersのクイックツアーを完了したら、ガイドをチェックして、カスタムモデルの作成、タスクのためのファインチューニング、スクリプトを使用したモデルのトレーニングなど、より具体的なことを学ぶことができます。🤗 Transformersのコアコンセプトについてもっと詳しく知りたい場合は、コンセプチュアルガイドを読んでみてください！
