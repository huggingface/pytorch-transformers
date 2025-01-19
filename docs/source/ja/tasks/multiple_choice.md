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

# Multiple choice

[[open-in-colab]]

多肢選択タスクは質問応答に似ていますが、いくつかの候補の回答がコンテキストとともに提供され、正しい回答を選択するようにモデルがトレーニングされる点が異なります。

このガイドでは、次の方法を説明します。

1. [SWAG](https://huggingface.co/datasets/swag) データセットの「通常」構成で [BERT](https://huggingface.co/google-bert/bert-base-uncased) を微調整して、最適なデータセットを選択します複数の選択肢と何らかのコンテキストを考慮して回答します。
2. 微調整したモデルを推論に使用します。

始める前に、必要なライブラリがすべてインストールされていることを確認してください。

```bash
pip install transformers datasets evaluate
```

モデルをアップロードしてコミュニティと共有できるように、Hugging Face アカウントにログインすることをお勧めします。プロンプトが表示されたら、トークンを入力してログインします。

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## Load SWAG dataset

まず、🤗 データセット ライブラリから SWAG データセットの「通常」構成をロードします。

```py
>>> from datasets import load_dataset

>>> swag = load_dataset("swag", "regular")
```

次に、例を見てみましょう。

```py
>>> swag["train"][0]
{'ending0': 'passes by walking down the street playing their instruments.',
 'ending1': 'has heard approaching them.',
 'ending2': "arrives and they're outside dancing and asleep.",
 'ending3': 'turns the lead singer watches the performance.',
 'fold-ind': '3416',
 'gold-source': 'gold',
 'label': 0,
 'sent1': 'Members of the procession walk down the street holding small horn brass instruments.',
 'sent2': 'A drum line',
 'startphrase': 'Members of the procession walk down the street holding small horn brass instruments. A drum line',
 'video-id': 'anetv_jkn6uvmqwh4'}
```

ここにはたくさんのフィールドがあるように見えますが、実際は非常に簡単です。

- `sent1` と `sent2`: これらのフィールドは文の始まりを示し、この 2 つを組み合わせると `startphrase` フィールドが得られます。
- `ending`: 文の終わり方として考えられる終わり方を示唆しますが、正しいのは 1 つだけです。
- `label`: 正しい文の終わりを識別します。

## Preprocess

次のステップでは、BERT トークナイザーをロードして、文の始まりと 4 つの可能な終わりを処理します。

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
```

作成する前処理関数は次のことを行う必要があります。

1. `sent1` フィールドのコピーを 4 つ作成し、それぞれを `sent2` と組み合わせて文の始まりを再現します。
2. `sent2` を 4 つの可能な文末尾のそれぞれと組み合わせます。
3. これら 2 つのリストをトークン化できるようにフラット化し、その後、各例に対応する `input_ids`、`attention_mask`、および `labels` フィールドが含まれるように非フラット化します。

```py
>>> ending_names = ["ending0", "ending1", "ending2", "ending3"]


>>> def preprocess_function(examples):
...     first_sentences = [[context] * 4 for context in examples["sent1"]]
...     question_headers = examples["sent2"]
...     second_sentences = [
...         [f"{header} {examples[end][i]}" for end in ending_names] for i, header in enumerate(question_headers)
...     ]

...     first_sentences = sum(first_sentences, [])
...     second_sentences = sum(second_sentences, [])

...     tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True)
...     return {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}
```

データセット全体に前処理関数を適用するには、🤗 Datasets [`~datasets.Dataset.map`] メソッドを使用します。 `batched=True` を設定してデータセットの複数の要素を一度に処理することで、`map` 関数を高速化できます。


```py
tokenized_swag = swag.map(preprocess_function, batched=True)
```

[`DataCollatorForMultipleChoice`] は、すべてのモデル入力を平坦化し、パディングを適用して、結果を非平坦化します。
```py
>>> from transformers import DataCollatorForMultipleChoice
>>> collator = DataCollatorForMultipleChoice(tokenizer=tokenizer)
```

## Evaluate

トレーニング中にメトリクスを含めると、多くの場合、モデルのパフォーマンスを評価するのに役立ちます。 🤗 [Evaluate](https://huggingface.co/docs/evaluate/index) ライブラリを使用して、評価メソッドをすばやくロードできます。このタスクでは、[accuracy](https://huggingface.co/spaces/evaluate-metric/accuracy) メトリクスを読み込みます (🤗 Evaluate [クイック ツアー](https://huggingface.co/docs/evaluate/a_quick_tour) を参照してください) ) メトリクスの読み込みと計算方法の詳細については、次を参照してください)。

```py
>>> import evaluate

>>> accuracy = evaluate.load("accuracy")
```

次に、予測とラベルを [`~evaluate.EvaluationModule.compute`] に渡して精度を計算する関数を作成します。

```py
>>> import numpy as np


>>> def compute_metrics(eval_pred):
...     predictions, labels = eval_pred
...     predictions = np.argmax(predictions, axis=1)
...     return accuracy.compute(predictions=predictions, references=labels)
```

これで`compute_metrics`関数の準備が整いました。トレーニングをセットアップするときにこの関数に戻ります。

## Train

<frameworkcontent>
<pt>
<Tip>

[`Trainer`] を使用したモデルの微調整に慣れていない場合は、[ここ](../training#train-with-pytorch-trainer) の基本的なチュートリアルをご覧ください。

</Tip>

これでモデルのトレーニングを開始する準備が整いました。 [`AutoModelForMultipleChoice`] を使用して BERT をロードします。

```py
>>> from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer

>>> model = AutoModelForMultipleChoice.from_pretrained("google-bert/bert-base-uncased")
```

この時点で残っている手順は次の 3 つだけです。

1. [`TrainingArguments`] でトレーニング ハイパーパラメータを定義します。唯一の必須パラメータは、モデルの保存場所を指定する `output_dir` です。 `push_to_hub=True`を設定して、このモデルをハブにプッシュします (モデルをアップロードするには、Hugging Face にサインインする必要があります)。各エポックの終了時に、[`Trainer`] は精度を評価し、トレーニング チェックポイントを保存します。
2. トレーニング引数を、モデル、データセット、トークナイザー、データ照合器、および `compute_metrics` 関数とともに [`Trainer`] に渡します。
3. [`~Trainer.train`] を呼び出してモデルを微調整します。

```py
>>> training_args = TrainingArguments(
...     output_dir="my_awesome_swag_model",
...     eval_strategy="epoch",
...     save_strategy="epoch",
...     load_best_model_at_end=True,
...     learning_rate=5e-5,
...     per_device_train_batch_size=16,
...     per_device_eval_batch_size=16,
...     num_train_epochs=3,
...     weight_decay=0.01,
...     push_to_hub=True,
... )

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=tokenized_swag["train"],
...     eval_dataset=tokenized_swag["validation"],
...     processing_class=tokenizer,
...     data_collator=collator,
...     compute_metrics=compute_metrics,
... )

>>> trainer.train()
```

トレーニングが完了したら、 [`~transformers.Trainer.push_to_hub`] メソッドを使用してモデルをハブに共有し、誰もがモデルを使用できますように。

```py
>>> trainer.push_to_hub()
```
</pt>
<tf>
<Tip>

Keras を使用したモデルの微調整に慣れていない場合は、[こちら](../training#train-a-tensorflow-model-with-keras) の基本的なチュートリアルをご覧ください。

</Tip>
TensorFlow でモデルを微調整するには、オプティマイザー関数、学習率スケジュール、およびいくつかのトレーニング ハイパーパラメーターをセットアップすることから始めます。

```py
>>> from transformers import create_optimizer

>>> batch_size = 16
>>> num_train_epochs = 2
>>> total_train_steps = (len(tokenized_swag["train"]) // batch_size) * num_train_epochs
>>> optimizer, schedule = create_optimizer(init_lr=5e-5, num_warmup_steps=0, num_train_steps=total_train_steps)
```

次に、[`TFAutoModelForMultipleChoice`] を使用して BERT をロードできます。

```py
>>> from transformers import TFAutoModelForMultipleChoice

>>> model = TFAutoModelForMultipleChoice.from_pretrained("google-bert/bert-base-uncased")
```

[`~transformers.TFPreTrainedModel.prepare_tf_dataset`] を使用して、データセットを `tf.data.Dataset` 形式に変換します。

```py
>>> data_collator = DataCollatorForMultipleChoice(tokenizer=tokenizer)
>>> tf_train_set = model.prepare_tf_dataset(
...     tokenized_swag["train"],
...     shuffle=True,
...     batch_size=batch_size,
...     collate_fn=data_collator,
... )

>>> tf_validation_set = model.prepare_tf_dataset(
...     tokenized_swag["validation"],
...     shuffle=False,
...     batch_size=batch_size,
...     collate_fn=data_collator,
... )
```

[`compile`](https://keras.io/api/models/model_training_apis/#compile-method) を使用してトレーニング用のモデルを設定します。 Transformers モデルにはすべてデフォルトのタスク関連の損失関数があるため、次の場合を除き、損失関数を指定する必要はないことに注意してください。

```py
>>> model.compile(optimizer=optimizer)  # No loss argument!
```

トレーニングを開始する前にセットアップする最後の 2 つのことは、予測から精度を計算することと、モデルをハブにプッシュする方法を提供することです。どちらも [Keras コールバック](../main_classes/keras_callbacks) を使用して行われます。

`compute_metrics` 関数を [`~transformers.KerasMetricCallback`] に渡します。

```py
>>> from transformers.keras_callbacks import KerasMetricCallback

>>> metric_callback = KerasMetricCallback(metric_fn=compute_metrics, eval_dataset=tf_validation_set)
```

[`~transformers.PushToHubCallback`] でモデルとトークナイザーをプッシュする場所を指定します。

```py
>>> from transformers.keras_callbacks import PushToHubCallback

>>> push_to_hub_callback = PushToHubCallback(
...     output_dir="my_awesome_model",
...     tokenizer=tokenizer,
... )
```

次に、コールバックをまとめてバンドルします。

```py
>>> callbacks = [metric_callback, push_to_hub_callback]
```

ついに、モデルのトレーニングを開始する準備が整いました。トレーニングおよび検証データセット、エポック数、コールバックを指定して [`fit`](https://keras.io/api/models/model_training_apis/#fit-method) を呼び出し、モデルを微調整します。

```py
>>> model.fit(x=tf_train_set, validation_data=tf_validation_set, epochs=2, callbacks=callbacks)
```

トレーニングが完了すると、モデルは自動的にハブにアップロードされ、誰でも使用できるようになります。

</tf>
</frameworkcontent>


<Tip>

複数選択用にモデルを微調整する方法の詳細な例については、対応するセクションを参照してください。
[PyTorch ノートブック](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/multiple_choice.ipynb)
または [TensorFlow ノートブック](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/multiple_choice-tf.ipynb)。

</Tip>


# Inference

モデルを微調整したので、それを推論に使用できるようになりました。

いくつかのテキストと 2 つの回答候補を考えてください。

```py
>>> prompt = "France has a bread law, Le Décret Pain, with strict rules on what is allowed in a traditional baguette."
>>> candidate1 = "The law does not apply to croissants and brioche."
>>> candidate2 = "The law applies to baguettes."
```

<frameworkcontent>
<pt>

各プロンプトと回答候補のペアをトークン化し、PyTorch テンソルを返します。いくつかの`lables`も作成する必要があります。

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("my_awesome_swag_model")
>>> inputs = tokenizer([[prompt, candidate1], [prompt, candidate2]], return_tensors="pt", padding=True)
>>> labels = torch.tensor(0).unsqueeze(0)
```

入力とラベルをモデルに渡し、`logits`を返します。

```py
>>> from transformers import AutoModelForMultipleChoice

>>> model = AutoModelForMultipleChoice.from_pretrained("my_awesome_swag_model")
>>> outputs = model(**{k: v.unsqueeze(0) for k, v in inputs.items()}, labels=labels)
>>> logits = outputs.logits
```

最も高い確率でクラスを取得します。

```py
>>> predicted_class = logits.argmax().item()
>>> predicted_class
'0'
```
</pt>
<tf>

各プロンプトと回答候補のペアをトークン化し、TensorFlow テンソルを返します。

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("my_awesome_swag_model")
>>> inputs = tokenizer([[prompt, candidate1], [prompt, candidate2]], return_tensors="tf", padding=True)
```

入力をモデルに渡し、`logits`を返します。

```py
>>> from transformers import TFAutoModelForMultipleChoice

>>> model = TFAutoModelForMultipleChoice.from_pretrained("my_awesome_swag_model")
>>> inputs = {k: tf.expand_dims(v, 0) for k, v in inputs.items()}
>>> outputs = model(inputs)
>>> logits = outputs.logits
```

最も高い確率でクラスを取得します。

```py
>>> predicted_class = int(tf.math.argmax(logits, axis=-1)[0])
>>> predicted_class
'0'
```
</tf>
</frameworkcontent>
