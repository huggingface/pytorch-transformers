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

# 快速上手

[[open-in-colab]]

快来使用 🤗 Transformers 吧！无论你是开发人员还是日常用户，这篇快速上手教程都将帮助你入门并且向你展示如何使用 [`pipeline`] 进行推理，使用 [AutoClass](./model_doc/auto) 加载一个预训练模型和预处理器，以及使用 PyTorch 或 TensorFlow 快速训练一个模型。如果你是一个初学者，我们建议你接下来查看我们的教程或者[课程](https://huggingface.co/course/chapter1/1)，来更深入地了解在这里介绍到的概念。

在开始之前，确保你已经安装了所有必要的库：

```bash
!pip install transformers datasets evaluate accelerate
```

你还需要安装喜欢的机器学习框架：

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

使用 [`pipeline`] 是利用预训练模型进行推理的最简单的方式。你能够将 [`pipeline`] 开箱即用地用于跨不同模态的多种任务。来看看它支持的任务列表：

| **任务**                     | **描述**                            | **模态**        | **Pipeline**                       |
|------------------------------|-----------------------------------|-----------------|-----------------------------------------------|
| 文本分类                      | 为给定的文本序列分配一个标签                    | NLP             | pipeline(task="sentiment-analysis")           |
| 文本生成                      | 根据给定的提示生成文本                       | NLP             | pipeline(task="text-generation")              |
| 命名实体识别                  | 为序列里的每个 token 分配一个标签（人, 组织, 地址等等） | NLP             | pipeline(task="ner")                          |
| 问答系统                      | 通过给定的上下文和问题, 在文本中提取答案             | NLP             | pipeline(task="question-answering")           |
| 掩盖填充                      | 预测出正确的在序列中被掩盖的token               | NLP             | pipeline(task="fill-mask")                    |
| 文本摘要                      | 为文本序列或文档生成总结                      | NLP             | pipeline(task="summarization")                |
| 文本翻译                      | 将文本从一种语言翻译为另一种语言                  | NLP             | pipeline(task="translation")                  |
| 图像分类                      | 为图像分配一个标签                         | Computer vision | pipeline(task="image-classification")         |
| 图像分割                      | 为图像中每个独立的像素分配标签（支持语义、全景和实例分割）     | Computer vision | pipeline(task="image-segmentation")           |
| 目标检测                      | 预测图像中目标对象的边界框和类别                  | Computer vision | pipeline(task="object-detection")             |
| 音频分类                      | 给音频文件分配一个标签                       | Audio           | pipeline(task="audio-classification")         |
| 自动语音识别                   | 将音频文件中的语音提取为文本                    | Audio           | pipeline(task="automatic-speech-recognition") |
| 视觉问答                      | 给定一个图像和一个问题，正确地回答有关图像的问题          | Multimodal      | pipeline(task="vqa")                          |

创建一个 [`pipeline`] 实例并且指定你想要将它用于的任务，就可以开始了。你可以将 [`pipeline`] 用于任何一个上面提到的任务，如果想知道支持的任务的完整列表，可以查阅 [pipeline API 参考](./main_classes/pipelines)。不过, 在这篇教程中，你将把 [`pipeline`] 用在一个情感分析示例上：

```py
>>> from transformers import pipeline

>>> classifier = pipeline("sentiment-analysis")
```

[`pipeline`] 会下载并缓存一个用于情感分析的默认的[预训练模型](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english)和分词器。现在你可以在目标文本上使用 `classifier` 了：

```py
>>> classifier("We are very happy to show you the 🤗 Transformers library.")
[{'label': 'POSITIVE', 'score': 0.9998}]
```

如果你有不止一个输入，可以把所有输入放入一个列表然后传给[`pipeline`]，它将会返回一个字典列表：

```py
>>> results = classifier(["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."])
>>> for result in results:
...     print(f"label: {result['label']}, with score: {round(result['score'], 4)}")
label: POSITIVE, with score: 0.9998
label: NEGATIVE, with score: 0.5309
```

[`pipeline`] 也可以为任何你喜欢的任务遍历整个数据集。在下面这个示例中，让我们选择自动语音识别作为我们的任务：

```py
>>> import torch
>>> from transformers import pipeline

>>> speech_recognizer = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")
```

加载一个你想遍历的音频数据集（查阅 🤗 Datasets [快速开始](https://huggingface.co/docs/datasets/quickstart#audio) 获得更多信息）。比如，加载 [MInDS-14](https://huggingface.co/datasets/PolyAI/minds14) 数据集：

```py
>>> from datasets import load_dataset, Audio

>>> dataset = load_dataset("PolyAI/minds14", name="en-US", split="train")  # doctest: +IGNORE_RESULT
```

你需要确保数据集中的音频的采样率与 [`facebook/wav2vec2-base-960h`](https://huggingface.co/facebook/wav2vec2-base-960h) 训练用到的音频的采样率一致：

```py
>>> dataset = dataset.cast_column("audio", Audio(sampling_rate=speech_recognizer.feature_extractor.sampling_rate))
```

当调用 `"audio"` 列时, 音频文件将会自动加载并重采样。
从前四个样本中提取原始波形数组，将它作为列表传给 pipeline：

```py
>>> result = speech_recognizer(dataset[:4]["audio"])
>>> print([d["text"] for d in result])
['I WOULD LIKE TO SET UP A JOINT ACCOUNT WITH MY PARTNER HOW DO I PROCEED WITH DOING THAT', "FODING HOW I'D SET UP A JOIN TO HET WITH MY WIFE AND WHERE THE AP MIGHT BE", "I I'D LIKE TOY SET UP A JOINT ACCOUNT WITH MY PARTNER I'M NOT SEEING THE OPTION TO DO IT ON THE AP SO I CALLED IN TO GET SOME HELP CAN I JUST DO IT OVER THE PHONE WITH YOU AND GIVE YOU THE INFORMATION OR SHOULD I DO IT IN THE AP AND I'M MISSING SOMETHING UQUETTE HAD PREFERRED TO JUST DO IT OVER THE PHONE OF POSSIBLE THINGS", 'HOW DO I THURN A JOIN A COUNT']
```

对于输入非常庞大的大型数据集（比如语音或视觉），你会想到使用一个生成器，而不是一个将所有输入都加载进内存的列表。查阅 [pipeline API 参考](./main_classes/pipelines) 来获取更多信息。

### 在 pipeline 中使用另一个模型和分词器

[`pipeline`] 可以容纳 [Hub](https://huggingface.co/models) 中的任何模型，这让 [`pipeline`] 更容易适用于其他用例。比如，你想要一个能够处理法语文本的模型，就可以使用 Hub 上的标记来筛选出合适的模型。靠前的筛选结果会返回一个为情感分析微调的多语言的 [BERT 模型](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment)，你可以将它用于法语文本：

```py
>>> model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
```

<frameworkcontent>
<pt>
使用 [`AutoModelForSequenceClassification`] 和 [`AutoTokenizer`] 来加载预训练模型和它关联的分词器（更多信息可以参考下一节的 `AutoClass`）：

```py
>>> from transformers import AutoTokenizer, AutoModelForSequenceClassification

>>> model = AutoModelForSequenceClassification.from_pretrained(model_name)
>>> tokenizer = AutoTokenizer.from_pretrained(model_name)
```
</pt>
<tf>
使用 [`TFAutoModelForSequenceClassification`] 和 [`AutoTokenizer`] 来加载预训练模型和它关联的分词器（更多信息可以参考下一节的 `TFAutoClass`）：

```py
>>> from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

>>> model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
>>> tokenizer = AutoTokenizer.from_pretrained(model_name)
```
</tf>
</frameworkcontent>

在 [`pipeline`] 中指定模型和分词器，现在你就可以在法语文本上使用 `classifier` 了：

```py
>>> classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
>>> classifier("Nous sommes très heureux de vous présenter la bibliothèque 🤗 Transformers.")
[{'label': '5 stars', 'score': 0.7273}]
```

如果你没有找到适合你的模型，就需要在你的数据上微调一个预训练模型了。查看 [微调教程](./training) 来学习怎样进行微调。最后，微调完模型后，考虑一下在 Hub 上与社区 [分享](./model_sharing) 这个模型，把机器学习普及到每一个人! 🤗

## AutoClass

<Youtube id="AhChOFRegn4"/>

在幕后，是由 [`AutoModelForSequenceClassification`] 和 [`AutoTokenizer`] 一起支持你在上面用到的 [`pipeline`]。[AutoClass](./model_doc/auto) 是一个能够通过预训练模型的名称或路径自动查找其架构的快捷方式。你只需要为你的任务选择合适的 `AutoClass` 和它关联的预处理类。

让我们回过头来看上一节的示例，看看怎样使用 `AutoClass` 来重现使用 [`pipeline`] 的结果。

### AutoTokenizer

分词器负责预处理文本，将文本转换为用于输入模型的数字数组。有多个用来管理分词过程的规则，包括如何拆分单词和在什么样的级别上拆分单词（在 [分词器总结](./tokenizer_summary) 学习更多关于分词的信息）。要记住最重要的是你需要实例化的分词器要与模型的名称相同, 来确保和模型训练时使用相同的分词规则。

使用 [`AutoTokenizer`] 加载一个分词器:

```py
>>> from transformers import AutoTokenizer

>>> model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
>>> tokenizer = AutoTokenizer.from_pretrained(model_name)
```

将文本传入分词器：

```py
>>> encoding = tokenizer("We are very happy to show you the 🤗 Transformers library.")
>>> print(encoding)
{'input_ids': [101, 11312, 10320, 12495, 19308, 10114, 11391, 10855, 10103, 100, 58263, 13299, 119, 102],
 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
```

分词器返回了含有如下内容的字典:

* [input_ids](./glossary#input-ids)：用数字表示的 token。
* [attention_mask](.glossary#attention-mask)：应该关注哪些 token 的指示。

分词器也可以接受列表作为输入，并填充和截断文本，返回具有统一长度的批次：

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

查阅[预处理](./preprocessing)教程来获得有关分词的更详细的信息，以及如何使用 [`AutoFeatureExtractor`] 和 [`AutoProcessor`] 来处理图像，音频，还有多模式输入。

</Tip>

### AutoModel

<frameworkcontent>
<pt>
🤗 Transformers 提供了一种简单统一的方式来加载预训练的实例. 这表示你可以像加载 [`AutoTokenizer`] 一样加载 [`AutoModel`]。唯一不同的地方是为你的任务选择正确的[`AutoModel`]。对于文本（或序列）分类，你应该加载[`AutoModelForSequenceClassification`]：

```py
>>> from transformers import AutoModelForSequenceClassification

>>> model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
>>> pt_model = AutoModelForSequenceClassification.from_pretrained(model_name)
```

<Tip>

通过 [任务摘要](./task_summary) 查找 [`AutoModel`] 支持的任务.

</Tip>

现在可以把预处理好的输入批次直接送进模型。你只需要通过 `**` 来解包字典:

```py
>>> pt_outputs = pt_model(**pt_batch)
```

模型在 `logits` 属性输出最终的激活结果. 在 `logits` 上应用 softmax 函数来查询概率:

```py
>>> from torch import nn

>>> pt_predictions = nn.functional.softmax(pt_outputs.logits, dim=-1)
>>> print(pt_predictions)
tensor([[0.0021, 0.0018, 0.0115, 0.2121, 0.7725],
        [0.2084, 0.1826, 0.1969, 0.1755, 0.2365]], grad_fn=<SoftmaxBackward0>)
```
</pt>
<tf>
🤗 Transformers 提供了一种简单统一的方式来加载预训练的实例。这表示你可以像加载 [`AutoTokenizer`] 一样加载 [`TFAutoModel`]。唯一不同的地方是为你的任务选择正确的 [`TFAutoModel`]，对于文本（或序列）分类，你应该加载 [`TFAutoModelForSequenceClassification`]：

```py
>>> from transformers import TFAutoModelForSequenceClassification

>>> model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
>>> tf_model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
```

<Tip>

通过 [任务摘要](./task_summary) 查找 [`AutoModel`] 支持的任务.

</Tip>

现在通过直接将字典的键传给张量，将预处理的输入批次传给模型。

```py
>>> tf_outputs = tf_model(tf_batch)
```

模型在 `logits` 属性输出最终的激活结果。在 `logits` 上应用softmax函数来查询概率：

```py
>>> import tensorflow as tf

>>> tf_predictions = tf.nn.softmax(tf_outputs.logits, axis=-1)
>>> tf_predictions  # doctest: +IGNORE_RESULT
```
</tf>
</frameworkcontent>

<Tip>

所有 🤗 Transformers 模型（PyTorch 或 TensorFlow）在最终的激活函数（比如 softmax）*之前* 输出张量，
因为最终的激活函数常常与 loss 融合。模型的输出是特殊的数据类，所以它们的属性可以在 IDE 中被自动补全。模型的输出就像一个元组或字典（你可以通过整数、切片或字符串来索引它），在这种情况下，为 None 的属性会被忽略。

</Tip>

### 保存模型

<frameworkcontent>
<pt>
当你的模型微调完成，你就可以使用 [`PreTrainedModel.save_pretrained`] 把它和它的分词器保存下来：

```py
>>> pt_save_directory = "./pt_save_pretrained"
>>> tokenizer.save_pretrained(pt_save_directory)  # doctest: +IGNORE_RESULT
>>> pt_model.save_pretrained(pt_save_directory)
```

当你准备再次使用这个模型时，就可以使用 [`PreTrainedModel.from_pretrained`] 加载它了：

```py
>>> pt_model = AutoModelForSequenceClassification.from_pretrained("./pt_save_pretrained")
```
</pt>
<tf>
当你的模型微调完成，你就可以使用 [`TFPreTrainedModel.save_pretrained`] 把它和它的分词器保存下来：

```py
>>> tf_save_directory = "./tf_save_pretrained"
>>> tokenizer.save_pretrained(tf_save_directory)  # doctest: +IGNORE_RESULT
>>> tf_model.save_pretrained(tf_save_directory)
```

当你准备再次使用这个模型时，就可以使用 [`TFPreTrainedModel.from_pretrained`] 加载它了：

```py
>>> tf_model = TFAutoModelForSequenceClassification.from_pretrained("./tf_save_pretrained")
```
</tf>
</frameworkcontent>

🤗 Transformers 有一个特别酷的功能，它能够保存一个模型，并且将它加载为 PyTorch 或 TensorFlow 模型。`from_pt` 或 `from_tf` 参数可以将模型从一个框架转换为另一个框架：

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

## 自定义模型构建

你可以修改模型的配置类来改变模型的构建方式。配置指明了模型的属性，比如隐藏层或者注意力头的数量。当你从自定义的配置类初始化模型时，你就开始自定义模型构建了。模型属性是随机初始化的，你需要先训练模型，然后才能得到有意义的结果。

通过导入 [`AutoConfig`] 来开始，之后加载你想修改的预训练模型。在 [`AutoConfig.from_pretrained`] 中，你能够指定想要修改的属性，比如注意力头的数量：

```py
>>> from transformers import AutoConfig

>>> my_config = AutoConfig.from_pretrained("distilbert/distilbert-base-uncased", n_heads=12)
```

<frameworkcontent>
<pt>
使用 [`AutoModel.from_config`] 根据你的自定义配置创建一个模型：

```py
>>> from transformers import AutoModel

>>> my_model = AutoModel.from_config(my_config)
```
</pt>
<tf>
使用 [`TFAutoModel.from_config`] 根据你的自定义配置创建一个模型：

```py
>>> from transformers import TFAutoModel

>>> my_model = TFAutoModel.from_config(my_config)
```
</tf>
</frameworkcontent>

查阅 [创建一个自定义结构](./create_a_model) 指南获取更多关于构建自定义配置的信息。

## Trainer - PyTorch 优化训练循环

所有的模型都是标准的 [`torch.nn.Module`](https://pytorch.org/docs/stable/nn.html#torch.nn.Module)，所以你可以在任何典型的训练模型中使用它们。当你编写自己的训练循环时，🤗 Transformers 为 PyTorch 提供了一个 [`Trainer`] 类，它包含了基础的训练循环并且为诸如分布式训练，混合精度等特性增加了额外的功能。

取决于你的任务, 你通常可以传递以下的参数给 [`Trainer`]：

1. [`PreTrainedModel`] 或者 [`torch.nn.Module`](https://pytorch.org/docs/stable/nn.html#torch.nn.Module)：

   ```py
   >>> from transformers import AutoModelForSequenceClassification

   >>> model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased")
   ```

2. [`TrainingArguments`] 含有你可以修改的模型超参数，比如学习率，批次大小和训练时的迭代次数。如果你没有指定训练参数，那么它会使用默认值：

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

3. 一个预处理类，比如分词器，特征提取器或者处理器：

   ```py
   >>> from transformers import AutoTokenizer

   >>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
   ```

4. 加载一个数据集：

   ```py
   >>> from datasets import load_dataset

   >>> dataset = load_dataset("rotten_tomatoes")  # doctest: +IGNORE_RESULT
   ```

5. 创建一个给数据集分词的函数，并且使用 [`~datasets.Dataset.map`] 应用到整个数据集：

   ```py
   >>> def tokenize_dataset(dataset):
   ...     return tokenizer(dataset["text"])

   >>> dataset = dataset.map(tokenize_dataset, batched=True)
   ```

6. 用来从数据集中创建批次的 [`DataCollatorWithPadding`]：

   ```py
   >>> from transformers import DataCollatorWithPadding

   >>> data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
   ```

现在把所有的类传给 [`Trainer`]：

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

一切准备就绪后，调用 [`~Trainer.train`] 进行训练：

```py
>>> trainer.train()  # doctest: +SKIP
```

<Tip>

对于像翻译或摘要这些使用序列到序列模型的任务，用 [`Seq2SeqTrainer`] 和 [`Seq2SeqTrainingArguments`] 来替代。

</Tip>

你可以通过子类化 [`Trainer`] 中的方法来自定义训练循环。这样你就可以自定义像损失函数，优化器和调度器这样的特性。查阅 [`Trainer`] 参考手册了解哪些方法能够被子类化。

另一个自定义训练循环的方式是通过[回调](./main_classes/callback)。你可以使用回调来与其他库集成，查看训练循环来报告进度或提前结束训练。回调不会修改训练循环。如果想自定义损失函数等，就需要子类化 [`Trainer`] 了。

## 使用 Tensorflow 训练

所有模型都是标准的 [`tf.keras.Model`](https://www.tensorflow.org/api_docs/python/tf/keras/Model)，所以你可以通过 [Keras](https://keras.io/) API 实现在 Tensorflow 中训练。🤗 Transformers 提供了 [`~TFPreTrainedModel.prepare_tf_dataset`] 方法来轻松地将数据集加载为 `tf.data.Dataset`，这样你就可以使用 Keras 的 [`compile`](https://keras.io/api/models/model_training_apis/#compile-method) 和 [`fit`](https://keras.io/api/models/model_training_apis/#fit-method) 方法马上开始训练。

1. 使用 [`TFPreTrainedModel`] 或者 [`tf.keras.Model`](https://www.tensorflow.org/api_docs/python/tf/keras/Model) 来开始：

   ```py
   >>> from transformers import TFAutoModelForSequenceClassification

   >>> model = TFAutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased")
   ```

2. 一个预处理类，比如分词器，特征提取器或者处理器：

   ```py
   >>> from transformers import AutoTokenizer

   >>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
   ```

3. 创建一个给数据集分词的函数

   ```py
   >>> def tokenize_dataset(dataset):
   ...     return tokenizer(dataset["text"])  # doctest: +SKIP
   ```

4. 使用 [`~datasets.Dataset.map`] 将分词器应用到整个数据集，之后将数据集和分词器传给 [`~TFPreTrainedModel.prepare_tf_dataset`]。如果你需要的话，也可以在这里改变批次大小和是否打乱数据集：

   ```py
   >>> dataset = dataset.map(tokenize_dataset)  # doctest: +SKIP
   >>> tf_dataset = model.prepare_tf_dataset(
   ...     dataset, batch_size=16, shuffle=True, tokenizer=tokenizer
   ... )  # doctest: +SKIP
   ```

5. 一切准备就绪后，调用 `compile` 和 `fit` 开始训练：

   ```py
   >>> from tensorflow.keras.optimizers import Adam

   >>> model.compile(optimizer=Adam(3e-5))
   >>> model.fit(dataset)  # doctest: +SKIP
   ```

## 接下来做什么?

现在你已经完成了 🤗 Transformers 的快速上手教程，来看看我们的指南并且学习如何做一些更具体的事情，比如写一个自定义模型，为某个任务微调一个模型以及如何使用脚本来训练模型。如果你有兴趣了解更多 🤗 Transformers 的核心章节，那就喝杯咖啡然后来看看我们的概念指南吧！
