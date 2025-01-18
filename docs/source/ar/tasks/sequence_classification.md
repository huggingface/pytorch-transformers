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

# تصنيف النص(Text classification)

[[open-in-colab]]

<Youtube id="leNG9fN9FQU"/>

تصنيف النص هو مهمة NLP شائعة حيث يُعيّن تصنيفًا أو فئة للنص. تستخدم بعض أكبر الشركات تصنيف النصوص في الإنتاج لمجموعة واسعة من التطبيقات العملية. أحد أكثر أشكال تصنيف النص شيوعًا هو تحليل المشاعر، والذي يقوم بتعيين تسمية مثل 🙂 إيجابية، 🙁 سلبية، أو 😐 محايدة لتسلسل نصي.

سيوضح لك هذا الدليل كيفية:

1. ضبط [DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased) على مجموعة بيانات [IMDb](https://huggingface.co/datasets/imdb) لتحديد ما إذا كانت مراجعة الفيلم إيجابية أو سلبية.
2. استخدام نموذج الضبط الدقيق للتنبؤ.

<Tip>

لرؤية جميع البنى ونقاط التحقق المتوافقة مع هذه المهمة، نوصي بالتحقق من [صفحة المهمة](https://huggingface.co/tasks/text-classification).

</Tip>

قبل أن تبدأ، تأكد من تثبيت جميع المكتبات الضرورية:

```bash
pip install transformers datasets evaluate accelerate
```

نحن نشجعك على تسجيل الدخول إلى حساب Hugging Face الخاص بك حتى تتمكن من تحميل ومشاركة نموذجك مع المجتمع. عند المطالبة، أدخل رمزك لتسجيل الدخول:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## تحميل مجموعة بيانات IMDb

ابدأ بتحميل مجموعة بيانات IMDb من مكتبة 🤗 Datasets:

```py
>>> from datasets import load_dataset

>>> imdb = load_dataset("imdb")
```

ثم ألق نظرة على مثال:

```py
>>> imdb["test"][0]
{
    "label": 0,
    "text": "I love sci-fi and am willing to put up with a lot. Sci-fi movies/TV are usually underfunded, under-appreciated and misunderstood. I tried to like this, I really did, but it is to good TV sci-fi as Babylon 5 is to Star Trek (the original). Silly prosthetics, cheap cardboard sets, stilted dialogues, CG that doesn't match the background, and painfully one-dimensional characters cannot be overcome with a 'sci-fi' setting. (I'm sure there are those of you out there who think Babylon 5 is good sci-fi TV. It's not. It's clichéd and uninspiring.) While US viewers might like emotion and character development, sci-fi is a genre that does not take itself seriously (cf. Star Trek). It may treat important issues, yet not as a serious philosophy. It's really difficult to care about the characters here as they are not simply foolish, just missing a spark of life. Their actions and reactions are wooden and predictable, often painful to watch. The makers of Earth KNOW it's rubbish as they have to always say \"Gene Roddenberry's Earth...\" otherwise people would not continue watching. Roddenberry's ashes must be turning in their orbit as this dull, cheap, poorly edited (watching it without advert breaks really brings this home) trudging Trabant of a show lumbers into space. Spoiler. So, kill off a main character. And then bring him back as another actor. Jeeez! Dallas all over again.",
}
```

هناك حقولان في هذه المجموعة من البيانات:

- `text`: نص مراجعة الفيلم.
- `label`: قيمة إما `0` لمراجعة سلبية أو `1` لمراجعة إيجابية.

## المعالجة المسبقة(Preprocess)

الخطوة التالية هي تحميل المُجزِّئ النص DistilBERT لتهيئة لحقل `text`:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
```

أنشئ دالة لتهيئة حقل `text` وتقصير السلاسل النصية بحيث لا يتجاوز طولها الحد الأقصى لإدخالات DistilBERT:

```py
>>> def preprocess_function(examples):
...     return tokenizer(examples["text"], truncation=True)
```

لتطبيق دالة التهيئة على مجموعة البيانات بأكملها، استخدم دالة 🤗 Datasets [`~datasets.Dataset.map`] . يمكنك تسريع `map` باستخدام `batched=True` لمعالجة دفعات من البيانات:

```py
tokenized_imdb = imdb.map(preprocess_function, batched=True)
```

الآن قم بإنشاء دفعة من الأمثلة باستخدام [`DataCollatorWithPadding`].  الأكثر كفاءة هو استخدام الحشو الديناميكي لجعل الجمل متساوية في الطول داخل كل دفعة، بدلًا من حشو كامل البيانات إلى الحد الأقصى للطول.

<frameworkcontent>
<pt>

```py
>>> from transformers import DataCollatorWithPadding

>>> data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```
</pt>
<tf>

```py
>>> from transformers import DataCollatorWithPadding

>>> data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")
```
</tf>
</frameworkcontent>

## التقييم(Evaluate)

يُعدّ تضمين مقياس أثناء التدريب مفيدًا لتقييم أداء النموذج. يمكنك تحميل طريقة تقييم بسرعة باستخدام مكتبة 🤗 [Evaluate](https://huggingface.co/docs/evaluate/index) . بالنسبة لهذه المهمة، قم بتحميل مقياس [الدقة](https://huggingface.co/spaces/evaluate-metric/accuracy) (راجع جولة 🤗 Evaluate [السريعة](https://huggingface.co/docs/evaluate/a_quick_tour) لمعرفة المزيد حول كيفية تحميل وحساب مقياس):

```py
>>> import evaluate

>>> accuracy = evaluate.load("accuracy")
```

ثم أنشئ دالة تقوم بتمرير تنبؤاتك وتصنيفاتك إلى [`~evaluate.EvaluationModule.compute`] لحساب الدقة:

```py
>>> import numpy as np

>>> def compute_metrics(eval_pred):
...     predictions, labels = eval_pred
...     predictions = np.argmax(predictions, axis=1)
...     return accuracy.compute(predictions=predictions, references=labels)
```

دالة `compute_metrics` جاهزة الآن، وستعود إليها عند إعداد التدريب.

## التدريب(Train)

قبل أن تبدأ في تدريب نموذجك، قم بإنشاء خريطة من المعرفات المتوقعة إلى تسمياتها باستخدام `id2label` و `label2id`:

```py
>>> id2label = {0: "NEGATIVE", 1: "POSITIVE"}
>>> label2id = {"NEGATIVE": 0, "POSITIVE": 1}
```

<frameworkcontent>
<pt>
<Tip>

إذا لم تكن على دراية بضبط نموذج دقيق باستخدام [`Trainer`], فالق نظرة على البرنامج التعليمي الأساسي [هنا](../training#train-with-pytorch-trainer)!

</Tip>

أنت مستعد الآن لبدء تدريب نموذجك! قم بتحميل DistilBERT مع [`AutoModelForSequenceClassification`] جنبًا إلى جنب مع عدد التصنيفات المتوقعة، وتصنيفات الخرائط:

```py
>>> from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

>>> model = AutoModelForSequenceClassification.from_pretrained(
...     "distilbert/distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id
... )
```

في هذه المرحلة، هناك ثلاث خطوات فقط متبقية:

1.  حدد مُعامِلات التدريب في [`TrainingArguments`]. المُعامل المطلوب الوحيد هو `output_dir`، لتحديد مكان حفظ النموذج. يمكنك رفع النموذج إلى Hub بتعيين `push_to_hub=True` (يجب تسجيل الدخول إلى Hugging Face لرفع النموذج). سيقوم `Trainer` بتقييم الدقة وحفظ نقاط التحقق في نهاية كل حقبة.
2.  مرر مُعامِلات التدريب إلى `Trainer` مع النموذج، ومجموعة البيانات، والمحلل اللغوي، ومُجمِّع البيانات، ووظيفة `compute_metrics`.
3.  استدعِ [`~Trainer.train`] لضبط النموذج.

```py
>>> training_args = TrainingArguments(
...     output_dir="my_awesome_model",
...     learning_rate=2e-5,
...     per_device_train_batch_size=16,
...     per_device_eval_batch_size=16,
...     num_train_epochs=2,
...     weight_decay=0.01,
...     eval_strategy="epoch",
...     save_strategy="epoch",
...     load_best_model_at_end=True,
...     push_to_hub=True,
... )

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=tokenized_imdb["train"],
...     eval_dataset=tokenized_imdb["test"],
...     processing_class=tokenizer,
...     data_collator=data_collator,
...     compute_metrics=compute_metrics,
... )

>>> trainer.train()
```

<Tip>

يستخدم [`Trainer`] الحشو الديناميكي افتراضيًا عند تمرير `tokenizer` إليه. في هذه الحالة،  لا تحتاج لتحديد مُجمِّع البيانات صراحةً. 

</Tip>

بعد اكتمال التدريب، شارك نموذجك على Hub باستخدام الطريقة [`~transformers.Trainer.push_to_hub`] ليستخدمه الجميع:

```py
>>> trainer.push_to_hub()
```
</pt>
<tf>
<Tip>

إذا لم تكن على دراية بضبط نموذج باستخدام Keras، قم بالاطلاع على البرنامج التعليمي الأساسي [هنا](../training#train-a-tensorflow-model-with-keras)!

</Tip>
لضبط نموذج في TensorFlow، ابدأ بإعداد دالة المحسن، وجدول معدل التعلم، وبعض معلمات التدريب:

```py
>>> from transformers import create_optimizer
>>> import tensorflow as tf

>>> batch_size = 16
>>> num_epochs = 5
>>> batches_per_epoch = len(tokenized_imdb["train"]) // batch_size
>>> total_train_steps = int(batches_per_epoch * num_epochs)
>>> optimizer, schedule = create_optimizer(init_lr=2e-5, num_warmup_steps=0, num_train_steps=total_train_steps)
```

ثم يمكنك تحميل DistilBERT مع [`TFAutoModelForSequenceClassification`] بالإضافة إلى عدد التصنيفات المتوقعة، وتعيينات التسميات:

```py
>>> from transformers import TFAutoModelForSequenceClassification

>>> model = TFAutoModelForSequenceClassification.from_pretrained(
...     "distilbert/distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id
... )
```

قم بتحويل مجموعات بياناتك إلى تنسيق `tf.data.Dataset` باستخدام [`~transformers.TFPreTrainedModel.prepare_tf_dataset`]:

```py
>>> tf_train_set = model.prepare_tf_dataset(
...     tokenized_imdb["train"],
...     shuffle=True,
...     batch_size=16,
...     collate_fn=data_collator,
... )

>>> tf_validation_set = model.prepare_tf_dataset(
...     tokenized_imdb["test"],
...     shuffle=False,
...     batch_size=16,
...     collate_fn=data_collator,
... )
```

قم بتهيئة النموذج للتدريب باستخدام [`compile`](https://keras.io/api/models/model_training_apis/#compile-method). لاحظ أن جميع نماذج Transformers لديها دالة خسارة ذات صلة بالمهمة بشكل افتراضي، لذلك لا تحتاج إلى تحديد واحدة ما لم ترغب في ذلك:

```py
>>> import tensorflow as tf

>>> model.compile(optimizer=optimizer)  # No loss argument!
```

آخر أمرين يجب إعدادهما قبل بدء التدريب هو حساب الدقة من التوقعات، وتوفير طريقة لدفع نموذجك إلى Hub. يتم ذلك باستخدام [Keras callbacks](../main_classes/keras_callbacks).

قم بتمرير دالة `compute_metrics` الخاصة بك إلى [`~transformers.KerasMetricCallback`]:

```py
>>> from transformers.keras_callbacks import KerasMetricCallback

>>> metric_callback = KerasMetricCallback(metric_fn=compute_metrics, eval_dataset=tf_validation_set)
```

حدد مكان دفع نموذجك والمجزئ اللغوي في [`~transformers.PushToHubCallback`]:

```py
>>> from transformers.keras_callbacks import PushToHubCallback

>>> push_to_hub_callback = PushToHubCallback(
...     output_dir="my_awesome_model",
...     tokenizer=tokenizer,
... )
```

ثم اجمع الاستدعاءات معًا:

```py
>>> callbacks = [metric_callback, push_to_hub_callback]
```

أخيرًا، أنت مستعد لبدء تدريب نموذجك! قم باستدعاء [`fit`](https://keras.io/api/models/model_training_apis/#fit-method) مع مجموعات بيانات التدريب والتحقق، وعدد الحقبات، واستدعاءاتك لضبط النموذج:

```py
>>> model.fit(x=tf_train_set, validation_data=tf_validation_set, epochs=3, callbacks=callbacks)
```

بمجرد اكتمال التدريب، يتم تحميل نموذجك تلقائيًا إلى Hub حتى يتمكن الجميع من استخدامه!
</tf>
</frameworkcontent>

<Tip>

للحصول على مثال أكثر عمقًا حول كيفية ضبط نموذج لتصنيف النصوص، قم بالاطلاع على الدفتر المقابل
[دفتر PyTorch](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification.ipynb)
أو [دفتر TensorFlow](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification-tf.ipynb).

</Tip>

## الاستدلال(Inference)

رائع، الآن بعد أن قمت بضبط نموذج، يمكنك استخدامه للاستدلال!

احصل على بعض النصوص التي ترغب في إجراء الاستدلال عليها:

```py
>>> text = "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."
```

أسهل طريقة لتجربة النموذج المضبوط للاستدلال هي استخدامه ضمن [`pipeline`]. قم بإنشاء `pipeline` لتحليل المشاعر مع نموذجك، ومرر نصك إليه:

```py
>>> from transformers import pipeline

>>> classifier = pipeline("sentiment-analysis", model="stevhliu/my_awesome_model")
>>> classifier(text)
[{'label': 'POSITIVE', 'score': 0.9994940757751465}]
```

يمكنك أيضًا تكرار نتائج `pipeline` يدويًا إذا أردت:

<frameworkcontent>
<pt>
قم يتجزئة النص وإرجاع تنسورات PyTorch:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("stevhliu/my_awesome_model")
>>> inputs = tokenizer(text, return_tensors="pt")
```

مرر المدخلات إلى النموذج واسترجع `logits`:

```py
>>> from transformers import AutoModelForSequenceClassification

>>> model = AutoModelForSequenceClassification.from_pretrained("stevhliu/my_awesome_model")
>>> with torch.no_grad():
...     logits = model(**inputs).logits
```

استخرج الفئة ذات الاحتمالية الأعلى، واستخدم `id2label` لتحويلها إلى تصنيف نصي:

```py
>>> predicted_class_id = logits.argmax().item()
>>> model.config.id2label[predicted_class_id]
'POSITIVE'
```
</pt>
<tf>
قم بتحليل النص وإرجاع تنسيقات TensorFlow:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("stevhliu/my_awesome_model")
>>> inputs = tokenizer(text, return_tensors="tf")
```

قم بتمرير مدخلاتك إلى النموذج وإرجاع `logits`:

```py
>>> from transformers import TFAutoModelForSequenceClassification

>>> model = TFAutoModelForSequenceClassification.from_pretrained("stevhliu/my_awesome_model")
>>> logits = model(**inputs).logits
```

استخرج الفئة ذات الاحتمالية الأعلى، واستخدم `id2label` لتحويلها إلى تصنيف نصي:

```py
>>> predicted_class_id = int(tf.math.argmax(logits, axis=-1)[0])
>>> model.config.id2label[predicted_class_id]
'POSITIVE'
```
</tf>
</frameworkcontent>
