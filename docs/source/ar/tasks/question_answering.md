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

#  الإجابة على الأسئلة (Question answering)

[[open-in-colab]]

<Youtube id="ajPx5LwJD-I"/>

تُقدّم مهام الإجابة على الأسئلة إجابةً بناءً على سؤال. إذا سبق لك أن سألت مساعدًا افتراضيًا مثل Alexa أو Siri أو Google عن حالة الطقس، فأنت قد استخدمت نموذج للإجابة على الأسئلة من قبل. هناك نوعان شائعان لمهام الإجابة على الأسئلة:

- الاستخراجية: استخراج الإجابة من السياق المحدد.
- التلخيصية: إنشاء إجابة من السياق تجيب على السؤال بشكل صحيح.

سيوضح لك هذا الدليل كيفية:

1. ضبط [DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased) على مجموعة بيانات [SQuAD](https://huggingface.co/datasets/squad) للإجابة على الأسئلة الاستخراجية.
2. استخدام النموذج المضبوط للاستدلال.

<Tip>

لمشاهدة جميع الهياكل والنسخ المتوافقة مع هذه المهمة، نوصي بالرجوع إلى [صفحة المهمة](https://huggingface.co/tasks/question-answering)

</Tip>

قبل البدء، تأكد من تثبيت جميع المكتبات الضرورية:

```bash
pip install transformers datasets evaluate
```

نشجعك على تسجيل الدخول إلى حساب Hugging Face الخاص بك حتى تتمكن من تحميل نموذجك ومشاركته مع المجتمع. عند المطالبة، أدخل الرمز المميز الخاص بك لتسجيل الدخول:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## تحميل مجموعة بيانات SQuAD

ابدأ بتحميل جزء أصغر من مجموعة بيانات SQuAD من مكتبة 🤗 Datasets. سيتيح لك ذلك فرصة للتجربة والتحقق من عمل كل شيء بشكل صحيح قبل قضاء المزيد من الوقت في التدريب على مجموعة البيانات الكاملة.

```py
>>> from datasets import load_dataset

>>> squad = load_dataset("squad", split="train[:5000]")
```

قم بتقسيم تقسيم `train` لمجموعة البيانات إلى مجموعة تدريب واختبار باستخدام طريقة [`~datasets.Dataset.train_test_split`]:

```py
>>> squad = squad.train_test_split(test_size=0.2)
```

ثم ألق نظرة على مثال:

```py
>>> squad["train"][0]
{'answers': {'answer_start': [515], 'text': ['Saint Bernadette Soubirous']},
 'context': 'Architecturally, the school has a Catholic character. Atop the Main Building\'s gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.',
 'id': '5733be284776f41900661182',
 'question': 'To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?',
 'title': 'University_of_Notre_Dame'
}
```

هناك العديد من الحقول المهمة هنا:

- `answers`: موقع بداية الرمز المميز للإجابة ونص الإجابة.
- `context`: معلومات أساسية يحتاج النموذج إلى استخراج الإجابة منها.
- `question`: السؤال الذي يجب على النموذج الإجابة عليه.

## المعالجة المسبقة (Preprocess)

<Youtube id="qgaM0weJHpA"/>

الخطوة التالية هي تحميل المحلل اللغوى DistilBERT لمعالجة حقلي `question` و `context`:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
```

هناك بعض خطوات المعالجة المسبقة الخاصة بمهام الإجابة على الأسئلة التي يجب أن تكون على دراية بها:

1. قد تحتوي بعض الأمثلة في مجموعة البيانات على `context` طويلًا يتجاوز الحد الأقصى لطول مدخل النموذج. للتعامل مع النصوص الأطول، يتم اقتطاع `context` فقط عن طريق تعيين `truncation="only_second"`.
2. بعد ذلك، يتم تحديد مواضع بداية ونهاية الإجابة في `context` الأصلي عن طريق تعيين
   `return_offset_mapping=True`.
3. باستخدام التعيين، يمكن الآن تحديد رموز بداية ونهاية الإجابة. استخدم طريقة [`~tokenizers.Encoding.sequence_ids`]
   لتحديد أجزاء الإزاحة التي تتوافق مع `question` و `context`.

فيما يلي كيفية إنشاء دالة لقص وتعيين رموز البداية والنهاية لـ `answer` إلى `context`:

```py
>>> def preprocess_function(examples):
...     questions = [q.strip() for q in examples["question"]]
...     inputs = tokenizer(
...         questions,
...         examples["context"],
...         max_length=384,
...         truncation="only_second",
...         return_offsets_mapping=True,
...         padding="max_length",
...     )

...     offset_mapping = inputs.pop("offset_mapping")
...     answers = examples["answers"]
...     start_positions = []
...     end_positions = []

...     for i, offset in enumerate(offset_mapping):
...         answer = answers[i]
...         start_char = answer["answer_start"][0]
...         end_char = answer["answer_start"][0] + len(answer["text"][0])
...         sequence_ids = inputs.sequence_ids(i)

...         # Find the start and end of the context
...         idx = 0
...         while sequence_ids[idx] != 1:
...             idx += 1
...         context_start = idx
...         while sequence_ids[idx] == 1:
...             idx += 1
...         context_end = idx - 1

...         # If the answer is not fully inside the context, label it (0, 0)
...         if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
...             start_positions.append(0)
...             end_positions.append(0)
...         else:
...             # Otherwise it's the start and end token positions
...             idx = context_start
...             while idx <= context_end and offset[idx][0] <= start_char:
...                 idx += 1
...             start_positions.append(idx - 1)

...             idx = context_end
...             while idx >= context_start and offset[idx][1] >= end_char:
...                 idx -= 1
...             end_positions.append(idx + 1)

...     inputs["start_positions"] = start_positions
...     inputs["end_positions"] = end_positions
...     return inputs
```

لتطبيق المعالجة المسبقة على كامل مجموعة البيانات، استخدم [`~datasets.Dataset.map`] من مكتبة 🤗 Datasets. يمكنك تسريع دالة `map` عن طريق تعيين `batched=True` لمعالجة عناصر متعددة من مجموعة البيانات دفعة واحدة. قم بإزالة أي أعمدة لا تحتاجها:

```py
>>> tokenized_squad = squad.map(preprocess_function, batched=True, remove_columns=squad["train"].column_names)
```

الآن قم بإنشاء دفعة من الأمثلة باستخدام [`DefaultDataCollator`]. بخلاف مجمّعات البيانات الأخرى في 🤗 Transformers، لا يطبق [`DefaultDataCollator`] أي معالجة مسبقة إضافية مثل الحشو.

<frameworkcontent>
<pt>
 
```py
>>> from transformers import DefaultDataCollator

>>> data_collator = DefaultDataCollator()
```
</pt>
<tf>
 
```py
>>> from transformers import DefaultDataCollator

>>> data_collator = DefaultDataCollator(return_tensors="tf")
```
</tf>
</frameworkcontent>

## التدريب (Train)

<frameworkcontent>
<pt>

<Tip>

إذا لم تكن معتادًا على ضبط نموذج باستخدام [`Trainer`], ألق نظرة على البرنامج التعليمي الأساسي [هنا](../training#train-with-pytorch-trainer)!

</Tip>

أنت جاهز لبدء تدريب نموذجك الآن! قم بتحميل DistilBERT باستخدام [`AutoModelForQuestionAnswering`]:

```py
>>> from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer

>>> model = AutoModelForQuestionAnswering.from_pretrained("distilbert/distilbert-base-uncased")
```

في هذه المرحلة، تبقى ثلاث خطوات فقط:

1. حدد المعاملات الفائقة للتدريب في [`TrainingArguments`]. المعامل الوحيد المطلوب هو `output_dir` الذي يحدد مكان حفظ نموذجك. ستدفع هذا النموذج إلى Hub عن طريق تعيين `push_to_hub=True` (يجب عليك تسجيل الدخول إلى Hugging Face لتحميل نموذجك).
2. مرر معاملات التدريب إلى [`Trainer`] جنبًا إلى جنب مع النموذج، ومجموعة البيانات، والمُحلّل النصي، ومُجمّع البيانات.
3. استدعِ ـ [`~Trainer.train`] لضبط النموذج.

```py
>>> training_args = TrainingArguments(
...     output_dir="my_awesome_qa_model",
...     eval_strategy="epoch",
...     learning_rate=2e-5,
...     per_device_train_batch_size=16,
...     per_device_eval_batch_size=16,
...     num_train_epochs=3,
...     weight_decay=0.01,
...     push_to_hub=True,
... )

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=tokenized_squad["train"],
...     eval_dataset=tokenized_squad["test"],
...     processing_class=tokenizer,
...     data_collator=data_collator,
... )

>>> trainer.train()
```

بمجرد اكتمال التدريب، شارك نموذجك في Hub باستخدام الدالة [`~transformers.Trainer.push_to_hub`] حتى يتمكن الجميع من استخدام نموذجك:

```py
>>> trainer.push_to_hub()
```
</pt>
<tf>
 
<Tip>

إذا لم تكن معتادًا على ضبط نموذج باستخدام Keras، فألق نظرة على البرنامج التعليمي الأساسي [هنا](../training#train-a-tensorflow-model-with-keras)!

</Tip>
لضبط نموذج في TensorFlow، ابدأ بإعداد دالة مُحسِّن، وجدول معدل التعلم، وبعض المعاملات الفائقة للتدريب:

```py
>>> from transformers import create_optimizer

>>> batch_size = 16
>>> num_epochs = 2
>>> total_train_steps = (len(tokenized_squad["train"]) // batch_size) * num_epochs
>>> optimizer, schedule = create_optimizer(
...     init_lr=2e-5,
...     num_warmup_steps=0,
...     num_train_steps=total_train_steps,
... )
```

ثم يمكنك تحميل DistilBERT باستخدام [`TFAutoModelForQuestionAnswering`]:

```py
>>> from transformers import TFAutoModelForQuestionAnswering

>>> model = TFAutoModelForQuestionAnswering.from_pretrained("distilbert/distilbert-base-uncased")
```

حوّل مجموعات البيانات الخاصة بك إلى تنسيق `tf.data.Dataset` باستخدام [`~transformers.TFPreTrainedModel.prepare_tf_dataset`]:

```py
>>> tf_train_set = model.prepare_tf_dataset(
...     tokenized_squad["train"],
...     shuffle=True,
...     batch_size=16,
...     collate_fn=data_collator,
... )

>>> tf_validation_set = model.prepare_tf_dataset(
...     tokenized_squad["test"],
...     shuffle=False,
...     batch_size=16,
...     collate_fn=data_collator,
... )
```

قم بتكوين النموذج للتدريب باستخدام [`compile`](https://keras.io/api/models/model_training_apis/#compile-method):

```py
>>> import tensorflow as tf

>>> model.compile(optimizer=optimizer)
```

آخر شيء يجب إعداده قبل بدء التدريب هو توفير طريقة لدفع نموذجك إلى Hub. يمكن القيام بذلك عن طريق تحديد مكان دفع نموذجك ومعالجك المعجمي في [`~transformers.PushToHubCallback`]:

```py
>>> from transformers.keras_callbacks import PushToHubCallback

>>> callback = PushToHubCallback(
...     output_dir="my_awesome_qa_model",
...     tokenizer=tokenizer,
... )
```

أخيرًا، أنت جاهز لبدء تدريب نموذجك! اتصل بـ [`fit`](https://keras.io/api/models/model_training_apis/#fit-method) مع مجموعات بيانات التدريب والتحقق من الصحة، وعدد العهود، ومعاودة الاتصال الخاصة بك لضبط النموذج:

```py
>>> model.fit(x=tf_train_set, validation_data=tf_validation_set, epochs=3, callbacks=[callback])
```
بمجرد اكتمال التدريب، يتم تحميل نموذجك تلقائيًا إلى Hub حتى يتمكن الجميع من استخدامه!
</tf>
</frameworkcontent>


<Tip>

للحصول على مثال أكثر تعمقًا حول كيفية ضبط نموذج للإجابة على الأسئلة، ألق نظرة على [دفتر ملاحظات PyTorch](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering.ipynb) المقابل
أو [دفتر ملاحظات TensorFlow](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering-tf.ipynb).

</Tip>

## التقييم (Evaluate)

يتطلب التقييم للإجابة على الأسئلة قدرًا كبيرًا من المعالجة اللاحقة. لتوفير وقتك، يتخطى هذا الدليل خطوة التقييم. لا يزال [`Trainer`] يحسب خسارة التقييم أثناء التدريب، مما يعني أنك لست تجهل تمامًا أداء نموذجك.

إذا كان لديك المزيد من الوقت وتهتم بكيفية تقييم نموذجك للإجابة على الأسئلة، فألق نظرة على فصل [الإجابة على الأسئلة](https://huggingface.co/course/chapter7/7?fw=pt#post-processing) من دورة 🤗 Hugging Face!

## الاستدلال (Inference)

رائع، الآن بعد أن قمت بضبط نموذج، يمكنك استخدامه للاستدلال!

حدد سؤالًا وسياقًا ليقوم النموذج بالتنبؤ بالإجابة عليه:

```py
>>> question = "How many programming languages does BLOOM support?"
>>> context = "BLOOM has 176 billion parameters and can generate text in 46 languages natural languages and 13 programming languages."
```

أبسط طريقة لتجربة نموذجك المُدرَّب للاستدلال هي استخدامه في [`pipeline`]. قم بإنشاء كائن لـ `pipeline` للإجابة على الأسئلة باستخدام نموذجك، ومرِّر النص إليه:

```py
>>> from transformers import pipeline

>>> question_answerer = pipeline("question-answering", model="my_awesome_qa_model")
>>> question_answerer(question=question, context=context)
{'score': 0.2058267742395401,
 'start': 10,
 'end': 95,
 'answer': '176 مليار معامل ويمكنه إنشاء نصوص بـ 46 لغة طبيعية و 13'}
```

يمكنك أيضًا تكرار نتائج `pipeline` يدويًا إذا أردت:

<frameworkcontent>
<pt>
 
 قسّم النص وأرجع تنسورات PyTorch:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("my_awesome_qa_model")
>>> inputs = tokenizer(question, context, return_tensors="pt")
```

مرر مدخلاتك إلى النموذج وأرجع `logits`:

```py
>>> import torch
>>> from transformers import AutoModelForQuestionAnswering

>>> model = AutoModelForQuestionAnswering.from_pretrained("my_awesome_qa_model")
>>> with torch.no_grad():
...     outputs = model(**inputs)
```

احصل على أعلى احتمال من مخرجات النموذج لموضعي البداية والنهاية:

```py
>>> answer_start_index = outputs.start_logits.argmax()
>>> answer_end_index = outputs.end_logits.argmax()
```

استخلاص الإجابة من الرموز المتوقعة:

```py
>>> predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
>>> tokenizer.decode(predict_answer_tokens)
'176 billion parameters and can generate text in 46 languages natural languages and 13'
```
</pt>
<tf>
قم بتحليل النص المعجمي وأعد موترات TensorFlow:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("my_awesome_qa_model")
>>> inputs = tokenizer(question, context, return_tensors="tf")
```

مرر مدخلاتك إلى النموذج وأعد `logits`:

```py
>>> from transformers import TFAutoModelForQuestionAnswering

>>> model = TFAutoModelForQuestionAnswering.from_pretrained("my_awesome_qa_model")
>>> outputs = model(**inputs)
```

احصل على أعلى احتمال من مخرجات النموذج لموضعي البداية والنهاية:

```py
>>> answer_start_index = int(tf.math.argmax(outputs.start_logits, axis=-1)[0])
>>> answer_end_index = int(tf.math.argmax(outputs.end_logits, axis=-1)[0])
```

استخلاص الإجابة من الرموز المتوقعة:

```py
>>> predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
>>> tokenizer.decode(predict_answer_tokens)
'176 billion parameters and can generate text in 46 languages natural languages and 13'
```
</tf>
</frameworkcontent>
