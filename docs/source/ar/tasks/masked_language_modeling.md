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

# نمذجة اللغة المقنعة (Masked language modeling)

[[open-in-colab]]

<Youtube id="mqElG5QJWUg"/>

تتنبأ نمذجة اللغة المقنعة برمز مقنع في تسلسل، ويمكن للنموذج الانتباه إلى الرموز بشكل ثنائي الاتجاه. هذا
يعني أن النموذج لديه إمكانية الوصول الكاملة إلى الرموز الموجودة على اليسار واليمين. تعد نمذجة اللغة المقنعة ممتازة للمهام التي
تتطلب فهمًا سياقيًا جيدًا لتسلسل كامل. BERT هو مثال على نموذج لغة مقنع.

سيوضح لك هذا الدليل كيفية:

1. تكييف [DistilRoBERTa](https://huggingface.co/distilbert/distilroberta-base) على مجموعة فرعية [r/askscience](https://www.reddit.com/r/askscience/) من مجموعة بيانات [ELI5](https://huggingface.co/datasets/eli5).
2. استخدام نموذج المدرب الخاص بك للاستدلال.

<Tip>

لمعرفة جميع البنى والنسخ المتوافقة مع هذه المهمة، نوصي بالتحقق من [صفحة المهمة](https://huggingface.co/tasks/fill-mask)

</Tip>

قبل أن تبدأ، تأكد من تثبيت جميع المكتبات الضرورية:

```bash
pip install transformers datasets evaluate
```

نحن نشجعك على تسجيل الدخول إلى حساب Hugging Face الخاص بك حتى تتمكن من تحميل ومشاركة نموذجك مع المجتمع. عندما تتم مطالبتك، أدخل رمزك لتسجيل الدخول:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## تحميل مجموعة بيانات ELI5

ابدأ بتحميل أول 5000 مثال من مجموعة بيانات [ELI5-Category](https://huggingface.co/datasets/eli5_category) باستخدام مكتبة 🤗 Datasets. سيعطيك هذا فرصة للتجربة والتأكد من أن كل شيء يعمل قبل قضاء المزيد من الوقت في التدريب على مجموعة البيانات الكاملة.

```py
>>> from datasets import load_dataset

>>> eli5 = load_dataset("eli5_category", split="train[:5000]")
```

قم بتقسيم مجموعة البيانات `train` إلى مجموعتي تدريب واختبار باستخدام الدالة [`~datasets.Dataset.train_test_split`]:

```py
>>> eli5 = eli5.train_test_split(test_size=0.2)
```

ثم ألق نظرة على مثال:

```py
>>> eli5["train"][0]
{'q_id': '7h191n',
 'title': 'What does the tax bill that was passed today mean? How will it affect Americans in each tax bracket?',
 'selftext': '',
 'category': 'Economics',
 'subreddit': 'explainlikeimfive',
 'answers': {'a_id': ['dqnds8l', 'dqnd1jl', 'dqng3i1', 'dqnku5x'],
  'text': ["The tax bill is 500 pages long and there were a lot of changes still going on right to the end. It's not just an adjustment to the income tax brackets, it's a whole bunch of changes. As such there is no good answer to your question. The big take aways are: - Big reduction in corporate income tax rate will make large companies very happy. - Pass through rate change will make certain styles of business (law firms, hedge funds) extremely happy - Income tax changes are moderate, and are set to expire (though it's the kind of thing that might just always get re-applied without being made permanent) - People in high tax states (California, New York) lose out, and many of them will end up with their taxes raised.",
   'None yet. It has to be reconciled with a vastly different house bill and then passed again.',
   'Also: does this apply to 2017 taxes? Or does it start with 2018 taxes?',
   'This article explains both the House and senate bills, including the proposed changes to your income taxes based on your income level. URL_0'],
  'score': [21, 19, 5, 3],
  'text_urls': [[],
   [],
   [],
   ['https://www.investopedia.com/news/trumps-tax-reform-what-can-be-done/']]},
 'title_urls': ['url'],
 'selftext_urls': ['url']}
```

على الرغم من أن هذا قد يبدو كثيرًا، إلا أنك مهتم حقًا بحقل `text`. ما هو رائع حول مهام نمذجة اللغة هو أنك لا تحتاج إلى تسميات (تُعرف أيضًا باسم المهمة غير الخاضعة للإشراف) لأن الكلمة التالية *هي* التسمية.

## معالجة مسبقة (Preprocess)

<Youtube id="8PmhEIXhBvI"/>

بالنسبة لنمذجة اللغة المقنعة، فإن الخطوة التالية هي تحميل معالج DistilRoBERTa لمعالجة حقل `text` الفرعي:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilroberta-base")
```

ستلاحظ من المثال أعلاه، أن حقل `text` موجود بالفعل داخل `answers`. هذا يعني أنك ستحتاج إلى استخراج حقل `text` الفرعي من بنيته المضمنة باستخدام الدالة [`flatten`](https://huggingface.co/docs/datasets/process#flatten):

```py
>>> eli5 = eli5.flatten()
>>> eli5["train"][0]
{'q_id': '7h191n',
 'title': 'What does the tax bill that was passed today mean? How will it affect Americans in each tax bracket?',
 'selftext': '',
 'category': 'Economics',
 'subreddit': 'explainlikeimfive',
 'answers.a_id': ['dqnds8l', 'dqnd1jl', 'dqng3i1', 'dqnku5x'],
 'answers.text': ["The tax bill is 500 pages long and there were a lot of changes still going on right to the end. It's not just an adjustment to the income tax brackets, it's a whole bunch of changes. As such there is no good answer to your question. The big take aways are: - Big reduction in corporate income tax rate will make large companies very happy. - Pass through rate change will make certain styles of business (law firms, hedge funds) extremely happy - Income tax changes are moderate, and are set to expire (though it's the kind of thing that might just always get re-applied without being made permanent) - People in high tax states (California, New York) lose out, and many of them will end up with their taxes raised.",
  'None yet. It has to be reconciled with a vastly different house bill and then passed again.',
  'Also: does this apply to 2017 taxes? Or does it start with 2018 taxes?',
  'This article explains both the House and senate bills, including the proposed changes to your income taxes based on your income level. URL_0'],
 'answers.score': [21, 19, 5, 3],
 'answers.text_urls': [[],
  [],
  [],
  ['https://www.investopedia.com/news/trumps-tax-reform-what-can-be-done/']],
 'title_urls': ['url'],
 'selftext_urls': ['url']}
```

كل حقل فرعي هو الآن عمود منفصل كما هو موضح بواسطة بادئة `answers`، وحقل `text` هو قائمة الآن. بدلاً من
معالجة كل جملة بشكل منفصل، قم بتحويل القائمة إلى سلسلة حتى تتمكن من معالجتها بشكل مشترك.

هنا أول دالة معالجة مسبقة لربط قائمة السلاسل لكل مثال ومعالجة النتيجة:

```py
>>> def preprocess_function(examples):
...     return tokenizer([" ".join(x) for x in examples["answers.text"]])
```

لتطبيق دالة المعالجة المسبقة على مجموعة البيانات بأكملها، استخدم الدالة 🤗 Datasets [`~datasets.Dataset.map`]. يمكنك تسريع دالة `map` عن طريق تعيين `batched=True` لمعالجة عدة عناصر في وقت واحد، وزيادة عدد العمليات باستخدام `num_proc`. احذف أي أعمدة غير ضرورية:

```py
>>> tokenized_eli5 = eli5.map(
...     preprocess_function,
...     batched=True,
...     num_proc=4,
...     remove_columns=eli5["train"].column_names,
... )
```


تحتوي مجموعة البيانات هذه على تسلسلات رمزية، ولكن بعضها أطول من الطول الأقصى للمدخلات للنموذج.

يمكنك الآن استخدام دالة معالجة مسبقة ثانية لـ:
- تجميع جميع التسلسلات
- تقسيم التسلسلات المجمّعة إلى أجزاء أقصر محددة بـ `block_size`، والتي يجب أن تكون أقصر من الحد الأقصى لطول المدخلات ومناسبة لذاكرة GPU.

```py
>>> block_size = 128

>>> def group_texts(examples):
...     # تجميع جميع النصوص.
...     concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
...     total_length = len(concatenated_examples[list(examples.keys())[0]])
...     # نتجاهل الجزء المتبقي الصغير، يمكننا إضافة الحشو إذا كان النموذج يدعمه بدلاً من هذا الإسقاط، يمكنك
...     # تخصيص هذا الجزء حسب احتياجاتك.
...     if total_length >= block_size:
...         total_length = (total_length // block_size) * block_size
...     # تقسيمها إلى أجزاء بحجم block_size.
...     result = {
...         k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
...         for k, t in concatenated_examples.items()
...     }
...     return result
```

طبق دالة `group_texts` على مجموعة البيانات بأكملها:

```py
>>> lm_dataset = tokenized_eli5.map(group_texts, batched=True, num_proc=4)
```

الآن، قم بإنشاء دفعة من الأمثلة باستخدام [`DataCollatorForLanguageModeling`]. من الأكثر كفاءة أن تقوم بـ *الحشو الديناميكي* ليصل طولها إلى أطول جملة في الدفعة أثناء التجميع، بدلاً من حشو مجموعة البيانات بأكملها إلى الطول الأقصى.

<frameworkcontent>
<pt>

استخدم رمز نهاية التسلسل كرمز الحشو وحدد `mlm_probability` لحجب الرموز عشوائياً كل مرة تكرر فيها البيانات:

```py
>>> from transformers import DataCollatorForLanguageModeling

>>> tokenizer.pad_token = tokenizer.eos_token
>>> data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
```
</pt>
<tf>

استخدم رمز نهاية التسلسل كرمز الحشو وحدد `mlm_probability` لحجب الرموز عشوائياً كل مرة تكرر فيها البيانات:

```py
>>> from transformers import DataCollatorForLanguageModeling

>>> data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15, return_tensors="tf")
```
</tf>
</frameworkcontent>

## التدريب (Train)

<frameworkcontent>
<pt>

<Tip>

إذا لم تكن على دراية بتعديل نموذج باستخدام [`Trainer`], ألق نظرة على الدليل الأساسي [هنا](../training#train-with-pytorch-trainer)!

</Tip>

أنت مستعد الآن لبدء تدريب نموذجك! قم بتحميل DistilRoBERTa باستخدام [`AutoModelForMaskedLM`]:

```py
>>> from transformers import AutoModelForMaskedLM

>>> model = AutoModelForMaskedLM.from_pretrained("distilbert/distilroberta-base")
```

في هذه المرحلة، تبقى ثلاث خطوات فقط:

1. حدد معلمات التدريب الخاصة بك في [`TrainingArguments`]. المعلمة الوحيدة المطلوبة هي `output_dir` والتي تحدد مكان حفظ نموذجك. ستقوم بدفع هذا النموذج إلى Hub عن طريق تعيين `push_to_hub=True` (يجب أن تكون مسجلاً الدخول إلى Hugging Face لتحميل نموذجك).
2. قم بتمرير معلمات التدريب إلى [`Trainer`] مع النموذج، ومجموعات البيانات، ومجمّع البيانات.
3. قم باستدعاء [`~Trainer.train`] لتعديل نموذجك.

```py
>>> training_args = TrainingArguments(
...     output_dir="my_awesome_eli5_mlm_model",
...     eval_strategy="epoch",
...     learning_rate=2e-5,
...     num_train_epochs=3,
...     weight_decay=0.01,
...     push_to_hub=True,
... )

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=lm_dataset["train"],
...     eval_dataset=lm_dataset["test"],
...     data_collator=data_collator,
...     tokenizer=tokenizer,
... )

>>> trainer.train()
```

بمجرد اكتمال التدريب، استخدم طريقة [`~transformers.Trainer.evaluate`] لتقييم النموذج والحصول على مقياس
    الحيرة:

```py
>>> import math

>>> eval_results = trainer.evaluate()
>>> print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
Perplexity: 8.76
```

ثم شارك نموذجك على Hub باستخدام طريقة [`~transformers.Trainer.push_to_hub`] حتى يتمكن الجميع من استخدام نموذجك:

```py
>>> trainer.push_to_hub()
```
</pt>
<tf>
<Tip>

إذا لم تكن على دراية بتعديل نموذج باستخدام Keras، ألق نظرة على الدليل الأساسي [هنا](../training#train-a-tensorflow-model-with-keras)!

</Tip>
لتعديل نموذج في TensorFlow، ابدأ بإعداد دالة محسن، وجدول معدل التعلم، وبعض معلمات التدريب:

```py
>>> from transformers import create_optimizer, AdamWeightDecay

>>> optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)
```

ثم يمكنك تحميل DistilRoBERTa باستخدام [`TFAutoModelForMaskedLM`]:

```py
>>> from transformers import TFAutoModelForMaskedLM

>>> model = TFAutoModelForMaskedLM.from_pretrained("distilbert/distilroberta-base")
```

قم بتحويل مجموعات بياناتك إلى تنسيق `tf.data.Dataset` باستخدام [`~transformers.TFPreTrainedModel.prepare_tf_dataset`]:

```py
>>> tf_train_set = model.prepare_tf_dataset(
...     lm_dataset["train"],
...     shuffle=True,
...     batch_size=16,
...     collate_fn=data_collator,
... )

>>> tf_test_set = model.prepare_tf_dataset(
...     lm_dataset["test"],
...     shuffle=False,
...     batch_size=16,
...     collate_fn=data_collator,
... )
```

قم بتهيئة النموذج للتدريب باستخدام [`compile`](https://keras.io/api/models/model_training_apis/#compile-method). لاحظ أن نماذج Transformers لديها جميعها دالة خسارة افتراضية ذات صلة بالمهمة، لذلك لا تحتاج إلى تحديد واحدة ما لم تكن تريد ذلك:

```py
>>> import tensorflow as tf

>>> model.compile(optimizer=optimizer)  # لا توجد حجة للخسارة!
```

يمكن القيام بذلك عن طريق تحديد مكان دفع نموذجك ومعالج الرموز في [`~transformers.PushToHubCallback`]:

```py
>>> from transformers.keras_callbacks import PushToHubCallback

>>> callback = PushToHubCallback(
...     output_dir="my_awesome_eli5_mlm_model",
...     tokenizer=tokenizer,
... )
```

أخيراً، أنت مستعد لبدء تدريب نموذجك! قم باستدعاء [`fit`](https://keras.io/api/models/model_training_apis/#fit-method) مع مجموعات بيانات التدريب والتحقق، وعدد العصور، والتعليقات الخاصة بك لتعديل النموذج:

```py
>>> model.fit(x=tf_train_set, validation_data=tf_test_set, epochs=3, callbacks=[callback])
```

بمجرد اكتمال التدريب، يتم تحميل نموذجك تلقائياً إلى Hub حتى يتمكن الجميع من استخدامه!
</tf>
</frameworkcontent>

<Tip>

لمثال أكثر تفصيلاً حول كيفية تعديل نموذج للنمذجة اللغوية المقنعة، ألق نظرة على الدفتر المقابل
[دفتر PyTorch](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb)
أو [دفتر TensorFlow](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling-tf.ipynb).

</Tip>

## الاستدلال

رائع، الآن بعد أن قمت بتعديل نموذج، يمكنك استخدامه للاستدلال!

جهّز بعض النصوص التي تريد أن يملأ النموذج الفراغات فيها، واستخدم الرمز الخاص `<mask>` للإشارة إلى الفراغ:

```py
>>> text = "The Milky Way is a <mask> galaxy."
```

أبسط طريقة لتجربة نموذجك المعدل للاستدلال هي استخدامه في [`pipeline`]. قم بإنشاء كائن  `pipeline` لملء الفراغ مع نموذجك، ومرر نصك إليه. إذا أردت، يمكنك استخدام معلمة `top_k` لتحديد عدد التنبؤات التي تريد إرجاعها:

```py
>>> from transformers import pipeline

>>> mask_filler = pipeline("fill-mask", "username/my_awesome_eli5_mlm_model")
>>> mask_filler(text, top_k=3)
[{'score': 0.5150994658470154,
  'token': 21300,
  'token_str': ' spiral',
  'sequence': 'The Milky Way is a spiral galaxy.'},
 {'score': 0.07087188959121704,
  'token': 2232,
  'token_str': ' massive',
  'sequence': 'The Milky Way is a massive galaxy.'},
 {'score': 0.06434620916843414,
  'token': 650,
  'token_str': ' small',
  'sequence': 'The Milky Way is a small galaxy.'}]
```

<frameworkcontent>
<pt>
قم بتجزئة النص وإرجاع `input_ids` كمتجهات PyTorch. ستحتاج أيضًا إلى تحديد موضع رمز `<mask>`:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("username/my_awesome_eli5_mlm_model")
>>> inputs = tokenizer(text, return_tensors="pt")
>>> mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
```

قم بتمرير المدخلات إلى النموذج وإرجاع `logits` للرمز المقنع:

```py
>>> from transformers import AutoModelForMaskedLM

>>> model = AutoModelForMaskedLM.from_pretrained("username/my_awesome_eli5_mlm_model")
>>> logits = model(**inputs).logits
>>> mask_token_logits = logits[0, mask_token_index, :]
```

ثم قم بإرجاع الرموز الثلاثة المقنعة ذات الاحتمالية الأعلى وطباعتها:

```py
>>> top_3_tokens = torch.topk(mask_token_logits, 3, dim=1).indices[0].tolist()

>>> for token in top_3_tokens:
...     print(text.replace(tokenizer.mask_token, tokenizer.decode([token])))
The Milky Way is a spiral galaxy.
The Milky Way is a massive galaxy.
The Milky Way is a small galaxy.
```
</pt>
<tf>
قم بتقسيم النص إلى رموز وإرجاع `input_ids` كـ TensorFlow tensors. ستحتاج أيضًا إلى تحديد موضع رمز `<mask>`:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("username/my_awesome_eli5_mlm_model")
>>> inputs = tokenizer(text, return_tensors="tf")
>>> mask_token_index = tf.where(inputs["input_ids"] == tokenizer.mask_token_id)[0, 1]
```

قم بتمرير المدخلات إلى النموذج وإرجاع `logits` للرمز المقنع:

```py
>>> from transformers import TFAutoModelForMaskedLM

>>> model = TFAutoModelForMaskedLM.from_pretrained("username/my_awesome_eli5_mlm_model")
>>> logits = model(**inputs).logits
>>> mask_token_logits = logits[0, mask_token_index, :]
```

ثم قم بإرجاع الرموز الثلاثة المقنعة ذات الاحتمالية الأعلى وطباعتها:

```py
>>> top_3_tokens = tf.math.top_k(mask_token_logits, 3).indices.numpy()

>>> for token in top_3_tokens:
...     print(text.replace(tokenizer.mask_token, tokenizer.decode([token])))
The Milky Way is a spiral galaxy.
The Milky Way is a massive galaxy.
The Milky Way is a small galaxy.
```
</tf>
</frameworkcontent>