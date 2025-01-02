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

# التلخيص (Summarization)

[[open-in-colab]]

<Youtube id="yHnr5Dk2zCI"/>

يقوم التلخيص بإنشاء نسخة مختصرة من مستند أو مقال، حيث يلتقط جميع المعلومات المهمة. بالإضافة إلى الترجمة، يعتبر التلخيص مثالاً آخر على مهمة يمكن صياغتها كتسلسل إلى تسلسل. يمكن أن يكون التلخيص:

- استخراجي: استخراج أهم المعلومات من مستند.
- تجريدي: إنشاء نص جديد يلخص أهم المعلومات.

سيوضح لك هذا الدليل كيفية:

1. ضبط دقيق [T5](https://huggingface.co/google-t5/t5-small) على مجموعة فرعية من مشاريع قوانين ولاية كاليفورنيا من مجموعة بيانات [BillSum](https://huggingface.co/datasets/billsum) للتلخيص التجريدي.
2. استخدام النموذج المضبوط بدقة للتنبؤ.

<Tip>

لمشاهدة جميع البنى ونقاط التفتيش المتوافقة مع هذه المهمة، نوصي بالتحقق من [صفحة المهمة](https://huggingface.co/tasks/summarization)

</Tip>

قبل البدء، تأكد من تثبيت جميع المكتبات الضرورية:

```bash
pip install transformers datasets evaluate rouge_score
```

نشجعك على تسجيل الدخول إلى حساب Hugging Face الخاص بك حتى تتمكن من تحميل نموذجك ومشاركته مع المجتمع. عند المطالبة، أدخل الرمز المميز لتسجيل الدخول:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## تحميل مجموعة بيانات BillSum

ابدأ بتحميل جزء صغير من بيانات مشاريع القوانين الخاصة بولاية كاليفورنيا من مجموعة بيانات BillSum في مكتبة 🤗 Datasets:

```py
>>> from datasets import load_dataset

>>> billsum = load_dataset("billsum", split="ca_test")
```

قسّم مجموعة البيانات إلى مجموعتي تدريب واختبار باستخدام الدالة [`~datasets.Dataset.train_test_split`]:

```py
>>> billsum = billsum.train_test_split(test_size=0.2)
```

ثم ألقِ نظرة على مثال:

```py
>>> billsum["train"][0]
{'summary': 'Existing law authorizes state agencies to enter into contracts for the acquisition of goods or services upon approval by the Department of General Services. Existing law sets forth various requirements and prohibitions for those contracts, including, but not limited to, a prohibition on entering into contracts for the acquisition of goods or services of $100,000 or more with a contractor that discriminates between spouses and domestic partners or same-sex and different-sex couples in the provision of benefits. Existing law provides that a contract entered into in violation of those requirements and prohibitions is void and authorizes the state or any person acting on behalf of the state to bring a civil action seeking a determination that a contract is in violation and therefore void. Under existing law, a willful violation of those requirements and prohibitions is a misdemeanor.\nThis bill would also prohibit a state agency from entering into contracts for the acquisition of goods or services of $100,000 or more with a contractor that discriminates between employees on the basis of gender identity in the provision of benefits, as specified. By expanding the scope of a crime, this bill would impose a state-mandated local program.\nThe California Constitution requires the state to reimburse local agencies and school districts for certain costs mandated by the state. Statutory provisions establish procedures for making that reimbursement.\nThis bill would provide that no reimbursement is required by this act for a specified reason.',
 'text': 'The people of the State of California do enact as follows:\n\n\nSECTION 1.\nSection 10295.35 is added to the Public Contract Code, to read:\n10295.35.\n(a) (1) Notwithstanding any other law, a state agency shall not enter into any contract for the acquisition of goods or services in the amount of one hundred thousand dollars ($100,000) or more with a contractor that, in the provision of benefits, discriminates between employees on the basis of an employee’s or dependent’s actual or perceived gender identity, including, but not limited to, the employee’s or dependent’s identification as transgender.\n(2) For purposes of this section, “contract” includes contracts with a cumulative amount of one hundred thousand dollars ($100,000) or more per contractor in each fiscal year.\n(3) For purposes of this section, an employee health plan is discriminatory if the plan is not consistent with Section 1365.5 of the Health and Safety Code and Section 10140 of the Insurance Code.\n(4) The requirements of this section shall apply only to those portions of a contractor’s operations that occur under any of the following conditions:\n(A) Within the state.\n(B) On real property outside the state if the property is owned by the state or if the state has a right to occupy the property, and if the contractor’s presence at that location is connected to a contract with the state.\n(C) Elsewhere in the United States where work related to a state contract is being performed.\n(b) Contractors shall treat as confidential, to the maximum extent allowed by law or by the requirement of the contractor’s insurance provider, any request by an employee or applicant for employment benefits or any documentation of eligibility for benefits submitted by an employee or applicant for employment.\n(c) After taking all reasonable measures to find a contractor that complies with this section, as determined by the state agency, the requirements of this section may be waived under any of the following circumstances:\n(1) There is only one prospective contractor willing to enter into a specific contract with the state agency.\n(2) The contract is necessary to respond to an emergency, as determined by the state agency, that endangers the public health, welfare, or safety, or the contract is necessary for the provision of essential services, and no entity that complies with the requirements of this section capable of responding to the emergency is immediately available.\n(3) The requirements of this section violate, or are inconsistent with, the terms or conditions of a grant, subvention, or agreement, if the agency has made a good faith attempt to change the terms or conditions of any grant, subvention, or agreement to authorize application of this section.\n(4) The contractor is providing wholesale or bulk water, power, or natural gas, the conveyance or transmission of the same, or ancillary services, as required for ensuring reliable services in accordance with good utility practice, if the purchase of the same cannot practically be accomplished through the standard competitive bidding procedures and the contractor is not providing direct retail services to end users.\n(d) (1) A contractor shall not be deemed to discriminate in the provision of benefits if the contractor, in providing the benefits, pays the actual costs incurred in obtaining the benefit.\n(2) If a contractor is unable to provide a certain benefit, despite taking reasonable measures to do so, the contractor shall not be deemed to discriminate in the provision of benefits.\n(e) (1) Every contract subject to this chapter shall contain a statement by which the contractor certifies that the contractor is in compliance with this section.\n(2) The department or other contracting agency shall enforce this section pursuant to its existing enforcement powers.\n(3) (A) If a contractor falsely certifies that it is in compliance with this section, the contract with that contractor shall be subject to Article 9 (commencing with Section 10420), unless, within a time period specified by the department or other contracting agency, the contractor provides to the department or agency proof that it has complied, or is in the process of complying, with this section.\n(B) The application of the remedies or penalties contained in Article 9 (commencing with Section 10420) to a contract subject to this chapter shall not preclude the application of any existing remedies otherwise available to the department or other contracting agency under its existing enforcement powers.\n(f) Nothing in this section is intended to regulate the contracting practices of any local jurisdiction.\n(g) This section shall be construed so as not to conflict with applicable federal laws, rules, or regulations. In the event that a court or agency of competent jurisdiction holds that federal law, rule, or regulation invalidates any clause, sentence, paragraph, or section of this code or the application thereof to any person or circumstances, it is the intent of the state that the court or agency sever that clause, sentence, paragraph, or section so that the remainder of this section shall remain in effect.\nSEC. 2.\nSection 10295.35 of the Public Contract Code shall not be construed to create any new enforcement authority or responsibility in the Department of General Services or any other contracting agency.\nSEC. 3.\nNo reimbursement is required by this act pursuant to Section 6 of Article XIII\u2009B of the California Constitution because the only costs that may be incurred by a local agency or school district will be incurred because this act creates a new crime or infraction, eliminates a crime or infraction, or changes the penalty for a crime or infraction, within the meaning of Section 17556 of the Government Code, or changes the definition of a crime within the meaning of Section 6 of Article XIII\u2009B of the California Constitution.',
 'title': 'An act to add Section 10295.35 to the Public Contract Code, relating to public contracts.'}
```

هناك مُدخلان سترغب في استخدامهما:

- `text`: نص القانون الذي سيكون مُدخلًا للنموذج.
- `summary`: نسخة مُختصرة من `text` والتي ستكون هدف النموذج.

## المعالجة المسبقة (Preprocess)

الخطوة التالية هي تحميل مجزء النصوص T5 لمعالجة `text` و `summary`:

```py
>>> from transformers import AutoTokenizer

>>> checkpoint = "google-t5/t5-small"
>>> tokenizer = AutoTokenizer.from_pretrained(checkpoint)
```

وظيفة المعالجة المسبقة التي تريد إنشاءها تحتاج إلى:

1. إضافة بادئة للمُدخل باستخدام توجيه حتى يعرف T5 أن هذه مهمة تلخيص. تتطلب بعض النماذج القادرة على مهام البرمجة اللغوية العصبية المتعددة توجيهات لمهام مُحددة.
2. استخدام مُعامل الكلمة الرئيسية `text_target` عند ترميز التصنيفات.
3. قصّ التسلسلات بحيث لا يزيد طولها عن الحد الأقصى الذي تم تعيينه بواسطة مُعامل `max_length`.

```py
>>> prefix = "summarize: "

>>> def preprocess_function(examples):
...     inputs = [prefix + doc for doc in examples["text"]]
...     model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

...     labels = tokenizer(text_target=examples["summary"], max_length=128, truncation=True)

...     model_inputs["labels"] = labels["input_ids"]
...     return model_inputs
```

لتطبيق دالة المعالجة المسبقة على مجموعة البيانات بأكملها، استخدم طريقة [`~datasets.Dataset.map`] الخاصة بـ 🤗 Datasets. يمكنك تسريع دالة `map` عن طريق تعيين `batched=True` لمعالجة عناصر متعددة من مجموعة البيانات في وقت واحد:

```py
>>> tokenized_billsum = billsum.map(preprocess_function, batched=True)
```

الآن قم بإنشاء دفعة من الأمثلة باستخدام [`DataCollatorForSeq2Seq`].  الأكثر كفاءة *الحشو الديناميكي* للجمل إلى أطول طول في دفعة أثناء عملية التجميع، بدلاً من حشو مجموعة البيانات بأكملها إلى الحد الأقصى للطول.

<frameworkcontent>
<pt>

```py
>>> from transformers import DataCollatorForSeq2Seq

>>> data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)
```
</pt>
<tf>

```py
>>> from transformers import DataCollatorForSeq2Seq

>>> data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint, return_tensors="tf")
```
</tf>
</frameworkcontent>

## التقييم (Evaluate)

يُعد تضمين مقياس أثناء التدريب مفيدًا غالبًا لتقييم أداء نموذجك. يمكنك تحميل طريقة تقييم بسرعة باستخدام مكتبة 🤗 [Evaluate](https://huggingface.co/docs/evaluate/index). لهذه المهمة، قم بتحميل مقياس [ROUGE](https://huggingface.co/spaces/evaluate-metric/rouge) (راجع [الجولة السريعة](https://huggingface.co/docs/evaluate/a_quick_tour) الخاصة بـ 🤗 Evaluate لمعرفة المزيد حول كيفية تحميل وحساب مقياس):

```py
>>> import evaluate

>>> rouge = evaluate.load("rouge")
```

ثم قم بإنشاء دالة تُمرر تنبؤاتك وتصنيفاتك إلى [`~evaluate.EvaluationModule.compute`] لحساب مقياس ROUGE:

```py
>>> import numpy as np

>>> def compute_metrics(eval_pred):
...     predictions, labels = eval_pred
...     decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
...     labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
...     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

...     result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

...     prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
...     result["gen_len"] = np.mean(prediction_lens)

...     return {k: round(v, 4) for k, v in result.items()}
```

دالة `compute_metrics` الخاصة بك جاهزة الآن، وستعود إليها عند إعداد التدريب الخاص بك.

## التدريب (Train)

<frameworkcontent>
<pt>

<Tip>

إذا لم تكن معتادًا على ضبط نموذج باستخدام [`Trainer`]، فألق نظرة على البرنامج التعليمي الأساسي [هنا](../training#train-with-pytorch-trainer)!

</Tip>

أنت جاهز لبدء تدريب نموذجك الآن! قم بتحميل T5 باستخدام [`AutoModelForSeq2SeqLM`]:

```py
>>> from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

>>> model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
```

في هذه المرحلة، لم يتبق سوى ثلاث خطوات:

1. حدد مُعامِلات التدريب الخاصة بك في [`Seq2SeqTrainingArguments`]. المعامل الوحيد المطلوب هو `output_dir` الذي يُحدد مكان حفظ نموذجك. ستدفع هذا النموذج إلى Hub عن طريق تعيين `push_to_hub=True` (تحتاج إلى تسجيل الدخول إلى Hugging Face لتحميل نموذجك). في نهاية كل حقبة، سيقوم [`Trainer`] بتقييم مقياس ROUGE وحفظ نقطة تفتيش التدريب.
2. مرر مُعامِلات التدريب إلى [`Seq2SeqTrainer`] جنبًا إلى جنب مع النموذج ومجموعة البيانات والمُحلِّل اللغوي وجامع البيانات ودالة `compute_metrics`.
3. استدعِ [`~Trainer.train`] لضبط نموذجك.

```py
>>> training_args = Seq2SeqTrainingArguments(
...     output_dir="my_awesome_billsum_model",
...     eval_strategy="epoch",
...     learning_rate=2e-5,
...     per_device_train_batch_size=16,
...     per_device_eval_batch_size=16,
...     weight_decay=0.01,
...     save_total_limit=3,
...     num_train_epochs=4,
...     predict_with_generate=True,
...     fp16=True, #change to bf16=True for XPU
...     push_to_hub=True,
... )

>>> trainer = Seq2SeqTrainer(
...     model=model,
...     args=training_args,
...     train_dataset=tokenized_billsum["train"],
...     eval_dataset=tokenized_billsum["test"],
...     processing_class=tokenizer,
...     data_collator=data_collator,
...     compute_metrics=compute_metrics,
... )

>>> trainer.train()
```

بمجرد اكتمال التدريب، شارك نموذجك مع Hub باستخدام طريقة [`~transformers.Trainer.push_to_hub`] حتى يتمكن الجميع من استخدام نموذجك:

```py
>>> trainer.push_to_hub()
```
</pt>
<tf>
<Tip>

إذا لم تكن معتادًا على ضبط نموذج باستخدام Keras، فألق نظرة على البرنامج التعليمي الأساسي [هنا](../training#train-a-tensorflow-model-with-keras)!

</Tip>
لضبط نموذج في TensorFlow، ابدأ بإعداد دالة مُحسِّن وجدول معدل التعلم وبعض معلمات التدريب:

```py
>>> from transformers import create_optimizer, AdamWeightDecay

>>> optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)
```

ثم يمكنك تحميل T5 باستخدام [`TFAutoModelForSeq2SeqLM`]:

```py
>>> from transformers import TFAutoModelForSeq2SeqLM

>>> model = TFAutoModelForSeq2SeqLM.from_pretrained(checkpoint)
```

حوّل مجموعات البيانات الخاصة بك إلى تنسيق `tf.data.Dataset` باستخدام [`~transformers.TFPreTrainedModel.prepare_tf_dataset`]:

```py
>>> tf_train_set = model.prepare_tf_dataset(
...     tokenized_billsum["train"],
...     shuffle=True,
...     batch_size=16,
...     collate_fn=data_collator,
... )

>>> tf_test_set = model.prepare_tf_dataset(
...     tokenized_billsum["test"],
...     shuffle=False,
...     batch_size=16,
...     collate_fn=data_collator,
... )
```

قم بتكوين النموذج للتدريب باستخدام [`compile`](https://keras.io/api/models/model_training_apis/#compile-method). لاحظ أن جميع نماذج Transformers لديها دالة خسارة ذات صلة بالمهمة افتراضيًا، لذلك لست بحاجة إلى تحديد واحدة ما لم تكن ترغب في ذلك:

```py
>>> import tensorflow as tf

>>> model.compile(optimizer=optimizer)  # No loss argument!
```

آخر شيئين يجب إعدادهما قبل بدء التدريب هما حساب درجة ROUGE من التنبؤات، وتوفير طريقة لدفع نموذجك إلى Hub. يتم كلاهما باستخدام [استدعاءات Keras](../main_classes/keras_callbacks).

مرر دالة `compute_metrics` الخاصة بك إلى [`~transformers.KerasMetricCallback`]:

```py
>>> from transformers.keras_callbacks import KerasMetricCallback

>>> metric_callback = KerasMetricCallback(metric_fn=compute_metrics, eval_dataset=tf_test_set)
```

حدد مكان دفع نموذجك ومُحلِّلك اللغوي في [`~transformers.PushToHubCallback`]:

```py
>>> from transformers.keras_callbacks import PushToHubCallback

>>> push_to_hub_callback = PushToHubCallback(
...     output_dir="my_awesome_billsum_model",
...     tokenizer=tokenizer,
... )
```

ثم اجمع استدعاءاتك معًا:

```py
>>> callbacks = [metric_callback, push_to_hub_callback]
```

أخيرًا، أنت جاهز لبدء تدريب نموذجك! اتصل بـ [`fit`](https://keras.io/api/models/model_training_apis/#fit-method) مع مجموعات بيانات التدريب والتحقق من الصحة وعدد الحقب واستدعاءاتك لضبط النموذج:

```py
>>> model.fit(x=tf_train_set, validation_data=tf_test_set, epochs=3, callbacks=callbacks)
```

بمجرد اكتمال التدريب، يتم تحميل نموذجك تلقائيًا إلى Hub حتى يتمكن الجميع من استخدامه!
</tf>
</frameworkcontent>

<Tip>

للحصول على مثال أكثر تعمقًا حول كيفية ضبط نموذج للتجميع، ألقِ نظرة على [دفتر ملاحظات PyTorch](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/summarization.ipynb)
أو [دفتر ملاحظات TensorFlow](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/summarization-tf.ipynb) المقابل.

</Tip>

## الاستدلال (Inference)

رائع، الآن بعد أن قمت بضبط نموذج، يمكنك استخدامه للاستدلال!

خدد بعض النصوص الذي ترغب في تلخيصها. بالنسبة لـ T5، تحتاج إلى إضافة بادئة إلى مُدخلاتك اعتمادًا على المهمة التي تعمل عليها. بالنسبة التلخيص، يجب عليك إضافة بادئة إلى مُدخلاتك كما هو موضح أدناه:

```py
>>> text = "summarize: The Inflation Reduction Act lowers prescription drug costs, health care costs, and energy costs. It's the most aggressive action on tackling the climate crisis in American history, which will lift up American workers and create good-paying, union jobs across the country. It'll lower the deficit and ask the ultra-wealthy and corporations to pay their fair share. And no one making under $400,000 per year will pay a penny more in taxes."
```

أبسط طريقة لتجربة نموذجك المضبوط للاستدلال هي استخدامه في [`pipeline`]. استخدم `pipeline` للتلخيص باستخدام نموذجك، ومرر نصك إليه:

```py
>>> from transformers import pipeline

>>> summarizer = pipeline("summarization", model="username/my_awesome_billsum_model")
>>> summarizer(text)
[{"summary_text": "The Inflation Reduction Act lowers prescription drug costs, health care costs, and energy costs. It's the most aggressive action on tackling the climate crisis in American history, which will lift up American workers and create good-paying, union jobs across the country."}]
```

يمكنك أيضًا تكرار نتائج `pipeline` يدويًا إذا أردت:

<frameworkcontent>
<pt>
قسم النص وإرجع `input_ids` كتنسورات PyTorch:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("username/my_awesome_billsum_model")
>>> inputs = tokenizer(text, return_tensors="pt").input_ids
```

استخدم طريقة [`~generation.GenerationMixin.generate`] لإنشاء التلخيص. لمزيد من التفاصيل حول استراتيجيات توليد النص المختلفة والمعلمات للتحكم في التوليد، راجع واجهة برمجة تطبيقات [توليد النص](../main_classes/text_generation).

```py
>>> from transformers import AutoModelForSeq2SeqLM

>>> model = AutoModelForSeq2SeqLM.from_pretrained("username/my_awesome_billsum_model")
>>> outputs = model.generate(inputs, max_new_tokens=100, do_sample=False)
```

فك تشفير معرفات الرموز المولدة مرة أخرى إلى نص:

```py
>>> tokenizer.decode(outputs[0], skip_special_tokens=True)
'the inflation reduction act lowers prescription drug costs, health care costs, and energy costs. it's the most aggressive action on tackling the climate crisis in american history. it will ask the ultra-wealthy and corporations to pay their fair share.'
```
</pt>
<tf>
قسم النص وإرجع `input_ids` كتنسورات TensorFlow:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("username/my_awesome_billsum_model")
>>> inputs = tokenizer(text, return_tensors="tf").input_ids
```

استخدم طريقة [`~transformers.generation_tf_utils.TFGenerationMixin.generate`] لإنشاء التلخيص. لمزيد من التفاصيل حول استراتيجيات توليد النص المختلفة والمعلمات للتحكم في التوليد، راجع واجهة برمجة تطبيقات [توليد النص](../main_classes/text_generation).

```py
>>> from transformers import TFAutoModelForSeq2SeqLM

>>> model = TFAutoModelForSeq2SeqLM.from_pretrained("username/my_awesome_billsum_model")
>>> outputs = model.generate(inputs, max_new_tokens=100, do_sample=False)
```

فك تشفير معرفات الرموز المولدة مرة أخرى إلى نص:

```py
>>> tokenizer.decode(outputs[0], skip_special_tokens=True)
'the inflation reduction act lowers prescription drug costs, health care costs, and energy costs. it's the most aggressive action on tackling the climate crisis in american history. it will ask the ultra-wealthy and corporations to pay their fair share.'
```
</tf>
</frameworkcontent>