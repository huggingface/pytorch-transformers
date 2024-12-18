# جولة سريعة

[[open-in-colab]]

ابدأ رحلتك مع مكتبة 🤗 Transformers! سواء كنت مطورًا أو مستخدمًا عاديًا، ستساعدك هذه الجولة السريعة على البدء وستُظهر لك كيفية استخدام [`pipeline`] للاستنتاج، وتحميل نموذج مُدرب مسبقًا ومعالج مُسبق مع [AutoClass](./model_doc/auto)، وتدريب نموذج بسرعة باستخدام PyTorch أو TensorFlow. إذا كنت مبتدئًا، نوصي بالاطلاع على دروسنا أو [الدورة](https://huggingface.co/course/chapter1/1) للحصول على شرح أكثر تعمقًا للمفاهيم المقدمة هنا.

قبل البدء، تأكد من تثبيت جميع المكتبات الضرورية:

```bash
!pip install transformers datasets evaluate accelerate
```

ستحتاج أيضًا إلى تثبيت إطار عمل التعلم الآلي المفضل لديك:

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

## خط الأنابيب

<Youtube id="tiZFewofSLM"/>

يمثل [`pipeline`] أسهل وأسرع طريقة لاستخدام نموذج مُدرب مسبقًا للاستنتاج. يمكنك استخدام [`pipeline`] جاهزًا للعديد من المهام عبر طرق مختلفة، والتي يظهر بعضها في الجدول أدناه:

<Tip>

للاطلاع على القائمة الكاملة للمهام المتاحة، راجع [مرجع واجهة برمجة التطبيقات الخاصة بخط الأنابيب](./main_classes/pipelines).

</Tip>

<div dir="rtl">

| **المهمة**                     | **الوصف**                                                                                              | **الطريقة**    | **معرف خط الأنابيب**                       |
|------------------------------|--------------------------------------------------------------------------------------------------------------|-----------------|-----------------------------------------------|
| تصنيف النص          | تعيين تسمية إلى تسلسل نص معين                                                                   | NLP             | pipeline(task=“sentiment-analysis”)           |
| توليد النص              | توليد نص بناءً على موجه معين                                                                                 | NLP             | pipeline(task=“text-generation”)              |
| تلخيص                | توليد ملخص لتسلسل نص أو مستند                                                         | NLP             | pipeline(task=“summarization”)                |
| تصنيف الصور         | تعيين تسمية لصورة معينة                                                                                   | رؤية حاسوبية | pipeline(task=“image-classification”)         |
| تجزئة الصورة           | تعيين تسمية لكل بكسل فردي في الصورة (يدعم التجزئة الدلالية، والمجملة، وتجزئة مثيلات) | رؤية حاسوبية | pipeline(task=“image-segmentation”)           |
| اكتشاف الأشياء             | التنبؤ بحدود الأشياء وفئاتها في صورة معينة                                                | رؤية حاسوبية | pipeline(task=“object-detection”)             |
| تصنيف الصوت         | تعيين تسمية لبيانات صوتية معينة                                                                            | صوتي           | pipeline(task=“audio-classification”)         |
| التعرف على الكلام التلقائي | نسخ الكلام إلى نص                                                                                  | صوتي           | pipeline(task=“automatic-speech-recognition”) |
| الإجابة على الأسئلة البصرية    | الإجابة على سؤال حول الصورة، مع إعطاء صورة وسؤال                                             | متعدد الوسائط      | pipeline(task=“vqa”)                          |
| الإجابة على أسئلة المستندات  | الإجابة على سؤال حول المستند، مع إعطاء مستند وسؤال                                        | متعدد الوسائط      | pipeline(task="document-question-answering")  |
| كتابة تعليق على الصورة             | إنشاء تعليق على صورة معينة                                                                         | متعدد الوسائط      | pipeline(task="image-to-text")                |

</div>
ابدأ بإنشاء مثيل من [`pipeline`] وتحديد المهمة التي تريد استخدامه لها. في هذا الدليل، ستستخدم خط الأنابيب للتحليل النصي كنموذج:

```py
>>> from transformers import pipeline

>>> classifier = pipeline("sentiment-analysis")
```

يقوم [`pipeline`] بتنزيل وتخزين نسخة احتياطية من نموذج افتراضي [مُدرب مسبقًا](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english) ومعالج للتحليل النصي. الآن يمكنك استخدام `classifier` على النص المستهدف:

```py
>>> classifier("We are very happy to show you the 🤗 Transformers library.")
[{'label': 'POSITIVE', 'score': 0.9998}]
```

إذا كان لديك أكثر من إدخال واحد، قم بتمرير إدخالاتك كقائمة إلى [`pipeline`] لإرجاع قائمة من القواميس:

```py
>>> results = classifier(["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."])
>>> for result in results:
...     print(f"label: {result['label']}, with score: {round(result['score'], 4)}")
label: POSITIVE, with score: 0.9998
label: NEGATIVE, with score: 0.5309
```
يمكن لخط الأنابيب أيضًا أن يتنقل خلال مجموعة بيانات كاملة لأي مهمة تريدها. كمثال على ذلك، دعنا نختار التعرف على الكلام التلقائي كمهمة لنا:

```py
>>> import torch
>>> from transformers import pipeline

>>> speech_recognizer = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")
```

قم بتحميل مجموعة بيانات صوتية (راجع دليل البدء السريع لـ 🤗 Datasets [Quick Start](https://huggingface.co/docs/datasets/quickstart#audio) للحصول على مزيد من التفاصيل) التي تريد التنقل خلالها. على سبيل المثال، قم بتحميل مجموعة بيانات [MInDS-14](https://huggingface.co/datasets/PolyAI/minds14):

```py
>>> from datasets import load_dataset, Audio

>>> dataset = load_dataset("PolyAI/minds14", name="en-US", split="train")  # doctest: +IGNORE_RESULT
```

يجب التأكد من أن نفس الجودة الصوتية (معدل أخذ العينات) لمجموعة البيانات يتطابق مع معدل أخذ العينات الذي تم تدريب [`facebook/wav2vec2-base-960h`](https://huggingface.co/facebook/wav2vec2-base-960h) عليه:

```py
>>> dataset = dataset.cast_column("audio", Audio(sampling_rate=speech_recognizer.feature_extractor.sampling_rate))
```

يتم تحميل الملفات الصوتية وإعادة تشكيلها تلقائيًا عند استدعاء العمود "audio".
استخرج المصفوفات الموجية الخام من أول 4 عينات ومررها كقائمة إلى خط الأنابيب:

```py
>>> result = speech_recognizer(dataset[:4]["audio"])
>>> print([d["text"] for d in result])
['I WOULD LIKE TO SET UP A JOINT ACCOUNT WITH MY PARTNER HOW DO I PROCEED WITH DOING THAT', "FONDERING HOW I'D SET UP A JOIN TO HELL T WITH MY WIFE AND WHERE THE AP MIGHT BE", "I I'D LIKE TOY SET UP A JOINT ACCOUNT WITH MY PARTNER I'M NOT SEEING THE OPTION TO DO IT ON THE APSO I CALLED IN TO GET SOME HELP CAN I JUST DO IT OVER THE PHONE WITH YOU AND GIVE YOU THE INFORMATION OR SHOULD I DO IT IN THE AP AN I'M MISSING SOMETHING UQUETTE HAD PREFERRED TO JUST DO IT OVER THE PHONE OF POSSIBLE THINGS", 'HOW DO I FURN A JOINA COUT']
```

بالنسبة لمجموعات البيانات الكبيرة  التي تحتوي على مدخلات ضخمة (كما هو الحال في البيانات الصوتية أو المرئية)، يفضل تمرير مولد (generator) بدلاً من قائمة لتحميل جميع المدخلات في الذاكرة دفعة واحدة. راجع [مرجع واجهة برمجة التطبيقات الخاصة بخط الأنابيب](./main_classes/pipelines) للحصول على مزيد من المعلومات.

### ااستخدم نموذجًا ومجزئًا آخرين في خط الأنابيب

يمكن لخط الأنابيب [`pipeline`] استيعاب أي نموذج من [Hub](https://huggingface.co/models)، مما يسهل التكيف مع حالات الاستخدام الأخرى. على سبيل المثال، إذا كنت تريد نموذجًا قادرًا على التعامل مع النص الفرنسي، فاستخدم العلامات على Hub لفلتره نموذج مناسب. تعيد النتيجة الأولى المرشحة نموذج BERT متعدد اللغات [BERT model](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment) الذي تم ضبطه مسبقًا للتحليل المشاعر والذي يمكنك استخدامه للنص الفرنسي:

```py
>>> model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
```

<frameworkcontent>
<pt>
استخدم [`AutoModelForSequenceClassification`] و [`AutoTokenizer`] لتحميل النموذج المُدرب مسبقًا ومعالجته المرتبط به (مزيد من المعلومات حول `AutoClass` في القسم التالي):

```py
>>> from transformers import AutoTokenizer, AutoModelForSequenceClassification

>>> model = AutoModelForSequenceClassification.from_pretrained(model_name)
>>> tokenizer = AutoTokenizer.from_pretrained(model_name)
```
</pt>
<tf>
استخدم [`TFAutoModelForSequenceClassification`] و [`AutoTokenizer`] لتحميل النموذج المُدرب مسبقًا ومعالجته المرتبط به (مزيد من المعلومات حول `TFAutoClass` في القسم التالي):

```py
>>> from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

>>> model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
>>> tokenizer = AutoTokenizer.from_pretrained(model_name)
```
</tf>
</frameworkcontent>

حدد النموذج والمعالج في [`pipeline`]. الآن يمكنك تطبيق `classifier` على النص الفرنسي:

```py
>>> classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
>>> classifier("Nous sommes très heureux de vous présenter la bibliothèque 🤗 Transformers.")
[{'label': '5 stars', 'score': 0.7273}]
```
إذا لم تجد نموذجًا جاهزًا يناسب مهمتك، فستحتاج إلى ضبط نموذج مُدرب مسبقًا على بياناتك. اطلع على [دليل الضبط الدقيق](./training) للتعرف على كيفية القيام بذلك. وبعد ضبط نموذجك المُدرب مسبقًا، يرجى مراعاة [المشاركة](./model_sharing) النموذج مع المجتمع على Hub لمساعدة الجميع في مجال التعلم الآلي! 🤗

## AutoClass

<Youtube id="AhChOFRegn4"/>

في الخلفية، تعمل فئتا [`AutoModelForSequenceClassification`] و [`AutoTokenizer`] معًا لتشغيل دالة pipeline()  الذي استخدمتها أعلاه. تعتبر [AutoClass](./model_doc/auto) اختصارًا يقوم تلقائيًا باسترداد بنية نموذج مُدرب مسبقًا من اسمه أو مساره. كل ما عليك فعله هو تحديد فئة `AutoClass` المناسبة لمهمتك وفئة المعالجة المرتبطة بها.

لنعد إلى المثال من القسم السابق ولنرى كيف يمكنك استخدام `AutoClass` لتكرار نتائج خط الأنابيب.

### المجزئ التلقائي (AutoTokenizer)

يتولى المجزئ مسؤولية تحويل النص إلى مصفوفة من الأرقام (رموز) يمكن للنموذج فهمها ومعالجتها. هناك قواعد متعددة تحكم عملية التجزئة، بما في ذلك كيفية تقسيم كلمة وما هو المستوى الذي يجب أن تقسيم الكلمات عنده (تعرف على المزيد حول المعالجة في [ملخص المجزئ](./tokenizer_summary)). أهم شيء يجب تذكره هو أنك تحتاج إلى إنشاء مثيل للمجزئ بنفس اسم النموذج لضمان استخدامك لقواعد التجزئة نفسها التي تم تدريب النموذج عليها.

قم بتحميل المجزئ باستخدام [`AutoTokenizer`]:

```py
>>> from transformers import AutoTokenizer

>>> model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
>>> tokenizer = AutoTokenizer.from_pretrained(model_name)
```

مرر نصك إلى المجزئ:

```py
>>> encoding = tokenizer("We are very happy to show you the 🤗 Transformers library.")
>>> print(encoding)
{'input_ids': [101, 11312, 10320, 12495, 19308, 10114, 11391, 10855, 10103, 100, 58263, 13299, 119, 102],
 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
```

يعيد المجزئ قاموسًا يحتوي على:

* [input_ids](./glossary#input-ids): التمثيلات الرقمية لرموزك.
* [attention_mask](./glossary#attention-mask): تشير إلى الرموز التي يجب الانتباه بها.

يمكن المجزئ أيضًا قبول قائمة من المدخلات، ويقوم بـ "حشو" و"تقصير" النص لإرجاع كدفعة بطول موحد:

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

اطلع على [الدليل التمهيدي للمعالجة المسبقة](./preprocessing) للحصول على مزيد من التفاصيل حول المعالجة، وكيفية استخدام [`AutoImageProcessor`] و [`AutoFeatureExtractor`] و [`AutoProcessor`] لمعالجة الصور والصوت والإدخالات متعددة الوسائط.

</Tip>

### AutoModel

<frameworkcontent>
<pt>
تقدم مكتبة 🤗 Transformers طريقة بسيطة وموحدة لتحميل نماذج مدربة مسبقًا. وهذا يعني أنه يمكنك تحميل [`AutoModel`] كما لو كنت تقوم بتحميل [`AutoTokenizer`]. الفرق الوحيد هو اختيار فئة [`AutoModel`] المناسبة للمهمة. بالنسبة لتصنيف النص (أو التسلسل)، يجب عليك تحميل [`AutoModelForSequenceClassification`]:

```py
>>> from transformers import AutoModelForSequenceClassification

>>> model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
>>> pt_model = AutoModelForSequenceClassification.from_pretrained(model_name)
```

<Tip>

راجع [ملخص المهمة](./task_summary) للاطلاع على المهام التي تدعمها فئة [`AutoModel`].

</Tip>

الآن قم بتمرير دفعة المدخلات المُعالجة مسبقًا مباشرة إلى النموذج. عليك فقط فك تعبئة القاموس عن طريق إضافة `**`:

# تدريب النموذج

الآن، مرر دفعة المدخلات المعالجة مسبقًا مباشرة إلى النموذج. ما عليك سوى فك تعبئة القاموس عن طريق إضافة `**`:

```py
>>> pt_outputs = pt_model(**pt_batch)
```

يُخرج النموذج التنشيطات النهائية في سمة `logits`. طبق دالة softmax على `logits` للحصول على الاحتمالات:

```py
>>> from torch import nn

>>> pt_predictions = nn.functional.softmax(pt_outputs.logits, dim=-1)
>>> print(pt_predictions)
tensor([[0.0021, 0.0018, 0.0115, 0.2121, 0.7725],
        [0.2084, 0.1826, 0.1969, 0.1755, 0.2365]], grad_fn=<SoftmaxBackward0>)
```
</pt>
<tf>
يوفر 🤗 Transformers طريقة بسيطة وموحدة لتحميل مثيلات مُدربة مسبقًا. وهذا يعني أنه يمكنك تحميل [`TFAutoModel`] مثل تحميل [`AutoTokenizer`]. والفرق الوحيد هو تحديد [`TFAutoModel`] الصحيح للمهمة. للتصنيف النصي (أو التسلسلي)، يجب تحميل [`TFAutoModelForSequenceClassification`]:

```py
>>> from transformers import TFAutoModelForSequenceClassification

>>> model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
>>> tf_model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
```

<Tip>

راجع [ملخص المهام](./task_summary) للمهام المدعومة بواسطة فئة [`AutoModel`].

</Tip>

الآن، مرر دفعة المدخلات المعالجة مسبقًا مباشرة إلى النموذج. يمكنك تمرير المصفوفات كما هي:

```py
>>> tf_outputs = tf_model(tf_batch)
```

يقوم النموذج بإخراج التنشيطات النهائية في سمة `logits`. طبق دالة softmax على `logits` لاسترداد الاحتمالات:

```py
>>> import tensorflow as tf

>>> tf_predictions = tf.nn.softmax(tf_outputs.logits, axis=-1)
>>> tf_predictions  # doctest: +IGNORE_RESULT
```
</tf>
</frameworkcontent>

<Tip>

تخرج جميع نماذج 🤗 Transformers (PyTorch أو TensorFlow) المصفوفات *قبل* دالة التنشيط النهائية (مثل softmax) لأن دالة التنشيط النهائية غالبًا ما تكون مدمجة مع دالة الخسارة. نواتج النموذج عبارة عن فئات بيانات خاصة، لذلك يتم استكمال سماتها تلقائيًا في IDE. وتتصرف مخرجات النموذج مثل زوج مرتب أو قاموس (يمكنك الفهرسة باستخدام عدد صحيح ، شريحة، أو سلسلة)، وفي هذه الحالة، يتم تجاهل السمات التي تساوي None.

</Tip>

### حفظ النموذج

<frameworkcontent>
<pt>
بمجرد ضبط نموذجك، يمكنك حفظه مع برنامج الترميز الخاص به باستخدام [`PreTrainedModel.save_pretrained`]:

```py
>>> pt_save_directory = "./pt_save_pretrained"
>>> tokenizer.save_pretrained(pt_save_directory)  # doctest: +IGNORE_RESULT
>>> pt_model.save_pretrained(pt_save_directory)
```

عندما تكون مستعدًا لاستخدام النموذج مرة أخرى، أعد تحميله باستخدام [`PreTrainedModel.from_pretrained`]:

```py
>>> pt_model = AutoModelForSequenceClassification.from_pretrained("./pt_save_pretrained")
```
</pt>
<tf>
بمجرد ضبط نموذجك، يمكنك حفظه مع برنامج الترميز الخاص به باستخدام [`TFPreTrainedModel.save_pretrained`]:

```py
>>> tf_save_directory = "./tf_save_pretrained"
>>> tokenizer.save_pretrained(tf_save_directory)  # doctest: +IGNORE_RESULT
>>> tf_model.save_pretrained(tf_save_directory)
```

عندما تكون مستعدًا لاستخدام النموذج مرة أخرى، أعد تحميله باستخدام [`TFPreTrainedModel.from_pretrained`]:

```py
>>> tf_model = TFAutoModelForSequenceClassification.from_pretrained("./tf_save_pretrained")
```
</tf>
</frameworkcontent>

من الميزات الرائعة في 🤗 Transformers القدرة على حفظ نموذج وإعادة تحميله كنموذج PyTorch أو TensorFlow. يمكن أن يحول معامل `from_pt` أو `from_tf` النموذج من إطار عمل إلى آخر:

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


## إنشاء نماذج مخصصة

يمكنك تعديل فئة تكوين النموذج لتغيير كيفية بناء النموذج. يحدد التكوين سمات النموذج، مثل عدد الطبقات المخفية أو رؤوس الاهتمام. تبدأ من الصفر عند تهيئة نموذج من فئة تكوين مخصصة. يتم تهيئة سمات النموذج بشكل عشوائي، ويجب تدريب النموذج قبل استخدامه للحصول على نتائج ذات معنى.

ابدأ باستيراد [`AutoConfig`]. ثم قم بتحميل النموذج المُدرب مسبقًا الذي تريد تعديله. ضمن [`AutoConfig.from_pretrained`]. يمكنك تحديد السمة التي تريد تغييرها، مثل عدد رؤوس الاهتمام:

```py
>>> from transformers import AutoConfig

>>> my_config = AutoConfig.from_pretrained("distilbert/distilbert-base-uncased", n_heads=12)
```

<frameworkcontent>
<pt>
قم بإنشاء نموذج من تكوينك المخصص باستخدام [`AutoModel.from_config`]:

```py
>>> from transformers import AutoModel

>>> my_model = AutoModel.from_config(my_config)
```
</pt>
<tf>
قم بإنشاء نموذج من تكوينك المخصص باستخدام [`TFAutoModel.from_config`]:

```py
>>> from transformers import TFAutoModel

>>> my_model = TFAutoModel.from_config(my_config)
```
</tf>
</frameworkcontent>

الق نظرة على دليل [إنشاء بنية مخصصة](./create_a_model) لمزيد من المعلومات حول بناء التكوينات المخصصة.

## المدرب - حلقة تدريب محسنة لـ PyTorch

جميع النماذج عبارة عن [`torch.nn.Module`](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) قياسية، لذا يمكنك استخدامها في أي حلقة تدريب نموذجية. في حين يمكنك كتابة حلقة التدريب الخاصة بك، يوفر 🤗 Transformers فئة [`Trainer`] لـ PyTorch، والتي تحتوي على حلقة التدريب الأساسية وتضيف وظائف إضافية لميزات مثل التدريب الموزع، والدقة المختلطة، والمزيد.

وفقًا لمهمتك، ستقوم عادةً بتمرير المعلمات التالية إلى [`Trainer`]:

1. ستبدأ بـ [`PreTrainedModel`] أو [`torch.nn.Module`](https://pytorch.org/docs/stable/nn.html#torch.nn.Module):

   ```py
   >>> from transformers import AutoModelForSequenceClassification

   >>> model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased")
   ```

2. تحتوي [`TrainingArguments`] على فرط معلمات النموذج التي يمكنك تغييرها مثل معدل التعلم، وحجم الدفعة، وعدد العصور التي يجب التدريب عليها. يتم استخدام القيم الافتراضية إذا لم تحدد أي حجج تدريب:

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

3. قم بتحميل فئة معالجة مسبقة مثل برنامج الترميز، أو معالج الصور، أو مستخرج الميزات، أو المعالج:

   ```py
   >>> from transformers import AutoTokenizer

   >>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
   ```

4. قم بتحميل مجموعة بيانات:

   ```py
   >>> from datasets import load_dataset

   >>> dataset = load_dataset("rotten_tomatoes")  # doctest: +IGNORE_RESULT
   ```

5. قم بإنشاء دالة لترميز مجموعة البيانات:

   ```py
   >>> def tokenize_dataset(dataset):
   ...     return tokenizer(dataset["text"])
   ```

   ثم قم بتطبيقه على مجموعة البيانات بأكملها باستخدام [`~datasets.Dataset.map`]:

   ```py
   >>> dataset = dataset.map(tokenize_dataset, batched=True)
   ```

6. [`DataCollatorWithPadding`] لإنشاء دفعة من الأمثلة من مجموعة البيانات الخاصة بك:

   ```py
   >>> from transformers import DataCollatorWithPadding

   >>> data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
   ```

الآن قم بتجميع جميع هذه الفئات في [`Trainer`]:

```py
>>> from transformers import Trainer

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=dataset["train"],
...     eval_dataset=dataset["test"],
...     tokenizer=tokenizer,
...     data_collator=data_collator,
... )  # doctest: +SKIP
```
عندما تكون مستعدًا، استدعِ [`~Trainer.train`] لبدء التدريب:

```py
>>> trainer.train()  # doctest: +SKIP
```

<Tip>

بالنسبة للمهام - مثل الترجمة أو التلخيص - التي تستخدم نموذج تسلسل إلى تسلسل، استخدم فئات [`Seq2SeqTrainer`] و [`Seq2SeqTrainingArguments`] بدلاً من ذلك.

</Tip>

يمكنك تخصيص سلوك حلقة التدريب عن طريق إنشاء فئة فرعية من الطرق داخل [`Trainer`]. يسمح لك ذلك بتخصيص ميزات مثل دالة الخسارة، والمحسن، والمجدول. راجع مرجع [`Trainer`] للتعرف على الطرق التي يمكن إنشاء فئات فرعية منها.

والطريقة الأخرى لتخصيص حلقة التدريب هي باستخدام [المستدعيات](./main_classes/callback). يمكنك استخدام المستدعيات للتكامل مع المكتبات الأخرى ومراقبة حلقة التدريب للإبلاغ عن التقدم أو إيقاف التدريب مبكرًا. لا تعدل المستدعيات أي شيء في حلقة التدريب نفسها. لتخصيص شيء مثل دالة الخسارة، تحتاج إلى إنشاء فئة فرعية من [`Trainer`] بدلاً من ذلك.

## التدريب باستخدام TensorFlow

جميع النماذج عبارة عن [`tf.keras.Model`](https://www.tensorflow.org/api_docs/python/tf/keras/Model) قياسية، لذا يمكن تدريبها في TensorFlow باستخدام واجهة برمجة تطبيقات Keras. يوفر 🤗 Transformers طريقة [`~TFPreTrainedModel.prepare_tf_dataset`] لتحميل مجموعة البيانات الخاصة بك بسهولة كـ `tf.data.Dataset` حتى تتمكن من البدء في التدريب على الفور باستخدام دالتي `compile` و`fit` في Keras.

1. ستبدأ بـ [`TFPreTrainedModel`] أو [`tf.keras.Model`](https://www.tensorflow.org/api_docs/python/tf/keras/Model):

   ```py
   >>> from transformers import TFAutoModelForSequenceClassification

   >>> model = TFAutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased")
   ```

2. قم بتحميل فئة معالجة مسبقة مثل برنامج الترميز، أو معالج الصور، أو مستخرج الميزات، أو المعالج:

   ```py
   >>> from transformers import AutoTokenizer

   >>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
   ```

3. قم بإنشاء دالة لترميز مجموعة البيانات:

   ```py
   >>> def tokenize_dataset(dataset):
   ...     return tokenizer(dataset["text"])  # doctest: +SKIP
   ```

4. قم بتطبيق برنامج الترميز على مجموعة البيانات بأكملها باستخدام [`~datasets.Dataset.map`] ثم مرر مجموعة البيانات وبرنامج الترميز إلى [`~TFPreTrainedModel.prepare_tf_dataset`]. يمكنك أيضًا تغيير حجم الدفعة وخلط مجموعة البيانات هنا إذا أردت:

   ```py
   >>> dataset = dataset.map(tokenize_dataset)  # doctest: +SKIP
   >>> tf_dataset = model.prepare_tf_dataset(
   ...     dataset["train"], batch_size=16, shuffle=True, tokenizer=tokenizer
   ... )  # doctest: +SKIP
   ```

5. عندما تكون مستعدًا، يمكنك استدعاء `compile` و`fit` لبدء التدريب. لاحظ أن جميع نماذج Transformers لديها دالة خسارة ذات صلة بالمهمة بشكل افتراضي، لذا فأنت لست بحاجة إلى تحديد واحدة ما لم ترغب في ذلك:

   ```py
   >>> from tensorflow.keras.optimizers import Adam

   >>> model.compile(optimizer='adam')  # لا توجد وسيطة دالة الخسارة!
   >>> model.fit(tf_dataset)  # doctest: +SKIP
   ```

## ماذا بعد؟

الآن بعد أن أكملت الجولة السريعة في 🤗 Transformers، راجع أدلتنا لمعرفة كيفية القيام بأشياء أكثر تحديدًا مثل كتابة نموذج مخصص، وضبط نموذج مسبق التدريب لمهمة معينة، وكيفية تدريب نموذج باستخدام نص برمجي. إذا كنت مهتمًا بمعرفة المزيد عن المفاهيم الأساسية لـ 🤗 Transformers، فاحصل على فنجان من القهوة واطلع على أدلة المفاهيم الخاصة بنا!
