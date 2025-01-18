# كيفية إنشاء خط أنابيب مخصص؟

في هذا الدليل، سنرى كيفية إنشاء خط أنابيب مخصص ومشاركته على [Hub](https://hf.co/models) أو إضافته إلى
مكتبة 🤗 Transformers.

أولاً وقبل كل شيء، تحتاج إلى تحديد الإدخالات الخام التي سيتمكن خط الأنابيب من التعامل معها. يمكن أن تكون هذه الإدخالات سلاسل نصية أو بايتات خام أو
قواميس أو أي شيء آخر يبدو أنه الإدخال المرغوب. حاول إبقاء هذه الإدخالات بسيطة قدر الإمكان
كما هو الحال في Python حيث يجعل ذلك التوافق أسهل (حتى من خلال لغات أخرى عبر JSON). ستكون هذه هي `الإدخالات` من
خط الأنابيب (`preprocess`).

ثم حدد `الإخراج`. نفس السياسة كما `الإدخالات`. كلما كان أبسط، كان ذلك أفضل. ستكون هذه هي الإخراج من
طريقة `postprocess`.

ابدأ بالوراثة من الفئة الأساسية `Pipeline` مع الطرق الأربع اللازمة لتنفيذ `preprocess`،
`_forward`، `postprocess`، و `_sanitize_parameters`.


```python
from transformers import Pipeline


class MyPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        if "maybe_arg" in kwargs:
            preprocess_kwargs["maybe_arg"] = kwargs["maybe_arg"]
        return preprocess_kwargs, {}, {}

    def preprocess(self, inputs, maybe_arg=2):
        model_input = Tensor(inputs["input_ids"])
        return {"model_input": model_input}

    def _forward(self, model_inputs):
        # model_inputs == {"model_input": model_input}
        outputs = self.model(**model_inputs)
        # Maybe {"logits": Tensor(...)}
        return outputs

    def postprocess(self, model_outputs):
        best_class = model_outputs["logits"].softmax(-1)
        return best_class
```

يهدف هيكل هذا التفصيل إلى دعم التوافق السلس نسبيًا مع CPU/GPU، مع دعم إجراء
pre/postprocessing على CPU على خيوط مختلفة

`preprocess` ستأخذ الإدخالات المحددة أصلاً، وتحويلها إلى شيء يمكن تغذيته إلى النموذج. قد
يحتوي على مزيد من المعلومات وعادة ما يكون `Dict`.

`_forward` هو تفاصيل التنفيذ وليس المقصود استدعاؤه مباشرة. `forward` هي الطريقة المفضلة
طريقة الاستدعاء حيث تحتوي على ضمانات للتأكد من أن كل شيء يعمل على الجهاز المتوقع. إذا كان أي شيء
مرتبط بنموذج حقيقي، فهو ينتمي إلى طريقة `_forward`، وأي شيء آخر هو في preprocess/postprocess.

ستأخذ طرق `postprocess` إخراج `_forward` وتحويله إلى الإخراج النهائي الذي تم تحديده
سابقا.

`_sanitize_parameters` موجودة للسماح للمستخدمين بتمرير أي معلمات كلما رغبوا في ذلك، سواء عند التهيئة
الوقت `pipeline(...., maybe_arg=4)` أو وقت الاستدعاء `pipe = pipeline(...); output = pipe(...., maybe_arg=4)`.

تكون إرجاعات `_sanitize_parameters` عبارة عن 3 قواميس kwargs التي سيتم تمريرها مباشرة إلى `preprocess`،
`_forward`، و `postprocess`. لا تملأ أي شيء إذا لم يستدع المتصل بأي معلمة إضافية. هذا
يسمح بالحفاظ على الحجج الافتراضية في تعريف الدالة وهو دائمًا أكثر "طبيعية".

مثال كلاسيكي سيكون حجة `top_k` في post processing في مهام التصنيف.

```python
>>> pipe = pipeline("my-new-task")
>>> pipe("This is a test")
[{"label": "1-star", "score": 0.8}, {"label": "2-star", "score": 0.1}, {"label": "3-star", "score": 0.05}
{"label": "4-star", "score": 0.025}, {"label": "5-star", "score": 0.025}]
>>> pipe("This is a test", top_k=2)
[{"label": "1-star", "score": 0.8}, {"label": "2-star", "score": 0.1}]
```

لتحقيق ذلك، سنقوم بتحديث طريقة `postprocess` بحجة افتراضية إلى `5`. وتحرير
`_sanitize_parameters` للسماح بهذه المعلمة الجديدة.


```python
def postprocess(self, model_outputs, top_k=5):
    best_class = model_outputs["logits"].softmax(-1)
    # Add logic to handle top_k
    return best_class


def _sanitize_parameters(self, **kwargs):
    preprocess_kwargs = {}
    if "maybe_arg" in kwargs:
        preprocess_kwargs["maybe_arg"] = kwargs["maybe_arg"]

    postprocess_kwargs = {}
    if "top_k" in kwargs:
        postprocess_kwargs["top_k"] = kwargs["top_k"]
    return preprocess_kwargs, {}, postprocess_kwargs
```

حاول الحفاظ على الإدخالات/الإخراج بسيطة قدر الإمكان ويفضل أن تكون JSON-serializable حيث يجعل استخدام خط الأنابيب سهلًا
دون الحاجة إلى فهم المستخدمين لأنواع جديدة من الكائنات. من الشائع أيضًا دعم العديد من أنواع
الحجج المختلفة من أجل سهولة الاستخدام (ملفات الصوت، والتي يمكن أن تكون أسماء ملفات أو عناوين URL أو بايتات خام)



## إضافته إلى قائمة المهام المدعومة

لتسجيل مهمة "new-task" في قائمة المهام المدعومة، يجب إضافتها إلى `PIPELINE_REGISTRY`:

```python
from transformers.pipelines import PIPELINE_REGISTRY

PIPELINE_REGISTRY.register_pipeline(
    "new-task",
    pipeline_class=MyPipeline,
    pt_model=AutoModelForSequenceClassification,
)
```

يمكنك تحديد نموذج افتراضي إذا أردت، وفي هذه الحالة يجب أن يأتي مع مراجعة محددة (والتي يمكن أن تكون اسم فرع أو هاش الالتزام، هنا أخذنا `"abcdef"`) وكذلك النوع:

```python
PIPELINE_REGISTRY.register_pipeline(
    "new-task",
    pipeline_class=MyPipeline,
    pt_model=AutoModelForSequenceClassification,
    default={"pt": ("user/awesome_model", "abcdef")},
    type="text",  # current support type: text, audio, image, multimodal
)
```

## مشاركة خط الأنابيب المخصص الخاص بك على Hub

لمشاركة خط الأنابيب المخصص الخاص بك على Hub، ما عليك سوى حفظ رمز مخصص لفئة `Pipeline` الفرعية في
ملف Python. على سبيل المثال، لنفترض أننا نريد استخدام خط أنابيب مخصص لتصنيف أزواج الجمل مثل هذا:

```py
import numpy as np

from transformers import Pipeline


def softmax(outputs):
    maxes = np.max(outputs, axis=-1, keepdims=True)
    shifted_exp = np.exp(outputs - maxes)
    return shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)


class PairClassificationPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        if "second_text" in kwargs:
            preprocess_kwargs["second_text"] = kwargs["second_text"]
        return preprocess_kwargs, {}, {}

    def preprocess(self, text, second_text=None):
        return self.tokenizer(text, text_pair=second_text, return_tensors=self.framework)

    def _forward(self, model_inputs):
        return self.model(**model_inputs)

    def postprocess(self, model_outputs):
        logits = model_outputs.logits[0].numpy()
        probabilities = softmax(logits)

        best_class = np.argmax(probabilities)
        label = self.model.config.id2label[best_class]
        score = probabilities[best_class].item()
        logits = logits.tolist()
        return {"label": label, "score": score, "logits": logits}
```

التنفيذ مستقل عن الإطار، وسيعمل لكل من نماذج PyTorch وTensorFlow. إذا قمنا بحفظ هذا في
ملف باسم `pair_classification.py`، فيمكننا بعد ذلك استيراده وتسجيله على النحو التالي:

```py
from pair_classification import PairClassificationPipeline
from transformers.pipelines import PIPELINE_REGISTRY
from transformers import AutoModelForSequenceClassification, TFAutoModelForSequenceClassification

PIPELINE_REGISTRY.register_pipeline(
    "pair-classification",
    pipeline_class=PairClassificationPipeline,
    pt_model=AutoModelForSequenceClassification,
    tf_model=TFAutoModelForSequenceClassification,
)
```
بمجرد القيام بذلك، يمكننا استخدامه مع نموذج مدرب مسبقًا. على سبيل المثال، تم تدريب `sgugger/finetuned-bert-mrpc`
على مجموعة بيانات MRPC، والتي تصنف أزواج الجمل كأقوال أو لا.

```py
from transformers import pipeline

classifier = pipeline("pair-classification", model="sgugger/finetuned-bert-mrpc")
```

بعد ذلك، يمكننا مشاركته على Hub باستخدام طريقة `push_to_hub`:

```py
classifier.push_to_hub("test-dynamic-pipeline")
```

سيقوم هذا بنسخ الملف الذي حددت فيه `PairClassificationPipeline` داخل المجلد `"test-dynamic-pipeline"`،
جنبًا إلى جنب مع حفظ نموذج ورمز محدد للخط أنابيب، قبل دفع كل شيء إلى المستودع
`{your_username}/test-dynamic-pipeline`. بعد ذلك، يمكن لأي شخص استخدامه طالما قاموا بتوفير الخيار
`trust_remote_code=True`:

```py
from transformers import pipeline

classifier = pipeline(model="{your_username}/test-dynamic-pipeline", trust_remote_code=True)
```

## إضافة خط الأنابيب إلى 🤗 Transformers

إذا كنت تريد المساهمة بخط أنابيبك في 🤗 Transformers، فستحتاج إلى إضافة وحدة نمطية جديدة في الوحدة الفرعية `pipelines`
برمز خط الأنابيب الخاص بك، ثم أضفه إلى قائمة المهام المحددة في `pipelines/__init__.py`.

بعد ذلك، ستحتاج إلى إضافة الاختبارات. قم بإنشاء ملف جديد `tests/test_pipelines_MY_PIPELINE.py` مع أمثلة للاختبارات الأخرى.

ستكون دالة `run_pipeline_test` عامة جدًا وستعمل على نماذج صغيرة عشوائية لكل هندسة معمارية ممكنة
كما هو محدد بواسطة `model_mapping` و `tf_model_mapping`.

هذا مهم جدًا لاختبار التوافق المستقبلي، مما يعني أنه إذا قام شخص ما بإضافة نموذج جديد لـ
`XXXForQuestionAnswering` فسيحاول اختبار خط الأنابيب تشغيله. نظرًا لأن النماذج عشوائية، فمن
من المستحيل التحقق من القيم الفعلية، ولهذا السبب يوجد مساعد `ANY` الذي سيحاول ببساطة مطابقة
إخراج نوع خط الأنابيب.

أنت أيضًا *تحتاج* إلى تنفيذ اختبارين (يفضل 4).

- `test_small_model_pt` : قم بتعريف نموذج صغير واحد لهذا الخط أنابيب (لا يهم إذا لم تكن النتائج منطقية)
  واختبار الإخراج من خط الأنابيب. يجب أن تكون النتائج هي نفسها `test_small_model_tf`.
- `test_small_model_tf` : قم بتعريف نموذج صغير واحد لهذا الخط أنابيب (لا يهم إذا لم تكن النتائج منطقية)
  واختبار الإخراج من خط الأنابيب. يجب أن تكون النتائج هي نفسها `test_small_model_pt`.
- `test_large_model_pt` (`اختياري`): اختبارات خط الأنابيب على خط أنابيب حقيقي حيث من المفترض أن تكون النتائج
  منطقي. هذه الاختبارات بطيئة ويجب تمييزها على هذا النحو. هنا الهدف هو عرض خط الأنابيب والتأكد من
  لا يوجد انجراف في الإصدارات المستقبلية.
- `test_large_model_tf` (`اختياري`): اختبارات خط الأنابيب على خط أنابيب حقيقي حيث من المفترض أن تكون النتائج
  منطقي. هذه الاختبارات بطيئة ويجب تمييزها على هذا النحو. هنا الهدف هو عرض خط الأنابيب والتأكد من
  لا يوجد انجراف في الإصدارات المستقبلية.