# إنشاء نموذج كبير

تتمثل إحدى العقبات التي تحول دون الوصول إلى النماذج الكبيرة المُدربة مسبقًا في حجم الذاكرة المطلوبة. فعند تحميل نموذج PyTorch مُدرب مسبقًا، عادة ما تقوم بما يلي:

1. إنشاء نموذج بوزن عشوائي.
2. تحميل الأوزان المُدربة مسبقًا.
3. وضع تلك الأوزان المُدربة مسبقًا في النموذج.

يتطلب كل من الخطوتين الأوليين نسخة كاملة من النموذج في الذاكرة، وإذا كان وزن النموذج عدة غيغابايت، فقد لا تتوفر لديك ذاكرة كافية لنسختين منه. وتتفاقم هذه المشكلة في بيئات التدريب الموزعة لأن كل عملية تقوم بتحميل نموذج مُدرب مسبقًا وتخزين نسختين في الذاكرة.

> [!TIP]
> يتم تهيئة النموذج الذي تم إنشاؤه عشوائيًا باستخدام تنسورات "فارغة"، والتي تشغل مساحة في الذاكرة دون ملئها. والقيم العشوائية هي أي شيء كان في هذا الجزء من الذاكرة في ذلك الوقت. ولتحسين سرعة التحميل، يتم تعيين معلمة [`_fast_init`](https://github.com/huggingface/transformers/blob/c9f6e5e35156e068b227dd9b15521767f6afd4d2/src/transformers/modeling_utils.py#L2710) افتراضيًا على `True` لتخطي التهيئة العشوائية لجميع الأوزان التي يتم تحميلها بشكل صحيح.

سيوضح لك هذا الدليل كيف يمكن لمكتبة Transformers أن تساعدك في تحميل النماذج الكبيرة المُدربة مسبقًا على الرغم من متطلبات الذاكرة الخاصة بها.

## نقاط التفتيش المجزأة

اعتبارًا من الإصدار Transformers v4.18.0، يتم تجزئة أي نقطة تفتيش أكبر من 10 غيغابايت تلقائيًا بواسطة طريقة [`~PreTrainedModel.save_pretrained`](https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.save_pretrained). حيث يتم تقسيمها إلى عدة نقاط تفتيش جزئية أصغر، ويتم إنشاء ملف فهرس يقوم بتعيين أسماء المعلمات إلى الملفات التي يتم تخزينها فيها.

يتم التحكم في حجم الجزء الأقصى باستخدام معلمة `max_shard_size`، ولكن افتراضيًا يكون 5 غيغابايت، لأنه من الأسهل تشغيله على مثيلات GPU من الطبقة المجانية دون نفاد الذاكرة.

على سبيل المثال، دعنا نقسم [BioMistral/BioMistral-7B](https://hf.co/BioMistral/BioMistral-7B).

```py
>>> with tempfile.TemporaryDirectory() as tmp_dir:
...     model.save_pretrained(tmp_dir, max_shard_size="5GB")
...     print(sorted(os.listdir(tmp_dir)))
['config.json', 'generation_config.json', 'model-00001-of-00006.safetensors', 'model-00002-of-00006.safetensors', 'model-00003-of-00006.safetensors', 'model-00004-of-00006.safetensors', 'model-00005-of-00006.safetensors', 'model-00006-of-00006.safetensors', 'model.safetensors.index.json']
```

يتم إعادة تحميل نقطة التفتيش المجزأة باستخدام طريقة [`~PreTrainedModel.from_pretrained`](https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.from_pretrained).

```py
>>> with tempfile.TemporaryDirectory() as tmp_dir:
...     model.save_pretrained(tmp_dir, max_shard_size="5GB")
...     new_model = AutoModel.from_pretrained(tmp_dir)
```

الميزة الرئيسية لنقاط التفتيش المجزأة للنماذج الكبيرة هي أنه يتم تحميل كل جزء بعد الجزء السابق، مما يحد من استخدام الذاكرة إلى حجم النموذج فقط وحجم الجزء الأكبر.

يمكنك أيضًا تحميل نقطة تفتيش مجزأة مباشرة داخل نموذج بدون استخدام طريقة [`~PreTrainedModel.from_pretrained`](https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.from_pretrained) (وهو ما يشبه طريقة `load_state_dict()` في PyTorch لنقطة تفتيش كاملة). في هذه الحالة، استخدم طريقة [`~modeling_utils.load_sharded_checkpoint`](https://huggingface.co/docs/transformers/main_classes/modeling_utils#transformers.modeling_utils.load_sharded_checkpoint).

```py
>>> from transformers.modeling_utils import load_sharded_checkpoint

>>> with tempfile.TemporaryDirectory() as tmp_dir:
...     model.save_pretrained(tmp_dir, max_shard_size="5GB")
...     load_sharded_checkpoint(model, tmp_dir)
```


### البيانات التعريفية المشتركة Shared metadata

يحدد ملف الفهرس المفاتيح الموجودة في نقطة التفتيش ومكان تخزين الأوزان المقابلة. ويمكن تحميل هذا الملف مثل أي ملف JSON آخر، ويمكنك الحصول على قاموس منه.

```py
>>> import json

>>> with tempfile.TemporaryDirectory() as tmp_dir:
...     model.save_pretrained(tmp_dir, max_shard_size="5GB")
...     with open(os.path.join(tmp_dir, "model.safetensors.index.json"), "r") as f:
...         index = json.load(f)

>>> print(index.keys())
dict_keys(['metadata', 'weight_map'])
```

توفر مفتاح `metadata` حجم النموذج الإجمالي.

```py
>>> index["metadata"]
{'total_size': 28966928384}
```

يقوم مفتاح `weight_map` بتعيين كل اسم معلمة (عادةً `state_dict` في نموذج PyTorch) إلى الجزء الذي يتم تخزينه فيه.

```py
>>> index["weight_map"]
{'lm_head.weight': 'model-00006-of-00006.safetensors',
 'model.embed_tokens.weight': 'model-00001-of-00006.safetensors',
 'model.layers.0.input_layernorm.weight': 'model-00001-of-00006.safetensors',
 'model.layers.0.mlp.down_proj.weight': 'model-00001-of-00006.safetensors',
 ...
}
```

## الاستدلال باستخدام النماذج الكبيرة في Accelerate

> [!TIP]
> تأكد من تثبيت Accelerate v0.9.0 أو إصدار أحدث وPyTorch v1.9.0 أو إصدار أحدث.

اعتبارًا من الإصدار Transformers v4.20.0، تم تعزيز طريقة [`~PreTrainedModel.from_pretrained`](https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.from_pretrained) بميزة [الاستدلال باستخدام النماذج الكبيرة](https://hf.co/docs/accelerate/usage_guides/big_modeling) في Accelerate للتعامل بكفاءة مع النماذج الكبيرة جدًا! يقوم الاستدلال باستخدام النماذج الكبيرة بإنشاء *هيكل نموذج* على جهاز PyTorch [**meta**](https://pytorch.org/docs/main/meta.html). ولا يتم إنشاء المعلمات التي تم تهيئتها بشكل عشوائي إلا عند تحميل الأوزان المُدربة مسبقًا. وبهذه الطريقة، لن تحتفظ بنسختين من النموذج في الذاكرة في نفس الوقت (واحدة للنموذج الذي تم تهيئته بشكل عشوائي والأخرى للأوزان المُدربة مسبقًا)، ويصبح الحد الأقصى لاستهلاك الذاكرة هو حجم النموذج الكامل فقط.

لتمكين الاستدلال باستخدام النماذج الكبيرة في مكتبة Transformers، قم بتعيين `low_cpu_mem_usage=True` في طريقة [`~PreTrainedModel.from_pretrained`](https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.from_pretrained).

```py
from transformers import AutoModelForCausalLM

gemma = AutoModelForCausalLM.from_pretrained("google/gemma-7b", low_cpu_mem_usage=True)
```

يقوم Accelerate تلقائيًا بتوزيع أوزان النموذج عبر جميع الأجهزة المتاحة، بدءًا من الجهاز الأسرع (GPU) أولاً، ثم تفريغها إلى الأجهزة الأبطأ (CPU وحتى القرص الصلب). يتم تمكين ذلك عن طريق تعيين `device_map="auto"` في طريقة [`~PreTrainedModel.from_pretrained`](https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.from_pretrained). وعند تمرير معلمة `device_map`، يتم تعيين `low_cpu_mem_usage` تلقائيًا إلى `True`، لذلك لا تحتاج إلى تحديدها.

```py
from transformers import AutoModelForCausalLM

# these loading methods are equivalent
gemma = AutoModelForCausalLM.from_pretrained("google/gemma-7b", device_map="auto")
gemma = AutoModelForCausalLM.from_pretrained("google/gemma-7b", device_map="auto", low_cpu_mem_usage=True)
```

يمكنك أيضًا كتابة خريطة الأجهزة الخاصة بك عن طريق تعيين كل طبقة إلى جهاز. يجب أن تقوم خريطة الأجهزة بتعيين جميع معلمات النموذج إلى جهاز، ولكن لا يلزم أن تحدد مكان وجود جميع الوحدات الفرعية لطبقة ما إذا كانت الطبقة بأكملها على نفس الجهاز.

```python
device_map = {"model.layers.1": 0, "model.layers.14": 1, "model.layers.31": "cpu", "lm_head": "disk"}
```

يمكنك الوصول إلى خاصية `hf_device_map` لمعرفة كيفية تقسيم Accelerate للنموذج عبر الأجهزة.

```py
gemma.hf_device_map
```

```python out
{'model.embed_tokens': 0,
 'model.layers.0': 0,
 'model.layers.1': 0,
 'model.layers.2': 0,
 'model.layers.3': 0,
 'model.layers.4': 0,
 'model.layers.5': 0,
 'model.layers.6': 0,
 'model.layers.7': 0,
 'model.layers.8': 0,
 'model.layers.9': 0,
 'model.layers.10': 0,
 'model.layers.11': 0,
 'model.layers.12': 0,
 'model.layers.13': 0,
 'model.layers.14': 'cpu',
 'model.layers.15': 'cpu',
 'model.layers.16': 'cpu',
 'model.layers.17': 'cpu',
 'model.layers.18': 'cpu',
 'model.layers.19': 'cpu',
 'model.layers.20': 'cpu',
 'model.layers.21': 'cpu',
 'model.layers.22': 'cpu',
 'model.layers.23': 'cpu',
 'model.layers.24': 'cpu',
 'model.layers.25': 'cpu',
 'model.layers.26': 'cpu',
 'model.layers.27': 'cpu',
 'model.layers.28': 'cpu',
 'model.layers.29': 'cpu',
 'model.layers.30': 'cpu',
 'model.layers.31': 'cpu',
 'model.norm': 'cpu',
 'lm_head': 'cpu'}
```

## نوع بيانات النموذج

عادةً ما يتم إنشاء أوزان نموذج PyTorch كـ torch.float32، ويمكن أن يمثل ذلك مشكلة إذا حاولت تحميل نموذج كنوع بيانات مختلف. على سبيل المثال، ستحتاج إلى ضعف الذاكرة لتحميل الأوزان في torch.float32 ثم مرة أخرى لتحميلها في نوع البيانات المطلوب، مثل torch.float16.

> [!WARNING]
> بسبب تصميم PyTorch، فإن معلمة `torch_dtype` تدعم فقط أنواع البيانات العائمة.

للتجنب إهدار الذاكرة بهذه الطريقة، قم بتعيين معلمة `torch_dtype` بشكل صريح إلى نوع البيانات المطلوب أو قم بتعيين `torch_dtype="auto"` لتحميل الأوزان بأكثر أنماط الذاكرة المثالية (يتم اشتقاق نوع البيانات تلقائيًا من أوزان النموذج).

<hfoptions id="dtype">
<hfoption id="specific dtype">

```py
from transformers import AutoModelForCausalLM

gemma = AutoModelForCausalLM.from_pretrained("google/gemma-7b", torch_dtype=torch.float16)
```

</hfoption>
<hfoption id="auto dtype">

```py
from transformers import AutoModelForCausalLM

gemma = AutoModelForCausalLM.from_pretrained("google/gemma-7b", torch_dtype="auto")
```

</hfoption>
</hfoptions>

يمكنك أيضًا تعيين نوع البيانات الذي سيتم استخدامه للنماذج التي تم إنشاؤها من الصفر.

```python
import torch
from transformers import AutoConfig, AutoModel

my_config = AutoConfig.from_pretrained("google/gemma-2b", torch_dtype=torch.float16)
model = AutoModel.from_config(my_config)
```