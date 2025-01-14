# DeepSpeed

[DeepSpeed](https://www.deepspeed.ai/) هي مكتبة تحسين PyTorch تجعل التدريب الموزع فعالًا من حيث الذاكرة وسريعًا. وفي جوهره [مُحسِّن Zero Redundancy (ZeRO)](https://hf.co/papers/1910.02054) الذي يمكّن من تدريب النماذج الكبيرة على نطاق واسع. يعمل ZeRO في عدة مراحل:

* ZeRO-1، تجزئة حالة المحسن عبر وحدات معالجة الرسومات (GPU)
* ZeRO-2، تجزئة التدرجات عبر وحدات معالجة الرسومات (GPU)
* ZeRO-3، تجزئة المعلمات عبر وحدات معالجة الرسومات (GPU)

في البيئات المحدودة بوحدة معالجة الرسومات (GPU)، يمكّن ZeRO أيضًا من نقل ذاكرة المحسن والحساب من وحدة معالجة الرسومات (GPU) إلى وحدة المعالجة المركزية (CPU) لتناسب وتدريب النماذج الكبيرة حقًا على وحدة معالجة الرسومات (GPU) واحدة. تم دمج DeepSpeed مع فئة [`Trainer`] في Transformers لجميع مراحل ZeRO والنقل. كل ما عليك فعله هو توفير ملف تكوين أو يمكنك استخدام قالب مقدم. بالنسبة للاستدلال، تدعم Transformers ZeRO-3 والنقل لأنه يسمح بتحميل النماذج الضخمة.

سيوضح لك هذا الدليل كيفية نشر التدريب باستخدام DeepSpeed، والميزات التي يمكنك تمكينها، وكيفية إعداد ملفات التكوين لمراحل ZeRO المختلفة، والنقل، والاستدلال، واستخدام DeepSpeed بدون [`Trainer`].

## التثبيت

DeepSpeed متاح لتثبيته من PyPI أو Transformers (للحصول على خيارات تثبيت أكثر تفصيلاً، راجع تفاصيل تثبيت DeepSpeed [التفاصيل](https://www.deepspeed.ai/tutorials/advanced-install/) أو قراءة [README](https://github.com/microsoft/deepspeed#installation) على GitHub).

<Tip>

إذا كنت تواجه صعوبات في تثبيت DeepSpeed، فراجع دليل تثبيت CUDA في DeepSpeed [دليل](../debugging#deepspeed-cuda-installation). في حين أن لدى DeepSpeed حزمة PyPI قابلة للتثبيت باستخدام pip، يوصى بشدة [بتثبيته من المصدر](https://www.deepspeed.ai/tutorials/advanced-install/#install-deepspeed-from-source) لمطابقة أجهزتك بشكل أفضل ولتدعم ميزات معينة، مثل 1-bit Adam، والتي ليست متاحة في توزيع PyPI.

</Tip>

<hfoptions id="install">
<hfoption id="PyPI">

```bash
pip install deepspeed
```

</hfoption>
<hfoption id="Transformers">

```bash
pip install transformers[deepspeed]
```

</hfoption>
</hfoptions>

## متطلبات الذاكرة

قبل البدء، من الجيد التحقق مما إذا كان لديك ذاكرة GPU وCPU كافية لتناسب نموذجك. توفر DeepSpeed أداة لتقدير متطلبات ذاكرة CPU/GPU. على سبيل المثال، لتقدير متطلبات الذاكرة لنموذج [bigscience/T0_3B](bigscience/T0_3B) على وحدة معالجة رسومات (GPU) واحدة:
```bash
$ python -c 'from transformers import AutoModel; \
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live; \
model = AutoModel.from_pretrained("bigscience/T0_3B"); \
estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=1, num_nodes=1)'
[...]
Estimated memory needed for params, optim states and gradients for a:
HW: Setup with 1 node, 1 GPU per node.
SW: Model with 2783M total params, 65M largest layer params.
  per CPU  |  per GPU |   Options
   70.00GB |   0.25GB | offload_param=cpu , offload_optimizer=cpu , zero_init=1
   70.00GB |   0.25GB | offload_param=cpu , offload_optimizer=cpu , zero_init=0
   62.23GB |   5.43GB | offload_param=none, offload_optimizer=cpu , zero_init=1
   62.23GB |   5.43GB | offload_param=none, offload_optimizer=cpu , zero_init=0
    0.37GB |  46.91GB | offload_param=none, offload_optimizer=none, zero_init=1
   15.56GB |  46.91GB | offload_param=none, offload_optimizer=none, zero_init=0
```

هذا يعني أنك إما بحاجة إلى وحدة معالجة رسومات (GPU) واحدة بسعة 80 جيجابايت دون نقل إلى وحدة المعالجة المركزية (CPU) أو وحدة معالجة رسومات (GPU) بسعة 8 جيجابايت ووحدة معالجة مركزية (CPU) بسعة 60 جيجابايت لنقلها إليها (هذه هي متطلبات الذاكرة للمعلمات وحالات المحسن والتدرجات فقط، وستحتاج إلى القليل من الذاكرة الإضافية لنوى CUDA والتنشيط). يجب عليك أيضًا مراعاة المقايضة بين التكلفة والسرعة لأنه سيكون من الأرخص استئجار أو شراء وحدة معالجة رسومات (GPU) أصغر ولكن سيستغرق تدريب نموذجك وقتًا أطول.

إذا كانت لديك ذاكرة GPU كافية، فتأكد من تعطيل النقل إلى وحدة المعالجة المركزية (CPU)/NVMe لجعل كل شيء أسرع.

## حدد مرحلة ZeRO

بعد تثبيت DeepSpeed والحصول على فكرة أفضل عن متطلبات الذاكرة الخاصة بك، تتمثل الخطوة التالية في تحديد مرحلة ZeRO لاستخدامها. حسب الترتيب الأسرع والأكثر كفاءة في الذاكرة:

| الأسرع          | الأكثر كفاءة في الذاكرة |
|------------------|------------------|
| ZeRO-1           | ZeRO-3 + النقل |
| ZeRO-2           | ZeRO-3           |
| ZeRO-2 + النقل | ZeRO-2 + النقل |
| ZeRO-3           | ZeRO-2           |
| ZeRO-3 + النقل | ZeRO-1           |

لمعرفة ما يناسبك، ابدأ بالنهج الأسرع وإذا نفدت من الذاكرة، فجرّب المرحلة التالية التي تكون أبطأ ولكنها أكثر كفاءة في الذاكرة. لا تتردد في العمل في أي اتجاه تفضله (بدءًا من الأكثر كفاءة في الذاكرة أو الأسرع) لاكتشاف التوازن المناسب بين السرعة واستخدام الذاكرة.

يمكنك استخدام عملية عامة (ابدأ بحجم دفعة يبلغ 1):

1. تمكين نقاط تفتيش التدرج
2. جرب ZeRO-2
3. جرب ZeRO-2 ونقل المحسن
4. جرب ZeRO-3
5. جرب ZeRO-3 ونقل المعلمات إلى وحدة المعالجة المركزية (CPU)
6. جرب ZeRO-3 ونقل المعلمات والمحسن إلى وحدة المعالجة المركزية (CPU)
7. جرب تقليل القيم الافتراضية المختلفة مثل شعاع بحث أضيق إذا كنت تستخدم طريقة [`~GenerationMixin.generate`]
8. جرب الدقة المختلطة نصفية الدقة (fp16 على معماريات GPU الأقدم وbf16 على Ampere) على الأوزان ذات الدقة الكاملة
9. أضف المزيد من الأجهزة إذا أمكن أو قم بتمكين Infinity لنقل المعلمات والمحسن إلى NVMe
10. بمجرد عدم نفاد الذاكرة، قم بقياس الإنتاجية الفعالة ثم حاول زيادة حجم الدفعة قدر الإمكان لتعظيم كفاءة وحدة معالجة الرسومات (GPU)
11. أخيرًا، حاول تحسين إعداد التدريب الخاص بك عن طريق تعطيل بعض ميزات النقل أو استخدام مرحلة ZeRO أسرع وزيادة/تقليل حجم الدفعة للعثور على أفضل مقايضة بين السرعة واستخدام الذاكرة


## ملف تكوين DeepSpeed

يعمل DeepSpeed مع فئة [`Trainer`] من خلال ملف تكوين يحتوي على جميع المعلمات اللازمة لتكوين كيفية إعداد تشغيل التدريب الخاص بك. عندما تنفذ نص البرنامج النصي للتدريب، يقوم DeepSpeed بتسجيل التكوين الذي تلقاه من [`Trainer`] في وحدة التحكم حتى تتمكن من رؤية التكوين المستخدم بالضبط.

<Tip>
<Tip>

يمكنك العثور على قائمة كاملة بخيارات تكوين DeepSpeed في مرجع تكوين DeepSpeed JSON [المرجع](https://www.deepspeed.ai/docs/config-json/). يمكنك أيضًا العثور على أمثلة عملية مختلفة لتكوين DeepSpeed على مستودع [DeepSpeedExamples](https://github.com/microsoft/DeepSpeedExamples) أو المستودع الرئيسي [DeepSpeed](https://github.com/microsoft/DeepSpeed). للعثور بسرعة على أمثلة محددة، يمكنك:

```bash
git clone https://github.com/microsoft/DeepSpeedExamples
cd DeepSpeedExamples
find . -name '*json'
# find examples with the Lamb optimizer
grep -i Lamb $(find . -name '*json')
```

</Tip>

يتم تمرير ملف تكوين DeepSpeed كمسار إلى ملف JSON إذا كنت تقوم بالتدريب من واجهة سطر الأوامر أو ككائن `dict` متداخل إذا كنت تستخدم [`Trainer`] في إعداد دفتر الملاحظات.

<hfoptions id="pass-config">
<hfoption id="path to file">

```py
TrainingArguments(..., deepspeed="path/to/deepspeed_config.json")
```

</hfoption>
<hfoption id="nested dict">

```py
ds_config_dict = dict(scheduler=scheduler_params, optimizer=optimizer_params)
args = TrainingArguments(..., deepspeed=ds_config_dict)
trainer = Trainer(model, args, ...)
```

</hfoption>
</hfoptions>

### معلمات DeepSpeed وTrainer

هناك ثلاثة أنواع من معلمات التكوين:

1. بعض معلمات التكوين مشتركة بين [`Trainer`] وDeepSpeed، ويمكن أن يكون من الصعب تحديد الأخطاء عندما تكون هناك تعريفات متضاربة. لتسهيل الأمر، يتم تكوين هذه المعلمات المشتركة من خلال حجة سطر الأوامر [`Trainer`].

2. يتم اشتقاق بعض معلمات التكوين تلقائيًا من تكوين النموذج، لذلك لا تحتاج إلى ضبط هذه القيم يدويًا. يستخدم [`Trainer`] قيمة التكوين `auto` لتحديد القيمة الأكثر صحة أو كفاءة. يمكنك تعيين معلمات التكوين الخاصة بك بشكل صريح، ولكن يجب عليك التأكد من اتفاق حجج [`Trainer`] ومعلمات تكوين DeepSpeed. قد تسبب عدم التطابقات فشل التدريب بطرق يصعب اكتشافها!

3. بعض معلمات التكوين المحددة لـ DeepSpeed فقط والتي تحتاج إلى إعدادها يدويًا بناءً على احتياجات التدريب الخاصة بك.

يمكنك أيضًا تعديل تكوين DeepSpeed وتحرير [`TrainingArguments`] منه:

1. قم بإنشاء أو تحميل تكوين DeepSpeed لاستخدامه كالتكوين الرئيسي
2. قم بإنشاء كائن [`TrainingArguments`] بناءً على قيم تكوين DeepSpeed هذه

يقوم [`Trainer`] بحساب بعض القيم، مثل `scheduler.params.total_num_steps`، أثناء التدريب.

### تكوين ZeRO

هناك ثلاثة تكوينات، لكل منها مرحلة ZeRO مختلفة. المرحلة 1 ليست مثيرة للاهتمام من حيث قابلية التوسع، ويركز هذا الدليل على المرحلتين 2 و3. يحتوي تكوين `zero_optimization` على جميع الخيارات لتمكينها وكيفية تكوينها. للحصول على شرح أكثر تفصيلاً لكل معلمة، راجع مرجع تكوين DeepSpeed JSON [المرجع](https://www.deepspeed.ai/docs/config-json/).

<Tip warning={true}>
لا يتحقق DeepSpeed من أسماء المعلمات وأي أخطاء إملائية تعود إلى الإعداد الافتراضي للمعلمة. يمكنك مشاهدة رسائل تسجيل تشغيل محرك DeepSpeed لمعرفة القيم التي سيتم استخدامها.

</Tip>

يجب إعداد التكوينات التالية باستخدام DeepSpeed لأن [`Trainer`] لا يوفر حجج سطر الأوامر المكافئة.

<hfoptions id="zero-config">
<hfoption id="ZeRO-1">

يقسم ZeRO-1 حالات المحسن عبر وحدات معالجة الرسومات (GPU)، ويمكنك توقع زيادة طفيفة في السرعة. يمكن إعداد تكوين ZeRO-1 على النحو التالي:

```yml
{
    "zero_optimization": {
        "stage": 1
    }
}
```

</hfoption>
<hfoption id="ZeRO-2">

يقسم ZeRO-2 المحسن والتدرجات عبر وحدات معالجة الرسومات (GPU). تُستخدم هذه المرحلة في المقام الأول للتدريب نظرًا لعدم أهمية ميزاتها للاستدلال. بعض المعلمات المهمة التي يجب تكوينها لتحقيق أداء أفضل تشمل:
* يجب تمكين `offload_optimizer` لتقليل استخدام ذاكرة وحدة معالجة الرسومات (GPU).
* عندما يتم تعيين `overlap_comm` على `true`، فإنه يتداول زيادة استخدام ذاكرة وحدة معالجة الرسومات (GPU) لتقليل الكمون allreduce. تستخدم هذه الميزة 4.5x قيم `allgather_bucket_size` و`reduce_bucket_size`. في هذا المثال، يتم تعيينها على `5e8` مما يعني أنها تتطلب 9 جيجابايت من ذاكرة وحدة معالجة الرسومات (GPU). إذا كانت ذاكرة وحدة معالجة الرسومات (GPU) لديك 8 جيجابايت أو أقل، فيجب عليك تقليل `overlap_comm` لتقليل متطلبات الذاكرة ومنع خطأ نفاد الذاكرة (OOM).
* يتداول `allgather_bucket_size` و`reduce_bucket_size` ذاكرة وحدة معالجة الرسومات (GPU) المتاحة مقابل السرعة. كلما صغرت قيمهما، كلما كان الاتصال أبطأ وزادت ذاكرة وحدة معالجة الرسومات (GPU) المتاحة. يمكنك الموازنة، على سبيل المثال، بين ما إذا كان حجم الدفعة الأكبر أكثر أهمية من وقت التدريب البطيء قليلاً.
* `round_robin_gradients` متاح في DeepSpeed 0.4.4 لنقل CPU. فهو يوازي نسخ التدرج إلى ذاكرة CPU بين الرتب عن طريق تجزئة التدرج الدقيق. تنمو الفائدة في الأداء مع خطوات تراكم التدرج (المزيد من النسخ بين خطوات المحسن) أو عدد وحدات معالجة الرسومات (GPU) (زيادة التوازي).

```yml
{
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "allgather_partitions": true,
        "allgather_bucket_size": 5e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 5e8,
        "contiguous_gradients": true,
        "round_robin_gradients": true
    }
}
```

</hfoption>
<hfoption id="ZeRO-3">

يقوم ZeRO-3 بتجزئة المحسن والتدرج والمعلمات عبر وحدات معالجة الرسومات. على عكس ZeRO-2، يمكن أيضًا استخدام ZeRO-3 للاستنتاج، بالإضافة إلى التدريب، لأنه يسمح بتحميل نماذج كبيرة على وحدات معالجة الرسومات متعددة. بعض المعلمات المهمة للتكوين تشمل:

* `device: "cpu"` يمكن أن يساعد إذا كنت تواجه مشكلة في نفاد ذاكرة وحدة معالجة الرسومات (GPU) ولديك ذاكرة CPU مجانية متاحة. يسمح هذا بإلغاء تحميل معلمات النموذج إلى وحدة المعالجة المركزية (CPU).
* `pin_memory: true` يمكن أن يحسن الإنتاجية، ولكن يصبح مقدار أقل من الذاكرة متاحًا للعمليات الأخرى لأن الذاكرة المثبتة محجوزة لعملية محددة طلبتها وعادة ما يتم الوصول إليها بشكل أسرع من ذاكرة وحدة المعالجة المركزية (CPU) العادية.
* `stage3_max_live_parameters` هو الحد الأعلى لعدد المعلمات الكاملة التي تريد الاحتفاظ بها في وحدة معالجة الرسومات في أي وقت. قلل من هذه القيمة إذا واجهت خطأ OOM.
* `stage3_max_reuse_distance` هي قيمة لتحديد متى يتم استخدام معلمة مرة أخرى في المستقبل، وهي تساعد في اتخاذ قرار بشأن ما إذا كان يجب التخلص من المعلمة أو الاحتفاظ بها. إذا كان سيتم إعادة استخدام المعلمة (إذا كانت القيمة أقل من `stage3_max_reuse_distance`)، فيتم الاحتفاظ بها لتقليل التخزين المؤقت للاتصال. هذا مفيد للغاية عندما يتم تمكين التحقق من التنشيط وتريد الاحتفاظ بالمعلمة في إعادة حساب التمرير إلى الأمام حتى تمريرة الخلف. ولكن قلل من هذه القيمة إذا واجهت خطأ OOM.
* `stage3_gather_16bit_weights_on_model_save` توحيد أوزان fp16 عند حفظ نموذج. بالنسبة للنماذج الكبيرة وحدات معالجة الرسومات متعددة، هذا مكلف من حيث الذاكرة والسرعة. يجب تمكينه إذا كنت تخطط لاستئناف التدريب.
* `sub_group_size` يتحكم في المعلمات التي يتم تحديثها أثناء خطوة المحسن. يتم تجميع المعلمات في دلوات من `sub_group_size` ويتم تحديث كل دلو في وقت واحد. عند استخدامه مع NVMe offload، `sub_group_size` يحدد متى يتم نقل حالات النموذج من وإلى ذاكرة وحدة المعالجة المركزية أثناء خطوة التحسين. يمنع هذا نفاد ذاكرة وحدة المعالجة المركزية (CPU) للنماذج الكبيرة للغاية. يمكن ترك `sub_group_size` بقيمته الافتراضية إذا لم تكن تستخدم NVMe offload، ولكن قد ترغب في تغييره إذا:

    1. واجه خطأ OOM أثناء خطوة المحسن. في هذه الحالة، قلل من `sub_group_size` لتقليل استخدام الذاكرة المؤقتة للذاكرة المؤقتة.
    2. تستغرق خطوة المحسن وقتًا طويلاً. في هذه الحالة، قم بزيادة `sub_group_size` لتحسين استخدام النطاق الترددي نتيجة زيادة مخازن البيانات.

* `reduce_bucket_size`، و`stage3_prefetch_bucket_size`، و`stage3_param_persistence_threshold` تعتمد على حجم مخفي للنموذج. يُنصح بتعيين هذه القيم إلى `auto` والسماح لـ [`Trainer`] بتعيين القيم تلقائيًا.

```yml
{
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true
    }
}
```

يمكنك استخدام [`deepspeed.zero.Init`](https://deepspeed.readthedocs.io/en/latest/zero3.html#deepspeed.zero.Init) كمدير سياق لتهيئة نموذج بشكل أسرع:

```py
from transformers import T5ForConditionalGeneration, T5Config
import deepspeed

with deepspeed.zero.Init():
    config = T5Config.from_pretrained("google-t5/t5-small")
    model = T5ForConditionalGeneration(config)
```

بالنسبة للنماذج المدربة مسبقًا، يجب أن يحتوي ملف تكوين DeepSped على "is_deepspeed_zero3_enabled: true" المحدد في [`TrainingArguments`] وأن يكون لديه تكوين ZeRO ممكّن. يجب إنشاء كائن [`TrainingArguments`] **قبل** استدعاء نموذج [`~PreTrainedModel.from_pretrained`].

```py
from transformers import AutoModel, Trainer, TrainingArguments

training_args = TrainingArguments(..., deepspeed=ds_config)
model = AutoModel.from_pretrained("google-t5/t5-small")
trainer = Trainer(model=model, args=training_args, ...)
```

ستحتاج إلى ZeRO-3 إذا لم تتسع الأوزان fp16 على GPU واحدة. إذا كنت قادرًا على تحميل أوزان fp16، فتأكد من تحديد "torch_dtype=torch.float16" في [`~PreTrainedModel.from_pretrained`].

هناك اعتبار آخر لـ ZeRO-3 وهو إذا كان لديك عدة وحدات GPU، ولا تحتوي وحدة GPU واحدة على جميع المعلمات ما لم تكن المعلمات للطبقة التي يتم تنفيذها حاليًا. للوصول إلى جميع المعلمات من جميع الطبقات في وقت واحد، مثل تحميل أوزان النموذج المسبق التدريب في [`~PreTrainedModel.from_pretrained`]، يتم تحميل طبقة واحدة في كل مرة وتقسيمها فورًا إلى جميع وحدات GPU. ويرجع ذلك إلى أنه بالنسبة للنماذج الكبيرة جدًا، لا يمكن تحميل الأوزان على وحدة GPU واحدة ثم توزيعها عبر وحدات GPU الأخرى بسبب قيود الذاكرة.

إذا صادفت وزن معلمة نموذج يشبه ما يلي، حيث "tensor([1.])" أو حجم المعلمة هو 1 بدلاً من شكل متعدد الأبعاد أكبر، فهذا يعني أن المعلمة مقسمة وهذا هو عنصر نائب ZeRO-3.

```py
tensor([1.0], device="cuda:0"، dtype=torch.float16، يتطلب_grad = صحيح)
```

<Tip>

للحصول على مزيد من المعلومات حول تهيئة النماذج الكبيرة باستخدام ZeRO-3 والوصول إلى المعلمات، راجع دليلي [بناء النماذج الضخمة](https://deepspeed.readthedocs.io/en/latest/zero3.html#constructing-massive-models) و [جمع المعلمات](https://deepspeed.readthedocs.io/en/latest/zero3.html#gathering-parameters).

</Tip>

### تكوين NVMe

تسمح [ZeRO-Infinity](https://hf.co/papers/2104.07857) بتفريغ حالات النموذج إلى وحدة المعالجة المركزية و/أو NVMe لتوفير المزيد من الذاكرة. تسمح خوارزميات التجزئة والتبليط الذكية لكل GPU بإرسال واستقبال كميات صغيرة جدًا من البيانات أثناء التفريغ بحيث يمكن لجهاز NVMe الحديث أن يحتوي على مجموعة ذاكرة إجمالية أكبر من الذاكرة المتاحة لعملية التدريب الخاصة بك. تتطلب ZeRO-Infinity ZeRO-3.

اعتمادًا على ذاكرة CPU و/أو NVMe المتوفرة، يمكنك تفريغ كل من [حالات المحسن](https://www.deepspeed.ai/docs/config-json/#optimizer-offloading) و [المعلمات](https://www.deepspeed.ai/docs/config-json/#parameter-offloading)، أو واحدة فقط منهما، أو لا شيء. يجب أيضًا التأكد من أن "nvme_path" يشير إلى جهاز NVMe، لأنه على الرغم من أنه لا يزال يعمل مع محرك أقراص ثابت عادي أو محرك أقراص صلبة، إلا أنه سيكون أبطأ بكثير. مع NVMe حديث، يمكنك توقع سرعات نقل ذروة تبلغ حوالي 3.5 جيجابايت/ثانية للقراءة وحوالي 3 جيجابايت/ثانية لعمليات الكتابة. أخيرًا، [قم بتشغيل معيار](https://github.com/microsoft/DeepSpeed/issues/998) على إعداد التدريب الخاص بك لتحديد تكوين "aio" الأمثل.

يحدد ملف تكوين ZeRO-3/Infinity أدناه معظم قيم المعلمات إلى "auto"، ولكن يمكنك أيضًا إضافة هذه القيم يدويًا.

```yml
{
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },

    "optimizer": {
        "type": "AdamW"،
        "params": {
            "lr": "auto"،
            "betas": "auto"،
            "eps": "auto"،
            "weight_decay": "auto"
        }
    },

    "scheduler": {
        "type": "WarmupLR"،
        "params": {
            "warmup_min_lr": "auto"،
            "warmup_max_lr": "auto"،
            "warmup_num_steps": "auto"
        }
    },

    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "nvme"،
            "nvme_path": "/local_nvme"،
            "pin_memory": true،
            "buffer_count": 4،
            "fast_init": false
        },
        "offload_param": {
            "device": "nvme"،
            "nvme_path": "/local_nvme"،
            "pin_memory": true،
            "buffer_count": 5،
            "buffer_size": 1e8،
            "max_in_cpu": 1e9
        }،
        "aio": {
            "block_size": 262144،
            "queue_depth": 32،
            "thread_count": 1،
            "single_submit": false،
            "overlap_events": true
        }،
        "overlap_comm": true،
        "contiguous_gradients": true،
        "sub_group_size": 1e9،
        "reduce_bucket_size": "auto"،
        "stage3_prefetch_bucket_size": "auto"،
        "stage3_param_persistence_threshold": "auto"،
        "stage3_max_live_parameters": 1e9،
        "stage3_max_reuse_distance": 1e9،
        "stage3_gather_16bit_weights_on_model_save": true
    }،

    "gradient_accumulation_steps": "auto"،
    "gradient_clipping": "auto"،
    "steps_per_print": 2000،
    "train_batch_size": "auto"،
    "train_micro_batch_size_per_gpu": "auto"،
    "wall_clock_breakdown": false
}
```

## ميزات DeepSpeed

هناك عدد من المعلمات المهمة التي يجب تحديدها في ملف تكوين DeepSpeed والتي يتم وصفها بإيجاز في هذا القسم.

### نقطة تفتيش التنشيط/التدرج

تتطلب نقطة تفتيش التنشيط والتدرج سرعة أكبر مقابل ذاكرة GPU أكبر، مما يتيح لك التغلب على السيناريوهات التي تنفد فيها ذاكرة GPU أو زيادة حجم دفعتك لتحقيق أداء أفضل. لتمكين هذه الميزة:

1. بالنسبة لنموذج Hugging Face، قم بتعيين "model.gradient_checkpointing_enable()" أو "--gradient_checkpointing" في [`Trainer`].
2. بالنسبة للنموذج غير Hugging Face، استخدم واجهة برمجة تطبيقات DeepSpeed [Activation Checkpointing API](https://deepspeed.readthedocs.io/en/latest/activation-checkpointing.html). يمكنك أيضًا استبدال رمز نمذجة المحولات واستبدال "torch.utils.checkpoint" بواجهة برمجة التطبيقات DeepSpeed. هذا النهج أكثر مرونة لأنه يمكنك تفريغ التنشيطات للأمام إلى ذاكرة وحدة المعالجة المركزية بدلاً من إعادة حسابها.

### المحسن والمجدول

يمكن مزج محسن DeepSpeed ومحول النماذج وجدولة مواعيده طالما أنك لا تقوم بتمكين "offload_optimizer". عندما يتم تمكين "offload_optimizer"، يمكنك استخدام محسن غير DeepSpeed (باستثناء LAMB) طالما أنه يحتوي على كل من التنفيذ CPU و GPU.
<Tip warning={true}>

يمكن تعيين معلمات المحسن والمجدول لملف التكوين من سطر الأوامر لتجنب الأخطاء التي يصعب العثور عليها. على سبيل المثال، إذا تم تعيين معدل التعلم على قيمة مختلفة في مكان آخر، فيمكنك تجاوزه من سطر الأوامر. وبصرف النظر عن معلمات المحسن والمجدول، سيتعين عليك التأكد من أن حجج سطر الأوامر لـ [`Trainer`] تتطابق مع تكوين DeepSpeed.

</Tip>

<hfoptions id="opt-sched">
<hfoption id="optimizer">

تقدم DeepSpeed العديد من [المحسنات](https://www.deepspeed.ai/docs/config-json/#optimizer-parameters) (Adam و AdamW و OneBitAdam و LAMB)، ولكن يمكنك أيضًا استيراد محسنات أخرى من PyTorch. إذا لم تقم بتكوين المحسن في التكوين، فإن [`Trainer`] يحدد تلقائيًا AdamW ويستخدم إما القيم المقدمة أو القيم الافتراضية للمعلمات التالية من سطر الأوامر: "lr"، "adam_beta1"، "adam_beta2"، "adam_epsilon"، "weight_decay".

يمكنك تعيين المعلمات إلى "auto" أو إدخال قيمك المرغوبة يدويًا.

```yaml
{
   "optimizer": {
       "type": "AdamW"،
       "params": {
         "lr": "auto"،
         "betas": "auto"،
         "eps": "auto"،
         "weight_decay": "auto"
       }
   }
}
```

يمكنك أيضًا استخدام محسن غير مدعوم عن طريق إضافة ما يلي إلى تكوين المستوى الأعلى.

```yaml
{
   "zero_allow_untested_optimizer": true
}
```

من DeepSpeed==0.8.3، إذا كنت تريد استخدام التفريغ، فستحتاج أيضًا إلى إضافة ما يلي إلى تكوين المستوى الأعلى لأن التفريغ يعمل بشكل أفضل مع محسن CPU Adam الخاص بـ DeepSpeed.

```yaml
{
   "zero_force_ds_cpu_optimizer": false
}
```

</hfoption>
<hfoption id="scheduler">

تدعم DeepSpeed مجدول معدلات التعلم LRRangeTest و OneCycle و WarmupLR و WarmupDecayLR [schedulers](https://www.deepspeed.ai/docs/config-json/#scheduler-parameters).

يوفر المحولون و DeepSpeed نفس المجدولين:

* WarmupLR هو نفسه مثل `--lr_scheduler_type constant_with_warmup` في المحولون
* WarmupDecayLR هو نفسه مثل `--lr_scheduler_type linear` في المحولون (هذا هو المجدول الافتراضي المستخدم في المحولون)

إذا لم تقم بتكوين المجدول في التكوين، فإن [`Trainer`] يحدد تلقائيًا WarmupDecayLR ويستخدم إما القيم المقدمة أو القيم الافتراضية للمعلمات التالية من سطر الأوامر: "warmup_min_lr"، "warmup_max_lr"، "warmup_num_steps"، "total_num_steps" (يتم حسابها تلقائيًا أثناء وقت التشغيل إذا لم يتم توفير "max_steps").

يمكنك تعيين المعلمات إلى "auto" أو إدخال قيمك المرغوبة يدويًا.

```yaml
{
   "scheduler": {
         "type": "WarmupDecayLR"،
         "params": {
             "total_num_steps": "auto"،
             "warmup_min_lr": "auto"،
             "warmup_max_lr": "auto"،
             "warmup_num_steps": "auto"
         }
     }
}
```

</hfoption>
</hfoptions>

### الدقة

تدعم Deepspeed الدقة fp32 و fp16 و bf16 المختلطة.

<hfoptions id="precision">
<hfoption id="fp32">

إذا لم يعمل نموذجك بشكل جيد مع الدقة المختلطة، على سبيل المثال إذا لم يتم تدريبه مسبقًا في الدقة المختلطة، فقد تواجه مشكلات في الفيض أو نقصان قد يتسبب في فقدان NaN. في هذه الحالات، يجب استخدام الدقة fp32 الكاملة عن طريق تعطيل وضع fp16 الافتراضي بشكل صريح.

```yaml
{
    "fp16": {
        "enabled": false
    }
}
```

بالنسبة لوحدات GPU Ampere و PyTorch > 1.7، فإنه يتحول تلقائيًا إلى تنسيق [tf32] الأكثر كفاءة (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices) لبعض العمليات ولكن النتائج لا تزال في fp32. يمكنك التحكم فيه من [`Trainer`] عن طريق تعيين "--tf32" لتمكينه، و "--tf32 0" أو "--no_tf32" لتعطيله.

</hfoption>
<hfoption id="fp16">
### تكوين fp16 مماثل لـ PyTorch AMP

لتقليل استخدام الذاكرة وتسريع سرعة التدريب، يمكنك تكوين الدقة المختلطة fp16 المشابهة لـ PyTorch AMP. يقوم [Trainer] تلقائيًا بتمكين أو تعطيل fp16 بناءً على قيمة args.fp16_backend، ويمكنك تعيين باقي التكوين. يتم تمكين fp16 من سطر الأوامر عند تمرير الحجج التالية: --fp16، --fp16_backend amp أو --fp16_full_eval.

```yaml
{
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    }
}
```

للحصول على خيارات تدريب DeepSpeed fp16 الإضافية، راجع مرجع [خيارات تدريب FP16](https://www.deepspeed.ai/docs/config-json/#fp16-training-options).

### تكوين دقة مختلطة fp16 مماثلة لـ Apex

لإعداد دقة مختلطة fp16 مماثلة لـ Apex، قم بضبط التكوين كما هو موضح أدناه باستخدام "auto" أو قيمك الخاصة. يقوم [Trainer] تلقائيًا بتكوين amp بناءً على قيم args.fp16_backend وargs.fp16_opt_level. يمكن أيضًا تمكينه من سطر الأوامر عند تمرير الحجج التالية: --fp16، --fp16_backend apex أو --fp16_opt_level 01.

```yaml
{
    "amp": {
        "enabled": "auto",
        "opt_level": "auto"
    }
}
```

### استخدام bf16

لاستخدام bf16، ستحتاج إلى DeepSpeed==0.6.0 على الأقل. يحتوي bf16 على نفس النطاق الديناميكي مثل fp32 ولا يتطلب ضبط مقياس الخسارة. ومع ذلك، إذا كنت تستخدم [تراكم الخرج](#gradient-accumulation) مع bf16، يتم تراكم الخرج في bf16، والذي قد لا يكون مرغوبًا فيه لأن تنسيق الدقة المنخفضة هذا يمكن أن يؤدي إلى تراكم الخسارة.

يمكن إعداد bf16 في ملف التكوين أو تمكينه من سطر الأوامر عند تمرير الحجج التالية: --bf16 أو --bf16_full_eval.

```yaml
{
    "bf16": {
        "enabled": "auto"
    }
}
```

### حجم الدُفعة

يمكن تكوين حجم الدفعة تلقائيًا أو تحديده صراحةً. إذا اخترت استخدام خيار "auto"، فسيقوم [Trainer] بتعيين train_micro_batch_size_per_gpu إلى قيمة args.per_device_train_batch_size وtrain_batch_size إلى args.world_size * args.per_device_train_batch_size * args.gradient_accumulation_steps.

```yaml
{
    "train_micro_batch_size_per_gpu": "auto",
    "train_batch_size": "auto"
}
```

### تراكم الخرج

يمكن تكوين تراكم الخرج تلقائيًا أو تحديده صراحةً. إذا اخترت استخدام خيار "auto"، فسيقوم [Trainer] بتعيينه إلى قيمة args.gradient_accumulation_steps.

```yaml
{
    "gradient_accumulation_steps": "auto"
}
```

### قص الخرج

يمكن تكوين قص الخرج تلقائيًا أو تحديده صراحةً. إذا اخترت استخدام خيار "auto"، فسيقوم [Trainer] بتعيينه إلى قيمة args.max_grad_norm.

```yaml
{
    "gradient_clipping": "auto"
}
```

### نوع بيانات الاتصال

بالنسبة لعمليات الجمعيات التواصلية مثل عمليات التخفيض والجمع والتشتت، يتم استخدام نوع بيانات منفصل.

يتم تنفيذ جميع عمليات الجمع والتشتت بنفس نوع البيانات الموجودة في البيانات. على سبيل المثال، إذا كنت تتدرب باستخدام bf16، فسيتم أيضًا جمع البيانات في bf16 لأن التجميع عملية غير مسببة للخسارة.

عمليات التخفيض مسببة للخسارة، على سبيل المثال عند حساب متوسط الخرج عبر وحدات معالجة الرسوميات (GPU) متعددة. عندما يتم تنفيذ الاتصال في fp16 أو bf16، من المحتمل أن يكون مسببًا للخسارة لأن إضافة أرقام متعددة في دقة منخفضة ليست دقيقة. وينطبق هذا بشكل خاص على bf16 الذي يتمتع بدقة أقل من fp16. لهذا السبب، يكون fp16 هو الافتراضي لعمليات التخفيض لأن الخسارة ضئيلة عند حساب متوسط الخرج.

يمكنك اختيار نوع بيانات الاتصال عن طريق تعيين معلمة communication_data_type في ملف التكوين. على سبيل المثال، يؤدي اختيار fp32 إلى إضافة قدر صغير من النفقات العامة، ولكنه يضمن أن يتم تراكم عملية التخفيض في fp32 وعند جاهزته، يتم تحويله إلى تنسيق الدقة النصفية الذي تتدرب فيه.

```yaml
{
    "communication_data_type": "fp32"
}
```

## النشر
```yaml
{
    "communication_data_type": "fp32"
}
```

## النشر

يمكن نشر DeepSpeed بواسطة برامج الإطلاق المختلفة مثل [torchrun](https://pytorch.org/docs/stable/elastic/run.html)، أو برنامج إطلاق DeepSpeed، أو [Accelerate](https://huggingface.co/docs/accelerate/basic_tutorials/launch#using-accelerate-launch). لنشره، أضف --deepspeed ds_config.json إلى سطر أوامر [Trainer]. يُنصح باستخدام أداة DeepSpeed [`add_config_arguments`](https://deepspeed.readthedocs.io/en/latest/initialize.html#argument-parsing) لإضافة أي حجج سطر أوامر ضرورية إلى رمزك.

سيوضح لك هذا الدليل كيفية نشر DeepSpeed باستخدام برنامج إطلاق DeepSpeed لمجموعات التدريب المختلفة. يمكنك الاطلاع على هذا [المنشور](https://github.com/huggingface/transformers/issues/8771#issuecomment-759248400) للحصول على أمثلة استخدام أكثر عملية.

### النشر على وحدات معالجة الرسوميات (GPU) المتعددة

لنشر DeepSpeed على وحدات معالجة الرسوميات (GPU) المتعددة، أضف معلمة --num_gpus. إذا كنت تريد استخدام جميع وحدات معالجة الرسوميات (GPU) المتوفرة، فلا يلزم إضافة --num_gpus. يستخدم المثال أدناه وحدتي معالجة رسوميات (GPU).

```bash
deepspeed --num_gpus=2 examples/pytorch/translation/run_translation.py \
--deepspeed tests/deepspeed/ds_config_zero3.json \
--model_name_or_path google-t5/t5-small --per_device_train_batch_size 1 \
--output_dir output_dir --overwrite_output_dir --fp16 \
--do_train --max_train_samples 500 --num_train_epochs 1 \
--dataset_name wmt16 --dataset_config "ro-en" \
--source_lang en --target_lang ro
```

### النشر على وحدة معالجة رسوميات (GPU) واحدة

لنشر DeepSpeed على وحدة معالجة رسوميات (GPU) واحدة، أضف معلمة --num_gpus. لا يلزم تعيين هذه القيمة صراحةً إذا كان لديك وحدة معالجة رسوميات (GPU) واحدة فقط لأن DeepSpeed ينشر جميع وحدات معالجة الرسوميات (GPU) التي يمكنه رؤيتها على عقدة معينة.

```bash
deepspeed --num_gpus=1 examples/pytorch/translation/run_translation.py \
--deepspeed tests/deepspeed/ds_config_zero2.json \
--model_name_or_path google-t5/t5-small --per_device_train_batch_size 1 \
--output_dir output_dir --overwrite_output_dir --fp16 \
--do_train --max_train_samples 500 --num_train_epochs 1 \
--dataset_name wmt16 --dataset_config "ro-en" \
--source_lang en --target_lang ro
```

ما زال DeepSpeed مفيدًا مع وحدة معالجة رسوميات (GPU) واحدة فقط لأنه يمكنك:

1. نقل بعض الحسابات والذاكرة إلى وحدة المعالجة المركزية (CPU) لتحرير المزيد من موارد وحدة معالجة الرسوميات (GPU) لكي يستخدمها نموذجك لحجم دفعة أكبر أو نموذج كبير جدًا لا يمكنه عادةً أن يتناسب.
2. تقليل تجزئة الذاكرة باستخدام نظام إدارة ذاكرة وحدة معالجة الرسوميات (GPU) الذكي الذي يسمح أيضًا بتناسب النماذج والدفعات الأكبر حجمًا.

<Tip>

قم بتعيين قيم allgather_bucket_size وreduce_bucket_size إلى 2e8 في ملف تكوين [Zero-2](#zero-configuration) للحصول على أداء أفضل على وحدة معالجة رسوميات (GPU) واحدة.

</Tip>

### النشر على عدة عقد

العقدة هي وحدة معالجة رسوميات (GPU) واحدة أو أكثر لتشغيل عبء العمل. يعد الإعداد متعدد العقد إعدادًا أكثر قوة ويمكن إطلاقه باستخدام برنامج إطلاق DeepSpeed. بالنسبة لهذا الدليل، دعنا نفترض وجود عقدتين بثماني وحدات معالجة رسوميات (GPU) لكل منهما. يمكن الوصول إلى العقدة الأولى باستخدام ssh hostname1 والعقدة الثانية باستخدام ssh hostname2. يجب أن تتمكن كلتا العقدتين من التواصل مع بعضهما البعض محليًا عبر ssh بدون كلمة مرور.

يتوقع DeepSpeed افتراضيًا أن يستخدم إعدادك متعدد العقد تخزينًا مشتركًا. إذا لم يكن الأمر كذلك ولا يمكن لكل عقدة سوى رؤية نظام الملفات المحلي، فيجب عليك ضبط ملف التكوين لتضمين ["checkpoint"](https://www.deepspeed.ai/docs/config-json/#checkpoint-options) للسماح بالتحميل بدون الوصول إلى نظام ملفات مشترك:

```yaml
{
  "checkpoint": {
    "use_node_local_storage": true
  }
}
```

يمكنك أيضًا استخدام حجة `--save_on_each_node` الخاصة بـ [Trainer] لإضافة "checkpoint" أعلاه تلقائيًا إلى تكوينك.

### استخدام torchrun

بالنسبة لـ [torchrun](https://pytorch.org/docs/stable/elastic/run.html)، يجب عليك تسجيل الدخول إلى كل عقدة عبر ssh وتشغيل الأمر التالي على كل منهما. ينتظر برنامج الإطلاق حتى تتم مزامنة كلتا العقدتين قبل بدء التدريب.
```bash
torchrun --nproc_per_node=8 --nnode=2 --node_rank=0 --master_addr=hostname1 \
--master_port=9901 your_program.py <normal cl args> --deepspeed ds_config.json
```

### استخدام برنامج إطلاق DeepSpeed

بالنسبة لبرنامج إطلاق DeepSpeed، ابدأ بإنشاء ملف hostfile.

```bash
hostname1 slots=8
hostname2 slots=8
```

بعد ذلك، يمكنك بدء التدريب باستخدام الأمر التالي. يقوم برنامج إطلاق DeepSpeed تلقائيًا ببدء الأمر على كلتا العقدتين في نفس الوقت.

```bash
deepspeed --num_gpus 8 --num_nodes 2 --hostfile hostfile --master_addr hostname1 --master_port=9901 \
your_program.py <normal cl args> --deepspeed ds_config.json
```

راجع دليل [تكوين الموارد (متعدد العقد)](https://www.deepspeed.ai/getting-started/#resource-configuration-multi-node) للحصول على مزيد من التفاصيل حول تكوين موارد الحوسبة متعددة العقد.

### SLURM

في بيئة SLURM، سيتعين عليك تكييف نص SLURM النصي مع بيئة SLURM الخاصة بك. قد يبدو مثال لنص SLURM على النحو التالي:

```bash
#SBATCH --job-name=test-nodes        # الاسم
#SBATCH --nodes=2                    # العقد
#SBATCH --ntasks-per-node=1          # حاسم - مهمة واحدة فقط لكل توزيع لكل عقدة!
#SBATCH --cpus-per-task=10           # عدد الأنوية لكل المهام
#SBATCH --gres=gpu:8                 # عدد وحدات معالجة الرسوميات (GPU)
#SBATCH --time 20:00:00              # وقت التنفيذ الأقصى (ساعة:دقيقة:ثانية)
#SBATCH --output=%x-%j.out           # اسم ملف الإخراج

export GPUS_PER_NODE=8
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=9901

srun --jobid $SLURM_JOBID bash -c 'python -m torch.distributed.run \
 --nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID \
 --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
your_program.py <normal cl args> --deepspeed ds_config.json'
```

بعد ذلك، يمكنك جدولة نشرك متعدد العقد باستخدام الأمر التالي الذي يبدأ التدريب في وقت واحد على جميع العقد.

```bash
sbatch launch.slurm
```

### دفتر الملاحظات

لا يدعم برنامج إطلاق DeepSpeed النشر من دفتر الملاحظات، لذلك سيتعين عليك محاكاة بيئة موزعة. ومع ذلك، فإن هذا يعمل فقط لوحدة معالجة رسوميات (GPU) واحدة. إذا كنت تريد استخدام أكثر من وحدة معالجة رسوميات (GPU) واحدة، فيجب عليك استخدام بيئة متعددة العمليات لكي يعمل DeepSpeed. وهذا يعني أنه يتعين عليك استخدام برنامج إطلاق DeepSpeed الذي لا يمكن محاكاته كما هو موضح هنا.

```py
# يتطلب DeepSpeed بيئة موزعة حتى عند استخدام عملية واحدة فقط.
# هذه تحاكي برنامج الإطلاق في دفتر الملاحظات
import os

os.environ["MASTER_ADDR"] = "localhost"
osMultiplier.environ["MASTER_PORT"] = "9994"  # غيّره إذا حدث خطأ " Runaway: Address already in use"
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"

# الآن تابع كالمعتاد، بالإضافة إلى تمرير ملف تكوين DeepSpeed
training_args = TrainingArguments(..., deepspeed="ds_config_zero3.json")
trainer = Trainer(...)
trainer.train()
```

إذا كنت تريد إنشاء ملف التكوين أثناء التنقل في دفتر الملاحظات في الدليل الحالي، فيمكنك تخصيص خلية لذلك.

```py
%%bash
cat <<'EOT' > ds_config_zero3.json
{
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },

    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto"
        }
    },

    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto"
        }
    },

    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true
    },

    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 2000,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false
}
EOT
```

إذا كان نص البرنامج النصي للتدريب موجودًا في ملف وليس في خلية دفتر ملاحظات، فيمكنك تشغيل `deepspeed` بشكل طبيعي من shell في خلية دفتر ملاحظات. على سبيل المثال، لتشغيل `run_translation.py`:

```py
!git clone https://github.com/huggingface/transformers
!cd transformers؛ deepspeed examples/pytorch/translation/run_translation.py ...
```

يمكنك أيضًا استخدام `%%bash` magic وكتابة التعليمات البرمجية متعددة الأسطر لتشغيل برنامج shell، ولكن لن تتمكن من عرض السجلات حتى اكتمال التدريب. مع `%%bash` magic، لا تحتاج إلى محاكاة بيئة موزعة.

```py
%%bash

git clone https://github.com/huggingface/transformers
cd transformers
deepspeed examples/pytorch/translation/run_translation.py ...
```

## حفظ أوزان النموذج

يحتفظ DeepSpeed بأوزان الدقة الكاملة الرئيسية fp32 في ملفات نقطة تفتيش محسن مخصصة (نمط glob يشبه `global_step*/*optim_states.pt`) ويتم حفظها في نقطة تفتيش عادية.

<hfoptions id="save">
<hfoption id="fp16">

يحفظ النموذج الذي تم تدريبه باستخدام ZeRO-2 أوزان pytorch_model.bin في fp16. لحفظ أوزان النموذج في fp16 لنموذج تم تدريبه باستخدام ZeRO-3، يجب عليك تعيين `"stage3_gather_16bit_weights_on_model_save": true` لأن أوزان النموذج مجزأة عبر وحدات معالجة الرسومات متعددة. وإلا، فإن [`Trainer`] لن يحفظ الأوزان في fp16 ولن يقوم بإنشاء ملف pytorch_model.bin. هذا لأن حالة DeepSpeed's state_dict تحتوي على عنصر نائب بدلاً من الأوزان الحقيقية ولن تتمكن من تحميلها.

```yaml
{
    "zero_optimization": {
        "stage3_gather_16bit_weights_on_model_save": true
    }
}
```

</hfoption>
<hfoption id="fp32">

لا يجب حفظ أوزان الدقة الكاملة أثناء التدريب لأنها قد تتطلب الكثير من الذاكرة. من الأفضل عادةً حفظ أوزان fp32 دون اتصال بمجرد اكتمال التدريب. ولكن إذا كان لديك الكثير من ذاكرة وحدة المعالجة المركزية (CPU) المجانية، فيمكن حفظ أوزان الدقة الكاملة أثناء التدريب. يغطي هذا القسم كلاً من النهج عبر الإنترنت وغير المتصل.

### عبر الإنترنت Online

يجب أن يكون قد تم حفظ نقطة تفتيش واحدة على الأقل لتحميل أحدث نقطة تفتيش كما هو موضح فيما يلي:

```py
from transformers.trainer_utils import get_last_checkpoint
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint

checkpoint_dir = get_last_checkpoint(trainer.args.output_dir)
fp32_model = load_state_dict_from_zero_checkpoint(trainer.model، checkpoint_dir)
```

إذا قمت بتمكين المعلمة `--load_best_model_at_end` لتتبع أفضل نقطة تفتيش في [`TrainingArguments`]]، فيمكنك إنهاء التدريب أولاً وحفظ النموذج النهائي بشكل صريح. بعد ذلك، يمكنك إعادة تحميله كما هو موضح أدناه:

```py
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint

checkpoint_dir = os.path.join(trainer.args.output_dir، "checkpoint-final")
trainer.deepspeed.save_checkpoint(checkpoint_dir)
fp32_model = load_state_dict_from_zero_checkpoint(trainer.model، checkpoint_dir)
```

<Tip>

بمجرد تشغيل load_state_dict_from_zero_checkpoint، لم يعد النموذج قابلًا للاستخدام في DeepSpeed في سياق نفس التطبيق. ستحتاج إلى تهيئة محرك DeepSpeed مرة أخرى لأن model.load_state_dict(state_dict) تزيل كل سحر DeepSpeed منه. استخدم هذا فقط في نهاية التدريب.

</Tip>

يمكنك أيضًا استخراج وحمل حالة fp32 الأوزان:

```py
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir) # موجود بالفعل على وحدة المعالجة المركزية
model = model.cpu()
model.load_state_dict(state_dict)
```

### غير متصل Offline

يوفر DeepSpeed نص برمجي zero_to_fp32.py في المستوى الأعلى لمجلد checkpoint لاستخراج الأوزان في أي نقطة. هذا نص برمجي مستقل ولا تحتاج إلى ملف تكوين أو [Trainer].

على سبيل المثال، إذا كان مجلد checkpoint الخاص بك يبدو كالتالي:

```bash
$ ls -l output_dir/checkpoint-1/
-rw-rw-r-- 1 stas stas 1.4K Mar 27 20:42 config.json
drwxrwxr-x 2 stas stas 4.0K Mar 25 19:52 global_step1/
-rw-rw-r-- Multiplier stas 12 Mar 27 13:16 latest
-rw-rw-r-- 1 stas stas 827K Mar 27 20:42 optimizer.pt
-rw-rw-r-- 1 stas stas 231M Mar 27 20:42 pytorch_model.bin
-rw-rw-r-- 1 stas stas 623 Mar 27 20:42 scheduler.pt
-rw-rw-r-- 1 stas stas 1.8K Mar 27 20:42 special_tokens_map.json
-rw-rw-r-- 1 stas stas 774K Mar 27 20:42 spiece.model
-rw-rw-r-- 1 stas stas 1.9K Mar 27 20:42 tokenizer_config.json
-rw-rw-r-- 1 stas stas 339 Mar 27 20:42 trainer_state.json
-rw-rw-r-- 1 stas stas 2.3K Mar 27 20:42 training_args.bin
-rwxrw-r-- 1 stas stas 5.5K Mar 27 13:16 zero_to_fp32.py*
```

لإعادة بناء أوزان fp32 من نقطة تفتيش DeepSpeed (ZeRO-2 أو ZeRO-3) subfolder `global_step1`، قم بتشغيل الأمر التالي لإنشاء وتوحيد أوزان fp32 الكاملة من وحدات GPU متعددة في ملف pytorch_model.bin واحد. يكتشف النص البرمجي تلقائيًا المجلد الفرعي الذي يحتوي على نقطة التفتيش.

```py
python zero_to_fp32.py . pytorch_model.bin
```

<Tip>

قم بتشغيل `python zero_to_fp32.py -h` لمزيد من تفاصيل الاستخدام. يتطلب النص البرمجي 2x RAM العام لأوزان fp32 النهائية.

</Tip>

</hfoption>
</hfoptions>

## ZeRO Inference

[استنتاج الصفر](https://www.deepspeed.ai/2022/09/09/zero-inference.html) يضع أوزان النموذج في ذاكرة CPU أو NVMe لتجنب إرهاق وحدة معالجة الرسومات GPU مما يجعل من الممكن تشغيل الاستدلال باستخدام نماذج ضخمة على وحدة معالجة الرسومات GPU. لا يتطلب الاستدلال أي كميات إضافية كبيرة من الذاكرة لحالات المحسن والتدرجات، لذا يمكنك ملاءمة دفعات و/أو تسلسلات أطول على نفس الأجهزة.

يستخدم استنتاج الصفر نفس ملف التكوين مثل [الصفر-3](#zero-configuration)، وتكوينات الصفر-2 والصفر-1 لا تعمل لأنها لا توفر أي فوائد للاستدلال.

لتشغيل استدلال الصفر، مرر حجج التدريب المعتادة إلى فئة [TrainingArguments] وأضف الحجة --do_eval.

```bash
deepspeed --num_gpus=2 your_program.py <normal cl args> --do_eval --deepspeed ds_config.json
```

## تكامل DeepSpeed غير المدرب

يعمل DeepSpeed أيضًا مع Transformers بدون فئة [Trainer]. تتم معالجة هذا بواسطة [HfDeepSpeedConfig] التي تهتم فقط بجمع معلمات الصفر-3 وتقسيم نموذج عبر وحدات معالجة الرسومات GPU متعددة عند استدعاء [~ PreTrainedModel.from_pretrained].

<Tip>

إذا كنت تريد أن تتم معالجة كل شيء تلقائيًا، فجرّب استخدام DeepSpeed مع [Trainer]! ستحتاج إلى اتباع [توثيق DeepSpeed](https://www.deepspeed.ai/)، وقم يدويًا بتكوين قيم المعلمات في ملف التكوين (لا يمكنك استخدام القيمة "auto").

</Tip>

لنشر الصفر-3 بكفاءة، يجب عليك إنشاء مثيل كائن [HfDeepSpeedConfig] قبل النموذج والاحتفاظ بذلك الكائن نشطًا:

<hfoptions id="models">
<hfoption id="pretrained model">

```py
from transformers.integrations import HfDeepSpeedConfig
from transformers import AutoModel
import deepspeed
<hfoptions id="models">
<hfoption id="pretrained model">

```py
from transformers.integrations import HfDeepSpeedConfig
from transformers import AutoModel
import deepspeed

ds_config = {...} # كائن تكوين DeepSpeed أو المسار إلى الملف
# يجب تشغيله قبل إنشاء مثيل النموذج للكشف عن الصفر 3
dschf = HfDeepSpeedConfig(ds_config) # احتفظ بهذا الكائن نشطًا
model = AutoModel.from_pretrained("openai-community/gpt2")
engine = deepspeed.initialize(model=model, config_params=ds_config, ...)
```

</hfoption>
<hfoption id="non-pretrained model">

[HfDeepSpeedConfig] غير مطلوب للصفر-1 أو الصفر-2.

```py
from transformers.integrations import HfDeepSpeedConfig
from transformers import AutoModel, AutoConfig
import deepspeed

ds_config = {...} # كائن تكوين DeepSpeed أو المسار إلى الملف
# يجب تشغيله قبل إنشاء مثيل النموذج للكشف عن الصفر 3
dschf = HfDeepSpeedConfig(ds_config) # احتفظ بهذا الكائن نشطًا
config = AutoConfig.from_pretrained("openai-community/gpt2")
model = AutoModel.from_config(config)
engine = deepspeed.initialize(model=model, config_params=ds_config, ...)
```

</hfoption>
</hfoptions>

### ZeRO Inference غير المدرب

لتشغيل استدلال الصفر بدون [مدرب] في الحالات التي لا يمكنك فيها وضع نموذج على وحدة معالجة الرسومات GPU واحدة، جرّب استخدام وحدات معالجة الرسومات GPU الإضافية و/أو التفريغ إلى ذاكرة CPU. الدقة المهمة التي يجب فهمها هنا هي أنه يمكن لمعمارية الصفر معالجة مدخلات مختلفة على وحدات معالجة الرسومات GPU المختلفة بالتوازي.

تأكد من:

* تعطيل التفريغ إلى وحدة المعالجة المركزية إذا كان لديك ذاكرة GPU كافية (نظرًا لأنه يبطئ الأمور).
* تمكين bf16 إذا كان لديك GPU Ampere أو أحدث للتسريع. إذا لم يكن لديك أحد هذه المعالجات الرسومية، فيمكنك تمكين fp16 طالما أنك لا تستخدم نموذجًا مدربًا في bf16 (نماذج T5) لأنه قد يؤدي إلى خطأ في الفيض.

الق نظرة على النص البرمجي التالي للحصول على فكرة أفضل حول كيفية تشغيل استدلال الصفر بدون [مدرب] على نموذج لن يناسب وحدة معالجة الرسومات GPU واحدة.

```py
#!/usr/bin/env python

# This script demonstrates how to use Deepspeed ZeRO in an inference mode when one can't fit a model
# into a single GPU
#
# 1. Use 1 GPU with CPU offload
# 2. Or use multiple GPUs instead
#
# First you need to install deepspeed: pip install deepspeed
#
# Here we use a 3B "bigscience/T0_3B" model which needs about 15GB GPU RAM - so 1 largish or 2
# small GPUs can handle it. or 1 small GPU and a lot of CPU memory.
#
# To use a larger model like "bigscience/T0" which needs about 50GB, unless you have an 80GB GPU -
# you will need 2-4 gpus. And then you can adapt the script to handle more gpus if you want to
# process multiple inputs at once.
#
# The provided deepspeed config also activates CPU memory offloading, so chances are that if you
# have a lot of available CPU memory and you don't mind a slowdown you should be able to load a
# model that doesn't normally fit into a single GPU. If you have enough GPU memory the program will
# run faster if you don't want offload to CPU - so disable that section then.
#
# To deploy on 1 gpu:
#
# deepspeed --num_gpus 1 t0.py
# or:
# python -m torch.distributed.run --nproc_per_node=1 t0.py
#
# To deploy on 2 gpus:
#
# deepspeed --num_gpus 2 t0.py
# or:
# python -m torch.distributed.run --nproc_per_node=2 t0.py

from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM
from transformers.integrations import HfDeepSpeedConfig
import deepspeed
import os
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # To avoid warnings about parallelism in tokenizers

# distributed setup
local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))
torch.cuda.set_device(local_rank)
deepspeed.init_distributed()

model_name = "bigscience/T0_3B"

config = AutoConfig.from_pretrained(model_name)
model_hidden_size = config.d_model

# batch size has to be divisible by world_size, but can be bigger than world_size
train_batch_size = 1 * world_size

# ds_config notes
#
# - enable bf16 if you use Ampere or higher GPU - this will run in mixed precision and will be
# faster.
#
# - for older GPUs you can enable fp16, but it'll only work for non-bf16 pretrained models - e.g.
# all official t5 models are bf16-pretrained
#
# - set offload_param.device to "none" or completely remove the `offload_param` section if you don't
# - want CPU offload
#
# - if using `offload_param` you can manually finetune stage3_param_persistence_threshold to control
# - which params should remain on gpus - the larger the value the smaller the offload size
#
# For in-depth info on Deepspeed config see
# https://huggingface.co/docs/transformers/main/main_classes/deepspeed

# keeping the same format as json for consistency, except it uses lower case for true/false
# fmt: off
ds_config = {
    "fp16": {
        "enabled": False
    },
    "bf16": {
        "enabled": False
    },
    "zero_optimization": {
        "stage": 3,
        "offload_param": {
            "device": "cpu",
            "pin_memory": True
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": model_hidden_size * model_hidden_size,
        "stage3_prefetch_bucket_size": 0.9 * model_hidden_size * model_hidden_size,
        "stage3_param_persistence_threshold": 10 * model_hidden_size
    },
    "steps_per_print": 2000,
    "train_batch_size": train_batch_size,
    "train_micro_batch_size_per_gpu": 1,
    "wall_clock_breakdown": False
}
# fmt: on

# next line instructs transformers to partition the model directly over multiple gpus using
# deepspeed.zero.Init when model's `from_pretrained` method is called.
#
# **it has to be run before loading the model AutoModelForSeq2SeqLM.from_pretrained(model_name)**
#
# otherwise the model will first be loaded normally and only partitioned at forward time which is
# less efficient and when there is little CPU RAM may fail
dschf = HfDeepSpeedConfig(ds_config)  # keep this object alive

# now a model can be loaded.
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# initialise Deepspeed ZeRO and store only the engine object
ds_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
ds_engine.module.eval()  # inference

# Deepspeed ZeRO can process unrelated inputs on each GPU. So for 2 gpus you process 2 inputs at once.
# If you use more GPUs adjust for more.
# And of course if you have just one input to process you then need to pass the same string to both gpus
# If you use only one GPU, then you will have only rank 0.
rank = torch.distributed.get_rank()
if rank == 0:
    text_in = "Is this review positive or negative? Review: this is the best cast iron skillet you will ever buy"
elif rank == 1:
    text_in = "Is this review positive or negative? Review: this is the worst restaurant ever"

tokenizer = AutoTokenizer.from_pretrained(model_name)
inputs = tokenizer.encode(text_in, return_tensors="pt").to(device=local_rank)
with torch.no_grad():
    outputs = ds_engine.module.generate(inputs, synced_gpus=True)
text_out = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"rank{rank}:\n   in={text_in}\n  out={text_out}")
```
احفظ البرنامج النصي باسم t0.py وابدأ تشغيله:

```bash
$ deepspeed --num_gpus 2 t0.py
rank0:
   in=Is this review positive or negative? Review: this is the best cast iron skillet you will ever buy
  out=Positive
rank1:
   in=Is this review positive or negative? Review: this is the worst restaurant ever
  out=negative
```

هذا مثال بسيط جدًا، وستحتاج إلى تكييفه مع حالتك الاستخدام.

### توليد

يتطلب استخدام وحدات معالجة الرسومات متعددة باستخدام ZeRO-3 للمولد مزامنة وحدات معالجة الرسومات عن طريق تعيين `synced_gpus=True` في طريقة [`~GenerationMixin.generate`]. وإلا، إذا انتهت إحدى وحدات معالجة الرسومات من التوليد قبل الأخرى، فسيعلق النظام بأكمله لأن وحدات معالجة الرسومات المتبقية لم تتلق شظية الوزن من وحدة معالجة الرسومات التي انتهت أولاً.

بالنسبة لـ Transformers>=4.28، إذا تم تعيين `synced_gpus` تلقائيًا على `True` إذا تم اكتشاف وحدات معالجة الرسومات المتعددة أثناء التوليد.

## استكشاف الأخطاء وإصلاحها

عندما تواجه مشكلة، يجب أن تفكر فيما إذا كان DeepSpeed هو سبب المشكلة لأنه غالبًا ما لا يكون كذلك (ما لم يكن من الواضح جدًا ويمكنك رؤية وحدات DeepSpeed في الاستثناء)! يجب أن تكون الخطوة الأولى هي إعادة المحاولة دون DeepSpeed، وإذا استمرت المشكلة، فيمكنك الإبلاغ عن المشكلة. إذا كانت المشكلة متعلقة بمشكلة أساسية في DeepSpeed وغير مرتبطة بتكامل Transformers، فقم بفتح مشكلة في [مستودع DeepSpeed](https://github.com/microsoft/DeepSpeed).

بالنسبة للمشكلات المتعلقة بتكامل Transformers، يرجى تقديم المعلومات التالية:

* ملف تكوين DeepSpeed الكامل

* وسائط سطر الأوامر لـ [`Trainer`]]، أو وسائط [`TrainingArguments`] إذا كنت تقوم بكتابة برنامج نصي لإعداد [`Trainer`] بنفسك (لا تقم بإلقاء [`TrainingArguments`] الذي يحتوي على عشرات الإدخالات غير ذات الصلة)

* الإخراج من:

```bash
python -c 'import torch؛ print(f"torch: {torch.__version__}")'
python -c 'import transformers؛ print(f"transformers: {transformers.__version__}")'
python -c 'import deepspeed؛ print(f"deepspeed: {deepspeed.__version__}")'
```

* رابط إلى Google Colab دفتر ملاحظات لإعادة إنتاج المشكلة

* إذا كان من المستحيل، مجموعة بيانات قياسية وغير مخصصة يمكننا استخدامها أيضًا لمحاولة استخدام مثال موجود لإعادة إنتاج المشكلة معه

توفر الأقسام التالية دليلًا لحل اثنتين من أكثر المشكلات شيوعًا.

### عملية DeepSpeed قتلت عند بدء التشغيل

عندما يتم إنهاء عملية DeepSpeed أثناء الإطلاق دون تتبع المكدس، فهذا يعني عادةً أن البرنامج حاول تخصيص المزيد من ذاكرة وحدة المعالجة المركزية (CPU) أكثر مما لدى نظامك أو حاولت العملية تخصيص المزيد من ذاكرة وحدة المعالجة المركزية (CPU) مما يؤدي إلى قيام نظام التشغيل بإنهاء العملية. في هذه الحالة، تحقق مما إذا كان ملف التكوين الخاص بك يحتوي على `offload_optimizer` أو `offload_param` أو كلاهما تم تكوينه لإلغاء التحميل إلى وحدة المعالجة المركزية (CPU). 

إذا كان لديك NVMe وZeRO-3، فقم بتجربة الإلغاء إلى NVMe ([تقدير](https://deepspeed.readthedocs.io/en/latest/memory.html) متطلبات الذاكرة لنموذجك).

### فقدان NaN

غالبًا ما يحدث فقدان NaN عندما يتم تدريب نموذج مسبقًا في bf16 ثم تحاول استخدامه مع fp16 (وهو أمر مهم بشكل خاص للنماذج المدربة على TPU). لحل هذا، استخدم fp32 أو bf16 إذا كان الأجهزة الخاصة بك تدعمها (TPU، Ampere GPUs أو أحدث).

قد تكون المشكلة الأخرى متعلقة باستخدام fp16. على سبيل المثال، إذا كان هذا هو تكوين fp16 الخاص بك:

```yaml
{
    "fp16": {
        "enabled": "auto"،
        "loss_scale": 0،
        "loss_scale_window": 1000،
        "initial_scale_power": 16،
        "hysteresis": 2،
        "min_loss_scale": 1
    }
}
```

قد تشاهد رسائل `OVERFLOW!` التالية في السجلات:

```bash
0%|                                                                                                                             | 0/189 [00:00<?, ?it/s]
 [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 262144, reducing to 262144
  1%|▌                                                                                                                    | 1/189 [00:00<01:26,  2.17it/s]
 [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 262144, reducing to 131072.0
  1%|█▏
 [...]
 [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 1, reducing to 1
 14%|████████████████▌                                                                                                   | 27/189 [00:14<01:13,  2.21it/s]
 [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 1, reducing to 1
 15%|█████████████████▏                                                                                                  | 28/189 [00:14<01:13,  2.18it/s]
 [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 1, reducing to 1
 15%|█████████████████▊                                                                                                  | 29/189 [00:15<01:13,  2.18it/s]
 [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 1, reducing to 1
[...]
```
هذا يعني أن مقياس الخسارة DeepSpeed غير قادر على العثور على معامل قياس للتغلب على فيض الخسارة. لإصلاحه، جرب قيمة أعلى لـ `initial_scale_power` (32 عادة ما تعمل).

## الموارد

DeepSpeed ZeRO هي تقنية قوية لتدريب وتحميل النماذج الكبيرة جدًا للتنفيذ باستخدام موارد GPU المحدودة، مما يجعلها أكثر سهولة في الوصول إلى الجميع. لمزيد من المعلومات حول DeepSpeed، يمكنك قراءة [المنشورات على المدونة](https://www.microsoft.com/en-us/research/search/?q=deepspeed)، و[الوثائق](https://www.deepspeed.ai/getting-started/)، و[مستودع GitHub](https://github.com/microsoft/deepspeed).

والأوراق التالية هي أيضًا مصدر رائع لمزيد من المعلومات حول ZeRO:

* [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://hf.co/papers/1910.02054)
* [ZeRO-Offload: Democratizing Billion-Scale Model Training](https://hf.co/papers/2101.06840)
* [ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning](https://hf.co/papers/2104.07857)