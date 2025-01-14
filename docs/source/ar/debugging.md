# استكشاف الأخطاء وإصلاحها

يمكن أن يكون التدريب على وحدات معالجة الرسومات (GPU) المتعددة مهمة صعبة، سواء كنت تواجه مشكلات في التثبيت أو مشكلات في الاتصال بين وحدات معالجة الرسومات الخاصة بك. يغطي دليل استكشاف الأخطاء وإصلاحها هذا بعض المشكلات التي قد تواجهها وكيفية حلها.

## تثبيت CUDA في DeepSpeed

إذا كنت تستخدم DeepSpeed، فمن المحتمل أنك قمت بتثبيته بالفعل باستخدام الأمر التالي.

```bash
pip install deepspeed
```

يقوم DeepSpeed بتجميع كود CUDA C++ ويمكن أن يكون مصدرًا محتملًا للأخطاء عند بناء ملحقات PyTorch التي تتطلب CUDA. تعتمد هذه الأخطاء على كيفية تثبيت CUDA على نظامك، ويركز هذا القسم على PyTorch المبني باستخدام *CUDA 10.2*.

<Tip>

بالنسبة لأي مشكلات أخرى في التثبيت، يرجى [فتح مشكلة](https://github.com/microsoft/DeepSpeed/issues) مع فريق DeepSpeed.

</Tip>

### حزم أدوات CUDA غير المتطابقة

يأتي PyTorch مع حزمة أدوات CUDA الخاصة به، ولكن لاستخدام DeepSpeed مع PyTorch، يجب أن يكون لديك إصدار متطابق من CUDA مثبت على مستوى النظام. على سبيل المثال، إذا قمت بتثبيت PyTorch مع `cudatoolkit==10.2` في بيئة Python الخاصة بك، فستحتاج أيضًا إلى تثبيت CUDA 10.2 على مستوى النظام. إذا لم يكن لديك CUDA مثبتًا على مستوى النظام، فيجب تثبيته أولاً.

قد يختلف الموقع الدقيق من نظام إلى آخر، ولكن `usr/local/cuda-10.2` هو الموقع الأكثر شيوعًا على العديد من أنظمة Unix. عندما يتم إعداد CUDA بشكل صحيح وإضافته إلى متغير البيئة `PATH` الخاص بك، يمكنك العثور على موقع التثبيت باستخدام الأمر التالي:

```bash
which nvcc
```

### حزم أدوات CUDA متعددة

قد يكون لديك أيضًا أكثر من حزمة أدوات CUDA مثبتة على مستوى النظام.

```bash
/usr/local/cuda-10.2
/usr/local/cuda-11.0
```

عادةً، يقوم مثبت الحزمة بتعيين المسارات إلى الإصدار الأخير الذي تم تثبيته. إذا فشلت حزمة البناء لأنها لا يمكنها العثور على الإصدار الصحيح من CUDA (على الرغم من تثبيته بالفعل على مستوى النظام)، فيجب عليك تكوين متغيرات البيئة `PATH` و`LD_LIBRARY_PATH` للإشارة إلى المسار الصحيح.

الق نظرة على محتويات هذه المتغيرات البيئية أولاً:

```bash
echo $PATH
echo $LD_LIBRARY_PATH
```

يسرد `PATH` مواقع الملفات القابلة للتنفيذ، و`LD_LIBRARY_PATH` يسرد أين تبحث عن المكتبات المشتركة. يتم منح الأولوية للمدخلات السابقة على اللاحقة، ويتم استخدام `:` للفصل بين الإدخالات المتعددة. لإخبار برنامج البناء بمكان العثور على حزمة أدوات CUDA المحددة التي تريدها، أدخل المسار الصحيح في القائمة أولاً. يقوم هذا الأمر بإلحاق المسار بالقيم الموجودة بدلاً من الكتابة فوقها.

```bash
# adjust the version and full path if needed
export PATH=/usr/local/cuda-10.2/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64:$LD_LIBRARY_PATH
```

بالإضافة إلى ذلك، يجب عليك أيضًا التحقق من وجود الدلائل التي تقوم بتعيينها بالفعل. يحتوي الدليل الفرعي `lib64` على كائنات CUDA `.so` المختلفة (مثل `libcudart.so`) وعلى الرغم من أنه من غير المحتمل أن يقوم نظامك بتسميتها بشكل مختلف، يجب عليك التحقق من الأسماء الفعلية وتغييرها وفقًا لذلك.

### إصدارات CUDA الأقدم

في بعض الأحيان، قد ترفض إصدارات CUDA الأقدم البناء مع برامج التجميع الأحدث. على سبيل المثال، إذا كان لديك `gcc-9` ولكن CUDA يريد `gcc-7`. عادةً، يؤدي تثبيت حزمة أدوات CUDA الأحدث إلى تمكين الدعم لمجمع أحدث.

يمكنك أيضًا تثبيت إصدار أقدم من المجمع بالإضافة إلى الإصدار الذي تستخدمه حاليًا (أو قد يكون مثبتًا بالفعل ولكنه غير مستخدم بشكل افتراضي ولا يمكن لنظام البناء رؤيته). لحل هذه المشكلة، يمكنك إنشاء رابط رمزي لمنح نظام البناء إمكانية رؤية المجمع الأقدم.

```bash
# تكييف المسار مع نظامك
sudo ln -s /usr/bin/gcc-7 /usr/local/cuda-10.2/bin/gcc
sudo ln -s /usr/bin/g++-7 /usr/local/cuda-10.2/bin/g++
```

### البناء المسبق

إذا كنت لا تزال تواجه مشكلات في تثبيت DeepSpeed أو إذا كنت تقوم ببناء DeepSpeed في وقت التشغيل، فيمكنك محاولة البناء المسبق لوحدات DeepSpeed قبل تثبيتها. لإجراء بناء محلي لـ DeepSpeed:

```bash
git clone https://github.com/microsoft/DeepSpeed/
cd DeepSpeed
rm -rf build
TORCH_CUDA_ARCH_LIST="8.6" DS_BUILD_CPU_ADAM=1 DS_BUILD_UTILS=1 pip install . \
--global-option="build_ext" --global-option="-j8" --no-cache -v \
--disable-pip-version-check 2>&1 | tee build.log
```

<Tip>

لاستخدام NVMe offload، أضف معلمة `DS_BUILD_AIO=1` إلى أمر البناء وتأكد من تثبيت حزمة libaio-dev على مستوى النظام.

</Tip>

بعد ذلك، سيتعين عليك تحديد بنية GPU الخاص بك عن طريق تحرير متغير `TORCH_CUDA_ARCH_LIST` (ابحث عن قائمة كاملة ببطاقات GPU من NVIDIA وبنيتها المقابلة على هذه [الصفحة](https://developer.nvidia.com/cuda-gpus)). للتحقق من إصدار PyTorch الذي يقابل بنيتك، قم بتشغيل الأمر التالي:

```bash
python -c "import torch; print(torch.cuda.get_arch_list())"
```

ابحث عن البنية لـ GPU باستخدام الأمر التالي:

<hfoptions id="arch">
<hfoption id="same GPUs">

```bash
CUDA_VISIBLE_DEVICES=0 python -c "import torch; print(torch.cuda.get_device_capability())"
```

</hfoption>
<hfoption id="specific GPU">

للعثور على البنية لـ GPU `0`:

```bash
CUDA_VISIBLE_DEVICES=0 python -c "import torch; \
print(torch.cuda.get_device_properties(torch.device('cuda')))
"_CudaDeviceProperties(name='GeForce RTX 3090'، major=8، minor=6، total_memory=24268MB، multi_processor_count=82)"
```

هذا يعني أن بنية GPU الخاص بك هي `8.6`.

</hfoption>
</hfoptions>

إذا حصلت على `8، 6`، فيمكنك تعيين `TORCH_CUDA_ARCH_LIST="8.6"`. بالنسبة لوحدات معالجة الرسومات المتعددة ذات البنى المختلفة، قم بإدراجها مثل `TORCH_CUDA_ARCH_LIST="6.1;8.6"`.

من الممكن أيضًا عدم تحديد `TORCH_CUDA_ARCH_LIST`، وسيقوم برنامج البناء تلقائيًا باستعلام بنية GPU للبناء. ومع ذلك، قد تتطابق أو لا تتطابق مع GPU الفعلي على الجهاز الهدف، ولهذا من الأفضل تحديد البنية الصحيحة بشكل صريح.

بالنسبة للتدريب على أجهزة متعددة ذات الإعداد نفسه، ستحتاج إلى إنشاء عجلة ثنائية:

```bash
git clone https://github.com/microsoft/DeepSpeed/
cd DeepSpeed
rm -rf build
TORCH_CUDA_ARCH_LIST="8.6" DS_BUILD_CPU_ADAM=1 DS_BUILD_UTILS=1 \
python setup.py build_ext -j8 bdist_wheel
```

ينشئ هذا الأمر عجلة ثنائية ستبدو شيئًا مثل `dist/deepspeed-0.3.13+8cd046f-cp38-cp38-linux_x86_64.whl`. الآن يمكنك تثبيت هذه العجلة محليًا أو على جهاز آخر.

```bash
pip install deepspeed-0.3.13+8cd046f-cp38-cp38-linux_x86_64.whl
```

## مشكلات شبكة Multi-GPU Debug

عند التدريب أو الاستدلال باستخدام `DistributedDataParallel` وGPU متعددة، إذا واجهت مشكلة في الاتصال بين العمليات و/أو العقد، فيمكنك استخدام البرنامج النصي التالي لتشخيص مشكلات الشبكة.

```bash
wget https://raw.githubusercontent.com/huggingface/transformers/main/scripts/distributed/torch-distributed-gpu-test.py
```

على سبيل المثال، لاختبار كيفية تفاعل وحدتي معالجة الرسومات (GPU) قم بما يلي:

```bash
python -m torch.distributed.run --nproc_per_node 2 --nnodes 1 torch-distributed-gpu-test.py
```
إذا تمكنت كلتا العمليتين من التحدث إلى بعضهما البعض وتخصيص ذاكرة GPU، فسيقوم كل منهما بطباعة حالة "موافق".

بالنسبة لوحدات معالجة الرسومات أو العقد الإضافية، قم بتعديل الحجج في البرنامج النصي.

ستجد الكثير من التفاصيل داخل برنامج التشخيص وحتى وصفة حول كيفية تشغيله في بيئة SLURM.
بالنسبة لوحدات معالجة الرسومات أو العقد الإضافية، قم بتعديل الحجج في البرنامج النصي.

ستجد الكثير من التفاصيل داخل برنامج التشخيص وحتى وصفة حول كيفية تشغيله في بيئة SLURM.

تتمثل إحدى مستويات التصحيح الإضافية في إضافة متغير البيئة `NCCL_DEBUG=INFO` كما يلي:

```bash
NCCL_DEBUG=INFO python -m torch.distributed.run --nproc_per_node 2 --nnodes 1 torch-distributed-gpu-test.py
```

سيؤدي هذا إلى تفريغ الكثير من معلومات التصحيح المتعلقة بـ NCCL، والتي يمكنك بعد ذلك البحث عنها عبر الإنترنت إذا وجدت أنه يتم الإبلاغ عن بعض المشكلات. أو إذا لم تكن متأكدًا من كيفية تفسير الإخراج، فيمكنك مشاركة ملف السجل في مشكلة.

## اكتشاف التدفق السفلي والفيضي

<Tip>

تتوفر هذه الميزة حاليًا لـ PyTorch فقط.

</Tip>

<Tip>

بالنسبة للتدريب متعدد وحدات معالجة الرسومات (GPU)، فإنه يتطلب DDP (`torch.distributed.launch`).

</Tip>

<Tip>

يمكن استخدام هذه الميزة مع أي نموذج يعتمد على `nn.Module`.

</Tip>

إذا بدأت في الحصول على `loss=NaN` أو إذا منع النموذج سلوكًا غير طبيعي آخر بسبب `inf` أو `nan` في التنشيطات أو الأوزان، فيجب اكتشاف المكان الذي يحدث فيه أول تدفق سفلي أو فيض وما الذي أدى إليه. لحسن الحظ، يمكنك القيام بذلك بسهولة عن طريق تنشيط وحدة نمطية خاصة ستقوم بالكشف التلقائي.

إذا كنت تستخدم [`Trainer`]]، فكل ما عليك فعله هو إضافة:

```bash
--debug underflow_overflow
```

إلى حجج سطر الأوامر العادية، أو تمرير `debug="underflow_overflow"` عند إنشاء كائن [`TrainingArguments`].

إذا كنت تستخدم حلقة التدريب الخاصة بك أو مدربًا آخر، فيمكنك تحقيق نفس الشيء باستخدام:

```python
from transformers.debug_utils import DebugUnderflowOverflow

debug_overflow = DebugUnderflowOverflow(model)
```

يقوم [`~debug_utils.DebugUnderflowOverflow`] بإدراج خطافات في النموذج الذي يقوم على الفور بعد كل مكالمة للأمام باختبار المتغيرات المدخلة والمخرجة وأوزان الوحدة النمطية المقابلة. بمجرد اكتشاف `inf` أو `nan` في عنصر واحد على الأقل من التنشيطات أو الأوزان، سيؤكد البرنامج ويطبع تقريرًا مثل هذا (تم اكتشافه باستخدام `google/mt5-small` في الدقة العائمة المختلطة fp16):

```
تم اكتشاف inf/nan أثناء batch_number=0
أطر 21 الأخيرة:
الحد الأدنى المطلق  الحد الأقصى المطلق  البيانات الوصفية
                  encoder.block.1.layer.1.DenseReluDense.dropout Dropout
0.00e+00 2.57e+02 input[0]
0.00e+00 2.85e+02 output
[...]
                  encoder.block.2.layer.0 T5LayerSelfAttention
6.78e-04 3.15e+03 input[0]
2.65e-04 3.42e+03 output[0]
             None output[1]
2.25e-01 1.00e+04 output[2]
                  encoder.block.2.layer.1.layer_norm T5LayerNorm
8.69e-02 4.18e-01 weight
2.65e-04 3.42e+03 input[0]
1.79e-06 4.65e+00 output
                  encoder.block.2.layer.1.DenseReluDense.wi_0 Linear
2.17e-07 4.50e+00 weight
1.79e-06 4.65e+00 input[0]
2.68e-06 3.70e+01 output
                  encoder.block.2.layer.1.DenseReluDense.wi_1 Linear
8.08e-07 2.66e+01 weight
1.79e-06 4.65e+00 input[0]
1.27e-04 2.37e+02 output
                  encoder.block.2.layer.1.DenseReluDense.wo Linear
1.01e-06 6.44e+00 weight
0.00e+00 9.74e+03 input[0]
3.18e-04 6.27e+04 output
                  encoder.block.2.layer.1.DenseReluDense T5DenseGatedGeluDense
1.79e-06 4.65e+00 input[0]
3.18e-04 6.27e+04 output
                  encoder.block.2.layer.1.dropout Dropout
3.18e-04 6.27e+04 input[0]
0.00e+00      inf output
```

تم تقصير إخراج المثال في المنتصف للإيجاز.
تم تقصير إخراج المثال في المنتصف للإيجاز.

يعرض العمود الثاني قيمة أكبر عنصر مطلق، لذا إذا نظرت عن كثب إلى الأطر القليلة الأخيرة، فستجد أن الإدخالات والمخرجات كانت في نطاق `1e4`. لذا عندما تم إجراء هذا التدريب في الدقة العائمة المختلطة fp16، تجاوزت الخطوة الأخيرة (نظرًا لأن أكبر رقم قبل `inf` في `fp16` هو `64e3`). لتجنب الفيضانات في `fp16`، يجب أن تظل التنشيطات أقل بكثير من `1e4`، لأن `1e4 * 1e4 = 1e8`، لذلك فإن أي ضرب للمصفوفة بأحجام تنشيط كبيرة سيؤدي إلى حالة فيض رقمي.

في بداية التتبع، يمكنك اكتشاف رقم الدفعة التي حدثت فيها المشكلة (هنا `تم اكتشاف inf/nan أثناء batch_number=0` يعني أن المشكلة حدثت في الدفعة الأولى).

يبدأ كل إطار تم الإبلاغ عنه عن طريق إعلان الإدخال الكامل المؤهل المقابل للوحدة النمطية التي يتم الإبلاغ عنها في هذا الإطار. إذا نظرنا فقط إلى هذا الإطار:

```
                  encoder.block.2.layer.1.layer_norm T5LayerNorm
8.69e-02 4.18e-01 weight
2.65e-04 3.42e+03 input[0]
1.79e-06 4.65e+00 output
```

هنا، يشير "encoder.block.2.layer.1.layer_norm" إلى أنه كان طبقة التطبيع للطبقة الأولى، من الكتلة الثانية للencoder. والمكالمات المحددة لـ "forward" هي "T5LayerNorm".

دعنا نلقي نظرة على الأطر القليلة الأخيرة من هذا التقرير:

```
Detected inf/nan during batch_number=0
Last 21 forward frames:
abs min  abs max  metadata
[...]
                  encoder.block.2.layer.1.DenseReluDense.wi_0 Linear

2.17e-07 4.50e+00 weight
1.79e-06 4.65e+00 input[0]
2.68e-06 3.70e+01 output
                  encoder.block.2.layer.1.DenseReluDense.wi_1 Linear

8.08e-07 2.66e+01 weight
1.79e-06 4.65e+00 input[0]
1.27e-04 2.37e+02 output
                  encoder.block.2.layer.1.DenseReluDense.wo Linear

1.01e-06 6.44e+00 weight
0.00e+00 9.74e+03 input[0]
3.18e-04 6.27e+04 output
                  encoder.block.2.layer.1.DenseReluDense T5DenseGatedGeluDense

1.79e-06 4.65e+00 input[0]
3.18e-04 6.27e+04 output
                  encoder.block.2.layer.1.dropout Dropout

3.18e-04 6.27e+04 input[0]
0.00e+00      inf output
```

يبلغ الإطار الأخير عن وظيفة "Dropout.forward" مع الإدخال الأول للإدخال الوحيد والإخراج الثاني
الإخراج الوحيد. يمكنك أن ترى أنه تم استدعاؤه من سمة "dropout" داخل فئة "DenseReluDense". يمكننا أن نرى
حدث ذلك أثناء الطبقة الأولى، من الكتلة الثانية، أثناء الدفعة الأولى. أخيرًا، كان الحد الأقصى المطلق
كانت عناصر الإدخال "6.27e+04" وكان نفس الإخراج "inf".

يمكنك أن ترى هنا أن "T5DenseGatedGeluDense.forward" أسفر عن تنشيطات الإخراج، والتي كان الحد الأقصى المطلق لها
حوالي 62.7K، وهو قريب جدًا من الحد الأعلى لـ fp16 وهو 64K. في الإطار التالي لدينا "Dropout" الذي يعيد التطبيع
الأوزان، بعد أن قام بإلغاء تنشيط بعض العناصر، مما يدفع القيمة القصوى المطلقة إلى أكثر من 64K، ونحصل على
فيض (inf).

كما ترى، فإن الأطر السابقة هي التي نحتاج إلى النظر فيها عندما تبدأ الأرقام في الدخول إلى أرقام كبيرة جدًا لـ fp16
الأرقام.

دعنا نقارن التقرير بالرمز من "models/t5/modeling_t5.py":

```python
class T5DenseGatedGeluDense(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wi_0 = nn.Linear(config.d_model, config.d_ff، bias=False)
        self.wi_1 = nn.Linear(config.d_model, config.d_ff، bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model، bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.gelu_act = ACT2FN["gelu_new"]

    def forward(self, hidden_states):
        hidden_gelu = self.gelu_act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states
```

لم يعد من الصعب الآن رؤية مكالمة `dropout` ، وجميع المكالمات السابقة أيضًا.

نظرًا لأن الكشف يحدث في خطاف للأمام، يتم طباعة هذه التقارير فورًا بعد كل عودة للأمام.

بالعودة إلى التقرير الكامل، للتصرف بناءً عليه وإصلاح المشكلة، يلزم الرجوع إلى بعض الأطر حيث بدأت الأرقام في الارتفاع والتحويل على الأرجح إلى وضع "fp32" هنا، بحيث لا تفيض الأرقام عند الضرب أو جمعها. بالطبع، قد تكون هناك حلول أخرى. على سبيل المثال، يمكننا إيقاف تشغيل "amp" مؤقتًا إذا تم تشغيله، بعد نقل "forward" الأصلي إلى برنامج مساعد wrapper، مثل:

```python
def _forward(self, hidden_states):
    hidden_gelu = self.gelu_act(self.wi_0(hidden_states))
    hidden_linear = self.wi_1(hidden_states)
    hidden_states = hidden_gelu * hidden_linear
    hidden_states = self.dropout(hidden_states)
    hidden_states = self.wo(hidden_states)
    return hidden_states


import torch


def forward(self, hidden_states):
    if torch.is_autocast_enabled():
        with torch.cuda.amp.autocast(enabled=False):
            return self._forward(hidden_states)
    else:
        return self._forward(hidden_states)
```

نظرًا لأن الكاشف التلقائي يبلغ فقط عن المدخلات والمخرجات للأطر الكاملة، بمجرد أن تعرف أين تبحث، فقد ترغب في تحليل المراحل الوسيطة لأي دالة "forward" محددة أيضًا. في هذه الحالة، يمكنك استخدام وظيفة المساعدة `detect_overflow` لإدخال الكاشف حيث تريده، على سبيل المثال:

```python
from debug_utils import detect_overflow


class T5LayerFF(nn.Module):
    [...]

    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        detect_overflow(forwarded_states, "after layer_norm")
        forwarded_states = self.DenseReluDense(forwarded_states)
        detect_overflow(forwarded_states, "after DenseReluDense")
        return hidden_states + self.dropout(forwarded_states)
```

يمكنك أن ترى أننا أضفنا 2 من هذه الآن ونحن نتتبع إذا تم اكتشاف "inf" أو "nan" لـ "forwarded_states" في مكان ما بينهما.

في الواقع، يقوم الكاشف بالفعل بالإبلاغ عن هذه المعلومات لأن كل مكالمة في المثال أعلاه هي `nn.Module`، ولكن دعنا نقول إذا كان لديك بعض الحسابات المباشرة المحلية، فهذا هو ما ستفعله.

بالإضافة إلى ذلك، إذا كنت تقوم بتنفيذ المصحح في كودك الخاص، فيمكنك ضبط عدد الأطر المطبوعة من الإعداد الافتراضي، على سبيل المثال:

```python
from transformers.debug_utils import DebugUnderflowOverflow

debug_overflow = DebugUnderflowOverflow(model, max_frames_to_save=100)
```

### تتبع قيمة الحد الأدنى والقصوى المطلقة الدفعة المحددة

يمكن استخدام نفس فئة التصحيح لتتبع الدفعات مع ميزة الكشف عن التدفق السفلي/الفيضي.

لنفترض أنك تريد مراقبة القيمة المطلقة الدنيا والقصوى لجميع مكونات كل مكالمة "forward" لدفعة معينة، والقيام بذلك فقط للدفعتين 1 و 3. ثم قم بتنفيذ هذه الفئة كما يلي:

```python
debug_overflow = DebugUnderflowOverflow(model, trace_batch_nums=[1, 3])
```

والآن سيتم تتبع الدفعات الكاملة 1 و 3 باستخدام نفس التنسيق الذي يستخدمه كاشف التدفق السفلي/الفيضي.

الدفعات مفهرسة من 0.

هذا مفيد إذا كنت تعلم أن البرنامج يبدأ التصرف بشكل غير طبيعي بعد رقم دفعة معين، بحيث يمكنك الانتقال مباشرة إلى تلك المنطقة. فيما يلي إخراج عينة مقطوعة لمثل هذا التكوين:

```
                  *** بدء رقم الدفعة = 1 ***
القيمة الصغرى المطلقة  القيمة القصوى المطلقة  البيانات الوصفية
                  shared Embedding
1.01e-06 7.92e+02 weight
0.00e+00 2.47e+04 input[0]
5.36e-05 7.92e+02 output
[...]
                  decoder.dropout Dropout
1.60e-07 2.27e+01 input[0]
0.00e+00 2.52e+01 output
                  مكدس فك التشفير T5
     ليس إخراج tensor
                  lm_head Linear
1.01e-06 7.92e+02 weight
0.00e+00 1.11e+00 input[0]
6.06e-02 8.39e+01 output
                   T5ForConditionalGeneration
     ليس إخراج tensor
*** بدء رقم الدفعة = 3 ***
القيمة الصغرى المطلقة  القيمة القصوى المطلقة  البيانات الوصفية
                  shared Embedding
1.01e-06 7.92e+02 weight
0.00e+00 2.78e+04 input[0]
5.36e-05 7.92e+02 output
[...]
```

هنا ستحصل على عدد كبير من الأطر التي تم تفريغها - بقدر ما كانت هناك مكالمات للأمام في نموذجك، لذلك قد يكون أو لا يكون ما تريده، ولكنه في بعض الأحيان قد يكون أسهل في استخدامه لأغراض التصحيح. على سبيل المثال، إذا بدأت المشكلة في الحدوث عند رقم الدفعة 150. لذا يمكنك تفريغ آثار الدفعات 149 و 150 ومقارنة المكان الذي بدأت فيه الأرقام في الاختلاف.

يمكنك أيضًا تحديد رقم الدفعة الذي يجب إيقاف التدريب بعده، باستخدام ما يلي:

```python
debug_overflow = DebugUnderflowOverflow(model, trace_batch_nums=[1, 3], abort_after_batch_num=3)
```