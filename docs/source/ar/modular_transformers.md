# المحولات النمطية

مكتبة `transformers` هي إطار عمل ذو فلسفة محدد؛ يتم تعريف فلسفتنا في [الدليل المفاهيمي](./philosophy).

جوهر هذه الفلسفة يتمثل في مبدأ [نموذج واحد، ملف واحد](https://huggingface.co/blog/transformers-design-philosophy)
في المكتبة. الجانب السلبي لهذا المكون هو تقييده لوراثة واستيراد مكونات الملفات.

نتيجة لذلك، تتكرر مكونات النموذج عبر العديد من الملفات. يحتوي `transformers` على عدد كبير من طبقات الانتباه، يقارب عدد النماذج، والكثير منها متطابق.  يتسبب هذا في تباعد عمليات التنفيذ المستقلة مع تطبيق الإصلاحات والتغييرات.
على أجزاء محددة من التعليمات البرمجية.

ولمعالجة ذلك، اعتمدنا مفهوم "النسخ" في المكتبة.  فبإضافة تعليق يُشير إلى أن التعليمات البرمجية هي نسخة من أخرى، نضمن من خلال أنظمة  CI والأوامر المحلية عدم تباعد النسخ.  لكن هذه العملية، رغم بساطتها، تُسبب إرهاقاً.  كما أنها تزيد العبء على المساهمين، وهو ما نهدف إلى تجاوزه.

غالباً ما تتطلب مساهمات النماذج إضافة تعليمات برمجية (حوالي 1000 سطر)، ومعالج (حوالي 500 سطر)، واختبارات، ووثائق، إلخ. ونادراً ما تقل مساهمات النماذج عن 3000-5000 سطر من التعليمات البرمجية،  معظمها أكواد نمطية.  هذا يرفع مستوى  المساهمات،

ونهدف مع المحولات النمطية إلى خفض هذا المستوى إلى حدّ مقبول.

## ما هو؟

تقدم المحولات النمطية مفهوم ملف "نمطي" لمجلد نموذج. يقبل هذا الملف النمطي تعليمات برمجية
غير مقبولة عادة في ملفات النمذجة/المعالجة، حيث يسمح بالاستيراد من نماذج مجاورة وكذلك
الوراثة من الفئات إلى فئات أخرى.

يعرّف هذا الملف النمطي النماذج والمعالجات وفئة التكوين التي سيتم تعريفها في وحداتهم
المتعلقة.

وأخيرًا، يقدم هذا الميزة أداة `linter` جديدة والتي ستعمل على "تفكيك" الملف النمطي إلى بنية "نموذج واحد، ملف واحد"
هيكل الدليل. سيتم إنشاء هذه الملفات تلقائيًا في كل مرة يتم فيها تشغيل البرنامج النصي؛ مما يقلل من المساهمات المطلوبة
إلى الملف النمطي، وبالتالي فقط إلى التغييرات بين النموذج المساهم والنماذج الأخرى.

سيقوم مستخدمو النموذج في النهاية باستيراد واستخدام واجهة الملف الواحد، لذا لا يتوقع حدوث أي تغيير هنا. من خلال القيام بذلك،
نأمل في الجمع بين أفضل ما في العالمين: تمكين المساهمات البسيطة مع الالتزام بفلسفتنا.

لذلك، هذا بديل لعلامات `# Copied from`، ويمكن توقع انتقال النماذج المساهمة سابقًا إلى
تنسيق المحولات النمطية الجديد في الأشهر المقبلة.

### التفاصيل

تُبسط أداة "linter" الوراثة، مُنشئةً جميع الملفات المفردة من الملف النمطي، مع الحفاظ على شفافيتها أمام مستخدمي Python. حاليًا، تُبسط الأداة مستوىً واحدًا من الوراثة

على سبيل المثال:
- إذا ورثت فئة التكوين من فئة أخرى وأضافت/حذفت معامل، فسيتم إما الإشارة إلى الملف المولد مباشرةً
  (في حالة الإضافة) أو إزالته تمامًا (في حالة الحذف).
- إذا ورثت فئة من فئة أخرى، على سبيل المثال: `class GemmaModel(LlamaModel):`، تُستنتج التبعيات تلقائيًا
  سيتم استنتاج جميع الوحدات الفرعية تلقائيًا من الفئة الأصلية.
- إذا قمت بتعريف وظائف جديدة في الملف `modular` واستخدمتها داخل الفئات، فستستنتج أداة linter ذلك تلقائيًا

يجب أن تكون قادرًا على كتابة كل شيء (المجزىء اللغوي، ومُعالِج الصور، والنموذج، والتكوين) في الملف `modular`، وسيتم إنشاء الملفات المُقابلة تلقائيًا.

### التطبيق

[TODO] نقدم اختبارًا جديدًا، للتأكد من أن المحتوى المولد يتطابق مع ما هو موجود في `modular_xxxx.py`

### الأمثلة

هنا مثال سريع باستخدام BERT و RoBERTa. النموذجان مرتبطان ارتباطًا وثيقًا: يختلف تنفيذهما النموذجي في طبقة تضمين.

بدلاً من إعادة تعريف النموذج بالكامل، إليك كيف يبدو ملف `modular_roberta.py` لفئات النمذجة والتكوين (لأغراض المثال، يتم تجاهل المجزىء اللغوي في هذا الوقت حيث أنه مختلف جدًا).

```python
from torch import nn
from ..bert.configuration_bert import BertConfig
from ..bert.modeling_bert import (
    BertModel,
    BertEmbeddings,
    BertForMaskedLM
)

# تكوين RoBERTa مطابق لتكوين BERT
class RobertaConfig(BertConfig):
  model_type = 'roberta'

# نعيد تعريف الإضافات هنا لتسليط الضوء على اختلاف معرف الحشو، ونعيد تعريف الإضافات الموضعية
class RobertaEmbeddings(BertEmbeddings):
    def __init__(self, config):
        super().__init__(config())

        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

# نموذج RoBERTa مطابق لنموذج BERT، باستثناء طبقة الإضافات.
# نعيد تعريف الإضافات أعلاه، لذا هنا لا توجد حاجة لعمل إضافي
class RobertaModel(BertModel):
  def __init__(self, config):
    super().__init__(config)
    self.embeddings = RobertaEmbeddings(config)

      
# الرؤوس الآن تحتاج فقط إلى إعادة تعريف النموذج داخل `RobertaModel` الصحيح
class RobertaForMaskedLM(BertForMaskedLM):
  def __init__(self, config):
    super().__init__(config)
    self.model = RobertaModel(config)
```

لاحظ أنه إذا لم تستخدم الاعتماد الذي حددته، فستحصل على الخطأ التالي:

```bash
ValueError: You defined `RobertaEmbeddings` in the modular_roberta.py, it should be used
                                    when you define `BertModel`, as it is one of it's direct dependencies. Make sure
                                    you use it in the `__init__` function.
```

بالإضافة إلى ذلك، قد تجد قائمة بالأمثلة هنا:

## ما هو ليس كذلك

ليس بديلاً لتعليمات برمجة النمذجة (بعد؟)، وإذا لم يكن نموذجك يعتمد على أي شيء آخر موجود من قبل، فيمكنك إضافة ملف `نمذجة` كالعادة.


## الاستخدام المتقدم

### إزالة السمات والوظائف
لإزالة السمات التي لا تستخدم في نموذجك النمطي، والتي لا تريد رؤيتها في النمذجة المفككة:

```python
class GemmaModel(LlamaModel):                 |           class GemmaModel(PreTrainedModel):
    def __init__(self, config):               |              def __init__(self, config):
        super().__init__(self, eos_token)     |                 super().__init__(config)
        del self.embed_tokens                 |                 self.padding_idx = config.pad_token_id
                                              |                 self.vocab_size = config.vocab_size
                                              |
                                              |                 self.layers = nn.ModuleList(
                                              |                     [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
                                              |                 )
                                              |                 self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
                                              |                 self.rotary_emb = LlamaRotaryEmbedding(config=config)
                                              |                 self.gradient_checkpointing = False
                                              |                 
                                              |                 # Initialize weights and apply final processing
                                              |                 self.post_init()
```
إذا قمت بالتحقق من `LlamaModel` الأصلي، فستجد `embed_tokens` الذي تمت إزالته هنا (كما هو متوقع!)

إزالة وظيفة مشابهة، تحتاج فقط إلى كتابتها مع `raise ValueError("")` لمحاكاة السلوك الذي تريده فعليًا عند إزالة وظيفة أصلية في بايثون.

```python
class GemmaTokenizer(LlamaTokenizer):
    ...

    def get_spm_processor(self):
        raise AttributeError("Not needed for Gemma")

    def unk_token_length(self):
        raise AttributeError("Not needed for Gemma")
```

### تعريف وظائف جديدة

إذا قمت بتعريف وظيفة جديدة في الملف `modular` لاستخدامها داخل فئة، على سبيل المثال

```python
def my_new_function(*args, **kwargs):
  # Do something here
  pass

class GemmaModel(LlamaModel):
    def forward(*args, **kwargs):
      # Call the function
      example = my_new_function(*args, **kwargs)
      # continue here
```

سيتم نسخ وظيفة `my_new_function` (وبشكل متكرر، أي وظائف أخرى جديدة يتم استدعاؤها في جسمها) تلقائيًا
في الملف الذي يتم استخدامه.

### استدعاء `super()`
قمنا مؤخرًا بشحن بعض الميزات التي تسمح لك بالانتقال من:
```python
class GemmaTokenizer(LlamaTokenizer, PretrainedTokenizerFast):         |           class GemmaModel(nn.Module):
    def __init__(self, eos_token="</s>"):                              |             def __init__(self):
        eos_token = AddedToken(eos_token)                              |                eos_token = AddedToken(eos_token)
        PretrainedTokenizerFast.__init__(self, eos_token)              |                super().__init__(eos_token)
```
هذا مفيد عندما لا تريد تفكيك استدعاء `super()`، وتريد التمييز بين أي استدعاء super init تقوم به!

### التسمية الخاصة
ندعم الآن أيضًا حالات خاصة مثل
```python
class GemmaVisionModel(CLIPModel):                                 
    pass
```
حيث اسم فئة `GemmaVision` الخاصة بك ليس هو نفسه `Gemma` النمطي. هذا مفيد للغاية للنماذج المركبة.
