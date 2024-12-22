<!---
Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/transformers-logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/transformers-logo-light.svg">
    <img alt="Hugging Face Transformers Library" src="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/transformers-logo-light.svg" width="352" height="59" style="max-width: 100%;">
  </picture>
  <br/>
  <br/>
</p>

<p align="center">
    <a href="https://circleci.com/gh/huggingface/transformers"><img alt="Build" src="https://img.shields.io/circleci/build/github/huggingface/transformers/main"></a>
    <a href="https://github.com/huggingface/transformers/blob/main/LICENSE"><img alt="GitHub" src="https://img.shields.io/github/license/huggingface/transformers.svg?color=blue"></a>
    <a href="https://huggingface.co/docs/transformers/index"><img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/docs/transformers/index.svg?down_color=red&down_message=offline&up_message=online"></a>
    <a href="https://github.com/huggingface/transformers/releases"><img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/transformers.svg"></a>
    <a href="https://github.com/huggingface/transformers/blob/main/CODE_OF_CONDUCT.md"><img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg"></a>
    <a href="https://zenodo.org/badge/latestdoi/155220641"><img src="https://zenodo.org/badge/155220641.svg" alt="DOI"></a>
</p>

<h4 align="center">
    <p>
        <a href="https://github.com/huggingface/transformers/blob/main/README.md">English</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_zh-hans.md">简体中文</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_zh-hant.md">繁體中文</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ko.md">한국어</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_es.md">Español</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ja.md">日本語</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_hd.md">हिन्दी</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ru.md">Русский</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_pt-br.md">Рortuguês</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_te.md">తెలుగు</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_fr.md">Français</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_de.md">Deutsch</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_vi.md">Tiếng Việt</a> |
		<b>العربية</b> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ur.md">اردو</a> |
    </p>
</h4>

<h3 align="center">
    <p>أحدث تقنيات التعلم الآلي لـ JAX وPyTorch وTensorFlow</p>
</h3>

<h3 align="center">
    <a href="https://hf.co/course"><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/course_banner.png"></a>
</h3>

يوفر 🤗 Transformers آلاف النماذج المُدربة مسبقًا لأداء المهام على طرائق مختلفة مثل النص والصورة والصوت.

يمكن تطبيق هذه النماذج على:

* 📝 النص، لمهام مثل تصنيف النص واستخراج المعلومات والرد على الأسئلة والتلخيص والترجمة وتوليد النص، في أكثر من 100 لغة.
* 🖼️ الصور، لمهام مثل تصنيف الصور وكشف الأشياء والتجزئة.
* 🗣️ الصوت، لمهام مثل التعرف على الكلام وتصنيف الصوت.

يمكن لنماذج المحول أيضًا أداء مهام على **طرائق متعددة مجتمعة**، مثل الرد على الأسئلة الجدولية والتعرف البصري على الحروف واستخراج المعلومات من المستندات الممسوحة ضوئيًا وتصنيف الفيديو والرد على الأسئلة المرئية.

يوفر 🤗 Transformers واجهات برمجة التطبيقات (APIs) لتحميل تلك النماذج المُدربة مسبقًا واستخدامها على نص معين، وضبطها بدقة على مجموعات البيانات الخاصة بك، ثم مشاركتها مع المجتمع على [مركز النماذج](https://huggingface.co/models) الخاص بنا. وفي الوقت نفسه، فإن كل وحدة نمطية Python التي تحدد بنية هي وحدة مستقلة تمامًا ويمكن تعديلها لتمكين تجارب البحث السريعة.

يتم دعم 🤗 Transformers بواسطة مكتبات التعلم العميق الثلاث الأكثر شيوعًا - [Jax](https://jax.readthedocs.io/en/latest/) و [PyTorch](https://pytorch.org/) و [TensorFlow](https://www.tensorflow.org/) - مع تكامل سلس بينها. من السهل تدريب نماذجك باستخدام واحدة قبل تحميلها للاستنتاج باستخدام الأخرى.

## العروض التوضيحية عبر الإنترنت

يمكنك اختبار معظم نماذجنا مباشرة على صفحاتها من [مركز النماذج](https://huggingface.co/models). كما نقدم [استضافة النماذج الخاصة وإصداراتها وواجهة برمجة تطبيقات الاستدلال](https://huggingface.co/pricing) للنماذج العامة والخاصة.

فيما يلي بعض الأمثلة:

في معالجة اللغات الطبيعية:
- [استكمال الكلمات المقنعة باستخدام BERT](https://huggingface.co/google-bert/bert-base-uncased?text=Paris+is+the+%5BMASK%5D+of+France)
- [التعرف على الكيانات المسماة باستخدام إليكترا](https://huggingface.co/dbmdz/electra-large-discriminator-finetuned-conll03-english?text=My+name+is+Sarah+and+I+live+in+London+city)
- [توليد النص باستخدام ميسترال](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
- [الاستدلال اللغوي الطبيعي باستخدام RoBERTa](https://huggingface.co/FacebookAI/roberta-large-mnli?text=The+dog+was+lost.+Nobody+lost+any+animal)
- [التلخيص باستخدام BART](https://huggingface.co/facebook/bart-large-cnn?text=The+tower+is+324+metres+%281%2C063+ft%29+tall%2C+about+the+same+height+as+an+81-storey+building%2C+and+the+tallest+structure+in+Paris.+Its+base+is+square%2C+measuring+125+metres+%28410+ft%29+on+each+side.+During+its+construction%2C+the+Eiffel+Tower+surpassed+the+Washington+Monument+to+become+the+tallest+man-made+structure+in+the+world%2C+a+title+it+held+for+41+years+until+the+Chrysler+Building+in+New+York+City+was+finished+in+1930.+It+was+the+first+structure+to+reach+a+height+of+300+metres.+Due+to+the+addition+of+a+broadcasting+aerial+at+the+top+of+the+tower+in+1957%2C+it+is+now+taller+than+the+Chrysler+Building+by+5.2+metres+%2817+ft%29.+Excluding+transmitters%2C+the+Eiffel+Tower+is+the+second+tallest+free-standing+structure+in+France+after+the+Millau+Viaduct)
- [الرد على الأسئلة باستخدام DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased-distilled-squad?text=Which+name+is+also+used+to+describe+the+Amazon+rainforest+in+English%3F&context=The+Amazon+rainforest+%28Portuguese%3A+Floresta+Amaz%C3%B4nica+or+Amaz%C3%B4nia%3B+Spanish%3A+Selva+Amaz%C3%B3nica%2C+Amazon%C3%ADa+or+usually+Amazonia%3B+French%3A+For%C3%AAt+amazonienne%3B+Dutch%3A+Amazoneregenwoud%29%2C+also+known+in+English+as+Amazonia+or+the+Amazon+Jungle%2C+is+a+moist+broadleaf+forest+that+covers+most+of+the+Amazon+basin+of+South+America.+This+basin+encompasses+7%2C000%2C000+square+kilometres+%282%2C700%2C000+sq+mi%29%2C+of+which+5%2C500%2C000+square+kilometres+%282%2C100%2C000+sq+mi%29+are+covered+by+the+rainforest.+This+region+includes+territory+belonging+to+nine+nations.+The+majority+of+the+forest+is+contained+within+Brazil%2C+with+60%25+of+the+rainforest%2C+followed+by+Peru+with+13%25%2C+Colombia+with+10%25%2C+and+with+minor+amounts+in+Venezuela%2C+Ecuador%2C+Bolivia%2C+Guyana%2C+Suriname+and+French+Guiana.+States+or+departments+in+four+nations+contain+%22Amazonas%22+in+their+names.+The+Amazon+represents+over+half+of+the+planet%27s+remaining+rainforests%2C+and+comprises+the+largest+and+most+biodiverse+tract+of+tropical+rainforest+in+the+world%2C+with+an+estimated+390+billion+individual+trees+divided+into+16%2C000+species)
- [الترجمة باستخدام T5](https://huggingface.co/google-t5/t5-base?text=My+name+is+Wolfgang+and+I+live+in+Berlin)

في رؤية الكمبيوتر:
- [تصنيف الصور باستخدام ViT](https://huggingface.co/google/vit-base-patch16-224)
- [كشف الأشياء باستخدام DETR](https://huggingface.co/facebook/detr-resnet-50)
- [التجزئة الدلالية باستخدام SegFormer](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512)
- [التجزئة الشاملة باستخدام Mask2Former](https://huggingface.co/facebook/mask2former-swin-large-coco-panoptic)
- [تقدير العمق باستخدام Depth Anything](https://huggingface.co/docs/transformers/main/model_doc/depth_anything)
- [تصنيف الفيديو باستخدام VideoMAE](https://huggingface.co/docs/transformers/model_doc/videomae)
- [التجزئة الشاملة باستخدام OneFormer](https://huggingface.co/shi-labs/oneformer_ade20k_dinat_large)

في الصوت:
- [الاعتراف التلقائي بالكلام مع Whisper](https://huggingface.co/openai/whisper-large-v3)
- [اكتشاف الكلمات الرئيسية باستخدام Wav2Vec2](https://huggingface.co/superb/wav2vec2-base-superb-ks)
- [تصنيف الصوت باستخدام محول طيف الصوت](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593)

في المهام متعددة الطرائق:
- [الرد على الأسئلة الجدولية باستخدام TAPAS](https://huggingface.co/google/tapas-base-finetuned-wtq)
- [الرد على الأسئلة المرئية باستخدام ViLT](https://huggingface.co/dandelin/vilt-b32-finetuned-vqa)
- [وصف الصورة باستخدام LLaVa](https://huggingface.co/llava-hf/llava-1.5-7b-hf)
- [تصنيف الصور بدون تدريب باستخدام SigLIP](https://huggingface.co/google/siglip-so400m-patch14-384)
- [الرد على أسئلة المستندات باستخدام LayoutLM](https://huggingface.co/impira/layoutlm-document-qa)
- [تصنيف الفيديو بدون تدريب باستخدام X-CLIP](https://huggingface.co/docs/transformers/model_doc/xclip)
- [كشف الأشياء بدون تدريب باستخدام OWLv2](https://huggingface.co/docs/transformers/en/model_doc/owlv2)
- [تجزئة الصور بدون تدريب باستخدام CLIPSeg](https://huggingface.co/docs/transformers/model_doc/clipseg)
- [توليد الأقنعة التلقائي باستخدام SAM](https://huggingface.co/docs/transformers/model_doc/sam)


## 100 مشروع يستخدم المحولات

🤗 Transformers هو أكثر من مجرد مجموعة أدوات لاستخدام النماذج المُدربة مسبقًا: إنه مجتمع من المشاريع المبنية حوله ومركز Hugging Face. نريد أن يمكّن 🤗 Transformers المطورين والباحثين والطلاب والأساتذة والمهندسين وأي شخص آخر من بناء مشاريعهم التي يحلمون بها.

للاحتفال بالـ 100,000 نجمة من النماذج المحولة، قررنا تسليط الضوء على المجتمع، وقد أنشأنا صفحة [awesome-transformers](./awesome-transformers.md) التي تُدرج 100 مشروعًا رائعًا تم بناؤها بالقرب من النماذج المحولة.

إذا كنت تمتلك أو تستخدم مشروعًا تعتقد أنه يجب أن يكون جزءًا من القائمة، فالرجاء فتح PR لإضافته!

## إذا كنت تبحث عن دعم مخصص من فريق Hugging Face

<a target="_blank" href="https://huggingface.co/support">
    <img alt="HuggingFace Expert Acceleration Program" src="https://cdn-media.huggingface.co/marketing/transformers/new-support-improved.png" style="max-width: 600px; border: 1px solid #eee; border-radius: 4px; box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);">
</a><br>

## جولة سريعة

لاستخدام نموذج على الفور على إدخال معين (نص أو صورة أو صوت، ...)، نوفر واجهة برمجة التطبيقات (API) الخاصة بـ `pipeline`. تجمع خطوط الأنابيب بين نموذج مُدرب مسبقًا ومعالجة ما قبل التدريب التي تم استخدامها أثناء تدريب هذا النموذج. فيما يلي كيفية استخدام خط أنابيب بسرعة لتصنيف النصوص الإيجابية مقابل السلبية:

```python
>>> from transformers import pipeline

# خصص خط أنابيب للتحليل الشعوري
>>> classifier = pipeline('sentiment-analysis')
>>> classifier('We are very happy to introduce pipeline to the transformers repository.')
[{'label': 'POSITIVE', 'score': 0.9996980428695679}]
```

يسمح السطر الثاني من التعليمات البرمجية بتحميل النموذج المُدرب مسبقًا الذي يستخدمه خط الأنابيب وتخزينه مؤقتًا، بينما يقوم السطر الثالث بتقييمه على النص المحدد. هنا، تكون الإجابة "إيجابية" بثقة تبلغ 99.97%.

تتوفر العديد من المهام على خط أنابيب مُدرب مسبقًا جاهز للاستخدام، في NLP ولكن أيضًا في رؤية الكمبيوتر والخطاب. على سبيل المثال، يمكننا بسهولة استخراج الأشياء المكتشفة في صورة:

``` python
>>> import requests
>>> from PIL import Image
>>> from transformers import pipeline

# قم بتنزيل صورة بها قطط لطيفة
>>> url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png"
>>> image_data = requests.get(url, stream=True).raw
>>> image = Image.open(image_data)

# خصص خط أنابيب لكشف الأشياء
>>> object_detector = pipeline('object-detection')
>>> object_detector(image)
[{'score': 0.9982201457023621،
  'label': 'remote'،
  'box': {'xmin': 40, 'ymin': 70, 'xmax': 175, 'ymax': 117}}،
 {'score': 0.9960021376609802،
  'label': 'remote'،
  'box': {'xmin': 333, 'ymin': 72, 'xmax': 368, 'ymax': 187}}،
 {'score': 0.9954745173454285،
  'label': 'couch'،
  'box': {'xmin': 0, 'ymin': 1, 'xmax': 639, 'ymax': 473}}،
 {'score': 0.9988006353378296،
  'label': 'cat'،
  'box': {'xmin': 13, 'ymin': 52, 'xmax': 314, 'ymax': 470}}،
 {'score': 0.9986783862113953،
  'label': 'cat'،
  'box': {'xmin': 345, 'ymin': 23, 'xmax': 640, 'ymax': 368}}]
```

هنا، نحصل على قائمة بالأشياء المكتشفة في الصورة، مع مربع يحيط بالشيء وتقييم الثقة. فيما يلي الصورة الأصلية على اليسار، مع عرض التوقعات على اليمين:

<h3 align="center">
    <a><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png" width="400"></a>
    <a><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample_post_processed.png" width="400"></a>
</h3>

يمكنك معرفة المزيد حول المهام التي تدعمها واجهة برمجة التطبيقات (API) الخاصة بـ `pipeline` في [هذا البرنامج التعليمي](https://huggingface.co/docs/transformers/task_summary).

بالإضافة إلى `pipeline`، لاستخدام أي من النماذج المُدربة مسبقًا على مهمتك، كل ما عليك هو ثلاثة أسطر من التعليمات البرمجية. فيما يلي إصدار PyTorch:
```python
>>> from transformers import AutoTokenizer، AutoModel

>>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
>>> model = AutoModel.from_pretrained("google-bert/bert-base-uncased")

>>> inputs = tokenizer("Hello world!"، return_tensors="pt")
>>> outputs = model(**inputs)
```

وهنا رمز مماثل لـ TensorFlow:
```python
>>> from transformers import AutoTokenizer، TFAutoModel

>>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
>>> model = TFAutoModel.from_pretrained("google-bert/bert-base-uncased")

>>> inputs = tokenizer("Hello world!"، return_tensors="tf")
>>> outputs = model(**inputs)
```

المُعلم مسؤول عن جميع المعالجة المسبقة التي يتوقعها النموذج المُدرب مسبقًا ويمكن استدعاؤه مباشرة على سلسلة واحدة (كما هو موضح في الأمثلة أعلاه) أو قائمة. سيقوم بإخراج قاموس يمكنك استخدامه في التعليمات البرمجية لأسفل أو تمريره مباشرة إلى نموذجك باستخدام عامل فك التعبئة **.

النموذج نفسه هو وحدة نمطية عادية [Pytorch `nn.Module`](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) أو [TensorFlow `tf.keras.Model`](https://www.tensorflow.org/api_docs/python/tf/keras/Model) (حسب backend) والتي يمكنك استخدامها كالمعتاد. [يوضح هذا البرنامج التعليمي](https://huggingface.co/docs/transformers/training) كيفية دمج مثل هذا النموذج في حلقة تدريب PyTorch أو TensorFlow التقليدية، أو كيفية استخدام واجهة برمجة تطبيقات `Trainer` لدينا لضبطها بدقة بسرعة على مجموعة بيانات جديدة.

## لماذا يجب أن أستخدم المحولات؟

1. نماذج سهلة الاستخدام وحديثة:
    - أداء عالي في فهم اللغة الطبيعية وتوليدها ورؤية الكمبيوتر والمهام الصوتية.
    - حاجز دخول منخفض للمربين والممارسين.
    - عدد قليل من التجريدات التي يواجهها المستخدم مع ثلاث فئات فقط للتعلم.
    - واجهة برمجة تطبيقات (API) موحدة لاستخدام جميع نماذجنا المُدربة مسبقًا.

1. تكاليف الكمبيوتر أقل، وبصمة كربونية أصغر:
    - يمكن للباحثين مشاركة النماذج المدربة بدلاً من إعادة التدريب دائمًا.
    - يمكن للممارسين تقليل وقت الكمبيوتر وتكاليف الإنتاج.
    - عشرات البنيات مع أكثر من 400,000 نموذج مُدرب مسبقًا عبر جميع الطرائق.

1. اختر الإطار المناسب لكل جزء من عمر النموذج:
    - تدريب النماذج الحديثة في 3 أسطر من التعليمات البرمجية.
    - قم بنقل نموذج واحد بين إطارات TF2.0/PyTorch/JAX حسب الرغبة.
    - اختر الإطار المناسب بسلاسة للتدريب والتقييم والإنتاج.

1. قم بسهولة بتخصيص نموذج أو مثال وفقًا لاحتياجاتك:
    - نوفر أمثلة لكل بنية لإعادة إنتاج النتائج التي نشرها مؤلفوها الأصليون.
    - يتم عرض داخليات النموذج بشكل متسق قدر الإمكان.
    - يمكن استخدام ملفات النموذج بشكل مستقل عن المكتبة للتجارب السريعة.

## لماذا لا يجب أن أستخدم المحولات؟

- ليست هذه المكتبة عبارة عن مجموعة أدوات من الصناديق المكونة للشبكات العصبية. لم يتم إعادة صياغة التعليمات البرمجية في ملفات النموذج باستخدام تجريدات إضافية عن قصد، بحيث يمكن للباحثين إجراء حلقات تكرار سريعة على كل من النماذج دون الغوص في تجريدات/ملفات إضافية.
- لا يُقصد بواجهة برمجة التطبيقات (API) للتدريب العمل على أي نموذج ولكنه مُستَهدف للعمل مع النماذج التي توفرها المكتبة. للحلقات العامة للتعلم الآلي، يجب استخدام مكتبة أخرى (ربما، [تسريع](https://huggingface.co/docs/accelerate)).
- في حين أننا نسعى جاهدين لتقديم أكبر عدد ممكن من حالات الاستخدام، فإن البرامج النصية الموجودة في مجلد [الأمثلة](https://github.com/huggingface/transformers/tree/main/examples) الخاص بنا هي مجرد أمثلة. من المتوقع ألا تعمل هذه البرامج النصية خارج الصندوق على مشكلتك المحددة وأنه سيُطلب منك تغيير بضع أسطر من التعليمات البرمجية لتكييفها مع احتياجاتك.

## التثبيت

### باستخدام pip

تم اختبار هذا المستودع على Python 3.9+، Flax 0.4.1+، PyTorch 2.0+، و TensorFlow 2.6+.

يجب تثبيت 🤗 Transformers في [بيئة افتراضية](https://docs.python.org/3/library/venv.html). إذا كنت غير معتاد على البيئات الافتراضية Python، فراجع [دليل المستخدم](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).

أولاً، قم بإنشاء بيئة افتراضية بالإصدار Python الذي تنوي استخدامه وقم بتنشيطه.

بعد ذلك، ستحتاج إلى تثبيت واحدة على الأقل من Flax أو PyTorch أو TensorFlow.
يرجى الرجوع إلى [صفحة تثبيت TensorFlow](https://www.tensorflow.org/install/)، و [صفحة تثبيت PyTorch](https://pytorch.org/get-started/locally/#start-locally) و/أو [صفحة تثبيت Flax](https://github.com/google/flax#quick-install) و [صفحة تثبيت Jax](https://github.com/google/jax#installation) بشأن أمر التثبيت المحدد لمنصتك.

عندما يتم تثبيت إحدى هذه المكتبات الخلفية، يمكن تثبيت 🤗 Transformers باستخدام pip كما يلي:

```bash
pip install transformers
```

إذا كنت ترغب في اللعب مع الأمثلة أو تحتاج إلى أحدث إصدار من التعليمات البرمجية ولا يمكنك الانتظار حتى يتم إصدار إصدار جديد، فيجب [تثبيت المكتبة من المصدر](https://huggingface.co/docs/transformers/installation#installing-from-source).

### باستخدام conda

يمكن تثبيت 🤗 Transformers باستخدام conda كما يلي:

```shell script
conda install conda-forge::transformers
```

> **_ملاحظة:_** تم إيقاف تثبيت `transformers` من قناة `huggingface`.

اتبع صفحات التثبيت الخاصة بـ Flax أو PyTorch أو TensorFlow لمعرفة كيفية تثبيتها باستخدام conda.

> **_ملاحظة:_**  على Windows، قد تتم مطالبتك بتنشيط وضع المطور للاستفادة من التخزين المؤقت. إذا لم يكن هذا خيارًا بالنسبة لك، فيرجى إعلامنا بذلك في [هذه المشكلة](https://github.com/huggingface/huggingface_hub/issues/1062).

## بنيات النماذج

**[جميع نقاط تفتيش النموذج](https://huggingface.co/models)** التي يوفرها 🤗 Transformers مدمجة بسلاسة من مركز [huggingface.co](https://huggingface.co/models) [model hub](https://huggingface.co/models)، حيث يتم تحميلها مباشرة من قبل [المستخدمين](https://huggingface.co/users) و [المنظمات](https://huggingface.co/organizations).

عدد نقاط التفتيش الحالية: ![](https://img.shields.io/endpoint?url=https://huggingface.co/api/shields/models&color=brightgreen)

يوفر 🤗 Transformers حاليًا البنيات التالية: راجع [هنا](https://huggingface.co/docs/transformers/model_summary) للحصول على ملخص لكل منها.

للتحقق مما إذا كان لكل نموذج تنفيذ في Flax أو PyTorch أو TensorFlow، أو كان لديه مُعلم مرفق مدعوم من مكتبة 🤗 Tokenizers، يرجى الرجوع إلى [هذا الجدول](https://huggingface.co/docs/transformers/index#supported-frameworks).

تم اختبار هذه التطبيقات على العديد من مجموعات البيانات (راجع البرامج النصية المثالية) ويجب أن تتطابق مع أداء التنفيذ الأصلي. يمكنك العثور على مزيد من التفاصيل حول الأداء في قسم الأمثلة من [الوثائق](https://github.com/huggingface/transformers/tree/main/examples).


## تعلم المزيد

| القسم | الوصف |
|-|-|
| [وثائق](https://huggingface.co/docs/transformers/) | وثائق واجهة برمجة التطبيقات (API) الكاملة والبرامج التعليمية |
| [ملخص المهام](https://huggingface.co/docs/transformers/task_summary) | المهام التي يدعمها 🤗 Transformers |
| [برنامج تعليمي لمعالجة مسبقة](https://huggingface.co/docs/transformers/preprocessing) | استخدام فئة `Tokenizer` لإعداد البيانات للنماذج |
| [التدريب والضبط الدقيق](https://huggingface.co/docs/transformers/training) | استخدام النماذج التي يوفرها 🤗 Transformers في حلقة تدريب PyTorch/TensorFlow وواجهة برمجة تطبيقات `Trainer` |
| [جولة سريعة: البرامج النصية للضبط الدقيق/الاستخدام](https://github.com/huggingface/transformers/tree/main/examples) | البرامج النصية المثالية للضبط الدقيق للنماذج على مجموعة واسعة من المهام |
| [مشاركة النماذج وتحميلها](https://huggingface.co/docs/transformers/model_sharing) | تحميل ومشاركة نماذجك المضبوطة بدقة مع المجتمع |

## الاستشهاد

لدينا الآن [ورقة](https://www.aclweb.org/anthology/2020.emnlp-demos.6/) يمكنك الاستشهاد بها لمكتبة 🤗 Transformers:
```bibtex
@inproceedings{wolf-etal-2020-transformers،
    title = "Transformers: State-of-the-Art Natural Language Processing"،
    author = "Thomas Wolf and Lysandre Debut and Victor Sanh and Julien Chaumond and Clement Delangue and Anthony Moi and Pierric Cistac and Tim Rault and R{\'e}mi Louf and Morgan Funtowicz and Joe Davison and Sam Shleifer and Patrick von Platen and Clara Ma and Yacine Jernite and Julien Plu and Canwen Xu and Teven Le Scao and Sylvain Gugger and Mariama Drame and Quentin Lhoest and Alexander M. Rush"،
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations"،
    month = oct،
    year = "2020"،
    address = "Online"،
    publisher = "Association for Computational Linguistics"،
    url = "https://www.aclweb.org/anthology/2020.emnlp-demos.6"،
    pages = "38--45"
}
```
