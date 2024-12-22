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
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ar.md">العربية</a> |
        <b>اردو</b> |
    </p>
</h4>

<h3 align="center">
    <p>جدید ترین مشین لرننگ برائے JAX، PyTorch اور TensorFlow</p>
</h3>

<h3 align="center">
    <a href="https://hf.co/course"><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/course_banner.png"></a>
</h3>

&#8207;🤗 Transformers مختلف طریقوں جیسے کہ متن، بصارت، اور آڈیو پر کام کرنے کے لیے ہزاروں پری ٹرینڈ ماڈلز فراہم کرتے ہیں۔

یہ ماڈلز درج ذیل پر لاگو کیے جا سکتے ہیں:

* 📝 متن، جیسے کہ متن کی درجہ بندی، معلومات کا استخراج، سوالات کے جوابات، خلاصہ، ترجمہ، اور متن کی تخلیق، 100 سے زائد زبانوں میں۔
* 🖼️ تصاویر، جیسے کہ تصویر کی درجہ بندی، اشیاء کی شناخت، اور تقسیم۔
* 🗣️ آڈیو، جیسے کہ تقریر کی شناخت اور آڈیو کی درجہ بندی۔

ٹرانسفارمر ماڈلز **مختلف طریقوں کو ملا کر** بھی کام انجام دے سکتے ہیں، جیسے کہ ٹیبل سوال جواب، بصری حروف کی شناخت، اسکین شدہ دستاویزات سے معلومات نکالنا، ویڈیو کی درجہ بندی، اور بصری سوال جواب۔

&#8207;🤗 Transformers ایسے APIs فراہم کرتا ہے جو آپ کو تیز رفتاری سے پری ٹرینڈ ماڈلز کو ایک دیے گئے متن پر ڈاؤن لوڈ اور استعمال کرنے، انہیں اپنے ڈیٹا سیٹس پر فائن ٹون کرنے، اور پھر ہمارے [ماڈل حب](https://huggingface.co/models) پر کمیونٹی کے ساتھ شیئر کرنے کی سہولت دیتا ہے۔ اسی وقت، ہر پائتھن ماڈیول جو ایک آرکیٹیکچر کو بیان کرتا ہے، مکمل طور پر خود مختار ہوتا ہے اور اسے تیز تحقیقاتی تجربات کے لیے تبدیل کیا جا سکتا ہے۔


&#8207;🤗 Transformers تین سب سے مشہور ڈیپ لرننگ لائبریریوں — [Jax](https://jax.readthedocs.io/en/latest/)، [PyTorch](https://pytorch.org/) اور [TensorFlow](https://www.tensorflow.org/) — کی مدد سے تیار کردہ ہے، جن کے درمیان بے حد ہموار انضمام ہے۔ اپنے ماڈلز کو ایک کے ساتھ تربیت دینا اور پھر دوسرے کے ساتھ inference کے لیے لوڈ کرنا انتہائی سادہ ہے۔

## آن لائن ڈیمو

آپ ہمارے زیادہ تر ماڈلز کو براہ راست ان کے صفحات پر [ماڈل ہب](https://huggingface.co/models) سے آزما سکتے ہیں۔ ہم عوامی اور نجی ماڈلز کے لیے [ذاتی ماڈل ہوسٹنگ، ورژننگ، اور انفرنس API](https://huggingface.co/pricing) بھی فراہم کرتے ہیں۔

یہاں چند مثالیں ہیں:

قدرتی زبان کی پروسیسنگ میں:

- [&#8207;BERT کے ساتھ ماسک شدہ الفاظ کی تکمیل](https://huggingface.co/google-bert/bert-base-uncased?text=Paris+is+the+%5BMASK%5D+of+France)
- [&#8207;Electra کے ساتھ نامزد اداروں کی شناخت](https://huggingface.co/dbmdz/electra-large-discriminator-finetuned-conll03-english?text=My+name+is+Sarah+and+I+live+in+London+city)
- [&#8207;Mistral کے ساتھ متنی جنریشن](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
- [&#8207;RoBERTa کے ساتھ قدرتی زبان کی دلیل](https://huggingface.co/FacebookAI/roberta-large-mnli?text=The+dog+was+lost.+Nobody+lost+any+animal)
- [&#8207;BART کے ساتھ خلاصہ کاری](https://huggingface.co/facebook/bart-large-cnn?text=The+tower+is+324+metres+%281%2C063+ft%29+tall%2C+about+the+same+height+as+an+81-storey+building%2C+and+the+tallest+structure+in+Paris.+Its+base+is+square%2C+measuring+125+metres+%28410+ft%29+on+each+side.+During+its+construction%2C+the+Eiffel+Tower+surpassed+the+Washington+Monument+to+become+the+tallest+man-made+structure+in+the+world%2C+a+title+it+held+for+41+years+until+the+Chrysler+Building+in+New+York+City+was+finished+in+1930.+It+was+the+first+structure+to+reach+a+height+of+300+metres.+Due+to+the+addition+of+a+broadcasting+aerial+at+the+top+of+the+tower+in+1957%2C+it+is+now+taller+than+the+Chrysler+Building+by+5.2+metres+%2817+ft%29.+Excluding+transmitters%2C+the+Eiffel+Tower+is+the+second+tallest+free-standing+structure+in+France+after+the+Millau+Viaduct)
- [&#8207;DistilBERT کے ساتھ سوالات کے جوابات](https://huggingface.co/distilbert/distilbert-base-uncased-distilled-squad?text=Which+name+is+also+used+to+describe+the+Amazon+rainforest+in+English%3F&context=The+Amazon+rainforest+%28Portuguese%3A+Floresta+Amaz%C3%B4nica+or+Amaz%C3%B4nia%3B+Spanish%3A+Selva+Amaz%C3%B3nica%2C+Amazon%C3%ADa+or+usually+Amazonia%3B+French%3A+For%C3%AAt+amazonienne%3B+Dutch%3A+Amazoneregenwoud%29%2C+also+known+in+English+as+Amazonia+or+the+Amazon+Jungle%2C+is+a+moist+broadleaf+forest+that+covers+most+of+the+Amazon+basin+of+South+America.+This+basin+encompasses+7%2C000%2C000+square+kilometres+%282%2C700%2C000+sq+mi%29%2C+of+which+5%2C500%2C000+square+kilometres+%282%2C100%2C000+sq+mi%29+are+covered+by+the+rainforest.+This+region+includes+territory+belonging+to+nine+nations.+The+majority+of+the+forest+is+contained+within+Brazil%2C+with+60%25+of+the+rainforest%2C+followed+by+Peru+with+13%25%2C+Colombia+with+10%25%2C+and+with+minor+amounts+in+Venezuela%2C+Ecuador%2C+Bolivia%2C+Guyana%2C+Suriname+and+French+Guiana.+States+or+departments+in+four+nations+contain+%22Amazonas%22+in+their+names.+The+Amazon+represents+over+half+of+the+planet%27s+remaining+rainforests%2C+and+comprises+the+largest+and+most+biodiverse+tract+of+tropical+rainforest+in+the+world%2C+with+an+estimated+390+billion+individual+trees+divided+into+16%2C000+species)
- [&#8207;T5 کے ساتھ ترجمہ](https://huggingface.co/google-t5/t5-base?text=My+name+is+Wolfgang+and+I+live+in+Berlin)

کمپیوٹر وژن میں:
- [&#8207;ViT کے ساتھ امیج کی درجہ بندی](https://huggingface.co/google/vit-base-patch16-224)
- [&#8207;DETR کے ساتھ اشیاء کی شناخت](https://huggingface.co/facebook/detr-resnet-50)
- [&#8207;SegFormer کے ساتھ سیمانٹک سیگمینٹیشن](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512)
- [&#8207;Mask2Former کے ساتھ پینوسٹک سیگمینٹیشن](https://huggingface.co/facebook/mask2former-swin-large-coco-panoptic)
- [&#8207;Depth Anything کے ساتھ گہرائی کا اندازہ](https://huggingface.co/docs/transformers/main/model_doc/depth_anything)
- [&#8207;VideoMAE کے ساتھ ویڈیو کی درجہ بندی](https://huggingface.co/docs/transformers/model_doc/videomae)
- [&#8207;OneFormer کے ساتھ یونیورسل سیگمینٹیشن](https://huggingface.co/shi-labs/oneformer_ade20k_dinat_large)


آڈیو:
- [خودکار تقریر کی پہچان Whisper کے ساتھ](https://huggingface.co/openai/whisper-large-v3)
- [کلیدی الفاظ کی تلاش Wav2Vec2 کے ساتھ](https://huggingface.co/superb/wav2vec2-base-superb-ks)
- [آڈیو کی درجہ بندی Audio Spectrogram Transformer کے ساتھ](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593)

ملٹی ماڈل ٹاسک میں:

- [ٹیبل سوال جواب کے لیے TAPAS](https://huggingface.co/google/tapas-base-finetuned-wtq)
- [ویژول سوال جواب کے لیے ViLT](https://huggingface.co/dandelin/vilt-b32-finetuned-vqa)
- [امیج کیپشننگ کے لیے LLaVa](https://huggingface.co/llava-hf/llava-1.5-7b-hf)
- [زیرو شاٹ امیج کلاسیفیکیشن کے لیے SigLIP](https://huggingface.co/google/siglip-so400m-patch14-384)
- [دستاویزی سوال جواب کے لیے LayoutLM](https://huggingface.co/impira/layoutlm-document-qa)
- [زیرو شاٹ ویڈیو کلاسیفیکیشن کے لیے X-CLIP](https://huggingface.co/docs/transformers/model_doc/xclip)
- [زیرو شاٹ آبجیکٹ ڈیٹیکشن کے لیے OWLv2](https://huggingface.co/docs/transformers/en/model_doc/owlv2)
- [زیرو شاٹ امیج سیگمنٹیشن کے لیے CLIPSeg](https://huggingface.co/docs/transformers/model_doc/clipseg)
- [خودکار ماسک جنریشن کے لیے SAM](https://huggingface.co/docs/transformers/model_doc/sam)


## ٹرانسفارمرز کے 100 منصوبے

&#8207;🤗 Transformers صرف پیشگی تربیت یافتہ ماڈلز کا ایک ٹول کٹ نہیں ہے: یہ ایک کمیونٹی ہے جو اس کے ارد گرد اور ہیگنگ فیس حب پر تعمیر شدہ منصوبوں کا مجموعہ ہے۔ ہم    چاہتے ہیں کہ🤗 Transformers ترقی کاروں، محققین، طلباء، پروفیسرز، انجینئرز، اور ہر کسی کو اپنے خوابوں کے منصوبے بنانے میں مدد فراہم کرے۔


&#8207;🤗 Transformers کے 100,000 ستاروں کی خوشی منانے کے لیے، ہم نے کمیونٹی پر روشنی ڈالنے کا فیصلہ کیا ہے، اور ہم نے [awesome-transformers](./awesome-transformers.md) کا صفحہ بنایا ہے جو 100 شاندار منصوبے درج کرتا ہے جو 🤗 Transformers کے ارد گرد بنائے گئے ہیں۔

اگر آپ کے پاس کوئی ایسا منصوبہ ہے جسے آپ سمجھتے ہیں کہ اس فہرست کا حصہ ہونا چاہیے، تو براہ کرم ایک PR کھولیں تاکہ اسے شامل کیا جا سکے!

## اگر آپ ہیگنگ فیس ٹیم سے حسب ضرورت معاونت تلاش کر رہے ہیں

<a target="_blank" href="https://huggingface.co/support">
    <img alt="HuggingFace Expert Acceleration Program" src="https://cdn-media.huggingface.co/marketing/transformers/new-support-improved.png" style="max-width: 600px; border: 1px solid #eee; border-radius: 4px; box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);">
</a><br>

## فوری ٹور

دیے گئے ان پٹ (متن، تصویر، آڈیو، ...) پر ماڈل کو فوری طور پر استعمال کرنے کے لیے، ہم pipeline API فراہم کرتے ہیں۔ پائپ لائنز ایک پیشگی تربیت یافتہ ماڈل کو اس ماڈل کی تربیت کے دوران استعمال ہونے والے پری پروسیسنگ کے ساتھ گروپ کرتی ہیں۔ یہاں یہ ہے کہ مثبت اور منفی متون کی درجہ بندی کے لیے پائپ لائن کو جلدی سے کیسے استعمال کیا جائے:


```python
>>> from transformers import pipeline

# جذبات کے تجزیے کے لیے ایک پائپ لائن مختص کریں
>>> classifier = pipeline('sentiment-analysis')
>>> classifier('We are very happy to introduce pipeline to the transformers repository.')
[{'label': 'POSITIVE', 'score': 0.9996980428695679}]
```

دوسری لائن کوڈ پائپ لائن کے ذریعہ استعمال ہونے والے پیشگی تربیت یافتہ ماڈل کو ڈاؤن لوڈ اور کیش کرتی ہے، جبکہ تیسری لائن اسے دیے گئے متن پر جانچتی ہے۔ یہاں، جواب "مثبت" ہے جس کی اعتماد کی شرح 99.97% ہے۔

بہت سے کاموں کے لیے ایک پیشگی تربیت یافتہ pipeline تیار ہے، NLP کے علاوہ کمپیوٹر ویژن اور آواز میں بھی۔ مثال کے طور پر، ہم تصویر میں دریافت شدہ اشیاء کو آسانی سے نکال سکتے ہیں:


``` python
>>> import requests
>>> from PIL import Image
>>> from transformers import pipeline

# جذبات کے تجزیے کے لیے ایک پائپ لائن مختص کریں
>>> url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png"
>>> image_data = requests.get(url, stream=True).raw
>>> image = Image.open(image_data)

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

یہاں، ہم کو تصویر میں دریافت شدہ اشیاء کی فہرست ملتی ہے، ہر ایک کے گرد ایک باکس اور اعتماد کا اسکور۔ یہاں اصل تصویر بائیں طرف ہے، اور پیشگوئیاں دائیں طرف ظاہر کی گئی ہیں:


<h3 align="center">
    <a><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png" width="400"></a>
    <a><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample_post_processed.png" width="400"></a>
</h3>

آپ `pipeline` API کی مدد سے معاونت شدہ کاموں کے بارے میں مزید جان سکتے ہیں [اس ٹیوٹوریل](https://huggingface.co/docs/transformers/task_summary) میں۔


&#8207;`pipeline` کے علاوہ، کسی بھی پیشگی تربیت یافتہ ماڈل کو آپ کے دیے گئے کام پر ڈاؤن لوڈ اور استعمال کرنے کے لیے، صرف تین لائنوں کا کوڈ کافی ہے۔ یہاں PyTorch ورژن ہے:

```python
>>> from transformers import AutoTokenizer، AutoModel

>>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
>>> model = AutoModel.from_pretrained("google-bert/bert-base-uncased")

>>> inputs = tokenizer("Hello world!"، return_tensors="pt")
>>> outputs = model(**inputs)
```

اور یہاں TensorFlow کے لیے مساوی کوڈ ہے:
```python
>>> from transformers import AutoTokenizer، TFAutoModel

>>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
>>> model = TFAutoModel.from_pretrained("google-bert/bert-base-uncased")

>>> inputs = tokenizer("Hello world!"، return_tensors="tf")
>>> outputs = model(**inputs)
```

ٹوکینائزر تمام پری پروسیسنگ کا ذمہ دار ہے جس کی پیشگی تربیت یافتہ ماڈل کو ضرورت ہوتی ہے اور اسے براہ راست ایک واحد سٹرنگ (جیسا کہ اوپر کی مثالوں میں) یا ایک فہرست پر کال کیا جا سکتا ہے۔ یہ ایک لغت فراہم کرے گا جسے آپ ڈاؤن اسٹریم کوڈ میں استعمال کر سکتے ہیں یا سادہ طور پر اپنے ماڈل کو ** دلیل انپیکنگ آپریٹر کے ذریعے براہ راست پاس کر سکتے ہیں۔

ماڈل خود ایک باقاعدہ [PyTorch `nn.Module`](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) یا [TensorFlow `tf.keras.Model`](https://www.tensorflow.org/api_docs/python/tf/keras/Model) (آپ کے بیک اینڈ پر منحصر ہے) ہے جسے آپ معمول کے مطابق استعمال کر سکتے ہیں۔ [یہ ٹیوٹوریل](https://huggingface.co/docs/transformers/training) وضاحت کرتا ہے کہ کلاسیکی PyTorch یا TensorFlow تربیتی لوپ میں ایسے ماڈل کو کیسے ضم کیا جائے، یا ہمارے `Trainer` API کا استعمال کرتے ہوئے نئے ڈیٹا سیٹ پر جلدی سے فائن ٹیون کیسے کیا جائے۔

## مجھے Transformers کیوں استعمال کرنا چاہیے؟

&#8207; 1. استعمال میں آسان جدید ترین ماڈلز:

 - قدرتی زبان کی سمجھ اور تخلیق، کمپیوٹر وژن، اور آڈیو کے کاموں میں اعلی کارکردگی۔
 - معلمین اور عملی ماہرین کے لیے کم داخلی رکاوٹ۔
 - سیکھنے کے لیے صرف تین کلاسز کے ساتھ چند یوزر فرینڈلی ایبسٹریکشنز۔
 - ہمارے تمام pretrained ماڈلز کے استعمال کے لیے ایک متحد API۔

&#8207; 2. کمپیوٹیشن کے اخراجات میں کمی، کاربن فٹ پرنٹ میں کمی:

- محققین ہمیشہ دوبارہ تربیت کرنے کی بجائے تربیت شدہ ماڈلز شیئر کر سکتے ہیں۔
- عملی ماہرین کمپیوٹ وقت اور پروڈکشن اخراجات کو کم کر سکتے ہیں۔
- ہر موڈیلٹی کے لیے 400,000 سے زیادہ pretrained ماڈلز کے ساتھ درجنوں آرکیٹیکچرز۔

&#8207; 3. ماڈل کے لائف ٹائم کے ہر حصے کے لیے صحیح
فریم ورک کا انتخاب کریں:

  - 3 لائنز کے کوڈ میں جدید ترین ماڈلز تربیت دیں۔
  - ایک ماڈل کو کسی بھی وقت TF2.0/PyTorch/JAX فریم ورکس کے درمیان منتقل کریں۔
  - تربیت، تشخیص، اور پروڈکشن کے لیے بغیر کسی رکاوٹ کے صحیح فریم ورک کا انتخاب کریں۔

&#8207; 4. اپنے ضروریات کے مطابق آسانی سے ماڈل یا ایک مثال کو حسب ضرورت بنائیں:

  - ہم ہر آرکیٹیکچر کے لیے مثالیں فراہم کرتے ہیں تاکہ اصل مصنفین کے شائع شدہ نتائج کو دوبارہ پیدا کیا جا سکے۔
  - ماڈلز کی اندرونی تفصیلات کو جتنا ممکن ہو یکساں طور پر ظاہر کیا جاتا ہے۔
  - فوری تجربات کے لیے ماڈل فائلز کو لائبریری سے آزادانہ طور پر استعمال کیا جا سکتا ہے۔

## مجھے Transformers کیوں استعمال نہیں کرنا چاہیے؟

- یہ لائبریری نیورل نیٹس کے لیے بلڈنگ بلاکس کا ماڈیولر ٹول باکس نہیں ہے۔ ماڈل فائلز میں موجود کوڈ جان بوجھ کر اضافی ایبسٹریکشنز کے ساتھ دوبارہ ترتیب نہیں دیا گیا ہے، تاکہ محققین بغیر اضافی ایبسٹریکشنز/فائلوں میں گئے ہوئے جلدی سے ہر ماڈل پر کام کر سکیں۔
- تربیتی API کا مقصد کسی بھی ماڈل پر کام کرنے کے لیے نہیں ہے بلکہ یہ لائبریری کے فراہم کردہ ماڈلز کے ساتھ کام کرنے کے لیے بہتر بنایا گیا ہے۔ عام مشین لرننگ لوپس کے لیے، آپ کو دوسری لائبریری (ممکنہ طور پر [Accelerate](https://huggingface.co/docs/accelerate)) استعمال کرنی چاہیے۔
- حالانکہ ہم جتنا ممکن ہو زیادہ سے زیادہ استعمال کے کیسز پیش کرنے کی کوشش کرتے ہیں، ہمارے [مثالوں کے فولڈر](https://github.com/huggingface/transformers/tree/main/examples) میں موجود اسکرپٹس صرف یہی ہیں: مثالیں۔ یہ توقع کی جاتی ہے کہ یہ آپ کے مخصوص مسئلے پر فوراً کام نہیں کریں گی اور آپ کو اپنی ضروریات کے مطابق کوڈ کی کچھ لائنیں تبدیل کرنی پڑیں گی۔

### انسٹالیشن

#### &#8207; pip کے ساتھ

یہ ریپوزٹری Python 3.9+، Flax 0.4.1+، PyTorch 2.0+، اور TensorFlow 2.6+ پر ٹیسٹ کی گئی ہے۔

آپ کو 🤗 Transformers کو ایک [ورچوئل ماحول](https://docs.python.org/3/library/venv.html) میں انسٹال کرنا چاہیے۔ اگر آپ Python ورچوئل ماحول سے واقف نہیں ہیں، تو [یوزر گائیڈ](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/) دیکھیں۔

پہلے، Python کے اس ورژن کے ساتھ ایک ورچوئل ماحول بنائیں جو آپ استعمال کر رہے ہیں اور اسے ایکٹیویٹ کریں۔

پھر، آپ کو کم از کم Flax، PyTorch، یا TensorFlow میں سے کسی ایک کو انسٹال کرنے کی ضرورت ہوگی۔
براہ کرم اپنے پلیٹ فارم کے لیے مخصوص انسٹالیشن کمانڈ کے حوالے سے [TensorFlow انسٹالیشن صفحہ](https://www.tensorflow.org/install/)، [PyTorch انسٹالیشن صفحہ](https://pytorch.org/get-started/locally/#start-locally) اور/یا [Flax](https://github.com/google/flax#quick-install) اور [Jax](https://github.com/google/jax#installation) انسٹالیشن صفحات دیکھیں۔

جب ان میں سے کوئی ایک بیک اینڈ انسٹال ہو جائے، تو 🤗 Transformers کو pip کے ذریعے مندرجہ ذیل طریقے سے انسٹال کیا جا سکتا ہے:

```bash
pip install transformers
```

اگر آپ مثالوں کے ساتھ کھیلنا چاہتے ہیں یا آپ کو کوڈ کا تازہ ترین ورژن چاہیے اور آپ نئے ریلیز کا انتظار نہیں کر سکتے، تو آپ کو [سورس سے لائبریری انسٹال کرنی ہوگی](https://huggingface.co/docs/transformers/installation#installing-from-source)۔

#### &#8207;conda کے ساتھ

&#8207;🤗 Transformers کو conda کے ذریعے مندرجہ ذیل طریقے سے انسٹال کیا جا سکتا ہے:

```shell script
conda install conda-forge::transformers
```

> **_نوٹ:_** `transformers` کو `huggingface` چینل سے انسٹال کرنا اب ختم کیا جا چکا ہے۔

Flax، PyTorch، یا TensorFlow کو conda کے ساتھ انسٹال کرنے کے لیے انسٹالیشن صفحات کی پیروی کریں۔

> **_نوٹ:_**  ونڈوز پر، آپ کو کیشنگ سے فائدہ اٹھانے کے لیے ڈویلپر موڈ کو ایکٹیویٹ کرنے کا پیغام دیا جا سکتا ہے۔ اگر یہ آپ کے لیے ممکن نہیں ہے، تو براہ کرم ہمیں [اس مسئلے](https://github.com/huggingface/huggingface_hub/issues/1062) میں بتائیں۔

### ماڈل کی تعمیرات

&#8207; 🤗 Transformers کی طرف سے فراہم کردہ **[تمام ماڈل چیک پوائنٹس](https://huggingface.co/models)** ہگنگ فیس کے ماڈل حب [model hub](https://huggingface.co/models) سے بآسانی مربوط ہیں، جہاں یہ براہ راست [صارفین](https://huggingface.co/users) اور [تنظیموں](https://huggingface.co/organizations) کے ذریعہ اپ لوڈ کیے جاتے ہیں۔

چیک پوائنٹس کی موجودہ تعداد: ![](https://img.shields.io/endpoint?url=https://huggingface.co/api/shields/models&color=brightgreen)

&#8207;🤗 Transformers فی الحال درج ذیل معماریاں فراہم کرتا ہے: ہر ایک کا اعلی سطحی خلاصہ دیکھنے کے لیے [یہاں](https://huggingface.co/docs/transformers/model_summary) دیکھیں۔

یہ چیک کرنے کے لیے کہ ہر ماڈل کی Flax، PyTorch یا TensorFlow میں کوئی عملداری ہے یا 🤗 Tokenizers لائبریری کے ذریعہ سپورٹ کردہ ٹوکنائزر کے ساتھ ہے، [اس جدول](https://huggingface.co/docs/transformers/index#supported-frameworks) کا حوالہ لیں۔

یہ عملداری مختلف ڈیٹا سیٹس پر ٹیسٹ کی گئی ہیں (مثال کے اسکرپٹس دیکھیں) اور اصل عملداری کی کارکردگی کے ہم آہنگ ہونی چاہئیں۔ آپ کو کارکردگی کی مزید تفصیلات [دستاویزات](https://github.com/huggingface/transformers/tree/main/examples) کے مثالوں کے سیکشن میں مل سکتی ہیں۔


## مزید معلومات حاصل کریں

| سیکشن | تفصیل |
|-|-|
| [دستاویزات](https://huggingface.co/docs/transformers/) | مکمل API دستاویزات اور ٹیوٹوریلز |
| [ٹاسک کا خلاصہ](https://huggingface.co/docs/transformers/task_summary) | 🤗 Transformers کے ذریعہ سپورٹ کردہ ٹاسک |
| [پری پروسیسنگ ٹیوٹوریل](https://huggingface.co/docs/transformers/preprocessing) | ماڈلز کے لیے ڈیٹا تیار کرنے کے لیے `Tokenizer` کلاس کا استعمال |
| [ٹریننگ اور فائن ٹیوننگ](https://huggingface.co/docs/transformers/training) | PyTorch/TensorFlow ٹریننگ لوپ میں 🤗 Transformers کی طرف سے فراہم کردہ ماڈلز کا استعمال اور `Trainer` API |
| [تیز دورہ: فائن ٹیوننگ/استعمال کے اسکرپٹس](https://github.com/huggingface/transformers/tree/main/examples) | مختلف قسم کے ٹاسک پر ماڈلز کو فائن ٹیون کرنے کے لیے مثال کے اسکرپٹس |
| [ماڈل کا اشتراک اور اپ لوڈ کرنا](https://huggingface.co/docs/transformers/model_sharing) | اپنی فائن ٹیون کردہ ماڈلز کو کمیونٹی کے ساتھ اپ لوڈ اور شیئر کریں |

## استشہاد

ہم نے اب ایک [تحقیقی مقالہ](https://www.aclweb.org/anthology/2020.emnlp-demos.6/) تیار کیا ہے جسے آپ 🤗 Transformers لائبریری کے لیے حوالہ دے سکتے ہیں:

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
