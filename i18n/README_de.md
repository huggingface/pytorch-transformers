<!---
Copyright 2024 The HuggingFace Team. All rights reserved.

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
        <a href="https://github.com/huggingface/transformers/">English</a> |
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
        <b>Deutsch</b> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_vi.md">Tiếng Việt</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ar.md">العربية</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ur.md">اردو</a> |
    </p>
</h4>

<h3 align="center">
    <p>Maschinelles Lernen auf dem neuesten Stand der Technik für JAX, PyTorch und TensorFlow</p>
</h3>

<h3 align="center">
    <a href="https://hf.co/course"><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/course_banner.png"></a>
</h3>

🤗 Transformers bietet Tausende von vortrainierten Modellen, um Aufgaben in verschiedenen Modalitäten wie Text, Bild und Audio durchzuführen.

Diese Modelle können angewendet werden, auf:

* 📝 Text - für Aufgaben wie Textklassifizierung, Informationsextraktion, Question Answering, automatische Textzusammenfassung, maschinelle Übersetzung und Textgenerierung in über 100 Sprachen.
* 🖼️ Bilder - für Aufgaben wie Bildklassifizierung, Objekterkennung und Segmentierung.
* 🗣️ Audio - für Aufgaben wie Spracherkennung und Audioklassifizierung.

Transformer-Modelle können auch Aufgaben für **mehrere Modalitäten in Kombination** durchführen, z. B. tabellenbasiertes Question Answering, optische Zeichenerkennung, Informationsextraktion aus gescannten Dokumenten, Videoklassifizierung und visuelles Question Answering.

🤗 Transformers bietet APIs, um diese vortrainierten Modelle schnell herunterzuladen und für einen gegebenen Text zu verwenden, sie auf Ihren eigenen Datensätzen zu feintunen und dann mit der Community in unserem [Model Hub](https://huggingface.co/models) zu teilen. Gleichzeitig ist jedes Python-Modul, das eine Architektur definiert, komplett eigenständig und kann modifiziert werden, um schnelle Forschungsexperimente zu ermöglichen.

🤗 Transformers unterstützt die nahtlose Integration von drei der beliebtesten Deep-Learning-Bibliotheken: [Jax](https://jax.readthedocs.io/en/latest/), [PyTorch](https://pytorch.org/) und [TensorFlow](https://www.tensorflow.org/). Trainieren Sie Ihr Modell in einem Framework und laden Sie es zur Inferenz unkompliziert mit einem anderen.

## Online-Demos

Sie können die meisten unserer Modelle direkt auf ihren Seiten im [Model Hub](https://huggingface.co/models) testen. Wir bieten auch [privates Modell-Hosting, Versionierung, & eine Inferenz-API](https://huggingface.co/pricing) für öffentliche und private Modelle an.

Hier sind einige Beispiele:

In der Computerlinguistik:

- [Maskierte Wortvervollständigung mit BERT](https://huggingface.co/google-bert/bert-base-uncased?text=Paris+is+the+%5BMASK%5D+of+France)
- [Eigennamenerkennung mit Electra](https://huggingface.co/dbmdz/electra-large-discriminator-finetuned-conll03-english?text=My+name+is+Sarah+and+I+live+in+London+city)
- [Textgenerierung mit GPT-2](https://huggingface.co/openai-community/gpt2?text=A+long+time+ago%2C+)
- [Natural Language Inference mit RoBERTa](https://huggingface.co/FacebookAI/roberta-large-mnli?text=The+dog+was+lost.+Nobody+lost+any+animal)
- [Automatische Textzusammenfassung mit BART](https://huggingface.co/facebook/bart-large-cnn?text=The+tower+is+324+metres+%281%2C063+ft%29+tall%2C+about+the+same+height+as+an+81-storey+building%2C+and+the+tallest+structure+in+Paris.+Its+base+is+square%2C+measuring+125+metres+%28410+ft%29+on+each+side.+During+its+construction%2C+the+Eiffel+Tower+surpassed+the+Washington+Monument+to+become+the+tallest+man-made+structure+in+the+world%2C+a+title+it+held+for+41+years+until+the+Chrysler+Building+in+New+York+City+was+finished+in+1930.+It+was+the+first+structure+to+reach+a+height+of+300+metres.+Due+to+the+addition+of+a+broadcasting+aerial+at+the+top+of+the+tower+in+1957%2C+it+is+now+taller+than+the+Chrysler+Building+by+5.2+metres+%2817+ft%29.+Excluding+transmitters%2C+the+Eiffel+Tower+is+the+second+tallest+free-standing+structure+in+France+after+the+Millau+Viaduct)
- [Question Answering mit DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased-distilled-squad?text=Which+name+is+also+used+to+describe+the+Amazon+rainforest+in+English%3F&context=The+Amazon+rainforest+%28Portuguese%3A+Floresta+Amaz%C3%B4nica+or+Amaz%C3%B4nia%3B+Spanish%3A+Selva+Amaz%C3%B3nica%2C+Amazon%C3%ADa+or+usually+Amazonia%3B+French%3A+For%C3%AAt+amazonienne%3B+Dutch%3A+Amazoneregenwoud%29%2C+also+known+in+English+as+Amazonia+or+the+Amazon+Jungle%2C+is+a+moist+broadleaf+forest+that+covers+most+of+the+Amazon+basin+of+South+America.+This+basin+encompasses+7%2C000%2C000+square+kilometres+%282%2C700%2C000+sq+mi%29%2C+of+which+5%2C500%2C000+square+kilometres+%282%2C100%2C000+sq+mi%29+are+covered+by+the+rainforest.+This+region+includes+territory+belonging+to+nine+nations.+The+majority+of+the+forest+is+contained+within+Brazil%2C+with+60%25+of+the+rainforest%2C+followed+by+Peru+with+13%25%2C+Colombia+with+10%25%2C+and+with+minor+amounts+in+Venezuela%2C+Ecuador%2C+Bolivia%2C+Guyana%2C+Suriname+and+French+Guiana.+States+or+departments+in+four+nations+contain+%22Amazonas%22+in+their+names.+The+Amazon+represents+over+half+of+the+planet%27s+remaining+rainforests%2C+and+comprises+the+largest+and+most+biodiverse+tract+of+tropical+rainforest+in+the+world%2C+with+an+estimated+390+billion+individual+trees+divided+into+16%2C000+species)
- [Maschinelle Übersetzung mit T5](https://huggingface.co/google-t5/t5-base?text=My+name+is+Wolfgang+and+I+live+in+Berlin)

In der Computer Vision:

- [Bildklassifizierung mit ViT](https://huggingface.co/google/vit-base-patch16-224)
- [Objekterkennung mit DETR](https://huggingface.co/facebook/detr-resnet-50)
- [Semantische Segmentierung mit SegFormer](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512)
- [Panoptische Segmentierung mit MaskFormer](https://huggingface.co/facebook/maskformer-swin-small-coco)
- [Depth Estimation mit DPT](https://huggingface.co/docs/transformers/model_doc/dpt)
- [Videoklassifizierung mit VideoMAE](https://huggingface.co/docs/transformers/model_doc/videomae)
- [Universelle Segmentierung mit OneFormer](https://huggingface.co/shi-labs/oneformer_ade20k_dinat_large)

Im Audio-Bereich:

- [Automatische Spracherkennung mit Wav2Vec2](https://huggingface.co/facebook/wav2vec2-base-960h)
- [Keyword Spotting mit Wav2Vec2](https://huggingface.co/superb/wav2vec2-base-superb-ks)
- [Audioklassifizierung mit Audio Spectrogram Transformer](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593)

In multimodalen Aufgaben:

- [Tabellenbasiertes Question Answering mit TAPAS](https://huggingface.co/google/tapas-base-finetuned-wtq)
- [Visuelles Question Answering mit ViLT](https://huggingface.co/dandelin/vilt-b32-finetuned-vqa)
- [Zero-Shot-Bildklassifizierung mit CLIP](https://huggingface.co/openai/clip-vit-large-patch14)
- [Dokumentenbasiertes Question Answering mit LayoutLM](https://huggingface.co/impira/layoutlm-document-qa)
- [Zero-Shot-Videoklassifizierung mit X-CLIP](https://huggingface.co/docs/transformers/model_doc/xclip)

## 100 Projekte, die 🤗 Transformers verwenden

🤗 Transformers ist mehr als nur ein Toolkit zur Verwendung von vortrainierten Modellen: Es ist eine Gemeinschaft von Projekten, die darum herum und um den Hugging Face Hub aufgebaut sind. Wir möchten, dass 🤗 Transformers es Entwicklern, Forschern, Studenten, Professoren, Ingenieuren und jedem anderen ermöglicht, ihre Traumprojekte zu realisieren.

Um die 100.000 Sterne von 🤗 Transformers zu feiern, haben wir beschlossen, die Gemeinschaft in den Mittelpunkt zu stellen und die Seite [awesome-transformers](./awesome-transformers.md) erstellt, die 100 unglaubliche Projekte auflistet, die zusammen mit 🤗 Transformers realisiert wurden.

Wenn Sie ein Projekt besitzen oder nutzen, von dem Sie glauben, dass es Teil der Liste sein sollte, öffnen Sie bitte einen PR, um es hinzuzufügen!

## Wenn Sie individuelle Unterstützung vom Hugging Face-Team möchten

<a target="_blank" href="https://huggingface.co/support">
    <img alt="HuggingFace Expert Acceleration Program" src="https://cdn-media.huggingface.co/marketing/transformers/new-support-improved.png" style="max-width: 600px; border: 1px solid #eee; border-radius: 4px; box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);">
</a><br>

## Schnelleinstieg

Um sofort ein Modell mit einer bestimmten Eingabe (Text, Bild, Audio ...) zu verwenden, bieten wir die `pipeline`-API an. Pipelines kombinieren ein vortrainiertes Modell mit der jeweiligen Vorverarbeitung, die während dessen Trainings verwendet wurde. Hier sehen Sie, wie man schnell eine Pipeline verwenden kann, um positive und negative Texte zu klassifizieren:

```python
>>> from transformers import pipeline

# Zuweisung einer Pipeline für die Sentiment-Analyse
>>> classifier = pipeline('sentiment-analysis')
>>> classifier('We are very happy to introduce pipeline to the transformers repository.')
[{'label': 'POSITIVE', 'score': 0.9996980428695679}]
```

Die zweite Codezeile lädt und cacht das vortrainierte Modell, das von der Pipeline verwendet wird, während die dritte es an dem gegebenen Text evaluiert. Hier ist die Antwort "positiv" mit einer Konfidenz von 99,97 %.

Viele Aufgaben, sowohl in der Computerlinguistik als auch in der Computer Vision und Sprachverarbeitung, haben eine vortrainierte `pipeline`, die sofort einsatzbereit ist. Z. B. können wir leicht erkannte Objekte in einem Bild extrahieren:

``` python
>>> import requests
>>> from PIL import Image
>>> from transformers import pipeline

# Download eines Bildes mit süßen Katzen
>>> url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png"
>>> image_data = requests.get(url, stream=True).raw
>>> image = Image.open(image_data)

# Zuweisung einer Pipeline für die Objekterkennung
>>> object_detector = pipeline('object-detection')
>>> object_detector(image)
[{'score': 0.9982201457023621,
  'label': 'remote',
  'box': {'xmin': 40, 'ymin': 70, 'xmax': 175, 'ymax': 117}},
 {'score': 0.9960021376609802,
  'label': 'remote',
  'box': {'xmin': 333, 'ymin': 72, 'xmax': 368, 'ymax': 187}},
 {'score': 0.9954745173454285,
  'label': 'couch',
  'box': {'xmin': 0, 'ymin': 1, 'xmax': 639, 'ymax': 473}},
 {'score': 0.9988006353378296,
  'label': 'cat',
  'box': {'xmin': 13, 'ymin': 52, 'xmax': 314, 'ymax': 470}},
 {'score': 0.9986783862113953,
  'label': 'cat',
  'box': {'xmin': 345, 'ymin': 23, 'xmax': 640, 'ymax': 368}}]
```

Hier erhalten wir eine Liste von Objekten, die im Bild erkannt wurden, mit einer Markierung, die das Objekt eingrenzt, und einem zugehörigen Konfidenzwert. Folgend ist das Originalbild links und die Vorhersagen rechts dargestellt:

<h3 align="center">
    <a><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png" width="400"></a>
    <a><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample_post_processed.png" width="400"></a>
</h3>

Sie können mehr über die von der `pipeline`-API unterstützten Aufgaben in [diesem Tutorial](https://huggingface.co/docs/transformers/task_summary) erfahren.

Zusätzlich zur `pipeline` benötigt es nur drei Zeilen Code, um eines der vortrainierten Modelle für Ihre Aufgabe herunterzuladen und zu verwenden. Hier ist der Code für die PyTorch-Version:

```python
>>> from transformers import AutoTokenizer, AutoModel

>>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
>>> model = AutoModel.from_pretrained("google-bert/bert-base-uncased")

>>> inputs = tokenizer("Hello world!", return_tensors="pt")
>>> outputs = model(**inputs)
```

Und hier ist der entsprechende Code für TensorFlow:

```python
>>> from transformers import AutoTokenizer, TFAutoModel

>>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
>>> model = TFAutoModel.from_pretrained("google-bert/bert-base-uncased")

>>> inputs = tokenizer("Hello world!", return_tensors="tf")
>>> outputs = model(**inputs)
```

Der Tokenizer ist für die gesamte Vorverarbeitung, die das vortrainierte Modell benötigt, verantwortlich und kann direkt auf einem einzelnen String (wie in den obigen Beispielen) oder einer Liste ausgeführt werden. Er gibt ein Dictionary aus, das Sie im darauffolgenden Code verwenden oder einfach direkt Ihrem Modell übergeben können, indem Sie den ** Operator zum Entpacken von Argumenten einsetzen.

Das Modell selbst ist ein reguläres [PyTorch `nn.Module`](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) oder ein [TensorFlow `tf.keras.Model`](https://www.tensorflow.org/api_docs/python/tf/keras/Model) (abhängig von Ihrem Backend), das Sie wie gewohnt verwenden können. [Dieses Tutorial](https://huggingface.co/docs/transformers/training) erklärt, wie man ein solches Modell in eine klassische PyTorch- oder TensorFlow-Trainingsschleife integrieren kann oder wie man unsere `Trainer`-API verwendet, um es schnell auf einem neuen Datensatz zu feintunen.

## Warum sollten Sie 🤗 Transformers verwenden?

1. Benutzerfreundliche Modelle auf dem neuesten Stand der Technik:
    - Hohe Leistung bei Aufgaben zu Natural Language Understanding & Generation, Computer Vision und Audio.
    - Niedrige Einstiegshürde für Bildungskräfte und Praktiker.
    - Wenige benutzerseitige Abstraktionen mit nur drei zu lernenden Klassen.
    - Eine einheitliche API für die Verwendung aller unserer vortrainierten Modelle.

1. Geringere Rechenkosten, kleinerer CO<sub>2</sub>-Fußabdruck:
    - Forscher können trainierte Modelle teilen, anstatt sie immer wieder neu zu trainieren.
    - Praktiker können die Rechenzeit und Produktionskosten reduzieren.
    - Dutzende Architekturen mit über 400.000 vortrainierten Modellen über alle Modalitäten hinweg.

1. Wählen Sie das richtige Framework für jeden Lebensabschnitt eines Modells:
    - Trainieren Sie Modelle auf neustem Stand der Technik in nur drei Codezeilen.
    - Verwenden Sie ein einzelnes Modell nach Belieben mit TF2.0-/PyTorch-/JAX-Frameworks.
    - Wählen Sie nahtlos das richtige Framework für Training, Evaluation und Produktiveinsatz.

1. Passen Sie ein Modell oder Beispiel leicht an Ihre Bedürfnisse an:
    - Wir bieten Beispiele für jede Architektur an, um die von ihren ursprünglichen Autoren veröffentlichten Ergebnisse zu reproduzieren.
    - Modellinterna sind so einheitlich wie möglich verfügbar gemacht.
    - Modelldateien können unabhängig von der Bibliothek für schnelle Experimente verwendet werden.

## Warum sollten Sie 🤗 Transformers nicht verwenden?

- Diese Bibliothek ist kein modularer Werkzeugkasten mit Bausteinen für neuronale Netze. Der Code in den Modelldateien ist absichtlich nicht mit zusätzlichen Abstraktionen refaktorisiert, sodass Forscher schnell mit jedem der Modelle iterieren können, ohne sich in zusätzliche Abstraktionen/Dateien vertiefen zu müssen.
- Die Trainings-API ist nicht dafür gedacht, mit beliebigen Modellen zu funktionieren, sondern ist für die Verwendung mit den von der Bibliothek bereitgestellten Modellen optimiert. Für generische Trainingsschleifen von maschinellem Lernen sollten Sie eine andere Bibliothek verwenden (möglicherweise [Accelerate](https://huggingface.co/docs/accelerate)).
- Auch wenn wir bestrebt sind, so viele Anwendungsfälle wie möglich zu veranschaulichen, sind die Beispielskripte in unserem [`examples`](./examples) Ordner genau das: Beispiele. Es ist davon auszugehen, dass sie nicht sofort auf Ihr spezielles Problem anwendbar sind und einige Codezeilen geändert werden müssen, um sie für Ihre Bedürfnisse anzupassen.

## Installation

### Mit pip

Dieses Repository wurde mit Python 3.9+, Flax 0.4.1+, PyTorch 2.0+ und TensorFlow 2.6+ getestet.

Sie sollten 🤗 Transformers in einer [virtuellen Umgebung](https://docs.python.org/3/library/venv.html) installieren. Wenn Sie mit virtuellen Python-Umgebungen nicht vertraut sind, schauen Sie sich den [Benutzerleitfaden](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/) an.

Erstellen und aktivieren Sie zuerst eine virtuelle Umgebung mit der Python-Version, die Sie verwenden möchten.

Dann müssen Sie entweder Flax, PyTorch oder TensorFlow installieren. Bitte beziehe dich entsprechend auf die jeweiligen Installationsanleitungen für [TensorFlow](https://www.tensorflow.org/install/), [PyTorch](https://pytorch.org/get-started/locally/#start-locally), und/oder [Flax](https://github.com/google/flax#quick-install) und [Jax](https://github.com/google/jax#installation) für den spezifischen Installationsbefehl für Ihre Plattform.

Wenn eines dieser Backends installiert ist, kann 🤗 Transformers wie folgt mit pip installiert werden:

```bash
pip install transformers
```

Wenn Sie mit den Beispielen experimentieren möchten oder die neueste Version des Codes benötigen und nicht auf eine neue Veröffentlichung warten können, müssen Sie [die Bibliothek von der Quelle installieren](https://huggingface.co/docs/transformers/installation#installing-from-source).

### Mit conda

🤗 Transformers kann wie folgt mit conda installiert werden:

```shell script
conda install conda-forge::transformers
```

> **_HINWEIS:_** Die Installation von `transformers` aus dem `huggingface`-Kanal ist veraltet.

Folgen Sie den Installationsanleitungen von Flax, PyTorch oder TensorFlow, um zu sehen, wie sie mit conda installiert werden können.

> **_HINWEIS:_** Auf Windows werden Sie möglicherweise aufgefordert, den Entwicklermodus zu aktivieren, um von Caching zu profitieren. Wenn das für Sie keine Option ist, lassen Sie es uns bitte in [diesem Issue](https://github.com/huggingface/huggingface_hub/issues/1062) wissen.

## Modellarchitekturen

**[Alle Modell-Checkpoints](https://huggingface.co/models)**, die von 🤗 Transformers bereitgestellt werden, sind nahtlos aus dem huggingface.co [Model Hub](https://huggingface.co/models) integriert, wo sie direkt von [Benutzern](https://huggingface.co/users) und [Organisationen](https://huggingface.co/organizations) hochgeladen werden.

Aktuelle Anzahl der Checkpoints: ![](https://img.shields.io/endpoint?url=https://huggingface.co/api/shields/models&color=brightgreen)

🤗 Transformers bietet derzeit die folgenden Architekturen an: siehe [hier](https://huggingface.co/docs/transformers/model_summary) für eine jeweilige Übersicht.

Um zu überprüfen, ob jedes Modell eine Implementierung in Flax, PyTorch oder TensorFlow hat oder über einen zugehörigen Tokenizer verfügt, der von der 🤗 Tokenizers-Bibliothek unterstützt wird, schauen Sie auf [diese Tabelle](https://huggingface.co/docs/transformers/index#supported-frameworks).

Diese Implementierungen wurden mit mehreren Datensätzen getestet (siehe Beispielskripte) und sollten den Leistungen der ursprünglichen Implementierungen entsprechen. Weitere Details zur Leistung finden Sie im Abschnitt der Beispiele in der [Dokumentation](https://github.com/huggingface/transformers/tree/main/examples).

## Mehr erfahren

| Abschnitt | Beschreibung |
|-|-|
| [Dokumentation](https://huggingface.co/docs/transformers/) | Vollständige API-Dokumentation und Tutorials |
| [Zusammenfassung der Aufgaben](https://huggingface.co/docs/transformers/task_summary) | Von 🤗 Transformers unterstützte Aufgaben |
| [Vorverarbeitungs-Tutorial](https://huggingface.co/docs/transformers/preprocessing) | Verwendung der `Tokenizer`-Klasse zur Vorverarbeitung der Daten für die Modelle |
| [Training und Feintuning](https://huggingface.co/docs/transformers/training) | Verwendung der von 🤗 Transformers bereitgestellten Modelle in einer PyTorch-/TensorFlow-Trainingsschleife und der `Trainer`-API |
| [Schnelleinstieg: Feintuning/Anwendungsskripte](https://github.com/huggingface/transformers/tree/main/examples) | Beispielskripte für das Feintuning von Modellen für eine breite Palette von Aufgaben |
| [Modellfreigabe und -upload](https://huggingface.co/docs/transformers/model_sharing) | Laden Sie Ihre feingetunten Modelle hoch und teilen Sie sie mit der Community |

## Zitation

Wir haben jetzt ein [Paper](https://www.aclweb.org/anthology/2020.emnlp-demos.6/), das Sie für die 🤗 Transformers-Bibliothek zitieren können:

```bibtex
@inproceedings{wolf-etal-2020-transformers,
    title = "Transformers: State-of-the-Art Natural Language Processing",
    author = "Thomas Wolf and Lysandre Debut and Victor Sanh and Julien Chaumond and Clement Delangue and Anthony Moi and Pierric Cistac and Tim Rault and Rémi Louf and Morgan Funtowicz and Joe Davison and Sam Shleifer and Patrick von Platen and Clara Ma and Yacine Jernite and Julien Plu and Canwen Xu and Teven Le Scao and Sylvain Gugger and Mariama Drame and Quentin Lhoest and Alexander M. Rush",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
    month = oct,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-demos.6",
    pages = "38--45"
}
```
