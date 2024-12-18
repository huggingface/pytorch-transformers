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

# Schnellstart

[[open-in-colab]]

Mit 🤗 Transformers können Sie sofort loslegen! Verwenden Sie die [`pipeline`] für schnelle Inferenz und laden Sie schnell ein vortrainiertes Modell und einen Tokenizer mit einer [AutoClass](./model_doc/auto), um Ihre Text-, Bild- oder Audioaufgabe zu lösen.

<Tip>

Alle in der Dokumentation vorgestellten Codebeispiele haben oben links einen Umschalter für PyTorch und TensorFlow. Wenn
nicht, wird erwartet, dass der Code für beide Backends ohne Änderungen funktioniert.

</Tip>

## Pipeline

[`pipeline`] ist der einfachste Weg, ein vortrainiertes Modell für eine bestimmte Aufgabe zu verwenden.

<Youtube id="tiZFewofSLM"/>

Die [`pipeline`] unterstützt viele gängige Aufgaben:

**Text**:
* Stimmungsanalyse: Klassifizierung der Polarität eines gegebenen Textes.
* Textgenerierung (auf Englisch): Generierung von Text aus einer gegebenen Eingabe.
* Name-Entity-Recognition (NER): Kennzeichnung jedes Worts mit der Entität, die es repräsentiert (Person, Datum, Ort usw.).
* Beantwortung von Fragen: Extrahieren der Antwort aus dem Kontext, wenn ein gewisser Kontext und eine Frage gegeben sind.
* Fill-mask: Ausfüllen von Lücken in einem Text mit maskierten Wörtern.
* Zusammenfassung: Erstellung einer Zusammenfassung einer langen Text- oder Dokumentensequenz.
* Übersetzung: Übersetzen eines Textes in eine andere Sprache.
* Merkmalsextraktion: Erstellen einer Tensordarstellung des Textes.

**Bild**:
* Bildklassifizierung: Klassifizierung eines Bildes.
* Bildsegmentierung: Klassifizierung jedes Pixels in einem Bild.
* Objekterkennung: Erkennen von Objekten innerhalb eines Bildes.

**Audio**:
* Audioklassifizierung: Zuweisung eines Labels zu einem bestimmten Audiosegment.
* Automatische Spracherkennung (ASR): Transkription von Audiodaten in Text.

<Tip>

Für mehr Details über die [`pipeline`] und assoziierte Aufgaben, schauen Sie in die Dokumentation [hier](./main_classes/pipelines).

</Tip>

### Verwendung der Pipeline

Im folgenden Beispiel werden Sie die [`pipeline`] für die Stimmungsanalyse verwenden.

Installieren Sie die folgenden Abhängigkeiten, falls Sie dies nicht bereits getan haben:

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

Importieren sie die [`pipeline`] und spezifizieren sie die Aufgabe, welche sie lösen möchten:

```py
>>> from transformers import pipeline

>>> classifier = pipeline("sentiment-analysis")
```

Die Pipeline lädt ein standardmäßiges [vortrainiertes Modell](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english) und einen Tokenizer für die Stimmungs-Analyse herunter und speichert sie. Jetzt können Sie den "Klassifikator" auf Ihren Zieltext anwenden:

```py
>>> classifier("We are very happy to show you the 🤗 Transformers library.")
[{'label': 'POSITIVE', 'score': 0.9998}]
```

For more than one sentence, pass a list of sentences to the [`pipeline`] which returns a list of dictionaries:

```py
>>> results = classifier(["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."])
>>> for result in results:
...     print(f"label: {result['label']}, with score: {round(result['score'], 4)}")
label: POSITIVE, with score: 0.9998
label: NEGATIVE, with score: 0.5309
```

Die [`pipeline`] kann auch über einen ganzen Datensatz iterieren. Starten wir mit der Installation der [🤗 Datasets](https://huggingface.co/docs/datasets/) Bibliothek:

```bash
pip install datasets
```

Erstellen wir eine [`pipeline`] mit der Aufgabe die wir lösen und dem Modell welches wir nutzen möchten.

```py
>>> import torch
>>> from transformers import pipeline

>>> speech_recognizer = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")
```

Als nächstes laden wir den Datensatz (siehe 🤗 Datasets [Quick Start](https://huggingface.co/docs/datasets/quickstart) für mehr Details) welches wir nutzen möchten. Zum Beispiel laden wir den [MInDS-14](https://huggingface.co/datasets/PolyAI/minds14) Datensatz:

```py
>>> from datasets import load_dataset, Audio

>>> dataset = load_dataset("PolyAI/minds14", name="en-US", split="train")  # doctest: +IGNORE_RESULT
```

Wir müssen sicherstellen, dass die Abtastrate des Datensatzes der Abtastrate entspricht, mit der `facebook/wav2vec2-base-960h` trainiert wurde.

```py
>>> dataset = dataset.cast_column("audio", Audio(sampling_rate=speech_recognizer.feature_extractor.sampling_rate))
```

Audiodateien werden automatisch geladen und neu abgetastet, wenn die Spalte "audio" aufgerufen wird.
Extrahieren wir die rohen Wellenform-Arrays der ersten 4 Beispiele und übergeben wir sie als Liste an die Pipeline:

```py
>>> result = speech_recognizer(dataset[:4]["audio"])
>>> print([d["text"] for d in result])
['I WOULD LIKE TO SET UP A JOINT ACCOUNT WITH MY PARTNER HOW DO I PROCEED WITH DOING THAT', "FODING HOW I'D SET UP A JOIN TO HET WITH MY WIFE AND WHERE THE AP MIGHT BE", "I I'D LIKE TOY SET UP A JOINT ACCOUNT WITH MY PARTNER I'M NOT SEEING THE OPTION TO DO IT ON THE AP SO I CALLED IN TO GET SOME HELP CAN I JUST DO IT OVER THE PHONE WITH YOU AND GIVE YOU THE INFORMATION OR SHOULD I DO IT IN THE AP AND I'M MISSING SOMETHING UQUETTE HAD PREFERRED TO JUST DO IT OVER THE PHONE OF POSSIBLE THINGS", 'HOW DO I THURN A JOIN A COUNT']
```

Bei einem größeren Datensatz mit vielen Eingaben (wie bei Sprache oder Bildverarbeitung) sollten Sie einen Generator anstelle einer Liste übergeben, der alle Eingaben in den Speicher lädt. Weitere Informationen finden Sie in der [Pipeline-Dokumentation](./main_classes/pipelines).

### Ein anderes Modell und einen anderen Tokenizer in der Pipeline verwenden

Die [`pipeline`] kann jedes Modell aus dem [Model Hub](https://huggingface.co/models) verwenden, wodurch es einfach ist, die [`pipeline`] für andere Anwendungsfälle anzupassen. Wenn Sie beispielsweise ein Modell wünschen, das französischen Text verarbeiten kann, verwenden Sie die Tags im Model Hub, um nach einem geeigneten Modell zu filtern. Das oberste gefilterte Ergebnis liefert ein mehrsprachiges [BERT-Modell](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment), das auf die Stimmungsanalyse abgestimmt ist. Großartig, verwenden wir dieses Modell!

```py
>>> model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
```

<frameworkcontent>
<pt>
Use the [`AutoModelForSequenceClassification`] and [`AutoTokenizer`] to load the pretrained model and it's associated tokenizer (more on an `AutoClass` below):

```py
>>> from transformers import AutoTokenizer, AutoModelForSequenceClassification

>>> model = AutoModelForSequenceClassification.from_pretrained(model_name)
>>> tokenizer = AutoTokenizer.from_pretrained(model_name)
```
</pt>
<tf>
Use the [`TFAutoModelForSequenceClassification`] and [`AutoTokenizer`] to load the pretrained model and it's associated tokenizer (more on an `TFAutoClass` below):

```py
>>> from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

>>> model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
>>> tokenizer = AutoTokenizer.from_pretrained(model_name)
```
</tf>
</frameworkcontent>

Dann können Sie das Modell und den Tokenizer in der [`pipeline`] angeben und den `Klassifikator` auf Ihren Zieltext anwenden:

```py
>>> classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
>>> classifier("Nous sommes très heureux de vous présenter la bibliothèque 🤗 Transformers.")
[{'label': '5 stars', 'score': 0.7273}]
```

Wenn Sie kein Modell für Ihren Anwendungsfall finden können, müssen Sie ein vortrainiertes Modell auf Ihren Daten feinabstimmen. Schauen Sie sich unser [Feinabstimmungs-Tutorial](./training) an, um zu erfahren, wie das geht. Und schließlich, nachdem Sie Ihr trainiertes Modell verfeinert haben, sollten Sie es mit der Community im Model Hub teilen (siehe Tutorial [hier](./model_sharing)), um NLP für alle zu demokratisieren! 🤗

## AutoClass

<Youtube id="AhChOFRegn4"/>

Unter der Haube arbeiten die Klassen [`AutoModelForSequenceClassification`] und [`AutoTokenizer`] zusammen, um die [`pipeline`] zu betreiben. Eine [`AutoClass`](./model_doc/auto) ist eine Abkürzung, die automatisch die Architektur eines trainierten Modells aus dessen Namen oder Pfad abruft. Sie müssen nur die passende `AutoClass` für Ihre Aufgabe und den zugehörigen Tokenizer mit [`AutoTokenizer`] auswählen.

Kehren wir zu unserem Beispiel zurück und sehen wir uns an, wie Sie die `AutoClass` verwenden können, um die Ergebnisse der [`pipeline`] zu replizieren.

### AutoTokenizer

Ein Tokenizer ist für die Vorverarbeitung von Text in ein für das Modell verständliches Format zuständig. Zunächst zerlegt der Tokenisierer den Text in Wörter, die *Token* genannt werden. Es gibt mehrere Regeln für den Tokenisierungsprozess, z. B. wie und auf welcher Ebene ein Wort aufgespalten wird (weitere Informationen über Tokenisierung [hier](./tokenizer_summary)). Das Wichtigste ist jedoch, dass Sie den Tokenizer mit demselben Modellnamen instanziieren müssen, um sicherzustellen, dass Sie dieselben Tokenisierungsregeln verwenden, mit denen ein Modell zuvor trainiert wurde.
Laden sie einen Tokenizer mit [`AutoTokenizer`]:

```py
>>> from transformers import AutoTokenizer

>>> model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
>>> tokenizer = AutoTokenizer.from_pretrained(model_name)
```

Anschließend wandelt der Tokenizer die Token in Zahlen um, um einen Tensor als Eingabe für das Modell zu konstruieren. Dieser wird als *Vokabular* des Modells bezeichnet.

Übergeben Sie Ihren Text an den Tokenizer:

```py
>>> encoding = tokenizer("We are very happy to show you the 🤗 Transformers library.")
>>> print(encoding)
{'input_ids': [101, 11312, 10320, 12495, 19308, 10114, 11391, 10855, 10103, 100, 58263, 13299, 119, 102],
 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
```

Der Tokenizer gibt ein Wörterbuch zurück, das Folgendes enthält:

* [input_ids](./glossary#input-ids): numerische Repräsentationen Ihrer Token.
* [atttention_mask](.glossary#attention-mask): gibt an, welche Token beachtet werden sollen.

Genau wie die [`pipeline`] akzeptiert der Tokenizer eine Liste von Eingaben. Darüber hinaus kann der Tokenizer den Text auch auffüllen und kürzen, um einen Stapel mit einheitlicher Länge zurückzugeben:

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

Lesen Sie das Tutorial [preprocessing](./preprocessing) für weitere Details zur Tokenisierung.

### AutoModel

<frameworkcontent>
<pt>
🤗 Transformers bietet eine einfache und einheitliche Möglichkeit, vortrainierte Instanzen zu laden. Das bedeutet, dass Sie ein [`AutoModel`] laden können, wie Sie einen [`AutoTokenizer`] laden würden. Der einzige Unterschied ist die Auswahl des richtigen [`AutoModel`] für die Aufgabe. Da Sie eine Text- oder Sequenzklassifizierung vornehmen, laden Sie [`AutoModelForSequenceClassification`]:

```py
>>> from transformers import AutoModelForSequenceClassification

>>> model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
>>> pt_model = AutoModelForSequenceClassification.from_pretrained(model_name)
```

<Tip>

In der [Aufgabenzusammenfassung](./task_summary) steht, welche [AutoModel]-Klasse für welche Aufgabe zu verwenden ist.

</Tip>

Jetzt können Sie Ihren vorverarbeiteten Stapel von Eingaben direkt an das Modell übergeben. Sie müssen nur das Wörterbuch entpacken, indem Sie `**` hinzufügen:

```py
>>> pt_outputs = pt_model(**pt_batch)
```

Das Modell gibt die endgültigen Aktivierungen in dem Attribut "logits" aus. Wenden Sie die Softmax-Funktion auf die "logits" an, um die Wahrscheinlichkeiten zu erhalten:

```py
>>> from torch import nn

>>> pt_predictions = nn.functional.softmax(pt_outputs.logits, dim=-1)
>>> print(pt_predictions)
tensor([[0.0021, 0.0018, 0.0115, 0.2121, 0.7725],
        [0.2084, 0.1826, 0.1969, 0.1755, 0.2365]], grad_fn=<SoftmaxBackward0>)
```
</pt>
<tf>
🤗 Transformers bietet eine einfache und einheitliche Methode zum Laden von vortrainierten Instanzen. Das bedeutet, dass Sie ein [`TFAutoModel`] genauso laden können, wie Sie einen [`AutoTokenizer`] laden würden. Der einzige Unterschied ist die Auswahl des richtigen [`TFAutoModel`] für die Aufgabe. Da Sie Text - oder Sequenz - Klassifizierung machen, laden Sie [`TFAutoModelForSequenceClassification`]:

```py
>>> from transformers import TFAutoModelForSequenceClassification

>>> model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
>>> tf_model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
```

<Tip>

In der [Aufgabenzusammenfassung](./task_summary) steht, welche [AutoModel]-Klasse für welche Aufgabe zu verwenden ist.

</Tip>

Jetzt können Sie Ihren vorverarbeiteten Stapel von Eingaben direkt an das Modell übergeben, indem Sie die Wörterbuchschlüssel direkt an die Tensoren übergeben:

```py
>>> tf_outputs = tf_model(tf_batch)
```

Das Modell gibt die endgültigen Aktivierungen in dem Attribut "logits" aus. Wenden Sie die Softmax-Funktion auf die "logits" an, um die Wahrscheinlichkeiten zu erhalten:

```py
>>> import tensorflow as tf

>>> tf_predictions = tf.nn.softmax(tf_outputs.logits, axis=-1)
>>> tf_predictions  # doctest: +IGNORE_RESULT
```
</tf>
</frameworkcontent>

<Tip>

Alle 🤗 Transformers-Modelle (PyTorch oder TensorFlow) geben die Tensoren *vor* der endgültigen Aktivierungsfunktion
Funktion (wie Softmax) aus, da die endgültige Aktivierungsfunktion oft mit dem Verlusten verschmolzen ist.

</Tip>

Modelle sind ein standardmäßiges [`torch.nn.Module`](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) oder ein [`tf.keras.Model`](https://www.tensorflow.org/api_docs/python/tf/keras/Model), sodass Sie sie in Ihrer üblichen Trainingsschleife verwenden können. Um jedoch die Dinge einfacher zu machen, bietet 🤗 Transformers eine [`Trainer`]-Klasse für PyTorch, die Funktionalität für verteiltes Training, gemischte Präzision und mehr bietet. Für TensorFlow können Sie die Methode `fit` aus [Keras](https://keras.io/) verwenden. Siehe das [training tutorial](./training) für weitere Details.

<Tip>

Transformers-Modellausgaben sind spezielle Datenklassen, so dass ihre Attribute in einer IDE automatisch vervollständigt werden.
Die Modellausgänge verhalten sich auch wie ein Tupel oder ein Wörterbuch (z.B. können Sie mit einem Integer, einem Slice oder einem String indexieren), wobei die Attribute, die "None" sind, ignoriert werden.

</Tip>

### Modell speichern

<frameworkcontent>
<pt>
Sobald Ihr Modell feinabgestimmt ist, können Sie es mit seinem Tokenizer speichern, indem Sie [`PreTrainedModel.save_pretrained`] verwenden:

```py
>>> pt_save_directory = "./pt_save_pretrained"
>>> tokenizer.save_pretrained(pt_save_directory)  # doctest: +IGNORE_RESULT
>>> pt_model.save_pretrained(pt_save_directory)
```

Wenn Sie bereit sind, das Modell erneut zu verwenden, laden Sie es mit [`PreTrainedModel.from_pretrained`]:

```py
>>> pt_model = AutoModelForSequenceClassification.from_pretrained("./pt_save_pretrained")
```
</pt>
<tf>
Sobald Ihr Modell feinabgestimmt ist, können Sie es mit seinem Tokenizer unter Verwendung von [`TFPreTrainedModel.save_pretrained`] speichern:

```py
>>> tf_save_directory = "./tf_save_pretrained"
>>> tokenizer.save_pretrained(tf_save_directory)  # doctest: +IGNORE_RESULT
>>> tf_model.save_pretrained(tf_save_directory)
```

Wenn Sie bereit sind, das Modell wieder zu verwenden, laden Sie es mit [`TFPreTrainedModel.from_pretrained`]:

```py
>>> tf_model = TFAutoModelForSequenceClassification.from_pretrained("./tf_save_pretrained")
```
</tf>
</frameworkcontent>

Ein besonders cooles 🤗 Transformers-Feature ist die Möglichkeit, ein Modell zu speichern und es entweder als PyTorch- oder TensorFlow-Modell wieder zu laden. Der Parameter "from_pt" oder "from_tf" kann das Modell von einem Framework in das andere konvertieren:

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

## Custom model builds

Sie können die Konfigurationsklasse des Modells ändern, um zu bestimmen, wie ein Modell aufgebaut ist. Die Konfiguration legt die Attribute eines Modells fest, z. B. die Anzahl der verborgenen Schichten oder der Aufmerksamkeitsköpfe. Wenn Sie ein Modell aus einer benutzerdefinierten Konfigurationsklasse initialisieren, beginnen Sie bei Null. Die Modellattribute werden zufällig initialisiert, und Sie müssen das Modell trainieren, bevor Sie es verwenden können, um aussagekräftige Ergebnisse zu erhalten.

Beginnen Sie mit dem Import von [`AutoConfig`] und laden Sie dann das trainierte Modell, das Sie ändern möchten. Innerhalb von [`AutoConfig.from_pretrained`] können Sie das Attribut angeben, das Sie ändern möchten, z. B. die Anzahl der Aufmerksamkeitsköpfe:

```py
>>> from transformers import AutoConfig

>>> my_config = AutoConfig.from_pretrained("distilbert/distilbert-base-uncased", n_heads=12)
```

<frameworkcontent>
<pt>
Create a model from your custom configuration with [`AutoModel.from_config`]:

```py
>>> from transformers import AutoModel

>>> my_model = AutoModel.from_config(my_config)
```
</pt>
<tf>
Create a model from your custom configuration with [`TFAutoModel.from_config`]:

```py
>>> from transformers import TFAutoModel

>>> my_model = TFAutoModel.from_config(my_config)
```
</tf>
</frameworkcontent>

Weitere Informationen zur Erstellung von benutzerdefinierten Konfigurationen finden Sie in der Anleitung [Erstellen einer benutzerdefinierten Architektur](./create_a_model).

## Wie geht es weiter?

Nachdem Sie nun die 🤗 Transformers-Kurztour abgeschlossen haben, schauen Sie sich unsere Anleitungen an und erfahren Sie, wie Sie spezifischere Dinge tun können, wie das Schreiben eines benutzerdefinierten Modells, die Feinabstimmung eines Modells für eine Aufgabe und wie man ein Modell mit einem Skript trainiert. Wenn Sie mehr über die Kernkonzepte von 🤗 Transformers erfahren möchten, nehmen Sie sich eine Tasse Kaffee und werfen Sie einen Blick auf unsere konzeptionellen Leitfäden!
