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

# Quick tour

[[open-in-colab]]

Entra in azione con 🤗 Transformers! Inizia utilizzando [`pipeline`] per un'inferenza veloce, carica un modello pre-allenato e un tokenizer con una [AutoClass](./model_doc/auto) per risolvere i tuoi compiti legati a testo, immagini o audio.

<Tip>

Tutti gli esempi di codice presenti in questa documentazione hanno un pulsante in alto a sinistra che permette di selezionare tra PyTorch e TensorFlow. Se
questo non è presente, ci si aspetta che il codice funzioni per entrambi i backend senza alcun cambiamento.

</Tip>

## Pipeline

[`pipeline`] è il modo più semplice per utilizzare un modello pre-allenato per un dato compito.

<Youtube id="tiZFewofSLM"/>

La [`pipeline`] supporta molti compiti comuni:

**Testo**:
* Analisi del Sentimento (Sentiment Analysis, in inglese): classifica la polarità di un testo dato.
* Generazione del Testo (Text Generation, in inglese): genera del testo a partire da un dato input.
* Riconoscimento di Entità (Name Entity Recognition o NER, in inglese): etichetta ogni parola con l'entità che questa rappresenta (persona, data, luogo, ecc.).
* Rispondere a Domande (Question answering, in inglese): estrae la risposta da un contesto, dato del contesto e una domanda.
* Riempimento di Maschere (Fill-mask, in inglese): riempie gli spazi mancanti in un testo che ha parole mascherate.
* Riassumere (Summarization, in inglese): genera una sintesi di una lunga sequenza di testo o di un documento.
* Traduzione (Translation, in inglese): traduce un testo in un'altra lingua.
* Estrazione di Caratteristiche (Feature Extraction, in inglese): crea un tensore che rappresenta un testo.

**Immagini**:
* Classificazione di Immagini (Image Classification, in inglese): classifica un'immagine.
* Segmentazione di Immagini (Image Segmentation, in inglese): classifica ogni pixel di un'immagine.
* Rilevazione di Oggetti (Object Detection, in inglese): rileva oggetti all'interno di un'immagine.

**Audio**:
* Classificazione di Audio (Audio Classification, in inglese): assegna un'etichetta ad un segmento di audio dato.
* Riconoscimento Vocale Automatico (Automatic Speech Recognition o ASR, in inglese): trascrive il contenuto di un audio dato in un testo.

<Tip>

Per maggiori dettagli legati alla [`pipeline`] e ai compiti ad essa associati, fai riferimento alla documentazione [qui](./main_classes/pipelines).

</Tip>

### Utilizzo della Pipeline

Nel seguente esempio, utilizzerai la [`pipeline`] per l'analisi del sentimento.

Installa le seguenti dipendenze se non lo hai già fatto:

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

Importa [`pipeline`] e specifica il compito che vuoi completare:

```py
>>> from transformers import pipeline

>>> classificatore = pipeline("sentiment-analysis", model="MilaNLProc/feel-it-italian-sentiment")
```

La pipeline scarica e salva il [modello pre-allenato](https://huggingface.co/MilaNLProc/feel-it-italian-sentiment) e il tokenizer per l'analisi del sentimento. Se non avessimo scelto un modello, la pipeline ne avrebbe scelto uno di default. Ora puoi utilizzare il `classifier` sul tuo testo obiettivo:

```py
>>> classificatore("Siamo molto felici di mostrarti la libreria 🤗 Transformers.")
[{'label': 'positive', 'score': 0.9997}]
```

Per più di una frase, passa una lista di frasi alla [`pipeline`] la quale restituirà una lista di dizionari:

```py
>>> risultati = classificatore(
...     ["Siamo molto felici di mostrarti la libreria 🤗 Transformers.", "Speriamo te non la odierai."]
... )
>>> for risultato in risultati:
...     print(f"etichetta: {risultato['label']}, con punteggio: {round(risultato['score'], 4)}")
etichetta: positive, con punteggio: 0.9998
etichetta: negative, con punteggio: 0.9998
```

La [`pipeline`] può anche iterare su un dataset intero. Inizia installando la libreria [🤗 Datasets](https://huggingface.co/docs/datasets/):

```bash
pip install datasets
```

Crea una [`pipeline`] con il compito che vuoi risolvere e con il modello che vuoi utilizzare.

```py
>>> import torch
>>> from transformers import pipeline

>>> riconoscitore_vocale = pipeline(
...     "automatic-speech-recognition", model="radiogroup-crits/wav2vec2-xls-r-1b-italian-doc4lm-5gram"
... )
```

Poi, carica un dataset (vedi 🤗 Datasets [Quick Start](https://huggingface.co/docs/datasets/quickstart) per maggiori dettagli) sul quale vuoi iterare. Per esempio, carichiamo il dataset [MInDS-14](https://huggingface.co/datasets/PolyAI/minds14):

```py
>>> from datasets import load_dataset, Audio

>>> dataset = load_dataset("PolyAI/minds14", name="it-IT", split="train")  # doctest: +IGNORE_RESULT
```

Dobbiamo assicurarci che la frequenza di campionamento del set di dati corrisponda alla frequenza di campionamento con cui è stato addestrato `radiogroup-crits/wav2vec2-xls-r-1b-italian-doc4lm-5gram`.

```py
>>> dataset = dataset.cast_column("audio", Audio(sampling_rate=riconoscitore_vocale.feature_extractor.sampling_rate))
```

I file audio vengono caricati automaticamente e ri-campionati quando chiamiamo la colonna "audio".
Estraiamo i vettori delle forme d'onda grezze delle prime 4 osservazioni e passiamoli come lista alla pipeline:

```py
>>> risultato = riconoscitore_vocale(dataset[:4]["audio"])
>>> print([d["text"] for d in risultato])
['dovrei caricare dei soldi sul mio conto corrente', 'buongiorno e senza vorrei depositare denaro sul mio conto corrente come devo fare per cortesia', 'sì salve vorrei depositare del denaro sul mio conto', 'e buon pomeriggio vorrei depositare dei soldi sul mio conto bancario volleo sapere come posso fare se e posso farlo online ed un altro conto o andandoo tramite bancomut']
```

Per un dataset più grande dove gli input sono di dimensione maggiore (come nel parlato/audio o nella visione), dovrai passare un generatore al posto di una lista che carica tutti gli input in memoria. Guarda la [documentazione della pipeline](./main_classes/pipelines) per maggiori informazioni.

### Utilizzare un altro modello e tokenizer nella pipeline

La [`pipeline`] può ospitare qualsiasi modello del [Model Hub](https://huggingface.co/models), rendendo semplice l'adattamento della [`pipeline`] per altri casi d'uso. Per esempio, se si vuole un modello capace di trattare testo in francese, usa i tag presenti nel Model Hub in modo da filtrare per ottenere un modello appropriato. Il miglior risultato filtrato restituisce un modello multi-lingua [BERT model](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment) fine-tuned per l'analisi del sentimento. Ottimo, utilizziamo questo modello!

```py
>>> model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
```

<frameworkcontent>
<pt>
Usa [`AutoModelForSequenceClassification`] e [`AutoTokenizer`] per caricare il modello pre-allenato e il suo tokenizer associato (maggiori informazioni su una `AutoClass` in seguito):

```py
>>> from transformers import AutoTokenizer, AutoModelForSequenceClassification

>>> model = AutoModelForSequenceClassification.from_pretrained(model_name)
>>> tokenizer = AutoTokenizer.from_pretrained(model_name)
```
</pt>
<tf>
Usa [`TFAutoModelForSequenceClassification`] e [`AutoTokenizer`] per caricare il modello pre-allenato e il suo tokenizer associato (maggiori informazioni su una `TFAutoClass` in seguito):

```py
>>> from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

>>> model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
>>> tokenizer = AutoTokenizer.from_pretrained(model_name)
```
</tf>
</frameworkcontent>

Poi puoi specificare il modello e il tokenizer nella [`pipeline`], e applicare il `classifier` sul tuo testo obiettivo:

```py
>>> classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
>>> classifier("Nous sommes très heureux de vous présenter la bibliothèque 🤗 Transformers.")
[{'label': '5 stars', 'score': 0.7273}]
```

Se non riesci a trovare un modello per il tuo caso d'uso, dovrai fare fine-tuning di un modello pre-allenato sui tuoi dati. Dai un'occhiata al nostro tutorial [fine-tuning tutorial](./training) per imparare come. Infine, dopo che hai completato il fine-tuning del tuo modello pre-allenato, considera per favore di condividerlo (vedi il tutorial [qui](./model_sharing)) con la comunità sul Model Hub per democratizzare l'NLP! 🤗

## AutoClass

<Youtube id="AhChOFRegn4"/>

Al suo interno, le classi [`AutoModelForSequenceClassification`] e [`AutoTokenizer`] lavorano assieme per dare potere alla [`pipeline`]. Una [AutoClass](./model_doc/auto) è una scorciatoia che automaticamente recupera l'architettura di un modello pre-allenato a partire dal suo nome o path. Hai solo bisogno di selezionare la `AutoClass` appropriata per il tuo compito e il suo tokenizer associato con [`AutoTokenizer`].

Ritorniamo al nostro esempio e vediamo come puoi utilizzare la `AutoClass` per replicare i risultati della [`pipeline`].

### AutoTokenizer

Un tokenizer è responsabile dell'elaborazione del testo in modo da trasformarlo in un formato comprensibile dal modello. Per prima cosa, il tokenizer dividerà il testo in parole chiamate *token*. Ci sono diverse regole che governano il processo di tokenizzazione, tra cui come dividere una parola e a quale livello (impara di più sulla tokenizzazione [qui](./tokenizer_summary)). La cosa più importante da ricordare comunque è che hai bisogno di inizializzare il tokenizer con lo stesso nome del modello in modo da assicurarti che stai utilizzando le stesse regole di tokenizzazione con cui il modello è stato pre-allenato.

Carica un tokenizer con [`AutoTokenizer`]:

```py
>>> from transformers import AutoTokenizer

>>> nome_del_modello = "nlptown/bert-base-multilingual-uncased-sentiment"
>>> tokenizer = AutoTokenizer.from_pretrained(nome_del_modello)
```

Dopodiché, il tokenizer converte i token in numeri in modo da costruire un tensore come input del modello. Questo è conosciuto come il *vocabolario* del modello.

Passa il tuo testo al tokenizer:

```py
>>> encoding = tokenizer("Siamo molto felici di mostrarti la libreria 🤗 Transformers.")
>>> print(encoding)
{'input_ids': [101, 56821, 10132, 14407, 13019, 13007, 10120, 47201, 10330, 10106, 91686, 100, 58263, 119, 102],
'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
```

Il tokenizer restituirà un dizionario contenente:

* [input_ids](./glossary#input-ids): rappresentazioni numeriche dei tuoi token.
* [attention_mask](.glossary#attention-mask): indica quali token devono essere presi in considerazione.

Come con la [`pipeline`], il tokenizer accetterà una lista di input. In più, il tokenizer può anche completare (pad, in inglese) e troncare il testo in modo da restituire un lotto (batch, in inglese) di lunghezza uniforme:

<frameworkcontent>
<pt>
```py
>>> pt_batch = tokenizer(
...     ["Siamo molto felici di mostrarti la libreria 🤗 Transformers.", "Speriamo te non la odierai."],
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
...     ["Siamo molto felici di mostrarti la libreria 🤗 Transformers.", "Speriamo te non la odierai."],
...     padding=True,
...     truncation=True,
...     max_length=512,
...     return_tensors="tf",
... )
```
</tf>
</frameworkcontent>

Leggi il tutorial sul [preprocessing](./preprocessing) per maggiori dettagli sulla tokenizzazione.

### AutoModel

<frameworkcontent>
<pt>
🤗 Transformers fornisce un metodo semplice e unificato per caricare istanze pre-allenate. Questo significa che puoi caricare un [`AutoModel`] come caricheresti un [`AutoTokenizer`]. L'unica differenza è selezionare l'[`AutoModel`] corretto per il compito di interesse. Dato che stai facendo classificazione di testi, o sequenze, carica [`AutoModelForSequenceClassification`]:

```py
>>> from transformers import AutoModelForSequenceClassification

>>> model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
>>> pt_model = AutoModelForSequenceClassification.from_pretrained(model_name)
```

<Tip>

Guarda il [task summary](./task_summary) per sapere quale classe di [`AutoModel`] utilizzare per quale compito.

</Tip>

Ora puoi passare il tuo lotto di input pre-processati direttamente al modello. Devi solo spacchettare il dizionario aggiungendo `**`:

```py
>>> pt_outputs = pt_model(**pt_batch)
```

Il modello produrrà le attivazioni finali nell'attributo `logits`. Applica la funzione softmax a `logits` per ottenere le probabilità:

```py
>>> from torch import nn

>>> pt_predictions = nn.functional.softmax(pt_outputs.logits, dim=-1)
>>> print(pt_predictions)
tensor([[0.0041, 0.0037, 0.0203, 0.2005, 0.7713],
        [0.3766, 0.3292, 0.1832, 0.0558, 0.0552]], grad_fn=<SoftmaxBackward0>)
```
</pt>
<tf>
🤗 Transformers fornisce un metodo semplice e unificato per caricare istanze pre-allenate. Questo significa che puoi caricare un [`TFAutoModel`] come caricheresti un [`AutoTokenizer`]. L'unica differenza è selezionare il [`TFAutoModel`] corretto per il compito di interesse. Dato che stai facendo classificazione di testi, o sequenze, carica [`TFAutoModelForSequenceClassification`]:

```py
>>> from transformers import TFAutoModelForSequenceClassification

>>> nome_del_modello = "nlptown/bert-base-multilingual-uncased-sentiment"
>>> tf_model = TFAutoModelForSequenceClassification.from_pretrained(nome_del_modello)
```

<Tip>

Guarda il [task summary](./task_summary) per sapere quale classe di [`AutoModel`] utilizzare per quale compito.

</Tip>

Ora puoi passare il tuo lotto di input pre-processati direttamente al modello passando le chiavi del dizionario al tensore:

```py
>>> tf_outputs = tf_model(tf_batch)
```

Il modello produrrà le attivazioni finali nell'attributo `logits`. Applica la funzione softmax a `logits` per ottenere le probabilità:
```py
>>> import tensorflow as tf

>>> tf_predictions = tf.nn.softmax(tf_outputs.logits, axis=-1)
>>> tf_predictions  # doctest: +IGNORE_RESULT
```
</tf>
</frameworkcontent>

<Tip>

Tutti i modelli di 🤗 Transformers (PyTorch e TensorFlow) restituiscono i tensori *prima* della funzione finale
di attivazione (come la softmax) perché la funzione di attivazione finale viene spesso unita a quella di perdita.

</Tip>

I modelli sono [`torch.nn.Module`](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) o [`tf.keras.Model`](https://www.tensorflow.org/api_docs/python/tf/keras/Model) standard così puoi utilizzarli all'interno del tuo training loop usuale. Tuttavia, per rendere le cose più semplici, 🤗 Transformers fornisce una classe [`Trainer`] per PyTorch che aggiunge delle funzionalità per l'allenamento distribuito, precisione mista, e altro ancora. Per TensorFlow, puoi utilizzare il metodo `fit` di [Keras](https://keras.io/). Fai riferimento al [tutorial per il training](./training) per maggiori dettagli.

<Tip>

Gli output del modello di 🤗 Transformers sono delle dataclasses speciali in modo che i loro attributi vengano auto-completati all'interno di un IDE.
Gli output del modello si comportano anche come una tupla o un dizionario (ad esempio, puoi indicizzare con un intero, una slice o una stringa) nel qual caso gli attributi che sono `None` vengono ignorati.

</Tip>

### Salva un modello

<frameworkcontent>
<pt>
Una volta completato il fine-tuning del tuo modello, puoi salvarlo con il suo tokenizer utilizzando [`PreTrainedModel.save_pretrained`]:

```py
>>> pt_save_directory = "./pt_save_pretrained"
>>> tokenizer.save_pretrained(pt_save_directory)  # doctest: +IGNORE_RESULT
>>> pt_model.save_pretrained(pt_save_directory)
```

Quando desideri utilizzare il tuo modello nuovamente, puoi ri-caricarlo con [`PreTrainedModel.from_pretrained`]:

```py
>>> pt_model = AutoModelForSequenceClassification.from_pretrained("./pt_save_pretrained")
```
</pt>
<tf>
Una volta completato il fine-tuning del tuo modello, puoi salvarlo con il suo tokenizer utilizzando [`TFPreTrainedModel.save_pretrained`]:

```py
>>> tf_save_directory = "./tf_save_pretrained"
>>> tokenizer.save_pretrained(tf_save_directory)  # doctest: +IGNORE_RESULT
>>> tf_model.save_pretrained(tf_save_directory)
```

Quando desideri utilizzare il tuo modello nuovamente, puoi ri-caricarlo con [`TFPreTrainedModel.from_pretrained`]:

```py
>>> tf_model = TFAutoModelForSequenceClassification.from_pretrained("./tf_save_pretrained")
```
</tf>
</frameworkcontent>

Una caratteristica particolarmente interessante di 🤗 Transformers è la sua abilità di salvare un modello e ri-caricarlo sia come modello di PyTorch che di TensorFlow. I parametri `from_pt` o `from_tf` possono convertire un modello da un framework all'altro:

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
