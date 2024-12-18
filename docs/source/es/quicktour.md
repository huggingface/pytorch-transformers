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

# Tour rápido

[[open-in-colab]]

¡Entra en marcha con los 🤗 Transformers! Comienza usando [`pipeline`] para una inferencia veloz, carga un modelo preentrenado y un tokenizador con una [AutoClass](./model_doc/auto) para resolver tu tarea de texto, visión o audio.

<Tip>

Todos los ejemplos de código presentados en la documentación tienen un botón arriba a la derecha para elegir si quieres ocultar o mostrar el código en Pytorch o TensorFlow.
Si no fuese así, se espera que el código funcione para ambos backends sin ningún cambio.

</Tip>

## Pipeline

[`pipeline`] es la forma más fácil de usar un modelo preentrenado para una tarea dada.

<Youtube id="tiZFewofSLM"/>

El [`pipeline`] soporta muchas tareas comunes listas para usar:

**Texto**:
* Análisis de Sentimiento (Sentiment Analysis, en inglés): clasifica la polaridad de un texto dado.
* Generación de Texto (Text Generation, en inglés): genera texto a partir de un input dado.
* Reconocimiento de Entidades (Name Entity Recognition o NER, en inglés): etiqueta cada palabra con la entidad que representa (persona, fecha, ubicación, etc.).
* Responder Preguntas (Question answering, en inglés): extrae la respuesta del contexto dado un contexto y una pregunta.
* Rellenar Máscara (Fill-mask, en inglés): rellena el espacio faltante dado un texto con palabras enmascaradas.
* Resumir (Summarization, en inglés): genera un resumen de una secuencia larga de texto o un documento.
* Traducción (Translation, en inglés): traduce un texto a otro idioma.
* Extracción de Características (Feature Extraction, en inglés): crea una representación tensorial del texto.

**Imagen**:
* Clasificación de Imágenes (Image Classification, en inglés): clasifica una imagen.
* Segmentación de Imágenes (Image Segmentation, en inglés): clasifica cada pixel de una imagen.
* Detección de Objetos (Object Detection, en inglés): detecta objetos dentro de una imagen.

**Audio**:
* Clasificación de Audios (Audio Classification, en inglés): asigna una etiqueta a un segmento de audio.
* Reconocimiento de Voz Automático (Automatic Speech Recognition o ASR, en inglés): transcribe datos de audio a un texto.

<Tip>

Para más detalles acerca del [`pipeline`] y tareas asociadas, consulta la documentación [aquí](./main_classes/pipelines).

</Tip>

### Uso del Pipeline

En el siguiente ejemplo, usarás el [`pipeline`] para análisis de sentimiento.

Instala las siguientes dependencias si aún no lo has hecho:

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

Importa [`pipeline`] y especifica la tarea que deseas completar:

```py
>>> from transformers import pipeline

>>> clasificador = pipeline("sentiment-analysis", model="pysentimiento/robertuito-sentiment-analysis")
```

El pipeline descarga y almacena en caché el [modelo preentrenado](https://huggingface.co/pysentimiento/robertuito-sentiment-analysis) y tokeniza para análisis de sentimiento. Si no hubieramos elegido un modelo el pipeline habría elegido uno por defecto. Ahora puedes usar `clasificador` en tu texto objetivo:

```py
>>> clasificador("Estamos muy felices de mostrarte la biblioteca de 🤗 Transformers.")
[{'label': 'POS', 'score': 0.9320}]
```

Para más de un enunciado, entrega una lista al [`pipeline`] que devolverá una lista de diccionarios:

El [`pipeline`] también puede iterar sobre un dataset entero. Comienza instalando la biblioteca [🤗 Datasets](https://huggingface.co/docs/datasets/):

```bash
pip install datasets
```

Crea un [`pipeline`] con la tarea que deseas resolver y el modelo que quieres usar. Coloca el parámetro `device` a `0` para poner los tensores en un dispositivo CUDA:

```py
>>> import torch
>>> from transformers import pipeline

>>> reconocedor_de_voz = pipeline(
...     "automatic-speech-recognition", model="jonatasgrosman/wav2vec2-large-xlsr-53-spanish", device=0
... )
```

A continuación, carga el dataset (ve 🤗 Datasets [Quick Start](https://huggingface.co/docs/datasets/quickstart.html) para más detalles) sobre el que quisieras iterar. Por ejemplo, vamos a cargar el dataset [MInDS-14](https://huggingface.co/datasets/PolyAI/minds14):

```py
>>> from datasets import load_dataset, Audio

>>> dataset = load_dataset("PolyAI/minds14", name="es-ES", split="train")  # doctest: +IGNORE_RESULT
```

Debemos asegurarnos de que la frecuencia de muestreo del conjunto de datos coincide con la frecuencia de muestreo con la que se entrenó `jonatasgrosman/wav2vec2-large-xlsr-53-spanish`.

```py
>>> dataset = dataset.cast_column("audio", Audio(sampling_rate=reconocedor_de_voz.feature_extractor.sampling_rate))
```

Los archivos de audio se cargan y remuestrean automáticamente cuando llamamos a la columna `"audio"`.
Extraigamos las matrices de onda cruda (raw waveform, en inglés) de las primeras 4 muestras y pasémosla como una lista al pipeline:

```py
>>> resultado = reconocedor_de_voz(dataset[:4]["audio"])
>>> print([d["text"] for d in resultado])
['ahora buenas eh a ver tengo un problema con vuestra aplicación resulta que que quiero hacer una transferencia bancaria a una cuenta conocida pero me da error la aplicación a ver que a ver que puede ser', 'la aplicación no cargue saldo de mi nueva cuenta', 'hola tengo un problema con la aplicación no carga y y tampoco veo que carga el saldo de mi cuenta nueva dice que la aplicación está siendo reparada y ahora no puedo acceder a mi cuenta no necesito inmediatamente', 'hora buena la aplicación no se carga la vida no carga el saldo de mi cuenta nueva dice que la villadenta siendo reparada y oro no puedo hacer a mi cuenta']
```

Para un dataset más grande, donde los inputs son de mayor tamaño (como en habla/audio o visión), querrás pasar un generador en lugar de una lista que carga todos los inputs en memoria. Ve la [documentación del pipeline](./main_classes/pipelines) para más información.

### Usa otro modelo y otro tokenizador en el pipeline

El [`pipeline`] puede acomodarse a cualquier modelo del [Model Hub](https://huggingface.co/models) haciendo más fácil adaptar el [`pipeline`] para otros casos de uso. Por ejemplo, si quisieras un modelo capaz de manejar texto en francés, usa los tags en el Model Hub para filtrar entre los modelos apropiados. El resultado mejor filtrado devuelve un [modelo BERT](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment) multilingual fine-tuned para el análisis de sentimiento. Genial, ¡vamos a usar este modelo!

```py
>>> model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
```

<frameworkcontent>
<pt>
Usa [`AutoModelForSequenceClassification`] y ['AutoTokenizer'] para cargar un modelo preentrenado y un tokenizador asociado (más en un `AutoClass` debajo):

```py
>>> from transformers import AutoTokenizer, AutoModelForSequenceClassification

>>> model = AutoModelForSequenceClassification.from_pretrained(model_name)
>>> tokenizer = AutoTokenizer.from_pretrained(model_name)
```

</pt>

<tf>
Usa [`TFAutoModelForSequenceClassification`] y ['AutoTokenizer'] para cargar un modelo preentrenado y un tokenizador asociado (más en un `TFAutoClass` debajo):

```py
>>> from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

>>> model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
>>> tokenizer = AutoTokenizer.from_pretrained(model_name)
```

</tf>
</frameworkcontent>

Después puedes especificar el modelo y el tokenizador en el [`pipeline`], y aplicar el `classifier` en tu texto objetivo:

```py
>>> classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
>>> classifier("Nous sommes très heureux de vous présenter la bibliothèque 🤗 Transformers.")
[{'label': '5 stars', 'score': 0.7273}]
```

Si no pudieras encontrar el modelo para tu caso respectivo de uso necesitarás ajustar un modelo preentrenado a tus datos. Mira nuestro [tutorial de fine-tuning](./training) para aprender cómo. Finalmente, después de que has ajustado tu modelo preentrenado, ¡por favor considera compartirlo (ve el tutorial [aquí](./model_sharing)) con la comunidad en el Model Hub para democratizar el NLP! 🤗

## AutoClass

<Youtube id="AhChOFRegn4"/>

Por debajo, las clases [`AutoModelForSequenceClassification`] y [`AutoTokenizer`] trabajan juntas para dar poder al [`pipeline`]. Una [AutoClass](./model_doc/auto) es un atajo que automáticamente recupera la arquitectura de un modelo preentrenado con su nombre o el path. Sólo necesitarás seleccionar el `AutoClass` apropiado para tu tarea y tu tokenizador asociado con [`AutoTokenizer`].

Regresemos a nuestro ejemplo y veamos cómo puedes usar el `AutoClass` para reproducir los resultados del [`pipeline`].

### AutoTokenizer

Un tokenizador es responsable de procesar el texto a un formato que sea entendible para el modelo. Primero, el tokenizador separará el texto en palabras llamadas *tokens*. Hay múltiples reglas que gobiernan el proceso de tokenización incluyendo el cómo separar una palabra y en qué nivel (aprende más sobre tokenización [aquí](./tokenizer_summary)). Lo más importante es recordar que necesitarás instanciar el tokenizador con el mismo nombre del modelo para asegurar que estás usando las mismas reglas de tokenización con las que el modelo fue preentrenado.

Carga un tokenizador con [`AutoTokenizer`]:

```py
>>> from transformers import AutoTokenizer

>>> nombre_del_modelo = "nlptown/bert-base-multilingual-uncased-sentiment"
>>> tokenizer = AutoTokenizer.from_pretrained(nombre_del_modelo)
```

Después, el tokenizador convierte los tokens a números para construir un tensor que servirá como input para el modelo. Esto es conocido como el *vocabulario* del modelo.

Pasa tu texto al tokenizador:

```py
>>> encoding = tokenizer("Estamos muy felices de mostrarte la biblioteca de 🤗 Transformers.")
>>> print(encoding)
{'input_ids': [101, 10602, 14000, 13653, 43353, 10107, 10102, 47201, 10218, 10106, 18283, 10102, 100, 58263, 119, 102],
 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
```

El tokenizador devolverá un diccionario conteniendo:

* [input_ids](./glossary#input-ids): representaciones numéricas de los tokens.
* [atttention_mask](.glossary#attention-mask): indica cuáles tokens deben ser atendidos.

Como con el [`pipeline`], el tokenizador aceptará una lista de inputs. Además, el tokenizador también puede rellenar (pad, en inglés) y truncar el texto para devolver un lote (batch, en inglés) de longitud uniforme:

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

Lee el tutorial de [preprocessing](./preprocessing) para más detalles acerca de la tokenización.

### AutoModel

<frameworkcontent>
<pt>
🤗 Transformers provee una forma simple y unificada de cargar tus instancias preentrenadas. Esto significa que puedes cargar un [`AutoModel`] como cargarías un [`AutoTokenizer`]. La única diferencia es seleccionar el [`AutoModel`] correcto para la tarea. Ya que estás clasificando texto, o secuencias, carga [`AutoModelForSequenceClassification`]:

```py
>>> from transformers import AutoModelForSequenceClassification

>>> model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
>>> pt_model = AutoModelForSequenceClassification.from_pretrained(model_name)
```

<Tip>

Ve el [task summary](./task_summary) para revisar qué clase del [`AutoModel`] deberías usar para cada tarea.

</Tip>

Ahora puedes pasar tu lote (batch) preprocesado de inputs directamente al modelo. Solo tienes que desempacar el diccionario añadiendo `**`:

```py
>>> pt_outputs = pt_model(**pt_batch)
```

El modelo producirá las activaciones finales en el atributo `logits`. Aplica la función softmax a `logits` para obtener las probabilidades:

```py
>>> from torch import nn

>>> pt_predictions = nn.functional.softmax(pt_outputs.logits, dim=-1)
>>> print(pt_predictions)
tensor([[0.0021, 0.0018, 0.0115, 0.2121, 0.7725],
        [0.2084, 0.1826, 0.1969, 0.1755, 0.2365]], grad_fn=<SoftmaxBackward0>)
```
</pt>
<tf>
🤗 Transformers provee una forma simple y unificada de cargar tus instancias preentrenadas. Esto significa que puedes cargar un [`TFAutoModel`] como cargarías un [`AutoTokenizer`]. La única diferencia es seleccionar el [`TFAutoModel`] correcto para la tarea. Ya que estás clasificando texto, o secuencias, carga [`TFAutoModelForSequenceClassification`]:

```py
>>> from transformers import TFAutoModelForSequenceClassification

>>> model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
>>> tf_model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
```

<Tip>
  Ve el [task summary](./task_summary) para revisar qué clase del [`AutoModel`]
  deberías usar para cada tarea.
</Tip>

Ahora puedes pasar tu lote preprocesado de inputs directamente al modelo pasando las llaves del diccionario directamente a los tensores:

```py
>>> tf_outputs = tf_model(tf_batch)
```

El modelo producirá las activaciones finales en el atributo `logits`. Aplica la función softmax a `logits` para obtener las probabilidades:

```py
>>> import tensorflow as tf

>>> tf_predictions = tf.nn.softmax(tf_outputs.logits, axis=-1)
>>> print(tf.math.round(tf_predictions * 10**4) / 10**4)
tf.Tensor(
[[0.0021 0.0018 0.0116 0.2121 0.7725]
 [0.2084 0.1826 0.1969 0.1755  0.2365]], shape=(2, 5), dtype=float32)
```
</tf>
</frameworkcontent>

<Tip>

Todos los modelos de 🤗 Transformers (PyTorch o TensorFlow) producirán los tensores *antes* de la función de activación
final (como softmax) porque la función de activación final es comúnmente fusionada con la pérdida.

</Tip>

Los modelos son [`torch.nn.Module`](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) o [`tf.keras.Model`](https://www.tensorflow.org/api_docs/python/tf/keras/Model) estándares así que podrás usarlos en tu training loop usual. Sin embargo, para facilitar las cosas, 🤗 Transformers provee una clase [`Trainer`] para PyTorch que añade funcionalidades para entrenamiento distribuido, precición mixta, y más. Para TensorFlow, puedes usar el método `fit` desde [Keras](https://keras.io/). Consulta el [tutorial de entrenamiento](./training) para más detalles.

<Tip>

Los outputs del modelo de 🤗 Transformers son dataclasses especiales por lo que sus atributos pueden ser completados en un IDE.
Los outputs del modelo también se comportan como tuplas o diccionarios (e.g., puedes indexar con un entero, un slice o una cadena) en cuyo caso los atributos que son `None` son ignorados.

</Tip>

### Guarda un modelo

<frameworkcontent>
<pt>
Una vez que se haya hecho fine-tuning a tu modelo puedes guardarlo con tu tokenizador usando [`PreTrainedModel.save_pretrained`]:

```py
>>> pt_save_directory = "./pt_save_pretrained"
>>> tokenizer.save_pretrained(pt_save_directory)  # doctest: +IGNORE_RESULT
>>> pt_model.save_pretrained(pt_save_directory)
```

Cuando quieras usar el modelo otra vez cárgalo con [`PreTrainedModel.from_pretrained`]:

```py
>>> pt_model = AutoModelForSequenceClassification.from_pretrained("./pt_save_pretrained")
```

</pt>

<tf>
Una vez que se haya hecho fine-tuning a tu modelo puedes guardarlo con tu tokenizador usando [`TFPreTrainedModel.save_pretrained`]:

```py
>>> tf_save_directory = "./tf_save_pretrained"
>>> tokenizer.save_pretrained(tf_save_directory)  # doctest: +IGNORE_RESULT
>>> tf_model.save_pretrained(tf_save_directory)
```

Cuando quieras usar el modelo otra vez cárgalo con [`TFPreTrainedModel.from_pretrained`]:

```py
>>> tf_model = TFAutoModelForSequenceClassification.from_pretrained("./tf_save_pretrained")
```
</tf>
</frameworkcontent>

Una característica particularmente interesante de 🤗 Transformers es la habilidad de guardar el modelo y cargarlo como un modelo de PyTorch o TensorFlow. El parámetro `from_pt` o `from_tf` puede convertir el modelo de un framework al otro:

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
