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
    <br>
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers_logo_name.png" width="400"/>
    <br>
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
        <b>Español</b> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ja.md">日本語</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_hd.md">हिन्दी</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ru.md">Русский</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_pt-br.md">Рortuguês</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_te.md">తెలుగు</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_fr.md">Français</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_de.md">Deutsch</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_vi.md">Tiếng Việt</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ar.md">العربية</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ur.md">اردو</a> |
    </p>
</h4>

<h3 align="center">
    <p>Lo último de Machine Learning para JAX, PyTorch y TensorFlow</p>
</h3>

<h3 align="center">
    <a href="https://hf.co/course"><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/course_banner.png"></a>
</h3>

🤗 Transformers aporta miles de modelos preentrenados para realizar tareas en diferentes modalidades como texto, visión, y audio.

Estos modelos pueden ser aplicados en:

* 📝 Texto, para tareas como clasificación de texto, extracción de información, responder preguntas, resumir, traducir, generación de texto, en más de 100 idiomas.
* 🖼️ Imágenes, para tareas como clasificación de imágenes, detección the objetos, y segmentación.
* 🗣️ Audio, para tareas como reconocimiento de voz y clasificación de audio.

Los modelos de Transformer también pueden realizar tareas en **muchas modalidades combinadas**, como responder preguntas, reconocimiento de carácteres ópticos,extracción de información de documentos escaneados, clasificación de video, y respuesta de preguntas visuales.

🤗 Transformers aporta APIs para descargar rápidamente y usar estos modelos preentrenados en un texto dado, afinarlos en tus propios sets de datos y compartirlos con la comunidad en nuestro [centro de modelos](https://huggingface.co/models). Al mismo tiempo, cada módulo de Python que define una arquitectura es completamente independiente y se puede modificar para permitir experimentos de investigación rápidos.

🤗 Transformers está respaldado por las tres bibliotecas de deep learning más populares — [Jax](https://jax.readthedocs.io/en/latest/), [PyTorch](https://pytorch.org/) y [TensorFlow](https://www.tensorflow.org/) — con una perfecta integración entre ellos. Es sencillo entrenar sus modelos con uno antes de cargarlos para la inferencia con el otro.

## Demostraciones en línea

Puedes probar la mayoría de nuestros modelos directamente en sus páginas desde el [centro de modelos](https://huggingface.co/models). También ofrecemos [alojamiento de modelos privados, control de versiones y una API de inferencia](https://huggingface.co/pricing) para modelos públicos y privados.

Aquí hay algunos ejemplos:

En procesamiento del lenguaje natural:
- [Terminación de palabras enmascaradas con BERT](https://huggingface.co/google-bert/bert-base-uncased?text=Paris+is+the+%5BMASK%5D+of+France)
- [Reconocimiento del nombre de la entidad con Electra](https://huggingface.co/dbmdz/electra-large-discriminator-finetuned-conll03-english?text=My+name+is+Sarah+and+I+live+in+London+city)
- [Generación de texto con GPT-2](https://huggingface.co/openai-community/gpt2?text=A+long+time+ago%2C+)
- [Inferencia del lenguaje natural con RoBERTa](https://huggingface.co/FacebookAI/roberta-large-mnli?text=The+dog+was+lost.+Nobody+lost+any+animal)
- [Resumen con BART](https://huggingface.co/facebook/bart-large-cnn?text=The+tower+is+324+metres+%281%2C063+ft%29+tall%2C+about+the+same+height+as+an+81-storey+building%2C+and+the+tallest+structure+in+Paris.+Its+base+is+square%2C+measuring+125+metres+%28410+ft%29+on+each+side.+During+its+construction%2C+the+Eiffel+Tower+surpassed+the+Washington+Monument+to+become+the+tallest+man-made+structure+in+the+world%2C+a+title+it+held+for+41+years+until+the+Chrysler+Building+in+New+York+City+was+finished+in+1930.+It+was+the+first+structure+to+reach+a+height+of+300+metres.+Due+to+the+addition+of+a+broadcasting+aerial+at+the+top+of+the+tower+in+1957%2C+it+is+now+taller+than+the+Chrysler+Building+by+5.2+metres+%2817+ft%29.+Excluding+transmitters%2C+the+Eiffel+Tower+is+the+second+tallest+free-standing+structure+in+France+after+the+Millau+Viaduct)
- [Responder a preguntas con DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased-distilled-squad?text=Which+name+is+also+used+to+describe+the+Amazon+rainforest+in+English%3F&context=The+Amazon+rainforest+%28Portuguese%3A+Floresta+Amaz%C3%B4nica+or+Amaz%C3%B4nia%3B+Spanish%3A+Selva+Amaz%C3%B3nica%2C+Amazon%C3%ADa+or+usually+Amazonia%3B+French%3A+For%C3%AAt+amazonienne%3B+Dutch%3A+Amazoneregenwoud%29%2C+also+known+in+English+as+Amazonia+or+the+Amazon+Jungle%2C+is+a+moist+broadleaf+forest+that+covers+most+of+the+Amazon+basin+of+South+America.+This+basin+encompasses+7%2C000%2C000+square+kilometres+%282%2C700%2C000+sq+mi%29%2C+of+which+5%2C500%2C000+square+kilometres+%282%2C100%2C000+sq+mi%29+are+covered+by+the+rainforest.+This+region+includes+territory+belonging+to+nine+nations.+The+majority+of+the+forest+is+contained+within+Brazil%2C+with+60%25+of+the+rainforest%2C+followed+by+Peru+with+13%25%2C+Colombia+with+10%25%2C+and+with+minor+amounts+in+Venezuela%2C+Ecuador%2C+Bolivia%2C+Guyana%2C+Suriname+and+French+Guiana.+States+or+departments+in+four+nations+contain+%22Amazonas%22+in+their+names.+The+Amazon+represents+over+half+of+the+planet%27s+remaining+rainforests%2C+and+comprises+the+largest+and+most+biodiverse+tract+of+tropical+rainforest+in+the+world%2C+with+an+estimated+390+billion+individual+trees+divided+into+16%2C000+species)
- [Traducción con T5](https://huggingface.co/google-t5/t5-base?text=My+name+is+Wolfgang+and+I+live+in+Berlin)

En visión de ordenador:
- [Clasificación de imágenes con ViT](https://huggingface.co/google/vit-base-patch16-224)
- [Detección de objetos con DETR](https://huggingface.co/facebook/detr-resnet-50)
- [Segmentación semántica con SegFormer](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512)
- [Segmentación panóptica con DETR](https://huggingface.co/facebook/detr-resnet-50-panoptic)
- [Segmentación Universal con OneFormer (Segmentación Semántica, de Instancia y Panóptica con un solo modelo)](https://huggingface.co/shi-labs/oneformer_ade20k_dinat_large)

En Audio:
- [Reconocimiento de voz automático con Wav2Vec2](https://huggingface.co/facebook/wav2vec2-base-960h)
- [Detección de palabras clave con Wav2Vec2](https://huggingface.co/superb/wav2vec2-base-superb-ks)

En tareas multimodales:
- [Respuesta visual a preguntas con ViLT](https://huggingface.co/dandelin/vilt-b32-finetuned-vqa)

**[Escribe con Transformer](https://transformer.huggingface.co)**, construido por el equipo de Hugging Face, es la demostración oficial de las capacidades de generación de texto de este repositorio.

## Si está buscando soporte personalizado del equipo de Hugging Face

<a target="_blank" href="https://huggingface.co/support">
    <img alt="HuggingFace Expert Acceleration Program" src="https://cdn-media.huggingface.co/marketing/transformers/new-support-improved.png" style="max-width: 600px; border: 1px solid #eee; border-radius: 4px; box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);">
</a><br>

## Tour rápido

Para usar inmediatamente un modelo en una entrada determinada (texto, imagen, audio, ...), proporcionamos la API de `pipeline`. Los pipelines agrupan un modelo previamente entrenado con el preprocesamiento que se usó durante el entrenamiento de ese modelo. Aquí se explica cómo usar rápidamente un pipeline para clasificar textos positivos frente a negativos:

```python
>>> from transformers import pipeline

# Allocate a pipeline for sentiment-analysis
>>> classifier = pipeline('sentiment-analysis')
>>> classifier('We are very happy to introduce pipeline to the transformers repository.')
[{'label': 'POSITIVE', 'score': 0.9996980428695679}]
```

La segunda línea de código descarga y almacena en caché el modelo previamente entrenado que usa la canalización, mientras que la tercera lo evalúa en el texto dado. Aquí la respuesta es "positiva" con una confianza del 99,97%.

Muchas tareas tienen un `pipeline` preentrenado listo para funcionar, en NLP pero también en visión por ordenador y habla. Por ejemplo, podemos extraer fácilmente los objetos detectados en una imagen:

``` python
>>> import requests
>>> from PIL import Image
>>> from transformers import pipeline

# Download an image with cute cats
>>> url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png"
>>> image_data = requests.get(url, stream=True).raw
>>> image = Image.open(image_data)

# Allocate a pipeline for object detection
>>> object_detector = pipeline('object_detection')
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

Aquí obtenemos una lista de objetos detectados en la imagen, con un cuadro que rodea el objeto y una puntuación de confianza. Aquí está la imagen original a la derecha, con las predicciones mostradas a la izquierda:

<h3 align="center">
    <a><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png" width="400"></a>
    <a><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample_post_processed.png" width="400"></a>
</h3>

Puedes obtener más información sobre las tareas admitidas por la API de `pipeline` en [este tutorial](https://huggingface.co/docs/transformers/task_summary).

Además de `pipeline`, para descargar y usar cualquiera de los modelos previamente entrenados en su tarea dada, todo lo que necesita son tres líneas de código. Aquí está la versión de PyTorch:
```python
>>> from transformers import AutoTokenizer, AutoModel

>>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
>>> model = AutoModel.from_pretrained("google-bert/bert-base-uncased")

>>> inputs = tokenizer("Hello world!", return_tensors="pt")
>>> outputs = model(**inputs)
```

Y aquí está el código equivalente para TensorFlow:
```python
>>> from transformers import AutoTokenizer, TFAutoModel

>>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
>>> model = TFAutoModel.from_pretrained("google-bert/bert-base-uncased")

>>> inputs = tokenizer("Hello world!", return_tensors="tf")
>>> outputs = model(**inputs)
```

El tokenizador es responsable de todo el preprocesamiento que espera el modelo preentrenado y se puede llamar directamente en una sola cadena (como en los ejemplos anteriores) o en una lista. Este dará como resultado un diccionario que puedes usar en el código descendente o simplemente pasarlo directamente a su modelo usando el operador de desempaquetado de argumento **.

El modelo en si es un [Pytorch `nn.Module`](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) normal o un [TensorFlow `tf.keras.Model`](https://www.tensorflow.org/api_docs/python/tf/keras/Model) (dependiendo De tu backend) que puedes usar de forma habitual. [Este tutorial](https://huggingface.co/docs/transformers/training) explica cómo integrar un modelo de este tipo en un ciclo de entrenamiento PyTorch o TensorFlow clásico, o como usar nuestra API `Trainer` para ajustar rápidamente un nuevo conjunto de datos.

## ¿Por qué debo usar transformers?

1. Modelos de última generación fáciles de usar:
    - Alto rendimiento en comprensión y generación de lenguaje natural, visión artificial y tareas de audio.
    - Baja barrera de entrada para educadores y profesionales.
    - Pocas abstracciones de cara al usuario con solo tres clases para aprender.
    - Una API unificada para usar todos nuestros modelos preentrenados.

1. Menores costes de cómputo, menor huella de carbono:
    - Los investigadores pueden compartir modelos entrenados en lugar de siempre volver a entrenar.
    - Los profesionales pueden reducir el tiempo de cómputo y los costos de producción.
    - Docenas de arquitecturas con más de 60 000 modelos preentrenados en todas las modalidades.

1. Elija el marco adecuado para cada parte de la vida útil de un modelo:
    - Entrene modelos de última generación en 3 líneas de código.
    - Mueva un solo modelo entre los marcos TF2.0/PyTorch/JAX a voluntad.
    - Elija sin problemas el marco adecuado para la formación, la evaluación y la producción.

1. Personalice fácilmente un modelo o un ejemplo según sus necesidades:
    - Proporcionamos ejemplos de cada arquitectura para reproducir los resultados publicados por sus autores originales..
    - Los internos del modelo están expuestos lo más consistentemente posible..
    - Los archivos modelo se pueden usar independientemente de la biblioteca para experimentos rápidos.

## ¿Por qué no debería usar transformers?

- Esta biblioteca no es una caja de herramientas modular de bloques de construcción para redes neuronales. El código en los archivos del modelo no se refactoriza con abstracciones adicionales a propósito, de modo que los investigadores puedan iterar rápidamente en cada uno de los modelos sin sumergirse en abstracciones/archivos adicionales.
- La API de entrenamiento no está diseñada para funcionar en ningún modelo, pero está optimizada para funcionar con los modelos proporcionados por la biblioteca. Para bucles genéricos de aprendizaje automático, debe usar otra biblioteca (posiblemente, [Accelerate](https://huggingface.co/docs/accelerate)).
- Si bien nos esforzamos por presentar tantos casos de uso como sea posible, los scripts en nuestra [carpeta de ejemplos](https://github.com/huggingface/transformers/tree/main/examples) son solo eso: ejemplos. Se espera que no funcionen de forma inmediata en su problema específico y que deba cambiar algunas líneas de código para adaptarlas a sus necesidades.

## Instalación

### Con pip

Este repositorio está probado en Python 3.9+, Flax 0.4.1+, PyTorch 2.0+ y TensorFlow 2.6+.

Deberías instalar 🤗 Transformers en un [entorno virtual](https://docs.python.org/3/library/venv.html). Si no estas familiarizado con los entornos virtuales de Python, consulta la [guía de usuario](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).

Primero, crea un entorno virtual con la versión de Python que vas a usar y actívalo.

Luego, deberás instalar al menos uno entre Flax, PyTorch o TensorFlow.
Por favor, ve a la [página de instalación de TensorFlow](https://www.tensorflow.org/install/), [página de instalación de PyTorch](https://pytorch.org/get-started/locally/#start-locally) y/o las páginas de instalación de [Flax](https://github.com/google/flax#quick-install) y [Jax](https://github.com/google/jax#installation) con respecto al comando de instalación específico para tu plataforma.

Cuando se ha instalado uno de esos backends, los 🤗 Transformers se pueden instalar usando pip de la siguiente manera:

```bash
pip install transformers
```

Si deseas jugar con los ejemplos o necesitas la última versión del código y no puedes esperar a una nueva versión, tienes que [instalar la librería de la fuente](https://huggingface.co/docs/transformers/installation#installing-from-source).

### Con conda

🤗 Transformers se puede instalar usando conda de la siguiente manera:

```shell script
conda install conda-forge::transformers
```

> **_NOTA:_** Instalar `transformers` desde el canal `huggingface` está obsoleto.

Sigue las páginas de instalación de Flax, PyTorch o TensorFlow para ver cómo instalarlos con conda.

> **_NOTA:_**  En Windows, es posible que se le pida que active el modo de desarrollador para beneficiarse del almacenamiento en caché. Si esta no es una opción para usted, háganoslo saber en [esta issue](https://github.com/huggingface/huggingface_hub/issues/1062).

## Arquitecturas modelo

**[Todos los puntos de control del modelo](https://huggingface.co/models)** aportados por 🤗 Transformers están perfectamente integrados desde huggingface.co [Centro de modelos](https://huggingface.co) donde son subidos directamente por los [usuarios](https://huggingface.co/users) y [organizaciones](https://huggingface.co/organizations).

Número actual de puntos de control: ![](https://img.shields.io/endpoint?url=https://huggingface.co/api/shields/models&color=brightgreen)

🤗 Transformers actualmente proporciona las siguientes arquitecturas: ver [aquí](https://huggingface.co/docs/transformers/model_summary) para un resumen de alto nivel de cada uno de ellas.

Para comprobar si cada modelo tiene una implementación en Flax, PyTorch o TensorFlow, o tiene un tokenizador asociado respaldado por la librería 🤗 Tokenizers, ve a [esta tabla](https://huggingface.co/docs/transformers/index#supported-frameworks).

Estas implementaciones se han probado en varios conjuntos de datos (consulte los scripts de ejemplo) y deberían coincidir con el rendimiento de las implementaciones originales. Puede encontrar más detalles sobre el rendimiento en la sección Examples de la [documentación](https://github.com/huggingface/transformers/tree/main/examples).


## Aprender más

| Sección | Descripción |
|-|-|
| [Documentación](https://huggingface.co/docs/transformers/) | Toda la documentación de la API y tutoriales |
| [Resumen de tareas](https://huggingface.co/docs/transformers/task_summary) | Tareas soportadas 🤗 Transformers |
| [Tutorial de preprocesamiento](https://huggingface.co/docs/transformers/preprocessing) | Usando la clase `Tokenizer` para preparar datos para los modelos |
| [Entrenamiento y puesta a punto](https://huggingface.co/docs/transformers/training) | Usando los modelos aportados por 🤗 Transformers en un bucle de entreno de PyTorch/TensorFlow y la API de `Trainer` |
| [Recorrido rápido: secuencias de comandos de ajuste/uso](https://github.com/huggingface/transformers/tree/main/examples) | Scripts de ejemplo para ajustar modelos en una amplia gama de tareas |
| [Compartir y subir modelos](https://huggingface.co/docs/transformers/model_sharing) | Carga y comparte tus modelos perfeccionados con la comunidad |
| [Migración](https://huggingface.co/docs/transformers/migration) | Migra a 🤗 Transformers desde `pytorch-transformers` o `pytorch-pretrained-bert` |

## Citación

Ahora nosotros tenemos un [paper](https://www.aclweb.org/anthology/2020.emnlp-demos.6/) que puedes citar para la librería de 🤗 Transformers:
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
