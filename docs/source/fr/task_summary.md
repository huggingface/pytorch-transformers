<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Ce que 🤗 Transformers peut faire

🤗 Transformers est une bibliothèque de modèles préentraînés à la pointe de la technologie pour les tâches de traitement du langage naturel (NLP), de vision par ordinateur et de traitement audio et de la parole. Non seulement la bibliothèque contient des modèles Transformer, mais elle inclut également des modèles non-Transformer comme des réseaux convolutionnels modernes pour les tâches de vision par ordinateur. Si vous regardez certains des produits grand public les plus populaires aujourd'hui, comme les smartphones, les applications et les téléviseurs, il est probable qu'une technologie d'apprentissage profond soit derrière. Vous souhaitez supprimer un objet de fond d'une photo prise avec votre smartphone ? C'est un exemple de tâche de segmentation panoptique (ne vous inquiétez pas si vous ne savez pas encore ce que cela signifie, nous le décrirons dans les sections suivantes !).

Cette page fournit un aperçu des différentes tâches de traitement de la parole et de l'audio, de vision par ordinateur et de NLP qui peuvent être résolues avec la bibliothèque 🤗 Transformers en seulement trois lignes de code !

## Audio

Les tâches de traitement audio et de la parole sont légèrement différentes des autres modalités principalement parce que l'audio en tant que donnée d'entrée est un signal continu. Contrairement au texte, un signal audio brut ne peut pas discrétisé de la manière dont une phrase peut être divisée en mots. Pour contourner cela, le signal audio brut est généralement échantillonné à intervalles réguliers. Si vous prenez plus d'échantillons dans un intervalle, le taux d'échantillonnage est plus élevé et l'audio ressemble davantage à la source audio originale.

Les approches précédentes prétraitaient l'audio pour en extraire des caractéristiques utiles. Il est maintenant plus courant de commencer les tâches de traitement audio et de la parole en donnant directement le signal audio brut à un encodeur de caractéristiques (*feature encoder* en anglais) pour extraire une représentation de l'audio. Cela correspond à l'étape de prétraitement et permet au modèle d'apprendre les caractéristiques les plus essentielles du signal.

### Classification audio

La classification audio est une tâche qui consiste à attribuer une classe, parmi un ensemble de classes prédéfini, à un audio. La classification audio englobe de nombreuses applications spécifiques, dont certaines incluent :

* la classification d'environnements sonores : attribuer une classe (catégorie) à l'audio pour indiquer l'environnement associé, tel que "bureau", "plage" ou "stade". 
* la détection d'événements sonores : étiqueter l'audio avec une étiquette d'événement sonore ("klaxon de voiture", "appel de baleine", "verre brisé")
* l'identification d'éléments sonores : attribuer des tags (*étiquettes* en français) à l'audio pour marquer des sons spécifiques, comme "chant des oiseaux" ou "identification du locuteur lors d'une réunion".
* la classification musicale : attribuer un genre à la musique, comme "metal", "hip-hop" ou "country".

```py
>>> from transformers import pipeline

>>> classifier = pipeline(task="audio-classification", model="superb/hubert-base-superb-er")
>>> preds = classifier("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
>>> preds = [{"score": round(pred["score"], 4), "label": pred["label"]} for pred in preds]
>>> preds
[{'score': 0.4532, 'label': 'hap'},
 {'score': 0.3622, 'label': 'sad'},
 {'score': 0.0943, 'label': 'neu'},
 {'score': 0.0903, 'label': 'ang'}]
```

### Reconnaissance vocale

La reconnaissance vocale (*Automatic Speech Recognition* ou ASR en anglais) transcrit la parole en texte. C'est l'une des tâches audio les plus courantes en partie parce que la parole est une forme de communication la plus naturelle pour nous, humains. Aujourd'hui, les systèmes ASR sont intégrés dans des produits technologiques "intelligents" comme les enceintes, les téléphones et les voitures. Il est désormais possible de demander à nos assistants virtuels de jouer de la musique, de définir des rappels et de nous indiquer la météo.

Mais l'un des principaux défis auxquels les architectures Transformer contribuent à résoudre est celui des langues à faibles ressources, c'est-à-dire des langues pour lesquelles il existe peu de données étiquetées. En préentraînant sur de grandes quantités de données vocales d'un autre language plus ou moins similaire, le réglage fin (*fine-tuning* en anglais) du modèle avec seulement une heure de données vocales étiquetées dans une langue à faibles ressources peut tout de même produire des résultats de haute qualité comparés aux systèmes ASR précédents entraînés sur 100 fois plus de données étiquetées.

```py
>>> from transformers import pipeline

>>> transcriber = pipeline(task="automatic-speech-recognition", model="openai/whisper-small")
>>> transcriber("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
{'text': ' I have a dream that one day this nation will rise up and live out the true meaning of its creed.'}
```

## Vision par ordinateur

L'une des premières réussites en vision par ordinateur a été la reconnaissance des numéros de code postal à l'aide d'un [réseau de neurones convolutionnel (CNN)](glossary#convolution). Une image est composée de pixels, chacun ayant une valeur numérique, ce qui permet de représenter facilement une image sous forme de matrice de valeurs de pixels. Chaque combinaison de valeurs de pixels correspond aux couleurs d'une image.

Il existe deux approches principales pour résoudre les tâches de vision par ordinateur :

1. Utiliser des convolutions pour apprendre les caractéristiques hiérarchiques d'une image, des détails de bas niveau aux éléments abstraits de plus haut niveau.
2. Diviser l'image en morceaux (*patches* en anglais) et utiliser un Transformer pour apprendre progressivement comment chaque morceau est lié aux autres pour former l'image complète. Contrairement à l'approche ascendante des CNNs, cette méthode ressemble à un processus où l'on démarre avec une image floue pour ensuite la mettre au point petit à petit. 

### Classification d'images

La classification d'images consiste à attribuer une classe, parmi un ensemble de classes prédéfini, à toute une image. Comme pour la plupart des tâches de classification, les cas d'utilisation pratiques sont nombreux, notamment :

- Santé : classification d'images médicales pour détecter des maladies ou surveiller l'état de santé des patients.
- Environnement : classification d'images satellites pour suivre la déforestation, aider à la gestion des terres ou détecter les incendies de forêt.
- Agriculture : classification d'images de cultures pour surveiller la santé des plantes ou des images satellites pour analyser l'utilisation des terres.
- Écologie : classification d'images d'espèces animales ou végétales pour suivre les populations fauniques ou les espèces menacées.

```py
>>> from transformers import pipeline

>>> classifier = pipeline(task="image-classification")
>>> preds = classifier(
...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
... )
>>> preds = [{"score": round(pred["score"], 4), "label": pred["label"]} for pred in preds]
>>> print(*preds, sep="\n")
{'score': 0.4335, 'label': 'lynx, catamount'}
{'score': 0.0348, 'label': 'cougar, puma, catamount, mountain lion, painter, panther, Felis concolor'}
{'score': 0.0324, 'label': 'snow leopard, ounce, Panthera uncia'}
{'score': 0.0239, 'label': 'Egyptian cat'}
{'score': 0.0229, 'label': 'tiger cat'}
```

### Détection d'objets

La détection d'objets, à la différence de la classification d'images, identifie plusieurs objets dans une image ainsi que leurs positions, généralement définies par des boîtes englobantes (*bounding boxes* en anglais). Voici quelques exemples d'applications :

- Véhicules autonomes : détection des objets de la circulation, tels que les véhicules, piétons et feux de signalisation.
- Télédétection : surveillance des catastrophes, planification urbaine et prévisions météorologiques.
- Détection de défauts : identification des fissures ou dommages structurels dans les bâtiments, ainsi que des défauts de fabrication.

```py
>>> from transformers import pipeline

>>> detector = pipeline(task="object-detection")
>>> preds = detector(
...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
... )
>>> preds = [{"score": round(pred["score"], 4), "label": pred["label"], "box": pred["box"]} for pred in preds]
>>> preds
[{'score': 0.9865,
  'label': 'cat',
  'box': {'xmin': 178, 'ymin': 154, 'xmax': 882, 'ymax': 598}}]
```

### Segmentation d'images

La segmentation d'images est une tâche qui consiste à attribuer une classe à chaque pixel d'une image, ce qui la rend plus précise que la détection d'objets, qui se limite aux boîtes englobantes (*bounding boxes* en anglais). Elle permet ainsi de détecter les objets à la précision du pixel. Il existe plusieurs types de segmentation d'images :

- Segmentation d'instances : en plus de classifier un objet, elle identifie chaque instance distincte d'un même objet (par exemple, "chien-1", "chien-2").
- Segmentation panoptique : combine segmentation sémantique et segmentation d'instances, attribuant à chaque pixel une classe sémantique **et** une instance spécifique.

Ces techniques sont utiles pour les véhicules autonomes, qui doivent cartographier leur environnement pixel par pixel pour naviguer en toute sécurité autour des piétons et des véhicules. Elles sont également précieuses en imagerie médicale, où la précision au niveau des pixels permet de détecter des anomalies cellulaires ou des caractéristiques d'organes. Dans le commerce en ligne, la segmentation est utilisée pour des essayages virtuels de vêtements ou des expériences de réalité augmentée, en superposant des objets virtuels sur des images du monde réel via la caméra.

```py
>>> from transformers import pipeline

>>> segmenter = pipeline(task="image-segmentation")
>>> preds = segmenter(
...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
... )
>>> preds = [{"score": round(pred["score"], 4), "label": pred["label"]} for pred in preds]
>>> print(*preds, sep="\n")
{'score': 0.9879, 'label': 'LABEL_184'}
{'score': 0.9973, 'label': 'snow'}
{'score': 0.9972, 'label': 'cat'}
```

### Estimation de la profondeur

L'estimation de la profondeur consiste à prédire la distance de chaque pixel d'une image par rapport à la caméra. Cette tâche est cruciale pour comprendre et reconstruire des scènes réelles. Par exemple, pour les voitures autonomes, il est essentiel de déterminer la distance des objets tels que les piétons, les panneaux de signalisation et les autres véhicules pour éviter les collisions. L'estimation de la profondeur permet également de créer des modèles 3D à partir d'images 2D, ce qui est utile pour générer des représentations détaillées de structures biologiques ou de bâtiments.

Il existe deux principales approches pour estimer la profondeur :

- Stéréo : la profondeur est estimée en comparant deux images d'une même scène prises sous des angles légèrement différents.
- Monoculaire : la profondeur est estimée à partir d'une seule image.

```py
>>> from transformers import pipeline

>>> depth_estimator = pipeline(task="depth-estimation")
>>> preds = depth_estimator(
...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
... )
```

## Traitement du langage naturel

Les tâches de traitement du langage naturel (*Natural Language Processing* ou *NLP* en anglais) sont courantes car le texte est une forme naturelle de communication pour nous. Pour qu'un modèle puisse traiter le texte, celui-ci doit être *tokenisé*, c'est-à-dire divisé en mots ou sous-mots appelés "*tokens*", puis converti en nombres. Ainsi, une séquence de texte peut être représentée comme une séquence de nombres, qui peut ensuite être utilisée comme données d'entrée pour un modèle afin de résoudre diverses tâches  de traitement du langage naturel.

### Classification de texte

La classification de texte attribue une classe à une séquence de texte (au niveau d'une phrase, d'un paragraphe ou d'un document) à partir d'un ensemble de classes prédéfini. Voici quelques applications pratiques :

- **Analyse des sentiments** : étiqueter le texte avec une polarité telle que `positive` ou `négative`, ce qui aide à la prise de décision dans des domaines comme la politique, la finance et le marketing.
- **Classification de contenu** : organiser et filtrer les informations en attribuant des *tags* sur des sujets spécifiques, comme `météo`, `sports` ou `finance`, dans les flux d'actualités et les réseaux sociaux.

```py
>>> from transformers import pipeline

>>> classifier = pipeline(task="sentiment-analysis")
>>> preds = classifier("Hugging Face is the best thing since sliced bread!")
>>> preds = [{"score": round(pred["score"], 4), "label": pred["label"]} for pred in preds]
>>> preds
[{'score': 0.9991, 'label': 'POSITIVE'}]
```

### Classification des tokens

Dans les tâches de traitement du language naturel, le texte est d'abord prétraité en le séparant en mots ou sous-mots individuels, appelés *[tokens](glossary#token)*. La classification des tokens attribue une classe à chaque token à partir d'un ensemble de classes prédéfini.

Voici deux types courants de classification des tokens :

- **Reconnaissance d'entités nommées (*Named Entity Recognition* ou *NER* en anglais)** : étiqueter un token selon une catégorie d'entité, telle qu'organisation, personne, lieu ou date. La NER est particulièrement utilisée dans les contextes biomédicaux pour identifier des gènes, des protéines et des noms de médicaments.
- **Étiquetage des parties du discours (*Part of Speech* ou *POS* en anglais)** : étiqueter un token en fonction de sa partie du discours, comme nom, verbe ou adjectif. Le POS est utile pour les systèmes de traduction afin de comprendre comment deux mots identiques peuvent avoir des rôles grammaticaux différents (par exemple, "banque" comme nom versus "banque" comme verbe).

```py
>>> from transformers import pipeline

>>> classifier = pipeline(task="ner")
>>> preds = classifier("Hugging Face is a French company based in New York City.")
>>> preds = [
...     {
...         "entity": pred["entity"],
...         "score": round(pred["score"], 4),
...         "index": pred["index"],
...         "word": pred["word"],
...         "start": pred["start"],
...         "end": pred["end"],
...     }
...     for pred in preds
... ]
>>> print(*preds, sep="\n")
{'entity': 'I-ORG', 'score': 0.9968, 'index': 1, 'word': 'Hu', 'start': 0, 'end': 2}
{'entity': 'I-ORG', 'score': 0.9293, 'index': 2, 'word': '##gging', 'start': 2, 'end': 7}
{'entity': 'I-ORG', 'score': 0.9763, 'index': 3, 'word': 'Face', 'start': 8, 'end': 12}
{'entity': 'I-MISC', 'score': 0.9983, 'index': 6, 'word': 'French', 'start': 18, 'end': 24}
{'entity': 'I-LOC', 'score': 0.999, 'index': 10, 'word': 'New', 'start': 42, 'end': 45}
{'entity': 'I-LOC', 'score': 0.9987, 'index': 11, 'word': 'York', 'start': 46, 'end': 50}
{'entity': 'I-LOC', 'score': 0.9992, 'index': 12, 'word': 'City', 'start': 51, 'end': 55}
```

### Réponse à des questions - (*Question Answering*)

La réponse à des questions (*Question Answering* ou *QA* en anglais) est une tâche de traitement du language naturel qui consiste à fournir une réponse à une question, parfois avec l'aide d'un contexte (domaine ouvert) et d'autres fois sans contexte (domaine fermé). Cette tâche intervient lorsqu'on interroge un assistant virtuel, par exemple pour savoir si un restaurant est ouvert. Elle est également utilisée pour le support client, technique, et pour aider les moteurs de recherche à fournir des informations pertinentes.

Il existe deux types courants de réponse à des questions :

- **Extractive** : pour une question donnée et un contexte fourni, la réponse est extraite directement du texte du contexte par le modèle.
- **Abstractive** : pour une question donnée et un contexte, la réponse est générée à partir du contexte. Cette approche utilise le [`Text2TextGenerationPipeline`] plutôt que le [`QuestionAnsweringPipeline`] montré ci-dessous.


```py
>>> from transformers import pipeline

>>> question_answerer = pipeline(task="question-answering")
>>> preds = question_answerer(
...     question="What is the name of the repository?",
...     context="The name of the repository is huggingface/transformers",
... )
>>> print(
...     f"score: {round(preds['score'], 4)}, start: {preds['start']}, end: {preds['end']}, answer: {preds['answer']}"
... )
score: 0.9327, start: 30, end: 54, answer: huggingface/transformers
```

### Résumé de texte - (*Summarization*)

Le résumé de text consiste à créer une version plus courte d'un texte tout en conservant l'essentiel du sens du document original. C'est une tâche de séquence à séquence qui produit un texte plus condensé à partir du texte initial. Cette technique est utile pour aider les lecteurs à saisir rapidement les points clés de longs documents, comme les projets de loi, les documents juridiques et financiers, les brevets, et les articles scientifiques.

Il existe deux types courants de summarization :

- **Extractive** : identifier et extraire les phrases les plus importantes du texte original.
- **Abstractive** : générer un résumé qui peut inclure des mots nouveaux non présents dans le texte d'origine. Le [`SummarizationPipeline`] utilise l'approche abstractive.

```py
>>> from transformers import pipeline

>>> summarizer = pipeline(task="summarization")
>>> summarizer(
...     "In this work, we presented the Transformer, the first sequence transduction model based entirely on attention, replacing the recurrent layers most commonly used in encoder-decoder architectures with multi-headed self-attention. For translation tasks, the Transformer can be trained significantly faster than architectures based on recurrent or convolutional layers. On both WMT 2014 English-to-German and WMT 2014 English-to-French translation tasks, we achieve a new state of the art. In the former task our best model outperforms even all previously reported ensembles."
... )
[{'summary_text': ' The Transformer is the first sequence transduction model based entirely on attention . It replaces the recurrent layers most commonly used in encoder-decoder architectures with multi-headed self-attention . For translation tasks, the Transformer can be trained significantly faster than architectures based on recurrent or convolutional layers .'}]
```

### Traduction

La traduction convertit un texte d'une langue à une autre. Elle facilite la communication entre personnes de différentes langues, permet de toucher des audiences plus larges et peut aussi servir d'outil d'apprentissage pour ceux qui apprennent une nouvelle langue. Comme le résumé de texte, la traduction est une tâche de séquence à séquence, où le modèle reçoit une séquence d'entrée (un texte est ici vu comme une séquence de mots, ou plus précisément de tokens) et produit une séquence de sortie dans la langue cible.

Initialement, les modèles de traduction étaient principalement monolingues, mais il y a eu récemment un intérêt croissant pour les modèles multilingues capables de traduire entre plusieurs paires de langues.

```py
>>> from transformers import pipeline

>>> text = "translate English to French: Hugging Face is a community-based open-source platform for machine learning."
>>> translator = pipeline(task="translation", model="google-t5/t5-small")
>>> translator(text)
[{'translation_text': "Hugging Face est une tribune communautaire de l'apprentissage des machines."}]
```

### Modélisation du langage

La modélisation du langage consiste à prédire un mot dans un texte. Cette tâche est devenue très populaire en traitement du language naturel, car un modèle de langage préentraîné sur cette tâche peut ensuite être ajusté (*finetuned*) pour accomplir de nombreuses autres tâches. Récemment, les grands modèles de langage (LLMs) ont suscité beaucoup d'intérêt pour leur capacité à apprendre avec peu ou pas de données spécifiques à une tâche, ce qui leur permet de résoudre des problèmes pour lesquels ils n'ont pas été explicitement entraînés. Ces modèles peuvent générer du texte fluide et convaincant, bien qu'il soit important de vérifier leur précision.

Il existe deux types de modélisation du langage :

- **Causale** : le modèle prédit le token suivant dans une séquence, avec les tokens futurs masqués.

    ```py
    >>> from transformers import pipeline

    >>> prompt = "Hugging Face is a community-based open-source platform for machine learning."
    >>> generator = pipeline(task="text-generation")
    >>> generator(prompt)  # doctest: +SKIP
    ```

- **Masquée** : le modèle prédit un token masqué dans une séquence en ayant accès à tous les autres tokens de la séquence (passé et futur).

    ```py
    >>> text = "Hugging Face is a community-based open-source <mask> for machine learning."
    >>> fill_mask = pipeline(task="fill-mask")
    >>> preds = fill_mask(text, top_k=1)
    >>> preds = [
    ...     {
    ...         "score": round(pred["score"], 4),
    ...         "token": pred["token"],
    ...         "token_str": pred["token_str"],
    ...         "sequence": pred["sequence"],
    ...     }
    ...     for pred in preds
    ... ]
    >>> preds
    [{'score': 0.2236,
      'token': 1761,
      'token_str': ' platform',
      'sequence': 'Hugging Face is a community-based open-source platform for machine learning.'}]
    ```

## Multimodal

Les tâches multimodales nécessitent qu'un modèle traite plusieurs types de données (texte, image, audio, vidéo) pour résoudre un problème spécifique. Par exemple, la génération de légendes pour les images est une tâche multimodale où le modèle prend une image en entrée et produit une séquence de texte décrivant l'image ou ses propriétés.

Bien que les modèles multimodaux traitent divers types de données, ils convertissent toutes ces données en *embeddings* (vecteurs ou listes de nombres contenant des informations significatives). Pour des tâches comme la génération de légendes pour les images, le modèle apprend les relations entre les *embeddings* d'images et ceux de texte.

### Réponse à des questions sur des documents - (*Document Question Answering*)

La réponse à des questions sur des documents consiste à répondre à des questions en langage naturel en utilisant un document comme référence. Contrairement à la réponse à des questions au niveau des tokens, qui prend du texte en entrée, cette tâche prend une image d'un document ainsi qu'une question concernant ce document, et fournit une réponse. Elle est utile pour analyser des données structurées et extraire des informations clées. Par exemple, à partir d'un reçu, on peut extraire des informations telles que le montant total et le change dû.

```py
>>> from transformers import pipeline
>>> from PIL import Image
>>> import requests

>>> url = "https://huggingface.co/datasets/hf-internal-testing/example-documents/resolve/main/jpeg_images/2.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> doc_question_answerer = pipeline("document-question-answering", model="magorshunov/layoutlm-invoices")
>>> preds = doc_question_answerer(
...     question="What is the total amount?",
...     image=image,
... )
>>> preds
[{'score': 0.8531, 'answer': '17,000', 'start': 4, 'end': 4}]
```

En espérant que cette page vous ait donné plus d'informations sur les différents types de tâches dans chaque modalité et l'importance pratique de chacune d'elles. Dans la [section suivante](tasks_explained), vous découvrirez **comment** 🤗 Transformers fonctionne pour résoudre ces tâches.
