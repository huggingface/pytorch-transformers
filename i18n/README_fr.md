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
    <img alt="Bibliothèque Hugging Face Transformers" src="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/transformers-logo-light.svg" width="352" height="59" style="max-width: 100%;">
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
        <b>Français</b> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_de.md">Deutsch</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_vi.md">Tiếng Việt</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ar.md">العربية</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ur.md">اردو</a> |
    </p>
</h4>

<h3 align="center">
    <p>Apprentissage automatique de pointe pour JAX, PyTorch et TensorFlow</p>
</h3>

<h3 align="center">
    <a href="https://hf.co/course"><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/course_banner.png"></a>
</h3>

🤗 Transformers fournit des milliers de modèles pré-entraînés pour effectuer des tâches sur différentes modalités telles que le texte, la vision et l'audio.

Ces modèles peuvent être appliqués à :

* 📝 Texte, pour des tâches telles que la classification de texte, l'extraction d'informations, la réponse aux questions, le résumé, la traduction et la génération de texte, dans plus de 100 langues.
* 🖼️ Images, pour des tâches telles que la classification d'images, la détection d'objets et la segmentation.
* 🗣️ Audio, pour des tâches telles que la reconnaissance vocale et la classification audio.

Les modèles de transformer peuvent également effectuer des tâches sur **plusieurs modalités combinées**, telles que la réponse aux questions sur des tableaux, la reconnaissance optique de caractères, l'extraction d'informations à partir de documents numérisés, la classification vidéo et la réponse aux questions visuelles.

🤗 Transformers fournit des API pour télécharger et utiliser rapidement ces modèles pré-entraînés sur un texte donné, les affiner sur vos propres ensembles de données, puis les partager avec la communauté sur notre [hub de modèles](https://huggingface.co/models). En même temps, chaque module Python définissant une architecture est complètement indépendant et peut être modifié pour permettre des expériences de recherche rapides.

🤗 Transformers est soutenu par les trois bibliothèques d'apprentissage profond les plus populaires — [Jax](https://jax.readthedocs.io/en/latest/), [PyTorch](https://pytorch.org/) et [TensorFlow](https://www.tensorflow.org/) — avec une intégration transparente entre eux. Il est facile de former vos modèles avec l'un avant de les charger pour l'inférence avec l'autre.

## Démos en ligne

Vous pouvez tester la plupart de nos modèles directement sur leurs pages du [hub de modèles](https://huggingface.co/models). Nous proposons également [l'hébergement privé de modèles, le versionning et une API d'inférence](https://huggingface.co/pricing) pour des modèles publics et privés.

Voici quelques exemples :

En traitement du langage naturel :
- [Complétion de mots masqués avec BERT](https://huggingface.co/google-bert/bert-base-uncased?text=Paris+is+the+%5BMASK%5D+of+France)
- [Reconnaissance d'entités nommées avec Electra](https://huggingface.co/dbmdz/electra-large-discriminator-finetuned-conll03-english?text=My+name+is+Sarah+and+I+live+in+London+city)
- [Génération de texte avec GPT-2](https://huggingface.co/openai-community/gpt2?text=A+long+time+ago%2C+)
- [Inférence de langage naturel avec RoBERTa](https://huggingface.co/FacebookAI/roberta-large-mnli?text=The+dog+was+lost.+Nobody+lost+any+animal)
- [Résumé avec BART](https://huggingface.co/facebook/bart-large-cnn?text=The+tower+is+324+metres+%281%2C063+ft%29+tall%2C+about+the+same+height+as+an+81-storey+building%2C+and+the+tallest+structure+in+Paris.+Its+base+is+square%2C+measuring+125+metres+%28410+ft%29+on+each+side.+During+its+construction%2C+the+Eiffel+Tower+surpassed+the+Washington+Monument+to+become+the+tallest+man-made+structure+in+the+world%2C+a+title+it+held+for+41+years+until+the+Chrysler+Building+in+New+York+City+was+finished+in+1930.+It+was+the+first+structure+to+reach+a+height+of+300+metres.+Due+to+the+addition+of+a+broadcasting+aerial+at+the+top+of+the+tower+in+1957%2C+it+is+now+taller+than+the+Chrysler+Building+by+5.2+metres+%2817+ft%29.+Excluding+transmitters%2C+the+Eiffel+Tower+is+the+second+tallest+free-standing+structure+in+France+after+the+Millau+Viaduct)
- [Réponse aux questions avec DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased-distilled-squad?text=Which+name+is+also+used+to+describe+the+Amazon+rainforest+in+English%3F&context=The+Amazon+rainforest+%28Portuguese%3A+Floresta+Amaz%C3%B4nica+or+Amaz%C3%B4nia%3B+Spanish%3A+Selva+Amaz%C3%B3nica%2C+Amazon%C3%ADa+or+usually+Amazonia%3B+French%3A+For%C3%AAt+amazonienne%3B+Dutch%3A+Amazoneregenwoud%29%2C+also+known+in+English+as+Amazonia+or+the+Amazon+Jungle%2C+is+a+moist+broadleaf+forest+that+covers+most+of+the+Amazon+basin+of+South+America.+This+basin+encompasses+7%2C000%2C000+square+kilometres+%282%2C700%2C000+sq+mi%29%2C+of+which+5%2C500%2C000+square+kilometres+%282%2C100%2C000+sq+mi%29+are+covered+by+the+rainforest.+This+region+includes+territory+belonging+to+nine+nations.+The+majority+of+the+forest+is+contained+within+Brazil%2C+with+60%25+of+the+rainforest%2C+followed+by+Peru+with+13%25%2C+Colombia+with+10%25%2C+and+with+minor+amounts+in+Venezuela%2C+Ecuador%2C+Bolivia%2C+Guyana%2C+Suriname+and+French+Guiana.+States+or+departments+in+four+nations+contain+%22Amazonas%22+in+their+names.+The+Amazon+represents+over+half+of+the+planet%27s+remaining+rainforests%2C+and+comprises+the+largest+and+most+biodiverse+tract+of+tropical+rainforest+in+the+world%2C+with+an+estimated+390+billion+individual+trees+divided+into+16%2C000+species)
- [Traduction avec T5](https://huggingface.co/google-t5/t5-base?text=My+name+is+Wolfgang+and+I+live+in+Berlin)

En vision par ordinateur :
- [Classification d'images avec ViT](https://huggingface.co/google/vit-base-patch16-224)
- [Détection d'objets avec DETR](https://huggingface.co/facebook/detr-resnet-50)
- [Segmentation sémantique avec SegFormer](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512)
- [Segmentation panoptique avec MaskFormer](https://huggingface.co/facebook/maskformer-swin-small-coco)
- [Estimation de profondeur avec DPT](https://huggingface.co/docs/transformers/model_doc/dpt)
- [Classification vidéo avec VideoMAE](https://huggingface.co/docs/transformers/model_doc/videomae)
- [Segmentation universelle avec OneFormer](https://huggingface.co/shi-labs/oneformer_ade20k_dinat_large)

En audio :
- [Reconnaissance automatique de la parole avec Wav2Vec2](https://huggingface.co/facebook/wav2vec2-base-960h)
- [Spotting de mots-clés avec Wav2Vec2](https://huggingface.co/superb/wav2vec2-base-superb-ks)
- [Classification audio avec Audio Spectrogram Transformer](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593)

Dans les tâches multimodales :
- [Réponses aux questions sur table avec TAPAS](https://huggingface.co/google/tapas-base-finetuned-wtq)
- [Réponses aux questions visuelles avec ViLT](https://huggingface.co/dandelin/vilt-b32-finetuned-vqa)
- [Classification d'images sans étiquette avec CLIP](https://huggingface.co/openai/clip-vit-large-patch14)
- [Réponses aux questions sur les documents avec LayoutLM](https://huggingface.co/impira/layoutlm-document-qa)
- [Classification vidéo sans étiquette avec X-CLIP](https://huggingface.co/docs/transformers/model_doc/xclip)


## 100 projets utilisant Transformers

Transformers est plus qu'une boîte à outils pour utiliser des modèles pré-entraînés : c'est une communauté de projets construits autour de lui et du Hub Hugging Face. Nous voulons que Transformers permette aux développeurs, chercheurs, étudiants, professeurs, ingénieurs et à quiconque d'imaginer et de réaliser leurs projets de rêve.

Afin de célébrer les 100 000 étoiles de transformers, nous avons décidé de mettre en avant la communauté et avons créé la page [awesome-transformers](./awesome-transformers.md) qui répertorie 100 projets incroyables construits autour de transformers.

Si vous possédez ou utilisez un projet que vous pensez devoir figurer dans la liste, veuillez ouvrir une pull request pour l'ajouter !

## Si vous recherchez un support personnalisé de la part de l'équipe Hugging Face

<a target="_blank" href="https://huggingface.co/support">
    <img alt="Programme d'accélération des experts HuggingFace" src="https://cdn-media.huggingface.co/marketing/transformers/new-support-improved.png" style="max-width: 600px; border: 1px solid #eee; border-radius: 4px; box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);">
</a><br>

## Tour rapide

Pour utiliser immédiatement un modèle sur une entrée donnée (texte, image, audio,...), nous fournissons l'API `pipeline`. Les pipelines regroupent un modèle pré-entraîné avec la préparation des données qui a été utilisée lors de l'entraînement de ce modèle. Voici comment utiliser rapidement un pipeline pour classer des textes en positif ou négatif :

```python
>>> from transformers import pipeline

# Allouer un pipeline pour l'analyse de sentiment
>>> classifieur = pipeline('sentiment-analysis')
>>> classifieur("Nous sommes très heureux d'introduire le pipeline dans le référentiel transformers.")
[{'label': 'POSITIF', 'score': 0.9996980428695679}]
```

La deuxième ligne de code télécharge et met en cache le modèle pré-entraîné utilisé par le pipeline, tandis que la troisième l'évalue sur le texte donné. Ici, la réponse est "positive" avec une confiance de 99,97%.

De nombreuses tâches ont une pipeline pré-entraîné prêt à l'emploi, en NLP, mais aussi en vision par ordinateur et en parole. Par exemple, nous pouvons facilement extraire les objets détectés dans une image :

```python
>>> import requests
>>> from PIL import Image
>>> from transformers import pipeline

# Télécharger une image avec de jolis chats
>>> url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png"
>>> donnees_image = requests.get(url, stream=True).raw
>>> image = Image.open(donnees_image)

# Allouer un pipeline pour la détection d'objets
>>> detecteur_objets = pipeline('object-detection')
>>> detecteur_objets(image)
[{'score': 0.9982201457023621,
  'label': 'télécommande',
  'box': {'xmin': 40, 'ymin': 70, 'xmax': 175, 'ymax': 117}},
 {'score': 0.9960021376609802,
  'label': 'télécommande',
  'box': {'xmin': 333, 'ymin': 72, 'xmax': 368, 'ymax': 187}},
 {'score': 0.9954745173454285,
  'label': 'canapé',
  'box': {'xmin': 0, 'ymin': 1, 'xmax': 639, 'ymax': 473}},
 {'score': 0.9988006353378296,
  'label': 'chat',
  'box': {'xmin': 13, 'ymin': 52, 'xmax': 314, 'ymax': 470}},
 {'score': 0.9986783862113953,
  'label': 'chat',
  'box': {'xmin': 345, 'ymin': 23, 'xmax': 640, 'ymax': 368}}]
```

Ici, nous obtenons une liste d'objets détectés dans l'image, avec une boîte entourant l'objet et un score de confiance. Voici l'image originale à gauche, avec les prédictions affichées à droite :

<h3 align="center">
    <a><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png" width="400"></a>
    <a><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample_post_processed.png" width="400"></a>
</h3>

Vous pouvez en savoir plus sur les tâches supportées par l'API pipeline dans [ce tutoriel](https://huggingface.co/docs/transformers/task_summary).

En plus de `pipeline`, pour télécharger et utiliser n'importe lequel des modèles pré-entraînés sur votre tâche donnée, il suffit de trois lignes de code. Voici la version PyTorch :

```python
>>> from transformers import AutoTokenizer, AutoModel

>>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
>>> model = AutoModel.from_pretrained("google-bert/bert-base-uncased")

inputs = tokenizer("Bonjour le monde !", return_tensors="pt")
outputs = model(**inputs)
```

Et voici le code équivalent pour TensorFlow :

```python
from transformers import AutoTokenizer, TFAutoModel

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
model = TFAutoModel.from_pretrained("google-bert/bert-base-uncased")

inputs = tokenizer("Bonjour le monde !", return_tensors="tf")
outputs = model(**inputs)
```

Le tokenizer est responsable de toutes les étapes de prétraitement que le modèle préentraîné attend et peut être appelé directement sur une seule chaîne de caractères (comme dans les exemples ci-dessus) ou sur une liste. Il produira un dictionnaire que vous pouvez utiliser dans votre code ou simplement passer directement à votre modèle en utilisant l'opérateur de déballage **.

Le modèle lui-même est un module [`nn.Module` PyTorch](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) ou un modèle [`tf.keras.Model` TensorFlow](https://www.tensorflow.org/api_docs/python/tf/keras/Model) (selon votre backend) que vous pouvez utiliser comme d'habitude. [Ce tutoriel](https://huggingface.co/docs/transformers/training) explique comment intégrer un tel modèle dans une boucle d'entraînement classique PyTorch ou TensorFlow, ou comment utiliser notre API `Trainer` pour affiner rapidement sur un nouvel ensemble de données.

## Pourquoi devrais-je utiliser transformers ?

1. Des modèles de pointe faciles à utiliser :
    - Hautes performances en compréhension et génération de langage naturel, en vision par ordinateur et en tâches audio.
    - Faible barrière à l'entrée pour les éducateurs et les praticiens.
    - Peu d'abstractions visibles pour l'utilisateur avec seulement trois classes à apprendre.
    - Une API unifiée pour utiliser tous nos modèles préentraînés.

1. Coûts informatiques réduits, empreinte carbone plus petite :
    - Les chercheurs peuvent partager des modèles entraînés au lieu de toujours les réentraîner.
    - Les praticiens peuvent réduire le temps de calcul et les coûts de production.
    - Des dizaines d'architectures avec plus de 400 000 modèles préentraînés dans toutes les modalités.

1. Choisissez le bon framework pour chaque partie de la vie d'un modèle :
    - Entraînez des modèles de pointe en 3 lignes de code.
    - Trasnférer un seul modèle entre les frameworks TF2.0/PyTorch/JAX à volonté.
    - Choisissez facilement le bon framework pour l'entraînement, l'évaluation et la production.

1. Personnalisez facilement un modèle ou un exemple selon vos besoins :
    - Nous fournissons des exemples pour chaque architecture afin de reproduire les résultats publiés par ses auteurs originaux.
    - Les détails internes du modèle sont exposés de manière aussi cohérente que possible.
    - Les fichiers de modèle peuvent être utilisés indépendamment de la bibliothèque pour des expériences rapides.

## Pourquoi ne devrais-je pas utiliser transformers ?

- Cette bibliothèque n'est pas une boîte à outils modulaire de blocs de construction pour les réseaux neuronaux. Le code dans les fichiers de modèle n'est pas refactored avec des abstractions supplémentaires à dessein, afin que les chercheurs puissent itérer rapidement sur chacun des modèles sans plonger dans des abstractions/fichiers supplémentaires.
- L'API d'entraînement n'est pas destinée à fonctionner avec n'importe quel modèle, mais elle est optimisée pour fonctionner avec les modèles fournis par la bibliothèque. Pour des boucles génériques d'apprentissage automatique, vous devriez utiliser une autre bibliothèque (éventuellement, [Accelerate](https://huggingface.co/docs/accelerate)).
- Bien que nous nous efforcions de présenter autant de cas d'utilisation que possible, les scripts de notre [dossier d'exemples](https://github.com/huggingface/transformers/tree/main/examples) ne sont que cela : des exemples. Il est prévu qu'ils ne fonctionnent pas immédiatement sur votre problème spécifique et que vous devrez probablement modifier quelques lignes de code pour les adapter à vos besoins.

## Installation

### Avec pip

Ce référentiel est testé sur Python 3.9+, Flax 0.4.1+, PyTorch 2.0+ et TensorFlow 2.6+.

Vous devriez installer 🤗 Transformers dans un [environnement virtuel](https://docs.python.org/3/library/venv.html). Si vous n'êtes pas familier avec les environnements virtuels Python, consultez le [guide utilisateur](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).

D'abord, créez un environnement virtuel avec la version de Python que vous allez utiliser et activez-le.

Ensuite, vous devrez installer au moins l'un de Flax, PyTorch ou TensorFlow.
Veuillez vous référer à la page d'installation de [TensorFlow](https://www.tensorflow.org/install/), de [PyTorch](https://pytorch.org/get-started/locally/#start-locally) et/ou de [Flax](https://github.com/google/flax#quick-install) et [Jax](https://github.com/google/jax#installation) pour connaître la commande d'installation spécifique à votre plateforme.

Lorsqu'un de ces backends est installé, 🤗 Transformers peut être installé avec pip comme suit :

```bash
pip install transformers
```

Si vous souhaitez jouer avec les exemples ou avez besoin de la dernière version du code et ne pouvez pas attendre une nouvelle version, vous devez [installer la bibliothèque à partir de la source](https://huggingface.co/docs/transformers/installation#installing-from-source).

### Avec conda

🤗 Transformers peut être installé avec conda comme suit :

```shell
conda install conda-forge::transformers
```

> **_NOTE:_** L'installation de `transformers` depuis le canal `huggingface` est obsolète.

Suivez les pages d'installation de Flax, PyTorch ou TensorFlow pour voir comment les installer avec conda.

> **_NOTE:_** Sur Windows, on peut vous demander d'activer le mode développeur pour bénéficier de la mise en cache. Si ce n'est pas une option pour vous, veuillez nous le faire savoir dans [cette issue](https://github.com/huggingface/huggingface_hub/issues/1062).

## Architectures de modèles

**[Tous les points de contrôle](https://huggingface.co/models)** de modèle fournis par 🤗 Transformers sont intégrés de manière transparente depuis le [hub de modèles](https://huggingface.co/models) huggingface.co, où ils sont téléchargés directement par les [utilisateurs](https://huggingface.co/users) et les [organisations](https://huggingface.co/organizations).

Nombre actuel de points de contrôle : ![](https://img.shields.io/endpoint?url=https://huggingface.co/api/shields/models&color=brightgreen)

🤗 Transformers fournit actuellement les architectures suivantes: consultez [ici](https://huggingface.co/docs/transformers/model_summary) pour un résumé global de chacune d'entre elles.

Pour vérifier si chaque modèle a une implémentation en Flax, PyTorch ou TensorFlow, ou s'il a un tokenizer associé pris en charge par la bibliothèque 🤗 Tokenizers, consultez [ce tableau](https://huggingface.co/docs/transformers/index#supported-frameworks).

Ces implémentations ont été testées sur plusieurs ensembles de données (voir les scripts d'exemple) et devraient correspondre aux performances des implémentations originales. Vous pouvez trouver plus de détails sur les performances dans la section Exemples de la [documentation](https://github.com/huggingface/transformers/tree/main/examples).

## En savoir plus

| Section | Description |
|-|-|
| [Documentation](https://huggingface.co/docs/transformers/) | Documentation complète de l'API et tutoriels |
| [Résumé des tâches](https://huggingface.co/docs/transformers/task_summary) | Tâches prises en charge par les 🤗 Transformers |
| [Tutoriel de prétraitement](https://huggingface.co/docs/transformers/preprocessing) | Utilisation de la classe `Tokenizer` pour préparer les données pour les modèles |
| [Entraînement et ajustement fin](https://huggingface.co/docs/transformers/training) | Utilisation des modèles fournis par les 🤗 Transformers dans une boucle d'entraînement PyTorch/TensorFlow et de l'API `Trainer` |
| [Tour rapide : Scripts d'ajustement fin/d'utilisation](https://github.com/huggingface/transformers/tree/main/examples) | Scripts d'exemple pour ajuster finement les modèles sur une large gamme de tâches |
| [Partage et téléversement de modèles](https://huggingface.co/docs/transformers/model_sharing) | Téléchargez et partagez vos modèles ajustés avec la communauté |

## Citation

Nous disposons désormais d'un [article](https://www.aclweb.org/anthology/2020.emnlp-demos.6/) que vous pouvez citer pour la bibliothèque 🤗 Transformers :
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
