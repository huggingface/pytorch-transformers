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

# Visite rapide

[[open-in-colab]]

Soyez opérationnel avec 🤗 Transformers ! Que vous soyez un développeur ou un utilisateur lambda, cette visite rapide vous aidera à démarrer et vous montrera comment utiliser le [`pipeline`] pour l'inférence, charger un modèle pré-entraîné et un préprocesseur avec une [AutoClass](./model_doc/auto), et entraîner rapidement un modèle avec PyTorch ou TensorFlow. Si vous êtes un débutant, nous vous recommandons de consulter nos tutoriels ou notre [cours](https://huggingface.co/course/chapter1/1) suivant pour des explications plus approfondies des concepts présentés ici.

Avant de commencer, assurez-vous que vous avez installé toutes les bibliothèques nécessaires :

```bash
!pip install transformers datasets evaluate accelerate
```

Vous aurez aussi besoin d'installer votre bibliothèque d'apprentissage profond favorite :

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

## Pipeline

<Youtube id="tiZFewofSLM"/>

Le [`pipeline`] est le moyen le plus simple d'utiliser un modèle pré-entraîné pour l'inférence. Vous pouvez utiliser le [`pipeline`] prêt à l'emploi pour de nombreuses tâches dans différentes modalités. Consultez le tableau ci-dessous pour connaître les tâches prises en charge :

| **Tâche**                     | **Description**                                                                                              | **Modalité**        | **Identifiant du pipeline**                   |
|------------------------------|--------------------------------------------------------------------------------------------------------------|----------------------|-----------------------------------------------|
| Classification de texte      | Attribue une catégorie à une séquence de texte donnée                                                        | Texte                | pipeline(task="sentiment-analysis")           |
| Génération de texte          | Génère du texte à partir d'une consigne donnée                                                               | Texte                | pipeline(task="text-generation")              |
| Reconnaissance de token nommé      | Attribue une catégorie à chaque token dans une séquence (personnes, organisation, localisation, etc.)                            | Texte                | pipeline(task="ner")                          |
| Question réponse             | Extrait une réponse du texte en fonction du contexte et d'une question                                       | Texte                | pipeline(task="question-answering")           |
| Prédiction de token masqué                    | Prédit correctement le token masqué dans une séquence                                                               | Texte                | pipeline(task="fill-mask")                    |
| Génération de résumé                | Génère un résumé d'une séquence de texte donnée ou d'un document                                                         | Texte                | pipeline(task="summarization")                |
| Traduction                  | Traduit du texte d'un langage à un autre                                                                      | Texte                | pipeline(task="translation")                  |
| Classification d'image       | Attribue une catégorie à une image                                                                           | Image                | pipeline(task="image-classification")         |
| Segmentation d'image           | Attribue une catégorie à chaque pixel d'une image (supporte la segmentation sémantique, panoptique et d'instance) | Image                | pipeline(task="image-segmentation")           |
| Détection d'objets             | Prédit les délimitations et catégories d'objets dans une image                                                | Image                | pipeline(task="object-detection")             |
| Classification d'audio       | Attribue une catégorie à un fichier audio                                                                    | Audio                | pipeline(task="audio-classification")         |
| Reconnaissance automatique de la parole | Extrait le discours d'un fichier audio en texte                                                                  | Audio                | pipeline(task="automatic-speech-recognition") |
| Question réponse visuels    | Etant données une image et une question, répond correctement à une question sur l'image                                   | Modalités multiples  | pipeline(task="vqa")                          |

Commencez par créer une instance de [`pipeline`] et spécifiez la tâche pour laquelle vous souhaitez l'utiliser. Vous pouvez utiliser le [`pipeline`] pour n'importe laquelle des tâches mentionnées dans le tableau précédent. Pour obtenir une liste complète des tâches prises en charge, consultez la documentation de l'[API pipeline](./main_classes/pipelines). Dans ce guide, nous utiliserons le [`pipeline`] pour l'analyse des sentiments à titre d'exemple :

```py
>>> from transformers import pipeline

>>> classifier = pipeline("sentiment-analysis")
```

Le [`pipeline`] télécharge et stocke en cache un [modèle pré-entraîné](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english) et un tokenizer par défaut pour l'analyse des sentiments. Vous pouvez maintenant utiliser le `classifier` sur le texte de votre choix :

```py
>>> classifier("We are very happy to show you the 🤗 Transformers library.")
[{'label': 'POSITIVE', 'score': 0.9998}]
```

Si vous voulez classifier plus qu'un texte, donnez une liste de textes au [`pipeline`] pour obtenir une liste de dictionnaires en retour :

```py
>>> results = classifier(["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."])
>>> for result in results:
...     print(f"label: {result['label']}, avec le score de: {round(result['score'], 4)}")
label: POSITIVE, avec le score de: 0.9998
label: NEGATIVE, avec le score de: 0.5309
```

Le [`pipeline`] peut aussi itérer sur un jeu de données entier pour n'importe quelle tâche. Prenons par exemple la reconnaissance automatique de la parole :

```py
>>> import torch
>>> from transformers import pipeline

>>> speech_recognizer = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")
```

Chargez un jeu de données audio (voir le 🤗 Datasets [Quick Start](https://huggingface.co/docs/datasets/quickstart#audio) pour plus de détails) sur lequel vous souhaitez itérer. Pour cet exemple, nous chargeons le jeu de données [MInDS-14](https://huggingface.co/datasets/PolyAI/minds14) :

```py
>>> from datasets import load_dataset, Audio

>>> dataset = load_dataset("PolyAI/minds14", name="en-US", split="train")  # doctest: +IGNORE_RESULT
```

Vous devez vous assurer que le taux d'échantillonnage de l'ensemble de données correspond au taux d'échantillonnage sur lequel [`facebook/wav2vec2-base-960h`](https://huggingface.co/facebook/wav2vec2-base-960h) a été entraîné :

```py
>>> dataset = dataset.cast_column("audio", Audio(sampling_rate=speech_recognizer.feature_extractor.sampling_rate))
```

Les fichiers audio sont automatiquement chargés et rééchantillonnés lors de l'appel de la colonne `"audio"`.
Extrayez les tableaux de formes d'ondes brutes des quatre premiers échantillons et passez-les comme une liste au pipeline :

```py
>>> result = speech_recognizer(dataset[:4]["audio"])
>>> print([d["text"] for d in result])
['I WOULD LIKE TO SET UP A JOINT ACCOUNT WITH MY PARTNER HOW DO I PROCEED WITH DOING THAT', "FODING HOW I'D SET UP A JOIN TO HET WITH MY WIFE AND WHERE THE AP MIGHT BE", "I I'D LIKE TOY SET UP A JOINT ACCOUNT WITH MY PARTNER I'M NOT SEEING THE OPTION TO DO IT ON THE AP SO I CALLED IN TO GET SOME HELP CAN I JUST DO IT OVER THE PHONE WITH YOU AND GIVE YOU THE INFORMATION OR SHOULD I DO IT IN THE AP AND I'M MISSING SOMETHING UQUETTE HAD PREFERRED TO JUST DO IT OVER THE PHONE OF POSSIBLE THINGS", 'HOW DO I THURN A JOIN A COUNT']
```

Pour les ensembles de données plus importants où les entrées sont volumineuses (comme dans les domaines de la parole ou de la vision), utilisez plutôt un générateur au lieu d'une liste pour charger toutes les entrées en mémoire. Pour plus d'informations, consultez la documentation de l'[API pipeline](./main_classes/pipelines).

### Utiliser une autre modèle et tokenizer dans le pipeline

Le [`pipeline`] peut être utilisé avec n'importe quel modèle du [Hub](https://huggingface.co/models), ce qui permet d'adapter facilement le [`pipeline`] à d'autres cas d'utilisation. Par exemple, si vous souhaitez un modèle capable de traiter du texte français, utilisez les filtres du Hub pour trouver un modèle approprié. Le premier résultat renvoie un [modèle BERT](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment) multilingue finetuné pour l'analyse des sentiments que vous pouvez utiliser pour le texte français :

```py
>>> model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
```

<frameworkcontent>
<pt>
Utilisez [`AutoModelForSequenceClassification`] et [`AutoTokenizer`] pour charger le modèle pré-entraîné et le tokenizer adapté (plus de détails sur une `AutoClass` dans la section suivante) :

```py
>>> from transformers import AutoTokenizer, AutoModelForSequenceClassification

>>> model = AutoModelForSequenceClassification.from_pretrained(model_name)
>>> tokenizer = AutoTokenizer.from_pretrained(model_name)
```
</pt>
<tf>
Utilisez [`TFAutoModelForSequenceClassification`] et [`AutoTokenizer`] pour charger le modèle pré-entraîné et le tokenizer adapté (plus de détails sur une `TFAutoClass` dans la section suivante) :

```py
>>> from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

>>> model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
>>> tokenizer = AutoTokenizer.from_pretrained(model_name)
```
</tf>
</frameworkcontent>

Spécifiez le modèle et le tokenizer dans le [`pipeline`], et utilisez le `classifier` sur le texte en français :

```py
>>> classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
>>> classifier("Nous sommes très heureux de vous présenter la bibliothèque 🤗 Transformers.")
[{'label': '5 stars', 'score': 0.7273}]
```

Si vous ne parvenez pas à trouver un modèle adapté à votre cas d'utilisation, vous devrez finetuner un modèle pré-entraîné sur vos données. Jetez un coup d'œil à notre [tutoriel sur le finetuning](./training) pour apprendre comment faire. Enfin, après avoir finetuné votre modèle pré-entraîné, pensez à [partager](./model_sharing) le modèle avec la communauté sur le Hub afin de démocratiser l'apprentissage automatique pour tous ! 🤗

## AutoClass

<Youtube id="AhChOFRegn4"/>

Les classes [`AutoModelForSequenceClassification`] et [`AutoTokenizer`] fonctionnent ensemble pour créer un [`pipeline`] comme celui que vous avez utilisé ci-dessus. Une [AutoClass](./model_doc/auto) est un raccourci qui récupère automatiquement l'architecture d'un modèle pré-entraîné à partir de son nom ou de son emplacement. Il vous suffit de sélectionner l'`AutoClass` appropriée à votre tâche et la classe de prétraitement qui lui est associée.

Reprenons l'exemple de la section précédente et voyons comment vous pouvez utiliser l'`AutoClass` pour reproduire les résultats du [`pipeline`].

### AutoTokenizer

Un tokenizer est chargé de prétraiter le texte pour en faire un tableau de chiffres qui servira d'entrée à un modèle. De nombreuses règles régissent le processus de tokenisation, notamment la manière de diviser un mot et le niveau auquel les mots doivent être divisés (pour en savoir plus sur la tokenisation, consultez le [résumé](./tokenizer_summary)). La chose la plus importante à retenir est que vous devez instancier un tokenizer avec le même nom de modèle pour vous assurer que vous utilisez les mêmes règles de tokenisation que celles avec lesquelles un modèle a été pré-entraîné.

Chargez un tokenizer avec [`AutoTokenizer`] :

```py
>>> from transformers import AutoTokenizer

>>> model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
>>> tokenizer = AutoTokenizer.from_pretrained(model_name)
```

Passez votre texte au tokenizer :

```py
>>> encoding = tokenizer("We are very happy to show you the 🤗 Transformers library.")
>>> print(encoding)
{'input_ids': [101, 11312, 10320, 12495, 19308, 10114, 11391, 10855, 10103, 100, 58263, 13299, 119, 102],
 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
```

Le tokenizer retourne un dictionnaire contenant :

* [input_ids](./glossary#input-ids): la représentation numérique des tokens.
* [attention_mask](.glossary#attention-mask): indique quels tokens doivent faire l'objet d'une attention particulière (plus particulièrement les tokens de remplissage).

Un tokenizer peut également accepter une liste de textes, et remplir et tronquer le texte pour retourner un échantillon de longueur uniforme :

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

<Tip>

Consultez le tutoriel [prétraitement](./preprocessing) pour plus de détails sur la tokenisation, et sur la manière d'utiliser un [`AutoImageProcessor`], un [`AutoFeatureExtractor`] et un [`AutoProcessor`] pour prétraiter les images, l'audio et les contenus multimodaux.

</Tip>

### AutoModel

<frameworkcontent>
<pt>
🤗 Transformers fournit un moyen simple et unifié de charger des instances pré-entraînées. Cela signifie que vous pouvez charger un [`AutoModel`] comme vous chargeriez un [`AutoTokenizer`]. La seule différence est de sélectionner l'[`AutoModel`] approprié pour la tâche. Pour une classification de texte (ou de séquence de textes), vous devez charger [`AutoModelForSequenceClassification`] :

```py
>>> from transformers import AutoModelForSequenceClassification

>>> model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
>>> pt_model = AutoModelForSequenceClassification.from_pretrained(model_name)
```

<Tip>

Voir le [résumé de la tâche](./task_summary) pour vérifier si elle est prise en charge par une classe [`AutoModel`].

</Tip>

Maintenant, passez votre échantillon d'entrées prétraitées directement au modèle. Il vous suffit de décompresser le dictionnaire en ajoutant `**` :

```py
>>> pt_outputs = pt_model(**pt_batch)
```

Le modèle produit les activations finales dans l'attribut `logits`. Appliquez la fonction softmax aux `logits` pour récupérer les probabilités :

```py
>>> from torch import nn

>>> pt_predictions = nn.functional.softmax(pt_outputs.logits, dim=-1)
>>> print(pt_predictions)
tensor([[0.0021, 0.0018, 0.0115, 0.2121, 0.7725],
        [0.2084, 0.1826, 0.1969, 0.1755, 0.2365]], grad_fn=<SoftmaxBackward0>)
```
</pt>
<tf>
🤗 Transformers fournit un moyen simple et unifié de charger des instances pré-entraînés. Cela signifie que vous pouvez charger un [`TFAutoModel`] comme vous chargeriez un [`AutoTokenizer`]. La seule différence est de sélectionner le [`TFAutoModel`] approprié pour la tâche. Pour une classification de texte (ou de séquence de textes), vous devez charger [`TFAutoModelForSequenceClassification`] :

```py
>>> from transformers import TFAutoModelForSequenceClassification

>>> model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
>>> tf_model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
```

<Tip>

Voir le [résumé de la tâche](./task_summary) pour vérifier si elle est prise en charge par une classe [`AutoModel`].

</Tip>

Passez maintenant votre échantillon d'entrées prétraitées directement au modèle en passant les clés du dictionnaire directement aux tensors :

```py
>>> tf_outputs = tf_model(tf_batch)
```

Le modèle produit les activations finales dans l'attribut `logits`. Appliquez la fonction softmax aux `logits` pour récupérer les probabilités :

```py
>>> import tensorflow as tf

>>> tf_predictions = tf.nn.softmax(tf_outputs.logits, axis=-1)
>>> tf_predictions  # doctest: +IGNORE_RESULT
```
</tf>
</frameworkcontent>

<Tip>

Tous les modèles 🤗 Transformers (PyTorch ou TensorFlow) produisent les tensors *avant* la fonction d'activation finale (comme softmax) car la fonction d'activation finale est souvent fusionnée avec le calcul de la perte. Les structures produites par le modèle sont des classes de données spéciales, de sorte que leurs attributs sont autocomplétés dans un environnement de développement. Les structures produites par le modèle se comportent comme un tuple ou un dictionnaire (vous pouvez les indexer avec un entier, une tranche ou une chaîne), auquel cas les attributs qui sont None sont ignorés.

</Tip>

### Sauvegarder un modèle

<frameworkcontent>
<pt>
Une fois que votre modèle est finetuné, vous pouvez le sauvegarder avec son tokenizer en utilisant [`PreTrainedModel.save_pretrained`] :

```py
>>> pt_save_directory = "./pt_save_pretrained"
>>> tokenizer.save_pretrained(pt_save_directory)  # doctest: +IGNORE_RESULT
>>> pt_model.save_pretrained(pt_save_directory)
```

Lorsque vous voulez réutiliser le modèle, rechargez-le avec [`PreTrainedModel.from_pretrained`] :

```py
>>> pt_model = AutoModelForSequenceClassification.from_pretrained("./pt_save_pretrained")
```
</pt>
<tf>
Une fois que votre modèle est finetuné, vous pouvez le sauvegarder avec son tokenizer en utilisant [`TFPreTrainedModel.save_pretrained`] :

```py
>>> tf_save_directory = "./tf_save_pretrained"
>>> tokenizer.save_pretrained(tf_save_directory)  # doctest: +IGNORE_RESULT
>>> tf_model.save_pretrained(tf_save_directory)
```

Lorsque vous voulez réutiliser le modèle, rechargez-le avec [`TFPreTrainedModel.from_pretrained`] :

```py
>>> tf_model = TFAutoModelForSequenceClassification.from_pretrained("./tf_save_pretrained")
```
</tf>
</frameworkcontent>

Une fonctionnalité particulièrement cool 🤗 Transformers est la possibilité d'enregistrer un modèle et de le recharger en tant que modèle PyTorch ou TensorFlow. Le paramètre `from_pt` ou `from_tf` permet de convertir le modèle d'un framework à l'autre :

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

## Constructions de modèles personnalisés

Vous pouvez modifier la configuration du modèle pour changer la façon dont un modèle est construit. La configuration spécifie les attributs d'un modèle, tels que le nombre de couches ou de têtes d'attention. Vous partez de zéro lorsque vous initialisez un modèle à partir d'une configuration personnalisée. Les attributs du modèle sont initialisés de manière aléatoire et vous devrez entraîner le modèle avant de pouvoir l'utiliser pour obtenir des résultats significatifs.

Commencez par importer [`AutoConfig`], puis chargez le modèle pré-entraîné que vous voulez modifier. Dans [`AutoConfig.from_pretrained`], vous pouvez spécifier l'attribut que vous souhaitez modifier, tel que le nombre de têtes d'attention :

```py
>>> from transformers import AutoConfig

>>> my_config = AutoConfig.from_pretrained("distilbert/distilbert-base-uncased", n_heads=12)
```

<frameworkcontent>
<pt>
Créez un modèle personnalisé à partir de votre configuration avec [`AutoModel.from_config`] :

```py
>>> from transformers import AutoModel

>>> my_model = AutoModel.from_config(my_config)
```
</pt>
<tf>
Créez un modèle personnalisé à partir de votre configuration avec [`TFAutoModel.from_config`] :

```py
>>> from transformers import TFAutoModel

>>> my_model = TFAutoModel.from_config(my_config)
```
</tf>
</frameworkcontent>

Consultez le guide [Créer une architecture personnalisée](./create_a_model) pour plus d'informations sur la création de configurations personnalisées.

## Trainer - une boucle d'entraînement optimisée par PyTorch

Tous les modèles sont des [`torch.nn.Module`](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) standard, vous pouvez donc les utiliser dans n'importe quelle boucle d'entraînement typique. Bien que vous puissiez écrire votre propre boucle d'entraînement, 🤗 Transformers fournit une classe [`Trainer`] pour PyTorch, qui contient la boucle d'entraînement de base et ajoute des fonctionnalités supplémentaires comme l'entraînement distribué, la précision mixte, et plus encore.

En fonction de votre tâche, vous passerez généralement les paramètres suivants à [`Trainer`] :

1. Un [`PreTrainedModel`] ou un [`torch.nn.Module`](https://pytorch.org/docs/stable/nn.html#torch.nn.Module):

   ```py
   >>> from transformers import AutoModelForSequenceClassification

   >>> model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased")
   ```

2. [`TrainingArguments`] contient les hyperparamètres du modèle que vous pouvez changer comme le taux d'apprentissage, la taille de l'échantillon, et le nombre d'époques pour s'entraîner. Les valeurs par défaut sont utilisées si vous ne spécifiez pas d'hyperparamètres d'apprentissage :

   ```py
   >>> from transformers import TrainingArguments

   >>> training_args = TrainingArguments(
   ...     output_dir="path/to/save/folder/",
   ...     learning_rate=2e-5,
   ...     per_device_train_batch_size=8,
   ...     per_device_eval_batch_size=8,
   ...     num_train_epochs=2,
   ... )
   ```

3. Une classe de prétraitement comme un tokenizer, un processeur d'images ou un extracteur de caractéristiques :

   ```py
   >>> from transformers import AutoTokenizer

   >>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
   ```

4. Chargez un jeu de données :

   ```py
   >>> from datasets import load_dataset

   >>> dataset = load_dataset("rotten_tomatoes")  # doctest: +IGNORE_RESULT
   ```

5. Créez une fonction qui transforme le texte du jeu de données en token :

   ```py
   >>> def tokenize_dataset(dataset):
   ...     return tokenizer(dataset["text"])
   ```

   Puis appliquez-la à l'intégralité du jeu de données avec [`~datasets.Dataset.map`]:

   ```py
   >>> dataset = dataset.map(tokenize_dataset, batched=True)
   ```

6. Un [`DataCollatorWithPadding`] pour créer un échantillon d'exemples à partir de votre jeu de données :

   ```py
   >>> from transformers import DataCollatorWithPadding

   >>> data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
   ```

Maintenant, rassemblez tous ces éléments dans un [`Trainer`] :

```py
>>> from transformers import Trainer

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=dataset["train"],
...     eval_dataset=dataset["test"],
...     processing_class=tokenizer,
...     data_collator=data_collator,
... )  # doctest: +SKIP
```

Une fois que vous êtes prêt, appelez la fonction [`~Trainer.train`] pour commencer l'entraînement :

```py
>>> trainer.train()  # doctest: +SKIP
```

<Tip>

Pour les tâches - comme la traduction ou la génération de résumé - qui utilisent un modèle séquence à séquence, utilisez plutôt les classes [`Seq2SeqTrainer`] et [`Seq2SeqTrainingArguments`].

</Tip>

Vous pouvez personnaliser le comportement de la boucle d'apprentissage en redéfinissant les méthodes à l'intérieur de [`Trainer`]. Cela vous permet de personnaliser des caractéristiques telles que la fonction de perte, l'optimiseur et le planificateur. Consultez la documentation de [`Trainer`] pour savoir quelles méthodes peuvent être redéfinies.

L'autre moyen de personnaliser la boucle d'apprentissage est d'utiliser les [Callbacks](./main_classes/callback). Vous pouvez utiliser les callbacks pour intégrer d'autres bibliothèques et inspecter la boucle d'apprentissage afin de suivre la progression ou d'arrêter l'apprentissage plus tôt. Les callbacks ne modifient rien dans la boucle d'apprentissage elle-même. Pour personnaliser quelque chose comme la fonction de perte, vous devez redéfinir le [`Trainer`] à la place.

## Entraînement avec TensorFlow

Tous les modèles sont des modèles standard [`tf.keras.Model`](https://www.tensorflow.org/api_docs/python/tf/keras/Model) afin qu'ils puissent être entraînés avec TensorFlow avec l'API [Keras](https://keras.io/). 🤗 Transformers fournit la fonction [`~TFPreTrainedModel.prepare_tf_dataset`] pour charger facilement votre jeu de données comme un `tf.data.Dataset` afin que vous puissiez commencer l'entraînement immédiatement avec les fonctions [`compile`](https://keras.io/api/models/model_training_apis/#compile-method) et [`fit`](https://keras.io/api/models/model_training_apis/#fit-method) de Keras.

1. Vous commencez avec un modèle [`TFPreTrainedModel`] ou [`tf.keras.Model`](https://www.tensorflow.org/api_docs/python/tf/keras/Model) :

   ```py
   >>> from transformers import TFAutoModelForSequenceClassification

   >>> model = TFAutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased")
   ```

2. Une classe de prétraitement comme un tokenizer, un processeur d'images ou un extracteur de caractéristiques :

   ```py
   >>> from transformers import AutoTokenizer

   >>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
   ```

3. Créez une fonction qui transforme le texte du jeu de données en token :

   ```py
   >>> def tokenize_dataset(dataset):
   ...     return tokenizer(dataset["text"])  # doctest: +SKIP
   ```

4. Appliquez le tokenizer à l'ensemble du jeu de données avec [`~datasets.Dataset.map`] et passez ensuite le jeu de données et le tokenizer à [`~TFPreTrainedModel.prepare_tf_dataset`]. Vous pouvez également modifier la taille de l'échantillon et mélanger le jeu de données ici si vous le souhaitez :

   ```py
   >>> dataset = dataset.map(tokenize_dataset)  # doctest: +SKIP
   >>> tf_dataset = model.prepare_tf_dataset(
   ...     dataset, batch_size=16, shuffle=True, tokenizer=tokenizer
   ... )  # doctest: +SKIP
   ```

5. Une fois que vous êtes prêt, appelez les fonctions `compile` et `fit` pour commencer l'entraînement :

   ```py
   >>> from tensorflow.keras.optimizers import Adam

   >>> model.compile(optimizer=Adam(3e-5))
   >>> model.fit(dataset)  # doctest: +SKIP
   ```

## Et après ?

Maintenant que vous avez terminé la visite rapide de 🤗 Transformers, consultez nos guides et apprenez à faire des choses plus spécifiques comme créer un modèle personnalisé, finetuner un modèle pour une tâche, et comment entraîner un modèle avec un script. Si vous souhaitez en savoir plus sur les concepts fondamentaux de 🤗 Transformers, jetez un œil à nos guides conceptuels !
