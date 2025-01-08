<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Comment 🤗 Transformers résout ces tâches

Dans [Ce que 🤗 Transformers peut faire](task_summary), vous avez découvert les tâches de traitement du langage naturel (NLP), de traitement de la parole et de l'audio, de vision par ordinateur, ainsi que certaines de leurs applications importantes. Cette page se penche sur la manière dont les modèles résolvent ces tâches et explique les processus en arrière-plan. Bien que différents modèles puissent utiliser diverses techniques ou approches innovantes, les modèles Transformer suivent généralement une idée commune. Grâce à leur architecture flexible, la plupart des modèles sont basés sur un encodeur, un décodeur ou une combinaison encodeur-décodeur. En plus des modèles Transformer, notre bibliothèque comprend également des réseaux de neurones convolutifs (CNN), qui restent utilisés pour les tâches de vision par ordinateur. Nous expliquerons aussi le fonctionnement d'un CNN moderne.

Voici comment différents modèles résolvent des tâches spécifiques :

- [Wav2Vec2](model_doc/wav2vec2) pour la classification audio et la reconnaissance vocale (*ASR* en anglais)
- [Vision Transformer (ViT)](model_doc/vit) et [ConvNeXT](model_doc/convnext) pour la classification d'images
- [DETR](model_doc/detr) pour la détection d'objets
- [Mask2Former](model_doc/mask2former) pour la segmentation d'images
- [GLPN](model_doc/glpn) pour l'estimation de la profondeur
- [BERT](model_doc/bert) pour les tâches de traitement du language naturel telles que la classification de texte, la classification des tokens et la réponse à des questions utilisant un encodeur
- [GPT2](model_doc/gpt2) pour les tâches de traitement du language naturel telles que la génération de texte utilisant un décodeur
- [BART](model_doc/bart) pour les tâches de traitement du language naturel telles que le résumé de texte et la traduction utilisant un encodeur-décodeur

<Tip>

Avant de poursuivre, il est utile d'avoir quelques connaissances de base sur l'architecture des Transformers. Comprendre le fonctionnement des encodeurs, des décodeurs et du mécanisme d'attention vous aidera à saisir comment les différents modèles Transformer fonctionnent. Si vous débutez ou avez besoin d'un rappel, consultez notre [cours](https://huggingface.co/course/chapter1/4?fw=pt) pour plus d'informations !

</Tip>

## Paroles et audio

[Wav2Vec2](model_doc/wav2vec2) est un modèle auto-supervisé qui est préentraîné sur des données de parole non étiquetées et ajusté sur des données étiquetées pour des tâches telles que la classification audio et la reconnaissance vocale (ASR).

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/wav2vec2_architecture.png"/>
</div>

Ce modèle comporte quatre composants principaux :

1. **Encodeur de caractéristiques** (*feature encoder*): Il prend le signal audio brut, le normalise pour avoir une moyenne nulle et une variance unitaire, et le convertit en une séquence de vecteurs de caractéristiques, chacun représentant une durée de 20 ms.

2. **Module de quantification** (*quantization module*): Les vecteurs de caractéristiques sont passés à ce module pour apprendre des unités de parole discrètes. Chaque vecteur est associé à un *codebook* (une collection de mots-clés), et l'unité de parole la plus représentative est sélectionnée parmi celles du codebook et transmise au modèle.

3. **Réseau de contexte** (*context network*): Environ la moitié des vecteurs de caractéristiques sont masqués aléatoirement. Les vecteurs masqués sont ensuite envoyés à un *réseau de contexte*, qui est un encodeur qui ajoute des embeddings positionnels relatifs.

4. **Tâche contrastive** (*contrastive task*): Le réseau de contexte est préentraîné avec une tâche contrastive. Le modèle doit prédire la véritable unité de parole quantifiée à partir de la prédiction masquée parmi un ensemble de fausses, ce qui pousse le modèle à trouver l'unité de parole quantifiée la plus proche de la prédiction.

Une fois préentraîné, wav2vec2 peut être ajusté sur vos propres données pour des tâches comme la classification audio ou la reconnaissance automatique de la parole !

### Classification audio

Pour utiliser le modèle préentraîné pour la classification audio, ajoutez une tête de classification de séquence au-dessus du modèle Wav2Vec2 de base. Cette tête de classification est une couche linéaire qui reçoit les états cachés (*hidden states*) de l'encodeur. Ces états cachés, qui représentent les caractéristiques apprises de chaque trame audio, peuvent avoir des longueurs variables. Pour obtenir un vecteur de longueur fixe, les états cachés sont d'abord regroupés, puis transformés en logits correspondant aux étiquettes de classe. La perte d'entropie croisée est calculée entre les logits et la cible pour déterminer la classe la plus probable.

Prêt à vous lancer dans la classification audio ? Consultez notre [guide complet de classification audio](tasks/audio_classification) pour apprendre à ajuster Wav2Vec2 et à l'utiliser pour l'inférence !

### Reconnaissance vocale

Pour utiliser le modèle préentraîné pour la reconnaissance vocale, ajoutez une tête de modélisation du langage au-dessus du modèle Wav2Vec2 de base pour la [classification temporelle connexionniste (CTC)](glossary#connectionist-temporal-classification-ctc). Cette tête de modélisation du langage est une couche linéaire qui prend les états cachés (*hidden states*) de l'encodeur et les convertit en logits. Chaque logit correspond à une classe de token (le nombre de tokens provient du vocabulaire de la tâche). La perte CTC est calculée entre les logits et les cibles (*targets*) pour identifier la séquence de tokens la plus probable, qui est ensuite décodée en transcription.

Prêt à vous lancer dans la reconnaissance automatique de la parole ? Consultez notre [guide complet de reconnaissance automatique de la parole](tasks/asr) pour apprendre à ajuster Wav2Vec2 et à l'utiliser pour l'inférence !

## Vision par ordinateur

Il existe deux façons d'aborder les tâches de vision par ordinateur :

1. **Diviser une image en une séquence de patches** et les traiter en parallèle avec un Transformer.
2. **Utiliser un CNN moderne**, comme [ConvNeXT](model_doc/convnext), qui repose sur des couches convolutionnelles mais adopte des conceptions de réseau modernes.

<Tip>

Une troisième approche combine les Transformers avec des convolutions (par exemple, [Convolutional Vision Transformer](model_doc/cvt) ou [LeViT](model_doc/levit)). Nous ne discuterons pas de ces approches ici, car elles mélangent simplement les deux approches que nous examinons.

</Tip>

ViT et ConvNeXT sont couramment utilisés pour la classification d'images. Pour d'autres tâches de vision par ordinateur comme la détection d'objets, la segmentation et l'estimation de la profondeur, nous examinerons respectivement DETR, Mask2Former et GLPN, qui sont mieux adaptés à ces tâches.

### Classification d'images

ViT et ConvNeXT peuvent tous deux être utilisés pour la classification d'images ; la principale différence réside dans leurs approches : ViT utilise un mécanisme d'attention tandis que ConvNeXT repose sur des convolutions.

#### Transformer

[ViT](model_doc/vit) remplace entièrement les convolutions par une architecture Transformer pure. Si vous êtes déjà familiarisé avec le Transformer original, vous trouverez que ViT suit des principes similaires, mais adaptés pour traiter les images comme des séquences de patches.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/vit_architecture.jpg"/>
</div>

Le principal changement introduit par ViT concerne la façon dont les images sont fournies à un Transformer :

1. **Tokenisation des images** : L'image est divisée en patches carrés non chevauchants, chacun étant transformé en un vecteur ou *embedding de patch*. Ces embeddings de patch sont générés à partir d'une couche convolutionnelle 2D pour adapter les dimensions d'entrée (par exemple, 768 valeurs pour chaque embedding de patch). Si vous avez une image de 224x224 pixels, elle peut être divisée en 196 patches de 16x16 pixels. Ainsi, une image est "tokenisée" en une séquence de patches.

2. **Token `[CLS]`** : Un *embedding apprenables* spécial, appelé token `[CLS]`, est ajouté au début des embeddings de patch, similaire à BERT. L'état caché final du token `[CLS]` est utilisé comme entrée pour la tête de classification attachée, tandis que les autres sorties sont ignorées. Ce token aide le modèle à encoder une représentation globale de l'image.

3. **Embeddings de position** : Pour que le modèle comprenne l'ordre des patches, des *embeddings de position* sont ajoutés aux embeddings de patch. Ces embeddings de position, également apprenables et de la même taille que les embeddings de patch, permettent au modèle de saisir la structure spatiale de l'image.

4. **Classification** : Les embeddings, enrichis des embeddings de position, sont ensuite traités par l'encodeur Transformer. La sortie associée au token `[CLS]` est passée à une tête de perceptron multicouche (MLP) pour la classification. La tête MLP convertit cette sortie en logits pour chaque étiquette de classe, et la perte d'entropie croisée est calculée pour déterminer la classe la plus probable.

Prêt à vous essayer à la classification d'images ? Consultez notre [guide complet de classification d'images](tasks/image_classification) pour apprendre à ajuster ViT et à l'utiliser pour l'inférence !

#### CNN

<Tip>

Cette section explique brièvement les convolutions, mais il serait utile d'avoir une compréhension préalable de la façon dont elles modifient la forme et la taille d'une image. Si vous n'êtes pas familier avec les convolutions, consultez le [chapitre sur les réseaux de neurones convolutionnels](https://github.com/fastai/fastbook/blob/master/13_convolutions.ipynb) du livre fastai !

</Tip>

[ConvNeXT](model_doc/convnext) est une architecture CNN qui adopte des conceptions de réseau modernes pour améliorer les performances. Cependant, les convolutions restent au cœur du modèle. D'un point de vue général, une [convolution](glossary#convolution) est une opération où une matrice plus petite (*noyau*) est multipliée par une petite fenêtre de pixels de l'image. Elle calcule certaines caractéristiques à partir de cette fenêtre, comme une texture particulière ou la courbure d'une ligne. Ensuite, elle se déplace vers la fenêtre suivante de pixels ; la distance parcourue par la convolution est appelée le *stride*.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/convolution.gif"/>
</div>

<small>Une convolution de base sans padding ni stride, tirée de <a href="https://arxiv.org/abs/1603.07285">Un guide des calculs de convolution pour l'apprentissage profond.</a></small>

Vous pouvez alimenter la sortie d'une couche convolutionnelle à une autre couche convolutionnelle. À chaque couche successive, le réseau apprend des caractéristiques de plus en plus complexes et abstraites, telles que des objets spécifiques comme des hot-dogs ou des fusées. Entre les couches convolutionnelles, il est courant d'ajouter des couches de pooling pour réduire la dimensionnalité et rendre le modèle plus robuste aux variations de position des caractéristiques.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/convnext_architecture.png"/>
</div>

ConvNeXT modernise un CNN de cinq manières :

1. **Modification du nombre de blocs** : ConvNeXT utilise une approche similaire à ViT en "patchifiant" l'image avec un stride plus grand et une taille de noyau correspondante, divisant ainsi l'image en patches non chevauchants.

2. **Couche de goulot d'étranglement** (*bottleneck layer*) : Cette couche réduit puis restaure le nombre de canaux pour accélérer les convolutions 1x1, permettant une plus grande profondeur du réseau. Un goulot d'étranglement inversé augmente d'abord le nombre de canaux avant de les réduire, optimisant ainsi l'utilisation de la mémoire.

3. **Convolution en profondeur** (*depthwise convolution*): Remplace la convolution 3x3 traditionnelle par une convolution appliquée à chaque canal d'entrée séparément, améliorant ainsi la largeur du réseau et ses performances.

4. **Augmentation de la taille du noyau** : ConvNeXT utilise un noyau de 7x7 pour imiter le champ réceptif global de ViT, ce qui permet de capturer des informations sur une plus grande partie de l'image.

5. **Changements de conception des couches** : Le modèle adopte des modifications inspirées des Transformers, telles que moins de couches d'activation et de normalisation, l'utilisation de GELU au lieu de ReLU, et LayerNorm plutôt que BatchNorm.

La sortie des blocs de convolution est ensuite passée à une tête de classification, qui convertit les sorties en logits et calcule la perte d'entropie croisée pour déterminer l'étiquette la plus probable.

### Object detection

[DETR](model_doc/detr), *DEtection TRansformer*, est un modèle de détection d'objets de bout en bout qui combine un CNN avec un encodeur-décodeur Transformer.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/detr_architecture.png"/>
</div>

Décomposons le fonctionnement de DETR (DEtection TRansformer) pour la détection d'objets :

1. **Extraction des caractéristiques avec le CNN** : Un CNN préentraîné, appelé *backbone*, prend une image et génère une carte de caractéristiques (*feature map*) à basse résolution. Une convolution 1x1 est ensuite appliquée pour réduire la dimensionnalité et créer une nouvelle carte de caractéristiques qui représente des abstractions de plus haut niveau de l'image. Cette dernière est ensuite aplatie en une séquence de vecteurs de caractéristiques, qui sont combinés avec des embeddings positionnels.

2. **Traitement avec l'encodeur et le décodeur** : Les vecteurs de caractéristiques sont passés à l'encodeur, qui apprend les représentations de l'image avec ses couches d'attention. Les états cachés de l'encodeur sont ensuite combinés avec des *objects queries* dans le décodeur. Ces *objects queries* sont des embeddings appris qui se concentrent sur différentes régions de l'image et sont mis à jour à chaque couche d'attention. Les états cachés du décodeur sont utilisés pour prédire les coordonnées de la boîte englobante (*bounding box*) et le label de la classe pour chaque objet query, ou `pas d'objet` si aucun objet n'est détecté.

3. **Perte de correspondance bipartite** : Lors de l'entraînement, DETR utilise une *perte de correspondance bipartite* pour comparer un nombre fixe de prédictions avec un ensemble fixe de labels de vérité terrain. Si le nombre de labels de vérité terrain est inférieur au nombre de *N* labels, ils sont complétés avec une classe `pas d'objet`. Cette fonction de perte encourage DETR à trouver une correspondance un à un entre les prédictions et les labels de vérité terrain. Si les boîtes englobantes ou les labels de classe ne sont pas corrects, une perte est encourue. De même, si DETR prédit un objet inexistant, il est pénalisé. Cela encourage DETR à trouver d'autres objets dans l'image au lieu de se concentrer sur un seul objet très proéminent.

Une tête de détection d'objets est ajoutée au-dessus de DETR pour trouver le label de la classe et les coordonnées de la boîte englobante. Cette tête de détection d'objets comprend deux composants : une couche linéaire pour transformer les états cachés du décodeur en logits sur les labels de classe, et un MLP pour prédire la boîte englobante.

Prêt à essayer la détection d'objets ? Consultez notre guide complet sur la [détection d'objets](tasks/object_detection) pour apprendre à affiner DETR et à l'utiliser pour l'inférence !

### Segmentation d'image

[Mask2Former](model_doc/mask2former) est une architecture polyvalente conçue pour traiter tous les types de tâches de segmentation d'image. Contrairement aux modèles de segmentation traditionnels, qui sont généralement spécialisés dans des sous-tâches spécifiques comme la segmentation d'instances, sémantique ou panoptique, Mask2Former aborde chaque tâche comme un problème de *classification de masques*. Cette approche regroupe les pixels en *N* segments et prédit pour chaque image *N* masques ainsi que leur étiquette de classe correspondante. Dans cette section, nous vous expliquerons le fonctionnement de Mask2Former et vous aurez la possibilité d'effectuer un réglage fin (*fine-tuning*) de SegFormer à la fin.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/mask2former_architecture.png"/>
</div>

Il y a trois composants principaux dans Mask2Former :

1. Un [backbone Swin](model_doc/swin) qui prend une image en entrée et génère une carte de caractéristiques (*feature map*) à basse résolution après trois convolutions successives de 3x3.

2. Cette carte de caractéristiques est ensuite envoyée à un *décodeur de pixels*, qui augmente progressivement la résolution des caractéristiques pour obtenir des embeddings par pixel en haute résolution. Le décodeur de pixels produit des caractéristiques multi-échelles, comprenant des résolutions de 1/32, 1/16, et 1/8 de l'image originale.

3. Les cartes de caractéristiques à différentes échelles sont successivement traitées par une couche de décodeur Transformer, permettant de capturer les petits objets à partir des caractéristiques haute résolution. Le point central de Mask2Former est le mécanisme de *masquage d'attention* dans le décodeur. Contrairement à l'attention croisée, qui peut se concentrer sur l'ensemble de l'image, l'attention masquée se focalise uniquement sur certaines zones spécifiques. Cette approche est plus rapide et améliore les performances en permettant au modèle de se concentrer sur les détails locaux de l'image.

4. À l'instar de [DETR](tasks_explained#object-detection), Mask2Former utilise également des requêtes d'objet apprises, qu'il combine avec les caractéristiques de l'image du décodeur de pixels pour faire des prédictions globales (c'est-à-dire, `étiquette de classe`, `prédiction de masque`). Les états cachés du décodeur sont passés dans une couche linéaire pour être transformés en logits correspondant aux étiquettes de classe. La perte d'entropie croisée est alors calculée entre les logits et l'étiquette de classe pour déterminer la plus probable.

   Les prédictions de masque sont générées en combinant les embeddings de pixels avec les états cachés finaux du décodeur. La perte d'entropie croisée sigmoïde et la perte de Dice sont calculées entre les logits et le masque de vérité terrain pour déterminer le masque le plus probable.

Prêt à vous lancer dans la détection d'objets ? Consultez notre [guide complet sur la segmentation d'image](tasks/semantic_segmentation) pour apprendre à affiner SegFormer et l'utiliser pour l'inférence !

### Estimation de la profondeur

[GLPN](model_doc/glpn), *Global-Local Path Network*, est un Transformer pour l'estimation de profondeur qui combine un encodeur [SegFormer](model_doc/segformer) avec un décodeur léger.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/glpn_architecture.jpg"/>
</div>
1. Comme avec ViT, une image est divisée en une séquence de patches, mais ces patches sont plus petits. Cette approche est particulièrement adaptée aux tâches de prédiction dense telles que la segmentation ou l'estimation de profondeur. Les patches d'image sont transformés en embeddings (voir la section [classification d'image](#image-classification) pour plus de détails sur la création des embeddings), puis envoyés à l'encodeur.

2. L'encodeur traite les embeddings de patches à travers plusieurs blocs d'encodeur. Chaque bloc comprend des couches d'attention et de Mix-FFN, conçues pour fournir des informations positionnelles. À la fin de chaque bloc, une couche de *fusion de patches* crée des représentations hiérarchiques. Les caractéristiques des groupes de patches voisins sont concaténées, et une couche linéaire est appliquée pour réduire le nombre de patches à une résolution de 1/4. Ce processus est répété dans les blocs suivants jusqu'à obtenir des caractéristiques d'image avec des résolutions de 1/8, 1/16, et 1/32.

3. Un décodeur léger prend la dernière carte de caractéristiques (à l'échelle 1/32) de l'encodeur et l'agrandit à l'échelle 1/16. Ensuite, cette caractéristique passe par un module de *Fusion de Caractéristiques Sélective (SFF)*, qui sélectionne et combine les caractéristiques locales et globales à partir d'une carte d'attention pour chaque caractéristique, puis l'agrandit à 1/8. Ce processus est répété jusqu'à ce que les caractéristiques décodées aient la même taille que l'image originale. La sortie est ensuite traitée par deux couches de convolution, suivies d'une activation sigmoïde pour prédire la profondeur de chaque pixel.

## Traitement du langage naturel

Le Transformer a été initialement conçu pour la traduction automatique, et depuis, il est devenu pratiquement l'architecture par défaut pour résoudre toutes les tâches de traitement du langage naturel (NLP). Certaines tâches se prêtent bien à la structure d'encodeur du Transformer, tandis que d'autres sont mieux adaptées au décodeur. D'autres tâches encore utilisent à la fois la structure encodeur-décodeur du Transformer.

### Classification de texte

[BERT](model_doc/bert) est un modèle basé uniquement sur l'encodeur, qui a été le premier à intégrer efficacement la bidirectionnalité profonde pour obtenir des représentations plus riches du texte en tenant compte des mots en amont et en aval.

1. BERT utilise la tokenisation [WordPiece](tokenizer_summary#wordpiece) pour générer des embeddings de tokens à partir du texte. Pour différencier une seule phrase d'une paire de phrases, un token spécial `[SEP]` est ajouté. De plus, un token spécial `[CLS]` est placé au début de chaque séquence de texte. La sortie finale associée au token `[CLS]` est utilisée comme entrée pour la tête de classification des tâches. BERT ajoute également un embedding de segment pour indiquer si un token appartient à la première ou à la deuxième phrase dans une paire.

2. BERT est préentraîné avec deux objectifs : le masquage de mots (masked language modeling) et la prédiction de la phrase suivante. Pour le masquage de mots, un pourcentage des tokens d'entrée est masqué aléatoirement, et le modèle doit prédire ces mots. Cela permet de surmonter le problème de la bidirectionnalité, où le modèle pourrait autrement tricher en voyant tous les mots et en "prédire" le mot suivant. Les états cachés finaux des tokens masqués sont passés à un réseau feedforward avec une fonction softmax sur le vocabulaire pour prédire le mot masqué.

   Le deuxième objectif de préentraînement est la prédiction de la phrase suivante. Le modèle doit déterminer si la phrase B suit la phrase A. Dans la moitié des cas, la phrase B est la phrase suivante, et dans l'autre moitié, elle est aléatoire. Cette prédiction (phrase suivante ou non) est envoyée à un réseau feedforward avec une softmax sur les deux classes (`IsNext` et `NotNext`).

3. Les embeddings d'entrée sont traités par plusieurs couches d'encodeur pour produire des états cachés finaux.

Pour utiliser le modèle préentraîné pour la classification de texte, ajoutez une tête de classification de séquence au-dessus du modèle BERT de base. Cette tête est une couche linéaire qui prend les états cachés finaux et les transforme en logits. La perte d'entropie croisée est ensuite calculée entre les logits et les cibles pour déterminer l'étiquette la plus probable.

Prêt à essayer la classification de texte ? Consultez notre [guide complet sur la classification de texte](tasks/sequence_classification) pour apprendre à effectuer un réglagle fin (*fine-tuning*) de DistilBERT et l'utiliser pour l'inférence !

### Classification de tokens

Pour utiliser BERT dans des tâches de classification de tokens, comme la reconnaissance d'entités nommées (NER), ajoutez une tête de classification de tokens au-dessus du modèle BERT de base. Cette tête est une couche linéaire qui prend les états cachés finaux et les transforme en logits. La perte d'entropie croisée est ensuite calculée entre les logits et les labels de chaque token pour déterminer l'étiquette la plus probable.

Prêt à essayer la classification de tokens ? Consultez notre [guide complet sur la classification de tokens](tasks/token_classification) pour découvrir comment effectuer un réglagle fin (*fine-tuning*) de DistilBERT et l'utiliser pour l'inférence !

### Réponse aux questions - (*Question Answering*)

Pour utiliser BERT pour la réponse aux questions, ajoutez une tête de classification de span au-dessus du modèle BERT de base. Cette tête est une couche linéaire qui transforme les états cachés finaux en logits pour les positions de début et de fin du `span` correspondant à la réponse. La perte d'entropie croisée est calculée entre les logits et les positions réelles pour déterminer le span de texte le plus probable en tant que réponse.

Prêt à essayer la réponse aux questions ? Consultez notre [guide complet sur la réponse aux questions](tasks/question_answering) pour découvrir comment effectuer un réglagle fin (*fine-tuning*) de DistilBERT et l'utiliser pour l'inférence !

<Tip>

💡 Une fois BERT préentraîné, il est incroyablement facile de l’adapter à diverses tâches ! Il vous suffit d’ajouter une tête spécifique au modèle préentraîné pour transformer les états cachés en la sortie souhaitée.

</Tip>

### Génération de texte

[GPT-2](model_doc/gpt2) est un modèle basé uniquement sur le décodeur, préentraîné sur une grande quantité de texte. Il peut générer du texte convaincant (bien que parfois inexact !) à partir d'une invite et accomplir d'autres tâches de NLP, comme la réponse aux questions, même s'il n'a pas été spécifiquement entraîné pour ces tâches.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/gpt2_architecture.png"/>
</div>

1. GPT-2 utilise le [byte pair encoding (BPE)](tokenizer_summary#bytepair-encoding-bpe) pour tokeniser les mots et générer des embeddings de tokens. Des encodages positionnels sont ajoutés pour indiquer la position de chaque token dans la séquence. Les embeddings d'entrée passent à travers plusieurs blocs de décodeur pour produire des états cachés finaux. Chaque bloc de décodeur utilise une couche d'*attention masquée*, ce qui signifie que GPT-2 ne peut pas se concentrer sur les tokens futurs et est uniquement autorisé à se focaliser sur les tokens à gauche dans le texte. Cela diffère du token [`mask`] de BERT, car ici, dans l'attention masquée, un masque d'attention est utilisé pour attribuer un score de `0` aux tokens futurs.

2. La sortie du décodeur est ensuite envoyée à une tête de modélisation du langage, qui effectue une transformation linéaire pour convertir les états cachés en logits. L'étiquette est le token suivant dans la séquence, obtenue en décalant les logits vers la droite d'une position. La perte d'entropie croisée est calculée entre les logits décalés et les étiquettes pour déterminer le token suivant le plus probable.

L'objectif de préentraînement de GPT-2 est basé sur la [modélisation du langage causale](glossary#causal-language-modeling), qui consiste à prédire le mot suivant dans une séquence. Cette approche rend GPT-2 particulièrement efficace pour les tâches de génération de texte.

Prêt à essayer la génération de texte ? Consultez notre [guide complet sur la modélisation du langage causale](tasks/language_modeling#causal-language-modeling) pour découvrir comment effectuer un réglagle fin (*fine-tuning*) de DistilGPT-2 et l'utiliser pour l'inférence !

<Tip>

Pour plus d'informations sur la génération de texte, consultez le guide sur les [stratégies de génération de texte](generation_strategies) !

</Tip>

### Résumé de texte

Les modèles encodeur-décodeur tels que [BART](model_doc/bart) et [T5](model_doc/t5) sont conçus pour les tâches de résumé en mode séquence-à-séquence. Dans cette section, nous expliquerons le fonctionnement de BART, puis vous aurez l'occasion de découvrir comment réaliser un réglagle fin (*fine-tuning*) de T5.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bart_architecture.png"/>
</div>

1. L'architecture de l'encodeur de BART est très similaire à celle de BERT, acceptant des embeddings de tokens et des embeddings positionnels du texte. BART est préentraîné en corrompant l'entrée et en la reconstruisant avec le décodeur. Contrairement à d'autres encodeurs utilisant des stratégies de corruption spécifiques, BART peut appliquer divers types de corruption, parmi lesquelles la stratégie de *text infilling* est la plus efficace. Dans le text infilling, plusieurs segments de texte sont remplacés par un **seul** token [`mask`]. Cette approche est cruciale car elle force le modèle à prédire les tokens masqués et à estimer le nombre de tokens manquants. Les embeddings d'entrée et les spans masqués sont passés à l'encodeur pour produire des états cachés finaux. Contrairement à BERT, BART ne comporte pas de réseau feedforward final pour prédire un mot.

2. La sortie de l'encodeur est transmise au décodeur, qui doit prédire à la fois les tokens masqués et les tokens non corrompus. Ce contexte supplémentaire aide le décodeur à restaurer le texte original. La sortie du décodeur est ensuite envoyée à une tête de modélisation du langage, qui transforme les états cachés en logits. La perte d'entropie croisée est calculée entre les logits et l'étiquette, qui est simplement le token décalé vers la droite.

Prêt à essayer le résumé ? Consultez notre [guide complet sur le résumé](tasks/summarization) pour apprendre à effectuer un réglage fin (*fine-tuning*) de T5 et l'utiliser pour l'inférence !

<Tip>

Pour plus d'informations sur la génération de texte, consultez le guide sur les [stratégies de génération de texte](generation_strategies) !

</Tip>

### Traduction

La traduction est un autre exemple de tâche séquence-à-séquence, ce qui signifie qu'un modèle encodeur-décodeur comme [BART](model_doc/bart) ou [T5](model_doc/t5) peut être utilisé pour cette tâche. Nous expliquerons ici comment BART fonctionne pour la traduction, puis vous pourrez découvrir comment affiner T5.

BART adapte le modèle à la traduction en ajoutant un encodeur séparé, initialisé aléatoirement, pour mapper la langue source en une entrée qui peut être décodée dans la langue cible. Les embeddings de cet encodeur sont ensuite passés à l'encodeur préentraîné au lieu des embeddings de mots originaux. L'encodeur source est entraîné en mettant à jour l'encodeur source, les embeddings positionnels et les embeddings d'entrée avec la perte d'entropie croisée provenant de la sortie du modèle. Les paramètres du modèle sont figés lors de cette première étape, et tous les paramètres du modèle sont entraînés ensemble lors de la deuxième étape.

BART a été suivi par une version multilingue, mBART, qui est spécifiquement conçue pour la traduction et préentraînée sur de nombreuses langues différentes.

Prêt à essayer la traduction ? Consultez notre [guide complet sur la traduction](tasks/translation) pour apprendre à affiner T5 et l'utiliser pour l'inférence !

<Tip>

Pour plus d'informations sur la génération de texte, consultez le guide sur les [stratégies de génération de texte](generation_strategies) !

</Tip>
