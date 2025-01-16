<!--Copyright 2021 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# DeiT

## Overview

DeiT モデルは、Hugo Touvron、Matthieu Cord、Matthijs Douze、Francisco Massa、Alexandre
Sablayrolles, Hervé Jégou.によって [Training data-efficient image Transformers & distillation through attention](https://arxiv.org/abs/2012.12877) で提案されました。
サブレイロール、エルヴェ・ジェグー。 [Dosovitskiy et al., 2020](https://arxiv.org/abs/2010.11929) で紹介された [Vision Transformer (ViT)](vit) は、既存の畳み込みニューラルと同等、またはそれを上回るパフォーマンスを発揮できることを示しました。
Transformer エンコーダ (BERT のような) を使用したネットワーク。ただし、その論文で紹介された ViT モデルには、次のトレーニングが必要でした。
外部データを使用して、数週間にわたる高価なインフラストラクチャ。 DeiT (データ効率の高い画像変換器) はさらに優れています
画像分類用に効率的にトレーニングされたトランスフォーマーにより、必要なデータとコンピューティング リソースがはるかに少なくなります。
オリジナルの ViT モデルとの比較。

論文の要約は次のとおりです。

*最近、純粋に注意に基づくニューラル ネットワークが、画像などの画像理解タスクに対処できることが示されました。
分類。ただし、これらのビジュアル トランスフォーマーは、
インフラストラクチャが高価であるため、その採用が制限されています。この作業では、コンボリューションフリーの競争力のあるゲームを作成します。
Imagenet のみでトレーニングしてトランスフォーマーを作成します。 1 台のコンピューターで 3 日以内にトレーニングを行います。私たちの基準となるビジョン
トランス (86M パラメータ) は、外部なしで ImageNet 上で 83.1% (単一クロップ評価) のトップ 1 の精度を達成します。
データ。さらに重要なのは、トランスフォーマーに特有の教師と生徒の戦略を導入することです。蒸留に依存している
学生が注意を払って教師から学ぶことを保証するトークン。私たちはこのトークンベースに興味を示します
特に convnet を教師として使用する場合。これにより、convnet と競合する結果を報告できるようになります。
Imagenet (最大 85.2% の精度が得られます) と他のタスクに転送するときの両方で。私たちはコードを共有し、
モデル。*

このモデルは、[nielsr](https://huggingface.co/nielsr) によって提供されました。このモデルの TensorFlow バージョンは、[amyeroberts](https://huggingface.co/amyeroberts) によって追加されました。

## Usage tips

- ViT と比較して、DeiT モデルはいわゆる蒸留トークンを使用して教師から効果的に学習します (これは、
  DeiT 論文は、ResNet のようなモデルです)。蒸留トークンは、バックプロパゲーションを通じて、と対話することによって学習されます。
  セルフアテンション層を介したクラス ([CLS]) とパッチ トークン。
- 抽出されたモデルを微調整するには 2 つの方法があります。(1) 上部に予測ヘッドを配置するだけの古典的な方法。
  クラス トークンの最終的な非表示状態を抽出し、蒸留シグナルを使用しない、または (2) 両方の
  予測ヘッドはクラス トークンの上と蒸留トークンの上にあります。その場合、[CLS] 予測は
  head は、head の予測とグラウンド トゥルース ラベル間の通常のクロスエントロピーを使用してトレーニングされます。
  蒸留予測ヘッドは、硬蒸留 (予測と予測の間のクロスエントロピー) を使用してトレーニングされます。
  蒸留ヘッドと教師が予測したラベル）。推論時に、平均予測を取得します。
  最終的な予測として両頭の間で。 (2) は「蒸留による微調整」とも呼ばれます。
  下流のデータセットですでに微調整されている教師。モデル的には (1) に相当します。
  [`DeiTForImageClassification`] と (2) に対応します。
  [`DeiTForImageClassificationWithTeacher`]。
- 著者らは (2) についてもソフト蒸留を試みたことに注意してください (この場合、蒸留予測ヘッドは
  教師のソフトマックス出力に一致するように KL ダイバージェンスを使用してトレーニングしました）が、ハード蒸留が最良の結果をもたらしました。
- リリースされたすべてのチェックポイントは、ImageNet-1k のみで事前トレーニングおよび微調整されました。外部データは使用されませんでした。これは
  JFT-300M データセット/Imagenet-21k などの外部データを使用した元の ViT モデルとは対照的です。
  事前トレーニング。
- DeiT の作者は、より効率的にトレーニングされた ViT モデルもリリースしました。これは、直接プラグインできます。
  [`ViTModel`] または [`ViTForImageClassification`]。データなどのテクニック
  はるかに大規模なデータセットでのトレーニングをシミュレートするために、拡張、最適化、正則化が使用されました。
  (ただし、事前トレーニングには ImageNet-1k のみを使用します)。 4 つのバリエーション (3 つの異なるサイズ) が利用可能です。
  *facebook/deit-tiny-patch16-224*、*facebook/deit-small-patch16-224*、*facebook/deit-base-patch16-224* および
  *facebook/deit-base-patch16-384*。以下を行うには [`DeiTImageProcessor`] を使用する必要があることに注意してください。
  モデル用の画像を準備します。

## Resources

DeiT を始めるのに役立つ公式 Hugging Face およびコミュニティ (🌎 で示されている) リソースのリスト。

<PipelineTag pipeline="image-classification"/>

- [`DeiTForImageClassification`] は、この [サンプル スクリプト](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification) および [ノートブック](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb)。
- 参照: [画像分類タスク ガイド](../tasks/image_classification)

それに加えて:

- [`DeiTForMaskedImageModeling`] は、この [サンプル スクリプト](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-pretraining) でサポートされています。

ここに含めるリソースの送信に興味がある場合は、お気軽にプル リクエストを開いてください。審査させていただきます。リソースは、既存のリソースを複製するのではなく、何か新しいものを示すことが理想的です。

## DeiTConfig

[[autodoc]] DeiTConfig

## DeiTFeatureExtractor

[[autodoc]] DeiTFeatureExtractor
    - __call__

## DeiTImageProcessor

[[autodoc]] DeiTImageProcessor
    - preprocess

## DeiTImageProcessorFast

[[autodoc]] DeiTImageProcessorFast
    - preprocess

<frameworkcontent>
<pt>

## DeiTModel

[[autodoc]] DeiTModel
    - forward

## DeiTForMaskedImageModeling

[[autodoc]] DeiTForMaskedImageModeling
    - forward

## DeiTForImageClassification

[[autodoc]] DeiTForImageClassification
    - forward

## DeiTForImageClassificationWithTeacher

[[autodoc]] DeiTForImageClassificationWithTeacher
    - forward

</pt>
<tf>

## TFDeiTModel

[[autodoc]] TFDeiTModel
    - call

## TFDeiTForMaskedImageModeling

[[autodoc]] TFDeiTForMaskedImageModeling
    - call

## TFDeiTForImageClassification

[[autodoc]] TFDeiTForImageClassification
    - call

## TFDeiTForImageClassificationWithTeacher

[[autodoc]] TFDeiTForImageClassificationWithTeacher
    - call

</tf>
</frameworkcontent>