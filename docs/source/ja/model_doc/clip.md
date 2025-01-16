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

# CLIP

## Overview

CLIP モデルは、Alec Radford、Jong Wook Kim、Chris Hallacy、Aditya Ramesh、Gabriel Goh Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) で提案されました。
サンディニ・アガルワル、ギリッシュ・サストリー、アマンダ・アスケル、パメラ・ミシュキン、ジャック・クラーク、グレッチェン・クルーガー、イリヤ・サツケヴァー。クリップ
(Contrastive Language-Image Pre-Training) は、さまざまな (画像、テキスト) ペアでトレーニングされたニューラル ネットワークです。かもね
直接最適化することなく、与えられた画像から最も関連性の高いテキスト スニペットを予測するように自然言語で指示されます。
GPT-2 および 3 のゼロショット機能と同様に、タスクに対して。

論文の要約は次のとおりです。

*最先端のコンピューター ビジョン システムは、あらかじめ定められたオブジェクト カテゴリの固定セットを予測するようにトレーニングされています。これ
制限された形式の監視では、指定するために追加のラベル付きデータが必要となるため、一般性と使いやすさが制限されます。
その他の視覚的なコンセプト。画像に関する生のテキストから直接学習することは、
より広範な監督源。どのキャプションが表示されるかを予測するという単純な事前トレーニング タスクが有効であることを示します。
400 のデータセットで SOTA 画像表現を最初から学習するための効率的かつスケーラブルな方法はどの画像ですか
インターネットから収集された数百万の（画像、テキスト）ペア。事前トレーニング後、自然言語を使用して参照します。
視覚的な概念を学習し（または新しい概念を説明し）、下流のタスクへのモデルのゼロショット転送を可能にします。私たちは勉強します
30 を超えるさまざまな既存のコンピューター ビジョン データセットでタスクをまたがってベンチマークを行うことにより、このアプローチのパフォーマンスを評価します。
OCR、ビデオ内のアクション認識、地理的位置特定、およびさまざまな種類のきめ細かいオブジェクト分類など。の
モデルはほとんどのタスクに簡単に移行でき、多くの場合、必要がなくても完全に監視されたベースラインと競合します。
データセット固有のトレーニングに適しています。たとえば、ImageNet ゼロショットではオリジナルの ResNet-50 の精度と一致します。
トレーニングに使用された 128 万のトレーニング サンプルを使用する必要はありません。コードをリリースし、事前トレーニング済み
モデルの重みはこの https URL で確認できます。*

このモデルは [valhalla](https://huggingface.co/valhalla) によって提供されました。元のコードは [ここ](https://github.com/openai/CLIP) にあります。

## Usage tips and example

CLIP は、マルチモーダルなビジョンおよび言語モデルです。画像とテキストの類似性やゼロショット画像に使用できます。
分類。 CLIP は、ViT のようなトランスフォーマーを使用して視覚的特徴を取得し、因果言語モデルを使用してテキストを取得します
特徴。次に、テキストと視覚の両方の特徴が、同じ次元の潜在空間に投影されます。ドット
投影された画像とテキストの特徴間の積が同様のスコアとして使用されます。

画像を Transformer エンコーダに供給するために、各画像は固定サイズの重複しないパッチのシーケンスに分割されます。
これらは線形に埋め込まれます。 [CLS] トークンは、イメージ全体の表現として機能するために追加されます。作家たち
また、絶対位置埋め込みを追加し、結果として得られるベクトルのシーケンスを標準の Transformer エンコーダに供給します。
[`CLIPImageProcessor`] を使用して、モデルの画像のサイズ変更 (または再スケール) および正規化を行うことができます。

[`CLIPTokenizer`] はテキストのエンコードに使用されます。 [`CLIPProcessor`] はラップします
[`CLIPImageProcessor`] と [`CLIPTokenizer`] を両方の単一インスタンスに統合
テキストをエンコードして画像を準備します。次の例は、次のメソッドを使用して画像とテキストの類似性スコアを取得する方法を示しています。
[`CLIPProcessor`] と [`CLIPModel`]。

```python
>>> from PIL import Image
>>> import requests

>>> from transformers import CLIPProcessor, CLIPModel

>>> model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
>>> processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

>>> outputs = model(**inputs)
>>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
>>> probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
```

## Resources

CLIP を使い始めるのに役立つ公式 Hugging Face およびコミュニティ (🌎 で示されている) リソースのリスト。

- [リモート センシング (衛星) 画像とキャプションを使用した CLIP の微調整](https://huggingface.co/blog/fine-tune-clip-rsicd)、[RSICD データセット] を使用して CLIP を微調整する方法に関するブログ投稿(https://github.com/201528014227051/RSICD_optimal) と、データ拡張によるパフォーマンスの変化の比較。
- この [サンプル スクリプト](https://github.com/huggingface/transformers/tree/main/examples/pytorch/contrastive-image-text) は、プレ- [COCO データセット](https://cocodataset.org/#home) を使用してトレーニングされたビジョンおよびテキスト エンコーダー。

<PipelineTag pipeline="image-to-text"/>

- 画像キャプションのビーム検索による推論に事前トレーニング済み CLIP を使用する方法に関する [ノートブック](https://colab.research.google.com/drive/1tuoAC5F4sC7qid56Z0ap-stR3rwdk0ZV?usp=sharing)。 🌎

**画像検索**

- 事前トレーニングされた CLIP を使用した画像検索と MRR (平均相互ランク) スコアの計算に関する [ノートブック](https://colab.research.google.com/drive/1bLVwVKpAndpEDHqjzxVPr_9nGrSbuOQd?usp=sharing)。 🌎
- 画像の取得と類似性スコアの表示に関する [ノートブック](https://colab.research.google.com/github/deep-diver/image_search_with_natural_language/blob/main/notebooks/Image_Search_CLIP.ipynb)。 🌎
- 多言語 CLIP を使用して画像とテキストを同じベクトル空間にマッピングする方法に関する [ノートブック](https://colab.research.google.com/drive/1xO-wC_m_GNzgjIBQ4a4znvQkvDoZJvH4?usp=sharing)。 🌎
- を使用してセマンティック イメージ検索で CLIP を実行する方法に関する [ノートブック](https://colab.research.google.com/github/vivien000/clip-demo/blob/master/clip.ipynb#scrollTo=uzdFhRGqiWkR) [Unsplash](https://unsplash.com) および [TMDB](https://www.themoviedb.org/) データセット。 🌎

**説明可能性**

- 入力トークンと画像セグメントの類似性を視覚化する方法に関する [ノートブック](https://colab.research.google.com/github/hila-chefer/Transformer-MM-Explainability/blob/main/CLIP_explainability.ipynb)。 🌎

ここに含めるリソースの送信に興味がある場合は、お気軽にプル リクエストを開いてください。審査させていただきます。
リソースは、既存のリソースを複製するのではなく、何か新しいものを示すことが理想的です。

## CLIPConfig

[[autodoc]] CLIPConfig
    - from_text_vision_configs

## CLIPTextConfig

[[autodoc]] CLIPTextConfig

## CLIPVisionConfig

[[autodoc]] CLIPVisionConfig

## CLIPTokenizer

[[autodoc]] CLIPTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## CLIPTokenizerFast

[[autodoc]] CLIPTokenizerFast

## CLIPImageProcessor

[[autodoc]] CLIPImageProcessor
    - preprocess

## CLIPImageProcessorFast

[[autodoc]] CLIPImageProcessorFast
    - preprocess

## CLIPFeatureExtractor

[[autodoc]] CLIPFeatureExtractor

## CLIPProcessor

[[autodoc]] CLIPProcessor

<frameworkcontent>
<pt>

## CLIPModel

[[autodoc]] CLIPModel
    - forward
    - get_text_features
    - get_image_features

## CLIPTextModel

[[autodoc]] CLIPTextModel
    - forward

## CLIPTextModelWithProjection

[[autodoc]] CLIPTextModelWithProjection
    - forward

## CLIPVisionModelWithProjection

[[autodoc]] CLIPVisionModelWithProjection
    - forward

## CLIPVisionModel

[[autodoc]] CLIPVisionModel
    - forward

</pt>
<tf>

## TFCLIPModel

[[autodoc]] TFCLIPModel
    - call
    - get_text_features
    - get_image_features

## TFCLIPTextModel

[[autodoc]] TFCLIPTextModel
    - call

## TFCLIPVisionModel

[[autodoc]] TFCLIPVisionModel
    - call

</tf>
<jax>

## FlaxCLIPModel

[[autodoc]] FlaxCLIPModel
    - __call__
    - get_text_features
    - get_image_features

## FlaxCLIPTextModel

[[autodoc]] FlaxCLIPTextModel
    - __call__

## FlaxCLIPTextModelWithProjection

[[autodoc]] FlaxCLIPTextModelWithProjection
    - __call__

## FlaxCLIPVisionModel

[[autodoc]] FlaxCLIPVisionModel
    - __call__

</jax>
</frameworkcontent>
