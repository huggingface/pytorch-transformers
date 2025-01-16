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

# BLIP

## Overview

BLIP モデルは、[BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation](https://arxiv.org/abs/2201.12086) で Junnan Li、Dongxu Li、Caiming Xiong、Steven Hoi によって提案されました。 。

BLIP は、次のようなさまざまなマルチモーダル タスクを実行できるモデルです。
- 視覚的な質問応答
- 画像とテキストの検索（画像とテキストのマッチング）
- 画像キャプション

論文の要約は次のとおりです。

*視覚言語事前トレーニング (VLP) により、多くの視覚言語タスクのパフォーマンスが向上しました。
ただし、既存の事前トレーニング済みモデルのほとんどは、理解ベースのタスクまたは世代ベースのタスクのいずれかでのみ優れています。さらに、最適ではない監視ソースである Web から収集されたノイズの多い画像とテキストのペアを使用してデータセットをスケールアップすることで、パフォーマンスの向上が大幅に達成されました。この論文では、視覚言語の理解と生成タスクの両方に柔軟に移行する新しい VLP フレームワークである BLIP を提案します。 BLIP は、キャプションをブートストラップすることでノイズの多い Web データを効果的に利用します。キャプショナーが合成キャプションを生成し、フィルターがノイズの多いキャプションを除去します。画像テキスト検索 (平均再現率 +2.7%@1)、画像キャプション作成 (CIDEr で +2.8%)、VQA ( VQA スコアは +1.6%)。 BLIP は、ゼロショット方式でビデオ言語タスクに直接転送した場合にも、強力な一般化能力を発揮します。コード、モデル、データセットがリリースされています。*

![BLIP.gif](https://cdn-uploads.huggingface.co/production/uploads/1670928184033-62441d1d9fdefb55a0b7d12c.gif)

このモデルは [ybelkada](https://huggingface.co/ybelkada) によって提供されました。
元のコードは [ここ](https://github.com/salesforce/BLIP) にあります。

## Resources

- [Jupyter ノートブック](https://github.com/huggingface/notebooks/blob/main/examples/image_captioning_blip.ipynb) カスタム データセットの画像キャプション用に BLIP を微調整する方法

## BlipConfig

[[autodoc]] BlipConfig
    - from_text_vision_configs

## BlipTextConfig

[[autodoc]] BlipTextConfig

## BlipVisionConfig

[[autodoc]] BlipVisionConfig

## BlipProcessor

[[autodoc]] BlipProcessor

## BlipImageProcessor

[[autodoc]] BlipImageProcessor
    - preprocess

## BlipImageProcessorFast

[[autodoc]] BlipImageProcessorFast
    - preprocess

<frameworkcontent>
<pt>

## BlipModel

[[autodoc]] BlipModel
    - forward
    - get_text_features
    - get_image_features

## BlipTextModel

[[autodoc]] BlipTextModel
    - forward

## BlipVisionModel

[[autodoc]] BlipVisionModel
    - forward

## BlipForConditionalGeneration

[[autodoc]] BlipForConditionalGeneration
    - forward

## BlipForImageTextRetrieval

[[autodoc]] BlipForImageTextRetrieval
    - forward

## BlipForQuestionAnswering

[[autodoc]] BlipForQuestionAnswering
    - forward

</pt>
<tf>

## TFBlipModel

[[autodoc]] TFBlipModel
    - call
    - get_text_features
    - get_image_features

## TFBlipTextModel

[[autodoc]] TFBlipTextModel
    - call

## TFBlipVisionModel

[[autodoc]] TFBlipVisionModel
    - call

## TFBlipForConditionalGeneration

[[autodoc]] TFBlipForConditionalGeneration
    - call

## TFBlipForImageTextRetrieval

[[autodoc]] TFBlipForImageTextRetrieval
    - call

## TFBlipForQuestionAnswering

[[autodoc]] TFBlipForQuestionAnswering
    - call
</tf>
</frameworkcontent>