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

# Decision Transformer

## Overview

Decision Transformer モデルは、[Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/abs/2106.01345) で提案されました。
Lili Chen, Kevin Lu, Aravind Rajeswaran, Kimin Lee, Aditya Grover, Michael Laskin, Pieter Abbeel, Aravind Srinivas, Igor Mordatch.

論文の要約は次のとおりです。

_強化学習（RL）をシーケンスモデリング問題として抽象化するフレームワークを紹介します。
これにより、Transformer アーキテクチャのシンプルさとスケーラビリティ、および関連する進歩を活用できるようになります。
GPT-x や BERT などの言語モデリングで。特に、Decision Transformer というアーキテクチャを紹介します。
RL の問題を条件付きシーケンス モデリングとして投げかけます。値関数に適合する以前の RL アプローチとは異なり、
ポリシー勾配を計算すると、Decision Transformer は因果的にマスクされたアルゴリズムを利用して最適なアクションを出力するだけです。
変成器。望ましいリターン (報酬)、過去の状態、アクションに基づいて自己回帰モデルを条件付けすることにより、
Decision Transformer モデルは、望ましいリターンを達成する将来のアクションを生成できます。そのシンプルさにも関わらず、
Decision Transformer は、最先端のモデルフリーのオフライン RL ベースラインのパフォーマンスと同等、またはそれを超えています。
Atari、OpenAI Gym、Key-to-Door タスク_

このバージョンのモデルは、状態がベクトルであるタスク用です。

このモデルは、[edbeeching](https://huggingface.co/edbeeching) によって提供されました。元のコードは [ここ](https://github.com/kzl/decision-transformer) にあります。

## DecisionTransformerConfig

[[autodoc]] DecisionTransformerConfig

## DecisionTransformerGPT2Model

[[autodoc]] DecisionTransformerGPT2Model - forward

## DecisionTransformerModel

[[autodoc]] DecisionTransformerModel - forward
