<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# IRIS

## Overview

The IRIS model was proposed in [TRANSFORMERS ARE SAMPLE-EFFICIENT WORLD MODELS](https://arxiv.org/abs/2209.00588) by Vincent Micheli, Eloi Alonso, François Fleuret.

The abstract from the paper is the following:

*Deep reinforcement learning agents are notoriously sample inefficient, which considerably limits their application to real-world problems. Recently, many model-based methods have been designed to address this issue, with learning in the imagination of a world model being one of the most prominent approaches. However, while virtually unlimited interaction with a simulated environment sounds appealing, the world model has to be accurate over extended periods of time. Motivated by the success of Transformers in sequence modeling tasks, we introduce IRIS, a data-efficient agent that learns in a world model composed of a discrete autoencoder and an autoregressive Transformer. With the equivalent of only two hours of gameplay in the Atari 100k benchmark, IRIS achieves a mean human normalized score of 1.046, and outperforms humans on 10 out of 26 games, setting a new state of the art for methods without lookahead search.*

Tips:

The hugging face version of the model provides the architecture of the original model for training and inference with an exact same output with precision of more 1e-3. The batch provided for training and inference is same as given in the original model, i.e., [observations,actions,rewards,ends, mask_padding]. It depends on you if you want to use the same method of data collection from env as provided in the original code or use or your own as the this version supports both. The env are Atari environments to train in as in the original code. If you want to use it, provide the batch consisting of [observations(image frames),actions,rewards,ends](from the env) & mask padding.

This model was contributed by [ruffy369](https://huggingface.co/ruffy369). The original code can be found [here](https://github.com/eloialonso/iris).

## IrisConfig

[[autodoc]] IrisConfig

## IrisModel

[[autodoc]] IrisModel
    - forward
