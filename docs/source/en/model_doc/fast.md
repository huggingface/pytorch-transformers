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

# FAST

## Overview

Fast model proposes an accurate and efficient scene text detection framework, termed FAST (i.e., faster 
arbitrarily-shaped text detector). 

FAST has two new designs. (1) We design a minimalist kernel representation (only has 1-channel output) to model text 
with arbitrary shape, as well as a GPU-parallel post-processing to efficiently assemble text lines with a negligible 
time overhead. (2) We search the network architecture tailored for text detection, leading to more powerful features 
than most networks that are searched for image classification.

## FastConfig

[[autodoc]] FastConfig

## FastImageProcessor

[[autodoc]] FastImageProcessor

## FastForSceneTextRecognition

[[autodoc]] FastForSceneTextRecognition
- forward

## FASTForImageCaptioningOutput

[[autodoc]] FASTForImageCaptioningOutput
- forward



