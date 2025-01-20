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

# KOSMOS-2.5

## Overview

Kosmos-2.5 is a multimodal literate model for machine reading of text-intensive images. Pre-trained on large-scale text-intensive images, Kosmos-2.5 excels in two distinct yet cooperative transcription tasks: (1) generating spatially-aware text blocks, where each block of text is assigned its spatial coordinates within the image, and (2) producing structured text output that captures styles and structures into the markdown format. This unified multimodal literate capability is achieved through a shared decoder-only auto-regressive Transformer architecture, task-specific prompts, and flexible text representations. We evaluate Kosmos-2.5 on end-to-end document-level text recognition and image-to-markdown text generation. Furthermore, the model can be readily adapted for any text-intensive image understanding task with different prompts through supervised fine-tuning, making it a general-purpose tool for real-world applications involving text-rich images. This work also paves the way for the future scaling of multimodal large language models.

The abstract from the paper is the following:

*We present Kosmos-2.5, a multimodal literate model for machine reading of text-intensive images. Pre-trained on large-scale text-intensive images, Kosmos-2.5 excels in two distinct yet cooperative transcription tasks: (1) generating spatially-aware text blocks, where each block of text is assigned its spatial coordinates within the image, and (2) producing structured text output that captures styles and structures into the markdown format. This unified multimodal literate capability is achieved through a shared Transformer architecture, task-specific prompts, and flexible text representations. We evaluate Kosmos-2.5 on end-to-end document-level text recognition and image-to-markdown text generation. Furthermore, the model can be readily adapted for any text-intensive image understanding task with different prompts through supervised fine-tuning, making it a general-purpose tool for real-world applications involving text-rich images. This work also paves the way for the future scaling of multimodal large language models.*

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/kosmos2_5_ocr.png"
alt="drawing" width="600"/>

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/kosmos2_5_md.png"
alt="drawing" width="600"/>

<small> Overview of tasks that KOSMOS-2.5 can handle. Taken from the <a href="https://arxiv.org/abs/2309.11419">original paper</a>. </small>

## Example
**Markdown Task:** For usage instructions, please refer to [md.py](https://huggingface.co/microsoft/kosmos-2.5/blob/main/md.py).

**OCR Task:** For usage instructions, please refer to [ocr.py](https://huggingface.co/microsoft/kosmos-2.5/blob/main/ocr.py).



## Kosmos2_5Config

[[autodoc]] Kosmos2_5Config

## Kosmos2_5ImageProcessor

[[autodoc]] Kosmos2_5ImageProcessor

## Kosmos2_5Processor

[[autodoc]] Kosmos2_5Processor
    - __call__

## Kosmos2_5Model

[[autodoc]] Kosmos2_5Model
    - forward

## Kosmos2_5ForConditionalGeneration

[[autodoc]] Kosmos2_5ForConditionalGeneration
    - forward
