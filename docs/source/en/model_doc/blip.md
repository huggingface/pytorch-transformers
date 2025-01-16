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

The BLIP model was proposed in [BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation](https://arxiv.org/abs/2201.12086) by Junnan Li, Dongxu Li, Caiming Xiong, Steven Hoi.

BLIP is a model that is able to perform various multi-modal tasks including:
- Visual Question Answering 
- Image-Text retrieval (Image-text matching)
- Image Captioning

The abstract from the paper is the following:

*Vision-Language Pre-training (VLP) has advanced the performance for many vision-language tasks. 
However, most existing pre-trained models only excel in either understanding-based tasks or generation-based tasks. Furthermore, performance improvement has been largely achieved by scaling up the dataset with noisy image-text pairs collected from the web, which is a suboptimal source of supervision. In this paper, we propose BLIP, a new VLP framework which transfers flexibly to both vision-language understanding and generation tasks. BLIP effectively utilizes the noisy web data by bootstrapping the captions, where a captioner generates synthetic captions and a filter removes the noisy ones. We achieve state-of-the-art results on a wide range of vision-language tasks, such as image-text retrieval (+2.7% in average recall@1), image captioning (+2.8% in CIDEr), and VQA (+1.6% in VQA score). BLIP also demonstrates strong generalization ability when directly transferred to videolanguage tasks in a zero-shot manner. Code, models, and datasets are released.*

![BLIP.gif](https://cdn-uploads.huggingface.co/production/uploads/1670928184033-62441d1d9fdefb55a0b7d12c.gif)

This model was contributed by [ybelkada](https://huggingface.co/ybelkada).
The original code can be found [here](https://github.com/salesforce/BLIP).

## Resources

- [Jupyter notebook](https://github.com/huggingface/notebooks/blob/main/examples/image_captioning_blip.ipynb) on how to fine-tune BLIP for image captioning on a custom dataset

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

`BlipModel` is going to be deprecated in future versions, please use `BlipForConditionalGeneration`, `BlipForImageTextRetrieval` or `BlipForQuestionAnswering` depending on your usecase.

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
