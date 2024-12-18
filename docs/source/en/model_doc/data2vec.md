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

# Data2Vec

## Overview

The Data2Vec model was proposed in [data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language](https://arxiv.org/pdf/2202.03555) by Alexei Baevski, Wei-Ning Hsu, Qiantong Xu, Arun Babu, Jiatao Gu and Michael Auli.
Data2Vec proposes a unified framework for self-supervised learning across different data modalities - text, audio and images.
Importantly, predicted targets for pre-training are contextualized latent representations of the inputs, rather than modality-specific, context-independent targets.

The abstract from the paper is the following:

*While the general idea of self-supervised learning is identical across modalities, the actual algorithms and
objectives differ widely because they were developed with a single modality in mind. To get us closer to general
self-supervised learning, we present data2vec, a framework that uses the same learning method for either speech,
NLP or computer vision. The core idea is to predict latent representations of the full input data based on a
masked view of the input in a selfdistillation setup using a standard Transformer architecture.
Instead of predicting modality-specific targets such as words, visual tokens or units of human speech which
are local in nature, data2vec predicts contextualized latent representations that contain information from
the entire input. Experiments on the major benchmarks of speech recognition, image classification, and
natural language understanding demonstrate a new state of the art or competitive performance to predominant approaches.
Models and code are available at www.github.com/pytorch/fairseq/tree/master/examples/data2vec.*

This model was contributed by [edugp](https://huggingface.co/edugp) and [patrickvonplaten](https://huggingface.co/patrickvonplaten).
[sayakpaul](https://github.com/sayakpaul) and [Rocketknight1](https://github.com/Rocketknight1) contributed Data2Vec for vision in TensorFlow.

The original code (for NLP and Speech) can be found [here](https://github.com/pytorch/fairseq/tree/main/examples/data2vec).
The original code for vision can be found [here](https://github.com/facebookresearch/data2vec_vision/tree/main/beit).

## Usage tips

- Data2VecAudio, Data2VecText, and Data2VecVision have all been trained using the same self-supervised learning method.
- For Data2VecAudio, preprocessing is identical to [`Wav2Vec2Model`], including feature extraction
- For Data2VecText, preprocessing is identical to [`RobertaModel`], including tokenization.
- For Data2VecVision, preprocessing is identical to [`BeitModel`], including feature extraction.

### Using Scaled Dot Product Attention (SDPA)

PyTorch includes a native scaled dot-product attention (SDPA) operator as part of `torch.nn.functional`. This function 
encompasses several implementations that can be applied depending on the inputs and the hardware in use. See the 
[official documentation](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) 
or the [GPU Inference](https://huggingface.co/docs/transformers/main/en/perf_infer_gpu_one#pytorch-scaled-dot-product-attention)
page for more information.

SDPA is used by default for `torch>=2.1.1` when an implementation is available, but you may also set 
`attn_implementation="sdpa"` in `from_pretrained()` to explicitly request SDPA to be used.

The SDPA implementation is currently available for the Data2VecAudio and Data2VecVision models.

```
from transformers import Data2VecVisionForImageClassification
model = Data2VecVisionForImageClassification.from_pretrained("facebook/data2vec-vision-base", attn_implementation="sdpa", torch_dtype=torch.float16)
...
```

For the best speedups, we recommend loading the model in half-precision (e.g. `torch.float16` or `torch.bfloat16`).

For the Data2VecVision model, on a local benchmark (NVIDIA GeForce RTX 2060-8GB, PyTorch 2.5.1, OS Ubuntu 20.04)
with `float16` and `facebook/data2vec-vision-base` model, we saw the following improvements during training and
inference:

#### Training

| num_training_steps | batch_size | image_size   | is_cuda | Time per batch (eager - s) | Time per batch (sdpa - s) | Speedup (%) | Eager peak mem (MB) | SDPA peak mem (MB) | Mem saving (%) |
|--------------------|------------|--------------|---------|----------------------------|---------------------------|-------------|----------------------|--------------------|----------------|
| 50                 | 2          | (1048, 640)  | True    | 0.996                      | 0.754                     | 32.147      | 6722.198            | 4264.653          | 57.626         |

#### Inference

|   Image batch size |   Eager (s/iter) | Eager CI, %   |   Eager memory (MB) |   SDPA (s/iter) | SDPA CI, %   |   SDPA memory (MB) |   SDPA speedup |   SDPA memory saved |
|-------------------:|-----------------:|:--------------|--------------------:|----------------:|:-------------|-------------------:|---------------:|--------------------:|
|                  1 |            0.011 | ±0.3%         |         3.76143e+08 |           0.01  | ±0.3%        |        3.74397e+08 |          1.101 |               0.466 |
|                  4 |            0.014 | ±0.1%         |         4.02756e+08 |           0.012 | ±0.2%        |        3.91373e+08 |          1.219 |               2.909 |
|                 16 |            0.046 | ±0.3%         |         4.96482e+08 |           0.035 | ±0.2%        |        4.51017e+08 |          1.314 |              10.081 |
|                 32 |            0.088 | ±0.1%         |         6.23903e+08 |           0.067 | ±0.1%        |        5.32974e+08 |          1.33  |              17.061 |

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with Data2Vec.

<PipelineTag pipeline="image-classification"/>

- [`Data2VecVisionForImageClassification`] is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb).
- To fine-tune [`TFData2VecVisionForImageClassification`] on a custom dataset, see [this notebook](https://colab.research.google.com/github/sayakpaul/TF-2.0-Hacks/blob/master/data2vec_vision_image_classification.ipynb).

**Data2VecText documentation resources**
- [Text classification task guide](../tasks/sequence_classification)
- [Token classification task guide](../tasks/token_classification)
- [Question answering task guide](../tasks/question_answering)
- [Causal language modeling task guide](../tasks/language_modeling)
- [Masked language modeling task guide](../tasks/masked_language_modeling)
- [Multiple choice task guide](../tasks/multiple_choice)

**Data2VecAudio documentation resources**
- [Audio classification task guide](../tasks/audio_classification)
- [Automatic speech recognition task guide](../tasks/asr)

**Data2VecVision documentation resources**
- [Image classification](../tasks/image_classification)
- [Semantic segmentation](../tasks/semantic_segmentation)

If you're interested in submitting a resource to be included here, please feel free to open a Pull Request and we'll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

## Data2VecTextConfig

[[autodoc]] Data2VecTextConfig

## Data2VecAudioConfig

[[autodoc]] Data2VecAudioConfig

## Data2VecVisionConfig

[[autodoc]] Data2VecVisionConfig

<frameworkcontent>
<pt>

## Data2VecAudioModel

[[autodoc]] Data2VecAudioModel
    - forward

## Data2VecAudioForAudioFrameClassification

[[autodoc]] Data2VecAudioForAudioFrameClassification
    - forward

## Data2VecAudioForCTC

[[autodoc]] Data2VecAudioForCTC
    - forward

## Data2VecAudioForSequenceClassification

[[autodoc]] Data2VecAudioForSequenceClassification
    - forward

## Data2VecAudioForXVector

[[autodoc]] Data2VecAudioForXVector
    - forward

## Data2VecTextModel

[[autodoc]] Data2VecTextModel
    - forward

## Data2VecTextForCausalLM

[[autodoc]] Data2VecTextForCausalLM
    - forward

## Data2VecTextForMaskedLM

[[autodoc]] Data2VecTextForMaskedLM
    - forward

## Data2VecTextForSequenceClassification

[[autodoc]] Data2VecTextForSequenceClassification
    - forward

## Data2VecTextForMultipleChoice

[[autodoc]] Data2VecTextForMultipleChoice
    - forward

## Data2VecTextForTokenClassification

[[autodoc]] Data2VecTextForTokenClassification
    - forward

## Data2VecTextForQuestionAnswering

[[autodoc]] Data2VecTextForQuestionAnswering
    - forward

## Data2VecVisionModel

[[autodoc]] Data2VecVisionModel
    - forward

## Data2VecVisionForImageClassification

[[autodoc]] Data2VecVisionForImageClassification
    - forward

## Data2VecVisionForSemanticSegmentation

[[autodoc]] Data2VecVisionForSemanticSegmentation
    - forward

</pt>
<tf>

## TFData2VecVisionModel

[[autodoc]] TFData2VecVisionModel
    - call

## TFData2VecVisionForImageClassification

[[autodoc]] TFData2VecVisionForImageClassification
    - call

## TFData2VecVisionForSemanticSegmentation

[[autodoc]] TFData2VecVisionForSemanticSegmentation
    - call

</tf>
</frameworkcontent>
