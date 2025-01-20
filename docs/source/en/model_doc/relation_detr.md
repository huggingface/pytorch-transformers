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

# Relation-DETR

## Overview


The Relation-DETR model was proposed in [Relation DETR: Exploring Explicit Position Relation Prior for Object Detection](https://arxiv.org/abs/2407.11699v1) by Xiuquan Hou, Meiqin Liu, Senlin Zhang, Ping Wei, Badong Chen, Xuguang Lan.

Relation-DETR is an object detection model that incorporates explicit position relation priors to enhance detection performance. By leveraging the strengths of transformer architectures while integrating spatial relation modeling, Relation-DETR achieves superior detection accuracy and fast convergence. This innovative design not only enhances the model's capability to capture complex object interactions but also ensures rapid convergence during training, making it an efficient and high-performance solution for object detection tasks.

The abstract from the paper is the following:

*This paper presents a general scheme for enhancing the convergence and performance of DETR (DEtection TRansformer). We investigate the slow convergence problem in transformers from a new perspective, suggesting that it arises from the self-attention that introduces no structural bias over inputs. To address this issue, we explore incorporating position relation prior as attention bias to augment object detection, following the verification of its statistical significance using a proposed quantitative macroscopic correlation (MC) metric. Our approach, termed Relation-DETR, introduces an encoder to construct position relation embeddings for progressive attention refinement, which further extends the traditional streaming pipeline of DETR into a contrastive relation pipeline to address the conflicts between non-duplicate predictions and positive supervision. Extensive experiments on both generic and task-specific datasets demonstrate the effectiveness of our approach. Under the same configurations, Relation-DETR achieves a significant improvement (+2.0% AP compared to DINO), state-of-the-art performance (51.7% AP for 1x and 52.1% AP for 2x settings), and a remarkably faster convergence speed (over 40% AP with only 2 training epochs) than existing DETR detectors on COCO val2017. Moreover, the proposed relation encoder serves as a universal plug-in-and-play component, bringing clear improvements for theoretically any DETR-like methods. Furthermore, we introduce a class-agnostic detection dataset, SA-Det-100k. The experimental results on the dataset illustrate that the proposed explicit position relation achieves a clear improvement of 1.3% AP, highlighting its potential towards universal object detection.*

<img src="https://raw.githubusercontent.com/xiuqhou/Relation-DETR/refs/heads/main/images/convergence_curve.png"
alt="drawing" width="600"/>

<small> Performance comparison between Relation-DETR and other DETR methods. Taken from the <a href="https://arxiv.org/abs/2407.11699">original paper.</a> </small>

The model version was contributed by [xiuqhou](https://github.com/xiuqhou). The original code can be found [here](https://github.com/xiuqhou/Relation-DETR/).


## Usage tips

Initially, an image is processed with a ResNet variant as referenced in the original code. This network extracts features from the final three layers of the architecture. Following this, a transformer encoder is employed to convert the multi-scale features into a sequential array of image features. Then, a decoder, equipped with auxiliary prediction heads and position relation encoders is used to refine the object queries. This process facilitates the direct generation of bounding boxes, eliminating the need for any additional post-processing to acquire the logits and coordinates for the bounding boxes.

```py
>>> import torch
>>> import requests

>>> from PIL import Image
>>> from transformers import RelationDetrForObjectDetection, RelationDetrImageProcessor

>>> url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = RelationDetrImageProcessor.from_pretrained("xiuqhou/relation-detr-resnet50")
>>> model = RelationDetrForObjectDetection.from_pretrained("xiuqhou/relation-detr-resnet50")

>>> inputs = image_processor(images=image, return_tensors="pt")

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> results = image_processor.post_process_object_detection(outputs, target_sizes=torch.tensor([(image.height, image.width)]), threshold=0.3)

>>> for result in results:
...     for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
...         score, label = score.item(), label_id.item()
...         box = [round(i, 2) for i in box.tolist()]
...         print(f"{model.config.id2label[label]}: {score:.2f} {box}")
cat: 0.96 [343.8, 24.9, 639.52, 371.71]
cat: 0.95 [12.6, 54.34, 316.37, 471.86]
remote: 0.95 [40.09, 73.49, 175.52, 118.06]
remote: 0.90 [333.09, 76.71, 369.77, 187.4]
couch: 0.90 [0.45, 0.53, 640.44, 475.54]
```

## RelationDetrConfig

[[autodoc]] RelationDetrConfig

## RelationDetrResNetConfig

[[autodoc]] RelationDetrResNetConfig

## RelationDetrImageProcessor

[[autodoc]] RelationDetrImageProcessor
    - preprocess
    - post_process_object_detection

## RelationDetrImageProcessorFast

[[autodoc]] RelationDetrImageProcessorFast
    - preprocess
    - post_process_object_detection

## RelationDetrModel

[[autodoc]] RelationDetrModel
    - forward

## RelationDetrForObjectDetection

[[autodoc]] RelationDetrForObjectDetection
    - forward
