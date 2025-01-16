# SAM-HQ

## Overview

SAM-HQ (High-Quality Segment Anything Model) was proposed in [Segment Anything in High Quality](https://arxiv.org/pdf/2306.01567.pdf) by Lei Ke, Mingqiao Ye, Martin Danelljan, Yifan Liu, Yu-Wing Tai, Chi-Keung Tang, Fisher Yu.

The model is an enhancement to the original SAM model that produces significantly higher quality segmentation masks while maintaining SAM's original promptable design, efficiency, and zero-shot generalizability.

![example image](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/sam-output.png)


SAM-HQ introduces several key improvements over the original SAM model:

1. High-Quality Output Token: A learnable token injected into SAM's mask decoder for higher quality mask prediction
2. Global-local Feature Fusion: Combines features from different stages of the model for improved mask details
3. Training Data: Uses a carefully curated dataset of 44K high-quality masks instead of SA-1B
4. Efficiency: Adds only 0.5% additional parameters while significantly improving mask quality
5. Zero-shot Capability: Maintains SAM's strong zero-shot performance while improving accuracy

The abstract from the paper is the following:

*The recent Segment Anything Model (SAM) represents a big leap in scaling up segmentation models, allowing for powerful zero-shot capabilities and flexible prompting. Despite being trained with 1.1 billion masks, SAM's mask prediction quality falls short in many cases, particularly when dealing with objects that have intricate structures. We propose HQ-SAM, equipping SAM with the ability to accurately segment any object, while maintaining SAM's original promptable design, efficiency, and zero-shot generalizability. Our careful design reuses and preserves the pre-trained model weights of SAM, while only introducing minimal additional parameters and computation. We design a learnable High-Quality Output Token, which is injected into SAM's mask decoder and is responsible for predicting the high-quality mask. Instead of only applying it on mask-decoder features, we first fuse them with early and final ViT features for improved mask details. To train our introduced learnable parameters, we compose a dataset of 44K fine-grained masks from several sources. HQ-SAM is only trained on the introduced dataset of 44k masks, which takes only 4 hours on 8 GPUs.*

Tips:

- SAM-HQ produces higher quality masks than the original SAM model, particularly for objects with intricate structures and fine details
- The model predicts binary masks with more accurate boundaries and better handling of thin structures
- Like SAM, the model performs better with input 2D points and/or input bounding boxes
- You can prompt multiple points for the same image and predict a single high-quality mask
- The model maintains SAM's zero-shot generalization capabilities
- SAM-HQ only adds ~0.5% additional parameters compared to SAM
- Fine-tuning the model is not supported yet

This model was contributed by [sushmanth](https://huggingface.co/sushmanth).
The original code can be found [here](https://github.com/SysCV/SAM-HQ).

Below is an example on how to run mask generation given an image and a 2D point:

```python
import torch
from PIL import Image
import requests
from transformers import SamHQModel, SamHQProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SamHQModel.from_pretrained("sushmanth/sam_hq_vit_b").to(device)
processor = SamHQProcessor.from_pretrained("sushmanth/sam_hq_vit_b")

img_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
input_points = [[[450, 600]]]  # 2D location of a window in the image

inputs = processor(raw_image, input_points=input_points, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)

masks = processor.image_processor.post_process_masks(
    outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
)
scores = outputs.iou_scores
```

You can also process your own masks alongside the input images in the processor to be passed to the model:

```python
import torch
from PIL import Image
import requests
from transformers import SamHQModel, SamHQProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SamHQModel.from_pretrained("sushmanth/sam_hq_vit_b").to(device)
processor = SamHQProcessor.from_pretrained("sushmanth/sam_hq_vit_b")

img_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
mask_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
segmentation_map = Image.open(requests.get(mask_url, stream=True).raw).convert("1")
input_points = [[[450, 600]]]  # 2D location of a window in the image

inputs = processor(raw_image, input_points=input_points, segmentation_maps=segmentation_map, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)

masks = processor.image_processor.post_process_masks(
    outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
)
scores = outputs.iou_scores
```


## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with SAM-HQ:

- Demo notebook for using the model (coming soon)
- Paper implementation and code: [SAM-HQ GitHub Repository](https://github.com/SysCV/SAM-HQ)

## SamHQConfig

[[autodoc]] SamHQConfig

## SamHQVisionConfig

[[autodoc]] SamHQVisionConfig

## SamHQMaskDecoderConfig

[[autodoc]] SamHQMaskDecoderConfig

## SamHQPromptEncoderConfig

[[autodoc]] SamHQPromptEncoderConfig

## SamHQProcessor

[[autodoc]] SamHQProcessor

## SamHQModel

[[autodoc]] SamHQModel
    - forward