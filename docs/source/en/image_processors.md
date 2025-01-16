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

# Image processors

Image processors converts images into pixel values, tensors that represent image colors and size. The pixel values are inputs to a vision or video model. To ensure a pretrained model receives the correct input, an image processor can perform the following operations to make sure an image is exactly like the images a model was pretrained on.

- [`~BaseImageProcessor.center_crop`] to resize an image
- [`~BaseImageProcessor.normalize`] or [`~BaseImageProcessor.rescale`] pixel values

Use [`~ImageProcessingMixin.from_pretrained`] to load an image processors configuration (image size, whether to normalize and rescale, etc.) from a vision model on the Hugging Face [Hub](https://hf.co) or local directory. The configuration for each pretrained model is saved in a [preprocessor_config.json](https://huggingface.co/google/vit-base-patch16-224/blob/main/preprocessor_config.json) file.

```py
from transformers import AutoImageProcessor

image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
```

Pass an image to the image processor to transform it into pixel values, and set `return_tensors="pt"` to return PyTorch tensors. Feel free to print out the inputs to see what the image looks like as a tensor.

```py
from PIL import Image
import requests

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/image_processor_example.png"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
inputs = image_processor(image, return_tensors="pt")
```

This guide covers the image processor class and how to preprocess images for vision models.

## Image processor classes

Image processors inherit from the [`BaseImageProcessor`] class which provides the [`~BaseImageProcessor.center_crop`], [`~BaseImageProcessor.normalize`], and [`~BaseImageProcessor.rescale`] functions. There are two types of image processors.

- [`BaseImageProcessor`] is a Python implementation.
- [`BaseImageProcessorFast`] is a faster [torchvision-backed](https://pytorch.org/vision/stable/index.html) version. For a batch of [torch.Tensor](https://pytorch.org/docs/stable/tensors.html) inputs, this can be up to 33x faster. [`BaseImageProcessorFast`] is not available for all vision models at the moment. Refer to a models API documentation to check if it is supported.

Each image processor subclasses the [`ImageProcessingMixin`] class which provides the [`~ImageProcessingMixin.from_pretrained`] and [`~ImageProcessingMixin.save_pretrained`] methods for loading and saving image processors.

There are two ways you can load an image processor, with [`AutoImageProcessor`] or a model-specific image processor.

<hfoptions id="image-processor-classes">
<hfoption id="AutoImageProcessor">

The [AutoClass](./model_doc/auto) API provides a convenient method to load an image processor without directly specifying the model the image processor is associated with.

Use [`~AutoImageProcessor.from_pretrained`] to load an image processor, and set `use_fast=True` to load a fast image processor if it's supported.

```py
from transformers import AutoImageProcessor

image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224", use_fast=True)
```

</hfoption>
<hfoption id="model-specific image processor">

Each image processor is associated with a specific pretrained vision model, and the image processors configuration contains the models expected size and whether to normalize and resize.

The image processor can be loaded directly from the model-specific class. Check a models API documentation to see whether it supports a fast image processor.

```py
from transformers import ViTImageProcessor

image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
```

To load a fast image processor, use the fast implementation class.

```py
from transformers import ViTImageProcessorFast

image_processor = ViTImageProcessorFast.from_pretrained("google/vit-base-patch16-224")
```

</hfoption>
</hfoptions>

## Fast image processors

[`BaseImageProcessorFast`] is based on [torchvision](https://pytorch.org/vision/stable/index.html) and is significantly faster, especially when processing on a GPU. This class can be used as a drop-in replacement for [`BaseImageProcessor`] if it's available for a model because it has the same design. Make sure [torchvision](https://pytorch.org/get-started/locally/#mac-installation) is installed, and set the `use_fast` parameter to `True`.

```py
from transformers import AutoImageProcessor

processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50", use_fast=True)
```

Control which device processing is performed on with the `device` parameter. Processing is performed on the same device as the input by default if the inputs are tensors, otherwise they are processed on the CPU. The example below places the fast processor on a GPU.

```py
from torchvision.io import read_image
from transformers import DetrImageProcessorFast

images = read_image("image.jpg")
processor = DetrImageProcessorFast.from_pretrained("facebook/detr-resnet-50")
images_processed = processor(images, return_tensors="pt", device="cuda")
```

<details>
<summary>Benchmarks</summary>

The benchmarks are obtained from an [AWS EC2 g5.2xlarge](https://aws.amazon.com/ec2/instance-types/g5/) instance with a NVIDIA A10G Tensor Core GPU.

<div class="flex">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/benchmark_results_full_pipeline_detr_fast_padded.png" />
</div>
<div class="flex">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/benchmark_results_full_pipeline_detr_fast_batched_compiled.png" />
</div>
<div class="flex">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/benchmark_results_full_pipeline_rt_detr_fast_single.png" />
</div>
<div class="flex">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/benchmark_results_full_pipeline_rt_detr_fast_batched.png" />
</div>
</details>

## Preprocess

Transformers' vision models expects the input as PyTorch tensors of pixel values. An image processor handles the conversion of images to pixel values, which is represented by the batch size, number of channels, height, and width. To achieve this, an image is resized (center cropped) and the pixel values are normalized and rescaled to the models expected values.

Image preprocessing is not the same as *image augmentation*. Image augmentation makes changes (brightness, colors, rotatation, etc.) to an image for the purpose of either creating new training examples or prevent overfitting. Image preprocessing makes changes to an image for the purpose of matching a pretrained model's expected input format.

Typically, images are augmented (to increase performance) and then preprocessed before being passed to a model. You can use any library ([Albumentations](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification_albumentations.ipynb), [Kornia](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification_kornia.ipynb)) for augmentation and an image processor for preprocessing.

This guide uses the torchvision [transforms](https://pytorch.org/vision/stable/transforms.html) module for augmentation.

Start by loading a small sample of the [food101](https://hf.co/datasets/food101) dataset.

```py
from datasets import load_dataset

dataset = load_dataset("food101", split="train[:100]")
```

From the [transforms](https://pytorch.org/vision/stable/transforms.html) module, use the [Compose](https://pytorch.org/vision/master/generated/torchvision.transforms.Compose.html) API to chain together [RandomResizedCrop](https://pytorch.org/vision/main/generated/torchvision.transforms.RandomResizedCrop.html) and [ColorJitter](https://pytorch.org/vision/main/generated/torchvision.transforms.ColorJitter.html). These transforms randomly crop and resize an image, and randomly adjusts an images colors.

The image size to randomly crop to can be retrieved from the image processor. For some models, an exact height and width are expected while for others, only the `shortest_edge` is required.

```py
from torchvision.transforms import RandomResizedCrop, ColorJitter, Compose

size = (
    image_processor.size["shortest_edge"]
    if "shortest_edge" in image_processor.size
    else (image_processor.size["height"], image_processor.size["width"])
)
_transforms = Compose([RandomResizedCrop(size), ColorJitter(brightness=0.5, hue=0.5)])
```

Apply the transforms to the images and convert them to the RGB format. Then pass the augmented images to the image processor to return the pixel values.

The `do_resize` parameter is set to `False` because the images have already been resized in the augmentation step by [RandomResizedCrop](https://pytorch.org/vision/main/generated/torchvision.transforms.RandomResizedCrop.html). If you don't augment the images, then the image processor automatically resizes and normalizes the images with the `image_mean` and `image_std` values. These values are found in the preprocessor configuration file.

```py
def transforms(examples):
    images = [_transforms(img.convert("RGB")) for img in examples["image"]]
    examples["pixel_values"] = image_processor(images, do_resize=False, return_tensors="pt")["pixel_values"]
    return examples
```

Apply the combined augmentation and preprocessing function to the entire dataset on the fly with [`~datasets.Dataset.set_transform`].

```py
dataset.set_transform(transforms)
```

Convert the pixel values back into an image to see how the image has been augmented and preprocessed.

```py
import numpy as np
import matplotlib.pyplot as plt

img = dataset[0]["pixel_values"]
plt.imshow(img.permute(1, 2, 0))
```

<div class="flex gap-4">
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/vision-preprocess-tutorial.png" />
    <figcaption class="mt-2 text-center text-sm text-gray-500">before</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/preprocessed_image.png" />
    <figcaption class="mt-2 text-center text-sm text-gray-500">after</figcaption>
  </div>
</div>

For other vision tasks like object detection or segmentation, the image processor includes post-processing methods to convert a models raw output into meaningful predictions like bounding boxes or segmentation maps.

### Padding

Some models, like [DETR](./model_doc/detr), applies [scale augmentation](https://paperswithcode.com/method/image-scale-augmentation) during training which can cause images in a batch to have different sizes. Images with different sizes can't be batched together.

To fix this, pad the images with the special padding token `0`. Use the [pad](https://github.com/huggingface/transformers/blob/9578c2597e2d88b6f0b304b5a05864fd613ddcc1/src/transformers/models/detr/image_processing_detr.py#L1151) method to pad the images, and define a custom collate function to batch them together.

```py
def collate_fn(batch):
    pixel_values = [item["pixel_values"] for item in batch]
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    labels = [item["labels"] for item in batch]
    batch = {}
    batch["pixel_values"] = encoding["pixel_values"]
    batch["pixel_mask"] = encoding["pixel_mask"]
    batch["labels"] = labels
    return batch
```
