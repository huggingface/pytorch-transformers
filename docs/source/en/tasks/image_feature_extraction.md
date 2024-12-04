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

# Image Feature Extraction

[[open-in-colab]]

Image feature extraction is the task of extracting semantically meaningful features given an image. This has many use cases, including image similarity and image retrieval. Moreover, most computer vision models can be used for image feature extraction, where one can remove the task-specific head (image classification, object detection etc) and get the features. These features are very useful on a higher level: edge detection, corner detection and so on. They may also contain information about the real world (e.g. what a cat looks like) depending on how deep the model is. Therefore, these outputs can be used to train new classifiers on a specific dataset.

In this guide, you will:

- Learn to build a simple image similarity system on top of the `image-feature-extraction` pipeline.
- Accomplish the same task with bare model inference.

## Image Similarity using `image-feature-extraction` Pipeline

We have two images of cats sitting on top of fish nets, one of them is generated. 

```python
from PIL import Image
import requests

img_urls = ["https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.png", "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.jpeg"]
image_real = Image.open(requests.get(img_urls[0], stream=True).raw).convert("RGB")
image_gen = Image.open(requests.get(img_urls[1], stream=True).raw).convert("RGB")
```

Let's see the pipeline in action. First, initialize the pipeline. If you don't pass any model to it, the pipeline will be automatically initialized with [google/vit-base-patch16-224](google/vit-base-patch16-224). If you'd like to calculate similarity, set `pool` to True.

```python
import torch
from transformers import pipeline
from accelerate.test_utils.testing import get_backend
# automatically detects the underlying device type (CUDA, CPU, XPU, MPS, etc.)
DEVICE, _, _ = get_backend()
pipe = pipeline(task="image-feature-extraction", model_name="google/vit-base-patch16-384", device=DEVICE, pool=True)
```

To infer with `pipe` pass both images to it.

```python
outputs = pipe([image_real, image_gen])
```

The output contains pooled embeddings of those two images.

```python
# get the length of a single output
print(len(outputs[0][0]))
# show outputs
print(outputs)

# 768
# [[[-0.03909236937761307, 0.43381670117378235, -0.06913255900144577,
```

To get the similarity score, we need to pass them to a similarity function. 

```python
from torch.nn.functional import cosine_similarity

similarity_score = cosine_similarity(torch.Tensor(outputs[0]),
                                     torch.Tensor(outputs[1]), dim=1)

print(similarity_score)

# tensor([0.6043])
```

If you want to get the last hidden states before pooling, avoid passing any value for the `pool` parameter, as it is set to `False` by default. These hidden states are useful for training new classifiers or models based on the features from the model.

```python
pipe = pipeline(task="image-feature-extraction", model_name="google/vit-base-patch16-224", device=DEVICE)
outputs = pipe(image_real)
```

Since the outputs are unpooled, we get the last hidden states where the first dimension is the batch size, and the last two are the embedding shape.

```python
import numpy as np
print(np.array(outputs).shape)
# (1, 197, 768)
```

## Getting Features and Similarities using `AutoModel`

We can also use `AutoModel` class of transformers to get the features. `AutoModel` loads any transformers model with no task-specific head, and we can use this to get the features.

```python
from transformers import AutoImageProcessor, AutoModel

processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = AutoModel.from_pretrained("google/vit-base-patch16-224").to(DEVICE)
```

Let's write a simple function for inference. We will pass the inputs to the `processor` first and pass its outputs to the `model`.

```python
def infer(image):
  inputs = processor(image, return_tensors="pt").to(DEVICE)
  outputs = model(**inputs)
  return outputs.pooler_output
```

We can pass the images directly to this function and get the embeddings.

```python
embed_real = infer(image_real)
embed_gen = infer(image_gen)
```

We can get the similarity again over the embeddings.

```python
from torch.nn.functional import cosine_similarity

similarity_score = cosine_similarity(embed_real, embed_gen, dim=1)
print(similarity_score)

# tensor([0.6061], device='cuda:0', grad_fn=<SumBackward1>)
```

