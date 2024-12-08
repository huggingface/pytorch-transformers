<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the MIT License; you may not use this file except in compliance with
the License.

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.


-->

# LightGlue

## Overview

The LightGlue model was proposed in [LightGlue: Local Feature Matching at Light Speed](https://arxiv.org/abs/2306.13643)
by Philipp Lindenberger, Paul-Edouard Sarlin and Marc Pollefeys.

Similar to [SuperGlue](https://huggingface.co/magic-leap-community/superglue_outdoor), this model consists of matching
two sets of local features extracted from two images, its goal is to be faster than SuperGlue. Paired with the 
[SuperPoint model](https://huggingface.co/magic-leap-community/superpoint), it can be used to match two images and 
estimate the pose between them. This model is useful for tasks such as image matching, homography estimation, etc.

The abstract from the paper is the following:

*We introduce LightGlue, a deep neural network that learns to match local features across images. We revisit multiple
design decisions of SuperGlue, the state of the art in sparse matching, and derive simple but effective improvements. 
Cumulatively, they make LightGlue more efficient - in terms of both memory and computation, more accurate, and much
easier to train. One key property is that LightGlue is adaptive to the difficulty of the problem: the inference is much
faster on image pairs that are intuitively easy to match, for example because of a larger visual overlap or limited
appearance change. This opens up exciting prospects for deploying deep matchers in latency-sensitive applications like
3D reconstruction. The code and trained models are publicly available at this [https URL](https://github.com/cvg/LightGlue)*

## How to use

Here is a quick example of using the model. Since this model is an image matching model, it requires pairs of images to be matched. 
The raw outputs contain the list of keypoints detected by the keypoint detector as well as the list of matches with their corresponding 
matching scores.
```python
from transformers import AutoImageProcessor, AutoModel
import torch
from PIL import Image
import requests

url_image1 = "https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/refs/heads/master/assets/phototourism_sample_images/united_states_capitol_98169888_3347710852.jpg"
image1 = Image.open(requests.get(url_image1, stream=True).raw)
url_image2 = "https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/refs/heads/master/assets/phototourism_sample_images/united_states_capitol_26757027_6717084061.jpg"
image2 = Image.open(requests.get(url_image2, stream=True).raw)

images = [image1, image2]

processor = AutoImageProcessor.from_pretrained("stevenbucaille/lightglue_superpoint")
model = AutoModel.from_pretrained("stevenbucaille/lightglue_superpoint")

inputs = processor(images, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

# Get the respective image masks 
image0_mask, image1_mask = outputs.mask[0] > 0

image0_indices = torch.nonzero(image0_mask).squeeze()
image1_indices = torch.nonzero(image1_mask).squeeze()

image0_matches = outputs.matches[0, 0][image0_indices]
image1_matches = outputs.matches[0, 1][image1_indices]

image0_matching_scores = outputs.matching_scores[0, 0][image0_indices]
image1_matching_scores = outputs.matching_scores[0, 1][image1_indices]
```

You can use the `post_process_keypoint_matching` method from the `LightGlueImageProcessor` to get the keypoints and matches in a more readable format:
```python
image_sizes = [[(image.height, image.width) for image in images]]
outputs = processor.post_process_keypoint_matching(outputs, image_sizes, threshold=0.2)
for i, output in enumerate(outputs):
    print("For the image pair", i)
    for keypoint0, keypoint1, matching_score in zip(
            output["keypoints0"], output["keypoints1"], output["matching_scores"]
    ):
        print(
            f"Keypoint at coordinate {keypoint0.numpy()} in the first image matches with keypoint at coordinate {keypoint1.numpy()} in the second image with a score of {matching_score}."
        )
```

From the outputs, you can visualize the matches between the two images using the following code:
```python
import matplotlib.pyplot as plt
import numpy as np

# Create side by side image
merged_image = np.zeros((max(image1.height, image2.height), image1.width + image2.width, 3))
merged_image[: image1.height, : image1.width] = np.array(image1) / 255.0
merged_image[: image2.height, image1.width :] = np.array(image2) / 255.0
plt.imshow(merged_image)
plt.axis("off")

# Retrieve the keypoints and matches
output = outputs[0]
keypoints0 = output["keypoints0"]
keypoints1 = output["keypoints1"]
matching_scores = output["matching_scores"]
keypoints0_x, keypoints0_y = keypoints0[:, 0].numpy(), keypoints0[:, 1].numpy()
keypoints1_x, keypoints1_y = keypoints1[:, 0].numpy(), keypoints1[:, 1].numpy()

# Plot the matches
for keypoint0_x, keypoint0_y, keypoint1_x, keypoint1_y, matching_score in zip(
        keypoints0_x, keypoints0_y, keypoints1_x, keypoints1_y, matching_scores
):
    plt.plot(
        [keypoint0_x, keypoint1_x + image1.width],
        [keypoint0_y, keypoint1_y],
        color=plt.get_cmap("RdYlGn")(matching_score.item()),
        alpha=0.9,
        linewidth=0.5,
    )
    plt.scatter(keypoint0_x, keypoint0_y, c="black", s=2)
    plt.scatter(keypoint1_x + image1.width, keypoint1_y, c="black", s=2)

# Save the plot
plt.savefig("matched_image.png", dpi=300, bbox_inches='tight')
plt.close()
```

![image/png](https://cdn-uploads.huggingface.co/production/uploads/632885ba1558dac67c440aa8/duPp09ty8NRZlMZS18ccP.png)

This model was contributed by [stevenbucaille](https://huggingface.co/stevenbucaille).
The original code can be found [here](https://github.com/cvg/LightGlue).

## LightGlueConfig

[[autodoc]] LightGlueConfig

## LightGlueImageProcessor

[[autodoc]] LightGlueImageProcessor

- preprocess
- post_process_keypoint_matching

## LightGlueForKeypointMatching

[[autodoc]] LightGlueForKeypointMatching

- forward
