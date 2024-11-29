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


## How to use

Here is a quick example of using the model. Since this model is an image matching model, it requires pairs of images to be matched:

```python
from transformers import AutoImageProcessor, AutoModel
import torch
from PIL import Image
import requests

url = "https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/assets/phototourism_sample_images/london_bridge_78916675_4568141288.jpg?raw=true"
im1 = Image.open(requests.get(url, stream=True).raw)
url = "https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/assets/phototourism_sample_images/london_bridge_19481797_2295892421.jpg?raw=true"
im2 = Image.open(requests.get(url, stream=True).raw)
images = [im1, im2]

processor = AutoImageProcessor.from_pretrained("stevenbucaille/superglue_outdoor")
model = AutoModel.from_pretrained("stevenbucaille/superglue_outdoor")

inputs = processor(images, return_tensors="pt")
outputs = model(**inputs)
```

The outputs contain the list of keypoints detected by the keypoint detector as well as the list of matches with their corresponding matching scores.
Due to the nature of LightGlue, to output a dynamic number of matches, you will need to use the mask attribute to retrieve the respective information:

```python
from transformers import AutoImageProcessor, AutoModel
import torch
from PIL import Image
import requests

url_image_1 = "https://github.com/cvg/LightGlue/blob/main/assets/sacre_coeur1.jpg?raw=true"
image_1 = Image.open(requests.get(url_image_1, stream=True).raw)
url_image_2 = "https://github.com/cvg/LightGlue/blob/main/assets/sacre_coeur2.jpg?raw=true"
image_2 = Image.open(requests.get(url_image_2, stream=True).raw)

images = [image_1, image_2]

processor = AutoImageProcessor.from_pretrained("stevenbucaille/superglue_outdoor")
model = AutoModel.from_pretrained("stevenbucaille/superglue_outdoor")

inputs = processor(images, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

# Get the respective image masks 
image0_mask, image1_mask = outputs_mask[0]

image0_indices = torch.nonzero(image0_mask).squeeze()
image1_indices = torch.nonzero(image1_mask).squeeze()

image0_matches = outputs.matches[0, 0][image0_indices]
image1_matches = outputs.matches[0, 1][image1_indices]

image0_matching_scores = outputs.matching_scores[0, 0][image0_indices]
image1_matching_scores = outputs.matching_scores[0, 1][image1_indices]
```

You can then print the matched keypoints on a side-by-side image to visualize the result :
```python
import cv2
import numpy as np

# Create side by side image
input_data = inputs['pixel_values']
height, width = input_data.shape[-2:]
matched_image = np.zeros((height, width * 2, 3))
matched_image[:, :width] = input_data.squeeze()[0].permute(1, 2, 0).cpu().numpy()
matched_image[:, width:] = input_data.squeeze()[1].permute(1, 2, 0).cpu().numpy()
matched_image = (matched_image * 255).astype(np.uint8)

# Retrieve matches by looking at which keypoints in image0 actually matched with keypoints in image1
image0_mask = outputs.mask[0, 0]
image0_indices = torch.nonzero(image0_mask).squeeze()
image0_matches_indices = torch.nonzero(outputs.matches[0, 0][image0_indices] != -1).squeeze()
image0_keypoints = outputs.keypoints[0, 0][image0_matches_indices]
image0_matches = outputs.matches[0, 0][image0_matches_indices]
image0_matching_scores = outputs.matching_scores[0, 0][image0_matches_indices]
# Retrieve matches from image1
image1_mask = outputs.mask[0, 1]
image1_indices = torch.nonzero(image1_mask).squeeze()
image1_keypoints = outputs.keypoints[0, 1][image0_matches]

# Draw matches
for (keypoint0, keypoint1, score) in zip(image0_keypoints, image1_keypoints, image0_matching_scores):
    keypoint0_x, keypoint0_y = int(keypoint0[0].item()), int(keypoint0[1].item())
    keypoint1_x, keypoint1_y = int(keypoint1[0].item() + width), int(keypoint1[1].item())
    color = tuple([int(score.item() * 255)] * 3)
    matched_image = cv2.line(matched_image, (keypoint0_x, keypoint0_y), (keypoint1_x, keypoint1_y), color)
cv2.imwrite(f"matched_image.png", matched_image)
```

This model was contributed by [stevenbucaille](https://huggingface.co/stevenbucaille).
# TODO 
The original code can be found [here](https://github.com/magicleap/SuperGluePretrainedNetwork).

## LightGlueConfig

[[autodoc]] LightGlueConfig

## LightGlueImageProcessor

[[autodoc]] LightGlueImageProcessor

- preprocess

## LightGlueForKeypointMatching

[[autodoc]] LightGlueForKeypointMatching

- forward
