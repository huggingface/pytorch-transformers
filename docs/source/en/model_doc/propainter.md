<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the S-Lab License, Version 1.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

https://github.com/sczhou/ProPainter/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# ProPainter

## Overview

The ProPainter model was proposed in [ProPainter: Improving Propagation and Transformer for Video Inpainting](https://arxiv.org/abs/2309.03897) by Shangchen Zhou, Chongyi Li, Kelvin C.K. Chan, Chen Change Loy.

ProPainter is an advanced framework designed for video frame editing, leveraging flow-based propagation and spatiotemporal transformers to achieve seamless inpainting and other sophisticated video manipulation tasks. ProPainter offers three key features for video editing:
a. **Object Removal**: Remove unwanted object(s) from a video
b. **Video Completion**: Fill in missing parts of a masked video with contextually relevant content
c. **Video Outpainting**: Expand the view of a video to include additional surrounding content

ProPainter includes three essential components: recurrent flow completion, dual-domain propagation, and mask-guided sparse Transformer. Initially, we utilize an efficient recurrent flow completion network to restore corrupted flow fields. We then perform propagation in both image and feature domains, which are jointly optimized. This combined approach allows us to capture correspondences from both global and local temporal frames, leading to more accurate and effective propagation. Finally, the mask-guided sparse Transformer blocks refine the propagated features using spatiotemporal attention, employing a sparse strategy that processes only a subset of tokens. This improves efficiency and reduces memory usage while preserving performance.

The abstract from the paper is the following:

*Flow-based propagation and spatiotemporal Transformer are two mainstream mechanisms in video inpainting (VI). Despite the effectiveness of these components, they still suffer from some limitations that affect their performance. Previous propagation-based approaches are performed separately either in the image or feature domain. Global image propagation isolated from learning may cause spatial misalignment due to inaccurate optical flow. Moreover, memory or computational constraints limit the temporal range of feature propagation and video Transformer, preventing exploration of correspondence information from distant frames. To address these issues, we propose an improved framework, called ProPainter, which involves enhanced ProPagation and an efficient Transformer. Specifically, we introduce dual-domain propagation that combines the advantages of image and feature warping, exploiting global correspondences reliably. We also propose a mask-guided sparse video Transformer, which achieves high efficiency by discarding unnecessary and redundant tokens. With these components, ProPainter outperforms prior arts by a large margin of 1.46 dB in PSNR while maintaining appealing efficiency.*

This model was contributed by [ruffy369](https://huggingface.co/ruffy369). The original code can be found [here](https://github.com/sczhou/ProPainter). The pre-trained checkpoints can be found on the [Hugging Face Hub](https://huggingface.co/models?sort=downloads&search=ruffy369%2Fpropainter).

## Usage tips:

- The model is used for both video inpainting and video outpainting. To switch between modes, `video_painting_mode` keyword argument has to be set in the `ProPainterVideoProcessor`. Choices are: `['video_inpainting', 'video_outpainting']`. By default the mode is `video_inpainting`. To perform outpainting, set `video_painting_mode='video_outpainting'` and provide a `tuple(scale_height, scale_width)` to the `scale_size` keyword argument in `ProPainterVideoProcessor`. In the usage example, we have demonstrated both ways of providing video frames and their corresponding masks regardless of whether the data is in `.mp4`, `.jpg`, or any other image/video format.

- After downloading the original checkpoints from [here](https://github.com/sczhou/ProPainter/releases/tag/v0.1.0), you can convert them using the **conversion script** available at
`src/transformers/models/propainter/convert_propainter_to_hf.py` with the following command:

```bash
python src/transformers/models/propainter/convert_propainter_to_hf.py \
    --pytorch-dump-folder-path /output/path --verify-logits
```

- You must remember this while providing the inputs as a single batch (one video), i.e., if the size of a single frame goes lower than 128 (height or width) then you **may** possibly encounter the error below. The solution is to keep the frame size to a minimum of **128**.
```
RuntimeError: CUDA error: device-side assert triggered
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
```


## Usage example

The model can accept videos frames and their corresponding masks frame(s) as input. Here's an example code for inference:

```python
import av
import cv2
import imageio
import numpy as np
import os
import torch

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import ProPainterVideoProcessor, ProPainterModel

np.random.seed(0)

def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    '''
    Sample a given number of frame indices from the video.
    Args:
        clip_len (`int`): Total number of frames to sample.
        frame_sample_rate (`int`): Sample every n-th frame.
        seg_len (`int`): Maximum allowed index of sample's last frame.
    Returns:
        indices (`List[int]`): List of sampled frame indices
    '''
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices


# Using .mp4 files for data:

# video clip consists of 80 frames(both masks and original video) (3 seconds at 24 FPS)
video_file_path = hf_hub_download(
    repo_id="ruffy369/propainter-object-removal", filename="object_removal_bmx/bmx.mp4", repo_type="dataset"
)
masks_file_path = hf_hub_download(
    repo_id="ruffy369/propainter-object-removal", filename="object_removal_bmx/bmx_masks.mp4", repo_type="dataset"
)
container_video = av.open(video_file_path)
container_masks = av.open(masks_file_path)

# sample 32 frames
indices = sample_frame_indices(clip_len=32, frame_sample_rate=1, seg_len=container_video.streams.video[0].frames)
video = read_video_pyav(container=container_video, indices=indices)

masks = read_video_pyav(container=container_masks, indices=indices)
video = list(video)
masks = list(masks)

# Forward pass:

device = "cuda" if torch.cuda.is_available() else "cpu"
video_processor = ProPainterVideoProcessor()
inputs = video_processor(video, masks = masks, return_tensors="pt").to(device)

model = ProPainterModel.from_pretrained("ruffy369/ProPainter").to(device)

# The first input in this always has a value for inference as its not utilised during training
with torch.no_grad():
    outputs = model(**inputs)

# To visualize the reconstructed frames with object removal video inpainting:
reconstructed_frames = outputs["reconstruction"][0] # As there is only a single video in batch for inferece
reconstructed_frames = [cv2.resize(frame, (240,432)) for frame in reconstructed_frames]
imageio.mimwrite(os.path.join(<PATH_TO_THE_FOLDER>, 'inpaint_out.mp4'), reconstructed_frames, fps=24, quality=7)

# Using .jpg files for data:

ds = load_dataset("ruffy369/propainter-object-removal")
ds_images = ds['train']["image"]
num_frames = 80
video = [np.array(ds_images[i]) for i in range(num_frames)]
#stack to convert H,W mask frame to compatible H,W,C frame as they are already in grayscale
masks = [np.stack([np.array(ds_images[i])], axis=-1) for i in range(num_frames, 2*num_frames)]

# Forward pass:

inputs = video_processor(video, masks = masks, return_tensors="pt").to(device)

# The first input in this always has a value for inference as its not utilised during training
with torch.no_grad():
    outputs = model(**inputs)

# To visualize the reconstructed frames with object removal video inpainting:
reconstructed_frames = outputs["reconstruction"][0] # As there is only a single video in batch for inferece
reconstructed_frames = [cv2.resize(frame, (240,432)) for frame in reconstructed_frames]
imageio.mimwrite(os.path.join(<PATH_TO_THE_FOLDER>, 'inpaint_out.mp4'), reconstructed_frames, fps=24, quality=7)

# Performing video outpainting:

# Forward pass:

inputs = video_processor(video, masks = masks, video_painting_mode = "video_outpainting", scale_size = (1.0,1.2), return_tensors="pt").to(device)

# The first input in this always has a value for inference as its not utilised during training
with torch.no_grad():
    outputs = model(**inputs)

# To visualize the reconstructed frames with object removal video inpainting:
reconstructed_frames = outputs["reconstruction"][0] # As there is only a single video in batch for inferece
reconstructed_frames = [cv2.resize(frame, (240,512)) for frame in reconstructed_frames]
imageio.mimwrite(os.path.join(<PATH_TO_THE_FOLDER>, 'outpaint_out.mp4'), reconstructed_frames, fps=24, quality=7)
```


## ProPainterConfig

[[autodoc]] ProPainterConfig

## ProPainterProcessor

[[autodoc]] ProPainterProcessor

## ProPainterVideoProcessor

[[autodoc]] ProPainterVideoProcessor

## ProPainterModel

[[autodoc]] ProPainterModel
    - forward
