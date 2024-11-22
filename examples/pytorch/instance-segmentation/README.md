<!---
Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Instance Segmentation Examples

This directory contains two scripts that demonstrate how to fine-tune [MaskFormer](https://huggingface.co/docs/transformers/model_doc/maskformer) and [Mask2Former](https://huggingface.co/docs/transformers/model_doc/mask2former) for instance segmentation using PyTorch.
For other instance segmentation models, such as [DETR](https://huggingface.co/docs/transformers/model_doc/detr) and [Conditional DETR](https://huggingface.co/docs/transformers/model_doc/conditional_detr), the scripts need to be adjusted to properly handle input and output data.

Content:
- [PyTorch Version with Trainer](#pytorch-version-with-trainer)
- [PyTorch Version with Accelerate](#pytorch-version-with-accelerate)
- [Reload and Perform Inference](#reload-and-perform-inference)
- [Note on Custom Data](#note-on-custom-data)

## PyTorch Version with Trainer

This example is based on the script [`run_instance_segmentation.py`](https://github.com/huggingface/transformers/blob/main/examples/pytorch/instance-segmentation/run_instance_segmentation.py).

The script uses the [🤗 Trainer API](https://huggingface.co/docs/transformers/main_classes/trainer) to manage training automatically, including distributed environments.

Here, we show how to fine-tune a [Mask2Former](https://huggingface.co/docs/transformers/model_doc/mask2former) model on a subsample of the [ADE20K](https://huggingface.co/datasets/zhoubolei/scene_parse_150) dataset. We created a [small dataset](https://huggingface.co/datasets/qubvel-hf/ade20k-mini) with approximately 2,000 images containing only "person" and "car" annotations; all other pixels are marked as "background."

Here is the `label2id` mapping for this dataset:

```python
label2id = {
    "background": 0,
    "person": 1,
    "car": 2,
}
```

Since the `background` label is not an instance and we don't want to predict it, we will use `do_reduce_labels` to remove it from the data.

Run the training with the following command:

```bash
python run_instance_segmentation.py \
    --model_name_or_path facebook/mask2former-swin-tiny-coco-instance \
    --output_dir finetune-instance-segmentation-ade20k-mini-mask2former \
    --dataset_name qubvel-hf/ade20k-mini \
    --do_reduce_labels \
    --image_height 256 \
    --image_width 256 \
    --do_train \
    --fp16 \
    --num_train_epochs 40 \
    --learning_rate 1e-5 \
    --lr_scheduler_type constant \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --dataloader_num_workers 8 \
    --dataloader_persistent_workers \
    --dataloader_prefetch_factor 4 \
    --do_eval \
    --evaluation_strategy epoch \
    --logging_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 2 \
    --push_to_hub
```

The resulting model can be viewed [here](https://huggingface.co/qubvel-hf/finetune-instance-segmentation-ade20k-mini-mask2former). Always refer to the original paper for details on training hyperparameters. To improve model quality, consider:
- Changing image size parameters (`--image_height`/`--image_width`)
- Adjusting training parameters such as learning rate, batch size, warmup, optimizer, and more (see [TrainingArguments](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments))
- Adding more image augmentations (we created a helpful [HF Space](https://huggingface.co/spaces/qubvel-hf/albumentations-demo) to choose some)

You can also replace the model [checkpoint](https://huggingface.co/models?search=maskformer).

## PyTorch Version with Accelerate

This example is based on the script [`run_instance_segmentation_no_trainer.py`](https://github.com/huggingface/transformers/blob/main/examples/pytorch/instance-segmentation/run_instance_segmentation_no_trainer.py).

The script uses [🤗 Accelerate](https://github.com/huggingface/accelerate) to write your own training loop in PyTorch and run it on various environments, including CPU, multi-CPU, GPU, multi-GPU, and TPU, with support for mixed precision.

First, configure the environment:

```bash
accelerate config
```

Answer the questions regarding your training environment. Then, run:

```bash
accelerate test
```

This command ensures everything is ready for training. Finally, launch training with:

```bash
accelerate launch run_instance_segmentation_no_trainer.py \
    --model_name_or_path facebook/mask2former-swin-tiny-coco-instance \
    --output_dir finetune-instance-segmentation-ade20k-mini-mask2former-no-trainer \
    --dataset_name qubvel-hf/ade20k-mini \
    --do_reduce_labels \
    --image_height 256 \
    --image_width 256 \
    --num_train_epochs 40 \
    --learning_rate 1e-5 \
    --lr_scheduler_type constant \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --dataloader_num_workers 8 \
    --push_to_hub
```

With this setup, you can train on multiple GPUs, log everything to trackers (like Weights and Biases, Tensorboard), and regularly push your model to the hub (with the repo name set to `args.output_dir` under your HF username).
With the default settings, the script fine-tunes a [Mask2Former](https://huggingface.co/docs/transformers/model_doc/mask2former) model on the sample of [ADE20K](https://huggingface.co/datasets/qubvel-hf/ade20k-mini) dataset. The resulting model can be viewed [here](https://huggingface.co/qubvel-hf/finetune-instance-segmentation-ade20k-mini-mask2former-no-trainer).

## Reload and Perform Inference

After training, you can easily load your trained model and perform inference as follows:

```python
import torch
import requests
import matplotlib.pyplot as plt

from PIL import Image
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

# Load image
image = Image.open(requests.get("http://farm4.staticflickr.com/3017/3071497290_31f0393363_z.jpg", stream=True).raw)

# Load model and image processor
device = "cuda"
checkpoint = "qubvel-hf/finetune-instance-segmentation-ade20k-mini-mask2former"

model = Mask2FormerForUniversalSegmentation.from_pretrained(checkpoint, device_map=device)
image_processor = Mask2FormerImageProcessor.from_pretrained(checkpoint)

# Run inference on image
inputs = image_processor(images=[image], return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)

# Post-process outputs
outputs = image_processor.post_process_instance_segmentation(outputs, target_sizes=[(image.height, image.width)])

print("Mask shape: ", outputs[0]["segmentation"].shape)
print("Mask values: ", outputs[0]["segmentation"].unique())
for segment in outputs[0]["segments_info"]:
    print("Segment: ", segment)
```

```
Mask shape:  torch.Size([427, 640])
Mask values:  tensor([-1.,  0.,  1.,  2.,  3.,  4.,  5.,  6.])
Segment:  {'id': 0, 'label_id': 0, 'was_fused': False, 'score': 0.946127}
Segment:  {'id': 1, 'label_id': 1, 'was_fused': False, 'score': 0.961582}
Segment:  {'id': 2, 'label_id': 1, 'was_fused': False, 'score': 0.968367}
Segment:  {'id': 3, 'label_id': 1, 'was_fused': False, 'score': 0.819527}
Segment:  {'id': 4, 'label_id': 1, 'was_fused': False, 'score': 0.655761}
Segment:  {'id': 5, 'label_id': 1, 'was_fused': False, 'score': 0.531299}
Segment:  {'id': 6, 'label_id': 1, 'was_fused': False, 'score': 0.929477}
```

Use the following code to visualize the results:

```python
import numpy as np
import matplotlib.pyplot as plt

segmentation = outputs[0]["segmentation"].numpy()

plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.imshow(np.array(image))
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(segmentation)
plt.axis("off")
plt.show()
```

![Result](https://i.imgur.com/rZmaRjD.png)

## Note on Custom Data

Here is a short script demonstrating how to create your own dataset for instance segmentation and push it to the hub:

> Note: Annotations should be represented as 3-channel images (similar to the [scene_parsing_150](https://huggingface.co/datasets/zhoubolei/scene_parse_150#instance_segmentation-1) dataset). The first channel is a semantic-segmentation map with values corresponding to `label2id`, the second is an instance-segmentation map where each instance has a unique value, and the third channel should be empty (filled with zeros).

```python
from datasets import Dataset, DatasetDict
from datasets import Image as DatasetImage

label2id = {
    "background": 0,
    "person": 1,
    "car": 2,
}

train_split = {
    "image": [<PIL Image 1>, <PIL Image 2>, <PIL Image 3>, ...],
    "annotation": [<PIL Image ann 1>, <PIL Image ann 2>, <PIL Image ann 3>, ...],
}

validation_split = {
    "image": [<PIL Image 101>, <PIL Image 102>, <PIL Image 103>, ...],
    "annotation": [<PIL Image ann 101>, <PIL Image ann 102>, <PIL Image ann 103>, ...],
}

def create_instance_segmentation_dataset(label2id, **splits):
    dataset_dict = {}
    for split_name, split in splits.items():
        split["semantic_class_to_id"] = [label2id] * len(split["image"])
        dataset_split = (
            Dataset.from_dict(split)
            .cast_column("image", DatasetImage())
            .cast_column("annotation", DatasetImage())
        )
        dataset_dict[split_name] = dataset_split
    return DatasetDict(dataset_dict)

dataset = create_instance_segmentation_dataset(label2id, train=train_split, validation=validation_split)
dataset.push_to_hub("qubvel-hf/ade20k-nano")
```

Use this dataset for fine-tuning by specifying its name with `--dataset_name <your_dataset_repo>`.

See also: [Dataset Creation Guide](https://huggingface.co/docs/datasets/image_dataset#create-an-image-dataset)