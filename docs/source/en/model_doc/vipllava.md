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

# VipLlava

## Overview

The VipLlava model was proposed in [Making Large Multimodal Models Understand Arbitrary Visual Prompts](https://arxiv.org/abs/2312.00784) by Mu Cai, Haotian Liu, Siva Karthik Mustikovela, Gregory P. Meyer, Yuning Chai, Dennis Park, Yong Jae Lee.

VipLlava enhances the training protocol of Llava by marking images and interact with the model using natural cues like a "red bounding box" or "pointed arrow" during training.

The abstract from the paper is the following:

*While existing large vision-language multimodal models focus on whole image understanding, there is a prominent gap in achieving region-specific comprehension. Current approaches that use textual coordinates or spatial encodings often fail to provide a user-friendly interface for visual prompting. To address this challenge, we introduce a novel multimodal model capable of decoding arbitrary visual prompts. This allows users to intuitively mark images and interact with the model using natural cues like a "red bounding box" or "pointed arrow". Our simple design directly overlays visual markers onto the RGB image, eliminating the need for complex region encodings, yet achieves state-of-the-art performance on region-understanding tasks like Visual7W, PointQA, and Visual Commonsense Reasoning benchmark. Furthermore, we present ViP-Bench, a comprehensive benchmark to assess the capability of models in understanding visual prompts across multiple dimensions, enabling future research in this domain. Code, data, and model are publicly available.*

The original code can be found [here](https://github.com/mu-cai/ViP-LLaVA).

This model was contributed by [Younes Belkada](https://huggingface.co/ybelkada)


## Usage tips:

- The architecture is similar than llava architecture except that the multi-modal projector takes a set of concatenated vision hidden states and has an additional layernorm layer on that module.

- We advise users to use `padding_side="left"` when computing batched generation as it leads to more accurate results. Simply make sure to call `processor.tokenizer.padding_side = "left"` before generating.

- Note the model has not been explicitly trained to process multiple images in the same prompt, although this is technically possible, you may experience inaccurate results.

> [!NOTE]
> LLaVA models after release v4.46 will raise warnings about adding `processor.patch_size = {{patch_size}}`, `processor.num_additional_image_tokens = {{num_additional_image_tokens}}` and processor.vision_feature_select_strategy = {{vision_feature_select_strategy}}`. It is strongly recommended to add the attributes to the processor if you own the model checkpoint, or open a PR if it is not owned by you.
Adding these attributes means that LLaVA will try to infer the number of image tokens required per image and expand the text with as many `<image>` placeholders as there will be tokens. Usually it is around 500 tokens per image, so make sure that the text is not truncated as otherwise there will be failure when merging the embeddings.
The attributes can be obtained from model config, as `model.config.vision_config.patch_size` or `model.config.vision_feature_select_strategy`. The `num_additional_image_tokens` should be `1` if the vision backbone adds a CLS token or `0` if nothing extra is added to the vision patches.


- For better results, we recommend users to use the processor's `apply_chat_template()` method to format your prompt correctly. For that you need to construct a conversation history, passing in a plain string will not format your prompt. Each message in the conversation history for chat templates is a dictionary with keys "role" and "content". The "content" should be a list of dictionaries, for "text" and "image" modalities, as follows:

```python
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("llava-hf/vip-llava-7b-hf")

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "What’s shown in this image?"},
        ],
    },
    {
        "role": "assistant",
        "content": [{"type": "text", "text": "This image shows a red stop sign."},]
    },
    {

        "role": "user",
        "content": [
            {"type": "text", "text": "Describe the image in more details."},
        ],
    },
]

text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

# Note that the template simply formats your prompt, you still have to tokenize it and obtain pixel values for your images
print(text_prompt)
>>> "###Human: <image>\nWhat’s shown in this image?###Assistant: This image shows a red stop sign.###Human: Describe the image in more details.###Assistant:"
```

- If you want to construct a chat prompt yourself, below is a list of prompt formats accepted by VipLLaVa checkpoints:
```bash
A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.###Human: <image>\n<prompt>###Assistant:
```

For multiple turns conversation:
```bash
A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.###Human: <image>\n<prompt1>###Assistant: <answer1>###Human: <prompt2>###Assistant:
```


## VipLlavaConfig

[[autodoc]] VipLlavaConfig

## VipLlavaForConditionalGeneration

[[autodoc]] VipLlavaForConditionalGeneration
    - forward
