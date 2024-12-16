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

# Aria

## Overview

The Aria model was proposed in [Aria: An Open Multimodal Native Mixture-of-Experts Model](https://huggingface.co/papers/2410.05993) by Li et al. from the Rhymes.AI team.

Aria is an open multimodal-native model with best-in-class performance across a wide range of multimodal, language, and coding tasks. It has a Mixture-of-Experts architecture, with respectively 3.9B and 3.5B activated parameters per visual token and text token. 

The abstract from the paper is the following:

*Information comes in diverse modalities. Multimodal native AI models are essential to integrate real-world information and deliver comprehensive understanding. While proprietary multimodal native models exist, their lack of openness imposes obstacles for adoptions, let alone adaptations. To fill this gap, we introduce Aria, an open multimodal native model with best-in-class performance across a wide range of multimodal, language, and coding tasks. Aria is a mixture-of-expert model with 3.9B and 3.5B activated parameters per visual token and text token, respectively. It outperforms Pixtral-12B and Llama3.2-11B, and is competitive against the best proprietary models on various multimodal tasks. We pre-train Aria from scratch following a 4-stage pipeline, which progressively equips the model with strong capabilities in language understanding, multimodal understanding, long context window, and instruction following. We open-source the model weights along with a codebase that facilitates easy adoptions and adaptations of Aria in real-world applications.*

This model was contributed by [m-ric](https://huggingface.co/m-ric).
The original code can be found [here](https://github.com/rhymes-ai/Aria).

## Usage tips

Here's how to use the model for vision tasks:
```python
import requests
import torch
from PIL import Image

from transformers import AriaProcessor, AriaForConditionalGeneration

model_id_or_path = "rhymes-ai/Aria"

model = AriaForConditionalGeneration.from_pretrained(
    model_id_or_path, device_map="auto"
)

processor = AriaProcessor.from_pretrained(model_id_or_path)

image = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"text": "what is the image?", "type": "text"},
        ],
    }
]

text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=text, images=image, return_tensors="pt")
inputs.to(model.device)

output = model.generate(
    **inputs,
    max_new_tokens=15,
    stop_strings=["<|im_end|>"],
    tokenizer=processor.tokenizer,
    do_sample=True,
    temperature=0.9,
)
output_ids = output[0][inputs["input_ids"].shape[1]:]
response = processor.decode(output_ids, skip_special_tokens=True)
```


## AriaImageProcessor

[[autodoc]] AriaImageProcessor

## AriaProcessor

[[autodoc]] AriaProcessor

## AriaTextConfig

[[autodoc]] AriaTextConfig

## AriaConfig

[[autodoc]] AriaConfig

## AriaTextModel

[[autodoc]] AriaTextModel

## AriaTextForCausalLM

[[autodoc]] AriaTextForCausalLM

## AriaForConditionalGeneration

[[autodoc]] AriaForConditionalGeneration
    - forward
