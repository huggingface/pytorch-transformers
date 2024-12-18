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

# ColPali

## Overview

The ColPali model was proposed in [ColPali: Efficient Document Retrieval with Vision Language Models](https://doi.org/10.48550/arXiv.2407.01449) by **Manuel Faysse***, **Hugues Sibille***, **Tony Wu***, Bilel Omrani, Gautier Viaud, Céline Hudelot, Pierre Colombo (* denotes equal contribution).

With our new model *ColPali*, we propose to leverage VLMs to construct efficient multi-vector embeddings in the visual space for document retrieval. By feeding the ViT output patches from PaliGemma-3B to a linear projection, we create a multi-vector representation of documents. We train the model to maximize the similarity between these document embeddings and the query embeddings, following the ColBERT method.

Using ColPali removes the need for potentially complex and brittle layout recognition and OCR pipelines with a single model that can take into account both the textual and visual content (layout, charts, ...) of a document. ColPali is also highly interpretable: similarity maps can be obtained between patches and query tokens. These maps highlight ColPali’s strong OCR capabilities and chart understanding.

**Paper abstract:**

> Documents are visually rich structures that convey information through text, but also figures, page layouts, tables, or even fonts. Since modern retrieval systems mainly rely on the textual information they extract from document pages to index documents -often through lengthy and brittle processes-, they struggle to exploit key visual cues efficiently. This limits their capabilities in many practical document retrieval applications such as Retrieval Augmented Generation (RAG). To benchmark current systems on visually rich document retrieval, we introduce the Visual Document Retrieval Benchmark *ViDoRe*, composed of various page-level retrieval tasks spanning multiple domains, languages, and practical settings. The inherent complexity and performance shortcomings of modern systems motivate a new concept; doing document retrieval by directly embedding the images of the document pages. We release *ColPali*, a Vision Language Model trained to produce high-quality multi-vector embeddings from images of document pages. Combined with a late interaction matching mechanism, *ColPali* largely outperforms modern document retrieval pipelines while being drastically simpler, faster and end-to-end trainable.
>
> We release models, data, code and benchmarks under open licenses at [https://huggingface.co/vidore](https://huggingface.co/vidore).

## Resources

- The official blog post detailing ColPali can be found [here](https://huggingface.co/blog/manu/colpali). 📝
- The original model implementation code for the ColPali model and for the `colpali-engine` package can be found [here](https://github.com/illuin-tech/colpali). 🌎
- Cookbooks for learning to use the transformers-native version of ColPali, fine-tuning, and similarity maps generation can be found [here](https://github.com/tonywu71/colpali-cookbooks). 📚

This model was contributed by [@tonywu71](https://huggingface.co/tonywu71) and [@yonigozlan](https://huggingface.co/yonigozlan).

## Usage

This example demonstrates how to use ColPali to embed both queries and images, calculate their similarity scores, and identify the most relevant matches. For a specific query, you can retrieve the top-k most similar images by selecting the ones with the highest similarity scores.

```python
import torch
from PIL import Image

from transformers import ColPaliForRetrieval, ColPaliProcessor

model_name = "vidore/colpali-v1.2-hf"

model = ColPaliForRetrieval.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",  # or "mps" if on Apple Silicon
).eval()

processor = ColPaliProcessor.from_pretrained(model_name)

# Your inputs (replace dummy images with screenshots of your documents)
images = [
    Image.new("RGB", (32, 32), color="white"),
    Image.new("RGB", (16, 16), color="black"),
]
queries = [
    "What is the organizational structure for our R&D department?",
    "Can you provide a breakdown of last year’s financial performance?",
]

# Process the inputs
batch_images = processor(images=images).to(model.device)
batch_queries = processor(text=queries).to(model.device)

# Forward pass
with torch.no_grad():
    image_embeddings = model(**batch_images)
    query_embeddings = model(**batch_queries)

# Score the queries against the images
scores = processor.score_retrieval(query_embeddings, image_embeddings)
```

## ColPaliConfig

[[autodoc]] ColPaliConfig

## ColPaliProcessor

[[autodoc]] ColPaliProcessor

## ColPaliForRetrieval

[[autodoc]] ColPaliForRetrieval
    - forward
