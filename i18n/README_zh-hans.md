<!---
Copyright 2020 The HuggingFace Team. All rights reserved.

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

<!---
A useful guide for English-Chinese translation of Hugging Face documentation
- Add space around English words and numbers when they appear between Chinese characters. E.g., 共 100 多种语言; 使用 transformers 库。
- Use square quotes, e.g.,「引用」

Dictionary

Hugging Face: 抱抱脸
token: 词符（并用括号标注原英文）
tokenize: 词符化（并用括号标注原英文）
tokenizer: 词符化器（并用括号标注原英文）
transformer: transformer（不翻译）
pipeline: 流水线
API: API (不翻译）
inference: 推理
Trainer: 训练器。当作为类名出现时不翻译。
pretrained/pretrain: 预训练
finetune: 微调
community: 社区
example: 当特指仓库中 example 目录时翻译为「用例」
Python data structures (e.g., list, set, dict): 翻译为列表，集合，词典，并用括号标注原英文
NLP/Natural Language Processing: 以 NLP 出现时不翻译，以 Natural Language Processing 出现时翻译为自然语言处理
checkpoint: 检查点
modality 模态
-->

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/transformers-logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/transformers-logo-light.svg">
    <img alt="Hugging Face Transformers Library" src="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/transformers-logo-light.svg" width="352" height="59" style="max-width: 100%;">
  </picture>
  <br/>
  <br/>
</p>

<p align="center">
    <a href="https://circleci.com/gh/huggingface/transformers"><img alt="Build" src="https://img.shields.io/circleci/build/github/huggingface/transformers/main"></a>
    <a href="https://github.com/huggingface/transformers/blob/main/LICENSE"><img alt="GitHub" src="https://img.shields.io/github/license/huggingface/transformers.svg?color=blue"></a>
    <a href="https://huggingface.co/docs/transformers/index"><img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/docs/transformers/index.svg?down_color=red&down_message=offline&up_message=online"></a>
    <a href="https://github.com/huggingface/transformers/releases"><img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/transformers.svg"></a>
    <a href="https://github.com/huggingface/transformers/blob/main/CODE_OF_CONDUCT.md"><img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg"></a>
    <a href="https://zenodo.org/badge/latestdoi/155220641"><img src="https://zenodo.org/badge/155220641.svg" alt="DOI"></a>
</p>

<h4 align="center">
    <p>
        <a href="https://github.com/huggingface/transformers/">English</a> |
        <b>简体中文</b> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_zh-hant.md">繁體中文</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ko.md">한국어</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_es.md">Español</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ja.md">日本語</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_hd.md">हिन्दी</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ru.md">Русский</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_pt-br.md">Рortuguês</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_te.md">తెలుగు</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_fr.md">Français</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_de.md">Deutsch</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_vi.md">Tiếng Việt</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ar.md">العربية</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ur.md">اردو</a> |
    </p>
</h4>

<h3 align="center">
    <p>为 Jax、PyTorch 和 TensorFlow 社区提供最先进的机器学习模型</p>
</h3>

<h3 align="center">
    <a href="https://hf.co/course"><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/course_banner.png"></a>
</h3>

🤗 Transformers 提供了数以千计的预训练模型，这些模型可以用于执行文本、视觉和音频等不同模态（modality）上的任务。

这些模型可以应用在：

- 📝 文本，用于文本分类、信息提取、问答、摘要、翻译和文本生成等任务，支持超过100种语言

- 🖼️ 图像，用于图像分类、目标检测和分割等任务

- 🗣️ 音频，用于语音识别和音频分类等任务

Transformer 模型还可以在多种模态组合上执行任务，例如表格问答、视觉字符识别（optical character recognition）、文档摘要、视频分类和视觉问答。

🤗 Transformers 提供了便于快速下载和使用的 API，可以快速下载并在给定文本上使用这些预训练模型，在你自己的数据库上对它们进行微调，然后通过 [model hub](https://huggingface.co/models) 与社区共享。同时，每个定义的 Python 模块均完全独立，可以进行修改，用于快速研究实验。

🤗 Transformers 为三个最热门的深度学习框架提供支持 — [Jax](https://jax.readthedocs.io/en/latest/), [PyTorch](https://pytorch.org/) 以及 [TensorFlow](https://www.tensorflow.org/) — 并在它们之间无缝集成。你可以轻松地先用一个训练你的模型，然后再用另一个框架加载它们进行推理。

## 在线演示

你可以直接在 [model hub](https://huggingface.co/models) 上测试我们的大部分模型。 我们还提供了 [私有模型托管、模型版本管理以及推理API](https://huggingface.co/pricing) 服务。

这里有一些例子：

自然语言处理：
- [用 BERT 做掩码填词](https://huggingface.co/google-bert/bert-base-uncased?text=Paris+is+the+%5BMASK%5D+of+France)
- [用 Electra 做命名实体识别](https://huggingface.co/dbmdz/electra-large-discriminator-finetuned-conll03-english?text=My+name+is+Sarah+and+I+live+in+London+city)
- [用 Mistral 做文本生成](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
- [用 RoBERTa 做自然语言推理](https://huggingface.co/FacebookAI/roberta-large-mnli?text=The+dog+was+lost.+Nobody+lost+any+animal)
- [用 BART 做文本摘要](https://huggingface.co/facebook/bart-large-cnn?text=The+tower+is+324+metres+%281%2C063+ft%29+tall%2C+about+the+same+height+as+an+81-storey+building%2C+and+the+tallest+structure+in+Paris.+Its+base+is+square%2C+measuring+125+metres+%28410+ft%29+on+each+side.+During+its+construction%2C+the+Eiffel+Tower+surpassed+the+Washington+Monument+to+become+the+tallest+man-made+structure+in+the+world%2C+a+title+it+held+for+41+years+until+the+Chrysler+Building+in+New+York+City+was+finished+in+1930.+It+was+the+first+structure+to+reach+a+height+of+300+metres.+Due+to+the+addition+of+a+broadcasting+aerial+at+the+top+of+the+tower+in+1957%2C+it+is+now+taller+than+the+Chrysler+Building+by+5.2+metres+%2817+ft%29.+Excluding+transmitters%2C+the+Eiffel+Tower+is+the+second+tallest+free-standing+structure+in+France+after+the+Millau+Viaduct)
- [用 DistilBERT 做问答](https://huggingface.co/distilbert/distilbert-base-uncased-distilled-squad?text=Which+name+is+also+used+to+describe+the+Amazon+rainforest+in+English%3F&context=The+Amazon+rainforest+%28Portuguese%3A+Floresta+Amaz%C3%B4nica+or+Amaz%C3%B4nia%3B+Spanish%3A+Selva+Amaz%C3%B3nica%2C+Amazon%C3%ADa+or+usually+Amazonia%3B+French%3A+For%C3%AAt+amazonienne%3B+Dutch%3A+Amazoneregenwoud%29%2C+also+known+in+English+as+Amazonia+or+the+Amazon+Jungle%2C+is+a+moist+broadleaf+forest+that+covers+most+of+the+Amazon+basin+of+South+America.+This+basin+encompasses+7%2C000%2C000+square+kilometres+%282%2C700%2C000+sq+mi%29%2C+of+which+5%2C500%2C000+square+kilometres+%282%2C100%2C000+sq+mi%29+are+covered+by+the+rainforest.+This+region+includes+territory+belonging+to+nine+nations.+The+majority+of+the+forest+is+contained+within+Brazil%2C+with+60%25+of+the+rainforest%2C+followed+by+Peru+with+13%25%2C+Colombia+with+10%25%2C+and+with+minor+amounts+in+Venezuela%2C+Ecuador%2C+Bolivia%2C+Guyana%2C+Suriname+and+French+Guiana.+States+or+departments+in+four+nations+contain+%22Amazonas%22+in+their+names.+The+Amazon+represents+over+half+of+the+planet%27s+remaining+rainforests%2C+and+comprises+the+largest+and+most+biodiverse+tract+of+tropical+rainforest+in+the+world%2C+with+an+estimated+390+billion+individual+trees+divided+into+16%2C000+species)
- [用 T5 做翻译](https://huggingface.co/google-t5/t5-base?text=My+name+is+Wolfgang+and+I+live+in+Berlin)

计算机视觉：
- [用 ViT 做图像分类](https://huggingface.co/google/vit-base-patch16-224)
- [用 DETR 做目标检测](https://huggingface.co/facebook/detr-resnet-50)
- [用 SegFormer 做语义分割](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512)
- [用 Mask2Former 做全景分割](https://huggingface.co/facebook/mask2former-swin-large-coco-panoptic)
- [用 Depth Anything 做深度估计](https://huggingface.co/docs/transformers/main/model_doc/depth_anything)
- [用 VideoMAE 做视频分类](https://huggingface.co/docs/transformers/model_doc/videomae)
- [用 OneFormer 做通用分割（Universal Segmentation）](https://huggingface.co/shi-labs/oneformer_ade20k_dinat_large)

音频处理：

- [用 Whisper 做自动语音识别](https://huggingface.co/openai/whisper-large-v3)
- [用 Wav2Vec2 做关键词检测](https://huggingface.co/superb/wav2vec2-base-superb-ks)
- [用 Audio Spectrogram Transformer 做音频分类](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593)

多模态任务：

- [Table Question Answering with TAPAS](https://huggingface.co/google/tapas-base-finetuned-wtq)
- [Visual Question Answering with ViLT](https://huggingface.co/dandelin/vilt-b32-finetuned-vqa)
- [Image captioning with LLaVa](https://huggingface.co/llava-hf/llava-1.5-7b-hf)
- [Zero-shot Image Classification with SigLIP](https://huggingface.co/google/siglip-so400m-patch14-384)
- [Document Question Answering with LayoutLM](https://huggingface.co/impira/layoutlm-document-qa)
- [Zero-shot Video Classification with X-CLIP](https://huggingface.co/docs/transformers/model_doc/xclip)
- [Zero-shot Object Detection with OWLv2](https://huggingface.co/docs/transformers/en/model_doc/owlv2)
- [Zero-shot Image Segmentation with CLIPSeg](https://huggingface.co/docs/transformers/model_doc/clipseg)
- [Automatic Mask Generation with SAM](https://huggingface.co/docs/transformers/model_doc/sam)

## 100 个使用 Transformers 的项目

Transformers 不只是一个集成了预训练模型的工具包：它还是一个围绕 Hugging Face Hub 所构建的社区。
我们希望 Transformers 能让开发者、研究人员、学生、教授、工程师以及每一个人能够构建他们理想中的项目。

为了庆祝 transformers 获得 100,000 个 star，我们决定聚焦社区，并创建了 [awesome-transformers](./awesome-transformers.md) 页面，
页面列出了 100 个使用 transformers 构建的不可思议的项目，如果你拥有或正在使用一个你认为应该被列入列表的项目，请提交一个 PR（pull request）添加它！

## 如果你在寻找由抱抱脸团队提供的定制化支持服务

<a target="_blank" href="https://huggingface.co/support">
    <img alt="HuggingFace Expert Acceleration Program" src="https://cdn-media.huggingface.co/marketing/transformers/new-support-improved.png" style="max-width: 600px; border: 1px solid #eee; border-radius: 4px; box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);">
</a><br>

## 快速上手

为了立即在给定输入（文本、图像、音频等）上使用模型，我们提供了 `pipeline`（流水线） API。Pipeline 将预训练模型与该模型训练期间使用的预处理步骤组合在一起。
以下是如何快速使用 `pipeline` 对正面文本和负面文本进行分类

```python
>>> from transformers import pipeline

# 使用情绪分析流水线
>>> classifier = pipeline('sentiment-analysis')
>>> classifier('We are very happy to introduce pipeline to the transformers repository.')
[{'label': 'POSITIVE', 'score': 0.9996980428695679}]
```

第二行代码下载并缓存了流水线使用的预训练模型，而第三行代码则在给定的文本上进行了评估。这里的答案“正面” (positive) 具有 99 的置信度。

许多任务都有一个开箱即用的 `pipeline` 可供使用，包括自然语言处理任务、计算机视觉任务和语音识别任务。
例如，我们可以轻松地提取图像中检测到的物体：
``` python
>>> import requests
>>> from PIL import Image
>>> from transformers import pipeline

# Download an image with cute cats
>>> url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png"
>>> image_data = requests.get(url, stream=True).raw
>>> image = Image.open(image_data)

# Allocate a pipeline for object detection
>>> object_detector = pipeline('object-detection')
>>> object_detector(image)
[{'score': 0.9982201457023621,
  'label': 'remote',
  'box': {'xmin': 40, 'ymin': 70, 'xmax': 175, 'ymax': 117}},
 {'score': 0.9960021376609802,
  'label': 'remote',
  'box': {'xmin': 333, 'ymin': 72, 'xmax': 368, 'ymax': 187}},
 {'score': 0.9954745173454285,
  'label': 'couch',
  'box': {'xmin': 0, 'ymin': 1, 'xmax': 639, 'ymax': 473}},
 {'score': 0.9988006353378296,
  'label': 'cat',
  'box': {'xmin': 13, 'ymin': 52, 'xmax': 314, 'ymax': 470}},
 {'score': 0.9986783862113953,
  'label': 'cat',
  'box': {'xmin': 345, 'ymin': 23, 'xmax': 640, 'ymax': 368}}]
```

如上，我们列出了图像中检测到的物体，并用方框标注出了图像并分别标出了置信度分数。

以下分别展示了原始图像和预测结果

<h3 align="center">
    <a><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png" width="400"></a>
    <a><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample_post_processed.png" width="400"></a>
</h3>

你可以在[这个教程](https://huggingface.co/docs/transformers/task_summary)中了解更多关于 `pipeline` API 的知识，以及能够支持的任务。

要在你的任务上下载和使用任意预训练模型也很简单，只需三行代码。以下是基于 PyTorch 的示例：

```python
>>> from transformers import AutoTokenizer, AutoModel

>>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
>>> model = AutoModel.from_pretrained("google-bert/bert-base-uncased")

>>> inputs = tokenizer("Hello world!", return_tensors="pt")
>>> outputs = model(**inputs)
```

以下是对应的 TensorFlow 代码：

```python
>>> from transformers import AutoTokenizer, TFAutoModel

>>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
>>> model = TFAutoModel.from_pretrained("google-bert/bert-base-uncased")

>>> inputs = tokenizer("Hello world!", return_tensors="tf")
>>> outputs = model(**inputs)
```

词符化器 (tokenizer) 为所有的预训练模型提供了预处理，并可以直接对单个字符串进行调用（比如上面的例子）或对列表 (list) 调用。它会输出一个你可以在下游代码里使用或直接通过 `**` 解包表达式传给模型的词典 (dict)。

模型本身是一个常规的 [Pytorch `nn.Module`](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) 或 [TensorFlow `tf.keras.Model`](https://www.tensorflow.org/api_docs/python/tf/keras/Model)（取决于你的后端），可以常规方式使用。 [这个教程](https://huggingface.co/transformers/training.html)解释了如何将这样的模型整合到经典的 PyTorch 或 TensorFlow 训练循环中，或是如何使用我们的 `Trainer` 训练器）API 来在一个新的数据集上快速微调。

## 为什么使用 transformers？

1. 易于使用的 Sota（最先进）模型：

    - 在自然语言理解与生成、计算机视觉和音频任务上表现卓越
   
    - 对于教育工作者和实践者的入门门槛低
   
    - 高级抽象，只需了解三个类
   
    - 对所有模型统一的 API

2. 更低计算开销，更少的碳排放：

    - 研究人员可以共享预训练模型而非总是从0开始训练模型
   
    - 工程师能够降低计算用时和生产成本
   
    - 数十种模型架构、涵盖所有模态的超过 400,000 个预训练模型

3. 对于模型生命周期的每一个部分都面面俱到：

    - 训练 Sota（最先进）模型，只需 3 行代码
   
    - 模型可以在不同深度学习框架间任意迁移，随你心意
   
    - 为训练、评估和生产选择最适合的框架，它们可以无缝衔接

4. 为你的需求轻松定制专属模型和用例：

    - 我们为每种模型架构提供了多个用例来复现原论文结果
   
    - 模型内部结构尽可能一致地被公开
   
    - 模型文件可单独使用，方便进行快速实验

## 什么情况下 transformers 并不适用？

- 本库并不是模块化的神经网络工具箱。模型文件中的代码特意呈若璞玉，未经额外抽象封装，以便研究人员快速迭代魔改而不致溺于抽象和文件跳转之中。

- `Trainer` API 并非兼容任何模型，只为本库之模型优化。若是在寻找适用于通用机器学习的训练循环实现，请另寻他库。

- 尽管我们已尽力而为，[examples 目录](https://github.com/huggingface/transformers/tree/main/examples)中的脚本也仅为用例而已。对于你的特定问题，它们并不一定开箱即用，可能需要改几行代码以适配你的需求。

## 安装

### 使用 pip

这个仓库已在 Python 3.9+、Flax 0.4.1+、PyTorch 2.0+ 和 TensorFlow 2.6+ 下经过测试。

你可以在[虚拟环境](https://docs.python.org/3/library/venv.html)中安装 🤗 Transformers。如果你还不熟悉 Python 的虚拟环境，请阅此[用户说明](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)。

首先，用你打算使用的 Python 版本创建一个虚拟环境并激活。

然后，你需要安装 Flax、PyTorch 或 TensorFlow 其中之一。关于在你使用的平台上安装这些框架，请参阅 [TensorFlow 安装页面](https://www.tensorflow.org/install/), [PyTorch 安装页面](https://pytorch.org/get-started/locally/#start-locally) 或 [Flax 安装页面](https://github.com/google/flax#quick-install)。

当这些后端之一安装成功后， 🤗 Transformers 可依此安装：

```bash
pip install transformers
```

如果你想要试试这些用例或者想在正式发布前试用最新版本，并且不想等待最新版本的发布，你可以[从源代码安装](https://huggingface.co/docs/transformers/installation#installing-from-source)。

### 使用 conda

🤗 Transformers 可以通过 conda 依此安装：

```shell script
conda install conda-forge::transformers
```

> **_注意:_** 从 `huggingface` 渠道安装 `transformers` 的方式已被弃用。

要通过 conda 安装 Flax、PyTorch 或 TensorFlow 其中之一，请参阅它们各自安装页的说明。

> **_注意:_** 在 Windows 系统上，你可能会被提示启用开发者模式以便于利用缓存优化安装。如果这个选项对你不可用，请在[问题](https://github.com/huggingface/huggingface_hub/issues/1062)中告知我们

## 模型架构

🤗 Transformers 支持的[**所有的模型检查点（checkpoints）**](https://huggingface.co/models)由[用户](https://huggingface.co/users)和[组织](https://huggingface.co/organizations)上传，均与 huggingface.co [model hub](https://huggingface.co) 无缝整合。

目前的检查点数量： ![](https://img.shields.io/endpoint?url=https://huggingface.co/api/shields/models&color=brightgreen)

🤗 Transformers 目前支持如下的架构: 关于它们的简要概述请查阅[这里](https://huggingface.co/docs/transformers/model_summary).

要检查某个模型是否已有 Flax、PyTorch 或 TensorFlow 的实现，或其是否在 🤗 Tokenizers 库中有对应词符化器（tokenizer），敬请参阅[此表](https://huggingface.co/docs/transformers/index#supported-frameworks)。

这些实现已经在若干个数据集上进行了测试（参见示例脚本），并且应该与原始实现的性能相匹配。你可以在[该文档]((https://github.com/huggingface/transformers/tree/main/examples))的用例部分找到更多关于性能的详细信息

## 了解更多

| 章节                                                                             | 描述 |
|--------------------------------------------------------------------------------|-|
| [文档](https://huggingface.co/docs/transformers/)                                | 完整的 API 文档和教程 |
| [任务概览](https://huggingface.co/docs/transformers/task_summary)                  | 🤗 Transformers 支持的任务 |
| [预处理教程](https://huggingface.co/docs/transformers/preprocessing)                | 使用 `Tokenizer` 来为模型准备数据 |
| [训练和微调](https://huggingface.co/docs/transformers/training)                     | 在 PyTorch/TensorFlow 的训练循环或 `Trainer` API 中使用 🤗 Transformers 提供的模型 |
| [快速上手：微调和用例脚本](https://github.com/huggingface/transformers/tree/main/examples) | 为各种任务提供的用例脚本 |
| [模型共享和上传](https://huggingface.co/docs/transformers/model_sharing)              | 和社区上传和分享你微调的模型 |

## 引用

我们已将该库对应的[论文](https://www.aclweb.org/anthology/2020.emnlp-demos.6/)正式发表，如果你使用了 🤗 Transformers 库，请引用:

```bibtex
@inproceedings{wolf-etal-2020-transformers,
    title = "Transformers: State-of-the-Art Natural Language Processing",
    author = "Thomas Wolf and Lysandre Debut and Victor Sanh and Julien Chaumond and Clement Delangue and Anthony Moi and Pierric Cistac and Tim Rault and Rémi Louf and Morgan Funtowicz and Joe Davison and Sam Shleifer and Patrick von Platen and Clara Ma and Yacine Jernite and Julien Plu and Canwen Xu and Teven Le Scao and Sylvain Gugger and Mariama Drame and Quentin Lhoest and Alexander M. Rush",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
    month = oct,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-demos.6",
    pages = "38--45"
}
```
