<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# 🤗 Transformers简介

为 [PyTorch](https://pytorch.org/)、[TensorFlow](https://www.tensorflow.org/) 和 [JAX](https://jax.readthedocs.io/en/latest/) 打造的先进的机器学习工具.

🤗 Transformers 提供了可以轻松地下载并且训练先进的预训练模型的 API 和工具。使用预训练模型可以减少计算消耗和碳排放，并且节省从头训练所需要的时间和资源。这些模型支持不同模态中的常见任务，比如：

📝 **自然语言处理**：文本分类、命名实体识别、问答、语言建模、摘要、翻译、多项选择和文本生成。<br>
🖼️ **机器视觉**：图像分类、目标检测和语义分割。<br>
🗣️ **音频**：自动语音识别和音频分类。<br>
🐙 **多模态**：表格问答、光学字符识别、从扫描文档提取信息、视频分类和视觉问答。

🤗 Transformers 支持在 PyTorch、TensorFlow 和 JAX 上的互操作性. 这给在模型的每个阶段使用不同的框架带来了灵活性；在一个框架中使用几行代码训练一个模型，然后在另一个框架中加载它并进行推理。模型也可以被导出为 ONNX 和 TorchScript 格式，用于在生产环境中部署。

马上加入在 [Hub](https://huggingface.co/models)、[论坛](https://discuss.huggingface.co/) 或者 [Discord](https://discord.com/invite/JfAtkvEtRb) 上正在快速发展的社区吧！

## 如果你需要来自 Hugging Face 团队的个性化支持

<a target="_blank" href="https://huggingface.co/support">
    <img alt="HuggingFace Expert Acceleration Program" src="https://cdn-media.huggingface.co/marketing/transformers/new-support-improved.png" style="width: 100%; max-width: 600px; border: 1px solid #eee; border-radius: 4px; box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);">
</a>

## 目录

这篇文档由以下 5 个章节组成：

- **开始使用** 包含了库的快速上手和安装说明，便于配置和运行。
- **教程** 是一个初学者开始的好地方。本章节将帮助你获得你会用到的使用这个库的基本技能。
- **操作指南** 向你展示如何实现一个特定目标，比如为语言建模微调一个预训练模型或者如何创造并分享个性化模型。
- **概念指南** 对 🤗 Transformers 的模型，任务和设计理念背后的基本概念和思想做了更多的讨论和解释。
- **API 介绍** 描述了所有的类和函数：

  - **主要类别** 详述了配置（configuration）、模型（model）、分词器（tokenizer）和流水线（pipeline）这几个最重要的类。
  - **模型** 详述了在这个库中和每个模型实现有关的类和函数。
  - **内部帮助** 详述了内部使用的工具类和函数。

### 支持的模型和框架

下表展示了库中对每个模型的支持情况，如是否具有 Python 分词器（表中的“Tokenizer slow”）、是否具有由 🤗 Tokenizers 库支持的快速分词器（表中的“Tokenizer fast”）、是否支持 Jax（通过 Flax）、PyTorch 与 TensorFlow。

<!--This table is updated automatically from the auto modules with _make fix-copies_. Do not update manually!-->

|                                  模型                                   | PyTorch 支持 | TensorFlow 支持 | Flax 支持 |
|:------------------------------------------------------------------------:|:---------------:|:------------------:|:------------:|
|                        [ALBERT](../en/model_doc/albert.md)                        |       ✅        |         ✅         |      ✅      |
|                         [ALIGN](../en/model_doc/align.md)                         |       ✅        |         ❌         |      ❌      |
|                       [AltCLIP](../en/model_doc/altclip)                       |       ✅        |         ❌         |      ❌      |
| [Audio Spectrogram Transformer](../en/model_doc/audio-spectrogram-transformer) |       ✅        |         ❌         |      ❌      |
|                    [Autoformer](../en/model_doc/autoformer)                    |       ✅        |         ❌         |      ❌      |
|                          [Bark](../en/model_doc/bark)                          |       ✅        |         ❌         |      ❌      |
|                          [BART](../en/model_doc/bart)                          |       ✅        |         ✅         |      ✅      |
|                       [BARThez](../en/model_doc/barthez)                       |       ✅        |         ✅         |      ✅      |
|                       [BARTpho](../en/model_doc/bartpho)                       |       ✅        |         ✅         |      ✅      |
|                          [BEiT](../en/model_doc/beit)                          |       ✅        |         ❌         |      ✅      |
|                          [BERT](../en/model_doc/bert)                          |       ✅        |         ✅         |      ✅      |
|               [Bert Generation](../en/model_doc/bert-generation)               |       ✅        |         ❌         |      ❌      |
|                 [BertJapanese](../en/model_doc/bert-japanese)                  |       ✅        |         ✅         |      ✅      |
|                      [BERTweet](../en/model_doc/bertweet)                      |       ✅        |         ✅         |      ✅      |
|                      [BigBird](../en/model_doc/big_bird)                       |       ✅        |         ❌         |      ✅      |
|               [BigBird-Pegasus](../en/model_doc/bigbird_pegasus)               |       ✅        |         ❌         |      ❌      |
|                        [BioGpt](../en/model_doc/biogpt)                        |       ✅        |         ❌         |      ❌      |
|                           [BiT](../en/model_doc/bit)                           |       ✅        |         ❌         |      ❌      |
|                    [Blenderbot](../en/model_doc/blenderbot)                    |       ✅        |         ✅         |      ✅      |
|              [BlenderbotSmall](../en/model_doc/blenderbot-small)               |       ✅        |         ✅         |      ✅      |
|                          [BLIP](../en/model_doc/blip)                          |       ✅        |         ✅         |      ❌      |
|                        [BLIP-2](../en/model_doc/blip-2)                        |       ✅        |         ❌         |      ❌      |
|                         [BLOOM](../en/model_doc/bloom)                         |       ✅        |         ❌         |      ✅      |
|                          [BORT](../en/model_doc/bort)                          |       ✅        |         ✅         |      ✅      |
|                   [BridgeTower](../en/model_doc/bridgetower)                   |       ✅        |         ❌         |      ❌      |
|                          [BROS](../en/model_doc/bros)                          |       ✅        |         ❌         |      ❌      |
|                          [ByT5](../en/model_doc/byt5)                          |       ✅        |         ✅         |      ✅      |
|                     [CamemBERT](../en/model_doc/camembert)                     |       ✅        |         ✅         |      ❌      |
|                        [CANINE](../en/model_doc/canine)                        |       ✅        |         ❌         |      ❌      |
|                  [Chinese-CLIP](../en/model_doc/chinese_clip)                  |       ✅        |         ❌         |      ❌      |
|                          [CLAP](../en/model_doc/clap)                          |       ✅        |         ❌         |      ❌      |
|                          [CLIP](../en/model_doc/clip)                          |       ✅        |         ✅         |      ✅      |
|                       [CLIPSeg](../en/model_doc/clipseg)                       |       ✅        |         ❌         |      ❌      |
|                          [CLVP](../en/model_doc/clvp)                          |       ✅        |         ❌         |      ❌      |
|                       [CodeGen](../en/model_doc/codegen)                       |       ✅        |         ❌         |      ❌      |
|                    [CodeLlama](../en/model_doc/code_llama)                     |       ✅        |         ❌         |      ✅      |
|              [Conditional DETR](../en/model_doc/conditional_detr)              |       ✅        |         ❌         |      ❌      |
|                      [ConvBERT](../en/model_doc/convbert)                      |       ✅        |         ✅         |      ❌      |
|                      [ConvNeXT](../en/model_doc/convnext)                      |       ✅        |         ✅         |      ❌      |
|                    [ConvNeXTV2](../en/model_doc/convnextv2)                    |       ✅        |         ✅         |      ❌      |
|                           [CPM](../en/model_doc/cpm)                           |       ✅        |         ✅         |      ✅      |
|                       [CPM-Ant](../en/model_doc/cpmant)                        |       ✅        |         ❌         |      ❌      |
|                          [CTRL](../en/model_doc/ctrl)                          |       ✅        |         ✅         |      ❌      |
|                           [CvT](../en/model_doc/cvt)                           |       ✅        |         ✅         |      ❌      |
|                   [Data2VecAudio](../en/model_doc/data2vec)                    |       ✅        |         ❌         |      ❌      |
|                    [Data2VecText](../en/model_doc/data2vec)                    |       ✅        |         ❌         |      ❌      |
|                   [Data2VecVision](../en/model_doc/data2vec)                   |       ✅        |         ✅         |      ❌      |
|                       [DeBERTa](../en/model_doc/deberta)                       |       ✅        |         ✅         |      ❌      |
|                    [DeBERTa-v2](../en/model_doc/deberta-v2)                    |       ✅        |         ✅         |      ❌      |
|          [Decision Transformer](../en/model_doc/decision_transformer)          |       ✅        |         ❌         |      ❌      |
|               [Deformable DETR](../en/model_doc/deformable_detr)               |       ✅        |         ❌         |      ❌      |
|                          [DeiT](../en/model_doc/deit)                          |       ✅        |         ✅         |      ❌      |
|                        [DePlot](../en/model_doc/deplot)                        |       ✅        |         ❌         |      ❌      |
|                [Depth Anything](../en/model_doc/depth_anything)                |       ✅        |         ❌         |      ❌      |
|                          [DETA](../en/model_doc/deta)                          |       ✅        |         ❌         |      ❌      |
|                          [DETR](../en/model_doc/detr)                          |       ✅        |         ❌         |      ❌      |
|                      [DialoGPT](../en/model_doc/dialogpt)                      |       ✅        |         ✅         |      ✅      |
|                         [DiNAT](../en/model_doc/dinat)                         |       ✅        |         ❌         |      ❌      |
|                        [DINOv2](../en/model_doc/dinov2)                        |       ✅        |         ❌         |      ❌      |
|                    [DistilBERT](../en/model_doc/distilbert)                    |       ✅        |         ✅         |      ✅      |
|                           [DiT](../en/model_doc/dit)                           |       ✅        |         ❌         |      ✅      |
|                       [DonutSwin](../en/model_doc/donut)                       |       ✅        |         ❌         |      ❌      |
|                           [DPR](../en/model_doc/dpr)                           |       ✅        |         ✅         |      ❌      |
|                           [DPT](../en/model_doc/dpt)                           |       ✅        |         ❌         |      ❌      |
|               [EfficientFormer](../en/model_doc/efficientformer)               |       ✅        |         ✅         |      ❌      |
|                  [EfficientNet](../en/model_doc/efficientnet)                  |       ✅        |         ❌         |      ❌      |
|                       [ELECTRA](../en/model_doc/electra)                       |       ✅        |         ✅         |      ✅      |
|                       [EnCodec](../en/model_doc/encodec)                       |       ✅        |         ❌         |      ❌      |
|               [Encoder decoder](../en/model_doc/encoder-decoder)               |       ✅        |         ✅         |      ✅      |
|                         [ERNIE](../en/model_doc/ernie)                         |       ✅        |         ❌         |      ❌      |
|                       [ErnieM](../en/model_doc/ernie_m)                        |       ✅        |         ❌         |      ❌      |
|                           [ESM](../en/model_doc/esm)                           |       ✅        |         ✅         |      ❌      |
|              [FairSeq Machine-Translation](../en/model_doc/fsmt)               |       ✅        |         ❌         |      ❌      |
|                        [Falcon](../en/model_doc/falcon)                        |       ✅        |         ❌         |      ❌      |
|         [FastSpeech2Conformer](../en/model_doc/fastspeech2_conformer)          |       ✅        |         ❌         |      ❌      |
|                       [FLAN-T5](../en/model_doc/flan-t5)                       |       ✅        |         ✅         |      ✅      |
|                      [FLAN-UL2](../en/model_doc/flan-ul2)                      |       ✅        |         ✅         |      ✅      |
|                      [FlauBERT](../en/model_doc/flaubert)                      |       ✅        |         ✅         |      ❌      |
|                         [FLAVA](../en/model_doc/flava)                         |       ✅        |         ❌         |      ❌      |
|                          [FNet](../en/model_doc/fnet)                          |       ✅        |         ❌         |      ❌      |
|                      [FocalNet](../en/model_doc/focalnet)                      |       ✅        |         ❌         |      ❌      |
|                  [Funnel Transformer](../en/model_doc/funnel)                  |       ✅        |         ✅         |      ❌      |
|                          [Fuyu](../en/model_doc/fuyu)                          |       ✅        |         ❌         |      ❌      |
|                         [Gemma](../en/model_doc/gemma)                         |       ✅        |         ❌         |      ✅      |
|                           [GIT](../en/model_doc/git)                           |       ✅        |         ❌         |      ❌      |
|                          [GLPN](../en/model_doc/glpn)                          |       ✅        |         ❌         |      ❌      |
|                       [GPT Neo](../en/model_doc/gpt_neo)                       |       ✅        |         ❌         |      ✅      |
|                      [GPT NeoX](../en/model_doc/gpt_neox)                      |       ✅        |         ❌         |      ❌      |
|             [GPT NeoX Japanese](../en/model_doc/gpt_neox_japanese)             |       ✅        |         ❌         |      ❌      |
|                         [GPT-J](../en/model_doc/gptj)                          |       ✅        |         ✅         |      ✅      |
|                       [GPT-Sw3](../en/model_doc/gpt-sw3)                       |       ✅        |         ✅         |      ✅      |
|                   [GPTBigCode](../en/model_doc/gpt_bigcode)                    |       ✅        |         ❌         |      ❌      |
|               [GPTSAN-japanese](../en/model_doc/gptsan-japanese)               |       ✅        |         ❌         |      ❌      |
|                    [Graphormer](../en/model_doc/graphormer)                    |       ✅        |         ❌         |      ❌      |
|                      [GroupViT](../en/model_doc/groupvit)                      |       ✅        |         ✅         |      ❌      |
|                       [HerBERT](../en/model_doc/herbert)                       |       ✅        |         ✅         |      ✅      |
|                        [Hubert](../en/model_doc/hubert)                        |       ✅        |         ✅         |      ❌      |
|                        [I-BERT](../en/model_doc/ibert)                         |       ✅        |         ❌         |      ❌      |
|                       [IDEFICS](../en/model_doc/idefics)                       |       ✅        |         ❌         |      ❌      |
|                      [ImageGPT](../en/model_doc/imagegpt)                      |       ✅        |         ❌         |      ❌      |
|                      [Informer](../en/model_doc/informer)                      |       ✅        |         ❌         |      ❌      |
|                  [InstructBLIP](../en/model_doc/instructblip)                  |       ✅        |         ❌         |      ❌      |
|                       [Jukebox](../en/model_doc/jukebox)                       |       ✅        |         ❌         |      ❌      |
|                      [KOSMOS-2](../en/model_doc/kosmos-2)                      |       ✅        |         ❌         |      ❌      |
|                      [LayoutLM](../en/model_doc/layoutlm)                      |       ✅        |         ✅         |      ❌      |
|                    [LayoutLMv2](../en/model_doc/layoutlmv2)                    |       ✅        |         ❌         |      ❌      |
|                    [LayoutLMv3](../en/model_doc/layoutlmv3)                    |       ✅        |         ✅         |      ❌      |
|                     [LayoutXLM](../en/model_doc/layoutxlm)                     |       ✅        |         ❌         |      ❌      |
|                           [LED](../en/model_doc/led)                           |       ✅        |         ✅         |      ❌      |
|                         [LeViT](../en/model_doc/levit)                         |       ✅        |         ❌         |      ❌      |
|                          [LiLT](../en/model_doc/lilt)                          |       ✅        |         ❌         |      ❌      |
|                         [LLaMA](../en/model_doc/llama)                         |       ✅        |         ❌         |      ✅      |
|                        [Llama2](../en/model_doc/llama2)                        |       ✅        |         ❌         |      ✅      |
|                         [LLaVa](../en/model_doc/llava)                         |       ✅        |         ❌         |      ❌      |
|                    [Longformer](../en/model_doc/longformer)                    |       ✅        |         ✅         |      ❌      |
|                        [LongT5](../en/model_doc/longt5)                        |       ✅        |         ❌         |      ✅      |
|                          [LUKE](../en/model_doc/luke)                          |       ✅        |         ❌         |      ❌      |
|                        [LXMERT](../en/model_doc/lxmert)                        |       ✅        |         ✅         |      ❌      |
|                        [M-CTC-T](../en/model_doc/mctct)                        |       ✅        |         ❌         |      ❌      |
|                       [M2M100](../en/model_doc/m2m_100)                        |       ✅        |         ❌         |      ❌      |
|                    [MADLAD-400](../en/model_doc/madlad-400)                    |       ✅        |         ✅         |      ✅      |
|                        [Marian](../en/model_doc/marian)                        |       ✅        |         ✅         |      ✅      |
|                      [MarkupLM](../en/model_doc/markuplm)                      |       ✅        |         ❌         |      ❌      |
|                   [Mask2Former](../en/model_doc/mask2former)                   |       ✅        |         ❌         |      ❌      |
|                    [MaskFormer](../en/model_doc/maskformer)                    |       ✅        |         ❌         |      ❌      |
|                        [MatCha](../en/model_doc/matcha)                        |       ✅        |         ❌         |      ❌      |
|                         [mBART](../en/model_doc/mbart)                         |       ✅        |         ✅         |      ✅      |
|                      [mBART-50](../en/model_doc/mbart50)                       |       ✅        |         ✅         |      ✅      |
|                          [MEGA](../en/model_doc/mega)                          |       ✅        |         ❌         |      ❌      |
|                 [Megatron-BERT](../en/model_doc/megatron-bert)                 |       ✅        |         ❌         |      ❌      |
|                 [Megatron-GPT2](../en/model_doc/megatron_gpt2)                 |       ✅        |         ✅         |      ✅      |
|                       [MGP-STR](../en/model_doc/mgp-str)                       |       ✅        |         ❌         |      ❌      |
|                       [Mistral](../en/model_doc/mistral)                       |       ✅        |         ❌         |      ✅      |
|                       [Mixtral](../en/model_doc/mixtral)                       |       ✅        |         ❌         |      ❌      |
|                         [mLUKE](../en/model_doc/mluke)                         |       ✅        |         ❌         |      ❌      |
|                           [MMS](../en/model_doc/mms)                           |       ✅        |         ✅         |      ✅      |
|                    [MobileBERT](../en/model_doc/mobilebert)                    |       ✅        |         ✅         |      ❌      |
|                  [MobileNetV1](../en/model_doc/mobilenet_v1)                   |       ✅        |         ❌         |      ❌      |
|                  [MobileNetV2](../en/model_doc/mobilenet_v2)                   |       ✅        |         ❌         |      ❌      |
|                     [MobileViT](../en/model_doc/mobilevit)                     |       ✅        |         ✅         |      ❌      |
|                   [MobileViTV2](../en/model_doc/mobilevitv2)                   |       ✅        |         ❌         |      ❌      |
|                         [MPNet](../en/model_doc/mpnet)                         |       ✅        |         ✅         |      ❌      |
|                           [MPT](../en/model_doc/mpt)                           |       ✅        |         ❌         |      ❌      |
|                           [MRA](../en/model_doc/mra)                           |       ✅        |         ❌         |      ❌      |
|                           [MT5](../en/model_doc/mt5)                           |       ✅        |         ✅         |      ✅      |
|                      [MusicGen](../en/model_doc/musicgen)                      |       ✅        |         ❌         |      ❌      |
|                           [MVP](../en/model_doc/mvp)                           |       ✅        |         ❌         |      ❌      |
|                           [NAT](../en/model_doc/nat)                           |       ✅        |         ❌         |      ❌      |
|                         [Nezha](../en/model_doc/nezha)                         |       ✅        |         ❌         |      ❌      |
|                          [NLLB](../en/model_doc/nllb)                          |       ✅        |         ❌         |      ❌      |
|                      [NLLB-MOE](../en/model_doc/nllb-moe)                      |       ✅        |         ❌         |      ❌      |
|                        [Nougat](../en/model_doc/nougat)                        |       ✅        |         ✅         |      ✅      |
|                 [Nyströmformer](../en/model_doc/nystromformer)                 |       ✅        |         ❌         |      ❌      |
|                     [OneFormer](../en/model_doc/oneformer)                     |       ✅        |         ❌         |      ❌      |
|                    [OpenAI GPT](../en/model_doc/openai-gpt)                    |       ✅        |         ✅         |      ❌      |
|                      [OpenAI GPT-2](../en/model_doc/gpt2)                      |       ✅        |         ✅         |      ✅      |
|                    [OpenLlama](../en/model_doc/open-llama)                     |       ✅        |         ❌         |      ❌      |
|                           [OPT](../en/model_doc/opt)                           |       ✅        |         ✅         |      ✅      |
|                       [OWL-ViT](../en/model_doc/owlvit)                        |       ✅        |         ❌         |      ❌      |
|                         [OWLv2](../en/model_doc/owlv2)                         |       ✅        |         ❌         |      ❌      |
|                  [PatchTSMixer](../en/model_doc/patchtsmixer)                  |       ✅        |         ❌         |      ❌      |
|                      [PatchTST](../en/model_doc/patchtst)                      |       ✅        |         ❌         |      ❌      |
|                       [Pegasus](../en/model_doc/pegasus)                       |       ✅        |         ✅         |      ✅      |
|                     [PEGASUS-X](../en/model_doc/pegasus_x)                     |       ✅        |         ❌         |      ❌      |
|                     [Perceiver](../en/model_doc/perceiver)                     |       ✅        |         ❌         |      ❌      |
|                     [Persimmon](../en/model_doc/persimmon)                     |       ✅        |         ❌         |      ❌      |
|                           [Phi](../en/model_doc/phi)                           |       ✅        |         ❌         |      ❌      |
|                       [PhoBERT](../en/model_doc/phobert)                       |       ✅        |         ✅         |      ✅      |
|                    [Pix2Struct](../en/model_doc/pix2struct)                    |       ✅        |         ❌         |      ❌      |
|                        [PLBart](../en/model_doc/plbart)                        |       ✅        |         ❌         |      ❌      |
|                    [PoolFormer](../en/model_doc/poolformer)                    |       ✅        |         ❌         |      ❌      |
|                     [Pop2Piano](../en/model_doc/pop2piano)                     |       ✅        |         ❌         |      ❌      |
|                    [ProphetNet](../en/model_doc/prophetnet)                    |       ✅        |         ❌         |      ❌      |
|                           [PVT](../en/model_doc/pvt)                           |       ✅        |         ❌         |      ❌      |
|                       [QDQBert](../en/model_doc/qdqbert)                       |       ✅        |         ❌         |      ❌      |
|                         [Qwen2](../en/model_doc/qwen2)                         |       ✅        |         ❌         |      ❌      |
|                           [RAG](../en/model_doc/rag)                           |       ✅        |         ✅         |      ❌      |
|                         [REALM](../en/model_doc/realm)                         |       ✅        |         ❌         |      ❌      |
|                      [Reformer](../en/model_doc/reformer)                      |       ✅        |         ❌         |      ❌      |
|                        [RegNet](../en/model_doc/regnet)                        |       ✅        |         ✅         |      ✅      |
|                       [RemBERT](../en/model_doc/rembert)                       |       ✅        |         ✅         |      ❌      |
|                        [ResNet](../en/model_doc/resnet)                        |       ✅        |         ✅         |      ✅      |
|                     [RetriBERT](../en/model_doc/retribert)                     |       ✅        |         ❌         |      ❌      |
|                       [RoBERTa](../en/model_doc/roberta)                       |       ✅        |         ✅         |      ✅      |
|          [RoBERTa-PreLayerNorm](../en/model_doc/roberta-prelayernorm)          |       ✅        |         ✅         |      ✅      |
|                      [RoCBert](../en/model_doc/roc_bert)                       |       ✅        |         ❌         |      ❌      |
|                      [RoFormer](../en/model_doc/roformer)                      |       ✅        |         ✅         |      ✅      |
|                          [RWKV](../en/model_doc/rwkv)                          |       ✅        |         ❌         |      ❌      |
|                         [RWKV6](../en/model_doc/rwkv6)                         |       ✅        |         ❌         |      ❌      |
|                           [SAM](../en/model_doc/sam)                           |       ✅        |         ✅         |      ❌      |
|                  [SeamlessM4T](../en/model_doc/seamless_m4t)                   |       ✅        |         ❌         |      ❌      |
|                [SeamlessM4Tv2](../en/model_doc/seamless_m4t_v2)                |       ✅        |         ❌         |      ❌      |
|                     [SegFormer](../en/model_doc/segformer)                     |       ✅        |         ✅         |      ❌      |
|                        [SegGPT](../en/model_doc/seggpt)                        |       ✅        |         ❌         |      ❌      |
|                           [SEW](../en/model_doc/sew)                           |       ✅        |         ❌         |      ❌      |
|                         [SEW-D](../en/model_doc/sew-d)                         |       ✅        |         ❌         |      ❌      |
|                        [SigLIP](../en/model_doc/siglip)                        |       ✅        |         ❌         |      ❌      |
|        [Speech Encoder decoder](../en/model_doc/speech-encoder-decoder)        |       ✅        |         ❌         |      ✅      |
|                 [Speech2Text](../en/model_doc/speech_to_text)                  |       ✅        |         ✅         |      ❌      |
|                      [SpeechT5](../en/model_doc/speecht5)                      |       ✅        |         ❌         |      ❌      |
|                      [Splinter](../en/model_doc/splinter)                      |       ✅        |         ❌         |      ❌      |
|                   [SqueezeBERT](../en/model_doc/squeezebert)                   |       ✅        |         ❌         |      ❌      |
|                      [StableLm](../en/model_doc/stablelm)                      |       ✅        |         ❌         |      ❌      |
|                    [Starcoder2](../en/model_doc/starcoder2)                    |       ✅        |         ❌         |      ❌      |
|                   [SwiftFormer](../en/model_doc/swiftformer)                   |       ✅        |         ❌         |      ❌      |
|                    [Swin Transformer](../en/model_doc/swin)                    |       ✅        |         ✅         |      ❌      |
|                 [Swin Transformer V2](../en/model_doc/swinv2)                  |       ✅        |         ❌         |      ❌      |
|                       [Swin2SR](../en/model_doc/swin2sr)                       |       ✅        |         ❌         |      ❌      |
|           [SwitchTransformers](../en/model_doc/switch_transformers)            |       ✅        |         ❌         |      ❌      |
|                            [T5](../en/model_doc/t5)                            |       ✅        |         ✅         |      ✅      |
|                        [T5v1.1](../en/model_doc/t5v1.1)                        |       ✅        |         ✅         |      ✅      |
|             [Table Transformer](../en/model_doc/table-transformer)             |       ✅        |         ❌         |      ❌      |
|                         [TAPAS](../en/model_doc/tapas)                         |       ✅        |         ✅         |      ❌      |
|                         [TAPEX](../en/model_doc/tapex)                         |       ✅        |         ✅         |      ✅      |
|       [Time Series Transformer](../en/model_doc/time_series_transformer)       |       ✅        |         ❌         |      ❌      |
|                   [TimeSformer](../en/model_doc/timesformer)                   |       ✅        |         ❌         |      ❌      |
|        [Trajectory Transformer](../en/model_doc/trajectory_transformer)        |       ✅        |         ❌         |      ❌      |
|                  [Transformer-XL](../en/model_doc/transfo-xl)                  |       ✅        |         ✅         |      ❌      |
|                         [TrOCR](../en/model_doc/trocr)                         |       ✅        |         ❌         |      ❌      |
|                          [TVLT](../en/model_doc/tvlt)                          |       ✅        |         ❌         |      ❌      |
|                           [TVP](../en/model_doc/tvp)                           |       ✅        |         ❌         |      ❌      |
|                           [UL2](../en/model_doc/ul2)                           |       ✅        |         ✅         |      ✅      |
|                          [UMT5](../en/model_doc/umt5)                          |       ✅        |         ❌         |      ❌      |
|                     [UniSpeech](../en/model_doc/unispeech)                     |       ✅        |         ❌         |      ❌      |
|                 [UniSpeechSat](../en/model_doc/unispeech-sat)                  |       ✅        |         ❌         |      ❌      |
|                       [UnivNet](../en/model_doc/univnet)                       |       ✅        |         ❌         |      ❌      |
|                       [UPerNet](../en/model_doc/upernet)                       |       ✅        |         ❌         |      ❌      |
|                           [VAN](../en/model_doc/van)                           |       ✅        |         ❌         |      ❌      |
|                      [VideoMAE](../en/model_doc/videomae)                      |       ✅        |         ❌         |      ❌      |
|                          [ViLT](../en/model_doc/vilt)                          |       ✅        |         ❌         |      ❌      |
|                      [VipLlava](../en/model_doc/vipllava)                      |       ✅        |         ❌         |      ❌      |
|        [Vision Encoder decoder](../en/model_doc/vision-encoder-decoder)        |       ✅        |         ✅         |      ✅      |
|       [VisionTextDualEncoder](../en/model_doc/vision-text-dual-encoder)        |       ✅        |         ✅         |      ✅      |
|                   [VisualBERT](../en/model_doc/visual_bert)                    |       ✅        |         ❌         |      ❌      |
|                           [ViT](../en/model_doc/vit)                           |       ✅        |         ✅         |      ✅      |
|                    [ViT Hybrid](../en/model_doc/vit_hybrid)                    |       ✅        |         ❌         |      ❌      |
|                        [VitDet](../en/model_doc/vitdet)                        |       ✅        |         ❌         |      ❌      |
|                       [ViTMAE](../en/model_doc/vit_mae)                        |       ✅        |         ✅         |      ❌      |
|                      [ViTMatte](../en/model_doc/vitmatte)                      |       ✅        |         ❌         |      ❌      |
|                       [ViTMSN](../en/model_doc/vit_msn)                        |       ✅        |         ❌         |      ❌      |
|                          [VITS](../en/model_doc/vits)                          |       ✅        |         ❌         |      ❌      |
|                         [ViViT](../en/model_doc/vivit)                         |       ✅        |         ❌         |      ❌      |
|                      [Wav2Vec2](../en/model_doc/wav2vec2)                      |       ✅        |         ✅         |      ✅      |
|                 [Wav2Vec2-BERT](../en/model_doc/wav2vec2-bert)                 |       ✅        |         ❌         |      ❌      |
|            [Wav2Vec2-Conformer](../en/model_doc/wav2vec2-conformer)            |       ✅        |         ❌         |      ❌      |
|              [Wav2Vec2Phoneme](../en/model_doc/wav2vec2_phoneme)               |       ✅        |         ✅         |      ✅      |
|                         [WavLM](../en/model_doc/wavlm)                         |       ✅        |         ❌         |      ❌      |
|                       [Whisper](../en/model_doc/whisper)                       |       ✅        |         ✅         |      ✅      |
|                        [X-CLIP](../en/model_doc/xclip)                         |       ✅        |         ❌         |      ❌      |
|                         [X-MOD](../en/model_doc/xmod)                          |       ✅        |         ❌         |      ❌      |
|                          [XGLM](../en/model_doc/xglm)                          |       ✅        |         ✅         |      ✅      |
|                           [XLM](../en/model_doc/xlm)                           |       ✅        |         ✅         |      ❌      |
|                [XLM-ProphetNet](../en/model_doc/xlm-prophetnet)                |       ✅        |         ❌         |      ❌      |
|                   [XLM-RoBERTa](../en/model_doc/xlm-roberta)                   |       ✅        |         ✅         |      ✅      |
|                [XLM-RoBERTa-XL](../en/model_doc/xlm-roberta-xl)                |       ✅        |         ❌         |      ❌      |
|                         [XLM-V](../en/model_doc/xlm-v)                         |       ✅        |         ✅         |      ✅      |
|                         [XLNet](../en/model_doc/xlnet)                         |       ✅        |         ✅         |      ❌      |
|                         [XLS-R](../en/model_doc/xls_r)                         |       ✅        |         ✅         |      ✅      |
|                 [XLSR-Wav2Vec2](../en/model_doc/xlsr_wav2vec2)                 |       ✅        |         ✅         |      ✅      |
|                         [YOLOS](../en/model_doc/yolos)                         |       ✅        |         ❌         |      ❌      |
|                          [YOSO](../en/model_doc/yoso)                          |       ✅        |         ❌         |      ❌      |

<!-- End table-->
