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

# How 🤗 Transformers solve tasks

In [What 🤗 Transformers can do](task_summary), you learned about natural language processing (NLP), speech and audio, computer vision tasks, and some important applications of them. This page will look closely at how models solve these tasks and explain what's happening under the hood. There are many ways to solve a given task, some models may implement certain techniques or even approach the task from a new angle, but for Transformer models, the general idea is the same. Owing to its flexible architecture, most models are a variant of an encoder, a decoder, or an encoder-decoder structure. In addition to Transformer models, our library also has several convolutional neural networks (CNNs), which are still used today for computer vision tasks. We'll also explain how a modern CNN works.

To explain how tasks are solved, we'll walk through what goes on inside the model to output useful predictions.

- [Wav2Vec2](model_doc/wav2vec2) for audio classification and automatic speech recognition (ASR)
- [Vision Transformer (ViT)](model_doc/vit) and [ConvNeXT](model_doc/convnext) for image classification
- [DETR](model_doc/detr) for object detection
- [Mask2Former](model_doc/mask2former) for image segmentation
- [GLPN](model_doc/glpn) for depth estimation
- [BERT](model_doc/bert) for NLP tasks like text classification, token classification and question answering that use an encoder
- [GPT2](model_doc/gpt2) for NLP tasks like text generation that use a decoder
- [BART](model_doc/bart) for NLP tasks like summarization and translation that use an encoder-decoder

<Tip>

Before you go further, it is good to have some basic knowledge of the original Transformer architecture. Knowing how encoders, decoders, and attention work will aid you in understanding how different Transformer models work. If you're just getting started or need a refresher, check out our [course](https://huggingface.co/course/chapter1/4?fw=pt) for more information! 

</Tip>

## Speech and audio

[Wav2Vec2](model_doc/wav2vec2) is a self-supervised model pretrained on unlabeled speech data and finetuned on labeled data for audio classification and automatic speech recognition. 

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/wav2vec2_architecture.png"/>
</div>

This model has four main components:

1. A *feature encoder* takes the raw audio waveform, normalizes it to zero mean and unit variance, and converts it into a sequence of feature vectors that are each 20ms long.

2. Waveforms are continuous by nature, so they can't be divided into separate units like a sequence of text can be split into words. That's why the feature vectors are passed to a *quantization module*, which aims to learn discrete speech units. The speech unit is chosen from a collection of codewords, known as a *codebook* (you can think of this as the vocabulary). From the codebook, the vector or speech unit, that best represents the continuous audio input is chosen and forwarded through the model.

3. About half of the feature vectors are randomly masked, and the masked feature vector is fed to a *context network*, which is a Transformer encoder that also adds relative positional embeddings.

4. The pretraining objective of the context network is a *contrastive task*. The model has to predict the true quantized speech representation of the masked prediction from a set of false ones, encouraging the model to find the most similar context vector and quantized speech unit (the target label).

Now that wav2vec2 is pretrained, you can finetune it on your data for audio classification or automatic speech recognition!

### Audio classification

To use the pretrained model for audio classification, add a sequence classification head on top of the base Wav2Vec2 model. The classification head is a linear layer that accepts the encoder's hidden states. The hidden states represent the learned features from each audio frame which can have varying lengths. To create one vector of fixed-length, the hidden states are pooled first and then transformed into logits over the class labels. The cross-entropy loss is calculated between the logits and target to find the most likely class.

Ready to try your hand at audio classification? Check out our complete [audio classification guide](tasks/audio_classification) to learn how to finetune Wav2Vec2 and use it for inference!

### Automatic speech recognition

To use the pretrained model for automatic speech recognition, add a language modeling head on top of the base Wav2Vec2 model for [connectionist temporal classification (CTC)](glossary#connectionist-temporal-classification-ctc). The language modeling head is a linear layer that accepts the encoder's hidden states and transforms them into logits. Each logit represents a token class (the number of tokens comes from the task vocabulary). The CTC loss is calculated between the logits and targets to find the most likely sequence of tokens, which are then decoded into a transcription.

Ready to try your hand at automatic speech recognition? Check out our complete [automatic speech recognition guide](tasks/asr) to learn how to finetune Wav2Vec2 and use it for inference!

## Computer vision

There are two ways to approach computer vision tasks:

1. Split an image into a sequence of patches and process them in parallel with a Transformer.
2. Use a modern CNN, like [ConvNeXT](model_doc/convnext), which relies on convolutional layers but adopts modern network designs.

<Tip>

A third approach mixes Transformers with convolutions (for example, [Convolutional Vision Transformer](model_doc/cvt) or [LeViT](model_doc/levit)). We won't discuss those because they just combine the two approaches we examine here.

</Tip>

ViT and ConvNeXT are commonly used for image classification, but for other vision tasks like object detection, segmentation, and depth estimation, we'll look at DETR, Mask2Former and GLPN, respectively; these models are better suited for those tasks.

### Image classification

ViT and ConvNeXT can both be used for image classification; the main difference is that ViT uses an attention mechanism while ConvNeXT uses convolutions.

#### Transformer

[ViT](model_doc/vit) replaces convolutions entirely with a pure Transformer architecture. If you're familiar with the original Transformer, then you're already most of the way toward understanding ViT.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/vit_architecture.jpg"/>
</div>

The main change ViT introduced was in how images are fed to a Transformer:

1. An image is split into square non-overlapping patches, each of which gets turned into a vector or *patch embedding*. The patch embeddings are generated from a convolutional 2D layer which creates the proper input dimensions (which for a base Transformer is 768 values for each patch embedding). If you had a 224x224 pixel image, you could split it into 196 16x16 image patches. Just like how text is tokenized into words, an image is "tokenized" into a sequence of patches.

2. A *learnable embedding* - a special `[CLS]` token - is added to the beginning of the patch embeddings just like BERT. The final hidden state of the `[CLS]` token is used as the input to the attached classification head; other outputs are ignored. This token helps the model learn how to encode a representation of the image.

3. The last thing to add to the patch and learnable embeddings are the *position embeddings* because the model doesn't know how the image patches are ordered. The position embeddings are also learnable and have the same size as the patch embeddings. Finally, all of the embeddings are passed to the Transformer encoder.

4. The output, specifically only the output with the `[CLS]` token, is passed to a multilayer perceptron head (MLP). ViT's pretraining objective is simply classification. Like other classification heads, the MLP head converts the output into logits over the class labels and calculates the cross-entropy loss to find the most likely class.

Ready to try your hand at image classification? Check out our complete [image classification guide](tasks/image_classification) to learn how to finetune ViT and use it for inference!

#### CNN

<Tip>

This section briefly explains convolutions, but it'd be helpful to have a prior understanding of how they change an image's shape and size. If you're unfamiliar with convolutions, check out the [Convolution Neural Networks chapter](https://github.com/fastai/fastbook/blob/master/13_convolutions.ipynb) from the fastai book!

</Tip>

[ConvNeXT](model_doc/convnext) is a CNN architecture that adopts new and modern network designs to improve performance. However, convolutions are still at the core of the model. From a high-level perspective, a [convolution](glossary#convolution) is an operation where a smaller matrix (*kernel*) is multiplied by a small window of the image pixels. It computes some features from it, such as a particular texture or curvature of a line. Then it slides over to the next window of pixels; the distance the convolution travels is known as the *stride*. 

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/convolution.gif"/>
</div>

<small>A basic convolution without padding or stride, taken from <a href="https://arxiv.org/abs/1603.07285">A guide to convolution arithmetic for deep learning.</a></small>

You can feed this output to another convolutional layer, and with each successive layer, the network learns more complex and abstract things like hotdogs or rockets. Between convolutional layers, it is common to add a pooling layer to reduce dimensionality and make the model more robust to variations of a feature's position.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/convnext_architecture.png"/>
</div>

ConvNeXT modernizes a CNN in five ways:

1. Change the number of blocks in each stage and "patchify" an image with a larger stride and corresponding kernel size. The non-overlapping sliding window makes this patchifying strategy similar to how ViT splits an image into patches.

2. A *bottleneck* layer shrinks the number of channels and then restores it because it is faster to do a 1x1 convolution, and you can increase the depth. An inverted bottleneck does the opposite by expanding the number of channels and shrinking them, which is more memory efficient.

3. Replace the typical 3x3 convolutional layer in the bottleneck layer with *depthwise convolution*, which applies a convolution to each input channel separately and then stacks them back together at the end. This widens the network width for improved performance.

4. ViT has a global receptive field which means it can see more of an image at once thanks to its attention mechanism. ConvNeXT attempts to replicate this effect by increasing the kernel size to 7x7.

5. ConvNeXT also makes several layer design changes that imitate Transformer models. There are fewer activation and normalization layers,  the activation function is switched to GELU instead of ReLU, and it uses LayerNorm instead of BatchNorm.

The output from the convolution blocks is passed to a classification head which converts the outputs into logits and calculates the cross-entropy loss to find the most likely label.

### Object detection

[DETR](model_doc/detr), *DEtection TRansformer*, is an end-to-end object detection model that combines a CNN with a Transformer encoder-decoder.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/detr_architecture.png"/>
</div>

1. A pretrained CNN *backbone* takes an image, represented by its pixel values, and creates a low-resolution feature map of it. A 1x1 convolution is applied to the feature map to reduce dimensionality and it creates a new feature map with a high-level image representation. Since the Transformer is a sequential model, the feature map is flattened into a sequence of feature vectors that are combined with positional embeddings.

2. The feature vectors are passed to the encoder, which learns the image representations using its attention layers. Next, the encoder hidden states are combined with *object queries* in the decoder. Object queries are learned embeddings that focus on the different regions of an image, and they're updated as they progress through each attention layer. The decoder hidden states are passed to a feedforward network that predicts the bounding box coordinates and class label for each object query, or `no object` if there isn't one.

    DETR decodes each object query in parallel to output *N* final predictions, where *N* is the number of queries. Unlike a typical autoregressive model that predicts one element at a time, object detection is a set prediction task (`bounding box`, `class label`) that makes *N* predictions in a single pass.

3. DETR uses a *bipartite matching loss* during training to compare a fixed number of predictions with a fixed set of ground truth labels. If there are fewer ground truth labels in the set of *N* labels, then they're padded with a `no object` class. This loss function encourages DETR to find a one-to-one assignment between the predictions and ground truth labels. If either the bounding boxes or class labels aren't correct, a loss is incurred. Likewise, if DETR predicts an object that doesn't exist, it is penalized. This encourages DETR to find other objects in an image instead of focusing on one really prominent object.

An object detection head is added on top of DETR to find the class label and the coordinates of the bounding box. There are two components to the object detection head: a linear layer to transform the decoder hidden states into logits over the class labels, and a MLP to predict the bounding box.

Ready to try your hand at object detection? Check out our complete [object detection guide](tasks/object_detection) to learn how to finetune DETR and use it for inference!

### Image segmentation

[Mask2Former](model_doc/mask2former) is a universal architecture for solving all types of image segmentation tasks. Traditional segmentation models are typically tailored towards a particular subtask of image segmentation, like instance, semantic or panoptic segmentation. Mask2Former frames each of those tasks as a *mask classification* problem. Mask classification groups pixels into *N* segments, and predicts *N* masks and their corresponding class label for a given image. We'll explain how Mask2Former works in this section, and then you can try finetuning SegFormer at the end.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/mask2former_architecture.png"/>
</div>

There are three main components to Mask2Former:

1. A [Swin](model_doc/swin) backbone accepts an image and creates a low-resolution image feature map from 3 consecutive 3x3 convolutions.

2. The feature map is passed to a *pixel decoder* which gradually upsamples the low-resolution features into high-resolution per-pixel embeddings. The pixel decoder actually generates multi-scale features (contains both low- and high-resolution features) with resolutions 1/32, 1/16, and 1/8th of the original image.

3. Each of these feature maps of differing scales is fed successively to one Transformer decoder layer at a time in order to capture small objects from the high-resolution features. The key to Mask2Former is the *masked attention* mechanism in the decoder. Unlike cross-attention which can attend to the entire image, masked attention only focuses on a certain area of the image. This is faster and leads to better performance because the local features of an image are enough for the model to learn from.

4. Like [DETR](tasks_explained#object-detection), Mask2Former also uses learned object queries and combines them with the image features from the pixel decoder to make a set prediction (`class label`, `mask prediction`). The decoder hidden states are passed into a linear layer and transformed into logits over the class labels. The cross-entropy loss is calculated between the logits and class label to find the most likely one.

    The mask predictions are generated by combining the pixel-embeddings with the final decoder hidden states. The sigmoid cross-entropy and dice loss is calculated between the logits and the ground truth mask to find the most likely mask.

Ready to try your hand at image segmentation? Check out our complete [image segmentation guide](tasks/semantic_segmentation) to learn how to finetune SegFormer and use it for inference!

### Depth estimation

[GLPN](model_doc/glpn), *Global-Local Path Network*, is a Transformer for depth estimation that combines a [SegFormer](model_doc/segformer) encoder with a lightweight decoder.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/glpn_architecture.jpg"/>
</div>

1. Like ViT, an image is split into a sequence of patches, except these image patches are smaller. This is better for dense prediction tasks like segmentation or depth estimation. The image patches are transformed into patch embeddings (see the [image classification](#image-classification) section for more details about how patch embeddings are created), which are fed to the encoder.

2. The encoder accepts the patch embeddings, and passes them through several encoder blocks. Each block consists of attention and Mix-FFN layers. The purpose of the latter is to provide positional information. At the end of each encoder block is a *patch merging* layer for creating hierarchical representations. The features of each group of neighboring patches are concatenated, and a linear layer is applied to the concatenated features to reduce the number of patches to a resolution of 1/4. This becomes the input to the next encoder block, where this whole process is repeated until you have image features with resolutions of 1/8, 1/16, and 1/32.

3. A lightweight decoder takes the last feature map (1/32 scale) from the encoder and upsamples it to 1/16 scale. From here, the feature is passed into a *Selective Feature Fusion (SFF)* module, which selects and combines local and global features from an attention map for each feature and then upsamples it to 1/8th. This process is repeated until the decoded features are the same size as the original image. The output is passed through two convolution layers and then a sigmoid activation is applied to predict the depth of each pixel.

## Natural language processing

The Transformer was initially designed for machine translation, and since then, it has practically become the default architecture for solving all NLP tasks. Some tasks lend themselves to the Transformer's encoder structure, while others are better suited for the decoder. Still, other tasks make use of both the Transformer's encoder-decoder structure.

### Text classification

[BERT](model_doc/bert) is an encoder-only model and is the first model to effectively implement deep bidirectionality to learn richer representations of the text by attending to words on both sides.

1. BERT uses [WordPiece](tokenizer_summary#wordpiece) tokenization to generate a token embedding of the text. To tell the difference between a single sentence and a pair of sentences, a special `[SEP]` token is added to differentiate them. A special `[CLS]` token is added to the beginning of every sequence of text. The final output with the `[CLS]` token is used as the input to the classification head for classification tasks. BERT also adds a segment embedding to denote whether a token belongs to the first or second sentence in a pair of sentences.

2. BERT is pretrained with two objectives: masked language modeling and next-sentence prediction. In masked language modeling, some percentage of the input tokens are randomly masked, and the model needs to predict these. This solves the issue of bidirectionality, where the model could cheat and see all the words and "predict" the next word. The final hidden states of the predicted mask tokens are passed to a feedforward network with a softmax over the vocabulary to predict the masked word.

    The second pretraining object is next-sentence prediction. The model must predict whether sentence B follows sentence A. Half of the time sentence B is the next sentence, and the other half of the time, sentence B is a random sentence. The prediction, whether it is the next sentence or not, is passed to a feedforward network with a softmax over the two classes (`IsNext` and `NotNext`).

3. The input embeddings are passed through multiple encoder layers to output some final hidden states.

To use the pretrained model for text classification, add a sequence classification head on top of the base BERT model. The sequence classification head is a linear layer that accepts the final hidden states and performs a linear transformation to convert them into logits. The cross-entropy loss is calculated between the logits and target to find the most likely label.

Ready to try your hand at text classification? Check out our complete [text classification guide](tasks/sequence_classification) to learn how to finetune DistilBERT and use it for inference!

### Token classification

To use BERT for token classification tasks like named entity recognition (NER), add a token classification head on top of the base BERT model. The token classification head is a linear layer that accepts the final hidden states and performs a linear transformation to convert them into logits. The cross-entropy loss is calculated between the logits and each token to find the most likely label.

Ready to try your hand at token classification? Check out our complete [token classification guide](tasks/token_classification) to learn how to finetune DistilBERT and use it for inference!

### Question answering

To use BERT for question answering, add a span classification head on top of the base BERT model. This linear layer accepts the final hidden states and performs a linear transformation to compute the `span` start and end logits corresponding to the answer. The cross-entropy loss is calculated between the logits and the label position to find the most likely span of text corresponding to the answer.

Ready to try your hand at question answering? Check out our complete [question answering guide](tasks/question_answering) to learn how to finetune DistilBERT and use it for inference!

<Tip>

💡 Notice how easy it is to use BERT for different tasks once it's been pretrained. You only need to add a specific head to the pretrained model to manipulate the hidden states into your desired output!

</Tip>

### Text generation

[GPT-2](model_doc/gpt2) is a decoder-only model pretrained on a large amount of text. It can generate convincing (though not always true!) text given a prompt and complete other NLP tasks like question answering despite not being explicitly trained to.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/gpt2_architecture.png"/>
</div>

1. GPT-2 uses [byte pair encoding (BPE)](tokenizer_summary#bytepair-encoding-bpe) to tokenize words and generate a token embedding. Positional encodings are added to the token embeddings to indicate the position of each token in the sequence. The input embeddings are passed through multiple decoder blocks to output some final hidden state. Within each decoder block, GPT-2 uses a *masked self-attention* layer which means GPT-2 can't attend to future tokens. It is only allowed to attend to tokens on the left. This is different from BERT's [`mask`] token because, in masked self-attention, an attention mask is used to set the score to `0` for future tokens.

2. The output from the decoder is passed to a language modeling head, which performs a linear transformation to convert the hidden states into logits. The label is the next token in the sequence, which are created by shifting the logits to the right by one. The cross-entropy loss is calculated between the shifted logits and the labels to output the next most likely token.

GPT-2's pretraining objective is based entirely on [causal language modeling](glossary#causal-language-modeling), predicting the next word in a sequence. This makes GPT-2 especially good at tasks that involve generating text.

Ready to try your hand at text generation? Check out our complete [causal language modeling guide](tasks/language_modeling#causal-language-modeling) to learn how to finetune DistilGPT-2 and use it for inference!

<Tip>

For more information about text generation, check out the [text generation strategies](generation_strategies) guide!

</Tip>

### Summarization

Encoder-decoder models like [BART](model_doc/bart) and [T5](model_doc/t5) are designed for the sequence-to-sequence pattern of a summarization task. We'll explain how BART works in this section, and then you can try finetuning T5 at the end.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bart_architecture.png"/>
</div>

1. BART's encoder architecture is very similar to BERT and accepts a token and positional embedding of the text. BART is pretrained by corrupting the input and then reconstructing it with the decoder. Unlike other encoders with specific corruption strategies, BART can apply any type of corruption. The *text infilling* corruption strategy works the best though. In text infilling, a number of text spans are replaced with a **single** [`mask`] token. This is important because the model has to predict the masked tokens, and it teaches the model to predict the number of missing tokens. The input embeddings and masked spans are passed through the encoder to output some final hidden states, but unlike BERT, BART doesn't add a final feedforward network at the end to predict a word.

2. The encoder's output is passed to the decoder, which must predict the masked tokens and any uncorrupted tokens from the encoder's output. This gives additional context to help the decoder restore the original text. The output from the decoder is passed to a language modeling head, which performs a linear transformation to convert the hidden states into logits. The cross-entropy loss is calculated between the logits and the label, which is just the token shifted to the right.

Ready to try your hand at summarization? Check out our complete [summarization guide](tasks/summarization) to learn how to finetune T5 and use it for inference!

<Tip>

For more information about text generation, check out the [text generation strategies](generation_strategies) guide!

</Tip>

### Translation

Translation is another example of a sequence-to-sequence task, which means you can use an encoder-decoder model like [BART](model_doc/bart) or [T5](model_doc/t5) to do it. We'll explain how BART works in this section, and then you can try finetuning T5 at the end.

BART adapts to translation by adding a separate randomly initialized encoder to map a source language to an input that can be decoded into the target language. This new encoder's embeddings are passed to the pretrained encoder instead of the original word embeddings. The source encoder is trained by updating the source encoder, positional embeddings, and input embeddings with the cross-entropy loss from the model output. The model parameters are frozen in this first step, and all the model parameters are trained together in the second step.

BART has since been followed up by a multilingual version, mBART, intended for translation and pretrained on many different languages.

Ready to try your hand at translation? Check out our complete [translation guide](tasks/translation) to learn how to finetune T5 and use it for inference!

<Tip>

For more information about text generation, check out the [text generation strategies](generation_strategies) guide!

</Tip>
