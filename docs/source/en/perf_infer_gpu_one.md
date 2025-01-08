<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# GPU inference

GPUs are the standard choice of hardware for machine learning, unlike CPUs, because they are optimized for memory bandwidth and parallelism. To keep up with the larger sizes of modern models or to run these large models on existing and older hardware, there are several optimizations you can use to speed up GPU inference. In this guide, you'll learn how to use FlashAttention-2 (a more memory-efficient attention mechanism), BetterTransformer (a PyTorch native fastpath execution), and bitsandbytes to quantize your model to a lower precision. Finally, learn how to use 🤗 Optimum to accelerate inference with ONNX Runtime on Nvidia and AMD GPUs.

<Tip>

The majority of the optimizations described here also apply to multi-GPU setups!

</Tip>

## FlashAttention-2

<Tip>

FlashAttention-2 is experimental and may change considerably in future versions.

</Tip>

[FlashAttention-2](https://huggingface.co/papers/2205.14135) is a faster and more efficient implementation of the standard attention mechanism that can significantly speedup inference by:

1. additionally parallelizing the attention computation over sequence length
2. partitioning the work between GPU threads to reduce communication and shared memory reads/writes between them

FlashAttention-2 is currently supported for the following architectures:
* [Aria](https://huggingface.co/docs/transformers/model_doc/aria#transformers.AriaForConditionalGeneration)
* [Bark](https://huggingface.co/docs/transformers/model_doc/bark#transformers.BarkModel)
* [Bamba](https://huggingface.co/docs/transformers/model_doc/bamba#transformers.BambaModel)
* [Bart](https://huggingface.co/docs/transformers/model_doc/bart#transformers.BartModel)
* [Chameleon](https://huggingface.co/docs/transformers/model_doc/chameleon#transformers.Chameleon)
* [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPModel)
* [Cohere](https://huggingface.co/docs/transformers/model_doc/cohere#transformers.CohereModel)
* [Cohere2](https://huggingface.co/docs/transformers/model_doc/cohere2#transformers.Cohere2Model)
* [GLM](https://huggingface.co/docs/transformers/model_doc/glm#transformers.GLMModel)
* [Dbrx](https://huggingface.co/docs/transformers/model_doc/dbrx#transformers.DbrxModel)
* [DiffLlama](https://huggingface.co/docs/transformers/model_doc/diffllama#transformers.DiffLlamaModel)
* [DistilBert](https://huggingface.co/docs/transformers/model_doc/distilbert#transformers.DistilBertModel)
* [Gemma](https://huggingface.co/docs/transformers/model_doc/gemma#transformers.GemmaModel)
* [Gemma2](https://huggingface.co/docs/transformers/model_doc/gemma2#transformers.Gemma2Model)
* [GPT2](https://huggingface.co/docs/transformers/model_doc/gpt2)
* [GPTBigCode](https://huggingface.co/docs/transformers/model_doc/gpt_bigcode#transformers.GPTBigCodeModel)
* [GPTNeo](https://huggingface.co/docs/transformers/model_doc/gpt_neo#transformers.GPTNeoModel)
* [GPTNeoX](https://huggingface.co/docs/transformers/model_doc/gpt_neox#transformers.GPTNeoXModel)
* [GPT-J](https://huggingface.co/docs/transformers/model_doc/gptj#transformers.GPTJModel)
* [Granite](https://huggingface.co/docs/transformers/model_doc/granite#transformers.GraniteModel)
* [GraniteMoe](https://huggingface.co/docs/transformers/model_doc/granitemoe#transformers.GraniteMoeModel)
* [Idefics2](https://huggingface.co/docs/transformers/model_doc/idefics2#transformers.Idefics2Model)
* [Idefics3](https://huggingface.co/docs/transformers/model_doc/idefics3#transformers.Idefics3Model)
* [Falcon](https://huggingface.co/docs/transformers/model_doc/falcon#transformers.FalconModel)
* [JetMoe](https://huggingface.co/docs/transformers/model_doc/jetmoe#transformers.JetMoeModel)
* [Jamba](https://huggingface.co/docs/transformers/model_doc/jamba#transformers.JambaModel)
* [Llama](https://huggingface.co/docs/transformers/model_doc/llama#transformers.LlamaModel)
* [Llava](https://huggingface.co/docs/transformers/model_doc/llava)
* [Llava-NeXT](https://huggingface.co/docs/transformers/model_doc/llava_next)
* [Llava-NeXT-Video](https://huggingface.co/docs/transformers/model_doc/llava_next_video)
* [LLaVA-Onevision](https://huggingface.co/docs/transformers/model_doc/llava_onevision)
* [Mimi](https://huggingface.co/docs/transformers/model_doc/mimi)
* [VipLlava](https://huggingface.co/docs/transformers/model_doc/vipllava)
* [VideoLlava](https://huggingface.co/docs/transformers/model_doc/video_llava)
* [M2M100](https://huggingface.co/docs/transformers/model_doc/m2m_100)
* [MBart](https://huggingface.co/docs/transformers/model_doc/mbart#transformers.MBartModel)
* [Mistral](https://huggingface.co/docs/transformers/model_doc/mistral#transformers.MistralModel)
* [Mixtral](https://huggingface.co/docs/transformers/model_doc/mixtral#transformers.MixtralModel)
* [ModernBert](https://huggingface.co/docs/transformers/model_doc/modernbert#transformers.ModernBert)
* [Moshi](https://huggingface.co/docs/transformers/model_doc/moshi#transformers.MoshiModel)
* [Musicgen](https://huggingface.co/docs/transformers/model_doc/musicgen#transformers.MusicgenModel)
* [MusicGen Melody](https://huggingface.co/docs/transformers/model_doc/musicgen_melody#transformers.MusicgenMelodyModel)
* [Nemotron](https://huggingface.co/docs/transformers/model_doc/nemotron)
* [NLLB](https://huggingface.co/docs/transformers/model_doc/nllb)
* [OLMo](https://huggingface.co/docs/transformers/model_doc/olmo#transformers.OlmoModel)
* [OLMo2](https://huggingface.co/docs/transformers/model_doc/olmo2#transformers.Olmo2Model)
* [OLMoE](https://huggingface.co/docs/transformers/model_doc/olmoe#transformers.OlmoeModel)
* [OPT](https://huggingface.co/docs/transformers/model_doc/opt#transformers.OPTModel)
* [PaliGemma](https://huggingface.co/docs/transformers/model_doc/paligemma#transformers.PaliGemmaForConditionalGeneration)
* [Phi](https://huggingface.co/docs/transformers/model_doc/phi#transformers.PhiModel)
* [Phi3](https://huggingface.co/docs/transformers/model_doc/phi3#transformers.Phi3Model)
* [PhiMoE](https://huggingface.co/docs/transformers/model_doc/phimoe#transformers.PhimoeModel)
* [StableLm](https://huggingface.co/docs/transformers/model_doc/stablelm#transformers.StableLmModel)
* [Starcoder2](https://huggingface.co/docs/transformers/model_doc/starcoder2#transformers.Starcoder2Model)
* [Qwen2](https://huggingface.co/docs/transformers/model_doc/qwen2#transformers.Qwen2Model)
* [Qwen2Audio](https://huggingface.co/docs/transformers/model_doc/qwen2_audio#transformers.Qwen2AudioEncoder)
* [Qwen2MoE](https://huggingface.co/docs/transformers/model_doc/qwen2_moe#transformers.Qwen2MoeModel)
* [Qwen2VL](https://huggingface.co/docs/transformers/model_doc/qwen2_vl#transformers.Qwen2VLModel)
* [RAG](https://huggingface.co/docs/transformers/model_doc/rag#transformers.RagModel)
* [SpeechEncoderDecoder](https://huggingface.co/docs/transformers/model_doc/speech_encoder_decoder#transformers.SpeechEncoderDecoderModel)
* [VisionEncoderDecoder](https://huggingface.co/docs/transformers/model_doc/vision_encoder_decoder#transformers.VisionEncoderDecoderModel)
* [VisionTextDualEncoder](https://huggingface.co/docs/transformers/model_doc/vision_text_dual_encoder#transformers.VisionTextDualEncoderModel)
* [Whisper](https://huggingface.co/docs/transformers/model_doc/whisper#transformers.WhisperModel)
* [Wav2Vec2](https://huggingface.co/docs/transformers/model_doc/wav2vec2#transformers.Wav2Vec2Model)
* [Hubert](https://huggingface.co/docs/transformers/model_doc/hubert#transformers.HubertModel)
* [data2vec_audio](https://huggingface.co/docs/transformers/main/en/model_doc/data2vec#transformers.Data2VecAudioModel)
* [Sew](https://huggingface.co/docs/transformers/main/en/model_doc/sew#transformers.SEWModel)
* [SigLIP](https://huggingface.co/docs/transformers/model_doc/siglip)
* [UniSpeech](https://huggingface.co/docs/transformers/v4.39.3/en/model_doc/unispeech#transformers.UniSpeechModel)
* [unispeech_sat](https://huggingface.co/docs/transformers/v4.39.3/en/model_doc/unispeech-sat#transformers.UniSpeechSatModel)

You can request to add FlashAttention-2 support for another model by opening a GitHub Issue or Pull Request.

Before you begin, make sure you have FlashAttention-2 installed.

<hfoptions id="install">
<hfoption id="NVIDIA">

```bash
pip install flash-attn --no-build-isolation
```

We strongly suggest referring to the detailed [installation instructions](https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features) to learn more about supported hardware and data types!

</hfoption>
<hfoption id="AMD">

FlashAttention-2 is also supported on AMD GPUs and current support is limited to **Instinct MI210**, **Instinct MI250** and **Instinct MI300**. We strongly suggest using this [Dockerfile](https://github.com/huggingface/optimum-amd/tree/main/docker/transformers-pytorch-amd-gpu-flash/Dockerfile) to use FlashAttention-2 on AMD GPUs.

</hfoption>
</hfoptions>

To enable FlashAttention-2, pass the argument `attn_implementation="flash_attention_2"` to [`~AutoModelForCausalLM.from_pretrained`]:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM

model_id = "tiiuae/falcon-7b"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
```

<Tip>

FlashAttention-2 can only be used when the model's dtype is `fp16` or `bf16`. Make sure to cast your model to the appropriate dtype and load them on a supported device before using FlashAttention-2.

<br>

You can also set `use_flash_attention_2=True` to enable FlashAttention-2 but it is deprecated in favor of `attn_implementation="flash_attention_2"`.

</Tip>

FlashAttention-2 can be combined with other optimization techniques like quantization to further speedup inference. For example, you can combine FlashAttention-2 with 8-bit or 4-bit quantization:

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM

model_id = "tiiuae/falcon-7b"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# load in 8bit
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_8bit=True,
    attn_implementation="flash_attention_2",
)

# load in 4bit
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_4bit=True,
    attn_implementation="flash_attention_2",
)
```

### Expected speedups

You can benefit from considerable speedups for inference, especially for inputs with long sequences. However, since FlashAttention-2 does not support computing attention scores with padding tokens, you must manually pad/unpad the attention scores for batched inference when the sequence contains padding tokens. This leads to a significant slowdown for batched generations with padding tokens.

To overcome this, you should use FlashAttention-2 without padding tokens in the sequence during training (by packing a dataset or [concatenating sequences](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py#L516) until reaching the maximum sequence length).

For a single forward pass on [tiiuae/falcon-7b](https://hf.co/tiiuae/falcon-7b) with a sequence length of 4096 and various batch sizes without padding tokens, the expected speedup is:

<div style="text-align: center">
<img src="https://huggingface.co/datasets/ybelkada/documentation-images/resolve/main/falcon-7b-inference-large-seqlen.png">
</div>

For a single forward pass on [meta-llama/Llama-7b-hf](https://hf.co/meta-llama/Llama-7b-hf) with a sequence length of 4096 and various batch sizes without padding tokens, the expected speedup is:

<div style="text-align: center">
<img src="https://huggingface.co/datasets/ybelkada/documentation-images/resolve/main/llama-7b-inference-large-seqlen.png">
</div>

For sequences with padding tokens (generating with padding tokens), you need to unpad/pad the input sequences to correctly compute the attention scores. With a relatively small sequence length, a single forward pass creates overhead leading to a small speedup (in the example below, 30% of the input is filled with padding tokens):

<div style="text-align: center">
<img src="https://huggingface.co/datasets/ybelkada/documentation-images/resolve/main/llama-2-small-seqlen-padding.png">
</div>

But for larger sequence lengths, you can expect even more speedup benefits:

<Tip>

FlashAttention is more memory efficient, meaning you can train on much larger sequence lengths without running into out-of-memory issues. You can potentially reduce memory usage up to 20x for larger sequence lengths. Take a look at the [flash-attention](https://github.com/Dao-AILab/flash-attention) repository for more details.

</Tip>

<div style="text-align: center">
<img src="https://huggingface.co/datasets/ybelkada/documentation-images/resolve/main/llama-2-large-seqlen-padding.png">
</div>

## PyTorch scaled dot product attention

PyTorch's [`torch.nn.functional.scaled_dot_product_attention`](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html) (SDPA) can also call FlashAttention and memory-efficient attention kernels under the hood. SDPA support is currently being added natively in Transformers and is used by default for `torch>=2.1.1` when an implementation is available. You may also set `attn_implementation="sdpa"` in `from_pretrained()` to explicitly request SDPA to be used.

For now, Transformers supports SDPA inference and training for the following architectures:
* [Albert](https://huggingface.co/docs/transformers/model_doc/albert#transformers.AlbertModel)
* [Aria](https://huggingface.co/docs/transformers/model_doc/aria#transformers.AriaForConditionalGeneration)
* [Audio Spectrogram Transformer](https://huggingface.co/docs/transformers/model_doc/audio-spectrogram-transformer#transformers.ASTModel)
* [Bamba](https://huggingface.co/docs/transformers/model_doc/bamba#transformers.BambaModel)
* [Bart](https://huggingface.co/docs/transformers/model_doc/bart#transformers.BartModel)
* [Beit](https://huggingface.co/docs/transformers/model_doc/beit#transformers.BeitModel)
* [Bert](https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertModel)
* [BioGpt](https://huggingface.co/docs/transformers/model_doc/biogpt#transformers.BioGptModel)
* [CamemBERT](https://huggingface.co/docs/transformers/model_doc/camembert#transformers.CamembertModel)
* [Chameleon](https://huggingface.co/docs/transformers/model_doc/chameleon#transformers.Chameleon)
* [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPModel)
* [GLM](https://huggingface.co/docs/transformers/model_doc/glm#transformers.GLMModel)
* [Cohere](https://huggingface.co/docs/transformers/model_doc/cohere#transformers.CohereModel)
* [Cohere2](https://huggingface.co/docs/transformers/model_doc/cohere2#transformers.Cohere2Model)
* [data2vec_audio](https://huggingface.co/docs/transformers/main/en/model_doc/data2vec#transformers.Data2VecAudioModel)
* [data2vec_vision](https://huggingface.co/docs/transformers/main/en/model_doc/data2vec#transformers.Data2VecVisionModel)
* [Dbrx](https://huggingface.co/docs/transformers/model_doc/dbrx#transformers.DbrxModel)
* [DeiT](https://huggingface.co/docs/transformers/model_doc/deit#transformers.DeiTModel)
* [DiffLlama](https://huggingface.co/docs/transformers/model_doc/diffllama#transformers.DiffLlamaModel)
* [Dinov2](https://huggingface.co/docs/transformers/en/model_doc/dinov2)
* [Dinov2_with_registers](https://huggingface.co/docs/transformers/en/model_doc/dinov2)
* [DistilBert](https://huggingface.co/docs/transformers/model_doc/distilbert#transformers.DistilBertModel)
* [Dpr](https://huggingface.co/docs/transformers/model_doc/dpr#transformers.DprReader)
* [EncoderDecoder](https://huggingface.co/docs/transformers/model_doc/encoder_decoder#transformers.EncoderDecoderModel)
* [Falcon](https://huggingface.co/docs/transformers/model_doc/falcon#transformers.FalconModel)
* [Gemma](https://huggingface.co/docs/transformers/model_doc/gemma#transformers.GemmaModel)
* [Gemma2](https://huggingface.co/docs/transformers/model_doc/gemma2#transformers.Gemma2Model)
* [Granite](https://huggingface.co/docs/transformers/model_doc/granite#transformers.GraniteModel)
* [GPT2](https://huggingface.co/docs/transformers/model_doc/gpt2)
* [GPTBigCode](https://huggingface.co/docs/transformers/model_doc/gpt_bigcode#transformers.GPTBigCodeModel)
* [GPTNeoX](https://huggingface.co/docs/transformers/model_doc/gpt_neox#transformers.GPTNeoXModel)
* [Hubert](https://huggingface.co/docs/transformers/model_doc/hubert#transformers.HubertModel)
* [Idefics](https://huggingface.co/docs/transformers/model_doc/idefics#transformers.IdeficsModel)
* [Idefics2](https://huggingface.co/docs/transformers/model_doc/idefics2#transformers.Idefics2Model)
* [Idefics3](https://huggingface.co/docs/transformers/model_doc/idefics3#transformers.Idefics3Model)
* [I-JEPA](https://huggingface.co/docs/transformers/model_doc/ijepa#transformers.IJepaModel)
* [GraniteMoe](https://huggingface.co/docs/transformers/model_doc/granitemoe#transformers.GraniteMoeModel)
* [JetMoe](https://huggingface.co/docs/transformers/model_doc/jetmoe#transformers.JetMoeModel)
* [Jamba](https://huggingface.co/docs/transformers/model_doc/jamba#transformers.JambaModel)
* [Llama](https://huggingface.co/docs/transformers/model_doc/llama#transformers.LlamaModel)
* [Llava](https://huggingface.co/docs/transformers/model_doc/llava)
* [Llava-NeXT](https://huggingface.co/docs/transformers/model_doc/llava_next)
* [Llava-NeXT-Video](https://huggingface.co/docs/transformers/model_doc/llava_next_video)
* [LLaVA-Onevision](https://huggingface.co/docs/transformers/model_doc/llava_onevision)
* [M2M100](https://huggingface.co/docs/transformers/model_doc/m2m_100#transformers.M2M100Model)
* [Mimi](https://huggingface.co/docs/transformers/model_doc/mimi)
* [Mistral](https://huggingface.co/docs/transformers/model_doc/mistral#transformers.MistralModel)
* [Mllama](https://huggingface.co/docs/transformers/model_doc/mllama#transformers.MllamaForConditionalGeneration)
* [Mixtral](https://huggingface.co/docs/transformers/model_doc/mixtral#transformers.MixtralModel)
* [ModernBert](https://huggingface.co/docs/transformers/model_doc/modernbert#transformers.ModernBert)
* [Moshi](https://huggingface.co/docs/transformers/model_doc/moshi#transformers.MoshiModel)
* [Musicgen](https://huggingface.co/docs/transformers/model_doc/musicgen#transformers.MusicgenModel)
* [MusicGen Melody](https://huggingface.co/docs/transformers/model_doc/musicgen_melody#transformers.MusicgenMelodyModel)
* [NLLB](https://huggingface.co/docs/transformers/model_doc/nllb)
* [OLMo](https://huggingface.co/docs/transformers/model_doc/olmo#transformers.OlmoModel)
* [OLMo2](https://huggingface.co/docs/transformers/model_doc/olmo2#transformers.Olmo2Model)
* [OLMoE](https://huggingface.co/docs/transformers/model_doc/olmoe#transformers.OlmoeModel)
* [OPT](https://huggingface.co/docs/transformers/en/model_doc/opt)
* [PaliGemma](https://huggingface.co/docs/transformers/model_doc/paligemma#transformers.PaliGemmaForConditionalGeneration)
* [Phi](https://huggingface.co/docs/transformers/model_doc/phi#transformers.PhiModel)
* [Phi3](https://huggingface.co/docs/transformers/model_doc/phi3#transformers.Phi3Model)
* [PhiMoE](https://huggingface.co/docs/transformers/model_doc/phimoe#transformers.PhimoeModel)
* [Idefics](https://huggingface.co/docs/transformers/model_doc/idefics#transformers.IdeficsModel)
* [Whisper](https://huggingface.co/docs/transformers/model_doc/whisper#transformers.WhisperModel)
* [mBart](https://huggingface.co/docs/transformers/model_doc/mbart#transformers.MBartModel)
* [Mistral](https://huggingface.co/docs/transformers/model_doc/mistral#transformers.MistralModel)
* [Mixtral](https://huggingface.co/docs/transformers/model_doc/mixtral#transformers.MixtralModel)
* [StableLm](https://huggingface.co/docs/transformers/model_doc/stablelm#transformers.StableLmModel)
* [Starcoder2](https://huggingface.co/docs/transformers/model_doc/starcoder2#transformers.Starcoder2Model)
* [Qwen2](https://huggingface.co/docs/transformers/model_doc/qwen2#transformers.Qwen2Model)
* [Qwen2Audio](https://huggingface.co/docs/transformers/model_doc/qwen2_audio#transformers.Qwen2AudioEncoder)
* [Qwen2MoE](https://huggingface.co/docs/transformers/model_doc/qwen2_moe#transformers.Qwen2MoeModel)
* [RoBERTa](https://huggingface.co/docs/transformers/model_doc/roberta#transformers.RobertaModel)
* [Sew](https://huggingface.co/docs/transformers/main/en/model_doc/sew#transformers.SEWModel)
* [SigLIP](https://huggingface.co/docs/transformers/model_doc/siglip)
* [StableLm](https://huggingface.co/docs/transformers/model_doc/stablelm#transformers.StableLmModel)
* [Starcoder2](https://huggingface.co/docs/transformers/model_doc/starcoder2#transformers.Starcoder2Model)
* [UniSpeech](https://huggingface.co/docs/transformers/v4.39.3/en/model_doc/unispeech#transformers.UniSpeechModel)
* [unispeech_sat](https://huggingface.co/docs/transformers/v4.39.3/en/model_doc/unispeech-sat#transformers.UniSpeechSatModel)
* [RoBERTa](https://huggingface.co/docs/transformers/model_doc/roberta#transformers.RobertaModel)
* [Qwen2VL](https://huggingface.co/docs/transformers/model_doc/qwen2_vl#transformers.Qwen2VLModel)
* [Musicgen](https://huggingface.co/docs/transformers/model_doc/musicgen#transformers.MusicgenModel)
* [MusicGen Melody](https://huggingface.co/docs/transformers/model_doc/musicgen_melody#transformers.MusicgenMelodyModel)
* [Nemotron](https://huggingface.co/docs/transformers/model_doc/nemotron)
* [SpeechEncoderDecoder](https://huggingface.co/docs/transformers/model_doc/speech_encoder_decoder#transformers.SpeechEncoderDecoderModel)
* [VideoLlava](https://huggingface.co/docs/transformers/model_doc/video_llava)
* [VipLlava](https://huggingface.co/docs/transformers/model_doc/vipllava)
* [VisionEncoderDecoder](https://huggingface.co/docs/transformers/model_doc/vision_encoder_decoder#transformers.VisionEncoderDecoderModel)
* [ViT](https://huggingface.co/docs/transformers/model_doc/vit#transformers.ViTModel)
* [ViTHybrid](https://huggingface.co/docs/transformers/model_doc/vit_hybrid#transformers.ViTHybridModel)
* [ViTMAE](https://huggingface.co/docs/transformers/model_doc/vit_mae#transformers.ViTMAEModel)
* [ViTMSN](https://huggingface.co/docs/transformers/model_doc/vit_msn#transformers.ViTMSNModel)
* [VisionTextDualEncoder](https://huggingface.co/docs/transformers/model_doc/vision_text_dual_encoder#transformers.VisionTextDualEncoderModel)
* [VideoMAE](https://huggingface.co/docs/transformers/model_doc/videomae#transformers.VideoMAEModell)
* [ViViT](https://huggingface.co/docs/transformers/model_doc/vivit#transformers.VivitModel)
* [wav2vec2](https://huggingface.co/docs/transformers/model_doc/wav2vec2#transformers.Wav2Vec2Model)
* [Whisper](https://huggingface.co/docs/transformers/model_doc/whisper#transformers.WhisperModel)
* [XLM-RoBERTa](https://huggingface.co/docs/transformers/model_doc/xlm-roberta#transformers.XLMRobertaModel)
* [XLM-RoBERTa-XL](https://huggingface.co/docs/transformers/model_doc/xlm-roberta-xl#transformers.XLMRobertaXLModel)
* [YOLOS](https://huggingface.co/docs/transformers/model_doc/yolos#transformers.YolosModel)

<Tip>

FlashAttention can only be used for models with the `fp16` or `bf16` torch type, so make sure to cast your model to the appropriate type first. The memory-efficient attention backend is able to handle `fp32` models.

</Tip>

<Tip>

SDPA does not support certain sets of attention parameters, such as `head_mask` and `output_attentions=True`.
In that case, you should see a warning message and we will fall back to the (slower) eager implementation.

</Tip>

By default, SDPA selects the most performant kernel available but you can check whether a backend is available in a given setting (hardware, problem size) with [`torch.nn.attention.sdpa_kernel`](https://pytorch.org/docs/stable/generated/torch.nn.attention.sdpa_kernel.html) as a context manager:

```diff
import torch
+ from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m", torch_dtype=torch.float16).to("cuda")

input_text = "Hello my dog is cute and"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

+ with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    outputs = model.generate(**inputs)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

If you see a bug with the traceback below, try using the nightly version of PyTorch which may have broader coverage for FlashAttention:

```bash
RuntimeError: No available kernel. Aborting execution.

# install PyTorch nightly
pip3 install -U --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118
```

## BetterTransformer

<Tip warning={true}>

Some BetterTransformer features are being upstreamed to Transformers with default support for native `torch.nn.scaled_dot_product_attention`. BetterTransformer still has a wider coverage than the Transformers SDPA integration, but you can expect more and more architectures to natively support SDPA in Transformers.

</Tip>

<Tip>

Check out our benchmarks with BetterTransformer and scaled dot product attention in the [Out of the box acceleration and memory savings of 🤗 decoder models with PyTorch 2.0](https://pytorch.org/blog/out-of-the-box-acceleration/) and learn more about the fastpath execution in the [BetterTransformer](https://medium.com/pytorch/bettertransformer-out-of-the-box-performance-for-huggingface-transformers-3fbe27d50ab2) blog post.

</Tip>

BetterTransformer accelerates inference with its fastpath (native PyTorch specialized implementation of Transformer functions) execution. The two optimizations in the fastpath execution are:

1. fusion, which combines multiple sequential operations into a single "kernel" to reduce the number of computation steps
2. skipping the inherent sparsity of padding tokens to avoid unnecessary computation with nested tensors

BetterTransformer also converts all attention operations to use the more memory-efficient [scaled dot product attention (SDPA)](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention), and it calls optimized kernels like [FlashAttention](https://huggingface.co/papers/2205.14135) under the hood.

Before you start, make sure you have 🤗 Optimum [installed](https://huggingface.co/docs/optimum/installation).

Then you can enable BetterTransformer with the [`PreTrainedModel.to_bettertransformer`] method:

```python
model = model.to_bettertransformer()
```

You can return the original Transformers model with the [`~PreTrainedModel.reverse_bettertransformer`] method. You should use this before saving your model to use the canonical Transformers modeling:

```py
model = model.reverse_bettertransformer()
model.save_pretrained("saved_model")
```

## bitsandbytes

bitsandbytes is a quantization library that includes support for 4-bit and 8-bit quantization. Quantization reduces your model size compared to its native full precision version, making it easier to fit large models onto GPUs with limited memory.

Make sure you have bitsandbytes and 🤗 Accelerate installed:

```bash
# these versions support 8-bit and 4-bit
pip install bitsandbytes>=0.39.0 accelerate>=0.20.0

# install Transformers
pip install transformers
```

### 4-bit

To load a model in 4-bit for inference, use the `load_in_4bit` parameter. The `device_map` parameter is optional, but we recommend setting it to `"auto"` to allow 🤗 Accelerate to automatically and efficiently allocate the model given the available resources in the environment.

```py
from transformers import AutoModelForCausalLM

model_name = "bigscience/bloom-2b5"
model_4bit = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto", load_in_4bit=True)
```

To load a model in 4-bit for inference with multiple GPUs, you can control how much GPU RAM you want to allocate to each GPU. For example, to distribute 600MB of memory to the first GPU and 1GB of memory to the second GPU:

```py
max_memory_mapping = {0: "600MB", 1: "1GB"}
model_name = "bigscience/bloom-3b"
model_4bit = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto", load_in_4bit=True, max_memory=max_memory_mapping
)
```

### 8-bit

<Tip>

If you're curious and interested in learning more about the concepts underlying 8-bit quantization, read the [Gentle Introduction to 8-bit Matrix Multiplication for transformers at scale using Hugging Face Transformers, Accelerate and bitsandbytes](https://huggingface.co/blog/hf-bitsandbytes-integration) blog post.

</Tip>

To load a model in 8-bit for inference, use the `load_in_8bit` parameter. The `device_map` parameter is optional, but we recommend setting it to `"auto"` to allow 🤗 Accelerate to automatically and efficiently allocate the model given the available resources in the environment:

```py
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

model_name = "bigscience/bloom-2b5"
model_8bit = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", quantization_config=BitsAndBytesConfig(load_in_8bit=True))
```

If you're loading a model in 8-bit for text generation, you should use the [`~transformers.GenerationMixin.generate`] method instead of the [`Pipeline`] function which is not optimized for 8-bit models and will be slower. Some sampling strategies, like nucleus sampling, are also not supported by the [`Pipeline`] for 8-bit models. You should also place all inputs on the same device as the model:

```py
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_name = "bigscience/bloom-2b5"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model_8bit = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", quantization_config=BitsAndBytesConfig(load_in_8bit=True))

prompt = "Hello, my llama is cute"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
generated_ids = model.generate(**inputs)
outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
```

To load a model in 8-bit for inference with multiple GPUs, you can control how much GPU RAM you want to allocate to each GPU. For example, to distribute 1GB of memory to the first GPU and 2GB of memory to the second GPU:

```py
max_memory_mapping = {0: "1GB", 1: "2GB"}
model_name = "bigscience/bloom-3b"
model_8bit = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto", load_in_8bit=True, max_memory=max_memory_mapping
)
```

<Tip>

Feel free to try running a 11 billion parameter [T5 model](https://colab.research.google.com/drive/1YORPWx4okIHXnjW7MSAidXN29mPVNT7F?usp=sharing) or the 3 billion parameter [BLOOM model](https://colab.research.google.com/drive/1qOjXfQIAULfKvZqwCen8-MoWKGdSatZ4?usp=sharing) for inference on Google Colab's free tier GPUs!

</Tip>

## 🤗 Optimum

<Tip>

Learn more details about using ORT with 🤗 Optimum in the [Accelerated inference on NVIDIA GPUs](https://huggingface.co/docs/optimum/onnxruntime/usage_guides/gpu#accelerated-inference-on-nvidia-gpus) and [Accelerated inference on AMD GPUs](https://huggingface.co/docs/optimum/onnxruntime/usage_guides/amdgpu#accelerated-inference-on-amd-gpus) guides. This section only provides a brief and simple example.

</Tip>

ONNX Runtime (ORT) is a model accelerator that supports accelerated inference on Nvidia GPUs, and AMD GPUs that use [ROCm](https://www.amd.com/en/products/software/rocm.html) stack. ORT uses optimization techniques like fusing common operations into a single node and constant folding to reduce the number of computations performed and speedup inference. ORT also places the most computationally intensive operations on the GPU and the rest on the CPU to intelligently distribute the workload between the two devices.

ORT is supported by 🤗 Optimum which can be used in 🤗 Transformers. You'll need to use an [`~optimum.onnxruntime.ORTModel`] for the task you're solving, and specify the `provider` parameter which can be set to either [`CUDAExecutionProvider`](https://huggingface.co/docs/optimum/onnxruntime/usage_guides/gpu#cudaexecutionprovider), [`ROCMExecutionProvider`](https://huggingface.co/docs/optimum/onnxruntime/usage_guides/amdgpu) or [`TensorrtExecutionProvider`](https://huggingface.co/docs/optimum/onnxruntime/usage_guides/gpu#tensorrtexecutionprovider). If you want to load a model that was not yet exported to ONNX, you can set `export=True` to convert your model on-the-fly to the ONNX format:

```py
from optimum.onnxruntime import ORTModelForSequenceClassification

ort_model = ORTModelForSequenceClassification.from_pretrained(
  "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
  export=True,
  provider="CUDAExecutionProvider",
)
```

Now you're free to use the model for inference:

```py
from optimum.pipelines import pipeline
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased-finetuned-sst-2-english")

pipeline = pipeline(task="text-classification", model=ort_model, tokenizer=tokenizer, device="cuda:0")
result = pipeline("Both the music and visual were astounding, not to mention the actors performance.")
```

## Combine optimizations

It is often possible to combine several of the optimization techniques described above to get the best inference performance possible for your model. For example, you can load a model in 4-bit, and then enable BetterTransformer with FlashAttention:

```py
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# load model in 4-bit
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m", torch_dtype="auto", quantization_config=quantization_config)

# enable BetterTransformer
model = model.to_bettertransformer()

input_text = "Hello my dog is cute and"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

# enable FlashAttention
with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    outputs = model.generate(**inputs)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
