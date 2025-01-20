# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Auto Model class."""

import warnings
from collections import OrderedDict

from ...utils import logging
from .auto_factory import (
    _BaseAutoBackboneClass,
    _BaseAutoModelClass,
    _LazyAutoMapping,
    auto_class_update,
)
from .configuration_auto import CONFIG_MAPPING_NAMES


logger = logging.get_logger(__name__)

MODEL_MAPPING_NAMES = OrderedDict(
    [
        # Base model mapping
        ("albert", "AlbertModel"),
        ("align", "AlignModel"),
        ("altclip", "AltCLIPModel"),
        ("aria", "AriaForConditionalGeneration"),
        ("aria_text", "AriaTextModel"),
        ("audio-spectrogram-transformer", "ASTModel"),
        ("autoformer", "AutoformerModel"),
        ("bamba", "BambaModel"),
        ("bark", "BarkModel"),
        ("bart", "BartModel"),
        ("beit", "BeitModel"),
        ("bert", "BertModel"),
        ("bert-generation", "BertGenerationEncoder"),
        ("big_bird", "BigBirdModel"),
        ("bigbird_pegasus", "BigBirdPegasusModel"),
        ("biogpt", "BioGptModel"),
        ("bit", "BitModel"),
        ("blenderbot", "BlenderbotModel"),
        ("blenderbot-small", "BlenderbotSmallModel"),
        ("blip", "BlipModel"),
        ("blip-2", "Blip2Model"),
        ("bloom", "BloomModel"),
        ("bridgetower", "BridgeTowerModel"),
        ("bros", "BrosModel"),
        ("camembert", "CamembertModel"),
        ("canine", "CanineModel"),
        ("chameleon", "ChameleonModel"),
        ("chinese_clip", "ChineseCLIPModel"),
        ("chinese_clip_vision_model", "ChineseCLIPVisionModel"),
        ("clap", "ClapModel"),
        ("clip", "CLIPModel"),
        ("clip_text_model", "CLIPTextModel"),
        ("clip_vision_model", "CLIPVisionModel"),
        ("clipseg", "CLIPSegModel"),
        ("clvp", "ClvpModelForConditionalGeneration"),
        ("code_llama", "LlamaModel"),
        ("codegen", "CodeGenModel"),
        ("cohere", "CohereModel"),
        ("cohere2", "Cohere2Model"),
        ("conditional_detr", "ConditionalDetrModel"),
        ("convbert", "ConvBertModel"),
        ("convnext", "ConvNextModel"),
        ("convnextv2", "ConvNextV2Model"),
        ("cpmant", "CpmAntModel"),
        ("ctrl", "CTRLModel"),
        ("cvt", "CvtModel"),
        ("dac", "DacModel"),
        ("data2vec-audio", "Data2VecAudioModel"),
        ("data2vec-text", "Data2VecTextModel"),
        ("data2vec-vision", "Data2VecVisionModel"),
        ("dbrx", "DbrxModel"),
        ("deberta", "DebertaModel"),
        ("deberta-v2", "DebertaV2Model"),
        ("decision_transformer", "DecisionTransformerModel"),
        ("deformable_detr", "DeformableDetrModel"),
        ("deit", "DeiTModel"),
        ("deta", "DetaModel"),
        ("detr", "DetrModel"),
        ("diffllama", "DiffLlamaModel"),
        ("dinat", "DinatModel"),
        ("dinov2", "Dinov2Model"),
        ("dinov2_with_registers", "Dinov2WithRegistersModel"),
        ("distilbert", "DistilBertModel"),
        ("donut-swin", "DonutSwinModel"),
        ("dpr", "DPRQuestionEncoder"),
        ("dpt", "DPTModel"),
        ("efficientformer", "EfficientFormerModel"),
        ("efficientnet", "EfficientNetModel"),
        ("electra", "ElectraModel"),
        ("encodec", "EncodecModel"),
        ("ernie", "ErnieModel"),
        ("ernie_m", "ErnieMModel"),
        ("esm", "EsmModel"),
        ("falcon", "FalconModel"),
        ("falcon_mamba", "FalconMambaModel"),
        ("fastspeech2_conformer", "FastSpeech2ConformerModel"),
        ("flaubert", "FlaubertModel"),
        ("flava", "FlavaModel"),
        ("fnet", "FNetModel"),
        ("focalnet", "FocalNetModel"),
        ("fsmt", "FSMTModel"),
        ("funnel", ("FunnelModel", "FunnelBaseModel")),
        ("gemma", "GemmaModel"),
        ("gemma2", "Gemma2Model"),
        ("git", "GitModel"),
        ("glm", "GlmModel"),
        ("glpn", "GLPNModel"),
        ("gpt-sw3", "GPT2Model"),
        ("gpt2", "GPT2Model"),
        ("gpt_bigcode", "GPTBigCodeModel"),
        ("gpt_neo", "GPTNeoModel"),
        ("gpt_neox", "GPTNeoXModel"),
        ("gpt_neox_japanese", "GPTNeoXJapaneseModel"),
        ("gptj", "GPTJModel"),
        ("gptsan-japanese", "GPTSanJapaneseForConditionalGeneration"),
        ("granite", "GraniteModel"),
        ("granitemoe", "GraniteMoeModel"),
        ("graphormer", "GraphormerModel"),
        ("grounding-dino", "GroundingDinoModel"),
        ("groupvit", "GroupViTModel"),
        ("helium", "HeliumModel"),
        ("hiera", "HieraModel"),
        ("hubert", "HubertModel"),
        ("ibert", "IBertModel"),
        ("idefics", "IdeficsModel"),
        ("idefics2", "Idefics2Model"),
        ("idefics3", "Idefics3Model"),
        ("idefics3_vision", "Idefics3VisionTransformer"),
        ("ijepa", "IJepaModel"),
        ("imagegpt", "ImageGPTModel"),
        ("informer", "InformerModel"),
        ("internlm3", "InternLM3Model"),
        ("jamba", "JambaModel"),
        ("jetmoe", "JetMoeModel"),
        ("jukebox", "JukeboxModel"),
        ("kosmos-2", "Kosmos2Model"),
        ("layoutlm", "LayoutLMModel"),
        ("layoutlmv2", "LayoutLMv2Model"),
        ("layoutlmv3", "LayoutLMv3Model"),
        ("led", "LEDModel"),
        ("levit", "LevitModel"),
        ("lilt", "LiltModel"),
        ("llama", "LlamaModel"),
        ("longformer", "LongformerModel"),
        ("longt5", "LongT5Model"),
        ("luke", "LukeModel"),
        ("lxmert", "LxmertModel"),
        ("m2m_100", "M2M100Model"),
        ("mamba", "MambaModel"),
        ("mamba2", "Mamba2Model"),
        ("marian", "MarianModel"),
        ("markuplm", "MarkupLMModel"),
        ("mask2former", "Mask2FormerModel"),
        ("maskformer", "MaskFormerModel"),
        ("maskformer-swin", "MaskFormerSwinModel"),
        ("mbart", "MBartModel"),
        ("mctct", "MCTCTModel"),
        ("mega", "MegaModel"),
        ("megatron-bert", "MegatronBertModel"),
        ("mgp-str", "MgpstrForSceneTextRecognition"),
        ("mimi", "MimiModel"),
        ("mistral", "MistralModel"),
        ("mixtral", "MixtralModel"),
        ("mobilebert", "MobileBertModel"),
        ("mobilenet_v1", "MobileNetV1Model"),
        ("mobilenet_v2", "MobileNetV2Model"),
        ("mobilevit", "MobileViTModel"),
        ("mobilevitv2", "MobileViTV2Model"),
        ("modernbert", "ModernBertModel"),
        ("moonshine", "MoonshineModel"),
        ("moshi", "MoshiModel"),
        ("mpnet", "MPNetModel"),
        ("mpt", "MptModel"),
        ("mra", "MraModel"),
        ("mt5", "MT5Model"),
        ("musicgen", "MusicgenModel"),
        ("musicgen_melody", "MusicgenMelodyModel"),
        ("mvp", "MvpModel"),
        ("nat", "NatModel"),
        ("nemotron", "NemotronModel"),
        ("nezha", "NezhaModel"),
        ("nllb-moe", "NllbMoeModel"),
        ("nystromformer", "NystromformerModel"),
        ("olmo", "OlmoModel"),
        ("olmo2", "Olmo2Model"),
        ("olmoe", "OlmoeModel"),
        ("omdet-turbo", "OmDetTurboForObjectDetection"),
        ("oneformer", "OneFormerModel"),
        ("open-llama", "OpenLlamaModel"),
        ("openai-gpt", "OpenAIGPTModel"),
        ("opt", "OPTModel"),
        ("owlv2", "Owlv2Model"),
        ("owlvit", "OwlViTModel"),
        ("patchtsmixer", "PatchTSMixerModel"),
        ("patchtst", "PatchTSTModel"),
        ("pegasus", "PegasusModel"),
        ("pegasus_x", "PegasusXModel"),
        ("perceiver", "PerceiverModel"),
        ("persimmon", "PersimmonModel"),
        ("phi", "PhiModel"),
        ("phi3", "Phi3Model"),
        ("phimoe", "PhimoeModel"),
        ("pixtral", "PixtralVisionModel"),
        ("plbart", "PLBartModel"),
        ("poolformer", "PoolFormerModel"),
        ("prophetnet", "ProphetNetModel"),
        ("pvt", "PvtModel"),
        ("pvt_v2", "PvtV2Model"),
        ("qdqbert", "QDQBertModel"),
        ("qwen2", "Qwen2Model"),
        ("qwen2_audio_encoder", "Qwen2AudioEncoder"),
        ("qwen2_moe", "Qwen2MoeModel"),
        ("qwen2_vl", "Qwen2VLModel"),
        ("recurrent_gemma", "RecurrentGemmaModel"),
        ("reformer", "ReformerModel"),
        ("regnet", "RegNetModel"),
        ("rembert", "RemBertModel"),
        ("resnet", "ResNetModel"),
        ("retribert", "RetriBertModel"),
        ("roberta", "RobertaModel"),
        ("roberta-prelayernorm", "RobertaPreLayerNormModel"),
        ("roc_bert", "RoCBertModel"),
        ("roformer", "RoFormerModel"),
        ("rt_detr", "RTDetrModel"),
        ("rwkv", "RwkvModel"),
        ("sam", "SamModel"),
        ("seamless_m4t", "SeamlessM4TModel"),
        ("seamless_m4t_v2", "SeamlessM4Tv2Model"),
        ("segformer", "SegformerModel"),
        ("seggpt", "SegGptModel"),
        ("sew", "SEWModel"),
        ("sew-d", "SEWDModel"),
        ("siglip", "SiglipModel"),
        ("siglip_vision_model", "SiglipVisionModel"),
        ("speech_to_text", "Speech2TextModel"),
        ("speecht5", "SpeechT5Model"),
        ("splinter", "SplinterModel"),
        ("squeezebert", "SqueezeBertModel"),
        ("stablelm", "StableLmModel"),
        ("starcoder2", "Starcoder2Model"),
        ("swiftformer", "SwiftFormerModel"),
        ("swin", "SwinModel"),
        ("swin2sr", "Swin2SRModel"),
        ("swinv2", "Swinv2Model"),
        ("switch_transformers", "SwitchTransformersModel"),
        ("t5", "T5Model"),
        ("table-transformer", "TableTransformerModel"),
        ("tapas", "TapasModel"),
        ("textnet", "TextNetModel"),
        ("time_series_transformer", "TimeSeriesTransformerModel"),
        ("timesformer", "TimesformerModel"),
        ("timm_backbone", "TimmBackbone"),
        ("timm_wrapper", "TimmWrapperModel"),
        ("trajectory_transformer", "TrajectoryTransformerModel"),
        ("transfo-xl", "TransfoXLModel"),
        ("tvlt", "TvltModel"),
        ("tvp", "TvpModel"),
        ("udop", "UdopModel"),
        ("umt5", "UMT5Model"),
        ("unispeech", "UniSpeechModel"),
        ("unispeech-sat", "UniSpeechSatModel"),
        ("univnet", "UnivNetModel"),
        ("van", "VanModel"),
        ("videomae", "VideoMAEModel"),
        ("vilt", "ViltModel"),
        ("vision-text-dual-encoder", "VisionTextDualEncoderModel"),
        ("visual_bert", "VisualBertModel"),
        ("vit", "ViTModel"),
        ("vit_hybrid", "ViTHybridModel"),
        ("vit_mae", "ViTMAEModel"),
        ("vit_msn", "ViTMSNModel"),
        ("vitdet", "VitDetModel"),
        ("vits", "VitsModel"),
        ("vivit", "VivitModel"),
        ("wav2vec2", "Wav2Vec2Model"),
        ("wav2vec2-bert", "Wav2Vec2BertModel"),
        ("wav2vec2-conformer", "Wav2Vec2ConformerModel"),
        ("wavlm", "WavLMModel"),
        ("whisper", "WhisperModel"),
        ("xclip", "XCLIPModel"),
        ("xglm", "XGLMModel"),
        ("xlm", "XLMModel"),
        ("xlm-prophetnet", "XLMProphetNetModel"),
        ("xlm-roberta", "XLMRobertaModel"),
        ("xlm-roberta-xl", "XLMRobertaXLModel"),
        ("xlnet", "XLNetModel"),
        ("xmod", "XmodModel"),
        ("yolos", "YolosModel"),
        ("yoso", "YosoModel"),
        ("zamba", "ZambaModel"),
    ]
)

MODEL_FOR_PRETRAINING_MAPPING_NAMES = OrderedDict(
    [
        # Model for pre-training mapping
        ("albert", "AlbertForPreTraining"),
        ("bart", "BartForConditionalGeneration"),
        ("bert", "BertForPreTraining"),
        ("big_bird", "BigBirdForPreTraining"),
        ("bloom", "BloomForCausalLM"),
        ("camembert", "CamembertForMaskedLM"),
        ("colpali", "ColPaliForRetrieval"),
        ("ctrl", "CTRLLMHeadModel"),
        ("data2vec-text", "Data2VecTextForMaskedLM"),
        ("deberta", "DebertaForMaskedLM"),
        ("deberta-v2", "DebertaV2ForMaskedLM"),
        ("distilbert", "DistilBertForMaskedLM"),
        ("electra", "ElectraForPreTraining"),
        ("ernie", "ErnieForPreTraining"),
        ("falcon_mamba", "FalconMambaForCausalLM"),
        ("flaubert", "FlaubertWithLMHeadModel"),
        ("flava", "FlavaForPreTraining"),
        ("fnet", "FNetForPreTraining"),
        ("fsmt", "FSMTForConditionalGeneration"),
        ("funnel", "FunnelForPreTraining"),
        ("gpt-sw3", "GPT2LMHeadModel"),
        ("gpt2", "GPT2LMHeadModel"),
        ("gpt_bigcode", "GPTBigCodeForCausalLM"),
        ("gptsan-japanese", "GPTSanJapaneseForConditionalGeneration"),
        ("hiera", "HieraForPreTraining"),
        ("ibert", "IBertForMaskedLM"),
        ("idefics", "IdeficsForVisionText2Text"),
        ("idefics2", "Idefics2ForConditionalGeneration"),
        ("idefics3", "Idefics3ForConditionalGeneration"),
        ("layoutlm", "LayoutLMForMaskedLM"),
        ("llava", "LlavaForConditionalGeneration"),
        ("llava_next", "LlavaNextForConditionalGeneration"),
        ("llava_next_video", "LlavaNextVideoForConditionalGeneration"),
        ("llava_onevision", "LlavaOnevisionForConditionalGeneration"),
        ("longformer", "LongformerForMaskedLM"),
        ("luke", "LukeForMaskedLM"),
        ("lxmert", "LxmertForPreTraining"),
        ("mamba", "MambaForCausalLM"),
        ("mamba2", "Mamba2ForCausalLM"),
        ("mega", "MegaForMaskedLM"),
        ("megatron-bert", "MegatronBertForPreTraining"),
        ("mllama", "MllamaForConditionalGeneration"),
        ("mobilebert", "MobileBertForPreTraining"),
        ("mpnet", "MPNetForMaskedLM"),
        ("mpt", "MptForCausalLM"),
        ("mra", "MraForMaskedLM"),
        ("mvp", "MvpForConditionalGeneration"),
        ("nezha", "NezhaForPreTraining"),
        ("nllb-moe", "NllbMoeForConditionalGeneration"),
        ("openai-gpt", "OpenAIGPTLMHeadModel"),
        ("paligemma", "PaliGemmaForConditionalGeneration"),
        ("qwen2_audio", "Qwen2AudioForConditionalGeneration"),
        ("retribert", "RetriBertModel"),
        ("roberta", "RobertaForMaskedLM"),
        ("roberta-prelayernorm", "RobertaPreLayerNormForMaskedLM"),
        ("roc_bert", "RoCBertForPreTraining"),
        ("rwkv", "RwkvForCausalLM"),
        ("splinter", "SplinterForPreTraining"),
        ("squeezebert", "SqueezeBertForMaskedLM"),
        ("switch_transformers", "SwitchTransformersForConditionalGeneration"),
        ("t5", "T5ForConditionalGeneration"),
        ("tapas", "TapasForMaskedLM"),
        ("transfo-xl", "TransfoXLLMHeadModel"),
        ("tvlt", "TvltForPreTraining"),
        ("unispeech", "UniSpeechForPreTraining"),
        ("unispeech-sat", "UniSpeechSatForPreTraining"),
        ("video_llava", "VideoLlavaForConditionalGeneration"),
        ("videomae", "VideoMAEForPreTraining"),
        ("vipllava", "VipLlavaForConditionalGeneration"),
        ("visual_bert", "VisualBertForPreTraining"),
        ("vit_mae", "ViTMAEForPreTraining"),
        ("wav2vec2", "Wav2Vec2ForPreTraining"),
        ("wav2vec2-conformer", "Wav2Vec2ConformerForPreTraining"),
        ("xlm", "XLMWithLMHeadModel"),
        ("xlm-roberta", "XLMRobertaForMaskedLM"),
        ("xlm-roberta-xl", "XLMRobertaXLForMaskedLM"),
        ("xlnet", "XLNetLMHeadModel"),
        ("xmod", "XmodForMaskedLM"),
    ]
)

MODEL_WITH_LM_HEAD_MAPPING_NAMES = OrderedDict(
    [
        # Model with LM heads mapping
        ("albert", "AlbertForMaskedLM"),
        ("bart", "BartForConditionalGeneration"),
        ("bert", "BertForMaskedLM"),
        ("big_bird", "BigBirdForMaskedLM"),
        ("bigbird_pegasus", "BigBirdPegasusForConditionalGeneration"),
        ("blenderbot-small", "BlenderbotSmallForConditionalGeneration"),
        ("bloom", "BloomForCausalLM"),
        ("camembert", "CamembertForMaskedLM"),
        ("codegen", "CodeGenForCausalLM"),
        ("convbert", "ConvBertForMaskedLM"),
        ("cpmant", "CpmAntForCausalLM"),
        ("ctrl", "CTRLLMHeadModel"),
        ("data2vec-text", "Data2VecTextForMaskedLM"),
        ("deberta", "DebertaForMaskedLM"),
        ("deberta-v2", "DebertaV2ForMaskedLM"),
        ("distilbert", "DistilBertForMaskedLM"),
        ("electra", "ElectraForMaskedLM"),
        ("encoder-decoder", "EncoderDecoderModel"),
        ("ernie", "ErnieForMaskedLM"),
        ("esm", "EsmForMaskedLM"),
        ("falcon_mamba", "FalconMambaForCausalLM"),
        ("flaubert", "FlaubertWithLMHeadModel"),
        ("fnet", "FNetForMaskedLM"),
        ("fsmt", "FSMTForConditionalGeneration"),
        ("funnel", "FunnelForMaskedLM"),
        ("git", "GitForCausalLM"),
        ("gpt-sw3", "GPT2LMHeadModel"),
        ("gpt2", "GPT2LMHeadModel"),
        ("gpt_bigcode", "GPTBigCodeForCausalLM"),
        ("gpt_neo", "GPTNeoForCausalLM"),
        ("gpt_neox", "GPTNeoXForCausalLM"),
        ("gpt_neox_japanese", "GPTNeoXJapaneseForCausalLM"),
        ("gptj", "GPTJForCausalLM"),
        ("gptsan-japanese", "GPTSanJapaneseForConditionalGeneration"),
        ("ibert", "IBertForMaskedLM"),
        ("layoutlm", "LayoutLMForMaskedLM"),
        ("led", "LEDForConditionalGeneration"),
        ("longformer", "LongformerForMaskedLM"),
        ("longt5", "LongT5ForConditionalGeneration"),
        ("luke", "LukeForMaskedLM"),
        ("m2m_100", "M2M100ForConditionalGeneration"),
        ("mamba", "MambaForCausalLM"),
        ("mamba2", "Mamba2ForCausalLM"),
        ("marian", "MarianMTModel"),
        ("mega", "MegaForMaskedLM"),
        ("megatron-bert", "MegatronBertForCausalLM"),
        ("mobilebert", "MobileBertForMaskedLM"),
        ("moonshine", "MoonshineForConditionalGeneration"),
        ("mpnet", "MPNetForMaskedLM"),
        ("mpt", "MptForCausalLM"),
        ("mra", "MraForMaskedLM"),
        ("mvp", "MvpForConditionalGeneration"),
        ("nezha", "NezhaForMaskedLM"),
        ("nllb-moe", "NllbMoeForConditionalGeneration"),
        ("nystromformer", "NystromformerForMaskedLM"),
        ("openai-gpt", "OpenAIGPTLMHeadModel"),
        ("pegasus_x", "PegasusXForConditionalGeneration"),
        ("plbart", "PLBartForConditionalGeneration"),
        ("pop2piano", "Pop2PianoForConditionalGeneration"),
        ("qdqbert", "QDQBertForMaskedLM"),
        ("reformer", "ReformerModelWithLMHead"),
        ("rembert", "RemBertForMaskedLM"),
        ("roberta", "RobertaForMaskedLM"),
        ("roberta-prelayernorm", "RobertaPreLayerNormForMaskedLM"),
        ("roc_bert", "RoCBertForMaskedLM"),
        ("roformer", "RoFormerForMaskedLM"),
        ("rwkv", "RwkvForCausalLM"),
        ("speech_to_text", "Speech2TextForConditionalGeneration"),
        ("squeezebert", "SqueezeBertForMaskedLM"),
        ("switch_transformers", "SwitchTransformersForConditionalGeneration"),
        ("t5", "T5ForConditionalGeneration"),
        ("tapas", "TapasForMaskedLM"),
        ("transfo-xl", "TransfoXLLMHeadModel"),
        ("wav2vec2", "Wav2Vec2ForMaskedLM"),
        ("whisper", "WhisperForConditionalGeneration"),
        ("xlm", "XLMWithLMHeadModel"),
        ("xlm-roberta", "XLMRobertaForMaskedLM"),
        ("xlm-roberta-xl", "XLMRobertaXLForMaskedLM"),
        ("xlnet", "XLNetLMHeadModel"),
        ("xmod", "XmodForMaskedLM"),
        ("yoso", "YosoForMaskedLM"),
    ]
)

MODEL_FOR_CAUSAL_LM_MAPPING_NAMES = OrderedDict(
    [
        # Model for Causal LM mapping
        ("aria_text", "AriaTextForCausalLM"),
        ("bamba", "BambaForCausalLM"),
        ("bart", "BartForCausalLM"),
        ("bert", "BertLMHeadModel"),
        ("bert-generation", "BertGenerationDecoder"),
        ("big_bird", "BigBirdForCausalLM"),
        ("bigbird_pegasus", "BigBirdPegasusForCausalLM"),
        ("biogpt", "BioGptForCausalLM"),
        ("blenderbot", "BlenderbotForCausalLM"),
        ("blenderbot-small", "BlenderbotSmallForCausalLM"),
        ("bloom", "BloomForCausalLM"),
        ("camembert", "CamembertForCausalLM"),
        ("code_llama", "LlamaForCausalLM"),
        ("codegen", "CodeGenForCausalLM"),
        ("cohere", "CohereForCausalLM"),
        ("cohere2", "Cohere2ForCausalLM"),
        ("cpmant", "CpmAntForCausalLM"),
        ("ctrl", "CTRLLMHeadModel"),
        ("data2vec-text", "Data2VecTextForCausalLM"),
        ("dbrx", "DbrxForCausalLM"),
        ("diffllama", "DiffLlamaForCausalLM"),
        ("electra", "ElectraForCausalLM"),
        ("emu3", "Emu3ForCausalLM"),
        ("ernie", "ErnieForCausalLM"),
        ("falcon", "FalconForCausalLM"),
        ("falcon_mamba", "FalconMambaForCausalLM"),
        ("fuyu", "FuyuForCausalLM"),
        ("gemma", "GemmaForCausalLM"),
        ("gemma2", "Gemma2ForCausalLM"),
        ("git", "GitForCausalLM"),
        ("glm", "GlmForCausalLM"),
        ("gpt-sw3", "GPT2LMHeadModel"),
        ("gpt2", "GPT2LMHeadModel"),
        ("gpt_bigcode", "GPTBigCodeForCausalLM"),
        ("gpt_neo", "GPTNeoForCausalLM"),
        ("gpt_neox", "GPTNeoXForCausalLM"),
        ("gpt_neox_japanese", "GPTNeoXJapaneseForCausalLM"),
        ("gptj", "GPTJForCausalLM"),
        ("granite", "GraniteForCausalLM"),
        ("granitemoe", "GraniteMoeForCausalLM"),
        ("helium", "HeliumForCausalLM"),
        ("internlm3", "InternLM3ForCausalLM"),
        ("jamba", "JambaForCausalLM"),
        ("jetmoe", "JetMoeForCausalLM"),
        ("llama", "LlamaForCausalLM"),
        ("mamba", "MambaForCausalLM"),
        ("mamba2", "Mamba2ForCausalLM"),
        ("marian", "MarianForCausalLM"),
        ("mbart", "MBartForCausalLM"),
        ("mega", "MegaForCausalLM"),
        ("megatron-bert", "MegatronBertForCausalLM"),
        ("mistral", "MistralForCausalLM"),
        ("mixtral", "MixtralForCausalLM"),
        ("mllama", "MllamaForCausalLM"),
        ("moshi", "MoshiForCausalLM"),
        ("mpt", "MptForCausalLM"),
        ("musicgen", "MusicgenForCausalLM"),
        ("musicgen_melody", "MusicgenMelodyForCausalLM"),
        ("mvp", "MvpForCausalLM"),
        ("nemotron", "NemotronForCausalLM"),
        ("olmo", "OlmoForCausalLM"),
        ("olmo2", "Olmo2ForCausalLM"),
        ("olmoe", "OlmoeForCausalLM"),
        ("open-llama", "OpenLlamaForCausalLM"),
        ("openai-gpt", "OpenAIGPTLMHeadModel"),
        ("opt", "OPTForCausalLM"),
        ("pegasus", "PegasusForCausalLM"),
        ("persimmon", "PersimmonForCausalLM"),
        ("phi", "PhiForCausalLM"),
        ("phi3", "Phi3ForCausalLM"),
        ("phimoe", "PhimoeForCausalLM"),
        ("plbart", "PLBartForCausalLM"),
        ("prophetnet", "ProphetNetForCausalLM"),
        ("qdqbert", "QDQBertLMHeadModel"),
        ("qwen2", "Qwen2ForCausalLM"),
        ("qwen2_moe", "Qwen2MoeForCausalLM"),
        ("recurrent_gemma", "RecurrentGemmaForCausalLM"),
        ("reformer", "ReformerModelWithLMHead"),
        ("rembert", "RemBertForCausalLM"),
        ("roberta", "RobertaForCausalLM"),
        ("roberta-prelayernorm", "RobertaPreLayerNormForCausalLM"),
        ("roc_bert", "RoCBertForCausalLM"),
        ("roformer", "RoFormerForCausalLM"),
        ("rwkv", "RwkvForCausalLM"),
        ("speech_to_text_2", "Speech2Text2ForCausalLM"),
        ("stablelm", "StableLmForCausalLM"),
        ("starcoder2", "Starcoder2ForCausalLM"),
        ("transfo-xl", "TransfoXLLMHeadModel"),
        ("trocr", "TrOCRForCausalLM"),
        ("whisper", "WhisperForCausalLM"),
        ("xglm", "XGLMForCausalLM"),
        ("xlm", "XLMWithLMHeadModel"),
        ("xlm-prophetnet", "XLMProphetNetForCausalLM"),
        ("xlm-roberta", "XLMRobertaForCausalLM"),
        ("xlm-roberta-xl", "XLMRobertaXLForCausalLM"),
        ("xlnet", "XLNetLMHeadModel"),
        ("xmod", "XmodForCausalLM"),
        ("zamba", "ZambaForCausalLM"),
    ]
)

MODEL_FOR_IMAGE_MAPPING_NAMES = OrderedDict(
    [
        # Model for Image mapping
        ("beit", "BeitModel"),
        ("bit", "BitModel"),
        ("conditional_detr", "ConditionalDetrModel"),
        ("convnext", "ConvNextModel"),
        ("convnextv2", "ConvNextV2Model"),
        ("data2vec-vision", "Data2VecVisionModel"),
        ("deformable_detr", "DeformableDetrModel"),
        ("deit", "DeiTModel"),
        ("deta", "DetaModel"),
        ("detr", "DetrModel"),
        ("dinat", "DinatModel"),
        ("dinov2", "Dinov2Model"),
        ("dinov2_with_registers", "Dinov2WithRegistersModel"),
        ("dpt", "DPTModel"),
        ("efficientformer", "EfficientFormerModel"),
        ("efficientnet", "EfficientNetModel"),
        ("focalnet", "FocalNetModel"),
        ("glpn", "GLPNModel"),
        ("hiera", "HieraModel"),
        ("ijepa", "IJepaModel"),
        ("imagegpt", "ImageGPTModel"),
        ("levit", "LevitModel"),
        ("mllama", "MllamaVisionModel"),
        ("mobilenet_v1", "MobileNetV1Model"),
        ("mobilenet_v2", "MobileNetV2Model"),
        ("mobilevit", "MobileViTModel"),
        ("mobilevitv2", "MobileViTV2Model"),
        ("nat", "NatModel"),
        ("poolformer", "PoolFormerModel"),
        ("pvt", "PvtModel"),
        ("regnet", "RegNetModel"),
        ("resnet", "ResNetModel"),
        ("segformer", "SegformerModel"),
        ("siglip_vision_model", "SiglipVisionModel"),
        ("swiftformer", "SwiftFormerModel"),
        ("swin", "SwinModel"),
        ("swin2sr", "Swin2SRModel"),
        ("swinv2", "Swinv2Model"),
        ("table-transformer", "TableTransformerModel"),
        ("timesformer", "TimesformerModel"),
        ("timm_backbone", "TimmBackbone"),
        ("timm_wrapper", "TimmWrapperModel"),
        ("van", "VanModel"),
        ("videomae", "VideoMAEModel"),
        ("vit", "ViTModel"),
        ("vit_hybrid", "ViTHybridModel"),
        ("vit_mae", "ViTMAEModel"),
        ("vit_msn", "ViTMSNModel"),
        ("vitdet", "VitDetModel"),
        ("vivit", "VivitModel"),
        ("yolos", "YolosModel"),
    ]
)

MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING_NAMES = OrderedDict(
    [
        ("deit", "DeiTForMaskedImageModeling"),
        ("focalnet", "FocalNetForMaskedImageModeling"),
        ("swin", "SwinForMaskedImageModeling"),
        ("swinv2", "Swinv2ForMaskedImageModeling"),
        ("vit", "ViTForMaskedImageModeling"),
    ]
)


MODEL_FOR_CAUSAL_IMAGE_MODELING_MAPPING_NAMES = OrderedDict(
    # Model for Causal Image Modeling mapping
    [
        ("imagegpt", "ImageGPTForCausalImageModeling"),
    ]
)

MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Image Classification mapping
        ("beit", "BeitForImageClassification"),
        ("bit", "BitForImageClassification"),
        ("clip", "CLIPForImageClassification"),
        ("convnext", "ConvNextForImageClassification"),
        ("convnextv2", "ConvNextV2ForImageClassification"),
        ("cvt", "CvtForImageClassification"),
        ("data2vec-vision", "Data2VecVisionForImageClassification"),
        (
            "deit",
            ("DeiTForImageClassification", "DeiTForImageClassificationWithTeacher"),
        ),
        ("dinat", "DinatForImageClassification"),
        ("dinov2", "Dinov2ForImageClassification"),
        ("dinov2_with_registers", "Dinov2WithRegistersForImageClassification"),
        (
            "efficientformer",
            (
                "EfficientFormerForImageClassification",
                "EfficientFormerForImageClassificationWithTeacher",
            ),
        ),
        ("efficientnet", "EfficientNetForImageClassification"),
        ("focalnet", "FocalNetForImageClassification"),
        ("hiera", "HieraForImageClassification"),
        ("ijepa", "IJepaForImageClassification"),
        ("imagegpt", "ImageGPTForImageClassification"),
        (
            "levit",
            ("LevitForImageClassification", "LevitForImageClassificationWithTeacher"),
        ),
        ("mobilenet_v1", "MobileNetV1ForImageClassification"),
        ("mobilenet_v2", "MobileNetV2ForImageClassification"),
        ("mobilevit", "MobileViTForImageClassification"),
        ("mobilevitv2", "MobileViTV2ForImageClassification"),
        ("nat", "NatForImageClassification"),
        (
            "perceiver",
            (
                "PerceiverForImageClassificationLearned",
                "PerceiverForImageClassificationFourier",
                "PerceiverForImageClassificationConvProcessing",
            ),
        ),
        ("poolformer", "PoolFormerForImageClassification"),
        ("pvt", "PvtForImageClassification"),
        ("pvt_v2", "PvtV2ForImageClassification"),
        ("regnet", "RegNetForImageClassification"),
        ("resnet", "ResNetForImageClassification"),
        ("segformer", "SegformerForImageClassification"),
        ("siglip", "SiglipForImageClassification"),
        ("swiftformer", "SwiftFormerForImageClassification"),
        ("swin", "SwinForImageClassification"),
        ("swinv2", "Swinv2ForImageClassification"),
        ("textnet", "TextNetForImageClassification"),
        ("timm_wrapper", "TimmWrapperForImageClassification"),
        ("van", "VanForImageClassification"),
        ("vit", "ViTForImageClassification"),
        ("vit_hybrid", "ViTHybridForImageClassification"),
        ("vit_msn", "ViTMSNForImageClassification"),
    ]
)

MODEL_FOR_IMAGE_SEGMENTATION_MAPPING_NAMES = OrderedDict(
    [
        # Do not add new models here, this class will be deprecated in the future.
        # Model for Image Segmentation mapping
        ("detr", "DetrForSegmentation"),
    ]
)

MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Semantic Segmentation mapping
        ("beit", "BeitForSemanticSegmentation"),
        ("data2vec-vision", "Data2VecVisionForSemanticSegmentation"),
        ("dpt", "DPTForSemanticSegmentation"),
        ("mobilenet_v2", "MobileNetV2ForSemanticSegmentation"),
        ("mobilevit", "MobileViTForSemanticSegmentation"),
        ("mobilevitv2", "MobileViTV2ForSemanticSegmentation"),
        ("segformer", "SegformerForSemanticSegmentation"),
        ("upernet", "UperNetForSemanticSegmentation"),
    ]
)

MODEL_FOR_INSTANCE_SEGMENTATION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Instance Segmentation mapping
        # MaskFormerForInstanceSegmentation can be removed from this mapping in v5
        ("maskformer", "MaskFormerForInstanceSegmentation"),
    ]
)

MODEL_FOR_UNIVERSAL_SEGMENTATION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Universal Segmentation mapping
        ("detr", "DetrForSegmentation"),
        ("mask2former", "Mask2FormerForUniversalSegmentation"),
        ("maskformer", "MaskFormerForInstanceSegmentation"),
        ("oneformer", "OneFormerForUniversalSegmentation"),
    ]
)

MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        ("timesformer", "TimesformerForVideoClassification"),
        ("videomae", "VideoMAEForVideoClassification"),
        ("vivit", "VivitForVideoClassification"),
    ]
)

MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES = OrderedDict(
    [
        ("blip", "BlipForConditionalGeneration"),
        ("blip-2", "Blip2ForConditionalGeneration"),
        ("chameleon", "ChameleonForConditionalGeneration"),
        ("git", "GitForCausalLM"),
        ("idefics2", "Idefics2ForConditionalGeneration"),
        ("idefics3", "Idefics3ForConditionalGeneration"),
        ("instructblip", "InstructBlipForConditionalGeneration"),
        ("instructblipvideo", "InstructBlipVideoForConditionalGeneration"),
        ("kosmos-2", "Kosmos2ForConditionalGeneration"),
        ("llava", "LlavaForConditionalGeneration"),
        ("llava_next", "LlavaNextForConditionalGeneration"),
        ("llava_next_video", "LlavaNextVideoForConditionalGeneration"),
        ("llava_onevision", "LlavaOnevisionForConditionalGeneration"),
        ("mllama", "MllamaForConditionalGeneration"),
        ("paligemma", "PaliGemmaForConditionalGeneration"),
        ("pix2struct", "Pix2StructForConditionalGeneration"),
        ("qwen2_vl", "Qwen2VLForConditionalGeneration"),
        ("video_llava", "VideoLlavaForConditionalGeneration"),
        ("vipllava", "VipLlavaForConditionalGeneration"),
        ("vision-encoder-decoder", "VisionEncoderDecoderModel"),
    ]
)

MODEL_FOR_RETRIEVAL_MAPPING_NAMES = OrderedDict(
    [
        ("colpali", "ColPaliForRetrieval"),
    ]
)

MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES = OrderedDict(
    [
        ("aria", "AriaForConditionalGeneration"),
        ("blip", "BlipForConditionalGeneration"),
        ("blip-2", "Blip2ForConditionalGeneration"),
        ("chameleon", "ChameleonForConditionalGeneration"),
        ("emu3", "Emu3ForConditionalGeneration"),
        ("fuyu", "FuyuForCausalLM"),
        ("git", "GitForCausalLM"),
        ("idefics", "IdeficsForVisionText2Text"),
        ("idefics2", "Idefics2ForConditionalGeneration"),
        ("idefics3", "Idefics3ForConditionalGeneration"),
        ("instructblip", "InstructBlipForConditionalGeneration"),
        ("kosmos-2", "Kosmos2ForConditionalGeneration"),
        ("llava", "LlavaForConditionalGeneration"),
        ("llava_next", "LlavaNextForConditionalGeneration"),
        ("llava_onevision", "LlavaOnevisionForConditionalGeneration"),
        ("mllama", "MllamaForConditionalGeneration"),
        ("paligemma", "PaliGemmaForConditionalGeneration"),
        ("pix2struct", "Pix2StructForConditionalGeneration"),
        ("pixtral", "LlavaForConditionalGeneration"),
        ("qwen2_vl", "Qwen2VLForConditionalGeneration"),
        ("udop", "UdopForConditionalGeneration"),
        ("vipllava", "VipLlavaForConditionalGeneration"),
        ("vision-encoder-decoder", "VisionEncoderDecoderModel"),
    ]
)

MODEL_FOR_MASKED_LM_MAPPING_NAMES = OrderedDict(
    [
        # Model for Masked LM mapping
        ("albert", "AlbertForMaskedLM"),
        ("bart", "BartForConditionalGeneration"),
        ("bert", "BertForMaskedLM"),
        ("big_bird", "BigBirdForMaskedLM"),
        ("camembert", "CamembertForMaskedLM"),
        ("convbert", "ConvBertForMaskedLM"),
        ("data2vec-text", "Data2VecTextForMaskedLM"),
        ("deberta", "DebertaForMaskedLM"),
        ("deberta-v2", "DebertaV2ForMaskedLM"),
        ("distilbert", "DistilBertForMaskedLM"),
        ("electra", "ElectraForMaskedLM"),
        ("ernie", "ErnieForMaskedLM"),
        ("esm", "EsmForMaskedLM"),
        ("flaubert", "FlaubertWithLMHeadModel"),
        ("fnet", "FNetForMaskedLM"),
        ("funnel", "FunnelForMaskedLM"),
        ("ibert", "IBertForMaskedLM"),
        ("layoutlm", "LayoutLMForMaskedLM"),
        ("longformer", "LongformerForMaskedLM"),
        ("luke", "LukeForMaskedLM"),
        ("mbart", "MBartForConditionalGeneration"),
        ("mega", "MegaForMaskedLM"),
        ("megatron-bert", "MegatronBertForMaskedLM"),
        ("mobilebert", "MobileBertForMaskedLM"),
        ("modernbert", "ModernBertForMaskedLM"),
        ("mpnet", "MPNetForMaskedLM"),
        ("mra", "MraForMaskedLM"),
        ("mvp", "MvpForConditionalGeneration"),
        ("nezha", "NezhaForMaskedLM"),
        ("nystromformer", "NystromformerForMaskedLM"),
        ("perceiver", "PerceiverForMaskedLM"),
        ("qdqbert", "QDQBertForMaskedLM"),
        ("reformer", "ReformerForMaskedLM"),
        ("rembert", "RemBertForMaskedLM"),
        ("roberta", "RobertaForMaskedLM"),
        ("roberta-prelayernorm", "RobertaPreLayerNormForMaskedLM"),
        ("roc_bert", "RoCBertForMaskedLM"),
        ("roformer", "RoFormerForMaskedLM"),
        ("squeezebert", "SqueezeBertForMaskedLM"),
        ("tapas", "TapasForMaskedLM"),
        ("wav2vec2", "Wav2Vec2ForMaskedLM"),
        ("xlm", "XLMWithLMHeadModel"),
        ("xlm-roberta", "XLMRobertaForMaskedLM"),
        ("xlm-roberta-xl", "XLMRobertaXLForMaskedLM"),
        ("xmod", "XmodForMaskedLM"),
        ("yoso", "YosoForMaskedLM"),
    ]
)

MODEL_FOR_OBJECT_DETECTION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Object Detection mapping
        ("conditional_detr", "ConditionalDetrForObjectDetection"),
        ("deformable_detr", "DeformableDetrForObjectDetection"),
        ("deta", "DetaForObjectDetection"),
        ("detr", "DetrForObjectDetection"),
        ("rt_detr", "RTDetrForObjectDetection"),
        ("table-transformer", "TableTransformerForObjectDetection"),
        ("yolos", "YolosForObjectDetection"),
    ]
)

MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Zero Shot Object Detection mapping
        ("grounding-dino", "GroundingDinoForObjectDetection"),
        ("omdet-turbo", "OmDetTurboForObjectDetection"),
        ("owlv2", "Owlv2ForObjectDetection"),
        ("owlvit", "OwlViTForObjectDetection"),
    ]
)

MODEL_FOR_DEPTH_ESTIMATION_MAPPING_NAMES = OrderedDict(
    [
        # Model for depth estimation mapping
        ("depth_anything", "DepthAnythingForDepthEstimation"),
        ("dpt", "DPTForDepthEstimation"),
        ("glpn", "GLPNForDepthEstimation"),
        ("zoedepth", "ZoeDepthForDepthEstimation"),
    ]
)
MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES = OrderedDict(
    [
        # Model for Seq2Seq Causal LM mapping
        ("bart", "BartForConditionalGeneration"),
        ("bigbird_pegasus", "BigBirdPegasusForConditionalGeneration"),
        ("blenderbot", "BlenderbotForConditionalGeneration"),
        ("blenderbot-small", "BlenderbotSmallForConditionalGeneration"),
        ("encoder-decoder", "EncoderDecoderModel"),
        ("fsmt", "FSMTForConditionalGeneration"),
        ("gptsan-japanese", "GPTSanJapaneseForConditionalGeneration"),
        ("led", "LEDForConditionalGeneration"),
        ("longt5", "LongT5ForConditionalGeneration"),
        ("m2m_100", "M2M100ForConditionalGeneration"),
        ("marian", "MarianMTModel"),
        ("mbart", "MBartForConditionalGeneration"),
        ("mt5", "MT5ForConditionalGeneration"),
        ("mvp", "MvpForConditionalGeneration"),
        ("nllb-moe", "NllbMoeForConditionalGeneration"),
        ("pegasus", "PegasusForConditionalGeneration"),
        ("pegasus_x", "PegasusXForConditionalGeneration"),
        ("plbart", "PLBartForConditionalGeneration"),
        ("prophetnet", "ProphetNetForConditionalGeneration"),
        ("qwen2_audio", "Qwen2AudioForConditionalGeneration"),
        ("seamless_m4t", "SeamlessM4TForTextToText"),
        ("seamless_m4t_v2", "SeamlessM4Tv2ForTextToText"),
        ("switch_transformers", "SwitchTransformersForConditionalGeneration"),
        ("t5", "T5ForConditionalGeneration"),
        ("umt5", "UMT5ForConditionalGeneration"),
        ("xlm-prophetnet", "XLMProphetNetForConditionalGeneration"),
    ]
)

MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES = OrderedDict(
    [
        ("moonshine", "MoonshineForConditionalGeneration"),
        ("pop2piano", "Pop2PianoForConditionalGeneration"),
        ("seamless_m4t", "SeamlessM4TForSpeechToText"),
        ("seamless_m4t_v2", "SeamlessM4Tv2ForSpeechToText"),
        ("speech-encoder-decoder", "SpeechEncoderDecoderModel"),
        ("speech_to_text", "Speech2TextForConditionalGeneration"),
        ("speecht5", "SpeechT5ForSpeechToText"),
        ("whisper", "WhisperForConditionalGeneration"),
    ]
)

MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Sequence Classification mapping
        ("albert", "AlbertForSequenceClassification"),
        ("bart", "BartForSequenceClassification"),
        ("bert", "BertForSequenceClassification"),
        ("big_bird", "BigBirdForSequenceClassification"),
        ("bigbird_pegasus", "BigBirdPegasusForSequenceClassification"),
        ("biogpt", "BioGptForSequenceClassification"),
        ("bloom", "BloomForSequenceClassification"),
        ("camembert", "CamembertForSequenceClassification"),
        ("canine", "CanineForSequenceClassification"),
        ("code_llama", "LlamaForSequenceClassification"),
        ("convbert", "ConvBertForSequenceClassification"),
        ("ctrl", "CTRLForSequenceClassification"),
        ("data2vec-text", "Data2VecTextForSequenceClassification"),
        ("deberta", "DebertaForSequenceClassification"),
        ("deberta-v2", "DebertaV2ForSequenceClassification"),
        ("diffllama", "DiffLlamaForSequenceClassification"),
        ("distilbert", "DistilBertForSequenceClassification"),
        ("electra", "ElectraForSequenceClassification"),
        ("ernie", "ErnieForSequenceClassification"),
        ("ernie_m", "ErnieMForSequenceClassification"),
        ("esm", "EsmForSequenceClassification"),
        ("falcon", "FalconForSequenceClassification"),
        ("flaubert", "FlaubertForSequenceClassification"),
        ("fnet", "FNetForSequenceClassification"),
        ("funnel", "FunnelForSequenceClassification"),
        ("gemma", "GemmaForSequenceClassification"),
        ("gemma2", "Gemma2ForSequenceClassification"),
        ("glm", "GlmForSequenceClassification"),
        ("gpt-sw3", "GPT2ForSequenceClassification"),
        ("gpt2", "GPT2ForSequenceClassification"),
        ("gpt_bigcode", "GPTBigCodeForSequenceClassification"),
        ("gpt_neo", "GPTNeoForSequenceClassification"),
        ("gpt_neox", "GPTNeoXForSequenceClassification"),
        ("gptj", "GPTJForSequenceClassification"),
        ("helium", "HeliumForSequenceClassification"),
        ("ibert", "IBertForSequenceClassification"),
        ("jamba", "JambaForSequenceClassification"),
        ("jetmoe", "JetMoeForSequenceClassification"),
        ("layoutlm", "LayoutLMForSequenceClassification"),
        ("layoutlmv2", "LayoutLMv2ForSequenceClassification"),
        ("layoutlmv3", "LayoutLMv3ForSequenceClassification"),
        ("led", "LEDForSequenceClassification"),
        ("lilt", "LiltForSequenceClassification"),
        ("llama", "LlamaForSequenceClassification"),
        ("longformer", "LongformerForSequenceClassification"),
        ("luke", "LukeForSequenceClassification"),
        ("markuplm", "MarkupLMForSequenceClassification"),
        ("mbart", "MBartForSequenceClassification"),
        ("mega", "MegaForSequenceClassification"),
        ("megatron-bert", "MegatronBertForSequenceClassification"),
        ("mistral", "MistralForSequenceClassification"),
        ("mixtral", "MixtralForSequenceClassification"),
        ("mobilebert", "MobileBertForSequenceClassification"),
        ("modernbert", "ModernBertForSequenceClassification"),
        ("mpnet", "MPNetForSequenceClassification"),
        ("mpt", "MptForSequenceClassification"),
        ("mra", "MraForSequenceClassification"),
        ("mt5", "MT5ForSequenceClassification"),
        ("mvp", "MvpForSequenceClassification"),
        ("nemotron", "NemotronForSequenceClassification"),
        ("nezha", "NezhaForSequenceClassification"),
        ("nystromformer", "NystromformerForSequenceClassification"),
        ("open-llama", "OpenLlamaForSequenceClassification"),
        ("openai-gpt", "OpenAIGPTForSequenceClassification"),
        ("opt", "OPTForSequenceClassification"),
        ("perceiver", "PerceiverForSequenceClassification"),
        ("persimmon", "PersimmonForSequenceClassification"),
        ("phi", "PhiForSequenceClassification"),
        ("phi3", "Phi3ForSequenceClassification"),
        ("phimoe", "PhimoeForSequenceClassification"),
        ("plbart", "PLBartForSequenceClassification"),
        ("qdqbert", "QDQBertForSequenceClassification"),
        ("qwen2", "Qwen2ForSequenceClassification"),
        ("qwen2_moe", "Qwen2MoeForSequenceClassification"),
        ("reformer", "ReformerForSequenceClassification"),
        ("rembert", "RemBertForSequenceClassification"),
        ("roberta", "RobertaForSequenceClassification"),
        ("roberta-prelayernorm", "RobertaPreLayerNormForSequenceClassification"),
        ("roc_bert", "RoCBertForSequenceClassification"),
        ("roformer", "RoFormerForSequenceClassification"),
        ("squeezebert", "SqueezeBertForSequenceClassification"),
        ("stablelm", "StableLmForSequenceClassification"),
        ("starcoder2", "Starcoder2ForSequenceClassification"),
        ("t5", "T5ForSequenceClassification"),
        ("tapas", "TapasForSequenceClassification"),
        ("transfo-xl", "TransfoXLForSequenceClassification"),
        ("umt5", "UMT5ForSequenceClassification"),
        ("xlm", "XLMForSequenceClassification"),
        ("xlm-roberta", "XLMRobertaForSequenceClassification"),
        ("xlm-roberta-xl", "XLMRobertaXLForSequenceClassification"),
        ("xlnet", "XLNetForSequenceClassification"),
        ("xmod", "XmodForSequenceClassification"),
        ("yoso", "YosoForSequenceClassification"),
        ("zamba", "ZambaForSequenceClassification"),
    ]
)

MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES = OrderedDict(
    [
        # Model for Question Answering mapping
        ("albert", "AlbertForQuestionAnswering"),
        ("bart", "BartForQuestionAnswering"),
        ("bert", "BertForQuestionAnswering"),
        ("big_bird", "BigBirdForQuestionAnswering"),
        ("bigbird_pegasus", "BigBirdPegasusForQuestionAnswering"),
        ("bloom", "BloomForQuestionAnswering"),
        ("camembert", "CamembertForQuestionAnswering"),
        ("canine", "CanineForQuestionAnswering"),
        ("convbert", "ConvBertForQuestionAnswering"),
        ("data2vec-text", "Data2VecTextForQuestionAnswering"),
        ("deberta", "DebertaForQuestionAnswering"),
        ("deberta-v2", "DebertaV2ForQuestionAnswering"),
        ("diffllama", "DiffLlamaForQuestionAnswering"),
        ("distilbert", "DistilBertForQuestionAnswering"),
        ("electra", "ElectraForQuestionAnswering"),
        ("ernie", "ErnieForQuestionAnswering"),
        ("ernie_m", "ErnieMForQuestionAnswering"),
        ("falcon", "FalconForQuestionAnswering"),
        ("flaubert", "FlaubertForQuestionAnsweringSimple"),
        ("fnet", "FNetForQuestionAnswering"),
        ("funnel", "FunnelForQuestionAnswering"),
        ("gpt2", "GPT2ForQuestionAnswering"),
        ("gpt_neo", "GPTNeoForQuestionAnswering"),
        ("gpt_neox", "GPTNeoXForQuestionAnswering"),
        ("gptj", "GPTJForQuestionAnswering"),
        ("ibert", "IBertForQuestionAnswering"),
        ("layoutlmv2", "LayoutLMv2ForQuestionAnswering"),
        ("layoutlmv3", "LayoutLMv3ForQuestionAnswering"),
        ("led", "LEDForQuestionAnswering"),
        ("lilt", "LiltForQuestionAnswering"),
        ("llama", "LlamaForQuestionAnswering"),
        ("longformer", "LongformerForQuestionAnswering"),
        ("luke", "LukeForQuestionAnswering"),
        ("lxmert", "LxmertForQuestionAnswering"),
        ("markuplm", "MarkupLMForQuestionAnswering"),
        ("mbart", "MBartForQuestionAnswering"),
        ("mega", "MegaForQuestionAnswering"),
        ("megatron-bert", "MegatronBertForQuestionAnswering"),
        ("mistral", "MistralForQuestionAnswering"),
        ("mixtral", "MixtralForQuestionAnswering"),
        ("mobilebert", "MobileBertForQuestionAnswering"),
        ("mpnet", "MPNetForQuestionAnswering"),
        ("mpt", "MptForQuestionAnswering"),
        ("mra", "MraForQuestionAnswering"),
        ("mt5", "MT5ForQuestionAnswering"),
        ("mvp", "MvpForQuestionAnswering"),
        ("nemotron", "NemotronForQuestionAnswering"),
        ("nezha", "NezhaForQuestionAnswering"),
        ("nystromformer", "NystromformerForQuestionAnswering"),
        ("opt", "OPTForQuestionAnswering"),
        ("qdqbert", "QDQBertForQuestionAnswering"),
        ("qwen2", "Qwen2ForQuestionAnswering"),
        ("qwen2_moe", "Qwen2MoeForQuestionAnswering"),
        ("reformer", "ReformerForQuestionAnswering"),
        ("rembert", "RemBertForQuestionAnswering"),
        ("roberta", "RobertaForQuestionAnswering"),
        ("roberta-prelayernorm", "RobertaPreLayerNormForQuestionAnswering"),
        ("roc_bert", "RoCBertForQuestionAnswering"),
        ("roformer", "RoFormerForQuestionAnswering"),
        ("splinter", "SplinterForQuestionAnswering"),
        ("squeezebert", "SqueezeBertForQuestionAnswering"),
        ("t5", "T5ForQuestionAnswering"),
        ("umt5", "UMT5ForQuestionAnswering"),
        ("xlm", "XLMForQuestionAnsweringSimple"),
        ("xlm-roberta", "XLMRobertaForQuestionAnswering"),
        ("xlm-roberta-xl", "XLMRobertaXLForQuestionAnswering"),
        ("xlnet", "XLNetForQuestionAnsweringSimple"),
        ("xmod", "XmodForQuestionAnswering"),
        ("yoso", "YosoForQuestionAnswering"),
    ]
)

MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING_NAMES = OrderedDict(
    [
        # Model for Table Question Answering mapping
        ("tapas", "TapasForQuestionAnswering"),
    ]
)

MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING_NAMES = OrderedDict(
    [
        ("blip", "BlipForQuestionAnswering"),
        ("blip-2", "Blip2ForConditionalGeneration"),
        ("vilt", "ViltForQuestionAnswering"),
    ]
)

MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES = OrderedDict(
    [
        ("layoutlm", "LayoutLMForQuestionAnswering"),
        ("layoutlmv2", "LayoutLMv2ForQuestionAnswering"),
        ("layoutlmv3", "LayoutLMv3ForQuestionAnswering"),
    ]
)

MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Token Classification mapping
        ("albert", "AlbertForTokenClassification"),
        ("bert", "BertForTokenClassification"),
        ("big_bird", "BigBirdForTokenClassification"),
        ("biogpt", "BioGptForTokenClassification"),
        ("bloom", "BloomForTokenClassification"),
        ("bros", "BrosForTokenClassification"),
        ("camembert", "CamembertForTokenClassification"),
        ("canine", "CanineForTokenClassification"),
        ("convbert", "ConvBertForTokenClassification"),
        ("data2vec-text", "Data2VecTextForTokenClassification"),
        ("deberta", "DebertaForTokenClassification"),
        ("deberta-v2", "DebertaV2ForTokenClassification"),
        ("diffllama", "DiffLlamaForTokenClassification"),
        ("distilbert", "DistilBertForTokenClassification"),
        ("electra", "ElectraForTokenClassification"),
        ("ernie", "ErnieForTokenClassification"),
        ("ernie_m", "ErnieMForTokenClassification"),
        ("esm", "EsmForTokenClassification"),
        ("falcon", "FalconForTokenClassification"),
        ("flaubert", "FlaubertForTokenClassification"),
        ("fnet", "FNetForTokenClassification"),
        ("funnel", "FunnelForTokenClassification"),
        ("gemma", "GemmaForTokenClassification"),
        ("gemma2", "Gemma2ForTokenClassification"),
        ("glm", "GlmForTokenClassification"),
        ("gpt-sw3", "GPT2ForTokenClassification"),
        ("gpt2", "GPT2ForTokenClassification"),
        ("gpt_bigcode", "GPTBigCodeForTokenClassification"),
        ("gpt_neo", "GPTNeoForTokenClassification"),
        ("gpt_neox", "GPTNeoXForTokenClassification"),
        ("helium", "HeliumForTokenClassification"),
        ("ibert", "IBertForTokenClassification"),
        ("layoutlm", "LayoutLMForTokenClassification"),
        ("layoutlmv2", "LayoutLMv2ForTokenClassification"),
        ("layoutlmv3", "LayoutLMv3ForTokenClassification"),
        ("lilt", "LiltForTokenClassification"),
        ("llama", "LlamaForTokenClassification"),
        ("longformer", "LongformerForTokenClassification"),
        ("luke", "LukeForTokenClassification"),
        ("markuplm", "MarkupLMForTokenClassification"),
        ("mega", "MegaForTokenClassification"),
        ("megatron-bert", "MegatronBertForTokenClassification"),
        ("mistral", "MistralForTokenClassification"),
        ("mixtral", "MixtralForTokenClassification"),
        ("mobilebert", "MobileBertForTokenClassification"),
        ("modernbert", "ModernBertForTokenClassification"),
        ("mpnet", "MPNetForTokenClassification"),
        ("mpt", "MptForTokenClassification"),
        ("mra", "MraForTokenClassification"),
        ("mt5", "MT5ForTokenClassification"),
        ("nemotron", "NemotronForTokenClassification"),
        ("nezha", "NezhaForTokenClassification"),
        ("nystromformer", "NystromformerForTokenClassification"),
        ("persimmon", "PersimmonForTokenClassification"),
        ("phi", "PhiForTokenClassification"),
        ("phi3", "Phi3ForTokenClassification"),
        ("qdqbert", "QDQBertForTokenClassification"),
        ("qwen2", "Qwen2ForTokenClassification"),
        ("qwen2_moe", "Qwen2MoeForTokenClassification"),
        ("rembert", "RemBertForTokenClassification"),
        ("roberta", "RobertaForTokenClassification"),
        ("roberta-prelayernorm", "RobertaPreLayerNormForTokenClassification"),
        ("roc_bert", "RoCBertForTokenClassification"),
        ("roformer", "RoFormerForTokenClassification"),
        ("squeezebert", "SqueezeBertForTokenClassification"),
        ("stablelm", "StableLmForTokenClassification"),
        ("starcoder2", "Starcoder2ForTokenClassification"),
        ("t5", "T5ForTokenClassification"),
        ("umt5", "UMT5ForTokenClassification"),
        ("xlm", "XLMForTokenClassification"),
        ("xlm-roberta", "XLMRobertaForTokenClassification"),
        ("xlm-roberta-xl", "XLMRobertaXLForTokenClassification"),
        ("xlnet", "XLNetForTokenClassification"),
        ("xmod", "XmodForTokenClassification"),
        ("yoso", "YosoForTokenClassification"),
    ]
)

MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES = OrderedDict(
    [
        # Model for Multiple Choice mapping
        ("albert", "AlbertForMultipleChoice"),
        ("bert", "BertForMultipleChoice"),
        ("big_bird", "BigBirdForMultipleChoice"),
        ("camembert", "CamembertForMultipleChoice"),
        ("canine", "CanineForMultipleChoice"),
        ("convbert", "ConvBertForMultipleChoice"),
        ("data2vec-text", "Data2VecTextForMultipleChoice"),
        ("deberta-v2", "DebertaV2ForMultipleChoice"),
        ("distilbert", "DistilBertForMultipleChoice"),
        ("electra", "ElectraForMultipleChoice"),
        ("ernie", "ErnieForMultipleChoice"),
        ("ernie_m", "ErnieMForMultipleChoice"),
        ("flaubert", "FlaubertForMultipleChoice"),
        ("fnet", "FNetForMultipleChoice"),
        ("funnel", "FunnelForMultipleChoice"),
        ("ibert", "IBertForMultipleChoice"),
        ("longformer", "LongformerForMultipleChoice"),
        ("luke", "LukeForMultipleChoice"),
        ("mega", "MegaForMultipleChoice"),
        ("megatron-bert", "MegatronBertForMultipleChoice"),
        ("mobilebert", "MobileBertForMultipleChoice"),
        ("mpnet", "MPNetForMultipleChoice"),
        ("mra", "MraForMultipleChoice"),
        ("nezha", "NezhaForMultipleChoice"),
        ("nystromformer", "NystromformerForMultipleChoice"),
        ("qdqbert", "QDQBertForMultipleChoice"),
        ("rembert", "RemBertForMultipleChoice"),
        ("roberta", "RobertaForMultipleChoice"),
        ("roberta-prelayernorm", "RobertaPreLayerNormForMultipleChoice"),
        ("roc_bert", "RoCBertForMultipleChoice"),
        ("roformer", "RoFormerForMultipleChoice"),
        ("squeezebert", "SqueezeBertForMultipleChoice"),
        ("xlm", "XLMForMultipleChoice"),
        ("xlm-roberta", "XLMRobertaForMultipleChoice"),
        ("xlm-roberta-xl", "XLMRobertaXLForMultipleChoice"),
        ("xlnet", "XLNetForMultipleChoice"),
        ("xmod", "XmodForMultipleChoice"),
        ("yoso", "YosoForMultipleChoice"),
    ]
)

MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING_NAMES = OrderedDict(
    [
        ("bert", "BertForNextSentencePrediction"),
        ("ernie", "ErnieForNextSentencePrediction"),
        ("fnet", "FNetForNextSentencePrediction"),
        ("megatron-bert", "MegatronBertForNextSentencePrediction"),
        ("mobilebert", "MobileBertForNextSentencePrediction"),
        ("nezha", "NezhaForNextSentencePrediction"),
        ("qdqbert", "QDQBertForNextSentencePrediction"),
    ]
)

MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Audio Classification mapping
        ("audio-spectrogram-transformer", "ASTForAudioClassification"),
        ("data2vec-audio", "Data2VecAudioForSequenceClassification"),
        ("hubert", "HubertForSequenceClassification"),
        ("sew", "SEWForSequenceClassification"),
        ("sew-d", "SEWDForSequenceClassification"),
        ("unispeech", "UniSpeechForSequenceClassification"),
        ("unispeech-sat", "UniSpeechSatForSequenceClassification"),
        ("wav2vec2", "Wav2Vec2ForSequenceClassification"),
        ("wav2vec2-bert", "Wav2Vec2BertForSequenceClassification"),
        ("wav2vec2-conformer", "Wav2Vec2ConformerForSequenceClassification"),
        ("wavlm", "WavLMForSequenceClassification"),
        ("whisper", "WhisperForAudioClassification"),
    ]
)

MODEL_FOR_CTC_MAPPING_NAMES = OrderedDict(
    [
        # Model for Connectionist temporal classification (CTC) mapping
        ("data2vec-audio", "Data2VecAudioForCTC"),
        ("hubert", "HubertForCTC"),
        ("mctct", "MCTCTForCTC"),
        ("sew", "SEWForCTC"),
        ("sew-d", "SEWDForCTC"),
        ("unispeech", "UniSpeechForCTC"),
        ("unispeech-sat", "UniSpeechSatForCTC"),
        ("wav2vec2", "Wav2Vec2ForCTC"),
        ("wav2vec2-bert", "Wav2Vec2BertForCTC"),
        ("wav2vec2-conformer", "Wav2Vec2ConformerForCTC"),
        ("wavlm", "WavLMForCTC"),
    ]
)

MODEL_FOR_AUDIO_FRAME_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Audio Classification mapping
        ("data2vec-audio", "Data2VecAudioForAudioFrameClassification"),
        ("unispeech-sat", "UniSpeechSatForAudioFrameClassification"),
        ("wav2vec2", "Wav2Vec2ForAudioFrameClassification"),
        ("wav2vec2-bert", "Wav2Vec2BertForAudioFrameClassification"),
        ("wav2vec2-conformer", "Wav2Vec2ConformerForAudioFrameClassification"),
        ("wavlm", "WavLMForAudioFrameClassification"),
    ]
)

MODEL_FOR_AUDIO_XVECTOR_MAPPING_NAMES = OrderedDict(
    [
        # Model for Audio Classification mapping
        ("data2vec-audio", "Data2VecAudioForXVector"),
        ("unispeech-sat", "UniSpeechSatForXVector"),
        ("wav2vec2", "Wav2Vec2ForXVector"),
        ("wav2vec2-bert", "Wav2Vec2BertForXVector"),
        ("wav2vec2-conformer", "Wav2Vec2ConformerForXVector"),
        ("wavlm", "WavLMForXVector"),
    ]
)

MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING_NAMES = OrderedDict(
    [
        # Model for Text-To-Spectrogram mapping
        ("fastspeech2_conformer", "FastSpeech2ConformerModel"),
        ("speecht5", "SpeechT5ForTextToSpeech"),
    ]
)

MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING_NAMES = OrderedDict(
    [
        # Model for Text-To-Waveform mapping
        ("bark", "BarkModel"),
        ("fastspeech2_conformer", "FastSpeech2ConformerWithHifiGan"),
        ("musicgen", "MusicgenForConditionalGeneration"),
        ("musicgen_melody", "MusicgenMelodyForConditionalGeneration"),
        ("seamless_m4t", "SeamlessM4TForTextToSpeech"),
        ("seamless_m4t_v2", "SeamlessM4Tv2ForTextToSpeech"),
        ("vits", "VitsModel"),
    ]
)

MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Zero Shot Image Classification mapping
        ("align", "AlignModel"),
        ("altclip", "AltCLIPModel"),
        ("blip", "BlipModel"),
        ("blip-2", "Blip2ForImageTextRetrieval"),
        ("chinese_clip", "ChineseCLIPModel"),
        ("clip", "CLIPModel"),
        ("clipseg", "CLIPSegModel"),
        ("siglip", "SiglipModel"),
    ]
)

MODEL_FOR_BACKBONE_MAPPING_NAMES = OrderedDict(
    [
        # Backbone mapping
        ("beit", "BeitBackbone"),
        ("bit", "BitBackbone"),
        ("convnext", "ConvNextBackbone"),
        ("convnextv2", "ConvNextV2Backbone"),
        ("dinat", "DinatBackbone"),
        ("dinov2", "Dinov2Backbone"),
        ("dinov2_with_registers", "Dinov2WithRegistersBackbone"),
        ("focalnet", "FocalNetBackbone"),
        ("hiera", "HieraBackbone"),
        ("maskformer-swin", "MaskFormerSwinBackbone"),
        ("nat", "NatBackbone"),
        ("pvt_v2", "PvtV2Backbone"),
        ("resnet", "ResNetBackbone"),
        ("rt_detr_resnet", "RTDetrResNetBackbone"),
        ("swin", "SwinBackbone"),
        ("swinv2", "Swinv2Backbone"),
        ("textnet", "TextNetBackbone"),
        ("timm_backbone", "TimmBackbone"),
        ("vitdet", "VitDetBackbone"),
        ("vitpose_backbone", "VitPoseBackbone"),
    ]
)

MODEL_FOR_MASK_GENERATION_MAPPING_NAMES = OrderedDict(
    [
        ("sam", "SamModel"),
    ]
)


MODEL_FOR_KEYPOINT_DETECTION_MAPPING_NAMES = OrderedDict(
    [
        ("superpoint", "SuperPointForKeypointDetection"),
    ]
)


MODEL_FOR_TEXT_ENCODING_MAPPING_NAMES = OrderedDict(
    [
        ("albert", "AlbertModel"),
        ("bert", "BertModel"),
        ("big_bird", "BigBirdModel"),
        ("clip_text_model", "CLIPTextModel"),
        ("data2vec-text", "Data2VecTextModel"),
        ("deberta", "DebertaModel"),
        ("deberta-v2", "DebertaV2Model"),
        ("distilbert", "DistilBertModel"),
        ("electra", "ElectraModel"),
        ("emu3", "Emu3TextModel"),
        ("flaubert", "FlaubertModel"),
        ("ibert", "IBertModel"),
        ("longformer", "LongformerModel"),
        ("mllama", "MllamaTextModel"),
        ("mobilebert", "MobileBertModel"),
        ("mt5", "MT5EncoderModel"),
        ("nystromformer", "NystromformerModel"),
        ("reformer", "ReformerModel"),
        ("rembert", "RemBertModel"),
        ("roberta", "RobertaModel"),
        ("roberta-prelayernorm", "RobertaPreLayerNormModel"),
        ("roc_bert", "RoCBertModel"),
        ("roformer", "RoFormerModel"),
        ("squeezebert", "SqueezeBertModel"),
        ("t5", "T5EncoderModel"),
        ("umt5", "UMT5EncoderModel"),
        ("xlm", "XLMModel"),
        ("xlm-roberta", "XLMRobertaModel"),
        ("xlm-roberta-xl", "XLMRobertaXLModel"),
    ]
)

MODEL_FOR_TIME_SERIES_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        ("patchtsmixer", "PatchTSMixerForTimeSeriesClassification"),
        ("patchtst", "PatchTSTForClassification"),
    ]
)

MODEL_FOR_TIME_SERIES_REGRESSION_MAPPING_NAMES = OrderedDict(
    [
        ("patchtsmixer", "PatchTSMixerForRegression"),
        ("patchtst", "PatchTSTForRegression"),
    ]
)

MODEL_FOR_IMAGE_TO_IMAGE_MAPPING_NAMES = OrderedDict(
    [
        ("swin2sr", "Swin2SRForImageSuperResolution"),
    ]
)

MODEL_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_MAPPING_NAMES)
MODEL_FOR_PRETRAINING_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_PRETRAINING_MAPPING_NAMES)
MODEL_WITH_LM_HEAD_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_WITH_LM_HEAD_MAPPING_NAMES)
MODEL_FOR_CAUSAL_LM_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_CAUSAL_LM_MAPPING_NAMES)
MODEL_FOR_CAUSAL_IMAGE_MODELING_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_CAUSAL_IMAGE_MODELING_MAPPING_NAMES
)
MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES
)
MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING_NAMES
)
MODEL_FOR_IMAGE_SEGMENTATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_IMAGE_SEGMENTATION_MAPPING_NAMES
)
MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING_NAMES
)
MODEL_FOR_INSTANCE_SEGMENTATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_INSTANCE_SEGMENTATION_MAPPING_NAMES
)
MODEL_FOR_UNIVERSAL_SEGMENTATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_UNIVERSAL_SEGMENTATION_MAPPING_NAMES
)
MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING_NAMES
)
MODEL_FOR_VISION_2_SEQ_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES)
MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES
)
MODEL_FOR_RETRIEVAL_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_RETRIEVAL_MAPPING_NAMES)
MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING_NAMES
)
MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES
)
MODEL_FOR_MASKED_LM_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_MASKED_LM_MAPPING_NAMES)
MODEL_FOR_IMAGE_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_IMAGE_MAPPING_NAMES)
MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING_NAMES
)
MODEL_FOR_OBJECT_DETECTION_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_OBJECT_DETECTION_MAPPING_NAMES)
MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING_NAMES
)
MODEL_FOR_DEPTH_ESTIMATION_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_DEPTH_ESTIMATION_MAPPING_NAMES)
MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES
)
MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES
)
MODEL_FOR_QUESTION_ANSWERING_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES
)
MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING_NAMES
)
MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES
)
MODEL_FOR_MULTIPLE_CHOICE_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES)
MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING_NAMES
)
MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES
)
MODEL_FOR_CTC_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_CTC_MAPPING_NAMES)
MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES)
MODEL_FOR_AUDIO_FRAME_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_AUDIO_FRAME_CLASSIFICATION_MAPPING_NAMES
)
MODEL_FOR_AUDIO_XVECTOR_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_AUDIO_XVECTOR_MAPPING_NAMES)

MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING_NAMES
)

MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING_NAMES)

MODEL_FOR_BACKBONE_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_BACKBONE_MAPPING_NAMES)

MODEL_FOR_MASK_GENERATION_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_MASK_GENERATION_MAPPING_NAMES)

MODEL_FOR_KEYPOINT_DETECTION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_KEYPOINT_DETECTION_MAPPING_NAMES
)

MODEL_FOR_TEXT_ENCODING_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_TEXT_ENCODING_MAPPING_NAMES)

MODEL_FOR_TIME_SERIES_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_TIME_SERIES_CLASSIFICATION_MAPPING_NAMES
)

MODEL_FOR_TIME_SERIES_REGRESSION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_TIME_SERIES_REGRESSION_MAPPING_NAMES
)

MODEL_FOR_IMAGE_TO_IMAGE_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_IMAGE_TO_IMAGE_MAPPING_NAMES)


class AutoModelForMaskGeneration(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_MASK_GENERATION_MAPPING


class AutoModelForKeypointDetection(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_KEYPOINT_DETECTION_MAPPING


class AutoModelForTextEncoding(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_TEXT_ENCODING_MAPPING


class AutoModelForImageToImage(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_IMAGE_TO_IMAGE_MAPPING


class AutoModel(_BaseAutoModelClass):
    _model_mapping = MODEL_MAPPING


AutoModel = auto_class_update(AutoModel)


class AutoModelForPreTraining(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_PRETRAINING_MAPPING


AutoModelForPreTraining = auto_class_update(AutoModelForPreTraining, head_doc="pretraining")


# Private on purpose, the public class will add the deprecation warnings.
class _AutoModelWithLMHead(_BaseAutoModelClass):
    _model_mapping = MODEL_WITH_LM_HEAD_MAPPING


_AutoModelWithLMHead = auto_class_update(_AutoModelWithLMHead, head_doc="language modeling")


class AutoModelForCausalLM(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_CAUSAL_LM_MAPPING


AutoModelForCausalLM = auto_class_update(AutoModelForCausalLM, head_doc="causal language modeling")


class AutoModelForMaskedLM(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_MASKED_LM_MAPPING


AutoModelForMaskedLM = auto_class_update(AutoModelForMaskedLM, head_doc="masked language modeling")


class AutoModelForSeq2SeqLM(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING


AutoModelForSeq2SeqLM = auto_class_update(
    AutoModelForSeq2SeqLM,
    head_doc="sequence-to-sequence language modeling",
    checkpoint_for_example="google-t5/t5-base",
)


class AutoModelForSequenceClassification(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING


AutoModelForSequenceClassification = auto_class_update(
    AutoModelForSequenceClassification, head_doc="sequence classification"
)


class AutoModelForQuestionAnswering(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_QUESTION_ANSWERING_MAPPING


AutoModelForQuestionAnswering = auto_class_update(AutoModelForQuestionAnswering, head_doc="question answering")


class AutoModelForTableQuestionAnswering(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING


AutoModelForTableQuestionAnswering = auto_class_update(
    AutoModelForTableQuestionAnswering,
    head_doc="table question answering",
    checkpoint_for_example="google/tapas-base-finetuned-wtq",
)


class AutoModelForVisualQuestionAnswering(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING


AutoModelForVisualQuestionAnswering = auto_class_update(
    AutoModelForVisualQuestionAnswering,
    head_doc="visual question answering",
    checkpoint_for_example="dandelin/vilt-b32-finetuned-vqa",
)


class AutoModelForDocumentQuestionAnswering(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING


AutoModelForDocumentQuestionAnswering = auto_class_update(
    AutoModelForDocumentQuestionAnswering,
    head_doc="document question answering",
    checkpoint_for_example='impira/layoutlm-document-qa", revision="52e01b3',
)


class AutoModelForTokenClassification(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING


AutoModelForTokenClassification = auto_class_update(AutoModelForTokenClassification, head_doc="token classification")


class AutoModelForMultipleChoice(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_MULTIPLE_CHOICE_MAPPING


AutoModelForMultipleChoice = auto_class_update(AutoModelForMultipleChoice, head_doc="multiple choice")


class AutoModelForNextSentencePrediction(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING


AutoModelForNextSentencePrediction = auto_class_update(
    AutoModelForNextSentencePrediction, head_doc="next sentence prediction"
)


class AutoModelForImageClassification(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING


AutoModelForImageClassification = auto_class_update(AutoModelForImageClassification, head_doc="image classification")


class AutoModelForZeroShotImageClassification(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING


AutoModelForZeroShotImageClassification = auto_class_update(
    AutoModelForZeroShotImageClassification, head_doc="zero-shot image classification"
)


class AutoModelForImageSegmentation(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_IMAGE_SEGMENTATION_MAPPING


AutoModelForImageSegmentation = auto_class_update(AutoModelForImageSegmentation, head_doc="image segmentation")


class AutoModelForSemanticSegmentation(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING


AutoModelForSemanticSegmentation = auto_class_update(
    AutoModelForSemanticSegmentation, head_doc="semantic segmentation"
)


class AutoModelForUniversalSegmentation(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_UNIVERSAL_SEGMENTATION_MAPPING


AutoModelForUniversalSegmentation = auto_class_update(
    AutoModelForUniversalSegmentation, head_doc="universal image segmentation"
)


class AutoModelForInstanceSegmentation(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_INSTANCE_SEGMENTATION_MAPPING


AutoModelForInstanceSegmentation = auto_class_update(
    AutoModelForInstanceSegmentation, head_doc="instance segmentation"
)


class AutoModelForObjectDetection(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_OBJECT_DETECTION_MAPPING


AutoModelForObjectDetection = auto_class_update(AutoModelForObjectDetection, head_doc="object detection")


class AutoModelForZeroShotObjectDetection(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING


AutoModelForZeroShotObjectDetection = auto_class_update(
    AutoModelForZeroShotObjectDetection, head_doc="zero-shot object detection"
)


class AutoModelForDepthEstimation(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_DEPTH_ESTIMATION_MAPPING


AutoModelForDepthEstimation = auto_class_update(AutoModelForDepthEstimation, head_doc="depth estimation")


class AutoModelForVideoClassification(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING


AutoModelForVideoClassification = auto_class_update(AutoModelForVideoClassification, head_doc="video classification")


class AutoModelForVision2Seq(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_VISION_2_SEQ_MAPPING


AutoModelForVision2Seq = auto_class_update(AutoModelForVision2Seq, head_doc="vision-to-text modeling")


class AutoModelForImageTextToText(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING


AutoModelForImageTextToText = auto_class_update(AutoModelForImageTextToText, head_doc="image-text-to-text modeling")


class AutoModelForAudioClassification(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING


AutoModelForAudioClassification = auto_class_update(AutoModelForAudioClassification, head_doc="audio classification")


class AutoModelForCTC(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_CTC_MAPPING


AutoModelForCTC = auto_class_update(AutoModelForCTC, head_doc="connectionist temporal classification")


class AutoModelForSpeechSeq2Seq(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING


AutoModelForSpeechSeq2Seq = auto_class_update(
    AutoModelForSpeechSeq2Seq, head_doc="sequence-to-sequence speech-to-text modeling"
)


class AutoModelForAudioFrameClassification(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_AUDIO_FRAME_CLASSIFICATION_MAPPING


AutoModelForAudioFrameClassification = auto_class_update(
    AutoModelForAudioFrameClassification, head_doc="audio frame (token) classification"
)


class AutoModelForAudioXVector(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_AUDIO_XVECTOR_MAPPING


class AutoModelForTextToSpectrogram(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING


class AutoModelForTextToWaveform(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING


class AutoBackbone(_BaseAutoBackboneClass):
    _model_mapping = MODEL_FOR_BACKBONE_MAPPING


AutoModelForAudioXVector = auto_class_update(AutoModelForAudioXVector, head_doc="audio retrieval via x-vector")


class AutoModelForMaskedImageModeling(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING


AutoModelForMaskedImageModeling = auto_class_update(AutoModelForMaskedImageModeling, head_doc="masked image modeling")


class AutoModelWithLMHead(_AutoModelWithLMHead):
    @classmethod
    def from_config(cls, config):
        warnings.warn(
            "The class `AutoModelWithLMHead` is deprecated and will be removed in a future version. Please use "
            "`AutoModelForCausalLM` for causal language models, `AutoModelForMaskedLM` for masked language models and "
            "`AutoModelForSeq2SeqLM` for encoder-decoder models.",
            FutureWarning,
        )
        return super().from_config(config)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        warnings.warn(
            "The class `AutoModelWithLMHead` is deprecated and will be removed in a future version. Please use "
            "`AutoModelForCausalLM` for causal language models, `AutoModelForMaskedLM` for masked language models and "
            "`AutoModelForSeq2SeqLM` for encoder-decoder models.",
            FutureWarning,
        )
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
