# Copyright 2020 The HuggingFace Team. All rights reserved.
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

# When adding a new object to this init, remember to add it twice: once inside the `_import_structure` dictionary and
# once inside the `if TYPE_CHECKING` branch. The `TYPE_CHECKING` should have import statements as usual, but they are
# only there for type checking. The `_import_structure` is a dictionary submodule to list of object names, and is used
# to defer the actual importing for when the objects are requested. This way `import transformers` provides the names
# in the namespace without actually importing anything (and especially none of the backends).

__version__ = "4.48.0.dev0"

from typing import TYPE_CHECKING

# Check the dependencies satisfy the minimal versions required.
from . import dependency_versions_check
from .utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_bitsandbytes_available,
    is_essentia_available,
    is_flax_available,
    is_g2p_en_available,
    is_keras_nlp_available,
    is_librosa_available,
    is_pretty_midi_available,
    is_scipy_available,
    is_sentencepiece_available,
    is_speech_available,
    is_tensorflow_text_available,
    is_tf_available,
    is_timm_available,
    is_tokenizers_available,
    is_torch_available,
    is_torchaudio_available,
    is_torchvision_available,
    is_vision_available,
    logging,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Base objects, independent of any specific backend
_import_structure = {
    "agents": [
        "Agent",
        "CodeAgent",
        "HfApiEngine",
        "ManagedAgent",
        "PipelineTool",
        "ReactAgent",
        "ReactCodeAgent",
        "ReactJsonAgent",
        "Tool",
        "Toolbox",
        "ToolCollection",
        "TransformersEngine",
        "launch_gradio_demo",
        "load_tool",
        "stream_to_gradio",
        "tool",
    ],
    "audio_utils": [],
    "benchmark": [],
    "commands": [],
    "configuration_utils": ["PretrainedConfig"],
    "convert_graph_to_onnx": [],
    "convert_slow_tokenizers_checkpoints_to_fast": [],
    "convert_tf_hub_seq_to_seq_bert_to_pytorch": [],
    "data": [
        "DataProcessor",
        "InputExample",
        "InputFeatures",
        "SingleSentenceClassificationProcessor",
        "SquadExample",
        "SquadFeatures",
        "SquadV1Processor",
        "SquadV2Processor",
        "glue_compute_metrics",
        "glue_convert_examples_to_features",
        "glue_output_modes",
        "glue_processors",
        "glue_tasks_num_labels",
        "squad_convert_examples_to_features",
        "xnli_compute_metrics",
        "xnli_output_modes",
        "xnli_processors",
        "xnli_tasks_num_labels",
    ],
    "data.data_collator": [
        "DataCollator",
        "DataCollatorForLanguageModeling",
        "DataCollatorForPermutationLanguageModeling",
        "DataCollatorForSeq2Seq",
        "DataCollatorForSOP",
        "DataCollatorForTokenClassification",
        "DataCollatorForWholeWordMask",
        "DataCollatorWithFlattening",
        "DataCollatorWithPadding",
        "DefaultDataCollator",
        "default_data_collator",
    ],
    "data.metrics": [],
    "data.processors": [],
    "debug_utils": [],
    "dependency_versions_check": [],
    "dependency_versions_table": [],
    "dynamic_module_utils": [],
    "feature_extraction_sequence_utils": ["SequenceFeatureExtractor"],
    "feature_extraction_utils": ["BatchFeature", "FeatureExtractionMixin"],
    "file_utils": [],
    "generation": [
        "AsyncTextIteratorStreamer",
        "CompileConfig",
        "GenerationConfig",
        "TextIteratorStreamer",
        "TextStreamer",
        "WatermarkingConfig",
    ],
    "hf_argparser": ["HfArgumentParser"],
    "hyperparameter_search": [],
    "image_transforms": [],
    "integrations": [
        "is_clearml_available",
        "is_comet_available",
        "is_dvclive_available",
        "is_neptune_available",
        "is_optuna_available",
        "is_ray_available",
        "is_ray_tune_available",
        "is_sigopt_available",
        "is_tensorboard_available",
        "is_wandb_available",
    ],
    "loss": [],
    "modelcard": ["ModelCard"],
    # Losses
    "modeling_tf_pytorch_utils": [
        "convert_tf_weight_name_to_pt_weight_name",
        "load_pytorch_checkpoint_in_tf2_model",
        "load_pytorch_model_in_tf2_model",
        "load_pytorch_weights_in_tf2_model",
        "load_tf2_checkpoint_in_pytorch_model",
        "load_tf2_model_in_pytorch_model",
        "load_tf2_weights_in_pytorch_model",
    ],
    # Models
    "models": [],
    "models.albert": ["AlbertConfig"],
    "models.align": [
        "AlignConfig",
        "AlignProcessor",
        "AlignTextConfig",
        "AlignVisionConfig",
    ],
    "models.altclip": [
        "AltCLIPConfig",
        "AltCLIPProcessor",
        "AltCLIPTextConfig",
        "AltCLIPVisionConfig",
    ],
    "models.aria": [
        "AriaConfig",
        "AriaProcessor",
        "AriaTextConfig",
    ],
    "models.audio_spectrogram_transformer": [
        "ASTConfig",
        "ASTFeatureExtractor",
    ],
    "models.auto": [
        "CONFIG_MAPPING",
        "FEATURE_EXTRACTOR_MAPPING",
        "IMAGE_PROCESSOR_MAPPING",
        "MODEL_NAMES_MAPPING",
        "PROCESSOR_MAPPING",
        "TOKENIZER_MAPPING",
        "AutoConfig",
        "AutoFeatureExtractor",
        "AutoImageProcessor",
        "AutoProcessor",
        "AutoTokenizer",
    ],
    "models.autoformer": ["AutoformerConfig"],
    "models.bamba": ["BambaConfig"],
    "models.bark": [
        "BarkCoarseConfig",
        "BarkConfig",
        "BarkFineConfig",
        "BarkProcessor",
        "BarkSemanticConfig",
    ],
    "models.bart": ["BartConfig", "BartTokenizer"],
    "models.barthez": [],
    "models.bartpho": [],
    "models.beit": ["BeitConfig"],
    "models.bert": [
        "BasicTokenizer",
        "BertConfig",
        "BertTokenizer",
        "WordpieceTokenizer",
    ],
    "models.bert_generation": ["BertGenerationConfig"],
    "models.bert_japanese": [
        "BertJapaneseTokenizer",
        "CharacterTokenizer",
        "MecabTokenizer",
    ],
    "models.bertweet": ["BertweetTokenizer"],
    "models.big_bird": ["BigBirdConfig"],
    "models.bigbird_pegasus": ["BigBirdPegasusConfig"],
    "models.biogpt": [
        "BioGptConfig",
        "BioGptTokenizer",
    ],
    "models.bit": ["BitConfig"],
    "models.blenderbot": [
        "BlenderbotConfig",
        "BlenderbotTokenizer",
    ],
    "models.blenderbot_small": [
        "BlenderbotSmallConfig",
        "BlenderbotSmallTokenizer",
    ],
    "models.blip": [
        "BlipConfig",
        "BlipProcessor",
        "BlipTextConfig",
        "BlipVisionConfig",
    ],
    "models.blip_2": [
        "Blip2Config",
        "Blip2Processor",
        "Blip2QFormerConfig",
        "Blip2VisionConfig",
    ],
    "models.bloom": ["BloomConfig"],
    "models.bridgetower": [
        "BridgeTowerConfig",
        "BridgeTowerProcessor",
        "BridgeTowerTextConfig",
        "BridgeTowerVisionConfig",
    ],
    "models.bros": [
        "BrosConfig",
        "BrosProcessor",
    ],
    "models.byt5": ["ByT5Tokenizer"],
    "models.camembert": ["CamembertConfig"],
    "models.canine": [
        "CanineConfig",
        "CanineTokenizer",
    ],
    "models.chameleon": [
        "ChameleonConfig",
        "ChameleonProcessor",
        "ChameleonVQVAEConfig",
    ],
    "models.chinese_clip": [
        "ChineseCLIPConfig",
        "ChineseCLIPProcessor",
        "ChineseCLIPTextConfig",
        "ChineseCLIPVisionConfig",
    ],
    "models.clap": [
        "ClapAudioConfig",
        "ClapConfig",
        "ClapProcessor",
        "ClapTextConfig",
    ],
    "models.clip": [
        "CLIPConfig",
        "CLIPProcessor",
        "CLIPTextConfig",
        "CLIPTokenizer",
        "CLIPVisionConfig",
    ],
    "models.clipseg": [
        "CLIPSegConfig",
        "CLIPSegProcessor",
        "CLIPSegTextConfig",
        "CLIPSegVisionConfig",
    ],
    "models.clvp": [
        "ClvpConfig",
        "ClvpDecoderConfig",
        "ClvpEncoderConfig",
        "ClvpFeatureExtractor",
        "ClvpProcessor",
        "ClvpTokenizer",
    ],
    "models.code_llama": [],
    "models.codegen": [
        "CodeGenConfig",
        "CodeGenTokenizer",
    ],
    "models.cohere": ["CohereConfig"],
    "models.cohere2": ["Cohere2Config"],
    "models.colpali": [
        "ColPaliConfig",
        "ColPaliProcessor",
    ],
    "models.conditional_detr": ["ConditionalDetrConfig"],
    "models.convbert": [
        "ConvBertConfig",
        "ConvBertTokenizer",
    ],
    "models.convnext": ["ConvNextConfig"],
    "models.convnextv2": ["ConvNextV2Config"],
    "models.cpm": [],
    "models.cpmant": [
        "CpmAntConfig",
        "CpmAntTokenizer",
    ],
    "models.ctrl": [
        "CTRLConfig",
        "CTRLTokenizer",
    ],
    "models.cvt": ["CvtConfig"],
    "models.dac": ["DacConfig", "DacFeatureExtractor"],
    "models.data2vec": [
        "Data2VecAudioConfig",
        "Data2VecTextConfig",
        "Data2VecVisionConfig",
    ],
    "models.dbrx": ["DbrxConfig"],
    "models.deberta": [
        "DebertaConfig",
        "DebertaTokenizer",
    ],
    "models.deberta_v2": ["DebertaV2Config"],
    "models.decision_transformer": ["DecisionTransformerConfig"],
    "models.deformable_detr": ["DeformableDetrConfig"],
    "models.deit": ["DeiTConfig"],
    "models.deprecated": [],
    "models.deprecated.bort": [],
    "models.deprecated.deta": ["DetaConfig"],
    "models.deprecated.efficientformer": ["EfficientFormerConfig"],
    "models.deprecated.ernie_m": ["ErnieMConfig"],
    "models.deprecated.gptsan_japanese": [
        "GPTSanJapaneseConfig",
        "GPTSanJapaneseTokenizer",
    ],
    "models.deprecated.graphormer": ["GraphormerConfig"],
    "models.deprecated.jukebox": [
        "JukeboxConfig",
        "JukeboxPriorConfig",
        "JukeboxTokenizer",
        "JukeboxVQVAEConfig",
    ],
    "models.deprecated.mctct": [
        "MCTCTConfig",
        "MCTCTFeatureExtractor",
        "MCTCTProcessor",
    ],
    "models.deprecated.mega": ["MegaConfig"],
    "models.deprecated.mmbt": ["MMBTConfig"],
    "models.deprecated.nat": ["NatConfig"],
    "models.deprecated.nezha": ["NezhaConfig"],
    "models.deprecated.open_llama": ["OpenLlamaConfig"],
    "models.deprecated.qdqbert": ["QDQBertConfig"],
    "models.deprecated.realm": [
        "RealmConfig",
        "RealmTokenizer",
    ],
    "models.deprecated.retribert": [
        "RetriBertConfig",
        "RetriBertTokenizer",
    ],
    "models.deprecated.speech_to_text_2": [
        "Speech2Text2Config",
        "Speech2Text2Processor",
        "Speech2Text2Tokenizer",
    ],
    "models.deprecated.tapex": ["TapexTokenizer"],
    "models.deprecated.trajectory_transformer": ["TrajectoryTransformerConfig"],
    "models.deprecated.transfo_xl": [
        "TransfoXLConfig",
        "TransfoXLCorpus",
        "TransfoXLTokenizer",
    ],
    "models.deprecated.tvlt": [
        "TvltConfig",
        "TvltFeatureExtractor",
        "TvltProcessor",
    ],
    "models.deprecated.van": ["VanConfig"],
    "models.deprecated.vit_hybrid": ["ViTHybridConfig"],
    "models.deprecated.xlm_prophetnet": ["XLMProphetNetConfig"],
    "models.depth_anything": ["DepthAnythingConfig"],
    "models.detr": ["DetrConfig"],
    "models.dialogpt": [],
    "models.diffllama": ["DiffLlamaConfig"],
    "models.dinat": ["DinatConfig"],
    "models.dinov2": ["Dinov2Config"],
    "models.dinov2_with_registers": ["Dinov2WithRegistersConfig"],
    "models.distilbert": [
        "DistilBertConfig",
        "DistilBertTokenizer",
    ],
    "models.dit": [],
    "models.donut": [
        "DonutProcessor",
        "DonutSwinConfig",
    ],
    "models.dpr": [
        "DPRConfig",
        "DPRContextEncoderTokenizer",
        "DPRQuestionEncoderTokenizer",
        "DPRReaderOutput",
        "DPRReaderTokenizer",
    ],
    "models.dpt": ["DPTConfig"],
    "models.efficientnet": ["EfficientNetConfig"],
    "models.electra": [
        "ElectraConfig",
        "ElectraTokenizer",
    ],
    "models.encodec": [
        "EncodecConfig",
        "EncodecFeatureExtractor",
    ],
    "models.encoder_decoder": ["EncoderDecoderConfig"],
    "models.ernie": ["ErnieConfig"],
    "models.esm": ["EsmConfig", "EsmTokenizer"],
    "models.falcon": ["FalconConfig"],
    "models.falcon_mamba": ["FalconMambaConfig"],
    "models.fastspeech2_conformer": [
        "FastSpeech2ConformerConfig",
        "FastSpeech2ConformerHifiGanConfig",
        "FastSpeech2ConformerTokenizer",
        "FastSpeech2ConformerWithHifiGanConfig",
    ],
    "models.flaubert": ["FlaubertConfig", "FlaubertTokenizer"],
    "models.flava": [
        "FlavaConfig",
        "FlavaImageCodebookConfig",
        "FlavaImageConfig",
        "FlavaMultimodalConfig",
        "FlavaTextConfig",
    ],
    "models.fnet": ["FNetConfig"],
    "models.focalnet": ["FocalNetConfig"],
    "models.fsmt": [
        "FSMTConfig",
        "FSMTTokenizer",
    ],
    "models.funnel": [
        "FunnelConfig",
        "FunnelTokenizer",
    ],
    "models.fuyu": ["FuyuConfig"],
    "models.gemma": ["GemmaConfig"],
    "models.gemma2": ["Gemma2Config"],
    "models.git": [
        "GitConfig",
        "GitProcessor",
        "GitVisionConfig",
    ],
    "models.glm": ["GlmConfig"],
    "models.glpn": ["GLPNConfig"],
    "models.gpt2": [
        "GPT2Config",
        "GPT2Tokenizer",
    ],
    "models.gpt_bigcode": ["GPTBigCodeConfig"],
    "models.gpt_neo": ["GPTNeoConfig"],
    "models.gpt_neox": ["GPTNeoXConfig"],
    "models.gpt_neox_japanese": ["GPTNeoXJapaneseConfig"],
    "models.gpt_sw3": [],
    "models.gptj": ["GPTJConfig"],
    "models.granite": ["GraniteConfig"],
    "models.granitemoe": ["GraniteMoeConfig"],
    "models.grounding_dino": [
        "GroundingDinoConfig",
        "GroundingDinoProcessor",
    ],
    "models.groupvit": [
        "GroupViTConfig",
        "GroupViTTextConfig",
        "GroupViTVisionConfig",
    ],
    "models.herbert": ["HerbertTokenizer"],
    "models.hiera": ["HieraConfig"],
    "models.hubert": ["HubertConfig"],
    "models.ibert": ["IBertConfig"],
    "models.idefics": ["IdeficsConfig"],
    "models.idefics2": ["Idefics2Config"],
    "models.idefics3": ["Idefics3Config"],
    "models.ijepa": ["IJepaConfig"],
    "models.imagegpt": ["ImageGPTConfig"],
    "models.informer": ["InformerConfig"],
    "models.instructblip": [
        "InstructBlipConfig",
        "InstructBlipProcessor",
        "InstructBlipQFormerConfig",
        "InstructBlipVisionConfig",
    ],
    "models.instructblipvideo": [
        "InstructBlipVideoConfig",
        "InstructBlipVideoProcessor",
        "InstructBlipVideoQFormerConfig",
        "InstructBlipVideoVisionConfig",
    ],
    "models.jamba": ["JambaConfig"],
    "models.jetmoe": ["JetMoeConfig"],
    "models.kosmos2": [
        "Kosmos2Config",
        "Kosmos2Processor",
    ],
    "models.layoutlm": [
        "LayoutLMConfig",
        "LayoutLMTokenizer",
    ],
    "models.layoutlmv2": [
        "LayoutLMv2Config",
        "LayoutLMv2FeatureExtractor",
        "LayoutLMv2ImageProcessor",
        "LayoutLMv2Processor",
        "LayoutLMv2Tokenizer",
    ],
    "models.layoutlmv3": [
        "LayoutLMv3Config",
        "LayoutLMv3FeatureExtractor",
        "LayoutLMv3ImageProcessor",
        "LayoutLMv3Processor",
        "LayoutLMv3Tokenizer",
    ],
    "models.layoutxlm": ["LayoutXLMProcessor"],
    "models.led": ["LEDConfig", "LEDTokenizer"],
    "models.levit": ["LevitConfig"],
    "models.lilt": ["LiltConfig"],
    "models.llama": ["LlamaConfig"],
    "models.llava": [
        "LlavaConfig",
        "LlavaProcessor",
    ],
    "models.llava_next": [
        "LlavaNextConfig",
        "LlavaNextProcessor",
    ],
    "models.llava_next_video": [
        "LlavaNextVideoConfig",
        "LlavaNextVideoProcessor",
    ],
    "models.llava_onevision": ["LlavaOnevisionConfig", "LlavaOnevisionProcessor"],
    "models.longformer": [
        "LongformerConfig",
        "LongformerTokenizer",
    ],
    "models.longt5": ["LongT5Config"],
    "models.luke": [
        "LukeConfig",
        "LukeTokenizer",
    ],
    "models.lxmert": [
        "LxmertConfig",
        "LxmertTokenizer",
    ],
    "models.m2m_100": ["M2M100Config"],
    "models.mamba": ["MambaConfig"],
    "models.mamba2": ["Mamba2Config"],
    "models.marian": ["MarianConfig"],
    "models.markuplm": [
        "MarkupLMConfig",
        "MarkupLMFeatureExtractor",
        "MarkupLMProcessor",
        "MarkupLMTokenizer",
    ],
    "models.mask2former": ["Mask2FormerConfig"],
    "models.maskformer": [
        "MaskFormerConfig",
        "MaskFormerSwinConfig",
    ],
    "models.mbart": ["MBartConfig"],
    "models.mbart50": [],
    "models.megatron_bert": ["MegatronBertConfig"],
    "models.megatron_gpt2": [],
    "models.mgp_str": [
        "MgpstrConfig",
        "MgpstrProcessor",
        "MgpstrTokenizer",
    ],
    "models.mimi": ["MimiConfig"],
    "models.mistral": ["MistralConfig"],
    "models.mixtral": ["MixtralConfig"],
    "models.mllama": [
        "MllamaConfig",
        "MllamaProcessor",
    ],
    "models.mluke": [],
    "models.mobilebert": [
        "MobileBertConfig",
        "MobileBertTokenizer",
    ],
    "models.mobilenet_v1": ["MobileNetV1Config"],
    "models.mobilenet_v2": ["MobileNetV2Config"],
    "models.mobilevit": ["MobileViTConfig"],
    "models.mobilevitv2": ["MobileViTV2Config"],
    "models.modernbert": ["ModernBertConfig"],
    "models.moshi": [
        "MoshiConfig",
        "MoshiDepthConfig",
    ],
    "models.mpnet": [
        "MPNetConfig",
        "MPNetTokenizer",
    ],
    "models.mpt": ["MptConfig"],
    "models.mra": ["MraConfig"],
    "models.mt5": ["MT5Config"],
    "models.musicgen": [
        "MusicgenConfig",
        "MusicgenDecoderConfig",
    ],
    "models.musicgen_melody": [
        "MusicgenMelodyConfig",
        "MusicgenMelodyDecoderConfig",
    ],
    "models.mvp": ["MvpConfig", "MvpTokenizer"],
    "models.myt5": ["MyT5Tokenizer"],
    "models.nemotron": ["NemotronConfig"],
    "models.nllb": [],
    "models.nllb_moe": ["NllbMoeConfig"],
    "models.nougat": ["NougatProcessor"],
    "models.nystromformer": ["NystromformerConfig"],
    "models.olmo": ["OlmoConfig"],
    "models.olmo2": ["Olmo2Config"],
    "models.olmoe": ["OlmoeConfig"],
    "models.omdet_turbo": [
        "OmDetTurboConfig",
        "OmDetTurboProcessor",
    ],
    "models.oneformer": [
        "OneFormerConfig",
        "OneFormerProcessor",
    ],
    "models.openai": [
        "OpenAIGPTConfig",
        "OpenAIGPTTokenizer",
    ],
    "models.opt": ["OPTConfig"],
    "models.owlv2": [
        "Owlv2Config",
        "Owlv2Processor",
        "Owlv2TextConfig",
        "Owlv2VisionConfig",
    ],
    "models.owlvit": [
        "OwlViTConfig",
        "OwlViTProcessor",
        "OwlViTTextConfig",
        "OwlViTVisionConfig",
    ],
    "models.paligemma": ["PaliGemmaConfig"],
    "models.patchtsmixer": ["PatchTSMixerConfig"],
    "models.patchtst": ["PatchTSTConfig"],
    "models.pegasus": [
        "PegasusConfig",
        "PegasusTokenizer",
    ],
    "models.pegasus_x": ["PegasusXConfig"],
    "models.perceiver": [
        "PerceiverConfig",
        "PerceiverTokenizer",
    ],
    "models.persimmon": ["PersimmonConfig"],
    "models.phi": ["PhiConfig"],
    "models.phi3": ["Phi3Config"],
    "models.phimoe": ["PhimoeConfig"],
    "models.phobert": ["PhobertTokenizer"],
    "models.pix2struct": [
        "Pix2StructConfig",
        "Pix2StructProcessor",
        "Pix2StructTextConfig",
        "Pix2StructVisionConfig",
    ],
    "models.pixtral": ["PixtralProcessor", "PixtralVisionConfig"],
    "models.plbart": ["PLBartConfig"],
    "models.poolformer": ["PoolFormerConfig"],
    "models.pop2piano": ["Pop2PianoConfig"],
    "models.prophetnet": [
        "ProphetNetConfig",
        "ProphetNetTokenizer",
    ],
    "models.pvt": ["PvtConfig"],
    "models.pvt_v2": ["PvtV2Config"],
    "models.qwen2": [
        "Qwen2Config",
        "Qwen2Tokenizer",
    ],
    "models.qwen2_audio": [
        "Qwen2AudioConfig",
        "Qwen2AudioEncoderConfig",
        "Qwen2AudioProcessor",
    ],
    "models.qwen2_moe": ["Qwen2MoeConfig"],
    "models.qwen2_vl": [
        "Qwen2VLConfig",
        "Qwen2VLProcessor",
    ],
    "models.rag": ["RagConfig", "RagRetriever", "RagTokenizer"],
    "models.recurrent_gemma": ["RecurrentGemmaConfig"],
    "models.reformer": ["ReformerConfig"],
    "models.regnet": ["RegNetConfig"],
    "models.rembert": ["RemBertConfig"],
    "models.resnet": ["ResNetConfig"],
    "models.roberta": [
        "RobertaConfig",
        "RobertaTokenizer",
    ],
    "models.roberta_prelayernorm": ["RobertaPreLayerNormConfig"],
    "models.roc_bert": [
        "RoCBertConfig",
        "RoCBertTokenizer",
    ],
    "models.roformer": [
        "RoFormerConfig",
        "RoFormerTokenizer",
    ],
    "models.rt_detr": ["RTDetrConfig", "RTDetrResNetConfig"],
    "models.rwkv": ["RwkvConfig"],
    "models.sam": [
        "SamConfig",
        "SamMaskDecoderConfig",
        "SamProcessor",
        "SamPromptEncoderConfig",
        "SamVisionConfig",
    ],
    "models.seamless_m4t": [
        "SeamlessM4TConfig",
        "SeamlessM4TFeatureExtractor",
        "SeamlessM4TProcessor",
    ],
    "models.seamless_m4t_v2": ["SeamlessM4Tv2Config"],
    "models.segformer": ["SegformerConfig"],
    "models.seggpt": ["SegGptConfig"],
    "models.sew": ["SEWConfig"],
    "models.sew_d": ["SEWDConfig"],
    "models.siglip": [
        "SiglipConfig",
        "SiglipProcessor",
        "SiglipTextConfig",
        "SiglipVisionConfig",
    ],
    "models.speech_encoder_decoder": ["SpeechEncoderDecoderConfig"],
    "models.speech_to_text": [
        "Speech2TextConfig",
        "Speech2TextFeatureExtractor",
        "Speech2TextProcessor",
    ],
    "models.speecht5": [
        "SpeechT5Config",
        "SpeechT5FeatureExtractor",
        "SpeechT5HifiGanConfig",
        "SpeechT5Processor",
    ],
    "models.splinter": [
        "SplinterConfig",
        "SplinterTokenizer",
    ],
    "models.squeezebert": [
        "SqueezeBertConfig",
        "SqueezeBertTokenizer",
    ],
    "models.stablelm": ["StableLmConfig"],
    "models.starcoder2": ["Starcoder2Config"],
    "models.superpoint": ["SuperPointConfig"],
    "models.swiftformer": ["SwiftFormerConfig"],
    "models.swin": ["SwinConfig"],
    "models.swin2sr": ["Swin2SRConfig"],
    "models.swinv2": ["Swinv2Config"],
    "models.switch_transformers": ["SwitchTransformersConfig"],
    "models.t5": ["T5Config"],
    "models.table_transformer": ["TableTransformerConfig"],
    "models.tapas": [
        "TapasConfig",
        "TapasTokenizer",
    ],
    "models.time_series_transformer": ["TimeSeriesTransformerConfig"],
    "models.timesformer": ["TimesformerConfig"],
    "models.timm_backbone": ["TimmBackboneConfig"],
    "models.timm_wrapper": ["TimmWrapperConfig"],
    "models.trocr": [
        "TrOCRConfig",
        "TrOCRProcessor",
    ],
    "models.tvp": [
        "TvpConfig",
        "TvpProcessor",
    ],
    "models.udop": [
        "UdopConfig",
        "UdopProcessor",
    ],
    "models.umt5": ["UMT5Config"],
    "models.unispeech": ["UniSpeechConfig"],
    "models.unispeech_sat": ["UniSpeechSatConfig"],
    "models.univnet": [
        "UnivNetConfig",
        "UnivNetFeatureExtractor",
    ],
    "models.upernet": ["UperNetConfig"],
    "models.video_llava": ["VideoLlavaConfig"],
    "models.videomae": ["VideoMAEConfig"],
    "models.vilt": [
        "ViltConfig",
        "ViltFeatureExtractor",
        "ViltImageProcessor",
        "ViltProcessor",
    ],
    "models.vipllava": ["VipLlavaConfig"],
    "models.vision_encoder_decoder": ["VisionEncoderDecoderConfig"],
    "models.vision_text_dual_encoder": [
        "VisionTextDualEncoderConfig",
        "VisionTextDualEncoderProcessor",
    ],
    "models.visual_bert": ["VisualBertConfig"],
    "models.vit": ["ViTConfig"],
    "models.vit_mae": ["ViTMAEConfig"],
    "models.vit_msn": ["ViTMSNConfig"],
    "models.vitdet": ["VitDetConfig"],
    "models.vitmatte": ["VitMatteConfig"],
    "models.vits": [
        "VitsConfig",
        "VitsTokenizer",
    ],
    "models.vivit": ["VivitConfig"],
    "models.wav2vec2": [
        "Wav2Vec2Config",
        "Wav2Vec2CTCTokenizer",
        "Wav2Vec2FeatureExtractor",
        "Wav2Vec2Processor",
        "Wav2Vec2Tokenizer",
    ],
    "models.wav2vec2_bert": [
        "Wav2Vec2BertConfig",
        "Wav2Vec2BertProcessor",
    ],
    "models.wav2vec2_conformer": ["Wav2Vec2ConformerConfig"],
    "models.wav2vec2_phoneme": ["Wav2Vec2PhonemeCTCTokenizer"],
    "models.wav2vec2_with_lm": ["Wav2Vec2ProcessorWithLM"],
    "models.wavlm": ["WavLMConfig"],
    "models.whisper": [
        "WhisperConfig",
        "WhisperFeatureExtractor",
        "WhisperProcessor",
        "WhisperTokenizer",
    ],
    "models.x_clip": [
        "XCLIPConfig",
        "XCLIPProcessor",
        "XCLIPTextConfig",
        "XCLIPVisionConfig",
    ],
    "models.xglm": ["XGLMConfig"],
    "models.xlm": ["XLMConfig", "XLMTokenizer"],
    "models.xlm_roberta": ["XLMRobertaConfig"],
    "models.xlm_roberta_xl": ["XLMRobertaXLConfig"],
    "models.xlnet": ["XLNetConfig"],
    "models.xmod": ["XmodConfig"],
    "models.yolos": ["YolosConfig"],
    "models.yoso": ["YosoConfig"],
    "models.zamba": ["ZambaConfig"],
    "models.zoedepth": ["ZoeDepthConfig"],
    "onnx": [],
    "pipelines": [
        "AudioClassificationPipeline",
        "AutomaticSpeechRecognitionPipeline",
        "CsvPipelineDataFormat",
        "DepthEstimationPipeline",
        "DocumentQuestionAnsweringPipeline",
        "FeatureExtractionPipeline",
        "FillMaskPipeline",
        "ImageClassificationPipeline",
        "ImageFeatureExtractionPipeline",
        "ImageSegmentationPipeline",
        "ImageTextToTextPipeline",
        "ImageToImagePipeline",
        "ImageToTextPipeline",
        "JsonPipelineDataFormat",
        "MaskGenerationPipeline",
        "NerPipeline",
        "ObjectDetectionPipeline",
        "PipedPipelineDataFormat",
        "Pipeline",
        "PipelineDataFormat",
        "QuestionAnsweringPipeline",
        "SummarizationPipeline",
        "TableQuestionAnsweringPipeline",
        "Text2TextGenerationPipeline",
        "TextClassificationPipeline",
        "TextGenerationPipeline",
        "TextToAudioPipeline",
        "TokenClassificationPipeline",
        "TranslationPipeline",
        "VideoClassificationPipeline",
        "VisualQuestionAnsweringPipeline",
        "ZeroShotAudioClassificationPipeline",
        "ZeroShotClassificationPipeline",
        "ZeroShotImageClassificationPipeline",
        "ZeroShotObjectDetectionPipeline",
        "pipeline",
    ],
    "processing_utils": ["ProcessorMixin"],
    "quantizers": [],
    "testing_utils": [],
    "tokenization_utils": ["PreTrainedTokenizer"],
    "tokenization_utils_base": [
        "AddedToken",
        "BatchEncoding",
        "CharSpan",
        "PreTrainedTokenizerBase",
        "SpecialTokensMixin",
        "TokenSpan",
    ],
    "trainer_callback": [
        "DefaultFlowCallback",
        "EarlyStoppingCallback",
        "PrinterCallback",
        "ProgressCallback",
        "TrainerCallback",
        "TrainerControl",
        "TrainerState",
    ],
    "trainer_utils": [
        "EvalPrediction",
        "IntervalStrategy",
        "SchedulerType",
        "enable_full_determinism",
        "set_seed",
    ],
    "training_args": ["TrainingArguments"],
    "training_args_seq2seq": ["Seq2SeqTrainingArguments"],
    "training_args_tf": ["TFTrainingArguments"],
    "utils": [
        "CONFIG_NAME",
        "MODEL_CARD_NAME",
        "PYTORCH_PRETRAINED_BERT_CACHE",
        "PYTORCH_TRANSFORMERS_CACHE",
        "SPIECE_UNDERLINE",
        "TF2_WEIGHTS_NAME",
        "TF_WEIGHTS_NAME",
        "TRANSFORMERS_CACHE",
        "WEIGHTS_NAME",
        "TensorType",
        "add_end_docstrings",
        "add_start_docstrings",
        "is_apex_available",
        "is_av_available",
        "is_bitsandbytes_available",
        "is_datasets_available",
        "is_faiss_available",
        "is_flax_available",
        "is_keras_nlp_available",
        "is_phonemizer_available",
        "is_psutil_available",
        "is_py3nvml_available",
        "is_pyctcdecode_available",
        "is_sacremoses_available",
        "is_safetensors_available",
        "is_scipy_available",
        "is_sentencepiece_available",
        "is_sklearn_available",
        "is_speech_available",
        "is_tensorflow_text_available",
        "is_tf_available",
        "is_timm_available",
        "is_tokenizers_available",
        "is_torch_available",
        "is_torch_mlu_available",
        "is_torch_musa_available",
        "is_torch_neuroncore_available",
        "is_torch_npu_available",
        "is_torch_tpu_available",
        "is_torchvision_available",
        "is_torch_xla_available",
        "is_torch_xpu_available",
        "is_vision_available",
        "logging",
    ],
    "utils.quantization_config": [
        "AqlmConfig",
        "AwqConfig",
        "BitNetConfig",
        "BitsAndBytesConfig",
        "CompressedTensorsConfig",
        "EetqConfig",
        "FbgemmFp8Config",
        "GPTQConfig",
        "HiggsConfig",
        "HqqConfig",
        "QuantoConfig",
        "TorchAoConfig",
        "VptqConfig",
    ],
}

# sentencepiece-backed objects
try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_sentencepiece_objects

    _import_structure["utils.dummy_sentencepiece_objects"] = [
        name for name in dir(dummy_sentencepiece_objects) if not name.startswith("_")
    ]
else:
    _import_structure["models.albert"].append("AlbertTokenizer")
    _import_structure["models.barthez"].append("BarthezTokenizer")
    _import_structure["models.bartpho"].append("BartphoTokenizer")
    _import_structure["models.bert_generation"].append("BertGenerationTokenizer")
    _import_structure["models.big_bird"].append("BigBirdTokenizer")
    _import_structure["models.camembert"].append("CamembertTokenizer")
    _import_structure["models.code_llama"].append("CodeLlamaTokenizer")
    _import_structure["models.cpm"].append("CpmTokenizer")
    _import_structure["models.deberta_v2"].append("DebertaV2Tokenizer")
    _import_structure["models.deprecated.ernie_m"].append("ErnieMTokenizer")
    _import_structure["models.deprecated.xlm_prophetnet"].append("XLMProphetNetTokenizer")
    _import_structure["models.fnet"].append("FNetTokenizer")
    _import_structure["models.gemma"].append("GemmaTokenizer")
    _import_structure["models.gpt_sw3"].append("GPTSw3Tokenizer")
    _import_structure["models.layoutxlm"].append("LayoutXLMTokenizer")
    _import_structure["models.llama"].append("LlamaTokenizer")
    _import_structure["models.m2m_100"].append("M2M100Tokenizer")
    _import_structure["models.marian"].append("MarianTokenizer")
    _import_structure["models.mbart"].append("MBartTokenizer")
    _import_structure["models.mbart50"].append("MBart50Tokenizer")
    _import_structure["models.mluke"].append("MLukeTokenizer")
    _import_structure["models.mt5"].append("MT5Tokenizer")
    _import_structure["models.nllb"].append("NllbTokenizer")
    _import_structure["models.pegasus"].append("PegasusTokenizer")
    _import_structure["models.plbart"].append("PLBartTokenizer")
    _import_structure["models.reformer"].append("ReformerTokenizer")
    _import_structure["models.rembert"].append("RemBertTokenizer")
    _import_structure["models.seamless_m4t"].append("SeamlessM4TTokenizer")
    _import_structure["models.siglip"].append("SiglipTokenizer")
    _import_structure["models.speech_to_text"].append("Speech2TextTokenizer")
    _import_structure["models.speecht5"].append("SpeechT5Tokenizer")
    _import_structure["models.t5"].append("T5Tokenizer")
    _import_structure["models.udop"].append("UdopTokenizer")
    _import_structure["models.xglm"].append("XGLMTokenizer")
    _import_structure["models.xlm_roberta"].append("XLMRobertaTokenizer")
    _import_structure["models.xlnet"].append("XLNetTokenizer")

# tokenizers-backed objects
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_tokenizers_objects

    _import_structure["utils.dummy_tokenizers_objects"] = [
        name for name in dir(dummy_tokenizers_objects) if not name.startswith("_")
    ]
else:
    # Fast tokenizers structure
    _import_structure["models.albert"].append("AlbertTokenizerFast")
    _import_structure["models.bart"].append("BartTokenizerFast")
    _import_structure["models.barthez"].append("BarthezTokenizerFast")
    _import_structure["models.bert"].append("BertTokenizerFast")
    _import_structure["models.big_bird"].append("BigBirdTokenizerFast")
    _import_structure["models.blenderbot"].append("BlenderbotTokenizerFast")
    _import_structure["models.blenderbot_small"].append("BlenderbotSmallTokenizerFast")
    _import_structure["models.bloom"].append("BloomTokenizerFast")
    _import_structure["models.camembert"].append("CamembertTokenizerFast")
    _import_structure["models.clip"].append("CLIPTokenizerFast")
    _import_structure["models.code_llama"].append("CodeLlamaTokenizerFast")
    _import_structure["models.codegen"].append("CodeGenTokenizerFast")
    _import_structure["models.cohere"].append("CohereTokenizerFast")
    _import_structure["models.convbert"].append("ConvBertTokenizerFast")
    _import_structure["models.cpm"].append("CpmTokenizerFast")
    _import_structure["models.deberta"].append("DebertaTokenizerFast")
    _import_structure["models.deberta_v2"].append("DebertaV2TokenizerFast")
    _import_structure["models.deprecated.realm"].append("RealmTokenizerFast")
    _import_structure["models.deprecated.retribert"].append("RetriBertTokenizerFast")
    _import_structure["models.distilbert"].append("DistilBertTokenizerFast")
    _import_structure["models.dpr"].extend(
        [
            "DPRContextEncoderTokenizerFast",
            "DPRQuestionEncoderTokenizerFast",
            "DPRReaderTokenizerFast",
        ]
    )
    _import_structure["models.electra"].append("ElectraTokenizerFast")
    _import_structure["models.fnet"].append("FNetTokenizerFast")
    _import_structure["models.funnel"].append("FunnelTokenizerFast")
    _import_structure["models.gemma"].append("GemmaTokenizerFast")
    _import_structure["models.gpt2"].append("GPT2TokenizerFast")
    _import_structure["models.gpt_neox"].append("GPTNeoXTokenizerFast")
    _import_structure["models.gpt_neox_japanese"].append("GPTNeoXJapaneseTokenizer")
    _import_structure["models.herbert"].append("HerbertTokenizerFast")
    _import_structure["models.layoutlm"].append("LayoutLMTokenizerFast")
    _import_structure["models.layoutlmv2"].append("LayoutLMv2TokenizerFast")
    _import_structure["models.layoutlmv3"].append("LayoutLMv3TokenizerFast")
    _import_structure["models.layoutxlm"].append("LayoutXLMTokenizerFast")
    _import_structure["models.led"].append("LEDTokenizerFast")
    _import_structure["models.llama"].append("LlamaTokenizerFast")
    _import_structure["models.longformer"].append("LongformerTokenizerFast")
    _import_structure["models.lxmert"].append("LxmertTokenizerFast")
    _import_structure["models.markuplm"].append("MarkupLMTokenizerFast")
    _import_structure["models.mbart"].append("MBartTokenizerFast")
    _import_structure["models.mbart50"].append("MBart50TokenizerFast")
    _import_structure["models.mobilebert"].append("MobileBertTokenizerFast")
    _import_structure["models.mpnet"].append("MPNetTokenizerFast")
    _import_structure["models.mt5"].append("MT5TokenizerFast")
    _import_structure["models.mvp"].append("MvpTokenizerFast")
    _import_structure["models.nllb"].append("NllbTokenizerFast")
    _import_structure["models.nougat"].append("NougatTokenizerFast")
    _import_structure["models.openai"].append("OpenAIGPTTokenizerFast")
    _import_structure["models.pegasus"].append("PegasusTokenizerFast")
    _import_structure["models.qwen2"].append("Qwen2TokenizerFast")
    _import_structure["models.reformer"].append("ReformerTokenizerFast")
    _import_structure["models.rembert"].append("RemBertTokenizerFast")
    _import_structure["models.roberta"].append("RobertaTokenizerFast")
    _import_structure["models.roformer"].append("RoFormerTokenizerFast")
    _import_structure["models.seamless_m4t"].append("SeamlessM4TTokenizerFast")
    _import_structure["models.splinter"].append("SplinterTokenizerFast")
    _import_structure["models.squeezebert"].append("SqueezeBertTokenizerFast")
    _import_structure["models.t5"].append("T5TokenizerFast")
    _import_structure["models.udop"].append("UdopTokenizerFast")
    _import_structure["models.whisper"].append("WhisperTokenizerFast")
    _import_structure["models.xglm"].append("XGLMTokenizerFast")
    _import_structure["models.xlm_roberta"].append("XLMRobertaTokenizerFast")
    _import_structure["models.xlnet"].append("XLNetTokenizerFast")
    _import_structure["tokenization_utils_fast"] = ["PreTrainedTokenizerFast"]


try:
    if not (is_sentencepiece_available() and is_tokenizers_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_sentencepiece_and_tokenizers_objects

    _import_structure["utils.dummy_sentencepiece_and_tokenizers_objects"] = [
        name for name in dir(dummy_sentencepiece_and_tokenizers_objects) if not name.startswith("_")
    ]
else:
    _import_structure["convert_slow_tokenizer"] = [
        "SLOW_TO_FAST_CONVERTERS",
        "convert_slow_tokenizer",
    ]

# Tensorflow-text-specific objects
try:
    if not is_tensorflow_text_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_tensorflow_text_objects

    _import_structure["utils.dummy_tensorflow_text_objects"] = [
        name for name in dir(dummy_tensorflow_text_objects) if not name.startswith("_")
    ]
else:
    _import_structure["models.bert"].append("TFBertTokenizer")

# keras-nlp-specific objects
try:
    if not is_keras_nlp_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_keras_nlp_objects

    _import_structure["utils.dummy_keras_nlp_objects"] = [
        name for name in dir(dummy_keras_nlp_objects) if not name.startswith("_")
    ]
else:
    _import_structure["models.gpt2"].append("TFGPT2Tokenizer")

# Vision-specific objects
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_vision_objects

    _import_structure["utils.dummy_vision_objects"] = [
        name for name in dir(dummy_vision_objects) if not name.startswith("_")
    ]
else:
    _import_structure["image_processing_base"] = ["ImageProcessingMixin"]
    _import_structure["image_processing_utils"] = ["BaseImageProcessor"]
    _import_structure["image_utils"] = ["ImageFeatureExtractionMixin"]
    _import_structure["models.aria"].extend(["AriaImageProcessor"])
    _import_structure["models.beit"].extend(["BeitFeatureExtractor", "BeitImageProcessor"])
    _import_structure["models.bit"].extend(["BitImageProcessor"])
    _import_structure["models.blip"].extend(["BlipImageProcessor"])
    _import_structure["models.bridgetower"].append("BridgeTowerImageProcessor")
    _import_structure["models.chameleon"].append("ChameleonImageProcessor")
    _import_structure["models.chinese_clip"].extend(["ChineseCLIPFeatureExtractor", "ChineseCLIPImageProcessor"])
    _import_structure["models.clip"].extend(["CLIPFeatureExtractor", "CLIPImageProcessor"])
    _import_structure["models.conditional_detr"].extend(
        ["ConditionalDetrFeatureExtractor", "ConditionalDetrImageProcessor"]
    )
    _import_structure["models.convnext"].extend(["ConvNextFeatureExtractor", "ConvNextImageProcessor"])
    _import_structure["models.deformable_detr"].extend(
        ["DeformableDetrFeatureExtractor", "DeformableDetrImageProcessor"]
    )
    _import_structure["models.deit"].extend(["DeiTFeatureExtractor", "DeiTImageProcessor"])
    _import_structure["models.deprecated.deta"].append("DetaImageProcessor")
    _import_structure["models.deprecated.efficientformer"].append("EfficientFormerImageProcessor")
    _import_structure["models.deprecated.tvlt"].append("TvltImageProcessor")
    _import_structure["models.deprecated.vit_hybrid"].extend(["ViTHybridImageProcessor"])
    _import_structure["models.detr"].extend(["DetrFeatureExtractor", "DetrImageProcessor"])
    _import_structure["models.donut"].extend(["DonutFeatureExtractor", "DonutImageProcessor"])
    _import_structure["models.dpt"].extend(["DPTFeatureExtractor", "DPTImageProcessor"])
    _import_structure["models.efficientnet"].append("EfficientNetImageProcessor")
    _import_structure["models.flava"].extend(["FlavaFeatureExtractor", "FlavaImageProcessor", "FlavaProcessor"])
    _import_structure["models.fuyu"].extend(["FuyuImageProcessor", "FuyuProcessor"])
    _import_structure["models.glpn"].extend(["GLPNFeatureExtractor", "GLPNImageProcessor"])
    _import_structure["models.grounding_dino"].extend(["GroundingDinoImageProcessor"])
    _import_structure["models.idefics"].extend(["IdeficsImageProcessor"])
    _import_structure["models.idefics2"].extend(["Idefics2ImageProcessor"])
    _import_structure["models.idefics3"].extend(["Idefics3ImageProcessor"])
    _import_structure["models.imagegpt"].extend(["ImageGPTFeatureExtractor", "ImageGPTImageProcessor"])
    _import_structure["models.instructblipvideo"].extend(["InstructBlipVideoImageProcessor"])
    _import_structure["models.layoutlmv2"].extend(["LayoutLMv2FeatureExtractor", "LayoutLMv2ImageProcessor"])
    _import_structure["models.layoutlmv3"].extend(["LayoutLMv3FeatureExtractor", "LayoutLMv3ImageProcessor"])
    _import_structure["models.levit"].extend(["LevitFeatureExtractor", "LevitImageProcessor"])
    _import_structure["models.llava_next"].append("LlavaNextImageProcessor")
    _import_structure["models.llava_next_video"].append("LlavaNextVideoImageProcessor")
    _import_structure["models.llava_onevision"].extend(
        ["LlavaOnevisionImageProcessor", "LlavaOnevisionVideoProcessor"]
    )
    _import_structure["models.mask2former"].append("Mask2FormerImageProcessor")
    _import_structure["models.maskformer"].extend(["MaskFormerFeatureExtractor", "MaskFormerImageProcessor"])
    _import_structure["models.mllama"].extend(["MllamaImageProcessor"])
    _import_structure["models.mobilenet_v1"].extend(["MobileNetV1FeatureExtractor", "MobileNetV1ImageProcessor"])
    _import_structure["models.mobilenet_v2"].extend(["MobileNetV2FeatureExtractor", "MobileNetV2ImageProcessor"])
    _import_structure["models.mobilevit"].extend(["MobileViTFeatureExtractor", "MobileViTImageProcessor"])
    _import_structure["models.nougat"].append("NougatImageProcessor")
    _import_structure["models.oneformer"].extend(["OneFormerImageProcessor"])
    _import_structure["models.owlv2"].append("Owlv2ImageProcessor")
    _import_structure["models.owlvit"].extend(["OwlViTFeatureExtractor", "OwlViTImageProcessor"])
    _import_structure["models.perceiver"].extend(["PerceiverFeatureExtractor", "PerceiverImageProcessor"])
    _import_structure["models.pix2struct"].extend(["Pix2StructImageProcessor"])
    _import_structure["models.pixtral"].append("PixtralImageProcessor")
    _import_structure["models.poolformer"].extend(["PoolFormerFeatureExtractor", "PoolFormerImageProcessor"])
    _import_structure["models.pvt"].extend(["PvtImageProcessor"])
    _import_structure["models.qwen2_vl"].extend(["Qwen2VLImageProcessor"])
    _import_structure["models.rt_detr"].extend(["RTDetrImageProcessor"])
    _import_structure["models.sam"].extend(["SamImageProcessor"])
    _import_structure["models.segformer"].extend(["SegformerFeatureExtractor", "SegformerImageProcessor"])
    _import_structure["models.seggpt"].extend(["SegGptImageProcessor"])
    _import_structure["models.siglip"].append("SiglipImageProcessor")
    _import_structure["models.superpoint"].extend(["SuperPointImageProcessor"])
    _import_structure["models.swin2sr"].append("Swin2SRImageProcessor")
    _import_structure["models.tvp"].append("TvpImageProcessor")
    _import_structure["models.video_llava"].append("VideoLlavaImageProcessor")
    _import_structure["models.videomae"].extend(["VideoMAEFeatureExtractor", "VideoMAEImageProcessor"])
    _import_structure["models.vilt"].extend(["ViltFeatureExtractor", "ViltImageProcessor", "ViltProcessor"])
    _import_structure["models.vit"].extend(["ViTFeatureExtractor", "ViTImageProcessor"])
    _import_structure["models.vitmatte"].append("VitMatteImageProcessor")
    _import_structure["models.vivit"].append("VivitImageProcessor")
    _import_structure["models.yolos"].extend(["YolosFeatureExtractor", "YolosImageProcessor"])
    _import_structure["models.zoedepth"].append("ZoeDepthImageProcessor")

try:
    if not is_torchvision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_torchvision_objects

    _import_structure["utils.dummy_torchvision_objects"] = [
        name for name in dir(dummy_torchvision_objects) if not name.startswith("_")
    ]
else:
    _import_structure["image_processing_utils_fast"] = ["BaseImageProcessorFast"]
    _import_structure["models.deformable_detr"].append("DeformableDetrImageProcessorFast")
    _import_structure["models.detr"].append("DetrImageProcessorFast")
    _import_structure["models.pixtral"].append("PixtralImageProcessorFast")
    _import_structure["models.rt_detr"].append("RTDetrImageProcessorFast")
    _import_structure["models.vit"].append("ViTImageProcessorFast")

try:
    if not is_torchvision_available() and not is_timm_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_timm_and_torchvision_objects

    _import_structure["utils.dummy_timm_and_torchvision_objects"] = [
        name for name in dir(dummy_timm_and_torchvision_objects) if not name.startswith("_")
    ]
else:
    _import_structure["models.timm_wrapper"].extend(["TimmWrapperImageProcessor"])

# PyTorch-backed objects
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_pt_objects

    _import_structure["utils.dummy_pt_objects"] = [name for name in dir(dummy_pt_objects) if not name.startswith("_")]
else:
    _import_structure["activations"] = []
    _import_structure["benchmark.benchmark"] = ["PyTorchBenchmark"]
    _import_structure["benchmark.benchmark_args"] = ["PyTorchBenchmarkArguments"]
    _import_structure["cache_utils"] = [
        "Cache",
        "CacheConfig",
        "DynamicCache",
        "EncoderDecoderCache",
        "HQQQuantizedCache",
        "HybridCache",
        "MambaCache",
        "OffloadedCache",
        "OffloadedStaticCache",
        "QuantizedCache",
        "QuantizedCacheConfig",
        "QuantoQuantizedCache",
        "SinkCache",
        "SlidingWindowCache",
        "StaticCache",
    ]
    _import_structure["data.datasets"] = [
        "GlueDataset",
        "GlueDataTrainingArguments",
        "LineByLineTextDataset",
        "LineByLineWithRefDataset",
        "LineByLineWithSOPTextDataset",
        "SquadDataset",
        "SquadDataTrainingArguments",
        "TextDataset",
        "TextDatasetForNextSentencePrediction",
    ]
    _import_structure["generation"].extend(
        [
            "AlternatingCodebooksLogitsProcessor",
            "BayesianDetectorConfig",
            "BayesianDetectorModel",
            "BeamScorer",
            "BeamSearchScorer",
            "ClassifierFreeGuidanceLogitsProcessor",
            "ConstrainedBeamSearchScorer",
            "Constraint",
            "ConstraintListState",
            "DisjunctiveConstraint",
            "EncoderNoRepeatNGramLogitsProcessor",
            "EncoderRepetitionPenaltyLogitsProcessor",
            "EosTokenCriteria",
            "EpsilonLogitsWarper",
            "EtaLogitsWarper",
            "ExponentialDecayLengthPenalty",
            "ForcedBOSTokenLogitsProcessor",
            "ForcedEOSTokenLogitsProcessor",
            "GenerationMixin",
            "HammingDiversityLogitsProcessor",
            "InfNanRemoveLogitsProcessor",
            "LogitNormalization",
            "LogitsProcessor",
            "LogitsProcessorList",
            "LogitsWarper",
            "MaxLengthCriteria",
            "MaxTimeCriteria",
            "MinLengthLogitsProcessor",
            "MinNewTokensLengthLogitsProcessor",
            "MinPLogitsWarper",
            "NoBadWordsLogitsProcessor",
            "NoRepeatNGramLogitsProcessor",
            "PhrasalConstraint",
            "PrefixConstrainedLogitsProcessor",
            "RepetitionPenaltyLogitsProcessor",
            "SequenceBiasLogitsProcessor",
            "StoppingCriteria",
            "StoppingCriteriaList",
            "StopStringCriteria",
            "SuppressTokensAtBeginLogitsProcessor",
            "SuppressTokensLogitsProcessor",
            "SynthIDTextWatermarkDetector",
            "SynthIDTextWatermarkingConfig",
            "SynthIDTextWatermarkLogitsProcessor",
            "TemperatureLogitsWarper",
            "TopKLogitsWarper",
            "TopPLogitsWarper",
            "TypicalLogitsWarper",
            "UnbatchedClassifierFreeGuidanceLogitsProcessor",
            "WatermarkDetector",
            "WatermarkLogitsProcessor",
            "WhisperTimeStampLogitsProcessor",
        ]
    )

    # PyTorch domain libraries integration
    _import_structure["integrations.executorch"] = [
        "TorchExportableModuleWithStaticCache",
        "convert_and_export_with_cache",
    ]

    _import_structure["modeling_flash_attention_utils"] = []
    _import_structure["modeling_outputs"] = []
    _import_structure["modeling_rope_utils"] = ["ROPE_INIT_FUNCTIONS"]
    _import_structure["modeling_utils"] = ["PreTrainedModel"]

    # PyTorch models structure

    _import_structure["models.albert"].extend(
        [
            "AlbertForMaskedLM",
            "AlbertForMultipleChoice",
            "AlbertForPreTraining",
            "AlbertForQuestionAnswering",
            "AlbertForSequenceClassification",
            "AlbertForTokenClassification",
            "AlbertModel",
            "AlbertPreTrainedModel",
            "load_tf_weights_in_albert",
        ]
    )

    _import_structure["models.align"].extend(
        [
            "AlignModel",
            "AlignPreTrainedModel",
            "AlignTextModel",
            "AlignVisionModel",
        ]
    )
    _import_structure["models.altclip"].extend(
        [
            "AltCLIPModel",
            "AltCLIPPreTrainedModel",
            "AltCLIPTextModel",
            "AltCLIPVisionModel",
        ]
    )
    _import_structure["models.aria"].extend(
        [
            "AriaForConditionalGeneration",
            "AriaPreTrainedModel",
            "AriaTextForCausalLM",
            "AriaTextModel",
            "AriaTextPreTrainedModel",
        ]
    )
    _import_structure["models.audio_spectrogram_transformer"].extend(
        [
            "ASTForAudioClassification",
            "ASTModel",
            "ASTPreTrainedModel",
        ]
    )
    _import_structure["models.auto"].extend(
        [
            "MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING",
            "MODEL_FOR_AUDIO_FRAME_CLASSIFICATION_MAPPING",
            "MODEL_FOR_AUDIO_XVECTOR_MAPPING",
            "MODEL_FOR_BACKBONE_MAPPING",
            "MODEL_FOR_CAUSAL_IMAGE_MODELING_MAPPING",
            "MODEL_FOR_CAUSAL_LM_MAPPING",
            "MODEL_FOR_CTC_MAPPING",
            "MODEL_FOR_DEPTH_ESTIMATION_MAPPING",
            "MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING",
            "MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING",
            "MODEL_FOR_IMAGE_MAPPING",
            "MODEL_FOR_IMAGE_SEGMENTATION_MAPPING",
            "MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING",
            "MODEL_FOR_IMAGE_TO_IMAGE_MAPPING",
            "MODEL_FOR_INSTANCE_SEGMENTATION_MAPPING",
            "MODEL_FOR_KEYPOINT_DETECTION_MAPPING",
            "MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING",
            "MODEL_FOR_MASKED_LM_MAPPING",
            "MODEL_FOR_MASK_GENERATION_MAPPING",
            "MODEL_FOR_MULTIPLE_CHOICE_MAPPING",
            "MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING",
            "MODEL_FOR_OBJECT_DETECTION_MAPPING",
            "MODEL_FOR_PRETRAINING_MAPPING",
            "MODEL_FOR_QUESTION_ANSWERING_MAPPING",
            "MODEL_FOR_RETRIEVAL_MAPPING",
            "MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING",
            "MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING",
            "MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING",
            "MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING",
            "MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING",
            "MODEL_FOR_TEXT_ENCODING_MAPPING",
            "MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING",
            "MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING",
            "MODEL_FOR_TIME_SERIES_CLASSIFICATION_MAPPING",
            "MODEL_FOR_TIME_SERIES_REGRESSION_MAPPING",
            "MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING",
            "MODEL_FOR_UNIVERSAL_SEGMENTATION_MAPPING",
            "MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING",
            "MODEL_FOR_VISION_2_SEQ_MAPPING",
            "MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING",
            "MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING",
            "MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING",
            "MODEL_MAPPING",
            "MODEL_WITH_LM_HEAD_MAPPING",
            "AutoBackbone",
            "AutoModel",
            "AutoModelForAudioClassification",
            "AutoModelForAudioFrameClassification",
            "AutoModelForAudioXVector",
            "AutoModelForCausalLM",
            "AutoModelForCTC",
            "AutoModelForDepthEstimation",
            "AutoModelForDocumentQuestionAnswering",
            "AutoModelForImageClassification",
            "AutoModelForImageSegmentation",
            "AutoModelForImageTextToText",
            "AutoModelForImageToImage",
            "AutoModelForInstanceSegmentation",
            "AutoModelForKeypointDetection",
            "AutoModelForMaskedImageModeling",
            "AutoModelForMaskedLM",
            "AutoModelForMaskGeneration",
            "AutoModelForMultipleChoice",
            "AutoModelForNextSentencePrediction",
            "AutoModelForObjectDetection",
            "AutoModelForPreTraining",
            "AutoModelForQuestionAnswering",
            "AutoModelForSemanticSegmentation",
            "AutoModelForSeq2SeqLM",
            "AutoModelForSequenceClassification",
            "AutoModelForSpeechSeq2Seq",
            "AutoModelForTableQuestionAnswering",
            "AutoModelForTextEncoding",
            "AutoModelForTextToSpectrogram",
            "AutoModelForTextToWaveform",
            "AutoModelForTokenClassification",
            "AutoModelForUniversalSegmentation",
            "AutoModelForVideoClassification",
            "AutoModelForVision2Seq",
            "AutoModelForVisualQuestionAnswering",
            "AutoModelForZeroShotImageClassification",
            "AutoModelForZeroShotObjectDetection",
            "AutoModelWithLMHead",
        ]
    )
    _import_structure["models.autoformer"].extend(
        [
            "AutoformerForPrediction",
            "AutoformerModel",
            "AutoformerPreTrainedModel",
        ]
    )
    _import_structure["models.bamba"].extend(
        [
            "BambaForCausalLM",
            "BambaModel",
            "BambaPreTrainedModel",
        ]
    )
    _import_structure["models.bark"].extend(
        [
            "BarkCausalModel",
            "BarkCoarseModel",
            "BarkFineModel",
            "BarkModel",
            "BarkPreTrainedModel",
            "BarkSemanticModel",
        ]
    )
    _import_structure["models.bart"].extend(
        [
            "BartForCausalLM",
            "BartForConditionalGeneration",
            "BartForQuestionAnswering",
            "BartForSequenceClassification",
            "BartModel",
            "BartPretrainedModel",
            "BartPreTrainedModel",
            "PretrainedBartModel",
        ]
    )
    _import_structure["models.beit"].extend(
        [
            "BeitBackbone",
            "BeitForImageClassification",
            "BeitForMaskedImageModeling",
            "BeitForSemanticSegmentation",
            "BeitModel",
            "BeitPreTrainedModel",
        ]
    )
    _import_structure["models.bert"].extend(
        [
            "BertForMaskedLM",
            "BertForMultipleChoice",
            "BertForNextSentencePrediction",
            "BertForPreTraining",
            "BertForQuestionAnswering",
            "BertForSequenceClassification",
            "BertForTokenClassification",
            "BertLMHeadModel",
            "BertModel",
            "BertPreTrainedModel",
            "load_tf_weights_in_bert",
        ]
    )
    _import_structure["models.bert_generation"].extend(
        [
            "BertGenerationDecoder",
            "BertGenerationEncoder",
            "BertGenerationPreTrainedModel",
            "load_tf_weights_in_bert_generation",
        ]
    )
    _import_structure["models.big_bird"].extend(
        [
            "BigBirdForCausalLM",
            "BigBirdForMaskedLM",
            "BigBirdForMultipleChoice",
            "BigBirdForPreTraining",
            "BigBirdForQuestionAnswering",
            "BigBirdForSequenceClassification",
            "BigBirdForTokenClassification",
            "BigBirdModel",
            "BigBirdPreTrainedModel",
            "load_tf_weights_in_big_bird",
        ]
    )
    _import_structure["models.bigbird_pegasus"].extend(
        [
            "BigBirdPegasusForCausalLM",
            "BigBirdPegasusForConditionalGeneration",
            "BigBirdPegasusForQuestionAnswering",
            "BigBirdPegasusForSequenceClassification",
            "BigBirdPegasusModel",
            "BigBirdPegasusPreTrainedModel",
        ]
    )
    _import_structure["models.biogpt"].extend(
        [
            "BioGptForCausalLM",
            "BioGptForSequenceClassification",
            "BioGptForTokenClassification",
            "BioGptModel",
            "BioGptPreTrainedModel",
        ]
    )
    _import_structure["models.bit"].extend(
        [
            "BitBackbone",
            "BitForImageClassification",
            "BitModel",
            "BitPreTrainedModel",
        ]
    )
    _import_structure["models.blenderbot"].extend(
        [
            "BlenderbotForCausalLM",
            "BlenderbotForConditionalGeneration",
            "BlenderbotModel",
            "BlenderbotPreTrainedModel",
        ]
    )
    _import_structure["models.blenderbot_small"].extend(
        [
            "BlenderbotSmallForCausalLM",
            "BlenderbotSmallForConditionalGeneration",
            "BlenderbotSmallModel",
            "BlenderbotSmallPreTrainedModel",
        ]
    )
    _import_structure["models.blip"].extend(
        [
            "BlipForConditionalGeneration",
            "BlipForImageTextRetrieval",
            "BlipForQuestionAnswering",
            "BlipModel",
            "BlipPreTrainedModel",
            "BlipTextModel",
            "BlipVisionModel",
        ]
    )
    _import_structure["models.blip_2"].extend(
        [
            "Blip2ForConditionalGeneration",
            "Blip2ForImageTextRetrieval",
            "Blip2Model",
            "Blip2PreTrainedModel",
            "Blip2QFormerModel",
            "Blip2TextModelWithProjection",
            "Blip2VisionModel",
            "Blip2VisionModelWithProjection",
        ]
    )
    _import_structure["models.bloom"].extend(
        [
            "BloomForCausalLM",
            "BloomForQuestionAnswering",
            "BloomForSequenceClassification",
            "BloomForTokenClassification",
            "BloomModel",
            "BloomPreTrainedModel",
        ]
    )
    _import_structure["models.bridgetower"].extend(
        [
            "BridgeTowerForContrastiveLearning",
            "BridgeTowerForImageAndTextRetrieval",
            "BridgeTowerForMaskedLM",
            "BridgeTowerModel",
            "BridgeTowerPreTrainedModel",
        ]
    )
    _import_structure["models.bros"].extend(
        [
            "BrosForTokenClassification",
            "BrosModel",
            "BrosPreTrainedModel",
            "BrosProcessor",
            "BrosSpadeEEForTokenClassification",
            "BrosSpadeELForTokenClassification",
        ]
    )
    _import_structure["models.camembert"].extend(
        [
            "CamembertForCausalLM",
            "CamembertForMaskedLM",
            "CamembertForMultipleChoice",
            "CamembertForQuestionAnswering",
            "CamembertForSequenceClassification",
            "CamembertForTokenClassification",
            "CamembertModel",
            "CamembertPreTrainedModel",
        ]
    )
    _import_structure["models.canine"].extend(
        [
            "CanineForMultipleChoice",
            "CanineForQuestionAnswering",
            "CanineForSequenceClassification",
            "CanineForTokenClassification",
            "CanineModel",
            "CaninePreTrainedModel",
            "load_tf_weights_in_canine",
        ]
    )
    _import_structure["models.chameleon"].extend(
        [
            "ChameleonForConditionalGeneration",
            "ChameleonModel",
            "ChameleonPreTrainedModel",
            "ChameleonProcessor",
            "ChameleonVQVAE",
        ]
    )
    _import_structure["models.chinese_clip"].extend(
        [
            "ChineseCLIPModel",
            "ChineseCLIPPreTrainedModel",
            "ChineseCLIPTextModel",
            "ChineseCLIPVisionModel",
        ]
    )
    _import_structure["models.clap"].extend(
        [
            "ClapAudioModel",
            "ClapAudioModelWithProjection",
            "ClapFeatureExtractor",
            "ClapModel",
            "ClapPreTrainedModel",
            "ClapTextModel",
            "ClapTextModelWithProjection",
        ]
    )
    _import_structure["models.clip"].extend(
        [
            "CLIPForImageClassification",
            "CLIPModel",
            "CLIPPreTrainedModel",
            "CLIPTextModel",
            "CLIPTextModelWithProjection",
            "CLIPVisionModel",
            "CLIPVisionModelWithProjection",
        ]
    )
    _import_structure["models.clipseg"].extend(
        [
            "CLIPSegForImageSegmentation",
            "CLIPSegModel",
            "CLIPSegPreTrainedModel",
            "CLIPSegTextModel",
            "CLIPSegVisionModel",
        ]
    )
    _import_structure["models.clvp"].extend(
        [
            "ClvpDecoder",
            "ClvpEncoder",
            "ClvpForCausalLM",
            "ClvpModel",
            "ClvpModelForConditionalGeneration",
            "ClvpPreTrainedModel",
        ]
    )
    _import_structure["models.codegen"].extend(
        [
            "CodeGenForCausalLM",
            "CodeGenModel",
            "CodeGenPreTrainedModel",
        ]
    )
    _import_structure["models.cohere"].extend(["CohereForCausalLM", "CohereModel", "CoherePreTrainedModel"])
    _import_structure["models.cohere2"].extend(["Cohere2ForCausalLM", "Cohere2Model", "Cohere2PreTrainedModel"])
    _import_structure["models.colpali"].extend(
        [
            "ColPaliForRetrieval",
            "ColPaliPreTrainedModel",
        ]
    )
    _import_structure["models.conditional_detr"].extend(
        [
            "ConditionalDetrForObjectDetection",
            "ConditionalDetrForSegmentation",
            "ConditionalDetrModel",
            "ConditionalDetrPreTrainedModel",
        ]
    )
    _import_structure["models.convbert"].extend(
        [
            "ConvBertForMaskedLM",
            "ConvBertForMultipleChoice",
            "ConvBertForQuestionAnswering",
            "ConvBertForSequenceClassification",
            "ConvBertForTokenClassification",
            "ConvBertModel",
            "ConvBertPreTrainedModel",
            "load_tf_weights_in_convbert",
        ]
    )
    _import_structure["models.convnext"].extend(
        [
            "ConvNextBackbone",
            "ConvNextForImageClassification",
            "ConvNextModel",
            "ConvNextPreTrainedModel",
        ]
    )
    _import_structure["models.convnextv2"].extend(
        [
            "ConvNextV2Backbone",
            "ConvNextV2ForImageClassification",
            "ConvNextV2Model",
            "ConvNextV2PreTrainedModel",
        ]
    )
    _import_structure["models.cpmant"].extend(
        [
            "CpmAntForCausalLM",
            "CpmAntModel",
            "CpmAntPreTrainedModel",
        ]
    )
    _import_structure["models.ctrl"].extend(
        [
            "CTRLForSequenceClassification",
            "CTRLLMHeadModel",
            "CTRLModel",
            "CTRLPreTrainedModel",
        ]
    )
    _import_structure["models.cvt"].extend(
        [
            "CvtForImageClassification",
            "CvtModel",
            "CvtPreTrainedModel",
        ]
    )
    _import_structure["models.dac"].extend(
        [
            "DacModel",
            "DacPreTrainedModel",
        ]
    )
    _import_structure["models.data2vec"].extend(
        [
            "Data2VecAudioForAudioFrameClassification",
            "Data2VecAudioForCTC",
            "Data2VecAudioForSequenceClassification",
            "Data2VecAudioForXVector",
            "Data2VecAudioModel",
            "Data2VecAudioPreTrainedModel",
            "Data2VecTextForCausalLM",
            "Data2VecTextForMaskedLM",
            "Data2VecTextForMultipleChoice",
            "Data2VecTextForQuestionAnswering",
            "Data2VecTextForSequenceClassification",
            "Data2VecTextForTokenClassification",
            "Data2VecTextModel",
            "Data2VecTextPreTrainedModel",
            "Data2VecVisionForImageClassification",
            "Data2VecVisionForSemanticSegmentation",
            "Data2VecVisionModel",
            "Data2VecVisionPreTrainedModel",
        ]
    )
    _import_structure["models.dbrx"].extend(
        [
            "DbrxForCausalLM",
            "DbrxModel",
            "DbrxPreTrainedModel",
        ]
    )
    _import_structure["models.deberta"].extend(
        [
            "DebertaForMaskedLM",
            "DebertaForQuestionAnswering",
            "DebertaForSequenceClassification",
            "DebertaForTokenClassification",
            "DebertaModel",
            "DebertaPreTrainedModel",
        ]
    )
    _import_structure["models.deberta_v2"].extend(
        [
            "DebertaV2ForMaskedLM",
            "DebertaV2ForMultipleChoice",
            "DebertaV2ForQuestionAnswering",
            "DebertaV2ForSequenceClassification",
            "DebertaV2ForTokenClassification",
            "DebertaV2Model",
            "DebertaV2PreTrainedModel",
        ]
    )
    _import_structure["models.decision_transformer"].extend(
        [
            "DecisionTransformerGPT2Model",
            "DecisionTransformerGPT2PreTrainedModel",
            "DecisionTransformerModel",
            "DecisionTransformerPreTrainedModel",
        ]
    )
    _import_structure["models.deformable_detr"].extend(
        [
            "DeformableDetrForObjectDetection",
            "DeformableDetrModel",
            "DeformableDetrPreTrainedModel",
        ]
    )
    _import_structure["models.deit"].extend(
        [
            "DeiTForImageClassification",
            "DeiTForImageClassificationWithTeacher",
            "DeiTForMaskedImageModeling",
            "DeiTModel",
            "DeiTPreTrainedModel",
        ]
    )
    _import_structure["models.deprecated.deta"].extend(
        [
            "DetaForObjectDetection",
            "DetaModel",
            "DetaPreTrainedModel",
        ]
    )
    _import_structure["models.deprecated.efficientformer"].extend(
        [
            "EfficientFormerForImageClassification",
            "EfficientFormerForImageClassificationWithTeacher",
            "EfficientFormerModel",
            "EfficientFormerPreTrainedModel",
        ]
    )
    _import_structure["models.deprecated.ernie_m"].extend(
        [
            "ErnieMForInformationExtraction",
            "ErnieMForMultipleChoice",
            "ErnieMForQuestionAnswering",
            "ErnieMForSequenceClassification",
            "ErnieMForTokenClassification",
            "ErnieMModel",
            "ErnieMPreTrainedModel",
        ]
    )
    _import_structure["models.deprecated.gptsan_japanese"].extend(
        [
            "GPTSanJapaneseForConditionalGeneration",
            "GPTSanJapaneseModel",
            "GPTSanJapanesePreTrainedModel",
        ]
    )
    _import_structure["models.deprecated.graphormer"].extend(
        [
            "GraphormerForGraphClassification",
            "GraphormerModel",
            "GraphormerPreTrainedModel",
        ]
    )
    _import_structure["models.deprecated.jukebox"].extend(
        [
            "JukeboxModel",
            "JukeboxPreTrainedModel",
            "JukeboxPrior",
            "JukeboxVQVAE",
        ]
    )
    _import_structure["models.deprecated.mctct"].extend(
        [
            "MCTCTForCTC",
            "MCTCTModel",
            "MCTCTPreTrainedModel",
        ]
    )
    _import_structure["models.deprecated.mega"].extend(
        [
            "MegaForCausalLM",
            "MegaForMaskedLM",
            "MegaForMultipleChoice",
            "MegaForQuestionAnswering",
            "MegaForSequenceClassification",
            "MegaForTokenClassification",
            "MegaModel",
            "MegaPreTrainedModel",
        ]
    )
    _import_structure["models.deprecated.mmbt"].extend(["MMBTForClassification", "MMBTModel", "ModalEmbeddings"])
    _import_structure["models.deprecated.nat"].extend(
        [
            "NatBackbone",
            "NatForImageClassification",
            "NatModel",
            "NatPreTrainedModel",
        ]
    )
    _import_structure["models.deprecated.nezha"].extend(
        [
            "NezhaForMaskedLM",
            "NezhaForMultipleChoice",
            "NezhaForNextSentencePrediction",
            "NezhaForPreTraining",
            "NezhaForQuestionAnswering",
            "NezhaForSequenceClassification",
            "NezhaForTokenClassification",
            "NezhaModel",
            "NezhaPreTrainedModel",
        ]
    )
    _import_structure["models.deprecated.open_llama"].extend(
        [
            "OpenLlamaForCausalLM",
            "OpenLlamaForSequenceClassification",
            "OpenLlamaModel",
            "OpenLlamaPreTrainedModel",
        ]
    )
    _import_structure["models.deprecated.qdqbert"].extend(
        [
            "QDQBertForMaskedLM",
            "QDQBertForMultipleChoice",
            "QDQBertForNextSentencePrediction",
            "QDQBertForQuestionAnswering",
            "QDQBertForSequenceClassification",
            "QDQBertForTokenClassification",
            "QDQBertLMHeadModel",
            "QDQBertModel",
            "QDQBertPreTrainedModel",
            "load_tf_weights_in_qdqbert",
        ]
    )
    _import_structure["models.deprecated.realm"].extend(
        [
            "RealmEmbedder",
            "RealmForOpenQA",
            "RealmKnowledgeAugEncoder",
            "RealmPreTrainedModel",
            "RealmReader",
            "RealmRetriever",
            "RealmScorer",
            "load_tf_weights_in_realm",
        ]
    )
    _import_structure["models.deprecated.retribert"].extend(
        [
            "RetriBertModel",
            "RetriBertPreTrainedModel",
        ]
    )
    _import_structure["models.deprecated.speech_to_text_2"].extend(
        ["Speech2Text2ForCausalLM", "Speech2Text2PreTrainedModel"]
    )
    _import_structure["models.deprecated.trajectory_transformer"].extend(
        [
            "TrajectoryTransformerModel",
            "TrajectoryTransformerPreTrainedModel",
        ]
    )
    _import_structure["models.deprecated.transfo_xl"].extend(
        [
            "AdaptiveEmbedding",
            "TransfoXLForSequenceClassification",
            "TransfoXLLMHeadModel",
            "TransfoXLModel",
            "TransfoXLPreTrainedModel",
            "load_tf_weights_in_transfo_xl",
        ]
    )
    _import_structure["models.deprecated.tvlt"].extend(
        [
            "TvltForAudioVisualClassification",
            "TvltForPreTraining",
            "TvltModel",
            "TvltPreTrainedModel",
        ]
    )
    _import_structure["models.deprecated.van"].extend(
        [
            "VanForImageClassification",
            "VanModel",
            "VanPreTrainedModel",
        ]
    )
    _import_structure["models.deprecated.vit_hybrid"].extend(
        [
            "ViTHybridForImageClassification",
            "ViTHybridModel",
            "ViTHybridPreTrainedModel",
        ]
    )
    _import_structure["models.deprecated.xlm_prophetnet"].extend(
        [
            "XLMProphetNetDecoder",
            "XLMProphetNetEncoder",
            "XLMProphetNetForCausalLM",
            "XLMProphetNetForConditionalGeneration",
            "XLMProphetNetModel",
            "XLMProphetNetPreTrainedModel",
        ]
    )
    _import_structure["models.depth_anything"].extend(
        [
            "DepthAnythingForDepthEstimation",
            "DepthAnythingPreTrainedModel",
        ]
    )
    _import_structure["models.detr"].extend(
        [
            "DetrForObjectDetection",
            "DetrForSegmentation",
            "DetrModel",
            "DetrPreTrainedModel",
        ]
    )
    _import_structure["models.diffllama"].extend(
        [
            "DiffLlamaForCausalLM",
            "DiffLlamaForQuestionAnswering",
            "DiffLlamaForSequenceClassification",
            "DiffLlamaForTokenClassification",
            "DiffLlamaModel",
            "DiffLlamaPreTrainedModel",
        ]
    )
    _import_structure["models.dinat"].extend(
        [
            "DinatBackbone",
            "DinatForImageClassification",
            "DinatModel",
            "DinatPreTrainedModel",
        ]
    )
    _import_structure["models.dinov2"].extend(
        [
            "Dinov2Backbone",
            "Dinov2ForImageClassification",
            "Dinov2Model",
            "Dinov2PreTrainedModel",
        ]
    )
    _import_structure["models.dinov2_with_registers"].extend(
        [
            "Dinov2WithRegistersBackbone",
            "Dinov2WithRegistersForImageClassification",
            "Dinov2WithRegistersModel",
            "Dinov2WithRegistersPreTrainedModel",
        ]
    )
    _import_structure["models.distilbert"].extend(
        [
            "DistilBertForMaskedLM",
            "DistilBertForMultipleChoice",
            "DistilBertForQuestionAnswering",
            "DistilBertForSequenceClassification",
            "DistilBertForTokenClassification",
            "DistilBertModel",
            "DistilBertPreTrainedModel",
        ]
    )
    _import_structure["models.donut"].extend(
        [
            "DonutSwinModel",
            "DonutSwinPreTrainedModel",
        ]
    )
    _import_structure["models.dpr"].extend(
        [
            "DPRContextEncoder",
            "DPRPretrainedContextEncoder",
            "DPRPreTrainedModel",
            "DPRPretrainedQuestionEncoder",
            "DPRPretrainedReader",
            "DPRQuestionEncoder",
            "DPRReader",
        ]
    )
    _import_structure["models.dpt"].extend(
        [
            "DPTForDepthEstimation",
            "DPTForSemanticSegmentation",
            "DPTModel",
            "DPTPreTrainedModel",
        ]
    )
    _import_structure["models.efficientnet"].extend(
        [
            "EfficientNetForImageClassification",
            "EfficientNetModel",
            "EfficientNetPreTrainedModel",
        ]
    )
    _import_structure["models.electra"].extend(
        [
            "ElectraForCausalLM",
            "ElectraForMaskedLM",
            "ElectraForMultipleChoice",
            "ElectraForPreTraining",
            "ElectraForQuestionAnswering",
            "ElectraForSequenceClassification",
            "ElectraForTokenClassification",
            "ElectraModel",
            "ElectraPreTrainedModel",
            "load_tf_weights_in_electra",
        ]
    )
    _import_structure["models.encodec"].extend(
        [
            "EncodecModel",
            "EncodecPreTrainedModel",
        ]
    )
    _import_structure["models.encoder_decoder"].append("EncoderDecoderModel")
    _import_structure["models.ernie"].extend(
        [
            "ErnieForCausalLM",
            "ErnieForMaskedLM",
            "ErnieForMultipleChoice",
            "ErnieForNextSentencePrediction",
            "ErnieForPreTraining",
            "ErnieForQuestionAnswering",
            "ErnieForSequenceClassification",
            "ErnieForTokenClassification",
            "ErnieModel",
            "ErniePreTrainedModel",
        ]
    )
    _import_structure["models.esm"].extend(
        [
            "EsmFoldPreTrainedModel",
            "EsmForMaskedLM",
            "EsmForProteinFolding",
            "EsmForSequenceClassification",
            "EsmForTokenClassification",
            "EsmModel",
            "EsmPreTrainedModel",
        ]
    )
    _import_structure["models.falcon"].extend(
        [
            "FalconForCausalLM",
            "FalconForQuestionAnswering",
            "FalconForSequenceClassification",
            "FalconForTokenClassification",
            "FalconModel",
            "FalconPreTrainedModel",
        ]
    )
    _import_structure["models.falcon_mamba"].extend(
        [
            "FalconMambaForCausalLM",
            "FalconMambaModel",
            "FalconMambaPreTrainedModel",
        ]
    )
    _import_structure["models.fastspeech2_conformer"].extend(
        [
            "FastSpeech2ConformerHifiGan",
            "FastSpeech2ConformerModel",
            "FastSpeech2ConformerPreTrainedModel",
            "FastSpeech2ConformerWithHifiGan",
        ]
    )
    _import_structure["models.flaubert"].extend(
        [
            "FlaubertForMultipleChoice",
            "FlaubertForQuestionAnswering",
            "FlaubertForQuestionAnsweringSimple",
            "FlaubertForSequenceClassification",
            "FlaubertForTokenClassification",
            "FlaubertModel",
            "FlaubertPreTrainedModel",
            "FlaubertWithLMHeadModel",
        ]
    )
    _import_structure["models.flava"].extend(
        [
            "FlavaForPreTraining",
            "FlavaImageCodebook",
            "FlavaImageModel",
            "FlavaModel",
            "FlavaMultimodalModel",
            "FlavaPreTrainedModel",
            "FlavaTextModel",
        ]
    )
    _import_structure["models.fnet"].extend(
        [
            "FNetForMaskedLM",
            "FNetForMultipleChoice",
            "FNetForNextSentencePrediction",
            "FNetForPreTraining",
            "FNetForQuestionAnswering",
            "FNetForSequenceClassification",
            "FNetForTokenClassification",
            "FNetModel",
            "FNetPreTrainedModel",
        ]
    )
    _import_structure["models.focalnet"].extend(
        [
            "FocalNetBackbone",
            "FocalNetForImageClassification",
            "FocalNetForMaskedImageModeling",
            "FocalNetModel",
            "FocalNetPreTrainedModel",
        ]
    )
    _import_structure["models.fsmt"].extend(["FSMTForConditionalGeneration", "FSMTModel", "PretrainedFSMTModel"])
    _import_structure["models.funnel"].extend(
        [
            "FunnelBaseModel",
            "FunnelForMaskedLM",
            "FunnelForMultipleChoice",
            "FunnelForPreTraining",
            "FunnelForQuestionAnswering",
            "FunnelForSequenceClassification",
            "FunnelForTokenClassification",
            "FunnelModel",
            "FunnelPreTrainedModel",
            "load_tf_weights_in_funnel",
        ]
    )
    _import_structure["models.fuyu"].extend(["FuyuForCausalLM", "FuyuPreTrainedModel"])
    _import_structure["models.gemma"].extend(
        [
            "GemmaForCausalLM",
            "GemmaForSequenceClassification",
            "GemmaForTokenClassification",
            "GemmaModel",
            "GemmaPreTrainedModel",
        ]
    )
    _import_structure["models.gemma2"].extend(
        [
            "Gemma2ForCausalLM",
            "Gemma2ForSequenceClassification",
            "Gemma2ForTokenClassification",
            "Gemma2Model",
            "Gemma2PreTrainedModel",
        ]
    )
    _import_structure["models.git"].extend(
        [
            "GitForCausalLM",
            "GitModel",
            "GitPreTrainedModel",
            "GitVisionModel",
        ]
    )
    _import_structure["models.glm"].extend(
        [
            "GlmForCausalLM",
            "GlmForSequenceClassification",
            "GlmForTokenClassification",
            "GlmModel",
            "GlmPreTrainedModel",
        ]
    )
    _import_structure["models.glpn"].extend(
        [
            "GLPNForDepthEstimation",
            "GLPNModel",
            "GLPNPreTrainedModel",
        ]
    )
    _import_structure["models.gpt2"].extend(
        [
            "GPT2DoubleHeadsModel",
            "GPT2ForQuestionAnswering",
            "GPT2ForSequenceClassification",
            "GPT2ForTokenClassification",
            "GPT2LMHeadModel",
            "GPT2Model",
            "GPT2PreTrainedModel",
            "load_tf_weights_in_gpt2",
        ]
    )
    _import_structure["models.gpt_bigcode"].extend(
        [
            "GPTBigCodeForCausalLM",
            "GPTBigCodeForSequenceClassification",
            "GPTBigCodeForTokenClassification",
            "GPTBigCodeModel",
            "GPTBigCodePreTrainedModel",
        ]
    )
    _import_structure["models.gpt_neo"].extend(
        [
            "GPTNeoForCausalLM",
            "GPTNeoForQuestionAnswering",
            "GPTNeoForSequenceClassification",
            "GPTNeoForTokenClassification",
            "GPTNeoModel",
            "GPTNeoPreTrainedModel",
            "load_tf_weights_in_gpt_neo",
        ]
    )
    _import_structure["models.gpt_neox"].extend(
        [
            "GPTNeoXForCausalLM",
            "GPTNeoXForQuestionAnswering",
            "GPTNeoXForSequenceClassification",
            "GPTNeoXForTokenClassification",
            "GPTNeoXModel",
            "GPTNeoXPreTrainedModel",
        ]
    )
    _import_structure["models.gpt_neox_japanese"].extend(
        [
            "GPTNeoXJapaneseForCausalLM",
            "GPTNeoXJapaneseModel",
            "GPTNeoXJapanesePreTrainedModel",
        ]
    )
    _import_structure["models.gptj"].extend(
        [
            "GPTJForCausalLM",
            "GPTJForQuestionAnswering",
            "GPTJForSequenceClassification",
            "GPTJModel",
            "GPTJPreTrainedModel",
        ]
    )
    _import_structure["models.granite"].extend(
        [
            "GraniteForCausalLM",
            "GraniteModel",
            "GranitePreTrainedModel",
        ]
    )
    _import_structure["models.granitemoe"].extend(
        [
            "GraniteMoeForCausalLM",
            "GraniteMoeModel",
            "GraniteMoePreTrainedModel",
        ]
    )
    _import_structure["models.grounding_dino"].extend(
        [
            "GroundingDinoForObjectDetection",
            "GroundingDinoModel",
            "GroundingDinoPreTrainedModel",
        ]
    )
    _import_structure["models.groupvit"].extend(
        [
            "GroupViTModel",
            "GroupViTPreTrainedModel",
            "GroupViTTextModel",
            "GroupViTVisionModel",
        ]
    )
    _import_structure["models.hiera"].extend(
        [
            "HieraBackbone",
            "HieraForImageClassification",
            "HieraForPreTraining",
            "HieraModel",
            "HieraPreTrainedModel",
        ]
    )
    _import_structure["models.hubert"].extend(
        [
            "HubertForCTC",
            "HubertForSequenceClassification",
            "HubertModel",
            "HubertPreTrainedModel",
        ]
    )
    _import_structure["models.ibert"].extend(
        [
            "IBertForMaskedLM",
            "IBertForMultipleChoice",
            "IBertForQuestionAnswering",
            "IBertForSequenceClassification",
            "IBertForTokenClassification",
            "IBertModel",
            "IBertPreTrainedModel",
        ]
    )
    _import_structure["models.idefics"].extend(
        [
            "IdeficsForVisionText2Text",
            "IdeficsModel",
            "IdeficsPreTrainedModel",
            "IdeficsProcessor",
        ]
    )
    _import_structure["models.idefics2"].extend(
        [
            "Idefics2ForConditionalGeneration",
            "Idefics2Model",
            "Idefics2PreTrainedModel",
            "Idefics2Processor",
        ]
    )
    _import_structure["models.idefics3"].extend(
        [
            "Idefics3ForConditionalGeneration",
            "Idefics3Model",
            "Idefics3PreTrainedModel",
            "Idefics3Processor",
            "Idefics3VisionConfig",
            "Idefics3VisionTransformer",
        ]
    )
    _import_structure["models.ijepa"].extend(
        [
            "IJepaForImageClassification",
            "IJepaModel",
            "IJepaPreTrainedModel",
        ]
    )
    _import_structure["models.imagegpt"].extend(
        [
            "ImageGPTForCausalImageModeling",
            "ImageGPTForImageClassification",
            "ImageGPTModel",
            "ImageGPTPreTrainedModel",
            "load_tf_weights_in_imagegpt",
        ]
    )
    _import_structure["models.informer"].extend(
        [
            "InformerForPrediction",
            "InformerModel",
            "InformerPreTrainedModel",
        ]
    )
    _import_structure["models.instructblip"].extend(
        [
            "InstructBlipForConditionalGeneration",
            "InstructBlipPreTrainedModel",
            "InstructBlipQFormerModel",
            "InstructBlipVisionModel",
        ]
    )
    _import_structure["models.instructblipvideo"].extend(
        [
            "InstructBlipVideoForConditionalGeneration",
            "InstructBlipVideoPreTrainedModel",
            "InstructBlipVideoQFormerModel",
            "InstructBlipVideoVisionModel",
        ]
    )
    _import_structure["models.jamba"].extend(
        [
            "JambaForCausalLM",
            "JambaForSequenceClassification",
            "JambaModel",
            "JambaPreTrainedModel",
        ]
    )
    _import_structure["models.jetmoe"].extend(
        [
            "JetMoeForCausalLM",
            "JetMoeForSequenceClassification",
            "JetMoeModel",
            "JetMoePreTrainedModel",
        ]
    )
    _import_structure["models.kosmos2"].extend(
        [
            "Kosmos2ForConditionalGeneration",
            "Kosmos2Model",
            "Kosmos2PreTrainedModel",
        ]
    )
    _import_structure["models.layoutlm"].extend(
        [
            "LayoutLMForMaskedLM",
            "LayoutLMForQuestionAnswering",
            "LayoutLMForSequenceClassification",
            "LayoutLMForTokenClassification",
            "LayoutLMModel",
            "LayoutLMPreTrainedModel",
        ]
    )
    _import_structure["models.layoutlmv2"].extend(
        [
            "LayoutLMv2ForQuestionAnswering",
            "LayoutLMv2ForSequenceClassification",
            "LayoutLMv2ForTokenClassification",
            "LayoutLMv2Model",
            "LayoutLMv2PreTrainedModel",
        ]
    )
    _import_structure["models.layoutlmv3"].extend(
        [
            "LayoutLMv3ForQuestionAnswering",
            "LayoutLMv3ForSequenceClassification",
            "LayoutLMv3ForTokenClassification",
            "LayoutLMv3Model",
            "LayoutLMv3PreTrainedModel",
        ]
    )
    _import_structure["models.led"].extend(
        [
            "LEDForConditionalGeneration",
            "LEDForQuestionAnswering",
            "LEDForSequenceClassification",
            "LEDModel",
            "LEDPreTrainedModel",
        ]
    )
    _import_structure["models.levit"].extend(
        [
            "LevitForImageClassification",
            "LevitForImageClassificationWithTeacher",
            "LevitModel",
            "LevitPreTrainedModel",
        ]
    )
    _import_structure["models.lilt"].extend(
        [
            "LiltForQuestionAnswering",
            "LiltForSequenceClassification",
            "LiltForTokenClassification",
            "LiltModel",
            "LiltPreTrainedModel",
        ]
    )
    _import_structure["models.llama"].extend(
        [
            "LlamaForCausalLM",
            "LlamaForQuestionAnswering",
            "LlamaForSequenceClassification",
            "LlamaForTokenClassification",
            "LlamaModel",
            "LlamaPreTrainedModel",
        ]
    )
    _import_structure["models.llava"].extend(
        [
            "LlavaForConditionalGeneration",
            "LlavaPreTrainedModel",
        ]
    )
    _import_structure["models.llava_next"].extend(
        [
            "LlavaNextForConditionalGeneration",
            "LlavaNextPreTrainedModel",
        ]
    )
    _import_structure["models.llava_next_video"].extend(
        [
            "LlavaNextVideoForConditionalGeneration",
            "LlavaNextVideoPreTrainedModel",
        ]
    )
    _import_structure["models.llava_onevision"].extend(
        [
            "LlavaOnevisionForConditionalGeneration",
            "LlavaOnevisionPreTrainedModel",
        ]
    )
    _import_structure["models.longformer"].extend(
        [
            "LongformerForMaskedLM",
            "LongformerForMultipleChoice",
            "LongformerForQuestionAnswering",
            "LongformerForSequenceClassification",
            "LongformerForTokenClassification",
            "LongformerModel",
            "LongformerPreTrainedModel",
        ]
    )
    _import_structure["models.longt5"].extend(
        [
            "LongT5EncoderModel",
            "LongT5ForConditionalGeneration",
            "LongT5Model",
            "LongT5PreTrainedModel",
        ]
    )
    _import_structure["models.luke"].extend(
        [
            "LukeForEntityClassification",
            "LukeForEntityPairClassification",
            "LukeForEntitySpanClassification",
            "LukeForMaskedLM",
            "LukeForMultipleChoice",
            "LukeForQuestionAnswering",
            "LukeForSequenceClassification",
            "LukeForTokenClassification",
            "LukeModel",
            "LukePreTrainedModel",
        ]
    )
    _import_structure["models.lxmert"].extend(
        [
            "LxmertEncoder",
            "LxmertForPreTraining",
            "LxmertForQuestionAnswering",
            "LxmertModel",
            "LxmertPreTrainedModel",
            "LxmertVisualFeatureEncoder",
        ]
    )
    _import_structure["models.m2m_100"].extend(
        [
            "M2M100ForConditionalGeneration",
            "M2M100Model",
            "M2M100PreTrainedModel",
        ]
    )
    _import_structure["models.mamba"].extend(
        [
            "MambaForCausalLM",
            "MambaModel",
            "MambaPreTrainedModel",
        ]
    )
    _import_structure["models.mamba2"].extend(
        [
            "Mamba2ForCausalLM",
            "Mamba2Model",
            "Mamba2PreTrainedModel",
        ]
    )
    _import_structure["models.marian"].extend(
        ["MarianForCausalLM", "MarianModel", "MarianMTModel", "MarianPreTrainedModel"]
    )
    _import_structure["models.markuplm"].extend(
        [
            "MarkupLMForQuestionAnswering",
            "MarkupLMForSequenceClassification",
            "MarkupLMForTokenClassification",
            "MarkupLMModel",
            "MarkupLMPreTrainedModel",
        ]
    )
    _import_structure["models.mask2former"].extend(
        [
            "Mask2FormerForUniversalSegmentation",
            "Mask2FormerModel",
            "Mask2FormerPreTrainedModel",
        ]
    )
    _import_structure["models.maskformer"].extend(
        [
            "MaskFormerForInstanceSegmentation",
            "MaskFormerModel",
            "MaskFormerPreTrainedModel",
            "MaskFormerSwinBackbone",
        ]
    )
    _import_structure["models.mbart"].extend(
        [
            "MBartForCausalLM",
            "MBartForConditionalGeneration",
            "MBartForQuestionAnswering",
            "MBartForSequenceClassification",
            "MBartModel",
            "MBartPreTrainedModel",
        ]
    )
    _import_structure["models.megatron_bert"].extend(
        [
            "MegatronBertForCausalLM",
            "MegatronBertForMaskedLM",
            "MegatronBertForMultipleChoice",
            "MegatronBertForNextSentencePrediction",
            "MegatronBertForPreTraining",
            "MegatronBertForQuestionAnswering",
            "MegatronBertForSequenceClassification",
            "MegatronBertForTokenClassification",
            "MegatronBertModel",
            "MegatronBertPreTrainedModel",
        ]
    )
    _import_structure["models.mgp_str"].extend(
        [
            "MgpstrForSceneTextRecognition",
            "MgpstrModel",
            "MgpstrPreTrainedModel",
        ]
    )
    _import_structure["models.mimi"].extend(
        [
            "MimiModel",
            "MimiPreTrainedModel",
        ]
    )
    _import_structure["models.mistral"].extend(
        [
            "MistralForCausalLM",
            "MistralForQuestionAnswering",
            "MistralForSequenceClassification",
            "MistralForTokenClassification",
            "MistralModel",
            "MistralPreTrainedModel",
        ]
    )
    _import_structure["models.mixtral"].extend(
        [
            "MixtralForCausalLM",
            "MixtralForQuestionAnswering",
            "MixtralForSequenceClassification",
            "MixtralForTokenClassification",
            "MixtralModel",
            "MixtralPreTrainedModel",
        ]
    )
    _import_structure["models.mllama"].extend(
        [
            "MllamaForCausalLM",
            "MllamaForConditionalGeneration",
            "MllamaPreTrainedModel",
            "MllamaProcessor",
            "MllamaTextModel",
            "MllamaVisionModel",
        ]
    )
    _import_structure["models.mobilebert"].extend(
        [
            "MobileBertForMaskedLM",
            "MobileBertForMultipleChoice",
            "MobileBertForNextSentencePrediction",
            "MobileBertForPreTraining",
            "MobileBertForQuestionAnswering",
            "MobileBertForSequenceClassification",
            "MobileBertForTokenClassification",
            "MobileBertModel",
            "MobileBertPreTrainedModel",
            "load_tf_weights_in_mobilebert",
        ]
    )
    _import_structure["models.mobilenet_v1"].extend(
        [
            "MobileNetV1ForImageClassification",
            "MobileNetV1Model",
            "MobileNetV1PreTrainedModel",
            "load_tf_weights_in_mobilenet_v1",
        ]
    )
    _import_structure["models.mobilenet_v2"].extend(
        [
            "MobileNetV2ForImageClassification",
            "MobileNetV2ForSemanticSegmentation",
            "MobileNetV2Model",
            "MobileNetV2PreTrainedModel",
            "load_tf_weights_in_mobilenet_v2",
        ]
    )
    _import_structure["models.mobilevit"].extend(
        [
            "MobileViTForImageClassification",
            "MobileViTForSemanticSegmentation",
            "MobileViTModel",
            "MobileViTPreTrainedModel",
        ]
    )
    _import_structure["models.mobilevitv2"].extend(
        [
            "MobileViTV2ForImageClassification",
            "MobileViTV2ForSemanticSegmentation",
            "MobileViTV2Model",
            "MobileViTV2PreTrainedModel",
        ]
    )
    _import_structure["models.modernbert"].extend(
        [
            "ModernBertForMaskedLM",
            "ModernBertForSequenceClassification",
            "ModernBertForTokenClassification",
            "ModernBertModel",
            "ModernBertPreTrainedModel",
        ]
    )
    _import_structure["models.moshi"].extend(
        [
            "MoshiForCausalLM",
            "MoshiForConditionalGeneration",
            "MoshiModel",
            "MoshiPreTrainedModel",
        ]
    )
    _import_structure["models.mpnet"].extend(
        [
            "MPNetForMaskedLM",
            "MPNetForMultipleChoice",
            "MPNetForQuestionAnswering",
            "MPNetForSequenceClassification",
            "MPNetForTokenClassification",
            "MPNetModel",
            "MPNetPreTrainedModel",
        ]
    )
    _import_structure["models.mpt"].extend(
        [
            "MptForCausalLM",
            "MptForQuestionAnswering",
            "MptForSequenceClassification",
            "MptForTokenClassification",
            "MptModel",
            "MptPreTrainedModel",
        ]
    )
    _import_structure["models.mra"].extend(
        [
            "MraForMaskedLM",
            "MraForMultipleChoice",
            "MraForQuestionAnswering",
            "MraForSequenceClassification",
            "MraForTokenClassification",
            "MraModel",
            "MraPreTrainedModel",
        ]
    )
    _import_structure["models.mt5"].extend(
        [
            "MT5EncoderModel",
            "MT5ForConditionalGeneration",
            "MT5ForQuestionAnswering",
            "MT5ForSequenceClassification",
            "MT5ForTokenClassification",
            "MT5Model",
            "MT5PreTrainedModel",
        ]
    )
    _import_structure["models.musicgen"].extend(
        [
            "MusicgenForCausalLM",
            "MusicgenForConditionalGeneration",
            "MusicgenModel",
            "MusicgenPreTrainedModel",
            "MusicgenProcessor",
        ]
    )
    _import_structure["models.musicgen_melody"].extend(
        [
            "MusicgenMelodyForCausalLM",
            "MusicgenMelodyForConditionalGeneration",
            "MusicgenMelodyModel",
            "MusicgenMelodyPreTrainedModel",
        ]
    )
    _import_structure["models.mvp"].extend(
        [
            "MvpForCausalLM",
            "MvpForConditionalGeneration",
            "MvpForQuestionAnswering",
            "MvpForSequenceClassification",
            "MvpModel",
            "MvpPreTrainedModel",
        ]
    )
    _import_structure["models.nemotron"].extend(
        [
            "NemotronForCausalLM",
            "NemotronForQuestionAnswering",
            "NemotronForSequenceClassification",
            "NemotronForTokenClassification",
            "NemotronModel",
            "NemotronPreTrainedModel",
        ]
    )
    _import_structure["models.nllb_moe"].extend(
        [
            "NllbMoeForConditionalGeneration",
            "NllbMoeModel",
            "NllbMoePreTrainedModel",
            "NllbMoeSparseMLP",
            "NllbMoeTop2Router",
        ]
    )
    _import_structure["models.nystromformer"].extend(
        [
            "NystromformerForMaskedLM",
            "NystromformerForMultipleChoice",
            "NystromformerForQuestionAnswering",
            "NystromformerForSequenceClassification",
            "NystromformerForTokenClassification",
            "NystromformerModel",
            "NystromformerPreTrainedModel",
        ]
    )
    _import_structure["models.olmo"].extend(
        [
            "OlmoForCausalLM",
            "OlmoModel",
            "OlmoPreTrainedModel",
        ]
    )
    _import_structure["models.olmo2"].extend(
        [
            "Olmo2ForCausalLM",
            "Olmo2Model",
            "Olmo2PreTrainedModel",
        ]
    )
    _import_structure["models.olmoe"].extend(
        [
            "OlmoeForCausalLM",
            "OlmoeModel",
            "OlmoePreTrainedModel",
        ]
    )
    _import_structure["models.omdet_turbo"].extend(
        [
            "OmDetTurboForObjectDetection",
            "OmDetTurboPreTrainedModel",
        ]
    )
    _import_structure["models.oneformer"].extend(
        [
            "OneFormerForUniversalSegmentation",
            "OneFormerModel",
            "OneFormerPreTrainedModel",
        ]
    )
    _import_structure["models.openai"].extend(
        [
            "OpenAIGPTDoubleHeadsModel",
            "OpenAIGPTForSequenceClassification",
            "OpenAIGPTLMHeadModel",
            "OpenAIGPTModel",
            "OpenAIGPTPreTrainedModel",
            "load_tf_weights_in_openai_gpt",
        ]
    )
    _import_structure["models.opt"].extend(
        [
            "OPTForCausalLM",
            "OPTForQuestionAnswering",
            "OPTForSequenceClassification",
            "OPTModel",
            "OPTPreTrainedModel",
        ]
    )
    _import_structure["models.owlv2"].extend(
        [
            "Owlv2ForObjectDetection",
            "Owlv2Model",
            "Owlv2PreTrainedModel",
            "Owlv2TextModel",
            "Owlv2VisionModel",
        ]
    )
    _import_structure["models.owlvit"].extend(
        [
            "OwlViTForObjectDetection",
            "OwlViTModel",
            "OwlViTPreTrainedModel",
            "OwlViTTextModel",
            "OwlViTVisionModel",
        ]
    )
    _import_structure["models.paligemma"].extend(
        [
            "PaliGemmaForConditionalGeneration",
            "PaliGemmaPreTrainedModel",
            "PaliGemmaProcessor",
        ]
    )
    _import_structure["models.patchtsmixer"].extend(
        [
            "PatchTSMixerForPrediction",
            "PatchTSMixerForPretraining",
            "PatchTSMixerForRegression",
            "PatchTSMixerForTimeSeriesClassification",
            "PatchTSMixerModel",
            "PatchTSMixerPreTrainedModel",
        ]
    )
    _import_structure["models.patchtst"].extend(
        [
            "PatchTSTForClassification",
            "PatchTSTForPrediction",
            "PatchTSTForPretraining",
            "PatchTSTForRegression",
            "PatchTSTModel",
            "PatchTSTPreTrainedModel",
        ]
    )
    _import_structure["models.pegasus"].extend(
        [
            "PegasusForCausalLM",
            "PegasusForConditionalGeneration",
            "PegasusModel",
            "PegasusPreTrainedModel",
        ]
    )
    _import_structure["models.pegasus_x"].extend(
        [
            "PegasusXForConditionalGeneration",
            "PegasusXModel",
            "PegasusXPreTrainedModel",
        ]
    )
    _import_structure["models.perceiver"].extend(
        [
            "PerceiverForImageClassificationConvProcessing",
            "PerceiverForImageClassificationFourier",
            "PerceiverForImageClassificationLearned",
            "PerceiverForMaskedLM",
            "PerceiverForMultimodalAutoencoding",
            "PerceiverForOpticalFlow",
            "PerceiverForSequenceClassification",
            "PerceiverModel",
            "PerceiverPreTrainedModel",
        ]
    )
    _import_structure["models.persimmon"].extend(
        [
            "PersimmonForCausalLM",
            "PersimmonForSequenceClassification",
            "PersimmonForTokenClassification",
            "PersimmonModel",
            "PersimmonPreTrainedModel",
        ]
    )
    _import_structure["models.phi"].extend(
        [
            "PhiForCausalLM",
            "PhiForSequenceClassification",
            "PhiForTokenClassification",
            "PhiModel",
            "PhiPreTrainedModel",
        ]
    )
    _import_structure["models.phi3"].extend(
        [
            "Phi3ForCausalLM",
            "Phi3ForSequenceClassification",
            "Phi3ForTokenClassification",
            "Phi3Model",
            "Phi3PreTrainedModel",
        ]
    )
    _import_structure["models.phimoe"].extend(
        [
            "PhimoeForCausalLM",
            "PhimoeForSequenceClassification",
            "PhimoeModel",
            "PhimoePreTrainedModel",
        ]
    )
    _import_structure["models.pix2struct"].extend(
        [
            "Pix2StructForConditionalGeneration",
            "Pix2StructPreTrainedModel",
            "Pix2StructTextModel",
            "Pix2StructVisionModel",
        ]
    )
    _import_structure["models.pixtral"].extend(["PixtralPreTrainedModel", "PixtralVisionModel"])
    _import_structure["models.plbart"].extend(
        [
            "PLBartForCausalLM",
            "PLBartForConditionalGeneration",
            "PLBartForSequenceClassification",
            "PLBartModel",
            "PLBartPreTrainedModel",
        ]
    )
    _import_structure["models.poolformer"].extend(
        [
            "PoolFormerForImageClassification",
            "PoolFormerModel",
            "PoolFormerPreTrainedModel",
        ]
    )
    _import_structure["models.pop2piano"].extend(
        [
            "Pop2PianoForConditionalGeneration",
            "Pop2PianoPreTrainedModel",
        ]
    )
    _import_structure["models.prophetnet"].extend(
        [
            "ProphetNetDecoder",
            "ProphetNetEncoder",
            "ProphetNetForCausalLM",
            "ProphetNetForConditionalGeneration",
            "ProphetNetModel",
            "ProphetNetPreTrainedModel",
        ]
    )
    _import_structure["models.pvt"].extend(
        [
            "PvtForImageClassification",
            "PvtModel",
            "PvtPreTrainedModel",
        ]
    )
    _import_structure["models.pvt_v2"].extend(
        [
            "PvtV2Backbone",
            "PvtV2ForImageClassification",
            "PvtV2Model",
            "PvtV2PreTrainedModel",
        ]
    )
    _import_structure["models.qwen2"].extend(
        [
            "Qwen2ForCausalLM",
            "Qwen2ForQuestionAnswering",
            "Qwen2ForSequenceClassification",
            "Qwen2ForTokenClassification",
            "Qwen2Model",
            "Qwen2PreTrainedModel",
        ]
    )
    _import_structure["models.qwen2_audio"].extend(
        [
            "Qwen2AudioEncoder",
            "Qwen2AudioForConditionalGeneration",
            "Qwen2AudioPreTrainedModel",
        ]
    )
    _import_structure["models.qwen2_moe"].extend(
        [
            "Qwen2MoeForCausalLM",
            "Qwen2MoeForQuestionAnswering",
            "Qwen2MoeForSequenceClassification",
            "Qwen2MoeForTokenClassification",
            "Qwen2MoeModel",
            "Qwen2MoePreTrainedModel",
        ]
    )
    _import_structure["models.qwen2_vl"].extend(
        [
            "Qwen2VLForConditionalGeneration",
            "Qwen2VLModel",
            "Qwen2VLPreTrainedModel",
        ]
    )
    _import_structure["models.rag"].extend(
        [
            "RagModel",
            "RagPreTrainedModel",
            "RagSequenceForGeneration",
            "RagTokenForGeneration",
        ]
    )
    _import_structure["models.recurrent_gemma"].extend(
        [
            "RecurrentGemmaForCausalLM",
            "RecurrentGemmaModel",
            "RecurrentGemmaPreTrainedModel",
        ]
    )
    _import_structure["models.reformer"].extend(
        [
            "ReformerForMaskedLM",
            "ReformerForQuestionAnswering",
            "ReformerForSequenceClassification",
            "ReformerModel",
            "ReformerModelWithLMHead",
            "ReformerPreTrainedModel",
        ]
    )
    _import_structure["models.regnet"].extend(
        [
            "RegNetForImageClassification",
            "RegNetModel",
            "RegNetPreTrainedModel",
        ]
    )
    _import_structure["models.rembert"].extend(
        [
            "RemBertForCausalLM",
            "RemBertForMaskedLM",
            "RemBertForMultipleChoice",
            "RemBertForQuestionAnswering",
            "RemBertForSequenceClassification",
            "RemBertForTokenClassification",
            "RemBertModel",
            "RemBertPreTrainedModel",
            "load_tf_weights_in_rembert",
        ]
    )
    _import_structure["models.resnet"].extend(
        [
            "ResNetBackbone",
            "ResNetForImageClassification",
            "ResNetModel",
            "ResNetPreTrainedModel",
        ]
    )
    _import_structure["models.roberta"].extend(
        [
            "RobertaForCausalLM",
            "RobertaForMaskedLM",
            "RobertaForMultipleChoice",
            "RobertaForQuestionAnswering",
            "RobertaForSequenceClassification",
            "RobertaForTokenClassification",
            "RobertaModel",
            "RobertaPreTrainedModel",
        ]
    )
    _import_structure["models.roberta_prelayernorm"].extend(
        [
            "RobertaPreLayerNormForCausalLM",
            "RobertaPreLayerNormForMaskedLM",
            "RobertaPreLayerNormForMultipleChoice",
            "RobertaPreLayerNormForQuestionAnswering",
            "RobertaPreLayerNormForSequenceClassification",
            "RobertaPreLayerNormForTokenClassification",
            "RobertaPreLayerNormModel",
            "RobertaPreLayerNormPreTrainedModel",
        ]
    )
    _import_structure["models.roc_bert"].extend(
        [
            "RoCBertForCausalLM",
            "RoCBertForMaskedLM",
            "RoCBertForMultipleChoice",
            "RoCBertForPreTraining",
            "RoCBertForQuestionAnswering",
            "RoCBertForSequenceClassification",
            "RoCBertForTokenClassification",
            "RoCBertModel",
            "RoCBertPreTrainedModel",
            "load_tf_weights_in_roc_bert",
        ]
    )
    _import_structure["models.roformer"].extend(
        [
            "RoFormerForCausalLM",
            "RoFormerForMaskedLM",
            "RoFormerForMultipleChoice",
            "RoFormerForQuestionAnswering",
            "RoFormerForSequenceClassification",
            "RoFormerForTokenClassification",
            "RoFormerModel",
            "RoFormerPreTrainedModel",
            "load_tf_weights_in_roformer",
        ]
    )
    _import_structure["models.rt_detr"].extend(
        [
            "RTDetrForObjectDetection",
            "RTDetrModel",
            "RTDetrPreTrainedModel",
            "RTDetrResNetBackbone",
            "RTDetrResNetPreTrainedModel",
        ]
    )
    _import_structure["models.rwkv"].extend(
        [
            "RwkvForCausalLM",
            "RwkvModel",
            "RwkvPreTrainedModel",
        ]
    )
    _import_structure["models.sam"].extend(
        [
            "SamModel",
            "SamPreTrainedModel",
        ]
    )
    _import_structure["models.seamless_m4t"].extend(
        [
            "SeamlessM4TCodeHifiGan",
            "SeamlessM4TForSpeechToSpeech",
            "SeamlessM4TForSpeechToText",
            "SeamlessM4TForTextToSpeech",
            "SeamlessM4TForTextToText",
            "SeamlessM4THifiGan",
            "SeamlessM4TModel",
            "SeamlessM4TPreTrainedModel",
            "SeamlessM4TTextToUnitForConditionalGeneration",
            "SeamlessM4TTextToUnitModel",
        ]
    )
    _import_structure["models.seamless_m4t_v2"].extend(
        [
            "SeamlessM4Tv2ForSpeechToSpeech",
            "SeamlessM4Tv2ForSpeechToText",
            "SeamlessM4Tv2ForTextToSpeech",
            "SeamlessM4Tv2ForTextToText",
            "SeamlessM4Tv2Model",
            "SeamlessM4Tv2PreTrainedModel",
        ]
    )
    _import_structure["models.segformer"].extend(
        [
            "SegformerDecodeHead",
            "SegformerForImageClassification",
            "SegformerForSemanticSegmentation",
            "SegformerModel",
            "SegformerPreTrainedModel",
        ]
    )
    _import_structure["models.seggpt"].extend(
        [
            "SegGptForImageSegmentation",
            "SegGptModel",
            "SegGptPreTrainedModel",
        ]
    )
    _import_structure["models.sew"].extend(
        [
            "SEWForCTC",
            "SEWForSequenceClassification",
            "SEWModel",
            "SEWPreTrainedModel",
        ]
    )
    _import_structure["models.sew_d"].extend(
        [
            "SEWDForCTC",
            "SEWDForSequenceClassification",
            "SEWDModel",
            "SEWDPreTrainedModel",
        ]
    )
    _import_structure["models.siglip"].extend(
        [
            "SiglipForImageClassification",
            "SiglipModel",
            "SiglipPreTrainedModel",
            "SiglipTextModel",
            "SiglipVisionModel",
        ]
    )
    _import_structure["models.speech_encoder_decoder"].extend(["SpeechEncoderDecoderModel"])
    _import_structure["models.speech_to_text"].extend(
        [
            "Speech2TextForConditionalGeneration",
            "Speech2TextModel",
            "Speech2TextPreTrainedModel",
        ]
    )
    _import_structure["models.speecht5"].extend(
        [
            "SpeechT5ForSpeechToSpeech",
            "SpeechT5ForSpeechToText",
            "SpeechT5ForTextToSpeech",
            "SpeechT5HifiGan",
            "SpeechT5Model",
            "SpeechT5PreTrainedModel",
        ]
    )
    _import_structure["models.splinter"].extend(
        [
            "SplinterForPreTraining",
            "SplinterForQuestionAnswering",
            "SplinterModel",
            "SplinterPreTrainedModel",
        ]
    )
    _import_structure["models.squeezebert"].extend(
        [
            "SqueezeBertForMaskedLM",
            "SqueezeBertForMultipleChoice",
            "SqueezeBertForQuestionAnswering",
            "SqueezeBertForSequenceClassification",
            "SqueezeBertForTokenClassification",
            "SqueezeBertModel",
            "SqueezeBertPreTrainedModel",
        ]
    )
    _import_structure["models.stablelm"].extend(
        [
            "StableLmForCausalLM",
            "StableLmForSequenceClassification",
            "StableLmForTokenClassification",
            "StableLmModel",
            "StableLmPreTrainedModel",
        ]
    )
    _import_structure["models.starcoder2"].extend(
        [
            "Starcoder2ForCausalLM",
            "Starcoder2ForSequenceClassification",
            "Starcoder2ForTokenClassification",
            "Starcoder2Model",
            "Starcoder2PreTrainedModel",
        ]
    )
    _import_structure["models.superpoint"].extend(
        [
            "SuperPointForKeypointDetection",
            "SuperPointPreTrainedModel",
        ]
    )
    _import_structure["models.swiftformer"].extend(
        [
            "SwiftFormerForImageClassification",
            "SwiftFormerModel",
            "SwiftFormerPreTrainedModel",
        ]
    )
    _import_structure["models.swin"].extend(
        [
            "SwinBackbone",
            "SwinForImageClassification",
            "SwinForMaskedImageModeling",
            "SwinModel",
            "SwinPreTrainedModel",
        ]
    )
    _import_structure["models.swin2sr"].extend(
        [
            "Swin2SRForImageSuperResolution",
            "Swin2SRModel",
            "Swin2SRPreTrainedModel",
        ]
    )
    _import_structure["models.swinv2"].extend(
        [
            "Swinv2Backbone",
            "Swinv2ForImageClassification",
            "Swinv2ForMaskedImageModeling",
            "Swinv2Model",
            "Swinv2PreTrainedModel",
        ]
    )
    _import_structure["models.switch_transformers"].extend(
        [
            "SwitchTransformersEncoderModel",
            "SwitchTransformersForConditionalGeneration",
            "SwitchTransformersModel",
            "SwitchTransformersPreTrainedModel",
            "SwitchTransformersSparseMLP",
            "SwitchTransformersTop1Router",
        ]
    )
    _import_structure["models.t5"].extend(
        [
            "T5EncoderModel",
            "T5ForConditionalGeneration",
            "T5ForQuestionAnswering",
            "T5ForSequenceClassification",
            "T5ForTokenClassification",
            "T5Model",
            "T5PreTrainedModel",
            "load_tf_weights_in_t5",
        ]
    )
    _import_structure["models.table_transformer"].extend(
        [
            "TableTransformerForObjectDetection",
            "TableTransformerModel",
            "TableTransformerPreTrainedModel",
        ]
    )
    _import_structure["models.tapas"].extend(
        [
            "TapasForMaskedLM",
            "TapasForQuestionAnswering",
            "TapasForSequenceClassification",
            "TapasModel",
            "TapasPreTrainedModel",
            "load_tf_weights_in_tapas",
        ]
    )
    _import_structure["models.time_series_transformer"].extend(
        [
            "TimeSeriesTransformerForPrediction",
            "TimeSeriesTransformerModel",
            "TimeSeriesTransformerPreTrainedModel",
        ]
    )
    _import_structure["models.timesformer"].extend(
        [
            "TimesformerForVideoClassification",
            "TimesformerModel",
            "TimesformerPreTrainedModel",
        ]
    )
    _import_structure["models.timm_backbone"].extend(["TimmBackbone"])
    _import_structure["models.timm_wrapper"].extend(
        ["TimmWrapperForImageClassification", "TimmWrapperModel", "TimmWrapperPreTrainedModel"]
    )
    _import_structure["models.trocr"].extend(
        [
            "TrOCRForCausalLM",
            "TrOCRPreTrainedModel",
        ]
    )
    _import_structure["models.tvp"].extend(
        [
            "TvpForVideoGrounding",
            "TvpModel",
            "TvpPreTrainedModel",
        ]
    )
    _import_structure["models.udop"].extend(
        [
            "UdopEncoderModel",
            "UdopForConditionalGeneration",
            "UdopModel",
            "UdopPreTrainedModel",
        ],
    )
    _import_structure["models.umt5"].extend(
        [
            "UMT5EncoderModel",
            "UMT5ForConditionalGeneration",
            "UMT5ForQuestionAnswering",
            "UMT5ForSequenceClassification",
            "UMT5ForTokenClassification",
            "UMT5Model",
            "UMT5PreTrainedModel",
        ]
    )
    _import_structure["models.unispeech"].extend(
        [
            "UniSpeechForCTC",
            "UniSpeechForPreTraining",
            "UniSpeechForSequenceClassification",
            "UniSpeechModel",
            "UniSpeechPreTrainedModel",
        ]
    )
    _import_structure["models.unispeech_sat"].extend(
        [
            "UniSpeechSatForAudioFrameClassification",
            "UniSpeechSatForCTC",
            "UniSpeechSatForPreTraining",
            "UniSpeechSatForSequenceClassification",
            "UniSpeechSatForXVector",
            "UniSpeechSatModel",
            "UniSpeechSatPreTrainedModel",
        ]
    )
    _import_structure["models.univnet"].extend(
        [
            "UnivNetModel",
        ]
    )
    _import_structure["models.upernet"].extend(
        [
            "UperNetForSemanticSegmentation",
            "UperNetPreTrainedModel",
        ]
    )
    _import_structure["models.video_llava"].extend(
        [
            "VideoLlavaForConditionalGeneration",
            "VideoLlavaPreTrainedModel",
            "VideoLlavaProcessor",
        ]
    )
    _import_structure["models.videomae"].extend(
        [
            "VideoMAEForPreTraining",
            "VideoMAEForVideoClassification",
            "VideoMAEModel",
            "VideoMAEPreTrainedModel",
        ]
    )
    _import_structure["models.vilt"].extend(
        [
            "ViltForImageAndTextRetrieval",
            "ViltForImagesAndTextClassification",
            "ViltForMaskedLM",
            "ViltForQuestionAnswering",
            "ViltForTokenClassification",
            "ViltModel",
            "ViltPreTrainedModel",
        ]
    )
    _import_structure["models.vipllava"].extend(
        [
            "VipLlavaForConditionalGeneration",
            "VipLlavaPreTrainedModel",
        ]
    )
    _import_structure["models.vision_encoder_decoder"].extend(["VisionEncoderDecoderModel"])
    _import_structure["models.vision_text_dual_encoder"].extend(["VisionTextDualEncoderModel"])
    _import_structure["models.visual_bert"].extend(
        [
            "VisualBertForMultipleChoice",
            "VisualBertForPreTraining",
            "VisualBertForQuestionAnswering",
            "VisualBertForRegionToPhraseAlignment",
            "VisualBertForVisualReasoning",
            "VisualBertModel",
            "VisualBertPreTrainedModel",
        ]
    )
    _import_structure["models.vit"].extend(
        [
            "ViTForImageClassification",
            "ViTForMaskedImageModeling",
            "ViTModel",
            "ViTPreTrainedModel",
        ]
    )
    _import_structure["models.vit_mae"].extend(
        [
            "ViTMAEForPreTraining",
            "ViTMAEModel",
            "ViTMAEPreTrainedModel",
        ]
    )
    _import_structure["models.vit_msn"].extend(
        [
            "ViTMSNForImageClassification",
            "ViTMSNModel",
            "ViTMSNPreTrainedModel",
        ]
    )
    _import_structure["models.vitdet"].extend(
        [
            "VitDetBackbone",
            "VitDetModel",
            "VitDetPreTrainedModel",
        ]
    )
    _import_structure["models.vitmatte"].extend(
        [
            "VitMatteForImageMatting",
            "VitMattePreTrainedModel",
        ]
    )
    _import_structure["models.vits"].extend(
        [
            "VitsModel",
            "VitsPreTrainedModel",
        ]
    )
    _import_structure["models.vivit"].extend(
        [
            "VivitForVideoClassification",
            "VivitModel",
            "VivitPreTrainedModel",
        ]
    )
    _import_structure["models.wav2vec2"].extend(
        [
            "Wav2Vec2ForAudioFrameClassification",
            "Wav2Vec2ForCTC",
            "Wav2Vec2ForMaskedLM",
            "Wav2Vec2ForPreTraining",
            "Wav2Vec2ForSequenceClassification",
            "Wav2Vec2ForXVector",
            "Wav2Vec2Model",
            "Wav2Vec2PreTrainedModel",
        ]
    )
    _import_structure["models.wav2vec2_bert"].extend(
        [
            "Wav2Vec2BertForAudioFrameClassification",
            "Wav2Vec2BertForCTC",
            "Wav2Vec2BertForSequenceClassification",
            "Wav2Vec2BertForXVector",
            "Wav2Vec2BertModel",
            "Wav2Vec2BertPreTrainedModel",
        ]
    )
    _import_structure["models.wav2vec2_conformer"].extend(
        [
            "Wav2Vec2ConformerForAudioFrameClassification",
            "Wav2Vec2ConformerForCTC",
            "Wav2Vec2ConformerForPreTraining",
            "Wav2Vec2ConformerForSequenceClassification",
            "Wav2Vec2ConformerForXVector",
            "Wav2Vec2ConformerModel",
            "Wav2Vec2ConformerPreTrainedModel",
        ]
    )
    _import_structure["models.wavlm"].extend(
        [
            "WavLMForAudioFrameClassification",
            "WavLMForCTC",
            "WavLMForSequenceClassification",
            "WavLMForXVector",
            "WavLMModel",
            "WavLMPreTrainedModel",
        ]
    )
    _import_structure["models.whisper"].extend(
        [
            "WhisperForAudioClassification",
            "WhisperForCausalLM",
            "WhisperForConditionalGeneration",
            "WhisperModel",
            "WhisperPreTrainedModel",
        ]
    )
    _import_structure["models.x_clip"].extend(
        [
            "XCLIPModel",
            "XCLIPPreTrainedModel",
            "XCLIPTextModel",
            "XCLIPVisionModel",
        ]
    )
    _import_structure["models.xglm"].extend(
        [
            "XGLMForCausalLM",
            "XGLMModel",
            "XGLMPreTrainedModel",
        ]
    )
    _import_structure["models.xlm"].extend(
        [
            "XLMForMultipleChoice",
            "XLMForQuestionAnswering",
            "XLMForQuestionAnsweringSimple",
            "XLMForSequenceClassification",
            "XLMForTokenClassification",
            "XLMModel",
            "XLMPreTrainedModel",
            "XLMWithLMHeadModel",
        ]
    )
    _import_structure["models.xlm_roberta"].extend(
        [
            "XLMRobertaForCausalLM",
            "XLMRobertaForMaskedLM",
            "XLMRobertaForMultipleChoice",
            "XLMRobertaForQuestionAnswering",
            "XLMRobertaForSequenceClassification",
            "XLMRobertaForTokenClassification",
            "XLMRobertaModel",
            "XLMRobertaPreTrainedModel",
        ]
    )
    _import_structure["models.xlm_roberta_xl"].extend(
        [
            "XLMRobertaXLForCausalLM",
            "XLMRobertaXLForMaskedLM",
            "XLMRobertaXLForMultipleChoice",
            "XLMRobertaXLForQuestionAnswering",
            "XLMRobertaXLForSequenceClassification",
            "XLMRobertaXLForTokenClassification",
            "XLMRobertaXLModel",
            "XLMRobertaXLPreTrainedModel",
        ]
    )
    _import_structure["models.xlnet"].extend(
        [
            "XLNetForMultipleChoice",
            "XLNetForQuestionAnswering",
            "XLNetForQuestionAnsweringSimple",
            "XLNetForSequenceClassification",
            "XLNetForTokenClassification",
            "XLNetLMHeadModel",
            "XLNetModel",
            "XLNetPreTrainedModel",
            "load_tf_weights_in_xlnet",
        ]
    )
    _import_structure["models.xmod"].extend(
        [
            "XmodForCausalLM",
            "XmodForMaskedLM",
            "XmodForMultipleChoice",
            "XmodForQuestionAnswering",
            "XmodForSequenceClassification",
            "XmodForTokenClassification",
            "XmodModel",
            "XmodPreTrainedModel",
        ]
    )
    _import_structure["models.yolos"].extend(
        [
            "YolosForObjectDetection",
            "YolosModel",
            "YolosPreTrainedModel",
        ]
    )
    _import_structure["models.yoso"].extend(
        [
            "YosoForMaskedLM",
            "YosoForMultipleChoice",
            "YosoForQuestionAnswering",
            "YosoForSequenceClassification",
            "YosoForTokenClassification",
            "YosoModel",
            "YosoPreTrainedModel",
        ]
    )
    _import_structure["models.zamba"].extend(
        [
            "ZambaForCausalLM",
            "ZambaForSequenceClassification",
            "ZambaModel",
            "ZambaPreTrainedModel",
        ]
    )
    _import_structure["models.zoedepth"].extend(
        [
            "ZoeDepthForDepthEstimation",
            "ZoeDepthPreTrainedModel",
        ]
    )
    _import_structure["optimization"] = [
        "Adafactor",
        "AdamW",
        "get_constant_schedule",
        "get_constant_schedule_with_warmup",
        "get_cosine_schedule_with_warmup",
        "get_cosine_with_hard_restarts_schedule_with_warmup",
        "get_inverse_sqrt_schedule",
        "get_linear_schedule_with_warmup",
        "get_polynomial_decay_schedule_with_warmup",
        "get_scheduler",
        "get_wsd_schedule",
    ]
    _import_structure["pytorch_utils"] = [
        "Conv1D",
        "apply_chunking_to_forward",
        "prune_layer",
    ]
    _import_structure["sagemaker"] = []
    _import_structure["time_series_utils"] = []
    _import_structure["trainer"] = ["Trainer"]
    _import_structure["trainer_pt_utils"] = ["torch_distributed_zero_first"]
    _import_structure["trainer_seq2seq"] = ["Seq2SeqTrainer"]

# TensorFlow-backed objects
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_tf_objects

    _import_structure["utils.dummy_tf_objects"] = [name for name in dir(dummy_tf_objects) if not name.startswith("_")]
else:
    _import_structure["activations_tf"] = []
    _import_structure["benchmark.benchmark_args_tf"] = ["TensorFlowBenchmarkArguments"]
    _import_structure["benchmark.benchmark_tf"] = ["TensorFlowBenchmark"]
    _import_structure["generation"].extend(
        [
            "TFForcedBOSTokenLogitsProcessor",
            "TFForcedEOSTokenLogitsProcessor",
            "TFForceTokensLogitsProcessor",
            "TFGenerationMixin",
            "TFLogitsProcessor",
            "TFLogitsProcessorList",
            "TFLogitsWarper",
            "TFMinLengthLogitsProcessor",
            "TFNoBadWordsLogitsProcessor",
            "TFNoRepeatNGramLogitsProcessor",
            "TFRepetitionPenaltyLogitsProcessor",
            "TFSuppressTokensAtBeginLogitsProcessor",
            "TFSuppressTokensLogitsProcessor",
            "TFTemperatureLogitsWarper",
            "TFTopKLogitsWarper",
            "TFTopPLogitsWarper",
        ]
    )
    _import_structure["keras_callbacks"] = ["KerasMetricCallback", "PushToHubCallback"]
    _import_structure["modeling_tf_outputs"] = []
    _import_structure["modeling_tf_utils"] = [
        "TFPreTrainedModel",
        "TFSequenceSummary",
        "TFSharedEmbeddings",
        "shape_list",
    ]
    # TensorFlow models structure
    _import_structure["models.albert"].extend(
        [
            "TFAlbertForMaskedLM",
            "TFAlbertForMultipleChoice",
            "TFAlbertForPreTraining",
            "TFAlbertForQuestionAnswering",
            "TFAlbertForSequenceClassification",
            "TFAlbertForTokenClassification",
            "TFAlbertMainLayer",
            "TFAlbertModel",
            "TFAlbertPreTrainedModel",
        ]
    )
    _import_structure["models.auto"].extend(
        [
            "TF_MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING",
            "TF_MODEL_FOR_CAUSAL_LM_MAPPING",
            "TF_MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING",
            "TF_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING",
            "TF_MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING",
            "TF_MODEL_FOR_MASKED_LM_MAPPING",
            "TF_MODEL_FOR_MASK_GENERATION_MAPPING",
            "TF_MODEL_FOR_MULTIPLE_CHOICE_MAPPING",
            "TF_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING",
            "TF_MODEL_FOR_PRETRAINING_MAPPING",
            "TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING",
            "TF_MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING",
            "TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING",
            "TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING",
            "TF_MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING",
            "TF_MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING",
            "TF_MODEL_FOR_TEXT_ENCODING_MAPPING",
            "TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING",
            "TF_MODEL_FOR_VISION_2_SEQ_MAPPING",
            "TF_MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING",
            "TF_MODEL_MAPPING",
            "TF_MODEL_WITH_LM_HEAD_MAPPING",
            "TFAutoModel",
            "TFAutoModelForAudioClassification",
            "TFAutoModelForCausalLM",
            "TFAutoModelForDocumentQuestionAnswering",
            "TFAutoModelForImageClassification",
            "TFAutoModelForMaskedImageModeling",
            "TFAutoModelForMaskedLM",
            "TFAutoModelForMaskGeneration",
            "TFAutoModelForMultipleChoice",
            "TFAutoModelForNextSentencePrediction",
            "TFAutoModelForPreTraining",
            "TFAutoModelForQuestionAnswering",
            "TFAutoModelForSemanticSegmentation",
            "TFAutoModelForSeq2SeqLM",
            "TFAutoModelForSequenceClassification",
            "TFAutoModelForSpeechSeq2Seq",
            "TFAutoModelForTableQuestionAnswering",
            "TFAutoModelForTextEncoding",
            "TFAutoModelForTokenClassification",
            "TFAutoModelForVision2Seq",
            "TFAutoModelForZeroShotImageClassification",
            "TFAutoModelWithLMHead",
        ]
    )
    _import_structure["models.bart"].extend(
        [
            "TFBartForConditionalGeneration",
            "TFBartForSequenceClassification",
            "TFBartModel",
            "TFBartPretrainedModel",
        ]
    )
    _import_structure["models.bert"].extend(
        [
            "TFBertForMaskedLM",
            "TFBertForMultipleChoice",
            "TFBertForNextSentencePrediction",
            "TFBertForPreTraining",
            "TFBertForQuestionAnswering",
            "TFBertForSequenceClassification",
            "TFBertForTokenClassification",
            "TFBertLMHeadModel",
            "TFBertMainLayer",
            "TFBertModel",
            "TFBertPreTrainedModel",
        ]
    )
    _import_structure["models.blenderbot"].extend(
        [
            "TFBlenderbotForConditionalGeneration",
            "TFBlenderbotModel",
            "TFBlenderbotPreTrainedModel",
        ]
    )
    _import_structure["models.blenderbot_small"].extend(
        [
            "TFBlenderbotSmallForConditionalGeneration",
            "TFBlenderbotSmallModel",
            "TFBlenderbotSmallPreTrainedModel",
        ]
    )
    _import_structure["models.blip"].extend(
        [
            "TFBlipForConditionalGeneration",
            "TFBlipForImageTextRetrieval",
            "TFBlipForQuestionAnswering",
            "TFBlipModel",
            "TFBlipPreTrainedModel",
            "TFBlipTextModel",
            "TFBlipVisionModel",
        ]
    )
    _import_structure["models.camembert"].extend(
        [
            "TFCamembertForCausalLM",
            "TFCamembertForMaskedLM",
            "TFCamembertForMultipleChoice",
            "TFCamembertForQuestionAnswering",
            "TFCamembertForSequenceClassification",
            "TFCamembertForTokenClassification",
            "TFCamembertModel",
            "TFCamembertPreTrainedModel",
        ]
    )
    _import_structure["models.clip"].extend(
        [
            "TFCLIPModel",
            "TFCLIPPreTrainedModel",
            "TFCLIPTextModel",
            "TFCLIPVisionModel",
        ]
    )
    _import_structure["models.convbert"].extend(
        [
            "TFConvBertForMaskedLM",
            "TFConvBertForMultipleChoice",
            "TFConvBertForQuestionAnswering",
            "TFConvBertForSequenceClassification",
            "TFConvBertForTokenClassification",
            "TFConvBertModel",
            "TFConvBertPreTrainedModel",
        ]
    )
    _import_structure["models.convnext"].extend(
        [
            "TFConvNextForImageClassification",
            "TFConvNextModel",
            "TFConvNextPreTrainedModel",
        ]
    )
    _import_structure["models.convnextv2"].extend(
        [
            "TFConvNextV2ForImageClassification",
            "TFConvNextV2Model",
            "TFConvNextV2PreTrainedModel",
        ]
    )
    _import_structure["models.ctrl"].extend(
        [
            "TFCTRLForSequenceClassification",
            "TFCTRLLMHeadModel",
            "TFCTRLModel",
            "TFCTRLPreTrainedModel",
        ]
    )
    _import_structure["models.cvt"].extend(
        [
            "TFCvtForImageClassification",
            "TFCvtModel",
            "TFCvtPreTrainedModel",
        ]
    )
    _import_structure["models.data2vec"].extend(
        [
            "TFData2VecVisionForImageClassification",
            "TFData2VecVisionForSemanticSegmentation",
            "TFData2VecVisionModel",
            "TFData2VecVisionPreTrainedModel",
        ]
    )
    _import_structure["models.deberta"].extend(
        [
            "TFDebertaForMaskedLM",
            "TFDebertaForQuestionAnswering",
            "TFDebertaForSequenceClassification",
            "TFDebertaForTokenClassification",
            "TFDebertaModel",
            "TFDebertaPreTrainedModel",
        ]
    )
    _import_structure["models.deberta_v2"].extend(
        [
            "TFDebertaV2ForMaskedLM",
            "TFDebertaV2ForMultipleChoice",
            "TFDebertaV2ForQuestionAnswering",
            "TFDebertaV2ForSequenceClassification",
            "TFDebertaV2ForTokenClassification",
            "TFDebertaV2Model",
            "TFDebertaV2PreTrainedModel",
        ]
    )
    _import_structure["models.deit"].extend(
        [
            "TFDeiTForImageClassification",
            "TFDeiTForImageClassificationWithTeacher",
            "TFDeiTForMaskedImageModeling",
            "TFDeiTModel",
            "TFDeiTPreTrainedModel",
        ]
    )
    _import_structure["models.deprecated.efficientformer"].extend(
        [
            "TFEfficientFormerForImageClassification",
            "TFEfficientFormerForImageClassificationWithTeacher",
            "TFEfficientFormerModel",
            "TFEfficientFormerPreTrainedModel",
        ]
    )
    _import_structure["models.deprecated.transfo_xl"].extend(
        [
            "TFAdaptiveEmbedding",
            "TFTransfoXLForSequenceClassification",
            "TFTransfoXLLMHeadModel",
            "TFTransfoXLMainLayer",
            "TFTransfoXLModel",
            "TFTransfoXLPreTrainedModel",
        ]
    )
    _import_structure["models.distilbert"].extend(
        [
            "TFDistilBertForMaskedLM",
            "TFDistilBertForMultipleChoice",
            "TFDistilBertForQuestionAnswering",
            "TFDistilBertForSequenceClassification",
            "TFDistilBertForTokenClassification",
            "TFDistilBertMainLayer",
            "TFDistilBertModel",
            "TFDistilBertPreTrainedModel",
        ]
    )
    _import_structure["models.dpr"].extend(
        [
            "TFDPRContextEncoder",
            "TFDPRPretrainedContextEncoder",
            "TFDPRPretrainedQuestionEncoder",
            "TFDPRPretrainedReader",
            "TFDPRQuestionEncoder",
            "TFDPRReader",
        ]
    )
    _import_structure["models.electra"].extend(
        [
            "TFElectraForMaskedLM",
            "TFElectraForMultipleChoice",
            "TFElectraForPreTraining",
            "TFElectraForQuestionAnswering",
            "TFElectraForSequenceClassification",
            "TFElectraForTokenClassification",
            "TFElectraModel",
            "TFElectraPreTrainedModel",
        ]
    )
    _import_structure["models.encoder_decoder"].append("TFEncoderDecoderModel")
    _import_structure["models.esm"].extend(
        [
            "TFEsmForMaskedLM",
            "TFEsmForSequenceClassification",
            "TFEsmForTokenClassification",
            "TFEsmModel",
            "TFEsmPreTrainedModel",
        ]
    )
    _import_structure["models.flaubert"].extend(
        [
            "TFFlaubertForMultipleChoice",
            "TFFlaubertForQuestionAnsweringSimple",
            "TFFlaubertForSequenceClassification",
            "TFFlaubertForTokenClassification",
            "TFFlaubertModel",
            "TFFlaubertPreTrainedModel",
            "TFFlaubertWithLMHeadModel",
        ]
    )
    _import_structure["models.funnel"].extend(
        [
            "TFFunnelBaseModel",
            "TFFunnelForMaskedLM",
            "TFFunnelForMultipleChoice",
            "TFFunnelForPreTraining",
            "TFFunnelForQuestionAnswering",
            "TFFunnelForSequenceClassification",
            "TFFunnelForTokenClassification",
            "TFFunnelModel",
            "TFFunnelPreTrainedModel",
        ]
    )
    _import_structure["models.gpt2"].extend(
        [
            "TFGPT2DoubleHeadsModel",
            "TFGPT2ForSequenceClassification",
            "TFGPT2LMHeadModel",
            "TFGPT2MainLayer",
            "TFGPT2Model",
            "TFGPT2PreTrainedModel",
        ]
    )
    _import_structure["models.gptj"].extend(
        [
            "TFGPTJForCausalLM",
            "TFGPTJForQuestionAnswering",
            "TFGPTJForSequenceClassification",
            "TFGPTJModel",
            "TFGPTJPreTrainedModel",
        ]
    )
    _import_structure["models.groupvit"].extend(
        [
            "TFGroupViTModel",
            "TFGroupViTPreTrainedModel",
            "TFGroupViTTextModel",
            "TFGroupViTVisionModel",
        ]
    )
    _import_structure["models.hubert"].extend(
        [
            "TFHubertForCTC",
            "TFHubertModel",
            "TFHubertPreTrainedModel",
        ]
    )

    _import_structure["models.idefics"].extend(
        [
            "TFIdeficsForVisionText2Text",
            "TFIdeficsModel",
            "TFIdeficsPreTrainedModel",
        ]
    )

    _import_structure["models.layoutlm"].extend(
        [
            "TFLayoutLMForMaskedLM",
            "TFLayoutLMForQuestionAnswering",
            "TFLayoutLMForSequenceClassification",
            "TFLayoutLMForTokenClassification",
            "TFLayoutLMMainLayer",
            "TFLayoutLMModel",
            "TFLayoutLMPreTrainedModel",
        ]
    )
    _import_structure["models.layoutlmv3"].extend(
        [
            "TFLayoutLMv3ForQuestionAnswering",
            "TFLayoutLMv3ForSequenceClassification",
            "TFLayoutLMv3ForTokenClassification",
            "TFLayoutLMv3Model",
            "TFLayoutLMv3PreTrainedModel",
        ]
    )
    _import_structure["models.led"].extend(["TFLEDForConditionalGeneration", "TFLEDModel", "TFLEDPreTrainedModel"])
    _import_structure["models.longformer"].extend(
        [
            "TFLongformerForMaskedLM",
            "TFLongformerForMultipleChoice",
            "TFLongformerForQuestionAnswering",
            "TFLongformerForSequenceClassification",
            "TFLongformerForTokenClassification",
            "TFLongformerModel",
            "TFLongformerPreTrainedModel",
        ]
    )
    _import_structure["models.lxmert"].extend(
        [
            "TFLxmertForPreTraining",
            "TFLxmertMainLayer",
            "TFLxmertModel",
            "TFLxmertPreTrainedModel",
            "TFLxmertVisualFeatureEncoder",
        ]
    )
    _import_structure["models.marian"].extend(["TFMarianModel", "TFMarianMTModel", "TFMarianPreTrainedModel"])
    _import_structure["models.mbart"].extend(
        ["TFMBartForConditionalGeneration", "TFMBartModel", "TFMBartPreTrainedModel"]
    )
    _import_structure["models.mistral"].extend(
        ["TFMistralForCausalLM", "TFMistralForSequenceClassification", "TFMistralModel", "TFMistralPreTrainedModel"]
    )
    _import_structure["models.mobilebert"].extend(
        [
            "TFMobileBertForMaskedLM",
            "TFMobileBertForMultipleChoice",
            "TFMobileBertForNextSentencePrediction",
            "TFMobileBertForPreTraining",
            "TFMobileBertForQuestionAnswering",
            "TFMobileBertForSequenceClassification",
            "TFMobileBertForTokenClassification",
            "TFMobileBertMainLayer",
            "TFMobileBertModel",
            "TFMobileBertPreTrainedModel",
        ]
    )
    _import_structure["models.mobilevit"].extend(
        [
            "TFMobileViTForImageClassification",
            "TFMobileViTForSemanticSegmentation",
            "TFMobileViTModel",
            "TFMobileViTPreTrainedModel",
        ]
    )
    _import_structure["models.mpnet"].extend(
        [
            "TFMPNetForMaskedLM",
            "TFMPNetForMultipleChoice",
            "TFMPNetForQuestionAnswering",
            "TFMPNetForSequenceClassification",
            "TFMPNetForTokenClassification",
            "TFMPNetMainLayer",
            "TFMPNetModel",
            "TFMPNetPreTrainedModel",
        ]
    )
    _import_structure["models.mt5"].extend(["TFMT5EncoderModel", "TFMT5ForConditionalGeneration", "TFMT5Model"])
    _import_structure["models.openai"].extend(
        [
            "TFOpenAIGPTDoubleHeadsModel",
            "TFOpenAIGPTForSequenceClassification",
            "TFOpenAIGPTLMHeadModel",
            "TFOpenAIGPTMainLayer",
            "TFOpenAIGPTModel",
            "TFOpenAIGPTPreTrainedModel",
        ]
    )
    _import_structure["models.opt"].extend(
        [
            "TFOPTForCausalLM",
            "TFOPTModel",
            "TFOPTPreTrainedModel",
        ]
    )
    _import_structure["models.pegasus"].extend(
        [
            "TFPegasusForConditionalGeneration",
            "TFPegasusModel",
            "TFPegasusPreTrainedModel",
        ]
    )
    _import_structure["models.rag"].extend(
        [
            "TFRagModel",
            "TFRagPreTrainedModel",
            "TFRagSequenceForGeneration",
            "TFRagTokenForGeneration",
        ]
    )
    _import_structure["models.regnet"].extend(
        [
            "TFRegNetForImageClassification",
            "TFRegNetModel",
            "TFRegNetPreTrainedModel",
        ]
    )
    _import_structure["models.rembert"].extend(
        [
            "TFRemBertForCausalLM",
            "TFRemBertForMaskedLM",
            "TFRemBertForMultipleChoice",
            "TFRemBertForQuestionAnswering",
            "TFRemBertForSequenceClassification",
            "TFRemBertForTokenClassification",
            "TFRemBertModel",
            "TFRemBertPreTrainedModel",
        ]
    )
    _import_structure["models.resnet"].extend(
        [
            "TFResNetForImageClassification",
            "TFResNetModel",
            "TFResNetPreTrainedModel",
        ]
    )
    _import_structure["models.roberta"].extend(
        [
            "TFRobertaForCausalLM",
            "TFRobertaForMaskedLM",
            "TFRobertaForMultipleChoice",
            "TFRobertaForQuestionAnswering",
            "TFRobertaForSequenceClassification",
            "TFRobertaForTokenClassification",
            "TFRobertaMainLayer",
            "TFRobertaModel",
            "TFRobertaPreTrainedModel",
        ]
    )
    _import_structure["models.roberta_prelayernorm"].extend(
        [
            "TFRobertaPreLayerNormForCausalLM",
            "TFRobertaPreLayerNormForMaskedLM",
            "TFRobertaPreLayerNormForMultipleChoice",
            "TFRobertaPreLayerNormForQuestionAnswering",
            "TFRobertaPreLayerNormForSequenceClassification",
            "TFRobertaPreLayerNormForTokenClassification",
            "TFRobertaPreLayerNormMainLayer",
            "TFRobertaPreLayerNormModel",
            "TFRobertaPreLayerNormPreTrainedModel",
        ]
    )
    _import_structure["models.roformer"].extend(
        [
            "TFRoFormerForCausalLM",
            "TFRoFormerForMaskedLM",
            "TFRoFormerForMultipleChoice",
            "TFRoFormerForQuestionAnswering",
            "TFRoFormerForSequenceClassification",
            "TFRoFormerForTokenClassification",
            "TFRoFormerModel",
            "TFRoFormerPreTrainedModel",
        ]
    )
    _import_structure["models.sam"].extend(
        [
            "TFSamModel",
            "TFSamPreTrainedModel",
        ]
    )
    _import_structure["models.segformer"].extend(
        [
            "TFSegformerDecodeHead",
            "TFSegformerForImageClassification",
            "TFSegformerForSemanticSegmentation",
            "TFSegformerModel",
            "TFSegformerPreTrainedModel",
        ]
    )
    _import_structure["models.speech_to_text"].extend(
        [
            "TFSpeech2TextForConditionalGeneration",
            "TFSpeech2TextModel",
            "TFSpeech2TextPreTrainedModel",
        ]
    )
    _import_structure["models.swiftformer"].extend(
        [
            "TFSwiftFormerForImageClassification",
            "TFSwiftFormerModel",
            "TFSwiftFormerPreTrainedModel",
        ]
    )
    _import_structure["models.swin"].extend(
        [
            "TFSwinForImageClassification",
            "TFSwinForMaskedImageModeling",
            "TFSwinModel",
            "TFSwinPreTrainedModel",
        ]
    )
    _import_structure["models.t5"].extend(
        [
            "TFT5EncoderModel",
            "TFT5ForConditionalGeneration",
            "TFT5Model",
            "TFT5PreTrainedModel",
        ]
    )
    _import_structure["models.tapas"].extend(
        [
            "TFTapasForMaskedLM",
            "TFTapasForQuestionAnswering",
            "TFTapasForSequenceClassification",
            "TFTapasModel",
            "TFTapasPreTrainedModel",
        ]
    )
    _import_structure["models.vision_encoder_decoder"].extend(["TFVisionEncoderDecoderModel"])
    _import_structure["models.vision_text_dual_encoder"].extend(["TFVisionTextDualEncoderModel"])
    _import_structure["models.vit"].extend(
        [
            "TFViTForImageClassification",
            "TFViTModel",
            "TFViTPreTrainedModel",
        ]
    )
    _import_structure["models.vit_mae"].extend(
        [
            "TFViTMAEForPreTraining",
            "TFViTMAEModel",
            "TFViTMAEPreTrainedModel",
        ]
    )
    _import_structure["models.wav2vec2"].extend(
        [
            "TFWav2Vec2ForCTC",
            "TFWav2Vec2ForSequenceClassification",
            "TFWav2Vec2Model",
            "TFWav2Vec2PreTrainedModel",
        ]
    )
    _import_structure["models.whisper"].extend(
        [
            "TFWhisperForConditionalGeneration",
            "TFWhisperModel",
            "TFWhisperPreTrainedModel",
        ]
    )
    _import_structure["models.xglm"].extend(
        [
            "TFXGLMForCausalLM",
            "TFXGLMModel",
            "TFXGLMPreTrainedModel",
        ]
    )
    _import_structure["models.xlm"].extend(
        [
            "TFXLMForMultipleChoice",
            "TFXLMForQuestionAnsweringSimple",
            "TFXLMForSequenceClassification",
            "TFXLMForTokenClassification",
            "TFXLMMainLayer",
            "TFXLMModel",
            "TFXLMPreTrainedModel",
            "TFXLMWithLMHeadModel",
        ]
    )
    _import_structure["models.xlm_roberta"].extend(
        [
            "TFXLMRobertaForCausalLM",
            "TFXLMRobertaForMaskedLM",
            "TFXLMRobertaForMultipleChoice",
            "TFXLMRobertaForQuestionAnswering",
            "TFXLMRobertaForSequenceClassification",
            "TFXLMRobertaForTokenClassification",
            "TFXLMRobertaModel",
            "TFXLMRobertaPreTrainedModel",
        ]
    )
    _import_structure["models.xlnet"].extend(
        [
            "TFXLNetForMultipleChoice",
            "TFXLNetForQuestionAnsweringSimple",
            "TFXLNetForSequenceClassification",
            "TFXLNetForTokenClassification",
            "TFXLNetLMHeadModel",
            "TFXLNetMainLayer",
            "TFXLNetModel",
            "TFXLNetPreTrainedModel",
        ]
    )
    _import_structure["optimization_tf"] = [
        "AdamWeightDecay",
        "GradientAccumulator",
        "WarmUp",
        "create_optimizer",
    ]
    _import_structure["tf_utils"] = []


try:
    if not (
        is_librosa_available()
        and is_essentia_available()
        and is_scipy_available()
        and is_torch_available()
        and is_pretty_midi_available()
    ):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import (
        dummy_essentia_and_librosa_and_pretty_midi_and_scipy_and_torch_objects,
    )

    _import_structure["utils.dummy_essentia_and_librosa_and_pretty_midi_and_scipy_and_torch_objects"] = [
        name
        for name in dir(dummy_essentia_and_librosa_and_pretty_midi_and_scipy_and_torch_objects)
        if not name.startswith("_")
    ]
else:
    _import_structure["models.pop2piano"].append("Pop2PianoFeatureExtractor")
    _import_structure["models.pop2piano"].append("Pop2PianoTokenizer")
    _import_structure["models.pop2piano"].append("Pop2PianoProcessor")

try:
    if not is_torchaudio_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import (
        dummy_torchaudio_objects,
    )

    _import_structure["utils.dummy_torchaudio_objects"] = [
        name for name in dir(dummy_torchaudio_objects) if not name.startswith("_")
    ]
else:
    _import_structure["models.musicgen_melody"].append("MusicgenMelodyFeatureExtractor")
    _import_structure["models.musicgen_melody"].append("MusicgenMelodyProcessor")


# FLAX-backed objects
try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_flax_objects

    _import_structure["utils.dummy_flax_objects"] = [
        name for name in dir(dummy_flax_objects) if not name.startswith("_")
    ]
else:
    _import_structure["generation"].extend(
        [
            "FlaxForcedBOSTokenLogitsProcessor",
            "FlaxForcedEOSTokenLogitsProcessor",
            "FlaxForceTokensLogitsProcessor",
            "FlaxGenerationMixin",
            "FlaxLogitsProcessor",
            "FlaxLogitsProcessorList",
            "FlaxLogitsWarper",
            "FlaxMinLengthLogitsProcessor",
            "FlaxTemperatureLogitsWarper",
            "FlaxSuppressTokensAtBeginLogitsProcessor",
            "FlaxSuppressTokensLogitsProcessor",
            "FlaxTopKLogitsWarper",
            "FlaxTopPLogitsWarper",
            "FlaxWhisperTimeStampLogitsProcessor",
        ]
    )
    _import_structure["modeling_flax_outputs"] = []
    _import_structure["modeling_flax_utils"] = ["FlaxPreTrainedModel"]
    _import_structure["models.albert"].extend(
        [
            "FlaxAlbertForMaskedLM",
            "FlaxAlbertForMultipleChoice",
            "FlaxAlbertForPreTraining",
            "FlaxAlbertForQuestionAnswering",
            "FlaxAlbertForSequenceClassification",
            "FlaxAlbertForTokenClassification",
            "FlaxAlbertModel",
            "FlaxAlbertPreTrainedModel",
        ]
    )
    _import_structure["models.auto"].extend(
        [
            "FLAX_MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING",
            "FLAX_MODEL_FOR_CAUSAL_LM_MAPPING",
            "FLAX_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING",
            "FLAX_MODEL_FOR_MASKED_LM_MAPPING",
            "FLAX_MODEL_FOR_MULTIPLE_CHOICE_MAPPING",
            "FLAX_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING",
            "FLAX_MODEL_FOR_PRETRAINING_MAPPING",
            "FLAX_MODEL_FOR_QUESTION_ANSWERING_MAPPING",
            "FLAX_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING",
            "FLAX_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING",
            "FLAX_MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING",
            "FLAX_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING",
            "FLAX_MODEL_FOR_VISION_2_SEQ_MAPPING",
            "FLAX_MODEL_MAPPING",
            "FlaxAutoModel",
            "FlaxAutoModelForCausalLM",
            "FlaxAutoModelForImageClassification",
            "FlaxAutoModelForMaskedLM",
            "FlaxAutoModelForMultipleChoice",
            "FlaxAutoModelForNextSentencePrediction",
            "FlaxAutoModelForPreTraining",
            "FlaxAutoModelForQuestionAnswering",
            "FlaxAutoModelForSeq2SeqLM",
            "FlaxAutoModelForSequenceClassification",
            "FlaxAutoModelForSpeechSeq2Seq",
            "FlaxAutoModelForTokenClassification",
            "FlaxAutoModelForVision2Seq",
        ]
    )

    # Flax models structure

    _import_structure["models.bart"].extend(
        [
            "FlaxBartDecoderPreTrainedModel",
            "FlaxBartForCausalLM",
            "FlaxBartForConditionalGeneration",
            "FlaxBartForQuestionAnswering",
            "FlaxBartForSequenceClassification",
            "FlaxBartModel",
            "FlaxBartPreTrainedModel",
        ]
    )
    _import_structure["models.beit"].extend(
        [
            "FlaxBeitForImageClassification",
            "FlaxBeitForMaskedImageModeling",
            "FlaxBeitModel",
            "FlaxBeitPreTrainedModel",
        ]
    )

    _import_structure["models.bert"].extend(
        [
            "FlaxBertForCausalLM",
            "FlaxBertForMaskedLM",
            "FlaxBertForMultipleChoice",
            "FlaxBertForNextSentencePrediction",
            "FlaxBertForPreTraining",
            "FlaxBertForQuestionAnswering",
            "FlaxBertForSequenceClassification",
            "FlaxBertForTokenClassification",
            "FlaxBertModel",
            "FlaxBertPreTrainedModel",
        ]
    )
    _import_structure["models.big_bird"].extend(
        [
            "FlaxBigBirdForCausalLM",
            "FlaxBigBirdForMaskedLM",
            "FlaxBigBirdForMultipleChoice",
            "FlaxBigBirdForPreTraining",
            "FlaxBigBirdForQuestionAnswering",
            "FlaxBigBirdForSequenceClassification",
            "FlaxBigBirdForTokenClassification",
            "FlaxBigBirdModel",
            "FlaxBigBirdPreTrainedModel",
        ]
    )
    _import_structure["models.blenderbot"].extend(
        [
            "FlaxBlenderbotForConditionalGeneration",
            "FlaxBlenderbotModel",
            "FlaxBlenderbotPreTrainedModel",
        ]
    )
    _import_structure["models.blenderbot_small"].extend(
        [
            "FlaxBlenderbotSmallForConditionalGeneration",
            "FlaxBlenderbotSmallModel",
            "FlaxBlenderbotSmallPreTrainedModel",
        ]
    )
    _import_structure["models.bloom"].extend(
        [
            "FlaxBloomForCausalLM",
            "FlaxBloomModel",
            "FlaxBloomPreTrainedModel",
        ]
    )
    _import_structure["models.clip"].extend(
        [
            "FlaxCLIPModel",
            "FlaxCLIPPreTrainedModel",
            "FlaxCLIPTextModel",
            "FlaxCLIPTextPreTrainedModel",
            "FlaxCLIPTextModelWithProjection",
            "FlaxCLIPVisionModel",
            "FlaxCLIPVisionPreTrainedModel",
        ]
    )
    _import_structure["models.dinov2"].extend(
        [
            "FlaxDinov2Model",
            "FlaxDinov2ForImageClassification",
            "FlaxDinov2PreTrainedModel",
        ]
    )
    _import_structure["models.distilbert"].extend(
        [
            "FlaxDistilBertForMaskedLM",
            "FlaxDistilBertForMultipleChoice",
            "FlaxDistilBertForQuestionAnswering",
            "FlaxDistilBertForSequenceClassification",
            "FlaxDistilBertForTokenClassification",
            "FlaxDistilBertModel",
            "FlaxDistilBertPreTrainedModel",
        ]
    )
    _import_structure["models.electra"].extend(
        [
            "FlaxElectraForCausalLM",
            "FlaxElectraForMaskedLM",
            "FlaxElectraForMultipleChoice",
            "FlaxElectraForPreTraining",
            "FlaxElectraForQuestionAnswering",
            "FlaxElectraForSequenceClassification",
            "FlaxElectraForTokenClassification",
            "FlaxElectraModel",
            "FlaxElectraPreTrainedModel",
        ]
    )
    _import_structure["models.encoder_decoder"].append("FlaxEncoderDecoderModel")
    _import_structure["models.gpt2"].extend(["FlaxGPT2LMHeadModel", "FlaxGPT2Model", "FlaxGPT2PreTrainedModel"])
    _import_structure["models.gpt_neo"].extend(
        ["FlaxGPTNeoForCausalLM", "FlaxGPTNeoModel", "FlaxGPTNeoPreTrainedModel"]
    )
    _import_structure["models.gptj"].extend(["FlaxGPTJForCausalLM", "FlaxGPTJModel", "FlaxGPTJPreTrainedModel"])
    _import_structure["models.llama"].extend(["FlaxLlamaForCausalLM", "FlaxLlamaModel", "FlaxLlamaPreTrainedModel"])
    _import_structure["models.gemma"].extend(["FlaxGemmaForCausalLM", "FlaxGemmaModel", "FlaxGemmaPreTrainedModel"])
    _import_structure["models.longt5"].extend(
        [
            "FlaxLongT5ForConditionalGeneration",
            "FlaxLongT5Model",
            "FlaxLongT5PreTrainedModel",
        ]
    )
    _import_structure["models.marian"].extend(
        [
            "FlaxMarianModel",
            "FlaxMarianMTModel",
            "FlaxMarianPreTrainedModel",
        ]
    )
    _import_structure["models.mbart"].extend(
        [
            "FlaxMBartForConditionalGeneration",
            "FlaxMBartForQuestionAnswering",
            "FlaxMBartForSequenceClassification",
            "FlaxMBartModel",
            "FlaxMBartPreTrainedModel",
        ]
    )
    _import_structure["models.mistral"].extend(
        [
            "FlaxMistralForCausalLM",
            "FlaxMistralModel",
            "FlaxMistralPreTrainedModel",
        ]
    )
    _import_structure["models.mt5"].extend(["FlaxMT5EncoderModel", "FlaxMT5ForConditionalGeneration", "FlaxMT5Model"])
    _import_structure["models.opt"].extend(
        [
            "FlaxOPTForCausalLM",
            "FlaxOPTModel",
            "FlaxOPTPreTrainedModel",
        ]
    )
    _import_structure["models.pegasus"].extend(
        [
            "FlaxPegasusForConditionalGeneration",
            "FlaxPegasusModel",
            "FlaxPegasusPreTrainedModel",
        ]
    )
    _import_structure["models.regnet"].extend(
        [
            "FlaxRegNetForImageClassification",
            "FlaxRegNetModel",
            "FlaxRegNetPreTrainedModel",
        ]
    )
    _import_structure["models.resnet"].extend(
        [
            "FlaxResNetForImageClassification",
            "FlaxResNetModel",
            "FlaxResNetPreTrainedModel",
        ]
    )
    _import_structure["models.roberta"].extend(
        [
            "FlaxRobertaForCausalLM",
            "FlaxRobertaForMaskedLM",
            "FlaxRobertaForMultipleChoice",
            "FlaxRobertaForQuestionAnswering",
            "FlaxRobertaForSequenceClassification",
            "FlaxRobertaForTokenClassification",
            "FlaxRobertaModel",
            "FlaxRobertaPreTrainedModel",
        ]
    )
    _import_structure["models.roberta_prelayernorm"].extend(
        [
            "FlaxRobertaPreLayerNormForCausalLM",
            "FlaxRobertaPreLayerNormForMaskedLM",
            "FlaxRobertaPreLayerNormForMultipleChoice",
            "FlaxRobertaPreLayerNormForQuestionAnswering",
            "FlaxRobertaPreLayerNormForSequenceClassification",
            "FlaxRobertaPreLayerNormForTokenClassification",
            "FlaxRobertaPreLayerNormModel",
            "FlaxRobertaPreLayerNormPreTrainedModel",
        ]
    )
    _import_structure["models.roformer"].extend(
        [
            "FlaxRoFormerForMaskedLM",
            "FlaxRoFormerForMultipleChoice",
            "FlaxRoFormerForQuestionAnswering",
            "FlaxRoFormerForSequenceClassification",
            "FlaxRoFormerForTokenClassification",
            "FlaxRoFormerModel",
            "FlaxRoFormerPreTrainedModel",
        ]
    )
    _import_structure["models.speech_encoder_decoder"].append("FlaxSpeechEncoderDecoderModel")
    _import_structure["models.t5"].extend(
        [
            "FlaxT5EncoderModel",
            "FlaxT5ForConditionalGeneration",
            "FlaxT5Model",
            "FlaxT5PreTrainedModel",
        ]
    )
    _import_structure["models.vision_encoder_decoder"].append("FlaxVisionEncoderDecoderModel")
    _import_structure["models.vision_text_dual_encoder"].extend(["FlaxVisionTextDualEncoderModel"])
    _import_structure["models.vit"].extend(["FlaxViTForImageClassification", "FlaxViTModel", "FlaxViTPreTrainedModel"])
    _import_structure["models.wav2vec2"].extend(
        [
            "FlaxWav2Vec2ForCTC",
            "FlaxWav2Vec2ForPreTraining",
            "FlaxWav2Vec2Model",
            "FlaxWav2Vec2PreTrainedModel",
        ]
    )
    _import_structure["models.whisper"].extend(
        [
            "FlaxWhisperForConditionalGeneration",
            "FlaxWhisperModel",
            "FlaxWhisperPreTrainedModel",
            "FlaxWhisperForAudioClassification",
        ]
    )
    _import_structure["models.xglm"].extend(
        [
            "FlaxXGLMForCausalLM",
            "FlaxXGLMModel",
            "FlaxXGLMPreTrainedModel",
        ]
    )
    _import_structure["models.xlm_roberta"].extend(
        [
            "FlaxXLMRobertaForMaskedLM",
            "FlaxXLMRobertaForMultipleChoice",
            "FlaxXLMRobertaForQuestionAnswering",
            "FlaxXLMRobertaForSequenceClassification",
            "FlaxXLMRobertaForTokenClassification",
            "FlaxXLMRobertaModel",
            "FlaxXLMRobertaForCausalLM",
            "FlaxXLMRobertaPreTrainedModel",
        ]
    )


# Direct imports for type-checking
if TYPE_CHECKING:
    # Configuration
    # Agents
    from .agents import (
        Agent,
        CodeAgent,
        HfApiEngine,
        ManagedAgent,
        PipelineTool,
        ReactAgent,
        ReactCodeAgent,
        ReactJsonAgent,
        Tool,
        Toolbox,
        ToolCollection,
        TransformersEngine,
        launch_gradio_demo,
        load_tool,
        stream_to_gradio,
        tool,
    )
    from .configuration_utils import PretrainedConfig

    # Data
    from .data import (
        DataProcessor,
        InputExample,
        InputFeatures,
        SingleSentenceClassificationProcessor,
        SquadExample,
        SquadFeatures,
        SquadV1Processor,
        SquadV2Processor,
        glue_compute_metrics,
        glue_convert_examples_to_features,
        glue_output_modes,
        glue_processors,
        glue_tasks_num_labels,
        squad_convert_examples_to_features,
        xnli_compute_metrics,
        xnli_output_modes,
        xnli_processors,
        xnli_tasks_num_labels,
    )
    from .data.data_collator import (
        DataCollator,
        DataCollatorForLanguageModeling,
        DataCollatorForPermutationLanguageModeling,
        DataCollatorForSeq2Seq,
        DataCollatorForSOP,
        DataCollatorForTokenClassification,
        DataCollatorForWholeWordMask,
        DataCollatorWithFlattening,
        DataCollatorWithPadding,
        DefaultDataCollator,
        default_data_collator,
    )
    from .feature_extraction_sequence_utils import SequenceFeatureExtractor

    # Feature Extractor
    from .feature_extraction_utils import BatchFeature, FeatureExtractionMixin

    # Generation
    from .generation import (
        AsyncTextIteratorStreamer,
        CompileConfig,
        GenerationConfig,
        TextIteratorStreamer,
        TextStreamer,
        WatermarkingConfig,
    )
    from .hf_argparser import HfArgumentParser

    # Integrations
    from .integrations import (
        is_clearml_available,
        is_comet_available,
        is_dvclive_available,
        is_neptune_available,
        is_optuna_available,
        is_ray_available,
        is_ray_tune_available,
        is_sigopt_available,
        is_tensorboard_available,
        is_wandb_available,
    )

    # Model Cards
    from .modelcard import ModelCard

    # TF 2.0 <=> PyTorch conversion utilities
    from .modeling_tf_pytorch_utils import (
        convert_tf_weight_name_to_pt_weight_name,
        load_pytorch_checkpoint_in_tf2_model,
        load_pytorch_model_in_tf2_model,
        load_pytorch_weights_in_tf2_model,
        load_tf2_checkpoint_in_pytorch_model,
        load_tf2_model_in_pytorch_model,
        load_tf2_weights_in_pytorch_model,
    )
    from .models.albert import AlbertConfig
    from .models.align import (
        AlignConfig,
        AlignProcessor,
        AlignTextConfig,
        AlignVisionConfig,
    )
    from .models.altclip import (
        AltCLIPConfig,
        AltCLIPProcessor,
        AltCLIPTextConfig,
        AltCLIPVisionConfig,
    )
    from .models.aria import (
        AriaConfig,
        AriaProcessor,
        AriaTextConfig,
    )
    from .models.audio_spectrogram_transformer import (
        ASTConfig,
        ASTFeatureExtractor,
    )
    from .models.auto import (
        CONFIG_MAPPING,
        FEATURE_EXTRACTOR_MAPPING,
        IMAGE_PROCESSOR_MAPPING,
        MODEL_NAMES_MAPPING,
        PROCESSOR_MAPPING,
        TOKENIZER_MAPPING,
        AutoConfig,
        AutoFeatureExtractor,
        AutoImageProcessor,
        AutoProcessor,
        AutoTokenizer,
    )
    from .models.autoformer import (
        AutoformerConfig,
    )
    from .models.bamba import BambaConfig
    from .models.bark import (
        BarkCoarseConfig,
        BarkConfig,
        BarkFineConfig,
        BarkProcessor,
        BarkSemanticConfig,
    )
    from .models.bart import BartConfig, BartTokenizer
    from .models.beit import BeitConfig
    from .models.bert import (
        BasicTokenizer,
        BertConfig,
        BertTokenizer,
        WordpieceTokenizer,
    )
    from .models.bert_generation import BertGenerationConfig
    from .models.bert_japanese import (
        BertJapaneseTokenizer,
        CharacterTokenizer,
        MecabTokenizer,
    )
    from .models.bertweet import BertweetTokenizer
    from .models.big_bird import BigBirdConfig
    from .models.bigbird_pegasus import (
        BigBirdPegasusConfig,
    )
    from .models.biogpt import (
        BioGptConfig,
        BioGptTokenizer,
    )
    from .models.bit import BitConfig
    from .models.blenderbot import (
        BlenderbotConfig,
        BlenderbotTokenizer,
    )
    from .models.blenderbot_small import (
        BlenderbotSmallConfig,
        BlenderbotSmallTokenizer,
    )
    from .models.blip import (
        BlipConfig,
        BlipProcessor,
        BlipTextConfig,
        BlipVisionConfig,
    )
    from .models.blip_2 import (
        Blip2Config,
        Blip2Processor,
        Blip2QFormerConfig,
        Blip2VisionConfig,
    )
    from .models.bloom import BloomConfig
    from .models.bridgetower import (
        BridgeTowerConfig,
        BridgeTowerProcessor,
        BridgeTowerTextConfig,
        BridgeTowerVisionConfig,
    )
    from .models.bros import (
        BrosConfig,
        BrosProcessor,
    )
    from .models.byt5 import ByT5Tokenizer
    from .models.camembert import (
        CamembertConfig,
    )
    from .models.canine import (
        CanineConfig,
        CanineTokenizer,
    )
    from .models.chameleon import (
        ChameleonConfig,
        ChameleonProcessor,
        ChameleonVQVAEConfig,
    )
    from .models.chinese_clip import (
        ChineseCLIPConfig,
        ChineseCLIPProcessor,
        ChineseCLIPTextConfig,
        ChineseCLIPVisionConfig,
    )
    from .models.clap import (
        ClapAudioConfig,
        ClapConfig,
        ClapProcessor,
        ClapTextConfig,
    )
    from .models.clip import (
        CLIPConfig,
        CLIPProcessor,
        CLIPTextConfig,
        CLIPTokenizer,
        CLIPVisionConfig,
    )
    from .models.clipseg import (
        CLIPSegConfig,
        CLIPSegProcessor,
        CLIPSegTextConfig,
        CLIPSegVisionConfig,
    )
    from .models.clvp import (
        ClvpConfig,
        ClvpDecoderConfig,
        ClvpEncoderConfig,
        ClvpFeatureExtractor,
        ClvpProcessor,
        ClvpTokenizer,
    )
    from .models.codegen import (
        CodeGenConfig,
        CodeGenTokenizer,
    )
    from .models.cohere import CohereConfig
    from .models.cohere2 import Cohere2Config
    from .models.colpali import (
        ColPaliConfig,
        ColPaliProcessor,
    )
    from .models.conditional_detr import (
        ConditionalDetrConfig,
    )
    from .models.convbert import (
        ConvBertConfig,
        ConvBertTokenizer,
    )
    from .models.convnext import ConvNextConfig
    from .models.convnextv2 import (
        ConvNextV2Config,
    )
    from .models.cpmant import (
        CpmAntConfig,
        CpmAntTokenizer,
    )
    from .models.ctrl import (
        CTRLConfig,
        CTRLTokenizer,
    )
    from .models.cvt import CvtConfig
    from .models.dac import (
        DacConfig,
        DacFeatureExtractor,
    )
    from .models.data2vec import (
        Data2VecAudioConfig,
        Data2VecTextConfig,
        Data2VecVisionConfig,
    )
    from .models.dbrx import DbrxConfig
    from .models.deberta import (
        DebertaConfig,
        DebertaTokenizer,
    )
    from .models.deberta_v2 import (
        DebertaV2Config,
    )
    from .models.decision_transformer import (
        DecisionTransformerConfig,
    )
    from .models.deformable_detr import (
        DeformableDetrConfig,
    )
    from .models.deit import DeiTConfig
    from .models.deprecated.deta import DetaConfig
    from .models.deprecated.efficientformer import (
        EfficientFormerConfig,
    )
    from .models.deprecated.ernie_m import ErnieMConfig
    from .models.deprecated.gptsan_japanese import (
        GPTSanJapaneseConfig,
        GPTSanJapaneseTokenizer,
    )
    from .models.deprecated.graphormer import GraphormerConfig
    from .models.deprecated.jukebox import (
        JukeboxConfig,
        JukeboxPriorConfig,
        JukeboxTokenizer,
        JukeboxVQVAEConfig,
    )
    from .models.deprecated.mctct import (
        MCTCTConfig,
        MCTCTFeatureExtractor,
        MCTCTProcessor,
    )
    from .models.deprecated.mega import MegaConfig
    from .models.deprecated.mmbt import MMBTConfig
    from .models.deprecated.nat import NatConfig
    from .models.deprecated.nezha import NezhaConfig
    from .models.deprecated.open_llama import (
        OpenLlamaConfig,
    )
    from .models.deprecated.qdqbert import QDQBertConfig
    from .models.deprecated.realm import (
        RealmConfig,
        RealmTokenizer,
    )
    from .models.deprecated.retribert import (
        RetriBertConfig,
        RetriBertTokenizer,
    )
    from .models.deprecated.speech_to_text_2 import (
        Speech2Text2Config,
        Speech2Text2Processor,
        Speech2Text2Tokenizer,
    )
    from .models.deprecated.tapex import TapexTokenizer
    from .models.deprecated.trajectory_transformer import (
        TrajectoryTransformerConfig,
    )
    from .models.deprecated.transfo_xl import (
        TransfoXLConfig,
        TransfoXLCorpus,
        TransfoXLTokenizer,
    )
    from .models.deprecated.tvlt import (
        TvltConfig,
        TvltFeatureExtractor,
        TvltProcessor,
    )
    from .models.deprecated.van import VanConfig
    from .models.deprecated.vit_hybrid import (
        ViTHybridConfig,
    )
    from .models.deprecated.xlm_prophetnet import (
        XLMProphetNetConfig,
    )
    from .models.depth_anything import DepthAnythingConfig
    from .models.detr import DetrConfig
    from .models.diffllama import DiffLlamaConfig
    from .models.dinat import DinatConfig
    from .models.dinov2 import Dinov2Config
    from .models.dinov2_with_registers import Dinov2WithRegistersConfig
    from .models.distilbert import (
        DistilBertConfig,
        DistilBertTokenizer,
    )
    from .models.donut import (
        DonutProcessor,
        DonutSwinConfig,
    )
    from .models.dpr import (
        DPRConfig,
        DPRContextEncoderTokenizer,
        DPRQuestionEncoderTokenizer,
        DPRReaderOutput,
        DPRReaderTokenizer,
    )
    from .models.dpt import DPTConfig
    from .models.efficientnet import (
        EfficientNetConfig,
    )
    from .models.electra import (
        ElectraConfig,
        ElectraTokenizer,
    )
    from .models.encodec import (
        EncodecConfig,
        EncodecFeatureExtractor,
    )
    from .models.encoder_decoder import EncoderDecoderConfig
    from .models.ernie import ErnieConfig
    from .models.esm import EsmConfig, EsmTokenizer
    from .models.falcon import FalconConfig
    from .models.falcon_mamba import FalconMambaConfig
    from .models.fastspeech2_conformer import (
        FastSpeech2ConformerConfig,
        FastSpeech2ConformerHifiGanConfig,
        FastSpeech2ConformerTokenizer,
        FastSpeech2ConformerWithHifiGanConfig,
    )
    from .models.flaubert import FlaubertConfig, FlaubertTokenizer
    from .models.flava import (
        FlavaConfig,
        FlavaImageCodebookConfig,
        FlavaImageConfig,
        FlavaMultimodalConfig,
        FlavaTextConfig,
    )
    from .models.fnet import FNetConfig
    from .models.focalnet import FocalNetConfig
    from .models.fsmt import (
        FSMTConfig,
        FSMTTokenizer,
    )
    from .models.funnel import (
        FunnelConfig,
        FunnelTokenizer,
    )
    from .models.fuyu import FuyuConfig
    from .models.gemma import GemmaConfig
    from .models.gemma2 import Gemma2Config
    from .models.git import (
        GitConfig,
        GitProcessor,
        GitVisionConfig,
    )
    from .models.glm import GlmConfig
    from .models.glpn import GLPNConfig
    from .models.gpt2 import (
        GPT2Config,
        GPT2Tokenizer,
    )
    from .models.gpt_bigcode import (
        GPTBigCodeConfig,
    )
    from .models.gpt_neo import GPTNeoConfig
    from .models.gpt_neox import GPTNeoXConfig
    from .models.gpt_neox_japanese import (
        GPTNeoXJapaneseConfig,
    )
    from .models.gptj import GPTJConfig
    from .models.granite import GraniteConfig
    from .models.granitemoe import GraniteMoeConfig
    from .models.grounding_dino import (
        GroundingDinoConfig,
        GroundingDinoProcessor,
    )
    from .models.groupvit import (
        GroupViTConfig,
        GroupViTTextConfig,
        GroupViTVisionConfig,
    )
    from .models.herbert import HerbertTokenizer
    from .models.hiera import HieraConfig
    from .models.hubert import HubertConfig
    from .models.ibert import IBertConfig
    from .models.idefics import (
        IdeficsConfig,
    )
    from .models.idefics2 import Idefics2Config
    from .models.idefics3 import Idefics3Config
    from .models.ijepa import IJepaConfig
    from .models.imagegpt import ImageGPTConfig
    from .models.informer import InformerConfig
    from .models.instructblip import (
        InstructBlipConfig,
        InstructBlipProcessor,
        InstructBlipQFormerConfig,
        InstructBlipVisionConfig,
    )
    from .models.instructblipvideo import (
        InstructBlipVideoConfig,
        InstructBlipVideoProcessor,
        InstructBlipVideoQFormerConfig,
        InstructBlipVideoVisionConfig,
    )
    from .models.jamba import JambaConfig
    from .models.jetmoe import JetMoeConfig
    from .models.kosmos2 import (
        Kosmos2Config,
        Kosmos2Processor,
    )
    from .models.layoutlm import (
        LayoutLMConfig,
        LayoutLMTokenizer,
    )
    from .models.layoutlmv2 import (
        LayoutLMv2Config,
        LayoutLMv2FeatureExtractor,
        LayoutLMv2ImageProcessor,
        LayoutLMv2Processor,
        LayoutLMv2Tokenizer,
    )
    from .models.layoutlmv3 import (
        LayoutLMv3Config,
        LayoutLMv3FeatureExtractor,
        LayoutLMv3ImageProcessor,
        LayoutLMv3Processor,
        LayoutLMv3Tokenizer,
    )
    from .models.layoutxlm import LayoutXLMProcessor
    from .models.led import LEDConfig, LEDTokenizer
    from .models.levit import LevitConfig
    from .models.lilt import LiltConfig
    from .models.llama import LlamaConfig
    from .models.llava import (
        LlavaConfig,
        LlavaProcessor,
    )
    from .models.llava_next import (
        LlavaNextConfig,
        LlavaNextProcessor,
    )
    from .models.llava_next_video import (
        LlavaNextVideoConfig,
        LlavaNextVideoProcessor,
    )
    from .models.llava_onevision import (
        LlavaOnevisionConfig,
        LlavaOnevisionProcessor,
    )
    from .models.longformer import (
        LongformerConfig,
        LongformerTokenizer,
    )
    from .models.longt5 import LongT5Config
    from .models.luke import (
        LukeConfig,
        LukeTokenizer,
    )
    from .models.lxmert import (
        LxmertConfig,
        LxmertTokenizer,
    )
    from .models.m2m_100 import M2M100Config
    from .models.mamba import MambaConfig
    from .models.mamba2 import Mamba2Config
    from .models.marian import MarianConfig
    from .models.markuplm import (
        MarkupLMConfig,
        MarkupLMFeatureExtractor,
        MarkupLMProcessor,
        MarkupLMTokenizer,
    )
    from .models.mask2former import (
        Mask2FormerConfig,
    )
    from .models.maskformer import (
        MaskFormerConfig,
        MaskFormerSwinConfig,
    )
    from .models.mbart import MBartConfig
    from .models.megatron_bert import (
        MegatronBertConfig,
    )
    from .models.mgp_str import (
        MgpstrConfig,
        MgpstrProcessor,
        MgpstrTokenizer,
    )
    from .models.mimi import (
        MimiConfig,
    )
    from .models.mistral import MistralConfig
    from .models.mixtral import MixtralConfig
    from .models.mllama import (
        MllamaConfig,
        MllamaProcessor,
    )
    from .models.mobilebert import (
        MobileBertConfig,
        MobileBertTokenizer,
    )
    from .models.mobilenet_v1 import (
        MobileNetV1Config,
    )
    from .models.mobilenet_v2 import (
        MobileNetV2Config,
    )
    from .models.mobilevit import (
        MobileViTConfig,
    )
    from .models.mobilevitv2 import (
        MobileViTV2Config,
    )
    from .models.modernbert import ModernBertConfig
    from .models.moshi import (
        MoshiConfig,
        MoshiDepthConfig,
    )
    from .models.mpnet import (
        MPNetConfig,
        MPNetTokenizer,
    )
    from .models.mpt import MptConfig
    from .models.mra import MraConfig
    from .models.mt5 import MT5Config
    from .models.musicgen import (
        MusicgenConfig,
        MusicgenDecoderConfig,
    )
    from .models.musicgen_melody import (
        MusicgenMelodyConfig,
        MusicgenMelodyDecoderConfig,
    )
    from .models.mvp import MvpConfig, MvpTokenizer
    from .models.myt5 import MyT5Tokenizer
    from .models.nemotron import NemotronConfig
    from .models.nllb_moe import NllbMoeConfig
    from .models.nougat import NougatProcessor
    from .models.nystromformer import (
        NystromformerConfig,
    )
    from .models.olmo import OlmoConfig
    from .models.olmo2 import Olmo2Config
    from .models.olmoe import OlmoeConfig
    from .models.omdet_turbo import (
        OmDetTurboConfig,
        OmDetTurboProcessor,
    )
    from .models.oneformer import (
        OneFormerConfig,
        OneFormerProcessor,
    )
    from .models.openai import (
        OpenAIGPTConfig,
        OpenAIGPTTokenizer,
    )
    from .models.opt import OPTConfig
    from .models.owlv2 import (
        Owlv2Config,
        Owlv2Processor,
        Owlv2TextConfig,
        Owlv2VisionConfig,
    )
    from .models.owlvit import (
        OwlViTConfig,
        OwlViTProcessor,
        OwlViTTextConfig,
        OwlViTVisionConfig,
    )
    from .models.paligemma import (
        PaliGemmaConfig,
    )
    from .models.patchtsmixer import (
        PatchTSMixerConfig,
    )
    from .models.patchtst import PatchTSTConfig
    from .models.pegasus import (
        PegasusConfig,
        PegasusTokenizer,
    )
    from .models.pegasus_x import (
        PegasusXConfig,
    )
    from .models.perceiver import (
        PerceiverConfig,
        PerceiverTokenizer,
    )
    from .models.persimmon import (
        PersimmonConfig,
    )
    from .models.phi import PhiConfig
    from .models.phi3 import Phi3Config
    from .models.phimoe import PhimoeConfig
    from .models.phobert import PhobertTokenizer
    from .models.pix2struct import (
        Pix2StructConfig,
        Pix2StructProcessor,
        Pix2StructTextConfig,
        Pix2StructVisionConfig,
    )
    from .models.pixtral import (
        PixtralProcessor,
        PixtralVisionConfig,
    )
    from .models.plbart import PLBartConfig
    from .models.poolformer import (
        PoolFormerConfig,
    )
    from .models.pop2piano import (
        Pop2PianoConfig,
    )
    from .models.prophetnet import (
        ProphetNetConfig,
        ProphetNetTokenizer,
    )
    from .models.pvt import PvtConfig
    from .models.pvt_v2 import PvtV2Config
    from .models.qwen2 import Qwen2Config, Qwen2Tokenizer
    from .models.qwen2_audio import (
        Qwen2AudioConfig,
        Qwen2AudioEncoderConfig,
        Qwen2AudioProcessor,
    )
    from .models.qwen2_moe import Qwen2MoeConfig
    from .models.qwen2_vl import (
        Qwen2VLConfig,
        Qwen2VLProcessor,
    )
    from .models.rag import RagConfig, RagRetriever, RagTokenizer
    from .models.recurrent_gemma import RecurrentGemmaConfig
    from .models.reformer import ReformerConfig
    from .models.regnet import RegNetConfig
    from .models.rembert import RemBertConfig
    from .models.resnet import ResNetConfig
    from .models.roberta import (
        RobertaConfig,
        RobertaTokenizer,
    )
    from .models.roberta_prelayernorm import (
        RobertaPreLayerNormConfig,
    )
    from .models.roc_bert import (
        RoCBertConfig,
        RoCBertTokenizer,
    )
    from .models.roformer import (
        RoFormerConfig,
        RoFormerTokenizer,
    )
    from .models.rt_detr import (
        RTDetrConfig,
        RTDetrResNetConfig,
    )
    from .models.rwkv import RwkvConfig
    from .models.sam import (
        SamConfig,
        SamMaskDecoderConfig,
        SamProcessor,
        SamPromptEncoderConfig,
        SamVisionConfig,
    )
    from .models.seamless_m4t import (
        SeamlessM4TConfig,
        SeamlessM4TFeatureExtractor,
        SeamlessM4TProcessor,
    )
    from .models.seamless_m4t_v2 import (
        SeamlessM4Tv2Config,
    )
    from .models.segformer import SegformerConfig
    from .models.seggpt import SegGptConfig
    from .models.sew import SEWConfig
    from .models.sew_d import SEWDConfig
    from .models.siglip import (
        SiglipConfig,
        SiglipProcessor,
        SiglipTextConfig,
        SiglipVisionConfig,
    )
    from .models.speech_encoder_decoder import SpeechEncoderDecoderConfig
    from .models.speech_to_text import (
        Speech2TextConfig,
        Speech2TextFeatureExtractor,
        Speech2TextProcessor,
    )
    from .models.speecht5 import (
        SpeechT5Config,
        SpeechT5FeatureExtractor,
        SpeechT5HifiGanConfig,
        SpeechT5Processor,
    )
    from .models.splinter import (
        SplinterConfig,
        SplinterTokenizer,
    )
    from .models.squeezebert import (
        SqueezeBertConfig,
        SqueezeBertTokenizer,
    )
    from .models.stablelm import StableLmConfig
    from .models.starcoder2 import Starcoder2Config
    from .models.superpoint import SuperPointConfig
    from .models.swiftformer import (
        SwiftFormerConfig,
    )
    from .models.swin import SwinConfig
    from .models.swin2sr import Swin2SRConfig
    from .models.swinv2 import Swinv2Config
    from .models.switch_transformers import (
        SwitchTransformersConfig,
    )
    from .models.t5 import T5Config
    from .models.table_transformer import (
        TableTransformerConfig,
    )
    from .models.tapas import (
        TapasConfig,
        TapasTokenizer,
    )
    from .models.time_series_transformer import (
        TimeSeriesTransformerConfig,
    )
    from .models.timesformer import (
        TimesformerConfig,
    )
    from .models.timm_backbone import TimmBackboneConfig
    from .models.timm_wrapper import TimmWrapperConfig
    from .models.trocr import (
        TrOCRConfig,
        TrOCRProcessor,
    )
    from .models.tvp import (
        TvpConfig,
        TvpProcessor,
    )
    from .models.udop import UdopConfig, UdopProcessor
    from .models.umt5 import UMT5Config
    from .models.unispeech import (
        UniSpeechConfig,
    )
    from .models.unispeech_sat import (
        UniSpeechSatConfig,
    )
    from .models.univnet import (
        UnivNetConfig,
        UnivNetFeatureExtractor,
    )
    from .models.upernet import UperNetConfig
    from .models.video_llava import VideoLlavaConfig
    from .models.videomae import VideoMAEConfig
    from .models.vilt import (
        ViltConfig,
        ViltFeatureExtractor,
        ViltImageProcessor,
        ViltProcessor,
    )
    from .models.vipllava import (
        VipLlavaConfig,
    )
    from .models.vision_encoder_decoder import VisionEncoderDecoderConfig
    from .models.vision_text_dual_encoder import (
        VisionTextDualEncoderConfig,
        VisionTextDualEncoderProcessor,
    )
    from .models.visual_bert import (
        VisualBertConfig,
    )
    from .models.vit import ViTConfig
    from .models.vit_mae import ViTMAEConfig
    from .models.vit_msn import ViTMSNConfig
    from .models.vitdet import VitDetConfig
    from .models.vitmatte import VitMatteConfig
    from .models.vits import (
        VitsConfig,
        VitsTokenizer,
    )
    from .models.vivit import VivitConfig
    from .models.wav2vec2 import (
        Wav2Vec2Config,
        Wav2Vec2CTCTokenizer,
        Wav2Vec2FeatureExtractor,
        Wav2Vec2Processor,
        Wav2Vec2Tokenizer,
    )
    from .models.wav2vec2_bert import (
        Wav2Vec2BertConfig,
        Wav2Vec2BertProcessor,
    )
    from .models.wav2vec2_conformer import (
        Wav2Vec2ConformerConfig,
    )
    from .models.wav2vec2_phoneme import Wav2Vec2PhonemeCTCTokenizer
    from .models.wav2vec2_with_lm import Wav2Vec2ProcessorWithLM
    from .models.wavlm import WavLMConfig
    from .models.whisper import (
        WhisperConfig,
        WhisperFeatureExtractor,
        WhisperProcessor,
        WhisperTokenizer,
    )
    from .models.x_clip import (
        XCLIPConfig,
        XCLIPProcessor,
        XCLIPTextConfig,
        XCLIPVisionConfig,
    )
    from .models.xglm import XGLMConfig
    from .models.xlm import XLMConfig, XLMTokenizer
    from .models.xlm_roberta import (
        XLMRobertaConfig,
    )
    from .models.xlm_roberta_xl import (
        XLMRobertaXLConfig,
    )
    from .models.xlnet import XLNetConfig
    from .models.xmod import XmodConfig
    from .models.yolos import YolosConfig
    from .models.yoso import YosoConfig
    from .models.zamba import ZambaConfig
    from .models.zoedepth import ZoeDepthConfig

    # Pipelines
    from .pipelines import (
        AudioClassificationPipeline,
        AutomaticSpeechRecognitionPipeline,
        CsvPipelineDataFormat,
        DepthEstimationPipeline,
        DocumentQuestionAnsweringPipeline,
        FeatureExtractionPipeline,
        FillMaskPipeline,
        ImageClassificationPipeline,
        ImageFeatureExtractionPipeline,
        ImageSegmentationPipeline,
        ImageTextToTextPipeline,
        ImageToImagePipeline,
        ImageToTextPipeline,
        JsonPipelineDataFormat,
        MaskGenerationPipeline,
        NerPipeline,
        ObjectDetectionPipeline,
        PipedPipelineDataFormat,
        Pipeline,
        PipelineDataFormat,
        QuestionAnsweringPipeline,
        SummarizationPipeline,
        TableQuestionAnsweringPipeline,
        Text2TextGenerationPipeline,
        TextClassificationPipeline,
        TextGenerationPipeline,
        TextToAudioPipeline,
        TokenClassificationPipeline,
        TranslationPipeline,
        VideoClassificationPipeline,
        VisualQuestionAnsweringPipeline,
        ZeroShotAudioClassificationPipeline,
        ZeroShotClassificationPipeline,
        ZeroShotImageClassificationPipeline,
        ZeroShotObjectDetectionPipeline,
        pipeline,
    )
    from .processing_utils import ProcessorMixin

    # Tokenization
    from .tokenization_utils import PreTrainedTokenizer
    from .tokenization_utils_base import (
        AddedToken,
        BatchEncoding,
        CharSpan,
        PreTrainedTokenizerBase,
        SpecialTokensMixin,
        TokenSpan,
    )

    # Trainer
    from .trainer_callback import (
        DefaultFlowCallback,
        EarlyStoppingCallback,
        PrinterCallback,
        ProgressCallback,
        TrainerCallback,
        TrainerControl,
        TrainerState,
    )
    from .trainer_utils import (
        EvalPrediction,
        IntervalStrategy,
        SchedulerType,
        enable_full_determinism,
        set_seed,
    )
    from .training_args import TrainingArguments
    from .training_args_seq2seq import Seq2SeqTrainingArguments
    from .training_args_tf import TFTrainingArguments

    # Files and general utilities
    from .utils import (
        CONFIG_NAME,
        MODEL_CARD_NAME,
        PYTORCH_PRETRAINED_BERT_CACHE,
        PYTORCH_TRANSFORMERS_CACHE,
        SPIECE_UNDERLINE,
        TF2_WEIGHTS_NAME,
        TF_WEIGHTS_NAME,
        TRANSFORMERS_CACHE,
        WEIGHTS_NAME,
        TensorType,
        add_end_docstrings,
        add_start_docstrings,
        is_apex_available,
        is_av_available,
        is_bitsandbytes_available,
        is_datasets_available,
        is_faiss_available,
        is_flax_available,
        is_keras_nlp_available,
        is_phonemizer_available,
        is_psutil_available,
        is_py3nvml_available,
        is_pyctcdecode_available,
        is_sacremoses_available,
        is_safetensors_available,
        is_scipy_available,
        is_sentencepiece_available,
        is_sklearn_available,
        is_speech_available,
        is_tensorflow_text_available,
        is_tf_available,
        is_timm_available,
        is_tokenizers_available,
        is_torch_available,
        is_torch_mlu_available,
        is_torch_musa_available,
        is_torch_neuroncore_available,
        is_torch_npu_available,
        is_torch_tpu_available,
        is_torch_xla_available,
        is_torch_xpu_available,
        is_torchvision_available,
        is_vision_available,
        logging,
    )

    # bitsandbytes config
    from .utils.quantization_config import (
        AqlmConfig,
        AwqConfig,
        BitNetConfig,
        BitsAndBytesConfig,
        CompressedTensorsConfig,
        EetqConfig,
        FbgemmFp8Config,
        GPTQConfig,
        HiggsConfig,
        HqqConfig,
        QuantoConfig,
        TorchAoConfig,
        VptqConfig,
    )

    try:
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_sentencepiece_objects import *
    else:
        from .models.albert import AlbertTokenizer
        from .models.barthez import BarthezTokenizer
        from .models.bartpho import BartphoTokenizer
        from .models.bert_generation import BertGenerationTokenizer
        from .models.big_bird import BigBirdTokenizer
        from .models.camembert import CamembertTokenizer
        from .models.code_llama import CodeLlamaTokenizer
        from .models.cpm import CpmTokenizer
        from .models.deberta_v2 import DebertaV2Tokenizer
        from .models.deprecated.ernie_m import ErnieMTokenizer
        from .models.deprecated.xlm_prophetnet import XLMProphetNetTokenizer
        from .models.fnet import FNetTokenizer
        from .models.gemma import GemmaTokenizer
        from .models.gpt_sw3 import GPTSw3Tokenizer
        from .models.layoutxlm import LayoutXLMTokenizer
        from .models.llama import LlamaTokenizer
        from .models.m2m_100 import M2M100Tokenizer
        from .models.marian import MarianTokenizer
        from .models.mbart import MBartTokenizer
        from .models.mbart50 import MBart50Tokenizer
        from .models.mluke import MLukeTokenizer
        from .models.mt5 import MT5Tokenizer
        from .models.nllb import NllbTokenizer
        from .models.pegasus import PegasusTokenizer
        from .models.plbart import PLBartTokenizer
        from .models.reformer import ReformerTokenizer
        from .models.rembert import RemBertTokenizer
        from .models.seamless_m4t import SeamlessM4TTokenizer
        from .models.siglip import SiglipTokenizer
        from .models.speech_to_text import Speech2TextTokenizer
        from .models.speecht5 import SpeechT5Tokenizer
        from .models.t5 import T5Tokenizer
        from .models.udop import UdopTokenizer
        from .models.xglm import XGLMTokenizer
        from .models.xlm_roberta import XLMRobertaTokenizer
        from .models.xlnet import XLNetTokenizer

    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_tokenizers_objects import *
    else:
        # Fast tokenizers imports
        from .models.albert import AlbertTokenizerFast
        from .models.bart import BartTokenizerFast
        from .models.barthez import BarthezTokenizerFast
        from .models.bert import BertTokenizerFast
        from .models.big_bird import BigBirdTokenizerFast
        from .models.blenderbot import BlenderbotTokenizerFast
        from .models.blenderbot_small import BlenderbotSmallTokenizerFast
        from .models.bloom import BloomTokenizerFast
        from .models.camembert import CamembertTokenizerFast
        from .models.clip import CLIPTokenizerFast
        from .models.code_llama import CodeLlamaTokenizerFast
        from .models.codegen import CodeGenTokenizerFast
        from .models.cohere import CohereTokenizerFast
        from .models.convbert import ConvBertTokenizerFast
        from .models.cpm import CpmTokenizerFast
        from .models.deberta import DebertaTokenizerFast
        from .models.deberta_v2 import DebertaV2TokenizerFast
        from .models.deprecated.realm import RealmTokenizerFast
        from .models.deprecated.retribert import RetriBertTokenizerFast
        from .models.distilbert import DistilBertTokenizerFast
        from .models.dpr import (
            DPRContextEncoderTokenizerFast,
            DPRQuestionEncoderTokenizerFast,
            DPRReaderTokenizerFast,
        )
        from .models.electra import ElectraTokenizerFast
        from .models.fnet import FNetTokenizerFast
        from .models.funnel import FunnelTokenizerFast
        from .models.gemma import GemmaTokenizerFast
        from .models.gpt2 import GPT2TokenizerFast
        from .models.gpt_neox import GPTNeoXTokenizerFast
        from .models.gpt_neox_japanese import GPTNeoXJapaneseTokenizer
        from .models.herbert import HerbertTokenizerFast
        from .models.layoutlm import LayoutLMTokenizerFast
        from .models.layoutlmv2 import LayoutLMv2TokenizerFast
        from .models.layoutlmv3 import LayoutLMv3TokenizerFast
        from .models.layoutxlm import LayoutXLMTokenizerFast
        from .models.led import LEDTokenizerFast
        from .models.llama import LlamaTokenizerFast
        from .models.longformer import LongformerTokenizerFast
        from .models.lxmert import LxmertTokenizerFast
        from .models.markuplm import MarkupLMTokenizerFast
        from .models.mbart import MBartTokenizerFast
        from .models.mbart50 import MBart50TokenizerFast
        from .models.mobilebert import MobileBertTokenizerFast
        from .models.mpnet import MPNetTokenizerFast
        from .models.mt5 import MT5TokenizerFast
        from .models.mvp import MvpTokenizerFast
        from .models.nllb import NllbTokenizerFast
        from .models.nougat import NougatTokenizerFast
        from .models.openai import OpenAIGPTTokenizerFast
        from .models.pegasus import PegasusTokenizerFast
        from .models.qwen2 import Qwen2TokenizerFast
        from .models.reformer import ReformerTokenizerFast
        from .models.rembert import RemBertTokenizerFast
        from .models.roberta import RobertaTokenizerFast
        from .models.roformer import RoFormerTokenizerFast
        from .models.seamless_m4t import SeamlessM4TTokenizerFast
        from .models.splinter import SplinterTokenizerFast
        from .models.squeezebert import SqueezeBertTokenizerFast
        from .models.t5 import T5TokenizerFast
        from .models.udop import UdopTokenizerFast
        from .models.whisper import WhisperTokenizerFast
        from .models.xglm import XGLMTokenizerFast
        from .models.xlm_roberta import XLMRobertaTokenizerFast
        from .models.xlnet import XLNetTokenizerFast
        from .tokenization_utils_fast import PreTrainedTokenizerFast

    try:
        if not (is_sentencepiece_available() and is_tokenizers_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummies_sentencepiece_and_tokenizers_objects import *
    else:
        from .convert_slow_tokenizer import (
            SLOW_TO_FAST_CONVERTERS,
            convert_slow_tokenizer,
        )

    try:
        if not is_tensorflow_text_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_tensorflow_text_objects import *
    else:
        from .models.bert import TFBertTokenizer

    try:
        if not is_keras_nlp_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_keras_nlp_objects import *
    else:
        from .models.gpt2 import TFGPT2Tokenizer

    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_vision_objects import *
    else:
        from .image_processing_base import ImageProcessingMixin
        from .image_processing_utils import BaseImageProcessor
        from .image_utils import ImageFeatureExtractionMixin
        from .models.aria import AriaImageProcessor
        from .models.beit import BeitFeatureExtractor, BeitImageProcessor
        from .models.bit import BitImageProcessor
        from .models.blip import BlipImageProcessor
        from .models.bridgetower import BridgeTowerImageProcessor
        from .models.chameleon import ChameleonImageProcessor
        from .models.chinese_clip import (
            ChineseCLIPFeatureExtractor,
            ChineseCLIPImageProcessor,
        )
        from .models.clip import CLIPFeatureExtractor, CLIPImageProcessor
        from .models.conditional_detr import (
            ConditionalDetrFeatureExtractor,
            ConditionalDetrImageProcessor,
        )
        from .models.convnext import ConvNextFeatureExtractor, ConvNextImageProcessor
        from .models.deformable_detr import DeformableDetrFeatureExtractor, DeformableDetrImageProcessor
        from .models.deit import DeiTFeatureExtractor, DeiTImageProcessor
        from .models.deprecated.deta import DetaImageProcessor
        from .models.deprecated.efficientformer import EfficientFormerImageProcessor
        from .models.deprecated.tvlt import TvltImageProcessor
        from .models.deprecated.vit_hybrid import ViTHybridImageProcessor
        from .models.detr import DetrFeatureExtractor, DetrImageProcessor
        from .models.donut import DonutFeatureExtractor, DonutImageProcessor
        from .models.dpt import DPTFeatureExtractor, DPTImageProcessor
        from .models.efficientnet import EfficientNetImageProcessor
        from .models.flava import (
            FlavaFeatureExtractor,
            FlavaImageProcessor,
            FlavaProcessor,
        )
        from .models.fuyu import FuyuImageProcessor, FuyuProcessor
        from .models.glpn import GLPNFeatureExtractor, GLPNImageProcessor
        from .models.grounding_dino import GroundingDinoImageProcessor
        from .models.idefics import IdeficsImageProcessor
        from .models.idefics2 import Idefics2ImageProcessor
        from .models.idefics3 import Idefics3ImageProcessor
        from .models.imagegpt import ImageGPTFeatureExtractor, ImageGPTImageProcessor
        from .models.instructblipvideo import InstructBlipVideoImageProcessor
        from .models.layoutlmv2 import (
            LayoutLMv2FeatureExtractor,
            LayoutLMv2ImageProcessor,
        )
        from .models.layoutlmv3 import (
            LayoutLMv3FeatureExtractor,
            LayoutLMv3ImageProcessor,
        )
        from .models.levit import LevitFeatureExtractor, LevitImageProcessor
        from .models.llava_next import LlavaNextImageProcessor
        from .models.llava_next_video import LlavaNextVideoImageProcessor
        from .models.llava_onevision import LlavaOnevisionImageProcessor, LlavaOnevisionVideoProcessor
        from .models.mask2former import Mask2FormerImageProcessor
        from .models.maskformer import (
            MaskFormerFeatureExtractor,
            MaskFormerImageProcessor,
        )
        from .models.mllama import MllamaImageProcessor
        from .models.mobilenet_v1 import (
            MobileNetV1FeatureExtractor,
            MobileNetV1ImageProcessor,
        )
        from .models.mobilenet_v2 import (
            MobileNetV2FeatureExtractor,
            MobileNetV2ImageProcessor,
        )
        from .models.mobilevit import MobileViTFeatureExtractor, MobileViTImageProcessor
        from .models.nougat import NougatImageProcessor
        from .models.oneformer import OneFormerImageProcessor
        from .models.owlv2 import Owlv2ImageProcessor
        from .models.owlvit import OwlViTFeatureExtractor, OwlViTImageProcessor
        from .models.perceiver import PerceiverFeatureExtractor, PerceiverImageProcessor
        from .models.pix2struct import Pix2StructImageProcessor
        from .models.pixtral import PixtralImageProcessor
        from .models.poolformer import (
            PoolFormerFeatureExtractor,
            PoolFormerImageProcessor,
        )
        from .models.pvt import PvtImageProcessor
        from .models.qwen2_vl import Qwen2VLImageProcessor
        from .models.rt_detr import RTDetrImageProcessor
        from .models.sam import SamImageProcessor
        from .models.segformer import SegformerFeatureExtractor, SegformerImageProcessor
        from .models.seggpt import SegGptImageProcessor
        from .models.siglip import SiglipImageProcessor
        from .models.superpoint import SuperPointImageProcessor
        from .models.swin2sr import Swin2SRImageProcessor
        from .models.tvp import TvpImageProcessor
        from .models.video_llava import VideoLlavaImageProcessor
        from .models.videomae import VideoMAEFeatureExtractor, VideoMAEImageProcessor
        from .models.vilt import ViltFeatureExtractor, ViltImageProcessor, ViltProcessor
        from .models.vit import ViTFeatureExtractor, ViTImageProcessor
        from .models.vitmatte import VitMatteImageProcessor
        from .models.vivit import VivitImageProcessor
        from .models.yolos import YolosFeatureExtractor, YolosImageProcessor
        from .models.zoedepth import ZoeDepthImageProcessor

    try:
        if not is_torchvision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_torchvision_objects import *
    else:
        from .image_processing_utils_fast import BaseImageProcessorFast
        from .models.deformable_detr import DeformableDetrImageProcessorFast
        from .models.detr import DetrImageProcessorFast
        from .models.pixtral import PixtralImageProcessorFast
        from .models.rt_detr import RTDetrImageProcessorFast
        from .models.vit import ViTImageProcessorFast

    try:
        if not is_torchvision_available() and not is_timm_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_timm_and_torchvision_objects import *
    else:
        from .models.timm_wrapper import TimmWrapperImageProcessor

    # Modeling
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_pt_objects import *
    else:
        # Benchmarks
        from .benchmark.benchmark import PyTorchBenchmark
        from .benchmark.benchmark_args import PyTorchBenchmarkArguments
        from .cache_utils import (
            Cache,
            CacheConfig,
            DynamicCache,
            EncoderDecoderCache,
            HQQQuantizedCache,
            HybridCache,
            MambaCache,
            OffloadedCache,
            OffloadedStaticCache,
            QuantizedCache,
            QuantizedCacheConfig,
            QuantoQuantizedCache,
            SinkCache,
            SlidingWindowCache,
            StaticCache,
        )
        from .data.datasets import (
            GlueDataset,
            GlueDataTrainingArguments,
            LineByLineTextDataset,
            LineByLineWithRefDataset,
            LineByLineWithSOPTextDataset,
            SquadDataset,
            SquadDataTrainingArguments,
            TextDataset,
            TextDatasetForNextSentencePrediction,
        )
        from .generation import (
            AlternatingCodebooksLogitsProcessor,
            BayesianDetectorConfig,
            BayesianDetectorModel,
            BeamScorer,
            BeamSearchScorer,
            ClassifierFreeGuidanceLogitsProcessor,
            ConstrainedBeamSearchScorer,
            Constraint,
            ConstraintListState,
            DisjunctiveConstraint,
            EncoderNoRepeatNGramLogitsProcessor,
            EncoderRepetitionPenaltyLogitsProcessor,
            EosTokenCriteria,
            EpsilonLogitsWarper,
            EtaLogitsWarper,
            ExponentialDecayLengthPenalty,
            ForcedBOSTokenLogitsProcessor,
            ForcedEOSTokenLogitsProcessor,
            GenerationMixin,
            HammingDiversityLogitsProcessor,
            InfNanRemoveLogitsProcessor,
            LogitNormalization,
            LogitsProcessor,
            LogitsProcessorList,
            LogitsWarper,
            MaxLengthCriteria,
            MaxTimeCriteria,
            MinLengthLogitsProcessor,
            MinNewTokensLengthLogitsProcessor,
            MinPLogitsWarper,
            NoBadWordsLogitsProcessor,
            NoRepeatNGramLogitsProcessor,
            PhrasalConstraint,
            PrefixConstrainedLogitsProcessor,
            RepetitionPenaltyLogitsProcessor,
            SequenceBiasLogitsProcessor,
            StoppingCriteria,
            StoppingCriteriaList,
            StopStringCriteria,
            SuppressTokensAtBeginLogitsProcessor,
            SuppressTokensLogitsProcessor,
            SynthIDTextWatermarkDetector,
            SynthIDTextWatermarkingConfig,
            SynthIDTextWatermarkLogitsProcessor,
            TemperatureLogitsWarper,
            TopKLogitsWarper,
            TopPLogitsWarper,
            TypicalLogitsWarper,
            UnbatchedClassifierFreeGuidanceLogitsProcessor,
            WatermarkDetector,
            WatermarkLogitsProcessor,
            WhisperTimeStampLogitsProcessor,
        )
        from .integrations.executorch import (
            TorchExportableModuleWithStaticCache,
            convert_and_export_with_cache,
        )
        from .modeling_rope_utils import ROPE_INIT_FUNCTIONS
        from .modeling_utils import PreTrainedModel
        from .models.albert import (
            AlbertForMaskedLM,
            AlbertForMultipleChoice,
            AlbertForPreTraining,
            AlbertForQuestionAnswering,
            AlbertForSequenceClassification,
            AlbertForTokenClassification,
            AlbertModel,
            AlbertPreTrainedModel,
            load_tf_weights_in_albert,
        )
        from .models.align import (
            AlignModel,
            AlignPreTrainedModel,
            AlignTextModel,
            AlignVisionModel,
        )
        from .models.altclip import (
            AltCLIPModel,
            AltCLIPPreTrainedModel,
            AltCLIPTextModel,
            AltCLIPVisionModel,
        )
        from .models.aria import (
            AriaForConditionalGeneration,
            AriaPreTrainedModel,
            AriaTextForCausalLM,
            AriaTextModel,
            AriaTextPreTrainedModel,
        )
        from .models.audio_spectrogram_transformer import (
            ASTForAudioClassification,
            ASTModel,
            ASTPreTrainedModel,
        )
        from .models.auto import (
            MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING,
            MODEL_FOR_AUDIO_FRAME_CLASSIFICATION_MAPPING,
            MODEL_FOR_AUDIO_XVECTOR_MAPPING,
            MODEL_FOR_BACKBONE_MAPPING,
            MODEL_FOR_CAUSAL_IMAGE_MODELING_MAPPING,
            MODEL_FOR_CAUSAL_LM_MAPPING,
            MODEL_FOR_CTC_MAPPING,
            MODEL_FOR_DEPTH_ESTIMATION_MAPPING,
            MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING,
            MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
            MODEL_FOR_IMAGE_MAPPING,
            MODEL_FOR_IMAGE_SEGMENTATION_MAPPING,
            MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING,
            MODEL_FOR_IMAGE_TO_IMAGE_MAPPING,
            MODEL_FOR_INSTANCE_SEGMENTATION_MAPPING,
            MODEL_FOR_KEYPOINT_DETECTION_MAPPING,
            MODEL_FOR_MASK_GENERATION_MAPPING,
            MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING,
            MODEL_FOR_MASKED_LM_MAPPING,
            MODEL_FOR_MULTIPLE_CHOICE_MAPPING,
            MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING,
            MODEL_FOR_OBJECT_DETECTION_MAPPING,
            MODEL_FOR_PRETRAINING_MAPPING,
            MODEL_FOR_QUESTION_ANSWERING_MAPPING,
            MODEL_FOR_RETRIEVAL_MAPPING,
            MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING,
            MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
            MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
            MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING,
            MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING,
            MODEL_FOR_TEXT_ENCODING_MAPPING,
            MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING,
            MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING,
            MODEL_FOR_TIME_SERIES_CLASSIFICATION_MAPPING,
            MODEL_FOR_TIME_SERIES_REGRESSION_MAPPING,
            MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
            MODEL_FOR_UNIVERSAL_SEGMENTATION_MAPPING,
            MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING,
            MODEL_FOR_VISION_2_SEQ_MAPPING,
            MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING,
            MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING,
            MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING,
            MODEL_MAPPING,
            MODEL_WITH_LM_HEAD_MAPPING,
            AutoBackbone,
            AutoModel,
            AutoModelForAudioClassification,
            AutoModelForAudioFrameClassification,
            AutoModelForAudioXVector,
            AutoModelForCausalLM,
            AutoModelForCTC,
            AutoModelForDepthEstimation,
            AutoModelForDocumentQuestionAnswering,
            AutoModelForImageClassification,
            AutoModelForImageSegmentation,
            AutoModelForImageTextToText,
            AutoModelForImageToImage,
            AutoModelForInstanceSegmentation,
            AutoModelForKeypointDetection,
            AutoModelForMaskedImageModeling,
            AutoModelForMaskedLM,
            AutoModelForMaskGeneration,
            AutoModelForMultipleChoice,
            AutoModelForNextSentencePrediction,
            AutoModelForObjectDetection,
            AutoModelForPreTraining,
            AutoModelForQuestionAnswering,
            AutoModelForSemanticSegmentation,
            AutoModelForSeq2SeqLM,
            AutoModelForSequenceClassification,
            AutoModelForSpeechSeq2Seq,
            AutoModelForTableQuestionAnswering,
            AutoModelForTextEncoding,
            AutoModelForTextToSpectrogram,
            AutoModelForTextToWaveform,
            AutoModelForTokenClassification,
            AutoModelForUniversalSegmentation,
            AutoModelForVideoClassification,
            AutoModelForVision2Seq,
            AutoModelForVisualQuestionAnswering,
            AutoModelForZeroShotImageClassification,
            AutoModelForZeroShotObjectDetection,
            AutoModelWithLMHead,
        )
        from .models.autoformer import (
            AutoformerForPrediction,
            AutoformerModel,
            AutoformerPreTrainedModel,
        )
        from .models.bamba import BambaForCausalLM, BambaModel, BambaPreTrainedModel
        from .models.bark import (
            BarkCausalModel,
            BarkCoarseModel,
            BarkFineModel,
            BarkModel,
            BarkPreTrainedModel,
            BarkSemanticModel,
        )
        from .models.bart import (
            BartForCausalLM,
            BartForConditionalGeneration,
            BartForQuestionAnswering,
            BartForSequenceClassification,
            BartModel,
            BartPreTrainedModel,
            BartPretrainedModel,
            PretrainedBartModel,
        )
        from .models.beit import (
            BeitBackbone,
            BeitForImageClassification,
            BeitForMaskedImageModeling,
            BeitForSemanticSegmentation,
            BeitModel,
            BeitPreTrainedModel,
        )
        from .models.bert import (
            BertForMaskedLM,
            BertForMultipleChoice,
            BertForNextSentencePrediction,
            BertForPreTraining,
            BertForQuestionAnswering,
            BertForSequenceClassification,
            BertForTokenClassification,
            BertLMHeadModel,
            BertModel,
            BertPreTrainedModel,
            load_tf_weights_in_bert,
        )
        from .models.bert_generation import (
            BertGenerationDecoder,
            BertGenerationEncoder,
            BertGenerationPreTrainedModel,
            load_tf_weights_in_bert_generation,
        )
        from .models.big_bird import (
            BigBirdForCausalLM,
            BigBirdForMaskedLM,
            BigBirdForMultipleChoice,
            BigBirdForPreTraining,
            BigBirdForQuestionAnswering,
            BigBirdForSequenceClassification,
            BigBirdForTokenClassification,
            BigBirdModel,
            BigBirdPreTrainedModel,
            load_tf_weights_in_big_bird,
        )
        from .models.bigbird_pegasus import (
            BigBirdPegasusForCausalLM,
            BigBirdPegasusForConditionalGeneration,
            BigBirdPegasusForQuestionAnswering,
            BigBirdPegasusForSequenceClassification,
            BigBirdPegasusModel,
            BigBirdPegasusPreTrainedModel,
        )
        from .models.biogpt import (
            BioGptForCausalLM,
            BioGptForSequenceClassification,
            BioGptForTokenClassification,
            BioGptModel,
            BioGptPreTrainedModel,
        )
        from .models.bit import (
            BitBackbone,
            BitForImageClassification,
            BitModel,
            BitPreTrainedModel,
        )
        from .models.blenderbot import (
            BlenderbotForCausalLM,
            BlenderbotForConditionalGeneration,
            BlenderbotModel,
            BlenderbotPreTrainedModel,
        )
        from .models.blenderbot_small import (
            BlenderbotSmallForCausalLM,
            BlenderbotSmallForConditionalGeneration,
            BlenderbotSmallModel,
            BlenderbotSmallPreTrainedModel,
        )
        from .models.blip import (
            BlipForConditionalGeneration,
            BlipForImageTextRetrieval,
            BlipForQuestionAnswering,
            BlipModel,
            BlipPreTrainedModel,
            BlipTextModel,
            BlipVisionModel,
        )
        from .models.blip_2 import (
            Blip2ForConditionalGeneration,
            Blip2ForImageTextRetrieval,
            Blip2Model,
            Blip2PreTrainedModel,
            Blip2QFormerModel,
            Blip2TextModelWithProjection,
            Blip2VisionModel,
            Blip2VisionModelWithProjection,
        )
        from .models.bloom import (
            BloomForCausalLM,
            BloomForQuestionAnswering,
            BloomForSequenceClassification,
            BloomForTokenClassification,
            BloomModel,
            BloomPreTrainedModel,
        )
        from .models.bridgetower import (
            BridgeTowerForContrastiveLearning,
            BridgeTowerForImageAndTextRetrieval,
            BridgeTowerForMaskedLM,
            BridgeTowerModel,
            BridgeTowerPreTrainedModel,
        )
        from .models.bros import (
            BrosForTokenClassification,
            BrosModel,
            BrosPreTrainedModel,
            BrosProcessor,
            BrosSpadeEEForTokenClassification,
            BrosSpadeELForTokenClassification,
        )
        from .models.camembert import (
            CamembertForCausalLM,
            CamembertForMaskedLM,
            CamembertForMultipleChoice,
            CamembertForQuestionAnswering,
            CamembertForSequenceClassification,
            CamembertForTokenClassification,
            CamembertModel,
            CamembertPreTrainedModel,
        )
        from .models.canine import (
            CanineForMultipleChoice,
            CanineForQuestionAnswering,
            CanineForSequenceClassification,
            CanineForTokenClassification,
            CanineModel,
            CaninePreTrainedModel,
            load_tf_weights_in_canine,
        )
        from .models.chameleon import (
            ChameleonForConditionalGeneration,
            ChameleonModel,
            ChameleonPreTrainedModel,
            ChameleonProcessor,
            ChameleonVQVAE,
        )
        from .models.chinese_clip import (
            ChineseCLIPModel,
            ChineseCLIPPreTrainedModel,
            ChineseCLIPTextModel,
            ChineseCLIPVisionModel,
        )
        from .models.clap import (
            ClapAudioModel,
            ClapAudioModelWithProjection,
            ClapFeatureExtractor,
            ClapModel,
            ClapPreTrainedModel,
            ClapTextModel,
            ClapTextModelWithProjection,
        )
        from .models.clip import (
            CLIPForImageClassification,
            CLIPModel,
            CLIPPreTrainedModel,
            CLIPTextModel,
            CLIPTextModelWithProjection,
            CLIPVisionModel,
            CLIPVisionModelWithProjection,
        )
        from .models.clipseg import (
            CLIPSegForImageSegmentation,
            CLIPSegModel,
            CLIPSegPreTrainedModel,
            CLIPSegTextModel,
            CLIPSegVisionModel,
        )
        from .models.clvp import (
            ClvpDecoder,
            ClvpEncoder,
            ClvpForCausalLM,
            ClvpModel,
            ClvpModelForConditionalGeneration,
            ClvpPreTrainedModel,
        )
        from .models.codegen import (
            CodeGenForCausalLM,
            CodeGenModel,
            CodeGenPreTrainedModel,
        )
        from .models.cohere import (
            CohereForCausalLM,
            CohereModel,
            CoherePreTrainedModel,
        )
        from .models.cohere2 import (
            Cohere2ForCausalLM,
            Cohere2Model,
            Cohere2PreTrainedModel,
        )
        from .models.colpali import (
            ColPaliForRetrieval,
            ColPaliPreTrainedModel,
        )
        from .models.conditional_detr import (
            ConditionalDetrForObjectDetection,
            ConditionalDetrForSegmentation,
            ConditionalDetrModel,
            ConditionalDetrPreTrainedModel,
        )
        from .models.convbert import (
            ConvBertForMaskedLM,
            ConvBertForMultipleChoice,
            ConvBertForQuestionAnswering,
            ConvBertForSequenceClassification,
            ConvBertForTokenClassification,
            ConvBertModel,
            ConvBertPreTrainedModel,
            load_tf_weights_in_convbert,
        )
        from .models.convnext import (
            ConvNextBackbone,
            ConvNextForImageClassification,
            ConvNextModel,
            ConvNextPreTrainedModel,
        )
        from .models.convnextv2 import (
            ConvNextV2Backbone,
            ConvNextV2ForImageClassification,
            ConvNextV2Model,
            ConvNextV2PreTrainedModel,
        )
        from .models.cpmant import (
            CpmAntForCausalLM,
            CpmAntModel,
            CpmAntPreTrainedModel,
        )
        from .models.ctrl import (
            CTRLForSequenceClassification,
            CTRLLMHeadModel,
            CTRLModel,
            CTRLPreTrainedModel,
        )
        from .models.cvt import (
            CvtForImageClassification,
            CvtModel,
            CvtPreTrainedModel,
        )
        from .models.dac import (
            DacModel,
            DacPreTrainedModel,
        )
        from .models.data2vec import (
            Data2VecAudioForAudioFrameClassification,
            Data2VecAudioForCTC,
            Data2VecAudioForSequenceClassification,
            Data2VecAudioForXVector,
            Data2VecAudioModel,
            Data2VecAudioPreTrainedModel,
            Data2VecTextForCausalLM,
            Data2VecTextForMaskedLM,
            Data2VecTextForMultipleChoice,
            Data2VecTextForQuestionAnswering,
            Data2VecTextForSequenceClassification,
            Data2VecTextForTokenClassification,
            Data2VecTextModel,
            Data2VecTextPreTrainedModel,
            Data2VecVisionForImageClassification,
            Data2VecVisionForSemanticSegmentation,
            Data2VecVisionModel,
            Data2VecVisionPreTrainedModel,
        )

        # PyTorch model imports
        from .models.dbrx import (
            DbrxForCausalLM,
            DbrxModel,
            DbrxPreTrainedModel,
        )
        from .models.deberta import (
            DebertaForMaskedLM,
            DebertaForQuestionAnswering,
            DebertaForSequenceClassification,
            DebertaForTokenClassification,
            DebertaModel,
            DebertaPreTrainedModel,
        )
        from .models.deberta_v2 import (
            DebertaV2ForMaskedLM,
            DebertaV2ForMultipleChoice,
            DebertaV2ForQuestionAnswering,
            DebertaV2ForSequenceClassification,
            DebertaV2ForTokenClassification,
            DebertaV2Model,
            DebertaV2PreTrainedModel,
        )
        from .models.decision_transformer import (
            DecisionTransformerGPT2Model,
            DecisionTransformerGPT2PreTrainedModel,
            DecisionTransformerModel,
            DecisionTransformerPreTrainedModel,
        )
        from .models.deformable_detr import (
            DeformableDetrForObjectDetection,
            DeformableDetrModel,
            DeformableDetrPreTrainedModel,
        )
        from .models.deit import (
            DeiTForImageClassification,
            DeiTForImageClassificationWithTeacher,
            DeiTForMaskedImageModeling,
            DeiTModel,
            DeiTPreTrainedModel,
        )
        from .models.deprecated.deta import (
            DetaForObjectDetection,
            DetaModel,
            DetaPreTrainedModel,
        )
        from .models.deprecated.efficientformer import (
            EfficientFormerForImageClassification,
            EfficientFormerForImageClassificationWithTeacher,
            EfficientFormerModel,
            EfficientFormerPreTrainedModel,
        )
        from .models.deprecated.ernie_m import (
            ErnieMForInformationExtraction,
            ErnieMForMultipleChoice,
            ErnieMForQuestionAnswering,
            ErnieMForSequenceClassification,
            ErnieMForTokenClassification,
            ErnieMModel,
            ErnieMPreTrainedModel,
        )
        from .models.deprecated.gptsan_japanese import (
            GPTSanJapaneseForConditionalGeneration,
            GPTSanJapaneseModel,
            GPTSanJapanesePreTrainedModel,
        )
        from .models.deprecated.graphormer import (
            GraphormerForGraphClassification,
            GraphormerModel,
            GraphormerPreTrainedModel,
        )
        from .models.deprecated.jukebox import (
            JukeboxModel,
            JukeboxPreTrainedModel,
            JukeboxPrior,
            JukeboxVQVAE,
        )
        from .models.deprecated.mctct import (
            MCTCTForCTC,
            MCTCTModel,
            MCTCTPreTrainedModel,
        )
        from .models.deprecated.mega import (
            MegaForCausalLM,
            MegaForMaskedLM,
            MegaForMultipleChoice,
            MegaForQuestionAnswering,
            MegaForSequenceClassification,
            MegaForTokenClassification,
            MegaModel,
            MegaPreTrainedModel,
        )
        from .models.deprecated.mmbt import (
            MMBTForClassification,
            MMBTModel,
            ModalEmbeddings,
        )
        from .models.deprecated.nat import (
            NatBackbone,
            NatForImageClassification,
            NatModel,
            NatPreTrainedModel,
        )
        from .models.deprecated.nezha import (
            NezhaForMaskedLM,
            NezhaForMultipleChoice,
            NezhaForNextSentencePrediction,
            NezhaForPreTraining,
            NezhaForQuestionAnswering,
            NezhaForSequenceClassification,
            NezhaForTokenClassification,
            NezhaModel,
            NezhaPreTrainedModel,
        )
        from .models.deprecated.open_llama import (
            OpenLlamaForCausalLM,
            OpenLlamaForSequenceClassification,
            OpenLlamaModel,
            OpenLlamaPreTrainedModel,
        )
        from .models.deprecated.qdqbert import (
            QDQBertForMaskedLM,
            QDQBertForMultipleChoice,
            QDQBertForNextSentencePrediction,
            QDQBertForQuestionAnswering,
            QDQBertForSequenceClassification,
            QDQBertForTokenClassification,
            QDQBertLMHeadModel,
            QDQBertModel,
            QDQBertPreTrainedModel,
            load_tf_weights_in_qdqbert,
        )
        from .models.deprecated.realm import (
            RealmEmbedder,
            RealmForOpenQA,
            RealmKnowledgeAugEncoder,
            RealmPreTrainedModel,
            RealmReader,
            RealmRetriever,
            RealmScorer,
            load_tf_weights_in_realm,
        )
        from .models.deprecated.retribert import (
            RetriBertModel,
            RetriBertPreTrainedModel,
        )
        from .models.deprecated.speech_to_text_2 import (
            Speech2Text2ForCausalLM,
            Speech2Text2PreTrainedModel,
        )
        from .models.deprecated.trajectory_transformer import (
            TrajectoryTransformerModel,
            TrajectoryTransformerPreTrainedModel,
        )
        from .models.deprecated.transfo_xl import (
            AdaptiveEmbedding,
            TransfoXLForSequenceClassification,
            TransfoXLLMHeadModel,
            TransfoXLModel,
            TransfoXLPreTrainedModel,
            load_tf_weights_in_transfo_xl,
        )
        from .models.deprecated.tvlt import (
            TvltForAudioVisualClassification,
            TvltForPreTraining,
            TvltModel,
            TvltPreTrainedModel,
        )
        from .models.deprecated.van import (
            VanForImageClassification,
            VanModel,
            VanPreTrainedModel,
        )
        from .models.deprecated.vit_hybrid import (
            ViTHybridForImageClassification,
            ViTHybridModel,
            ViTHybridPreTrainedModel,
        )
        from .models.deprecated.xlm_prophetnet import (
            XLMProphetNetDecoder,
            XLMProphetNetEncoder,
            XLMProphetNetForCausalLM,
            XLMProphetNetForConditionalGeneration,
            XLMProphetNetModel,
            XLMProphetNetPreTrainedModel,
        )
        from .models.depth_anything import (
            DepthAnythingForDepthEstimation,
            DepthAnythingPreTrainedModel,
        )
        from .models.detr import (
            DetrForObjectDetection,
            DetrForSegmentation,
            DetrModel,
            DetrPreTrainedModel,
        )
        from .models.diffllama import (
            DiffLlamaForCausalLM,
            DiffLlamaForQuestionAnswering,
            DiffLlamaForSequenceClassification,
            DiffLlamaForTokenClassification,
            DiffLlamaModel,
            DiffLlamaPreTrainedModel,
        )
        from .models.dinat import (
            DinatBackbone,
            DinatForImageClassification,
            DinatModel,
            DinatPreTrainedModel,
        )
        from .models.dinov2 import (
            Dinov2Backbone,
            Dinov2ForImageClassification,
            Dinov2Model,
            Dinov2PreTrainedModel,
        )
        from .models.dinov2_with_registers import (
            Dinov2WithRegistersBackbone,
            Dinov2WithRegistersForImageClassification,
            Dinov2WithRegistersModel,
            Dinov2WithRegistersPreTrainedModel,
        )
        from .models.distilbert import (
            DistilBertForMaskedLM,
            DistilBertForMultipleChoice,
            DistilBertForQuestionAnswering,
            DistilBertForSequenceClassification,
            DistilBertForTokenClassification,
            DistilBertModel,
            DistilBertPreTrainedModel,
        )
        from .models.donut import (
            DonutSwinModel,
            DonutSwinPreTrainedModel,
        )
        from .models.dpr import (
            DPRContextEncoder,
            DPRPretrainedContextEncoder,
            DPRPreTrainedModel,
            DPRPretrainedQuestionEncoder,
            DPRPretrainedReader,
            DPRQuestionEncoder,
            DPRReader,
        )
        from .models.dpt import (
            DPTForDepthEstimation,
            DPTForSemanticSegmentation,
            DPTModel,
            DPTPreTrainedModel,
        )
        from .models.efficientnet import (
            EfficientNetForImageClassification,
            EfficientNetModel,
            EfficientNetPreTrainedModel,
        )
        from .models.electra import (
            ElectraForCausalLM,
            ElectraForMaskedLM,
            ElectraForMultipleChoice,
            ElectraForPreTraining,
            ElectraForQuestionAnswering,
            ElectraForSequenceClassification,
            ElectraForTokenClassification,
            ElectraModel,
            ElectraPreTrainedModel,
            load_tf_weights_in_electra,
        )
        from .models.encodec import (
            EncodecModel,
            EncodecPreTrainedModel,
        )
        from .models.encoder_decoder import EncoderDecoderModel
        from .models.ernie import (
            ErnieForCausalLM,
            ErnieForMaskedLM,
            ErnieForMultipleChoice,
            ErnieForNextSentencePrediction,
            ErnieForPreTraining,
            ErnieForQuestionAnswering,
            ErnieForSequenceClassification,
            ErnieForTokenClassification,
            ErnieModel,
            ErniePreTrainedModel,
        )
        from .models.esm import (
            EsmFoldPreTrainedModel,
            EsmForMaskedLM,
            EsmForProteinFolding,
            EsmForSequenceClassification,
            EsmForTokenClassification,
            EsmModel,
            EsmPreTrainedModel,
        )
        from .models.falcon import (
            FalconForCausalLM,
            FalconForQuestionAnswering,
            FalconForSequenceClassification,
            FalconForTokenClassification,
            FalconModel,
            FalconPreTrainedModel,
        )
        from .models.falcon_mamba import (
            FalconMambaForCausalLM,
            FalconMambaModel,
            FalconMambaPreTrainedModel,
        )
        from .models.fastspeech2_conformer import (
            FastSpeech2ConformerHifiGan,
            FastSpeech2ConformerModel,
            FastSpeech2ConformerPreTrainedModel,
            FastSpeech2ConformerWithHifiGan,
        )
        from .models.flaubert import (
            FlaubertForMultipleChoice,
            FlaubertForQuestionAnswering,
            FlaubertForQuestionAnsweringSimple,
            FlaubertForSequenceClassification,
            FlaubertForTokenClassification,
            FlaubertModel,
            FlaubertPreTrainedModel,
            FlaubertWithLMHeadModel,
        )
        from .models.flava import (
            FlavaForPreTraining,
            FlavaImageCodebook,
            FlavaImageModel,
            FlavaModel,
            FlavaMultimodalModel,
            FlavaPreTrainedModel,
            FlavaTextModel,
        )
        from .models.fnet import (
            FNetForMaskedLM,
            FNetForMultipleChoice,
            FNetForNextSentencePrediction,
            FNetForPreTraining,
            FNetForQuestionAnswering,
            FNetForSequenceClassification,
            FNetForTokenClassification,
            FNetModel,
            FNetPreTrainedModel,
        )
        from .models.focalnet import (
            FocalNetBackbone,
            FocalNetForImageClassification,
            FocalNetForMaskedImageModeling,
            FocalNetModel,
            FocalNetPreTrainedModel,
        )
        from .models.fsmt import (
            FSMTForConditionalGeneration,
            FSMTModel,
            PretrainedFSMTModel,
        )
        from .models.funnel import (
            FunnelBaseModel,
            FunnelForMaskedLM,
            FunnelForMultipleChoice,
            FunnelForPreTraining,
            FunnelForQuestionAnswering,
            FunnelForSequenceClassification,
            FunnelForTokenClassification,
            FunnelModel,
            FunnelPreTrainedModel,
            load_tf_weights_in_funnel,
        )
        from .models.fuyu import (
            FuyuForCausalLM,
            FuyuPreTrainedModel,
        )
        from .models.gemma import (
            GemmaForCausalLM,
            GemmaForSequenceClassification,
            GemmaForTokenClassification,
            GemmaModel,
            GemmaPreTrainedModel,
        )
        from .models.gemma2 import (
            Gemma2ForCausalLM,
            Gemma2ForSequenceClassification,
            Gemma2ForTokenClassification,
            Gemma2Model,
            Gemma2PreTrainedModel,
        )
        from .models.git import (
            GitForCausalLM,
            GitModel,
            GitPreTrainedModel,
            GitVisionModel,
        )
        from .models.glm import (
            GlmForCausalLM,
            GlmForSequenceClassification,
            GlmForTokenClassification,
            GlmModel,
            GlmPreTrainedModel,
        )
        from .models.glpn import (
            GLPNForDepthEstimation,
            GLPNModel,
            GLPNPreTrainedModel,
        )
        from .models.gpt2 import (
            GPT2DoubleHeadsModel,
            GPT2ForQuestionAnswering,
            GPT2ForSequenceClassification,
            GPT2ForTokenClassification,
            GPT2LMHeadModel,
            GPT2Model,
            GPT2PreTrainedModel,
            load_tf_weights_in_gpt2,
        )
        from .models.gpt_bigcode import (
            GPTBigCodeForCausalLM,
            GPTBigCodeForSequenceClassification,
            GPTBigCodeForTokenClassification,
            GPTBigCodeModel,
            GPTBigCodePreTrainedModel,
        )
        from .models.gpt_neo import (
            GPTNeoForCausalLM,
            GPTNeoForQuestionAnswering,
            GPTNeoForSequenceClassification,
            GPTNeoForTokenClassification,
            GPTNeoModel,
            GPTNeoPreTrainedModel,
            load_tf_weights_in_gpt_neo,
        )
        from .models.gpt_neox import (
            GPTNeoXForCausalLM,
            GPTNeoXForQuestionAnswering,
            GPTNeoXForSequenceClassification,
            GPTNeoXForTokenClassification,
            GPTNeoXModel,
            GPTNeoXPreTrainedModel,
        )
        from .models.gpt_neox_japanese import (
            GPTNeoXJapaneseForCausalLM,
            GPTNeoXJapaneseModel,
            GPTNeoXJapanesePreTrainedModel,
        )
        from .models.gptj import (
            GPTJForCausalLM,
            GPTJForQuestionAnswering,
            GPTJForSequenceClassification,
            GPTJModel,
            GPTJPreTrainedModel,
        )
        from .models.granite import (
            GraniteForCausalLM,
            GraniteModel,
            GranitePreTrainedModel,
        )
        from .models.granitemoe import (
            GraniteMoeForCausalLM,
            GraniteMoeModel,
            GraniteMoePreTrainedModel,
        )
        from .models.grounding_dino import (
            GroundingDinoForObjectDetection,
            GroundingDinoModel,
            GroundingDinoPreTrainedModel,
        )
        from .models.groupvit import (
            GroupViTModel,
            GroupViTPreTrainedModel,
            GroupViTTextModel,
            GroupViTVisionModel,
        )
        from .models.hiera import (
            HieraBackbone,
            HieraForImageClassification,
            HieraForPreTraining,
            HieraModel,
            HieraPreTrainedModel,
        )
        from .models.hubert import (
            HubertForCTC,
            HubertForSequenceClassification,
            HubertModel,
            HubertPreTrainedModel,
        )
        from .models.ibert import (
            IBertForMaskedLM,
            IBertForMultipleChoice,
            IBertForQuestionAnswering,
            IBertForSequenceClassification,
            IBertForTokenClassification,
            IBertModel,
            IBertPreTrainedModel,
        )
        from .models.idefics import (
            IdeficsForVisionText2Text,
            IdeficsModel,
            IdeficsPreTrainedModel,
            IdeficsProcessor,
        )
        from .models.idefics2 import (
            Idefics2ForConditionalGeneration,
            Idefics2Model,
            Idefics2PreTrainedModel,
            Idefics2Processor,
        )
        from .models.idefics3 import (
            Idefics3ForConditionalGeneration,
            Idefics3Model,
            Idefics3PreTrainedModel,
            Idefics3Processor,
            Idefics3VisionConfig,
            Idefics3VisionTransformer,
        )
        from .models.ijepa import (
            IJepaForImageClassification,
            IJepaModel,
            IJepaPreTrainedModel,
        )
        from .models.imagegpt import (
            ImageGPTForCausalImageModeling,
            ImageGPTForImageClassification,
            ImageGPTModel,
            ImageGPTPreTrainedModel,
            load_tf_weights_in_imagegpt,
        )
        from .models.informer import (
            InformerForPrediction,
            InformerModel,
            InformerPreTrainedModel,
        )
        from .models.instructblip import (
            InstructBlipForConditionalGeneration,
            InstructBlipPreTrainedModel,
            InstructBlipQFormerModel,
            InstructBlipVisionModel,
        )
        from .models.instructblipvideo import (
            InstructBlipVideoForConditionalGeneration,
            InstructBlipVideoPreTrainedModel,
            InstructBlipVideoQFormerModel,
            InstructBlipVideoVisionModel,
        )
        from .models.jamba import (
            JambaForCausalLM,
            JambaForSequenceClassification,
            JambaModel,
            JambaPreTrainedModel,
        )
        from .models.jetmoe import (
            JetMoeForCausalLM,
            JetMoeForSequenceClassification,
            JetMoeModel,
            JetMoePreTrainedModel,
        )
        from .models.kosmos2 import (
            Kosmos2ForConditionalGeneration,
            Kosmos2Model,
            Kosmos2PreTrainedModel,
        )
        from .models.layoutlm import (
            LayoutLMForMaskedLM,
            LayoutLMForQuestionAnswering,
            LayoutLMForSequenceClassification,
            LayoutLMForTokenClassification,
            LayoutLMModel,
            LayoutLMPreTrainedModel,
        )
        from .models.layoutlmv2 import (
            LayoutLMv2ForQuestionAnswering,
            LayoutLMv2ForSequenceClassification,
            LayoutLMv2ForTokenClassification,
            LayoutLMv2Model,
            LayoutLMv2PreTrainedModel,
        )
        from .models.layoutlmv3 import (
            LayoutLMv3ForQuestionAnswering,
            LayoutLMv3ForSequenceClassification,
            LayoutLMv3ForTokenClassification,
            LayoutLMv3Model,
            LayoutLMv3PreTrainedModel,
        )
        from .models.led import (
            LEDForConditionalGeneration,
            LEDForQuestionAnswering,
            LEDForSequenceClassification,
            LEDModel,
            LEDPreTrainedModel,
        )
        from .models.levit import (
            LevitForImageClassification,
            LevitForImageClassificationWithTeacher,
            LevitModel,
            LevitPreTrainedModel,
        )
        from .models.lilt import (
            LiltForQuestionAnswering,
            LiltForSequenceClassification,
            LiltForTokenClassification,
            LiltModel,
            LiltPreTrainedModel,
        )
        from .models.llama import (
            LlamaForCausalLM,
            LlamaForQuestionAnswering,
            LlamaForSequenceClassification,
            LlamaForTokenClassification,
            LlamaModel,
            LlamaPreTrainedModel,
        )
        from .models.llava import (
            LlavaForConditionalGeneration,
            LlavaPreTrainedModel,
        )
        from .models.llava_next import (
            LlavaNextForConditionalGeneration,
            LlavaNextPreTrainedModel,
        )
        from .models.llava_next_video import (
            LlavaNextVideoForConditionalGeneration,
            LlavaNextVideoPreTrainedModel,
        )
        from .models.llava_onevision import (
            LlavaOnevisionForConditionalGeneration,
            LlavaOnevisionPreTrainedModel,
        )
        from .models.longformer import (
            LongformerForMaskedLM,
            LongformerForMultipleChoice,
            LongformerForQuestionAnswering,
            LongformerForSequenceClassification,
            LongformerForTokenClassification,
            LongformerModel,
            LongformerPreTrainedModel,
        )
        from .models.longt5 import (
            LongT5EncoderModel,
            LongT5ForConditionalGeneration,
            LongT5Model,
            LongT5PreTrainedModel,
        )
        from .models.luke import (
            LukeForEntityClassification,
            LukeForEntityPairClassification,
            LukeForEntitySpanClassification,
            LukeForMaskedLM,
            LukeForMultipleChoice,
            LukeForQuestionAnswering,
            LukeForSequenceClassification,
            LukeForTokenClassification,
            LukeModel,
            LukePreTrainedModel,
        )
        from .models.lxmert import (
            LxmertEncoder,
            LxmertForPreTraining,
            LxmertForQuestionAnswering,
            LxmertModel,
            LxmertPreTrainedModel,
            LxmertVisualFeatureEncoder,
        )
        from .models.m2m_100 import (
            M2M100ForConditionalGeneration,
            M2M100Model,
            M2M100PreTrainedModel,
        )
        from .models.mamba import (
            MambaForCausalLM,
            MambaModel,
            MambaPreTrainedModel,
        )
        from .models.mamba2 import (
            Mamba2ForCausalLM,
            Mamba2Model,
            Mamba2PreTrainedModel,
        )
        from .models.marian import MarianForCausalLM, MarianModel, MarianMTModel, MarianPreTrainedModel
        from .models.markuplm import (
            MarkupLMForQuestionAnswering,
            MarkupLMForSequenceClassification,
            MarkupLMForTokenClassification,
            MarkupLMModel,
            MarkupLMPreTrainedModel,
        )
        from .models.mask2former import (
            Mask2FormerForUniversalSegmentation,
            Mask2FormerModel,
            Mask2FormerPreTrainedModel,
        )
        from .models.maskformer import (
            MaskFormerForInstanceSegmentation,
            MaskFormerModel,
            MaskFormerPreTrainedModel,
            MaskFormerSwinBackbone,
        )
        from .models.mbart import (
            MBartForCausalLM,
            MBartForConditionalGeneration,
            MBartForQuestionAnswering,
            MBartForSequenceClassification,
            MBartModel,
            MBartPreTrainedModel,
        )
        from .models.megatron_bert import (
            MegatronBertForCausalLM,
            MegatronBertForMaskedLM,
            MegatronBertForMultipleChoice,
            MegatronBertForNextSentencePrediction,
            MegatronBertForPreTraining,
            MegatronBertForQuestionAnswering,
            MegatronBertForSequenceClassification,
            MegatronBertForTokenClassification,
            MegatronBertModel,
            MegatronBertPreTrainedModel,
        )
        from .models.mgp_str import (
            MgpstrForSceneTextRecognition,
            MgpstrModel,
            MgpstrPreTrainedModel,
        )
        from .models.mimi import (
            MimiModel,
            MimiPreTrainedModel,
        )
        from .models.mistral import (
            MistralForCausalLM,
            MistralForQuestionAnswering,
            MistralForSequenceClassification,
            MistralForTokenClassification,
            MistralModel,
            MistralPreTrainedModel,
        )
        from .models.mixtral import (
            MixtralForCausalLM,
            MixtralForQuestionAnswering,
            MixtralForSequenceClassification,
            MixtralForTokenClassification,
            MixtralModel,
            MixtralPreTrainedModel,
        )
        from .models.mllama import (
            MllamaForCausalLM,
            MllamaForConditionalGeneration,
            MllamaPreTrainedModel,
            MllamaProcessor,
            MllamaTextModel,
            MllamaVisionModel,
        )
        from .models.mobilebert import (
            MobileBertForMaskedLM,
            MobileBertForMultipleChoice,
            MobileBertForNextSentencePrediction,
            MobileBertForPreTraining,
            MobileBertForQuestionAnswering,
            MobileBertForSequenceClassification,
            MobileBertForTokenClassification,
            MobileBertModel,
            MobileBertPreTrainedModel,
            load_tf_weights_in_mobilebert,
        )
        from .models.mobilenet_v1 import (
            MobileNetV1ForImageClassification,
            MobileNetV1Model,
            MobileNetV1PreTrainedModel,
            load_tf_weights_in_mobilenet_v1,
        )
        from .models.mobilenet_v2 import (
            MobileNetV2ForImageClassification,
            MobileNetV2ForSemanticSegmentation,
            MobileNetV2Model,
            MobileNetV2PreTrainedModel,
            load_tf_weights_in_mobilenet_v2,
        )
        from .models.mobilevit import (
            MobileViTForImageClassification,
            MobileViTForSemanticSegmentation,
            MobileViTModel,
            MobileViTPreTrainedModel,
        )
        from .models.mobilevitv2 import (
            MobileViTV2ForImageClassification,
            MobileViTV2ForSemanticSegmentation,
            MobileViTV2Model,
            MobileViTV2PreTrainedModel,
        )
        from .models.modernbert import (
            ModernBertForMaskedLM,
            ModernBertForSequenceClassification,
            ModernBertForTokenClassification,
            ModernBertModel,
            ModernBertPreTrainedModel,
        )
        from .models.moshi import (
            MoshiForCausalLM,
            MoshiForConditionalGeneration,
            MoshiModel,
            MoshiPreTrainedModel,
        )
        from .models.mpnet import (
            MPNetForMaskedLM,
            MPNetForMultipleChoice,
            MPNetForQuestionAnswering,
            MPNetForSequenceClassification,
            MPNetForTokenClassification,
            MPNetModel,
            MPNetPreTrainedModel,
        )
        from .models.mpt import (
            MptForCausalLM,
            MptForQuestionAnswering,
            MptForSequenceClassification,
            MptForTokenClassification,
            MptModel,
            MptPreTrainedModel,
        )
        from .models.mra import (
            MraForMaskedLM,
            MraForMultipleChoice,
            MraForQuestionAnswering,
            MraForSequenceClassification,
            MraForTokenClassification,
            MraModel,
            MraPreTrainedModel,
        )
        from .models.mt5 import (
            MT5EncoderModel,
            MT5ForConditionalGeneration,
            MT5ForQuestionAnswering,
            MT5ForSequenceClassification,
            MT5ForTokenClassification,
            MT5Model,
            MT5PreTrainedModel,
        )
        from .models.musicgen import (
            MusicgenForCausalLM,
            MusicgenForConditionalGeneration,
            MusicgenModel,
            MusicgenPreTrainedModel,
            MusicgenProcessor,
        )
        from .models.musicgen_melody import (
            MusicgenMelodyForCausalLM,
            MusicgenMelodyForConditionalGeneration,
            MusicgenMelodyModel,
            MusicgenMelodyPreTrainedModel,
        )
        from .models.mvp import (
            MvpForCausalLM,
            MvpForConditionalGeneration,
            MvpForQuestionAnswering,
            MvpForSequenceClassification,
            MvpModel,
            MvpPreTrainedModel,
        )
        from .models.nemotron import (
            NemotronForCausalLM,
            NemotronForQuestionAnswering,
            NemotronForSequenceClassification,
            NemotronForTokenClassification,
            NemotronModel,
            NemotronPreTrainedModel,
        )
        from .models.nllb_moe import (
            NllbMoeForConditionalGeneration,
            NllbMoeModel,
            NllbMoePreTrainedModel,
            NllbMoeSparseMLP,
            NllbMoeTop2Router,
        )
        from .models.nystromformer import (
            NystromformerForMaskedLM,
            NystromformerForMultipleChoice,
            NystromformerForQuestionAnswering,
            NystromformerForSequenceClassification,
            NystromformerForTokenClassification,
            NystromformerModel,
            NystromformerPreTrainedModel,
        )
        from .models.olmo import (
            OlmoForCausalLM,
            OlmoModel,
            OlmoPreTrainedModel,
        )
        from .models.olmo2 import (
            Olmo2ForCausalLM,
            Olmo2Model,
            Olmo2PreTrainedModel,
        )
        from .models.olmoe import (
            OlmoeForCausalLM,
            OlmoeModel,
            OlmoePreTrainedModel,
        )
        from .models.omdet_turbo import (
            OmDetTurboForObjectDetection,
            OmDetTurboPreTrainedModel,
        )
        from .models.oneformer import (
            OneFormerForUniversalSegmentation,
            OneFormerModel,
            OneFormerPreTrainedModel,
        )
        from .models.openai import (
            OpenAIGPTDoubleHeadsModel,
            OpenAIGPTForSequenceClassification,
            OpenAIGPTLMHeadModel,
            OpenAIGPTModel,
            OpenAIGPTPreTrainedModel,
            load_tf_weights_in_openai_gpt,
        )
        from .models.opt import (
            OPTForCausalLM,
            OPTForQuestionAnswering,
            OPTForSequenceClassification,
            OPTModel,
            OPTPreTrainedModel,
        )
        from .models.owlv2 import (
            Owlv2ForObjectDetection,
            Owlv2Model,
            Owlv2PreTrainedModel,
            Owlv2TextModel,
            Owlv2VisionModel,
        )
        from .models.owlvit import (
            OwlViTForObjectDetection,
            OwlViTModel,
            OwlViTPreTrainedModel,
            OwlViTTextModel,
            OwlViTVisionModel,
        )
        from .models.paligemma import (
            PaliGemmaForConditionalGeneration,
            PaliGemmaPreTrainedModel,
            PaliGemmaProcessor,
        )
        from .models.patchtsmixer import (
            PatchTSMixerForPrediction,
            PatchTSMixerForPretraining,
            PatchTSMixerForRegression,
            PatchTSMixerForTimeSeriesClassification,
            PatchTSMixerModel,
            PatchTSMixerPreTrainedModel,
        )
        from .models.patchtst import (
            PatchTSTForClassification,
            PatchTSTForPrediction,
            PatchTSTForPretraining,
            PatchTSTForRegression,
            PatchTSTModel,
            PatchTSTPreTrainedModel,
        )
        from .models.pegasus import (
            PegasusForCausalLM,
            PegasusForConditionalGeneration,
            PegasusModel,
            PegasusPreTrainedModel,
        )
        from .models.pegasus_x import (
            PegasusXForConditionalGeneration,
            PegasusXModel,
            PegasusXPreTrainedModel,
        )
        from .models.perceiver import (
            PerceiverForImageClassificationConvProcessing,
            PerceiverForImageClassificationFourier,
            PerceiverForImageClassificationLearned,
            PerceiverForMaskedLM,
            PerceiverForMultimodalAutoencoding,
            PerceiverForOpticalFlow,
            PerceiverForSequenceClassification,
            PerceiverModel,
            PerceiverPreTrainedModel,
        )
        from .models.persimmon import (
            PersimmonForCausalLM,
            PersimmonForSequenceClassification,
            PersimmonForTokenClassification,
            PersimmonModel,
            PersimmonPreTrainedModel,
        )
        from .models.phi import (
            PhiForCausalLM,
            PhiForSequenceClassification,
            PhiForTokenClassification,
            PhiModel,
            PhiPreTrainedModel,
        )
        from .models.phi3 import (
            Phi3ForCausalLM,
            Phi3ForSequenceClassification,
            Phi3ForTokenClassification,
            Phi3Model,
            Phi3PreTrainedModel,
        )
        from .models.phimoe import (
            PhimoeForCausalLM,
            PhimoeForSequenceClassification,
            PhimoeModel,
            PhimoePreTrainedModel,
        )
        from .models.pix2struct import (
            Pix2StructForConditionalGeneration,
            Pix2StructPreTrainedModel,
            Pix2StructTextModel,
            Pix2StructVisionModel,
        )
        from .models.pixtral import (
            PixtralPreTrainedModel,
            PixtralVisionModel,
        )
        from .models.plbart import (
            PLBartForCausalLM,
            PLBartForConditionalGeneration,
            PLBartForSequenceClassification,
            PLBartModel,
            PLBartPreTrainedModel,
        )
        from .models.poolformer import (
            PoolFormerForImageClassification,
            PoolFormerModel,
            PoolFormerPreTrainedModel,
        )
        from .models.pop2piano import (
            Pop2PianoForConditionalGeneration,
            Pop2PianoPreTrainedModel,
        )
        from .models.prophetnet import (
            ProphetNetDecoder,
            ProphetNetEncoder,
            ProphetNetForCausalLM,
            ProphetNetForConditionalGeneration,
            ProphetNetModel,
            ProphetNetPreTrainedModel,
        )
        from .models.pvt import (
            PvtForImageClassification,
            PvtModel,
            PvtPreTrainedModel,
        )
        from .models.pvt_v2 import (
            PvtV2Backbone,
            PvtV2ForImageClassification,
            PvtV2Model,
            PvtV2PreTrainedModel,
        )
        from .models.qwen2 import (
            Qwen2ForCausalLM,
            Qwen2ForQuestionAnswering,
            Qwen2ForSequenceClassification,
            Qwen2ForTokenClassification,
            Qwen2Model,
            Qwen2PreTrainedModel,
        )
        from .models.qwen2_audio import (
            Qwen2AudioEncoder,
            Qwen2AudioForConditionalGeneration,
            Qwen2AudioPreTrainedModel,
        )
        from .models.qwen2_moe import (
            Qwen2MoeForCausalLM,
            Qwen2MoeForQuestionAnswering,
            Qwen2MoeForSequenceClassification,
            Qwen2MoeForTokenClassification,
            Qwen2MoeModel,
            Qwen2MoePreTrainedModel,
        )
        from .models.qwen2_vl import (
            Qwen2VLForConditionalGeneration,
            Qwen2VLModel,
            Qwen2VLPreTrainedModel,
        )
        from .models.rag import (
            RagModel,
            RagPreTrainedModel,
            RagSequenceForGeneration,
            RagTokenForGeneration,
        )
        from .models.recurrent_gemma import (
            RecurrentGemmaForCausalLM,
            RecurrentGemmaModel,
            RecurrentGemmaPreTrainedModel,
        )
        from .models.reformer import (
            ReformerForMaskedLM,
            ReformerForQuestionAnswering,
            ReformerForSequenceClassification,
            ReformerModel,
            ReformerModelWithLMHead,
            ReformerPreTrainedModel,
        )
        from .models.regnet import (
            RegNetForImageClassification,
            RegNetModel,
            RegNetPreTrainedModel,
        )
        from .models.rembert import (
            RemBertForCausalLM,
            RemBertForMaskedLM,
            RemBertForMultipleChoice,
            RemBertForQuestionAnswering,
            RemBertForSequenceClassification,
            RemBertForTokenClassification,
            RemBertModel,
            RemBertPreTrainedModel,
            load_tf_weights_in_rembert,
        )
        from .models.resnet import (
            ResNetBackbone,
            ResNetForImageClassification,
            ResNetModel,
            ResNetPreTrainedModel,
        )
        from .models.roberta import (
            RobertaForCausalLM,
            RobertaForMaskedLM,
            RobertaForMultipleChoice,
            RobertaForQuestionAnswering,
            RobertaForSequenceClassification,
            RobertaForTokenClassification,
            RobertaModel,
            RobertaPreTrainedModel,
        )
        from .models.roberta_prelayernorm import (
            RobertaPreLayerNormForCausalLM,
            RobertaPreLayerNormForMaskedLM,
            RobertaPreLayerNormForMultipleChoice,
            RobertaPreLayerNormForQuestionAnswering,
            RobertaPreLayerNormForSequenceClassification,
            RobertaPreLayerNormForTokenClassification,
            RobertaPreLayerNormModel,
            RobertaPreLayerNormPreTrainedModel,
        )
        from .models.roc_bert import (
            RoCBertForCausalLM,
            RoCBertForMaskedLM,
            RoCBertForMultipleChoice,
            RoCBertForPreTraining,
            RoCBertForQuestionAnswering,
            RoCBertForSequenceClassification,
            RoCBertForTokenClassification,
            RoCBertModel,
            RoCBertPreTrainedModel,
            load_tf_weights_in_roc_bert,
        )
        from .models.roformer import (
            RoFormerForCausalLM,
            RoFormerForMaskedLM,
            RoFormerForMultipleChoice,
            RoFormerForQuestionAnswering,
            RoFormerForSequenceClassification,
            RoFormerForTokenClassification,
            RoFormerModel,
            RoFormerPreTrainedModel,
            load_tf_weights_in_roformer,
        )
        from .models.rt_detr import (
            RTDetrForObjectDetection,
            RTDetrModel,
            RTDetrPreTrainedModel,
            RTDetrResNetBackbone,
            RTDetrResNetPreTrainedModel,
        )
        from .models.rwkv import (
            RwkvForCausalLM,
            RwkvModel,
            RwkvPreTrainedModel,
        )
        from .models.sam import (
            SamModel,
            SamPreTrainedModel,
        )
        from .models.seamless_m4t import (
            SeamlessM4TCodeHifiGan,
            SeamlessM4TForSpeechToSpeech,
            SeamlessM4TForSpeechToText,
            SeamlessM4TForTextToSpeech,
            SeamlessM4TForTextToText,
            SeamlessM4THifiGan,
            SeamlessM4TModel,
            SeamlessM4TPreTrainedModel,
            SeamlessM4TTextToUnitForConditionalGeneration,
            SeamlessM4TTextToUnitModel,
        )
        from .models.seamless_m4t_v2 import (
            SeamlessM4Tv2ForSpeechToSpeech,
            SeamlessM4Tv2ForSpeechToText,
            SeamlessM4Tv2ForTextToSpeech,
            SeamlessM4Tv2ForTextToText,
            SeamlessM4Tv2Model,
            SeamlessM4Tv2PreTrainedModel,
        )
        from .models.segformer import (
            SegformerDecodeHead,
            SegformerForImageClassification,
            SegformerForSemanticSegmentation,
            SegformerModel,
            SegformerPreTrainedModel,
        )
        from .models.seggpt import (
            SegGptForImageSegmentation,
            SegGptModel,
            SegGptPreTrainedModel,
        )
        from .models.sew import (
            SEWForCTC,
            SEWForSequenceClassification,
            SEWModel,
            SEWPreTrainedModel,
        )
        from .models.sew_d import (
            SEWDForCTC,
            SEWDForSequenceClassification,
            SEWDModel,
            SEWDPreTrainedModel,
        )
        from .models.siglip import (
            SiglipForImageClassification,
            SiglipModel,
            SiglipPreTrainedModel,
            SiglipTextModel,
            SiglipVisionModel,
        )
        from .models.speech_encoder_decoder import SpeechEncoderDecoderModel
        from .models.speech_to_text import (
            Speech2TextForConditionalGeneration,
            Speech2TextModel,
            Speech2TextPreTrainedModel,
        )
        from .models.speecht5 import (
            SpeechT5ForSpeechToSpeech,
            SpeechT5ForSpeechToText,
            SpeechT5ForTextToSpeech,
            SpeechT5HifiGan,
            SpeechT5Model,
            SpeechT5PreTrainedModel,
        )
        from .models.splinter import (
            SplinterForPreTraining,
            SplinterForQuestionAnswering,
            SplinterModel,
            SplinterPreTrainedModel,
        )
        from .models.squeezebert import (
            SqueezeBertForMaskedLM,
            SqueezeBertForMultipleChoice,
            SqueezeBertForQuestionAnswering,
            SqueezeBertForSequenceClassification,
            SqueezeBertForTokenClassification,
            SqueezeBertModel,
            SqueezeBertPreTrainedModel,
        )
        from .models.stablelm import (
            StableLmForCausalLM,
            StableLmForSequenceClassification,
            StableLmForTokenClassification,
            StableLmModel,
            StableLmPreTrainedModel,
        )
        from .models.starcoder2 import (
            Starcoder2ForCausalLM,
            Starcoder2ForSequenceClassification,
            Starcoder2ForTokenClassification,
            Starcoder2Model,
            Starcoder2PreTrainedModel,
        )
        from .models.superpoint import (
            SuperPointForKeypointDetection,
            SuperPointPreTrainedModel,
        )
        from .models.swiftformer import (
            SwiftFormerForImageClassification,
            SwiftFormerModel,
            SwiftFormerPreTrainedModel,
        )
        from .models.swin import (
            SwinBackbone,
            SwinForImageClassification,
            SwinForMaskedImageModeling,
            SwinModel,
            SwinPreTrainedModel,
        )
        from .models.swin2sr import (
            Swin2SRForImageSuperResolution,
            Swin2SRModel,
            Swin2SRPreTrainedModel,
        )
        from .models.swinv2 import (
            Swinv2Backbone,
            Swinv2ForImageClassification,
            Swinv2ForMaskedImageModeling,
            Swinv2Model,
            Swinv2PreTrainedModel,
        )
        from .models.switch_transformers import (
            SwitchTransformersEncoderModel,
            SwitchTransformersForConditionalGeneration,
            SwitchTransformersModel,
            SwitchTransformersPreTrainedModel,
            SwitchTransformersSparseMLP,
            SwitchTransformersTop1Router,
        )
        from .models.t5 import (
            T5EncoderModel,
            T5ForConditionalGeneration,
            T5ForQuestionAnswering,
            T5ForSequenceClassification,
            T5ForTokenClassification,
            T5Model,
            T5PreTrainedModel,
            load_tf_weights_in_t5,
        )
        from .models.table_transformer import (
            TableTransformerForObjectDetection,
            TableTransformerModel,
            TableTransformerPreTrainedModel,
        )
        from .models.tapas import (
            TapasForMaskedLM,
            TapasForQuestionAnswering,
            TapasForSequenceClassification,
            TapasModel,
            TapasPreTrainedModel,
            load_tf_weights_in_tapas,
        )
        from .models.time_series_transformer import (
            TimeSeriesTransformerForPrediction,
            TimeSeriesTransformerModel,
            TimeSeriesTransformerPreTrainedModel,
        )
        from .models.timesformer import (
            TimesformerForVideoClassification,
            TimesformerModel,
            TimesformerPreTrainedModel,
        )
        from .models.timm_backbone import TimmBackbone
        from .models.timm_wrapper import (
            TimmWrapperForImageClassification,
            TimmWrapperModel,
            TimmWrapperPreTrainedModel,
        )
        from .models.trocr import (
            TrOCRForCausalLM,
            TrOCRPreTrainedModel,
        )
        from .models.tvp import (
            TvpForVideoGrounding,
            TvpModel,
            TvpPreTrainedModel,
        )
        from .models.udop import (
            UdopEncoderModel,
            UdopForConditionalGeneration,
            UdopModel,
            UdopPreTrainedModel,
        )
        from .models.umt5 import (
            UMT5EncoderModel,
            UMT5ForConditionalGeneration,
            UMT5ForQuestionAnswering,
            UMT5ForSequenceClassification,
            UMT5ForTokenClassification,
            UMT5Model,
            UMT5PreTrainedModel,
        )
        from .models.unispeech import (
            UniSpeechForCTC,
            UniSpeechForPreTraining,
            UniSpeechForSequenceClassification,
            UniSpeechModel,
            UniSpeechPreTrainedModel,
        )
        from .models.unispeech_sat import (
            UniSpeechSatForAudioFrameClassification,
            UniSpeechSatForCTC,
            UniSpeechSatForPreTraining,
            UniSpeechSatForSequenceClassification,
            UniSpeechSatForXVector,
            UniSpeechSatModel,
            UniSpeechSatPreTrainedModel,
        )
        from .models.univnet import UnivNetModel
        from .models.upernet import (
            UperNetForSemanticSegmentation,
            UperNetPreTrainedModel,
        )
        from .models.video_llava import (
            VideoLlavaForConditionalGeneration,
            VideoLlavaPreTrainedModel,
            VideoLlavaProcessor,
        )
        from .models.videomae import (
            VideoMAEForPreTraining,
            VideoMAEForVideoClassification,
            VideoMAEModel,
            VideoMAEPreTrainedModel,
        )
        from .models.vilt import (
            ViltForImageAndTextRetrieval,
            ViltForImagesAndTextClassification,
            ViltForMaskedLM,
            ViltForQuestionAnswering,
            ViltForTokenClassification,
            ViltModel,
            ViltPreTrainedModel,
        )
        from .models.vipllava import (
            VipLlavaForConditionalGeneration,
            VipLlavaPreTrainedModel,
        )
        from .models.vision_encoder_decoder import VisionEncoderDecoderModel
        from .models.vision_text_dual_encoder import VisionTextDualEncoderModel
        from .models.visual_bert import (
            VisualBertForMultipleChoice,
            VisualBertForPreTraining,
            VisualBertForQuestionAnswering,
            VisualBertForRegionToPhraseAlignment,
            VisualBertForVisualReasoning,
            VisualBertModel,
            VisualBertPreTrainedModel,
        )
        from .models.vit import (
            ViTForImageClassification,
            ViTForMaskedImageModeling,
            ViTModel,
            ViTPreTrainedModel,
        )
        from .models.vit_mae import (
            ViTMAEForPreTraining,
            ViTMAEModel,
            ViTMAEPreTrainedModel,
        )
        from .models.vit_msn import (
            ViTMSNForImageClassification,
            ViTMSNModel,
            ViTMSNPreTrainedModel,
        )
        from .models.vitdet import (
            VitDetBackbone,
            VitDetModel,
            VitDetPreTrainedModel,
        )
        from .models.vitmatte import (
            VitMatteForImageMatting,
            VitMattePreTrainedModel,
        )
        from .models.vits import (
            VitsModel,
            VitsPreTrainedModel,
        )
        from .models.vivit import (
            VivitForVideoClassification,
            VivitModel,
            VivitPreTrainedModel,
        )
        from .models.wav2vec2 import (
            Wav2Vec2ForAudioFrameClassification,
            Wav2Vec2ForCTC,
            Wav2Vec2ForMaskedLM,
            Wav2Vec2ForPreTraining,
            Wav2Vec2ForSequenceClassification,
            Wav2Vec2ForXVector,
            Wav2Vec2Model,
            Wav2Vec2PreTrainedModel,
        )
        from .models.wav2vec2_bert import (
            Wav2Vec2BertForAudioFrameClassification,
            Wav2Vec2BertForCTC,
            Wav2Vec2BertForSequenceClassification,
            Wav2Vec2BertForXVector,
            Wav2Vec2BertModel,
            Wav2Vec2BertPreTrainedModel,
        )
        from .models.wav2vec2_conformer import (
            Wav2Vec2ConformerForAudioFrameClassification,
            Wav2Vec2ConformerForCTC,
            Wav2Vec2ConformerForPreTraining,
            Wav2Vec2ConformerForSequenceClassification,
            Wav2Vec2ConformerForXVector,
            Wav2Vec2ConformerModel,
            Wav2Vec2ConformerPreTrainedModel,
        )
        from .models.wavlm import (
            WavLMForAudioFrameClassification,
            WavLMForCTC,
            WavLMForSequenceClassification,
            WavLMForXVector,
            WavLMModel,
            WavLMPreTrainedModel,
        )
        from .models.whisper import (
            WhisperForAudioClassification,
            WhisperForCausalLM,
            WhisperForConditionalGeneration,
            WhisperModel,
            WhisperPreTrainedModel,
        )
        from .models.x_clip import (
            XCLIPModel,
            XCLIPPreTrainedModel,
            XCLIPTextModel,
            XCLIPVisionModel,
        )
        from .models.xglm import (
            XGLMForCausalLM,
            XGLMModel,
            XGLMPreTrainedModel,
        )
        from .models.xlm import (
            XLMForMultipleChoice,
            XLMForQuestionAnswering,
            XLMForQuestionAnsweringSimple,
            XLMForSequenceClassification,
            XLMForTokenClassification,
            XLMModel,
            XLMPreTrainedModel,
            XLMWithLMHeadModel,
        )
        from .models.xlm_roberta import (
            XLMRobertaForCausalLM,
            XLMRobertaForMaskedLM,
            XLMRobertaForMultipleChoice,
            XLMRobertaForQuestionAnswering,
            XLMRobertaForSequenceClassification,
            XLMRobertaForTokenClassification,
            XLMRobertaModel,
            XLMRobertaPreTrainedModel,
        )
        from .models.xlm_roberta_xl import (
            XLMRobertaXLForCausalLM,
            XLMRobertaXLForMaskedLM,
            XLMRobertaXLForMultipleChoice,
            XLMRobertaXLForQuestionAnswering,
            XLMRobertaXLForSequenceClassification,
            XLMRobertaXLForTokenClassification,
            XLMRobertaXLModel,
            XLMRobertaXLPreTrainedModel,
        )
        from .models.xlnet import (
            XLNetForMultipleChoice,
            XLNetForQuestionAnswering,
            XLNetForQuestionAnsweringSimple,
            XLNetForSequenceClassification,
            XLNetForTokenClassification,
            XLNetLMHeadModel,
            XLNetModel,
            XLNetPreTrainedModel,
            load_tf_weights_in_xlnet,
        )
        from .models.xmod import (
            XmodForCausalLM,
            XmodForMaskedLM,
            XmodForMultipleChoice,
            XmodForQuestionAnswering,
            XmodForSequenceClassification,
            XmodForTokenClassification,
            XmodModel,
            XmodPreTrainedModel,
        )
        from .models.yolos import (
            YolosForObjectDetection,
            YolosModel,
            YolosPreTrainedModel,
        )
        from .models.yoso import (
            YosoForMaskedLM,
            YosoForMultipleChoice,
            YosoForQuestionAnswering,
            YosoForSequenceClassification,
            YosoForTokenClassification,
            YosoModel,
            YosoPreTrainedModel,
        )
        from .models.zamba import (
            ZambaForCausalLM,
            ZambaForSequenceClassification,
            ZambaModel,
            ZambaPreTrainedModel,
        )
        from .models.zoedepth import (
            ZoeDepthForDepthEstimation,
            ZoeDepthPreTrainedModel,
        )

        # Optimization
        from .optimization import (
            Adafactor,
            AdamW,
            get_constant_schedule,
            get_constant_schedule_with_warmup,
            get_cosine_schedule_with_warmup,
            get_cosine_with_hard_restarts_schedule_with_warmup,
            get_inverse_sqrt_schedule,
            get_linear_schedule_with_warmup,
            get_polynomial_decay_schedule_with_warmup,
            get_scheduler,
            get_wsd_schedule,
        )
        from .pytorch_utils import Conv1D, apply_chunking_to_forward, prune_layer

        # Trainer
        from .trainer import Trainer
        from .trainer_pt_utils import torch_distributed_zero_first
        from .trainer_seq2seq import Seq2SeqTrainer

    # TensorFlow
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # Import the same objects as dummies to get them in the namespace.
        # They will raise an import error if the user tries to instantiate / use them.
        from .utils.dummy_tf_objects import *
    else:
        from .benchmark.benchmark_args_tf import TensorFlowBenchmarkArguments

        # Benchmarks
        from .benchmark.benchmark_tf import TensorFlowBenchmark
        from .generation import (
            TFForcedBOSTokenLogitsProcessor,
            TFForcedEOSTokenLogitsProcessor,
            TFForceTokensLogitsProcessor,
            TFGenerationMixin,
            TFLogitsProcessor,
            TFLogitsProcessorList,
            TFLogitsWarper,
            TFMinLengthLogitsProcessor,
            TFNoBadWordsLogitsProcessor,
            TFNoRepeatNGramLogitsProcessor,
            TFRepetitionPenaltyLogitsProcessor,
            TFSuppressTokensAtBeginLogitsProcessor,
            TFSuppressTokensLogitsProcessor,
            TFTemperatureLogitsWarper,
            TFTopKLogitsWarper,
            TFTopPLogitsWarper,
        )
        from .keras_callbacks import KerasMetricCallback, PushToHubCallback
        from .modeling_tf_utils import (
            TFPreTrainedModel,
            TFSequenceSummary,
            TFSharedEmbeddings,
            shape_list,
        )

        # TensorFlow model imports
        from .models.albert import (
            TFAlbertForMaskedLM,
            TFAlbertForMultipleChoice,
            TFAlbertForPreTraining,
            TFAlbertForQuestionAnswering,
            TFAlbertForSequenceClassification,
            TFAlbertForTokenClassification,
            TFAlbertMainLayer,
            TFAlbertModel,
            TFAlbertPreTrainedModel,
        )
        from .models.auto import (
            TF_MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING,
            TF_MODEL_FOR_CAUSAL_LM_MAPPING,
            TF_MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING,
            TF_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
            TF_MODEL_FOR_MASK_GENERATION_MAPPING,
            TF_MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING,
            TF_MODEL_FOR_MASKED_LM_MAPPING,
            TF_MODEL_FOR_MULTIPLE_CHOICE_MAPPING,
            TF_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING,
            TF_MODEL_FOR_PRETRAINING_MAPPING,
            TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING,
            TF_MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING,
            TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
            TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
            TF_MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING,
            TF_MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING,
            TF_MODEL_FOR_TEXT_ENCODING_MAPPING,
            TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
            TF_MODEL_FOR_VISION_2_SEQ_MAPPING,
            TF_MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING,
            TF_MODEL_MAPPING,
            TF_MODEL_WITH_LM_HEAD_MAPPING,
            TFAutoModel,
            TFAutoModelForAudioClassification,
            TFAutoModelForCausalLM,
            TFAutoModelForDocumentQuestionAnswering,
            TFAutoModelForImageClassification,
            TFAutoModelForMaskedImageModeling,
            TFAutoModelForMaskedLM,
            TFAutoModelForMaskGeneration,
            TFAutoModelForMultipleChoice,
            TFAutoModelForNextSentencePrediction,
            TFAutoModelForPreTraining,
            TFAutoModelForQuestionAnswering,
            TFAutoModelForSemanticSegmentation,
            TFAutoModelForSeq2SeqLM,
            TFAutoModelForSequenceClassification,
            TFAutoModelForSpeechSeq2Seq,
            TFAutoModelForTableQuestionAnswering,
            TFAutoModelForTextEncoding,
            TFAutoModelForTokenClassification,
            TFAutoModelForVision2Seq,
            TFAutoModelForZeroShotImageClassification,
            TFAutoModelWithLMHead,
        )
        from .models.bart import (
            TFBartForConditionalGeneration,
            TFBartForSequenceClassification,
            TFBartModel,
            TFBartPretrainedModel,
        )
        from .models.bert import (
            TFBertForMaskedLM,
            TFBertForMultipleChoice,
            TFBertForNextSentencePrediction,
            TFBertForPreTraining,
            TFBertForQuestionAnswering,
            TFBertForSequenceClassification,
            TFBertForTokenClassification,
            TFBertLMHeadModel,
            TFBertMainLayer,
            TFBertModel,
            TFBertPreTrainedModel,
        )
        from .models.blenderbot import (
            TFBlenderbotForConditionalGeneration,
            TFBlenderbotModel,
            TFBlenderbotPreTrainedModel,
        )
        from .models.blenderbot_small import (
            TFBlenderbotSmallForConditionalGeneration,
            TFBlenderbotSmallModel,
            TFBlenderbotSmallPreTrainedModel,
        )
        from .models.blip import (
            TFBlipForConditionalGeneration,
            TFBlipForImageTextRetrieval,
            TFBlipForQuestionAnswering,
            TFBlipModel,
            TFBlipPreTrainedModel,
            TFBlipTextModel,
            TFBlipVisionModel,
        )
        from .models.camembert import (
            TFCamembertForCausalLM,
            TFCamembertForMaskedLM,
            TFCamembertForMultipleChoice,
            TFCamembertForQuestionAnswering,
            TFCamembertForSequenceClassification,
            TFCamembertForTokenClassification,
            TFCamembertModel,
            TFCamembertPreTrainedModel,
        )
        from .models.clip import (
            TFCLIPModel,
            TFCLIPPreTrainedModel,
            TFCLIPTextModel,
            TFCLIPVisionModel,
        )
        from .models.convbert import (
            TFConvBertForMaskedLM,
            TFConvBertForMultipleChoice,
            TFConvBertForQuestionAnswering,
            TFConvBertForSequenceClassification,
            TFConvBertForTokenClassification,
            TFConvBertModel,
            TFConvBertPreTrainedModel,
        )
        from .models.convnext import (
            TFConvNextForImageClassification,
            TFConvNextModel,
            TFConvNextPreTrainedModel,
        )
        from .models.convnextv2 import (
            TFConvNextV2ForImageClassification,
            TFConvNextV2Model,
            TFConvNextV2PreTrainedModel,
        )
        from .models.ctrl import (
            TFCTRLForSequenceClassification,
            TFCTRLLMHeadModel,
            TFCTRLModel,
            TFCTRLPreTrainedModel,
        )
        from .models.cvt import (
            TFCvtForImageClassification,
            TFCvtModel,
            TFCvtPreTrainedModel,
        )
        from .models.data2vec import (
            TFData2VecVisionForImageClassification,
            TFData2VecVisionForSemanticSegmentation,
            TFData2VecVisionModel,
            TFData2VecVisionPreTrainedModel,
        )
        from .models.deberta import (
            TFDebertaForMaskedLM,
            TFDebertaForQuestionAnswering,
            TFDebertaForSequenceClassification,
            TFDebertaForTokenClassification,
            TFDebertaModel,
            TFDebertaPreTrainedModel,
        )
        from .models.deberta_v2 import (
            TFDebertaV2ForMaskedLM,
            TFDebertaV2ForMultipleChoice,
            TFDebertaV2ForQuestionAnswering,
            TFDebertaV2ForSequenceClassification,
            TFDebertaV2ForTokenClassification,
            TFDebertaV2Model,
            TFDebertaV2PreTrainedModel,
        )
        from .models.deit import (
            TFDeiTForImageClassification,
            TFDeiTForImageClassificationWithTeacher,
            TFDeiTForMaskedImageModeling,
            TFDeiTModel,
            TFDeiTPreTrainedModel,
        )
        from .models.deprecated.efficientformer import (
            TFEfficientFormerForImageClassification,
            TFEfficientFormerForImageClassificationWithTeacher,
            TFEfficientFormerModel,
            TFEfficientFormerPreTrainedModel,
        )
        from .models.deprecated.transfo_xl import (
            TFAdaptiveEmbedding,
            TFTransfoXLForSequenceClassification,
            TFTransfoXLLMHeadModel,
            TFTransfoXLMainLayer,
            TFTransfoXLModel,
            TFTransfoXLPreTrainedModel,
        )
        from .models.distilbert import (
            TFDistilBertForMaskedLM,
            TFDistilBertForMultipleChoice,
            TFDistilBertForQuestionAnswering,
            TFDistilBertForSequenceClassification,
            TFDistilBertForTokenClassification,
            TFDistilBertMainLayer,
            TFDistilBertModel,
            TFDistilBertPreTrainedModel,
        )
        from .models.dpr import (
            TFDPRContextEncoder,
            TFDPRPretrainedContextEncoder,
            TFDPRPretrainedQuestionEncoder,
            TFDPRPretrainedReader,
            TFDPRQuestionEncoder,
            TFDPRReader,
        )
        from .models.electra import (
            TFElectraForMaskedLM,
            TFElectraForMultipleChoice,
            TFElectraForPreTraining,
            TFElectraForQuestionAnswering,
            TFElectraForSequenceClassification,
            TFElectraForTokenClassification,
            TFElectraModel,
            TFElectraPreTrainedModel,
        )
        from .models.encoder_decoder import TFEncoderDecoderModel
        from .models.esm import (
            TFEsmForMaskedLM,
            TFEsmForSequenceClassification,
            TFEsmForTokenClassification,
            TFEsmModel,
            TFEsmPreTrainedModel,
        )
        from .models.flaubert import (
            TFFlaubertForMultipleChoice,
            TFFlaubertForQuestionAnsweringSimple,
            TFFlaubertForSequenceClassification,
            TFFlaubertForTokenClassification,
            TFFlaubertModel,
            TFFlaubertPreTrainedModel,
            TFFlaubertWithLMHeadModel,
        )
        from .models.funnel import (
            TFFunnelBaseModel,
            TFFunnelForMaskedLM,
            TFFunnelForMultipleChoice,
            TFFunnelForPreTraining,
            TFFunnelForQuestionAnswering,
            TFFunnelForSequenceClassification,
            TFFunnelForTokenClassification,
            TFFunnelModel,
            TFFunnelPreTrainedModel,
        )
        from .models.gpt2 import (
            TFGPT2DoubleHeadsModel,
            TFGPT2ForSequenceClassification,
            TFGPT2LMHeadModel,
            TFGPT2MainLayer,
            TFGPT2Model,
            TFGPT2PreTrainedModel,
        )
        from .models.gptj import (
            TFGPTJForCausalLM,
            TFGPTJForQuestionAnswering,
            TFGPTJForSequenceClassification,
            TFGPTJModel,
            TFGPTJPreTrainedModel,
        )
        from .models.groupvit import (
            TFGroupViTModel,
            TFGroupViTPreTrainedModel,
            TFGroupViTTextModel,
            TFGroupViTVisionModel,
        )
        from .models.hubert import (
            TFHubertForCTC,
            TFHubertModel,
            TFHubertPreTrainedModel,
        )
        from .models.idefics import (
            TFIdeficsForVisionText2Text,
            TFIdeficsModel,
            TFIdeficsPreTrainedModel,
        )
        from .models.layoutlm import (
            TFLayoutLMForMaskedLM,
            TFLayoutLMForQuestionAnswering,
            TFLayoutLMForSequenceClassification,
            TFLayoutLMForTokenClassification,
            TFLayoutLMMainLayer,
            TFLayoutLMModel,
            TFLayoutLMPreTrainedModel,
        )
        from .models.layoutlmv3 import (
            TFLayoutLMv3ForQuestionAnswering,
            TFLayoutLMv3ForSequenceClassification,
            TFLayoutLMv3ForTokenClassification,
            TFLayoutLMv3Model,
            TFLayoutLMv3PreTrainedModel,
        )
        from .models.led import (
            TFLEDForConditionalGeneration,
            TFLEDModel,
            TFLEDPreTrainedModel,
        )
        from .models.longformer import (
            TFLongformerForMaskedLM,
            TFLongformerForMultipleChoice,
            TFLongformerForQuestionAnswering,
            TFLongformerForSequenceClassification,
            TFLongformerForTokenClassification,
            TFLongformerModel,
            TFLongformerPreTrainedModel,
        )
        from .models.lxmert import (
            TFLxmertForPreTraining,
            TFLxmertMainLayer,
            TFLxmertModel,
            TFLxmertPreTrainedModel,
            TFLxmertVisualFeatureEncoder,
        )
        from .models.marian import (
            TFMarianModel,
            TFMarianMTModel,
            TFMarianPreTrainedModel,
        )
        from .models.mbart import (
            TFMBartForConditionalGeneration,
            TFMBartModel,
            TFMBartPreTrainedModel,
        )
        from .models.mistral import (
            TFMistralForCausalLM,
            TFMistralForSequenceClassification,
            TFMistralModel,
            TFMistralPreTrainedModel,
        )
        from .models.mobilebert import (
            TFMobileBertForMaskedLM,
            TFMobileBertForMultipleChoice,
            TFMobileBertForNextSentencePrediction,
            TFMobileBertForPreTraining,
            TFMobileBertForQuestionAnswering,
            TFMobileBertForSequenceClassification,
            TFMobileBertForTokenClassification,
            TFMobileBertMainLayer,
            TFMobileBertModel,
            TFMobileBertPreTrainedModel,
        )
        from .models.mobilevit import (
            TFMobileViTForImageClassification,
            TFMobileViTForSemanticSegmentation,
            TFMobileViTModel,
            TFMobileViTPreTrainedModel,
        )
        from .models.mpnet import (
            TFMPNetForMaskedLM,
            TFMPNetForMultipleChoice,
            TFMPNetForQuestionAnswering,
            TFMPNetForSequenceClassification,
            TFMPNetForTokenClassification,
            TFMPNetMainLayer,
            TFMPNetModel,
            TFMPNetPreTrainedModel,
        )
        from .models.mt5 import (
            TFMT5EncoderModel,
            TFMT5ForConditionalGeneration,
            TFMT5Model,
        )
        from .models.openai import (
            TFOpenAIGPTDoubleHeadsModel,
            TFOpenAIGPTForSequenceClassification,
            TFOpenAIGPTLMHeadModel,
            TFOpenAIGPTMainLayer,
            TFOpenAIGPTModel,
            TFOpenAIGPTPreTrainedModel,
        )
        from .models.opt import TFOPTForCausalLM, TFOPTModel, TFOPTPreTrainedModel
        from .models.pegasus import (
            TFPegasusForConditionalGeneration,
            TFPegasusModel,
            TFPegasusPreTrainedModel,
        )
        from .models.rag import (
            TFRagModel,
            TFRagPreTrainedModel,
            TFRagSequenceForGeneration,
            TFRagTokenForGeneration,
        )
        from .models.regnet import (
            TFRegNetForImageClassification,
            TFRegNetModel,
            TFRegNetPreTrainedModel,
        )
        from .models.rembert import (
            TFRemBertForCausalLM,
            TFRemBertForMaskedLM,
            TFRemBertForMultipleChoice,
            TFRemBertForQuestionAnswering,
            TFRemBertForSequenceClassification,
            TFRemBertForTokenClassification,
            TFRemBertModel,
            TFRemBertPreTrainedModel,
        )
        from .models.resnet import (
            TFResNetForImageClassification,
            TFResNetModel,
            TFResNetPreTrainedModel,
        )
        from .models.roberta import (
            TFRobertaForCausalLM,
            TFRobertaForMaskedLM,
            TFRobertaForMultipleChoice,
            TFRobertaForQuestionAnswering,
            TFRobertaForSequenceClassification,
            TFRobertaForTokenClassification,
            TFRobertaMainLayer,
            TFRobertaModel,
            TFRobertaPreTrainedModel,
        )
        from .models.roberta_prelayernorm import (
            TFRobertaPreLayerNormForCausalLM,
            TFRobertaPreLayerNormForMaskedLM,
            TFRobertaPreLayerNormForMultipleChoice,
            TFRobertaPreLayerNormForQuestionAnswering,
            TFRobertaPreLayerNormForSequenceClassification,
            TFRobertaPreLayerNormForTokenClassification,
            TFRobertaPreLayerNormMainLayer,
            TFRobertaPreLayerNormModel,
            TFRobertaPreLayerNormPreTrainedModel,
        )
        from .models.roformer import (
            TFRoFormerForCausalLM,
            TFRoFormerForMaskedLM,
            TFRoFormerForMultipleChoice,
            TFRoFormerForQuestionAnswering,
            TFRoFormerForSequenceClassification,
            TFRoFormerForTokenClassification,
            TFRoFormerModel,
            TFRoFormerPreTrainedModel,
        )
        from .models.sam import (
            TFSamModel,
            TFSamPreTrainedModel,
        )
        from .models.segformer import (
            TFSegformerDecodeHead,
            TFSegformerForImageClassification,
            TFSegformerForSemanticSegmentation,
            TFSegformerModel,
            TFSegformerPreTrainedModel,
        )
        from .models.speech_to_text import (
            TFSpeech2TextForConditionalGeneration,
            TFSpeech2TextModel,
            TFSpeech2TextPreTrainedModel,
        )
        from .models.swiftformer import (
            TFSwiftFormerForImageClassification,
            TFSwiftFormerModel,
            TFSwiftFormerPreTrainedModel,
        )
        from .models.swin import (
            TFSwinForImageClassification,
            TFSwinForMaskedImageModeling,
            TFSwinModel,
            TFSwinPreTrainedModel,
        )
        from .models.t5 import (
            TFT5EncoderModel,
            TFT5ForConditionalGeneration,
            TFT5Model,
            TFT5PreTrainedModel,
        )
        from .models.tapas import (
            TFTapasForMaskedLM,
            TFTapasForQuestionAnswering,
            TFTapasForSequenceClassification,
            TFTapasModel,
            TFTapasPreTrainedModel,
        )
        from .models.vision_encoder_decoder import TFVisionEncoderDecoderModel
        from .models.vision_text_dual_encoder import TFVisionTextDualEncoderModel
        from .models.vit import (
            TFViTForImageClassification,
            TFViTModel,
            TFViTPreTrainedModel,
        )
        from .models.vit_mae import (
            TFViTMAEForPreTraining,
            TFViTMAEModel,
            TFViTMAEPreTrainedModel,
        )
        from .models.wav2vec2 import (
            TFWav2Vec2ForCTC,
            TFWav2Vec2ForSequenceClassification,
            TFWav2Vec2Model,
            TFWav2Vec2PreTrainedModel,
        )
        from .models.whisper import (
            TFWhisperForConditionalGeneration,
            TFWhisperModel,
            TFWhisperPreTrainedModel,
        )
        from .models.xglm import (
            TFXGLMForCausalLM,
            TFXGLMModel,
            TFXGLMPreTrainedModel,
        )
        from .models.xlm import (
            TFXLMForMultipleChoice,
            TFXLMForQuestionAnsweringSimple,
            TFXLMForSequenceClassification,
            TFXLMForTokenClassification,
            TFXLMMainLayer,
            TFXLMModel,
            TFXLMPreTrainedModel,
            TFXLMWithLMHeadModel,
        )
        from .models.xlm_roberta import (
            TFXLMRobertaForCausalLM,
            TFXLMRobertaForMaskedLM,
            TFXLMRobertaForMultipleChoice,
            TFXLMRobertaForQuestionAnswering,
            TFXLMRobertaForSequenceClassification,
            TFXLMRobertaForTokenClassification,
            TFXLMRobertaModel,
            TFXLMRobertaPreTrainedModel,
        )
        from .models.xlnet import (
            TFXLNetForMultipleChoice,
            TFXLNetForQuestionAnsweringSimple,
            TFXLNetForSequenceClassification,
            TFXLNetForTokenClassification,
            TFXLNetLMHeadModel,
            TFXLNetMainLayer,
            TFXLNetModel,
            TFXLNetPreTrainedModel,
        )

        # Optimization
        from .optimization_tf import (
            AdamWeightDecay,
            GradientAccumulator,
            WarmUp,
            create_optimizer,
        )

    try:
        if not (
            is_librosa_available()
            and is_essentia_available()
            and is_scipy_available()
            and is_torch_available()
            and is_pretty_midi_available()
        ):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_essentia_and_librosa_and_pretty_midi_and_scipy_and_torch_objects import *
    else:
        from .models.pop2piano import (
            Pop2PianoFeatureExtractor,
            Pop2PianoProcessor,
            Pop2PianoTokenizer,
        )

    try:
        if not is_torchaudio_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_torchaudio_objects import *
    else:
        from .models.musicgen_melody import MusicgenMelodyFeatureExtractor, MusicgenMelodyProcessor
    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # Import the same objects as dummies to get them in the namespace.
        # They will raise an import error if the user tries to instantiate / use them.
        from .utils.dummy_flax_objects import *
    else:
        from .generation import (
            FlaxForcedBOSTokenLogitsProcessor,
            FlaxForcedEOSTokenLogitsProcessor,
            FlaxForceTokensLogitsProcessor,
            FlaxGenerationMixin,
            FlaxLogitsProcessor,
            FlaxLogitsProcessorList,
            FlaxLogitsWarper,
            FlaxMinLengthLogitsProcessor,
            FlaxSuppressTokensAtBeginLogitsProcessor,
            FlaxSuppressTokensLogitsProcessor,
            FlaxTemperatureLogitsWarper,
            FlaxTopKLogitsWarper,
            FlaxTopPLogitsWarper,
            FlaxWhisperTimeStampLogitsProcessor,
        )
        from .modeling_flax_utils import FlaxPreTrainedModel

        # Flax model imports
        from .models.albert import (
            FlaxAlbertForMaskedLM,
            FlaxAlbertForMultipleChoice,
            FlaxAlbertForPreTraining,
            FlaxAlbertForQuestionAnswering,
            FlaxAlbertForSequenceClassification,
            FlaxAlbertForTokenClassification,
            FlaxAlbertModel,
            FlaxAlbertPreTrainedModel,
        )
        from .models.auto import (
            FLAX_MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING,
            FLAX_MODEL_FOR_CAUSAL_LM_MAPPING,
            FLAX_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
            FLAX_MODEL_FOR_MASKED_LM_MAPPING,
            FLAX_MODEL_FOR_MULTIPLE_CHOICE_MAPPING,
            FLAX_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING,
            FLAX_MODEL_FOR_PRETRAINING_MAPPING,
            FLAX_MODEL_FOR_QUESTION_ANSWERING_MAPPING,
            FLAX_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
            FLAX_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
            FLAX_MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING,
            FLAX_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
            FLAX_MODEL_FOR_VISION_2_SEQ_MAPPING,
            FLAX_MODEL_MAPPING,
            FlaxAutoModel,
            FlaxAutoModelForCausalLM,
            FlaxAutoModelForImageClassification,
            FlaxAutoModelForMaskedLM,
            FlaxAutoModelForMultipleChoice,
            FlaxAutoModelForNextSentencePrediction,
            FlaxAutoModelForPreTraining,
            FlaxAutoModelForQuestionAnswering,
            FlaxAutoModelForSeq2SeqLM,
            FlaxAutoModelForSequenceClassification,
            FlaxAutoModelForSpeechSeq2Seq,
            FlaxAutoModelForTokenClassification,
            FlaxAutoModelForVision2Seq,
        )
        from .models.bart import (
            FlaxBartDecoderPreTrainedModel,
            FlaxBartForCausalLM,
            FlaxBartForConditionalGeneration,
            FlaxBartForQuestionAnswering,
            FlaxBartForSequenceClassification,
            FlaxBartModel,
            FlaxBartPreTrainedModel,
        )
        from .models.beit import (
            FlaxBeitForImageClassification,
            FlaxBeitForMaskedImageModeling,
            FlaxBeitModel,
            FlaxBeitPreTrainedModel,
        )
        from .models.bert import (
            FlaxBertForCausalLM,
            FlaxBertForMaskedLM,
            FlaxBertForMultipleChoice,
            FlaxBertForNextSentencePrediction,
            FlaxBertForPreTraining,
            FlaxBertForQuestionAnswering,
            FlaxBertForSequenceClassification,
            FlaxBertForTokenClassification,
            FlaxBertModel,
            FlaxBertPreTrainedModel,
        )
        from .models.big_bird import (
            FlaxBigBirdForCausalLM,
            FlaxBigBirdForMaskedLM,
            FlaxBigBirdForMultipleChoice,
            FlaxBigBirdForPreTraining,
            FlaxBigBirdForQuestionAnswering,
            FlaxBigBirdForSequenceClassification,
            FlaxBigBirdForTokenClassification,
            FlaxBigBirdModel,
            FlaxBigBirdPreTrainedModel,
        )
        from .models.blenderbot import (
            FlaxBlenderbotForConditionalGeneration,
            FlaxBlenderbotModel,
            FlaxBlenderbotPreTrainedModel,
        )
        from .models.blenderbot_small import (
            FlaxBlenderbotSmallForConditionalGeneration,
            FlaxBlenderbotSmallModel,
            FlaxBlenderbotSmallPreTrainedModel,
        )
        from .models.bloom import (
            FlaxBloomForCausalLM,
            FlaxBloomModel,
            FlaxBloomPreTrainedModel,
        )
        from .models.clip import (
            FlaxCLIPModel,
            FlaxCLIPPreTrainedModel,
            FlaxCLIPTextModel,
            FlaxCLIPTextModelWithProjection,
            FlaxCLIPTextPreTrainedModel,
            FlaxCLIPVisionModel,
            FlaxCLIPVisionPreTrainedModel,
        )
        from .models.dinov2 import (
            FlaxDinov2ForImageClassification,
            FlaxDinov2Model,
            FlaxDinov2PreTrainedModel,
        )
        from .models.distilbert import (
            FlaxDistilBertForMaskedLM,
            FlaxDistilBertForMultipleChoice,
            FlaxDistilBertForQuestionAnswering,
            FlaxDistilBertForSequenceClassification,
            FlaxDistilBertForTokenClassification,
            FlaxDistilBertModel,
            FlaxDistilBertPreTrainedModel,
        )
        from .models.electra import (
            FlaxElectraForCausalLM,
            FlaxElectraForMaskedLM,
            FlaxElectraForMultipleChoice,
            FlaxElectraForPreTraining,
            FlaxElectraForQuestionAnswering,
            FlaxElectraForSequenceClassification,
            FlaxElectraForTokenClassification,
            FlaxElectraModel,
            FlaxElectraPreTrainedModel,
        )
        from .models.encoder_decoder import FlaxEncoderDecoderModel
        from .models.gemma import (
            FlaxGemmaForCausalLM,
            FlaxGemmaModel,
            FlaxGemmaPreTrainedModel,
        )
        from .models.gpt2 import (
            FlaxGPT2LMHeadModel,
            FlaxGPT2Model,
            FlaxGPT2PreTrainedModel,
        )
        from .models.gpt_neo import (
            FlaxGPTNeoForCausalLM,
            FlaxGPTNeoModel,
            FlaxGPTNeoPreTrainedModel,
        )
        from .models.gptj import (
            FlaxGPTJForCausalLM,
            FlaxGPTJModel,
            FlaxGPTJPreTrainedModel,
        )
        from .models.llama import (
            FlaxLlamaForCausalLM,
            FlaxLlamaModel,
            FlaxLlamaPreTrainedModel,
        )
        from .models.longt5 import (
            FlaxLongT5ForConditionalGeneration,
            FlaxLongT5Model,
            FlaxLongT5PreTrainedModel,
        )
        from .models.marian import (
            FlaxMarianModel,
            FlaxMarianMTModel,
            FlaxMarianPreTrainedModel,
        )
        from .models.mbart import (
            FlaxMBartForConditionalGeneration,
            FlaxMBartForQuestionAnswering,
            FlaxMBartForSequenceClassification,
            FlaxMBartModel,
            FlaxMBartPreTrainedModel,
        )
        from .models.mistral import (
            FlaxMistralForCausalLM,
            FlaxMistralModel,
            FlaxMistralPreTrainedModel,
        )
        from .models.mt5 import (
            FlaxMT5EncoderModel,
            FlaxMT5ForConditionalGeneration,
            FlaxMT5Model,
        )
        from .models.opt import FlaxOPTForCausalLM, FlaxOPTModel, FlaxOPTPreTrainedModel
        from .models.pegasus import (
            FlaxPegasusForConditionalGeneration,
            FlaxPegasusModel,
            FlaxPegasusPreTrainedModel,
        )
        from .models.regnet import (
            FlaxRegNetForImageClassification,
            FlaxRegNetModel,
            FlaxRegNetPreTrainedModel,
        )
        from .models.resnet import (
            FlaxResNetForImageClassification,
            FlaxResNetModel,
            FlaxResNetPreTrainedModel,
        )
        from .models.roberta import (
            FlaxRobertaForCausalLM,
            FlaxRobertaForMaskedLM,
            FlaxRobertaForMultipleChoice,
            FlaxRobertaForQuestionAnswering,
            FlaxRobertaForSequenceClassification,
            FlaxRobertaForTokenClassification,
            FlaxRobertaModel,
            FlaxRobertaPreTrainedModel,
        )
        from .models.roberta_prelayernorm import (
            FlaxRobertaPreLayerNormForCausalLM,
            FlaxRobertaPreLayerNormForMaskedLM,
            FlaxRobertaPreLayerNormForMultipleChoice,
            FlaxRobertaPreLayerNormForQuestionAnswering,
            FlaxRobertaPreLayerNormForSequenceClassification,
            FlaxRobertaPreLayerNormForTokenClassification,
            FlaxRobertaPreLayerNormModel,
            FlaxRobertaPreLayerNormPreTrainedModel,
        )
        from .models.roformer import (
            FlaxRoFormerForMaskedLM,
            FlaxRoFormerForMultipleChoice,
            FlaxRoFormerForQuestionAnswering,
            FlaxRoFormerForSequenceClassification,
            FlaxRoFormerForTokenClassification,
            FlaxRoFormerModel,
            FlaxRoFormerPreTrainedModel,
        )
        from .models.speech_encoder_decoder import FlaxSpeechEncoderDecoderModel
        from .models.t5 import (
            FlaxT5EncoderModel,
            FlaxT5ForConditionalGeneration,
            FlaxT5Model,
            FlaxT5PreTrainedModel,
        )
        from .models.vision_encoder_decoder import FlaxVisionEncoderDecoderModel
        from .models.vision_text_dual_encoder import FlaxVisionTextDualEncoderModel
        from .models.vit import (
            FlaxViTForImageClassification,
            FlaxViTModel,
            FlaxViTPreTrainedModel,
        )
        from .models.wav2vec2 import (
            FlaxWav2Vec2ForCTC,
            FlaxWav2Vec2ForPreTraining,
            FlaxWav2Vec2Model,
            FlaxWav2Vec2PreTrainedModel,
        )
        from .models.whisper import (
            FlaxWhisperForAudioClassification,
            FlaxWhisperForConditionalGeneration,
            FlaxWhisperModel,
            FlaxWhisperPreTrainedModel,
        )
        from .models.xglm import (
            FlaxXGLMForCausalLM,
            FlaxXGLMModel,
            FlaxXGLMPreTrainedModel,
        )
        from .models.xlm_roberta import (
            FlaxXLMRobertaForCausalLM,
            FlaxXLMRobertaForMaskedLM,
            FlaxXLMRobertaForMultipleChoice,
            FlaxXLMRobertaForQuestionAnswering,
            FlaxXLMRobertaForSequenceClassification,
            FlaxXLMRobertaForTokenClassification,
            FlaxXLMRobertaModel,
            FlaxXLMRobertaPreTrainedModel,
        )


else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
        extra_objects={"__version__": __version__},
    )


if not is_tf_available() and not is_torch_available() and not is_flax_available():
    logger.warning_advice(
        "None of PyTorch, TensorFlow >= 2.0, or Flax have been found. "
        "Models won't be available and only tokenizers, configuration "
        "and file/data utilities can be used."
    )
