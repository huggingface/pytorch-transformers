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
"""Auto Tokenizer class."""

import importlib
import json
import os
import warnings
from collections import OrderedDict
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union

from ...configuration_utils import PretrainedConfig
from ...dynamic_module_utils import get_class_from_dynamic_module, resolve_trust_remote_code
from ...modeling_gguf_pytorch_utils import load_gguf_checkpoint
from ...tokenization_utils import PreTrainedTokenizer
from ...tokenization_utils_base import TOKENIZER_CONFIG_FILE
from ...utils import (
    cached_file,
    extract_commit_hash,
    is_g2p_en_available,
    is_sentencepiece_available,
    is_tokenizers_available,
    logging,
)
from ..encoder_decoder import EncoderDecoderConfig
from .auto_factory import _LazyAutoMapping
from .configuration_auto import (
    CONFIG_MAPPING_NAMES,
    AutoConfig,
    config_class_to_model_type,
    model_type_to_module_name,
    replace_list_option_in_docstrings,
)


if is_tokenizers_available():
    from ...tokenization_utils_fast import PreTrainedTokenizerFast
else:
    PreTrainedTokenizerFast = None


logger = logging.get_logger(__name__)

if TYPE_CHECKING:
    # This significantly improves completion suggestion performance when
    # the transformers package is used with Microsoft's Pylance language server.
    TOKENIZER_MAPPING_NAMES: OrderedDict[str, Tuple[Optional[str], Optional[str]]] = OrderedDict()
else:
    TOKENIZER_MAPPING_NAMES = OrderedDict(
        [
            (
                "albert",
                (
                    "AlbertTokenizer" if is_sentencepiece_available() else None,
                    "AlbertTokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            ("align", ("BertTokenizer", "BertTokenizerFast" if is_tokenizers_available() else None)),
            ("aria", ("LlamaTokenizer", "LlamaTokenizerFast" if is_tokenizers_available() else None)),
            ("bark", ("BertTokenizer", "BertTokenizerFast" if is_tokenizers_available() else None)),
            ("bart", ("BartTokenizer", "BartTokenizerFast")),
            (
                "barthez",
                (
                    "BarthezTokenizer" if is_sentencepiece_available() else None,
                    "BarthezTokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            ("bartpho", ("BartphoTokenizer", None)),
            ("bert", ("BertTokenizer", "BertTokenizerFast" if is_tokenizers_available() else None)),
            ("bert-generation", ("BertGenerationTokenizer" if is_sentencepiece_available() else None, None)),
            ("bert-japanese", ("BertJapaneseTokenizer", None)),
            ("bertweet", ("BertweetTokenizer", None)),
            (
                "big_bird",
                (
                    "BigBirdTokenizer" if is_sentencepiece_available() else None,
                    "BigBirdTokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            ("bigbird_pegasus", ("PegasusTokenizer", "PegasusTokenizerFast" if is_tokenizers_available() else None)),
            ("biogpt", ("BioGptTokenizer", None)),
            ("blenderbot", ("BlenderbotTokenizer", "BlenderbotTokenizerFast")),
            ("blenderbot-small", ("BlenderbotSmallTokenizer", None)),
            ("blip", ("BertTokenizer", "BertTokenizerFast" if is_tokenizers_available() else None)),
            ("blip-2", ("GPT2Tokenizer", "GPT2TokenizerFast" if is_tokenizers_available() else None)),
            ("bloom", (None, "BloomTokenizerFast" if is_tokenizers_available() else None)),
            ("bridgetower", ("RobertaTokenizer", "RobertaTokenizerFast" if is_tokenizers_available() else None)),
            ("bros", ("BertTokenizer", "BertTokenizerFast" if is_tokenizers_available() else None)),
            ("byt5", ("ByT5Tokenizer", None)),
            (
                "camembert",
                (
                    "CamembertTokenizer" if is_sentencepiece_available() else None,
                    "CamembertTokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            ("canine", ("CanineTokenizer", None)),
            (
                "chameleon",
                (
                    "LlamaTokenizer" if is_sentencepiece_available() else None,
                    "LlamaTokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            ("chinese_clip", ("BertTokenizer", "BertTokenizerFast" if is_tokenizers_available() else None)),
            (
                "clap",
                (
                    "RobertaTokenizer",
                    "RobertaTokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            (
                "clip",
                (
                    "CLIPTokenizer",
                    "CLIPTokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            (
                "clipseg",
                (
                    "CLIPTokenizer",
                    "CLIPTokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            ("clvp", ("ClvpTokenizer", None)),
            (
                "code_llama",
                (
                    "CodeLlamaTokenizer" if is_sentencepiece_available() else None,
                    "CodeLlamaTokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            ("codegen", ("CodeGenTokenizer", "CodeGenTokenizerFast" if is_tokenizers_available() else None)),
            ("cohere", (None, "CohereTokenizerFast" if is_tokenizers_available() else None)),
            ("cohere2", (None, "CohereTokenizerFast" if is_tokenizers_available() else None)),
            ("colpali", ("LlamaTokenizer", "LlamaTokenizerFast" if is_tokenizers_available() else None)),
            ("convbert", ("ConvBertTokenizer", "ConvBertTokenizerFast" if is_tokenizers_available() else None)),
            (
                "cpm",
                (
                    "CpmTokenizer" if is_sentencepiece_available() else None,
                    "CpmTokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            ("cpmant", ("CpmAntTokenizer", None)),
            ("ctrl", ("CTRLTokenizer", None)),
            ("data2vec-audio", ("Wav2Vec2CTCTokenizer", None)),
            ("data2vec-text", ("RobertaTokenizer", "RobertaTokenizerFast" if is_tokenizers_available() else None)),
            ("dbrx", ("GPT2Tokenizer", "GPT2TokenizerFast" if is_tokenizers_available() else None)),
            ("deberta", ("DebertaTokenizer", "DebertaTokenizerFast" if is_tokenizers_available() else None)),
            (
                "deberta-v2",
                (
                    "DebertaV2Tokenizer" if is_sentencepiece_available() else None,
                    "DebertaV2TokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            (
                "diffllama",
                (
                    "LlamaTokenizer" if is_sentencepiece_available() else None,
                    "LlamaTokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            ("distilbert", ("DistilBertTokenizer", "DistilBertTokenizerFast" if is_tokenizers_available() else None)),
            (
                "dpr",
                (
                    "DPRQuestionEncoderTokenizer",
                    "DPRQuestionEncoderTokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            ("electra", ("ElectraTokenizer", "ElectraTokenizerFast" if is_tokenizers_available() else None)),
            ("emu3", ("GPT2Tokenizer", "GPT2TokenizerFast" if is_tokenizers_available() else None)),
            ("ernie", ("BertTokenizer", "BertTokenizerFast" if is_tokenizers_available() else None)),
            ("ernie_m", ("ErnieMTokenizer" if is_sentencepiece_available() else None, None)),
            ("esm", ("EsmTokenizer", None)),
            ("falcon", (None, "PreTrainedTokenizerFast" if is_tokenizers_available() else None)),
            ("falcon_mamba", (None, "GPTNeoXTokenizerFast" if is_tokenizers_available() else None)),
            (
                "fastspeech2_conformer",
                ("FastSpeech2ConformerTokenizer" if is_g2p_en_available() else None, None),
            ),
            ("flaubert", ("FlaubertTokenizer", None)),
            ("fnet", ("FNetTokenizer", "FNetTokenizerFast" if is_tokenizers_available() else None)),
            ("fsmt", ("FSMTTokenizer", None)),
            ("funnel", ("FunnelTokenizer", "FunnelTokenizerFast" if is_tokenizers_available() else None)),
            (
                "gemma",
                (
                    "GemmaTokenizer" if is_sentencepiece_available() else None,
                    "GemmaTokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            (
                "gemma2",
                (
                    "GemmaTokenizer" if is_sentencepiece_available() else None,
                    "GemmaTokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            ("git", ("BertTokenizer", "BertTokenizerFast" if is_tokenizers_available() else None)),
            ("glm", (None, "PreTrainedTokenizerFast" if is_tokenizers_available() else None)),
            ("gpt-sw3", ("GPTSw3Tokenizer" if is_sentencepiece_available() else None, None)),
            ("gpt2", ("GPT2Tokenizer", "GPT2TokenizerFast" if is_tokenizers_available() else None)),
            ("gpt_bigcode", ("GPT2Tokenizer", "GPT2TokenizerFast" if is_tokenizers_available() else None)),
            ("gpt_neo", ("GPT2Tokenizer", "GPT2TokenizerFast" if is_tokenizers_available() else None)),
            ("gpt_neox", (None, "GPTNeoXTokenizerFast" if is_tokenizers_available() else None)),
            ("gpt_neox_japanese", ("GPTNeoXJapaneseTokenizer", None)),
            ("gptj", ("GPT2Tokenizer", "GPT2TokenizerFast" if is_tokenizers_available() else None)),
            ("gptsan-japanese", ("GPTSanJapaneseTokenizer", None)),
            ("grounding-dino", ("BertTokenizer", "BertTokenizerFast" if is_tokenizers_available() else None)),
            ("groupvit", ("CLIPTokenizer", "CLIPTokenizerFast" if is_tokenizers_available() else None)),
            ("helium", (None, "PreTrainedTokenizerFast" if is_tokenizers_available() else None)),
            ("herbert", ("HerbertTokenizer", "HerbertTokenizerFast" if is_tokenizers_available() else None)),
            ("hubert", ("Wav2Vec2CTCTokenizer", None)),
            ("ibert", ("RobertaTokenizer", "RobertaTokenizerFast" if is_tokenizers_available() else None)),
            ("idefics", (None, "LlamaTokenizerFast" if is_tokenizers_available() else None)),
            ("idefics2", ("LlamaTokenizer", "LlamaTokenizerFast" if is_tokenizers_available() else None)),
            ("idefics3", ("LlamaTokenizer", "LlamaTokenizerFast" if is_tokenizers_available() else None)),
            ("instructblip", ("GPT2Tokenizer", "GPT2TokenizerFast" if is_tokenizers_available() else None)),
            ("instructblipvideo", ("GPT2Tokenizer", "GPT2TokenizerFast" if is_tokenizers_available() else None)),
            (
                "jamba",
                (
                    "LlamaTokenizer" if is_sentencepiece_available() else None,
                    "LlamaTokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            (
                "jetmoe",
                (
                    "LlamaTokenizer" if is_sentencepiece_available() else None,
                    "LlamaTokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            ("jukebox", ("JukeboxTokenizer", None)),
            (
                "kosmos-2",
                (
                    "XLMRobertaTokenizer" if is_sentencepiece_available() else None,
                    "XLMRobertaTokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            ("layoutlm", ("LayoutLMTokenizer", "LayoutLMTokenizerFast" if is_tokenizers_available() else None)),
            ("layoutlmv2", ("LayoutLMv2Tokenizer", "LayoutLMv2TokenizerFast" if is_tokenizers_available() else None)),
            ("layoutlmv3", ("LayoutLMv3Tokenizer", "LayoutLMv3TokenizerFast" if is_tokenizers_available() else None)),
            ("layoutxlm", ("LayoutXLMTokenizer", "LayoutXLMTokenizerFast" if is_tokenizers_available() else None)),
            ("led", ("LEDTokenizer", "LEDTokenizerFast" if is_tokenizers_available() else None)),
            ("lilt", ("LayoutLMv3Tokenizer", "LayoutLMv3TokenizerFast" if is_tokenizers_available() else None)),
            (
                "llama",
                (
                    "LlamaTokenizer" if is_sentencepiece_available() else None,
                    "LlamaTokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            ("llava", ("LlamaTokenizer", "LlamaTokenizerFast" if is_tokenizers_available() else None)),
            ("llava_next", ("LlamaTokenizer", "LlamaTokenizerFast" if is_tokenizers_available() else None)),
            ("llava_next_video", ("LlamaTokenizer", "LlamaTokenizerFast" if is_tokenizers_available() else None)),
            ("llava_onevision", ("LlamaTokenizer", "LlamaTokenizerFast" if is_tokenizers_available() else None)),
            ("longformer", ("LongformerTokenizer", "LongformerTokenizerFast" if is_tokenizers_available() else None)),
            (
                "longt5",
                (
                    "T5Tokenizer" if is_sentencepiece_available() else None,
                    "T5TokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            ("luke", ("LukeTokenizer", None)),
            ("lxmert", ("LxmertTokenizer", "LxmertTokenizerFast" if is_tokenizers_available() else None)),
            ("m2m_100", ("M2M100Tokenizer" if is_sentencepiece_available() else None, None)),
            ("mamba", (None, "GPTNeoXTokenizerFast" if is_tokenizers_available() else None)),
            ("mamba2", (None, "GPTNeoXTokenizerFast" if is_tokenizers_available() else None)),
            ("marian", ("MarianTokenizer" if is_sentencepiece_available() else None, None)),
            (
                "mbart",
                (
                    "MBartTokenizer" if is_sentencepiece_available() else None,
                    "MBartTokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            (
                "mbart50",
                (
                    "MBart50Tokenizer" if is_sentencepiece_available() else None,
                    "MBart50TokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            ("mega", ("RobertaTokenizer", "RobertaTokenizerFast" if is_tokenizers_available() else None)),
            ("megatron-bert", ("BertTokenizer", "BertTokenizerFast" if is_tokenizers_available() else None)),
            ("mgp-str", ("MgpstrTokenizer", None)),
            (
                "mistral",
                (
                    "LlamaTokenizer" if is_sentencepiece_available() else None,
                    "LlamaTokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            (
                "mixtral",
                (
                    "LlamaTokenizer" if is_sentencepiece_available() else None,
                    "LlamaTokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            ("mllama", ("LlamaTokenizer", "LlamaTokenizerFast" if is_tokenizers_available() else None)),
            ("mluke", ("MLukeTokenizer" if is_sentencepiece_available() else None, None)),
            ("mobilebert", ("MobileBertTokenizer", "MobileBertTokenizerFast" if is_tokenizers_available() else None)),
            ("modernbert", (None, "PreTrainedTokenizerFast" if is_tokenizers_available() else None)),
            ("moonshine", (None, "PreTrainedTokenizerFast" if is_tokenizers_available() else None)),
            ("moshi", (None, "PreTrainedTokenizerFast" if is_tokenizers_available() else None)),
            ("mpnet", ("MPNetTokenizer", "MPNetTokenizerFast" if is_tokenizers_available() else None)),
            ("mpt", (None, "GPTNeoXTokenizerFast" if is_tokenizers_available() else None)),
            ("mra", ("RobertaTokenizer", "RobertaTokenizerFast" if is_tokenizers_available() else None)),
            (
                "mt5",
                (
                    "MT5Tokenizer" if is_sentencepiece_available() else None,
                    "MT5TokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            ("musicgen", ("T5Tokenizer", "T5TokenizerFast" if is_tokenizers_available() else None)),
            ("musicgen_melody", ("T5Tokenizer", "T5TokenizerFast" if is_tokenizers_available() else None)),
            ("mvp", ("MvpTokenizer", "MvpTokenizerFast" if is_tokenizers_available() else None)),
            ("myt5", ("MyT5Tokenizer", None)),
            ("nezha", ("BertTokenizer", "BertTokenizerFast" if is_tokenizers_available() else None)),
            (
                "nllb",
                (
                    "NllbTokenizer" if is_sentencepiece_available() else None,
                    "NllbTokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            (
                "nllb-moe",
                (
                    "NllbTokenizer" if is_sentencepiece_available() else None,
                    "NllbTokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            (
                "nystromformer",
                (
                    "AlbertTokenizer" if is_sentencepiece_available() else None,
                    "AlbertTokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            ("olmo", (None, "GPTNeoXTokenizerFast" if is_tokenizers_available() else None)),
            ("olmo2", (None, "GPTNeoXTokenizerFast" if is_tokenizers_available() else None)),
            ("olmoe", (None, "GPTNeoXTokenizerFast" if is_tokenizers_available() else None)),
            (
                "omdet-turbo",
                ("CLIPTokenizer", "CLIPTokenizerFast" if is_tokenizers_available() else None),
            ),
            ("oneformer", ("CLIPTokenizer", "CLIPTokenizerFast" if is_tokenizers_available() else None)),
            (
                "openai-gpt",
                ("OpenAIGPTTokenizer", "OpenAIGPTTokenizerFast" if is_tokenizers_available() else None),
            ),
            ("opt", ("GPT2Tokenizer", "GPT2TokenizerFast" if is_tokenizers_available() else None)),
            ("owlv2", ("CLIPTokenizer", "CLIPTokenizerFast" if is_tokenizers_available() else None)),
            ("owlvit", ("CLIPTokenizer", "CLIPTokenizerFast" if is_tokenizers_available() else None)),
            ("paligemma", ("LlamaTokenizer", "LlamaTokenizerFast" if is_tokenizers_available() else None)),
            (
                "pegasus",
                (
                    "PegasusTokenizer" if is_sentencepiece_available() else None,
                    "PegasusTokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            (
                "pegasus_x",
                (
                    "PegasusTokenizer" if is_sentencepiece_available() else None,
                    "PegasusTokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            (
                "perceiver",
                (
                    "PerceiverTokenizer",
                    None,
                ),
            ),
            (
                "persimmon",
                (
                    "LlamaTokenizer" if is_sentencepiece_available() else None,
                    "LlamaTokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            ("phi", ("CodeGenTokenizer", "CodeGenTokenizerFast" if is_tokenizers_available() else None)),
            ("phi3", ("LlamaTokenizer", "LlamaTokenizerFast" if is_tokenizers_available() else None)),
            ("phimoe", ("LlamaTokenizer", "LlamaTokenizerFast" if is_tokenizers_available() else None)),
            ("phobert", ("PhobertTokenizer", None)),
            ("pix2struct", ("T5Tokenizer", "T5TokenizerFast" if is_tokenizers_available() else None)),
            ("pixtral", (None, "PreTrainedTokenizerFast" if is_tokenizers_available() else None)),
            ("plbart", ("PLBartTokenizer" if is_sentencepiece_available() else None, None)),
            ("prophetnet", ("ProphetNetTokenizer", None)),
            ("qdqbert", ("BertTokenizer", "BertTokenizerFast" if is_tokenizers_available() else None)),
            (
                "qwen2",
                (
                    "Qwen2Tokenizer",
                    "Qwen2TokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            ("qwen2_audio", ("Qwen2Tokenizer", "Qwen2TokenizerFast" if is_tokenizers_available() else None)),
            (
                "qwen2_moe",
                (
                    "Qwen2Tokenizer",
                    "Qwen2TokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            ("qwen2_vl", ("Qwen2Tokenizer", "Qwen2TokenizerFast" if is_tokenizers_available() else None)),
            ("rag", ("RagTokenizer", None)),
            ("realm", ("RealmTokenizer", "RealmTokenizerFast" if is_tokenizers_available() else None)),
            (
                "recurrent_gemma",
                (
                    "GemmaTokenizer" if is_sentencepiece_available() else None,
                    "GemmaTokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            (
                "reformer",
                (
                    "ReformerTokenizer" if is_sentencepiece_available() else None,
                    "ReformerTokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            (
                "rembert",
                (
                    "RemBertTokenizer" if is_sentencepiece_available() else None,
                    "RemBertTokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            ("retribert", ("RetriBertTokenizer", "RetriBertTokenizerFast" if is_tokenizers_available() else None)),
            ("roberta", ("RobertaTokenizer", "RobertaTokenizerFast" if is_tokenizers_available() else None)),
            (
                "roberta-prelayernorm",
                ("RobertaTokenizer", "RobertaTokenizerFast" if is_tokenizers_available() else None),
            ),
            ("roc_bert", ("RoCBertTokenizer", None)),
            ("roformer", ("RoFormerTokenizer", "RoFormerTokenizerFast" if is_tokenizers_available() else None)),
            ("rwkv", (None, "GPTNeoXTokenizerFast" if is_tokenizers_available() else None)),
            (
                "seamless_m4t",
                (
                    "SeamlessM4TTokenizer" if is_sentencepiece_available() else None,
                    "SeamlessM4TTokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            (
                "seamless_m4t_v2",
                (
                    "SeamlessM4TTokenizer" if is_sentencepiece_available() else None,
                    "SeamlessM4TTokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            ("siglip", ("SiglipTokenizer" if is_sentencepiece_available() else None, None)),
            ("speech_to_text", ("Speech2TextTokenizer" if is_sentencepiece_available() else None, None)),
            ("speech_to_text_2", ("Speech2Text2Tokenizer", None)),
            ("speecht5", ("SpeechT5Tokenizer" if is_sentencepiece_available() else None, None)),
            ("splinter", ("SplinterTokenizer", "SplinterTokenizerFast")),
            (
                "squeezebert",
                ("SqueezeBertTokenizer", "SqueezeBertTokenizerFast" if is_tokenizers_available() else None),
            ),
            ("stablelm", (None, "GPTNeoXTokenizerFast" if is_tokenizers_available() else None)),
            ("starcoder2", ("GPT2Tokenizer", "GPT2TokenizerFast" if is_tokenizers_available() else None)),
            (
                "switch_transformers",
                (
                    "T5Tokenizer" if is_sentencepiece_available() else None,
                    "T5TokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            (
                "t5",
                (
                    "T5Tokenizer" if is_sentencepiece_available() else None,
                    "T5TokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            ("tapas", ("TapasTokenizer", None)),
            ("tapex", ("TapexTokenizer", None)),
            ("transfo-xl", ("TransfoXLTokenizer", None)),
            ("tvp", ("BertTokenizer", "BertTokenizerFast" if is_tokenizers_available() else None)),
            (
                "udop",
                (
                    "UdopTokenizer" if is_sentencepiece_available() else None,
                    "UdopTokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            (
                "umt5",
                (
                    "T5Tokenizer" if is_sentencepiece_available() else None,
                    "T5TokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            ("video_llava", ("LlamaTokenizer", "LlamaTokenizerFast" if is_tokenizers_available() else None)),
            ("vilt", ("BertTokenizer", "BertTokenizerFast" if is_tokenizers_available() else None)),
            ("vipllava", ("LlamaTokenizer", "LlamaTokenizerFast" if is_tokenizers_available() else None)),
            ("visual_bert", ("BertTokenizer", "BertTokenizerFast" if is_tokenizers_available() else None)),
            ("vits", ("VitsTokenizer", None)),
            ("wav2vec2", ("Wav2Vec2CTCTokenizer", None)),
            ("wav2vec2-bert", ("Wav2Vec2CTCTokenizer", None)),
            ("wav2vec2-conformer", ("Wav2Vec2CTCTokenizer", None)),
            ("wav2vec2_phoneme", ("Wav2Vec2PhonemeCTCTokenizer", None)),
            ("whisper", ("WhisperTokenizer", "WhisperTokenizerFast" if is_tokenizers_available() else None)),
            ("xclip", ("CLIPTokenizer", "CLIPTokenizerFast" if is_tokenizers_available() else None)),
            (
                "xglm",
                (
                    "XGLMTokenizer" if is_sentencepiece_available() else None,
                    "XGLMTokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            ("xlm", ("XLMTokenizer", None)),
            ("xlm-prophetnet", ("XLMProphetNetTokenizer" if is_sentencepiece_available() else None, None)),
            (
                "xlm-roberta",
                (
                    "XLMRobertaTokenizer" if is_sentencepiece_available() else None,
                    "XLMRobertaTokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            (
                "xlm-roberta-xl",
                (
                    "XLMRobertaTokenizer" if is_sentencepiece_available() else None,
                    "XLMRobertaTokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            (
                "xlnet",
                (
                    "XLNetTokenizer" if is_sentencepiece_available() else None,
                    "XLNetTokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            (
                "xmod",
                (
                    "XLMRobertaTokenizer" if is_sentencepiece_available() else None,
                    "XLMRobertaTokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            (
                "yoso",
                (
                    "AlbertTokenizer" if is_sentencepiece_available() else None,
                    "AlbertTokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            (
                "zamba",
                (
                    "LlamaTokenizer" if is_sentencepiece_available() else None,
                    "LlamaTokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            (
                "zamba2",
                (
                    "LlamaTokenizer" if is_sentencepiece_available() else None,
                    "LlamaTokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
        ]
    )

TOKENIZER_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, TOKENIZER_MAPPING_NAMES)

CONFIG_TO_TYPE = {v: k for k, v in CONFIG_MAPPING_NAMES.items()}


def tokenizer_class_from_name(class_name: str):
    if class_name == "PreTrainedTokenizerFast":
        return PreTrainedTokenizerFast

    for module_name, tokenizers in TOKENIZER_MAPPING_NAMES.items():
        if class_name in tokenizers:
            module_name = model_type_to_module_name(module_name)

            module = importlib.import_module(f".{module_name}", "transformers.models")
            try:
                return getattr(module, class_name)
            except AttributeError:
                continue

    for config, tokenizers in TOKENIZER_MAPPING._extra_content.items():
        for tokenizer in tokenizers:
            if getattr(tokenizer, "__name__", None) == class_name:
                return tokenizer

    # We did not fine the class, but maybe it's because a dep is missing. In that case, the class will be in the main
    # init and we return the proper dummy to get an appropriate error message.
    main_module = importlib.import_module("transformers")
    if hasattr(main_module, class_name):
        return getattr(main_module, class_name)

    return None


def get_tokenizer_config(
    pretrained_model_name_or_path: Union[str, os.PathLike],
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    force_download: bool = False,
    resume_download: Optional[bool] = None,
    proxies: Optional[Dict[str, str]] = None,
    token: Optional[Union[bool, str]] = None,
    revision: Optional[str] = None,
    local_files_only: bool = False,
    subfolder: str = "",
    **kwargs,
):
    """
    Loads the tokenizer configuration from a pretrained model tokenizer configuration.

    Args:
        pretrained_model_name_or_path (`str` or `os.PathLike`):
            This can be either:

            - a string, the *model id* of a pretrained model configuration hosted inside a model repo on
              huggingface.co.
            - a path to a *directory* containing a configuration file saved using the
              [`~PreTrainedTokenizer.save_pretrained`] method, e.g., `./my_model_directory/`.

        cache_dir (`str` or `os.PathLike`, *optional*):
            Path to a directory in which a downloaded pretrained model configuration should be cached if the standard
            cache should not be used.
        force_download (`bool`, *optional*, defaults to `False`):
            Whether or not to force to (re-)download the configuration files and override the cached versions if they
            exist.
        resume_download:
            Deprecated and ignored. All downloads are now resumed by default when possible.
            Will be removed in v5 of Transformers.
        proxies (`Dict[str, str]`, *optional*):
            A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
            'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
        token (`str` or *bool*, *optional*):
            The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
            when running `huggingface-cli login` (stored in `~/.huggingface`).
        revision (`str`, *optional*, defaults to `"main"`):
            The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
            git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
            identifier allowed by git.
        local_files_only (`bool`, *optional*, defaults to `False`):
            If `True`, will only try to load the tokenizer configuration from local files.
        subfolder (`str`, *optional*, defaults to `""`):
            In case the tokenizer config is located inside a subfolder of the model repo on huggingface.co, you can
            specify the folder name here.

    <Tip>

    Passing `token=True` is required when you want to use a private model.

    </Tip>

    Returns:
        `Dict`: The configuration of the tokenizer.

    Examples:

    ```python
    # Download configuration from huggingface.co and cache.
    tokenizer_config = get_tokenizer_config("google-bert/bert-base-uncased")
    # This model does not have a tokenizer config so the result will be an empty dict.
    tokenizer_config = get_tokenizer_config("FacebookAI/xlm-roberta-base")

    # Save a pretrained tokenizer locally and you can reload its config
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
    tokenizer.save_pretrained("tokenizer-test")
    tokenizer_config = get_tokenizer_config("tokenizer-test")
    ```"""
    use_auth_token = kwargs.pop("use_auth_token", None)
    if use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
            FutureWarning,
        )
        if token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        token = use_auth_token

    commit_hash = kwargs.get("_commit_hash", None)
    resolved_config_file = cached_file(
        pretrained_model_name_or_path,
        TOKENIZER_CONFIG_FILE,
        cache_dir=cache_dir,
        force_download=force_download,
        resume_download=resume_download,
        proxies=proxies,
        token=token,
        revision=revision,
        local_files_only=local_files_only,
        subfolder=subfolder,
        _raise_exceptions_for_gated_repo=False,
        _raise_exceptions_for_missing_entries=False,
        _raise_exceptions_for_connection_errors=False,
        _commit_hash=commit_hash,
    )
    if resolved_config_file is None:
        logger.info("Could not locate the tokenizer configuration file, will try to use the model config instead.")
        return {}
    commit_hash = extract_commit_hash(resolved_config_file, commit_hash)

    with open(resolved_config_file, encoding="utf-8") as reader:
        result = json.load(reader)
    result["_commit_hash"] = commit_hash
    return result


class AutoTokenizer:
    r"""
    This is a generic tokenizer class that will be instantiated as one of the tokenizer classes of the library when
    created with the [`AutoTokenizer.from_pretrained`] class method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoTokenizer is designed to be instantiated "
            "using the `AutoTokenizer.from_pretrained(pretrained_model_name_or_path)` method."
        )

    @classmethod
    @replace_list_option_in_docstrings(TOKENIZER_MAPPING_NAMES)
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        r"""
        Instantiate one of the tokenizer classes of the library from a pretrained model vocabulary.

        The tokenizer class to instantiate is selected based on the `model_type` property of the config object (either
        passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it's missing, by
        falling back to using pattern matching on `pretrained_model_name_or_path`:

        List options

        Params:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                Can be either:

                    - A string, the *model id* of a predefined tokenizer hosted inside a model repo on huggingface.co.
                    - A path to a *directory* containing vocabulary files required by the tokenizer, for instance saved
                      using the [`~PreTrainedTokenizer.save_pretrained`] method, e.g., `./my_model_directory/`.
                    - A path or url to a single saved vocabulary file if and only if the tokenizer only requires a
                      single vocabulary file (like Bert or XLNet), e.g.: `./my_model_directory/vocab.txt`. (Not
                      applicable to all derived classes)
            inputs (additional positional arguments, *optional*):
                Will be passed along to the Tokenizer `__init__()` method.
            config ([`PretrainedConfig`], *optional*)
                The configuration object used to determine the tokenizer class to instantiate.
            cache_dir (`str` or `os.PathLike`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download the model weights and configuration files and override the
                cached versions if they exist.
            resume_download:
                Deprecated and ignored. All downloads are now resumed by default when possible.
                Will be removed in v5 of Transformers.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.
            subfolder (`str`, *optional*):
                In case the relevant files are located inside a subfolder of the model repo on huggingface.co (e.g. for
                facebook/rag-token-base), specify it here.
            use_fast (`bool`, *optional*, defaults to `True`):
                Use a [fast Rust-based tokenizer](https://huggingface.co/docs/tokenizers/index) if it is supported for
                a given model. If a fast tokenizer is not available for a given model, a normal Python-based tokenizer
                is returned instead.
            tokenizer_type (`str`, *optional*):
                Tokenizer type to be loaded.
            trust_remote_code (`bool`, *optional*, defaults to `False`):
                Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
                should only be set to `True` for repositories you trust and in which you have read the code, as it will
                execute code present on the Hub on your local machine.
            kwargs (additional keyword arguments, *optional*):
                Will be passed to the Tokenizer `__init__()` method. Can be used to set special tokens like
                `bos_token`, `eos_token`, `unk_token`, `sep_token`, `pad_token`, `cls_token`, `mask_token`,
                `additional_special_tokens`. See parameters in the `__init__()` for more details.

        Examples:

        ```python
        >>> from transformers import AutoTokenizer

        >>> # Download vocabulary from huggingface.co and cache.
        >>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

        >>> # Download vocabulary from huggingface.co (user-uploaded) and cache.
        >>> tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-cased")

        >>> # If vocabulary files are in a directory (e.g. tokenizer was saved using *save_pretrained('./test/saved_model/')*)
        >>> # tokenizer = AutoTokenizer.from_pretrained("./test/bert_saved_model/")

        >>> # Download vocabulary from huggingface.co and define model-specific arguments
        >>> tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base", add_prefix_space=True)
        ```"""
        use_auth_token = kwargs.pop("use_auth_token", None)
        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
                FutureWarning,
            )
            if kwargs.get("token", None) is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            kwargs["token"] = use_auth_token

        config = kwargs.pop("config", None)
        kwargs["_from_auto"] = True

        use_fast = kwargs.pop("use_fast", True)
        tokenizer_type = kwargs.pop("tokenizer_type", None)
        trust_remote_code = kwargs.pop("trust_remote_code", None)
        gguf_file = kwargs.get("gguf_file", None)

        # First, let's see whether the tokenizer_type is passed so that we can leverage it
        if tokenizer_type is not None:
            tokenizer_class = None
            tokenizer_class_tuple = TOKENIZER_MAPPING_NAMES.get(tokenizer_type, None)

            if tokenizer_class_tuple is None:
                raise ValueError(
                    f"Passed `tokenizer_type` {tokenizer_type} does not exist. `tokenizer_type` should be one of "
                    f"{', '.join(c for c in TOKENIZER_MAPPING_NAMES.keys())}."
                )

            tokenizer_class_name, tokenizer_fast_class_name = tokenizer_class_tuple

            if use_fast:
                if tokenizer_fast_class_name is not None:
                    tokenizer_class = tokenizer_class_from_name(tokenizer_fast_class_name)
                else:
                    logger.warning(
                        "`use_fast` is set to `True` but the tokenizer class does not have a fast version. "
                        " Falling back to the slow version."
                    )
            if tokenizer_class is None:
                tokenizer_class = tokenizer_class_from_name(tokenizer_class_name)

            if tokenizer_class is None:
                raise ValueError(f"Tokenizer class {tokenizer_class_name} is not currently imported.")

            return tokenizer_class.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)

        # Next, let's try to use the tokenizer_config file to get the tokenizer class.
        tokenizer_config = get_tokenizer_config(pretrained_model_name_or_path, **kwargs)
        if "_commit_hash" in tokenizer_config:
            kwargs["_commit_hash"] = tokenizer_config["_commit_hash"]
        config_tokenizer_class = tokenizer_config.get("tokenizer_class")
        tokenizer_auto_map = None
        if "auto_map" in tokenizer_config:
            if isinstance(tokenizer_config["auto_map"], (tuple, list)):
                # Legacy format for dynamic tokenizers
                tokenizer_auto_map = tokenizer_config["auto_map"]
            else:
                tokenizer_auto_map = tokenizer_config["auto_map"].get("AutoTokenizer", None)

        # If that did not work, let's try to use the config.
        if config_tokenizer_class is None:
            if not isinstance(config, PretrainedConfig):
                if gguf_file:
                    gguf_path = cached_file(pretrained_model_name_or_path, gguf_file, **kwargs)
                    config_dict = load_gguf_checkpoint(gguf_path, return_tensors=False)["config"]
                    config = AutoConfig.for_model(**config_dict)
                else:
                    config = AutoConfig.from_pretrained(
                        pretrained_model_name_or_path, trust_remote_code=trust_remote_code, **kwargs
                    )
            config_tokenizer_class = config.tokenizer_class
            if hasattr(config, "auto_map") and "AutoTokenizer" in config.auto_map:
                tokenizer_auto_map = config.auto_map["AutoTokenizer"]

        has_remote_code = tokenizer_auto_map is not None
        has_local_code = type(config) in TOKENIZER_MAPPING or (
            config_tokenizer_class is not None
            and (
                tokenizer_class_from_name(config_tokenizer_class) is not None
                or tokenizer_class_from_name(config_tokenizer_class + "Fast") is not None
            )
        )
        trust_remote_code = resolve_trust_remote_code(
            trust_remote_code, pretrained_model_name_or_path, has_local_code, has_remote_code
        )

        if has_remote_code and trust_remote_code:
            if use_fast and tokenizer_auto_map[1] is not None:
                class_ref = tokenizer_auto_map[1]
            else:
                class_ref = tokenizer_auto_map[0]
            tokenizer_class = get_class_from_dynamic_module(class_ref, pretrained_model_name_or_path, **kwargs)
            _ = kwargs.pop("code_revision", None)
            if os.path.isdir(pretrained_model_name_or_path):
                tokenizer_class.register_for_auto_class()
            return tokenizer_class.from_pretrained(
                pretrained_model_name_or_path, *inputs, trust_remote_code=trust_remote_code, **kwargs
            )
        elif config_tokenizer_class is not None:
            tokenizer_class = None
            if use_fast and not config_tokenizer_class.endswith("Fast"):
                tokenizer_class_candidate = f"{config_tokenizer_class}Fast"
                tokenizer_class = tokenizer_class_from_name(tokenizer_class_candidate)
            if tokenizer_class is None:
                tokenizer_class_candidate = config_tokenizer_class
                tokenizer_class = tokenizer_class_from_name(tokenizer_class_candidate)
            if tokenizer_class is None:
                raise ValueError(
                    f"Tokenizer class {tokenizer_class_candidate} does not exist or is not currently imported."
                )
            return tokenizer_class.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)

        # Otherwise we have to be creative.
        # if model is an encoder decoder, the encoder tokenizer class is used by default
        if isinstance(config, EncoderDecoderConfig):
            if type(config.decoder) is not type(config.encoder):  # noqa: E721
                logger.warning(
                    f"The encoder model config class: {config.encoder.__class__} is different from the decoder model "
                    f"config class: {config.decoder.__class__}. It is not recommended to use the "
                    "`AutoTokenizer.from_pretrained()` method in this case. Please use the encoder and decoder "
                    "specific tokenizer classes."
                )
            config = config.encoder

        model_type = config_class_to_model_type(type(config).__name__)
        if model_type is not None:
            tokenizer_class_py, tokenizer_class_fast = TOKENIZER_MAPPING[type(config)]

            if tokenizer_class_fast and (use_fast or tokenizer_class_py is None):
                return tokenizer_class_fast.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
            else:
                if tokenizer_class_py is not None:
                    return tokenizer_class_py.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
                else:
                    raise ValueError(
                        "This tokenizer cannot be instantiated. Please make sure you have `sentencepiece` installed "
                        "in order to use this tokenizer."
                    )

        raise ValueError(
            f"Unrecognized configuration class {config.__class__} to build an AutoTokenizer.\n"
            f"Model type should be one of {', '.join(c.__name__ for c in TOKENIZER_MAPPING.keys())}."
        )

    def register(config_class, slow_tokenizer_class=None, fast_tokenizer_class=None, exist_ok=False):
        """
        Register a new tokenizer in this mapping.


        Args:
            config_class ([`PretrainedConfig`]):
                The configuration corresponding to the model to register.
            slow_tokenizer_class ([`PretrainedTokenizer`], *optional*):
                The slow tokenizer to register.
            fast_tokenizer_class ([`PretrainedTokenizerFast`], *optional*):
                The fast tokenizer to register.
        """
        if slow_tokenizer_class is None and fast_tokenizer_class is None:
            raise ValueError("You need to pass either a `slow_tokenizer_class` or a `fast_tokenizer_class")
        if slow_tokenizer_class is not None and issubclass(slow_tokenizer_class, PreTrainedTokenizerFast):
            raise ValueError("You passed a fast tokenizer in the `slow_tokenizer_class`.")
        if fast_tokenizer_class is not None and issubclass(fast_tokenizer_class, PreTrainedTokenizer):
            raise ValueError("You passed a slow tokenizer in the `fast_tokenizer_class`.")

        if (
            slow_tokenizer_class is not None
            and fast_tokenizer_class is not None
            and issubclass(fast_tokenizer_class, PreTrainedTokenizerFast)
            and fast_tokenizer_class.slow_tokenizer_class != slow_tokenizer_class
        ):
            raise ValueError(
                "The fast tokenizer class you are passing has a `slow_tokenizer_class` attribute that is not "
                "consistent with the slow tokenizer class you passed (fast tokenizer has "
                f"{fast_tokenizer_class.slow_tokenizer_class} and you passed {slow_tokenizer_class}. Fix one of those "
                "so they match!"
            )

        # Avoid resetting a set slow/fast tokenizer if we are passing just the other ones.
        if config_class in TOKENIZER_MAPPING._extra_content:
            existing_slow, existing_fast = TOKENIZER_MAPPING[config_class]
            if slow_tokenizer_class is None:
                slow_tokenizer_class = existing_slow
            if fast_tokenizer_class is None:
                fast_tokenizer_class = existing_fast

        TOKENIZER_MAPPING.register(config_class, (slow_tokenizer_class, fast_tokenizer_class), exist_ok=exist_ok)
