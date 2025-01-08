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
"""
File utilities: utilities related to download and cache models

This module should not be update anymore and is only left for backward compatibility.
"""

from huggingface_hub import get_full_repo_name  # for backward compatibility
from huggingface_hub.constants import HF_HUB_DISABLE_TELEMETRY as DISABLE_TELEMETRY  # for backward compatibility

from . import __version__

# Backward compatibility imports, to make sure all those objects can be found in file_utils
from .utils import (
    CLOUDFRONT_DISTRIB_PREFIX,
    CONFIG_NAME,
    DUMMY_INPUTS,
    DUMMY_MASK,
    ENV_VARS_TRUE_AND_AUTO_VALUES,
    ENV_VARS_TRUE_VALUES,
    FEATURE_EXTRACTOR_NAME,
    FLAX_WEIGHTS_NAME,
    HF_MODULES_CACHE,
    HUGGINGFACE_CO_PREFIX,
    HUGGINGFACE_CO_RESOLVE_ENDPOINT,
    MODEL_CARD_NAME,
    MULTIPLE_CHOICE_DUMMY_INPUTS,
    PYTORCH_PRETRAINED_BERT_CACHE,
    PYTORCH_TRANSFORMERS_CACHE,
    S3_BUCKET_PREFIX,
    SENTENCEPIECE_UNDERLINE,
    SPIECE_UNDERLINE,
    TF2_WEIGHTS_NAME,
    TF_WEIGHTS_NAME,
    TORCH_FX_REQUIRED_VERSION,
    TRANSFORMERS_CACHE,
    TRANSFORMERS_DYNAMIC_MODULE_NAME,
    USE_JAX,
    USE_TF,
    USE_TORCH,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    ContextManagers,
    DummyObject,
    EntryNotFoundError,
    ExplicitEnum,
    ModelOutput,
    PaddingStrategy,
    PushToHubMixin,
    RepositoryNotFoundError,
    RevisionNotFoundError,
    TensorType,
    _LazyModule,
    add_code_sample_docstrings,
    add_end_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    cached_property,
    copy_func,
    default_cache_path,
    define_sagemaker_information,
    get_cached_models,
    get_file_from_repo,
    get_torch_version,
    has_file,
    http_user_agent,
    is_apex_available,
    is_bs4_available,
    is_coloredlogs_available,
    is_datasets_available,
    is_detectron2_available,
    is_faiss_available,
    is_flax_available,
    is_ftfy_available,
    is_g2p_en_available,
    is_in_notebook,
    is_ipex_available,
    is_librosa_available,
    is_offline_mode,
    is_onnx_available,
    is_pandas_available,
    is_phonemizer_available,
    is_protobuf_available,
    is_psutil_available,
    is_py3nvml_available,
    is_pyctcdecode_available,
    is_pytesseract_available,
    is_pytorch_quantization_available,
    is_rjieba_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_scipy_available,
    is_sentencepiece_available,
    is_seqio_available,
    is_sklearn_available,
    is_soundfile_available,
    is_spacy_available,
    is_speech_available,
    is_tensor,
    is_tensorflow_probability_available,
    is_tf2onnx_available,
    is_tf_available,
    is_timm_available,
    is_tokenizers_available,
    is_torch_available,
    is_torch_bf16_available,
    is_torch_cuda_available,
    is_torch_fx_available,
    is_torch_fx_proxy,
    is_torch_mps_available,
    is_torch_tf32_available,
    is_torch_xla_available,
    is_torchaudio_available,
    is_training_run_on_sagemaker,
    is_vision_available,
    replace_return_docstrings,
    requires_backends,
    to_numpy,
    to_py_obj,
    torch_only_method,
)
