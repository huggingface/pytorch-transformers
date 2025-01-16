# Copyright 2022 The HuggingFace Team. All rights reserved.
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
Import utilities: Utilities related to imports and our lazy inits.
"""

import importlib.machinery
import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import warnings
from collections import OrderedDict
from functools import lru_cache
from itertools import chain
from types import ModuleType
from typing import Any, Dict, FrozenSet, Optional, Set, Tuple, Union

from packaging import version

from . import logging


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# TODO: This doesn't work for all packages (`bs4`, `faiss`, etc.) Talk to Sylvain to see how to do with it better.
def _is_package_available(pkg_name: str, return_version: bool = False) -> Union[Tuple[bool, str], bool]:
    # Check if the package spec exists and grab its version to avoid importing a local directory
    package_exists = importlib.util.find_spec(pkg_name) is not None
    package_version = "N/A"
    if package_exists:
        try:
            # Primary method to get the package version
            package_version = importlib.metadata.version(pkg_name)
        except importlib.metadata.PackageNotFoundError:
            # Fallback method: Only for "torch" and versions containing "dev"
            if pkg_name == "torch":
                try:
                    package = importlib.import_module(pkg_name)
                    temp_version = getattr(package, "__version__", "N/A")
                    # Check if the version contains "dev"
                    if "dev" in temp_version:
                        package_version = temp_version
                        package_exists = True
                    else:
                        package_exists = False
                except ImportError:
                    # If the package can't be imported, it's not available
                    package_exists = False
            else:
                # For packages other than "torch", don't attempt the fallback and set as not available
                package_exists = False
        logger.debug(f"Detected {pkg_name} version: {package_version}")
    if return_version:
        return package_exists, package_version
    else:
        return package_exists


ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}
ENV_VARS_TRUE_AND_AUTO_VALUES = ENV_VARS_TRUE_VALUES.union({"AUTO"})

USE_TF = os.environ.get("USE_TF", "AUTO").upper()
USE_TORCH = os.environ.get("USE_TORCH", "AUTO").upper()
USE_JAX = os.environ.get("USE_FLAX", "AUTO").upper()

# Try to run a native pytorch job in an environment with TorchXLA installed by setting this value to 0.
USE_TORCH_XLA = os.environ.get("USE_TORCH_XLA", "1").upper()

FORCE_TF_AVAILABLE = os.environ.get("FORCE_TF_AVAILABLE", "AUTO").upper()

# `transformers` requires `torch>=1.11` but this variable is exposed publicly, and we can't simply remove it.
# This is the version of torch required to run torch.fx features and torch.onnx with dictionary inputs.
TORCH_FX_REQUIRED_VERSION = version.parse("1.10")

ACCELERATE_MIN_VERSION = "0.26.0"
FSDP_MIN_VERSION = "1.12.0"
GGUF_MIN_VERSION = "0.10.0"
XLA_FSDPV2_MIN_VERSION = "2.2.0"
HQQ_MIN_VERSION = "0.2.1"
VPTQ_MIN_VERSION = "0.0.4"


_accelerate_available, _accelerate_version = _is_package_available("accelerate", return_version=True)
_apex_available = _is_package_available("apex")
_aqlm_available = _is_package_available("aqlm")
_vptq_available, _vptq_version = _is_package_available("vptq", return_version=True)
_av_available = importlib.util.find_spec("av") is not None
_decord_available = importlib.util.find_spec("decord") is not None
_bitsandbytes_available = _is_package_available("bitsandbytes")
_eetq_available = _is_package_available("eetq")
_fbgemm_gpu_available = _is_package_available("fbgemm_gpu")
_galore_torch_available = _is_package_available("galore_torch")
_lomo_available = _is_package_available("lomo_optim")
_grokadamw_available = _is_package_available("grokadamw")
_schedulefree_available = _is_package_available("schedulefree")
# `importlib.metadata.version` doesn't work with `bs4` but `beautifulsoup4`. For `importlib.util.find_spec`, reversed.
_bs4_available = importlib.util.find_spec("bs4") is not None
_coloredlogs_available = _is_package_available("coloredlogs")
# `importlib.metadata.util` doesn't work with `opencv-python-headless`.
_cv2_available = importlib.util.find_spec("cv2") is not None
_yt_dlp_available = importlib.util.find_spec("yt_dlp") is not None
_datasets_available = _is_package_available("datasets")
_detectron2_available = _is_package_available("detectron2")
# We need to check both `faiss` and `faiss-cpu`.
_faiss_available = importlib.util.find_spec("faiss") is not None
try:
    _faiss_version = importlib.metadata.version("faiss")
    logger.debug(f"Successfully imported faiss version {_faiss_version}")
except importlib.metadata.PackageNotFoundError:
    try:
        _faiss_version = importlib.metadata.version("faiss-cpu")
        logger.debug(f"Successfully imported faiss version {_faiss_version}")
    except importlib.metadata.PackageNotFoundError:
        _faiss_available = False
_ftfy_available = _is_package_available("ftfy")
_g2p_en_available = _is_package_available("g2p_en")
_hadamard_available = _is_package_available("fast_hadamard_transform")
_ipex_available, _ipex_version = _is_package_available("intel_extension_for_pytorch", return_version=True)
_jieba_available = _is_package_available("jieba")
_jinja_available = _is_package_available("jinja2")
_kenlm_available = _is_package_available("kenlm")
_keras_nlp_available = _is_package_available("keras_nlp")
_levenshtein_available = _is_package_available("Levenshtein")
_librosa_available = _is_package_available("librosa")
_natten_available = _is_package_available("natten")
_nltk_available = _is_package_available("nltk")
_onnx_available = _is_package_available("onnx")
_openai_available = _is_package_available("openai")
_optimum_available = _is_package_available("optimum")
_auto_gptq_available = _is_package_available("auto_gptq")
_gptqmodel_available = _is_package_available("gptqmodel")
# `importlib.metadata.version` doesn't work with `awq`
_auto_awq_available = importlib.util.find_spec("awq") is not None
_quanto_available = _is_package_available("quanto")
_is_optimum_quanto_available = False
try:
    importlib.metadata.version("optimum_quanto")
    _is_optimum_quanto_available = True
except importlib.metadata.PackageNotFoundError:
    _is_optimum_quanto_available = False
# For compressed_tensors, only check spec to allow compressed_tensors-nightly package
_compressed_tensors_available = importlib.util.find_spec("compressed_tensors") is not None
_pandas_available = _is_package_available("pandas")
_peft_available = _is_package_available("peft")
_phonemizer_available = _is_package_available("phonemizer")
_uroman_available = _is_package_available("uroman")
_psutil_available = _is_package_available("psutil")
_py3nvml_available = _is_package_available("py3nvml")
_pyctcdecode_available = _is_package_available("pyctcdecode")
_pygments_available = _is_package_available("pygments")
_pytesseract_available = _is_package_available("pytesseract")
_pytest_available = _is_package_available("pytest")
_pytorch_quantization_available = _is_package_available("pytorch_quantization")
_rjieba_available = _is_package_available("rjieba")
_sacremoses_available = _is_package_available("sacremoses")
_safetensors_available = _is_package_available("safetensors")
_scipy_available = _is_package_available("scipy")
_sentencepiece_available = _is_package_available("sentencepiece")
_is_seqio_available = _is_package_available("seqio")
_is_gguf_available, _gguf_version = _is_package_available("gguf", return_version=True)
_sklearn_available = importlib.util.find_spec("sklearn") is not None
if _sklearn_available:
    try:
        importlib.metadata.version("scikit-learn")
    except importlib.metadata.PackageNotFoundError:
        _sklearn_available = False
_smdistributed_available = importlib.util.find_spec("smdistributed") is not None
_soundfile_available = _is_package_available("soundfile")
_spacy_available = _is_package_available("spacy")
_sudachipy_available, _sudachipy_version = _is_package_available("sudachipy", return_version=True)
_tensorflow_probability_available = _is_package_available("tensorflow_probability")
_tensorflow_text_available = _is_package_available("tensorflow_text")
_tf2onnx_available = _is_package_available("tf2onnx")
_timm_available = _is_package_available("timm")
_tokenizers_available = _is_package_available("tokenizers")
_torchaudio_available = _is_package_available("torchaudio")
_torchao_available = _is_package_available("torchao")
_torchdistx_available = _is_package_available("torchdistx")
_torchvision_available, _torchvision_version = _is_package_available("torchvision", return_version=True)
_mlx_available = _is_package_available("mlx")
_hqq_available, _hqq_version = _is_package_available("hqq", return_version=True)
_tiktoken_available = _is_package_available("tiktoken")
_blobfile_available = _is_package_available("blobfile")
_liger_kernel_available = _is_package_available("liger_kernel")
_triton_available = _is_package_available("triton")
_spqr_available = _is_package_available("spqr_quant")

_torch_version = "N/A"
_torch_available = False
if USE_TORCH in ENV_VARS_TRUE_AND_AUTO_VALUES and USE_TF not in ENV_VARS_TRUE_VALUES:
    _torch_available, _torch_version = _is_package_available("torch", return_version=True)
else:
    logger.info("Disabling PyTorch because USE_TF is set")
    _torch_available = False


_tf_version = "N/A"
_tf_available = False
if FORCE_TF_AVAILABLE in ENV_VARS_TRUE_VALUES:
    _tf_available = True
else:
    if USE_TF in ENV_VARS_TRUE_AND_AUTO_VALUES and USE_TORCH not in ENV_VARS_TRUE_VALUES:
        # Note: _is_package_available("tensorflow") fails for tensorflow-cpu. Please test any changes to the line below
        # with tensorflow-cpu to make sure it still works!
        _tf_available = importlib.util.find_spec("tensorflow") is not None
        if _tf_available:
            candidates = (
                "tensorflow",
                "tensorflow-cpu",
                "tensorflow-gpu",
                "tf-nightly",
                "tf-nightly-cpu",
                "tf-nightly-gpu",
                "tf-nightly-rocm",
                "intel-tensorflow",
                "intel-tensorflow-avx512",
                "tensorflow-rocm",
                "tensorflow-macos",
                "tensorflow-aarch64",
            )
            _tf_version = None
            # For the metadata, we have to look for both tensorflow and tensorflow-cpu
            for pkg in candidates:
                try:
                    _tf_version = importlib.metadata.version(pkg)
                    break
                except importlib.metadata.PackageNotFoundError:
                    pass
            _tf_available = _tf_version is not None
        if _tf_available:
            if version.parse(_tf_version) < version.parse("2"):
                logger.info(
                    f"TensorFlow found but with version {_tf_version}. Transformers requires version 2 minimum."
                )
                _tf_available = False
    else:
        logger.info("Disabling Tensorflow because USE_TORCH is set")


_essentia_available = importlib.util.find_spec("essentia") is not None
try:
    _essentia_version = importlib.metadata.version("essentia")
    logger.debug(f"Successfully imported essentia version {_essentia_version}")
except importlib.metadata.PackageNotFoundError:
    _essentia_version = False


_pretty_midi_available = importlib.util.find_spec("pretty_midi") is not None
try:
    _pretty_midi_version = importlib.metadata.version("pretty_midi")
    logger.debug(f"Successfully imported pretty_midi version {_pretty_midi_version}")
except importlib.metadata.PackageNotFoundError:
    _pretty_midi_available = False


ccl_version = "N/A"
_is_ccl_available = (
    importlib.util.find_spec("torch_ccl") is not None
    or importlib.util.find_spec("oneccl_bindings_for_pytorch") is not None
)
try:
    ccl_version = importlib.metadata.version("oneccl_bind_pt")
    logger.debug(f"Detected oneccl_bind_pt version {ccl_version}")
except importlib.metadata.PackageNotFoundError:
    _is_ccl_available = False


_flax_available = False
if USE_JAX in ENV_VARS_TRUE_AND_AUTO_VALUES:
    _flax_available, _flax_version = _is_package_available("flax", return_version=True)
    if _flax_available:
        _jax_available, _jax_version = _is_package_available("jax", return_version=True)
        if _jax_available:
            logger.info(f"JAX version {_jax_version}, Flax version {_flax_version} available.")
        else:
            _flax_available = _jax_available = False
            _jax_version = _flax_version = "N/A"


_torch_fx_available = False
if _torch_available:
    torch_version = version.parse(_torch_version)
    _torch_fx_available = (torch_version.major, torch_version.minor) >= (
        TORCH_FX_REQUIRED_VERSION.major,
        TORCH_FX_REQUIRED_VERSION.minor,
    )


_torch_xla_available = False
if USE_TORCH_XLA in ENV_VARS_TRUE_VALUES:
    _torch_xla_available, _torch_xla_version = _is_package_available("torch_xla", return_version=True)
    if _torch_xla_available:
        logger.info(f"Torch XLA version {_torch_xla_version} available.")


def is_kenlm_available():
    return _kenlm_available


def is_cv2_available():
    return _cv2_available


def is_yt_dlp_available():
    return _yt_dlp_available


def is_torch_available():
    return _torch_available


def is_accelerate_available(min_version: str = ACCELERATE_MIN_VERSION):
    return _accelerate_available and version.parse(_accelerate_version) >= version.parse(min_version)


def is_torch_deterministic():
    """
    Check whether pytorch uses deterministic algorithms by looking if torch.set_deterministic_debug_mode() is set to 1 or 2"
    """
    import torch

    if torch.get_deterministic_debug_mode() == 0:
        return False
    else:
        return True


def is_hadamard_available():
    return _hadamard_available


def is_hqq_available(min_version: str = HQQ_MIN_VERSION):
    return _hqq_available and version.parse(_hqq_version) >= version.parse(min_version)


def is_pygments_available():
    return _pygments_available


def get_torch_version():
    return _torch_version


def is_torch_sdpa_available():
    if not is_torch_available():
        return False
    elif _torch_version == "N/A":
        return False

    # NOTE: We require torch>=2.1 (and not torch>=2.0) to use SDPA in Transformers for two reasons:
    # - Allow the global use of the `scale` argument introduced in https://github.com/pytorch/pytorch/pull/95259
    # - Memory-efficient attention supports arbitrary attention_mask: https://github.com/pytorch/pytorch/pull/104310
    # NOTE: MLU is OK with non-contiguous inputs.
    if is_torch_mlu_available():
        return version.parse(_torch_version) >= version.parse("2.1.0")
    # NOTE: NPU can use SDPA in Transformers with torch>=2.1.0.
    if is_torch_npu_available():
        return version.parse(_torch_version) >= version.parse("2.1.0")
    # NOTE: We require torch>=2.1.1 to avoid a numerical issue in SDPA with non-contiguous inputs: https://github.com/pytorch/pytorch/issues/112577
    return version.parse(_torch_version) >= version.parse("2.1.1")


def is_torch_flex_attn_available():
    if not is_torch_available():
        return False
    elif _torch_version == "N/A":
        return False

    # TODO check if some bugs cause push backs on the exact version
    # NOTE: We require torch>=2.5.0 as it is the first release
    return version.parse(_torch_version) >= version.parse("2.5.0")


def is_torchvision_available():
    return _torchvision_available


def is_torchvision_v2_available():
    if not is_torchvision_available():
        return False

    # NOTE: We require torchvision>=0.15 as v2 transforms are available from this version: https://pytorch.org/vision/stable/transforms.html#v1-or-v2-which-one-should-i-use
    return version.parse(_torchvision_version) >= version.parse("0.15")


def is_galore_torch_available():
    return _galore_torch_available


def is_lomo_available():
    return _lomo_available


def is_grokadamw_available():
    return _grokadamw_available


def is_schedulefree_available():
    return _schedulefree_available


def is_pyctcdecode_available():
    return _pyctcdecode_available


def is_librosa_available():
    return _librosa_available


def is_essentia_available():
    return _essentia_available


def is_pretty_midi_available():
    return _pretty_midi_available


def is_torch_cuda_available():
    if is_torch_available():
        import torch

        return torch.cuda.is_available()
    else:
        return False


def is_mamba_ssm_available():
    if is_torch_available():
        import torch

        if not torch.cuda.is_available():
            return False
        else:
            return _is_package_available("mamba_ssm")
    return False


def is_mamba_2_ssm_available():
    if is_torch_available():
        import torch

        if not torch.cuda.is_available():
            return False
        else:
            if _is_package_available("mamba_ssm"):
                import mamba_ssm

                if version.parse(mamba_ssm.__version__) >= version.parse("2.0.4"):
                    return True
    return False


def is_causal_conv1d_available():
    if is_torch_available():
        import torch

        if not torch.cuda.is_available():
            return False
        return _is_package_available("causal_conv1d")
    return False


def is_mambapy_available():
    if is_torch_available():
        return _is_package_available("mambapy")
    return False


def is_torch_mps_available(min_version: Optional[str] = None):
    if is_torch_available():
        import torch

        if hasattr(torch.backends, "mps"):
            backend_available = torch.backends.mps.is_available() and torch.backends.mps.is_built()
            if min_version is not None:
                flag = version.parse(_torch_version) >= version.parse(min_version)
                backend_available = backend_available and flag
            return backend_available
    return False


def is_torch_bf16_gpu_available():
    if not is_torch_available():
        return False

    import torch

    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def is_torch_bf16_cpu_available():
    if not is_torch_available():
        return False

    import torch

    try:
        # multiple levels of AttributeError depending on the pytorch version so do them all in one check
        _ = torch.cpu.amp.autocast
    except AttributeError:
        return False

    return True


def is_torch_bf16_available():
    # the original bf16 check was for gpu only, but later a cpu/bf16 combo has emerged so this util
    # has become ambiguous and therefore deprecated
    warnings.warn(
        "The util is_torch_bf16_available is deprecated, please use is_torch_bf16_gpu_available "
        "or is_torch_bf16_cpu_available instead according to whether it's used with cpu or gpu",
        FutureWarning,
    )
    return is_torch_bf16_gpu_available()


@lru_cache()
def is_torch_fp16_available_on_device(device):
    if not is_torch_available():
        return False

    import torch

    try:
        x = torch.zeros(2, 2, dtype=torch.float16).to(device)
        _ = x @ x

        # At this moment, let's be strict of the check: check if `LayerNorm` is also supported on device, because many
        # models use this layer.
        batch, sentence_length, embedding_dim = 3, 4, 5
        embedding = torch.randn(batch, sentence_length, embedding_dim, dtype=torch.float16, device=device)
        layer_norm = torch.nn.LayerNorm(embedding_dim, dtype=torch.float16, device=device)
        _ = layer_norm(embedding)

    except:  # noqa: E722
        # TODO: more precise exception matching, if possible.
        # most backends should return `RuntimeError` however this is not guaranteed.
        return False

    return True


@lru_cache()
def is_torch_bf16_available_on_device(device):
    if not is_torch_available():
        return False

    import torch

    if device == "cuda":
        return is_torch_bf16_gpu_available()

    try:
        x = torch.zeros(2, 2, dtype=torch.bfloat16).to(device)
        _ = x @ x
    except:  # noqa: E722
        # TODO: more precise exception matching, if possible.
        # most backends should return `RuntimeError` however this is not guaranteed.
        return False

    return True


def is_torch_tf32_available():
    if not is_torch_available():
        return False

    import torch

    if not torch.cuda.is_available() or torch.version.cuda is None:
        return False
    if torch.cuda.get_device_properties(torch.cuda.current_device()).major < 8:
        return False
    if int(torch.version.cuda.split(".")[0]) < 11:
        return False
    if version.parse(version.parse(torch.__version__).base_version) < version.parse("1.7"):
        return False

    return True


def is_torch_fx_available():
    return _torch_fx_available


def is_peft_available():
    return _peft_available


def is_bs4_available():
    return _bs4_available


def is_tf_available():
    return _tf_available


def is_coloredlogs_available():
    return _coloredlogs_available


def is_tf2onnx_available():
    return _tf2onnx_available


def is_onnx_available():
    return _onnx_available


def is_openai_available():
    return _openai_available


def is_flax_available():
    return _flax_available


def is_flute_available():
    try:
        return importlib.util.find_spec("flute") is not None and importlib.metadata.version("flute-kernel") >= "0.3.0"
    except importlib.metadata.PackageNotFoundError:
        return False


def is_ftfy_available():
    return _ftfy_available


def is_g2p_en_available():
    return _g2p_en_available


@lru_cache()
def is_torch_tpu_available(check_device=True):
    "Checks if `torch_xla` is installed and potentially if a TPU is in the environment"
    warnings.warn(
        "`is_torch_tpu_available` is deprecated and will be removed in 4.41.0. "
        "Please use the `is_torch_xla_available` instead.",
        FutureWarning,
    )

    if not _torch_available:
        return False
    if importlib.util.find_spec("torch_xla") is not None:
        if check_device:
            # We need to check if `xla_device` can be found, will raise a RuntimeError if not
            try:
                import torch_xla.core.xla_model as xm

                _ = xm.xla_device()
                return True
            except RuntimeError:
                return False
        return True
    return False


@lru_cache
def is_torch_xla_available(check_is_tpu=False, check_is_gpu=False):
    """
    Check if `torch_xla` is available. To train a native pytorch job in an environment with torch xla installed, set
    the USE_TORCH_XLA to false.
    """
    assert not (check_is_tpu and check_is_gpu), "The check_is_tpu and check_is_gpu cannot both be true."

    if not _torch_xla_available:
        return False

    import torch_xla

    if check_is_gpu:
        return torch_xla.runtime.device_type() in ["GPU", "CUDA"]
    elif check_is_tpu:
        return torch_xla.runtime.device_type() == "TPU"

    return True


@lru_cache()
def is_torch_neuroncore_available(check_device=True):
    if importlib.util.find_spec("torch_neuronx") is not None:
        return is_torch_xla_available()
    return False


@lru_cache()
def is_torch_npu_available(check_device=False):
    "Checks if `torch_npu` is installed and potentially if a NPU is in the environment"
    if not _torch_available or importlib.util.find_spec("torch_npu") is None:
        return False

    import torch
    import torch_npu  # noqa: F401

    if check_device:
        try:
            # Will raise a RuntimeError if no NPU is found
            _ = torch.npu.device_count()
            return torch.npu.is_available()
        except RuntimeError:
            return False
    return hasattr(torch, "npu") and torch.npu.is_available()


@lru_cache()
def is_torch_mlu_available(check_device=False):
    """
    Checks if `mlu` is available via an `cndev-based` check which won't trigger the drivers and leave mlu
    uninitialized.
    """
    if not _torch_available or importlib.util.find_spec("torch_mlu") is None:
        return False

    import torch
    import torch_mlu  # noqa: F401

    pytorch_cndev_based_mlu_check_previous_value = os.environ.get("PYTORCH_CNDEV_BASED_MLU_CHECK")
    try:
        os.environ["PYTORCH_CNDEV_BASED_MLU_CHECK"] = str(1)
        available = torch.mlu.is_available()
    finally:
        if pytorch_cndev_based_mlu_check_previous_value:
            os.environ["PYTORCH_CNDEV_BASED_MLU_CHECK"] = pytorch_cndev_based_mlu_check_previous_value
        else:
            os.environ.pop("PYTORCH_CNDEV_BASED_MLU_CHECK", None)

    return available


@lru_cache()
def is_torch_musa_available(check_device=False):
    "Checks if `torch_musa` is installed and potentially if a MUSA is in the environment"
    if not _torch_available or importlib.util.find_spec("torch_musa") is None:
        return False

    import torch
    import torch_musa  # noqa: F401

    torch_musa_min_version = "0.33.0"
    if _accelerate_available and version.parse(_accelerate_version) < version.parse(torch_musa_min_version):
        return False

    if check_device:
        try:
            # Will raise a RuntimeError if no MUSA is found
            _ = torch.musa.device_count()
            return torch.musa.is_available()
        except RuntimeError:
            return False
    return hasattr(torch, "musa") and torch.musa.is_available()


def is_torchdynamo_available():
    if not is_torch_available():
        return False

    return version.parse(_torch_version) >= version.parse("2.0.0")


def is_torch_compile_available():
    if not is_torch_available():
        return False

    import torch

    # We don't do any version check here to support nighlies marked as 1.14. Ultimately needs to check version against
    # 2.0 but let's do it later.
    return hasattr(torch, "compile")


def is_torchdynamo_compiling():
    if not is_torch_available():
        return False

    # Importing torch._dynamo causes issues with PyTorch profiler (https://github.com/pytorch/pytorch/issues/130622)
    # hence rather relying on `torch.compiler.is_compiling()` when possible (torch>=2.3)
    try:
        import torch

        return torch.compiler.is_compiling()
    except Exception:
        try:
            import torch._dynamo as dynamo  # noqa: F401

            return dynamo.is_compiling()
        except Exception:
            return False


def is_torch_tensorrt_fx_available():
    if importlib.util.find_spec("torch_tensorrt") is None:
        return False
    return importlib.util.find_spec("torch_tensorrt.fx") is not None


def is_datasets_available():
    return _datasets_available


def is_detectron2_available():
    return _detectron2_available


def is_rjieba_available():
    return _rjieba_available


def is_psutil_available():
    return _psutil_available


def is_py3nvml_available():
    return _py3nvml_available


def is_sacremoses_available():
    return _sacremoses_available


def is_apex_available():
    return _apex_available


def is_aqlm_available():
    return _aqlm_available


def is_vptq_available(min_version: str = VPTQ_MIN_VERSION):
    return _vptq_available and version.parse(_vptq_version) >= version.parse(min_version)


def is_av_available():
    return _av_available


def is_decord_available():
    return _decord_available


def is_ninja_available():
    r"""
    Code comes from *torch.utils.cpp_extension.is_ninja_available()*. Returns `True` if the
    [ninja](https://ninja-build.org/) build system is available on the system, `False` otherwise.
    """
    try:
        subprocess.check_output("ninja --version".split())
    except Exception:
        return False
    else:
        return True


def is_ipex_available():
    def get_major_and_minor_from_version(full_version):
        return str(version.parse(full_version).major) + "." + str(version.parse(full_version).minor)

    if not is_torch_available() or not _ipex_available:
        return False

    torch_major_and_minor = get_major_and_minor_from_version(_torch_version)
    ipex_major_and_minor = get_major_and_minor_from_version(_ipex_version)
    if torch_major_and_minor != ipex_major_and_minor:
        logger.warning(
            f"Intel Extension for PyTorch {ipex_major_and_minor} needs to work with PyTorch {ipex_major_and_minor}.*,"
            f" but PyTorch {_torch_version} is found. Please switch to the matching version and run again."
        )
        return False
    return True


@lru_cache
def is_torch_xpu_available(check_device=False):
    """
    Checks if XPU acceleration is available either via `intel_extension_for_pytorch` or
    via stock PyTorch (>=2.4) and potentially if a XPU is in the environment
    """
    if not is_torch_available():
        return False

    torch_version = version.parse(_torch_version)
    if is_ipex_available():
        import intel_extension_for_pytorch  # noqa: F401
    elif torch_version.major < 2 or (torch_version.major == 2 and torch_version.minor < 4):
        return False

    import torch

    if check_device:
        try:
            # Will raise a RuntimeError if no XPU  is found
            _ = torch.xpu.device_count()
            return torch.xpu.is_available()
        except RuntimeError:
            return False
    return hasattr(torch, "xpu") and torch.xpu.is_available()


@lru_cache()
def is_bitsandbytes_available():
    if not is_torch_available() or not _bitsandbytes_available:
        return False

    import torch

    # `bitsandbytes` versions older than 0.43.1 eagerly require CUDA at import time,
    # so those versions of the library are practically only available when CUDA is too.
    if version.parse(importlib.metadata.version("bitsandbytes")) < version.parse("0.43.1"):
        return torch.cuda.is_available()

    # Newer versions of `bitsandbytes` can be imported on systems without CUDA.
    return True


def is_bitsandbytes_multi_backend_available() -> bool:
    if not is_bitsandbytes_available():
        return False

    import bitsandbytes as bnb

    return "multi_backend" in getattr(bnb, "features", set())


def is_flash_attn_2_available():
    if not is_torch_available():
        return False

    if not _is_package_available("flash_attn"):
        return False

    # Let's add an extra check to see if cuda is available
    import torch

    if not (torch.cuda.is_available() or is_torch_mlu_available()):
        return False

    if torch.version.cuda:
        return version.parse(importlib.metadata.version("flash_attn")) >= version.parse("2.1.0")
    elif torch.version.hip:
        # TODO: Bump the requirement to 2.1.0 once released in https://github.com/ROCmSoftwarePlatform/flash-attention
        return version.parse(importlib.metadata.version("flash_attn")) >= version.parse("2.0.4")
    elif is_torch_mlu_available():
        return version.parse(importlib.metadata.version("flash_attn")) >= version.parse("2.3.3")
    else:
        return False


@lru_cache()
def is_flash_attn_greater_or_equal_2_10():
    if not _is_package_available("flash_attn"):
        return False

    return version.parse(importlib.metadata.version("flash_attn")) >= version.parse("2.1.0")


@lru_cache()
def is_flash_attn_greater_or_equal(library_version: str):
    if not _is_package_available("flash_attn"):
        return False

    return version.parse(importlib.metadata.version("flash_attn")) >= version.parse(library_version)


@lru_cache()
def is_torch_greater_or_equal(library_version: str):
    if not _is_package_available("torch"):
        return False

    return version.parse(importlib.metadata.version("torch")) >= version.parse(library_version)


def is_torchdistx_available():
    return _torchdistx_available


def is_faiss_available():
    return _faiss_available


def is_scipy_available():
    return _scipy_available


def is_sklearn_available():
    return _sklearn_available


def is_sentencepiece_available():
    return _sentencepiece_available


def is_seqio_available():
    return _is_seqio_available


def is_gguf_available(min_version: str = GGUF_MIN_VERSION):
    return _is_gguf_available and version.parse(_gguf_version) >= version.parse(min_version)


def is_protobuf_available():
    if importlib.util.find_spec("google") is None:
        return False
    return importlib.util.find_spec("google.protobuf") is not None


def is_fsdp_available(min_version: str = FSDP_MIN_VERSION):
    return is_torch_available() and version.parse(_torch_version) >= version.parse(min_version)


def is_optimum_available():
    return _optimum_available


def is_auto_awq_available():
    return _auto_awq_available


def is_optimum_quanto_available():
    # `importlib.metadata.version` doesn't work with `optimum.quanto`, need to put `optimum_quanto`
    return _is_optimum_quanto_available


def is_compressed_tensors_available():
    return _compressed_tensors_available


def is_auto_gptq_available():
    return _auto_gptq_available


def is_gptqmodel_available():
    return _gptqmodel_available


def is_eetq_available():
    return _eetq_available


def is_fbgemm_gpu_available():
    return _fbgemm_gpu_available


def is_levenshtein_available():
    return _levenshtein_available


def is_optimum_neuron_available():
    return _optimum_available and _is_package_available("optimum.neuron")


def is_safetensors_available():
    return _safetensors_available


def is_tokenizers_available():
    return _tokenizers_available


@lru_cache
def is_vision_available():
    _pil_available = importlib.util.find_spec("PIL") is not None
    if _pil_available:
        try:
            package_version = importlib.metadata.version("Pillow")
        except importlib.metadata.PackageNotFoundError:
            try:
                package_version = importlib.metadata.version("Pillow-SIMD")
            except importlib.metadata.PackageNotFoundError:
                return False
        logger.debug(f"Detected PIL version {package_version}")
    return _pil_available


def is_pytesseract_available():
    return _pytesseract_available


def is_pytest_available():
    return _pytest_available


def is_spacy_available():
    return _spacy_available


def is_tensorflow_text_available():
    return is_tf_available() and _tensorflow_text_available


def is_keras_nlp_available():
    return is_tensorflow_text_available() and _keras_nlp_available


def is_in_notebook():
    try:
        # Test adapted from tqdm.autonotebook: https://github.com/tqdm/tqdm/blob/master/tqdm/autonotebook.py
        get_ipython = sys.modules["IPython"].get_ipython
        if "IPKernelApp" not in get_ipython().config:
            raise ImportError("console")
        # Removed the lines to include VSCode
        if "DATABRICKS_RUNTIME_VERSION" in os.environ and os.environ["DATABRICKS_RUNTIME_VERSION"] < "11.0":
            # Databricks Runtime 11.0 and above uses IPython kernel by default so it should be compatible with Jupyter notebook
            # https://docs.microsoft.com/en-us/azure/databricks/notebooks/ipython-kernel
            raise ImportError("databricks")

        return importlib.util.find_spec("IPython") is not None
    except (AttributeError, ImportError, KeyError):
        return False


def is_pytorch_quantization_available():
    return _pytorch_quantization_available


def is_tensorflow_probability_available():
    return _tensorflow_probability_available


def is_pandas_available():
    return _pandas_available


def is_sagemaker_dp_enabled():
    # Get the sagemaker specific env variable.
    sagemaker_params = os.getenv("SM_FRAMEWORK_PARAMS", "{}")
    try:
        # Parse it and check the field "sagemaker_distributed_dataparallel_enabled".
        sagemaker_params = json.loads(sagemaker_params)
        if not sagemaker_params.get("sagemaker_distributed_dataparallel_enabled", False):
            return False
    except json.JSONDecodeError:
        return False
    # Lastly, check if the `smdistributed` module is present.
    return _smdistributed_available


def is_sagemaker_mp_enabled():
    # Get the sagemaker specific mp parameters from smp_options variable.
    smp_options = os.getenv("SM_HP_MP_PARAMETERS", "{}")
    try:
        # Parse it and check the field "partitions" is included, it is required for model parallel.
        smp_options = json.loads(smp_options)
        if "partitions" not in smp_options:
            return False
    except json.JSONDecodeError:
        return False

    # Get the sagemaker specific framework parameters from mpi_options variable.
    mpi_options = os.getenv("SM_FRAMEWORK_PARAMS", "{}")
    try:
        # Parse it and check the field "sagemaker_distributed_dataparallel_enabled".
        mpi_options = json.loads(mpi_options)
        if not mpi_options.get("sagemaker_mpi_enabled", False):
            return False
    except json.JSONDecodeError:
        return False
    # Lastly, check if the `smdistributed` module is present.
    return _smdistributed_available


def is_training_run_on_sagemaker():
    return "SAGEMAKER_JOB_NAME" in os.environ


def is_soundfile_available():
    return _soundfile_available


def is_timm_available():
    return _timm_available


def is_natten_available():
    return _natten_available


def is_nltk_available():
    return _nltk_available


def is_torchaudio_available():
    return _torchaudio_available


def is_torchao_available():
    return _torchao_available


def is_speech_available():
    # For now this depends on torchaudio but the exact dependency might evolve in the future.
    return _torchaudio_available


def is_spqr_available():
    return _spqr_available


def is_phonemizer_available():
    return _phonemizer_available


def is_uroman_available():
    return _uroman_available


def torch_only_method(fn):
    def wrapper(*args, **kwargs):
        if not _torch_available:
            raise ImportError(
                "You need to install pytorch to use this method or class, "
                "or activate it with environment variables USE_TORCH=1 and USE_TF=0."
            )
        else:
            return fn(*args, **kwargs)

    return wrapper


def is_ccl_available():
    return _is_ccl_available


def is_sudachi_available():
    return _sudachipy_available


def get_sudachi_version():
    return _sudachipy_version


def is_sudachi_projection_available():
    if not is_sudachi_available():
        return False

    # NOTE: We require sudachipy>=0.6.8 to use projection option in sudachi_kwargs for the constructor of BertJapaneseTokenizer.
    # - `projection` option is not supported in sudachipy<0.6.8, see https://github.com/WorksApplications/sudachi.rs/issues/230
    return version.parse(_sudachipy_version) >= version.parse("0.6.8")


def is_jumanpp_available():
    return (importlib.util.find_spec("rhoknp") is not None) and (shutil.which("jumanpp") is not None)


def is_cython_available():
    return importlib.util.find_spec("pyximport") is not None


def is_jieba_available():
    return _jieba_available


def is_jinja_available():
    return _jinja_available


def is_mlx_available():
    return _mlx_available


def is_tiktoken_available():
    return _tiktoken_available and _blobfile_available


def is_liger_kernel_available():
    if not _liger_kernel_available:
        return False

    return version.parse(importlib.metadata.version("liger_kernel")) >= version.parse("0.3.0")


def is_triton_available():
    return _triton_available


# docstyle-ignore
AV_IMPORT_ERROR = """
{0} requires the PyAv library but it was not found in your environment. You can install it with:
```
pip install av
```
Please note that you may need to restart your runtime after installation.
"""

# docstyle-ignore
YT_DLP_IMPORT_ERROR = """
{0} requires the YT-DLP library but it was not found in your environment. You can install it with:
```
pip install yt-dlp
```
Please note that you may need to restart your runtime after installation.
"""

DECORD_IMPORT_ERROR = """
{0} requires the PyAv library but it was not found in your environment. You can install it with:
```
pip install decord
```
Please note that you may need to restart your runtime after installation.
"""

# docstyle-ignore
CV2_IMPORT_ERROR = """
{0} requires the OpenCV library but it was not found in your environment. You can install it with:
```
pip install opencv-python
```
Please note that you may need to restart your runtime after installation.
"""


# docstyle-ignore
DATASETS_IMPORT_ERROR = """
{0} requires the 🤗 Datasets library but it was not found in your environment. You can install it with:
```
pip install datasets
```
In a notebook or a colab, you can install it by executing a cell with
```
!pip install datasets
```
then restarting your kernel.

Note that if you have a local folder named `datasets` or a local python file named `datasets.py` in your current
working directory, python may try to import this instead of the 🤗 Datasets library. You should rename this folder or
that python file if that's the case. Please note that you may need to restart your runtime after installation.
"""


# docstyle-ignore
TOKENIZERS_IMPORT_ERROR = """
{0} requires the 🤗 Tokenizers library but it was not found in your environment. You can install it with:
```
pip install tokenizers
```
In a notebook or a colab, you can install it by executing a cell with
```
!pip install tokenizers
```
Please note that you may need to restart your runtime after installation.
"""


# docstyle-ignore
SENTENCEPIECE_IMPORT_ERROR = """
{0} requires the SentencePiece library but it was not found in your environment. Checkout the instructions on the
installation page of its repo: https://github.com/google/sentencepiece#installation and follow the ones
that match your environment. Please note that you may need to restart your runtime after installation.
"""


# docstyle-ignore
PROTOBUF_IMPORT_ERROR = """
{0} requires the protobuf library but it was not found in your environment. Checkout the instructions on the
installation page of its repo: https://github.com/protocolbuffers/protobuf/tree/master/python#installation and follow the ones
that match your environment. Please note that you may need to restart your runtime after installation.
"""


# docstyle-ignore
FAISS_IMPORT_ERROR = """
{0} requires the faiss library but it was not found in your environment. Checkout the instructions on the
installation page of its repo: https://github.com/facebookresearch/faiss/blob/master/INSTALL.md and follow the ones
that match your environment. Please note that you may need to restart your runtime after installation.
"""


# docstyle-ignore
PYTORCH_IMPORT_ERROR = """
{0} requires the PyTorch library but it was not found in your environment. Checkout the instructions on the
installation page: https://pytorch.org/get-started/locally/ and follow the ones that match your environment.
Please note that you may need to restart your runtime after installation.
"""


# docstyle-ignore
TORCHVISION_IMPORT_ERROR = """
{0} requires the Torchvision library but it was not found in your environment. Checkout the instructions on the
installation page: https://pytorch.org/get-started/locally/ and follow the ones that match your environment.
Please note that you may need to restart your runtime after installation.
"""

# docstyle-ignore
PYTORCH_IMPORT_ERROR_WITH_TF = """
{0} requires the PyTorch library but it was not found in your environment.
However, we were able to find a TensorFlow installation. TensorFlow classes begin
with "TF", but are otherwise identically named to our PyTorch classes. This
means that the TF equivalent of the class you tried to import would be "TF{0}".
If you want to use TensorFlow, please use TF classes instead!

If you really do want to use PyTorch please go to
https://pytorch.org/get-started/locally/ and follow the instructions that
match your environment.
"""

# docstyle-ignore
TF_IMPORT_ERROR_WITH_PYTORCH = """
{0} requires the TensorFlow library but it was not found in your environment.
However, we were able to find a PyTorch installation. PyTorch classes do not begin
with "TF", but are otherwise identically named to our TF classes.
If you want to use PyTorch, please use those classes instead!

If you really do want to use TensorFlow, please follow the instructions on the
installation page https://www.tensorflow.org/install that match your environment.
"""

# docstyle-ignore
BS4_IMPORT_ERROR = """
{0} requires the Beautiful Soup library but it was not found in your environment. You can install it with pip:
`pip install beautifulsoup4`. Please note that you may need to restart your runtime after installation.
"""


# docstyle-ignore
SKLEARN_IMPORT_ERROR = """
{0} requires the scikit-learn library but it was not found in your environment. You can install it with:
```
pip install -U scikit-learn
```
In a notebook or a colab, you can install it by executing a cell with
```
!pip install -U scikit-learn
```
Please note that you may need to restart your runtime after installation.
"""


# docstyle-ignore
TENSORFLOW_IMPORT_ERROR = """
{0} requires the TensorFlow library but it was not found in your environment. Checkout the instructions on the
installation page: https://www.tensorflow.org/install and follow the ones that match your environment.
Please note that you may need to restart your runtime after installation.
"""


# docstyle-ignore
DETECTRON2_IMPORT_ERROR = """
{0} requires the detectron2 library but it was not found in your environment. Checkout the instructions on the
installation page: https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md and follow the ones
that match your environment. Please note that you may need to restart your runtime after installation.
"""


# docstyle-ignore
FLAX_IMPORT_ERROR = """
{0} requires the FLAX library but it was not found in your environment. Checkout the instructions on the
installation page: https://github.com/google/flax and follow the ones that match your environment.
Please note that you may need to restart your runtime after installation.
"""

# docstyle-ignore
FTFY_IMPORT_ERROR = """
{0} requires the ftfy library but it was not found in your environment. Checkout the instructions on the
installation section: https://github.com/rspeer/python-ftfy/tree/master#installing and follow the ones
that match your environment. Please note that you may need to restart your runtime after installation.
"""

LEVENSHTEIN_IMPORT_ERROR = """
{0} requires the python-Levenshtein library but it was not found in your environment. You can install it with pip: `pip
install python-Levenshtein`. Please note that you may need to restart your runtime after installation.
"""

# docstyle-ignore
G2P_EN_IMPORT_ERROR = """
{0} requires the g2p-en library but it was not found in your environment. You can install it with pip:
`pip install g2p-en`. Please note that you may need to restart your runtime after installation.
"""

# docstyle-ignore
PYTORCH_QUANTIZATION_IMPORT_ERROR = """
{0} requires the pytorch-quantization library but it was not found in your environment. You can install it with pip:
`pip install pytorch-quantization --extra-index-url https://pypi.ngc.nvidia.com`
Please note that you may need to restart your runtime after installation.
"""

# docstyle-ignore
TENSORFLOW_PROBABILITY_IMPORT_ERROR = """
{0} requires the tensorflow_probability library but it was not found in your environment. You can install it with pip as
explained here: https://github.com/tensorflow/probability. Please note that you may need to restart your runtime after installation.
"""

# docstyle-ignore
TENSORFLOW_TEXT_IMPORT_ERROR = """
{0} requires the tensorflow_text library but it was not found in your environment. You can install it with pip as
explained here: https://www.tensorflow.org/text/guide/tf_text_intro.
Please note that you may need to restart your runtime after installation.
"""

# docstyle-ignore
TORCHAUDIO_IMPORT_ERROR = """
{0} requires the torchaudio library but it was not found in your environment. Please install it and restart your
runtime.
"""

# docstyle-ignore
PANDAS_IMPORT_ERROR = """
{0} requires the pandas library but it was not found in your environment. You can install it with pip as
explained here: https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html.
Please note that you may need to restart your runtime after installation.
"""


# docstyle-ignore
PHONEMIZER_IMPORT_ERROR = """
{0} requires the phonemizer library but it was not found in your environment. You can install it with pip:
`pip install phonemizer`. Please note that you may need to restart your runtime after installation.
"""
# docstyle-ignore
UROMAN_IMPORT_ERROR = """
{0} requires the uroman library but it was not found in your environment. You can install it with pip:
`pip install uroman`. Please note that you may need to restart your runtime after installation.
"""


# docstyle-ignore
SACREMOSES_IMPORT_ERROR = """
{0} requires the sacremoses library but it was not found in your environment. You can install it with pip:
`pip install sacremoses`. Please note that you may need to restart your runtime after installation.
"""

# docstyle-ignore
SCIPY_IMPORT_ERROR = """
{0} requires the scipy library but it was not found in your environment. You can install it with pip:
`pip install scipy`. Please note that you may need to restart your runtime after installation.
"""


# docstyle-ignore
SPEECH_IMPORT_ERROR = """
{0} requires the torchaudio library but it was not found in your environment. You can install it with pip:
`pip install torchaudio`. Please note that you may need to restart your runtime after installation.
"""

# docstyle-ignore
TIMM_IMPORT_ERROR = """
{0} requires the timm library but it was not found in your environment. You can install it with pip:
`pip install timm`. Please note that you may need to restart your runtime after installation.
"""

# docstyle-ignore
NATTEN_IMPORT_ERROR = """
{0} requires the natten library but it was not found in your environment. You can install it by referring to:
shi-labs.com/natten . You can also install it with pip (may take longer to build):
`pip install natten`. Please note that you may need to restart your runtime after installation.
"""

NUMEXPR_IMPORT_ERROR = """
{0} requires the numexpr library but it was not found in your environment. You can install it by referring to:
https://numexpr.readthedocs.io/en/latest/index.html.
"""


# docstyle-ignore
NLTK_IMPORT_ERROR = """
{0} requires the NLTK library but it was not found in your environment. You can install it by referring to:
https://www.nltk.org/install.html. Please note that you may need to restart your runtime after installation.
"""


# docstyle-ignore
VISION_IMPORT_ERROR = """
{0} requires the PIL library but it was not found in your environment. You can install it with pip:
`pip install pillow`. Please note that you may need to restart your runtime after installation.
"""


# docstyle-ignore
PYTESSERACT_IMPORT_ERROR = """
{0} requires the PyTesseract library but it was not found in your environment. You can install it with pip:
`pip install pytesseract`. Please note that you may need to restart your runtime after installation.
"""

# docstyle-ignore
PYCTCDECODE_IMPORT_ERROR = """
{0} requires the pyctcdecode library but it was not found in your environment. You can install it with pip:
`pip install pyctcdecode`. Please note that you may need to restart your runtime after installation.
"""

# docstyle-ignore
ACCELERATE_IMPORT_ERROR = """
{0} requires the accelerate library >= {ACCELERATE_MIN_VERSION} it was not found in your environment.
You can install or update it with pip: `pip install --upgrade accelerate`. Please note that you may need to restart your
runtime after installation.
"""

# docstyle-ignore
CCL_IMPORT_ERROR = """
{0} requires the torch ccl library but it was not found in your environment. You can install it with pip:
`pip install oneccl_bind_pt -f https://developer.intel.com/ipex-whl-stable`
Please note that you may need to restart your runtime after installation.
"""

# docstyle-ignore
ESSENTIA_IMPORT_ERROR = """
{0} requires essentia library. But that was not found in your environment. You can install them with pip:
`pip install essentia==2.1b6.dev1034`
Please note that you may need to restart your runtime after installation.
"""

# docstyle-ignore
LIBROSA_IMPORT_ERROR = """
{0} requires thes librosa library. But that was not found in your environment. You can install them with pip:
`pip install librosa`
Please note that you may need to restart your runtime after installation.
"""

# docstyle-ignore
PRETTY_MIDI_IMPORT_ERROR = """
{0} requires thes pretty_midi library. But that was not found in your environment. You can install them with pip:
`pip install pretty_midi`
Please note that you may need to restart your runtime after installation.
"""


CYTHON_IMPORT_ERROR = """
{0} requires the Cython library but it was not found in your environment. You can install it with pip: `pip install
Cython`. Please note that you may need to restart your runtime after installation.
"""

JIEBA_IMPORT_ERROR = """
{0} requires the jieba library but it was not found in your environment. You can install it with pip: `pip install
jieba`. Please note that you may need to restart your runtime after installation.
"""

PEFT_IMPORT_ERROR = """
{0} requires the peft library but it was not found in your environment. You can install it with pip: `pip install
peft`. Please note that you may need to restart your runtime after installation.
"""

JINJA_IMPORT_ERROR = """
{0} requires the jinja library but it was not found in your environment. You can install it with pip: `pip install
jinja2`. Please note that you may need to restart your runtime after installation.
"""

BACKENDS_MAPPING = OrderedDict(
    [
        ("av", (is_av_available, AV_IMPORT_ERROR)),
        ("bs4", (is_bs4_available, BS4_IMPORT_ERROR)),
        ("cv2", (is_cv2_available, CV2_IMPORT_ERROR)),
        ("datasets", (is_datasets_available, DATASETS_IMPORT_ERROR)),
        ("decord", (is_decord_available, DECORD_IMPORT_ERROR)),
        ("detectron2", (is_detectron2_available, DETECTRON2_IMPORT_ERROR)),
        ("essentia", (is_essentia_available, ESSENTIA_IMPORT_ERROR)),
        ("faiss", (is_faiss_available, FAISS_IMPORT_ERROR)),
        ("flax", (is_flax_available, FLAX_IMPORT_ERROR)),
        ("ftfy", (is_ftfy_available, FTFY_IMPORT_ERROR)),
        ("g2p_en", (is_g2p_en_available, G2P_EN_IMPORT_ERROR)),
        ("pandas", (is_pandas_available, PANDAS_IMPORT_ERROR)),
        ("phonemizer", (is_phonemizer_available, PHONEMIZER_IMPORT_ERROR)),
        ("uroman", (is_uroman_available, UROMAN_IMPORT_ERROR)),
        ("pretty_midi", (is_pretty_midi_available, PRETTY_MIDI_IMPORT_ERROR)),
        ("levenshtein", (is_levenshtein_available, LEVENSHTEIN_IMPORT_ERROR)),
        ("librosa", (is_librosa_available, LIBROSA_IMPORT_ERROR)),
        ("protobuf", (is_protobuf_available, PROTOBUF_IMPORT_ERROR)),
        ("pyctcdecode", (is_pyctcdecode_available, PYCTCDECODE_IMPORT_ERROR)),
        ("pytesseract", (is_pytesseract_available, PYTESSERACT_IMPORT_ERROR)),
        ("sacremoses", (is_sacremoses_available, SACREMOSES_IMPORT_ERROR)),
        ("pytorch_quantization", (is_pytorch_quantization_available, PYTORCH_QUANTIZATION_IMPORT_ERROR)),
        ("sentencepiece", (is_sentencepiece_available, SENTENCEPIECE_IMPORT_ERROR)),
        ("sklearn", (is_sklearn_available, SKLEARN_IMPORT_ERROR)),
        ("speech", (is_speech_available, SPEECH_IMPORT_ERROR)),
        ("tensorflow_probability", (is_tensorflow_probability_available, TENSORFLOW_PROBABILITY_IMPORT_ERROR)),
        ("tf", (is_tf_available, TENSORFLOW_IMPORT_ERROR)),
        ("tensorflow_text", (is_tensorflow_text_available, TENSORFLOW_TEXT_IMPORT_ERROR)),
        ("timm", (is_timm_available, TIMM_IMPORT_ERROR)),
        ("torchaudio", (is_torchaudio_available, TORCHAUDIO_IMPORT_ERROR)),
        ("natten", (is_natten_available, NATTEN_IMPORT_ERROR)),
        ("nltk", (is_nltk_available, NLTK_IMPORT_ERROR)),
        ("tokenizers", (is_tokenizers_available, TOKENIZERS_IMPORT_ERROR)),
        ("torch", (is_torch_available, PYTORCH_IMPORT_ERROR)),
        ("torchvision", (is_torchvision_available, TORCHVISION_IMPORT_ERROR)),
        ("vision", (is_vision_available, VISION_IMPORT_ERROR)),
        ("scipy", (is_scipy_available, SCIPY_IMPORT_ERROR)),
        ("accelerate", (is_accelerate_available, ACCELERATE_IMPORT_ERROR)),
        ("oneccl_bind_pt", (is_ccl_available, CCL_IMPORT_ERROR)),
        ("cython", (is_cython_available, CYTHON_IMPORT_ERROR)),
        ("jieba", (is_jieba_available, JIEBA_IMPORT_ERROR)),
        ("peft", (is_peft_available, PEFT_IMPORT_ERROR)),
        ("jinja", (is_jinja_available, JINJA_IMPORT_ERROR)),
        ("yt_dlp", (is_yt_dlp_available, YT_DLP_IMPORT_ERROR)),
    ]
)


def requires_backends(obj, backends):
    if not isinstance(backends, (list, tuple)):
        backends = [backends]

    name = obj.__name__ if hasattr(obj, "__name__") else obj.__class__.__name__

    # Raise an error for users who might not realize that classes without "TF" are torch-only
    if "torch" in backends and "tf" not in backends and not is_torch_available() and is_tf_available():
        raise ImportError(PYTORCH_IMPORT_ERROR_WITH_TF.format(name))

    # Raise the inverse error for PyTorch users trying to load TF classes
    if "tf" in backends and "torch" not in backends and is_torch_available() and not is_tf_available():
        raise ImportError(TF_IMPORT_ERROR_WITH_PYTORCH.format(name))

    checks = (BACKENDS_MAPPING[backend] for backend in backends)
    failed = [msg.format(name) for available, msg in checks if not available()]
    if failed:
        raise ImportError("".join(failed))


class DummyObject(type):
    """
    Metaclass for the dummy objects. Any class inheriting from it will return the ImportError generated by
    `requires_backend` each time a user tries to access any method of that class.
    """

    def __getattribute__(cls, key):
        if key.startswith("_") and key != "_from_config":
            return super().__getattribute__(key)
        requires_backends(cls, cls._backends)


def is_torch_fx_proxy(x):
    if is_torch_fx_available():
        import torch.fx

        return isinstance(x, torch.fx.Proxy)
    return False


BACKENDS_T = FrozenSet[str]
IMPORT_STRUCTURE_T = Dict[BACKENDS_T, Dict[str, Set[str]]]


class _LazyModule(ModuleType):
    """
    Module class that surfaces all objects but only performs associated imports when the objects are requested.
    """

    # Very heavily inspired by optuna.integration._IntegrationModule
    # https://github.com/optuna/optuna/blob/master/optuna/integration/__init__.py
    def __init__(
        self,
        name: str,
        module_file: str,
        import_structure: IMPORT_STRUCTURE_T,
        module_spec: importlib.machinery.ModuleSpec = None,
        extra_objects: Dict[str, object] = None,
    ):
        super().__init__(name)

        self._object_missing_backend = {}
        if any(isinstance(key, frozenset) for key in import_structure.keys()):
            self._modules = set()
            self._class_to_module = {}
            self.__all__ = []

            _import_structure = {}

            for backends, module in import_structure.items():
                missing_backends = []
                for backend in backends:
                    if backend not in BACKENDS_MAPPING:
                        raise ValueError(
                            f"Error: the following backend: '{backend}' was specified around object {module} but isn't specified in the backends mapping."
                        )
                    callable, error = BACKENDS_MAPPING[backend]
                    if not callable():
                        missing_backends.append(backend)
                self._modules = self._modules.union(set(module.keys()))

                for key, values in module.items():
                    if len(missing_backends):
                        self._object_missing_backend[key] = missing_backends

                    for value in values:
                        self._class_to_module[value] = key
                        if len(missing_backends):
                            self._object_missing_backend[value] = missing_backends
                    _import_structure.setdefault(key, []).extend(values)

                # Needed for autocompletion in an IDE
                self.__all__.extend(list(module.keys()) + list(chain(*module.values())))

            self.__file__ = module_file
            self.__spec__ = module_spec
            self.__path__ = [os.path.dirname(module_file)]
            self._objects = {} if extra_objects is None else extra_objects
            self._name = name
            self._import_structure = _import_structure

        # This can be removed once every exportable object has a `export()` export.
        else:
            self._modules = set(import_structure.keys())
            self._class_to_module = {}
            for key, values in import_structure.items():
                for value in values:
                    self._class_to_module[value] = key
            # Needed for autocompletion in an IDE
            self.__all__ = list(import_structure.keys()) + list(chain(*import_structure.values()))
            self.__file__ = module_file
            self.__spec__ = module_spec
            self.__path__ = [os.path.dirname(module_file)]
            self._objects = {} if extra_objects is None else extra_objects
            self._name = name
            self._import_structure = import_structure

    # Needed for autocompletion in an IDE
    def __dir__(self):
        result = super().__dir__()
        # The elements of self.__all__ that are submodules may or may not be in the dir already, depending on whether
        # they have been accessed or not. So we only add the elements of self.__all__ that are not already in the dir.
        for attr in self.__all__:
            if attr not in result:
                result.append(attr)
        return result

    def __getattr__(self, name: str) -> Any:
        if name in self._objects:
            return self._objects[name]
        if name in self._object_missing_backend.keys():
            missing_backends = self._object_missing_backend[name]

            class Placeholder(metaclass=DummyObject):
                _backends = missing_backends

                def __init__(self, *args, **kwargs):
                    requires_backends(self, missing_backends)

            Placeholder.__name__ = name
            Placeholder.__module__ = self.__spec__

            value = Placeholder
        elif name in self._class_to_module.keys():
            module = self._get_module(self._class_to_module[name])
            value = getattr(module, name)
        elif name in self._modules:
            value = self._get_module(name)
        else:
            raise AttributeError(f"module {self.__name__} has no attribute {name}")

        setattr(self, name, value)
        return value

    def _get_module(self, module_name: str):
        try:
            return importlib.import_module("." + module_name, self.__name__)
        except Exception as e:
            raise RuntimeError(
                f"Failed to import {self.__name__}.{module_name} because of the following error (look up to see its"
                f" traceback):\n{e}"
            ) from e

    def __reduce__(self):
        return (self.__class__, (self._name, self.__file__, self._import_structure))


class OptionalDependencyNotAvailable(BaseException):
    """Internally used error class for signalling an optional dependency was not found."""


def direct_transformers_import(path: str, file="__init__.py") -> ModuleType:
    """Imports transformers directly

    Args:
        path (`str`): The path to the source file
        file (`str`, *optional*): The file to join with the path. Defaults to "__init__.py".

    Returns:
        `ModuleType`: The resulting imported module
    """
    name = "transformers"
    location = os.path.join(path, file)
    spec = importlib.util.spec_from_file_location(name, location, submodule_search_locations=[path])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module = sys.modules[name]
    return module


def export(*, backends=()):
    """
    This decorator enables two things:
    - Attaching a `__backends` tuple to an object to see what are the necessary backends for it
      to execute correctly without instantiating it
    - The '@export' string is used to dynamically import objects
    """
    for backend in backends:
        if backend not in BACKENDS_MAPPING:
            raise ValueError(f"Backend should be defined in the BACKENDS_MAPPING. Offending backend: {backend}")

    if not isinstance(backends, tuple):
        raise ValueError("Backends should be a tuple.")

    def inner_fn(fun):
        fun.__backends = backends
        return fun

    return inner_fn


BASE_FILE_REQUIREMENTS = {
    lambda e: "modeling_tf_" in e: ("tf",),
    lambda e: "modeling_flax_" in e: ("flax",),
    lambda e: "modeling_" in e: ("torch",),
    lambda e: e.startswith("tokenization_") and e.endswith("_fast"): ("tokenizers",),
}


def fetch__all__(file_content):
    """
    Returns the content of the __all__ variable in the file content.
    Returns None if not defined, otherwise returns a list of strings.
    """

    if "__all__" not in file_content:
        return []

    lines = file_content.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("__all__"):
            start_index = index

    lines = lines[start_index:]

    if not lines[0].startswith("__all__"):
        raise ValueError(
            "fetch__all__ accepts a list of lines, with the first line being the __all__ variable declaration"
        )

    # __all__ is defined on a single line
    if lines[0].endswith("]"):
        return [obj.strip("\"' ") for obj in lines[0].split("=")[1].strip(" []").split(",")]

    # __all__ is defined on multiple lines
    else:
        _all = []
        for __all__line_index in range(1, len(lines)):
            if lines[__all__line_index].strip() == "]":
                return _all
            else:
                _all.append(lines[__all__line_index].strip("\"', "))

        return _all


@lru_cache()
def create_import_structure_from_path(module_path):
    """
    This method takes the path to a file/a folder and returns the import structure.
    If a file is given, it will return the import structure of the parent folder.

    Import structures are designed to be digestible by `_LazyModule` objects. They are
    created from the __all__ definitions in each files as well as the `@export` decorators
    above methods and objects.

    The import structure allows explicit display of the required backends for a given object.
    These backends are specified in two ways:

    1. Through their `@export`, if they are exported with that decorator. This `@export` decorator
       accepts a `backend` tuple kwarg mentioning which backends are required to run this object.

    2. If an object is defined in a file with "default" backends, it will have, at a minimum, this
       backend specified. The default backends are defined according to the filename:

       - If a file is named like `modeling_*.py`, it will have a `torch` backend
       - If a file is named like `modeling_tf_*.py`, it will have a `tf` backend
       - If a file is named like `modeling_flax_*.py`, it will have a `flax` backend
       - If a file is named like `tokenization_*_fast.py`, it will have a `tokenizers` backend

    Backends serve the purpose of displaying a clear error message to the user in case the backends are not installed.
    Should an object be imported without its required backends being in the environment, any attempt to use the
    object will raise an error mentioning which backend(s) should be added to the environment in order to use
    that object.

    Here's an example of an input import structure at the src.transformers.models level:

    {
        'albert': {
            frozenset(): {
                'configuration_albert': {'AlbertConfig', 'AlbertOnnxConfig'}
            },
            frozenset({'tokenizers'}): {
                'tokenization_albert_fast': {'AlbertTokenizerFast'}
            },
        },
        'align': {
            frozenset(): {
                'configuration_align': {'AlignConfig', 'AlignTextConfig', 'AlignVisionConfig'},
                'processing_align': {'AlignProcessor'}
            },
        },
        'altclip': {
            frozenset(): {
                'configuration_altclip': {'AltCLIPConfig', 'AltCLIPTextConfig', 'AltCLIPVisionConfig'},
                'processing_altclip': {'AltCLIPProcessor'},
            }
        }
    }
    """
    import_structure = {}
    if os.path.isdir(module_path):
        directory = module_path
        adjacent_modules = []

        for f in os.listdir(module_path):
            if f != "__pycache__" and os.path.isdir(os.path.join(module_path, f)):
                import_structure[f] = create_import_structure_from_path(os.path.join(module_path, f))

            elif not os.path.isdir(os.path.join(directory, f)):
                adjacent_modules.append(f)

    else:
        directory = os.path.dirname(module_path)
        adjacent_modules = [f for f in os.listdir(directory) if not os.path.isdir(os.path.join(directory, f))]

    # We're only taking a look at files different from __init__.py
    # We could theoretically export things directly from the __init__.py
    # files, but this is not supported at this time.
    if "__init__.py" in adjacent_modules:
        adjacent_modules.remove("__init__.py")

    # Modular files should not be imported
    def find_substring(substring, list_):
        return any(substring in x for x in list_)

    if find_substring("modular_", adjacent_modules) and find_substring("modeling_", adjacent_modules):
        adjacent_modules = [module for module in adjacent_modules if "modular_" not in module]

    module_requirements = {}
    for module_name in adjacent_modules:
        # Only modules ending in `.py` are accepted here.
        if not module_name.endswith(".py"):
            continue

        with open(os.path.join(directory, module_name), encoding="utf-8") as f:
            file_content = f.read()

        # Remove the .py suffix
        module_name = module_name[:-3]

        previous_line = ""
        previous_index = 0

        # Some files have some requirements by default.
        # For example, any file named `modeling_tf_xxx.py`
        # should have TensorFlow as a required backend.
        base_requirements = ()
        for string_check, requirements in BASE_FILE_REQUIREMENTS.items():
            if string_check(module_name):
                base_requirements = requirements
                break

        # Objects that have a `@export` assigned to them will get exported
        # with the backends specified in the decorator as well as the file backends.
        exported_objects = set()
        if "@export" in file_content:
            lines = file_content.split("\n")
            for index, line in enumerate(lines):
                # This allows exporting items with other decorators. We'll take a look
                # at the line that follows at the same indentation level.
                if line.startswith((" ", "\t", "@", ")")) and not line.startswith("@export"):
                    continue

                # Skipping line enables putting whatever we want between the
                # export() call and the actual class/method definition.
                # This is what enables having # Copied from statements, docs, etc.
                skip_line = False

                if "@export" in previous_line:
                    skip_line = False

                    # Backends are defined on the same line as export
                    if "backends" in previous_line:
                        backends_string = previous_line.split("backends=")[1].split("(")[1].split(")")[0]
                        backends = tuple(sorted([b.strip("'\",") for b in backends_string.split(", ") if b]))

                    # Backends are defined in the lines following export, for example such as:
                    # @export(
                    #     backends=(
                    #             "sentencepiece",
                    #             "torch",
                    #             "tf",
                    #     )
                    # )
                    #
                    # or
                    #
                    # @export(
                    #     backends=(
                    #             "sentencepiece", "tf"
                    #     )
                    # )
                    elif "backends" in lines[previous_index + 1]:
                        backends = []
                        for backend_line in lines[previous_index:index]:
                            if "backends" in backend_line:
                                backend_line = backend_line.split("=")[1]
                            if '"' in backend_line or "'" in backend_line:
                                if ", " in backend_line:
                                    backends.extend(backend.strip("()\"', ") for backend in backend_line.split(", "))
                                else:
                                    backends.append(backend_line.strip("()\"', "))

                            # If the line is only a ')', then we reached the end of the backends and we break.
                            if backend_line.strip() == ")":
                                break
                        backends = tuple(backends)

                    # No backends are registered for export
                    else:
                        backends = ()

                    backends = frozenset(backends + base_requirements)
                    if backends not in module_requirements:
                        module_requirements[backends] = {}
                    if module_name not in module_requirements[backends]:
                        module_requirements[backends][module_name] = set()

                    if not line.startswith("class") and not line.startswith("def"):
                        skip_line = True
                    else:
                        start_index = 6 if line.startswith("class") else 4
                        object_name = line[start_index:].split("(")[0].strip(":")
                        module_requirements[backends][module_name].add(object_name)
                        exported_objects.add(object_name)

                if not skip_line:
                    previous_line = line
                    previous_index = index

        # All objects that are in __all__ should be exported by default.
        # These objects are exported with the file backends.
        if "__all__" in file_content:
            for _all_object in fetch__all__(file_content):
                if _all_object not in exported_objects:
                    backends = frozenset(base_requirements)
                    if backends not in module_requirements:
                        module_requirements[backends] = {}
                    if module_name not in module_requirements[backends]:
                        module_requirements[backends][module_name] = set()

                    module_requirements[backends][module_name].add(_all_object)

    import_structure = {**module_requirements, **import_structure}
    return import_structure


def spread_import_structure(nested_import_structure):
    """
    This method takes as input an unordered import structure and brings the required backends at the top-level,
    aggregating modules and objects under their required backends.

    Here's an example of an input import structure at the src.transformers.models level:

    {
        'albert': {
            frozenset(): {
                'configuration_albert': {'AlbertConfig', 'AlbertOnnxConfig'}
            },
            frozenset({'tokenizers'}): {
                'tokenization_albert_fast': {'AlbertTokenizerFast'}
            },
        },
        'align': {
            frozenset(): {
                'configuration_align': {'AlignConfig', 'AlignTextConfig', 'AlignVisionConfig'},
                'processing_align': {'AlignProcessor'}
            },
        },
        'altclip': {
            frozenset(): {
                'configuration_altclip': {'AltCLIPConfig', 'AltCLIPTextConfig', 'AltCLIPVisionConfig'},
                'processing_altclip': {'AltCLIPProcessor'},
            }
        }
    }

    Here's an example of an output import structure at the src.transformers.models level:

    {
        frozenset({'tokenizers'}): {
            'albert.tokenization_albert_fast': {'AlbertTokenizerFast'}
        },
        frozenset(): {
            'albert.configuration_albert': {'AlbertConfig', 'AlbertOnnxConfig'},
            'align.processing_align': {'AlignProcessor'},
            'align.configuration_align': {'AlignConfig', 'AlignTextConfig', 'AlignVisionConfig'},
            'altclip.configuration_altclip': {'AltCLIPConfig', 'AltCLIPTextConfig', 'AltCLIPVisionConfig'},
            'altclip.processing_altclip': {'AltCLIPProcessor'}
        }
    }

    """

    def propagate_frozenset(unordered_import_structure):
        tuple_first_import_structure = {}
        for _key, _value in unordered_import_structure.items():
            if not isinstance(_value, dict):
                tuple_first_import_structure[_key] = _value

            elif any(isinstance(v, frozenset) for v in _value.keys()):
                # Here we want to switch around key and v
                for k, v in _value.items():
                    if isinstance(k, frozenset):
                        if k not in tuple_first_import_structure:
                            tuple_first_import_structure[k] = {}
                        tuple_first_import_structure[k][_key] = v

            else:
                tuple_first_import_structure[_key] = propagate_frozenset(_value)

        return tuple_first_import_structure

    def flatten_dict(_dict, previous_key=None):
        items = []
        for _key, _value in _dict.items():
            _key = f"{previous_key}.{_key}" if previous_key is not None else _key
            if isinstance(_value, dict):
                items.extend(flatten_dict(_value, _key).items())
            else:
                items.append((_key, _value))
        return dict(items)

    # The tuples contain the necessary backends. We want these first, so we propagate them up the
    # import structure.
    ordered_import_structure = nested_import_structure

    # 6 is a number that gives us sufficient depth to go through all files and foreseeable folder depths
    # while not taking too long to parse.
    for i in range(6):
        ordered_import_structure = propagate_frozenset(ordered_import_structure)

    # We then flatten the dict so that it references a module path.
    flattened_import_structure = {}
    for key, value in ordered_import_structure.copy().items():
        if isinstance(key, str):
            del ordered_import_structure[key]
        else:
            flattened_import_structure[key] = flatten_dict(value)

    return flattened_import_structure


def define_import_structure(module_path: str) -> IMPORT_STRUCTURE_T:
    """
    This method takes a module_path as input and creates an import structure digestible by a _LazyModule.

    Here's an example of an output import structure at the src.transformers.models level:

    {
        frozenset({'tokenizers'}): {
            'albert.tokenization_albert_fast': {'AlbertTokenizerFast'}
        },
        frozenset(): {
            'albert.configuration_albert': {'AlbertConfig', 'AlbertOnnxConfig'},
            'align.processing_align': {'AlignProcessor'},
            'align.configuration_align': {'AlignConfig', 'AlignTextConfig', 'AlignVisionConfig'},
            'altclip.configuration_altclip': {'AltCLIPConfig', 'AltCLIPTextConfig', 'AltCLIPVisionConfig'},
            'altclip.processing_altclip': {'AltCLIPProcessor'}
        }
    }

    The import structure is a dict defined with frozensets as keys, and dicts of strings to sets of objects.
    """
    import_structure = create_import_structure_from_path(module_path)
    return spread_import_structure(import_structure)
