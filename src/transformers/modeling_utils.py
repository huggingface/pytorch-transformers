# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors, Facebook AI Research authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
import collections
import copy
import functools
import gc
import importlib.metadata
import inspect
import itertools
import json
import os
import re
import shutil
import tempfile
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial, wraps
from threading import Thread
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union
from zipfile import is_zipfile

import torch
from huggingface_hub import split_torch_state_dict_into_shards
from packaging import version
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss, Identity
from torch.utils.checkpoint import checkpoint

from .activations import get_activation
from .configuration_utils import PretrainedConfig
from .dynamic_module_utils import custom_object_save
from .generation import CompileConfig, GenerationConfig, GenerationMixin
from .integrations import PeftAdapterMixin, deepspeed_config, is_deepspeed_zero3_enabled
from .integrations.flash_attention import flash_attention_forward
from .integrations.flex_attention import flex_attention_forward
from .integrations.sdpa_attention import sdpa_attention_forward
from .loss.loss_utils import LOSS_MAPPING
from .pytorch_utils import (  # noqa: F401
    Conv1D,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    id_tensor_storage,
    prune_conv1d_layer,
    prune_layer,
    prune_linear_layer,
    translate_to_torch_parallel_style,
)
from .quantizers import AutoHfQuantizer, HfQuantizer
from .quantizers.quantizers_utils import get_module_from_name
from .safetensors_conversion import auto_conversion
from .utils import (
    ACCELERATE_MIN_VERSION,
    ADAPTER_SAFE_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    CONFIG_NAME,
    DUMMY_INPUTS,
    FLAX_WEIGHTS_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    TF2_WEIGHTS_NAME,
    TF_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    ContextManagers,
    ModelOutput,
    PushToHubMixin,
    cached_file,
    copy_func,
    download_url,
    extract_commit_hash,
    has_file,
    is_accelerate_available,
    is_bitsandbytes_available,
    is_flash_attn_2_available,
    is_offline_mode,
    is_optimum_available,
    is_peft_available,
    is_remote_url,
    is_safetensors_available,
    is_torch_flex_attn_available,
    is_torch_greater_or_equal,
    is_torch_sdpa_available,
    is_torch_xla_available,
    logging,
    replace_return_docstrings,
    strtobool,
)
from .utils.hub import create_and_tag_model_card, get_checkpoint_shard_files
from .utils.import_utils import (
    ENV_VARS_TRUE_VALUES,
    is_sagemaker_mp_enabled,
    is_torch_fx_proxy,
    is_torchdynamo_compiling,
)
from .utils.quantization_config import BitsAndBytesConfig, QuantizationMethod


XLA_USE_BF16 = os.environ.get("XLA_USE_BF16", "0").upper()
XLA_DOWNCAST_BF16 = os.environ.get("XLA_DOWNCAST_BF16", "0").upper()


if is_accelerate_available():
    from accelerate import dispatch_model, infer_auto_device_map, init_empty_weights
    from accelerate.hooks import add_hook_to_module
    from accelerate.utils import (
        check_tied_parameters_on_same_device,
        extract_model_from_parallel,
        find_tied_parameters,
        get_balanced_memory,
        get_max_memory,
        load_offloaded_weights,
        offload_weight,
        save_offload_index,
        set_module_tensor_to_device,
    )

    accelerate_version = version.parse(importlib.metadata.version("accelerate"))
    if accelerate_version >= version.parse("0.31"):
        from accelerate.utils.modeling import get_state_dict_from_offload

if is_safetensors_available():
    from safetensors import safe_open
    from safetensors.torch import load_file as safe_load_file
    from safetensors.torch import save_file as safe_save_file

logger = logging.get_logger(__name__)


_init_weights = True
_is_quantized = False
_is_ds_init_called = False


def is_fsdp_enabled():
    return (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and strtobool(os.environ.get("ACCELERATE_USE_FSDP", "False")) == 1
        and strtobool(os.environ.get("FSDP_CPU_RAM_EFFICIENT_LOADING", "False")) == 1
    )


def is_local_dist_rank_0():
    return (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and int(os.environ.get("LOCAL_RANK", -1)) == 0
    )


if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

if is_peft_available():
    from .utils import find_adapter_config_file

SpecificPreTrainedModelType = TypeVar("SpecificPreTrainedModelType", bound="PreTrainedModel")

TORCH_INIT_FUNCTIONS = {
    "uniform_": nn.init.uniform_,
    "normal_": nn.init.normal_,
    "trunc_normal_": nn.init.trunc_normal_,
    "constant_": nn.init.constant_,
    "xavier_uniform_": nn.init.xavier_uniform_,
    "xavier_normal_": nn.init.xavier_normal_,
    "kaiming_uniform_": nn.init.kaiming_uniform_,
    "kaiming_normal_": nn.init.kaiming_normal_,
    "uniform": nn.init.uniform,
    "normal": nn.init.normal,
    "xavier_uniform": nn.init.xavier_uniform,
    "xavier_normal": nn.init.xavier_normal,
    "kaiming_uniform": nn.init.kaiming_uniform,
    "kaiming_normal": nn.init.kaiming_normal,
}


@contextmanager
def no_init_weights(_enable=True):
    """
    Context manager to globally disable weight initialization to speed up loading large models.

    TODO(Patrick): Delete safety argument `_enable=True` at next major version. .
    """
    global _init_weights
    old_init_weights = _init_weights

    if _enable:
        _init_weights = False

        def _skip_init(*args, **kwargs):
            pass

        # # Save the original initialization functions
        for name, init_func in TORCH_INIT_FUNCTIONS.items():
            setattr(torch.nn.init, name, _skip_init)
    try:
        yield
    finally:
        _init_weights = old_init_weights
        if _enable:
            # # Restore the original initialization functions
            for name, init_func in TORCH_INIT_FUNCTIONS.items():
                setattr(torch.nn.init, name, init_func)


@contextmanager
def set_quantized_state():
    global _is_quantized
    _is_quantized = True
    try:
        yield
    finally:
        _is_quantized = False


# Skip recursive calls to deepspeed.zero.Init to avoid pinning errors.
# This issue occurs with ZeRO stage 3 when using NVMe offloading.
# For more details, refer to issue #34429.
@contextmanager
def set_zero3_state():
    global _is_ds_init_called
    _is_ds_init_called = True
    try:
        yield
    finally:
        _is_ds_init_called = False


def get_parameter_device(parameter: Union[nn.Module, "ModuleUtilsMixin"]):
    try:
        return next(parameter.parameters()).device
    except StopIteration:
        # For nn.DataParallel compatibility in PyTorch 1.5

        def find_tensor_attributes(module: nn.Module) -> List[Tuple[str, Tensor]]:
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
            return tuples

        gen = parameter._named_members(get_members_fn=find_tensor_attributes)
        first_tuple = next(gen)
        return first_tuple[1].device


def get_first_parameter_dtype(parameter: Union[nn.Module, "ModuleUtilsMixin"]):
    """
    Returns the first parameter dtype (can be non-floating) or asserts if none were found.
    """
    try:
        return next(parameter.parameters()).dtype
    except StopIteration:
        # For nn.DataParallel compatibility in PyTorch > 1.5

        def find_tensor_attributes(module: nn.Module) -> List[Tuple[str, Tensor]]:
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
            return tuples

        gen = parameter._named_members(get_members_fn=find_tensor_attributes)
        first_tuple = next(gen)
        return first_tuple[1].dtype


def get_parameter_dtype(parameter: Union[nn.Module, "ModuleUtilsMixin"]):
    """
    Returns the first found floating dtype in parameters if there is one, otherwise returns the last dtype it found.
    """
    last_dtype = None
    for t in parameter.parameters():
        last_dtype = t.dtype
        if t.is_floating_point():
            # Adding fix for https://github.com/pytorch/xla/issues/4152
            # Fixes issue where the model code passes a value that is out of range for XLA_USE_BF16=1
            # and XLA_DOWNCAST_BF16=1 so the conversion would cast it to -inf
            # NOTE: `is_torch_xla_available()` is checked last as it induces a graph break in torch dynamo
            if XLA_USE_BF16 in ENV_VARS_TRUE_VALUES and is_torch_xla_available():
                return torch.bfloat16
            if XLA_DOWNCAST_BF16 in ENV_VARS_TRUE_VALUES and is_torch_xla_available():
                if t.dtype == torch.float:
                    return torch.bfloat16
                if t.dtype == torch.double:
                    return torch.float32
            return t.dtype

    if last_dtype is not None:
        # if no floating dtype was found return whatever the first dtype is
        return last_dtype

    # For nn.DataParallel compatibility in PyTorch > 1.5
    def find_tensor_attributes(module: nn.Module) -> List[Tuple[str, Tensor]]:
        tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
        return tuples

    gen = parameter._named_members(get_members_fn=find_tensor_attributes)
    last_tuple = None
    for tuple in gen:
        last_tuple = tuple
        if tuple[1].is_floating_point():
            return tuple[1].dtype

    if last_tuple is not None:
        # fallback to the last dtype
        return last_tuple[1].dtype

    # fallback to buffer dtype
    for t in parameter.buffers():
        last_dtype = t.dtype
        if t.is_floating_point():
            return t.dtype
    return last_dtype


def get_state_dict_float_dtype(state_dict):
    """
    Returns the first found floating dtype in `state_dict` or asserts if none were found.
    """
    for t in state_dict.values():
        if t.is_floating_point():
            return t.dtype

    raise ValueError("couldn't find any floating point dtypes in state_dict")


def get_state_dict_dtype(state_dict):
    """
    Returns the first found floating dtype in `state_dict` if there is one, otherwise returns the first dtype.
    """
    for t in state_dict.values():
        if t.is_floating_point():
            return t.dtype

    # if no floating dtype was found return whatever the first dtype is
    else:
        return next(state_dict.values()).dtype


def dtype_byte_size(dtype):
    """
    Returns the size (in bytes) occupied by one parameter of type `dtype`.

    Example:

    ```py
    >>> dtype_byte_size(torch.float32)
    4
    ```
    """
    if dtype == torch.bool:
        return 1 / 8
    bit_search = re.search(r"[^\d](\d+)_?", str(dtype))
    if bit_search is None:
        raise ValueError(f"`dtype` is not a valid dtype: {dtype}.")
    bit_size = int(bit_search.groups()[0])
    return bit_size // 8


def check_support_param_buffer_assignment(model_to_load, state_dict, start_prefix=""):
    """
    Checks if `model_to_load` supports param buffer assignment (such
    as when loading in empty weights) by first checking
    if the model explicitly disables it, then by ensuring that the state dict keys
    are a subset of the model's parameters.

    Note: We fully disable this if we are using `deepspeed`
    """
    if model_to_load.device.type == "meta":
        return False

    if len([key for key in state_dict if key.startswith(start_prefix)]) == 0:
        return False

    if is_deepspeed_zero3_enabled():
        return False

    # Some models explicitly do not support param buffer assignment
    if not getattr(model_to_load, "_supports_param_buffer_assignment", True):
        logger.debug(
            f"{model_to_load.__class__.__name__} does not support param buffer assignment, loading will be slower"
        )
        return False

    # If the model does, the incoming `state_dict` and the `model_to_load` must be the same dtype
    first_key = next(iter(model_to_load.state_dict().keys()))
    if start_prefix + first_key in state_dict:
        return state_dict[start_prefix + first_key].dtype == model_to_load.state_dict()[first_key].dtype

    # For cases when the `state_dict` doesn't contain real weights to the model (`test_model_weights_reload_no_missing_tied_weights`)
    return False


def load_sharded_checkpoint(model, folder, strict=True, prefer_safe=True):
    """
    This is the same as
    [`torch.nn.Module.load_state_dict`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=load_state_dict#torch.nn.Module.load_state_dict)
    but for a sharded checkpoint.

    This load is performed efficiently: each checkpoint shard is loaded one by one in RAM and deleted after being
    loaded in the model.

    Args:
        model (`torch.nn.Module`): The model in which to load the checkpoint.
        folder (`str` or `os.PathLike`): A path to a folder containing the sharded checkpoint.
        strict (`bool`, *optional`, defaults to `True`):
            Whether to strictly enforce that the keys in the model state dict match the keys in the sharded checkpoint.
        prefer_safe (`bool`, *optional*, defaults to `False`)
            If both safetensors and PyTorch save files are present in checkpoint and `prefer_safe` is True, the
            safetensors files will be loaded. Otherwise, PyTorch files are always loaded when possible.

    Returns:
        `NamedTuple`: A named tuple with `missing_keys` and `unexpected_keys` fields
            - `missing_keys` is a list of str containing the missing keys
            - `unexpected_keys` is a list of str containing the unexpected keys
    """
    # Load the index
    index_file = os.path.join(folder, WEIGHTS_INDEX_NAME)
    safe_index_file = os.path.join(folder, SAFE_WEIGHTS_INDEX_NAME)

    index_present = os.path.isfile(index_file)
    safe_index_present = os.path.isfile(safe_index_file)

    if not index_present and not (safe_index_present and is_safetensors_available()):
        filenames = (
            (WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_INDEX_NAME) if is_safetensors_available() else (WEIGHTS_INDEX_NAME,)
        )
        raise ValueError(f"Can't find a checkpoint index ({' or '.join(filenames)}) in {folder}.")

    load_safe = False
    if safe_index_present:
        if prefer_safe:
            if is_safetensors_available():
                load_safe = True  # load safe due to preference
            else:
                logger.warning(
                    f"Cannot load sharded checkpoint at {folder} safely since safetensors is not installed!"
                )
        elif not index_present:
            load_safe = True  # load safe since we have no other choice

    load_index = safe_index_file if load_safe else index_file

    with open(load_index, "r", encoding="utf-8") as f:
        index = json.load(f)

    shard_files = list(set(index["weight_map"].values()))

    # If strict=True, error before loading any of the state dicts.
    loaded_keys = index["weight_map"].keys()
    model_keys = model.state_dict().keys()
    missing_keys = [key for key in model_keys if key not in loaded_keys]
    unexpected_keys = [key for key in loaded_keys if key not in model_keys]
    if strict and (len(missing_keys) > 0 or len(unexpected_keys) > 0):
        error_message = f"Error(s) in loading state_dict for {model.__class__.__name__}"
        if len(missing_keys) > 0:
            str_missing_keys = ",".join([f'"{k}"' for k in missing_keys])
            error_message += f"\nMissing key(s): {str_missing_keys}."
        if len(unexpected_keys) > 0:
            str_unexpected_keys = ",".join([f'"{k}"' for k in unexpected_keys])
            error_message += f"\nMissing key(s): {str_unexpected_keys}."
        raise RuntimeError(error_message)

    weights_only_kwarg = {"weights_only": True}
    loader = safe_load_file if load_safe else partial(torch.load, map_location="cpu", **weights_only_kwarg)

    for shard_file in shard_files:
        state_dict = loader(os.path.join(folder, shard_file))
        model.load_state_dict(state_dict, strict=False)

        # Make sure memory is freed before we load the next state dict.
        del state_dict
        gc.collect()

    # Return the same thing as PyTorch load_state_dict function.
    return torch.nn.modules.module._IncompatibleKeys(missing_keys, unexpected_keys)


def load_state_dict(
    checkpoint_file: Union[str, os.PathLike],
    is_quantized: bool = False,
    map_location: Optional[Union[str, torch.device]] = None,
    weights_only: bool = True,
):
    """
    Reads a PyTorch checkpoint file, returning properly formatted errors if they arise.
    """
    if checkpoint_file.endswith(".safetensors") and is_safetensors_available():
        # Check format of the archive
        with safe_open(checkpoint_file, framework="pt") as f:
            metadata = f.metadata()
        if metadata is not None and metadata.get("format") not in ["pt", "tf", "flax", "mlx"]:
            raise OSError(
                f"The safetensors archive passed at {checkpoint_file} does not contain the valid metadata. Make sure "
                "you save your model with the `save_pretrained` method."
            )
        return safe_load_file(checkpoint_file)
    try:
        if map_location is None:
            if (
                (
                    is_deepspeed_zero3_enabled()
                    and torch.distributed.is_initialized()
                    and torch.distributed.get_rank() > 0
                )
                or (is_fsdp_enabled() and not is_local_dist_rank_0())
            ) and not is_quantized:
                map_location = "meta"
            else:
                map_location = "cpu"
        extra_args = {}
        # mmap can only be used with files serialized with zipfile-based format.
        if (
            isinstance(checkpoint_file, str)
            and map_location != "meta"
            and version.parse(torch.__version__) >= version.parse("2.1.0")
            and is_zipfile(checkpoint_file)
        ):
            extra_args = {"mmap": True}
        weights_only_kwarg = {"weights_only": weights_only}
        return torch.load(
            checkpoint_file,
            map_location=map_location,
            **weights_only_kwarg,
            **extra_args,
        )
    except Exception as e:
        try:
            with open(checkpoint_file) as f:
                if f.read(7) == "version":
                    raise OSError(
                        "You seem to have cloned a repository without having git-lfs installed. Please install "
                        "git-lfs and run `git lfs install` followed by `git lfs pull` in the folder "
                        "you cloned."
                    )
                else:
                    raise ValueError(
                        f"Unable to locate the file {checkpoint_file} which is necessary to load this pretrained "
                        "model. Make sure you have saved the model properly."
                    ) from e
        except (UnicodeDecodeError, ValueError):
            raise OSError(
                f"Unable to load weights from pytorch checkpoint file for '{checkpoint_file}' "
                f"at '{checkpoint_file}'. "
                "If you tried to load a PyTorch model from a TF 2.0 checkpoint, please set from_tf=True."
            )


def set_initialized_submodules(model, state_dict_keys):
    """
    Sets the `_is_hf_initialized` flag in all submodules of a given model when all its weights are in the loaded state
    dict.
    """
    not_initialized_submodules = {}
    for module_name, module in model.named_modules():
        loaded_keys = {k.replace(f"{module_name}.", "") for k in state_dict_keys if k.startswith(f"{module_name}.")}
        # When checking if the root module is loaded all state_dict_keys must be used.
        if module_name == "":
            loaded_keys = set(state_dict_keys)
        if loaded_keys.issuperset(module.state_dict()):
            module._is_hf_initialized = True
        else:
            not_initialized_submodules[module_name] = module
    return not_initialized_submodules


def _end_ptr(tensor: torch.Tensor) -> int:
    # extract the end of the pointer if the tensor is a slice of a bigger tensor
    if tensor.nelement():
        stop = tensor.view(-1)[-1].data_ptr() + tensor.element_size()
    else:
        stop = tensor.data_ptr()
    return stop


def _get_tied_weight_keys(module: nn.Module, prefix=""):
    tied_weight_keys = []
    if getattr(module, "_tied_weights_keys", None) is not None:
        names = [f"{prefix}.{k}" if prefix else k for k in module._tied_weights_keys]
        tied_weight_keys.extend(names)
    if getattr(module, "_dynamic_tied_weights_keys", None) is not None:
        names = [f"{prefix}.{k}" if prefix else k for k in module._dynamic_tied_weights_keys]
        tied_weight_keys.extend(names)
    for name, submodule in module.named_children():
        local_prefix = f"{prefix}.{name}" if prefix else name
        tied_weight_keys.extend(_get_tied_weight_keys(submodule, prefix=local_prefix))
    return tied_weight_keys


def _find_disjoint(tensors: List[Set[str]], state_dict: Dict[str, torch.Tensor]) -> Tuple[List[Set[str]], List[str]]:
    filtered_tensors = []
    for shared in tensors:
        if len(shared) < 2:
            filtered_tensors.append(shared)
            continue

        areas = []
        for name in shared:
            tensor = state_dict[name]
            areas.append((tensor.data_ptr(), _end_ptr(tensor), name))
        areas.sort()

        _, last_stop, last_name = areas[0]
        filtered_tensors.append({last_name})
        for start, stop, name in areas[1:]:
            if start >= last_stop:
                filtered_tensors.append({name})
            else:
                filtered_tensors[-1].add(name)
            last_stop = stop
    disjoint_tensors = []
    shared_tensors = []
    for tensors in filtered_tensors:
        if len(tensors) == 1:
            disjoint_tensors.append(tensors.pop())
        else:
            shared_tensors.append(tensors)
    return shared_tensors, disjoint_tensors


def _find_identical(tensors: List[Set[str]], state_dict: Dict[str, torch.Tensor]) -> Tuple[List[Set[str]], Set[str]]:
    shared_tensors = []
    identical = []
    for shared in tensors:
        if len(shared) < 2:
            continue

        areas = collections.defaultdict(set)
        for name in shared:
            tensor = state_dict[name]
            area = (tensor.device, tensor.data_ptr(), _end_ptr(tensor))
            areas[area].add(name)
        if len(areas) == 1:
            identical.append(shared)
        else:
            shared_tensors.append(shared)
    return shared_tensors, identical


def _load_state_dict_into_model(model_to_load, state_dict, start_prefix, assign_to_params_buffers=False):
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    error_msgs = []

    # PyTorch's `_load_from_state_dict` does not copy parameters in a module's descendants
    # so we need to apply the function recursively.
    def load(module: nn.Module, state_dict, prefix="", assign_to_params_buffers=False):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        local_metadata["assign_to_params_buffers"] = assign_to_params_buffers

        args = (state_dict, prefix, local_metadata, True, [], [], error_msgs)
        # Parameters of module and children will start with prefix. We can exit early if there are none in this
        # state_dict
        if len([key for key in state_dict if key.startswith(prefix)]) > 0:
            if is_deepspeed_zero3_enabled():
                import deepspeed

                # In sharded models, each shard has only part of the full state_dict, so only gather
                # parameters that are in the current state_dict.
                named_parameters = dict(module.named_parameters(prefix=prefix[:-1], recurse=False))
                params_to_gather = [named_parameters[k] for k in state_dict.keys() if k in named_parameters]
                if len(params_to_gather) > 0:
                    # because zero3 puts placeholders in model params, this context
                    # manager gathers (unpartitions) the params of the current layer, then loads from
                    # the state dict and then re-partitions them again
                    with deepspeed.zero.GatheredParameters(params_to_gather, modifier_rank=0):
                        if torch.distributed.get_rank() == 0:
                            module._load_from_state_dict(*args)
            else:
                module._load_from_state_dict(*args)

        for name, child in module._modules.items():
            if child is not None:
                load(child, state_dict, prefix + name + ".", assign_to_params_buffers)

    load(model_to_load, state_dict, prefix=start_prefix, assign_to_params_buffers=assign_to_params_buffers)
    # Delete `state_dict` so it could be collected by GC earlier. Note that `state_dict` is a copy of the argument, so
    # it's safe to delete it.
    del state_dict

    return error_msgs


def find_submodule_and_param_name(model, long_key, start_prefix):
    """
    A helper util to find the last sub-module and the param/buffer name. If `start_prefix` is supplied it'll be removed
    from the start of the key
    """

    if len(start_prefix) > 0 and long_key.startswith(start_prefix):
        long_key = ".".join(long_key.split(".")[1:])

    split_key = long_key.split(".")
    submodule = model
    while len(split_key) > 1:
        if hasattr(submodule, split_key[0]):
            submodule = getattr(submodule, split_key[0])
            del split_key[0]
        else:
            submodule = None
            break
    if submodule == model:
        submodule = None
    return submodule, split_key[0]


def _move_model_to_meta(model, loaded_state_dict_keys, start_prefix):
    """
    Moves `loaded_state_dict_keys` in model to meta device which frees up the memory taken by those params.

    `start_prefix` is used for models which insert their name into model keys, e.g. `bert` in
    `bert.pooler.dense.weight`

    """

    # dematerialize param storage for keys that are going to be replaced by state_dict, by
    # putting those on the meta device
    for k in loaded_state_dict_keys:
        submodule, param_name = find_submodule_and_param_name(model, k, start_prefix)
        if submodule is not None:
            # selectively switch to the meta device only those params/buffers that will
            # be next replaced from state_dict. This a complex way to do p.to_("meta")
            # since we have no in-place to_ for tensors.
            new_val = getattr(submodule, param_name)
            if isinstance(new_val, torch.nn.Parameter):
                # isinstance returns False for Params on meta device, so switch after the check
                new_val = torch.nn.Parameter(new_val.to("meta"))
            else:
                new_val = new_val.to("meta")
            setattr(submodule, param_name, new_val)


def _load_state_dict_into_meta_model(
    model,
    state_dict,
    start_prefix,
    expected_keys,
    device_map=None,
    offload_folder=None,
    offload_index=None,
    state_dict_folder=None,
    state_dict_index=None,
    dtype=None,
    hf_quantizer=None,
    is_safetensors=False,
    keep_in_fp32_modules=None,
    unexpected_keys=None,  # passing `unexpected` for cleanup from quantization items
    pretrained_model_name_or_path=None,  # for flagging the user when the model contains renamed keys
):
    """
    This is somewhat similar to `_load_state_dict_into_model`, but deals with a model that has some or all of its
    params on a `meta` device. It replaces the model params with the data from the `state_dict`, while moving the
    params back to the normal device, but only for `loaded_state_dict_keys`.

    `start_prefix` is used for models which insert their name into model keys, e.g. `bert` in
    `bert.pooler.dense.weight`

    """

    # XXX: remaining features to implement to be fully compatible with _load_state_dict_into_model
    # - deepspeed zero 3 support
    # - need to copy metadata if any - see _load_state_dict_into_model
    # - handling error_msgs - mimicking the error handling in module._load_from_state_dict()

    error_msgs = []

    is_quantized = hf_quantizer is not None

    is_torch_e4m3fn_available = hasattr(torch, "float8_e4m3fn")

    for param_name, param in state_dict.items():
        if param_name not in expected_keys:
            continue

        if param_name.startswith(start_prefix):
            param_name = param_name[len(start_prefix) :]

        module_name = param_name
        set_module_kwargs = {}

        # We convert floating dtypes to the `dtype` passed except for float8_e4m3fn type. We also want to keep the buffers/params
        # in int/uint/bool and not cast them.
        is_param_float8_e4m3fn = is_torch_e4m3fn_available and param.dtype == torch.float8_e4m3fn
        if dtype is not None and torch.is_floating_point(param) and not is_param_float8_e4m3fn:
            if (
                keep_in_fp32_modules is not None
                and any(
                    module_to_keep_in_fp32 in param_name.split(".") for module_to_keep_in_fp32 in keep_in_fp32_modules
                )
                and dtype == torch.float16
            ):
                param = param.to(torch.float32)

                # For backward compatibility with older versions of `accelerate`
                # TODO: @sgugger replace this check with version check at the next `accelerate` release
                if "dtype" in list(inspect.signature(set_module_tensor_to_device).parameters):
                    set_module_kwargs["dtype"] = torch.float32
            else:
                param = param.to(dtype)

        # For compatibility with PyTorch load_state_dict which converts state dict dtype to existing dtype in model, and which
        # uses `param.copy_(input_param)` that preserves the contiguity of the parameter in the model.
        # Reference: https://github.com/pytorch/pytorch/blob/db79ceb110f6646523019a59bbd7b838f43d4a86/torch/nn/modules/module.py#L2040C29-L2040C29

        old_param = model
        splits = param_name.split(".")
        for split in splits:
            # We shouldn't hit the default value unless for quant methods like hqq that modifies expected_keys.
            old_param = getattr(old_param, split, None)
            if old_param is None:
                break

        if not isinstance(old_param, (torch.nn.Parameter, torch.Tensor)):
            old_param = None

        if old_param is not None:
            if dtype is None:
                param = param.to(old_param.dtype)

            if old_param.is_contiguous():
                param = param.contiguous()

        set_module_kwargs["value"] = param

        if device_map is None:
            param_device = "cpu"
        else:
            # find next higher level module that is defined in device_map:
            # bert.lm_head.weight -> bert.lm_head -> bert -> ''
            while len(module_name) > 0 and module_name not in device_map:
                module_name = ".".join(module_name.split(".")[:-1])
            if module_name == "" and "" not in device_map:
                # TODO: group all errors and raise at the end.
                raise ValueError(f"{param_name} doesn't have any device set.")
            param_device = device_map[module_name]

        if param_device == "disk":
            if not is_safetensors:
                offload_index = offload_weight(param, param_name, offload_folder, offload_index)
        elif param_device == "cpu" and state_dict_index is not None:
            state_dict_index = offload_weight(param, param_name, state_dict_folder, state_dict_index)
        elif (
            not is_quantized
            or (not hf_quantizer.requires_parameters_quantization)
            or (
                not hf_quantizer.check_quantized_param(
                    model, param, param_name, state_dict, param_device=param_device, device_map=device_map
                )
            )
        ):
            if is_fsdp_enabled():
                param_device = "cpu" if is_local_dist_rank_0() else "meta"

            # For backward compatibility with older versions of `accelerate` and for non-quantized params
            set_module_tensor_to_device(model, param_name, param_device, **set_module_kwargs)
        else:
            hf_quantizer.create_quantized_param(model, param, param_name, param_device, state_dict, unexpected_keys)
            # For quantized modules with FSDP/DeepSpeed Stage 3, we need to quantize the parameter on the GPU
            # and then cast it to CPU to avoid excessive memory usage on each GPU
            # in comparison to the sharded model across GPUs.
            if is_fsdp_enabled() or is_deepspeed_zero3_enabled():
                module, tensor_name = get_module_from_name(model, param_name)
                value = getattr(module, tensor_name)
                param_to = "cpu"
                if is_fsdp_enabled() and not is_local_dist_rank_0():
                    param_to = "meta"
                val_kwargs = {}
                if hasattr(module, "weight") and module.weight.__class__.__name__ == "Int8Params":
                    val_kwargs["requires_grad"] = False
                value = type(value)(value.data.to(param_to), **val_kwargs, **value.__dict__)
                setattr(module, tensor_name, value)
            # TODO: consider removing used param_parts from state_dict before return

    return error_msgs, offload_index, state_dict_index


def _add_variant(weights_name: str, variant: Optional[str] = None) -> str:
    if variant is not None:
        splits = weights_name.split(".")
        splits = splits[:-1] + [variant] + splits[-1:]
        weights_name = ".".join(splits)

    return weights_name


class ModuleUtilsMixin:
    """
    A few utilities for `torch.nn.Modules`, to be used as a mixin.
    """

    @staticmethod
    def _hook_rss_memory_pre_forward(module, *args, **kwargs):
        try:
            import psutil
        except ImportError:
            raise ImportError("You need to install psutil (pip install psutil) to use memory tracing.")

        process = psutil.Process(os.getpid())
        mem = process.memory_info()
        module.mem_rss_pre_forward = mem.rss
        return None

    @staticmethod
    def _hook_rss_memory_post_forward(module, *args, **kwargs):
        try:
            import psutil
        except ImportError:
            raise ImportError("You need to install psutil (pip install psutil) to use memory tracing.")

        process = psutil.Process(os.getpid())
        mem = process.memory_info()
        module.mem_rss_post_forward = mem.rss
        mem_rss_diff = module.mem_rss_post_forward - module.mem_rss_pre_forward
        module.mem_rss_diff = mem_rss_diff + (module.mem_rss_diff if hasattr(module, "mem_rss_diff") else 0)
        return None

    def add_memory_hooks(self):
        """
        Add a memory hook before and after each sub-module forward pass to record increase in memory consumption.

        Increase in memory consumption is stored in a `mem_rss_diff` attribute for each module and can be reset to zero
        with `model.reset_memory_hooks_state()`.
        """
        for module in self.modules():
            module.register_forward_pre_hook(self._hook_rss_memory_pre_forward)
            module.register_forward_hook(self._hook_rss_memory_post_forward)
        self.reset_memory_hooks_state()

    def reset_memory_hooks_state(self):
        """
        Reset the `mem_rss_diff` attribute of each module (see [`~modeling_utils.ModuleUtilsMixin.add_memory_hooks`]).
        """
        for module in self.modules():
            module.mem_rss_diff = 0
            module.mem_rss_post_forward = 0
            module.mem_rss_pre_forward = 0

    @property
    def device(self) -> torch.device:
        """
        `torch.device`: The device on which the module is (assuming that all the module parameters are on the same
        device).
        """
        return get_parameter_device(self)

    @property
    def dtype(self) -> torch.dtype:
        """
        `torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        return get_parameter_dtype(self)

    def invert_attention_mask(self, encoder_attention_mask: Tensor) -> Tensor:
        """
        Invert an attention mask (e.g., switches 0. and 1.).

        Args:
            encoder_attention_mask (`torch.Tensor`): An attention mask.

        Returns:
            `torch.Tensor`: The inverted attention mask.
        """
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        if encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
        # T5 has a mask that can compare sequence ids, we can simulate this here with this transposition
        # Cf. https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6/mesh_tensorflow
        # /transformer/transformer_layers.py#L270
        # encoder_extended_attention_mask = (encoder_extended_attention_mask ==
        # encoder_extended_attention_mask.transpose(-1, -2))
        encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * torch.finfo(self.dtype).min

        return encoder_extended_attention_mask

    @staticmethod
    def create_extended_attention_mask_for_decoder(input_shape, attention_mask, device=None):
        if device is not None:
            warnings.warn(
                "The `device` argument is deprecated and will be removed in v5 of Transformers.", FutureWarning
            )
        else:
            device = attention_mask.device
        batch_size, seq_length = input_shape
        seq_ids = torch.arange(seq_length, device=device)
        causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
        # in case past_key_values are used we need to add a prefix ones mask to the causal mask
        # causal and attention masks must have same type with pytorch version < 1.3
        causal_mask = causal_mask.to(attention_mask.dtype)

        if causal_mask.shape[1] < attention_mask.shape[1]:
            prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
            causal_mask = torch.cat(
                [
                    torch.ones((batch_size, seq_length, prefix_seq_len), device=device, dtype=causal_mask.dtype),
                    causal_mask,
                ],
                axis=-1,
            )

        extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
        return extended_attention_mask

    def get_extended_attention_mask(
        self, attention_mask: Tensor, input_shape: Tuple[int], device: torch.device = None, dtype: torch.float = None
    ) -> Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (`Tuple[int]`):
                The shape of the input to the model.

        Returns:
            `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
        """
        if dtype is None:
            dtype = self.dtype

        if not (attention_mask.dim() == 2 and self.config.is_decoder):
            # show warning only if it won't be shown in `create_extended_attention_mask_for_decoder`
            if device is not None:
                warnings.warn(
                    "The `device` argument is deprecated and will be removed in v5 of Transformers.", FutureWarning
                )
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder:
                extended_attention_mask = ModuleUtilsMixin.create_extended_attention_mask_for_decoder(
                    input_shape, attention_mask, device
                )
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and the dtype's smallest value for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
        return extended_attention_mask

    def get_head_mask(
        self, head_mask: Optional[Tensor], num_hidden_layers: int, is_attention_chunked: bool = False
    ) -> Tensor:
        """
        Prepare the head mask if needed.

        Args:
            head_mask (`torch.Tensor` with shape `[num_heads]` or `[num_hidden_layers x num_heads]`, *optional*):
                The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 for discard).
            num_hidden_layers (`int`):
                The number of hidden layers in the model.
            is_attention_chunked (`bool`, *optional*, defaults to `False`):
                Whether or not the attentions scores are computed by chunks or not.

        Returns:
            `torch.Tensor` with shape `[num_hidden_layers x batch x num_heads x seq_length x seq_length]` or list with
            `[None]` for each layer.
        """
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers

        return head_mask

    def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
        """-> [num_hidden_layers x batch x num_heads x seq_length x seq_length]"""
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
        assert head_mask.dim() == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
        head_mask = head_mask.to(dtype=self.dtype)  # switch to float if need + fp16 compatibility
        return head_mask

    def num_parameters(self, only_trainable: bool = False, exclude_embeddings: bool = False) -> int:
        """
        Get number of (optionally, trainable or non-embeddings) parameters in the module.

        Args:
            only_trainable (`bool`, *optional*, defaults to `False`):
                Whether or not to return only the number of trainable parameters

            exclude_embeddings (`bool`, *optional*, defaults to `False`):
                Whether or not to return only the number of non-embeddings parameters

        Returns:
            `int`: The number of parameters.
        """

        if exclude_embeddings:
            embedding_param_names = [
                f"{name}.weight" for name, module_type in self.named_modules() if isinstance(module_type, nn.Embedding)
            ]
            total_parameters = [
                parameter for name, parameter in self.named_parameters() if name not in embedding_param_names
            ]
        else:
            total_parameters = list(self.parameters())

        total_numel = []
        is_loaded_in_4bit = getattr(self, "is_loaded_in_4bit", False)

        if is_loaded_in_4bit:
            if is_bitsandbytes_available():
                import bitsandbytes as bnb
            else:
                raise ValueError(
                    "bitsandbytes is not installed but it seems that the model has been loaded in 4bit precision, something went wrong"
                    " make sure to install bitsandbytes with `pip install bitsandbytes`. You also need a GPU. "
                )

        for param in total_parameters:
            if param.requires_grad or not only_trainable:
                # For 4bit models, we need to multiply the number of parameters by 2 as half of the parameters are
                # used for the 4bit quantization (uint8 tensors are stored)
                if is_loaded_in_4bit and isinstance(param, bnb.nn.Params4bit):
                    if hasattr(param, "element_size"):
                        num_bytes = param.element_size()
                    elif hasattr(param, "quant_storage"):
                        num_bytes = param.quant_storage.itemsize
                    else:
                        num_bytes = 1
                    total_numel.append(param.numel() * 2 * num_bytes)
                else:
                    total_numel.append(param.numel())

        return sum(total_numel)

    def estimate_tokens(self, input_dict: Dict[str, Union[torch.Tensor, Any]]) -> int:
        """
        Helper function to estimate the total number of tokens from the model inputs.

        Args:
            inputs (`dict`): The model inputs.

        Returns:
            `int`: The total number of tokens.
        """
        if not hasattr(self, "warnings_issued"):
            self.warnings_issued = {}
        if self.main_input_name in input_dict:
            return input_dict[self.main_input_name].numel()
        elif "estimate_tokens" not in self.warnings_issued:
            logger.warning(
                "Could not estimate the number of tokens of the input, floating-point operations will not be computed"
            )
            self.warnings_issued["estimate_tokens"] = True
        return 0

    def floating_point_ops(
        self, input_dict: Dict[str, Union[torch.Tensor, Any]], exclude_embeddings: bool = True
    ) -> int:
        """
        Get number of (optionally, non-embeddings) floating-point operations for the forward and backward passes of a
        batch with this transformer model. Default approximation neglects the quadratic dependency on the number of
        tokens (valid if `12 * d_model << sequence_length`) as laid out in [this
        paper](https://arxiv.org/pdf/2001.08361.pdf) section 2.1. Should be overridden for transformers with parameter
        re-use e.g. Albert or Universal Transformers, or if doing long-range modeling with very high sequence lengths.

        Args:
            batch_size (`int`):
                The batch size for the forward pass.

            sequence_length (`int`):
                The number of tokens in each line of the batch.

            exclude_embeddings (`bool`, *optional*, defaults to `True`):
                Whether or not to count embedding and softmax operations.

        Returns:
            `int`: The number of floating-point operations.
        """

        return 6 * self.estimate_tokens(input_dict) * self.num_parameters(exclude_embeddings=exclude_embeddings)


# TODO (joao): remove `GenerationMixin` inheritance in v4.50
class PreTrainedModel(nn.Module, ModuleUtilsMixin, GenerationMixin, PushToHubMixin, PeftAdapterMixin):
    r"""
    Base class for all models.

    [`PreTrainedModel`] takes care of storing the configuration of the models and handles methods for loading,
    downloading and saving models as well as a few methods common to all models to:

        - resize the input embeddings,
        - prune heads in the self-attention heads.

    Class attributes (overridden by derived classes):

        - **config_class** ([`PretrainedConfig`]) -- A subclass of [`PretrainedConfig`] to use as configuration class
          for this model architecture.
        - **load_tf_weights** (`Callable`) -- A python *method* for loading a TensorFlow checkpoint in a PyTorch model,
          taking as arguments:

            - **model** ([`PreTrainedModel`]) -- An instance of the model on which to load the TensorFlow checkpoint.
            - **config** ([`PreTrainedConfig`]) -- An instance of the configuration associated to the model.
            - **path** (`str`) -- A path to the TensorFlow checkpoint.

        - **base_model_prefix** (`str`) -- A string indicating the attribute associated to the base model in derived
          classes of the same architecture adding modules on top of the base model.
        - **is_parallelizable** (`bool`) -- A flag indicating whether this model supports model parallelization.
        - **main_input_name** (`str`) -- The name of the principal input to the model (often `input_ids` for NLP
          models, `pixel_values` for vision models and `input_values` for speech models).
    """

    config_class = None
    base_model_prefix = ""
    main_input_name = "input_ids"
    model_tags = None

    _auto_class = None
    _no_split_modules = None
    _skip_keys_device_placement = None
    _keep_in_fp32_modules = None

    # a list of `re` patterns of `state_dict` keys that should be removed from the list of missing
    # keys we find (keys inside the model but not in the checkpoint) and avoid unnecessary warnings.
    _keys_to_ignore_on_load_missing = None
    # a list of `re` patterns of `state_dict` keys that should be removed from the list of
    # unexpected keys we find (keys inside the checkpoint but not the model) and avoid unnecessary
    # warnings.
    _keys_to_ignore_on_load_unexpected = None
    # a list of `state_dict` keys to ignore when saving the model (useful for keys that aren't
    # trained, but which are either deterministic or tied variables)
    _keys_to_ignore_on_save = None
    # a list of `state_dict` keys that are potentially tied to another key in the state_dict.
    _tied_weights_keys = None

    is_parallelizable = False
    supports_gradient_checkpointing = False
    _is_stateful = False

    # Flash Attention 2 support
    _supports_flash_attn_2 = False

    # SDPA support
    _supports_sdpa = False

    # Flex Attention support
    _supports_flex_attn = False

    # Has support for a `Cache` instance as `past_key_values`? Does it support a `StaticCache`?
    _supports_cache_class = False
    _supports_static_cache = False

    # Has support for a `QuantoQuantizedCache` instance as `past_key_values`
    _supports_quantized_cache = False

    # A tensor parallel plan to be applied to the model when TP is enabled. For
    # top-level models, this attribute is currently defined in respective model
    # code. For base models, this attribute comes from
    # `config.base_model_tp_plan` during `post_init`.
    _tp_plan = None

    @property
    def dummy_inputs(self) -> Dict[str, torch.Tensor]:
        """
        `Dict[str, torch.Tensor]`: Dummy inputs to do a forward pass in the network.
        """
        return {"input_ids": torch.tensor(DUMMY_INPUTS)}

    @property
    def framework(self) -> str:
        """
        :str: Identifies that this is a PyTorch model.
        """
        return "pt"

    def __init__(self, config: PretrainedConfig, *inputs, **kwargs):
        super().__init__()
        if not isinstance(config, PretrainedConfig):
            raise ValueError(
                f"Parameter config in `{self.__class__.__name__}(config)` should be an instance of class "
                "`PretrainedConfig`. To create a model from a pretrained model use "
                f"`model = {self.__class__.__name__}.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        if not getattr(config, "_attn_implementation_autoset", False):
            # config usually has a `torch_dtype` but we need the next line for the `no_super_init` tests
            dtype = config.torch_dtype if hasattr(config, "torch_dtype") else torch.get_default_dtype()
            config = self._autoset_attn_implementation(config, torch_dtype=dtype, check_device_map=False)
        self.config = config

        # for initialization of the loss
        loss_type = self.__class__.__name__
        if loss_type not in LOSS_MAPPING:
            loss_groups = f"({'|'.join(LOSS_MAPPING)})"
            loss_type = re.findall(loss_groups, self.__class__.__name__)
            if len(loss_type) > 0:
                loss_type = loss_type[0]
            else:
                loss_type = None
        self.loss_type = loss_type

        self.name_or_path = config.name_or_path
        self.warnings_issued = {}
        self.generation_config = GenerationConfig.from_model_config(config) if self.can_generate() else None
        # Overwrite the class attribute to make it an instance attribute, so models like
        # `InstructBlipForConditionalGeneration` can dynamically update it without modifying the class attribute
        # when a different component (e.g. language_model) is used.
        self._keep_in_fp32_modules = copy.copy(self.__class__._keep_in_fp32_modules)

    def post_init(self):
        """
        A method executed at the end of each Transformer model initialization, to execute code that needs the model's
        modules properly initialized (such as weight initialization).
        """
        self.init_weights()
        self._backward_compatibility_gradient_checkpointing()
        # If current model is a base model, attach `base_model_tp_plan` from config
        if self.base_model is self:
            self._tp_plan = self.config.base_model_tp_plan

    def dequantize(self):
        """
        Potentially dequantize the model in case it has been quantized by a quantization method that support
        dequantization.
        """
        hf_quantizer = getattr(self, "hf_quantizer", None)

        if hf_quantizer is None:
            raise ValueError("You need to first quantize your model in order to dequantize it")

        return hf_quantizer.dequantize(self)

    def _backward_compatibility_gradient_checkpointing(self):
        if self.supports_gradient_checkpointing and getattr(self.config, "gradient_checkpointing", False):
            self.gradient_checkpointing_enable()
            # Remove the attribute now that is has been consumed, so it's no saved in the config.
            delattr(self.config, "gradient_checkpointing")

    def add_model_tags(self, tags: Union[List[str], str]) -> None:
        r"""
        Add custom tags into the model that gets pushed to the Hugging Face Hub. Will
        not overwrite existing tags in the model.

        Args:
            tags (`Union[List[str], str]`):
                The desired tags to inject in the model

        Examples:

        ```python
        from transformers import AutoModel

        model = AutoModel.from_pretrained("google-bert/bert-base-cased")

        model.add_model_tags(["custom", "custom-bert"])

        # Push the model to your namespace with the name "my-custom-bert".
        model.push_to_hub("my-custom-bert")
        ```
        """
        if isinstance(tags, str):
            tags = [tags]

        if self.model_tags is None:
            self.model_tags = []

        for tag in tags:
            if tag not in self.model_tags:
                self.model_tags.append(tag)

    @classmethod
    def _from_config(cls, config, **kwargs):
        """
        All context managers that the model should be initialized under go here.

        Args:
            torch_dtype (`torch.dtype`, *optional*):
                Override the default `torch.dtype` and load the model under this dtype.
        """
        # when we init a model from within another model (e.g. VLMs) and dispatch on FA2
        # a warning is raised that dtype should be fp16. Since we never pass dtype from within
        # modeling code, we can try to infer it here same way as done in `from_pretrained`
        torch_dtype = kwargs.pop("torch_dtype", config.torch_dtype)
        if isinstance(torch_dtype, str):
            torch_dtype = getattr(torch, torch_dtype)

        use_flash_attention_2 = kwargs.pop("use_flash_attention_2", False)

        # override default dtype if needed
        dtype_orig = None
        if torch_dtype is not None:
            dtype_orig = cls._set_default_torch_dtype(torch_dtype)

        config = copy.deepcopy(config)  # We do not want to modify the config inplace in _from_config.

        if config._attn_implementation_internal is not None:
            # In this case, the config has been created with the attn_implementation set by the user, which we
            # should respect.
            attn_implementation = config._attn_implementation_internal
        else:
            attn_implementation = None

        config._attn_implementation = kwargs.pop("attn_implementation", attn_implementation)
        if not getattr(config, "_attn_implementation_autoset", False):
            config = cls._autoset_attn_implementation(
                config,
                use_flash_attention_2=use_flash_attention_2,
                check_device_map=False,
                torch_dtype=torch_dtype,
            )

        if is_deepspeed_zero3_enabled() and not _is_quantized and not _is_ds_init_called:
            import deepspeed

            logger.info("Detected DeepSpeed ZeRO-3: activating zero.init() for this model")
            # this immediately partitions the model across all gpus, to avoid the overhead in time
            # and memory copying it on CPU or each GPU first
            init_contexts = [deepspeed.zero.Init(config_dict_or_path=deepspeed_config()), set_zero3_state()]
            with ContextManagers(init_contexts):
                model = cls(config, **kwargs)

        else:
            model = cls(config, **kwargs)

        # restore default dtype if it was modified
        if dtype_orig is not None:
            torch.set_default_dtype(dtype_orig)

        return model

    @classmethod
    def _autoset_attn_implementation(
        cls,
        config,
        use_flash_attention_2: bool = False,
        torch_dtype: Optional[torch.dtype] = None,
        device_map: Optional[Union[str, Dict[str, int]]] = None,
        check_device_map: bool = True,
    ):
        """
        Automatically checks and dispatches to a default attention implementation. In order of priority:
            1. An implementation specified in `config._attn_implementation` (due for example to the argument attn_implementation="sdpa" in from_pretrained).
            2. DEPRECATED: if use_flash_attention_2 is set to `True` and `flash_attn` is available, flash attention. (`LlamaFlashAttention` for example)
            3. SDPA implementation, if available and supported by the model type. (`LlamaSdpaAttention` for example)
            4. The default model's implementation otherwise (`LlamaAttention` for example) .
        """
        # Here we use config._attn_implementation_internal to check whether the attention implementation was explicitely set by the user.
        # The property `PretrainedConfig._attn_implementation` is never `None`, for backward compatibility (always fall back on "eager").
        # The `hasattr` here is used as some Transformers tests for some reason do not call PretrainedConfig __init__ (e.g. test_no_super_init_config_and_model)
        requested_attn_implementation = None
        if hasattr(config, "_attn_implementation_internal") and config._attn_implementation_internal is not None:
            if config._attn_implementation != "flash_attention_2" and use_flash_attention_2:
                raise ValueError(
                    f'Both attn_implementation="{config._attn_implementation}" and `use_flash_attention_2=True` were used when loading the model, which are not compatible.'
                    ' We recommend to just use `attn_implementation="flash_attention_2"` when loading the model.'
                )

            if not isinstance(config._attn_implementation, dict) and config._attn_implementation not in [
                "eager"
            ] + list(ALL_ATTENTION_FUNCTIONS.keys()):
                message = f'Specified `attn_implementation="{config._attn_implementation}"` is not supported. The only possible arguments are `attn_implementation="eager"` (manual attention implementation)'
                if cls._supports_flash_attn_2:
                    message += ', `"attn_implementation=flash_attention_2"` (implementation using flash attention 2)'
                if cls._supports_sdpa:
                    message += ', `"attn_implementation=sdpa"` (implementation using torch.nn.functional.scaled_dot_product_attention)'
                if cls._supports_flex_attn:
                    message += (
                        ', `"attn_implementation=flex_attention"` (implementation using torch\'s flex_attention)'
                    )
                raise ValueError(message + ".")

            # If a config is passed with a preset attn_implementation, we skip the automatic dispatch and use the user-provided config, with hard checks that the requested attention implementation is available.
            requested_attn_implementation = config._attn_implementation_internal

        # Composite models consisting of several PretrainedModels have to specify attention impl as a dict
        # where keys are sub-config names. But most people will specify one `str` which means that should dispatch it
        # for all sub-models.
        # Below we check if a config is composite and manually prepare a dict of attn impl if not already passed as a dict.
        # Later each sub-module will dispatch with its own attn impl, by calling `XXXModel._from_config(config.text_config)`
        # If any of sub-modules doesn't support requested attn, an error will be raised. See https://github.com/huggingface/transformers/pull/32238
        for key in config.sub_configs.keys():
            sub_config = getattr(config, key)
            curr_attn_implementation = (
                requested_attn_implementation
                if not isinstance(requested_attn_implementation, dict)
                else requested_attn_implementation.get(key, None)
            )
            sub_config._attn_implementation_internal = curr_attn_implementation

        if use_flash_attention_2:
            logger.warning_once(
                'The model was loaded with use_flash_attention_2=True, which is deprecated and may be removed in a future release. Please use `attn_implementation="flash_attention_2"` instead.'
            )
            config._attn_implementation = "flash_attention_2"

        if config._attn_implementation == "flash_attention_2":
            cls._check_and_enable_flash_attn_2(
                config,
                torch_dtype=torch_dtype,
                device_map=device_map,
                hard_check_only=False,
                check_device_map=check_device_map,
            )
        elif requested_attn_implementation == "flex_attention":
            config = cls._check_and_enable_flex_attn(config, hard_check_only=True)
        elif requested_attn_implementation in [None, "sdpa"] and not is_torch_xla_available():
            # use_flash_attention_2 takes priority over SDPA, hence SDPA treated in this elif.
            config = cls._check_and_enable_sdpa(
                config,
                hard_check_only=False if requested_attn_implementation is None else True,
            )

            if (
                torch.version.hip is not None
                and config._attn_implementation == "sdpa"
                and torch.cuda.device_count() > 1
            ):
                logger.warning_once(
                    "Using the `SDPA` attention implementation on multi-gpu setup with ROCM may lead to performance issues due to the FA backend. Disabling it to use alternative backends."
                )
                torch.backends.cuda.enable_flash_sdp(False)
        elif requested_attn_implementation in list(ALL_ATTENTION_FUNCTIONS.keys()):
            config._attn_implementation = requested_attn_implementation
        elif isinstance(requested_attn_implementation, dict):
            config._attn_implementation = None
        else:
            config._attn_implementation = "eager"

        config._attn_implementation_autoset = True
        return config

    @classmethod
    def _set_default_torch_dtype(cls, dtype: torch.dtype) -> torch.dtype:
        """
        Change the default dtype and return the previous one. This is needed when wanting to instantiate the model
        under specific dtype.

        Args:
            dtype (`torch.dtype`):
                a floating dtype to set to.

        Returns:
            `torch.dtype`: the original `dtype` that can be used to restore `torch.set_default_dtype(dtype)` if it was
            modified. If it wasn't, returns `None`.

        Note `set_default_dtype` currently only works with floating-point types and asserts if for example,
        `torch.int64` is passed. So if a non-float `dtype` is passed this functions will throw an exception.
        """
        if not dtype.is_floating_point:
            raise ValueError(
                f"Can't instantiate {cls.__name__} model under dtype={dtype} since it is not a floating point dtype"
            )

        logger.info(f"Instantiating {cls.__name__} model under default dtype {dtype}.")
        dtype_orig = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        return dtype_orig

    @property
    def base_model(self) -> nn.Module:
        """
        `torch.nn.Module`: The main body of the model.
        """
        return getattr(self, self.base_model_prefix, self)

    @classmethod
    def can_generate(cls) -> bool:
        """
        Returns whether this model can generate sequences with `.generate()`.

        Returns:
            `bool`: Whether this model can generate sequences with `.generate()`.
        """
        # Directly inherits `GenerationMixin` -> can generate
        if "GenerationMixin" in str(cls.__bases__):
            return True
        # Model class overwrites `generate` (e.g. time series models) -> can generate
        if str(cls.__name__) in str(cls.generate):
            return True
        # The class inherits from a class that can generate (recursive check) -> can generate
        for base in cls.__bases__:
            if not hasattr(base, "can_generate"):
                continue
            if "PreTrainedModel" not in str(base) and base.can_generate():
                return True
        # BC: Detects whether `prepare_inputs_for_generation` has been overwritten in the model. Prior to v4.45, this
        # was how we detected whether a model could generate.
        if "GenerationMixin" not in str(cls.prepare_inputs_for_generation):
            logger.warning_once(
                f"{cls.__name__} has generative capabilities, as `prepare_inputs_for_generation` is explicitly "
                "overwritten. However, it doesn't directly inherit from `GenerationMixin`. From 👉v4.50👈 onwards, "
                "`PreTrainedModel` will NOT inherit from `GenerationMixin`, and this model will lose the ability "
                "to call `generate` and other related functions."
                "\n  - If you're using `trust_remote_code=True`, you can get rid of this warning by loading the "
                "model with an auto class. See https://huggingface.co/docs/transformers/en/model_doc/auto#auto-classes"
                "\n  - If you are the owner of the model architecture code, please modify your model class such that "
                "it inherits from `GenerationMixin` (after `PreTrainedModel`, otherwise you'll get an exception)."
                "\n  - If you are not the owner of the model architecture class, please contact the model code owner "
                "to update it."
            )
            return True
        # Otherwise, can't generate
        return False

    @classmethod
    def _check_and_enable_flash_attn_2(
        cls,
        config,
        torch_dtype: Optional[torch.dtype] = None,
        device_map: Optional[Union[str, Dict[str, int]]] = None,
        check_device_map: bool = True,
        hard_check_only: bool = False,
    ) -> PretrainedConfig:
        """
        Checks the availability of Flash Attention 2 and compatibility with the current model.

        If all checks pass and `hard_check_only` is False, the method will set the config attribute `attn_implementation` to "flash_attention_2" so that the model can initialize the correct attention module.
        """
        if not cls._supports_flash_attn_2:
            raise ValueError(
                f"{cls.__name__} does not support Flash Attention 2.0 yet. Please request to add support where"
                f" the model is hosted, on its model hub page: https://huggingface.co/{config._name_or_path}/discussions/new"
                " or in the Transformers GitHub repo: https://github.com/huggingface/transformers/issues/new"
            )

        if not is_flash_attn_2_available():
            preface = "FlashAttention2 has been toggled on, but it cannot be used due to the following error:"
            install_message = "Please refer to the documentation of https://huggingface.co/docs/transformers/perf_infer_gpu_one#flashattention-2 to install Flash Attention 2."

            if importlib.util.find_spec("flash_attn") is None:
                raise ImportError(f"{preface} the package flash_attn seems to be not installed. {install_message}")

            flash_attention_version = version.parse(importlib.metadata.version("flash_attn"))
            if torch.version.cuda:
                if flash_attention_version < version.parse("2.1.0"):
                    raise ImportError(
                        f"{preface} you need flash_attn package version to be greater or equal than 2.1.0. Detected version {flash_attention_version}. {install_message}"
                    )
                elif not torch.cuda.is_available():
                    raise ValueError(
                        f"{preface} Flash Attention 2 is not available on CPU. Please make sure torch can access a CUDA device."
                    )
                else:
                    raise ImportError(f"{preface} Flash Attention 2 is not available. {install_message}")
            elif torch.version.hip:
                if flash_attention_version < version.parse("2.0.4"):
                    raise ImportError(
                        f"{preface} you need flash_attn package version to be greater or equal than 2.0.4. Make sure to have that version installed - detected version {flash_attention_version}. {install_message}"
                    )
                else:
                    raise ImportError(f"{preface} Flash Attention 2 is not available. {install_message}")

        _is_bettertransformer = getattr(cls, "use_bettertransformer", False)

        if _is_bettertransformer:
            raise ValueError(
                "Flash Attention 2 and BetterTransformer API are not compatible. Please make sure to disable BetterTransformers by doing model.reverse_bettertransformer()"
            )

        if torch_dtype is None:
            logger.warning_once(
                "You are attempting to use Flash Attention 2.0 without specifying a torch dtype. This might lead to unexpected behaviour"
            )
        elif torch_dtype is not None and torch_dtype not in [torch.float16, torch.bfloat16]:
            logger.warning_once(
                "Flash Attention 2.0 only supports torch.float16 and torch.bfloat16 dtypes, but"
                f" the current dype in {cls.__name__} is {torch_dtype}. You should run training or inference using Automatic Mixed-Precision via the `with torch.autocast(device_type='torch_device'):` decorator,"
                ' or load the model with the `torch_dtype` argument. Example: `model = AutoModel.from_pretrained("openai/whisper-tiny", attn_implementation="flash_attention_2", torch_dtype=torch.float16)`'
            )

        # The check `torch.empty(0).device.type != "cuda"` is needed as the model may be initialized after `torch.set_default_device` has been called,
        # or the model may be initialized under the context manager `with torch.device("cuda"):`.
        if check_device_map and device_map is None and torch.empty(0).device.type != "cuda":
            if torch.cuda.is_available():
                logger.warning_once(
                    "You are attempting to use Flash Attention 2.0 with a model not initialized on GPU. Make sure to move the model to GPU"
                    " after initializing it on CPU with `model.to('cuda')`."
                )
            else:
                raise ValueError(
                    "You are attempting to use Flash Attention 2.0 with a model not initialized on GPU and with no GPU available. "
                    "This is not supported yet. Please make sure to have access to a GPU and either initialise the model on a GPU by passing a device_map "
                    "or initialising the model on CPU and then moving it to GPU."
                )
        elif (
            check_device_map
            and device_map is not None
            and isinstance(device_map, dict)
            and ("cpu" in device_map.values() or "disk" in device_map.values())
        ):
            raise ValueError(
                "You are attempting to use Flash Attention 2.0 with a model dispatched on CPU or disk. This is not supported. Please make sure to "
                "initialise the model on a GPU by passing a device_map that contains only GPU devices as keys."
            )
        if not hard_check_only:
            config._attn_implementation = "flash_attention_2"
        return config

    @classmethod
    def _check_and_enable_sdpa(cls, config, hard_check_only: bool = False) -> PretrainedConfig:
        """
        Checks the availability of SDPA for a given model.

        If all checks pass and `hard_check_only` is False, the method will set the config attribute `_attn_implementation` to "sdpa" so that the model can initialize the correct attention module.
        """
        if hard_check_only:
            if not cls._supports_sdpa:
                raise ValueError(
                    f"{cls.__name__} does not support an attention implementation through torch.nn.functional.scaled_dot_product_attention yet."
                    " Please request the support for this architecture: https://github.com/huggingface/transformers/issues/28005. If you believe"
                    ' this error is a bug, please open an issue in Transformers GitHub repository and load your model with the argument `attn_implementation="eager"` meanwhile. Example: `model = AutoModel.from_pretrained("openai/whisper-tiny", attn_implementation="eager")`'
                )
            if not is_torch_sdpa_available():
                raise ImportError(
                    "PyTorch SDPA requirements in Transformers are not met. Please install torch>=2.1.1."
                )

        if not is_torch_sdpa_available() or not cls._supports_sdpa:
            return config

        _is_bettertransformer = getattr(cls, "use_bettertransformer", False)
        if _is_bettertransformer:
            return config

        if not hard_check_only:
            config._attn_implementation = "sdpa"
        return config

    @classmethod
    def _check_and_enable_flex_attn(cls, config, hard_check_only: bool = False) -> PretrainedConfig:
        """
        Checks the availability of Flex Attention for a given model.

        If all checks pass and `hard_check_only` is False, the method will set the config attribute `_attn_implementation` to "flex_attention" so that the model can initialize the correct attention module.
        """
        if hard_check_only:
            if not cls._supports_flex_attn:
                raise ValueError(
                    f"{cls.__name__} does not support an attention implementation through torch's flex_attention."
                    " Please request the support for this architecture: https://github.com/huggingface/transformers/issues/34809."
                    " If you believe this error is a bug, please open an issue in Transformers GitHub repository"
                    ' and load your model with the argument `attn_implementation="eager"` meanwhile.'
                    ' Example: `model = AutoModel.from_pretrained("openai/whisper-tiny", attn_implementation="eager")`'
                )
            if not is_torch_flex_attn_available():
                raise ImportError(
                    "PyTorch Flex Attention requirements in Transformers are not met. Please install torch>=2.5.0."
                )

        if not is_torch_flex_attn_available() or not cls._supports_flex_attn:
            return config

        if not hard_check_only:
            config._attn_implementation = "flex_attention"

        return config

    def enable_input_require_grads(self):
        """
        Enables the gradients for the input embeddings. This is useful for fine-tuning adapter weights while keeping
        the model weights fixed.
        """

        def make_inputs_require_grads(module, input, output):
            output.requires_grad_(True)

        self._require_grads_hook = self.get_input_embeddings().register_forward_hook(make_inputs_require_grads)

    def disable_input_require_grads(self):
        """
        Removes the `_require_grads_hook`.
        """
        self._require_grads_hook.remove()

    def get_input_embeddings(self) -> nn.Module:
        """
        Returns the model's input embeddings.

        Returns:
            `nn.Module`: A torch module mapping vocabulary to hidden states.
        """
        base_model = getattr(self, self.base_model_prefix, self)
        if base_model is not self:
            return base_model.get_input_embeddings()
        else:
            raise NotImplementedError

    def set_input_embeddings(self, value: nn.Module):
        """
        Set model's input embeddings.

        Args:
            value (`nn.Module`): A module mapping vocabulary to hidden states.
        """
        base_model = getattr(self, self.base_model_prefix, self)
        if base_model is not self:
            base_model.set_input_embeddings(value)
        else:
            raise NotImplementedError

    def get_output_embeddings(self) -> nn.Module:
        """
        Returns the model's output embeddings.

        Returns:
            `nn.Module`: A torch module mapping hidden states to vocabulary.
        """
        return None  # Overwrite for models with output embeddings

    def _init_weights(self, module):
        """
        Initialize the weights. This method should be overridden by derived class and is
        the only initialization method that will be called when loading a checkpoint
        using `from_pretrained`. Any attempt to initialize outside of this function
        will be useless as the torch.nn.init function are all replaced with skip.
        """
        pass

    def _initialize_weights(self, module):
        """
        Initialize the weights if they are not already initialized.
        """
        if getattr(module, "_is_hf_initialized", False):
            return
        self._init_weights(module)
        module._is_hf_initialized = True

    def tie_weights(self):
        """
        Tie the weights between the input embeddings and the output embeddings.

        If the `torchscript` flag is set in the configuration, can't handle parameter sharing so we are cloning the
        weights instead.
        """
        if getattr(self.config.get_text_config(decoder=True), "tie_word_embeddings", True):
            output_embeddings = self.get_output_embeddings()
            if output_embeddings is not None:
                self._tie_or_clone_weights(output_embeddings, self.get_input_embeddings())

        if getattr(self.config, "is_encoder_decoder", False) and getattr(self.config, "tie_encoder_decoder", False):
            if hasattr(self, self.base_model_prefix):
                self = getattr(self, self.base_model_prefix)
            tied_weights = self._tie_encoder_decoder_weights(
                self.encoder, self.decoder, self.base_model_prefix, "encoder"
            )
            # Setting a dynamic variable instead of `_tied_weights_keys` because it's a class
            # attributed not an instance member, therefore modifying it will modify the entire class
            # Leading to issues on subsequent calls by different tests or subsequent calls.
            self._dynamic_tied_weights_keys = tied_weights

        for module in self.modules():
            if hasattr(module, "_tie_weights"):
                module._tie_weights()

    @staticmethod
    def _tie_encoder_decoder_weights(
        encoder: nn.Module, decoder: nn.Module, base_model_prefix: str, base_encoder_name: str
    ):
        uninitialized_encoder_weights: List[str] = []
        tied_weights: List[str] = []
        if decoder.__class__ != encoder.__class__:
            logger.info(
                f"{decoder.__class__} and {encoder.__class__} are not equal. In this case make sure that all encoder"
                " weights are correctly initialized."
            )

        def tie_encoder_to_decoder_recursively(
            decoder_pointer: nn.Module,
            encoder_pointer: nn.Module,
            module_name: str,
            base_encoder_name: str,
            uninitialized_encoder_weights: List[str],
            depth=0,
            total_decoder_name="",
            total_encoder_name="",
        ):
            assert isinstance(decoder_pointer, nn.Module) and isinstance(
                encoder_pointer, nn.Module
            ), f"{decoder_pointer} and {encoder_pointer} have to be of type nn.Module"
            if hasattr(decoder_pointer, "weight"):
                assert hasattr(encoder_pointer, "weight")
                encoder_pointer.weight = decoder_pointer.weight
                tied_weights.append(f"{base_encoder_name}{total_encoder_name}.weight")
                if hasattr(decoder_pointer, "bias"):
                    assert hasattr(encoder_pointer, "bias")
                    tied_weights.append(f"{base_encoder_name}{total_encoder_name}.bias")
                    encoder_pointer.bias = decoder_pointer.bias
                return

            encoder_modules = encoder_pointer._modules
            decoder_modules = decoder_pointer._modules
            if len(decoder_modules) > 0:
                assert (
                    len(encoder_modules) > 0
                ), f"Encoder module {encoder_pointer} does not match decoder module {decoder_pointer}"

                all_encoder_weights = {module_name + "/" + sub_name for sub_name in encoder_modules.keys()}
                encoder_layer_pos = 0
                for name, module in decoder_modules.items():
                    if name.isdigit():
                        encoder_name = str(int(name) + encoder_layer_pos)
                        decoder_name = name
                        if not isinstance(decoder_modules[decoder_name], type(encoder_modules[encoder_name])) and len(
                            encoder_modules
                        ) != len(decoder_modules):
                            # this can happen if the name corresponds to the position in a list module list of layers
                            # in this case the decoder has added a cross-attention that the encoder does not have
                            # thus skip this step and subtract one layer pos from encoder
                            encoder_layer_pos -= 1
                            continue
                    elif name not in encoder_modules:
                        continue
                    elif depth > 500:
                        raise ValueError(
                            "Max depth of recursive function `tie_encoder_to_decoder` reached. It seems that there is"
                            " a circular dependency between two or more `nn.Modules` of your model."
                        )
                    else:
                        decoder_name = encoder_name = name
                    tie_encoder_to_decoder_recursively(
                        decoder_modules[decoder_name],
                        encoder_modules[encoder_name],
                        module_name + "/" + name,
                        base_encoder_name,
                        uninitialized_encoder_weights,
                        depth=depth + 1,
                        total_encoder_name=f"{total_encoder_name}.{encoder_name}",
                        total_decoder_name=f"{total_decoder_name}.{decoder_name}",
                    )
                    all_encoder_weights.remove(module_name + "/" + encoder_name)

                uninitialized_encoder_weights += list(all_encoder_weights)

        # tie weights recursively
        tie_encoder_to_decoder_recursively(
            decoder, encoder, base_model_prefix, base_encoder_name, uninitialized_encoder_weights
        )

        if len(uninitialized_encoder_weights) > 0:
            logger.warning(
                f"The following encoder weights were not tied to the decoder {uninitialized_encoder_weights}"
            )
        return tied_weights

    def _tie_or_clone_weights(self, output_embeddings, input_embeddings):
        """Tie or clone module weights depending of whether we are using TorchScript or not"""
        if self.config.torchscript:
            output_embeddings.weight = nn.Parameter(input_embeddings.weight.clone())
        else:
            output_embeddings.weight = input_embeddings.weight

        if getattr(output_embeddings, "bias", None) is not None:
            output_embeddings.bias.data = nn.functional.pad(
                output_embeddings.bias.data,
                (
                    0,
                    output_embeddings.weight.shape[0] - output_embeddings.bias.shape[0],
                ),
                "constant",
                0,
            )
        if hasattr(output_embeddings, "out_features") and hasattr(input_embeddings, "num_embeddings"):
            output_embeddings.out_features = input_embeddings.num_embeddings

    def _get_no_split_modules(self, device_map: str):
        """
        Get the modules of the model that should not be spit when using device_map. We iterate through the modules to
        get the underlying `_no_split_modules`.

        Args:
            device_map (`str`):
                The device map value. Options are ["auto", "balanced", "balanced_low_0", "sequential"]

        Returns:
            `List[str]`: List of modules that should not be split
        """
        _no_split_modules = set()
        modules_to_check = [self]
        while len(modules_to_check) > 0:
            module = modules_to_check.pop(-1)
            # if the module does not appear in _no_split_modules, we also check the children
            if module.__class__.__name__ not in _no_split_modules:
                if isinstance(module, PreTrainedModel):
                    if module._no_split_modules is None:
                        raise ValueError(
                            f"{module.__class__.__name__} does not support `device_map='{device_map}'`. To implement support, the model "
                            "class needs to implement the `_no_split_modules` attribute."
                        )
                    else:
                        _no_split_modules = _no_split_modules | set(module._no_split_modules)
                modules_to_check += list(module.children())
        return list(_no_split_modules)

    def resize_token_embeddings(
        self,
        new_num_tokens: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        mean_resizing: bool = True,
    ) -> nn.Embedding:
        """
        Resizes input token embeddings matrix of the model if `new_num_tokens != config.vocab_size`.

        Takes care of tying weights embeddings afterwards if the model class has a `tie_weights()` method.

        Arguments:
            new_num_tokens (`int`, *optional*):
                The new number of tokens in the embedding matrix. Increasing the size will add newly initialized
                vectors at the end. Reducing the size will remove vectors from the end. If not provided or `None`, just
                returns a pointer to the input tokens `torch.nn.Embedding` module of the model without doing anything.
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the embedding matrix to a multiple of the provided value.If `new_num_tokens` is set to
                `None` will just pad the embedding to a multiple of `pad_to_multiple_of`.

                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                `>= 7.5` (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128. For more
                details about this, or help on choosing the correct value for resizing, refer to this guide:
                https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc
            mean_resizing (`bool`):
                Whether to initialize the added embeddings from a multivariate normal distribution that has old embeddings' mean and
                covariance or to initialize them with a normal distribution that has a mean of zero and std equals `config.initializer_range`.

                Setting `mean_resizing` to `True` is useful when increasing the size of the embeddings of causal language models,
                where the generated tokens' probabilities won't be affected by the added embeddings because initializing the new embeddings with the
                old embeddings' mean will reduce the kl-divergence between the next token probability before and after adding the new embeddings.
                Refer to this article for more information: https://nlp.stanford.edu/~johnhew/vocab-expansion.html

        Return:
            `torch.nn.Embedding`: Pointer to the input tokens Embeddings Module of the model.
        """
        model_embeds = self._resize_token_embeddings(new_num_tokens, pad_to_multiple_of, mean_resizing)
        if new_num_tokens is None and pad_to_multiple_of is None:
            return model_embeds

        # Since we are basically resuing the same old embeddings with new weight values, gathering is required
        is_quantized = hasattr(self, "hf_quantizer") and self.hf_quantizer is not None
        if is_deepspeed_zero3_enabled() and not is_quantized:
            import deepspeed

            with deepspeed.zero.GatheredParameters(model_embeds.weight, modifier_rank=None):
                vocab_size = model_embeds.weight.shape[0]
        else:
            vocab_size = model_embeds.weight.shape[0]

        # Update base model and current model config.
        self.config.get_text_config().vocab_size = vocab_size
        self.vocab_size = vocab_size

        # Tie weights again if needed
        self.tie_weights()

        return model_embeds

    def _resize_token_embeddings(self, new_num_tokens, pad_to_multiple_of=None, mean_resizing=True):
        old_embeddings = self.get_input_embeddings()
        new_embeddings = self._get_resized_embeddings(
            old_embeddings, new_num_tokens, pad_to_multiple_of, mean_resizing
        )
        if hasattr(old_embeddings, "_hf_hook"):
            hook = old_embeddings._hf_hook
            add_hook_to_module(new_embeddings, hook)
        old_embeddings_requires_grad = old_embeddings.weight.requires_grad
        new_embeddings.requires_grad_(old_embeddings_requires_grad)
        self.set_input_embeddings(new_embeddings)
        is_quantized = hasattr(self, "hf_quantizer") and self.hf_quantizer is not None

        # Update new_num_tokens with the actual size of new_embeddings
        if pad_to_multiple_of is not None:
            if is_deepspeed_zero3_enabled() and not is_quantized:
                import deepspeed

                with deepspeed.zero.GatheredParameters(new_embeddings.weight, modifier_rank=None):
                    new_num_tokens = new_embeddings.weight.shape[0]
            else:
                new_num_tokens = new_embeddings.weight.shape[0]

        # if word embeddings are not tied, make sure that lm head is resized as well
        if (
            self.get_output_embeddings() is not None
            and not self.config.get_text_config(decoder=True).tie_word_embeddings
        ):
            old_lm_head = self.get_output_embeddings()
            if isinstance(old_lm_head, torch.nn.Embedding):
                new_lm_head = self._get_resized_embeddings(old_lm_head, new_num_tokens, mean_resizing=mean_resizing)
            else:
                new_lm_head = self._get_resized_lm_head(old_lm_head, new_num_tokens, mean_resizing=mean_resizing)
            if hasattr(old_lm_head, "_hf_hook"):
                hook = old_lm_head._hf_hook
                add_hook_to_module(new_lm_head, hook)
            old_lm_head_requires_grad = old_lm_head.weight.requires_grad
            new_lm_head.requires_grad_(old_lm_head_requires_grad)
            self.set_output_embeddings(new_lm_head)

        return self.get_input_embeddings()

    def _get_resized_embeddings(
        self,
        old_embeddings: nn.Embedding,
        new_num_tokens: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        mean_resizing: bool = True,
    ) -> nn.Embedding:
        """
        Build a resized Embedding Module from a provided token Embedding Module. Increasing the size will add newly
        initialized vectors at the end. Reducing the size will remove vectors from the end

        Args:
            old_embeddings (`torch.nn.Embedding`):
                Old embeddings to be resized.
            new_num_tokens (`int`, *optional*):
                New number of tokens in the embedding matrix.

                Increasing the size will add newly initialized vectors at the end. Reducing the size will remove
                vectors from the end. If not provided or `None`, just returns a pointer to the input tokens
                `torch.nn.Embedding` module of the model without doing anything.
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the embedding matrix to a multiple of the provided value. If `new_num_tokens` is set to
                `None` will just pad the embedding to a multiple of `pad_to_multiple_of`.

                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                `>= 7.5` (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128. For more
                details about this, or help on choosing the correct value for resizing, refer to this guide:
                https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc
            mean_resizing (`bool`):
                Whether to initialize the added embeddings from a multivariate normal distribution that has old embeddings' mean and
                covariance or to initialize them with a normal distribution that has a mean of zero and std equals `config.initializer_range`.

                Setting `mean_resizing` to `True` is useful when increasing the size of the embeddings of causal language models,
                where the generated tokens' probabilities will not be affected by the added embeddings because initializing the new embeddings with the
                old embeddings' mean will reduce the kl-divergence between the next token probability before and after adding the new embeddings.
                Refer to this article for more information: https://nlp.stanford.edu/~johnhew/vocab-expansion.html


        Return:
            `torch.nn.Embedding`: Pointer to the resized Embedding Module or the old Embedding Module if
            `new_num_tokens` is `None`
        """

        if pad_to_multiple_of is not None:
            if not isinstance(pad_to_multiple_of, int):
                raise ValueError(
                    f"Asking to pad the embedding matrix to a multiple of `{pad_to_multiple_of}`, which is not and integer. Please make sure to pass an integer"
                )
            if new_num_tokens is None:
                new_num_tokens = old_embeddings.weight.shape[0]
            new_num_tokens = ((new_num_tokens + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
        else:
            logger.info(
                "You are resizing the embedding layer without providing a `pad_to_multiple_of` parameter. This means that the new embedding"
                f" dimension will be {new_num_tokens}. This might induce some performance reduction as *Tensor Cores* will not be available."
                " For more details about this, or help on choosing the correct value for resizing, refer to this guide:"
                " https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc"
            )

        if new_num_tokens is None:
            return old_embeddings

        is_quantized = hasattr(self, "hf_quantizer") and self.hf_quantizer is not None
        if is_deepspeed_zero3_enabled() and not is_quantized:
            import deepspeed

            with deepspeed.zero.GatheredParameters(old_embeddings.weight, modifier_rank=None):
                old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
        else:
            old_num_tokens, old_embedding_dim = old_embeddings.weight.size()

        if old_num_tokens == new_num_tokens and not is_deepspeed_zero3_enabled():
            return old_embeddings

        if not isinstance(old_embeddings, nn.Embedding):
            raise TypeError(
                f"Old embeddings are of type {type(old_embeddings)}, which is not an instance of {nn.Embedding}. You"
                " should either use a different resize function or make sure that `old_embeddings` are an instance of"
                f" {nn.Embedding}."
            )

        # Build new embeddings

        # When using DeepSpeed ZeRO-3, we shouldn't create new embeddings with DeepSpeed init
        # because the shape of the new embedding layer is used across various modeling files
        # as well as to update config vocab size. Shape will be 0 when using DeepSpeed init leading
        # to errors when training.
        new_embeddings = nn.Embedding(
            new_num_tokens,
            old_embedding_dim,
            device=old_embeddings.weight.device,
            dtype=old_embeddings.weight.dtype,
        )

        if new_num_tokens > old_num_tokens and not mean_resizing:
            # initialize new embeddings (in particular added tokens) with a mean of 0 and std equals `config.initializer_range`.
            self._init_weights(new_embeddings)

        elif new_num_tokens > old_num_tokens and mean_resizing:
            # initialize new embeddings  (in particular added tokens). The new embeddings will be initialized
            # from a multivariate normal distribution that has old embeddings' mean and covariance.
            # as described in this article: https://nlp.stanford.edu/~johnhew/vocab-expansion.html
            logger.warning_once(
                "The new embeddings will be initialized from a multivariate normal distribution that has old embeddings' mean and covariance. "
                "As described in this article: https://nlp.stanford.edu/~johnhew/vocab-expansion.html. "
                "To disable this, use `mean_resizing=False`"
            )

            added_num_tokens = new_num_tokens - old_num_tokens
            if is_deepspeed_zero3_enabled() and not is_quantized:
                import deepspeed

                with deepspeed.zero.GatheredParameters([old_embeddings.weight], modifier_rank=None):
                    self._init_added_embeddings_weights_with_mean(
                        old_embeddings, new_embeddings, old_embedding_dim, old_num_tokens, added_num_tokens
                    )
            else:
                self._init_added_embeddings_weights_with_mean(
                    old_embeddings, new_embeddings, old_embedding_dim, old_num_tokens, added_num_tokens
                )

        # Copy token embeddings from the previous weights

        # numbers of tokens to copy
        n = min(old_num_tokens, new_num_tokens)

        if is_deepspeed_zero3_enabled() and not is_quantized:
            import deepspeed

            params = [old_embeddings.weight, new_embeddings.weight]
            with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                new_embeddings.weight.data[:n, :] = old_embeddings.weight.data[:n, :]
        else:
            new_embeddings.weight.data[:n, :] = old_embeddings.weight.data[:n, :]

        # Replace weights in old_embeddings and return to maintain the same embedding type.
        # This ensures correct functionality when a Custom Embedding class is passed as input.
        # The input and output embedding types remain consistent. (c.f. https://github.com/huggingface/transformers/pull/31979)
        if is_deepspeed_zero3_enabled() and not is_quantized:
            import deepspeed

            params = [old_embeddings.weight, new_embeddings.weight]
            with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                old_embeddings.weight = new_embeddings.weight
                old_embeddings.num_embeddings = new_embeddings.weight.data.shape[0]

                # If the new number of tokens is smaller than the original `padding_idx`, the `padding_idx`
                # will be set to `None` in the resized embeddings.
                if old_embeddings.padding_idx is not None and (new_num_tokens - 1) < old_embeddings.padding_idx:
                    old_embeddings.padding_idx = None
        else:
            old_embeddings.weight.data = new_embeddings.weight.data
            old_embeddings.num_embeddings = new_embeddings.weight.data.shape[0]
            if old_embeddings.padding_idx is not None and (new_num_tokens - 1) < old_embeddings.padding_idx:
                old_embeddings.padding_idx = None

        return old_embeddings

    def _get_resized_lm_head(
        self,
        old_lm_head: nn.Linear,
        new_num_tokens: Optional[int] = None,
        transposed: Optional[bool] = False,
        mean_resizing: bool = True,
    ) -> nn.Linear:
        """
        Build a resized Linear Module from a provided old Linear Module. Increasing the size will add newly initialized
        vectors at the end. Reducing the size will remove vectors from the end

        Args:
            old_lm_head (`torch.nn.Linear`):
                Old lm head liner layer to be resized.
            new_num_tokens (`int`, *optional*):
                New number of tokens in the linear matrix.

                Increasing the size will add newly initialized vectors at the end. Reducing the size will remove
                vectors from the end. If not provided or `None`, just returns a pointer to the input tokens
                `torch.nn.Linear` module of the model without doing anything. transposed (`bool`, *optional*, defaults
                to `False`): Whether `old_lm_head` is transposed or not. If True `old_lm_head.size()` is `lm_head_dim,
                vocab_size` else `vocab_size, lm_head_dim`.
            mean_resizing (`bool`):
                Whether to initialize the added embeddings from a multivariate normal distribution that has old embeddings' mean and
                covariance or to initialize them with a normal distribution that has a mean of zero and std equals `config.initializer_range`.

                Setting `mean_resizing` to `True` is useful when increasing the size of the embeddings of causal language models,
                where the generated tokens' probabilities will not be affected by the added embeddings because initializing the new embeddings with the
                old embeddings' mean will reduce the kl-divergence between the next token probability before and after adding the new embeddings.
                Refer to this article for more information: https://nlp.stanford.edu/~johnhew/vocab-expansion.html

        Return:
            `torch.nn.Linear`: Pointer to the resized Linear Module or the old Linear Module if `new_num_tokens` is
            `None`
        """
        if new_num_tokens is None:
            return old_lm_head

        is_quantized = hasattr(self, "hf_quantizer") and self.hf_quantizer is not None
        if is_deepspeed_zero3_enabled() and not is_quantized:
            import deepspeed

            with deepspeed.zero.GatheredParameters(old_lm_head.weight, modifier_rank=None):
                old_num_tokens, old_lm_head_dim = (
                    old_lm_head.weight.size() if not transposed else old_lm_head.weight.t().size()
                )
        else:
            old_num_tokens, old_lm_head_dim = (
                old_lm_head.weight.size() if not transposed else old_lm_head.weight.t().size()
            )

        if old_num_tokens == new_num_tokens and not is_deepspeed_zero3_enabled():
            return old_lm_head

        if not isinstance(old_lm_head, nn.Linear):
            raise TypeError(
                f"Old language model head is of type {type(old_lm_head)}, which is not an instance of {nn.Linear}. You"
                " should either use a different resize function or make sure that `old_lm_head` are an instance of"
                f" {nn.Linear}."
            )

        # Build new lm head
        new_lm_head_shape = (old_lm_head_dim, new_num_tokens) if not transposed else (new_num_tokens, old_lm_head_dim)
        has_new_lm_head_bias = old_lm_head.bias is not None

        # When using DeepSpeed ZeRO-3, we shouldn't create new embeddings with DeepSpeed init
        # because the shape of the new embedding layer is used across various modeling files
        # as well as to update config vocab size. Shape will be 0 when using DeepSpeed init leading
        # to errors when training.
        new_lm_head = nn.Linear(
            *new_lm_head_shape,
            bias=has_new_lm_head_bias,
            device=old_lm_head.weight.device,
            dtype=old_lm_head.weight.dtype,
        )

        if new_num_tokens > old_num_tokens and not mean_resizing:
            # initialize new embeddings (in particular added tokens) with a mean of 0 and std equals `config.initializer_range`.
            self._init_weights(new_lm_head)

        elif new_num_tokens > old_num_tokens and mean_resizing:
            # initialize new lm_head weights (in particular added tokens). The new lm_head weights
            # will be initialized from a multivariate normal distribution that has old embeddings' mean and covariance.
            # as described in this article: https://nlp.stanford.edu/~johnhew/vocab-expansion.html
            logger.warning_once(
                "The new lm_head weights will be initialized from a multivariate normal distribution that has old embeddings' mean and covariance. "
                "As described in this article: https://nlp.stanford.edu/~johnhew/vocab-expansion.html. "
                "To disable this, use `mean_resizing=False`"
            )

            added_num_tokens = new_num_tokens - old_num_tokens
            if is_deepspeed_zero3_enabled() and not is_quantized:
                import deepspeed

                params = [old_lm_head.weight]
                if has_new_lm_head_bias:
                    params += [old_lm_head.bias]
                with deepspeed.zero.GatheredParameters(params, modifier_rank=None):
                    self._init_added_lm_head_weights_with_mean(
                        old_lm_head, new_lm_head, old_lm_head_dim, old_num_tokens, added_num_tokens, transposed
                    )
                    if has_new_lm_head_bias:
                        self._init_added_lm_head_bias_with_mean(old_lm_head, new_lm_head, added_num_tokens)

            else:
                self._init_added_lm_head_weights_with_mean(
                    old_lm_head, new_lm_head, old_lm_head_dim, old_num_tokens, added_num_tokens, transposed
                )
                if has_new_lm_head_bias:
                    self._init_added_lm_head_bias_with_mean(old_lm_head, new_lm_head, added_num_tokens)

        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)

        if is_deepspeed_zero3_enabled() and not is_quantized:
            import deepspeed

            params = [old_lm_head.weight, old_lm_head.bias, new_lm_head.weight, new_lm_head.bias]
            with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                self._copy_lm_head_original_to_resized(
                    new_lm_head, old_lm_head, num_tokens_to_copy, transposed, has_new_lm_head_bias
                )
        else:
            self._copy_lm_head_original_to_resized(
                new_lm_head, old_lm_head, num_tokens_to_copy, transposed, has_new_lm_head_bias
            )

        return new_lm_head

    def _init_added_embeddings_weights_with_mean(
        self, old_embeddings, new_embeddings, old_embedding_dim, old_num_tokens, added_num_tokens
    ):
        old_embeddings_weight = old_embeddings.weight.data.to(torch.float32)
        mean_embeddings = torch.mean(old_embeddings_weight, axis=0)
        old_centered_embeddings = old_embeddings_weight - mean_embeddings
        covariance = old_centered_embeddings.T @ old_centered_embeddings / old_num_tokens

        # Check if the covariance is positive definite.
        eigenvalues = torch.linalg.eigvals(covariance)
        is_covariance_psd = bool(
            (covariance == covariance.T).all() and not torch.is_complex(eigenvalues) and (eigenvalues > 0).all()
        )
        if is_covariance_psd:
            # If covariances is positive definite, a distribution can be created. and we can sample new weights from it.
            distribution = torch.distributions.multivariate_normal.MultivariateNormal(
                mean_embeddings, covariance_matrix=1e-9 * covariance
            )
            new_embeddings.weight.data[-1 * added_num_tokens :, :] = distribution.sample(
                sample_shape=(added_num_tokens,)
            ).to(old_embeddings.weight.dtype)
        else:
            # Otherwise, just initialize with the mean. because distribtion will not be created.
            new_embeddings.weight.data[-1 * added_num_tokens :, :] = (
                mean_embeddings[None, :].repeat(added_num_tokens, 1).to(old_embeddings.weight.dtype)
            )

    def _init_added_lm_head_weights_with_mean(
        self,
        old_lm_head,
        new_lm_head,
        old_lm_head_dim,
        old_num_tokens,
        added_num_tokens,
        transposed=False,
    ):
        if transposed:
            # Transpose to the desired shape for the function.
            new_lm_head.weight.data = new_lm_head.weight.data.T
            old_lm_head.weight.data = old_lm_head.weight.data.T

        # The same initilization logic as Embeddings.
        self._init_added_embeddings_weights_with_mean(
            old_lm_head, new_lm_head, old_lm_head_dim, old_num_tokens, added_num_tokens
        )

        if transposed:
            # Transpose again to the correct shape.
            new_lm_head.weight.data = new_lm_head.weight.data.T
            old_lm_head.weight.data = old_lm_head.weight.data.T

    def _init_added_lm_head_bias_with_mean(self, old_lm_head, new_lm_head, added_num_tokens):
        bias_mean = torch.mean(old_lm_head.bias.data, axis=0, dtype=torch.float32)
        bias_std = torch.std(old_lm_head.bias.data, axis=0).to(torch.float32)
        new_lm_head.bias.data[-1 * added_num_tokens :].normal_(mean=bias_mean, std=1e-9 * bias_std)

    def _copy_lm_head_original_to_resized(
        self, new_lm_head, old_lm_head, num_tokens_to_copy, transposed, has_new_lm_head_bias
    ):
        # Copy old lm head weights to new lm head
        if not transposed:
            new_lm_head.weight.data[:num_tokens_to_copy, :] = old_lm_head.weight.data[:num_tokens_to_copy, :]
        else:
            new_lm_head.weight.data[:, :num_tokens_to_copy] = old_lm_head.weight.data[:, :num_tokens_to_copy]

        # Copy bias weights to new lm head
        if has_new_lm_head_bias:
            new_lm_head.bias.data[:num_tokens_to_copy] = old_lm_head.bias.data[:num_tokens_to_copy]

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        raise NotImplementedError(
            f"`resize_position_embeddings` is not implemented for {self.__class__}`. To implement it, you should "
            f"overwrite this method in the class {self.__class__} in `modeling_{self.__class__.__module__}.py`"
        )

    def get_position_embeddings(self) -> Union[nn.Embedding, Tuple[nn.Embedding]]:
        raise NotImplementedError(
            f"`get_position_embeddings` is not implemented for {self.__class__}`. To implement it, you should "
            f"overwrite this method in the class {self.__class__} in `modeling_{self.__class__.__module__}.py`"
        )

    def init_weights(self):
        """
        If needed prunes and maybe initializes weights. If using a custom `PreTrainedModel`, you need to implement any
        initialization logic in `_init_weights`.
        """
        # Prune heads if needed
        if self.config.pruned_heads:
            self.prune_heads(self.config.pruned_heads)

        if _init_weights:
            # Initialize weights
            self.apply(self._initialize_weights)

            # Tie weights should be skipped when not initializing all weights
            # since from_pretrained(...) calls tie weights anyways
            self.tie_weights()

    def prune_heads(self, heads_to_prune: Dict[int, List[int]]):
        """
        Prunes heads of the base model.

        Arguments:
            heads_to_prune (`Dict[int, List[int]]`):
                Dictionary with keys being selected layer indices (`int`) and associated values being the list of heads
                to prune in said layer (list of `int`). For instance {1: [0, 2], 2: [2, 3]} will prune heads 0 and 2 on
                layer 1 and heads 2 and 3 on layer 2.
        """
        # save new sets of pruned heads as union of previously stored pruned heads and newly pruned heads
        for layer, heads in heads_to_prune.items():
            union_heads = set(self.config.pruned_heads.get(layer, [])) | set(heads)
            self.config.pruned_heads[layer] = list(union_heads)  # Unfortunately we have to store it as list for JSON

        self.base_model._prune_heads(heads_to_prune)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """
        Activates gradient checkpointing for the current model.

        Note that in other frameworks this feature can be referred to as "activation checkpointing" or "checkpoint
        activations".

        We pass the `__call__` method of the modules instead of `forward` because `__call__` attaches all the hooks of
        the module. https://discuss.pytorch.org/t/any-different-between-model-input-and-model-forward-input/3690/2

        Args:
            gradient_checkpointing_kwargs (dict, *optional*):
                Additional keyword arguments passed along to the `torch.utils.checkpoint.checkpoint` function.
        """
        if not self.supports_gradient_checkpointing:
            raise ValueError(f"{self.__class__.__name__} does not support gradient checkpointing.")

        if gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = {"use_reentrant": True}

        gradient_checkpointing_func = functools.partial(checkpoint, **gradient_checkpointing_kwargs)

        # For old GC format (transformers < 4.35.0) for models that live on the Hub
        # we will fall back to the overwritten `_set_gradient_checkpointing` method
        _is_using_old_format = "value" in inspect.signature(self._set_gradient_checkpointing).parameters

        if not _is_using_old_format:
            self._set_gradient_checkpointing(enable=True, gradient_checkpointing_func=gradient_checkpointing_func)
        else:
            self.apply(partial(self._set_gradient_checkpointing, value=True))
            logger.warning(
                "You are using an old version of the checkpointing format that is deprecated (We will also silently ignore `gradient_checkpointing_kwargs` in case you passed it)."
                "Please update to the new format on your modeling file. To use the new format, you need to completely remove the definition of the method `_set_gradient_checkpointing` in your model."
            )

        if getattr(self, "_hf_peft_config_loaded", False):
            # When using PEFT + gradient checkpointing + Trainer we need to make sure the input has requires_grad=True
            # we do it also on PEFT: https://github.com/huggingface/peft/blob/85013987aa82aa1af3da1236b6902556ce3e483e/src/peft/peft_model.py#L334
            # When training with PEFT, only LoRA layers will have requires grad set to True, but the output of frozen layers need to propagate
            # the gradients to make sure the gradient flows.
            self.enable_input_require_grads()

    def _set_gradient_checkpointing(self, enable: bool = True, gradient_checkpointing_func: Callable = checkpoint):
        is_gradient_checkpointing_set = False

        # Apply it on the top-level module in case the top-level modules supports it
        # for example, LongT5Stack inherits from `PreTrainedModel`.
        if hasattr(self, "gradient_checkpointing"):
            self._gradient_checkpointing_func = gradient_checkpointing_func
            self.gradient_checkpointing = enable
            is_gradient_checkpointing_set = True

        for module in self.modules():
            if hasattr(module, "gradient_checkpointing"):
                module._gradient_checkpointing_func = gradient_checkpointing_func
                module.gradient_checkpointing = enable
                is_gradient_checkpointing_set = True

        if not is_gradient_checkpointing_set:
            raise ValueError(
                f"{self.__class__.__name__} is not compatible with gradient checkpointing. Make sure all the architecture support it by setting a boolean attribute"
                " `gradient_checkpointing` to modules of the model that uses checkpointing."
            )

    def gradient_checkpointing_disable(self):
        """
        Deactivates gradient checkpointing for the current model.

        Note that in other frameworks this feature can be referred to as "activation checkpointing" or "checkpoint
        activations".
        """
        if self.supports_gradient_checkpointing:
            # For old GC format (transformers < 4.35.0) for models that live on the Hub
            # we will fall back to the overwritten `_set_gradient_checkpointing` methid
            _is_using_old_format = "value" in inspect.signature(self._set_gradient_checkpointing).parameters
            if not _is_using_old_format:
                self._set_gradient_checkpointing(enable=False)
            else:
                logger.warning(
                    "You are using an old version of the checkpointing format that is deprecated (We will also silently ignore `gradient_checkpointing_kwargs` in case you passed it)."
                    "Please update to the new format on your modeling file. To use the new format, you need to completely remove the definition of the method `_set_gradient_checkpointing` in your model."
                )
                self.apply(partial(self._set_gradient_checkpointing, value=False))

        if getattr(self, "_hf_peft_config_loaded", False):
            self.disable_input_require_grads()

    @property
    def is_gradient_checkpointing(self) -> bool:
        """
        Whether gradient checkpointing is activated for this model or not.

        Note that in other frameworks this feature can be referred to as "activation checkpointing" or "checkpoint
        activations".
        """
        return any(hasattr(m, "gradient_checkpointing") and m.gradient_checkpointing for m in self.modules())

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        state_dict: Optional[dict] = None,
        save_function: Callable = torch.save,
        push_to_hub: bool = False,
        max_shard_size: Union[int, str] = "5GB",
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        token: Optional[Union[str, bool]] = None,
        save_peft_format: bool = True,
        **kwargs,
    ):
        """
        Save a model and its configuration file to a directory, so that it can be re-loaded using the
        [`~PreTrainedModel.from_pretrained`] class method.

        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to which to save. Will be created if it doesn't exist.
            is_main_process (`bool`, *optional*, defaults to `True`):
                Whether the process calling this is the main process or not. Useful when in distributed training like
                TPUs and need to call this function on all processes. In this case, set `is_main_process=True` only on
                the main process to avoid race conditions.
            state_dict (nested dictionary of `torch.Tensor`):
                The state dictionary of the model to save. Will default to `self.state_dict()`, but can be used to only
                save parts of the model or if special precautions need to be taken when recovering the state dictionary
                of a model (like when using model parallelism).
            save_function (`Callable`):
                The function to use to save the state dictionary. Useful on distributed training like TPUs when one
                need to replace `torch.save` by another method.
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            max_shard_size (`int` or `str`, *optional*, defaults to `"5GB"`):
                The maximum size for a checkpoint before being sharded. Checkpoints shard will then be each of size
                lower than this size. If expressed as a string, needs to be digits followed by a unit (like `"5MB"`).
                We default it to 5GB in order for models to be able to run easily on free-tier google colab instances
                without CPU OOM issues.

                <Tip warning={true}>

                If a single weight of the model is bigger than `max_shard_size`, it will be in its own checkpoint shard
                which will be bigger than `max_shard_size`.

                </Tip>

            safe_serialization (`bool`, *optional*, defaults to `True`):
                Whether to save the model using `safetensors` or the traditional PyTorch way (that uses `pickle`).
            variant (`str`, *optional*):
                If specified, weights are saved in the format pytorch_model.<variant>.bin.
            token (`str` or `bool`, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, or not specified, will use
                the token generated when running `huggingface-cli login` (stored in `~/.huggingface`).
            save_peft_format (`bool`, *optional*, defaults to `True`):
                For backward compatibility with PEFT library, in case adapter weights are attached to the model, all
                keys of the state dict of adapters needs to be pre-pended with `base_model.model`. Advanced users can
                disable this behaviours by setting `save_peft_format` to `False`.
            kwargs (`Dict[str, Any]`, *optional*):
                Additional key word arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """
        use_auth_token = kwargs.pop("use_auth_token", None)
        ignore_metadata_errors = kwargs.pop("ignore_metadata_errors", False)

        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
                FutureWarning,
            )
            if token is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            token = use_auth_token

        if token is not None:
            kwargs["token"] = token

        _hf_peft_config_loaded = getattr(self, "_hf_peft_config_loaded", False)

        hf_quantizer = getattr(self, "hf_quantizer", None)
        quantization_serializable = (
            hf_quantizer is not None
            and isinstance(hf_quantizer, HfQuantizer)
            and hf_quantizer.is_serializable(safe_serialization=safe_serialization)
        )

        if hf_quantizer is not None and not _hf_peft_config_loaded and not quantization_serializable:
            raise ValueError(
                f"The model is quantized with {hf_quantizer.quantization_config.quant_method} and is not serializable - check out the warnings from"
                " the logger on the traceback to understand the reason why the quantized model is not serializable."
            )

        if "save_config" in kwargs:
            warnings.warn(
                "`save_config` is deprecated and will be removed in v5 of Transformers. Use `is_main_process` instead."
            )
            is_main_process = kwargs.pop("save_config")
        if safe_serialization and not is_safetensors_available():
            raise ImportError("`safe_serialization` requires the `safetensors library: `pip install safetensors`.")

        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        os.makedirs(save_directory, exist_ok=True)

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
            repo_id = self._create_repo(repo_id, **kwargs)
            files_timestamps = self._get_files_timestamps(save_directory)

        # Only save the model itself if we are using distributed training
        model_to_save = unwrap_model(self)

        # save the string version of dtype to the config, e.g. convert torch.float32 => "float32"
        # we currently don't use this setting automatically, but may start to use with v5
        dtype = get_parameter_dtype(model_to_save)
        model_to_save.config.torch_dtype = str(dtype).split(".")[1]

        # Attach architecture to the config
        model_to_save.config.architectures = [model_to_save.__class__.__name__]

        # Unset attn implementation so it can be set to another one when loading back
        model_to_save.config._attn_implementation_autoset = False

        # If we have a custom model, we copy the file defining it in the folder and set the attributes so it can be
        # loaded from the Hub.
        if self._auto_class is not None:
            custom_object_save(self, save_directory, config=self.config)

        # Save the config
        if is_main_process:
            if not _hf_peft_config_loaded:
                # If the model config has set attributes that should be in the generation config, move them there.
                misplaced_generation_parameters = model_to_save.config._get_non_default_generation_parameters()
                if self.can_generate() and len(misplaced_generation_parameters) > 0:
                    warnings.warn(
                        "Moving the following attributes in the config to the generation config: "
                        f"{misplaced_generation_parameters}. You are seeing this warning because you've set "
                        "generation parameters in the model config, as opposed to in the generation config.",
                        UserWarning,
                    )
                    for param_name, param_value in misplaced_generation_parameters.items():
                        setattr(model_to_save.generation_config, param_name, param_value)
                        setattr(model_to_save.config, param_name, None)

                model_to_save.config.save_pretrained(save_directory)
            if self.can_generate():
                model_to_save.generation_config.save_pretrained(save_directory)

            if _hf_peft_config_loaded:
                logger.info(
                    "Detected adapters on the model, saving the model in the PEFT format, only adapter weights will be saved."
                )
                state_dict = model_to_save.get_adapter_state_dict()

                if save_peft_format:
                    logger.info(
                        "To match the expected format of the PEFT library, all keys of the state dict of adapters will be pre-pended with `base_model.model`."
                    )
                    peft_state_dict = {}
                    for key, value in state_dict.items():
                        peft_state_dict[f"base_model.model.{key}"] = value
                    state_dict = peft_state_dict

                active_adapter = self.active_adapters()

                if len(active_adapter) > 1:
                    raise ValueError(
                        "Multiple active adapters detected, saving multiple active adapters is not supported yet. You can save adapters separately one by one "
                        "by iteratively calling `model.set_adapter(adapter_name)` then `model.save_pretrained(...)`"
                    )
                active_adapter = active_adapter[0]

                current_peft_config = self.peft_config[active_adapter]
                current_peft_config.save_pretrained(save_directory)

        # for offloaded modules
        module_map = {}

        # Save the model
        if state_dict is None:
            # if any model parameters are offloaded, make module map
            if (
                hasattr(self, "hf_device_map")
                and len(set(self.hf_device_map.values())) > 1
                and ("cpu" in self.hf_device_map.values() or "disk" in self.hf_device_map.values())
            ):
                warnings.warn(
                    "Attempting to save a model with offloaded modules. Ensure that unallocated cpu memory exceeds the `shard_size` (5GB default)"
                )
                for name, module in model_to_save.named_modules():
                    if name == "":
                        continue
                    module_state_dict = module.state_dict()

                    for key in module_state_dict:
                        module_map[name + f".{key}"] = module
            state_dict = model_to_save.state_dict()

        # Translate state_dict from smp to hf if saving with smp >= 1.10
        if IS_SAGEMAKER_MP_POST_1_10:
            for smp_to_hf, _ in smp.state.module_manager.translate_functions:
                state_dict = smp_to_hf(state_dict)

        # Handle the case where some state_dict keys shouldn't be saved
        if self._keys_to_ignore_on_save is not None:
            for ignore_key in self._keys_to_ignore_on_save:
                if ignore_key in state_dict.keys():
                    del state_dict[ignore_key]

        # Rename state_dict keys before saving to file. Do nothing unless overriden in a particular model.
        # (initially introduced with TimmWrapperModel to remove prefix and make checkpoints compatible with timm)
        state_dict = self._fix_state_dict_keys_on_save(state_dict)

        if safe_serialization:
            # Safetensors does not allow tensor aliasing.
            # We're going to remove aliases before saving
            ptrs = collections.defaultdict(list)
            for name, tensor in state_dict.items():
                # Sometimes in the state_dict we have non-tensor objects.
                # e.g. in bitsandbytes we have some `str` objects in the state_dict
                if isinstance(tensor, torch.Tensor):
                    ptrs[id_tensor_storage(tensor)].append(name)
                else:
                    # In the non-tensor case, fall back to the pointer of the object itself
                    ptrs[id(tensor)].append(name)

            # These are all the pointers of shared tensors
            if hasattr(self, "hf_device_map"):
                # if the model has offloaded parameters, we must check using find_tied_parameters()
                tied_params = find_tied_parameters(self)
                if tied_params:
                    tied_names = tied_params[0]
                    shared_ptrs = {
                        ptr: names for ptr, names in ptrs.items() if any(name in tied_names for name in names)
                    }
                else:
                    shared_ptrs = {}
            else:
                shared_ptrs = {ptr: names for ptr, names in ptrs.items() if len(names) > 1}

            # Recursively descend to find tied weight keys
            _tied_weights_keys = _get_tied_weight_keys(self)
            error_names = []
            to_delete_names = set()
            for names in shared_ptrs.values():
                # Removing the keys which are declared as known duplicates on
                # load. This allows to make sure the name which is kept is consistent.
                if _tied_weights_keys is not None:
                    found = 0
                    for name in sorted(names):
                        matches_pattern = any(re.search(pat, name) for pat in _tied_weights_keys)
                        if matches_pattern and name in state_dict:
                            found += 1
                            if found < len(names):
                                to_delete_names.add(name)
            # We are entering a place where the weights and the transformers configuration do NOT match.
            shared_names, disjoint_names = _find_disjoint(shared_ptrs.values(), state_dict)
            # Those are actually tensor sharing but disjoint from each other, we can safely clone them
            # Reloaded won't have the same property, but it shouldn't matter in any meaningful way.
            for name in disjoint_names:
                state_dict[name] = state_dict[name].clone()

            # When not all duplicates have been cleaned, still remove those keys, but put a clear warning.
            # If the link between tensors was done at runtime then `from_pretrained` will not get
            # the key back leading to random tensor. A proper warning will be shown
            # during reload (if applicable), but since the file is not necessarily compatible with
            # the config, better show a proper warning.
            shared_names, identical_names = _find_identical(shared_names, state_dict)
            # delete tensors that have identical storage
            for inames in identical_names:
                known = inames.intersection(to_delete_names)
                for name in known:
                    del state_dict[name]
                unknown = inames.difference(to_delete_names)
                if len(unknown) > 1:
                    error_names.append(unknown)

            if shared_names:
                error_names.append(set(shared_names))

            if len(error_names) > 0:
                raise RuntimeError(
                    f"The weights trying to be saved contained shared tensors {error_names} that are mismatching the transformers base configuration. Try saving using `safe_serialization=False` or remove this tensor sharing.",
                )

        # Shard the model if it is too big.
        if not _hf_peft_config_loaded:
            weights_name = SAFE_WEIGHTS_NAME if safe_serialization else WEIGHTS_NAME
            weights_name = _add_variant(weights_name, variant)
        else:
            weights_name = ADAPTER_SAFE_WEIGHTS_NAME if safe_serialization else ADAPTER_WEIGHTS_NAME

        filename_pattern = weights_name.replace(".bin", "{suffix}.bin").replace(".safetensors", "{suffix}.safetensors")
        state_dict_split = split_torch_state_dict_into_shards(
            state_dict, filename_pattern=filename_pattern, max_shard_size=max_shard_size
        )
        # Save index if sharded
        index = None
        if state_dict_split.is_sharded:
            index = {
                "metadata": state_dict_split.metadata,
                "weight_map": state_dict_split.tensor_to_filename,
            }

        # Clean the folder from a previous save
        for filename in os.listdir(save_directory):
            full_filename = os.path.join(save_directory, filename)
            # If we have a shard file that is not going to be replaced, we delete it, but only from the main process
            # in distributed settings to avoid race conditions.
            weights_no_suffix = weights_name.replace(".bin", "").replace(".safetensors", "")

            # make sure that file to be deleted matches format of sharded file, e.g. pytorch_model-00001-of-00005
            filename_no_suffix = filename.replace(".bin", "").replace(".safetensors", "")
            reg = re.compile(r"(.*?)-\d{5}-of-\d{5}")

            if (
                filename.startswith(weights_no_suffix)
                and os.path.isfile(full_filename)
                and filename not in state_dict_split.filename_to_tensors.keys()
                and is_main_process
                and reg.fullmatch(filename_no_suffix) is not None
            ):
                os.remove(full_filename)
        # Save the model
        filename_to_tensors = state_dict_split.filename_to_tensors.items()
        if module_map:
            filename_to_tensors = logging.tqdm(filename_to_tensors, desc="Saving checkpoint shards")
        for shard_file, tensors in filename_to_tensors:
            shard = {}
            for tensor in tensors:
                shard[tensor] = state_dict[tensor].contiguous()
                # delete reference, see https://github.com/huggingface/transformers/pull/34890
                del state_dict[tensor]

            # remake shard with onloaded parameters if necessary
            if module_map:
                if accelerate_version < version.parse("0.31"):
                    raise ImportError(
                        f"You need accelerate version to be greater or equal than 0.31 to save models with offloaded parameters. Detected version {accelerate_version}. "
                        f"Please upgrade accelerate with `pip install -U accelerate`"
                    )
                # init state_dict for this shard
                shard_state_dict = {name: "" for name in shard}
                for module_name in shard:
                    module = module_map[module_name]
                    # update state dict with onloaded parameters
                    shard_state_dict = get_state_dict_from_offload(module, module_name, shard_state_dict)

                # assign shard to be the completed state dict
                shard = shard_state_dict
                del shard_state_dict
                gc.collect()

            if safe_serialization:
                # At some point we will need to deal better with save_function (used for TPU and other distributed
                # joyfulness), but for now this enough.
                safe_save_file(shard, os.path.join(save_directory, shard_file), metadata={"format": "pt"})
            else:
                save_function(shard, os.path.join(save_directory, shard_file))

        del state_dict

        if index is None:
            path_to_weights = os.path.join(save_directory, weights_name)
            logger.info(f"Model weights saved in {path_to_weights}")
        else:
            save_index_file = SAFE_WEIGHTS_INDEX_NAME if safe_serialization else WEIGHTS_INDEX_NAME
            save_index_file = os.path.join(save_directory, _add_variant(save_index_file, variant))
            # Save the index as well
            with open(save_index_file, "w", encoding="utf-8") as f:
                content = json.dumps(index, indent=2, sort_keys=True) + "\n"
                f.write(content)
            logger.info(
                f"The model is bigger than the maximum size per checkpoint ({max_shard_size}) and is going to be "
                f"split in {len(state_dict_split.filename_to_tensors)} checkpoint shards. You can find where each parameters has been saved in the "
                f"index located at {save_index_file}."
            )

        if push_to_hub:
            # Eventually create an empty model card
            model_card = create_and_tag_model_card(
                repo_id, self.model_tags, token=token, ignore_metadata_errors=ignore_metadata_errors
            )

            # Update model card if needed:
            model_card.save(os.path.join(save_directory, "README.md"))

            self._upload_modified_files(
                save_directory,
                repo_id,
                files_timestamps,
                commit_message=commit_message,
                token=token,
            )

    @wraps(PushToHubMixin.push_to_hub)
    def push_to_hub(self, *args, **kwargs):
        tags = self.model_tags if self.model_tags is not None else []

        tags_kwargs = kwargs.get("tags", [])
        if isinstance(tags_kwargs, str):
            tags_kwargs = [tags_kwargs]

        for tag in tags_kwargs:
            if tag not in tags:
                tags.append(tag)

        if tags:
            kwargs["tags"] = tags
        return super().push_to_hub(*args, **kwargs)

    def get_memory_footprint(self, return_buffers=True):
        r"""
        Get the memory footprint of a model. This will return the memory footprint of the current model in bytes.
        Useful to benchmark the memory footprint of the current model and design some tests. Solution inspired from the
        PyTorch discussions: https://discuss.pytorch.org/t/gpu-memory-that-model-uses/56822/2

        Arguments:
            return_buffers (`bool`, *optional*, defaults to `True`):
                Whether to return the size of the buffer tensors in the computation of the memory footprint. Buffers
                are tensors that do not require gradients and not registered as parameters. E.g. mean and std in batch
                norm layers. Please see: https://discuss.pytorch.org/t/what-pytorch-means-by-buffers/120266/2
        """
        mem = sum([param.nelement() * param.element_size() for param in self.parameters()])
        if return_buffers:
            mem_bufs = sum([buf.nelement() * buf.element_size() for buf in self.buffers()])
            mem = mem + mem_bufs
        return mem

    @wraps(torch.nn.Module.cuda)
    def cuda(self, *args, **kwargs):
        if getattr(self, "quantization_method", None) == QuantizationMethod.HQQ:
            raise ValueError("`.cuda` is not supported for HQQ-quantized models.")
        # Checks if the model has been loaded in 4-bit or 8-bit with BNB
        if getattr(self, "quantization_method", None) == QuantizationMethod.BITS_AND_BYTES:
            if getattr(self, "is_loaded_in_8bit", False):
                raise ValueError(
                    "Calling `cuda()` is not supported for `8-bit` quantized models. "
                    " Please use the model as it is, since the model has already been set to the correct devices."
                )
            elif version.parse(importlib.metadata.version("bitsandbytes")) < version.parse("0.43.2"):
                raise ValueError(
                    "Calling `cuda()` is not supported for `4-bit` quantized models with the installed version of bitsandbytes. "
                    f"The current device is `{self.device}`. If you intended to move the model, please install bitsandbytes >= 0.43.2."
                )
        else:
            return super().cuda(*args, **kwargs)

    @wraps(torch.nn.Module.to)
    def to(self, *args, **kwargs):
        # For BNB/GPTQ models, we prevent users from casting the model to another dtype to restrict unwanted behaviours.
        # the correct API should be to load the model with the desired dtype directly through `from_pretrained`.
        dtype_present_in_args = "dtype" in kwargs

        if not dtype_present_in_args:
            for arg in args:
                if isinstance(arg, torch.dtype):
                    dtype_present_in_args = True
                    break

        if getattr(self, "quantization_method", None) == QuantizationMethod.HQQ:
            raise ValueError("`.to` is not supported for HQQ-quantized models.")
        # Checks if the model has been loaded in 4-bit or 8-bit with BNB
        if getattr(self, "quantization_method", None) == QuantizationMethod.BITS_AND_BYTES:
            if dtype_present_in_args:
                raise ValueError(
                    "You cannot cast a bitsandbytes model in a new `dtype`. Make sure to load the model using `from_pretrained` using the"
                    " desired `dtype` by passing the correct `torch_dtype` argument."
                )

            if getattr(self, "is_loaded_in_8bit", False):
                raise ValueError(
                    "`.to` is not supported for `8-bit` bitsandbytes models. Please use the model as it is, since the"
                    " model has already been set to the correct devices and casted to the correct `dtype`."
                )
            elif version.parse(importlib.metadata.version("bitsandbytes")) < version.parse("0.43.2"):
                raise ValueError(
                    "Calling `to()` is not supported for `4-bit` quantized models with the installed version of bitsandbytes. "
                    f"The current device is `{self.device}`. If you intended to move the model, please install bitsandbytes >= 0.43.2."
                )
        elif getattr(self, "quantization_method", None) == QuantizationMethod.GPTQ:
            if dtype_present_in_args:
                raise ValueError(
                    "You cannot cast a GPTQ model in a new `dtype`. Make sure to load the model using `from_pretrained` using the desired"
                    " `dtype` by passing the correct `torch_dtype` argument."
                )
        return super().to(*args, **kwargs)

    def half(self, *args):
        # Checks if the model is quantized
        if getattr(self, "is_quantized", False):
            raise ValueError(
                "`.half()` is not supported for quantized model. Please use the model as it is, since the"
                " model has already been casted to the correct `dtype`."
            )
        else:
            return super().half(*args)

    def float(self, *args):
        # Checks if the model is quantized
        if getattr(self, "is_quantized", False):
            raise ValueError(
                "`.float()` is not supported for quantized model. Please use the model as it is, since the"
                " model has already been casted to the correct `dtype`."
            )
        else:
            return super().float(*args)

    @classmethod
    def from_pretrained(
        cls: Type[SpecificPreTrainedModelType],
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        use_safetensors: Optional[bool] = None,
        weights_only: bool = True,
        **kwargs,
    ) -> SpecificPreTrainedModelType:
        r"""
        Instantiate a pretrained pytorch model from a pre-trained model configuration.

        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated). To train
        the model, you should first set it back in training mode with `model.train()`.

        The warning *Weights from XXX not initialized from pretrained model* means that the weights of XXX do not come
        pretrained with the rest of the model. It is up to you to train those weights with a downstream fine-tuning
        task.

        The warning *Weights from XXX not used in YYY* means that the layer XXX is not used by YYY, therefore those
        weights are discarded.

        If model weights are the same precision as the base model (and is a supported model), weights will be lazily loaded
        in using the `meta` device and brought into memory once an input is passed through that layer regardless of
        `low_cpu_mem_usage`.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                    - A path to a *directory* containing model weights saved using
                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
                    - A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
                      this case, `from_tf` should be set to `True` and a configuration object should be provided as
                      `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
                      PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
                    - A path or url to a model folder containing a *flax checkpoint file* in *.msgpack* format (e.g,
                      `./flax_model/` containing `flax_model.msgpack`). In this case, `from_flax` should be set to
                      `True`.
                    - `None` if you are both providing the configuration and state dictionary (resp. with keyword
                      arguments `config` and `state_dict`).
            model_args (sequence of positional arguments, *optional*):
                All remaining positional arguments will be passed to the underlying model's `__init__` method.
            config (`Union[PretrainedConfig, str, os.PathLike]`, *optional*):
                Can be either:

                    - an instance of a class derived from [`PretrainedConfig`],
                    - a string or path valid as input to [`~PretrainedConfig.from_pretrained`].

                Configuration for the model to use instead of an automatically loaded configuration. Configuration can
                be automatically loaded when:

                    - The model is a model provided by the library (loaded with the *model id* string of a pretrained
                      model).
                    - The model was saved using [`~PreTrainedModel.save_pretrained`] and is reloaded by supplying the
                      save directory.
                    - The model is loaded by supplying a local directory as `pretrained_model_name_or_path` and a
                      configuration JSON file named *config.json* is found in the directory.
            state_dict (`Dict[str, torch.Tensor]`, *optional*):
                A state dictionary to use instead of a state dictionary loaded from saved weights file.

                This option can be used if you want to create a model from a pretrained configuration but load your own
                weights. In this case though, you should check if using [`~PreTrainedModel.save_pretrained`] and
                [`~PreTrainedModel.from_pretrained`] is not a simpler option.
            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            from_tf (`bool`, *optional*, defaults to `False`):
                Load the model weights from a TensorFlow checkpoint save file (see docstring of
                `pretrained_model_name_or_path` argument).
            from_flax (`bool`, *optional*, defaults to `False`):
                Load the model weights from a Flax checkpoint save file (see docstring of
                `pretrained_model_name_or_path` argument).
            ignore_mismatched_sizes (`bool`, *optional*, defaults to `False`):
                Whether or not to raise an error if some of the weights from the checkpoint do not have the same size
                as the weights of the model (if for instance, you are instantiating a model with 10 labels from a
                checkpoint with 3 labels).
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            resume_download:
                Deprecated and ignored. All downloads are now resumed by default when possible.
                Will be removed in v5 of Transformers.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            output_loading_info(`bool`, *optional*, defaults to `False`):
                Whether ot not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only(`bool`, *optional*, defaults to `False`):
                Whether or not to only look at local files (i.e., do not try to download the model).
            token (`str` or `bool`, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, or not specified, will use
                the token generated when running `huggingface-cli login` (stored in `~/.huggingface`).
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.

                <Tip>

                To test a pull request you made on the Hub, you can pass `revision="refs/pr/<pr_number>"`.

                </Tip>

            mirror (`str`, *optional*):
                Mirror source to accelerate downloads in China. If you are from China and have an accessibility
                problem, you can set this option to resolve it. Note that we do not guarantee the timeliness or safety.
                Please refer to the mirror site for more information.
            _fast_init(`bool`, *optional*, defaults to `True`):
                Whether or not to disable fast initialization.

                <Tip warning={true}>

                One should only disable *_fast_init* to ensure backwards compatibility with `transformers.__version__ <
                4.6.0` for seeded model initialization. This argument will be removed at the next major version. See
                [pull request 11471](https://github.com/huggingface/transformers/pull/11471) for more information.

                </Tip>
            attn_implementation (`str`, *optional*):
                The attention implementation to use in the model (if relevant). Can be any of `"eager"` (manual implementation of the attention), `"sdpa"` (using [`F.scaled_dot_product_attention`](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html)), or `"flash_attention_2"` (using [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)). By default, if available, SDPA will be used for torch>=2.1.1. The default is otherwise the manual `"eager"` implementation.

            > Parameters for big model inference

            low_cpu_mem_usage(`bool`, *optional*):
                Tries not to use more than 1x model size in CPU memory (including peak memory) while loading the model.
                Generally should be combined with a `device_map` (such as `"auto"`) for best results.
                This is an experimental feature and a subject to change at any moment.
                </Tip>
                    If the model weights are in the same precision as the model loaded in, `low_cpu_mem_usage` (without
                    `device_map`) is redundant and will not provide any benefit in regards to CPU memory usage. However,
                    this should still be enabled if you are passing in a `device_map`.
                </Tip>
            torch_dtype (`str` or `torch.dtype`, *optional*):
                Override the default `torch.dtype` and load the model under a specific `dtype`. The different options
                are:

                1. `torch.float16` or `torch.bfloat16` or `torch.float`: load in a specified
                  `dtype`, ignoring the model's `config.torch_dtype` if one exists. If not specified
                  - the model will get loaded in `torch.float` (fp32).

                2. `"auto"` - A `torch_dtype` entry in the `config.json` file of the model will be
                  attempted to be used. If this entry isn't found then next check the `dtype` of the first weight in
                  the checkpoint that's of a floating point type and use that as `dtype`. This will load the model
                  using the `dtype` it was saved in at the end of the training. It can't be used as an indicator of how
                  the model was trained. Since it could be trained in one of half precision dtypes, but saved in fp32.

                3. A string that is a valid `torch.dtype`. E.g. "float32" loads the model in `torch.float32`, "float16" loads in `torch.float16` etc.

                <Tip>

                For some models the `dtype` they were trained in is unknown - you may try to check the model's paper or
                reach out to the authors and ask them to add this information to the model's card and to insert the
                `torch_dtype` entry in `config.json` on the hub.

                </Tip>

            device_map (`str` or `Dict[str, Union[int, str, torch.device]]` or `int` or `torch.device`, *optional*):
                A map that specifies where each submodule should go. It doesn't need to be refined to each
                parameter/buffer name, once a given module name is inside, every submodule of it will be sent to the
                same device. If we only pass the device (*e.g.*, `"cpu"`, `"cuda:1"`, `"mps"`, or a GPU ordinal rank
                like `1`) on which the model will be allocated, the device map will map the entire model to this
                device. Passing `device_map = 0` means put the whole model on GPU 0.

                To have Accelerate compute the most optimized `device_map` automatically, set `device_map="auto"`. For
                more information about each option see [designing a device
                map](https://hf.co/docs/accelerate/main/en/usage_guides/big_modeling#designing-a-device-map).
            max_memory (`Dict`, *optional*):
                A dictionary device identifier to maximum memory. Will default to the maximum memory available for each
                GPU and the available CPU RAM if unset.
            offload_folder (`str` or `os.PathLike`, *optional*):
                If the `device_map` contains any value `"disk"`, the folder where we will offload weights.
            offload_state_dict (`bool`, *optional*):
                If `True`, will temporarily offload the CPU state dict to the hard drive to avoid getting out of CPU
                RAM if the weight of the CPU state dict + the biggest shard of the checkpoint does not fit. Defaults to
                `True` when there is some disk offload.
            offload_buffers (`bool`, *optional*):
                Whether or not to offload the buffers with the model parameters.
            quantization_config (`Union[QuantizationConfigMixin,Dict]`, *optional*):
                A dictionary of configuration parameters or a QuantizationConfigMixin object for quantization (e.g
                bitsandbytes, gptq). There may be other quantization-related kwargs, including `load_in_4bit` and
                `load_in_8bit`, which are parsed by QuantizationConfigParser. Supported only for bitsandbytes
                quantizations and not preferred. consider inserting all such arguments into quantization_config
                instead.
            subfolder (`str`, *optional*, defaults to `""`):
                In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can
                specify the folder name here.
            variant (`str`, *optional*):
                If specified load weights from `variant` filename, *e.g.* pytorch_model.<variant>.bin. `variant` is
                ignored when using `from_tf` or `from_flax`.
            use_safetensors (`bool`, *optional*, defaults to `None`):
                Whether or not to use `safetensors` checkpoints. Defaults to `None`. If not specified and `safetensors`
                is not installed, it will be set to `False`.

            weights_only (`bool`, *optional*, defaults to `True`):
                Indicates whether unpickler should be restricted to loading only tensors, primitive types,
                dictionaries and any types added via torch.serialization.add_safe_globals().
                When set to False, we can load wrapper tensor subclass weights.

            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
                `output_attentions=True`). Behaves differently depending on whether a `config` is provided or
                automatically loaded:

                    - If a configuration is provided with `config`, `**kwargs` will be directly passed to the
                      underlying model's `__init__` method (we assume all relevant updates to the configuration have
                      already been done)
                    - If a configuration is not provided, `kwargs` will be first passed to the configuration class
                      initialization function ([`~PretrainedConfig.from_pretrained`]). Each key of `kwargs` that
                      corresponds to a configuration attribute will be used to override said attribute with the
                      supplied `kwargs` value. Remaining keys that do not correspond to any configuration attribute
                      will be passed to the underlying model's `__init__` function.

        <Tip>

        Activate the special ["offline-mode"](https://huggingface.co/transformers/installation.html#offline-mode) to
        use this method in a firewalled environment.

        </Tip>

        Examples:

        ```python
        >>> from transformers import BertConfig, BertModel

        >>> # Download model and configuration from huggingface.co and cache.
        >>> model = BertModel.from_pretrained("google-bert/bert-base-uncased")
        >>> # Model was saved using *save_pretrained('./test/saved_model/')* (for example purposes, not runnable).
        >>> model = BertModel.from_pretrained("./test/saved_model/")
        >>> # Update configuration during loading.
        >>> model = BertModel.from_pretrained("google-bert/bert-base-uncased", output_attentions=True)
        >>> assert model.config.output_attentions == True
        >>> # Loading from a TF checkpoint file instead of a PyTorch model (slower, for example purposes, not runnable).
        >>> config = BertConfig.from_json_file("./tf_model/my_tf_model_config.json")
        >>> model = BertModel.from_pretrained("./tf_model/my_tf_checkpoint.ckpt.index", from_tf=True, config=config)
        >>> # Loading from a Flax checkpoint file instead of a PyTorch model (slower)
        >>> model = BertModel.from_pretrained("google-bert/bert-base-uncased", from_flax=True)
        ```

        * `low_cpu_mem_usage` algorithm:

        This is an experimental function that loads the model using ~1x model size CPU memory

        Here is how it works:

        1. save which state_dict keys we have
        2. drop state_dict before the model is created, since the latter takes 1x model size CPU memory
        3. after the model has been instantiated switch to the meta device all params/buffers that
        are going to be replaced from the loaded state_dict
        4. load state_dict 2nd time
        5. replace the params/buffers from the state_dict

        Currently, it can't handle deepspeed ZeRO stage 3 and ignores loading errors

        """
        state_dict = kwargs.pop("state_dict", None)
        from_tf = kwargs.pop("from_tf", False)
        from_flax = kwargs.pop("from_flax", False)
        resume_download = kwargs.pop("resume_download", None)
        proxies = kwargs.pop("proxies", None)
        output_loading_info = kwargs.pop("output_loading_info", False)
        use_auth_token = kwargs.pop("use_auth_token", None)
        trust_remote_code = kwargs.pop("trust_remote_code", None)
        _ = kwargs.pop("mirror", None)
        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)
        _fast_init = kwargs.pop("_fast_init", True)
        torch_dtype = kwargs.pop("torch_dtype", None)
        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", None)
        device_map = kwargs.pop("device_map", None)
        max_memory = kwargs.pop("max_memory", None)
        offload_folder = kwargs.pop("offload_folder", None)
        offload_state_dict = kwargs.pop("offload_state_dict", False)
        offload_buffers = kwargs.pop("offload_buffers", False)
        load_in_8bit = kwargs.pop("load_in_8bit", False)
        load_in_4bit = kwargs.pop("load_in_4bit", False)
        quantization_config = kwargs.pop("quantization_config", None)
        subfolder = kwargs.pop("subfolder", "")
        commit_hash = kwargs.pop("_commit_hash", None)
        variant = kwargs.pop("variant", None)
        adapter_kwargs = kwargs.pop("adapter_kwargs", {})
        adapter_name = kwargs.pop("adapter_name", "default")
        use_flash_attention_2 = kwargs.pop("use_flash_attention_2", False)
        generation_config = kwargs.pop("generation_config", None)

        gguf_file = kwargs.pop("gguf_file", None)
        # Cache path to the GGUF file
        gguf_path = None

        tp_plan = kwargs.pop("tp_plan", None)
        if tp_plan is not None and tp_plan != "auto":
            # TODO: we can relax this check when we support taking tp_plan from a json file, for example.
            raise ValueError(f"tp_plan supports 'auto' only for now but got {tp_plan}.")

        if is_fsdp_enabled():
            low_cpu_mem_usage = True

        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
                FutureWarning,
            )
            if token is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            token = use_auth_token

        if token is not None and adapter_kwargs is not None and "token" not in adapter_kwargs:
            adapter_kwargs["token"] = token

        if use_safetensors is None and not is_safetensors_available():
            use_safetensors = False
        if trust_remote_code is True:
            logger.warning(
                "The argument `trust_remote_code` is to be used with Auto classes. It has no effect here and is"
                " ignored."
            )

        if gguf_file is not None and not is_accelerate_available():
            raise ValueError("accelerate is required when loading a GGUF file `pip install accelerate`.")

        if commit_hash is None:
            if not isinstance(config, PretrainedConfig):
                # We make a call to the config file first (which may be absent) to get the commit hash as soon as possible
                resolved_config_file = cached_file(
                    pretrained_model_name_or_path,
                    CONFIG_NAME,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    revision=revision,
                    subfolder=subfolder,
                    _raise_exceptions_for_gated_repo=False,
                    _raise_exceptions_for_missing_entries=False,
                    _raise_exceptions_for_connection_errors=False,
                )
                commit_hash = extract_commit_hash(resolved_config_file, commit_hash)
            else:
                commit_hash = getattr(config, "_commit_hash", None)

        if is_peft_available():
            _adapter_model_path = adapter_kwargs.pop("_adapter_model_path", None)

            if _adapter_model_path is None:
                _adapter_model_path = find_adapter_config_file(
                    pretrained_model_name_or_path,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    _commit_hash=commit_hash,
                    **adapter_kwargs,
                )
            if _adapter_model_path is not None and os.path.isfile(_adapter_model_path):
                with open(_adapter_model_path, "r", encoding="utf-8") as f:
                    _adapter_model_path = pretrained_model_name_or_path
                    pretrained_model_name_or_path = json.load(f)["base_model_name_or_path"]
        else:
            _adapter_model_path = None

        # change device_map into a map if we passed an int, a str or a torch.device
        if isinstance(device_map, torch.device):
            device_map = {"": device_map}
        elif isinstance(device_map, str) and device_map not in ["auto", "balanced", "balanced_low_0", "sequential"]:
            try:
                device_map = {"": torch.device(device_map)}
            except RuntimeError:
                raise ValueError(
                    "When passing device_map as a string, the value needs to be a device name (e.g. cpu, cuda:0) or "
                    f"'auto', 'balanced', 'balanced_low_0', 'sequential' but found {device_map}."
                )
        elif isinstance(device_map, int):
            if device_map < 0:
                raise ValueError(
                    "You can't pass device_map as a negative int. If you want to put the model on the cpu, pass device_map = 'cpu' "
                )
            else:
                device_map = {"": device_map}

        if device_map is not None:
            if low_cpu_mem_usage is None:
                low_cpu_mem_usage = True
            elif not low_cpu_mem_usage:
                raise ValueError("Passing along a `device_map` requires `low_cpu_mem_usage=True`")

        if low_cpu_mem_usage:
            if is_deepspeed_zero3_enabled():
                raise ValueError(
                    "DeepSpeed Zero-3 is not compatible with `low_cpu_mem_usage=True` or with passing a `device_map`."
                )
            elif not is_accelerate_available():
                raise ImportError(
                    f"Using `low_cpu_mem_usage=True` or a `device_map` requires Accelerate: `pip install 'accelerate>={ACCELERATE_MIN_VERSION}'`"
                )

        # handling bnb config from kwargs, remove after `load_in_{4/8}bit` deprecation.
        if load_in_4bit or load_in_8bit:
            if quantization_config is not None:
                raise ValueError(
                    "You can't pass `load_in_4bit`or `load_in_8bit` as a kwarg when passing "
                    "`quantization_config` argument at the same time."
                )

            # preparing BitsAndBytesConfig from kwargs
            config_dict = {k: v for k, v in kwargs.items() if k in inspect.signature(BitsAndBytesConfig).parameters}
            config_dict = {**config_dict, "load_in_4bit": load_in_4bit, "load_in_8bit": load_in_8bit}
            quantization_config, kwargs = BitsAndBytesConfig.from_dict(
                config_dict=config_dict, return_unused_kwargs=True, **kwargs
            )
            logger.warning(
                "The `load_in_4bit` and `load_in_8bit` arguments are deprecated and will be removed in the future versions. "
                "Please, pass a `BitsAndBytesConfig` object in `quantization_config` argument instead."
            )

        from_pt = not (from_tf | from_flax)

        user_agent = {"file_type": "model", "framework": "pytorch", "from_auto_class": from_auto_class}
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline

        if is_offline_mode() and not local_files_only:
            logger.info("Offline mode: forcing local_files_only=True")
            local_files_only = True

        # Load config if we don't provide a configuration
        if not isinstance(config, PretrainedConfig):
            config_path = config if config is not None else pretrained_model_name_or_path
            config, model_kwargs = cls.config_class.from_pretrained(
                config_path,
                cache_dir=cache_dir,
                return_unused_kwargs=True,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                subfolder=subfolder,
                _from_auto=from_auto_class,
                _from_pipeline=from_pipeline,
                **kwargs,
            )
        else:
            # In case one passes a config to `from_pretrained` + "attn_implementation"
            # override the `_attn_implementation` attribute to `attn_implementation` of the kwargs
            # Please see: https://github.com/huggingface/transformers/issues/28038

            # Overwrite `config._attn_implementation` by the one from the kwargs --> in auto-factory
            # we pop attn_implementation from the kwargs but this handles the case where users
            # passes manually the config to `from_pretrained`.
            config = copy.deepcopy(config)

            kwarg_attn_imp = kwargs.pop("attn_implementation", None)
            if kwarg_attn_imp is not None:
                config._attn_implementation = kwarg_attn_imp

            model_kwargs = kwargs

        pre_quantized = getattr(config, "quantization_config", None) is not None
        if pre_quantized or quantization_config is not None:
            if pre_quantized:
                config.quantization_config = AutoHfQuantizer.merge_quantization_configs(
                    config.quantization_config, quantization_config
                )
            else:
                config.quantization_config = quantization_config

            hf_quantizer = AutoHfQuantizer.from_config(
                config.quantization_config,
                pre_quantized=pre_quantized,
            )

        else:
            hf_quantizer = None

        if hf_quantizer is not None:
            hf_quantizer.validate_environment(
                torch_dtype=torch_dtype,
                from_tf=from_tf,
                from_flax=from_flax,
                device_map=device_map,
                weights_only=weights_only,
            )
            torch_dtype = hf_quantizer.update_torch_dtype(torch_dtype)
            device_map = hf_quantizer.update_device_map(device_map)

            # In order to ensure popular quantization methods are supported. Can be disable with `disable_telemetry`
            user_agent["quant"] = hf_quantizer.quantization_config.quant_method.value

            # Force-set to `True` for more mem efficiency
            if low_cpu_mem_usage is None:
                low_cpu_mem_usage = True
                logger.warning("`low_cpu_mem_usage` was None, now default to True since model is quantized.")
        is_quantized = hf_quantizer is not None

        # This variable will flag if we're loading a sharded checkpoint. In this case the archive file is just the
        # index of the files.
        is_sharded = False
        sharded_metadata = None
        # Load model
        loading_info = None

        # Keep in fp32 modules
        keep_in_fp32_modules = None
        use_keep_in_fp32_modules = False

        if gguf_file is not None and hf_quantizer is not None:
            raise ValueError(
                "You cannot combine Quantization and loading a model from a GGUF file, try again by making sure you did not passed a `quantization_config` or that you did not load a quantized model from the Hub."
            )

        if pretrained_model_name_or_path is not None and gguf_file is None:
            pretrained_model_name_or_path = str(pretrained_model_name_or_path)
            is_local = os.path.isdir(pretrained_model_name_or_path)
            if is_local:
                if from_tf and os.path.isfile(
                    os.path.join(pretrained_model_name_or_path, subfolder, TF_WEIGHTS_NAME + ".index")
                ):
                    # Load from a TF 1.0 checkpoint in priority if from_tf
                    archive_file = os.path.join(pretrained_model_name_or_path, subfolder, TF_WEIGHTS_NAME + ".index")
                elif from_tf and os.path.isfile(
                    os.path.join(pretrained_model_name_or_path, subfolder, TF2_WEIGHTS_NAME)
                ):
                    # Load from a TF 2.0 checkpoint in priority if from_tf
                    archive_file = os.path.join(pretrained_model_name_or_path, subfolder, TF2_WEIGHTS_NAME)
                elif from_flax and os.path.isfile(
                    os.path.join(pretrained_model_name_or_path, subfolder, FLAX_WEIGHTS_NAME)
                ):
                    # Load from a Flax checkpoint in priority if from_flax
                    archive_file = os.path.join(pretrained_model_name_or_path, subfolder, FLAX_WEIGHTS_NAME)
                elif use_safetensors is not False and os.path.isfile(
                    os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(SAFE_WEIGHTS_NAME, variant))
                ):
                    # Load from a safetensors checkpoint
                    archive_file = os.path.join(
                        pretrained_model_name_or_path, subfolder, _add_variant(SAFE_WEIGHTS_NAME, variant)
                    )
                elif use_safetensors is not False and os.path.isfile(
                    os.path.join(
                        pretrained_model_name_or_path, subfolder, _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant)
                    )
                ):
                    # Load from a sharded safetensors checkpoint
                    archive_file = os.path.join(
                        pretrained_model_name_or_path, subfolder, _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant)
                    )
                    is_sharded = True
                elif not use_safetensors and os.path.isfile(
                    os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(WEIGHTS_NAME, variant))
                ):
                    # Load from a PyTorch checkpoint
                    archive_file = os.path.join(
                        pretrained_model_name_or_path, subfolder, _add_variant(WEIGHTS_NAME, variant)
                    )
                elif not use_safetensors and os.path.isfile(
                    os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(WEIGHTS_INDEX_NAME, variant))
                ):
                    # Load from a sharded PyTorch checkpoint
                    archive_file = os.path.join(
                        pretrained_model_name_or_path, subfolder, _add_variant(WEIGHTS_INDEX_NAME, variant)
                    )
                    is_sharded = True
                # At this stage we don't have a weight file so we will raise an error.
                elif not use_safetensors and (
                    os.path.isfile(os.path.join(pretrained_model_name_or_path, subfolder, TF_WEIGHTS_NAME + ".index"))
                    or os.path.isfile(os.path.join(pretrained_model_name_or_path, subfolder, TF2_WEIGHTS_NAME))
                ):
                    raise EnvironmentError(
                        f"Error no file named {_add_variant(WEIGHTS_NAME, variant)} found in directory"
                        f" {pretrained_model_name_or_path} but there is a file for TensorFlow weights. Use"
                        " `from_tf=True` to load this model from those weights."
                    )
                elif not use_safetensors and os.path.isfile(
                    os.path.join(pretrained_model_name_or_path, subfolder, FLAX_WEIGHTS_NAME)
                ):
                    raise EnvironmentError(
                        f"Error no file named {_add_variant(WEIGHTS_NAME, variant)} found in directory"
                        f" {pretrained_model_name_or_path} but there is a file for Flax weights. Use `from_flax=True`"
                        " to load this model from those weights."
                    )
                elif use_safetensors:
                    raise EnvironmentError(
                        f"Error no file named {_add_variant(SAFE_WEIGHTS_NAME, variant)} found in directory"
                        f" {pretrained_model_name_or_path}."
                    )
                else:
                    raise EnvironmentError(
                        f"Error no file named {_add_variant(WEIGHTS_NAME, variant)}, {_add_variant(SAFE_WEIGHTS_NAME, variant)},"
                        f" {TF2_WEIGHTS_NAME}, {TF_WEIGHTS_NAME + '.index'} or {FLAX_WEIGHTS_NAME} found in directory"
                        f" {pretrained_model_name_or_path}."
                    )
            elif os.path.isfile(os.path.join(subfolder, pretrained_model_name_or_path)):
                archive_file = pretrained_model_name_or_path
                is_local = True
            elif os.path.isfile(os.path.join(subfolder, pretrained_model_name_or_path + ".index")):
                if not from_tf:
                    raise ValueError(
                        f"We found a TensorFlow checkpoint at {pretrained_model_name_or_path + '.index'}, please set "
                        "from_tf to True to load from this checkpoint."
                    )
                archive_file = os.path.join(subfolder, pretrained_model_name_or_path + ".index")
                is_local = True
            elif is_remote_url(pretrained_model_name_or_path):
                filename = pretrained_model_name_or_path
                resolved_archive_file = download_url(pretrained_model_name_or_path)
            else:
                # set correct filename
                if from_tf:
                    filename = TF2_WEIGHTS_NAME
                elif from_flax:
                    filename = FLAX_WEIGHTS_NAME
                elif use_safetensors is not False:
                    filename = _add_variant(SAFE_WEIGHTS_NAME, variant)
                else:
                    filename = _add_variant(WEIGHTS_NAME, variant)

                try:
                    # Load from URL or cache if already cached
                    cached_file_kwargs = {
                        "cache_dir": cache_dir,
                        "force_download": force_download,
                        "proxies": proxies,
                        "resume_download": resume_download,
                        "local_files_only": local_files_only,
                        "token": token,
                        "user_agent": user_agent,
                        "revision": revision,
                        "subfolder": subfolder,
                        "_raise_exceptions_for_gated_repo": False,
                        "_raise_exceptions_for_missing_entries": False,
                        "_commit_hash": commit_hash,
                    }
                    resolved_archive_file = cached_file(pretrained_model_name_or_path, filename, **cached_file_kwargs)

                    # Since we set _raise_exceptions_for_missing_entries=False, we don't get an exception but a None
                    # result when internet is up, the repo and revision exist, but the file does not.
                    if resolved_archive_file is None and filename == _add_variant(SAFE_WEIGHTS_NAME, variant):
                        # Maybe the checkpoint is sharded, we try to grab the index name in this case.
                        resolved_archive_file = cached_file(
                            pretrained_model_name_or_path,
                            _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant),
                            **cached_file_kwargs,
                        )
                        if resolved_archive_file is not None:
                            is_sharded = True
                        elif use_safetensors:
                            if revision == "main":
                                resolved_archive_file, revision, is_sharded = auto_conversion(
                                    pretrained_model_name_or_path, **cached_file_kwargs
                                )
                            cached_file_kwargs["revision"] = revision
                            if resolved_archive_file is None:
                                raise EnvironmentError(
                                    f"{pretrained_model_name_or_path} does not appear to have a file named"
                                    f" {_add_variant(SAFE_WEIGHTS_NAME, variant)} or {_add_variant(SAFE_WEIGHTS_INDEX_NAME, variant)} "
                                    "and thus cannot be loaded with `safetensors`. Please make sure that the model has "
                                    "been saved with `safe_serialization=True` or do not set `use_safetensors=True`."
                                )
                        else:
                            # This repo has no safetensors file of any kind, we switch to PyTorch.
                            filename = _add_variant(WEIGHTS_NAME, variant)
                            resolved_archive_file = cached_file(
                                pretrained_model_name_or_path, filename, **cached_file_kwargs
                            )
                    if resolved_archive_file is None and filename == _add_variant(WEIGHTS_NAME, variant):
                        # Maybe the checkpoint is sharded, we try to grab the index name in this case.
                        resolved_archive_file = cached_file(
                            pretrained_model_name_or_path,
                            _add_variant(WEIGHTS_INDEX_NAME, variant),
                            **cached_file_kwargs,
                        )
                        if resolved_archive_file is not None:
                            is_sharded = True
                    if not local_files_only and not is_offline_mode():
                        if resolved_archive_file is not None:
                            if filename in [WEIGHTS_NAME, WEIGHTS_INDEX_NAME]:
                                # If the PyTorch file was found, check if there is a safetensors file on the repository
                                # If there is no safetensors file on the repositories, start an auto conversion
                                safe_weights_name = SAFE_WEIGHTS_INDEX_NAME if is_sharded else SAFE_WEIGHTS_NAME
                                has_file_kwargs = {
                                    "revision": revision,
                                    "proxies": proxies,
                                    "token": token,
                                    "cache_dir": cache_dir,
                                    "local_files_only": local_files_only,
                                }
                                cached_file_kwargs = {
                                    "cache_dir": cache_dir,
                                    "force_download": force_download,
                                    "resume_download": resume_download,
                                    "local_files_only": local_files_only,
                                    "user_agent": user_agent,
                                    "subfolder": subfolder,
                                    "_raise_exceptions_for_gated_repo": False,
                                    "_raise_exceptions_for_missing_entries": False,
                                    "_commit_hash": commit_hash,
                                    **has_file_kwargs,
                                }
                                if not has_file(pretrained_model_name_or_path, safe_weights_name, **has_file_kwargs):
                                    Thread(
                                        target=auto_conversion,
                                        args=(pretrained_model_name_or_path,),
                                        kwargs={"ignore_errors_during_conversion": True, **cached_file_kwargs},
                                        name="Thread-auto_conversion",
                                    ).start()
                        else:
                            # Otherwise, no PyTorch file was found, maybe there is a TF or Flax model file.
                            # We try those to give a helpful error message.
                            has_file_kwargs = {
                                "revision": revision,
                                "proxies": proxies,
                                "token": token,
                                "cache_dir": cache_dir,
                                "local_files_only": local_files_only,
                            }
                            if has_file(pretrained_model_name_or_path, TF2_WEIGHTS_NAME, **has_file_kwargs):
                                raise EnvironmentError(
                                    f"{pretrained_model_name_or_path} does not appear to have a file named"
                                    f" {_add_variant(WEIGHTS_NAME, variant)} but there is a file for TensorFlow weights."
                                    " Use `from_tf=True` to load this model from those weights."
                                )
                            elif has_file(pretrained_model_name_or_path, FLAX_WEIGHTS_NAME, **has_file_kwargs):
                                raise EnvironmentError(
                                    f"{pretrained_model_name_or_path} does not appear to have a file named"
                                    f" {_add_variant(WEIGHTS_NAME, variant)} but there is a file for Flax weights. Use"
                                    " `from_flax=True` to load this model from those weights."
                                )
                            elif variant is not None and has_file(
                                pretrained_model_name_or_path, WEIGHTS_NAME, **has_file_kwargs
                            ):
                                raise EnvironmentError(
                                    f"{pretrained_model_name_or_path} does not appear to have a file named"
                                    f" {_add_variant(WEIGHTS_NAME, variant)} but there is a file without the variant"
                                    f" {variant}. Use `variant=None` to load this model from those weights."
                                )
                            else:
                                raise EnvironmentError(
                                    f"{pretrained_model_name_or_path} does not appear to have a file named"
                                    f" {_add_variant(WEIGHTS_NAME, variant)}, {_add_variant(SAFE_WEIGHTS_NAME, variant)},"
                                    f" {TF2_WEIGHTS_NAME}, {TF_WEIGHTS_NAME} or {FLAX_WEIGHTS_NAME}."
                                )

                except EnvironmentError:
                    # Raise any environment error raise by `cached_file`. It will have a helpful error message adapted
                    # to the original exception.
                    raise
                except Exception as e:
                    # For any other exception, we throw a generic error.
                    raise EnvironmentError(
                        f"Can't load the model for '{pretrained_model_name_or_path}'. If you were trying to load it"
                        " from 'https://huggingface.co/models', make sure you don't have a local directory with the"
                        f" same name. Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a"
                        f" directory containing a file named {_add_variant(WEIGHTS_NAME, variant)},"
                        f" {TF2_WEIGHTS_NAME}, {TF_WEIGHTS_NAME} or {FLAX_WEIGHTS_NAME}."
                    ) from e

            if is_local:
                logger.info(f"loading weights file {archive_file}")
                resolved_archive_file = archive_file
            else:
                logger.info(f"loading weights file {filename} from cache at {resolved_archive_file}")
        elif gguf_file:
            from .modeling_gguf_pytorch_utils import load_gguf_checkpoint

            # Case 1: the GGUF file is present locally
            if os.path.isfile(gguf_file):
                gguf_path = gguf_file
            # Case 2: The GGUF path is a location on the Hub
            # Load from URL or cache if already cached
            else:
                cached_file_kwargs = {
                    "cache_dir": cache_dir,
                    "force_download": force_download,
                    "proxies": proxies,
                    "resume_download": resume_download,
                    "local_files_only": local_files_only,
                    "token": token,
                    "user_agent": user_agent,
                    "revision": revision,
                    "subfolder": subfolder,
                    "_raise_exceptions_for_gated_repo": False,
                    "_raise_exceptions_for_missing_entries": False,
                    "_commit_hash": commit_hash,
                }

                gguf_path = cached_file(pretrained_model_name_or_path, gguf_file, **cached_file_kwargs)

            # we need a dummy model to help rename state_dict
            with torch.device("meta"):
                dummy_model = cls(config)
            state_dict = load_gguf_checkpoint(gguf_path, return_tensors=True, model_to_load=dummy_model)["tensors"]

            resolved_archive_file = None
            is_sharded = False
        else:
            resolved_archive_file = None

        # We'll need to download and cache each checkpoint shard if the checkpoint is sharded.
        if is_sharded:
            # resolved_archive_file becomes a list of files that point to the different checkpoint shards in this case.
            resolved_archive_file, sharded_metadata = get_checkpoint_shard_files(
                pretrained_model_name_or_path,
                resolved_archive_file,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                local_files_only=local_files_only,
                token=token,
                user_agent=user_agent,
                revision=revision,
                subfolder=subfolder,
                _commit_hash=commit_hash,
            )

        if (
            is_safetensors_available()
            and isinstance(resolved_archive_file, str)
            and resolved_archive_file.endswith(".safetensors")
        ):
            with safe_open(resolved_archive_file, framework="pt") as f:
                metadata = f.metadata()

            if metadata is None:
                # Assume it's a pytorch checkpoint (introduced for timm checkpoints)
                pass
            elif metadata.get("format") == "pt":
                pass
            elif metadata.get("format") == "tf":
                from_tf = True
                logger.info("A TensorFlow safetensors file is being loaded in a PyTorch model.")
            elif metadata.get("format") == "flax":
                from_flax = True
                logger.info("A Flax safetensors file is being loaded in a PyTorch model.")
            elif metadata.get("format") == "mlx":
                # This is a mlx file, we assume weights are compatible with pt
                pass
            else:
                raise ValueError(
                    f"Incompatible safetensors file. File metadata is not ['pt', 'tf', 'flax', 'mlx'] but {metadata.get('format')}"
                )

        from_pt = not (from_tf | from_flax)

        # load pt weights early so that we know which dtype to init the model under

        if from_pt:
            if not is_sharded and state_dict is None:
                # Time to load the checkpoint
                state_dict = load_state_dict(resolved_archive_file, weights_only=weights_only)

            # set dtype to instantiate the model under:
            # 1. If torch_dtype is not None, we use that dtype
            # 2. If torch_dtype is "auto", we auto-detect dtype from the loaded state_dict, by checking its first
            #    weights entry that is of a floating type - we assume all floating dtype weights are of the same dtype
            # we also may have config.torch_dtype available, but we won't rely on it till v5
            dtype_orig = None

            if torch_dtype is not None:
                if isinstance(torch_dtype, str):
                    if torch_dtype == "auto":
                        if hasattr(config, "torch_dtype") and config.torch_dtype is not None:
                            torch_dtype = config.torch_dtype
                            logger.info(f"Will use torch_dtype={torch_dtype} as defined in model's config object")
                        else:
                            if is_sharded and "dtype" in sharded_metadata:
                                torch_dtype = sharded_metadata["dtype"]
                            elif not is_sharded:
                                torch_dtype = get_state_dict_dtype(state_dict)
                            else:
                                one_state_dict = load_state_dict(resolved_archive_file[0], weights_only=weights_only)
                                torch_dtype = get_state_dict_dtype(one_state_dict)
                                del one_state_dict  # free CPU memory
                            logger.info(
                                "Since the `torch_dtype` attribute can't be found in model's config object, "
                                "will use torch_dtype={torch_dtype} as derived from model's weights"
                            )
                    elif hasattr(torch, torch_dtype):
                        torch_dtype = getattr(torch, torch_dtype)
                        for sub_config_key in config.sub_configs.keys():
                            sub_config = getattr(config, sub_config_key)
                            sub_config.torch_dtype = torch_dtype
                elif isinstance(torch_dtype, torch.dtype):
                    for sub_config_key in config.sub_configs.keys():
                        sub_config = getattr(config, sub_config_key)
                        sub_config.torch_dtype = torch_dtype
                elif isinstance(torch_dtype, dict):
                    for key, curr_dtype in torch_dtype.items():
                        if hasattr(config, key):
                            value = getattr(config, key)
                            value.torch_dtype = curr_dtype
                    # main torch dtype for modules that aren't part of any sub-config
                    torch_dtype = torch_dtype.get("")
                    config.torch_dtype = torch_dtype
                    if isinstance(torch_dtype, str) and hasattr(torch, torch_dtype):
                        torch_dtype = getattr(torch, torch_dtype)
                    elif torch_dtype is None:
                        torch_dtype = torch.float32
                else:
                    raise ValueError(
                        f"`torch_dtype` can be one of: `torch.dtype`, `'auto'`, a string of a valid `torch.dtype` or a `dict` with valid `torch_dtype` "
                        f"for each sub-config in composite configs, but received {torch_dtype}"
                    )

                dtype_orig = cls._set_default_torch_dtype(torch_dtype)
            else:
                # set fp32 as the default dtype for BC
                default_dtype = str(torch.get_default_dtype()).split(".")[-1]
                config.torch_dtype = default_dtype
                for key in config.sub_configs.keys():
                    value = getattr(config, key)
                    value.torch_dtype = default_dtype

            # Check if `_keep_in_fp32_modules` is not None
            use_keep_in_fp32_modules = (cls._keep_in_fp32_modules is not None) and (
                (torch_dtype == torch.float16) or hasattr(hf_quantizer, "use_keep_in_fp32_modules")
            )

            if is_sharded:
                loaded_state_dict_keys = sharded_metadata["all_checkpoint_keys"]
            else:
                loaded_state_dict_keys = list(state_dict.keys())
            if (
                gguf_path is None
                and (low_cpu_mem_usage or (use_keep_in_fp32_modules and is_accelerate_available()))
                and pretrained_model_name_or_path is not None
            ):
                # In case some weights need to be kept in float32 and accelerate is not installed,
                # we later on want to take the path where state_dict is not None, that is the one
                # that do not require accelerate.
                state_dict = None

        config.name_or_path = pretrained_model_name_or_path

        # Instantiate model.
        init_contexts = [no_init_weights(_enable=_fast_init)]
        tp_device = None

        if is_deepspeed_zero3_enabled() and not is_quantized and not _is_ds_init_called:
            import deepspeed

            logger.info("Detected DeepSpeed ZeRO-3: activating zero.init() for this model")
            init_contexts = [
                deepspeed.zero.Init(config_dict_or_path=deepspeed_config()),
                set_zero3_state(),
            ] + init_contexts
        elif low_cpu_mem_usage:
            if not is_accelerate_available():
                raise ImportError(
                    f"Using `low_cpu_mem_usage=True` or a `device_map` requires Accelerate: `pip install 'accelerate>={ACCELERATE_MIN_VERSION}'`"
                )
            init_contexts.append(init_empty_weights())
        elif tp_plan is not None:
            if not torch.distributed.is_initialized():
                raise ValueError("Tensor Parallel requires torch.distributed to be initialized first.")

            # Detect the accelerator on the machine. If no accelerator is available, it returns CPU.
            device_type = torch._C._get_accelerator().type
            device_module = torch.get_device_module(device_type)
            # Get device with index assuming equal number of devices per host
            tp_device = torch.device(device_type, torch.distributed.get_rank() % device_module.device_count())
            init_contexts.append(tp_device)

        if is_deepspeed_zero3_enabled() and is_quantized:
            init_contexts.append(set_quantized_state())

        config = copy.deepcopy(config)  # We do not want to modify the config inplace in from_pretrained.
        if not getattr(config, "_attn_implementation_autoset", False):
            config = cls._autoset_attn_implementation(
                config, use_flash_attention_2=use_flash_attention_2, torch_dtype=torch_dtype, device_map=device_map
            )

        with ContextManagers(init_contexts):
            # Let's make sure we don't run the init function of buffer modules
            model = cls(config, *model_args, **model_kwargs)

        # make sure we use the model's config since the __init__ call might have copied it
        config = model.config

        # Check first if we are `from_pt`
        if use_keep_in_fp32_modules:
            if is_accelerate_available() and not is_deepspeed_zero3_enabled():
                low_cpu_mem_usage = True
            keep_in_fp32_modules = model._keep_in_fp32_modules
        else:
            keep_in_fp32_modules = []

        if hf_quantizer is not None:
            hf_quantizer.preprocess_model(
                model=model, device_map=device_map, keep_in_fp32_modules=keep_in_fp32_modules
            )

            # We store the original dtype for quantized models as we cannot easily retrieve it
            # once the weights have been quantized
            # Note that once you have loaded a quantized model, you can't change its dtype so this will
            # remain a single source of truth
            config._pre_quantization_dtype = torch_dtype

        if isinstance(device_map, str):
            special_dtypes = {}

            if hf_quantizer is not None:
                special_dtypes.update(hf_quantizer.get_special_dtypes_update(model, torch_dtype))

            special_dtypes.update(
                {
                    name: torch.float32
                    for name, _ in model.named_parameters()
                    if any(m in name for m in keep_in_fp32_modules)
                }
            )

            target_dtype = torch_dtype

            if hf_quantizer is not None:
                target_dtype = hf_quantizer.adjust_target_dtype(target_dtype)

            no_split_modules = model._get_no_split_modules(device_map)
            if device_map not in ["auto", "balanced", "balanced_low_0", "sequential"]:
                raise ValueError(
                    "If passing a string for `device_map`, please choose 'auto', 'balanced', 'balanced_low_0' or "
                    "'sequential'."
                )

            device_map_kwargs = {"no_split_module_classes": no_split_modules}
            if "special_dtypes" in inspect.signature(infer_auto_device_map).parameters:
                device_map_kwargs["special_dtypes"] = special_dtypes
            elif len(special_dtypes) > 0:
                logger.warning(
                    "This model has some weights that should be kept in higher precision, you need to upgrade "
                    "`accelerate` to properly deal with them (`pip install --upgrade accelerate`)."
                )
            if device_map != "sequential":
                max_memory = get_balanced_memory(
                    model,
                    dtype=target_dtype,
                    low_zero=(device_map == "balanced_low_0"),
                    max_memory=max_memory,
                    **device_map_kwargs,
                )
            else:
                max_memory = get_max_memory(max_memory)
            if hf_quantizer is not None:
                max_memory = hf_quantizer.adjust_max_memory(max_memory)
            device_map_kwargs["max_memory"] = max_memory

            # Make sure tied weights are tied before creating the device map.
            model.tie_weights()
            device_map = infer_auto_device_map(model, dtype=target_dtype, **device_map_kwargs)

            if hf_quantizer is not None:
                hf_quantizer.validate_environment(device_map=device_map)

        elif device_map is not None:
            model.tie_weights()
            tied_params = find_tied_parameters(model)
            # check if we don't have tied param in different devices
            check_tied_parameters_on_same_device(tied_params, device_map)

        if from_tf:
            if resolved_archive_file.endswith(".index"):
                # Load from a TensorFlow 1.X checkpoint - provided by original authors
                model = cls.load_tf_weights(model, config, resolved_archive_file[:-6])  # Remove the '.index'
            else:
                # Load from our TensorFlow 2.0 checkpoints
                try:
                    from .modeling_tf_pytorch_utils import load_tf2_checkpoint_in_pytorch_model

                    model, loading_info = load_tf2_checkpoint_in_pytorch_model(
                        model, resolved_archive_file, allow_missing_keys=True, output_loading_info=True
                    )
                except ImportError:
                    logger.error(
                        "Loading a TensorFlow model in PyTorch, requires both PyTorch and TensorFlow to be installed."
                        " Please see https://pytorch.org/ and https://www.tensorflow.org/install/ for installation"
                        " instructions."
                    )
                    raise
        elif from_flax:
            try:
                from .modeling_flax_pytorch_utils import load_flax_checkpoint_in_pytorch_model

                model = load_flax_checkpoint_in_pytorch_model(model, resolved_archive_file)
            except ImportError:
                logger.error(
                    "Loading a Flax model in PyTorch, requires both PyTorch and Flax to be installed. Please see"
                    " https://pytorch.org/ and https://flax.readthedocs.io/en/latest/installation.html for"
                    " installation instructions."
                )
                raise
        elif from_pt:
            # restore default dtype
            if dtype_orig is not None:
                torch.set_default_dtype(dtype_orig)

            load_contexts = []
            # Make sure we load onto targeted device
            if tp_device is not None:
                load_contexts.append(tp_device)

            with ContextManagers(load_contexts):
                (
                    model,
                    missing_keys,
                    unexpected_keys,
                    mismatched_keys,
                    offload_index,
                    error_msgs,
                ) = cls._load_pretrained_model(
                    model,
                    state_dict,
                    loaded_state_dict_keys,  # XXX: rename?
                    resolved_archive_file,
                    pretrained_model_name_or_path,
                    ignore_mismatched_sizes=ignore_mismatched_sizes,
                    sharded_metadata=sharded_metadata,
                    _fast_init=_fast_init,
                    low_cpu_mem_usage=low_cpu_mem_usage,
                    device_map=device_map,
                    offload_folder=offload_folder,
                    offload_state_dict=offload_state_dict,
                    dtype=torch_dtype,
                    hf_quantizer=hf_quantizer,
                    keep_in_fp32_modules=keep_in_fp32_modules,
                    gguf_path=gguf_path,
                    weights_only=weights_only,
                )

        # make sure token embedding weights are still tied if needed
        model.tie_weights()

        # Set model in evaluation mode to deactivate DropOut modules by default
        model.eval()

        # If it is a model with generation capabilities, attempt to load the generation config
        if model.can_generate() and generation_config is not None:
            logger.info("The user-defined `generation_config` will be used to override the default generation config.")
            model.generation_config = model.generation_config.from_dict(generation_config.to_dict())
        elif model.can_generate() and pretrained_model_name_or_path is not None:
            try:
                model.generation_config = GenerationConfig.from_pretrained(
                    pretrained_model_name_or_path,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    revision=revision,
                    subfolder=subfolder,
                    _from_auto=from_auto_class,
                    _from_pipeline=from_pipeline,
                    **kwargs,
                )
            except OSError:
                logger.info(
                    "Generation config file not found, using a generation config created from the model config."
                )
                pass

        # Dispatch model with hooks on all devices if necessary
        if device_map is not None:
            device_map_kwargs = {
                "device_map": device_map,
                "offload_dir": offload_folder,
                "offload_index": offload_index,
                "offload_buffers": offload_buffers,
            }
            if "skip_keys" in inspect.signature(dispatch_model).parameters:
                device_map_kwargs["skip_keys"] = model._skip_keys_device_placement
            # For HQQ method we force-set the hooks for single GPU envs
            if (
                "force_hooks" in inspect.signature(dispatch_model).parameters
                and hf_quantizer is not None
                and hf_quantizer.quantization_config.quant_method == QuantizationMethod.HQQ
            ):
                device_map_kwargs["force_hooks"] = True
            if (
                hf_quantizer is not None
                and hf_quantizer.quantization_config.quant_method == QuantizationMethod.FBGEMM_FP8
                and isinstance(device_map, dict)
                and ("cpu" in device_map.values() or "disk" in device_map.values())
            ):
                device_map_kwargs["offload_buffers"] = True

            if not is_fsdp_enabled() and not is_deepspeed_zero3_enabled():
                dispatch_model(model, **device_map_kwargs)

        if hf_quantizer is not None:
            hf_quantizer.postprocess_model(model, config=config)
            model.hf_quantizer = hf_quantizer

        if _adapter_model_path is not None:
            model.load_adapter(
                _adapter_model_path,
                adapter_name=adapter_name,
                token=token,
                adapter_kwargs=adapter_kwargs,
            )

        if output_loading_info:
            if loading_info is None:
                loading_info = {
                    "missing_keys": missing_keys,
                    "unexpected_keys": unexpected_keys,
                    "mismatched_keys": mismatched_keys,
                    "error_msgs": error_msgs,
                }
            return model, loading_info

        if tp_plan is not None:
            assert tp_device is not None, "tp_device not set!"
            if not model.supports_tp_plan:
                raise NotImplementedError("This model does not have a tensor parallel plan.")
            # Assuming sharding the model onto the world
            world_size = torch.distributed.get_world_size()
            device_mesh = torch.distributed.init_device_mesh(tp_device.type, (world_size,))
            # Apply Tensor Parallelism
            model.tensor_parallel(device_mesh)

        return model

    @staticmethod
    def _fix_state_dict_key_on_load(key) -> Tuple[str, bool]:
        """Replace legacy parameter names with their modern equivalents. E.g. beta -> bias, gamma -> weight."""

        # Rename LayerNorm beta & gamma params for some early models ported from Tensorflow (e.g. Bert)
        # This rename is logged.
        if key.endswith("LayerNorm.beta"):
            return key.replace("LayerNorm.beta", "LayerNorm.bias"), True
        if key.endswith("LayerNorm.gamma"):
            return key.replace("LayerNorm.gamma", "LayerNorm.weight"), True

        # Rename weight norm parametrizations to match changes across torch versions.
        # Impacts a number of speech/wav2vec models. e.g. Hubert, Wav2Vec2, and others.
        # This rename is not logged.
        if hasattr(nn.utils.parametrizations, "weight_norm"):
            if key.endswith("weight_g"):
                return key.replace("weight_g", "parametrizations.weight.original0"), True
            if key.endswith("weight_v"):
                return key.replace("weight_v", "parametrizations.weight.original1"), True
        else:
            if key.endswith("parametrizations.weight.original0"):
                return key.replace("parametrizations.weight.original0", "weight_g"), True
            if key.endswith("parametrizations.weight.original1"):
                return key.replace("parametrizations.weight.original1", "weight_v"), True

        return key, False

    @classmethod
    def _fix_state_dict_keys_on_load(cls, state_dict):
        """Fixes state dict keys by replacing legacy parameter names with their modern equivalents.
        Logs if any parameters have been renamed.
        """

        renamed_keys = {}
        state_dict_keys = list(state_dict.keys())
        for key in state_dict_keys:
            new_key, has_changed = cls._fix_state_dict_key_on_load(key)
            if has_changed:
                state_dict[new_key] = state_dict.pop(key)

                # track gamma/beta rename for logging
                if key.endswith("LayerNorm.gamma"):
                    renamed_keys["LayerNorm.gamma"] = (key, new_key)
                elif key.endswith("LayerNorm.beta"):
                    renamed_keys["LayerNorm.beta"] = (key, new_key)

        if renamed_keys:
            warning_msg = f"A pretrained model of type `{cls.__name__}` "
            warning_msg += "contains parameters that have been renamed internally (a few are listed below but more are present in the model):\n"
            for old_key, new_key in renamed_keys.values():
                warning_msg += f"* `{old_key}` -> `{new_key}`\n"
            warning_msg += "If you are using a model from the Hub, consider submitting a PR to adjust these weights and help future users."
            logger.info_once(warning_msg)

        return state_dict

    @staticmethod
    def _fix_state_dict_key_on_save(key) -> Tuple[str, bool]:
        """
        Similar to `_fix_state_dict_key_on_load` allows to define hook for state dict key renaming on model save.
        Do nothing by default, but can be overridden in particular models.
        """
        return key, False

    def _fix_state_dict_keys_on_save(self, state_dict):
        """
        Similar to `_fix_state_dict_keys_on_load` allows to define hook for state dict key renaming on model save.
        Apply `_fix_state_dict_key_on_save` to all keys in `state_dict`.
        """
        return {self._fix_state_dict_key_on_save(key)[0]: value for key, value in state_dict.items()}

    @classmethod
    def _load_pretrained_model(
        cls,
        model,
        state_dict,
        loaded_keys,
        resolved_archive_file,
        pretrained_model_name_or_path,
        ignore_mismatched_sizes=False,
        sharded_metadata=None,
        _fast_init=True,
        low_cpu_mem_usage=False,
        device_map=None,
        offload_folder=None,
        offload_state_dict=None,
        dtype=None,
        hf_quantizer=None,
        keep_in_fp32_modules=None,
        gguf_path=None,
        weights_only=True,
    ):
        is_safetensors = False
        is_quantized = hf_quantizer is not None
        state_dict_folder = None
        state_dict_index = None

        if device_map is not None and "disk" in device_map.values():
            archive_file = (
                resolved_archive_file[0] if isinstance(resolved_archive_file, (list, tuple)) else resolved_archive_file
            )
            is_safetensors = archive_file.endswith(".safetensors")
            if offload_folder is None and not is_safetensors:
                raise ValueError(
                    "The current `device_map` had weights offloaded to the disk. Please provide an `offload_folder`"
                    " for them. Alternatively, make sure you have `safetensors` installed if the model you are using"
                    " offers the weights in this format."
                )
            if offload_folder is not None:
                os.makedirs(offload_folder, exist_ok=True)
            if offload_state_dict is None:
                offload_state_dict = True

        is_sharded_safetensors = is_safetensors and sharded_metadata is not None

        # tie the model weights before retrieving the state_dict
        model.tie_weights()

        # Retrieve missing & unexpected_keys
        model_state_dict = model.state_dict()
        expected_keys = list(model_state_dict.keys())
        prefix = model.base_model_prefix

        if hf_quantizer is not None:
            expected_keys = hf_quantizer.update_expected_keys(model, expected_keys, loaded_keys)

        original_loaded_keys = loaded_keys
        loaded_keys = [cls._fix_state_dict_key_on_load(key)[0] for key in loaded_keys]

        if len(prefix) > 0:
            has_prefix_module = any(s.startswith(prefix) for s in loaded_keys)
            expects_prefix_module = any(s.startswith(prefix) for s in expected_keys)
        else:
            has_prefix_module = False
            expects_prefix_module = False

        # key re-naming operations are never done on the keys
        # that are loaded, but always on the keys of the newly initialized model
        remove_prefix_from_model = not has_prefix_module and expects_prefix_module
        add_prefix_to_model = has_prefix_module and not expects_prefix_module

        if remove_prefix_from_model:
            _prefix = f"{prefix}."
            expected_keys_not_prefixed = [s for s in expected_keys if not s.startswith(_prefix)]
            expected_keys = [s[len(_prefix) :] if s.startswith(_prefix) else s for s in expected_keys]
        elif add_prefix_to_model:
            expected_keys = [".".join([prefix, s]) for s in expected_keys]

        missing_keys = sorted(set(expected_keys) - set(loaded_keys))
        unexpected_keys = set(loaded_keys) - set(expected_keys)

        # Remove nonpersistent buffers from unexpected keys: they are not in the state dict but will be in the model
        # buffers
        model_buffers = {n for n, _ in model.named_buffers()}
        if remove_prefix_from_model:
            model_buffers = {key[len(_prefix) :] if key.startswith(_prefix) else key for key in model_buffers}
        elif add_prefix_to_model:
            model_buffers = {".".join([prefix, key]) for key in model_buffers}
        unexpected_keys = sorted(unexpected_keys - model_buffers)

        # Clean up buffer for `inv-freq` because RoPE embedding moved under base model (https://github.com/huggingface/transformers/pull/34858)
        has_inv_freq_buffers = any(buffer.endswith("rotary_emb.inv_freq") for buffer in model_buffers)
        if has_inv_freq_buffers:
            unexpected_keys = {k for k in unexpected_keys if "rotary_emb.inv_freq" not in k}

        model.tie_weights()
        if device_map is None and not is_fsdp_enabled() and not is_deepspeed_zero3_enabled():
            ptrs = collections.defaultdict(list)
            for name, tensor in model.state_dict().items():
                id_tensor = id_tensor_storage(tensor)
                ptrs[id_tensor].append(name)

            # These are all the pointers of shared tensors.
            tied_params = [names for _, names in ptrs.items() if len(names) > 1]
        else:
            # id function doesn't work for meta tensor so we need this function
            tied_params = find_tied_parameters(model)

        for group in tied_params:
            if remove_prefix_from_model:
                group = [key[len(_prefix) :] if key.startswith(_prefix) else key for key in group]
            elif add_prefix_to_model:
                group = [".".join([prefix, key]) for key in group]
            missing_in_group = [k for k in missing_keys if k in group]
            if len(missing_in_group) > 0 and len(missing_in_group) < len(group):
                missing_keys = [k for k in missing_keys if k not in missing_in_group]

        # Some models may have keys that are not in the state by design, removing them before needlessly warning
        # the user.
        if cls._keys_to_ignore_on_load_missing is not None:
            for pat in cls._keys_to_ignore_on_load_missing:
                missing_keys = [k for k in missing_keys if re.search(pat, k) is None]

        if cls._keys_to_ignore_on_load_unexpected is not None:
            for pat in cls._keys_to_ignore_on_load_unexpected:
                unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]
        if hf_quantizer is not None:
            missing_keys = hf_quantizer.update_missing_keys(model, missing_keys, prefix)

        # retrieve weights on meta device and put them back on CPU.
        # This is not ideal in terms of memory, but if we don't do that not, we can't initialize them in the next step
        if low_cpu_mem_usage:
            for key in missing_keys:
                if key in list(model_state_dict.keys()):
                    key = key
                elif f"{prefix}.{key}" in list(model_state_dict.keys()):
                    key = f"{prefix}.{key}"
                elif key.startswith(prefix) and ".".join(key.split(".")[1:]) in list(model_state_dict.keys()):
                    key = ".".join(key.split(".")[1:])
                param = model_state_dict[key]

                # upcast in fp32 if any
                target_dtype = dtype
                if (
                    keep_in_fp32_modules is not None
                    and dtype == torch.float16
                    and any(
                        module_to_keep_in_fp32 in key.split(".") for module_to_keep_in_fp32 in keep_in_fp32_modules
                    )
                ):
                    target_dtype = torch.float32

                if param.device == torch.device("meta"):
                    value = torch.empty(*param.size(), dtype=target_dtype)
                    if (
                        not is_quantized
                        or (getattr(hf_quantizer, "requires_parameters_quantization", False))
                        or not hf_quantizer.check_quantized_param(
                            model, param_value=value, param_name=key, state_dict={}
                        )
                    ):
                        set_module_tensor_to_device(model, key, "cpu", value)
                    else:
                        hf_quantizer.create_quantized_param(model, value, key, "cpu", state_dict, unexpected_keys)

        # retrieve uninitialized modules and initialize before maybe overriding that with the pretrained weights.
        if _fast_init:
            if not ignore_mismatched_sizes:
                if remove_prefix_from_model:
                    _loaded_keys = [f"{prefix}.{k}" for k in loaded_keys]
                elif add_prefix_to_model:
                    _loaded_keys = [k[len(prefix) + 1 :] for k in loaded_keys]
                else:
                    _loaded_keys = loaded_keys
                not_initialized_submodules = set_initialized_submodules(model, _loaded_keys)
                # If we're about to tie the output embeds to the input embeds we don't need to init them
                if (
                    hasattr(model.config.get_text_config(decoder=True), "tie_word_embeddings")
                    and model.config.get_text_config(decoder=True).tie_word_embeddings
                ):
                    output_embeddings = model.get_output_embeddings()
                    if output_embeddings is not None:
                        # Still need to initialize if there is a bias term since biases are not tied.
                        if not hasattr(output_embeddings, "bias") or output_embeddings.bias is None:
                            output_embeddings._is_hf_initialized = True
            else:
                not_initialized_submodules = dict(model.named_modules())
            # This will only initialize submodules that are not marked as initialized by the line above.
            if is_deepspeed_zero3_enabled() and not is_quantized:
                import deepspeed

                not_initialized_parameters = list(
                    set(
                        itertools.chain.from_iterable(
                            submodule.parameters(recurse=False) for submodule in not_initialized_submodules.values()
                        )
                    )
                )
                with deepspeed.zero.GatheredParameters(not_initialized_parameters, modifier_rank=0):
                    model.apply(model._initialize_weights)
            else:
                model.apply(model._initialize_weights)

        # Set some modules to fp32 if any
        if keep_in_fp32_modules is not None:
            for name, param in model.named_parameters():
                if any(module_to_keep_in_fp32 in name.split(".") for module_to_keep_in_fp32 in keep_in_fp32_modules):
                    # param = param.to(torch.float32) does not work here as only in the local scope.
                    param.data = param.data.to(torch.float32)

        # Make sure we are able to load base models as well as derived models (with heads)
        start_prefix = ""
        model_to_load = model
        if len(cls.base_model_prefix) > 0 and not hasattr(model, cls.base_model_prefix) and has_prefix_module:
            start_prefix = cls.base_model_prefix + "."
        if len(cls.base_model_prefix) > 0 and hasattr(model, cls.base_model_prefix) and not has_prefix_module:
            model_to_load = getattr(model, cls.base_model_prefix)
            base_model_expected_keys = list(model_to_load.state_dict().keys())
            if any(key in expected_keys_not_prefixed and key not in base_model_expected_keys for key in loaded_keys):
                raise ValueError(
                    "The state dictionary of the model you are trying to load is corrupted. Are you sure it was "
                    "properly saved?"
                )
            if device_map is not None:
                device_map = {k.replace(f"{cls.base_model_prefix}.", ""): v for k, v in device_map.items()}

        def _find_mismatched_keys(
            state_dict,
            model_state_dict,
            loaded_keys,
            original_loaded_keys,
            add_prefix_to_model,
            remove_prefix_from_model,
            ignore_mismatched_sizes,
        ):
            mismatched_keys = []
            if ignore_mismatched_sizes:
                for checkpoint_key, model_key in zip(original_loaded_keys, loaded_keys):
                    # If the checkpoint is sharded, we may not have the key here.
                    if checkpoint_key not in state_dict:
                        continue
                    if remove_prefix_from_model:
                        # The model key starts with `prefix` but `checkpoint_key` doesn't so we add it.
                        model_key = f"{prefix}.{model_key}"
                    elif add_prefix_to_model:
                        # The model key doesn't start with `prefix` but `checkpoint_key` does so we remove it.
                        model_key = ".".join(model_key.split(".")[1:])

                    if (
                        model_key in model_state_dict
                        and state_dict[checkpoint_key].shape != model_state_dict[model_key].shape
                    ):
                        if (
                            state_dict[checkpoint_key].shape[-1] == 1
                            and state_dict[checkpoint_key].numel() * 2 == model_state_dict[model_key].numel()
                        ):
                            # This skips size mismatches for 4-bit weights. Two 4-bit values share an 8-bit container, causing size differences.
                            # Without matching with module type or paramter type it seems like a practical way to detect valid 4bit weights.
                            pass
                        else:
                            mismatched_keys.append(
                                (checkpoint_key, state_dict[checkpoint_key].shape, model_state_dict[model_key].shape)
                            )
                            del state_dict[checkpoint_key]
            return mismatched_keys

        if resolved_archive_file is not None:
            folder = os.path.sep.join(resolved_archive_file[0].split(os.path.sep)[:-1])
        else:
            folder = None
        if device_map is not None and is_safetensors:
            param_device_map = expand_device_map(device_map, original_loaded_keys, start_prefix)
            str_dtype = str(dtype).replace("torch.", "") if dtype is not None else "float32"
            if sharded_metadata is None:
                archive_file = (
                    resolved_archive_file[0]
                    if isinstance(resolved_archive_file, (list, tuple))
                    else resolved_archive_file
                )
                weight_map = {p: archive_file for p in original_loaded_keys}
            else:
                weight_map = {p: os.path.join(folder, f) for p, f in sharded_metadata["weight_map"].items()}
            offload_index = {
                p[len(start_prefix) :]: {"safetensors_file": f, "weight_name": p, "dtype": str_dtype}
                for p, f in weight_map.items()
                if p.startswith(start_prefix) and param_device_map[p[len(start_prefix) :]] == "disk"
            }
        else:
            offload_index = None

        if state_dict is not None:
            # Whole checkpoint
            mismatched_keys = _find_mismatched_keys(
                state_dict,
                model_state_dict,
                loaded_keys,
                original_loaded_keys,
                add_prefix_to_model,
                remove_prefix_from_model,
                ignore_mismatched_sizes,
            )

            # For GGUF models `state_dict` is never set to None as the state dict is always small
            if gguf_path or low_cpu_mem_usage:
                fixed_state_dict = cls._fix_state_dict_keys_on_load(state_dict)
                error_msgs, offload_index, state_dict_index = _load_state_dict_into_meta_model(
                    model_to_load,
                    fixed_state_dict,
                    start_prefix,
                    expected_keys,
                    device_map=device_map,
                    offload_folder=offload_folder,
                    offload_index=offload_index,
                    state_dict_folder=state_dict_folder,
                    state_dict_index=state_dict_index,
                    dtype=dtype,
                    hf_quantizer=hf_quantizer,
                    is_safetensors=is_safetensors,
                    keep_in_fp32_modules=keep_in_fp32_modules,
                    unexpected_keys=unexpected_keys,
                )
            else:
                # Sharded checkpoint or whole but low_cpu_mem_usage==True
                assign_to_params_buffers = check_support_param_buffer_assignment(
                    model_to_load, state_dict, start_prefix
                )
                fixed_state_dict = cls._fix_state_dict_keys_on_load(state_dict)
                error_msgs = _load_state_dict_into_model(
                    model_to_load, fixed_state_dict, start_prefix, assign_to_params_buffers
                )

        else:
            # This should always be a list but, just to be sure.
            if not isinstance(resolved_archive_file, list):
                resolved_archive_file = [resolved_archive_file]

            error_msgs = []
            mismatched_keys = []
            if not is_safetensors:
                offload_index = {} if device_map is not None and "disk" in device_map.values() else None
            if offload_state_dict:
                state_dict_folder = tempfile.mkdtemp()
                state_dict_index = {}
            else:
                state_dict_folder = None
                state_dict_index = None

            if is_sharded_safetensors:
                disk_only_shard_files = get_disk_only_shard_files(
                    device_map, sharded_metadata=sharded_metadata, start_prefix=start_prefix
                )
                disk_only_shard_files = [os.path.join(folder, f) for f in disk_only_shard_files]
            else:
                disk_only_shard_files = []

            if len(resolved_archive_file) > 1:
                resolved_archive_file = logging.tqdm(resolved_archive_file, desc="Loading checkpoint shards")
            assign_to_params_buffers = None
            for shard_file in resolved_archive_file:
                # Skip the load for shards that only contain disk-offloaded weights when using safetensors for the offload.
                if shard_file in disk_only_shard_files:
                    continue
                map_location = None
                if (
                    device_map is not None
                    and hf_quantizer is not None
                    and hf_quantizer.quantization_config.quant_method == QuantizationMethod.TORCHAO
                    and hf_quantizer.quantization_config.quant_type == "int4_weight_only"
                ):
                    map_location = torch.device([d for d in device_map.values() if d not in ["cpu", "disk"]][0])
                state_dict = load_state_dict(
                    shard_file, is_quantized=is_quantized, map_location=map_location, weights_only=weights_only
                )

                # Mistmatched keys contains tuples key/shape1/shape2 of weights in the checkpoint that have a shape not
                # matching the weights in the model.
                mismatched_keys += _find_mismatched_keys(
                    state_dict,
                    model_state_dict,
                    loaded_keys,
                    original_loaded_keys,
                    add_prefix_to_model,
                    remove_prefix_from_model,
                    ignore_mismatched_sizes,
                )
                if low_cpu_mem_usage:
                    if is_fsdp_enabled() and not is_local_dist_rank_0() and not is_quantized:
                        for key, param in model_to_load.state_dict().items():
                            if param.device == torch.device("meta"):
                                set_module_tensor_to_device(
                                    model_to_load, key, "cpu", torch.empty(*param.size(), dtype=dtype)
                                )
                    else:
                        fixed_state_dict = cls._fix_state_dict_keys_on_load(state_dict)
                        new_error_msgs, offload_index, state_dict_index = _load_state_dict_into_meta_model(
                            model_to_load,
                            fixed_state_dict,
                            start_prefix,
                            expected_keys,
                            device_map=device_map,
                            offload_folder=offload_folder,
                            offload_index=offload_index,
                            state_dict_folder=state_dict_folder,
                            state_dict_index=state_dict_index,
                            dtype=dtype,
                            hf_quantizer=hf_quantizer,
                            is_safetensors=is_safetensors,
                            keep_in_fp32_modules=keep_in_fp32_modules,
                            unexpected_keys=unexpected_keys,
                        )
                        error_msgs += new_error_msgs
                else:
                    # Sharded checkpoint or whole but low_cpu_mem_usage==True
                    if assign_to_params_buffers is None:
                        assign_to_params_buffers = check_support_param_buffer_assignment(
                            model_to_load, state_dict, start_prefix
                        )
                    fixed_state_dict = cls._fix_state_dict_keys_on_load(state_dict)
                    error_msgs += _load_state_dict_into_model(
                        model_to_load, fixed_state_dict, start_prefix, assign_to_params_buffers
                    )

                # force memory release
                del state_dict
                gc.collect()

            if offload_index is not None and len(offload_index) > 0:
                if model != model_to_load:
                    # We need to add the prefix of the base model
                    prefix = cls.base_model_prefix
                    if not is_safetensors:
                        for weight_name in offload_index:
                            shutil.move(
                                os.path.join(offload_folder, f"{weight_name}.dat"),
                                os.path.join(offload_folder, f"{prefix}.{weight_name}.dat"),
                            )
                    offload_index = {f"{prefix}.{key}": value for key, value in offload_index.items()}
                if not is_safetensors:
                    save_offload_index(offload_index, offload_folder)
                    offload_index = None

            if offload_state_dict:
                # Load back temporarily offloaded state dict
                load_offloaded_weights(model_to_load, state_dict_index, state_dict_folder)
                shutil.rmtree(state_dict_folder)

        if len(error_msgs) > 0:
            error_msg = "\n\t".join(error_msgs)
            if "size mismatch" in error_msg:
                error_msg += (
                    "\n\tYou may consider adding `ignore_mismatched_sizes=True` in the model `from_pretrained` method."
                )
            raise RuntimeError(f"Error(s) in loading state_dict for {model.__class__.__name__}:\n\t{error_msg}")

        if len(unexpected_keys) > 0:
            archs = [] if model.config.architectures is None else model.config.architectures
            warner = logger.warning if model.__class__.__name__ in archs else logger.info
            warner(
                f"Some weights of the model checkpoint at {pretrained_model_name_or_path} were not used when"
                f" initializing {model.__class__.__name__}: {unexpected_keys}\n- This IS expected if you are"
                f" initializing {model.__class__.__name__} from the checkpoint of a model trained on another task or"
                " with another architecture (e.g. initializing a BertForSequenceClassification model from a"
                " BertForPreTraining model).\n- This IS NOT expected if you are initializing"
                f" {model.__class__.__name__} from the checkpoint of a model that you expect to be exactly identical"
                " (initializing a BertForSequenceClassification model from a BertForSequenceClassification model)."
            )
        else:
            logger.info(f"All model checkpoint weights were used when initializing {model.__class__.__name__}.\n")
        if len(missing_keys) > 0:
            logger.warning(
                f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path} and are newly initialized: {missing_keys}\nYou should probably"
                " TRAIN this model on a down-stream task to be able to use it for predictions and inference."
            )
        elif len(mismatched_keys) == 0:
            logger.info(
                f"All the weights of {model.__class__.__name__} were initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path}.\nIf your task is similar to the task the model of the checkpoint"
                f" was trained on, you can already use {model.__class__.__name__} for predictions without further"
                " training."
            )
        if len(mismatched_keys) > 0:
            mismatched_warning = "\n".join(
                [
                    f"- {key}: found shape {shape1} in the checkpoint and {shape2} in the model instantiated"
                    for key, shape1, shape2 in mismatched_keys
                ]
            )
            logger.warning(
                f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path} and are newly initialized because the shapes did not"
                f" match:\n{mismatched_warning}\nYou should probably TRAIN this model on a down-stream task to be able"
                " to use it for predictions and inference."
            )

        return model, missing_keys, unexpected_keys, mismatched_keys, offload_index, error_msgs

    def retrieve_modules_from_names(self, names, add_prefix=False, remove_prefix=False):
        module_keys = {".".join(key.split(".")[:-1]) for key in names}

        # torch.nn.ParameterList is a special case where two parameter keywords
        # are appended to the module name, *e.g.* bert.special_embeddings.0
        module_keys = module_keys.union(
            {".".join(key.split(".")[:-2]) for key in names if len(key) > 0 and key[-1].isdigit()}
        )

        retrieved_modules = []
        # retrieve all modules that has at least one missing weight name
        for name, module in self.named_modules():
            if remove_prefix:
                _prefix = f"{self.base_model_prefix}."
                name = name[len(_prefix) :] if name.startswith(_prefix) else name
            elif add_prefix:
                name = ".".join([self.base_model_prefix, name]) if len(name) > 0 else self.base_model_prefix

            if name in module_keys:
                retrieved_modules.append(module)

        return retrieved_modules

    @staticmethod
    def _load_pretrained_model_low_mem(
        model,
        loaded_state_dict_keys,
        resolved_archive_file,
        start_prefix="",
        hf_quantizer=None,
        pretrained_model_name_or_path=None,
        weights_only=True,
    ):
        """
        This is an experimental function that loads the model using ~1.x model size CPU memory

        Before you call it do:

        1. save which state_dict keys are available
        2. drop state_dict before model is created, since the latter takes 1x model size memory

        Here then we continue:

        3. switch to the meta device all params/buffers that are going to be replaced from the loaded state_dict
        4. load state_dict 2nd time
        5. replace the params/buffers from the state_dict

        Currently, it doesn't handle missing_keys, unexpected_keys, mismatched_keys. It can't handle deepspeed. To
        handle bitsandbytes, needs non-empty hf_quantizer argument.
        """

        _move_model_to_meta(model, loaded_state_dict_keys, start_prefix)
        state_dict = load_state_dict(resolved_archive_file, weights_only=weights_only)
        expected_keys = loaded_state_dict_keys  # plug for missing expected_keys. TODO: replace with proper keys
        fixed_state_dict = model._fix_state_dict_keys_on_load(state_dict)
        error_msgs = _load_state_dict_into_meta_model(
            model,
            fixed_state_dict,
            start_prefix,
            expected_keys=expected_keys,
            hf_quantizer=hf_quantizer,
        )
        return error_msgs

    @classmethod
    def register_for_auto_class(cls, auto_class="AutoModel"):
        """
        Register this class with a given auto class. This should only be used for custom models as the ones in the
        library are already mapped with an auto class.

        <Tip warning={true}>

        This API is experimental and may have some slight breaking changes in the next releases.

        </Tip>

        Args:
            auto_class (`str` or `type`, *optional*, defaults to `"AutoModel"`):
                The auto class to register this new model with.
        """
        if not isinstance(auto_class, str):
            auto_class = auto_class.__name__

        import transformers.models.auto as auto_module

        if not hasattr(auto_module, auto_class):
            raise ValueError(f"{auto_class} is not a valid auto class.")

        cls._auto_class = auto_class

    def to_bettertransformer(self) -> "PreTrainedModel":
        """
        Converts the model to use [PyTorch's native attention
        implementation](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html), integrated to
        Transformers through [Optimum library](https://huggingface.co/docs/optimum/bettertransformer/overview). Only a
        subset of all Transformers models are supported.

        PyTorch's attention fastpath allows to speed up inference through kernel fusions and the use of [nested
        tensors](https://pytorch.org/docs/stable/nested.html). Detailed benchmarks can be found in [this blog
        post](https://medium.com/pytorch/bettertransformer-out-of-the-box-performance-for-huggingface-transformers-3fbe27d50ab2).

        Returns:
            [`PreTrainedModel`]: The model converted to BetterTransformer.
        """
        if not is_optimum_available():
            raise ImportError("The package `optimum` is required to use Better Transformer.")

        from optimum.version import __version__ as optimum_version

        if version.parse(optimum_version) < version.parse("1.7.0"):
            raise ImportError(
                f"Please install optimum>=1.7.0 to use Better Transformer. The version {optimum_version} was found."
            )

        from optimum.bettertransformer import BetterTransformer

        return BetterTransformer.transform(self)

    def reverse_bettertransformer(self):
        """
        Reverts the transformation from [`~PreTrainedModel.to_bettertransformer`] so that the original modeling is
        used, for example in order to save the model.

        Returns:
            [`PreTrainedModel`]: The model converted back to the original modeling.
        """
        if not is_optimum_available():
            raise ImportError("The package `optimum` is required to use Better Transformer.")

        from optimum.version import __version__ as optimum_version

        if version.parse(optimum_version) < version.parse("1.7.0"):
            raise ImportError(
                f"Please install optimum>=1.7.0 to use Better Transformer. The version {optimum_version} was found."
            )

        from optimum.bettertransformer import BetterTransformer

        return BetterTransformer.reverse(self)

    def warn_if_padding_and_no_attention_mask(self, input_ids, attention_mask):
        """
        Shows a one-time warning if the input_ids appear to contain padding and no attention mask was given.
        """

        # Skip the check during tracing.
        if is_torch_fx_proxy(input_ids) or torch.jit.is_tracing() or is_torchdynamo_compiling():
            return

        if (attention_mask is not None) or (self.config.pad_token_id is None):
            return

        # Check only the first and last input IDs to reduce overhead.
        if self.config.pad_token_id in input_ids[:, [-1, 0]]:
            warn_string = (
                "We strongly recommend passing in an `attention_mask` since your input_ids may be padded. See "
                "https://huggingface.co/docs/transformers/troubleshooting"
                "#incorrect-output-when-padding-tokens-arent-masked."
            )

            # If the pad token is equal to either BOS, EOS, or SEP, we do not know whether the user should use an
            # attention_mask or not. In this case, we should still show a warning because this is a rare case.
            if (
                (self.config.bos_token_id is not None and self.config.bos_token_id == self.config.pad_token_id)
                or (self.config.eos_token_id is not None and self.config.eos_token_id == self.config.pad_token_id)
                or (self.config.sep_token_id is not None and self.config.sep_token_id == self.config.pad_token_id)
            ):
                warn_string += (
                    f"\nYou may ignore this warning if your `pad_token_id` ({self.config.pad_token_id}) is identical "
                    f"to the `bos_token_id` ({self.config.bos_token_id}), `eos_token_id` ({self.config.eos_token_id}), "
                    f"or the `sep_token_id` ({self.config.sep_token_id}), and your input is not padded."
                )

            logger.warning_once(warn_string)

    @property
    def supports_tp_plan(self):
        """
        Returns whether the model has a tensor parallelism plan.
        """
        if self._tp_plan is not None:
            return True
        # Check if base model has a TP plan
        if getattr(self.base_model, "_tp_plan", None) is not None:
            return True
        return False

    def tensor_parallel(self, device_mesh):
        """
        Tensor parallelize the model across the given device mesh.

        Args:
            device_mesh (`torch.distributed.DeviceMesh`):
                The device mesh to use for tensor parallelism.
        """
        if not is_torch_greater_or_equal("2.5"):
            raise EnvironmentError("tensor parallel is only supported for `torch>=2.5`.")

        # Tensor parallelize a nn.Module based on the `_tp_plan` attribute of the module.
        # No op if `_tp_plan` attribute does not exist under the module.
        # This is a helper function to be used with `model.apply` to recursively
        # parallelize a model.
        def tplize(mod: torch.nn.Module) -> None:
            tp_plan = getattr(mod, "_tp_plan", None)
            if tp_plan is None:
                return
            logger.debug(f"Applying tensor parallel to {mod.__class__.__name__}: {tp_plan}")
            # In model configs, we use a neutral type (string) to specify
            # parallel styles, here we translate them into torch TP types.
            # Using tree_map because `tp_plan` is a dict.
            tp_plan = torch.utils._pytree.tree_map(
                translate_to_torch_parallel_style,
                tp_plan,
            )
            # Apply TP to current module.
            torch.distributed.tensor.parallel.parallelize_module(
                mod,
                device_mesh=device_mesh,
                parallelize_plan=tp_plan,
            )

        # `apply` is a native method of `nn.Module` that recursively applies a
        # function to every submodule.
        self.apply(tplize)

    @property
    def loss_function(self):
        loss_type = getattr(self, "loss_type", None)

        if loss_type is None or loss_type not in LOSS_MAPPING:
            logger.warning_once(
                f"`loss_type={loss_type}` was set in the config but it is unrecognised."
                f"Using the default loss: `ForCausalLMLoss`."
            )
            loss_type = "ForCausalLM"
        return LOSS_MAPPING[loss_type]

    def get_compiled_call(self, compile_config: CompileConfig):
        """Return a `torch.compile`'d version of `self.__call__`. This is useful to dynamically choose between
        non-compiled/compiled `forward` during inference, especially to switch between prefill (where we don't
        want to use compiled version to avoid recomputing the graph with new shapes) and iterative decoding
        (where we want the speed-ups of compiled version with static shapes)."""
        # Only reset it if not present or different from previous config
        default_config = getattr(self.generation_config, "compile_config", CompileConfig())
        if (
            not hasattr(self, "_compiled_call")
            or getattr(self, "_last_compile_config", default_config) != compile_config
        ):
            self._last_compile_config = compile_config
            self._compiled_call = torch.compile(self.__call__, **compile_config.to_dict())
        return self._compiled_call


PreTrainedModel.push_to_hub = copy_func(PreTrainedModel.push_to_hub)
if PreTrainedModel.push_to_hub.__doc__ is not None:
    PreTrainedModel.push_to_hub.__doc__ = PreTrainedModel.push_to_hub.__doc__.format(
        object="model", object_class="AutoModel", object_files="model file"
    )


class PoolerStartLogits(nn.Module):
    """
    Compute SQuAD start logits from sequence hidden states.

    Args:
        config ([`PretrainedConfig`]):
            The config used by the model, will be used to grab the `hidden_size` of the model.
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, 1)

    def forward(
        self, hidden_states: torch.FloatTensor, p_mask: Optional[torch.FloatTensor] = None
    ) -> torch.FloatTensor:
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, seq_len, hidden_size)`):
                The final hidden states of the model.
            p_mask (`torch.FloatTensor` of shape `(batch_size, seq_len)`, *optional*):
                Mask for tokens at invalid position, such as query and special symbols (PAD, SEP, CLS). 1.0 means token
                should be masked.

        Returns:
            `torch.FloatTensor`: The start logits for SQuAD.
        """
        x = self.dense(hidden_states).squeeze(-1)

        if p_mask is not None:
            if get_parameter_dtype(self) == torch.float16:
                x = x * (1 - p_mask) - 65500 * p_mask
            else:
                x = x * (1 - p_mask) - 1e30 * p_mask

        return x


class PoolerEndLogits(nn.Module):
    """
    Compute SQuAD end logits from sequence hidden states.

    Args:
        config ([`PretrainedConfig`]):
            The config used by the model, will be used to grab the `hidden_size` of the model and the `layer_norm_eps`
            to use.
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.dense_0 = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.activation = nn.Tanh()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dense_1 = nn.Linear(config.hidden_size, 1)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        start_states: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        p_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, seq_len, hidden_size)`):
                The final hidden states of the model.
            start_states (`torch.FloatTensor` of shape `(batch_size, seq_len, hidden_size)`, *optional*):
                The hidden states of the first tokens for the labeled span.
            start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                The position of the first token for the labeled span.
            p_mask (`torch.FloatTensor` of shape `(batch_size, seq_len)`, *optional*):
                Mask for tokens at invalid position, such as query and special symbols (PAD, SEP, CLS). 1.0 means token
                should be masked.

        <Tip>

        One of `start_states` or `start_positions` should be not `None`. If both are set, `start_positions` overrides
        `start_states`.

        </Tip>

        Returns:
            `torch.FloatTensor`: The end logits for SQuAD.
        """
        assert (
            start_states is not None or start_positions is not None
        ), "One of start_states, start_positions should be not None"
        if start_positions is not None:
            slen, hsz = hidden_states.shape[-2:]
            start_positions = start_positions[:, None, None].expand(-1, -1, hsz)  # shape (bsz, 1, hsz)
            start_states = hidden_states.gather(-2, start_positions)  # shape (bsz, 1, hsz)
            start_states = start_states.expand(-1, slen, -1)  # shape (bsz, slen, hsz)

        x = self.dense_0(torch.cat([hidden_states, start_states], dim=-1))
        x = self.activation(x)
        x = self.LayerNorm(x)
        x = self.dense_1(x).squeeze(-1)

        if p_mask is not None:
            if get_parameter_dtype(self) == torch.float16:
                x = x * (1 - p_mask) - 65500 * p_mask
            else:
                x = x * (1 - p_mask) - 1e30 * p_mask

        return x


class PoolerAnswerClass(nn.Module):
    """
    Compute SQuAD 2.0 answer class from classification and start tokens hidden states.

    Args:
        config ([`PretrainedConfig`]):
            The config used by the model, will be used to grab the `hidden_size` of the model.
    """

    def __init__(self, config):
        super().__init__()
        self.dense_0 = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.activation = nn.Tanh()
        self.dense_1 = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        start_states: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        cls_index: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, seq_len, hidden_size)`):
                The final hidden states of the model.
            start_states (`torch.FloatTensor` of shape `(batch_size, seq_len, hidden_size)`, *optional*):
                The hidden states of the first tokens for the labeled span.
            start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                The position of the first token for the labeled span.
            cls_index (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Position of the CLS token for each sentence in the batch. If `None`, takes the last token.

        <Tip>

        One of `start_states` or `start_positions` should be not `None`. If both are set, `start_positions` overrides
        `start_states`.

        </Tip>

        Returns:
            `torch.FloatTensor`: The SQuAD 2.0 answer class.
        """
        # No dependency on end_feature so that we can obtain one single `cls_logits` for each sample.
        hsz = hidden_states.shape[-1]
        assert (
            start_states is not None or start_positions is not None
        ), "One of start_states, start_positions should be not None"
        if start_positions is not None:
            start_positions = start_positions[:, None, None].expand(-1, -1, hsz)  # shape (bsz, 1, hsz)
            start_states = hidden_states.gather(-2, start_positions).squeeze(-2)  # shape (bsz, hsz)

        if cls_index is not None:
            cls_index = cls_index[:, None, None].expand(-1, -1, hsz)  # shape (bsz, 1, hsz)
            cls_token_state = hidden_states.gather(-2, cls_index).squeeze(-2)  # shape (bsz, hsz)
        else:
            cls_token_state = hidden_states[:, -1, :]  # shape (bsz, hsz)

        x = self.dense_0(torch.cat([start_states, cls_token_state], dim=-1))
        x = self.activation(x)
        x = self.dense_1(x).squeeze(-1)

        return x


@dataclass
class SquadHeadOutput(ModelOutput):
    """
    Base class for outputs of question answering models using a [`~modeling_utils.SQuADHead`].

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned if both `start_positions` and `end_positions` are provided):
            Classification loss as the sum of start token, end token (and is_impossible if provided) classification
            losses.
        start_top_log_probs (`torch.FloatTensor` of shape `(batch_size, config.start_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided):
            Log probabilities for the top config.start_n_top start token possibilities (beam-search).
        start_top_index (`torch.LongTensor` of shape `(batch_size, config.start_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided):
            Indices for the top config.start_n_top start token possibilities (beam-search).
        end_top_log_probs (`torch.FloatTensor` of shape `(batch_size, config.start_n_top * config.end_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided):
            Log probabilities for the top `config.start_n_top * config.end_n_top` end token possibilities
            (beam-search).
        end_top_index (`torch.LongTensor` of shape `(batch_size, config.start_n_top * config.end_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided):
            Indices for the top `config.start_n_top * config.end_n_top` end token possibilities (beam-search).
        cls_logits (`torch.FloatTensor` of shape `(batch_size,)`, *optional*, returned if `start_positions` or `end_positions` is not provided):
            Log probabilities for the `is_impossible` label of the answers.

    """

    loss: Optional[torch.FloatTensor] = None
    start_top_log_probs: Optional[torch.FloatTensor] = None
    start_top_index: Optional[torch.LongTensor] = None
    end_top_log_probs: Optional[torch.FloatTensor] = None
    end_top_index: Optional[torch.LongTensor] = None
    cls_logits: Optional[torch.FloatTensor] = None


class SQuADHead(nn.Module):
    r"""
    A SQuAD head inspired by XLNet.

    Args:
        config ([`PretrainedConfig`]):
            The config used by the model, will be used to grab the `hidden_size` of the model and the `layer_norm_eps`
            to use.
    """

    def __init__(self, config):
        super().__init__()
        self.start_n_top = config.start_n_top
        self.end_n_top = config.end_n_top

        self.start_logits = PoolerStartLogits(config)
        self.end_logits = PoolerEndLogits(config)
        self.answer_class = PoolerAnswerClass(config)

    @replace_return_docstrings(output_type=SquadHeadOutput, config_class=PretrainedConfig)
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        cls_index: Optional[torch.LongTensor] = None,
        is_impossible: Optional[torch.LongTensor] = None,
        p_mask: Optional[torch.FloatTensor] = None,
        return_dict: bool = False,
    ) -> Union[SquadHeadOutput, Tuple[torch.FloatTensor]]:
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, seq_len, hidden_size)`):
                Final hidden states of the model on the sequence tokens.
            start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Positions of the first token for the labeled span.
            end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Positions of the last token for the labeled span.
            cls_index (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Position of the CLS token for each sentence in the batch. If `None`, takes the last token.
            is_impossible (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Whether the question has a possible answer in the paragraph or not.
            p_mask (`torch.FloatTensor` of shape `(batch_size, seq_len)`, *optional*):
                Mask for tokens at invalid position, such as query and special symbols (PAD, SEP, CLS). 1.0 means token
                should be masked.
            return_dict (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        Returns:
        """
        start_logits = self.start_logits(hidden_states, p_mask=p_mask)

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, let's remove the dimension added by batch splitting
            for x in (start_positions, end_positions, cls_index, is_impossible):
                if x is not None and x.dim() > 1:
                    x.squeeze_(-1)

            # during training, compute the end logits based on the ground truth of the start position
            end_logits = self.end_logits(hidden_states, start_positions=start_positions, p_mask=p_mask)

            loss_fct = CrossEntropyLoss()
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

            if cls_index is not None and is_impossible is not None:
                # Predict answerability from the representation of CLS and START
                cls_logits = self.answer_class(hidden_states, start_positions=start_positions, cls_index=cls_index)
                loss_fct_cls = nn.BCEWithLogitsLoss()
                cls_loss = loss_fct_cls(cls_logits, is_impossible)

                # note(zhiliny): by default multiply the loss by 0.5 so that the scale is comparable to start_loss and end_loss
                total_loss += cls_loss * 0.5

            return SquadHeadOutput(loss=total_loss) if return_dict else (total_loss,)

        else:
            # during inference, compute the end logits based on beam search
            bsz, slen, hsz = hidden_states.size()
            start_log_probs = nn.functional.softmax(start_logits, dim=-1)  # shape (bsz, slen)

            start_top_log_probs, start_top_index = torch.topk(
                start_log_probs, self.start_n_top, dim=-1
            )  # shape (bsz, start_n_top)
            start_top_index_exp = start_top_index.unsqueeze(-1).expand(-1, -1, hsz)  # shape (bsz, start_n_top, hsz)
            start_states = torch.gather(hidden_states, -2, start_top_index_exp)  # shape (bsz, start_n_top, hsz)
            start_states = start_states.unsqueeze(1).expand(-1, slen, -1, -1)  # shape (bsz, slen, start_n_top, hsz)

            hidden_states_expanded = hidden_states.unsqueeze(2).expand_as(
                start_states
            )  # shape (bsz, slen, start_n_top, hsz)
            p_mask = p_mask.unsqueeze(-1) if p_mask is not None else None
            end_logits = self.end_logits(hidden_states_expanded, start_states=start_states, p_mask=p_mask)
            end_log_probs = nn.functional.softmax(end_logits, dim=1)  # shape (bsz, slen, start_n_top)

            end_top_log_probs, end_top_index = torch.topk(
                end_log_probs, self.end_n_top, dim=1
            )  # shape (bsz, end_n_top, start_n_top)
            end_top_log_probs = end_top_log_probs.view(-1, self.start_n_top * self.end_n_top)
            end_top_index = end_top_index.view(-1, self.start_n_top * self.end_n_top)

            start_states = torch.einsum("blh,bl->bh", hidden_states, start_log_probs)
            cls_logits = self.answer_class(hidden_states, start_states=start_states, cls_index=cls_index)

            if not return_dict:
                return (start_top_log_probs, start_top_index, end_top_log_probs, end_top_index, cls_logits)
            else:
                return SquadHeadOutput(
                    start_top_log_probs=start_top_log_probs,
                    start_top_index=start_top_index,
                    end_top_log_probs=end_top_log_probs,
                    end_top_index=end_top_index,
                    cls_logits=cls_logits,
                )


class SequenceSummary(nn.Module):
    r"""
    Compute a single vector summary of a sequence hidden states.

    Args:
        config ([`PretrainedConfig`]):
            The config used by the model. Relevant arguments in the config class of the model are (refer to the actual
            config class of your model for the default values it uses):

            - **summary_type** (`str`) -- The method to use to make this summary. Accepted values are:

                - `"last"` -- Take the last token hidden state (like XLNet)
                - `"first"` -- Take the first token hidden state (like Bert)
                - `"mean"` -- Take the mean of all tokens hidden states
                - `"cls_index"` -- Supply a Tensor of classification token position (GPT/GPT-2)
                - `"attn"` -- Not implemented now, use multi-head attention

            - **summary_use_proj** (`bool`) -- Add a projection after the vector extraction.
            - **summary_proj_to_labels** (`bool`) -- If `True`, the projection outputs to `config.num_labels` classes
              (otherwise to `config.hidden_size`).
            - **summary_activation** (`Optional[str]`) -- Set to `"tanh"` to add a tanh activation to the output,
              another string or `None` will add no activation.
            - **summary_first_dropout** (`float`) -- Optional dropout probability before the projection and activation.
            - **summary_last_dropout** (`float`)-- Optional dropout probability after the projection and activation.
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__()

        self.summary_type = getattr(config, "summary_type", "last")
        if self.summary_type == "attn":
            # We should use a standard multi-head attention module with absolute positional embedding for that.
            # Cf. https://github.com/zihangdai/xlnet/blob/master/modeling.py#L253-L276
            # We can probably just use the multi-head attention module of PyTorch >=1.1.0
            raise NotImplementedError

        self.summary = Identity()
        if hasattr(config, "summary_use_proj") and config.summary_use_proj:
            if hasattr(config, "summary_proj_to_labels") and config.summary_proj_to_labels and config.num_labels > 0:
                num_classes = config.num_labels
            else:
                num_classes = config.hidden_size
            self.summary = nn.Linear(config.hidden_size, num_classes)

        activation_string = getattr(config, "summary_activation", None)
        self.activation: Callable = get_activation(activation_string) if activation_string else Identity()

        self.first_dropout = Identity()
        if hasattr(config, "summary_first_dropout") and config.summary_first_dropout > 0:
            self.first_dropout = nn.Dropout(config.summary_first_dropout)

        self.last_dropout = Identity()
        if hasattr(config, "summary_last_dropout") and config.summary_last_dropout > 0:
            self.last_dropout = nn.Dropout(config.summary_last_dropout)

    def forward(
        self, hidden_states: torch.FloatTensor, cls_index: Optional[torch.LongTensor] = None
    ) -> torch.FloatTensor:
        """
        Compute a single vector summary of a sequence hidden states.

        Args:
            hidden_states (`torch.FloatTensor` of shape `[batch_size, seq_len, hidden_size]`):
                The hidden states of the last layer.
            cls_index (`torch.LongTensor` of shape `[batch_size]` or `[batch_size, ...]` where ... are optional leading dimensions of `hidden_states`, *optional*):
                Used if `summary_type == "cls_index"` and takes the last token of the sequence as classification token.

        Returns:
            `torch.FloatTensor`: The summary of the sequence hidden states.
        """
        if self.summary_type == "last":
            output = hidden_states[:, -1]
        elif self.summary_type == "first":
            output = hidden_states[:, 0]
        elif self.summary_type == "mean":
            output = hidden_states.mean(dim=1)
        elif self.summary_type == "cls_index":
            if cls_index is None:
                cls_index = torch.full_like(
                    hidden_states[..., :1, :],
                    hidden_states.shape[-2] - 1,
                    dtype=torch.long,
                )
            else:
                cls_index = cls_index.unsqueeze(-1).unsqueeze(-1)
                cls_index = cls_index.expand((-1,) * (cls_index.dim() - 1) + (hidden_states.size(-1),))
            # shape of cls_index: (bsz, XX, 1, hidden_size) where XX are optional leading dim of hidden_states
            output = hidden_states.gather(-2, cls_index).squeeze(-2)  # shape (bsz, XX, hidden_size)
        elif self.summary_type == "attn":
            raise NotImplementedError

        output = self.first_dropout(output)
        output = self.summary(output)
        output = self.activation(output)
        output = self.last_dropout(output)

        return output


def unwrap_model(model: nn.Module, recursive: bool = False) -> nn.Module:
    """
    Recursively unwraps a model from potential containers (as used in distributed training).

    Args:
        model (`torch.nn.Module`): The model to unwrap.
        recursive (`bool`, *optional*, defaults to `False`):
            Whether to recursively extract all cases of `module.module` from `model` as well as unwrap child sublayers
            recursively, not just the top-level distributed containers.
    """
    # Use accelerate implementation if available (should always be the case when using torch)
    # This is for pytorch, as we also have to handle things like dynamo
    if is_accelerate_available():
        kwargs = {}
        if recursive:
            if not is_accelerate_available("0.29.0"):
                raise RuntimeError(
                    "Setting `recursive=True` to `unwrap_model` requires `accelerate` v0.29.0. Please upgrade your version of accelerate"
                )
            else:
                kwargs["recursive"] = recursive
        return extract_model_from_parallel(model, **kwargs)
    else:
        # since there could be multiple levels of wrapping, unwrap recursively
        if hasattr(model, "module"):
            return unwrap_model(model.module)
        else:
            return model


def expand_device_map(device_map, param_names, start_prefix):
    """
    Expand a device map to return the correspondance parameter name to device.
    """
    new_device_map = {}
    param_names = [p[len(start_prefix) :] for p in param_names if p.startswith(start_prefix)]
    for module, device in device_map.items():
        new_device_map.update(
            {p: device for p in param_names if p == module or p.startswith(f"{module}.") or module == ""}
        )
    return new_device_map


def get_disk_only_shard_files(device_map, sharded_metadata, start_prefix):
    """
    Returns the list of shard files containing only weights offloaded to disk.
    """

    weight_map = {
        p[len(start_prefix) :]: v for p, v in sharded_metadata["weight_map"].items() if p.startswith(start_prefix)
    }
    files_content = collections.defaultdict(list)
    for weight_name, filename in weight_map.items():
        while len(weight_name) > 0 and weight_name not in device_map:
            weight_name = ".".join(weight_name.split(".")[:-1])
        files_content[filename].append(device_map[weight_name])

    return [fname for fname, devices in files_content.items() if set(devices) == {"disk"}]


ALL_ATTENTION_FUNCTIONS: Dict[str, Dict[str, Callable]] = {}

ALL_ATTENTION_FUNCTIONS.update(
    {
        "flash_attention_2": flash_attention_forward,
        "flex_attention": flex_attention_forward,
        "sdpa": sdpa_attention_forward,
    }
)
