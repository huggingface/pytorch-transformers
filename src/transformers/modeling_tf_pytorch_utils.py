# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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
"""PyTorch - TF 2.0 general utilities."""

import os
import re

import numpy

from .utils import (
    ExplicitEnum,
    expand_dims,
    is_numpy_array,
    is_safetensors_available,
    is_torch_tensor,
    logging,
    reshape,
    squeeze,
    tensor_size,
)
from .utils import transpose as transpose_func


if is_safetensors_available():
    from safetensors import safe_open


logger = logging.get_logger(__name__)


class TransposeType(ExplicitEnum):
    """
    Possible ...
    """

    NO = "no"
    SIMPLE = "simple"
    CONV1D = "conv1d"
    CONV2D = "conv2d"


def convert_tf_weight_name_to_pt_weight_name(
    tf_name, start_prefix_to_remove="", tf_weight_shape=None, name_scope=None
):
    """
    Convert a TF 2.0 model variable name in a pytorch model weight name.

    Conventions for TF2.0 scopes -> PyTorch attribute names conversions:

        - '$1___$2' is replaced by $2 (can be used to duplicate or remove layers in TF2.0 vs PyTorch)
        - '_._' is replaced by a new level separation (can be used to convert TF2.0 lists in PyTorch nn.ModulesList)

    return tuple with:

        - pytorch model weight name
        - transpose: `TransposeType` member indicating whether and how TF2.0 and PyTorch weights matrices should be
          transposed with regards to each other
    """
    if name_scope is not None:
        if not tf_name.startswith(name_scope) and "final_logits_bias" not in tf_name:
            raise ValueError(
                f"Weight name {tf_name} does not start with name_scope {name_scope}. This is an internal error "
                "in Transformers, so (unless you were doing something really evil) please open an issue to report it!"
            )
        tf_name = tf_name[len(name_scope) :]
        tf_name = tf_name.lstrip("/")
    tf_name = tf_name.replace(":0", "")  # device ids
    tf_name = re.sub(
        r"/[^/]*___([^/]*)/", r"/\1/", tf_name
    )  # '$1___$2' is replaced by $2 (can be used to duplicate or remove layers in TF2.0 vs PyTorch)
    tf_name = tf_name.replace(
        "_._", "/"
    )  # '_._' is replaced by a level separation (can be used to convert TF2.0 lists in PyTorch nn.ModulesList)
    tf_name = re.sub(r"//+", "/", tf_name)  # Remove empty levels at the end
    tf_name = tf_name.split("/")  # Convert from TF2.0 '/' separators to PyTorch '.' separators
    # Some weights have a single name without "/" such as final_logits_bias in BART
    if len(tf_name) > 1:
        tf_name = tf_name[1:]  # Remove level zero

    tf_weight_shape = list(tf_weight_shape)

    # When should we transpose the weights
    if tf_name[-1] == "kernel" and tf_weight_shape is not None and len(tf_weight_shape) == 4:
        transpose = TransposeType.CONV2D
    elif tf_name[-1] == "kernel" and tf_weight_shape is not None and len(tf_weight_shape) == 3:
        transpose = TransposeType.CONV1D
    elif bool(
        tf_name[-1] in ["kernel", "pointwise_kernel", "depthwise_kernel"]
        or "emb_projs" in tf_name
        or "out_projs" in tf_name
    ):
        transpose = TransposeType.SIMPLE
    else:
        transpose = TransposeType.NO

    # Convert standard TF2.0 names in PyTorch names
    if tf_name[-1] == "kernel" or tf_name[-1] == "embeddings" or tf_name[-1] == "gamma":
        tf_name[-1] = "weight"
    if tf_name[-1] == "beta":
        tf_name[-1] = "bias"

    # The SeparableConv1D TF layer contains two weights that are translated to PyTorch Conv1D here
    if tf_name[-1] == "pointwise_kernel" or tf_name[-1] == "depthwise_kernel":
        tf_name[-1] = tf_name[-1].replace("_kernel", ".weight")

    # Remove prefix if needed
    tf_name = ".".join(tf_name)
    if start_prefix_to_remove:
        tf_name = tf_name.replace(start_prefix_to_remove, "", 1)

    return tf_name, transpose


def apply_transpose(transpose: TransposeType, weight, match_shape=None, pt_to_tf=True):
    """
    Apply a transpose to some weight then tries to reshape the weight to the same shape as a given shape, all in a
    framework agnostic way.
    """
    if transpose is TransposeType.CONV2D:
        # Conv2D weight:
        #    PT: (num_out_channel, num_in_channel, kernel[0], kernel[1])
        # -> TF: (kernel[0], kernel[1], num_in_channel, num_out_channel)
        axes = (2, 3, 1, 0) if pt_to_tf else (3, 2, 0, 1)
        weight = transpose_func(weight, axes=axes)
    elif transpose is TransposeType.CONV1D:
        # Conv1D weight:
        #    PT: (num_out_channel, num_in_channel, kernel)
        # -> TF: (kernel, num_in_channel, num_out_channel)
        weight = transpose_func(weight, axes=(2, 1, 0))
    elif transpose is TransposeType.SIMPLE:
        weight = transpose_func(weight)

    if match_shape is None:
        return weight

    if len(match_shape) < len(weight.shape):
        weight = squeeze(weight)
    elif len(match_shape) > len(weight.shape):
        weight = expand_dims(weight, axis=0)

    if list(match_shape) != list(weight.shape):
        try:
            weight = reshape(weight, match_shape)
        except AssertionError as e:
            e.args += (match_shape, match_shape)
            raise e

    return weight


#####################
# PyTorch => TF 2.0 #
#####################


def load_pytorch_checkpoint_in_tf2_model(
    tf_model,
    pytorch_checkpoint_path,
    tf_inputs=None,
    allow_missing_keys=False,
    output_loading_info=False,
    _prefix=None,
    tf_to_pt_weight_rename=None,
):
    """Load pytorch checkpoints in a TF 2.0 model"""
    try:
        import tensorflow as tf  # noqa: F401
        import torch  # noqa: F401
        from safetensors.torch import load_file as safe_load_file  # noqa: F401
    except ImportError:
        logger.error(
            "Loading a PyTorch model in TensorFlow, requires both PyTorch and TensorFlow to be installed. Please see "
            "https://pytorch.org/ and https://www.tensorflow.org/install/ for installation instructions."
        )
        raise

    # Treats a single file as a collection of shards with 1 shard.
    if isinstance(pytorch_checkpoint_path, str):
        pytorch_checkpoint_path = [pytorch_checkpoint_path]

    # Loads all shards into a single state dictionary
    pt_state_dict = {}
    for path in pytorch_checkpoint_path:
        pt_path = os.path.abspath(path)
        logger.info(f"Loading PyTorch weights from {pt_path}")
        if pt_path.endswith(".safetensors"):
            state_dict = safe_load_file(pt_path)
        else:
            weights_only_kwarg = {"weights_only": True}
            state_dict = torch.load(pt_path, map_location="cpu", **weights_only_kwarg)

        pt_state_dict.update(state_dict)

    logger.info(f"PyTorch checkpoint contains {sum(t.numel() for t in pt_state_dict.values()):,} parameters")

    return load_pytorch_weights_in_tf2_model(
        tf_model,
        pt_state_dict,
        tf_inputs=tf_inputs,
        allow_missing_keys=allow_missing_keys,
        output_loading_info=output_loading_info,
        _prefix=_prefix,
        tf_to_pt_weight_rename=tf_to_pt_weight_rename,
    )


def load_pytorch_model_in_tf2_model(tf_model, pt_model, tf_inputs=None, allow_missing_keys=False):
    """Load pytorch checkpoints in a TF 2.0 model"""
    pt_state_dict = pt_model.state_dict()

    return load_pytorch_weights_in_tf2_model(
        tf_model, pt_state_dict, tf_inputs=tf_inputs, allow_missing_keys=allow_missing_keys
    )


def load_pytorch_weights_in_tf2_model(
    tf_model,
    pt_state_dict,
    tf_inputs=None,
    allow_missing_keys=False,
    output_loading_info=False,
    _prefix=None,
    tf_to_pt_weight_rename=None,
):
    """Load pytorch state_dict in a TF 2.0 model."""
    try:
        import tensorflow as tf  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        logger.error(
            "Loading a PyTorch model in TensorFlow, requires both PyTorch and TensorFlow to be installed. Please see "
            "https://pytorch.org/ and https://www.tensorflow.org/install/ for installation instructions."
        )
        raise

    # Numpy doesn't understand bfloat16, so upcast to a dtype that doesn't lose precision
    pt_state_dict = {
        k: v.numpy() if v.dtype != torch.bfloat16 else v.float().numpy() for k, v in pt_state_dict.items()
    }
    return load_pytorch_state_dict_in_tf2_model(
        tf_model,
        pt_state_dict,
        tf_inputs=tf_inputs,
        allow_missing_keys=allow_missing_keys,
        output_loading_info=output_loading_info,
        _prefix=_prefix,
        tf_to_pt_weight_rename=tf_to_pt_weight_rename,
    )


def _log_key_warnings(missing_keys, unexpected_keys, mismatched_keys, class_name):
    if len(unexpected_keys) > 0:
        logger.warning(
            "Some weights of the PyTorch model were not used when initializing the TF 2.0 model"
            f" {class_name}: {unexpected_keys}\n- This IS expected if you are initializing"
            f" {class_name} from a PyTorch model trained on another task or with another architecture"
            " (e.g. initializing a TFBertForSequenceClassification model from a BertForPreTraining model).\n- This IS"
            f" NOT expected if you are initializing {class_name} from a PyTorch model that you expect"
            " to be exactly identical (e.g. initializing a TFBertForSequenceClassification model from a"
            " BertForSequenceClassification model)."
        )
    else:
        logger.warning(f"All PyTorch model weights were used when initializing {class_name}.\n")
    if len(missing_keys) > 0:
        logger.warning(
            f"Some weights or buffers of the TF 2.0 model {class_name} were not initialized from the"
            f" PyTorch model and are newly initialized: {missing_keys}\nYou should probably TRAIN this model on a"
            " down-stream task to be able to use it for predictions and inference."
        )
    else:
        logger.warning(
            f"All the weights of {class_name} were initialized from the PyTorch model.\n"
            "If your task is similar to the task the model of the checkpoint was trained on, "
            f"you can already use {class_name} for predictions without further training."
        )

    if len(mismatched_keys) > 0:
        mismatched_warning = "\n".join(
            [
                f"- {key}: found shape {shape1} in the checkpoint and {shape2} in the model instantiated"
                for key, shape1, shape2 in mismatched_keys
            ]
        )
        logger.warning(
            f"Some weights of {class_name} were not initialized from the model checkpoint"
            f" are newly initialized because the shapes did not"
            f" match:\n{mismatched_warning}\nYou should probably TRAIN this model on a down-stream task to be able"
            " to use it for predictions and inference."
        )


def load_pytorch_state_dict_in_tf2_model(
    tf_model,
    pt_state_dict,
    tf_inputs=None,
    allow_missing_keys=False,
    output_loading_info=False,
    _prefix=None,
    tf_to_pt_weight_rename=None,
    ignore_mismatched_sizes=False,
    skip_logger_warnings=False,
):
    """Load a pytorch state_dict in a TF 2.0 model. pt_state_dict can be either an actual dict or a lazy-loading
    safetensors archive created with the safe_open() function."""
    import tensorflow as tf

    if tf_inputs is None:
        tf_inputs = tf_model.dummy_inputs

    if _prefix is None:
        _prefix = ""
    if tf_inputs:
        with tf.name_scope(_prefix):
            tf_model(tf_inputs, training=False)  # Make sure model is built
    # Convert old format to new format if needed from a PyTorch state_dict
    tf_keys_to_pt_keys = {}
    for key in pt_state_dict.keys():
        new_key = None
        if "gamma" in key:
            new_key = key.replace("gamma", "weight")
        if "beta" in key:
            new_key = key.replace("beta", "bias")
        if "running_var" in key:
            new_key = key.replace("running_var", "moving_variance")
        if "running_mean" in key:
            new_key = key.replace("running_mean", "moving_mean")

        # New `weight_norm` from https://github.com/huggingface/transformers/pull/24030
        key_components = key.split(".")
        name = None
        if key_components[-3::2] == ["parametrizations", "original0"]:
            name = key_components[-2] + "_g"
        elif key_components[-3::2] == ["parametrizations", "original1"]:
            name = key_components[-2] + "_v"
        if name is not None:
            key_components = key_components[:-3] + [name]
            new_key = ".".join(key_components)

        if new_key is None:
            new_key = key
        tf_keys_to_pt_keys[new_key] = key

    # Matt: All TF models store the actual model stem in a MainLayer class, including the base model.
    # In PT, the derived models (with heads) use the base model class as the stem instead,
    # and there is no MainLayer class. This means that TF base classes have one
    # extra layer in their weight names, corresponding to the MainLayer class. This code block compensates for that.
    start_prefix_to_remove = ""
    if not any(s.startswith(tf_model.base_model_prefix) for s in tf_keys_to_pt_keys.keys()):
        start_prefix_to_remove = tf_model.base_model_prefix + "."

    symbolic_weights = tf_model.trainable_weights + tf_model.non_trainable_weights
    tf_loaded_numel = 0
    all_pytorch_weights = set(tf_keys_to_pt_keys.keys())
    missing_keys = []
    mismatched_keys = []
    is_safetensor_archive = hasattr(pt_state_dict, "get_tensor")
    for symbolic_weight in symbolic_weights:
        sw_name = symbolic_weight.name
        name, transpose = convert_tf_weight_name_to_pt_weight_name(
            sw_name,
            start_prefix_to_remove=start_prefix_to_remove,
            tf_weight_shape=symbolic_weight.shape,
            name_scope=_prefix,
        )
        if tf_to_pt_weight_rename is not None:
            aliases = tf_to_pt_weight_rename(name)  # Is a tuple to account for possible name aliasing
            for alias in aliases:  # The aliases are in priority order, take the first one that matches
                if alias in tf_keys_to_pt_keys:
                    name = alias
                    break
            else:
                # If none of the aliases match, just use the first one (it'll be reported as missing)
                name = aliases[0]

        # Find associated numpy array in pytorch model state dict
        if name not in tf_keys_to_pt_keys:
            if allow_missing_keys:
                missing_keys.append(name)
                continue
            elif tf_model._keys_to_ignore_on_load_missing is not None:
                # authorized missing keys don't have to be loaded
                if any(re.search(pat, name) is not None for pat in tf_model._keys_to_ignore_on_load_missing):
                    continue
            raise AttributeError(f"{name} not found in PyTorch model")
        state_dict_name = tf_keys_to_pt_keys[name]
        if is_safetensor_archive:
            array = pt_state_dict.get_tensor(state_dict_name)
        else:
            array = pt_state_dict[state_dict_name]
        try:
            array = apply_transpose(transpose, array, symbolic_weight.shape)
        except tf.errors.InvalidArgumentError as e:
            if not ignore_mismatched_sizes:
                error_msg = str(e)
                error_msg += (
                    "\n\tYou may consider adding `ignore_mismatched_sizes=True` in the model `from_pretrained` method."
                )
                raise tf.errors.InvalidArgumentError(error_msg)
            else:
                mismatched_keys.append((name, array.shape, symbolic_weight.shape))
                continue

        tf_loaded_numel += tensor_size(array)

        symbolic_weight.assign(tf.cast(array, symbolic_weight.dtype))
        del array  # Immediately free memory to keep peak usage as low as possible
        all_pytorch_weights.discard(name)

    logger.info(f"Loaded {tf_loaded_numel:,} parameters in the TF 2.0 model.")

    unexpected_keys = list(all_pytorch_weights)

    if tf_model._keys_to_ignore_on_load_missing is not None:
        for pat in tf_model._keys_to_ignore_on_load_missing:
            missing_keys = [k for k in missing_keys if re.search(pat, k) is None]
    if tf_model._keys_to_ignore_on_load_unexpected is not None:
        for pat in tf_model._keys_to_ignore_on_load_unexpected:
            unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]
    if not skip_logger_warnings:
        _log_key_warnings(missing_keys, unexpected_keys, mismatched_keys, class_name=tf_model.__class__.__name__)

    if output_loading_info:
        loading_info = {
            "missing_keys": missing_keys,
            "unexpected_keys": unexpected_keys,
            "mismatched_keys": mismatched_keys,
        }
        return tf_model, loading_info

    return tf_model


def load_sharded_pytorch_safetensors_in_tf2_model(
    tf_model,
    safetensors_shards,
    tf_inputs=None,
    allow_missing_keys=False,
    output_loading_info=False,
    _prefix=None,
    tf_to_pt_weight_rename=None,
    ignore_mismatched_sizes=False,
):
    all_loading_infos = []
    for shard in safetensors_shards:
        with safe_open(shard, framework="tf") as safetensors_archive:
            tf_model, loading_info = load_pytorch_state_dict_in_tf2_model(
                tf_model,
                safetensors_archive,
                tf_inputs=tf_inputs,
                allow_missing_keys=allow_missing_keys,
                output_loading_info=True,
                _prefix=_prefix,
                tf_to_pt_weight_rename=tf_to_pt_weight_rename,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
                skip_logger_warnings=True,  # We will emit merged warnings at the end
            )
        all_loading_infos.append(loading_info)
    # Now we just need to merge the loading info
    # Keys are missing only if they're missing in *every* shard
    missing_keys = sorted(set.intersection(*[set(info["missing_keys"]) for info in all_loading_infos]))
    # Keys are unexpected/mismatched if they're unexpected/mismatched in *any* shard
    unexpected_keys = sum([info["unexpected_keys"] for info in all_loading_infos], [])
    mismatched_keys = sum([info["mismatched_keys"] for info in all_loading_infos], [])

    _log_key_warnings(missing_keys, unexpected_keys, mismatched_keys, class_name=tf_model.__class__.__name__)

    if output_loading_info:
        loading_info = {
            "missing_keys": missing_keys,
            "unexpected_keys": unexpected_keys,
            "mismatched_keys": mismatched_keys,
        }
        return tf_model, loading_info

    return tf_model


#####################
# TF 2.0 => PyTorch #
#####################


def load_tf2_checkpoint_in_pytorch_model(
    pt_model, tf_checkpoint_path, tf_inputs=None, allow_missing_keys=False, output_loading_info=False
):
    """
    Load TF 2.0 HDF5 checkpoint in a PyTorch model We use HDF5 to easily do transfer learning (see
    https://github.com/tensorflow/tensorflow/blob/ee16fcac960ae660e0e4496658a366e2f745e1f0/tensorflow/python/keras/engine/network.py#L1352-L1357).
    """
    try:
        import tensorflow as tf  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires both PyTorch and TensorFlow to be installed. Please see "
            "https://pytorch.org/ and https://www.tensorflow.org/install/ for installation instructions."
        )
        raise

    import transformers

    from .modeling_tf_utils import load_tf_weights

    logger.info(f"Loading TensorFlow weights from {tf_checkpoint_path}")

    # Instantiate and load the associated TF 2.0 model
    tf_model_class_name = "TF" + pt_model.__class__.__name__  # Add "TF" at the beginning
    tf_model_class = getattr(transformers, tf_model_class_name)
    tf_model = tf_model_class(pt_model.config)

    if tf_inputs is None:
        tf_inputs = tf_model.dummy_inputs

    if tf_inputs is not None:
        tf_model(tf_inputs, training=False)  # Make sure model is built

    load_tf_weights(tf_model, tf_checkpoint_path)

    return load_tf2_model_in_pytorch_model(
        pt_model, tf_model, allow_missing_keys=allow_missing_keys, output_loading_info=output_loading_info
    )


def load_tf2_model_in_pytorch_model(pt_model, tf_model, allow_missing_keys=False, output_loading_info=False):
    """Load TF 2.0 model in a pytorch model"""
    weights = tf_model.weights

    return load_tf2_weights_in_pytorch_model(
        pt_model, weights, allow_missing_keys=allow_missing_keys, output_loading_info=output_loading_info
    )


def load_tf2_weights_in_pytorch_model(pt_model, tf_weights, allow_missing_keys=False, output_loading_info=False):
    """Load TF2.0 symbolic weights in a PyTorch model"""
    try:
        import tensorflow as tf  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires both PyTorch and TensorFlow to be installed. Please see "
            "https://pytorch.org/ and https://www.tensorflow.org/install/ for installation instructions."
        )
        raise

    tf_state_dict = {tf_weight.name: tf_weight.numpy() for tf_weight in tf_weights}
    return load_tf2_state_dict_in_pytorch_model(
        pt_model, tf_state_dict, allow_missing_keys=allow_missing_keys, output_loading_info=output_loading_info
    )


def load_tf2_state_dict_in_pytorch_model(pt_model, tf_state_dict, allow_missing_keys=False, output_loading_info=False):
    import torch

    new_pt_params_dict = {}
    current_pt_params_dict = dict(pt_model.named_parameters())

    # Make sure we are able to load PyTorch base models as well as derived models (with heads)
    # TF models always have a prefix, some of PyTorch models (base ones) don't
    start_prefix_to_remove = ""
    if not any(s.startswith(pt_model.base_model_prefix) for s in current_pt_params_dict.keys()):
        start_prefix_to_remove = pt_model.base_model_prefix + "."

    # Build a map from potential PyTorch weight names to TF 2.0 Variables
    tf_weights_map = {}
    for name, tf_weight in tf_state_dict.items():
        pt_name, transpose = convert_tf_weight_name_to_pt_weight_name(
            name, start_prefix_to_remove=start_prefix_to_remove, tf_weight_shape=tf_weight.shape
        )
        tf_weights_map[pt_name] = (tf_weight, transpose)

    all_tf_weights = set(tf_weights_map.keys())
    loaded_pt_weights_data_ptr = {}
    missing_keys_pt = []
    for pt_weight_name, pt_weight in current_pt_params_dict.items():
        # Handle PyTorch shared weight ()not duplicated in TF 2.0
        if pt_weight.data_ptr() in loaded_pt_weights_data_ptr:
            new_pt_params_dict[pt_weight_name] = loaded_pt_weights_data_ptr[pt_weight.data_ptr()]
            continue

        pt_weight_name_to_check = pt_weight_name
        # New `weight_norm` from https://github.com/huggingface/transformers/pull/24030
        key_components = pt_weight_name.split(".")
        name = None
        if key_components[-3::2] == ["parametrizations", "original0"]:
            name = key_components[-2] + "_g"
        elif key_components[-3::2] == ["parametrizations", "original1"]:
            name = key_components[-2] + "_v"
        if name is not None:
            key_components = key_components[:-3] + [name]
            pt_weight_name_to_check = ".".join(key_components)

        # Find associated numpy array in pytorch model state dict
        if pt_weight_name_to_check not in tf_weights_map:
            if allow_missing_keys:
                missing_keys_pt.append(pt_weight_name)
                continue

            raise AttributeError(f"{pt_weight_name} not found in TF 2.0 model")

        array, transpose = tf_weights_map[pt_weight_name_to_check]

        array = apply_transpose(transpose, array, pt_weight.shape, pt_to_tf=False)

        if numpy.isscalar(array):
            array = numpy.array(array)
        if not is_torch_tensor(array) and not is_numpy_array(array):
            array = array.numpy()
        if is_numpy_array(array):
            # Convert to torch tensor
            array = torch.from_numpy(array)

        new_pt_params_dict[pt_weight_name] = array
        loaded_pt_weights_data_ptr[pt_weight.data_ptr()] = array
        all_tf_weights.discard(pt_weight_name)

    missing_keys, unexpected_keys = pt_model.load_state_dict(new_pt_params_dict, strict=False)
    missing_keys += missing_keys_pt

    # Some models may have keys that are not in the state by design, removing them before needlessly warning
    # the user.
    if pt_model._keys_to_ignore_on_load_missing is not None:
        for pat in pt_model._keys_to_ignore_on_load_missing:
            missing_keys = [k for k in missing_keys if re.search(pat, k) is None]

    if pt_model._keys_to_ignore_on_load_unexpected is not None:
        for pat in pt_model._keys_to_ignore_on_load_unexpected:
            unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]

    if len(unexpected_keys) > 0:
        logger.warning(
            "Some weights of the TF 2.0 model were not used when initializing the PyTorch model"
            f" {pt_model.__class__.__name__}: {unexpected_keys}\n- This IS expected if you are initializing"
            f" {pt_model.__class__.__name__} from a TF 2.0 model trained on another task or with another architecture"
            " (e.g. initializing a BertForSequenceClassification model from a TFBertForPreTraining model).\n- This IS"
            f" NOT expected if you are initializing {pt_model.__class__.__name__} from a TF 2.0 model that you expect"
            " to be exactly identical (e.g. initializing a BertForSequenceClassification model from a"
            " TFBertForSequenceClassification model)."
        )
    else:
        logger.warning(f"All TF 2.0 model weights were used when initializing {pt_model.__class__.__name__}.\n")
    if len(missing_keys) > 0:
        logger.warning(
            f"Some weights of {pt_model.__class__.__name__} were not initialized from the TF 2.0 model and are newly"
            f" initialized: {missing_keys}\nYou should probably TRAIN this model on a down-stream task to be able to"
            " use it for predictions and inference."
        )
    else:
        logger.warning(
            f"All the weights of {pt_model.__class__.__name__} were initialized from the TF 2.0 model.\n"
            "If your task is similar to the task the model of the checkpoint was trained on, "
            f"you can already use {pt_model.__class__.__name__} for predictions without further training."
        )

    logger.info(f"Weights or buffers not loaded from TF 2.0 model: {all_tf_weights}")

    if output_loading_info:
        loading_info = {"missing_keys": missing_keys, "unexpected_keys": unexpected_keys}
        return pt_model, loading_info

    return pt_model
