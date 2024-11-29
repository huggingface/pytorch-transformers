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
"""TF general model utils."""

from __future__ import annotations

import functools
import gc
import inspect
import json
import os
import pickle
import re
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import h5py
import numpy as np
import tensorflow as tf
from packaging.version import parse

from . import DataCollatorWithPadding, DefaultDataCollator
from .activations_tf import get_tf_activation
from .configuration_utils import PretrainedConfig
from .dynamic_module_utils import custom_object_save
from .generation import GenerationConfig, TFGenerationMixin
from .tf_utils import (
    convert_batch_encoding,
    expand_1d,
    load_attributes_from_hdf5_group,
    save_attributes_to_hdf5_group,
    shape_list,
)
from .utils import (
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    TF2_WEIGHTS_INDEX_NAME,
    TF2_WEIGHTS_NAME,
    TF_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    ModelOutput,
    PushToHubMixin,
    cached_file,
    download_url,
    find_labels,
    has_file,
    is_offline_mode,
    is_remote_url,
    is_safetensors_available,
    is_tf_symbolic_tensor,
    logging,
    requires_backends,
    working_or_temp_dir,
)
from .utils.hub import convert_file_size_to_int, get_checkpoint_shard_files


if is_safetensors_available():
    from safetensors import safe_open
    from safetensors.tensorflow import save_file as safe_save_file

if TYPE_CHECKING:
    from . import PreTrainedTokenizerBase

logger = logging.get_logger(__name__)

if "TF_USE_LEGACY_KERAS" not in os.environ:
    os.environ["TF_USE_LEGACY_KERAS"] = "1"  # Compatibility fix to make sure tf.keras stays at Keras 2
elif os.environ["TF_USE_LEGACY_KERAS"] != "1":
    logger.warning(
        "Transformers is only compatible with Keras 2, but you have explicitly set `TF_USE_LEGACY_KERAS` to `0`. "
        "This may result in unexpected behaviour or errors if Keras 3 objects are passed to Transformers models."
    )

try:
    import tf_keras as keras
    from tf_keras import backend as K
except (ModuleNotFoundError, ImportError):
    import keras
    from keras import backend as K

    if parse(keras.__version__).major > 2:
        raise ValueError(
            "Your currently installed version of Keras is Keras 3, but this is not yet supported in "
            "Transformers. Please install the backwards-compatible tf-keras package with "
            "`pip install tf-keras`."
        )


tf_logger = tf.get_logger()

TFModelInputType = Union[
    List[tf.Tensor],
    List[np.ndarray],
    Dict[str, tf.Tensor],
    Dict[str, np.ndarray],
    tf.Tensor,
    np.ndarray,
]


def dummy_loss(y_true, y_pred):
    if y_pred.shape.rank <= 1:
        return y_pred
    else:
        reduction_axes = list(range(1, y_pred.shape.rank))
        return tf.reduce_mean(y_pred, axis=reduction_axes)


class TFModelUtilsMixin:
    """
    A few utilities for `keras.Model`, to be used as a mixin.
    """

    def num_parameters(self, only_trainable: bool = False) -> int:
        """
        Get the number of (optionally, trainable) parameters in the model.

        Args:
            only_trainable (`bool`, *optional*, defaults to `False`):
                Whether or not to return only the number of trainable parameters

        Returns:
            `int`: The number of parameters.
        """
        if only_trainable:
            return int(sum(np.prod(w.shape.as_list()) for w in self.trainable_variables))
        else:
            return self.count_params()


def keras_serializable(cls):
    """
    Decorate a Keras Layer class to support Keras serialization.

    This is done by:

    1. Adding a `transformers_config` dict to the Keras config dictionary in `get_config` (called by Keras at
       serialization time.
    2. Wrapping `__init__` to accept that `transformers_config` dict (passed by Keras at deserialization time) and
       convert it to a config object for the actual layer initializer.
    3. Registering the class as a custom object in Keras (if the Tensorflow version supports this), so that it does not
       need to be supplied in `custom_objects` in the call to `keras.models.load_model`.

    Args:
        cls (a `keras.layers.Layers subclass`):
            Typically a `TF.MainLayer` class in this project, in general must accept a `config` argument to its
            initializer.

    Returns:
        The same class object, with modifications for Keras deserialization.
    """
    initializer = cls.__init__

    config_class = getattr(cls, "config_class", None)
    if config_class is None:
        raise AttributeError("Must set `config_class` to use @keras_serializable")

    @functools.wraps(initializer)
    def wrapped_init(self, *args, **kwargs):
        config = args[0] if args and isinstance(args[0], PretrainedConfig) else kwargs.pop("config", None)

        if isinstance(config, dict):
            config = config_class.from_dict(config)
            initializer(self, config, *args, **kwargs)
        elif isinstance(config, PretrainedConfig):
            if len(args) > 0:
                initializer(self, *args, **kwargs)
            else:
                initializer(self, config, *args, **kwargs)
        else:
            raise ValueError("Must pass either `config` (PretrainedConfig) or `config` (dict)")

        self._config = config
        self._kwargs = kwargs

    cls.__init__ = wrapped_init

    if not hasattr(cls, "get_config"):
        raise TypeError("Only use @keras_serializable on keras.layers.Layer subclasses")
    if hasattr(cls.get_config, "_is_default"):

        def get_config(self):
            cfg = super(cls, self).get_config()
            cfg["config"] = self._config.to_dict()
            cfg.update(self._kwargs)
            return cfg

        cls.get_config = get_config

    cls._keras_serializable = True
    if hasattr(keras.utils, "register_keras_serializable"):
        cls = keras.utils.register_keras_serializable()(cls)
    return cls


class TFCausalLanguageModelingLoss:
    """
    Loss function suitable for causal language modeling (CLM), that is, the task of guessing the next token.

    <Tip>

    Any label of -100 will be ignored (along with the corresponding logits) in the loss computation.

    </Tip>
    """

    def hf_compute_loss(self, labels, logits):
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=keras.losses.Reduction.NONE)
        if self.config.tf_legacy_loss:
            # make sure only labels that are not equal to -100 affect the loss
            active_loss = tf.not_equal(tf.reshape(labels, (-1,)), -100)
            reduced_logits = tf.boolean_mask(tf.reshape(logits, (-1, shape_list(logits)[2])), active_loss)
            labels = tf.boolean_mask(tf.reshape(labels, (-1,)), active_loss)
            return loss_fn(labels, reduced_logits)

        # Clip negative labels to zero here to avoid NaNs and errors - those positions will get masked later anyway
        unmasked_loss = loss_fn(tf.nn.relu(labels), logits)
        # make sure only labels that are not equal to -100 affect the loss
        loss_mask = tf.cast(labels != -100, dtype=unmasked_loss.dtype)
        masked_loss = unmasked_loss * loss_mask
        reduced_masked_loss = tf.reduce_sum(masked_loss) / tf.reduce_sum(loss_mask)
        return tf.reshape(reduced_masked_loss, (1,))


class TFQuestionAnsweringLoss:
    """
    Loss function suitable for question answering.
    """

    def hf_compute_loss(self, labels, logits):
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=keras.losses.Reduction.NONE)
        start_loss = loss_fn(labels["start_position"], logits[0])
        end_loss = loss_fn(labels["end_position"], logits[1])

        return (start_loss + end_loss) / 2.0


class TFTokenClassificationLoss:
    """
    Loss function suitable for token classification.

    <Tip>

    Any label of -100 will be ignored (along with the corresponding logits) in the loss computation.

    </Tip>
    """

    def hf_compute_loss(self, labels, logits):
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=keras.losses.Reduction.NONE)
        if tf.executing_eagerly():  # Data-dependent conditionals are forbidden in XLA
            if tf.math.reduce_any(labels == -1):
                tf.print("Using `-1` to mask the loss for the token is deprecated. Please use `-100` instead.")

        if self.config.tf_legacy_loss:
            # make sure only labels that are not equal to -100
            # are taken into account as loss
            if tf.math.reduce_any(labels == -1):
                tf.print("Using `-1` to mask the loss for the token is deprecated. Please use `-100` instead.")
                active_loss = tf.reshape(labels, (-1,)) != -1
            else:
                active_loss = tf.reshape(labels, (-1,)) != -100
            reduced_logits = tf.boolean_mask(tf.reshape(logits, (-1, shape_list(logits)[2])), active_loss)
            labels = tf.boolean_mask(tf.reshape(labels, (-1,)), active_loss)

            return loss_fn(labels, reduced_logits)

        # Clip negative labels to zero here to avoid NaNs and errors - those positions will get masked later anyway
        unmasked_loss = loss_fn(tf.nn.relu(labels), logits)
        # make sure only labels that are not equal to -100 or -1
        # are taken into account as loss
        loss_mask = tf.cast(labels >= 0, dtype=unmasked_loss.dtype)
        # Avoid possible division by zero later
        # Masked positions will have a loss of NaN because -100 and -1 are not valid labels
        masked_loss = unmasked_loss * loss_mask
        reduced_masked_loss = tf.reduce_sum(masked_loss) / tf.reduce_sum(loss_mask)
        return tf.reshape(reduced_masked_loss, (1,))


class TFSequenceClassificationLoss:
    """
    Loss function suitable for sequence classification.
    """

    def hf_compute_loss(self, labels, logits):
        if logits.shape.rank == 1 or logits.shape[1] == 1:
            loss_fn = keras.losses.MeanSquaredError(reduction=keras.losses.Reduction.NONE)
            if labels.shape.rank == 1:
                # MeanSquaredError returns a scalar loss if the labels are 1D, so avoid that
                labels = tf.expand_dims(labels, axis=-1)
        else:
            loss_fn = keras.losses.SparseCategoricalCrossentropy(
                from_logits=True, reduction=keras.losses.Reduction.NONE
            )

        return loss_fn(labels, logits)


class TFMultipleChoiceLoss:
    """Loss function suitable for multiple choice tasks."""

    def hf_compute_loss(self, labels, logits):
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=keras.losses.Reduction.NONE)
        return loss_fn(labels, logits)


class TFMaskedLanguageModelingLoss(TFCausalLanguageModelingLoss):
    """
    Loss function suitable for masked language modeling (MLM), that is, the task of guessing the masked tokens.

    <Tip>

    Any label of -100 will be ignored (along with the corresponding logits) in the loss computation.

    </Tip>
    """


class TFNextSentencePredictionLoss:
    """
    Loss function suitable for next sentence prediction (NSP), that is, the task of guessing the next sentence.

    <Tip>

    Any label of -100 will be ignored (along with the corresponding logits) in the loss computation.

    </Tip>
    """

    def hf_compute_loss(self, labels, logits):
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=keras.losses.Reduction.NONE)
        if self.config.tf_legacy_loss:
            # make sure only labels that are not equal to -100
            # are taken into account as loss
            next_sentence_active_loss = tf.not_equal(tf.reshape(labels, (-1,)), -100)
            next_sentence_reduced_logits = tf.boolean_mask(tf.reshape(logits, (-1, 2)), next_sentence_active_loss)
            next_sentence_label = tf.boolean_mask(tf.reshape(labels, (-1,)), next_sentence_active_loss)

            return loss_fn(next_sentence_label, next_sentence_reduced_logits)

        # make sure only labels that are not equal to -100
        # are taken into account as loss

        # Clip negative labels to zero here to avoid NaNs and errors - those positions will get masked later anyway
        unmasked_ns_loss = loss_fn(y_true=tf.nn.relu(labels), y_pred=logits)
        ns_loss_mask = tf.cast(labels != -100, dtype=unmasked_ns_loss.dtype)
        # Just zero out samples where label is -100, no reduction
        masked_ns_loss = unmasked_ns_loss * ns_loss_mask

        return masked_ns_loss


def booleans_processing(config, **kwargs):
    """
    Process the input booleans of each model.

    Args:
        config ([`PretrainedConfig`]):
            The config of the running model.
        **kwargs:
            The boolean parameters

    Returns:
        A dictionary with the proper values for each boolean
    """
    final_booleans = {}

    # Pure conv models (such as ConvNext) do not have `output_attentions`. If the signature has
    # `output_attentions`, it will be present here in `kwargs`, even if unset (in that case, as `None`)
    if "output_attentions" in kwargs:
        final_booleans["output_attentions"] = (
            kwargs["output_attentions"] if kwargs["output_attentions"] is not None else config.output_attentions
        )
    final_booleans["output_hidden_states"] = (
        kwargs["output_hidden_states"] if kwargs["output_hidden_states"] is not None else config.output_hidden_states
    )
    final_booleans["return_dict"] = kwargs["return_dict"] if kwargs["return_dict"] is not None else config.return_dict

    if "use_cache" in kwargs:
        final_booleans["use_cache"] = (
            kwargs["use_cache"] if kwargs["use_cache"] is not None else getattr(config, "use_cache", None)
        )
    return final_booleans


def unpack_inputs(func):
    """
    Decorator that processes the inputs to a Keras layer, passing them to the layer as keyword arguments. This enables
    downstream use of the inputs by their variable name, even if they arrive packed as a dictionary in the first input
    (common case in Keras).

    Args:
        func (`callable`):
            The callable function of the TensorFlow model.


    Returns:
        A callable that wraps the original `func` with the behavior described above.
    """

    original_signature = inspect.signature(func)

    @functools.wraps(func)
    def run_call_with_unpacked_inputs(self, *args, **kwargs):
        # isolates the actual `**kwargs` for the decorated function
        kwargs_call = {key: val for key, val in kwargs.items() if key not in dict(original_signature.parameters)}
        fn_args_and_kwargs = {key: val for key, val in kwargs.items() if key not in kwargs_call}
        fn_args_and_kwargs.update({"kwargs_call": kwargs_call})

        # move any arg into kwargs, if they exist
        fn_args_and_kwargs.update(dict(zip(func.__code__.co_varnames[1:], args)))

        # Encoder Decoder models delegate the application of the configuration options to their inner models.
        if "EncoderDecoder" in self.__class__.__name__:
            config = None
        else:
            config = self.config

        unpacked_inputs = input_processing(func, config, **fn_args_and_kwargs)
        return func(self, **unpacked_inputs)

    # Keras enforces the first layer argument to be passed, and checks it through `inspect.getfullargspec()`. This
    # function does not follow wrapper chains (i.e. ignores `functools.wraps()`), meaning that without the line below
    # Keras would attempt to check the first argument against the literal signature of the wrapper.
    run_call_with_unpacked_inputs.__signature__ = original_signature

    return run_call_with_unpacked_inputs


def input_processing(func, config, **kwargs):
    """
    Process the input of each TensorFlow model including the booleans. In case of a list of symbolic inputs, each input
    has to be named accordingly to the parameters name, i.e. `input_ids = keras.Input(shape=(128,), dtype='int32',
    name="input_ids")` otherwise the order of the tensors will not be guaranteed during the training.

    Args:
        func (`callable`):
            The callable function of the TensorFlow model.
        config ([`PretrainedConfig`]):
            The config of the running model.
        **kwargs:
            The inputs of the model.

    Returns:
        Two lists, one for the missing layers, and another one for the unexpected layers.
    """
    signature = dict(inspect.signature(func).parameters)
    has_kwargs = bool(signature.pop("kwargs", None))
    signature.pop("self", None)
    parameter_names = list(signature.keys())
    main_input_name = parameter_names[0]
    main_input = kwargs.pop(main_input_name, None)
    output = {}
    allowed_types = (tf.Tensor, bool, int, ModelOutput, tuple, list, dict, np.ndarray)

    if "inputs" in kwargs["kwargs_call"]:
        warnings.warn(
            "The `inputs` argument is deprecated and will be removed in a future version, use `input_ids` instead.",
            FutureWarning,
        )

        output["input_ids"] = kwargs["kwargs_call"].pop("inputs")

    if "decoder_cached_states" in kwargs["kwargs_call"]:
        warnings.warn(
            "The `decoder_cached_states` argument is deprecated and will be removed in a future version, use"
            " `past_key_values` instead.",
            FutureWarning,
        )
        output["past_key_values"] = kwargs["kwargs_call"].pop("decoder_cached_states")

    if "past" in kwargs["kwargs_call"] and "past_key_values" in parameter_names:
        warnings.warn(
            "The `past` argument is deprecated and will be removed in a future version, use `past_key_values`"
            " instead.",
            FutureWarning,
        )
        kwargs["past_key_values"] = kwargs["kwargs_call"].pop("past")
    elif "past_key_values" in kwargs["kwargs_call"] and "past" in parameter_names:
        kwargs["past"] = kwargs["kwargs_call"].pop("past_key_values")

    if has_kwargs:
        output["kwargs"] = kwargs.pop("kwargs_call", {})
    else:
        if len(kwargs["kwargs_call"]) > 0:
            raise ValueError(
                "The following keyword arguments are not supported by this model:"
                f" {list(kwargs['kwargs_call'].keys())}."
            )
        kwargs.pop("kwargs_call")

    for k, v in kwargs.items():
        if isinstance(v, allowed_types) or tf.is_tensor(v) or v is None:
            output[k] = v
        else:
            raise ValueError(f"Data of type {type(v)} is not allowed only {allowed_types} is accepted for {k}.")

    if isinstance(main_input, (tuple, list)):
        for i, input in enumerate(main_input):
            # EagerTensors don't allow to use the .name property so we check for a real Tensor
            if is_tf_symbolic_tensor(input):
                # Tensor names have always the pattern `name:id` then we check only the
                # `name` part
                tensor_name = input.name.split(":")[0]

                if tensor_name in parameter_names:
                    output[tensor_name] = input
                else:
                    output[parameter_names[i]] = input
            elif isinstance(input, allowed_types) or input is None:
                output[parameter_names[i]] = input
            else:
                raise ValueError(
                    f"Data of type {type(input)} is not allowed only {allowed_types} is accepted for"
                    f" {parameter_names[i]}."
                )
    elif isinstance(main_input, Mapping):
        if "inputs" in main_input:
            warnings.warn(
                "The `inputs` argument is deprecated and will be removed in a future version, use `input_ids`"
                " instead.",
                FutureWarning,
            )

            output["input_ids"] = main_input.pop("inputs")

        if "decoder_cached_states" in main_input:
            warnings.warn(
                "The `decoder_cached_states` argument is deprecated and will be removed in a future version, use"
                " `past_key_values` instead.",
                FutureWarning,
            )
            output["past_key_values"] = main_input.pop("decoder_cached_states")

        for k, v in dict(main_input).items():
            if isinstance(v, allowed_types) or v is None:
                output[k] = v
            elif k not in parameter_names and "args" not in parameter_names:
                logger.warning(
                    f"The parameter {k} does not belongs to the parameter list {parameter_names} and will be ignored."
                )
                continue
            else:
                raise ValueError(f"Data of type {type(v)} is not allowed only {allowed_types} is accepted for {k}.")
    else:
        if tf.is_tensor(main_input) or main_input is None:
            output[main_input_name] = main_input
        else:
            raise ValueError(
                f"Data of type {type(main_input)} is not allowed only {allowed_types} is accepted for"
                f" {main_input_name}."
            )

    # Populates any unspecified argument with their default value, according to the signature.
    for name in parameter_names:
        if name not in list(output.keys()) and name != "args":
            output[name] = kwargs.pop(name, signature[name].default)

    # When creating a SavedModel TF calls the method with LayerCall.__call__(args, **kwargs)
    # So to respect the proper output we have to add this exception
    if "args" in output:
        if output["args"] is not None and is_tf_symbolic_tensor(output["args"]):
            tensor_name = output["args"].name.split(":")[0]
            output[tensor_name] = output["args"]
        else:
            # `args` in this case is always the first parameter, then `input_ids`
            output["input_ids"] = output["args"]

        del output["args"]

    if "kwargs" in output:
        del output["kwargs"]

    cast_output = {}
    for key, val in output.items():
        if isinstance(val, tf.Tensor) and val.dtype == tf.int64:
            cast_output[key] = tf.cast(val, tf.int32)
        elif isinstance(val, np.ndarray) and val.dtype == np.int64:
            cast_output[key] = val.astype(np.int32)
        else:
            cast_output[key] = val

    output = cast_output
    del cast_output

    if config is not None:
        boolean_dict = {
            k: v
            for k, v in output.items()
            if k in ["return_dict", "output_attentions", "output_hidden_states", "use_cache"]
        }

        output.update(
            booleans_processing(
                config=config,
                **boolean_dict,
            )
        )

    return output


def dtype_byte_size(dtype):
    """
    Returns the size (in bytes) occupied by one parameter of type `dtype`.

    Example:

    ```py
    >>> dtype_byte_size(tf.float32)
    4
    ```
    """
    if dtype == tf.bool:
        return 1 / 8
    bit_search = re.search(r"[^\d](\d+)$", dtype.name)
    if bit_search is None:
        raise ValueError(f"`dtype` is not a valid dtype: {dtype}.")
    bit_size = int(bit_search.groups()[0])
    return bit_size // 8


def strip_model_name_and_prefix(name, _prefix=None):
    if _prefix is not None and name.startswith(_prefix):
        name = name[len(_prefix) :]
        if name.startswith("/"):
            name = name[1:]
    if "model." not in name and len(name.split("/")) > 1:
        name = "/".join(name.split("/")[1:])
    return name


def tf_shard_checkpoint(weights, max_shard_size="10GB", weights_name: str = TF2_WEIGHTS_NAME):
    """
    Splits a model state dictionary in sub-checkpoints so that the final size of each sub-checkpoint does not exceed a
    given size.

    The sub-checkpoints are determined by iterating through the `state_dict` in the order of its keys, so there is no
    optimization made to make each sub-checkpoint as close as possible to the maximum size passed. For example, if the
    limit is 10GB and we have weights of sizes [6GB, 6GB, 2GB, 6GB, 2GB, 2GB] they will get sharded as [6GB], [6+2GB],
    [6+2+2GB] and not [6+2+2GB], [6+2GB], [6GB].

    <Tip warning={true}>

    If one of the model's weight is bigger that `max_shard_size`, it will end up in its own sub-checkpoint which will
    have a size greater than `max_shard_size`.

    </Tip>

    Args:
        weights (`Dict[str, tf.RessourceVariable]`): The list of tf.RessourceVariable of a model to save.
        max_shard_size (`int` or `str`, *optional*, defaults to `"10GB"`):
            The maximum size of each sub-checkpoint. If expressed as a string, needs to be digits followed by a unit
            (like `"5MB"`).
    """
    max_shard_size = convert_file_size_to_int(max_shard_size)

    sharded_state_dicts = []
    current_block = []
    current_block_size = 0
    total_size = 0

    for item in weights:
        weight_size = item.numpy().size * dtype_byte_size(item.dtype)

        # If this weight is going to tip up over the maximal size, we split.
        if current_block_size + weight_size > max_shard_size:
            sharded_state_dicts.append(current_block)
            current_block = []
            current_block_size = 0

        current_block.append(item)
        current_block_size += weight_size
        total_size += weight_size

    # Add the last block
    sharded_state_dicts.append(current_block)

    # If we only have one shard, we return it
    if len(sharded_state_dicts) == 1:
        return {weights_name: sharded_state_dicts[0]}, None

    # Otherwise, let's build the index
    weight_map = {}
    shards = {}
    for idx, shard in enumerate(sharded_state_dicts):
        shard_file = weights_name.replace(".h5", f"-{idx+1:05d}-of-{len(sharded_state_dicts):05d}.h5")
        shard_file = shard_file.replace(
            ".safetensors", f"-{idx + 1:05d}-of-{len(sharded_state_dicts):05d}.safetensors"
        )
        shards[shard_file] = shard
        for weight in shard:
            weight_name = weight.name
            weight_map[weight_name] = shard_file

    # Add the metadata
    metadata = {"total_size": total_size}
    index = {"metadata": metadata, "weight_map": weight_map}
    return shards, index


def load_tf_sharded_weights(model, shard_files, ignore_mismatched_sizes=False, strict=False, _prefix=None):
    """
    This is the same as `load_tf_weights` but for a sharded checkpoint. Detect missing and unexpected layers and load
    the TF weights from the shard file accordingly to their names and shapes.

    This load is performed efficiently: each checkpoint shard is loaded one by one in RAM and deleted after being
    loaded in the model.

    Args:
        model (`keras.models.Model`): The model in which to load the checkpoint.
        shard_files (`str` or `os.PathLike`): A list containing the sharded checkpoint names.
        ignore_mismatched_sizes`bool`, *optional`, defaults to `True`):
            Whether or not to ignore the mismatch between the sizes
        strict (`bool`, *optional*, defaults to `True`):
            Whether to strictly enforce that the keys in the model state dict match the keys in the sharded checkpoint.

    Returns:
        Three lists, one for the missing layers, another one for the unexpected layers, and a last one for the
        mismatched layers.
    """

    # Load the index
    unexpected_keys = set()
    saved_keys = set()
    mismatched_keys = set()

    # Since TF adds the name of the class to its weights, and uses the index and not the name of the layer to load
    # the weight, we have to get rid of the first prefix of the name of the layer.
    model_keys = set()
    model_layer_map = {}
    for i, k in enumerate(model.weights):
        layer_name = k.name
        if _prefix is not None and layer_name.startswith(_prefix):
            layer_name = layer_name[len(_prefix) :]
            layer_name = layer_name.lstrip("/")
        if not ("model." in layer_name or len(layer_name.split("/")) == 1):
            layer_name = "/".join(layer_name.split("/")[1:])
        model_keys.add(layer_name)
        model_layer_map[layer_name] = i

    for shard_file in shard_files:
        saved_weight_names_set, unexpected_keys_set, mismatched_keys_set = load_tf_shard(
            model,
            model_layer_map,
            shard_file,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            _prefix=_prefix,
        )
        saved_keys.update(saved_weight_names_set)
        unexpected_keys.update(unexpected_keys_set)
        mismatched_keys.update(mismatched_keys_set)
        gc.collect()

    missing_keys = model_keys - saved_keys
    if strict and (len(missing_keys) > 0 or len(unexpected_keys) > 0):
        error_message = f"Error(s) in loading state_dict for {model.__class__.__name__}"
        if len(missing_keys) > 0:
            str_missing_keys = ",".join([f'"{k}"' for k in missing_keys])
            error_message += f"\nMissing key(s): {str_missing_keys}."
        if len(unexpected_keys) > 0:
            str_unexpected_keys = ",".join([f'"{k}"' for k in unexpected_keys])
            error_message += f"\nMissing key(s): {str_unexpected_keys}."
        raise RuntimeError(error_message)

    return missing_keys, unexpected_keys, mismatched_keys


def load_tf_shard(model, model_layer_map, resolved_archive_file, ignore_mismatched_sizes=False, _prefix=None):
    """
    Loads a shard from a sharded checkpoint file. Can be either H5 or Safetensors.
    Handles missing keys and unexpected keys.

    Args:
        model (`keras.models.Model`): Model in which the weights are loaded
        model_layer_map (`Dict`): A dictionary mapping the layer name to the index of the layer in the model.
        resolved_archive_file (`str`): Path to the checkpoint file from which the weights will be loaded
        ignore_mismatched_sizes (`bool`, *optional*, defaults to `False`): Whether to ignore the mismatched keys

    Returns:
        `keras.models.Model`: Three lists, one for the layers that were found and succesfully restored (from the
        shard file), one for the mismatched layers, and another one for the unexpected layers.
    """
    saved_weight_names_set = set()
    saved_weights = {}
    mismatched_keys = set()
    unexpected_keys = set()
    # Read the H5 file
    try:
        with h5py.File(resolved_archive_file, "r") as sharded_checkpoint_file:
            # Retrieve the name of each layer from the H5 file
            saved_h5_model_layers_name = set(load_attributes_from_hdf5_group(sharded_checkpoint_file, "layer_names"))
            weight_value_tuples = []

            # Compute missing and unexpected sub layers
            # Store the weights in list of tuples that looks like [(weight_object, value_of_weight),...]
            for layer_name in saved_h5_model_layers_name:
                h5_layer_object = sharded_checkpoint_file[layer_name]
                saved_weights[layer_name] = np.asarray(h5_layer_object)

                saved_weight_names_set.add(layer_name)

                if layer_name not in model_layer_map:
                    unexpected_keys.add(layer_name)
                else:
                    symbolic_weight = model.weights[model_layer_map[layer_name]]

                    saved_weight_value = saved_weights[layer_name]
                    # If the current weight is found
                    if saved_weight_value is not None:
                        # Check if the shape of the current weight and the one from the H5 file are different
                        if K.int_shape(symbolic_weight) != saved_weight_value.shape:
                            # If yes we reshape the weight from the H5 file accordingly to the current weight
                            # If the two shapes are not compatible we raise an issue
                            try:
                                array = np.reshape(saved_weight_value, K.int_shape(symbolic_weight))
                            except ValueError as e:
                                if ignore_mismatched_sizes:
                                    mismatched_keys.add(
                                        (layer_name, saved_weight_value.shape, K.int_shape(symbolic_weight))
                                    )
                                    continue
                                else:
                                    raise e
                        else:
                            array = saved_weight_value

                    # We create the tuple that will be loaded and add it to the final list
                    weight_value_tuples.append((symbolic_weight, array))

        K.batch_set_value(weight_value_tuples)

        return saved_weight_names_set, unexpected_keys, mismatched_keys

    except Exception as e:
        try:
            with open(resolved_archive_file) as f:
                if f.read().startswith("version"):
                    raise OSError(
                        "You seem to have cloned a repository without having git-lfs installed. Please install "
                        "git-lfs and run `git lfs install` followed by `git lfs pull` in the folder "
                        "you cloned."
                    )
                else:
                    raise ValueError(
                        f"Unable to locate the file {resolved_archive_file} which is necessary to load this pretrained"
                        " model. Make sure you have saved the model properly."
                    ) from e
        except (UnicodeDecodeError, ValueError):
            raise OSError(
                f"Unable to load weights from TF checkpoint file for '{resolved_archive_file}' "
                f"at '{resolved_archive_file}'. "
                "If you tried to load a TF model from a sharded checkpoint, you should try converting the model "
                "by loading it in pytorch and saving it localy. A convertion script should be realeased soon."
            )


def load_tf_sharded_weights_from_safetensors(
    model, shard_files, ignore_mismatched_sizes=False, strict=False, _prefix=None
):
    """
    This is the same as `load_tf_weights_from_safetensors` but for a sharded TF-format safetensors checkpoint.
    Detect missing and unexpected layers and load the TF weights from the shard file accordingly to their names and
    shapes.

    This load is performed efficiently: each checkpoint shard is loaded one by one in RAM and deleted after being
    loaded in the model.

    Args:
        model (`keras.models.Model`): The model in which to load the checkpoint.
        shard_files (`str` or `os.PathLike`): A list containing the sharded checkpoint names.
        ignore_mismatched_sizes`bool`, *optional`, defaults to `True`):
            Whether or not to ignore the mismatch between the sizes
        strict (`bool`, *optional*, defaults to `True`):
            Whether to strictly enforce that the keys in the model state dict match the keys in the sharded checkpoint.

    Returns:
        Three lists, one for the missing layers, another one for the unexpected layers, and a last one for the
        mismatched layers.
    """

    # Load the index
    unexpected_keys = set()
    all_missing_keys = []
    mismatched_keys = set()

    for shard_file in shard_files:
        missing_layers, unexpected_layers, mismatched_layers = load_tf_weights_from_safetensors(
            model,
            shard_file,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            _prefix=_prefix,
        )
        all_missing_keys.append(set(missing_layers))
        unexpected_keys.update(unexpected_layers)
        mismatched_keys.update(mismatched_layers)
        gc.collect()
    missing_keys = set.intersection(*all_missing_keys)

    if strict and (len(missing_keys) > 0 or len(unexpected_keys) > 0):
        error_message = f"Error(s) in loading state_dict for {model.__class__.__name__}"
        if len(missing_keys) > 0:
            str_missing_keys = ",".join([f'"{k}"' for k in missing_keys])
            error_message += f"\nMissing key(s): {str_missing_keys}."
        if len(unexpected_keys) > 0:
            str_unexpected_keys = ",".join([f'"{k}"' for k in unexpected_keys])
            error_message += f"\nMissing key(s): {str_unexpected_keys}."
        raise RuntimeError(error_message)

    return missing_keys, unexpected_keys, mismatched_keys


def load_tf_weights(model, resolved_archive_file, ignore_mismatched_sizes=False, _prefix=None):
    """
    Detect missing and unexpected layers and load the TF weights from the shard file accordingly to their names and
    shapes.

    Args:
        model (`keras.models.Model`):
            The model to load the weights into.
        resolved_archive_file (`str`):
            The location of the H5 file.
        ignore_mismatched_sizes (`bool`, *optional*, defaults to `False`):
            Whether or not to ignore weights with shapes that don't match between the checkpoint of the model.

    Returns:
        Three lists, one for the missing layers, another one for the unexpected layers, and a last one for the
        mismatched layers.
    """
    if resolved_archive_file.endswith(".safetensors"):
        load_function = load_tf_weights_from_safetensors
    else:
        load_function = load_tf_weights_from_h5

    return load_function(
        model, resolved_archive_file, ignore_mismatched_sizes=ignore_mismatched_sizes, _prefix=_prefix
    )


def load_tf_weights_from_h5(model, resolved_archive_file, ignore_mismatched_sizes=False, _prefix=None):
    mismatched_layers = []

    # Read the H5 file
    with h5py.File(resolved_archive_file, "r") as sharded_checkpoint_file:
        # Retrieve the name of each layer from the H5 file
        saved_h5_model_layers_name = set(load_attributes_from_hdf5_group(sharded_checkpoint_file, "layer_names"))

        # Find the missing layers from the high level list of layers
        missing_layers = list({layer.name for layer in model.layers} - saved_h5_model_layers_name)

        # Find the unexpected layers from the high level list of layers
        unexpected_layers = list(saved_h5_model_layers_name - {layer.name for layer in model.layers})
        saved_weight_names_set = set()
        symbolic_weights_names = set()
        weight_value_tuples = []

        # Compute missing and unexpected sub layers
        # Store the weights in list of tuples that looks like [(weight_object, value_of_weight),...]
        for layer in model.layers:
            # if layer_name from the H5 file belongs to the layers from the instantiated model
            if layer.name in saved_h5_model_layers_name:
                # Get the H5 layer object from its name
                h5_layer_object = sharded_checkpoint_file[layer.name]
                # Get all the weights as a list from the layer object
                symbolic_weights = layer.trainable_weights + layer.non_trainable_weights
                saved_weights = {}

                # Create a dict from the H5 saved model that looks like {"weight_name": weight_value}
                # And a set with only the names
                for weight_name in load_attributes_from_hdf5_group(h5_layer_object, "weight_names"):
                    # TF names always start with the model name so we ignore it
                    name = "/".join(weight_name.split("/")[1:])

                    if _prefix is not None:
                        name = _prefix + "/" + name

                    saved_weights[name] = np.asarray(h5_layer_object[weight_name])

                    # Add the updated name to the final list for computing missing/unexpected values
                    saved_weight_names_set.add(name)

                # Loop over each weights from the instantiated model and compare with the weights from the H5 file
                for symbolic_weight in symbolic_weights:
                    # TF names always start with the model name so we ignore it
                    if _prefix is not None:
                        delimeter = len(_prefix.split("/"))
                        symbolic_weight_name = "/".join(
                            symbolic_weight.name.split("/")[:delimeter]
                            + symbolic_weight.name.split("/")[delimeter + 1 :]
                        )
                    else:
                        symbolic_weight_name = "/".join(symbolic_weight.name.split("/")[1:])

                    # here we check if the current weight is among the weights from the H5 file
                    # If yes, get the weight_value of the corresponding weight from the H5 file
                    # If not, make the value to None
                    saved_weight_value = saved_weights.get(symbolic_weight_name, None)

                    # Retrocompatibility patch: some embeddings are stored with the weights name (e.g. Bart's
                    # `model.shared/embeddings:0` are stored as `model.shared/weights:0`)
                    if saved_weight_value is None and symbolic_weight_name.endswith("embeddings:0"):
                        symbolic_weight_name = symbolic_weight_name[:-12] + "weight:0"
                        saved_weight_value = saved_weights.get(symbolic_weight_name, None)

                    # Add the updated name to the final list for computing missing/unexpected values
                    symbolic_weights_names.add(symbolic_weight_name)

                    # If the current weight is found
                    if saved_weight_value is not None:
                        # Check if the shape of the current weight and the one from the H5 file are different
                        if K.int_shape(symbolic_weight) != saved_weight_value.shape:
                            # If yes we reshape the weight from the H5 file accordingly to the current weight
                            # If the two shapes are not compatible we raise an issue
                            try:
                                array = np.reshape(saved_weight_value, K.int_shape(symbolic_weight))
                            except ValueError as e:
                                if ignore_mismatched_sizes:
                                    mismatched_layers.append(
                                        (symbolic_weight_name, saved_weight_value.shape, K.int_shape(symbolic_weight))
                                    )
                                    continue
                                else:
                                    raise e
                        else:
                            array = saved_weight_value

                        # We create the tuple that will be loaded and add it to the final list
                        weight_value_tuples.append((symbolic_weight, array))

    # Load all the weights
    K.batch_set_value(weight_value_tuples)

    # Compute the missing and unexpected layers
    missing_layers.extend(list(symbolic_weights_names - saved_weight_names_set))
    unexpected_layers.extend(list(saved_weight_names_set - symbolic_weights_names))

    return missing_layers, unexpected_layers, mismatched_layers


def load_tf_weights_from_safetensors(model, resolved_archive_file, ignore_mismatched_sizes=False, _prefix=None):
    # Read the safetensors file
    with safe_open(resolved_archive_file, framework="tf") as safetensors_archive:
        mismatched_layers = []
        weight_names = [strip_model_name_and_prefix(w.name, _prefix=_prefix) for w in model.weights]
        loaded_weight_names = list(safetensors_archive.keys())
        # Find the missing layers from the high level list of layers
        missing_layers = list(set(weight_names) - set(loaded_weight_names))
        # Find the unexpected layers from the high level list of layers
        unexpected_layers = list(set(loaded_weight_names) - set(weight_names))

        for weight in model.weights:
            weight_name = strip_model_name_and_prefix(weight.name, _prefix=_prefix)
            if weight_name in loaded_weight_names:
                weight_value = safetensors_archive.get_tensor(weight_name)
                # Check if the shape of the current weight and the one from the H5 file are different
                if K.int_shape(weight) != weight_value.shape:
                    # If yes we reshape the weight from the H5 file accordingly to the current weight
                    # If the two shapes are not compatible we raise an issue
                    try:
                        weight_value = tf.reshape(weight_value, K.int_shape(weight))
                    except (ValueError, tf.errors.InvalidArgumentError) as e:
                        if ignore_mismatched_sizes:
                            mismatched_layers.append((weight_name, weight_value.shape, K.int_shape(weight)))
                            continue
                        else:
                            raise e

                K.set_value(weight, weight_value)  # weight.assign() might break if weight is a DTensor
    return missing_layers, unexpected_layers, mismatched_layers


def init_copy_embeddings(old_embeddings, new_num_tokens):
    r"""
    This function aims to reduce the embeddings in case new_num_tokens < old_num_tokens or to pad with -1 in case
    new_num_tokens > old_num_tokens. A mask is also computed in order to know which weight in the embeddings should be
    kept or not. Example:

        - if new_num_tokens=5 and old_num_tokens=4 and old_embeddings=[w1,w2,w3,w4]

            -  mask=[True,True,True,True,False] and current_weights=[w1,w2,w3,w4,-1]
        - if new_num_tokens=4 and old_num_tokens=5 and old_embeddings=[w1,w2,w3,w4,w5]

            - mask=[True,True,True,True] and current_weights=[w1,w2,w3,w4]
    """
    old_num_tokens, old_embedding_dim = shape_list(old_embeddings)
    size_diff = new_num_tokens - old_num_tokens

    # initialize new embeddings
    # Copy token embeddings from the previous ones
    if tf.math.greater(size_diff, 0):
        # if the new size is greater than the old one, we extend the current embeddings with a padding until getting new size
        # and we create a mask to properly identify the padded values and be replaced by the values of the newly created
        # embeddings
        current_weights = tf.pad(
            old_embeddings.value(), tf.convert_to_tensor([[0, size_diff], [0, 0]]), constant_values=-1
        )
        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
        mask = tf.fill(tf.convert_to_tensor([num_tokens_to_copy, 1]), True)
        mask = tf.pad(mask, tf.convert_to_tensor([[0, size_diff], [0, 0]]), constant_values=False)
    else:
        # if the new size if lower than the old one, we take the current embeddings until the new size
        current_weights = tf.slice(
            old_embeddings.value(),
            tf.convert_to_tensor([0, 0]),
            tf.convert_to_tensor([new_num_tokens, old_embedding_dim]),
        )
        mask = tf.fill(tf.convert_to_tensor([new_num_tokens, 1]), True)

    return mask, current_weights


class TFPreTrainedModel(keras.Model, TFModelUtilsMixin, TFGenerationMixin, PushToHubMixin):
    r"""
    Base class for all TF models.

    [`TFPreTrainedModel`] takes care of storing the configuration of the models and handles methods for loading,
    downloading and saving models as well as a few methods common to all models to:

        - resize the input embeddings,
        - prune heads in the self-attention heads.

    Class attributes (overridden by derived classes):

        - **config_class** ([`PretrainedConfig`]) -- A subclass of [`PretrainedConfig`] to use as configuration class
          for this model architecture.
        - **base_model_prefix** (`str`) -- A string indicating the attribute associated to the base model in derived
          classes of the same architecture adding modules on top of the base model.
        - **main_input_name** (`str`) -- The name of the principal input to the model (often `input_ids` for NLP
          models, `pixel_values` for vision models and `input_values` for speech models).
    """

    config_class = None
    base_model_prefix = ""
    main_input_name = "input_ids"
    _auto_class = None
    _using_dummy_loss = None
    _label_to_output_map = None

    # a list of re pattern of tensor names to ignore from the model when loading the model weights
    # (and avoid unnecessary warnings).
    _keys_to_ignore_on_load_missing = None
    # a list of re pattern of tensor names to ignore from the weights when loading the model weights
    # (and avoid unnecessary warnings).
    _keys_to_ignore_on_load_unexpected = None
    _requires_load_weight_prefix = False

    @property
    def dummy_inputs(self) -> Dict[str, tf.Tensor]:
        """
        Dummy inputs to build the network.

        Returns:
            `Dict[str, tf.Tensor]`: The dummy inputs.
        """
        dummies = {}
        for key, spec in self.input_signature.items():
            # 2 is the most correct arbitrary size. I will not be taking questions
            dummy_shape = [dim if dim is not None else 2 for dim in spec.shape]
            if spec.shape[0] is None:
                # But let's make the batch size 1 to save memory anyway
                dummy_shape[0] = 1
            dummies[key] = tf.ones(shape=dummy_shape, dtype=spec.dtype)
            if key == "token_type_ids":
                # Some models have token_type_ids but with a vocab_size of 1
                dummies[key] = tf.zeros_like(dummies[key])
        if self.config.add_cross_attention and "encoder_hidden_states" in inspect.signature(self.call).parameters:
            if "encoder_hidden_states" not in dummies:
                if self.main_input_name == "input_ids":
                    dummies["encoder_hidden_states"] = tf.ones(
                        shape=(1, 2, self.config.hidden_size), dtype=tf.float32, name="encoder_hidden_states"
                    )
                else:
                    raise NotImplementedError(
                        "Model has cross-attention but we couldn't infer the shape for the encoder hidden states. Please manually override dummy_inputs!"
                    )
        return dummies

    def build_in_name_scope(self):
        with tf.name_scope(self.name):
            self.build(input_shape=None)

    @property
    def framework(self) -> str:
        """
        :str: Identifies that this is a TensorFlow model.
        """
        return "tf"

    def build(self, input_shape=None):
        pass  # This is just here to make sure we don't call the superclass build()

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)
        if not isinstance(config, PretrainedConfig):
            raise TypeError(
                f"Parameter config in `{self.__class__.__name__}(config)` should be an instance of class "
                "`PretrainedConfig`. To create a model from a pretrained model use "
                f"`model = {self.__class__.__name__}.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        # Save config and origin of the pretrained weights if given in model
        self.config = config
        self.name_or_path = config.name_or_path
        self.generation_config = GenerationConfig.from_model_config(config) if self.can_generate() else None
        self._set_save_spec(self.input_signature)

    def get_config(self):
        return self.config.to_dict()

    @functools.wraps(keras.Model.fit)
    def fit(self, *args, **kwargs):
        args, kwargs = convert_batch_encoding(*args, **kwargs)
        return super().fit(*args, **kwargs)

    @functools.wraps(keras.Model.train_on_batch)
    def train_on_batch(self, *args, **kwargs):
        args, kwargs = convert_batch_encoding(*args, **kwargs)
        return super().train_on_batch(*args, **kwargs)

    @functools.wraps(keras.Model.test_on_batch)
    def test_on_batch(self, *args, **kwargs):
        args, kwargs = convert_batch_encoding(*args, **kwargs)
        return super().test_on_batch(*args, **kwargs)

    @functools.wraps(keras.Model.predict_on_batch)
    def predict_on_batch(self, *args, **kwargs):
        args, kwargs = convert_batch_encoding(*args, **kwargs)
        return super().predict_on_batch(*args, **kwargs)

    @functools.wraps(keras.Model.predict)
    def predict(self, *args, **kwargs):
        args, kwargs = convert_batch_encoding(*args, **kwargs)
        return super().predict(*args, **kwargs)

    @functools.wraps(keras.Model.evaluate)
    def evaluate(self, *args, **kwargs):
        args, kwargs = convert_batch_encoding(*args, **kwargs)
        return super().evaluate(*args, **kwargs)

    @classmethod
    def from_config(cls, config, **kwargs):
        if isinstance(config, PretrainedConfig):
            return cls._from_config(config, **kwargs)
        return cls._from_config(cls.config_class.from_dict(config, **kwargs))

    @classmethod
    def _from_config(cls, config, **kwargs):
        """
        All context managers that the model should be initialized under go here.
        """
        return cls(config, **kwargs)

    def get_head_mask(self, head_mask: tf.Tensor | None, num_hidden_layers: int) -> tf.Tensor:
        """
        Prepare the head mask if needed.

        Args:
            head_mask (`tf.Tensor` with shape `[num_heads]` or `[num_hidden_layers x num_heads]`, *optional*):
                The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 for discard).
            num_hidden_layers (`int`):
                The number of hidden layers in the model.

        Returns:
            `tf.Tensor` with shape `[num_hidden_layers x batch x num_heads x seq_length x seq_length]` or list with
            `[None]` for each layer.
        """
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
        else:
            head_mask = [None] * num_hidden_layers

        return head_mask

    def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
        """-> [num_hidden_layers x batch x num_heads x seq_length x seq_length]"""
        if head_mask.shape.rank == 1:
            head_mask = head_mask[None, None, :, None, None]
            head_mask = tf.repeat(head_mask, repeats=num_hidden_layers, axis=0)
        elif head_mask.shape.rank == 2:
            head_mask = head_mask[:, None, :, None, None]
        assert head_mask.shape.rank == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
        head_mask = tf.cast(head_mask, tf.float32)  # switch to float if need + fp16 compatibility
        return head_mask

    @tf.function
    def serving(self, inputs):
        """
        Args:
        Method used for serving the model. Does not have a specific signature, but will be specialized as concrete
        functions when saving with `save_pretrained`.
            inputs (`Dict[str, tf.Tensor]`):
                The input of the saved model as a dictionary of tensors.
        """
        output = self.call(inputs)

        return self.serving_output(output)

    @property
    def input_signature(self) -> Dict[str, tf.TensorSpec]:
        """
        This property should return a dict mapping input names to tf.TensorSpec objects, representing the expected
        shape and dtype for model inputs. It is used for both serving and for generating dummy inputs.
        """
        model_inputs = list(inspect.signature(self.call).parameters)
        sig = {}
        if "input_ids" in model_inputs:
            if self.__class__.__name__.endswith("ForMultipleChoice"):
                text_dims = 3
            else:
                text_dims = 2
            for input_name in (
                "input_ids",
                "attention_mask",
                "token_type_ids",
                "decoder_input_ids",
                "decoder_attention_mask",
            ):
                if input_name in model_inputs:
                    sig[input_name] = tf.TensorSpec([None] * text_dims, tf.int32, name=input_name)
        if "pixel_values" in model_inputs:
            pixel_values_shape = [None, None, None, None]
            if hasattr(self.config, "vision_config"):
                vision_config = self.config.vision_config
            else:
                vision_config = self.config
            if hasattr(vision_config, "num_channels"):
                pixel_values_shape[1] = vision_config.num_channels
            else:
                raise NotImplementedError(
                    "Could not infer number of channels from config, please override input_signature to specify input shapes."
                )
            if hasattr(vision_config, "image_size"):
                pixel_values_shape[2] = pixel_values_shape[3] = vision_config.image_size
            elif hasattr(vision_config, "input_size"):
                pixel_values_shape[2] = pixel_values_shape[3] = vision_config.input_size
            else:
                raise NotImplementedError(
                    "Could not infer input image shape from config, please override input_signature to specify input shapes."
                )
            sig["pixel_values"] = tf.TensorSpec(pixel_values_shape, tf.float32, name="pixel_values")
        if "input_features" in model_inputs:
            raise NotImplementedError("Audio models need a manually defined input_signature")
        return sig

    def serving_output(self, output):
        """
        Prepare the output of the saved model. Can be overridden if specific serving modifications are required.
        """
        if not isinstance(output, ModelOutput):
            return output
        for key in output:
            if key.endswith("hidden_states") and not getattr(self.config, "output_hidden_states", False):
                output[key] = None
            elif key.endswith("attentions") and not getattr(self.config, "output_attentions", False):
                output[key] = None
            elif key == "past_key_values" and not getattr(self.config, "use_cache", False):
                output[key] = None
            elif key == "cross_attentions" and not (
                getattr(self.config, "output_attentions", False) and getattr(self.config, "add_cross_attention", False)
            ):
                output[key] = None
            if isinstance(output[key], (tuple, list)):
                try:
                    output[key] = tf.convert_to_tensor(output[key])
                except (ValueError, tf.errors.InvalidArgumentError):
                    pass  # Layers may not have the same dimensions
        return output

    @classmethod
    def can_generate(cls) -> bool:
        """
        Returns whether this model can generate sequences with `.generate()`.

        Returns:
            `bool`: Whether this model can generate sequences with `.generate()`.
        """
        # Detects whether `prepare_inputs_for_generation` has been overwritten, which is a requirement for generation.
        # Alternativelly, the model can also have a custom `generate` function.
        if "GenerationMixin" in str(cls.prepare_inputs_for_generation) and "GenerationMixin" in str(cls.generate):
            return False
        return True

    def get_input_embeddings(self) -> keras.layers.Layer:
        """
        Returns the model's input embeddings layer.

        Returns:
            `tf.Variable`: The embeddings layer mapping vocabulary to hidden states.
        """
        main_layer = getattr(self, self.base_model_prefix, self)

        if main_layer is not self:
            return main_layer.get_input_embeddings()
        else:
            raise NotImplementedError

    def _save_checkpoint(self, checkpoint_dir, epoch):
        if not os.path.isdir(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        # We avoid tf.train.checkpoint or saving weights in TF format, even though that includes optimizer
        # state for us, because it requires special handling for objects like custom losses, which we use
        # internally and which users are likely to use too
        weights_path = os.path.join(checkpoint_dir, "weights.h5")
        self.save_weights(weights_path)
        extra_data = {"epoch": epoch, "optimizer_state": self.optimizer.get_weights()}
        extra_data_path = os.path.join(checkpoint_dir, "extra_data.pickle")
        with open(extra_data_path, "wb") as f:
            pickle.dump(extra_data, f)

    def prepare_tf_dataset(
        self,
        dataset: "datasets.Dataset",  # noqa:F821
        batch_size: int = 8,
        shuffle: bool = True,
        tokenizer: Optional["PreTrainedTokenizerBase"] = None,
        collate_fn: Optional[Callable] = None,
        collate_fn_args: Optional[Dict[str, Any]] = None,
        drop_remainder: Optional[bool] = None,
        prefetch: bool = True,
    ):
        """
        Wraps a HuggingFace [`~datasets.Dataset`] as a `tf.data.Dataset` with collation and batching. This method is
        designed to create a "ready-to-use" dataset that can be passed directly to Keras methods like `fit()` without
        further modification. The method will drop columns from the dataset if they don't match input names for the
        model. If you want to specify the column names to return rather than using the names that match this model, we
        recommend using `Dataset.to_tf_dataset()` instead.

        Args:
            dataset (`Any`):
                A [~`datasets.Dataset`] to be wrapped as a `tf.data.Dataset`.
            batch_size (`int`, *optional*, defaults to 8):
                The size of batches to return.
            shuffle (`bool`, defaults to `True`):
                Whether to return samples from the dataset in random order. Usually `True` for training datasets and
                `False` for validation/test datasets.
            tokenizer ([`PreTrainedTokenizerBase`], *optional*):
                A `PreTrainedTokenizer` that will be used to pad samples to create batches. Has no effect if a specific
                `collate_fn` is passed instead.
            collate_fn (`Callable`, *optional*):
                A function that collates samples from the dataset into a single batch. Defaults to
                `DefaultDataCollator` if no `tokenizer` is supplied or `DataCollatorWithPadding` if a `tokenizer` is
                passed.
            collate_fn_args (`Dict[str, Any]`, *optional*):
                A dict of arguments to pass to the `collate_fn` alongside the list of samples.
            drop_remainder (`bool`, *optional*):
                Whether to drop the final batch, if the batch_size does not evenly divide the dataset length. Defaults
                to the same setting as `shuffle`.
            prefetch (`bool`, defaults to `True`):
                Whether to add prefetching to the end of the `tf.data` pipeline. This is almost always beneficial for
                performance, but can be disabled in edge cases.


        Returns:
            `Dataset`: A `tf.data.Dataset` which is ready to pass to the Keras API.
        """
        requires_backends(self, ["datasets"])
        import datasets

        if collate_fn is None:
            if tokenizer is None:
                collate_fn = DefaultDataCollator(return_tensors="np")
            else:
                collate_fn = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="np")
        if collate_fn_args is None:
            collate_fn_args = {}

        if not isinstance(dataset, datasets.Dataset):
            raise TypeError("Dataset argument should be a datasets.Dataset!")
        model_inputs = list(inspect.signature(self.call).parameters)
        model_labels = find_labels(self.__class__)
        if "cols_to_retain" in list(inspect.signature(dataset._get_output_signature).parameters.keys()):
            output_signature, _ = dataset._get_output_signature(
                dataset,
                batch_size=None,
                collate_fn=collate_fn,
                collate_fn_args=collate_fn_args,
                cols_to_retain=model_inputs,
            )
        else:
            # TODO Matt: This is a workaround for older versions of datasets that are missing the `cols_to_retain`
            #            argument. We should remove this once the minimum supported version of datasets is > 2.3.2
            unwanted_columns = [
                feature
                for feature in dataset.features
                if feature not in model_inputs and feature not in ("label_ids", "label")
            ]
            dataset = dataset.remove_columns(unwanted_columns)
            output_signature, _ = dataset._get_output_signature(
                dataset, batch_size=None, collate_fn=collate_fn, collate_fn_args=collate_fn_args
            )
        output_columns = list(output_signature.keys())
        feature_cols = [col for col in output_columns if col in model_inputs and col not in model_labels]
        label_cols = [col for col in output_columns if col in model_labels]

        # Backwards compatibility for older versions of datasets. Previously, if `columns` or `label_cols`
        # were a single element list, the returned element spec would be a single element. Now, passing [feature]
        # will return a dict structure {"feature": feature}, and passing a single string will return a single element.
        feature_cols = feature_cols[0] if len(feature_cols) == 1 else feature_cols
        label_cols = label_cols[0] if len(label_cols) == 1 else label_cols

        if drop_remainder is None:
            drop_remainder = shuffle
        tf_dataset = dataset.to_tf_dataset(
            columns=feature_cols,
            label_cols=label_cols,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_remainder=drop_remainder,
            collate_fn=collate_fn,
            collate_fn_args=collate_fn_args,
            prefetch=prefetch,
        )
        return tf_dataset

    def compile(
        self,
        optimizer="rmsprop",
        loss="auto_with_warning",
        metrics=None,
        loss_weights=None,
        weighted_metrics=None,
        run_eagerly=None,
        steps_per_execution=None,
        **kwargs,
    ):
        """
        This is a thin wrapper that sets the model's loss output head as the loss if the user does not specify a loss
        function themselves.
        """
        if loss in ("auto_with_warning", "passthrough"):  # "passthrough" for workflow backward compatibility
            logger.info(
                "No loss specified in compile() - the model's internal loss computation will be used as the "
                "loss. Don't panic - this is a common way to train TensorFlow models in Transformers! "
                "To disable this behaviour please pass a loss argument, or explicitly pass "
                "`loss=None` if you do not want your model to compute a loss. You can also specify `loss='auto'` to "
                "get the internal loss without printing this info string."
            )
            loss = "auto"
        if loss == "auto":
            loss = dummy_loss
            self._using_dummy_loss = True
        else:
            self._using_dummy_loss = False
        parent_args = list(inspect.signature(keras.Model.compile).parameters.keys())
        # This argument got renamed, we need to support both versions
        if "steps_per_execution" in parent_args:
            super().compile(
                optimizer=optimizer,
                loss=loss,
                metrics=metrics,
                loss_weights=loss_weights,
                weighted_metrics=weighted_metrics,
                run_eagerly=run_eagerly,
                steps_per_execution=steps_per_execution,
                **kwargs,
            )
        else:
            super().compile(
                optimizer=optimizer,
                loss=loss,
                metrics=metrics,
                loss_weights=loss_weights,
                weighted_metrics=weighted_metrics,
                run_eagerly=run_eagerly,
                experimental_steps_per_execution=steps_per_execution,
                **kwargs,
            )

    def compute_loss(self, *args, **kwargs):
        if hasattr(keras.Model, "compute_loss"):
            # This will be true in TF 2.8 or greater
            return super().compute_loss(*args, **kwargs)
        else:
            warnings.warn(
                "The old compute_loss method is deprecated as it conflicts with the Keras compute_loss "
                "method added in TF 2.8. If you want the original HF compute_loss, please call "
                "hf_compute_loss() instead. From TF versions >= 2.8, or Transformers versions >= 5, "
                "calling compute_loss() will get the Keras method instead.",
                FutureWarning,
            )
            return self.hf_compute_loss(*args, **kwargs)

    def get_label_to_output_name_mapping(self):
        arg_names = list(inspect.signature(self.call).parameters)
        if self._label_to_output_map is not None:
            return self._label_to_output_map
        elif "start_positions" in arg_names:
            return {"start_positions": "start_logits", "end_positions": "end_logits"}
        elif "sentence_order_label" in arg_names:
            return {"labels": "prediction_logits", "sentence_order_label": "sop_logits"}
        elif "next_sentence_label" in arg_names:
            return {"labels": "prediction_logits", "next_sentence_label": "seq_relationship_logits"}
        elif "mc_labels" in arg_names:
            return {"labels": "logits", "mc_labels": "mc_logits"}
        else:
            return {}

    def train_step(self, data):
        """
        A modification of Keras's default `train_step` that correctly handles matching outputs to labels for our models
        and supports directly training on the loss output head. In addition, it ensures input keys are copied to the
        labels where appropriate. It will also copy label keys into the input dict when using the dummy loss, to ensure
        that they are available to the model during the forward pass.
        """

        # We hardcode the most common renamings; models with weirder names can set `self._label_to_output_map`
        arg_names = list(inspect.signature(self.call).parameters)
        label_kwargs = find_labels(self.__class__)
        label_to_output = self.get_label_to_output_name_mapping()
        output_to_label = {val: key for key, val in label_to_output.items()}
        if not self._using_dummy_loss and parse(tf.__version__) < parse("2.11.0"):
            # Newer TF train steps leave this out
            data = expand_1d(data)
        x, y, sample_weight = keras.utils.unpack_x_y_sample_weight(data)
        # If the inputs are mutable dictionaries, make a shallow copy of them because we will modify
        # them during input/label pre-processing. This avoids surprising the user by wrecking their data.
        # In addition, modifying mutable Python inputs makes XLA compilation impossible.
        if isinstance(x, dict):
            x = x.copy()
        if isinstance(y, dict):
            y = y.copy()

        # When using a dummy loss, we ensure that separate labels are copied to the correct model arguments,
        # if those keys are not already present in the input dict
        if self._using_dummy_loss and y is not None:
            # If y is a tensor and the model only has one label-like input, map y to that input
            if len(label_kwargs) == 1 and isinstance(y, tf.Tensor):
                if isinstance(x, tf.Tensor):
                    x = {arg_names[0]: x}
                label_kwarg = next(iter(label_kwargs))
                if label_kwarg not in x:
                    x[label_kwarg] = y
            # Otherwise, copy keys from y to x as long as they weren't already present in x
            elif isinstance(y, dict):
                if isinstance(x, tf.Tensor):
                    x = {arg_names[0]: x}
                for key, val in y.items():
                    if key in arg_names and key not in x:
                        x[key] = val
                    elif output_to_label.get(key, None) in arg_names and key not in x:
                        x[output_to_label[key]] = val
        if y is None:
            y = {key: val for key, val in x.items() if key in label_kwargs}
            if not y and not self._using_dummy_loss:
                raise ValueError("Could not find label column(s) in input dict and no separate labels were provided!")

        if isinstance(y, dict):
            # Rename labels at this point to match output heads
            y = {label_to_output.get(key, key): val for key, val in y.items()}

        # Run forward pass.
        with tf.GradientTape() as tape:
            if self._using_dummy_loss and "return_loss" in arg_names:
                y_pred = self(x, training=True, return_loss=True)
            else:
                y_pred = self(x, training=True)
            if self._using_dummy_loss:
                loss = self.compiled_loss(y_pred.loss, y_pred.loss, sample_weight, regularization_losses=self.losses)
            else:
                loss = None

            # This next block matches outputs to label keys. Tensorflow's standard method for doing this
            # can get very confused if any of the keys contain nested values (e.g. lists/tuples of Tensors)
            if isinstance(y, dict) and len(y) == 1:
                if list(y.keys())[0] in y_pred.keys():
                    y_pred = y_pred[list(y.keys())[0]]
                elif list(y_pred.keys())[0] == "loss":
                    y_pred = y_pred[1]
                else:
                    y_pred = y_pred[0]
                _, y = y.popitem()
            elif isinstance(y, dict):
                # If the labels are a dict, match keys from the output by name
                y_pred = {key: val for key, val in y_pred.items() if key in y}
            elif isinstance(y, tuple) or isinstance(y, list):
                # If the labels are a tuple/list, match keys to the output by order, skipping the loss.
                if list(y_pred.keys())[0] == "loss":
                    y_pred = y_pred.to_tuple()[1:]
                else:
                    y_pred = y_pred.to_tuple()
                y_pred = y_pred[: len(y)]  # Remove unused fields in case those cause problems
            else:
                # If the labels are a single tensor, match them to the first non-loss tensor in the output
                if list(y_pred.keys())[0] == "loss":
                    y_pred = y_pred[1]
                else:
                    y_pred = y_pred[0]

            if loss is None:
                loss = self.compiled_loss(y, y_pred, sample_weight, regularization_losses=self.losses)

        # Run backwards pass.
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)

        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        # Collect metrics to return
        return_metrics = {}
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result
        return return_metrics

    def test_step(self, data):
        """
        A modification of Keras's default `train_step` that correctly handles matching outputs to labels for our models
        and supports directly training on the loss output head. In addition, it ensures input keys are copied to the
        labels where appropriate. It will also copy label keys into the input dict when using the dummy loss, to ensure
        that they are available to the model during the forward pass.
        """
        # We hardcode the most common renamings; models with weirder names can set `self._label_to_output_map`
        arg_names = list(inspect.signature(self.call).parameters)
        label_kwargs = find_labels(self.__class__)
        label_to_output = self.get_label_to_output_name_mapping()
        output_to_label = {val: key for key, val in label_to_output.items()}
        if not self._using_dummy_loss and parse(tf.__version__) < parse("2.11.0"):
            # Newer versions leave this out
            data = expand_1d(data)
        x, y, sample_weight = keras.utils.unpack_x_y_sample_weight(data)
        # If the inputs are mutable dictionaries, make a shallow copy of them because we will modify
        # them during input/label pre-processing. This avoids surprising the user by wrecking their data.
        # In addition, modifying mutable Python inputs makes XLA compilation impossible.
        if isinstance(x, dict):
            x = x.copy()
        if isinstance(y, dict):
            y = y.copy()

        # When using a dummy loss, we ensure that separate labels are copied to the correct model arguments,
        # if those keys are not already present in the input dict
        if self._using_dummy_loss and y is not None:
            arg_names = list(inspect.signature(self.call).parameters)
            # If y is a tensor and the model only has one label-like input, map y to that input
            if len(label_kwargs) == 1 and isinstance(y, tf.Tensor):
                if isinstance(x, tf.Tensor):
                    x = {arg_names[0]: x}
                label_kwarg = next(iter(label_kwargs))
                if label_kwarg not in x:
                    x[label_kwarg] = y
            # Otherwise, copy keys from y to x as long as they weren't already present in x
            elif isinstance(y, dict):
                if isinstance(x, tf.Tensor):
                    x = {arg_names[0]: x}
                for key, val in y.items():
                    if key in arg_names and key not in x:
                        x[key] = val
                    elif output_to_label.get(key, None) in arg_names and key not in x:
                        x[output_to_label[key]] = val
        if y is None:
            y = {key: val for key, val in x.items() if key in label_kwargs}
            if not y and not self._using_dummy_loss:
                raise ValueError("Could not find label column(s) in input dict and no separate labels were provided!")

        if isinstance(y, dict):
            # Rename labels at this point to match output heads
            y = {label_to_output.get(key, key): val for key, val in y.items()}

        # Run forward pass.
        if self._using_dummy_loss and "return_loss" in arg_names:
            y_pred = self(x, return_loss=True, training=False)
        else:
            y_pred = self(x, training=False)
        if self._using_dummy_loss:
            loss = self.compiled_loss(y_pred.loss, y_pred.loss, sample_weight, regularization_losses=self.losses)
        else:
            loss = None

        # This next block matches outputs to label keys. Tensorflow's standard method for doing this
        # can get very confused if any of the keys contain nested values (e.g. lists/tuples of Tensors)
        if isinstance(y, dict) and len(y) == 1:
            if list(y.keys())[0] in y_pred.keys():
                y_pred = y_pred[list(y.keys())[0]]
            elif list(y_pred.keys())[0] == "loss":
                y_pred = y_pred[1]
            else:
                y_pred = y_pred[0]
            _, y = y.popitem()
        elif isinstance(y, dict):
            # If the labels are a dict, match keys from the output by name
            y_pred = {key: val for key, val in y_pred.items() if key in y}
        elif isinstance(y, tuple) or isinstance(y, list):
            # If the labels are a tuple/list, match keys to the output by order, skipping the loss.
            if list(y_pred.keys())[0] == "loss":
                y_pred = y_pred.to_tuple()[1:]
            else:
                y_pred = y_pred.to_tuple()
            y_pred = y_pred[: len(y)]  # Remove unused fields in case those cause problems
        else:
            # If the labels are a single tensor, match them to the first non-loss tensor in the output
            if list(y_pred.keys())[0] == "loss":
                y_pred = y_pred[1]
            else:
                y_pred = y_pred[0]

        if loss is None:
            loss = self.compiled_loss(y, y_pred, sample_weight, regularization_losses=self.losses)

        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        # Collect metrics to return
        return_metrics = {}
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result
        return return_metrics

    def create_model_card(
        self,
        output_dir,
        model_name: str,
        language: Optional[str] = None,
        license: Optional[str] = None,
        tags: Optional[str] = None,
        finetuned_from: Optional[str] = None,
        tasks: Optional[str] = None,
        dataset_tags: Optional[Union[str, List[str]]] = None,
        dataset: Optional[Union[str, List[str]]] = None,
        dataset_args: Optional[Union[str, List[str]]] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            output_dir (`str` or `os.PathLike`):
                The folder in which to create the model card.
            model_name (`str`, *optional*):
                The name of the model.
            language (`str`, *optional*):
                The language of the model (if applicable)
            license (`str`, *optional*):
                The license of the model. Will default to the license of the pretrained model used, if the original
                model given to the `Trainer` comes from a repo on the Hub.
            tags (`str` or `List[str]`, *optional*):
                Some tags to be included in the metadata of the model card.
            finetuned_from (`str`, *optional*):
                The name of the model used to fine-tune this one (if applicable). Will default to the name of the repo
                of the original model given to the `Trainer` (if it comes from the Hub).
            tasks (`str` or `List[str]`, *optional*):
                One or several task identifiers, to be included in the metadata of the model card.
            dataset_tags (`str` or `List[str]`, *optional*):
                One or several dataset tags, to be included in the metadata of the model card.
            dataset (`str` or `List[str]`, *optional*):
                One or several dataset identifiers, to be included in the metadata of the model card.
            dataset_args (`str` or `List[str]`, *optional*):
               One or several dataset arguments, to be included in the metadata of the model card.
        """
        # Avoids a circular import by doing this when necessary.
        from .modelcard import TrainingSummary  # tests_ignore

        training_summary = TrainingSummary.from_keras(
            self,
            keras_history=self.history,
            language=language,
            license=license,
            tags=tags,
            model_name=model_name,
            finetuned_from=finetuned_from,
            tasks=tasks,
            dataset_tags=dataset_tags,
            dataset=dataset,
            dataset_args=dataset_args,
        )
        model_card = training_summary.to_model_card()
        with open(os.path.join(output_dir, "README.md"), "w") as f:
            f.write(model_card)

    def set_input_embeddings(self, value):
        """
        Set model's input embeddings

        Args:
            value (`tf.Variable`):
                The new weights mapping hidden states to vocabulary.
        """
        main_layer = getattr(self, self.base_model_prefix)

        if main_layer is None:
            raise NotImplementedError("The model does not implements the base_model_prefix attribute.")

        try:
            main_layer.set_input_embeddings(value)
        except AttributeError:
            logger.info("Building the model")
            self.build_in_name_scope()
            main_layer.set_input_embeddings(value)

    def get_output_embeddings(self) -> Union[None, keras.layers.Layer]:
        """
        Returns the model's output embeddings

        Returns:
            `tf.Variable`: The new weights mapping vocabulary to hidden states.
        """
        if self.get_lm_head() is not None:
            lm_head = self.get_lm_head()

            try:
                return lm_head.get_output_embeddings()
            except AttributeError:
                logger.info("Building the model")
                self.build_in_name_scope()

                return lm_head().get_output_embeddings()

        return None  # Overwrite for models with output embeddings

    def set_output_embeddings(self, value):
        """
        Set model's output embeddings

        Args:
            value (`tf.Variable`):
                The new weights mapping hidden states to vocabulary.
        """
        if self.get_lm_head() is not None:
            lm_head = self.get_lm_head()
            try:
                lm_head.set_output_embeddings(value)
            except AttributeError:
                logger.info("Building the model")
                self.build_in_name_scope()
                lm_head.set_output_embeddings(value)

    def get_output_layer_with_bias(self) -> Union[None, keras.layers.Layer]:
        """
        Get the layer that handles a bias attribute in case the model has an LM head with weights tied to the
        embeddings

        Return:
            `keras.layers.Layer`: The layer that handles the bias, None if not an LM model.
        """
        warnings.warn(
            "The method get_output_layer_with_bias is deprecated. Please use `get_lm_head` instead.", FutureWarning
        )
        return self.get_lm_head()

    def get_prefix_bias_name(self) -> Union[None, str]:
        """
        Get the concatenated _prefix name of the bias from the model name to the parent layer

        Return:
            `str`: The _prefix name of the bias.
        """
        warnings.warn("The method get_prefix_bias_name is deprecated. Please use `get_bias` instead.", FutureWarning)
        return None

    def get_bias(self) -> Union[None, Dict[str, tf.Variable]]:
        """
        Dict of bias attached to an LM head. The key represents the name of the bias attribute.

        Return:
            `tf.Variable`: The weights representing the bias, None if not an LM model.
        """
        if self.get_lm_head() is not None:
            lm_head = self.get_lm_head()
            try:
                return lm_head.get_bias()
            except AttributeError:
                self.build_in_name_scope()

                return lm_head.get_bias()
        return None

    def set_bias(self, value):
        """
        Set all the bias in the LM head.

        Args:
            value (`Dict[tf.Variable]`):
                All the new bias attached to an LM head.
        """
        if self.get_lm_head() is not None:
            lm_head = self.get_lm_head()
            try:
                lm_head.set_bias(value)
            except AttributeError:
                self.build_in_name_scope()
                lm_head.set_bias(value)

    def get_lm_head(self) -> keras.layers.Layer:
        """
        The LM Head layer. This method must be overwritten by all the models that have a lm head.

        Return:
            `keras.layers.Layer`: The LM head layer if the model has one, None if not.
        """
        return None

    def resize_token_embeddings(
        self, new_num_tokens: Optional[int] = None
    ) -> Union[keras.layers.Embedding, tf.Variable]:
        """
        Resizes input token embeddings matrix of the model if `new_num_tokens != config.vocab_size`.

        Takes care of tying weights embeddings afterwards if the model class has a `tie_weights()` method.

        Arguments:
            new_num_tokens (`int`, *optional*):
                The number of new tokens in the embedding matrix. Increasing the size will add newly initialized
                vectors at the end. Reducing the size will remove vectors from the end. If not provided or `None`, just
                returns a pointer to the input tokens without doing anything.

        Return:
            `tf.Variable` or `keras.layers.Embedding`: Pointer to the input tokens of the model.
        """
        # TODO (joao): flagged for replacement (by `_v2_resized_token_embeddings`) due to embeddings refactor

        # Run the new code path if the model has a keras embeddings layer
        if isinstance(self.get_input_embeddings(), keras.layers.Embedding):
            return self._v2_resized_token_embeddings(new_num_tokens)

        if new_num_tokens is None or new_num_tokens == self.config.vocab_size:
            return self._get_word_embedding_weight(self.get_input_embeddings())

        model_embeds = self._resize_token_embeddings(new_num_tokens)

        # Update base model and current model config
        self.config.vocab_size = new_num_tokens

        return model_embeds

    def _v2_resized_token_embeddings(self, new_num_tokens: Optional[int] = None) -> keras.layers.Embedding:
        """
        Resizes input token embeddings matrix of the model if `new_num_tokens != config.vocab_size`.

        Arguments:
            new_num_tokens (`int`, *optional*):
                The number of new tokens in the embedding matrix. Increasing the size will add newly initialized
                vectors at the end. Reducing the size will remove vectors from the end. If not provided or `None`, just
                returns a pointer to the input tokens without doing anything.

        Return:
            `keras.layers.Embedding`: Pointer to the input tokens of the model.
        """
        if new_num_tokens is None or new_num_tokens == self.config.vocab_size:
            return self.get_input_embeddings()

        model_embeds = self._v2_resize_token_embeddings(new_num_tokens)

        # Update base model and current model config
        self.config.vocab_size = new_num_tokens

        return model_embeds

    def _get_word_embedding_weight(model, embedding_layer):
        # TODO (joao): flagged for delection due to embeddings refactor

        # If the variable holds the weights themselves, return them
        if isinstance(embedding_layer, tf.Tensor):
            return embedding_layer
        # Otherwise, try to get them from the layer's attributes

        embeds = getattr(embedding_layer, "weight", None)
        if embeds is not None:
            return embeds

        embeds = getattr(embedding_layer, "decoder", None)
        if embeds is not None:
            return embeds

        # The reason why the attributes don't exist might be
        # because the model is not built, so retry getting
        # the argument after building the model
        model.build_in_name_scope()

        embeds = getattr(embedding_layer, "weight", None)
        if embeds is not None:
            return embeds

        embeds = getattr(embedding_layer, "decoder", None)
        if embeds is not None:
            return embeds

        return None

    def _resize_token_embeddings(self, new_num_tokens):
        # TODO (joao): flagged for replacement (by `_v2_resize_token_embeddings`) due to embeddings refactor
        old_embeddings = self._get_word_embedding_weight(self.get_input_embeddings())
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)

        # if word embeddings are not tied, make sure that lm head bias is resized as well
        if self.get_bias() is not None:
            old_lm_head_bias = self.get_bias()
            new_lm_head_bias = self._get_resized_lm_head_bias(old_lm_head_bias, new_num_tokens)

            self.set_bias(new_lm_head_bias)

        # if word embeddings are not tied, make sure that lm head decoder is resized as well
        if self.get_output_embeddings() is not None:
            old_lm_head_decoder = self._get_word_embedding_weight(self.get_output_embeddings())
            new_lm_head_decoder = self._get_resized_lm_head_decoder(old_lm_head_decoder, new_num_tokens)

            self.set_output_embeddings(new_lm_head_decoder)

        self.set_input_embeddings(new_embeddings)

        return self.get_input_embeddings()

    def _v2_resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.get_input_embeddings()
        new_embeddings = self._v2_get_resized_embeddings(old_embeddings, new_num_tokens)
        self.set_input_embeddings(new_embeddings)

        # If word embeddings are not tied, make sure that lm head bias is resized as well
        if self.get_bias() is not None:
            old_lm_head_bias = self.get_bias()
            new_lm_head_bias = self._v2_get_resized_lm_head_bias(old_lm_head_bias, new_num_tokens)
            self.set_bias(new_lm_head_bias)

        # If word embeddings are not tied, make sure that lm head decoder is resized as well.
        tied_weights = self.get_input_embeddings() == self.get_output_embeddings()
        if self.get_output_embeddings() is not None and not tied_weights:
            old_lm_head_decoder = self._get_word_embedding_weight(self.get_output_embeddings())
            # TODO (joao): this one probably needs a v2 version with other models
            new_lm_head_decoder = self._get_resized_lm_head_decoder(old_lm_head_decoder, new_num_tokens)
            self.set_output_embeddings(new_lm_head_decoder)

        return self.get_input_embeddings()

    def _get_resized_lm_head_bias(self, old_lm_head_bias, new_num_tokens):
        """
        Build a resized bias from the old ones. Increasing the size will add newly initialized vectors at the end.
        Reducing the size will remove vectors from the end

        Args:
            old_lm_head_bias (`tf.Variable`):
                Old lm head bias to be resized.
            new_num_tokens (`int`, *optional*):
                New number of tokens in the linear matrix.

                Increasing the size will add newly initialized vectors at the end. Reducing the size will remove
                vectors from the end. If not provided or `None`, just returns None

        Return:
            `tf.Variable`: Pointer to the resized bias.
        """
        # TODO (joao): flagged for replacement (by `_v2_get_resized_lm_head_bias`) due to embeddings refactor
        new_lm_head_bias = {}

        for attr, weight in old_lm_head_bias.items():
            first_dim, old_num_tokens = (None, shape_list(weight)[0]) if tf.rank(weight) == 1 else shape_list(weight)
            size_diff = new_num_tokens - old_num_tokens
            final_shape = [new_num_tokens] if first_dim is None else [first_dim, new_num_tokens]

            # initialize new bias
            if tf.math.greater(size_diff, 0):
                padding_shape = [[0, size_diff]] if first_dim is None else [[0, 0], [0, size_diff]]
                current_bias = tf.pad(weight.value(), tf.convert_to_tensor(padding_shape), constant_values=-1)
                num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
                mask_shape = [num_tokens_to_copy] if first_dim is None else [1, num_tokens_to_copy]
                bias_mask = tf.fill(tf.convert_to_tensor(mask_shape), True)
                bias_mask = tf.pad(bias_mask, tf.convert_to_tensor(padding_shape), constant_values=False)
            else:
                slice_from = [0] if first_dim is None else [0, 0]
                current_bias = tf.slice(
                    weight.value(), tf.convert_to_tensor(slice_from), tf.convert_to_tensor(final_shape)
                )
                bias_mask = tf.fill(tf.convert_to_tensor(final_shape), True)

            new_bias = self.add_weight(
                shape=final_shape,
                initializer="zeros",
                trainable=True,
                name=weight.name.split(":")[0],
            )
            init_bias = tf.where(bias_mask, current_bias, new_bias.value())

            new_bias.assign(init_bias)
            new_lm_head_bias[attr] = new_bias

        return new_lm_head_bias

    def _v2_get_resized_lm_head_bias(
        self, old_lm_head_bias: Dict[str, tf.Variable], new_num_tokens: int
    ) -> Dict[str, tf.Tensor]:
        """
        Build a resized bias from the old ones. Increasing the size will add newly initialized vectors at the end.
        Reducing the size will remove vectors from the end

        Args:
            old_lm_head_bias (`Dict[str, tf.Variable]`):
                Old lm head bias to be resized.
            new_num_tokens (`int`):
                New number of tokens in the linear matrix. Increasing the size will add newly initialized vectors at
                the end. Reducing the size will remove vectors from the end.

        Return:
            `tf.Tensor`: Values for the resized bias.
        """
        new_lm_head_bias = {}

        for attr, weight in old_lm_head_bias.items():
            # Determine the size difference (depending on the shape)
            first_dim, old_num_tokens = (None, shape_list(weight)[0]) if tf.rank(weight) == 1 else shape_list(weight)
            size_diff = new_num_tokens - old_num_tokens

            # Copy the old bias values to the new bias
            if old_num_tokens > new_num_tokens:
                new_bias = weight.value()[..., :new_num_tokens]
            else:
                padding_shape = [[0, size_diff]] if first_dim is None else [[0, 0], [0, size_diff]]
                new_bias = tf.pad(weight.value(), tf.convert_to_tensor(padding_shape))

            new_lm_head_bias[attr] = new_bias
        return new_lm_head_bias

    def _get_resized_lm_head_decoder(self, old_lm_head_decoder, new_num_tokens):
        """
        Build a resized decoder from the old ones. Increasing the size will add newly initialized vectors at the end.
        Reducing the size will remove vectors from the end

        Args:
            old_lm_head_decoder (`tf.Variable`):
                Old lm head decoder to be resized.
            new_num_tokens (`int`, *optional*):
                New number of tokens in the linear matrix.

                Increasing the size will add newly initialized vectors at the end. Reducing the size will remove
                vectors from the end. If not provided or `None`, just returns None

        Return:
            `tf.Variable`: Pointer to the resized decoder or None if the output embeddings are different from the input
            ones.
        """
        new_lm_head_decoder = old_lm_head_decoder
        is_input_output_equals = tf.reduce_any(
            self._get_word_embedding_weight(self.get_input_embeddings()) == old_lm_head_decoder
        )

        if old_lm_head_decoder is not None and not is_input_output_equals:
            old_embedding_dim = shape_list(old_lm_head_decoder)[1]
            decoder_mask, current_decoder = init_copy_embeddings(old_lm_head_decoder, new_num_tokens)
            new_lm_head_decoder = self.add_weight(
                shape=(new_num_tokens, old_embedding_dim),
                initializer="zeros",
                trainable=True,
                name=old_lm_head_decoder.name.split(":")[0],
            )
            init_decoder = tf.where(decoder_mask, current_decoder, new_lm_head_decoder.value())

            new_lm_head_decoder.assign(init_decoder)

        return new_lm_head_decoder

    def _get_resized_embeddings(self, old_embeddings, new_num_tokens=None) -> tf.Variable:
        """
        Build a resized Embedding weights from a provided token Embedding weights. Increasing the size will add newly
        initialized vectors at the end. Reducing the size will remove vectors from the end

        Args:
            old_embeddings (`tf.Variable`):
                Old embeddings to be resized.
            new_num_tokens (`int`, *optional*):
                New number of tokens in the embedding matrix.

                Increasing the size will add newly initialized vectors at the end. Reducing the size will remove
                vectors from the end. If not provided or `None`, just returns a pointer to the input tokens
                `tf.Variable` module of the model without doing anything.

        Return:
            `tf.Variable`: Pointer to the resized Embedding Module or the old Embedding Module if `new_num_tokens` is
            `None`
        """
        # TODO (joao): flagged for replacement (by `_v2_get_resized_embeddings`) due to embeddings refactor
        old_embedding_dim = shape_list(old_embeddings)[1]
        init_range = getattr(self.config, "initializer_range", 0.02)
        embeddings_mask, current_embeddings = init_copy_embeddings(old_embeddings, new_num_tokens)
        new_embeddings = self.add_weight(
            name=old_embeddings.name.split(":")[0],
            shape=[new_num_tokens, old_embedding_dim],
            initializer=get_initializer(init_range),
            dtype=tf.float32,
        )
        init_embeddings = tf.where(embeddings_mask, current_embeddings, new_embeddings.value())

        new_embeddings.assign(init_embeddings)

        return new_embeddings

    def _v2_get_resized_embeddings(
        self, old_embeddings: keras.layers.Embedding, new_num_tokens: int
    ) -> keras.layers.Embedding:
        """
        Build a resized Embedding layer from a provided Embedding layer. Increasing the size will add newly initialized
        vectors at the end. Reducing the size will remove vectors from the end.

        Args:
            old_embeddings (`keras.layers.Embedding`):
                Old embeddings to be resized.
            new_num_tokens (`int`, *optional*):
                New number of tokens in the embedding matrix.

        Return:
            `keras.layers.Embedding`: Resized Embedding layer.
        """

        # Get the initialization range for the embeddings
        init_range = 0.02  # default value
        potential_initialization_variable_names = [
            "initializer_range",  # most common
            "initializer_factor",  # e.g. T5
            "init_std",  # e.g BART
        ]
        for var_name in potential_initialization_variable_names:
            if hasattr(self.config, var_name):
                init_range = getattr(self.config, var_name)

        # Get a new (initialized) embeddings layer
        new_embeddings = keras.layers.Embedding(
            input_dim=new_num_tokens,
            output_dim=old_embeddings.output_dim,
            embeddings_initializer=keras.initializers.TruncatedNormal(stddev=init_range),
            name=old_embeddings.embeddings.name[:-13],  # exact same scoped name except "/embeddings:0"
        )
        new_embeddings(tf.constant([[0]]))

        # Copy the old embeddings to the new embeddings
        if old_embeddings.input_dim >= new_num_tokens:
            init_embeddings = old_embeddings.embeddings[:new_num_tokens]
        else:
            init_embeddings = tf.concat(
                [old_embeddings.embeddings, new_embeddings.embeddings[old_embeddings.input_dim :]], axis=0
            )
        new_embeddings.embeddings.assign(init_embeddings)
        return new_embeddings

    def prune_heads(self, heads_to_prune):
        """
        Prunes heads of the base model.

        Arguments:
            heads_to_prune (`Dict[int, List[int]]`):
                Dictionary with keys being selected layer indices (`int`) and associated values being the list of heads
                to prune in said layer (list of `int`). For instance {1: [0, 2], 2: [2, 3]} will prune heads 0 and 2 on
                layer 1 and heads 2 and 3 on layer 2.
        """
        raise NotImplementedError

    def save_pretrained(
        self,
        save_directory,
        saved_model=False,
        version=1,
        push_to_hub=False,
        signatures=None,
        max_shard_size: Union[int, str] = "5GB",
        create_pr: bool = False,
        safe_serialization: bool = False,
        token: Optional[Union[str, bool]] = None,
        **kwargs,
    ):
        """
        Save a model and its configuration file to a directory, so that it can be re-loaded using the
        [`~TFPreTrainedModel.from_pretrained`] class method.

        Arguments:
            save_directory (`str`):
                Directory to which to save. Will be created if it doesn't exist.
            saved_model (`bool`, *optional*, defaults to `False`):
                If the model has to be saved in saved model format as well or not.
            version (`int`, *optional*, defaults to 1):
                The version of the saved model. A saved model needs to be versioned in order to be properly loaded by
                TensorFlow Serving as detailed in the official documentation
                https://www.tensorflow.org/tfx/serving/serving_basic
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            signatures (`dict` or `tf.function`, *optional*):
                Model's signature used for serving. This will be passed to the `signatures` argument of model.save().
            max_shard_size (`int` or `str`, *optional*, defaults to `"10GB"`):
                The maximum size for a checkpoint before being sharded. Checkpoints shard will then be each of size
                lower than this size. If expressed as a string, needs to be digits followed by a unit (like `"5MB"`).

                <Tip warning={true}>

                If a single weight of the model is bigger than `max_shard_size`, it will be in its own checkpoint shard
                which will be bigger than `max_shard_size`.

                </Tip>

            create_pr (`bool`, *optional*, defaults to `False`):
                Whether or not to create a PR with the uploaded files or directly commit.
            safe_serialization (`bool`, *optional*, defaults to `False`):
                Whether to save the model using `safetensors` or the traditional TensorFlow way (that uses `h5`).
            token (`str` or `bool`, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, or not specified, will use
                the token generated when running `huggingface-cli login` (stored in `~/.huggingface`).
            kwargs (`Dict[str, Any]`, *optional*):
                Additional key word arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """
        use_auth_token = kwargs.pop("use_auth_token", None)

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

        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        os.makedirs(save_directory, exist_ok=True)

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
            repo_id = self._create_repo(repo_id, **kwargs)
            files_timestamps = self._get_files_timestamps(save_directory)

        if saved_model:
            # If `torch_dtype` is in the config with a torch dtype class as the value, we need to change it to string.
            # (Although TF doesn't care about this attribute, we can't just remove it or set it to `None`.)
            if getattr(self.config, "torch_dtype", None) is not None and not isinstance(self.config.torch_dtype, str):
                self.config.torch_dtype = str(self.config.torch_dtype).split(".")[1]
            if signatures is None:
                serving_default = self.serving.get_concrete_function(self.input_signature)
                if any(spec.dtype == tf.int32 for spec in self.input_signature.values()):
                    int64_spec = {
                        key: tf.TensorSpec(
                            shape=spec.shape, dtype=tf.int64 if spec.dtype == tf.int32 else spec.dtype, name=spec.name
                        )
                        for key, spec in self.input_signature.items()
                    }
                    int64_serving = self.serving.get_concrete_function(int64_spec)
                    signatures = {"serving_default": serving_default, "int64_serving": int64_serving}
                else:
                    signatures = serving_default
            saved_model_dir = os.path.join(save_directory, "saved_model", str(version))
            self.save(saved_model_dir, include_optimizer=False, signatures=signatures)
            logger.info(f"Saved model created in {saved_model_dir}")

        # Save configuration file
        self.config.architectures = [self.__class__.__name__[2:]]

        # If we have a custom model, we copy the file defining it in the folder and set the attributes so it can be
        # loaded from the Hub.
        if self._auto_class is not None:
            custom_object_save(self, save_directory, config=self.config)

        self.config.save_pretrained(save_directory)
        if self.can_generate():
            self.generation_config.save_pretrained(save_directory)

        # If we save using the predefined names, we can load using `from_pretrained`
        weights_name = SAFE_WEIGHTS_NAME if safe_serialization else TF2_WEIGHTS_NAME
        output_model_file = os.path.join(save_directory, weights_name)

        shards, index = tf_shard_checkpoint(self.weights, max_shard_size, weights_name=weights_name)

        # Clean the folder from a previous save
        for filename in os.listdir(save_directory):
            full_filename = os.path.join(save_directory, filename)
            # If we have a shard file that is not going to be replaced, we delete it, but only from the main process
            # in distributed settings to avoid race conditions.
            weights_no_suffix = weights_name.replace(".bin", "").replace(".safetensors", "")
            if (
                filename.startswith(weights_no_suffix)
                and os.path.isfile(full_filename)
                and filename not in shards.keys()
            ):
                os.remove(full_filename)

        if index is None:
            if safe_serialization:
                state_dict = {strip_model_name_and_prefix(w.name): w.value() for w in self.weights}
                safe_save_file(state_dict, output_model_file, metadata={"format": "tf"})
            else:
                self.save_weights(output_model_file)
            logger.info(f"Model weights saved in {output_model_file}")
        else:
            save_index_file = SAFE_WEIGHTS_INDEX_NAME if safe_serialization else TF2_WEIGHTS_INDEX_NAME
            save_index_file = os.path.join(save_directory, save_index_file)
            # Save the index as well
            with open(save_index_file, "w", encoding="utf-8") as index_file:
                content = json.dumps(index, indent=2, sort_keys=True) + "\n"
                index_file.write(content)
            logger.info(
                f"The model is bigger than the maximum size per checkpoint ({max_shard_size}) and is going to be "
                f"split in {len(shards)} checkpoint shards. You can find where each parameters has been saved in the "
                f"index located at {save_index_file}."
            )
            for shard_file, shard in shards.items():
                if safe_serialization:
                    shard_state_dict = {strip_model_name_and_prefix(w.name): w.value() for w in shard}
                    safe_save_file(
                        shard_state_dict, os.path.join(save_directory, shard_file), metadata={"format": "tf"}
                    )
                else:
                    with h5py.File(os.path.join(save_directory, shard_file), mode="w") as shard_file:
                        layers = []
                        for layer in sorted(shard, key=lambda x: x.name):
                            if "model." in layer.name or len(layer.name.split("/")) == 1:
                                layer_name = layer.name
                            else:
                                layer_name = "/".join(layer.name.split("/")[1:])
                            param_dset = shard_file.create_dataset(
                                layer_name, layer.numpy().shape, dtype=layer.numpy().dtype
                            )
                            param_dset[:] = layer.numpy()
                            layers.append(layer_name.encode("utf8"))
                        save_attributes_to_hdf5_group(shard_file, "layer_names", layers)

        if push_to_hub:
            self._upload_modified_files(
                save_directory,
                repo_id,
                files_timestamps,
                commit_message=commit_message,
                token=token,
            )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        use_safetensors: bool = None,
        **kwargs,
    ):
        r"""
        Instantiate a pretrained TF 2.0 model from a pre-trained model configuration.

        The warning *Weights from XXX not initialized from pretrained model* means that the weights of XXX do not come
        pretrained with the rest of the model. It is up to you to train those weights with a downstream fine-tuning
        task.

        The warning *Weights from XXX not used in YYY* means that the layer XXX is not used by YYY, therefore those
        weights are discarded.

        Parameters:
            pretrained_model_name_or_path (`str`, *optional*):
                Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                    - A path to a *directory* containing model weights saved using
                      [`~TFPreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
                    - A path or url to a *PyTorch state_dict save file* (e.g, `./pt_model/pytorch_model.bin`). In this
                      case, `from_pt` should be set to `True` and a configuration object should be provided as `config`
                      argument. This loading path is slower than converting the PyTorch model in a TensorFlow model
                      using the provided conversion scripts and loading the TensorFlow model afterwards.
                    - `None` if you are both providing the configuration and state dictionary (resp. with keyword
                      arguments `config` and `state_dict`).
            model_args (sequence of positional arguments, *optional*):
                All remaining positional arguments will be passed to the underlying model's `__init__` method.
            config (`Union[PretrainedConfig, str]`, *optional*):
                Can be either:

                    - an instance of a class derived from [`PretrainedConfig`],
                    - a string valid as input to [`~PretrainedConfig.from_pretrained`].

                Configuration for the model to use instead of an automatically loaded configuration. Configuration can
                be automatically loaded when:

                    - The model is a model provided by the library (loaded with the *model id* string of a pretrained
                      model).
                    - The model was saved using [`~TFPreTrainedModel.save_pretrained`] and is reloaded by supplying the
                      save directory.
                    - The model is loaded by supplying a local directory as `pretrained_model_name_or_path` and a
                      configuration JSON file named *config.json* is found in the directory.
            from_pt (`bool`, *optional*, defaults to `False`):
                Load the model weights from a PyTorch state_dict save file (see docstring of
                `pretrained_model_name_or_path` argument).
            ignore_mismatched_sizes (`bool`, *optional*, defaults to `False`):
                Whether or not to raise an error if some of the weights from the checkpoint do not have the same size
                as the weights of the model (if for instance, you are instantiating a model with 10 labels from a
                checkpoint with 3 labels).
            cache_dir (`str`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            resume_download:
                Deprecated and ignored. All downloads are now resumed by default when possible.
                Will be removed in v5 of Transformers.
            proxies:
                (`Dict[str, str], `optional`): A dictionary of proxy servers to use by protocol or endpoint, e.g.,
                `{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
                output_loading_info(`bool`, *optional*, defaults to `False`): Whether ot not to also return a
                dictionary containing missing keys, unexpected keys and error messages.
            local_files_only(`bool`, *optional*, defaults to `False`):
                Whether or not to only look at local files (e.g., not try downloading the model).
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
            subfolder (`str`, *optional*, defaults to `""`):
                In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can
                specify the folder name here.
            tf_to_pt_weight_rename (`Callable`, *optional*):
                A function that is called to transform the names of weights during the PyTorch to TensorFlow
                crossloading process. This is not necessary for most models, but is useful to allow composite models to
                be crossloaded correctly.
            use_safetensors (`bool`, *optional*, defaults to `None`):
                Whether or not to use `safetensors` checkpoints. Defaults to `None`. If not specified and `safetensors`
                is not installed, it will be set to `False`.
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

        Examples:

        ```python
        >>> from transformers import BertConfig, TFBertModel

        >>> # Download model and configuration from huggingface.co and cache.
        >>> model = TFBertModel.from_pretrained("google-bert/bert-base-uncased")
        >>> # Model was saved using *save_pretrained('./test/saved_model/')* (for example purposes, not runnable).
        >>> model = TFBertModel.from_pretrained("./test/saved_model/")
        >>> # Update configuration during loading.
        >>> model = TFBertModel.from_pretrained("google-bert/bert-base-uncased", output_attentions=True)
        >>> assert model.config.output_attentions == True
        >>> # Loading from a Pytorch model file instead of a TensorFlow checkpoint (slower, for example purposes, not runnable).
        >>> config = BertConfig.from_json_file("./pt_model/my_pt_model_config.json")
        >>> model = TFBertModel.from_pretrained("./pt_model/my_pytorch_model.bin", from_pt=True, config=config)
        ```"""
        from_pt = kwargs.pop("from_pt", False)
        resume_download = kwargs.pop("resume_download", None)
        proxies = kwargs.pop("proxies", None)
        output_loading_info = kwargs.pop("output_loading_info", False)
        use_auth_token = kwargs.pop("use_auth_token", None)
        trust_remote_code = kwargs.pop("trust_remote_code", None)
        _ = kwargs.pop("mirror", None)
        load_weight_prefix = kwargs.pop("load_weight_prefix", None)
        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)
        subfolder = kwargs.pop("subfolder", "")
        commit_hash = kwargs.pop("_commit_hash", None)
        tf_to_pt_weight_rename = kwargs.pop("tf_to_pt_weight_rename", None)

        # Not relevant for TF models
        _ = kwargs.pop("adapter_kwargs", None)

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

        if trust_remote_code is True:
            logger.warning(
                "The argument `trust_remote_code` is to be used with Auto classes. It has no effect here and is"
                " ignored."
            )

        user_agent = {"file_type": "model", "framework": "tensorflow", "from_auto_class": from_auto_class}
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline

        if is_offline_mode() and not local_files_only:
            logger.info("Offline mode: forcing local_files_only=True")
            local_files_only = True

        if use_safetensors is None and not is_safetensors_available():
            use_safetensors = False

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
                _from_auto=from_auto_class,
                _from_pipeline=from_pipeline,
                _commit_hash=commit_hash,
                **kwargs,
            )
        else:
            model_kwargs = kwargs

        if commit_hash is None:
            commit_hash = getattr(config, "_commit_hash", None)

        # This variable will flag if we're loading a sharded checkpoint. In this case the archive file is just the
        # index of the files.
        is_sharded = False
        # Load model
        if pretrained_model_name_or_path is not None:
            pretrained_model_name_or_path = str(pretrained_model_name_or_path)
            is_local = os.path.isdir(pretrained_model_name_or_path)
            if is_local:
                if from_pt and os.path.isfile(os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)):
                    # Load from a PyTorch checkpoint in priority if from_pt
                    archive_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
                elif from_pt and os.path.isfile(os.path.join(pretrained_model_name_or_path, WEIGHTS_INDEX_NAME)):
                    # Load from a sharded PyTorch checkpoint
                    archive_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_INDEX_NAME)
                    is_sharded = True
                elif use_safetensors is not False and os.path.isfile(
                    os.path.join(pretrained_model_name_or_path, SAFE_WEIGHTS_NAME)
                ):
                    # Load from a safetensors checkpoint
                    archive_file = os.path.join(pretrained_model_name_or_path, SAFE_WEIGHTS_NAME)
                elif use_safetensors is not False and os.path.isfile(
                    os.path.join(pretrained_model_name_or_path, SAFE_WEIGHTS_INDEX_NAME)
                ):
                    # Load from a sharded safetensors checkpoint
                    archive_file = os.path.join(pretrained_model_name_or_path, SAFE_WEIGHTS_INDEX_NAME)
                    is_sharded = True
                elif os.path.isfile(os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_NAME)):
                    # Load from a TF 2.0 checkpoint
                    archive_file = os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_NAME)
                elif os.path.isfile(os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_INDEX_NAME)):
                    # Load from a sharded TF 2.0 checkpoint
                    archive_file = os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_INDEX_NAME)
                    is_sharded = True

                # At this stage we don't have a weight file so we will raise an error.
                elif use_safetensors:
                    raise EnvironmentError(
                        f"Error no file named {SAFE_WEIGHTS_NAME} or {SAFE_WEIGHTS_INDEX_NAME} found in directory {pretrained_model_name_or_path}. "
                        f"Please make sure that the model has been saved with `safe_serialization=True` or do not "
                        f"set `use_safetensors=True`."
                    )
                elif os.path.isfile(os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)) or os.path.isfile(
                    os.path.join(pretrained_model_name_or_path, WEIGHTS_INDEX_NAME)
                ):
                    raise EnvironmentError(
                        f"Error no file named {TF2_WEIGHTS_NAME} or {SAFE_WEIGHTS_NAME} found in directory {pretrained_model_name_or_path} "
                        "but there is a file for PyTorch weights. Use `from_pt=True` to load this model from those "
                        "weights."
                    )
                else:
                    raise EnvironmentError(
                        f"Error no file named {TF2_WEIGHTS_NAME}, {SAFE_WEIGHTS_NAME} or {WEIGHTS_NAME} found in directory "
                        f"{pretrained_model_name_or_path}."
                    )
            elif os.path.isfile(pretrained_model_name_or_path):
                archive_file = pretrained_model_name_or_path
                is_local = True
            elif os.path.isfile(pretrained_model_name_or_path + ".index"):
                archive_file = pretrained_model_name_or_path + ".index"
                is_local = True
            elif is_remote_url(pretrained_model_name_or_path):
                filename = pretrained_model_name_or_path
                resolved_archive_file = download_url(pretrained_model_name_or_path)
            else:
                # set correct filename
                if from_pt:
                    filename = WEIGHTS_NAME
                elif use_safetensors is not False:
                    filename = SAFE_WEIGHTS_NAME
                else:
                    filename = TF2_WEIGHTS_NAME

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
                    if resolved_archive_file is None and filename == SAFE_WEIGHTS_NAME:
                        # Did not find the safetensors file, let's fallback to TF.
                        # No support for sharded safetensors yet, so we'll raise an error if that's all we find.
                        filename = TF2_WEIGHTS_NAME
                        resolved_archive_file = cached_file(
                            pretrained_model_name_or_path, TF2_WEIGHTS_NAME, **cached_file_kwargs
                        )
                    if resolved_archive_file is None and filename == TF2_WEIGHTS_NAME:
                        # Maybe the checkpoint is sharded, we try to grab the index name in this case.
                        resolved_archive_file = cached_file(
                            pretrained_model_name_or_path, TF2_WEIGHTS_INDEX_NAME, **cached_file_kwargs
                        )
                        if resolved_archive_file is not None:
                            is_sharded = True
                    if resolved_archive_file is None and filename == WEIGHTS_NAME:
                        # Maybe the checkpoint is sharded, we try to grab the index name in this case.
                        resolved_archive_file = cached_file(
                            pretrained_model_name_or_path, WEIGHTS_INDEX_NAME, **cached_file_kwargs
                        )
                        if resolved_archive_file is not None:
                            is_sharded = True
                    if resolved_archive_file is None:
                        # Otherwise, maybe there is a PyTorch or Flax model file.  We try those to give a helpful error
                        # message.
                        has_file_kwargs = {
                            "revision": revision,
                            "proxies": proxies,
                            "token": token,
                            "cache_dir": cache_dir,
                            "local_files_only": local_files_only,
                        }
                        if has_file(pretrained_model_name_or_path, SAFE_WEIGHTS_INDEX_NAME, **has_file_kwargs):
                            is_sharded = True
                        elif has_file(pretrained_model_name_or_path, WEIGHTS_NAME, **has_file_kwargs):
                            raise EnvironmentError(
                                f"{pretrained_model_name_or_path} does not appear to have a file named"
                                f" {TF2_WEIGHTS_NAME} but there is a file for PyTorch weights. Use `from_pt=True` to"
                                " load this model from those weights."
                            )
                        else:
                            raise EnvironmentError(
                                f"{pretrained_model_name_or_path} does not appear to have a file named {WEIGHTS_NAME},"
                                f" {TF2_WEIGHTS_NAME} or {TF_WEIGHTS_NAME}"
                            )

                except EnvironmentError:
                    # Raise any environment error raise by `cached_file`. It will have a helpful error message adapted
                    # to the original exception.
                    raise
                except Exception:
                    # For any other exception, we throw a generic error.

                    raise EnvironmentError(
                        f"Can't load the model for '{pretrained_model_name_or_path}'. If you were trying to load it"
                        " from 'https://huggingface.co/models', make sure you don't have a local directory with the"
                        f" same name. Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a"
                        f" directory containing a file named {WEIGHTS_NAME}, {TF2_WEIGHTS_NAME} or {TF_WEIGHTS_NAME}"
                    )
            if is_local:
                logger.info(f"loading weights file {archive_file}")
                resolved_archive_file = archive_file
                filename = resolved_archive_file.split(os.path.sep)[-1]
            else:
                logger.info(f"loading weights file {filename} from cache at {resolved_archive_file}")
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
                _commit_hash=commit_hash,
            )

        safetensors_from_pt = False
        if filename == SAFE_WEIGHTS_NAME:
            with safe_open(resolved_archive_file, framework="tf") as f:
                safetensors_metadata = f.metadata()
            if safetensors_metadata is None or safetensors_metadata.get("format") not in ["pt", "tf", "flax", "mlx"]:
                raise OSError(
                    f"The safetensors archive passed at {resolved_archive_file} does not contain the valid metadata."
                    " Make sure you save your model with the `save_pretrained` method."
                )
            safetensors_from_pt = safetensors_metadata.get("format") == "pt"
        elif filename == SAFE_WEIGHTS_INDEX_NAME:
            with safe_open(resolved_archive_file[0], framework="tf") as f:
                safetensors_metadata = f.metadata()
            if safetensors_metadata is None or safetensors_metadata.get("format") not in ["pt", "tf", "flax", "mlx"]:
                raise OSError(
                    f"The safetensors archive passed at {resolved_archive_file} does not contain the valid metadata."
                    " Make sure you save your model with the `save_pretrained` method."
                )
            safetensors_from_pt = safetensors_metadata.get("format") == "pt"

        config.name_or_path = pretrained_model_name_or_path

        # composed models, *e.g.* TFRag, require special treatment when it comes to loading
        # pre-trained weights.
        if cls._requires_load_weight_prefix and model_kwargs.get("name") is not None:
            model_kwargs["load_weight_prefix"] = load_weight_prefix + "/" + model_kwargs.get("name")

        # Instantiate model.
        model = cls(config, *model_args, **model_kwargs)

        if tf_to_pt_weight_rename is None and hasattr(model, "tf_to_pt_weight_rename"):
            # TODO Matt: This is a temporary workaround to allow weight renaming, but requires a method
            #            to be defined for each class that requires a rename. We can probably just have a class-level
            #            dict and a single top-level method or something and cut down a lot of boilerplate code
            tf_to_pt_weight_rename = model.tf_to_pt_weight_rename

        if from_pt:
            from .modeling_tf_pytorch_utils import load_pytorch_checkpoint_in_tf2_model

            # Load from a PyTorch checkpoint
            return load_pytorch_checkpoint_in_tf2_model(
                model,
                resolved_archive_file,
                allow_missing_keys=True,
                output_loading_info=output_loading_info,
                _prefix=load_weight_prefix,
                tf_to_pt_weight_rename=tf_to_pt_weight_rename,
            )

        # we might need to extend the variable scope for composite models
        if load_weight_prefix is not None:
            with tf.compat.v1.variable_scope(load_weight_prefix):
                model.build_in_name_scope()  # build the network with dummy inputs
        else:
            model.build_in_name_scope()  # build the network with dummy inputs

        if safetensors_from_pt and not is_sharded:
            from .modeling_tf_pytorch_utils import load_pytorch_state_dict_in_tf2_model

            with safe_open(resolved_archive_file, framework="tf") as safetensors_archive:
                # Load from a PyTorch safetensors checkpoint
                # We load in TF format here because PT weights often need to be transposed, and this is much
                # faster on GPU. Loading as numpy and transposing on CPU adds several seconds to load times.
                return load_pytorch_state_dict_in_tf2_model(
                    model,
                    safetensors_archive,
                    tf_inputs=False,  # No need to build the model again
                    allow_missing_keys=True,
                    output_loading_info=output_loading_info,
                    _prefix=load_weight_prefix,
                    ignore_mismatched_sizes=ignore_mismatched_sizes,
                    tf_to_pt_weight_rename=tf_to_pt_weight_rename,
                )
        elif safetensors_from_pt:
            from .modeling_tf_pytorch_utils import load_sharded_pytorch_safetensors_in_tf2_model

            return load_sharded_pytorch_safetensors_in_tf2_model(
                model,
                resolved_archive_file,
                tf_inputs=False,
                allow_missing_keys=True,
                output_loading_info=output_loading_info,
                _prefix=load_weight_prefix,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
                tf_to_pt_weight_rename=tf_to_pt_weight_rename,
            )

        # 'by_name' allow us to do transfer learning by skipping/adding layers
        # see https://github.com/tensorflow/tensorflow/blob/00fad90125b18b80fe054de1055770cfb8fe4ba3/tensorflow/python/keras/engine/network.py#L1339-L1357
        try:
            if is_sharded:
                for file in resolved_archive_file:
                    os.path.isfile(file), f"Error retrieving files {file}"
                if filename == SAFE_WEIGHTS_INDEX_NAME:
                    missing_keys, unexpected_keys, mismatched_keys = load_tf_sharded_weights_from_safetensors(
                        model,
                        resolved_archive_file,
                        ignore_mismatched_sizes=ignore_mismatched_sizes,
                        _prefix=load_weight_prefix,
                    )
                else:
                    missing_keys, unexpected_keys, mismatched_keys = load_tf_sharded_weights(
                        model,
                        resolved_archive_file,
                        ignore_mismatched_sizes=ignore_mismatched_sizes,
                        _prefix=load_weight_prefix,
                    )
            else:
                # Handles both H5 and safetensors
                missing_keys, unexpected_keys, mismatched_keys = load_tf_weights(
                    model,
                    resolved_archive_file,
                    ignore_mismatched_sizes=ignore_mismatched_sizes,
                    _prefix=load_weight_prefix,
                )
        except OSError as e:
            try:
                with open(resolved_archive_file) as f:
                    if f.read().startswith("version"):
                        raise OSError(
                            "You seem to have cloned a repository without having git-lfs installed. Please install "
                            "git-lfs and run `git lfs install` followed by `git lfs pull` in the folder "
                            "you cloned."
                        )
                    else:
                        raise ValueError from e
            except (UnicodeDecodeError, ValueError):
                raise OSError(
                    "Unable to load weights from h5 file. "
                    "If you tried to load a TF 2.0 model from a PyTorch checkpoint, please set from_pt=True. "
                )

        if cls._keys_to_ignore_on_load_missing is not None:
            for pat in cls._keys_to_ignore_on_load_missing:
                missing_keys = [k for k in missing_keys if re.search(pat, k) is None]

        if cls._keys_to_ignore_on_load_unexpected is not None:
            for pat in cls._keys_to_ignore_on_load_unexpected:
                unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]

        if len(unexpected_keys) > 0:
            logger.warning(
                f"Some layers from the model checkpoint at {pretrained_model_name_or_path} were not used when"
                f" initializing {model.__class__.__name__}: {unexpected_keys}\n- This IS expected if you are"
                f" initializing {model.__class__.__name__} from the checkpoint of a model trained on another task or"
                " with another architecture (e.g. initializing a BertForSequenceClassification model from a"
                " BertForPreTraining model).\n- This IS NOT expected if you are initializing"
                f" {model.__class__.__name__} from the checkpoint of a model that you expect to be exactly identical"
                " (initializing a BertForSequenceClassification model from a BertForSequenceClassification model)."
            )
        else:
            logger.warning(f"All model checkpoint layers were used when initializing {model.__class__.__name__}.\n")

        if len(missing_keys) > 0:
            logger.warning(
                f"Some layers of {model.__class__.__name__} were not initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path} and are newly initialized: {missing_keys}\nYou should probably"
                " TRAIN this model on a down-stream task to be able to use it for predictions and inference."
            )
        elif len(mismatched_keys) == 0:
            logger.warning(
                f"All the layers of {model.__class__.__name__} were initialized from the model checkpoint at"
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

        # If it is a model with generation capabilities, attempt to load the generation config
        if model.can_generate():
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

        if output_loading_info:
            loading_info = {
                "missing_keys": missing_keys,
                "unexpected_keys": unexpected_keys,
                "mismatched_keys": mismatched_keys,
            }

            return model, loading_info

        return model

    def push_to_hub(
        self,
        repo_id: str,
        use_temp_dir: Optional[bool] = None,
        commit_message: Optional[str] = None,
        private: Optional[bool] = None,
        max_shard_size: Optional[Union[int, str]] = "10GB",
        token: Optional[Union[bool, str]] = None,
        # (`use_auth_token` is deprecated: we have to keep it here as we don't have **kwargs)
        use_auth_token: Optional[Union[bool, str]] = None,
        create_pr: bool = False,
        **base_model_card_args,
    ) -> str:
        """
        Upload the model files to the 🤗 Model Hub while synchronizing a local clone of the repo in `repo_path_or_name`.

        Parameters:
            repo_id (`str`):
                The name of the repository you want to push your model to. It should contain your organization name
                when pushing to a given organization.
            use_temp_dir (`bool`, *optional*):
                Whether or not to use a temporary directory to store the files saved before they are pushed to the Hub.
                Will default to `True` if there is no directory named like `repo_id`, `False` otherwise.
            commit_message (`str`, *optional*):
                Message to commit while pushing. Will default to `"Upload model"`.
            private (`bool`, *optional*):
                Whether to make the repo private. If `None` (default), the repo will be public unless the organization's default is private. This value is ignored if the repo already exists.
            token (`bool` or `str`, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
                when running `huggingface-cli login` (stored in `~/.huggingface`). Will default to `True` if `repo_url`
                is not specified.
            max_shard_size (`int` or `str`, *optional*, defaults to `"10GB"`):
                Only applicable for models. The maximum size for a checkpoint before being sharded. Checkpoints shard
                will then be each of size lower than this size. If expressed as a string, needs to be digits followed
                by a unit (like `"5MB"`).
            create_pr (`bool`, *optional*, defaults to `False`):
                Whether or not to create a PR with the uploaded files or directly commit.

        Examples:

        ```python
        from transformers import TFAutoModel

        model = TFAutoModel.from_pretrained("google-bert/bert-base-cased")

        # Push the model to your namespace with the name "my-finetuned-bert".
        model.push_to_hub("my-finetuned-bert")

        # Push the model to an organization with the name "my-finetuned-bert".
        model.push_to_hub("huggingface/my-finetuned-bert")
        ```
        """
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

        if "repo_path_or_name" in base_model_card_args:
            warnings.warn(
                "The `repo_path_or_name` argument is deprecated and will be removed in v5 of Transformers. Use "
                "`repo_id` instead."
            )
            repo_id = base_model_card_args.pop("repo_path_or_name")
        # Deprecation warning will be sent after for repo_url and organization
        repo_url = base_model_card_args.pop("repo_url", None)
        organization = base_model_card_args.pop("organization", None)

        if os.path.isdir(repo_id):
            working_dir = repo_id
            repo_id = repo_id.split(os.path.sep)[-1]
        else:
            working_dir = repo_id.split("/")[-1]

        repo_id = self._create_repo(
            repo_id, private=private, token=token, repo_url=repo_url, organization=organization
        )

        if use_temp_dir is None:
            use_temp_dir = not os.path.isdir(working_dir)

        with working_or_temp_dir(working_dir=working_dir, use_temp_dir=use_temp_dir) as work_dir:
            files_timestamps = self._get_files_timestamps(work_dir)

            # Save all files.
            self.save_pretrained(work_dir, max_shard_size=max_shard_size)
            if hasattr(self, "history") and hasattr(self, "create_model_card"):
                # This is a Keras model and we might be able to fish out its History and make a model card out of it
                base_model_card_args = {
                    "output_dir": work_dir,
                    "model_name": Path(repo_id).name,
                }
                base_model_card_args.update(base_model_card_args)
                self.create_model_card(**base_model_card_args)

            self._upload_modified_files(
                work_dir,
                repo_id,
                files_timestamps,
                commit_message=commit_message,
                token=token,
                create_pr=create_pr,
            )

    @classmethod
    def register_for_auto_class(cls, auto_class="TFAutoModel"):
        """
        Register this class with a given auto class. This should only be used for custom models as the ones in the
        library are already mapped with an auto class.

        <Tip warning={true}>

        This API is experimental and may have some slight breaking changes in the next releases.

        </Tip>

        Args:
            auto_class (`str` or `type`, *optional*, defaults to `"TFAutoModel"`):
                The auto class to register this new model with.
        """
        if not isinstance(auto_class, str):
            auto_class = auto_class.__name__

        import transformers.models.auto as auto_module

        if not hasattr(auto_module, auto_class):
            raise ValueError(f"{auto_class} is not a valid auto class.")

        cls._auto_class = auto_class


class TFConv1D(keras.layers.Layer):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

    Basically works like a linear layer but the weights are transposed.

    Args:
        nf (`int`):
            The number of output features.
        nx (`int`):
            The number of input features.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation to use to initialize the weights.
        kwargs (`Dict[str, Any]`, *optional*):
            Additional keyword arguments passed along to the `__init__` of `keras.layers.Layer`.
    """

    def __init__(self, nf, nx, initializer_range=0.02, **kwargs):
        super().__init__(**kwargs)
        self.nf = nf
        self.nx = nx
        self.initializer_range = initializer_range

    def build(self, input_shape):
        if self.built:
            return
        self.built = True
        self.weight = self.add_weight(
            "weight", shape=[self.nx, self.nf], initializer=get_initializer(self.initializer_range)
        )
        self.bias = self.add_weight("bias", shape=[1, self.nf], initializer=tf.zeros_initializer())

    def call(self, x):
        bz, sl = shape_list(x)[:2]

        x = tf.reshape(x, [-1, self.nx])
        x = tf.matmul(x, self.weight) + self.bias

        x = tf.reshape(x, [bz, sl, self.nf])

        return x


class TFSharedEmbeddings(keras.layers.Layer):
    r"""
    Construct shared token embeddings.

    The weights of the embedding layer is usually shared with the weights of the linear decoder when doing language
    modeling.

    Args:
        vocab_size (`int`):
            The size of the vocabulary, e.g., the number of unique tokens.
        hidden_size (`int`):
            The size of the embedding vectors.
        initializer_range (`float`, *optional*):
            The standard deviation to use when initializing the weights. If no value is provided, it will default to
            \\(1/\sqrt{hidden\_size}\\).
        kwargs (`Dict[str, Any]`, *optional*):
            Additional keyword arguments passed along to the `__init__` of `keras.layers.Layer`.
    """

    # TODO (joao): flagged for delection due to embeddings refactor

    def __init__(self, vocab_size: int, hidden_size: int, initializer_range: Optional[float] = None, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.initializer_range = hidden_size**-0.5 if initializer_range is None else initializer_range
        warnings.warn(
            "`TFSharedEmbeddings` is scheduled for deletion in v4.32, use `keras.layers.Embedding` instead.",
            DeprecationWarning,
        )

    def build(self, input_shape):
        """
        Build shared token embedding layer Shared weights logic adapted from
        https://github.com/tensorflow/models/blob/a009f4fb9d2fc4949e32192a944688925ef78659/official/transformer/v2/embedding_layer.py#L24
        """
        self.weight = self.add_weight(
            "weight", shape=[self.vocab_size, self.hidden_size], initializer=get_initializer(self.initializer_range)
        )
        super().build(input_shape)

    def get_config(self):
        config = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "initializer_range": self.initializer_range,
        }
        base_config = super().get_config()

        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs: tf.Tensor, mode: str = "embedding") -> tf.Tensor:
        """
        Get token embeddings of inputs or decode final hidden state.

        Args:
            inputs (`tf.Tensor`):
                In embedding mode, should be an int64 tensor with shape `[batch_size, length]`.

                In linear mode, should be a float tensor with shape `[batch_size, length, hidden_size]`.
            mode (`str`, defaults to `"embedding"`):
               A valid value is either `"embedding"` or `"linear"`, the first one indicates that the layer should be
               used as an embedding layer, the second one that the layer should be used as a linear decoder.

        Returns:
            `tf.Tensor`: In embedding mode, the output is a float32 embedding tensor, with shape `[batch_size, length,
            embedding_size]`.

            In linear mode, the output is a float32 with shape `[batch_size, length, vocab_size]`.

        Raises:
            ValueError: if `mode` is not valid.

        Shared weights logic is adapted from
        [here](https://github.com/tensorflow/models/blob/a009f4fb9d2fc4949e32192a944688925ef78659/official/transformer/v2/embedding_layer.py#L24).
        """
        if mode == "embedding":
            return self._embedding(inputs)
        elif mode == "linear":
            return self._linear(inputs)
        else:
            raise ValueError(f"mode {mode} is not valid.")

    def _embedding(self, input_ids):
        """Applies embedding based on inputs tensor."""
        return tf.gather(self.weight, input_ids)

    def _linear(self, inputs):
        """
        Computes logits by running inputs through a linear layer.

        Args:
            inputs: A float32 tensor with shape [..., hidden_size]

        Returns:
            float32 tensor with shape [..., vocab_size].
        """
        first_dims = shape_list(inputs)[:-1]
        x = tf.reshape(inputs, [-1, self.hidden_size])
        logits = tf.matmul(x, self.weight, transpose_b=True)

        return tf.reshape(logits, first_dims + [self.vocab_size])


class TFSequenceSummary(keras.layers.Layer):
    """
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

        initializer_range (`float`, *optional*, defaults to 0.02): The standard deviation to use to initialize the weights.
        kwargs (`Dict[str, Any]`, *optional*):
            Additional keyword arguments passed along to the `__init__` of `keras.layers.Layer`.
    """

    def __init__(self, config: PretrainedConfig, initializer_range: float = 0.02, **kwargs):
        super().__init__(**kwargs)

        self.summary_type = config.summary_type if hasattr(config, "summary_use_proj") else "last"
        if self.summary_type == "attn":
            # We should use a standard multi-head attention module with absolute positional embedding for that.
            # Cf. https://github.com/zihangdai/xlnet/blob/master/modeling.py#L253-L276
            # We can probably just use the multi-head attention module of PyTorch >=1.1.0
            raise NotImplementedError

        self.has_summary = hasattr(config, "summary_use_proj") and config.summary_use_proj
        if self.has_summary:
            if hasattr(config, "summary_proj_to_labels") and config.summary_proj_to_labels and config.num_labels > 0:
                num_classes = config.num_labels
            else:
                num_classes = config.hidden_size
            self.summary = keras.layers.Dense(
                num_classes, kernel_initializer=get_initializer(initializer_range), name="summary"
            )

        self.has_activation = False
        activation_string = getattr(config, "summary_activation", None)
        if activation_string is not None:
            self.has_activation = True
            self.activation = get_tf_activation(activation_string)

        self.has_first_dropout = hasattr(config, "summary_first_dropout") and config.summary_first_dropout > 0
        if self.has_first_dropout:
            self.first_dropout = keras.layers.Dropout(config.summary_first_dropout)

        self.has_last_dropout = hasattr(config, "summary_last_dropout") and config.summary_last_dropout > 0
        if self.has_last_dropout:
            self.last_dropout = keras.layers.Dropout(config.summary_last_dropout)
        self.hidden_size = config.hidden_size

    def call(self, inputs, cls_index=None, training=False):
        if not isinstance(inputs, (dict, tuple, list)):
            hidden_states = inputs
        elif isinstance(inputs, (tuple, list)):
            hidden_states = inputs[0]
            cls_index = inputs[1] if len(inputs) > 1 else None
            assert len(inputs) <= 2, "Too many inputs."
        else:
            hidden_states = inputs.get("hidden_states")
            cls_index = inputs.get("cls_index", None)

        if self.summary_type == "last":
            output = hidden_states[:, -1]
        elif self.summary_type == "first":
            output = hidden_states[:, 0]
        elif self.summary_type == "mean":
            output = tf.reduce_mean(hidden_states, axis=1)
        elif self.summary_type == "cls_index":
            hidden_shape = shape_list(hidden_states)  # e.g. [batch, num choices, seq length, hidden dims]
            if cls_index is None:
                cls_index = tf.fill(
                    hidden_shape[:-2], hidden_shape[-2] - 1
                )  # A tensor full of shape [batch] or [batch, num choices] full of sequence length
            cls_shape = shape_list(cls_index)
            if len(cls_shape) <= len(hidden_shape) - 2:
                cls_index = tf.expand_dims(cls_index, axis=-1)
            # else:
            # cls_index = cls_index[..., tf.newaxis]
            # cls_index = cls_index.expand((-1,) * (cls_index.dim()-1) + (hidden_states.size(-1),))
            # shape of cls_index: (bsz, XX, 1, hidden_size) where XX are optional leading dim of hidden_states
            output = tf.gather(hidden_states, cls_index, batch_dims=len(hidden_shape) - 2)
            output = tf.squeeze(
                output, axis=len(hidden_shape) - 2
            )  # shape of output: (batch, num choices, hidden_size)
        elif self.summary_type == "attn":
            raise NotImplementedError

        if self.has_first_dropout:
            output = self.first_dropout(output, training=training)

        if self.has_summary:
            output = self.summary(output)

        if self.has_activation:
            output = self.activation(output)

        if self.has_last_dropout:
            output = self.last_dropout(output, training=training)

        return output

    def build(self, input_shape):
        if self.built:
            return
        self.built = True
        if getattr(self, "summary", None) is not None:
            with tf.name_scope("summary"):
                self.summary.build(self.hidden_size)


def get_initializer(initializer_range: float = 0.02) -> keras.initializers.TruncatedNormal:
    """
    Creates a `keras.initializers.TruncatedNormal` with the given range.

    Args:
        initializer_range (*float*, defaults to 0.02): Standard deviation of the initializer range.

    Returns:
        `keras.initializers.TruncatedNormal`: The truncated normal initializer.
    """
    return keras.initializers.TruncatedNormal(stddev=initializer_range)
