import copy
import importlib.metadata
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from packaging import version

from .configuration_utils import PretrainedConfig
from .utils import (
    is_hqq_available,
    is_optimum_quanto_available,
    is_torchdynamo_compiling,
    logging,
)
from .utils.deprecation import deprecate_kwarg


if is_hqq_available():
    from hqq.core.quantize import Quantizer as HQQQuantizer

logger = logging.get_logger(__name__)


class Cache(torch.nn.Module):
    """
    Base, abstract class for all caches. The actual data structure is specific to each subclass.
    """

    def __init__(self):
        super().__init__()

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. These are specific to each subclass and allow new types of
                cache to be created.

        Return:
            A tuple containing the updated key and value states.
        """
        raise NotImplementedError("Make sure to implement `update` in a subclass.")

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # TODO: deprecate this function in favor of `cache_position`
        raise NotImplementedError("Make sure to implement `get_seq_length` in a subclass.")

    def get_max_cache_shape(self) -> Optional[int]:
        """Returns the maximum sequence length (i.e. max capacity) of the cache object"""
        raise NotImplementedError("Make sure to implement `get_max_cache_shape` in a subclass.")

    def get_usable_length(self, new_seq_length: int, layer_idx: Optional[int] = 0) -> int:
        """Given the sequence length of the new inputs, returns the usable length of the cache."""
        # Cache without size limit -> all cache is usable
        # Cache with size limit -> if the length cache plus the length of the new inputs is larger the maximum cache
        #   length, we will need to evict part of the cache (and thus not all cache is usable)
        max_length = self.get_max_cache_shape()
        previous_seq_length = self.get_seq_length(layer_idx)
        if max_length is not None and previous_seq_length + new_seq_length > max_length:
            return max_length - new_seq_length
        return previous_seq_length

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        for layer_idx in range(len(self.key_cache)):
            if self.key_cache[layer_idx] != []:
                device = self.key_cache[layer_idx].device
                self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx.to(device))
            if self.value_cache[layer_idx] != []:
                device = self.value_cache[layer_idx].device
                self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx.to(device))

    @property
    def seen_tokens(self):
        logger.warning_once(
            "The `seen_tokens` attribute is deprecated and will be removed in v4.41. Use the `cache_position` "
            "model input instead."
        )
        if hasattr(self, "_seen_tokens"):
            return self._seen_tokens
        else:
            return None


@dataclass
class CacheConfig:
    """
    Base class for cache configs
    """

    cache_implementation: None

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        """
        Constructs a CacheConfig instance from a dictionary of parameters.
        Args:
            config_dict (Dict[str, Any]): Dictionary containing configuration parameters.
            **kwargs: Additional keyword arguments to override dictionary values.

        Returns:
            CacheConfig: Instance of CacheConfig constructed from the dictionary.
        """
        config = cls(**config_dict)
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)
        return config

    # Copied from transformers.utils.quantization_config.QuantizationConfigMixin.to_json_file
    def to_json_file(self, json_file_path: Union[str, os.PathLike]):
        """
        Save this instance to a JSON file.

        Args:
            json_file_path (`str` or `os.PathLike`):
                Path to the JSON file in which this configuration instance's parameters will be saved.
            use_diff (`bool`, *optional*, defaults to `True`):
                If set to `True`, only the difference between the config instance and the default
                `QuantizationConfig()` is serialized to JSON file.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            config_dict = self.to_dict()
            json_string = json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

            writer.write(json_string)

    # Copied from transformers.utils.quantization_config.QuantizationConfigMixin.to_dict
    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary. Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        return copy.deepcopy(self.__dict__)

    # Copied from transformers.utils.quantization_config.QuantizationConfigMixin.__iter__
    def __iter__(self):
        """allows `dict(obj)` for situations where obj may be a dict or QuantizationConfigMixin"""
        for attr, value in copy.deepcopy(self.__dict__).items():
            yield attr, value

    # Copied from transformers.utils.quantization_config.QuantizationConfigMixin.__repr__
    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string()}"

    def to_json_string(self):
        """
        Serializes this instance to a JSON formatted string.
        Returns:
            str: JSON formatted string representing the configuration instance.
        """
        return json.dumps(self.__dict__, indent=2) + "\n"

    # Copied from transformers.utils.quantization_config.QuantizationConfigMixin.update
    def update(self, **kwargs):
        """
        Updates attributes of this class instance with attributes from `kwargs` if they match existing attributes,
        returning all the unused kwargs.

        Args:
            kwargs (`Dict[str, Any]`):
                Dictionary of attributes to tentatively update this class.

        Returns:
            `Dict[str, Any]`: Dictionary containing all the key-value pairs that were not used to update the instance.
        """
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                to_remove.append(key)

        # Remove all the attributes that were updated, without modifying the input dict
        unused_kwargs = {key: value for key, value in kwargs.items() if key not in to_remove}
        return unused_kwargs


@dataclass
class QuantizedCacheConfig(CacheConfig):
    """
    Configuration class for quantized cache settings.

    Attributes:
        backend (`str`, *optional*, defaults to `"quanto"`):
            Backend to use when performing quantization, Can be one of [`quanto`, `HQQ`]
        nbits (`Optional[int]`, *optional*, defaults to 4):
            Number of bits, can be 2 or 4 for the `quanto` backend and one of [1, 2, 3, 4, 8] for the `HQQ` backend. Defaults to 2.
        axis_key (`int`, *optional*, defaults to 0):
            Axis over which to perform grouping for the key tensors. Can be [0, -1] for `quanto` backend and [0, 1] for `HQQ` backend.
        axis_value (`int`, *optional*, defaults to 0):
            Axis over which to perform grouping for the value tensors. Can be [0, -1] for `quanto` backend and [0, 1] for `HQQ` backend.
        q_group_size (`Optional[int]`, *optional*, defaults to 64):
            Size of the quantization group, should be a divisor of the model's hidden dimension.
            Defaults to 64.
        residual_length (`Optional[int]`, *optional*, defaults to 128):
            Length of the residual cache which will always be stored in original presicion.
            Defaults to 128.
        compute_dtype (`torch.dtype`, *optional*, defaults to `torch.float16`):
            The defualt dtype used for computations in the model. Keys and Values will be cast to this dtype after dequantization.
        device (`str`, *optional*, defaults to `"cpu"`):
            Device on which to perform computations, should be same as the model's device.
    """

    def __init__(
        self,
        backend: str = "quanto",
        nbits: Optional[int] = 4,
        axis_key: Optional[int] = 0,
        axis_value: Optional[int] = 0,
        q_group_size: Optional[int] = 64,
        residual_length: Optional[int] = 128,
        compute_dtype: Optional[torch.dtype] = torch.float16,
        device: Optional[str] = "cpu",
    ):
        self.backend = backend
        self.nbits = nbits
        self.axis_key = axis_key
        self.axis_value = axis_value
        self.q_group_size = q_group_size
        self.residual_length = residual_length
        self.compute_dtype = compute_dtype
        self.device = device

    def validate(self):
        """Validates if the arguments passed are correct"""

        incorrect_arg_msg = (
            "Some of the keys in `cache_config` are defined incorrectly. `{key}` should be {correct_value}` "
            "but found {found_value}"
        )
        # Check that the values are reasonable in general (nbits, axis)
        # Later in QuantizedCache init we check if they are supported for that particular backend
        if self.nbits not in [1, 2, 3, 4, 8]:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="nbits",
                    correct_value="2 or 4 or 8",
                    found_value=self.nbits,
                ),
            )
        if self.q_group_size <= 0:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="q_group_size",
                    correct_value="a positive integer",
                    found_value=self.q_group_size,
                ),
            )
        if self.residual_length < 0:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="residual_length",
                    correct_value="a positive integer",
                    found_value=self.residual_length,
                ),
            )

        if self.axis_key not in [0, 1, -1]:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="axis_key",
                    correct_value="`1` or `0`, `-1`",
                    found_value=self.axis_key,
                ),
            )

        if self.axis_value not in [0, 1, -1]:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="axis_value",
                    correct_value="`1` or `0` or `-1`",
                    found_value=self.axis_value,
                ),
            )


@dataclass
class StaticCacheConfig(CacheConfig):
    """
    Configuration class for static cache settings.
    """

    cache_implementation = "static"

    def __init__(self, batch_size: int, max_cache_len: int, device="cpu"):
        self.batch_size = batch_size
        self.max_cache_len = max_cache_len
        self.device = device

    def validate(self):
        """Validates if the arguments passed are correct"""

        incorrect_arg_msg = (
            "Some of the keys in `cache_config` are defined incorrectly. `{key}` should be {correct_value}` "
            "but found {found_value}"
        )

        if self.batch_size <= 0:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="batch_size",
                    correct_value="> 0",
                    found_value=self.batch_size,
                ),
            )

        if self.max_cache_len <= 0:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="max_cache_len",
                    correct_value="> 0",
                    found_value=self.max_cache_len,
                ),
            )


class DynamicCache(Cache):
    """
    A cache that grows dynamically as more tokens are generated. This is the default for generative models.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.

    Example:

        ```python
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

        >>> inputs = tokenizer(text="My name is Qwen2", return_tensors="pt")

        >>> # Prepare a cache class and pass it to model's forward
        >>> past_key_values = DynamicCache()
        >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
        >>> outputs.past_key_values # access cache filled with key/values from generation
        DynamicCache()
        ```
    """

    @deprecate_kwarg("num_hidden_layers", version="4.47.0")
    def __init__(self, num_hidden_layers: Optional[int] = None) -> None:
        super().__init__()
        self._seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx])

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.key_cache)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Update the cache
        if key_states is not None:
            if len(self.key_cache) <= layer_idx:
                # There may be skipped layers, fill them with empty lists
                for _ in range(len(self.key_cache), layer_idx):
                    self.key_cache.append([])
                    self.value_cache.append([])
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
            elif (
                len(self.key_cache[layer_idx]) == 0
            ):  # fills previously skipped layers; checking for tensor causes errors
                self.key_cache[layer_idx] = key_states
                self.value_cache[layer_idx] = value_states
            else:
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # TODO: deprecate this function in favor of `cache_position`
        is_empty_layer = (
            len(self.key_cache) == 0  # no cache in any layer
            or len(self.key_cache) <= layer_idx  # skipped `layer_idx` and hasn't run a layer with cache after it
            or len(self.key_cache[layer_idx]) == 0  # the layer has no cache
        )
        layer_seq_length = self.key_cache[layer_idx].shape[-2] if not is_empty_layer else 0
        return layer_seq_length

    def get_max_cache_shape(self) -> Optional[int]:
        """Returns the maximum sequence length of the cache object. DynamicCache does not have a maximum length."""
        return None

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        """Converts the `DynamicCache` instance into the its equivalent in the legacy cache format. Used for
        backward compatibility."""
        legacy_cache = ()
        for layer_idx in range(len(self)):
            legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
        return legacy_cache

    @classmethod
    @deprecate_kwarg("num_hidden_layers", version="4.47.0")
    def from_legacy_cache(
        cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None, num_hidden_layers: int = None
    ) -> "DynamicCache":
        """Converts a cache in the legacy cache format into an equivalent `DynamicCache`. Used for
        backward compatibility."""
        cache = cls()
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        return cache

    def crop(self, max_length: int):
        """Crop the past key values up to a new `max_length` in terms of tokens. `max_length` can also be
        negative to remove `max_length` tokens. This is used in assisted decoding and contrastive search."""
        # In case it is negative
        if max_length < 0:
            max_length = self.get_seq_length() - abs(max_length)

        if self.get_seq_length() <= max_length:
            return

        self._seen_tokens = max_length
        for idx in range(len(self.key_cache)):
            if self.key_cache[idx] != []:
                self.key_cache[idx] = self.key_cache[idx][..., :max_length, :]
                self.value_cache[idx] = self.value_cache[idx][..., :max_length, :]

    @deprecate_kwarg("num_hidden_layers", version="4.47.0")
    def batch_split(
        self, full_batch_size: int, split_size: int, num_hidden_layers: int = None
    ) -> List["DynamicCache"]:
        """Split the current instance into a list of `DynamicCache` by the batch size. This will be used by
        `_split_model_inputs()` in `generation.utils`"""
        out = []
        for i in range(0, full_batch_size, split_size):
            current_split = DynamicCache()
            current_split._seen_tokens = self._seen_tokens
            current_split.key_cache = [tensor[i : i + split_size] for tensor in self.key_cache]
            current_split.value_cache = [tensor[i : i + split_size] for tensor in self.value_cache]
            out.append(current_split)
        return out

    @classmethod
    @deprecate_kwarg("num_hidden_layers", version="4.47.0")
    def from_batch_splits(cls, splits: List["DynamicCache"], num_hidden_layers: int = None) -> "DynamicCache":
        """This is the opposite of the above `batch_split()` method. This will be used by `stack_model_outputs` in
        `generation.utils`"""
        cache = cls()
        for idx in range(len(splits[0])):
            key_cache = [current.key_cache[idx] for current in splits if current.key_cache[idx] != []]
            value_cache = [current.value_cache[idx] for current in splits if current.value_cache[idx] != []]
            if key_cache != []:
                layer_keys = torch.cat(key_cache, dim=0)
                layer_values = torch.cat(value_cache, dim=0)
                cache.update(layer_keys, layer_values, idx)
        return cache

    def batch_repeat_interleave(self, repeats: int):
        """Repeat the cache `repeats` times in the batch dimension. Used in contrastive search."""
        for layer_idx in range(len(self)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx].repeat_interleave(repeats, dim=0)
            self.value_cache[layer_idx] = self.value_cache[layer_idx].repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices: torch.Tensor):
        """Only keep the `indices` in the batch dimension of the cache. Used in contrastive search."""
        for layer_idx in range(len(self)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx][indices, ...]
            self.value_cache[layer_idx] = self.value_cache[layer_idx][indices, ...]


class OffloadedCache(DynamicCache):
    """
    A drop-in replacement for DynamicCache that conserves GPU memory at the expense of more CPU memory.
    Useful for generating from models with very long context.

    In addition to the default CUDA stream, where all forward() computations happen,
    this class uses another stream, the prefetch stream, which it creates itself.
    Since scheduling of operations on separate streams happens independently, this class uses
    the prefetch stream to asynchronously prefetch the KV cache of layer k+1 when layer k is executing.
    The movement of the layer k-1 cache to the CPU is handled by the default stream as a simple way to
    ensure the eviction is scheduled after all computations on that cache are finished.
    """

    def __init__(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("OffloadedCache can only be used with a GPU")
        super().__init__()
        self.original_device = []
        self.prefetch_stream = torch.cuda.Stream()
        self.beam_idx = None  # used to delay beam search operations

    def prefetch_layer(self, layer_idx: int):
        "Starts prefetching the next layer cache"
        if layer_idx < len(self):
            with torch.cuda.stream(self.prefetch_stream):
                # Prefetch next layer tensors to GPU
                device = self.original_device[layer_idx]
                self.key_cache[layer_idx] = self.key_cache[layer_idx].to(device, non_blocking=True)
                self.value_cache[layer_idx] = self.value_cache[layer_idx].to(device, non_blocking=True)

    def evict_previous_layer(self, layer_idx: int):
        "Moves the previous layer cache to the CPU"
        if len(self) > 2:
            # We do it on the default stream so it occurs after all earlier computations on these tensors are done
            prev_layer_idx = (layer_idx - 1) % len(self)
            self.key_cache[prev_layer_idx] = self.key_cache[prev_layer_idx].to("cpu", non_blocking=True)
            self.value_cache[prev_layer_idx] = self.value_cache[prev_layer_idx].to("cpu", non_blocking=True)

    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        "Gets the cache for this layer to the device. Prefetches the next and evicts the previous layer."
        if layer_idx < len(self):
            # Evict the previous layer if necessary
            torch.cuda.current_stream().synchronize()
            self.evict_previous_layer(layer_idx)
            # Load current layer cache to its original device if not already there
            original_device = self.original_device[layer_idx]
            self.prefetch_stream.synchronize()
            key_tensor = self.key_cache[layer_idx]
            value_tensor = self.value_cache[layer_idx]
            # Now deal with beam search ops which were delayed
            if self.beam_idx is not None:
                self.beam_idx = self.beam_idx.to(original_device)
                key_tensor = key_tensor.index_select(0, self.beam_idx)
                value_tensor = value_tensor.index_select(0, self.beam_idx)
            # Prefetch the next layer
            self.prefetch_layer((layer_idx + 1) % len(self))
            return (key_tensor, value_tensor)
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Saves the beam indices and reorders the cache when the tensor is back to its device."""
        # We delay this operation until the tensors are back to their original
        # device because performing torch.index_select on the CPU is very slow
        del self.beam_idx
        self.beam_idx = beam_idx.clone()

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.
        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `OffloadedCache`.
        Return:
            A tuple containing the updated key and value states.
        """
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Update the cache
        if len(self.key_cache) < layer_idx:
            raise ValueError("OffloadedCache does not support model usage where layers are skipped. Use DynamicCache.")
        elif len(self.key_cache) == layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
            self.original_device.append(key_states.device)
            self.evict_previous_layer(layer_idx)
        else:
            key_tensor, value_tensor = self[layer_idx]
            self.key_cache[layer_idx] = torch.cat([key_tensor, key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([value_tensor, value_states], dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    # According to https://docs.python.org/3/library/exceptions.html#NotImplementedError
    # if a method is not supposed to be supported in a subclass we should set it to None
    from_legacy_cache = None

    to_legacy_cache = None


class QuantizedCache(DynamicCache):
    """
    A quantizer cache similar to what is described in the [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache paper](https://arxiv.org/abs/2402.02750).
    It allows the model to generate longer sequence length without allocating too much memory for Key and Value cache by applying quantization.

    The cache has two types of storage, one for original precision and one for the quantized cache. A `residual length` is set as a maximum capacity for the
    original precision cache. When the length goes beyond maximum capacity, the original precision cache is discarded and moved into the quantized cache. The
    quantization is done per-channel with a set `q_group_size` for both Keys and Values, in contrast to what was described in the paper.

    It stores Keys and Values a list of quantized tensors (tuples in case we need to store metadata), one for each layer. Additionally, it stores the Key and
    Value in original precision states as a list of tensors, one for each layer. The size of each tensor
    is `[batch_size, num_heads, seq_len - residual_length, head_dim]`
    """

    def __init__(self, cache_config: QuantizedCacheConfig) -> None:
        super().__init__()
        self._quantized_key_cache: List[torch.Tensor] = []
        self._quantized_value_cache: List[torch.Tensor] = []

        self.nbits = cache_config.nbits
        self.residual_length = cache_config.residual_length
        self.q_group_size = cache_config.q_group_size
        self.axis_key = cache_config.axis_key
        self.axis_value = cache_config.axis_value
        self.compute_dtype = cache_config.compute_dtype
        self.device = cache_config.device

        super().__init__()

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        if len(self.key_cache) < layer_idx:
            raise ValueError("QuantizedCache does not support model usage where layers are skipped. Use DynamicCache.")
        elif len(self.key_cache) == layer_idx:
            self._quantized_key_cache.append(self._quantize(key_states.contiguous(), axis=self.axis_key))
            self._quantized_value_cache.append(self._quantize(value_states.contiguous(), axis=self.axis_value))
            self.key_cache.append(torch.zeros(0, dtype=key_states.dtype, device=key_states.device))
            self.value_cache.append(torch.zeros(0, dtype=key_states.dtype, device=key_states.device))
            keys_to_return, values_to_return = key_states, value_states
        else:
            dequant_key = self._dequantize(self._quantized_key_cache[layer_idx])
            dequant_value = self._dequantize(self._quantized_value_cache[layer_idx])
            keys_to_return = [dequant_key, self.key_cache[layer_idx], key_states]
            values_to_return = [dequant_value, self.value_cache[layer_idx], value_states]

            keys_to_return = torch.cat(keys_to_return, dim=-2)
            values_to_return = torch.cat(values_to_return, dim=-2)
            if (
                self.key_cache[layer_idx].dim() == 4
                and self.key_cache[layer_idx].shape[-2] + 1 >= self.residual_length
            ):
                self._quantized_key_cache[layer_idx] = self._quantize(keys_to_return.contiguous(), axis=self.axis_key)
                self._quantized_value_cache[layer_idx] = self._quantize(
                    values_to_return.contiguous(), axis=self.axis_value
                )
                self.key_cache[layer_idx] = torch.zeros(0, dtype=key_states.dtype, device=key_states.device)
                self.value_cache[layer_idx] = torch.zeros(0, dtype=key_states.dtype, device=key_states.device)
            else:
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        return keys_to_return, values_to_return

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        if len(self.key_cache) <= layer_idx:
            return 0
        # since we cannot get the seq_length of each layer directly and rely on `_seen_tokens` which is
        # updated every "layer_idx" == 0, this is a hack to get the actual seq_length for the given layer_idx
        # this part of code otherwise fails when used to verify attn_weight shape in some models
        return self._seen_tokens if layer_idx == 0 else self._seen_tokens - 1

    def _quantize(self, tensor, axis):
        """Quantizes a key/value using a defined quantization method."""
        raise NotImplementedError("Make sure to implement `_quantize` in a subclass.")

    def _dequantize(self, q_tensor):
        """Dequantizes back the tensor that was quantized by `self._quantize()`"""
        raise NotImplementedError("Make sure to implement `_dequantize` in a subclass.")


class QuantoQuantizedCache(QuantizedCache):
    """
    Quantized Cache class that uses `quanto` as a backend to perform quantization. Current implementation supports `int2` and `int4` dtypes only.

    Parameters:
        cache_config (`QuantizedCacheConfig`):
            A configuration containing all the arguments to be used by the quantizer, including axis, qtype and group size.

    Example:

        ```python
        >>> # Run pip install quanto first if you don't have it yet
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM, QuantoQuantizedCache, QuantizedCacheConfig

        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

        >>> inputs = tokenizer(text="My name is Qwen2", return_tensors="pt")

        >>> # Prepare a cache class and pass it to model's forward
        >>> cache_config = QuantizedCacheConfig(nbits=4)
        >>> past_key_values = QuantoQuantizedCache(cache_config=cache_config)
        >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
        >>> outputs.past_key_values # access cache filled with key/values from generation
        QuantoQuantizedCache()
        ```
    """

    def __init__(self, cache_config: CacheConfig) -> None:
        super().__init__(cache_config)

        if is_optimum_quanto_available():
            optimum_quanto_version = version.parse(importlib.metadata.version("optimum-quanto"))
            if optimum_quanto_version <= version.parse("0.2.5"):
                raise ImportError(
                    f"You need optimum-quanto package version to be greater or equal than 0.2.5 to use `QuantoQuantizedCache`. Detected version {optimum_quanto_version}."
                )
            from optimum.quanto import MaxOptimizer, qint2, qint4

        if self.nbits not in [2, 4]:
            raise ValueError(f"`nbits` for `quanto` backend has to be one of [`2`, `4`] but got {self.nbits}")

        if self.axis_key not in [0, -1]:
            raise ValueError(f"`axis_key` for `quanto` backend has to be one of [`0`, `-1`] but got {self.axis_key}")

        if self.axis_value not in [0, -1]:
            raise ValueError(
                f"`axis_value` for `quanto` backend has to be one of [`0`, `-1`] but got {self.axis_value}"
            )

        self.qtype = qint4 if self.nbits == 4 else qint2
        self.optimizer = MaxOptimizer()  # hardcode as it's the only one for per-channel quantization

    def _quantize(self, tensor, axis):
        # We have two different API since in optimum-quanto, we don't use AffineQuantizer anymore
        if is_optimum_quanto_available():
            from optimum.quanto import quantize_weight

            scale, zeropoint = self.optimizer(tensor, self.qtype, axis, self.q_group_size)
            qtensor = quantize_weight(tensor, self.qtype, axis, scale, zeropoint, self.q_group_size)
            return qtensor

    def _dequantize(self, qtensor):
        return qtensor.dequantize()


class HQQQuantizedCache(QuantizedCache):
    """
    Quantized Cache class that uses `HQQ` as a backend to perform quantization. Current implementation supports `int2`, `int4`, `int8` dtypes.

    Parameters:
        cache_config (`QuantizedCacheConfig`):
            A configuration containing all the arguments to be used by the quantizer, including axis, qtype and group size.

    Example:

        ```python
        >>> # Run pip install hqq first if you don't have it yet
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM, HQQQuantizedCache, QuantizedCacheConfig

        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

        >>> inputs = tokenizer(text="My name is Qwen2", return_tensors="pt")

        >>> # Prepare a cache class and pass it to model's forward
        >>> cache_config = QuantizedCacheConfig(nbits=4, axis_key=1, axis_value=1)
        >>> past_key_values = HQQQuantizedCache(cache_config=cache_config)
        >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
        >>> outputs.past_key_values # access cache filled with key/values from generation
        HQQQuantizedCache()
        ```
    """

    def __init__(self, cache_config: CacheConfig) -> None:
        super().__init__(cache_config)
        if self.nbits not in [1, 2, 3, 4, 8]:
            raise ValueError(
                f"`nbits` for `HQQ` backend has to be one of [`1`, `2`, `3`, `4`, `8`] but got {self.nbits}"
            )

        if self.axis_key not in [0, 1]:
            raise ValueError(f"`axis_key` for `HQQ` backend has to be one of [`0`, `1`] but got {self.axis_key}")

        if self.axis_value not in [0, 1]:
            raise ValueError(f"`axis_value` for `HQQ` backend has to be one of [`0`, `1`] but got {self.axis_value}")

        self.quantizer = HQQQuantizer

    def _quantize(self, tensor, axis):
        qtensor, meta = self.quantizer.quantize(
            tensor,
            axis=axis,
            device=self.device,
            compute_dtype=self.compute_dtype,
            nbits=self.nbits,
            group_size=self.q_group_size,
        )
        meta["compute_dtype"] = self.compute_dtype
        self.quantizer.cuda(qtensor, meta=meta, device=self.device)  # Move to device and cast to dtype
        return qtensor, meta

    def _dequantize(self, qtensor):
        quant_tensor, meta = qtensor
        tensor = self.quantizer.dequantize(quant_tensor, meta)
        return tensor


class SinkCache(Cache):
    """
    A cache that as described in the [Attention Sinks paper](https://arxiv.org/abs/2309.17453). It allows the model to
    generate beyond the length of its context window, without losing fluency in the conversation. As it discards past
    tokens, the model will lose the ability to generate tokens that depend on the context that was discarded.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.

    Parameters:
        window_length (`int`):
            The length of the context window.
        num_sink_tokens (`int`):
            The number of sink tokens. See the original paper for more information.

    Example:

        ```python
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM, SinkCache

        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

        >>> inputs = tokenizer(text="My name is Qwen2", return_tensors="pt")

        >>> # Prepare a cache class and pass it to model's forward
        >>> past_key_values = SinkCache(window_length=256, num_sink_tokens=4)
        >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
        >>> outputs.past_key_values # access cache filled with key/values from generation
        SinkCache()
        ```
    """

    is_sliding = True

    def __init__(self, window_length: int, num_sink_tokens: int) -> None:
        super().__init__()
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.window_length = window_length
        self.num_sink_tokens = num_sink_tokens
        self.cos_sin_rerotation_cache = {}
        self._cos_cache = None
        self._sin_cache = None
        self._seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen

    @staticmethod
    def _rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_key_rotary_pos_emb(
        self, key_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        rotated_key_states = (key_states * cos) + (self._rotate_half(key_states) * sin)
        return rotated_key_states

    def _get_rerotation_cos_sin(
        self, key_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if key_states.shape[-2] not in self.cos_sin_rerotation_cache:
            # Upcast to float32 temporarily for better accuracy
            cos = cos.to(torch.float32)
            sin = sin.to(torch.float32)

            # Compute the cos and sin required for back- and forward-rotating to one position earlier in the sequence
            original_cos = cos[self.num_sink_tokens + key_states.shape[-2] :]
            shifted_cos = cos[self.num_sink_tokens : -key_states.shape[-2]]
            original_sin = sin[self.num_sink_tokens + key_states.shape[-2] :]
            shifted_sin = sin[self.num_sink_tokens : -key_states.shape[-2]]
            rerotation_cos = original_cos * shifted_cos + original_sin * shifted_sin
            rerotation_sin = -original_sin * shifted_cos + original_cos * shifted_sin

            self.cos_sin_rerotation_cache[key_states.shape[-2]] = (
                rerotation_cos.to(key_states.dtype).unsqueeze(0),
                rerotation_sin.to(key_states.dtype).unsqueeze(0),
            )
        return self.cos_sin_rerotation_cache[key_states.shape[-2]]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # TODO: deprecate this function in favor of `cache_position`
        # Workaround to make 'key_states.shape[-2] + past_key_value.get_seq_length(self.layer_idx)' <= window_length
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_max_cache_shape(self) -> Optional[int]:
        """Returns the maximum sequence length of the cache object, in case of SinkCache it is the window length."""
        return self.window_length

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. The following arguments can be used in `SinkCache`: `sin`,
                `cos` and `partial_rotation_size`. These arguments are used with models using RoPE, to recompute the
                rotation as the tokens are shifted.

        Return:
            A tuple containing the updated key and value states.
        """
        # Optional kwargs for `SinkCache` -- needed on models using RoPE. `partial_rotation_size` is used on models
        # with partially rotated position embeddings, like Phi or Persimmon.
        sin = cache_kwargs.get("sin")
        cos = cache_kwargs.get("cos")
        partial_rotation_size = cache_kwargs.get("partial_rotation_size")
        using_rope = cos is not None and sin is not None

        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Update the sin/cos cache, which holds sin/cos values for all possible positions
        if using_rope and layer_idx == 0:
            # BC: some models still pass `sin`/`cos` with 2 dims. In those models, they are the full sin/cos. Remove
            # after all RoPE models have a llama-like cache utilization.
            if cos.dim() == 2:
                self._cos_cache = cos
                self._sin_cache = sin
            else:
                if self._cos_cache is None:
                    self._cos_cache = cos[0, ...]
                    self._sin_cache = sin[0, ...]
                elif self._cos_cache.shape[0] < self.window_length:
                    self._cos_cache = torch.cat([self._cos_cache, cos[0, ...]], dim=0)
                    self._sin_cache = torch.cat([self._sin_cache, sin[0, ...]], dim=0)

        # [bsz, num_heads, seq_len, head_dim]
        if len(self.key_cache) <= layer_idx:
            # Empty cache
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)

        elif key_states.shape[-2] + self.get_seq_length(layer_idx) < self.window_length:
            # Growing cache
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        else:
            # Shifting cache
            keys_to_keep = self.key_cache[layer_idx][
                :, :, -self.window_length + self.num_sink_tokens + key_states.shape[-2] :
            ]

            # On RoPE models, we need to recompute the Key rotation as the tokens are shifted
            if using_rope:
                rerotation_cos, rerotation_sin = self._get_rerotation_cos_sin(
                    key_states, self._cos_cache[: self.window_length], self._sin_cache[: self.window_length]
                )
                if partial_rotation_size is not None:
                    keys_to_keep, keys_pass = (
                        keys_to_keep[..., :partial_rotation_size],
                        keys_to_keep[..., partial_rotation_size:],
                    )
                keys_to_keep = self._apply_key_rotary_pos_emb(keys_to_keep, rerotation_cos, rerotation_sin)
                if partial_rotation_size is not None:
                    keys_to_keep = torch.cat((keys_to_keep, keys_pass), dim=-1)

            # Concatenate sink tokens, shifted & rotated tokens (if needed), and new tokens
            sink_keys = self.key_cache[layer_idx][:, :, : self.num_sink_tokens]
            self.key_cache[layer_idx] = torch.cat([sink_keys, keys_to_keep, key_states], dim=-2)

            sink_values = self.value_cache[layer_idx][:, :, : self.num_sink_tokens]
            values_to_keep = self.value_cache[layer_idx][
                :, :, -self.window_length + self.num_sink_tokens + value_states.shape[-2] :
            ]
            self.value_cache[layer_idx] = torch.cat([sink_values, values_to_keep, value_states], dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]


class StaticCache(Cache):
    """
    Static Cache class to be used with `torch.compile(model)` and `torch.export()`.

    Parameters:
        config (`PretrainedConfig`):
            The configuration file defining the shape-related attributes required to initialize the static cache.
        batch_size (`int`):
            The batch size with which the model will be used. Note that a new instance must be instantiated if a
            smaller batch size is used. If you are manually setting the batch size, make sure to take into account the number of beams if you are running beam search
        max_cache_len (`int`):
            The maximum sequence length with which the model will be used.
        device (`torch.device` or `str`):
            The device on which the cache should be initialized. Should be the same as the layer.
        dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
            The default `dtype` to use when initializing the layer.
        layer_device_map(`Dict[int, Union[str, torch.device, int]]]`, `optional`):
            Mapping between the layers and its device. This is required when you are manually initializing the cache and the model is splitted between differents gpus.
            You can know which layers mapped to which device by checking the associated device_map: `model.hf_device_map`.

    Example:

        ```python
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM, StaticCache

        >>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

        >>> inputs = tokenizer(text="My name is Llama", return_tensors="pt")

        >>> # Prepare a cache class and pass it to model's forward
        >>> # Leave empty space for 10 new tokens, which can be used when calling forward iteratively 10 times to generate
        >>> max_generated_length = inputs.input_ids.shape[1] + 10
        >>> past_key_values = StaticCache(config=model.config, batch_size=1, max_cache_len=max_generated_length, device=model.device, dtype=model.dtype)
        >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
        >>> outputs.past_key_values # access cache filled with key/values from generation
        StaticCache()
        ```
    """

    # TODO (joao): remove `=None` in non-optional arguments in v4.46. Remove from `OBJECTS_TO_IGNORE` as well.
    def __init__(
        self,
        config: PretrainedConfig,
        batch_size: int = None,
        max_cache_len: int = None,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
        max_batch_size: Optional[int] = None,
        layer_device_map: Optional[Dict[int, Union[str, torch.device, int]]] = None,
    ) -> None:
        super().__init__()
        if batch_size is not None:
            logger.warning_once(
                f"The 'batch_size' argument of {self.__class__.__name__} is deprecated and will be removed in "
                "v4.49. Use the more precisely named 'max_batch_size' argument instead."
            )

        self.max_batch_size = batch_size or max_batch_size
        self.max_cache_len = config.max_position_embeddings if max_cache_len is None else max_cache_len

        # Some model define a custom `head_dim` != config.hidden_size // config.num_attention_heads
        self.head_dim = (
            config.head_dim if hasattr(config, "head_dim") else config.hidden_size // config.num_attention_heads
        )

        self.dtype = dtype
        self.num_key_value_heads = (
            config.num_attention_heads
            if getattr(config, "num_key_value_heads", None) is None
            else config.num_key_value_heads
        )

        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        # Note: There will be significant perf decrease if switching to use 5D tensors instead.
        cache_shape = (self.max_batch_size, self.num_key_value_heads, self.max_cache_len, self.head_dim)
        for idx in range(config.num_hidden_layers):
            if layer_device_map is not None:
                layer_device = layer_device_map[idx]
            else:
                layer_device = device
            new_layer_key_cache = torch.zeros(cache_shape, dtype=self.dtype, device=layer_device)
            new_layer_value_cache = torch.zeros(cache_shape, dtype=self.dtype, device=layer_device)
            # Notes:
            # 1. `mark_static_address` is used to tag the cache as an fixed data pointer, preventing cuda graph
            #     breaks when updating the cache. It can't be used if the cache code is being compiled (but in that case
            #     it is not needed anyway)
            # 2. `torch.export()` requires mutations to be registered as buffers.
            if not is_torchdynamo_compiling():
                self.register_buffer(f"key_cache_{idx}", torch.zeros(cache_shape, dtype=dtype, device=layer_device))
                self.register_buffer(f"value_cache_{idx}", torch.zeros(cache_shape, dtype=dtype, device=layer_device))
                new_layer_key_cache = getattr(self, f"key_cache_{idx}")
                new_layer_value_cache = getattr(self, f"value_cache_{idx}")
                torch._dynamo.mark_static_address(new_layer_key_cache)
                torch._dynamo.mark_static_address(new_layer_value_cache)
            self.key_cache.append(new_layer_key_cache)
            self.value_cache.append(new_layer_value_cache)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.
        It is VERY important to index using a tensor, otherwise you introduce a copy to the device.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. The `StaticCache` needs the `cache_position` input
                to know how where to write in the cache.

        Return:
            A tuple containing the updated key and value states.
        """

        cache_position = cache_kwargs.get("cache_position")

        k_out = self.key_cache[layer_idx]
        v_out = self.value_cache[layer_idx]
        key_states = key_states.to(k_out.dtype)
        value_states = value_states.to(v_out.dtype)

        if cache_position is None:
            k_out.copy_(key_states)
            v_out.copy_(value_states)
        else:
            # Note: here we use `tensor.index_copy_(dim, index, tensor)` that is equivalent to
            # `tensor[:, :, index] = tensor`, but the first one is compile-friendly and it does explicitly an in-place
            # operation, that avoids copies and uses less memory.
            try:
                k_out.index_copy_(2, cache_position, key_states)
                v_out.index_copy_(2, cache_position, value_states)
            except NotImplementedError:
                # The operator 'aten::index_copy.out' is not currently implemented for the MPS device.
                k_out[:, :, cache_position] = key_states
                v_out[:, :, cache_position] = value_states

        return k_out, v_out

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states that were seen by the model."""
        # Occupied cache == any slot in the 3rd dim (sequence length) holds a non-zero value. To save on compute, let's
        # limit the check to the first batch member and head dimension.
        # TODO: deprecate this function in favor of `cache_position`
        return (self.key_cache[layer_idx][0, 0].any(dim=-1)).sum()

    def get_max_cache_shape(self) -> Optional[int]:
        return self.max_cache_len

    def reset(self):
        """Resets the cache values while preserving the objects"""
        for layer_idx in range(len(self.key_cache)):
            # In-place ops prevent breaking the static address
            self.key_cache[layer_idx].zero_()
            self.value_cache[layer_idx].zero_()

    @property
    def batch_size(self):
        logger.warning_once(
            f"The 'batch_size' attribute of {self.__class__.__name__} is deprecated and will be removed in "
            "v4.49. Use the more precisely named 'self.max_batch_size' attribute instead."
        )
        return self.max_batch_size


class SlidingWindowCache(StaticCache):
    """
    Sliding Window Cache class to be used with `torch.compile` for models like Mistral that support sliding window attention.
    Every time when we try to update the cache, we compute the `indices` based on `cache_position >= self.config.sliding_window - 1`,
    if true(which means the cache can not hold all the old key value states and new states together because of the sliding window constraint),
    we need to do a cycle shift based on `indices` to replace the oldest states by the new key value states passed in.

    The `to_shift` is only true once we are above sliding_window. Thus with `sliding_window==64`:

    indices = (slicing + to_shift[-1].int()-1) % self.config.sliding_window
    tensor([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
        19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
        37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
        55, 56, 57, 58, 59, 60, 61, 62, 63,  0])

    We overwrite the cache using these, then we always write at cache_position (clamped to `sliding_window`)

    Parameters:
        config (`PretrainedConfig`):
            The configuration file defining the shape-related attributes required to initialize the static cache.
        batch_size (`int`):
            The batch size with which the model will be used. Note that a new instance must be instantiated if a
            smaller batch size is used.
        max_cache_len (`int`):
            The maximum sequence length with which the model will be used.
        device (`torch.device` or `str`):
            The device on which the cache should be initialized. Should be the same as the layer.
        dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
            The default `dtype` to use when initializing the layer.
        layer_device_map(`Dict[int, Union[str, torch.device, int]]]`, `optional`):
            Mapping between the layers and its device. This is required when you are manually initializing the cache and the model is splitted between differents gpus.
            You can know which layers mapped to which device by checking the associated device_map: `model.hf_device_map`.

    Example:

        ```python
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM, SlidingWindowCache

        >>> model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
        >>> tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")

        >>> inputs = tokenizer(text="My name is Mistral", return_tensors="pt")

        >>> # Prepare a cache class and pass it to model's forward
        >>> # Leave empty space for 10 new tokens, which can be used when calling forward iteratively 10 times to generate
        >>> max_generated_length = inputs.input_ids.shape[1] + 10
        >>> past_key_values = SlidingWindowCache(config=model.config, batch_size=1, max_cache_len=max_generated_length, device=model.device, dtype=model.dtype)
        >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
        >>> outputs.past_key_values # access cache filled with key/values from generation
        SlidingWindowCache()
        ```
    """

    is_sliding = True

    # TODO (joao): remove `=None` in non-optional arguments in v4.46. Remove from `OBJECTS_TO_IGNORE` as well.
    def __init__(
        self,
        config: PretrainedConfig,
        batch_size: int = None,
        max_cache_len: int = None,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
        max_batch_size: Optional[int] = None,
        layer_device_map: Optional[Dict[int, Union[str, torch.device, int]]] = None,
    ) -> None:
        if not hasattr(config, "sliding_window") or config.sliding_window is None:
            raise ValueError(
                "Setting `cache_implementation` to 'sliding_window' requires the model config supporting "
                "sliding window attention, please check if there is a `sliding_window` field in the model "
                "config and it's not set to None."
            )
        max_cache_len = min(config.sliding_window, max_cache_len)
        super().__init__(
            config=config,
            batch_size=batch_size,
            max_cache_len=max_cache_len,
            device=device,
            dtype=dtype,
            max_batch_size=max_batch_size,
            layer_device_map=layer_device_map,
        )

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor]:
        cache_position = cache_kwargs.get("cache_position")
        k_out = self.key_cache[layer_idx]
        v_out = self.value_cache[layer_idx]

        # assume this only happens in prefill phase when prompt length > sliding_window_size (= max_cache_len)
        if cache_position.shape[0] > self.max_cache_len:
            k_out = key_states[:, :, -self.max_cache_len :, :]
            v_out = value_states[:, :, -self.max_cache_len :, :]
            # Assumption: caches are all zeros at this point, `+=` is equivalent to `=` but compile-friendly
            self.key_cache[layer_idx] += k_out
            self.value_cache[layer_idx] += v_out
            # we should return the whole states instead of k_out, v_out to take the whole prompt
            # into consideration when building kv cache instead of just throwing away tokens outside of the window
            return key_states, value_states

        slicing = torch.ones(self.max_cache_len, dtype=torch.long, device=value_states.device).cumsum(0)
        cache_position = cache_position.clamp(0, self.max_cache_len - 1)
        to_shift = cache_position >= self.max_cache_len - 1
        indices = (slicing + to_shift[-1].int() - 1) % self.max_cache_len

        k_out = k_out[:, :, indices]
        v_out = v_out[:, :, indices]

        try:
            k_out.index_copy_(2, cache_position, key_states)
            v_out.index_copy_(2, cache_position, value_states)
        except NotImplementedError:
            # The operator 'aten::index_copy.out' is not currently implemented for the MPS device.
            k_out[:, :, cache_position] = key_states
            v_out[:, :, cache_position] = value_states

        # `_.zero()` followed by `+=` is equivalent `=`, but compile-friendly (without graph breaks due to assignment)
        self.key_cache[layer_idx].zero_()
        self.value_cache[layer_idx].zero_()

        self.key_cache[layer_idx] += k_out
        self.value_cache[layer_idx] += v_out

        return k_out, v_out

    def get_max_cache_shape(self) -> Optional[int]:
        return self.max_cache_len

    def reset(self):
        for layer_idx in range(len(self.key_cache)):
            # In-place ops prevent breaking the static address
            self.key_cache[layer_idx].zero_()
            self.value_cache[layer_idx].zero_()


class EncoderDecoderCache(Cache):
    """
    Base, abstract class for all encoder-decoder caches. Can be used to hold combinations of self-attention and
    cross-attention caches.

    Example:

        ```python
        >>> from transformers import AutoProcessor, AutoModelForCausalLM, DynamicCache, EncoderDecoderCache

        >>> model = AutoModelForCausalLM.from_pretrained("openai/whisper-small")
        >>> processor = AutoProcessor.from_pretrained("openai/whisper-small")

        >>> inputs = processor(audio=YOUR-AUDIO, return_tensors="pt")

        >>> # Prepare cache classes for encoder and decoder and pass it to model's forward
        >>> self_attention_cache = DynamicCache()
        >>> cross_attention_cache = DynamicCache()
        >>> past_key_values = EncoderDecoderCache(self_attention_cache, cross_attention_cache)
        >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
        >>> outputs.past_key_values # access cache filled with key/values from generation
        EncoderDecoderCache()
        ```

    """

    def __init__(self, self_attention_cache: Cache, cross_attention_cache: Cache):
        super().__init__()
        self.self_attention_cache = self_attention_cache
        self.cross_attention_cache = cross_attention_cache

        self.is_updated = {}
        for layer_idx in range(len(cross_attention_cache.key_cache)):
            self.is_updated[layer_idx] = bool(cross_attention_cache.get_seq_length(layer_idx) > 0)

    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            return (
                self.self_attention_cache.key_cache[layer_idx],
                self.self_attention_cache.value_cache[layer_idx],
                self.cross_attention_cache.key_cache[layer_idx],
                self.cross_attention_cache.value_cache[layer_idx],
            )
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.self_attention_cache)

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        """Converts the `EncoderDecoderCache` instance into  its equivalent in the legacy cache format."""
        legacy_cache = ()
        if len(self.cross_attention_cache) > 0:
            for self_attn, cross_attn in zip(
                self.self_attention_cache.to_legacy_cache(), self.cross_attention_cache.to_legacy_cache()
            ):
                legacy_cache += (self_attn + cross_attn,)
        else:
            legacy_cache = self.self_attention_cache.to_legacy_cache()
        return legacy_cache

    @classmethod
    def from_legacy_cache(
        cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    ) -> "EncoderDecoderCache":
        """Converts a cache in the legacy cache format into an equivalent `EncoderDecoderCache`."""
        cache = cls(
            self_attention_cache=DynamicCache(),
            cross_attention_cache=DynamicCache(),
        )
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx][:2]
                cache.self_attention_cache.update(key_states, value_states, layer_idx)
                if len(past_key_values[layer_idx]) > 2:
                    key_states, value_states = past_key_values[layer_idx][2:]
                    cache.cross_attention_cache.update(key_states, value_states, layer_idx)
                    cache.is_updated[layer_idx] = True
        return cache

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # check if empty list because in case of static cache it will be a tensors and we can't check `if not torch.Tensor`
        return self.self_attention_cache.get_seq_length(layer_idx)

    def reset(self):
        if hasattr(self.self_attention_cache, "reset"):
            self.self_attention_cache.reset()
        if hasattr(self.cross_attention_cache, "reset"):
            self.cross_attention_cache.reset()
        elif not hasattr(self.self_attention_cache, "reset") and not hasattr(self.cross_attention_cache, "reset"):
            raise ValueError(
                "Neither self nor cross-attention cache have valid `.reset()` methods. `.reset()` should "
                "only be called on compatible cache classes, such as `StaticCache` or `SlidingWindowCache`. "
                f"Got {self.self_attention_cache.__str__()} for the self attention cache and "
                f"{self.cross_attention_cache.__str__()} for the cross attention cache."
            )
        for layer_idx in self.is_updated:
            self.is_updated[layer_idx] = False

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        self.self_attention_cache.reorder_cache(beam_idx)
        self.cross_attention_cache.reorder_cache(beam_idx)

    def check_dynamic_cache(self, method: str):
        if not (
            isinstance(self.self_attention_cache, DynamicCache)
            and isinstance(self.cross_attention_cache, DynamicCache)
        ):
            raise ValueError(
                f"`{method}` is only defined for dynamic cache, got {self.self_attention_cache.__str__()} for the self "
                f"attention cache and {self.cross_attention_cache.__str__()} for the cross attention cache."
            )

    # TODO(gante, sanchit-gandhi): move following functionality into `.generate`
    def crop(self, maximum_length: int):
        """Crop the past key values up to a new `maximum_length` in terms of tokens. `maximum_length` can also be
        negative to remove `maximum_length` tokens. This is used in assisted decoding and contrastive search."""
        self.check_dynamic_cache(self.crop.__name__)
        self.self_attention_cache.crop(maximum_length)

    @deprecate_kwarg("num_hidden_layers", version="4.47.0")
    def batch_split(
        self, full_batch_size: int, split_size: int, num_hidden_layers: int = None
    ) -> "List[EncoderDecoderCache]":
        """Split the current instance into a list of `DynamicCache` by the batch size. This will be used by
        `_split_model_inputs()` in `generation.utils`"""
        self.check_dynamic_cache(self.batch_split.__name__)
        self_attention_cache = self.self_attention_cache.batch_split(full_batch_size, split_size)
        cross_attention_cache = self.cross_attention_cache.batch_split(full_batch_size, split_size)

        out = []
        for self_attn, cross_attn in zip(self_attention_cache, cross_attention_cache):
            out.append(EncoderDecoderCache(self_attn, cross_attn))
        return out

    @classmethod
    @deprecate_kwarg("num_hidden_layers", version="4.47.0")
    def from_batch_splits(
        cls, splits: List["EncoderDecoderCache"], num_hidden_layers: int = None
    ) -> "EncoderDecoderCache":
        """This is the opposite of the above `batch_split()` method. This will be used by `stack_model_outputs` in
        `generation.utils`"""
        self_attention_cache = DynamicCache()
        cross_attention_cache = DynamicCache()
        for idx in range(len(splits[0])):
            layer_keys = torch.cat([current.self_attention_cache.key_cache[idx] for current in splits], dim=0)
            layer_values = torch.cat([current.self_attention_cache.value_cache[idx] for current in splits], dim=0)
            self_attention_cache.update(layer_keys, layer_values, idx)

            layer_keys = torch.cat([current.cross_attention_cache.key_cache[idx] for current in splits], dim=0)
            layer_values = torch.cat([current.cross_attention_cache.value_cache[idx] for current in splits], dim=0)
            cross_attention_cache.update(layer_keys, layer_values, idx)
        return cls(self_attention_cache, cross_attention_cache)

    def batch_repeat_interleave(self, repeats: int):
        """Repeat the cache `repeats` times in the batch dimension. Used in contrastive search."""
        self.check_dynamic_cache(self.batch_repeat_interleave.__name__)
        self.self_attention_cache.batch_repeat_interleave(repeats)
        self.cross_attention_cache.batch_repeat_interleave(repeats)

    def batch_select_indices(self, indices: torch.Tensor):
        """Only keep the `indices` in the batch dimension of the cache. Used in contrastive search."""
        self.check_dynamic_cache(self.batch_select_indices.__name__)
        self.self_attention_cache.batch_select_indices(indices)
        self.cross_attention_cache.batch_select_indices(indices)


class HybridCache(Cache):
    """
    Hybrid Cache class to be used with `torch.compile` for Gemma2 models that alternate between a local sliding window attention
    and global attention in every other layer. Under the hood, Hybrid Cache leverages ["SlidingWindowCache"] for sliding window attention
    and ["StaticCache"] for global attention. For more information, see the documentation of each subcomponeent cache class.

    Parameters:
        config (`PretrainedConfig):
            The configuration file defining the shape-related attributes required to initialize the static cache.
        batch_size (`int`):
            The batch size with which the model will be used. Note that a new instance must be instantiated if a
            smaller batch size is used.
        max_cache_len (`int`):
            The maximum sequence length with which the model will be used.
        device (`torch.device` or `str`, *optional*, defaults to `"cpu"`):
            The device on which the cache should be initialized. Should be the same as the layer.
        dtype (torch.dtype, *optional*, defaults to `torch.float32`):
            The default `dtype` to use when initializing the layer.
        layer_device_map(`Dict[int, Union[str, torch.device, int]]]`, `optional`):
            Mapping between the layers and its device. This is required when you are manually initializing the cache and the model is splitted between differents gpus.
            You can know which layers mapped to which device by checking the associated device_map: `model.hf_device_map`.

    Example:

        ```python
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM, HybridCache

        >>> model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b")
        >>> tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")

        >>> inputs = tokenizer(text="My name is Gemma", return_tensors="pt")

        >>> # Prepare a cache class and pass it to model's forward
        >>> # Leave empty space for 10 new tokens, which can be used when calling forward iteratively 10 times to generate
        >>> max_generated_length = inputs.input_ids.shape[1] + 10
        >>> past_key_values = HybridCache(config=model.config, batch_size=1, max_cache_len=max_generated_length, device=model.device, dtype=model.dtype)
        >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
        >>> outputs.past_key_values # access cache filled with key/values from generation
        HybridCache()
        ```
    """

    # TODO (joao): remove `=None` in non-optional arguments in v4.46. Remove from `OBJECTS_TO_IGNORE` as well.
    def __init__(
        self,
        config: PretrainedConfig,
        batch_size: int = None,
        max_cache_len: int = None,
        device: Union[torch.device, str] = "cpu",
        dtype: torch.dtype = torch.float32,
        max_batch_size: Optional[int] = None,
        layer_device_map: Optional[Dict[int, Union[str, torch.device, int]]] = None,
    ) -> None:
        super().__init__()
        if batch_size is not None:
            logger.warning_once(
                f"The 'batch_size' argument of {self.__class__.__name__} is deprecated and will be removed in "
                "v4.49. Use the more precisely named 'max_batch_size' argument instead."
            )
        if not hasattr(config, "sliding_window") or config.sliding_window is None:
            raise ValueError(
                "Setting `cache_implementation` to 'sliding_window' requires the model config supporting "
                "sliding window attention, please check if there is a `sliding_window` field in the model "
                "config and it's not set to None."
            )
        self.max_cache_len = max_cache_len
        self.max_batch_size = batch_size or max_batch_size
        # Some model define a custom `head_dim` != config.hidden_size // config.num_attention_heads
        self.head_dim = (
            config.head_dim if hasattr(config, "head_dim") else config.hidden_size // config.num_attention_heads
        )

        self.dtype = dtype
        self.num_key_value_heads = (
            config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        )
        layer_switch = config.sliding_window_pattern if hasattr(config, "sliding_window_pattern") else 2  # 2 is for BC
        self.is_sliding = torch.tensor(
            [bool((i + 1) % layer_switch) for i in range(config.num_hidden_layers)], dtype=torch.bool, device=device
        )
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        global_cache_shape = (self.max_batch_size, self.num_key_value_heads, max_cache_len, self.head_dim)
        sliding_cache_shape = (
            self.max_batch_size,
            self.num_key_value_heads,
            min(config.sliding_window, max_cache_len),
            self.head_dim,
        )
        for i in range(config.num_hidden_layers):
            if layer_device_map is not None:
                layer_device = layer_device_map[i]
            else:
                layer_device = device
            # Note: `mark_static_address` is used to tag the cache as an fixed data pointer, preventing cuda graph
            # breaks when updating the cache.
            cache_shape = global_cache_shape if not self.is_sliding[i] else sliding_cache_shape
            new_layer_key_cache = torch.zeros(cache_shape, dtype=self.dtype, device=layer_device)
            new_layer_value_cache = torch.zeros(cache_shape, dtype=self.dtype, device=layer_device)
            torch._dynamo.mark_static_address(new_layer_key_cache)
            torch._dynamo.mark_static_address(new_layer_value_cache)
            self.key_cache.append(new_layer_key_cache)
            self.value_cache.append(new_layer_value_cache)

    def _sliding_update(self, cache_position, layer_idx, key_states, value_states, k_out, v_out, max_cache_len):
        if cache_position.shape[0] > max_cache_len:
            k_out = key_states[:, :, -max_cache_len:, :]
            v_out = value_states[:, :, -max_cache_len:, :]
            # Assumption: caches are all zeros at this point, `+=` is equivalent to `=` but compile-friendly
            self.key_cache[layer_idx] += k_out
            self.value_cache[layer_idx] += v_out
            # we should return the whole states instead of k_out, v_out to take the whole prompt
            # into consideration when building kv cache instead of just throwing away tokens outside of the window
            return key_states, value_states

        slicing = torch.ones(max_cache_len, dtype=torch.long, device=value_states.device).cumsum(0)
        cache_position = cache_position.clamp(0, max_cache_len - 1)
        to_shift = cache_position >= max_cache_len - 1
        indices = (slicing + to_shift[-1].int() - 1) % max_cache_len
        k_out = k_out[:, :, indices]
        v_out = v_out[:, :, indices]

        k_out[:, :, cache_position] = key_states
        v_out[:, :, cache_position] = value_states
        # `_.zero()` followed by `+=` is equivalent `=`, but compile-friendly (without graph breaks due to assignment)
        self.key_cache[layer_idx].zero_()
        self.value_cache[layer_idx].zero_()

        self.key_cache[layer_idx] += k_out
        self.value_cache[layer_idx] += v_out
        return k_out, v_out

    def _static_update(self, cache_position, layer_idx, key_states, value_states, k_out, v_out, max_cache_len):
        k_out[:, :, cache_position] = key_states
        v_out[:, :, cache_position] = value_states

        self.key_cache[layer_idx] = k_out
        self.value_cache[layer_idx] = v_out
        return k_out, v_out

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor]:
        cache_position = cache_kwargs.get("cache_position")
        sliding_window = cache_kwargs.get("sliding_window")
        k_out = self.key_cache[layer_idx]
        v_out = self.value_cache[layer_idx]
        if sliding_window:
            update_fn = self._sliding_update
        else:
            update_fn = self._static_update

        return update_fn(
            cache_position,
            layer_idx,
            key_states,
            value_states,
            k_out,
            v_out,
            k_out.shape[2],
        )

    def get_max_cache_shape(self) -> Optional[int]:
        return self.max_cache_len

    def get_seq_length(self, layer_idx: Optional[int] = 0):
        # Occupied cache == any slot in the 3rd dim (sequence length) holds a non-zero value. To save on compute, let's
        # limit the check to the first batch member and head dimension.
        # TODO: deprecate this function in favor of `cache_position`
        if layer_idx != 0:
            raise ValueError(
                "`get_seq_length` on `HybridCache` may get inconsistent results depending on the layer index. "
                "Using the `layer_idx` argument is not supported."
            )
        return (self.key_cache[layer_idx][0, 0].any(dim=-1)).sum()

    def reset(self):
        """Resets the cache values while preserving the objects"""
        for layer_idx in range(len(self.key_cache)):
            # In-place ops prevent breaking the static address
            self.key_cache[layer_idx].zero_()
            self.value_cache[layer_idx].zero_()

    @property
    def batch_size(self):
        logger.warning_once(
            f"The 'batch_size' attribute of {self.__class__.__name__} is deprecated and will be removed in "
            "v4.49. Use the more precisely named 'self.max_batch_size' attribute instead."
        )
        return self.max_batch_size


class MambaCache:
    """
    Cache for mamba model which does not have attention mechanism and key value states.

    Arguments:
        config (`PretrainedConfig):
            The configuration file defining the shape-related attributes required to initialize the static cache.
        batch_size (`int`):
            The batch size with which the model will be used. Note that a new instance must be instantiated if a
            smaller batch size is used.
        dtype (`torch.dtype`, *optional*, defaults to `torch.float16`):
            The default `dtype` to use when initializing the layer.
        device (`torch.device` or `str`, *optional*):
            The device on which the cache should be initialized. Should be the same as the layer.

    Attributes:
        dtype: (`torch.dtype`):
            The default `dtype` used to initializing the cache.
        intermediate_size: (`int`):
            Model's intermediate_size taken from config.
        ssm_state_size: (`int`):
            Model's state_size taken from config.
        conv_kernel_size: (`int`):
            Model's convolution kernel size taken from config
        conv_states: (`torch.Tensor`):
            A tensor of shape `[layer_idx, batch_size, intermediate_size, conv_kernel_size]` that holds convolutional states.
        ssm_states: (`torch.Tensor`):
            A tensor of shape `[layer_idx, batch_size, intermediate_size, ssm_state_size]` that holds ssm states

    Example:

        ```python
        >>> from transformers import AutoTokenizer, MambaForCausalLM, MambaCache

        >>> model = MambaForCausalLM.from_pretrained("state-spaces/mamba-130m-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")

        >>> inputs = tokenizer(text="My name is Mamba", return_tensors="pt")

        >>> # Prepare a cache class and pass it to model's forward
        >>> past_key_values = MambaCache(config=model.config, batch_size=1, device=model.device, dtype=model.dtype)
        >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
        >>> outputs.past_key_values
        MambaCache()
        ```
    """

    # TODO (joao): remove `=None` in non-optional arguments in v4.46. Remove from `OBJECTS_TO_IGNORE` as well.
    def __init__(
        self,
        config: PretrainedConfig,
        batch_size: int = None,
        dtype: torch.dtype = torch.float16,
        device: Optional[Union[torch.device, str]] = None,
        max_batch_size: Optional[int] = None,
    ):
        if batch_size is not None:
            logger.warning_once(
                f"The 'batch_size' argument of {self.__class__.__name__} is deprecated and will be removed in "
                "v4.49. Use the more precisely named 'max_batch_size' argument instead."
            )
        self.dtype = dtype
        self.max_batch_size = batch_size or max_batch_size
        self.intermediate_size = config.intermediate_size
        self.ssm_state_size = config.state_size
        self.conv_kernel_size = config.conv_kernel

        self.conv_states: torch.Tensor = torch.zeros(
            config.num_hidden_layers,
            self.max_batch_size,
            self.intermediate_size,
            self.conv_kernel_size,
            device=device,
            dtype=dtype,
        )
        self.ssm_states: torch.Tensor = torch.zeros(
            config.num_hidden_layers,
            self.max_batch_size,
            self.intermediate_size,
            self.ssm_state_size,
            device=device,
            dtype=dtype,
        )

        torch._dynamo.mark_static_address(self.conv_states)
        torch._dynamo.mark_static_address(self.ssm_states)

    def update_conv_state(
        self, layer_idx: int, new_conv_state: torch.Tensor, cache_position: torch.LongTensor
    ) -> torch.Tensor:
        conv_state = self.conv_states[layer_idx]
        cache_position = cache_position.clamp(0, self.conv_kernel_size - 1)

        conv_state = conv_state.roll(shifts=-1, dims=-1)
        conv_state[:, :, cache_position] = new_conv_state.to(device=conv_state.device, dtype=conv_state.dtype)
        self.conv_states[layer_idx].zero_()
        self.conv_states[layer_idx] += conv_state
        return self.conv_states[layer_idx]

    def update_ssm_state(self, layer_idx: int, new_ssm_state: torch.Tensor):
        self.ssm_states[layer_idx] = new_ssm_state.to(self.ssm_states.device)
        return self.ssm_states[layer_idx]

    def reset(self):
        self.conv_states.zero_()
        self.ssm_states.zero_()

    @property
    def batch_size(self):
        logger.warning_once(
            f"The 'batch_size' attribute of {self.__class__.__name__} is deprecated and will be removed in "
            "v4.49. Use the more precisely named 'self.max_batch_size' attribute instead."
        )
        return self.max_batch_size


class OffloadedStaticCache(StaticCache):
    """
    Static cache class to be used with `torch.compile(model)` that offloads to the CPU or
    another device.

    Args:
        config (`PretrainedConfig):
            The configuration file defining the shape-related attributes required to initialize
            the static cache.
        max_batch_size (`int`):
            The maximum batch size with which the model will be used.
        max_cache_len (`int`):
            The maximum sequence length with which the model will be used.
        device (`Union[str, torch.device]`):
            The device on which the cache should be initialized. Should be the same as the
            layer device.
        dtype (`torch.dtype`, *optional*):
            The default `dtype` to use when initializing the cache.
        offload_device (`Union[str, torch.device]`, *optional*, defaults to `cpu`):
            The device to offload to. Defaults to CPU.
        layer_device_map (`Dict[int, Union[str, torch.device, int]]`, *optional*):
            Mapping between the layers and its device. This is required when you are manually initializing the cache and the model is splitted between differents gpus.
            You can know which layers mapped to which device by checking the associated device_map: `model.hf_device_map`.

    Attributes:
        key_cache (`List[torch.Tensor]`):
            Off-loaded key cache tensors. First one will be on device, where-as the others are
            off-loaded.
        value_cache (`List[torch.Tensor]`):
            Off-loaded value cache tensors. First one will be on device, where-as the others are
            off-loaded.
        max_batch_size (`int`):
            The maximum batch size with which this cache can be used.
        max_cache_len (`int`):
            The maximum sequence length with which this cache can be used.
        device (`torch.device`):
            The device on which the cache is used.
        offload_device (`torch.device`):
            The device used to offload to.
        dtype (`torch.dtype`):
            The `dtype` used to initializing the cache.

    Example:

        ```python
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM, OffloadedStaticCache

        >>> model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
        >>> tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

        >>> inputs = tokenizer(text="My name is GPT2", return_tensors="pt")

        >>> # Prepare a cache class and pass it to model's forward
        >>> # Leave empty space for 10 new tokens, which can be used when calling forward iteratively 10 times to generate
        >>> max_generated_length = inputs.input_ids.shape[1] + 10
        >>> past_key_values = OffloadedStaticCache(config=model.config, max_batch_size=1, max_cache_len=max_generated_length, device=model.device, dtype=model.dtype)
        >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
        >>> past_kv_length = outputs.past_key_values # access cache filled with key/values from generation
        ```
    """

    def __init__(
        self,
        config: PretrainedConfig,
        max_batch_size: int,
        max_cache_len: Optional[int],
        device: Union[str, torch.device],
        dtype: Optional[torch.dtype] = None,
        offload_device: Union[str, torch.device] = torch.device("cpu"),
        layer_device_map: Optional[Dict[int, Union[str, torch.device, int]]] = None,
    ) -> None:
        self.max_batch_size = max_batch_size
        self.max_cache_len = config.max_position_embeddings if max_cache_len is None else max_cache_len
        self.device = torch.device(device) if layer_device_map is None else layer_device_map[0]
        self.offload_device = torch.device(offload_device)
        self.dtype = dtype if dtype is not None else torch.float32

        # Some model define a custom `head_dim` != config.hidden_size // config.num_attention_heads
        head_dim = config.head_dim if hasattr(config, "head_dim") else config.hidden_size // config.num_attention_heads

        num_key_value_heads = (
            config.num_attention_heads
            if getattr(config, "num_key_value_heads", None) is None
            else config.num_key_value_heads
        )

        cache_shape = (max_batch_size, num_key_value_heads, self.max_cache_len, head_dim)

        # Create offloaded CPU tensors.
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

        for i in range(config.num_hidden_layers):
            # First layer is always on-device.
            device = self.device if i == 0 else self.offload_device

            key_cache, value_cache = self._create_key_value_cache_tensors(cache_shape, device)

            self.key_cache.append(key_cache)
            self.value_cache.append(value_cache)

        # Create device tensors.
        self._device_key_cache: List[torch.Tensor] = []
        self._device_value_cache: List[torch.Tensor] = []

        for i in range(2):
            key_cache, value_cache = self._create_key_value_cache_tensors(cache_shape, self.device)

            self._device_key_cache.append(key_cache)
            self._device_value_cache.append(value_cache)

        # For backwards compatibility.
        # TODO(gante): Remove this.
        self._seen_tokens = 0

        # Create new CUDA stream for parallel prefetching.
        self._prefetch_stream = torch.cuda.Stream() if self.device.type == "cuda" else None

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.
        It is VERY important to index using a tensor, otherwise you introduce a copy to the device.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, *optional*):
                Additional arguments for the cache subclass. The `OffloadedStaticCache` needs the
                `cache_position` input to know how where to write in the cache.

        Return:
            A tuple containing the updated key and value states.
        """

        if layer_idx == 0:
            # Update seen tokens.
            # TODO(gante): Remove this.
            self._seen_tokens += key_states.shape[-2]

            # Always there.
            k_out = self.key_cache[0]
            v_out = self.value_cache[0]
        else:
            # Wait for prefetch stream.
            if self._prefetch_stream is not None:
                torch.cuda.default_stream(self.device).wait_stream(self._prefetch_stream)

            k_out = self._device_key_cache[layer_idx & 1]
            v_out = self._device_value_cache[layer_idx & 1]

        self._prefetch_layer(layer_idx + 1)

        cache_position = cache_kwargs.get("cache_position") if cache_kwargs is not None else None
        if cache_position is None:
            k_out.copy_(key_states)
            v_out.copy_(value_states)

            # Copy the values to the offloaded device as well.
            if layer_idx == 0:
                self.key_cache[layer_idx].copy_(key_states.to(self.offload_device))
                self.value_cache[layer_idx].copy_(value_states.to(self.offload_device))
        else:
            # Note: here we use `tensor.index_copy_(dim, index, tensor)` that is equivalent to
            # `tensor[:, :, index] = tensor`, but the first one is compile-friendly and it does
            # explicitly an in-place operation, that avoids copies and uses less memory.
            try:
                k_out.index_copy_(2, cache_position, key_states)
                v_out.index_copy_(2, cache_position, value_states)
            except NotImplementedError:
                # The operator 'aten::index_copy.out' is not currently implemented for the MPS
                # device.
                k_out[:, :, cache_position] = key_states
                v_out[:, :, cache_position] = value_states

            # Copy the values to the offloaded device as well.
            if layer_idx != 0:
                cache_position = cache_position.to(self.offload_device)
                key_states = key_states.to(self.offload_device)
                value_states = value_states.to(self.offload_device)

                try:
                    self.key_cache[layer_idx].index_copy_(2, cache_position, key_states)
                    self.value_cache[layer_idx].index_copy_(2, cache_position, value_states)
                except NotImplementedError:
                    # The operator 'aten::index_copy.out' is not currently implemented for the MPS
                    # device.
                    self.key_cache[layer_idx][:, :, cache_position] = key_states
                    self.value_cache[layer_idx][:, :, cache_position] = value_states

        return k_out, v_out

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states that were seen by the model."""

        # TODO(gante): Remove this.
        return self._seen_tokens

    def get_max_cache_shape(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states."""

        return self.max_cache_len

    def reset(self) -> None:
        """Resets the cache values while preserving the objects."""

        # For backwards compatibility.
        # TODO(gante): Remove this.
        self._seen_tokens = 0

        # Zero out cache.
        for layer_idx in range(len(self.key_cache)):
            # In-place ops prevent breaking the static address.
            self.key_cache[layer_idx].zero_()
            self.value_cache[layer_idx].zero_()

    @property
    def seen_tokens(self) -> int:
        # For backwards compatibility.
        # TODO(gante): Remove this.
        return self._seen_tokens

    def _create_key_value_cache_tensors(
        self, shape: Tuple[int, ...], device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Creates K/V cache tensors on a device. Pins memory for CPU tensors. Marks them as static
        addresses for non-CPU tensors.

        Args:
            shape (`Tuple[int, ...]`): Shape.
            device (`torch.device`): Device.

        Returns:
            Key and value cache tensors as a tuple.
        """

        is_cpu_device = device == torch.device("cpu")

        key_cache = torch.zeros(shape, dtype=self.dtype, device=device, pin_memory=is_cpu_device)
        value_cache = torch.zeros(shape, dtype=self.dtype, device=device, pin_memory=is_cpu_device)

        # Note: `mark_static_address` is used to tag the cache as a fixed data pointer,
        # preventing compiled graph breaks when updating the cache.
        torch._dynamo.mark_static_address(key_cache)
        torch._dynamo.mark_static_address(value_cache)

        return key_cache, value_cache

    def _prefetch_layer(self, layer_idx: int) -> None:
        """Prefetch a layer to the device. Needs to be called in order of layer indices."""

        # Don't fetch layers that do not exist.
        if layer_idx >= len(self.key_cache):
            return

        # Alternate between two on-device caches.
        if self._prefetch_stream is not None:
            with torch.cuda.stream(self._prefetch_stream):
                self._prefetch_layer_in_context(layer_idx)
        else:
            self._prefetch_layer_in_context(layer_idx)

    def _prefetch_layer_in_context(self, layer_idx: int) -> None:
        """Performs the actual copy of the layer to device cache."""

        self._device_key_cache[layer_idx & 1].copy_(self.key_cache[layer_idx], non_blocking=True)
        self._device_value_cache[layer_idx & 1].copy_(self.value_cache[layer_idx], non_blocking=True)
