#!/usr/bin/env python
# coding=utf-8

# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
import copy
import importlib.metadata
import json
import os
from dataclasses import dataclass
from enum import Enum
from inspect import Parameter, signature
from typing import Any, Dict, List, Optional, Union

from packaging import version

from ..utils import (
    is_auto_awq_available,
    is_gptqmodel_available,
    is_hqq_available,
    is_torch_available,
    is_torchao_available,
    logging,
)
from .import_utils import is_auto_gptq_available


if is_torch_available():
    import torch

logger = logging.get_logger(__name__)


class QuantizationMethod(str, Enum):
    BITS_AND_BYTES = "bitsandbytes"
    GPTQ = "gptq"
    AWQ = "awq"
    AQLM = "aqlm"
    VPTQ = "vptq"
    QUANTO = "quanto"
    EETQ = "eetq"
    HIGGS = "higgs"
    HQQ = "hqq"
    COMPRESSED_TENSORS = "compressed-tensors"
    FBGEMM_FP8 = "fbgemm_fp8"
    TORCHAO = "torchao"
    BITNET = "bitnet"


class AWQLinearVersion(str, Enum):
    GEMM = "gemm"
    GEMV = "gemv"
    EXLLAMA = "exllama"
    IPEX = "ipex"

    @staticmethod
    def from_str(version: str):
        version = version.lower()
        if version == "gemm":
            return AWQLinearVersion.GEMM
        elif version == "gemv":
            return AWQLinearVersion.GEMV
        elif version == "exllama":
            return AWQLinearVersion.EXLLAMA
        elif version == "ipex":
            return AWQLinearVersion.IPEX
        else:
            raise ValueError(f"Unknown AWQLinearVersion {version}")


class AwqBackendPackingMethod(str, Enum):
    AUTOAWQ = "autoawq"
    LLMAWQ = "llm-awq"


@dataclass
class QuantizationConfigMixin:
    """
    Mixin class for quantization config
    """

    quant_method: QuantizationMethod

    @classmethod
    def from_dict(cls, config_dict, return_unused_kwargs=False, **kwargs):
        """
        Instantiates a [`QuantizationConfigMixin`] from a Python dictionary of parameters.

        Args:
            config_dict (`Dict[str, Any]`):
                Dictionary that will be used to instantiate the configuration object.
            return_unused_kwargs (`bool`,*optional*, defaults to `False`):
                Whether or not to return a list of unused keyword arguments. Used for `from_pretrained` method in
                `PreTrainedModel`.
            kwargs (`Dict[str, Any]`):
                Additional parameters from which to initialize the configuration object.

        Returns:
            [`QuantizationConfigMixin`]: The configuration object instantiated from those parameters.
        """
        config = cls(**config_dict)

        to_remove = []
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)

        if return_unused_kwargs:
            return config, kwargs
        else:
            return config

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

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary. Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        return copy.deepcopy(self.__dict__)

    def __iter__(self):
        """allows `dict(obj)` for situations where obj may be a dict or QuantizationConfigMixin"""
        for attr, value in copy.deepcopy(self.__dict__).items():
            yield attr, value

    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string()}"

    def to_json_string(self, use_diff: bool = True) -> str:
        """
        Serializes this instance to a JSON string.

        Args:
            use_diff (`bool`, *optional*, defaults to `True`):
                If set to `True`, only the difference between the config instance and the default `PretrainedConfig()`
                is serialized to JSON string.

        Returns:
            `str`: String containing all the attributes that make up this configuration instance in JSON format.
        """
        if use_diff is True:
            config_dict = self.to_diff_dict()
        else:
            config_dict = self.to_dict()
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

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
class HqqConfig(QuantizationConfigMixin):
    """
    This is wrapper around hqq's BaseQuantizeConfig.

    Args:
        nbits (`int`, *optional*, defaults to 4):
            Number of bits. Supported values are (8, 4, 3, 2, 1).
        group_size (`int`, *optional*, defaults to 64):
            Group-size value. Supported values are any value that is divisble by weight.shape[axis]).
        view_as_float (`bool`, *optional*, defaults to `False`):
            View the quantized weight as float (used in distributed training) if set to `True`.
        axis (`Optional[int]`, *optional*):
            Axis along which grouping is performed. Supported values are 0 or 1.
        dynamic_config (dict, *optional*):
            Parameters for dynamic configuration. The key is the name tag of the layer and the value is a quantization config.
            If set, each layer specified by its id will use its dedicated quantization configuration.
        skip_modules (`List[str]`, *optional*, defaults to `['lm_head']`):
            List of `nn.Linear` layers to skip.
        kwargs (`Dict[str, Any]`, *optional*):
            Additional parameters from which to initialize the configuration object.
    """

    def __init__(
        self,
        nbits: int = 4,
        group_size: int = 64,
        view_as_float: bool = False,
        axis: Optional[int] = None,
        dynamic_config: Optional[dict] = None,
        skip_modules: List[str] = ["lm_head"],
        **kwargs,
    ):
        if is_hqq_available():
            from hqq.core.quantize import BaseQuantizeConfig as HQQBaseQuantizeConfig
        else:
            raise ImportError(
                "A valid HQQ version (>=0.2.1) is not available. Please follow the instructions to install it: `https://github.com/mobiusml/hqq/`."
            )

        for deprecated_key in ["quant_zero", "quant_scale", "offload_meta"]:
            if deprecated_key in kwargs:
                logger.info(
                    deprecated_key + " is deprecated. This parameter will be ignored in quantization settings."
                )

        if axis is None:
            axis = 1
            logger.info("Setting axis=1 as faster backends such as TorchAO or BitBlas are only compatible with it.")

        if axis not in [0, 1]:
            raise ValueError("Invalid axis value. Only 0 and 1 are allowed.")

        if dynamic_config is not None:
            self.quant_config = {}
            for key in dynamic_config:
                self.quant_config[key] = HQQBaseQuantizeConfig(**dynamic_config[key])
        else:
            self.quant_config = HQQBaseQuantizeConfig(
                **{
                    "nbits": nbits,
                    "group_size": group_size,
                    "view_as_float": view_as_float,
                    "axis": axis,
                }
            )

        self.quant_method = QuantizationMethod.HQQ
        self.skip_modules = skip_modules

        self.post_init()

    def post_init(self):
        r"""
        Safety checker that arguments are correct - also replaces some NoneType arguments with their default values.
        """
        pass

    @classmethod
    def from_dict(cls, config: Dict[str, Any]):
        """
        Override from_dict, used in AutoQuantizationConfig.from_dict in quantizers/auto.py
        """
        instance = cls()
        instance.quant_config = config["quant_config"]
        instance.skip_modules = config["skip_modules"]
        return instance

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary. Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        return {
            "quant_config": self.quant_config,
            "quant_method": self.quant_method,
            "skip_modules": self.skip_modules,
        }

    def __repr__(self):
        config_dict = self.to_dict()
        return f"{self.__class__.__name__} {json.dumps(config_dict, indent=2, sort_keys=True)}\n"

    def to_diff_dict(self) -> Dict[str, Any]:
        """
        Removes all attributes from config which correspond to the default config attributes for better readability and
        serializes to a Python dictionary.
        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        config_dict = self.to_dict()

        # get the default config dict
        default_config_dict = HqqConfig().to_dict()

        serializable_config_dict = {}

        # only serialize values that differ from the default config
        for key, value in config_dict.items():
            if value != default_config_dict[key]:
                serializable_config_dict[key] = value

        return serializable_config_dict


@dataclass
class BitsAndBytesConfig(QuantizationConfigMixin):
    """
    This is a wrapper class about all possible attributes and features that you can play with a model that has been
    loaded using `bitsandbytes`.

    This replaces `load_in_8bit` or `load_in_4bit`therefore both options are mutually exclusive.

    Currently only supports `LLM.int8()`, `FP4`, and `NF4` quantization. If more methods are added to `bitsandbytes`,
    then more arguments will be added to this class.

    Args:
        load_in_8bit (`bool`, *optional*, defaults to `False`):
            This flag is used to enable 8-bit quantization with LLM.int8().
        load_in_4bit (`bool`, *optional*, defaults to `False`):
            This flag is used to enable 4-bit quantization by replacing the Linear layers with FP4/NF4 layers from
            `bitsandbytes`.
        llm_int8_threshold (`float`, *optional*, defaults to 6.0):
            This corresponds to the outlier threshold for outlier detection as described in `LLM.int8() : 8-bit Matrix
            Multiplication for Transformers at Scale` paper: https://arxiv.org/abs/2208.07339 Any hidden states value
            that is above this threshold will be considered an outlier and the operation on those values will be done
            in fp16. Values are usually normally distributed, that is, most values are in the range [-3.5, 3.5], but
            there are some exceptional systematic outliers that are very differently distributed for large models.
            These outliers are often in the interval [-60, -6] or [6, 60]. Int8 quantization works well for values of
            magnitude ~5, but beyond that, there is a significant performance penalty. A good default threshold is 6,
            but a lower threshold might be needed for more unstable models (small models, fine-tuning).
        llm_int8_skip_modules (`List[str]`, *optional*):
            An explicit list of the modules that we do not want to convert in 8-bit. This is useful for models such as
            Jukebox that has several heads in different places and not necessarily at the last position. For example
            for `CausalLM` models, the last `lm_head` is kept in its original `dtype`.
        llm_int8_enable_fp32_cpu_offload (`bool`, *optional*, defaults to `False`):
            This flag is used for advanced use cases and users that are aware of this feature. If you want to split
            your model in different parts and run some parts in int8 on GPU and some parts in fp32 on CPU, you can use
            this flag. This is useful for offloading large models such as `google/flan-t5-xxl`. Note that the int8
            operations will not be run on CPU.
        llm_int8_has_fp16_weight (`bool`, *optional*, defaults to `False`):
            This flag runs LLM.int8() with 16-bit main weights. This is useful for fine-tuning as the weights do not
            have to be converted back and forth for the backward pass.
        bnb_4bit_compute_dtype (`torch.dtype` or str, *optional*, defaults to `torch.float32`):
            This sets the computational type which might be different than the input type. For example, inputs might be
            fp32, but computation can be set to bf16 for speedups.
        bnb_4bit_quant_type (`str`,  *optional*, defaults to `"fp4"`):
            This sets the quantization data type in the bnb.nn.Linear4Bit layers. Options are FP4 and NF4 data types
            which are specified by `fp4` or `nf4`.
        bnb_4bit_use_double_quant (`bool`, *optional*, defaults to `False`):
            This flag is used for nested quantization where the quantization constants from the first quantization are
            quantized again.
        bnb_4bit_quant_storage (`torch.dtype` or str, *optional*, defaults to `torch.uint8`):
            This sets the storage type to pack the quanitzed 4-bit prarams.
        kwargs (`Dict[str, Any]`, *optional*):
            Additional parameters from which to initialize the configuration object.
    """

    def __init__(
        self,
        load_in_8bit=False,
        load_in_4bit=False,
        llm_int8_threshold=6.0,
        llm_int8_skip_modules=None,
        llm_int8_enable_fp32_cpu_offload=False,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=None,
        bnb_4bit_quant_type="fp4",
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_storage=None,
        **kwargs,
    ):
        self.quant_method = QuantizationMethod.BITS_AND_BYTES

        if load_in_4bit and load_in_8bit:
            raise ValueError("load_in_4bit and load_in_8bit are both True, but only one can be used at the same time")

        self._load_in_8bit = load_in_8bit
        self._load_in_4bit = load_in_4bit
        self.llm_int8_threshold = llm_int8_threshold
        self.llm_int8_skip_modules = llm_int8_skip_modules
        self.llm_int8_enable_fp32_cpu_offload = llm_int8_enable_fp32_cpu_offload
        self.llm_int8_has_fp16_weight = llm_int8_has_fp16_weight
        self.bnb_4bit_quant_type = bnb_4bit_quant_type
        self.bnb_4bit_use_double_quant = bnb_4bit_use_double_quant

        if bnb_4bit_compute_dtype is None:
            self.bnb_4bit_compute_dtype = torch.float32
        elif isinstance(bnb_4bit_compute_dtype, str):
            self.bnb_4bit_compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
        elif isinstance(bnb_4bit_compute_dtype, torch.dtype):
            self.bnb_4bit_compute_dtype = bnb_4bit_compute_dtype
        else:
            raise ValueError("bnb_4bit_compute_dtype must be a string or a torch.dtype")

        if bnb_4bit_quant_storage is None:
            self.bnb_4bit_quant_storage = torch.uint8
        elif isinstance(bnb_4bit_quant_storage, str):
            if bnb_4bit_quant_storage not in ["float16", "float32", "int8", "uint8", "float64", "bfloat16"]:
                raise ValueError(
                    "`bnb_4bit_quant_storage` must be a valid string (one of 'float16', 'float32', 'int8', 'uint8', 'float64', 'bfloat16') "
                )
            self.bnb_4bit_quant_storage = getattr(torch, bnb_4bit_quant_storage)
        elif isinstance(bnb_4bit_quant_storage, torch.dtype):
            self.bnb_4bit_quant_storage = bnb_4bit_quant_storage
        else:
            raise ValueError("bnb_4bit_quant_storage must be a string or a torch.dtype")

        if kwargs:
            logger.warning(f"Unused kwargs: {list(kwargs.keys())}. These kwargs are not used in {self.__class__}.")

        self.post_init()

    @property
    def load_in_4bit(self):
        return self._load_in_4bit

    @load_in_4bit.setter
    def load_in_4bit(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError("load_in_4bit must be a boolean")

        if self.load_in_8bit and value:
            raise ValueError("load_in_4bit and load_in_8bit are both True, but only one can be used at the same time")
        self._load_in_4bit = value

    @property
    def load_in_8bit(self):
        return self._load_in_8bit

    @load_in_8bit.setter
    def load_in_8bit(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError("load_in_8bit must be a boolean")

        if self.load_in_4bit and value:
            raise ValueError("load_in_4bit and load_in_8bit are both True, but only one can be used at the same time")
        self._load_in_8bit = value

    def post_init(self):
        r"""
        Safety checker that arguments are correct - also replaces some NoneType arguments with their default values.
        """
        if not isinstance(self.load_in_4bit, bool):
            raise TypeError("load_in_4bit must be a boolean")

        if not isinstance(self.load_in_8bit, bool):
            raise TypeError("load_in_8bit must be a boolean")

        if not isinstance(self.llm_int8_threshold, float):
            raise TypeError("llm_int8_threshold must be a float")

        if self.llm_int8_skip_modules is not None and not isinstance(self.llm_int8_skip_modules, list):
            raise TypeError("llm_int8_skip_modules must be a list of strings")
        if not isinstance(self.llm_int8_enable_fp32_cpu_offload, bool):
            raise TypeError("llm_int8_enable_fp32_cpu_offload must be a boolean")

        if not isinstance(self.llm_int8_has_fp16_weight, bool):
            raise TypeError("llm_int8_has_fp16_weight must be a boolean")

        if self.bnb_4bit_compute_dtype is not None and not isinstance(self.bnb_4bit_compute_dtype, torch.dtype):
            raise TypeError("bnb_4bit_compute_dtype must be torch.dtype")

        if not isinstance(self.bnb_4bit_quant_type, str):
            raise TypeError("bnb_4bit_quant_type must be a string")

        if not isinstance(self.bnb_4bit_use_double_quant, bool):
            raise TypeError("bnb_4bit_use_double_quant must be a boolean")

        if self.load_in_4bit and not version.parse(importlib.metadata.version("bitsandbytes")) >= version.parse(
            "0.39.0"
        ):
            raise ValueError(
                "4 bit quantization requires bitsandbytes>=0.39.0 - please upgrade your bitsandbytes version"
            )

    def is_quantizable(self):
        r"""
        Returns `True` if the model is quantizable, `False` otherwise.
        """
        return self.load_in_8bit or self.load_in_4bit

    def quantization_method(self):
        r"""
        This method returns the quantization method used for the model. If the model is not quantizable, it returns
        `None`.
        """
        if self.load_in_8bit:
            return "llm_int8"
        elif self.load_in_4bit and self.bnb_4bit_quant_type == "fp4":
            return "fp4"
        elif self.load_in_4bit and self.bnb_4bit_quant_type == "nf4":
            return "nf4"
        else:
            return None

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary. Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        output = copy.deepcopy(self.__dict__)
        output["bnb_4bit_compute_dtype"] = str(output["bnb_4bit_compute_dtype"]).split(".")[1]
        output["bnb_4bit_quant_storage"] = str(output["bnb_4bit_quant_storage"]).split(".")[1]
        output["load_in_4bit"] = self.load_in_4bit
        output["load_in_8bit"] = self.load_in_8bit

        return output

    def __repr__(self):
        config_dict = self.to_dict()
        return f"{self.__class__.__name__} {json.dumps(config_dict, indent=2, sort_keys=True)}\n"

    def to_diff_dict(self) -> Dict[str, Any]:
        """
        Removes all attributes from config which correspond to the default config attributes for better readability and
        serializes to a Python dictionary.

        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        config_dict = self.to_dict()

        # get the default config dict
        default_config_dict = BitsAndBytesConfig().to_dict()

        serializable_config_dict = {}

        # only serialize values that differ from the default config
        for key, value in config_dict.items():
            if value != default_config_dict[key]:
                serializable_config_dict[key] = value

        return serializable_config_dict


class ExllamaVersion(int, Enum):
    ONE = 1
    TWO = 2


@dataclass
class GPTQConfig(QuantizationConfigMixin):
    """
    This is a wrapper class about all possible attributes and features that you can play with a model that has been
    loaded using `optimum` api for gptq quantization relying on auto_gptq backend.

    Args:
        bits (`int`):
            The number of bits to quantize to, supported numbers are (2, 3, 4, 8).
        tokenizer (`str` or `PreTrainedTokenizerBase`, *optional*):
            The tokenizer used to process the dataset. You can pass either:
                - A custom tokenizer object.
                - A string, the *model id* of a predefined tokenizer hosted inside a model repo on huggingface.co.
                - A path to a *directory* containing vocabulary files required by the tokenizer, for instance saved
                    using the [`~PreTrainedTokenizer.save_pretrained`] method, e.g., `./my_model_directory/`.
        dataset (`Union[List[str]]`, *optional*):
            The dataset used for quantization. You can provide your own dataset in a list of string or just use the
            original datasets used in GPTQ paper ['wikitext2','c4','c4-new']
        group_size (`int`, *optional*, defaults to 128):
            The group size to use for quantization. Recommended value is 128 and -1 uses per-column quantization.
        damp_percent (`float`, *optional*, defaults to 0.1):
            The percent of the average Hessian diagonal to use for dampening. Recommended value is 0.1.
        desc_act (`bool`, *optional*, defaults to `False`):
            Whether to quantize columns in order of decreasing activation size. Setting it to False can significantly
            speed up inference but the perplexity may become slightly worse. Also known as act-order.
        sym (`bool`, *optional*, defaults to `True`):
            Whether to use symetric quantization.
        true_sequential (`bool`, *optional*, defaults to `True`):
            Whether to perform sequential quantization even within a single Transformer block. Instead of quantizing
            the entire block at once, we perform layer-wise quantization. As a result, each layer undergoes
            quantization using inputs that have passed through the previously quantized layers.
        checkpoint_format (`str`, *optional*, defaults to `"gptq"`):
            GPTQ weight format. `gptq`(v1) is supported by both gptqmodel and auto-gptq. `gptq_v2` is gptqmodel only.
        meta (`Dict[str, any]`, *optional*):
            Properties, such as tooling:version, that do not directly contributes to quantization or quant inference are stored in meta.
            i.e. `meta.quantizer`: ["optimum:_version_", "gptqmodel:_version_"]
        backend (`str`, *optional*):
            Controls which gptq kernel to be used. Valid values for gptqmodel are `auto`, `auto_trainable` and more. For auto-gptq, only
            valid value is None and `auto_trainable`. Ref gptqmodel backends: https://github.com/ModelCloud/GPTQModel/blob/main/gptqmodel/utils/backend.py
        use_cuda_fp16 (`bool`, *optional*, defaults to `False`):
            Whether or not to use optimized cuda kernel for fp16 model. Need to have model in fp16. Auto-gptq only.
        model_seqlen (`int`, *optional*):
            The maximum sequence length that the model can take.
        block_name_to_quantize (`str`, *optional*):
            The transformers block name to quantize. If None, we will infer the block name using common patterns (e.g. model.layers)
        module_name_preceding_first_block (`List[str]`, *optional*):
            The layers that are preceding the first Transformer block.
        batch_size (`int`, *optional*, defaults to 1):
            The batch size used when processing the dataset
        pad_token_id (`int`, *optional*):
            The pad token id. Needed to prepare the dataset when `batch_size` > 1.
        use_exllama (`bool`, *optional*):
            Whether to use exllama backend. Defaults to `True` if unset. Only works with `bits` = 4.
        max_input_length (`int`, *optional*):
            The maximum input length. This is needed to initialize a buffer that depends on the maximum expected input
            length. It is specific to the exllama backend with act-order.
        exllama_config (`Dict[str, Any]`, *optional*):
            The exllama config. You can specify the version of the exllama kernel through the `version` key. Defaults
            to `{"version": 1}` if unset.
        cache_block_outputs (`bool`, *optional*, defaults to `True`):
            Whether to cache block outputs to reuse as inputs for the succeeding block.
        modules_in_block_to_quantize (`List[List[str]]`, *optional*):
            List of list of module names to quantize in the specified block. This argument is useful to exclude certain linear modules from being quantized.
            The block to quantize can be specified by setting `block_name_to_quantize`. We will quantize each list sequentially. If not set, we will quantize all linear layers.
            Example: `modules_in_block_to_quantize =[["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"], ["self_attn.o_proj"]]`.
            In this example, we will first quantize the q,k,v layers simultaneously since they are independent.
            Then, we will quantize `self_attn.o_proj` layer with the q,k,v layers quantized. This way, we will get
            better results since it reflects the real input `self_attn.o_proj` will get when the model is quantized.
    """

    def __init__(
        self,
        bits: int,
        tokenizer: Any = None,
        dataset: Optional[Union[List[str], str]] = None,
        group_size: int = 128,
        damp_percent: float = 0.1,
        desc_act: bool = False,
        sym: bool = True,
        true_sequential: bool = True,
        checkpoint_format: str = "gptq",
        meta: Optional[Dict[str, any]] = None,
        backend: Optional[str] = None,
        use_cuda_fp16: bool = False,
        model_seqlen: Optional[int] = None,
        block_name_to_quantize: Optional[str] = None,
        module_name_preceding_first_block: Optional[List[str]] = None,
        batch_size: int = 1,
        pad_token_id: Optional[int] = None,
        use_exllama: Optional[bool] = None,
        max_input_length: Optional[int] = None,
        exllama_config: Optional[Dict[str, Any]] = None,
        cache_block_outputs: bool = True,
        modules_in_block_to_quantize: Optional[List[List[str]]] = None,
        **kwargs,
    ):
        self.quant_method = QuantizationMethod.GPTQ
        self.bits = bits
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.group_size = group_size
        self.damp_percent = damp_percent
        self.desc_act = desc_act
        self.sym = sym
        self.true_sequential = true_sequential
        self.checkpoint_format = checkpoint_format.lower()
        self.meta = meta
        self.backend = backend.lower() if isinstance(backend, str) else backend
        self.use_cuda_fp16 = use_cuda_fp16
        self.model_seqlen = model_seqlen
        self.block_name_to_quantize = block_name_to_quantize
        self.module_name_preceding_first_block = module_name_preceding_first_block
        self.batch_size = batch_size
        self.pad_token_id = pad_token_id
        self.use_exllama = use_exllama
        self.max_input_length = max_input_length
        self.exllama_config = exllama_config
        self.disable_exllama = kwargs.pop("disable_exllama", None)
        self.cache_block_outputs = cache_block_outputs
        self.modules_in_block_to_quantize = modules_in_block_to_quantize
        self.post_init()

    def get_loading_attributes(self):
        attibutes_dict = copy.deepcopy(self.__dict__)
        loading_attibutes = [
            "disable_exllama",
            "use_exllama",
            "exllama_config",
            "use_cuda_fp16",
            "max_input_length",
            "backend",
        ]
        loading_attibutes_dict = {i: j for i, j in attibutes_dict.items() if i in loading_attibutes}
        return loading_attibutes_dict

    def post_init(self):
        r"""
        Safety checker that arguments are correct
        """
        if self.bits not in [2, 3, 4, 8]:
            raise ValueError(f"Only support quantization to [2,3,4,8] bits but found {self.bits}")
        if self.group_size != -1 and self.group_size <= 0:
            raise ValueError("group_size must be greater than 0 or equal to -1")
        if not (0 < self.damp_percent < 1):
            raise ValueError("damp_percent must between 0 and 1.")
        if self.dataset is not None:
            if isinstance(self.dataset, str):
                if self.dataset in ["ptb", "ptb-new"]:
                    raise ValueError(
                        f"""{self.dataset} dataset was deprecated. You can only choose between
                        ['wikitext2','c4','c4-new']"""
                    )
                if self.dataset not in ["wikitext2", "c4", "c4-new"]:
                    raise ValueError(
                        f"""You have entered a string value for dataset. You can only choose between
                        ['wikitext2','c4','c4-new'], but we found {self.dataset}"""
                    )
            elif not isinstance(self.dataset, list):
                raise ValueError(
                    f"""dataset needs to be either a list of string or a value in
                    ['wikitext2','c4','c4-new'], but we found {self.dataset}"""
                )

        # make sure backend is back/forward compatible with both gptqmodel (full) and auto-gptq (partial)
        if is_gptqmodel_available():
            # convert auto-gptq control into gptqmodel backend
            if self.backend is None:
                self.backend = "auto_trainable" if self.use_exllama is not None and not self.use_exllama else "auto"
        else:
            # convert gptqmodel backend `auto_trainable` into auto-gptq control
            if self.backend == "auto_trainable":
                self.use_exllama = False

        # auto-gptq specific kernel control logic
        if self.disable_exllama is None and self.use_exllama is None:
            # New default behaviour
            self.use_exllama = True
        elif self.disable_exllama is not None and self.use_exllama is None:
            # Follow pattern of old config
            logger.warning(
                "Using `disable_exllama` is deprecated and will be removed in version 4.37. Use `use_exllama` instead and specify the version with `exllama_config`."
                "The value of `use_exllama` will be overwritten by `disable_exllama` passed in `GPTQConfig` or stored in your config file."
            )
            self.use_exllama = not self.disable_exllama
            self.disable_exllama = None
        elif self.disable_exllama is not None and self.use_exllama is not None:
            # Only happens if user explicitly passes in both arguments
            raise ValueError("Cannot specify both `disable_exllama` and `use_exllama`. Please use just `use_exllama`")

        if self.exllama_config is None:
            self.exllama_config = {"version": ExllamaVersion.ONE}
        else:
            if "version" not in self.exllama_config:
                raise ValueError("`exllama_config` needs to have a `version` key.")
            elif self.exllama_config["version"] not in [ExllamaVersion.ONE, ExllamaVersion.TWO]:
                exllama_version = self.exllama_config["version"]
                raise ValueError(
                    f"Only supported versions are in [ExllamaVersion.ONE, ExllamaVersion.TWO] - not recognized version {exllama_version}"
                )

        if self.bits == 4 and self.use_exllama:
            if self.exllama_config["version"] == ExllamaVersion.ONE:
                logger.info(
                    "You have activated exllama backend. Note that you can get better inference "
                    "speed using exllamav2 kernel by setting `exllama_config`."
                )
            elif self.exllama_config["version"] == ExllamaVersion.TWO:
                if is_auto_gptq_available():
                    optimum_version = version.parse(importlib.metadata.version("optimum"))
                    autogptq_version = version.parse(importlib.metadata.version("auto_gptq"))
                    if optimum_version <= version.parse("1.13.2") or autogptq_version <= version.parse("0.4.2"):
                        raise ValueError(
                            f"You need optimum > 1.13.2 and auto-gptq > 0.4.2 . Make sure to have that version installed - detected version : optimum {optimum_version} and autogptq {autogptq_version}"
                        )
        if self.modules_in_block_to_quantize is not None:
            optimum_version = version.parse(importlib.metadata.version("optimum"))
            if optimum_version < version.parse("1.15.0"):
                raise ValueError(
                    "You current version of `optimum` does not support `modules_in_block_to_quantize` quantization argument, please upgrade `optimum` package to a version superior than 1.15.0 ."
                )

    def to_dict(self):
        config_dict = super().to_dict()
        config_dict.pop("disable_exllama", None)
        return config_dict

    def to_dict_optimum(self):
        """
        Get compatible dict for optimum gptq config
        """
        quant_dict = self.to_dict()
        # make it compatible with optimum config
        quant_dict["disable_exllama"] = not self.use_exllama
        return quant_dict

    @classmethod
    def from_dict_optimum(cls, config_dict):
        """
        Get compatible class with optimum gptq config dict
        """

        if "disable_exllama" in config_dict:
            config_dict["use_exllama"] = not config_dict["disable_exllama"]
            # switch to None to not trigger the warning
            config_dict["disable_exllama"] = None

        config = cls(**config_dict)
        return config


@dataclass
class AwqConfig(QuantizationConfigMixin):
    """
    This is a wrapper class about all possible attributes and features that you can play with a model that has been
    loaded using `auto-awq` library awq quantization relying on auto_awq backend.

    Args:
        bits (`int`, *optional*, defaults to 4):
            The number of bits to quantize to.
        group_size (`int`, *optional*, defaults to 128):
            The group size to use for quantization. Recommended value is 128 and -1 uses per-column quantization.
        zero_point (`bool`, *optional*, defaults to `True`):
            Whether to use zero point quantization.
        version (`AWQLinearVersion`, *optional*, defaults to `AWQLinearVersion.GEMM`):
            The version of the quantization algorithm to use. GEMM is better for big batch_size (e.g. >= 8) otherwise,
            GEMV is better (e.g. < 8 ). GEMM models are compatible with Exllama kernels.
        backend (`AwqBackendPackingMethod`, *optional*, defaults to `AwqBackendPackingMethod.AUTOAWQ`):
            The quantization backend. Some models might be quantized using `llm-awq` backend. This is useful for users
            that quantize their own models using `llm-awq` library.
        do_fuse (`bool`, *optional*, defaults to `False`):
            Whether to fuse attention and mlp layers together for faster inference
        fuse_max_seq_len (`int`, *optional*):
            The Maximum sequence length to generate when using fusing.
        modules_to_fuse (`dict`, *optional*, default to `None`):
            Overwrite the natively supported fusing scheme with the one specified by the users.
        modules_to_not_convert (`list`, *optional*, default to `None`):
            The list of modules to not quantize, useful for quantizing models that explicitly require to have
            some modules left in their original precision (e.g. Whisper encoder, Llava encoder, Mixtral gate layers).
            Note you cannot quantize directly with transformers, please refer to `AutoAWQ` documentation for quantizing HF models.
        exllama_config (`Dict[str, Any]`, *optional*):
            You can specify the version of the exllama kernel through the `version` key, the maximum sequence
            length through the `max_input_len` key, and the maximum batch size through the `max_batch_size` key.
            Defaults to `{"version": 2, "max_input_len": 2048, "max_batch_size": 8}` if unset.
    """

    def __init__(
        self,
        bits: int = 4,
        group_size: int = 128,
        zero_point: bool = True,
        version: AWQLinearVersion = AWQLinearVersion.GEMM,
        backend: AwqBackendPackingMethod = AwqBackendPackingMethod.AUTOAWQ,
        do_fuse: Optional[bool] = None,
        fuse_max_seq_len: Optional[int] = None,
        modules_to_fuse: Optional[dict] = None,
        modules_to_not_convert: Optional[List] = None,
        exllama_config: Optional[Dict[str, int]] = None,
        **kwargs,
    ):
        self.quant_method = QuantizationMethod.AWQ

        self.bits = bits
        self.group_size = group_size
        self.zero_point = zero_point
        self.version = version
        self.backend = backend
        self.fuse_max_seq_len = fuse_max_seq_len
        self.modules_to_not_convert = modules_to_not_convert
        self.exllama_config = exllama_config

        self.modules_to_fuse = modules_to_fuse
        if do_fuse is None:
            self.do_fuse = modules_to_fuse is not None and len(modules_to_fuse) > 0
        else:
            self.do_fuse = do_fuse
        self.fuse_max_seq_len = fuse_max_seq_len

        self.post_init()

    def post_init(self):
        r"""
        Safety checker that arguments are correct
        """
        if self.backend not in [AwqBackendPackingMethod.AUTOAWQ, AwqBackendPackingMethod.LLMAWQ]:
            raise ValueError(
                f"Only supported quantization backends in {AwqBackendPackingMethod.AUTOAWQ} and {AwqBackendPackingMethod.LLMAWQ} - not recognized backend {self.backend}"
            )

        self.version = AWQLinearVersion.from_str(self.version)
        if self.version not in [
            AWQLinearVersion.GEMM,
            AWQLinearVersion.GEMV,
            AWQLinearVersion.EXLLAMA,
            AWQLinearVersion.IPEX,
        ]:
            raise ValueError(
                f"Only supported versions are in [AWQLinearVersion.GEMM, AWQLinearVersion.GEMV, AWQLinearVersion.EXLLAMA, AWQLinearVersion.IPEX] - not recognized version {self.version}"
            )

        if self.backend == AwqBackendPackingMethod.LLMAWQ:
            compute_capability = torch.cuda.get_device_capability()
            major, minor = compute_capability
            if major < 8:
                raise ValueError("LLM-AWQ backend is only supported on GPUs with compute capability >= 8.0")

        if self.do_fuse and self.fuse_max_seq_len is None:
            raise ValueError(
                "You cannot enable fused modules without specifying a `fuse_max_seq_len`, make sure to pass a valid `fuse_max_seq_len` for your usecase"
            )

        if self.do_fuse:
            awq_version_supports_fusing = False
            MIN_AWQ_VERSION = "0.1.7"
            if is_auto_awq_available():
                awq_version_supports_fusing = version.parse(importlib.metadata.version("autoawq")) >= version.parse(
                    MIN_AWQ_VERSION
                )

            if not awq_version_supports_fusing:
                raise ValueError(
                    f"You current version of `autoawq` does not support module fusing, please upgrade `autoawq` package to at least {MIN_AWQ_VERSION}."
                )

        if self.modules_to_not_convert is not None:
            awq_version_supports_non_conversion = False
            MIN_AWQ_VERSION = "0.1.8"
            if is_auto_awq_available():
                awq_version_supports_non_conversion = version.parse(
                    importlib.metadata.version("autoawq")
                ) >= version.parse(MIN_AWQ_VERSION)

            if not awq_version_supports_non_conversion:
                raise ValueError(
                    f"You current version of `autoawq` does not support module quantization skipping, please upgrade `autoawq` package to at least {MIN_AWQ_VERSION}."
                )

        if self.do_fuse and self.modules_to_fuse is not None:
            required_keys = [
                "hidden_size",
                "num_attention_heads",
                "num_key_value_heads",
                "mlp",
                "attention",
                "layernorm",
                "use_alibi",
            ]
            if not all(key in self.modules_to_fuse for key in required_keys):
                raise ValueError(
                    f"Required fields are missing in the fusing mapping, required fields are {required_keys}"
                )

        if self.version == AWQLinearVersion.EXLLAMA:
            awq_version_supports_exllama = False
            MIN_AWQ_VERSION = "0.2.0"
            if is_auto_awq_available():
                awq_version_supports_exllama = version.parse(importlib.metadata.version("autoawq")) >= version.parse(
                    MIN_AWQ_VERSION
                )

            if not awq_version_supports_exllama:
                raise ValueError(
                    f"You current version of `autoawq` does not support exllama backend, "
                    f"please upgrade `autoawq` package to at least {MIN_AWQ_VERSION}."
                )

            if self.exllama_config is None:
                self.exllama_config = {"version": ExllamaVersion.TWO, "max_input_len": 2048, "max_batch_size": 8}
            else:
                if "version" not in self.exllama_config:
                    raise ValueError("`exllama_config` needs to have a `version` key.")
                elif self.exllama_config["version"] not in [ExllamaVersion.ONE, ExllamaVersion.TWO]:
                    exllama_version = self.exllama_config["version"]
                    raise ValueError(
                        f"Only supported versions are in [ExllamaVersion.ONE, ExllamaVersion.TWO] - not recognized version {exllama_version}"
                    )

    def get_loading_attributes(self):
        attibutes_dict = copy.deepcopy(self.__dict__)
        loading_attibutes = ["version", "do_fuse", "modules_to_fuse", "fuse_max_seq_len", "exllama_config"]
        loading_attibutes_dict = {i: j for i, j in attibutes_dict.items() if i in loading_attibutes}
        return loading_attibutes_dict


@dataclass
class AqlmConfig(QuantizationConfigMixin):
    """
    This is a wrapper class about `aqlm` parameters.

    Args:
        in_group_size (`int`, *optional*, defaults to 8):
            The group size along the input dimension.
        out_group_size (`int`, *optional*, defaults to 1):
            The group size along the output dimension. It's recommended to always use 1.
        num_codebooks (`int`, *optional*, defaults to 1):
            Number of codebooks for the Additive Quantization procedure.
        nbits_per_codebook (`int`, *optional*, defaults to 16):
            Number of bits encoding a single codebook vector. Codebooks size is 2**nbits_per_codebook.
        linear_weights_not_to_quantize (`Optional[List[str]]`, *optional*):
            List of full paths of `nn.Linear` weight parameters that shall not be quantized.
        kwargs (`Dict[str, Any]`, *optional*):
            Additional parameters from which to initialize the configuration object.
    """

    def __init__(
        self,
        in_group_size: int = 8,
        out_group_size: int = 1,
        num_codebooks: int = 1,
        nbits_per_codebook: int = 16,
        linear_weights_not_to_quantize: Optional[List[str]] = None,
        **kwargs,
    ):
        self.quant_method = QuantizationMethod.AQLM
        self.in_group_size = in_group_size
        self.out_group_size = out_group_size
        self.num_codebooks = num_codebooks
        self.nbits_per_codebook = nbits_per_codebook
        self.linear_weights_not_to_quantize = linear_weights_not_to_quantize

        self.post_init()

    def post_init(self):
        r"""
        Safety checker that arguments are correct - also replaces some NoneType arguments with their default values.
        """
        if not isinstance(self.in_group_size, int):
            raise TypeError("in_group_size must be a float")
        if not isinstance(self.out_group_size, int):
            raise TypeError("out_group_size must be a float")
        if not isinstance(self.num_codebooks, int):
            raise TypeError("num_codebooks must be a float")
        if not isinstance(self.nbits_per_codebook, int):
            raise TypeError("nbits_per_codebook must be a float")

        if self.linear_weights_not_to_quantize is not None and not isinstance(
            self.linear_weights_not_to_quantize, list
        ):
            raise ValueError("linear_weights_not_to_quantize must be a list of strings")

        if self.linear_weights_not_to_quantize is None:
            self.linear_weights_not_to_quantize = []


@dataclass
class VptqLayerConfig(QuantizationConfigMixin):
    """
    This is used to explain vptq config params for each layer
    Args:
        enable_norm (`bool`, *optional*, defaults to `True`): to control if we have scale/bias for fp-weight
        enable_perm (`bool`, *optional*, defaults to `True`): to perm input_channel or not
        group_num (`int`, *optional*, defaults to `1`): how many single groups for vector-quantization
        group_size (`int`, *optional*, defaults to `-1`): depends on out-features
        indices_as_float (`bool`, *optional*, defaults to `False`): for Finetuning
        is_indice_packed (`bool`, *optional*, defaults to `True`): should always be True
        num_centroids (`list`, *optional*, defaults to `[-1, -1]`): centriod numbers of clusters
        num_res_centroids (`list`, *optional*, defaults to `[-1, -1]`): ditto for residual
        outlier_size (`int`, *optional*, defaults to `1`): outliers
        vector_lens (`list`, *optional*, defaults to `[-1, -1]`): centroid vector length in quantization
    """

    def __init__(
        self,
        enable_norm: bool = True,
        enable_perm: bool = True,
        group_num: int = 1,
        group_size: int = -1,
        in_features: int = -1,
        indices_as_float: bool = False,
        is_indice_packed: bool = True,
        num_centroids: tuple = [-1, -1],
        num_res_centroids: tuple = [-1, -1],
        out_features: int = -1,
        outlier_size: int = 0,
        vector_lens: tuple = [-1, -1],
        **kwargs,
    ):
        self.enable_norm = enable_norm
        self.enable_perm = enable_perm
        self.group_num = group_num
        self.group_size = group_size
        self.in_features = in_features
        self.indices_as_float = indices_as_float
        self.is_indice_packed = is_indice_packed
        self.num_centroids = num_centroids
        self.num_res_centroids = num_res_centroids
        self.out_features = out_features
        self.outlier_size = outlier_size
        self.vector_lens = vector_lens
        self.post_init()

    def post_init(self):
        r"""
        Safety checker that arguments are correct
        """
        if self.is_indice_packed is False:
            raise ValueError("is_indice_packed should always be True")


@dataclass
class VptqConfig(QuantizationConfigMixin):
    """
    This is a wrapper class about `vptq` parameters.

    Args:
        enable_proxy_error (`bool`, *optional*, defaults to `False`): calculate proxy error for each layer
        config_for_layers (`Dict`, *optional*, defaults to `{}`): quantization params for each layer
        shared_layer_config (`Dict`, *optional*, defaults to `{}`): shared quantization params among layers
        modules_to_not_convert (`list`, *optional*, default to `None`):
            The list of modules to not quantize, useful for quantizing models that explicitly require to have
            some modules left in their original precision (e.g. Whisper encoder, Llava encoder, Mixtral gate layers).
        kwargs (`Dict[str, Any]`, *optional*):
            Additional parameters from which to initialize the configuration object.
    """

    def __init__(
        self,
        enable_proxy_error: bool = False,
        config_for_layers: Dict[str, Any] = {},
        shared_layer_config: Dict[str, Any] = {},
        modules_to_not_convert: Optional[List] = None,
        **kwargs,
    ):
        self.quant_method = QuantizationMethod.VPTQ
        self.enable_proxy_error = enable_proxy_error
        self.config_for_layers: Dict[str, Any] = config_for_layers
        self.shared_layer_config: Dict[str, Any] = shared_layer_config
        self.modules_to_not_convert = modules_to_not_convert
        self.post_init()

    def post_init(self):
        r"""
        Safety checker that arguments are correct
        """
        for layer_name, layer_param in self.config_for_layers.items():
            VptqLayerConfig(**layer_param)
        if self.enable_proxy_error is True:
            raise ValueError("enable_proxy_error should always be False until we support training")


@dataclass
class QuantoConfig(QuantizationConfigMixin):
    """
    This is a wrapper class about all possible attributes and features that you can play with a model that has been
    loaded using `quanto`.

    Args:
        weights (`str`, *optional*, defaults to `"int8"`):
            The target dtype for the weights after quantization. Supported values are ("float8","int8","int4","int2")
        activations (`str`, *optional*):
            The target dtype for the activations after quantization. Supported values are (None,"int8","float8")
        modules_to_not_convert (`list`, *optional*, default to `None`):
            The list of modules to not quantize, useful for quantizing models that explicitly require to have
            some modules left in their original precision (e.g. Whisper encoder, Llava encoder, Mixtral gate layers).
    """

    def __init__(
        self,
        weights="int8",
        activations=None,
        modules_to_not_convert: Optional[List] = None,
        **kwargs,
    ):
        self.quant_method = QuantizationMethod.QUANTO
        self.weights = weights
        self.activations = activations
        self.modules_to_not_convert = modules_to_not_convert
        self.post_init()

    def post_init(self):
        r"""
        Safety checker that arguments are correct
        """
        accepted_weights = ["float8", "int8", "int4", "int2"]
        accepted_activations = [None, "int8", "float8"]
        if self.weights not in accepted_weights:
            raise ValueError(f"Only support weights in {accepted_weights} but found {self.weights}")
        if self.activations not in accepted_activations:
            raise ValueError(f"Only support weights in {accepted_activations} but found {self.activations}")


@dataclass
class EetqConfig(QuantizationConfigMixin):
    """
    This is a wrapper class about all possible attributes and features that you can play with a model that has been
    loaded using `eetq`.

    Args:
        weights (`str`, *optional*, defaults to `"int8"`):
            The target dtype for the weights. Supported value is only "int8"
        modules_to_not_convert (`list`, *optional*, default to `None`):
            The list of modules to not quantize, useful for quantizing models that explicitly require to have
            some modules left in their original precision.
    """

    def __init__(
        self,
        weights: str = "int8",
        modules_to_not_convert: Optional[List] = None,
        **kwargs,
    ):
        self.quant_method = QuantizationMethod.EETQ
        self.weights = weights
        self.modules_to_not_convert = modules_to_not_convert
        self.post_init()

    def post_init(self):
        r"""
        Safety checker that arguments are correct
        """
        accepted_weights = ["int8"]
        if self.weights not in accepted_weights:
            raise ValueError(f"Only support weights in {accepted_weights} but found {self.weights}")


class CompressedTensorsConfig(QuantizationConfigMixin):
    """
    This is a wrapper class that handles compressed-tensors quantization config options.
    It is a wrapper around `compressed_tensors.QuantizationConfig`
    Args:
        config_groups (`typing.Dict[str, typing.Union[ForwardRef('QuantizationScheme'), typing.List[str]]]`, *optional*):
            dictionary mapping group name to a quantization scheme definition
        format (`str`, *optional*, defaults to `"dense"`):
            format the model is represented as. Set `run_compressed` True to execute model as the
            compressed format if not `dense`
        quantization_status (`QuantizationStatus`, *optional*, defaults to `"initialized"`):
            status of model in the quantization lifecycle, ie 'initialized', 'calibration', 'frozen'
        kv_cache_scheme (`typing.Union[QuantizationArgs, NoneType]`, *optional*):
            specifies quantization of the kv cache. If None, kv cache is not quantized.
        global_compression_ratio (`typing.Union[float, NoneType]`, *optional*):
            0-1 float percentage of model compression
        ignore (`typing.Union[typing.List[str], NoneType]`, *optional*):
            layer names or types to not quantize, supports regex prefixed by 're:'
        sparsity_config (`typing.Dict[str, typing.Any]`, *optional*):
            configuration for sparsity compression
        quant_method (`str`, *optional*, defaults to `"compressed-tensors"`):
            do not override, should be compressed-tensors
        run_compressed (`bool`, *optional*, defaults to `True`): alter submodules (usually linear) in order to
            emulate compressed model execution if True, otherwise use default submodule
    """

    def __init__(
        self,
        config_groups: Dict[str, Union["QuantizationScheme", List[str]]] = None,  # noqa: F821
        format: str = "dense",
        quantization_status: "QuantizationStatus" = "initialized",  # noqa: F821
        kv_cache_scheme: Optional["QuantizationArgs"] = None,  # noqa: F821
        global_compression_ratio: Optional[float] = None,
        ignore: Optional[List[str]] = None,
        sparsity_config: Dict[str, Any] = None,
        quant_method: str = "compressed-tensors",
        run_compressed: bool = True,
        **kwargs,
    ):
        from compressed_tensors.config import SparsityCompressionConfig
        from compressed_tensors.quantization import QuantizationConfig

        self.quantization_config = None
        self.sparsity_config = None

        self.run_compressed = run_compressed

        # parse from dict to load nested QuantizationScheme objects
        if config_groups or kv_cache_scheme:
            self.quantization_config = QuantizationConfig.parse_obj(
                {
                    "config_groups": config_groups,
                    "quant_method": quant_method,
                    "format": format,
                    "quantization_status": quantization_status,
                    "kv_cache_scheme": kv_cache_scheme,
                    "global_compression_ratio": global_compression_ratio,
                    "ignore": ignore,
                    "run_compressed": run_compressed,
                    **kwargs,
                }
            )

        if sparsity_config:
            self.sparsity_config = SparsityCompressionConfig.load_from_registry(
                sparsity_config.get("format"), **sparsity_config
            )

        super().__init__(quant_method=QuantizationMethod.COMPRESSED_TENSORS)

    @classmethod
    def from_dict(cls, config_dict, return_unused_kwargs=False, **kwargs):
        """
        Instantiates a [`CompressedTensorsConfig`] from a Python dictionary of parameters.
        Optionally unwraps any args from the nested quantization_config

        Args:
            config_dict (`Dict[str, Any]`):
                Dictionary that will be used to instantiate the configuration object.
            return_unused_kwargs (`bool`,*optional*, defaults to `False`):
                Whether or not to return a list of unused keyword arguments. Used for `from_pretrained` method in
                `PreTrainedModel`.
            kwargs (`Dict[str, Any]`):
                Additional parameters from which to initialize the configuration object.

        Returns:
            [`QuantizationConfigMixin`]: The configuration object instantiated from those parameters.

        """

        if "quantization_config" in config_dict:
            config_dict = dict(
                sparsity_config=config_dict.get("sparsity_config"),
                **config_dict["quantization_config"],
            )

        return super().from_dict(config_dict, return_unused_kwargs=return_unused_kwargs, **kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """
        Quantization config to be added to config.json

        Serializes this instance to a Python dictionary. Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        quantization_config = {}
        if self.quantization_config is not None:
            quantization_config = self.quantization_config.dict()
        else:
            quantization_config["quant_method"] = QuantizationMethod.COMPRESSED_TENSORS

        if self.sparsity_config is not None:
            quantization_config["sparsity_config"] = self.sparsity_config.dict()
        else:
            quantization_config["sparsity_config"] = {}

        return quantization_config

    def to_diff_dict(self) -> Dict[str, Any]:
        """
        Removes all attributes from config which correspond to the default config attributes for better readability and
        serializes to a Python dictionary.
        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        config_dict = self.to_dict()

        # get the default config dict
        default_config_dict = CompressedTensorsConfig().to_dict()

        serializable_config_dict = {}

        # only serialize values that differ from the default config
        for key, value in config_dict.items():
            if value != default_config_dict[key]:
                serializable_config_dict[key] = value

        return serializable_config_dict

    def get_loading_attributes(self):
        return {"run_compressed": self.run_compressed}


@dataclass
class FbgemmFp8Config(QuantizationConfigMixin):
    """
    This is a wrapper class about all possible attributes and features that you can play with a model that has been
    loaded using fbgemm fp8 quantization.

    Args:
        activation_scale_ub (`float`, *optional*, defaults to 1200.0):
            The activation scale upper bound. This is used when quantizing the input activation.
        modules_to_not_convert (`list`, *optional*, default to `None`):
            The list of modules to not quantize, useful for quantizing models that explicitly require to have
            some modules left in their original precision.
    """

    def __init__(
        self,
        activation_scale_ub: float = 1200.0,
        modules_to_not_convert: Optional[List] = None,
        **kwargs,
    ):
        self.quant_method = QuantizationMethod.FBGEMM_FP8
        self.activation_scale_ub = activation_scale_ub
        self.modules_to_not_convert = modules_to_not_convert

    def get_loading_attributes(self):
        attibutes_dict = copy.deepcopy(self.__dict__)
        loading_attibutes = ["activation_scale_ub"]
        loading_attibutes_dict = {i: j for i, j in attibutes_dict.items() if i in loading_attibutes}
        return loading_attibutes_dict


@dataclass
class HiggsConfig(QuantizationConfigMixin):
    """
    HiggsConfig is a configuration class for quantization using the HIGGS method.

    Args:
        bits (int, *optional*, defaults to 4):
            Number of bits to use for quantization. Can be 2, 3 or 4. Default is 4.
        p (int, *optional*, defaults to 2):
            Quantization grid dimension. 1 and 2 are supported. 2 is always better in practice. Default is 2.
        modules_to_not_convert (`list`, *optional*, default to ["lm_head"]):
            List of linear layers that should not be quantized.
        hadamard_size (int, *optional*, defaults to 512):
            Hadamard size for the HIGGS method. Default is 512. Input dimension of matrices is padded to this value. Decreasing this below 512 will reduce the quality of the quantization.
        group_size (int, *optional*, defaults to 256):
            Group size for the HIGGS method. Can be 64, 128 or 256. Decreasing it barely affects the performance. Default is 256. Must be a divisor of hadamard_size.
    """

    def __init__(
        self,
        bits: int = 4,
        p: int = 2,
        modules_to_not_convert: Optional[List[str]] = None,
        hadamard_size: int = 512,
        group_size: int = 256,
        **kwargs,
    ):
        if modules_to_not_convert is None:
            modules_to_not_convert = ["lm_head"]
        self.quant_method = QuantizationMethod.HIGGS
        self.bits = bits
        self.p = p
        self.modules_to_not_convert = modules_to_not_convert
        self.hadamard_size = hadamard_size
        self.group_size = group_size

        self.post_init()

    def post_init(self):
        r"""
        Safety checker that arguments are correct - also replaces some NoneType arguments with their default values.
        """
        if self.bits not in [2, 3, 4]:
            raise ValueError("bits must be 2, 3, or 4")
        if self.p not in [1, 2]:
            raise ValueError("p must be 1 or 2. 2 is always better in practice")
        if self.group_size not in [64, 128, 256]:
            raise ValueError("group_size must be 64, 128, or 256")
        if self.hadamard_size % self.group_size != 0:
            raise ValueError("hadamard_size must be divisible by group_size")


@dataclass
class TorchAoConfig(QuantizationConfigMixin):
    """This is a config class for torchao quantization/sparsity techniques.

    Args:
        quant_type (`str`):
            The type of quantization we want to use, currently supporting: `int4_weight_only`, `int8_weight_only` and `int8_dynamic_activation_int8_weight`.
        modules_to_not_convert (`list`, *optional*, default to `None`):
            The list of modules to not quantize, useful for quantizing models that explicitly require to have
            some modules left in their original precision.
        kwargs (`Dict[str, Any]`, *optional*):
            The keyword arguments for the chosen type of quantization, for example, int4_weight_only quantization supports two keyword arguments
            `group_size` and `inner_k_tiles` currently. More API examples and documentation of arguments can be found in
            https://github.com/pytorch/ao/tree/main/torchao/quantization#other-available-quantization-techniques

    Example:

    ```python
    quantization_config = TorchAoConfig("int4_weight_only", group_size=32)
    # int4_weight_only quant is only working with *torch.bfloat16* dtype right now
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda", torch_dtype=torch.bfloat16, quantization_config=quantization_config)
    ```
    """

    def __init__(self, quant_type: str, modules_to_not_convert: Optional[List] = None, **kwargs):
        self.quant_method = QuantizationMethod.TORCHAO
        self.quant_type = quant_type
        self.modules_to_not_convert = modules_to_not_convert
        # when we load from serailized config, "quant_type_kwargs" will be the key
        if "quant_type_kwargs" in kwargs:
            self.quant_type_kwargs = kwargs["quant_type_kwargs"]
        else:
            self.quant_type_kwargs = kwargs

        self.post_init()

    def post_init(self):
        r"""
        Safety checker that arguments are correct - also replaces some NoneType arguments with their default values.
        """
        if is_torchao_available():
            if not version.parse(importlib.metadata.version("torchao")) >= version.parse("0.4.0"):
                raise ValueError("Requires torchao 0.4.0 version and above")
        else:
            raise ValueError(
                "TorchAoConfig requires torchao to be installed, please install with `pip install torchao`"
            )

        _STR_TO_METHOD = self._get_torchao_quant_type_to_method()
        if self.quant_type not in _STR_TO_METHOD.keys():
            raise ValueError(
                f"Requested quantization type: {self.quant_type} is not supported yet, please add support in TorchAoConfig and TorchAoHfQuantizer."
            )

        method = _STR_TO_METHOD[self.quant_type]
        sig = signature(method)
        all_kwargs = [
            param.name
            for param in sig.parameters.values()
            if param.kind in [Parameter.KEYWORD_ONLY, Parameter.POSITIONAL_OR_KEYWORD]
        ]
        for k in self.quant_type_kwargs:
            if k not in all_kwargs:
                raise ValueError(
                    f"Unexpected keyword arg: {k} for API: {method}, accepted keyword args are: {all_kwargs}"
                )

    def _get_torchao_quant_type_to_method(self):
        if is_torchao_available():
            from torchao.quantization import (
                int4_weight_only,
                int8_dynamic_activation_int8_weight,
                int8_weight_only,
            )

            return {
                "int4_weight_only": int4_weight_only,
                "int8_weight_only": int8_weight_only,
                "int8_dynamic_activation_int8_weight": int8_dynamic_activation_int8_weight,
            }
        else:
            raise ValueError(
                "TorchAoConfig requires torchao to be installed, please install with `pip install torchao`"
            )

    def get_apply_tensor_subclass(self):
        _STR_TO_METHOD = self._get_torchao_quant_type_to_method()
        return _STR_TO_METHOD[self.quant_type](**self.quant_type_kwargs)

    def __repr__(self):
        config_dict = self.to_dict()
        return f"{self.__class__.__name__} {json.dumps(config_dict, indent=2, sort_keys=True)}\n"


@dataclass
class BitNetConfig(QuantizationConfigMixin):
    def __init__(
        self,
        modules_to_not_convert: Optional[List] = None,
        **kwargs,
    ):
        self.quant_method = QuantizationMethod.BITNET
        self.modules_to_not_convert = modules_to_not_convert
        self.post_init()

    def post_init(self):
        r"""
        Safety checker that arguments are correct
        """
        pass
