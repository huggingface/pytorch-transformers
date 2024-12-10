# Copyright 2023 The HuggingFace Team. All rights reserved.
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
import importlib
import inspect
import warnings
from typing import Any, Dict, List, Optional, Union

from packaging import version

from ..utils import (
    check_peft_version,
    find_adapter_config_file,
    is_accelerate_available,
    is_peft_available,
    is_torch_available,
    logging,
)


if is_torch_available():
    import torch

if is_accelerate_available():
    from accelerate import dispatch_model
    from accelerate.utils import get_balanced_memory, infer_auto_device_map

# Minimum PEFT version supported for the integration
MIN_PEFT_VERSION = "0.5.0"


logger = logging.get_logger(__name__)


class PeftAdapterMixin:
    """
    A class containing all functions for loading and using adapters weights that are supported in PEFT library. For
    more details about adapters and injecting them on a transformer-based model, check out the documentation of PEFT
    library: https://huggingface.co/docs/peft/index

    Currently supported PEFT methods are all non-prefix tuning methods. Below is the list of supported PEFT methods
    that anyone can load, train and run with this mixin class:
    - Low Rank Adapters (LoRA): https://huggingface.co/docs/peft/conceptual_guides/lora
    - IA3: https://huggingface.co/docs/peft/conceptual_guides/ia3
    - AdaLora: https://arxiv.org/abs/2303.10512

    Other PEFT models such as prompt tuning, prompt learning are out of scope as these adapters are not "injectable"
    into a torch module. For using these methods, please refer to the usage guide of PEFT library.

    With this mixin, if the correct PEFT version is installed, it is possible to:

    - Load an adapter stored on a local path or in a remote Hub repository, and inject it in the model
    - Attach new adapters in the model and train them with Trainer or by your own.
    - Attach multiple adapters and iteratively activate / deactivate them
    - Activate / deactivate all adapters from the model.
    - Get the `state_dict` of the active adapter.
    """

    _hf_peft_config_loaded = False

    def load_adapter(
        self,
        peft_model_id: Optional[str] = None,
        adapter_name: Optional[str] = None,
        revision: Optional[str] = None,
        token: Optional[str] = None,
        device_map: Optional[str] = "auto",
        max_memory: Optional[str] = None,
        offload_folder: Optional[str] = None,
        offload_index: Optional[int] = None,
        peft_config: Dict[str, Any] = None,
        adapter_state_dict: Optional[Dict[str, "torch.Tensor"]] = None,
        low_cpu_mem_usage: bool = False,
        is_trainable: bool = False,
        adapter_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Load adapter weights from file or remote Hub folder. If you are not familiar with adapters and PEFT methods, we
        invite you to read more about them on PEFT official documentation: https://huggingface.co/docs/peft

        Requires peft as a backend to load the adapter weights.

        Args:
            peft_model_id (`str`, *optional*):
                The identifier of the model to look for on the Hub, or a local path to the saved adapter config file
                and adapter weights.
            adapter_name (`str`, *optional*):
                The adapter name to use. If not set, will use the default adapter.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.

                <Tip>

                To test a pull request you made on the Hub, you can pass `revision="refs/pr/<pr_number>"`.

                </Tip>

            token (`str`, `optional`):
                Whether to use authentication token to load the remote folder. Useful to load private repositories
                that are on HuggingFace Hub. You might need to call `huggingface-cli login` and paste your tokens to
                cache it.
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
            offload_folder (`str` or `os.PathLike`, `optional`):
                If the `device_map` contains any value `"disk"`, the folder where we will offload weights.
            offload_index (`int`, `optional`):
                `offload_index` argument to be passed to `accelerate.dispatch_model` method.
            peft_config (`Dict[str, Any]`, *optional*):
                The configuration of the adapter to add, supported adapters are non-prefix tuning and adaption prompts
                methods. This argument is used in case users directly pass PEFT state dicts
            adapter_state_dict (`Dict[str, torch.Tensor]`, *optional*):
                The state dict of the adapter to load. This argument is used in case users directly pass PEFT state
                dicts
            low_cpu_mem_usage (`bool`, *optional*, defaults to `False`):
                Reduce memory usage while loading the PEFT adapter. This should also speed up the loading process.
                Requires PEFT version 0.13.0 or higher.
            is_trainable (`bool`, *optional*, defaults to `False`):
                Whether the adapter should be trainable or not. If `False`, the adapter will be frozen and can only be
                used for inference.
            adapter_kwargs (`Dict[str, Any]`, *optional*):
                Additional keyword arguments passed along to the `from_pretrained` method of the adapter config and
                `find_adapter_config_file` method.
        """
        check_peft_version(min_version=MIN_PEFT_VERSION)

        # peft only supports low_cpu_mem_usage starting from v0.13.0
        peft_load_kwargs = {}
        if low_cpu_mem_usage:
            min_version_lcmu = "0.13.0"
            if version.parse(importlib.metadata.version("peft")) >= version.parse(min_version_lcmu):
                peft_load_kwargs["low_cpu_mem_usage"] = low_cpu_mem_usage
            else:
                raise ValueError(
                    "The version of PEFT you are using does not support `low_cpu_mem_usage` yet, "
                    f"please install PEFT >= {min_version_lcmu}."
                )

        adapter_name = adapter_name if adapter_name is not None else "default"
        if adapter_kwargs is None:
            adapter_kwargs = {}

        from peft import PeftConfig, inject_adapter_in_model, load_peft_weights
        from peft.utils import set_peft_model_state_dict

        if self._hf_peft_config_loaded and adapter_name in self.peft_config:
            raise ValueError(f"Adapter with name {adapter_name} already exists. Please use a different name.")

        if peft_model_id is None and (adapter_state_dict is None and peft_config is None):
            raise ValueError(
                "You should either pass a `peft_model_id` or a `peft_config` and `adapter_state_dict` to load an adapter."
            )

        if "device" not in adapter_kwargs:
            device = self.device if not hasattr(self, "hf_device_map") else list(self.hf_device_map.values())[0]
        else:
            device = adapter_kwargs.pop("device")

        # To avoid PEFT errors later on with safetensors.
        if isinstance(device, torch.device):
            device = str(device)

        # We keep `revision` in the signature for backward compatibility
        if revision is not None and "revision" not in adapter_kwargs:
            adapter_kwargs["revision"] = revision
        elif revision is not None and "revision" in adapter_kwargs and revision != adapter_kwargs["revision"]:
            logger.error(
                "You passed a `revision` argument both in `adapter_kwargs` and as a standalone argument. "
                "The one in `adapter_kwargs` will be used."
            )

        # Override token with adapter_kwargs' token
        if "token" in adapter_kwargs:
            token = adapter_kwargs.pop("token")

        if peft_config is None:
            adapter_config_file = find_adapter_config_file(
                peft_model_id,
                token=token,
                **adapter_kwargs,
            )

            if adapter_config_file is None:
                raise ValueError(
                    f"adapter model file not found in {peft_model_id}. Make sure you are passing the correct path to the "
                    "adapter model."
                )

            peft_config = PeftConfig.from_pretrained(
                peft_model_id,
                token=token,
                **adapter_kwargs,
            )
            peft_config.inference_mode = not is_trainable

        # Create and add fresh new adapters into the model.
        inject_adapter_in_model(peft_config, self, adapter_name, **peft_load_kwargs)

        if not self._hf_peft_config_loaded:
            self._hf_peft_config_loaded = True

        if peft_model_id is not None:
            adapter_state_dict = load_peft_weights(peft_model_id, token=token, device=device, **adapter_kwargs)

        # We need to pre-process the state dict to remove unneeded prefixes - for backward compatibility
        processed_adapter_state_dict = {}
        prefix = "base_model.model."
        for key, value in adapter_state_dict.items():
            if key.startswith(prefix):
                new_key = key[len(prefix) :]
            else:
                new_key = key
            processed_adapter_state_dict[new_key] = value

        # Load state dict
        incompatible_keys = set_peft_model_state_dict(
            self, processed_adapter_state_dict, adapter_name, **peft_load_kwargs
        )

        if incompatible_keys is not None:
            err_msg = ""
            origin_name = peft_model_id if peft_model_id is not None else "state_dict"
            # Check for unexpected keys.
            if hasattr(incompatible_keys, "unexpected_keys") and len(incompatible_keys.unexpected_keys) > 0:
                err_msg = (
                    f"Loading adapter weights from {origin_name} led to unexpected keys not found in the model: "
                    f"{', '.join(incompatible_keys.unexpected_keys)}. "
                )

            # Check for missing keys.
            missing_keys = getattr(incompatible_keys, "missing_keys", None)
            if missing_keys:
                # Filter missing keys specific to the current adapter, as missing base model keys are expected.
                lora_missing_keys = [k for k in missing_keys if "lora_" in k and adapter_name in k]
                if lora_missing_keys:
                    err_msg += (
                        f"Loading adapter weights from {origin_name} led to missing keys in the model: "
                        f"{', '.join(lora_missing_keys)}"
                    )

            if err_msg:
                logger.warning(err_msg)

        if peft_config.inference_mode:
            self.eval()

        # Re-dispatch model and hooks in case the model is offloaded to CPU / Disk.
        if (
            (getattr(self, "hf_device_map", None) is not None)
            and (len(set(self.hf_device_map.values()).intersection({"cpu", "disk"})) > 0)
            and len(self.peft_config) == 1
        ):
            self._dispatch_accelerate_model(
                device_map=device_map,
                max_memory=max_memory,
                offload_folder=offload_folder,
                offload_index=offload_index,
            )

    def add_adapter(self, adapter_config, adapter_name: Optional[str] = None) -> None:
        r"""
        If you are not familiar with adapters and PEFT methods, we invite you to read more about them on the PEFT
        official documentation: https://huggingface.co/docs/peft

        Adds a fresh new adapter to the current model for training purpose. If no adapter name is passed, a default
        name is assigned to the adapter to follow the convention of PEFT library (in PEFT we use "default" as the
        default adapter name).

        Args:
            adapter_config (`~peft.PeftConfig`):
                The configuration of the adapter to add, supported adapters are non-prefix tuning and adaption prompts
                methods
            adapter_name (`str`, *optional*, defaults to `"default"`):
                The name of the adapter to add. If no name is passed, a default name is assigned to the adapter.
        """
        check_peft_version(min_version=MIN_PEFT_VERSION)

        from peft import PeftConfig, inject_adapter_in_model

        adapter_name = adapter_name or "default"

        if not self._hf_peft_config_loaded:
            self._hf_peft_config_loaded = True
        elif adapter_name in self.peft_config:
            raise ValueError(f"Adapter with name {adapter_name} already exists. Please use a different name.")

        if not isinstance(adapter_config, PeftConfig):
            raise TypeError(f"adapter_config should be an instance of PeftConfig. Got {type(adapter_config)} instead.")

        # Retrieve the name or path of the model, one could also use self.config._name_or_path
        # but to be consistent with what we do in PEFT: https://github.com/huggingface/peft/blob/6e783780ca9df3a623992cc4d1d665001232eae0/src/peft/mapping.py#L100
        adapter_config.base_model_name_or_path = self.__dict__.get("name_or_path", None)
        inject_adapter_in_model(adapter_config, self, adapter_name)

        self.set_adapter(adapter_name)

    def set_adapter(self, adapter_name: Union[List[str], str]) -> None:
        """
        If you are not familiar with adapters and PEFT methods, we invite you to read more about them on the PEFT
        official documentation: https://huggingface.co/docs/peft

        Sets a specific adapter by forcing the model to use a that adapter and disable the other adapters.

        Args:
            adapter_name (`Union[List[str], str]`):
                The name of the adapter to set. Can be also a list of strings to set multiple adapters.
        """
        check_peft_version(min_version=MIN_PEFT_VERSION)
        if not self._hf_peft_config_loaded:
            raise ValueError("No adapter loaded. Please load an adapter first.")
        elif isinstance(adapter_name, list):
            missing = set(adapter_name) - set(self.peft_config)
            if len(missing) > 0:
                raise ValueError(
                    f"Following adapter(s) could not be found: {', '.join(missing)}. Make sure you are passing the correct adapter name(s)."
                    f" current loaded adapters are: {list(self.peft_config.keys())}"
                )
        elif adapter_name not in self.peft_config:
            raise ValueError(
                f"Adapter with name {adapter_name} not found. Please pass the correct adapter name among {list(self.peft_config.keys())}"
            )

        from peft.tuners.tuners_utils import BaseTunerLayer
        from peft.utils import ModulesToSaveWrapper

        _adapters_has_been_set = False

        for _, module in self.named_modules():
            if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
                # For backward compatbility with previous PEFT versions
                if hasattr(module, "set_adapter"):
                    module.set_adapter(adapter_name)
                else:
                    module.active_adapter = adapter_name
                _adapters_has_been_set = True

        if not _adapters_has_been_set:
            raise ValueError(
                "Did not succeeded in setting the adapter. Please make sure you are using a model that supports adapters."
            )

    def disable_adapters(self) -> None:
        r"""
        If you are not familiar with adapters and PEFT methods, we invite you to read more about them on the PEFT
        official documentation: https://huggingface.co/docs/peft

        Disable all adapters that are attached to the model. This leads to inferring with the base model only.
        """
        check_peft_version(min_version=MIN_PEFT_VERSION)

        if not self._hf_peft_config_loaded:
            raise ValueError("No adapter loaded. Please load an adapter first.")

        from peft.tuners.tuners_utils import BaseTunerLayer
        from peft.utils import ModulesToSaveWrapper

        for _, module in self.named_modules():
            if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
                # The recent version of PEFT need to call `enable_adapters` instead
                if hasattr(module, "enable_adapters"):
                    module.enable_adapters(enabled=False)
                else:
                    module.disable_adapters = True

    def enable_adapters(self) -> None:
        """
        If you are not familiar with adapters and PEFT methods, we invite you to read more about them on the PEFT
        official documentation: https://huggingface.co/docs/peft

        Enable adapters that are attached to the model.
        """
        check_peft_version(min_version=MIN_PEFT_VERSION)

        if not self._hf_peft_config_loaded:
            raise ValueError("No adapter loaded. Please load an adapter first.")

        from peft.tuners.tuners_utils import BaseTunerLayer

        for _, module in self.named_modules():
            if isinstance(module, BaseTunerLayer):
                # The recent version of PEFT need to call `enable_adapters` instead
                if hasattr(module, "enable_adapters"):
                    module.enable_adapters(enabled=True)
                else:
                    module.disable_adapters = False

    def active_adapters(self) -> List[str]:
        """
        If you are not familiar with adapters and PEFT methods, we invite you to read more about them on the PEFT
        official documentation: https://huggingface.co/docs/peft

        Gets the current active adapters of the model. In case of multi-adapter inference (combining multiple adapters
        for inference) returns the list of all active adapters so that users can deal with them accordingly.

        For previous PEFT versions (that does not support multi-adapter inference), `module.active_adapter` will return
        a single string.
        """
        check_peft_version(min_version=MIN_PEFT_VERSION)

        if not is_peft_available():
            raise ImportError("PEFT is not available. Please install PEFT to use this function: `pip install peft`.")

        if not self._hf_peft_config_loaded:
            raise ValueError("No adapter loaded. Please load an adapter first.")

        from peft.tuners.tuners_utils import BaseTunerLayer

        for _, module in self.named_modules():
            if isinstance(module, BaseTunerLayer):
                active_adapters = module.active_adapter
                break

        # For previous PEFT versions
        if isinstance(active_adapters, str):
            active_adapters = [active_adapters]

        return active_adapters

    def active_adapter(self) -> str:
        warnings.warn(
            "The `active_adapter` method is deprecated and will be removed in a future version.", FutureWarning
        )

        return self.active_adapters()[0]

    def get_adapter_state_dict(self, adapter_name: Optional[str] = None) -> dict:
        """
        If you are not familiar with adapters and PEFT methods, we invite you to read more about them on the PEFT
        official documentation: https://huggingface.co/docs/peft

        Gets the adapter state dict that should only contain the weights tensors of the specified adapter_name adapter.
        If no adapter_name is passed, the active adapter is used.

        Args:
            adapter_name (`str`, *optional*):
                The name of the adapter to get the state dict from. If no name is passed, the active adapter is used.
        """
        check_peft_version(min_version=MIN_PEFT_VERSION)

        if not self._hf_peft_config_loaded:
            raise ValueError("No adapter loaded. Please load an adapter first.")

        from peft import get_peft_model_state_dict

        if adapter_name is None:
            adapter_name = self.active_adapters()[0]

        adapter_state_dict = get_peft_model_state_dict(self, adapter_name=adapter_name)
        return adapter_state_dict

    def _dispatch_accelerate_model(
        self,
        device_map: str,
        max_memory: Optional[int] = None,
        offload_folder: Optional[str] = None,
        offload_index: Optional[int] = None,
    ) -> None:
        """
        Optional re-dispatch the model and attach new hooks to the model in case the model has been loaded with
        accelerate (i.e. with `device_map=xxx`)

        Args:
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
            offload_index (`int`, *optional*):
                The offload_index argument to be passed to `accelerate.dispatch_model` method.
        """
        dispatch_model_kwargs = {}
        # Safety checker for previous `accelerate` versions
        # `offload_index` was introduced in https://github.com/huggingface/accelerate/pull/873/
        if "offload_index" in inspect.signature(dispatch_model).parameters:
            dispatch_model_kwargs["offload_index"] = offload_index

        no_split_module_classes = self._no_split_modules

        if device_map != "sequential":
            max_memory = get_balanced_memory(
                self,
                max_memory=max_memory,
                no_split_module_classes=no_split_module_classes,
                low_zero=(device_map == "balanced_low_0"),
            )
        if isinstance(device_map, str):
            device_map = infer_auto_device_map(
                self, max_memory=max_memory, no_split_module_classes=no_split_module_classes
            )
        dispatch_model(
            self,
            device_map=device_map,
            offload_dir=offload_folder,
            **dispatch_model_kwargs,
        )
