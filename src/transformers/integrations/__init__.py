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
from typing import TYPE_CHECKING

from ..utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available


_import_structure = {
    "aqlm": ["replace_with_aqlm_linear"],
    "awq": [
        "fuse_awq_modules",
        "post_init_awq_exllama_modules",
        "post_init_awq_ipex_modules",
        "replace_quantization_scales",
        "replace_with_awq_linear",
    ],
    "bitnet": [
        "BitLinear",
        "pack_weights",
        "replace_with_bitnet_linear",
        "unpack_weights",
    ],
    "bitsandbytes": [
        "dequantize_and_replace",
        "get_keys_to_not_convert",
        "replace_8bit_linear",
        "replace_with_bnb_linear",
        "set_module_8bit_tensor_to_device",
        "set_module_quantized_tensor_to_device",
        "validate_bnb_backend_availability",
    ],
    "deepspeed": [
        "HfDeepSpeedConfig",
        "HfTrainerDeepSpeedConfig",
        "deepspeed_config",
        "deepspeed_init",
        "deepspeed_load_checkpoint",
        "deepspeed_optim_sched",
        "is_deepspeed_available",
        "is_deepspeed_zero3_enabled",
        "set_hf_deepspeed_config",
        "unset_hf_deepspeed_config",
    ],
    "eetq": ["replace_with_eetq_linear"],
    "fbgemm_fp8": ["FbgemmFp8Linear", "replace_with_fbgemm_fp8_linear"],
    "fsdp": ["is_fsdp_managed_module"],
    "ggml": [
        "GGUF_CONFIG_MAPPING",
        "GGUF_TOKENIZER_MAPPING",
        "_gguf_parse_value",
        "load_dequant_gguf_tensor",
        "load_gguf",
    ],
    "higgs": ["HiggsLinear", "dequantize_higgs", "quantize_with_higgs", "replace_with_higgs_linear"],
    "hqq": ["prepare_for_hqq_linear"],
    "integration_utils": [
        "INTEGRATION_TO_CALLBACK",
        "AzureMLCallback",
        "ClearMLCallback",
        "CodeCarbonCallback",
        "CometCallback",
        "DagsHubCallback",
        "DVCLiveCallback",
        "FlyteCallback",
        "MLflowCallback",
        "NeptuneCallback",
        "NeptuneMissingConfiguration",
        "TensorBoardCallback",
        "WandbCallback",
        "get_available_reporting_integrations",
        "get_reporting_integration_callbacks",
        "hp_params",
        "is_azureml_available",
        "is_clearml_available",
        "is_codecarbon_available",
        "is_comet_available",
        "is_dagshub_available",
        "is_dvclive_available",
        "is_flyte_deck_standard_available",
        "is_flytekit_available",
        "is_mlflow_available",
        "is_neptune_available",
        "is_optuna_available",
        "is_ray_available",
        "is_ray_tune_available",
        "is_sigopt_available",
        "is_tensorboard_available",
        "is_wandb_available",
        "rewrite_logs",
        "run_hp_search_optuna",
        "run_hp_search_ray",
        "run_hp_search_sigopt",
        "run_hp_search_wandb",
    ],
    "peft": ["PeftAdapterMixin"],
    "quanto": ["replace_with_quanto_layers"],
    "spqr": ["replace_with_spqr_linear"],
    "vptq": ["replace_with_vptq_linear"],
}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["executorch"] = [
        "TorchExportableModuleWithStaticCache",
        "convert_and_export_with_cache",
    ]

if TYPE_CHECKING:
    from .aqlm import replace_with_aqlm_linear
    from .awq import (
        fuse_awq_modules,
        post_init_awq_exllama_modules,
        post_init_awq_ipex_modules,
        replace_quantization_scales,
        replace_with_awq_linear,
    )
    from .bitnet import (
        BitLinear,
        pack_weights,
        replace_with_bitnet_linear,
        unpack_weights,
    )
    from .bitsandbytes import (
        dequantize_and_replace,
        get_keys_to_not_convert,
        replace_8bit_linear,
        replace_with_bnb_linear,
        set_module_8bit_tensor_to_device,
        set_module_quantized_tensor_to_device,
        validate_bnb_backend_availability,
    )
    from .deepspeed import (
        HfDeepSpeedConfig,
        HfTrainerDeepSpeedConfig,
        deepspeed_config,
        deepspeed_init,
        deepspeed_load_checkpoint,
        deepspeed_optim_sched,
        is_deepspeed_available,
        is_deepspeed_zero3_enabled,
        set_hf_deepspeed_config,
        unset_hf_deepspeed_config,
    )
    from .eetq import replace_with_eetq_linear
    from .fbgemm_fp8 import FbgemmFp8Linear, replace_with_fbgemm_fp8_linear
    from .fsdp import is_fsdp_managed_module
    from .ggml import (
        GGUF_CONFIG_MAPPING,
        GGUF_TOKENIZER_MAPPING,
        _gguf_parse_value,
        load_dequant_gguf_tensor,
        load_gguf,
    )
    from .higgs import HiggsLinear, dequantize_higgs, quantize_with_higgs, replace_with_higgs_linear
    from .hqq import prepare_for_hqq_linear
    from .integration_utils import (
        INTEGRATION_TO_CALLBACK,
        AzureMLCallback,
        ClearMLCallback,
        CodeCarbonCallback,
        CometCallback,
        DagsHubCallback,
        DVCLiveCallback,
        FlyteCallback,
        MLflowCallback,
        NeptuneCallback,
        NeptuneMissingConfiguration,
        TensorBoardCallback,
        WandbCallback,
        get_available_reporting_integrations,
        get_reporting_integration_callbacks,
        hp_params,
        is_azureml_available,
        is_clearml_available,
        is_codecarbon_available,
        is_comet_available,
        is_dagshub_available,
        is_dvclive_available,
        is_flyte_deck_standard_available,
        is_flytekit_available,
        is_mlflow_available,
        is_neptune_available,
        is_optuna_available,
        is_ray_available,
        is_ray_tune_available,
        is_sigopt_available,
        is_tensorboard_available,
        is_wandb_available,
        rewrite_logs,
        run_hp_search_optuna,
        run_hp_search_ray,
        run_hp_search_sigopt,
        run_hp_search_wandb,
    )
    from .peft import PeftAdapterMixin
    from .quanto import replace_with_quanto_layers
    from .spqr import replace_with_spqr_linear
    from .vptq import replace_with_vptq_linear

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .executorch import TorchExportableModuleWithStaticCache, convert_and_export_with_cache

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
