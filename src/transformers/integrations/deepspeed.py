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
Integration with Deepspeed
"""

import copy
import importlib.metadata as importlib_metadata
import importlib.util
import weakref
from functools import partialmethod

from ..dependency_versions_check import dep_version_check
from ..utils import is_accelerate_available, is_torch_available, is_torch_mlu_available, logging


if is_torch_available():
    import torch


logger = logging.get_logger(__name__)


def is_deepspeed_available():
    package_exists = importlib.util.find_spec("deepspeed") is not None

    # Check we're not importing a "deepspeed" directory somewhere but the actual library by trying to grab the version
    # AND checking it has an author field in the metadata that is HuggingFace.
    if package_exists:
        try:
            if is_torch_mlu_available():
                _ = importlib_metadata.metadata("deepspeed-mlu")
                return True
            _ = importlib_metadata.metadata("deepspeed")
            return True
        except importlib_metadata.PackageNotFoundError:
            return False


if is_accelerate_available() and is_deepspeed_available():
    from accelerate.utils.deepspeed import HfDeepSpeedConfig as DeepSpeedConfig
else:
    # Inherits from a dummy `object` if accelerate is not available, so that python succeeds to import this file.
    # Deepspeed glue code will never inherit this dummy object as it checks if accelerate is available.
    from builtins import object as DeepSpeedConfig


if is_deepspeed_available():
    from deepspeed.sequence.layer import _SeqAllToAll
    from deepspeed.utils import groups as deepspeed_groups


class HfDeepSpeedConfig(DeepSpeedConfig):
    """
    This object contains a DeepSpeed configuration dictionary and can be quickly queried for things like zero stage.

    A `weakref` of this object is stored in the module's globals to be able to access the config from areas where
    things like the Trainer object is not available (e.g. `from_pretrained` and `_get_resized_embeddings`). Therefore
    it's important that this object remains alive while the program is still running.

    [`Trainer`] uses the `HfTrainerDeepSpeedConfig` subclass instead. That subclass has logic to sync the configuration
    with values of [`TrainingArguments`] by replacing special placeholder values: `"auto"`. Without this special logic
    the DeepSpeed configuration is not modified in any way.

    Args:
        config_file_or_dict (`Union[str, Dict]`): path to DeepSpeed config file or dict.

    """

    def __init__(self, config_file_or_dict):
        # set global weakref object
        set_hf_deepspeed_config(self)
        dep_version_check("accelerate")
        dep_version_check("deepspeed")
        super().__init__(config_file_or_dict)


class HfTrainerDeepSpeedConfig(HfDeepSpeedConfig):
    """
    The `HfTrainerDeepSpeedConfig` object is meant to be created during `TrainingArguments` object creation and has the
    same lifespan as the latter.
    """

    def __init__(self, config_file_or_dict):
        super().__init__(config_file_or_dict)
        self._dtype = None
        self.mismatches = []

    def dtype(self):
        if self._dtype is None:
            raise ValueError("trainer_config_process() wasn't called yet to tell dtype")
        return self._dtype

    def is_auto(self, ds_key_long):
        val = self.get_value(ds_key_long)
        if val is None:
            return False
        else:
            return val == "auto"

    def fill_match(self, ds_key_long, hf_val, hf_key=None, must_match=True):
        """
        A utility method that massages the config file and can optionally verify that the values match.

        1. Replace "auto" values with `TrainingArguments` value.

        2. If it wasn't "auto" and `must_match` is true, then check that DS config matches Trainer
        config values and if mismatched add the entry to `self.mismatched` - will assert during
        `trainer_config_finalize` for one or more mismatches.

        """
        config, ds_key = self.find_config_node(ds_key_long)
        if config is None:
            return

        if config.get(ds_key) == "auto":
            config[ds_key] = hf_val
            return

        if not must_match:
            return

        ds_val = config.get(ds_key)
        if ds_val is not None and ds_val != hf_val:
            self.mismatches.append(f"- ds {ds_key_long}={ds_val} vs hf {hf_key}={hf_val}")

    fill_only = partialmethod(fill_match, must_match=False)

    def trainer_config_process(self, args, auto_find_batch_size=False):
        """
        Adjust the config with `TrainingArguments` values. This stage is run during `TrainingArguments` object
        creation., sequence_parallel_size, sequence_parallel_rank)
        """
        if getattr(self, "sequence_parallel_size") and self.sequence_parallel_size() > 1:
            world_size = getattr(self, "data_parallel_size", args.world_size // self.sequence_parallel_size())()
        else:
            world_size = args.world_size
        # DeepSpeed does:
        # train_batch_size = world_size * train_micro_batch_size_per_gpu * gradient_accumulation_steps
        train_batch_size = world_size * args.per_device_train_batch_size * args.gradient_accumulation_steps
        self.fill_match(
            "train_micro_batch_size_per_gpu",
            args.per_device_train_batch_size,
            "per_device_train_batch_size",
            not auto_find_batch_size,
        )
        self.fill_match(
            "gradient_accumulation_steps",
            args.gradient_accumulation_steps,
            "gradient_accumulation_steps",
        )
        self.fill_match(
            "train_batch_size",
            train_batch_size,
            "train_batch_size (calculated)",
            not auto_find_batch_size,
        )
        self.fill_match("gradient_clipping", args.max_grad_norm, "max_grad_norm")

        self.fill_match("optimizer.params.lr", args.learning_rate, "learning_rate")
        self.fill_match(
            "optimizer.params.betas",
            [args.adam_beta1, args.adam_beta2],
            "adam_beta1+adam_beta2",
        )
        self.fill_match("optimizer.params.eps", args.adam_epsilon, "adam_epsilon")
        self.fill_match("optimizer.params.weight_decay", args.weight_decay, "weight_decay")

        self.fill_only("scheduler.params.warmup_min_lr", 0)  # not a trainer arg
        self.fill_match("scheduler.params.warmup_max_lr", args.learning_rate, "learning_rate")
        # total_num_steps - will get set in trainer_config_finalize

        # fp16
        if args.fp16 or args.fp16_full_eval:
            fp16_backend = "apex" if args.fp16_backend == "apex" else "amp"
        else:
            fp16_backend = None

        if args.save_on_each_node:
            # deepspeed uses shared storage by default. Let's override this setting if save_on_each_node == True
            self.config["checkpoint"] = self.config.get("checkpoint", {})
            self.config["checkpoint"]["use_node_local_storage"] = args.save_on_each_node

        # amp: similar to the pytorch native amp - it has a bunch of optional params but we won't set
        # any here unless the user did the work
        self.fill_match(
            "fp16.enabled",
            ((args.fp16 or args.fp16_full_eval) and fp16_backend == "amp"),
            "fp16|fp16_full_eval+fp16_backend(amp)",
        )

        # apex: delegates amp work to apex (which needs to be available), but it cannot be used with any
        # ZeRO features
        self.fill_match("amp.enabled", fp16_backend == "apex", "fp16+fp16_backend(apex)")
        self.fill_match("amp.opt_level", args.fp16_opt_level, "fp16_opt_level")

        self.fill_match("bf16.enabled", (args.bf16 or args.bf16_full_eval), "bf16|bf16_full_eval")

        # deepspeed's default mode is fp16 unless there is a config that says differently
        if self.is_true("bf16.enabled"):
            self._dtype = torch.bfloat16
        elif self.is_false("fp16.enabled"):
            self._dtype = torch.float32
        else:
            self._dtype = torch.float16

    def trainer_config_finalize(self, args, model, num_training_steps):
        """
        This stage is run after we have the model and know num_training_steps.

        Now we can complete the configuration process.
        """
        # zero

        # deal with config keys that use `auto` value and rely on model's hidden_size
        hidden_size_based_keys = [
            "zero_optimization.reduce_bucket_size",
            "zero_optimization.stage3_prefetch_bucket_size",
            "zero_optimization.stage3_param_persistence_threshold",
        ]
        hidden_size_auto_keys = [x for x in hidden_size_based_keys if self.is_auto(x)]

        if len(hidden_size_auto_keys) > 0:
            if hasattr(model.config, "hidden_size"):
                hidden_size = model.config.hidden_size
            elif hasattr(model.config, "hidden_sizes"):
                # if there are many hidden sizes pick the largest one
                hidden_size = max(model.config.hidden_sizes)
            elif hasattr(model.config, "text_config") and hasattr(model.config.text_config, "hidden_size"):
                hidden_size = model.config.text_config.hidden_size
            elif hasattr(model.config, "text_config") and hasattr(model.config.text_config, "hidden_sizes"):
                # if there are many hidden sizes pick the largest one
                hidden_size = max(model.config.text_config.hidden_sizes)
            else:
                raise ValueError(
                    "The model's config file has neither `hidden_size` nor `hidden_sizes` entry, "
                    "therefore it's not possible to automatically fill out the following `auto` entries "
                    f"in the DeepSpeed config file: {hidden_size_auto_keys}. You can fix that by replacing "
                    "`auto` values for these keys with an integer value of your choice."
                )

            self.fill_only("zero_optimization.reduce_bucket_size", hidden_size * hidden_size)
            if self.is_zero3():
                # automatically assign the optimal config values based on model config
                self.fill_only(
                    "zero_optimization.stage3_prefetch_bucket_size",
                    int(0.9 * hidden_size * hidden_size),
                )
                self.fill_only(
                    "zero_optimization.stage3_param_persistence_threshold",
                    10 * hidden_size,
                )

        # scheduler
        self.fill_match(
            "scheduler.params.total_num_steps",
            num_training_steps,
            "num_training_steps (calculated)",
        )
        self.fill_match(
            "scheduler.params.warmup_num_steps",
            args.get_warmup_steps(num_training_steps),
            "warmup_steps",
        )

        if len(self.mismatches) > 0:
            mismatches = "\n".join(self.mismatches)
            raise ValueError(
                "Please correct the following DeepSpeed config values that mismatch TrainingArguments"
                f" values:\n{mismatches}\nThe easiest method is to set these DeepSpeed config values to 'auto'."
            )


# keep the config object global to be able to access it anywhere during TrainingArguments life-cycle
_hf_deepspeed_config_weak_ref = None


def set_hf_deepspeed_config(hf_deepspeed_config_obj):
    # this is a special weakref global object to allow us to get to Deepspeed config from APIs
    # that don't have an easy way to get to the Deepspeed config outside of the Trainer domain.
    global _hf_deepspeed_config_weak_ref
    # will go away automatically when HfDeepSpeedConfig is destroyed (when TrainingArguments is destroyed)
    _hf_deepspeed_config_weak_ref = weakref.ref(hf_deepspeed_config_obj)


def unset_hf_deepspeed_config():
    # useful for unit tests to ensure the global state doesn't leak - call from `tearDown` method
    global _hf_deepspeed_config_weak_ref
    _hf_deepspeed_config_weak_ref = None


def is_deepspeed_zero3_enabled():
    if _hf_deepspeed_config_weak_ref is not None and _hf_deepspeed_config_weak_ref() is not None:
        return _hf_deepspeed_config_weak_ref().is_zero3()
    else:
        return False


def is_deepspeed_sp_enabled():
    if _hf_deepspeed_config_weak_ref is not None and _hf_deepspeed_config_weak_ref() is not None:
        return _hf_deepspeed_config_weak_ref().is_sequence_parallel()
    else:
        return False


def deepspeed_config():
    if _hf_deepspeed_config_weak_ref is not None and _hf_deepspeed_config_weak_ref() is not None:
        return _hf_deepspeed_config_weak_ref().config
    else:
        return None


def deepspeed_optim_sched(trainer, hf_deepspeed_config, args, num_training_steps, model_parameters):
    """
    A convenience wrapper that deals with optimizer and lr scheduler configuration.
    """
    from accelerate.utils import DummyOptim, DummyScheduler

    config = hf_deepspeed_config.config

    # Mixing and matching DS schedulers and optimizers is supported unless Offload is enabled in which case it's:
    # 1. DS scheduler + DS optimizer: Yes
    # 2. HF scheduler + HF optimizer: Mostly*
    # 3. DS scheduler + HF optimizer: Mostly*
    # 4. HF scheduler + DS optimizer: Yes
    #
    # Mostly*: All non-native DeepSpeed optimizers that have both CPU and GPU implementation should work (except LAMB)

    optimizer = None
    if "optimizer" in config:
        if args.adafactor:
            raise ValueError(
                "--adafactor was passed, but also found `optimizer` configured in the DeepSpeed config. "
                "Only one optimizer can be configured."
            )
        optimizer = DummyOptim(params=model_parameters)
    else:
        if hf_deepspeed_config.is_offload():
            logger.info(
                "Detected ZeRO Offload and non-DeepSpeed optimizers: This combination should work as long as the"
                " custom optimizer has both CPU and GPU implementation (except LAMB)"
            )

        # ds supports Adam, OneBitAdam, and Lamb optimizers and can import other optimizers from torch.
        # But trainer uses AdamW by default.
        optimizer = trainer.create_optimizer()
        # To use other optimizers requires voiding warranty with: `zero_allow_untested_optimizer`
        config["zero_allow_untested_optimizer"] = True

    lr_scheduler = None
    if "scheduler" in config:
        lr_scheduler = DummyScheduler(optimizer)
    else:
        if isinstance(optimizer, DummyOptim):

            def _lr_scheduler_callable(optimizer):
                # create a shallow copy first, so later modifications do not affect original trainer
                trainer_copy = copy.copy(trainer)
                # at the time _lr_scheduler_callable is called, trainer.lr_scheduler has been set
                # update it to None so that we can re-create a new scheduler
                trainer_copy.lr_scheduler = None
                lr_scheduler = trainer_copy.create_scheduler(
                    num_training_steps=num_training_steps, optimizer=optimizer
                )
                return lr_scheduler

            lr_scheduler = DummyScheduler(optimizer, lr_scheduler_callable=_lr_scheduler_callable)
        else:
            lr_scheduler = trainer.create_scheduler(num_training_steps=num_training_steps, optimizer=optimizer)

    return optimizer, lr_scheduler


def deepspeed_init(trainer, num_training_steps, inference=False):
    """
    Init DeepSpeed, after updating the DeepSpeed configuration with any relevant Trainer's args.

    If `resume_from_checkpoint` was passed then an attempt to resume from a previously saved checkpoint will be made.

    Args:
        trainer: Trainer object
        num_training_steps: per single gpu
        resume_from_checkpoint: path to a checkpoint if to resume from after normal DeepSpeedEngine load
        inference: launch in inference mode (no optimizer and no lr scheduler)
        auto_find_batch_size: whether to ignore the `train_micro_batch_size_per_gpu` argument as it's being
            set automatically by the auto batch size finder

    Returns: optimizer, lr_scheduler

    We may use `deepspeed_init` more than once during the life of Trainer, when we do - it's a temp hack based on:
    https://github.com/microsoft/DeepSpeed/issues/1394#issuecomment-937405374 until Deepspeed fixes a bug where it
    can't resume from a checkpoint after it did some stepping https://github.com/microsoft/DeepSpeed/issues/1612

    """
    from deepspeed.utils import logger as ds_logger

    model = trainer.model
    args = trainer.args

    hf_deepspeed_config = trainer.accelerator.state.deepspeed_plugin.hf_ds_config

    # resume config update - some bits like `model` and `num_training_steps` only become available during train
    hf_deepspeed_config.trainer_config_finalize(args, model, num_training_steps)

    # set the Deepspeed log level consistent with the Trainer
    ds_logger.setLevel(args.get_process_log_level())

    if inference:
        # only Z3 makes sense for the inference
        if not hf_deepspeed_config.is_zero3():
            raise ValueError("ZeRO inference only makes sense with ZeRO Stage 3 - please adjust your config")

        # in case the training config is re-used for inference
        hf_deepspeed_config.del_config_sub_tree("optimizer")
        hf_deepspeed_config.del_config_sub_tree("lr_scheduler")
        optimizer, lr_scheduler = None, None
        model_parameters = None
    else:
        trainer.optimizer = None  # important for when deepspeed_init is used as re-init
        model_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        optimizer, lr_scheduler = deepspeed_optim_sched(
            trainer, hf_deepspeed_config, args, num_training_steps, model_parameters
        )

    # keep for quick debug:
    # from pprint import pprint; pprint(config)

    return optimizer, lr_scheduler


def deepspeed_load_checkpoint(deepspeed_engine, checkpoint_path, load_module_strict=True):
    # it's possible that the user is trying to resume from model_path, which doesn't necessarily
    # contain a deepspeed checkpoint. e.g. examples just check if the dir exists and assume it's
    # a resume from a checkpoint and not just a local pretrained weight. So we check here if the
    # path contains what looks like a deepspeed checkpoint
    import glob

    deepspeed_checkpoint_dirs = sorted(glob.glob(f"{checkpoint_path}/global_step*"))

    if len(deepspeed_checkpoint_dirs) > 0:
        logger.info(f"Attempting to resume from {checkpoint_path}")
        # this magically updates self.optimizer and self.lr_scheduler
        load_path, _ = deepspeed_engine.load_checkpoint(
            checkpoint_path,
            load_module_strict=load_module_strict,
            load_optimizer_states=True,
            load_lr_scheduler_states=True,
        )
        if load_path is None:
            raise ValueError(f"[deepspeed] failed to resume from checkpoint {checkpoint_path}")
    else:
        raise ValueError(f"Can't find a valid checkpoint at {checkpoint_path}")


def deepspeed_ulysses_attention(attn_func, seq_dim=1, head_dim=2):
    def wrapped(*args, **kwargs):
        if is_deepspeed_sp_enabled():
            spg = deepspeed_groups._get_sequence_parallel_group()
            scatter_idx = head_dim  # Scatter on num_heads dimension
            gather_idx = seq_dim  # Gather on seq_len dimension
            batch_dim_idx = 0  # Synonymous with the batch_first==true
            args = list(args)
            args[0] = _SeqAllToAll.apply(spg, args[0], scatter_idx, gather_idx, batch_dim_idx)
            args[1] = _SeqAllToAll.apply(spg, args[1], scatter_idx, gather_idx, batch_dim_idx)
            args[2] = _SeqAllToAll.apply(spg, args[2], scatter_idx, gather_idx, batch_dim_idx)
            args = tuple(args)

        attn_output = attn_func(*args, **kwargs)

        if is_deepspeed_sp_enabled():
            scatter_idx = seq_dim  # Scatter back on seq_len dimension
            gather_idx = head_dim  # Gather on num_heads dimension
            batch_dim_idx = 0
            attn_output = _SeqAllToAll.apply(spg, attn_output, scatter_idx, gather_idx, batch_dim_idx)

        return attn_output

    return wrapped


def support_deepspeed_ulysses(module):
    module._supports_sequence_parallel = True

    original_forward = module.forward

    def wrapped_forward(
        query_states: torch.Tensor,
        *args,
        **kwargs,
    ):
        # lazily set if sequence parallelism is enabled to ensure deepspeed is initialized first
        if is_deepspeed_sp_enabled():
            module.sp_group_size = deepspeed_groups._get_sequence_parallel_world_size()

        return original_forward(
            query_states,
            *args,
            **kwargs,
        )

    module.forward = wrapped_forward

    return module
