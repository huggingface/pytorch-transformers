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

import dataclasses
import io
import itertools
import json
import os
import unittest
from copy import deepcopy
from functools import partial

import datasets
from parameterized import parameterized

import tests.trainer.test_trainer
import transformers
from tests.trainer.test_trainer import TrainerIntegrationCommon  # noqa
from transformers import AutoModel, TrainingArguments, is_torch_available, logging
from transformers.integrations.deepspeed import (
    HfDeepSpeedConfig,
    is_deepspeed_available,
    unset_hf_deepspeed_config,
)
from transformers.testing_utils import (
    CaptureLogger,
    CaptureStd,
    CaptureStderr,
    LoggingLevel,
    TestCasePlus,
    backend_device_count,
    execute_subprocess_async,
    mockenv_context,
    require_deepspeed,
    require_optuna,
    require_torch_accelerator,
    require_torch_multi_accelerator,
    slow,
    torch_device,
)
from transformers.trainer_utils import get_last_checkpoint, set_seed
from transformers.utils import SAFE_WEIGHTS_NAME, is_torch_bf16_available_on_device


if is_torch_available():
    import torch

    from tests.trainer.test_trainer import (  # noqa
        RegressionModelConfig,
        RegressionPreTrainedModel,
    )

    # hack to restore original logging level pre #21700
    get_regression_trainer = partial(tests.trainer.test_trainer.get_regression_trainer, log_level="info")


set_seed(42)

# default torch.distributed port
DEFAULT_MASTER_PORT = "10999"

T5_SMALL = "google-t5/t5-small"
T5_TINY = "patrickvonplaten/t5-tiny-random"
GPT2_TINY = "sshleifer/tiny-gpt2"
GPTJ_TINY = "hf-internal-testing/tiny-random-gptj"


def load_json(path):
    with open(path) as f:
        return json.load(f)


def get_master_port(real_launcher=False):
    """
    When using a single gpu launcher emulation (i.e. not deepspeed or python -m torch.distributed)
    the issue is that once the port is tied it can't be used anywhere else outside of this process,
    since torch.dist doesn't free the port until the process exits. Therefore for the sake of being
    able to run both emulated launcher and normal launcher tests we need 2 distinct ports.

    This function will give the right port in the right context. For real launcher it'll give the
    base port, for emulated launcher it'll give the base port + 1. In both cases a string is
    returned.

    Args:
        `real_launcher`: whether a real launcher is going to be used, or the emulated one

    """

    master_port_base = os.environ.get("DS_TEST_PORT", DEFAULT_MASTER_PORT)
    if not real_launcher:
        master_port_base = str(int(master_port_base) + 1)
    return master_port_base


def require_deepspeed_aio(test_case):
    """
    Decorator marking a test that requires deepspeed aio (nvme)
    """
    if not is_deepspeed_available():
        return unittest.skip(reason="test requires deepspeed")(test_case)

    import deepspeed
    from deepspeed.ops.aio import AsyncIOBuilder

    if not deepspeed.ops.__compatible_ops__[AsyncIOBuilder.NAME]:
        return unittest.skip(reason="test requires deepspeed async-io")(test_case)
    else:
        return test_case


if is_deepspeed_available():
    from deepspeed.utils import logger as deepspeed_logger  # noqa
    from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
    from transformers.integrations.deepspeed import deepspeed_config, is_deepspeed_zero3_enabled  # noqa


def get_launcher(distributed=False):
    # 1. explicitly set --num_nodes=1 just in case these tests end up run on a multi-node setup
    # - it won't be able to handle that
    # 2. for now testing with just 2 gpus max (since some quality tests may give different
    # results with mode gpus because we use very little data)
    num_gpus = min(2, backend_device_count(torch_device)) if distributed else 1
    master_port = get_master_port(real_launcher=True)
    return f"deepspeed --num_nodes 1 --num_gpus {num_gpus} --master_port {master_port}".split()


ZERO2 = "zero2"
ZERO3 = "zero3"

FP16 = "fp16"
BF16 = "bf16"

HF_OPTIM = "hf_optim"
HF_SCHEDULER = "hf_scheduler"
DS_OPTIM = "ds_optim"
DS_SCHEDULER = "ds_scheduler"

optims = [HF_OPTIM, DS_OPTIM]
schedulers = [HF_SCHEDULER, DS_SCHEDULER]

stages = [ZERO2, ZERO3]
if is_torch_bf16_available_on_device(torch_device):
    dtypes = [FP16, BF16]
else:
    dtypes = [FP16]


def parameterized_custom_name_func(func, param_num, param):
    # customize the test name generator function as we want both params to appear in the sub-test
    # name, as by default it shows only the first param
    param_based_name = parameterized.to_safe_name("_".join(str(x) for x in param.args))
    return f"{func.__name__}_{param_based_name}"


# Cartesian-product of zero stages with models to test
params = list(itertools.product(stages, dtypes))

params_with_optims_and_schedulers = list(itertools.product(stages, dtypes, optims, schedulers))


@require_deepspeed
@require_torch_accelerator
class CoreIntegrationDeepSpeed(TestCasePlus, TrainerIntegrationCommon):
    """
    Testing non-Trainer DeepSpeed integration
    """

    def setUp(self):
        super().setUp()

        master_port = get_master_port(real_launcher=False)
        self.dist_env_1_gpu = {
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": master_port,
            "RANK": "0",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": "1",
        }

    def tearDown(self):
        super().tearDown()

        # reset the ds config global so that tests state doesn't leak
        unset_hf_deepspeed_config()

    def test_init_zero3_fp16(self):
        # test that zero.Init() works correctly under zero3/fp16
        ds_config = {
            "train_batch_size": 1,
            "zero_optimization": {
                "stage": 3,
            },
        }

        dschf = HfDeepSpeedConfig(ds_config)

        self.assertTrue(dschf.is_zero3())
        self.assertTrue(is_deepspeed_zero3_enabled())

        with LoggingLevel(logging.INFO):
            with mockenv_context(**self.dist_env_1_gpu):
                logger = logging.get_logger("transformers.modeling_utils")
                with CaptureLogger(logger) as cl:
                    AutoModel.from_pretrained(T5_TINY)
        self.assertIn("Detected DeepSpeed ZeRO-3", cl.out)

        # now remove zero optimization
        del ds_config["zero_optimization"]
        dschf = HfDeepSpeedConfig(ds_config)

        self.assertFalse(dschf.is_zero3())
        self.assertFalse(is_deepspeed_zero3_enabled())

        with LoggingLevel(logging.INFO):
            with mockenv_context(**self.dist_env_1_gpu):
                logger = logging.get_logger("transformers.modeling_utils")
                with CaptureLogger(logger) as cl:
                    AutoModel.from_pretrained(T5_TINY)
        self.assertNotIn("Detected DeepSpeed ZeRO-3", cl.out)

    def test_init_zero3_missing_params(self):
        # test that zero.Init() for missing parameters works correctly under zero3
        import deepspeed
        import torch

        from transformers.models.gpt2.modeling_gpt2 import GPT2PreTrainedModel

        class TinyGPT2WithUninitializedWeights(GPT2PreTrainedModel):
            def __init__(self, config):
                super().__init__(config)
                self.transformer = AutoModel.from_pretrained(GPT2_TINY, config=config)
                self.new_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=True)

            def forward(self, *args, **kwargs):
                transformer_outputs = self.transformer(*args, **kwargs)
                hidden_states = transformer_outputs[0]
                return self.new_head(hidden_states).float()

            def _init_weights(self, module):
                super()._init_weights(module)
                if module is self.new_head:
                    self.new_head.weight.data.fill_(-100.0)
                    self.new_head.bias.data.fill_(+100.0)

        ds_config = {
            "train_batch_size": 1,
            "zero_optimization": {
                "stage": 3,
            },
        }

        dschf = HfDeepSpeedConfig(ds_config)

        self.assertTrue(dschf.is_zero3())
        self.assertTrue(is_deepspeed_zero3_enabled())

        with LoggingLevel(logging.INFO):
            with mockenv_context(**self.dist_env_1_gpu):
                logger = logging.get_logger("transformers.modeling_utils")
                with CaptureLogger(logger) as cl:
                    model = TinyGPT2WithUninitializedWeights.from_pretrained(GPT2_TINY)
        self.assertIn("Detected DeepSpeed ZeRO-3", cl.out)
        self.assertRegex(cl.out, r"newly initialized.*new_head\.bias.*new_head\.weight")
        with deepspeed.zero.GatheredParameters([model.new_head.weight, model.new_head.bias]):
            self.assertTrue(
                torch.allclose(model.new_head.weight, torch.tensor(-100.0, device=model.new_head.weight.device)),
            )
            self.assertTrue(
                torch.allclose(model.new_head.bias, torch.tensor(+100.0, device=model.new_head.bias.device)),
            )

        # now remove zero optimization
        del ds_config["zero_optimization"]
        dschf = HfDeepSpeedConfig(ds_config)

        self.assertFalse(dschf.is_zero3())
        self.assertFalse(is_deepspeed_zero3_enabled())

        with LoggingLevel(logging.INFO):
            with mockenv_context(**self.dist_env_1_gpu):
                logger = logging.get_logger("transformers.modeling_utils")
                with CaptureLogger(logger) as cl:
                    model = TinyGPT2WithUninitializedWeights.from_pretrained(GPT2_TINY)
        self.assertNotIn("Detected DeepSpeed ZeRO-3", cl.out)
        self.assertRegex(cl.out, r"newly initialized.*new_head\.bias.*new_head\.weight")
        self.assertTrue(
            torch.allclose(model.new_head.weight, torch.tensor(-100.0, device=model.new_head.weight.device)),
        )
        self.assertTrue(
            torch.allclose(model.new_head.bias, torch.tensor(+100.0, device=model.new_head.bias.device)),
        )

    def test_arange_bf16(self):
        # Tests that configuring DeepSpeed with 16 bits does not cause float `torch.arange()` tensors to be cast down.
        # NOTE -- this assumes that the function calls have the following downcast-preventing pattern, i.e.
        # `torch.arange(...,dtype=torch.int64)` followed by a cast like `.to(torch.float32)`. 🚨 If this pattern is
        # NOT applied (e.g. `torch.arange(...,dtype=torch.float32)` is used), DeepSpeed can automatically cast it down
        # at init time. See https://github.com/huggingface/transformers/issues/28685 for more info.

        ds_config = {
            "train_batch_size": 1,
            "zero_optimization": {
                "stage": 3,
            },
            "bf16": {"enabled": True},
        }

        dschf = HfDeepSpeedConfig(ds_config)

        self.assertTrue(dschf.is_zero3())
        self.assertTrue(is_deepspeed_zero3_enabled())

        with LoggingLevel(logging.INFO):
            with mockenv_context(**self.dist_env_1_gpu):
                logger = logging.get_logger("transformers.modeling_utils")
                with CaptureLogger(logger) as cl:
                    model = AutoModel.from_pretrained(GPTJ_TINY)
        self.assertIn("Detected DeepSpeed ZeRO-3", cl.out)

        # The model weights are in BF16 as per deepspeed config
        self.assertTrue(str(model.h[0].attn.q_proj.weight.dtype) == "torch.bfloat16")
        good_deepspeed_sin_cos = model.h[0].attn.embed_positions

        # Monkeypatches the function that creates RoPE embeddings using the INCORRECT torch.arange() pattern, and
        # then recreates the model
        def bad_deepspeed_create_sinusoidal_positions(num_pos: int, dim: int) -> torch.Tensor:
            inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.int64) / dim))
            # Incorrect pattern here: torch.arange has dtype=torch.float32 as its argument, and it will automatically
            # converted to BF16 by DeepSpeed
            sinusoid_inp = torch.einsum("i , j -> i j", torch.arange(num_pos, dtype=inv_freq.dtype), inv_freq)
            return torch.cat((torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)), dim=1)

        good_deepspeed_create_sinusoidal_positions = transformers.models.gptj.modeling_gptj.create_sinusoidal_positions
        transformers.models.gptj.modeling_gptj.create_sinusoidal_positions = bad_deepspeed_create_sinusoidal_positions

        with LoggingLevel(logging.INFO):
            with mockenv_context(**self.dist_env_1_gpu):
                logger = logging.get_logger("transformers.modeling_utils")
                with CaptureLogger(logger) as cl:
                    model = AutoModel.from_pretrained(GPTJ_TINY)
        self.assertIn("Detected DeepSpeed ZeRO-3", cl.out)

        self.assertTrue(str(model.h[0].attn.q_proj.weight.dtype) == "torch.bfloat16")
        bad_deepspeed_sin_cos = model.h[0].attn.embed_positions

        # Compares the two values: the two sets of values are different, and the correct one matches the torch
        # (i.e. outside DeepSpeed) version.
        good_torch_sin_cos = good_deepspeed_create_sinusoidal_positions(
            model.config.max_position_embeddings, model.config.rotary_dim
        )
        self.assertFalse(torch.allclose(good_deepspeed_sin_cos, bad_deepspeed_sin_cos))
        self.assertTrue(torch.allclose(good_torch_sin_cos, good_deepspeed_sin_cos.cpu()))

        # Finally, we can see that the incorrect pattern is okay on vanilla torch, demostrating that this issue is
        # exclusive to DeepSpeed
        bad_torch_sin_cos = bad_deepspeed_create_sinusoidal_positions(
            model.config.max_position_embeddings, model.config.rotary_dim
        )
        self.assertTrue(torch.allclose(bad_torch_sin_cos, good_torch_sin_cos))


class TrainerIntegrationDeepSpeedWithCustomConfig(TestCasePlus):
    def setUp(self):
        super().setUp()

        args = TrainingArguments(".")
        self.n_epochs = args.num_train_epochs
        self.batch_size = args.train_batch_size

        master_port = get_master_port(real_launcher=False)
        self.dist_env_1_gpu = {
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": master_port,
            "RANK": "0",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": "1",
        }

        self.ds_config_file = {
            "zero2": f"{self.test_file_dir_str}/ds_config_zero2.json",
            "zero3": f"{self.test_file_dir_str}/ds_config_zero3.json",
        }

        # use self.get_config_dict(stage) to use these to ensure the original is not modified
        with io.open(self.ds_config_file[ZERO2], "r", encoding="utf-8") as f:
            config_zero2 = json.load(f)
        with io.open(self.ds_config_file[ZERO3], "r", encoding="utf-8") as f:
            config_zero3 = json.load(f)
            # The following setting slows things down, so don't enable it by default unless needed by a test.
            # It's in the file as a demo for users since we want everything to work out of the box even if slower.
            config_zero3["zero_optimization"]["stage3_gather_16bit_weights_on_model_save"] = False

        self.ds_config_dict = {
            "zero2": config_zero2,
            "zero3": config_zero3,
        }

    def tearDown(self):
        super().tearDown()

        # reset the ds config global so that tests state doesn't leak
        unset_hf_deepspeed_config()

    def get_config_dict(self, stage):
        # As some tests modify the dict, always make a copy
        return deepcopy(self.ds_config_dict[stage])


@require_deepspeed
@require_torch_accelerator
class TrainerIntegrationDeepSpeed(TrainerIntegrationDeepSpeedWithCustomConfig, TrainerIntegrationCommon):
    """

    This class is for testing directly via get_regression_trainer

    It mixes in `TrainerIntegrationCommon` which already has a lot of helper validation methods
    which we can re-use here.

    Important: this class' setup can only work with a single gpu because it runs within the current
    pytest worker. For multi-gpu tests use TestDeepSpeedWithLauncher.

    Note: if any of the tests of this class get run there will be at least one gpu occupied by them
    until this pytest worker exits. This is because the gpu memory allocated by the cuda-kernels
    won't be released until this pytest worker exits.

    This may appear as some run-away tests if you watch `nvidia-smi` while other tests that fork new
    processes are run. So there will be one or two "stale" processes reported in `nvidia-smi`. This
    is not a bug.
    """

    # --- These tests are enough to run on one of zero stages --- #

    def test_hf_ds_config_mismatch(self):
        ds_config = self.get_config_dict(ZERO2)

        # Purposefully configure these values to mismatch TrainingArguments values.
        # This currently doesn't cover all keys (but it could)
        per_device_train_batch_size = 2
        ds_config["train_micro_batch_size_per_gpu"] = per_device_train_batch_size + 2

        ds_config["train_batch_size"] = 1000

        gradient_accumulation_steps = 2
        ds_config["gradient_accumulation_steps"] = gradient_accumulation_steps + 2

        max_grad_norm = 1.0
        ds_config["gradient_clipping"] = max_grad_norm + 0.1

        adam_beta1, adam_beta2 = 0.9, 0.99
        ds_config["optimizer"]["params"]["betas"] = [adam_beta1 - 0.1, adam_beta2 - 0.1]

        fp16 = True
        ds_config["fp16"]["enabled"] = not fp16

        keys = [
            "per_device_train_batch_size",
            "train_batch_size",
            "gradient_accumulation_steps",
            "max_grad_norm",
            "betas",
            "fp16",
        ]

        with mockenv_context(**self.dist_env_1_gpu):
            trainer = get_regression_trainer(
                local_rank=0,
                fp16=fp16,
                deepspeed=ds_config,
                per_device_train_batch_size=per_device_train_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                max_grad_norm=max_grad_norm,
                adam_beta1=adam_beta1,
                adam_beta2=adam_beta2,
                output_dir=self.get_auto_remove_tmp_dir(),
            )
            with self.assertRaises(Exception) as context:
                trainer.train()

        for key in keys:
            self.assertTrue(
                key in str(context.exception),
                f"{key} is not in the exception message:\n{context.exception}",
            )

    # Test various combos
    # 1. DS scheduler + DS optimizer: this is already tested by most other tests
    # 2. HF scheduler + HF optimizer:
    # 3. DS scheduler + HF optimizer:
    # 4. HF scheduler + DS optimizer:

    def test_hf_scheduler_hf_optimizer(self):
        a = 0
        with mockenv_context(**self.dist_env_1_gpu):
            ds_config_zero2_dict = self.get_config_dict(ZERO2)
            del ds_config_zero2_dict["optimizer"]  # force default HF Trainer optimizer
            del ds_config_zero2_dict["scheduler"]  # force default HF Trainer scheduler
            ds_config_zero2_dict["zero_optimization"]["offload_optimizer"]["device"] = "none"
            ds_config_zero2_dict["fp16"]["initial_scale_power"] = 1  # force optimizer on the first step
            trainer = get_regression_trainer(
                a=a, local_rank=0, fp16=True, deepspeed=ds_config_zero2_dict, output_dir=self.get_auto_remove_tmp_dir()
            )
            trainer.train()
        new_a = trainer.model.a.item()
        self.assertNotEqual(new_a, a)

    def test_ds_scheduler_hf_optimizer(self):
        a = 0
        with mockenv_context(**self.dist_env_1_gpu):
            ds_config_zero2_dict = self.get_config_dict(ZERO2)
            del ds_config_zero2_dict["optimizer"]  # force default HF Trainer optimizer
            ds_config_zero2_dict["zero_optimization"]["offload_optimizer"]["device"] = "none"
            ds_config_zero2_dict["fp16"]["initial_scale_power"] = 1  # force optimizer on the first step
            trainer = get_regression_trainer(
                a=a, local_rank=0, fp16=True, deepspeed=ds_config_zero2_dict, output_dir=self.get_auto_remove_tmp_dir()
            )
            trainer.train()
        new_a = trainer.model.a.item()
        self.assertNotEqual(new_a, a)

    def test_hf_scheduler_ds_optimizer(self):
        a = 0
        with mockenv_context(**self.dist_env_1_gpu):
            ds_config_zero2_dict = self.get_config_dict(ZERO2)
            del ds_config_zero2_dict["scheduler"]  # force default HF Trainer scheduler
            ds_config_zero2_dict["zero_optimization"]["offload_optimizer"]["device"] = "none"
            ds_config_zero2_dict["fp16"]["initial_scale_power"] = 1  # force optimizer on the first step
            trainer = get_regression_trainer(
                a=a, local_rank=0, fp16=True, deepspeed=ds_config_zero2_dict, output_dir=self.get_auto_remove_tmp_dir()
            )
            trainer.train()
        new_a = trainer.model.a.item()
        self.assertNotEqual(new_a, a)

    @require_deepspeed_aio
    def test_stage3_nvme_offload(self):
        with mockenv_context(**self.dist_env_1_gpu):
            # this actually doesn't have to be on NVMe, any storage will do since this test only
            # runs a simple check that we can use some directory as if it were NVMe
            nvme_path = self.get_auto_remove_tmp_dir()
            nvme_config = {"device": "nvme", "nvme_path": nvme_path}
            ds_config_zero3_dict = self.get_config_dict(ZERO3)
            ds_config_zero3_dict["zero_optimization"]["offload_optimizer"] = nvme_config
            ds_config_zero3_dict["zero_optimization"]["offload_param"] = nvme_config
            ds_config_zero3_dict["zero_optimization"]["stage3_gather_16bit_weights_on_model_save"] = True
            trainer = get_regression_trainer(
                local_rank=0, fp16=True, deepspeed=ds_config_zero3_dict, output_dir=self.get_auto_remove_tmp_dir()
            )
            with CaptureLogger(deepspeed_logger) as cl:
                trainer.train()
            self.assertIn("DeepSpeed info", cl.out, "expected DeepSpeed logger output but got none")

    @require_optuna
    def test_hyperparameter_search(self):
        with mockenv_context(**self.dist_env_1_gpu):
            ds_config_zero3_dict = self.get_config_dict(ZERO3)

            # hyperparameter_search requires model_init() to recreate the model for each trial
            def model_init():
                config = RegressionModelConfig(a=0, b=0, double_output=False)
                model = RegressionPreTrainedModel(config)
                return model

            trainer = get_regression_trainer(
                local_rank=0,
                fp16=True,
                model_init=model_init,
                deepspeed=ds_config_zero3_dict,
                output_dir=self.get_auto_remove_tmp_dir(),
            )

            n_trials = 3
            with CaptureLogger(deepspeed_logger) as cl:
                with CaptureStd() as cs:
                    trainer.hyperparameter_search(direction="maximize", n_trials=n_trials)
            self.assertIn("DeepSpeed info", cl.out, "expected DeepSpeed logger output but got none")
            self.assertIn(f"Trial {n_trials-1} finished with value", cs.err, "expected hyperparameter_search output")
            self.assertIn("Best is trial", cs.err, "expected hyperparameter_search output")

    # --- These tests need to run on both zero stages --- #

    @parameterized.expand(params, name_func=parameterized_custom_name_func)
    def test_hf_optimizer_with_offload(self, stage, dtype):
        # non-DS optimizers can be used with ZERO-offload (as long as they have both CPU and GPU implementation (except LAMB))
        ds_config_dict = self.get_config_dict(stage)
        del ds_config_dict["optimizer"]  # force default HF Trainer optimizer
        # force cpu offload
        ds_config_dict["zero_optimization"]["offload_optimizer"]["device"] = "cpu"
        ds_config_dict["zero_force_ds_cpu_optimizer"] = False  # offload is not efficient w/o CPUAdam
        with mockenv_context(**self.dist_env_1_gpu):
            kwargs = {"local_rank": 0, "deepspeed": ds_config_dict, "output_dir": self.get_auto_remove_tmp_dir()}
            kwargs[dtype] = True
            trainer = get_regression_trainer(**kwargs)
            with CaptureLogger(deepspeed_logger) as cl:
                trainer.train()
            self.assertIn("DeepSpeed info", cl.out, "expected DeepSpeed logger output but got none")

    @parameterized.expand(params, name_func=parameterized_custom_name_func)
    def test_fake_notebook_no_launcher(self, stage, dtype):
        # this setup emulates a notebook where a launcher needs to be emulated by hand

        # note that unittest resets sys.stdout each test, so `CaptureStd` will work here to capture
        # DeepSpeed log if this test happens to run first in this pytest worker. But it will fail if
        # it's run not as a first test as `sys.stdout` will no longer be the same. So we either have
        # to reset `deepspeed_logger.handlers[0].setStream(sys.stdout)` or directly capture from the deepspeed_logger.
        with mockenv_context(**self.dist_env_1_gpu):
            kwargs = {
                "local_rank": 0,
                "deepspeed": self.get_config_dict(stage),
                "output_dir": self.get_auto_remove_tmp_dir(),
            }
            kwargs[dtype] = True
            trainer = get_regression_trainer(**kwargs)

            with CaptureLogger(deepspeed_logger) as cl:
                trainer.train()
            self.assertIn("DeepSpeed info", cl.out, "expected DeepSpeed logger output but got none")

    @parameterized.expand(params, name_func=parameterized_custom_name_func)
    def test_early_get_last_lr(self, stage, dtype):
        # with deepspeed's fp16 and dynamic loss scale enabled the optimizer/scheduler steps may
        # not run for the first few dozen steps while loss scale is too large, and thus during
        # that time `get_last_lr` will fail if called during that warm up stage,
        #
        # setting `logging_steps=1` forces an early `trainer._maybe_log_save_evaluate()` which calls
        # `self.lr_scheduler.get_last_lr()` and originally it'd fail on the very first step.
        with mockenv_context(**self.dist_env_1_gpu):
            a = b = 0.0
            kwargs = {
                "a": a,
                "b": b,
                "local_rank": 0,
                "train_len": 8,
                "deepspeed": self.get_config_dict(stage),
                "per_device_train_batch_size": 8,
                "logging_steps": 1,
                "output_dir": self.get_auto_remove_tmp_dir(),
            }
            kwargs[dtype] = True
            trainer = get_regression_trainer(**kwargs)

            trainer.train()
            post_train_a = trainer.model.a.item()

            # XXX: for some reason the following check fails with zero3/fp16 and any/bf16 - not a
            # broken but a different qualitative outcome - as if optimizer did run
            # oddly getting 1.0 for both a and b from 0.0 - there is a bug somewhere
            # print(trainer.model.a.item())
            # print(trainer.model.b.item())
            # need to investigate at some point
            if (stage == ZERO3 and dtype == FP16) or (dtype == BF16):
                self.skipTest(reason="When using zero3/fp16 or any/bf16 the optimizer seems run oddly")

            # it's enough that train didn't fail for this test, but we must check that
            # optimizer/scheduler didn't run (since if it did this test isn't testing the right thing)
            self.assertEqual(post_train_a, a)

    @parameterized.expand(params, name_func=parameterized_custom_name_func)
    def test_gradient_accumulation(self, stage, dtype):
        # this test measures that we get identical weights and similar loss with:
        # 1. per_device_train_batch_size=8, gradient_accumulation_steps=1
        # 2. per_device_train_batch_size=4, gradient_accumulation_steps=2
        # since the 2nd should produce the effective batch of 1st, with the same results
        #
        # I can get an identical loss for a small train_len=32, plus the power of the initial
        # dynamic loss scale value set to:
        #   "fp16.initial_scale_power": 1
        # plus having the same WarmupLR's warmup_min_lr == warmup_max_lr in the config file
        # but for some reason going to train_len=64 the weights, weights start to mismatch with this setup.
        # the culprit seems to be `initial_scale_power` - putting it back to its default 32 keeps the weights identical

        train_len = 64
        a = b = 0.0

        kwargs = {
            "a": a,
            "b": b,
            "local_rank": 0,
            "train_len": train_len,
            "deepspeed": self.get_config_dict(stage),
            "output_dir": self.get_auto_remove_tmp_dir(),
        }
        kwargs[dtype] = True

        with mockenv_context(**self.dist_env_1_gpu):
            no_grad_accum_trainer = get_regression_trainer(
                **kwargs,
                per_device_train_batch_size=16,
                gradient_accumulation_steps=1,
            )
            no_grad_accum_result = no_grad_accum_trainer.train()
            no_grad_accum_loss = no_grad_accum_result.training_loss
            no_grad_accum_a = no_grad_accum_trainer.model.a.item()
            no_grad_accum_b = no_grad_accum_trainer.model.b.item()
            # make sure the optimizer kicked in - if it hasn't changed from the original value of a then make train_len bigger
            self.assertNotEqual(no_grad_accum_a, a)

        with mockenv_context(**self.dist_env_1_gpu):
            yes_grad_accum_trainer = get_regression_trainer(
                **kwargs,
                per_device_train_batch_size=4,
                gradient_accumulation_steps=4,
            )
            yes_grad_accum_result = yes_grad_accum_trainer.train()
            yes_grad_accum_loss = yes_grad_accum_result.training_loss
            yes_grad_accum_a = yes_grad_accum_trainer.model.a.item()
            yes_grad_accum_b = yes_grad_accum_trainer.model.b.item()
            self.assertNotEqual(yes_grad_accum_a, a)

        # training with half the batch size but accumulation steps as 2 should give the same
        # weights, but sometimes get a slight difference still of 1e-6
        self.assertAlmostEqual(no_grad_accum_a, yes_grad_accum_a, places=5)
        self.assertAlmostEqual(no_grad_accum_b, yes_grad_accum_b, places=5)

        # Relative difference. See the note above how to get identical loss on a small bs
        self.assertTrue((no_grad_accum_loss - yes_grad_accum_loss) / (no_grad_accum_loss + 1e-15) <= 1e-3)

    # NOTE: Currently a disabled test. In the future we should re-enable it.
    # Issue resolves around Zero-3 w/ DPO/TRL + DeepSpeed
    # As well as Zero-3 inference
    # Related PR: https://github.com/huggingface/transformers/pull/32299
    # def test_missed_zero3_init(self):
    #     from transformers import Trainer  # noqa

    #     with mockenv_context(**self.dist_env_1_gpu):
    #         model = AutoModel.from_pretrained(T5_TINY)
    #         training_args = TrainingArguments(
    #             output_dir="./test_missed_zero3_init",
    #             deepspeed=self.get_config_dict(ZERO3),
    #         )
    #         with self.assertRaises(
    #             ValueError, msg="Model was not initialized with `Zero-3` despite being configured."
    #         ):
    #             _ = Trainer(
    #                 model=model,
    #                 args=training_args,
    #             )
    #         # Now do it properly, triggered from our `TrainingArguments` earlier
    #         model = AutoModel.from_pretrained(T5_TINY)
    #         trainer = Trainer(
    #             model=model,
    #             args=training_args,
    #         )
    #         assert trainer.is_deepspeed_enabled
    #         assert model._transformers_zero3_init_used

    def check_saved_checkpoints_deepspeed(self, output_dir, freq, total, stage, dtype):
        # adapted from TrainerIntegrationCommon.check_saved_checkpoints
        file_list = [SAFE_WEIGHTS_NAME, "training_args.bin", "trainer_state.json", "config.json"]

        if stage == ZERO2:
            ds_file_list = ["mp_rank_00_model_states.pt"]
        elif stage == ZERO3:
            ds_file_list = ["zero_pp_rank_0_mp_rank_00_model_states.pt"]
        else:
            raise ValueError(f"unknown stage {stage}")

        if dtype == "bf16":
            ds_file_list.append("bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt")

        for step in range(freq, total, freq):
            checkpoint = os.path.join(output_dir, f"checkpoint-{step}")
            self.assertTrue(os.path.isdir(checkpoint), f"[{stage}] {checkpoint} dir is not found")
            # common files
            for filename in file_list:
                path = os.path.join(checkpoint, filename)
                self.assertTrue(os.path.isfile(path), f"[{stage}] {path} is not found")

            # ds files
            ds_path = os.path.join(checkpoint, f"global_step{step}")
            for filename in ds_file_list:
                # filename = os.path.join(path, filename)
                # print(filename)
                path = os.path.join(ds_path, filename)
                self.assertTrue(os.path.isfile(path), f"[{stage}] {path} is not found")

    @parameterized.expand(params, name_func=parameterized_custom_name_func)
    def test_save_checkpoints(self, stage, dtype):
        # adapted from  TrainerIntegrationTest.test_save_checkpoints

        freq = 5
        output_dir = self.get_auto_remove_tmp_dir()
        ds_config_dict = self.get_config_dict(stage)
        if dtype == FP16:
            ds_config_dict["fp16"]["initial_scale_power"] = 1  # force optimizer on the first step
        # XXX:
        if stage == ZERO3:
            ds_config_dict["zero_optimization"]["stage3_gather_16bit_weights_on_model_save"] = True

        # save checkpoints
        with mockenv_context(**self.dist_env_1_gpu):
            kwargs = {
                "output_dir": output_dir,
                "save_steps": freq,
                "deepspeed": ds_config_dict,
            }
            kwargs[dtype] = True
            trainer = get_regression_trainer(**kwargs)
            trainer.train()

        total = int(self.n_epochs * 64 / self.batch_size)
        self.check_saved_checkpoints_deepspeed(output_dir, freq, total, stage, dtype)

    @parameterized.expand(params, name_func=parameterized_custom_name_func)
    def test_can_resume_training_errors(self, stage, dtype):
        with mockenv_context(**self.dist_env_1_gpu):
            ds_config_dict = self.get_config_dict(stage)
            output_dir = self.get_auto_remove_tmp_dir()
            kwargs = {"output_dir": output_dir, "deepspeed": ds_config_dict}
            kwargs[dtype] = True
            trainer = get_regression_trainer(**kwargs)

            # 1. fail to find any checkpoint - due a fresh output_dir
            with self.assertRaises(Exception) as context:
                trainer.train(resume_from_checkpoint=True)
            self.assertTrue(
                "No valid checkpoint found in output directory" in str(context.exception),
                f"got exception: {context.exception}",
            )

            # 2. fail to find a bogus checkpoint
            with self.assertRaises(Exception) as context:
                checkpoint = os.path.join(output_dir, "checkpoint-5")
                trainer.train(resume_from_checkpoint=f"{checkpoint}-bogus")

    @parameterized.expand(params_with_optims_and_schedulers, name_func=parameterized_custom_name_func)
    def test_can_resume_training_normal(self, stage, dtype, optim, scheduler):
        # adapted from TrainerIntegrationTest.test_can_resume_training
        # test normal resume for each stage separately, error-handling is tested in a different test

        # ToDo: Currently, hf_optim + hf_scheduler resumes with the correct states and
        # also has same losses for few steps but then slowly diverges. Need to figure it out.
        if optim == HF_OPTIM and scheduler == HF_SCHEDULER:
            self.skipTest(reason="hf_optim + hf_scheduler resumes with the correct states but slowly diverges")

        output_dir = self.get_auto_remove_tmp_dir("./xxx", after=False)
        ds_config_dict = self.get_config_dict(stage)
        if dtype == FP16:
            ds_config_dict["fp16"]["initial_scale_power"] = 1  # force optimizer on the first step
        # XXX:
        if stage == ZERO3:
            ds_config_dict["zero_optimization"]["stage3_gather_16bit_weights_on_model_save"] = True

        if optim == HF_OPTIM:
            del ds_config_dict["optimizer"]

        if scheduler == HF_SCHEDULER:
            del ds_config_dict["scheduler"]

        kwargs = {
            "output_dir": output_dir,
            "train_len": 128,
            "save_steps": 5,
            "learning_rate": 0.1,
            "deepspeed": ds_config_dict,
        }
        kwargs[dtype] = True

        with mockenv_context(**self.dist_env_1_gpu):
            trainer = get_regression_trainer(**kwargs)
            trainer.train()
            (a, b) = trainer.model.a.item(), trainer.model.b.item()
            state = dataclasses.asdict(trainer.state)

            checkpoint = os.path.join(output_dir, "checkpoint-5")

            # Reinitialize trainer
            trainer = get_regression_trainer(**kwargs)

            trainer.train(resume_from_checkpoint=checkpoint)
            (a1, b1) = trainer.model.a.item(), trainer.model.b.item()
            state1 = dataclasses.asdict(trainer.state)
            self.assertEqual(a, a1)
            self.assertEqual(b, b1)
            self.check_trainer_state_are_the_same(state, state1)

            # Now check with a later checkpoint that it also works when we span over one epoch
            checkpoint = os.path.join(output_dir, "checkpoint-15")

            # Reinitialize trainer and load model
            trainer = get_regression_trainer(**kwargs)

            trainer.train(resume_from_checkpoint=checkpoint)
            (a1, b1) = trainer.model.a.item(), trainer.model.b.item()
            state1 = dataclasses.asdict(trainer.state)
            self.assertEqual(a, a1)
            self.assertEqual(b, b1)
            self.check_trainer_state_are_the_same(state, state1)

            # Finally, should be able to resume with the same trainer/same deepspeed engine instance
            # XXX: but currently this not possible due DS bug: https://github.com/microsoft/DeepSpeed/issues/1612
            # trainer.train(resume_from_checkpoint=checkpoint)
            # a workaround needs to be used that re-creates the deepspeed engine

    @parameterized.expand(params, name_func=parameterized_custom_name_func)
    def test_load_state_dict_from_zero_checkpoint(self, stage, dtype):
        # test that we can load fp32 weights directly from the zero checkpoint into the current model

        output_dir = self.get_auto_remove_tmp_dir()  # "./xxx", after=False, before=False)

        ds_config_dict = self.get_config_dict(stage)

        kwargs = {
            "output_dir": output_dir,
            "train_len": 4,
            "per_device_train_batch_size": 4,
            "num_train_epochs": 1,
            "save_strategy": "steps",
            "save_steps": 1,
            "learning_rate": 0.1,
            "deepspeed": ds_config_dict,
        }
        kwargs[dtype] = True

        with mockenv_context(**self.dist_env_1_gpu):
            trainer = get_regression_trainer(**kwargs)
            trainer.train()
            (a, b) = trainer.model.a.item(), trainer.model.b.item()
            state = dataclasses.asdict(trainer.state)

            checkpoint_dir = get_last_checkpoint(output_dir)
            model = load_state_dict_from_zero_checkpoint(trainer.model, checkpoint_dir)

            (a1, b1) = model.a.item(), model.b.item()
            state1 = dataclasses.asdict(trainer.state)
            self.assertEqual(a, a1)
            self.assertEqual(b, b1)
            self.check_trainer_state_are_the_same(state, state1)

    def test_ds_config_object(self):
        # test that we can switch from zero2 to zero3 in the same process for example
        # test is_zero, etc.
        output_dir = self.get_auto_remove_tmp_dir()
        kwargs = {"output_dir": output_dir, "train_len": 8, "fp16": True}

        ds_config_zero3_dict = self.get_config_dict(ZERO3)
        ds_config_zero2_dict = self.get_config_dict(ZERO2)

        with mockenv_context(**self.dist_env_1_gpu):
            trainer = get_regression_trainer(deepspeed=ds_config_zero3_dict, **kwargs)
            self.assertTrue(is_deepspeed_zero3_enabled())

            # test we can repeat that and with train this time
            trainer = get_regression_trainer(deepspeed=ds_config_zero3_dict, **kwargs)
            trainer.train()
            self.assertTrue(is_deepspeed_zero3_enabled())

            # test zero3 is disabled
            trainer = get_regression_trainer(deepspeed=ds_config_zero2_dict, **kwargs)
            self.assertFalse(is_deepspeed_zero3_enabled())

            # check config obj
            config = deepspeed_config()
            self.assertTrue(bool(config), "Deepspeed config should be accessible")

            # with accelerate integration below line is additionally required for this test to pass
            trainer.accelerator.state._reset_state()
            del trainer
            # now weakref should gc the global and we shouldn't get anything here
            config = deepspeed_config()
            self.assertFalse(is_deepspeed_zero3_enabled())
            self.assertFalse(bool(config), "Deepspeed config should not be accessible")

    @parameterized.expand(params, name_func=parameterized_custom_name_func)
    def test_load_best_model(self, stage, dtype):
        # Test that forced deepspeed reinit doesn't break the model. the forced re-init after
        # loading the best model in Trainer is there to workaround this bug in Deepspeed
        # https://github.com/microsoft/DeepSpeed/issues/1612
        #
        # The test is derived from a repro script submitted in this Issue:
        # https://github.com/huggingface/transformers/issues/17114
        #
        # One additional feature of this test is that we use a non-AdamW optimizer to test that
        # deepspeed doesn't fallback to AdamW, which would prevent the optimizer states from loading
        # correctly

        from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer  # noqa

        output_dir = self.get_auto_remove_tmp_dir()  # "./xxx", after=False, before=False)

        ds_config_dict = self.get_config_dict(stage)
        del ds_config_dict["optimizer"]  # will use HF Trainer optimizer
        del ds_config_dict["scheduler"]  # will use HF Trainer scheduler
        ds_config_dict["zero_force_ds_cpu_optimizer"] = False  # offload is not efficient w/o CPUAdam
        # must use this setting to get the reload path exercised
        ds_config_dict["zero_optimization"]["stage3_gather_16bit_weights_on_model_save"] = True

        with mockenv_context(**self.dist_env_1_gpu):
            args_dict = {
                "per_device_train_batch_size": 1,
                "per_device_eval_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "learning_rate": 1e-4,
                "num_train_epochs": 1,
                "do_train": True,
                "do_eval": True,
                "optim": "adafactor",
                "eval_strategy": "steps",
                "eval_steps": 1,
                "save_strategy": "steps",
                "save_steps": 1,
                "load_best_model_at_end": True,
                "max_steps": 1,
                "deepspeed": ds_config_dict,
                "report_to": "none",
            }

            training_args = TrainingArguments(output_dir, **args_dict)
            tokenizer = T5Tokenizer.from_pretrained(T5_TINY)
            model = T5ForConditionalGeneration.from_pretrained(T5_TINY)

            def _add_eos_to_examples(example):
                example["input_text"] = f"question: {example['question']}  context: {example['context']}"
                example["target_text"] = example["answers"]["text"][0] if len(example["answers"]["text"]) > 0 else ""
                return example

            def _convert_to_features(example_batch):
                input_encodings = tokenizer.batch_encode_plus(
                    example_batch["input_text"], pad_to_max_length=True, max_length=512, truncation=True
                )
                target_encodings = tokenizer.batch_encode_plus(
                    example_batch["target_text"], pad_to_max_length=True, max_length=16, truncation=True
                )

                encodings = {
                    "input_ids": input_encodings["input_ids"],
                    "attention_mask": input_encodings["attention_mask"],
                    "labels": target_encodings["input_ids"],
                }

                return encodings

            def get_dataset():
                data_file = str(self.tests_dir / "fixtures/tests_samples/SQUAD/sample.json")
                data_files = {"train": data_file, "validation": data_file}
                raw_datasets = datasets.load_dataset("json", data_files=data_files, field="data")
                train_dataset = raw_datasets["train"].map(_add_eos_to_examples).map(_convert_to_features, batched=True)
                valid_dataset = deepcopy(train_dataset)
                return train_dataset, valid_dataset

            train_dataset, eval_dataset = get_dataset()

            trainer = Trainer(
                model=model,
                processing_class=tokenizer,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
            )
            trainer.train()  # crash 1 was here
            trainer.evaluate()  # crash 2 was here


@slow
@require_deepspeed
@require_torch_accelerator
class TestDeepSpeedWithLauncher(TestCasePlus):
    """This class is for testing via an external script - can do multiple gpus"""

    # Tests to devise #
    #
    # 1. predict_with_generate on multigpu - need to figure out how to give input sequences so that
    # the 2 gpus will generate prediction sequences that aren't of the same length - this is because
    # we had to code a special feature to sync the gpus when the predicted sequences aren't of the
    # same length. In general this will tested as a side-effect through a variety of other tests -
    # it'll simply hang trying to synchronize with other gpus if this problem is encountered. So as
    # long as we have a few full tests running on zero3 + predict_with_generate this should be
    # mostly covered.
    #
    # but there are 5 variations on beam search in `generate`- with identical code branched with `if
    # synced_gpus`
    #
    # 2. most tests should probably be run on both: zero2 and zero3 configs
    #

    @parameterized.expand(params, name_func=parameterized_custom_name_func)
    @require_torch_multi_accelerator
    def test_basic_distributed(self, stage, dtype):
        self.run_and_check(stage=stage, dtype=dtype, distributed=True)

    def test_do_eval_no_train(self):
        # testing only zero3 since zero2 makes no sense with inference
        self.run_and_check(
            stage=ZERO3,
            dtype=FP16,
            eval_steps=1,
            distributed=False,
            do_train=False,
            do_eval=True,
        )

    @parameterized.expand(params, name_func=parameterized_custom_name_func)
    def test_fp32_non_distributed(self, stage, dtype):
        # real model needs too much GPU memory under stage2+fp32, so using tiny random model here -
        # therefore no quality checks, just basic completion checks are done
        self.run_and_check(
            stage=stage,
            dtype=dtype,
            model_name=T5_TINY,
            distributed=False,
            do_train=True,
            do_eval=True,
            quality_checks=False,
            fp32=True,
        )

    @parameterized.expand(params, name_func=parameterized_custom_name_func)
    @require_torch_multi_accelerator
    def test_fp32_distributed(self, stage, dtype):
        # real model needs too much GPU memory under stage2+fp32, so using tiny random model here -
        # therefore no quality checks, just basic completion checks are done
        self.run_and_check(
            stage=stage,
            dtype=dtype,
            model_name=T5_TINY,
            distributed=True,
            do_train=True,
            do_eval=True,
            quality_checks=False,
            fp32=True,
        )

    @parameterized.expand(params, name_func=parameterized_custom_name_func)
    def test_resume_train_not_from_ds_checkpoint(self, stage, dtype):
        # do normal training and then resume not from the deepspeed checkpoint but explicitly from
        # the saved model dir

        do_train = True
        do_eval = False
        kwargs = {
            "stage": stage,
            "dtype": dtype,
            "eval_steps": 1,
            "distributed": True,
            "do_train": do_train,
            "do_eval": do_eval,
        }

        # 1. normal training
        output_dir = self.run_and_check(**kwargs)

        # 2. now resume explicitly from the saved weights, by passing --model_name_or_path output_dir
        # - i.e. the same path the model was saved to in step 1
        output_dir = self.run_trainer(**kwargs, model_name=output_dir)

        self.do_checks(output_dir, do_train=do_train, do_eval=do_eval)

    @parameterized.expand(["bf16", "fp16", "fp32"])
    @require_torch_multi_accelerator
    def test_inference(self, dtype):
        if dtype == "bf16" and not is_torch_bf16_available_on_device(torch_device):
            self.skipTest(reason="test requires bfloat16 hardware support")

        # this is just inference, so no optimizer should be loaded
        # it only works for z3 (makes no sense with z1-z2)
        fp32 = True if dtype == "fp32" else False
        self.run_and_check(
            stage=ZERO3,
            dtype=FP16,
            model_name=T5_TINY,
            distributed=True,
            do_train=False,
            do_eval=True,
            quality_checks=False,
            fp32=fp32,
        )

    def do_checks(self, output_dir, do_train=True, do_eval=True, quality_checks=True):
        if do_train:
            train_metrics = load_json(os.path.join(output_dir, "train_results.json"))
            self.assertIn("train_samples_per_second", train_metrics)
            if quality_checks:
                self.assertGreater(train_metrics["train_samples_per_second"], 0.5)

        if do_eval:
            eval_metrics = load_json(os.path.join(output_dir, "eval_results.json"))
            self.assertIn("eval_bleu", eval_metrics)
            if quality_checks:
                self.assertGreater(eval_metrics["eval_bleu"], 1)

    # XXX: need to do better validation beyond just that the run was successful
    def run_and_check(
        self,
        stage,
        dtype,
        model_name: str = T5_SMALL,
        eval_steps: int = 10,
        distributed: bool = True,
        do_train: bool = True,
        do_eval: bool = True,
        quality_checks: bool = True,
        fp32: bool = False,
        extra_args_str: str = None,
        remove_args_str: str = None,
    ):
        # we are doing quality testing so using a small real model
        output_dir = self.run_trainer(
            stage=stage,
            dtype=dtype,
            model_name=model_name,
            eval_steps=eval_steps,
            num_train_epochs=1,
            do_train=do_train,
            do_eval=do_eval,
            distributed=distributed,
            fp32=fp32,
            extra_args_str=extra_args_str,
            remove_args_str=remove_args_str,
        )

        self.do_checks(output_dir, do_train=do_train, do_eval=do_eval, quality_checks=quality_checks)

        return output_dir

    def run_trainer(
        self,
        stage: str,
        dtype: str,
        model_name: str,
        eval_steps: int = 10,
        num_train_epochs: int = 1,
        do_train: bool = False,
        do_eval: bool = True,
        distributed: bool = True,
        fp32: bool = False,
        extra_args_str: str = None,
        remove_args_str: str = None,
    ):
        max_len = 32
        data_dir = self.test_file_dir / "../fixtures/tests_samples/wmt_en_ro"
        output_dir = self.get_auto_remove_tmp_dir()
        args = f"""
            --model_name_or_path {model_name}
            --train_file {data_dir}/train.json
            --validation_file {data_dir}/val.json
            --output_dir {output_dir}
            --overwrite_output_dir
            --max_source_length {max_len}
            --max_target_length {max_len}
            --val_max_target_length {max_len}
            --warmup_steps 8
            --predict_with_generate
            --save_steps 0
            --eval_steps {eval_steps}
            --group_by_length
            --label_smoothing_factor 0.1
            --source_lang en
            --target_lang ro
            --report_to none
        """.split()
        args.extend(["--source_prefix", '"translate English to Romanian: "'])

        if not fp32:
            args.extend([f"--{dtype}"])

        actions = 0
        if do_train:
            actions += 1
            args.extend(
                f"""
            --do_train
            --num_train_epochs {str(num_train_epochs)}
            --max_train_samples 16
            --per_device_train_batch_size 2
            --learning_rate 3e-3
            """.split()
            )

        if do_eval:
            actions += 1
            args.extend(
                """
            --do_eval
            --max_eval_samples 16
            --per_device_eval_batch_size 2
            """.split()
            )

        assert actions > 0, "need at least do_train or do_eval for the test to run"

        if extra_args_str is not None:
            args.extend(extra_args_str.split())

        # currently only works for bool args
        if remove_args_str is not None:
            remove_args = remove_args_str.split()
            args = [x for x in args if x not in remove_args]

        ds_args = f"--deepspeed {self.test_file_dir_str}/ds_config_{stage}.json".split()
        script = [f"{self.examples_dir_str}/pytorch/translation/run_translation.py"]
        launcher = get_launcher(distributed)

        cmd = launcher + script + args + ds_args
        # keep for quick debug
        # print(" ".join([f"\nPYTHONPATH={self.src_dir_str}"] +cmd)); die
        execute_subprocess_async(cmd, env=self.get_env())

        return output_dir

    @parameterized.expand(params, name_func=parameterized_custom_name_func)
    def test_clm(self, stage, dtype):
        # this test exercises model.resize_token_embeddings() which requires param gathering outside
        # of forward - it's not used by `run_translation.py`, but it is in `run_clm.py`

        data_dir = self.tests_dir / "fixtures"
        output_dir = self.get_auto_remove_tmp_dir()
        args = f"""
            --model_name_or_path {GPT2_TINY}
            --train_file {data_dir}/sample_text.txt
            --validation_file {data_dir}/sample_text.txt
            --output_dir {output_dir}
            --overwrite_output_dir
            --do_train
            --do_eval
            --max_train_samples 16
            --max_eval_samples 16
            --per_device_train_batch_size 2
            --per_device_eval_batch_size 2
            --num_train_epochs 1
            --warmup_steps 8
            --block_size 64
            --report_to none
            """.split()

        args.extend([f"--{dtype}"])

        ds_args = f"--deepspeed {self.test_file_dir_str}/ds_config_{stage}.json".split()
        script = [f"{self.examples_dir_str}/pytorch/language-modeling/run_clm.py"]
        launcher = get_launcher(distributed=True)

        cmd = launcher + script + args + ds_args
        # keep for quick debug
        # print(" ".join([f"\nPYTHONPATH={self.src_dir_str}"] +cmd)); die
        execute_subprocess_async(cmd, env=self.get_env())

    def test_clm_from_config_zero3_fp16(self):
        # this test exercises AutoModel.from_config(config) - to ensure zero.Init is called

        data_dir = self.tests_dir / "fixtures"
        output_dir = self.get_auto_remove_tmp_dir()
        args = f"""
            --model_type gpt2
            --tokenizer_name {GPT2_TINY}
            --train_file {data_dir}/sample_text.txt
            --validation_file {data_dir}/sample_text.txt
            --output_dir {output_dir}
            --overwrite_output_dir
            --do_train
            --max_train_samples 4
            --per_device_train_batch_size 2
            --num_train_epochs 1
            --warmup_steps 8
            --block_size 8
            --fp16
            --report_to none
            """.split()

        ds_args = f"--deepspeed {self.test_file_dir_str}/ds_config_zero3.json".split()
        script = [f"{self.examples_dir_str}/pytorch/language-modeling/run_clm.py"]
        launcher = get_launcher(distributed=True)

        cmd = launcher + script + args + ds_args
        # keep for quick debug
        # print(" ".join([f"\nPYTHONPATH={self.src_dir_str}"] +cmd)); die
        with CaptureStderr() as cs:
            execute_subprocess_async(cmd, env=self.get_env())
        self.assertIn("Detected DeepSpeed ZeRO-3", cs.err)
