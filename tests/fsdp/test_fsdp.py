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

import itertools
import os
import subprocess
import unittest
from copy import deepcopy
from functools import partial

from parameterized import parameterized

import tests.trainer.test_trainer
from tests.trainer.test_trainer import TrainerIntegrationCommon  # noqa
from transformers import is_torch_available
from transformers.testing_utils import (
    TestCasePlus,
    backend_device_count,
    execute_subprocess_async,
    mockenv_context,
    require_accelerate,
    require_fsdp,
    require_torch_accelerator,
    require_torch_multi_accelerator,
    slow,
    torch_device,
)
from transformers.trainer_callback import TrainerState
from transformers.trainer_utils import FSDPOption, set_seed
from transformers.utils import is_accelerate_available, is_torch_bf16_available_on_device


if is_torch_available():
    from transformers.pytorch_utils import is_torch_greater_or_equal_than_2_1
    from transformers.trainer import FSDP_MODEL_NAME
else:
    is_torch_greater_or_equal_than_2_1 = False

# default torch.distributed port
DEFAULT_MASTER_PORT = "10999"
dtypes = ["fp16"]
if is_torch_bf16_available_on_device(torch_device):
    dtypes += ["bf16"]
sharding_strategies = ["full_shard", "shard_grad_op"]
state_dict_types = ["FULL_STATE_DICT", "SHARDED_STATE_DICT"]
set_seed(42)
params = list(itertools.product(sharding_strategies, dtypes))


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


if is_torch_available():
    from tests.trainer.test_trainer import (  # noqa
        RegressionModelConfig,
        RegressionPreTrainedModel,
    )

    # hack to restore original logging level pre #21700
    get_regression_trainer = partial(tests.trainer.test_trainer.get_regression_trainer, log_level="info")

require_fsdp_version = require_fsdp
if is_accelerate_available():
    from accelerate.utils.constants import (
        FSDP_PYTORCH_VERSION,
        FSDP_SHARDING_STRATEGY,
    )

    require_fsdp_version = partial(require_fsdp, min_version=FSDP_PYTORCH_VERSION)


def get_launcher(distributed=False, use_accelerate=False):
    # 1. explicitly set --num_nodes=1 just in case these tests end up run on a multi-node setup
    # - it won't be able to handle that
    # 2. for now testing with just 2 gpus max (since some quality tests may give different
    # results with mode gpus because we use very little data)
    num_gpus = min(2, backend_device_count(torch_device)) if distributed else 1
    master_port = get_master_port(real_launcher=True)
    if use_accelerate:
        return f"""accelerate launch
        --num_processes {num_gpus}
        --main_process_port {master_port}
        --use_fsdp
        --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP
        --fsdp_state_dict_type SHARDED_STATE_DICT
        --fsdp_transformer_layer_cls_to_wrap BertLayer""".split()
    return f"torchrun --nnodes 1 --nproc-per-node {num_gpus} --master-port {master_port}".split()


def _parameterized_custom_name_func(func, param_num, param):
    # customize the test name generator function as we want both params to appear in the sub-test
    # name, as by default it shows only the first param
    param_based_name = parameterized.to_safe_name("_".join(str(x) for x in param.args))
    return f"{func.__name__}_{param_based_name}"


@require_accelerate
@require_torch_accelerator
@require_fsdp_version
class TrainerIntegrationFSDP(TestCasePlus, TrainerIntegrationCommon):
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

        self.fsdp_config = {
            "backward_prefetch": "backward_pre",
            "forward_prefetch": "False",
            "limit_all_gathers": "False",
            "use_orig_params": "True",
            "sync_module_states": "True",
            "cpu_ram_efficient_loading": "True",
            "activation_checkpointing": "False",
            "min_num_params": 1,
        }

    def tearDown(self):
        super().tearDown()

    @parameterized.expand(params, name_func=_parameterized_custom_name_func)
    def test_fsdp_config(self, sharding_strategy, dtype):
        output_dir = self.get_auto_remove_tmp_dir()
        kwargs = {
            "output_dir": output_dir,
            "train_len": 128,
            "save_steps": 5,
            "learning_rate": 0.1,
            "fsdp": f"{sharding_strategy} offload auto_wrap",
            "fsdp_config": self.fsdp_config,
        }
        kwargs[dtype] = True
        with mockenv_context(**self.dist_env_1_gpu):
            trainer = get_regression_trainer(**kwargs)
            self.assertEqual(trainer.args.fsdp[0], sharding_strategy)
            self.assertEqual(trainer.args.fsdp[1], FSDPOption.OFFLOAD)
            self.assertEqual(trainer.args.fsdp[2], FSDPOption.AUTO_WRAP)
            for k, v in trainer.args.fsdp_config.items():
                self.assertEqual(v, self.fsdp_config[k])
            self.assertEqual(os.environ.get("ACCELERATE_USE_FSDP", "false"), "true")

    @parameterized.expand(params, name_func=_parameterized_custom_name_func)
    def test_fsdp_config_transformers_auto_wrap(self, sharding_strategy, dtype):
        output_dir = self.get_auto_remove_tmp_dir()
        fsdp_config = deepcopy(self.fsdp_config)
        del fsdp_config["min_num_params"]
        fsdp_config["transformer_layer_cls_to_wrap"] = "BertLayer"
        kwargs = {
            "output_dir": output_dir,
            "train_len": 128,
            "save_steps": 5,
            "learning_rate": 0.1,
            "fsdp": f"{sharding_strategy} offload auto_wrap",
            "fsdp_config": fsdp_config,
        }
        kwargs[dtype] = True
        prefix = "FSDP_"
        with mockenv_context(**self.dist_env_1_gpu):
            trainer = get_regression_trainer(**kwargs)
            self.assertEqual(trainer.args.fsdp[0], sharding_strategy)
            self.assertEqual(trainer.args.fsdp[1], FSDPOption.OFFLOAD)
            self.assertEqual(trainer.args.fsdp[2], FSDPOption.AUTO_WRAP)
            fsdp_sharding_strategy = str(FSDP_SHARDING_STRATEGY.index(sharding_strategy.upper()) + 1)
            self.assertEqual(os.environ[f"{prefix}SHARDING_STRATEGY"], fsdp_sharding_strategy)
            self.assertEqual(os.environ[f"{prefix}OFFLOAD_PARAMS"], "true")
            self.assertEqual(os.environ[f"{prefix}AUTO_WRAP_POLICY"], "TRANSFORMER_BASED_WRAP")
            self.assertEqual(
                os.environ[f"{prefix}TRANSFORMER_CLS_TO_WRAP"], ",".join(fsdp_config["transformer_layer_cls_to_wrap"])
            )
            self.assertEqual(os.environ[f"{prefix}BACKWARD_PREFETCH"], fsdp_config["backward_prefetch"].upper())
            self.assertEqual(os.environ[f"{prefix}FORWARD_PREFETCH"], fsdp_config["forward_prefetch"])
            self.assertEqual(os.environ[f"{prefix}USE_ORIG_PARAMS"], fsdp_config["use_orig_params"])
            self.assertEqual(os.environ[f"{prefix}SYNC_MODULE_STATES"], fsdp_config["sync_module_states"])
            self.assertEqual(
                os.environ[f"{prefix}CPU_RAM_EFFICIENT_LOADING"], fsdp_config["cpu_ram_efficient_loading"]
            )
            self.assertEqual(os.environ.get("ACCELERATE_USE_FSDP", "false"), "true")

    @parameterized.expand(params, name_func=_parameterized_custom_name_func)
    @require_torch_multi_accelerator
    @slow
    def test_basic_run(self, sharding_strategy, dtype):
        launcher = get_launcher(distributed=True, use_accelerate=False)
        output_dir = self.get_auto_remove_tmp_dir()
        args = self.get_base_args(output_dir, 1, 50).split() + [f"--{dtype}"]
        fsdp_args = ["--fsdp", f"{sharding_strategy} auto_wrap", "--fsdp_transformer_layer_cls_to_wrap", "BertLayer"]
        script = [f"{self.examples_dir_str}/pytorch/text-classification/run_glue.py"]
        cmd = launcher + script + args + fsdp_args
        execute_subprocess_async(cmd, env=self.get_env())

    @parameterized.expand(params, name_func=_parameterized_custom_name_func)
    @require_torch_multi_accelerator
    @slow
    def test_basic_run_with_gradient_accumulation(self, sharding_strategy, dtype):
        launcher = get_launcher(distributed=True, use_accelerate=False)
        output_dir = self.get_auto_remove_tmp_dir()
        args = self.get_base_args(output_dir, 1, 50).split() + [f"--{dtype}", "--gradient_accumulation_steps", "2"]
        fsdp_args = ["--fsdp", f"{sharding_strategy} auto_wrap", "--fsdp_transformer_layer_cls_to_wrap", "BertLayer"]
        script = [f"{self.examples_dir_str}/pytorch/text-classification/run_glue.py"]
        cmd = launcher + script + args + fsdp_args
        execute_subprocess_async(cmd, env=self.get_env())

    @parameterized.expand(dtypes)
    @require_torch_multi_accelerator
    @slow
    @unittest.skipIf(not is_torch_greater_or_equal_than_2_1, reason="This test on pytorch 2.0 takes 4 hours.")
    def test_basic_run_with_cpu_offload(self, dtype):
        launcher = get_launcher(distributed=True, use_accelerate=False)
        output_dir = self.get_auto_remove_tmp_dir()
        args = self.get_base_args(output_dir, 1, 50).split() + [f"--{dtype}", "--max_steps", "10"]
        fsdp_args = ["--fsdp", "full_shard auto_wrap offload", "--fsdp_transformer_layer_cls_to_wrap", "BertLayer"]
        script = [f"{self.examples_dir_str}/pytorch/text-classification/run_glue.py"]
        cmd = launcher + script + args + fsdp_args
        execute_subprocess_async(cmd, env=self.get_env())

    @parameterized.expand(state_dict_types, name_func=_parameterized_custom_name_func)
    @require_torch_multi_accelerator
    @slow
    def test_training_and_can_resume_normally(self, state_dict_type):
        output_dir = self.get_auto_remove_tmp_dir("./xxx", after=False)

        sharding_strategy = "full_shard"
        use_accelerate = state_dict_type == "SHARDED_STATE_DICT"
        launcher = get_launcher(True, use_accelerate=use_accelerate)
        args = self.get_base_args(output_dir, 2, 25).split()
        script = [f"{self.examples_dir_str}/pytorch/text-classification/run_glue.py"]
        logs = self.run_cmd_and_get_logs(use_accelerate, sharding_strategy, launcher, script, args, output_dir)

        # resume from ckpt
        checkpoint = os.path.join(output_dir, "checkpoint-115")
        resume_args = args + f"--resume_from_checkpoint {checkpoint}".split()

        is_fsdp_ckpt = os.path.isdir(checkpoint) and (
            # this checks the FSDP state dict when `SHARDED_STATE_DICT` is used
            any(
                FSDP_MODEL_NAME in folder_name
                for folder_name in os.listdir(checkpoint)
                if os.path.isdir(os.path.join(checkpoint, folder_name))
            )
            # this checks the FSDP state dict when `FULL_STATE_DICT` is used
            or os.path.isfile(os.path.join(checkpoint, f"{FSDP_MODEL_NAME}.bin"))
        )
        self.assertTrue(is_fsdp_ckpt)

        logs_resume = self.run_cmd_and_get_logs(
            use_accelerate, sharding_strategy, launcher, script, resume_args, output_dir
        )

        for log, log1 in zip(logs, logs_resume):
            if "learning_rate" in log:
                self.assertAlmostEqual(log["learning_rate"], log1["learning_rate"], delta=1e-5)

    @require_torch_multi_accelerator
    @slow
    @require_torch_accelerator
    @require_fsdp
    def test_fsdp_cpu_offloading(self):
        try:
            subprocess.run(
                "accelerate launch utils/testing_scripts/fsdp_cpu_offloading.py --config utils/testing_scripts/dummy_fsdp_config.yml",
                shell=True,
                check=True,
            )
        except:  # noqa
            raise AssertionError("CPU offloading failed with FSDP!")

    def run_cmd_and_get_logs(self, use_accelerate, sharding_strategy, launcher, script, args, output_dir):
        if not use_accelerate:
            fsdp_args = [
                "--fsdp",
                f"{sharding_strategy} auto_wrap",
                "--fsdp_transformer_layer_cls_to_wrap",
                "BertLayer",
            ]
            cmd = launcher + script + args + fsdp_args
        else:
            fsdp_config = f"""
                --fsdp_sharding_strategy {FSDP_SHARDING_STRATEGY.index(sharding_strategy.upper()) + 1}
            """.split()
            cmd = launcher + fsdp_config + script + args

        # keep for quick debug
        # print(" ".join([f"\nPYTHONPATH={self.src_dir_str}"] +cmd)); die
        execute_subprocess_async(cmd, env=self.get_env())
        logs = TrainerState.load_from_json(os.path.join(output_dir, "trainer_state.json")).log_history
        return logs

    def get_base_args(self, output_dir, num_epochs, logging_steps):
        return f"""
            --model_name_or_path google-bert/bert-base-cased
            --task_name mrpc
            --output_dir {output_dir}
            --overwrite_output_dir
            --do_train
            --max_seq_length 128
            --per_device_train_batch_size 16
            --learning_rate 5e-5
            --num_train_epochs {num_epochs}
            --lr_scheduler_type cosine
            --logging_steps {logging_steps}
            --save_strategy epoch
            --do_eval
            --eval_strategy epoch
            --report_to none
        """
