# coding=utf-8
# Copyright 2020-present the HuggingFace Inc. team.
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
Utilities for the Trainer and TFTrainer class. Should be independent from PyTorch and TensorFlow.
"""

import copy
import gc
import os
import random
import re
import time
import tracemalloc
from typing import Any, Dict, NamedTuple, Optional, Tuple, Union

import numpy as np

from .file_utils import (
    is_sagemaker_distributed_available,
    is_tf_available,
    is_torch_available,
    is_torch_cuda_available,
    is_torch_tpu_available,
)
from .tokenization_utils_base import ExplicitEnum


def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).

    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # ^^ safe to call this function even if cuda is not available
    if is_tf_available():
        import tensorflow as tf

        tf.random.set_seed(seed)


class EvalPrediction(NamedTuple):
    """
    Evaluation output (always contains labels), to be used to compute metrics.

    Parameters:
        predictions (:obj:`np.ndarray`): Predictions of the model.
        label_ids (:obj:`np.ndarray`): Targets to be matched.
    """

    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: np.ndarray


class PredictionOutput(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Optional[np.ndarray]
    metrics: Optional[Dict[str, float]]


class TrainOutput(NamedTuple):
    global_step: int
    training_loss: float
    metrics: Dict[str, float]


PREFIX_CHECKPOINT_DIR = "checkpoint"
_re_checkpoint = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"\-(\d+)$")


def get_last_checkpoint(folder):
    content = os.listdir(folder)
    checkpoints = [
        path
        for path in content
        if _re_checkpoint.search(path) is not None and os.path.isdir(os.path.join(folder, path))
    ]
    if len(checkpoints) == 0:
        return
    return os.path.join(folder, max(checkpoints, key=lambda x: int(_re_checkpoint.search(x).groups()[0])))


class EvaluationStrategy(ExplicitEnum):
    NO = "no"
    STEPS = "steps"
    EPOCH = "epoch"


class BestRun(NamedTuple):
    """
    The best run found by an hyperparameter search (see :class:`~transformers.Trainer.hyperparameter_search`).

    Parameters:
        run_id (:obj:`str`):
            The id of the best run (if models were saved, the corresponding checkpoint will be in the folder ending
            with run-{run_id}).
        objective (:obj:`float`):
            The objective that was obtained for this run.
        hyperparameters (:obj:`Dict[str, Any]`):
            The hyperparameters picked to get this run.
    """

    run_id: str
    objective: float
    hyperparameters: Dict[str, Any]


def default_compute_objective(metrics: Dict[str, float]) -> float:
    """
    The default objective to maximize/minimize when doing an hyperparameter search. It is the evaluation loss if no
    metrics are provided to the :class:`~transformers.Trainer`, the sum of all metrics otherwise.

    Args:
        metrics (:obj:`Dict[str, float]`): The metrics returned by the evaluate method.

    Return:
        :obj:`float`: The objective to minimize or maximize
    """
    metrics = copy.deepcopy(metrics)
    loss = metrics.pop("eval_loss", None)
    _ = metrics.pop("epoch", None)
    # Remove speed metrics
    speed_metrics = [m for m in metrics.keys() if m.endswith("_runtime") or m.endswith("_samples_per_second")]
    for sm in speed_metrics:
        _ = metrics.pop(sm, None)
    return loss if len(metrics) == 0 else sum(metrics.values())


def default_hp_space_optuna(trial) -> Dict[str, float]:
    from .integrations import is_optuna_available

    assert is_optuna_available(), "This function needs Optuna installed: `pip install optuna`"
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 1, 5),
        "seed": trial.suggest_int("seed", 1, 40),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16, 32, 64]),
    }


def default_hp_space_ray(trial) -> Dict[str, float]:
    from .integrations import is_ray_tune_available

    assert is_ray_tune_available(), "This function needs ray installed: `pip " "install ray[tune]`"
    from ray import tune

    return {
        "learning_rate": tune.loguniform(1e-6, 1e-4),
        "num_train_epochs": tune.choice(list(range(1, 6))),
        "seed": tune.uniform(1, 40),
        "per_device_train_batch_size": tune.choice([4, 8, 16, 32, 64]),
    }


class HPSearchBackend(ExplicitEnum):
    OPTUNA = "optuna"
    RAY = "ray"


default_hp_space = {
    HPSearchBackend.OPTUNA: default_hp_space_optuna,
    HPSearchBackend.RAY: default_hp_space_ray,
}


def is_main_process(local_rank):
    """
    Whether or not the current process is the local process, based on `xm.get_ordinal()` (for TPUs) first, then on
    `local_rank`.
    """
    if is_torch_tpu_available():
        import torch_xla.core.xla_model as xm

        return xm.get_ordinal() == 0
    return local_rank in [-1, 0]


def total_processes_number(local_rank):
    """
    Return the number of processes launched in parallel. Works with `torch.distributed` and TPUs.
    """
    if is_torch_tpu_available():
        import torch_xla.core.xla_model as xm

        return xm.xrt_world_size()
    elif is_sagemaker_distributed_available():
        import smdistributed.dataparallel.torch.distributed as dist

        return dist.get_world_size()
    elif local_rank != -1 and is_torch_available():
        import torch

        return torch.distributed.get_world_size()
    return 1


def speed_metrics(split, start_time, num_samples=None):
    """
    Measure and return speed performance metrics.

    This function requires a time snapshot `start_time` before the operation to be measured starts and this function
    should be run immediately after the operation to be measured has completed.

    Args:

    - split: name to prefix metric (like train, eval, test...)
    - start_time: operation start time
    - num_samples: number of samples processed
    """
    runtime = time.time() - start_time
    result = {f"{split}_runtime": round(runtime, 4)}
    if num_samples is not None:
        samples_per_second = 1 / (runtime / num_samples)
        result[f"{split}_samples_per_second"] = round(samples_per_second, 3)
    return result


class SchedulerType(ExplicitEnum):
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"
    CONSTANT = "constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"


class TrainerMemoryTracker:
    def __init__(self, skip_memory_metrics=False):
        if is_torch_cuda_available():
            import torch

            self.torch = torch
            self.gpu = {}
        else:
            self.torch = None

        self.stage = "none"
        self.cpu = {}
        self.init_reported = False
        self.skip_memory_metrics = skip_memory_metrics

    def start(self, stage):
        if self.skip_memory_metrics:
            return

        self.stage = stage

        if self.torch is not None:
            self.torch.cuda.reset_peak_memory_stats()
            self.torch.cuda.empty_cache()

        gc.collect()

        # gpu
        if self.torch is not None:
            self.gpu[self.stage] = {}
            self.gpu[self.stage]["alloc"] = self.torch.cuda.memory_allocated()
            self.gpu[self.stage]["peaked"] = 0

        # cpu
        self.cpu[self.stage] = {}
        tracemalloc.start()

    def stop(self):
        if self.skip_memory_metrics:
            return

        if self.torch is not None:
            self.torch.cuda.empty_cache()

        gc.collect()

        # gpu
        if self.torch is not None:
            mem_cur = self.torch.cuda.memory_allocated()
            # this is the difference between the start and the end allocated memory
            self.gpu[self.stage]["alloc"] = mem_cur - self.gpu[self.stage]["alloc"]  # can be negative
            # this is the difference if any between the start and the peak
            self.gpu[self.stage]["peaked"] = max(0, self.torch.cuda.max_memory_allocated() - mem_cur)

        # cpu
        cpu_mem_used_delta, cpu_mem_used_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()  # reset accounting
        self.cpu[self.stage]["alloc"] = cpu_mem_used_delta  # can be negative
        self.cpu[self.stage]["peaked"] = max(0, cpu_mem_used_peak - cpu_mem_used_delta)

    def update_metrics(self, stage, metrics):
        if self.skip_memory_metrics:
            return
        # since we don't have a way to return init metrics, we push them into the first of train/val/predict
        stages = [stage]
        if not self.init_reported:
            stages.insert(0, "init")
            self.init_reported = True

        for stage in stages:
            for t in ["alloc", "peaked"]:
                metrics[f"{stage}_mem_cpu_{t}_delta"] = self.cpu[stage][t]
                if self.torch is not None:
                    metrics[f"{stage}_mem_gpu_{t}_delta"] = self.gpu[stage][t]
