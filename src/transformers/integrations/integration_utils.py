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
Integrations with other Python libraries.
"""

import functools
import importlib.metadata
import importlib.util
import json
import numbers
import os
import pickle
import shutil
import sys
import tempfile
from dataclasses import asdict, fields
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Union

import numpy as np
import packaging.version

from .. import PreTrainedModel, TFPreTrainedModel
from .. import __version__ as version
from ..utils import (
    PushToHubMixin,
    flatten_dict,
    is_datasets_available,
    is_pandas_available,
    is_tf_available,
    is_torch_available,
    logging,
)


logger = logging.get_logger(__name__)

if is_torch_available():
    import torch

# comet_ml requires to be imported before any ML frameworks
_MIN_COMET_VERSION = "3.43.2"
try:
    _comet_version = importlib.metadata.version("comet_ml")
    _is_comet_installed = True

    _is_comet_recent_enough = packaging.version.parse(_comet_version) >= packaging.version.parse(_MIN_COMET_VERSION)

    # Check if the Comet API Key is set
    import comet_ml

    if comet_ml.config.get_config("comet.api_key") is not None:
        _is_comet_configured = True
    else:
        _is_comet_configured = False
except (importlib.metadata.PackageNotFoundError, ImportError, ValueError, TypeError, AttributeError, KeyError):
    _comet_version = None
    _is_comet_installed = False
    _is_comet_recent_enough = False
    _is_comet_configured = False

_has_neptune = (
    importlib.util.find_spec("neptune") is not None or importlib.util.find_spec("neptune-client") is not None
)
if TYPE_CHECKING and _has_neptune:
    try:
        _neptune_version = importlib.metadata.version("neptune")
        logger.info(f"Neptune version {_neptune_version} available.")
    except importlib.metadata.PackageNotFoundError:
        try:
            _neptune_version = importlib.metadata.version("neptune-client")
            logger.info(f"Neptune-client version {_neptune_version} available.")
        except importlib.metadata.PackageNotFoundError:
            _has_neptune = False

from .. import modelcard  # noqa: E402
from ..trainer_callback import ProgressCallback, TrainerCallback  # noqa: E402
from ..trainer_utils import PREFIX_CHECKPOINT_DIR, BestRun, IntervalStrategy  # noqa: E402
from ..training_args import ParallelMode  # noqa: E402
from ..utils import ENV_VARS_TRUE_VALUES, is_torch_xla_available  # noqa: E402


# Integration functions:
def is_wandb_available():
    # any value of WANDB_DISABLED disables wandb
    if os.getenv("WANDB_DISABLED", "").upper() in ENV_VARS_TRUE_VALUES:
        logger.warning(
            "Using the `WANDB_DISABLED` environment variable is deprecated and will be removed in v5. Use the "
            "--report_to flag to control the integrations used for logging result (for instance --report_to none)."
        )
        return False
    return importlib.util.find_spec("wandb") is not None


def is_clearml_available():
    return importlib.util.find_spec("clearml") is not None


def is_comet_available():
    if os.getenv("COMET_MODE", "").upper() == "DISABLED":
        logger.warning(
            "Using the `COMET_MODE=DISABLED` environment variable is deprecated and will be removed in v5. Use the "
            "--report_to flag to control the integrations used for logging result (for instance --report_to none)."
        )
        return False

    if _is_comet_installed is False:
        return False

    if _is_comet_recent_enough is False:
        logger.warning(
            "comet_ml version %s is installed, but version %s or higher is required. "
            "Please update comet_ml to the latest version to enable Comet logging with pip install 'comet-ml>=%s'.",
            _comet_version,
            _MIN_COMET_VERSION,
            _MIN_COMET_VERSION,
        )
        return False

    if _is_comet_configured is False:
        logger.warning(
            "comet_ml is installed but the Comet API Key is not configured. "
            "Please set the `COMET_API_KEY` environment variable to enable Comet logging. "
            "Check out the documentation for other ways of configuring it: "
            "https://www.comet.com/docs/v2/guides/experiment-management/configure-sdk/#set-the-api-key"
        )
        return False

    return True


def is_tensorboard_available():
    return importlib.util.find_spec("tensorboard") is not None or importlib.util.find_spec("tensorboardX") is not None


def is_optuna_available():
    return importlib.util.find_spec("optuna") is not None


def is_ray_available():
    return importlib.util.find_spec("ray") is not None


def is_ray_tune_available():
    if not is_ray_available():
        return False
    return importlib.util.find_spec("ray.tune") is not None


def is_sigopt_available():
    return importlib.util.find_spec("sigopt") is not None


def is_azureml_available():
    if importlib.util.find_spec("azureml") is None:
        return False
    if importlib.util.find_spec("azureml.core") is None:
        return False
    return importlib.util.find_spec("azureml.core.run") is not None


def is_mlflow_available():
    if os.getenv("DISABLE_MLFLOW_INTEGRATION", "FALSE").upper() == "TRUE":
        return False
    return importlib.util.find_spec("mlflow") is not None


def is_dagshub_available():
    return None not in [importlib.util.find_spec("dagshub"), importlib.util.find_spec("mlflow")]


def is_neptune_available():
    return _has_neptune


def is_codecarbon_available():
    return importlib.util.find_spec("codecarbon") is not None


def is_flytekit_available():
    return importlib.util.find_spec("flytekit") is not None


def is_flyte_deck_standard_available():
    if not is_flytekit_available():
        return False
    return importlib.util.find_spec("flytekitplugins.deck") is not None


def is_dvclive_available():
    return importlib.util.find_spec("dvclive") is not None


def hp_params(trial):
    if is_optuna_available():
        import optuna

        if isinstance(trial, optuna.trial.BaseTrial):
            return trial.params
    if is_ray_tune_available():
        if isinstance(trial, dict):
            return trial

    if is_sigopt_available():
        if isinstance(trial, dict):
            return trial

    if is_wandb_available():
        if isinstance(trial, dict):
            return trial

    raise RuntimeError(f"Unknown type for trial {trial.__class__}")


def run_hp_search_optuna(trainer, n_trials: int, direction: str, **kwargs) -> BestRun:
    import optuna
    from accelerate.utils.memory import release_memory

    if trainer.args.process_index == 0:

        def _objective(trial: optuna.Trial, checkpoint_dir=None):
            checkpoint = None
            if checkpoint_dir:
                for subdir in os.listdir(checkpoint_dir):
                    if subdir.startswith(PREFIX_CHECKPOINT_DIR):
                        checkpoint = os.path.join(checkpoint_dir, subdir)
            trainer.objective = None
            if trainer.args.world_size > 1:
                if trainer.args.parallel_mode != ParallelMode.DISTRIBUTED:
                    raise RuntimeError("only support DDP optuna HPO for ParallelMode.DISTRIBUTED currently.")
                trainer.hp_space(trial)
                fixed_trial = optuna.trial.FixedTrial(trial.params, trial.number)
                trial_main_rank_list = [fixed_trial]
                torch.distributed.broadcast_object_list(trial_main_rank_list, src=0)
                trainer.train(resume_from_checkpoint=checkpoint, trial=trial)
            else:
                trainer.train(resume_from_checkpoint=checkpoint, trial=trial)
            # If there hasn't been any evaluation during the training loop.
            if getattr(trainer, "objective", None) is None:
                metrics = trainer.evaluate()
                trainer.objective = trainer.compute_objective(metrics)

            # Free GPU memory
            trainer.model_wrapped, trainer.model = release_memory(trainer.model_wrapped, trainer.model)
            trainer.accelerator.clear()

            return trainer.objective

        timeout = kwargs.pop("timeout", None)
        n_jobs = kwargs.pop("n_jobs", 1)
        gc_after_trial = kwargs.pop("gc_after_trial", False)
        directions = direction if isinstance(direction, list) else None
        direction = None if directions is not None else direction
        study = optuna.create_study(direction=direction, directions=directions, **kwargs)
        study.optimize(_objective, n_trials=n_trials, timeout=timeout, n_jobs=n_jobs, gc_after_trial=gc_after_trial)
        if not study._is_multi_objective():
            best_trial = study.best_trial
            return BestRun(str(best_trial.number), best_trial.value, best_trial.params)
        else:
            best_trials = study.best_trials
            return [BestRun(str(best.number), best.values, best.params) for best in best_trials]
    else:
        for i in range(n_trials):
            trainer.objective = None
            trial_main_rank_list = [None]
            if trainer.args.parallel_mode != ParallelMode.DISTRIBUTED:
                raise RuntimeError("only support DDP optuna HPO for ParallelMode.DISTRIBUTED currently.")
            torch.distributed.broadcast_object_list(trial_main_rank_list, src=0)
            trainer.train(resume_from_checkpoint=None, trial=trial_main_rank_list[0])
            # If there hasn't been any evaluation during the training loop.
            if getattr(trainer, "objective", None) is None:
                metrics = trainer.evaluate()
                trainer.objective = trainer.compute_objective(metrics)
        return None


def run_hp_search_ray(trainer, n_trials: int, direction: str, **kwargs) -> BestRun:
    import ray
    import ray.train

    def _objective(trial: dict, local_trainer):
        try:
            from transformers.utils.notebook import NotebookProgressCallback

            if local_trainer.pop_callback(NotebookProgressCallback):
                local_trainer.add_callback(ProgressCallback)
        except ModuleNotFoundError:
            pass

        local_trainer.objective = None

        checkpoint = ray.train.get_checkpoint()
        if checkpoint:
            # Upon trial resume, the local_trainer's objective gets reset to None.
            # If `local_trainer.train` is a noop (training has already reached
            # the target number of epochs/steps), then this would
            # trigger an unnecessary extra checkpoint at the end of training.
            # -> Set the objective to a dummy value upon resume as a workaround.
            local_trainer.objective = "objective"

            with checkpoint.as_directory() as checkpoint_dir:
                checkpoint_path = next(Path(checkpoint_dir).glob(f"{PREFIX_CHECKPOINT_DIR}*")).as_posix()
                local_trainer.train(resume_from_checkpoint=checkpoint_path, trial=trial)
        else:
            local_trainer.train(trial=trial)

        # If there hasn't been any evaluation during the training loop.
        if getattr(local_trainer, "objective", None) is None:
            metrics = local_trainer.evaluate()
            local_trainer.objective = local_trainer.compute_objective(metrics)

            metrics.update({"objective": local_trainer.objective, "done": True})

            with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                local_trainer._tune_save_checkpoint(checkpoint_dir=temp_checkpoint_dir)
                checkpoint = ray.train.Checkpoint.from_directory(temp_checkpoint_dir)
                ray.train.report(metrics, checkpoint=checkpoint)

    if not trainer._memory_tracker.skip_memory_metrics:
        from ..trainer_utils import TrainerMemoryTracker

        logger.warning(
            "Memory tracking for your Trainer is currently "
            "enabled. Automatically disabling the memory tracker "
            "since the memory tracker is not serializable."
        )
        trainer._memory_tracker = TrainerMemoryTracker(skip_memory_metrics=True)

    # The model and TensorBoard writer do not pickle so we have to remove them (if they exists)
    # while doing the ray hp search.
    _tb_writer = trainer.pop_callback(TensorBoardCallback)
    trainer.model = None

    # Setup default `resources_per_trial`.
    if "resources_per_trial" not in kwargs:
        # Default to 1 CPU and 1 GPU (if applicable) per trial.
        kwargs["resources_per_trial"] = {"cpu": 1}
        if trainer.args.n_gpu > 0:
            kwargs["resources_per_trial"]["gpu"] = 1
        resource_msg = "1 CPU" + (" and 1 GPU" if trainer.args.n_gpu > 0 else "")
        logger.info(
            "No `resources_per_trial` arg was passed into "
            "`hyperparameter_search`. Setting it to a default value "
            f"of {resource_msg} for each trial."
        )
    # Make sure each trainer only uses GPUs that were allocated per trial.
    gpus_per_trial = kwargs["resources_per_trial"].get("gpu", 0)
    trainer.args._n_gpu = gpus_per_trial

    # Setup default `progress_reporter`.
    if "progress_reporter" not in kwargs:
        from ray.tune import CLIReporter

        kwargs["progress_reporter"] = CLIReporter(metric_columns=["objective"])

    if "scheduler" in kwargs:
        from ray.tune.schedulers import ASHAScheduler, HyperBandForBOHB, MedianStoppingRule, PopulationBasedTraining

        # Check for `do_eval` and `eval_during_training` for schedulers that require intermediate reporting.
        if isinstance(
            kwargs["scheduler"], (ASHAScheduler, MedianStoppingRule, HyperBandForBOHB, PopulationBasedTraining)
        ) and (not trainer.args.do_eval or trainer.args.eval_strategy == IntervalStrategy.NO):
            raise RuntimeError(
                "You are using {cls} as a scheduler but you haven't enabled evaluation during training. "
                "This means your trials will not report intermediate results to Ray Tune, and "
                "can thus not be stopped early or used to exploit other trials parameters. "
                "If this is what you want, do not use {cls}. If you would like to use {cls}, "
                "make sure you pass `do_eval=True` and `eval_strategy='steps'` in the "
                "Trainer `args`.".format(cls=type(kwargs["scheduler"]).__name__)
            )

    trainable = ray.tune.with_parameters(_objective, local_trainer=trainer)

    @functools.wraps(trainable)
    def dynamic_modules_import_trainable(*args, **kwargs):
        """
        Wrapper around `tune.with_parameters` to ensure datasets_modules are loaded on each Actor.

        Without this, an ImportError will be thrown. See https://github.com/huggingface/transformers/issues/11565.

        Assumes that `_objective`, defined above, is a function.
        """
        if is_datasets_available():
            import datasets.load

            dynamic_modules_path = os.path.join(datasets.load.init_dynamic_modules(), "__init__.py")
            # load dynamic_modules from path
            spec = importlib.util.spec_from_file_location("datasets_modules", dynamic_modules_path)
            datasets_modules = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = datasets_modules
            spec.loader.exec_module(datasets_modules)
        return trainable(*args, **kwargs)

    # special attr set by tune.with_parameters
    if hasattr(trainable, "__mixins__"):
        dynamic_modules_import_trainable.__mixins__ = trainable.__mixins__

    analysis = ray.tune.run(
        dynamic_modules_import_trainable,
        config=trainer.hp_space(None),
        num_samples=n_trials,
        **kwargs,
    )
    best_trial = analysis.get_best_trial(metric="objective", mode=direction[:3], scope=trainer.args.ray_scope)
    best_run = BestRun(best_trial.trial_id, best_trial.last_result["objective"], best_trial.config, analysis)
    if _tb_writer is not None:
        trainer.add_callback(_tb_writer)
    return best_run


def run_hp_search_sigopt(trainer, n_trials: int, direction: str, **kwargs) -> BestRun:
    import sigopt

    if trainer.args.process_index == 0:
        if importlib.metadata.version("sigopt") >= "8.0.0":
            sigopt.set_project("huggingface")

            experiment = sigopt.create_experiment(
                name="huggingface-tune",
                type="offline",
                parameters=trainer.hp_space(None),
                metrics=[{"name": "objective", "objective": direction, "strategy": "optimize"}],
                parallel_bandwidth=1,
                budget=n_trials,
            )

            logger.info(f"created experiment: https://app.sigopt.com/experiment/{experiment.id}")

            for run in experiment.loop():
                with run:
                    trainer.objective = None
                    if trainer.args.world_size > 1:
                        if trainer.args.parallel_mode != ParallelMode.DISTRIBUTED:
                            raise RuntimeError("only support DDP Sigopt HPO for ParallelMode.DISTRIBUTED currently.")
                        trainer._hp_search_setup(run.run)
                        torch.distributed.broadcast_object_list(pickle.dumps(trainer.args), src=0)
                        trainer.train(resume_from_checkpoint=None)
                    else:
                        trainer.train(resume_from_checkpoint=None, trial=run.run)
                    # If there hasn't been any evaluation during the training loop.
                    if getattr(trainer, "objective", None) is None:
                        metrics = trainer.evaluate()
                        trainer.objective = trainer.compute_objective(metrics)
                    run.log_metric("objective", trainer.objective)

            best = list(experiment.get_best_runs())[0]
            best_run = BestRun(best.id, best.values["objective"].value, best.assignments)
        else:
            from sigopt import Connection

            conn = Connection()
            proxies = kwargs.pop("proxies", None)
            if proxies is not None:
                conn.set_proxies(proxies)

            experiment = conn.experiments().create(
                name="huggingface-tune",
                parameters=trainer.hp_space(None),
                metrics=[{"name": "objective", "objective": direction, "strategy": "optimize"}],
                parallel_bandwidth=1,
                observation_budget=n_trials,
                project="huggingface",
            )
            logger.info(f"created experiment: https://app.sigopt.com/experiment/{experiment.id}")

            while experiment.progress.observation_count < experiment.observation_budget:
                suggestion = conn.experiments(experiment.id).suggestions().create()
                trainer.objective = None
                if trainer.args.world_size > 1:
                    if trainer.args.parallel_mode != ParallelMode.DISTRIBUTED:
                        raise RuntimeError("only support DDP Sigopt HPO for ParallelMode.DISTRIBUTED currently.")
                    trainer._hp_search_setup(suggestion)
                    torch.distributed.broadcast_object_list(pickle.dumps(trainer.args), src=0)
                    trainer.train(resume_from_checkpoint=None)
                else:
                    trainer.train(resume_from_checkpoint=None, trial=suggestion)
                # If there hasn't been any evaluation during the training loop.
                if getattr(trainer, "objective", None) is None:
                    metrics = trainer.evaluate()
                    trainer.objective = trainer.compute_objective(metrics)

                values = [{"name": "objective", "value": trainer.objective}]
                obs = conn.experiments(experiment.id).observations().create(suggestion=suggestion.id, values=values)
                logger.info(f"[suggestion_id, observation_id]: [{suggestion.id}, {obs.id}]")
                experiment = conn.experiments(experiment.id).fetch()

            best = list(conn.experiments(experiment.id).best_assignments().fetch().iterate_pages())[0]
            best_run = BestRun(best.id, best.value, best.assignments)
        return best_run
    else:
        for i in range(n_trials):
            trainer.objective = None
            args_main_rank = list(pickle.dumps(trainer.args))
            if trainer.args.parallel_mode != ParallelMode.DISTRIBUTED:
                raise RuntimeError("only support DDP Sigopt HPO for ParallelMode.DISTRIBUTED currently.")
            torch.distributed.broadcast_object_list(args_main_rank, src=0)
            args = pickle.loads(bytes(args_main_rank))
            for key, value in asdict(args).items():
                if key != "local_rank":
                    setattr(trainer.args, key, value)
            trainer.train(resume_from_checkpoint=None)
            # If there hasn't been any evaluation during the training loop.
            if getattr(trainer, "objective", None) is None:
                metrics = trainer.evaluate()
                trainer.objective = trainer.compute_objective(metrics)
        return None


def run_hp_search_wandb(trainer, n_trials: int, direction: str, **kwargs) -> BestRun:
    from ..integrations import is_wandb_available

    if not is_wandb_available():
        raise ImportError("This function needs wandb installed: `pip install wandb`")
    import wandb

    # add WandbCallback if not already added in trainer callbacks
    reporting_to_wandb = False
    for callback in trainer.callback_handler.callbacks:
        if isinstance(callback, WandbCallback):
            reporting_to_wandb = True
            break
    if not reporting_to_wandb:
        trainer.add_callback(WandbCallback())
    trainer.args.report_to = ["wandb"]
    best_trial = {"run_id": None, "objective": None, "hyperparameters": None}
    sweep_id = kwargs.pop("sweep_id", None)
    project = kwargs.pop("project", None)
    name = kwargs.pop("name", None)
    entity = kwargs.pop("entity", None)
    metric = kwargs.pop("metric", "eval/loss")

    sweep_config = trainer.hp_space(None)
    sweep_config["metric"]["goal"] = direction
    sweep_config["metric"]["name"] = metric
    if name:
        sweep_config["name"] = name

    def _objective():
        run = wandb.run if wandb.run else wandb.init()
        trainer.state.trial_name = run.name
        run.config.update({"assignments": {}, "metric": metric})
        config = wandb.config

        trainer.objective = None

        trainer.train(resume_from_checkpoint=None, trial=vars(config)["_items"])
        # If there hasn't been any evaluation during the training loop.
        if getattr(trainer, "objective", None) is None:
            metrics = trainer.evaluate()
            trainer.objective = trainer.compute_objective(metrics)
            format_metrics = rewrite_logs(metrics)
            if metric not in format_metrics:
                logger.warning(
                    f"Provided metric {metric} not found. This might result in unexpected sweeps charts. The available"
                    f" metrics are {format_metrics.keys()}"
                )
        best_score = False
        if best_trial["run_id"] is not None:
            if direction == "minimize":
                best_score = trainer.objective < best_trial["objective"]
            elif direction == "maximize":
                best_score = trainer.objective > best_trial["objective"]

        if best_score or best_trial["run_id"] is None:
            best_trial["run_id"] = run.id
            best_trial["objective"] = trainer.objective
            best_trial["hyperparameters"] = dict(config)

        return trainer.objective

    sweep_id = wandb.sweep(sweep_config, project=project, entity=entity) if not sweep_id else sweep_id
    logger.info(f"wandb sweep id - {sweep_id}")
    wandb.agent(sweep_id, function=_objective, count=n_trials)

    return BestRun(best_trial["run_id"], best_trial["objective"], best_trial["hyperparameters"])


def get_available_reporting_integrations():
    integrations = []
    if is_azureml_available() and not is_mlflow_available():
        integrations.append("azure_ml")
    if is_comet_available():
        integrations.append("comet_ml")
    if is_dagshub_available():
        integrations.append("dagshub")
    if is_dvclive_available():
        integrations.append("dvclive")
    if is_mlflow_available():
        integrations.append("mlflow")
    if is_neptune_available():
        integrations.append("neptune")
    if is_tensorboard_available():
        integrations.append("tensorboard")
    if is_wandb_available():
        integrations.append("wandb")
    if is_codecarbon_available():
        integrations.append("codecarbon")
    if is_clearml_available():
        integrations.append("clearml")
    return integrations


def rewrite_logs(d):
    new_d = {}
    eval_prefix = "eval_"
    eval_prefix_len = len(eval_prefix)
    test_prefix = "test_"
    test_prefix_len = len(test_prefix)
    for k, v in d.items():
        if k.startswith(eval_prefix):
            new_d["eval/" + k[eval_prefix_len:]] = v
        elif k.startswith(test_prefix):
            new_d["test/" + k[test_prefix_len:]] = v
        else:
            new_d["train/" + k] = v
    return new_d


class TensorBoardCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that sends the logs to [TensorBoard](https://www.tensorflow.org/tensorboard).

    Args:
        tb_writer (`SummaryWriter`, *optional*):
            The writer to use. Will instantiate one if not set.
    """

    def __init__(self, tb_writer=None):
        has_tensorboard = is_tensorboard_available()
        if not has_tensorboard:
            raise RuntimeError(
                "TensorBoardCallback requires tensorboard to be installed. Either update your PyTorch version or"
                " install tensorboardX."
            )
        if has_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter  # noqa: F401

                self._SummaryWriter = SummaryWriter
            except ImportError:
                try:
                    from tensorboardX import SummaryWriter

                    self._SummaryWriter = SummaryWriter
                except ImportError:
                    self._SummaryWriter = None
        else:
            self._SummaryWriter = None
        self.tb_writer = tb_writer

    def _init_summary_writer(self, args, log_dir=None):
        log_dir = log_dir or args.logging_dir
        if self._SummaryWriter is not None:
            self.tb_writer = self._SummaryWriter(log_dir=log_dir)

    def on_train_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return

        log_dir = None

        if state.is_hyper_param_search:
            trial_name = state.trial_name
            if trial_name is not None:
                log_dir = os.path.join(args.logging_dir, trial_name)

        if self.tb_writer is None:
            self._init_summary_writer(args, log_dir)

        if self.tb_writer is not None:
            self.tb_writer.add_text("args", args.to_json_string())
            if "model" in kwargs:
                model = kwargs["model"]
                if hasattr(model, "config") and model.config is not None:
                    model_config_json = model.config.to_json_string()
                    self.tb_writer.add_text("model_config", model_config_json)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero:
            return

        if self.tb_writer is None:
            self._init_summary_writer(args)

        if self.tb_writer is not None:
            logs = rewrite_logs(logs)
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    self.tb_writer.add_scalar(k, v, state.global_step)
                elif isinstance(v, str):
                    self.tb_writer.add_text(k, v, state.global_step)
                else:
                    logger.warning(
                        "Trainer is attempting to log a value of "
                        f'"{v}" of type {type(v)} for key "{k}" as a scalar. '
                        "This invocation of Tensorboard's writer.add_scalar() "
                        "is incorrect so we dropped this attribute."
                    )
            self.tb_writer.flush()

    def on_train_end(self, args, state, control, **kwargs):
        if self.tb_writer:
            self.tb_writer.close()
            self.tb_writer = None


def save_model_architecture_to_file(model: Any, output_dir: str):
    with open(f"{output_dir}/model_architecture.txt", "w+") as f:
        if isinstance(model, PreTrainedModel):
            print(model, file=f)
        elif is_tf_available() and isinstance(model, TFPreTrainedModel):

            def print_to_file(s):
                print(s, file=f)

            model.summary(print_fn=print_to_file)
        elif is_torch_available() and (
            isinstance(model, (torch.nn.Module, PushToHubMixin)) and hasattr(model, "base_model")
        ):
            print(model, file=f)


class WandbLogModel(str, Enum):
    """Enum of possible log model values in W&B."""

    CHECKPOINT = "checkpoint"
    END = "end"
    FALSE = "false"

    @property
    def is_enabled(self) -> bool:
        """Check if the value corresponds to a state where the `WANDB_LOG_MODEL` setting is enabled."""
        return self in (WandbLogModel.CHECKPOINT, WandbLogModel.END)

    @classmethod
    def _missing_(cls, value: Any) -> "WandbLogModel":
        if not isinstance(value, str):
            raise ValueError(f"Expecting to have a string `WANDB_LOG_MODEL` setting, but got {type(value)}")
        if value.upper() in ENV_VARS_TRUE_VALUES:
            raise DeprecationWarning(
                f"Setting `WANDB_LOG_MODEL` as {os.getenv('WANDB_LOG_MODEL')} is deprecated and will be removed in "
                "version 5 of transformers. Use one of `'end'` or `'checkpoint'` instead."
            )
            logger.info(f"Setting `WANDB_LOG_MODEL` from {os.getenv('WANDB_LOG_MODEL')} to `end` instead")
            return WandbLogModel.END
        logger.warning(
            f"Received unrecognized `WANDB_LOG_MODEL` setting value={value}; so disabling `WANDB_LOG_MODEL`"
        )
        return WandbLogModel.FALSE


class WandbCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that logs metrics, media, model checkpoints to [Weight and Biases](https://www.wandb.com/).
    """

    def __init__(self):
        has_wandb = is_wandb_available()
        if not has_wandb:
            raise RuntimeError("WandbCallback requires wandb to be installed. Run `pip install wandb`.")
        if has_wandb:
            import wandb

            self._wandb = wandb
        self._initialized = False
        self._log_model = WandbLogModel(os.getenv("WANDB_LOG_MODEL", "false"))

    def setup(self, args, state, model, **kwargs):
        """
        Setup the optional Weights & Biases (*wandb*) integration.

        One can subclass and override this method to customize the setup if needed. Find more information
        [here](https://docs.wandb.ai/guides/integrations/huggingface). You can also override the following environment
        variables:

        Environment:
        - **WANDB_LOG_MODEL** (`str`, *optional*, defaults to `"false"`):
            Whether to log model and checkpoints during training. Can be `"end"`, `"checkpoint"` or `"false"`. If set
            to `"end"`, the model will be uploaded at the end of training. If set to `"checkpoint"`, the checkpoint
            will be uploaded every `args.save_steps` . If set to `"false"`, the model will not be uploaded. Use along
            with [`~transformers.TrainingArguments.load_best_model_at_end`] to upload best model.

            <Deprecated version="5.0">

            Setting `WANDB_LOG_MODEL` as `bool` will be deprecated in version 5 of 🤗 Transformers.

            </Deprecated>
        - **WANDB_WATCH** (`str`, *optional* defaults to `"false"`):
            Can be `"gradients"`, `"all"`, `"parameters"`, or `"false"`. Set to `"all"` to log gradients and
            parameters.
        - **WANDB_PROJECT** (`str`, *optional*, defaults to `"huggingface"`):
            Set this to a custom string to store results in a different project.
        - **WANDB_DISABLED** (`bool`, *optional*, defaults to `False`):
            Whether to disable wandb entirely. Set `WANDB_DISABLED=true` to disable.
        """
        if self._wandb is None:
            return
        self._initialized = True

        # prepare to handle potential configuration issues during setup
        from wandb.sdk.lib.config_util import ConfigError as WandbConfigError

        if state.is_world_process_zero:
            logger.info(
                'Automatic Weights & Biases logging enabled, to disable set os.environ["WANDB_DISABLED"] = "true"'
            )
            combined_dict = {**args.to_dict()}

            if hasattr(model, "config") and model.config is not None:
                model_config = model.config if isinstance(model.config, dict) else model.config.to_dict()
                combined_dict = {**model_config, **combined_dict}
            if hasattr(model, "peft_config") and model.peft_config is not None:
                peft_config = model.peft_config
                combined_dict = {**{"peft_config": peft_config}, **combined_dict}
            trial_name = state.trial_name
            init_args = {}
            if trial_name is not None:
                init_args["name"] = trial_name
                init_args["group"] = args.run_name
            elif args.run_name is not None:
                init_args["name"] = args.run_name
                if args.run_name == args.output_dir:
                    self._wandb.termwarn(
                        "The `run_name` is currently set to the same value as `TrainingArguments.output_dir`. If this was "
                        "not intended, please specify a different run name by setting the `TrainingArguments.run_name` parameter.",
                        repeat=False,
                    )

            if self._wandb.run is None:
                self._wandb.init(
                    project=os.getenv("WANDB_PROJECT", "huggingface"),
                    **init_args,
                )
            # add config parameters (run may have been created manually)
            self._wandb.config.update(combined_dict, allow_val_change=True)

            # define default x-axis (for latest wandb versions)
            if getattr(self._wandb, "define_metric", None):
                self._wandb.define_metric("train/global_step")
                self._wandb.define_metric("*", step_metric="train/global_step", step_sync=True)

            # keep track of model topology and gradients, unsupported on TPU
            _watch_model = os.getenv("WANDB_WATCH", "false")
            if not is_torch_xla_available() and _watch_model in ("all", "parameters", "gradients"):
                self._wandb.watch(model, log=_watch_model, log_freq=max(100, state.logging_steps))
            self._wandb.run._label(code="transformers_trainer")

            # add number of model parameters to wandb config
            try:
                self._wandb.config["model/num_parameters"] = model.num_parameters()
            except AttributeError:
                logger.info(
                    "Could not log the number of model parameters in Weights & Biases due to an AttributeError."
                )
            except WandbConfigError:
                logger.warning(
                    "A ConfigError was raised whilst setting the number of model parameters in Weights & Biases config."
                )

            # log the initial model architecture to an artifact
            if self._log_model.is_enabled:
                with tempfile.TemporaryDirectory() as temp_dir:
                    model_name = (
                        f"model-{self._wandb.run.id}"
                        if (args.run_name is None or args.run_name == args.output_dir)
                        else f"model-{self._wandb.run.name}"
                    )
                    model_artifact = self._wandb.Artifact(
                        name=model_name,
                        type="model",
                        metadata={
                            "model_config": model.config.to_dict() if hasattr(model, "config") else None,
                            "num_parameters": self._wandb.config.get("model/num_parameters"),
                            "initial_model": True,
                        },
                    )
                    # add the architecture to a separate text file
                    save_model_architecture_to_file(model, temp_dir)

                    for f in Path(temp_dir).glob("*"):
                        if f.is_file():
                            with model_artifact.new_file(f.name, mode="wb") as fa:
                                fa.write(f.read_bytes())
                    self._wandb.run.log_artifact(model_artifact, aliases=["base_model"])

                    badge_markdown = (
                        f'[<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge'
                        f'-28.svg" alt="Visualize in Weights & Biases" width="20'
                        f'0" height="32"/>]({self._wandb.run.get_url()})'
                    )

                    modelcard.AUTOGENERATED_TRAINER_COMMENT += f"\n{badge_markdown}"

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if self._wandb is None:
            return
        hp_search = state.is_hyper_param_search
        if hp_search:
            self._wandb.finish()
            self._initialized = False
            args.run_name = None
        if not self._initialized:
            self.setup(args, state, model, **kwargs)

    def on_train_end(self, args, state, control, model=None, processing_class=None, **kwargs):
        if self._wandb is None:
            return
        if self._log_model.is_enabled and self._initialized and state.is_world_process_zero:
            from ..trainer import Trainer

            fake_trainer = Trainer(args=args, model=model, processing_class=processing_class, eval_dataset=["fake"])
            with tempfile.TemporaryDirectory() as temp_dir:
                fake_trainer.save_model(temp_dir)
                metadata = (
                    {
                        k: v
                        for k, v in dict(self._wandb.summary).items()
                        if isinstance(v, numbers.Number) and not k.startswith("_")
                    }
                    if not args.load_best_model_at_end
                    else {
                        f"eval/{args.metric_for_best_model}": state.best_metric,
                        "train/total_floss": state.total_flos,
                        "model/num_parameters": self._wandb.config.get("model/num_parameters"),
                    }
                )
                metadata["final_model"] = True
                logger.info("Logging model artifacts. ...")
                model_name = (
                    f"model-{self._wandb.run.id}"
                    if (args.run_name is None or args.run_name == args.output_dir)
                    else f"model-{self._wandb.run.name}"
                )
                # add the model architecture to a separate text file
                save_model_architecture_to_file(model, temp_dir)

                artifact = self._wandb.Artifact(name=model_name, type="model", metadata=metadata)
                for f in Path(temp_dir).glob("*"):
                    if f.is_file():
                        with artifact.new_file(f.name, mode="wb") as fa:
                            fa.write(f.read_bytes())
                self._wandb.run.log_artifact(artifact, aliases=["final_model"])

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        single_value_scalars = [
            "train_runtime",
            "train_samples_per_second",
            "train_steps_per_second",
            "train_loss",
            "total_flos",
        ]

        if self._wandb is None:
            return
        if not self._initialized:
            self.setup(args, state, model)
        if state.is_world_process_zero:
            for k, v in logs.items():
                if k in single_value_scalars:
                    self._wandb.run.summary[k] = v
            non_scalar_logs = {k: v for k, v in logs.items() if k not in single_value_scalars}
            non_scalar_logs = rewrite_logs(non_scalar_logs)
            self._wandb.log({**non_scalar_logs, "train/global_step": state.global_step})

    def on_save(self, args, state, control, **kwargs):
        if self._log_model == WandbLogModel.CHECKPOINT and self._initialized and state.is_world_process_zero:
            checkpoint_metadata = {
                k: v
                for k, v in dict(self._wandb.summary).items()
                if isinstance(v, numbers.Number) and not k.startswith("_")
            }
            checkpoint_metadata["model/num_parameters"] = self._wandb.config.get("model/num_parameters")

            ckpt_dir = f"checkpoint-{state.global_step}"
            artifact_path = os.path.join(args.output_dir, ckpt_dir)
            logger.info(f"Logging checkpoint artifacts in {ckpt_dir}. ...")
            checkpoint_name = (
                f"model-{self._wandb.run.id}"
                if (args.run_name is None or args.run_name == args.output_dir)
                else f"model-{self._wandb.run.name}"
            )
            artifact = self._wandb.Artifact(name=checkpoint_name, type="model", metadata=checkpoint_metadata)
            artifact.add_dir(artifact_path)
            self._wandb.log_artifact(
                artifact, aliases=[f"epoch_{round(state.epoch, 2)}", f"checkpoint_global_step_{state.global_step}"]
            )

    def on_predict(self, args, state, control, metrics, **kwargs):
        if self._wandb is None:
            return
        if not self._initialized:
            self.setup(args, state, **kwargs)
        if state.is_world_process_zero:
            metrics = rewrite_logs(metrics)
            self._wandb.log(metrics)


class CometCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that sends the logs to [Comet ML](https://www.comet.com/site/).
    """

    def __init__(self):
        if _is_comet_installed is False or _is_comet_recent_enough is False:
            raise RuntimeError(
                f"CometCallback requires comet-ml>={_MIN_COMET_VERSION} to be installed. Run `pip install comet-ml>={_MIN_COMET_VERSION}`."
            )
        self._initialized = False
        self._log_assets = False
        self._experiment = None

    def setup(self, args, state, model):
        """
        Setup the optional Comet integration.

        Environment:
        - **COMET_MODE** (`str`, *optional*, default to `get_or_create`):
            Control whether to create and log to a new Comet experiment or append to an existing experiment.
            It accepts the following values:
                * `get_or_create`: Decides automatically depending if
                  `COMET_EXPERIMENT_KEY` is set and whether an Experiment
                  with that key already exists or not.
                * `create`: Always create a new Comet Experiment.
                * `get`: Always try to append to an Existing Comet Experiment.
                  Requires `COMET_EXPERIMENT_KEY` to be set.
                * `ONLINE`: **deprecated**, used to create an online
                  Experiment. Use `COMET_START_ONLINE=1` instead.
                * `OFFLINE`: **deprecated**, used to created an offline
                  Experiment. Use `COMET_START_ONLINE=0` instead.
                * `DISABLED`: **deprecated**, used to disable Comet logging.
                  Use the `--report_to` flag to control the integrations used
                  for logging result instead.
        - **COMET_PROJECT_NAME** (`str`, *optional*):
            Comet project name for experiments.
        - **COMET_LOG_ASSETS** (`str`, *optional*, defaults to `TRUE`):
            Whether or not to log training assets (tf event logs, checkpoints, etc), to Comet. Can be `TRUE`, or
            `FALSE`.

        For a number of configurable items in the environment, see
        [here](https://www.comet.com/docs/v2/guides/experiment-management/configure-sdk/#explore-comet-configuration-options).
        """
        self._initialized = True
        log_assets = os.getenv("COMET_LOG_ASSETS", "FALSE").upper()
        if log_assets in {"TRUE", "1"}:
            self._log_assets = True
        if state.is_world_process_zero:
            comet_old_mode = os.getenv("COMET_MODE")

            mode = None
            online = None

            if comet_old_mode is not None:
                comet_old_mode = comet_old_mode.lower()

                if comet_old_mode == "online":
                    online = True
                elif comet_old_mode == "offline":
                    online = False
                elif comet_old_mode in ("get", "get_or_create", "create"):
                    mode = comet_old_mode
                elif comet_old_mode:
                    logger.warning("Invalid COMET_MODE env value %r, Comet logging is disabled", comet_old_mode)
                    return

            # For HPO, we always create a new experiment for each trial
            if state.is_hyper_param_search:
                if mode is not None:
                    logger.warning(
                        "Hyperparameter Search is enabled, forcing the creation of new experimetns, COMET_MODE value %r  is ignored",
                        comet_old_mode,
                    )
                mode = "create"

            import comet_ml

            # Do not use the default run_name as the experiment name
            if args.run_name is not None and args.run_name != args.output_dir:
                experiment_config = comet_ml.ExperimentConfig(name=args.run_name)
            else:
                experiment_config = comet_ml.ExperimentConfig()

            self._experiment = comet_ml.start(online=online, mode=mode, experiment_config=experiment_config)
            self._experiment.__internal_api__set_model_graph__(model, framework="transformers")

            params = {"args": args.to_dict()}

            if hasattr(model, "config") and model.config is not None:
                model_config = model.config.to_dict()
                params["config"] = model_config
            if hasattr(model, "peft_config") and model.peft_config is not None:
                peft_config = model.peft_config
                params["peft_config"] = peft_config

            self._experiment.__internal_api__log_parameters__(
                params, framework="transformers", source="manual", flatten_nested=True
            )

            if state.is_hyper_param_search:
                optimization_id = getattr(state, "trial_name", None)
                optimization_params = getattr(state, "trial_params", None)

                self._experiment.log_optimization(optimization_id=optimization_id, parameters=optimization_params)

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if not self._initialized:
            self.setup(args, state, model)

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if not self._initialized:
            self.setup(args, state, model)
        if state.is_world_process_zero:
            if self._experiment is not None:
                rewritten_logs = rewrite_logs(logs)
                self._experiment.__internal_api__log_metrics__(
                    rewritten_logs, step=state.global_step, epoch=state.epoch, framework="transformers"
                )

    def on_train_end(self, args, state, control, **kwargs):
        if self._initialized and state.is_world_process_zero:
            if self._experiment is not None:
                if self._log_assets is True:
                    logger.info("Logging checkpoints. This may take time.")
                    self._experiment.log_asset_folder(
                        args.output_dir, recursive=True, log_file_name=True, step=state.global_step
                    )

            # We create one experiment per trial in HPO mode
            if state.is_hyper_param_search:
                self._experiment.clean()
                self._initialized = False

    def on_predict(self, args, state, control, metrics, **kwargs):
        if not self._initialized:
            self.setup(args, state, model=None)
        if state.is_world_process_zero and self._experiment is not None:
            rewritten_metrics = rewrite_logs(metrics)
            self._experiment.__internal_api__log_metrics__(
                rewritten_metrics, step=state.global_step, epoch=state.epoch, framework="transformers"
            )


class AzureMLCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that sends the logs to [AzureML](https://pypi.org/project/azureml-sdk/).
    """

    def __init__(self, azureml_run=None):
        if not is_azureml_available():
            raise RuntimeError("AzureMLCallback requires azureml to be installed. Run `pip install azureml-sdk`.")
        self.azureml_run = azureml_run

    def on_init_end(self, args, state, control, **kwargs):
        from azureml.core.run import Run

        if self.azureml_run is None and state.is_world_process_zero:
            self.azureml_run = Run.get_context()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.azureml_run and state.is_world_process_zero:
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    self.azureml_run.log(k, v, description=k)


class MLflowCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that sends the logs to [MLflow](https://www.mlflow.org/). Can be disabled by setting
    environment variable `DISABLE_MLFLOW_INTEGRATION = TRUE`.
    """

    def __init__(self):
        if not is_mlflow_available():
            raise RuntimeError("MLflowCallback requires mlflow to be installed. Run `pip install mlflow`.")
        import mlflow

        self._MAX_PARAM_VAL_LENGTH = mlflow.utils.validation.MAX_PARAM_VAL_LENGTH
        self._MAX_PARAMS_TAGS_PER_BATCH = mlflow.utils.validation.MAX_PARAMS_TAGS_PER_BATCH

        self._initialized = False
        self._auto_end_run = False
        self._log_artifacts = False
        self._ml_flow = mlflow

    def setup(self, args, state, model):
        """
        Setup the optional MLflow integration.

        Environment:
        - **HF_MLFLOW_LOG_ARTIFACTS** (`str`, *optional*):
            Whether to use MLflow `.log_artifact()` facility to log artifacts. This only makes sense if logging to a
            remote server, e.g. s3 or GCS. If set to `True` or *1*, will copy each saved checkpoint on each save in
            [`TrainingArguments`]'s `output_dir` to the local or remote artifact storage. Using it without a remote
            storage will just copy the files to your artifact location.
        - **MLFLOW_TRACKING_URI** (`str`, *optional*):
            Whether to store runs at a specific path or remote server. Unset by default, which skips setting the
            tracking URI entirely.
        - **MLFLOW_EXPERIMENT_NAME** (`str`, *optional*, defaults to `None`):
            Whether to use an MLflow experiment_name under which to launch the run. Default to `None` which will point
            to the `Default` experiment in MLflow. Otherwise, it is a case sensitive name of the experiment to be
            activated. If an experiment with this name does not exist, a new experiment with this name is created.
        - **MLFLOW_TAGS** (`str`, *optional*):
            A string dump of a dictionary of key/value pair to be added to the MLflow run as tags. Example:
            `os.environ['MLFLOW_TAGS']='{"release.candidate": "RC1", "release.version": "2.2.0"}'`.
        - **MLFLOW_NESTED_RUN** (`str`, *optional*):
            Whether to use MLflow nested runs. If set to `True` or *1*, will create a nested run inside the current
            run.
        - **MLFLOW_RUN_ID** (`str`, *optional*):
            Allow to reattach to an existing run which can be usefull when resuming training from a checkpoint. When
            `MLFLOW_RUN_ID` environment variable is set, `start_run` attempts to resume a run with the specified run ID
            and other parameters are ignored.
        - **MLFLOW_FLATTEN_PARAMS** (`str`, *optional*, defaults to `False`):
            Whether to flatten the parameters dictionary before logging.
        - **MLFLOW_MAX_LOG_PARAMS** (`int`, *optional*):
            Set the maximum number of parameters to log in the run.
        """
        self._log_artifacts = os.getenv("HF_MLFLOW_LOG_ARTIFACTS", "FALSE").upper() in ENV_VARS_TRUE_VALUES
        self._nested_run = os.getenv("MLFLOW_NESTED_RUN", "FALSE").upper() in ENV_VARS_TRUE_VALUES
        self._tracking_uri = os.getenv("MLFLOW_TRACKING_URI", None)
        self._experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", None)
        self._flatten_params = os.getenv("MLFLOW_FLATTEN_PARAMS", "FALSE").upper() in ENV_VARS_TRUE_VALUES
        self._run_id = os.getenv("MLFLOW_RUN_ID", None)
        self._max_log_params = os.getenv("MLFLOW_MAX_LOG_PARAMS", None)

        # "synchronous" flag is only available with mlflow version >= 2.8.0
        # https://github.com/mlflow/mlflow/pull/9705
        # https://github.com/mlflow/mlflow/releases/tag/v2.8.0
        self._async_log = packaging.version.parse(self._ml_flow.__version__) >= packaging.version.parse("2.8.0")

        logger.debug(
            f"MLflow experiment_name={self._experiment_name}, run_name={args.run_name}, nested={self._nested_run},"
            f" tracking_uri={self._tracking_uri}"
        )
        if state.is_world_process_zero:
            if not self._ml_flow.is_tracking_uri_set():
                if self._tracking_uri:
                    self._ml_flow.set_tracking_uri(self._tracking_uri)
                    logger.debug(f"MLflow tracking URI is set to {self._tracking_uri}")
                else:
                    logger.debug(
                        "Environment variable `MLFLOW_TRACKING_URI` is not provided and therefore will not be"
                        " explicitly set."
                    )
            else:
                logger.debug(f"MLflow tracking URI is set to {self._ml_flow.get_tracking_uri()}")

            if self._ml_flow.active_run() is None or self._nested_run or self._run_id:
                if self._experiment_name:
                    # Use of set_experiment() ensure that Experiment is created if not exists
                    self._ml_flow.set_experiment(self._experiment_name)
                self._ml_flow.start_run(run_name=args.run_name, nested=self._nested_run)
                logger.debug(f"MLflow run started with run_id={self._ml_flow.active_run().info.run_id}")
                self._auto_end_run = True
            combined_dict = args.to_dict()
            if hasattr(model, "config") and model.config is not None:
                model_config = model.config.to_dict()
                combined_dict = {**model_config, **combined_dict}
            combined_dict = flatten_dict(combined_dict) if self._flatten_params else combined_dict
            # remove params that are too long for MLflow
            for name, value in list(combined_dict.items()):
                # internally, all values are converted to str in MLflow
                if len(str(value)) > self._MAX_PARAM_VAL_LENGTH:
                    logger.warning(
                        f'Trainer is attempting to log a value of "{value}" for key "{name}" as a parameter. MLflow\'s'
                        " log_param() only accepts values no longer than 250 characters so we dropped this attribute."
                        " You can use `MLFLOW_FLATTEN_PARAMS` environment variable to flatten the parameters and"
                        " avoid this message."
                    )
                    del combined_dict[name]
            # MLflow cannot log more than 100 values in one go, so we have to split it
            combined_dict_items = list(combined_dict.items())
            if self._max_log_params and self._max_log_params.isdigit():
                max_log_params = int(self._max_log_params)
                if max_log_params < len(combined_dict_items):
                    logger.debug(
                        f"Reducing the number of parameters to log from {len(combined_dict_items)} to {max_log_params}."
                    )
                    combined_dict_items = combined_dict_items[:max_log_params]
            for i in range(0, len(combined_dict_items), self._MAX_PARAMS_TAGS_PER_BATCH):
                if self._async_log:
                    self._ml_flow.log_params(
                        dict(combined_dict_items[i : i + self._MAX_PARAMS_TAGS_PER_BATCH]), synchronous=False
                    )
                else:
                    self._ml_flow.log_params(dict(combined_dict_items[i : i + self._MAX_PARAMS_TAGS_PER_BATCH]))
            mlflow_tags = os.getenv("MLFLOW_TAGS", None)
            if mlflow_tags:
                mlflow_tags = json.loads(mlflow_tags)
                self._ml_flow.set_tags(mlflow_tags)
        self._initialized = True

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if not self._initialized:
            self.setup(args, state, model)

    def on_log(self, args, state, control, logs, model=None, **kwargs):
        if not self._initialized:
            self.setup(args, state, model)
        if state.is_world_process_zero:
            metrics = {}
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    metrics[k] = v
                elif isinstance(v, torch.Tensor) and v.numel() == 1:
                    metrics[k] = v.item()
                else:
                    logger.warning(
                        f'Trainer is attempting to log a value of "{v}" of type {type(v)} for key "{k}" as a metric. '
                        "MLflow's log_metric() only accepts float and int types so we dropped this attribute."
                    )

            if self._async_log:
                self._ml_flow.log_metrics(metrics=metrics, step=state.global_step, synchronous=False)
            else:
                self._ml_flow.log_metrics(metrics=metrics, step=state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        if self._initialized and state.is_world_process_zero:
            if self._auto_end_run and self._ml_flow.active_run():
                self._ml_flow.end_run()

    def on_save(self, args, state, control, **kwargs):
        if self._initialized and state.is_world_process_zero and self._log_artifacts:
            ckpt_dir = f"checkpoint-{state.global_step}"
            artifact_path = os.path.join(args.output_dir, ckpt_dir)
            logger.info(f"Logging checkpoint artifacts in {ckpt_dir}. This may take time.")
            self._ml_flow.pyfunc.log_model(
                ckpt_dir,
                artifacts={"model_path": artifact_path},
                python_model=self._ml_flow.pyfunc.PythonModel(),
            )

    def __del__(self):
        # if the previous run is not terminated correctly, the fluent API will
        # not let you start a new run before the previous one is killed
        if (
            self._auto_end_run
            and callable(getattr(self._ml_flow, "active_run", None))
            and self._ml_flow.active_run() is not None
        ):
            self._ml_flow.end_run()


class DagsHubCallback(MLflowCallback):
    """
    A [`TrainerCallback`] that logs to [DagsHub](https://dagshub.com/). Extends [`MLflowCallback`]
    """

    def __init__(self):
        super().__init__()
        if not is_dagshub_available():
            raise ImportError("DagsHubCallback requires dagshub to be installed. Run `pip install dagshub`.")

        from dagshub.upload import Repo

        self.Repo = Repo

    def setup(self, *args, **kwargs):
        """
        Setup the DagsHub's Logging integration.

        Environment:
        - **HF_DAGSHUB_LOG_ARTIFACTS** (`str`, *optional*):
                Whether to save the data and model artifacts for the experiment. Default to `False`.
        """

        self.log_artifacts = os.getenv("HF_DAGSHUB_LOG_ARTIFACTS", "FALSE").upper() in ENV_VARS_TRUE_VALUES
        self.name = os.getenv("HF_DAGSHUB_MODEL_NAME") or "main"
        self.remote = os.getenv("MLFLOW_TRACKING_URI")
        self.repo = self.Repo(
            owner=self.remote.split(os.sep)[-2],
            name=self.remote.split(os.sep)[-1].split(".")[0],
            branch=os.getenv("BRANCH") or "main",
        )
        self.path = Path("artifacts")

        if self.remote is None:
            raise RuntimeError(
                "DagsHubCallback requires the `MLFLOW_TRACKING_URI` environment variable to be set. Did you run"
                " `dagshub.init()`?"
            )

        super().setup(*args, **kwargs)

    def on_train_end(self, args, state, control, **kwargs):
        if self.log_artifacts:
            if getattr(self, "train_dataloader", None):
                torch.save(self.train_dataloader.dataset, os.path.join(args.output_dir, "dataset.pt"))

            self.repo.directory(str(self.path)).add_dir(args.output_dir)


class NeptuneMissingConfiguration(Exception):
    def __init__(self):
        super().__init__(
            """
        ------ Unsupported ---- We were not able to create new runs. You provided a custom Neptune run to
        `NeptuneCallback` with the `run` argument. For the integration to work fully, provide your `api_token` and
        `project` by saving them as environment variables or passing them to the callback.
        """
        )


class NeptuneCallback(TrainerCallback):
    """TrainerCallback that sends the logs to [Neptune](https://app.neptune.ai).

    Args:
        api_token (`str`, *optional*): Neptune API token obtained upon registration.
            You can leave this argument out if you have saved your token to the `NEPTUNE_API_TOKEN` environment
            variable (strongly recommended). See full setup instructions in the
            [docs](https://docs.neptune.ai/setup/installation).
        project (`str`, *optional*): Name of an existing Neptune project, in the form "workspace-name/project-name".
            You can find and copy the name in Neptune from the project settings -> Properties. If None (default), the
            value of the `NEPTUNE_PROJECT` environment variable is used.
        name (`str`, *optional*): Custom name for the run.
        base_namespace (`str`, *optional*, defaults to "finetuning"): In the Neptune run, the root namespace
            that will contain all of the metadata logged by the callback.
        log_parameters (`bool`, *optional*, defaults to `True`):
            If True, logs all Trainer arguments and model parameters provided by the Trainer.
        log_checkpoints (`str`, *optional*): If "same", uploads checkpoints whenever they are saved by the Trainer.
            If "last", uploads only the most recently saved checkpoint. If "best", uploads the best checkpoint (among
            the ones saved by the Trainer). If `None`, does not upload checkpoints.
        run (`Run`, *optional*): Pass a Neptune run object if you want to continue logging to an existing run.
            Read more about resuming runs in the [docs](https://docs.neptune.ai/logging/to_existing_object).
        **neptune_run_kwargs (*optional*):
            Additional keyword arguments to be passed directly to the
            [`neptune.init_run()`](https://docs.neptune.ai/api/neptune#init_run) function when a new run is created.

    For instructions and examples, see the [Transformers integration
    guide](https://docs.neptune.ai/integrations/transformers) in the Neptune documentation.
    """

    integration_version_key = "source_code/integrations/transformers"
    model_parameters_key = "model_parameters"
    trial_name_key = "trial"
    trial_params_key = "trial_params"
    trainer_parameters_key = "trainer_parameters"
    flat_metrics = {"train/epoch"}

    def __init__(
        self,
        *,
        api_token: Optional[str] = None,
        project: Optional[str] = None,
        name: Optional[str] = None,
        base_namespace: str = "finetuning",
        run=None,
        log_parameters: bool = True,
        log_checkpoints: Optional[str] = None,
        **neptune_run_kwargs,
    ):
        if not is_neptune_available():
            raise ValueError(
                "NeptuneCallback requires the Neptune client library to be installed. "
                "To install the library, run `pip install neptune`."
            )

        try:
            from neptune import Run
            from neptune.internal.utils import verify_type
        except ImportError:
            from neptune.new.internal.utils import verify_type
            from neptune.new.metadata_containers.run import Run

        verify_type("api_token", api_token, (str, type(None)))
        verify_type("project", project, (str, type(None)))
        verify_type("name", name, (str, type(None)))
        verify_type("base_namespace", base_namespace, str)
        verify_type("run", run, (Run, type(None)))
        verify_type("log_parameters", log_parameters, bool)
        verify_type("log_checkpoints", log_checkpoints, (str, type(None)))

        self._base_namespace_path = base_namespace
        self._log_parameters = log_parameters
        self._log_checkpoints = log_checkpoints
        self._initial_run: Optional[Run] = run

        self._run = None
        self._is_monitoring_run = False
        self._run_id = None
        self._force_reset_monitoring_run = False
        self._init_run_kwargs = {"api_token": api_token, "project": project, "name": name, **neptune_run_kwargs}

        self._volatile_checkpoints_dir = None
        self._should_upload_checkpoint = self._log_checkpoints is not None
        self._recent_checkpoint_path = None

        if self._log_checkpoints in {"last", "best"}:
            self._target_checkpoints_namespace = f"checkpoints/{self._log_checkpoints}"
            self._should_clean_recently_uploaded_checkpoint = True
        else:
            self._target_checkpoints_namespace = "checkpoints"
            self._should_clean_recently_uploaded_checkpoint = False

    def _stop_run_if_exists(self):
        if self._run:
            self._run.stop()
            del self._run
            self._run = None

    def _initialize_run(self, **additional_neptune_kwargs):
        try:
            from neptune import init_run
            from neptune.exceptions import NeptuneMissingApiTokenException, NeptuneMissingProjectNameException
        except ImportError:
            from neptune.new import init_run
            from neptune.new.exceptions import NeptuneMissingApiTokenException, NeptuneMissingProjectNameException

        self._stop_run_if_exists()

        try:
            run_params = additional_neptune_kwargs.copy()
            run_params.update(self._init_run_kwargs)
            self._run = init_run(**run_params)
            self._run_id = self._run["sys/id"].fetch()
        except (NeptuneMissingProjectNameException, NeptuneMissingApiTokenException) as e:
            raise NeptuneMissingConfiguration() from e

    def _use_initial_run(self):
        self._run = self._initial_run
        self._is_monitoring_run = True
        self._run_id = self._run["sys/id"].fetch()
        self._initial_run = None

    def _ensure_run_with_monitoring(self):
        if self._initial_run is not None:
            self._use_initial_run()
        else:
            if not self._force_reset_monitoring_run and self._is_monitoring_run:
                return

            if self._run and not self._is_monitoring_run and not self._force_reset_monitoring_run:
                self._initialize_run(with_id=self._run_id)
                self._is_monitoring_run = True
            else:
                self._initialize_run()
                self._force_reset_monitoring_run = False

    def _ensure_at_least_run_without_monitoring(self):
        if self._initial_run is not None:
            self._use_initial_run()
        else:
            if not self._run:
                self._initialize_run(
                    with_id=self._run_id,
                    capture_stdout=False,
                    capture_stderr=False,
                    capture_hardware_metrics=False,
                    capture_traceback=False,
                )
                self._is_monitoring_run = False

    @property
    def run(self):
        if self._run is None:
            self._ensure_at_least_run_without_monitoring()
        return self._run

    @property
    def _metadata_namespace(self):
        return self.run[self._base_namespace_path]

    def _log_integration_version(self):
        self.run[NeptuneCallback.integration_version_key] = version

    def _log_trainer_parameters(self, args):
        self._metadata_namespace[NeptuneCallback.trainer_parameters_key] = args.to_sanitized_dict()

    def _log_model_parameters(self, model):
        from neptune.utils import stringify_unsupported

        if model and hasattr(model, "config") and model.config is not None:
            self._metadata_namespace[NeptuneCallback.model_parameters_key] = stringify_unsupported(
                model.config.to_dict()
            )

    def _log_hyper_param_search_parameters(self, state):
        if state and hasattr(state, "trial_name"):
            self._metadata_namespace[NeptuneCallback.trial_name_key] = state.trial_name

        if state and hasattr(state, "trial_params") and state.trial_params is not None:
            self._metadata_namespace[NeptuneCallback.trial_params_key] = state.trial_params

    def _log_model_checkpoint(self, source_directory: str, checkpoint: str):
        target_path = relative_path = os.path.join(source_directory, checkpoint)

        if self._volatile_checkpoints_dir is not None:
            consistent_checkpoint_path = os.path.join(self._volatile_checkpoints_dir, checkpoint)
            try:
                # Remove leading ../ from a relative path.
                cpkt_path = relative_path.replace("..", "").lstrip(os.path.sep)
                copy_path = os.path.join(consistent_checkpoint_path, cpkt_path)
                shutil.copytree(relative_path, copy_path)
                target_path = consistent_checkpoint_path
            except IOError as e:
                logger.warning(
                    "NeptuneCallback was unable to made a copy of checkpoint due to I/O exception: '{}'. "
                    "Could fail trying to upload.".format(e)
                )

        self._metadata_namespace[self._target_checkpoints_namespace].upload_files(target_path)

        if self._should_clean_recently_uploaded_checkpoint and self._recent_checkpoint_path is not None:
            self._metadata_namespace[self._target_checkpoints_namespace].delete_files(self._recent_checkpoint_path)

        self._recent_checkpoint_path = relative_path

    def on_init_end(self, args, state, control, **kwargs):
        self._volatile_checkpoints_dir = None
        if self._log_checkpoints and (args.overwrite_output_dir or args.save_total_limit is not None):
            self._volatile_checkpoints_dir = tempfile.TemporaryDirectory().name

        if self._log_checkpoints == "best" and not args.load_best_model_at_end:
            raise ValueError("To save the best model checkpoint, the load_best_model_at_end argument must be enabled.")

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if not state.is_world_process_zero:
            return

        self._ensure_run_with_monitoring()
        self._force_reset_monitoring_run = True

        self._log_integration_version()
        if self._log_parameters:
            self._log_trainer_parameters(args)
            self._log_model_parameters(model)

        if state.is_hyper_param_search:
            self._log_hyper_param_search_parameters(state)

    def on_train_end(self, args, state, control, **kwargs):
        self._stop_run_if_exists()

    def __del__(self):
        if self._volatile_checkpoints_dir is not None:
            shutil.rmtree(self._volatile_checkpoints_dir, ignore_errors=True)

        self._stop_run_if_exists()

    def on_save(self, args, state, control, **kwargs):
        if self._should_upload_checkpoint:
            self._log_model_checkpoint(args.output_dir, f"checkpoint-{state.global_step}")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if self._log_checkpoints == "best":
            best_metric_name = args.metric_for_best_model
            if not best_metric_name.startswith("eval_"):
                best_metric_name = f"eval_{best_metric_name}"

            metric_value = metrics.get(best_metric_name)

            operator = np.greater if args.greater_is_better else np.less

            self._should_upload_checkpoint = state.best_metric is None or operator(metric_value, state.best_metric)

    @classmethod
    def get_run(cls, trainer):
        for callback in trainer.callback_handler.callbacks:
            if isinstance(callback, cls):
                return callback.run

        raise Exception("The trainer doesn't have a NeptuneCallback configured.")

    def on_log(self, args, state, control, logs: Optional[Dict[str, float]] = None, **kwargs):
        if not state.is_world_process_zero:
            return

        if logs is not None:
            for name, value in rewrite_logs(logs).items():
                if isinstance(value, (int, float)):
                    if name in NeptuneCallback.flat_metrics:
                        self._metadata_namespace[name] = value
                    else:
                        self._metadata_namespace[name].log(value, step=state.global_step)


class CodeCarbonCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that tracks the CO2 emission of training.
    """

    def __init__(self):
        if not is_codecarbon_available():
            raise RuntimeError(
                "CodeCarbonCallback requires `codecarbon` to be installed. Run `pip install codecarbon`."
            )
        elif torch.version.hip:
            raise RuntimeError(
                "CodeCarbonCallback requires `codecarbon` package, which is not compatible with AMD ROCm (https://github.com/mlco2/codecarbon/pull/490). When using the Trainer, please specify the `report_to` argument (https://huggingface.co/docs/transformers/v4.39.3/en/main_classes/trainer#transformers.TrainingArguments.report_to) to disable CodeCarbonCallback."
            )

        import codecarbon

        self._codecarbon = codecarbon
        self.tracker = None

    def on_init_end(self, args, state, control, **kwargs):
        if self.tracker is None and state.is_local_process_zero:
            # CodeCarbon will automatically handle environment variables for configuration
            self.tracker = self._codecarbon.EmissionsTracker(output_dir=args.output_dir)

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if self.tracker and state.is_local_process_zero:
            self.tracker.start()

    def on_train_end(self, args, state, control, **kwargs):
        if self.tracker and state.is_local_process_zero:
            self.tracker.stop()


class ClearMLCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that sends the logs to [ClearML](https://clear.ml/).

    Environment:
    - **CLEARML_PROJECT** (`str`, *optional*, defaults to `HuggingFace Transformers`):
        ClearML project name.
    - **CLEARML_TASK** (`str`, *optional*, defaults to `Trainer`):
        ClearML task name.
    - **CLEARML_LOG_MODEL** (`bool`, *optional*, defaults to `False`):
        Whether to log models as artifacts during training.
    """

    log_suffix = ""

    _hparams_section = "Transformers"
    _model_config_section = "Model Configuration"
    _ignore_hparams_overrides = "_ignore_hparams_ui_overrides_"
    _ignoge_model_config_overrides = "_ignore_model_config_ui_overrides_"
    _model_config_description = "The configuration of model number {}."
    _model_config_description_note = (
        "Note that, when cloning this task and running it remotely,"
        " the configuration might be applied to another model instead of this one."
        " To avoid this, initialize the task externally by calling `Task.init`"
        " before the `ClearMLCallback` is instantiated."
    )
    _train_run_counter = 0
    _model_connect_counter = 0
    _task_created_in_callback = False
    _should_close_on_train_end = None

    def __init__(self):
        if is_clearml_available():
            import clearml

            self._clearml = clearml
        else:
            raise RuntimeError("ClearMLCallback requires 'clearml' to be installed. Run `pip install clearml`.")

        self._initialized = False
        self._clearml_task = None

        self._log_model = False
        self._checkpoints_saved = []

    def setup(self, args, state, model, processing_class, **kwargs):
        if self._clearml is None:
            return
        if self._initialized:
            return
        ClearMLCallback._train_run_counter += 1
        ClearMLCallback._model_connect_counter += 1
        ClearMLCallback.log_suffix = (
            "" if ClearMLCallback._train_run_counter == 1 else "_" + str(ClearMLCallback._train_run_counter)
        )
        if state.is_world_process_zero:
            logger.info("Automatic ClearML logging enabled.")
            if self._clearml_task is None:
                if ClearMLCallback._should_close_on_train_end is None:
                    if not self._clearml.Task.running_locally() or self._clearml.Task.current_task():
                        ClearMLCallback._should_close_on_train_end = False
                    else:
                        ClearMLCallback._should_close_on_train_end = True

                # This might happen when running inside of a pipeline, where the task is already initialized
                # from outside of Hugging Face
                if self._clearml.Task.running_locally() and self._clearml.Task.current_task():
                    self._clearml_task = self._clearml.Task.current_task()
                    self._log_model = os.getenv(
                        "CLEARML_LOG_MODEL",
                        "FALSE" if not ClearMLCallback._task_created_in_callback else "TRUE",
                    ).upper() in ENV_VARS_TRUE_VALUES.union({"TRUE"})
                    logger.info("External ClearML Task has been connected.")
                else:
                    self._clearml_task = self._clearml.Task.init(
                        project_name=os.getenv("CLEARML_PROJECT", "HuggingFace Transformers"),
                        task_name=os.getenv("CLEARML_TASK", "Trainer"),
                        auto_connect_frameworks={"tensorboard": False, "pytorch": False},
                        output_uri=True,
                    )
                    self._log_model = os.getenv("CLEARML_LOG_MODEL", "TRUE").upper() in ENV_VARS_TRUE_VALUES.union(
                        {"TRUE"}
                    )
                    ClearMLCallback._task_created_in_callback = True
                    logger.info("ClearML Task has been initialized.")
                self._initialized = True

            suffixed_hparams_section = ClearMLCallback._hparams_section + ClearMLCallback.log_suffix
            ignore_hparams_config_section = suffixed_hparams_section + "/" + ClearMLCallback._ignore_hparams_overrides
            if self._clearml.Task.running_locally():
                self._copy_training_args_as_hparams(args, suffixed_hparams_section)
                self._clearml_task.set_parameter(
                    name=ignore_hparams_config_section,
                    value=True,
                    value_type=bool,
                    description=(
                        "If True, ignore Transformers hyperparameters overrides done in the UI/backend "
                        + "when running remotely. Otherwise, the overrides will be applied when running remotely"
                    ),
                )
            elif not self._clearml_task.get_parameter(ignore_hparams_config_section, default=True, cast=True):
                self._clearml_task.connect(args, suffixed_hparams_section)
            else:
                self._copy_training_args_as_hparams(
                    args, ClearMLCallback._hparams_section + ClearMLCallback.log_suffix
                )

            if getattr(model, "config", None) is not None:
                ignore_model_config_section = (
                    suffixed_hparams_section + "/" + ClearMLCallback._ignoge_model_config_overrides
                )
                configuration_object_description = ClearMLCallback._model_config_description.format(
                    ClearMLCallback._model_connect_counter
                )
                if ClearMLCallback._model_connect_counter != ClearMLCallback._train_run_counter:
                    configuration_object_description += " " + ClearMLCallback._model_config_description_note
                if self._clearml.Task.running_locally():
                    self._clearml_task.set_parameter(
                        name=ignore_model_config_section,
                        value=True,
                        value_type=bool,
                        description=(
                            "If True, ignore Transformers model configuration overrides done in the UI/backend "
                            + "when running remotely. Otherwise, the overrides will be applied when running remotely"
                        ),
                    )
                    self._clearml_task.set_configuration_object(
                        name=ClearMLCallback._model_config_section + ClearMLCallback.log_suffix,
                        config_dict=model.config.to_dict(),
                        description=configuration_object_description,
                    )
                elif not self._clearml_task.get_parameter(ignore_model_config_section, default=True, cast=True):
                    model.config = model.config.from_dict(
                        self._clearml_task.get_configuration_object_as_dict(
                            ClearMLCallback._model_config_section + ClearMLCallback.log_suffix
                        )
                    )
                else:
                    self._clearml_task.set_configuration_object(
                        name=ClearMLCallback._model_config_section + ClearMLCallback.log_suffix,
                        config_dict=model.config.to_dict(),
                        description=configuration_object_description,
                    )

    def on_train_begin(self, args, state, control, model=None, processing_class=None, **kwargs):
        if self._clearml is None:
            return
        self._checkpoints_saved = []
        if state.is_hyper_param_search:
            self._initialized = False
        if not self._initialized:
            self.setup(args, state, model, processing_class, **kwargs)

    def on_train_end(self, args, state, control, **kwargs):
        if ClearMLCallback._should_close_on_train_end:
            self._clearml_task.close()
            ClearMLCallback._train_run_counter = 0

    def on_log(self, args, state, control, model=None, processing_class=None, logs=None, **kwargs):
        if self._clearml is None:
            return
        if not self._initialized:
            self.setup(args, state, model, processing_class, **kwargs)
        if state.is_world_process_zero:
            eval_prefix = "eval_"
            eval_prefix_len = len(eval_prefix)
            test_prefix = "test_"
            test_prefix_len = len(test_prefix)
            single_value_scalars = [
                "train_runtime",
                "train_samples_per_second",
                "train_steps_per_second",
                "train_loss",
                "total_flos",
                "epoch",
            ]
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    if k in single_value_scalars:
                        self._clearml_task.get_logger().report_single_value(
                            name=k + ClearMLCallback.log_suffix, value=v
                        )
                    elif k.startswith(eval_prefix):
                        self._clearml_task.get_logger().report_scalar(
                            title="eval" + ClearMLCallback.log_suffix,
                            series=k[eval_prefix_len:],
                            value=v,
                            iteration=state.global_step,
                        )
                    elif k.startswith(test_prefix):
                        self._clearml_task.get_logger().report_scalar(
                            title="test" + ClearMLCallback.log_suffix,
                            series=k[test_prefix_len:],
                            value=v,
                            iteration=state.global_step,
                        )
                    else:
                        self._clearml_task.get_logger().report_scalar(
                            title="train" + ClearMLCallback.log_suffix,
                            series=k,
                            value=v,
                            iteration=state.global_step,
                        )
                else:
                    logger.warning(
                        "Trainer is attempting to log a value of "
                        f'"{v}" of type {type(v)} for key "{k}" as a scalar. '
                        "This invocation of ClearML logger's  report_scalar() "
                        "is incorrect so we dropped this attribute."
                    )

    def on_save(self, args, state, control, **kwargs):
        if self._log_model and self._clearml_task and state.is_world_process_zero:
            ckpt_dir = f"checkpoint-{state.global_step}"
            artifact_path = os.path.join(args.output_dir, ckpt_dir)
            name = ckpt_dir + ClearMLCallback.log_suffix
            logger.info(f"Logging checkpoint artifact `{name}`. This may take some time.")
            output_model = self._clearml.OutputModel(task=self._clearml_task, name=name)
            output_model.connect(task=self._clearml_task, name=name)
            output_model.update_weights_package(
                weights_path=artifact_path,
                target_filename=ckpt_dir,
                iteration=state.global_step,
                auto_delete_file=False,
            )
            self._checkpoints_saved.append(output_model)
            while args.save_total_limit and args.save_total_limit < len(self._checkpoints_saved):
                try:
                    self._clearml.model.Model.remove(
                        self._checkpoints_saved[0],
                        delete_weights_file=True,
                        force=True,
                        raise_on_errors=True,
                    )
                except Exception as e:
                    logger.warning(
                        "Could not remove checkpoint `{}` after going over the `save_total_limit`. Error is: {}".format(
                            self._checkpoints_saved[0].name, e
                        )
                    )
                    break
                self._checkpoints_saved = self._checkpoints_saved[1:]

    def _copy_training_args_as_hparams(self, training_args, prefix):
        as_dict = {
            field.name: getattr(training_args, field.name)
            for field in fields(training_args)
            if field.init and not field.name.endswith("_token")
        }
        flat_dict = {str(k): v for k, v in self._clearml.utilities.proxy_object.flatten_dictionary(as_dict).items()}
        self._clearml_task._arguments.copy_from_dict(flat_dict, prefix=prefix)


class FlyteCallback(TrainerCallback):
    """A [`TrainerCallback`] that sends the logs to [Flyte](https://flyte.org/).
    NOTE: This callback only works within a Flyte task.

    Args:
        save_log_history (`bool`, *optional*, defaults to `True`):
            When set to True, the training logs are saved as a Flyte Deck.

        sync_checkpoints (`bool`, *optional*, defaults to `True`):
            When set to True, checkpoints are synced with Flyte and can be used to resume training in the case of an
            interruption.

    Example:

    ```python
    # Note: This example skips over some setup steps for brevity.
    from flytekit import current_context, task


    @task
    def train_hf_transformer():
        cp = current_context().checkpoint
        trainer = Trainer(..., callbacks=[FlyteCallback()])
        output = trainer.train(resume_from_checkpoint=cp.restore())
    ```
    """

    def __init__(self, save_log_history: bool = True, sync_checkpoints: bool = True):
        super().__init__()
        if not is_flytekit_available():
            raise ImportError("FlyteCallback requires flytekit to be installed. Run `pip install flytekit`.")

        if not is_flyte_deck_standard_available() or not is_pandas_available():
            logger.warning(
                "Syncing log history requires both flytekitplugins-deck-standard and pandas to be installed. "
                "Run `pip install flytekitplugins-deck-standard pandas` to enable this feature."
            )
            save_log_history = False

        from flytekit import current_context

        self.cp = current_context().checkpoint
        self.save_log_history = save_log_history
        self.sync_checkpoints = sync_checkpoints

    def on_save(self, args, state, control, **kwargs):
        if self.sync_checkpoints and state.is_world_process_zero:
            ckpt_dir = f"checkpoint-{state.global_step}"
            artifact_path = os.path.join(args.output_dir, ckpt_dir)

            logger.info(f"Syncing checkpoint in {ckpt_dir} to Flyte. This may take time.")
            self.cp.save(artifact_path)

    def on_train_end(self, args, state, control, **kwargs):
        if self.save_log_history:
            import pandas as pd
            from flytekit import Deck
            from flytekitplugins.deck.renderer import TableRenderer

            log_history_df = pd.DataFrame(state.log_history)
            Deck("Log History", TableRenderer().to_html(log_history_df))


class DVCLiveCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that sends the logs to [DVCLive](https://www.dvc.org/doc/dvclive).

    Use the environment variables below in `setup` to configure the integration. To customize this callback beyond
    those environment variables, see [here](https://dvc.org/doc/dvclive/ml-frameworks/huggingface).

    Args:
        live (`dvclive.Live`, *optional*, defaults to `None`):
            Optional Live instance. If None, a new instance will be created using **kwargs.
        log_model (Union[Literal["all"], bool], *optional*, defaults to `None`):
            Whether to use `dvclive.Live.log_artifact()` to log checkpoints created by [`Trainer`]. If set to `True`,
            the final checkpoint is logged at the end of training. If set to `"all"`, the entire
            [`TrainingArguments`]'s `output_dir` is logged at each checkpoint.
    """

    def __init__(
        self,
        live: Optional[Any] = None,
        log_model: Optional[Union[Literal["all"], bool]] = None,
        **kwargs,
    ):
        if not is_dvclive_available():
            raise RuntimeError("DVCLiveCallback requires dvclive to be installed. Run `pip install dvclive`.")
        from dvclive import Live

        self._initialized = False
        self.live = None
        if isinstance(live, Live):
            self.live = live
        elif live is not None:
            raise RuntimeError(f"Found class {live.__class__} for live, expected dvclive.Live")

        self._log_model = log_model
        if self._log_model is None:
            log_model_env = os.getenv("HF_DVCLIVE_LOG_MODEL", "FALSE")
            if log_model_env.upper() in ENV_VARS_TRUE_VALUES:
                self._log_model = True
            elif log_model_env.lower() == "all":
                self._log_model = "all"

    def setup(self, args, state, model):
        """
        Setup the optional DVCLive integration. To customize this callback beyond the environment variables below, see
        [here](https://dvc.org/doc/dvclive/ml-frameworks/huggingface).

        Environment:
        - **HF_DVCLIVE_LOG_MODEL** (`str`, *optional*):
            Whether to use `dvclive.Live.log_artifact()` to log checkpoints created by [`Trainer`]. If set to `True` or
            *1*, the final checkpoint is logged at the end of training. If set to `all`, the entire
            [`TrainingArguments`]'s `output_dir` is logged at each checkpoint.
        """
        from dvclive import Live

        self._initialized = True
        if state.is_world_process_zero:
            if not self.live:
                self.live = Live()
            self.live.log_params(args.to_dict())

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if not self._initialized:
            self.setup(args, state, model)

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if not self._initialized:
            self.setup(args, state, model)
        if state.is_world_process_zero:
            from dvclive.plots import Metric
            from dvclive.utils import standardize_metric_name

            for key, value in logs.items():
                if Metric.could_log(value):
                    self.live.log_metric(standardize_metric_name(key, "dvclive.huggingface"), value)
                else:
                    logger.warning(
                        "Trainer is attempting to log a value of "
                        f'"{value}" of type {type(value)} for key "{key}" as a scalar. '
                        "This invocation of DVCLive's Live.log_metric() "
                        "is incorrect so we dropped this attribute."
                    )
            self.live.next_step()

    def on_save(self, args, state, control, **kwargs):
        if self._log_model == "all" and self._initialized and state.is_world_process_zero:
            self.live.log_artifact(args.output_dir)

    def on_train_end(self, args, state, control, **kwargs):
        if self._initialized and state.is_world_process_zero:
            from transformers.trainer import Trainer

            if self._log_model is True:
                fake_trainer = Trainer(
                    args=args,
                    model=kwargs.get("model"),
                    processing_class=kwargs.get("processing_class"),
                    eval_dataset=["fake"],
                )
                name = "best" if args.load_best_model_at_end else "last"
                output_dir = os.path.join(args.output_dir, name)
                fake_trainer.save_model(output_dir)
                self.live.log_artifact(output_dir, name=name, type="model", copy=True)
            self.live.end()


INTEGRATION_TO_CALLBACK = {
    "azure_ml": AzureMLCallback,
    "comet_ml": CometCallback,
    "mlflow": MLflowCallback,
    "neptune": NeptuneCallback,
    "tensorboard": TensorBoardCallback,
    "wandb": WandbCallback,
    "codecarbon": CodeCarbonCallback,
    "clearml": ClearMLCallback,
    "dagshub": DagsHubCallback,
    "flyte": FlyteCallback,
    "dvclive": DVCLiveCallback,
}


def get_reporting_integration_callbacks(report_to):
    if report_to is None:
        return []

    if isinstance(report_to, str):
        if "none" == report_to:
            return []
        elif "all" == report_to:
            report_to = get_available_reporting_integrations()
        else:
            report_to = [report_to]

    for integration in report_to:
        if integration not in INTEGRATION_TO_CALLBACK:
            raise ValueError(
                f"{integration} is not supported, only {', '.join(INTEGRATION_TO_CALLBACK.keys())} are supported."
            )

    return [INTEGRATION_TO_CALLBACK[integration] for integration in report_to]
