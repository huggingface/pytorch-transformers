<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Hyperparameter search

Hyperparameter search discovers an optimal set of hyperparameters that produces the best model performance. [`Trainer`] supports several hyperparameter search backends - [Optuna](https://optuna.readthedocs.io/en/stable/index.html), [SigOpt](https://docs.sigopt.com/), [Weights & Biases](https://docs.wandb.ai/), [Ray Tune](https://docs.ray.io/en/latest/tune/index.html) - through  [`~Trainer.hyperparameter_search`] to optimize an objective or even multiple objectives.

This guide will go over how to set up a hyperparameter search for each of the backends.

```bash
pip install optuna/sigopt/wandb/ray[tune]
```

To use [`~Trainer.hyperparameter_search`], you need to create a `model_init` function. This function includes basic model information (arguments and configuration) because it needs to be reinitialized for each search trial in the run.

> [!WARNING]
> The `model_init` function is incompatible with the [optimizers](./main_classes/trainer#transformers.Trainer.optimizers) parameter. Subclass [`Trainer`] and override the [`~Trainer.create_optimizer_and_scheduler`] method to create a custom optimizer and scheduler.

An example `model_init` function is shown below.

```py
def model_init(trial):
    return AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=True if model_args.use_auth_token else None,
    )
```

Pass `model_init` to [`Trainer`] along with everything else you need for training. Then you can call [`~Trainer.hyperparameter_search`] to start the search.

[`~Trainer.hyperparameter_search`] accepts a [direction](./main_classes/trainer#transformers.Trainer.hyperparameter_search.direction) parameter to specify whether to minimize, maximize, or minimize and maximize multiple objectives. You'll also need to set the [backend](./main_classes/trainer#transformers.Trainer.hyperparameter_search.backend) you're using, an [object](./main_classes/trainer#transformers.Trainer.hyperparameter_search.hp_space) containing the hyperparameters to optimize for, the [number of trials](./main_classes/trainer#transformers.Trainer.hyperparameter_search.n_trials) to run, and a [compute_objective](./main_classes/trainer#transformers.Trainer.hyperparameter_search.compute_objective) to return the objective values.

> [!TIP]
> If [compute_objective](./main_classes/trainer#transformers.Trainer.hyperparameter_search.compute_objective) isn't defined, the default [compute_objective](./main_classes/trainer#transformers.Trainer.hyperparameter_search.compute_objective) is called which is the sum of an evaluation metric like F1.

```py
from transformers import Trainer

trainer = Trainer(
    model=None,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
    processing_class=tokenizer,
    model_init=model_init,
    data_collator=data_collator,
)
trainer.hyperparameter_search(...)
```

The following examples demonstrate how to perform a hyperparameter search for the learning rate and training batch size using the different backends.

<hfoptions id="backends">
<hfoption id="Optuna">

[Optuna](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/002_configurations.html#sphx-glr-tutorial-10-key-features-002-configurations-py) optimizes categorical, integers, and floats.

```py
def optuna_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [16, 32, 64, 128]),
    }

best_trials = trainer.hyperparameter_search(
    direction=["minimize", "maximize"],
    backend="optuna",
    hp_space=optuna_hp_space,
    n_trials=20,
    compute_objective=compute_objective,
)
```

</hfoption>
<hfoption id="Ray Tune">

[Ray Tune](https://docs.ray.io/en/latest/tune/api/search_space.html) optimizes floats, integers, and categorical parameters but it also offers multiple sampling distributions for each parameter such as uniform and log-uniform.

```py
def ray_hp_space(trial):
    return {
        "learning_rate": tune.loguniform(1e-6, 1e-4),
        "per_device_train_batch_size": tune.choice([16, 32, 64, 128]),
    }

best_trials = trainer.hyperparameter_search( 
    direction=["minimize", "maximize"],
    backend="ray",
    hp_space=ray_hp_space,
    n_trials=20,
    compute_objective=compute_objective,
)
```

</hfoption>
<hfoption id="SigOpt">

[SigOpt](https://docs.sigopt.com/ai-module-api-references/api_reference/objects/object_parameter) optimizes double, integer, and categorical parameters.

```py
def sigopt_hp_space(trial):
    return [
        {"bounds": {"min": 1e-6, "max": 1e-4}, "name": "learning_rate", "type": "double"},
        {
            "categorical_values": ["16", "32", "64", "128"],
            "name": "per_device_train_batch_size",
            "type": "categorical",
        },
    ]

best_trials = trainer.hyperparameter_search( 
    direction=["minimize", "maximize"],
    backend="sigopt",
    hp_space=sigopt_hp_space,
    n_trials=20,
    compute_objective=compute_objective,
)
```

</hfoption>
<hfoption id="Weights & Biases">

[Weights & Biases](https://docs.wandb.ai/guides/sweeps/sweep-config-keys) also optimizes integers, floats, and categorical parameters but it also includes support for different search strategies and distribution options.

```py
def wandb_hp_space(trial):
    return {
        "method": "random",
        "metric": {"name": "objective", "goal": "minimize"},
        "parameters": {
            "learning_rate": {"distribution": "uniform", "min": 1e-6, "max": 1e-4},
            "per_device_train_batch_size": {"values": [16, 32, 64, 128]},
        },
    }

best_trials = trainer.hyperparameter_search( 
    direction=["minimize", "maximize"],
    backend="wandb",
    hp_space=wandb_hp_space,
    n_trials=20,
    compute_objective=compute_objective,
)
```

</hfoption>
</hfoptions>

## Distributed Data Parallel

[`Trainer`] only supports hyperparameter search for distributed data parallel (DDP) on the Optuna and SigOpt backends. Only the rank-zero process is used to generate the search trial, and the resulting parameters are passed along to the other ranks.
