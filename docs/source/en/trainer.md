<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Trainer

[`Trainer`] is a complete training and evaluation loop for Transformers' PyTorch models. Plug a model, preprocessor, dataset, and training arguments into [`Trainer`] and let it handle the rest to start training faster.

This guide will show you how [`Trainer`] works and how to customize it for your use case with a callback.

[`Trainer`] is powered by [Accelerate](https://hf.co/docs/accelerate/index), a library for handling large models for distributed training, so make sure it is installed.

```bash
!pip install accelerate

# upgrade to the latest version
# !pip install accelerate --upgrade
```

[`Trainer`] contains all the necessary components of a training loop.

1. calculate the loss from a training step
2. calculate the gradients with the [`~accelerate.Accelerator.backward`] method
3. update the weights based on the gradients
4. repeat until the predetermined number of epochs is reached

Manually coding this training loop everytime can be inconvenient or a barrier if you're just getting started with machine learning. [`Trainer`] abstracts this process, allowing you to focus on the model, dataset, and training design choices.

Configure your training with hyperparameters and options from [`TrainingArguments`] which supports a ton of features such as distributed training, torch.compile, mixed precision training, and saving the model to the Hub.

> [!TIP]
> The number of available parameters available in [`TrainingArguments`] may be intimidating at first. If there is a specific hyperparameter or feature you want to use, try searching for it directly. Otherwise, feel free to start with the default values and gradually customize them as you become more familiar with the training process.

The example below demonstrates an example instance of [`TrainingArguments`] that evaluates and saves the model at the end of each epoch. It also loads the best model found during training and pushes it to the Hub.

```py
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="your-model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=True,
)
```

Pass your model, dataset, preprocessor, and [`TrainingArguments`] to [`Trainer`] and call [`~Trainer.train`] to start training.

> [!TIP]
> Refer to the [Finetuning](./training) guide for a more complete overview of the training process.

```py
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
```

## Checkpoints

[`Trainer`] saves checkpoints (the optimizer state is not saved by default) to the directory set to `output_dir` in [`TrainingArguments`] to a subfolder named `checkpoint-000`. The number at the end is the training step at which the checkpoint was saved.

Saving checkpoints are useful for resuming training or recovering your training progress if you encounter an error. Set the `resume_from_checkpoint` parameter in [`~Trainer.train`] to resume training from the last checkpoint or a specific checkpoint.

<hfoptions id="ckpt">
<hfoption id="latest checkpoint">

```py
trainer.train(resume_from_checkpoint=True)
```

</hfoption>
<hfoption id="specific checkpoint">

```py
trainer.train(resume_from_checkpoint="your-model/checkpoint-1000")
```

</hfoption>
</hfoptions>

Checkpoints can be saved to the Hub by setting `push_to_hub=True` in [`TrainingArguments`]. The default method (`"every_save"`) saves a checkpoint to the Hub is every time a model is saved, which is typically the final model at the end of training. Some other options for deciding how to save checkpoints to the Hub include:

- `hub_strategy="end"` only pushes a checkpoint when [`~Trainer.save_model`] is called
- `hub_strategy="checkpoint"` pushes the latest checkpoint to a subfolder named *last-checkpoint* from which training can be resumed
- `hub_strategy="all_checkpoints"` pushes all checkpoints to the Hub with one checkpoint per subfolder in your model repository

[`Trainer`] attempts to maintain the same Python, NumPy, and PyTorch RNG states when you resume training from a checkpoint. But PyTorch has various non-deterministic settings which can't guarantee the RNG states are identical. To enable full determinism, refer to the [Controlling sources of randomness](https://pytorch.org/docs/stable/notes/randomness#controlling-sources-of-randomness) guide to learn what settings to adjust to make training fully deterministic (some settings may result in slower training).

## Logging

[`Trainer`] is set to `logging.INFO` by default to report errors, warnings, and other basic information. Use [`~TrainingArguments.log_level`] to change the logging level and log verbosity.

The example below sets the main code and modules to use the same log level.

```py
logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

log_level = training_args.get_process_log_level()
logger.setLevel(log_level)
datasets.utils.logging.set_verbosity(log_level)
transformers.utils.logging.set_verbosity(log_level)

trainer = Trainer(...)
```

In a distributed environment, [`Trainer`] replicas are set to `logging.WARNING` to only report errors and warnings. Use [`~TrainingArguments.log_level_replica`] to change the logging level and log verbosity. To configure the log level for each node, use [`~TrainingArguments.log_on_each_node`] to determine whether to use a specific log level on each node or only the main node.

Use different combinations of `log_level` and `log_level_replica` to configure what gets logged on each node.

<hfoptions id="nodes">
<hfoption id="single node">

```bash
my_app.py ... --log_level warning --log_level_replica error
```

</hfoption>
<hfoption id="multi-node">

Add `log_on_each_node 0` for distributed environments.

```bash
my_app.py ... --log_level warning --log_level_replica error --log_on_each_node 0

# set to only report errors
my_app.py ... --log_level error --log_level_replica error --log_on_each_node 0
```

</hfoption>
</hfoptions>

> [!TIP]
> [`Trainer`] sets the log level separately for each node in the [`~Trainer.__init__`] method, so you may want to consider setting this sooner if you're using other Transformers functionalities before creating the [`Trainer`] instance.

## Customize

Tailor [`Trainer`] to your use case by subclassing or overriding its methods to support the functionality you want to add or use, without rewriting the entire training loop from scratch. The table below lists some of the methods that can be customized.

| method | description |
|---|---|
| [`~Trainer.get_train_dataloader`] | create a training DataLoader |
| [`~Trainer.get_eval_dataloader`] | create an evaluation DataLoader |
| [`~Trainer.get_test_dataloader`] | create a test DataLoader |
| [`~Trainer.log`] | log information about the training process |
| [`~Trainer.create_optimizer_and_scheduler`] | create an optimizer and learning rate scheduler (can also be separately customized with [`~Trainer.create_optimizer`] and [`~Trainer.screate_scheduler`] if they weren't passed in `__init__` |
| [`~Trainer.compute_loss`] | compute the loss of a batch of training inputs |
| [`~Trainer.training_step`] | perform the training step |
| [`~Trainer.prediction_step`] | perform the prediction and test step |
| [`~Trainer.evaluate`] | evaluate the model and return the evaluation metric |
| [`~Trainer.predict`] | make a prediction (with metrics if labels are available) on the test set |

For example, to use weighted loss, rewrite [`~Trainer.compute_loss`] inside your custom [`Trainer`].

```py
from torch import nn
from transformers import Trainer

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss for 3 labels with different weights
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 3.0], device=model.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
```

### Callbacks

[Callbacks](./main_classes/callback) are another way to customize [`Trainer`], but they *don't change anything* inside the training loop. Instead, a callback inspects the training loop state and then executes some action (early stopping, logging, etc.) depending on the state. For example, you can't implement a custom loss function with a callback because that requires overriding [`~Trainer.compute_loss`].

To use a callback, create a class that inherits from [`TrainerCallback`] and implements the functionality you want. Then you can pass the callback to the `callback` parameter in [`Trainer`]. The example below implements an early stopping callback that stops training after 10 steps.

```py
from transformers import TrainerCallback, Trainer

class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, num_steps=10):
        self.num_steps = num_steps

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.num_steps:
            return {"should_training_stop": True}
        else:
            return {}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback()],
)
```

## Accelerate

## Optimization

### NEFTune

### GaLore

### Liger