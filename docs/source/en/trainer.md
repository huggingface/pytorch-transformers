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

[Accelerate](https://hf.co/docs/accelerate/index) is a library that simplifies training in distributed environments and across different hardware. Its integration with [`Trainer`] means [`Trainer`] supports distributed training frameworks like [Fully Sharded Data Parallel (FSDP)](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/) and [DeepSpeed](https://www.deepspeed.ai/).

> [!TIP]
> Learn more about FSDP sharding strategies, CPU offloading, and more with [`Trainer`] in the [Fully Sharded Data Parallel](./fsdp) guide.

To use Accelerate with [`Trainer`], run the [accelerate_config](https://hf.co/docs/accelerate/package_reference/cli#accelerate-config) command to configure your training environment. This command creates a `config_file.yaml` file that stores the configuration settings of your training environment and it's used whenever you launch your training script. Some example distributed training configurations are shown below.

<hfoptions id="distributed-training">
<hfoption id="DistributedDataParallel">

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0 #change rank as per the node
main_process_ip: 192.168.20.1
main_process_port: 9898
main_training_function: main
mixed_precision: fp16
num_machines: 2
num_processes: 8
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

</hfoption>
<hfoption id="FullyShardedDataParallel">

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
downcast_bf16: 'no'
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch_policy: BACKWARD_PRE
  fsdp_forward_prefetch: true
  fsdp_offload_params: false
  fsdp_sharding_strategy: 1
  fsdp_state_dict_type: FULL_STATE_DICT
  fsdp_sync_module_states: true
  fsdp_transformer_layer_cls_to_wrap: BertLayer
  fsdp_use_orig_params: true
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 2
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

</hfoption>
<hfoption id="DeepSpeed">

```yaml
compute_environment: LOCAL_MACHINE
deepspeed_config:
  deepspeed_config_file: /home/user/configs/ds_zero3_config.json
  zero3_init_flag: true
distributed_type: DEEPSPEED
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
num_machines: 1
num_processes: 4
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

</hfoption>
<hfoption id="DeepSpeed with Accelerate plugin">

```yaml
compute_environment: LOCAL_MACHINE
deepspeed_config:
  gradient_accumulation_steps: 1
  gradient_clipping: 0.7
  offload_optimizer_device: cpu
  offload_param_device: cpu
  zero3_init_flag: true
  zero_stage: 2
distributed_type: DEEPSPEED
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 4
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

</hfoption>
</hfoptions>

Run [accelerate_launch](https://hf.co/docs/accelerate/package_reference/cli#accelerate-launch) to start training with the configurations set in `config_file.yaml`. This file is saved to the Accelerate cache folder and automatically loaded when you run `accelerate_launch`.

The example below launches the [run_glue.py](../../../examples/pytorch/text-classification/run_glue) script with the FSDP configuration shown earlier. Parameters from the `config_file.yaml` file can also be directly set in the command line.

```bash
accelerate launch \
    ./examples/pytorch/text-classification/run_glue.py \
    --model_name_or_path google-bert/bert-base-cased \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --max_seq_length 128 \
    --per_device_train_batch_size 16 \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --output_dir /tmp/$TASK_NAME/ \
    --overwrite_output_dir
```

> [!TIP]
> Refer to the [Launching your Accelerate scripts](https://hf.co/docs/accelerate/basic_tutorials/launch) tutorial to learn more about `accelerate_launch` and custom configurations.

## Optimization

[`Trainer`] supports various optimizations to improve *training* performance - reduce memory and increase training speed - and *model* performance.

### GaLore

[Gradient Low-Rank Projection (GaLore)](https://hf.co/papers/2403.03507) significantly reduces memory usage when training large language models (LLMs). One of GaLores key benefits is *full-parameter* learning, unlike low-rank adaptation methods like [LoRA](https://hf.co/papers/2106.09685), which produces better model performance.

Install the [GaLore](https://github.com/jiaweizzhao/GaLore) library, [TRL](https://hf.co/docs/trl/index), and [Datasets](https://hf.co/docs/datasets/index).

```bash
pip install galore-torch trl datasets
```

Then pick a GaLore optimizer (`"galore_adamw"`, `"galore_adafactor"`, `"galore_adamw_8bit`") and pass it to the `optim` parameter in [`TrainingArguments`]. Use the `optim_target_modules` parameter to specify which modules to adapt (can be a list of strings, regex, or a full path).

Extra parameters supported by GaLore, `rank`, `update_proj_gap`, and `scale` should be passed to the `optim_args` parameter in [`TrainingArguments`].

The example below shows how to enable GaLore with [`~trl.SFTTrainer`] that targets the `attn` and `mlp` layers with regex.

> [!TIP]
> It can take some time before training starts (~3 minutes for a 2B model on a NVIDIA A100).

<hfoptions id="galore">
<hfoption id="GaLore optimizer">

```py
import torch
import datasets
import trl
from transformers import TrainingArguments, AutoConfig, AutoTokenizer, AutoModelForCausalLM

train_dataset = datasets.load_dataset('imdb', split='train')
args = TrainingArguments(
    output_dir="./test-galore",
    max_steps=100,
    per_device_train_batch_size=2,
    optim="galore_adamw",
    optim_target_modules=[r".*.attn.*", r".*.mlp.*"],
    optim_args="rank=64, update_proj_gap=100, scale=0.10",
)
config = AutoConfig.from_pretrained("google/gemma-2b")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
model = AutoModelForCausalLM.from_config("google/gemma-2b").to(0)
trainer = trl.SFTTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    dataset_text_field='text',
    max_seq_length=512,
)
trainer.train()
```

</hfoption>
<hfoption id="GaLore optimizer with layerwise optimization">

Append `layerwise` to the optimizer name to enable layerwise optimization. For example, `"galore_adamw"` becomes `"galore_adamw_layerwise"`. This feature is still experimental and does not support Distributed Data Parallel (DDP). The code below can only be run on a [single GPU](https://github.com/jiaweizzhao/GaLore?tab=readme-ov-file#train-7b-model-with-a-single-gpu-with-24gb-memory). Other features like gradient clipping and DeepSpeed may not be available out of the box. Feel free to open an [issue](https://github.com/huggingface/transformers/issues) if you encounter any problems!

```py
import torch
import datasets
import trl
from transformers import TrainingArguments, AutoConfig, AutoTokenizer, AutoModelForCausalLM

train_dataset = datasets.load_dataset('imdb', split='train')
args = TrainingArguments(
    output_dir="./test-galore",
    max_steps=100,
    per_device_train_batch_size=2,
    optim="galore_adamw_layerwise",
    optim_target_modules=[r".*.attn.*", r".*.mlp.*"],
    optim_args="rank=64, update_proj_gap=100, scale=0.10",
)
config = AutoConfig.from_pretrained("google/gemma-2b")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
model = AutoModelForCausalLM.from_config("google/gemma-2b").to(0)
trainer = trl.SFTTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    dataset_text_field='text',
    max_seq_length=512,
)
trainer.train()
```

</hfoption>
</hfoptions>

Only linear layers that are considered GaLore layers can be trained with low-rank decomposition. The rest of the model layers are optimized in the usual way.

### Liger

[Liger Kernel](https://github.com/linkedin/Liger-Kernel) is a collection of layers such as RMSNorm, RoPE, SwiGLU, CrossEntropy, FusedLinearCrossEntropy and more that have been fused into a single Triton kernel for training LLMs. These kernels are also compatible with FlashAttention, FSDP, and DeepSpeed. As a result, Liger Kernel can increase multi-GPU training throughput and reduce memory usage. This is useful for multi-head training and supporting larger vocabulary sizes, larger batch sizes and longer context lengths.

```bash
pip install liger-kernel
```

Enable Liger Kernel for training by setting `use_liger_kernel=True` in [`TrainingArguments`]. This patches the corresponding layers in the model with Ligers kernels.

> [!TIP]
> Liger Kernel supports Llama, Gemma, Mistral, and Mixtral models. Refer to the [patching](https://github.com/linkedin/Liger-Kernel#patching) list for the latest list of supported models.

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
    use_liger_kernel=True
)
```

### NEFTune

[NEFTune](https://hf.co/papers/2310.05914) adds noise to the embedding vectors during training to improve model performance. Enable it in [`Trainer`] with the `neftune_noise_alpha` parameter in [`TrainingArguments`] to control how much noise is added.

```py
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(..., neftune_noise_alpha=0.1)
trainer = Trainer(..., args=training_args)
```

The original embedding layer is restored after training to avoid any unexpected behavior.
