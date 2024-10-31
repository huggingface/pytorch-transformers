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

# Accelerate

[Accelerate](https://hf.co/docs/accelerate/index) is a library designed to simplify distributed training on any type of setup with PyTorch by uniting the most common frameworks ([Fully Sharded Data Parallel (FSDP)](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/) and [DeepSpeed](https://www.deepspeed.ai/)) for it into a single interface. [`Trainer`] is powered by Accelerate under the hood, enabling loading big models and distributed training.

This guide will show you two ways to use Accelerate with Transformers, using FSDP as the backend. The first method demonstrates distributed training with [`Trainer`], and the second method demonstrates adapting a PyTorch training loop. For more detailed information about Accelerate, please refer to the [documentation](https://hf.co/docs/accelerate/index).

```bash
pip install accelerate
```

Start by running [accelerate config](https://hf.co/docs/accelerate/main/en/package_reference/cli#accelerate-config) in the command line to answer a series of prompts about your training system. This creates and saves a configuration file to help Accelerate correctly set up training based on your setup.

```bash
accelerate config
```

Depending on your setup and the answers you provide, an example configuration file for distributing training with FSDP on one machine with two GPUs.

```yaml
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: FSDP
downcast_bf16: 'no'
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch_policy: BACKWARD_PRE
  fsdp_forward_prefetch: false
  fsdp_cpu_ram_efficient_loading: true
  fsdp_offload_params: false
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_state_dict_type: SHARDED_STATE_DICT
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

## Trainer

Pass the path to the saved configuration file to [`TrainingArguments`], and from there, pass your [`TrainingArguments`] to [`Trainer`].

```py
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="your-model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    fsdp_config="path/to/fsdp_config",
    fsdp_strategy="full_shard",
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=True,
)

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

## Native PyTorch

Accelerate can also be added to any PyTorch training loop to enable distributed training. The [`~accelerate.Accelerator`] is the main entry point for adapting your PyTorch code to work with Accelerate. It automatically detects your distributed training setup and initializes all the necessary components for training. You don't need to explicitly place your model on a device because [`~accelerate.Accelerator`] knows which device to move your model to.

```py
from accelerate import Accelerator

accelerator = Accelerator()
device = accelerator.device
```

All PyTorch objects (model, optimizer, scheduler, dataloaders) should be passed to the [`~accelerate.Accelerator.prepare`] method now. This method moves your model to the appropriate device or devices, adapts the optimizer and scheduler to use [`~accelerate.AcceleratedOptimizer`] and [`~accelerate.AcceleratedScheduler`], and creates a new shardable dataloader.

```py
train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
    train_dataloader, eval_dataloader, model, optimizer
)
```

Replace `loss.backward` in your training loop with Accelerates [`~accelerate.Accelerator.backward`] method to scale the gradients and determine the appropriate `backward` method to use depending on your framework (for example, DeepSpeed or Megatron).

```py
for epoch in range(num_epochs):
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
```

Combine everything into a function and make it callable as a script.

```py
from accelerate import Accelerator
  
def main():
  accelerator = Accelerator()

  model, optimizer, training_dataloader, scheduler = accelerator.prepare(
      model, optimizer, training_dataloader, scheduler
  )

  for batch in training_dataloader:
      optimizer.zero_grad()
      inputs, targets = batch
      outputs = model(inputs)
      loss = loss_function(outputs, targets)
      accelerator.backward(loss)
      optimizer.step()
      scheduler.step()

if __name__ == "__main__":
    main()
```

From the command line, call [accelerate launch](https://hf.co/docs/accelerate/main/en/package_reference/cli#accelerate-launch) to run your training script. Any additional arguments or parameters can be passed here as well.

To launch your training script on two GPUs, add the `--num_processes` argument.

```bash
accelerate launch --num_processes=2 your_script.py
```

Refer to the [Launching Accelerate scripts](https://hf.co/docs/accelerate/main/en/basic_tutorials/launch) for more details.
