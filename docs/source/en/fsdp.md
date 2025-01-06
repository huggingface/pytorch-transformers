<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Fully Sharded Data Parallel

[Fully Sharded Data Parallel (FSDP)](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/) is a data parallel method that shards a model's parameters, gradients and optimizer states across the number of available GPUs (also called workers or *rank*). Unlike [DistributedDataParallel (DDP)](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html), FSDP reduces memory-usage because a model is replicated on each GPU. This improves GPU memory-efficiency and allows you to train much larger models on fewer GPUs. FSDP is integrated with the Accelerate, a library for easily managing training in distributed environments, which means it is available for use from the [`Trainer`] class.

Before you start, make sure Accelerate is installed and at least PyTorch 2.1.0 or newer.

```bash
pip install accelerate
```

## FSDP configuration

To start, run the [`accelerate config`](https://huggingface.co/docs/accelerate/package_reference/cli#accelerate-config) command to create a configuration file for your training environment. Accelerate uses this configuration file to automatically setup the correct training environment based on your selected training options in `accelerate config`.

```bash
accelerate config
```

When you run `accelerate config`, you'll be prompted with a series of options to configure your training environment. This section covers some of the most important FSDP options. To learn more about the other available FSDP options, take a look at the [fsdp_config](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments.fsdp_config) parameters.

### Sharding strategy

FSDP offers a number of sharding strategies to select from:

* `FULL_SHARD` - shards model parameters, gradients and optimizer states across workers; select `1` for this option
* `SHARD_GRAD_OP`- shard gradients and optimizer states across workers; select `2` for this option
* `NO_SHARD` - don't shard anything (this is equivalent to DDP); select `3` for this option
* `HYBRID_SHARD` - shard model parameters, gradients and optimizer states within each worker where each worker also has a full copy; select `4` for this option
* `HYBRID_SHARD_ZERO2` - shard gradients and optimizer states within each worker where each worker also has a full copy; select `5` for this option

This is enabled by the `fsdp_sharding_strategy` flag.

### CPU offload

You could also offload parameters and gradients when they are not in use to the CPU to save even more GPU memory and help you fit large models where even FSDP may not be sufficient. This is enabled by setting `fsdp_offload_params: true` when running `accelerate config`.

### Wrapping policy

FSDP is applied by wrapping each layer in the network. The wrapping is usually applied in a nested way where the full weights are discarded after each forward pass to save memory for use in the next layer. The *auto wrapping* policy is the simplest way to implement this and you don't need to change any code. You should select `fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP` to wrap a Transformer layer and `fsdp_transformer_layer_cls_to_wrap` to specify which layer to wrap (for example `BertLayer`).

Otherwise, you can choose a size-based wrapping policy where FSDP is applied to a layer if it exceeds a certain number of parameters. This is enabled by setting `fsdp_wrap_policy: SIZE_BASED_WRAP` and `min_num_param` to the desired size threshold.

### Checkpointing

Intermediate checkpoints should be saved with `fsdp_state_dict_type: SHARDED_STATE_DICT` because saving the full state dict with CPU offloading on rank 0 takes a lot of time and often results in `NCCL Timeout` errors due to indefinite hanging during broadcasting. You can resume training with the sharded state dicts with the [`~accelerate.Accelerator.load_state`] method.

```py
# directory containing checkpoints
accelerator.load_state("ckpt")
```

However, when training ends, you want to save the full state dict because sharded state dict is only compatible with FSDP.

```py
if trainer.is_fsdp_enabled:
    trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

trainer.save_model(script_args.output_dir)
```

### TPU

[PyTorch XLA](https://pytorch.org/xla/release/2.1/index.html) supports FSDP training for TPUs and it can be enabled by modifying the FSDP configuration file generated by `accelerate config`. In addition to the sharding strategies and wrapping options specified above, you can add the parameters shown below to the file.

```yaml
xla: True # must be set to True to enable PyTorch/XLA
xla_fsdp_settings: # XLA-specific FSDP parameters
xla_fsdp_grad_ckpt: True # use gradient checkpointing
```

The [`xla_fsdp_settings`](https://github.com/pytorch/xla/blob/2e6e183e0724818f137c8135b34ef273dea33318/torch_xla/distributed/fsdp/xla_fully_sharded_data_parallel.py#L128) allow you to configure additional XLA-specific parameters for FSDP.

## Launch training

An example FSDP configuration file may look like:

```yaml
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: FSDP
downcast_bf16: 'no'
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch_policy: BACKWARD_PRE
  fsdp_cpu_ram_efficient_loading: true
  fsdp_forward_prefetch: false
  fsdp_offload_params: true
  fsdp_sharding_strategy: 1
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

To launch training, run the [`accelerate launch`](https://huggingface.co/docs/accelerate/package_reference/cli#accelerate-launch) command and it'll automatically use the configuration file you previously created with `accelerate config`.

```bash
accelerate launch my-trainer-script.py
```

```bash
accelerate launch --fsdp="full shard" --fsdp_config="path/to/fsdp_config/ my-trainer-script.py
```

## Next steps

FSDP can be a powerful tool for training really large models and you have access to more than one GPU or TPU. By sharding the model parameters, optimizer and gradient states, and even offloading them to the CPU when they're inactive, FSDP can reduce the high cost of large-scale training. If you're interested in learning more, the following may be helpful:

* Follow along with the more in-depth Accelerate guide for [FSDP](https://huggingface.co/docs/accelerate/usage_guides/fsdp).
* Read the [Introducing PyTorch Fully Sharded Data Parallel (FSDP) API](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/) blog post.
* Read the [Scaling PyTorch models on Cloud TPUs with FSDP](https://pytorch.org/blog/scaling-pytorch-models-on-cloud-tpus-with-fsdp/) blog post.
