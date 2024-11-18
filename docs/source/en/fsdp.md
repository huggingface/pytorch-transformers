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

# FullyShardedDataParallel

[Fully Sharded Data Parallel (FSDP)](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/) is a [parallelism](./perf_train_gpu_many) method that combines the advantages of data and model parallelism for distributed training.

Unlike [DistributedDataParallel (DDP)](./perf_train_gpu_many#distributeddataparallel), FSDP saves more memory because it doesn't replicate a model on each GPU. It shards the models parameters, gradients and optimizer states across GPUs. Each model shard processes a portion of the data and the results are synchronized to speed up training.

This guide covers how to setup training a model with FSDP using [Accelerate](https://hf.co/docs/accelerate/index), a library for managing distributed training.

```bash
pip install accelerate
```

## Configuration options

Always start by running the [accelerate config](https://hf.co/docs/accelerate/package_reference/cli#accelerate-config) command to help Accelerate setup the correct distributed training environment.

```bash
accelerate config
```

The section below discusses some of the more important FSDP configuration options. Learn more about other available options in the [fsdp_config](https://hf.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments.fsdp_config) parameter.

### Sharding strategy

FSDP offers several sharding strategies to distribute a model. Refer to the table below to help you choose the best strategy for your setup. Specify a strategy with the `fsdp_sharding_strategy` parameter in the configuration file.

| sharding strategy | description | parameter value |
|---|---|---|
| `FULL_SHARD` | shards model parameters, gradients, and optimizer states | `1` |
| `SHARD_GRAD_OP` | shards gradients and optimizer states | `2` |
| `NO_SHARD` | don't shard the model | `3` |
| `HYBRID_SHARD` | shards model parameters, gradients, and optimizer states within each GPU | `4` |
| `HYBRID_SHARD_ZERO2` | shards gradients and optimizer states within each GPU | `5` |

### CPU offload

Offload model parameters and gradients when they aren't being used to the CPU to save additional GPU memory. This is useful for scenarios where a model is too large even with FSDP.

Specify `fsdp_offload_params: true` in the configuration file to enable offloading.

### Wrapping policy

FSDP is applied by wrapping each layer in the network. The wrapping is usually applied in a nested way where the full weights are discarded after each forward pass to save memory for the next layer.

There are several wrapping policies available, but the *auto wrapping* policy is the simplest and doesn't require any changes to your code. Specify `fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP` to wrap a Transformer layer and `fsdp_transformer_layer_cls_to_wrap` to determine which layer to wrap (for example, `BertLayer`).

Size-based wrapping is also available. If a layer exceeds a certain number of parameters, it is wrapped. Specify `fsdp_wrap_policy: SIZED_BASED_WRAP` and `min_num_param` to set the minimum number of parameters for a layer to be wrapped.

### Checkpoints

Intermediate checkpoints should be saved as a sharded state dict because saving the full state dict - even with CPU offloading - is time consuming and can cause `NCCL Timeout` errors due to indefinite hanging during broadcasting.

Specify `fsdp_state_dict_type: SHARDED_STATE_DICT` in the configuration file to save the sharded state dict. Now you can resume training from the sharded state dict with the [`~accelerate.Accelerator.load_state`] method.

```py
accelerator.load_state("directory/containing/checkpoints")
```

Once training is complete though, you should save the full state dict because the sharded state dict is only compatible with FSDP.

```py
if trainer.is_fsdp_enabled:
  trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

trainer.save_model(script_args.output_dir)
```

### TPU

[PyTorch XLA](https://pytorch.org/xla/release/2.1/index.html), a package for running PyTorch on XLA devices, enables FSDP on TPUs. Modify the configuration file to include the parameters below. Refer to the [xla_fsdp_settings](https://github.com/pytorch/xla/blob/2e6e183e0724818f137c8135b34ef273dea33318/torch_xla/distributed/fsdp/xla_fully_sharded_data_parallel.py#L128) parameter for additional XLA-specific parameters you can configure for FSDP.

```yaml
xla: True # must be set to True to enable PyTorch/XLA
xla_fsdp_settings: # XLA specific FSDP parameters
xla_fsdp_grad_ckpt: True # enable gradient checkpointing
```

## Training

After running [accelerate config](https://hf.co/docs/accelerate/package_reference/cli#accelerate-config), your configuration file should be ready. An example configuration file is show below that fully shards the parameter, gradient and optimizer states on two GPUs. Your file may look different depending on how you setup your configuration.

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

Run the [accelerate launch](https://hf.co/docs/accelerate/package_reference/cli#accelerate-launch) command to launch a training script with the FSDP configurations you chose in the configuration file.

```bash
accelerate launch my-training-script.py
```

It is also possible to directly specify some of the FSDP arguments in the command line.

```bash
accelerate launch --fsdp="full shard" --fsdp_config="path/to/fsdp_config/" my-training-script.py
```

## Resources

FSDP is a powerful tool for training large models with fewer GPUs compared to some other parallelism strategies. Refer to the following resources below to learn even more about FSDP.

- Follow along with the more in-depth Accelerate guide for [FSDP](https://hf.co/docs/accelerate/usage_guides/fsdp).
- Read the [Introducing PyTorch Fully Sharded Data Parallel (FSDP) API](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/) blog post.
- Read the [Scaling PyTorch models on Cloud TPUs with FSDP](https://pytorch.org/blog/scaling-pytorch-models-on-cloud-tpus-with-fsdp/) blog post.
