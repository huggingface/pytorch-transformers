<!--
Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# 完全分片数据并行

[完全分片数据并行（FSDP）](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/)是一种数据并行方法，
它将模型的参数、梯度和优化器状态在可用 GPU（也称为 Worker 或 *rank*）的数量上进行分片。
与[分布式数据并行（DDP）](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)不同，
FSDP 减少了内存使用量，因为模型在每个 GPU 上都被复制了一次。这就提高了 GPU 内存效率，
使您能够用较少的 GPU 训练更大的模型。FSDP 已经集成到 Accelerate 中，
这是一个用于在分布式环境中轻松管理训练的库，这意味着可以从 [`Trainer`] 类中调用这个库。

在开始之前，请确保已安装 Accelerate，并且至少使用 PyTorch 2.1.0 或更高版本。

```bash
pip install accelerate
```

## FSDP 配置

首先，运行 [`accelerate config`](https://huggingface.co/docs/accelerate/package_reference/cli#accelerate-config)
命令为您的训练环境创建一个配置文件。Accelerate 使用此配置文件根据您在 `accelerate config`
中选择的训练选项来自动搭建正确的训练环境。

```bash
accelerate config
```

运行 `accelerate config` 时，您将被提示一系列选项来配置训练环境。
本节涵盖了一些最重要的 FSDP 选项。要了解有关其他可用的 FSDP 选项的更多信息，
请查阅 [fsdp_config](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments.fsdp_config) 参数。

### 分片策略

FSDP 提供了多种可选择的分片策略：

- `FULL_SHARD` - 将模型参数、梯度和优化器状态跨 Worker 进行分片；为此选项选择 `1`
- `SHARD_GRAD_OP`- 将梯度和优化器状态跨 Worker 进行分片；为此选项选择 `2`
- `NO_SHARD` - 不分片任何内容（这等同于 DDP）；为此选项选择 `3`
- `HYBRID_SHARD` - 在每个 Worker 中分片模型参数、梯度和优化器状态，其中每个 Worker 也有完整副本；为此选项选择 `4`
- `HYBRID_SHARD_ZERO2` - 在每个 Worker 中分片梯度和优化器状态，其中每个 Worker 也有完整副本；为此选项选择 `5`

这由 `fsdp_sharding_strategy` 标志启用。

### CPU 卸载

当参数和梯度在不使用时可以卸载到 CPU 上，以节省更多 GPU 内存并帮助您适应即使 FSDP 也不足以容纳大型模型的情况。
在运行 `accelerate config` 时，通过设置 `fsdp_offload_params: true` 来启用此功能。

### 包装策略

FSDP 是通过包装网络中的每个层来应用的。通常，包装是以嵌套方式应用的，其中完整的权重在每次前向传递后被丢弃，
以便在下一层使用内存。**自动包装**策略是实现这一点的最简单方法，您不需要更改任何代码。
您应该选择 `fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP` 来包装一个 Transformer 层，
并且 `fsdp_transformer_layer_cls_to_wrap` 来指定要包装的层（例如 `BertLayer`）。

否则，您可以选择基于大小的包装策略，其中如果一层的参数超过一定数量，则应用 FSDP。通过设置
`fsdp_wrap_policy: SIZE_BASED_WRAP` 和 `min_num_param` 来启用此功能，将参数设置为所需的大小阈值。

### 检查点

应该使用 `fsdp_state_dict_type: SHARDED_STATE_DICT` 来保存中间检查点，
因为在排名 0 上保存完整状态字典需要很长时间，通常会导致 `NCCL Timeout` 错误，因为在广播过程中会无限期挂起。
您可以使用 [`~accelerate.Accelerator.load_state`] 方法加载分片状态字典以恢复训练。

```py
# 包含检查点的目录
accelerator.load_state("ckpt")
```

然而，当训练结束时，您希望保存完整状态字典，因为分片状态字典仅与 FSDP 兼容。

```py
if trainer.is_fsdp_enabled:
    trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

trainer.save_model(script_args.output_dir)
```

### TPU

[PyTorch XLA](https://pytorch.org/xla/release/2.1/index.html) 支持用于 TPUs 的 FSDP 训练，
可以通过修改由 `accelerate config` 生成的 FSDP 配置文件来启用。除了上面指定的分片策略和包装选项外，
您还可以将以下参数添加到文件中。

```yaml
xla: True # 必须设置为 True 以启用 PyTorch/XLA
xla_fsdp_settings: # XLA 特定的 FSDP 参数
xla_fsdp_grad_ckpt: True # 使用梯度检查点
```

[`xla_fsdp_settings`](https://github.com/pytorch/xla/blob/2e6e183e0724818f137c8135b34ef273dea33318/torch_xla/distributed/fsdp/xla_fully_sharded_data_parallel.py#L128)
允许您配置用于 FSDP 的额外 XLA 特定参数。

## 启动训练

FSDP 配置文件示例如下所示：

```yaml
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: FSDP
downcast_bf16: "no"
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

要启动训练，请运行 [`accelerate launch`](https://huggingface.co/docs/accelerate/package_reference/cli#accelerate-launch)
命令，它将自动使用您之前使用 `accelerate config` 创建的配置文件。

```bash
accelerate launch my-trainer-script.py
```

```bash
accelerate launch --fsdp="full shard" --fsdp_config="path/to/fsdp_config/ my-trainer-script.py
```

## 下一步

FSDP 在大规模模型训练方面是一个强大的工具，您可以使用多个 GPU 或 TPU。
通过分片模型参数、优化器和梯度状态，甚至在它们不活动时将其卸载到 CPU 上，
FSDP 可以减少大规模训练的高成本。如果您希望了解更多信息，下面的内容可能会有所帮助：

- 深入参考 Accelerate 指南，了解有关
  [FSDP](https://huggingface.co/docs/accelerate/usage_guides/fsdp)的更多信息。
- 阅读[介绍 PyTorch 完全分片数据并行（FSDP）API](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/) 博文。
- 阅读[使用 FSDP 在云 TPU 上扩展 PyTorch 模型](https://pytorch.org/blog/scaling-pytorch-models-on-cloud-tpus-with-fsdp/)博文。
