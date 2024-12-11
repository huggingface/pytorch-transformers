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

# DeepSpeed

[DeepSpeed](https://www.deepspeed.ai/) is designed to optimize distributed training for large models with data, model, pipeline, and even a combination of all three [parallelism](./perf_train_gpu_many) strategies to provide better memory efficiency and faster training speeds. This is achieved by the [Zero Redundancy Optimizer (ZeRO)](https://hf.co/papers/1910.02054) which consists of three stages.

| ZeRO stage | description |
|---|---|
| 1 | partition optimizer states |
| 2 | partition optimizer and gradient states |
| 3 | partition optimizer, gradient, and parameters |

Each stage progressively saves more memory, allowing really large models to fit and be trained on a single GPU. DeepSpeed is integrated with [`Trainer`] for all ZeRO stages and offloading optimizer memory and computations from the GPU to the CPU. Provide a config file or one of the example templates to [`Trainer`] to enable DeepSpeed features.

This guide walks you through setting up a DeepSpeed config file, how to enable its features in [`Trainer`], and deploy training.

Install DeepSpeed from either PyPI or Transformers. For more detailed installation instructions, refer to the DeepSpeed [installation](https://www.deepspeed.ai/tutorials/advanced-install/) or GitHUB [README](https://github.com/microsoft/deepspeed#installation).

<hfoptions id="installation">
<hfoption id="PyPI">

```bash
pip install deepspeed
```

</hfoption>
<hfoption id="Transformers">

```bash
pip install transformers[deepspeed]
```

</hfoption>
</hfoptions>

> [!WARNING]
> Refer to the [DeepSpeed CUDA installation](./debugging#deepspeed-cuda-issues) if you're having trouble with your installation. While DeepSpeed has a pip installable package, it is highly recommended to [install it from source](https://www.deepspeed.ai/tutorials/advanced-install/#install-deepspeed-from-source) to ensure it matches your hardware and to support certain features which aren't available in the PyPI distribution.

DeepSpeed provides a tool for estimating the required CPU and GPU memory for the parameters, optimizer and gradient states. You'll also need some memory for the CUDA kernels and activations.

Run the command below to check the memory requirements for [bigscience/T0_3B](https://huggingface.co/docs/transformers/main/en/bigscience/T0_3B) on a single GPU.

```bash
$ python -c 'from transformers import AutoModel; \
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live; \
model = AutoModel.from_pretrained("bigscience/T0_3B"); \
estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=1, num_nodes=1)'
[...]
Estimated memory needed for params, optim states and gradients for a:
HW: Setup with 1 node, 1 GPU per node.
SW: Model with 2783M total params, 65M largest layer params.
  per CPU  |  per GPU |   Options
   70.00GB |   0.25GB | offload_param=cpu , offload_optimizer=cpu , zero_init=1
   70.00GB |   0.25GB | offload_param=cpu , offload_optimizer=cpu , zero_init=0
   62.23GB |   5.43GB | offload_param=none, offload_optimizer=cpu , zero_init=1
   62.23GB |   5.43GB | offload_param=none, offload_optimizer=cpu , zero_init=0
    0.37GB |  46.91GB | offload_param=none, offload_optimizer=none, zero_init=1
   15.56GB |  46.91GB | offload_param=none, offload_optimizer=none, zero_init=0
```

> [!TIP]
> If you have enough GPU memory, disable CPU and NVMe offload to speed everything up.

## Choose a ZeRO stage

Consider the table below to help you choose the appropriate ZeRO stage for training because there is a trade-off between training speed and memory usage. The table orders the ZeRO stages from fastest to slowest and from least memory usage to most.

| fastest | memory usage |
|---|---|
| ZeRO-1 | ZeRO-3 + offload |
| ZeRO-2 | ZeRO-3 |
| ZeRO-2 + offload | ZeRO-2 + offload |
| ZeRO-3 | ZeRO-2 |
| ZeRO-3 + offload | ZeRO-1 |

Decide the type of performance you're optimizing for, speed or memory, and then work backwards to discover the best ZeRO stage for your use case. For example, if you're optimizing for speed, start with the fastest ZeRO stage and if you run out of memory, try the next stage which is slower but more memory efficient.

## Config file

Enable DeepSpeed in [`Trainer`] with a config file containing all the parameters for how to configure and setup your training. When the training script is executed, DeepSpeed logs the configuration from [`Trainer`] to the console so you can see exactly what's being used.

> [!TIP]
> Find a complete list of DeepSpeed configuration options on the [DeepSpeed Configuration JSON](https://www.deepspeed.ai/docs/config-json/) reference. There are also practical examples of various DeepSpeed configuration examples in the [DeepSpeedExamples](https://github.com/microsoft/DeepSpeedExamples) main [DeepSpeed](https://github.com/microsoft/DeepSpeed) repository. Run the command below to quickly find specific examples.
>
> ```bash
> git clone https://github.com/microsoft/DeepSpeedExamples
> cd DeepSpeedExamples
> find . -name '*json'
> # find examples with the Lamb optimizer
> grep -i Lamb $(find . -name '*json')
> ```

The config file is passed as a path to a JSON file if you're training from the command line interface or as a nested dict object if you're using [`Trainer`] in a notebook.

<hfoptions id="pass-config">
<hfoption id="path to file">

```py
TrainingArguments(
    deepspeed="path/to/deepspeed_config.json",
    ...,
)
```

</hfoption>
<hfoption id="nested dict">

```py
ds_config_dict = dict(scheduler=scheduler_params, optimizer=optimizer_params)
args = TrainingArguments(
    deepspeed=ds_config_dict,
    ...,
)
trainer = Trainer(
    model,
    args,
    ...,
)
```

</hfoption>
</hfoptions>

### DeepSpeed versus Trainer parameters

There are three types of config parameters.

1. Some config parameters are shared by DeepSpeed and [`Trainer`] making it difficult to identify errors when there are conflicting definitions. In this case, configure these parameters from the [`Trainer`] command line arguments.
1. Some config parameters are automatically derived from the model configuration and don't need to be manually configured. [`Trainer`] uses the config value `auto` to set the most correct or efficient option. You could define these parameters explicitly, but you must take care to ensure the [`Trainer`] and DeepSpeed config parameters match. Mismatches may cause training to fail in very difficult to detect ways.
1. Some config parameters are specific to DeepSpeed and should be manually set based on your training requirements.

There are two ways to modify the config parameters.

> [!TIP]
> Some values, such as `scheduler.params.total_num_steps`, are calculated by the [`Trainer`] during training.

1. Create or load a DeepSpeed config to use as the main config.
1. Create a [`TrainingArguments`] object based on the DeepSpeed config values.

### ZeRO stage

Each ZeRO stage has its own config, as defined in `zero_optimization`.

For a more detailed explanation of each parameter, refer to the [DeepSpeed Configuration JSON](https://www.deepspeed.ai/docs/config-json/) reference. These parameters must be setup with DeepSpeed because [`Trainer`] doesn't provide equivalent command line arguments.

> [!WARNING]
> DeepSpeed doesn't validate parameter names and any typos will fallback on the parameters default setting. Observe the DeepSpeed engine startup log messages to see what values are being used.

<hfoptions id="zero-config">
<hfoption id="ZeRO-1">

ZeRO-1 shards the optimizer states across GPUs and you can expect a small speed up.

```yml
{
    "zero_optimization": {
        "stage": 1
    }
}
```

</hfoption>
<hfoption id="ZeRO-2">

ZeRO-2 shards the optimizer and gradient states across GPUs. This stage is primarily used for training since its features are not relevant to inference. Some important parameters to configure for better performance include the following.

* `offload_optimizer` should be enabled to reduce GPU memory usage.
* `overlap_comm` when set to `true` uses increased GPU memory usage in exchange for lower allreduce latency. This feature uses 4.5x the `allgather_bucket_size` and `reduce_bucket_size` values. In this example, they're set to `5e8` which means it requires 9GB of GPU memory. If your GPU memory is 8GB or less, you should reduce `overlap_comm` to lower the memory requirements and prevent an out-of-memory (OOM) error.
* `allgather_bucket_size` and `reduce_bucket_size` trade-off available GPU memory for communication speed. The smaller their values, the slower communication is and the more GPU memory is available. You can balance, for example, whether a bigger batch size is more important than a slightly slower training time.
* `round_robin_gradients` is available in DeepSpeed 0.4.4 for CPU offloading. It parallelizes gradient copying to CPU memory among ranks by fine-grained gradient partitioning. Performance benefit grows with gradient accumulation steps (more copying between optimizer steps) or GPU count (increased parallelism).

```yml
{
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "allgather_partitions": true,
        "allgather_bucket_size": 5e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 5e8,
        "contiguous_gradients": true
        "round_robin_gradients": true
    }
}
```

</hfoption>
<hfoption id="ZeRO-3">

ZeRO-3 shards the optimizer and gradient states, and parameters across GPUs. Unlike ZeRO-2, ZeRO-3 can also be used for inference in addition to training because it loads large models onto multiple GPUs. Some important parameters to configure include the following.

* `device: "cpu"` can help if you're running out of GPU memory and if you have free CPU memory available. This offloads model parameters to the CPU.
* `pin_memory: true` can improve throughput, but less memory becomes available for other processes because the pinned memory is reserved for the specific process that requested it and it's typically accessed much faster than normal CPU memory.
* `stage3_max_live_parameters` is the upper limit on how many full parameters to keep on the GPU at any given time. Reduce this value if you encounter an OOM error.
* `stage3_max_reuse_distance` is a value for determining when a parameter is used again in the future, and it helps decide whether to throw the parameter away or to keep it. If the parameter is going to be reused (if the value is less than `stage3_max_reuse_distance`), then it is kept to reduce communication overhead. This is helpful when activation checkpointing is enabled and you want to keep the parameter in the forward recompute until the backward pass. But reduce this value if you encounter an OOM error.
* `stage3_gather_16bit_weights_on_model_save` consolidates fp16 weights when a model is saved. For large models and multiple GPUs, this is expensive in terms of memory and speed. You should enable it if you're planning on resuming training.
* `sub_group_size` controls which parameters are updated during the optimizer step. Parameters are grouped into buckets of `sub_group_size` and each bucket is updated one at a time. When used with NVMe offload, `sub_group_size` determines when model states are moved in and out of CPU memory during the optimization step. This prevents running out of CPU memory for extremely large models. `sub_group_size` can be left to its default value if you aren't using NVMe offload, but you may want to change it if you:

    1. Run into an OOM error during the optimization step. In this case, reduce `sub_group_size` to reduce memory usage of the temporary buffers.
    2. The optimization step is taking a really long time. In this case, increase `sub_group_size` to improve bandwidth utilization as a result of increased data buffers.

* `reduce_bucket_size`, `stage3_prefetch_bucket_size`, and `stage3_param_persistence_threshold` are dependent on a models hidden size. It is recommended to set these values to `auto` and allow the [`Trainer`] to automatically assign the values.

```yml
{
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true
    }
}
```

#### Initialize large models

With ZeRO-3, use the [deepspeed.zero.Init](https://deepspeed.readthedocs.io/en/latest/zero3.html#deepspeed.zero.Init) context manager to initialize a model faster.

```py
from transformers import T5ForConditionalGeneration, T5Config
import deepspeed

with deepspeed.zero.Init():
    config = T5Config.from_pretrained("google-t5/t5-small")
    model = T5ForConditionalGeneration(config)
```

The DeepSped config file needs to have `is_deepspeed_zero3_enabled: true` setup in [`TrainingArguments`] and it needs a ZeRO configuration enabled. The [`TrainingArguments`] object must be created **before** calling [`~PreTrainedModel.from_pretrained`].

> [!TIP]
> You'll need ZeRO-3 when the fp16 weights don't fit on a single GPU. But if you're able to load the fp16 weights, set `torch_dtype=torch.float16` in [`~PreTrainedModel.from_pretrained`].

```py
from transformers import AutoModel, Trainer, TrainingArguments

training_args = TrainingArguments(..., deepspeed=ds_config)
model = AutoModel.from_pretrained("google-t5/t5-small")
trainer = Trainer(model=model, args=training_args, ...)
```

When there are multiple GPUs, no single GPU has all the parameters unless it's the parameters of the currently executing layer. To access all parameters from all the layers at once, such as loading pretrained model weights in [`~PreTrainedModel.from_pretrained`], one layer is loaded at a time and immediately partitioned to all GPUs. For very large models, it isn't possible to load the weights onto one GPU and then distribute them across the other GPUs due to memory limitations.

If you encounter a model parameter weight where `tensor([1.])` or the parameter size is 1 instead of a larger multidimensional shape, it means the parameter is partitioned and this is a ZeRO-3 placeholder.

```py
tensor([1.0], device="cuda:0", dtype=torch.float16, requires_grad=True)
```

[!TIP]
> For more information about initializing large models with ZeRO-3 and accessing the parameters, take a look at the [Constructing Massive Models](https://deepspeed.readthedocs.io/en/latest/zero3.html#constructing-massive-models) and [Gathering Parameters](https://deepspeed.readthedocs.io/en/latest/zero3.html#gathering-parameters) guides.

</hfoption>
</hfoptions>

### NVMe

[ZeRO-Infinity](https://hf.co/papers/2104.07857) offloads model states to the CPU and/or NVMe to save even more memory. Smart partitioning and tiling algorithms allow each GPU to send and receive very small amounts of data during offloading such that a modern NVMe can fit an even larger total memory pool than is available to your training process. ZeRO-Infinity requires ZeRO-3.

Depending on the CPU and NVMe memory available, you can offload both the [optimizer states](https://www.deepspeed.ai/docs/config-json/#optimizer-offloading) and [parameters](https://www.deepspeed.ai/docs/config-json/#parameter-offloading), just one of them, or none of them. Make sure the `nvme_path` points to a NVMe device, because while it still works with a regular hard drive or solid state drive, it'll be significantly slower. With a modern NVMe, you can expect peak transfer speeds of ~3.5GB/s for read operations and ~3GB/s for write operations.

Consider running a [benchmark](https://github.com/microsoft/DeepSpeed/issues/998) on your training setup to determine the optimal `aio` configuration.

The example ZeRO-3 and ZeRO-Infinity config below sets most of the parameter values to `auto`, but you can also manually set configure these values.

```yaml
{
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },

    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto"
        }
    },

    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto"
        }
    },

    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "nvme",
            "nvme_path": "/local_nvme",
            "pin_memory": true,
            "buffer_count": 4,
            "fast_init": false
        },
        "offload_param": {
            "device": "nvme",
            "nvme_path": "/local_nvme",
            "pin_memory": true,
            "buffer_count": 5,
            "buffer_size": 1e8,
            "max_in_cpu": 1e9
        },
        "aio": {
            "block_size": 262144,
            "queue_depth": 32,
            "thread_count": 1,
            "single_submit": false,
            "overlap_events": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true
    },

    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 2000,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false
}
```

## Training features

DeepSpeed supports many training features that can be configured in the config file. This section describes some of the most important features.

### Activation and gradient checkpointing

### Batch size

### Communication data type

### Gradient accumulation

### Gradient clipping

### Mixed precision training

### Optimizer and scheduler

## Deploy

### Multi-node

### SLURM

### Notebook

## Save model weights

### fp16

### fp32
