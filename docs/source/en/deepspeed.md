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

### Gradient checkpointing

Gradient checkpointing saves memory by only storing some of the intermediate activations instead of storing *all* of them. It is useful for fitting larger models on the GPU without running out of memory or to increase the batch size for better performance. Training speed is slower though.

* For a Hugging Face model, set `model.gradient_checkpointing_enable()` or add `--gradient_checkpointing` in the [`TrainingArguments`].
* For a non-Hugging Face model, use the DeepSpeed [Activation Checkpointing API](https://deepspeed.readthedocs.io/en/latest/activation-checkpointing.html). Replacing Transformers modeling code and [torch.utils.checkpoint](https://pytorch.org/docs/stable/checkpoint.html) with the DeepSpeed API gives you more flexibility because you can offload the forward activations to the CPU memory instead of recalculating them.

### Batch size

The batch size can be automatically configured or manually set. When you choose the `"auto"` option, [`Trainer`] sets `train_micro_batch_size_per_gpu` and `train_batch_size` to the value of `world_size * per_device_train_batch_size * gradient_accumulation_steps`.

```yaml
{
    "train_micro_batch_size_per_gpu": "auto",
    "train_batch_size": "auto"
}
```

### Communication data type

A separate data type is used for communication collectives like reduction, gathering and scattering operations.

All gather and scatter operations are performed in the same data type the data is in. For example, if you're training in bf16, the data is also gathered in bf16 because gathering is a non-lossy operation.

Reduce operations are lossy, for example, when gradients are averaged across multiple GPUs. When the communication is done if fp16 or bf16, it's more likely to be lossy because adding multiple numbers in low precision isn't exact. This is especially the case with bf16 which has a lower precision than fp16. For this reason, fp16 is the default for reduction operations because the loss is minimal when averaging gradients.

Choose the communication data type by setting the `communication_data_type` parameter in the config file. For example, choosing fp32 adds a small amount of overhead but ensures the reduction operation is accumulated in fp32 and when it is ready, it's downcasted to whichever half-precision data type you're training in.

```yaml
{
    "communication_data_type": "fp32"
}
```

### Gradient accumulation

Gradient accumulation accumulates gradients over several mini-batches of data before updating parameters. It stores less gradients and enables training with a larger *effective batch size*. Training speed is slower though, but it's useful for overcoming memory constraints.

Gradient accumulation can be automatically configured or manually set. When you choose the `"auto"` option, [`Trainer`] sets it to the value of `gradient_accumulation_steps`.

```yaml
{
    "gradient_accumulation_steps": "auto"
}
```

### Gradient clipping

Gradient clipping is useful for preventing exploding gradients which can lead to instability during training. It sets a maximum threshold value and rescales the gradients if their norm exceeds the threshold.

Gradient clipping can be automatically configured or manually set. When you choose the `"auto"` option, [`Trainer`] sets it to the value of `max_grad_norm`.

```yaml
{
    "gradient_clipping": "auto"
}
```

### Mixed precision training

Mixed precision accelerates training speed by performing some calculations in half-precision, but it also maintains some calculations in full-precision to preserve accuracy. DeepSpeed supports fp32, fp16, and bf16 data types.

<hfoptions id="precision">
<hfoption id="fp32">

Train in fp32 if a model wasn't pretrained in mixed precision because it may cause underflow or overflow errors. Disable fp16, the default, in this case.

```yaml
{
    "fp16": {
        "enabled": false
    }
}
```

For Ampere GPUs and PyTorch 1.7+, the more efficient [tf32](https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices) mode is automatically enabled for some operations but the results are still in fp32. Configure it from the [`Trainer`] by setting `--tf32` to enable it, and `--tf32 0` or `--no_tf32` to disable it.

</hfoption>
<hfoption id="fp16">

To configure AMP-like fp16 mixed precision, set up the config as shown below with `"auto"` or your own values. [`Trainer`] automatically enables or disables fp16 based on the value of `fp16_backend`, and the rest of the config can be set by you. fp16 is enabled from the command line when the following arguments are passed: `--fp16`, `--fp16_backend amp` or `--fp16_full_eval`.

```yaml
{
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    }
}
```

For additional DeepSpeed fp16 training options, take a look at the [FP16 Training Options](https://www.deepspeed.ai/docs/config-json/#fp16-training-options) reference.

To configure Apex-like fp16 mixed precision, set up the config as shown below with `"auto"` or your own values. [`Trainer`] automatically configures `amp` based on the values of `fp16_backend` and `fp16_opt_level`. It can also be enabled from the command line when the following arguments are passed: `--fp16`, `--fp16_backend apex` or `--fp16_opt_level 01`.

```yaml
{
    "amp": {
        "enabled": "auto",
        "opt_level": "auto"
    }
}
```

</hfoption>
<hfoption id="bf16">

> [!TIP]
> bf16 requires DeepSpeed 0.6.0.

bf16 has the same dynamic range as fp32, and doesn’t require loss scaling unlike fp16. However, if you use [gradient accumulation](#gradient-accumulation) with bf16, gradients are accumulated in bf16 which may not be desirable because bf16s low precision can lead to lossy accumulation.

bf16 can be set up in the config file or enabled from the command line when the following arguments are passed: `--bf16` or `--bf16_full_eval`.

```yaml
{
    "bf16": {
        "enabled": "auto"
    }
}
```

</hfoption>
</hfoptions>

### Optimizer and scheduler

DeepSpeed and Transformers optimizers and schedulers can be mixed and matched if `offload_optimizer` isn't enabled. When `offload_optimizer` is enabled, use a non-DeepSpeed optimizer (except for LAMB) as long as it has it a CPU and GPU implementation.

Set the optimizer and scheduler parameters for the config file from the command line to avoid hard to find errors. For example, if the learning rate is set to a different value in another place, you can override it from the command line.

<hfoptions id="opt-sched">
<hfoption id="optimizer">

DeepSpeed offers several [optimizers](https://www.deepspeed.ai/docs/config-json/#optimizer-parameters) (Adam, AdamW, OneBitAdam, and LAMB) but you can also import other optimizers from PyTorch. If you don't configure the optimizer in the config, [`Trainer`] automatically selects AdamW and either uses the supplied values or the default values for the following parameters from the command line: `lr`, `adam_beta1`, `adam_beta2`, `adam_epsilon`, `weight_decay`.

You can set the parameters to `"auto"` or manually input your own values.

```yaml
{
   "optimizer": {
       "type": "AdamW",
       "params": {
         "lr": "auto",
         "betas": "auto",
         "eps": "auto",
         "weight_decay": "auto"
       }
   }
}
```

Use an unsupported optimizer by adding the following to the top level configuration.

```yaml
{
   "zero_allow_untested_optimizer": true
}
```

From DeepSpeed 0.8.3+, if you want to use offload, you'll also need to add the following to the top level configuration because offload works best with DeepSpeed's CPU Adam optimizer.

```yaml
{
   "zero_force_ds_cpu_optimizer": false
}
```

</hfoption>
<hfoption id="scheduler">

DeepSpeed supports the LRRangeTest, OneCycle, WarmupLR and WarmupDecayLR learning rate [schedulers](https://www.deepspeed.ai/docs/config-json/#scheduler-parameters).

Transformers and DeepSpeed provide two of the same schedulers:

* WarmupLR is the same as `--lr_scheduler_type constant_with_warmup` in Transformers.
* WarmupDecayLR is the same as  `--lr_scheduler_type linear` in Transformers (this is the default scheduler used in Transformers).

If you don't configure the scheduler in the config file, [`Trainer`] automatically selects WarmupDecayLR and either uses the supplied values or the default values for the following parameters from the command line: `warmup_min_lr`, `warmup_max_lr`, `warmup_num_steps`, `total_num_steps` (automatically calculated during run time if `max_steps` is not provided).

You can set the parameters to `"auto"` or manually input your own values.

```yaml
{
   "scheduler": {
         "type": "WarmupDecayLR",
         "params": {
             "total_num_steps": "auto",
             "warmup_min_lr": "auto",
             "warmup_max_lr": "auto",
             "warmup_num_steps": "auto"
         }
     }
}
```

</hfoption>
</hfoptions>

## Deploy

DeepSpeed can be deployed with its native launcher, [torchrun](https://pytorch.org/docs/stable/elastic/run.html) or [Accelerate](https://huggingface.co/docs/accelerate/basic_tutorials/launch#using-accelerate-launch).

Add the `--deepspeed ds_config.json` argument to the [`Trainer`] command line. It is recommended to use DeepSpeeds [add_config_arguments](https://deepspeed.readthedocs.io/en/latest/initialize.html#argument-parsing) utility to add any other command line arguments to your code.

<hfoptions id="deploy">
<hfoption id="multi-GPU">

To deploy DeepSpeed on multiple GPUs, add `--num_gpus`. You don't need to add `--num_gpus` if you're planning on using all available GPUs.

```bash
deepspeed --num_gpus=2 examples/pytorch/translation/run_translation.py \
--deepspeed tests/deepspeed/ds_config_zero3.json \
--model_name_or_path google-t5/t5-small --per_device_train_batch_size 1 \
--output_dir output_dir --overwrite_output_dir --fp16 \
--do_train --max_train_samples 500 --num_train_epochs 1 \
--dataset_name wmt16 --dataset_config "ro-en" \
--source_lang en --target_lang ro
```

</hfoption>
<hfoption id="single-GPU">

DeepSpeed is still useful with just one GPU because you can:

1. Offload some computations and memory to the CPU to make more GPU resources available to your model to use a larger batch size or fit a very large model that normally won't fit.
2. Minimize memory fragmentation with its smart GPU memory management system which also allows you to fit bigger models and data batches.

To deploy DeepSpeed on a single GPU, add `--num_gpus`. You don't need to add `--num_gpus` if you only have one GPU because DeepSpeed deploys all GPUs it can see on a given node.

> [!TIP]
> Set the `allgather_bucket_size` and `reduce_bucket_size` values to 2e8 in the [ZeRO-2](#zero-configuration) configuration file to get better performance on a single GPU.

```bash
deepspeed --num_gpus=1 examples/pytorch/translation/run_translation.py \
--deepspeed tests/deepspeed/ds_config_zero2.json \
--model_name_or_path google-t5/t5-small --per_device_train_batch_size 1 \
--output_dir output_dir --overwrite_output_dir --fp16 \
--do_train --max_train_samples 500 --num_train_epochs 1 \
--dataset_name wmt16 --dataset_config "ro-en" \
--source_lang en --target_lang ro
```

</hfoption>
</hfoptions>

### Multi-node

A multi-node setup consists of multiple nodes, where each node has one of more GPUs running a workload. DeepSpeed expects a shared storage system, but if this is not the case, you need to adjust the config file to include a [checkpoint](https://www.deepspeed.ai/docs/config-json/#checkpoint-options) to allow loading without access to a shared filesystem.

```yaml
{
  "checkpoint": {
    "use_node_local_storage": true
  }
}
```

You could also use the `--save_on_each_node` parameter in [`TrainingArguments`] to automatically add the above `checkpoint` to your config.

The examples below for the torchrun and DeepSpeed launcher shows how to deploy two nodes with eight GPUs each. Access the first node with `ssh hotname1` and the second node with `ssh hostname2`. Both nodes must be able to communicate with each other locally over ssh without a password.

<hfoptions id="multinode">
<hfoption id="torchrun">

With [torchrun](https://pytorch.org/docs/stable/elastic/run.html), ssh to each node and run the following command on both of them. The launcher waits until both nodes are synchronized before launching the training.

```bash
torchrun --nproc_per_node=8 --nnode=2 --node_rank=0 --master_addr=hostname1 \
--master_port=9901 your_program.py <normal cl args> --deepspeed ds_config.json
```

</hfoption>
<hfoption id="DeepSpeed">

Create a `hostfile` for the DeepSpeed launcher.

```bash
hostname1 slots=8
hostname2 slots=8
```

The DeepSpeed launcher automatically launches the command on both nodes at once with the command below.

```bash
deepspeed --num_gpus 8 --num_nodes 2 --hostfile hostfile --master_addr hostname1 --master_port=9901 \
your_program.py <normal cl args> --deepspeed ds_config.json
```

Check out the [Resource Configuration (multi-node)](https://www.deepspeed.ai/getting-started/#resource-configuration-multi-node) guide for more details about configuring multi-node compute resources.

</hfoption>
</hfoptions>

### Slurm

[Slurm](https://slurm.schedmd.com/documentation.html) is a cluster management and job scheduling system. An example Slurm script is shown below.

```bash
#SBATCH --job-name=test-nodes        # name
#SBATCH --nodes=2                    # nodes
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --gres=gpu:8                 # number of gpus
#SBATCH --time 20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name

export GPUS_PER_NODE=8
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=9901

srun --jobid $SLURM_JOBID bash -c 'python -m torch.distributed.run \
 --nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID \
 --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
your_program.py <normal cl args> --deepspeed ds_config.json'
```

Launch training simultaneously on all nodes with the command below.

```bash
sbatch launch.slurm
```

### Jupyter Notebook

To use DeepSpeed in a Jupyter Notebook, you need to emulate a distributed environment because the launcher doesn't support deployment from a notebook. This is only supported for one GPU. To use multiple GPUs, you must use a multi-process environment, which means you have to use the DeepSpeed launcher which can't be emulated as shown here.

```py
# emulate a launcher in the notebook
import os

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "9994"  # modify if RuntimeError: Address already in use
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"

training_args = TrainingArguments(..., deepspeed="ds_config_zero3.json")
trainer = Trainer(...)
trainer.train()
```

Create a config file on the fly in the notebook in the current directory with a dedicated cell.

```py
%%bash
cat <<'EOT' > ds_config_zero3.json
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
    },

    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 2000,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false
}
EOT
```

If the training script is in a file and not a notebook cell, launch DeepSpeed from the shell in the notebook cell.

```py
!git clone https://github.com/huggingface/transformers
!cd transformers; deepspeed examples/pytorch/translation/run_translation.py ...
```

Another option is to use `%%bash` to run the shell program without emulating the distributed environment. However, you won't be able to view the logs until training is complete.

```py
%%bash

git clone https://github.com/huggingface/transformers
cd transformers
deepspeed examples/pytorch/translation/run_translation.py ...
```

## Save model weights

DeepSpeed stores the main fp32 weights in custom checkpoint optimizer files (`global_step*/*optim_states.pt`) which are saved under the normal checkpoint.

### fp16

ZeRO-2 saves the model weights in fp16. To save the weights in fp16 for ZeRO-3, set `"stage3_gather_16bit_weights_on_model_save": true` in the config file, because the weights are distributed across multiple GPUs.

If you don't, [`Trainer`] won't save the weights in fp16 and it won't create a `pytorch_model.bin` file. This is because DeepSpeed's state_dict contains a placeholder instead of the real weights, so you won't be able to load it.

```yaml
{
    "zero_optimization": {
        "stage": 3,
        "stage3_gather_16bit_weights_on_model_save": true
    }
}
```

### fp32

fp32 weights shouldn't be saved during training because it can require a lot of memory, unless you have a lot of free CPU memory. It is usually best to save the fp32 weights offline after training is complete.

<hfoptions id="save">
<hfoption id="offline">

DeepSpeed provies a [zero_to_fp32.py](https://github.com/microsoft/DeepSpeed/blob/91829476a8fd4d0d9268c03c1d56795d20a51c12/deepspeed/utils/zero_to_fp32.py#L14) script at the top-level checkpoint folder for extracting weights at any point. This is a standalone script and you don't need a config file or [`Trainer`].

For example, if your checkpoint folder looks like the one shown below, then you can run the following command to create and consolidate the fp32 weights from multiple GPUs into a single `pytorch_model.bin` file. The script automatically discovers the subfolder `global_step1` which contains the checkpoint.

```bash
$ ls -l output_dir/checkpoint-1/
-rw-rw-r-- 1 stas stas 1.4K Mar 27 20:42 config.json
drwxrwxr-x 2 stas stas 4.0K Mar 25 19:52 global_step1/
-rw-rw-r-- 1 stas stas   12 Mar 27 13:16 latest
-rw-rw-r-- 1 stas stas 827K Mar 27 20:42 optimizer.pt
-rw-rw-r-- 1 stas stas 231M Mar 27 20:42 pytorch_model.bin
-rw-rw-r-- 1 stas stas  623 Mar 27 20:42 scheduler.pt
-rw-rw-r-- 1 stas stas 1.8K Mar 27 20:42 special_tokens_map.json
-rw-rw-r-- 1 stas stas 774K Mar 27 20:42 spiece.model
-rw-rw-r-- 1 stas stas 1.9K Mar 27 20:42 tokenizer_config.json
-rw-rw-r-- 1 stas stas  339 Mar 27 20:42 trainer_state.json
-rw-rw-r-- 1 stas stas 2.3K Mar 27 20:42 training_args.bin
-rwxrw-r-- 1 stas stas 5.5K Mar 27 13:16 zero_to_fp32.py*
```

> [!TIP]
> Run `python zero_to_fp32.py -h` for more usage details. The script requires 2x the general RAM of the final fp32 weights.

```bash
python zero_to_fp32.py . pytorch_model.bin
```

</hfoption>
<hfoption id="online">

Adding the `--load_best_model_at_end` parameter in [`TrainingArguments`] tracks the best checkpoint so you can finish training first and save the final model explicitly. Reload the model as shown below.

> [!WARNING]
> Once [load_state_dict_from_zero_checkpoint](https://deepspeed.readthedocs.io/en/stable/model-checkpointing.html#deepspeed.utils.zero_to_fp32.load_state_dict_from_zero_checkpoint) is run, the model is no longer usable in DeepSpeed in the context of the same application. You'll need to reinitialize the DeepSpeed engine because `model.load_state_dict(state_dict)` removes all the DeepSpeed magic from it. Only use this function once training is complete.

```py
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint

checkpoint_dir = os.path.join(trainer.args.output_dir, "checkpoint-final")
trainer.deepspeed.save_checkpoint(checkpoint_dir)
fp32_model = load_state_dict_from_zero_checkpoint(trainer.model, checkpoint_dir)
```

You must have saved at least one checkpoint to load the latest checkpoint as shown in the example below.

```py
from transformers.trainer_utils import get_last_checkpoint
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint

checkpoint_dir = get_last_checkpoint(trainer.args.output_dir)
fp32_model = load_state_dict_from_zero_checkpoint(trainer.model, checkpoint_dir)
```

Use `load_state_dict` to extract and load the state_dict of the fp32 weights.

```py
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir)
model = model.cpu()
model.load_state_dict(state_dict)
```

</hfoption>
</hfoptions>

## Non-Trainer integration

DeepSpeed also works with Transformers without [`Trainer`]. The [`~integrations.HfDeepSpeedConfig`] is responsible for gathering ZeRO-3 parameters and partitioning a model across multiple GPUs when [`~PreTrainedModel.from_pretrained`] is called.

You must instantiate [`HfDeepSpeedConfig`] before loading a model to efficiently deploy ZeRO-3.

<hfoptions id="models">
<hfoption id="pretrained model">

```py
from transformers.integrations import HfDeepSpeedConfig
from transformers import AutoModel
import deepspeed

# DeepSpeed config object or path to the file
ds_config = {...}
# must run before instantiating the model to detect ZeRO-3
dschf = HfDeepSpeedConfig(ds_config)  # keep this object alive
model = AutoModel.from_pretrained("openai-community/gpt2")
engine = deepspeed.initialize(model=model, config_params=ds_config, ...)
```

</hfoption>
<hfoption id="non-pretrained model">

[`HfDeepSpeedConfig`] is not required for ZeRO-1 or ZeRO-2.

```py
from transformers.integrations import HfDeepSpeedConfig
from transformers import AutoModel, AutoConfig
import deepspeed

# DeepSpeed config object or path to the file
ds_config = {...}
# must run before instantiating the model to detect zero 3
dschf = HfDeepSpeedConfig(ds_config)  # keep this object alive
# randomly intialize model weights
config = AutoConfig.from_pretrained("openai-community/gpt2")
model = AutoModel.from_config(config)
engine = deepspeed.initialize(model=model, config_params=ds_config, ...)
```

</hfoption>
</hfoptions>

## Troubleshoot

One of the first things to check when you encounter an error is whether DeepSpeed is the cause because often it isn't. Retry your setup without DeepSpeed, and if the error persists, report the issue. If the issue is unrelated to the Transformers integration, please open the issue on the DeepSpeed [repository](https://github.com/microsoft/DeepSpeed).

For issues related to the Transformers integration, please provide the following information.

* The full DeepSpeed config file.
* The command line arguments for [`Trainer`] or the [`TrainingArguments`] if you're scripting the [`Trainer`] setup yourself (don't dump the entire [`TrainingArguments`] which contains many irrelevant entries).
* The outputs of the following commands.

    ```bash
    python -c 'import torch; print(f"torch: {torch.__version__}")'
    python -c 'import transformers; print(f"transformers: {transformers.__version__}")'
    python -c 'import deepspeed; print(f"deepspeed: {deepspeed.__version__}")'
    ```

* A link to a Google Colab notebook to reproduce the issue.
* A standard or non-custom dataset or an existing example to reproduce the issue.

The following sections provide a guide for resolving two of the most common issues.

### Process killed at startup

When the DeepSpeed process is killed during launch without a traceback, that usually means the program tried to allocate more CPU memory than is available on your system. Or the process may have tried to allocate more CPU memory than allowed, leading the OS kernel to termine the process.

In this case, check whether your config file has either `offload_optimizer`, `offlload_param`, or both configured to offload to the CPU.

If you have NVM3 and ZeRO-3 set up, experiment with offloading to the NVMe ([estimate](https://deepspeed.readthedocs.io/en/latest/memory.html) the memory requirements of a model first) instead.

### NaN loss

NaN loss often occurs when a model is pretrained in bf16 and you try to use it with fp16 (especially relevant to TPU trained models). To resolve this, use fp32 or bf16 if your hardware (TPUs, Ampere GPUs or newer) supports it.

It is also possible that fp16 is causing overflow. For example, if your config file looks like the one below, you may see the following overflow errors in the logs.

```yaml
{
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    }
}
```

The `OVERFLOW!` error below is a result of the DeepSpeed loss scaler unable to find a scaling coefficient to overcome the loss overflow. Try a higher `initial_scale_power` value in this case (32 usually works).

```bash
0%|                                                                                                                             | 0/189 [00:00<?, ?it/s]
 [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 262144, reducing to 262144
  1%|▌                                                                                                                    | 1/189 [00:00<01:26,  2.17it/s]
 [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 262144, reducing to 131072.0
  1%|█▏
 [...]
 [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 1, reducing to 1
 14%|████████████████▌                                                                                                   | 27/189 [00:14<01:13,  2.21it/s]
 [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 1, reducing to 1
 15%|█████████████████▏                                                                                                  | 28/189 [00:14<01:13,  2.18it/s]
 [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 1, reducing to 1
 15%|█████████████████▊                                                                                                  | 29/189 [00:15<01:13,  2.18it/s]
 [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 1, reducing to 1
[...]
```

## Resources

DeepSpeed is a powerful technology for scaling large model training. To learn more about DeepSpeed, take a look at their [blog posts](https://www.microsoft.com/en-us/research/search/?q=deepspeed), [documentation](https://www.deepspeed.ai/getting-started/), and [GitHub](https://github.com/microsoft/deepspeed).

The papers below provide additional details about ZeRO.

* [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://hf.co/papers/1910.02054)
* [ZeRO-Offload: Democratizing Billion-Scale Model Training](https://hf.co/papers/2101.06840)
* [ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning](https://hf.co/papers/2104.07857)
