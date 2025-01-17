<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Build your own machine

One of the most important consideration when building a machine for deep learning is the GPU choice. GPUs are the standard workhorse for deep learning owing to their tensor cores for performing very efficient matrix multiplication and high memory bandwidth. To train large models, you either need a more powerful GPU, multiple GPUs, or take advantage of techniques that offload some of the load to the CPU or NVMe.

This guide provides some practical tips for setting up a GPU for deep learning. For a more detailed discussion and comparison of GPUs, take a look at the [Which GPU(s) to Get for Deep Learning](https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/) blog post.

## Power

High-end consumer GPUs may have two or three PCIe 8-pin power sockets, and you should make sure you have the same number of 12V PCIe 8-pin cables connected to each socket. Don't use a *pigtail cable*, a single cable with two splits at one end, to connect two sockets or else you won't get full performance from your GPU.

Each PCIe 8-pin power cable should be connected to a 12V rail on the power supply unit (PSU) and can deliver up to 150W. Other GPUs may use a PCIe 12-pin connector which can deliver up to 500-600W. Lower-end GPUs may only use a PCIe 6-pin connector which supplies up to 75W.

It is important the PSU has stable voltage otherwise it may not be able to supply the GPU with enough power to function properly during peak usage.

## Cooling

An overheated GPU throttles its performance and can even shutdown if it's too hot to prevent damage. Keeping the GPU temperature low, anywhere between 158 - 167F, is essential for delivering full perfomance and maintaining its lifespan. Once temperatures reach 183 - 194F, the GPU may begin to throttle performance.

## Multi-GPU connectivity

When your setup uses multiple GPUs, it is important to consider how they're connected. [NVLink](https://www.nvidia.com/en-us/design-visualization/nvlink-bridges/) connections are faster than PCIe bridges, but you should also consider the [parallelism](./perf_train_gpu_many) strategy you're using. For example, in DistributedDataParallel, GPUs communicate less frequently compared to ZeRO-DP. In this case, a slower connection is not as important.

Run the command below to check how your GPUs are connected.

```bash
nvidia-smi topo -m
```

<hfoptions id="nvlink">
<hfoption id="NVLink">

[NVLink](https://www.nvidia.com/en-us/design-visualization/nvlink-bridges/) is a high-speed communication system designed by NVIDIA for connecting multiple NVIDIA GPUs. Training [openai-community/gpt2](https://huggingface.co/openai-community/gpt2) on a small sample of the [wikitext](https://huggingface.co/datasets/Salesforce/wikitext) dataset is ~23% faster with NVLink.

On a machine with two GPUs connected with NVLink, an example output of `nvidia-smi topo -m` is shown below.

```bash
        GPU0    GPU1    CPU Affinity    NUMA Affinity
GPU0     X      NV2     0-23            N/A
GPU1    NV2      X      0-23            N/A
```

`NV2` indicates `GPU0` and `GPU1` are connected by 2 NVLinks.

</hfoption>
<hfoption id="without NVLink">

On a machine with two GPUs connected with a PCIe bridge, an example output of `nvidia-smi topo -m` is shown below.

```bash
        GPU0    GPU1    CPU Affinity    NUMA Affinity
GPU0     X      PHB     0-11            N/A
GPU1    PHB      X      0-11            N/A
```

`PHB` indicates `GPU0` and `GPU1` are connected by a PCIe bridge.

</hfoption>
</hfoptions>
