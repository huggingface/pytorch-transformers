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

# Multi-GPU debugging

Distributed training can be tricky because you have to ensure you're using the correct CUDA version across your system, you may encounter inter-communication issues between GPUs, and there may be underflow or overflow problems in your model. This guide covers how to debug these issues, especially as it relates to DeepSpeed and PyTorch.

## DeepSpeed CUDA issues

DeepSpeed compiles CUDA C++ which can be a potential source of errors when building PyTorch extensions that require CUDA. These errors depend on how CUDA is installed on your system. This section focuses on PyTorch built with *CUDA 10.2*

```bash
pip install deepspeed
```

> [!TIP]
> For any other installation issues, please [open an issue](https://github.com/microsoft/DeepSpeed/issues) with the DeepSpeed team.

### Non-identical CUDA toolkits

PyTorch comes with its own CUDA toolkit, but to use DeepSpeed with PyTorch, you need to have an identical version of CUDA installed system-wide. For example, if you installed PyTorch with `cudatoolkit==10.2` in your Python environment, then you'll also need to have CUDA 10.2 installed everywhere.

The exact location can vary from system to system, but `usr/local/cuda-10.2` is the most common location on many Unix systems. When CUDA is correctly setup and added to your `PATH` environment variable, you can find the installation location with the following command.

```bash
which nvcc
```

### Multiple CUDA toolkits

You may also have more than one CUDA toolkit installed on your system.

```bash
/usr/local/cuda-10.2
/usr/local/cuda-11.0
```

Typically, package installers set the paths to whatever the last version was installed. If the package build fails because it can't find the right CUDA version (despite it being installed already), then you need to configure the `PATH` and `LD_LIBRARY_PATH` environment variables to point to the correct path.

Take a look at the contents of the following environment variables first.

```bash
echo $PATH
echo $LD_LIBRARY_PATH
```

`PATH` lists the locations of the executables and `LD_LIBRARY_PATH` lists where to look for shared libraries. Earlier entries are prioritized over later ones, and `:` is used to separate multiple entries. To find a specific CUDA toolkit, insert the correct path to list first. This command prepends rather than overwrites the existing values.

```bash
# adjust the version and full path if needed
export PATH=/usr/local/cuda-10.2/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64:$LD_LIBRARY_PATH
```

In addition, you should also check that the assigned directories you actually exist. The `lib64` sub-directory contains various CUDA `.so` objects (like `libcudart.so`), and while it is unlikely your system names them differently, you should check the actual names and change them accordingly.

### Older CUDA versions

Sometimes, older CUDA versions may refuse to build with newer compilers. For example, if you have `gcc-9` but CUDA wants `gcc-7`. Usually, installing the latest CUDA toolkit enables support for the newer compiler.

You could also install an older version of the compiler in addition to the one you're currently using (or it may already be installed but it's not used by default and the build system can't see it). To resolve this, create a symlink to give the build system visibility to the older compiler.

```bash
# adjust the path to your system
sudo ln -s /usr/bin/gcc-7  /usr/local/cuda-10.2/bin/gcc
sudo ln -s /usr/bin/g++-7  /usr/local/cuda-10.2/bin/g++
```

### Prebuild

If you're still having issues with installing DeepSpeed or if you're building DeepSpeed at run time, try to prebuild the DeepSpeed modules before installing them. Run the commands below to make a local build for DeepSpeed.

```bash
git clone https://github.com/microsoft/DeepSpeed/
cd DeepSpeed
rm -rf build
TORCH_CUDA_ARCH_LIST="8.6" DS_BUILD_CPU_ADAM=1 DS_BUILD_UTILS=1 pip install . \
--global-option="build_ext" --global-option="-j8" --no-cache -v \
--disable-pip-version-check 2>&1 | tee build.log
```

> [TIP]
> Add the `DS_BUILD_AIO=1` parameter to the build command to use NVMe offload. Make sure you install the libaio-dev package across your system.

Next, specify your GPU's architecture by editing the `TORCH_CUDA_ARCH_LIST` variable (find a complete list of NVIDIA GPUs and their corresponding architectures on this [page](https://developer.nvidia.com/cuda-gpus)). To check the PyTorch version that corresponds to your architecture, run the following command.

```bash
python -c "import torch; print(torch.cuda.get_arch_list())"
```

Find the architecture for a GPU with the following command.

<hfoptions id="arch">
<hfoption id="same GPUs">

```bash
CUDA_VISIBLE_DEVICES=0 python -c "import torch; print(torch.cuda.get_device_capability())"
```

</hfoption>
<hfoption id="specific GPU">

Run the following command to find the architecture for GPU `0`. The results will show a value for `major` and `minor`, which is your GPU architecture. The GPU architecture below is `8.6`.

```bash
CUDA_VISIBLE_DEVICES=0 python -c "import torch; \
print(torch.cuda.get_device_properties(torch.device('cuda')))
"_CudaDeviceProperties(name='GeForce RTX 3090', major=8, minor=6, total_memory=24268MB, multi_processor_count=82)"
```

</hfoption>
</hfoptions>

If you get `8, 6`, then you can set `TORCH_CUDA_ARCH_LIST="8.6"`. For multiple GPUs with different architectures, list them like `TORCH_CUDA_ARCH_LIST="6.1;8.6"`.

It is also possible to not specify `TORCH_CUDA_ARCH_LIST` and the build program automatically queries the GPU architecture of the build. However, it may or may not match the actual GPU on the target machine which is why it is better to explicitly specify the correct architecture.

For training on multiple machines with the same setup, you'll need to make a binary wheel as shown below.

```bash
git clone https://github.com/microsoft/DeepSpeed/
cd DeepSpeed
rm -rf build
TORCH_CUDA_ARCH_LIST="8.6" DS_BUILD_CPU_ADAM=1 DS_BUILD_UTILS=1 \
python setup.py build_ext -j8 bdist_wheel
```

This command generates a binary wheel that'll look something like `dist/deepspeed-0.3.13+8cd046f-cp38-cp38-linux_x86_64.whl`. Install this wheel locally or on another machine.

```bash
pip install deepspeed-0.3.13+8cd046f-cp38-cp38-linux_x86_64.whl
```

## Communication issues

Distributed training involves communication between processes and or nodes and this can be a potential source of errors.

Download the script below to diagnose network issues, and then run it to test GPU communication. The example command below tests how two GPUs communicate. Adjust the `--nproc_per_node` and `--nnodes` parameters to adapt it to your system.

```bash
wget https://raw.githubusercontent.com/huggingface/transformers/main/scripts/distributed/torch-distributed-gpu-test.py
python -m torch.distributed.run --nproc_per_node 2 --nnodes 1 torch-distributed-gpu-test.py
```

The script prints an `OK` status if both GPUs are able to communicate and allocate memory. Take a closer look at the diagnostic script for more details and a recipe for running it in a SLURM environment.

Add the `NCCL_DEBUG=INFO` environment variable to report more NCCL-related debugging information.

```bash
NCCL_DEBUG=INFO python -m torch.distributed.run --nproc_per_node 2 --nnodes 1 torch-distributed-gpu-test.py
```

## Underflow and overflow detection

Underflow and overflow can occur when activations or weights are `inf`, `nan`, and when `loss=NaN`. This may indicate an underflow or overflow issue. To detect these issues, activate the `DebugUnderflowOverflow` module in [`TrainingArguments.debug`] or import and add the module to your own training loop or another trainer class.

<hfoption id="overflow">
<hfoption id="Trainer">

```py
from transformers import TrainingArguments

args = TrainingArguments(
    debug="underflow_overflow",
    ...
)
```

</hfoption>
<hfoption id="PyTorch training loop">

```py
from transformers.debug_utils import DebugUnderflowOverflow

debug_overflow = DebugUnderflowOverflow(model)
```

</hfoption>
</hfoptions>

The [`~debug_utils.DebugUnderflowOverflow`] module inserts hooks into the model to test the input and output variables and the corresponding model weights after each forward call. If `inf` or `nan` is detected in at least one element of the activations or weights, the module prints a report like the one shown below.

The example below is for fp16 mixed precision training with a [google/mt5-small](https://huggingface.co/google/mt5-small).

```shell
Detected inf/nan during batch_number=0
Last 21 forward frames:
abs min  abs max  metadata
                  encoder.block.1.layer.1.DenseReluDense.dropout Dropout
0.00e+00 2.57e+02 input[0]
0.00e+00 2.85e+02 output
[...]
                  encoder.block.2.layer.0 T5LayerSelfAttention
6.78e-04 3.15e+03 input[0]
2.65e-04 3.42e+03 output[0]
             None output[1]
2.25e-01 1.00e+04 output[2]
                  encoder.block.2.layer.1.layer_norm T5LayerNorm
8.69e-02 4.18e-01 weight
2.65e-04 3.42e+03 input[0]
1.79e-06 4.65e+00 output
                  encoder.block.2.layer.1.DenseReluDense.wi_0 Linear
2.17e-07 4.50e+00 weight
1.79e-06 4.65e+00 input[0]
2.68e-06 3.70e+01 output
                  encoder.block.2.layer.1.DenseReluDense.wi_1 Linear
8.08e-07 2.66e+01 weight
1.79e-06 4.65e+00 input[0]
1.27e-04 2.37e+02 output
                  encoder.block.2.layer.1.DenseReluDense.dropout Dropout
0.00e+00 8.76e+03 input[0]
0.00e+00 9.74e+03 output
                  encoder.block.2.layer.1.DenseReluDense.wo Linear
1.01e-06 6.44e+00 weight
0.00e+00 9.74e+03 input[0]
3.18e-04 6.27e+04 output
                  encoder.block.2.layer.1.DenseReluDense T5DenseGatedGeluDense
1.79e-06 4.65e+00 input[0]
3.18e-04 6.27e+04 output
                  encoder.block.2.layer.1.dropout Dropout
3.18e-04 6.27e+04 input[0]
0.00e+00      inf output
```

At the start of the report, you can see which batch number the error occurred. In this case, it occurred on the first batch.

Each frame describes the module it is reporting on. For example, the frame below inspected `encoder.block.2.layer.1.layer_norm`. This indicates the layer norm in the first layer of the second block of the encoder. The forward calls are to `T5LayerNorm`.

```shell
                  encoder.block.2.layer.1.layer_norm T5LayerNorm
8.69e-02 4.18e-01 weight
2.65e-04 3.42e+03 input[0]
1.79e-06 4.65e+00 output
```

The last frame reports on the `Dropout.forward` function. It called the `dropout` attribute from inside the `DenseReluDense` class. You can observe that the overflow (`inf`) occurred in the first layer of the encoders second block in the first batch. The absolute largest input element was 6.27e+04.

```shell
                  encoder.block.2.layer.1.DenseReluDense T5DenseGatedGeluDense
1.79e-06 4.65e+00 input[0]
3.18e-04 6.27e+04 output
                  encoder.block.2.layer.1.dropout Dropout
3.18e-04 6.27e+04 input[0]
0.00e+00      inf output
```

The `T5DenseGatedGeluDense.forward` function output activations had an absolute maximum value of 6.27e+04 which is close to fp16s maximum limit of 6.4e+04. In the next step, `Dropout` renormalizes the weights, after zeroing some elements, which pushes the absolute maximum value to greater than 6.4e+04 resulting in an overflow.

Now that you know where the error is happening, you can investigate the modeling code in [modeling_t5.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py).

```py
class T5DenseGatedGeluDense(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.gelu_act = ACT2FN["gelu_new"]

    def forward(self, hidden_states):
        hidden_gelu = self.gelu_act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states
```

One solution is to back a few steps before the values started growing too large and switch to fp32 so the numbers don't overflow when multiplied or summed. Another potential solution is to temporarily disable mixed precision training (`amp`).

```py
import torch

def forward(self, hidden_states):
    if torch.is_autocast_enabled():
        with torch.cuda.amp.autocast(enabled=False):
            return self._forward(hidden_states)
    else:
        return self._forward(hidden_states)
```

The report only returns inputs and outputs of full frames, so you may also want to analyze the intermediate values of any `forward` function as well. Add the `detect_overflow` function after the forward calls to track `inf` or `nan` values in the intermediate `forwarded_states`.

```py
from debug_utils import detect_overflow

class T5LayerFF(nn.Module):
    [...]

    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        detect_overflow(forwarded_states, "after layer_norm")
        forwarded_states = self.DenseReluDense(forwarded_states)
        detect_overflow(forwarded_states, "after DenseReluDense")
        return hidden_states + self.dropout(forwarded_states)
```

Finally, you can configure the number of frames printed by [`~debug_utils.DebugUnderflowOverflow`].

```py
from transformers.debug_utils import DebugUnderflowOverflow

debug_overflow = DebugUnderflowOverflow(model, max_frames_to_save=100)
```

### Batch tracing

[`~debug_utils.DebugUnderflowOverflow`] is able to trace the absolute minimum and maximum values in each batch with the underflow and overflow feature disabled. This is useful for identifying where errors are occurring in the model.

The example below shows how to trace the minimum and maximum values in batches 1 and 3 (batches are zero-indexd).

```py
debug_overflow = DebugUnderflowOverflow(model, trace_batch_nums=[1, 3])
```

```shell
                  *** Starting batch number=1 ***
abs min  abs max  metadata
                  shared Embedding
1.01e-06 7.92e+02 weight
0.00e+00 2.47e+04 input[0]
5.36e-05 7.92e+02 output
[...]
                  decoder.dropout Dropout
1.60e-07 2.27e+01 input[0]
0.00e+00 2.52e+01 output
                  decoder T5Stack
     not a tensor output
                  lm_head Linear
1.01e-06 7.92e+02 weight
0.00e+00 1.11e+00 input[0]
6.06e-02 8.39e+01 output
                   T5ForConditionalGeneration
     not a tensor output

                  *** Starting batch number=3 ***
abs min  abs max  metadata
                  shared Embedding
1.01e-06 7.92e+02 weight
0.00e+00 2.78e+04 input[0]
5.36e-05 7.92e+02 output
[...]
```

[`~debug_utils.DebugUnderflowOverflow`] reports on a large number of frames which is easier for debugging. Once you know where a problem is occurring, say batch 150, then you can focus the trace for batches 149 and 150 and compare where the numbers are diverging.

It is also possible to abort the trace after a certain batch number, for example, batch 3.

```py
debug_overflow = DebugUnderflowOverflow(model, trace_batch_nums=[1, 3], abort_after_batch_num=3)
```
