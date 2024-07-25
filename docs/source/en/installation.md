<!---
Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Installation

Transformers works with [PyTorch](https://pytorch.org/get-started/locally/), [TensorFlow 2.0](https://www.tensorflow.org/install/pip), and [Flax](https://flax.readthedocs.io/en/latest/). It has been tested on Python 3.6+, PyTorch 1.1.0+, TensorFlow 2.0+, and Flax.

## Virtual environment

A virtual environment can help you manage different projects and avoid compatibility issues between dependencies. Take a look at the [Install packages in a virtual environment using pip and venv](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) guide if you're unfamiliar with Python virtual environments.

Create a virtual environment in your project directory.

```bash
python -m venv .env
```

Activate the virtual environment.

<hfoptions id="venv">
<hfoption id="macOS/Linux">

```bash
source ./env/bin/activate
```

</hfoption>
<hfoption id="Windows">

```bash
.env/Scripts/activate
```

</hfoption>
</hfoptions>

## pip

[pip](https://pip.pypa.io/en/stable/) is a package installer for Python. Install Transformers with pip in your newly created virtual environment.

```bash
pip install transformers
```

For GPU acceleration, install the appropriate CUDA drivers for [PyTorch](https://pytorch.org/get-started/locally) and TensorFlow(https://www.tensorflow.org/install/pip).

Run the command below to check if your system detects an NVIDIA GPU.

```bash
nvidia-smi
```

To install a CPU-only version of Transformers and a machine learning framework, run the following command.

<hfoptions id="cpu-only">
<hfoption id="PyTorch">

```bash
pip install 'transformers[torch]'
```

</hfoption>
<hfoption id="TensorFlow">

For M1 ARM-based hardware, you need to install CMake and pkg-config first.

```bash
brew install cmake
brew install pkg-config
```

Install TensorFlow 2.0.

```bash
pip install 'transformers[tf-cpu]'
```

</hfoption>
<hfoption id="Flax">

```bash
pip install 'transformers[flax]'
```

</hfoption>
</hfoptions>

Test whether your install was successful with the following command to quickly inference with a pretrained model. It should return a label and score for the provided text.

```bash
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('hugging face is the best'))"
[{'label': 'POSITIVE', 'score': 0.9998704791069031}]
```

### Source install

Installing from source installs the *latest* version rather than the *stable* version of the library. This ensures you get the most up-to-date changes in the library, which is useful for experimenting with the latest features or fixing a bug that hasn't been officially released yet.

The downside is that the latest version may not always be stable. If you encounter any problems, please open a [GitHub Issue](https://github.com/huggingface/transformers/issues) so we can fix it as soon as possible.

Install from source with the following command.

```bash
pip install git+https://github.com/huggingface/transformers
```

Check if the install was successful with the command below to quickly inference with a pretrained model. It should return a label and score for the provided text.

```bash
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('hugging face is the best'))"
[{'label': 'POSITIVE', 'score': 0.9998704791069031}]
```

### Editable install

An [editable install](https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs) is useful if you're developing locally with Transformers. It links your local copy of Transformers to the Transformers [repository](https://github.com/huggingface/transformers) instead of copying the files. The files are added to Python's import path.

```bash
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install -e .
```

> [!WARNING]
> You must keep the `transformers` folder to keep using it locally.

Update your local version of Transformers with the latest changes in the main repository with the following command.

```bash
cd ~/transformers/
git pull
```

## conda

[conda](https://docs.conda.io/projects/conda/en/stable/#) is a language-agnostic package manager. Install Transformers from the [conda-forge](https://anaconda.org/conda-forge/transformers) channel in your newly created virtual environment.

```bash
conda install conda-forge::transformers
```

## Setup

After installation, you can configure the Transformers cache location or setup the library for offline usage if you want.

### Cache directory

When you load a pretrained model with [`~PreTrainedModel.from_pretrained`], the model is downloaded from the Hub and locally cached. Every time you load a model, it checks whether the cached model is up-to-date. If it's the same, then the local model is loaded. If it's not the same, the updated model is downloaded and cached. This ensures you always have the latest version of a model.

The default directory given by the shell environment variable `TRANSFORMERS_CACHE` is `~/.cache/huggingface/hub`.

On Windows, the default directory is given by `C:\Users\username\.cache\huggingface\hub`.

Cache a model in a different directory by changing the path in the following shell environment variables (listed by priority).

1. [HF_HUB_CACHE](https://hf.co/docs/huggingface_hub/package_reference/environment_variables#hfhubcache) or `TRANSFORMERS_CACHE` (default)
2. [HF_HOME](https://hf.co/docs/huggingface_hub/package_reference/environment_variables#hfhome)
3. [XDG_CACHE_HOME](https://hf.co/docs/huggingface_hub/package_reference/environment_variables#xdgcachehome) + `/huggingface` (only if `HF_HOME` is not set)

Older versions of Transformers uses the shell environment variables `PYTORCH_TRANSFORMERS_CACHE` or `PYTORCH_PRETRAINED_BERT_CACHE`. You should keep these unless you specify the newer shell environment variable `TRANSFORMERS_CACHE`.

### Offline mode

To use Transformers in an offline or firewalled environment requires having the downloaded and cached files ahead of time. Download a model repository from the Hub with the [`~huggingface_hub.snapshot_download`] method.

> [!TIP]
> Refer to the [Download files from the Hub](https://hf.co/docs/huggingface_hub/guides/download) guide for more options for downloading files from the Hub. You can download files from specific revisions, download from the CLI, and even filter which files to download from a repository.

```py
from huggingface_hub import snapshot_download

snapshot_download(repo_id="meta-llama/Llama-2-7b-hf", repo_type="model")
```

Set the environment variable `HF_HUB_OFFLINE=1` to prevent HTTP calls to the Hub to download the model.

```bash
HF_HUB_OFFLINE=1 \
python examples/pytorch/language-modeling/run_clm.py --model_name_or_path meta-llama/Llama-2-7b-hf --dataset_name wikitext ...
```

Another option for only loading cached files is to set the `local_files_only` parameter to `True` in the [`~PreTrainedModel.from_pretrained`] method.

```py
from transformers import LlamaForCausalLM

model = LlamaForCausalLM.from_pretrained("./path/to/local/directory", local_files_only=True)
```
