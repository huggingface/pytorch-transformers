<!---
Copyright 2022 The HuggingFace Team. All rights reserved.

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

Install 🤗 Transformers for whichever deep learning library you're working with, setup your cache, and optionally configure 🤗 Transformers to run offline.

🤗 Transformers is tested on Python 3.6+, PyTorch 1.1.0+, TensorFlow 2.0+, and Flax. Follow the installation instructions below for the deep learning library you are using:

* [PyTorch](https://pytorch.org/get-started/locally/) installation instructions.
* [TensorFlow 2.0](https://www.tensorflow.org/install/pip) installation instructions.
* [Flax](https://flax.readthedocs.io/en/latest/) installation instructions.

## Install with pip

You should install 🤗 Transformers in a [virtual environment](https://docs.python.org/3/library/venv.html). If you're unfamiliar with Python virtual environments, take a look at this [guide](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/). A virtual environment makes it easier to manage different projects, and avoid compatibility issues between dependencies.

Now you're ready to install 🤗 Transformers with the following command:

```bash
pip install transformers
```

For GPU acceleration, install the appropriate CUDA drivers for [PyTorch](https://pytorch.org/get-started/locally) and TensorFlow(https://www.tensorflow.org/install/pip).

Run the command below to check if your system detects an NVIDIA GPU.

```bash
nvidia-smi
```

For CPU-support only, you can conveniently install 🤗 Transformers and a deep learning library in one line. For example, install 🤗 Transformers and PyTorch with:

```bash
pip install 'transformers[torch]'
```

🤗 Transformers and TensorFlow 2.0:

```bash
pip install 'transformers[tf-cpu]'
```

<Tip warning={true}>

M1 / ARM Users

You will need to install the following before installing TensorFlow 2.0
```bash
brew install cmake
brew install pkg-config
```

</Tip>

🤗 Transformers and Flax:

```bash
pip install 'transformers[flax]'
```

Finally, check if 🤗 Transformers has been properly installed by running the following command. It will download a pretrained model:

```bash
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('we love you'))"
```

Then print out the label and score:

```bash
[{'label': 'POSITIVE', 'score': 0.9998704791069031}]
```

## Install from source

Install 🤗 Transformers from source with the following command:

```bash
pip install git+https://github.com/huggingface/transformers
```

This command installs the bleeding edge `main` version rather than the latest `stable` version. The `main` version is useful for staying up-to-date with the latest developments. For instance, if a bug has been fixed since the last official release but a new release hasn't been rolled out yet. However, this means the `main` version may not always be stable. We strive to keep the `main` version operational, and most issues are usually resolved within a few hours or a day. If you run into a problem, please open an [Issue](https://github.com/huggingface/transformers/issues) so we can fix it even sooner!

Check if 🤗 Transformers has been properly installed by running the following command:

```bash
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('I love you'))"
```

## Editable install

You will need an editable install if you'd like to:

* Use the `main` version of the source code.
* Contribute to 🤗 Transformers and need to test changes in the code.

Clone the repository and install 🤗 Transformers with the following commands:

```bash
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install -e .
```

These commands will link the folder you cloned the repository to and your Python library paths. Python will now look inside the folder you cloned to in addition to the normal library paths. For example, if your Python packages are typically installed in `~/anaconda3/envs/main/lib/python3.7/site-packages/`, Python will also search the folder you cloned to: `~/transformers/`.

<Tip warning={true}>

You must keep the `transformers` folder if you want to keep using the library.

</Tip>

Now you can easily update your clone to the latest version of 🤗 Transformers with the following command:

```bash
cd ~/transformers/
git pull
```

Your Python environment will find the `main` version of 🤗 Transformers on the next run.

## Install with conda

Install from the conda channel `conda-forge`:

```bash
conda install conda-forge::transformers
```

## Cache setup

Pretrained models are downloaded and locally cached at: `~/.cache/huggingface/hub`. This is the default directory given by the shell environment variable `TRANSFORMERS_CACHE`. On Windows, the default directory is given by `C:\Users\username\.cache\huggingface\hub`. You can change the shell environment variables shown below - in order of priority - to specify a different cache directory:

1. Shell environment variable (default): `HF_HUB_CACHE` or `TRANSFORMERS_CACHE`.
2. Shell environment variable: `HF_HOME`.
3. Shell environment variable: `XDG_CACHE_HOME` + `/huggingface`.

<Tip>

🤗 Transformers will use the shell environment variables `PYTORCH_TRANSFORMERS_CACHE` or `PYTORCH_PRETRAINED_BERT_CACHE` if you are coming from an earlier iteration of this library and have set those environment variables, unless you specify the shell environment variable `TRANSFORMERS_CACHE`.

</Tip>

## Offline mode

Run 🤗 Transformers in a firewalled or offline environment with locally cached files by setting the environment variable `HF_HUB_OFFLINE=1`.

<Tip>

Add [🤗 Datasets](https://huggingface.co/docs/datasets/) to your offline training workflow with the environment variable `HF_DATASETS_OFFLINE=1`.

</Tip>

```bash
HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
python examples/pytorch/translation/run_translation.py --model_name_or_path google-t5/t5-small --dataset_name wmt16 --dataset_config ro-en ...
```

This script should run without hanging or waiting to timeout because it won't attempt to download the model from the Hub.

You can also bypass loading a model from the Hub from each [`~PreTrainedModel.from_pretrained`] call with the [`local_files_only`] parameter. When set to `True`, only local files are loaded:

```py
from transformers import T5Model

model = T5Model.from_pretrained("./path/to/local/directory", local_files_only=True)
```

### Fetch models and tokenizers to use offline

Another option for using 🤗 Transformers offline is to download the files ahead of time, and then point to their local path when you need to use them offline. There are three ways to do this:

* Download a file through the user interface on the [Model Hub](https://huggingface.co/models) by clicking on the ↓ icon.

    ![download-icon](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/download-icon.png)

* Use the [`PreTrainedModel.from_pretrained`] and [`PreTrainedModel.save_pretrained`] workflow:

    1. Download your files ahead of time with [`PreTrainedModel.from_pretrained`]:

    ```py
    >>> from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    >>> tokenizer = AutoTokenizer.from_pretrained("bigscience/T0_3B")
    >>> model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/T0_3B")
    ```

    2. Save your files to a specified directory with [`PreTrainedModel.save_pretrained`]:

    ```py
    >>> tokenizer.save_pretrained("./your/path/bigscience_t0")
    >>> model.save_pretrained("./your/path/bigscience_t0")
    ```

    3. Now when you're offline, reload your files with [`PreTrainedModel.from_pretrained`] from the specified directory:

    ```py
    >>> tokenizer = AutoTokenizer.from_pretrained("./your/path/bigscience_t0")
    >>> model = AutoModel.from_pretrained("./your/path/bigscience_t0")
    ```

* Programmatically download files with the [huggingface_hub](https://github.com/huggingface/huggingface_hub/tree/main/src/huggingface_hub) library:

    1. Install the `huggingface_hub` library in your virtual environment:

    ```bash
    python -m pip install huggingface_hub
    ```

    2. Use the [`hf_hub_download`](https://huggingface.co/docs/hub/adding-a-library#download-files-from-the-hub) function to download a file to a specific path. For example, the following command downloads the `config.json` file from the [T0](https://huggingface.co/bigscience/T0_3B) model to your desired path:

    ```py
    >>> from huggingface_hub import hf_hub_download

    >>> hf_hub_download(repo_id="bigscience/T0_3B", filename="config.json", cache_dir="./your/path/bigscience_t0")
    ```

Once your file is downloaded and locally cached, specify it's local path to load and use it:

```py
>>> from transformers import AutoConfig

>>> config = AutoConfig.from_pretrained("./your/path/bigscience_t0/config.json")
```

<Tip>

See the [How to download files from the Hub](https://huggingface.co/docs/hub/how-to-downstream) section for more details on downloading files stored on the Hub.

</Tip>

## Troubleshooting

See below for some of the more common installation issues and how to resolve them.

### Unsupported Python version

Ensure you are using Python 3.9 or later. Run the command below to check your Python version.

```
python --version
```

### Missing dependencies

Install all required dependencies by running the following command. Ensure you’re in the project directory before executing the command.

```
pip install -r requirements.txt
```

### Windows-specific

If you encounter issues on Windows, you may need to activate Developer Mode. Navigate to Windows Settings > For Developers > Developer Mode.

Alternatively, create and activate a virtual environment as shown below.

```
python -m venv env
.\env\Scripts\activate
```


