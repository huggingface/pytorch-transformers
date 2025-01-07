<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Training scripts

Transformers provides many example training scripts for deep learning frameworks (PyTorch, TensorFlow, Flax) and tasks in [transformers/examples](https://github.com/huggingface/transformers/tree/main/examples). There are additional scripts in [transformers/research projects](https://github.com/huggingface/transformers/tree/main/examples/research_projects) and [transformers/legacy](https://github.com/huggingface/transformers/tree/main/examples/legacy), but these aren't actively maintained and requires a specific version of Transformers.

Example scripts are only examples and you may need to adapt the script to your use-case. To help you with this, most scripts are very transparent in how data is preprocessed, allowing you to edit it as necessary.

For any feature you'd like to implement in an example script, please discuss it on the [forum](https://discuss.huggingface.co/) or in an [issue](https://github.com/huggingface/transformers/issues) before submitting a pull request. While we welcome contributions, it is unlikely a pull request that adds more functionality is added at the cost of readability.

This guide will show you how to run an example summarization training script in [PyTorch](https://github.com/huggingface/transformers/tree/main/examples/pytorch/summarization) and [TensorFlow](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/summarization).

## Setup

Install Transformers from source in a new virtual environment to run the latest version of the example script.

```bash
git clone https://github.com/huggingface/transformers
cd transformers
pip install .
```

Run the command below to checkout a script from a specific or older version of Transformers.

```bash
git checkout tags/v3.5.1
```

After you've setup the correct version, navigate to the example folder of your choice and install the example specific requirements.

```bash
pip install -r requirements.txt
```

## Run a script

Start with a smaller dataset by including the `max_train_samples`, `max_eval_samples`, and `max_predict_samples` parameters to truncate the dataset to a maximum number of samples. This helps ensure training works as expected before committing to the entire dataset which can take hours to complete.

> [!WARNING]
> Not all example scripts support the `max_predict_samples` parameter. Run the command below to check whether a script supports it or not.
> ```bash
> examples/pytorch/summarization/run_summarization.py -h
> ```

The example below fine-tunes [T5-small](https://huggingface.co/google-t5/t5-small) on the [CNN/DailyMail](https://huggingface.co/datasets/abisee/cnn_dailymail) dataset. T5 requires an additional `source_prefix` parameter to prompt it to summarize.

<hfoptions id="script">
<hfoption id="PyTorch">

The example script downloads and preprocesses a dataset, and then fine-tunes it with [`Trainer`] with a supported model architecture. 

Resuming training from a checkpoint is very useful if training is interrupted because you don't have to start over again. There are two ways to resume training from a checkpoint.

* `--output dir previous_output_dir` resumes training from the latest checkpoint stored in `output_dir`. Remove the `--overwrite_output_dir` parameter if you're using this method.
* `--resume_from_checkpoint path_to_specific_checkpoint` resumes training from a specific checkpoint folder.

Share your model on the [Hub](https://huggingface.co/) with the `--push_to_hub` parameter. It creates a repository and uploads the model to the folder name specified in `--output_dir`. You could also use the `--push_to_hub_model_id` parameter to specify the repository name.

```bash
python examples/pytorch/summarization/run_summarization.py \
    --model_name_or_path google-t5/t5-small \
    # remove the `max_train_samples`, `max_eval_samples` and `max_predict_samples` if everything works
    --max_train_samples 50 \
    --max_eval_samples 50 \
    --max_predict_samples 50 \
    --do_train \
    --do_eval \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir /tmp/tst-summarization \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --push_to_hub \
    --push_to_hub_model_id finetuned-t5-cnn_dailymail \
    # remove if using `output_dir previous_output_dir`
    # --overwrite_output_dir \
    --output_dir previous_output_dir \
    # --resume_from_checkpoint path_to_specific_checkpoint \
    --predict_with_generate \
```

For mixed precision and distributed training, include the following parameters and launch training with [torchrun](https://pytorch.org/docs/stable/elastic/run.html).

* Add the `fp16` or `bf16` parameters to enable mixed precision training. XPU devices only supports `bf16`.
* Add the `nproc_per_node` parameter to set number of GPUs to train with.

```bash
torchrun \
    --nproc_per_node 8 pytorch/summarization/run_summarization.py \
    --fp16 \
    ...
    ...
```

PyTorch supports TPUs, hardware designed to accelerate performance, through the [PyTorch/XLA](https://github.com/pytorch/xla/blob/master/README.md) package. Launch the `xla_spawn.py` script and use `num _cores` to set the number of TPU cores to train with.

```bash
python xla_spawn.py --num_cores 8 pytorch/summarization/run_summarization.py \
    --model_name_or_path google-t5/t5-small \
    ...
    ...
```

</hfoption>
<hfoption id="TensorFlow">

```bash
python examples/tensorflow/summarization/run_summarization.py  \
    --model_name_or_path google-t5/t5-small \
    # remove the `max_train_samples`, `max_eval_samples` and `max_predict_samples` if everything works
    --max_train_samples 50 \
    --max_eval_samples 50 \
    --max_predict_samples 50 \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --output_dir /tmp/tst-summarization  \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --num_train_epochs 3 \
    --do_train \
    --do_eval \
```

TensorFlow uses the [MirroredStrategy](https://www.tensorflow.org/guide/distributed_training#mirroredstrategy) for distributed training and doesn't require adding any additional parameters. The script uses multiple GPUs by default if they are available.

For TPU training, TensorFlow scripts use the [TPUStrategy](https://www.tensorflow.org/guide/distributed_training#tpustrategy). Pass the TPU resource name to the `--tpu` parameter.

```bash
python run_summarization.py  \
    --tpu name_of_tpu_resource \
    ...
    ...
```

</hfoption>
</hfoptions>

## Accelerate

[Accelerate](https://huggingface.co/docs/accelerate) is designed to simplify distributed training while offering complete visibility into the PyTorch training loop. If you're planning on training with a script with Accelerate, use the `_no_trainer.py` version of the script.

Install Accelerate from source to ensure you have the latest version.

```bash
pip install git+https://github.com/huggingface/accelerate
```

Run the [accelerate config](https://huggingface.co/docs/accelerate/package_reference/cli#accelerate-config) command to answer a few questions about your training setup. This creates and saves a config file about your system. 

```bash
accelerate config
```

You can use [accelerate test](https://huggingface.co/docs/accelerate/package_reference/cli#accelerate-test) to ensure your system is properly configured.

```bash
accelerate test
```

Run [accelerate launch](https://huggingface.co/docs/accelerate/package_reference/cli#accelerate-launch) to start training.

```bash
accelerate launch run_summarization_no_trainer.py \
    --model_name_or_path google-t5/t5-small \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir ~/tmp/tst-summarization \
```

## Custom dataset

The summarization scripts supports custom datasets as long as they are a CSV or JSONL file. When using your own dataset, you need to specify the following additional parameters.

* `train_file` and `validation_file` specify the path to your training and validation files.
* `text_column` is the input text to summarize.
* `summary_column` is the target text to output.

An example command for summarizing a custom dataset is shown below.

```bash
python examples/pytorch/summarization/run_summarization.py \
    --model_name_or_path google-t5/t5-small \
    --do_train \
    --do_eval \
    --train_file path_to_csv_or_jsonlines_file \
    --validation_file path_to_csv_or_jsonlines_file \
    --text_column text_column_name \
    --summary_column summary_column_name \
    --source_prefix "summarize: " \
    --output_dir /tmp/tst-summarization \
    --overwrite_output_dir \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --predict_with_generate \
```
