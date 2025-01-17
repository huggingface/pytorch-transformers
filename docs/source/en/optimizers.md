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

# Optimizers

Transformers offers two optimizers natively, AdamW and AdaFactor. But it also provides integrations for other more specialized optimizers. Install the library that offers the optimizer and drop it in to the `optim` parameter in [`TrainingArguments`].

This guide will show you how to use these optimizers with [`Trainer`] using the [`TrainingArguments`] shown below.

```py
import torch
from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM, Trainer

args = TrainingArguments(
    output_dir="./test-optimizer",
    max_steps=1000,
    per_device_train_batch_size=4,
    logging_strategy="steps",
    logging_steps=1,
    learning_rate=2e-5,
    save_strategy="no",
    run_name="optimizer-name",
)
```

## GrokAdamW

```bash
pip install grokadamw
```

[GrokAdamW](https://github.com/cognitivecomputations/grokadamw) is an optimizer designed to help models that benefit from *grokking*, a term used to describe delayed generalization because of slow-varying gradients. It is particularly useful for models requiring more advanced optimization techniques to achieve better performance and stability.

```diff
import torch
from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="./test-grokadamw",
    max_steps=1000,
    per_device_train_batch_size=4,
+   optim="grokadamw",
    logging_strategy="steps",
    logging_steps=1,
    learning_rate=2e-5,
    save_strategy="no",
    run_name="grokadamw",
)
```

## LOMO

```bash
pip install lomo-optim
```

[Low-Memory Optimization (LOMO)](https://github.com/OpenLMLab/LOMO) is a family of optimizers, [LOMO](https://huggingface.co/papers/2306.09782) and [AdaLomo](https://hf.co/papers/2310.10195), designed for low-memory full-parameter finetuning of LLMs. Both LOMO optimizers fuse the gradient computation and parameter update in one step to reduce memory usage. AdaLomo builds on top of LOMO by incorporating an adaptive learning rate for each parameter like the Adam optimizer.

> [!TIP]
> It is recommended to use AdaLomo without `grad_norm` for better performance and higher throughput.

```diff
args = TrainingArguments(
    output_dir="./test-lomo",
    max_steps=1000,
    per_device_train_batch_size=4,
+   optim="adalomo",
    gradient_checkpointing=True,
    gradient_checkpointing=True,
    logging_strategy="steps",
    logging_steps=1,
    learning_rate=2e-6,
    save_strategy="no",
    run_name="adalomo",
)
```

## Schedule Free

```bash
pip install schedulefree
```

[Schedule Free optimizer (SFO)](https://hf.co/papers/2405.15682) replaces the base optimizers momentum with a combination of averaging and interpolation. Unlike a traditional scheduler, SFO completely removes the need to anneal the learning rate.

SFO supports both the AdamW (`schedule_free_adamw`) and SGD (`schedule_free_sgd`) optimizers.

```diff
args = TrainingArguments(
    output_dir="./test-schedulefree",
    max_steps=1000,
    per_device_train_batch_size=4,
+   optim="schedule_free_adamw",
    gradient_checkpointing=True,
    logging_strategy="steps",
    logging_steps=1,
    learning_rate=2e-6,
    save_strategy="no",
    run_name="sfo",
)
```
