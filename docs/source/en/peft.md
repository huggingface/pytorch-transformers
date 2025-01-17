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

# PEFT

[[open-in-colab]]

[PEFT](https://huggingface.co/docs/peft/index), a library of parameter-efficient finetuning methods, enable training and storing large models often on consumer GPUs. These methods only finetune a small number of extra model parameters, also known as adapters, on top of the pretrained model. A significant amount of memory is saved because the GPU doesn't need to store the optimizer states and gradients for the pretrained base model. Adapters are very lightweight, making it convenient to share, store, and load them.

This guide provides a short overview of the PEFT library and how to use it for training with Transformers. For more details, refer to the PEFT [documentation](https://huggingface.co/docs/peft/index).

Install PEFT with the command below.

<hfoptions id="install">
<hfoption id="pip">

```bash
pip install -U peft
```

</hfoption>
<hfoption id="source">

```bash
pip install git+https://github.com/huggingface/peft.git
```

</hfoption>
</hfoptions>

> [!TIP]
> PEFT currently supports the LoRA, IA3, and AdaLoRA methods for Transformers. To use another PEFT method, such as prompt learniing or prompt tuning, you'll need to use the PEFT library directly.

[Low-Rank Adaptation (LoRA)](https://huggingface.co/docs/peft/conceptual_guides/adapter#low-rank-adaptation-lora) is a very common PEFT method that decomposes the weight matrix into two smaller trainable matrices. Start by defining a [`~peft.LoraConfig`] object with the parameters shown below.

```py
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM

# create LoRA configuration object
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, # type of task to train on
    inference_mode=False, # set to False for training
    r=8, # dimension of the smaller matrices
    lora_alpha=32, # scaling factor
    lora_dropout=0.1 # dropout of LoRA layers
)
```

Add [`~peft.LoraConfig`] to the model with [`~integrations.PeftAdapterMixin.add_adapter`]. The model is ready to be passed to [`Trainer`] for training.

```py
model.add_adapter(lora_config, adapter_name="lora_1")
trainer = Trainer(model=model, ...)
trainer.train()
```

To add an additional trainable adapter on top of a model with an existing adapter attached, specify the modules you want to train in [modules_to_save()](https://huggingface.co/docs/peft/package_reference/lora#peft.LoraConfig.modules_to_save).

For example, to train the `lm_head` module on top of a causal language model with a LoRA adapter attached, set `modules_to_save=["lm_head"]`. Add the adapter to the model as shown below, and then pass it to [`Trainer`].

```py
from transformers import AutoModelForCausalLM
from peft import LoraConfig

model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b")

lora_config = LoraConfig(
    target_modules=["q_proj", "k_proj"],
    modules_to_save=["lm_head"],
)

model.add_adapter(lora_config)
trainer = Trainer(model=model, ...)
trainer.train()
```

Save your adapter with [`~PreTrainedModel.save_pretrained`] to reuse it.

## Load adapter

To load an adapter with Transformers, the Hub repository or local directory must contain an `adapter_config.json` file and the adapter weights. Load the adapter with [`~PreTrainedModel.from_pretrained`] or with [`~integrations.PeftAdapterMixin.load_adapter`].

<hfoptions id="load">
<hfoption id="from_pretrained">

```py
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("klcsp/gemma7b-lora-alpaca-11-v1")
```

</hfoption>
<hfoption id="load_adapter">

```py
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("google/gemma-7b")
model.load_adapter("klcsp/gemma7b-lora-alpaca-11-v1")
```

For very large models, it is helpful to load a quantized version of the model in 8 or 4-bit precision to save memory. Transformers supports quantization with its [bitsandbytes](https://huggingface.co/docs/bitsandbytes/index) integration. Specify in [`BitsAndBytesConfig`] whether you want to load a model in 8 or 4-bit precision.

For multiple devices, add `device_map="auto"` to automatically distribute the model across your hardware.

```py
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

model = AutoModelForCausalLM.from_pretrained(
    "klcsp/gemma7b-lora-alpaca-11-v1",
    quantization_config=BitsAndBytesConfig(load_in_8bit=True),
    device_map="auto",
)
```

## Set adapter

[`~integrations.PeftAdapterMixin.add_adapter`] adds a new adapter to a model. To add a second adapter, the new adapter must be the same type as the first adapter. Use the [`~integrations.PeftAdapterMixin.add_adapter.adapter_name`] parameter to assign a name to the adapter.

```py
model.add_adapter(lora_config, adapter_name="lora_2")
```

Once added, use [`~integrations.PeftAdapterMixin.set_adapter`] to force a model to use the specified adapter and disable the other adapters.

```py
model.set_adapter("lora_2")
```

## Enable and disable adapter

[`~integrations.PeftAdapterMixin.enable_adapters`] is a broader function that enables *all* adapters attached to a model, and [`~integrations.PeftAdapterMixin.disable_adapters`] disables *all* attached adapters.

```py
model.add_adapter(lora_1)
model.add_adapter(lora_2)
model.enable_adapters()

# disable all adapters
model.disable_adapters()
```
