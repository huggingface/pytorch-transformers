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

# Contribute

Transformers supports many quantization methods such as QLoRA, GPTQ, LLM.int8, and AWQ. However, there are still many more quantization approaches that haven't been integrated yet. To make adding and using these quantization methods with Transformers easier, use the [`~quantizers.HfQuantizer`] class.  [`~quantizers.HfQuantizer`] is designed to be an internal helper class for adding a quantization method instead of something applied to every PyTorch module.

This guide will show you how to integrate a new quantization method with [`~quantizers.HfQuantizer`].

## Requirements

Before integrating a new quantization method into Transformers, ensure the method meets the following requirements. Only quantization methods that can be run with PyTorch modules are supported.

- The quantization method is available through a Python package that is pip-installable (it is also fine if you can only install the package from source). Ideally, pre-compiled kernels are included in the pip package.
- The method can run on commonly-used hardware (CPU, GPU, etc.).
- The method is wrapped in a [nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html) ([`~bitsandbytes.nn.Linear8bitLt`], [`~bitsandbytes.nn.Linear4bit`]), and the quantized linear layer should have the following definition.

    ```py
    class Linear4bit(nn.Module):
        def __init__(self, ...):
            ...
        
        def forward(self, x):
            return my_4bit_kernel(x, self.weight, self.bias)
    ```

    This way, Transformers models are easily quantized by replacing instances of [nn.Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) with a target class.

- The quantization method should be serializable. You can save the quantized weights locally or push them to the Hub.
- Make sure the package containing the quantization kernels/primitive is stable (no frequent breaking changes).

Some quantization methods may require "pre-quantizing" the model through data calibration (AWQ). In this case, we prefer to only support inference in Transformers and let the third-party library maintained by the ML community deal handle the model quantization itself.

## Create new HFQuantizer class

1. Create a new quantization config class inside [src/transformers/utils/quantization_config.py](https://github.com/huggingface/transformers/blob/abbffc4525566a48a9733639797c812301218b83/src/transformers/utils/quantization_config.py). Add the new quantization config to the [_import_structure](https://github.com/huggingface/transformers/blob/abbffc4525566a48a9733639797c812301218b83/src/transformers/__init__.py#L1088) inside Transformers' [src/transformers/__init__.py](https://github.com/huggingface/transformers/blob/abbffc4525566a48a9733639797c812301218b83/src/transformers/__init__.py) file.

2. Create a new file inside [src/transformers/quantizers/](https://github.com/huggingface/transformers/tree/abbffc4525566a48a9733639797c812301218b83/src/transformers/quantizers) named `quantizer_your_method.py`, and make it inherit from [`~quantizers.HfQuantizer]. Make sure to add the new quantizer and quantization config in the quantization auto-mapping in [src/transformers/quantizers/auto.py](https://github.com/huggingface/transformers/blob/abbffc4525566a48a9733639797c812301218b83/src/transformers/quantizers/auto.py).

3. Define the following class attributes and property methods for your quantization method.

    - `requires_calibration`: Whether the quantization method requires a data calibration process. If set to `True`, you can only support inference (with quantized weights) and not inference and quantization.
    - `required_packages`: A list of strings of the required packages to use the quantized weights. You might need to define some new utility methods such as `is_auto_awq_available` in [transformers/src/utils/import_utils.py](https://github.com/huggingface/transformers/blob/abbffc4525566a48a9733639797c812301218b83/src/transformers/utils/import_utils.py).
    - `requires_parameters_quantization`: Only required if your quantization method requires extra attention to the underlying [nn.Parameter](https://pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html) object. For example, bitsandbytes uses [`~bitsandbytes.nn.Params4bit`] and [`~bitsandbytes.nn.Int8Params`], which requires some extra attention when quantizing the model. Most of the recent quantization method packs int2 and int4 weights inside [torch.uint8](https://pytorch.org/docs/stable/tensors.html) weights, so this flag should not be really required (set to `False` by default).
    - `is_serializable`: A property method to determine whether the method is serializable or not.
    - `is_trainable`:  A property method to determine whether you can fine-tune models on top of the quantization method (with or without PEFT approaches).

4. Write the `validate_environment` and `update_torch_dtype` methods. These methods are called before creating the quantized model to ensure users use the right configuration. Refer to other quantizers for an example of it is implemented.

5. Write the `_process_model_before_weight_loading` method. In Transformers, the quantized models are initialized first on the `"meta"` device before loading the weights. This means the `_process_model_before_weight_loading` method takes care of manipulating the model skeleton to replace some modules ([nn.Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)) with the target modules (quantization modules).

    You can define module replacement logic or any other utility method by creating a new file in [transformers/src/integrations/](https://github.com/huggingface/transformers/tree/abbffc4525566a48a9733639797c812301218b83/src/transformers/integrations) and exposing the relevant methods in that folder's `__init__.py` file. The best starting point would be to have a look at another quantization method such as [quantizer_awq.py](https://github.com/huggingface/transformers/blob/abbffc4525566a48a9733639797c812301218b83/src/transformers/quantizers/quantizer_awq.py).

6. Write the `_process_model_after_weight_loading` method. This method enables implementing additional features that require manipulating the model after loading the weights.

7. Document everything! Make sure your quantization method is documented by adding a new file under `docs/source/en/quantization`.

8. You should add tests by adding the package in our nightly Dockerfile inside `docker/transformers-quantization-latest-gpu` and then adding a new test file in `tests/quantization/xxx`. Feel free to check out existing quantization methods to see how it is implemented.
