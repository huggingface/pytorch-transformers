<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Contribute

Transformers is fortunate to have a passionate community of developers and researchers contribute models to the library. As an open-source first project, we're invested in empowering the community to actively add models.

When you add a model to Transformers, you'll learn:

- more about open-source best practices
- about a models architecture
- about the design principles behind Transformers
- how to efficiently test large models
- how to use Python utilities like [Black](https://black.readthedocs.io/en/stable/) and [Ruff](https://docs.astral.sh/ruff/) to create clean and readable code

It is challenging but also rewarding.

This guide will walk you through adding an example "BrandNewBert" PyTorch model to Transformers.

## New model addition

Open a [New model addition](https://github.com/huggingface/transformers/issues/new?assignees=&labels=New+model&template=new-model-addition.yml) issue to add a specific model.

> [!TIP]
> To add any model, filter by the [New model](https://github.com/huggingface/transformers/labels/New%20model) label for an open model request.

Now is a good time to get familiar with BrandNewBert. It can be helpful to read a models research paper to understand its technical details and implementation. You don't necessarily have to worry too much about the more theoretical aspects of the paper. Instead, focus on the practical details. Use the questions below to guide your reading.

- What type of model is BrandNewBert? Is it a encoder, decoder, or encoder-decoder model?
- What tasks can BrandNewBert be used for?
- What makes BrandNewBert different from other models?
- What models in Transformers are most similar to BrandNewBert?
- What tokenizer does BrandNewBert use?

## Transformers overview

Transformers is an opinionated library with its own unique philosophy and design choices. These choices help us scale Transformers while maintaining a sustainable level of maintenance.

> [!TIP]
> Learn more about our design principles in the [Philosophy](./philosophy) page.

Some of these design choices are:

- composition > over-abstraction
- duplicate code isn't always bad if it greatly improves readability and accessibility
- model files are self-contained and all the necessary model code is found in the `modeling.py` file

These design choices are important *for everyone* interacting with the model. It is easier to read, understand, and modify.

This section describes how the model and configuration classes interact and the Transformers code style you should adopt.

### Model and configuration

All Transformers' models inherit from a base [`PreTrainedModel`] and [`PretrainedConfig`] class. The configuration is the blueprint to the model.

<!-- <insert diagram here> -->

To keep the code readable, there is never more than two levels of abstraction for any model. The example model here, BrandNewBert, traces its inheritance from `BrandNewBertPreTrainedModel` and [`PreTrainedModel`]. It is important that a new model only depends on [`PreTrainedModel`] because it allows a model to be loaded and saved with [`~PreTrainedModel.from_pretrained`] and [`PreTrainedModel.save_pretrained`].

Other important functions like the forward method are defined in the `modeling.py` file.

Specific model heads (for example, for sequence classification or language modeling) should use the base model as a component that is called in the forward pass rather than inherting from it. This keeps abstraction low.

New models require a configuration, for example `BrandNewBertConfig`, that is stored as an attribute of [`PreTrainedModel`].

```py
model = BrandNewBertModel.from_pretrained("brandy/brand_new_bert")
model.config
```

Like [`PreTrainedModel`], [`PretrainedConfig`] provides [`~PretrainedConfig.from_pretrained`] and [`PretrainedConfig.save_pretrained`] methods.

When you use [`~PreTrainedModel.save_pretrained`], it automatically calls the configurations [`~PretrainedConfig.save_pretrained`] method so that both the model and configuration are saved together.

A model is saved to a `model.safetensors` file and a configuration is saved to a `config.json` file.

### Code style

Transformers prefers a clean and readable code style over a more abstracted one. Some of the coed style choices include:

- The forward pass is written in the `modeling.py` file, completely independent of other models in the library. To reuse a block from another model, copy the code and paste it with a `# Copied from` comment above it. For example, the `RobertaSelfAttention` class is copied from the `BertSelfAttention` class.

  ```py
  # Copied from transformers.models.bert.modeling_bert.BertSelfAttention with Bert->Roberta
  class RobertaSelfAttention(nn.Module):
  ```

  Refer to the [Check copies](./pr_checks#check-copies) section for more information about the `# Copied from` comment.

- The code should be accessible to users from a non-native English background. Pick descriptive variable names and avoid abbreviations. For example, "activation" is preferred over "act". One letter variables names are highly discouraged unless it's an infex in a for loop.

- Explicit code is preferred over shorter code even if it's longer.

- Avoid subclassing [nn.Sequential](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html). Subclass [nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module) instead so the code can be quickly debugged with print statements or breakpoints.

- Function signatures should be type-annotated. Otherwise, good variable names are preferred because they're more readable and understandable.

## Add a new model

With some background knowledge about your model and the Transformers library, you're ready to add BrandNewBert now!

> [!TIP]
> Each contributor has a unique style and workflow for porting models to Transformers. It may be helpful to take a look at how [GPT2](https://medium.com/huggingface/from-tensorflow-to-pytorch-265f40ef2a28) and [WMT19](https://huggingface.co/blog/porting-fsmt) were ported.

Some final tips to keep in mind are:

- Don't reinvent the wheel! Take your time to explore existing models and tokenizers to see what you can copy and reuse. [Grep](https://www.gnu.org/software/grep/) and [ripgrep](https://github.com/BurntSushi/ripgrep) are great tools for this.
- This is an engineering challenge more than a scientific one. Focus on the more practical aspects (set up an efficient debugging environment for example) instead of theoretical ones.
- Don't be shy to ask for help! We are here to support you. 🤗

### Dev environment

Click on the **Fork** button on the [Transformers](https://github.com/huggingface/transformers) repository to create your own copy of it to work on. Then clone the repository to your local disk and add the base repository as the remote.

```bash
git clone https://github.com/[your Github handle]/transformers.git
cd transformers
git remote add upstream https://github.com/huggingface/transformers.git
```

Create a virtual environment and do an [editable install](./installation#editable-install) of the library with the "dev" or development dependencies.

```bash
python -m venv .env
source .env/bin/activate
pip install -e ".[dev]"
```

Due to the number of optional dependencies as Transformers grows, this command may fail. In that case, install the "quality" dependencies. Also make sure you have a deep learning framework installed.

```bash
pip install -e ".[quality]"
```

Return to the parent directory and clone and install the original BrandNewBert repository.

```bash
git clone https://github.com/org_that_created_brand_new_bert_org/brand_new_bert.git
cd brand_new_bert
pip install -e .
```

Return to your clone of Transformers to begin porting BrandNewBert.

```bash
cd transformers
```

There are two debugging environments for running the original model, a notebook ([Google Colab](https://colab.research.google.com/notebooks/intro.ipynb) or [Jupyter](https://jupyter.org/)) or a local Python script.

> [!WARNING]
> We don't recommend setting up a GPU environment to run the original model. This can be costly and only verified when the model is working in Transformers. Instead, work in a CPU environment at first.

Notebooks are great for executing code cell-by-cell which can better help split logical components from one another. It can also accelerate debugging cycles because intermediate results can be stored. Notebooks can also be shared which is useful for working with contributors.

The downside of notebooks is that if you aren't used to them, it may take some time to get used to.

> [!TIP]
> If the model architecture is identical to an existing model, skip ahead to add a [conversion script](#conversion-script), because you can reuse the architecture of the existing model.

Run the command below to start and complete the questionnaire with some basic information about the new model. This command jumpstarts the process by automatically generating some model code that you'll need to adapt.

```bash
transformers-cli add-new-model-like
```

### Create a pull request

Before you start adapting the code, create a pull request to track your progress and get feedback from the Transformers team. Title your pull request "[WIP] Add BrandNewBert" so it's clear that this is a work in progress.

Create a branch with a descriptive name from your main branch.

```bash
git checkout -b add_brand_new_bert
```

Commit the code, and then fetch and rebase on the main branch.

```bash
git add .
git commit
git fetch upstream
git rebase upstream/main
```

Push any changes to your branch and click on **Compare & pull request** to open a pull request on GitHub. Open the pull request as a *draft* to signal it's a work in progress.

```bash
git push -u origin a-descriptive-name-for-my-changes
```

Include relevant Hugging Face team members GitHub handles in the pull request for questions, feedback, comments, and reviews. Direct team members to specific parts of the code you want by clicking on the **Files changed** tab, and then clicking on **+** to the left of the line number to add a comment. When a question or problem is solved, click on **Resolve** to indicate the issue is resolved. This keeps the conversation organized and clean.

Remember to periodically commit and push your work, and update your work with the current main branch.

```bash
git fetch upstream
git merge upstream/main
```

### Run original checkpoint

Before you start working on your model implementation, you should work on the original model implementation first to understand how it works.

This can be difficult if the original model repository is lacking documentation or if the codebase is complex. But you should use this as your motivation to implement the model in Transformers. Your contribution makes it more accessible and user-friendly to everyone.

Orient yourself with the original repository by doing the following.

- Locate the pretrained weights.
- Figure out how to the load pretrained weights into the model.
- Figure out how to run the tokenizer indepdently of the model.
- Trace one forward pass to understand which classes and functions are required. These are probably the only classes and functions you'll have to implement.
- Locate all the important components (model class, model subclasses, self-attention layer, etc.) of the model.
- Figure out how to debug the model in the original repository. Add print statements, use interactive debuggers like [ipdb](https://github.com/gotcha/ipdb), or a efficient integrated development environment (IDE) like [PyCharm](https://www.jetbrains.com/pycharm/).

The last point is especially important because you'll need a thorough understanding of what's happening inside the original model before you can reimplement it in Transformers. Feel free to open issues and pull requests in the original repository if you encounter any issues.

A good first step is to load a *small* pretrained checkpoint and try to reproduce a single forward pass with an example integer vector of inputs. For example, in pseudocode, this could look like the following.

```py
model = BrandNewBertModel.load_pretrained_checkpoint("/path/to/checkpoint/")
input_ids = [0, 4, 5, 2, 3, 7, 9]  # vector of input ids
original_output = model.predict(input_ids)
```

If you run into issues, you'll need to choose one of the following debugging decomposition strategies depending on the original models codebase.

<hfoptions id="debug-strategy">
<hfoption id="sub-components">

This strategy relies on breaking the original model into smaller sub-components, such as when the code can be easily run in eager mode. While more difficult, there are some advantages to this approach.

1. It is easier later to compare the original model to your implementation. You can automatically verify that each individual component matches its corresponding component in Transformers' implementation. This is better than relying on a visual comparison based on print statements.
2. It is easier to port individal components instead of the entire model.
3. It is easier for understanding how a model works by breaking it up into its components.
4. It is easier to prevent regressions at a later stage when you change your code thanks to component-by-component tests.

> [!TIP]
> Refer to the ELECTRA [integration checks](https://gist.github.com/LysandreJik/db4c948f6b4483960de5cbac598ad4ed) for a good example of how to decompose a model into smaller components.

</hfoption>
<hfoption id="model and tokenizer">

This strategy is viable when the original codebase is too complex, only allows intermediate components to be run in compiled mode, or if it's too time-consuming (maybe even impossible) to separate the model into smaller sub-components.

For example, the MeshTensorFlow implementation of [T5](https://github.com/tensorflow/mesh/tree/master/mesh_tensorflow) is too complex and doesn't offer a simple way to decompose the model into its sub-components. In this situation, you'll have to relay on verifying print statements.

</hfoption>
</hfoptions>

Whichever strategy you choose, it is recommended to debug the initial layers first and the final layers last. Retrieve the output, either with print statements or sub-component functions, of the following layers in this order.

1. input ids passed to the model
2. word embeddings
3. input of the first Transformer layer
4. output of the first Transformer layer
5. output of the following n-1 Transformer layers
6. output of the whole model

The input ids should just be an array of integers like `input_ids = [0, 4, 4, 3, 2, 4, 1, 7, 19]`.

Layer outputs often consist of multi-dimensional float arrays.

```py
[[
 [-0.1465, -0.6501,  0.1993,  ...,  0.1451,  0.3430,  0.6024],
 [-0.4417, -0.5920,  0.3450,  ..., -0.3062,  0.6182,  0.7132],
 [-0.5009, -0.7122,  0.4548,  ..., -0.3662,  0.6091,  0.7648],
 ...,
 [-0.5613, -0.6332,  0.4324,  ..., -0.3792,  0.7372,  0.9288],
 [-0.5416, -0.6345,  0.4180,  ..., -0.3564,  0.6992,  0.9191],
 [-0.5334, -0.6403,  0.4271,  ..., -0.3339,  0.6533,  0.8694]]],
```

Every Transformers model output should have a precision or error tolerance of *1e-3*. This accounts for any output differences that arise from using a different library framework. Compare the intermediate outputs of the original model with the Transformers implementation to ensure they're nearly identical. Having an *efficient* debugging environment is crucial for this step.

Here are some tips for an efficient debugging environment.

- To debug intermediate results, it depends on the machine learning framework the original model repository is using. For PyTorch, you should write a script to decompose the original model into smaller sub-components to retrieve the intermediate values. For TensorFlow, you may need to use [tf.print](https://www.tensorflow.org/api_docs/python/tf/print). For Flax, make sure the model is *not jitted* during the forward pass (refer to this GitHub [Issue](https://github.com/google/jax/issues/196) for more details).

- It is faster to debug with a smaller pretrained checkpoint versus a larger checkpoint where the forward pass takes more than 10 seconds. If only large checkpoints are available, create a dummy model with randomly initialized weights and save those weights to compare against the Transformers implementation.

- Find the easiest way to call the model's forward pass. Ideally, this function (may be called `predict`, `evaluate`, `forward`, or `__call__`) should only call the forward pass *once*. It is more difficult to debug a function that calls the forward pass multiple times.

- Separate tokenization from the forward pass. Locate where a string input is changed to input ids in the forward pass and start here. You may need to create a small script or modify the original code to directly input the input ids instead of an input string.

- Ensure the model is *not* in training mode. This can produce random outputs due to multiple dropout layers in a model. The forward pass in your debugging environment should be *deterministic* so that the dropout layers aren't used.

Once you're able to run the original checkpoint, you're ready to start adapting the model code for Transformers.

### Adapt the model code

The `transformers-cli add-new-model-like` command should have generated a model and configuration file.

- `src/transformers/models/brand_new_bert/modeling_brand_new_bert.py`
- `src/transformers/models/brand_new_bert/configuration_brand_new_bert.py`

The automatically generated code in the `modeling.py` file will have the same architecture as BERT if you answered it's an encoder-only model or it will have the same architecture as BART if you answered it's an encoder-decoder model. The generated code is just a starting point. Based on your research on the new model, you'll need to implement those specific changes by adapting the generated code. This may involve changes to the self-attention layer, the order of the normalization layer, and so on.

At this point, your code doesn't have to be clean or even fully correct! It is more efficiently to quickly create a first draft and then iteratively improve on it. The only thing that matters is that your model should be able to be instantiated from Transformers. The command below creates a model from the configuration with random weights, verifying that the the `__init__` method works.

```py
from transformers import BrandNewBert, BrandNewBertConfig
model = BrandNewBert(BrandNewBertConfig())
```

Random initialization occurs in BrandNewBertPreTrainedModel's `_init_weights` method. All leaf modules are initialized depending on the configuration's variables.

```py
def _init_weights(self, module):
    """Initialize the weights"""
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
```

The initialization scheme can look different if you need to adapt it to your model. For example, [`Wave2Vec2ForPreTraining`] initializes [nn.Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) in its last two linear layers.

The `_is_hf_initialized` flag makes sure the submodule is only initialized once. Setting `module.project_q` and `module.project_hid` to `True` ensures the custom initialization is not overriden later. The `_init_weights` function won't be applied to these modules.

```py
def _init_weights(self, module):
    """Initialize the weights"""
    if isinstance(module, Wav2Vec2ForPreTraining):
        module.project_hid.reset_parameters()
        module.project_q.reset_parameters()
        module.project_hid._is_hf_initialized = True
        module.project_q._is_hf_initialized = True
    elif isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if module.bias is not None:
            module.bias.data.zero_()
```

### Conversion script

The original checkpoint must be converted to a Transformers compatible checkpoint.

> [!TIP]
> Try looking for an existing conversion script to copy, adapt, and reuse for your model!
>
> - If you're porting a model from TensorFlow to PyTorch, a good starting point may be the BERT [conversion script](https://github.com/huggingface/transformers/blob/7acfa95afb8194f8f9c1f4d2c6028224dbed35a2/src/transformers/models/bert/modeling_bert.py#L91).
> - If you're porting a model from PyTorch to PyTorch, a good starting point may be the BART [conversion script](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bart/convert_bart_original_pytorch_checkpoint_to_pytorch.py).

Make sure **all** required weights are initialized and print out all the checkpoint weights that weren't used for initialization to make sure the model has been converted correctly.

You may encounter wrong shape statements of name assignments during the conversion. This is most likely because of incorrect parameters in BrandNewBertConfig, the wrong architecture, a bug in the `init` method of your implementation, or you need to transpose one of the checkpoint weights.

Keep iterating with the [Adapt the model code](#adapt-the-model-code) section until all the checkpoint weights are correctly loaded. Once you can load a checkpoint in your model, save it to a folder. This should contain a `model.safetensors` file and a `config.json` file.

```py
model.save_pretrained("/path/to/converted/checkpoint/folder")
```

To help with conversion, the next section briefly describes how PyTorch models stores and defines layer weights and names.

#### PyTorch layer weights and names

It is helpful to create a basic PyTorch model to understand how layer names are defined and weights are initialized.

```py
from torch import nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(10, 10)
        self.intermediate = nn.Linear(10, 10)
        self.layer_norm = nn.LayerNorm(10)
```

PyTorch layer names are defined by the class attribute name of the layer (dense, intermediate, layer_norm). Create a instance of SimpleModel to fill all the layers with random weights.

```py
model = SimpleModel()
print(model)
SimpleModel(
  (dense): Linear(in_features=10, out_features=10, bias=True)
  (intermediate): Linear(in_features=10, out_features=10, bias=True)
  (layer_norm): LayerNorm((10,), eps=1e-05, elementwise_affine=True)
)
```

The weight values of a specific layer are randomly initialized.

```py
print(model.dense.weight.data)
tensor([[-0.0818,  0.2207, -0.0749, -0.0030,  0.0045, -0.1569, -0.1598,  0.0212,
         -0.2077,  0.2157],
        [ 0.1044,  0.0201,  0.0990,  0.2482,  0.3116,  0.2509,  0.2866, -0.2190,
          0.2166, -0.0212],
        [-0.2000,  0.1107, -0.1999, -0.3119,  0.1559,  0.0993,  0.1776, -0.1950,
         -0.1023, -0.0447],
        [-0.0888, -0.1092,  0.2281,  0.0336,  0.1817, -0.0115,  0.2096,  0.1415,
         -0.1876, -0.2467],
        [ 0.2208, -0.2352, -0.1426, -0.2636, -0.2889, -0.2061, -0.2849, -0.0465,
          0.2577,  0.0402],
        [ 0.1502,  0.2465,  0.2566,  0.0693,  0.2352, -0.0530,  0.1859, -0.0604,
          0.2132,  0.1680],
        [ 0.1733, -0.2407, -0.1721,  0.1484,  0.0358, -0.0633, -0.0721, -0.0090,
          0.2707, -0.2509],
        [-0.1173,  0.1561,  0.2945,  0.0595, -0.1996,  0.2988, -0.0802,  0.0407,
          0.1829, -0.1568],
        [-0.1164, -0.2228, -0.0403,  0.0428,  0.1339,  0.0047,  0.1967,  0.2923,
          0.0333, -0.0536],
        [-0.1492, -0.1616,  0.1057,  0.1950, -0.2807, -0.2710, -0.1586,  0.0739,
          0.2220,  0.2358]]).
```

In the conversion script, the random weights should be replaced with the exact weights from the corresponding layer in the original checkpoint.

```py
# retrieve matching layer weights with recursive algorithm
layer_name = "dense"
pretrained_weight = array_of_dense_layer

model_pointer = getattr(model, "dense")
model_pointer.weight.data = torch.from_numpy(pretrained_weight)
```

Verify the randomly initialized weights and their corresponding pretrained checkpoint weights have the identical **shape** and **name**. Add assert statements for the shape and print out the checkpoint weight names.

```py
assert (
    model_pointer.weight.shape == pretrained_weight.shape
), f"Pointer shape of random weight {model_pointer.shape} and array shape of checkpoint weight {pretrained_weight.shape} mismatched"

logger.info(f"Initialize PyTorch weight {layer_name} from {pretrained_weight.name}")
```

When the shape or name don't match, you may have assigned the incorrect checkpoint weight to a randomly initialized layer. An incorrect shape may be because of an BrandNewBert's parameters don't exactly match the original model's parameters. But it could be that the PyTorch layer implementation requires the weights to be transposed first.

### Implement the forward pass

The forward pass should be implemented next if the model loads correctly. It takes some inputs and returns the model output.

```py
model = BrandNewBertModel.from_pretrained("/path/to/converted/checkpoint/folder")
input_ids = [0, 4, 4, 3, 2, 4, 1, 7, 19]
output = model(input_ids).last_hidden_states
```

Don't be discouraged if your forward pass isn't identical with the output from the original model or if it returns an error! Check that the forward pass doesn't throw any errors. This is often because the dimensions are wrong (dimensionality mismatch) or because the wrong data type is used ([torch.long](https://pytorch.org/docs/stable/generated/torch.Tensor.long.html) instead of [torch.float32](https://pytorch.org/docs/stable/tensors.html)).

Your output should have a precision of *1e-3*. Ensure the output shapes and output values are identical. Common reasons for why the outputs aren't identical include:

- Some layers were not added (activation layer or a residual connection).
- The word embedding matix is not tied.
- The wrong positional embeddings are used because the original implementation includes an offset.
- Dropout is applied during the forward pass. Fix this error by making sure `model.training` is `False` and passing `self.training` to [torch.nn.functional.dropout](https://pytorch.org/docs/stable/nn.functional.html?highlight=dropout#torch.nn.functional.dropout).

Compare the forward pass of the original model and your implementation to check if there are any differences. Ideally, debug and print out the intermediate outputs of both implementations of the forward pass to pinpoint where the original implementation differs from yours.

1. Make sure the hardcoded `input_ids` in both implementations are identical.
2. Verify the outputs of the first transformation of `input_ids` (usually the word embeddings) are identical, and work your way through to the last layer.

Any difference between the two implementations should point to the bug in your implementation.

One of the best strategies is to add many print statements to the same positions in both implementations, and then successively remove them when they output identical values for the intermediate outputs.

When both implementations produce the same output, verify the outputs are within the precision of *1e-3*.

```py
torch.allclose(original_output, output, atol=1e-3)
```

This is typically the most difficult part of the process. Congratulations if you've made it this far! 

And if you're stuck or struggling with this step, don't hesitate to ask for help on your pull request.

### Add model tests

While the model works, you still need to add tests to ensure it is compatible with Transformers and all the tests pass. Tests are important because they help users understand your work by looking at specific tests, and because they prevent your model from breaking in the future if any changes are made.

[Cookiecutter](https://cookiecutter.readthedocs.io/en/stable/) should have added a test file for your model. Run the test file below to make sure all common tests pass.

```bash
pytest tests/models/brand_new_bert/test_modeling_brand_new_bert.py
```

The integration tests should be added first because they serve the same purpose as the debugging scripts you used earlier to implement the new model in Transformers. A template of those model tests, BrandNewBertModelIntegrationTests, was added by Cookiecutter and only needs to be filled out by you. To ensure it passes, run the following command.

<hfoptions id="integration-test">
<hfoption id="macOS">

```bash
RUN_SLOW=1 pytest -sv tests/models/brand_new_bert/test_modeling_brand_new_bert.py::BrandNewBertModelIntegrationTests
```

</hfoption>
<hfoption id="Windows">

```bash
SET RUN_SLOW=1 pytest -sv tests/models/brand_new_bert/test_modeling_brand_new_bert.py::BrandNewBertModelIntegrationTests
```

</hfoption>
</hfoptions>

All features unique to BrandNewBert should be tested in a separate test under `BrandNewBertModelTester/BrandNewBertModelTest`. This test is often overlooked, but it is extremely important because:

- it helps transfer knowledge you acquired during the process to the community by showing how the novel features of the new model works
- future contributors can quickly test changes to the model by running these special tests

### Implement tokenizer

With the model out of the way, time to focus on the tokenizer. The tokenizer should be identical or very similar to an existing tokenizer in Transformers.

Find and load the original tokenizer file into your implementation. Create a script in the original repository that inputs a string and returns the `input_ids`. The pseudocode should look similar to this.

```py
input_str = "This is a long example input string containing special characters .$?-, numbers 2872 234 12 and words."
model = BrandNewBertModel.load_pretrained_checkpoint("/path/to/checkpoint/")
input_ids = model.tokenize(input_str)
```

You may need to search the original repository to find the correct tokenizer function or modify the existing tokenizer in your clone of the original repository to only return the `input_ids`. The script for your tokenizer should look something like this.

```py
from transformers import BrandNewBertTokenizer

input_str = "This is a long example input string containing special characters .$?-, numbers 2872 234 12 and words."
tokenizer = BrandNewBertTokenizer.from_pretrained("/path/to/tokenizer/folder/")
input_ids = tokenizer(input_str).input_ids
```

When both implementations have the same `input_ids`, add a tokenizer test file. This file is analogous to the modeling test files. The tokenizer test files should contain a couple of hardcoded integration tests.

### Run integration tests

Now that you have a model and tokenizer, add end-to-end integration tests using both the model and tokenizer to `tests/models/brand_new_bert/test_modeling_brand-new_bert.py`.

The test should provide a meaningful text-to-text example that the model works as expected. For example, you can include a source-to-target translation pair, an article-to-summary pair, or a question-to-answer pair.

If the checkpoint hasn't been finetuned on a downstream task, then the model tests will suffice.

Finally, try to make sure your tests can run on a GPU by adding `.to(self.device)` statements to the models internal tensors. Don't worry if you don't have access to a GPU, we can take care of that for you if that's the case.

### Add documentation

Your model is only useful if users know how to use it. This is why it's important to add documentation and docstrings. Cookiecutter added a template file, `docs/source/model_doc/brand_new_bert.md`, that you can fill out with information about your model.

This is generally a user's first interaction with a model, so the documentation should be clear and concise. It is often very useful to add examples of how the model should be used.

Make sure docstrings are added to `src/transformers/models/brand_new_bert/modeling_brand_new_bert/py` and includes all necessary inputs and outputs. Review our [guide](https://github.com/huggingface/transformers/tree/main/docs#writing-documentation---specification) for writing documentation and docstrings.

### Refactor

Time to tidy things up and make sure the code style is consistent with the rest of the library. Run the following command to automatically fix incorrect styles.

```bash
make style
```

To verify the code style passes quality checks, run the command below.

```bash
make quality
```

There may be other failing tests or checks (missing docstring or incorrect naming) on your pull request due to Transformers strict design tests. We can help you with these issues if you're stuck.

After ensuring the code runs correctly, you may want to refactor it to make it more readable or cleaner.

### Upload to the Hub

Convert and upload all checkpoints to the [Hub](https://hf.co/models). Add a model card to provide more transparency and context about the model. The model card should highlight specific characteristics of a checkpoint, how the model was trained, and code examples of how to use it.

> [!TIP]
> In many cases, adding an interactive notebook users can run is a great way to showcase how to use the model for inference or finetune it on a downstream task. While not mandatory, including a notebook can drive greater adoption of your model.

You should also consult with the Transformers team to decide on an appropriate name for the model, and getting the required access rights to upload the model.

Use the [`~PreTrainedModel.push_to_hub`] method to upload the model.

```py
brand_new_bert.push_to_hub("brand_new_bert")
```

Refer to the [Share](./model_sharing) guide for more information about uploading models to the Hub.

### Merge your model

You're finally ready to merge your pull request and officially add the model to Transformers! Make sure all the tests are passing and all comments and feedback have been addressed.

Congratulations on adding a new model to Transformers! 🥳

This is a very significant contribution. Your work here makes Transformers more accessible to developers and researchers around the world. You should be proud of your contribution and share your accomplishment with the community!