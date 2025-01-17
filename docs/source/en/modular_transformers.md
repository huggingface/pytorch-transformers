# Modular Transformers

Modular Transformers lowers the bar for contributing models and significantly reduces the code required to add a model by allowing imports and inheritance.

One of Transformers' core design feature is the [single model, single file](https://huggingface.co/blog/transformers-design-philosophy) policy. Model components - such as attention layers - are repeated across many files and any independent implementations tend to diverge as fixes and changes are applied to specific parts of the code.

The [`# Copied from`](./pr_checks#check-copies) statements prevents the code from diverging, and it is enforced by our continuous integration tests and local commands. The downside is that this approach is tedious and adds significantly more lines of code, most of which is boilerplate.

## Motivation

Modular Transformers addresses these issues by adding a *modular* file to a model folder. The modular file can import code from other models and inherit code from other classes unlike traditional modeling and processing files.

> [!TIP]
> Modular Transformers isn't meant to replace the modeling code, and if your model isn't based on an existing model, you'll need to add a `modeling.py` file manually.

A modular file contains model, processor, and configuration class code that would otherwise be in separate files under the single model, single file policy.

Model users still import and use the single-file interface they've grown familiar with. In doing so, we hope to enable simpler contributions while sticking to our philosophy.

## Create a modeling.py file

A linter "unravels" the modular file into a `modeling.py` file to preserve the single model, single file directory structure (modeling, processor, etc.). Inheritance is flattened to only a **single** level.

Run the command below to automatically generate a `modeling.py` file from a modular file.

```bash
python utils/modular_model_converter.py --files-to-parse src/transformers/models/<your_model>/modular_<your_model>.py
```

For example:

- If a configuration class inherits from another class, but adds and deletes an argument, the generated file directly references it if an argument is added or completely removes it if an argument is deleted.
- If a class inherits from another, like `GemmaModel(LlamaModel)`, the dependencies are automatically inferred. All submodules are also automatically inferred from the superclass.
- If a new function is defined in the modular file and used inside classes, the linter automatically infers these as well.

You should be able to write everything (tokenizer, image processor, model, config, etc.) in a modular and their corresponding single-files are generated.

Run the command below to ensure the generated content matches `modular_<your_model>.py`.

```bash
python utils/check_modular_conversion.py --files src/transformers/models/<your_model>/modular_<your_model>.py
```

The example below demonstrates how a model can be added with significantly fewer lines of code with Modular Transformers.

### BERT and RoBERTa

BERT and RoBERTa, two very similar models, differ solely in how the embedding layer is implemented.

Instead of redefining the model entirely, consider the `modular_roberta.py` file shown below for the modeling and configuration classes (the tokenizer isn't shown in this example).

```py
from torch import nn
from ..bert.configuration_bert import BertConfig
from ..bert.modeling_bert import (
    BertModel,
    BertEmbeddings,
    BertForMaskedLM
)

# RoBERTa and BERT config is identical
class RobertaConfig(BertConfig):
  model_type = 'roberta'

# Redefine the embeddings to highlight the padding id difference, and redefine the position embeddings
class RobertaEmbeddings(BertEmbeddings):
    def __init__(self, config):
        super().__init__(config())

        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

# RoBERTa and BERT model is identical except for the embedding layer, which is defined above, so no need for additional changes here
class RobertaModel(BertModel):
  def __init__(self, config):
    super().__init__(config)
    self.embeddings = RobertaEmbeddings(config)

      
# The model heads now only need to redefine the model inside to `RobertaModel`
class RobertaForMaskedLM(BertForMaskedLM):
  def __init__(self, config):
    super().__init__(config)
    self.model = RobertaModel(config)
```

If you don't use the defined dependency, you'll receive the following error.

```
ValueError: You defined `RobertaEmbeddings` in the modular_roberta.py, it should be used when you define `BertModel`, as it is one of it's direct dependencies. Make sure you use it in the `__init__` function.
```

## Removing attributes and functions

Use `del` to remove attributes that aren't used in your model or if you don't want to include it in the unravelled `modeling.py` file. The example [`GemmaModel`] below removes the `embed_tokens` from the original [`LlamaModel`] it inherits from.

```py
class GemmaModel(LlamaModel):                 |           class GemmaModel(PreTrainedModel):
    def __init__(self, config):               |              def __init__(self, config):
        super().__init__(self, eos_token)     |                 super().__init__(config)
        del self.embed_tokens                 |                 self.padding_idx = config.pad_token_id
                                              |                 self.vocab_size = config.vocab_size
                                              |
                                              |                 self.layers = nn.ModuleList(
                                              |                     [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
                                              |                 )
                                              |                 self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
                                              |                 self.rotary_emb = LlamaRotaryEmbedding(config=config)
                                              |                 self.gradient_checkpointing = False
                                              |                 
                                              |                 # Initialize weights and apply final processing
                                              |                 self.post_init()
```

Remove a function by writing it with a `raise AttributeError("")` to mimic the behavior you actually want when you remove a parent function in Python.

```py
class GemmaTokenizer(LlamaTokenizer):
    ...

    def get_spm_processor(self):
        raise AttributeError("Not needed for Gemma")

    def unk_token_length(self):
        raise AttributeError("Not needed for Gemma")
```

## Define new functions

New functions can be defined in the modular file and used inside a class. The new function - and recursively, any other new function called in its body - is automatically copy-pasted in the file where it is used.

```py
def my_new_function(*args, **kwargs):
  # Do something here
  pass

class GemmaModel(LlamaModel):
    def forward(*args, **kwargs):
      # Call the function
      example = my_new_function(*args, **kwargs)
      # Continue here
```

## Calling super()

You don't have to unravel a call to `super()` or if you want to differentiate which `super().__init__()` call you're doing.

The example below shows how you only need to add `eos_token` to the `__init__` instead of calling `super().__init__(eos_token)`.

```py
class GemmaTokenizer(LlamaTokenizer, PretrainedTokenizerFast):         |           class GemmaModel(nn.Module):
    def __init__(self, eos_token="</s>"):                              |             def __init__(self):
        eos_token = AddedToken(eos_token)                              |                eos_token = AddedToken(eos_token)
        PretrainedTokenizerFast.__init__(self, eos_token)              |                super().__init__(eos_token)
```

## Special naming

Special naming for classes is also supported, which is useful for composite models.

The example below shows how you can use `GemmaVisionModel` even though it's not the same as the modular Gemma model.

```py
class GemmaVisionModel(CLIPModel):                                 
    pass
```
