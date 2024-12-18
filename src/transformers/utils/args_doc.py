import inspect
from functools import wraps
from typing import get_args

import regex as re

from .doc import PT_SAMPLE_DOCSTRINGS, _prepare_output_docstrings
from .generic import ModelOutput


class ModelArgs:
    labels = r""" of shape `(batch_size, sequence_length)`:
        Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
        config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
        (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
    """

    num_logits_to_keep = r""":
        Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
        `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
        token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
    """

    input_ids = r""":
        Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.
        Indices can be obtained using `AutoTokenizer`. See `PreTrainedTokenizer.encode` and
        `PreTrainedTokenizer.__call__` for details.

        [What are input IDs?](../glossary#input-ids)
    """

    attention_mask = r""" of shape `(batch_size, sequence_length)`:
        Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

        - 1 for tokens that are **not masked**,
        - 0 for tokens that are **masked**.

        [What are attention masks?](../glossary#attention-mask)

        Indices can be obtained using `AutoTokenizer`. See `PreTrainedTokenizer.encode` and
        `PreTrainedTokenizer.__call__` for details.
    """

    position_ids = r""":
        Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

        [What are position IDs?](../glossary#position-ids)
    """

    past_key_values = r""":
        Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
        blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
        returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

        Two formats are allowed:
            - a `~cache_utils.Cache` instance, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache);
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

        The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
        legacy cache format will be returned.

        If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
        have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
        of shape `(batch_size, sequence_length)`.
    """

    past_key_value = r""":deprecated in favor of `past_key_values`"""

    inputs_embeds = r""":
        Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
        is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
        model's internal embedding lookup matrix.
    """

    use_cache = r""":
        If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
        `past_key_values`).
    """

    output_attentions = r""":
        Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
        tensors for more detail.
    """

    output_hidden_states = r""":
        Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
        more detail.
    """

    return_dict = r""":
        Whether or not to return a `~utils.ModelOutput` instead of a plain tuple.
    """

    cache_position = r""":
        Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
        this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
        the complete sequence length.
    """

    hidden_states = r""": input to the layer of shape `(batch, seq_len, embed_dim)"""

    position_embeddings = r""":
        Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
        with `head_dim` being the embedding dimension of each attention head.
    """

    config = r""":
        Model configuration class with all the parameters of the model. Initializing with a config file does not
        load the weights associated with the model, only the configuration. Check out the
        [`~PreTrainedModel.from_pretrained`] method to load the model weights.
    """

    start_positions = r""" of shape `(batch_size,)`:
        Labels for position (index) of the start of the labelled span for computing the token classification loss.
        Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
        are not taken into account for computing the loss.
    """

    end_positions = r""" of shape `(batch_size,)`:
        Labels for position (index) of the end of the labelled span for computing the token classification loss.
        Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
        are not taken into account for computing the loss.
    """

    output_router_logits = r"""
        Whether or not to return the logits of all the routers. They are useful for computing the router loss, and
        should not be returned during inference.
    """


class ClassDocstring:
    PreTrainedModel = r"""
        This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
        library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
        etc.)

        This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
        Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
        and behavior.
    """

    Model = r"""
        The bare {model_name} Model outputting raw hidden-states without any specific head on top."""

    ForSequenceClassification = r"""
        The {model_name} Model with a sequence classification head on top (linear layer).

        [`LlamaForSequenceClassification`] uses the last token in order to do the classification, as other causal models
        (e.g. GPT-2) do.

        Since it does classification on the last token, it requires to know the position of the last token. If a
        `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
        no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
        padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
        each row of the batch).
    """

    ForQuestionAnswering = r"""
        The {model_name} transformer with a span classification head on top for extractive question-answering tasks like
        SQuAD (a linear layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """

    ForTokenClassificatio = r"""
        The {model_name} transformer with a token classification head on top (a linear layer on top of the hidden-states
        output) e.g. for Named-Entity-Recognition (NER) tasks.
    """

    Config = r"""
    This is the configuration class to store the configuration of a [`{model_name}Model`] or a [`TF{model_name}Model`]. It is
    used to instantiate a DeBERTa model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the {model_name}
    [{}](https://huggingface.co/{model_checkpoint}) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    """


class ClassAttrs:
    base_model_prefix = r"""TODO """
    supports_gradient_checkpointing = r"""TODO """
    _no_split_module = r"""TODO """
    _skip_keys_device_placement = r"""TODO """
    _supports_flash_attn_2 = r"""TODO """
    _supports_sdpa = r"""TODO """
    _supports_cache_class = r"""TODO """
    _supports_quantized_cache = r"""TODO """
    _supports_static_cache = r"""TODO """
    _init_weights = r"""TODO """


ARGS_TO_IGNORE = {"self", "kwargs", "args", "deprecated_arguments"}


def get_indent_level(func):
    # Get the source code of the function
    source_code = inspect.getsource(func)

    # Get the first line of the source (the function definition)
    first_line = source_code.splitlines()[0]

    # Calculate the indentation level (number of spaces at the start)
    indent_level = len(first_line) - len(first_line.lstrip())

    return indent_level


def parse_docstring(docstring):
    args_pattern = args_pattern = re.compile(r"(Args:)(\n.*)?(\n)?$", re.DOTALL)

    args_match = args_pattern.search(docstring)
    args_section = args_match.group(2).lstrip("\n") if args_match else None

    params = {}
    if args_section:
        param_pattern = re.compile(
            r"^\s*(\w+)\s*\((.*?)\):\s*(.*?)(?=\n^\s*\w+\s*\(|\n\s*$)", re.DOTALL | re.MULTILINE
        )
        for match in param_pattern.finditer(args_section):
            param_name = match.group(1)
            param_type = match.group(2)
            param_description = match.group(3).strip()
            params[param_name] = {"type": param_type, "description": param_description}
    docstring = re.sub(r"Args:[\S\s]*(?=Example|Return)", "", docstring)
    return params, docstring


def contains_type(type_hint, target_type) -> bool:
    args = get_args(type_hint)
    if args == ():
        try:
            return issubclass(type_hint, target_type)
        except Exception as _:
            return issubclass(type(type_hint), target_type)
    _type = any(contains_type(arg, target_type) for arg in args)
    return _type


def auto_docstring(func):
    """
    Wrapper that automatically generates docstring using ARG_TO_DOC.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    # Use inspect to retrieve the function's signature
    sig = inspect.signature(func)
    indent_level = get_indent_level(func)
    # Build the docstring dynamically
    docstring = "Args:\n"
    # Adding description for each parameter from ARG_TO_DOC
    undocumented_parameters = []
    documented_params = set()

    func_documentation = func.__doc__
    if func_documentation is not None:
        documented_params, func_documentation = parse_docstring(func_documentation)

    for param_name, param in sig.parameters.items():
        if param_name in ModelArgs.__dict__:
            if param.annotation != inspect.Parameter.empty:
                param_type = param.annotation
                if "typing" in str(param_type):
                    param_type = "".join(str(param_type).split("typing.")).replace("transformers.", "~")
                else:
                    param_type = f"{param_type.__module__.replace('transformers.','~').replace('builtins','')}.{param.annotation.__name__}"
            else:
                param_type = ""
            # Check if the parameter has a default value (considered optional)
            # is_optional = param.default != inspect.Parameter.empty
            param_default = ""
            if param.default != inspect._empty and param.default is not None:
                param_default = f", defaults to `{str(param.default)}`"
            indented_doc = getattr(ModelArgs, param_name)  # .replace("\n    ", "\n")
            docstring += f"{' '*indent_level}{param_name} (`{param_type}`{param_default}){indented_doc}\n"
        elif param_name in ARGS_TO_IGNORE:
            continue
        elif param_name in documented_params:
            docstring += f"{' '*indent_level}{param_name} {documented_params[param_name]}\n"
        else:
            undocumented_parameters.append(
                f"🚨 `{param_name}` is part of {func.__qualname__}'s signature, but not documented. Make sure to add it to the docstring of the function in {func.__code__.co_filename}."
            )

    model_class = func.__qualname__.split(".")[0]
    task = rf"({'|'.join(PT_SAMPLE_DOCSTRINGS.keys())})"
    model_task = re.search(task, model_class)
    example_annotation = ""
    if model_task is not None:
        task = model_task.group()
        example_annotation = PT_SAMPLE_DOCSTRINGS[task]

    config_class = "".join([k.title() for k in func.__module__.split(".")[-1].split("_")[1:]]) + "Config"
    # return_annotation = sig.return_annotation
    # return_type =  f"{return_annotation.__module__.replace("transformers.","~").replace("builtins","")}.{return_annotation.__name__}"
    if sig.return_annotation is not None and sig.return_annotation != inspect._empty:
        return_docstring = _prepare_output_docstrings(
            sig.return_annotation, config_class, add_intro=contains_type(sig.return_annotation, ModelOutput)
        )
        docstring += return_docstring

    docstring += example_annotation

    if len(undocumented_parameters) > 0:
        print("\n".join(undocumented_parameters))
    if func_documentation is not None:
        docstring += func_documentation
    # Assign the dynamically generated docstring to the wrapper function
    wrapper.__doc__ = docstring
    return wrapper


def auto_class_docstring(cls):
    """
    Wrapper that automatically generates a docstring for classes based on their attributes and methods.
    """
    docstring = "DUMMUY DOCSTRING YOU DUMB"
    indent_level = get_indent_level(cls) + 8

    name = re.findall(rf"({'|'.join(ClassDocstring.__dict__.keys())})", cls.__name__)
    if name == [] and cls.__doc__ is None:
        raise ValueError(
            f"`{cls.__name__}` is not part of the auto doc. Here are the available classes: {ClassDocstring.__dict__.keys()}"
        )
    if name != []:
        name = name[0]
        path = inspect.getsourcefile(cls)
        model_name = "".join([k.title() for k in path[:-3].split("/")[-1].split("_")[1:]])
        pre_block = getattr(ClassDocstring, name).format(model_name=model_name, model_checkpoint="dummy-path")
        # Start building the docstring
        docstring = f"{pre_block}\n\n"
        attr_docs = ""
        # Get all attributes and methods of the class
        for attr_name, attr_value in cls.__dict__.items():
            if not callable(attr_value) and not attr_name.startswith("__"):
                if attr_value.__class__.__name__ == "property":
                    attr_type = "property"
                else:
                    attr_type = type(attr_value).__name__
                if "Config" in name:
                    raise ValueError("Config should have explicit docstring")
                attribute_mapping = ClassAttrs
                indented_doc = getattr(attribute_mapping, attr_name, "")
                attr_docs += f"{' ' * (indent_level+4)}{attr_name} (`{attr_type}`): {indented_doc}\n"
        if len(attr_docs.replace(" ", "")):
            docstring += f"{' ' * indent_level}Attributes:\n" + attr_docs
    else:
        print(
            f"You used `@auto_class_docstring` decorator on `{cls.__name__}` but this class is not part of the AutoMappings. Remove the decorator"
        )
    # Assign the dynamically generated docstring to the wrapper class
    if cls.__doc__ is not None:
        docstring += cls.__doc__
    cls.__doc__ = docstring
    # cls.__name__ = cls.__name__
    return cls
