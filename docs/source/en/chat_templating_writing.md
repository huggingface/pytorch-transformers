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

# Template writing

A chat template is a [Jinja](https://jinja.palletsprojects.com/en/3.1.x/templates/) template stored in the tokenizers [chat_template](https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizer.chat_template) attribute. Jinja is a templating language that allows you to write Python-like code and syntax. A chat template performs the following three roles.

1. Print the role enclosed in `<|` and `|>` (`<|user|>`, `<|assistant|>`, etc.).
2. Print the message followed by an end-of-sequence (EOS) token.
3. Print the assistant token if [add_generation_prompt=True](./chat_templating#add_generation_prompt) so the model knows to generate an assistant response.

An example template is shown below.

```jinja
{%- for message in messages %}
    {{- '<|' + message['role'] + |>\n' }}
    {{- message['content'] + eos_token }}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|assistant|>\n' }}
{%- endif %}
```

The template can be customized to handle more complex use cases. This guide will show you how to add and edit templates and some template writing tips.

## Create a template

Create a template by writing a Jinja template and then setting it as the chat template in the tokenizer. For example, the template below adds `[ASST]` and `[/ASST]` tags to the assistant messages.

```jinja
{%- for message in messages %}
    {%- if message['role'] == 'user' %}
        {{- bos_token + '[INST] ' + message['content'].strip() + ' [/INST]' }}
    {%- elif message['role'] == 'system' %}
        {{- '<<SYS>>\\n' + message['content'].strip() + '\\n<</SYS>>\\n\\n' }}
    {%- elif message['role'] == 'assistant' %}
        {{- '[ASST] '  + message['content'] + ' [/ASST]' + eos_token }}
    {%- endif %}
{%- endfor %}
```

Set the template in the tokenizer, and the next time you use [`~PreTrainedTokenizerBase.apply_chat_template`], the new template is used.

```py
template = tokenizer.chat_template
template = template.replace("SYS", "SYSTEM")  # Change the system token
tokenizer.chat_template = template  # Set the new template
```

The template is saved in the `tokenizer_config.json` file. Save it to the Hub with [`~PushToHubMixin.push_to_hub`] so you can reuse it later and make sure everyone is using the right template for your model.

```py
tokenizer.push_to_hub("model_name")
```

## Template writing tips

The easiest way to start writing Jinja templates is to refer to existing templates. Use `print(tokenizer.chat_template)` on any chat model to see what template it's using. Try starting with simple models that don't call any tools or support RAG. Finally, take a look at the [Jinja documentation](https://jinja.palletsprojects.com/en/3.1.x/templates/#synopsis) for more details about formatting and syntax.

This section curates some best practices for writing clean and efficient Jinja templates.

### Trimming whitespace

Jinja prints any whitespace before or after a block of text. This can be an issue for chat templates because whitespace usage should be intentional. Add `-` to strip any whitespace before a block.

```jinja
{%- for message in messages %}
    {{- message['role'] + message['content'] }}
{%- endfor %}
```

The incorrect whitespace usage example below may introduce a newline and indentation in the output.

```jinja
{% for message in messages %}
    {{ message['role'] + message['content'] }}
{% endfor %}
```

### Special variables

There are five special variables available inside a template. You can pass virtually any additional arguments to [`~PreTrainedTokenizerBase.apply_chat_template`] and it will be available inside the template as a variable. However, you should try to keep the number of variables to the five below to make it easier for users to use the chat model without writing custom code to handle model-specific arguments.

- `messages` contains the chat history as a list of message dicts.
- `tools` contains a list of tools in JSON schema format.
- `documents` contains a list of documents with the format `{"title": Title, "contents": "Contents"}` (designed for RAG models).
- `add_generation_prompt` is a boolean that determines whether to add an assistant header at the end of the conversation.
- `bos_token` and `eos_token` are special tokens extracted from a tokenizers `special_tokens_map`.

### Callable functions

There are two callable functions available inside a template.

- `raise_exception(msg)` raises a `TemplateException`. This is useful for debugging or warning users about incorrect template usage.
- `strftime_now(format_str)` retrieves the current date and time in a specific format which could be useful to include in system messages. It is equivalent to [datetime.now().strftime(format_str)](https://docs.python.org/3/library/datetime.html#datetime.datetime.now) in Python.

### Compatibility with non-Python Jinja

Jinka is implemented in multiple languages. and they generally have the same syntax. Writing a template in Python allows you to use Python methods such as [lower](https://docs.python.org/3/library/stdtypes.html#str.lower) on strings or [items](https://docs.python.org/3/library/stdtypes.html#dict.items) on dicts. But this won't work if the template is used in a non-Python implementation, for example, when deploying with Javascript or Rust.

Make the changes below to ensure compatibility across all Jinja implementations.

- Replace Python methods with Jinja filters. For example, replace `string.lower()` with `string|lower` or `dict.items()` with `dict|dictitems`. Most of the changes follow the same pattern except `string.strip()`, which is replaced with `string|trim`. Refer to the list of [built-in filters](https://jinja.palletsprojects.com/en/3.1.x/templates/#builtin-filters) for a complete list of filters.
- Replace `True`, `False`, and `None` (these are Python specific) with `true`, `false`, and `none` respectively.
- Directly rendering a dict or list may return different results in other implementations. For example, string entries may change from single-quote to double-quote. To avoid this, add the [tojson](https://jinja.palletsprojects.com/en/3.1.x/templates/#jinja-filters.tojson) filter to maintain consistency.

### Big templates

Newer models or models with features like [tool-calling](./chat_extras#tools) and [RAG](./chat_extras#retrieval-augmented-generation-rag) require larger templates than can be more than 100 lines. It may be easier to write larger templates in a separate file. The line numbers in the separate file corresponds exactly to the line numbers in template parsing or execution errors, making it easier to debug any potential issues.

Write the template in a separate file and extract it to the chat template.

```py
open("template.jinja", "w").write(tokenizer.chat_template)
```

You could also load an edited template back into the tokenizer.

```py
tokenizer.chat_template = open("template.jinja").read()
```

## Contribute

Add a chat template by setting the `chat_template` attribute in the tokenizer and testing it with [`~PreTrainedTokenizerBase.apply_chat_template`]. If it works as expected, then you can upload it to the Hub with with [`~PushToHubMixin.push_to_hub`].

Even if you're not the model owner, it is still helpful to add a template for a model with an empty chat template or a model that is using a default class template. Open a [pull request](https://hf.co/docs/hub/repositories-pull-requests-discussions) on the model repository to add the template.

```py
tokenizer.chat_template = template
tokenizer.push_to_hub("model_name")
```
