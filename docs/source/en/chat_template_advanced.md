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

# Advanced Usage and Customizing Your Chat Templates

In this page, we’ll explore more advanced techniques for working with chat templates in Transformers. Whether you’re looking to write your own templates, create custom components, or optimize your templates for efficiency, we’ll cover everything you need to take your templates to the next level. Let’s dive into the tools and strategies that will help you get the most out of your chat models.


## How do chat templates work?

The chat template for a model is stored on the `tokenizer.chat_template` attribute. Let's take a look at a `Zephyr` chat template, though note this
one is a little simplified from the actual one!

```
{%- for message in messages %}
    {{- '<|' + message['role'] + '|>\n' }}
    {{- message['content'] + eos_token }}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|assistant|>\n' }}
{%- endif %}
```

If you've never seen one of these before, this is a [Jinja template](https://jinja.palletsprojects.com/en/3.1.x/templates/).
Jinja is a templating language that allows you to write simple code that generates text. In many ways, the code and
syntax resembles Python. In pure Python, this template would look something like this:

```python
for message in messages:
    print(f'<|{message["role"]}|>')
    print(message['content'] + eos_token)
if add_generation_prompt:
    print('<|assistant|>')
```

Effectively, the template does three things:
1. For each message, print the role enclosed in `<|` and `|>`, like `<|user|>` or `<|assistant|>`.
2. Next, print the content of the message, followed by the end-of-sequence token.
3. Finally, if `add_generation_prompt` is set, print the assistant token, so that the model knows to start generating
   an assistant response.

This is a pretty simple template but Jinja gives you a lot of flexibility to do more complex things! Let's see a Jinja
template that can format inputs similarly to the way LLaMA formats them (note that the real LLaMA template includes 
handling for default system messages and slightly different system message handling in general - don't use this one 
in your actual code!)

```
{%- for message in messages %}
    {%- if message['role'] == 'user' %}
        {{- bos_token + '[INST] ' + message['content'] + ' [/INST]' }}
    {%- elif message['role'] == 'system' %}
        {{- '<<SYS>>\\n' + message['content'] + '\\n<</SYS>>\\n\\n' }}
    {%- elif message['role'] == 'assistant' %}
        {{- ' '  + message['content'] + ' ' + eos_token }}
    {%- endif %}
{%- endfor %}
```

Hopefully if you stare at this for a little bit you can see what this template is doing - it adds specific tokens like
`[INST]` and `[/INST]` based on the role of each message. User, assistant and system messages are clearly
distinguishable to the model because of the tokens they're wrapped in.


## How do I create a chat template?

Simple, just write a jinja template and set `tokenizer.chat_template`. You may find it easier to start with an 
existing template from another model and simply edit it for your needs! For example, we could take the LLaMA template
above and add "[ASST]" and "[/ASST]" to assistant messages:

```
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

Now, simply set the `tokenizer.chat_template` attribute. Next time you use [`~PreTrainedTokenizer.apply_chat_template`], it will
use your new template! This attribute will be saved in the `tokenizer_config.json` file, so you can use
[`~utils.PushToHubMixin.push_to_hub`] to upload your new template to the Hub and make sure everyone's using the right
template for your model!

```python
template = tokenizer.chat_template
template = template.replace("SYS", "SYSTEM")  # Change the system token
tokenizer.chat_template = template  # Set the new template
tokenizer.push_to_hub("model_name")  # Upload your new template to the Hub!
```

The method [`~PreTrainedTokenizer.apply_chat_template`] which uses your chat template is called by the [`TextGenerationPipeline`] class, so 
once you set the correct chat template, your model will automatically become compatible with [`TextGenerationPipeline`].

<Tip>
If you're fine-tuning a model for chat, in addition to setting a chat template, you should probably add any new chat
control tokens as special tokens in the tokenizer. Special tokens are never split, 
ensuring that your control tokens are always handled as single tokens rather than being tokenized in pieces. You 
should also set the tokenizer's `eos_token` attribute to the token that marks the end of assistant generations in your
template. This will ensure that text generation tools can correctly figure out when to stop generating text.
</Tip>


## Why do some models have multiple templates?

Some models use different templates for different use cases. For example, they might use one template for normal chat
and another for tool-use, or retrieval-augmented generation. In these cases, `tokenizer.chat_template` is a dictionary.
This can cause some confusion, and where possible, we recommend using a single template for all use-cases. You can use
Jinja statements like `if tools is defined` and `{% macro %}` definitions to easily wrap multiple code paths in a
single template.

When a tokenizer has multiple templates, `tokenizer.chat_template` will be a `dict`, where each key is the name
of a template. The `apply_chat_template` method has special handling for certain template names: Specifically, it will
look for a template named `default` in most cases, and will raise an error if it can't find one. However, if a template
named `tool_use` exists when the user has passed a `tools` argument, it will use that instead. To access templates
with other names, pass the name of the template you want to the `chat_template` argument of
`apply_chat_template()`.

We find that this can be a bit confusing for users, though - so if you're writing a template yourself, we recommend
trying to put it all in a single template where possible!


## What template should I use?

When setting the template for a model that's already been trained for chat, you should ensure that the template
exactly matches the message formatting that the model saw during training, or else you will probably experience
performance degradation. This is true even if you're training the model further - you will probably get the best 
performance if you keep the chat tokens constant. This is very analogous to tokenization - you generally get the
best performance for inference or fine-tuning when you precisely match the tokenization used during training.

If you're training a model from scratch, or fine-tuning a base language model for chat, on the other hand,
you have a lot of freedom to choose an appropriate template! LLMs are smart enough to learn to handle lots of different
input formats. One popular choice is the `ChatML` format, and this is a good, flexible choice for many use-cases. 
It looks like this:

```
{%- for message in messages %}
    {{- '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}
{%- endfor %}
```

If you like this one, here it is in one-liner form, ready to copy into your code. The one-liner also includes
handy support for [generation prompts](#what-are-generation-prompts), but note that it doesn't add BOS or EOS tokens!
If your model expects those, they won't be added automatically by `apply_chat_template` - in other words, the
text will be tokenized with `add_special_tokens=False`. This is to avoid potential conflicts between the template and
the `add_special_tokens` logic. If your model expects special tokens, make sure to add them to the template!

```python
tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
```

This template wraps each message in `<|im_start|>` and `<|im_end|>` tokens, and simply writes the role as a string, which
allows for flexibility in the roles you train with. The output looks like this:

```text
<|im_start|>system
You are a helpful chatbot that will do its best not to say anything so stupid that people tweet about it.<|im_end|>
<|im_start|>user
How are you?<|im_end|>
<|im_start|>assistant
I'm doing great!<|im_end|>
```

The "user", "system" and "assistant" roles are the standard for chat, and we recommend using them when it makes sense,
particularly if you want your model to operate well with [`TextGenerationPipeline`]. However, you are not limited
to these roles - templating is extremely flexible, and any string can be a role.

## I want to add some chat templates! How should I get started?

If you have any chat models, you should set their `tokenizer.chat_template` attribute and test it using
[`~PreTrainedTokenizer.apply_chat_template`], then push the updated tokenizer to the Hub. This applies even if you're
not the model owner - if you're using a model with an empty chat template, or one that's still using the default class
template, please open a [pull request](https://huggingface.co/docs/hub/repositories-pull-requests-discussions) to the model repository so that this attribute can be set properly!

Once the attribute is set, that's it, you're done! `tokenizer.apply_chat_template` will now work correctly for that
model, which means it is also automatically supported in places like `TextGenerationPipeline`!

By ensuring that models have this attribute, we can make sure that the whole community gets to use the full power of
open-source models. Formatting mismatches have been haunting the field and silently harming performance for too long - 
it's time to put an end to them!


<Tip>

The easiest way to get started with writing Jinja templates is to take a look at some existing ones. You can use
`print(tokenizer.chat_template)` for any chat model to see what template it's using. In general, models that support tool use have 
much more complex templates than other models - so when you're just getting started, they're probably a bad example
to learn from! You can also take a look at the 
[Jinja documentation](https://jinja.palletsprojects.com/en/3.1.x/templates/#synopsis) for details
of general Jinja formatting and syntax.

</Tip>

Jinja templates in `transformers` are identical to Jinja templates elsewhere. The main thing to know is that 
the conversation history will be accessible inside your template as a variable called `messages`.  
You will be able to access `messages` in your template just like you can in Python, which means you can loop over 
it with `{% for message in messages %}` or access individual messages with `{{ messages[0] }}`, for example.

You can also use the following tips to write clean, efficient Jinja templates:

### Trimming whitespace

By default, Jinja will print any whitespace that comes before or after a block. This can be a problem for chat
templates, which generally want to be very precise with whitespace! To avoid this, we strongly recommend writing
your templates like this:

```
{%- for message in messages %}
    {{- message['role'] + message['content'] }}
{%- endfor %}
```

rather than like this:

```
{% for message in messages %}
    {{ message['role'] + message['content'] }}
{% endfor %}
```

Adding `-` will strip any whitespace that comes before the block. The second example looks innocent, but the newline
and indentation may end up being included in the output, which is probably not what you want!

### Special variables

Inside your template, you will have access several special variables. The most important of these is `messages`, 
which contains the chat history as a list of message dicts. However, there are several others. Not every
variable will be used in every template. The most common other variables are:

- `tools` contains a list of tools in JSON schema format. Will be `None` or undefined if no tools are passed.
- `documents` contains a list of documents in the format `{"title": "Title", "contents": "Contents"}`, used for retrieval-augmented generation. Will be `None` or undefined if no documents are passed.
- `add_generation_prompt` is a bool that is `True` if the user has requested a generation prompt, and `False` otherwise. If this is set, your template should add the header for an assistant message to the end of the conversation. If your model doesn't have a specific header for assistant messages, you can ignore this flag.
- **Special tokens** like `bos_token` and `eos_token`. These are extracted from `tokenizer.special_tokens_map`. The exact tokens available inside each template will differ depending on the parent tokenizer.

<Tip>

You can actually pass any `kwarg` to `apply_chat_template`, and it will be accessible inside the template as a variable. In general,
we recommend trying to stick to the core variables above, as it will make your model harder to use if users have
to write custom code to pass model-specific `kwargs`. However, we're aware that this field moves quickly, so if you
have a new use-case that doesn't fit in the core API, feel free to use a new `kwarg` for it! If a new `kwarg`
becomes common we may promote it into the core API and create a standard, documented format for it.

</Tip>

### Callable functions

There is also a short list of callable functions available to you inside your templates. These are:

- `raise_exception(msg)`: Raises a `TemplateException`. This is useful for debugging, and for telling users when they're
doing something that your template doesn't support.
- `strftime_now(format_str)`: Equivalent to `datetime.now().strftime(format_str)` in Python. This is used for getting
the current date/time in a specific format, which is sometimes included in system messages.

### Compatibility with non-Python Jinja

There are multiple implementations of Jinja in various languages. They generally have the same syntax,
but a key difference is that when you're writing a template in Python you can use Python methods, such as
`.lower()` on strings or `.items()` on dicts. This will break if someone tries to use your template on a non-Python
implementation of Jinja. Non-Python implementations are particularly common in deployment environments, where JS
and Rust are very popular. 

Don't panic, though! There are a few easy changes you can make to your templates to ensure they're compatible across
all implementations of Jinja:

- Replace Python methods with Jinja filters. These usually have the same name, for example `string.lower()` becomes
  `string|lower`, and `dict.items()` becomes `dict|items`. One notable change is that `string.strip()` becomes `string|trim`.
  See the [list of built-in filters](https://jinja.palletsprojects.com/en/3.1.x/templates/#builtin-filters)
  in the Jinja documentation for more.
- Replace `True`, `False` and `None`, which are Python-specific, with `true`, `false` and `none`.
- Directly rendering a dict or list may give different results in other implementations (for example, string entries
  might change from single-quoted to double-quoted). Adding the `tojson` filter can help to ensure consistency here.

### Writing generation prompts

We mentioned above that `add_generation_prompt` is a special variable that will be accessible inside your template,
and is controlled by the user setting the `add_generation_prompt` flag. If your model expects a header for
assistant messages, then your template must support adding the header when `add_generation_prompt` is set.

Here is an example of a template that formats messages ChatML-style, with generation prompt support:

```text
{{- bos_token }}
{%- for message in messages %}
    {{- '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}
```

The exact content of the assistant header will depend on your specific model, but it should always be **the string
that represents the start of an assistant message**, so that if the user applies your template with 
`add_generation_prompt=True` and then generates text, the model will write an assistant response. Also note that some
models do not need a generation prompt, because assistant messages always begin immediately after user messages. 
This is particularly common for LLaMA and Mistral models, where assistant messages begin immediately after the `[/INST]`
token that ends user messages. In these cases, the template can ignore the `add_generation_prompt` flag.

Generation prompts are important! If your model requires a generation prompt but it is not set in the template, then
model generations will likely be severely degraded, or the model may display unusual behaviour like continuing 
the final user message! 

### Writing and debugging larger templates

When this feature was introduced, most templates were quite small, the Jinja equivalent of a "one-liner" script. 
However, with new models and features like tool-use and RAG, some templates can be 100 lines long or more. When
writing templates like these, it's a good idea to write them in a separate file, using a text editor. You can easily 
extract a chat template to a file:

```python
open("template.jinja", "w").write(tokenizer.chat_template)
```

Or load the edited template back into the tokenizer:

```python
tokenizer.chat_template = open("template.jinja").read()
```

As an added bonus, when you write a long, multi-line template in a separate file, line numbers in that file will
exactly correspond to line numbers in template parsing or execution errors. This will make it much easier to
identify the source of issues.



## Writing templates for tools

Although chat templates do not enforce a specific API for tools (or for anything, really), we recommend 
template authors try to stick to a standard API where possible. The whole point of chat templates is to allow code
to be transferable across models, so deviating from the standard tools API means users will have to write
custom code to use tools with your model. Sometimes it's unavoidable, but often with clever templating you can
make the standard API work!

Below, we'll list the elements of the standard API, and give tips on writing templates that will work well with it.

### Tool definitions

Your template should expect that the variable `tools` will either be null (if no tools are passed), or is a list 
of JSON schema dicts. Our chat template methods allow users to pass tools as either JSON schema or Python functions, but when
functions are passed, we automatically generate JSON schema and pass that to your template. As a result, the 
`tools` variable that your template receives will always be a list of JSON schema. Here is
a sample tool JSON schema:

```json
{
  "type": "function", 
  "function": {
    "name": "multiply", 
    "description": "A function that multiplies two numbers", 
    "parameters": {
      "type": "object", 
      "properties": {
        "a": {
          "type": "number", 
          "description": "The first number to multiply"
        }, 
        "b": {
          "type": "number",
          "description": "The second number to multiply"
        }
      }, 
      "required": ["a", "b"]
    }
  }
}
```

And here is some example code for handling tools in your chat template. Remember, this is just an example for a
specific format - your model will probably need different formatting!

```text
{%- if tools %}
    {%- for tool in tools %}
        {{- '<tool>' + tool['function']['name'] + '\n' }}
        {%- for argument in tool['function']['parameters']['properties'] %}
            {{- argument + ': ' + tool['function']['parameters']['properties'][argument]['description'] + '\n' }}
        {%- endfor %}
        {{- '\n</tool>' }}
    {%- endif %}
{%- endif %}
```

The specific tokens and tool descriptions your template renders should of course be chosen to match the ones your model
was trained with. There is no requirement that your **model** understands JSON schema input, only that your template can translate
JSON schema into your model's format. For example, [Command-R](https://huggingface.co/CohereForAI/c4ai-command-r-plus-08-2024) 
was trained with tools defined using Python function headers, but the Command-R tool template accepts JSON schema, 
converts types internally and renders the input tools as Python headers. You can do a lot with templates!

### Tool calls

Tool calls, if present, will be a list attached to a message with the "assistant" role. Note that `tool_calls` is 
always a list, even though most tool-calling models only support single tool calls at a time, which means
the list will usually only have a single element. Here is a sample message dict containing a tool call:

```json
{
  "role": "assistant",
  "tool_calls": [
    {
      "type": "function",
      "function": {
        "name": "multiply",
        "arguments": {
          "a": 5,
          "b": 6
        }
      }
    }
  ]
}
```

And a common pattern for handling them would be something like this:

```text
{%- if message['role'] == 'assistant' and 'tool_calls' in message %}
    {%- for tool_call in message['tool_calls'] %}
            {{- '<tool_call>' + tool_call['function']['name'] + '\n' + tool_call['function']['arguments']|tojson + '\n</tool_call>' }}
        {%- endif %}
    {%- endfor %}
{%- endif %}
```

Again, you should render the tool call with the formatting and special tokens that your model expects.

### Tool responses

Tool responses have a simple format: They are a message dict with the "tool" role, a "name" key giving the name
of the called function, and a "content" key containing the result of the tool call. Here is a sample tool response:

```json
{
  "role": "tool",
  "name": "multiply",
  "content": "30"
}
```

You don't need to use all of the keys in the tool response. For example, if your model doesn't expect the function
name to be included in the tool response, then rendering it can be as simple as:

```text
{%- if message['role'] == 'tool' %}
    {{- "<tool_result>" + message['content'] + "</tool_result>" }}
{%- endif %}
```

Again, remember that the actual formatting and special tokens are model-specific - you should take a lot of care
to ensure that tokens, whitespace and everything else exactly match the format your model was trained with!
