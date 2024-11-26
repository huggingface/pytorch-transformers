<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
``
⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Tiktoken and interaction with Transformers

Support for tiktoken model files is seamlessly integrated in 🤗 transformers when loading models 
`from_pretrained` with a `tokenizer.model` tiktoken file on the Hub, which is automatically converted into our 
[fast tokenizer](https://huggingface.co/docs/transformers/main/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast). 

### Known models that were released with a `tiktoken.model`:
	- gpt2
	- llama3

## Example usage
 
In order to load `tiktoken` files in `transformers`, ensure that the `tokenizer.model` file is a tiktoken file and it 
will automatically be loaded when loading `from_pretrained`. Here is how one would load a tokenizer and a model, which 
 can be loaded from the exact same file:

```py
from transformers import AutoTokenizer

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="original") 
```
## Create tiktoken tokenizer

The `tokenizer.model` file contains no information about additional tokens or pattern strings. If these are important, convert the tokenizer to `tokenizer.json`, the appropriate format for [`PreTrainedTokenizerFast`].

Generate the `tokenizer.model` file with [tiktoken.get_encoding](https://github.com/openai/tiktoken/blob/63527649963def8c759b0f91f2eb69a40934e468/tiktoken/registry.py#L63) and then convert it to `tokenizer.json` with [`convert_tiktoken_to_fast`].

```py

from transformers.integrations.tiktoken import convert_tiktoken_to_fast
from tiktoken import get_encoding

# You can load your custom encoding or the one provided by OpenAI
encoding = get_encoding("gpt2")
convert_tiktoken_to_fast(encoding, "config/save/dir")
```

The resulting `tokenizer.json` file is saved to the specified directory and can be loaded with [`PreTrainedTokenizerFast`].

```py
tokenizer = PreTrainedTokenizerFast.from_pretrained("config/save/dir")
```

