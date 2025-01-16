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

# tiktoken

[tiktoken](https://github.com/openai/tiktoken) is a [byte-pair encoding (BPE)](./tokenizer_summary#byte-pair-encoding-bpe) tokenizer by OpenAI. It includes several tokenization schemes or encodings for how text should be tokenized.

There are currently two models trained and released with tiktoken, GPT2 and Llama3. Transformers supports models with a [tokenizer.model](https://hf.co/meta-llama/Meta-Llama-3-8B/blob/main/original/tokenizer.model) tiktoken file. The tiktoken file is automatically converted into Transformers Rust-based [`PreTrainedTokenizerFast`].

Add the `subfolder` parameter to [`~PreTrainedModel.from_pretrained`] to specify where the `tokenizer.model` tiktoken file is located.

```py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", subfolder="original") 
```

## Create a tiktoken tokenizer

The tiktoken `tokenizer.model` file contains no information about additional tokens or pattern strings. If these are important, convert the tokenizer to `tokenizer.json` (the appropriate format for [`PreTrainedTokenizerFast`]).

Generate the tiktoken `tokenizer.model` file with the [tiktoken.get_encoding](https://github.com/openai/tiktoken/blob/63527649963def8c759b0f91f2eb69a40934e468/tiktoken/registry.py#L63) function, and convert it to `tokenizer.json` with [convert_tiktoken_to_fast](https://github.com/huggingface/transformers/blob/99e0ab6ed888136ea4877c6d8ab03690a1478363/src/transformers/integrations/tiktoken.py#L8).

```py
from transformers.integrations.tiktoken import convert_tiktoken_to_fast
from tiktoken import get_encoding

# Load your custom encoding or the one provided by OpenAI
encoding = get_encoding("gpt2")
convert_tiktoken_to_fast(encoding, "config/save/dir")
```

The resulting `tokenizer.json` file is saved to the specified directory and loaded with [`~PreTrainedTokenizerFast.from_pretrained`].

```py
tokenizer = PreTrainedTokenizerFast.from_pretrained("config/save/dir")
```

Visualize how the tiktoken tokenizer works by selecting Llama3 in the Tokenizer Playground below.

<iframe
	src="https://xenova-the-tokenizer-playground.static.hf.space"
	frameborder="0"
	width="850"
	height="600"
></iframe>
