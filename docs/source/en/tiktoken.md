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

# Tiktoken

[Tiktoken](https://github.com/openai/tiktoken) is a [byte-pair encoding (BPE)](./tokenizer_summary#byte-pair-encoding-bpe) tokenizer by OpenAI, inclyding several tokenization schemes or encodings for how text should be tokenized.

There are currently two models trained and released with tiktoken, GPT2 and Llama3. Transformers supports loading these models with the [tokenizer.model](https://hf.co/meta-llama/Meta-Llama-3-8B/blob/main/original/tokenizer.model) tiktoken file. The tiktoken file is automatically converted into Transformers Rust-based [`PreTrainedTokenizerFast`].

Add the `subfolder` parameter to [`~PreTrainedModel.from_pretrained`] to specify where the `tokenizer.model` file is located.

```py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", subfolder="original") 
```

You can visualize how the tiktoken tokenizer works for Llama3 with the Tokenizer Playground below.

<iframe
	src="https://xenova-the-tokenizer-playground.static.hf.space"
	frameborder="0"
	width="850"
	height="450"
></iframe>
