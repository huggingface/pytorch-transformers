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

# Generation strategies

A decoding strategy informs how a model should select the next generated token. There are many types of decoding strategies, and choosing the appropriate one has a significant impact on the quality of the generated text.

This guide will help you understand the different decoding strategies available in Transformers and how and when to use them.

## Default decoding strategy

Greedy search is the default decoding strategy. It selects the next most likely token at each step. Unless specified in the [`GenerationConfig`], this strategy generates a maximum of 20 tokens.

Greedy search works well for tasks with relatively short outputs. However, it breaks down when generating longer sequences because it begins to repeat itself.

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
inputs = tokenizer("I look forward to", return_tensors="pt").to("cuda")

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype=torch.float16).to("cuda")
# explicitly set to default length because Llama2 generation length is 4096
outputs = model.generate(**inputs, max_new_tokens=20)
tokenizer.batch_decode(outputs, skip_special_tokens=True)
'Hugging Face is an open-source company that provides a suite of tools and services for building, deploying, and maintaining natural language processing'
```

## Contrastive search

[Contrastive search](https://huggingface.co/papers/2202.06417) is a decoding strategy that aims to reduce repetition even while generating longer sequences. This strategy compares how similar a generated token is against previous tokens, and if they're more similar, a penalty is applied.

Enable contrastive search with the `penalty_alpha` and `top_k` parameters. The `penalty_alpha` manages the penalty applied and `top_k` is the number of most likely tokens to return.

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
inputs = tokenizer("Hugging Face is an open-source company", return_tensors="pt").to("cuda")

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype=torch.float16).to("cuda")
# explicitly set to 100 because Llama2 generation length is 4096
outputs = model.generate(**inputs, max_new_tokens=100, penalty_alpha=0.6, top_k=4)
tokenizer.batch_decode(outputs, skip_special_tokens=True)
'Hugging Face is an open-source company that provides a platform for building and deploying AI models.\nHugging Face is an open-source company that provides a platform for building and deploying AI models. The platform allows developers to build and deploy AI models, as well as collaborate with other developers.\nHugging Face was founded in 2019 by Thibault Wittemberg and Clément Delangue. The company is based in Paris, France.\nHugging Face has'
```

## Beam search

Beam search keeps track of several generated sequences (beams) at each time step. After a certain number of steps, it selects the sequence with the highest *overall* probability. Unlike greedy search, this strategy can "look ahead" and pick a sequence with a higher probability overall even if the initial tokens have a lower probability.

> [TIP]
> Check out the [beam search visualizer](https://huggingface.co/spaces/m-ric/beam_search_visualizer) to see how beam search works.

Enable beam search with the `num_beams` parameter (should be greater than 1 otherwise it's equivalent to greedy search).

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
inputs = tokenizer("Hugging Face is an open-source company", return_tensors="pt").to("cuda")

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype=torch.float16).to("cuda")
# explicitly set to 100 because Llama2 generation length is 4096
outputs = model.generate(**inputs, max_new_tokens=50, num_beams=2)
tokenizer.batch_decode(outputs, skip_special_tokens=True)
"['Hugging Face is an open-source company that develops and maintains the Hugging Face platform, which is a collection of tools and libraries for building and deploying natural language processing (NLP) models. Hugging Face was founded in 2018 by Thomas Wolf']"
```

## Diverse beam search

[Diverse beam search](https://hf.co/papers/1610.02424) is a variant of beam search that produces more diverse output candidates to choose from. This strategy measures the dissimilarity of sequences and a penalty is applied if sequences are too similar. To avoid high computation costs, the number of beams is divided into groups.

Enable diverse beam search with the `num_beams`, `num_beam_groups` and `diversity_penalty` parameters (the `num_beams` parameter should be divisible by `num_beam_groups`).

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
inputs = tokenizer("Hugging Face is an open-source company", return_tensors="pt").to("cuda")

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype=torch.float16).to("cuda")
# explicitly set to 100 because Llama2 generation length is 4096
outputs = model.generate(**inputs, max_new_tokens=50, num_beams=6, num_beam_groups=3, diversity_penalty=1.0, do_sample=False)
tokenizer.batch_decode(outputs, skip_special_tokens=True)
'Hugging Face is an open-source company 🤗\nWe are an open-source company. Our mission is to democratize AI and make it accessible to everyone. We believe that AI should be used for the benefit of humanity, not for the benefit of a'
```

## Multinomial sampling

Search methods selects the most likely tokens. Sampling, or multinomial sampling, randomly selects a token based on the probability distribution over the entire models vocabulary. This means every token with a non-zero probability has a chance to be selected. Sampling strategies reduce repetition and can generate more creative and diverse outputs.

Enable multinomial sampling with `do_sample=True` and `num_beams=1`.

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
inputs = tokenizer("Hugging Face is an open-source company", return_tensors="pt").to("cuda")

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype=torch.float16).to("cuda")
# explicitly set to 100 because Llama2 generation length is 4096
outputs = model.generate(**inputs, max_new_tokens=50, do_sample=True, num_beams=1)
tokenizer.batch_decode(outputs, skip_special_tokens=True)
'Hugging Face is an open-source company 🤗\nWe are open-source and believe that open-source is the best way to build technology. Our mission is to make AI accessible to everyone, and we believe that open-source is the best way to achieve that.'
```

## Beam search multinomial sampling

This decoding strategy is a combination of beam search and multinomial sampling. It generates multiple beams and uses a sampling strategy for each beam.

Enable beam search multinomial sampling by setting `num_beams` to a value greater than 1 and `do_sample=True`.

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
inputs = tokenizer("Hugging Face is an open-source company", return_tensors="pt").to("cuda")

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype=torch.float16).to("cuda")
# explicitly set to 100 because Llama2 generation length is 4096
outputs = model.generate(**inputs, max_new_tokens=50, do_sample=True, num_beams=4)
'Hugging Face is an open-source company 100% dedicated to making AI more accessible. We believe that AI should be available to everyone, and we’re working hard to make that a reality.\nWe’re a team of passionate engineers, designers,'
```

## Speculative decoding

[Speculative](https://hf.co/papers/2211.17192) or assistive decoding isn't a search or sampling strategy. This method is especially useful for LLMs where it can be more costly and slower to generate tokens. Instead, speculative decoding adds a second smaller model to generate candidate tokens. The main model verifies the candidate tokens in a single `forward` pass, which speeds up the decoding process overall. Refer to the [speculative decoding](./llm_optims#speculative-decoding) guide to learn more.

Currently, only greed search and multinomial sampling are supported with speculative decoding. Batched inputs aren't supported either.

Enable speculative decoding with the `assistant_model` parameter. You'll notice the fastest speed up with an assistant model that is much smaller than the main model. Add `do_sample=True` to enable token validation with resampling.

<hfoptions id="spec-decoding">
<hfoption id="greedy search">

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-1.7B")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-1.7B")
assistant_model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M")
inputs = tokenizer("Hugging Face is an open-source company", return_tensors="pt")

outputs = model.generate(**inputs, assistant_model=assistant_model)
tokenizer.batch_decode(outputs, skip_special_tokens=True)
'Hugging Face is an open-source company that provides a platform for developers to build and deploy machine'
```

<Tip>

If you're using a `pipeline` object, all you need to do is to pass the assistant checkpoint under `assistant_model`

```python
>>> from transformers import pipeline
>>> import torch

>>> pipe = pipeline(
...     "text-generation",
...     model="meta-llama/Llama-3.1-8B",
...     assistant_model="meta-llama/Llama-3.2-1B",  # This extra line is all that's needed, also works with UAD
...     torch_dtype=torch.bfloat16
>>> )
>>> pipe_output = pipe("Once upon a time, ", max_new_tokens=50, do_sample=False)
>>> pipe_output[0]["generated_text"]
'Once upon a time, 3D printing was a niche technology that was only'
```

</Tip>


</hfoption>
<hfoption id="multinomial sampling">

Add the `temperature` parameter to control sampling randomness. For speculative decoding, a lower temperature may improve latency.

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-1.7B")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-1.7B")
assistant_model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M")
inputs = tokenizer("Hugging Face is an open-source company", return_tensors="pt")

outputs = model.generate(**inputs, assistant_model=assistant_model, do_sample=True, temperature=0.5)
tokenizer.batch_decode(outputs, skip_special_tokens=True)
'Hugging Face is an open-source company that is dedicated to creating a better world through technology.'
```

</hfoption>
</hfoptions>

### Prompt lookup decoding

[Prompt lookup decoding](./llm_optims#prompt-lookup-decoding) is a variant of speculative decoding that uses overlapping n-grams as the candidate tokens. It works well for input-grounded tasks such as summarization. Refer to the [prompt lookup decoding](./llm_optims#prompt-lookup-decoding) guide to learn more.

Enable prompt lookup decoding with the `prompt_lookup_num_tokens` parameter.

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-1.7B")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-1.7B", torch_dtype=torch.float16).to("cuda")
assistant_model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M", torch_dtype=torch.float16).to("cuda")
inputs = tokenizer("Hugging Face is an open-source company", return_tensors="pt").to("cuda")

outputs = model.generate(**inputs, assistant_model=assistant_model, max_new_tokens=20, prompt_lookup_num_tokens=5)
tokenizer.batch_decode(outputs, skip_special_tokens=True)
'Hugging Face is an open-source company that provides a platform for developers to build and deploy machine learning models. It offers a variety of tools'
```

## DoLa

[Decoding by Contrasting Layers (DoLa)](https://hf.co/papers/2309.03883) is a contrastive decoding strategy for improving factuality and reducing hallucination. This strategy works by contrasting the logit diffferences between the final and early layers. As a result, factual knowledge localized to a particular layers are amplified. DoLa is not recommended for smaller models like GPT-2.

Enable DoLa with the following parameters.

- `dola_layers` are the candidate layers to be contrasted with the final layer. It can be a string with `low` or `high` to contrast the lower or higher parts of a layer. `high` is recommended for short-answer tasks like TruthfulQA. `low` is recommended for long-answer reasoning tasks like GSM8K, StrategyQA, FACTOR, and VicunaQA.

  When a model has tied word embeddings, layer 0 is skipped and it begins from layer 2.

  It can also be a list of integers that represent the layer indices between 0 and the total number of layers. Layer 0 is the word embedding, 1 is the first transformer layer, and so on. Refer to the table below for the range of layer indices depending on the number of model layers.

| layers | low | high |
|---|---|---|
| > 40 | (0, 20, 2) | (N - 20, N, 2) |
| <= 40 | range(0, N // 2, 2) | range(N // 2, N, 2) |

- `repetition_penalty` reduces repetition and it is recommended to set it to 1.2.

<hfoptions id="dola">
<hfoption id="contrast higher layers">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-1.7B")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-1.7B", torch_dtype=torch.float16).to("cuda")
inputs = tokenizer("What is the highest peak in the world??", return_tensors="pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens=50, dola_layers="high", do_sample=False)
tokenizer.batch_decode(outputs, skip_special_tokens=True)
" Mount EverestMount Everest, called Himalaya in Nepali, is the world's highest peak, lying almost 9.5 kilometers above the sea level and the tallest mountain from 19,036.91 ft. The mountain was"
```

</hfoption>
<hfoption id="contrast specific layers">

Contrast layers 18 and 20 with the final layer.

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-1.7B")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-1.7B", torch_dtype=torch.float16).to("cuda")
inputs = tokenizer("What is the highest peak in the world?", return_tensors="pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens=50, dola_layers=[18,20], do_sample=False, repetition_penalty=1.2)
tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True)
" Mount EverestMount Everest, called Himalaya in Nepali, is the world's highest peak above sea level and it rises to an incredible height of 29,028 feet above the ocean. Its summit is over a mile taller than Mt"


```

</hfoption>
</hfoptions>