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

# MinGRU

## Overview

The MinGRU model was proposed in [Were RNNs All We Needed?](https://arxiv.org/abs/2410.01201) by Leo Feng, Frederick Tung, Mohamed Osama Ahmed, Yoshua Bengio, and Hossein Hajimirsadegh. 
MinGRU is a novel recurrent architecture such as S4, Mamba, and Aaren which (i) no longer need to BPTT and can be efficiently trained in parallel and (ii) use significantly fewer parameters than their traditional GRU counterpart. MinGRU is 175x faster than GRU for a sequence length of 512 while matching the empirical performance of recent sequence models.

The abstract from the paper is the following:

*The scalability limitations of Transformers regarding sequence length have renewed interest in recurrent sequence models that are parallelizable during training. As a result, many novel recurrent architectures, such as S4, Mamba, and Aaren, have been proposed that achieve comparable performance. In this work, we revisit traditional recurrent neural networks (RNNs) from over a decade ago: LSTMs (1997) and GRUs (2014). While these models were slow due to requiring to backpropagate through time (BPTT), we show that by removing their hidden state dependencies from their input, forget, and update gates, LSTMs and GRUs no longer need to BPTT and can be efficiently trained in parallel. Building on this, we introduce minimal versions (minLSTMs and minGRUs) that (1) use significantly fewer parameters than their traditional counterparts and (2) are fully parallelizable during training (175x faster for a sequence of length 512). Lastly, we show that these stripped-down versions of decade-old RNNs match the empirical performance of recent sequence models.* 

Its architecture is an stripped-down version of GRU, with two major differences:
* drops previous hidden state dependencies from gates
* drops range restriction of candidate states
* the output is time-independent in scale

Note that the provided checkpoint is on a tiny corpus of Shakespeare works.

This model was contributed by [José Ángel González (jogonba2)](https://huggingface.co/jogonba2) and [Symanto Research](https://huggingface.co/symanto)
The original code and checkpoints have not been released.

## Usage example 

### Causal Language Modeling (+Pretraining)
Here is a quick example of how to use MinGRU for causal language modeling:

```python
>>> from transformers import AutoModelForCausalLM, AutoTokenizer

>>> # load model and feature extractor
>>> tokenizer = AutoTokenizer.from_pretrained("symanto/mingru-shakespeare")
>>> model = AutoModelForCausalLM.from_pretrained("symanto/mingru-shakespeare")

>>> # Prepare the prefix
>>> prefix = "Second Citizen:\nNay, but speak not maliciously.\n"
>>> input_ids = torch.LongTensor(tokenizer(prefix)["input_ids"]).unsqueeze(0)

>>> # Generate
>>> decoding_args = {"max_new_tokens": 200, "do_sample": True, "top_p": 0.9}
>>> completion = model.generate(input_ids=input_ids, **decoding_args)
```

You can pretrain this model for causal language modeling as [usual using 🤗 Transformers](https://huggingface.co/learn/nlp-course/chapter7/6).

### Sequence Classification
Here is a quick example of how to use MinGRU for sequence classification:

```python
>>> from transformers import AutoModelForSequenceClassification, AutoTokenizer

>>> # load model and feature extractor
>>> tokenizer = AutoTokenizer.from_pretrained("...")
>>> model = AutoModelForSequenceClassification.from_pretrained("...")

>>> # Prepare your input
>>> text = "Sunday afternoon walking through Venice in the sun with @user"
>>> input_ids = torch.LongTensor(tokenizer(prefix)["input_ids"]).unsqueeze(0)

>>> # Predict
>>> pred = model(input_ids=input_ids)
```

### Token Classification
Here is a quick example of how to use MinGRU for token classification:

```python
>>> from transformers import AutoModelForTokenClassification, AutoTokenizer

>>> # load model and feature extractor
>>> tokenizer = AutoTokenizer.from_pretrained("...")
>>> model = AutoModelForTokenClassification.from_pretrained("...")

>>> # Prepare your input
>>> text = "The European Comission said on Thursday it disagreed with German advice"
>>> input_ids = torch.LongTensor(tokenizer(prefix)["input_ids"]).unsqueeze(0)

>>> # Predict
>>> pred = model(input_ids=input_ids)
```

## MinGRUConfig

[[autodoc]] MinGRUConfig

## MinGRUModel

[[autodoc]] MinGRUModel
    - forward

[[autodoc]] MinGRUForCausalLM
    - forward

[[autodoc]] MinGRUForSequenceClassification
    - forward

[[autodoc]] MinGRUForTokenClassification
    - forward