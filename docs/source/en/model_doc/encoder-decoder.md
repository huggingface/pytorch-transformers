<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Encoder Decoder Models

## Overview

The [`EncoderDecoderModel`] can be used to initialize a sequence-to-sequence model with any
pretrained autoencoding model as the encoder and any pretrained autoregressive model as the decoder.

The effectiveness of initializing sequence-to-sequence models with pretrained checkpoints for sequence generation tasks
was shown in [Leveraging Pre-trained Checkpoints for Sequence Generation Tasks](https://arxiv.org/abs/1907.12461) by
Sascha Rothe, Shashi Narayan, Aliaksei Severyn.

After such an [`EncoderDecoderModel`] has been trained/fine-tuned, it can be saved/loaded just like
any other models (see the examples for more information).

An application of this architecture could be to leverage two pretrained [`BertModel`] as the encoder
and decoder for a summarization model as was shown in: [Text Summarization with Pretrained Encoders](https://arxiv.org/abs/1908.08345) by Yang Liu and Mirella Lapata.

## Randomly initializing `EncoderDecoderModel` from model configurations.

[`EncoderDecoderModel`] can be randomly initialized from an encoder and a decoder config. In the following example, we show how to do this using the default [`BertModel`] configuration for the encoder and the default [`BertForCausalLM`] configuration for the decoder.

```python
>>> from transformers import BertConfig, EncoderDecoderConfig, EncoderDecoderModel

>>> config_encoder = BertConfig()
>>> config_decoder = BertConfig()

>>> config = EncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)
>>> model = EncoderDecoderModel(config=config)
```

## Initialising `EncoderDecoderModel` from a pretrained encoder and a pretrained decoder.

[`EncoderDecoderModel`] can be initialized from a pretrained encoder checkpoint and a pretrained decoder checkpoint. Note that any pretrained auto-encoding model, *e.g.* BERT, can serve as the encoder and both pretrained auto-encoding models, *e.g.* BERT, pretrained causal language models, *e.g.* GPT2, as well as the pretrained decoder part of sequence-to-sequence models, *e.g.* decoder of BART, can be used as the decoder.
Depending on which architecture you choose as the decoder, the cross-attention layers might be randomly initialized.
Initializing [`EncoderDecoderModel`] from a pretrained encoder and decoder checkpoint requires the model to be fine-tuned on a downstream task, as has been shown in [the *Warm-starting-encoder-decoder blog post*](https://huggingface.co/blog/warm-starting-encoder-decoder).
To do so, the `EncoderDecoderModel` class provides a [`EncoderDecoderModel.from_encoder_decoder_pretrained`] method.

```python
>>> from transformers import EncoderDecoderModel, BertTokenizer

>>> tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
>>> model = EncoderDecoderModel.from_encoder_decoder_pretrained("google-bert/bert-base-uncased", "google-bert/bert-base-uncased")
```

## Loading an existing `EncoderDecoderModel` checkpoint and perform inference.

To load fine-tuned checkpoints of the `EncoderDecoderModel` class, [`EncoderDecoderModel`] provides the `from_pretrained(...)` method just like any other model architecture in Transformers.

To perform inference, one uses the [`generate`] method, which allows to autoregressively generate text. This method supports various forms of decoding, such as greedy, beam search and multinomial sampling.

```python
>>> from transformers import AutoTokenizer, EncoderDecoderModel

>>> # load a fine-tuned seq2seq model and corresponding tokenizer
>>> model = EncoderDecoderModel.from_pretrained("patrickvonplaten/bert2bert_cnn_daily_mail")
>>> tokenizer = AutoTokenizer.from_pretrained("patrickvonplaten/bert2bert_cnn_daily_mail")

>>> # let's perform inference on a long piece of text
>>> ARTICLE_TO_SUMMARIZE = (
...     "PG&E stated it scheduled the blackouts in response to forecasts for high winds "
...     "amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were "
...     "scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."
... )
>>> input_ids = tokenizer(ARTICLE_TO_SUMMARIZE, return_tensors="pt").input_ids

>>> # autoregressively generate summary (uses greedy decoding by default)
>>> generated_ids = model.generate(input_ids)
>>> generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
>>> print(generated_text)
nearly 800 thousand customers were affected by the shutoffs. the aim is to reduce the risk of wildfires. nearly 800, 000 customers were expected to be affected by high winds amid dry conditions. pg & e said it scheduled the blackouts to last through at least midday tomorrow.
```

## Loading a PyTorch checkpoint into `TFEncoderDecoderModel`.

[`TFEncoderDecoderModel.from_pretrained`] currently doesn't support initializing the model from a
pytorch checkpoint. Passing `from_pt=True` to this method will throw an exception. If there are only pytorch
checkpoints for a particular encoder-decoder model, a workaround is:

```python
>>> # a workaround to load from pytorch checkpoint
>>> from transformers import EncoderDecoderModel, TFEncoderDecoderModel

>>> _model = EncoderDecoderModel.from_pretrained("patrickvonplaten/bert2bert-cnn_dailymail-fp16")

>>> _model.encoder.save_pretrained("./encoder")
>>> _model.decoder.save_pretrained("./decoder")

>>> model = TFEncoderDecoderModel.from_encoder_decoder_pretrained(
...     "./encoder", "./decoder", encoder_from_pt=True, decoder_from_pt=True
... )
>>> # This is only for copying some specific attributes of this particular model.
>>> model.config = _model.config
```

## Training

Once the model is created, it can be fine-tuned similar to BART, T5 or any other encoder-decoder model.
As you can see, only 2 inputs are required for the model in order to compute a loss: `input_ids` (which are the
`input_ids` of the encoded input sequence) and `labels` (which are the `input_ids` of the encoded
target sequence).

```python
>>> from transformers import BertTokenizer, EncoderDecoderModel

>>> tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
>>> model = EncoderDecoderModel.from_encoder_decoder_pretrained("google-bert/bert-base-uncased", "google-bert/bert-base-uncased")

>>> model.config.decoder_start_token_id = tokenizer.cls_token_id
>>> model.config.pad_token_id = tokenizer.pad_token_id

>>> input_ids = tokenizer(
...     "The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side.During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was  finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft).Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct.",
...     return_tensors="pt",
... ).input_ids

>>> labels = tokenizer(
...     "the eiffel tower surpassed the washington monument to become the tallest structure in the world. it was the first structure to reach a height of 300 metres in paris in 1930. it is now taller than the chrysler building by 5. 2 metres ( 17 ft ) and is the second tallest free - standing structure in paris.",
...     return_tensors="pt",
... ).input_ids

>>> # the forward function automatically creates the correct decoder_input_ids
>>> loss = model(input_ids=input_ids, labels=labels).loss
```

Detailed [colab](https://colab.research.google.com/drive/1WIk2bxglElfZewOHboPFNj8H44_VAyKE?usp=sharing#scrollTo=ZwQIEhKOrJpl) for training.

This model was contributed by [thomwolf](https://github.com/thomwolf). This model's TensorFlow and Flax versions
were contributed by [ydshieh](https://github.com/ydshieh).


## EncoderDecoderConfig

[[autodoc]] EncoderDecoderConfig

<frameworkcontent>
<pt>

## EncoderDecoderModel

[[autodoc]] EncoderDecoderModel
    - forward
    - from_encoder_decoder_pretrained

</pt>
<tf>

## TFEncoderDecoderModel

[[autodoc]] TFEncoderDecoderModel
    - call
    - from_encoder_decoder_pretrained

</tf>
<jax>

## FlaxEncoderDecoderModel

[[autodoc]] FlaxEncoderDecoderModel
    - __call__
    - from_encoder_decoder_pretrained

</jax>
</frameworkcontent>

## Create a model

To create an `EncoderDecoderModel`, you can use the `from_encoder_decoder_pretrained` method. This method allows you to initialize the model from a pretrained encoder and decoder checkpoint. Here is an example:

```python
from transformers import EncoderDecoderModel

# Initialize a Bert2Bert model from pretrained checkpoints
model = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-uncased", "bert-base-uncased")

# Save the model
model.save_pretrained("./bert2bert")

# Load the model
model = EncoderDecoderModel.from_pretrained("./bert2bert")
```

## Finetune a model

Once the model is created, it can be fine-tuned similar to BART, T5, or any other encoder-decoder model. Here is an example:

```python
from transformers import BertTokenizer, EncoderDecoderModel

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-uncased", "bert-base-uncased")

model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.pad_token_id = tokenizer.pad_token_id

input_ids = tokenizer(
    "The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side.During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was  finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft).Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct.",
    return_tensors="pt",
).input_ids

labels = tokenizer(
    "the eiffel tower surpassed the washington monument to become the tallest structure in the world. it was the first structure to reach a height of 300 metres in paris in 1930. it is now taller than the chrysler building by 5. 2 metres ( 17 ft ) and is the second tallest free - standing structure in paris.",
    return_tensors="pt",
).input_ids

# The forward function automatically creates the correct decoder_input_ids
loss = model(input_ids=input_ids, labels=labels).loss
```

## Warning about configuration values

> [!WARNING]
> It is important to correctly set the configuration values for the `EncoderDecoderModel`. Incorrect configuration values can lead to unexpected behavior. For more information on how to set the configuration values, refer to the [related issue](https://github.com/huggingface/transformers/issues/15479).
