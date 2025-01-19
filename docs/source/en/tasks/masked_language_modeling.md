<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Masked language modeling

[[open-in-colab]]

<Youtube id="mqElG5QJWUg"/>

Masked language modeling predicts a masked token in a sequence, and the model can attend to tokens bidirectionally. This
means the model has full access to the tokens on the left and right. Masked language modeling is great for tasks that
require a good contextual understanding of an entire sequence. BERT is an example of a masked language model.

This guide will show you how to:

1. Finetune [DistilRoBERTa](https://huggingface.co/distilbert/distilroberta-base) on the [r/askscience](https://www.reddit.com/r/askscience/) subset of the [ELI5](https://huggingface.co/datasets/eli5) dataset.
2. Use your finetuned model for inference.

<Tip>

To see all architectures and checkpoints compatible with this task, we recommend checking the [task-page](https://huggingface.co/tasks/fill-mask)

</Tip>

Before you begin, make sure you have all the necessary libraries installed:

```bash
pip install transformers datasets evaluate
```

We encourage you to log in to your Hugging Face account so you can upload and share your model with the community. When prompted, enter your token to log in:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## Load ELI5 dataset

Start by loading the first 5000 examples from the [ELI5-Category](https://huggingface.co/datasets/eli5_category) dataset with the 🤗 Datasets library. This'll give you a chance to experiment and make sure everything works before spending more time training on the full dataset.

```py
>>> from datasets import load_dataset

>>> eli5 = load_dataset("eli5_category", split="train[:5000]")
```

Split the dataset's `train` split into a train and test set with the [`~datasets.Dataset.train_test_split`] method:

```py
>>> eli5 = eli5.train_test_split(test_size=0.2)
```

Then take a look at an example:

```py
>>> eli5["train"][0]
{'q_id': '7h191n',
 'title': 'What does the tax bill that was passed today mean? How will it affect Americans in each tax bracket?',
 'selftext': '',
 'category': 'Economics',
 'subreddit': 'explainlikeimfive',
 'answers': {'a_id': ['dqnds8l', 'dqnd1jl', 'dqng3i1', 'dqnku5x'],
  'text': ["The tax bill is 500 pages long and there were a lot of changes still going on right to the end. It's not just an adjustment to the income tax brackets, it's a whole bunch of changes. As such there is no good answer to your question. The big take aways are: - Big reduction in corporate income tax rate will make large companies very happy. - Pass through rate change will make certain styles of business (law firms, hedge funds) extremely happy - Income tax changes are moderate, and are set to expire (though it's the kind of thing that might just always get re-applied without being made permanent) - People in high tax states (California, New York) lose out, and many of them will end up with their taxes raised.",
   'None yet. It has to be reconciled with a vastly different house bill and then passed again.',
   'Also: does this apply to 2017 taxes? Or does it start with 2018 taxes?',
   'This article explains both the House and senate bills, including the proposed changes to your income taxes based on your income level. URL_0'],
  'score': [21, 19, 5, 3],
  'text_urls': [[],
   [],
   [],
   ['https://www.investopedia.com/news/trumps-tax-reform-what-can-be-done/']]},
 'title_urls': ['url'],
 'selftext_urls': ['url']}
```

While this may look like a lot, you're only really interested in the `text` field. What's cool about language modeling tasks is you don't need labels (also known as an unsupervised task) because the next word *is* the label.

## Preprocess

<Youtube id="8PmhEIXhBvI"/>

For masked language modeling, the next step is to load a DistilRoBERTa tokenizer to process the `text` subfield:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilroberta-base")
```

You'll notice from the example above, the `text` field is actually nested inside `answers`. This means you'll need to extract the `text` subfield from its nested structure with the [`flatten`](https://huggingface.co/docs/datasets/process#flatten) method:

```py
>>> eli5 = eli5.flatten()
>>> eli5["train"][0]
{'q_id': '7h191n',
 'title': 'What does the tax bill that was passed today mean? How will it affect Americans in each tax bracket?',
 'selftext': '',
 'category': 'Economics',
 'subreddit': 'explainlikeimfive',
 'answers.a_id': ['dqnds8l', 'dqnd1jl', 'dqng3i1', 'dqnku5x'],
 'answers.text': ["The tax bill is 500 pages long and there were a lot of changes still going on right to the end. It's not just an adjustment to the income tax brackets, it's a whole bunch of changes. As such there is no good answer to your question. The big take aways are: - Big reduction in corporate income tax rate will make large companies very happy. - Pass through rate change will make certain styles of business (law firms, hedge funds) extremely happy - Income tax changes are moderate, and are set to expire (though it's the kind of thing that might just always get re-applied without being made permanent) - People in high tax states (California, New York) lose out, and many of them will end up with their taxes raised.",
  'None yet. It has to be reconciled with a vastly different house bill and then passed again.',
  'Also: does this apply to 2017 taxes? Or does it start with 2018 taxes?',
  'This article explains both the House and senate bills, including the proposed changes to your income taxes based on your income level. URL_0'],
 'answers.score': [21, 19, 5, 3],
 'answers.text_urls': [[],
  [],
  [],
  ['https://www.investopedia.com/news/trumps-tax-reform-what-can-be-done/']],
 'title_urls': ['url'],
 'selftext_urls': ['url']}
```

Each subfield is now a separate column as indicated by the `answers` prefix, and the `text` field is a list now. Instead
of tokenizing each sentence separately, convert the list to a string so you can jointly tokenize them.

Here is a first preprocessing function to join the list of strings for each example and tokenize the result:

```py
>>> def preprocess_function(examples):
...     return tokenizer(" ".join(eli5["train"][0]["answers"]["text"]))
```

To apply this preprocessing function over the entire dataset, use the 🤗 Datasets [`~datasets.Dataset.map`] method. You can speed up the `map` function by setting `batched=True` to process multiple elements of the dataset at once, and increasing the number of processes with `num_proc`. Remove any columns you don't need:

```py
>>> tokenized_eli5 = eli5.map(
...     preprocess_function,
...     batched=True,
...     num_proc=4,
...     remove_columns=eli5["train"].column_names,
... )
```

This dataset contains the token sequences, but some of these are longer than the maximum input length for the model.

You can now use a second preprocessing function to
- concatenate all the sequences
- split the concatenated sequences into shorter chunks defined by `block_size`, which should be both shorter than the maximum input length and short enough for your GPU RAM. 

```py
>>> block_size = 128


>>> def group_texts(examples):
...     # Concatenate all texts.
...     concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
...     total_length = len(concatenated_examples[list(examples.keys())[0]])
...     # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
...     # customize this part to your needs.
...     if total_length >= block_size:
...         total_length = (total_length // block_size) * block_size
...     # Split by chunks of block_size.
...     result = {
...         k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
...         for k, t in concatenated_examples.items()
...     }
...     return result
```

Apply the `group_texts` function over the entire dataset:

```py
>>> lm_dataset = tokenized_eli5.map(group_texts, batched=True, num_proc=4)
```

Now create a batch of examples using [`DataCollatorForLanguageModeling`]. It's more efficient to *dynamically pad* the sentences to the longest length in a batch during collation, instead of padding the whole dataset to the maximum length.

<frameworkcontent>
<pt>

Use the end-of-sequence token as the padding token and specify `mlm_probability` to randomly mask tokens each time you iterate over the data:

```py
>>> from transformers import DataCollatorForLanguageModeling

>>> tokenizer.pad_token = tokenizer.eos_token
>>> data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
```
</pt>
<tf>

Use the end-of-sequence token as the padding token and specify `mlm_probability` to randomly mask tokens each time you iterate over the data:

```py
>>> from transformers import DataCollatorForLanguageModeling

>>> data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15, return_tensors="tf")
```
</tf>
</frameworkcontent>

## Train

<frameworkcontent>
<pt>
<Tip>

If you aren't familiar with finetuning a model with the [`Trainer`], take a look at the basic tutorial [here](../training#train-with-pytorch-trainer)!

</Tip>

You're ready to start training your model now! Load DistilRoBERTa with [`AutoModelForMaskedLM`]:

```py
>>> from transformers import AutoModelForMaskedLM

>>> model = AutoModelForMaskedLM.from_pretrained("distilbert/distilroberta-base")
```

At this point, only three steps remain:

1. Define your training hyperparameters in [`TrainingArguments`]. The only required parameter is `output_dir` which specifies where to save your model. You'll push this model to the Hub by setting `push_to_hub=True` (you need to be signed in to Hugging Face to upload your model).
2. Pass the training arguments to [`Trainer`] along with the model, datasets, and data collator.
3. Call [`~Trainer.train`] to finetune your model.

```py
>>> training_args = TrainingArguments(
...     output_dir="my_awesome_eli5_mlm_model",
...     eval_strategy="epoch",
...     learning_rate=2e-5,
...     num_train_epochs=3,
...     weight_decay=0.01,
...     push_to_hub=True,
... )

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=lm_dataset["train"],
...     eval_dataset=lm_dataset["test"],
...     data_collator=data_collator,
...     tokenizer=tokenizer,
... )

>>> trainer.train()
```

Once training is completed, use the [`~transformers.Trainer.evaluate`] method to evaluate your model and get its perplexity:

```py
>>> import math

>>> eval_results = trainer.evaluate()
>>> print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
Perplexity: 8.76
```

Then share your model to the Hub with the [`~transformers.Trainer.push_to_hub`] method so everyone can use your model:

```py
>>> trainer.push_to_hub()
```
</pt>
<tf>
<Tip>

If you aren't familiar with finetuning a model with Keras, take a look at the basic tutorial [here](../training#train-a-tensorflow-model-with-keras)!

</Tip>
To finetune a model in TensorFlow, start by setting up an optimizer function, learning rate schedule, and some training hyperparameters:

```py
>>> from transformers import create_optimizer, AdamWeightDecay

>>> optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)
```

Then you can load DistilRoBERTa with [`TFAutoModelForMaskedLM`]:

```py
>>> from transformers import TFAutoModelForMaskedLM

>>> model = TFAutoModelForMaskedLM.from_pretrained("distilbert/distilroberta-base")
```

Convert your datasets to the `tf.data.Dataset` format with [`~transformers.TFPreTrainedModel.prepare_tf_dataset`]:

```py
>>> tf_train_set = model.prepare_tf_dataset(
...     lm_dataset["train"],
...     shuffle=True,
...     batch_size=16,
...     collate_fn=data_collator,
... )

>>> tf_test_set = model.prepare_tf_dataset(
...     lm_dataset["test"],
...     shuffle=False,
...     batch_size=16,
...     collate_fn=data_collator,
... )
```

Configure the model for training with [`compile`](https://keras.io/api/models/model_training_apis/#compile-method). Note that Transformers models all have a default task-relevant loss function, so you don't need to specify one unless you want to:

```py
>>> import tensorflow as tf

>>> model.compile(optimizer=optimizer)  # No loss argument!
```

This can be done by specifying where to push your model and tokenizer in the [`~transformers.PushToHubCallback`]:

```py
>>> from transformers.keras_callbacks import PushToHubCallback

>>> callback = PushToHubCallback(
...     output_dir="my_awesome_eli5_mlm_model",
...     tokenizer=tokenizer,
... )
```

Finally, you're ready to start training your model! Call [`fit`](https://keras.io/api/models/model_training_apis/#fit-method) with your training and validation datasets, the number of epochs, and your callback to finetune the model:

```py
>>> model.fit(x=tf_train_set, validation_data=tf_test_set, epochs=3, callbacks=[callback])
```

Once training is completed, your model is automatically uploaded to the Hub so everyone can use it!
</tf>
</frameworkcontent>

<Tip>

For a more in-depth example of how to finetune a model for masked language modeling, take a look at the corresponding
[PyTorch notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb)
or [TensorFlow notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling-tf.ipynb).

</Tip>

## Inference

Great, now that you've finetuned a model, you can use it for inference!

Come up with some text you'd like the model to fill in the blank with, and use the special `<mask>` token to indicate the blank:

```py
>>> text = "The Milky Way is a <mask> galaxy."
```

The simplest way to try out your finetuned model for inference is to use it in a [`pipeline`]. Instantiate a `pipeline` for fill-mask with your model, and pass your text to it. If you like, you can use the `top_k` parameter to specify how many predictions to return:

```py
>>> from transformers import pipeline

>>> mask_filler = pipeline("fill-mask", "username/my_awesome_eli5_mlm_model")
>>> mask_filler(text, top_k=3)
[{'score': 0.5150994658470154,
  'token': 21300,
  'token_str': ' spiral',
  'sequence': 'The Milky Way is a spiral galaxy.'},
 {'score': 0.07087188959121704,
  'token': 2232,
  'token_str': ' massive',
  'sequence': 'The Milky Way is a massive galaxy.'},
 {'score': 0.06434620916843414,
  'token': 650,
  'token_str': ' small',
  'sequence': 'The Milky Way is a small galaxy.'}]
```

<frameworkcontent>
<pt>
Tokenize the text and return the `input_ids` as PyTorch tensors. You'll also need to specify the position of the `<mask>` token:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("username/my_awesome_eli5_mlm_model")
>>> inputs = tokenizer(text, return_tensors="pt")
>>> mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
```

Pass your inputs to the model and return the `logits` of the masked token:

```py
>>> from transformers import AutoModelForMaskedLM

>>> model = AutoModelForMaskedLM.from_pretrained("username/my_awesome_eli5_mlm_model")
>>> logits = model(**inputs).logits
>>> mask_token_logits = logits[0, mask_token_index, :]
```

Then return the three masked tokens with the highest probability and print them out:

```py
>>> top_3_tokens = torch.topk(mask_token_logits, 3, dim=1).indices[0].tolist()

>>> for token in top_3_tokens:
...     print(text.replace(tokenizer.mask_token, tokenizer.decode([token])))
The Milky Way is a spiral galaxy.
The Milky Way is a massive galaxy.
The Milky Way is a small galaxy.
```
</pt>
<tf>
Tokenize the text and return the `input_ids` as TensorFlow tensors. You'll also need to specify the position of the `<mask>` token:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("username/my_awesome_eli5_mlm_model")
>>> inputs = tokenizer(text, return_tensors="tf")
>>> mask_token_index = tf.where(inputs["input_ids"] == tokenizer.mask_token_id)[0, 1]
```

Pass your inputs to the model and return the `logits` of the masked token:

```py
>>> from transformers import TFAutoModelForMaskedLM

>>> model = TFAutoModelForMaskedLM.from_pretrained("username/my_awesome_eli5_mlm_model")
>>> logits = model(**inputs).logits
>>> mask_token_logits = logits[0, mask_token_index, :]
```

Then return the three masked tokens with the highest probability and print them out:

```py
>>> top_3_tokens = tf.math.top_k(mask_token_logits, 3).indices.numpy()

>>> for token in top_3_tokens:
...     print(text.replace(tokenizer.mask_token, tokenizer.decode([token])))
The Milky Way is a spiral galaxy.
The Milky Way is a massive galaxy.
The Milky Way is a small galaxy.
```
</tf>
</frameworkcontent>
