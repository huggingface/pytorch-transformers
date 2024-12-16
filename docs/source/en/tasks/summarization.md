<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Summarization

[[open-in-colab]]

<Youtube id="yHnr5Dk2zCI"/>

Summarization creates a shorter version of a document or an article that captures all the important information. Along with translation, it is another example of a task that can be formulated as a sequence-to-sequence task. Summarization can be:

- Extractive: extract the most relevant information from a document.
- Abstractive: generate new text that captures the most relevant information.

This guide will show you how to:

1. Finetune [T5](https://huggingface.co/google-t5/t5-small) on the California state bill subset of the [BillSum](https://huggingface.co/datasets/billsum) dataset for abstractive summarization.
2. Use your finetuned model for inference.

<Tip>

To see all architectures and checkpoints compatible with this task, we recommend checking the [task-page](https://huggingface.co/tasks/summarization)

</Tip>

Before you begin, make sure you have all the necessary libraries installed:

```bash
pip install transformers datasets evaluate rouge_score
```

We encourage you to login to your Hugging Face account so you can upload and share your model with the community. When prompted, enter your token to login:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## Load BillSum dataset

Start by loading the smaller California state bill subset of the BillSum dataset from the 🤗 Datasets library:

```py
>>> from datasets import load_dataset

>>> billsum = load_dataset("billsum", split="ca_test")
```

Split the dataset into a train and test set with the [`~datasets.Dataset.train_test_split`] method:

```py
>>> billsum = billsum.train_test_split(test_size=0.2)
```

Then take a look at an example:

```py
>>> billsum["train"][0]
{'summary': 'Existing law authorizes state agencies to enter into contracts for the acquisition of goods or services upon approval by the Department of General Services. Existing law sets forth various requirements and prohibitions for those contracts, including, but not limited to, a prohibition on entering into contracts for the acquisition of goods or services of $100,000 or more with a contractor that discriminates between spouses and domestic partners or same-sex and different-sex couples in the provision of benefits. Existing law provides that a contract entered into in violation of those requirements and prohibitions is void and authorizes the state or any person acting on behalf of the state to bring a civil action seeking a determination that a contract is in violation and therefore void. Under existing law, a willful violation of those requirements and prohibitions is a misdemeanor.\nThis bill would also prohibit a state agency from entering into contracts for the acquisition of goods or services of $100,000 or more with a contractor that discriminates between employees on the basis of gender identity in the provision of benefits, as specified. By expanding the scope of a crime, this bill would impose a state-mandated local program.\nThe California Constitution requires the state to reimburse local agencies and school districts for certain costs mandated by the state. Statutory provisions establish procedures for making that reimbursement.\nThis bill would provide that no reimbursement is required by this act for a specified reason.',
 'text': 'The people of the State of California do enact as follows:\n\n\nSECTION 1.\nSection 10295.35 is added to the Public Contract Code, to read:\n10295.35.\n(a) (1) Notwithstanding any other law, a state agency shall not enter into any contract for the acquisition of goods or services in the amount of one hundred thousand dollars ($100,000) or more with a contractor that, in the provision of benefits, discriminates between employees on the basis of an employee’s or dependent’s actual or perceived gender identity, including, but not limited to, the employee’s or dependent’s identification as transgender.\n(2) For purposes of this section, “contract” includes contracts with a cumulative amount of one hundred thousand dollars ($100,000) or more per contractor in each fiscal year.\n(3) For purposes of this section, an employee health plan is discriminatory if the plan is not consistent with Section 1365.5 of the Health and Safety Code and Section 10140 of the Insurance Code.\n(4) The requirements of this section shall apply only to those portions of a contractor’s operations that occur under any of the following conditions:\n(A) Within the state.\n(B) On real property outside the state if the property is owned by the state or if the state has a right to occupy the property, and if the contractor’s presence at that location is connected to a contract with the state.\n(C) Elsewhere in the United States where work related to a state contract is being performed.\n(b) Contractors shall treat as confidential, to the maximum extent allowed by law or by the requirement of the contractor’s insurance provider, any request by an employee or applicant for employment benefits or any documentation of eligibility for benefits submitted by an employee or applicant for employment.\n(c) After taking all reasonable measures to find a contractor that complies with this section, as determined by the state agency, the requirements of this section may be waived under any of the following circumstances:\n(1) There is only one prospective contractor willing to enter into a specific contract with the state agency.\n(2) The contract is necessary to respond to an emergency, as determined by the state agency, that endangers the public health, welfare, or safety, or the contract is necessary for the provision of essential services, and no entity that complies with the requirements of this section capable of responding to the emergency is immediately available.\n(3) The requirements of this section violate, or are inconsistent with, the terms or conditions of a grant, subvention, or agreement, if the agency has made a good faith attempt to change the terms or conditions of any grant, subvention, or agreement to authorize application of this section.\n(4) The contractor is providing wholesale or bulk water, power, or natural gas, the conveyance or transmission of the same, or ancillary services, as required for ensuring reliable services in accordance with good utility practice, if the purchase of the same cannot practically be accomplished through the standard competitive bidding procedures and the contractor is not providing direct retail services to end users.\n(d) (1) A contractor shall not be deemed to discriminate in the provision of benefits if the contractor, in providing the benefits, pays the actual costs incurred in obtaining the benefit.\n(2) If a contractor is unable to provide a certain benefit, despite taking reasonable measures to do so, the contractor shall not be deemed to discriminate in the provision of benefits.\n(e) (1) Every contract subject to this chapter shall contain a statement by which the contractor certifies that the contractor is in compliance with this section.\n(2) The department or other contracting agency shall enforce this section pursuant to its existing enforcement powers.\n(3) (A) If a contractor falsely certifies that it is in compliance with this section, the contract with that contractor shall be subject to Article 9 (commencing with Section 10420), unless, within a time period specified by the department or other contracting agency, the contractor provides to the department or agency proof that it has complied, or is in the process of complying, with this section.\n(B) The application of the remedies or penalties contained in Article 9 (commencing with Section 10420) to a contract subject to this chapter shall not preclude the application of any existing remedies otherwise available to the department or other contracting agency under its existing enforcement powers.\n(f) Nothing in this section is intended to regulate the contracting practices of any local jurisdiction.\n(g) This section shall be construed so as not to conflict with applicable federal laws, rules, or regulations. In the event that a court or agency of competent jurisdiction holds that federal law, rule, or regulation invalidates any clause, sentence, paragraph, or section of this code or the application thereof to any person or circumstances, it is the intent of the state that the court or agency sever that clause, sentence, paragraph, or section so that the remainder of this section shall remain in effect.\nSEC. 2.\nSection 10295.35 of the Public Contract Code shall not be construed to create any new enforcement authority or responsibility in the Department of General Services or any other contracting agency.\nSEC. 3.\nNo reimbursement is required by this act pursuant to Section 6 of Article XIII\u2009B of the California Constitution because the only costs that may be incurred by a local agency or school district will be incurred because this act creates a new crime or infraction, eliminates a crime or infraction, or changes the penalty for a crime or infraction, within the meaning of Section 17556 of the Government Code, or changes the definition of a crime within the meaning of Section 6 of Article XIII\u2009B of the California Constitution.',
 'title': 'An act to add Section 10295.35 to the Public Contract Code, relating to public contracts.'}
```

There are two fields that you'll want to use:

- `text`: the text of the bill which'll be the input to the model.
- `summary`: a condensed version of `text` which'll be the model target.

## Preprocess

The next step is to load a T5 tokenizer to process `text` and `summary`:

```py
>>> from transformers import AutoTokenizer

>>> checkpoint = "google-t5/t5-small"
>>> tokenizer = AutoTokenizer.from_pretrained(checkpoint)
```

The preprocessing function you want to create needs to:

1. Prefix the input with a prompt so T5 knows this is a summarization task. Some models capable of multiple NLP tasks require prompting for specific tasks.
2. Use the keyword `text_target` argument when tokenizing labels.
3. Truncate sequences to be no longer than the maximum length set by the `max_length` parameter.

```py
>>> prefix = "summarize: "


>>> def preprocess_function(examples):
...     inputs = [prefix + doc for doc in examples["text"]]
...     model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

...     labels = tokenizer(text_target=examples["summary"], max_length=128, truncation=True)

...     model_inputs["labels"] = labels["input_ids"]
...     return model_inputs
```

To apply the preprocessing function over the entire dataset, use 🤗 Datasets [`~datasets.Dataset.map`] method. You can speed up the `map` function by setting `batched=True` to process multiple elements of the dataset at once:

```py
>>> tokenized_billsum = billsum.map(preprocess_function, batched=True)
```

Now create a batch of examples using [`DataCollatorForSeq2Seq`]. It's more efficient to *dynamically pad* the sentences to the longest length in a batch during collation, instead of padding the whole dataset to the maximum length.

<frameworkcontent>
<pt>

```py
>>> from transformers import DataCollatorForSeq2Seq

>>> data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)
```
</pt>
<tf>

```py
>>> from transformers import DataCollatorForSeq2Seq

>>> data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint, return_tensors="tf")
```
</tf>
</frameworkcontent>

## Evaluate

Including a metric during training is often helpful for evaluating your model's performance. You can quickly load a evaluation method with the 🤗 [Evaluate](https://huggingface.co/docs/evaluate/index) library. For this task, load the [ROUGE](https://huggingface.co/spaces/evaluate-metric/rouge) metric (see the 🤗 Evaluate [quick tour](https://huggingface.co/docs/evaluate/a_quick_tour) to learn more about how to load and compute a metric):

```py
>>> import evaluate

>>> rouge = evaluate.load("rouge")
```

Then create a function that passes your predictions and labels to [`~evaluate.EvaluationModule.compute`] to calculate the ROUGE metric:

```py
>>> import numpy as np


>>> def compute_metrics(eval_pred):
...     predictions, labels = eval_pred
...     decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
...     labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
...     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

...     result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

...     prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
...     result["gen_len"] = np.mean(prediction_lens)

...     return {k: round(v, 4) for k, v in result.items()}
```

Your `compute_metrics` function is ready to go now, and you'll return to it when you setup your training.

## Train

<frameworkcontent>
<pt>
<Tip>

If you aren't familiar with finetuning a model with the [`Trainer`], take a look at the basic tutorial [here](../training#train-with-pytorch-trainer)!

</Tip>

You're ready to start training your model now! Load T5 with [`AutoModelForSeq2SeqLM`]:

```py
>>> from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

>>> model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
```

At this point, only three steps remain:

1. Define your training hyperparameters in [`Seq2SeqTrainingArguments`]. The only required parameter is `output_dir` which specifies where to save your model. You'll push this model to the Hub by setting `push_to_hub=True` (you need to be signed in to Hugging Face to upload your model). At the end of each epoch, the [`Trainer`] will evaluate the ROUGE metric and save the training checkpoint.
2. Pass the training arguments to [`Seq2SeqTrainer`] along with the model, dataset, tokenizer, data collator, and `compute_metrics` function.
3. Call [`~Trainer.train`] to finetune your model.

```py
>>> training_args = Seq2SeqTrainingArguments(
...     output_dir="my_awesome_billsum_model",
...     eval_strategy="epoch",
...     learning_rate=2e-5,
...     per_device_train_batch_size=16,
...     per_device_eval_batch_size=16,
...     weight_decay=0.01,
...     save_total_limit=3,
...     num_train_epochs=4,
...     predict_with_generate=True,
...     fp16=True, #change to bf16=True for XPU
...     push_to_hub=True,
... )

>>> trainer = Seq2SeqTrainer(
...     model=model,
...     args=training_args,
...     train_dataset=tokenized_billsum["train"],
...     eval_dataset=tokenized_billsum["test"],
...     processing_class=tokenizer,
...     data_collator=data_collator,
...     compute_metrics=compute_metrics,
... )

>>> trainer.train()
```

Once training is completed, share your model to the Hub with the [`~transformers.Trainer.push_to_hub`] method so everyone can use your model:

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

Then you can load T5 with [`TFAutoModelForSeq2SeqLM`]:

```py
>>> from transformers import TFAutoModelForSeq2SeqLM

>>> model = TFAutoModelForSeq2SeqLM.from_pretrained(checkpoint)
```

Convert your datasets to the `tf.data.Dataset` format with [`~transformers.TFPreTrainedModel.prepare_tf_dataset`]:

```py
>>> tf_train_set = model.prepare_tf_dataset(
...     tokenized_billsum["train"],
...     shuffle=True,
...     batch_size=16,
...     collate_fn=data_collator,
... )

>>> tf_test_set = model.prepare_tf_dataset(
...     tokenized_billsum["test"],
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

The last two things to setup before you start training is to compute the ROUGE score from the predictions, and provide a way to push your model to the Hub. Both are done by using [Keras callbacks](../main_classes/keras_callbacks).

Pass your `compute_metrics` function to [`~transformers.KerasMetricCallback`]:

```py
>>> from transformers.keras_callbacks import KerasMetricCallback

>>> metric_callback = KerasMetricCallback(metric_fn=compute_metrics, eval_dataset=tf_test_set)
```

Specify where to push your model and tokenizer in the [`~transformers.PushToHubCallback`]:

```py
>>> from transformers.keras_callbacks import PushToHubCallback

>>> push_to_hub_callback = PushToHubCallback(
...     output_dir="my_awesome_billsum_model",
...     tokenizer=tokenizer,
... )
```

Then bundle your callbacks together:

```py
>>> callbacks = [metric_callback, push_to_hub_callback]
```

Finally, you're ready to start training your model! Call [`fit`](https://keras.io/api/models/model_training_apis/#fit-method) with your training and validation datasets, the number of epochs, and your callbacks to finetune the model:

```py
>>> model.fit(x=tf_train_set, validation_data=tf_test_set, epochs=3, callbacks=callbacks)
```

Once training is completed, your model is automatically uploaded to the Hub so everyone can use it!
</tf>
</frameworkcontent>

<Tip>

For a more in-depth example of how to finetune a model for summarization, take a look at the corresponding
[PyTorch notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/summarization.ipynb)
or [TensorFlow notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/summarization-tf.ipynb).

</Tip>

## Inference

Great, now that you've finetuned a model, you can use it for inference!

Come up with some text you'd like to summarize. For T5, you need to prefix your input depending on the task you're working on. For summarization you should prefix your input as shown below:

```py
>>> text = "summarize: The Inflation Reduction Act lowers prescription drug costs, health care costs, and energy costs. It's the most aggressive action on tackling the climate crisis in American history, which will lift up American workers and create good-paying, union jobs across the country. It'll lower the deficit and ask the ultra-wealthy and corporations to pay their fair share. And no one making under $400,000 per year will pay a penny more in taxes."
```

The simplest way to try out your finetuned model for inference is to use it in a [`pipeline`]. Instantiate a `pipeline` for summarization with your model, and pass your text to it:

```py
>>> from transformers import pipeline

>>> summarizer = pipeline("summarization", model="username/my_awesome_billsum_model")
>>> summarizer(text)
[{"summary_text": "The Inflation Reduction Act lowers prescription drug costs, health care costs, and energy costs. It's the most aggressive action on tackling the climate crisis in American history, which will lift up American workers and create good-paying, union jobs across the country."}]
```

You can also manually replicate the results of the `pipeline` if you'd like:


<frameworkcontent>
<pt>
Tokenize the text and return the `input_ids` as PyTorch tensors:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("username/my_awesome_billsum_model")
>>> inputs = tokenizer(text, return_tensors="pt").input_ids
```

Use the [`~generation.GenerationMixin.generate`] method to create the summarization. For more details about the different text generation strategies and parameters for controlling generation, check out the [Text Generation](../main_classes/text_generation) API.

```py
>>> from transformers import AutoModelForSeq2SeqLM

>>> model = AutoModelForSeq2SeqLM.from_pretrained("username/my_awesome_billsum_model")
>>> outputs = model.generate(inputs, max_new_tokens=100, do_sample=False)
```

Decode the generated token ids back into text:

```py
>>> tokenizer.decode(outputs[0], skip_special_tokens=True)
'the inflation reduction act lowers prescription drug costs, health care costs, and energy costs. it's the most aggressive action on tackling the climate crisis in american history. it will ask the ultra-wealthy and corporations to pay their fair share.'
```
</pt>
<tf>
Tokenize the text and return the `input_ids` as TensorFlow tensors:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("username/my_awesome_billsum_model")
>>> inputs = tokenizer(text, return_tensors="tf").input_ids
```

Use the [`~transformers.generation_tf_utils.TFGenerationMixin.generate`] method to create the summarization. For more details about the different text generation strategies and parameters for controlling generation, check out the [Text Generation](../main_classes/text_generation) API.

```py
>>> from transformers import TFAutoModelForSeq2SeqLM

>>> model = TFAutoModelForSeq2SeqLM.from_pretrained("username/my_awesome_billsum_model")
>>> outputs = model.generate(inputs, max_new_tokens=100, do_sample=False)
```

Decode the generated token ids back into text:

```py
>>> tokenizer.decode(outputs[0], skip_special_tokens=True)
'the inflation reduction act lowers prescription drug costs, health care costs, and energy costs. it's the most aggressive action on tackling the climate crisis in american history. it will ask the ultra-wealthy and corporations to pay their fair share.'
```
</tf>
</frameworkcontent>
