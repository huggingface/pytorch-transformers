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

# Automatic speech recognition

[[open-in-colab]]

<Youtube id="TksaY_FDgnk"/>

Automatic speech recognition (ASR) converts a speech signal to text, mapping a sequence of audio inputs to text outputs. Virtual assistants like Siri and Alexa use ASR models to help users every day, and there are many other useful user-facing applications like live captioning and note-taking during meetings.

This guide will show you how to:

1. Fine-tune [Wav2Vec2](https://huggingface.co/facebook/wav2vec2-base) on the [MInDS-14](https://huggingface.co/datasets/PolyAI/minds14) dataset to transcribe audio to text.
2. Use your fine-tuned model for inference.

<Tip>

To see all architectures and checkpoints compatible with this task, we recommend checking the [task-page](https://huggingface.co/tasks/automatic-speech-recognition)

</Tip>

Before you begin, make sure you have all the necessary libraries installed:

```bash
pip install transformers datasets evaluate jiwer
```

We encourage you to login to your Hugging Face account so you can upload and share your model with the community. When prompted, enter your token to login:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## Load MInDS-14 dataset

Start by loading a smaller subset of the [MInDS-14](https://huggingface.co/datasets/PolyAI/minds14) dataset from the 🤗 Datasets library. This will give you a chance to experiment and make sure everything works before spending more time training on the full dataset.

```py
>>> from datasets import load_dataset, Audio

>>> minds = load_dataset("PolyAI/minds14", name="en-US", split="train[:100]")
```

Split the dataset's `train` split into a train and test set with the [`~Dataset.train_test_split`] method:

```py
>>> minds = minds.train_test_split(test_size=0.2)
```

Then take a look at the dataset:

```py
>>> minds
DatasetDict({
    train: Dataset({
        features: ['path', 'audio', 'transcription', 'english_transcription', 'intent_class', 'lang_id'],
        num_rows: 16
    })
    test: Dataset({
        features: ['path', 'audio', 'transcription', 'english_transcription', 'intent_class', 'lang_id'],
        num_rows: 4
    })
})
```

While the dataset contains a lot of useful information, like `lang_id` and `english_transcription`, this guide focuses on the `audio` and `transcription`. Remove the other columns with the [`~datasets.Dataset.remove_columns`] method:

```py
>>> minds = minds.remove_columns(["english_transcription", "intent_class", "lang_id"])
```

Review the example again:

```py
>>> minds["train"][0]
{'audio': {'array': array([-0.00024414,  0.        ,  0.        , ...,  0.00024414,
          0.00024414,  0.00024414], dtype=float32),
  'path': '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~APP_ERROR/602ba9e2963e11ccd901cd4f.wav',
  'sampling_rate': 8000},
 'path': '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~APP_ERROR/602ba9e2963e11ccd901cd4f.wav',
 'transcription': "hi I'm trying to use the banking app on my phone and currently my checking and savings account balance is not refreshing"}
```

There are two fields:

- `audio`: a 1-dimensional `array` of the speech signal that must be called to load and resample the audio file.
- `transcription`: the target text.

## Preprocess

The next step is to load a Wav2Vec2 processor to process the audio signal:

```py
>>> from transformers import AutoProcessor

>>> processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base")
```

The MInDS-14 dataset has a sampling rate of 8000Hz (you can find this information in its [dataset card](https://huggingface.co/datasets/PolyAI/minds14)), which means you'll need to resample the dataset to 16000Hz to use the pretrained Wav2Vec2 model:

```py
>>> minds = minds.cast_column("audio", Audio(sampling_rate=16_000))
>>> minds["train"][0]
{'audio': {'array': array([-2.38064706e-04, -1.58618059e-04, -5.43987835e-06, ...,
          2.78103951e-04,  2.38446111e-04,  1.18740834e-04], dtype=float32),
  'path': '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~APP_ERROR/602ba9e2963e11ccd901cd4f.wav',
  'sampling_rate': 16000},
 'path': '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~APP_ERROR/602ba9e2963e11ccd901cd4f.wav',
 'transcription': "hi I'm trying to use the banking app on my phone and currently my checking and savings account balance is not refreshing"}
```

As you can see in the `transcription` above, the text contains a mix of uppercase and lowercase characters. The Wav2Vec2 tokenizer is only trained on uppercase characters so you'll need to make sure the text matches the tokenizer's vocabulary:

```py
>>> def uppercase(example):
...     return {"transcription": example["transcription"].upper()}


>>> minds = minds.map(uppercase)
```

Now create a preprocessing function that:

1. Calls the `audio` column to load and resample the audio file.
2. Extracts the `input_values` from the audio file and tokenize the `transcription` column with the processor.

```py
>>> def prepare_dataset(batch):
...     audio = batch["audio"]
...     batch = processor(audio["array"], sampling_rate=audio["sampling_rate"], text=batch["transcription"])
...     batch["input_length"] = len(batch["input_values"][0])
...     return batch
```

To apply the preprocessing function over the entire dataset, use 🤗 Datasets [`~datasets.Dataset.map`] function. You can speed up `map` by increasing the number of processes with the `num_proc` parameter. Remove the columns you don't need with the [`~datasets.Dataset.remove_columns`] method:

```py
>>> encoded_minds = minds.map(prepare_dataset, remove_columns=minds.column_names["train"], num_proc=4)
```

🤗 Transformers doesn't have a data collator for ASR, so you'll need to adapt the [`DataCollatorWithPadding`] to create a batch of examples. It'll also dynamically pad your text and labels to the length of the longest element in its batch (instead of the entire dataset) so they are a uniform length. While it is possible to pad your text in the `tokenizer` function by setting `padding=True`, dynamic padding is more efficient.

Unlike other data collators, this specific data collator needs to apply a different padding method to `input_values` and `labels`:

```py
>>> import torch

>>> from dataclasses import dataclass, field
>>> from typing import Any, Dict, List, Optional, Union


>>> @dataclass
... class DataCollatorCTCWithPadding:
...     processor: AutoProcessor
...     padding: Union[bool, str] = "longest"

...     def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
...         # split inputs and labels since they have to be of different lengths and need
...         # different padding methods
...         input_features = [{"input_values": feature["input_values"][0]} for feature in features]
...         label_features = [{"input_ids": feature["labels"]} for feature in features]

...         batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")

...         labels_batch = self.processor.pad(labels=label_features, padding=self.padding, return_tensors="pt")

...         # replace padding with -100 to ignore loss correctly
...         labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

...         batch["labels"] = labels

...         return batch
```

Now instantiate your `DataCollatorForCTCWithPadding`:

```py
>>> data_collator = DataCollatorCTCWithPadding(processor=processor, padding="longest")
```

## Evaluate

Including a metric during training is often helpful for evaluating your model's performance. You can quickly load an evaluation method with the 🤗 [Evaluate](https://huggingface.co/docs/evaluate/index) library. For this task, load the [word error rate](https://huggingface.co/spaces/evaluate-metric/wer) (WER) metric (refer to the 🤗 Evaluate [quick tour](https://huggingface.co/docs/evaluate/a_quick_tour) to learn more about loading and computing metrics):

```py
>>> import evaluate

>>> wer = evaluate.load("wer")
```

Then create a function that passes your predictions and labels to [`~evaluate.EvaluationModule.compute`] to calculate the WER:

```py
>>> import numpy as np


>>> def compute_metrics(pred):
...     pred_logits = pred.predictions
...     pred_ids = np.argmax(pred_logits, axis=-1)

...     pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

...     pred_str = processor.batch_decode(pred_ids)
...     label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

...     wer = wer.compute(predictions=pred_str, references=label_str)

...     return {"wer": wer}
```

Your `compute_metrics` function is ready to go now, and you'll return to it when you setup your training.

## Train

<frameworkcontent>
<pt>
<Tip>

If you aren't familiar with finetuning a model with the [`Trainer`], take a look at the basic tutorial [here](../training#train-with-pytorch-trainer)!

</Tip>

You are now ready to start training your model! Load Wav2Vec2 with [`AutoModelForCTC`]. Specify the reduction to apply with the `ctc_loss_reduction` parameter. It is often better to use the average instead of the default summation:

```py
>>> from transformers import AutoModelForCTC, TrainingArguments, Trainer

>>> model = AutoModelForCTC.from_pretrained(
...     "facebook/wav2vec2-base",
...     ctc_loss_reduction="mean",
...     pad_token_id=processor.tokenizer.pad_token_id,
... )
```

At this point, only three steps remain:

1. Define your training hyperparameters in [`TrainingArguments`]. The only required parameter is `output_dir` which specifies where to save your model. You'll push this model to the Hub by setting `push_to_hub=True` (you need to be signed in to Hugging Face to upload your model). At the end of each epoch, the [`Trainer`] will evaluate the WER and save the training checkpoint.
2. Pass the training arguments to [`Trainer`] along with the model, dataset, tokenizer, data collator, and `compute_metrics` function.
3. Call [`~Trainer.train`] to fine-tune your model.

```py
>>> training_args = TrainingArguments(
...     output_dir="my_awesome_asr_mind_model",
...     per_device_train_batch_size=8,
...     gradient_accumulation_steps=2,
...     learning_rate=1e-5,
...     warmup_steps=500,
...     max_steps=2000,
...     gradient_checkpointing=True,
...     fp16=True,
...     group_by_length=True,
...     eval_strategy="steps",
...     per_device_eval_batch_size=8,
...     save_steps=1000,
...     eval_steps=1000,
...     logging_steps=25,
...     load_best_model_at_end=True,
...     metric_for_best_model="wer",
...     greater_is_better=False,
...     push_to_hub=True,
... )

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=encoded_minds["train"],
...     eval_dataset=encoded_minds["test"],
...     processing_class=processor,
...     data_collator=data_collator,
...     compute_metrics=compute_metrics,
... )

>>> trainer.train()
```

Once training is completed, share your model to the Hub with the [`~transformers.Trainer.push_to_hub`] method so it can be accessible to everyone:

```py
>>> trainer.push_to_hub()
```
</pt>
</frameworkcontent>

<Tip>

For a more in-depth example of how to fine-tune a model for automatic speech recognition, take a look at this blog [post](https://huggingface.co/blog/fine-tune-wav2vec2-english) for English ASR and this [post](https://huggingface.co/blog/fine-tune-xlsr-wav2vec2) for multilingual ASR.

</Tip>

## Inference

Great, now that you've fine-tuned a model, you can use it for inference!

Load an audio file you'd like to run inference on. Remember to resample the sampling rate of the audio file to match the sampling rate of the model if you need to!

```py
>>> from datasets import load_dataset, Audio

>>> dataset = load_dataset("PolyAI/minds14", "en-US", split="train")
>>> dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
>>> sampling_rate = dataset.features["audio"].sampling_rate
>>> audio_file = dataset[0]["audio"]["path"]
```

The simplest way to try out your fine-tuned model for inference is to use it in a [`pipeline`]. Instantiate a `pipeline` for automatic speech recognition with your model, and pass your audio file to it:

```py
>>> from transformers import pipeline

>>> transcriber = pipeline("automatic-speech-recognition", model="stevhliu/my_awesome_asr_minds_model")
>>> transcriber(audio_file)
{'text': 'I WOUD LIKE O SET UP JOINT ACOUNT WTH Y PARTNER'}
```

<Tip>

The transcription is decent, but it could be better! Try finetuning your model on more examples to get even better results!

</Tip>

You can also manually replicate the results of the `pipeline` if you'd like:

<frameworkcontent>
<pt>
Load a processor to preprocess the audio file and transcription and return the `input` as PyTorch tensors:

```py
>>> from transformers import AutoProcessor

>>> processor = AutoProcessor.from_pretrained("stevhliu/my_awesome_asr_mind_model")
>>> inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
```

Pass your inputs to the model and return the logits:

```py
>>> from transformers import AutoModelForCTC

>>> model = AutoModelForCTC.from_pretrained("stevhliu/my_awesome_asr_mind_model")
>>> with torch.no_grad():
...     logits = model(**inputs).logits
```

Get the predicted `input_ids` with the highest probability, and use the processor to decode the predicted `input_ids` back into text:

```py
>>> import torch

>>> predicted_ids = torch.argmax(logits, dim=-1)
>>> transcription = processor.batch_decode(predicted_ids)
>>> transcription
['I WOUL LIKE O SET UP JOINT ACOUNT WTH Y PARTNER']
```
</pt>
</frameworkcontent>
