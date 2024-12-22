<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# How to create a custom pipeline?

In this guide, we will see how to create a custom pipeline and share it on the [Hub](https://hf.co/models) or add it to the
🤗 Transformers library.

First and foremost, you need to decide the raw entries the pipeline will be able to take. It can be strings, raw bytes,
dictionaries or whatever seems to be the most likely desired input. Try to keep these inputs as pure Python as possible
as it makes compatibility easier (even through other languages via JSON). Those will be the `inputs` of the
pipeline (`preprocess`).

Then define the `outputs`. Same policy as the `inputs`. The simpler, the better. Those will be the outputs of
`postprocess` method.

Start by inheriting the base class `Pipeline` with the 4 methods needed to implement `preprocess`,
`_forward`, `postprocess`, and `_sanitize_parameters`.


```python
from transformers import Pipeline


class MyPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        if "maybe_arg" in kwargs:
            preprocess_kwargs["maybe_arg"] = kwargs["maybe_arg"]
        return preprocess_kwargs, {}, {}

    def preprocess(self, inputs, maybe_arg=2):
        model_input = Tensor(inputs["input_ids"])
        return {"model_input": model_input}

    def _forward(self, model_inputs):
        # model_inputs == {"model_input": model_input}
        outputs = self.model(**model_inputs)
        # Maybe {"logits": Tensor(...)}
        return outputs

    def postprocess(self, model_outputs):
        best_class = model_outputs["logits"].softmax(-1)
        return best_class
```

The structure of this breakdown is to support relatively seamless support for CPU/GPU, while supporting doing
pre/postprocessing on the CPU on different threads

`preprocess` will take the originally defined inputs, and turn them into something feedable to the model. It might
contain more information and is usually a `Dict`.

`_forward` is the implementation detail and is not meant to be called directly. `forward` is the preferred
called method as it contains safeguards to make sure everything is working on the expected device. If anything is
linked to a real model it belongs in the `_forward` method, anything else is in the preprocess/postprocess.

`postprocess` methods will take the output of `_forward` and turn it into the final output that was decided
earlier.

`_sanitize_parameters` exists to allow users to pass any parameters whenever they wish, be it at initialization
time `pipeline(...., maybe_arg=4)` or at call time `pipe = pipeline(...); output = pipe(...., maybe_arg=4)`.

The returns of `_sanitize_parameters` are the 3 dicts of kwargs that will be passed directly to `preprocess`,
`_forward`, and `postprocess`. Don't fill anything if the caller didn't call with any extra parameter. That
allows to keep the default arguments in the function definition which is always more "natural".

A classic example would be a `top_k` argument in the post processing in classification tasks.

```python
>>> pipe = pipeline("my-new-task")
>>> pipe("This is a test")
[{"label": "1-star", "score": 0.8}, {"label": "2-star", "score": 0.1}, {"label": "3-star", "score": 0.05}
{"label": "4-star", "score": 0.025}, {"label": "5-star", "score": 0.025}]

>>> pipe("This is a test", top_k=2)
[{"label": "1-star", "score": 0.8}, {"label": "2-star", "score": 0.1}]
```

In order to achieve that, we'll update our `postprocess` method with a default parameter to `5`. and edit
`_sanitize_parameters` to allow this new parameter.


```python
def postprocess(self, model_outputs, top_k=5):
    best_class = model_outputs["logits"].softmax(-1)
    # Add logic to handle top_k
    return best_class


def _sanitize_parameters(self, **kwargs):
    preprocess_kwargs = {}
    if "maybe_arg" in kwargs:
        preprocess_kwargs["maybe_arg"] = kwargs["maybe_arg"]

    postprocess_kwargs = {}
    if "top_k" in kwargs:
        postprocess_kwargs["top_k"] = kwargs["top_k"]
    return preprocess_kwargs, {}, postprocess_kwargs
```

Try to keep the inputs/outputs very simple and ideally JSON-serializable as it makes the pipeline usage very easy
without requiring users to understand new kinds of objects. It's also relatively common to support many different types
of arguments for ease of use (audio files, which can be filenames, URLs or pure bytes)



## Adding it to the list of supported tasks

To register your `new-task` to the list of supported tasks, you have to add it to the `PIPELINE_REGISTRY`:

```python
from transformers.pipelines import PIPELINE_REGISTRY

PIPELINE_REGISTRY.register_pipeline(
    "new-task",
    pipeline_class=MyPipeline,
    pt_model=AutoModelForSequenceClassification,
)
```

You can specify a default model if you want, in which case it should come with a specific revision (which can be the name of a branch or a commit hash, here we took `"abcdef"`) as well as the type:

```python
PIPELINE_REGISTRY.register_pipeline(
    "new-task",
    pipeline_class=MyPipeline,
    pt_model=AutoModelForSequenceClassification,
    default={"pt": ("user/awesome_model", "abcdef")},
    type="text",  # current support type: text, audio, image, multimodal
)
```

## Share your pipeline on the Hub

To share your custom pipeline on the Hub, you just have to save the custom code of your `Pipeline` subclass in a
python file. For instance, let's say we want to use a custom pipeline for sentence pair classification like this:

```py
import numpy as np

from transformers import Pipeline


def softmax(outputs):
    maxes = np.max(outputs, axis=-1, keepdims=True)
    shifted_exp = np.exp(outputs - maxes)
    return shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)


class PairClassificationPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        if "second_text" in kwargs:
            preprocess_kwargs["second_text"] = kwargs["second_text"]
        return preprocess_kwargs, {}, {}

    def preprocess(self, text, second_text=None):
        return self.tokenizer(text, text_pair=second_text, return_tensors=self.framework)

    def _forward(self, model_inputs):
        return self.model(**model_inputs)

    def postprocess(self, model_outputs):
        logits = model_outputs.logits[0].numpy()
        probabilities = softmax(logits)

        best_class = np.argmax(probabilities)
        label = self.model.config.id2label[best_class]
        score = probabilities[best_class].item()
        logits = logits.tolist()
        return {"label": label, "score": score, "logits": logits}
```

The implementation is framework agnostic, and will work for PyTorch and TensorFlow models. If we have saved this in
a file named `pair_classification.py`, we can then import it and register it like this.

```py
from pair_classification import PairClassificationPipeline
from transformers.pipelines import PIPELINE_REGISTRY
from transformers import AutoModelForSequenceClassification, TFAutoModelForSequenceClassification

PIPELINE_REGISTRY.register_pipeline(
    "pair-classification",
    pipeline_class=PairClassificationPipeline,
    pt_model=AutoModelForSequenceClassification,
    tf_model=TFAutoModelForSequenceClassification,
)
```

The [register_pipeline](https://github.com/huggingface/transformers/blob/9feae5fb0164e89d4998e5776897c16f7330d3df/src/transformers/pipelines/base.py#L1387) function registers the pipeline details (task type, pipeline class, supported backends) to a models `config.json` file.

```json
  "custom_pipelines": {
    "pair-classification": {
      "impl": "pair_classification.PairClassificationPipeline",
      "pt": [
        "AutoModelForSequenceClassification"
      ],
      "tf": [
        "TFAutoModelForSequenceClassification"
      ],
    }
  },
```

Once this is done, we can use it with a pretrained model. For instance `sgugger/finetuned-bert-mrpc` has been
fine-tuned on the MRPC dataset, which classifies pairs of sentences as paraphrases or not.

```py
from transformers import pipeline

classifier = pipeline("pair-classification", model="sgugger/finetuned-bert-mrpc")
```

Then we can share it on the Hub by using the `push_to_hub` method:

```py
classifier.push_to_hub("test-dynamic-pipeline")
```

This will copy the file where you defined `PairClassificationPipeline` inside the folder `"test-dynamic-pipeline"`,
along with saving the model and tokenizer of the pipeline, before pushing everything into the repository
`{your_username}/test-dynamic-pipeline`. After that, anyone can use it as long as they provide the option
`trust_remote_code=True`:

```py
from transformers import pipeline

classifier = pipeline(model="{your_username}/test-dynamic-pipeline", trust_remote_code=True)
```

## Add the pipeline to 🤗 Transformers

If you want to contribute your pipeline to 🤗 Transformers, you will need to add a new module in the `pipelines` submodule
with the code of your pipeline, then add it to the list of tasks defined in `pipelines/__init__.py`.

Then you will need to add tests. Create a new file `tests/test_pipelines_MY_PIPELINE.py` with examples of the other tests.

The `run_pipeline_test` function will be very generic and run on small random models on every possible
architecture as defined by `model_mapping` and `tf_model_mapping`.

This is very important to test future compatibility, meaning if someone adds a new model for
`XXXForQuestionAnswering` then the pipeline test will attempt to run on it. Because the models are random it's
impossible to check for actual values, that's why there is a helper `ANY` that will simply attempt to match the
output of the pipeline TYPE.

You also *need* to implement 2 (ideally 4) tests.

- `test_small_model_pt` : Define 1 small model for this pipeline (doesn't matter if the results don't make sense)
  and test the pipeline outputs. The results should be the same as `test_small_model_tf`.
- `test_small_model_tf` : Define 1 small model for this pipeline (doesn't matter if the results don't make sense)
  and test the pipeline outputs. The results should be the same as `test_small_model_pt`.
- `test_large_model_pt` (`optional`): Tests the pipeline on a real pipeline where the results are supposed to
  make sense. These tests are slow and should be marked as such. Here the goal is to showcase the pipeline and to make
  sure there is no drift in future releases.
- `test_large_model_tf` (`optional`): Tests the pipeline on a real pipeline where the results are supposed to
  make sense. These tests are slow and should be marked as such. Here the goal is to showcase the pipeline and to make
  sure there is no drift in future releases.
