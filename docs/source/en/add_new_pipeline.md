<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Add a new pipeline

You can make [`Pipeline`] your own by subclassing it, and then implementing a few methods. Share the code with the community on the [Hub](https://hf.co) and register the pipeline with Transformers so that everyone can quickly and easily use it.

This guide will walk you through the process of adding a new pipeline to Transformers.

## Design choices

At a bare minimum, you only need to provide [`Pipeline`] with an appropriate input for a task. This is also where you should begin when designing your pipeline.

Decide what input types [`Pipeline`] can accept. It can be strings, raw bytes, dictionaries, and so on. Try to keep the inputs in pure Python where possible because it's more compatible. Next, decide on the output [`Pipeline`] should return. Again, keeping the output in Python is the simplest and best option because it's easier to work with.

Keeping the inputs and outputs simple, and ideally JSON-serializable, makes it easier for users to run your [`Pipeline`] without needing to learn new object types. It's also common to support many different input types, for even greater ease of use. For example, making an audio file acceptable from a filename, URL, or raw bytes gives the user more flexibility in how they provide the audio data.

## Implement a pipeline

With an input and output decided, you can start implementing [`Pipeline`]. Your pipeline should inherit from the base [`Pipeline`] class and include 4 methods.

```py
from transformers import Pipeline

class MyPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):

    def preprocess(self, inputs, args=2):

    def _forward(self, model_inputs):

    def postprocess(self, model_outputs):
```

1. `preprocess` takes the inputs and transforms them into the appropriate input format for the model.

```py
def preprocess(self, inputs, maybe_arg=2):
    model_input = Tensor(inputs["input_ids"])
    return {"model_input": model_input}
```

1. `_forward` shouldn't be called directly. `forward` is the preferred method because it includes safeguards to make sure everything works correctly on the expected device. Anything linked to the model belongs in `_forward`, and everything else belongs in either `preprocess` or `postprocess`.

```py
def _forward(self, model_inputs):
    outputs = self.model(**model_inputs)
    return outputs
```

1. `postprocess` generates the final output from the models output in `_forward`.

```py
def postprocess(self, model_outputs, top_k=5):
    best_class = model_outputs["logits"].softmax(-1)
    return best_class
```

1. `_sanitize_parameters` lets users pass additional parameters to [`Pipeline`]. This could be during the initialization or when [`Pipeline`] is called. `_sanitize_parameters` returns 3 dicts of additional keyword arguments that are passed directly to `preprocess`, `_forward`, and `postprocess`. Don't add anything if a user didn't call the pipeline with an extra parameters! This keeps the default arguments in the function definition which is always more *natural*.

For example, add a `top_k` parameter in `postprocess` to return the top 5 most likely classes. Then in `_sanitize_parameters`, check if the user passed in `top_k` and add it to `postprocess_kwargs`.

```py
def _sanitize_parameters(self, **kwargs):
    preprocess_kwargs = {}
    if "maybe_arg" in kwargs:
        preprocess_kwargs["maybe_arg"] = kwargs["maybe_arg"]

    postprocess_kwargs = {}
    if "top_k" in kwargs:
        postprocess_kwargs["top_k"] = kwargs["top_k"]
    return preprocess_kwargs, {}, postprocess_kwargs
```

Now the pipeline can return the top most likely labels if they choose to.

```py
from transformers import pipeline

pipeline = pipeline("my-task")
# returns 3 most likely labels
pipeline("This is the best meal I've ever had", top_k=3)
# returns 5 most likely labels by default
pipeline("This is the best meal I've ever had")
```

## Register a pipeline

Register the new task your pipeline supports in the `PIPELINE_REGISTRY`. The registry defines:

- the machine learning framework the pipeline supports with either `pt_model` or `tf_model` (add both to ensure it works with either frameworks)
- a default model which should come from a specific revision (branch, or commit hash) where the model works as expected with `default`
- the expected input with `type`

```py
from transformers.pipelines import PIPELINE_REGISTRY
from transformers import AutoModelForSequenceClassification, TFAutoModelForSequenceClassification

PIPELINE_REGISTRY.register_pipeline(
    "new-task",
    pipeline_class=MyPipeline,
    pt_model=AutoModelForSequenceClassification,
    tf_model=TFAutoModelForSequenceClassification,
    default={"pt": ("user/awesome-model", "branch-name")},
    type="text",
)
```

## Share your pipeline

Share your pipeline with the community on the [Hub](https://hf.co) or you can add it directly to Transformers.

It's faster to upload your pipeline code to the Hub because it doesn't require any review from the Transformers team. Adding the pipeline to Transformers may be slower because it requires a review and you need to add tests to ensure the [`Pipeline`] works.

### Upload to the Hub

Add your pipeline code to the Hub in a Python file.

For example, a custom pipeline for sentence pair classification might look the following code below.

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

Call [`~Pipeline.push_to_hub`] to push the pipeline to the Hub. The Python file containing the code is copied to the Hub, and the pipelines model and tokenizer are also saved and pushed to the Hub. Your pipeline should now be available on the Hub under your namespace.

```py
from transformers import pipeline

pipeline = pipeline(task="pair-classification", model="sgugger/finetuned-bert-mrpc")
pipeline.push_to_hub("pair-classification-pipeline")
```

To use the pipeline, add `trust_remote_code=True` when loading the pipeline.

```py
from transformers import pipeline

pipeline = pipeline(task="pair-classification", trust_remote_code=True)
```

### Add to Transformers

Adding a custom pipeline to Transformers requires adding tests to make sure everything works as expected, and requesting a review from the Transformers team.

Add your pipeline code as a new module to the [pipelines](https://github.com/huggingface/transformers/tree/main/src/transformers/pipelines) submodule, and add it to the list of tasks defined in [pipelines/__init__.py](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/__init__.py).

Next, add a new test for the pipeline in [transformers/tests/pipelines](https://github.com/huggingface/transformers/tree/main/tests/pipelines). You can look at the other tests for examples of how to test your pipeline. For example, take a look at the text classification pipeline test.

The [run_pipeline_test](https://github.com/huggingface/transformers/blob/db70426854fe7850f2c5834d633aff637f14772e/tests/pipelines/test_pipelines_text_classification.py#L186) function should be very generic and run on the models defined in [model_mapping](https://github.com/huggingface/transformers/blob/db70426854fe7850f2c5834d633aff637f14772e/tests/pipelines/test_pipelines_text_classification.py#L48) and [tf_model_mapping](https://github.com/huggingface/transformers/blob/db70426854fe7850f2c5834d633aff637f14772e/tests/pipelines/test_pipelines_text_classification.py#L49). This is important for testing future compatibility with new models.

You'll also notice `ANY` is used throughout the `run_pipeline_test` function. The models are random, so you can't check the actual values. Using `ANY` allows the test to just match the output of the pipeline type instead.

Finally, you should also implement the following 4 tests.

1. [test_small_model_pt](https://github.com/huggingface/transformers/blob/db70426854fe7850f2c5834d633aff637f14772e/tests/pipelines/test_pipelines_text_classification.py#L59) and [test_small_model_tf](https://github.com/huggingface/transformers/blob/db70426854fe7850f2c5834d633aff637f14772e/tests/pipelines/test_pipelines_text_classification.py#L150), use a small model for these pipelines to make sure they return the correct outputs. The results don't have to make sense. Each pipeline should return the same result.
1. [test_large_model_pt](https://github.com/huggingface/transformers/blob/db70426854fe7850f2c5834d633aff637f14772e/tests/pipelines/test_pipelines_zero_shot_image_classification.py#L187) nad [test_large_model_tf](https://github.com/huggingface/transformers/blob/db70426854fe7850f2c5834d633aff637f14772e/tests/pipelines/test_pipelines_zero_shot_image_classification.py#L220), use a realistic model for these pipelines to make sure they return meaningful results. These tests are slow and should be marked as slow.
