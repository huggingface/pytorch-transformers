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

# XLA

[[open-in-colab]]

[Accelerated Linear Algebra (XLA)](https://openxla.org/xla) is a linear algebra compiler that optimizes model runtime across different hardware and frameworks.

This guide will look specifically at how to accelerate *TensorFlow* models with XLA.

## TensorFlow

XLA can potentially accelerate a TensorFlow model without making any source code changes. It is already packaged with the TensorFlow library, and it is triggered with `jit_compile` in any graph creating function such as [tf.function](https://www.tensorflow.org/api_docs/python/tf/function).

If you're using Keras methods like [fit](https://keras.io/api/models/model_training_apis/#fit-method) and [predict](https://keras.io/api/models/model_training_apis/#predict-method), enable XLA by passing `jit_compile=True` to [compile](https://keras.io/api/models/model_training_apis/#compile-method).

```py
model.compile(jit_compile=True)
```

XLA can be used to accelerate any arbitrary [tf.function](https://www.tensorflow.org/api_docs/python/tf/function).

Models with a TensorFlow implementation like [GPT2](./model_doc/gpt2), [T5](./model_doc/t5), [OPT](./model_doc/opt), and [Whisper](./model_doc/whisper) are XLA compatible. The amount of speed up depends on a model, but in general, TensorFlow models in Transformers get a ~100x speed up.

### Functions

A typical forward pass in a TensorFlow model is shown below. To run a forward pass with XLA, wrap the model with [tf.function](https://www.tensorflow.org/api_docs/python/tf/function) and set `jit_compile=True`.

```diff
import tensorflow as tf

model = tf.keras.Sequential(
    [tf.keras.layers.Dense(10, input_shape=(10,), activation="relu"), tf.keras.layers.Dense(5, activation="softmax")]
)
# Generate random inputs for the model.
batch_size = 16
input_vector_dim = 10
random_inputs = tf.random.normal((batch_size, input_vector_dim))

# Run a forward pass.
- _ = model(random_inputs)
+ xla_fn = tf.function(model, jit_compile=True)
+ _ = xla_fn(random_inputs)
```

The default `call` function of the model is used to compile the XLA graph. But if there's any other model function you want to compile with XLA, wrap them with [tf.function](https://www.tensorflow.org/api_docs/python/tf/function).

```py
my_xla_fn = tf.function(model.my_xla_fn, jit_compile=True)
```

### Text generation

You could also compile other model functions with XLA. For example, enable XLA for text generation by wrapping the [`~TFGenerationMixin.generate`] method with [tf.function](https://www.tensorflow.org/api_docs/python/tf/function).

```py
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForCausalLM
# Will error if the minimal version of Transformers is not installed.
from transformers.utils import check_min_version

check_min_version("4.21.0")

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2", padding_side="left", pad_token="</s>")
model = TFAutoModelForCausalLM.from_pretrained("openai-community/gpt2")
input_string = ["TensorFlow is"]

xla_generate = tf.function(model.generate, jit_compile=True)

tokenized_input = tokenizer(input_string, return_tensors="tf")
generated_tokens = xla_generate(**tokenized_input, num_beams=2)

decoded_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
print(f"Generated -- {decoded_text}")
"Generated -- TensorFlow is an open-source, open-source, distributed-source application framework for the"
```

## Tracing

When executing an XLA-enabled function for the first time, it tries to infer the computation graph in a process known as *tracing*. This is a time-consuming step, but any consecutive calls to the function will be much faster because it won't have to trace the computation graph again.

To ensure a function is only traced once, the inputs must have the same shape as when the graph was built. This usually isn't an issue for fixed input shapes like images, but it can be an issue for inputs with variable shapes like text.

One way to handle this is to pad your text so it always has the same shape. Configure the padding options such as [pad_to_multiple_of](https://hf.co/docs/transformers/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.pad.pad_to_multiple_of) in the tokenizer.

```py
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2", padding_side="left", pad_token="</s>")
model = TFAutoModelForCausalLM.from_pretrained("openai-community/gpt2")
input_string = ["TensorFlow is"]

xla_generate = tf.function(model.generate, jit_compile=True)

# Call tokenizer with padding options.
tokenized_input = tokenizer(input_string, pad_to_multiple_of=8, padding=True, return_tensors="tf")

generated_tokens = xla_generate(**tokenized_input, num_beams=2)
decoded_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
print(f"Generated -- {decoded_text}")
```

In addition to the input shape, any changes to the generation options at any point also triggers tracing.

## Resources

Learn more about XLA with the following resources.

- A [notebook](https://colab.research.google.com/github/huggingface/blog/blob/main/notebooks/91_tf_xla_generate.ipynb) demonstrating XLA-compatible encoder-decoder and decoder-only text generation models.
- The [Faster Text Generation with TensorFlow and XLA](https://hf.co/blog/tf-xla-generate) blog post compares benchmarks for XLA-compatible models and provides a friendly introduction to XLA in TensorFlow.
- The [How Hugging Face improved Text Generation performance with XLA](https://blog.tensorflow.org/2022/11/how-hugging-face-improved-text-generation-performance-with-xla.html) blog post discusses the design philosophy behind adding XLA to TensorFlow models in Transformers.
- The [Introduction to graphs and tf.function](https://www.tensorflow.org/guide/intro_to_graphs) guide.
- The [Better performance with tf.function](https://www.tensorflow.org/guide/function) guide.
- The [XLA](https://openxla.org/xla) documentation.
