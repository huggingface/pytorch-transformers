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

# Prompt engineering

[[open-in-colab]]

Prompt engineering or prompting, refers to using natural language to improve large language model (LLM) performance on a variety of tasks. LLMs have tremendous capacity as a result of their training and size, such that a prompt can steer the model towards generating a desired output. In many cases ([but not all](#finetuning)), you don't need a finetuned model for a task, you just need a good prompt.

Try prompting a LLM to classify some text. When you create a prompt, it's very important to provide specific instructions about the task you want to perform and what the result should look like.

```py
from transformers import pipeline
import torch

pipeline = pipeline(task="text-generation", model="mistralai/Mistal-7B-Instruct-v0.1", torch_dtype=torch.bfloat16, device_map="auto")
prompt = """Classify the text into neutral, negative or positive.
Text: This movie is definitely one of my favorite movies of its kind. The interaction between respectable and morally strong characters is an ode to chivalry and the honor code amongst thieves and policemen.
Sentiment:
"""

outputs = pipeline(prompt, max_new_tokens=10)
for output in outputs:
    print(f"Result: {output['generated_text']}")
Result: Classify the text into neutral, negative or positive. 
Text: This movie is definitely one of my favorite movies of its kind. The interaction between respectable and morally strong characters is an ode to chivalry and the honor code amongst thieves and policemen.
Sentiment:
Positive
```

The challenge lies in designing prompts that produces the results you're expecting, which can be tricky, because language is so incredibly nuanced and expressive.

This guide covers prompt engineering best practices, techniques, and examples for how to solve language and reasoning tasks.

## Best practices

1. Try to pick the latest models for the best performance. Keep in mind that LLMs can come in two flavors, [base](https://hf.co/mistralai/Mistral-7B-v0.1) and [instruction-tuned](https://hf.co/mistralai/Mistral-7B-Instruct-v0.1) (or chat).

    Base models are excellent at completing text given an initial prompt, but they're not as good at following instructions. Instruction-tuned models are specifically trained versions of the base models on instructional or conversational data. This makes instruction-tuned models a better fit for prompting.

    > [!WARNING]
    > Modern LLMs are typically decoder-only models, but there are some encoder-decoder LLMs like [Flan-T5](../model_doc/flan-t5) or [BART](../model_doc/bart) that may be used for prompting. For encoder-decoder models, make sure you set the pipeline task identifier to `text2text-generation` instead of `text-generation`.

2. Start with a short and simple prompt, and iterate on it to get better results.

3. Put instructions at the beginning or end of a prompt. For longer prompts, models may apply optimizations to prevent attention from scaling quadratically, which places more emphasis at the beginning and end of a prompt.

4. Clearly separate instructions from the text of interest.

5. Be specific and descriptive about the task and the desired output, including for example, its format, length, style, and language. Avoid ambiguous and vague descriptions and instructions.

6. Instructions should focus on "what to do" rather than "what not to do".

7. Help lead the model generate the correct output by writing the first word or even the first sentence.

8. Try other techniques like [few-shot](#few-shot) and [chain-of-thought](#chain-of-thought) to improve results.

9. Test your prompts with different models to assess their robustness.

10. Version and track your prompt performance.

## Techniques

Crafting a good prompt alone, also known as zero-shot prompting, may not be enough to get the results you want. You may need to try a few prompting techniques to get the best performance.

This section covers a few of these techniques.

### Few-shot

Few-shot prompting improves accuracy and performance by including specific examples of what a model should generate given an input. The explicit examples give the model a better understanding of the task and the output format you're looking for. Try experimenting with different numbers of examples (2, 4, 8, etc.) to the model to see how it affects performance.

The example below provides the model with 1 example (1-shot) of the output format, a date in MM/DD/YYYY format, it should return.

```py
from transformers import pipeline
import torch

pipeline = pipeline(model="mistralai/Mistral-7B-Instruct-v0.1", torch_dtype=torch.bfloat16, device_map="auto")
prompt = """Text: The first human went into space and orbited the Earth on April 12, 1961.
Date: 04/12/1961
Text: The first-ever televised presidential debate in the United States took place on September 28, 1960, between presidential candidates John F. Kennedy and Richard Nixon. 
Date:"""

outputs = pipeline(prompt, max_new_tokens=12, do_sample=True, top_k=10)
for output in outputs:
    print(f"Result: {output['generated_text']}")
Result: Text: The first human went into space and orbited the Earth on April 12, 1961.
Date: 04/12/1961
Text: The first-ever televised presidential debate in the United States took place on September 28, 1960, between presidential candidates John F. Kennedy and Richard Nixon. 
Date: 09/28/1960
```

The downside of few-shot prompting is that you need to create lengthier prompts which increases computation and latency. There is also a limit to prompt lengths. Finally, a model can learn unintended patterns from your examples and it doesn't work well on complex reasoning tasks.

### Chain-of-thought

Chain-of-thought (CoT) is effective at generating more coherent and well-reasoned outputs by providing a series of prompts that help a model "think" more thoroughly about a topic.

The example below provides the model with several prompts that forces it to work through several intermediate reasoning steps.

```py
from transformers import pipeline
import torch

pipeline = pipeline(model="mistralai/Mistral-7B-Instruct-v0.1", torch_dtype=torch.bfloat16, device_map="auto")
prompt = """Let's go through this step-by-step:
1. You start with 15 muffins.
2. You eat 2 muffins, leaving you with 13 muffins.
3. You give 5 muffins to your neighbor, leaving you with 8 muffins.
4. Your partner buys 6 more muffins, bringing the total number of muffins to 14.
5. Your partner eats 2 muffins, leaving you with 12 muffins.
If you eat 6 muffins, how many are left?"""

outputs = pipeline(prompt, max_new_tokens=20, do_sample=True, top_k=10)
for output in outputs:
    print(f"Result: {output['generated_text']}")
Result: Let's go through this step-by-step:
1. You start with 15 muffins.
2. You eat 2 muffins, leaving you with 13 muffins.
3. You give 5 muffins to your neighbor, leaving you with 8 muffins.
4. Your partner buys 6 more muffins, bringing the total number of muffins to 14.
5. Your partner eats 2 muffins, leaving you with 12 muffins.
If you eat 6 muffins, how many are left?
Answer: 6
```

Like [few-shot](#few-shot) prompting, the downside of CoT is that it requires more effort to design a series of prompts that help the model reason through a complex task and the prompt length increases latency.

## Finetuning

While prompting is a powerful way to work with LLMs, there are scenarios where a finetuned model or even finetuning a model works better.

Here are some examples scenarios where a finetuned model makes sense.

- Your domain is extremely different from what a LLM was pretrained on, and extensive prompting didn't produce the results you want.
- Your model needs to work well in a low-resource language.
- Your model needs to be trained on sensitive data that have strict regulatory requirements.
- You're using a small model due to cost, privacy, infrastructure, or other constraints.

In all of these scenarios, ensure that you have a large enough domain-specific dataset to train your model with, have enough time and resources, and the cost of finetuning is worth it. Otherwise, you may be better off trying to optimize your prompt!

## Examples

Here are some examples of prompting a LLM for different tasks.

<hfoptions id="tasks">
<hfoption id="named entity recognition">

```py
from transformers import pipeline
import torch

pipeline = pipeline(model="mistralai/Mistral-7B-Instruct-v0.1", torch_dtype=torch.bfloat16, device_map="auto")
prompt = """Return a list of named entities in the text.
Text: The company was founded in 2016 by French entrepreneurs Clément Delangue, Julien Chaumond, and Thomas Wolf in New York City, originally as a company that developed a chatbot app targeted at teenagers.
Named entities:
"""

outputs = pipeline(prompt, max_new_tokens=50, return_full_text=False)
for output in outputs:
    print(f"Result: {output['generated_text']}")
Result:  [Clément Delangue, Julien Chaumond, Thomas Wolf, company, New York City, chatbot app, teenagers]
```

</hfoption>
<hfoption id="translation">

```py
from transformers import pipeline
import torch

pipeline = pipeline(model="mistralai/Mistral-7B-Instruct-v0.1", torch_dtype=torch.bfloat16, device_map="auto")
prompt = """Translate the English text to French.
Text: Sometimes, I've believed as many as six impossible things before breakfast.
Translation:
"""

outputs = pipeline(prompt, max_new_tokens=20, do_sample=True, top_k=10, return_full_text=False)
for output in outputs:
    print(f"Result: {output['generated_text']}")
Result: À l'occasion, j'ai croyu plus de six choses impossibles
```

</hfoption>
<hfoption id="summarization">

```py
from transformers import pipeline
import torch

pipeline = pipeline(model="mistralai/Mistral-7B-Instruct-v0.1", torch_dtype=torch.bfloat16, device_map="auto")
prompt = """Permaculture is a design process mimicking the diversity, functionality and resilience of natural ecosystems. The principles and practices are drawn from traditional ecological knowledge of indigenous cultures combined with modern scientific understanding and technological innovations. Permaculture design provides a framework helping individuals and communities develop innovative, creative and effective strategies for meeting basic needs while preparing for and mitigating the projected impacts of climate change.
Write a summary of the above text.
Summary:
"""

outputs = pipeline(prompt, max_new_tokens=30, do_sample=True, top_k=10, return_full_text=False)
for output in outputs:
    print(f"Result: {output['generated_text']}")
Result: Permaculture is the design process that involves mimicking natural ecosystems to provide sustainable solutions to basic needs. It is a holistic approach that comb
```

</hfoption>
<hfoption id="question answering">

```py
from transformers import pipeline
import torch

pipeline = pipeline(model="mistralai/Mistral-7B-Instruct-v0.1", torch_dtype=torch.bfloat16, device_map="auto")
prompt = """Answer the question using the context below.
Context: Gazpacho is a cold soup and drink made of raw, blended vegetables. Most gazpacho includes stale bread, tomato, cucumbers, onion, bell peppers, garlic, olive oil, wine vinegar, water, and salt. Northern recipes often include cumin and/or pimentón (smoked sweet paprika). Traditionally, gazpacho was made by pounding the vegetables in a mortar with a pestle; this more laborious method is still sometimes used as it helps keep the gazpacho cool and avoids the foam and silky consistency of smoothie versions made in blenders or food processors.
Question: What modern tool is used to make gazpacho?
Answer:
"""

outputs = pipeline(prompt, max_new_tokens=10, do_sample=True, top_k=10, return_full_text=False)
for output in outputs:
    print(f"Result: {output['generated_text']}")
Result: A blender or food processor is the modern tool
```

</hfoption>
</hfoptions>
