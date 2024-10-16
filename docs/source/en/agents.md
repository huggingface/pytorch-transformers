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

# Agents

[[open-in-colab]]

An agent is a system where a large language model (LLM) can execute more complex tasks through *planning* and using *tools*.

- Planning helps a LLM reason its way through a task by breaking it down into smaller subtasks. For example, [`CodeAgent`] plans a series of actions to take and then generates Python code to execute all the actions at once.

    Another planning method is by self-reflection and refinement of its previous actions to improve its performance. The [`ReactJsonAgent`] is an example of this type of planning, and it's based on the [ReAct](https://hf.co/papers/2210.03629) framework. This agent plans and executes actions one at a time based on the feedback it receives from each action.

- Tools give a LLM access to external functions or APIs that it can use to help it complete a task. For example, [gradio-tools](https://github.com/freddyaboulton/gradio-tools) gives a LLM access to any of the [Gradio](https://www.gradio.app/) apps available on Hugging Face [Spaces](https://hf.co/spaces). These apps can be used for a wide range of tasks such as image generation, video generation, audio transcription, and more.

To use agents in Transformers, make sure you have the extra `agents` dependencies installed.

```bash
!pip install transformers[agents]
```

Create an agent instance (refer to the [Agents](./main_classes/agent#agents) API for supported agents in Transformers) and a list of tools available for it to use, then [`~ReactAgent.run`] the agent on your task. The example below demonstrates how a ReAct agent reasons through a task.

```py
from transformers import ReactCodeAgent

agent = ReactCodeAgent(tools=[])
agent.run(
    "How many more blocks (also denoted as layers) in BERT base encoder than the encoder from the architecture proposed in Attention is All You Need?",
)
```

```bash
======== New task ========
How many more blocks (also denoted as layers) in BERT base encoder than the encoder from the architecture proposed in Attention is All You Need?
==== Agent is executing the code below:
bert_layers = 12  # BERT base encoder has 12 layers
attention_layers = 6  # Encoder in Attention is All You Need has 6 layers
layer_diff = bert_layers - attention_layers
print("The difference in layers between BERT base encoder and Attention is All You Need is", layer_diff)
====
Print outputs:
The difference in layers between BERT base encoder and Attention is All You Need is 6

==== Agent is executing the code below:
final_answer("BERT base encoder has {} more layers than the encoder from Attention is All You Need.".format(layer_diff))
====
Print outputs:

>>> Final answer:
BERT base encoder has 6 more layers than the encoder from Attention is All You Need.
```

This guide will walk you through in more detail how to initialize an agent.

## LLM

An agent uses a LLM to plan and execute a task; it is the engine that powers the agent. To choose and build your own LLM engine, you need a method that:

1. the input uses the [chat template](./chat_templating) format, `List[Dict[str, str]]`, and it returns a string
2. the LLM stops generating outputs when it encounters the sequences in `stop_sequences`

```py
def llm_engine(messages, stop_sequences=["Task"]) -> str:
    response = client.chat_completion(messages, stop=stop_sequences, max_tokens=1000)
    answer = response.choices[0].message.content
    return answer
```

Next, initialize an engine to load a model. To run an agent locally, create a [`TransformersEngine`] to load a preinitialized [`Pipeline`].

However, you could also leverage Hugging Face's powerful inference infrastructure, [Inference API](https://hf.co/docs/api-inference/index) or [Inference Endpoints](https://hf.co/docs/inference-endpoints/index), to run your model. This is useful for loading larger models that are typically required for agentic behavior. In this case, load the [`HfApiEngine`] to run the agent.

The agent requires a list of tools it can use to complete a task. If you aren't using any additional tools, pass an empty list. The default tools provided by Transformers are loaded automatically, but you can optionally set `add_base_tools=True` to explicitly enable them.

<hfoptions id="engine">
<hfoption id="TransformersEngine">

```py
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TransformersEngine, CodeAgent

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct").to("cuda")
pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
llm_engine = TransformersEngine(pipeline)
agent = CodeAgent(tools=[], llm_engine=llm_engine)
agent.run(
    "What causes bread to rise?",
)
```

</hfoption>
<hfoption id="HfApiEngine">

```py
from transformers import CodeAgent, HfApiEngine

llm_engine = HfApiEngine(model="meta-llama/Meta-Llama-3-70B-Instruct")
agent = CodeAgent(tools=[], llm_engine=llm_engine)
agent.run(
    "Could you translate this sentence from French, say it out loud and return the audio.",
    sentence="Où est la boulangerie la plus proche?",
)
```

</hfoption>
</hfoptions>

The agent supports [constrained generation](https://hf.co/docs/text-generation-inference/conceptual/guidance) for generating outputs according to a specific structure with the `grammar` parameter. The `grammar` parameter should be specified in the `llm_engine` method or you can set it when initializing an agent.

Lastly, an agent accepts additional inputs such as text and audio. In the [`HfApiEngine`] example above, the agent accepted a sentence to translate. But you could also pass a path to a local or remote file for the agent to access. The example below demonstrates how to pass a path to an audio file.

```py
from transformers import ReactCodeAgent

agent = ReactCodeAgent(tools=[], llm_engine=llm_engine)
agent.run("Why doesn't he know many people in New York?", audio="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/recording.mp3")
```

## System prompt

A system prompt describes how an agent should behave, a description of the available tools, and the expected output format.

Tools are defined by the `<<tool_descriptions>>` token which is dynamically replaced during runtime with the actual tool. The tool description is derived from the tool name, description, inputs, output type, and a Jinja2 template. Refer to the [Tools](./tools) guide for more information about how to describe tools.

The example below is the system prompt for [`ReactCodeAgent`].

```py
You will be given a task to solve as best you can.
You have access to the following tools:
<<tool_descriptions>>

To solve the task, you must plan forward to proceed in a series of steps, in a cycle of 'Thought:', 'Code:', and 'Observation:' sequences.

At each step, in the 'Thought:' sequence, you should first explain your reasoning towards solving the task, then the tools that you want to use.
Then in the 'Code:' sequence, you should write the code in simple Python. The code sequence must end with '/End code' sequence.
During each intermediate step, you can use 'print()' to save whatever important information you will then need.
These print outputs will then be available in the 'Observation:' field, for using this information as input for the next step.

In the end you have to return a final answer using the `final_answer` tool.

Here are a few examples using notional tools:
---
{examples}

Above example were using notional tools that might not exist for you. You only have acces to those tools:
<<tool_names>>
You also can perform computations in the python code you generate.

Always provide a 'Thought:' and a 'Code:\n```py' sequence ending with '```<end_code>' sequence. You MUST provide at least the 'Code:' sequence to move forward.

Remember to not perform too many operations in a single code block! You should split the task into intermediate code blocks.
Print results at the end of each step to save the intermediate results. Then use final_answer() to return the final result.

Remember to make sure that variables you use are all defined.

Now Begin!
```

The system prompt can be tailored to the intended task. For example, you can add a better explanation of the output format or you can overwrite the system prompt template entirely with your own custom system prompt as shown below.

> [!WARNING]
> If you're writing a custom system prompt, make sure to include `<<tool_descriptions>>` in the template so the agent is aware of the available tools.

```py
from transformers import ReactJsonAgent
from transformers.agents import PythonInterpreterTool

agent = ReactJsonAgent(tools=[PythonInterpreterTool()], system_prompt="{your_custom_prompt}")
```

## Code execution

For safety, only the tools you provide (and the default Transformers tools) and the `print` function are executed. The interpreter doesn't allow importing modules that aren't on a safe list.

To import modules that aren't on the list, add them as a list to the `additional_authorized_imports` parameter when initializing an agent.

```py
from transformers import ReactCodeAgent

agent = ReactCodeAgent(tools=[], additional_authorized_imports=['requests', 'bs4'])
agent.run("Could you get me the title of the page at url 'https://huggingface.co/blog'?")
```

Code execution stops if a tool isn't on the safe list, it isn't authorized, or if the code generated by the agent returns a Python error.

> [!WARNING]
> A LLM can generate any arbitrary code that can be executed, so don't add any unsafe imports!

## Multi-agent

[Multi-agent](https://hf.co/papers/2308.08155) refers to multiple agents working together to solve a task. Performance is typically better because each agent is specialized for a particular subtask.

Multi-agents are created through a [`ManagedAgent`] class, where a *manager agent* oversees how other agents work together. The manager agent requires an agent and their name and description. These are added to the manager agents system prompt which lets it know how to call and use them.

The multi-agent example below creates a web search agent that is managed by another [`ReactCodeAgent`].

```py
from transformers.agents import ReactCodeAgent, HfApiEngine, DuckDuckGoSearchTool, ManagedAgent

llm_engine = HfApiEngine()
web_agent = ReactCodeAgent(tools=[DuckDuckGoSearchTool()], llm_engine=llm_engine)
managed_web_agent = ManagedAgent(
    agent=web_agent,
    name="web_search",
    description="Runs web searches for you. Give it your query as an argument."
)
manager_agent = ReactCodeAgent(
    tools=[], llm_engine=llm_engine, managed_agents=[managed_web_agent]
)
manager_agent.run("Who is the CEO of Hugging Face?")
```

## Gradio integration

[Gradio](https://www.gradio.app/) is a library for quickly creating and sharing machine learning apps. The [gradio.Chatbot](https://www.gradio.app/docs/gradio/chatbot) supports chatting with a Transformers agent with the [`stream_to_gradio`] function.

Load a tool and LLM with an agent, and then create a Gradio app. The key is to use [`stream_to_gradio`] to stream the agents messages and display how it's reasoning through a task.

```py
import gradio as gr
from transformers import (
    load_tool,
    ReactCodeAgent,
    HfApiEngine,
    stream_to_gradio,
)

# Import tool from Hub
image_generation_tool = load_tool("m-ric/text-to-image")
llm_engine = HfApiEngine("meta-llama/Meta-Llama-3-70B-Instruct")

# Initialize the agent with the image generation tool
agent = ReactCodeAgent(tools=[image_generation_tool], llm_engine=llm_engine)

def interact_with_agent(task):
    messages = []
    messages.append(gr.ChatMessage(role="user", content=task))
    yield messages
    for msg in stream_to_gradio(agent, task):
        messages.append(msg)
        yield messages + [
            gr.ChatMessage(role="assistant", content="⏳ Task not finished yet!")
        ]
    yield messages

with gr.Blocks() as demo:
    text_input = gr.Textbox(lines=1, label="Chat Message", value="Make me a picture of the Statue of Liberty.")
    submit = gr.Button("Run illustrator agent!")
    chatbot = gr.Chatbot(
        label="Agent",
        type="messages",
        avatar_images=(
            None,
            "https://em-content.zobj.net/source/twitter/53/robot-face_1f916.png",
        ),
    )
    submit.click(interact_with_agent, [text_input], [chatbot])

if __name__ == "__main__":
    demo.launch()
```

## Troubleshoot

For a better idea of what is happening when you call an agent, it is always a good idea to check the system prompt template first.

```py
print(agent.system_prompt_template)
```

If the agent is behaving unexpectedly, remember to explain the task you want to perform as clearly as possible. Every [`~Agent.run`] is different and minor variations in your system prompt may yield completely different results.

To find out what happened after a run, check the following agent attributes.

- `agent.logs` stores the finegrained agent logs. At every step of the agents run, everything is stored in a dictionary and appended to `agent.logs`.
- `agent.write_inner_memory_from_logs` only stores a high-level overview of the agents run. For example, at each step, it stores the LLM output as a message and the tool call output as a separate message. Not every detail from a step is transcripted by `write_inner_memory_from_logs`.

## Resources

Learn more about ReAct agents in the [Open-source LLMs as LangChain Agents](https://hf.co/blog/open-source-llms-as-agents) blog post.
