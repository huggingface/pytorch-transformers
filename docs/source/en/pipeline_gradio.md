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

# Machine learning apps

[Gradio](https://www.gradio.app/), a fast and easy library for building and sharing machine learning apps, is integrated with [`Pipeline`] to quickly create a simple interface for inference.

Before you begin, make sure Gradio is installed.

```py
!pip install gradio
```

Create a pipeline for your task, and then pass it to Gradio's [Interface.from_pipeline](https://www.gradio.app/docs/gradio/interface#interface-from_pipeline) function to create the interface. Gradio automatically determines the appropriate input and output components for a [`Pipeline`].

Add [launch](https://www.gradio.app/main/docs/gradio/blocks#blocks-launch) to create a web server and start up the app.

```py
from transformers import pipeline
import gradio as gr

pipeline = pipeline("image-classification", model="google/vit-base-patch16-224")
gr.Interface.from_pipeline(pipeline).launch()
```

The web app runs on a local server by default. To share the app with other users, set `share=True` in [launch](https://www.gradio.app/main/docs/gradio/blocks#blocks-launch) to generate a temporary public link. For a more permanent solution, host the app on Hugging Face [Spaces](https://hf.co/spaces).

```py
gr.Interface.from_pipeline(pipeline).launch(share=True)
```

The Space below is created with the code above and hosted on Spaces.

<iframe
	src="https://stevhliu-gradio-pipeline-demo.hf.space"
	frameborder="0"
	width="850"
	height="450"
></iframe>
