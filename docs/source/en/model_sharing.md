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

# Share

The Hugging Face [Hub](https://hf.co/models) is a platform for sharing, discovering, and consuming models of all different types and sizes. We highly recommend sharing your model on the Hub to push open-source machine learning forward for everyone!

This guide will show you how to share a model to the Hub from Transformers.

## Setup

To share a model to the Hub, you need a Hugging Face [account](https://hf.co/join). Create a [User Access Token](https://hf.co/docs/hub/security-tokens#user-access-tokens) and login to your account from either the CLI or a notebook.

<hfoptions id="share">
<hfoption id="CLI">

```bash
huggingface-cli login
```

</hfoption>
<hfoption id="notebook">

```py
from huggingface_hub import notebook_login

notebook_login()
```

</hfoption>
</hfoptions>

## Repository features

<Youtube id="XvSGPZFEjDY"/>

Each model repository supports versioning, commit history, and visualizing diffs.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/vis_diff.png"/>
</div>

The repository's built-in versioning is based on [Git](https://git-scm.com/) and [Git Large File Storage (LFS)](https://git-lfs.github.com/). Version control enables revisions, a way to specify a model version with a commit hash, tag or branch.

For example, specify the `revision` parameter in [`~PreTrainedModel.from_pretrained`] to load a specific model version.

```py
model = AutoModel.from_pretrained(
    "julien-c/EsperBERTo-small", revision="v2.0.1"
)
```

Model repositories also support [gating](https://hf.co/docs/hub/models-gated) for more control over how and who can access a model. Gating is common for allowing a select group of users to preview a research model before it's made public.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/gated-model.png"/>
</div>

The model repository also includes an inference [widget](https://hf.co/docs/hub/models-widgets) for users to directly interact with a model.

Check out the Hub [Models](https://hf.co/docs/hub/models) documentation to learn more about.

## Model framework conversion

Reach a wider audience by converting a model to be compatible with all machine learning frameworks (PyTorch, TensorFlow, Flax). While users can still load a model if they're using a different framework, it is slower because Transformers converts the checkpoint on the fly. It is faster to convert the checkpoint beforehand.

<hfoptions id="convert">
<hfoption id="PyTorch">

Set `from_tf=True` to convert a checkpoint from TensorFlow to PyTorch and then save it.

```py
import DistilBertForSequenceClassification

pt_model = DistilBertForSequenceClassification.from_pretrained("path/to/awesome-name-you-picked", from_tf=True)
pt_model.save_pretrained("path/to/awesome-name-you-picked")
```

</hfoption>
<hfoption id="TensorFlow">

Set `from_pt=True` to convert a checkpoint from PyTorch to TensorFlow and then save it.

```py
import TFDistilBertForSequenceClassification

tf_model = TFDistilBertForSequenceClassification.from_pretrained("path/to/awesome-name-you-picked", from_pt=True)
tf_model.save_pretrained("path/to/awesome-name-you-picked")
```

</hfoption>
<hfoption id="Flax">

Set `from_pt=True` to convert a checkpoint from PyTorch to Flax and then save it.

```py
flax_model = FlaxDistilBertForSequenceClassification.from_pretrained(
    "path/to/awesome-name-you-picked", from_pt=True
)
flax_model.save_pretrained("path/to/awesome-name-you-picked")
```

</hfoption>
</hfoptions>

## Upload model

There are several ways to upload a model to the Hub depending on your workflow preference. You can push a model with the [`Trainer`], call the [`~PreTrainedModel.push_to_hub`] method directly on a model, or use the Hub's web interface.

<Youtube id="Z1-XMy-GNLQ"/>

### Trainer

The [`Trainer`], Transformers' training API, allows pushing a model directly to the Hub after training. Set `push_to_hub=True` in the [`TrainingArguments`] class and pass it to the [`Trainer`]. Once training is complete, call [`~transformers.Trainer.push_to_hub`] to upload the model.

The [`~transformers.Trainer.push_to_hub`] method automatically adds useful information like the training hyperparameters and results to the model card.

```py
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(output_dir="my-awesome-model", push_to_hub=True)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)
trainer.push_to_hub()
```

#### TensorFlow models

For TensorFlow models, add the [`PushToHubCallback`] to [fit](https://keras.io/api/models/model_training_apis/#fit-method).

```py
from transformers import PushToHubCallback

push_to_hub_callback = PushToHubCallback(
    output_dir="./your_model_save_path", tokenizer=tokenizer, hub_model_id="your-username/my-awesome-model"
)
model.fit(tf_train_dataset, validation_data=tf_validation_dataset, epochs=3, callbacks=push_to_hub_callback)
```

### PreTrainedModel.push_to_hub

Call [`~PreTrainedModel.push_to_hub`] directly on a model to upload it to the Hub. It creates a repository under your namespace with the model name specified in [`~PreTrainedModel.push_to_hub`].

```py
model.push_to_hub("my-awesome-model")
```

Other objects like a tokenizer or TensorFlow model are also pushed to the Hub in the same way.

```py
tokenizer.push_to_hub("my-awesome-model")
```

Your Hugging Face profile should now display the newly created model repository. Navigate to the **Files** tab to see all the uploaded files

Refer to the [Upload files to the Hub](https://hf.co/docs/hub/how-to-upstream) guide for more details about pushing files to the Hub.

### Hub web interface

For a no-code approach, upload a model with the Hub's web interface.

Create a new repository by selecting [**New Model**](https://huggingface.co/new).

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/new_model_repo.png"/>
</div>

Add some details about your model:

- Select the **owner** of the repository. This can be yourself or any of the organizations you belong to.
- Pick a name for your model, which will also be the repository name.
- Choose whether your model is public or private.
- Specify the license usage for your model.

Click on **Create model** to create the model repository.

Now select the **Files** tab and click on the **Add file** button to drag-and-drop a file to your repository. Add a commit message and click on **Commit changes to `main`** to commit the file.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/upload_file.png"/>
</div>

## Model card

[Model cards](https://hf.co/docs/hub/model-cards#model-cards) inform users about a model's performance, limitations, potential biases, and ethical considerations. It is highly recommended to add a model card to your repository!

A model card is a `README.md` file in your repository. Add this file by:

- manually creating and uploading a `README.md` file
- clicking on the **Edit model card** button in the repository

Take a look at the Llama 3.1 [model card](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) for an example of the type of information to include on a model card.

Learn more about other model card metadata (carbon emissions, license, link to paper, etc.) to include in the [Model Cards](https://hf.co/docs/hub/model-cards#model-cards) guide.
