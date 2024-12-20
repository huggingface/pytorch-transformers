# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Convert ColPali weights from the original repository to the HF model format.

Original repository: https://github.com/illuin-tech/colpali.

NOTE: This script was originally run using `torch==2.5.1` and with:

```bash
python src/transformers/models/colpali/convert_colpali_weights_to_hf.py \
    --model_id vidore/colpali-v1.2-merged \
    --revision 89fd9736194236a1ecb7a9ec9b04f537f6f896af \
    --original_vlm_name_or_path google/paligemma-3b-mix-448 \
    --output_dir vidore/colpali-v1.2-hf-internal \
    --push_to_hub

python src/transformers/models/colpali/convert_colpali_weights_to_hf.py \
    --model_id vidore/colpali-v1.3-merged \
    --revision 5b955e3415a7c5468ab33119d98d6d45c3a5b2c3 \
    --original_vlm_name_or_path google/paligemma-3b-mix-448 \
    --output_dir vidore/colpali-v1.3-hf \
    --push_to_hub
```
"""

import argparse
import glob
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

from transformers import AutoConfig
from transformers.models.colpali import ColPaliForRetrieval
from transformers.models.colpali.configuration_colpali import ColPaliConfig
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


ORIGINAL_DTYPE = torch.bfloat16


def rename_state_dict_keys(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        if key.startswith("custom_text_proj"):
            new_key = key.replace("custom_text_proj", "embedding_proj_layer")
        if key.startswith("model."):
            new_key = key.replace("model.", "vlm.", 1)
        new_state_dict[new_key] = value
    return new_state_dict


def load_original_state_dict(model_id: str, revision: Optional[str] = None) -> Dict[str, torch.Tensor]:
    directory_path = snapshot_download(
        repo_id=model_id,
        revision=revision,
        allow_patterns=["*.safetensors"],
    )

    original_state_dict = {}
    for path in glob.glob(f"{directory_path}/*"):
        if path.endswith(".safetensors"):
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    original_state_dict[key] = f.get_tensor(key)

    # Some weights are tied, so `lm.head`` is not saved. Let's clone to load state dict.
    if "lm_head.weight" not in original_state_dict:
        original_state_dict["vlm.language_model.lm_head.weight"] = original_state_dict[
            "model.language_model.model.embed_tokens.weight"
        ].clone()

    return original_state_dict


@torch.no_grad()
def convert_colpali_weights_to_hf(
    model_id: str,
    output_dir: str,
    push_to_hub: bool,
    revision: Optional[str] = None,
    original_vlm_name_or_path: Optional[str] = None,
):
    # Load the original model data
    original_config = AutoConfig.from_pretrained(
        model_id,
        revision=revision,
    )
    if original_vlm_name_or_path is not None:
        original_config._name_or_path = original_vlm_name_or_path
    if hasattr(original_config, "architectures"):
        delattr(original_config, "architectures")

    original_state_dict = load_original_state_dict(model_id, revision=revision)

    # Format the state_dict keys
    original_state_dict = rename_state_dict_keys(original_state_dict)

    # Create the new config
    config = ColPaliConfig(
        vlm_config=original_config,
        embedding_dim=128,  # hardcoded in the original model
    )
    config.model_type = "colpali"
    config.is_composition = False

    # Load the untrained model
    model = ColPaliForRetrieval(config=config).to("cpu").eval()
    print("Created model with new config and randomly initialized weights")

    # NOTE: The model was initialized with float32 weights. We need to convert it to the desired precision.
    # There are two ways to set the model's dtype:
    # - Using `model.from_pretrained(..., torch_dtype=dtype_precision)` doesn't convert the hyperparameters to the desired precision.
    # - Using `model.to(dtype_precision)` converts all values - including the hyperparameters - to the desired precision.
    # The following snippet allows a fine-grained control over the model's dtype, making sure that all
    # the new weights' dtypes match the original model.
    for param in model.parameters():
        param.data = param.data.to(ORIGINAL_DTYPE)
    print(f"Converted the new model weights to `{ORIGINAL_DTYPE}`")

    # Load the original weights
    model.load_state_dict(original_state_dict)
    print("Loaded original model weights")

    # Tie the weights (following ColPali's `__init__`` step)
    if model.vlm.language_model._tied_weights_keys is not None:
        model._tied_weights_keys = [f"vlm.language_model.{k}" for k in model.vlm.language_model._tied_weights_keys]

    # Sanity check: ensure all keys are the same
    state_dict_keys_old = set(original_state_dict.keys())
    state_dict_keys_new = set(model.state_dict().keys())
    disjoint_keys = state_dict_keys_old.symmetric_difference(state_dict_keys_new)
    if disjoint_keys:
        raise ValueError(f"Incompatible keys: {disjoint_keys}")

    # Save the model
    if push_to_hub:
        model.push_to_hub(output_dir, private=True)
        print(f"Model pushed to the hub at `{output_dir}`")
    else:
        Path(output_dir).mkdir(exist_ok=True, parents=True)
        model.save_pretrained(output_dir)
        print(f"Model saved to `{output_dir}`")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        This script converts the original ColPali model to the HF model format.

        Example usage:
        ```bash
        python src/transformers/models/colpali/convert_colpali_weights_to_hf.py \
            --model_id vidore/colpali-v1.2-merged \
            --revision 89fd9736194236a1ecb7a9ec9b04f537f6f896af \
            --original_vlm_name_or_path google/paligemma-3b-mix-448 \
            --output_dir vidore/colpali-v1.2-hf \
            --push_to_hub
        ```
        """
    )
    parser.add_argument(
        "--model_id",
        help="Model ID of the original model to convert",
    )
    parser.add_argument(
        "--output_dir",
        help="Location to write HF model and tokenizer",
    )
    parser.add_argument(
        "--push_to_hub",
        help="Whether or not to push the model to the hub at `output_dir` instead of saving it locally",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--revision",
        help="Revision of the model to download",
        default=None,
    )
    parser.add_argument(
        "--original_vlm_name_or_path",
        help="Name or path of the original VLM backbone model",
        default=None,
    )
    args = parser.parse_args()

    convert_colpali_weights_to_hf(
        model_id=args.model_id,
        output_dir=args.output_dir,
        push_to_hub=args.push_to_hub,
        revision=args.revision,
        original_vlm_name_or_path=args.original_vlm_name_or_path,
    )
