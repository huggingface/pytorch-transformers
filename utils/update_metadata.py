# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
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
Utility that updates the metadata of the Transformers library in the repository `huggingface/transformers-metadata`.

Usage for an update (as used by the GitHub action `update_metadata`):

```bash
python utils/update_metadata.py --token <token> --commit_sha <commit_sha>
```

Usage to check all pipelines are properly defined in the constant `PIPELINE_TAGS_AND_AUTO_MODELS` of this script, so
that new pipelines are properly added as metadata (as used in `make repo-consistency`):

```bash
python utils/update_metadata.py --check-only
```
"""

import argparse
import collections
import os
import re
import tempfile
from typing import Dict, List, Tuple

import pandas as pd
from datasets import Dataset
from huggingface_hub import hf_hub_download, upload_folder

from transformers.utils import direct_transformers_import


# All paths are set with the intent you should run this script from the root of the repo with the command
# python utils/update_metadata.py
TRANSFORMERS_PATH = "src/transformers"


# This is to make sure the transformers module imported is the one in the repo.
transformers_module = direct_transformers_import(TRANSFORMERS_PATH)


# Regexes that match TF/Flax/PT model names.
_re_tf_models = re.compile(r"TF(.*)(?:Model|Encoder|Decoder|ForConditionalGeneration)")
_re_flax_models = re.compile(r"Flax(.*)(?:Model|Encoder|Decoder|ForConditionalGeneration)")
# Will match any TF or Flax model too so need to be in an else branch afterthe two previous regexes.
_re_pt_models = re.compile(r"(.*)(?:Model|Encoder|Decoder|ForConditionalGeneration|ForRetrieval)")


# Fill this with tuples (pipeline_tag, model_mapping, auto_model)
PIPELINE_TAGS_AND_AUTO_MODELS = [
    ("pretraining", "MODEL_FOR_PRETRAINING_MAPPING_NAMES", "AutoModelForPreTraining"),
    ("feature-extraction", "MODEL_MAPPING_NAMES", "AutoModel"),
    ("image-feature-extraction", "MODEL_FOR_IMAGE_MAPPING_NAMES", "AutoModel"),
    ("audio-classification", "MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES", "AutoModelForAudioClassification"),
    ("text-generation", "MODEL_FOR_CAUSAL_LM_MAPPING_NAMES", "AutoModelForCausalLM"),
    ("automatic-speech-recognition", "MODEL_FOR_CTC_MAPPING_NAMES", "AutoModelForCTC"),
    ("image-classification", "MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES", "AutoModelForImageClassification"),
    ("image-segmentation", "MODEL_FOR_IMAGE_SEGMENTATION_MAPPING_NAMES", "AutoModelForImageSegmentation"),
    ("image-text-to-text", "MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES", "AutoModelForImageTextToText"),
    ("image-to-image", "MODEL_FOR_IMAGE_TO_IMAGE_MAPPING_NAMES", "AutoModelForImageToImage"),
    ("fill-mask", "MODEL_FOR_MASKED_LM_MAPPING_NAMES", "AutoModelForMaskedLM"),
    ("object-detection", "MODEL_FOR_OBJECT_DETECTION_MAPPING_NAMES", "AutoModelForObjectDetection"),
    (
        "zero-shot-object-detection",
        "MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING_NAMES",
        "AutoModelForZeroShotObjectDetection",
    ),
    ("question-answering", "MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES", "AutoModelForQuestionAnswering"),
    ("text2text-generation", "MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES", "AutoModelForSeq2SeqLM"),
    ("text-classification", "MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES", "AutoModelForSequenceClassification"),
    ("automatic-speech-recognition", "MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES", "AutoModelForSpeechSeq2Seq"),
    (
        "table-question-answering",
        "MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING_NAMES",
        "AutoModelForTableQuestionAnswering",
    ),
    ("token-classification", "MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES", "AutoModelForTokenClassification"),
    ("multiple-choice", "MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES", "AutoModelForMultipleChoice"),
    (
        "next-sentence-prediction",
        "MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING_NAMES",
        "AutoModelForNextSentencePrediction",
    ),
    (
        "audio-frame-classification",
        "MODEL_FOR_AUDIO_FRAME_CLASSIFICATION_MAPPING_NAMES",
        "AutoModelForAudioFrameClassification",
    ),
    ("audio-xvector", "MODEL_FOR_AUDIO_XVECTOR_MAPPING_NAMES", "AutoModelForAudioXVector"),
    (
        "document-question-answering",
        "MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES",
        "AutoModelForDocumentQuestionAnswering",
    ),
    (
        "visual-question-answering",
        "MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING_NAMES",
        "AutoModelForVisualQuestionAnswering",
    ),
    ("image-to-text", "MODEL_FOR_FOR_VISION_2_SEQ_MAPPING_NAMES", "AutoModelForVision2Seq"),
    (
        "zero-shot-image-classification",
        "MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING_NAMES",
        "AutoModelForZeroShotImageClassification",
    ),
    ("depth-estimation", "MODEL_FOR_DEPTH_ESTIMATION_MAPPING_NAMES", "AutoModelForDepthEstimation"),
    ("video-classification", "MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING_NAMES", "AutoModelForVideoClassification"),
    ("mask-generation", "MODEL_FOR_MASK_GENERATION_MAPPING_NAMES", "AutoModelForMaskGeneration"),
    ("text-to-audio", "MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING_NAMES", "AutoModelForTextToSpectrogram"),
    ("text-to-audio", "MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING_NAMES", "AutoModelForTextToWaveform"),
]


def camel_case_split(identifier: str) -> List[str]:
    """
    Split a camel-cased name into words.

    Args:
        identifier (`str`): The camel-cased name to parse.

    Returns:
        `List[str]`: The list of words in the identifier (as seprated by capital letters).

    Example:

    ```py
    >>> camel_case_split("CamelCasedClass")
    ["Camel", "Cased", "Class"]
    ```
    """
    # Regex thanks to https://stackoverflow.com/questions/29916065/how-to-do-camelcase-split-in-python
    matches = re.finditer(".+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)", identifier)
    return [m.group(0) for m in matches]


def get_frameworks_table() -> pd.DataFrame:
    """
    Generates a dataframe containing the supported auto classes for each model type, using the content of the auto
    modules.
    """
    # Dictionary model names to config.
    config_maping_names = transformers_module.models.auto.configuration_auto.CONFIG_MAPPING_NAMES
    model_prefix_to_model_type = {
        config.replace("Config", ""): model_type for model_type, config in config_maping_names.items()
    }

    # Dictionaries flagging if each model prefix has a backend in PT/TF/Flax.
    pt_models = collections.defaultdict(bool)
    tf_models = collections.defaultdict(bool)
    flax_models = collections.defaultdict(bool)

    # Let's lookup through all transformers object (once) and find if models are supported by a given backend.
    for attr_name in dir(transformers_module):
        lookup_dict = None
        if _re_tf_models.match(attr_name) is not None:
            lookup_dict = tf_models
            attr_name = _re_tf_models.match(attr_name).groups()[0]
        elif _re_flax_models.match(attr_name) is not None:
            lookup_dict = flax_models
            attr_name = _re_flax_models.match(attr_name).groups()[0]
        elif _re_pt_models.match(attr_name) is not None:
            lookup_dict = pt_models
            attr_name = _re_pt_models.match(attr_name).groups()[0]

        if lookup_dict is not None:
            while len(attr_name) > 0:
                if attr_name in model_prefix_to_model_type:
                    lookup_dict[model_prefix_to_model_type[attr_name]] = True
                    break
                # Try again after removing the last word in the name
                attr_name = "".join(camel_case_split(attr_name)[:-1])

    all_models = set(list(pt_models.keys()) + list(tf_models.keys()) + list(flax_models.keys()))
    all_models = list(all_models)
    all_models.sort()

    data = {"model_type": all_models}
    data["pytorch"] = [pt_models[t] for t in all_models]
    data["tensorflow"] = [tf_models[t] for t in all_models]
    data["flax"] = [flax_models[t] for t in all_models]

    # Now let's find the right processing class for each model. In order we check if there is a Processor, then a
    # Tokenizer, then a FeatureExtractor, then an ImageProcessor
    processors = {}
    for t in all_models:
        if t in transformers_module.models.auto.processing_auto.PROCESSOR_MAPPING_NAMES:
            processors[t] = "AutoProcessor"
        elif t in transformers_module.models.auto.tokenization_auto.TOKENIZER_MAPPING_NAMES:
            processors[t] = "AutoTokenizer"
        elif t in transformers_module.models.auto.image_processing_auto.IMAGE_PROCESSOR_MAPPING_NAMES:
            processors[t] = "AutoImageProcessor"
        elif t in transformers_module.models.auto.feature_extraction_auto.FEATURE_EXTRACTOR_MAPPING_NAMES:
            processors[t] = "AutoFeatureExtractor"
        else:
            # Default to AutoTokenizer if a model has nothing, for backward compatibility.
            processors[t] = "AutoTokenizer"

    data["processor"] = [processors[t] for t in all_models]

    return pd.DataFrame(data)


def update_pipeline_and_auto_class_table(table: Dict[str, Tuple[str, str]]) -> Dict[str, Tuple[str, str]]:
    """
    Update the table maping models to pipelines and auto classes without removing old keys if they don't exist anymore.

    Args:
        table (`Dict[str, Tuple[str, str]]`):
            The existing table mapping model names to a tuple containing the pipeline tag and the auto-class name with
            which they should be used.

    Returns:
        `Dict[str, Tuple[str, str]]`: The updated table in the same format.
    """
    auto_modules = [
        transformers_module.models.auto.modeling_auto,
        transformers_module.models.auto.modeling_tf_auto,
        transformers_module.models.auto.modeling_flax_auto,
    ]
    for pipeline_tag, model_mapping, auto_class in PIPELINE_TAGS_AND_AUTO_MODELS:
        model_mappings = [model_mapping, f"TF_{model_mapping}", f"FLAX_{model_mapping}"]
        auto_classes = [auto_class, f"TF_{auto_class}", f"Flax_{auto_class}"]
        # Loop through all three frameworks
        for module, cls, mapping in zip(auto_modules, auto_classes, model_mappings):
            # The type of pipeline may not exist in this framework
            if not hasattr(module, mapping):
                continue
            # First extract all model_names
            model_names = []
            for name in getattr(module, mapping).values():
                if isinstance(name, str):
                    model_names.append(name)
                else:
                    model_names.extend(list(name))

            # Add pipeline tag and auto model class for those models
            table.update({model_name: (pipeline_tag, cls) for model_name in model_names})

    return table


def update_metadata(token: str, commit_sha: str):
    """
    Update the metadata for the Transformers repo in `huggingface/transformers-metadata`.

    Args:
        token (`str`): A valid token giving write access to `huggingface/transformers-metadata`.
        commit_sha (`str`): The commit SHA on Transformers corresponding to this update.
    """
    frameworks_table = get_frameworks_table()
    frameworks_dataset = Dataset.from_pandas(frameworks_table)

    resolved_tags_file = hf_hub_download(
        "huggingface/transformers-metadata", "pipeline_tags.json", repo_type="dataset", token=token
    )
    tags_dataset = Dataset.from_json(resolved_tags_file)
    table = {
        tags_dataset[i]["model_class"]: (tags_dataset[i]["pipeline_tag"], tags_dataset[i]["auto_class"])
        for i in range(len(tags_dataset))
    }
    table = update_pipeline_and_auto_class_table(table)

    # Sort the model classes to avoid some nondeterministic updates to create false update commits.
    model_classes = sorted(table.keys())
    tags_table = pd.DataFrame(
        {
            "model_class": model_classes,
            "pipeline_tag": [table[m][0] for m in model_classes],
            "auto_class": [table[m][1] for m in model_classes],
        }
    )
    tags_dataset = Dataset.from_pandas(tags_table)

    hub_frameworks_json = hf_hub_download(
        repo_id="huggingface/transformers-metadata",
        filename="frameworks.json",
        repo_type="dataset",
        token=token,
    )
    with open(hub_frameworks_json) as f:
        hub_frameworks_json = f.read()

    hub_pipeline_tags_json = hf_hub_download(
        repo_id="huggingface/transformers-metadata",
        filename="pipeline_tags.json",
        repo_type="dataset",
        token=token,
    )
    with open(hub_pipeline_tags_json) as f:
        hub_pipeline_tags_json = f.read()

    with tempfile.TemporaryDirectory() as tmp_dir:
        frameworks_dataset.to_json(os.path.join(tmp_dir, "frameworks.json"))
        tags_dataset.to_json(os.path.join(tmp_dir, "pipeline_tags.json"))

        with open(os.path.join(tmp_dir, "frameworks.json")) as f:
            frameworks_json = f.read()
        with open(os.path.join(tmp_dir, "pipeline_tags.json")) as f:
            pipeline_tags_json = f.read()

        frameworks_equal = hub_frameworks_json == frameworks_json
        hub_pipeline_tags_equal = hub_pipeline_tags_json == pipeline_tags_json

        if frameworks_equal and hub_pipeline_tags_equal:
            print("No updates on the Hub, not pushing the metadata files.")
            return

        if commit_sha is not None:
            commit_message = (
                f"Update with commit {commit_sha}\n\nSee: "
                f"https://github.com/huggingface/transformers/commit/{commit_sha}"
            )
        else:
            commit_message = "Update"

        upload_folder(
            repo_id="huggingface/transformers-metadata",
            folder_path=tmp_dir,
            repo_type="dataset",
            token=token,
            commit_message=commit_message,
        )


def check_pipeline_tags():
    """
    Check all pipeline tags are properly defined in the `PIPELINE_TAGS_AND_AUTO_MODELS` constant of this script.
    """
    in_table = {tag: cls for tag, _, cls in PIPELINE_TAGS_AND_AUTO_MODELS}
    pipeline_tasks = transformers_module.pipelines.SUPPORTED_TASKS
    missing = []
    for key in pipeline_tasks:
        if key not in in_table:
            model = pipeline_tasks[key]["pt"]
            if isinstance(model, (list, tuple)):
                model = model[0]
            model = model.__name__
            if model not in in_table.values():
                missing.append(key)

    if len(missing) > 0:
        msg = ", ".join(missing)
        raise ValueError(
            "The following pipeline tags are not present in the `PIPELINE_TAGS_AND_AUTO_MODELS` constant inside "
            f"`utils/update_metadata.py`: {msg}. Please add them!"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", type=str, help="The token to use to push to the transformers-metadata dataset.")
    parser.add_argument("--commit_sha", type=str, help="The sha of the commit going with this update.")
    parser.add_argument("--check-only", action="store_true", help="Activate to just check all pipelines are present.")
    args = parser.parse_args()

    if args.check_only:
        check_pipeline_tags()
    else:
        update_metadata(args.token, args.commit_sha)
