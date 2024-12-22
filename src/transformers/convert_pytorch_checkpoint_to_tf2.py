# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
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
"""Convert pytorch checkpoints to TensorFlow"""

import argparse
import os

from . import (
    AlbertConfig,
    BartConfig,
    BertConfig,
    CamembertConfig,
    CTRLConfig,
    DistilBertConfig,
    DPRConfig,
    ElectraConfig,
    FlaubertConfig,
    GPT2Config,
    LayoutLMConfig,
    LxmertConfig,
    OpenAIGPTConfig,
    RobertaConfig,
    T5Config,
    TFAlbertForPreTraining,
    TFBartForConditionalGeneration,
    TFBartForSequenceClassification,
    TFBertForPreTraining,
    TFBertForQuestionAnswering,
    TFBertForSequenceClassification,
    TFCamembertForMaskedLM,
    TFCTRLLMHeadModel,
    TFDistilBertForMaskedLM,
    TFDistilBertForQuestionAnswering,
    TFDPRContextEncoder,
    TFDPRQuestionEncoder,
    TFDPRReader,
    TFElectraForPreTraining,
    TFFlaubertWithLMHeadModel,
    TFGPT2LMHeadModel,
    TFLayoutLMForMaskedLM,
    TFLxmertForPreTraining,
    TFLxmertVisualFeatureEncoder,
    TFOpenAIGPTLMHeadModel,
    TFRobertaForCausalLM,
    TFRobertaForMaskedLM,
    TFRobertaForSequenceClassification,
    TFT5ForConditionalGeneration,
    TFTransfoXLLMHeadModel,
    TFWav2Vec2Model,
    TFXLMRobertaForMaskedLM,
    TFXLMWithLMHeadModel,
    TFXLNetLMHeadModel,
    TransfoXLConfig,
    Wav2Vec2Config,
    Wav2Vec2Model,
    XLMConfig,
    XLMRobertaConfig,
    XLNetConfig,
    is_torch_available,
    load_pytorch_checkpoint_in_tf2_model,
)
from .utils import CONFIG_NAME, WEIGHTS_NAME, cached_file, logging


if is_torch_available():
    import numpy as np
    import torch

    from . import (
        AlbertForPreTraining,
        BartForConditionalGeneration,
        BertForPreTraining,
        BertForQuestionAnswering,
        BertForSequenceClassification,
        CamembertForMaskedLM,
        CTRLLMHeadModel,
        DistilBertForMaskedLM,
        DistilBertForQuestionAnswering,
        DPRContextEncoder,
        DPRQuestionEncoder,
        DPRReader,
        ElectraForPreTraining,
        FlaubertWithLMHeadModel,
        GPT2LMHeadModel,
        LayoutLMForMaskedLM,
        LxmertForPreTraining,
        LxmertVisualFeatureEncoder,
        OpenAIGPTLMHeadModel,
        RobertaForMaskedLM,
        RobertaForSequenceClassification,
        T5ForConditionalGeneration,
        TransfoXLLMHeadModel,
        XLMRobertaForMaskedLM,
        XLMWithLMHeadModel,
        XLNetLMHeadModel,
    )


logging.set_verbosity_info()

MODEL_CLASSES = {
    "bart": (
        BartConfig,
        TFBartForConditionalGeneration,
        TFBartForSequenceClassification,
        BartForConditionalGeneration,
    ),
    "bert": (
        BertConfig,
        TFBertForPreTraining,
        BertForPreTraining,
    ),
    "google-bert/bert-large-uncased-whole-word-masking-finetuned-squad": (
        BertConfig,
        TFBertForQuestionAnswering,
        BertForQuestionAnswering,
    ),
    "google-bert/bert-large-cased-whole-word-masking-finetuned-squad": (
        BertConfig,
        TFBertForQuestionAnswering,
        BertForQuestionAnswering,
    ),
    "google-bert/bert-base-cased-finetuned-mrpc": (
        BertConfig,
        TFBertForSequenceClassification,
        BertForSequenceClassification,
    ),
    "dpr": (
        DPRConfig,
        TFDPRQuestionEncoder,
        TFDPRContextEncoder,
        TFDPRReader,
        DPRQuestionEncoder,
        DPRContextEncoder,
        DPRReader,
    ),
    "openai-community/gpt2": (
        GPT2Config,
        TFGPT2LMHeadModel,
        GPT2LMHeadModel,
    ),
    "xlnet": (
        XLNetConfig,
        TFXLNetLMHeadModel,
        XLNetLMHeadModel,
    ),
    "xlm": (
        XLMConfig,
        TFXLMWithLMHeadModel,
        XLMWithLMHeadModel,
    ),
    "xlm-roberta": (
        XLMRobertaConfig,
        TFXLMRobertaForMaskedLM,
        XLMRobertaForMaskedLM,
    ),
    "transfo-xl": (
        TransfoXLConfig,
        TFTransfoXLLMHeadModel,
        TransfoXLLMHeadModel,
    ),
    "openai-community/openai-gpt": (
        OpenAIGPTConfig,
        TFOpenAIGPTLMHeadModel,
        OpenAIGPTLMHeadModel,
    ),
    "roberta": (
        RobertaConfig,
        TFRobertaForCausalLM,
        TFRobertaForMaskedLM,
        RobertaForMaskedLM,
    ),
    "layoutlm": (
        LayoutLMConfig,
        TFLayoutLMForMaskedLM,
        LayoutLMForMaskedLM,
    ),
    "FacebookAI/roberta-large-mnli": (
        RobertaConfig,
        TFRobertaForSequenceClassification,
        RobertaForSequenceClassification,
    ),
    "camembert": (
        CamembertConfig,
        TFCamembertForMaskedLM,
        CamembertForMaskedLM,
    ),
    "flaubert": (
        FlaubertConfig,
        TFFlaubertWithLMHeadModel,
        FlaubertWithLMHeadModel,
    ),
    "distilbert": (
        DistilBertConfig,
        TFDistilBertForMaskedLM,
        DistilBertForMaskedLM,
    ),
    "distilbert-base-distilled-squad": (
        DistilBertConfig,
        TFDistilBertForQuestionAnswering,
        DistilBertForQuestionAnswering,
    ),
    "lxmert": (
        LxmertConfig,
        TFLxmertForPreTraining,
        LxmertForPreTraining,
    ),
    "lxmert-visual-feature-encoder": (
        LxmertConfig,
        TFLxmertVisualFeatureEncoder,
        LxmertVisualFeatureEncoder,
    ),
    "Salesforce/ctrl": (
        CTRLConfig,
        TFCTRLLMHeadModel,
        CTRLLMHeadModel,
    ),
    "albert": (
        AlbertConfig,
        TFAlbertForPreTraining,
        AlbertForPreTraining,
    ),
    "t5": (
        T5Config,
        TFT5ForConditionalGeneration,
        T5ForConditionalGeneration,
    ),
    "electra": (
        ElectraConfig,
        TFElectraForPreTraining,
        ElectraForPreTraining,
    ),
    "wav2vec2": (
        Wav2Vec2Config,
        TFWav2Vec2Model,
        Wav2Vec2Model,
    ),
}


def convert_pt_checkpoint_to_tf(
    model_type, pytorch_checkpoint_path, config_file, tf_dump_path, compare_with_pt_model=False, use_cached_models=True
):
    if model_type not in MODEL_CLASSES:
        raise ValueError(f"Unrecognized model type, should be one of {list(MODEL_CLASSES.keys())}.")

    config_class, model_class, pt_model_class, aws_config_map = MODEL_CLASSES[model_type]

    # Initialise TF model
    if config_file in aws_config_map:
        config_file = cached_file(config_file, CONFIG_NAME, force_download=not use_cached_models)
    config = config_class.from_json_file(config_file)
    config.output_hidden_states = True
    config.output_attentions = True
    print(f"Building TensorFlow model from configuration: {config}")
    tf_model = model_class(config)

    # Load weights from tf checkpoint
    if pytorch_checkpoint_path in aws_config_map.keys():
        pytorch_checkpoint_path = cached_file(
            pytorch_checkpoint_path, WEIGHTS_NAME, force_download=not use_cached_models
        )
    # Load PyTorch checkpoint in tf2 model:
    tf_model = load_pytorch_checkpoint_in_tf2_model(tf_model, pytorch_checkpoint_path)

    if compare_with_pt_model:
        tfo = tf_model(tf_model.dummy_inputs, training=False)  # build the network

        weights_only_kwarg = {"weights_only": True}
        state_dict = torch.load(
            pytorch_checkpoint_path,
            map_location="cpu",
            **weights_only_kwarg,
        )
        pt_model = pt_model_class.from_pretrained(
            pretrained_model_name_or_path=None, config=config, state_dict=state_dict
        )

        with torch.no_grad():
            pto = pt_model(**pt_model.dummy_inputs)

        np_pt = pto[0].numpy()
        np_tf = tfo[0].numpy()
        diff = np.amax(np.abs(np_pt - np_tf))
        print(f"Max absolute difference between models outputs {diff}")
        assert diff <= 2e-2, f"Error, model absolute difference is >2e-2: {diff}"

    # Save pytorch-model
    print(f"Save TensorFlow model to {tf_dump_path}")
    tf_model.save_weights(tf_dump_path, save_format="h5")


def convert_all_pt_checkpoints_to_tf(
    args_model_type,
    tf_dump_path,
    model_shortcut_names_or_path=None,
    config_shortcut_names_or_path=None,
    compare_with_pt_model=False,
    use_cached_models=False,
    remove_cached_files=False,
    only_convert_finetuned_models=False,
):
    if args_model_type is None:
        model_types = list(MODEL_CLASSES.keys())
    else:
        model_types = [args_model_type]

    for j, model_type in enumerate(model_types, start=1):
        print("=" * 100)
        print(f" Converting model type {j}/{len(model_types)}: {model_type}")
        print("=" * 100)
        if model_type not in MODEL_CLASSES:
            raise ValueError(f"Unrecognized model type {model_type}, should be one of {list(MODEL_CLASSES.keys())}.")

        config_class, model_class, pt_model_class, aws_model_maps, aws_config_map = MODEL_CLASSES[model_type]

        if model_shortcut_names_or_path is None:
            model_shortcut_names_or_path = list(aws_model_maps.keys())
        if config_shortcut_names_or_path is None:
            config_shortcut_names_or_path = model_shortcut_names_or_path

        for i, (model_shortcut_name, config_shortcut_name) in enumerate(
            zip(model_shortcut_names_or_path, config_shortcut_names_or_path), start=1
        ):
            print("-" * 100)
            if "-squad" in model_shortcut_name or "-mrpc" in model_shortcut_name or "-mnli" in model_shortcut_name:
                if not only_convert_finetuned_models:
                    print(f"    Skipping finetuned checkpoint {model_shortcut_name}")
                    continue
                model_type = model_shortcut_name
            elif only_convert_finetuned_models:
                print(f"    Skipping not finetuned checkpoint {model_shortcut_name}")
                continue
            print(
                f"    Converting checkpoint {i}/{len(aws_config_map)}: {model_shortcut_name} - model_type {model_type}"
            )
            print("-" * 100)

            if config_shortcut_name in aws_config_map:
                config_file = cached_file(config_shortcut_name, CONFIG_NAME, force_download=not use_cached_models)
            else:
                config_file = config_shortcut_name

            if model_shortcut_name in aws_model_maps:
                model_file = cached_file(model_shortcut_name, WEIGHTS_NAME, force_download=not use_cached_models)
            else:
                model_file = model_shortcut_name

            if os.path.isfile(model_shortcut_name):
                model_shortcut_name = "converted_model"

            convert_pt_checkpoint_to_tf(
                model_type=model_type,
                pytorch_checkpoint_path=model_file,
                config_file=config_file,
                tf_dump_path=os.path.join(tf_dump_path, model_shortcut_name + "-tf_model.h5"),
                compare_with_pt_model=compare_with_pt_model,
            )
            if remove_cached_files:
                os.remove(config_file)
                os.remove(model_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--tf_dump_path", default=None, type=str, required=True, help="Path to the output Tensorflow dump file."
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        help=(
            f"Model type selected in the list of {list(MODEL_CLASSES.keys())}. If not given, will download and "
            "convert all the models from AWS."
        ),
    )
    parser.add_argument(
        "--pytorch_checkpoint_path",
        default=None,
        type=str,
        help=(
            "Path to the PyTorch checkpoint path or shortcut name to download from AWS. "
            "If not given, will download and convert all the checkpoints from AWS."
        ),
    )
    parser.add_argument(
        "--config_file",
        default=None,
        type=str,
        help=(
            "The config json file corresponding to the pre-trained model. \n"
            "This specifies the model architecture. If not given and "
            "--pytorch_checkpoint_path is not given or is a shortcut name "
            "use the configuration associated to the shortcut name on the AWS"
        ),
    )
    parser.add_argument(
        "--compare_with_pt_model", action="store_true", help="Compare Tensorflow and PyTorch model predictions."
    )
    parser.add_argument(
        "--use_cached_models",
        action="store_true",
        help="Use cached models if possible instead of updating to latest checkpoint versions.",
    )
    parser.add_argument(
        "--remove_cached_files",
        action="store_true",
        help="Remove pytorch models after conversion (save memory when converting in batches).",
    )
    parser.add_argument("--only_convert_finetuned_models", action="store_true", help="Only convert finetuned models.")
    args = parser.parse_args()

    # if args.pytorch_checkpoint_path is not None:
    #     convert_pt_checkpoint_to_tf(args.model_type.lower(),
    #                                 args.pytorch_checkpoint_path,
    #                                 args.config_file if args.config_file is not None else args.pytorch_checkpoint_path,
    #                                 args.tf_dump_path,
    #                                 compare_with_pt_model=args.compare_with_pt_model,
    #                                 use_cached_models=args.use_cached_models)
    # else:
    convert_all_pt_checkpoints_to_tf(
        args.model_type.lower() if args.model_type is not None else None,
        args.tf_dump_path,
        model_shortcut_names_or_path=[args.pytorch_checkpoint_path]
        if args.pytorch_checkpoint_path is not None
        else None,
        config_shortcut_names_or_path=[args.config_file] if args.config_file is not None else None,
        compare_with_pt_model=args.compare_with_pt_model,
        use_cached_models=args.use_cached_models,
        remove_cached_files=args.remove_cached_files,
        only_convert_finetuned_models=args.only_convert_finetuned_models,
    )
