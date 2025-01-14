# Copyright 2020 The HuggingFace Team. All rights reserved.
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

import os
import sys


SRC_DIR = os.path.join(os.path.dirname(__file__), "src")
sys.path.append(SRC_DIR)


from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    add_start_docstrings,
)


dependencies = ["torch", "numpy", "tokenizers", "filelock", "requests", "tqdm", "regex", "sentencepiece", "sacremoses", "importlib_metadata", "huggingface_hub"]


@add_start_docstrings(AutoConfig.__doc__)
def config(*args, **kwargs):
    r"""
                # Using torch.hub !
                import torch

                config = torch.hub.load('huggingface/transformers', 'config', 'google-bert/bert-base-uncased')  # Download configuration from huggingface.co and cache.
                config = torch.hub.load('huggingface/transformers', 'config', './test/bert_saved_model/')  # E.g. config (or model) was saved using `save_pretrained('./test/saved_model/')`
                config = torch.hub.load('huggingface/transformers', 'config', './test/bert_saved_model/my_configuration.json')
                config = torch.hub.load('huggingface/transformers', 'config', 'google-bert/bert-base-uncased', output_attentions=True, foo=False)
                assert config.output_attentions == True
                config, unused_kwargs = torch.hub.load('huggingface/transformers', 'config', 'google-bert/bert-base-uncased', output_attentions=True, foo=False, return_unused_kwargs=True)
                assert config.output_attentions == True
                assert unused_kwargs == {'foo': False}

            """

    return AutoConfig.from_pretrained(*args, **kwargs)


@add_start_docstrings(AutoTokenizer.__doc__)
def tokenizer(*args, **kwargs):
    r"""
        # Using torch.hub !
        import torch

        tokenizer = torch.hub.load('huggingface/transformers', 'tokenizer', 'google-bert/bert-base-uncased')  # Download vocabulary from huggingface.co and cache.
        tokenizer = torch.hub.load('huggingface/transformers', 'tokenizer', './test/bert_saved_model/')  # E.g. tokenizer was saved using `save_pretrained('./test/saved_model/')`

    """

    return AutoTokenizer.from_pretrained(*args, **kwargs)


@add_start_docstrings(AutoModel.__doc__)
def model(*args, **kwargs):
    r"""
            # Using torch.hub !
            import torch

            model = torch.hub.load('huggingface/transformers', 'model', 'google-bert/bert-base-uncased')  # Download model and configuration from huggingface.co and cache.
            model = torch.hub.load('huggingface/transformers', 'model', './test/bert_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
            model = torch.hub.load('huggingface/transformers', 'model', 'google-bert/bert-base-uncased', output_attentions=True)  # Update configuration during loading
            assert model.config.output_attentions == True
            # Loading from a TF checkpoint file instead of a PyTorch model (slower)
            config = AutoConfig.from_pretrained('./tf_model/bert_tf_model_config.json')
            model = torch.hub.load('huggingface/transformers', 'model', './tf_model/bert_tf_checkpoint.ckpt.index', from_tf=True, config=config)

        """

    return AutoModel.from_pretrained(*args, **kwargs)


@add_start_docstrings(AutoModelForCausalLM.__doc__)
def modelForCausalLM(*args, **kwargs):
    r"""
        # Using torch.hub !
        import torch

        model = torch.hub.load('huggingface/transformers', 'modelForCausalLM', 'openai-community/gpt2')  # Download model and configuration from huggingface.co and cache.
        model = torch.hub.load('huggingface/transformers', 'modelForCausalLM', './test/saved_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
        model = torch.hub.load('huggingface/transformers', 'modelForCausalLM', 'openai-community/gpt2', output_attentions=True)  # Update configuration during loading
        assert model.config.output_attentions == True
        # Loading from a TF checkpoint file instead of a PyTorch model (slower)
        config = AutoConfig.from_pretrained('./tf_model/gpt_tf_model_config.json')
        model = torch.hub.load('huggingface/transformers', 'modelForCausalLM', './tf_model/gpt_tf_checkpoint.ckpt.index', from_tf=True, config=config)

    """
    return AutoModelForCausalLM.from_pretrained(*args, **kwargs)


@add_start_docstrings(AutoModelForMaskedLM.__doc__)
def modelForMaskedLM(*args, **kwargs):
    r"""
            # Using torch.hub !
            import torch

            model = torch.hub.load('huggingface/transformers', 'modelForMaskedLM', 'google-bert/bert-base-uncased')  # Download model and configuration from huggingface.co and cache.
            model = torch.hub.load('huggingface/transformers', 'modelForMaskedLM', './test/bert_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
            model = torch.hub.load('huggingface/transformers', 'modelForMaskedLM', 'google-bert/bert-base-uncased', output_attentions=True)  # Update configuration during loading
            assert model.config.output_attentions == True
            # Loading from a TF checkpoint file instead of a PyTorch model (slower)
            config = AutoConfig.from_pretrained('./tf_model/bert_tf_model_config.json')
            model = torch.hub.load('huggingface/transformers', 'modelForMaskedLM', './tf_model/bert_tf_checkpoint.ckpt.index', from_tf=True, config=config)

        """

    return AutoModelForMaskedLM.from_pretrained(*args, **kwargs)


@add_start_docstrings(AutoModelForSequenceClassification.__doc__)
def modelForSequenceClassification(*args, **kwargs):
    r"""
            # Using torch.hub !
            import torch

            model = torch.hub.load('huggingface/transformers', 'modelForSequenceClassification', 'google-bert/bert-base-uncased')  # Download model and configuration from huggingface.co and cache.
            model = torch.hub.load('huggingface/transformers', 'modelForSequenceClassification', './test/bert_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
            model = torch.hub.load('huggingface/transformers', 'modelForSequenceClassification', 'google-bert/bert-base-uncased', output_attentions=True)  # Update configuration during loading
            assert model.config.output_attentions == True
            # Loading from a TF checkpoint file instead of a PyTorch model (slower)
            config = AutoConfig.from_pretrained('./tf_model/bert_tf_model_config.json')
            model = torch.hub.load('huggingface/transformers', 'modelForSequenceClassification', './tf_model/bert_tf_checkpoint.ckpt.index', from_tf=True, config=config)

        """

    return AutoModelForSequenceClassification.from_pretrained(*args, **kwargs)


@add_start_docstrings(AutoModelForQuestionAnswering.__doc__)
def modelForQuestionAnswering(*args, **kwargs):
    r"""
        # Using torch.hub !
        import torch

        model = torch.hub.load('huggingface/transformers', 'modelForQuestionAnswering', 'google-bert/bert-base-uncased')  # Download model and configuration from huggingface.co and cache.
        model = torch.hub.load('huggingface/transformers', 'modelForQuestionAnswering', './test/bert_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
        model = torch.hub.load('huggingface/transformers', 'modelForQuestionAnswering', 'google-bert/bert-base-uncased', output_attentions=True)  # Update configuration during loading
        assert model.config.output_attentions == True
        # Loading from a TF checkpoint file instead of a PyTorch model (slower)
        config = AutoConfig.from_pretrained('./tf_model/bert_tf_model_config.json')
        model = torch.hub.load('huggingface/transformers', 'modelForQuestionAnswering', './tf_model/bert_tf_checkpoint.ckpt.index', from_tf=True, config=config)

    """
    return AutoModelForQuestionAnswering.from_pretrained(*args, **kwargs)
