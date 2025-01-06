# coding=utf-8
# Copyright 2019-present CNRS, Facebook Inc. and the HuggingFace Inc. team.
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
"""Flaubert configuration"""

from collections import OrderedDict
from typing import Mapping

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class FlaubertConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`FlaubertModel`] or a [`TFFlaubertModel`]. It is
    used to instantiate a FlauBERT model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the FlauBERT
    [flaubert/flaubert_base_uncased](https://huggingface.co/flaubert/flaubert_base_uncased) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        pre_norm (`bool`, *optional*, defaults to `False`):
            Whether to apply the layer normalization before or after the feed forward layer following the attention in
            each layer (Vaswani et al., Tensor2Tensor for Neural Machine Translation. 2018)
        layerdrop (`float`, *optional*, defaults to 0.0):
            Probability to drop layers during training (Fan et al., Reducing Transformer Depth on Demand with
            Structured Dropout. ICLR 2020)
        vocab_size (`int`, *optional*, defaults to 30145):
            Vocabulary size of the FlauBERT model. Defines the number of different tokens that can be represented by
            the `inputs_ids` passed when calling [`FlaubertModel`] or [`TFFlaubertModel`].
        emb_dim (`int`, *optional*, defaults to 2048):
            Dimensionality of the encoder layers and the pooler layer.
        n_layer (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        n_head (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for the attention mechanism
        gelu_activation (`bool`, *optional*, defaults to `True`):
            Whether or not to use a *gelu* activation instead of *relu*.
        sinusoidal_embeddings (`bool`, *optional*, defaults to `False`):
            Whether or not to use sinusoidal positional embeddings instead of absolute positional embeddings.
        causal (`bool`, *optional*, defaults to `False`):
            Whether or not the model should behave in a causal manner. Causal models use a triangular attention mask in
            order to only attend to the left-side context instead if a bidirectional context.
        asm (`bool`, *optional*, defaults to `False`):
            Whether or not to use an adaptive log softmax projection layer instead of a linear layer for the prediction
            layer.
        n_langs (`int`, *optional*, defaults to 1):
            The number of languages the model handles. Set to 1 for monolingual models.
        use_lang_emb (`bool`, *optional*, defaults to `True`)
            Whether to use language embeddings. Some models use additional language embeddings, see [the multilingual
            models page](http://huggingface.co/transformers/multilingual.html#xlm-language-embeddings) for information
            on how to use them.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        embed_init_std (`float`, *optional*, defaults to 2048^-0.5):
            The standard deviation of the truncated_normal_initializer for initializing the embedding matrices.
        init_std (`int`, *optional*, defaults to 50257):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices except the
            embedding matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        bos_index (`int`, *optional*, defaults to 0):
            The index of the beginning of sentence token in the vocabulary.
        eos_index (`int`, *optional*, defaults to 1):
            The index of the end of sentence token in the vocabulary.
        pad_index (`int`, *optional*, defaults to 2):
            The index of the padding token in the vocabulary.
        unk_index (`int`, *optional*, defaults to 3):
            The index of the unknown token in the vocabulary.
        mask_index (`int`, *optional*, defaults to 5):
            The index of the masking token in the vocabulary.
        is_encoder(`bool`, *optional*, defaults to `True`):
            Whether or not the initialized model should be a transformer encoder or decoder as seen in Vaswani et al.
        summary_type (`string`, *optional*, defaults to "first"):
            Argument used when doing sequence summary. Used in the sequence classification and multiple choice models.

            Has to be one of the following options:

                - `"last"`: Take the last token hidden state (like XLNet).
                - `"first"`: Take the first token hidden state (like BERT).
                - `"mean"`: Take the mean of all tokens hidden states.
                - `"cls_index"`: Supply a Tensor of classification token position (like GPT/GPT-2).
                - `"attn"`: Not implemented now, use multi-head attention.
        summary_use_proj (`bool`, *optional*, defaults to `True`):
            Argument used when doing sequence summary. Used in the sequence classification and multiple choice models.

            Whether or not to add a projection after the vector extraction.
        summary_activation (`str`, *optional*):
            Argument used when doing sequence summary. Used in the sequence classification and multiple choice models.

            Pass `"tanh"` for a tanh activation to the output, any other value will result in no activation.
        summary_proj_to_labels (`bool`, *optional*, defaults to `True`):
            Used in the sequence classification and multiple choice models.

            Whether the projection outputs should have `config.num_labels` or `config.hidden_size` classes.
        summary_first_dropout (`float`, *optional*, defaults to 0.1):
            Used in the sequence classification and multiple choice models.

            The dropout ratio to be used after the projection and activation.
        start_n_top (`int`, *optional*, defaults to 5):
            Used in the SQuAD evaluation script.
        end_n_top (`int`, *optional*, defaults to 5):
            Used in the SQuAD evaluation script.
        mask_token_id (`int`, *optional*, defaults to 0):
            Model agnostic parameter to identify masked tokens when generating text in an MLM context.
        lang_id (`int`, *optional*, defaults to 1):
            The ID of the language used by the model. This parameter is used when generating text in a given language.
    """

    model_type = "flaubert"
    attribute_map = {
        "hidden_size": "emb_dim",
        "num_attention_heads": "n_heads",
        "num_hidden_layers": "n_layers",
        "n_words": "vocab_size",  # For backward compatibility
    }

    def __init__(
        self,
        pre_norm=False,
        layerdrop=0.0,
        vocab_size=30145,
        emb_dim=2048,
        n_layers=12,
        n_heads=16,
        dropout=0.1,
        attention_dropout=0.1,
        gelu_activation=True,
        sinusoidal_embeddings=False,
        causal=False,
        asm=False,
        n_langs=1,
        use_lang_emb=True,
        max_position_embeddings=512,
        embed_init_std=2048**-0.5,
        layer_norm_eps=1e-12,
        init_std=0.02,
        bos_index=0,
        eos_index=1,
        pad_index=2,
        unk_index=3,
        mask_index=5,
        is_encoder=True,
        summary_type="first",
        summary_use_proj=True,
        summary_activation=None,
        summary_proj_to_labels=True,
        summary_first_dropout=0.1,
        start_n_top=5,
        end_n_top=5,
        mask_token_id=0,
        lang_id=0,
        pad_token_id=2,
        bos_token_id=0,
        **kwargs,
    ):
        """Constructs FlaubertConfig."""
        self.pre_norm = pre_norm
        self.layerdrop = layerdrop
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.gelu_activation = gelu_activation
        self.sinusoidal_embeddings = sinusoidal_embeddings
        self.causal = causal
        self.asm = asm
        self.n_langs = n_langs
        self.use_lang_emb = use_lang_emb
        self.layer_norm_eps = layer_norm_eps
        self.bos_index = bos_index
        self.eos_index = eos_index
        self.pad_index = pad_index
        self.unk_index = unk_index
        self.mask_index = mask_index
        self.is_encoder = is_encoder
        self.max_position_embeddings = max_position_embeddings
        self.embed_init_std = embed_init_std
        self.init_std = init_std
        self.summary_type = summary_type
        self.summary_use_proj = summary_use_proj
        self.summary_activation = summary_activation
        self.summary_proj_to_labels = summary_proj_to_labels
        self.summary_first_dropout = summary_first_dropout
        self.start_n_top = start_n_top
        self.end_n_top = end_n_top
        self.mask_token_id = mask_token_id
        self.lang_id = lang_id

        if "n_words" in kwargs:
            self.n_words = kwargs["n_words"]

        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, **kwargs)


class FlaubertOnnxConfig(OnnxConfig):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        if self.task == "multiple-choice":
            dynamic_axis = {0: "batch", 1: "choice", 2: "sequence"}
        else:
            dynamic_axis = {0: "batch", 1: "sequence"}
        return OrderedDict(
            [
                ("input_ids", dynamic_axis),
                ("attention_mask", dynamic_axis),
            ]
        )


__all__ = ["FlaubertConfig", "FlaubertOnnxConfig"]
