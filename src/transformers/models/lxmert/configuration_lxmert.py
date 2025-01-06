# coding=utf-8
# Copyright 2018, Hao Tan, Mohit Bansal
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
"""LXMERT model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class LxmertConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`LxmertModel`] or a [`TFLxmertModel`]. It is used
    to instantiate a LXMERT model according to the specified arguments, defining the model architecture. Instantiating
    a configuration with the defaults will yield a similar configuration to that of the Lxmert
    [unc-nlp/lxmert-base-uncased](https://huggingface.co/unc-nlp/lxmert-base-uncased) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the LXMERT model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`LxmertModel`] or [`TFLxmertModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_qa_labels (`int`, *optional*, defaults to 9500):
            This represents the total number of different question answering (QA) labels there are. If using more than
            one dataset with QA, the user will need to account for the total number of labels that all of the datasets
            have in total.
        num_object_labels (`int`, *optional*, defaults to 1600):
            This represents the total number of semantically unique objects that lxmert will be able to classify a
            pooled-object feature as belonging too.
        num_attr_labels (`int`, *optional*, defaults to 400):
            This represents the total number of semantically unique attributes that lxmert will be able to classify a
            pooled-object feature as possessing.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the *token_type_ids* passed into [`BertModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        l_layers (`int`, *optional*, defaults to 9):
            Number of hidden layers in the Transformer language encoder.
        x_layers (`int`, *optional*, defaults to 5):
            Number of hidden layers in the Transformer cross modality encoder.
        r_layers (`int`, *optional*, defaults to 5):
            Number of hidden layers in the Transformer visual encoder.
        visual_feat_dim (`int`, *optional*, defaults to 2048):
            This represents the last dimension of the pooled-object features used as input for the model, representing
            the size of each object feature itself.
        visual_pos_dim (`int`, *optional*, defaults to 4):
            This represents the number of spacial features that are mixed into the visual features. The default is set
            to 4 because most commonly this will represent the location of a bounding box. i.e., (x, y, width, height)
        visual_loss_normalizer (`float`, *optional*, defaults to 6.67):
            This represents the scaling factor in which each visual loss is multiplied by if during pretraining, one
            decided to train with multiple vision-based loss objectives.
        task_matched (`bool`, *optional*, defaults to `True`):
            This task is used for sentence-image matching. If the sentence correctly describes the image the label will
            be 1. If the sentence does not correctly describe the image, the label will be 0.
        task_mask_lm (`bool`, *optional*, defaults to `True`):
            Whether or not to add masked language modeling (as used in pretraining models such as BERT) to the loss
            objective.
        task_obj_predict (`bool`, *optional*, defaults to `True`):
            Whether or not to add object prediction, attribute prediction and feature regression to the loss objective.
        task_qa (`bool`, *optional*, defaults to `True`):
            Whether or not to add the question-answering loss to the objective
        visual_obj_loss (`bool`, *optional*, defaults to `True`):
            Whether or not to calculate the object-prediction loss objective
        visual_attr_loss (`bool`, *optional*, defaults to `True`):
            Whether or not to calculate the attribute-prediction loss objective
        visual_feat_loss (`bool`, *optional*, defaults to `True`):
            Whether or not to calculate the feature-regression loss objective
    """

    model_type = "lxmert"
    attribute_map = {}

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_attention_heads=12,
        num_qa_labels=9500,
        num_object_labels=1600,
        num_attr_labels=400,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        l_layers=9,
        x_layers=5,
        r_layers=5,
        visual_feat_dim=2048,
        visual_pos_dim=4,
        visual_loss_normalizer=6.67,
        task_matched=True,
        task_mask_lm=True,
        task_obj_predict=True,
        task_qa=True,
        visual_obj_loss=True,
        visual_attr_loss=True,
        visual_feat_loss=True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.num_qa_labels = num_qa_labels
        self.num_object_labels = num_object_labels
        self.num_attr_labels = num_attr_labels
        self.l_layers = l_layers
        self.x_layers = x_layers
        self.r_layers = r_layers
        self.visual_feat_dim = visual_feat_dim
        self.visual_pos_dim = visual_pos_dim
        self.visual_loss_normalizer = visual_loss_normalizer
        self.task_matched = task_matched
        self.task_mask_lm = task_mask_lm
        self.task_obj_predict = task_obj_predict
        self.task_qa = task_qa
        self.visual_obj_loss = visual_obj_loss
        self.visual_attr_loss = visual_attr_loss
        self.visual_feat_loss = visual_feat_loss
        self.num_hidden_layers = {"vision": r_layers, "cross_encoder": x_layers, "language": l_layers}
        super().__init__(**kwargs)


__all__ = ["LxmertConfig"]
