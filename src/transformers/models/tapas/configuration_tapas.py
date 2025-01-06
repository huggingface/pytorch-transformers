# coding=utf-8
# Copyright 2020 Google Research and The HuggingFace Inc. team.
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
TAPAS configuration. Based on the BERT configuration with added parameters.

Hyperparameters are taken from run_task_main.py and hparam_utils.py of the original implementation. URLS:

- https://github.com/google-research/tapas/blob/master/tapas/run_task_main.py
- https://github.com/google-research/tapas/blob/master/tapas/utils/hparam_utils.py

"""

from ...configuration_utils import PretrainedConfig


class TapasConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`TapasModel`]. It is used to instantiate a TAPAS
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the TAPAS
    [google/tapas-base-finetuned-sqa](https://huggingface.co/google/tapas-base-finetuned-sqa) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Hyperparameters additional to BERT are taken from run_task_main.py and hparam_utils.py of the original
    implementation. Original implementation available at https://github.com/google-research/tapas/tree/master.

    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the TAPAS model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`TapasModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"swish"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 1024):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_sizes (`List[int]`, *optional*, defaults to `[3, 256, 256, 2, 256, 256, 10]`):
            The vocabulary sizes of the `token_type_ids` passed when calling [`TapasModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        positive_label_weight (`float`, *optional*, defaults to 10.0):
            Weight for positive labels.
        num_aggregation_labels (`int`, *optional*, defaults to 0):
            The number of aggregation operators to predict.
        aggregation_loss_weight (`float`, *optional*, defaults to 1.0):
            Importance weight for the aggregation loss.
        use_answer_as_supervision (`bool`, *optional*):
            Whether to use the answer as the only supervision for aggregation examples.
        answer_loss_importance (`float`, *optional*, defaults to 1.0):
            Importance weight for the regression loss.
        use_normalized_answer_loss (`bool`, *optional*, defaults to `False`):
            Whether to normalize the answer loss by the maximum of the predicted and expected value.
        huber_loss_delta (`float`, *optional*):
            Delta parameter used to calculate the regression loss.
        temperature (`float`, *optional*, defaults to 1.0):
            Value used to control (OR change) the skewness of cell logits probabilities.
        aggregation_temperature (`float`, *optional*, defaults to 1.0):
            Scales aggregation logits to control the skewness of probabilities.
        use_gumbel_for_cells (`bool`, *optional*, defaults to `False`):
            Whether to apply Gumbel-Softmax to cell selection.
        use_gumbel_for_aggregation (`bool`, *optional*, defaults to `False`):
            Whether to apply Gumbel-Softmax to aggregation selection.
        average_approximation_function (`string`, *optional*, defaults to `"ratio"`):
            Method to calculate the expected average of cells in the weak supervision case. One of `"ratio"`,
            `"first_order"` or `"second_order"`.
        cell_selection_preference (`float`, *optional*):
            Preference for cell selection in ambiguous cases. Only applicable in case of weak supervision for
            aggregation (WTQ, WikiSQL). If the total mass of the aggregation probabilities (excluding the "NONE"
            operator) is higher than this hyperparameter, then aggregation is predicted for an example.
        answer_loss_cutoff (`float`, *optional*):
            Ignore examples with answer loss larger than cutoff.
        max_num_rows (`int`, *optional*, defaults to 64):
            Maximum number of rows.
        max_num_columns (`int`, *optional*, defaults to 32):
            Maximum number of columns.
        average_logits_per_cell (`bool`, *optional*, defaults to `False`):
            Whether to average logits per cell.
        select_one_column (`bool`, *optional*, defaults to `True`):
            Whether to constrain the model to only select cells from a single column.
        allow_empty_column_selection (`bool`, *optional*, defaults to `False`):
            Whether to allow not to select any column.
        init_cell_selection_weights_to_zero (`bool`, *optional*, defaults to `False`):
            Whether to initialize cell selection weights to 0 so that the initial probabilities are 50%.
        reset_position_index_per_cell (`bool`, *optional*, defaults to `True`):
            Whether to restart position indexes at every cell (i.e. use relative position embeddings).
        disable_per_token_loss (`bool`, *optional*, defaults to `False`):
            Whether to disable any (strong or weak) supervision on cells.
        aggregation_labels (`Dict[int, label]`, *optional*):
            The aggregation labels used to aggregate the results. For example, the WTQ models have the following
            aggregation labels: `{0: "NONE", 1: "SUM", 2: "AVERAGE", 3: "COUNT"}`
        no_aggregation_label_index (`int`, *optional*):
            If the aggregation labels are defined and one of these labels represents "No aggregation", this should be
            set to its index. For example, the WTQ models have the "NONE" aggregation label at index 0, so that value
            should be set to 0 for these models.


    Example:

    ```python
    >>> from transformers import TapasModel, TapasConfig

    >>> # Initializing a default (SQA) Tapas configuration
    >>> configuration = TapasConfig()
    >>> # Initializing a model from the configuration
    >>> model = TapasModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "tapas"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=1024,
        type_vocab_sizes=[3, 256, 256, 2, 256, 256, 10],
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        positive_label_weight=10.0,
        num_aggregation_labels=0,
        aggregation_loss_weight=1.0,
        use_answer_as_supervision=None,
        answer_loss_importance=1.0,
        use_normalized_answer_loss=False,
        huber_loss_delta=None,
        temperature=1.0,
        aggregation_temperature=1.0,
        use_gumbel_for_cells=False,
        use_gumbel_for_aggregation=False,
        average_approximation_function="ratio",
        cell_selection_preference=None,
        answer_loss_cutoff=None,
        max_num_rows=64,
        max_num_columns=32,
        average_logits_per_cell=False,
        select_one_column=True,
        allow_empty_column_selection=False,
        init_cell_selection_weights_to_zero=False,
        reset_position_index_per_cell=True,
        disable_per_token_loss=False,
        aggregation_labels=None,
        no_aggregation_label_index=None,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        # BERT hyperparameters (with updated max_position_embeddings and type_vocab_sizes)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_sizes = type_vocab_sizes
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps

        # Fine-tuning task hyperparameters
        self.positive_label_weight = positive_label_weight
        self.num_aggregation_labels = num_aggregation_labels
        self.aggregation_loss_weight = aggregation_loss_weight
        self.use_answer_as_supervision = use_answer_as_supervision
        self.answer_loss_importance = answer_loss_importance
        self.use_normalized_answer_loss = use_normalized_answer_loss
        self.huber_loss_delta = huber_loss_delta
        self.temperature = temperature
        self.aggregation_temperature = aggregation_temperature
        self.use_gumbel_for_cells = use_gumbel_for_cells
        self.use_gumbel_for_aggregation = use_gumbel_for_aggregation
        self.average_approximation_function = average_approximation_function
        self.cell_selection_preference = cell_selection_preference
        self.answer_loss_cutoff = answer_loss_cutoff
        self.max_num_rows = max_num_rows
        self.max_num_columns = max_num_columns
        self.average_logits_per_cell = average_logits_per_cell
        self.select_one_column = select_one_column
        self.allow_empty_column_selection = allow_empty_column_selection
        self.init_cell_selection_weights_to_zero = init_cell_selection_weights_to_zero
        self.reset_position_index_per_cell = reset_position_index_per_cell
        self.disable_per_token_loss = disable_per_token_loss

        # Aggregation hyperparameters
        self.aggregation_labels = aggregation_labels
        self.no_aggregation_label_index = no_aggregation_label_index

        if isinstance(self.aggregation_labels, dict):
            self.aggregation_labels = {int(k): v for k, v in aggregation_labels.items()}


__all__ = ["TapasConfig"]
