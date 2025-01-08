# coding=utf-8
# Copyright 2019-present, the HuggingFace Inc. team, The Google AI Language Team and Facebook, Inc.
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

import math
from typing import Callable, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax

from ...modeling_flax_outputs import (
    FlaxBaseModelOutput,
    FlaxMaskedLMOutput,
    FlaxMultipleChoiceModelOutput,
    FlaxQuestionAnsweringModelOutput,
    FlaxSequenceClassifierOutput,
    FlaxTokenClassifierOutput,
)
from ...modeling_flax_utils import ACT2FN, FlaxPreTrainedModel, append_call_sample_docstring, overwrite_call_docstring
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_distilbert import DistilBertConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "distilbert-base-uncased"
_CONFIG_FOR_DOC = "DistilBertConfig"


FLAX_DISTILBERT_START_DOCSTRING = r"""

    This model inherits from [`FlaxPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading, saving and converting weights from PyTorch models)

    This model is also a
    [flax.linen.Module](https://flax.readthedocs.io/en/latest/api_reference/flax.linen/module.html) subclass. Use it as
    a regular Flax linen Module and refer to the Flax documentation for all matter related to general usage and
    behavior.

    Finally, this model supports inherent JAX features such as:

    - [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
    - [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
    - [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
    - [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

    Parameters:
        config ([`DistilBertConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

DISTILBERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`numpy.ndarray` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`numpy.ndarray` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    # create the sinusoidal pattern for the positional encoding
    angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return jnp.array(pos_encoding)


class FlaxEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    config: DistilBertConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.word_embeddings = nn.Embed(
            self.config.vocab_size,
            self.config.dim,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
        )
        if not self.config.sinusoidal_pos_embds:
            self.position_embeddings = nn.Embed(
                self.config.max_position_embeddings,
                self.config.dim,
                embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            )
        else:
            self.pos_encoding = positional_encoding(self.config.max_position_embeddings, self.config.dim)
        self.LayerNorm = nn.LayerNorm(epsilon=1e-12, dtype=self.dtype)
        self.dropout = nn.Dropout(rate=self.config.dropout)

    def __call__(self, input_ids, deterministic: bool = True):
        # Embed
        batch_size, seq_length = input_ids.shape
        inputs_embeds = self.word_embeddings(input_ids.astype("i4"))
        if not self.config.sinusoidal_pos_embds:
            position_ids = jnp.arange(seq_length).astype("i4")
            position_ids = jnp.broadcast_to(position_ids, shape=(batch_size, seq_length))
            position_embeds = self.position_embeddings(position_ids.astype("i4"))
        else:
            position_embeds = self.pos_encoding[:, :seq_length, :]
            # explicitly cast the positions here, since self.embed_positions are not registered as parameters
            position_embeds = position_embeds.astype(inputs_embeds.dtype)

        # Sum all embeddings
        hidden_states = inputs_embeds + position_embeds

        # Layer Norm
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        return hidden_states


class FlaxMultiHeadSelfAttention(nn.Module):
    config: DistilBertConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.n_heads = self.config.n_heads
        self.dim = self.config.dim
        self.dropout = nn.Dropout(rate=self.config.attention_dropout)

        if not (self.dim % self.n_heads == 0):
            raise ValueError(f"Hidden size {self.dim} not dividable by number of heads {self.n_heads}")

        self.q_lin = nn.Dense(
            self.dim,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
        )
        self.k_lin = nn.Dense(
            self.dim,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
        )
        self.v_lin = nn.Dense(
            self.dim,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
        )
        self.out_lin = nn.Dense(
            self.dim,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
        )

    def __call__(
        self,
        query,
        key,
        value,
        mask,
        deterministic: bool = True,
        output_attentions: bool = False,
    ):
        bs, q_len, dim = query.shape
        k_len = key.shape[1]
        # assert dim == self.dim, f'Dimensions do not match: {dim} input vs {self.dim} configured'
        # assert key.size() == value.size()

        dim_per_head = self.dim // self.n_heads

        mask_reshp = (bs, 1, 1, k_len)

        def shape(x):
            """separate heads"""
            return x.reshape(bs, -1, self.n_heads, dim_per_head).transpose(0, 2, 1, 3)

        def unshape(x):
            """group heads"""
            return x.transpose(0, 2, 1, 3).reshape(bs, -1, self.n_heads * dim_per_head)

        q = shape(self.q_lin(query))  # (bs, n_heads, q_len, dim_per_head)
        k = shape(self.k_lin(key))  # (bs, n_heads, k_len, dim_per_head)
        v = shape(self.v_lin(value))  # (bs, n_heads, k_len, dim_per_head)

        q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_len, dim_per_head)
        scores = jnp.matmul(q, k.transpose(0, 1, 3, 2))  # (bs, n_heads, q_len, k_len)
        mask = jnp.reshape(mask, mask_reshp)

        mask = mask.astype(scores.dtype)
        scores = scores - 1e30 * (1.0 - mask)

        weights = nn.softmax(scores, axis=-1)  # (bs, n_heads, q_len, k_len)
        weights = self.dropout(weights, deterministic=deterministic)

        context = jnp.matmul(weights, v)  # (bs, n_heads, q_len, dim_per_head)
        context = unshape(context)  # (bs, q_len, dim)
        context = self.out_lin(context)  # (bs, q_len, dim)

        if output_attentions:
            return (context, weights)
        else:
            return (context,)


class FlaxFFN(nn.Module):
    config: DistilBertConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.dropout = nn.Dropout(rate=self.config.dropout)
        self.chunk_size_feed_forward = self.config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.lin1 = nn.Dense(
            self.config.hidden_dim,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
        )
        self.lin2 = nn.Dense(
            self.config.dim,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
        )

        self.activation = ACT2FN[self.config.activation]

    def __call__(self, hidden_states, deterministic: bool = True):
        hidden_states = self.lin1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.lin2(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        return hidden_states


class FlaxTransformerBlock(nn.Module):
    config: DistilBertConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        assert (
            self.config.dim % self.config.n_heads == 0
        ), f"Hidden size {self.config.dim} not dividable by number of heads {self.config.n_heads}"

        self.attention = FlaxMultiHeadSelfAttention(self.config, dtype=self.dtype)
        self.sa_layer_norm = nn.LayerNorm(epsilon=1e-12, dtype=self.dtype)

        self.ffn = FlaxFFN(self.config, dtype=self.dtype)
        self.output_layer_norm = nn.LayerNorm(epsilon=1e-12, dtype=self.dtype)

    def __call__(
        self,
        hidden_states,
        attn_mask,
        output_attentions: bool = False,
        deterministic: bool = True,
    ):
        # Self-Attention
        sa_output = self.attention(
            query=hidden_states,
            key=hidden_states,
            value=hidden_states,
            mask=attn_mask,
            output_attentions=output_attentions,
            deterministic=deterministic,
        )
        if output_attentions:
            sa_output, sa_weights = sa_output
        else:
            assert type(sa_output) is tuple
            sa_output = sa_output[0]
        sa_output = self.sa_layer_norm(sa_output + hidden_states)

        # Feed Forward Network
        ffn_output = self.ffn(sa_output, deterministic=deterministic)
        ffn_output = self.output_layer_norm(ffn_output + sa_output)
        output = (ffn_output,)
        if output_attentions:
            output = (sa_weights,) + output
        return output


class FlaxTransformer(nn.Module):
    config: DistilBertConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.layers = [
            FlaxTransformerBlock(self.config, name=str(i), dtype=self.dtype) for i in range(self.config.n_layers)
        ]

    def __call__(
        self,
        hidden_states,
        attention_mask,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        deterministic: bool = True,
        return_dict: bool = False,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for layer_module in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states=hidden_states,
                attn_mask=attention_mask,
                output_attentions=output_attentions,
                deterministic=deterministic,
            )
            hidden_states = layer_outputs[-1]

            if output_attentions:
                assert len(layer_outputs) == 2
                attentions = layer_outputs[0]
                all_attentions = all_attentions + (attentions,)
            else:
                assert len(layer_outputs) == 1

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_attentions, all_hidden_states] if v is not None)
        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )


class FlaxTransformerEncoder(nn.Module):
    config: DistilBertConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.layer = FlaxTransformer(self.config, dtype=self.dtype)

    def __call__(
        self,
        hidden_states,
        attention_mask,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        deterministic: bool = True,
        return_dict: bool = False,
    ):
        return self.layer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            deterministic=deterministic,
            return_dict=return_dict,
        )


class FlaxDistilBertLMDecoder(nn.Module):
    config: DistilBertConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    bias_init: Callable[..., np.ndarray] = jax.nn.initializers.zeros

    def setup(self):
        self.bias = self.param("bias", self.bias_init, (self.config.vocab_size,))

    def __call__(self, inputs, kernel):
        inputs = jnp.asarray(inputs, self.dtype)
        kernel = jnp.asarray(kernel, self.dtype)
        y = lax.dot_general(inputs, kernel, (((inputs.ndim - 1,), (0,)), ((), ())))
        bias = jnp.asarray(self.bias, self.dtype)
        y = y + bias
        return y


class FlaxDistilBertPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = DistilBertConfig
    base_model_prefix = "distilbert"
    module_class: nn.Module = None

    def __init__(
        self,
        config: DistilBertConfig,
        input_shape: Tuple = (1, 1),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # init input tensors
        input_ids = jnp.zeros(input_shape, dtype="i4")
        attention_mask = jnp.ones_like(input_ids)

        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        random_params = self.module.init(rngs, input_ids, attention_mask, return_dict=False)["params"]

        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    @add_start_docstrings_to_model_forward(DISTILBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def __call__(
        self,
        input_ids,
        attention_mask=None,
        head_mask=None,
        params: dict = None,
        dropout_rng: jax.random.PRNGKey = None,
        train: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        return self.module.apply(
            {"params": params or self.params},
            jnp.array(input_ids, dtype="i4"),
            jnp.array(attention_mask, dtype="i4"),
            not train,
            output_attentions,
            output_hidden_states,
            return_dict,
            rngs=rngs,
        )


class FlaxDistilBertModule(nn.Module):
    config: DistilBertConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.embeddings = FlaxEmbeddings(self.config, dtype=self.dtype)
        self.transformer = FlaxTransformerEncoder(self.config, dtype=self.dtype)

    def __call__(
        self,
        input_ids,
        attention_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        input_embeds = self.embeddings(input_ids, deterministic=deterministic)
        return self.transformer(
            hidden_states=input_embeds,
            attention_mask=attention_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


@add_start_docstrings(
    "The bare DistilBert Model transformer outputting raw hidden-states without any specific head on top.",
    FLAX_DISTILBERT_START_DOCSTRING,
)
class FlaxDistilBertModel(FlaxDistilBertPreTrainedModel):
    module_class = FlaxDistilBertModule


append_call_sample_docstring(FlaxDistilBertModel, _CHECKPOINT_FOR_DOC, None, _CONFIG_FOR_DOC)


class FlaxDistilBertForMaskedLMModule(nn.Module):
    config: DistilBertConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.distilbert = FlaxDistilBertModule(self.config, dtype=self.dtype)
        self.vocab_transform = nn.Dense(
            self.config.dim,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
        )
        self.vocab_layer_norm = nn.LayerNorm(epsilon=1e-12, dtype=self.dtype)
        if self.config.tie_word_embeddings:
            self.vocab_projector = FlaxDistilBertLMDecoder(
                self.config,
                dtype=self.dtype,
            )
        else:
            self.vocab_projector = nn.Dense(
                self.config.vocab_size,
                dtype=self.dtype,
                kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            )

    def __call__(
        self,
        input_ids,
        attention_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        dlbrt_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            deterministic=deterministic,
            return_dict=return_dict,
        )
        hidden_states = dlbrt_output[0]
        prediction_logits = self.vocab_transform(hidden_states)
        prediction_logits = ACT2FN[self.config.activation](prediction_logits)
        prediction_logits = self.vocab_layer_norm(prediction_logits)

        if self.config.tie_word_embeddings:
            shared_embedding = self.distilbert.variables["params"]["embeddings"]["word_embeddings"]["embedding"]
            prediction_logits = self.vocab_projector(prediction_logits, shared_embedding.T)
        else:
            prediction_logits = self.vocab_projector(prediction_logits)

        if not return_dict:
            output = (prediction_logits,) + dlbrt_output[1:]
            return output

        return FlaxMaskedLMOutput(
            logits=prediction_logits,
            hidden_states=dlbrt_output.hidden_states,
            attentions=dlbrt_output.attentions,
        )


@add_start_docstrings("""DistilBert Model with a `language modeling` head on top.""", FLAX_DISTILBERT_START_DOCSTRING)
class FlaxDistilBertForMaskedLM(FlaxDistilBertPreTrainedModel):
    module_class = FlaxDistilBertForMaskedLMModule


append_call_sample_docstring(FlaxDistilBertForMaskedLM, _CHECKPOINT_FOR_DOC, FlaxMaskedLMOutput, _CONFIG_FOR_DOC)


class FlaxDistilBertForSequenceClassificationModule(nn.Module):
    config: DistilBertConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.distilbert = FlaxDistilBertModule(config=self.config, dtype=self.dtype)
        self.pre_classifier = nn.Dense(
            self.config.dim,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
        )
        self.dropout = nn.Dropout(rate=self.config.seq_classif_dropout)
        self.classifier = nn.Dense(
            self.config.num_labels,
            dtype=self.dtype,
        )

    def __call__(
        self,
        input_ids,
        attention_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # Model
        distilbert_output = self.distilbert(
            input_ids,
            attention_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = ACT2FN["relu"](pooled_output)
        pooled_output = self.dropout(pooled_output, deterministic=deterministic)
        logits = self.classifier(pooled_output)  # (bs, dim)

        if not return_dict:
            return (logits,) + distilbert_output[1:]

        return FlaxSequenceClassifierOutput(
            logits=logits,
            hidden_states=distilbert_output.hidden_states,
            attentions=distilbert_output.attentions,
        )


@add_start_docstrings(
    """
    DistilBert Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    FLAX_DISTILBERT_START_DOCSTRING,
)
class FlaxDistilBertForSequenceClassification(FlaxDistilBertPreTrainedModel):
    module_class = FlaxDistilBertForSequenceClassificationModule


append_call_sample_docstring(
    FlaxDistilBertForSequenceClassification,
    _CHECKPOINT_FOR_DOC,
    FlaxSequenceClassifierOutput,
    _CONFIG_FOR_DOC,
)


class FlaxDistilBertForMultipleChoiceModule(nn.Module):
    config: DistilBertConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.distilbert = FlaxDistilBertModule(config=self.config, dtype=self.dtype)
        self.pre_classifier = nn.Dense(
            self.config.dim,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
        )
        self.dropout = nn.Dropout(rate=self.config.seq_classif_dropout)
        self.classifier = nn.Dense(
            1,
            dtype=self.dtype,
        )

    def __call__(
        self,
        input_ids,
        attention_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1]
        input_ids = input_ids.reshape(-1, input_ids.shape[-1]) if input_ids is not None else None
        attention_mask = attention_mask.reshape(-1, attention_mask.shape[-1]) if attention_mask is not None else None

        # Model
        outputs = self.distilbert(
            input_ids,
            attention_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_state = outputs[0]
        pooled_output = hidden_state[:, 0]
        pooled_output = self.pre_classifier(pooled_output)
        pooled_output = ACT2FN["relu"](pooled_output)
        pooled_output = self.dropout(pooled_output, deterministic=deterministic)
        logits = self.classifier(pooled_output)

        reshaped_logits = logits.reshape(-1, num_choices)

        if not return_dict:
            return (reshaped_logits,) + outputs[2:]

        return FlaxMultipleChoiceModelOutput(
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    DistilBert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and
    a softmax) e.g. for RocStories/SWAG tasks.
    """,
    FLAX_DISTILBERT_START_DOCSTRING,
)
class FlaxDistilBertForMultipleChoice(FlaxDistilBertPreTrainedModel):
    module_class = FlaxDistilBertForMultipleChoiceModule


overwrite_call_docstring(
    FlaxDistilBertForMultipleChoice, DISTILBERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
)
append_call_sample_docstring(
    FlaxDistilBertForMultipleChoice,
    _CHECKPOINT_FOR_DOC,
    FlaxMultipleChoiceModelOutput,
    _CONFIG_FOR_DOC,
)


class FlaxDistilBertForTokenClassificationModule(nn.Module):
    config: DistilBertConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.distilbert = FlaxDistilBertModule(config=self.config, dtype=self.dtype)
        self.dropout = nn.Dropout(rate=self.config.dropout)
        self.classifier = nn.Dense(self.config.num_labels, dtype=self.dtype)

    def __call__(
        self,
        input_ids,
        attention_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # Model
        outputs = self.distilbert(
            input_ids,
            attention_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        logits = self.classifier(hidden_states)

        if not return_dict:
            return (logits,) + outputs[1:]

        return FlaxTokenClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    DistilBert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g.
    for Named-Entity-Recognition (NER) tasks.
    """,
    FLAX_DISTILBERT_START_DOCSTRING,
)
class FlaxDistilBertForTokenClassification(FlaxDistilBertPreTrainedModel):
    module_class = FlaxDistilBertForTokenClassificationModule


append_call_sample_docstring(
    FlaxDistilBertForTokenClassification,
    _CHECKPOINT_FOR_DOC,
    FlaxTokenClassifierOutput,
    _CONFIG_FOR_DOC,
)


class FlaxDistilBertForQuestionAnsweringModule(nn.Module):
    config: DistilBertConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.distilbert = FlaxDistilBertModule(config=self.config, dtype=self.dtype)
        self.qa_outputs = nn.Dense(self.config.num_labels, dtype=self.dtype)
        assert self.config.num_labels == 2
        self.dropout = nn.Dropout(rate=self.config.qa_dropout)

    def __call__(
        self,
        input_ids,
        attention_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Model
        distilbert_output = self.distilbert(
            input_ids,
            attention_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = distilbert_output[0]

        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        logits = self.qa_outputs(hidden_states)
        start_logits, end_logits = logits.split(self.config.num_labels, axis=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if not return_dict:
            return (start_logits, end_logits) + distilbert_output[1:]

        return FlaxQuestionAnsweringModelOutput(
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=distilbert_output.hidden_states,
            attentions=distilbert_output.attentions,
        )


@add_start_docstrings(
    """
    DistilBert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a
    linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    FLAX_DISTILBERT_START_DOCSTRING,
)
class FlaxDistilBertForQuestionAnswering(FlaxDistilBertPreTrainedModel):
    module_class = FlaxDistilBertForQuestionAnsweringModule


append_call_sample_docstring(
    FlaxDistilBertForQuestionAnswering,
    _CHECKPOINT_FOR_DOC,
    FlaxQuestionAnsweringModelOutput,
    _CONFIG_FOR_DOC,
)


__all__ = [
    "FlaxDistilBertForMaskedLM",
    "FlaxDistilBertForMultipleChoice",
    "FlaxDistilBertForQuestionAnswering",
    "FlaxDistilBertForSequenceClassification",
    "FlaxDistilBertForTokenClassification",
    "FlaxDistilBertModel",
    "FlaxDistilBertPreTrainedModel",
]
