# coding=utf-8
# Copyright 2020, The RAG Authors and The HuggingFace Inc. team.
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

"""TFRAG model implementation."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from ...configuration_utils import PretrainedConfig
from ...generation import TFLogitsProcessorList
from ...modeling_tf_utils import (
    TFCausalLanguageModelingLoss,
    TFModelInputType,
    TFPreTrainedModel,
    keras,
    shape_list,
    unpack_inputs,
)
from ...utils import ModelOutput, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_rag import RagConfig
from .retrieval_rag import RagRetriever


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "RagConfig"


@dataclass
class TFRetrievAugLMMarginOutput(ModelOutput):
    """
    Base class for retriever augmented marginalized models outputs.

    Args:
        loss (`tf.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss.
        logits (`tf.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head. The score is possibly marginalized over all documents for
            each vocabulary token.
        past_key_values (`List[tf.Tensor]`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            List of `tf.Tensor` of length `config.n_layers`, with each tensor of shape `(2, batch_size, num_heads,
            sequence_length, embed_size_per_head)`).

            Contains precomputed hidden-states (key and values in the attention blocks) of the decoder that can be used
            (see `past_key_values` input) to speed up sequential decoding.
        doc_scores (`tf.Tensor` of shape `(batch_size, config.n_docs)`):
            Score between each retrieved document embeddings (see `retrieved_doc_embeds`) and
            `question_encoder_last_hidden_state`.
        retrieved_doc_embeds (`tf.Tensor` of shape `(batch_size, config.n_docs, hidden_size)`, *optional*, returned when *output_retrieved=True*):
            Embedded documents retrieved by the retriever. Is used with `question_encoder_last_hidden_state` to compute
            the `doc_scores`.
        retrieved_doc_ids (`tf.Tensor` (int32) of shape `(batch_size, config.n_docs)`, *optional*, returned when *output_retrieved=True*):
            The indexes of the embedded documents retrieved by the retriever.
        context_input_ids (`tf.Tensor`(int32) of shape `(batch_size * config.n_docs, config.max_combined_length)`, *optional*, returned when *output_retrieved=True*):
            Input ids post-processed from the retrieved documents and the question encoder input_ids by the retriever.
        context_attention_mask (`tf.Tensor` (int32) of shape `(batch_size * config.n_docs, config.max_combined_length)`, *optional*, returned when *output_retrieved=True*):
            Attention mask post-processed from the retrieved documents and the question encoder `input_ids` by the
            retriever.
        question_encoder_last_hidden_state (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden states at the output of the last layer of the question encoder pooled output of the
            model.
        question_enc_hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings and one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden states of the question encoder at the output of each layer plus the initial embedding outputs.
        question_enc_attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the question encoder, after the attention softmax, used to compute the weighted
            average in the self-attention heads.
        generator_enc_last_hidden_state (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the generator encoder of the model.
        generator_enc_hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings and one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden states of the generator encoder at the output of each layer plus the initial embedding outputs.
        generator_enc_attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the generator encoder, after the attention softmax, used to compute the weighted
            average in the self-attention heads.
        generator_dec_hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings and one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden states of the generator decoder at the output of each layer plus the initial embedding outputs.
        generator_dec_attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the generator decoder, after the attention softmax, used to compute the weighted
            average in the self-attention heads.
    """

    loss: tf.Tensor | None = None
    logits: tf.Tensor = None
    past_key_values: List[tf.Tensor] | None = None
    doc_scores: tf.Tensor | None = None
    retrieved_doc_embeds: tf.Tensor | None = None
    retrieved_doc_ids: tf.Tensor | None = None
    context_input_ids: tf.Tensor | None = None
    context_attention_mask: tf.Tensor | None = None
    question_encoder_last_hidden_state: tf.Tensor | None = None
    question_enc_hidden_states: Tuple[tf.Tensor, ...] | None = None
    question_enc_attentions: Tuple[tf.Tensor, ...] | None = None
    generator_enc_last_hidden_state: tf.Tensor | None = None
    generator_enc_hidden_states: Tuple[tf.Tensor, ...] | None = None
    generator_enc_attentions: Tuple[tf.Tensor, ...] | None = None
    generator_dec_hidden_states: Tuple[tf.Tensor, ...] | None = None
    generator_dec_attentions: Tuple[tf.Tensor, ...] | None = None


@dataclass
class TFRetrievAugLMOutput(ModelOutput):
    """
    Args:
        logits (`tf.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head. The score is possibly marginalized over all documents for
            each vocabulary token.
        past_key_values (`List[tf.Tensor]`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            List of `tf.Tensor` of length `config.n_layers`, with each tensor of shape `(2, batch_size, num_heads,
            sequence_length, embed_size_per_head)`).

            Contains precomputed hidden-states (key and values in the attention blocks) of the decoder that can be used
            (see `past_key_values` input) to speed up sequential decoding.
        doc_scores (`tf.Tensor` of shape `(batch_size, config.n_docs)`):
            Score between each retrieved document embeddings (see `retrieved_doc_embeds`) and
            `question_encoder_last_hidden_state`.
        retrieved_doc_embeds (`tf.Tensor` of shape `(batch_size, config.n_docs, hidden_size)`, *optional*, returned when *output_retrieved=True*):
            Embedded documents retrieved by the retriever. Is used with `question_encoder_last_hidden_state` to compute
            the `doc_scores`.
        retrieved_doc_ids (`tf.Tensor` of shape `(batch_size, config.n_docs)`, *optional*, returned when *output_retrieved=True*):
            The indexes of the embedded documents retrieved by the retriever.
        context_input_ids (`tf.Tensor` of shape `(batch_size * config.n_docs, config.max_combined_length)`, *optional*, returned when *output_retrieved=True*):
            Input ids post-processed from the retrieved documents and the question encoder input_ids by the retriever.
        context_attention_mask (`tf.Tensor` of shape `(batch_size * config.n_docs, config.max_combined_length)`, *optional*, returned when *output_retrieved=True*):
            Attention mask post-processed from the retrieved documents and the question encoder `input_ids` by the
            retriever.
        question_encoder_last_hidden_state (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden states at the output of the last layer of the question encoder pooled output of the
            model.
        question_enc_hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings and one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden states of the question encoder at the output of each layer plus the initial embedding outputs.
        question_enc_attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the question encoder, after the attention softmax, used to compute the weighted
            average in the self-attention heads.
        generator_enc_last_hidden_state (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the generator encoder of the model.
        generator_enc_hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings and one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden states of the generator encoder at the output of each layer plus the initial embedding outputs.
        generator_enc_attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the generator encoder, after the attention softmax, used to compute the weighted
            average in the self-attention heads.
        generator_dec_hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings and one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden states of the generator decoder at the output of each layer plus the initial embedding outputs.
        generator_dec_attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the generator decoder, after the attention softmax, used to compute the weighted
            average in the self-attention heads.
    """

    logits: tf.Tensor = None
    past_key_values: List[tf.Tensor] | None = None
    doc_scores: tf.Tensor | None = None
    retrieved_doc_embeds: tf.Tensor | None = None
    retrieved_doc_ids: tf.Tensor | None = None
    context_input_ids: tf.Tensor | None = None
    context_attention_mask: tf.Tensor | None = None
    question_encoder_last_hidden_state: tf.Tensor | None = None
    question_enc_hidden_states: Tuple[tf.Tensor, ...] | None = None
    question_enc_attentions: Tuple[tf.Tensor, ...] | None = None
    generator_enc_last_hidden_state: tf.Tensor | None = None
    generator_enc_hidden_states: Tuple[tf.Tensor, ...] | None = None
    generator_enc_attentions: Tuple[tf.Tensor, ...] | None = None
    generator_dec_hidden_states: Tuple[tf.Tensor, ...] | None = None
    generator_dec_attentions: Tuple[tf.Tensor, ...] | None = None


class TFRagPreTrainedModel(TFPreTrainedModel):
    r"""
    RAG models were released with the paper [Retrieval-Augmented Generation for Knowledge-Intensive NLP
    Tasks](https://arxiv.org/abs/2005.11401) by Patrick Lewis, Ethan Perez, Aleksandra Piktus et al.

    RAG is a retriever augmented model and encapsulate three components: a question encoder, a dataset retriever and a
    generator, the encoder and generator are trainable while the retriever is just an indexed dataset.

    """

    config_class = RagConfig
    base_model_prefix = "rag"
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    @classmethod
    def from_pretrained_question_encoder_generator(
        cls,
        question_encoder_pretrained_model_name_or_path: str = None,
        generator_pretrained_model_name_or_path: str = None,
        retriever: RagRetriever = None,
        *model_args,
        **kwargs,
    ) -> TFPreTrainedModel:
        r"""
        Instantiates an question encoder and a generator from one or two base classes of the library from pretrained
        model checkpoints.

        Params:
            question_encoder_pretrained_model_name_or_path (`str`, *optional*):
                Information necessary to initiate the question encoder. Can be either:

                    - A string with the *shortcut name* of a pretrained model to load from cache or download, e.g.,
                      `google-bert/bert-base-uncased`.
                    - A string with the *identifier name* of a pretrained model that was user-uploaded to our S3, e.g.,
                      `dbmdz/bert-base-german-cased`.
                    - A path to a *directory* containing model weights saved using
                      [`~TFPreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
                    - A path or url to a *pytorch index checkpoint file* (e.g, `./pt_model/`). In this case,
                      `question_encoder_from_pt` should be set to `True`.

            generator_pretrained_model_name_or_path (`str`, *optional*, defaults to `None`):
                Information necessary to initiate the generator. Can be either:

                    - A string with the *shortcut name* of a pretrained model to load from cache or download, e.g.,
                      `google-t5/t5-small`.
                    - A string with the *identifier name* of a pretrained model that was user-uploaded to our S3, e.g.,
                      `facebook/bart-base`.
                    - A path to a *directory* containing model weights saved using
                      [`~TFPreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
                    - A path or url to a *pytorch checkpoint file* (e.g, `./pt_model/`). In this case,
                      `generator_from_pt` should be set to `True`.

            model_args (remaining positional arguments, *optional*):
                All remaining positional arguments will be passed to the underlying model's `__init__` method.
            retriever ([`RagRetriever`], *optional*):
                The retriever to use.
            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
                `output_attentions=True`).

                - To update the question_encoder configuration, use the prefix *question_encoder_* for each
                  configuration parameter.
                - To update the generator configuration, use the prefix *generator_* for each configuration parameter.
                - To update the parent model configuration, do not use a prefix for each configuration parameter.

                Behaves differently depending on whether a `config` is provided or automatically loaded.

        Example:

        ```python
        >>> from transformers import RagRetriever, TFRagModel

        >>> # initialize a RAG from two pretrained models.
        >>> model = TFRagModel.from_pretrained_question_encoder_generator(
        ...     "facebook/dpr-question_encoder-single-nq-base", "google-t5/t5-small"
        ... )
        >>> # alternatively, initialize from pytorch pretrained models can also be done
        >>> model = TFRagModel.from_pretrained_question_encoder_generator(
        ...     "facebook/dpr-question_encoder-single-nq-base",
        ...     "facebook/bart-base",
        ...     generator_from_pt=True,
        ...     question_encoder_from_pt=True,
        ... )

        >>> # saving model after fine-tuning
        >>> model.save_pretrained("./rag")

        >>> # load retriever
        >>> retriever = RagRetriever.from_pretrained(
        ...     "facebook/rag-token-base", index_name="exact", use_dummy_dataset=True
        ... )
        >>> # load fine-tuned model with retriever
        >>> model = TFRagModel.from_pretrained("./rag", retriever=retriever)
        ```"""

        kwargs_question_encoder = {
            argument[len("question_encoder_") :]: value
            for argument, value in kwargs.items()
            if argument.startswith("question_encoder_")
        }

        kwargs_generator = {
            argument[len("generator_") :]: value
            for argument, value in kwargs.items()
            if argument.startswith("generator_")
        }

        # remove question_encoder, generator kwargs from kwargs
        for key in kwargs_question_encoder.keys():
            del kwargs["question_encoder_" + key]
        for key in kwargs_generator.keys():
            del kwargs["generator_" + key]

        # Load and initialize the question_encoder and generator
        # The distinction between question_encoder and generator at the model level is made
        # by the value of the flag `is_generator` that we need to set correctly.
        question_encoder = kwargs_question_encoder.pop("model", None)
        if question_encoder is None:
            assert question_encoder_pretrained_model_name_or_path is not None, (
                "If `model` is not defined as an argument, a `question_encoder_pretrained_model_name_or_path` has to"
                " be defined"
            )

            from ..auto.modeling_tf_auto import TFAutoModel

            if "config" not in kwargs_question_encoder:
                from ..auto.configuration_auto import AutoConfig

                question_encoder_config = AutoConfig.from_pretrained(question_encoder_pretrained_model_name_or_path)
                kwargs_question_encoder["config"] = question_encoder_config

            question_encoder = TFAutoModel.from_pretrained(
                question_encoder_pretrained_model_name_or_path,
                name="question_encoder",
                load_weight_prefix=cls.load_weight_prefix,
                *model_args,
                **kwargs_question_encoder,
            )

        generator = kwargs_generator.pop("generator", None)
        if generator is None:
            assert generator_pretrained_model_name_or_path is not None, (
                "If `generator_model` is not defined as an argument, a `generator_pretrained_model_name_or_path` has"
                " to be defined"
            )

            from ..auto.modeling_tf_auto import TFAutoModelForSeq2SeqLM

            if "config" not in kwargs_generator:
                from ..auto.configuration_auto import AutoConfig

                generator_config = AutoConfig.from_pretrained(generator_pretrained_model_name_or_path)
                kwargs_generator["config"] = generator_config

            generator = TFAutoModelForSeq2SeqLM.from_pretrained(
                generator_pretrained_model_name_or_path,
                name="generator",
                load_weight_prefix=cls.load_weight_prefix,
                **kwargs_generator,
            )

        # instantiate config with corresponding kwargs
        config = kwargs.get("config", None)
        if config is None:
            config = RagConfig.from_question_encoder_generator_configs(
                question_encoder.config, generator.config, **kwargs
            )

        return cls(question_encoder=question_encoder, generator=generator, config=config, retriever=retriever)


RAG_START_DOCSTRING = r"""

    RAG is a sequence-to-sequence model which encapsulates two core components: a question encoder and a generator.
    During a forward pass, we encode the input with the question encoder and pass it to the retriever to extract
    relevant context documents. The documents are then prepended to the input. Such contextualized inputs is passed to
    the generator.

    The question encoder can be any *autoencoding* model, preferably [`TFDPRQuestionEncoder`], and the generator can be
    any *seq2seq* model, preferably [`TFBartForConditionalGeneration`].

    The model can be initialized with a [`RagRetriever`] for end-to-end generation or used in combination with the
    outputs of a retriever in multiple steps---see examples for more details. The model is compatible any
    *autoencoding* model as the `question_encoder` and any *seq2seq* model with language model head as the `generator`.
    It has been tested with [`TFDPRQuestionEncoder`] as the `question_encoder` and [`TFBartForConditionalGeneration`]
    as the `generator`.

    This model inherits from [`TFPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a Tensorflow [keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model)
    subclass. Use it as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to
    general usage and behavior.

    The model is in a developing state as it is now fully supports in eager-mode only, and may not be exported in
    SavedModel format.

    Args:
        config ([`RagConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~TFPreTrainedModel.from_pretrained`] method to load the model weights.
        question_encoder ([`TFPreTrainedModel`]):
            An encoder model compatible with the faiss index encapsulated by the `retriever`.
        generator ([`TFPreTrainedModel`]):
            A seq2seq model used as the generator in the RAG architecture.
        retriever ([`RagRetriever`]):
            A retriever class encapsulating a faiss index queried to obtain context documents for current inputs.
"""


RAG_FORWARD_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`tf.Tensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. [`RagConfig`], used to initialize the model, specifies
            which generator to use, it also specifies a compatible generator tokenizer. Use that tokenizer class to
            obtain the indices.
        attention_mask (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        encoder_outputs (`tuple(tuple(tf.Tensor)`, *optional*)
            Tuple consists of (`generator_enc_last_hidden_state`, *optional*: `generator_enc_hidden_states`,
            *optional*: `generator_enc_attentions`). `generator_enc_last_hidden_state` of shape `(batch_size, n_docs *
            sequence_length, hidden_size)` is a sequence of hidden-states at the output of the last layer of the
            generator's encoder.

            Used by the ([`TFRagModel`]) model during decoding.
        decoder_input_ids (`tf.Tensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Provide for generation tasks. `None` by default, construct as per instructions for the generator model
            you're using with your RAG instance.
        decoder_attention_mask (`torch.BoolTensor` of shape `(batch_size,  target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default.
        past_key_values (`tuple(tuple(tf.Tensor))`):
            Tuple consists of two elements: `encoder_outputs` of the RAG model (see `encoder_outputs`) and
            `past_key_values` of the underlying generator. Can be used to speed up decoding. `past_key_values` are used
            in the ([`RagTokenForGeneration`]) model during decoding.
        doc_scores (`tf.Tensor` of shape `(batch_size, config.n_docs)`):
            Score between each retrieved document embeddings (see `retrieved_doc_embeds`) and
            `question_encoder_last_hidden_state`. If the model has is not initialized with a `retriever` `doc_scores`
            has to be provided to the forward pass. `doc_scores` can be computed via
            `question_encoder_last_hidden_state` and `retrieved_doc_embeds`, see examples for more information.
        context_input_ids (`tf.Tensor` of shape `(batch_size * config.n_docs, config.max_combined_length)`, *optional*, returned when *output_retrieved=True*):
            Input IDs post-processed from the retrieved documents and the question encoder `input_ids` by the
            retriever.

            If the model has is not initialized with a `retriever` ``context_input_ids` has to be provided to the
            forward pass. `context_input_ids` are returned by [`~RagRetriever.__call__`]. context_attention_mask
            (`tf.Tensor` of shape `(batch_size * config.n_docs, config.max_combined_length)`, *optional*, returned when
            *output_retrieved=True*): Attention mask post-processed from the retrieved documents and the question
            encoder `input_ids` by the retriever.

            If the model has is not initialized with a `retriever` `context_attention_mask` has to be provided to the
            forward pass. `context_attention_mask` are returned by [`~RagRetriever.__call__`].
        use_cache (`bool`, *optional*, defaults to `True`):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        output_retrieved(`bool`, *optional*):
            Whether or not to return the `retrieved_doc_embeds`, `retrieved_doc_ids`, `context_input_ids` and
            `context_attention_mask`. See returned tensors for more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`TFRetrievAugLMOutput`] instead of a plain tuple.
        n_docs (`int`, *optional*, defaults to `config.n_docs``)
            Number of documents to retrieve and/or number of documents for which to generate an answer.
"""


@add_start_docstrings_to_model_forward(RAG_START_DOCSTRING)
class TFRagModel(TFRagPreTrainedModel):
    load_weight_prefix = "tf_rag_model_1"

    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        question_encoder: Optional[TFPreTrainedModel] = None,
        generator: Optional[TFPreTrainedModel] = None,
        retriever: Optional[RagRetriever] = None,
        load_weight_prefix: Optional[str] = None,
        **kwargs,
    ):
        assert config is not None or (
            question_encoder is not None and generator is not None
        ), "Either a configuration or an question_encoder and a generator has to be provided."

        if config is None:
            config = RagConfig.from_question_encoder_generator_configs(
                question_encoder.config, generator.config, **kwargs
            )
        else:
            assert isinstance(config, self.config_class), f"config: {config} has to be of type {self.config_class}"
        super().__init__(config, **kwargs)

        if question_encoder is None:
            from ..auto.modeling_tf_auto import TFAutoModel

            question_encoder = TFAutoModel.from_config(config.question_encoder, name="question_encoder")

        if generator is None:
            from ..auto.modeling_tf_auto import TFAutoModelForSeq2SeqLM

            load_weight_prefix = load_weight_prefix if load_weight_prefix is not None else self.load_weight_prefix
            generator = TFAutoModelForSeq2SeqLM.from_config(
                config.generator, name="generator", load_weight_prefix=load_weight_prefix + "/generator"
            )

        self.retriever = retriever
        if self.retriever is not None:
            assert isinstance(
                retriever, RagRetriever
            ), f"`self.retriever` is of type {type(self.retriever)}, but should be of type `RagRetriever`"
            self.retriever = retriever

        self.question_encoder = question_encoder
        self.generator = generator

    def set_retriever(self, retriever: RagRetriever):
        self.retriever = retriever

    @unpack_inputs
    @add_start_docstrings_to_model_forward(RAG_FORWARD_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFRetrievAugLMOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        encoder_outputs: np.ndarray | tf.Tensor | None = None,
        decoder_input_ids: np.ndarray | tf.Tensor | None = None,
        decoder_attention_mask: np.ndarray | tf.Tensor | None = None,
        past_key_values: Tuple[Tuple[Union[np.ndarray, tf.Tensor]]] | None = None,
        doc_scores: np.ndarray | tf.Tensor | None = None,
        context_input_ids: np.ndarray | tf.Tensor | None = None,
        context_attention_mask: np.ndarray | tf.Tensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_retrieved: bool | None = None,
        n_docs: int | None = None,
        return_dict: bool | None = None,
        training: bool = False,
        **kwargs,
    ) -> TFRetrievAugLMOutput:
        r"""
        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, RagRetriever, TFRagModel
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/rag-token-base")
        >>> retriever = RagRetriever.from_pretrained(
        ...     "facebook/rag-token-base", index_name="exact", use_dummy_dataset=True
        ... )
        >>> # initialize with RagRetriever to do everything in one forward call
        >>> model = TFRagModel.from_pretrained("facebook/rag-token-base", retriever=retriever, from_pt=True)

        >>> input_dict = tokenizer.prepare_seq2seq_batch(
        ...     "How many people live in Paris?", "In Paris, there are 10 million people.", return_tensors="tf"
        ... )
        >>> input_ids = input_dict["input_ids"]
        >>> outputs = model(input_ids)
        ```"""
        assert (
            "decoder_cached_states" not in kwargs
        ), "Please use past_key_values to cache intermediate outputs"  # from modeling_tf_bart.py

        # aliasing to minimize code changing
        n_docs = n_docs if n_docs is not None else self.config.n_docs

        # whether retriever has to be used
        has_to_retrieve = (
            self.retriever is not None
            and (context_input_ids is None or context_attention_mask is None or doc_scores is None)
            and encoder_outputs is None
        )

        # encoder_outputs are pre-computed during RAG-token generation
        if encoder_outputs is None:
            if has_to_retrieve:
                question_enc_outputs = self.question_encoder(
                    input_ids, attention_mask=attention_mask, return_dict=True, training=training
                )
                # see https://github.com/huggingface/transformers/blob/main/src/transformers/models/dpr/modeling_tf_dpr.py#L91
                question_encoder_last_hidden_state = question_enc_outputs[
                    0
                ]  # hidden states of question encoder => pooler_output

                retriever_outputs = self.retriever(
                    input_ids,
                    question_encoder_last_hidden_state.numpy(),
                    prefix=self.generator.config.prefix,
                    n_docs=n_docs,
                    return_tensors="tf",
                )
                context_input_ids, context_attention_mask, retrieved_doc_embeds, retrieved_doc_ids = (
                    retriever_outputs["context_input_ids"],
                    retriever_outputs["context_attention_mask"],
                    retriever_outputs["retrieved_doc_embeds"],
                    retriever_outputs["doc_ids"],
                )

                context_input_ids = tf.cast(context_input_ids, tf.int32)
                context_attention_mask = tf.cast(context_attention_mask, tf.int32)
                retrieved_doc_embeds = tf.cast(retrieved_doc_embeds, tf.float32)
                retrieved_doc_ids = tf.cast(retrieved_doc_ids, tf.int32)

                # compute doc_scores
                doc_scores = tf.squeeze(
                    tf.matmul(
                        tf.expand_dims(question_encoder_last_hidden_state, axis=1),
                        retrieved_doc_embeds,
                        transpose_b=True,
                    ),
                    axis=1,
                )

            else:
                assert context_input_ids is not None, (
                    "Make sure that `context_input_ids` are passed, if no `retriever` is set. Alternatively, you can"
                    " set a retriever using the `set_retriever(...)` function."
                )
                assert context_attention_mask is not None, (
                    "Make sure that `context_attention_mask` are passed, if no `retriever` is set. Alternatively, you"
                    " can set a retriever using the `set_retriever(...)` function."
                )
                assert doc_scores is not None, (
                    "Make sure that `doc_scores` are passed, if no `retriever` is set. Alternatively, you can set a"
                    " retriever using the `set_retriever(...)` function."
                )

        assert (
            doc_scores is not None
        ), "Make sure that `doc_scores` are passed when passing `encoder_outputs` to the forward function."

        assert (doc_scores.shape[1] % n_docs) == 0, (
            f" The first dimension of `context_input_ids` should be a multiple of `n_docs`={n_docs}, but is"
            f" {context_input_ids.shape[0]}."
        )

        # Decoder input without context documents
        if decoder_input_ids is not None:
            decoder_input_ids = tf.repeat(decoder_input_ids, n_docs, axis=0)

        if decoder_attention_mask is not None:
            decoder_attention_mask = tf.repeat(decoder_attention_mask, n_docs, axis=0)

        gen_outputs = self.generator(
            context_input_ids,
            attention_mask=context_attention_mask,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=True,
            training=training,
        )

        if not has_to_retrieve:
            question_encoder_last_hidden_state = None
            question_enc_hidden_states = None
            question_enc_attentions = None
            retrieved_doc_embeds = None
            retrieved_doc_ids = None
        else:
            question_enc_hidden_states = question_enc_outputs.hidden_states
            question_enc_attentions = question_enc_outputs.attentions

        if not has_to_retrieve or not output_retrieved:
            # don't output retrieved docs
            context_input_ids = (None,)
            context_attention_mask = None
            retrieved_doc_embeds = None
            retrieved_doc_ids = None

        return TFRetrievAugLMOutput(
            logits=gen_outputs.logits,
            doc_scores=doc_scores,
            past_key_values=gen_outputs.past_key_values,
            context_input_ids=context_input_ids,
            context_attention_mask=context_attention_mask,
            retrieved_doc_embeds=retrieved_doc_embeds,
            retrieved_doc_ids=retrieved_doc_ids,
            question_encoder_last_hidden_state=question_encoder_last_hidden_state,
            question_enc_hidden_states=question_enc_hidden_states,
            question_enc_attentions=question_enc_attentions,
            generator_enc_last_hidden_state=gen_outputs.encoder_last_hidden_state,
            generator_enc_hidden_states=gen_outputs.encoder_hidden_states,
            generator_enc_attentions=gen_outputs.encoder_attentions,
            generator_dec_hidden_states=gen_outputs.decoder_hidden_states,
            generator_dec_attentions=gen_outputs.decoder_attentions,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        with tf.name_scope(self.generator.name):
            self.generator.build(None)
        with tf.name_scope(self.question_encoder.name):
            self.question_encoder.build(None)


@add_start_docstrings_to_model_forward(
    """
    A TF RAG-token model implementation. It performs RAG-token specific marginalization in the forward pass.
    """,
    RAG_START_DOCSTRING,
)
class TFRagTokenForGeneration(TFRagPreTrainedModel, TFCausalLanguageModelingLoss):
    load_weight_prefix = "tf_rag_token_for_generation_1/rag"

    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        question_encoder: Optional[TFPreTrainedModel] = None,
        generator: Optional[TFPreTrainedModel] = None,
        retriever: Optional[RagRetriever] = None,
        **kwargs,
    ):
        assert config is not None or (
            question_encoder is not None and generator is not None
        ), "Either a configuration or an encoder and a generator has to be provided."

        if config is None:
            config = RagConfig.from_question_encoder_generator_configs(
                question_encoder.config, generator.config, **kwargs
            )

        super().__init__(config)

        # instantiate model
        self.rag = TFRagModel(
            config=config,
            question_encoder=question_encoder,
            generator=generator,
            retriever=retriever,
            load_weight_prefix=self.load_weight_prefix,
            name="rag",
        )

    def set_retriever(self, retriever: RagRetriever):
        self.rag.retriever = retriever

    # Adapted from https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_tf_bart.py
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        use_cache=None,
        encoder_outputs=None,
        doc_scores=None,
        n_docs=None,
        **kwargs,
    ):
        if past_key_values is not None:
            # if past is defined use only last decoder_input_ids
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,
            "encoder_outputs": encoder_outputs,
            "doc_scores": doc_scores,
            "context_attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "do_marginalize": True,
            "n_docs": n_docs,
        }

    @property
    def retriever(self):
        return self.rag.retriever

    @property
    def generator(self):
        return self.rag.generator

    @property
    def question_encoder(self):
        return self.rag.question_encoder

    @staticmethod
    def _gather_beams(nested, beam_indices, batch_axis=0):
        """
        RAG-specific `_gather_beams`: gathers the beam slices indexed by beam_indices into new beam array. If the
        nested tensor has a shape mismatch with the beam indices, then it means it is the cache. In that case, isolates
        and takes care of the extra dimension for ndocs.
        """

        def gather_fn(tensor):
            is_rag_cache = tensor.shape[0] != beam_indices.shape[0]
            if is_rag_cache:
                n_docs = tensor.shape[0] // beam_indices.shape[0]
                batch_size = beam_indices.shape[0]
                # reshapes into (batch size, num beams, n_docs, ...), the cache format expected by RAG
                tensor = tf.reshape(tensor, (batch_size, -1, n_docs, *tensor.shape[2:]))

            gathered_tensor = tf.gather(params=tensor, indices=beam_indices, axis=1, batch_dims=1)

            if is_rag_cache:
                # reshapes back into the shape expected by beam search
                gathered_tensor = tf.reshape(gathered_tensor, (batch_size * n_docs, -1, *gathered_tensor.shape[3:]))

            return gathered_tensor

        return tf.nest.map_structure(gather_fn, nested)

    def marginalize(self, seq_logits, doc_scores, n_docs=None):
        n_docs = n_docs if n_docs is not None else self.config.n_docs

        # RAG-token marginalization
        seq_logprobs = tf.nn.log_softmax(seq_logits, axis=-1)
        seq_logprobs = tf.reshape(seq_logprobs, [seq_logits.shape[0] // n_docs, n_docs, -1, seq_logits.shape[-1]])
        doc_logprobs = tf.nn.log_softmax(doc_scores, axis=1)
        doc_logprobs = tf.expand_dims(doc_logprobs, axis=-1)
        doc_logprobs = tf.expand_dims(doc_logprobs, axis=-1)  # twice
        log_prob_sum = seq_logprobs + doc_logprobs
        return tf.reduce_logsumexp(log_prob_sum, axis=1)

    @unpack_inputs
    @add_start_docstrings_to_model_forward(RAG_FORWARD_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFRetrievAugLMMarginOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        decoder_input_ids: np.ndarray | tf.Tensor | None = None,
        decoder_attention_mask: np.ndarray | tf.Tensor | None = None,
        encoder_outputs: np.ndarray | tf.Tensor | None = None,
        past_key_values: Tuple[Tuple[Union[np.ndarray, tf.Tensor]]] | None = None,
        doc_scores: np.ndarray | tf.Tensor | None = None,
        context_input_ids: np.ndarray | tf.Tensor | None = None,
        context_attention_mask: np.ndarray | tf.Tensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_retrieved: bool | None = None,
        n_docs: int | None = None,
        do_marginalize: bool | None = None,
        labels: np.ndarray | tf.Tensor | None = None,
        reduce_loss: bool | None = None,
        return_dict: bool | None = None,
        training: bool = False,
        **kwargs,  # needs kwargs for generation
    ) -> TFRetrievAugLMMarginOutput:
        r"""
        do_marginalize (`bool`, *optional*):
            If `True`, the logits are marginalized over all documents by making use of
            `torch.nn.functional.log_softmax`.
        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the cross entropy classification loss according to Rag-Token model formulation See
            https://arxiv.org/pdf/2005.11401.pdf Section 2.1 for details about Rag-Token formulation. Indices should be
            in `[0, ..., config.vocab_size - 1]`.
        reduce_loss (`bool`, *optional*):
            Only relevant if `labels` is passed. If `True`, the NLL loss is reduced using the `tf.Tensor.sum`
            operation.
        kwargs (`Dict[str, any]`, *optional*, defaults to `{}`):
            Legacy dictionary, which is required so that model can use *generate()* function.

        Returns:

        Example:

        ```python
        >>> import tensorflow as tf
        >>> from transformers import AutoTokenizer, RagRetriever, TFRagTokenForGeneration

        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/rag-token-nq")
        >>> retriever = RagRetriever.from_pretrained(
        ...     "facebook/rag-token-nq", index_name="exact", use_dummy_dataset=True
        ... )
        >>> # initialize with RagRetriever to do everything in one forward call
        >>> model = TFRagTokenForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever, from_pt=True)

        >>> input_dict = tokenizer.prepare_seq2seq_batch(
        ...     "How many people live in Paris?", "In Paris, there are 10 million people.", return_tensors="tf"
        ... )
        >>> outputs = model(input_dict, output_retrieved=True)

        >>> # or use retriever separately
        >>> # 1. Encode
        >>> input_ids = input_dict["input_ids"]
        >>> question_hidden_states = model.question_encoder(input_ids)[0]
        >>> # 2. Retrieve
        >>> docs_dict = retriever(input_ids.numpy(), question_hidden_states.numpy(), return_tensors="tf")
        >>> doc_scores = tf.squeeze(
        ...     tf.matmul(
        ...         tf.expand_dims(question_hidden_states, axis=1), docs_dict["retrieved_doc_embeds"], transpose_b=True
        ...     ),
        ...     axis=1,
        ... )
        >>> # 3. Forward to generator
        >>> outputs = model(
        ...     inputs=None,
        ...     context_input_ids=docs_dict["context_input_ids"],
        ...     context_attention_mask=docs_dict["context_attention_mask"],
        ...     doc_scores=doc_scores,
        ...     decoder_input_ids=input_dict["labels"],
        ... )

        >>> # or directly generate
        >>> generated = model.generate(
        ...     context_input_ids=docs_dict["context_input_ids"],
        ...     context_attention_mask=docs_dict["context_attention_mask"],
        ...     doc_scores=doc_scores,
        ... )
        >>> generated_string = tokenizer.batch_decode(generated, skip_special_tokens=True)
        ```"""

        assert (
            "decoder_cached_states" not in kwargs
        ), "Please use past_key_values to cache intermediate outputs"  # from modeling_tf_bart.py

        do_marginalize = do_marginalize if do_marginalize else self.config.do_marginalize
        reduce_loss = reduce_loss if reduce_loss else self.config.reduce_loss

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = labels
            use_cache = False

        outputs = self.rag(
            input_ids,
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            context_input_ids=context_input_ids,
            context_attention_mask=context_attention_mask,
            doc_scores=doc_scores,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_retrieved=output_retrieved,
            n_docs=n_docs,
            training=training,
        )

        loss = None
        logits = outputs.logits
        if labels is not None:
            assert decoder_input_ids is not None
            loss = self.get_nll(
                outputs.logits,
                outputs.doc_scores,
                labels,
                reduce_loss=reduce_loss,
                epsilon=self.config.label_smoothing,
                n_docs=n_docs,
            )

        if do_marginalize:
            logits = self.marginalize(logits, outputs.doc_scores, n_docs)

        return TFRetrievAugLMMarginOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            doc_scores=outputs.doc_scores,
            context_input_ids=outputs.context_input_ids,
            context_attention_mask=outputs.context_attention_mask,
            retrieved_doc_embeds=outputs.retrieved_doc_embeds,
            retrieved_doc_ids=outputs.retrieved_doc_ids,
            question_encoder_last_hidden_state=outputs.question_encoder_last_hidden_state,
            question_enc_hidden_states=outputs.question_enc_hidden_states,
            question_enc_attentions=outputs.question_enc_attentions,
            generator_enc_last_hidden_state=outputs.generator_enc_last_hidden_state,
            generator_enc_hidden_states=outputs.generator_enc_hidden_states,
            generator_enc_attentions=outputs.generator_enc_attentions,
            generator_dec_hidden_states=outputs.generator_dec_hidden_states,
            generator_dec_attentions=outputs.generator_dec_attentions,
        )

    def generate(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: tf.Tensor | None = None,
        context_input_ids=None,
        context_attention_mask=None,
        doc_scores=None,
        n_docs=None,
        generation_config=None,
        logits_processor=TFLogitsProcessorList(),
        **kwargs,
    ):
        """
        Implements TFRAG token decoding.

        Args:
            input_ids (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                The sequence used as a prompt for the generation. If `input_ids` is not passed, then
                `context_input_ids` has to be provided.
            attention_mask (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            context_input_ids (`tf.Tensor` of shape `(batch_size * config.n_docs, config.max_combined_length)`, *optional*, returned when *output_retrieved=True*):
                Input IDs post-processed from the retrieved documents and the question encoder `input_ids` by the
                retriever.

                If the model has is not initialized with a `retriever`, `context_input_ids` has to be provided to the
                forward pass. `context_input_ids` are returned by [`~RagRetriever.__call__`].
            context_attention_mask (`tf.Tensor` of shape `(batch_size * config.n_docs, config.max_combined_length)`, *optional*, returned when *output_retrieved=True*):
                Attention mask post-processed from the retrieved documents and the question encoder `input_ids` by the
                retriever.

                If the model has is not initialized with a `retriever`, `context_input_ids` has to be provided to the
                forward pass. `context_input_ids` are returned by [`~RagRetriever.__call__`].
            doc_scores (`tf.Tensor` of shape `(batch_size, config.n_docs)`):
                Score between each retrieved document embeddings (see `retrieved_doc_embeds`) and
                `question_encoder_last_hidden_state`.

                If the model has is not initialized with a `retriever`, `context_input_ids` has to be provided to the
                forward pass. `context_input_ids` are returned by [`~RagRetriever.__call__`].
            n_docs (`int`, *optional*, defaults to `config.n_docs`)
                Number of documents to retrieve and/or number of documents for which to generate an answer.
            generation_config (`~generation.GenerationConfig`, *optional*):
                The generation configuration to be used as base parametrization for the generation call. `**kwargs`
                passed to generate matching the attributes of `generation_config` will override them. If
                `generation_config` is not provided, the default will be used, which had the following loading
                priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
                configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
                default values, whose documentation should be checked to parameterize generation.
            logits_processor (`TFLogitsProcessorList`, *optional*):
                Custom logits processors that complement the default logits processors built from arguments and a
                model's config. If a logit processor is passed that is already created with the arguments or a model's
                config an error is thrown.
            kwargs (`Dict[str, Any]`, *optional*):
                Ad hoc parametrization of `generate_config` and/or additional model-specific kwargs that will be
                forwarded to the `forward` function of the model.

        Return:
            `tf.Tensor` of shape `(batch_size * num_return_sequences, sequence_length)`: The generated sequences. The
            second dimension (sequence_length) is either equal to `max_length` or shorter if all batches finished early
            due to the `eos_token_id`.
        """
        # Handle `generation_config` and kwargs that might update it
        if generation_config is None:
            generation_config = self.generation_config
        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(**kwargs)  # All unused kwargs must be model kwargs

        # set default parameters
        n_docs = n_docs if n_docs is not None else self.config.n_docs

        # retrieve docs
        if self.retriever is not None and context_input_ids is None:
            question_hidden_states = self.question_encoder(input_ids, attention_mask=attention_mask)[0]
            out = self.retriever(
                input_ids,
                question_hidden_states.numpy().astype(np.float32),
                prefix=self.generator.config.prefix,
                n_docs=n_docs,
                return_tensors="tf",
            )
            context_input_ids, context_attention_mask, retrieved_doc_embeds = (
                out["context_input_ids"],
                out["context_attention_mask"],
                out["retrieved_doc_embeds"],
            )

            context_input_ids = tf.cast(context_input_ids, tf.int32)
            context_attention_mask = tf.cast(context_attention_mask, tf.int32)
            retrieved_doc_embeds = tf.cast(retrieved_doc_embeds, tf.float32)

            # compute doc_scores
            doc_scores = tf.matmul(
                tf.expand_dims(question_hidden_states, axis=1), retrieved_doc_embeds, transpose_b=True
            )
            doc_scores = tf.squeeze(doc_scores, axis=1)

        assert (context_input_ids.shape[0] % n_docs) == 0, (
            f" The first dimension of `context_input_ids` should be a multiple of `n_docs`={n_docs}, but is"
            f" {context_input_ids.shape[0]}."
        )

        batch_size = context_input_ids.shape[0] // n_docs

        encoder = self.rag.generator.get_encoder()
        encoder_outputs = encoder(
            input_ids=context_input_ids,
            attention_mask=context_attention_mask,
            output_attentions=generation_config.output_attentions,
            output_hidden_states=generation_config.output_hidden_states,
            return_dict=True,
        )

        decoder_input_ids = tf.fill(
            (batch_size * generation_config.num_beams, 1),
            tf.cast(generation_config.decoder_start_token_id, tf.int32),
        )
        last_hidden_state = encoder_outputs["last_hidden_state"]

        def extend_enc_output(tensor, num_beams=None):
            """
            Broadcast tensor with `num_beams` replica, with correct order Input: tensor of shape (batch_size*n_docs ,
            d) Output: tensor of shape (batch_size*num_beams*n_docs , d)
            """

            # expand batch_size & num_beam dimensions
            d_shape_list = tensor.shape[1:]

            # split n_docs dimensions
            new_shape = (batch_size, 1, n_docs) + d_shape_list
            tensor = tf.reshape(tensor, new_shape)

            # repeat same last hidden states over `num_beams` dimension
            new_shape = (batch_size, num_beams, n_docs) + d_shape_list
            tensor = tf.broadcast_to(tensor, new_shape)

            # merge `batch_size`, `num_beams`, `num_docs` dims again
            new_shape = (batch_size * num_beams * n_docs,) + d_shape_list
            return tf.reshape(tensor, new_shape)

        # correctly extend last_hidden_state and attention mask
        context_attention_mask = extend_enc_output(context_attention_mask, num_beams=generation_config.num_beams)
        encoder_outputs["last_hidden_state"] = extend_enc_output(
            last_hidden_state, num_beams=generation_config.num_beams
        )

        doc_scores = tf.repeat(doc_scores, generation_config.num_beams, axis=0)

        # define start_len & additional parameters
        model_kwargs["doc_scores"] = doc_scores
        model_kwargs["encoder_outputs"] = encoder_outputs
        model_kwargs["attention_mask"] = context_attention_mask
        model_kwargs["n_docs"] = n_docs

        pre_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=tf.shape(decoder_input_ids)[-1],
            logits_processor=logits_processor,
        )

        if generation_config.num_beams == 1:
            return self.greedy_search(
                input_ids=decoder_input_ids,
                max_length=generation_config.max_length,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                logits_processor=pre_processor,
                output_attentions=generation_config.output_attentions,
                output_hidden_states=generation_config.output_hidden_states,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                **model_kwargs,
            )
        elif generation_config.num_beams > 1:
            if generation_config.num_beams < generation_config.num_return_sequences:
                raise ValueError(
                    "Beam search decoding cannot return more sequences than it has beams. Please set num_beams >="
                    f" num_return_sequences, got {generation_config.num_beams} and"
                    f" {generation_config.num_return_sequences} (respectivelly)"
                )

            def unflatten_beam_dim(tensor):
                """Unflattens the first, flat batch*beam dimension of a non-scalar array."""
                shape = shape_list(tensor)
                return tf.reshape(tensor, [-1, generation_config.num_beams] + shape[1:])

            decoder_input_ids = unflatten_beam_dim(decoder_input_ids)
            model_kwargs["attention_mask"] = unflatten_beam_dim(model_kwargs["attention_mask"])
            model_kwargs["encoder_outputs"]["last_hidden_state"] = unflatten_beam_dim(
                model_kwargs["encoder_outputs"]["last_hidden_state"]
            )

            return self.beam_search(
                input_ids=decoder_input_ids,
                max_length=generation_config.max_length,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                logits_processor=pre_processor,
                output_attentions=generation_config.output_attentions,
                output_hidden_states=generation_config.output_hidden_states,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                **model_kwargs,
            )
        else:
            raise ValueError(
                f"`num_beams` has to be an integer strictly superior to 0 (≥ 1), but is {generation_config.num_beams}"
            )

    def get_input_embeddings(self):
        return self.rag.generator.get_input_embeddings()

    def get_output_embeddings(self):
        return self.rag.generator.get_output_embeddings()

    # Adapted from tf_t5's & tf_bart's _shift_right
    def shift_tokens_right(self, input_ids, start_token_id=None):
        """Shift input ids one token to the right, and pad with start_token_id"""

        if start_token_id is None:
            start_token_id = self.generator.config.decoder_start_token_id
            assert start_token_id is not None, (
                "self.generator.config.decoder_start_token_id has to be defined. In Rag we commonly use Bart as"
                " generator, see Bart docs for more information"
            )

        pad_token_id = self.generator.config.pad_token_id
        assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."

        start_tokens = tf.fill((shape_list(input_ids)[0], 1), tf.cast(start_token_id, input_ids.dtype))
        shifted_input_ids = tf.concat([start_tokens, input_ids[:, :-1]], -1)

        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids = tf.where(
            shifted_input_ids == -100,
            tf.fill(shape_list(shifted_input_ids), tf.cast(pad_token_id, input_ids.dtype)),
            shifted_input_ids,
        )

        # "Verify that `labels` has only positive values and -100"
        assert_gte0 = tf.debugging.assert_greater_equal(shifted_input_ids, tf.cast(0, shifted_input_ids.dtype))

        # Make sure the assertion op is called by wrapping the result in an identity no-op
        with tf.control_dependencies([assert_gte0]):
            shifted_input_ids = tf.identity(shifted_input_ids)

        return shifted_input_ids

    # nll stands for 'negative log likelihood'
    def get_nll(self, seq_logits, doc_scores, target, reduce_loss=False, epsilon=0.0, n_docs=None):
        n_docs = n_docs if n_docs is not None else self.config.n_docs
        # shift tokens left (from original Pytorch's version)

        target = tf.concat(
            [target[:, 1:], tf.fill([target.shape[0], 1], tf.cast(self.config.generator.pad_token_id, target.dtype))],
            axis=1,
        )
        rag_logprobs = self.marginalize(seq_logits, doc_scores, n_docs)
        loss = self.hf_compute_loss(target, rag_logprobs, from_logits=True, reduce_loss=reduce_loss)

        return loss

    # Adopted modeling_tf_bart + add smooth_loss to match with pytorch version
    def hf_compute_loss(self, labels, y_pred, smooth_epsilon=0.0, from_logits=True, reduce_loss=False):
        """CrossEntropyLoss that ignores pad tokens"""
        # Matt: As written, this loss is not XLA-compatible, but it's doing some very weird things
        #       and I don't feel comfortable converting it.
        loss_fn = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction=keras.losses.Reduction.SUM,
        )

        if from_logits is False:  # convert to logits
            eps = 1e-9
            y_pred = tf.clip_by_value(y_pred, clip_value_min=eps, clip_value_max=1 - eps)
            y_pred = tf.math.log(y_pred)

        logits = y_pred
        melted_labels = tf.reshape(labels, (-1,))
        active_loss = tf.not_equal(melted_labels, self.config.generator.pad_token_id)

        reduced_logits = tf.boolean_mask(tf.reshape(logits, (-1, logits.shape[2])), active_loss)
        labels = tf.boolean_mask(melted_labels, active_loss)
        nll_loss = loss_fn(labels, reduced_logits)

        smooth_loss = -tf.reduce_sum(reduced_logits, axis=-1)
        smooth_loss = tf.reduce_sum(smooth_loss)  # sum and squeeze like torch
        eps_i = smooth_epsilon / reduced_logits.shape[-1]

        loss = (1.0 - smooth_epsilon) * nll_loss + eps_i * smooth_loss

        return loss

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "rag", None) is not None:
            with tf.name_scope(self.rag.name):
                self.rag.build(None)


@add_start_docstrings_to_model_forward(
    """
    A TF RAG-sequence model implementation. It performs RAG-sequence specific marginalization in the forward pass.
    """,
    RAG_START_DOCSTRING,
)
class TFRagSequenceForGeneration(TFRagPreTrainedModel, TFCausalLanguageModelingLoss):
    load_weight_prefix = "tf_rag_sequence_for_generation_1/rag"

    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        question_encoder: Optional[TFPreTrainedModel] = None,
        generator: Optional[TFPreTrainedModel] = None,
        retriever: Optional[RagRetriever] = None,
        **kwargs,
    ):
        assert config is not None or (
            question_encoder is not None and generator is not None
        ), "Either a configuration or an encoder and a generator has to be provided."

        if config is None:
            config = RagConfig.from_question_encoder_generator_configs(
                question_encoder.config, generator.config, **kwargs
            )

        super().__init__(config)

        # instantiate model
        self.rag = TFRagModel(
            config=config,
            question_encoder=question_encoder,
            generator=generator,
            retriever=retriever,
            load_weight_prefix=self.load_weight_prefix,
            name="rag",
        )

    def set_retriever(self, retriever: RagRetriever):
        self.rag.retriever = retriever

    @property
    def retriever(self):
        return self.rag.retriever

    @property
    def generator(self):
        return self.rag.generator

    @property
    def question_encoder(self):
        return self.rag.question_encoder

    @unpack_inputs
    @add_start_docstrings_to_model_forward(RAG_FORWARD_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFRetrievAugLMMarginOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        decoder_input_ids: np.ndarray | tf.Tensor | None = None,
        decoder_attention_mask: np.ndarray | tf.Tensor | None = None,
        encoder_outputs: np.ndarray | tf.Tensor | None = None,
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        doc_scores: np.ndarray | tf.Tensor | None = None,
        context_input_ids: np.ndarray | tf.Tensor | None = None,
        context_attention_mask: np.ndarray | tf.Tensor | None = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_retrieved: Optional[bool] = None,
        n_docs: Optional[int] = None,
        exclude_bos_score: Optional[bool] = None,
        labels: np.ndarray | tf.Tensor | None = None,
        reduce_loss: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
        **kwargs,  # needs kwargs for generation
    ) -> Union[Tuple[tf.Tensor], TFRetrievAugLMMarginOutput]:
        r"""
        exclude_bos_score (`bool`, *optional*):
            Only relevant if `labels` is passed. If `True`, the score of the BOS token is disregarded when computing
            the loss.
        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the cross entropy classification loss according to Rag-Sequence model formulation See
            https://arxiv.org/pdf/2005.11401.pdf Section 2.1 for details about Rag-Sequence formulation. Indices should
            be in `[0, ..., config.vocab_size - 1]`.
        reduce_loss (`bool`, *optional*):
            Only relevant if `labels` is passed. If `True`, the NLL loss is reduced using the `tf.Tensor.sum`
            operation.
        kwargs (`Dict[str, any]`, *optional*, defaults to `{}`):
            Legacy dictionary, which is required so that model can use *generate()* function.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, RagRetriever, TFRagSequenceForGeneration

        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/rag-sequence-nq")
        >>> retriever = RagRetriever.from_pretrained(
        ...     "facebook/rag-sequence-nq", index_name="exact", use_dummy_dataset=True
        ... )
        >>> # initialize with RagRetriever to do everything in one forward call
        >>> model = TFRagSequenceForGeneration.from_pretrained(
        ...     "facebook/rag-sequence-nq", retriever=retriever, from_pt=True
        ... )

        >>> input_dict = tokenizer.prepare_seq2seq_batch(
        ...     "How many people live in Paris?", "In Paris, there are 10 million people.", return_tensors="tf"
        ... )
        >>> outputs = model(input_dict, output_retrieved=True)

        >>> # or use retriever separately
        >>> # 1. Encode
        >>> input_ids = input_dict["input_ids"]
        >>> question_hidden_states = model.question_encoder(input_ids)[0]
        >>> # 2. Retrieve
        >>> docs_dict = retriever(input_ids.numpy(), question_hidden_states.numpy(), return_tensors="tf")
        >>> doc_scores = tf.squeeze(
        ...     tf.matmul(
        ...         tf.expand_dims(question_hidden_states, axis=1), docs_dict["retrieved_doc_embeds"], transpose_b=True
        ...     ),
        ...     axis=1,
        ... )
        >>> # 3. Forward to generator
        >>> outputs = model(
        ...     inputs=None,
        ...     context_input_ids=docs_dict["context_input_ids"],
        ...     context_attention_mask=docs_dict["context_attention_mask"],
        ...     doc_scores=doc_scores,
        ...     decoder_input_ids=input_dict["labels"],
        ... )

        >>> # or directly generate
        >>> generated = model.generate(
        ...     context_input_ids=docs_dict["context_input_ids"],
        ...     context_attention_mask=docs_dict["context_attention_mask"],
        ...     doc_scores=doc_scores,
        ... )
        >>> generated_string = tokenizer.batch_decode(generated, skip_special_tokens=True)
        ```"""

        assert (
            "decoder_cached_states" not in kwargs
        ), "Please use past_key_values to cache intermediate outputs"  # from modeling_tf_bart.py

        exclude_bos_score = exclude_bos_score if exclude_bos_score else self.config.exclude_bos_score
        reduce_loss = reduce_loss if reduce_loss else self.config.reduce_loss

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = labels
            use_cache = False

        outputs = self.rag(
            input_ids,
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            context_input_ids=context_input_ids,
            context_attention_mask=context_attention_mask,
            doc_scores=doc_scores,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_retrieved=output_retrieved,
            n_docs=n_docs,
            training=training,
        )

        loss = None
        if labels is not None:
            loss = self.get_nll(
                outputs.logits,
                outputs.doc_scores,
                labels,
                reduce_loss=reduce_loss,
                epsilon=self.config.label_smoothing,
                n_docs=n_docs,
            )

        return TFRetrievAugLMMarginOutput(
            loss=loss,
            logits=outputs.logits,
            doc_scores=outputs.doc_scores,
            past_key_values=outputs.past_key_values,
            context_input_ids=outputs.context_input_ids,
            context_attention_mask=outputs.context_attention_mask,
            retrieved_doc_embeds=outputs.retrieved_doc_embeds,
            retrieved_doc_ids=outputs.retrieved_doc_ids,
            question_encoder_last_hidden_state=outputs.question_encoder_last_hidden_state,
            question_enc_hidden_states=outputs.question_enc_hidden_states,
            question_enc_attentions=outputs.question_enc_attentions,
            generator_enc_last_hidden_state=outputs.generator_enc_last_hidden_state,
            generator_enc_hidden_states=outputs.generator_enc_hidden_states,
            generator_enc_attentions=outputs.generator_enc_attentions,
            generator_dec_hidden_states=outputs.generator_dec_hidden_states,
            generator_dec_attentions=outputs.generator_dec_attentions,
        )

    def get_nll(
        self, seq_logits, doc_scores, target, reduce_loss=False, epsilon=0.0, exclude_bos_score=False, n_docs=None
    ):
        # shift tokens left
        target = tf.concat(
            [target[:, 1:], tf.fill([target.shape[0], 1], tf.cast(self.config.generator.pad_token_id, target.dtype))],
            axis=1,
        )

        # bos_token_id is None for T5
        bos_token_id = self.config.bos_token_id or self.config.generator.bos_token_id
        n_docs = n_docs if n_docs is not None else self.config.n_docs
        equal_bos_token_id_all = tf.reduce_all(tf.equal(target[:, 0], bos_token_id))
        use_bos = bos_token_id is not None and equal_bos_token_id_all

        def _mask_pads(ll, smooth_obj):
            pad_mask = tf.equal(target, tf.cast(self.config.generator.pad_token_id, target.dtype))
            if tf.reduce_any(pad_mask):
                ll = tf.where(pad_mask, 0.0, ll)
                smooth_obj = tf.where(pad_mask, 0.0, smooth_obj)
            return tf.squeeze(ll, axis=-1), tf.squeeze(smooth_obj, axis=-1)

        # seq_logits.shape = (batch*n_docs, tgt_len , vocabs)
        seq_logprobs = tf.nn.log_softmax(seq_logits, axis=-1)
        seq_logprobs = tf.reshape(
            seq_logprobs, (seq_logits.shape[0] // n_docs, n_docs, -1, seq_logits.shape[-1])
        )  # (batch_size, n_docs, tgt_len, vocabs)
        doc_logprobs = tf.nn.log_softmax(doc_scores, axis=1)
        doc_logprobs = tf.expand_dims(doc_logprobs, axis=-1)
        doc_logprobs = tf.expand_dims(doc_logprobs, axis=-1)  # done twice to get 4-D

        # RAG-sequence marginalization
        first_token_scores = seq_logprobs[:, :, :1, :]
        second_token_scores = seq_logprobs[:, :, 1:2, :]
        remainder = seq_logprobs[:, :, 2:, :]
        rag_logprobs = tf.concat([first_token_scores, second_token_scores + doc_logprobs, remainder], axis=2)

        # calculate loss
        target = tf.expand_dims(target, axis=1)  # n_docs dimension
        target = tf.expand_dims(target, axis=-1)  # logits dimension
        target = tf.repeat(target, n_docs, axis=1)
        assert len(target.shape) == len(rag_logprobs.shape)

        # last-axis gathering only - use 2D-reshape-trick for Torch's style nD gathering
        def torch_gather(param, id_tensor):
            # 2d-gather torch equivalent: https://stackoverflow.com/questions/52129909/tensorflow-equivalent-of-torch-gather
            def gather2d(target, id_tensor):
                idx = tf.stack([tf.range(tf.shape(id_tensor)[0], dtype=id_tensor.dtype), id_tensor[:, 0]], axis=-1)
                result = tf.gather_nd(target, idx)
                return tf.expand_dims(result, axis=-1)

            target = tf.reshape(param, (-1, param.shape[-1]))  # reshape 2D
            target_shape = id_tensor.shape

            id_tensor = tf.reshape(id_tensor, (-1, 1))  # also 2D-index
            result = gather2d(target, id_tensor)
            return tf.reshape(result, target_shape)

        ll = torch_gather(rag_logprobs, id_tensor=target)
        smooth_obj = tf.reduce_sum(rag_logprobs, axis=-1, keepdims=True)  # total sum of all (normalised) logits

        ll, smooth_obj = _mask_pads(ll, smooth_obj)

        # sum over tokens, exclude bos while scoring
        if exclude_bos_score and use_bos:
            ll = tf.reduce_sum(ll[:, :, 1:], axis=2)
        else:
            ll = tf.reduce_sum(ll, axis=2)

        smooth_obj = tf.reduce_sum(smooth_obj, axis=2)
        ll = tf.math.reduce_logsumexp(ll, axis=1)  # logsumexp over docs
        smooth_obj = tf.math.reduce_logsumexp(smooth_obj, axis=1)

        nll_loss = -ll
        smooth_loss = -smooth_obj

        if reduce_loss:
            nll_loss = tf.reduce_sum(nll_loss)
            smooth_loss = tf.reduce_sum(smooth_loss)

        eps_i = epsilon / rag_logprobs.shape[-1]
        loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
        return loss

    def generate(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: tf.Tensor | None = None,
        context_input_ids=None,
        context_attention_mask=None,
        doc_scores=None,
        do_deduplication=None,  # defaults to True
        num_return_sequences=None,  # defaults to 1
        num_beams=None,  # defaults to 1
        n_docs=None,
        **model_kwargs,
    ):
        """
        Implements RAG sequence "thorough" decoding. Read the [`~generation.GenerationMixin.generate`]` documentation
        for more information on how to set other generate input parameters

        Args:
            input_ids (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                The sequence used as a prompt for the generation. If `input_ids` is not passed, then
                `context_input_ids` has to be provided.
            attention_mask (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`: - 1 for
                tokens that are **not masked**, - 0 for tokens that are **masked**. [What are attention
                masks?](../glossary#attention-mask)
            context_input_ids (`tf.Tensor` of shape `(batch_size * config.n_docs, config.max_combined_length)`, *optional*, returned when *output_retrieved=True*):
                Input IDs post-processed from the retrieved documents and the question encoder input_ids by the
                retriever.
            context_attention_mask (`tf.Tensor` of shape `(batch_size * config.n_docs, config.max_combined_length)`, *optional*, returned when *output_retrieved=True*):
                Attention mask post-processed from the retrieved documents and the question encoder `input_ids` by the
                retriever. If the model has is not initialized with a `retriever` or `input_ids` is not given,
                `context_input_ids` and `context_attention_mask` have to be provided to the forward pass. They are
                returned by [`~RagRetriever.__call__`].
            doc_scores (`tf.Tensor` of shape `(batch_size, config.n_docs)`):
                Score between each retrieved document embeddings (see `retrieved_doc_embeds`) and
                `question_encoder_last_hidden_state`. If the model has is not initialized with a `retriever` or
                `input_ids` is not given, `doc_scores` has to be provided to the forward pass. `doc_scores` are
                returned by [`~RagRetriever.__call__`].
            do_deduplication (`bool`, *optional*):
                Whether or not to deduplicate the generations from different context documents for a given input. Has
                to be set to `False` if used while training with distributed backend.
            num_return_sequences(`int`, *optional*, defaults to 1):
                The number of independently computed returned sequences for each element in the batch. Note that this
                is not the value we pass to the `generator`'s `[`~generation.GenerationMixin.generate`]` function,
                where we set `num_return_sequences` to `num_beams`.
            num_beams (`int`, *optional*, defaults to 1):
                Number of beams for beam search. 1 means no beam search.
            n_docs (`int`, *optional*, defaults to `config.n_docs`)
                Number of documents to retrieve and/or number of documents for which to generate an answer.
            kwargs (`Dict[str, Any]`, *optional*):
                Additional kwargs will be passed to [`~generation.GenerationMixin.generate`]

        Return:
            `tf.Tensor` of shape `(batch_size * num_return_sequences, sequence_length)`: The generated sequences. The
            second dimension (sequence length) is either equal to `max_length` or shorter if all batches finished early
            due to the `eos_token_id`.
        """

        n_docs = n_docs if n_docs is not None else self.config.n_docs
        do_deduplication = do_deduplication if do_deduplication is not None else self.config.do_deduplication
        num_doc_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
        )
        num_beams = num_beams if num_beams is not None else self.config.num_beams

        assert (
            input_ids is not None or context_input_ids is not None
        ), " At least one of input_ids or context_input_ids must be given"

        if self.retriever is not None and context_input_ids is None:
            question_hidden_states = self.question_encoder(input_ids, attention_mask=attention_mask)[0]
            context_input_ids = self.retriever(
                input_ids,
                question_hidden_states.numpy(),
                prefix=self.generator.config.prefix,
                n_docs=n_docs,
                return_tensors="tf",
            )["context_input_ids"]

        hypos = []
        model_kwargs["num_beams"] = num_beams
        model_kwargs["num_return_sequences"] = num_beams  # put here so that not confused with num_doc_return_sequences
        model_kwargs["attention_mask"] = None

        batch_size = input_ids.shape[0] if input_ids is not None else context_input_ids.shape[0] // n_docs

        for index in range(batch_size):
            # first, generate beams from documents:
            generator_input_ids = context_input_ids[index * n_docs : (index + 1) * n_docs]  # (n_docs, max_len)

            output_sequences = self.generator.generate(
                generator_input_ids,
                **model_kwargs,
            )  # n_docs * n_beam, tgt_len
            if do_deduplication:
                # do_deduplication -- for TF, work on Eager mode only!
                output_sequences = tf.stack(list({str(k.numpy().tolist()): k for k in output_sequences}.values()))

            num_candidates = output_sequences.shape[
                0
            ]  # after deduplication, this number can be less than n_docs*n_beam

            # then, run model forwards to get nll scores:
            if input_ids is not None:
                new_input_ids = tf.tile(input_ids[index : index + 1], (num_candidates, 1))
                outputs = self(new_input_ids, labels=output_sequences, exclude_bos_score=True)
            else:  # input_ids is None, need context_input_ids/mask and doc_scores
                assert context_attention_mask is not None, (
                    "Make sure that `context_attention_mask` are passed, if no `input_ids` is set. Alternatively, you"
                    " can set a retriever using the `set_retriever(...)` function."
                )
                assert doc_scores is not None, (
                    "Make sure that `doc_scores` are passed, if no `input_ids` is set. Alternatively, you can set a"
                    " retriever using the `set_retriever(...)` function."
                )

                individual_input_ids = tf.tile(
                    generator_input_ids, (num_candidates, 1)
                )  # (num_candidates*n_docs, max_len)

                individual_attention_mask = context_attention_mask[index * n_docs : (index + 1) * n_docs]
                individual_attention_mask = tf.tile(individual_attention_mask, (num_candidates, 1))

                individual_doc_scores = doc_scores[index : (index + 1), :]  # doc_scores.shape = [batch, n_docs]
                individual_doc_scores = tf.tile(individual_doc_scores, (num_candidates, 1))  # [num_candidates, n_docs]

                outputs = self(
                    input_ids=None,
                    context_input_ids=individual_input_ids,
                    context_attention_mask=individual_attention_mask,
                    doc_scores=individual_doc_scores,
                    labels=output_sequences,
                    exclude_bos_score=True,
                )

            top_cand_inds = tf.math.top_k((-outputs["loss"]), k=num_doc_return_sequences)[1]

            # add hypothesis
            hypos.append(tf.gather(output_sequences, top_cand_inds))

        return self._cat_and_pad(hypos, pad_token_id=self.config.generator.pad_token_id)

    @staticmethod
    def _cat_and_pad(tensors, pad_token_id):
        # used by generate(): tensors is a (batched) list of (candidates, len); len is varied across batch

        # Initialize padded tensor with shape ( all_candidates , max_candidate_length ),
        # where all_candidates counted from all inputs
        new_shape = sum([t.shape[0] for t in tensors]), max([t.shape[1] for t in tensors])
        output = tf.fill(new_shape, pad_token_id)

        # Normal tensor doesn't support slice assignment, so we need tf.Variable
        output = tf.Variable(output)

        # Assign, and then convert back to tensor
        ind = 0
        for t in tensors:
            output[ind : ind + t.shape[0], : t.shape[1]].assign(t)
            ind += t.shape[0]

        output = tf.convert_to_tensor(output)
        return tf.cast(output, tensors[0][0][0].dtype)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "rag", None) is not None:
            with tf.name_scope(self.rag.name):
                self.rag.build(None)


__all__ = ["TFRagModel", "TFRagPreTrainedModel", "TFRagSequenceForGeneration", "TFRagTokenForGeneration"]
