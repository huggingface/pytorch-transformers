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
"""RAG model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import add_start_docstrings


RAG_CONFIG_DOC = r"""
    [`RagConfig`] stores the configuration of a *RagModel*. Configuration objects inherit from [`PretrainedConfig`] and
    can be used to control the model outputs. Read the documentation from [`PretrainedConfig`] for more information.

    Args:
        title_sep (`str`, *optional*, defaults to  `" / "`):
            Separator inserted between the title and the text of the retrieved document when calling [`RagRetriever`].
        doc_sep (`str`, *optional*, defaults to  `" // "`):
            Separator inserted between the text of the retrieved document and the original input when calling
            [`RagRetriever`].
        n_docs (`int`, *optional*, defaults to 5):
            Number of documents to retrieve.
        max_combined_length (`int`, *optional*, defaults to 300):
            Max length of contextualized input returned by [`~RagRetriever.__call__`].
        retrieval_vector_size (`int`, *optional*, defaults to 768):
            Dimensionality of the document embeddings indexed by [`RagRetriever`].
        retrieval_batch_size (`int`, *optional*, defaults to 8):
            Retrieval batch size, defined as the number of queries issues concurrently to the faiss index encapsulated
            [`RagRetriever`].
        dataset (`str`, *optional*, defaults to `"wiki_dpr"`):
            A dataset identifier of the indexed dataset in HuggingFace Datasets (list all available datasets and ids
            using `datasets.list_datasets()`).
        dataset_split (`str`, *optional*, defaults to `"train"`)
            Which split of the `dataset` to load.
        index_name (`str`, *optional*, defaults to `"compressed"`)
            The index name of the index associated with the `dataset`. One can choose between `"legacy"`, `"exact"` and
            `"compressed"`.
        index_path (`str`, *optional*)
            The path to the serialized faiss index on disk.
        passages_path (`str`, *optional*):
            A path to text passages compatible with the faiss index. Required if using
            [`~models.rag.retrieval_rag.LegacyIndex`]
        use_dummy_dataset (`bool`, *optional*, defaults to `False`)
            Whether to load a "dummy" variant of the dataset specified by `dataset`.
        label_smoothing (`float`, *optional*, defaults to 0.0):
            Only relevant if `return_loss` is set to `True`. Controls the `epsilon` parameter value for label smoothing
            in the loss calculation. If set to 0, no label smoothing is performed.
        do_marginalize (`bool`, *optional*, defaults to `False`):
            If `True`, the logits are marginalized over all documents by making use of
            `torch.nn.functional.log_softmax`.
        reduce_loss (`bool`, *optional*, defaults to `False`):
            Whether or not to reduce the NLL loss using the `torch.Tensor.sum` operation.
        do_deduplication (`bool`, *optional*, defaults to `True`):
            Whether or not to deduplicate the generations from different context documents for a given input. Has to be
            set to `False` if used while training with distributed backend.
        exclude_bos_score (`bool`, *optional*, defaults to `False`):
            Whether or not to disregard the BOS token when computing the loss.
        output_retrieved(`bool`, *optional*, defaults to `False`):
            If set to `True`, `retrieved_doc_embeds`, `retrieved_doc_ids`, `context_input_ids` and
            `context_attention_mask` are returned. See returned tensors for more detail.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        forced_eos_token_id (`int`, *optional*):
            The id of the token to force as the last generated token when `max_length` is reached. Usually set to
            `eos_token_id`.
"""


@add_start_docstrings(RAG_CONFIG_DOC)
class RagConfig(PretrainedConfig):
    model_type = "rag"
    is_composition = True

    def __init__(
        self,
        vocab_size=None,
        is_encoder_decoder=True,
        prefix=None,
        bos_token_id=None,
        pad_token_id=None,
        eos_token_id=None,
        decoder_start_token_id=None,
        title_sep=" / ",
        doc_sep=" // ",
        n_docs=5,
        max_combined_length=300,
        retrieval_vector_size=768,
        retrieval_batch_size=8,
        dataset="wiki_dpr",
        dataset_split="train",
        index_name="compressed",
        index_path=None,
        passages_path=None,
        use_dummy_dataset=False,
        reduce_loss=False,
        label_smoothing=0.0,
        do_deduplication=True,
        exclude_bos_score=False,
        do_marginalize=False,
        output_retrieved=False,
        use_cache=True,
        forced_eos_token_id=None,
        dataset_revision=None,
        **kwargs,
    ):
        super().__init__(
            bos_token_id=bos_token_id,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            decoder_start_token_id=decoder_start_token_id,
            forced_eos_token_id=forced_eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            prefix=prefix,
            vocab_size=vocab_size,
            **kwargs,
        )
        if "question_encoder" not in kwargs or "generator" not in kwargs:
            raise ValueError(
                f"A configuraton of type {self.model_type} cannot be instantiated because "
                f"both `question_encoder` and `generator` sub-configurations were not passed, only {kwargs}"
            )
        question_encoder_config = kwargs.pop("question_encoder")
        question_encoder_model_type = question_encoder_config.pop("model_type")
        decoder_config = kwargs.pop("generator")
        decoder_model_type = decoder_config.pop("model_type")

        from ..auto.configuration_auto import AutoConfig

        self.question_encoder = AutoConfig.for_model(question_encoder_model_type, **question_encoder_config)
        self.generator = AutoConfig.for_model(decoder_model_type, **decoder_config)

        self.reduce_loss = reduce_loss
        self.label_smoothing = label_smoothing
        self.exclude_bos_score = exclude_bos_score
        self.do_marginalize = do_marginalize

        self.title_sep = title_sep
        self.doc_sep = doc_sep
        self.n_docs = n_docs
        self.max_combined_length = max_combined_length

        self.dataset = dataset
        self.dataset_split = dataset_split
        self.index_name = index_name

        self.retrieval_vector_size = retrieval_vector_size
        self.retrieval_batch_size = retrieval_batch_size
        self.passages_path = passages_path
        self.index_path = index_path
        self.use_dummy_dataset = use_dummy_dataset
        self.dataset_revision = dataset_revision

        self.output_retrieved = output_retrieved

        self.do_deduplication = do_deduplication

        self.use_cache = use_cache

        if self.forced_eos_token_id is None:
            self.forced_eos_token_id = getattr(self.generator, "forced_eos_token_id", None)

    @classmethod
    def from_question_encoder_generator_configs(
        cls, question_encoder_config: PretrainedConfig, generator_config: PretrainedConfig, **kwargs
    ) -> PretrainedConfig:
        r"""
        Instantiate a [`EncoderDecoderConfig`] (or a derived class) from a pre-trained encoder model configuration and
        decoder model configuration.

        Returns:
            [`EncoderDecoderConfig`]: An instance of a configuration object
        """
        return cls(question_encoder=question_encoder_config.to_dict(), generator=generator_config.to_dict(), **kwargs)


__all__ = ["RagConfig"]
