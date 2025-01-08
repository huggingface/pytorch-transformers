import os
from typing import Dict, List, Union

import tensorflow as tf
from keras_nlp.tokenizers import BytePairTokenizer
from tensorflow_text import pad_model_inputs

from ...modeling_tf_utils import keras
from .tokenization_gpt2 import GPT2Tokenizer


class TFGPT2Tokenizer(keras.layers.Layer):
    """
    This is an in-graph tokenizer for GPT2. It should be initialized similarly to other tokenizers, using the
    `from_pretrained()` method. It can also be initialized with the `from_tokenizer()` method, which imports settings
    from an existing standard tokenizer object.

    In-graph tokenizers, unlike other Hugging Face tokenizers, are actually Keras layers and are designed to be run
    when the model is called, rather than during preprocessing. As a result, they have somewhat more limited options
    than standard tokenizer classes. They are most useful when you want to create an end-to-end model that goes
    straight from `tf.string` inputs to outputs.

    Args:
        vocab (Dict[str, int]): Vocabulary dict for Byte Pair Tokenizer
        merges (List[str]): Merges list for Byte Pair Tokenizer
    """

    def __init__(self, vocab: Dict[str, int], merges: List[str], max_length: int = None, pad_token_id: int = None):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.max_length = max_length
        self.vocab = vocab
        self.merges = merges
        self.tf_tokenizer = BytePairTokenizer(vocab, merges, sequence_length=max_length)

    @classmethod
    def from_tokenizer(cls, tokenizer: GPT2Tokenizer, *args, **kwargs):
        """Creates TFGPT2Tokenizer from GPT2Tokenizer

        Args:
            tokenizer (GPT2Tokenizer)

        Examples:

        ```python
        from transformers import AutoTokenizer, TFGPT2Tokenizer

        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        tf_tokenizer = TFGPT2Tokenizer.from_tokenizer(tokenizer)
        ```
        """
        merges = [" ".join(m) for m in tokenizer.bpe_ranks.keys()]
        vocab = tokenizer.get_vocab()
        return cls(vocab, merges, *args, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], *init_inputs, **kwargs):
        """Creates TFGPT2Tokenizer from pretrained GPT2Tokenizer

        Args:
            pretrained_model_name_or_path (Union[str, os.PathLike]): Path to pretrained model

        Examples:

        ```python
        from transformers import TFGPT2Tokenizer

        tf_tokenizer = TFGPT2Tokenizer.from_pretrained("openai-community/gpt2")
        ```
        """
        tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path, *init_inputs, **kwargs)
        return cls.from_tokenizer(tokenizer, *init_inputs, **kwargs)

    @classmethod
    def from_config(cls, config):
        """Creates TFGPT2Tokenizer from configurations

        Args:
            config (Dict): Dictionary with keys such as stated in `get_config`.
        """
        return cls(**config)

    def get_config(self):
        return {
            "vocab": self.vocab,
            "merges": self.merges,
            "max_length": self.max_length,
            "pad_token_id": self.pad_token_id,
        }

    def call(self, x, max_length: int = None):
        input_ids = self.tf_tokenizer(x)
        attention_mask = tf.ones_like(input_ids)

        if self.pad_token_id is not None:
            # pad the tokens up to max length
            max_length = max_length if max_length is not None else self.max_length

            if max_length is not None:
                input_ids, attention_mask = pad_model_inputs(
                    input_ids, max_seq_length=max_length, pad_value=self.pad_token_id
                )

        return {"attention_mask": attention_mask, "input_ids": input_ids}


__all__ = ["TFGPT2Tokenizer"]
