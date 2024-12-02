from typing import Dict

import numpy as np

from ..utils import add_end_docstrings, is_tf_available, is_torch_available, logging
from .base import GenericTensor, Pipeline, PipelineException, build_pipeline_init_args


if is_tf_available():
    import tensorflow as tf

    from ..tf_utils import stable_softmax


if is_torch_available():
    import torch


logger = logging.get_logger(__name__)


@add_end_docstrings(
    build_pipeline_init_args(has_tokenizer=True),
    r"""
        top_k (`int`, *optional*, defaults to 5):
            The number of predictions to return.
        targets (`str` or `List[str]`, *optional*):
            When passed, the model will limit the scores to the passed targets instead of looking up in the whole
            vocab. If the provided targets are not in the model vocab, they will be tokenized and the first resulting
            token will be used (with a warning, and that might be slower).
        tokenizer_kwargs (`dict`, *optional*):
            Additional dictionary of keyword arguments passed along to the tokenizer.""",
)
class FillMaskPipeline(Pipeline):
    """
    Masked language modeling prediction pipeline using any `ModelWithLMHead`. See the [masked language modeling
    examples](../task_summary#masked-language-modeling) for more information.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> fill_masker = pipeline(model="google-bert/bert-base-uncased")
    >>> fill_masker("This is a simple [MASK].")
    [{'score': 0.042, 'token': 3291, 'token_str': 'problem', 'sequence': 'this is a simple problem.'}, {'score': 0.031, 'token': 3160, 'token_str': 'question', 'sequence': 'this is a simple question.'}, {'score': 0.03, 'token': 8522, 'token_str': 'equation', 'sequence': 'this is a simple equation.'}, {'score': 0.027, 'token': 2028, 'token_str': 'one', 'sequence': 'this is a simple one.'}, {'score': 0.024, 'token': 3627, 'token_str': 'rule', 'sequence': 'this is a simple rule.'}]
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This mask filling pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"fill-mask"`.

    The models that this pipeline can use are models that have been trained with a masked language modeling objective,
    which includes the bi-directional models in the library. See the up-to-date list of available models on
    [huggingface.co/models](https://huggingface.co/models?filter=fill-mask).

    <Tip>

    This pipeline only works for inputs with exactly one token masked. Experimental: We added support for multiple
    masks. The returned values are raw model output, and correspond to disjoint probabilities where one might expect
    joint probabilities (See [discussion](https://github.com/huggingface/transformers/pull/10222)).

    </Tip>

    <Tip>

    This pipeline now supports tokenizer_kwargs. For example try:

    ```python
    >>> from transformers import pipeline

    >>> fill_masker = pipeline(model="google-bert/bert-base-uncased")
    >>> tokenizer_kwargs = {"truncation": True}
    >>> fill_masker(
    ...     "This is a simple [MASK]. " + "...with a large amount of repeated text appended. " * 100,
    ...     tokenizer_kwargs=tokenizer_kwargs,
    ... )
    ```


    </Tip>


    """

    def get_masked_index(self, input_ids: GenericTensor) -> np.ndarray:
        if self.framework == "tf":
            masked_index = tf.where(input_ids == self.tokenizer.mask_token_id).numpy()
        elif self.framework == "pt":
            masked_index = torch.nonzero(input_ids == self.tokenizer.mask_token_id, as_tuple=False)
        else:
            raise ValueError("Unsupported framework")
        return masked_index

    def _ensure_exactly_one_mask_token(self, input_ids: GenericTensor) -> np.ndarray:
        masked_index = self.get_masked_index(input_ids)
        numel = np.prod(masked_index.shape)
        if numel < 1:
            raise PipelineException(
                "fill-mask",
                self.model.base_model_prefix,
                f"No mask_token ({self.tokenizer.mask_token}) found on the input",
            )

    def ensure_exactly_one_mask_token(self, model_inputs: GenericTensor):
        if isinstance(model_inputs, list):
            for model_input in model_inputs:
                self._ensure_exactly_one_mask_token(model_input["input_ids"][0])
        else:
            for input_ids in model_inputs["input_ids"]:
                self._ensure_exactly_one_mask_token(input_ids)

    def preprocess(
        self, inputs, return_tensors=None, tokenizer_kwargs=None, **preprocess_parameters
    ) -> Dict[str, GenericTensor]:
        if return_tensors is None:
            return_tensors = self.framework
        if tokenizer_kwargs is None:
            tokenizer_kwargs = {}

        model_inputs = self.tokenizer(inputs, return_tensors=return_tensors, **tokenizer_kwargs)
        self.ensure_exactly_one_mask_token(model_inputs)
        return model_inputs

    def _forward(self, model_inputs):
        model_outputs = self.model(**model_inputs)
        model_outputs["input_ids"] = model_inputs["input_ids"]
        return model_outputs

    def postprocess(self, model_outputs, top_k=5, target_ids=None):
        # Cap top_k if there are targets
        if target_ids is not None and target_ids.shape[0] < top_k:
            top_k = target_ids.shape[0]
        input_ids = model_outputs["input_ids"][0]
        outputs = model_outputs["logits"]

        if self.framework == "tf":
            masked_index = tf.where(input_ids == self.tokenizer.mask_token_id).numpy()[:, 0]

            outputs = outputs.numpy()

            logits = outputs[0, masked_index, :]
            probs = stable_softmax(logits, axis=-1)
            if target_ids is not None:
                probs = tf.gather_nd(tf.squeeze(probs, 0), target_ids.reshape(-1, 1))
                probs = tf.expand_dims(probs, 0)

            topk = tf.math.top_k(probs, k=top_k)
            values, predictions = topk.values.numpy(), topk.indices.numpy()
        else:
            masked_index = torch.nonzero(input_ids == self.tokenizer.mask_token_id, as_tuple=False).squeeze(-1)
            # Fill mask pipeline supports only one ${mask_token} per sample

            logits = outputs[0, masked_index, :]
            probs = logits.softmax(dim=-1)
            if target_ids is not None:
                probs = probs[..., target_ids]

            values, predictions = probs.topk(top_k)

        result = []
        single_mask = values.shape[0] == 1
        for i, (_values, _predictions) in enumerate(zip(values.tolist(), predictions.tolist())):
            row = []
            for v, p in zip(_values, _predictions):
                # Copy is important since we're going to modify this array in place
                tokens = input_ids.numpy().copy()
                if target_ids is not None:
                    p = target_ids[p].tolist()

                tokens[masked_index[i]] = p
                # Filter padding out:
                tokens = tokens[np.where(tokens != self.tokenizer.pad_token_id)]
                # Originally we skip special tokens to give readable output.
                # For multi masks though, the other [MASK] would be removed otherwise
                # making the output look odd, so we add them back
                sequence = self.tokenizer.decode(tokens, skip_special_tokens=single_mask)
                proposition = {"score": v, "token": p, "token_str": self.tokenizer.decode([p]), "sequence": sequence}
                row.append(proposition)
            result.append(row)
        if single_mask:
            return result[0]
        return result

    def get_target_ids(self, targets, top_k=None):
        if isinstance(targets, str):
            targets = [targets]
        try:
            vocab = self.tokenizer.get_vocab()
        except Exception:
            vocab = {}
        target_ids = []
        for target in targets:
            id_ = vocab.get(target, None)
            if id_ is None:
                input_ids = self.tokenizer(
                    target,
                    add_special_tokens=False,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                    max_length=1,
                    truncation=True,
                )["input_ids"]
                if len(input_ids) == 0:
                    logger.warning(
                        f"The specified target token `{target}` does not exist in the model vocabulary. "
                        "We cannot replace it with anything meaningful, ignoring it"
                    )
                    continue
                id_ = input_ids[0]
                # XXX: If users encounter this pass
                # it becomes pretty slow, so let's make sure
                # The warning enables them to fix the input to
                # get faster performance.
                logger.warning(
                    f"The specified target token `{target}` does not exist in the model vocabulary. "
                    f"Replacing with `{self.tokenizer.convert_ids_to_tokens(id_)}`."
                )
            target_ids.append(id_)
        target_ids = list(set(target_ids))
        if len(target_ids) == 0:
            raise ValueError("At least one target must be provided when passed.")
        target_ids = np.array(target_ids)
        return target_ids

    def _sanitize_parameters(self, top_k=None, targets=None, tokenizer_kwargs=None):
        preprocess_params = {}

        if tokenizer_kwargs is not None:
            preprocess_params["tokenizer_kwargs"] = tokenizer_kwargs

        postprocess_params = {}

        if targets is not None:
            target_ids = self.get_target_ids(targets, top_k)
            postprocess_params["target_ids"] = target_ids

        if top_k is not None:
            postprocess_params["top_k"] = top_k

        if self.tokenizer.mask_token_id is None:
            raise PipelineException(
                "fill-mask", self.model.base_model_prefix, "The tokenizer does not define a `mask_token`."
            )
        return preprocess_params, {}, postprocess_params

    def __call__(self, inputs, **kwargs):
        """
        Fill the masked token in the text(s) given as inputs.

        Args:
            inputs (`str` or `List[str]`):
                One or several texts (or one list of prompts) with masked tokens.
            targets (`str` or `List[str]`, *optional*):
                When passed, the model will limit the scores to the passed targets instead of looking up in the whole
                vocab. If the provided targets are not in the model vocab, they will be tokenized and the first
                resulting token will be used (with a warning, and that might be slower).
            top_k (`int`, *optional*):
                When passed, overrides the number of predictions to return.

        Return:
            A list or a list of list of `dict`: Each result comes as list of dictionaries with the following keys:

            - **sequence** (`str`) -- The corresponding input with the mask token prediction.
            - **score** (`float`) -- The corresponding probability.
            - **token** (`int`) -- The predicted token id (to replace the masked one).
            - **token_str** (`str`) -- The predicted token (to replace the masked one).
        """
        outputs = super().__call__(inputs, **kwargs)
        if isinstance(inputs, list) and len(inputs) == 1:
            return outputs[0]
        return outputs
