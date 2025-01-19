# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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
from copy import deepcopy
from dataclasses import dataclass
from typing import ClassVar, List, Optional, Tuple, Union

from transformers.models.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor

from ...cache_utils import Cache
from ...configuration_utils import PretrainedConfig
from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput, is_valid_image
from ...modeling_outputs import BaseModelOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...models.auto import AutoModelForImageTextToText
from ...processing_utils import (
    ProcessingKwargs,
    Unpack,
)
from ...tokenization_utils_base import (
    PreTokenizedInput,
    TextInput,
)
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_torch_available,
    logging,
    replace_return_docstrings,
)
from ..auto import CONFIG_MAPPING, AutoConfig


if is_torch_available():
    import torch
    from torch import nn


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "ColQwen2Config"

COLQWEN2_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`ColQwen2Config`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


class ColQwen2Config(PretrainedConfig):
    r"""
    Configuration class to store the configuration of a [`ColQ2en2ForRetrieval`]. It is used to instantiate an instance
    of `ColQwen2ForRetrieval` according to the specified arguments, defining the model architecture following the methodology
    from the "ColQwen2: Efficient Document Retrieval with Vision Language Models" paper.

    Creating a configuration with the default settings will result in a configuration where the VLM backbone is set to the
    default PaliGemma configuration, i.e the one from [vidore/colpali-v1.2](https://huggingface.co/vidore/colpali-v1.2).

    The ColQwen2 config is very similar to [`Qwen2VLConfig`], but with an extra attribute defining the embedding dimension.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vlm_config (`PretrainedConfig`, *optional*):
            Configuration of the VLM backbone model.
        text_config (`PretrainedConfig`, *optional*):
            Configuration of the text backbone model. Overrides the `text_config` attribute of the `vlm_config` if provided.
        embedding_dim (`int`, *optional*, defaults to 128):
            Dimension of the multi-vector embeddings produced by the model.

    Example:

    ```python
    from transformers.models.colpali import ColQwen2Config, ColQwen2ForRetrieval

    config = ColQwen2Config()
    model = ColQwen2ForRetrieval(config)
    ```
    """

    model_type = "colqwen2"
    sub_configs = {"vlm_config": PretrainedConfig, "text_config": AutoConfig}

    def __init__(
        self,
        vlm_config=None,
        text_config=None,
        embedding_dim: int = 128,
        **kwargs,
    ):
        if vlm_config is None:
            vlm_config = CONFIG_MAPPING["qwen2_vl"]()
            logger.info(
                "`vlm_config` is `None`. Initializing `vlm_config` with the `Qwen2VLConfig` with default values."
            )
        elif isinstance(vlm_config, dict):
            vlm_config = deepcopy(vlm_config)
            if "model_type" not in vlm_config:
                raise KeyError(
                    "The `model_type` key is missing in the `vlm_config` dictionary. Please provide the model type."
                )
            elif vlm_config["model_type"] != "qwen2_vl":
                raise ValueError(
                    f"Invalid model type for `vlm_config`. Expected `qwen2_vl`, but got {vlm_config['model_type']}."
                )
            vlm_config = CONFIG_MAPPING[vlm_config["model_type"]](**vlm_config)
        elif isinstance(vlm_config, PretrainedConfig):
            vlm_config = vlm_config
        else:
            raise TypeError(
                f"Invalid type for `vlm_config`. Expected `PretrainedConfig`, `dict`, or `None`, but got {type(vlm_config)}."
            )

        self.vlm_config = vlm_config
        self.text_config = text_config = text_config if text_config is not None else vlm_config.text_config
        if isinstance(self.text_config, dict):
            text_config["model_type"] = text_config["model_type"] if "model_type" in text_config else "qwen2"
            self.text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)

        self.embedding_dim = embedding_dim

        super().__init__(**kwargs)


class ColQwen2ProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": "longest",
        },
        "images_kwargs": {
            "data_format": "channels_first",
            "do_convert_rgb": True,
        },
        "common_kwargs": {"return_tensors": "pt"},
    }


def round_by_factor(number: float, factor: int) -> int:
    """
    Returns the closest integer to 'number' that is divisible by 'factor'.

    Args:
        number (float): The number to round.
        factor (int): The factor to round to.
    """
    return round(number / factor) * factor


def ceil_by_factor(number: float, factor: int) -> int:
    """
    Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'.

    Args:
        number (float): The number to round.
        factor (int): The factor to round to.
    """
    return math.ceil(number / factor) * factor


def floor_by_factor(number: float, factor: int) -> int:
    """
    Returns the largest integer less than or equal to 'number' that is divisible by 'factor'.

    Args:
        number (float): The number to round.
        factor (int): The factor to round to.
    """
    return math.floor(number / factor) * factor


class ColQwen2Processor(Qwen2VLProcessor):
    r"""
    Constructs a ColQwen2 processor which wraps a Qwen2VLProcessor and special methods to process images and queries, as
    well as to compute the late-interaction retrieval score.

    [`ColQwen2Processor`] offers all the functionalities of [`Qwen2VLProcessor`]. See the [`~Qwen2VLProcessor.__call__`]
    for more information.

    Args:
        image_processor ([`SiglipImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`LlamaTokenizerFast`], *optional*):
            The tokenizer is a required input.
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
    """

    image_processor_class = "Qwen2VLImageProcessor"

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        visual_prompt_prefix: str = "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe the image.<|im_end|><|endoftext|>",
        query_prefix: str = "Query: ",
        num_image_tokens: int = 768,
        **kwargs,
    ):
        super().__init__(image_processor, tokenizer, chat_template=chat_template, **kwargs)

        self.visual_prompt_prefix = visual_prompt_prefix
        self.query_prefix = query_prefix

        self.tokenizer.padding_side = "left"

        self.min_pixels = 4 * 28 * 28
        self.max_pixels = num_image_tokens * 28 * 28
        self.factor = 28
        self.max_ratio = 200

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        audio=None,
        videos=None,
        **kwargs: Unpack[ColQwen2ProcessorKwargs],
    ) -> BatchFeature:
        """
        # TODO: Validate docstring

        Main method to prepare for the model either (1) one or several texts, either (2) one or several image(s). This method is custom
        wrapper around the Qwen2VLProcessor's [`~Qwen2VLProcessor.__call__`] method adapted for the ColQwen2 model. It cannot process
        both text and images at the same time.

        When preparing the the text(s), this method forwards the `text` and `kwargs` arguments to LlamaTokenizerFast's
        [`~LlamaTokenizerFast.__call__`].
        When preparing the the image(s), this method forwards the `images` and `kwargs` arguments to SiglipImageProcessor's
        [`~SiglipImageProcessor.__call__`].
        Please refer to the doctsring of the above two methods for more information.

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. In case of a NumPy array/PyTorch tensor, each image should be of shape (C, H, W), where C is a
                number of channels, H and W are image height and width.
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
        """
        kwargs["padding"] = "longest" if "padding" not in kwargs else kwargs["padding"]
        kwargs["return_tensors"] = "pt" if "return_tensors" not in kwargs else kwargs["return_tensors"]
        output_kwargs = self._merge_kwargs(
            ColQwen2ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        suffix = output_kwargs["text_kwargs"].pop("suffix", None)

        return_token_type_ids = True if suffix is not None else False

        if text is None and images is None:
            raise ValueError("Either text or images must be provided")
        if text is not None and images is not None:
            raise ValueError("Only one of text or images can be processed at a time")

        if images is not None:
            if is_valid_image(images):
                images = [images]
            elif isinstance(images, list) and is_valid_image(images[0]):
                pass
            elif not (isinstance(images, list) and isinstance(images[0], list) and is_valid_image(images[0][0])):
                raise ValueError("images must be an image, list of images or list of list of images")

            texts_doc = [self.visual_prompt_prefix] * len(images)
            images: List[ImageInput] = [self.smart_resize(image.convert("RGB")) for image in images]

            image_inputs = self.image_processor(images=images, videos=None, **output_kwargs["images_kwargs"])
            image_grid_thw = image_inputs["image_grid_thw"]

            if image_grid_thw is not None:
                merge_length = self.image_processor.merge_size**2
                index = 0
                for i in range(len(texts_doc)):
                    while self.image_token in texts_doc[i]:
                        texts_doc[i] = texts_doc[i].replace(
                            self.image_token, "<|placeholder|>" * (image_grid_thw[index].prod() // merge_length), 1
                        )
                        index += 1
                    texts_doc[i] = texts_doc[i].replace("<|placeholder|>", self.image_token)

            text_inputs = self.tokenizer(texts_doc, **output_kwargs["text_kwargs"])

            return_data = BatchFeature(data={**text_inputs, **image_inputs})

            # NOTE: Make sure the scatter operation in DDP is done correctly when training on multiple GPUs.
            offsets = return_data["image_grid_thw"][:, 1] * return_data["image_grid_thw"][:, 2]

            # Separate pixel_values for each image.
            pixel_values = torch.split(return_data["pixel_values"], offsets.tolist())

            # Pad pixel_values to the same sequence length to later stack it into a single tensor.
            max_length = max([len(pixel_value) for pixel_value in pixel_values])

            pixel_values = [
                torch.cat([pv, torch.zeros((max_length - len(pv), pv.shape[1]), dtype=pv.dtype, device=pv.device)])
                for pv in pixel_values
            ]
            return_data["pixel_values"] = torch.stack(pixel_values)

            if return_token_type_ids:
                labels = return_data["input_ids"].masked_fill(return_data["token_type_ids"] == 0, -100)
                return_data.update({"labels": labels})

            return return_data

        elif text is not None:
            if isinstance(text, str):
                text = [text]
            elif not (isinstance(text, list) and isinstance(text[0], str)):
                raise ValueError("Text must be a string or a list of strings")

            if suffix is None:
                suffix = self.query_augmentation_token * 10

            texts_query: List[str] = []

            for query in text:
                augmented_query = self.query_prefix + query + suffix
                texts_query.append(augmented_query)

            output_kwargs["text_kwargs"]["max_length"] = output_kwargs["text_kwargs"].get("max_length", 50)

            batch_query = self.tokenizer(
                texts_query,
                return_token_type_ids=False,
                **output_kwargs["text_kwargs"],
            )

            return batch_query

    @property
    def query_augmentation_token(self) -> str:
        """
        Return the query augmentation token.

        Query augmentation buffers are used as reasoning buffers during inference.
        """
        return self.tokenizer.pad_token

    @staticmethod
    def smart_resize_helper(
        width: int,
        height: int,
        factor: int,
        max_ratio: int,
        min_pixels: int,
        max_pixels: int,
    ) -> Tuple[int, int]:
        """
        Returns the image size so that the following conditions are met:
        1. Both dimensions (height and width) are divisible by 'factor'.
        2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
        3. The aspect ratio of the image is maintained as closely as possible.

        Args:
            width (int): The width of the image.
            height (int): The height of the image.
            factor (int): The factor to round to.
            max_ratio (int): The maximum aspect ratio allowed.
            min_pixels (int): The minimum number of pixels allowed.
            max_pixels (int): The maximum number of pixels allowed.
        """

        if max(height, width) / min(height, width) > max_ratio:
            raise ValueError(
                f"Absolute aspect ratio must be smaller than {max_ratio}, "
                f"got {max(height, width) / min(height, width)}"
            )

        height_new = max(factor, round_by_factor(height, factor))
        width_new = max(factor, round_by_factor(width, factor))

        if height_new * width_new > max_pixels:
            beta = math.sqrt((height * width) / max_pixels)
            height_new = floor_by_factor(height / beta, factor)
            width_new = floor_by_factor(width / beta, factor)
        elif height_new * width_new < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            height_new = ceil_by_factor(height * beta, factor)
            width_new = ceil_by_factor(width * beta, factor)
        else:
            pass

        return height_new, width_new

    def smart_resize(self, image: "PIL.Image.Image") -> "PIL.Image.Image":
        """
        Resize and convert the image to the required format.
        """
        image_size = image.size
        new_height, new_width = self.smart_resize_helper(
            width=image_size[0],
            height=image_size[1],
            factor=self.factor,
            max_ratio=self.max_ratio,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        return image.resize((new_width, new_height))

    def process_images(
        self,
        images: ImageInput = None,
        **kwargs: Unpack[ColQwen2ProcessorKwargs],
    ) -> BatchFeature:
        """
        Prepare for the model one or several image(s). This method is a wrapper around the `__call__` method of the ColQwen2Processor's
        [`ColQwen2Processor.__call__`].

        This method forwards the `images` and `kwargs` arguments to SiglipImageProcessor's [`~SiglipImageProcessor.__call__`].

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. In case of a NumPy array/PyTorch tensor, each image should be of shape (C, H, W), where C is a
                number of channels, H and W are image height and width.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
        """
        return self.__call__(images=images, **kwargs)

    def process_queries(
        self,
        text: Union[TextInput, List[TextInput]],
        **kwargs: Unpack[ColQwen2ProcessorKwargs],
    ) -> BatchFeature:
        """
        Prepare for the model one or several texts. This method is a wrapper around the `__call__` method of the ColQwen2Processor's
        [`ColQwen2Processor.__call__`].

        This method forwards the `text` and `kwargs` arguments to LlamaTokenizerFast's [`~LlamaTokenizerFast.__call__`].

        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
        """
        return self.__call__(text=text, **kwargs)

    def score_retrieval(
        self,
        query_embeddings: Union["torch.Tensor", List["torch.Tensor"]],
        passage_embeddings: Union["torch.Tensor", List["torch.Tensor"]],
        batch_size: int = 128,
        output_dtype: Optional["torch.dtype"] = None,
        output_device: Union["torch.device", str] = "cpu",
    ) -> "torch.Tensor":
        """
        Compute the late-interaction/MaxSim score (ColBERT-like) for the given multi-vector
        query embeddings (`qs`) and passage embeddings (`ps`). For ColQwen2, a passage is the
        image of a document page.

        Because the embedding tensors are multi-vector and can thus have different shapes, they
        should be fed as:
        (1) a list of tensors, where the i-th tensor is of shape (sequence_length_i, embedding_dim)
        (2) a single tensor of shape (n_passages, max_sequence_length, embedding_dim) -> usually
            obtained by padding the list of tensors.

        Args:
            query_embeddings (`Union[torch.Tensor, List[torch.Tensor]`): Query embeddings.
            passage_embeddings (`Union[torch.Tensor, List[torch.Tensor]`): Passage embeddings.
            batch_size (`int`, *optional*, defaults to 128): Batch size for computing scores.
            output_dtype (`torch.dtype`, *optional*, defaults to `torch.float32`): The dtype of the output tensor.
                If `None`, the dtype of the input embeddings is used.
            output_device (`torch.device` or `str`, *optional*, defaults to "cpu"): The device of the output tensor.

        Returns:
            `torch.Tensor`: A tensor of shape `(n_queries, n_passages)` containing the scores. The score
            tensor is saved on the "cpu" device.
        """

        if len(query_embeddings) == 0:
            raise ValueError("No queries provided")
        if len(passage_embeddings) == 0:
            raise ValueError("No passages provided")

        if query_embeddings[0].device != passage_embeddings[0].device:
            raise ValueError("Queries and passages must be on the same device")

        if query_embeddings[0].dtype != passage_embeddings[0].dtype:
            raise ValueError("Queries and passages must have the same dtype")

        if output_dtype is None:
            output_dtype = query_embeddings[0].dtype

        scores: List[torch.Tensor] = []

        for i in range(0, len(query_embeddings), batch_size):
            batch_scores: List[torch.Tensor] = []
            batch_queries = torch.nn.utils.rnn.pad_sequence(
                query_embeddings[i : i + batch_size], batch_first=True, padding_value=0
            )
            for j in range(0, len(passage_embeddings), batch_size):
                batch_passages = torch.nn.utils.rnn.pad_sequence(
                    passage_embeddings[j : j + batch_size], batch_first=True, padding_value=0
                )
                batch_scores.append(
                    torch.einsum("bnd,csd->bcns", batch_queries, batch_passages).max(dim=3)[0].sum(dim=2)
                )
            scores.append(torch.cat(batch_scores, dim=1).to(output_dtype).to(output_device))

        return torch.cat(scores, dim=0)


@add_start_docstrings(
    "The bare ColQwen2 model outputting raw hidden-states without any specific head on top.",
    COLQWEN2_START_DOCSTRING,
)
class ColQwen2PreTrainedModel(PreTrainedModel):
    config_class = ColQwen2Config
    base_model_prefix = "model"
    _no_split_modules = []

    def _init_weights(self, module):
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.vlm_config.text_config.initializer_range
        )

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


@dataclass
class ColQwen2ForRetrievalOutput(BaseModelOutputWithPast):
    """
    Base class for ColQwen2 embeddings output.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        embeddings (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            The embeddings of the model.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        image_hidden_states (`torch.FloatTensor`, *optional*):
            A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
            image_hidden_states of the model produced by the vision encoder after projecting last hidden state.
    """

    loss: Optional[torch.FloatTensor] = None
    embeddings: torch.Tensor = None
    past_key_values: Optional[Union[List[torch.FloatTensor], Cache]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[torch.FloatTensor] = None


COLQWEN2_FOR_RETRIEVAL_INPUT_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.
            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.
            [What are input IDs?](../glossary#input-ids)
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)):
            The tensors corresponding to the input images. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`SiglipImageProcessor.__call__`] for details ([]`PaliGemmaProcessor`] uses
            [`SiglipImageProcessor`] for processing images). If none, ColQwen2 will only process text (query embeddings).
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)
            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.
            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).
            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.
            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        kwargs (`Dict[str, Any]`, *optional*):
            Additional key word arguments passed along to the vlm backbone model.
"""


@add_start_docstrings(
    """
    # TODO: Update
    ColQwen2 leverages Vision Language Models (VLMs) to construct efficient multi-vector embeddings in the visual space for document retrieval.
    By feeding the ViT output patches from Qwen2-VL to a linear projection, we create a multi-vector representation of documents. The model
    is trained to maximize the similarity between these document embeddings and the query embeddings, following the ColBERT method.

    Using ColQwen2 removes the need for potentially complex and brittle layout recognition and OCR pipelines with a single model that can take into account
    both the textual and visual content (layout, charts, ...) of a document.

    ColQwen2 is part of the ColVision model family, which was introduced with ColPali in the following paper: [*ColPali: Efficient Document Retrieval with
    Vision Language Models*](https://arxiv.org/abs/2407.01449).

    Resources:
    - A blog post detailing ColPali, a vision retrieval model, can be found [here](https://huggingface.co/blog/manu/colpali). 📝
    - The code for using and training the original ColQwen2 model and for the `colpali-engine` package can be found [here](https://github.com/illuin-tech/colpali). 🌎
    - Cookbooks for learning to use the Hf version of ColQwen2, fine-tuning, and similarity maps generation can be found [here](https://github.com/tonywu71/colpali-cookbooks). 📚
    """
)
class ColQwen2ForRetrieval(PreTrainedModel):
    main_input_name: ClassVar[str] = "input_ids"

    def __init__(self, config: ColQwen2Config):
        super().__init__(config)
        self.config = config
        self.vocab_size = config.vlm_config.text_config.vocab_size

        vlm = AutoModelForImageTextToText.from_config(config.vlm_config)
        if vlm.language_model._tied_weights_keys is not None:
            self._tied_weights_keys = [f"vlm.language_model.{k}" for k in vlm.language_model._tied_weights_keys]
        self.vlm = vlm

        self.embedding_dim = self.config.embedding_dim
        self.embedding_proj_layer = nn.Linear(
            self.config.vlm_config.text_config.hidden_size,
            self.embedding_dim,
        )

        self.post_init()

    def inner_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, ModelOutput]:
        """
        Forward through the Qwen2VL VLM backbone. Uses a custom forward method to fix gradient flow when using
        training with multiple GPUs.
        """
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)

            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.get_dtype())
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                video_mask = (input_ids == self.config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        outputs = self.vlm(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return outputs

    @add_start_docstrings_to_model_forward(COLQWEN2_FOR_RETRIEVAL_INPUT_DOCSTRING)
    @replace_return_docstrings(output_type=ColQwen2ForRetrievalOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        *args,
        **kwargs,
    ) -> Union[Tuple, ColQwen2ForRetrievalOutput]:  # noqa: F821
        r"""
        # TODO: Update docstring
        Returns:
        ```"""
        if "pixel_values" in kwargs:
            kwargs["pixel_values"] = kwargs["pixel_values"].to(dtype=self.dtype)
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        position_ids, rope_deltas = self.get_rope_index(
            input_ids=kwargs["input_ids"],
            image_grid_thw=kwargs.get("image_grid_thw", None),
            video_grid_thw=None,
            attention_mask=kwargs.get("attention_mask", None),
        )

        outputs = self.inner_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
            *args,
            **kwargs,
        )

        last_hidden_states = outputs[0]  # (batch_size, sequence_length, hidden_size)
        embeddings = self.embedding_proj_layer(last_hidden_states)  # (batch_size, sequence_length, dim)

        # L2 normalization
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)  # (batch_size, sequence_length, dim)

        embeddings = embeddings * attention_mask.unsqueeze(-1)  # (batch_size, sequence_length, dim)

        loss = None
        if not return_dict:
            output = (embeddings,) + outputs[2:]
            output[2] = output[2] if output_hidden_states is not None else None
            output[-1] = (outputs.image_hidden_states if pixel_values is not None else None,)
            return (loss,) + output if loss is not None else output

        return ColQwen2ForRetrievalOutput(
            loss=loss,
            embeddings=embeddings,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states if pixel_values is not None else None,
        )


__all__ = [
    "ColQwen2ForRetrieval",
    "ColQwen2PreTrainedModel",
    "ColQwen2Processor",
]
