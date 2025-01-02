# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...modeling_outputs import ImageClassifierOutput, ModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
    add_start_docstrings_to_model_forward,
    is_timm_available,
    replace_return_docstrings,
    requires_backends,
)
from .configuration_timm_wrapper import TimmWrapperConfig


if is_timm_available():
    import timm


@dataclass
class TimmWrapperModelOutput(ModelOutput):
    """
    Output class for models TimmWrapperModel, containing the last hidden states, an optional pooled output,
    and optional hidden states.

    Args:
        last_hidden_state (`torch.FloatTensor`):
            The last hidden state of the model, output before applying the classification head.
        pooler_output (`torch.FloatTensor`, *optional*):
            The pooled output derived from the last hidden state, if applicable.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            A tuple containing the intermediate hidden states of the model at the output of each layer or specified layers.
            Returned if `output_hidden_states=True` is set or if `config.output_hidden_states=True`.
        attentions (`tuple(torch.FloatTensor)`, *optional*):
            A tuple containing the intermediate attention weights of the model at the output of each layer.
            Returned if `output_attentions=True` is set or if `config.output_attentions=True`.
            Note: Currently, Timm models do not support attentions output.
    """

    last_hidden_state: torch.FloatTensor
    pooler_output: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


TIMM_WRAPPER_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See [`TimmWrapperImageProcessor.preprocess`]
            for details.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. Not compatible with timm wrapped models.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. Not compatible with timm wrapped models.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        **kwargs:
            Additional keyword arguments passed along to the `timm` model forward.
"""


class TimmWrapperPreTrainedModel(PreTrainedModel):
    main_input_name = "pixel_values"
    config_class = TimmWrapperConfig
    _no_split_modules = []

    # used in Trainer to avoid passing `loss_kwargs` to model forward
    accepts_loss_kwargs = False

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["vision", "timm"])
        super().__init__(*args, **kwargs)

    @staticmethod
    def _fix_state_dict_key_on_load(key):
        """
        Overrides original method that renames `gamma` and `beta` to `weight` and `bias`.
        We don't want this behavior for timm wrapped models. Instead, this method adds a
        "timm_model." prefix to enable loading official timm Hub checkpoints.
        """
        if "timm_model." not in key:
            return f"timm_model.{key}"
        return key

    def _fix_state_dict_key_on_save(self, key):
        """
        Overrides original method to remove "timm_model." prefix from state_dict keys.
        Makes the saved checkpoint compatible with the `timm` library.
        """
        return key.replace("timm_model.", "")

    def load_state_dict(self, state_dict, *args, **kwargs):
        """
        Override original method to fix state_dict keys on load for cases when weights are loaded
        without using the `from_pretrained` method (e.g., in Trainer to resume from checkpoint).
        """
        state_dict = self._fix_state_dict_keys_on_load(state_dict)
        return super().load_state_dict(state_dict, *args, **kwargs)

    def _init_weights(self, module):
        """
        Initialize weights function to properly initialize Linear layer weights.
        Since model architectures may vary, we assume only the classifier requires
        initialization, while all other weights should be loaded from the checkpoint.
        """
        if isinstance(module, (nn.Linear)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()


class TimmWrapperModel(TimmWrapperPreTrainedModel):
    """
    Wrapper class for timm models to be used in transformers.
    """

    def __init__(self, config: TimmWrapperConfig):
        super().__init__(config)
        # using num_classes=0 to avoid creating classification head
        self.timm_model = timm.create_model(config.architecture, pretrained=False, num_classes=0)
        self.post_init()

    @add_start_docstrings_to_model_forward(TIMM_WRAPPER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TimmWrapperModelOutput, config_class=TimmWrapperConfig)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[Union[bool, List[int]]] = None,
        return_dict: Optional[bool] = None,
        do_pooling: Optional[bool] = None,
        **kwargs,
    ) -> Union[TimmWrapperModelOutput, Tuple[Tensor, ...]]:
        r"""
        do_pooling (`bool`, *optional*):
            Whether to do pooling for the last_hidden_state in `TimmWrapperModel` or not. If `None` is passed, the
            `do_pooling` value from the config is used.

        Returns:

        Examples:
        ```python
        >>> import torch
        >>> from PIL import Image
        >>> from urllib.request import urlopen
        >>> from transformers import AutoModel, AutoImageProcessor

        >>> # Load image
        >>> image = Image.open(urlopen(
        ...     'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
        ... ))

        >>> # Load model and image processor
        >>> checkpoint = "timm/resnet50.a1_in1k"
        >>> image_processor = AutoImageProcessor.from_pretrained(checkpoint)
        >>> model = AutoModel.from_pretrained(checkpoint).eval()

        >>> # Preprocess image
        >>> inputs = image_processor(image)

        >>> # Forward pass
        >>> with torch.no_grad():
        ...     outputs = model(**inputs)

        >>> # Get pooled output
        >>> pooled_output = outputs.pooler_output

        >>> # Get last hidden state
        >>> last_hidden_state = outputs.last_hidden_state
        ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        do_pooling = do_pooling if do_pooling is not None else self.config.do_pooling

        if output_attentions:
            raise ValueError("Cannot set `output_attentions` for timm models.")

        if output_hidden_states and not hasattr(self.timm_model, "forward_intermediates"):
            raise ValueError(
                "The 'output_hidden_states' option cannot be set for this timm model. "
                "To enable this feature, the 'forward_intermediates' method must be implemented "
                "in the timm model (available in timm versions > 1.*). Please consider using a "
                "different architecture or updating the timm package to a compatible version."
            )

        pixel_values = pixel_values.to(self.device, self.dtype)

        if output_hidden_states:
            # to enable hidden states selection
            if isinstance(output_hidden_states, (list, tuple)):
                kwargs["indices"] = output_hidden_states
            last_hidden_state, hidden_states = self.timm_model.forward_intermediates(pixel_values, **kwargs)
        else:
            last_hidden_state = self.timm_model.forward_features(pixel_values, **kwargs)
            hidden_states = None

        if do_pooling:
            # classification head is not created, applying pooling only
            pooler_output = self.timm_model.forward_head(last_hidden_state)
        else:
            pooler_output = None

        if not return_dict:
            outputs = (last_hidden_state, pooler_output, hidden_states)
            outputs = tuple(output for output in outputs if output is not None)
            return outputs

        return TimmWrapperModelOutput(
            last_hidden_state=last_hidden_state,
            pooler_output=pooler_output,
            hidden_states=hidden_states,
        )


class TimmWrapperForImageClassification(TimmWrapperPreTrainedModel):
    """
    Wrapper class for timm models to be used in transformers for image classification.
    """

    def __init__(self, config: TimmWrapperConfig):
        super().__init__(config)

        if config.num_labels == 0:
            raise ValueError(
                "You are trying to load weights into `TimmWrapperForImageClassification` from a checkpoint with no classifier head. "
                "Please specify the number of classes, e.g. `model = TimmWrapperForImageClassification.from_pretrained(..., num_labels=10)`, "
                "or use `TimmWrapperModel` for feature extraction."
            )

        self.timm_model = timm.create_model(config.architecture, pretrained=False, num_classes=config.num_labels)
        self.num_labels = config.num_labels
        self.post_init()

    @add_start_docstrings_to_model_forward(TIMM_WRAPPER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ImageClassifierOutput, config_class=TimmWrapperConfig)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[Union[bool, List[int]]] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[ImageClassifierOutput, Tuple[Tensor, ...]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Returns:

        Examples:
        ```python
        >>> import torch
        >>> from PIL import Image
        >>> from urllib.request import urlopen
        >>> from transformers import AutoModelForImageClassification, AutoImageProcessor

        >>> # Load image
        >>> image = Image.open(urlopen(
        ...     'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
        ... ))

        >>> # Load model and image processor
        >>> checkpoint = "timm/resnet50.a1_in1k"
        >>> image_processor = AutoImageProcessor.from_pretrained(checkpoint)
        >>> model = AutoModelForImageClassification.from_pretrained(checkpoint).eval()

        >>> # Preprocess image
        >>> inputs = image_processor(image)

        >>> # Forward pass
        >>> with torch.no_grad():
        ...     logits = model(**inputs).logits

        >>> # Get top 5 predictions
        >>> top5_probabilities, top5_class_indices = torch.topk(logits.softmax(dim=1) * 100, k=5)
        ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        if output_attentions:
            raise ValueError("Cannot set `output_attentions` for timm models.")

        if output_hidden_states and not hasattr(self.timm_model, "forward_intermediates"):
            raise ValueError(
                "The 'output_hidden_states' option cannot be set for this timm model. "
                "To enable this feature, the 'forward_intermediates' method must be implemented "
                "in the timm model (available in timm versions > 1.*). Please consider using a "
                "different architecture or updating the timm package to a compatible version."
            )

        pixel_values = pixel_values.to(self.device, self.dtype)

        if output_hidden_states:
            # to enable hidden states selection
            if isinstance(output_hidden_states, (list, tuple)):
                kwargs["indices"] = output_hidden_states
            last_hidden_state, hidden_states = self.timm_model.forward_intermediates(pixel_values, **kwargs)
            logits = self.timm_model.forward_head(last_hidden_state)
        else:
            logits = self.timm_model(pixel_values, **kwargs)
            hidden_states = None

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            outputs = (loss, logits, hidden_states)
            outputs = tuple(output for output in outputs if output is not None)
            return outputs

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states,
        )


__all__ = ["TimmWrapperPreTrainedModel", "TimmWrapperModel", "TimmWrapperForImageClassification"]
