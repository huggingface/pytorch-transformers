# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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

from typing import Optional, Tuple, Union

import torch

from ...modeling_outputs import BackboneOutput
from ...modeling_utils import PreTrainedModel
from ...utils import is_timm_available, is_torch_available, requires_backends
from ...utils.backbone_utils import BackboneMixin
from .configuration_timm_backbone import TimmBackboneConfig


if is_timm_available():
    import timm


if is_torch_available():
    from torch import Tensor


class TimmBackbone(PreTrainedModel, BackboneMixin):
    """
    Wrapper class for timm models to be used as backbones. This enables using the timm models interchangeably with the
    other models in the library keeping the same API.
    """

    main_input_name = "pixel_values"
    supports_gradient_checkpointing = False
    config_class = TimmBackboneConfig

    def __init__(self, config, **kwargs):
        requires_backends(self, "timm")
        super().__init__(config)
        self.config = config

        if config.backbone is None:
            raise ValueError("backbone is not set in the config. Please set it to a timm model name.")

        if hasattr(config, "out_features") and config.out_features is not None:
            raise ValueError("out_features is not supported by TimmBackbone. Please use out_indices instead.")

        pretrained = getattr(config, "use_pretrained_backbone", None)
        if pretrained is None:
            raise ValueError("use_pretrained_backbone is not set in the config. Please set it to True or False.")

        # We just take the final layer by default. This matches the default for the transformers models.
        out_indices = config.out_indices if getattr(config, "out_indices", None) is not None else (-1,)

        in_chans = kwargs.pop("in_chans", config.num_channels)
        self._backbone = timm.create_model(
            config.backbone,
            pretrained=pretrained,
            # This is currently not possible for transformer architectures.
            features_only=config.features_only,
            in_chans=in_chans,
            out_indices=out_indices,
            **kwargs,
        )

        # Converts all `BatchNorm2d` and `SyncBatchNorm` or `BatchNormAct2d` and `SyncBatchNormAct2d` layers of provided module into `FrozenBatchNorm2d` or `FrozenBatchNormAct2d` respectively
        if getattr(config, "freeze_batch_norm_2d", False):
            self.freeze_batch_norm_2d()

        # These are used to control the output of the model when called. If output_hidden_states is True, then
        # return_layers is modified to include all layers.
        self._return_layers = {
            layer["module"]: str(layer["index"]) for layer in self._backbone.feature_info.get_dicts()
        }
        self._all_layers = {layer["module"]: str(i) for i, layer in enumerate(self._backbone.feature_info.info)}
        super()._init_backbone(config)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        requires_backends(cls, ["vision", "timm"])
        from ...models.timm_backbone import TimmBackboneConfig

        config = kwargs.pop("config", TimmBackboneConfig())

        use_timm = kwargs.pop("use_timm_backbone", True)
        if not use_timm:
            raise ValueError("use_timm_backbone must be True for timm backbones")

        num_channels = kwargs.pop("num_channels", config.num_channels)
        features_only = kwargs.pop("features_only", config.features_only)
        use_pretrained_backbone = kwargs.pop("use_pretrained_backbone", config.use_pretrained_backbone)
        out_indices = kwargs.pop("out_indices", config.out_indices)
        config = TimmBackboneConfig(
            backbone=pretrained_model_name_or_path,
            num_channels=num_channels,
            features_only=features_only,
            use_pretrained_backbone=use_pretrained_backbone,
            out_indices=out_indices,
        )
        return super()._from_config(config, **kwargs)

    def freeze_batch_norm_2d(self):
        timm.utils.model.freeze_batch_norm_2d(self._backbone)

    def unfreeze_batch_norm_2d(self):
        timm.utils.model.unfreeze_batch_norm_2d(self._backbone)

    def _init_weights(self, module):
        """
        Empty init weights function to ensure compatibility of the class in the library.
        """
        pass

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[BackboneOutput, Tuple[Tensor, ...]]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        if output_attentions:
            raise ValueError("Cannot output attentions for timm backbones at the moment")

        if output_hidden_states:
            # We modify the return layers to include all the stages of the backbone
            self._backbone.return_layers = self._all_layers
            hidden_states = self._backbone(pixel_values, **kwargs)
            self._backbone.return_layers = self._return_layers
            feature_maps = tuple(hidden_states[i] for i in self.out_indices)
        else:
            feature_maps = self._backbone(pixel_values, **kwargs)
            hidden_states = None

        feature_maps = tuple(feature_maps)
        hidden_states = tuple(hidden_states) if hidden_states is not None else None

        if not return_dict:
            output = (feature_maps,)
            if output_hidden_states:
                output = output + (hidden_states,)
            return output

        return BackboneOutput(feature_maps=feature_maps, hidden_states=hidden_states, attentions=None)


__all__ = ["TimmBackbone"]
