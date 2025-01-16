# coding=utf-8
# Copyright 2024 Baidu Inc and The HuggingFace Inc. team.
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


from ..rt_detr.configuration_rt_detr_resnet import RTDetrResNetConfig
from ..rt_detr.modeling_rt_detr_resnet import RTDetrResNetBackbone, RTDetrResNetPreTrainedModel, RTDetrResNetEmbeddings, RTDetrResNetConvLayer
from torch import nn


class DFineResNetConfig(RTDetrResNetConfig):
    model_type = "d-fine-resnet"
    def __init__(
        self,
        stem_channels=[3, 32, 48],
        **super_kwargs,
    ):
        super().__init__(**super_kwargs)
        self.stem_channels = stem_channels


class DFineResNetPreTrainedModel(RTDetrResNetPreTrainedModel):
    pass


class DFineResNetConvLayer(RTDetrResNetConvLayer):
    pass


class DFineResNetEmbeddings(RTDetrResNetEmbeddings):
    def __init__(self, config: DFineResNetConfig):
        super().__init__(config=config)

        self.embedder = nn.Sequential(
            *[
                DFineResNetConvLayer(
                    config.stem_channels[0],
                    config.stem_channels[1],
                    kernel_size=3,
                    stride=2,
                    activation=config.hidden_act,
                ),
                DFineResNetConvLayer(
                    config.stem_channels[1],
                    config.stem_channels[1] // 2,
                    kernel_size=2,
                    stride=1,
                    activation=config.hidden_act,
                ),
                DFineResNetConvLayer(
                    config.stem_channels[1] // 2,
                    config.stem_channels[1],
                    kernel_size=2,
                    stride=1,
                    activation=config.hidden_act,
                ),
                DFineResNetConvLayer(
                    config.stem_channels[1] * 2,
                    config.stem_channels[1],
                    kernel_size=3,
                    stride=2,
                    activation=config.hidden_act,
                ),
                DFineResNetConvLayer(
                    config.stem_channels[1],
                    config.stem_channels[2],
                    kernel_size=1,
                    stride=1,
                    activation=config.hidden_act,
                )
            ]
        )


class DFineResNetBackbone(RTDetrResNetBackbone):
    def __init__(self, config: DFineResNetConfig):
        super().__init__(config=config)
        self.embedder = DFineResNetEmbeddings(config=config)


__all__ = ["DFineResNetConfig", "DFineResNetBackbone", "DFineResNetPreTrainedModel"]