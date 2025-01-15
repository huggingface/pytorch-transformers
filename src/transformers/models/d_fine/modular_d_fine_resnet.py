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
from ..rt_detr.modeling_rt_detr_resnet import RTDetrResNetBackbone, RTDetrResNetPreTrainedModel
from torch import nn


class DFineResNetConfig(RTDetrResNetConfig):
    def __init__(
        self,
        **super_kwargs,
    ):
        super().__init__(**super_kwargs)
        self.depths = [5, 5, 5, 5]


class DFineResNetPreTrainedModel(RTDetrResNetPreTrainedModel):
    pass


class DFineResNetEncoder(nn.Module):
    def __init__(self, config: DFineResNetConfig):
        super().__init__()


class DFineResNetBackbone(RTDetrResNetBackbone):
    def __init__(self, config: DFineResNetConfig):
        super().__init__(config=config)
        self.encoder = DFineResNetEncoder(config=config)


__all__ = ["DFineResNetConfig", "DFineResNetBackbone", "DFineResNetPreTrainedModel"]