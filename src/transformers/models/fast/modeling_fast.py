# coding=utf-8
# Copyright 2021 Microsoft Research and The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch FAST model."""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils import is_timm_available


if is_timm_available():
    from timm import create_model


from transformers import (
    AutoBackbone,
    FastConfig,
    PreTrainedModel,
    add_start_docstrings,
    is_timm_available,
    requires_backends
)
from transformers.utils import ModelOutput, add_start_docstrings_to_model_forward, replace_return_docstrings


_CONFIG_FOR_DOC = "FastConfig"

FAST_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`FastConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

FAST_FOR_CAPTIONING_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`FastImageProcessor.__call__`] for details.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


def get_same_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        padding1 = get_same_padding(kernel_size[0])
        padding2 = get_same_padding(kernel_size[1])
        return padding1, padding2
    return kernel_size // 2


class FASTConvLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        bias=False,
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride

        padding = get_same_padding(self.kernel_size)
        # if isinstance(padding, int):
        #     padding *= self.dilation
        # else:
        #     padding[0] *= self.dilation
        #     padding[1] *= self.dilation

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )

    def forward(self, hidden_states):
        if self.training:
            if hasattr(self, "fused_conv"):
                delattr(self, "fused_conv")
            hidden_states = self.conv(hidden_states)
            return hidden_states
        else:
            if not hasattr(self, "fused_conv"):
                setattr(self, "fused_conv", self.conv)
            hidden_states = self.fused_conv(hidden_states)
            return hidden_states

    def fuse_conv_batch_norm(self, conv, batch_norm):
        """During inference, the functionary of batch norm layers is turned off but
        only the mean and var alone channels are used, which exposes the chance to fuse it with the preceding conv
        layers to save computations and simplify network structures."""
        if isinstance(batch_norm, nn.Identity):
            return conv
        conv_w = conv.weight
        conv_b = conv.bias if conv.bias is not None else torch.zeros_like(batch_norm.running_mean)

        factor = batch_norm.weight / torch.sqrt(batch_norm.running_var + batch_norm.eps)
        conv.weight = nn.Parameter(conv_w * factor.reshape([conv.out_channels, 1, 1, 1]))
        conv.bias = nn.Parameter((conv_b - batch_norm.running_mean) * factor + batch_norm.bias)
        return conv


class FASTRepConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        padding = (int((kernel_size[0] - 1) / 2), int((kernel_size[1] - 1) / 2))

        self.activation = nn.ReLU(inplace=True)

        self.main_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.main_batch_norm = nn.BatchNorm2d(num_features=out_channels)

        ver_pad = (int((kernel_size[0] - 1) / 2), 0)
        hor_pad = (0, int((kernel_size[1] - 1) / 2))

        if kernel_size[1] != 1:
            self.vertical_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(kernel_size[0], 1),
                stride=stride,
                padding=ver_pad,
                bias=False,
            )
            self.vertical_batch_norm = nn.BatchNorm2d(num_features=out_channels)
        else:
            self.vertical_conv, self.vertical_batch_norm = None, None

        if kernel_size[0] != 1:  # 卷积核的高大于1 -> 有水平卷积
            self.horizontal_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, kernel_size[1]),
                stride=stride,
                padding=hor_pad,
                bias=False,
            )
            self.horizontal_batch_norm = nn.BatchNorm2d(num_features=out_channels)
        else:
            self.horizontal_conv, self.horizontal_batch_norm = None, None

        self.rbr_identity = (
            nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
        )

    def forward(self, hidden_states):
        if self.training:
            if hasattr(self, "fused_conv"):
                self.__delattr__("fused_conv")

            main_outputs = self.main_conv(hidden_states)
            main_outputs = self.main_batch_norm(main_outputs)
            if self.vertical_conv is not None:
                vertical_outputs = self.vertical_conv(hidden_states)
                vertical_outputs = self.vertical_batch_norm(vertical_outputs)
            else:
                vertical_outputs = 0

            if self.horizontal_conv is not None:
                horizontal_outputs = self.horizontal_conv(hidden_states)
                horizontal_outputs = self.horizontal_batch_norm(horizontal_outputs)
            else:
                horizontal_outputs = 0

            if self.rbr_identity is None:
                id_out = 0
            else:
                id_out = self.rbr_identity(hidden_states)

            return self.activation(main_outputs + vertical_outputs + horizontal_outputs + id_out)
        else:
            if not hasattr(self, "fused_conv"):
                self.prepare_for_eval()
            return self.activation(self.fused_conv(hidden_states))

    def _identity_to_conv(self, identity):
        if identity is None:
            return 0, 0
        if not hasattr(self, "id_tensor"):
            input_dim = self.in_channels
            kernel_value = np.zeros((self.in_channels, input_dim, 1, 1), dtype=np.float32)
            for i in range(self.in_channels):
                kernel_value[i, i % input_dim, 0, 0] = 1
            id_tensor = torch.from_numpy(kernel_value).to(identity.weight.device)
            self.id_tensor = self._pad_to_mxn_tensor(id_tensor)
        kernel = self.id_tensor
        running_mean = identity.running_mean
        running_var = identity.running_var
        gamma = identity.weight
        beta = identity.bias
        eps = identity.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def _fuse_batch_norm_tensor(self, conv, batch_norm):
        kernel = conv.weight
        kernel = self._pad_to_mxn_tensor(kernel)
        running_mean = batch_norm.running_mean
        running_var = batch_norm.running_var
        gamma = batch_norm.weight
        beta = batch_norm.bias
        eps = batch_norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def get_equivalent_kernel_bias(self):
        kernel_mxn, bias_mxn = self._fuse_batch_norm_tensor(self.main_conv, self.main_batch_norm)
        if self.vertical_conv is not None:
            kernel_mx1, bias_mx1 = self._fuse_batch_norm_tensor(self.vertical_conv, self.vertical_batch_norm)
        else:
            kernel_mx1, bias_mx1 = 0, 0
        if self.horizontal_conv is not None:
            kernel_1xn, bias_1xn = self._fuse_batch_norm_tensor(self.horizontal_conv, self.horizontal_batch_norm)
        else:
            kernel_1xn, bias_1xn = 0, 0
        kernel_id, bias_id = self._identity_to_conv(self.rbr_identity)
        kernel_mxn = kernel_mxn + kernel_mx1 + kernel_1xn + kernel_id
        bias_mxn = bias_mxn + bias_mx1 + bias_1xn + bias_id
        return kernel_mxn, bias_mxn

    def _pad_to_mxn_tensor(self, kernel):
        kernel_height, kernel_width = self.kernel_size
        height, width = kernel.shape[2:]
        pad_left_right = (kernel_width - width) // 2
        pad_top_down = (kernel_height - height) // 2
        return torch.nn.functional.pad(kernel, [pad_left_right, pad_left_right, pad_top_down, pad_top_down])

    def prepare_for_eval(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        self.fused_conv = nn.Conv2d(
            in_channels=self.main_conv.in_channels,
            out_channels=self.main_conv.out_channels,
            kernel_size=self.main_conv.kernel_size,
            stride=self.main_conv.stride,
            padding=self.main_conv.padding,
            bias=True,
        )
        self.fused_conv.weight.data = kernel
        self.fused_conv.bias.data = bias
        for para in self.fused_conv.parameters():
            para.detach_()


class FastPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = FastConfig
    base_model_prefix = "fast"
    main_input_name = "pixel_values"

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()


class FASTNeck(nn.Module):
    def __init__(self, config):
        super().__init__()
        reduce_layer_configs = list(
            zip(
                config.neck_in_channels,
                config.neck_out_channels,
                config.neck_kernel_size,
                config.neck_stride,
            )
        )
        self.num_layers = len(reduce_layer_configs)
        for layer_ix in range(0, len(reduce_layer_configs)):
            setattr(self, f"reduce_layer{layer_ix + 1}", FASTRepConvLayer(*reduce_layer_configs[layer_ix]))

    def _upsample(self, layer_out, height, width):
        return F.upsample(layer_out, size=(height, width), mode="bilinear")

    def forward(self, hidden_states):
        first_layer_hidden = hidden_states[0]
        first_layer_hidden = self.reduce_layer1(first_layer_hidden)
        output_stages = [first_layer_hidden]

        for layer_ix in range(1, self.num_layers):
            layer_out = getattr(self, f"reduce_layer{layer_ix + 1}")(hidden_states[layer_ix])
            _, _, height, width = first_layer_hidden.size()
            layer_out = self._upsample(layer_out, height, width)
            output_stages.append(layer_out)

        combined_hidden_states = torch.cat(output_stages, 1)
        return combined_hidden_states


class FASTHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv = FASTRepConvLayer(
            config.head_conv_in_channels,
            config.head_conv_out_channels,
            config.head_conv_kernel_size,
            config.head_conv_stride,
        )

        self.final = FASTConvLayer(
            config.head_final_in_channels,
            config.head_final_out_channels,
            config.head_final_kernel_size,
            config.head_final_stride,
            config.head_final_bias,
        )

        self.pooling_size = config.head_pooling_size

        self.pooling_1s = nn.MaxPool2d(kernel_size=self.pooling_size, stride=1, padding=(self.pooling_size - 1) // 2)
        self.pooling_2s = nn.MaxPool2d(
            kernel_size=self.pooling_size // 2 + 1, stride=1, padding=(self.pooling_size // 2) // 2
        )

        if config.head_dropout_ratio > 0:
            self.dropout = nn.Dropout2d(config.head_dropout_ratio)
        else:
            self.dropout = None

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        if self.dropout is not None:
            hidden_states = self.dropout(hidden_states)
        hidden_states = self.final(hidden_states)
        return hidden_states

    def _max_pooling(self, x, scale=1):
        if scale == 1:
            x = self.pooling_1s(x)
        elif scale == 2:
            x = self.pooling_2s(x)
        return x


def emb_loss(
    emb, instance, kernel, training_mask, feature_dim=4, delta_v=0.5, delta_d=1.5, weights=(1.0, 1.0), bg_sample=False
):
    training_mask = (training_mask > 0.5).long()
    kernel = (kernel > 0.5).long()
    instance = instance * training_mask
    instance_kernel = (instance * kernel).view(-1)
    instance = instance.view(-1)
    emb = emb.view(feature_dim, -1)

    unique_labels, unique_ids = torch.unique(instance_kernel, sorted=True, return_inverse=True)
    num_instance = unique_labels.size(0)
    if num_instance <= 1:
        return 0

    emb_mean = emb.new_zeros((feature_dim, num_instance), dtype=torch.float32)
    for i, lb in enumerate(unique_labels):
        if lb == 0:
            continue
        ind_k = instance_kernel == lb
        emb_mean[:, i] = torch.mean(emb[:, ind_k], dim=1)

    l_agg = emb.new_zeros(num_instance, dtype=torch.float32)  # bug
    for i, lb in enumerate(unique_labels):
        if lb == 0:
            continue
        ind = instance == lb
        emb_ = emb[:, ind]
        dist = (emb_ - emb_mean[:, i : i + 1]).norm(p=2, dim=0)
        dist = F.relu(dist - delta_v) ** 2
        l_agg[i] = torch.mean(torch.log(dist + 1.0))
    l_agg = torch.mean(l_agg[1:])

    if num_instance > 2:
        emb_interleave = emb_mean.permute(1, 0).repeat(num_instance, 1)
        emb_band = emb_mean.permute(1, 0).repeat(1, num_instance).view(-1, feature_dim)
        # print(seg_band)

        mask = (1 - torch.eye(num_instance, dtype=torch.int8)).view(-1, 1).repeat(1, feature_dim)
        mask = mask.view(num_instance, num_instance, -1)
        mask[0, :, :] = 0
        mask[:, 0, :] = 0
        mask = mask.view(num_instance * num_instance, -1)
        # print(mask)

        dist = emb_interleave - emb_band
        dist = dist[mask > 0].view(-1, feature_dim).norm(p=2, dim=1)
        dist = F.relu(2 * delta_d - dist) ** 2
        l_dis = torch.mean(torch.log(dist + 1.0))

        if bg_sample:
            l_dis = [torch.log(dist + 1.0)]
            emb_bg = emb[:, instance == 0].view(feature_dim, -1)
            if emb_bg.size(1) > 100:
                rand_ind = np.random.permutation(emb_bg.size(1))[:100]
                emb_bg = emb_bg[:, rand_ind]
            if emb_bg.size(1) > 0:
                for i, lb in enumerate(unique_labels):
                    if lb == 0:
                        continue
                    dist = (emb_bg - emb_mean[:, i : i + 1]).norm(p=2, dim=0)
                    dist = F.relu(2 * delta_d - dist) ** 2
                    l_dis_bg = torch.mean(torch.log(dist + 1.0), 0, keepdim=True)
                    l_dis.append(l_dis_bg)
            l_dis = torch.mean(torch.cat(l_dis))
    else:
        l_dis = 0

    l_agg = weights[0] * l_agg
    l_dis = weights[1] * l_dis
    l_reg = torch.mean(torch.log(torch.norm(emb_mean, 2, 0) + 1.0)) * 0.001
    loss = l_agg + l_dis + l_reg
    return loss


def emb_loss_batch(emb, instance, kernel, training_mask, reduce=True, loss_weight=0.25):
    loss_batch = emb.new_zeros((emb.size(0)), dtype=torch.float32)

    for i in range(loss_batch.size(0)):
        loss_batch[i] = emb_loss(emb[i], instance[i], kernel[i], training_mask[i])

    loss_batch = loss_weight * loss_batch

    if reduce:
        loss_batch = torch.mean(loss_batch)

    return loss_batch


def dice_loss_with_masks(input, target, mask, reduce=True):
    loss_weight = 0.5
    batch_size = input.size(0)
    input = torch.sigmoid(input)

    input = input.contiguous().view(batch_size, -1)
    target = target.contiguous().view(batch_size, -1).float()
    mask = mask.contiguous().view(batch_size, -1).float()

    input = input * mask
    target = target * mask

    a = torch.sum(input * target, dim=1)
    b = torch.sum(input * input, dim=1) + 0.001
    c = torch.sum(target * target, dim=1) + 0.001
    d = (2 * a) / (b + c)
    loss = 1 - d

    loss = loss_weight * loss

    if reduce:
        loss = torch.mean(loss)

    return loss


def ohem_single(score, gt_text, training_mask):
    pos_num = int(torch.sum(gt_text > 0.5)) - int(torch.sum((gt_text > 0.5) & (training_mask <= 0.5)))

    if pos_num == 0:
        # selected_mask = gt_text.copy() * 0 # may be not good
        selected_mask = training_mask
        selected_mask = selected_mask.view(1, selected_mask.shape[0], selected_mask.shape[1]).float()
        return selected_mask

    neg_num = int(torch.sum(gt_text <= 0.5))
    neg_num = int(min(pos_num * 3, neg_num))

    if neg_num == 0:
        selected_mask = training_mask
        selected_mask = selected_mask.view(1, selected_mask.shape[0], selected_mask.shape[1]).float()
        return selected_mask

    neg_score = score[gt_text <= 0.5]
    neg_score_sorted, _ = torch.sort(-neg_score)
    threshold = -neg_score_sorted[neg_num - 1]

    selected_mask = ((score >= threshold) | (gt_text > 0.5)) & (training_mask > 0.5)
    selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).float()
    return selected_mask


def ohem_batch(scores, gt_texts, training_masks):
    selected_masks = []
    for i in range(scores.shape[0]):
        selected_masks.append(ohem_single(scores[i, :, :], gt_texts[i, :, :], training_masks[i, :, :]))

    selected_masks = torch.cat(selected_masks, 0).float()
    return selected_masks


def iou_single(a, b, mask, n_class):
    EPS = 1e-6
    valid = mask == 1
    a = a[valid]
    b = b[valid]
    miou = []
    for i in range(n_class):
        inter = ((a == i) & (b == i)).float()
        union = ((a == i) | (b == i)).float()

        miou.append(torch.sum(inter) / (torch.sum(union) + EPS))
    miou = sum(miou) / len(miou)
    return miou


def iou(a, b, mask, n_class=2, reduce=True):
    batch_size = a.size(0)

    a = a.view(batch_size, -1)
    b = b.view(batch_size, -1)
    mask = mask.view(batch_size, -1)

    iou = a.new_zeros((batch_size,), dtype=torch.float32)
    for i in range(batch_size):
        iou[i] = iou_single(a[i], b[i], mask[i], n_class)

    if reduce:
        iou = torch.mean(iou)
    return iou


@dataclass
class FastForSceneTextRecognitionOutput(ModelOutput):
    """
    Adapted from the base class for vision model's outputs that also contains image embeddings of the pooling of the
    last hidden states. This class also adds the loss term from the text decoder as well as the image-text similarity
    scores.

    Args:
        loss (`torch.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Languge modeling loss from the text decoder.
        text_hidden (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional*):
            The image hidden states.
    """

    loss: Optional[torch.Tensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


@add_start_docstrings(
    """FAST (faster arbitararily-shaped text detector) proposes an accurate and efficient scene text detection
    framework, termed FAST (i.e., faster arbitrarily-shaped text detector).FAST has two new designs. (1) They design a
    minimalist kernel representation (only has 1-channel output) to model text with arbitrary shape, as well as a
    GPU-parallel post-processing to efficiently assemble text lines with a negligible time overhead. (2) We search the
    network architecture tailored for text detection, leading to more powerful features than most networks that are
    searched for image classification.""",
    FAST_START_DOCSTRING,
)
class FastForSceneTextRecognition(FastPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        if config.use_timm_backbone:
            requires_backends(self, ["timm"])
            kwargs = {}
            if config.dilation:
                kwargs["output_stride"] = 16
            backbone = create_model(
                config.backbone,
                pretrained=config.use_pretrained_backbone,
                features_only=True,
                out_indices=(1, 2, 3, 4),
                in_chans=config.num_channels,
                **kwargs,
            )
        else:
            #TODO: check how to configure this backbone
            backbone = AutoBackbone.from_config(config.backbone_config)
        self.backbone = backbone
        self.neck = FASTNeck(config=config)
        self.text_detection_head = FASTHead(config=config)

        self.pooling_1s = nn.MaxPool2d(
            kernel_size=config.head_pooling_size, stride=1, padding=(config.head_pooling_size - 1) // 2
        )
        self.pooling_2s = nn.MaxPool2d(
            kernel_size=config.head_pooling_size // 2 + 1, stride=1, padding=(config.head_pooling_size // 2) // 2
        )
        self.post_init()

    def _upsample(self, x, size, scale=1):
        _, _, H, W = size
        return F.interpolate(x, size=(H // scale, W // scale), mode="bilinear")

    def _max_pooling(self, x, scale=1):
        if scale == 1:
            x = self.pooling_1s(x)
        elif scale == 2:
            x = self.pooling_2s(x)
        return x

    def loss(self, hidden, labels):
        gt_texts = labels["gt_texts"]
        gt_kernels = labels["gt_kernels"]
        training_masks = labels["training_masks"]
        gt_instances = labels["gt_instances"]

        kernels = hidden[:, 0, :, :]  # 4*640*640
        texts = self._max_pooling(kernels, scale=1)  # 4*640*640
        embs = hidden[:, 1:, :, :]  # 4*4*640*640

        selected_masks = ohem_batch(texts, gt_texts, training_masks)
        loss_text = dice_loss_with_masks(texts, gt_texts, selected_masks, reduce=False)

        selected_masks = gt_texts * training_masks
        loss_kernel = dice_loss_with_masks(kernels, gt_kernels, selected_masks, reduce=False)
        loss_kernel = torch.mean(loss_kernel, dim=0)

        loss_emb = emb_loss_batch(embs, gt_instances, gt_kernels, training_masks, reduce=False)

        return torch.mean(loss_text) + torch.mean(loss_kernel) + torch.mean(loss_emb)

    @add_start_docstrings_to_model_forward(FAST_FOR_CAPTIONING_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FastForSceneTextRecognitionOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = None,
        labels: Dict = None,
    ):
        r"""
        labels (`Dict[str, torch.Tensor]`, *optional*):
            Should contain 3 keys: gt_texts,gt_kernels,gt_instances

        Returns:

                Examples:

        ```python
        >>> from transformers import FastImageProcessor, FastForSceneTextRecognition
        >>> from PIL import Image
        >>> import requests

        >>> url = "https://huggingface.co/datasets/Raghavan/fast_model_samples/resolve/main/img657.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        >>> processor = FastImageProcessor.from_pretrained("Raghavan/fast_base_tt_800_finetune_ic17mlt")
        >>> model = FastForSceneTextRecognition.from_pretrained("Raghavan/fast_base_tt_800_finetune_ic17mlt")
        >>> inputs = processor(image, return_tensors="pt")
        >>> # forward pass
        >>> outputs = model(pixel_values=inputs["pixel_values"])
        >>> target_sizes = [(image.shape[1], image.shape[2]) for image in inputs["pixel_values"]]
        >>> threshold = 0.85
        >>> text_locations = processor.post_process_text_detection(outputs, target_sizes, threshold, bbox_type="poly")
        >>> print(text_locations[0]["bboxes"][0][:10])
        [484, 175, 484, 178, 483, 179, 452, 179, 452, 182]
        ```
        """
        # outputs = {}
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        features = (
            self.backbone(pixel_values) if self.config.use_timm_backbone else self.backbone(pixel_values).feature_maps
        )

        hidden_states = self.neck(features)

        text_detection_output = self.text_detection_head(hidden_states)

        all_hidden_states = (features, hidden_states)

        loss = None
        if labels:
            out = self._upsample(text_detection_output, pixel_values.size(), scale=1)
            loss = self.loss(out, labels)
        text_detection_output = self._upsample(text_detection_output, pixel_values.size(), scale=4)

        if not return_dict:
            output = (loss, text_detection_output) if loss is not None else (text_detection_output,)
            return output + (all_hidden_states,) if output_hidden_states else output

        return FastForSceneTextRecognitionOutput(
            loss=loss,
            last_hidden_state=text_detection_output,
            hidden_states=all_hidden_states if output_hidden_states else None,
        )
