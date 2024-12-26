from ..rt_detr.configuration_rt_detr import RTDetrConfig
from ..rt_detr.modeling_rt_detr import (
    RTDetrDecoderLayer,
    RTDetrDecoder,
    RTDetrMLPPredictionHead,
    RTDetrDecoderOutput,
    inverse_sigmoid,
)

import torch
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F
import math
from typing import List, Optional
import functools
from ...activations import ACT2CLS


class DFineConfig(RTDetrConfig):
    model_type = "d-fine"

    def __init__(self, decoder_offset_scale=0.5, eval_idx=-1, layer_scale=2, **super_kwargs):
        super().__init__(**super_kwargs)

        self.decoder_offset_scale = decoder_offset_scale
        self.eval_idx = eval_idx
        self.layer_scale = layer_scale


def deformable_attention_core_func_v2(
    value: torch.Tensor,
    value_spatial_shapes,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    num_points_list: List[int],
    method="default",
):
    """
    Args:
        value (Tensor): [bs, value_length, n_head, c]
        value_spatial_shapes (Tensor|List): [n_levels, 2]
        value_level_start_index (Tensor|List): [n_levels]
        sampling_locations (Tensor): [bs, query_length, n_head, n_levels * n_points, 2]
        attention_weights (Tensor): [bs, query_length, n_head, n_levels * n_points]

    Returns:
        output (Tensor): [bs, Length_{query}, C]
    """
    bs, n_head, c, _ = value[0].shape
    _, Len_q, _, _, _ = sampling_locations.shape

    # sampling_offsets [8, 480, 8, 12, 2]
    if method == "default":
        sampling_grids = 2 * sampling_locations - 1

    elif method == "discrete":
        sampling_grids = sampling_locations

    sampling_grids = sampling_grids.permute(0, 2, 1, 3, 4).flatten(0, 1)
    sampling_locations_list = sampling_grids.split(num_points_list, dim=-2)

    sampling_value_list = []
    for level, (h, w) in enumerate(value_spatial_shapes):
        value_l = value[level].reshape(bs * n_head, c, h, w)
        sampling_grid_l: torch.Tensor = sampling_locations_list[level]

        if method == "default":
            sampling_value_l = F.grid_sample(
                value_l, sampling_grid_l, mode="bilinear", padding_mode="zeros", align_corners=False
            )

        elif method == "discrete":
            # n * m, seq, n, 2
            sampling_coord = (sampling_grid_l * torch.tensor([[w, h]], device=value_l.device) + 0.5).to(torch.int64)

            # FIX ME? for rectangle input
            sampling_coord = sampling_coord.clamp(0, h - 1)
            sampling_coord = sampling_coord.reshape(bs * n_head, Len_q * num_points_list[level], 2)

            s_idx = (
                torch.arange(sampling_coord.shape[0], device=value_l.device)
                .unsqueeze(-1)
                .repeat(1, sampling_coord.shape[1])
            )
            sampling_value_l: torch.Tensor = value_l[s_idx, :, sampling_coord[..., 1], sampling_coord[..., 0]]  # n l c

            sampling_value_l = sampling_value_l.permute(0, 2, 1).reshape(bs * n_head, c, Len_q, num_points_list[level])

        sampling_value_list.append(sampling_value_l)

    attn_weights = attention_weights.permute(0, 2, 1, 3).reshape(bs * n_head, 1, Len_q, sum(num_points_list))
    weighted_sample_locs = torch.concat(sampling_value_list, dim=-1) * attn_weights
    output = weighted_sample_locs.sum(-1).reshape(bs, n_head * c, Len_q)

    return output.permute(0, 2, 1)


class DFineMultiscaleDeformableAttention(nn.Module):
    def __init__(
        self,
        config: DFineConfig,
        method="default",
    ):
        """
        D-Fine version of multiscale deformable attention
        """
        super(DFineMultiscaleDeformableAttention, self).__init__()
        self.d_model = config.d_model
        self.n_levels = config.num_feature_levels
        self.n_heads = config.decoder_attention_heads
        self.n_points = config.decoder_n_points
        self.offset_scale = config.decoder_offset_scale

        if isinstance(self.n_points, list):
            assert len(self.n_points) == self.n_levels, ""
            num_points_list = self.n_points
        else:
            num_points_list = [self.n_points for _ in range(self.n_levels)]

        self.num_points_list = num_points_list

        num_points_scale = [1 / n for n in num_points_list for _ in range(n)]
        self.register_buffer("num_points_scale", torch.tensor(num_points_scale, dtype=torch.float32))

        self.total_points = self.n_heads * sum(num_points_list)
        self.method = method

        self.head_dim = self.d_model // self.n_heads
        assert self.head_dim * self.n_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.sampling_offsets = nn.Linear(self.d_model, self.total_points * 2)
        self.attention_weights = nn.Linear(self.d_model, self.total_points)

        self.ms_deformable_attn_core = functools.partial(deformable_attention_core_func_v2, method=self.method)

        self._reset_parameters()

        if method == "discrete":
            for p in self.sampling_offsets.parameters():
                p.requires_grad = False

    def _reset_parameters(self):
        # sampling_offsets
        init.constant_(self.sampling_offsets.weight, 0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
        grid_init = grid_init.reshape(self.num_heads, 1, 2).tile([1, sum(self.num_points_list), 1])
        scaling = torch.concat([torch.arange(1, n + 1) for n in self.num_points_list]).reshape(1, -1, 1)
        grid_init *= scaling
        self.sampling_offsets.bias.data[...] = grid_init.flatten()

        # attention_weights
        init.constant_(self.attention_weights.weight, 0)
        init.constant_(self.attention_weights.bias, 0)

    def forward(
        self, query: torch.Tensor, reference_points: torch.Tensor, value: torch.Tensor, value_spatial_shapes: List[int]
    ):
        """
        Args:
            query (Tensor): [bs, query_length, C]
            reference_points (Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (Tensor): [bs, value_length, C]
            value_spatial_shapes (List): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, Len_q = query.shape[:2]

        sampling_offsets: torch.Tensor = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.reshape(bs, Len_q, self.num_heads, sum(self.num_points_list), 2)

        attention_weights = self.attention_weights(query).reshape(bs, Len_q, self.num_heads, sum(self.num_points_list))
        attention_weights = F.softmax(attention_weights, dim=-1)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.tensor(value_spatial_shapes)
            offset_normalizer = offset_normalizer.flip([1]).reshape(1, 1, 1, self.num_levels, 1, 2)
            sampling_locations = (
                reference_points.reshape(bs, Len_q, 1, self.num_levels, 1, 2) + sampling_offsets / offset_normalizer
            )
        elif reference_points.shape[-1] == 4:
            # reference_points [8, 480, None, 1,  4]
            # sampling_offsets [8, 480, 8,    12, 2]
            num_points_scale = self.num_points_scale.to(dtype=query.dtype).unsqueeze(-1)
            offset = sampling_offsets * num_points_scale * reference_points[:, :, None, :, 2:] * self.offset_scale
            sampling_locations = reference_points[:, :, None, :, :2] + offset
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".format(reference_points.shape[-1])
            )

        output = self.ms_deformable_attn_core(
            value, value_spatial_shapes, sampling_locations, attention_weights, self.num_points_list
        )

        return output


class DFineDecoderLayer(RTDetrDecoderLayer):
    def __init__(self, config: DFineConfig):
        # initialize parent class
        super().__init__(config)

        # override the encoder attention module with d-fine version
        self.encoder_attn = DFineMultiscaleDeformableAttention(config=config)
        # gate
        self.gateway = Gate(config.d_model)
        self._reset_parameters()

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[torch.Tensor] = None,
        reference_points=None,
        spatial_shapes=None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ):
        residual = hidden_states

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=encoder_attention_mask,
            position_embeddings=position_embeddings,
            output_attentions=output_attentions,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        second_residual = hidden_states

        # Cross-Attention
        cross_attn_weights = None
        hidden_states, cross_attn_weights = self.encoder_attn(
            query=hidden_states,
            value=encoder_hidden_states,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
        )

        hidden_states = self.gateway(
            second_residual, nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        )

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states.clamp(min=-65504, max=65504))

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        return outputs

    def _reset_parameters(self):
        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.fc2.weight)


def weighting_function(reg_max, up, reg_scale):
    """
    Generates the non-uniform Weighting Function W(n) for bounding box regression.

    Args:
        reg_max (int): Max number of the discrete bins.
        up (Tensor): Controls upper bounds of the sequence,
                     where maximum offset is ±up * H / W.
        reg_scale (float): Controls the curvature of the Weighting Function.
                           Larger values result in flatter weights near the central axis W(reg_max/2)=0
                           and steeper weights at both ends.
    Returns:
        Tensor: Sequence of Weighting Function.
    """
    upper_bound1 = abs(up[0]) * abs(reg_scale)
    upper_bound2 = abs(up[0]) * abs(reg_scale) * 2
    step = (upper_bound1 + 1) ** (2 / (reg_max - 2))
    left_values = [-((step) ** i) + 1 for i in range(reg_max // 2 - 1, 0, -1)]
    right_values = [(step) ** i - 1 for i in range(1, reg_max // 2)]
    values = [-upper_bound2] + left_values + [torch.zeros_like(up[0][None])] + right_values + [upper_bound2]
    return torch.cat(values, 0)


class DFineMLPPredictionHead(RTDetrMLPPredictionHead):
    pass


def box_xyxy_to_cxcywh(x: torch.Tensor) -> torch.Tensor:
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def distance2bbox(points, distance, reg_scale):
    """
    Decodes edge-distances into bounding box coordinates.

    Args:
        points (Tensor): (B, N, 4) or (N, 4) format, representing [x, y, w, h],
                         where (x, y) is the center and (w, h) are width and height.
        distance (Tensor): (B, N, 4) or (N, 4), representing distances from the
                           point to the left, top, right, and bottom boundaries.

        reg_scale (float): Controls the curvature of the Weighting Function.

    Returns:
        Tensor: Bounding boxes in (N, 4) or (B, N, 4) format [cx, cy, w, h].
    """
    reg_scale = abs(reg_scale)
    x1 = points[..., 0] - (0.5 * reg_scale + distance[..., 0]) * (points[..., 2] / reg_scale)
    y1 = points[..., 1] - (0.5 * reg_scale + distance[..., 1]) * (points[..., 3] / reg_scale)
    x2 = points[..., 0] + (0.5 * reg_scale + distance[..., 2]) * (points[..., 2] / reg_scale)
    y2 = points[..., 1] + (0.5 * reg_scale + distance[..., 3]) * (points[..., 3] / reg_scale)

    bboxes = torch.stack([x1, y1, x2, y2], -1)

    return box_xyxy_to_cxcywh(bboxes)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act="relu"):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.act = ACT2CLS[act]()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class LQE(nn.Module):
    def __init__(self, k, hidden_dim, num_layers, reg_max):
        super(LQE, self).__init__()
        self.k = k
        self.reg_max = reg_max
        self.reg_conf = MLP(4 * (k + 1), hidden_dim, 1, num_layers)
        init.constant_(self.reg_conf.layers[-1].bias, 0)
        init.constant_(self.reg_conf.layers[-1].weight, 0)

    def forward(self, scores, pred_corners):
        B, L, _ = pred_corners.size()
        prob = F.softmax(pred_corners.reshape(B, L, 4, self.reg_max + 1), dim=-1)
        prob_topk, _ = prob.topk(self.k, dim=-1)
        stat = torch.cat([prob_topk, prob_topk.mean(dim=-1, keepdim=True)], dim=-1)
        quality_score = self.reg_conf(stat.reshape(B, L, -1))
        return scores + quality_score


class DFineDecoderOutput(RTDetrDecoderOutput):
    pass


class DFineDecoder(RTDetrDecoder):
    """
    D-FINE Decoder implementing Fine-grained Distribution Refinement (FDR).

    This decoder refines object detection predictions through iterative updates across multiple layers,
    utilizing attention mechanisms, location quality estimators, and distribution refinement techniques
    to improve bounding box accuracy and robustness.
    """

    def __init__(self, config: DFineConfig, num_layers, reg_max, reg_scale, up):
        super().__init__(config=config)
        self.d_model = config.d_model
        self.num_layers = num_layers
        self.layer_scale = config.layer_scale
        self.eval_idx = config.eval_idx if config.eval_idx >= 0 else num_layers + config.eval_idx
        self.num_head = config.decoder_attention_heads
        self.up, self.reg_scale, self.reg_max = up, reg_scale, reg_max
        self.layers = nn.ModuleList(
            [DFineDecoderLayer(config=config) for _ in range(config.decoder_layers)]
            + [DFineDecoderLayer(config=config) for _ in range(config.decoder_layers - self.eval_idx - 1)]
        )
        self.lqe_layers = nn.ModuleList([LQE(4, 64, 2, reg_max) for _ in range(config.decoder_layers)])

    def value_op(self, memory, value_proj, value_scale, memory_mask, memory_spatial_shapes):
        """
        Preprocess values for MSDeformableAttention.
        """
        value = value_proj(memory) if value_proj is not None else memory
        value = F.interpolate(memory, size=value_scale) if value_scale is not None else value
        if memory_mask is not None:
            value = value * memory_mask.to(value.dtype).unsqueeze(-1)
        value = value.reshape(value.shape[0], value.shape[1], self.num_head, -1)
        split_shape = [h * w for h, w in memory_spatial_shapes]
        return value.permute(0, 2, 3, 1).split(split_shape, dim=-1)

    def forward(
        self,
        inputs_embeds,
        ref_points_unact,
        memory,
        spatial_shapes,
        bbox_head,
        score_head,
        pre_bbox_head,
        integral,
        up,
        reg_scale,
        encoder_attn_mask=None,
        memory_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is not None:
            hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and memory is not None) else None
        intermediate = ()
        intermediate_reference_points = ()
        intermediate_logits = ()

        output_detach = pred_corners_undetach = 0
        value = self.value_op(memory, None, None, memory_mask, spatial_shapes)

        project = weighting_function(self.reg_max, up, reg_scale)
        ref_points_detach = F.sigmoid(ref_points_unact)

        for i, decoder_layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = self.query_pos_head(ref_points_detach).clamp(min=-10, max=10)

            # TODO Adjust scale if needed for detachable wider layers
            if i >= self.eval_idx + 1 and self.layer_scale > 1:
                query_pos_embed = F.interpolate(query_pos_embed, scale_factor=self.layer_scale)
                value = self.value_op(memory, None, query_pos_embed.shape[-1], memory_mask, spatial_shapes)
                hidden_states = F.interpolate(hidden_states, size=query_pos_embed.shape[-1])
                output_detach = hidden_states.detach()

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            output = decoder_layer(
                hidden_states=hidden_states,
                position_embeddings=query_pos_embed,
                reference_points=ref_points_input,
                spatial_shapes=spatial_shapes,
                encoder_hidden_states=value,
                encoder_attention_mask=encoder_attn_mask,
                output_attentions=output_attentions,
            )

            hidden_states = output[0]

            if i == 0:
                # Initial bounding box predictions with inverse sigmoid refinement
                pre_bboxes = F.sigmoid(pre_bbox_head(output) + inverse_sigmoid(ref_points_detach))
                pre_scores = score_head[0](hidden_states)
                ref_points_initial = pre_bboxes.detach()

            # Refine bounding box corners using FDR, integrating previous layer's corrections
            pred_corners = bbox_head[i](hidden_states + output_detach) + pred_corners_undetach
            inter_ref_bbox = distance2bbox(ref_points_initial, integral(pred_corners, project), reg_scale)

            pred_corners_undetach = pred_corners
            ref_points_detach = inter_ref_bbox.detach()
            output_detach = hidden_states.detach()

            # Store intermediate results
            intermediate += (hidden_states,)
            intermediate_reference_points += (inter_ref_bbox,)
            intermediate_logits += (pre_scores,) if i == 0 else (score_head[i](hidden_states),)

            if output_attentions:
                all_self_attns += (output[1],)
                if memory is not None:
                    all_cross_attentions += (output[2],)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    intermediate,
                    intermediate_logits,
                    intermediate_reference_points,
                    all_hidden_states,
                    all_self_attns,
                    all_cross_attentions,
                ]
                if v is not None
            )

        return DFineDecoderOutput(
            last_hidden_state=hidden_states,
            intermediate_hidden_states=torch.stack(intermediate, dim=1),
            intermediate_logits=torch.stack(intermediate_logits, dim=1),
            intermediate_reference_points=torch.stack(intermediate_reference_points, dim=1),
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


class Gate(nn.Module):
    def __init__(self, d_model):
        super(Gate, self).__init__()
        self.gate = nn.Linear(2 * d_model, 2 * d_model)
        bias = self._bias_init_with_prob(0.5)
        init.constant_(self.gate.bias, bias)
        init.constant_(self.gate.weight, 0)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x1, x2):
        gate_input = torch.cat([x1, x2], dim=-1)
        gates = torch.sigmoid(self.gate(gate_input))
        gate1, gate2 = gates.chunk(2, dim=-1)
        return self.norm(gate1 * x1 + gate2 * x2)

    def _bias_init_with_prob(self, prior_prob=0.01):
        """initialize conv/fc bias value according to a given probability value."""
        bias_init = float(-math.log((1 - prior_prob) / prior_prob))
        return bias_init
