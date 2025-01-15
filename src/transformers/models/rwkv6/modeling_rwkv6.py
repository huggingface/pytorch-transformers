# coding=utf-8
# Copyright 2024 The RWKV team and HuggingFace Inc. team.
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
"""PyTorch RWKV6 World model."""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.generation import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)
from transformers.models.rwkv.modeling_rwkv import RwkvForCausalLM, RwkvModel, RwkvBlock
from transformers.models.rwkv.modeling_rwkv import RwkvCausalLMOutput, RwkvOutput

from .configuration_rwkv6 import Rwkv6Config


def check_dependencies():
    missing_deps = []

    try:
        import triton  # noqa: F401
    except ImportError:
        missing_deps.append("triton>=3.0.0")

    try:
        import rwkvfla  # noqa: F401
    except ImportError:
        missing_deps.append("rwkv-fla")

    if missing_deps:
        install_instructions = """
Required dependencies are missing. Please install them using:

{}

""".strip()
        install_commands = "\n".join(f"pip install {dep}" for dep in missing_deps)
        print(install_instructions.format(install_commands))
        return False
    else:
        return True


if check_dependencies():
    # flake8: noqa: E402
    from rwkvfla.ops.rwkv6.chunk import chunk_rwkv6  # pylint: disable=C0411
    from rwkvfla.ops.rwkv6.fused_recurrent import fused_recurrent_rwkv6  # pylint: disable=C0411
    from rwkvfla.ops.rwkv6.recurrent_naive import native_recurrent_rwkv6  # pylint: disable=C0411
else:
    from .wkv6 import native_recurrent_rwkv6

    chunk_rwkv6 = native_recurrent_rwkv6
    fused_recurrent_rwkv6 = native_recurrent_rwkv6


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "RWKV/rwkv-6-world-1b6"
_CONFIG_FOR_DOC = "Rwkv6Config"


def rwkv6_linear_attention(
    training,
    receptance,
    key,
    value,
    time_decay,
    time_first,
    state,
):
    one_token = key.size(1) == 1
    batch, seq_length, _ = receptance.shape
    num_heads, head_size = time_first.shape
    # pylint: disable=line-too-long
    key = key.view(batch, seq_length, num_heads, head_size).transpose(1, 2)  # B, T, H, K -> B, H, T, K
    value = value.view(batch, seq_length, num_heads, head_size).transpose(1, 2)  # B, T, H, K - > B, H, T, V
    receptance = receptance.view(batch, seq_length, num_heads, head_size).transpose(1, 2)  # B, H, T, K
    time_decay = (
        -torch.exp(time_decay).view(batch, seq_length, num_heads, head_size).permute(0, 2, 1, 3)
    )  # B, T, H, K -> B, H, T, K
    time_first = time_first.reshape(num_heads, head_size)  # H, K
    if receptance.device.type == "cpu":
        out, state = native_recurrent_rwkv6(
            receptance, key, value, time_decay, time_first, scale=1.0, initial_state=state, output_final_state=True
        )
    elif one_token:
        out, state = fused_recurrent_rwkv6(
            receptance, key, value, time_decay, time_first, scale=1.0, initial_state=state, output_final_state=True
        )
    else:
        out, state = chunk_rwkv6(
            receptance, key, value, time_decay, time_first, scale=1.0, initial_state=state, output_final_state=True
        )
    return out.transpose(1, 2), state


class Rwkv6SelfAttention(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        hidden_size = config.hidden_size
        attention_hidden_size = config.attention_hidden_size
        self.attention_hidden_size = attention_hidden_size
        head_size = config.head_size
        num_heads = attention_hidden_size // head_size

        self.time_maa_x = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.time_maa_w = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.time_maa_k = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.time_maa_v = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.time_maa_r = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.time_maa_g = nn.Parameter(torch.empty(1, 1, hidden_size))

        time_mix_extra_dim = 32  # generate TIME_MIX for w,k,v,r,g
        if hidden_size == 4096:  # 7b
            time_mix_extra_dim = 64
        self.time_maa_w1 = nn.Parameter(torch.empty(hidden_size, time_mix_extra_dim * 5))
        self.time_maa_w2 = nn.Parameter(torch.empty(5, time_mix_extra_dim, hidden_size))

        self.time_decay = nn.Parameter(torch.empty(1, 1, attention_hidden_size))

        time_decay_extra_dim = 64
        if hidden_size == 4096:  # 7b
            time_decay_extra_dim = 128
        self.time_decay_w1 = nn.Parameter(torch.empty(hidden_size, time_decay_extra_dim))
        self.time_decay_w2 = nn.Parameter(torch.empty(time_decay_extra_dim, attention_hidden_size))

        self.time_faaaa = nn.Parameter(torch.empty(num_heads, config.head_size))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(hidden_size, attention_hidden_size, bias=False)
        self.key = nn.Linear(hidden_size, attention_hidden_size, bias=False)
        self.value = nn.Linear(hidden_size, attention_hidden_size, bias=False)
        self.gate = nn.Linear(hidden_size, attention_hidden_size, bias=False)
        self.output = nn.Linear(attention_hidden_size, hidden_size, bias=False)
        self.ln_x = nn.GroupNorm(num_heads, hidden_size, eps=(1e-5) * (config.head_size_divisor**2))

    def extract_key_value(self, hidden, state=None):
        # Mix hidden with the previous timestep to produce key, value,
        # receptance
        if hidden.size(1) == 1 and state is not None:
            shifted = state[0][:, :, self.layer_id]
        else:
            shifted = self.time_shift(hidden)
            if state is not None:
                shifted[:, 0] = state[0][:, :, self.layer_id]
        if len(shifted.size()) == 2:
            shifted = shifted.unsqueeze(1)

        x = hidden

        B, T, C = hidden.shape

        xx = shifted - x

        xxx = x + xx * self.time_maa_x
        xxx = torch.tanh(xxx @ self.time_maa_w1).view(B * T, 5, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_w2).view(5, B, T, -1)
        mw, mk, mv, mr, mg = xxx.unbind(dim=0)

        time_decay = x + xx * (self.time_maa_w + mw)
        key = x + xx * (self.time_maa_k + mk)
        value = x + xx * (self.time_maa_v + mv)
        receptance = x + xx * (self.time_maa_r + mr)
        gate = x + xx * (self.time_maa_g + mg)

        receptance = self.receptance(receptance)
        key = self.key(key)
        value = self.value(value)
        gate = F.silu(self.gate(gate))

        time_decay = torch.tanh(time_decay @ self.time_decay_w1) @ self.time_decay_w2
        time_decay = self.time_decay + time_decay

        if state is not None:
            state[0][:, :, self.layer_id] = hidden[:, -1]

        return receptance, key, value, gate, time_decay, state

    def forward(self, hidden, state=None, use_cache=False):
        receptance, key, value, gate, time_decay, state = self.extract_key_value(hidden, state=state)

        B, T, _ = receptance.shape
        H, S = self.time_faaaa.shape

        layer_state = state[1][:, :, :, :, self.layer_id] if state is not None else None
        out, layer_state = rwkv6_linear_attention(
            self.training,
            receptance,
            key,
            value,
            time_decay,
            self.time_faaaa,
            layer_state,
        )

        if layer_state is not None:
            state[1][:, :, :, :, self.layer_id] = layer_state

        out = out.reshape(B * T, H * S)
        out = F.group_norm(
            out,
            num_groups=H,
            weight=self.ln_x.weight,
            bias=self.ln_x.bias,
            eps=self.ln_x.eps,
        ).reshape(B, T, H * S)
        out = out * gate
        out = self.output(out)
        return out, state


class Rwkv6FeedForward(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        hidden_size = config.hidden_size
        # https://github.com/BlinkDL/RWKV-LM/blob/3db37a72356b736966ddd377268f02b80963af3f/RWKV-v4neo/train.py#L168
        intermediate_size = (
            config.intermediate_size
            if config.intermediate_size is not None
            else int((config.hidden_size * 3.5) // 32 * 32)
        )

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.time_maa_k = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.time_maa_r = nn.Parameter(torch.empty(1, 1, hidden_size))

        self.key = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.receptance = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden, state=None):
        if hidden.size(1) == 1 and state is not None:
            shifted = state[2][:, :, self.layer_id]
        else:
            shifted = self.time_shift(hidden)
            if state is not None:
                shifted[:, 0] = state[2][:, :, self.layer_id]
        if len(shifted.size()) == 2:
            shifted = shifted.unsqueeze(1)

        delta_hidden_to_shifted = shifted - hidden
        key = hidden + delta_hidden_to_shifted * self.time_maa_k
        receptance = hidden + delta_hidden_to_shifted * self.time_maa_r

        key = torch.square(torch.relu(self.key(key)))
        value = self.value(key)
        receptance = torch.sigmoid(self.receptance(receptance))

        if state is not None:
            state[2][:, :, self.layer_id] = hidden[:, -1]

        return receptance * value, state


class Rwkv6Block(RwkvBlock):
    def __init__(self, config, layer_id):
        super().__init__(config, layer_id)

        self.attention = Rwkv6SelfAttention(config, layer_id)
        self.feed_forward = Rwkv6FeedForward(config, layer_id)


class Rwkv6PreTrainedModel(PreTrainedModel):
    # pylint: disable=line-too-long
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = Rwkv6Config
    base_model_prefix = "rwkv6"
    _no_split_modules = ["Rwkv6Block"]
    _keep_in_fp32_modules = ["time_decay", "time_first"]
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, Rwkv6SelfAttention):
            layer_id = module.layer_id
            num_hidden_layers = module.config.num_hidden_layers
            hidden_size = module.config.hidden_size
            attention_hidden_size = module.attention_hidden_size
            head_size = module.config.head_size
            num_heads = attention_hidden_size // head_size

            ratio_0_to_1 = layer_id / (num_hidden_layers - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / num_hidden_layers)  # 1 to ~0

            time_weight = torch.tensor(
                [i / hidden_size for i in range(hidden_size)],
                dtype=module.time_maa_k.dtype,
                device=module.time_maa_k.device,
            )
            time_weight = time_weight[None, None, :]

            decay_speed = [
                -6.0 + 5.0 * (h / (attention_hidden_size - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
                for h in range(attention_hidden_size)
            ]
            decay_speed = torch.tensor(decay_speed, dtype=module.time_decay.dtype, device=module.time_decay.device)
            tmp = torch.tensor(
                [
                    (1.0 - (i / (attention_hidden_size - 1.0))) * ratio_0_to_1 + 0.1 * ((i + 1) % 3 - 1)
                    for i in range(attention_hidden_size)
                ],
                dtype=module.time_faaaa.dtype,
                device=module.time_faaaa.device,
            )

            with torch.no_grad():
                module.time_maa_x.data = 1.0 - torch.pow(time_weight, ratio_1_to_almost0)
                module.time_maa_w.data = 1.0 - torch.pow(time_weight, ratio_1_to_almost0)
                module.time_maa_k.data = 1.0 - torch.pow(time_weight, ratio_1_to_almost0)
                module.time_maa_v.data = 1.0 - (torch.pow(time_weight, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
                module.time_maa_r.data = 1.0 - torch.pow(time_weight, 0.5 * ratio_1_to_almost0)
                module.time_maa_g.data = 1.0 - torch.pow(time_weight, 0.5 * ratio_1_to_almost0)

                TIME_MIX_EXTRA_DIM = 32  # generate TIME_MIX for w,k,v,r,g
                module.time_maa_w1.data = torch.zeros(
                    hidden_size,
                    TIME_MIX_EXTRA_DIM * 5,
                    dtype=module.time_maa_w1.dtype,
                    device=module.time_maa_w1.device,
                ).uniform_(-1e-4, 1e-4)
                module.time_maa_w2.data = torch.zeros(
                    5,
                    TIME_MIX_EXTRA_DIM,
                    hidden_size,
                    dtype=module.time_maa_w2.dtype,
                    device=module.time_maa_w2.device,
                ).uniform_(-1e-4, 1e-4)

                TIME_DECAY_EXTRA_DIM = 64
                module.time_decay_w1.data = torch.zeros(
                    hidden_size,
                    TIME_DECAY_EXTRA_DIM,
                    dtype=module.time_decay_w1.dtype,
                    device=module.time_decay_w1.device,
                ).uniform_(-1e-4, 1e-4)
                module.time_decay_w2.data = torch.zeros(
                    TIME_DECAY_EXTRA_DIM,
                    attention_hidden_size,
                    dtype=module.time_decay_w2.dtype,
                    device=module.time_decay_w2.device,
                ).uniform_(-1e-4, 1e-4)

                module.time_decay.data = decay_speed.reshape(num_heads, head_size)
                module.time_faaaa.data = tmp.reshape(num_heads, head_size)

        elif isinstance(module, Rwkv6FeedForward):
            layer_id = module.layer_id
            num_hidden_layers = module.config.num_hidden_layers
            hidden_size = module.config.hidden_size

            ratio_1_to_almost0 = 1.0 - (layer_id / num_hidden_layers)  # 1 to ~0

            time_weight = torch.tensor(
                [i / hidden_size for i in range(hidden_size)],
                dtype=module.time_maa_k.dtype,
                device=module.time_maa_k.device,
            )
            time_weight = time_weight[None, None, :]

            with torch.no_grad():
                module.time_maa_k.data = 1.0 - torch.pow(time_weight, ratio_1_to_almost0)
                module.time_maa_r.data = 1.0 - torch.pow(time_weight, ratio_1_to_almost0)


RWKV6_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.) This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module)
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.

    Parameters:
        config ([`Rwkv6Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

RWKV6_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`):
            `input_ids_length` = `sequence_length` if `past_key_values` is `None` else
            `past_key_values[0][0].shape[-2]` (`sequence_length` of input past key value states). Indices of input
            sequence tokens in the vocabulary. If `past_key_values` is used, only `input_ids` that do not have their
            past calculated should be passed as `input_ids`. Indices can be obtained using [`AutoTokenizer`]. See
            [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for details. [What are input
            IDs?](../glossary#input-ids)
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        state (tuple of five `torch.FloatTensor` of shape `(batch_size, hidden_size, num_hidden_layers)`, *optional*):
            If passed along, the model uses the previous state in all the blocks (which will give the output for the
            `input_ids` provided as if the model add `state_input_ids + input_ids` as context).
        use_cache (`bool`, *optional*):
            If set to `True`, the last state is returned and can be used to quickly generate the next logits.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare RWKV6 Model transformer outputting raw hidden-states without any specific head on top.",
    RWKV6_START_DOCSTRING,
)
class Rwkv6Model(Rwkv6PreTrainedModel, RwkvModel):
    def __init__(self, config):
        super().__init__(config)

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList([Rwkv6Block(config, layer_id=idx) for idx in range(config.num_hidden_layers)])
        self.ln_out = nn.LayerNorm(config.hidden_size)

        self.layers_are_rescaled = False
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(RWKV6_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=RwkvOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,  # noqa
        inputs_embeds: Optional[torch.FloatTensor] = None,
        state: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, RwkvOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # training is supportable with the Triton kernel
        # rwkv6 only support inference in huggingface.
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.training == self.layers_are_rescaled and (
            self.embeddings.weight.dtype == torch.float16 or self.embeddings.weight.dtype == torch.bfloat16
        ):
            self._rescale_layers()

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        if state is None:
            state = []
            head_size = self.config.head_size
            num_heads = self.config.attention_hidden_size // head_size
            state_attn_x = torch.zeros(
                (inputs_embeds.size(0), self.config.hidden_size, self.config.num_hidden_layers),
                dtype=inputs_embeds.dtype,
                requires_grad=False,
                device=inputs_embeds.device,
            ).contiguous()
            state_attn_kv = torch.zeros(
                (
                    inputs_embeds.size(0),
                    num_heads,
                    head_size,
                    head_size,
                    self.config.num_hidden_layers,
                ),
                dtype=torch.float32,
                requires_grad=False,
                device=inputs_embeds.device,
            ).contiguous()
            state_ffn_x = torch.zeros(
                (inputs_embeds.size(0), self.config.hidden_size, self.config.num_hidden_layers),
                dtype=inputs_embeds.dtype,
                requires_grad=False,
                device=inputs_embeds.device,
            ).contiguous()
            state.append(state_attn_x)
            state.append(state_attn_kv)
            state.append(state_ffn_x)

        hidden_states = inputs_embeds

        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for idx, block in enumerate(self.blocks):
            hidden_states, state, attentions = block(
                hidden_states, state=state, use_cache=use_cache, output_attentions=output_attentions,
            )
            if (
                self.layers_are_rescaled
                and self.config.rescale_every > 0
                and (idx + 1) % self.config.rescale_every == 0
            ):
                hidden_states = hidden_states / 2

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if output_attentions:
                all_self_attentions = all_self_attentions + (attentions,)

        hidden_states = self.ln_out(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return (hidden_states, state, all_hidden_states, all_self_attentions)

        return RwkvOutput(
            last_hidden_state=hidden_states,
            state=state,
            hidden_states=all_hidden_states,  # None
            attentions=all_self_attentions,  # None
        )


@add_start_docstrings(
    """
    The RWKV6 Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    RWKV6_START_DOCSTRING,
)
class Rwkv6ForCausalLM(Rwkv6PreTrainedModel, RwkvForCausalLM, GenerationMixin):
    _tied_weights_keys = ["head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.rwkv = Rwkv6Model(config)
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(RWKV6_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=RwkvCausalLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        state: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, RwkvCausalLMOutput]:
        # pylint: disable=line-too-long
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.rwkv(
            input_ids,
            inputs_embeds=inputs_embeds,
            state=state,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]

        logits = self.head(hidden_states)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return RwkvCausalLMOutput(
            loss=loss,
            logits=logits,
            state=outputs.state,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
