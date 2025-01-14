# Copyright 2020 The HuggingFace Team. All rights reserved.
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
"""
Integration with MsT
"""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from ..modeling_outputs import CausalLMOutputWithPast
from ..models.gemma2.modeling_gemma2 import Gemma2ForCausalLM, Gemma2MLP
from ..models.llama.modeling_llama import LlamaForCausalLM, LlamaMLP
from ..models.mistral.modeling_mistral import MistralForCausalLM, MistralMLP
from ..models.qwen2.modeling_qwen2 import Qwen2ForCausalLM, Qwen2MLP


def minis_mlp_forward(self, x):
    bsz, q_len, _ = x.size()
    chunk_size = self.hidden_size

    x_list = list(x.split(chunk_size, dim=1))

    output_list = [None for _ in range(len(x_list))]

    for i in range(len(x_list)):
        x = x_list[i]
        output_list[i] = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    down_proj = torch.cat(output_list, dim=1)

    return down_proj


class _LM_head(torch.autograd.Function):
    @classmethod
    def forward(cls, ctx, hidden_states, indices, weights):
        logits = F.linear(hidden_states, weights).float()
        loss_fct = torch.nn.CrossEntropyLoss(reduction="sum")
        loss_i = loss_fct(logits, indices)

        weights.count += 1

        ctx.save_for_backward(hidden_states, indices, weights)

        return loss_i

    @classmethod
    def backward(cls, ctx, dneg_logprobs):
        # load saved tensors
        hidden_states, indices, weights = ctx.saved_tensors
        weights.count -= 1

        ignore_index = -100
        mask = indices != ignore_index
        reverse_mask = indices == ignore_index

        if not mask.any():
            # If all indices are -100, return zero gradients
            return torch.zeros_like(hidden_states), None, None

        logits = F.linear(hidden_states, weights).float()

        grad_input = F.softmax(logits, dim=-1)
        grad_input[mask, indices[mask]] -= 1
        # grad_input[mask] /= batch_size
        grad_input[reverse_mask] = 0
        grad_input *= dneg_logprobs
        grad_input = grad_input.to(hidden_states.dtype)

        if hasattr(weights, "grad") and weights.grad is not None:
            torch.addmm(
                weights.grad,
                grad_input.T,
                hidden_states,
                out=weights.grad,
            )
        else:
            weights.grad = grad_input.T @ hidden_states

        logits = None
        grad_input = grad_input @ weights

        if weights.count == 0:
            return grad_input, None, weights.grad
        else:
            return grad_input, None, None


class LMheadWarpper(nn.Module):
    def __init__(self, original_weight=None):
        super().__init__()
        self.LM_head_weight = original_weight
        self.LM_head = _LM_head.apply
        self.LM_head_weight.count = 0

    def forward(self, hidden_states, labels):
        loss = self.LM_head(hidden_states, labels, self.LM_head_weight)
        return loss


def minis_processing(hidden_states, labels, lm_head, mini_s):
    bsz, q_len, hidden_size = hidden_states.size()
    tmp = q_len // mini_s

    if labels is None:
        hidden_states = hidden_states[..., -1:, :]
        logits = lm_head(hidden_states)
        logits = logits.float()
        return logits, None

    hidden_states = hidden_states[..., :-1, :]

    labels = labels[..., 1:].contiguous()
    labels = labels.to(hidden_states.device)

    LMhead = LMheadWarpper(lm_head.weight)

    loss = None
    for i in range(mini_s):
        shift_hidden_states = hidden_states[..., i * tmp : (i + 1) * tmp, :].contiguous()
        shift_hidden_states = shift_hidden_states.view(-1, hidden_size)
        shift_labels = labels[..., i * tmp : (i + 1) * tmp].contiguous()
        shift_labels = shift_labels.view(-1)

        loss_i = LMhead(shift_hidden_states, shift_labels)

        # if not torch.isnan(loss_i):
        if loss is None:
            loss = loss_i
        else:
            loss = loss + loss_i
        # print(i, loss_i, loss)

    loss = loss / torch.sum(torch.ne(labels, -100))
    return None, loss


def minis_CausalLM_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> Union[Tuple, CausalLMOutputWithPast]:
    r"""
    Args:
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

    Returns:

    Example:

    ```python
    >>> from transformers import AutoTokenizer, LlamaForCausalLM

    >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

    >>> prompt = "Hey, are you conscious? Can you talk to me?"
    >>> inputs = tokenizer(prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
    ```"""
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )

    hidden_states = outputs[0]

    minis = math.ceil(self.config.vocab_size / self.config.hidden_size)

    logits, loss = minis_processing(hidden_states, labels, self.lm_head, minis)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def replace_with_minis():
    LlamaMLP.forward = minis_mlp_forward
    Gemma2MLP.forward = minis_mlp_forward
    Qwen2MLP.forward = minis_mlp_forward
    MistralMLP.forward = minis_mlp_forward
    LlamaForCausalLM.forward = minis_CausalLM_forward
    Gemma2ForCausalLM.forward = minis_CausalLM_forward
    Qwen2ForCausalLM.forward = minis_CausalLM_forward
    MistralForCausalLM.forward = minis_CausalLM_forward


def replace_model_with_minis(model):
    model.gradient_checkpointing_enable()
    replace_with_minis()
