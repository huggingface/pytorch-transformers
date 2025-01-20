"""PyTorch xLSTM Model."""

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from ...generation import GenerationMixin
from ...modeling_utils import PreTrainedModel
from ...utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_torchdynamo_compiling,
    is_xlstm_available,
)


if is_xlstm_available():
    from xlstm.xlstm_large.model import (
        RMSNorm,
        mLSTMBlock,
        mLSTMStateType,
        soft_cap,
        xLSTMLargeConfig,
    )
else:
    mLSTMBlock = None
    xLSTMLargeConfig = None

from .configuration_xlstm import xLSTMConfig


_CHECKPOINT_FOR_DOC = "NX-AI/xLSTM-7b"
_CONFIG_FOR_DOC = "xLSTMConfig"


class xLSTMCache:
    """
    Cache / RNN State handler for xLSTM.

    Args:
        config: xLSTMConfig
        batch_size: int
        dtype: torch.dtype
        device: torch.device

    Attributes:
        seqlen_offset: int
        dtype: torch.dtype
    """

    def __init__(
        self, config: xLSTMConfig, batch_size: int, dtype: torch.dtype = torch.bfloat16, device: Optional[str] = None
    ):
        self.seqlen_offset = torch.tensor(0, dtype=torch.int64, device=device)
        self.dtype = dtype
        self.config = config
        self.rnn_state: mLSTMStateType = {
            layer: (
                torch.zeros(
                    [batch_size, config.num_heads, config.qk_head_dim, config.v_head_dim], dtype=dtype, device=device
                ),
                torch.zeros([batch_size, config.num_heads, config.qk_head_dim], dtype=dtype, device=device),
                torch.zeros([batch_size, config.num_heads, 1], dtype=dtype, device=device),
            )
            for layer in range(config.num_blocks)
        }
        self.rnn_state_initial = True

    def reset(self):
        self.rnn_state = {
            layer: (
                torch.zeros_like(self.rnn_state[layer][0]),
                torch.zeros_like(self.rnn_state[layer][1]),
                torch.zeros_like(self.rnn_state[layer][2]),
            )
            for layer in self.rnn_state
        }
        self.rnn_state_initial = True


class xLSTMPreTrainedModel(PreTrainedModel):
    """
    An abstract class for an interface to loading a pre-trained xLSTM model.
    """

    config_class = xLSTMConfig
    base_model_prefix = "backbone"
    _no_split_modules = ["xLSTMBlock"]
    supports_gradient_checkpointing = True
    _is_stateful = True

    def _init_weights(self, module):
        """Initialize the weights."""
        # TODO: this is a dummy, check with original settings.
        pass


@dataclass
class xLSTMOutput(ModelOutput):
    """
    Class for the xLSTM model outputs

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, embedding_dim)`):
            Sequence of hidden-states at the output of the last layer of the model.
        cache_params (`xLSTMCache`):
            The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
            avoid providing the old `input_ids`.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, embedding_dim)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

    """

    last_hidden_state: Optional[torch.FloatTensor]
    cache_params: Optional[xLSTMCache] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class xLSTMCausalLMOutput(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        cache_params (`xLSTMCache`):
            The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
            avoid providing the old `input_ids`.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, embedding_dim)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    cache_params: Optional[xLSTMCache] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


XLSTM_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`xLSTMConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

XLSTM_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`):
            Indices of input sequence tokens in the vocabulary.

            If `cache_params.seqlen_offset>0`, only `input_ids` that do not have their past calculated should be passed as
            `input_ids`.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, embedding_dim)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        cache_params (`xLSTMCache`, *optional*):
            If passed along, the model uses the previous state in all the blocks (which will give the output for the
            `input_ids` provided as if the model add `state_input_ids + input_ids` as context).
        use_cache (`bool`, *optional*):
            If set to `True`, the `cache_params` is returned and can be used to quickly generate the next logits.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            The position of the current input in the cache. This is used to ensure that the cache is correctly updated.
            If `cache_params` is passed, `cache_position` should also be passed.
        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
"""


@add_start_docstrings(
    "The bare xLSTM Model transformer outputting raw hidden-states without any specific head on top.",
    XLSTM_START_DOCSTRING,
)
class xLSTMModel(xLSTMPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embeddings = nn.Embedding(config.vocab_size, config.embedding_dim)
        # use config explicitly to mitigate unused variable tests
        xlstm_block_config = xLSTMLargeConfig(
            vocab_size=config.vocab_size,
            embedding_dim=config.embedding_dim,
            num_blocks=config.num_blocks,
            num_heads=config.num_heads,
            use_bias=config.use_bias,
            add_out_norm=config.add_out_norm,
            norm_eps=config.norm_eps,
            norm_reduction_force_float32=config.norm_reduction_force_float32,
            # mlstm_layer
            qk_dim_factor=config.qk_dim_factor,
            v_dim_factor=config.v_dim_factor,
            # mlstm backend
            chunkwise_kernel=config.chunkwise_kernel,
            sequence_kernel=config.sequence_kernel,
            step_kernel=config.step_kernel,
            mode=config.mode,
            chunk_size=config.chunk_size,
            return_last_states=config.return_last_states,
            autocast_kernel_dtype=config.autocast_kernel_dtype,
            eps=config.eps,
            inference_state_dtype=config.inference_state_dtype,
            # feedforward
            ffn_proj_factor=config.ffn_proj_factor,
            ffn_round_up_to_multiple_of=config.ffn_round_up_to_multiple_of,
            # capping
            gate_soft_cap=config.gate_soft_cap,
            output_logit_soft_cap=config.output_logit_soft_cap,
            weight_mode=config.weight_mode,
        )

        self.blocks = nn.ModuleList([mLSTMBlock(xlstm_block_config) for _ in range(config.num_blocks)])

        self.gradient_checkpointing = False
        self.out_norm = RMSNorm(config.embedding_dim, eps=config.norm_eps)
        # Initialize weights and apply final processing
        self.post_init()

    def _init_weights(self, module):
        # Not implemented yet - use pretrained model.
        pass

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, new_embedding):
        self.embeddings = new_embedding

    @add_start_docstrings_to_model_forward(XLSTM_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=xLSTMOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        cache_params: Optional[xLSTMCache] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[Tuple, xLSTMOutput]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):  # ^ is python for xor
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        if use_cache:
            if cache_params is None:
                cache_params = xLSTMCache(
                    self.config, inputs_embeds.size(0), device=inputs_embeds.device, dtype=inputs_embeds.dtype
                )
        else:
            cache_params = None

        hidden_states = inputs_embeds

        if (
            not self.training
            and self.config.max_inference_chunksize < hidden_states.shape[1]
            and not output_hidden_states
        ):
            all_hidden_states = None
            offset = 0
            with torch.no_grad():
                if cache_params is None:
                    cache_params = xLSTMCache(config=self.config, batch_size=hidden_states.shape[0])
                final_state = torch.zeros_like(hidden_states)
                while offset < hidden_states.shape[1]:
                    hidden_states_chunk = hidden_states[
                        :, offset : min(offset + self.config.max_inference_chunksize, hidden_states.shape[1])
                    ]
                    for i, xlstm_block in enumerate(self.blocks):
                        hidden_states_chunk, rnn_state = xlstm_block(
                            hidden_states_chunk,
                            state=cache_params.rnn_state[i],
                        )
                        for state_idx in range(len(cache_params.rnn_state[i])):
                            local_rnn_state = rnn_state[state_idx]
                            local_rnn_state = rnn_state[state_idx]
                            cache_params.rnn_state[i][state_idx].copy_(local_rnn_state)
                        cache_params.rnn_state_initial = False
                    final_state[
                        :, offset : min(offset + self.config.max_inference_chunksize, hidden_states.shape[1])
                    ] = hidden_states_chunk
                    offset += self.config.max_inference_chunksize
                hidden_states = final_state
        else:
            all_hidden_states = () if output_hidden_states else None
            for i, xlstm_block in enumerate(self.blocks):
                if self.gradient_checkpointing and self.training:
                    hidden_states, rnn_state = self._gradient_checkpointing_func(
                        xlstm_block.__call__,
                        hidden_states,
                        cache_params.rnn_state[i] if cache_params is not None else None,
                    )
                else:
                    hidden_states, rnn_state = xlstm_block(
                        hidden_states,
                        state=cache_params.rnn_state[i] if cache_params is not None else None,
                    )
                if cache_params:
                    for state_idx in range(len(cache_params.rnn_state[i])):
                        local_rnn_state = rnn_state[state_idx]
                        local_rnn_state = rnn_state[state_idx]
                        cache_params.rnn_state[i][state_idx].copy_(local_rnn_state)
                    cache_params.rnn_state_initial = False

                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

        if use_cache:
            cache_params.seqlen_offset += inputs_embeds.shape[1]

        hidden_states = self.out_norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, cache_params, all_hidden_states] if v is not None)

        return xLSTMOutput(
            last_hidden_state=hidden_states,
            cache_params=cache_params if use_cache else None,
            hidden_states=all_hidden_states,
        )


@add_start_docstrings(
    """
    The xLSTM Model transformer with a language modeling head on top (linear layer with weights not tied to the input
    embeddings).
    """,
    XLSTM_START_DOCSTRING,
)
class xLSTMForCausalLM(xLSTMPreTrainedModel, GenerationMixin):
    _tied_weights_keys = []

    def __init__(self, config):
        super().__init__(config)
        self.backbone = xLSTMModel(config)
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        # self.register_load_state_dict_pre_hook(self.load_hook)
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        return self.backbone.set_input_embeddings(new_embeddings)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        inputs_embeds=None,
        use_cache=None,
        cache_params: Optional[xLSTMCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # Overwritten -- uses `cache_params` as opposed to `past_key_values`
        # Does not support using additional convolution states via inputs_embeds
        # as opposed to Mamba, currently.
        if use_cache:
            # `cache_position` should have been initialized in `generate`
            if cache_position is None:
                raise ValueError(
                    "`cache_position` should not be None as it should have been initialized in "
                    "`model.generate`, you are responsible for passing in a valid `cache_position` if "
                    "you are calling `prepare_inputs_for_generation` directly with `use_cache=True`"
                )
            # If the first cache position is non-zero, we assume we are in generation mode.
            # Thus, the cache_params state is assumed to be the state before the last token
            # (lastly generated token), and all previous tokens are already ingested.
            # This should as well support generation from scratch with the [BOS] token inserted first.

            # if is_torchdynamo_compiling() or cache_position[0] > 0:
            if cache_params is not None:
                input_ids = input_ids[:, -1:]
                if inputs_embeds is not None:
                    inputs_embeds = inputs_embeds[:, -1:]

        attention_mask = None

        if inputs_embeds is not None and cache_params is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "cache_params": cache_params,
                "use_cache": use_cache,
                "cache_position": cache_position,
            }
        )
        return model_inputs

    @add_start_docstrings_to_model_forward(XLSTM_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=xLSTMCausalLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_params: Optional[xLSTMCache] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,  # for now we need this for generation
    ) -> Union[Tuple, xLSTMCausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        inserted_bos_token = False
        if (
            (cache_params is None or cache_params.rnn_state_initial)
            and self.config.force_bos_token_insert
            and input_ids is not None
        ):
            if not is_torchdynamo_compiling():
                if bool(torch.all(input_ids[0, 0] != self.config.bos_token_id).cpu()):
                    input_ids = torch.cat(
                        [self.config.bos_token_id + input_ids.new_zeros([input_ids.shape[0], 1]), input_ids], dim=1
                    )
                    inserted_bos_token = True

        xlstm_outputs = self.backbone(
            input_ids,
            cache_params=cache_params,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=use_cache,
            cache_position=cache_position,
            attention_mask=attention_mask,
        )
        hidden_states = xlstm_outputs[0]

        if inserted_bos_token:
            hidden_states = hidden_states[:, 1:]
            if hasattr(xlstm_outputs, "hidden_states"):
                xlstm_outputs_mod = (
                    hidden_states,
                    xlstm_outputs.cache_params,
                    tuple(hidden_state[:, 1:] for hidden_state in xlstm_outputs.hidden_states)
                    if xlstm_outputs.hidden_states is not None
                    else None,
                )
            elif len(xlstm_outputs) == 3:
                xlstm_outputs_mod = (
                    hidden_states,
                    xlstm_outputs[1],
                    tuple(hidden_state[:, 1:] for hidden_state in xlstm_outputs[2])
                    if xlstm_outputs[2] is not None
                    else None,
                )
            elif len(xlstm_outputs) == 2:
                xlstm_outputs_mod = (hidden_states, xlstm_outputs[1])
            elif len(xlstm_outputs) == 1:
                xlstm_outputs_mod = (hidden_states,)
            if isinstance(xlstm_outputs, xLSTMOutput):
                xlstm_outputs = xLSTMOutput(*xlstm_outputs_mod)
            else:
                xlstm_outputs = xlstm_outputs_mod

        logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype)).float()

        if not self.training and self.config.max_inference_chunksize < logits.shape[1]:
            offset = 0
            with torch.no_grad():
                while offset < logits.shape[1]:
                    logits[:, offset : min(offset + self.config.max_inference_chunksize, logits.shape[1])] = soft_cap(
                        logits[:, offset : min(offset + self.config.max_inference_chunksize, logits.shape[1])],
                        self.config.output_logit_soft_cap,
                    )
                    offset += self.config.max_inference_chunksize
        else:
            logits = soft_cap(logits, self.config.output_logit_soft_cap)

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
            output = (logits,) + xlstm_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return xLSTMCausalLMOutput(
            loss=loss,
            logits=logits,
            cache_params=xlstm_outputs.cache_params,
            hidden_states=xlstm_outputs.hidden_states,
        )
