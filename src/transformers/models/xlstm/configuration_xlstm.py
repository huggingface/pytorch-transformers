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
"""XLSTM configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import is_xlstm_available, logging


if is_xlstm_available():
    from xlstm.xlstm_large.model import (
        BackendModeType,
        ChunkwiseKernelType,
        DtypeType,
        SequenceKernelType,
        StepKernelType,
        WeightModeType,
        round_up_to_next_multiple_of,
        xLSTMLargeConfig,
    )
else:
    xLSTMLargeConfig = None
    BackendModeType = None
    ChunkwiseKernelType = None
    DtypeType = None
    SequenceKernelType = None
    StepKernelType = None
    WeightModeType = None


logger = logging.get_logger(__name__)


class xLSTMConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`xLSTM`]. It is used to instantiate a xLSTM
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the xLSTM-7b [NX-AI/xLSTM-7b](https://huggingface.co/NX-AI/xLSTM-7b) model.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (int, optional, *optional*, defaults to 50304):
            Vocabulary size of the xLSTM model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`xLSTMModel`]. Defaults to the GPT2-NeoX tokenizer size.
        embedding_dim (int, optional, *optional*, defaults to 4096):
            Dimensionality of the embeddings or hidden states.
        num_blocks (int, optional, *optional*, defaults to 32):
            Number of blocks of the xLSTM model.
        num_heads (int, optional, *optional*, defaults to 8):
            Number of heads for the xLSTM Layer/Cell.
        use_bias (bool, optional, *optional*, defaults to `False`):
            Whether to use biases in the xLSTM model.
        norm_reduction_force_float32 (bool, optional, *optional*, defaults to `True`):
            Whether to force the float32 norm reduction op to be done in fp32 precision.
        tie_word_embeddings (bool, optional, *optional*, defaults to `False`):
            Whether to tie word embeddings to the lm head weights.
        add_out_norm (bool, optional, *optional*, defaults to `True`):
            Whether to add an output norm after the blocks before the LMHead.
        norm_eps (float, optional, *optional*, defaults to 1e-06):
            Norm eps for RMSNorm and Layer Norm.
        qk_dim_factor (float, optional, *optional*, defaults to 0.5):
            Scale factor for the query and key dimension.
        v_dim_factor (float, optional, *optional*, defaults to 1.0):
            Scale factor for the value dimension.
        chunkwise_kernel (ChunkwiseKernelType, optional, *optional*, defaults to `"chunkwise--native_autograd"`):
            Kernel type for chunkwise processing mode.
        sequence_kernel (SequenceKernelType, optional, *optional*, defaults to `"native_sequence__native"`):
            Kernel type for sequence processing mode.
        step_kernel (StepKernelType, optional, *optional*, defaults to `"native"`):
            Kernel type for step processing mode.
        mode (BackendModeType, optional, *optional*, defaults to `"inference"`):
            Operation mode (inference is needed for generation).
        chunk_size (int, optional, *optional*, defaults to 64):
            Internal chunk size.
        return_last_states (bool, optional, *optional*, defaults to `True`):
            If to return the last states / cache internally. Needed as True for generation.
        autocast_kernel_dtype (DtypeType, optional, *optional*, defaults to `"bfloat16"`):
            Kernel dtype for the states.
        eps (float, optional, *optional*, defaults to 1e-06):
            Epsilon for the mLSTM cell post norm.
        inference_state_dtype (DtypeType, optional, *optional*, defaults to `"float32"`):
            Kernel dtype for states in inference.
        ffn_proj_factor (float, optional, *optional*, defaults to 2.667):
            Size factor of the post-up projection gated Feed Forward network.
        ffn_round_up_to_multiple_of (int, optional, *optional*, defaults to 64):
            Size factor round value of the post-up projection gated Feed Forward network.
        gate_soft_cap (float, optional, *optional*, defaults to 15.0):
            Gate soft cap scale.
        output_logit_soft_cap (float, optional, *optional*, defaults to 30.0):
            Output logit soft cap scale.
        weight_mode (`Literal`, *optional*, defaults to `"single"`):
            Whether parallel linear layers are separated or fused (single).
        use_cache (bool, optional, *optional*, defaults to `True`):
            Whether to use the cache (xLSTMCache).
        pad_token_id (int, optional, *optional*, defaults to 1):
            Pad token id needed for generation.
        bos_token_id (int, optional, *optional*, defaults to 0):
            BOS token id needed for generation.
        eos_token_id (int, optional, *optional*, defaults to 2):
            EOS token id needed for generation.
        force_bos_token_insert (bool, optional, *optional*, defaults to `True`):
            Whether to force the insertion of a BOS token for prompting.
        max_inference_chunksize (int, optional, *optional*, defaults to 16384):
            Limit the chunk size for inference to save memory.

    Example:

    ```python
    >>> from transformers import xLSTMConfig, xLSTMModel

    >>> # Initializing a xLSTM configuration
    >>> configuration = xLSTMConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = xLSTMModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "xlstm"

    def __init__(
        self,
        vocab_size: int = 50304,
        embedding_dim: int = 4096,
        num_blocks: int = 32,
        num_heads: int = 8,
        use_bias: bool = False,
        norm_reduction_force_float32: bool = True,
        tie_word_embeddings: bool = False,
        add_out_norm: bool = True,
        norm_eps: float = 1e-6,
        # mlstm_layer
        qk_dim_factor: float = 0.5,
        v_dim_factor: float = 1.0,
        # mlstm backend
        chunkwise_kernel: ChunkwiseKernelType = "chunkwise--native_autograd",
        sequence_kernel: SequenceKernelType = "native_sequence__native",
        step_kernel: StepKernelType = "native",
        # nedded to enable generation
        mode: BackendModeType = "inference",
        chunk_size: int = 64,
        # needed to be true for generation
        return_last_states: bool = True,
        autocast_kernel_dtype: DtypeType = "bfloat16",
        eps: float = 1e-6,
        inference_state_dtype: DtypeType = "float32",
        # feedforward
        ffn_proj_factor: float = 2.667,
        ffn_round_up_to_multiple_of: int = 64,
        # capping
        gate_soft_cap: float = 15.0,
        output_logit_soft_cap: float = 30.0,
        # weights
        weight_mode: WeightModeType = "single",
        # HF interface
        use_cache: bool = True,
        pad_token_id: int = 1,
        bos_token_id: int = 0,
        eos_token_id: int = 2,
        force_bos_token_insert: bool = True,
        max_inference_chunksize: int = 16384,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.use_bias = use_bias
        self.tie_word_embeddings = tie_word_embeddings
        self.add_out_norm = add_out_norm
        self.norm_eps = norm_eps
        self.norm_reduction_force_float32 = norm_reduction_force_float32
        # mlstm_layer
        self.qk_dim_factor = qk_dim_factor
        self.v_dim_factor = v_dim_factor
        # mlstm backend
        self.chunkwise_kernel = chunkwise_kernel
        self.sequence_kernel = sequence_kernel
        self.step_kernel = step_kernel
        self.mode = mode
        self.chunk_size = chunk_size
        self.return_last_states = return_last_states
        self.autocast_kernel_dtype = autocast_kernel_dtype
        self.eps = eps
        self.inference_state_dtype = inference_state_dtype
        # feedforward
        self.ffn_proj_factor = ffn_proj_factor
        self.ffn_round_up_to_multiple_of = ffn_round_up_to_multiple_of
        # capping
        self.gate_soft_cap = gate_soft_cap
        self.output_logit_soft_cap = output_logit_soft_cap
        self.weight_mode = weight_mode

        self.use_cache = use_cache
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.force_bos_token_insert = force_bos_token_insert
        self.max_inference_chunksize = max_inference_chunksize

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def qk_dim(self):
        return round_up_to_next_multiple_of(
            self.embedding_dim * self.qk_dim_factor,
            multiple_of=64,
        )

    @property
    def v_dim(self):
        return round_up_to_next_multiple_of(
            self.embedding_dim * self.v_dim_factor,
            multiple_of=64,
        )

    @property
    def qk_head_dim(self):
        return self.qk_dim // self.num_heads

    @property
    def v_head_dim(self):
        return self.v_dim // self.num_heads

    def to_xlstm_block_config(self):
        return xLSTMLargeConfig(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            num_blocks=self.num_blocks,
            num_heads=self.num_heads,
            use_bias=self.use_bias,
            add_out_norm=self.add_out_norm,
            norm_eps=self.norm_eps,
            norm_reduction_force_float32=self.norm_reduction_force_float32,
            # mlstm_layer
            qk_dim_factor=self.qk_dim_factor,
            v_dim_factor=self.v_dim_factor,
            # mlstm backend
            chunkwise_kernel=self.chunkwise_kernel,
            sequence_kernel=self.sequence_kernel,
            step_kernel=self.step_kernel,
            mode=self.mode,
            chunk_size=self.chunk_size,
            return_last_states=self.return_last_states,
            autocast_kernel_dtype=self.autocast_kernel_dtype,
            eps=self.eps,
            inference_state_dtype=self.inference_state_dtype,
            # feedforward
            ffn_proj_factor=self.ffn_proj_factor,
            ffn_round_up_to_multiple_of=self.ffn_round_up_to_multiple_of,
            # capping
            gate_soft_cap=self.gate_soft_cap,
            output_logit_soft_cap=self.output_logit_soft_cap,
            weight_mode=self.weight_mode,
        )
