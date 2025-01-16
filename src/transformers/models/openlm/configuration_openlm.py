# Follows OLMo's HF template

"""
OpenLM configuration
"""

from transformers import AutoConfig, PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class OpenLMConfig(PretrainedConfig):
    model_type = "openlm"

    def __init__(self,
        dim: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
        vocab_size: int = 2048,
        norm_eps: float = 1e-5,
        seq_len: int = 2048,
        post_embed_norm: bool = False,
        weight_tying: bool = False,
        model_norm: str = "gain_only_lp_layer_norm",
        attn_name: str = "auto",
        apply_qk_norm: bool = False,
        moe_loss_weight: float = 0.1,
        moe_capacity_factor: float = 1.25,
        moe_expert_model_parallelism: bool = False,
        moe_weight_parallelism: bool = False,
        moe_num_experts: int = 8,
        moe_top_k: int = 2,
        moe_freq: int = 0,
        positional_embedding_type: str = "rotary",
        ffn_type: str = "swiglu",
        **kwargs
    ):
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.vocab_size = vocab_size
        self.norm_eps = norm_eps
        self.seq_len = seq_len
        self.post_embed_norm = post_embed_norm
        self.weight_tying = weight_tying
        self.model_norm = model_norm
        self.attn_name = attn_name
        self.apply_qk_norm = apply_qk_norm
        self.moe_loss_weight = moe_loss_weight
        self.moe_capacity_factor = moe_capacity_factor
        self.moe_expert_model_parallelism = moe_expert_model_parallelism
        self.moe_weight_parallelism = moe_weight_parallelism
        self.moe_num_experts = moe_num_experts
        self.moe_top_k = moe_top_k
        self.moe_freq = moe_freq
        self.positional_embedding_type = positional_embedding_type
        self.ffn_type = ffn_type

        super().__init__(**kwargs)
