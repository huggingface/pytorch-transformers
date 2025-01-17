# Note that zamba does not have the `apply_rotary_pos_emb` function!
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from transformers.models.zamba.modeling_zamba import ZambaAttention


# When following ZambaAttention dependencies, the function `apply_rotary_pos_emb` is not present
# by default as it is absent from the class definition (and the file altogether).
# Note that this syntax should be able to add both `apply_rotary_pos_emb` as imported directly, but
# `rotate_half` as well as a dependency from the imported function!!
class TestAttention(ZambaAttention):
    def __init__(self):
        pass

    def forward(self):
        _ = apply_rotary_pos_emb(1, 1, 1, 1)
