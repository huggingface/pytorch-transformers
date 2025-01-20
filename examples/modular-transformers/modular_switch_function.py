# Note that llama and cohere have different definitions for rotate_half
from transformers.models.cohere.modeling_cohere import rotate_half  # noqa
from transformers.models.llama.modeling_llama import LlamaAttention


# When following LlamaAttention dependencies, we will grab the function `rotate_half` defined
# in `modeling_llama.py`. But here we imported it explicitly from Cohere, so it should use Cohere's
# definition instead
class SwitchFunctionAttention(LlamaAttention):
    pass
