"""
Here, because clip is not consistent with the use of the "Text" and "Vision" prefixes, we cannot simply use
```
class Multimodal2VisionModel(CLIPVisionModel):
    pass
```
with the hope that all dependencies will be renamed as `Multimodal2VisionClass`. For this reason, if we want consistency and
use the "Vision" part everywhere, we need to overwrite the intermediate classes and add the prefix everytime.
This adds noise to the modular, but is unfortunately unavoidable.
"""

from torch import nn

from transformers.models.clip.modeling_clip import (
    CLIPMLP,
    CLIPAttention,
    CLIPEncoder,
    CLIPEncoderLayer,
    CLIPFlashAttention2,
    CLIPPreTrainedModel,
    CLIPSdpaAttention,
    CLIPVisionModel,
    CLIPVisionTransformer,
)
from transformers.utils import add_start_docstrings


class Multimodal2VisionAttention(CLIPAttention):
    pass


# Check that adding the second base class correctly set the parent, even though in clip it does not have the "Vision" part
class Multimodal2VisionSdpaAttention(CLIPSdpaAttention, Multimodal2VisionAttention):
    pass


# Check that adding the second base class correctly set the parent, even though in clip it does not have the "Vision" part
class Multimodal2VisionFlashAttention2(CLIPFlashAttention2, Multimodal2VisionAttention):
    pass


MULTIMODAL2_VISION_ATTENTION_CLASSES = {
    "eager": Multimodal2VisionAttention,
    "sdpa": Multimodal2VisionSdpaAttention,
    "flash_attention_2": Multimodal2VisionFlashAttention2,
}


class Multimodal2VisionMLP(CLIPMLP):
    pass


class Multimodal2VisionEncoderLayer(CLIPEncoderLayer):
    def __init__(self, config):
        super().__init__()
        self.self_attn = MULTIMODAL2_VISION_ATTENTION_CLASSES[config._attn_implementation](config)
        self.mlp = Multimodal2VisionMLP(config)


class Multimodal2VisionEncoder(CLIPEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList([Multimodal2VisionEncoderLayer(config) for _ in range(config.num_hidden_layers)])


# Finally here the `Vision` part was correct in CLIP, but we still need to tell it that the encoder arg should use it as well
class Multimodal2VisionTransformer(CLIPVisionTransformer):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = Multimodal2VisionEncoder(config)


class Multimodal2VisionPreTrainedModel(CLIPPreTrainedModel):
    def _init_weights(self, module):
        if isinstance(module, Multimodal2VisionMLP):
            pass


MULTIMODAL2_VISION_START_DOCSTRING = "doc"


# Here the only arg `self.vision_model = CLIPVisionTransformer(config)` in CLIPVisionModel already has the "Vision" part, so
# no need to overwrite it, it will look for `Multimodal2VisionTransformer` which has already being redefined above
# Note: we may want to redefine decorator as well for full consistency, as CLIP does not use "CLIP_VISION_START_DOCSTRING" but only
# "CLIP_START_DOCSTRING"
@add_start_docstrings("New doc", MULTIMODAL2_VISION_START_DOCSTRING)
class Multimodal2VisionModel(CLIPVisionModel, Multimodal2VisionPreTrainedModel):
    _no_split_modules = ["Multimodal2VisionEncoderLayer"]
