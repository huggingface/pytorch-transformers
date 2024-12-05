from transformers.models.clip.modeling_clip import CLIPEncoderLayer


# Check if we can correctly grab dependencies with correct naming from all UPPERCASE old model
class FromUppercaseModelEncoderLayer(CLIPEncoderLayer):
    pass
