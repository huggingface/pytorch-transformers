from transformers.models.llama.modeling_llama import LlamaModel


# Check that we can correctly change the prefix (here add Text part at the end of the name)
class Multimodal1TextModel(LlamaModel):
    pass
