import torch
import torch.utils.checkpoint

from transformers.models.blip.image_processing_blip import BlipImageProcessor


class ImgprocModelImageProcessor(BlipImageProcessor):
    def new_image_processing_method(self, pixel_values: torch.FloatTensor):
        return pixel_values / 2
