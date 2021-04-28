from transformers import DetrConfig, DetrForPanopticSegmentation
import torch

config = DetrConfig(masks=True)
model = DetrForPanopticSegmentation(config)

pixel_values = torch.randn([2, 3, 800, 800])
pixel_mask = torch.randint(0,1, (2, 800, 800))
outputs = model(pixel_values, pixel_mask)