import torch
import torch.nn as nn

class LinearLoRA(nn.Module):
    def __init__(self, linear, lora_params):
        super().__init__()
        self.linear = linear
        self.lora_params = lora_params
    
    def forward(self, x):
        lora_out = self.lora_params @ x.reshape(x.shape[-1], x.shape[-2])
        lora_out = lora_out.reshape(x.shape)
        return self.linear(x) 