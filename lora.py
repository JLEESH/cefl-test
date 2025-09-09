import torch
import torch.nn as nn

class LinearLoRA(nn.Module):
    def __init__(self, linear):
        super().__init__()
        self.linear = linear
        #self.lora_params = lora_params
        self.lora_params = None

    def forward(self, x):
        if self.lora_params is not None:
            lora_out = self.lora_params @ x.reshape(x.shape[-1], x.shape[-2]) #self.lora_params @ x.reshape(x.shape[-1], x.shape[-2])
            lora_out = lora_out.reshape(x.shape)
            return lora_out + self.linear(x)
        else:
            #lora_out = 0
            return self.linear(x)
        # out = lora_out + self.linear(x)
        # return out
    
    def attach_w(self, w):
        self.lora_params = w
        
    def remove_w(self):
        self.lora_params = None