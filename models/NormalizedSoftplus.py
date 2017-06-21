
# coding: utf-8

# In[ ]:

import torch
import math

class NSoftPlus(torch.nn.Module):
    def __init__(self):
        super(NSoftPlus, self).__init__()
        self.log2 = math.log(2)
    
    def forward(self, x):
        return torch.log(torch.exp(x).add(1))/self.log2 - 1.0
