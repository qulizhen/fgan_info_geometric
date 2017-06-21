
# coding: utf-8

# In[ ]:

import torch

class MatsushitaTransform(torch.nn.Module):
    def __init__(self):
        super(MatsushitaTransform, self).__init__()
    
    def forward(self, x):
        return (x + torch.sqrt(1 + torch.pow(x, 2))) * 0.5


class MatsushitaTransformOne(torch.nn.Module):
    def __init__(self):
        super(MatsushitaTransformOne, self).__init__()

    def forward(self, x):
        return x + torch.sqrt(1 + torch.pow(x, 2)) - 1


class MatsushitaLinkFunc(torch.nn.Module):
    def __init__(self):
        super(MatsushitaLinkFunc, self).__init__()

    def forward(self, x):
        return 0.5 * (1 + 0.5 * x / torch.sqrt(0.25 * torch.pow(x, 2) + 1))


class NormalizedMatsushita(torch.nn.Module):
    def __init__(self, mu=0):
        super(NormalizedMatsushita, self).__init__()
        self.mu = mu
        self.one_minus_mu = 1 - self.mu
        self.one_minus_mu_square = self.one_minus_mu * self.one_minus_mu

    def forward(self, x):
        return 0.5 * (x + torch.sqrt(self.one_minus_mu_square + torch.pow(x, 2)) - self.one_minus_mu)