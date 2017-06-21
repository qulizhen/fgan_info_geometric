# coding: utf-8

# In[ ]:

import torch
from torch.autograd.function import Function

class LeastSquareFunc(Function):

    def forward(self, input):
        mask_le_mone = input.le(-1).type_as(input)
        self.mask_ge_one = input.ge(1).type_as(input)
        index_gt_mone = input.gt(-1).type_as(input)
        index_lt_one = input.lt(1).type_as(input)
        self.mask_mone_one = index_lt_one * index_gt_mone
        mone = input.new().resize_as_(input).fill_(-1)
        mone= mone * mask_le_mone
        between_one = torch.pow(1 + input, 2) -1
        between_one = between_one * self.mask_mone_one
        ge_one = input * 4 - 1
        ge_one = ge_one * self.mask_ge_one
        between_one = mone + between_one
        ge_one = between_one + ge_one
        self.input = input
        return ge_one

    def backward(self, grad_output):
        grad_between = (self.input * 2 + 2) * self.mask_mone_one
        grad_ge_one = 4 * self.mask_ge_one
        grad_input = grad_output * (grad_between + grad_ge_one)

        return grad_input





class LeastSquare(torch.nn.Module):
    def __init__(self):
        super(LeastSquare, self).__init__()

    def forward(self, input):
        return LeastSquareFunc()(input)
