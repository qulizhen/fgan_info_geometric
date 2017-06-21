from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import torch
import torch.nn as nn
from models.Matsushita import MatsushitaTransform,MatsushitaTransformOne,NormalizedMatsushita

class MLP_G(nn.Module):
    def __init__(self, isize, nz, nc, ngf, ngpu, hidden_activation='', mu=0.5, last_layer='none'):
        super(MLP_G, self).__init__()
        self.ngpu = ngpu
        if hidden_activation == 'matsu':
            first_activation = MatsushitaTransform()
            second_activation = MatsushitaTransform()
            third_activation = MatsushitaTransform()
        elif hidden_activation == 'matsu1':
            first_activation = MatsushitaTransformOne()
            second_activation = MatsushitaTransformOne()
            third_activation = MatsushitaTransformOne()
        elif hidden_activation == 'elu':
            first_activation = nn.ELU(alpha=mu)
            second_activation = nn.ELU(alpha=mu)
            third_activation = nn.ELU(alpha=mu)
        elif hidden_activation == 'murelu':
            first_activation = NormalizedMatsushita(mu)
            second_activation = NormalizedMatsushita(mu)
            third_activation = NormalizedMatsushita(mu)
        else:
            first_activation = nn.ReLU(False)
            second_activation = nn.ReLU(False)
            third_activation = nn.ReLU(False)
        main = nn.Sequential(
            # Z goes into a linear of size: ngf
            nn.Linear(nz, ngf),
            first_activation,
            nn.Linear(ngf, ngf),
            second_activation,
            nn.Linear(ngf, ngf),
            third_activation,
            nn.Linear(ngf, nc * isize * isize),
        )

        if last_layer == 'tanh':
            main.add_module('final.{0}.tanh'.format(nc),
                        nn.Tanh())
        elif last_layer == 'sigmoid':
            main.add_module('final.{0}.sigmoid'.format(nc),
                        nn.Sigmoid())

        self.main = main
        self.nc = nc
        self.isize = isize
        self.nz = nz

    def forward(self, input):
        input = input.view(input.size(0), input.size(1))
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output.view(output.size(0), self.nc, self.isize, self.isize)


class MLP_D(nn.Module):
    def __init__(self, isize, nz, nc, ndf, ngpu):
        super(MLP_D, self).__init__()
        self.ngpu = ngpu

        main = nn.Sequential(
            # Z goes into a linear of size: ndf
            nn.Linear(nc * isize * isize, ndf),
            nn.ReLU(False),
            nn.Linear(ndf, ndf),
            nn.ReLU(False),
            nn.Linear(ndf, ndf),
            nn.ReLU(False),
            nn.Linear(ndf, 1),
        )
        self.main = main
        self.nc = nc
        self.isize = isize
        self.nz = nz

    def forward(self, input):
        input = input.view(input.size(0),
                           input.size(1) * input.size(2) * input.size(3))
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        #output = output.mean(0)
        return output.view(input.size(0))
