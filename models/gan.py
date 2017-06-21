from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import torch
import torch.nn as nn
from models.Matsushita import NormalizedMatsushita,MatsushitaLinkFunc
from models.NormalizedSoftplus import NSoftPlus
from models.LeastSquare import LeastSquare

class GAN_G(nn.Module):
    def __init__(self, isize, nz, nc, ngf, ngpu, hidden_activation, mu, last_layer='sigmoid'):
        super(GAN_G, self).__init__()
        self.ngpu = ngpu
        if hidden_activation == 'elu':
            first_activation = nn.ELU(mu)
            second_activation = nn.ELU(mu)
        elif hidden_activation == 'murelu':
            first_activation = NormalizedMatsushita(mu)
            second_activation = NormalizedMatsushita(mu)
        elif hidden_activation == 'ls':
            first_activation = LeastSquare()
            second_activation = LeastSquare()
        elif hidden_activation == 'sp':
            first_activation = NSoftPlus()
            second_activation = NSoftPlus()
        else:
            first_activation = nn.ReLU(False)
            second_activation = nn.ReLU(False)

        main = nn.Sequential(
            # Z goes into a linear of size: ngf
            nn.Linear(nz, ngf),
            nn.BatchNorm1d(ngf),
            first_activation,
            nn.Linear(ngf, ngf),
            nn.BatchNorm1d(ngf),
            second_activation,
            nn.Linear(ngf, nc * isize * isize)
        )
        if last_layer == 'sigmoid':
            main.add_module('top_sigmoid', torch.nn.Sigmoid())
        elif last_layer == 'tanh':
            main.add_module('top_tanh',torch.nn.Tanh())

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


class GAN_D(nn.Module):
    def __init__(self, isize, nz, nc, ndf, ngpu,hidden_activation = 'relu', last_layer='', alpha=1.0):
        super(GAN_D, self).__init__()
        self.ngpu = ngpu

        if hidden_activation == 'elu':
            first_activation = nn.ELU(alpha=alpha)
            second_activation = nn.ELU(alpha=alpha)
        else:
            first_activation = nn.ReLU(False)
            second_activation = nn.ReLU(False)

        main = nn.Sequential(
            # Z goes into a linear of size: ndf
            nn.Linear(nc * isize * isize, ndf),
            first_activation,
            nn.Linear(ndf, ndf),
            second_activation,
            nn.Linear(ndf, 1)
        )
        if last_layer == 'sigmoid':
            main.add_module('top_sigmoid', torch.nn.Sigmoid())
        elif last_layer == 'tanh':
            main.add_module('top_tanh',torch.nn.Tanh())
        elif last_layer == 'matsu':
            main.add_module('final.{0}.Matsushita'.format(nc),
                            MatsushitaLinkFunc())

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
        return output.view(input.size(0))