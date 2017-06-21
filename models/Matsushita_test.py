
# coding: utf-8

# In[ ]:
import os,sys
par_dir = os.path.join(os.pardir, os.pardir)
curr_path = os.path.abspath(__file__)
root_path = os.path.abspath(os.path.join(curr_path, par_dir))

sys.path.append(str(root_path))

import unittest
import torch
from torch import nn
from unittest import TestCase
from torch.autograd import Variable

from models.Matsushita import MatsushitaTransform,NormalizedMatsushita
from torch.autograd import gradcheck

from models.NormalizedSoftplus import NSoftPlus
from models.LeastSquare import LeastSquare

class MatsushitaModuleTest(TestCase):

    def test_forward_prop(self):
        m = MatsushitaTransform()
        v = Variable(torch.FloatTensor([1 , 2]))
        o = m.forward(v)
        ground_truth = torch.FloatTensor([1.2071067811865475 , 2.118033988749895])
        self.assertTrue(torch.equal(ground_truth,o.data))

    def test_forward_prop_least_square(self):
        m = LeastSquare()
        v = Variable(torch.FloatTensor([-2 , 1, 0.5, -1, 2]))
        o = m.forward(v)
        ground_truth = torch.FloatTensor([-1, 3, 1.25, -1, 7])
        self.assertTrue(torch.equal(ground_truth,o.data))

    def test_backward_prop_least_square(self):
        m = LeastSquare()
        v = Variable(torch.FloatTensor([-2 , 1, 0.5, -1, 2]), requires_grad=True)
        o = m.forward(v)
        l = o.sum()
        l.backward()
        ground_truth = torch.FloatTensor([0, 4, 3, 0, 4])
        self.assertTrue(torch.equal(ground_truth, v.grad.data))

    def test_scala_backward(self):
        m = MatsushitaTransform()
        criterion = torch.nn.MSELoss()
        y = Variable(torch.FloatTensor([1.118033988749895]), requires_grad=False)
        v = Variable(torch.FloatTensor([2]), requires_grad=True)
        o = m.forward(v)
        loss = criterion(o, y)
        loss.backward()
        self.assertEqual(v.grad[0], 1.8944)
        # 1.8944

    def test_backward_prop(self):
        input = (Variable(torch.randn(20, 10).double(), requires_grad=True),)
        test = gradcheck(MatsushitaTransform(), input, eps=1e-6, atol=1e-4)
        self.assertTrue(test)

    def test_backward_prop_normalized(self):
        input = (Variable(torch.randn(20, 10).double(), requires_grad=True),)
        test = gradcheck(NormalizedMatsushita(), input, eps=1e-6, atol=1e-4)
        self.assertTrue(test)

    def test_leastSquare(self):
        input = (Variable(torch.randn(20, 10).double(), requires_grad=True),)
        test = gradcheck(LeastSquare(), input, eps=1e-6, atol=1e-4)
        self.assertTrue(test, 'least square failed')

    def test_NSoftPlus(self):
        input = (Variable(torch.randn(20, 10).double(), requires_grad=True),)
        test = gradcheck(NSoftPlus(), input, eps=1e-6, atol=1e-4)
        self.assertTrue(test)

if __name__ == '__main__':
    unittest.main()