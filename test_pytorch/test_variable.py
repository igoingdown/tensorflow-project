#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
===============================================================================
author: 赵明星
desc:   测试torch中variable的基本功能。
===============================================================================
"""

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import torch
from torch.autograd import Variable


def test_variable():
    tensor = torch.randn(2, 2)
    variable = Variable(tensor, requires_grad=True)
    print "tensor: ", tensor, "\nvariable: ", variable

    t_out = torch.mean(tensor * tensor)
    v_out = torch.mean(variable * variable)
    print "t_out: ", t_out, "\nv_out: ", v_out
    v_out.backward()
    print "grad: ",  variable.grad.data.numpy()
    print "grad size: ", variable.grad.size()
    print "data:", variable.data.numpy()


if __name__ == "__main__":
    test_variable()
