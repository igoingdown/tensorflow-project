#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
===============================================================================
author: 赵明星
desc:   测试torch和numpy的基本功能及区别。
===============================================================================
"""

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import torch
import numpy as np


def test():
    np_data = np.arange(6).reshape((2, 3))
    torch_data = torch.from_numpy(np_data)
    tensor2array = torch_data.numpy()
    print np_data, "\n", torch_data, "\n", tensor2array

    data = [[-1, -2], [-4, -5]]
    np_array = np.array(data)
    tensor = torch.FloatTensor(data)
    variable = torch.autograd.Variable(tensor)
    print torch.mm(tensor, tensor), np.matmul(data,data)
    print tensor.dot(tensor), "\n", np_array.dot(np_array)
    tensor = variable.data
    np_array = tensor.numpy()
    data = np_array.tolist()
    print data

if __name__ == "__main__":
    test()
