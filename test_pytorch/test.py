#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
===============================================================================
author: 赵明星
desc:   测试torch中各种工具的基本功能。
===============================================================================
"""

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import torch
import torch.nn.functional as a_function
import matplotlib.pyplot as plt
from torch.autograd import Variable

def foo():

    n_data = torch.ones(10, 2)
    print n_data.size()
    pass

    x0 = torch.normal(2 * n_data, 1)
    print x0



if __name__ == '__main__':
    foo()

