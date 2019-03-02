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

    # x = Variable(torch.randn(2, 3))
    # m = torch.nn.Softmax()
    # print x
    # print m(x)
    # print torch.max(m(x), 1)
    # print x.data.contiguous()

    # y = torch.randn(3, 2)
    # print y
    # print(y.stride(), y.is_contiguous())
    #
    # print "-" * 100
    # try:
    #     print y.t().is_contiguous()
    # except Exception :
    #     print "aa"
    #
    # y.view(2, 3)  # ok

    # x = torch.randn(2, 3)
    # print x
    # for t in torch.chunk(x, 2):
    #     print t
    # print torch.chunk(x, 3, 1)
    # print '-' * 100
    # print torch.split(x, 4, 1)
    # pass

    # x = torch.FloatTensor([1, 2 , 3])
    # print x
    # print torch.unsqueeze(x, 0)
    # print torch.get_num_threads()

    x = torch.range(1, 6).view(2, 3)
    print x
    print torch.topk(x, 2, 1)
    print torch.diag(x, -1)
    print x.index(1)
    print x.stride()
    print torch.randn(4, 5, 6).stride()
    print x.storage()
    print dir(x.storage())
    print x.storage().is_pinned()
    print x.storage().data_ptr()
if __name__ == '__main__':
    foo()

