#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
===============================================================================
author: 赵明星
desc:   测试torch中activation function的基本功能。
===============================================================================
"""

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import torch
import torch.nn.functional as a_function
import matplotlib.pyplot as plt
from torch.autograd import Variable


def test_activation_function():
    x = torch.linspace(-5, 5, 200)
    x = Variable(x)
    x_np = x.data.numpy()

    y_1 = a_function.relu(x).data.numpy()
    y_2 = a_function.sigmoid(x).data.numpy()
    y_3 = a_function.tanh(x).data.numpy()
    y_4 = a_function.softplus(x).data.numpy()
    # y_5 = a_function.softmax(x)
    # softmax只适合做概率分类的激活函数

    plt.figure(1, figsize=(8, 6))
    plt.subplot(221)
    plt.plot(x_np, y_1, c="red", label="relu")
    plt.ylim(-1, 5)
    plt.legend(loc="best")

    plt.subplot(222)
    plt.plot(x_np, y_2, c="red", label="sigmoid")
    plt.ylim(-0.2, 1.2)
    plt.legend(loc="best")

    plt.subplot(223)
    plt.plot(x_np, y_3, c="red", label="tanh")
    plt.ylim(-1.2, 1.2)
    plt.legend(loc="best")

    plt.subplot(224)
    plt.plot(x_np, y_4, c="red", label="softplus")
    plt.ylim(-0.2, 6)
    plt.legend(loc="best")

    plt.show()


if __name__ == "__main__":
    test_activation_function()
