#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
===============================================================================
author: 赵明星
desc:   测试torch和numpy的基本功能。
===============================================================================
"""

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import torch
import numpy as np
import torch.nn.functional as a_function
import matplotlib.pyplot as plt
from torch.autograd import Variable


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


def test_variable():
    tensor = torch.FloatTensor([[1, 2], [3, 4]])
    variable = torch.autograd.Variable(tensor, requires_grad=True)
    print "tensor: ", tensor, "\nvariable: ", variable

    t_out = torch.mean(tensor * tensor)
    v_out = torch.mean(variable * variable)
    print "t_out: ", t_out, "\nv_out: ", v_out
    v_out.backward()
    print variable.grad
    print variable.data


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



class Net_1(torch.nn.Module):
    def __init__(self, n_features, n_hidden, n_output):
        pass
        super(Net_1, self).__init__()
        self.hidden = torch.nn.Linear(n_features, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, 1)

    def forward(self, x):
        x = a_function.relu(self.hidden(x))
        x = self.predict(x)
        return x


def test_regression():
    x = torch.unsqueeze(torch.linspace(-1, 1), dim=1)
    y = x.pow(2) + 0.2 * torch.rand(x.size())
    x = Variable(x)
    y = Variable(y)

    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.show()

    net = Net_1(1, 10, 1)
    print net

    optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
    loss_func = torch.nn.MSELoss()

    for t in range(100):
        prediction = net(x)
        loss = loss_func(prediction, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    test_variable()
    test()
    test_activation_function()
    test_regression()