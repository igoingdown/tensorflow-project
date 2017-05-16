#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
===============================================================================
author: 赵明星
desc:   基于torch实现的regression。
===============================================================================
"""

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import torch
import torch.nn.functional as a_function
import matplotlib.pyplot as plt
from torch.autograd import Variable


plt.ion()
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
    plt.pause(3)

    net = Net_1(1, 10, 1)
    print net

    optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
    loss_func = torch.nn.MSELoss()
    # 回归问题的loss function用mean square error loss function。

    for t in range(100):
        prediction = net(x)
        loss = loss_func(prediction, y)

        # TODO: 先将网络中上一步用于更新参数的梯度全部置零，
        #       进行本步的反向传播，计算本步的梯度，更新参数。
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if t % 5 == 0:
            plt.cla()
            plt.scatter(x.data.numpy(), y.data.numpy())
            plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
            plt.text(0.5, 0, "Loss=%.4f" % loss.data[0],
                     fontdict={'size': 15, 'color': 'red'})
            plt.pause(0.2)

if __name__ == "__main__":
    test_regression()
