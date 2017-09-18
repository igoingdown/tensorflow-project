#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
===============================================================================
author: 赵明星
desc:   基于torch实现的classification。
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
        self.predict = torch.nn.Linear(n_hidden, 2)

    def forward(self, x):
        x = a_function.relu(self.hidden(x))
        x = self.predict(x)
        return x


def test_classification():
    n_data = torch.ones(100, 2)
    print n_data.size()
    x_0 = torch.normal(2 * n_data, 1)
    y_0 = torch.zeros(100)
    x_1 = torch.normal(-2 * n_data, 1)
    y_1 = torch.ones(100)
    x = torch.cat((x_0, x_1), 0).type(torch.FloatTensor)
    y = torch.cat((y_0, y_1), 0).type(torch.LongTensor)
    # print "x.size: ", x.size()
    # print "y.size: ", y.size()
    # print y
    x, y = Variable(x), Variable(y)

    plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1],
                c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
    plt.pause(0.01)
    # net = Net_1(2, 10, 2)
    net = torch.nn.Sequential(torch.nn.Linear(2, 10), torch.nn.ReLU(), torch.nn.Linear(10, 2))
    print net

    optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
    loss_func = torch.nn.CrossEntropyLoss()
    # 分类问题的loss function使用cross entropy loss function。

    for t in range(30):
        out = net(x)
        loss = loss_func(out, y)

        # TODO: 先将网络中上一步用于更新参数的梯度全部置零，
        #       进行本步的反向传播，计算本步的梯度，更新参数。
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if t % 2 == 0:
            plt.cla()
            print len(torch.max(a_function.softmax(out), 1))
            # torch.max(input, dim)返回值为一个含有两个Variable的元组，
            #     存放的分别是dim维度的最大值和该最大值在dim维度的index，
            #     这对于分类问题非常好用！

            tmp = a_function.softmax(out)
            prediction = torch.max(a_function.softmax(out), 1)[1]
            prediction_y = prediction.data.numpy().squeeze()
            target_y = y.data.numpy()
            plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1],
                        c= prediction_y, s=100, lw=0)
            accuracy = sum(prediction_y == target_y) / 200.0
            print accuracy
            plt.text(0.5, 0, "accuracy=%.4f" % accuracy,
                     fontdict={'size': 15, 'color': 'red'})
            plt.pause(0.01)


if __name__ == "__main__":
    test_classification()
