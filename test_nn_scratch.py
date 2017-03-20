#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
===============================================================================
author: 赵明星
desc:   手写神经网络，不使用tensorflow类似的高级架构。
===============================================================================
"""

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import numpy as np
from sklearn import datasets, linear_model
import sklearn
import matplotlib.pyplot as plt


class Config:
    nn_input_dim = 2
    nn_output_dim = 2
    epsilon = 0.01
    reg_lambda = 0.01


def generate_data():
    np.random.seed(0)
    X, y = datasets.make_moons(200, noise=0.20)
    return X, y


def visualize(X, y, model):
    pass


if __name__ == "__main__":
    X, y = generate_data()
    print len(X)
    print X, "\n\n\n"
    print y
    plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    clf = sklearn.linear_model.LogisticRegressionCV()
    clf.fit(X, y)
    plt.plot_decision_boundary(lambda x: clf.predict(x))
    plt.title("Logistic Regression")
    plt.show()






















