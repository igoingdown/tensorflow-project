#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
===============================================================================
author: 赵明星
desc:   测试numpy的基本功能。
===============================================================================
"""

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import numpy as np
import random
from time_wrapper import *

x_data = np.linspace(-1, 1, 10)[:, np.newaxis]
print x_data

y_data = np.random.rand(10, 1)
print y_data

print np.linspace(-1, 1, 10)

x_data_without_0_teacher = [1, 2, 3]
y_data_without_0 = [1, 2, 3]
z_data_without_0 = [1, 2]
a_data_without_teacher_money_dad_should = [1]

a_x, a_y, a_z = [], [], []
print "~~~~", a_x
print "~~~~", a_y
print "~~~~", a_z


@time_recorder
def foo(x, y, z, a):
    print x
    print y
    print z
    print a

if __name__ == '__main__':
    foo(x_data_without_0_teacher, y_data_without_0,
        z_data_without_0, a_data_without_teacher_money_dad_should)
    random_sample_01 = np.random.random_sample()
    print random_sample_01
    two_dim_list = [[1,2], [3, 4], [5, 6, 7]]
    one_dim_list = [1, 3, 8]
    print random.choice(two_dim_list)
    print random.choice(one_dim_list)
    print [2, 4, 8] + [9, 10]
    print np.ones(10)
    np_arr_1 = np.arange(12)
    print "np_arr_1:\n{0}".format(np_arr_1)
    np_arr_2 = np.reshape(np_arr_1, (3, 4))
    print "np_arr_2:\n{0}".format(np_arr_2)
    print np.argmax(np_arr_2)
    print "first axis max index:\n{0}".format(np.argmax(np_arr_2, axis=0))
    print "second axis max index:\n{0}".format(np.argmax(np_arr_2, axis=1))
    train_X = np.linspace(-1, 1, 101)
    train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.33 + 10

    print "train x data is:\n{0}".format(train_X)
    print "\n\n\n"
    print "train y data is:\n{0}".format(train_Y)


