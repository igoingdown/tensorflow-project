#!/usr/bin/python
# -*- coding:utf-8 -*-

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import numpy as np
from time_wrapper import *

x_data = np.linspace(-1, 1, 10)[:, np.newaxis]
print x_data

y_data = np.random.rand(10, 1)
print y_data

print np.linspace(-1, 1, 10)

x_data_without_0_teacher_money_dad_should_buy_something_anything_you_go_home_where = [1, 2, 3]
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
    foo(x_data_without_0_teacher_money_dad_should_buy_something_anything_you_go_home_where, y_data_without_0,
        z_data_without_0, a_data_without_teacher_money_dad_should)
