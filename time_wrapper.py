#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
===============================================================================
author: 赵明星
desc:   实现通用的函数的time wrapper并测试。
===============================================================================
"""

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import time

def time_recorder(func):
    def wrapper(*args, **kwargs):
        print "function {0} begin......".format(func.__name__)
        start_time = time.clock()
        res = func(*args, **kwargs)
        end_time = time.clock()
        dur = end_time - start_time
        print "function {0} over......".format(func.__name__)
        print "total time running function {0} used: {1}s\n\n".format(func.__name__, dur)
        return res
    return wrapper
