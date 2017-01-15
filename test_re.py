#!/usr/bin/python
# -*- coding:utf-8 -*-


"""
===============================================================================
author: 赵明星
desc:   学习正则表达式(re)的教高级用法。
===============================================================================
"""

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import re
import locale

content = "ab* hilary_ 48838dw @we DDFFD3*"

print re.split("([`|~!@#\$%\^&\*\(\)\-_=\+\[\{\]\};:'\",<\.>/\?\s])",
               content)
print re.split("(\w+)|", content)

print locale.getdefaultlocale()

numbers = [x for x in range(10)]
print numbers




