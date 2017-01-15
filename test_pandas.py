#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
===============================================================================
author: 赵明星
desc:   学习pandas基本用法。
===============================================================================
"""

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

with open("test_file.txt", "w") as f:
    for i in range(10):
        for j in range(10):
            f.write(str(i))
            if j != 9:
                f.write("\t")
        f.write("\n")


series_1 = pd.Series(np.random.randn(3))
series_2 = pd.Series([1, 2, np.nan, 3])
print series_1, "\n\n", series_2
dates = pd.date_range('20161001', periods=10)
df = pd.DataFrame(np.random.randn(10, 4), index=dates, columns=list('ABCD'))
print df, '\n\n', df.dtypes, '\n\n', df.tail(2), '\n\n\n', df.index
print '\n\n', df.describe()
print '\n\nfd.T is:\n{0}'.format(df.T)
print '\n\ndf[\'A\'] is {0}\n'.format(df['A'])
print '\n\ndf.loc[\'20161001\', \'A\'] is {0}'.format(df.loc['20161001', 'A'])
print '\n\ndf.iloc[3] is:\n{0}\n\n'.format(df.iloc[3])
print '\n\ndf.cumsum is\n{0}\n\n'.format(df.cumsum())
df = df.cumsum()
plt.figure()
df.plot()
plt.legend(loc="best")
df.to_csv("test.csv")

df_2 = pd.DataFrame({'A': 1, 'B': series_1})
print df_2, '\n\n', df_2.dtypes, '\n\n', df_2.head(2), '\n\n', df_2.index