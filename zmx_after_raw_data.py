#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
===============================================================================
author: 赵明星
desc:   设计这个脚本仅仅为了让模型在本地run起来。
===============================================================================
"""

import sys
reload(sys)
sys.setdefaultencoding('utf-8')


class Textloader(object):
    def __init__(self):
        self.nl_word_oov_begin = 10
        self.nl_character_oov_begin = 10
        self.nl_phrase_oov_begin = 10
        self.kb_phrase_oov_begin = 10
        self.kb_character_oov_begin = 10
        self.kb_word_oov_begin = 10
