#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2018/2/11 18:10
# @Author  : xingyuezhiji
# @Email   : zhong180@126.com
# @File    : analyze.py
# @Software: PyCharm Community Edition
#coding=utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')

import warnings
def ignore_warn(*args ,**kwargs):
    pass
warnings.warn = ignore_warn

from scipy import stats
from scipy.stats import norm, skew

train = pd.read_csv('f_train_20180204.csv',encoding='gbk')
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))
corrmat = train.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat,vmax=0.9,square=True)
plt.show()