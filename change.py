#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2018/2/11 18:40
# @Author  : xingyuezhiji
# @Email   : zhong180@126.com
# @File    : change.py
# @Software: PyCharm Community Edition
#coding=utf-8

import pandas as pd
import numpy as np
# from fancyimpute import KNN
from dateutil.parser import parse

def change(name):
    df = pd.read_csv(name,encoding='gb2312')
    index = ['孕前BMI','VAR00007','SNP34','ALT','SNP43','SNP19',]
    index1 = ['SNP34', 'SNP37', 'SNP55', 'SNP54']
    for i in index:
        # index.remove(i)
        for j in index:
            if i == j:
                continue
            else:
                df['{}/{}'.format(i,j)] = df[i]/df[j]
    # for i in index:
    #     index.remove(i)
    #     for j in index:
    #         if i == j:
    #             continue
    #         else:
    #             df['{}*{}'.format(i,j)] = df[i]*df[j]
    return df

df1 = change('f_train_20180204.csv')
print(df1.shape)
df1.to_csv('train_change.csv', index=False)
df2 = change('f_test_a_20180204.csv')
df2.to_csv('test_change.csv', index=False)
# df3 = change('d_test_B_20180128.csv')
# df3.to_csv('test_change_B.csv', index=False)
# df4 = change('d_train_20180102_add.csv')
# print(df4.shape)
# df4.to_csv('train_change_add.csv', index=False)