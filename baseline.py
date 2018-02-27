#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2018/2/10 10:22
# @Author  : xingyuezhiji
# @Email   : zhong180@126.com
# @File    : baseline.py
# @Software: PyCharm Community Edition
#coding=utf-8
import time
import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb
from dateutil.parser import parse
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation, metrics
from sklearn.model_selection import GridSearchCV
from sklearn import cross_validation
from sklearn.ensemble import GradientBoostingClassifier

train = pd.read_csv('f_train_20180204.csv', encoding='gb2312')
test = pd.read_csv('f_test_a_20180204.csv', encoding='gb2312')
train_all = pd.read_csv('train_change.csv', encoding='gb2312')
test_real = pd.read_csv('test_change.csv', encoding='gb2312')

train.fillna(train.mean(axis=0), inplace=True)
test.fillna(test.mean(axis=0), inplace=True)
train_all.fillna(train_all.mean(axis=0), inplace=True)
test_real.fillna(test_real.mean(axis=0), inplace=True)
predictors1 = []
for i in range(40,55):
    # if  i == 37 or i == 52 or i == 53 or i == 54 or i == 43:
    #     continue
    predictors1.append('SNP'+str(i))

train.drop(predictors1, axis=1, inplace=True)
train_all.drop(predictors1, axis=1, inplace=True)
train.drop(['id'], axis=1, inplace=True)
train_all.drop(['id'], axis=1, inplace=True)

# train.drop(['wbc','身高','SNP2','SNP49','hsCRP','ApoB','SNP25','SNP26','SNP52','SNP45','SNP54',
#             'SNP16','SNP22','产次','SNP36','SNP38','舒张压','SNP40','SNP41','SNP7','SNP8','SNP13'], axis=1, inplace=True)
predictors = [f for f in train_all.columns if f not in ['label']]
# predictors1 = ['SNP6','SNP7','SNP12','RBP4','ACEID','SNP47','SNP37','SNP42','Lpa',
#                'SNP3','SNP4','SNP11','SNP18','SNP19','孕前体重','DM家族史']

# x = train[predictors]
# y = train['label']
#划分训练集和测试集
X_train,X_test,y_train,y_test = cross_validation.train_test_split(train_all[predictors],
                                 train_all['label'],test_size=0.2,random_state=0)
# print(x)
clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            n_estimators=70, n_jobs=1, oob_score=False, random_state=410,
            verbose=0, warm_start=False)
# clf.fit(x,y)
# y_pred = clf.predict(test[predictors])
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
sub = clf.predict(test_real[predictors])

gbc = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=70,
                 subsample=1.0, criterion='friedman_mse', min_samples_split=2,
                 min_samples_leaf=1, min_weight_fraction_leaf=0.,
                 max_depth=3, min_impurity_split=1e-7, init=None,
                 random_state=410, max_features=None, verbose=0,
                 max_leaf_nodes=None, warm_start=False,
                 presort='auto')
gbc.fit(X_train,y_train)
y_pred1 = gbc.predict(X_test)
# param_test1 = {'min_samples_split':range(2,20,2), 'min_samples_leaf':range(10,60,10)}
# gsearch1 = GridSearchCV(cv=5, error_score='raise',
#        estimator=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
#             max_depth=None, max_features='auto', max_leaf_nodes=None,
#             min_impurity_split=1e-07, min_samples_leaf=1,
#             min_samples_split=10, min_weight_fraction_leaf=0.0,
#             n_estimators=100, n_jobs=1, oob_score=False, random_state=410,
#             verbose=0, warm_start=False),
#        fit_params={}, iid=True, n_jobs=1,
#        param_grid={'n_estimators': range(10,101,10),
#                      'min_samples_split': [2, 5, 10],'max_depth': [10, 50, 100]},
#        pre_dispatch='2*n_jobs', refit=True, return_train_score=True,
#        scoring='roc_auc', verbose=0)
# gsearch1.fit(x,y)
# print(gsearch1.best_params_)
# gsearch1.fit(x,y)
# y_pred = gsearch1.predict(test[predictors])



# submission = pd.DataFrame({'sub': sub})
# submission.to_csv('pred.csv',header=None,index=False)
# print(metrics.roc_auc_score(test['label'],y_pred))
# print(metrics.accuracy_score(test['label'],y_pred))
# print(clf.oob_score)
print(metrics.roc_auc_score(y_test,y_pred))
print(metrics.accuracy_score(y_test,y_pred))
print(metrics.roc_auc_score(y_test,y_pred1))
print(metrics.accuracy_score(y_test,y_pred1))
# submission.to_csv(r'sub{}_{}.csv'.format(round(metrics.accuracy_score(y_test,y_pred),3),
#             datetime.datetime.now().strftime('%Y%m%d_%H%M%S')), header=None,index=False, float_format='%.4f')


