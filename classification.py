#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2018/2/11 23:07
# @Author  : xingyuezhiji
# @Email   : zhong180@126.com
# @File    : classification.py
# @Software: PyCharm Community Edition
#coding=utf-8
import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, \
    GradientBoostingClassifier,ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from mlxtend.classifier import StackingClassifier,EnsembleVoteClassifier
from sklearn import cross_validation
from sklearn.cross_validation import KFold
clf1 = KNeighborsClassifier(4)
clf2 = DecisionTreeClassifier(criterion="gini")
clf3 = LogisticRegression()
lr = LogisticRegression()
gb =GradientBoostingClassifier()
classifiers = [
    # StackingClassifier(classifiers=[clf1, clf2, clf3],
    #                       meta_classifier=lr),
    # EnsembleVoteClassifier(clfs=[clf1, clf2, clf3],voting='soft', verbose=0),
    # SVC(kernel="linear", C=0.025),
    ExtraTreesClassifier(n_estimators=100,criterion="entropy",max_depth=None,
                 min_samples_split=2,min_samples_leaf=1,min_weight_fraction_leaf=0.,
                 max_features="auto",max_leaf_nodes=None,min_impurity_split=1e-7,
                 bootstrap=False,oob_score=False,n_jobs=1,random_state=410,
                 verbose=0,warm_start=False,class_weight=None),
    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                           max_depth=None, max_features='auto', max_leaf_nodes=None,
                           min_impurity_split=1e-07, min_samples_leaf=1,
                           min_samples_split=2, min_weight_fraction_leaf=0.0,
                           n_estimators=60, n_jobs=1, oob_score=True, random_state=410,
                           verbose=0, warm_start=False),
    AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion="entropy",
                 splitter="best",max_depth=1,min_samples_split=2,min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,max_features=None,random_state=None,
                 max_leaf_nodes=None,min_impurity_split=1e-7,class_weight=None,presort=False),
                 n_estimators=60,learning_rate=0.1,
                 algorithm='SAMME.R',random_state=410),
    GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=70,
                               subsample=1.0, criterion='friedman_mse', min_samples_split=2,
                               min_samples_leaf=1, min_weight_fraction_leaf=0.,
                               max_depth=3, min_impurity_split=1e-7, init=None,
                               random_state=410, max_features=None, verbose=0,
                               max_leaf_nodes=None, warm_start=False,
                               presort='auto'),
    LinearDiscriminantAnalysis(solver='lsqr', shrinkage=0.001, priors=None,
                 n_components=110, store_covariance=False, tol=1e-4)]

# Logging for Visual Comparison
log_cols = ["Classifier", "Accuracy", "Log Loss"]
log = pd.DataFrame(columns=log_cols)
train = pd.read_csv('train_change.csv', encoding='gb2312')
test = pd.read_csv('test_change.csv', encoding='gb2312')
train.fillna(train.mean(axis=0), inplace=True)
test.fillna(test.mean(axis=0), inplace=True)
predictors1 = []
# for i in range(40,55):
#     # if  i == 37 or i == 52 or i == 53 or i == 54 or i == 43:
#     #     continue
#     predictors1.append('SNP'+str(i))
# for i in range(10,20):
#     # if  i == 37 or i == 52 or i == 53 or i == 54 or i == 43:
#     #     continue
#     predictors1.append('SNP'+str(i))
train.drop(predictors1, axis=1, inplace=True)
test.drop(predictors1, axis=1, inplace=True)
train.drop(['id'], axis=1, inplace=True)
test.drop(['id'], axis=1, inplace=True)
predictors = [f for f in train.columns if f not in ['label']]
x = train[predictors]
y = train['label']
kf = KFold(train.shape[0],n_folds=5,random_state=1)
#划分训练集和测试集
X_train,X_test,y_train,y_test = cross_validation.train_test_split(train[predictors],
                                 train['label'],test_size=0.20,random_state=200)
ensemble = []
test_ensemble = []
for clf in classifiers:
    clf.fit(X_train, y_train)
    # for tr, te in kf:
    #     train_predictors = train[predictors].iloc[tr] # 将predictors作为测试特征
    #     train_target = train['label'].iloc[tr]
    #     clf.fit(train_predictors, train_target)
    #     test_prediction = clf.predict(train[predictors].iloc[te])
    name = clf.__class__.__name__

    print("=" * 30)
    print(name)

    print('****Results****')

    train_predictions = clf.predict(X_test)

    ensemble.append(train_predictions)

    acc = accuracy_score(y_test, train_predictions)
    # acc = accuracy_score(train['label'].iloc[tr], test_prediction)
    print("Accuracy: {:.4%}".format(acc))


for clf in classifiers:
    clf.fit(x,y)
    test_predictions = clf.predict(test[predictors])
    test_ensemble.append(test_predictions)


ensemble = (np.sum(ensemble,axis=0))//3
print(ensemble)
print(accuracy_score(y_test, ensemble))
test_ensemble = (np.sum(test_ensemble,axis=0))//3
submission = pd.DataFrame({'sub': test_ensemble})
# if accuracy_score(y_test, ensemble) >= 0.79:
#     submission.to_csv(r'sub{}_{}.csv'.format(round(accuracy_score(y_test,ensemble),3),
#             datetime.datetime.now().strftime('%Y%m%d_%H%M%S')), header=None,index=False, float_format='%.4f')
