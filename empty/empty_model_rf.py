# -*- coding: utf-8 -*-
# Author: wangyuan
# mail: wyxidian@gmail.com
# Created Time: 
#!/bin/bash
import numpy as np
import scipy as sp
from sklearn import tree
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
import pydotplus
import os
import sys
from sklearn.externals.six import StringIO
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier


# 输入：
#     数据格式：url    score1  score2  score3  ...    label
# 输出：决策树模型empty_model.m 决策树结构tree.pdf
# path = sys.path[0]
# os.chdir(os.path.dirname(path))
# curDir = os.getcwd()


''''' 数据读入 '''
urls = []
datas = []
labels = []
with open("train.txt") as ifile:
    for line in ifile:
        line = line.strip().split("\t")
        if (len(line) < 3):
            continue
        url = line[0]
        feature = line[1].strip().split(" ")
        if (len(feature) < 16):
            continue
        feature = [int(i) for i in feature]
        datas.append(feature)
        urls.append(url)
        labels.append(line[2])

x = np.array(datas)
labels = ["normal" if l == "0" else "empty" for l in labels]
labels = np.array(labels)
y = np.zeros(labels.shape)

''''' 标签转换为0/1 '''
y[labels == 'empty'] = 1

''''' 拆分训练数据与测试数据 '''
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

''''' 使用信息熵作为划分标准，对决策树进行训练 '''

# clf = tree.DecisionTreeClassifier(criterion='entropy',max_depth=6)
clf = RandomForestClassifier(n_jobs=2)
print(clf)
clf.fit(x_train, y_train)

''''' 把决策树结构写入文件 '''
#写入pdf
# dot_data = StringIO()
# tree.export_graphviz(clf, out_file=dot_data)
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# graph.write_pdf("tree.pdf")
# 写入dot文件
# with open("tree.dot", 'w') as f:
#     f = tree.export_graphviz(clf, out_file=f)
joblib.dump(clf, "empty.model")
# os.unlink('tree.dot')
# dot_data = StringIO()
# tree.export_graphviz(clf, out_file=dot_data)
# graph = pydot.graph_from_dot_data(dot_data)
# graph.write_pdf("tree.pdf")

''''' 系数反映每个特征的影响力。越大表示该特征在分类中起到的作用越大 '''
print(clf.feature_importances_)
print clf.get_params(1)

'''''测试结果的打印'''
answer = clf.predict(x_train)
# print(x_train)
# print(answer)
# print(y_train)
print(np.mean(answer == y_train))

'''''准确率与召回率'''
precision, recall, thresholds = precision_recall_curve(y_train, clf.predict(x_train))
print precision
print recall
print thresholds
answer = clf.predict_proba(x)[:, 1]
answer = [1 if i>0.5 else 0 for i in answer]
print(classification_report(y, answer, target_names=['normal', 'empty']))



# answer = clf.predict_proba(x)[:, 1]
# answer = [1 if i>0.5 else 0 for i in answer]
# print(classification_report(y, answer, target_names=['notmatching', 'matching']))








# clf = RandomForestClassifier(n_jobs=2)
# clf.fit(x_train, y_train)
# answer = clf.predict_proba(x)[:, 1]
# answer = [1 if i>0.5 else 0 for i in answer]
# print(classification_report(y, answer, target_names=['normal', 'empty']))
#
# answer = clf.predict_proba(x_test)[:, 1]
# answer = [1 if i>0.5 else 0 for i in answer]
# print(classification_report(y_test, answer, target_names=['normal', 'empty']))
#
# joblib.dump(clf, 'rf.model')
# print '所有的树:%s' % clf.estimators_

# print clf.classes_
# print clf.n_classes_
print '各feature的重要性：%s' % clf.feature_importances_
print clf.n_outputs_
#