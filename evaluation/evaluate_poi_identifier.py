#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""
import os
import joblib
import sys

import numpy as np
sys.path.append(os.path.abspath("../tools/"))
from feature_format import featureFormat, targetFeatureSplit

data_dict = joblib.load(open("../final_project/final_project_dataset.pkl", "rb") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list, sort_keys = '../tools/python2_lesson14_keys.pkl')
labels, features = targetFeatureSplit(data)



from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, random_state=42, test_size=0.3)
print(labels_test.count(1.0))
print(features_test)
print(len(features_train))
print(len(features_test))

from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf.fit(features_train, labels_train)
predictions = clf.predict(features_test)
from sklearn.metrics import accuracy_score

test0 = [0.0] * 29
print(accuracy_score(labels_test, predictions))
print(clf.score(features_test, labels_test))
from pprint import pprint
print(labels_test)
print(predictions.tolist())

temp3 = 0
temp2 = 0
for i,element in enumerate(labels_test):
    if element == 1.0:
        temp2 += 1
        if element == predictions[i]:
            temp3 += 1
temp4 = 0
temp5 =0
for element in predictions:
    if element == 1.0:
        temp4 += 1
for i,element in enumerate(labels_test):
    if element == 0.0 and element == predictions[i]:
        temp5 += 1

print("poi in test",  temp2)
print("poi truely predicted ", temp3)
print("poi predicted", temp4)
print("not poi truely predicted", temp5)
