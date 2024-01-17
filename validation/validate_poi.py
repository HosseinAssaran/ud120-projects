#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""
import json
import os
import joblib
import sys

sys.path.append(os.path.abspath("../tools/"))
from feature_format import featureFormat, targetFeatureSplit

data_dict = joblib.load(open("../final_project/final_project_dataset.pkl", "rb") )
#print(json.dumps({key: (value.get("poi"), value.get("salary")) for key, value in data_dict.items()}, indent=4))
### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
print(data)
labels, features = targetFeatureSplit(data)
print(labels)
print(features)

from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, random_state=42, test_size=0.3)
print(labels_test)
print(features_test)
print(len(features_train))
print(len(features_test))

from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf.fit(features_train, labels_train)
predictions = clf.predict(features_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(labels_test, predictions))
print(clf.score(features_test, labels_test))
### it's all yours from here forward!  


