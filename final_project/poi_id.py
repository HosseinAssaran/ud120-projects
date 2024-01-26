#!/usr/bin/python

import sys
import pickle
import os

sys.path.append(os.path.abspath(("../tools/")))

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from pprint import pprint
import pandas as pd

### Task 1: Select what features you'll use.

### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

print("Get data dict with len", len(data_dict))

# Convert data to pandas data frame to make it easier to manipulate
df = pd.DataFrame.from_dict(data_dict, orient='index')

print("Remove 'email address' from data frame")
df.drop('email_address', inplace=True, axis=1)

# Assuming 'poi' is the target variable
y = df['poi']  # Target
# df.drop('poi', inplace=True, axis=1)

# Replace 'NaN' with appropriate imputations
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
df_imputed = imputer.fit_transform(df)
print("Data frame imputed.")

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif 
# Select the top k features based on chi-squared test
k_best = 2  # You can adjust this based on your requirements
skb = SelectKBest(f_classif, k=k_best)
skb.fit_transform(df_imputed, y)
# Get the selected feature names
selected_features = df.columns[skb.get_support()]
# Get the original feature names
feature_names = df.columns
# Combine feature names with their scores
feature_scores = pd.DataFrame({'Feature': feature_names, 'Score': skb.scores_})
sorted_features = feature_scores.sort_values(by='Score', ascending=False)
selected_features_with_kbest = sorted_features.head(k_best)['Feature'].values
print("Selected features with kbes")
print(selected_features_with_kbest)

k_features = 2
from sklearn.ensemble import RandomForestClassifier
# Create a Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
# Fit the classifier to the data
rf_classifier.fit(df_imputed, y)
# Get feature importances
feature_importances = pd.Series(rf_classifier.feature_importances_, index=df.columns)
# Sort features based on importance
sorted_features = feature_importances.sort_values(ascending=False)
# Select top k features (adjust k as needed)
selected_features_with_random_forest = sorted_features.head(k_features).index.values
print("Selected features with random forest")
print(selected_features_with_random_forest)
print("Choose kbest features:")
selected_features = selected_features_with_kbest

### Task 2: Remove outliers
bonus = df["bonus"]
salary = df["salary"]
bonus = pd.to_numeric(bonus, errors='coerce')
salary = pd.to_numeric(salary, errors='coerce')
import matplotlib.pyplot
matplotlib.pyplot.scatter(bonus, salary)
matplotlib.pyplot.xlabel("bonus")
matplotlib.pyplot.ylabel("salary")
matplotlib.pyplot.show()
print("Outlier is ",bonus[bonus > 80000000])
df = df.drop('TOTAL')
y = y.drop('TOTAL')
print("drop 'TOTAL")
bonus = df["bonus"]
salary = df["salary"]
bonus = pd.to_numeric(bonus, errors='coerce')
salary = pd.to_numeric(salary, errors='coerce')
matplotlib.pyplot.scatter(bonus, salary)
matplotlib.pyplot.xlabel("bonus")
matplotlib.pyplot.ylabel("salary")
matplotlib.pyplot.show()
print("New outliers ignored because they inlcude valuable data:")
print(bonus[(bonus > 5000000) & (salary > 1000000)])

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.

data_dict.pop('TOTAL', 0)
for name, details in data_dict.items():
    details.pop('email_address', None)

my_dataset = data_dict
features_list = selected_features
print("feature list:")
print(features_list)
print(len(data_dict))
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

print("label length:", len(labels))
print("feaure length", len(features))

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

# from sklearn.svm import SVC
# clf = SVC()
# from sklearn.ensemble import GradientBoostingClassifier
# clf = GradientBoostingClassifier()
# from sklearn.linear_model import LogisticRegression
# clf = LogisticRegression()
# from sklearn.neighbors import KNeighborsClassifier
# clf = KNeighborsClassifier()
# from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(features, labels)
prediction = clf.predict(features)
print("prediction: ", prediction)
print("labels:     ", labels)

from sklearn.metrics import accuracy_score
print("train using all data accuracy:")
print(accuracy_score(prediction, labels))

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print("train length {}, test length {}".format(len(labels_train), len(labels_test)))
print(clf.score(features_test,labels_test))

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)