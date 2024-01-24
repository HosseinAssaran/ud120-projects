#!/usr/bin/python

import sys
import pickle
import os

sys.path.append(os.path.abspath(("../tools/")))

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.

### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)
from pprint import pprint
print(len(data_dict))
# print(list(data_dict.keys()))
# print(list(data_dict.values()))
# pprint(data_dict["YEAGER F SCOTT"])

for name, details in data_dict.items():
    details.pop('email_address', None)
pprint(data_dict["YEAGER F SCOTT"])

import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif 

df = pd.DataFrame.from_dict(data_dict, orient='index')
# Assuming 'poi' is the target variable
# Replace 'NaN' with appropriate imputations
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(df)
print(X_imputed)

y = df['poi']  # Target
print(X_imputed, y)

# Select the top k features based on chi-squared test
k_best = 2  # You can adjust this based on your requirements
skb = SelectKBest(f_classif, k=k_best)
X_new = skb.fit_transform(X_imputed, y)
# Get the selected feature names
selected_features = df.columns[skb.get_support()]
print(selected_features)


# Get the original feature names
feature_names = df.columns

# Combine feature names with their scores
feature_scores = pd.DataFrame({'Feature': feature_names, 'Score': skb.scores_})

sorted_features = feature_scores.sort_values(by='Score', ascending=False)

selected_features = sorted_features.head(k_best)['Feature'].values
print(selected_features)
# exit()
# # Print the variances before applying the threshold
# print("Variances Before Threshold:")
# # print(X.var())
# from sklearn.feature_selection import VarianceThreshold
# # Apply VarianceThreshold
# variance_threshold = 0.8  # Set your desired threshold
# selector = VarianceThreshold(threshold=variance_threshold)
# df_high_variance = pd.DataFrame(selector.fit_transform(X), columns=X.columns[selector.get_support()])
# # Display the selected features
# print("Selected Features:")
# print(len(df_high_variance.columns))

# # df.replace('NaN', pd.NA, inplace=True)
# # Replace 'NaN' with appropriate imputations
# from sklearn.impute import SimpleImputer
# imputer = SimpleImputer(strategy='mean')
# X_imputed = imputer.fit_transform(X)
# print(X_imputed)

# # Select the top k features based on chi-squared test
# k_best = 15  # You can adjust this based on your requirements
# skb = SelectKBest(f_classif, k=k_best)
# X_new = skb.fit_transform(X_imputed, y)
# # Get the selected feature names
# selected_features = X.columns[skb.get_support()]
# print(selected_features)

# from sklearn.decomposition import PCA
# # Standardize the data (important for PCA)
# X_standardized = (X_imputed - X_imputed.mean()) / X_imputed.std()

# # Apply PCA and specify the number of components you want to retain
# n_components = 5  # Adjust this based on your requirements
# pca = PCA(n_components=n_components)
# X_pca = pca.fit_transform(X_standardized)

# # Create a DataFrame with the principal components
# pca_df = pd.DataFrame(data=X_pca, columns=[f'PC{i+1}' for i in range(n_components)])

# # Concatenate the target variable 'poi' with the principal components
# final_data = pd.concat([y, pca_df], axis=1)
# print(pca_df)

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import ExtraTreesClassifier

# # Create a Random Forest classifier
# rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# # Fit the classifier to the data
# rf_classifier.fit(X_imputed, y)

# # Get feature importances
# feature_importances = pd.Series(rf_classifier.feature_importances_, index=X.columns)

# # Sort features based on importance
# sorted_features = feature_importances.sort_values(ascending=False)

# # Select top k features (adjust k as needed)
# k_features = 15
# selected_features = sorted_features.head(k_features).index
# print(selected_features)

### Task 2: Remove outliers
x = df["bonus"]
y = df["salary"]
print(x)
print(len(y))
x = pd.to_numeric(x, errors='coerce')
y = pd.to_numeric(y, errors='coerce')
print("outlier",x[x> 80000000])


X = df.drop('TOTAL')
y = y.drop('TOTAL')

print("new outlier",x[(x > 5000000) & (y > 1000000)])
# print("Pop out: ", X)
x = X["bonus"]
import matplotlib.pyplot
matplotlib.pyplot.scatter(x, y)
matplotlib.pyplot.xlabel("bonus")
matplotlib.pyplot.ylabel("poi")
# matplotlib.pyplot.show()
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.

data_dict.pop('TOTAL', 0)
my_dataset = data_dict
features_list = selected_features
print(features_list)

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

print(labels)
print(features)
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
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

clf.fit(features, labels)
prediction = clf.predict(features)
print(prediction)
print(labels)
from sklearn.metrics import accuracy_score
print(accuracy_score(prediction,labels))

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
print(clf.score(features_test,labels_test))

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)