#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]



################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from time import time

#clf = DecisionTreeClassifier(min_samples_split=40)
clf = SVC(kernel="rbf", C=100.0)
#clf = GaussianNB()
#clf = KNeighborsClassifier(n_neighbors=1)
#clf = RandomForestClassifier(n_estimators=100)
#clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=200)
#clf = AdaBoostClassifier(n_estimators=500, random_state=0)

t0 = time()
clf.fit(features_train, labels_train)
print("Training time:", round(time()-t0, 3), "s")
t0 = time()
pred = clf.predict(features_test)
print("Prediction time:", round(time()-t0, 3), "s")


accuracy = accuracy_score(labels_test, pred)
print ('Accuracy = {0}'.format(clf.score(features_test, labels_test)))

print(accuracy)


try:
    prettyPicture(clf, features_test, labels_test)
    #### initial visualization
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.scatter(bumpy_fast, grade_fast, color="b", label="fast")
    plt.scatter(grade_slow, bumpy_slow, color="r", label="slow")
    plt.legend()
    plt.xlabel("bumpiness")
    plt.ylabel("grade")
    plt.show()
except NameError:
    pass

