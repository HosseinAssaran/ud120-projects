#!/usr/bin/python3
import os
import joblib
import sys
import matplotlib.pyplot
sys.path.append(os.path.abspath("../tools/"))
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = joblib.load( open("../final_project/final_project_dataset.pkl", "rb") )
features = ["salary", "bonus"]
print("Pop out: ", data_dict.pop('TOTAL', 0))
data = featureFormat(data_dict, features)

# print(data)
# print(type(data))
### your code below
x = data[:, 0]
y = data[:, 1]
matplotlib.pyplot.scatter(x, y)
matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()


