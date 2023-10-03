#!/usr/bin/python3

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import joblib
import re

enron_data = joblib.load(open("../final_project/final_project_dataset.pkl", "rb"))

print(enron_data)
print(len(enron_data))
print(type(enron_data))
print(enron_data.keys())
for i in sorted(enron_data.keys()):
    print(i, end="-- ")
print(len(enron_data['METTS MARK']))
print(enron_data['METTS MARK'])

print("In the E+F dataset,There is total poi of", sum(1 for key in enron_data.keys() if enron_data[key]["poi"] == 1))
filename = "../final_project/poi_names.txt"
pattern = r"\([yn]\) \w+, \w+"  # Pattern to match "(y) Lastname, Firstname"
count = 0
with open(filename, "r") as file:
    for line in file:
        if re.match(pattern, line):
            count += 1

print("POI Count:", count)

print(enron_data['PRENTICE JAMES']['total_stock_value'])
print(enron_data['COLWELL WESLEY']['from_this_person_to_poi'])
print(enron_data['SKILLING JEFFREY K']['exercised_stock_options'])

print(enron_data['LAY KENNETH L']['total_payments'])
print(enron_data['FASTOW ANDREW S']['total_payments'])
print(enron_data['SKILLING JEFFREY K']['total_payments'])
print(sum(1 for key in enron_data.keys() if enron_data[key]["salary"] != 'NaN'))
print(sum(1 for key in enron_data.keys() if enron_data[key]["email_address"] != 'NaN'))
tp_nan = sum(1 for key in enron_data.keys() if enron_data[key]["total_payments"] == 'NaN')
print("the number of total payment is NaN = ", tp_nan)
print("the percentage of total payment is NaN = ", tp_nan/len(enron_data)*100)
tp_nan_poi = sum(1 for key in enron_data.keys() if (enron_data[key]["total_payments"] == 'NaN') & (enron_data[key]["poi"] == 1))
print("the number of poi which total payment is NaN = ", tp_nan_poi)
