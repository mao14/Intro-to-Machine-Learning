#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
features = ["salary", "bonus"]

#To remove Total outlier:
if True:
    data_dict.pop('TOTAL', 0)

data = featureFormat(data_dict, features)

### your code below
#This code is to show the total outlier:
if False:
    bonus = []
    data1 = data.tolist()

    for i in data1:
        bonus.append(i[1])

    max_outlier = max(bonus)

    for k,v in data_dict.items():
        if v['bonus'] == max_outlier:
            print(k, ":", v['bonus'])

#This code is to show outliers with more than 1mm in salary and bonus over 5mm:

if True:
    outliers = []
    data1 = data.tolist()

    for i in data1:
        if i[0] > 1000000 and i[1] > 5000000:
            outliers.append(i[1])

    outliers.sort(reverse=True)

    max_outlier1 = outliers[0]
    max_outlier2 = outliers[1]

    for k, v in data_dict.items():
        if v['bonus'] == max_outlier1 or v['bonus'] == max_outlier2:
            print(k, ":", v['bonus'])

# Show graphic:
if False:
    for point in data:
        salary = point[0]
        bonus = point[1]
        matplotlib.pyplot.scatter( salary, bonus )

    matplotlib.pyplot.xlabel("salary")
    matplotlib.pyplot.ylabel("bonus")
    matplotlib.pyplot.show()
