#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
import pprint

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score

features_train, features_test, labels_train, labels_test = train_test_split(features,labels, test_size=0.3, random_state=42)

# Finding accuracy
if True:
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(features_train,labels_train)
    pred = clf.predict(features_test)

    accuracy = accuracy_score(pred, labels_test)

# Finding how many POIs are in the test set for your POI identifier
if False:
    print(sum(labels_test))

# How many people total are in your test set
if False:
    print(len(labels_test))

# If your identifier  predicted 0 (not POI for everyone in the test set, what would your accuracy be?)
if False:
    print(float((len(labels_test)-sum(labels_test))/len(labels_test)))

#Look at the predictions of your model and compare them to the true test labels. Do you get any true positives?
if False:
    print(pred)
    print(labels_test)

#you can just guess the more common class label for every point, not a very insightful strategy
#Precision and recall can help illuminate your performance better.

if True:
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score

    print('precision score: ',precision_score(labels_test, pred ))
    print('recall score: ',recall_score(labels_test, pred ))
