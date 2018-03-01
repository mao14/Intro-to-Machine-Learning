#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier
from sklearn.pipeline import Pipeline

from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.cross_validation import train_test_split

from sklearn.grid_search import GridSearchCV
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, make_scorer

#Functions for the Project

def remove_outliers(data_dict):
    outliers = ['TOTAL','THE TRAVEL AGENCY IN THE PARK']
    for outlier in outliers:
        data_dict.pop(outlier, 0)
    return data_dict

def add_features(features_list):
    for person in data_dict:
        try:
            poi_messages = data_dict[person]['from_this_person_to_poi'] + data_dict[person]['from_poi_to_this_person']+ data_dict[person]['shared_receipt_with_poi']
            total_messages = data_dict[person]['to_messages'] + data_dict[person]['from_messages']
            poi_ratio = float(poi_messages) / float(total_messages)
            data_dict[person]['poi_ratio'] = poi_ratio
        except:
            data_dict[person]['poi_ratio'] = 'NaN'

        features_list = features_list + ['poi_ratio']

    return features_list

def generate_classifiers():

    #Here we process different classifiers at the same time
    clf_classifiers = []

    clf_naive_bayes = GaussianNB()
    parameters_naive = {}
    clf_classifiers.append((clf_naive_bayes,parameters_naive))

    clf_svm = svm.SVC()
    parameters_svm = {
        'C': [1e3, 5e3, 1e4, 5e4, 1e5],
        'kernel' : ['linear','rbf'],
        'gamma' : [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
    }
    clf_classifiers.append((clf_svm,parameters_svm))

    clf_tree = tree.DecisionTreeClassifier()
    parameters_tree = {
        'min_samples_split' : [5,10,20,30,40],
    }
    clf_classifiers.append((clf_tree,parameters_tree))

    clf_adaboost = AdaBoostClassifier()
    parameters_adaboost = {
        'n_estimators' : [40,80,84,85,90],
        'learning_rate' : [0.2,0.4,0.7,0.9]
    }
    clf_classifiers.append((clf_adaboost,parameters_adaboost))

    clf_forest = RandomForestClassifier()
    parameters_forest = {
        'n_estimators' : [5,10,20,30,40],
        'min_samples_split' : [5,10,20,30,40],
        'max_leaf_nodes' : [10,12]
    }
    clf_classifiers.append((clf_forest,parameters_forest))

    return clf_classifiers

def optimize_classifiers(clf_classifiers, features_train, labels_train):

    #This def returns the classifiers optimized with its parameters
    best_classifier = []

    for clf, param in clf_classifiers:
        scorer = make_scorer(f1_score)
        clf = GridSearchCV(clf, param, scoring=scorer)
        clf = clf.fit(features_train,labels_train)
        clf = clf.best_estimator_

        best_classifier.append(clf)

    return best_classifier

def train_classifiers(feature_train, labels_train, pca_supervised=False,unsupervised=True,pca_unsupervised=False):

    """
    Trains the Model with different classification and clustering algorithms .
    It returns a list of estimators with the best parameters for a particular algorithm.
    """

    clf_supervised = generate_classifiers()

    if pca_supervised:
        clf_supervised = pca_transform(clf_supervised)

    clf_supervised = optimize_classifiers(clf_supervised, feature_train, labels_train)
    clf_unsupervised = []

    if unsupervised:
        clf_kmeans = KMeans(n_clusters = 3, tol = 0.01)

        if pca_unsupervised:
            pca = PCA(n_components= 26, withen=False)
            clf_kmeans = Pipeline([('pca',pca),('kmeans',kmeans)])
        clf_kmeans.fit(features_train)

    return clf_supervised #+ [clf_kmeans]

"""
def evaluate_classifiers(clf_classifiers, features_test, labels_test):

    clf_scores = []

    for classifier in clf_classifiers:
        pred = classifier.predict(features_test)
        accuracy = accuracy_score(pred, labels_test)

        precision = precision_score(pred, labels_test)
        recall = recall_score(pred, labels_test)
        f1 = f1_score(pred, labels_test)

        clf_scores.append(classifier, accuracy, precision, recall, f1)

    return clf_scores
"""
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary','exercised_stock_options','deferral_payments',
                'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income',
                 'expenses', 'long_term_incentive', 'restricted_stock', 'director_fees',
                 'to_messages', 'from_poi_to_this_person',
                 'from_messages', 'from_this_person_to_poi',
                  'shared_receipt_with_poi']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict = remove_outliers(data_dict)

### Task 3: Create new feature(s)
features_list = add_features(features_list)

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)

# targetFeatureSplit separates the data into a label and a list of features
# the first item always has to be 'poi'
labels, features = targetFeatureSplit(data)

#Scaling of features
scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)

#Using feature_selection find the most discriminative features to uses
    #features = SelectKBest(f_classif, k = 15).fit_transform(features, labels)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

#t = time.time()
model = train_classifiers(features_train, labels_train)

for i in range(len(model)):
    test_classifier(model[i],my_dataset,features_list)



### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

#dump_classifier_and_data(clf, my_dataset, features_list)
