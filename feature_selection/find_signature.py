#!/usr/bin/python

import pickle
import numpy
numpy.random.seed(42)


### The words (features) and authors (labels), already largely processed.
### These files should have been created from the previous (Lesson 10)
### mini-project.
words_file = "../text_learning/your_word_data.pkl"
authors_file = "../text_learning/your_email_authors.pkl"
word_data = pickle.load( open(words_file, "r"))
authors = pickle.load( open(authors_file, "r") )



### test_size is the percentage of events assigned to the test set (the
### remainder go into training)
### feature matrices changed to dense representations for compatibility with
### classifier functions in versions 0.15.2 and earlier
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(word_data, authors, test_size=0.1, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')

features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test).toarray()

"""
Exercise from Lesson 12. 28
In order to figure out what words are causing the problem, you need to go back to the TfIdf
and use the feature numbers that you obtained in the previous part of the mini-project to get
the associated words. You can return a list of all the words in the TfIdf by calling
get_feature_names() on it; pull out the word that.s causing most of the discrimination of
the decision tree. What is it? Does it make sense as a word that.s uniquely tied to either
Chris Germany or Sara Shackleton, a signature of sorts?
"""
if False:
    vectorizer.fit_transform(features_train)
    #feature_word_33614 = vectorizer.get_feature_names()[33614]
    #print('feature_word: ', feature_word_33614)

    feature_word1 = vectorizer.get_feature_names()[18849]
    print('feature_word1: ', feature_word1)

    feature_word2 = vectorizer.get_feature_names()[21323]
    print('feature_word2: ', feature_word2)

### a classic way to overfit is to use a small number
### of data points and a large number of features;
### train on only 150 events to put ourselves in this regime
if True:
    features_train = features_train[:150].toarray()
    labels_train   = labels_train[:150]


### your code goes here


from sklearn import tree
from sklearn.metrics import accuracy_score

if True:
    clf = tree.DecisionTreeClassifier(min_samples_split=40)
    clf.fit(features_train,labels_train)
    pred = clf.predict(features_test)

if True:
    print(accuracy_score(pred,labels_test))

"""
Take your (overfit) decision tree and use the feature_importances_ attribute to get a
list of the relative importance of all the features being used. We suggest iterating through
this list (itss long, since this is text data) and only printing out the feature importance if
its above some threshold (say, 0.2. Remember, if all words were equally important,
each one would give an importance of far less than 0.01).
What's the importance of the most important feature? What is the number of this feature?
"""
if False:

    for number, value in enumerate(clf.feature_importances_):
        if value > 0.2:
            print(number, value)
