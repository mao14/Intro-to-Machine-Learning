#!/usr/bin/python

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

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

# How many data points are in the enron dataset?
if False:
    count = 0
    for i in enron_data.keys():
        count += 1
    print(count)

#For each person, how many features are available?
if False:
    for k,v in enron_data.items():
        print(k, len(v))

#How many POI's are in the dataset?
if False:
    count = 0
    for i in enron_data.values():
        if i["poi"] == 1:
            count += 1
    print(count)

#What is the total value of the stock belonging to James Prentice?
if False:
    print(enron_data['PRENTICE JAMES']['total_stock_value'])

#How many messages do we have from Wesley Colwell to POI's?
if False:
    print(enron_data['COLWELL WESLEY']['from_this_person_to_poi'])

#What is the value of the stock options exercised by Jeffrey K Skilling?
if False:
    print(enron_data['SKILLING JEFFREY K']['exercised_stock_options'])

#Of Lay, Skilling and Fastow, who took home the most money, how much?
if False:
    print enron_data['SKILLING JEFFREY K']['total_payments']
    print enron_data['LAY KENNETH L']['total_payments']
    print enron_data['FASTOW ANDREW S']['total_payments']

#How many folks have a quantified_salary, known email_address?
if False:
    count_salary = 0
    count_email = 0
    for i in enron_data.values():
        if i['salary'] != 'NaN':
            count_salary += 1
    print(count_salary)

    for i in enron_data.values():
        if i['email_address'] != 'NaN':
            count_email += 1
    print(count_email)

#What percentage of people have NaN for their total_payments?
if False:
    count_NaN = 0
    for i in enron_data.values():
        if i['total_payments'] == 'NaN':
            count_NaN += 1
    print(float(count_NaN)/len(enron_data.keys()))

#What percentage of POI's have NaN for their total_payments?
if True:
    count_NaN = 0
    for i in enron_data.values():
        if i['total_payments'] == 'NaN' and i["poi"]:
            count_NaN += 1
    print(float(count_NaN))
