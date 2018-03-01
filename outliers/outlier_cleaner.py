#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):

    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where
        each tuple is of the form (age, net_worth, error).
    """

    cleaned_data = []
    temp = []
    ### your code goes here

    age = []
    net_worth = []
    errors = []

    for i in range(0,89):
        age.append(ages[i][0])
        net_worth.append(net_worths[i][0])
        errors.append(abs(predictions[i][0]-net_worths[i][0]))

    temp = (zip(age,net_worth,errors))
    temp.sort(key=lambda x: x[2], reverse=True)
    cleaned_data = temp[int(len(temp)*0.1):]
    return cleaned_data
