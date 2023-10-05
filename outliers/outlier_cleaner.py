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
    # errors = list()
    ### your code goes here
    # for prediction, networth, age in zip(predictions, net_worths, ages):
    #     error = prediction - networth
    #     cleaned_data.append((age, prediction, networth, error))

    #cleaned_data.append((1, "hello"))
    for i in range(len(predictions)):
        error = abs(predictions[i] - net_worths[i])
        cleaned_data.append((ages[i], net_worths[i], error))

    cleaned_data.sort(key=lambda x: x[-1])
    #print(cleaned_data)

    print(len(cleaned_data))

    len_to_remove = int(len(cleaned_data)*10/100)
    del cleaned_data[-len_to_remove:]
    print(len(cleaned_data))

    #print(cleaned_data)


    
    return cleaned_data

