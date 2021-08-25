import pandas as pd
from sklearn.model_selection import train_test_split


def csvToDictArray():
    """Parse incoming data set into an array of dictionaries."""
    attributes = ['Age','workclass','fnlwgt','education','education_num','marital_status','occupation','relationship','race','sex','capital_gain','capital_loss','hours_per_week','native_country','label']
    adults = pd.read_csv('adult.csv',names= attributes)
    #adults_test = pd.read_csv('adult_test.csv',names= attributes)
    #train_data = adults.drop('label',axis=1)
    #test_data = adults_test.drop('label',axis=1)
    data = adults
    #label = adults['label'].append(adults_test['label'])
    data_binary = pd.get_dummies(data)
    #label_binary = pd.get_dummies(label)
    #x_train, x_test, y_train, y_test = train_test_split(data_binary,label_binay)
    

    data_set = []
    attributes = list(data_binary.head(0))
    #print(data_binary.head(5))
    #print(len(list(data_binary.index)))
    valuesPerAttr = {attr: [] for attr in attributes}   # Dictionary of all values per data attribute.

    for i in range(len(list(data_binary.index))):
        item = {}
        #print(i)
        for j in range(len(attributes)):
            attr = attributes[j]
            value = float(data_binary[attr][i])
            item[attr] = value
            valuesPerAttr[attr].append(value)
        if(bool(item)):
            data_set.append(item)

    return [data_set, attributes , valuesPerAttr]
