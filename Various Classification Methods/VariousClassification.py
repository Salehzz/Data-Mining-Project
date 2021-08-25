
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import svm

#reading data and making data readable
attributes = ['Age','workclass','fnlwgt','education','education_num','marital_status','occupation','relationship','race','sex','capital_gain','capital_loss','hours_per_week','native_country','label']
adults = pd.read_csv('adult.csv',names= attributes)
full_dataset = adults
train_data = adults.drop('label',axis=1)
label = adults['label']
data_binary = pd.get_dummies(train_data)
#spliting data to train and test
x_train, x_test, y_train, y_test = train_test_split(data_binary,label)

algorithmsperformance = []

# KNN
knn_scores = []
train_scores = []
test_scores = []
for n in range(1,25,4):
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(x_train,y_train)
    train_score = knn.score(x_train,y_train)
    test_score = knn.score(x_test,y_test)
    train_scores.append(train_score)
    test_scores.append(test_score)
    print(f'KNN : Training score = {train_score} and Test score = {test_score}','for ', n , 'Neighbors')
    knn_scores.append({'algorithm':'KNN', 'training_score':train_score})
    
plt.scatter(x=range(1,25,4),y=train_scores,c='#d62728')
plt.scatter(x=range(1,25,4),y=test_scores,c='#2ca02c')
plt.show()
#find the best quantity for neighbors
test_scores = np.array(test_scores)
idknn = np.argmax(test_scores)
train_score = train_scores[idknn]
test_score = max(test_scores)
print(f'Best K Neighbors : Training score = {train_score} and Test score = {test_score}','for ', idknn*5+1 , 'Neighbors')
algorithmsperformance.append({'algorithm':'K Neighbors', 'training_score':train_score, 'testing_score':test_score})

# Gaussian Naive Bayes
GNB = GaussianNB()
# Binary data
GNB.fit(x_train,y_train)
train_score = GNB.score(x_train,y_train)
test_score = GNB.score(x_test,y_test)
print(f'Gaussian Naive Bayes : Training score = {train_score} and Test score = {test_score}')
algorithmsperformance.append({'algorithm':'Gaussian Naive Bayes', 'training_score':train_score, 'testing_score':test_score})


# LogisticRegression
logClassifier = LogisticRegression(max_iter = 1000)
logClassifier.fit(x_train,y_train)
train_score = logClassifier.score(x_train,y_train)
test_score = logClassifier.score(x_test,y_test)
print(f'LogisticRegression : Training score = {train_score} and Test score = {test_score}')
algorithmsperformance.append({'algorithm':'LogisticRegression', 'training_score':train_score, 'testing_score':test_score})


#Random Forest
rndTree = RandomForestClassifier()
rndTree.fit(x_train,y_train)
train_score = rndTree.score(x_train,y_train)
test_score = rndTree.score(x_test,y_test)
print(f'Random Forests : Training score = {train_score} and Test score = {test_score}')
algorithmsperformance.append({'algorithm':'Random Forests', 'training_score':train_score, 'testing_score':test_score})

#SVM
svc = svm.SVC(kernel='linear')
scaler = StandardScaler()
scaler.fit(data_binary,label)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)
svc.fit(x_train_scaled,y_train)
train_score = svc.score(x_train_scaled,y_train)
test_score = svc.score(x_test_scaled,y_test)
print(f'SVM : Training score = {train_score} and Test score = {test_score}')
algorithmsperformance.append({'algorithm':'SVM', 'training_score':train_score, 'testing_score':test_score})

#Print all the Scores
for i in algorithmsperformance:
    print(i)