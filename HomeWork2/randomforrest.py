import csv
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data = csv.reader(open('diabetes.csv'))
train = []
for row in data:
    train.append(list(row))
y = []
x = []
for i in range(1, len(train)):
    y.append((train[i][8]))
for i in range(1, len(train)):
    mylist = []
    for k in range(8):
        mylist.append((train[i][k]))
    x.append(mylist)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
classifier = RandomForestClassifier()
classifier.fit(x_train, y_train)
y_prediction = classifier.predict(x_test)
print("Result : ")
print(classification_report(y_test, y_prediction))
print("Confusion Matrix :")
confusion = confusion_matrix(y_test, y_prediction)
for i in confusion:
    print(i)
print()
tp = confusion[0][0]
tn = confusion[1][1]
print("Accuracy : "+str((tp+tn)/(len(y_test))))
metrics.plot_roc_curve(classifier, x_test, y_test)
plt.show()