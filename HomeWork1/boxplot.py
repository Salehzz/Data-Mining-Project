import matplotlib.pyplot as plt
import csv
import pandas as pd

def renderboxplot(data):
    dictionary = {
        "L": [[], [], [], []],
        "R": [[], [], [], []],
        "B": [[], [], [], []],
    }
    for row in data:
        dictionary[row[0]][0].append(int(row[1]))
        dictionary[row[0]][1].append(int(row[2]))
        dictionary[row[0]][2].append(int(row[3]))
        dictionary[row[0]][3].append(int(row[4]))
    _ , (ax1, ax2, ax3) = plt.subplots(nrows = 3)
    ax1.boxplot(dictionary["R"])
    ax2.boxplot(dictionary["L"])
    ax3.boxplot(dictionary["B"])
myfile = open("balance-scale.data","r")
data = csv.reader(myfile)
renderboxplot(data)
plt.show()