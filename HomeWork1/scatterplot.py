import matplotlib.pyplot as plt
import csv
import pandas as pd

data = 0
lrb = 0
color = 0

def renderscatterplot():
    x = []
    y = []
    for row in data:
        row = row.split(',')
        if(row[0]==lrb):
            x.append(int(row[1]) * int(row[2]))
            y.append(int(row[3]) * int(row[4][0]))
    plt.scatter(x, y, color=color, label=lrb )

myfile = open("balance-scale.data","r")
#myfile = csv.reader(myfile)
#print(myfile)
data = myfile
lrb = "L"
color = "red"
renderscatterplot()
myfile = open("balance-scale.data","r")
data = myfile
lrb = "R"
color = "green"
renderscatterplot()
myfile = open("balance-scale.data","r")
data = myfile
lrb = "B"
color = "blue"
renderscatterplot()
plt.show()