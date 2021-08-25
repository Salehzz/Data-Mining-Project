from clusterData import clusterTwoColumns
from parseCSV import csvToDictArray


# Parse CSV file
parsedData = csvToDictArray()
attributes = parsedData[1]
min_den = 10
gridSize = len(attributes)

# Execute clustering strategy.
for i in range(len(attributes)-2):
    for j in range(i+1, len(attributes)-2):
        clusterTwoColumns(attributes[i], attributes[j], parsedData, min_den, gridSize)
