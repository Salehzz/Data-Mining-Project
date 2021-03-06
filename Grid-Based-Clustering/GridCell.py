class GridCell:
    """Grid cell class"""

    # We create the density_threshold variable outside of the constructor
    # because we want to share this value amongst ALL instantiations of
    # the Grid Cell class.
    min_den = 1

    def __init__(self, min_den, xVals, yVals, xPos, yPos):
        """Constructor for Grid cell class."""

        # Instance Variables
        self.cluster = -1           # Cluster cell belongs to.
        self.items = []             # Array of objects contained in cell.
        self.min_den = min_den      # Minimum density of cell.
        self.xVals = xVals          # Tuple containing the minimum and maximum x-values for cell.
        self.yVals = yVals          # Tuple containing the minimum and maximum y-values for cell.
        self.xPos = xPos            # X-Position of the cell.
        self.yPos = yPos            # Y-Position of the cell.

        # X-Bin of the cell.
        self.xBin = "[" + str(xVals[0]) + ".." + str(xVals[1]) + "]"
        # Y-Bin of the cell.
        self.yBin = "[" + str(yVals[0]) + ".." + str(yVals[1]) + "]"

    def isWithinValueRange(self, data):
        """Checks if an attribute value falls within the min and max value range of cell."""
        withinXRange = False
        withinYRange = False
        #print(data['x'],self.xVals[1])
        try:
            if data['x'] >= self.xVals[0] and data['x'] <= self.xVals[1]:
                withinXRange = True

            if data['y'] >= self.yVals[0] and data['y'] <= self.yVals[1]:
                withinYRange = True
        except:
            pass
        return withinXRange and withinYRange

    def addItem(self, item, xAttr, yAttr):
        """Adds item to cell."""
        item["xVal"] = item[xAttr]
        item["yVal"] = item[yAttr]
        item[xAttr] = self.xBin
        item[yAttr] = self.yBin
        self.items.append(item)

    def assignToCluster(self, cluster):
        """Assigns cell to a cluster"""
        self.cluster = cluster

    def getDensityCount(self):
        """Returns a count of the cell's density."""
        return len(self.items)

    def getPosition(self):
        """Returns the cell's position."""
        return [self.xPos, self.yPos]

    def isAssignedToCluster(self):
        """Returns whether or not cell is assigned to cluster."""
        return self.cluster > 0

    def isDense(self):
        """Returns whether or not grid cell is dense."""
        return len(self.items) >= self.min_den

    def isAdjacentCell(self, cell):
        """Checks if the input dense-cell is adjacent to itself."""
        xPos = cell.getPosition()[0]
        yPos = cell.getPosition()[1]

        # Check if input cell is above cell
        if xPos == self.xPos and (yPos - 1) == self.yPos:
            return True

        # Check if input cell is below cell
        if xPos == self.xPos and (yPos + 1) == self.yPos:
            return True

        # Check if input cell is to the right of cell
        if (xPos - 1) == self.xPos and yPos == self.yPos:
            return True

        # Check if input cell is to the left of cell
        if (xPos + 1) == self.xPos and yPos == self.yPos:
            return True

        return False

    def getCellCluster(self):
        return self.cluster

    def getCellItems(self):
        return self.items

    def getBins(self):
        return { "xBin": self.xBin, "yBin": self.yBin }
