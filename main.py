'''
Author: Kevin He
Date: 2022-06-30
'''

from copy import deepcopy
import math
from matplotlib import pyplot as plt
import pandas as pd
import random

# Constants
COLORS = ["red", "orange", "yellow", "green", "blue", "indigo", "purple"]

# Classes
class Point:
    '''
    Class Name: Point
    Description: To store information of points on the graph.
                 This includes, positional data, and the classifier type
    '''
    def __init__(self, p_x, p_y, p_classifier=None):
        self.x = p_x
        self.y = p_y
        self.classifier = p_classifier  # None, 1+

    def __str__(self):
        return "x: " + str(self.x) + ", y: " + str(self.y) + ", classifier: " + str(self.classifier)

    def setClassifier(self, p_classifier):
        assert type(p_classifier) == int, "Int is required!"
        self.classifier = p_classifier

class Centroid:
    '''
    Class Name: Centroid
    Description: Stores information on the centroid points.
                 This includes, positional data of where the centroid is, classifier type, and a list of points
                 that are part of the same classifier.
    '''
    def __init__(self, p_x, p_y, p_classifier=None):
        self.x = p_x
        self.y = p_y
        self.classifier = p_classifier  # None, 1+
        self.pointsList = []

    def __str__(self):
        return "x: " + str(self.x) + ", y: " + str(self.y) + ", classifier: " + str(self.classifier)

    def __len__(self):
        return len(self.pointsList)

    def setClassifier(self, p_classifier):
        assert type(p_classifier) == int, "Int is required!"
        self.classifier = p_classifier

    def addPoint(self, p_point):
        assert type(p_point) == Point, "Point is required!"
        self.pointsList.append(p_point)

    def resetPoints(self):
        self.pointsList.clear()

def GeneratePointsList(p_dataList):
    '''
    Geneerates a List of points given an array of data
    :param p_dataList: 2D array of Integers
    :return: List of Points
    '''
    pList = []
    for i, data in enumerate(p_dataList):
        p = Point(data[0], data[1])
        pList.append(p)

    return pList

def EuclideanDistancePointCentroid(p_point, p_centroid):
    '''
    Obtain the Euclidean Distance Between a given Point and a Centroid
    :param p_point: A Point to calculate Euclidean Distance with
    :param p_centroid: A Centroid to calculate Euclidean Distance with
    :return: float value of the Euclidean distance between Point and Centroid
    '''
    a = math.pow(p_point.x - p_centroid.x, 2)
    b = math.pow(p_point.y - p_centroid.y, 2)
    return math.sqrt(a + b)

def PlotData(p_pointsList, p_centroidList=[], p_xTitle="f1", p_yTitle="f2"):
    '''
    Generates a graph and output and image.
    Given a list of Points, it will graph the points onto the graph
    In addition, given a list of Centroids, and it'll graph the centroids too
    :param p_pointsList: List of Points
    :param p_centroidList: List of Centroids
    :param p_xTitle: Custom X-Axis name
    :param p_yTitle: Custom Y-Axis name
    :return: None
    '''
    plt.rcParams["figure.figsize"] = [10, 8]
    plt.rcParams["figure.autolayout"] = True
    fig = plt.figure()
    ax = fig.add_subplot()

    for ci, i, in enumerate(p_pointsList):
        mycolor = COLORS[i.classifier] if i.classifier is not None else "#000000"
        ax.scatter(i.x, i.y, c=mycolor, s=150, linewidths=1, edgecolors='#8D99AE')

    for ci, i, in enumerate(p_centroidList):
        mycolor = COLORS[i.classifier] if i.classifier is not None else "#000000"
        ax.scatter(i.x, i.y, c=mycolor, s=50, linewidths=2, edgecolors='#8D99AE')

    plt.xlim(0, 90)
    plt.ylim(0, 90)
    plt.xlabel(p_xTitle)
    plt.ylabel(p_yTitle)
    plt.title("Scatter Plot")
    plt.show()
    return

def CreateCentroids(p_pointsList, p_k):
    '''
    Given a list of Points, it will pick random points from the list. Points picked must be unique
    Pick upto p_k amount of random Points
    For each random point, create a Centroid and take the random point's position on the graph
    Assign the Point with a classifier (related to the Centroid)
    Return the Centroid List
    :param p_pointsList: List of Points
    :param p_k: integer number corresponding to k clusters
    :return: List of Centroids
    '''
    plLength = len(p_pointsList)
    counter = 0
    randomPointsList = []

    while counter < p_k:
        rNum = random.randint(0, plLength - 1)
        rPoint = p_pointsList[rNum]
        if rPoint not in randomPointsList:
            randomPointsList.append(rPoint)
            counter += 1

    centroidList = []
    for i, data in enumerate(randomPointsList):
        data.setClassifier(i)
        c = Centroid(data.x, data.y, data.classifier)
        centroidList.append(c)

    return centroidList


def MinimumDistance(p_eDistanceList):
    '''
    Given a list of float distances, choose the index with the smallest distance
    :param p_eDistanceList: List of floats
    :return: Index with the smallest distance
    '''
    count = -1
    distance = 9999999  # maxsize

    for i, data in enumerate(p_eDistanceList):
        if data < distance:
            distance = data
            count = i

    return count


def ResetCentroidList(p_centroidList):
    '''
    Removes all Points from the each Centroid in a list
    :param p_centroidList: List of Centroids
    :return: None
    '''
    for i in p_centroidList:
        i.resetPoints()

    return


def RepositionCentroidList(p_centroidList):
    '''
    Given a list of Centroids, Change their position based on the average positions of each
    point with the same classifier as the Centroid
    :param p_centroidList: List of Centroids (Referenced)
    :return: None
    '''
    for i in p_centroidList:
        totalX = 0
        totalY = 0
        count = 0
        for j in i.pointsList:
            totalX += j.x
            totalY += j.y
            count += 1
        i.x = totalX / count
        i.y = totalY / count
    return


def CentroidData(p_centroidList):
    '''
    Outputs data of the Centroids given a list
    Will print out Centroid number, colour, size, and position
    :param p_centroidList: List of Centroids
    :return: None
    '''
    for data in p_centroidList:
        print("Centroid Classifier", "#" + str(data.classifier), "[" + COLORS[data.classifier].upper() + "]")
        print("  Size:", len(data))
        print("  X Position: {:.4f} Y Position: {:.4f}".format(data.x, data.y))
        print()

    return


def kmeans(p_pointList, p_k=1):
    '''
    The Algorithm used to generate k-means clustering.
    After the algorithm is done, it will draw a graph and output what the final result of the clustering
    :param p_pointList: List of Points
    :param p_k: Integer based on how many clusters
    :return: None
    '''
    classifierModified = True
    f_pointsList = deepcopy(p_pointList)
    f_centroidsList = CreateCentroids(f_pointsList, p_k)
    PlotData(f_pointsList, f_centroidsList)
    print()

    counter = 0
    while classifierModified:
        counter += 1
        classifierModified = False
        ResetCentroidList(f_centroidsList)

        for i in f_pointsList:  # main loop
            edistanceList = []
            for j in f_centroidsList:  # loop through centroids
                dist = EuclideanDistancePointCentroid(i, j)
                edistanceList.append(dist)

            verdict = MinimumDistance(edistanceList)
            if verdict != i.classifier:
                if classifierModified == False:
                    classifierModified = True

                i.setClassifier(verdict)

            f_centroidsList[i.classifier].addPoint(i)

        RepositionCentroidList(f_centroidsList)
    PlotData(f_pointsList, f_centroidsList)
    CentroidData(f_centroidsList)


if __name__ == '__main__':
    # Obtains the Data
    mainData = pd.read_csv("data/kmeans.csv", index_col=None, header=0, engine='python')
    # Convert from dataframe to a 2D list
    mainData = mainData.values.tolist()

    # Convert the 2D list to List of Points
    pointsList = GeneratePointsList(mainData)
    # Plot the starting of what the data looks like
    PlotData(pointsList)
    # Generate the k-means clustering
    kmeans(pointsList,2)

