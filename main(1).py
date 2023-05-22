import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt
from sklearn import metrics, svm, datasets
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from fastdtw import dtw
from sklearn.preprocessing import StandardScaler

# Daten laden
data = pd.read_csv("./recording-20220629-182534.csv", delimiter=';')
data2 = pd.read_csv("./recording-20220629-182610.csv", delimiter=';')

# Aufteilung in Features und Zielvariable
X = data[['accX', 'accY', 'accZ', 'gyroX', 'gyroY', 'gyroZ', 'time']]
y = data['spellName']

X2 = data2[['accX', 'accY', 'accZ', 'gyroX', 'gyroY', 'gyroZ', 'time']]
y2 = data2['spellName']

X = X.set_index("time")
X2 = X2.set_index("time")

#print(X)

#dist, path = dtw(X, X2)
#print(dist)
#print(path)

dfs = [X, X2]
labels = [y[0], y2[0]]

def get_distance(a, b):
    dist, path = dtw(a, b)
    return dist

def compare_with_data(a):
    lowestDistIndex = -1
    lowestDist = float("inf")
    for index, df in enumerate(dfs):
        curr = get_distance(a, df)
        if curr<lowestDist:
            lowestDistIndex=index
    return labels[lowestDistIndex]

print(compare_with_data(X2))

# KNN-Classifier initialisieren
#knn = NearestNeighbors(n_neighbors=1, metric=get_distance)
#knn.fit([[1],[2]], [y, y2])
#y_pred = knn.predict(X)
#print(y_pred)


