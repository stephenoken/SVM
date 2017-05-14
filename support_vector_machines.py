import sys
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData

import matplotlib.pyplot as plt
import copy
import numpy as np
import pylab as pl


# features_train, labels_train, features_test, labels_test = makeTerrainData()


########################## SVM #################################
### we handle the import statement and SVC creation for you here
from sklearn.svm import SVC

def classify(features_train, labels_train):
    # clf = SVC(kernel="rbf", gamma=1000.0)
    # clf = SVC(kernel="rbf", gamma=1.0)
    clf = SVC(kernel="rbf", C=1.0)
    clf.fit(features_train,labels_train)
    return clf
#### now your job is to fit the classifier
#### using the training features/labels, and to
#### make a set of predictions on the test data



#### store your predictions in a list named pred
# pred = classifiy()
# print(pred)
# print(labels_test)




# from sklearn.metrics import accuracy_score
# acc = accuracy_score(pred, labels_test)

# def submitAccuracy():
#         return acc

# print(submitAccuracy())
