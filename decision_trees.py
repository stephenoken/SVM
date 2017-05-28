import sys
from class_vis import prettyPicture, output_image 
from prep_terrain_data import makeTerrainData 

import matplotlib.pyplot as plt
import numpy as np 
import pylab as pl 

from sklearn import tree
from sklearn.metrics import accuracy_score

features_train, labels_train, features_test, labels_test = makeTerrainData()

def classifly(train_features, train_labels):
    classifier = tree.DecisionTreeClassifier(min_samples_split=50)
    return classifier.fit(train_features, train_labels)

clf = classifly(features_train,labels_train)

print(accuracy_score(clf.predict(features_test), labels_test))
prettyPicture(clf, features_test, labels_test)
output_image("test.png", "png", open("test.png", "rb").read())

