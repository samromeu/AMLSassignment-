# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 16:15:55 2018

@author: Sam
"""

import cv2
import os
datasetDirectory = 'C:\\Users\\Sam\\Desktop\\UNI\\Fourth Year\\Machine learning\\Assignment'
dataset = []

def findFaces(image):
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    # Read the image
    grayimage = cv2.imread(image,0)
    
    # Detect faces in the image
    faces = faceCascade.detectMultiScale(grayimage, scaleFactor=1.1, minNeighbors=1)
    
    return len(faces)

for filename in os.listdir(datasetDirectory):
    if filename.endswith('.png'):
        if findFaces(filename) > 0:
            print(filename,"contains a face")
            dataset.append(filename)
        else:
            print(filename, "does not contain a face")
            
print(dataset)