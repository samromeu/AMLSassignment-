# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 16:15:55 2018

@author: Sam
"""

import cv2
import os
import csv

datasetDirectory = 'C:\\Users\\Sam\\Desktop\\UNI\\Fourth Year\\Machine learning\\Assignment'
faceDirectory = 'C:\\Users\\Sam\\Desktop\\UNI\\Fourth Year\\Machine learning\\Assignment\\faces'
dataset = []
csvData = []

def findFace(image):
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
    # Read the image
    grayimage = cv2.imread(image,0)    
    # Detect faces in the image
    faces = faceCascade.detectMultiScale(grayimage, scaleFactor=1.05, minNeighbors=3)   
    return len(faces)

for filename in os.listdir(datasetDirectory):
    if filename.endswith('.png'):
        if findFace(filename) > 0:
            print(filename,"contains a face")
            image = cv2.imread(filename)
            cv2.imwrite(os.path.join(faceDirectory , filename), image)
        else:
            print(filename, "does not contain a face")
            
print(dataset)