# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 16:15:55 2018

@author: Sam
"""

import cv2
import os

def findFaces(image):
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    # Read the image
    grayimage = cv2.imread(image,0)
    
    # Detect faces in the image
    faces = faceCascade.detectMultiScale(grayimage, scaleFactor=1.1, minNeighbors=5)
    
    return len(faces)

image = "1.png"
print(findFaces(image))