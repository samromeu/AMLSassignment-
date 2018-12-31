# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 15:11:16 2018

@author: Sam
"""

import cv2
import csv
import os
import numpy as np
from tqdm import tqdm

datasetDirectory = 'C:\\Users\\Sam\\Desktop\\UNI\\Fourth Year\\Machine learning\\Assignment\\faces'
trainingDirectory = 'C:\\Users\\Sam\\Desktop\\UNI\\Fourth Year\\Machine learning\\Assignment\\training'
testingDirectory = 'C:\\Users\\Sam\\Desktop\\UNI\\Fourth Year\\Machine learning\\Assignment\\testing'

splitRatio = 0.8

train_thresh = round(splitRatio * len(os.listdir(datasetDirectory)))

for i in range(len(os.listdir(datasetDirectory))):
    filename = os.listdir(datasetDirectory)[i]
    if filename.endswith('.png'):
        if i < train_thresh:
            image = cv2.imread(filename)
            currentDir = "training"
            cv2.imwrite(os.path.join(trainingDirectory , filename), image)
        elif i > train_thresh and i <= len(os.listdir(datasetDirectory)):
            image = cv2.imread(filename)
            currentDir = "testing"
            cv2.imwrite(os.path.join(testingDirectory , filename), image)
    
        print("Saving", filename, "to", currentDir)
