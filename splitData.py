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

datasetDirectory = './faces'
trainingDirectory = './training'
testingDirectory = './testing'

splitRatio = 0.8 #Sets the training:test ratio at 80:20

train_thresh = round(splitRatio * len(os.listdir(datasetDirectory))) #Calculates the number of images in the training set

for i in range(len(os.listdir(datasetDirectory))): #Iterates through every file in the face directory
    filename = os.listdir(datasetDirectory)[i] #Indexes every file with number 'i'
    
    if filename.endswith('.png'): #Only applies the following code to png images
        if i < train_thresh: #For all images below the training threshold (80% of images)
            image = cv2.imread(filename) #Read in image
            currentDir = "training"
            cv2.imwrite(os.path.join(trainingDirectory , filename), image) #Save image to the training directory folder
        
        elif i >= train_thresh and i <= len(os.listdir(datasetDirectory)): #For all images above training threshold (20% of images)
            image = cv2.imread(filename) #Read in image
            currentDir = "testing"
            cv2.imwrite(os.path.join(testingDirectory , filename), image) #Save image to testing directory folder
    
        print("Saving", filename, "to", currentDir) #Print the current image being saved and what folder it is being saved to

