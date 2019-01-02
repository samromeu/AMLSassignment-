# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 23:15:49 2018

@author: Sam
"""
import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories
from random import shuffle # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import tqdm  
import csv
import itertools
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import cross_val_score

datasetDirectory = './faces'
TRAIN_DIR = './training'
validationDirectory = './validation'
TEST_DIR = './testing'

IMG_SIZE = 64 #Sets the pixel length and height for all images
class_type = 'binary' # Determines whether the model will be performung a binary or multiclass classification
class_column = 3 #Determines which column in the attribute_list.csv file is to be examined. E.g. 1 for hair colour, 2 for eyeglasses 

def label_img(img, class_column, class_type): #Function to label each image from 'attribute_list.csv'
    with open('attribute_list.csv', mode='r') as f:
        label_file = csv.reader(f, delimiter=',', quotechar='"', quoting =csv.QUOTE_MINIMAL)
        for row in label_file:
            if img.split('.')[-2] == row[0]: #If img number corresponds to img in attribute list
                if class_type == 'binary':
                    if row[class_column] == "-1": #If the label for that column is equal to -1
                        return 0 #Label as [0,1] (not smiling) (one hot-vector encoding)
                    else:
                        return 1 #Else label as [1,0] (smiling)
                elif class_type == 'multiclass':
                    if row[class_column] == "1": #If the label for that column is equal to -1
                        return 1 #Corresponds to Blond hair
                    elif row[class_column] == "2":
                        return 2 #Corresponds to Ginger hair
                    elif row[class_column] == "3":
                        return 3 #Corresponds to Brown hair
                    elif row[class_column] == "4":
                        return 4 #Corresponds to Black hair
                    elif row[class_column] == "5":
                        return 5 #Corresponds to Gray hair
                    else: 
                        return 0 #Corresponds to no hair

def load_training_data():  #Function to convert training images to pixel data and label them  
    train_data = [] #Creates empty array for training data
    for img in tqdm(os.listdir(TRAIN_DIR)): #Iterates through all images in the training directory
        label = label_img(img, class_column, class_type) #Labels images with their corresponding feature label
        path = os.path.join(TRAIN_DIR, img) 
        img = cv2.imread(path) #Reads in image
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE)) #Resizes image to (IMG_SIZE,IMG_SIZE) to reduce feature size
        img = img.flatten() #Converts (64x64x3) vector into 1D array with each, each column of pixel data corresponding to a separate feature
        train_data.append([np.array(img), label]) #Appends the flattened image data along with its output label to train_data array
            
    shuffle(train_data) #Shuffles training data to allow for random training every time 
    return train_data

def load_test_data(): #Function to convert testing images to pixel data and label them  
    test_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        label = label_img(img, class_column, class_type)
        path = os.path.join(TEST_DIR, img)
        img_num = img.split('.')[0] #Also saves the file number to evaluate accuracy when compared to attribute_list
        img = cv2.imread(path)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        img = img.flatten()
        test_data.append([np.array(img), label, img_num])
        
    shuffle(test_data)
    return test_data

train_data = load_training_data() #Loads training data
test_data = load_test_data() #Loads testing data


trainImages = np.array([i[0] for i in train_data]) #Extracts flattened pixel data from training_data
trainLabels = np.array([i[1] for i in train_data]) #Extracts correspodning labels from training_data

testImages = np.array([i[0] for i in test_data]) #Extracts flattened pixel data from test_data
testLabels = np.array([i[1] for i in test_data]) #Extracts correspodning labels from test_data
testNum = np.array([i[2] for i in test_data])


classifier = svm.SVC(gamma=0.001, verbose = 1, kernel='poly',decision_function_shape='ovo') #Builds SVM model using a polynomial kernel
classifier.fit(trainImages,trainLabels) #Trains model using training images and training labels
accuracy_scores = cross_val_score(classifier, testImages, testLabels, scoring='accuracy', cv=10) #Evaluates accuracy of model using test images and labels and 10 fold cross-validation

labeledPredictions = []

predictions = classifier.predict(testImages) #Predicts classes for test images
for i in range(len(predictions)):
    labeledPredictions.append([testNum[i]+'.png',predictions[i]]) #Labels predictions with the name of the correspodning file being predicted
    
labeledPredictions.sort(key=lambda x: int(os.path.splitext(x[0])[0])) #Sorts filenames in ascending order 
