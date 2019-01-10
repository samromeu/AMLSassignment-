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
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

datasetDirectory = './faces'
TRAIN_DIR = './training'
validationDirectory = './validation'
TEST_DIR = './testing'

IMG_SIZE = 64 #Sets the pixel length and height for all images
class_column = 1 #Determines which column in the attribute_list.csv file is to be examined. E.g. 1 for hair colour, 2 for eyeglasses 

if class_column == 1: 
    classes = ['Bald','Blond','Ginger','Brown','Black','Gray']
    class_type = 'multiclass'
elif class_column == 2:
    classes = ['No glasses','Glasses']
    class_type = 'binary'
elif class_column == 3:
    classes = ['Not smiling','Smiling']
    class_type = 'binary'
elif class_column == 4:
    classes = ['Old','Young']
    class_type = 'binary'
elif class_column == 5:
    classes = ['Cartoon','Human']
    class_type = 'binary'

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
#

trainImages = np.array([i[0] for i in train_data]) #Extracts flattened pixel data from training_data
trainLabels = np.array([i[1] for i in train_data]) #Extracts correspodning labels from training_data

testImages = np.array([i[0] for i in test_data]) #Extracts flattened pixel data from test_data
testLabels = np.array([i[1] for i in test_data]) #Extracts correspodning labels from test_data
testNum = np.array([i[2] for i in test_data])

parameters = {'kernel': ['linear','poly'], 'C': [0.01, 0.1, 1], 'tol':[1e-3,1e-2,1e-1]} #Parameters for gridsearch/randomsearch

classifier = svm.SVC(gamma='auto', C = 0.01, verbose = 1, kernel='poly') #Builds SVM model using a polynomial kernel
classifier.fit(trainImages,trainLabels) #Trains model using training images and training labels
accuracy_scores = cross_val_score(classifier, testImages, testLabels, scoring='accuracy', cv=5) #Evaluates accuracy of model using test images and labels and 10 fold cross-validation

labeledPredictions = []

predictions = classifier.predict(testImages) #Predicts classes for test images
for i in range(len(predictions)):
    labeledPredictions.append([testNum[i]+'.png',predictions[i]]) #Labels predictions with the name of the correspodning file being predicted
    
labeledPredictions.sort(key=lambda x: int(os.path.splitext(x[0])[0])) #Sorts filenames in ascending order 

print(classification_report(testLabels, predictions, target_names=classes))
print("Labeled predictions for test set:", labeledPredictions)
cm = confusion_matrix(testLabels, predictions)
print(cm, classes)

#plot_learning_curve(classifier, title = 'Learning curve for SVM', testImages, testLabels)


"Function to plot learning curve obtained from SKlearn's website"

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt
