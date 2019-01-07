# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 15:41:34 2018

@author: Sam
"""
import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories
from random import shuffle # mixing up currently ordered data that might lead our network astray in training.
from tqdm import tqdm  
import csv
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Convolution2D
from keras.layers. normalization import BatchNormalization
from keras.callbacks import EarlyStopping
K.set_image_dim_ordering('tf')


IMG_SIZE = 64 #Sets the pixel length and height for all images
class_column = 5 #Determines which column in the attribute_list.csv file is to be examined. E.g. 1 for hair colour, 2 for eyeglasses
LR = 1e-3

datasetDirectory = './faces' 
TRAIN_DIR = './training'
validationDirectory = './validation'
TEST_DIR = './testing'

"""
This section of if statements prepares the respective labels for the confusion matrix depending on which 
class_column is selected. It also tells the model whether it should build a multiclass or a binary classifier.
"""
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

"""
The 'label_img' function uses the provided attribute_list to label images with their respective
class using one hot vector encoding, as well as labeling the images with their respective filename
"""

def label_img(img, class_column, class_type): #Function to label each image from 'attribute_list.csv'
    with open('attribute_list.csv', mode='r') as f:
        label_file = csv.reader(f, delimiter=',', quotechar='"', quoting =csv.QUOTE_MINIMAL)
        for row in label_file:
            if img.split('.')[-2] == row[0]: #If img number corresponds to img in attribute list
                if class_type == 'binary':
                    if row[class_column] == "-1": #If the label for that column is equal to -1
                        return [1,0] #Label as [0,1] (one hot-vector encoding)
                    else:
                        return [0,1] #Else label as [1,0] 
                elif class_type == 'multiclass':
                    if row[class_column] == "0": #If the label for that column is equal to 1
                        return [1,0,0,0,0,0] #Corresponds to bald
                    elif row[class_column] == "1":
                        return [0,1,0,0,0,0] #Corresponds to Blond hair
                    elif row[class_column] == "2":
                        return [0,0,1,0,0,0] #Corresponds to Ginger hair
                    elif row[class_column] == "3":
                        return [0,0,0,1,0,0] #Corresponds to Brown hair
                    elif row[class_column] == "4":
                        return [0,0,0,0,1,0] #Corresponds to Black hair
                    elif row[class_column] == "5": 
                        return [0,0,0,0,0,1]  #Corresponds to Gray hair
                    else: return [0,0,0,0,0,0]

"""
Converts training images to raw pixel data and labels them using the attribute list
"""    

def load_training_data():
    train_data = [] #Creates empty array for training data
    for img in tqdm(os.listdir(TRAIN_DIR)): #Iterates through all images in the training directory
        label = label_img(img, class_column, class_type) #Labels images with their corresponding feature label
        path = os.path.join(TRAIN_DIR, img)
        img = cv2.imread(path) #Reads in image
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE)) #Resizes image to (IMG_SIZE,IMG_SIZE) to reduce feature size
        train_data.append([np.array(img), label]) #Appends the image data along with its output label to train_data array
            
    shuffle(train_data) #Shuffles training data to allow for random training every time
    return train_data


"""
Converts testing images to raw pixel data and labels them using the attribute list
"""    
def load_test_data():
    test_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        label = label_img(img, class_column, class_type)
        path = os.path.join(TEST_DIR, img)
        img_num = img.split('.')[0]
        img = cv2.imread(path)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        test_data.append([np.array(img), label, img_num])
        
    shuffle(test_data)
    return test_data


train_data = load_training_data() #Loads training data
test_data = load_test_data() #Loads testing data

trainImages = np.array([i[0] for i in train_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 3) #Reshapes training image data to 1D vector and appends to an array
trainLabels = np.array([i[1] for i in train_data]) #Appends training labels to an array

testImages = np.array([i[0] for i in test_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 3) #Reshapes testing image data to 1D vector and appends to an array
testLabels = np.array([i[1] for i in test_data]) #Appends training labels to an array
testNum = np.array([i[2] for i in test_data]) #Appends filename to an array


"""
Build CNN model used for image classification. All individual layers are commented and an explanation
of the function of each one is provided in the report
"""

def cnn_model(class_type):
# Initialising 
    cnn_model = Sequential()
    
    # 1st conv. layer
    cnn_model.add(Conv2D(32, (3, 3), input_shape = (IMG_SIZE, IMG_SIZE, 3), activation = 'tanh'))
    cnn_model.add(MaxPooling2D(pool_size = (2, 2)))
    
    # 2nd conv. layer
    cnn_model.add(Conv2D(32, (3, 3), activation = 'tanh')) 
    cnn_model.add(MaxPooling2D(pool_size = (2, 2)))
    
    # 3nd conv. layer
    cnn_model.add(Conv2D(64, (3, 3), activation = 'tanh')) 
    cnn_model.add(MaxPooling2D(pool_size = (2, 2)))
    
    # Flattening
    cnn_model.add(Flatten())
    
    # Full connection
    cnn_model.add(Dense(units = 64, activation = 'tanh'))
    cnn_model.add(Dropout(0.5))
    
    if class_type == 'multiclass': #6 output units with softmax activation if multiclass
        cnn_model.add(Dense(units = 6, activation = 'softmax'))
    elif class_type == 'binary': #2 output units with sigmoid activation if binary
        cnn_model.add(Dense(units = 2, activation = 'sigmoid'))
    
    
    cnn_model.compile(optimizer = 'adam', 
                           loss = 'binary_crossentropy', 
                           metrics = ['accuracy']) 

    cnn_model.summary()
    return cnn_model

"""
Declares callback to implement early stopping 
"""
earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=0,
                              verbose=0, mode='auto')


"""
Builds cnn_model by examining the class type
"""
cnn_model = cnn_model(class_type)

"""
Fits the model using the training data with a validation split of 20%, a batch size of 50 and 5 epochs
"""
history = cnn_model.fit(trainImages, trainLabels, validation_split=0.2, batch_size = 50,
                        epochs = 5, verbose = 1, callbacks = [earlyStopping])

"""
Evaluates the model using test data and returns loss and accuracy
"""
loss, acc = cnn_model.evaluate(testImages, testLabels, verbose = 1)

"""
Predicts the class for each test image
"""
rounded_predictions = cnn_model.predict_classes(testImages)

filepredictions = []
labels = []

"""
Appends all testlabels to a list
"""
for i in testLabels:
    labels.append(np.argmax(i)) 

"""
Appends both the test filenames and the rounded predictions to an array so we can see what each
image has been predicted as.
"""    
for i in range(len(test_data)):
    filepredictions.append([testNum[i]+'.png', rounded_predictions[i]])
filepredictions.sort(key=lambda x: int(os.path.splitext(x[0])[0]))

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('CNN Model Accuracy for Cartoon vs. Human with early stopping and dropout')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Cross-validaion score', 'Training'], loc='bottom right')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss for Cartoon vs. Human without early stopping and dropout')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Cross-validaion score', 'Training'], loc='upper left')
plt.show()

"""
Creates a confusion matrix for the test data
"""
cm = confusion_matrix(labels, rounded_predictions)

"""
Function obtained from sklearn's website used to plot a confusion matrix
"""
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()