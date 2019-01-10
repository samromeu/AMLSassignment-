import cv2
import os
import csv

datasetDirectory = './Assignment'
faceDirectory = './faces'
noiseDirectory = './noise'
csvData = []

def findFace(image): #Function to examine image and determine whether or not it contains a face (0 for no 1 for yes)
    # Apply frontalface training data
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml") 
    # Read the image in grayscale (fewer features to examine)
    grayimage = cv2.imread(image,0)    
    # Detect faces in the image
    faces = faceCascade.detectMultiScale(grayimage, scaleFactor=1.1, minNeighbors=3) 
    
    # scaleFactor determines the % by which an image is shrunk after each iteration before determinign whether or not
    # it contains a face. 1.1 means the image will shrink by 10% each iteration
    
    # minNeighbors determines the number of times a potential face needs to be identified before it is tested positive
    # for containing a face
   
    return len(faces)

for filename in os.listdir(datasetDirectory): #Iterates through every file in the dataset directory
    if filename.endswith('.png'): #Only applies the following code to png images
        if findFace(filename) > 0: #If the file conatains a face
            print(filename,"contains a face") #Print that the file contains a face
            image = cv2.imread(filename) #Read in said filename as an image
            cv2.imwrite(os.path.join(faceDirectory , filename), image) #Write this image to the faceDirectory folder
            faceval = 1 #Set current facetag to 1 
        else:
            print(filename, "does not contain a face")
            image = cv2.imread(filename)
            cv2.imwrite(os.path.join(noiseDirectory , filename), image)
            faceval = -1
        csvData.append([filename,faceval]) #Append the filename along with its facetag to an array called csvData
        
csvData.sort(key=lambda x: int(os.path.splitext(x[0])[0])) #Sort the csvData array by ascending file number

with open('task_A.csv', 'w') as csvFile: #Create a new .csv file called task_A.csv
    writer = csv.writer(csvFile)
    writer.writerows(csvData) #Wrote the csvData array to this file
csvFile.close()
