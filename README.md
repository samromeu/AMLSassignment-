First of all, all .py files should be placed inide the same folder.
Next, four empty folders called 'testing', 'training', 'noise' and 'faces' should be created. The folder containing the original 5000 images should be in a folder called 'dataset'.
Then, run task_A.py, which will iterate through images in the dataset and place the images containing faces in the 'faces' folder and the noisy images in the 'noise' folder.
Next, run splitData.py, which will divide the data into training and testing, 80% and 20% respectively.
From here, you can run either the cnnmodel or the svmmodel, making sure to change the 'class_column' attribute in each file before compiling and runnning to choose the column of the attribute list you wish to train on (1 being hair colour, 2 being eyeglasses etc.).
The predictions and classification report printed will be of the test set conatining 20% of the 'faces' images, and had nothing to do with the training of the model. 
FOR THE SVM MODEL
The training curve is plotted by training the model a second time. If you wish to do so, uncomment line 129 in the SVM model.
