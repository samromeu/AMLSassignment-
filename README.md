First of all, all .py files should be placed inide the same folder. Next, four empty folders called 'testing', 'training', 'noise' and faces should be created. The folder containing the original 5000 images should be called 'dataset'
Next, run task_A.py, which will iterate through images in the dataset and place the images containing faces in the 'faces' folder and the noisy images in the 'noise' folder.
Next, run splitData.py, which will divide the data into training and testing, 80% and 20% respectively.
From here, you can run either the cnnmodel or the svmmodel, making sure to change the 'class_column' attribute in each file before compiling and runnning to choose the column of the attribute list you wish to train on (1 being hair colour, 2 being eyeglasses etc.)
Both will output training curves along with the labeled predictions and accuracy.
