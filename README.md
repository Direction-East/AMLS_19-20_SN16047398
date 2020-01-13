# AMLS_19-20 assignment1
## Xudong Wu SN:16047398

### This is the code implementation repository for UCL ELEC0134(19-20) assignment1.


## main.py
Running the script main.py, the program will start from feature extraction and data preprocessing of the dataset given, and then go through the model training and testing for each task (A1->A2->B1->B2) in order. A table with the training error and test error of all 4 models will be generated as the final output of the script. **CAUTION: The whole process takes very very very long time.** Mainly because the dataset is large. Most of the time were spent on the feature extraction (dlib's 68 point extraction for all images) and the CNN training process. **It is recommended to use the already trained model from the trained model in the Dataset folder** It is set train new model, if wanted to using trained model, simply comment the lines that the training takes place (indicated with comments in the script)

## Dataset folder
This folder contains the original dataset given and the additional test dataset. Also the trained CNN models for task B1 and B2 are also stored here in their corresponding folders. **They will be replaced if the models are being retrained**

## A1, A2, B1, B2 folders
These folders contains core codes for the assignment. The feature extraction and model training and testing happens here.
