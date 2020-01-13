# import libararys
import os
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report,accuracy_score

# import data pre-processing functions
from A1.A_feature_extraction import *
from B1.B_feature_extraction import *

# import data models
from A1.A1_model_training import *
from A2.A2_model_training import *
from B1.B1_model_training import *
from B2.B2_model_training import *
from B2.B2_prediction_test import *

# path to all images and parameter settings
global basedir, A_image_dir, B_image_dir, target_size
basedir = './Dataset/dataset'
add_testset_dir = './Dataset/dataset_test'
A_images_dir = os.path.join(basedir,'celeba/img')
A_labels_filename = 'celeba/labels.csv'
B_images_dir = os.path.join(basedir,'cartoon_set/img')
B_labels_filename = 'cartoon_set/labels.csv'
target_size = None
add_testset_A_img = os.path.join(add_testset_dir, 'celeba_test/img')
add_testset_B_dir_path = os.path.join(add_testset_dir,'cartoon_set_test/img')
add_testset_A_labels = 'cartoon_set_test/labels.csv'
add_testset_B_labels_path = os.path.join(add_testset_dir,'celeba_test/labels.csv')


# ======================================================================================================================
# Data preprocessing
# # task A
X, Y_gender, Y_smile = A_get_tvt_dataset(basedir, A_images_dir, A_labels_filename)
# # A1
x_train_gender, x_test_gender, y_train_gender, y_test_gender = train_test_split(X, Y_gender,random_state=0)
# # A2
x_train_smile, x_test_smile, y_train_smile, y_test_smile = train_test_split(X, Y_smile,random_state=0)

# task B

# ############################ uncomment these for retrain task B ##########################
all_image, X_cartoon, y_faceshape, y_eyecolour, faceshapeLabels, eyecolourLabels = B_extract_features_labels(basedir, B_images_dir, B_labels_filename)
# # B1 data for svm model
# x_train_faceshape_svm, x_test_faceshape_svm, y_train_faceshape_svm, y_test_faceshape_svm = train_test_split(X_cartoon, y_faceshape,random_state=0)
# # B1 data for cnn model
x_train_faceshape, x_test_faceshape, y_train_faceshape, y_test_faceshape = train_test_split(all_image, faceshapeLabels,random_state=0)
# # B2 data for svm model
# x_train_eyecolour_svm, x_test_eyecolour_svm, y_train_eyecolour_svm, y_test_eyecolour_svm = train_test_split(X_cartoon, y_eyecolour, random_state = 0)
# # B2 data for cnn model
x_train_eyecolour, x_test_eyecolour, y_train_eyecolour, y_test_eyecolour = train_test_split(all_image, eyecolourLabels,random_state=0)
############################ uncomment these for retrain task B ##########################
# additional test dataset for A
# test_features, test_genderLabels, test_smileLabels = A_get_tvt_dataset(add_testset_dir,add_testset_A_img, add_testset_A_labels)
# additional test dataset for B
test_all_image, test_faceshapeLabels, test_eyecolourLabels = B_load_additional_test_set(add_testset_B_dir_path, add_testset_B_labels_path)
# ======================================================================================================================
# Task A1
model_A1 = A1_classifier(x_train_gender, x_test_gender, y_train_gender, y_test_gender)                 # Build model object.
acc_A1_train = model_A1.svm_train()
acc_A1_test = model_A1.train_validation() # Train model based on the training set (you should fine-tune your model based on validation set.)
# acc_A1_train = model_A1.train_validation()
# acc_A1_test = model_A1.test(test_features, test_genderLabels)   # Test model based on the test set.

print(acc_A1_train)
print(model_A1.train_valid_error)
print(model_A1.accuracy_score)
print("A1 finished")
# Clean up memory/GPU etc...             # Some code to free memory if necessary.
#
#
# ======================================================================================================================
# Task A2
model_A2 = A2_classifier(x_train_smile, x_test_smile, y_train_smile, y_test_smile)
acc_A2_train = model_A2.svm_train()
acc_A2_test = model_A2.train_validation()
# acc_A2_test = model_A2.test(test_features, test_smileLabels)

print(acc_A2_train)
print(model_A2.train_valid_error)
print(model_A2.accuracy_score)
print("A2 finished")
# Clean up memory/GPU etc...
#
#
# ======================================================================================================================
# Task B1
B1_model_name = 'face_shape_model'
# change this directory if you don't want the saved models being replaced
B1_model_dir = './Dataset/B1_trained_CNN_model/'
B1_meta_file_path = os.path.join(B1_model_dir, 'face_shape_model.meta')
B1_checkpoint_path = B1_model_dir
#
# comment the line below to retrain the CNN model
# remember to uncomment assign acc_B1_train to 'TBD' if this line is commented
print("Training CNN for task B1, this is going to take some time")
acc_B1_train = B1_train(100, x_train_faceshape, y_train_faceshape, x_test_faceshape, y_test_faceshape, B1_model_name)
# acc_B1_train = 'TBD'
#
print("Test trained CNN for task B1")
acc_B1_test = show_test_accuracy(test_all_image, B1_meta_file_path, B1_checkpoint_path, test_faceshapeLabels)
print(acc_B1_train)
print("B1 finish")
# Clean up memory/GPU etc...
#
#
# # ======================================================================================================================
# # Task B2
B2_model_name = 'eye-colour-model'
# change this directory if you don't want the saved models being replaced
B2_model_dir = './Dataset/B2_trained_CNN_model/'
B2_meta_file_path = os.path.join(B2_model_dir, 'eye-colour-model.meta')
B2_checkpoint_path = B2_model_dir

# comment the line below to retrain the CNN model
# remember to uncomment assign acc_B2_train to 'TBD' if this line is commented
print("Training CNN for task B2, this is going to take some time")
acc_B2_train = B2_train(300, x_train_eyecolour, y_train_eyecolour, x_test_eyecolour, y_test_eyecolour, B2_model_name)
# acc_B2_train = 'TBD'

print("Test trained CNN for task B2")
acc_B2_test = show_test_accuracy(test_all_image, B2_meta_file_path, B2_checkpoint_path, test_eyecolourLabels)
print(acc_B2_train)
print("B2_finished")
Clean up memory/GPU etc...
#
#
# # ======================================================================================================================
# Print out your results with following format:

print('TA1:{},{};TA2:{},{};TB1:{},{};TB2:{},{};'.format(acc_A1_train, acc_A1_test,
                                                        acc_A2_train, acc_A2_test,
                                                        acc_B1_train, acc_B1_test,
                                                        acc_B2_train, acc_B2_test))
#
# # If you are not able to finish a task, fill the corresponding variable with 'TBD'. For example:
# # acc_A1_train = 'TBD'
# # acc_A1_test = 'TBD'
