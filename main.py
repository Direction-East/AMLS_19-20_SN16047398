# IMPORT LIBARARYS
import os
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report,accuracy_score

from A1.A_feature_extraction import *
from A1.A1_model_training import *
from B2.B2A1_model_training import *


# PATH TO ALL IMAGES AND PARAMETER SETTINGS
global basedir, A_image_dir, B_image_dir, target_size
basedir = './Dataset/dataset'
A_images_dir = os.path.join(basedir,'celeba/img')
A_labels_filename = 'celeba/labels.csv'
B_images_dir = os.path.join(basedir,'cartoon_set/img')
B_labels_filename = 'cartoon_set/labels.csv'
target_size = None


# # ======================================================================================================================
# # Data preprocessing
# data_train, data_val, data_test = data_preprocessing(args...)
# task A
X, Y_gender, Y_smile = A_get_tvt_dataset(basedir, A_images_dir, A_labels_filename)
# A1
x_train_gender, x_test_gender, y_train_gender, y_test_gender = train_test_split(X, Y_gender,random_state=0)
# A2
x_train_smile, x_test_smile, y_train_smile, y_test_smile = train_test_split(X, Y_smile,random_state=0)

# task B
all_image, X_cartoon, y_faceshape, y_eyecolour, faceshapeLabels, eyecolourLabels = B_extract_features_labels(basedir, B_images_dir, B_labels_filename)
# B1
x_train_faceshape_svm, x_test_faceshape_svm, y_train_faceshape_svm, y_test_faceshape_svm = train_test_split(X_cartoon, y_faceshape,random_state=0)
x_train_faceshape, x_test_faceshape, y_train_faceshape, y_test_faceshape = train_test_split(all_image, faceshapeLabels,random_state=0)
# B2
x_train_eyecolour_svm, x_test_eyecolour_svm, y_train_eyecolour_svm, y_test_eyecolour_svm = train_test_split(X_cartoon, y_eyecolour, random_state = 0)
x_train_eyecolour, x_test_eyecolour, y_train_eyecolour, y_test_eyecolour = train_test_split(all_image, eyecolourLabels,random_state=0)
# # ======================================================================================================================
# # Task A1
model_A1 = A1_classifier(x_train_gender, x_test_gender, y_train_gender, y_test_gender)                 # Build model object.
###### should get an accuracy here /downarrow ###############
acc_A1_train = model_A1.svm_train() # Train model based on the training set (you should fine-tune your model based on validation set.)
acc_A1_test = model_A1.test()   # Test model based on the test set.
print(model_A1.accuracy_score)
# Clean up memory/GPU etc...             # Some code to free memory if necessary.
#
#
# # ======================================================================================================================
# # Task A2
model_A1 = A1_classifier(x_train_smile, x_test_smile, y_train_smile, y_test_smile)
###### should get an accuracy here /downarrow ###############
acc_A2_train = model_A2.svm_train()
acc_A2_test = model_A2.test()
print(model_A1.accuracy_score)
# Clean up memory/GPU etc...
#
#
# # ======================================================================================================================
# # Task B1
# training
train(num_iteration=300, x_train_eyecolour, y_train_eyecolour, x_test_eyecolour, y_test_eyecolour)
# model_B1 = B1(args...)
# acc_B1_train = model_B1.train(args...)
# acc_B1_test = model_B1.test(args...)
# Clean up memory/GPU etc...
#
#
# # ======================================================================================================================
# # Task B2
# model_B2 = B2(args...)
# acc_B2_train = model_B2.train(args...)
# acc_B2_test = model_B2.test(args...)
# Clean up memory/GPU etc...
#
#
# # ======================================================================================================================
# ## Print out your results with following format:
# print('TA1:{},{};TA2:{},{};TB1:{},{};TB2:{},{};'.format(acc_A1_train, acc_A1_test,
#                                                         acc_A2_train, acc_A2_test,
#                                                         acc_B1_train, acc_B1_test,
#                                                         acc_B2_train, acc_B2_test))
#
# # If you are not able to finish a task, fill the corresponding variable with 'TBD'. For example:
# # acc_A1_train = 'TBD'
# # acc_A1_test = 'TBD'
