import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report,accuracy_score

class A1_classifier:
    def __init__(self, x_train_gender, x_test_gender, y_train_gender, y_test_gender):
        self.x_train_gender = x_train_gender
        self.x_test_gender = x_test_gender
        self.y_train_gender = y_train_gender
        self.y_test_gender = y_test_gender
        self.svm_classifier = SVC(kernel='linear')
        self.accuracy_score = 0
    def svm_train(self):
        train_images = self.x_train_gender.reshape((self.x_train_gender.shape[0], 68*2))
        train_labels = list(zip(*self.y_train_gender))[0]
        self.svm_classifier.fit(train_images, train_labels)
        ############ CV needed here###########
    def test(self):
        test_images = self.x_test_gender.reshape((self.x_test_gender.shape[0], 68*2))
        test_labels = list(zip(* self.y_test_gender))[0]
        pred = self.svm_classifier.predict(test_images)
        self.accuracy_score = accuracy_score(test_labels, pred)








# # sklearn functions implementation
# def img_SVM(training_images, training_labels, test_images, test_labels):
#     classifier = SVC(kernel='linear')
#     classifier.fit(training_images, training_labels)
#     pred = classifier.predict(test_images)
#     print("Accuracy:", accuracy_score(test_labels, pred))
#
#    # print(pred)
#     return pred
#
# pred=img_SVM(x_train_gender.reshape((x_train_gender.shape[0], 68*2)), list(zip(*y_train_gender))[0], x_test_gender.reshape((x_test_gender.shape[0], 68*2)), list(zip(* y_test_gender))[0])
# pred_smile=img_SVM(x_train_smile.reshape((x_train_smile.shape[0], 68*2)), list(zip(*y_train_smile))[0], x_test_smile.reshape((x_test_smile.shape[0], 68*2)), list(zip(* y_test_smile))[0])
