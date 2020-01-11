import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report,accuracy_score

class A2_classifier:
    def __init__(self, x_train_smile, x_test_smile, y_train_smile, y_test_smile):
        self.x_train_smile = x_train_smile
        self.x_test_smile = x_test_smile
        self.y_train_smile = y_train_smile
        self.y_test_smile = y_test_smile
        self.svm_classifier = SVC(kernel='linear')
        self.accuracy_score = 0
    def svm_train(self):
        train_images = self.x_train_smile.reshape((self.x_train_smile.shape[0], 68*2))
        train_labels = list(zip(*self.y_train_smile))[0]
        self.svm_classifier.fit(train_images, train_labels)
        train_score_at_this_stage = self.svm_classifier.score(train_images[-10:,:], train_labels[-10:])
        return train_score_at_this_stage
        ############ CV needed here###########
    def test(self):
        test_images = self.x_test_smile.reshape((self.x_test_smile.shape[0], 68*2))
        test_labels = list(zip(* self.y_train_smile))[0]
        pred = self.svm_classifier.predict(test_images)
        self.accuracy_score = accuracy_score(test_labels, pred)
        return accuracy_score
