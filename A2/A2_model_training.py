import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report,accuracy_score


# ========================================================================
# similar structure as A1_model_training, can be integrated into one module
# =========================================================================

class A2_classifier:
    def __init__(self, x_train_smile, x_test_smile, y_train_smile, y_test_smile):
        self.x_train_smile = x_train_smile
        self.x_test_smile = x_test_smile
        self.y_train_smile = y_train_smile
        self.y_test_smile = y_test_smile
        # Best params for this task (params are different from A1)
        self.tuned_svm_classifier = SVC(kernel='rbf', C = 10)
        self.accuracy_score = 0
        self.train_valid_error = 0

    def svm_train(self):
        train_images = self.x_train_smile.reshape((self.x_train_smile.shape[0], 68*2))
        train_labels = list(zip(*self.y_train_smile))[0]
        self.tuned_svm_classifier.fit(train_images, train_labels)
        train_score_at_this_stage = self.tuned_svm_classifier.score(train_images[-100:,:], train_labels[-100:])
        return train_score_at_this_stage

    def test(self, test_features, test_smileLabels):
        # test the trained model
        # prepare the data for the classifier prediction
        test_images = test_features.reshape((test_features.shape[0], 68*2))
        test_labels = list(zip(* test_smileLabels))[0]
        pred = self.tuned_svm_classifier.predict(test_images)
        self.train_valid_error = accuracy_score(test_labels, pred)
        return self.train_valid_error

    def train_validation(self):
        test_images = self.x_test_smile.reshape((self.x_test_smile.shape[0], 68*2))
        test_labels = list(zip(* self.y_test_smile))[0]
        pred = self.tuned_svm_classifier.predict(test_images)
        self.accuracy_score = accuracy_score(test_labels, pred)
        return self.accuracy_score

    def SVM_CV_multiclass(self, training_images, training_labels, test_images, test_labels):
        tuned_parameters = [{'kernel': ['linear'], 'C': [0.001, 0.01, 0.1, 1]}]
    #     tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
    #                          'C': [1, 10, 100, 1000]},
    #                         {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
        scores = ['precision', 'recall']

        for score in scores:
            print("# Tuning hyper-parameters for %s" % score)
            print()

            clf = GridSearchCV(
                SVC(), tuned_parameters, scoring='%s_macro' % score
            )
            clf.fit(training_images, training_labels)

            print("Best parameters set found on development set:")
            print()
            print(clf.best_params_)
            print()
            print("Grid scores on development set:")
            print()
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r"
                      % (mean, std * 2, params))
            print()

            pred = clf.predict(test_images)
            print("Test Accuracy:", accuracy_score(test_labels, pred))
