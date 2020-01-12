import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report,accuracy_score

class A1_classifier:
    def __init__(self, x_train_gender, x_test_gender, y_train_gender, y_test_gender):
        # training and test data
        self.x_train_gender = x_train_gender
        self.x_test_gender = x_test_gender
        self.y_train_gender = y_train_gender
        self.y_test_gender = y_test_gender

        # init svm classifier
        # the parameter here is the best params found by SVM_CV_multiclass() function bwlow
        self.tuned_svm_classifier = SVC(kernel='linear', C = 0.01)
        self.accuracy_score = 0

    def svm_train(self):
        # train svm model
        train_images = self.x_train_gender.reshape((self.x_train_gender.shape[0], 68*2))
        train_labels = list(zip(*self.y_train_gender))[0]
        self.tuned_svm_classifier.fit(train_images, train_labels)

        # output the training score at this stage
        train_score_at_this_stage = self.tuned_svm_classifier.score(train_images[-10:,:], train_labels[-10:])
        return train_score_at_this_stage

    def test(self):
        # test the trained model
        # prepare the data for the classifier prediction
        test_images = self.x_test_gender.reshape((self.x_test_gender.shape[0], 68*2))
        test_labels = list(zip(* self.y_test_gender))[0]
        pred = self.tuned_svm_classifier.predict(test_images)
        self.accuracy_score = accuracy_score(test_labels, pred)
        return self.accuracy_score

    def SVM_CV_multiclass(self, training_images, training_labels, test_images, test_labels):
        # parameter settings to be searched over to find the best param
        tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                             'C': [1, 10, 100, 1000]},
                            {'kernel': ['linear'], 'C': [0.001, 0.01, 0.1, 1]}]

        # two kinds of metrics
        scores = ['precision', 'recall']

        for score in scores:
            print("# Tuning hyper-parameters for %s" % score)
            print()

            # apply grid search cross validation on svm model
            clf = GridSearchCV(
                SVC(), tuned_parameters, scoring='%s_macro' % score
            )
            clf.fit(training_images, training_labels)

            # present the results
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
