from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from itertools import cycle
from paths import *
from scipy import interp
from sklearn import datasets
from sklearn import linear_model
from sklearn import metrics
from sklearn import svm, datasets
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import numpy as np
import os
import urllib
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from collections import OrderedDict
class ScikitPredictor:
    
    classifier = None

    # All Data we have
    X = None
    y = None
    
    # Data used for Training the model
    trainX = None
    trainY = None

    # Never seen before data! Used only for final testing, after (!) model tuning to report final performance measures
    testX = None
    testY = None

    # Method to hand over all data to the predictor. Train/Validation/Test-Splits are created afterwards by splitData.
    def setData(self, csv_path):
        f = open(csv_path)
        f.readline()  # skip the header
        data = np.loadtxt(f, delimiter=',')
        self.X = data[:, 0:-1]
        self.y = data[:, -1]
        print("Predictor: Got "+str(len(self.X))+" samples with "+str(len(self.X[0]))+" features.")
        return self

    # Example: splitData(0.2, 0.2) takes away 20% of the Data for testing
    #          and uses 20% of the remaining data for model selection
    def splitData(self, testSplitRatio):
        self.trainX, self.testX, self.trainY, self.testY = \
            train_test_split(self.X, self.y, test_size=testSplitRatio, stratify=self.y)
        
        print("Predictor: Using "+str(len(self.trainX))+" samples for training and "+str(len(self.testX))+" for testing.")
        
        return self

    # Sets the classifier for this predictor
    def setClassifier(self, classifier):
        self.classifier = classifier
        return self

    # Performances cross validation with numFolds-folds on validation-data. Use this method as metric for parameter tuning.
    def crossValidate(self, numFolds):
        cv = StratifiedKFold(n_splits=numFolds, shuffle=True)
        scores = cross_val_score(self.classifier, self.trainX, self.trainY, cv=cv)

        # mean score and the 95% confidence interval 
        print("Avg Accuracy by "+str(numFolds)+"-Fold-CF: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    # Trains the previously given classifier over given amount of iterations on the train-dataset. Reports Acc on Train-Data afterwards
    def train(self, iterations):
        self.classifier.set_params(max_iter=iterations)
        self.classifier.fit(self.trainX, self.trainY) 
        acc = self.classifier.score(self.trainX, self.trainY)
        print("Acc on learned data: "+str(acc))
        return self

    # Classifies an array of samples with the previously trained classifier
    def classify(self, samples):
        return(self.classifier.predict(samples))

    # Evaluate previously trained model on unseen test-data. Prints various metrics.
    def evaluateOnTestData(self):
        print("Evaluation on unseen test-data:")
        y_pred = self.classify(self.testX)
        y_true = self.testY
        print(metrics.classification_report(y_true, y_pred, target_names = ['non-binder', 'binder']))
        print("Accurancy: " + str(self.accuracy(y_true, y_pred)))
        print("AUC: " + str(self.auc()))
        return self

    # Returns the acc for samples with classes y_true that are classified as y_pred
    # Parameters also may be an array of classifications. The Average acc is then returned.
    def accuracy(self, y_true, y_pred):
        if(isinstance(y_true, list)):
            acc = 0
            for yt, yp in zip(y_true, y_pred):
                acc += self.accuracy(yt, yp)
            return acc/len(y_true)
        else:
            return accuracy_score(y_true, y_pred)

    # Gives the AUC for unseen test-data
    def auc(self):
        probas = self.classifier.predict_proba(self.testX)

        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(self.testY, probas[:, 1])
        return auc(fpr, tpr)

    # Plots the ROC-Curve over the test data
    
    def plot_roc(self, features, filename):
        probas = self.classifier.predict_proba(self.testX)

        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(self.testY, probas[:, 1])
        roc_auc = auc(fpr, tpr)
    

        plt.plot(fpr, tpr, lw=2, color='blue',
                 label='ROC (area = %0.2f)' % (roc_auc))

        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k',
                 label='Luck')
        all_params = self.classifier.get_params()
        all_params = OrderedDict(sorted(all_params.items(), key=lambda t: t[0]))
        selected_params = ['learning_rate_init', 'momentum', 'solver', 'hidden_layer_sizes', 'activation']
        params = {k + ": " + str(all_params[k]) for k in selected_params} 
        plt.text(0, -0.51, str("\n".join(params)))
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC-Curve (Featureset '+ features +')')
        plt.legend(loc="lower right")
        plt.gca().set_position((.1, .3, .8, .6))
        plt.savefig('rocs/'+filename + '.png')
        plt.close()

    def optimize(self):
        parameters = {
           # 'solver':             ['lbfgs', 'sgd', 'adam'],
            #'momentum':           [1, 0.5,  0.1],
           # 'learning_rate_init': [1, 0.5, 0.01],
            'hidden_layer_sizes': [(10), (25), (100), (30, 10), (5, 20, 5), (10, 20, 10), (20, 100, 50)],
            #'activation' :        ['identity', 'logistic', 'tanh', 'relu']
        }
        cv = StratifiedKFold(n_splits=4, shuffle=True)
        clf = GridSearchCV(self.classifier, parameters, n_jobs = 4, cv = cv, verbose = True)
        clf.fit(self.trainX, self.trainY)
        print("Best model has a score (acc) of "+str(clf.best_score_)+" with hyperparameters:\n\t"+str(clf.best_params_))
        return clf.best_estimator_

    def saveTrainedClassifier(self, feature_extractor, path):
        joblib.dump({'features': feature_extractor, 'model': self.classifier}, path)
        return self

    def loadTrainedClassifier(self, path):
        loaded = joblib.load(path)
        self.classifier = loaded['model']
        return loaded['features']