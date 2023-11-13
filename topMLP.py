import pandas as pd
from sklearn.neural_network import MLPClassifier
from splitter import Splitter
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from itertools import repeat

class TopMLP:
  def top_mlp(features_train, features_test, target_train, target_test):

    # Define hyper-parameter values
    param_grid = {
      'activation': ['tanh', 'logistic', 'relu'],
      'hidden_layer_sizes': [(30, 50), (10, 10, 10), (30, 50), (10, 10, 10), (10, 10, 10),],
      'solver': ['adam', 'sgd'] 
    }

    # Create the Top Multi-Layered Perceptron classifier
    clf = MLPClassifier(max_iter=2000)

    # Create an instance for GridSearchCV
    grid_search = GridSearchCV(clf, param_grid)
    # Train the grid search 
    grid_search.fit(features_train, target_train)

    # Get the best set of parameters and the model
    best_parameters = grid_search.best_estimator_

    # Predict the labels
    tmlp_predict_test = best_parameters.predict(features_test)

    # Calculate the accuracy of the Base-MLP classifier
    tmlp_accuracy = accuracy_score(target_test, tmlp_predict_test)

    # Build the confusion matrix
    tmlp_confusion_matrix = confusion_matrix(target_test, tmlp_predict_test)

    # Calculate the precision
    tmlp_precision = precision_score(target_test, tmlp_predict_test, average=None, zero_division=0)

    # Calculate the recall
    tmlp_recall = recall_score(target_test, tmlp_predict_test, average=None)

    # Calculate the F1-measure
    tmlp_f1 = f1_score(target_test, tmlp_predict_test, average=None)

    tmlp_f1_macro = f1_score(target_test, tmlp_predict_test, average='macro')

    tmlp_f1_weighted = f1_score(target_test, tmlp_predict_test, average='weighted')

    return tmlp_accuracy, tmlp_precision, tmlp_recall, tmlp_f1, tmlp_confusion_matrix, tmlp_f1_macro, tmlp_f1_weighted