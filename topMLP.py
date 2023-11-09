import pandas as pd
from sklearn.neural_network import MLPClassifier
from splitter import Splitter
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

class TopMLP:
  def top_mlp(features_train, features_test, target_train, target_test):
    # Define hyper-parameter values
    param_grid = {
      'activation': ['sigmoid', 'tanh', 'relu'],
      'hidden_layer_sizes': [(30, 50), (10, 10, 10)],
      'solver': ['adam', 'sgd'] 
    }

    # Create the Top Multi-Layered Perceptron classifier
    clf = MLPClassifier()

    # Create an instance for GridSearchCV
    grid_search = GridSearchCV(clf, param_grid)
    # Train the grid search 
    grid_search.fit(features_train, target_train)

    # Get the best set of parameters and the model
    best_parameters = grid_search.best_estimator_

    # Predict the labels
    tmlp_predict_test = best_parameters.predict(features_test)

    # Calculate the accuracy of the Base-MLP classifier
    bmlp_accuracy = accuracy_score(target_test, tmlp_predict_test)

    return best_parameters