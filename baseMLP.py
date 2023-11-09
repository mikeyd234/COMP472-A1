import pandas as pd
from sklearn.neural_network import MLPClassifier
from splitter import Splitter
from sklearn.metrics import accuracy_score

class BaseMLP:
  def base_mlp(features_train, features_test, target_train, target_test):
    # Create the Base Multi-Layered Perceptron classifier with the given parameters
    clf = MLPClassifier(hidden_layer_sizes=(100, 100), activation='logistic', solver='sgd')
    # Train the Base-MLP
    clf.fit(features_train, target_train)

    # Predict the labels
    bmlp_predict_test = clf.predict(features_test)

    # Calculate the accuracy of the Base-MLP classifier
    bmlp_accuracy = accuracy_score(target_test, bmlp_predict_test)

