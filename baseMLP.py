import pandas as pd
from sklearn.neural_network import MLPClassifier
from splitter import Splitter
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

class BaseMLP:
  def base_mlp(features_train, features_test, target_train, target_test):
    # Create the Base Multi-Layered Perceptron classifier with the given parameters
    clf = MLPClassifier(hidden_layer_sizes=(100, 100), activation='logistic', solver='sgd', max_iter=2000)
    # Train the Base-MLP
    clf.fit(features_train, target_train)

    # Predict the labels
    bmlp_predict_test = clf.predict(features_test)

    # Calculate the accuracy of the Base-MLP classifier
    bmlp_accuracy = accuracy_score(target_test, bmlp_predict_test)

    # Build the confusion matrix
    bmlp_confusion_matrix = confusion_matrix(target_test, bmlp_predict_test)

    # Calculate the precision
    bmlp_precision = precision_score(target_test, bmlp_predict_test, average=None, zero_division=0)

    # Calculate the recall
    bmlp_recall = recall_score(target_test, bmlp_predict_test, average=None)

    # Calculate the F1-measure
    bmlp_f1 = f1_score(target_test, bmlp_predict_test, average=None)

    bmlp_f1_macro = f1_score(target_test, bmlp_predict_test, average='macro')

    bmlp_f1_weighted = f1_score(target_test, bmlp_predict_test, average='weighted')

    return bmlp_accuracy, bmlp_precision, bmlp_recall, bmlp_f1, bmlp_confusion_matrix, bmlp_f1_macro, bmlp_f1_weighted


