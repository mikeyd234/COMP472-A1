import pandas as pd #file reader 
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder #convert type into numerical values 
from sklearn.tree import plot_tree, DecisionTreeClassifier #decision tree 
from sklearn.model_selection import train_test_split #used in the splitting of validation and test data 
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score


class BaseDT:
    def base_dt(features_train, features_test, target_train, target_test):
        # Initialize the Decision Tree Classifier with default parameters
        classifier = DecisionTreeClassifier(random_state=42)

        # Train the Decision Tree Classifier
        classifier.fit(features_train, target_train)

        # Predict on the test set
        target_pred = classifier.predict(features_test)

        # Calculate the evaluation metrics
        accuracy = accuracy_score(target_test, target_pred)
        precision = precision_score(target_test, target_pred, average=None)
        recall = recall_score(target_test, target_pred, average=None)
        f1 = f1_score(target_test, target_pred, average=None)
        confusion_mat = confusion_matrix(target_test, target_pred)

        # Calculate macro F1 and weighted F1 separately
        f1_macro = f1_score(target_test, target_pred, average='macro')
        f1_weighted = f1_score(target_test, target_pred, average='weighted')

        # Plot the decision tree
        plt.figure(figsize=(20,10))
        plot_tree(classifier, filled=True)
        plt.title("Decision Tree for Dataset")
        plt.show()
        
        return accuracy, precision, recall, f1, confusion_mat, f1_macro, f1_weighted