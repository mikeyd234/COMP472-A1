# Decision Tree with the default parameters. 
# Show the decision tree graphically 
# for the abalone dataset, you can restrict the tree depth for visualisation purposes

import pandas as pd #file reader 
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder #convert type into numerical values 
from sklearn.tree import plot_tree, DecisionTreeClassifier #decision tree 
from sklearn.model_selection import train_test_split #used in the splitting of validation and test data 


class BaseDT:
    def base_dt(features_train, features_test, target_train, target_test):
        # Initialize the Decision Tree Classifier with default parameters
        classifier = DecisionTreeClassifier(random_state=42)

        # Train the Decision Tree Classifier
        classifier.fit(features_train, target_train)

        # Predict on the test set
        target_pred = classifier.predict(features_test)

        # Evaluate the predictions
        accuracy = (target_pred == target_test).mean()
        print(f"Accuracy of the Base Decision Tree classifier: {accuracy:.2f}")

        # Plot the decision tree
        plt.figure(figsize=(20,10))
        plot_tree(classifier, filled=True)
        plt.title("Decision Tree for Abalone Dataset")
        plt.show()
        

