import pandas as pd #file reader 
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder #convert type into numerical values 
from sklearn.tree import plot_tree, DecisionTreeClassifier #decision tree 
from sklearn.model_selection import train_test_split #used in the splitting of validation and test data 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

class TopDT:
    def top_dt(features_train, features_test, target_train, target_test):
        # Define the parameter grid to search
        param_grid = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 10, 20]
        }

        # Initialize the classifier
        dt_classifier = DecisionTreeClassifier(random_state=42)

        # Initialize the GridSearchCV object
        grid_search = GridSearchCV(dt_classifier, param_grid, cv=5, scoring='accuracy')

        # Fit the grid search to the data
        grid_search.fit(features_train, target_train)

        # Get the best estimator and print the details
        best_classifier = grid_search.best_estimator_
    
        # Use the best classifier to predict on the test set
        target_pred = best_classifier.predict(features_test)

        print("Best parameters found:", grid_search.best_params_)
        print("Best score found:", grid_search.best_score_)

        # Calculate the evaluation metrics
        accuracy = accuracy_score(target_test, target_pred)
        precision = precision_score(target_test, target_pred, average=None)
        recall = recall_score(target_test, target_pred, average=None)
        f1 = f1_score(target_test, target_pred, average=None)
        confusion_mat = confusion_matrix(target_test, target_pred)

        # Calculate macro F1 and weighted F1 separately
        f1_macro = f1_score(target_test, target_pred, average='macro')
        f1_weighted = f1_score(target_test, target_pred, average='weighted')

        # Plot the best decision tree
        plt.figure(figsize=(20, 10))
        plot_tree(best_classifier, filled=True, max_depth=3)  # Set max_depth to your preferred value for visualization
        plt.title('Best Decision Tree from GridSearchCV')
        plt.show()

        return accuracy, precision, recall, f1, confusion_mat, f1_macro, f1_weighted