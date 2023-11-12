import pandas as pd #file reader 
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder #convert type into numerical values 
from sklearn.tree import plot_tree, DecisionTreeClassifier #decision tree 
from sklearn.model_selection import train_test_split #used in the splitting of validation and test data 
from sklearn.model_selection import GridSearchCV

class TopDT:
    def top_dt(self, features_train, target_train):
        # Define the parameter grid to search
        param_grid = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 10, 20],  # Replace with values of your choice
            'min_samples_split': [2, 10, 20]  # Replace with values of your choice
        }

        # Initialize the classifier
        dt_classifier = DecisionTreeClassifier(random_state=42)

        # Initialize the GridSearchCV object
        grid_search = GridSearchCV(dt_classifier, param_grid, cv=5, scoring='accuracy')

        # Fit the grid search to the data
        grid_search.fit(features_train, target_train)

        # Get the best estimator and print the details
        best_classifier = grid_search.best_estimator_
        print("Best parameters found:", grid_search.best_params_)
        print("Best score found:", grid_search.best_score_)

        # Plot the best decision tree
        plt.figure(figsize=(20, 10))
        plot_tree(best_classifier, filled=True, max_depth=3)  # Set max_depth to your preferred value for visualization
        plt.title('Best Decision Tree from GridSearchCV')
        plt.show()
        
        return best_classifier