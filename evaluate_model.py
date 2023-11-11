import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV

class EvaluateModel:
    def __init__(self, model_name, predictions, target_test, penguin):
        self.model_name = model_name
        self.predictions = predictions
        self.target_test = target_test
        # If evaluating model with penguin dataset, penguin = True
        self.penguin = penguin
        self.runs = []

    def evaluate(self):
        """ Function for evaluating the performance of a given model """
        # Open the file in append mode
        if self.penguin:
            file_name = "penguin-performance.txt"
        else:
            file_name = "abalone-performance.txt"
        with open(file_name, "a") as file:
            file.write("\n\n")
            file.write("*" * 50 + "\n")
            file.write(self.model_name + "\n")
            file.write("*" * 50 + "\n")

            # (B) Confusion Matrix
            file.write("\n(B) Confusion Matrix:\n")
            file.write(f"{confusion_matrix(self.target_test, self.predictions)}\n")

            # (C) Precision, Recall, and F1-measure for each class
            file.write("\n(C) Precision, Recall, and F1-measure for each class:\n")
            file.write(f"{classification_report(self.target_test, self.predictions)}\n")

            # (D) Accuracy, Macro-average F1, and Weighted-average F1
            file.write("\n(D) Accuracy, Macro-average F1, and Weighted-average F1:\n")
            accuracy = accuracy_score(self.target_test, self.predictions)
            macro_f1 = f1_score(self.target_test, self.predictions, average='macro')
            weighted_f1 = f1_score(self.target_test, self.predictions, average='weighted')
            self.runs.append({'accuracy': accuracy, 'macro_f1': macro_f1, 'weighted_f1': weighted_f1})
            file.write(f"Accuracy: {accuracy}\n")
            file.write(f"Macro-average F1: {macro_f1}\n")
            file.write(f"Weighted-average F1: {weighted_f1}\n")

    def calculate_average_and_variance(self):
        """ Function to calculate the average and variance of performance metrics across multiple runs """
        accuracy_values = [run['accuracy'] for run in self.runs]
        macro_f1_values = [run['macro_f1'] for run in self.runs]
        weighted_f1_values = [run['weighted_f1'] for run in self.runs]

        average_accuracy = np.mean(accuracy_values)
        variance_accuracy = np.var(accuracy_values)

        average_macro_f1 = np.mean(macro_f1_values)
        variance_macro_f1 = np.var(macro_f1_values)

        average_weighted_f1 = np.mean(weighted_f1_values)
        variance_weighted_f1 = np.var(weighted_f1_values)

        if self.penguin:
            file_name = "penguin-performance.txt"
        else:
            file_name = "abalone-performance.txt"

         with open(file_name, "a") as file:
            file.write("\n\n")
            file.write("*" * 50 + "\n")
            file.write(f"Average and Variance for {self.model_name}\n")
            file.write("*" * 50 + "\n")
            file.write(f"(A) Average Accuracy: {average_accuracy}, Variance: {variance_accuracy}\n")
            file.write(f"(B) Average Macro-average F1: {average_macro_f1}, Variance: {variance_macro_f1}\n")
            file.write(f"(C) Average Weighted-average F1: {average_weighted_f1}, Variance: {variance_weighted_f1}\n")


