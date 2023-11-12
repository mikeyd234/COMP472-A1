import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV

class EvaluateModel:
    def __init__(self, model_name, accuracy, f1, macro_f1, weighted_f1, confusion, precision, recall, penguin):
        self.model_name = model_name
        self.accuracy = accuracy
        self.f1 = f1
        self.macro_f1 = macro_f1
        self.weighted_f1 = weighted_f1
        self.confusion = confusion
        self.precision = precision
        self.recall = recall
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
            file.write(f"{self.confusion}")

            # (C) Precision, Recall, and F1-measure for each class
            file.write("\n(C) Precision, Recall, and F1-measure for each class:\n")
            file.write(f"Precision: {self.precision}\n")
            file.write(f"Recall: {self.recall}\n")
            file.write(f"F1-measure: {self.f1}\n")

            # (D) Accuracy, Macro-average F1, and Weighted-average F1
            file.write("\n(D) Accuracy, Macro-average F1, and Weighted-average F1:\n")
            file.write(f"Accuracy: {self.accuracy}\n")
            file.write(f"Macro-average F1: {self.macro_f1}\n")
            file.write(f"Weighted-average F1: {self.weighted_f1}\n")
            self.runs.append({'accuracy': self.accuracy, 'macro_f1': self.macro_f1, 'weighted_f1': self.weighted_f1})

    def calculate_average_and_variance(self):
        """ Function to calculate the average and variance of performance metrics across multiple runs """
        if not self.runs:
            print("No runs available for calculation.")
            return

        accuracy_values = [run.get('accuracy', 0) for run in self.runs]
        macro_f1_values = [run.get('macro_f1', 0) for run in self.runs]
        weighted_f1_values = [run.get('weighted_f1', 0) for run in self.runs]

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


