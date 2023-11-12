import pandas as pd
from plotter import Plotter
from splitter import Splitter
from baseMLP import BaseMLP
from topMLP import TopMLP
from evaluate_model import EvaluateModel

# Access the dataset
filepath = 'C:/Users/Michael/source/repos/COMP472-A1/COMP472-A1-datasets/abalone.csv'
abalone_dataset = pd.read_csv(filepath)

# Display the first few columns from the dataset 
# Note: dataset can be used as is, therefore no need to convert any features
print(abalone_dataset.head())

# Plot the percentage of the instances
abalone_plot = Plotter(abalone_dataset, penguin=False)

abalone_plot.plot()

# Split the dataset
abalone_split = Splitter(abalone_dataset, penguin=False) 

features_train, features_test, target_train, target_test = abalone_split.split()

print("Training target: ", target_train)
print("Test target: ", target_test)

print("Select the machine learning model:")
print("1. Base MLP")
print("2. Top MLP")
method_choice = input("Enter 1 or 2: ")

if method_choice == '1':
    for i in range(5):
        print(f"Run {i + 1}")
        # Train and test the Base-MLP classifier
        abalone_base_mlp = BaseMLP()

        result = BaseMLP.base_mlp(features_train, features_test, target_train, target_test)

        # Unpack the results
        accuracy, precision, recall, f1, confusion_matrix, f1_macro, f1_weighted = result

        evaluate_mlp = EvaluateModel(model_name = "BaseMLP", accuracy = accuracy, f1 = f1, macro_f1 = f1_macro, weighted_f1 = f1_weighted, confusion = confusion_matrix, precision = precision, recall = recall, penguin=False)

        evaluate_mlp.evaluate()

    evaluate_mlp.calculate_average_and_variance()

if method_choice == '2':
    for i in range(5):
        print(f"Run {i + 1}")
        # Train and test the Top-MLP classifier
        abalone_top_mlp = TopMLP()

        result = TopMLP.top_mlp(features_train, features_test, target_train, target_test)

        # Unpack the results
        accuracy, precision, recall, f1, confusion_matrix, f1_macro, f1_weighted = result

        evaluate_mlp = EvaluateModel(model_name = "TopMLP", accuracy = accuracy, f1 = f1, macro_f1 = f1_macro, weighted_f1 = f1_weighted, confusion = confusion_matrix, precision = precision, recall = recall, penguin=False)

        evaluate_mlp.evaluate()

    evaluate_mlp.calculate_average_and_variance()