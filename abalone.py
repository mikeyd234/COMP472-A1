import pandas as pd
from plotter import Plotter
from splitter import Splitter
from baseMLP import BaseMLP
from topMLP import TopMLP

# Access the dataset
filepath = 'C:/Users/pavit/OneDrive/Documents/Fall_2023/COMP_472/COMP472-A1/COMP472-A1-datasets/abalone.csv'
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

# Train and test the Base-MLP classifier
abalone_base_mlp = BaseMLP()

abalone_base_mlp.base_mlp(features_train, features_test, target_train, target_test)

# Train and test the Top-MLP classifier
abalone_top_mlp = TopMLP()

abalone_top_mlp.top_mlp(features_train, features_test, target_train, target_test)
