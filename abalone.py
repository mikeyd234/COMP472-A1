import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from plotter import Plotter
from splitter import Splitter

# Access the dataset
filepath = 'C:/Users/pavit/OneDrive/Documents/Fall_2023/COMP_472/COMP472-A1/COMP472-A1-datasets/abalone.csv'
abalone_dataset = pd.read_csv(filepath)

# Display the first few columns from the dataset 
# Note: dataset can be used as is, therefore no need to convert any features
print(abalone_dataset.head())

# Plot the percentage of the instances
abalone_plot = Plotter(abalone_dataset)

abalone_plot.plot()

# Split the dataset
abalone_split = Splitter(abalone_dataset) 
