import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Load the Penguin dataset
file_path = "C:/Users/Michael/source/repos/COMP472-A1/COMP472-A1-datasets/penguins.csv"
penguins_data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(penguins_data.head())

# Method 1: Convert "island" and "sex" features into 1-hot vectors (dummy-coded data)
# Create a copy of the original dataset to preserve the original data
penguins_data_onehot = penguins_data.copy()

# Use OneHotEncoder to convert "island" and "sex" into 1-hot vectors
encoder = OneHotEncoder(sparse=False)

# Transform "island" and "sex" columns into 1-hot encoded columns
island_encoded = encoder.fit_transform(penguins_data[['island']])
sex_encoded = encoder.fit_transform(penguins_data[['sex']])

# Create new column names for the 1-hot encoded columns
island_categories = encoder.categories_[0]
sex_categories = penguins_data['sex'].unique()

# Create new DataFrame for the 1-hot encoded columns
island_df = pd.DataFrame(island_encoded, columns=[f'island_{cat}' for cat in island_categories])
sex_df = pd.DataFrame(sex_encoded, columns=[f'sex_{cat}' for cat in sex_categories])

# Reset the index of the island_df
island_df = island_df.reset_index(drop=True)

# Concatenate the new 1-hot encoded columns with the original DataFrame
penguins_data_onehot = pd.concat([penguins_data_onehot, island_df, sex_df], axis=1)

# Drop the original "island" and "sex" columns
penguins_data_onehot = penguins_data_onehot.drop(['island', 'sex'], axis=1)

# Display the modified dataset with 1-hot encoded features
print("Method 1: 1-hot encoded dataset")
print(penguins_data_onehot.head())
