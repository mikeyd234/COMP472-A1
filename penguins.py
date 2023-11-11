import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from plotter import Plotter
from splitter import Splitter
from baseMLP import BaseMLP
from topMLP import TopMLP

# Load the Penguin dataset
file_path = "C:/Users/Michael/source/repos/COMP472-A1/COMP472-A1-datasets/penguins.csv"
penguins_data = pd.read_csv(file_path)

# Display all columns in the dataset
pd.set_option('display.max_columns', None)
print(penguins_data.head())

# Prompt user to select method for data conversion
print("Select the method to convert features:")
print("1. Convert to 1-hot vectors (dummy-coded data)")
print("2. Convert to categories")
method_choice = input("Enter 1 or 2: ")

if method_choice == '1':
    # Method 1: Convert "island" and "sex" features into 1-hot vectors (dummy-coded data)
    # Create a copy of the original dataset to preserve the original data
    penguins_data_onehot = penguins_data.copy()

    # Create a OneHotEncoder for "island" and fit/transform it
    encoder_island = OneHotEncoder(sparse=False)
    island_encoded = encoder_island.fit_transform(penguins_data[['island']])

    # Create a OneHotEncoder for "sex" and fit/transform it
    encoder_sex = OneHotEncoder(sparse=False)
    sex_encoded = encoder_sex.fit_transform(penguins_data[['sex']])

    # Get the feature names for the one-hot encoded columns
    island_feature_names = encoder_island.get_feature_names_out(['island'])
    sex_feature_names = encoder_sex.get_feature_names_out(['sex'])

    # Create DataFrames for the encoded features
    island_df = pd.DataFrame(island_encoded, columns=island_feature_names)
    sex_df = pd.DataFrame(sex_encoded, columns=sex_feature_names)

    # Concatenate the new 1-hot encoded columns with the original DataFrame
    penguins_data_onehot = pd.concat([penguins_data_onehot, island_df, sex_df], axis=1)

    # Drop the original "island" and "sex" columns
    penguins_data_onehot = penguins_data_onehot.drop(['island', 'sex'], axis=1)

    # Display the modified dataset with 1-hot encoded features
    print("Method 1: 1-hot encoded dataset")
    print(penguins_data_onehot.head())

    penguins_data = penguins_data_onehot

elif method_choice == '2':

    # Method 2: Convert "island" and "sex" features into categories
    penguins_data_categories = penguins_data.copy()

    # Use LabelEncoder to convert "island" and "sex" columns into categories
    label_encoder_island = LabelEncoder()
    label_encoder_sex = LabelEncoder()

    # Encode "island" and "sex" columns
    penguins_data_categories['island'] = label_encoder_island.fit_transform(penguins_data['island'])
    penguins_data_categories['sex'] = label_encoder_sex.fit_transform(penguins_data['sex'])

    # Display the modified dataset with encoded features
    print("Method 2: Encoded dataset")
    print(penguins_data_categories.head())

    penguins_data = penguins_data_categories

else:
    print("Invalid choice. Please enter 1 or 2.")

penguin_plot = Plotter(penguins_data, penguin=True)

penguin_plot.plot()

penguin_split = Splitter(penguins_data, penguin = True)

features_train, features_test, target_train, target_test = penguin_split.split()

print("Training target: ", target_train)
print("Test target: ", target_test)

# Train and test the Base-MLP classifier
penguin_base_mlp = BaseMLP()

penguin_base_mlp.base_mlp(features_train, features_test, target_train, target_test)

# Train and test the Top-MLP classifier
penguin_top_mlp = TopMLP()

penguin_top_mlp.top_mlp(features_train, features_test, target_train, target_test)


