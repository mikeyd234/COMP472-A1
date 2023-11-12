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
        



# # abalone data set 
# abalone_filepath = '/COMP472-A1-datasets/abalone.csv'
# abalone_data = pd.read_csv(abalone_filepath)

# #One hot encoding for the category 'Type'
# onehot_encoding = OneHotEncoder(sparse=False)
# type_encoded = onehot_encoding.fit_transform(abalone_data[['Type']])

# #put encoded data back into the table 
# type_encoded_df = pd.DataFrame(type_encoded, columns=onehot_encoding.get_feature_names_out(['Type']))
# abalone_encoded = pd.concat([abalone_data.drop('Type', axis=1), type_encoded_df], axis=1)

# #original sex of the abalone from the dataset
# #dropping features so that the model could learn from the other variables
# original_sex = abalone_encoded.drop(['Type_F', 'Type_I', 'Type_M'], axis=1)

# #actual outcome the model predict 
# prediction_sex = abalone_encoded[['Type_F', 'Type_I', 'Type_M']]

# #training the data
# original_train, original_test, prediction_train, prediction_test = train_test_split(original_sex, prediction_sex, test_size=0.2, random_state=42)








# # data: Type,LongestShell,Diameter,Height,WholeWeight,ShuckedWeight,VisceraWeight,ShellWeight,Rings
# # these are the target data 
# type_abalone = abalone_data['Type']
# longest_shell = abalone_data['LongestShell']
# diameter_abalone = abalone_data['Diameter']
# height_abalone = abalone_data['Height']
# whole_weight = abalone_data['WholeWeight']
# shucked_weight = abalone_data['ShuckedWeight']
# viscera_weight = abalone_data['VisceraWeight']
# shell_weight = abalone_data['ShellWeight']
# rings_abalone = abalone_data['Rings']





