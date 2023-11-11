# Decision Tree with the default parameters. 
# Show the decision tree graphically 
# for the abalone dataset, you can restrict the tree depth for visualisation purposes

from sklearn.tree import DecisionTreeClassifier #decision tree 
from sklearn.model_selection import train_test_split #used in the splitting of validation and test data 
import pandas as pd #file reader 

# abalone data set 
abalone_filepath = '/COMP472-A1-datasets/abalone.csv'
abalone_data = pd.read_csv(abalone_filepath)

# data: Type,LongestShell,Diameter,Height,WholeWeight,ShuckedWeight,VisceraWeight,ShellWeight,Rings
# these are the target data 
type_abalone = abalone_data['Type']
longest_shell = abalone_data['LongestShell']
diameter_abalone = abalone_data['Diameter']
height_abalone = abalone_data['Height']
whole_weight = abalone_data['WholeWeight']
shucked_weight = abalone_data['ShuckedWeight']
viscera_weight = abalone_data['VisceraWeight']
shell_weight = abalone_data['ShellWeight']
rings_abalone = abalone_data['Rings']




