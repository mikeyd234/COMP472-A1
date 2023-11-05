import matplotlib.pyplot as plt
import pandas as pd

class plotter:
    def __init__(data, penguin):
        self.data = data
        # If plotting penguin datset, penguin = true
        self.penguin = penguin
    
    def plot(self):
        # Count the instances of the output classes
        if penguin:
            class_cnts = self.data['species'].value_counts()
        else:
            class_cnts = self.data['Type'].value_counts()

        # Find the percentage
        class_perc = class_cnts / class_counts.sum() * 100

        # Plot the class percentages
        plt.figure(figsize=(8, 6))
        class_percentages.plot(kind='bar', color='skyblue')
        plt.title("Percentage of Instances in Each Class")
        plt.xlabel("Class")
        plt.ylabel("Percentage")
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        # Save the plot as a GIF file (replace 'penguin-classes.gif' with the desired filename)
        plt.savefig('penguin-classes.gif', format='gif')
        plt.show()


    




