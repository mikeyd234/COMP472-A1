import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

class Plotter:
    def __init__(self, data, penguin):
        self.data = data
        # If plotting penguin datset, penguin = true
        self.penguin = penguin
    
    def plot(self):
        # Count the instances of the output classes
        if self.penguin:
            class_cnts = self.data['species'].value_counts()
        else:
            class_cnts = self.data['Type'].value_counts()

        # Find the percentage
        class_perc = class_cnts / class_cnts.sum() * 100

        # Plot the class percentages
        plt.figure(figsize=(8, 6))
        class_perc.plot(kind='bar', color='skyblue')
        plt.title("Percentage of Instances in Each Class")
        plt.xlabel("Class")
        plt.ylabel("Percentage")
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        # Save the plot as a GIF file (replace 'penguin-classes.gif' with the desired filename)
        plt.savefig('temp-classes.png', format='png')
        
        image = Image.open('temp-classes.png')
        if self.penguin:
            image.save('penguin-classes.gif', save_all=True, append_images=[image], duration=100, loop=0)
        else:
            image.save('abalone-classes.gif', save_all=True, append_images=[image], duration=100, loop=0)
    




