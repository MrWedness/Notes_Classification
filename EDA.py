import numpy as np
import zipfile
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

with zipfile.ZipFile('images.npy.zip', 'r') as zip_ref:
    zip_ref.extractall()

labels = np.load('labels.npy')

images = np.load('images.npy')

# First, create a train/test split
train_images, temp_images, train_labels, temp_labels = train_test_split(
    images, labels, test_size=0.4, random_state=42  # 60% train, 40% temp
)

# Then, split the temp set into validation and test sets
val_images, test_images, val_labels, test_labels = train_test_split(
    temp_images, temp_labels, test_size=0.5, random_state=42  # 20% val, 20% test
)

plt.imshow(images[0], cmap='gray')

label_graph = pd.Series(labels).map(label_mapped).value_counts()

label_graph.plot(kind='bar')

# Dataframe of my plots
label_graph = pd.Series(labels).map(label_mapped).value_counts()

label_df = label_graph.reset_index()

label_df.columns = ['Label', 'Frequency']

print(label_df)





