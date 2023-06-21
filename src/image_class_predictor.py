import os
import cv2
import numpy as np
from tkinter import Tk, filedialog
from keras.models import load_model

# Define the dataset path and the list of classes
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_path = os.path.join(project_path, "Plant_Data")
classes = os.listdir(os.path.join(dataset_path, "train"))

# Define the input shape of the images
input_shape = (128, 128, 3)

# Load the trained model
model = load_model(project_path+"\\plant_classifier.h5")

# Create a file dialog to choose an image
root = Tk()
root.withdraw()
file_path = filedialog.askopenfilename()

# Load the image and preprocess it
img = cv2.imread(file_path)
img = cv2.resize(img, (128, 128))
img = img.astype("float32") / 255.0
img = np.expand_dims(img, axis=0)

# Use the model to predict the class of the image
preds = model.predict(img)[0]
"""
# Get the indices of the top 5 classes with the highest probabilities
top_classes = preds.argsort()[::-1][:5]

# Print the top 5 class labels and their corresponding probabilities
for i, idx in enumerate(top_classes):
    class_name = classes[idx]
    print(class_name, ": ", round(preds[idx]*100, 2), "%")
"""
for i in range(99):
    print(classes[i],":",round(preds[i]*100, 2), "%")