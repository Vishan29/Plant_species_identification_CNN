import cv2
import os
import numpy as np
from tkinter import Tk, filedialog
from keras.models import load_model

# Define the dataset path and the list of classes
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_path = os.path.join(project_path, "Plant_Data")
classes = os.listdir(os.path.join(dataset_path, "train"))

model = load_model("plant_classifier.h5")

# Function to preprocess the input image
def preprocess_image(image):
    image = cv2.resize(image, (128, 128))
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Open file dialog to select an image
Tk().withdraw()
image_path = filedialog.askopenfilename(title="Select an image", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])

# Load the image
image = cv2.imread(image_path)

# Preprocess the image
preprocessed_image = preprocess_image(image)

# Make predictions
predictions = model.predict(preprocessed_image)
predicted_class_index = np.argmax(predictions[0])
predicted_class = classes[predicted_class_index]
probability = predictions[0][predicted_class_index]

# Print the predicted class and probability
print("Predicted class:", predicted_class)
print("Probability:", probability)
