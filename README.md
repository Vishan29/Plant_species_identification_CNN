# Plant_species_identification_CNN
Project to identify a plant species from an image. To implement this, we will be using CNN, for which, we will use a dataset from kaggle, which contains around 19000 images split into training, testing and validation.
The images in these directories are further split into 99 classes repesenting 99 different species:

## Dataset Link: https://www.kaggle.com/datasets/muhammadjawad1998/plants-dataset99-classes

## To build model
Download the dataset from the kaggle link given above and extract it to the root directory of the project. Now run the data_preprocess_cnn.ipynb notebook, to read the images from dataset and build and test the model accordingly.

## To predict using CNN model.
Run the plant_predictor.py code, and select an image from the folder. The program will make use of the built cnn model and give the predicted species
