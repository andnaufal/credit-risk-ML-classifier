import os
import pandas as pd
import pickle

# Define the directory where the models are saved
model_dir = "/Users/naufal/Documents/data_science_project/credit-risk-classifier/src/models"

# Load trained models
models = {}
for filename in os.listdir(model_dir):
    if filename.endswith(".pkl"):  # Check if the file is a pickle file
        # Extract the model name from the filename
        model_name = filename.split("_")[0]
        with open(os.path.join(model_dir, filename), 'rb') as file:
            models[model_name] = pickle.load(file)

# Load new data for prediction
new_data = pd.read_csv("new_data.csv")

# Make predictions using each model
for model_name, model in models.items():
    predictions = model.predict(new_data)
    # Do something with predictions (e.g., save them to a file)
