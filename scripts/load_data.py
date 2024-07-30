from datasets import load_dataset
import pandas as pd
import os

# Load the dataset from Hugging Face
dataset = load_dataset('cnn_dailymail', '3.0.0', split='validation')

# Convert the dataset to a Pandas DataFrame
df = dataset.to_pandas()

# Ensure the directory exists
os.makedirs('data', exist_ok=True)

# Save the DataFrame to a CSV file
df.to_csv('data/cnn_dailymail_3.0.0.csv', index=False)

# Confirm the file is saved
print("CSV file saved successfully!")