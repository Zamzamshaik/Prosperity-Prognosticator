import kagglehub
import os
import shutil
import pandas as pd

# Dataset reference from Kaggle
DATASET_NAME = "manishkc06/startup-success-prediction"

# Define storage path
RAW_DATA_DIR = "../data/raw"

def download_dataset():
    print("Downloading dataset from Kaggle...")

    # Download dataset
    path = kagglehub.dataset_download(DATASET_NAME)

    print("Dataset downloaded to temporary path:", path)

    # Create raw data directory if not exists
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    # Find CSV file inside downloaded folder
    for file in os.listdir(path):
        if file.endswith(".csv"):
            source_file = os.path.join(path, file)
            destination_file = os.path.join(RAW_DATA_DIR, file)

            shutil.copy(source_file, destination_file)

            print("Dataset copied to:", destination_file)
            return destination_file

    print("No CSV file found in dataset folder.")
    return None


if __name__ == "__main__":
    saved_path = download_dataset()
    print("Final raw dataset location:", saved_path)