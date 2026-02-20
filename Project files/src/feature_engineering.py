import pandas as pd
import os
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "startup_processed.csv")
SPLIT_PATH = os.path.join(BASE_DIR, "..", "data", "split")

os.makedirs(SPLIT_PATH, exist_ok=True)

def feature_engineering():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("Processed file not found! Run preprocessing.py first.")

    data = pd.read_csv(DATA_PATH)
    print("Loaded Data Shape:", data.shape)

    # Separate target
    y = data['status']
    X = data.drop(columns=['status'])

    # Drop string columns
    string_cols = X.select_dtypes(include=['object']).columns
    print("Dropping String Columns:", string_cols.tolist())
    X = X.drop(columns=string_cols)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training Shape:", X_train.shape)
    print("Testing Shape:", X_test.shape)

    # Save split data
    X_train.to_csv(os.path.join(SPLIT_PATH, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(SPLIT_PATH, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(SPLIT_PATH, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(SPLIT_PATH, "y_test.csv"), index=False)

    print("✅ Train-Test Split Saved Successfully at:", SPLIT_PATH)

if __name__ == "__main__":
    feature_engineering()