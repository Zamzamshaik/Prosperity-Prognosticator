import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SPLIT_PATH = os.path.join(BASE_DIR, "..", "data", "split")
MODEL_PATH = os.path.join(BASE_DIR, "..", "models")

os.makedirs(MODEL_PATH, exist_ok=True)

def train_model():
    # Check if split files exist
    x_train_path = os.path.join(SPLIT_PATH, "X_train.csv")
    x_test_path = os.path.join(SPLIT_PATH, "X_test.csv")
    y_train_path = os.path.join(SPLIT_PATH, "y_train.csv")
    y_test_path = os.path.join(SPLIT_PATH, "y_test.csv")

    if not os.path.exists(x_train_path):
        raise FileNotFoundError("❌ X_train.csv not found! Run feature_engineering.py first.")

    # Load data
    X_train = pd.read_csv(x_train_path)
    X_test = pd.read_csv(x_test_path)
    y_train = pd.read_csv(y_train_path).values.ravel()
    y_test = pd.read_csv(y_test_path).values.ravel()

    print("Training Data Shape:", X_train.shape)

    # Model
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight='balanced'
    )

    # Train
    model.fit(X_train, y_train)
    print("✅ Model Training Completed.")

    # Predict
    y_pred = model.predict(X_test)

    # Evaluation
    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Save model
    joblib.dump(model, os.path.join(MODEL_PATH, "startup_success_model.pkl"))

    # Save feature columns (IMPORTANT for API)
    joblib.dump(X_train.columns.tolist(), os.path.join(MODEL_PATH, "feature_columns.pkl"))

    print("✅ Model and feature columns saved in models/")

    # Cross Validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print("CV Score:", cv_scores.mean())

if __name__ == "__main__":
    train_model()