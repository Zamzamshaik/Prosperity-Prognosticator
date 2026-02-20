import joblib
import numpy as np
import os

__all__ = ["model", "feature_columns"]

# Get project root (one level above backend)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
FEATURE_PATH = os.path.join(BASE_DIR, "models", "feature_columns.pkl")

print("Loading Model From:", MODEL_PATH)
print("Loading Features From:", FEATURE_PATH)

# Load model and feature columns
model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURE_PATH)

def prepare_input(user_input: dict):
    """
    Convert frontend input into full 26 feature vector
    (VERY IMPORTANT to avoid feature mismatch error)
    """
    # Create empty feature array with same columns as training
    input_df = {col: 0 for col in feature_columns}

    # Map frontend fields to model features (edit if needed)
    if "founded_year" in user_input:
        input_df["founded_year"] = float(user_input["founded_year"])

    if "funding_total_usd" in user_input:
        input_df["funding_total_usd"] = float(user_input["funding_total_usd"])

    if "has_vc" in user_input:
        input_df["has_VC"] = 1 if user_input["has_vc"] == "Yes" else 0

    if "has_angel" in user_input:
        input_df["has_angel"] = 1 if user_input["has_angel"] == "Yes" else 0

    # Convert to numpy array in correct column order
    final_input = np.array([list(input_df.values())])
    return final_input


def predict_startup(user_input):
    """
    Predict startup success probability
    """
    processed_input = prepare_input(user_input)

    prediction = model.predict(processed_input)[0]
    probability = model.predict_proba(processed_input)[0][1]

    return int(prediction), float(probability)