import pandas as pd
import joblib
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

SPLIT_PATH = "../data/split/"
MODEL_PATH = "../models/"
os.makedirs(MODEL_PATH, exist_ok=True)


def hyperparameter_tuning():

    X_train = pd.read_csv(SPLIT_PATH + "X_train.csv")
    y_train = pd.read_csv(SPLIT_PATH + "y_train.csv").values.ravel()

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }

    rf = RandomForestClassifier(random_state=42)

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        verbose=2
    )

    grid_search.fit(X_train, y_train)

    print("\nBest Parameters:", grid_search.best_params_)
    print("Best CV Score:", grid_search.best_score_)

    best_model = grid_search.best_estimator_

    joblib.dump(best_model, MODEL_PATH + "startup_success_model_tuned.pkl")

    print("Tuned Model Saved Successfully.")


if __name__ == "__main__":
    hyperparameter_tuning()