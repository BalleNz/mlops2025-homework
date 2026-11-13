"""Model quality gate: fails the pipeline when accuracy < accuracy_min."""

import json
import os
import pickle
import sys

import pandas as pd
import yaml
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

METRICS_PATH = "metrics/metrics.json"


def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)


def compute_accuracy(params):
    """Recomputes accuracy on the same test split as evaluate.py."""
    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)

    df = pd.read_csv("data/processed/dataset.csv")
    X = df[["total_bill", "size"]]
    y = df["high_tip"]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=params["test_size"], random_state=params["seed"]
    )
    return accuracy_score(y_test, model.predict(X_test))


def read_saved_accuracy():
    """Reads accuracy written by evaluate.py, if the file exists."""
    if not os.path.exists(METRICS_PATH):
        return None
    with open(METRICS_PATH, "r") as f:
        return json.load(f).get("accuracy")


def validate_model():
    params = load_params()
    accuracy_min = float(params["accuracy_min"])

    accuracy = compute_accuracy(params)

    saved = read_saved_accuracy()
    if saved is not None and abs(saved - accuracy) > 1e-9:
        print(
            f"Warning: accuracy in {METRICS_PATH} ({saved:.4f}) "
            f"differs from recomputed value ({accuracy:.4f})"
        )

    print(f"Accuracy: {accuracy:.4f}, accuracy_min: {accuracy_min:.4f}")

    if accuracy < accuracy_min:
        print("Model validation FAILED: accuracy is below accuracy_min")
        sys.exit(1)

    print("Model validation passed")


if __name__ == "__main__":
    validate_model()
