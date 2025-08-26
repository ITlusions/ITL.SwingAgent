"""Train or incrementally update an ML model using stored training data."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
from sklearn.linear_model import SGDRegressor
import joblib

DATA_PATH = Path("data/training_dataset.npz")
MODEL_PATH = Path("data/ml_model.pkl")


def load_training_data(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Training dataset not found: {path}")
    data = np.load(path, allow_pickle=True)
    return data["X"], data["y"]


def load_or_initialize(model_path: Path) -> SGDRegressor:
    if model_path.exists():
        return joblib.load(model_path)
    return SGDRegressor()


def train_incremental(X: np.ndarray, y: np.ndarray, model_path: Path = MODEL_PATH) -> SGDRegressor:
    model = load_or_initialize(model_path)
    if hasattr(model, "partial_fit"):
        model.partial_fit(X, y)
    else:
        model.fit(X, y)
    joblib.dump(model, model_path)
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description="Train or update the ML model")
    ap.add_argument("--data", default=str(DATA_PATH))
    ap.add_argument("--model", default=str(MODEL_PATH))
    args = ap.parse_args()

    X, y = load_training_data(Path(args.data))
    train_incremental(X, y, Path(args.model))
    print(f"Trained model on {len(y)} samples")

if __name__ == "__main__":
    main()
