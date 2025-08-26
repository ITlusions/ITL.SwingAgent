#!/usr/bin/env python
"""Train basic ML models using the vector store.

Loads vectors and outcomes from the Swing Agent vector store,
splits them into train/test sets and trains:

* LogisticRegression classifier for hit_target vs hit_stop
* LinearRegression model for realized_r regression

Models are saved with joblib to a models/ directory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split

from swing_agent.vectorstore import VectorStore, _ensure_db, get_session


def load_dataset(db_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Load feature vectors and labels from the vector store."""
    _ensure_db(db_path)
    x_list: list[np.ndarray] = []
    y_cls: list[int] = []
    y_reg: list[float] = []

    with get_session() as session:
        rows = session.query(VectorStore).all()
        for row in rows:
            if row.vec_json is None:
                continue
            vec = np.array(json.loads(row.vec_json), dtype=float)
            reason = row.exit_reason
            if not reason and row.payload:
                payload = row.payload
                if isinstance(payload, str):
                    try:
                        payload = json.loads(payload)
                    except Exception:
                        payload = None
                if isinstance(payload, dict):
                    reason = payload.get("exit_reason")
            if row.realized_r is None or reason is None:
                continue
            reason_upper = reason.upper()
            if reason_upper in {"TP", "TARGET", "HIT_TARGET"}:
                cls = 1
            elif reason_upper in {"SL", "STOP", "HIT_STOP"}:
                cls = 0
            else:
                continue
            x_list.append(vec)
            y_cls.append(cls)
            y_reg.append(row.realized_r)

    if not x_list:
        return None
    return np.array(x_list), np.array(y_cls), np.array(y_reg)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train ML models from vector store data.")
    ap.add_argument("--db", default="data/swing_agent.sqlite", help="Database path or URL")
    ap.add_argument("--model-dir", default="models", help="Directory to save models")
    args = ap.parse_args()

    dataset = load_dataset(args.db)
    if dataset is None:
        print("No training data found.")
        return
    features, y_cls, y_reg = dataset
    # Split for classification
    x_cls_train, x_cls_test, y_cls_train, y_cls_test = train_test_split(
        features, y_cls, test_size=0.2, random_state=args.random_state, stratify=y_cls
    )
    # Split for regression
    x_reg_train, x_reg_test, y_reg_train, y_reg_test = train_test_split(
        features, y_reg, test_size=0.2, random_state=args.random_state
    )

    cls_model = LogisticRegression(max_iter=1000)
    cls_model.fit(x_cls_train, y_cls_train)
    cls_pred = cls_model.predict(x_cls_test)
    acc = accuracy_score(y_cls_test, cls_pred)
    print(f"Classification accuracy: {acc:.3f}")

    reg_model = LinearRegression()
    reg_model.fit(x_reg_train, y_reg_train)
    reg_pred = reg_model.predict(x_reg_test)
    mse = mean_squared_error(y_reg_test, reg_pred)
    print(f"Regression MSE: {mse:.3f}")

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(cls_model, model_dir / "classification_model.joblib")
    joblib.dump(reg_model, model_dir / "realized_r_model.joblib")
    print(f"Saved models to {model_dir}")


if __name__ == "__main__":
    main()
