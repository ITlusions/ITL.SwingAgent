"""Probability calibration utilities for SwingAgent.

This module provides a simple interface for calibrating predicted win
rates using historical signal outcomes. It fits an isotonic regression
model on past predictions and realized results and exposes a
``calibrated_winrate`` function used by :mod:`agent` before signals are
stored.

The calibration model is loaded lazily from the signals database.  If
insufficient data is available the input probability is returned
unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import sqlite3

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

__all__ = ["calibrated_winrate", "load_calibrator"]


_model: Optional[IsotonicRegression] = None
_model_db: Optional[Path] = None


def _fit_model(db_path: Path) -> Optional[IsotonicRegression]:
    """Fit an isotonic regression model from historical signals.

    Parameters
    ----------
    db_path:
        Path to the SQLite database containing the ``signals`` table.

    Returns
    -------
    Optional[IsotonicRegression]
        Fitted model or ``None`` if there is insufficient data.
    """
    query = (
        "SELECT expected_winrate, realized_r FROM signals "
        "WHERE evaluated = 1 AND expected_winrate IS NOT NULL "
        "AND realized_r IS NOT NULL"
    )
    try:
        with sqlite3.connect(db_path) as con:
            df = pd.read_sql_query(query, con)
    except sqlite3.Error:
        return None

    if len(df) < MIN_SAMPLE_SIZE:
        return None

    y = (df["realized_r"] > 0).astype(float).to_numpy()
    p = df["expected_winrate"].clip(0, 1).to_numpy()
    model = IsotonicRegression(out_of_bounds="clip")
    model.fit(p, y)
    return model


def load_calibrator(db_path: str | Path = "data/swing_agent.sqlite") -> Optional[IsotonicRegression]:
    """Load (or fit) the calibration model for the given database."""
    global _model, _model_db
    db = Path(db_path)
    if _model is None or _model_db != db:
        _model = _fit_model(db)
        _model_db = db
    return _model


def calibrated_winrate(prob: float, db_path: str | Path = "data/swing_agent.sqlite") -> float:
    """Return calibrated win rate for the given probability.

    If a calibration model is available, the probability is transformed
    using isotonic regression; otherwise it is returned unchanged.
    """
    model = load_calibrator(db_path)
    prob = float(prob)
    if model is None:
        return prob
    return float(model.predict(np.array([prob]))[0])
