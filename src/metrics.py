"""Evaluation metrics aligned with the 4-8 PM product decision."""
import numpy as np
import pandas as pd


def score_prediction(actual: pd.Series, predicted: pd.Series, timestamps: pd.Series) -> dict[str, float]:
    valid = actual.notna() & predicted.notna()
    y = actual[valid].astype(float).to_numpy()
    p = predicted[valid].astype(float).to_numpy()
    time = pd.to_datetime(timestamps[valid])
    if len(y) == 0:
        raise ValueError("no overlapping values to score")
    peak = (time.dt.hour >= 16) & (time.dt.hour < 20)
    denom = np.maximum(np.abs(y), 0.01)
    return {
        "mae_kwh": float(np.mean(np.abs(y - p))),
        "mape_pct": float(np.mean(np.abs(y - p) / denom) * 100),
        "peak_window_error_kwh": float(abs(y[peak].sum() - p[peak].sum())),
        "bill_total_error_kwh": float(abs(y.sum() - p.sum())),
    }
