"""Evaluation metrics aligned with the 4-8 PM product decision.

Near-zero policy for MAPE: the denominator is floored at ``MAPE_FLOOR_KWH`` per interval, so an
interval with (almost) no consumption cannot make the score infinite. WAPE (total absolute error
over total actual energy) has no such issue and is the headline metric.
"""
import numpy as np
import pandas as pd

MAPE_FLOOR_KWH = 0.01
PEAK_HOURS = (16, 20)  # 4-8 PM, end exclusive


def _peak_mask(time: pd.Series) -> np.ndarray:
    hour = pd.to_datetime(time).dt.hour
    return ((hour >= PEAK_HOURS[0]) & (hour < PEAK_HOURS[1])).to_numpy()


def score_prediction(actual: pd.Series, predicted: pd.Series, timestamps: pd.Series,
                     mape_floor_kwh: float = MAPE_FLOOR_KWH) -> dict[str, float]:
    valid = (actual.notna() & predicted.notna()).to_numpy()
    y = actual[valid].astype(float).to_numpy()
    p = predicted[valid].astype(float).to_numpy()
    time = pd.to_datetime(timestamps[valid])
    if len(y) == 0:
        raise ValueError("no overlapping values to score")
    peak = _peak_mask(time)
    error = np.abs(y - p)
    return {
        "mae_kwh": float(error.mean()),
        "rmse_kwh": float(np.sqrt(np.mean((y - p) ** 2))),
        "wape_pct": float(100 * error.sum() / max(np.abs(y).sum(), 1e-9)),
        "mape_pct": float(np.mean(error / np.maximum(np.abs(y), mape_floor_kwh)) * 100),
        "peak_window_error_kwh": float(abs(y[peak].sum() - p[peak].sum())),
        "peak_window_mae_kwh": float(error[peak].mean()) if peak.any() else float("nan"),
        "bill_total_error_kwh": float(abs(y.sum() - p.sum())),
    }
