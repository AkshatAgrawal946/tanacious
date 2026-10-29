"""Evaluation metrics aligned with the 4-8 PM product decision.

Near-zero policy for MAPE: the denominator is floored at ``MAPE_FLOOR_KWH`` per interval, so an
interval with (almost) no consumption cannot make the score infinite. WAPE (total absolute error
over total actual energy) has no such issue and is the headline metric for model comparison.
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


def daily_peak_errors(actual: pd.Series, predicted: pd.Series, timestamps: pd.Series) -> dict[str, float]:
    """Per-day peak magnitude error (% of the true daily peak) and peak timing error (minutes),
    averaged over days with complete actuals."""
    frame = pd.DataFrame({"y": actual.to_numpy(), "p": predicted.to_numpy(), "t": pd.to_datetime(timestamps).to_numpy()})
    magnitude, timing = [], []
    for _, day in frame.groupby(frame["t"].dt.normalize()):
        if day["y"].isna().any() or day["p"].isna().any() or len(day) < 2:
            continue
        true_at, pred_at = day.loc[day["y"].idxmax()], day.loc[day["p"].idxmax()]
        magnitude.append(100 * abs(pred_at["p"] - true_at["y"]) / max(true_at["y"], MAPE_FLOOR_KWH))
        timing.append(abs((pred_at["t"] - true_at["t"]).total_seconds()) / 60)
    return {"peak_magnitude_error_pct": float(np.mean(magnitude)) if magnitude else float("nan"),
            "peak_timing_error_min": float(np.mean(timing)) if timing else float("nan"),
            "days_scored": len(magnitude)}


def peak_window_energy_error_pct(actual: pd.Series, predicted: pd.Series, timestamps: pd.Series) -> float:
    """Absolute error of total 4-8 PM energy, as % of the true 4-8 PM energy (days with actuals)."""
    peak = _peak_mask(pd.Series(timestamps)) & actual.notna().to_numpy() & predicted.notna().to_numpy()
    true = actual.to_numpy()[peak].sum()
    return float(100 * abs(predicted.to_numpy()[peak].sum() - true) / true) if true > 0 else float("nan")


def interval_coverage(actual: pd.Series, lower: pd.Series, upper: pd.Series) -> float:
    """Share of actual values inside [lower, upper] (e.g. the P10-P90 band; nominal 80%)."""
    valid = actual.notna() & lower.notna() & upper.notna()
    if not valid.any():
        return float("nan")
    inside = (actual[valid] >= lower[valid]) & (actual[valid] <= upper[valid])
    return float(inside.mean())


def forecast_scores(actual: pd.Series, predicted: pd.Series, timestamps: pd.Series,
                    lower: pd.Series | None = None, upper: pd.Series | None = None) -> dict[str, float]:
    """Every forecasting metric for one set of aligned intervals (NaN actuals are excluded)."""
    actual, predicted = actual.reset_index(drop=True), predicted.reset_index(drop=True)
    timestamps = pd.Series(pd.to_datetime(timestamps)).reset_index(drop=True)
    scores = score_prediction(actual, predicted, timestamps)
    scores.pop("bill_total_error_kwh")
    scores["peak_window_energy_error_pct"] = peak_window_energy_error_pct(actual, predicted, timestamps)
    scores.update(daily_peak_errors(actual, predicted, timestamps))
    if lower is not None and upper is not None:
        scores["p10_p90_coverage"] = interval_coverage(actual, lower.reset_index(drop=True), upper.reset_index(drop=True))
    scores["intervals_scored"] = int((actual.notna() & predicted.notna()).sum())
    return scores
