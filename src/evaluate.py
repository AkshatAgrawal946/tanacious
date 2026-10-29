"""Accuracy check for the Random Forest reconstruction (Task A). Forecasting (Task B) has its own
chronological protocol in ``src.forecasting.backtest``.

Scenarios (building-level holdouts; no hyperparameters are tuned on any of them):

* Bill-only buildings (hostels, library): reconstructed from every real reading (smart meters and
  pilot meters) and scored against the hidden ground truth outside the pilot-meter weeks.
* Unmetered (``evaluate_model``, metered buildings): each metered building is held out in turn,
  reconstructed from the other buildings plus its own bill, and scored against its real readings.
  This is the "new building type with no meter" test.
* Partially metered (``evaluate_partial_metering``): a metered building keeps a random fraction of
  its days; the rest are hidden, reconstructed with the bill, and scored on the hidden intervals only.

Observed intervals are scored separately from reconstructed ones: reconstruction must return them
unchanged, so their error has to be exactly zero. Hidden ground truth is only ever loaded through
``load_ground_truth`` and only used for scoring, never for training.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src import ml_model
from src.metrics import score_prediction
from src.reconstruct import ShapeFn, reconstruct_building_month

EVALUATION_ONLY = "evaluation_only"


def load_ground_truth(data_dir: str | Path = "data/raw") -> pd.DataFrame:
    """Synthetic hidden truth, for scoring only. The frame is tagged so training refuses it."""
    hidden = pd.read_csv(Path(data_dir) / "ground_truth_15min.csv", parse_dates=["timestamp"])
    hidden.attrs[EVALUATION_ONLY] = True
    return hidden


def flat_shape(timestamps: pd.DatetimeIndex, building: str, building_type: str) -> np.ndarray:
    """Simplest baseline: spread the bill evenly over every interval of the month."""
    return np.ones(len(timestamps))


def _reconstruct(reference: pd.DataFrame, bills: pd.DataFrame, building: str, shape: ShapeFn) -> pd.DataFrame:
    months = sorted(bills.loc[bills["building"] == building, "month"].astype(str))
    frames = [reconstruct_building_month(reference, bills, building, month, shape) for month in months]
    return pd.concat(frames).set_index("timestamp")


def _held_out(interval: pd.DataFrame, bills: pd.DataFrame, building: str) -> pd.Series:
    """Reconstruct a metered building from the *other* buildings' readings only."""
    reference = interval[interval["building"] != building]
    model = ml_model.train(reference, bills)
    return _reconstruct(reference, bills, building, ml_model.shape_fn(model, reference, bills))["kWh_consumed"]


def _weekday_peak_hour(values: pd.Series) -> int:
    weekday = values[values.index.day_of_week < 5]
    return int(weekday.groupby(weekday.index.hour).mean().idxmax())


def evaluate_model(interval: pd.DataFrame, bills: pd.DataFrame, hidden: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (per-interval predictions next to the truth, per-building scores of reconstructed intervals)."""
    if interval.attrs.get(EVALUATION_ONLY):
        raise ValueError("evaluation ground truth was passed as training data")
    bills = bills.assign(month=bills["month"].astype(str))
    forest = ml_model.shape_fn(ml_model.train(interval, bills), interval, bills)
    frames = []
    for building, building_type in bills.groupby("building")["building_type"].first().items():
        readings = interval.loc[(interval["building"] == building) & interval["kWh_consumed"].notna(), "timestamp"]
        if building in set(hidden["building"]):
            actual = hidden[hidden["building"] == building].set_index("timestamp")["kWh_consumed"]
            predicted = _reconstruct(interval, bills, building, forest)["kWh_consumed"]
            evaluated_against = "hidden ground truth"
            # Pilot-meter weeks are real readings the model keeps as-is, so they are not scored.
            is_pilot = predicted.index.isin(readings)
        else:
            actual = interval[interval["building"] == building].set_index("timestamp")["kWh_consumed"]
            predicted, evaluated_against, is_pilot = _held_out(interval, bills, building), "held-out meter", False
        frame = pd.DataFrame({"predicted": predicted, "actual": actual.reindex(predicted.index), "is_pilot": is_pilot})
        frame = frame.rename_axis("timestamp").reset_index()
        frames.append(frame.assign(building=building, building_type=building_type, evaluated_against=evaluated_against))
    series = pd.concat(frames, ignore_index=True)[
        ["timestamp", "building", "building_type", "evaluated_against", "is_pilot", "actual", "predicted"]]
    return series, score_series(series)


def observed_preservation_error(series: pd.DataFrame) -> float:
    """Largest absolute change reconstruction made to an observed (pilot) reading; must be 0."""
    observed = series[series["is_pilot"]].dropna(subset=["actual"])
    return float((observed["predicted"] - observed["actual"]).abs().max()) if len(observed) else 0.0


def score_series(series: pd.DataFrame) -> pd.DataFrame:
    """Per-building scores for any slice of ``evaluate_model`` output (e.g. some months).
    Pilot-meter intervals are left out: the model keeps those real readings as-is."""
    rows = []
    for (building, building_type, evaluated_against), group in series[~series["is_pilot"]].groupby(
            ["building", "building_type", "evaluated_against"]):
        group = group.dropna(subset=["actual"])
        peak = (group["timestamp"].dt.hour >= 16) & (group["timestamp"].dt.hour < 20)
        scores = score_prediction(group["actual"], group["predicted"], group["timestamp"])
        rows.append({
            "building": building, "building_type": building_type, "evaluated_against": evaluated_against,
            "mae_kwh": scores["mae_kwh"], "mape_pct": scores["mape_pct"], "wape_pct": scores["wape_pct"],
            "peak_window_error_pct": 100 * scores["peak_window_error_kwh"] / group.loc[peak, "actual"].sum(),
            "true_peak_hour": _weekday_peak_hour(group.set_index("timestamp")["actual"]),
            "predicted_peak_hour": _weekday_peak_hour(group.set_index("timestamp")["predicted"]),
        })
    return pd.DataFrame(rows)


def _hide_days(readings: pd.DataFrame, observed_fraction: float, seed: int) -> tuple[pd.DataFrame, pd.Series]:
    """Keep a random ``observed_fraction`` of whole days; return (kept readings, hidden mask)."""
    days = readings["timestamp"].dt.normalize()
    unique = np.sort(days.unique())
    rng = np.random.default_rng(seed)
    kept_days = rng.choice(unique, size=int(round(observed_fraction * len(unique))), replace=False)
    hidden = ~days.isin(kept_days)
    return readings[~hidden], hidden


def evaluate_partial_metering(interval: pd.DataFrame, bills: pd.DataFrame, buildings: list[str] | None = None,
                              observed_fractions: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75),
                              seed: int = ml_model.SEED) -> pd.DataFrame:
    """Reconstruction accuracy on intervals hidden from a metered building, by how much of the
    building stays metered (0.0 = completely unmetered). The Random Forest is compared with the
    flat bill split, and observed intervals are checked to come back unchanged."""
    if interval.attrs.get(EVALUATION_ONLY):
        raise ValueError("evaluation ground truth was passed as training data")
    bills = bills.assign(month=bills["month"].astype(str))
    metered = interval.groupby("building")["kWh_consumed"].count()
    buildings = buildings or sorted(metered[metered > 0.5 * 96 * 365].index)
    rows = []
    for building in buildings:
        own = interval[interval["building"] == building].dropna(subset=["kWh_consumed"])
        truth = own.set_index("timestamp")["kWh_consumed"]
        for fraction in observed_fractions:
            kept, hidden_mask = _hide_days(own, fraction, seed)
            reference = pd.concat([interval[interval["building"] != building], kept], ignore_index=True)
            forest = ml_model.shape_fn(ml_model.train(reference, bills), reference, bills)
            for model_name, shape in (("random_forest", forest), ("flat_bill_split", flat_shape)):
                result = _reconstruct(reference, bills, building, shape)
                scored = own.loc[hidden_mask.to_numpy(), "timestamp"]
                predicted = result.loc[scored, "kWh_consumed"]
                scores = score_prediction(truth.loc[scored].reset_index(drop=True), predicted.reset_index(drop=True),
                                          pd.Series(scored.to_numpy()))
                observed = result[result["is_observed"]]
                preserved = float((observed["kWh_consumed"] - truth.reindex(observed.index)).abs().max()) if len(observed) else 0.0
                rows.append({"building": building, "observed_fraction": fraction, "model": model_name,
                             "scored_intervals": len(scored), **scores, "observed_max_abs_change_kwh": preserved})
    return pd.DataFrame(rows)
