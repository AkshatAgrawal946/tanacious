"""Head-to-head evaluation of the baseline and the Random Forest on the same intervals.

* Bill-only buildings (hostels, library): both models reconstruct from the metered buildings
  and are scored against the hidden ground truth.
* Metered buildings (academic, lab): each is held out in turn, reconstructed from the other
  meters plus its own bill, and scored against its real readings.

Both models go through the same bill reconciliation, so only the predicted shape differs.
"""
import pandas as pd

from src import ml_model
from src.metrics import score_prediction
from src.reconstruct import ShapeFn, reconstruct_building_month

MODELS = ("Baseline", "Random forest")


def _reconstruct(reference: pd.DataFrame, bills: pd.DataFrame, building: str,
                 shape: ShapeFn | None) -> pd.Series:
    months = sorted(bills.loc[bills["building"] == building, "month"].astype(str))
    frames = [reconstruct_building_month(reference, bills, building, month, shape) for month in months]
    return pd.concat(frames).set_index("timestamp")["kWh_consumed"]


def _held_out(interval: pd.DataFrame, bills: pd.DataFrame, building: str, feature_set: str) -> pd.DataFrame:
    """Reconstruct a metered building from the *other* meters only, by both models."""
    reference = interval[interval["building"] != building]
    model = ml_model.train(reference, bills, feature_set)
    return pd.DataFrame({
        "Baseline": _reconstruct(reference, bills, building, None),
        "Random forest": _reconstruct(reference, bills, building, ml_model.shape_fn(model, reference)),
    })


def _weekday_peak_hour(values: pd.Series) -> int:
    weekday = values[values.index.day_of_week < 5]
    return int(weekday.groupby(weekday.index.hour).mean().idxmax())


def compare_models(interval: pd.DataFrame, bills: pd.DataFrame, hidden: pd.DataFrame,
                   feature_set: str = ml_model.DEFAULT_FEATURE_SET) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (per-interval predictions, per-building metrics)."""
    bills = bills.assign(month=bills["month"].astype(str))
    forest = ml_model.shape_fn(ml_model.train(interval, bills, feature_set), interval)
    frames = []
    for building, building_type in bills.groupby("building")["building_type"].first().items():
        if building in set(interval["building"]):
            actual = interval[interval["building"] == building].set_index("timestamp")["kWh_consumed"]
            predicted, evaluated_against = _held_out(interval, bills, building, feature_set), "held-out meter"
        else:
            actual = hidden[hidden["building"] == building].set_index("timestamp")["kWh_consumed"]
            predicted = pd.DataFrame({
                "Baseline": _reconstruct(interval, bills, building, None),
                "Random forest": _reconstruct(interval, bills, building, forest),
            })
            evaluated_against = "hidden ground truth"
        frame = predicted.assign(actual=actual.reindex(predicted.index)).rename_axis("timestamp").reset_index()
        frames.append(frame.assign(building=building, building_type=building_type, evaluated_against=evaluated_against))
    series = pd.concat(frames, ignore_index=True)[
        ["timestamp", "building", "building_type", "evaluated_against", "actual", *MODELS]]

    rows = []
    for (building, building_type, evaluated_against), group in series.groupby(
            ["building", "building_type", "evaluated_against"]):
        group = group.dropna(subset=["actual"])
        actual = group.set_index("timestamp")["actual"]
        peak = (group["timestamp"].dt.hour >= 16) & (group["timestamp"].dt.hour < 20)
        for model in MODELS:
            scores = score_prediction(group["actual"], group[model], group["timestamp"])
            rows.append({
                "building": building, "building_type": building_type, "evaluated_against": evaluated_against,
                "model": model, "mae_kwh": scores["mae_kwh"], "mape_pct": scores["mape_pct"],
                "peak_window_error_pct": 100 * scores["peak_window_error_kwh"] / group.loc[peak, "actual"].sum(),
                "true_peak_hour": _weekday_peak_hour(actual),
                "predicted_peak_hour": _weekday_peak_hour(group.set_index("timestamp")[model]),
            })
    return series, pd.DataFrame(rows)
